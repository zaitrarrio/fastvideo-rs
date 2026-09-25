//! Device-allocation ledger: which named component holds how many bytes of
//! the allocator pool, so a phase's memory line explains every GiB.
//!
//! Components record what they hold when they load and release it when they
//! drop: [`track`] measures the pool's live bytes across a load (exact on a
//! device, whatever the component believes it allocated), [`set`] / [`add`] /
//! [`clear`] book sizes the owner knows (the offload ring, the AdaLN table,
//! step caches). What the pool holds beyond the booked total is
//! `activations`: the tensors of the running forward or decode.
//!
//! [`crate::wan::offload::MemoryLog::mark`] appends it to each phase's
//! `memory` line (numbers illustrative):
//!
//! ```text
//! h3 memory denoise: peak 16.10 GiB used, ...; ledger: text_refiner 0.05 | dit_nonlinear 0.06 |
//!   dit_ring 0.00 (max 1.44) | adaln_table 0.00 (max 0.04) | step_cache 0.00 (max 2.20) |
//!   activations 12.30 | pool_reserved 5.10 | non_pool 0.80 | total 16.90 of budget 32.00 GiB
//! ```
//!
//! Booked values carry the phase's maximum as well as the current value (a
//! ring freed at the end of the denoise still counts at the denoise's peak);
//! `activations` is the pool's peak minus the booked maxima, `pool_reserved`
//! the cached-but-free blocks at the reserved peak, `non_pool` the context,
//! workspaces and anything else on the card outside the pool.

use std::sync::Mutex;

/// Text encoder weights kept between prompts (resident FP8/bf16 decoder).
pub const TEXT_ENCODER: &str = "text_encoder";
/// A streamed DiT's device ring (`slots x block`).
pub const DIT_RING: &str = "dit_ring";
/// A streamed refiner's device ring.
pub const REFINER_RING: &str = "refiner_ring";
/// What a DiT keeps on the device outside its streamed linears: every block
/// when resident; the skeletons (norms, tables, biases), the in/out
/// projections and the global modulation otherwise.
pub const DIT_NONLINEAR: &str = "dit_nonlinear";
/// The text refiner (H3) or connectors (LTX) outside their ring.
pub const TEXT_REFINER: &str = "text_refiner";
/// Per-step AdaLN / modulation table uploaded for the denoise.
pub const ADALN_TABLE: &str = "adaln_table";
/// Video decoder weights.
pub const VAE: &str = "vae";
/// Audio decoder (and vocoder) weights.
pub const AUDIO_VAE: &str = "audio_vae";
/// Latent upsampler weights.
pub const UPSAMPLER: &str = "upsampler";
/// Buffers a step cache keeps between forwards (TeaCache / FBCache).
pub const STEP_CACHE: &str = "step_cache";

/// One booked category: bytes now and the most it held since the last
/// [`reset_phase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Entry {
    pub now: u64,
    pub phase_max: u64,
}

static LEDGER: Mutex<Vec<(&'static str, Entry)>> = Mutex::new(Vec::new());

fn with<R>(f: impl FnOnce(&mut Vec<(&'static str, Entry)>) -> R) -> R {
    f(&mut LEDGER.lock().expect("ledger lock"))
}

fn entry<'a>(v: &'a mut Vec<(&'static str, Entry)>, cat: &'static str) -> &'a mut Entry {
    let i = match v.iter().position(|(c, _)| *c == cat) {
        Some(i) => i,
        None => {
            v.push((cat, Entry::default()));
            v.len() - 1
        }
    };
    &mut v[i].1
}

/// Book `cat` at `bytes`.
pub fn set(cat: &'static str, bytes: u64) {
    with(|v| {
        let e = entry(v, cat);
        e.now = bytes;
        e.phase_max = e.phase_max.max(bytes);
    });
}

/// Add (or, negative, release) bytes of `cat`.
pub fn add(cat: &'static str, delta: i64) {
    with(|v| {
        let e = entry(v, cat);
        e.now = e.now.saturating_add_signed(delta);
        e.phase_max = e.phase_max.max(e.now);
    });
}

/// Release everything booked under `cat`.
pub fn clear(cat: &'static str) {
    set(cat, 0);
}

pub fn get(cat: &'static str) -> Entry {
    with(|v| {
        v.iter()
            .find(|(c, _)| *c == cat)
            .map(|(_, e)| *e)
            .unwrap_or_default()
    })
}

/// Every category with a non-zero current or phase-maximum value, in the
/// order they were first booked.
pub fn entries() -> Vec<(&'static str, Entry)> {
    with(|v| {
        v.iter()
            .filter(|(_, e)| e.now > 0 || e.phase_max > 0)
            .copied()
            .collect()
    })
}

/// Start a new phase: each maximum restarts from the current value.
pub fn reset_phase() {
    with(|v| {
        for (_, e) in v.iter_mut() {
            e.phase_max = e.now;
        }
    });
}

/// Forget everything (tests, or a new pipeline in the same process).
pub fn reset() {
    with(Vec::clear);
}

/// Pool bytes live right now after the queued work finished, or `None`
/// without a device.
fn pool_live() -> Option<u64> {
    super::device::synchronize().ok()?;
    super::device::pool_usage().map(|u| u.used)
}

/// Run a load and book the pool bytes it left live under `cat` (added to
/// what `cat` already holds). Returns the result and the measured bytes;
/// without a device nothing is measured (`None`), and the caller may book a
/// size it computed instead.
pub fn track<R>(cat: &'static str, f: impl FnOnce() -> R) -> (R, Option<u64>) {
    let before = pool_live();
    let out = f();
    let measured = match (before, pool_live()) {
        (Some(b), Some(a)) => {
            let delta = a as i64 - b as i64;
            add(cat, delta);
            Some(delta.max(0) as u64)
        }
        _ => None,
    };
    (out, measured)
}

/// Book `claimed` (a size a component computed from its shapes) when no
/// measurement exists, and say so when a measurement disagrees with it by
/// more than 5 % and 64 MiB: the 22.7-vs-45.4 GiB text encoder of a past run
/// was exactly such a claim.
pub fn reconcile(cat: &'static str, measured: Option<u64>, claimed: u64) {
    match measured {
        None => add(cat, claimed as i64),
        Some(m) => {
            let diff = m.abs_diff(claimed);
            if diff > (64 << 20) && diff * 20 > claimed.max(m) {
                super::log::info(format_args!(
                    "ledger {cat}: measured {:.2} GiB on the device, the component reports {:.2} GiB",
                    m as f64 / GIB,
                    claimed as f64 / GIB
                ));
            }
        }
    }
}

const GIB: f64 = (1u64 << 30) as f64;

/// What one owner (a pipeline, a transient decoder) booked, taken back out
/// of the ledger when it is released or dropped, so owners sharing a
/// category never erase each other's entries.
#[derive(Debug, Default)]
pub struct Booking {
    items: Vec<(&'static str, u64)>,
}

impl Booking {
    /// [`track`] a load under `cat`; when nothing could be measured (no
    /// device) and `claimed` returns a size, that size is booked instead.
    /// A measurement far from the claim is logged ([`reconcile`]).
    pub fn track<R>(
        &mut self,
        cat: &'static str,
        f: impl FnOnce() -> R,
        claimed: impl FnOnce(&R) -> Option<u64>,
    ) -> R {
        let (out, measured) = track(cat, f);
        let claim = claimed(&out);
        let bytes = match (measured, claim) {
            (Some(m), Some(c)) => {
                reconcile(cat, Some(m), c);
                m
            }
            (Some(m), None) => m,
            (None, Some(c)) => {
                add(cat, c as i64);
                c
            }
            (None, None) => 0,
        };
        self.items.push((cat, bytes));
        out
    }

    /// Book `bytes` under `cat` for this owner.
    pub fn book(&mut self, cat: &'static str, bytes: u64) {
        add(cat, bytes as i64);
        self.items.push((cat, bytes));
    }

    /// Take everything this owner booked under `cat` back out.
    pub fn release(&mut self, cat: &'static str) {
        self.items.retain(|&(c, b)| {
            if c == cat {
                add(c, -(b as i64));
                false
            } else {
                true
            }
        });
    }

    /// Bytes this owner holds under `cat`.
    pub fn held(&self, cat: &'static str) -> u64 {
        self.items
            .iter()
            .filter(|(c, _)| *c == cat)
            .map(|(_, b)| b)
            .sum()
    }
}

impl Drop for Booking {
    fn drop(&mut self) {
        for (c, b) in self.items.drain(..) {
            add(c, -(b as i64));
        }
    }
}

/// A phase's accounting: booked categories, the untracked remainder, pool
/// fragmentation, what lives outside the pool, and the budget.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseLedger {
    pub entries: Vec<(&'static str, Entry)>,
    /// Pool peak minus the booked maxima: the phase's activations.
    pub activations: u64,
    /// Pool reserved peak minus used peak (cached, fragmented blocks).
    pub pool_reserved: u64,
    /// Device bytes outside the pool (context, workspaces, other processes).
    pub non_pool: u64,
    /// `peak_used + non_pool`: what the phase needed from the card.
    pub total: u64,
    pub budget: Option<u64>,
}

impl PhaseLedger {
    pub fn new(
        entries: Vec<(&'static str, Entry)>,
        peak_used: u64,
        peak_reserved: u64,
        non_pool: u64,
        budget: Option<u64>,
    ) -> Self {
        let booked: u64 = entries.iter().map(|(_, e)| e.phase_max).sum();
        Self {
            entries,
            activations: peak_used.saturating_sub(booked),
            pool_reserved: peak_reserved.saturating_sub(peak_used),
            non_pool,
            total: peak_used + non_pool,
            budget,
        }
    }

    /// `name GiB | ...`, with `(max X)` where a category peaked above its
    /// value at the end of the phase.
    pub fn describe(&self) -> String {
        let g = |b: u64| b as f64 / GIB;
        let mut parts: Vec<String> = self
            .entries
            .iter()
            .map(|(c, e)| {
                if e.phase_max > e.now {
                    format!("{c} {:.2} (max {:.2})", g(e.now), g(e.phase_max))
                } else {
                    format!("{c} {:.2}", g(e.now))
                }
            })
            .collect();
        parts.push(format!("activations {:.2}", g(self.activations)));
        parts.push(format!("pool_reserved {:.2}", g(self.pool_reserved)));
        parts.push(format!("non_pool {:.2}", g(self.non_pool)));
        let mut line = parts.join(" | ");
        line.push_str(&format!(" | total {:.2}", g(self.total)));
        if let Some(b) = self.budget {
            line.push_str(&format!(" of budget {:.2}", g(b)));
        }
        line.push_str(" GiB");
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The ledger is process-wide and other tests book into it; these tests
    // use categories of their own and never reset it.
    const G: u64 = 1 << 30;

    fn mine(prefix: &str) -> Vec<(&'static str, Entry)> {
        entries()
            .into_iter()
            .filter(|(c, _)| c.starts_with(prefix))
            .collect()
    }

    #[test]
    fn a_phase_explains_every_byte_of_its_peak() {
        let (enc, dit, ring, table) = ("t1_text_encoder", "t1_dit", "t1_ring", "t1_table");
        set(enc, 22 * G);
        set(dit, G / 2);
        // A phase starts from the current values.
        with(|v| {
            for (c, e) in v.iter_mut() {
                if c.starts_with("t1_") {
                    e.phase_max = e.now;
                }
            }
        });
        // The ring exists during the denoise and is released at its end.
        set(ring, 2 * G);
        add(table, (G / 4) as i64);
        clear(ring);
        assert_eq!(
            get(ring),
            Entry {
                now: 0,
                phase_max: 2 * G
            }
        );
        let peak_used = 22 * G + G / 2 + 2 * G + G / 4 + 11 * G;
        let l = PhaseLedger::new(mine("t1_"), peak_used, peak_used + 6 * G, G, Some(32 * G));
        // Booked maxima + activations = the pool peak, to the byte.
        let booked: u64 = l.entries.iter().map(|(_, e)| e.phase_max).sum();
        assert_eq!(booked + l.activations, peak_used);
        assert_eq!(l.activations, 11 * G);
        assert_eq!(l.pool_reserved, 6 * G);
        assert_eq!(l.total, peak_used + G);
        let line = l.describe();
        assert!(line.contains("t1_text_encoder 22.00"), "{line}");
        assert!(line.contains("t1_ring 0.00 (max 2.00)"), "{line}");
        assert!(line.contains("activations 11.00"), "{line}");
        assert!(line.contains("of budget 32.00 GiB"), "{line}");
        for c in [enc, dit, ring, table] {
            clear(c);
        }
    }

    #[test]
    fn without_a_device_a_claim_is_booked() {
        let cat = "t2_vae";
        let ((), measured) = track(cat, || ());
        assert_eq!(measured, None);
        reconcile(cat, measured, 5 * G);
        assert_eq!(get(cat).now, 5 * G);
        clear(cat);
        assert_eq!(get(cat).now, 0);
    }

    #[test]
    fn owners_sharing_a_category_release_only_their_own_bytes() {
        let (dit, enc) = ("t3_dit", "t3_text_encoder");
        let mut a = Booking::default();
        let mut b = Booking::default();
        let v = a.track(dit, || 7u8, |_| Some(3 * G));
        assert_eq!(v, 7);
        b.book(dit, 2 * G);
        a.book(enc, 22 * G);
        assert_eq!(get(dit).now, 5 * G);
        a.release(enc);
        assert_eq!(get(enc).now, 0);
        assert_eq!(a.held(dit), 3 * G);
        drop(a);
        assert_eq!(get(dit).now, 2 * G);
        drop(b);
        assert_eq!(get(dit).now, 0);
    }
}
