//! Device-path accounting: host fallbacks and host↔device transfers.
//!
//! Every tensor op is structured as "run the device kernel; if that is not
//! possible, compute on host". On a GPU run the host branch is always a
//! bug, so it is counted here (per op name) and returned as an error: GPU work
//! never silently runs on the CPU. Transfers are counted
//! separately: uploads of host-built inputs (noise, RoPE tables, timesteps) and
//! downloads of results are expected at the boundaries, but a transfer inside
//! a forward is a regression. `fv-gpucheck` snapshots these around each output.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use super::tensor::{Result, TensorError};

static FALLBACKS: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());
static H2D_COUNT: AtomicU64 = AtomicU64::new(0);
static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static D2H_COUNT: AtomicU64 = AtomicU64::new(0);
/// Every NVRTC kernel launch, counted at the one macro they all go through.
/// Launch overhead is a few microseconds each, so this is what turns "maybe we
/// are launch-bound" into an arithmetic claim.
static LAUNCHES: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// Host computations with a live device, per op.
    pub host_fallbacks: BTreeMap<&'static str, u64>,
    pub h2d_count: u64,
    pub h2d_bytes: u64,
    pub d2h_count: u64,
    pub d2h_bytes: u64,
    pub launches: u64,
}

impl Snapshot {
    pub fn total_fallbacks(&self) -> u64 {
        self.host_fallbacks.values().sum()
    }

    /// Counts accumulated since `earlier`.
    pub fn since(&self, earlier: &Snapshot) -> Snapshot {
        let mut host_fallbacks = BTreeMap::new();
        for (op, &n) in &self.host_fallbacks {
            let d = n - earlier.host_fallbacks.get(op).copied().unwrap_or(0);
            if d > 0 {
                host_fallbacks.insert(*op, d);
            }
        }
        Snapshot {
            host_fallbacks,
            h2d_count: self.h2d_count - earlier.h2d_count,
            h2d_bytes: self.h2d_bytes - earlier.h2d_bytes,
            d2h_count: self.d2h_count - earlier.d2h_count,
            d2h_bytes: self.d2h_bytes - earlier.d2h_bytes,
            launches: self.launches - earlier.launches,
        }
    }
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        host_fallbacks: FALLBACKS.lock().expect("stats lock").clone(),
        h2d_count: H2D_COUNT.load(Relaxed),
        h2d_bytes: H2D_BYTES.load(Relaxed),
        d2h_count: D2H_COUNT.load(Relaxed),
        d2h_bytes: D2H_BYTES.load(Relaxed),
        launches: LAUNCHES.load(Relaxed),
    }
}

/// Counted by the `launch!` macro, so no call site can forget.
#[inline]
pub(crate) fn record_launch() {
    LAUNCHES.fetch_add(1, Relaxed);
}

pub fn reset() {
    FALLBACKS.lock().expect("stats lock").clear();
    for c in [&H2D_COUNT, &H2D_BYTES, &D2H_COUNT, &D2H_BYTES, &LAUNCHES] {
        c.store(0, Relaxed);
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn record_h2d(elems: usize) {
    H2D_COUNT.fetch_add(1, Relaxed);
    H2D_BYTES.fetch_add((elems * 4) as u64, Relaxed);
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn record_d2h(elems: usize) {
    D2H_COUNT.fetch_add(1, Relaxed);
    D2H_BYTES.fetch_add((elems * 4) as u64, Relaxed);
}

/// True when ops are expected to run on the device (live CUDA device and
/// residency on). Host computation in that state is a fallback.
pub(crate) fn device_expected() -> bool {
    super::device::has_live_device() && super::resident::residency_enabled()
}

/// Call immediately before an op computes on host. On a CPU run this is a
/// no-op (the host path is the reference implementation). With a device
/// expected it is always an error: GPU code never silently runs on the CPU.
/// The attempt is still counted so reports show which op was missing a kernel.
pub(crate) fn host_fallback(op: &'static str, detail: impl std::fmt::Display) -> Result<()> {
    if !device_expected() {
        return Ok(());
    }
    *FALLBACKS.lock().expect("stats lock").entry(op).or_insert(0) += 1;
    Err(TensorError::Message(format!(
        "`{op}` has no device path for this call ({detail}); refusing to run GPU work on the CPU"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_fallback_is_silent_without_a_device() {
        reset();
        host_fallback("add", "cpu run").unwrap();
        assert_eq!(snapshot().total_fallbacks(), 0);
    }

    #[test]
    fn since_subtracts_per_op() {
        let a = Snapshot {
            host_fallbacks: [("add", 2u64)].into_iter().collect(),
            h2d_count: 1,
            ..Snapshot::default()
        };
        let b = Snapshot {
            host_fallbacks: [("add", 5u64), ("cat", 1)].into_iter().collect(),
            h2d_count: 4,
            ..Snapshot::default()
        };
        let d = b.since(&a);
        assert_eq!(d.host_fallbacks.get("add"), Some(&3));
        assert_eq!(d.host_fallbacks.get("cat"), Some(&1));
        assert_eq!(d.h2d_count, 3);
    }
}
