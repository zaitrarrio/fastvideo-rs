//! Regression fixtures for ADR-0020: every pre-B4a controller (`legacy.rs`,
//! verbatim) and its replacement on the one implementation, driven through
//! the same runtime protocol on the same synthetic tensors. Each test
//! compares every decision (compute / reuse, reason, the accumulator and
//! the indicator bit for bit) and the denoised latent bit for bit; the
//! latent feeds the next step's signal, so one diverging decision diverges
//! everything after it.
//!
//! The protocols mirror the family runtimes (file:line on main `2e3c71b1`):
//! Wan / Sana EasyCache `wan/pipeline.rs:1802-1895`,
//! `sana_video/pipeline.rs:238-280`; A14B `wan/transformer.rs:1827-1906`;
//! LingBot `lingbot/pipeline.rs:534-570`; Wan Sol TeaCache
//! `wan/transformer.rs:1223-1303, :1662-1681`; H3 `h3/transformer.rs:3258-3331`;
//! Cosmos `cosmos/transformer.rs:539-593`, `cosmos3/pipeline.rs:242-276`;
//! TeaCache4Wan2.1 `wan/pipeline.rs:1632-1671, :1946-2000`; LTX FBCache
//! `ltx2/transformer.rs:1439-1512`; Taylor `wan/transformer.rs:1305-1338,
//! :1908-1960`.

use super::legacy;
use super::{Decision, Lane, Refresh, Signal, StepCache};
use crate::cosmos::sol as cosmos;
use crate::h3::sol as h3;
use crate::lingbot::sol as lingbot;
use crate::ltx2::fbcache as ltx;
use crate::wan::sol_cache as wan;

/// A deterministic generator (64-bit LCG, top 24 bits).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

/// A stand-in transformer: a step- and branch-dependent smooth map of `x`
/// (the drift between steps is what the caches measure).
fn model(x: &[f32], step: usize, branch: f32, w: &[f32]) -> Vec<f32> {
    let t = 1.0 - step as f32 * 0.013;
    x.iter()
        .zip(w)
        .map(|(v, w)| v * (0.9 + 0.02 * branch) * t + w * (0.3 + 0.01 * step as f32))
        .collect()
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn lincomb(sa: f32, a: &[f32], sb: f32, b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| sa * x + sb * y).collect()
}

/// The cudarc host path of `mean |a − b|` (`wan/sol_cache.rs:134`).
fn mean_abs_delta(a: &[f32], b: &[f32]) -> f64 {
    let (diff, _) = super::distance::abs_sums(a, b);
    diff / a.len().max(1) as f64
}

/// `mean |t|` (`wan/sol_cache.rs:128`).
fn mean_abs(t: &[f32]) -> f64 {
    let (_, sum) = super::distance::abs_sums(t, t);
    sum / t.len().max(1) as f64
}

fn euler(x: &[f32], v: &[f32]) -> Vec<f32> {
    lincomb(1.0, x, -0.07, v)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn ob(v: Option<f64>) -> Option<u64> {
    v.map(f64::to_bits)
}

/// One recorded decision, comparable across the two implementations.
#[derive(Debug, PartialEq, Eq)]
struct Rec {
    compute: bool,
    reason: &'static str,
    acc: u64,
    indicator: Option<u64>,
}

impl Rec {
    fn of(d: &Decision) -> Self {
        Self {
            compute: d.compute(),
            reason: d.reason,
            acc: d.accumulator.to_bits(),
            indicator: ob(d.indicator),
        }
    }
}

// ---------------------------------------------------------------------------
// EasyCache: Wan whole-stack (and SANA), A14B, LingBot
// ---------------------------------------------------------------------------

/// Wan `dit_cfg_easy`: cond decides, uncond follows; residual reuse.
fn wan_easy_run(new: bool, n: usize, thr: f64, retain: usize, cool: usize) -> (Vec<Rec>, Vec<u32>) {
    let mut rng = Lcg(7 + n as u64);
    let w = rng.vec(64, 1.0);
    let mut x = rng.vec(64, 2.0);
    let mut old = legacy::wan_easy::EasyCache::new(n, thr, retain, cool).unwrap();
    let mut neu = wan::easycache(n, thr, retain, cool).unwrap();
    let (mut prev_in, mut last_in, mut last_out) =
        (None::<Vec<f32>>, None::<Vec<f32>>, None::<Vec<f32>>);
    let (mut res_c, mut res_u) = (None::<Vec<f32>>, None::<Vec<f32>>);
    let mut recs = Vec::new();
    let scale = 5.0f32;
    for step in 0..n {
        let latent_in = x.clone();
        let (compute, rec) = if new {
            let signal = if neu.needs_signal(Lane::COND, step) {
                Signal::Drift {
                    input_change: mean_abs_delta(&latent_in, prev_in.as_deref().unwrap()),
                    output_norm: mean_abs(last_out.as_deref().unwrap()),
                }
            } else {
                Signal::None
            };
            let d = neu.decide(Lane::COND, step, signal).unwrap();
            (d.compute(), Rec::of(&d))
        } else {
            let (change, norm) = if old.needs_cond_signal(step) {
                (
                    mean_abs_delta(&latent_in, prev_in.as_deref().unwrap()),
                    mean_abs(last_out.as_deref().unwrap()),
                )
            } else {
                (0.0, 1.0)
            };
            let d = old.decide_cond(step, change, norm);
            let rec = Rec {
                compute: d.compute,
                reason: d.reason,
                acc: d.accumulator.to_bits(),
                indicator: ob(d.estimate),
            };
            (d.compute, rec)
        };
        prev_in = Some(latent_in.clone());
        recs.push(rec);
        let cond = if compute {
            let out = model(&latent_in, step, 1.0, &w);
            let (full_in, out_change) = match (&last_in, &last_out) {
                (Some(pi), Some(po)) => (mean_abs_delta(&latent_in, pi), mean_abs_delta(&out, po)),
                _ => (0.0, 0.0),
            };
            if new {
                neu.note_computed(
                    Lane::COND,
                    Refresh::Drift {
                        full_input_change: full_in,
                        output_change: out_change,
                        output_norm: None,
                    },
                )
                .unwrap();
            } else {
                let _ = old.note_cond_computed(full_in, out_change);
            }
            last_in = Some(latent_in.clone());
            last_out = Some(out.clone());
            res_c = Some(sub(&out, &latent_in));
            out
        } else {
            add(&latent_in, res_c.as_deref().unwrap())
        };
        let follow = if new {
            neu.uncond_compute(Lane::COND, res_u.is_some())
        } else {
            old.uncond_compute(res_u.is_some())
        };
        let uncond = if follow {
            let out = model(&latent_in, step, 0.0, &w);
            res_u = Some(sub(&out, &latent_in));
            out
        } else {
            add(&latent_in, res_u.as_deref().unwrap())
        };
        x = euler(&x, &lincomb(1.0 - scale, &uncond, scale, &cond));
    }
    (recs, bits(&x))
}

#[test]
fn wan_easycache_matches_the_pre_b4a_controller() {
    for (n, thr, retain, cool) in [
        (50, 0.05, 7, 1),
        (50, 0.036, 7, 1),
        (40, 0.10, 5, 1),
        (30, 0.5, 2, 3),
        (12, 2.0, 0, 0),
    ] {
        let a = wan_easy_run(false, n, thr, retain, cool);
        let b = wan_easy_run(true, n, thr, retain, cool);
        assert_eq!(a, b, "{n} {thr} {retain} {cool}");
    }
    assert!(wan_easy_run(false, 12, 2.0, 0, 0)
        .0
        .iter()
        .any(|r| !r.compute));
}

/// SANA `denoise`: `(0.0, 0.0)` on a forced step, `compute || residual.is_none()`.
fn sana_easy_run(new: bool, n: usize, thr: f64) -> (Vec<Rec>, Vec<u32>) {
    let mut rng = Lcg(11);
    let w = rng.vec(48, 1.0);
    let mut x = rng.vec(48, 2.0);
    let mut old = legacy::wan_easy::EasyCache::new(n, thr, 7, 1).unwrap();
    let mut neu = wan::easycache(n, thr, 7, 1).unwrap();
    let (mut prev_input, mut last_full, mut residual) = (
        None::<Vec<f32>>,
        None::<(Vec<f32>, Vec<f32>)>,
        None::<Vec<f32>>,
    );
    let mut recs = Vec::new();
    for step in 0..n {
        let compute = if new {
            let signal = if neu.needs_signal(Lane::COND, step) {
                Signal::Drift {
                    input_change: mean_abs_delta(&x, prev_input.as_deref().unwrap()),
                    output_norm: mean_abs(&last_full.as_ref().unwrap().1),
                }
            } else {
                Signal::None
            };
            let d = neu.decide(Lane::COND, step, signal).unwrap();
            recs.push(Rec::of(&d));
            d.compute() || residual.is_none()
        } else {
            let (dx, onorm) = if old.needs_cond_signal(step) {
                (
                    mean_abs_delta(&x, prev_input.as_deref().unwrap()),
                    mean_abs(&last_full.as_ref().unwrap().1),
                )
            } else {
                (0.0, 0.0)
            };
            let d = old.decide_cond(step, dx, onorm);
            recs.push(Rec {
                compute: d.compute,
                reason: d.reason,
                acc: d.accumulator.to_bits(),
                indicator: ob(d.estimate),
            });
            d.compute || residual.is_none()
        };
        let noise = if compute {
            let out = model(&x, step, 1.0, &w);
            let (din, dout) = match &last_full {
                Some((li, lo)) => (mean_abs_delta(&x, li), mean_abs_delta(&out, lo)),
                None => (0.0, 0.0),
            };
            if new {
                neu.note_computed(
                    Lane::COND,
                    Refresh::Drift {
                        full_input_change: din,
                        output_change: dout,
                        output_norm: None,
                    },
                )
                .unwrap();
            } else {
                old.note_cond_computed(din, dout);
            }
            residual = Some(sub(&out, &x));
            last_full = Some((x.clone(), out.clone()));
            out
        } else {
            add(&x, residual.as_deref().unwrap())
        };
        prev_input = Some(x.clone());
        x = euler(&x, &noise);
    }
    (recs, bits(&x))
}

#[test]
fn sana_easycache_matches_the_pre_b4a_controller() {
    for thr in [0.1, 0.02, 1.0] {
        assert_eq!(
            sana_easy_run(false, 50, thr),
            sana_easy_run(true, 50, thr),
            "{thr}"
        );
    }
}

/// A14B `begin_a14b_tail` / `finish_a14b_tail`, the experts switching at
/// the boundary (one lane each).
fn a14b_run(new: bool, n: usize, thr: f64, max_reuse: usize) -> (Vec<Rec>, Vec<u32>) {
    use legacy::wan_a14b::{A14bCacheController, A14bExpert as Old};
    let mut rng = Lcg(23);
    let w = rng.vec(40, 1.0);
    let mut x = rng.vec(40, 2.0);
    let mut old = A14bCacheController::new(n, thr, 5, 3, max_reuse).unwrap();
    let mut neu = wan::a14b_easycache(n, thr, 5, 3, max_reuse).unwrap();
    let mut recs = Vec::new();
    // Per expert: previous input, last compute (input, residual), residuals.
    let mut prev = [None::<Vec<f32>>, None];
    let mut last = [None::<(Vec<f32>, Vec<f32>)>, None];
    let mut res = [[None::<Vec<f32>>, None], [None, None]];
    for step in 0..n {
        let e = usize::from(step >= n / 2);
        let (old_e, new_lane) = if e == 0 {
            (Old::HighNoise, wan::A14bExpert::HighNoise.lane())
        } else {
            (Old::LowNoise, wan::A14bExpert::LowNoise.lane())
        };
        // Block 0 always runs: the prefix.
        let prefix = model(&x, step, 0.5, &w);
        let needs = if new {
            neu.needs_signal(new_lane, step)
        } else {
            old.needs_input_signal(old_e, step)
        };
        let change = if needs {
            mean_abs_delta(&prefix, prev[e].as_deref().unwrap())
        } else {
            0.0
        };
        prev[e] = Some(prefix.clone());
        let compute = if new {
            let signal = if needs {
                Signal::Distance(change)
            } else {
                Signal::None
            };
            let d = neu.decide(new_lane, step, signal).unwrap();
            recs.push(Rec::of(&d));
            d.compute()
        } else {
            let d = old.decide(old_e, step, change);
            recs.push(Rec {
                compute: d.compute,
                reason: d.reason,
                acc: d.accumulator.to_bits(),
                indicator: ob(d.estimate),
            });
            d.compute
        };
        let mut reuse = !compute;
        let mut outs = Vec::new();
        for (branch, cond) in [(0usize, true), (1, false)] {
            let have = res[e][branch].is_some();
            let this_reuse = if reuse && !have {
                reuse = false;
                false
            } else {
                reuse
            };
            if this_reuse && cond {
                if new {
                    neu.note_reused(new_lane);
                } else {
                    old.note_reused(old_e);
                }
            }
            let hidden = if this_reuse {
                add(&prefix, res[e][branch].as_deref().unwrap())
            } else {
                let hidden = model(&prefix, step, branch as f32, &w);
                let residual = sub(&hidden, &prefix);
                if cond {
                    let (full_in, out_change) = match &last[e] {
                        Some((pi, po)) => {
                            (mean_abs_delta(&prefix, pi), mean_abs_delta(&residual, po))
                        }
                        None => (0.0, 0.0),
                    };
                    let norm = mean_abs(&residual);
                    if new {
                        neu.note_computed(
                            new_lane,
                            Refresh::Drift {
                                full_input_change: full_in,
                                output_change: out_change,
                                output_norm: Some(norm),
                            },
                        )
                        .unwrap();
                    } else {
                        old.note_computed(old_e, full_in, out_change, norm);
                    }
                    last[e] = Some((prefix.clone(), residual.clone()));
                }
                res[e][branch] = Some(residual);
                hidden
            };
            outs.push(hidden);
        }
        x = euler(&x, &lincomb(-2.0, &outs[1], 3.0, &outs[0]));
    }
    (recs, bits(&x))
}

#[test]
fn a14b_easycache_matches_the_pre_b4a_controller() {
    for (n, thr, max_reuse) in [(40, 0.30, 1), (40, 1.5, 3), (24, 0.05, 0), (40, 4.0, 0)] {
        let a = a14b_run(false, n, thr, max_reuse);
        assert_eq!(
            a,
            a14b_run(true, n, thr, max_reuse),
            "{n} {thr} {max_reuse}"
        );
    }
    assert!(a14b_run(false, 40, 4.0, 0).0.iter().any(|r| !r.compute));
}

/// LingBot `denoise`: the cached CFG velocity while the latent stays close.
fn lingbot_run(
    new: bool,
    cfg: lingbot::EasyCacheConfig,
    n: usize,
) -> (Vec<bool>, Vec<u32>, (usize, usize)) {
    let mut rng = Lcg(31);
    let w = rng.vec(32, 0.2);
    let mut latents = rng.vec(32, 1.0);
    let mut old = legacy::lingbot_easy::EasyCache::new(
        legacy::lingbot_easy::EasyCacheConfig {
            threshold: cfg.threshold,
            head_steps: cfg.head_steps,
            tail_steps: cfg.tail_steps,
            max_reuse: cfg.max_reuse,
        },
        n,
    );
    let mut neu = lingbot::EasyCache::new(cfg, n);
    let mut last_v = None::<Vec<f32>>;
    let mut seen = Vec::new();
    for i in 0..n {
        let reuse = if new {
            last_v.is_some() && neu.reuse(i, &latents).unwrap()
        } else {
            last_v.is_some() && old.reuse(i, &latents)
        };
        let v = if reuse {
            last_v.clone().unwrap()
        } else {
            let v = lincomb(1.0, &model(&latents, i, 1.0, &w), -0.5, &latents);
            if new {
                neu.computed(&latents).unwrap();
            } else {
                old.computed(&latents);
            }
            last_v = Some(v.clone());
            v
        };
        seen.push(reuse);
        latents = lincomb(1.0, &latents, -0.02, &v);
    }
    let counts = if new {
        (neu.computed_steps, neu.reused_steps)
    } else {
        (old.computed_steps, old.reused_steps)
    };
    (seen, bits(&latents), counts)
}

#[test]
fn lingbot_easycache_matches_the_pre_b4a_controller() {
    let cfgs = [
        lingbot::EasyCacheConfig::BASE,
        lingbot::EasyCacheConfig::REFINER,
        lingbot::EasyCacheConfig {
            threshold: 10.0,
            head_steps: 1,
            tail_steps: 1,
            max_reuse: 1,
        },
        lingbot::EasyCacheConfig {
            threshold: 10.0,
            head_steps: 0,
            tail_steps: 0,
            max_reuse: 0,
        },
    ];
    for cfg in cfgs {
        for n in [8, 40] {
            let a = lingbot_run(false, cfg, n);
            assert_eq!(a, lingbot_run(true, cfg, n), "{cfg:?} {n}");
        }
    }
    assert!(lingbot_run(false, cfgs[2], 8).0.iter().any(|r| *r));
}

// ---------------------------------------------------------------------------
// TeaCache: Wan Sol, H3, Cosmos, TeaCache4Wan2.1
// ---------------------------------------------------------------------------

/// The Wan device `relative_l1` on the host (`wan/transformer.rs:1202`).
fn wan_rel(a: &[f32], b: &[f32]) -> f64 {
    let (num, den) = super::distance::abs_sums(a, b);
    let n = a.len().max(1) as f64;
    (num / n) / (den / n).max(1e-8)
}

/// Wan Sol TeaCache, `begin_sol_tea` / `finish_sol_tea`; even steps run the
/// two branches one by one, odd steps as one CFG batch.
#[allow(clippy::too_many_arguments)]
fn wan_tea_run(
    new: bool,
    n: usize,
    thr: f64,
    start: usize,
    end: usize,
    max_hits: usize,
    periodic: usize,
    coefficients: &[f64],
) -> (Vec<Rec>, Vec<u32>) {
    use legacy::wan_tea::{SolTeaCache, TeaBranch};
    let mut rng = Lcg(41);
    let w = rng.vec(32, 1.0);
    let mut x = rng.vec(32, 2.0);
    let mut old =
        SolTeaCache::new(thr, start, end, max_hits, periodic, coefficients.to_vec()).unwrap();
    let mut neu =
        wan::sol_teacache(thr, start, end, max_hits, periodic, coefficients.to_vec()).unwrap();
    let lane = |b: TeaBranch| match b {
        TeaBranch::Cond => Lane::COND,
        TeaBranch::Uncond => Lane::UNCOND,
    };
    let mut signals = [None::<Vec<f32>>, None];
    let mut res = [None::<Vec<f32>>, None];
    let mut batch_res = None::<Vec<f32>>;
    let mut recs = Vec::new();
    let decide = |old: &mut SolTeaCache,
                  neu: &mut super::TeaCache,
                  b: TeaBranch,
                  step: usize,
                  sig: &[f32],
                  prev: Option<&Vec<f32>>,
                  recs: &mut Vec<Rec>| {
        if new {
            let s = if neu.needs_signal(lane(b), step) {
                Signal::Distance(wan_rel(sig, prev.unwrap()))
            } else {
                Signal::None
            };
            let d = neu.decide(lane(b), step, s).unwrap();
            recs.push(Rec::of(&d));
            d.compute()
        } else {
            let rel = if old.needs_signal(b, step) {
                wan_rel(sig, prev.unwrap())
            } else {
                0.0
            };
            let d = old.decide(b, step, rel);
            recs.push(Rec {
                compute: d.compute,
                reason: d.reason,
                acc: d.accumulator.to_bits(),
                indicator: ob(d.indicator),
            });
            d.compute
        }
    };
    for step in 0..n {
        // The timestep projection: a smooth function of the step.
        let tproj: Vec<f32> = w
            .iter()
            .map(|v| v * (1.0 - step as f32 / n as f32))
            .collect();
        let tproj_u: Vec<f32> = tproj.iter().map(|v| v * 0.97 + 0.01).collect();
        let outs = if step % 2 == 0 {
            let mut outs = Vec::new();
            for (b, sig, branch) in [
                (TeaBranch::Cond, &tproj, 1.0f32),
                (TeaBranch::Uncond, &tproj_u, 0.0),
            ] {
                let i = b as usize;
                let compute = decide(
                    &mut old,
                    &mut neu,
                    b,
                    step,
                    sig,
                    signals[i].as_ref(),
                    &mut recs,
                );
                signals[i] = Some(sig.clone());
                let hidden = if compute {
                    let h = model(&x, step, branch, &w);
                    if new {
                        neu.note_computed(lane(b), Refresh::None).unwrap();
                    } else {
                        old.note_computed(b);
                    }
                    res[i] = Some(sub(&h, &x));
                    h
                } else {
                    if new {
                        neu.note_reused(lane(b));
                    } else {
                        old.note_reused(b);
                    }
                    add(&x, res[i].as_deref().unwrap())
                };
                outs.push(hidden);
            }
            outs
        } else {
            let mut compute = false;
            for (b, sig) in [(TeaBranch::Uncond, &tproj_u), (TeaBranch::Cond, &tproj)] {
                let i = b as usize;
                compute |= decide(
                    &mut old,
                    &mut neu,
                    b,
                    step,
                    sig,
                    signals[i].as_ref(),
                    &mut recs,
                );
                signals[i] = Some(sig.clone());
            }
            let both: Vec<f32> = if compute || batch_res.is_none() {
                let mut h = model(&x, step, 1.0, &w);
                h.extend(model(&x, step, 0.0, &w));
                for b in [TeaBranch::Uncond, TeaBranch::Cond] {
                    if new {
                        neu.note_computed(lane(b), Refresh::None).unwrap();
                    } else {
                        old.note_computed(b);
                    }
                }
                let mut xx = x.clone();
                xx.extend(x.iter());
                batch_res = Some(sub(&h, &xx));
                h
            } else {
                for b in [TeaBranch::Uncond, TeaBranch::Cond] {
                    if new {
                        neu.note_reused(lane(b));
                    } else {
                        old.note_reused(b);
                    }
                }
                let mut xx = x.clone();
                xx.extend(x.iter());
                add(&xx, batch_res.as_deref().unwrap())
            };
            vec![both[..x.len()].to_vec(), both[x.len()..].to_vec()]
        };
        x = euler(&x, &lincomb(3.0, &outs[0], -2.0, &outs[1]));
    }
    (recs, bits(&x))
}

/// `(steps, threshold, start, end, max_hits, periodic, coefficients)`.
type WanTeaCase = (usize, f64, usize, usize, usize, usize, &'static [f64]);

#[test]
fn wan_sol_teacache_matches_the_pre_b4a_controller() {
    let cases: [WanTeaCase; 6] = [
        (20, 0.12, 2, 18, 0, 0, &[1.0, 0.0]),
        (20, 0.5, 2, 18, 0, 0, &[1.0, 0.0]),
        (20, 0.5, 1, 19, 2, 0, &[1.0, 0.0]),
        (20, 0.5, 1, 19, 0, 3, &[1.0, 0.0]),
        (30, 0.3, 0, 30, 2, 4, &[2.0, 0.5, 0.0]),
        (30, 3.0, 2, 28, 0, 0, &[-3.0, 4.0, 1.0, 0.01]),
    ];
    let mut reused = 0;
    for (n, thr, start, end, hits, periodic, c) in cases {
        let a = wan_tea_run(false, n, thr, start, end, hits, periodic, c);
        let b = wan_tea_run(true, n, thr, start, end, hits, periodic, c);
        reused += a.0.iter().filter(|r| !r.compute).count();
        assert_eq!(a, b, "{n} {thr} {start} {end} {hits} {periodic} {c:?}");
    }
    assert!(reused > 0);
}

/// H3 `begin_sol_tea` / `finish_sol_tea` with `end_denoise` between two
/// requests.
fn h3_run(
    new: bool,
    thr: f64,
    retain: usize,
    cool: usize,
    n: usize,
    c: &[f64],
) -> (Vec<Rec>, Vec<u32>) {
    let mut rng = Lcg(53);
    let w = rng.vec(48, 1.0);
    let mut x = rng.vec(48, 2.0);
    let mut old = legacy::h3_tea::H3TeaCache::new(thr, retain, cool, n, c.to_vec()).unwrap();
    let mut neu = h3::teacache(thr, retain, cool, n, c.to_vec()).unwrap();
    let mut recs = Vec::new();
    for _request in 0..2 {
        let (mut signal, mut residual) = (None::<Vec<f32>>, None::<Vec<f32>>);
        for step in 0..n {
            // The block-0 modulated probe: the hidden through a fixed map.
            let probe = model(&x, step, 0.25, &w);
            let compute = if new {
                let s = if neu.needs_signal(Lane::ONLY, step) {
                    let (d, p) = super::distance::abs_sums(&probe, signal.as_deref().unwrap());
                    Signal::Distance(h3::relative_l1_from_sums(d, p))
                } else {
                    Signal::None
                };
                let d = neu.decide(Lane::ONLY, step, s).unwrap();
                recs.push(Rec::of(&d));
                d.compute()
            } else {
                let rel = if old.needs_signal(step) {
                    legacy::h3_tea::relative_l1(&probe, signal.as_deref().unwrap())
                } else {
                    0.0
                };
                let d = old.decide(step, rel);
                recs.push(Rec {
                    compute: d.compute,
                    reason: d.reason,
                    acc: d.accumulator.to_bits(),
                    indicator: ob(d.indicator),
                });
                d.compute
            };
            signal = Some(probe);
            let hidden = if compute {
                let h = model(&x, step, 1.0, &w);
                if new {
                    neu.note_computed(Lane::ONLY, Refresh::None).unwrap();
                } else {
                    old.note_computed();
                }
                residual = Some(sub(&h, &x));
                h
            } else {
                add(&x, residual.as_deref().unwrap())
            };
            x = euler(&x, &hidden);
        }
        if new {
            neu.reset();
        } else {
            old.reset();
        }
    }
    (recs, bits(&x))
}

#[test]
fn h3_teacache_matches_the_pre_b4a_controller() {
    let cases: [(f64, usize, usize, usize, &[f64]); 5] = [
        (0.10, 5, 1, 49, &[1.0, 0.0]),
        (1.0, 3, 3, 8, &[1.0, 0.0]),
        (0.5, 1, 1, 4, &[1.0, 0.0]),
        (0.3, 2, 1, 20, &[0.5, 1.0, 0.0]),
        (0.1, 8, 0, 8, &[1.0, 0.0]),
    ];
    for (thr, retain, cool, n, c) in cases {
        let a = h3_run(false, thr, retain, cool, n, c);
        assert_eq!(
            a,
            h3_run(true, thr, retain, cool, n, c),
            "{thr} {retain} {cool} {n}"
        );
    }
    assert!(h3_run(false, 1.0, 3, 3, 8, &[1.0, 0.0])
        .0
        .iter()
        .any(|r| !r.compute));
    // The relative L1 the host path computes is the same function.
    let mut rng = Lcg(5);
    let (a, b) = (rng.vec(100, 1.0), rng.vec(100, 1.0));
    assert_eq!(
        legacy::h3_tea::relative_l1(&a, &b).to_bits(),
        h3::relative_l1(&a, &b).to_bits()
    );
}

/// Cosmos `begin_sol_tea` (cond decides, uncond follows) and Cosmos3's loop.
fn cosmos_run(new: bool, n: usize) -> (Vec<bool>, Vec<u32>) {
    let mut rng = Lcg(61);
    let w = rng.vec(24, 1.0);
    let mut x = rng.vec(24, 2.0);
    let mut old = legacy::cosmos_tea::SolCosmosTea::official();
    let mut neu = cosmos::SolCosmosTea::official();
    let (mut rc, mut ru) = (None::<Vec<f32>>, None::<Vec<f32>>);
    let mut seen = Vec::new();
    for step in 0..n {
        // The time embed drifts slowly after step 10, so reuses happen.
        let temb: Vec<f32> = w
            .iter()
            .map(|v| v * (1.0 + 0.001 * step as f32 + 0.3 * f32::from(u8::from(step < 10))))
            .collect();
        let compute = if new {
            neu.decide(step, &temb).unwrap()
        } else {
            old.decide(step, &temb)
        };
        let follow = if new { neu.follow() } else { old.follow() };
        seen.push(compute);
        let mut outs = Vec::new();
        for (cmp, slot, branch) in [(compute, &mut rc, 1.0f32), (follow, &mut ru, 0.0)] {
            let h = if cmp {
                let h = model(&x, step, branch, &w);
                *slot = Some(sub(&h, &x));
                h
            } else {
                add(&x, slot.as_deref().unwrap())
            };
            outs.push(h);
        }
        x = euler(&x, &lincomb(-1.0, &outs[1], 2.0, &outs[0]));
    }
    (seen, bits(&x))
}

#[test]
fn cosmos_teacache_matches_the_pre_b4a_controller() {
    for n in [12, 35, 60] {
        let a = cosmos_run(false, n);
        assert_eq!(a, cosmos_run(true, n), "{n}");
    }
    assert!(cosmos_run(false, 35).0.iter().any(|c| !c));
    // The window alone, a zero cap included (every step computes).
    for (thr, start, cap) in [(1.15, 10, 3), (0.2, 2, 1), (0.2, 0, 0), (5.0, 3, 6)] {
        let mut old = legacy::cosmos_tea::TeaCacheWindow::official();
        old.threshold = thr;
        old.start_step = start;
        old.max_consecutive = cap;
        let mut neu = cosmos::teacache_window(thr, start, cap);
        let mut rng = Lcg(3);
        for step in 0..40 {
            let ind = f64::from(rng.next().abs());
            let a = old.decide(step, ind);
            let d = neu.decide(Lane::ONLY, step, Signal::Distance(ind)).unwrap();
            if d.reuse() {
                neu.note_reused(Lane::ONLY);
            }
            assert_eq!(a, d.compute(), "{thr} {start} {cap} {step}");
        }
    }
}

/// TeaCache4Wan2.1 `dit_cfg`: decide on the latents once a previous step
/// exists; a reuse returns `latents + prev_residual`.
/// Per decision: reuse and the f32 accumulator's bits (the legacy field,
/// the new state's f64 accumulator cast back to f32).
fn tea4wan_run(
    new: bool,
    thresh: f32,
    ret: bool,
    n: usize,
) -> (Vec<(bool, Option<u32>)>, Vec<u32>) {
    let coefficients = if ret {
        legacy::wan_teacache4wan21::RET_COEFFICIENTS
    } else {
        legacy::wan_teacache4wan21::COEFFICIENTS
    };
    let mut old = legacy::wan_teacache4wan21::TeaCache::new(thresh, coefficients);
    let mut neu = wan::TeaCache4Wan21::new(thresh, ret).unwrap();
    let mut rng = Lcg(71);
    let w = rng.vec(32, 0.5);
    let mut x = rng.vec(32, 1.0);
    let (mut prev, mut residual) = (None::<Vec<f32>>, None::<Vec<f32>>);
    let mut seen = Vec::new();
    let mut acc = None;
    for step in 0..n {
        let reuse = match &prev {
            Some(p) => {
                let (num, den) = super::distance::abs_sums(&x, p);
                let rel = num / (den + 1e-6);
                if new {
                    let r = neu.reuse(rel).unwrap();
                    acc = Some((neu.state().accumulator(Lane::ONLY) as f32).to_bits());
                    r
                } else {
                    let r = old.maybe_reuse(rel);
                    acc = Some(old.accumulated.to_bits());
                    r
                }
            }
            None => false,
        };
        seen.push((reuse, acc.take()));
        let out = if reuse {
            add(&x, residual.as_deref().unwrap())
        } else {
            let out = model(&x, step, 1.0, &w);
            if !new {
                old.store();
            }
            prev = Some(x.clone());
            residual = Some(sub(&out, &x));
            out
        };
        x = euler(&x, &out);
    }
    (seen, bits(&x))
}

#[test]
fn teacache4wan21_matches_the_pre_b4a_controller() {
    for (thresh, ret) in [
        (0.08f32, false),
        (0.2, false),
        (0.05, true),
        (5.0, true),
        (0.0, false),
    ] {
        for n in [10, 50] {
            let a = tea4wan_run(false, thresh, ret, n);
            assert_eq!(a, tea4wan_run(true, thresh, ret, n), "{thresh} {ret} {n}");
        }
    }
    assert!(tea4wan_run(false, 5.0, true, 50).0.iter().any(|r| r.0));
}

/// TeaCache4Wan2.1 accumulates in f32 and compares `acc_f32 < thr_f32`. On
/// a recorded signal, find the decision where the f32 running sum lies
/// above the f64 one, and put the threshold exactly on the f32 sum: the
/// f32 rule computes there (a tie, `!(acc < thr)`), an f64 accumulator
/// would reuse. Old and new must agree on every decision and every
/// accumulator bit.
#[test]
fn teacache4wan21_f32_accumulator_ties_and_disagrees_with_f64() {
    let poly = |x: f64| {
        let c = legacy::wan_teacache4wan21::COEFFICIENTS;
        ((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x) + c[4]
    };
    let mut rng = Lcg(101);
    let rels: Vec<f64> = (0..64)
        .map(|_| f64::from(rng.next().abs()) * 0.02)
        .collect();
    // The running sums (the warm-up decision accumulates too).
    let (mut s32, mut s64) = (0.0f32, 0.0f64);
    let mut split = None;
    for (k, &r) in rels.iter().enumerate() {
        let ind = poly(r);
        assert!(ind > 0.0);
        s32 += ind as f32;
        s64 += ind;
        if k >= 2 && s64 < f64::from(s32) {
            split = Some((k, s32));
            break;
        }
    }
    let (k, thr) = split.unwrap();
    // An f64 accumulator would reuse at decision k; the f32 rule computes.
    assert!(s64 < f64::from(thr));
    let mut old =
        legacy::wan_teacache4wan21::TeaCache::new(thr, legacy::wan_teacache4wan21::COEFFICIENTS);
    old.store();
    let mut neu = wan::TeaCache4Wan21::new(thr, false).unwrap();
    let mut decided = Vec::new();
    for &r in &rels[..=k] {
        let a = old.maybe_reuse(r);
        let b = neu.reuse(r).unwrap();
        let acc = neu.state().accumulator(Lane::ONLY);
        assert_eq!(a, b);
        decided.push(b);
        if a {
            // The stored accumulator is exactly the f32 sum, not a wider one.
            assert_eq!(acc.to_bits(), f64::from(old.accumulated).to_bits());
        }
    }
    // Decisions 1..k reuse (0 is the warm-up), k computes on the tie.
    assert!(!decided[0]);
    assert!(decided[1..k].iter().all(|&r| r), "{decided:?}");
    assert!(!decided[k], "the f32 tie computes");
    // The threshold compute cleared both accumulators.
    assert_eq!(old.accumulated, 0.0);
    assert_eq!(neu.state().accumulator(Lane::ONLY), 0.0);
}

/// One exact tie per comparison family, old against new: dyadic distances
/// that sum to the threshold exactly, so `acc >= thr` (H3, Wan Sol, Cosmos,
/// Wan EasyCache) and `!(acc < thr)` (LTX FBCache, A14B, LingBot) both
/// compute on the tie.
#[test]
fn threshold_ties_match_the_pre_b4a_controllers() {
    // H3: retain 1, no cool-down, threshold 0.125, distances 0.0625.
    let mut old = legacy::h3_tea::H3TeaCache::new(0.125, 1, 0, 8, vec![1.0, 0.0]).unwrap();
    let mut neu = h3::teacache(0.125, 1, 0, 8, vec![1.0, 0.0]).unwrap();
    let mut seq = Vec::new();
    for step in 0..6 {
        let a = old.decide(step, if old.needs_signal(step) { 0.0625 } else { 0.0 });
        let s = if neu.needs_signal(Lane::ONLY, step) {
            Signal::Distance(0.0625)
        } else {
            Signal::None
        };
        let b = neu.decide(Lane::ONLY, step, s).unwrap();
        assert_eq!(
            (a.compute, a.reason, a.accumulator),
            (b.compute(), b.reason, b.accumulator)
        );
        if a.compute {
            old.note_computed();
            neu.note_computed(Lane::ONLY, Refresh::None).unwrap();
        }
        seq.push(b.compute());
    }
    assert_eq!(seq, [true, false, true, false, true, false], "h3");

    // Wan Sol: start 1, threshold 0.125, the accumulator cleared by the compute.
    let mut old = legacy::wan_tea::SolTeaCache::new(0.125, 1, 100, 0, 0, vec![1.0, 0.0]).unwrap();
    let mut neu = wan::sol_teacache(0.125, 1, 100, 0, 0, vec![1.0, 0.0]).unwrap();
    let cond = legacy::wan_tea::TeaBranch::Cond;
    let mut seq = Vec::new();
    for step in 0..6 {
        let a = old.decide(
            cond,
            step,
            if old.needs_signal(cond, step) {
                0.0625
            } else {
                0.0
            },
        );
        let s = if neu.needs_signal(Lane::COND, step) {
            Signal::Distance(0.0625)
        } else {
            Signal::None
        };
        let b = neu.decide(Lane::COND, step, s).unwrap();
        assert_eq!(
            (a.compute, a.reason, a.accumulator),
            (b.compute(), b.reason, b.accumulator)
        );
        if a.compute {
            old.note_computed(cond);
            neu.note_computed(Lane::COND, Refresh::None).unwrap();
        } else {
            old.note_reused(cond);
            neu.note_reused(Lane::COND);
        }
        seq.push(b.compute());
    }
    assert_eq!(seq, [true, false, true, false, true, false], "wan sol");

    // Cosmos window: start 1, cap 10, threshold 0.125.
    let mut old = legacy::cosmos_tea::TeaCacheWindow::official();
    old.threshold = 0.125;
    old.start_step = 1;
    old.max_consecutive = 10;
    let mut neu = cosmos::teacache_window(0.125, 1, 10);
    let mut seq = Vec::new();
    for step in 0..6 {
        let a = old.decide(step, 0.0625);
        let b = neu
            .decide(Lane::ONLY, step, Signal::Distance(0.0625))
            .unwrap();
        if b.reuse() {
            neu.note_reused(Lane::ONLY);
        }
        assert_eq!(a, b.compute());
        seq.push(a);
    }
    assert_eq!(seq, [true, false, true, false, true, false], "cosmos");

    // Wan EasyCache: k = 0.5, input change 0.125, norm 1: estimate 0.0625,
    // threshold 0.125.
    let mut old = legacy::wan_easy::EasyCache::new(8, 0.125, 1, 0).unwrap();
    let mut neu = wan::easycache(8, 0.125, 1, 0).unwrap();
    let mut seq = Vec::new();
    for step in 0..6 {
        let a = old.decide_cond(step, 0.125, 1.0);
        let b = neu
            .decide(
                Lane::COND,
                step,
                Signal::Drift {
                    input_change: 0.125,
                    output_norm: 1.0,
                },
            )
            .unwrap();
        assert_eq!(
            (a.compute, a.reason, a.accumulator),
            (b.compute(), b.reason, b.accumulator)
        );
        if a.compute {
            old.note_cond_computed(1.0, 0.5);
            neu.note_computed(
                Lane::COND,
                Refresh::Drift {
                    full_input_change: 1.0,
                    output_change: 0.5,
                    output_norm: None,
                },
            )
            .unwrap();
        }
        seq.push(b.compute());
    }
    assert_eq!(seq, [true, true, false, true, false, true], "wan easycache");

    // LTX FBCache: warm-up 1, threshold 0.09375, distances 0.046875.
    let mut old = legacy::ltx_fb::FbCache::official();
    old.threshold = 0.09375;
    let mut neu = super::FbCache::new(ltx::FbCacheParams {
        name: "ltx2 fbcache",
        threshold: 0.09375,
        warmup: 1,
        max_consecutive: 10,
    });
    let mut seq = Vec::new();
    for step in 0..6 {
        old.begin_step(step);
        neu.begin_step(step);
        let a = old.decide(if old.needs_signal() { 0.046875 } else { 0.0 });
        let s = if neu.needs_signal_next() {
            Signal::Distance(0.046875)
        } else {
            Signal::None
        };
        let b = neu.decide_next(s).unwrap();
        assert_eq!(
            (!a.skip, a.reason, a.accumulator),
            (b.compute(), b.reason, b.accumulator)
        );
        if a.skip {
            old.note_reused(0);
            neu.note_reused(Lane(0));
        } else {
            old.note_computed(0);
            neu.note_computed(Lane(0), Refresh::None).unwrap();
        }
        seq.push(b.compute());
    }
    assert_eq!(seq, [true, false, true, false, true, false], "ltx fbcache");

    // A14B: start 1, no tail, no reuse cap; k = 0.5, input change 0.25,
    // norm 1: estimate 0.125, threshold 0.25.
    use legacy::wan_a14b::{A14bCacheController, A14bExpert as Old};
    let mut old = A14bCacheController::new(8, 0.25, 1, 0, 0).unwrap();
    let mut neu = wan::a14b_easycache(8, 0.25, 1, 0, 0).unwrap();
    let lane = wan::A14bExpert::HighNoise.lane();
    let mut seq = Vec::new();
    for step in 0..7 {
        let needs = old.needs_input_signal(Old::HighNoise, step);
        assert_eq!(needs, neu.needs_signal(lane, step));
        let a = old.decide(Old::HighNoise, step, if needs { 0.25 } else { 0.0 });
        let s = if needs {
            Signal::Distance(0.25)
        } else {
            Signal::None
        };
        let b = neu.decide(lane, step, s).unwrap();
        assert_eq!(
            (a.compute, a.reason, a.accumulator),
            (b.compute(), b.reason, b.accumulator)
        );
        if a.compute {
            old.note_computed(Old::HighNoise, 1.0, 0.5, 1.0);
            neu.note_computed(
                lane,
                Refresh::Drift {
                    full_input_change: 1.0,
                    output_change: 0.5,
                    output_norm: Some(1.0),
                },
            )
            .unwrap();
        } else {
            old.note_reused(Old::HighNoise);
            neu.note_reused(lane);
        }
        seq.push(b.compute());
    }
    assert_eq!(seq, [true, true, false, true, false, true, false], "a14b");

    // LingBot: the threshold is exactly the relative L1 of the step's latent.
    let reference = vec![1.0f32; 4];
    let latent = vec![1.25f32; 4];
    let rel = lingbot::EasyCache::relative_l1(&latent, &reference);
    let cfg = lingbot::EasyCacheConfig {
        threshold: rel,
        head_steps: 0,
        tail_steps: 0,
        max_reuse: 5,
    };
    let mut old = legacy::lingbot_easy::EasyCache::new(
        legacy::lingbot_easy::EasyCacheConfig {
            threshold: rel,
            head_steps: 0,
            tail_steps: 0,
            max_reuse: 5,
        },
        8,
    );
    let mut neu = lingbot::EasyCache::new(cfg, 8);
    old.computed(&reference);
    neu.computed(&reference).unwrap();
    let a = old.reuse(1, &latent);
    let b = neu.reuse(1, &latent).unwrap();
    assert_eq!(a, b);
    assert!(!b, "lingbot: rel == threshold computes");
    let nearer = vec![1.125f32; 4];
    assert!(old.reuse(2, &nearer) && neu.reuse(2, &nearer).unwrap());
}

// ---------------------------------------------------------------------------
// FbCache (LTX-2.5 stage 1)
// ---------------------------------------------------------------------------

/// `begin_fb_pass` / `decide_fb_after_block0` / `finish_fb`, three CFG
/// passes per step, block-0 residual signal (host relative L1).
fn fb_run(new: bool, thr: f64, max_consecutive: usize, n: usize) -> (Vec<Rec>, Vec<u32>) {
    let mut old = legacy::ltx_fb::FbCache::official();
    old.threshold = thr;
    old.max_consecutive = max_consecutive;
    let mut neu = super::FbCache::new(ltx::FbCacheParams {
        name: "ltx2 fbcache",
        threshold: thr,
        warmup: ltx::WARMUP,
        max_consecutive,
    });
    let mut rng = Lcg(83);
    let w = rng.vec(40, 1.0);
    let mut x = rng.vec(40, 2.0);
    let mut signals: Vec<Option<Vec<f32>>> = Vec::new();
    let mut res: Vec<Option<Vec<f32>>> = Vec::new();
    let mut recs = Vec::new();
    for step in 0..n {
        if new {
            neu.begin_step(step);
        } else {
            old.begin_step(step);
        }
        let mut outs = Vec::new();
        for pass in 0..3 {
            let after0 = model(&x, step, 0.1 * pass as f32, &w);
            // A zero block-0 residual on pass 2 of step 4 (inf distance next).
            let signal = if pass == 2 && step == 4 {
                vec![0.0; x.len()]
            } else {
                sub(&after0, &x)
            };
            let (skip, p) = if new {
                let cur = neu.current_pass();
                let s = if neu.needs_signal_next() {
                    Signal::Distance(ltx::relative_l1(&signal, signals[cur].as_deref().unwrap()))
                } else {
                    Signal::None
                };
                let d = neu.decide_next(s).unwrap();
                recs.push(Rec::of(&d));
                (d.reuse(), neu.last_pass())
            } else {
                let distance = if old.needs_signal() {
                    legacy::ltx_fb::relative_l1(
                        &signal,
                        signals[old.current_pass()].as_deref().unwrap(),
                    )
                } else {
                    0.0
                };
                let d = old.decide(distance);
                recs.push(Rec {
                    compute: !d.skip,
                    reason: d.reason,
                    acc: d.accumulator.to_bits(),
                    indicator: ob(d.distance),
                });
                (d.skip, old.last_pass())
            };
            while signals.len() <= p {
                signals.push(None);
                res.push(None);
            }
            signals[p] = Some(signal);
            let h = if skip {
                if new {
                    neu.note_reused(Lane(p));
                } else {
                    old.note_reused(p);
                }
                add(&x, res[p].as_deref().unwrap())
            } else {
                let h = model(&after0, step, pass as f32, &w);
                res[p] = Some(sub(&h, &x));
                if new {
                    neu.note_computed(Lane(p), Refresh::None).unwrap();
                } else {
                    old.note_computed(p);
                }
                h
            };
            outs.push(h);
        }
        let g = lincomb(2.0, &outs[0], -1.0, &outs[1]);
        x = euler(&x, &lincomb(1.5, &g, -0.5, &outs[2]));
    }
    (recs, bits(&x))
}

#[test]
fn ltx_fbcache_matches_the_pre_b4a_controller() {
    for (thr, cap, n) in [
        (0.08, 10, 30),
        (0.5, 10, 30),
        (0.5, 2, 30),
        (2.0, 0, 12),
        (3.0, 4, 40),
    ] {
        let a = fb_run(false, thr, cap, n);
        assert_eq!(a, fb_run(true, thr, cap, n), "{thr} {cap} {n}");
    }
    assert!(fb_run(false, 3.0, 4, 40).0.iter().any(|r| !r.compute));
}

// ---------------------------------------------------------------------------
// Taylor (Wan TaylorSeer lite)
// ---------------------------------------------------------------------------

/// `begin_taylor` / `update_taylor` / `predict_taylor` with host factors.
fn taylor_run(new: bool, interval: usize, warmup: usize, n: usize, order: usize) -> Vec<Vec<u32>> {
    let mut old = legacy::wan_taylor::TaylorSchedule::new(interval, warmup, n - 2, order).unwrap();
    let mut neu = wan::taylor(interval, warmup, n - 2, order).unwrap();
    let mut rng = Lcg(97);
    let w = rng.vec(16, 1.0);
    let x = rng.vec(16, 2.0);
    let mut factors: Vec<Vec<f32>> = Vec::new();
    let mut last_update: Option<isize> = None;
    let mut outs = Vec::new();
    for f in 0..n {
        let compute = if new {
            neu.decide(Lane::ONLY, f, Signal::None).unwrap().compute()
        } else {
            old.begin_forward().compute
        };
        let out = if compute {
            let features = model(&x, f, 1.0, &w);
            if new {
                let delta = neu.update_delta().unwrap();
                factors =
                    super::taylor::host_update("t", &factors, &features, delta, order).unwrap();
                neu.note_computed(Lane::ONLY, Refresh::None).unwrap();
            } else {
                let current = old.current_step();
                let delta = last_update.map(|l| current - l).unwrap_or(1);
                factors =
                    legacy::wan_taylor::taylor_update(&factors, &features, delta, order).unwrap();
                last_update = Some(current);
            }
            features
        } else if new {
            let offset = neu.forecast_offset().unwrap();
            super::taylor::host_predict("t", &factors, offset).unwrap()
        } else {
            let offset = old.current_step() - last_update.unwrap();
            legacy::wan_taylor::taylor_predict(&factors, offset).unwrap()
        };
        outs.push(bits(&out));
    }
    outs
}

#[test]
fn taylor_matches_the_pre_b4a_schedule_and_factors() {
    for (interval, warmup, n, order) in [(3, 3, 50, 1), (3, 3, 20, 2), (2, 1, 30, 3), (5, 1, 40, 1)]
    {
        let a = taylor_run(false, interval, warmup, n, order);
        assert_eq!(
            a,
            taylor_run(true, interval, warmup, n, order),
            "{interval} {warmup} {n}"
        );
    }
    // The coefficient helpers are the device path's arithmetic too.
    for offset in [1isize, 2, 3, 5] {
        let c = super::taylor::forecast_coefficients(4, offset);
        let mut pow = 1.0f64;
        let mut fact = 1.0f64;
        for (order, coeff) in c.iter().enumerate().skip(1) {
            pow *= offset as f64;
            fact *= order as f64;
            assert_eq!(coeff.to_bits(), ((pow / fact) as f32).to_bits());
        }
    }
}
