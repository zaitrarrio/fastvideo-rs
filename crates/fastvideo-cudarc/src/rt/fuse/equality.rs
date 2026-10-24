//! The unified fusion module against the pre-B4b code (`legacy.rs`), bit for
//! bit (`docs/adr/0022-block-fusion.md`, "Evidence"):
//!
//! * every host twin (Wan's four ops, H3's four, `quant`'s row references)
//!   on random inputs over shapes that cross the norm kernels' 256-lane
//!   stride, every policy;
//! * H3's fused last residual + next norm against the two twins in a row;
//! * every kernel choice and mode argument the policies select against the
//!   literals the launchers passed, for every family op and every LTX-2
//!   residual / norm combination, and every kernel name in the NVRTC source;
//! * the switches against the old getters (overrides, nesting, the plan's
//!   installed value) and the plans' reads against the old reads, for every
//!   spelling of the settings.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::Arc;

use fastvideo_models::techniques::settings::{Live, Source};

use super::legacy;
use super::policy::{self, Kernel, Residual, RoundingPolicy};
use super::switch::{self, BlockFusion, Switch};
use crate::h3::fused16::{self, AdaRows, NormOut};
use crate::rt::fused::Rope;
use crate::rt::quant::bf16_round;
use crate::rt::tensor::{with_bf16_act, CudaTensor, TensorDType};
use crate::wan::fuse as wan;

/// xorshift64*: reproducible inputs without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[-scale, scale)`, not rounded to bf16.
    fn f(&mut self, scale: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (2.0 * u - 1.0) * scale
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f(scale)).collect()
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn bf16_tensor(v: Vec<f32>, shape: &[usize]) -> CudaTensor {
    CudaTensor::host_only_dtype(v, shape.to_vec(), TensorDType::Bf16)
}

fn f32_tensor(v: Vec<f32>, shape: &[usize]) -> CudaTensor {
    CudaTensor::host_only_dtype(v, shape.to_vec(), TensorDType::F32)
}

fn bits(t: &CudaTensor) -> Vec<u32> {
    t.host_cow()
        .expect("host values")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn same(label: &str, a: &CudaTensor, b: &CudaTensor) {
    assert_eq!(a.shape, b.shape, "{label}: shape");
    assert_eq!(a.is_bf16(), b.is_bf16(), "{label}: dtype");
    assert!(bits(a) == bits(b), "{label}: values differ");
}

fn norm_out(n: NormOut) -> CudaTensor {
    match n {
        NormOut::T(t) => t,
        #[cfg(feature = "cuda")]
        NormOut::Mx(_) => panic!("host twins return bf16"),
    }
}

// ---- Wan ---------------------------------------------------------------------

/// `(batch, seq, dim)` cases: one row, odd widths, widths past 256.
const WAN_SHAPES: [(usize, usize, usize); 5] =
    [(1, 1, 8), (1, 3, 33), (2, 3, 64), (2, 2, 300), (1, 2, 1536)];

#[test]
fn wan_residual_norms_match_legacy() {
    let mut rng = Rng(0x5eed_0001);
    for &(b, s, d) in &WAN_SHAPES {
        for round_inputs in [false, true] {
            let mut v = |n: usize, k: f32| {
                let x = rng.vec(n, k);
                if round_inputs {
                    x.into_iter().map(bf16_round).collect()
                } else {
                    x
                }
            };
            let shape = [b, s, d];
            let h = bf16_tensor(v(b * s * d, 3.0), &shape);
            let a = bf16_tensor(v(b * s * d, 2.0), &shape);
            let e = f32_tensor(v(b * 6 * d, 1.0), &[b, 6, d]);
            let w = f32_tensor(v(d, 1.5), &[d]);
            let bias = f32_tensor(v(d, 0.5), &[d]);
            let label = format!("wan {shape:?} rounded={round_inputs}");
            with_bf16_act(true, || {
                for (gate, eps) in [(2usize, 1e-6f32), (0, 1e-5)] {
                    let (nh, nn) = wan::self_residual_norm(&h, &a, &e, gate, &w, &bias, eps)
                        .expect("new")
                        .expect("bf16 path");
                    let (oh, on) =
                        legacy::wan::self_residual_norm(&h, &a, &e, gate, &w, &bias, eps)
                            .expect("legacy")
                            .expect("bf16 path");
                    same(&format!("{label} self hidden"), &nh, &oh);
                    same(&format!("{label} self normed"), &nn, &on);
                }
                for (sc, sh) in [(4usize, 3usize), (1, 0)] {
                    let (nh, nn) = wan::cross_residual_norm_mod(&h, &a, &e, sc, sh, 1e-6)
                        .expect("new")
                        .expect("bf16 path");
                    let (oh, on) = legacy::wan::cross_residual_norm_mod(&h, &a, &e, sc, sh, 1e-6)
                        .expect("legacy")
                        .expect("bf16 path");
                    same(&format!("{label} cross hidden"), &nh, &oh);
                    same(&format!("{label} cross normed"), &nn, &on);
                }
                for gate in [5usize, 1] {
                    let n = wan::gate_residual(&h, &a, &e, gate)
                        .expect("new")
                        .expect("bf16 path");
                    let o = legacy::wan::gate_residual(&h, &a, &e, gate)
                        .expect("legacy")
                        .expect("bf16 path");
                    same(&format!("{label} gate residual"), &n, &o);
                }
            });
            // f32 activations: both decline.
            with_bf16_act(false, || {
                assert!(wan::gate_residual(&h, &a, &e, 5).expect("new").is_none());
                assert!(legacy::wan::gate_residual(&h, &a, &e, 5)
                    .expect("legacy")
                    .is_none());
            });
        }
    }
}

#[test]
fn wan_qk_norm_rope_matches_legacy() {
    let mut rng = Rng(0x5eed_0002);
    for &(b, s, heads, d, extra) in &[
        (1usize, 1usize, 1usize, 4usize, 0usize),
        (1, 4, 2, 8, 16),
        (2, 3, 3, 64, 0),
        (1, 2, 12, 128, 128),
    ] {
        let width = heads * d;
        let proj_w = 3 * width + extra;
        let x = bf16_tensor(rng.vec(b * s * proj_w, 4.0), &[b, s, proj_w]);
        let w = f32_tensor(rng.vec(width, 2.0), &[width]);
        let ang = rng.vec(s * d, 6.0);
        let cos = f32_tensor(ang.iter().map(|a| a.cos()).collect(), &[s, d]);
        let sin = f32_tensor(ang.iter().map(|a| a.sin()).collect(), &[s, d]);
        for col_off in [0, width, 2 * width] {
            for with_rope in [false, true] {
                let rope = || {
                    with_rope.then_some(Rope {
                        cos: &cos,
                        sin: &sin,
                    })
                };
                let label = format!("wan qk b{b} s{s} h{heads} d{d} col{col_off} rope={with_rope}");
                with_bf16_act(true, || {
                    let n = wan::qk_norm_rope(&x, col_off, heads, &w, rope(), 1e-6)
                        .expect("new")
                        .expect("bf16 path");
                    let o = legacy::wan::qk_norm_rope(&x, col_off, heads, &w, rope(), 1e-6)
                        .expect("legacy")
                        .expect("bf16 path");
                    same(&label, &n, &o);
                });
            }
        }
    }
}

// ---- H3 ----------------------------------------------------------------------

fn ada(rng: &mut Rng, hidden: usize, t: usize, rows: usize) -> AdaRows {
    // f32 `[T, 6, H]`; scale slots hold `1 + scale` around 1.
    let tab: Vec<f32> = (0..t * 6 * hidden)
        .map(|i| {
            let v = rng.f(1.0);
            if (i / hidden) % 6 == 1 || (i / hidden) % 6 == 4 {
                1.0 + v
            } else {
                v
            }
        })
        .collect();
    let idx: Vec<u32> = (0..rows).map(|_| rng.below(t / 2 + 1) as u32).collect();
    AdaRows {
        tab: f32_tensor(tab, &[t, 6, hidden]),
        hidden,
        idx: Arc::new(idx),
        #[cfg(feature = "cuda")]
        idx_dev: None,
    }
}

/// `(rows, hidden)`: below, at and across the 256-lane stride.
const H3_SHAPES: [(usize, usize); 5] = [(1, 32), (3, 256), (2, 300), (4, 600), (2, 5120)];

#[test]
fn h3_norms_and_residuals_match_legacy() {
    let mut rng = Rng(0x5eed_0003);
    for &(rows, hidden) in &H3_SHAPES {
        let t = 8;
        let ada = ada(&mut rng, hidden, t, rows);
        // f32 operands: both twins round them to bf16 first.
        let x = f32_tensor(rng.vec(rows * hidden, 4.0), &[1, rows, hidden]);
        let u = bf16_tensor(rng.vec(rows * hidden, 2.0), &[1, rows, hidden]);
        let w = f32_tensor(rng.vec(hidden, 1.5), &[hidden]);
        let label = format!("h3 {rows}x{hidden}");
        for base in [0usize, 3] {
            for (sc, sh) in [(1usize, 0usize), (4, 3)] {
                let n = norm_out(
                    fused16::norm_mod(&x, &w, &ada, base, sc, sh, 1e-6, false).expect("new"),
                );
                let o = norm_out(
                    legacy::h3::norm_mod(&x, &w, &ada, base, sc, sh, 1e-6, false).expect("legacy"),
                );
                same(&format!("{label} norm_mod base {base}"), &n, &o);
            }
            let slots = (2usize, 4usize, 3usize);
            let (nh, nn) = fused16::res_gate_norm_mod(&x, &u, &w, &ada, base, slots, 1e-6, false)
                .expect("new");
            let (oh, on) =
                legacy::h3::res_gate_norm_mod(&x, &u, &w, &ada, base, slots, 1e-6, false)
                    .expect("legacy");
            same(&format!("{label} res_gate hidden"), &nh, &oh);
            same(
                &format!("{label} res_gate normed"),
                &norm_out(nn),
                &norm_out(on),
            );
            for gate in [5usize, 2] {
                let n = fused16::gate_residual(&x, &u, &ada, base, gate).expect("new");
                let o = legacy::h3::gate_residual(&x, &u, &ada, base, gate).expect("legacy");
                same(&format!("{label} gate_residual"), &n, &o);
            }
        }
    }
}

#[test]
fn h3_qk_norm_rope_matches_legacy() {
    let mut rng = Rng(0x5eed_0004);
    for &(b, s, heads, d) in &[
        (1usize, 1usize, 1usize, 16usize),
        (1, 5, 2, 64),
        (2, 3, 3, 128),
    ] {
        let inner = heads * d;
        let width = 4 * inner;
        let x = f32_tensor(rng.vec(b * s * width, 4.0), &[b, s, width]);
        let w = f32_tensor(rng.vec(d, 2.0), &[d]);
        for r in [0usize, 8, d] {
            let ang = rng.vec(s * r.max(1), 6.0);
            let cos = f32_tensor(ang.iter().take(s * r).map(|a| a.cos()).collect(), &[s, r]);
            let sin = f32_tensor(ang.iter().take(s * r).map(|a| a.sin()).collect(), &[s, r]);
            let rope = (r > 0).then_some((&cos, &sin));
            for col in [0, inner, 3 * inner] {
                let n = fused16::qk_norm_rope(&x, &w, rope, heads, d, col, 1e-6).expect("new");
                let o =
                    legacy::h3::qk_norm_rope(&x, &w, rope, heads, d, col, 1e-6).expect("legacy");
                same(
                    &format!("h3 qk b{b} s{s} h{heads} d{d} r{r} col{col}"),
                    &n,
                    &o,
                );
            }
        }
    }
}

#[test]
fn quant_row_references_match_legacy() {
    let mut rng = Rng(0x5eed_0005);
    for dim in [1usize, 7, 256, 257, 777, 5120] {
        let (x, w) = (rng.vec(dim, 4.0), rng.vec(dim, 1.5));
        let (sc, sh) = (rng.vec(dim, 1.0), rng.vec(dim, 1.0));
        let n = crate::rt::quant::norm_mod_row(&x, &w, &sc, &sh, 1e-6);
        let o = legacy::quant::norm_mod_row(&x, &w, &sc, &sh, 1e-6);
        assert!(
            n.iter()
                .map(|v| v.to_bits())
                .eq(o.iter().map(|v| v.to_bits())),
            "norm_mod_row {dim}"
        );
        for j in 0..dim {
            let (r, g, br) = (x[j], sc[j], w[j]);
            assert_eq!(
                crate::rt::quant::gate_residual_eager(r, g, br).to_bits(),
                legacy::quant::gate_residual_eager(r, g, br).to_bits(),
                "gate_residual_eager {r} {g} {br}"
            );
        }
    }
}

/// H3's fused last residual + next norm (`fvf_h3_gate_res_norm_mod`, device
/// only) has the rounding points of the two ops it replaces: the policy's
/// host reference equals the legacy twins run in a row.
#[test]
fn h3_gate_res_norm_mod_policy_is_the_two_op_chain() {
    let mut rng = Rng(0x5eed_0006);
    for dim in [32usize, 300, 5120] {
        let (res, br) = (rng.vec(dim, 4.0), rng.vec(dim, 2.0));
        let (g, w) = (rng.vec(dim, 1.0), rng.vec(dim, 1.5));
        let sc: Vec<f32> = rng.vec(dim, 1.0).iter().map(|v| 1.0 + v).collect();
        let sh = rng.vec(dim, 1.0);
        let out = super::host::res_norm_row(
            &policy::H3_GATE_RES_NORM_MOD,
            &super::host::RowOperands {
                x: &res,
                u: Some(&br),
                gate: Some(&g),
                w: Some(&w),
                scale: Some(&sc),
                shift: Some(&sh),
                eps: 1e-6,
                ..Default::default()
            },
        )
        .expect("host twin");
        let (rb, bb): (Vec<f32>, Vec<f32>) = (
            res.iter().map(|&v| bf16_round(v)).collect(),
            br.iter().map(|&v| bf16_round(v)).collect(),
        );
        let hidden: Vec<f32> = (0..dim)
            .map(|j| legacy::quant::gate_residual_eager(rb[j], g[j], bb[j]))
            .collect();
        let normed = legacy::quant::norm_mod_row(&hidden, &w, &sc, &sh, 1e-6);
        assert_eq!(out.hidden, Some(hidden), "hidden {dim}");
        assert_eq!(out.normed, Some(normed), "normed {dim}");
    }
}

/// `n` triples `(x, u, g)` (`x`, `u` bf16 values, `g` f32) where the f32 FMA
/// `fma(g, u, x)` and the f32 product-then-sum `x + u·g` round to different
/// bf16 values: sums aimed at a bf16 rounding midpoint, kept only when the
/// two contractions land on either side. Random inputs alone cross such a
/// boundary about once in 2^16 elements, too rarely to pin a contraction.
fn contraction_sensitive(rng: &mut Rng, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut xs, mut us, mut gs) = (Vec::new(), Vec::new(), Vec::new());
    while xs.len() < n {
        let x = bf16_round(rng.f(4.0));
        let t = bf16_round(x + rng.f(1.0));
        // The midpoint above |t| (a bf16 drops the low 16 bits of the f32).
        let m = f32::from_bits(t.to_bits() + 0x8000);
        let u = bf16_round(0.5 + rng.f(0.25).abs() * 6.0);
        let g = f32::from_bits(
            ((m - x) / u)
                .to_bits()
                .wrapping_add(rng.below(5) as u32)
                .wrapping_sub(2),
        );
        if !(g.is_finite() && m.is_finite()) {
            continue;
        }
        if bf16_round(g.mul_add(u, x)).to_bits() != bf16_round(x + u * g).to_bits() {
            xs.push(x);
            us.push(u);
            gs.push(g);
        }
    }
    (xs, us, gs)
}

/// The residuals' contraction (H3's FMA, Wan's product then sum) on inputs
/// where it decides the stored bf16 hidden, against the legacy twins.
#[test]
fn residual_contractions_are_pinned() {
    let mut rng = Rng(0x5eed_0007);
    let d = 64;
    let (x, u, g) = contraction_sensitive(&mut rng, d);
    // H3: one row, the gate slot holds `g` (base 0, idx 0, slot 2).
    let mut tab = rng.vec(6 * d, 1.0);
    tab[2 * d..3 * d].copy_from_slice(&g);
    let ada = AdaRows {
        tab: f32_tensor(tab.clone(), &[1, 6, d]),
        hidden: d,
        idx: Arc::new(vec![0]),
        #[cfg(feature = "cuda")]
        idx_dev: None,
    };
    let (xt, ut) = (
        f32_tensor(x.clone(), &[1, 1, d]),
        f32_tensor(u.clone(), &[1, 1, d]),
    );
    let w = f32_tensor(rng.vec(d, 1.5), &[d]);
    let (nh, _) =
        fused16::res_gate_norm_mod(&xt, &ut, &w, &ada, 0, (2, 4, 3), 1e-6, false).expect("new");
    let (oh, _) = legacy::h3::res_gate_norm_mod(&xt, &ut, &w, &ada, 0, (2, 4, 3), 1e-6, false)
        .expect("legacy");
    same("h3 fma residual", &nh, &oh);
    let muladd: Vec<u32> = (0..d)
        .map(|j| bf16_round(x[j] + u[j] * g[j]).to_bits())
        .collect();
    assert!(
        bits(&oh).iter().zip(&muladd).all(|(a, b)| a != b),
        "inputs discriminate"
    );
    // Wan: the same row, the gate in slot 2 of a one-batch table.
    let e = f32_tensor(tab, &[1, 6, d]);
    let (h16, a16) = (bf16_tensor(x, &[1, 1, d]), bf16_tensor(u, &[1, 1, d]));
    let bias = f32_tensor(rng.vec(d, 0.5), &[d]);
    with_bf16_act(true, || {
        let (nh, nn) = wan::self_residual_norm(&h16, &a16, &e, 2, &w, &bias, 1e-6)
            .expect("new")
            .expect("bf16 path");
        let (oh, on) = legacy::wan::self_residual_norm(&h16, &a16, &e, 2, &w, &bias, 1e-6)
            .expect("legacy")
            .expect("bf16 path");
        same("wan mul-add residual", &nh, &oh);
        same("wan mul-add residual normed", &nn, &on);
        let n = wan::gate_residual(&h16, &a16, &e, 2)
            .expect("new")
            .expect("bf16 path");
        let o = legacy::wan::gate_residual(&h16, &a16, &e, 2)
            .expect("legacy")
            .expect("bf16 path");
        same("wan mul-add gate residual", &n, &o);
    });
}

// ---- Kernel selection ----------------------------------------------------------

#[test]
fn every_policy_selects_the_legacy_kernel_and_modes() {
    use legacy::launch as old;
    let k = |p: &RoundingPolicy| p.kernel().expect("a kernel");
    let (name, r, l) = old::WAN_SELF;
    assert_eq!(
        k(&policy::WAN_SELF_RESIDUAL_NORM),
        Kernel::WanResLn {
            res_mode: r,
            ln_mode: l
        }
    );
    assert_eq!(k(&policy::WAN_SELF_RESIDUAL_NORM).name(), name);
    let (name, r, l) = old::WAN_CROSS;
    assert_eq!(
        k(&policy::WAN_CROSS_RESIDUAL_NORM_MOD),
        Kernel::WanResLn {
            res_mode: r,
            ln_mode: l
        }
    );
    assert_eq!(k(&policy::WAN_CROSS_RESIDUAL_NORM_MOD).name(), name);
    let (name, r, keep) = old::WAN_GATE_RESIDUAL;
    assert_eq!(
        k(&policy::WAN_GATE_RESIDUAL),
        Kernel::WanResGate {
            res_mode: r,
            keep_f32: keep
        }
    );
    assert_eq!(k(&policy::WAN_GATE_RESIDUAL).name(), name);
    // FASTVIDEO_WAN_FUSE=0: the same policies as two kernels.
    for (p, (r, l)) in [
        (
            policy::WAN_SELF_RESIDUAL_NORM,
            (old::WAN_SELF.1, old::WAN_SELF.2),
        ),
        (
            policy::WAN_CROSS_RESIDUAL_NORM_MOD,
            (old::WAN_CROSS.1, old::WAN_CROSS.2),
        ),
    ] {
        let ((rn, rm, keep), (ln, lm)) = old::wan_unfused(r, l);
        let (a, b) = p.split().expect("two kernels");
        assert_eq!(
            a,
            Kernel::WanResGate {
                res_mode: rm,
                keep_f32: keep
            }
        );
        assert_eq!((a.name(), b.name()), (rn, ln));
        assert_eq!(b, Kernel::WanLn { ln_mode: lm });
    }
    let (name, rp) = old::H3_GATE_RESIDUAL;
    assert_eq!(
        k(&policy::H3_GATE_RESIDUAL),
        Kernel::H3GateResidual { round_product: rp }
    );
    assert_eq!(k(&policy::H3_GATE_RESIDUAL).name(), name);
    assert_eq!(k(&policy::H3_NORM_MOD).name(), old::H3_NORM_MOD);
    assert_eq!(
        k(&policy::H3_RES_GATE_NORM_MOD).name(),
        old::H3_RES_GATE_NORM_MOD
    );
    assert_eq!(
        k(&policy::H3_GATE_RES_NORM_MOD).name(),
        old::H3_GATE_RES_NORM_MOD
    );
    // LTX: every residual and norm the launcher accepts.
    for (res, then_mod) in [
        (old::LtxRes::None, false),
        (old::LtxRes::None, true),
        (old::LtxRes::Plain, false),
        (old::LtxRes::Plain, true),
        (old::LtxRes::Gated, false),
        (old::LtxRes::Gated, true),
    ] {
        let residual = match res {
            old::LtxRes::None => Residual::None,
            old::LtxRes::Plain => Residual::Add,
            old::LtxRes::Gated => Residual::RoundedProduct,
        };
        let (name, res_mode, mode) = old::ltx(res, then_mod);
        let got = k(&policy::ltx_res_norm_mod(residual, then_mod));
        assert_eq!(
            got,
            Kernel::LtxResNormMod { res_mode, mode },
            "{res:?} {then_mod}"
        );
        assert_eq!(got.name(), name);
    }
    // Only Wan has a two-kernel form.
    assert!(policy::H3_NORM_MOD.split().is_err());
    assert!(policy::ltx_res_norm_mod(Residual::Add, false)
        .split()
        .is_err());
}

#[test]
fn the_ltx_family_maps_its_operands_onto_its_policies() {
    use crate::ltx2::fuse::{policy as ltx_policy, Mod, Norm, Residual as LtxResidual};
    let t = f32_tensor(vec![0.0; 4], &[1, 4]);
    let m = Mod {
        tab: &t,
        scale: 0,
        shift: 0,
    };
    let gated = LtxResidual::Gated {
        u: &t,
        gates: &t,
        row: 0,
    };
    let plain = LtxResidual::Plain(&t);
    for (res, r) in [
        (None, legacy::launch::LtxRes::None),
        (Some(&plain), legacy::launch::LtxRes::Plain),
        (Some(&gated), legacy::launch::LtxRes::Gated),
    ] {
        for (norm, then_mod) in [(Norm::AdaLn(m), false), (Norm::ThenMod(&t, m), true)] {
            let (_, res_mode, mode) = legacy::launch::ltx(r, then_mod);
            assert_eq!(
                ltx_policy(res, &norm).kernel(),
                Ok(Kernel::LtxResNormMod { res_mode, mode })
            );
        }
    }
}

/// Every kernel a policy selects is an entry point of the NVRTC source,
/// whose bytes `rt::kernel_src` pins: B4b changes no kernel.
#[test]
fn every_selected_kernel_is_in_the_pinned_source() {
    let src = crate::rt::kernel_src::KERNEL_SRC;
    let has = |name: &str| src.contains(&format!("extern \"C\" __global__ void {name}("));
    for (label, p) in policy::RES_NORM_POLICIES {
        let k = p.kernel().unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(has(k.name()), "{label}: {}", k.name());
        if let Ok((a, b)) = p.split() {
            assert!(has(a.name()) && has(b.name()), "{label}: split");
        }
    }
    for ((label, p), old) in policy::QK_POLICIES.iter().zip(legacy::launch::QK) {
        assert_eq!(p.kernel, old, "{label}");
        assert!(has(p.kernel), "{label}");
    }
}

/// Every policy a family runs is distinct (no two family ops collapsed onto
/// one set of rounding points by the unification).
#[test]
fn the_policies_are_distinct() {
    let ps = policy::RES_NORM_POLICIES;
    for (i, (a, pa)) in ps.iter().enumerate() {
        for (b, pb) in &ps[i + 1..] {
            assert_ne!(pa, pb, "{a} and {b}");
        }
    }
}

// ---- Switches ------------------------------------------------------------------

/// A settings source over fixed values (unset: absent).
struct Fixed(BTreeMap<&'static str, &'static str>);

impl Source for Fixed {
    fn var(&self, name: &str) -> Option<String> {
        self.0.get(name).map(|v| (*v).to_owned())
    }
    fn is_set(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
    fn env_var(&self, name: &str) -> Result<String, std::env::VarError> {
        self.var(name).ok_or(std::env::VarError::NotPresent)
    }
    fn env_var_os(&self, name: &str) -> Option<OsString> {
        self.var(name).map(OsString::from)
    }
}

/// Every spelling the flag parser treats differently.
const SPELLINGS: [Option<&str>; 9] = [
    None,
    Some("0"),
    Some("1"),
    Some("false"),
    Some(" OFF "),
    Some("off"),
    Some("on"),
    Some(""),
    Some("garbage"),
];

#[test]
fn the_plans_read_what_the_old_plans_read() {
    for fuse in SPELLINGS {
        for glue in SPELLINGS {
            let mut m = BTreeMap::new();
            if let Some(v) = fuse {
                m.insert("FASTVIDEO_H3_FUSE", v);
                m.insert("FASTVIDEO_LTX_FUSE", v);
            }
            if let Some(v) = glue {
                m.insert("FASTVIDEO_H3_GLUE", v);
            }
            let src = Fixed(m);
            let h3 = BlockFusion::h3(&src);
            assert_eq!(
                (h3.on(Switch::H3), h3.on(Switch::H3Glue)),
                legacy::getters::h3_plan(&src),
                "{fuse:?} {glue:?}"
            );
            let ltx = BlockFusion::ltx(&src);
            assert_eq!(
                ltx.on(Switch::Ltx),
                legacy::getters::ltx_plan(&src),
                "{fuse:?}"
            );
            // An LTX plan leaves H3's and Wan's switches to the process.
            assert_eq!(ltx.get(Switch::H3), None);
            assert_eq!(h3.get(Switch::Wan), None);
        }
    }
}

/// The new switches against the old getters in this process: unforced,
/// every forced combination and nesting, and under the value a plan reads
/// from the live settings (what the fuse stage installs).
#[test]
fn the_switches_resolve_as_the_old_getters() {
    use legacy::getters as old;
    let check = |what: &str| {
        assert_eq!(
            switch::enabled(Switch::H3),
            old::fuse_enabled(),
            "{what}: h3"
        );
        assert_eq!(
            switch::enabled(Switch::H3Glue),
            old::glue_enabled(),
            "{what}: glue"
        );
        assert_eq!(
            switch::enabled(Switch::Ltx),
            old::ltx_enabled(),
            "{what}: ltx"
        );
        assert_eq!(
            switch::enabled(Switch::Wan),
            old::wan_enabled(),
            "{what}: wan"
        );
    };
    let plans = [
        None,
        Some(BlockFusion::h3(&Live)),
        Some(BlockFusion::ltx(&Live)),
        Some(BlockFusion::h3(&Live).with(Switch::Ltx, BlockFusion::ltx(&Live).on(Switch::Ltx))),
    ];
    for plan in plans {
        switch::installed(plan, || {
            check("unforced");
            for fuse in [false, true] {
                for glue in [false, true] {
                    switch::with_forced(Switch::H3, fuse, || {
                        old::with_fuse(fuse, || {
                            check("h3 forced");
                            switch::with_forced(Switch::H3Glue, glue, || {
                                old::with_glue(glue, || check("h3 + glue forced"))
                            });
                            check("glue restored");
                        })
                    });
                    // The glue forced alone, under the fusions' own value.
                    switch::with_forced(Switch::H3Glue, glue, || {
                        old::with_glue(glue, || check("glue forced"))
                    });
                }
                switch::with_forced(Switch::Ltx, fuse, || {
                    old::ltx_with_fuse(fuse, || {
                        check("ltx forced");
                        switch::with_forced(Switch::Ltx, !fuse, || {
                            old::ltx_with_fuse(!fuse, || check("ltx nested"))
                        });
                        check("ltx restored");
                    })
                });
                switch::with_forced(Switch::Wan, fuse, || {
                    old::wan_with_fuse(fuse, || check("wan forced"))
                });
            }
            check("all restored");
        });
    }
}

/// The families' own getters are the switches.
#[test]
fn the_family_getters_are_the_switches() {
    for on in [false, true] {
        fused16::with_fuse(on, || assert_eq!(fused16::fuse_enabled(), on));
        fused16::with_fuse(true, || {
            fused16::with_glue(on, || assert_eq!(fused16::glue_enabled(), on))
        });
        fused16::with_fuse(false, || {
            fused16::with_glue(true, || assert!(!fused16::glue_enabled()))
        });
        crate::ltx2::fuse::with_fuse(on, || assert_eq!(crate::ltx2::fuse::enabled(), on));
        wan::with_fuse(on, || assert_eq!(wan::enabled(), on));
        // A plan's installed value reaches the family getters.
        let plan = BlockFusion::default()
            .with(Switch::H3, on)
            .with(Switch::H3Glue, true)
            .with(Switch::Ltx, on);
        switch::installed(Some(plan), || {
            assert_eq!(fused16::fuse_enabled(), on);
            assert_eq!(fused16::glue_enabled(), on);
            assert_eq!(crate::ltx2::fuse::enabled(), on);
        });
    }
}
