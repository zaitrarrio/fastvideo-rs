//! The one host reference of the fused block ops: a row of a residual and /
//! or modulated norm ([`res_norm_row`]) and a q/k row's norm and rotation
//! ([`qk_norm`], [`qk_rope_head`]), each taking its rounding points from a
//! policy ([`super::policy`]). The families' CPU twins (`wan::fuse`,
//! `h3::fused16`, `quant::norm_mod_row` and its siblings) are these
//! functions with their family's policy; how a family indexes its
//! modulation table stays in the family.
//!
//! Bit-identical to the twins it replaced (`rt/fuse/equality.rs`, against
//! the pre-B4b code in `rt/fuse/legacy.rs`): every expression below is the
//! one the old twin wrote, in the same order.

use thiserror::Error;

use super::policy::{
    HostTwin, Modulation, NormInput, Normalizer, QkPolicy, Residual, RopeCombine, RopePairing,
    RoundingPolicy,
};
use crate::rt::quant::{bf16_round, row_rsqrt};

/// Why the host reference cannot run a policy.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostError {
    /// The policy has no host twin (LTX-2), or a combination no twin runs.
    #[error("fused op has no host twin: {0}")]
    Unsupported(String),
    /// An operand the policy reads was not given, or has the wrong length.
    #[error("fused op host twin: {0}")]
    Operand(&'static str),
}

/// The operands of one row (`dim` values each; `scale` is `1 + scale` for
/// [`Modulation::WeightFmaScaleShift`], the raw scale otherwise).
#[derive(Debug, Clone, Copy, Default)]
pub struct RowOperands<'a> {
    /// The residual stream (or the norm's input without a residual).
    pub x: &'a [f32],
    /// The branch added to it.
    pub u: Option<&'a [f32]>,
    /// The branch's gate row.
    pub gate: Option<&'a [f32]>,
    /// The norm's weight.
    pub w: Option<&'a [f32]>,
    /// The norm's bias.
    pub b: Option<&'a [f32]>,
    pub scale: Option<&'a [f32]>,
    pub shift: Option<&'a [f32]>,
    /// The norm's epsilon.
    pub eps: f32,
}

/// One row's outputs: the stored hidden (with a residual) and the normed
/// value (with a norm), both bf16 values held in f32.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RowOut {
    pub hidden: Option<Vec<f32>>,
    pub normed: Option<Vec<f32>>,
}

fn operand<'a>(
    v: Option<&'a [f32]>,
    dim: usize,
    what: &'static str,
) -> Result<&'a [f32], HostError> {
    match v {
        Some(s) if s.len() >= dim => Ok(&s[..dim]),
        _ => Err(HostError::Operand(what)),
    }
}

fn unsupported(p: &impl std::fmt::Debug) -> HostError {
    HostError::Unsupported(format!("{p:?}"))
}

/// The operands as the twin reads them: as stored, or rounded to bf16 first
/// ([`HostTwin::KernelLanes`]).
fn read(twin: HostTwin, v: &[f32]) -> Vec<f32> {
    match twin {
        HostTwin::KernelLanes => v.iter().map(|&x| bf16_round(x)).collect(),
        _ => v.to_vec(),
    }
}

/// LayerNorm statistics, summed in index order: `(mean, 1 / sqrt(var + eps))`.
pub fn ln_stats(v: &[f32], eps: f32) -> (f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, 1.0 / (var + eps).sqrt())
}

/// The RMS factor summed in index order: `1 / sqrt(mean(x²) + eps)`.
pub fn rms_factor(x: &[f32], eps: f32) -> f32 {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    1.0 / (ss / x.len() as f32 + eps).sqrt()
}

/// The residual's f32 value before its one rounding to bf16 (`x` without a
/// residual).
pub fn combine(residual: Residual, x: f32, u: f32, g: f32) -> f32 {
    match residual {
        Residual::None => x,
        Residual::Add => x + u,
        Residual::MulAdd => x + u * g,
        Residual::Fma => g.mul_add(u, x),
        Residual::RoundedProduct => x + bf16_round(g * u),
    }
}

/// LayerNorm in index order: `(v - mean) · inv` per channel ([`ln_stats`]).
pub fn normalize_ln_sequential(v: &[f32], eps: f32) -> Vec<f32> {
    let (mean, inv) = ln_stats(v, eps);
    v.iter().map(|&x| (x - mean) * inv).collect()
}

/// RMSNorm with H3's kernel-order statistic: `v · row_rsqrt(v)` per channel.
pub fn normalize_rms_lanes(v: &[f32], eps: f32) -> Vec<f32> {
    let r = row_rsqrt(v, eps);
    v.iter().map(|&x| x * r).collect()
}

/// [`Modulation::Affine`]: `bf16(n·w + b)`.
pub fn modulate_affine(n: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    (0..n.len())
        .map(|j| bf16_round(n[j] * w[j] + b[j]))
        .collect()
}

/// [`Modulation::RoundedNormScaleShift`]: `bf16(bf16(n)·(1 + scale) + shift)`.
pub fn modulate_rounded_norm(n: &[f32], scale: &[f32], shift: &[f32]) -> Vec<f32> {
    (0..n.len())
        .map(|j| bf16_round(bf16_round(n[j]) * (1.0 + scale[j]) + shift[j]))
        .collect()
}

/// [`Modulation::WeightFmaScaleShift`]: `bf16(fma(n·w, scale1p, shift))`.
pub fn modulate_weight_fma(n: &[f32], w: &[f32], scale1p: &[f32], shift: &[f32]) -> Vec<f32> {
    (0..n.len())
        .map(|j| bf16_round((n[j] * w[j]).mul_add(scale1p[j], shift[j])))
        .collect()
}

/// One row of `policy` (`dim = ops.x.len()`).
pub fn res_norm_row(policy: &RoundingPolicy, ops: &RowOperands<'_>) -> Result<RowOut, HostError> {
    if policy.host == HostTwin::None {
        return Err(unsupported(policy));
    }
    let dim = ops.x.len();
    let x = read(policy.host, ops.x);
    // The residual's f32 value (`sum`) before its one rounding to bf16.
    let sum: Option<Vec<f32>> = match policy.residual {
        Residual::None => None,
        Residual::Add => {
            let u = read(policy.host, operand(ops.u, dim, "branch")?);
            Some(
                (0..dim)
                    .map(|j| combine(Residual::Add, x[j], u[j], 0.0))
                    .collect(),
            )
        }
        r => {
            let u = read(policy.host, operand(ops.u, dim, "branch")?);
            let g = operand(ops.gate, dim, "gate")?;
            Some((0..dim).map(|j| combine(r, x[j], u[j], g[j])).collect())
        }
    };
    let hidden: Option<Vec<f32>> = sum
        .as_ref()
        .map(|s| s.iter().map(|&v| bf16_round(v)).collect());
    let Some(norm) = policy.norm else {
        return Ok(RowOut {
            hidden,
            normed: None,
        });
    };
    let v: &[f32] = match (&sum, &hidden, norm.input) {
        (None, _, _) => &x,
        (Some(s), _, NormInput::F32Sum) => s,
        (_, Some(h), NormInput::StoredBf16) => h,
        _ => return Err(unsupported(policy)),
    };
    let nv = match (norm.normalizer, policy.host) {
        (Normalizer::LayerNorm, HostTwin::Sequential) => normalize_ln_sequential(v, ops.eps),
        (Normalizer::Rms, HostTwin::KernelLanes) => normalize_rms_lanes(v, ops.eps),
        _ => return Err(unsupported(policy)),
    };
    let normed = match norm.modulation {
        Modulation::Affine => modulate_affine(
            &nv,
            operand(ops.w, dim, "weight")?,
            operand(ops.b, dim, "bias")?,
        ),
        Modulation::RoundedNormScaleShift => modulate_rounded_norm(
            &nv,
            operand(ops.scale, dim, "scale")?,
            operand(ops.shift, dim, "shift")?,
        ),
        Modulation::WeightFmaScaleShift => modulate_weight_fma(
            &nv,
            operand(ops.w, dim, "weight")?,
            operand(ops.scale, dim, "scale")?,
            operand(ops.shift, dim, "shift")?,
        ),
        Modulation::RoundedScaleShift | Modulation::WeightRoundedScaleShift => {
            return Err(unsupported(policy))
        }
    };
    Ok(RowOut {
        hidden,
        normed: Some(normed),
    })
}

/// The q/k norm of `x` (one head for [`super::QkScope::PerHead`], every
/// head for `AcrossHeads`): `x · rms`, rounded when `round_normalized`, times
/// `w`, rounded when `round_weighted`. Not yet rounded otherwise.
pub fn qk_norm(policy: &QkPolicy, x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, HostError> {
    if policy.host == HostTwin::None || policy.combine != RopeCombine::MulAdd {
        return Err(unsupported(policy));
    }
    if w.len() < x.len() {
        return Err(HostError::Operand("qk norm weight"));
    }
    let x = read(policy.host, x);
    let inv = rms_factor(&x, eps);
    Ok(x.iter()
        .zip(w)
        .map(|(&v, &wv)| {
            let n = v * inv;
            let n = if policy.round_normalized {
                bf16_round(n)
            } else {
                n
            };
            let n = n * wv;
            if policy.round_weighted {
                bf16_round(n)
            } else {
                n
            }
        })
        .collect())
}

/// One head's output (`d` channels of [`qk_norm`]) rotated by its sequence
/// row's tables, or passed through without a rotation; bf16.
pub fn qk_rope_head(
    policy: &QkPolicy,
    n: &[f32],
    rope: Option<(&[f32], &[f32])>,
) -> Result<Vec<f32>, HostError> {
    let pass = |v: f32| {
        if policy.round_weighted {
            v
        } else {
            bf16_round(v)
        }
    };
    let Some((c, sn)) = rope else {
        return Ok(n.iter().map(|&v| pass(v)).collect());
    };
    let d = n.len();
    match policy.pairing {
        RopePairing::Interleaved => {
            if !d.is_multiple_of(2) || c.len() < d || sn.len() < d {
                return Err(HostError::Operand("interleaved rope tables"));
            }
            Ok((0..d)
                .map(|p| {
                    let even = p - (p & 1);
                    let (x1, x2) = (n[even], n[even + 1]);
                    let (cv, sv) = (c[even], sn[even + 1]);
                    bf16_round(if p & 1 == 1 {
                        x2 * cv + x1 * sv
                    } else {
                        x1 * cv - x2 * sv
                    })
                })
                .collect())
        }
        RopePairing::RotateHalf => {
            let r = c.len();
            if r > d || sn.len() < r {
                return Err(HostError::Operand("rotate_half rope tables"));
            }
            let half = r / 2;
            Ok((0..d)
                .map(|p| {
                    if p >= r {
                        return bf16_round(n[p]);
                    }
                    let partner = if p < half { p + half } else { p - half };
                    let rot = if p < half { -n[partner] } else { n[partner] };
                    bf16_round(n[p] * c[p] + rot * sn[p])
                })
                .collect())
        }
        RopePairing::RotateHalfHeadMajor => Err(unsupported(policy)),
    }
}
