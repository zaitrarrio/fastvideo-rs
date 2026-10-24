//! The rounding points of every fused block op, as typed values, and the
//! kernel each one selects (`docs/adr/0022-block-fusion.md`, "Inventory").
//!
//! A [`RoundingPolicy`] says, for a residual in front of a modulated norm (or
//! a residual alone), where every intermediate is rounded to bf16 and where
//! it stays f32: how the residual combines its operands ([`Residual`]), what
//! the norm reads ([`NormInput`]), which normalizer ([`Normalizer`]) and how
//! the modulation rounds ([`Modulation`]). [`QkPolicy`] does the same for the
//! q/k RMSNorm with its rotary embedding. Every family op is one named
//! constant below; [`RoundingPolicy::kernel`] maps a policy onto the kernel
//! instantiation that implements exactly those rounding points, so the
//! launchers take their mode arguments from the policy, never from literals.

use thiserror::Error;

/// A policy no kernel implements (a combination of rounding points that no
/// family runs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("no fused kernel implements {0:?}")]
pub struct PolicyError(pub RoundingPolicy);

/// How the residual in front of a norm combines the stream `x`, the branch
/// `u` and its gate `g` (a modulation-table row). The stored hidden is the
/// result rounded to bf16 once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residual {
    /// No residual: the norm reads `x` as stored.
    None,
    /// `x + u` in f32.
    Add,
    /// `x + u·g`: the product, then the sum, each an f32 operation (no FMA;
    /// the Wan kernels spell both with `_rn` intrinsics).
    MulAdd,
    /// `fma(g, u, x)`: one f32 rounding.
    Fma,
    /// `x + bf16(g·u)`: the product rounded to bf16 before the sum (eager
    /// torch's `residual + gate * branch` on bf16 tensors).
    RoundedProduct,
}

/// What the norm after a residual reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormInput {
    /// The unrounded f32 sum (one pass, the hidden is stored rounded beside).
    F32Sum,
    /// The stored bf16 hidden.
    StoredBf16,
}

/// The row statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalizer {
    /// `(v - mean) · rsqrt(var + eps)`, two passes.
    LayerNorm,
    /// `v · rsqrt(mean(v²) + eps)`.
    Rms,
}

/// The modulation after the normalizer (`n`: the normalized value; `w`, `b`
/// the norm's affine; `scale`, `shift` modulation-table rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modulation {
    /// `bf16(n·w + b)`: a LayerNorm affine, one rounding (Wan `norm2`).
    Affine,
    /// `bf16(bf16(n)·(1 + scale) + shift)`: the norm rounded to bf16 before
    /// the f32 modulation (Wan's `FP32LayerNorm` then AdaLN).
    RoundedNormScaleShift,
    /// `bf16(fma(n·w, scale1p, shift))` with `scale1p = 1 + scale` formed in
    /// f32 beforehand: one f32 chain, one rounding (Sol-H3
    /// `fused_rmsnorm_modulate`).
    WeightFmaScaleShift,
    /// `bf16(bf16(n·bf16(1 + scale)) + shift)` (LTX-2 `rms_adaln`).
    RoundedScaleShift,
    /// `bf16(bf16(bf16(n·w)·bf16(1 + scale)) + shift)` (LTX-2's text
    /// cross-attention: `rms_norm(x, w)`, then `scale_shift`).
    WeightRoundedScaleShift,
}

/// A modulated norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Norm {
    pub input: NormInput,
    pub normalizer: Normalizer,
    pub modulation: Modulation,
}

/// The CPU twin of a policy: how [`super::host`] reads the operands and
/// reduces a row. It is not a rounding point of the device kernel; it
/// records what each family's host twin has always done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTwin {
    /// Operands as stored; statistics summed in index order, `1 / sqrt`
    /// (Wan: within a bf16 ulp of the kernel, which reduces by its block
    /// tree and `rsqrtf`).
    Sequential,
    /// Operands rounded to bf16 first; RMS statistics in the kernel's lane
    /// order with a correctly rounded rsqrt (`quant::row_rsqrt`), so the
    /// norms are bit-identical to H3's kernels; q/k summed in index order.
    KernelLanes,
    /// No host twin: the CPU runs the unfused op chain (LTX-2, whose kernels
    /// use the hardware `rsqrtf`, which no host code reproduces bit for bit).
    None,
}

/// The rounding points of a residual and / or modulated norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundingPolicy {
    pub residual: Residual,
    /// `None`: the residual alone (a block's last residual).
    pub norm: Option<Norm>,
    pub host: HostTwin,
}

/// One kernel instantiation and its mode arguments (`rt/kernels/*.cu`,
/// `wan/kernels/block.cu`, `h3/kernels/block_ops.cu`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// `wan_res_ln`: residual (`res_mode`) and LayerNorm (`ln_mode`) fused.
    WanResLn { res_mode: i32, ln_mode: i32 },
    /// `wan_res_gate`: the residual alone, the f32 sum stored when `keep_f32`.
    WanResGate { res_mode: i32, keep_f32: bool },
    /// `wan_ln`: the norm of a stored row.
    WanLn { ln_mode: i32 },
    /// `h3_norm_mod`.
    H3NormMod,
    /// `h3_res_gate_norm_mod`.
    H3ResGateNormMod,
    /// `h3_gate_residual` (`round_product` = 1: the eager product rounding).
    H3GateResidual { round_product: i32 },
    /// `fvf_h3_gate_res_norm_mod`.
    H3GateResNormMod,
    /// `fvf_ltx_res_norm_mod`.
    LtxResNormMod { res_mode: i32, mode: i32 },
}

impl Kernel {
    /// The `extern "C" __global__` entry point.
    pub const fn name(&self) -> &'static str {
        match self {
            Kernel::WanResLn { .. } => "wan_res_ln",
            Kernel::WanResGate { .. } => "wan_res_gate",
            Kernel::WanLn { .. } => "wan_ln",
            Kernel::H3NormMod => "h3_norm_mod",
            Kernel::H3ResGateNormMod => "h3_res_gate_norm_mod",
            Kernel::H3GateResidual { .. } => "h3_gate_residual",
            Kernel::H3GateResNormMod => "fvf_h3_gate_res_norm_mod",
            Kernel::LtxResNormMod { .. } => "fvf_ltx_res_norm_mod",
        }
    }
}

const fn norm(input: NormInput, normalizer: Normalizer, modulation: Modulation) -> Option<Norm> {
    Some(Norm {
        input,
        normalizer,
        modulation,
    })
}

// ---- Wan (`wan::fuse`, FastVideo's `WanTransformerBlock.forward`) ----------

/// Self-attention residual + `norm2`: `hidden = bf16(h + a·gate)`, the
/// LayerNorm (f32 affine) of the unrounded sum.
pub const WAN_SELF_RESIDUAL_NORM: RoundingPolicy = RoundingPolicy {
    residual: Residual::MulAdd,
    norm: norm(NormInput::F32Sum, Normalizer::LayerNorm, Modulation::Affine),
    host: HostTwin::Sequential,
};

/// Cross-attention residual + the FFN's AdaLN: `hidden = bf16(h + c)`, then
/// `bf16(bf16(LN(hidden))·(1 + scale) + shift)`.
pub const WAN_CROSS_RESIDUAL_NORM_MOD: RoundingPolicy = RoundingPolicy {
    residual: Residual::Add,
    norm: norm(
        NormInput::StoredBf16,
        Normalizer::LayerNorm,
        Modulation::RoundedNormScaleShift,
    ),
    host: HostTwin::Sequential,
};

/// The block's last residual: `bf16(h + ff·gate)` in f32.
pub const WAN_GATE_RESIDUAL: RoundingPolicy = RoundingPolicy {
    residual: Residual::MulAdd,
    norm: None,
    host: HostTwin::Sequential,
};

// ---- H3 (`h3::fused16`, Sol-H3 `fusions.py`) -------------------------------

/// `bf16(rms(x)·w·(1 + scale) + shift)`, one f32 chain.
pub const H3_NORM_MOD: RoundingPolicy = RoundingPolicy {
    residual: Residual::None,
    norm: norm(
        NormInput::StoredBf16,
        Normalizer::Rms,
        Modulation::WeightFmaScaleShift,
    ),
    host: HostTwin::KernelLanes,
};

/// `hidden = bf16(fma(gate, branch, res))`, the norm of the unrounded sum.
pub const H3_RES_GATE_NORM_MOD: RoundingPolicy = RoundingPolicy {
    residual: Residual::Fma,
    norm: norm(
        NormInput::F32Sum,
        Normalizer::Rms,
        Modulation::WeightFmaScaleShift,
    ),
    host: HostTwin::KernelLanes,
};

/// The block's last residual as eager torch runs it: `bf16(res +
/// bf16(gate·branch))`.
pub const H3_GATE_RESIDUAL: RoundingPolicy = RoundingPolicy {
    residual: Residual::RoundedProduct,
    norm: None,
    host: HostTwin::KernelLanes,
};

/// [`H3_GATE_RESIDUAL`] and the next block's [`H3_NORM_MOD`] of the stored
/// hidden in one kernel (`FASTVIDEO_H3_FUSE`): the same rounding points as
/// the two ops in a row. Device only (the host runs the two twins).
pub const H3_GATE_RES_NORM_MOD: RoundingPolicy = RoundingPolicy {
    residual: Residual::RoundedProduct,
    norm: norm(
        NormInput::StoredBf16,
        Normalizer::Rms,
        Modulation::WeightFmaScaleShift,
    ),
    host: HostTwin::KernelLanes,
};

// ---- LTX-2 (`ltx2::fuse`, the bf16 op chain it replaces) -------------------

/// LTX-2's modulated norm behind `residual` ([`Residual::None`],
/// [`Residual::RoundedProduct`]: `residual_gate_add_e`, or [`Residual::Add`])
/// with `rms_adaln` (`then_mod = false`) or `rms_norm(x, w)` then
/// `scale_shift` (`then_mod = true`). Every intermediate is stored bf16, as
/// the op chain stores it.
pub const fn ltx_res_norm_mod(residual: Residual, then_mod: bool) -> RoundingPolicy {
    RoundingPolicy {
        residual,
        norm: norm(
            NormInput::StoredBf16,
            Normalizer::Rms,
            if then_mod {
                Modulation::WeightRoundedScaleShift
            } else {
                Modulation::RoundedScaleShift
            },
        ),
        host: HostTwin::None,
    }
}

impl RoundingPolicy {
    /// The kernel that runs exactly this policy (fused), or [`PolicyError`]
    /// when no family kernel does.
    pub fn kernel(&self) -> Result<Kernel, PolicyError> {
        use Modulation as M;
        use NormInput as I;
        use Normalizer as N;
        use Residual as R;
        let k = match (self.residual, self.norm) {
            // Wan: LayerNorm kernels; res_mode 0 is the gated residual, 1 the
            // plain one; ln_mode 0 the affine, 1 the rounded-norm AdaLN.
            (R::MulAdd, None) => Kernel::WanResGate {
                res_mode: 0,
                keep_f32: false,
            },
            (
                r @ (R::MulAdd | R::Add),
                Some(Norm {
                    input,
                    normalizer: N::LayerNorm,
                    modulation: m @ (M::Affine | M::RoundedNormScaleShift),
                }),
            ) if matches!((r, input), (R::MulAdd, I::F32Sum) | (R::Add, I::StoredBf16)) => {
                Kernel::WanResLn {
                    res_mode: wan_res_mode(r),
                    ln_mode: i32::from(m == M::RoundedNormScaleShift),
                }
            }
            // H3: RMSNorm with the weight-FMA modulation.
            (R::RoundedProduct, None) => Kernel::H3GateResidual { round_product: 1 },
            (
                r,
                Some(Norm {
                    input,
                    normalizer: N::Rms,
                    modulation: M::WeightFmaScaleShift,
                }),
            ) => match (r, input) {
                (R::None, _) => Kernel::H3NormMod,
                (R::Fma, I::F32Sum) => Kernel::H3ResGateNormMod,
                (R::RoundedProduct, I::StoredBf16) => Kernel::H3GateResNormMod,
                _ => return Err(PolicyError(*self)),
            },
            // LTX-2: every value stored bf16; res_mode 0 none, 1 the rounded
            // gated product, 2 the plain add; mode 0 rms_adaln, 1 norm then
            // scale_shift.
            (
                r @ (R::None | R::RoundedProduct | R::Add),
                Some(Norm {
                    input: I::StoredBf16,
                    normalizer: N::Rms,
                    modulation: m @ (M::RoundedScaleShift | M::WeightRoundedScaleShift),
                }),
            ) => Kernel::LtxResNormMod {
                res_mode: match r {
                    R::None => 0,
                    R::RoundedProduct => 1,
                    _ => 2,
                },
                mode: i32::from(m == M::WeightRoundedScaleShift),
            },
            _ => return Err(PolicyError(*self)),
        };
        Ok(k)
    }

    /// The same policy as two kernels, the residual storing what the norm
    /// reads (`FASTVIDEO_WAN_FUSE=0`: the f32 sum for [`NormInput::F32Sum`],
    /// the bf16 hidden otherwise). Only Wan has this split; the other
    /// families' unfused paths are their op chains.
    pub fn split(&self) -> Result<(Kernel, Kernel), PolicyError> {
        match self.kernel()? {
            Kernel::WanResLn { res_mode, ln_mode } => {
                let keep_f32 = matches!(self.norm, Some(n) if n.input == NormInput::F32Sum);
                Ok((
                    Kernel::WanResGate { res_mode, keep_f32 },
                    Kernel::WanLn { ln_mode },
                ))
            }
            _ => Err(PolicyError(*self)),
        }
    }
}

fn wan_res_mode(r: Residual) -> i32 {
    if r == Residual::Add {
        1
    } else {
        0
    }
}

// ---- q/k RMSNorm + RoPE ----------------------------------------------------

/// Over which channels the q/k RMS statistic runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkScope {
    /// All heads of the projection (`[heads·d]` weight): Wan, LTX-2.
    AcrossHeads,
    /// Each head's `d` channels (`[d]` weight): H3.
    PerHead,
}

/// Which channels a rotation pairs and where its cos / sin live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePairing {
    /// Pairs `(2i, 2i + 1)` within a head; `[S, d]` tables, cos read at the
    /// pair's even slot and sin at its odd one (Wan, `_apply_rotary_emb`).
    Interleaved,
    /// rotate_half over a head's first `r` channels (`p` with `p ± r/2`),
    /// `[S, r]` tables shared by the heads; channels `>= r` pass through
    /// (H3 partial RoPE).
    RotateHalf,
    /// rotate_half within each head with head-major `[H·S, r]` tables, every
    /// row holding its values twice when `dup` (LTX-2 `DeviceRope`).
    RotateHalfHeadMajor,
}

/// How the rotation's two products combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeCombine {
    /// Two f32 products and an f32 sum / difference.
    MulAdd,
    /// `fma(n, cos, other·sin)` (LTX-2).
    Fma,
}

/// The rounding points of a q/k RMSNorm followed by RoPE (bf16 BHSD out).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkPolicy {
    pub scope: QkScope,
    /// `bf16(x · rsqrt(..))` before the weight.
    pub round_normalized: bool,
    /// `bf16(n · w)` before the rotation.
    pub round_weighted: bool,
    pub pairing: RopePairing,
    pub combine: RopeCombine,
    pub host: HostTwin,
    /// The kernel (`extern "C"` entry point).
    pub kernel: &'static str,
}

/// Wan: RMSNorm across heads, `bf16(bf16(x·inv)·w)`, the interleaved rotation
/// in f32, one rounding (FastVideo `RMSNorm.forward_native`).
pub const WAN_QK_NORM_ROPE: QkPolicy = QkPolicy {
    scope: QkScope::AcrossHeads,
    round_normalized: true,
    round_weighted: true,
    pairing: RopePairing::Interleaved,
    combine: RopeCombine::MulAdd,
    host: HostTwin::Sequential,
    kernel: "wan_qk_norm_rope16",
};

/// H3: per-head RMSNorm, the weight and the partial rotate_half in f32, one
/// rounding (Sol-H3 `_qknorm_partial_rope_kernel`).
pub const H3_QK_NORM_ROPE: QkPolicy = QkPolicy {
    scope: QkScope::PerHead,
    round_normalized: false,
    round_weighted: false,
    pairing: RopePairing::RotateHalf,
    combine: RopeCombine::MulAdd,
    host: HostTwin::KernelLanes,
    kernel: "h3_qk_norm_rope",
};

/// LTX-2: RMSNorm across heads, `bf16(x·rs·w)` stored, the split rotary with
/// an FMA, one rounding (`rms_norm_across_heads` → `split_heads_bhsd` →
/// `DeviceRope::apply`).
pub const LTX_QK_NORM_ROPE: QkPolicy = QkPolicy {
    scope: QkScope::AcrossHeads,
    round_normalized: false,
    round_weighted: true,
    pairing: RopePairing::RotateHalfHeadMajor,
    combine: RopeCombine::Fma,
    host: HostTwin::None,
    kernel: "fvf_ltx_qk_norm_rope",
};

/// Every residual / norm policy of the three families, with its name (the
/// ADR's inventory rows).
pub const RES_NORM_POLICIES: [(&str, RoundingPolicy); 13] = [
    ("wan self_residual_norm", WAN_SELF_RESIDUAL_NORM),
    ("wan cross_residual_norm_mod", WAN_CROSS_RESIDUAL_NORM_MOD),
    ("wan gate_residual", WAN_GATE_RESIDUAL),
    ("h3 norm_mod", H3_NORM_MOD),
    ("h3 res_gate_norm_mod", H3_RES_GATE_NORM_MOD),
    ("h3 gate_residual", H3_GATE_RESIDUAL),
    ("h3 gate_res_norm_mod", H3_GATE_RES_NORM_MOD),
    (
        "ltx norm_mod adaln",
        ltx_res_norm_mod(Residual::None, false),
    ),
    (
        "ltx norm_mod then_mod",
        ltx_res_norm_mod(Residual::None, true),
    ),
    (
        "ltx res_norm_mod gated adaln",
        ltx_res_norm_mod(Residual::RoundedProduct, false),
    ),
    (
        "ltx res_norm_mod gated then_mod",
        ltx_res_norm_mod(Residual::RoundedProduct, true),
    ),
    (
        "ltx res_norm_mod plain adaln",
        ltx_res_norm_mod(Residual::Add, false),
    ),
    (
        "ltx res_norm_mod plain then_mod",
        ltx_res_norm_mod(Residual::Add, true),
    ),
];

/// The q/k policies, with their names.
pub const QK_POLICIES: [(&str, QkPolicy); 3] = [
    ("wan qk_norm_rope", WAN_QK_NORM_ROPE),
    ("h3 qk_norm_rope", H3_QK_NORM_ROPE),
    ("ltx qk_norm_rope", LTX_QK_NORM_ROPE),
];
