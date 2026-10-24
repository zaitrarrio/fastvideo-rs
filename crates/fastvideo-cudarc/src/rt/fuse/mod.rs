//! The DiT block fusions of every family, in one place (plan B4b,
//! `docs/adr/0022-block-fusion.md`).
//!
//! Wan (`wan::fuse`), H3 (`h3::fused16`) and LTX-2 (`ltx2::fuse`) each fuse
//! the elementwise chain around their block's GEMMs and attention: a residual
//! in front of a modulated norm, the block's last gated residual, and the q/k
//! norm with its rotary embedding. They are one computation per op with
//! different rounding points (where a value is rounded to bf16 and where it
//! stays f32), a different normalizer (LayerNorm or RMSNorm) and a different
//! modulation-table layout. This module holds what they share:
//!
//! * [`policy`]: the rounding points of every fused op as a typed value
//!   ([`RoundingPolicy`] for residual + norm, [`QkPolicy`] for q/k norm +
//!   RoPE), one named constant per family op, and the kernel each policy
//!   selects ([`Kernel`]). The kernels themselves stay as they were, one
//!   instantiation per family (the source text is pinned by
//!   `rt::kernel_src`'s hash): unifying their source would change their
//!   arithmetic order, so the policy chooses among them instead;
//! * [`host`]: the one host reference of those ops, parameterized by the
//!   policy, which the families' CPU twins run (Wan's and H3's; LTX-2 has
//!   none: its CPU path is the unfused op chain);
//! * [`switch`]: the fusion switches (`FASTVIDEO_H3_FUSE`, `FASTVIDEO_H3_GLUE`,
//!   `FASTVIDEO_LTX_FUSE`, `FASTVIDEO_WAN_FUSE`) as one implementation, and
//!   [`BlockFusion`], the value an H3 or LTX-2 load plan decides from its
//!   settings snapshot and the fuse stage installs on the pipeline's DiT
//!   (ADR-0017 "How B4 and a lazy cold start plug in").
//!
//! What is one family's alone stays in that family's module: the device
//! launchers (operand checks, table pointers, MXFP8 / NVFP4 outputs), LTX-2's
//! per-head output gates, H3's head merges, SwiGLU producers and q/k/v/gate
//! split.

pub mod host;
pub mod policy;
pub mod switch;

#[cfg(test)]
mod equality;
#[cfg(test)]
pub(crate) mod legacy;

pub use policy::{
    HostTwin, Kernel, Modulation, Norm, NormInput, Normalizer, PolicyError, QkPolicy, QkScope,
    Residual, RopeCombine, RopePairing, RoundingPolicy,
};
pub use switch::{enabled, installed, with_forced, BlockFusion, Switch};
