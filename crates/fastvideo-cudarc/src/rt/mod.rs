//! The device runtime shared by every model family: tensors, the device
//! context, kernels, ops, attention backends, quantization, weight loading,
//! offload, logging, tracing and the frame writer.
//!
//! Extracted from [`crate::wan`] (plan B1, `docs/plan/2026-10-23-execution-plan.md`),
//! byte-identical in behaviour: every module here was `wan::<name>` and
//! `wan/mod.rs` re-exports each one under its old path until the last user
//! moves. Nothing in `rt` is specific to Wan; the Wan family lives in
//! [`crate::wan`], and the pieces of `rt` that still name a family type
//! (`quant::H3QuantPlan`, the LTX seam in `nvfp4_linear`, `writer`'s use of
//! `wan::pipeline::PipelineError`) are listed as follow-ups in that plan.

#[cfg(feature = "cuda")]
pub(crate) mod act16;
pub mod affine;
pub mod attn;
#[cfg(feature = "cuda")]
pub mod attn_dc;
pub mod attn_fp4;
pub mod attn_fp8;
pub mod attn_sage;
pub mod bf16_gemm;
#[cfg(feature = "cuda")]
pub mod conv;
pub mod cudnn_sdpa;
pub mod device;
pub mod dump;
pub mod envflag;
pub mod evalstats;
#[cfg(feature = "cuda")]
pub mod fp8;
pub mod fused;
pub mod gpu_trace;
pub mod graph;
pub mod hopper;
pub mod hostmem;
pub mod inject;
/// The kernel source string, concatenated by `build.rs` from the region files.
pub(crate) mod kernel_src;
#[cfg(feature = "cuda")]
pub mod kernels;
pub mod ledger;
pub mod log;
pub mod nn;
pub mod nvfp4;
#[cfg(feature = "nvfp4-gemm")]
pub mod nvfp4_gemm;
#[cfg(feature = "cuda")]
pub mod nvfp4_linear;
#[cfg(feature = "cuda")]
pub mod nvrtc_sass;
pub mod offload;
pub mod ops;
pub mod quant;
pub mod resident;
pub mod sdpa;
pub mod sdpa_rule;
pub mod sla;
pub mod sol_ops;
pub mod sp;
#[cfg(feature = "cuda")]
pub mod stage_upload;
pub mod stats;
pub mod taehv;
pub mod taehv_ref;
pub mod tensor;
pub mod vsa;
pub mod weights;
pub mod writer;
