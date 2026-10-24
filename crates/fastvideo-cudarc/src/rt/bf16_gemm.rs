//! BF16 policy for DiT GEMMs.
//!
//! `FASTVIDEO_BF16` unset/`1`/`true`: cuBLAS runs F32 buffers with
//! `CUBLAS_COMPUTE_32F_FAST_16BF` (bfloat16 math, like upstream's bf16
//! autocast) on Tensor Core GPUs. `FASTVIDEO_BF16=0` keeps FP32/TF32 math.
//! See [`super::device::GemmMath`].

use super::envflag::CachedBool;

static BF16_CACHE: CachedBool = CachedBool::new();

/// BF16 GEMM policy (default on). Read once when the device context is built.
pub fn bf16_enabled() -> bool {
    BF16_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_BF16", true))
}
