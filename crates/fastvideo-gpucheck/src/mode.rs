//! Precision modes. The cudarc crate caches its `FASTVIDEO_*` flags on first
//! read, so the mode must be applied at process start, before any cudarc call.
//!
//! The load-time settings snapshot (`docs/adr/0010-settings-layer.md`) does
//! not change that: a load reads its settings as they were when it started,
//! but `FASTVIDEO_BF16` (`rt::bf16_gemm`) and `FASTVIDEO_TF32` (`rt::hopper`)
//! are cached once per process on their first read, whenever that is, and
//! `FASTVIDEO_TEACACHE` is read by each Wan generate. Setting the mode before
//! the first cudarc call stays the rule
//! (`rt::envflag::tests::a_cache_filled_inside_a_load_keeps_the_snapshot_value`).

use clap::ValueEnum;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// FP32 math everywhere (no TF32 in cuBLAS or cuDNN): the GPU must match
    /// the CPU path tightly. Any gap here is a kernel bug, not precision.
    Exact,
    /// Production defaults (bf16 cuBLAS compute, TF32 convs): looser limits
    /// that bound the quality cost of the fast math.
    Fast,
}

impl Mode {
    pub fn apply_env(self) {
        if self == Mode::Exact {
            for (k, v) in [
                ("FASTVIDEO_BF16", "0"),
                ("FASTVIDEO_TF32", "0"),
                ("FASTVIDEO_TEACACHE", "0"),
            ] {
                if fastvideo_models::settings::env_var(k).is_ok_and(|cur| cur != v) {
                    eprintln!(
                        "mode=exact overrides {k}={} → {v}",
                        fastvideo_models::settings::env_var(k).unwrap_or_default()
                    );
                }
                std::env::set_var(k, v);
            }
        }
    }
}

/// GPU-vs-CPU-path limits (relative L2 unless noted) per mode.
///
/// `exact` compares cudarc's GPU kernels + cuBLAS (TF32 off, F32) with its
/// own host path: only reduction-order / fast-math round-off should remain.
/// `fast` bounds bf16/TF32 math. Both modes run the same order-2 UniPC and
/// DMD samplers on device and host. Calibrate from the
/// values recorded in reports; never loosen a limit to make a failing run pass
/// without understanding the gap.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Limits {
    /// Single kernel / op.
    pub op: f64,
    /// One full forward of a multi-layer module (UMT5, DiT, VAE decode).
    pub forward: f64,
    /// Final latents after a short sampler run.
    pub denoise: f64,
}

pub fn limits(mode: Mode) -> Limits {
    match mode {
        Mode::Exact => Limits {
            op: 1e-5,
            forward: 1e-3,
            denoise: 2e-3,
        },
        Mode::Fast => Limits {
            op: 5e-3,
            forward: 2e-2,
            // bf16 error compounds over sampler steps; samplers themselves are
            // identical on both paths.
            denoise: 0.1,
        },
    }
}
