//! Precision modes. The cudarc crate caches its `FASTVIDEO_*` flags on first
//! read, so the mode must be applied at process start, before any cudarc call.

use clap::ValueEnum;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// F32 everywhere, no TF32, host-equivalent (order-2) UniPC: must match the
    /// Candle oracle tightly. Any gap here is a port/kernel bug, not precision.
    Exact,
    /// Production defaults (BF16 GEMMs, TF32, device order-1 UniPC): looser
    /// limits that bound the quality cost of the fast paths.
    Fast,
}

impl Mode {
    pub fn apply_env(self) {
        if self == Mode::Exact {
            for (k, v) in [
                ("FASTVIDEO_BF16", "0"),
                ("FASTVIDEO_TF32", "0"),
                ("FASTVIDEO_DEVICE_SCHED", "0"),
                ("FASTVIDEO_TEACACHE", "0"),
            ] {
                if std::env::var(k).is_ok_and(|cur| cur != v) {
                    eprintln!("mode=exact overrides {k}={} → {v}", std::env::var(k).unwrap_or_default());
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
/// `fast` bounds BF16 + TF32 + the order-1 device sampler. Calibrate from the
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
            // Device UniPC is order-1 only (host path is order-2 bh2), so the
            // fast sampler is expected to drift; this bounds, not zeroes, it.
            denoise: 0.25,
        },
    }
}
