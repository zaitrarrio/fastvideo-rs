//! Cosmos3-Super host math for `fastvideo-cudarc::cosmos3`.
//! See docs/ports/cosmos3.md.
//!
//! TeaCache and the official canvas live in [`crate::cosmos::sol`]. This
//! module re-exports them; it does not invent a second cache.

pub mod config;
pub mod prompt;
pub mod rope;
pub mod schedule;

pub use crate::cosmos::sol::{
    fp4_linear, official_requested, teacache_requested, SolCosmosTea, TeaCacheWindow,
    FP4_SKIP_FIRST, FP4_SKIP_LAST, OFFICIAL_FLOW_SHIFT, OFFICIAL_FPS, OFFICIAL_FRAMES,
    OFFICIAL_GUIDANCE, OFFICIAL_HEIGHT, OFFICIAL_STEPS, OFFICIAL_WIDTH, TEACACHE_MAX_CONSECUTIVE,
    TEACACHE_START_STEP, TEACACHE_THRESHOLD,
};
pub use config::{Cosmos3Preset, Cosmos3TransformerConfig};

/// Published baseline (4× GB200, sequence parallel, warmup excluded).
pub const PUBLISHED_BASELINE_S: f64 = 130.41;
pub const PUBLISHED_BASELINE_DENOISE_S: f64 = 121.4198;
pub const PUBLISHED_BASELINE_DECODE_S: f64 = 5.8017;
/// `site_docs/pipelines/cosmos3.md`: TeaCache + step-selective NVFP4.
pub const PUBLISHED_FULLOPT_SPEEDUP: f64 = 2.26;

/// Latent grid of a `(frames, height, width)` request: Wan 2.2 VAE (×4 time,
/// ×16 space), then 2×2 patches with the height/width padded up to even.
pub fn latent_grid(frames: usize, height: usize, width: usize, patch: usize) -> ([usize; 3], [usize; 3]) {
    let lt = (frames - 1) / 4 + 1;
    let (lh, lw) = (height / 16, width / 16);
    ([lt, lh, lw], [lt, lh.div_ceil(patch), lw.div_ceil(patch)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_grid() {
        let (lat, grid) = latent_grid(189, 720, 1280, 2);
        assert_eq!(lat, [48, 45, 80]);
        assert_eq!(grid, [48, 23, 40]);
        assert_eq!(grid.iter().product::<usize>(), 44_160);
    }
}
