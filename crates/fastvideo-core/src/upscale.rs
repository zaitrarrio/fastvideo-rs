//! `generate --upscale 2|3|4`: the Real-ESRGAN compact upscaler
//! (`fastvideo_cudarc::rt::esrgan`, `docs/ports/upscaler.md`) over what a
//! family's generate wrote, for every family whose output is PNG frames.

use std::path::{Path, PathBuf};

use crate::error::{FastVideoError, Result};
use crate::family::Output;

/// What the CLI asked for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpscaleOptions {
    /// 2, 3 or 4.
    pub factor: usize,
    /// `--upscale-weights`; else the environment and the weights roots
    /// (`fastvideo_cudarc::rt::esrgan::resolve_weights`).
    pub weights: Option<PathBuf>,
}

pub use fastvideo_cudarc::rt::esrgan::UpscaleReport;

/// Upscale the frames of `out` (the directory of its first frame) by
/// `options.factor` into `frames-up/` and `output-up.mp4` beside them, the
/// mp4 at the original's frame rate with the family's audio track (H3's
/// `audio.wav`) when there is one. `weights_root` is the run's `--weights`
/// (an `auxiliary/upscalers/` sibling of its tree is searched).
pub fn upscale_output(
    out: &Output,
    options: &UpscaleOptions,
    weights_root: Option<&Path>,
) -> Result<UpscaleReport> {
    let msg = |e: String| FastVideoError::Message(e);
    if !(2..=4).contains(&options.factor) {
        return Err(msg(format!("--upscale {}: want 2, 3 or 4", options.factor)));
    }
    let first = out
        .frame_paths
        .first()
        .ok_or_else(|| msg(format!("--upscale: {} wrote no frames", out.family)))?;
    let dir = Path::new(first)
        .parent()
        .ok_or_else(|| msg(format!("--upscale: {first} has no directory")))?;
    let weights =
        fastvideo_cudarc::rt::esrgan::resolve_weights(options.weights.as_deref(), weights_root)
            .map_err(|e| msg(e.to_string()))?;
    let audio = out.wav.as_deref().map(Path::new).filter(|p| p.is_file());
    fastvideo_cudarc::rt::esrgan::upscale_dir(dir, options.factor, &weights, None, audio)
        .map_err(|e| msg(e.to_string()))
}
