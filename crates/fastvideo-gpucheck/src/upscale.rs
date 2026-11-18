//! `--upscale N` for the generation stages: the Real-ESRGAN compact
//! upscaler (`fastvideo_cudarc::rt::esrgan`) over a clip directory after
//! the pipeline wrote it, its numbers for `benchmark.json`.

use std::path::{Path, PathBuf};

use serde_json::json;

use crate::report::StageResult;

/// A resolved `--upscale N` / `--upscale-weights`.
#[derive(Debug, Clone)]
pub struct UpscaleArgs {
    pub factor: usize,
    pub weights: PathBuf,
}

impl UpscaleArgs {
    /// `None` without `--upscale`; with it, the factor checked (2 ..= 4) and
    /// the weights resolved now (a missing file fails before the clip runs).
    pub fn resolve(
        factor: Option<usize>,
        weights: Option<&Path>,
        weights_root: Option<&Path>,
    ) -> StageResult<Option<Self>> {
        let Some(factor) = factor else {
            return Ok(None);
        };
        if !(2..=4).contains(&factor) {
            return Err(anyhow::anyhow!("--upscale {factor}: want 2, 3 or 4").into());
        }
        let weights = fastvideo_cudarc::rt::esrgan::resolve_weights(weights, weights_root)?;
        Ok(Some(Self { factor, weights }))
    }

    /// Upscale `<dir>/frame-NNN.png` into `<dir>/frames-up/` and
    /// `<dir>/output-up.mp4`; the report as JSON (`upscale_s`, sizes, paths).
    pub fn run(
        &self,
        dir: &Path,
        fps: Option<u32>,
        audio: Option<&Path>,
    ) -> StageResult<serde_json::Value> {
        let r =
            fastvideo_cudarc::rt::esrgan::upscale_dir(dir, self.factor, &self.weights, fps, audio)?;
        eprintln!(
            "upscale ×{}: {} frames {}x{} → {}x{} in {:.2}s ({}; read {:.2}s, load {:.2}s, write {:.2}s)",
            r.factor, r.frames, r.in_width, r.in_height, r.out_width, r.out_height, r.upscale_s, r.device, r.read_s, r.load_s, r.write_s
        );
        Ok(json!({
            "model": "realesr-animevideov3",
            "factor": r.factor,
            "frames": r.frames,
            "in": [r.in_width, r.in_height],
            "out": [r.out_width, r.out_height],
            "batch": r.batch,
            "device": r.device,
            "upscale_s": r.upscale_s,
            "read_s": r.read_s,
            "load_s": r.load_s,
            "write_s": r.write_s,
            "fps": r.frames as f64 / r.upscale_s.max(1e-9),
            "mp4": r.mp4,
            "frames_dir": dir.join("frames-up").display().to_string(),
            "weights": r.weights,
        }))
    }
}
