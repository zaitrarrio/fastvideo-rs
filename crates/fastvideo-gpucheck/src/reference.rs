//! GPU-vs-CPU-path references produced on the test box itself.
//!
//! The same `fv-gpucheck` binary runs a stage twice in separate processes
//! (cudarc caches its device and precision flags per process):
//!
//! 1. `--device cpu --dump DIR`: cudarc's host path, exact mode → outputs saved.
//! 2. `--device cuda --reference DIR`: the GPU path recomputes the same outputs
//!    from the same seeded weights/inputs and every output is compared.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context};
use serde_json::json;

use crate::metrics::{diff, jf, psnr};
use crate::report::{Report, StageResult};
use crate::st::{self, F32Tensor};

pub enum RefIo {
    Dump {
        path: PathBuf,
        outputs: Vec<(String, F32Tensor)>,
        videos: Option<PathBuf>,
    },
    Compare {
        path: PathBuf,
        reference: HashMap<String, F32Tensor>,
        videos: Option<PathBuf>,
    },
}

/// What an output is: plain tensor, or `[1, 3, F, H, W]` video in `[-1, 1]`
/// (reported with PSNR and saved as an mp4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Out {
    Tensor,
    Video,
}

pub const VIDEO_FPS: u32 = 16;

/// Write `frames/frame-%03d.png` + `<name>.mp4` under `dir/<name>/`. Encoding
/// problems (e.g. no ffmpeg) are reported, not fatal: the numbers are the test.
fn save_video(report: &mut Report, dir: &Path, name: &str, shape: &[usize], data: &[f32]) {
    let timer = Instant::now();
    let clip_dir = dir.join(name);
    let result = (|| -> anyhow::Result<(usize, String)> {
        let video = fastvideo_cudarc::CudaTensor::from_vec(data.to_vec(), shape.to_vec())?;
        let frames = clip_dir.join("frames");
        let paths = fastvideo_cudarc::wan::pipeline::write_frames(&video, &frames)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mp4 = fastvideo_cudarc::wan::pipeline::mux_mp4(&frames, VIDEO_FPS)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let dest = clip_dir.join(format!("{name}.mp4"));
        std::fs::rename(&mp4, &dest)?;
        Ok((paths.len(), dest.to_string_lossy().into_owned()))
    })();
    let seconds = timer.elapsed().as_secs_f64();
    match result {
        Ok((frames, mp4)) => report.note(
            format!("video/{name}"),
            json!({"mp4": mp4, "frames": frames, "shape": shape, "fps": VIDEO_FPS, "encode_seconds": seconds}),
        ),
        Err(e) => report.note(
            format!("video/{name}"),
            json!({"mp4": null, "error": format!("{e:#}"), "shape": shape, "encode_seconds": seconds}),
        ),
    }
}

impl RefIo {
    pub fn new(dump: Option<&Path>, reference: Option<&Path>, stage: &str) -> anyhow::Result<Self> {
        match (dump, reference) {
            (Some(dir), None) => Ok(RefIo::Dump {
                path: dir.join(format!("{stage}.safetensors")),
                outputs: Vec::new(),
                videos: None,
            }),
            (None, Some(dir)) => {
                let path = dir.join(format!("{stage}.safetensors"));
                let meta: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(path.with_extension("json")).with_context(|| {
                        format!(
                            "no reference manifest for {} (run with --dump first)",
                            path.display()
                        )
                    })?,
                )?;
                if meta["device"] != "cpu" || meta["mode"] != "exact" {
                    bail!(
                        "{}: reference must come from --device cpu --mode exact, got {meta}",
                        path.display()
                    );
                }
                // References are keyed by the CPU-path sources that produce
                // them (scripts/gpu/lib.sh fv_ref_key), not the binary: GPU-only
                // changes reuse them, CPU-path changes must re-dump.
                if let (Some(want), Ok(have)) =
                    (meta["ref_key"].as_str(), std::env::var("FV_REF_KEY"))
                {
                    if want != have {
                        bail!("reference key {want} != this run's {have}: CPU-path sources changed; re-dump");
                    }
                }
                Ok(RefIo::Compare {
                    reference: st::load(&path)?,
                    path,
                    videos: None,
                })
            }
            _ => {
                bail!("pass exactly one of --dump <dir> (CPU path) or --reference <dir> (GPU run)")
            }
        }
    }

    /// Save every video output as an mp4 under `dir`.
    pub fn with_videos(mut self, dir: PathBuf) -> Self {
        match &mut self {
            RefIo::Dump { videos, .. } | RefIo::Compare { videos, .. } => *videos = Some(dir),
        }
        self
    }

    pub fn is_dump(&self) -> bool {
        matches!(self, RefIo::Dump { .. })
    }

    /// Record (dump) or compare one output. `rel_limit` bounds relative L2;
    /// `seconds` is how long the output took to generate (recorded in the report).
    pub fn output(
        &mut self,
        report: &mut Report,
        name: &str,
        shape: &[usize],
        data: Vec<f32>,
        rel_limit: f64,
        seconds: f64,
        kind: Out,
    ) -> StageResult<()> {
        let bad = crate::metrics::non_finite(&data);
        let video = kind == Out::Video;
        if video {
            let dir = match self {
                RefIo::Dump { videos, .. } | RefIo::Compare { videos, .. } => videos.clone(),
            };
            if let Some(dir) = dir {
                save_video(report, &dir, name, shape, &data);
            }
        }
        match self {
            RefIo::Dump { outputs, .. } => {
                report.check(
                    format!("{name}/finite"),
                    bad == 0,
                    json!({"non_finite": bad, "numel": data.len(), "seconds": seconds}),
                    json!({"non_finite": 0}),
                )?;
                outputs.push((name.to_string(), F32Tensor::new(shape.to_vec(), data)?));
                Ok(())
            }
            RefIo::Compare {
                reference, path, ..
            } => {
                let want = reference
                    .get(name)
                    .with_context(|| format!("{} has no output `{name}`", path.display()))?;
                if want.shape != shape {
                    return report.check(
                        format!("{name}/shape"),
                        false,
                        json!({"gpu": shape, "cpu_path": want.shape}),
                        json!({}),
                    );
                }
                let d = diff(&data, &want.data);
                let mut values = d.to_json();
                values["seconds"] = json!(seconds);
                if video {
                    values["psnr_db"] = jf(psnr(&data, &want.data, 2.0));
                }
                report.check(
                    name,
                    d.within(rel_limit),
                    values,
                    json!({"rel_l2": rel_limit}),
                )
            }
        }
    }

    pub fn finish(
        self,
        report: &mut Report,
        mode: crate::mode::Mode,
        device: &str,
    ) -> anyhow::Result<()> {
        if let RefIo::Dump { path, outputs, .. } = self {
            let refs: Vec<(&str, &F32Tensor)> =
                outputs.iter().map(|(n, t)| (n.as_str(), t)).collect();
            st::save(&path, &refs)?;
            let meta = json!({
                "device": device,
                "mode": mode,
                "git_sha": std::env::var("FV_GIT_SHA").ok(),
                "ref_key": std::env::var("FV_REF_KEY").ok(),
                "outputs": outputs.iter().map(|(n, t)| json!({"name": n, "shape": t.shape})).collect::<Vec<_>>(),
            });
            std::fs::write(
                path.with_extension("json"),
                serde_json::to_string_pretty(&meta)?,
            )?;
            report.set("reference_written", path);
        }
        Ok(())
    }
}
