//! HunyuanVideo 1.5 stages. `gen` matches the H3/LTX report: clip, step
//! timings, peak MiB.

use std::path::{Path, PathBuf};

use anyhow::Context;
use fastvideo_cudarc::hunyuan15::{Hunyuan15Output, Hunyuan15Pipeline, Hunyuan15Request};
use fastvideo_models::hunyuan15::Hunyuan15Preset;
use serde_json::json;

use crate::report::{Report, StageResult};

#[derive(clap::Subcommand, Debug)]
pub enum Stage {
    /// Print the preset canvas and schedule the port targets.
    Info {
        #[arg(long, default_value = "hy15_480p_t2v")]
        preset: String,
    },
    /// Generate one clip end to end and report timings and peak VRAM.
    Gen {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        prompt: String,
        /// `hy15_480p_t2v`, `hy15_480p_i2v_distilled`, `hy15_720p_t2v`,
        /// `hy15_720p_i2v_distilled`, `hy15_1080p_sr`.
        #[arg(long, default_value = "hy15_480p_t2v")]
        preset: String,
        #[arg(long, default_value_t = 1024)]
        seed: u64,
        #[arg(long)]
        height: Option<usize>,
        #[arg(long)]
        width: Option<usize>,
        #[arg(long)]
        num_frames: Option<usize>,
        #[arg(long)]
        num_steps: Option<usize>,
        #[arg(long, default_value = "gpucheck-out/hunyuan-gen")]
        clip: PathBuf,
        #[arg(long)]
        warm: bool,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Info { preset } => {
            let p = parse_preset(preset)?;
            let (h, w, f) = p.canvas();
            report.set(
                "preset",
                json!({
                    "name": p.as_str(),
                    "height": h,
                    "width": w,
                    "num_frames": f,
                    "steps": p.default_steps(),
                    "flow_shift": p.flow_shift(),
                }),
            );
            Ok(())
        }
        Stage::Gen {
            weights,
            prompt,
            preset,
            seed,
            height,
            width,
            num_frames,
            num_steps,
            clip,
            warm,
            device,
        } => gen(
            report,
            weights,
            prompt,
            parse_preset(preset)?,
            *seed,
            *height,
            *width,
            *num_frames,
            *num_steps,
            clip,
            *warm,
            device,
        ),
    }
}

fn parse_preset(s: &str) -> anyhow::Result<Hunyuan15Preset> {
    Hunyuan15Preset::from_cli(s).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown --preset '{s}' (hy15_480p_t2v|hy15_480p_i2v_distilled|hy15_720p_t2v|hy15_720p_i2v_distilled|hy15_1080p_sr)"
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn gen(
    report: &mut Report,
    weights: &Path,
    prompt: &str,
    preset: Hunyuan15Preset,
    seed: u64,
    height: Option<usize>,
    width: Option<usize>,
    num_frames: Option<usize>,
    num_steps: Option<usize>,
    clip: &Path,
    warm: bool,
    device: &str,
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let mut request = Hunyuan15Request::from_preset(preset, prompt, seed);
    if let Some(h) = height {
        request.height = h;
    }
    if let Some(w) = width {
        request.width = w;
    }
    if let Some(f) = num_frames {
        request.num_frames = f;
    }
    if let Some(s) = num_steps {
        request.num_steps = s;
    }
    report.set(
        "request",
        json!({
            "prompt": prompt,
            "preset": preset.as_str(),
            "seed": seed,
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "num_steps": request.num_steps,
            "warm": warm,
        }),
    );

    let peak = crate::gpu::PeakMem::start();
    let load_t = std::time::Instant::now();
    let pipeline = Hunyuan15Pipeline::load(weights, preset)?;
    let load_s = load_t.elapsed().as_secs_f64();
    report.note("load", json!({"seconds": load_s}));

    let timings = |out: &Hunyuan15Output, total: f64| {
        let t = &out.timings;
        json!({
            "total_s": total,
            "text_s": t.text_s,
            "denoise_s": t.denoise_s,
            "step_s": t.step_s,
            "decode_s": t.decode_s,
            "write_s": t.write_s,
        })
    };

    if warm {
        let timer = std::time::Instant::now();
        let cold = pipeline.generate(&request, &clip.join("cold"))?;
        report.note(
            "cold_generation",
            timings(&cold, timer.elapsed().as_secs_f64()),
        );
        fastvideo_cudarc::wan::stats::phase_reset();
    }

    let timer = std::time::Instant::now();
    let out = pipeline.generate(&request, clip)?;
    let total = timer.elapsed().as_secs_f64();
    let mut values = timings(&out, total);
    values["warm"] = json!(warm);
    values["load_s"] = json!(load_s);
    values["peak_mib"] = json!(peak.stop());
    report.note("timings", values);
    report.set(
        "output",
        json!({"frames": out.frames, "first_frame": out.frame_paths.first()}),
    );
    report.check(
        "frames",
        out.frames > 0,
        json!({"decoded": out.frames}),
        json!({"expected": ">0"}),
    )?;
    let probe = out
        .frame_paths
        .get(out.frame_paths.len() / 2)
        .ok_or_else(|| anyhow::anyhow!("no frames were written"))?;
    let img = image::open(probe)
        .with_context(|| format!("open {probe}"))?
        .to_rgb8();
    let values: Vec<f32> = img.as_raw().iter().map(|&b| f32::from(b) / 255.0).collect();
    let (mean, std) = crate::metrics::mean_std(&values);
    report.check(
        "middle_frame_not_flat",
        std > 0.02 && mean > 0.02 && mean < 0.98,
        json!({"mean": mean, "std": std, "frame": probe}),
        json!({"std_min": 0.02}),
    )?;
    Ok(())
}
