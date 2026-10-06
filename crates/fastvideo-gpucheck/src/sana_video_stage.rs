//! SANA-Video 2B stages (docs/ports/sana-video.md). `gen` is the Sol-Engine
//! benchmark arm: one warm process, the published canvas by default
//! (832x480, 81 frames, 50 steps, cfg 6), `--arm baseline|full`.

use std::path::{Path, PathBuf};

use anyhow::Context;
use fastvideo_cudarc::sana_video::{SanaVideoOutput, SanaVideoPipeline, SanaVideoRequest};
use fastvideo_models::sana_video::{sol, SanaOptimizations};
use serde_json::json;

use crate::report::{Report, StageResult};

#[derive(clap::Subcommand, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Stage {
    /// Print the published canvas, schedule and arm switches.
    Info {
        #[arg(long, default_value = "baseline")]
        arm: String,
    },
    /// Generate one clip end to end; timings, reuse count and peak VRAM.
    Gen {
        /// Diffusers tree (`Efficient-Large-Model/SANA-Video_2B_480p_diffusers`).
        #[arg(long)]
        weights: PathBuf,
        /// VAE directory when the tree has no `vae/` (any Wan 2.1 Diffusers `vae/`).
        #[arg(long)]
        vae: Option<PathBuf>,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "")]
        negative_prompt: String,
        /// `baseline` or `full` (EasyCache 0.1 + QKV merge + bf16 linear attention);
        /// `FASTVIDEO_SANA_*` overrides individual switches.
        #[arg(long, default_value = "baseline")]
        arm: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        height: Option<usize>,
        #[arg(long)]
        width: Option<usize>,
        #[arg(long)]
        num_frames: Option<usize>,
        #[arg(long)]
        num_steps: Option<usize>,
        #[arg(long)]
        guidance_scale: Option<f32>,
        #[arg(long)]
        motion_score: Option<u32>,
        /// Raw f32 initial noise (parity runs).
        #[arg(long)]
        noise: Option<PathBuf>,
        #[arg(long, default_value = "gpucheck-out/sana-video-gen")]
        clip: PathBuf,
        /// One untimed generation first (the published numbers are warm).
        #[arg(long)]
        warm: bool,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
}

fn arm(name: &str) -> anyhow::Result<SanaOptimizations> {
    let get = |k: &str| std::env::var(k).ok();
    SanaOptimizations::parse(
        Some(name),
        get("FASTVIDEO_SANA_EASYCACHE").as_deref(),
        get("FASTVIDEO_SANA_QKV_MERGE").as_deref(),
        get("FASTVIDEO_SANA_LINATTN_BF16").as_deref(),
    )
    .map_err(anyhow::Error::msg)
}

fn opt_json(o: &SanaOptimizations) -> serde_json::Value {
    json!({"easycache": o.easycache, "qkv_merge": o.qkv_merge, "linattn_bf16": o.linattn_bf16})
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Info { arm: a } => {
            let o = arm(a)?;
            report.set(
                "published",
                json!({
                    "height": sol::PUBLISHED_HEIGHT, "width": sol::PUBLISHED_WIDTH,
                    "frames": sol::PUBLISHED_FRAMES, "steps": sol::PUBLISHED_STEPS,
                    "guidance": sol::PUBLISHED_GUIDANCE, "speedup": sol::PUBLISHED_SPEEDUP,
                    "arm": a, "switches": opt_json(&o),
                }),
            );
            Ok(())
        }
        Stage::Gen {
            weights,
            vae,
            prompt,
            negative_prompt,
            arm: a,
            seed,
            height,
            width,
            num_frames,
            num_steps,
            guidance_scale,
            motion_score,
            noise,
            clip,
            warm,
            device,
        } => {
            let mut r = SanaVideoRequest::published(prompt.clone(), *seed);
            r.negative_prompt = negative_prompt.clone();
            r.height = height.unwrap_or(r.height);
            r.width = width.unwrap_or(r.width);
            r.num_frames = num_frames.unwrap_or(r.num_frames);
            r.num_steps = num_steps.unwrap_or(r.num_steps);
            r.guidance_scale = guidance_scale.unwrap_or(r.guidance_scale);
            r.motion_score = *motion_score;
            r.noise_path = noise.clone();
            gen(report, weights, vae.as_deref(), a, &r, clip, *warm, device)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gen(
    report: &mut Report,
    weights: &Path,
    vae: Option<&Path>,
    arm_name: &str,
    request: &SanaVideoRequest,
    clip: &Path,
    warm: bool,
    device: &str,
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let o = arm(arm_name)?;
    report.set(
        "request",
        json!({
            "prompt": request.prompt, "seed": request.seed, "height": request.height,
            "width": request.width, "num_frames": request.num_frames,
            "num_steps": request.num_steps, "guidance_scale": request.guidance_scale,
            "flow_shift": request.flow_shift, "arm": arm_name, "switches": opt_json(&o),
            "warm": warm,
        }),
    );
    let peak = crate::gpu::PeakMem::start();
    let load_t = std::time::Instant::now();
    let pipeline = SanaVideoPipeline::load(weights, o, vae)?;
    let load_s = load_t.elapsed().as_secs_f64();
    report.note("load", json!({"seconds": load_s}));

    let timings = |out: &SanaVideoOutput, total: f64| {
        let t = &out.timings;
        json!({
            "total_s": total, "text_s": t.text_s, "denoise_s": t.denoise_s,
            "step_s": t.step_s, "decode_s": t.decode_s, "write_s": t.write_s,
            "reused_steps": t.reused_steps,
        })
    };
    if warm {
        let timer = std::time::Instant::now();
        let cold = pipeline.generate(request, &clip.join("cold"))?;
        report.note("cold_generation", timings(&cold, timer.elapsed().as_secs_f64()));
        fastvideo_cudarc::wan::stats::phase_reset();
    }
    let timer = std::time::Instant::now();
    let out = pipeline.generate(request, clip)?;
    let total = timer.elapsed().as_secs_f64();
    let mut values = timings(&out, total);
    values["warm"] = json!(warm);
    values["load_s"] = json!(load_s);
    values["peak_mib"] = json!(peak.stop());
    // Comparable with Sol-Engine's warm end-to-end (text encode → decode,
    // frame writing excluded).
    values["generate_s"] = json!(total - out.timings.write_s);
    std::fs::create_dir_all(clip).ok();
    std::fs::write(
        clip.join("benchmark.json"),
        serde_json::to_vec_pretty(&values).unwrap_or_default(),
    )
    .with_context(|| format!("write {}", clip.join("benchmark.json").display()))?;
    report.note("timings", values);
    report.set(
        "output",
        json!({"frames": out.frames, "first_frame": out.frame_paths.first()}),
    );
    report.check(
        "frames",
        out.frames == request.num_frames,
        json!({"decoded": out.frames}),
        json!({"expected": request.num_frames}),
    )?;
    let probe = out
        .frame_paths
        .get(out.frame_paths.len() / 2)
        .ok_or_else(|| anyhow::anyhow!("no frames were written"))?;
    let img = image::open(probe)
        .with_context(|| format!("open {probe}"))?
        .to_rgb8();
    let px: Vec<f32> = img.as_raw().iter().map(|&b| f32::from(b) / 255.0).collect();
    let (mean, std) = crate::metrics::mean_std(&px);
    report.check(
        "middle_frame_not_flat",
        std > 0.02 && mean > 0.02 && mean < 0.98,
        json!({"mean": mean, "std": std, "frame": probe}),
        json!({"std_min": 0.02}),
    )?;
    Ok(())
}
