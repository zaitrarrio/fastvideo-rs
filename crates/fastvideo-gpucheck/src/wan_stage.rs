//! Wan 2.1 / FastWan stages. `gen` is the matrix's Wan cell: every prompt of
//! a prompt set through one resident [`WanPipeline`] (text encoder included),
//! warm after an untimed first generation, with `benchmark.json` beside the
//! clip directory as the H3 and LTX-2 cells write it.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::Subcommand;
use fastvideo_cudarc::wan::pipeline::WanOutput;
use fastvideo_cudarc::{GenerateConfig, LoadParts, WanPipeline};
use serde_json::json;

use crate::benchmark::PromptSpec;
use crate::report::{Report, StageResult};

/// The Wan negative prompt (FastVideo's `WanT2V480PConfig` default). Encoded
/// only when a guidance scale above 1 uses it.
pub const WAN_NEGATIVE: &str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards";

#[derive(Subcommand)]
pub enum Stage {
    /// Generate clips end to end (UMT5 → DiT → decode → mp4, PNGs after) and
    /// write `benchmark.json`. Defaults are the FastWan 1.3B DMD recipe:
    /// 480x832, 81 frames, 3 steps at timesteps 1000/757/522, shift 8.
    Gen {
        /// Diffusers root (`transformer/`, `vae/`, `text_encoder/`, `tokenizer/`).
        #[arg(long)]
        weights: PathBuf,
        /// Registry preset of the checkpoint (architecture and sampler family).
        #[arg(long, default_value = "fast_wan_t2v_480p")]
        preset: String,
        #[arg(long, default_value = "a cat walking on the grass")]
        prompt: String,
        /// A prompt set (`scripts/gpu/prompts-eval.json`): every prompt in this
        /// process, each clip under `<clip-dir>/<name>/`.
        #[arg(long)]
        prompts: Option<PathBuf>,
        #[arg(long, default_value_t = 1024)]
        seed: u64,
        #[arg(long, default_value = WAN_NEGATIVE)]
        negative: String,
        #[arg(long, default_value_t = 480)]
        height: usize,
        #[arg(long, default_value_t = 832)]
        width: usize,
        #[arg(long, default_value_t = 81)]
        num_frames: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        /// UniPC instead of the DMD sampler (base Wan checkpoints).
        #[arg(long)]
        unipc: bool,
        #[arg(long, default_value_t = 1.0)]
        guidance: f32,
        #[arg(long, default_value_t = 8.0)]
        flow_shift: f64,
        #[arg(long, default_value_t = 16)]
        fps: u32,
        #[arg(long)]
        no_mp4: bool,
        #[arg(long, default_value = "gpucheck-out/wan-gen")]
        clip_dir: PathBuf,
        /// Decode with the full Wan VAE even for the distilled presets (which
        /// default to TAEHV when its weights are found): FASTVIDEO_WAN_VAE=full.
        #[arg(long)]
        full_vae: bool,
        /// UMT5 prompt cache directory. Default: `text-cache` next to the clip
        /// directory. A prompt seen before costs a file read.
        #[arg(long)]
        text_cache: Option<PathBuf>,
        /// Encode every prompt (no disk cache).
        #[arg(long)]
        no_text_cache: bool,
        /// One untimed generation first (recorded as `cold_generation`).
        #[arg(long)]
        warm: bool,
        /// First frame (PNG/JPEG) for image-to-video: Wan 2.2 TI2V-5B pins
        /// its latent to frame 0; the Wan 2.1 I2V presets pack it as the
        /// 36-channel condition.
        #[arg(long)]
        image: Option<PathBuf>,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
    /// The open-ended causal SF-Wan rollout (`wan::stream`): `--parity`
    /// against the bounded 81-frame path, then each `--run` for its video
    /// seconds (see `wan_stream.rs` for the run syntax).
    Stream {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value = "sf_wan_t2v_1_3b")]
        preset: String,
        #[arg(long, default_value = "a cat walking on the grass")]
        prompt: String,
        /// The prompt a run's `switch_at` changes to.
        #[arg(long, default_value = "a dog running along a beach at sunset")]
        switch_prompt: String,
        #[arg(long, default_value_t = 1024)]
        seed: u64,
        #[arg(long, default_value_t = 480)]
        height: usize,
        #[arg(long, default_value_t = 832)]
        width: usize,
        #[arg(long, default_value_t = 16)]
        fps: u32,
        #[arg(long)]
        parity: bool,
        /// `name,seconds=S[,rope=rel|abs][,sink=N][,window=N][,switch_at=S][,switch=keep|reset][,drop_rgb=1]`
        #[arg(long = "run")]
        runs: Vec<String>,
        /// Video seconds per statistics window.
        #[arg(long, default_value_t = 10.0)]
        window_s: f64,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
    /// Wan 2.2 TI2V-5B module parity against a reference dump
    /// (`scripts/gpu/upstream/oracle_wan22.py`, Diffusers): VAE encode and
    /// decode of the dump's video / latents, and one DiT forward per
    /// `<case>_dit_latents` (text-to-video `t2v_`, per-frame timesteps
    /// `i2v_`) with the dump's inputs injected. Our tensors go to `--dump-out`
    /// under the reference's names for `compare-dumps`; decode PSNR and
    /// max-abs are checked here.
    Oracle {
        /// Diffusers root (`transformer/`, `vae/`).
        #[arg(long)]
        weights: PathBuf,
        /// The reference dump directory.
        #[arg(long)]
        reference: PathBuf,
        /// Where our tensors go (created).
        #[arg(long)]
        dump_out: PathBuf,
        #[arg(long, default_value = "wan_2_2_ti2v_5b")]
        preset: String,
        /// Also decode the reference latents through TAEHV (`taew2_2`) from
        /// this file or directory: PSNR against the reference VAE decode.
        #[arg(long)]
        taehv: Option<PathBuf>,
        /// Minimum decode PSNR (dB, range 2) against the reference.
        #[arg(long, default_value_t = 40.0)]
        min_psnr: f64,
        #[arg(long)]
        skip_dit: bool,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Gen {
            weights,
            preset,
            prompt,
            prompts,
            seed,
            negative,
            height,
            width,
            num_frames,
            steps,
            unipc,
            guidance,
            flow_shift,
            fps,
            no_mp4,
            clip_dir,
            warm,
            full_vae,
            text_cache,
            no_text_cache,
            image,
            device,
        } => {
            if *full_vae {
                std::env::set_var("FASTVIDEO_WAN_VAE", "full");
            }
            let (set, multi) = match prompts {
                Some(file) => (crate::benchmark::load_prompts(file, *seed)?, true),
                None => (
                    vec![PromptSpec {
                        name: "default".into(),
                        prompt: prompt.clone(),
                        seed: *seed,
                    }],
                    false,
                ),
            };
            let tokenizer = weights.join("tokenizer").join("tokenizer.json");
            if !tokenizer.is_file() {
                return Err(anyhow::anyhow!(
                    "{} missing (the Diffusers root needs tokenizer/)",
                    tokenizer.display()
                )
                .into());
            }
            let dmd = !*unipc;
            let base = GenerateConfig {
                negative_prompt: negative.clone(),
                height: *height,
                width: *width,
                num_frames: *num_frames,
                num_inference_steps: *steps,
                guidance_scale: *guidance,
                flow_shift: *flow_shift,
                is_dmd: dmd,
                dmd_steps: dmd.then(|| dmd_steps(*steps)),
                tokenizer_path: Some(tokenizer.to_string_lossy().into_owned()),
                image_path: image.as_ref().map(|p| p.to_string_lossy().into_owned()),
                fps: *fps,
                text_cache: (!*no_text_cache).then(|| {
                    text_cache.clone().unwrap_or_else(|| {
                        clip_dir
                            .parent()
                            .unwrap_or(Path::new("."))
                            .join("text-cache")
                    })
                }),
                ..GenerateConfig::default()
            };
            gen(
                report,
                &GenArgs {
                    weights,
                    preset,
                    set: &set,
                    multi,
                    base,
                    mp4: !*no_mp4,
                    clip_dir,
                    warm: *warm,
                    device,
                },
            )
        }
        Stage::Stream {
            weights,
            preset,
            prompt,
            switch_prompt,
            seed,
            height,
            width,
            fps,
            parity,
            runs,
            window_s,
            device,
        } => crate::wan_stream::run(
            report,
            &crate::wan_stream::Args {
                weights,
                preset,
                prompt,
                switch_prompt,
                seed: *seed,
                height: *height,
                width: *width,
                fps: *fps,
                parity: *parity,
                runs,
                window_s: *window_s,
                device,
            },
        ),
        Stage::Oracle {
            weights,
            reference,
            dump_out,
            preset,
            taehv,
            min_psnr,
            skip_dit,
            device,
        } => crate::wan_oracle::run(
            report,
            &crate::wan_oracle::Args {
                weights,
                reference,
                out: dump_out,
                preset,
                taehv: taehv.as_deref(),
                min_psnr: *min_psnr,
                skip_dit: *skip_dit,
                device,
            },
        ),
    }
}

/// The checkpoint a preset names, for `benchmark.json`.
fn model_name(preset: &str) -> &'static str {
    match preset {
        "wan_2_2_ti2v_5b" => "Wan2.2-TI2V-5B",
        "fast_wan_2_2_ti2v_5b" => "FastWan2.2-TI2V-5B",
        "wan_t2v_14b" => "Wan2.1-T2V-14B",
        "sf_wan_t2v_1_3b" => "SFWan2.1-T2V-1.3B",
        "wan_t2v_1_3b" => "Wan2.1-T2V-1.3B",
        _ => "FastWan2.1-T2V-1.3B",
    }
}

/// FastWan 1.3B's DMD timesteps, or `steps` evenly spaced from 1000.
fn dmd_steps(steps: usize) -> Vec<i32> {
    let base = fastvideo_models::schedulers::FAST_WAN_1_3B_DMD_STEPS;
    if steps == base.len() {
        return base.to_vec();
    }
    (0..steps)
        .map(|i| 1000 - (i as i32 * 1000 / steps.max(1) as i32))
        .collect()
}

struct GenArgs<'a> {
    weights: &'a Path,
    preset: &'a str,
    set: &'a [PromptSpec],
    multi: bool,
    base: GenerateConfig,
    mp4: bool,
    clip_dir: &'a Path,
    warm: bool,
    device: &'a str,
}

/// SHA-256 over every PNG frame's bytes in order: two clips with the same
/// digest are byte-identical (the identity check for exact switches).
fn frames_sha256(paths: &[String]) -> anyhow::Result<String> {
    let mut all = Vec::new();
    for p in paths {
        let bytes = std::fs::read(p).with_context(|| format!("read {p}"))?;
        all.extend_from_slice(&crate::benchmark::sha256(&bytes));
    }
    Ok(crate::benchmark::sha256_hex_bytes(&all))
}

fn timings_json(out: &WanOutput, total: f64) -> serde_json::Value {
    let t = &out.timings;
    json!({
        "total_s": total,
        "text_s": t.text_s,
        "text_cache": out.text_cache,
        "denoise_s": t.denoise_s,
        "step_s": t.step_s,
        "decode_s": t.decode_s,
        "video_vae_s": t.vae_s,
        "video_rgb_s": t.rgb_s,
        "video_push_s": t.push_s,
        "video_wait_s": t.wait_s,
        "video_encode_s": t.encode_s,
        "write_s": t.write_s,
        "decoder": out.decoder,
    })
}

fn gen(report: &mut Report, a: &GenArgs<'_>) -> StageResult<()> {
    report.set("device", crate::gpu::init(a.device)?);
    report.set(
        "request",
        json!({
            "weights": a.weights, "preset": a.preset,
            "height": a.base.height, "width": a.base.width, "num_frames": a.base.num_frames,
            "steps": a.base.num_inference_steps, "dmd": a.base.is_dmd, "dmd_steps": a.base.dmd_steps,
            "guidance": a.base.guidance_scale, "flow_shift": a.base.flow_shift, "warm": a.warm,
            "vsa": fastvideo_cudarc::wan::nn::vsa_enabled(),
            "text_cache": a.base.text_cache,
            "vae": std::env::var("FASTVIDEO_WAN_VAE").unwrap_or_else(|_| "auto".into()),
            "cond_cache": std::env::var("FASTVIDEO_WAN_COND_CACHE").map_or(true, |v| !matches!(v.trim(), "0" | "false" | "off")),
            "prompt_set": a.multi.then(|| a.set.iter().map(|p| json!({"name": p.name, "seed": p.seed})).collect::<Vec<_>>()),
        }),
    );
    let mut load_mem = Some(crate::gpu::PeakMem::start());
    let timer = std::time::Instant::now();
    let pipe = WanPipeline::load_with(a.weights, a.preset, LoadParts { text_encoder: true })
        .map_err(|e| anyhow::anyhow!("load {}: {e}", a.weights.display()))?;
    let load_s = timer.elapsed().as_secs_f64();
    report.note(
        "load",
        json!({"seconds": load_s, "mem_used_mib": crate::gpu::mem_info().map(|(f, t)| (t - f) >> 20)}),
    );
    let request = |spec: &PromptSpec| GenerateConfig {
        prompt: spec.prompt.clone(),
        seed: spec.seed,
        ..a.base.clone()
    };
    if a.warm {
        let timer = std::time::Instant::now();
        let cold = pipe
            .generate_to(&request(&a.set[0]), &a.clip_dir.join("cold"), a.mp4)
            .map_err(|e| anyhow::anyhow!("cold generation: {e}"))?;
        report.note(
            "cold_generation",
            timings_json(&cold, timer.elapsed().as_secs_f64()),
        );
        fastvideo_cudarc::wan::stats::phase_reset();
    }
    let mut docs: Vec<(PromptSpec, serde_json::Value)> = Vec::new();
    for spec in a.set {
        let dir = if a.multi {
            a.clip_dir.join(&spec.name)
        } else {
            a.clip_dir.to_path_buf()
        };
        let ck = |name: &str| {
            if a.multi {
                format!("{name}/{}", spec.name)
            } else {
                name.to_string()
            }
        };
        let mem = load_mem.take().unwrap_or_else(crate::gpu::PeakMem::start);
        fastvideo_cudarc::wan::evalstats::reset();
        let timer = std::time::Instant::now();
        let out = pipe
            .generate_to(&request(spec), &dir, a.mp4)
            .map_err(|e| anyhow::anyhow!("generate {}: {e}", spec.name))?;
        let wall = timer.elapsed().as_secs_f64();
        let peak_mib = mem.stop();
        let counters = fastvideo_cudarc::wan::evalstats::snapshot();
        let t = &out.timings;
        let total = wall - t.write_s;
        let digest = frames_sha256(&out.frame_paths)?;
        let mut values = timings_json(&out, total);
        values["peak_mib"] = json!(peak_mib);
        values["frames_sha256"] = json!(digest);
        report.note(
            if a.multi {
                format!("timings/{}", spec.name)
            } else {
                "timings".to_string()
            },
            values,
        );
        eprintln!(
            "wan gen {}: total {total:.2}s (text {:.2}s, denoise {:.2}s, decode {:.2}s) png {:.2}s peak {:?} MiB frames {digest}",
            spec.name, t.text_s, t.denoise_s, t.decode_s, t.write_s, peak_mib
        );
        let task = if a.base.image_path.is_some() {
            "i2v"
        } else {
            "t2v"
        };
        let doc = json!({
            "model": model_name(a.preset),
            "preset": a.preset,
            "dtype": "float32 weights, bf16 tensor-core GEMMs",
            "task": task,
            "workload": {
                "task": task,
                "prompt_name": spec.name,
                "prompt_sha256": crate::benchmark::sha256_hex(&spec.prompt),
                "seed": spec.seed,
                "height": a.base.height,
                "width": a.base.width,
                "num_frames": a.base.num_frames,
                "measured_steps": t.step_s.len(),
            },
            "total_s": total,
            "inference_time_s": total,
            "e2e_seconds": total,
            "generate_wall_s": wall,
            "png_frames_s": t.write_s,
            "denoise_s": t.denoise_s,
            "decode_s": t.decode_s,
            "text_s": t.text_s,
            "text_cache": out.text_cache,
            "load_s": load_s,
            "decoder": out.decoder,
            "stage_seconds": {
                "text": t.text_s,
                "denoise": t.denoise_s,
                "video_decode": t.decode_s,
                "video_vae": t.vae_s,
                "video_rgb": t.rgb_s,
                "video_push": t.push_s,
                "video_wait": t.wait_s,
                "video_encode_tail": t.encode_s,
                "png_frames": t.write_s,
            },
            "step_seconds": t.step_s,
            "peak_memory_mb": peak_mib,
            "max_device_memory_used_mib": peak_mib,
            "frames_sha256": digest,
        });
        let doc = crate::benchmark::merge(crate::benchmark::common("wan", a.warm), doc);
        docs.push((spec.clone(), crate::benchmark::merge(doc, counters)));
        report.check(
            ck("frames"),
            out.frames == a.base.num_frames && out.frame_paths.len() == a.base.num_frames,
            json!({"decoded": out.frames, "written": out.frame_paths.len()}),
            json!({"expected": a.base.num_frames}),
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
            ck("middle_frame_not_flat"),
            std > 0.02 && mean > 0.02 && mean < 0.98,
            json!({"mean": mean, "std": std, "frame": probe}),
            json!({"std_min": 0.02}),
        )?;
    }
    let path = crate::benchmark::path_beside(a.clip_dir);
    let doc = if a.multi {
        crate::benchmark::summarize(&docs)
    } else {
        docs.pop().map(|(_, d)| d).unwrap_or_default()
    };
    crate::benchmark::write(&path, &doc)?;
    report.set("benchmark_json", path.display().to_string());
    Ok(())
}
