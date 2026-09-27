//! MMAudio stages. `v2a` runs our MMAudio port on a finished clip the way
//! strobe's sidecar runs upstream MMAudio on it (duration = `--duration`,
//! default frames / fps; 25 Euler steps, CFG 4.5, negative prompt ""), writes
//! the waveform and a muxed mp4, and times `--runs` generations (the first
//! is the warm-up when `--runs` > 1). With `FASTVIDEO_DUMP_DIR` /
//! `FASTVIDEO_INJECT_DIR` it is the oracle side (docs/oracle.md). `t2a` is
//! text-to-audio (no video features), one wav per `--seeds` entry from one
//! loaded pipeline.

use std::path::PathBuf;

use clap::Subcommand;
use fastvideo_cudarc::mmaudio::pipeline::write_wav;
use fastvideo_cudarc::mmaudio::{MmAudioPipeline, MmAudioRequest, VideoFrames};
use fastvideo_models::mmaudio::MmAudioPreset;
use serde_json::json;

use crate::report::{Report, StageResult};

#[derive(Subcommand)]
pub enum Stage {
    V2a {
        /// Weight root (`scripts/gpu/fetch-mmaudio.py` layout).
        #[arg(long, default_value = "/workspace/weights/mmaudio-44k-v2")]
        weights: PathBuf,
        #[arg(long)]
        video: PathBuf,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value = "")]
        negative: String,
        /// Seconds (default: frame count / frame rate).
        #[arg(long)]
        duration: Option<f64>,
        #[arg(long, default_value_t = 1000)]
        seed: u64,
        #[arg(long, default_value_t = 25)]
        steps: usize,
        #[arg(long, default_value_t = 4.5)]
        cfg: f32,
        #[arg(long, default_value_t = 1)]
        runs: usize,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
    T2a {
        /// Weight root (`scripts/gpu/fetch-mmaudio.py` layout).
        #[arg(long, default_value = "/workspace/weights/mmaudio-44k-v2")]
        weights: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "")]
        negative: String,
        /// Seconds (the preset's training length, 8 s, by default).
        #[arg(long)]
        duration: Option<f64>,
        /// Comma-separated seeds; each writes `<out>/seed-<seed>.wav`.
        #[arg(long, default_value = "1000", value_delimiter = ',')]
        seeds: Vec<u64>,
        #[arg(long, default_value_t = 25)]
        steps: usize,
        #[arg(long, default_value_t = 4.5)]
        cfg: f32,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
}

fn run_t2a(report: &mut Report, stage: &Stage) -> StageResult<()> {
    let Stage::T2a {
        weights,
        prompt,
        negative,
        duration,
        seeds,
        steps,
        cfg,
        out,
        device,
    } = stage
    else {
        unreachable!("t2a stage")
    };
    report.set("device", crate::gpu::init(device)?);
    std::fs::create_dir_all(out).map_err(anyhow::Error::from)?;
    let t = std::time::Instant::now();
    let pipe = MmAudioPipeline::load(weights, MmAudioPreset::Large44kV2)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", weights.display()))?;
    let load_s = t.elapsed().as_secs_f64();
    let mut runs = Vec::new();
    for &seed in seeds {
        let mut req = MmAudioRequest::t2a(prompt.clone(), seed);
        req.negative_prompt = negative.clone();
        req.num_steps = *steps;
        req.cfg_strength = *cfg;
        if let Some(d) = duration {
            req.duration_s = *d;
        }
        let t = std::time::Instant::now();
        let o = pipe
            .generate(&req)
            .map_err(|e| anyhow::anyhow!("generate: {e}"))?;
        let wall = t.elapsed().as_secs_f64();
        let wav = out.join(format!("seed-{seed}.wav"));
        write_wav(&wav, &o.waveform, o.sample_rate).map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!(
            "mmaudio t2a seed {seed}: {wall:.3}s duration {:.3}s -> {}",
            o.duration_s,
            wav.display()
        );
        runs.push(json!({"seed": seed, "wall_s": wall, "duration_s": o.duration_s, "sample_rate": o.sample_rate, "wav": wav}));
    }
    let doc = json!({
        "pipeline": "MMAudioT2A (fastvideo-rs cudarc)",
        "variant": "large_44k_v2",
        "prompt": prompt,
        "load_s": load_s,
        "runs": runs,
    });
    std::fs::write(
        out.join("mmaudio.json"),
        serde_json::to_string_pretty(&doc).unwrap_or_default(),
    )
    .map_err(anyhow::Error::from)?;
    report.set("mmaudio", doc);
    Ok(())
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    if matches!(stage, Stage::T2a { .. }) {
        return run_t2a(report, stage);
    }
    let Stage::V2a {
        weights,
        video,
        prompt,
        negative,
        duration,
        seed,
        steps,
        cfg,
        runs,
        out,
        device,
    } = stage
    else {
        unreachable!("v2a stage")
    };
    report.set("device", crate::gpu::init(device)?);
    std::fs::create_dir_all(out).map_err(anyhow::Error::from)?;
    let t = std::time::Instant::now();
    let pipe = MmAudioPipeline::load(weights, MmAudioPreset::Large44kV2)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", weights.display()))?;
    let load_s = t.elapsed().as_secs_f64();
    let frames = VideoFrames::from_file(video).map_err(|e| anyhow::anyhow!("{e}"))?;
    let fps = if frames.times.len() > 1 { 1.0 / frames.times[1] } else { 16.0 };
    let clip_s = duration.unwrap_or(frames.times.len() as f64 / fps);
    let req = MmAudioRequest {
        prompt: prompt.clone(),
        negative_prompt: negative.clone(),
        seed: *seed,
        duration_s: clip_s,
        num_steps: *steps,
        cfg_strength: *cfg,
        video: Some(frames),
    };
    let mut timings = Vec::new();
    let mut last = None;
    let mem = crate::gpu::PeakMem::start();
    for i in 0..(*runs).max(1) {
        let t = std::time::Instant::now();
        let o = pipe.generate(&req).map_err(|e| anyhow::anyhow!("generate: {e}"))?;
        let wall = t.elapsed().as_secs_f64();
        let tm = &o.timings;
        eprintln!(
            "mmaudio v2a run {i}: {wall:.3}s (pre {:.3} clip {:.3} sync {:.3} text {:.3} dit {:.3} vae {:.3} voc {:.3}) duration {:.3}s rtf {:.3}",
            tm.preprocess_s, tm.clip_s, tm.sync_s, tm.text_s, tm.dit_s, tm.vae_s, tm.vocoder_s, o.duration_s, wall / clip_s
        );
        timings.push(json!({
            "wall_s": wall, "preprocess_s": tm.preprocess_s, "clip_s": tm.clip_s, "sync_s": tm.sync_s,
            "text_s": tm.text_s, "dit_s": tm.dit_s, "vae_s": tm.vae_s, "vocoder_s": tm.vocoder_s,
        }));
        last = Some((o, wall));
    }
    let peak_mib = mem.stop();
    let (o, wall) = last.expect("one run");
    let wav = out.join("mmaudio.wav");
    write_wav(&wav, &o.waveform, o.sample_rate).map_err(|e| anyhow::anyhow!("{e}"))?;
    let raw = out.join("mmaudio.f32");
    let bytes: Vec<u8> = o.waveform.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&raw, bytes).map_err(anyhow::Error::from)?;
    let muxed = out.join("sidecar.mp4");
    std::fs::copy(video, &muxed).map_err(anyhow::Error::from)?;
    fastvideo_cudarc::mmaudio::sidecar::mux(&muxed, &wav).map_err(|e| anyhow::anyhow!("{e}"))?;
    let doc = json!({
        "pipeline": "MMAudioV2A (fastvideo-rs cudarc)",
        "variant": "large_44k_v2",
        "clip_s": clip_s,
        "duration_s": o.duration_s,
        "samples": o.waveform.len(),
        "sample_rate": o.sample_rate,
        "audio_s": wall,
        "audio_rtf": wall / clip_s,
        "load_s": load_s,
        "runs": timings,
        "peak_mib": peak_mib,
        "wav": wav, "muxed_path": muxed,
    });
    std::fs::write(out.join("mmaudio.json"), serde_json::to_string_pretty(&doc).unwrap_or_default())
        .map_err(anyhow::Error::from)?;
    report.set("mmaudio", doc);
    Ok(())
}
