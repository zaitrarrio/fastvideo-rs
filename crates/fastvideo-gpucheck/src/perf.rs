//! Performance probe (fit + projection, fail before the expensive run) and
//! the full clip run with per-step fail-fast guards and quality gates.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use fastvideo_cudarc::wan::pipeline::{mux_mp4, write_frames, PipelineError};
use fastvideo_cudarc::{CudaTensor, DenoiseStep, GenerateConfig, LoadParts, WanPipeline};
use serde::Serialize;
use serde_json::json;

use crate::gpu::PeakMem;
use crate::metrics::{mean_std, non_finite};
use crate::quality::{contact_sheet, video_stats, QualityGates};
use crate::report::{Report, StageError, StageResult};
use crate::st::{self, F32Tensor};

#[derive(Debug, Clone, Serialize)]
pub struct ClipSpec {
    pub height: usize,
    pub width: usize,
    pub frames: usize,
    pub steps: usize,
    pub guidance: f32,
    pub flow_shift: f64,
    pub dmd: bool,
    pub seed: u64,
    pub fps: u32,
}

impl ClipSpec {
    pub fn latent_frames(&self) -> usize {
        (self.frames - 1) / 4 + 1
    }

    /// DiT tokens per latent frame (patch 1×2×2 on the 8× VAE latent).
    pub fn tokens_per_frame(&self) -> usize {
        (self.height / 16) * (self.width / 16)
    }

    pub fn dmd_steps(&self) -> Vec<i32> {
        // FastWan 1.3B schedule, truncated/extended evenly to `steps`.
        let base = fastvideo_models::schedulers::FAST_WAN_1_3B_DMD_STEPS;
        if self.steps == base.len() {
            return base.to_vec();
        }
        (0..self.steps)
            .map(|i| 1000 - (i as i32 * 1000 / self.steps as i32))
            .collect()
    }

    pub fn batch(&self) -> usize {
        if (self.guidance - 1.0).abs() < 1e-6 {
            1
        } else {
            2
        }
    }

    pub fn generate_config(&self) -> GenerateConfig {
        GenerateConfig {
            height: self.height,
            width: self.width,
            num_frames: self.frames,
            num_inference_steps: self.steps,
            guidance_scale: self.guidance,
            flow_shift: self.flow_shift,
            is_dmd: self.dmd,
            dmd_steps: self.dmd.then(|| self.dmd_steps()),
            seed: self.seed,
            fps: self.fps,
            ..GenerateConfig::default()
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.frames % 4 == 1,
            "frames must be 4n+1 (got {})",
            self.frames
        );
        anyhow::ensure!(
            self.height % 16 == 0 && self.width % 16 == 0,
            "height/width must be multiples of 16 (got {}x{})",
            self.height,
            self.width
        );
        anyhow::ensure!(self.steps > 0, "steps must be > 0");
        Ok(())
    }
}

fn load_embeds(path: &Path) -> anyhow::Result<CudaTensor> {
    let mut m = st::load(path)?;
    let e = st::take(&mut m, "embeds", path)?;
    anyhow::ensure!(
        e.shape.len() == 3 && e.shape[0] == 2,
        "{}: embeds must be [2, text_len, dim] ([neg, prompt]), got {:?}",
        path.display(),
        e.shape
    );
    Ok(CudaTensor::from_vec(e.data, e.shape)?)
}

fn load_pipeline(report: &mut Report, weights: &Path) -> anyhow::Result<WanPipeline> {
    let t = Instant::now();
    let pipe = WanPipeline::load_with(
        weights,
        "wan_t2v_1_3b",
        LoadParts {
            text_encoder: false,
        },
    )?;
    report.note(
        "load",
        json!({"seconds": t.elapsed().as_secs_f64(), "mem_used_mib": used_mib()}),
    );
    Ok(pipe)
}

fn used_mib() -> Option<u64> {
    crate::gpu::mem_info().map(|(free, total)| (total - free) / (1 << 20))
}

/// Least-squares `y = a·x + b·x²` with both coefficients clamped ≥ 0.
fn fit_linear_quadratic(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let (mut s11, mut s12, mut s22, mut t1, mut t2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys) {
        let (f1, f2) = (x, x * x);
        s11 += f1 * f1;
        s12 += f1 * f2;
        s22 += f2 * f2;
        t1 += f1 * y;
        t2 += f2 * y;
    }
    let det = s11 * s22 - s12 * s12;
    if det.abs() > 1e-30 {
        let a = (t1 * s22 - t2 * s12) / det;
        let b = (s11 * t2 - s12 * t1) / det;
        if a >= 0.0 && b >= 0.0 {
            return (a, b);
        }
    }
    // Fall back to the better single-term fit.
    let a_only = if s11 > 0.0 { t1 / s11 } else { 0.0 };
    let b_only = if s22 > 0.0 { t2 / s22 } else { 0.0 };
    let err = |a: f64, b: f64| -> f64 {
        xs.iter()
            .zip(ys)
            .map(|(&x, &y)| (a * x + b * x * x - y).powi(2))
            .sum()
    };
    if err(a_only.max(0.0), 0.0) <= err(0.0, b_only.max(0.0)) {
        (a_only.max(0.0), 0.0)
    } else {
        (0.0, b_only.max(0.0))
    }
}

/// Least-squares `y = c + d·x`.
fn fit_affine(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let d = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (my - d * mx, d)
}

pub struct ProbeArgs<'a> {
    pub weights: &'a Path,
    pub embeds: &'a Path,
    pub device: &'a str,
    pub spec: ClipSpec,
    pub probe_latent_frames: Vec<usize>,
    pub budget_min: f64,
    pub vram_headroom: f64,
}

/// Measure DiT forward time/memory at several sequence lengths, fit, and
/// project the target clip. Exits with a budget error *before* the clip run
/// when the projection doesn't fit — the cheapest place to discover that a
/// kernel is too slow or an 8s clip won't fit in VRAM.
pub fn probe(report: &mut Report, args: ProbeArgs<'_>) -> StageResult<()> {
    let spec = &args.spec;
    spec.validate()?;
    let info = crate::gpu::init(args.device)?;
    report.set("device", &info);
    report.set("spec", spec);
    let pipe = load_pipeline(report, args.weights)?;
    let embeds = load_embeds(args.embeds)?;
    let (text_len, text_dim) = (embeds.shape[1], embeds.shape[2]);
    let b = spec.batch();
    let enc = if b == 2 {
        embeds.clone()
    } else {
        embeds.narrow(0, 1, 1)?
    };
    let (zh, zw) = (spec.height / 8, spec.width / 8);
    let tpf = spec.tokens_per_frame();

    let forward = |t_lat: usize| -> anyhow::Result<(f64, Option<u64>)> {
        let n = b * 16 * t_lat * zh * zw;
        let lat = CudaTensor::from_vec(
            crate::rand_weights::randn(7, n, 1.0),
            vec![b, 16, t_lat, zh, zw],
        )?;
        let ts = CudaTensor::from_vec(vec![999.0; b], vec![b])?;
        let enc_b =
            CudaTensor::from_vec(enc.host_cow()?.into_owned(), vec![b, text_len, text_dim])?;
        let mem = PeakMem::start();
        let t = Instant::now();
        let out = pipe.transformer().forward_ctx(&lat, &ts, &enc_b, None)?;
        // Download forces a sync, so the timing includes all queued kernels.
        let host = out.host_cow()?;
        let secs = t.elapsed().as_secs_f64();
        anyhow::ensure!(non_finite(&host) == 0, "non-finite DiT output at T={t_lat}");
        Ok((secs, mem.stop()))
    };

    let smallest = *args
        .probe_latent_frames
        .iter()
        .min()
        .context("no probe sizes")?;
    let (warm_s, _) = forward(smallest)?;
    report.note(
        "warmup",
        json!({"latent_frames": smallest, "seconds": warm_s}),
    );

    let budget_s = args.budget_min * 60.0;
    let (mut xs, mut ts, mut ms) = (Vec::new(), Vec::new(), Vec::new());
    for &t_lat in &args.probe_latent_frames {
        let (secs, peak) = forward(t_lat)?;
        let tokens = (t_lat * tpf) as f64;
        report.note(
            format!("forward_T{t_lat}"),
            json!({"tokens": tokens, "batch": b, "seconds": secs, "peak_mib": peak}),
        );
        xs.push(tokens);
        ts.push(secs);
        if let Some(p) = peak {
            ms.push(p as f64);
        }
        if secs * spec.steps as f64 > budget_s {
            return Err(StageError::Budget(format!(
                "a single probe forward at {tokens} tokens already takes {secs:.1}s; \
                 {} steps exceed the {:.0} min budget before reaching the target size",
                spec.steps, args.budget_min
            )));
        }
    }

    let target_tokens = (spec.latent_frames() * tpf) as f64;
    let (a, q) = fit_linear_quadratic(&xs, &ts);
    let fwd_target = a * target_tokens + q * target_tokens * target_tokens;
    let denoise_s = fwd_target * spec.steps as f64;

    // VAE: feat-cache decode is per latent frame, so time is ~linear in T.
    let vae_lat = 2usize;
    let z = CudaTensor::from_vec(
        crate::rand_weights::randn(11, 16 * vae_lat * zh * zw, 1.0),
        vec![1, 16, vae_lat, zh, zw],
    )?;
    let mem = PeakMem::start();
    let t = Instant::now();
    let video = pipe.decode_latents(&z)?;
    let _ = video.host_cow()?;
    let vae_probe_s = t.elapsed().as_secs_f64();
    let vae_peak = mem.stop();
    let vae_s = vae_probe_s / vae_lat as f64 * spec.latent_frames() as f64;
    report.note(
        "vae_T2",
        json!({"seconds": vae_probe_s, "peak_mib": vae_peak}),
    );

    let total_s = denoise_s + vae_s;
    let mut projection = json!({
        "target_tokens": target_tokens,
        "fit_seconds": {"linear": a, "quadratic": q},
        "dit_forward_s": fwd_target,
        "denoise_s": denoise_s,
        "vae_decode_s": vae_s,
        "total_s": total_s,
        "budget_s": budget_s,
    });
    if ms.len() == xs.len() && ms.len() >= 2 {
        let (c, d) = fit_affine(&xs, &ms);
        let dit_peak = c + d * target_tokens;
        let peak = dit_peak.max(vae_peak.unwrap_or(0) as f64);
        projection["peak_mib"] = json!(peak);
        projection["total_mib"] = json!(info.total_mib);
        if let Some(total) = info.total_mib {
            let limit = total as f64 * args.vram_headroom;
            report.set("projection", &projection);
            if peak > limit {
                return Err(StageError::Budget(format!(
                    "projected peak VRAM {peak:.0} MiB > {limit:.0} MiB ({:.0}% of {total} MiB)",
                    args.vram_headroom * 100.0
                )));
            }
        }
    }
    report.set("projection", &projection);
    if total_s > budget_s {
        return Err(StageError::Budget(format!(
            "projected clip time {:.1} min (denoise {:.1} + vae {:.1}) > budget {:.0} min",
            total_s / 60.0,
            denoise_s / 60.0,
            vae_s / 60.0,
            args.budget_min
        )));
    }
    Ok(())
}

pub struct ClipArgs<'a> {
    pub weights: &'a Path,
    pub embeds: &'a Path,
    pub device: &'a str,
    pub spec: ClipSpec,
    pub out_dir: PathBuf,
    pub budget_min: f64,
    pub gates: QualityGates,
    pub save_mp4: bool,
}

pub fn clip(report: &mut Report, args: ClipArgs<'_>) -> StageResult<()> {
    let spec = &args.spec;
    spec.validate()?;
    report.set("device", crate::gpu::init(args.device)?);
    report.set("spec", spec);
    report.set("gates", args.gates);
    let run_timer = Instant::now();
    let pipe = load_pipeline(report, args.weights)?;
    let embeds = load_embeds(args.embeds)?;
    let cfg = spec.generate_config();
    let noise = pipe.initial_latents(&cfg)?;
    let budget_s = args.budget_min * 60.0;

    #[derive(Default)]
    struct Trace {
        steps: Vec<serde_json::Value>,
        /// Latents after step 1: every precision path starts that step from
        /// the same seeded noise, so `compare` isolates one-forward error.
        first: Option<Vec<f32>>,
        abort: Option<StageError>,
    }
    let trace = RefCell::new(Trace::default());
    let denoise_timer = Instant::now();
    let mut last = Instant::now();
    let mem = PeakMem::start();
    let mut observer = |s: &DenoiseStep<'_>| -> fastvideo_cudarc::wan::pipeline::Result<()> {
        let host = s.latents.host_cow()?;
        let step_s = last.elapsed().as_secs_f64();
        last = Instant::now();
        let bad = non_finite(&host);
        let (mean, std) = mean_std(&host);
        eprintln!(
            "step {}/{} t={:.0} {:.1}s latent mean={mean:.4} std={std:.4}",
            s.index + 1,
            s.total,
            s.timestep,
            step_s
        );
        let mut tr = trace.borrow_mut();
        tr.steps.push(json!({"t": s.timestep, "seconds": step_s, "mean": mean, "std": std, "non_finite": bad}));
        if s.index == 0 {
            tr.first = Some(host.into_owned());
        }
        if bad > 0 {
            let msg = format!("{bad} non-finite latents after step {}", s.index + 1);
            tr.abort = Some(StageError::Check(msg.clone()));
            return Err(PipelineError::Message(msg));
        }
        let elapsed = denoise_timer.elapsed().as_secs_f64();
        let projected = elapsed / (s.index + 1) as f64 * s.total as f64;
        if projected > budget_s {
            let msg = format!(
                "projected denoise {:.1} min after {} step(s) > budget {:.0} min",
                projected / 60.0,
                s.index + 1,
                args.budget_min
            );
            tr.abort = Some(StageError::Budget(msg.clone()));
            return Err(PipelineError::Message(msg));
        }
        Ok(())
    };
    let denoised = pipe.denoise(&cfg, noise, &embeds, Some(&mut observer));
    drop(observer);
    let denoise_peak = mem.stop();
    let denoise_s = denoise_timer.elapsed().as_secs_f64();
    let mut trace = trace.into_inner();
    report.set("steps", &trace.steps);
    let first = trace.first.take();
    let latents = match (denoised, trace.abort) {
        (_, Some(abort)) => return Err(abort),
        (Err(e), None) => return Err(StageError::Error(e.into())),
        (Ok(l), None) => l,
    };

    let clip_dir = args.out_dir;
    std::fs::create_dir_all(&clip_dir)?;
    let lat_host = F32Tensor::new(latents.shape.clone(), latents.host_cow()?.into_owned())?;
    let mut saved = vec![("latents", &lat_host)];
    let first_host = first
        .map(|v| F32Tensor::new(latents.shape.clone(), v))
        .transpose()?;
    if let Some(f) = &first_host {
        saved.push(("step1", f));
    }
    st::save(&clip_dir.join("latents.safetensors"), &saved)?;

    let mem = PeakMem::start();
    let t = Instant::now();
    let video = pipe.decode_latents(&latents)?;
    let video_host = video.host_cow()?.into_owned();
    let vae_s = t.elapsed().as_secs_f64();
    let vae_peak = mem.stop();

    let t = Instant::now();
    let frames_dir = clip_dir.join("frames");
    let paths = write_frames(&video, &frames_dir)?;
    let mp4 = if args.save_mp4 {
        mux_mp4(&frames_dir, spec.fps).map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        String::new()
    };
    contact_sheet(
        &video_host,
        &video.shape,
        &clip_dir.join("contact_sheet.png"),
    )?;
    let write_s = t.elapsed().as_secs_f64();

    let total_s = run_timer.elapsed().as_secs_f64();
    let seconds_of_video = spec.frames as f64 / f64::from(spec.fps);
    report.set(
        "timings",
        json!({
            "denoise_s": denoise_s,
            "per_step_s": denoise_s / spec.steps as f64,
            "vae_decode_s": vae_s,
            "write_s": write_s,
            "total_s": total_s,
            "video_seconds": seconds_of_video,
            "seconds_per_video_second": total_s / seconds_of_video,
            "peak_mib": {"denoise": denoise_peak, "vae": vae_peak},
        }),
    );
    // Recorded, not just printed: kernel launches are the number that decides
    // whether the remaining gap to upstream is launch overhead or arithmetic,
    // and "how many launches per step" is only answerable from a real clip.
    {
        let st = fastvideo_cudarc::wan::stats::snapshot();
        report.set(
            "device_stats",
            json!({
                "kernel_launches": st.launches,
                "launches_per_step": st.launches as f64 / spec.steps as f64,
                "h2d_count": st.h2d_count,
                "d2h_count": st.d2h_count,
                "h2d_mib": st.h2d_bytes >> 20,
                "d2h_mib": st.d2h_bytes >> 20,
                "host_fallbacks": st.host_fallbacks,
            }),
        );
        // Empty unless FASTVIDEO_PROFILE=1: phase timing synchronizes, so a
        // profiled run is deliberately not the run we quote timings from.
        let phases = fastvideo_cudarc::wan::stats::phase_report();
        if !phases.is_empty() {
            let total: f64 = phases.iter().map(|(_, _, s)| s).sum();
            report.set(
                "dit_phases",
                json!({
                    "total_s": total,
                    "by_phase": phases
                        .iter()
                        .map(|(n, c, s)| json!({"phase": n, "calls": c, "seconds": s,
                                                "pct": if total > 0.0 { 100.0 * s / total } else { 0.0 }}))
                        .collect::<Vec<_>>(),
                }),
            );
        }
    }
    report.set(
        "artifacts",
        json!({"frames": paths.len(), "mp4": mp4, "dir": clip_dir}),
    );

    report.check(
        "frame_count",
        paths.len() == spec.frames,
        json!({"frames": paths.len()}),
        json!({"expected": spec.frames}),
    )?;
    let stats = video_stats(&video_host, &video.shape)?;
    let failures = stats.failures(&args.gates);
    report.check(
        "video_quality",
        failures.is_empty(),
        json!({"stats": stats, "failures": failures}),
        serde_json::to_value(args.gates)?,
    )?;
    if total_s > budget_s * 1.25 {
        return Err(StageError::Budget(format!(
            "clip took {:.1} min, over 125% of the {:.0} min budget",
            total_s / 60.0,
            args.budget_min
        )));
    }
    Ok(())
}

/// Two clip runs of the same seed and prompt under different precision paths.
///
/// Step 1 starts from identical noise in both runs, so its latents bound the
/// single-forward error of path `b` and are gated tightly. Later steps feed
/// that error back through the model, and the trajectories drift apart
/// (bf16 vs FP32 on the 2s DMD clip: rel_l2 0.23, 23 dB, same content), so
/// the final latents and frames get sanity limits only.
pub fn compare(report: &mut Report, a: &Path, b: &Path, gates: CompareGates) -> StageResult<()> {
    let load = |dir: &Path| -> anyhow::Result<std::collections::HashMap<String, F32Tensor>> {
        st::load(&dir.join("latents.safetensors"))
    };
    let (mut ma, mut mb) = (load(a)?, load(b)?);
    let (la, lb) = (
        st::take(&mut ma, "latents", &a.join("latents.safetensors"))?,
        st::take(&mut mb, "latents", &b.join("latents.safetensors"))?,
    );
    let step1 = match (ma.remove("step1"), mb.remove("step1")) {
        (Some(x), Some(y)) => Some(crate::metrics::diff(&y.data, &x.data)),
        _ => None,
    };
    let d = crate::metrics::diff(&lb.data, &la.data);

    let frames = |dir: &Path| -> anyhow::Result<Vec<PathBuf>> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir.join("frames"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "png"))
            .collect();
        v.sort();
        Ok(v)
    };
    let (fa, fb) = (frames(a)?, frames(b)?);
    let (mut mse, mut n) = (0.0f64, 0usize);
    for (pa, pb) in fa.iter().zip(&fb) {
        let ia = image::open(pa)?.into_rgb8();
        let ib = image::open(pb)?.into_rgb8();
        if ia.dimensions() != ib.dimensions() {
            return Err(StageError::Check(format!(
                "frame size mismatch {}",
                pa.display()
            )));
        }
        for (x, y) in ia.as_raw().iter().zip(ib.as_raw()) {
            let d = f64::from(*x) - f64::from(*y);
            mse += d * d;
        }
        n += ia.as_raw().len();
    }
    mse /= n.max(1) as f64;
    let p = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };

    // Every metric lands in the report before the first gate can stop the stage.
    report.set(
        "metrics",
        json!({
            "step1": step1.as_ref().map(|s| s.to_json()),
            "latents": d.to_json(),
            "frames_psnr_db": crate::metrics::jf(p),
        }),
    );
    match &step1 {
        Some(s) => report.check(
            "step1_latents",
            s.within(gates.max_step1_rel),
            s.to_json(),
            json!({"rel_l2": gates.max_step1_rel}),
        )?,
        None => {
            return Err(StageError::Check(
                "clip dirs have no step1 latents; rerun the clip stages".into(),
            ))
        }
    }
    report.check(
        "latents",
        d.within(gates.max_latent_rel),
        d.to_json(),
        json!({"rel_l2": gates.max_latent_rel}),
    )?;
    report.check(
        "frame_count",
        fa.len() == fb.len() && !fa.is_empty(),
        json!({"a": fa.len(), "b": fb.len()}),
        json!({}),
    )?;
    report.check(
        "frames_psnr",
        p >= gates.min_psnr,
        json!({"psnr_db": crate::metrics::jf(p)}),
        json!({"psnr_db_min": gates.min_psnr}),
    )?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct CompareGates {
    pub max_step1_rel: f64,
    pub max_latent_rel: f64,
    pub min_psnr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_quadratic() {
        let xs = [1000.0, 3000.0, 9000.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2e-4 * x + 3e-8 * x * x).collect();
        let (a, b) = fit_linear_quadratic(&xs, &ys);
        assert!(
            (a - 2e-4).abs() < 1e-9 && (b - 3e-8).abs() < 1e-12,
            "{a} {b}"
        );
    }

    #[test]
    fn fit_affine_recovers_line() {
        let (c, d) = fit_affine(&[1.0, 2.0, 4.0], &[5.0, 7.0, 11.0]);
        assert!((c - 3.0).abs() < 1e-9 && (d - 2.0).abs() < 1e-9);
    }

    #[test]
    fn eight_second_clip_shape() {
        let s = ClipSpec {
            height: 480,
            width: 832,
            frames: 129,
            steps: 3,
            guidance: 1.0,
            flow_shift: 8.0,
            dmd: true,
            seed: 0,
            fps: 16,
        };
        s.validate().unwrap();
        assert_eq!(s.latent_frames(), 33);
        assert_eq!(s.tokens_per_frame(), 1560);
        assert_eq!(s.dmd_steps(), vec![1000, 757, 522]);
    }
}
