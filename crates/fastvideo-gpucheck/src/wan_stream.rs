//! `wan stream`: the open-ended causal SF-Wan rollout (`wan::stream`,
//! serve work package E6) on real weights.
//!
//! One resident pipeline, then:
//!
//! * `--parity`: the bounded 81-frame causal generation (`denoise`, whole-clip
//!   TAEHV decode) against seven streamed blocks with the bounded path's
//!   settings (absolute RoPE, 21-frame window, no sink): latents, and the
//!   per-block carried-state decode against the whole-clip decode of the same
//!   latents (max abs, PSNR);
//! * every `--run name,seconds=S[,rope=rel|abs][,sink=N][,window=N]
//!   [,switch_at=S][,switch=keep|reset][,drop_rgb=1]`: a rollout for `S`
//!   seconds of video at `--fps`, blocks handed through a depth-4 channel to
//!   a consumer thread (the design's executor → pacer hand-off) that measures
//!   picture statistics per window of video time. Reports time to first
//!   frame, per-block latency, steady-state frames per second, device memory
//!   and KV bytes over time, and drift indicators.

use std::sync::mpsc::sync_channel;
use std::time::Instant;

use anyhow::{anyhow, bail};
use fastvideo_cudarc::wan::stream::{
    CausalRollout, HostBlock, PromptSwitch, RolloutConfig, RopePolicy,
};
use fastvideo_cudarc::wan::tensor::CudaTensor;
use fastvideo_cudarc::{GenerateConfig, LoadParts, WanPipeline};
use serde_json::{json, Value};

use crate::report::{Report, StageResult};

pub struct Args<'a> {
    pub weights: &'a std::path::Path,
    pub preset: &'a str,
    pub prompt: &'a str,
    pub switch_prompt: &'a str,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub fps: u32,
    pub parity: bool,
    pub runs: &'a [String],
    pub window_s: f64,
    pub device: &'a str,
}

#[derive(Debug, Clone)]
struct RunSpec {
    name: String,
    seconds: f64,
    rope: RopePolicy,
    sink: usize,
    window: usize,
    switch_at: Option<f64>,
    switch: PromptSwitch,
    drop_rgb: bool,
}

fn parse_run(s: &str) -> anyhow::Result<RunSpec> {
    let mut parts = s.split(',');
    let name = parts.next().filter(|n| !n.is_empty()).ok_or_else(|| anyhow!("--run {s}: no name"))?;
    let mut r = RunSpec {
        name: name.to_string(),
        seconds: 10.0,
        rope: RopePolicy::Relativistic,
        sink: 3,
        window: 21,
        switch_at: None,
        switch: PromptSwitch::Keep,
        drop_rgb: false,
    };
    for kv in parts {
        let (k, v) = kv.split_once('=').ok_or_else(|| anyhow!("--run {s}: `{kv}` is not key=value"))?;
        match k {
            "seconds" => r.seconds = v.parse()?,
            "rope" => {
                r.rope = match v {
                    "rel" | "relativistic" => RopePolicy::Relativistic,
                    "abs" | "absolute" => RopePolicy::Absolute,
                    _ => bail!("--run {s}: rope={v}"),
                }
            }
            "sink" => r.sink = v.parse()?,
            "window" => r.window = v.parse()?,
            "switch_at" => r.switch_at = Some(v.parse()?),
            "switch" => {
                r.switch = match v {
                    "keep" => PromptSwitch::Keep,
                    "reset" => PromptSwitch::Reset,
                    _ => bail!("--run {s}: switch={v}"),
                }
            }
            "drop_rgb" => r.drop_rgb = v == "1",
            _ => bail!("--run {s}: unknown key {k}"),
        }
    }
    Ok(r)
}

fn host(t: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.host_cow().map_err(|e| anyhow!("{e}"))?.into_owned())
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[((v.len() - 1) as f64 * p).round() as usize]
}

pub fn run(report: &mut Report, a: &Args<'_>) -> StageResult<()> {
    report.set("device", crate::gpu::init(a.device)?);
    let runs: Vec<RunSpec> = a.runs.iter().map(|s| parse_run(s)).collect::<anyhow::Result<_>>()?;
    let tokenizer = a.weights.join("tokenizer").join("tokenizer.json");
    if !tokenizer.is_file() {
        return Err(anyhow!("{} missing", tokenizer.display()).into());
    }
    let tokenizer = tokenizer.to_string_lossy().into_owned();
    let t = Instant::now();
    let pipe = WanPipeline::load_with(a.weights, a.preset, LoadParts { text_encoder: true })
        .map_err(|e| anyhow!("load {}: {e}", a.weights.display()))?;
    report.note(
        "load",
        json!({"seconds": t.elapsed().as_secs_f64(),
               "mem_used_mib": crate::gpu::mem_info().map(|(f, t)| (t - f) >> 20)}),
    );
    let base = RolloutConfig {
        prompt: a.prompt.to_string(),
        height: a.height,
        width: a.width,
        seed: a.seed,
        tokenizer_path: Some(tokenizer.clone()),
        ..RolloutConfig::default()
    };
    if a.parity {
        parity(report, &pipe, &base)?;
    }
    for r in &runs {
        one_run(report, &pipe, &base, r, a)?;
    }
    Ok(())
}

/// Bounded 81-frame causal path against seven streamed blocks.
fn parity(report: &mut Report, pipe: &WanPipeline, base: &RolloutConfig) -> StageResult<()> {
    let tae = pipe.taehv().ok_or_else(|| anyhow!("no TAEHV loaded"))?;
    let cfg = GenerateConfig {
        prompt: base.prompt.clone(),
        seed: base.seed,
        height: base.height,
        width: base.width,
        num_frames: 81,
        num_inference_steps: base.dmd_steps.len(),
        guidance_scale: 1.0,
        is_dmd: true,
        dmd_steps: Some(base.dmd_steps.clone()),
        flow_shift: base.flow_shift,
        tokenizer_path: base.tokenizer_path.clone(),
        ..GenerateConfig::default()
    };
    let t = Instant::now();
    let emb = pipe.encode_prompt(&cfg).map_err(|e| anyhow!("{e}"))?;
    let lat0 = pipe.initial_latents(&cfg).map_err(|e| anyhow!("{e}"))?;
    let bounded = pipe.denoise(&cfg, lat0, &emb, None).map_err(|e| anyhow!("{e}"))?;
    let whole = tae.decode(&bounded).map_err(|e| anyhow!("{e}"))?;
    let bounded_s = t.elapsed().as_secs_f64();
    // [1, 3, F, H, W] → [F, 3, H, W]
    let (f, h, w) = (whole.shape[2], whole.shape[3], whole.shape[4]);
    let whole_f = whole
        .reshape(vec![3, f, h, w])
        .and_then(|x| x.permute(&[1, 0, 2, 3]))
        .map_err(|e| anyhow!("{e}"))?;
    let whole_h = host(&whole_f)?;
    let bounded_h = host(&bounded)?;

    let mut arms = serde_json::Map::new();
    for (name, rope) in [("absolute", RopePolicy::Absolute), ("relativistic", RopePolicy::Relativistic)] {
        let cfg = RolloutConfig {
            rope,
            sink_frames: 0,
            local_attn_frames: 21,
            rgb8: false,
            ..base.clone()
        };
        let mut ro = CausalRollout::open(pipe, cfg).map_err(|e| anyhow!("{e}"))?;
        let (mut lats, mut frames, mut sizes) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..7 {
            let b = ro.next_block().map_err(|e| anyhow!("{e}"))?;
            sizes.push(b.num_frames());
            lats.push(b.latents);
            frames.push(b.frames);
        }
        drop(ro);
        let lat = CudaTensor::cat(&lats.iter().collect::<Vec<_>>(), 2).map_err(|e| anyhow!("{e}"))?;
        let fr = CudaTensor::cat(&frames.iter().collect::<Vec<_>>(), 0).map_err(|e| anyhow!("{e}"))?;
        let (lat_h, fr_h) = (host(&lat)?, host(&fr)?);
        let lat_d = crate::metrics::diff(&lat_h, &bounded_h);
        // Per-block decode of the stream's own latents against their
        // whole-clip decode: the carried-state question, independent of the
        // denoiser.
        let own_whole = tae.decode(&lat).map_err(|e| anyhow!("{e}"))?;
        let own_whole = own_whole
            .reshape(vec![3, f, h, w])
            .and_then(|x| x.permute(&[1, 0, 2, 3]))
            .map_err(|e| anyhow!("{e}"))?;
        let own_h = host(&own_whole)?;
        let dec_d = crate::metrics::diff(&fr_h, &own_h);
        let dec_psnr = crate::metrics::psnr(&fr_h, &own_h, 2.0);
        let vs_bounded = crate::metrics::diff(&fr_h, &whole_h);
        let vs_bounded_psnr = crate::metrics::psnr(&fr_h, &whole_h, 2.0);
        let bitwise_latents = lat_h == bounded_h;
        // The same whole-clip decode in 3-latent chunks: the per-block path
        // then runs the same convolutions on the same frame counts, so any
        // remaining difference would be the carried state itself.
        let chunk3 = tae
            .decode_streaming_chunked(&lat, 3, &mut |_, _| Ok(()))
            .and_then(|x| x.reshape(vec![3, f, h, w]))
            .and_then(|x| x.permute(&[1, 0, 2, 3]))
            .map_err(|e| anyhow!("{e}"))?;
        let chunk3_h = host(&chunk3)?;
        let c3 = crate::metrics::diff(&fr_h, &chunk3_h);
        // 8-bit frames as the stream emits them.
        let q = |v: &[f32]| -> Vec<u8> { v.iter().map(|x| ((x + 1.0) * 127.5).clamp(0.0, 255.0) as u8).collect() };
        let (q_fr, q_own) = (q(&fr_h), q(&own_h));
        let u8_diff = q_fr.iter().zip(&q_own).filter(|(a, b)| a != b).count();
        let u8_max = q_fr.iter().zip(&q_own).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        arms.insert(
            name.into(),
            json!({
                "block_frames": sizes,
                "latents_vs_bounded": lat_d.to_json(),
                "latents_bitwise_equal": bitwise_latents,
                "per_block_vs_whole_clip_decode": {"max_abs": dec_d.max_abs, "psnr_db": dec_psnr,
                    "rel_l2": dec_d.rel_l2, "bitwise_equal": fr_h == own_h},
                "frames_vs_bounded_clip": {"max_abs": vs_bounded.max_abs, "psnr_db": vs_bounded_psnr},
                "per_block_vs_whole_clip_chunk3": {"max_abs": c3.max_abs, "bitwise_equal": fr_h == chunk3_h},
                "rgb8_vs_whole_clip": {"differing_values": u8_diff, "of": q_fr.len(), "max_levels": u8_max},
            }),
        );
        report.note(format!("parity/{name}"), arms[name].clone());
        if rope == RopePolicy::Absolute {
            report.check(
                "parity/absolute_stream_is_the_bounded_path",
                lat_d.max_abs <= 1e-3 && fr_h.len() == whole_h.len(),
                json!({"latent_max_abs": lat_d.max_abs, "bitwise": bitwise_latents, "frames": sizes.iter().sum::<usize>()}),
                json!({"latent_max_abs": 1e-3, "frames": 81}),
            )?;
        }
        report.check(
            format!("parity/{name}_per_block_decode"),
            dec_d.max_abs <= 0.02 && dec_psnr >= 60.0,
            json!({"max_abs": dec_d.max_abs, "psnr_db": dec_psnr, "chunk3_max_abs": c3.max_abs}),
            json!({"max_abs": 0.02, "psnr_db_min": 60.0}),
        )?;
    }
    report.note("parity", json!({"bounded_s": bounded_s, "arms": Value::Object(arms)}));
    Ok(())
}

/// Picture statistics of one block (RGB8, every other pixel).
#[derive(Default, Clone, Copy)]
struct Pic {
    luma: f64,
    std: f64,
    mad: f64,
    seam_mad: f64,
    sharp: f64,
    clipped: f64,
}

fn frame_luma(rgb: &[u8], h: usize, w: usize, stride: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((h / stride) * (w / stride));
    for y in (0..h).step_by(stride) {
        for x in (0..w).step_by(stride) {
            let p = (y * w + x) * 3;
            out.push(0.299 * f32::from(rgb[p]) + 0.587 * f32::from(rgb[p + 1]) + 0.114 * f32::from(rgb[p + 2]));
        }
    }
    out
}

fn block_pic(b: &HostBlock, prev_last: &mut Option<Vec<f32>>) -> Pic {
    let (h, w) = (b.height, b.width);
    let plane = h * w * 3;
    let (sh, sw) = (h.div_ceil(2), w.div_ceil(2));
    let lumas: Vec<Vec<f32>> = (0..b.frames)
        .map(|i| frame_luma(&b.rgb[i * plane..(i + 1) * plane], h, w, 2))
        .collect();
    let mut p = Pic::default();
    let n = lumas.len().max(1) as f64;
    for l in &lumas {
        let m = l.iter().map(|&v| f64::from(v)).sum::<f64>() / l.len() as f64;
        let var = l.iter().map(|&v| (f64::from(v) - m).powi(2)).sum::<f64>() / l.len() as f64;
        p.luma += m / n;
        p.std += var.sqrt() / n;
        p.clipped += l.iter().filter(|&&v| !(3.0..=252.0).contains(&v)).count() as f64 / l.len() as f64 / n;
        let mut g = 0.0f64;
        for y in 0..sh - 1 {
            for x in 0..sw - 1 {
                let c = l[y * sw + x];
                g += f64::from((l[y * sw + x + 1] - c).abs() + (l[(y + 1) * sw + x] - c).abs());
            }
        }
        p.sharp += g / ((sh - 1) * (sw - 1)) as f64 / n;
    }
    let mad = |a: &[f32], b: &[f32]| {
        a.iter().zip(b).map(|(x, y)| f64::from((x - y).abs())).sum::<f64>() / a.len() as f64
    };
    let pairs = lumas.len().saturating_sub(1).max(1) as f64;
    for i in 1..lumas.len() {
        p.mad += mad(&lumas[i], &lumas[i - 1]) / pairs;
    }
    if let (Some(prev), Some(first)) = (prev_last.as_ref(), lumas.first()) {
        p.seam_mad = mad(prev, first);
    }
    *prev_last = lumas.last().cloned();
    p
}

fn one_run(
    report: &mut Report,
    pipe: &WanPipeline,
    base: &RolloutConfig,
    r: &RunSpec,
    a: &Args<'_>,
) -> StageResult<()> {
    let fps = f64::from(a.fps.max(1));
    let want_frames = (r.seconds * fps).ceil() as usize;
    let cfg = RolloutConfig {
        rope: r.rope,
        sink_frames: r.sink,
        local_attn_frames: r.window,
        prompt_switch: r.switch,
        rgb8: true,
        ..base.clone()
    };
    let (tx, rx) = sync_channel::<(HostBlock, Option<u64>, usize, f64)>(4);
    let window_s = a.window_s;
    // The consumer: picture statistics per window of video time.
    let consumer = std::thread::spawn(move || {
        let mut prev_last = None;
        let mut windows: Vec<Value> = Vec::new();
        let mut acc: Vec<(Pic, f64)> = Vec::new();
        let (mut mem, mut kv) = (None, 0usize);
        let mut win_start = 0.0f64;
        let mut first_rx = None;
        let mut frames = 0usize;
        let flush = |acc: &mut Vec<(Pic, f64)>, windows: &mut Vec<Value>, t0: f64, mem: Option<u64>, kv: usize| {
            if acc.is_empty() {
                return;
            }
            let wsum: f64 = acc.iter().map(|(_, w)| w).sum();
            let avg = |f: fn(&Pic) -> f64| acc.iter().map(|(p, w)| f(p) * w).sum::<f64>() / wsum;
            let seams: Vec<f64> = acc.iter().map(|(p, _)| p.seam_mad).filter(|&s| s > 0.0).collect();
            windows.push(json!({
                "t0_s": t0, "blocks": acc.len(),
                "luma": avg(|p| p.luma), "std": avg(|p| p.std), "mad": avg(|p| p.mad),
                "seam_mad": if seams.is_empty() { 0.0 } else { seams.iter().sum::<f64>() / seams.len() as f64 },
                "sharpness": avg(|p| p.sharp), "clipped": avg(|p| p.clipped),
                "mem_used_mib": mem, "kv_mib": kv >> 20,
            }));
            acc.clear();
        };
        for (b, m, k, _) in rx {
            first_rx.get_or_insert_with(Instant::now);
            let t = b.first_frame as f64 / fps;
            if t >= win_start + window_s {
                flush(&mut acc, &mut windows, win_start, mem, kv);
                win_start = (t / window_s).floor() * window_s;
            }
            let p = if b.rgb.is_empty() { Pic::default() } else { block_pic(&b, &mut prev_last) };
            frames += b.frames;
            acc.push((p, b.frames as f64));
            mem = m;
            kv = k;
        }
        flush(&mut acc, &mut windows, win_start, mem, kv);
        (windows, frames)
    });

    let t_open = Instant::now();
    let mut ro = CausalRollout::open(pipe, cfg).map_err(|e| anyhow!("{e}"))?;
    let open_s = t_open.elapsed().as_secs_f64();
    let mut blocks: Vec<Value> = Vec::new();
    let mut totals = Vec::new();
    let mut frames = 0usize;
    let mut ttff = None;
    let mut switch_info = None;
    let mut mem_first_full = None;
    let mut mem_max = 0u64;
    let mut mems: Vec<f64> = Vec::new();
    let t_run = Instant::now();
    let mut steady_from: Option<(Instant, usize)> = None;
    let mut count = 0usize;
    while frames < want_frames {
        if let Some(at) = r.switch_at.filter(|_| switch_info.is_none()) {
            if frames as f64 / fps >= at {
                let t = Instant::now();
                let v = ro.set_prompt(a.switch_prompt).map_err(|e| anyhow!("{e}"))?;
                switch_info = Some(json!({"at_frame": frames, "encode_s": t.elapsed().as_secs_f64(), "version": v}));
            }
        }
        let b = ro.next_block().map_err(|e| anyhow!("{e}"))?;
        let tm = b.timings;
        ttff.get_or_insert(t_open.elapsed().as_secs_f64());
        frames += b.num_frames();
        totals.push(tm.total_s);
        let mem = crate::gpu::mem_info().map(|(f, t)| (t - f) >> 20);
        if count == 7 {
            steady_from = Some((Instant::now(), frames));
            mem_first_full = mem;
        }
        count += 1;
        if let Some(m) = mem {
            mem_max = mem_max.max(m);
            mems.push(m as f64);
        }
        let kv_now = ro.kv_bytes();
        blocks.push(json!([b.index, b.num_frames(), tm.denoise_s, tm.context_s, tm.decode_s, tm.rgb_s, tm.total_s, mem, kv_now >> 20]));
        let mut hb = b.into_host();
        if r.drop_rgb {
            hb.rgb.clear();
        }
        if tx.send((hb, mem, kv_now, 0.0)).is_err() {
            break;
        }
    }
    let t_end = Instant::now();
    let kv_bytes = ro.kv_bytes();
    drop(ro);
    drop(tx);
    let (windows, rx_frames) = consumer.join().map_err(|_| anyhow!("consumer panicked"))?;
    let wall = (t_end - t_run).as_secs_f64();
    let steady_fps = steady_from
        .map(|(t, f0)| (frames - f0) as f64 / (t_end - t).as_secs_f64().max(1e-9))
        .unwrap_or(0.0);
    let steady_engine: f64 = totals.iter().skip(8).sum();
    let steady_engine_fps = if steady_engine > 0.0 {
        (frames - steady_from.map_or(frames, |(_, f0)| f0)) as f64 / steady_engine
    } else {
        0.0
    };
    let mut tt = totals.clone();
    let summary = json!({
        "spec": {"seconds": r.seconds, "rope": format!("{:?}", r.rope), "sink": r.sink, "window": r.window,
                 "switch_at": r.switch_at, "switch": format!("{:?}", r.switch)},
        "frames": frames, "consumer_frames": rx_frames, "blocks": totals.len(),
        "open_s": open_s, "ttff_s": ttff, "first_block_s": totals.first(),
        "block_s": {"p50": pct(&mut tt, 0.5), "p90": pct(&mut tt, 0.9), "max": pct(&mut tt, 1.0)},
        "wall_s": wall, "fps_wall": frames as f64 / wall,
        "fps_steady_wall": steady_fps, "fps_steady_engine": steady_engine_fps,
        "mem_used_mib_at_block7": mem_first_full, "mem_used_mib_max": mem_max,
        "kv_mib_end": kv_bytes >> 20,
        "windows": windows,
        "block_rows": ["index", "frames", "denoise_s", "context_s", "decode_s", "rgb_s", "total_s", "mem_used_mib", "kv_mib"],
        "switch": switch_info,
        "per_block": blocks,
    });
    report.note(format!("run/{}", r.name), summary);
    // Settled device memory after the window filled against the end of the
    // run (medians, so one allocator high-water mark does not count).
    if mem_first_full.is_some() && mems.len() >= 48 {
        let base = pct(&mut mems[8..28].to_vec(), 0.5);
        let end = pct(&mut mems[mems.len() - 20..].to_vec(), 0.5);
        report.check(
            format!("run/{}/no_device_memory_growth", r.name),
            end <= base + 256.0,
            json!({"settled_mib": base, "end_mib": end, "transient_max_mib": mem_max}),
            json!({"growth_mib_max": 256}),
        )?;
    }
    report.check(
        format!("run/{}/frames", r.name),
        frames >= want_frames && rx_frames == frames,
        json!({"frames": frames, "consumer": rx_frames}),
        json!({"min": want_frames}),
    )?;
    Ok(())
}
