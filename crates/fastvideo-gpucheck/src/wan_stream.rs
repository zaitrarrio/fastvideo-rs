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
//!   [,switch_at=S[/S2/...]][,switch=keep|reset|recache|recache_sink][,drop_rgb=1]
//!   [,recache=EVERY:KEEP][,graphs=0|1][,fresh=0|1][,sheet=0|1][,keep_s=S]
//!   [,longlive=1]`: a rollout for `S`
//!   seconds of video at `--fps`, blocks handed through a depth-4 channel to
//!   a consumer thread (the design's executor → pacer hand-off) that measures
//!   picture statistics per window of video time. Reports time to first
//!   frame, per-block latency, steady-state frames per second, device memory
//!   and KV bytes over time, and drift indicators: luma, contrast, motion,
//!   the top band of the picture against the rest (banding), the same on the
//!   DiT latents (whether the denoiser or the decoder makes an artefact), a
//!   hash of the latents per window (graph against eager runs), and at the
//!   start of each window a fresh-state TAEHV decode of the last four blocks
//!   against the streamed frames (the carried decoder state). `sheet=1`
//!   (default) writes `sheets/<name>.jpg` next to the report: one tile per
//!   window, the streamed frame over its fresh-state decode. `keep_s=S`
//!   decodes the first `S` seconds of latents again at the end as one clip
//!   (TAEHV's default 4-latent chunks, one carried state) and compares it
//!   with the streamed 3-latent blocks frame by frame.
//!
//! LongLive (`--longlive DIR`, docs/serve/research-longlive.md): the
//! LongLive-1.3B transformer (the converted `longlive_base` + `lora`
//! safetensors, renamed and merged by `wan::longlive`) replaces
//! `--weights/transformer`. `longlive=1` in a run (put it first: later keys
//! override it) takes LongLive's window 12, sink 3, absolute RoPE and the
//! prompt-switch KV re-cache (`switch=recache`; `switch=keep` is the
//! no-re-cache ablation). Several `switch_at` times (`/`-separated) switch
//! through `--switch-prompts` (one prompt per line, or LongLive's
//! `interactive_example.jsonl`: `{"prompts": [...]}` on a line, whose first
//! prompt replaces `--prompt`), else to `--switch-prompt` each time.

use std::sync::mpsc::sync_channel;
use std::time::Instant;

use anyhow::{anyhow, bail};
use fastvideo_cudarc::wan::longlive::LongLiveConfig;
use fastvideo_cudarc::wan::stream::{
    CausalRollout, HostBlock, PromptSwitch, Recache, RolloutConfig, RopePolicy,
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
    /// Prompts of successive switches (`--switch-prompts`); empty: always
    /// `switch_prompt`.
    pub switch_prompts: &'a [String],
    /// LongLive-1.3B converted checkpoint dir (`--longlive`).
    pub longlive: Option<&'a std::path::Path>,
    /// Merge LongLive's `lora.safetensors` (off: the base generator).
    pub longlive_lora: bool,
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
    switch_at: Vec<f64>,
    switch: PromptSwitch,
    drop_rgb: bool,
    recache: Option<Recache>,
    graphs: Option<bool>,
    fresh: bool,
    sheet: bool,
    keep_s: f64,
}

fn parse_run(s: &str) -> anyhow::Result<RunSpec> {
    let mut parts = s.split(',');
    let name = parts.next().filter(|n| !n.is_empty()).ok_or_else(|| anyhow!("--run {s}: no name"))?;
    let mut r = RunSpec {
        name: name.to_string(),
        seconds: 10.0,
        rope: RopePolicy::RebasedSink,
        sink: RolloutConfig::default().sink_frames,
        window: 21,
        switch_at: Vec::new(),
        switch: PromptSwitch::Keep,
        drop_rgb: false,
        recache: None,
        graphs: None,
        fresh: true,
        sheet: true,
        keep_s: 0.0,
    };
    for kv in parts {
        let (k, v) = kv.split_once('=').ok_or_else(|| anyhow!("--run {s}: `{kv}` is not key=value"))?;
        match k {
            "seconds" => r.seconds = v.parse()?,
            "rope" => {
                r.rope = match v {
                    "rel" | "relativistic" => RopePolicy::Relativistic,
                    "abs" | "absolute" => RopePolicy::Absolute,
                    "rebased" | "rebased_sink" => RopePolicy::RebasedSink,
                    _ => bail!("--run {s}: rope={v}"),
                }
            }
            "sink" => r.sink = v.parse()?,
            "window" => r.window = v.parse()?,
            "switch_at" => {
                r.switch_at = v.split('/').map(str::parse).collect::<Result<Vec<f64>, _>>()?;
                r.switch_at.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            "switch" => {
                r.switch = match v {
                    "keep" => PromptSwitch::Keep,
                    "reset" => PromptSwitch::Reset,
                    "recache" => PromptSwitch::Recache { global_sink: true },
                    "recache_sink" => PromptSwitch::Recache { global_sink: false },
                    _ => bail!("--run {s}: switch={v}"),
                }
            }
            "longlive" if v == "1" => {
                let ll = LongLiveConfig::interactive();
                r.window = ll.local_attn_frames;
                r.sink = ll.sink_frames;
                r.rope = ll.rope(false);
                r.switch = ll.prompt_switch();
            }
            "drop_rgb" => r.drop_rgb = v == "1",
            "recache" => {
                let (e, k) = v.split_once(':').ok_or_else(|| anyhow!("--run {s}: recache=EVERY_BLOCKS:KEEP_FRAMES"))?;
                r.recache = Some(Recache { every_blocks: e.parse()?, keep_frames: k.parse()? });
            }
            "graphs" => r.graphs = Some(v == "1"),
            "fresh" => r.fresh = v == "1",
            "sheet" => r.sheet = v == "1",
            "keep_s" => r.keep_s = v.parse()?,
            _ => bail!("--run {s}: unknown key {k}"),
        }
    }
    Ok(r)
}

/// `--switch-prompts`: LongLive's `interactive_example.jsonl` (line `line`
/// is `{"prompts": [p0, p1, ...]}`: `p0` starts the run, the rest are the
/// switches) or plain text (one switch prompt per non-empty line).
pub fn read_switch_prompts(path: &std::path::Path, line: usize) -> anyhow::Result<(Option<String>, Vec<String>)> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    parse_switch_prompts(&text, line).map_err(|e| anyhow!("{}: {e}", path.display()))
}

fn parse_switch_prompts(text: &str, line: usize) -> anyhow::Result<(Option<String>, Vec<String>)> {
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.first().is_some_and(|l| l.starts_with('{')) {
        let l = lines.get(line).ok_or_else(|| anyhow!("no line {line} ({} lines)", lines.len()))?;
        let v: Value = serde_json::from_str(l)?;
        let ps: Vec<String> = v["prompts"]
            .as_array()
            .ok_or_else(|| anyhow!("line {line}: no \"prompts\" array"))?
            .iter()
            .map(|p| p.as_str().map(str::to_string).ok_or_else(|| anyhow!("line {line}: a prompt is not a string")))
            .collect::<anyhow::Result<_>>()?;
        let mut it = ps.into_iter();
        let first = it.next().ok_or_else(|| anyhow!("line {line}: no prompts"))?;
        return Ok((Some(first), it.collect()));
    }
    Ok((None, lines.into_iter().map(str::to_string).collect()))
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
    let dit = match a.longlive {
        Some(dir) => {
            let ll = LongLiveConfig::interactive();
            let mut w = fastvideo_cudarc::wan::longlive::LongLiveWeights::in_dir(dir, &ll);
            if !a.longlive_lora {
                w.lora = None;
            } else if w.lora.is_none() {
                return Err(anyhow!("--longlive {}: no lora.safetensors (--longlive-no-lora for the base)", dir.display()).into());
            }
            let (map, rep) = fastvideo_cudarc::wan::longlive::load_transformer_map(&w).map_err(|e| anyhow!("{e}"))?;
            report.note(
                "longlive_weights",
                json!({"dir": dir.display().to_string(), "tensors": rep.tensors, "lora_modules_merged": rep.merged,
                       "lora_scale": w.lora_scale, "skipped": rep.skipped, "seconds": t.elapsed().as_secs_f64()}),
            );
            Some(map)
        }
        None => None,
    };
    let pipe = WanPipeline::load_with_dit(a.weights, a.preset, LoadParts { text_encoder: true }, dit)
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
        graph_parity(report, &pipe, &base, a.switch_prompt)?;
        parity(report, &pipe, &base)?;
    }
    for r in &runs {
        one_run(report, &pipe, &base, r, a)?;
    }
    Ok(())
}

/// Graph mode (E7) against eager blocks (`graphs: false`, the
/// `FASTVIDEO_WAN_GRAPH=0` path), bit for bit: the default rollout through
/// the window filling, the rolls, a prompt switch (keep) and two resets, so
/// every block of the second and third fills and the steady state replay
/// graphs captured earlier.
fn graph_parity(
    report: &mut Report,
    pipe: &WanPipeline,
    base: &RolloutConfig,
    switch_prompt: &str,
) -> StageResult<()> {
    // (blocks, then what happens before the next leg)
    const LEGS: [(usize, &str); 4] = [(12, "switch"), (3, "reset"), (10, "reset"), (4, "")];
    const SHORT: [(usize, &str); 2] = [(10, "reset"), (10, "")];
    for (arm, rope, legs) in [
        ("rebased", RopePolicy::RebasedSink, &LEGS[..]),
        ("relativistic", RopePolicy::Relativistic, &SHORT[..]),
        ("absolute", RopePolicy::Absolute, &SHORT[..]),
    ] {
        graph_parity_arm(report, pipe, base, switch_prompt, arm, rope, legs)?;
    }
    Ok(())
}

fn graph_parity_arm(
    report: &mut Report,
    pipe: &WanPipeline,
    base: &RolloutConfig,
    switch_prompt: &str,
    arm: &str,
    rope: RopePolicy,
    legs: &[(usize, &str)],
) -> StageResult<()> {
    let run = |graphs: bool| -> anyhow::Result<(Vec<(Vec<u32>, Vec<u32>, f64)>, Option<Value>)> {
        let cfg = RolloutConfig { graphs, rope, rgb8: false, ..base.clone() };
        let mut ro = CausalRollout::open(pipe, cfg).map_err(|e| anyhow!("{e}"))?;
        let mut out = Vec::new();
        for &(n, then) in legs {
            for _ in 0..n {
                let b = ro.next_block().map_err(|e| anyhow!("{e}"))?;
                let bits = |t: &CudaTensor| -> anyhow::Result<Vec<u32>> {
                    Ok(host(t)?.iter().map(|v| v.to_bits()).collect())
                };
                out.push((bits(&b.latents)?, bits(&b.frames)?, b.timings.total_s));
            }
            match then {
                "switch" => {
                    ro.set_prompt(switch_prompt).map_err(|e| anyhow!("{e}"))?;
                }
                "reset" => ro.reset(None),
                _ => {}
            }
        }
        let g = ro.graph_report().map(|r| {
            json!({"eager_blocks": r.eager_blocks, "captured_blocks": r.captured_blocks,
                   "replayed_blocks": r.replayed_blocks, "keys": r.keys,
                   "denoise_kernels": r.denoise_kernels, "context_kernels": r.context_kernels,
                   "failed": r.failed})
        });
        Ok((out, g))
    };
    let t = Instant::now();
    let (eager, _) = run(false)?;
    let eager_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (graph, g) = run(true)?;
    let graph_s = t.elapsed().as_secs_f64();
    let differing: Vec<usize> = eager
        .iter()
        .zip(&graph)
        .enumerate()
        .filter(|(_, (a, b))| a.0 != b.0 || a.1 != b.1)
        .map(|(i, _)| i)
        .collect();
    let first_diff = differing.first().map(|&i| {
        let (a, b) = (&eager[i], &graph[i]);
        let f = |v: &[u32]| v.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>();
        json!({"block": i,
               "latents": crate::metrics::diff(&f(&a.0), &f(&b.0)).to_json(),
               "frames": crate::metrics::diff(&f(&a.1), &f(&b.1)).to_json()})
    });
    let times = |v: &[(Vec<u32>, Vec<u32>, f64)]| v.iter().map(|x| x.2).collect::<Vec<_>>();
    let failed = g.as_ref().and_then(|g| g.get("failed").cloned()).unwrap_or(Value::Null);
    report.note(
        format!("parity/graph_vs_eager/{arm}"),
        json!({"blocks": eager.len(), "differing_blocks": differing, "first_difference": first_diff,
               "graph": g, "eager_wall_s": eager_s, "graph_wall_s": graph_s,
               "eager_block_s": times(&eager), "graph_block_s": times(&graph)}),
    );
    report.check(
        format!("parity/graph_bitwise_equals_eager/{arm}"),
        differing.is_empty() && failed.is_null() && eager.len() == graph.len(),
        json!({"differing_blocks": differing.len(), "blocks": eager.len(), "failed": failed}),
        json!({"differing_blocks": 0}),
    )?;
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
    for (name, rope) in [("absolute", RopePolicy::Absolute)] {
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

    // The rebased sink against FastVideo's relativistic policy, past the
    // first roll (14 blocks, sink 3): the same attention offsets, different
    // rounding points.
    let roll = |rope: RopePolicy| -> anyhow::Result<(Vec<f32>, f64)> {
        let cfg = RolloutConfig { rope, sink_frames: 3, rgb8: false, ..base.clone() };
        let mut ro = CausalRollout::open(pipe, cfg).map_err(|e| anyhow!("{e}"))?;
        let (mut lats, mut secs) = (Vec::new(), 0.0);
        for _ in 0..14 {
            let b = ro.next_block().map_err(|e| anyhow!("{e}"))?;
            if b.index >= 8 {
                secs += b.timings.total_s;
            }
            lats.extend_from_slice(&host(&b.latents)?);
        }
        Ok((lats, secs / 6.0))
    };
    let (rel, rel_s) = roll(RopePolicy::Relativistic)?;
    let (reb, reb_s) = roll(RopePolicy::RebasedSink)?;
    let (abs, abs_s) = roll(RopePolicy::Absolute)?;
    let n7 = rel.len() / 2;
    let d = |a: &[f32], b: &[f32]| crate::metrics::diff(a, b).to_json();
    report.note(
        "parity/sink3_policies",
        json!({
            "rebased_vs_relativistic": {"blocks_0_6": d(&reb[..n7], &rel[..n7]), "blocks_7_13": d(&reb[n7..], &rel[n7..])},
            "absolute_vs_relativistic": {"blocks_0_6": d(&abs[..n7], &rel[..n7]), "blocks_7_13": d(&abs[n7..], &rel[n7..])},
            "steady_block_s": {"relativistic": rel_s, "rebased": reb_s, "absolute": abs_s},
        }),
    );
    Ok(())
}

/// Rows of the top band (the first eighth of the picture) that the banding
/// statistics single out.
fn top_rows(h: usize) -> usize {
    (h / 8).max(1)
}

/// `(mean, std, hu)` of rows `[r0, r1)` of a `w`-wide plane. `hu` is the mean
/// variance along a row over the variance of the whole band: about 1 for
/// texture, near 0 for horizontal stripes (rows that differ from each other
/// but not along themselves).
fn band_stats(p: &[f32], w: usize, r0: usize, r1: usize) -> (f64, f64, f64) {
    let rows = &p[r0 * w..r1 * w];
    let n = rows.len().max(1) as f64;
    let mean = rows.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = rows.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n;
    let mut along = 0.0;
    for r in rows.chunks_exact(w) {
        let m = r.iter().map(|&v| f64::from(v)).sum::<f64>() / w as f64;
        along += r.iter().map(|&v| (f64::from(v) - m).powi(2)).sum::<f64>() / w as f64;
    }
    along /= (r1 - r0).max(1) as f64;
    (mean, var.sqrt(), if var > 1e-9 { along / var } else { 1.0 })
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
    /// The top band (first eighth of the rows) and the rest.
    top_luma: f64,
    top_std: f64,
    top_hu: f64,
    rest_hu: f64,
    top_sat: f64,
    rest_sat: f64,
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

/// Mean saturation (max − min channel, 0-255) of the top band and the rest.
fn frame_sat(rgb: &[u8], h: usize, w: usize, stride: usize) -> (f64, f64) {
    let top = top_rows(h.div_ceil(stride));
    let (mut s, mut n) = ([0.0f64; 2], [0usize; 2]);
    for (yi, y) in (0..h).step_by(stride).enumerate() {
        let k = usize::from(yi >= top);
        for x in (0..w).step_by(stride) {
            let p = (y * w + x) * 3;
            let px = &rgb[p..p + 3];
            let (mx, mn) = (px.iter().max().copied().unwrap_or(0), px.iter().min().copied().unwrap_or(0));
            s[k] += f64::from(mx - mn);
            n[k] += 1;
        }
    }
    (s[0] / n[0].max(1) as f64, s[1] / n[1].max(1) as f64)
}

fn block_pic(b: &HostBlock, prev_last: &mut Option<Vec<f32>>) -> Pic {
    let (h, w) = (b.height, b.width);
    let plane = h * w * 3;
    let (sh, sw) = (h.div_ceil(2), w.div_ceil(2));
    let top = top_rows(sh);
    let lumas: Vec<Vec<f32>> = (0..b.frames)
        .map(|i| frame_luma(&b.rgb[i * plane..(i + 1) * plane], h, w, 2))
        .collect();
    let mut p = Pic::default();
    let n = lumas.len().max(1) as f64;
    for (i, l) in lumas.iter().enumerate() {
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
        let (tm, ts, thu) = band_stats(l, sw, 0, top);
        let (_, _, rhu) = band_stats(l, sw, top, sh);
        p.top_luma += tm / n;
        p.top_std += ts / n;
        p.top_hu += thu / n;
        p.rest_hu += rhu / n;
        let (tsat, rsat) = frame_sat(&b.rgb[i * plane..(i + 1) * plane], h, w, 2);
        p.top_sat += tsat / n;
        p.rest_sat += rsat / n;
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

/// Statistics of one block's DiT latents `[1, C, T, h, w]`: the top band
/// (first eighth of the latent rows) against the rest, per channel plane,
/// averaged.
#[derive(Default, Clone, Copy)]
struct LatStats {
    top_mean: f64,
    top_std: f64,
    top_hu: f64,
    rest_mean: f64,
    rest_std: f64,
    rest_hu: f64,
    absmax: f64,
}

fn lat_stats(v: &[f32], shape: &[usize]) -> LatStats {
    let (h, w) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let top = top_rows(h);
    let k = (v.len() / (h * w).max(1)).max(1) as f64;
    let mut s = LatStats::default();
    for p in v.chunks_exact(h * w) {
        let (tm, ts, thu) = band_stats(p, w, 0, top);
        let (rm, rs, rhu) = band_stats(p, w, top, h);
        s.top_mean += tm / k;
        s.top_std += ts / k;
        s.top_hu += thu / k;
        s.rest_mean += rm / k;
        s.rest_std += rs / k;
        s.rest_hu += rhu / k;
    }
    s.absmax = v.iter().fold(0.0f64, |m, &x| m.max(f64::from(x.abs())));
    s
}

/// FNV-1a over the bits (graph and eager runs must agree bit for bit).
fn fnv(v: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// A fresh-state decode of the last blocks against the streamed frames of
/// the newest one (RGB8).
struct Fresh {
    /// The first frame of the block, fresh-state decode.
    first: Vec<u8>,
    mad: f64,
    top_mad: f64,
    max: u8,
}

/// What the main thread adds to a block for the consumer.
struct Extra {
    lat: LatStats,
    hash: u64,
    /// The first block of a statistics window: its first frame goes on the
    /// contact sheet (with the fresh-state decode, when measured).
    sample: bool,
    fresh: Option<Fresh>,
    recaches: usize,
}

/// `(mad, top-band mad, max)` between two RGB8 frame stacks of `h` rows.
fn rgb_diff(a: &[u8], b: &[u8], h: usize, w: usize) -> (f64, f64, u8) {
    let row = w * 3;
    let top = top_rows(h);
    let (mut s, mut st, mut nt, mut mx) = (0u64, 0u64, 0u64, 0u8);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = x.abs_diff(y);
        s += u64::from(d);
        mx = mx.max(d);
        if (i / row) % h < top {
            st += u64::from(d);
            nt += 1;
        }
    }
    (s as f64 / a.len().max(1) as f64, st as f64 / nt.max(1) as f64, mx)
}

/// Half-size RGB8 tile (2x2 box filter) of a `h x w` frame.
fn half(rgb: &[u8], h: usize, w: usize) -> Vec<u8> {
    let (th, tw) = (h / 2, w / 2);
    let mut out = vec![0u8; th * tw * 3];
    for y in 0..th {
        for x in 0..tw {
            for c in 0..3 {
                let at = |yy: usize, xx: usize| u32::from(rgb[(yy * w + xx) * 3 + c]);
                let v = at(2 * y, 2 * x) + at(2 * y + 1, 2 * x) + at(2 * y, 2 * x + 1) + at(2 * y + 1, 2 * x + 1);
                out[(y * tw + x) * 3 + c] = ((v + 2) / 4) as u8;
            }
        }
    }
    out
}

/// A streamed tile and, when measured, its fresh-state decode.
type Tile = (Vec<u8>, Option<Vec<u8>>);

/// One tile per window (the streamed frame, with the fresh-state decode
/// below it when measured), six to a row, as a JPEG.
fn write_sheet(path: &std::path::Path, tiles: &[Tile], h: usize, w: usize) -> anyhow::Result<()> {
    let (th, tw) = (h / 2, w / 2);
    let rows_per = if tiles.iter().any(|t| t.1.is_some()) { 2 } else { 1 };
    let cols = tiles.len().clamp(1, 6);
    let grid_rows = tiles.len().div_ceil(cols);
    let (iw, ih) = (cols * (tw + 4), grid_rows * (rows_per * th + 8));
    let mut img = image::RgbImage::from_pixel(iw as u32, ih as u32, image::Rgb([32, 32, 32]));
    for (i, (s, f)) in tiles.iter().enumerate() {
        let (x0, y0) = ((i % cols) * (tw + 4), (i / cols) * (rows_per * th + 8));
        for (k, t) in [Some(s), f.as_ref()].into_iter().enumerate() {
            let Some(t) = t else { continue };
            for y in 0..th {
                for x in 0..tw {
                    let p = (y * tw + x) * 3;
                    img.put_pixel((x0 + x) as u32, (y0 + k * th + y) as u32, image::Rgb([t[p], t[p + 1], t[p + 2]]));
                }
            }
        }
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, 80).encode_image(&img)?;
    Ok(())
}

/// The first latents decoded again as one clip (TAEHV's default 4-latent
/// chunks through one carried state, a fresh one) against the streamed
/// frames (3-latent blocks), per window of video time.
fn whole_clip_check(
    tae: &fastvideo_cudarc::wan::taehv::TaeHv,
    lats: &[CudaTensor],
    streamed: &[u8],
    (h, w): (usize, usize),
    window_frames: usize,
) -> anyhow::Result<Value> {
    let lat = CudaTensor::cat(&lats.iter().collect::<Vec<_>>(), 2).map_err(|e| anyhow!("{e}"))?;
    let t = lat.shape[2];
    let plane = h * w * 3;
    let mut st = tae.decode_state();
    let (mut s, mut frame) = (0usize, 0usize);
    // Per window: frames, differing values, values, max, sum |d| top, n top,
    // sum |d| rest, n rest.
    let mut wins: Vec<[f64; 8]> = Vec::new();
    let top = top_rows(h) * w * 3;
    while s < t {
        let len = 4.min(t - s);
        let z = lat.narrow(2, s, len).map_err(|e| anyhow!("{e}"))?;
        s += len;
        let Some(piece) = tae.decode_step(&mut st, &z).map_err(|e| anyhow!("{e}"))? else {
            continue;
        };
        let rgb = fastvideo_cudarc::wan::pipeline::frames_to_rgb8(&piece).map_err(|e| anyhow!("{e}"))?;
        for f in rgb.chunks_exact(plane) {
            let Some(o) = streamed.get(frame * plane..(frame + 1) * plane) else { break };
            let wi = frame / window_frames.max(1);
            if wins.len() <= wi {
                wins.resize(wi + 1, [0.0; 8]);
            }
            let acc = &mut wins[wi];
            acc[0] += 1.0;
            for (i, (&a, &b)) in f.iter().zip(o).enumerate() {
                let d = a.abs_diff(b);
                acc[1] += f64::from(u8::from(d != 0));
                acc[2] += 1.0;
                acc[3] = acc[3].max(f64::from(d));
                if i < top {
                    acc[4] += f64::from(d);
                    acc[5] += 1.0;
                } else {
                    acc[6] += f64::from(d);
                    acc[7] += 1.0;
                }
            }
            frame += 1;
        }
    }
    Ok(json!({
        "latent_frames": t, "frames_compared": frame,
        "windows": wins.iter().map(|a| json!({
            "frames": a[0], "differing_fraction": a[1] / a[2].max(1.0), "max_levels": a[3],
            "top_mean_abs": a[4] / a[5].max(1.0), "rest_mean_abs": a[6] / a[7].max(1.0),
        })).collect::<Vec<_>>(),
    }))
}

/// One statistics window's blocks: picture, latent statistics, frames.
type WinAcc = Vec<(Pic, LatStats, f64)>;

/// What the consumer hands back: windows, frames, contact-sheet tiles and
/// the frame size.
type Consumed = (Vec<Value>, usize, Vec<Tile>, (usize, usize));

#[allow(clippy::too_many_arguments)]
fn flush_window(
    acc: &mut WinAcc,
    windows: &mut Vec<Value>,
    t0: f64,
    mem: Option<u64>,
    kv: usize,
    hash: u64,
    fresh: Option<(f64, f64, u8)>,
    recaches: usize,
) {
    if acc.is_empty() {
        return;
    }
    let wsum: f64 = acc.iter().map(|(_, _, w)| w).sum();
    let avg = |f: fn(&Pic) -> f64| acc.iter().map(|(p, _, w)| f(p) * w).sum::<f64>() / wsum;
    let lavg = |f: fn(&LatStats) -> f64| acc.iter().map(|(_, l, w)| f(l) * w).sum::<f64>() / wsum;
    let seams: Vec<f64> = acc.iter().map(|(p, _, _)| p.seam_mad).filter(|&s| s > 0.0).collect();
    windows.push(json!({
        "t0_s": t0, "blocks": acc.len(),
        "luma": avg(|p| p.luma), "std": avg(|p| p.std), "mad": avg(|p| p.mad),
        "seam_mad": if seams.is_empty() { 0.0 } else { seams.iter().sum::<f64>() / seams.len() as f64 },
        "sharpness": avg(|p| p.sharp), "clipped": avg(|p| p.clipped),
        "top_luma": avg(|p| p.top_luma), "top_std": avg(|p| p.top_std),
        "top_hu": avg(|p| p.top_hu), "rest_hu": avg(|p| p.rest_hu),
        "top_sat": avg(|p| p.top_sat), "rest_sat": avg(|p| p.rest_sat),
        "lat_top_mean": lavg(|l| l.top_mean), "lat_rest_mean": lavg(|l| l.rest_mean),
        "lat_top_std": lavg(|l| l.top_std), "lat_rest_std": lavg(|l| l.rest_std),
        "lat_top_hu": lavg(|l| l.top_hu), "lat_rest_hu": lavg(|l| l.rest_hu),
        "lat_absmax": acc.iter().map(|(_, l, _)| l.absmax).fold(0.0, f64::max),
        "latent_hash": format!("{hash:016x}"),
        "fresh_decode": fresh.map(|(m, t, x)| json!({"mad": m, "top_mad": t, "max": x})),
        "recaches": recaches,
        "mem_used_mib": mem, "kv_mib": kv >> 20,
    }));
    acc.clear();
}

/// The consumer thread: picture statistics per window of video time and
/// the contact-sheet tiles.
fn consume(
    rx: std::sync::mpsc::Receiver<(HostBlock, Option<u64>, usize, Extra)>,
    fps: f64,
    window_s: f64,
    sheet: bool,
) -> Consumed {
    let mut prev_last = None;
    let mut windows: Vec<Value> = Vec::new();
    let mut acc: WinAcc = Vec::new();
    let mut win_hash = 0u64;
    let mut win_fresh: Option<(f64, f64, u8)> = None;
    let mut recaches = 0usize;
    let mut tiles: Vec<Tile> = Vec::new();
    let mut dims = (0usize, 0usize);
    let (mut mem, mut kv) = (None, 0usize);
    let mut win_start = 0.0f64;
    let mut frames = 0usize;
    for (b, m, k, x) in rx {
        let t = b.first_frame as f64 / fps;
        if t >= win_start + window_s {
            flush_window(&mut acc, &mut windows, win_start, mem, kv, win_hash, win_fresh.take(), recaches);
            win_hash = 0;
            win_start = (t / window_s).floor() * window_s;
        }
        let p = if b.rgb.is_empty() { Pic::default() } else { block_pic(&b, &mut prev_last) };
        if sheet && x.sample && !b.rgb.is_empty() {
            dims = (b.height, b.width);
            let plane = b.height * b.width * 3;
            tiles.push((
                half(&b.rgb[..plane], b.height, b.width),
                x.fresh.as_ref().map(|f| half(&f.first, b.height, b.width)),
            ));
        }
        if let Some(f) = &x.fresh {
            win_fresh = Some((f.mad, f.top_mad, f.max));
        }
        win_hash = win_hash.rotate_left(7) ^ x.hash;
        recaches = x.recaches;
        frames += b.frames;
        acc.push((p, x.lat, b.frames as f64));
        mem = m;
        kv = k;
    }
    flush_window(&mut acc, &mut windows, win_start, mem, kv, win_hash, win_fresh.take(), recaches);
    (windows, frames, tiles, dims)
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
        recache: r.recache,
        graphs: r.graphs.unwrap_or(base.graphs),
        ..base.clone()
    };
    let tae = pipe.taehv().ok_or_else(|| anyhow!("no TAEHV loaded"))?;
    let (tx, rx) = sync_channel::<(HostBlock, Option<u64>, usize, Extra)>(4);
    let (window_s, sheet) = (a.window_s, r.sheet);
    let consumer = std::thread::spawn(move || consume(rx, fps, window_s, sheet));

    let t_open = Instant::now();
    let mut ro = CausalRollout::open(pipe, cfg).map_err(|e| anyhow!("{e}"))?;
    let open_s = t_open.elapsed().as_secs_f64();
    let mut blocks: Vec<Value> = Vec::new();
    let mut totals = Vec::new();
    let mut frames = 0usize;
    let mut ttff = None;
    let mut switch_info: Vec<Value> = Vec::new();
    let mut mem_first_full = None;
    let mut mem_max = 0u64;
    let mut mems: Vec<f64> = Vec::new();
    let t_run = Instant::now();
    let mut steady_from: Option<(Instant, usize)> = None;
    let mut count = 0usize;
    let mut next_sample = 0.0f64;
    let mut ring: std::collections::VecDeque<CudaTensor> = Default::default();
    let (mut keep_lat, mut keep_rgb) = (Vec::new(), Vec::new());
    let mut extra_s = 0.0f64;
    while frames < want_frames {
        if let Some(&at) = r.switch_at.get(switch_info.len()) {
            if frames as f64 / fps >= at {
                let i = switch_info.len();
                let p = if a.switch_prompts.is_empty() {
                    a.switch_prompt
                } else {
                    a.switch_prompts[i % a.switch_prompts.len()].as_str()
                };
                let t = Instant::now();
                let v = ro.set_prompt(p).map_err(|e| anyhow!("{e}"))?;
                switch_info.push(json!({"at_frame": frames, "encode_s": t.elapsed().as_secs_f64(), "version": v,
                                        "prompt": p}));
            }
        }
        let mut b = ro.next_block().map_err(|e| anyhow!("{e}"))?;
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
        blocks.push(json!([b.index, b.num_frames(), tm.denoise_s, tm.context_s, tm.decode_s, tm.rgb_s, tm.total_s, mem, kv_now >> 20, tm.recache_s]));
        // Diagnostics, outside the block's own timings.
        let t_extra = Instant::now();
        let lat_h = host(&b.latents)?;
        let lat = lat_stats(&lat_h, &b.latents.shape);
        let hash = fnv(&lat_h);
        drop(lat_h);
        if r.fresh {
            ring.push_back(b.latents.clone());
            while ring.len() > 4 {
                ring.pop_front();
            }
        }
        let t_video = b.first_frame as f64 / fps;
        let sample = t_video >= next_sample;
        if sample {
            next_sample = ((t_video / window_s).floor() + 1.0) * window_s;
        }
        let mut fresh = None;
        if let (true, true, Some(rgb)) = (sample, r.fresh, b.rgb.as_ref()) {
            let z = CudaTensor::cat(&ring.iter().collect::<Vec<_>>(), 2).map_err(|e| anyhow!("{e}"))?;
            let dec = tae.decode(&z).map_err(|e| anyhow!("{e}"))?;
            let (f, h, w) = (dec.shape[2], dec.shape[3], dec.shape[4]);
            let n = b.num_frames();
            let own = dec
                .reshape(vec![3, f, h, w])
                .and_then(|x| x.permute(&[1, 0, 2, 3]))
                .and_then(|x| x.narrow(0, f - n, n))
                .map_err(|e| anyhow!("{e}"))?;
            let q = fastvideo_cudarc::wan::pipeline::frames_to_rgb8(&own).map_err(|e| anyhow!("{e}"))?;
            let (mad, top_mad, max) = rgb_diff(&q, rgb, h, w);
            fresh = Some(Fresh { first: q[..h * w * 3].to_vec(), mad, top_mad, max });
        }
        if r.keep_s > 0.0 && t_video < r.keep_s {
            keep_lat.push(b.latents.clone());
            if let Some(rgb) = b.rgb.as_ref() {
                keep_rgb.extend_from_slice(rgb);
            }
        }
        extra_s += t_extra.elapsed().as_secs_f64();
        let x = Extra { lat, hash, sample, fresh, recaches: ro.recaches() };
        b.latents = CudaTensor::zeros(&[1]);
        let mut hb = b.into_host();
        if r.drop_rgb {
            hb.rgb.clear();
        }
        if tx.send((hb, mem, kv_now, x)).is_err() {
            break;
        }
    }
    let t_end = Instant::now();
    let kv_bytes = ro.kv_bytes();
    let graph = ro.graph_report().map(|r| {
        json!({"eager_blocks": r.eager_blocks, "captured_blocks": r.captured_blocks,
               "replayed_blocks": r.replayed_blocks, "keys": r.keys,
               "denoise_kernels": r.denoise_kernels, "context_kernels": r.context_kernels,
               "failed": r.failed})
    });
    let recaches = ro.recaches();
    let switch_recaches = ro.switch_recaches();
    drop(ro);
    drop(tx);
    drop(ring);
    let (windows, rx_frames, tiles, dims) = consumer.join().map_err(|_| anyhow!("consumer panicked"))?;
    let sheet_path = if r.sheet && !tiles.is_empty() {
        let p = report.dir().join("sheets").join(format!("{}.jpg", r.name));
        write_sheet(&p, &tiles, dims.0, dims.1)?;
        Some(p.to_string_lossy().into_owned())
    } else {
        None
    };
    let whole_clip = if keep_lat.is_empty() {
        None
    } else {
        Some(whole_clip_check(tae, &keep_lat, &keep_rgb, (a.height, a.width), (window_s * fps) as usize)?)
    };
    drop((keep_lat, keep_rgb));
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
                 "switch_at": r.switch_at, "switch": format!("{:?}", r.switch),
                 "recache": r.recache.map(|c| json!({"every_blocks": c.every_blocks, "keep_frames": c.keep_frames})),
                 "graphs": r.graphs, "fresh": r.fresh, "keep_s": r.keep_s},
        "frames": frames, "consumer_frames": rx_frames, "blocks": totals.len(),
        "open_s": open_s, "ttff_s": ttff, "first_block_s": totals.first(),
        "block_s": {"p50": pct(&mut tt, 0.5), "p90": pct(&mut tt, 0.9), "max": pct(&mut tt, 1.0)},
        "wall_s": wall, "fps_wall": frames as f64 / wall, "diagnostics_s": extra_s,
        "fps_steady_wall": steady_fps, "fps_steady_engine": steady_engine_fps,
        "mem_used_mib_at_block7": mem_first_full, "mem_used_mib_max": mem_max,
        "kv_mib_end": kv_bytes >> 20,
        "graph": graph,
        "recaches": recaches,
        "switch_recaches": switch_recaches,
        "sheet": sheet_path,
        "whole_clip_decode": whole_clip,
        "windows": windows,
        "block_rows": ["index", "frames", "denoise_s", "context_s", "decode_s", "rgb_s", "total_s", "mem_used_mib", "kv_mib", "recache_s"],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longlive_run_spec() {
        let r = parse_run("ll,longlive=1,seconds=60,switch_at=15/30/45").unwrap();
        assert_eq!((r.window, r.sink, r.rope), (12, 3, RopePolicy::Absolute));
        assert_eq!(r.switch, PromptSwitch::Recache { global_sink: true });
        assert_eq!(r.switch_at, [15.0, 30.0, 45.0]);
        let r = parse_run("ll,longlive=1,switch=keep,switch_at=45/15").unwrap();
        assert_eq!(r.switch, PromptSwitch::Keep);
        assert_eq!(r.switch_at, [15.0, 45.0]);
        let r = parse_run("x,switch=recache_sink,rope=rel").unwrap();
        assert_eq!(r.switch, PromptSwitch::Recache { global_sink: false });
        assert!(parse_run("x,switch=nope").is_err());
    }

    #[test]
    fn switch_prompts_from_jsonl_or_lines() {
        let jsonl = "{\"prompts\": [\"a\", \"b\", \"c\"]}\n{\"prompts\": [\"x\", \"y\"]}\n";
        assert_eq!(parse_switch_prompts(jsonl, 0).unwrap(), (Some("a".into()), vec!["b".into(), "c".into()]));
        assert_eq!(parse_switch_prompts(jsonl, 1).unwrap(), (Some("x".into()), vec!["y".into()]));
        assert!(parse_switch_prompts(jsonl, 2).is_err());
        assert_eq!(parse_switch_prompts("p1\n\np2\n", 0).unwrap(), (None, vec!["p1".into(), "p2".into()]));
    }
}
