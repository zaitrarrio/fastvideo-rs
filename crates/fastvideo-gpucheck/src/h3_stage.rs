//! MiniMax-H3 / FastH3 stages: each judges one part of the port against the
//! reference dump written by `scripts/gpu/h3_oracle.py` (see docs/ports/h3.md,
//! section j). Owned by the H3 track; `main.rs` only dispatches here.
//!
//! Every stage reads the oracle's *inputs* as well as its outputs (token ids,
//! noise, fixed latents), so a stage's error belongs to the code it names and
//! not to whatever ran before it. All stages share one report name (`h3`);
//! pass the global `--tag <stage>` to keep their JSON files apart.

use std::path::{Path, PathBuf};

use anyhow::Context;
use fastvideo_cudarc::llm::DecoderConfig;
use fastvideo_cudarc::wan::weights::WeightMap;
use fastvideo_cudarc::CudaTensor;
use serde_json::json;

use crate::metrics::diff;
use crate::model::measure;
use crate::report::{Report, StageResult};
use crate::st;

#[derive(clap::Subcommand, Debug)]
pub enum Stage {
    /// Print the inference contract and geometry the port targets.
    Info,
    /// Tokenizer parity and the Qwen3-VL hidden states H3 conditions on.
    Text {
        /// Root of the FastH3 snapshot (`tokenizer/`, `text_encoder/`, ...).
        #[arg(long)]
        weights: PathBuf,
        /// `--out` file of `h3_oracle.py` (stage `text`).
        #[arg(long)]
        oracle: PathBuf,
        /// `--meta` file of `h3_oracle.py`; supplies the prompt and the
        /// reference's ids for the added marker tokens. Defaults to the oracle
        /// path with a `.json` extension.
        #[arg(long)]
        meta: Option<PathBuf>,
        /// The prompt, when no meta file is at hand.
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// `hidden_states[50]`: bf16 on both sides through 50 layers.
        #[arg(long, default_value_t = 2e-2)]
        max_rel: f64,
        /// `hidden_states[1]`: one layer.
        #[arg(long, default_value_t = 2e-3)]
        max_rel_layer0: f64,
    },
    /// The BigVGAN audio decoder on the oracle's fixed latent, float32 on both sides.
    AudioVae {
        /// Root of the FastH3 snapshot (reads `audio_vae/`).
        #[arg(long)]
        weights: PathBuf,
        /// `--out` file of `h3_oracle.py` (stage `audio`): `audio_latent`, `audio_wave`.
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Waveform samples live in [-1, 1]; both sides are float32.
        #[arg(long, default_value_t = 1e-4)]
        max_abs: f64,
    },
    /// The ViT video decoder (tiled, temporally chunked) on the oracle's fixed latent.
    Vae {
        /// Root of the FastH3 snapshot (reads `vae/`).
        #[arg(long)]
        weights: PathBuf,
        /// `--out` file of `h3_oracle.py` (stage `vae`): `vae_latent`, `vae_video_raw`.
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// ImageNet-normalized RGB, before the clamp; float32 on both sides in
        /// `--mode exact`. With bf16 linears (`--mode fast`) pass a budget from
        /// the meta's `vae_fp16_autocast_vs_fp32`.
        #[arg(long, default_value_t = 2e-3)]
        max_abs: f64,
        #[arg(long, default_value_t = 1e-3)]
        max_rel: f64,
    },
    /// One dense DiT forward on the oracle's packed input, with the
    /// intermediate hooks (temb, AdaLN block 0, refined text, block outputs).
    /// Run with `--mode fast`: exact mode would load 37 GiB of linears as f32.
    Dit {
        /// Root of the FastH3 snapshot (reads `transformer/`).
        #[arg(long)]
        weights: PathBuf,
        /// `--out` file of `h3_oracle.py` (stage `dit`).
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// `dit_video` / `dit_audio`: bf16 on both sides through 50 blocks.
        #[arg(long, default_value_t = 3e-2)]
        max_rel: f64,
    },
    /// The full 8-step ladder in dense mode against the oracle's per-step
    /// latents: scheduler sign, both shifts, the step ratio. `--mode fast`.
    Loop {
        #[arg(long)]
        weights: PathBuf,
        /// `--out` file of `h3_oracle.py` (stage `loop`).
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// After the first step: one forward's worth of divergence.
        #[arg(long, default_value_t = 3e-2)]
        max_rel_first: f64,
        /// After the last step: eight forwards, each fed the previous one's error.
        #[arg(long, default_value_t = 1e-1)]
        max_rel_last: f64,
    },
    /// VSA-H3 on the device against its plain-loop statement, and against
    /// dense attention at sparsity 0 without the gate. No weights needed.
    Vsa {
        #[arg(long, default_value = "cuda")]
        device: String,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// bf16 tensor-core fine stage against a float64 host reference.
        #[arg(long, default_value_t = 1e-2)]
        max_rel: f64,
    },
    /// Generate one clip end to end (VSA-H3 unless `--dense`) and report
    /// timings and peak VRAM. `--mode fast`.
    Gen {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        prompt: String,
        /// 5 to 15.
        #[arg(long, default_value_t = 5)]
        seconds: usize,
        #[arg(long, default_value_t = 1024)]
        seed: u64,
        /// Dense attention without the compression gate (the parity mode).
        #[arg(long)]
        dense: bool,
        #[arg(long)]
        no_mp4: bool,
        /// Where `frame-NNN.png`, `audio.wav` and `output.mp4` go.
        #[arg(long, default_value = "gpucheck-out/h3-gen")]
        clip_dir: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Info => {
            let c = fastvideo_models::h3::config::H3InferenceContract::fasth3_8step();
            report.set("contract", format!("{c:?}"));
            Ok(())
        }
        Stage::Text { weights, oracle, meta, prompt, device, max_rel, max_rel_layer0 } => {
            text(report, weights, oracle, meta.as_deref(), prompt.as_deref(), device, *max_rel, *max_rel_layer0)
        }
        Stage::AudioVae { weights, oracle, device, max_abs } => audio_vae(report, weights, oracle, device, *max_abs),
        Stage::Vae { weights, oracle, device, max_abs, max_rel } => vae(report, weights, oracle, device, *max_abs, *max_rel),
        Stage::Dit { weights, oracle, device, max_rel } => dit(report, weights, oracle, device, *max_rel),
        Stage::Loop { weights, oracle, device, max_rel_first, max_rel_last } => ladder(report, weights, oracle, device, *max_rel_first, *max_rel_last),
        Stage::Vsa { device, seed, max_rel } => vsa(report, device, *seed, *max_rel),
        Stage::Gen { weights, prompt, seconds, seed, dense, no_mp4, clip_dir, device } => {
            gen(report, weights, prompt, *seconds, *seed, *dense, !*no_mp4, clip_dir, device)
        }
    }
}

/// A float tensor of small non-negative integers, as integers.
fn ints(t: &st::F32Tensor, name: &str) -> anyhow::Result<Vec<u32>> {
    t.data
        .iter()
        .map(|&v| {
            if v < 0.0 || v.fract() != 0.0 || v > 16_777_216.0 {
                anyhow::bail!("{name}: {v} is not an integer id");
            }
            Ok(v as u32)
        })
        .collect()
}

fn read_meta(oracle: &Path, meta: Option<&Path>) -> anyhow::Result<Option<serde_json::Value>> {
    let path = meta.map_or_else(|| oracle.with_extension("json"), Path::to_path_buf);
    if !path.exists() {
        if meta.is_some() {
            anyhow::bail!("{} does not exist", path.display());
        }
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(Some(serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?))
}

#[allow(clippy::too_many_arguments)]
fn text(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    meta: Option<&Path>,
    prompt: Option<&str>,
    device: &str,
    max_rel: f64,
    max_rel_layer0: f64,
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let mut orc = st::load(oracle)?;
    let want_ids = ints(&st::take(&mut orc, "text_ids", oracle)?, "text_ids")?;
    let meta = read_meta(oracle, meta)?;
    if meta.as_ref().is_some_and(|m| m.get("text_is_synthetic").is_some()) {
        return Err(anyhow::anyhow!("{}: `text` is a synthetic stand-in; rerun h3_oracle.py with the text stage", oracle.display()).into());
    }

    // --- tokenizer parity: ours on the prompt vs the reference's ids ---------
    let prompt = match (prompt, meta.as_ref().and_then(|m| m.get("prompt")).and_then(|p| p.as_str())) {
        (Some(p), _) => p.to_string(),
        (None, Some(p)) => p.to_string(),
        (None, None) => return Err(anyhow::anyhow!("no prompt: pass --meta <h3_oracle meta json> or --prompt").into()),
    };
    let tokenizer = fastvideo_models::h3::tokenizer::H3Tokenizer::from_file(&weights.join("tokenizer").join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e))?;
    let ours: serde_json::Map<String, serde_json::Value> =
        tokenizer.added_special_token_ids().iter().map(|(t, id)| (t.clone(), json!(id))).collect();
    let reference = meta.as_ref().and_then(|m| m.get("added_special_token_ids")).cloned();
    let added_ok = match &reference {
        Some(serde_json::Value::Object(r)) => !r.is_empty() && r.iter().all(|(t, id)| ours.get(t) == Some(id)),
        _ => tokenizer.added_ids_match_reference(),
    };
    report.check(
        "added_special_tokens",
        added_ok,
        json!({"ours": ours}),
        json!({"reference": reference.unwrap_or_else(|| json!("151669..=151675 (no meta file)"))}),
    )?;
    let got_ids = tokenizer.encode(&prompt).map_err(|e| anyhow::anyhow!(e))?;
    let first_diff = got_ids.iter().zip(&want_ids).position(|(a, b)| a != b);
    report.check(
        "token_ids",
        got_ids == want_ids,
        json!({"ours": got_ids.len(), "first_difference_at": first_diff}),
        json!({"reference": want_ids.len(), "exact": true}),
    )?;

    // --- the decoder, on the reference's ids so arithmetic is judged alone ----
    let map = WeightMap::open(&weights.join("text_encoder"))?;
    let cfg = DecoderConfig::qwen3_vl_32b_text().for_bf16_reference();
    let tap = fastvideo_models::h3::config::H3TextEncoderConfig::fasth3_8step().output_hidden_state_index;
    let named: Vec<(usize, &str, f64)> = [(0, "text_h0", 0.0), (1, "text_h1", max_rel_layer0), (tap, "text", max_rel)]
        .into_iter()
        .filter(|(_, name, _)| orc.contains_key(*name))
        .collect();
    let taps: Vec<usize> = named.iter().map(|(k, _, _)| *k).collect();
    let positions: Vec<u32> = (0..want_ids.len() as u32).collect();
    let attend = vec![true; want_ids.len()];
    let (states, seconds) = measure(report, "text_forward", || {
        let out = fastvideo_cudarc::llm::hidden_states(&map, &cfg, &want_ids, &positions, &attend, &taps)?;
        out.iter().map(|t| Ok(t.host_cow()?.into_owned())).collect::<anyhow::Result<Vec<_>>>()
    })?;
    report.note("text_forward", json!({"seconds": seconds, "tokens": want_ids.len(), "layers_run": tap}));
    // Every metric lands before the first gate can stop the stage.
    let diffs: Vec<_> = named
        .iter()
        .zip(&states)
        .map(|((_, name, limit), got)| Ok((*name, *limit, diff(got, &st::take(&mut orc, name, oracle)?.data))))
        .collect::<anyhow::Result<_>>()?;
    for (name, limit, d) in &diffs {
        // The embedding gather has nothing to round: bf16 rows read as f32.
        let ok = if *limit == 0.0 { d.max_abs == 0.0 && d.non_finite == 0 } else { d.within(*limit) && d.cosine >= 0.999 };
        report.check(*name, ok, d.to_json(), json!({"rel_l2": limit, "cosine_min": 0.999}))?;
    }
    Ok(())
}

fn audio_vae(report: &mut Report, weights: &Path, oracle: &Path, device: &str, max_abs: f64) -> StageResult<()> {
    use fastvideo_cudarc::h3::audio_vae::H3AudioDecoder;
    use fastvideo_models::h3::config::H3AudioVaeConfig;

    report.set("device", crate::gpu::init(device)?);
    let mut orc = st::load(oracle)?;
    let latent = st::take(&mut orc, "audio_latent", oracle)?;
    let want = st::take(&mut orc, "audio_wave", oracle)?;
    drop(orc);

    let timer = std::time::Instant::now();
    let decoder = H3AudioDecoder::load(H3AudioVaeConfig::fasth3_8step(), &WeightMap::open(&weights.join("audio_vae"))?)?;
    report.note("load_audio_vae", json!({"seconds": timer.elapsed().as_secs_f64()}));

    let input = CudaTensor::from_vec(latent.data, latent.shape.clone())?;
    let (wave, seconds) = measure(report, "audio_decode", || {
        let out = decoder.decode(&input)?;
        Ok((out.shape.clone(), out.host_cow()?.into_owned()))
    })?;
    report.note("audio_decode", json!({"seconds": seconds, "latent": latent.shape, "wave": wave.0}));
    report.check("audio_wave_shape", wave.0 == want.shape, json!({"ours": wave.0}), json!({"reference": want.shape}))?;
    let d = diff(&wave.1, &want.data);
    // A clamp to [-1, 1] on both sides can hide an overdriven decode; say how much of it is railed.
    let railed = want.data.iter().filter(|v| v.abs() >= 1.0).count() as f64 / want.data.len().max(1) as f64;
    report.note("audio_wave_railed", json!({"fraction_of_reference_at_full_scale": railed}));
    report.check("audio_wave", d.non_finite == 0 && d.max_abs <= max_abs, d.to_json(), json!({"max_abs": max_abs}))?;
    Ok(())
}

fn vae(report: &mut Report, weights: &Path, oracle: &Path, device: &str, max_abs: f64, max_rel: f64) -> StageResult<()> {
    use fastvideo_cudarc::h3::vae::H3VideoDecoder;
    use fastvideo_models::h3::config::H3VideoVaeConfig;

    report.set("device", crate::gpu::init(device)?);
    let mut orc = st::load(oracle)?;
    let latent = st::take(&mut orc, "vae_latent", oracle)?;
    let want = st::take(&mut orc, "vae_video_raw", oracle)?;
    drop(orc);

    let timer = std::time::Instant::now();
    let decoder = H3VideoDecoder::load(H3VideoVaeConfig::fasth3_8step(), &WeightMap::open(&weights.join("vae"))?)?;
    report.note("load_vae", json!({"seconds": timer.elapsed().as_secs_f64()}));

    // The reference is [1, 3, F, H, W]; chunks arrive as [3, f, H, W] and are
    // placed by their frame offset.
    let [_, channels, frames, height, width] = want.shape[..] else {
        return Err(anyhow::anyhow!("vae_video_raw has shape {:?}, expected [1, 3, F, H, W]", want.shape).into());
    };
    let plane = height * width;
    let mut video = vec![f32::NAN; channels * frames * plane];
    let input = CudaTensor::from_vec(latent.data, latent.shape.clone())?;
    let mem = crate::gpu::PeakMem::start();
    let (emitted, seconds) = measure(report, "vae_decode", || {
        Ok(decoder.decode_raw_streaming(&input, &mut |offset, chunk| {
            let [c, f, h, w] = chunk.shape[..] else {
                return Err(fastvideo_cudarc::wan::tensor::TensorError::Message(format!("chunk shape {:?}", chunk.shape)));
            };
            if c != channels || h != height || w != width || offset + f > frames {
                return Err(fastvideo_cudarc::wan::tensor::TensorError::Message(format!(
                    "chunk {:?} at frame {offset} does not fit the reference {:?}",
                    chunk.shape, want.shape
                )));
            }
            let host = chunk.host_cow()?;
            for ch in 0..c {
                let dst = (ch * frames + offset) * plane;
                video[dst..dst + f * plane].copy_from_slice(&host[ch * f * plane..(ch + 1) * f * plane]);
            }
            Ok(())
        })?)
    })?;
    report.note("vae_decode", json!({"seconds": seconds, "frames": emitted, "latent": latent.shape, "peak_mib": mem.stop()}));
    report.check("vae_frames", emitted == frames, json!({"ours": emitted}), json!({"reference": frames}))?;
    let d = diff(&video, &want.data);
    // Where along time the error sits separates a chunk cross-fade bug from a ViT bug.
    let per_frame: Vec<f64> = (0..frames)
        .map(|f| {
            (0..channels)
                .flat_map(|ch| {
                    let base = (ch * frames + f) * plane;
                    video[base..base + plane].iter().zip(&want.data[base..base + plane])
                })
                .fold(0.0f64, |m, (a, b)| m.max(f64::from((a - b).abs())))
        })
        .collect();
    report.note("vae_max_abs_per_frame", json!({"values": per_frame}));
    report.check("vae_video_raw", d.non_finite == 0 && d.max_abs <= max_abs && d.within(max_rel), d.to_json(), json!({"max_abs": max_abs, "rel_l2": max_rel}))?;
    Ok(())
}

/// The oracle's request, rebuilt on our side and checked against the layout
/// tensors it saved, so a packing difference is named before any weight loads.
struct OracleRequest {
    layout: fastvideo_models::h3::packing::H3PackedLayout,
    /// `[Nv, 96]` patchified `video_noise`.
    video_rows: Vec<f32>,
    /// `[2 Na, 32]`.
    audio_rows: Vec<f32>,
    latent_shape: [usize; 4],
}

fn oracle_request(
    report: &mut Report,
    orc: &mut std::collections::HashMap<String, st::F32Tensor>,
    oracle: &Path,
    cfg: &fastvideo_models::h3::config::H3TransformerConfig,
    text_tokens: usize,
) -> StageResult<OracleRequest> {
    use fastvideo_models::h3::packing::{patchify, H3PackedLayout};
    use fastvideo_models::h3::schedule::H3JointSchedule;

    let video = st::take(orc, "video_noise", oracle)?;
    let audio = st::take(orc, "audio_noise", oracle)?;
    let [1, c, t, h, w] = video.shape[..] else {
        return Err(anyhow::anyhow!("video_noise has shape {:?}, expected [1, C, T, H, W]", video.shape).into());
    };
    if audio.shape.len() != 2 || audio.shape[0] % 2 != 0 || audio.shape[1] != cfg.audio_in_channels {
        return Err(anyhow::anyhow!("audio_noise has shape {:?}, expected [2 Na, {}]", audio.shape, cfg.audio_in_channels).into());
    }
    let layout = H3PackedLayout::new(text_tokens, (t, h, w), audio.shape[0] / 2, cfg.patch_size).map_err(|e| anyhow::anyhow!(e))?;

    let want_pos = st::take(orc, "position_ids", oracle)?;
    let ours: Vec<f32> = layout.position_ids.iter().flatten().map(|&p| p as f32).collect();
    let worst = ours.iter().zip(&want_pos.data).map(|(a, b)| f64::from((a - b).abs())).fold(0.0, f64::max);
    report.check(
        "position_ids",
        ours.len() == want_pos.data.len() && worst == 0.0,
        json!({"rows": layout.sequence_length(), "max_abs": worst}),
        json!({"rows": want_pos.shape.first(), "exact_as_f32": true}),
    )?;
    let want_tags = ints(&st::take(orc, "token_tags", oracle)?, "token_tags")?;
    let tags_ok = want_tags.len() == layout.token_tags.len() && want_tags.iter().zip(&layout.token_tags).all(|(a, b)| *a == u32::from(*b));
    report.check("token_tags", tags_ok, json!({"rows": layout.token_tags.len()}), json!({"exact": true}))?;

    let schedule = H3JointSchedule::fasth3_8step();
    for (name, ours) in [
        ("video_sigmas", &schedule.video.sigmas),
        ("audio_sigmas", &schedule.audio.sigmas),
        ("video_timesteps", &schedule.video.timesteps),
        ("audio_timesteps", &schedule.audio.timesteps),
    ] {
        let want = st::take(orc, name, oracle)?;
        let same = want.data.len() == ours.len() && want.data.iter().zip(ours.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        report.check(name, same, json!({"ours": ours}), json!({"reference": want.data, "bitwise": true}))?;
    }
    Ok(OracleRequest {
        video_rows: patchify(&video.data, [c, t, h, w], cfg.patch_size).map_err(|e| anyhow::anyhow!(e))?,
        audio_rows: audio.data,
        latent_shape: [c, t, h, w],
        layout,
    })
}

fn dit(report: &mut Report, weights: &Path, oracle: &Path, device: &str, max_rel: f64) -> StageResult<()> {
    use fastvideo_cudarc::h3::transformer::{AttnMode, DeviceLayout, H3TextRefiner, H3Transformer};
    use fastvideo_models::h3::config::{H3TransformerConfig, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
    use fastvideo_models::h3::schedule::H3JointSchedule;

    report.set("device", crate::gpu::init(device)?);
    let cfg = H3TransformerConfig::fasth3_8step();
    let hidden = cfg.hidden_size;
    let mut orc = st::load(oracle)?;
    let text = st::take(&mut orc, "text", oracle)?;
    let request = oracle_request(report, &mut orc, oracle, &cfg, text.shape[1])?;
    let layout = &request.layout;
    report.set("request", json!({"latent": request.latent_shape, "text_tokens": layout.text.len, "audio_rows": layout.audio.len, "video_rows": layout.video.len, "sequence": layout.sequence_length()}));

    // Which rung the oracle ran: its sorted-unique timesteps name it.
    let schedule = H3JointSchedule::fasth3_8step();
    let want_ts = st::take(&mut orc, "dit_timesteps", oracle)?;
    let step = (0..schedule.num_steps())
        .find(|&i| schedule.row_timesteps(i).is_ok_and(|r| r.timesteps.len() == want_ts.data.len() && r.timesteps.iter().zip(&want_ts.data).all(|(a, b)| a.to_bits() == b.to_bits())))
        .ok_or_else(|| anyhow::anyhow!("dit_timesteps {:?} match no rung of the FastH3 ladder", want_ts.data))?;
    let want_index = ints(&st::take(&mut orc, "dit_timestep_indices", oracle)?, "dit_timestep_indices")?;
    let ours_index = layout.timestep_indices(&schedule.row_timesteps(step).map_err(|e| anyhow::anyhow!(e))?);
    report.check(
        "timestep_indices",
        want_index.len() == ours_index.len() && want_index.iter().zip(&ours_index).all(|(a, b)| *a as usize == *b),
        json!({"step": step}),
        json!({"exact": true}),
    )?;

    let map = WeightMap::open(&weights.join("transformer"))?;
    let mem = crate::gpu::PeakMem::start();

    // --- text refiner, on the oracle's text ----------------------------------
    let text_dev = CudaTensor::from_vec(text.data, text.shape.clone())?;
    let (refined, seconds) = measure(report, "text_refiner", || {
        let refiner = H3TextRefiner::load(&cfg, &map)?;
        Ok(refiner.forward(&text_dev)?)
    })?;
    report.note("text_refiner", json!({"seconds": seconds}));
    let mut results = vec![("text_refined".to_string(), 5e-3, diff(&refined.host_cow()?, &st::take(&mut orc, "text_refined", oracle)?.data))];

    // --- load: AdaLN table, then the resident stack (dense: no gates) ---------
    let timer = std::time::Instant::now();
    let model = H3Transformer::load(cfg.clone(), &map, &schedule, false)?;
    report.note("load_dit", json!({"seconds": timer.elapsed().as_secs_f64()}));
    let table = model.adaln_table();
    // `temb` rows are the sorted-unique timesteps: video (smaller t) then audio.
    let want_temb = st::take(&mut orc, "temb", oracle)?;
    let ours_temb: Vec<f32> = table.temb[2 * step].iter().chain(&table.temb[2 * step + 1]).copied().collect();
    results.push(("temb".into(), 1e-5, diff(&ours_temb, &want_temb.data)));
    // adaln_block0 is [6 params, n_t * 3 rows, hidden]; a T2AV forward reads
    // rows 0 (t_video, video), 1 (t_video, text) and 5 (t_audio, audio).
    let want_adaln = st::take(&mut orc, "adaln_block0", oracle)?;
    let rows = want_adaln.shape.get(1).copied().unwrap_or(0);
    let (mut ours_adaln, mut ref_adaln) = (Vec::new(), Vec::new());
    for (tag, row) in [(TAG_VIDEO, 0usize), (TAG_TEXT, 1), (TAG_AUDIO, 5)] {
        let slot = table.block_slot(step, 0, tag);
        for p in 0..6 {
            // The table stores 1 + scale for the two scale parameters (1 and 4).
            let plus = if p == 1 || p == 4 { 1.0 } else { 0.0 };
            ours_adaln.extend(slot[p * hidden..(p + 1) * hidden].iter().map(|v| v - plus));
            ref_adaln.extend_from_slice(&want_adaln.data[(p * rows + row) * hidden..(p * rows + row + 1) * hidden]);
        }
    }
    results.push(("adaln_block0".into(), 2e-3, diff(&ours_adaln, &ref_adaln)));

    // --- one forward ------------------------------------------------------------
    let stride = orc
        .get("block_0")
        .map(|t| layout.sequence_length().div_ceil(t.shape[1].max(1)))
        .unwrap_or(16);
    let strided: Vec<usize> = (0..layout.sequence_length()).step_by(stride.max(1)).collect();
    let device_layout = DeviceLayout::new(&cfg, layout.clone())?;
    let video_rows = CudaTensor::from_vec(request.video_rows, vec![layout.video.len, cfg.video_patch_dim()])?;
    let audio_rows = CudaTensor::from_vec(request.audio_rows, vec![layout.audio.len, cfg.audio_in_channels])?;
    let wanted: Vec<String> = orc.keys().filter(|k| k.starts_with("block_")).cloned().collect();
    let mut dumps: Vec<(String, Vec<f32>)> = Vec::new();
    let ((video_v, audio_v), seconds) = measure(report, "dit_forward", || {
        let out = model.forward(
            step,
            &video_rows,
            &audio_rows,
            &refined,
            &device_layout,
            AttnMode::Dense,
            Some(&mut |name, x| {
                if wanted.iter().any(|w| w == name) {
                    let rows = x.reshape(vec![x.shape[1], x.shape[2]])?.index_select_rows(&strided)?;
                    dumps.push((name.to_string(), rows.host_cow()?.into_owned()));
                }
                Ok(())
            }),
        )?;
        Ok((out.0.host_cow()?.into_owned(), out.1.host_cow()?.into_owned()))
    })?;
    report.note("dit_forward", json!({"seconds": seconds, "peak_mib": mem.stop(), "block_row_stride": stride}));

    for (name, got) in &dumps {
        let index: usize = name.trim_start_matches("block_").parse().unwrap_or(0);
        // Divergence is allowed to grow with depth: bf16 on both sides.
        let limit = if index == 0 { 5e-3 } else if index < cfg.num_layers / 2 { 2e-2 } else { 5e-2 };
        results.push((name.clone(), limit, diff(got, &st::take(&mut orc, name, oracle)?.data)));
    }
    results.push(("dit_video".into(), max_rel, diff(&video_v, &st::take(&mut orc, "dit_video", oracle)?.data)));
    results.push(("dit_audio".into(), max_rel, diff(&audio_v, &st::take(&mut orc, "dit_audio", oracle)?.data)));

    // Every metric lands before the first gate can stop the stage: where the
    // error starts to grow is the diagnosis.
    report.set("metrics", results.iter().map(|(n, _, d)| (n.clone(), d.to_json())).collect::<serde_json::Map<_, _>>());
    for (name, limit, d) in &results {
        let cosine_min = if name.starts_with("dit_") { 0.999 } else { 0.0 };
        report.check(name.as_str(), d.within(*limit) && d.cosine >= cosine_min, d.to_json(), json!({"rel_l2": limit, "cosine_min": cosine_min}))?;
    }
    Ok(())
}

fn ladder(report: &mut Report, weights: &Path, oracle: &Path, device: &str, max_rel_first: f64, max_rel_last: f64) -> StageResult<()> {
    use fastvideo_cudarc::h3::pipeline::denoise;
    use fastvideo_cudarc::h3::transformer::{AttnMode, DeviceLayout, H3TextRefiner, H3Transformer};
    use fastvideo_models::h3::config::H3TransformerConfig;
    use fastvideo_models::h3::schedule::H3JointSchedule;

    report.set("device", crate::gpu::init(device)?);
    let cfg = H3TransformerConfig::fasth3_8step();
    let schedule = H3JointSchedule::fasth3_8step();
    let mut orc = st::load(oracle)?;
    let text = st::take(&mut orc, "text", oracle)?;
    let request = oracle_request(report, &mut orc, oracle, &cfg, text.shape[1])?;
    let want_video = st::take(&mut orc, "loop_video", oracle)?;
    let want_audio = st::take(&mut orc, "loop_audio", oracle)?;
    drop(orc);
    let layout = request.layout.clone();

    let map = WeightMap::open(&weights.join("transformer"))?;
    let refined = H3TextRefiner::load(&cfg, &map)?.forward(&CudaTensor::from_vec(text.data, text.shape.clone())?)?;
    let timer = std::time::Instant::now();
    let model = H3Transformer::load(cfg.clone(), &map, &schedule, false)?;
    report.note("load_dit", json!({"seconds": timer.elapsed().as_secs_f64()}));

    let device_layout = DeviceLayout::new(&cfg, layout.clone())?;
    let video_rows = CudaTensor::from_vec(request.video_rows, vec![layout.video.len, cfg.video_patch_dim()])?.to_device()?;
    let audio_rows = CudaTensor::from_vec(request.audio_rows, vec![layout.audio.len, cfg.audio_in_channels])?.to_device()?;
    let (nv, na) = (layout.video.len * cfg.video_patch_dim(), layout.audio.len * cfg.audio_in_channels);
    let mut steps = Vec::new();
    let mem = crate::gpu::PeakMem::start();
    let (_, seconds) = measure(report, "loop", || {
        let mut last = std::time::Instant::now();
        denoise(&model, &device_layout, &refined, video_rows, audio_rows, &schedule, AttnMode::Dense, &mut |step, video, audio| {
            let (dv, da) = (
                diff(&video.host_cow()?, &want_video.data[step * nv..(step + 1) * nv]),
                diff(&audio.host_cow()?, &want_audio.data[step * na..(step + 1) * na]),
            );
            eprintln!("[INFO] h3 loop step {step}: {:.1}s video rel {:.3e} audio rel {:.3e}", last.elapsed().as_secs_f64(), dv.rel_l2, da.rel_l2);
            last = std::time::Instant::now();
            steps.push((dv, da));
            Ok(())
        })?;
        Ok(())
    })?;
    report.note("loop", json!({"seconds": seconds, "peak_mib": mem.stop()}));
    report.set("per_step", steps.iter().map(|(v, a)| json!({"video": v.to_json(), "audio": a.to_json()})).collect::<Vec<_>>());
    let last = steps.len().saturating_sub(1);
    for (index, limit) in [(0usize, max_rel_first), (last, max_rel_last)] {
        let Some((v, a)) = steps.get(index) else {
            return Err(anyhow::anyhow!("the loop produced no step {index}").into());
        };
        report.check(format!("loop_video_step{index}"), v.within(limit), v.to_json(), json!({"rel_l2": limit}))?;
        report.check(format!("loop_audio_step{index}"), a.within(limit), a.to_json(), json!({"rel_l2": limit}))?;
    }
    Ok(())
}

fn vsa(report: &mut Report, device: &str, seed: u64, max_rel: f64) -> StageResult<()> {
    use fastvideo_cudarc::h3::vsa::{attention_host, H3Vsa, H3VsaConfig};
    use fastvideo_cudarc::wan::nn::scaled_dot_product_attention;
    use fastvideo_models::h3::packing::H3PackedLayout;

    report.set("device", crate::gpu::init(device)?);
    // Partial tiles on every axis, a short text tile, a short audio tile and an
    // odd tile count: 3 + 2 prefix tiles, 3 x 2 x 3 = 18 video tiles, n = 23.
    let layout = H3PackedLayout::new(150, (9, 12, 20), 50, [1, 2, 2]).map_err(|e| anyhow::anyhow!(e))?;
    // The tensor-core fine kernel is written for head dim 128.
    let (heads, dim, seq) = (4usize, 128usize, layout.sequence_length());
    let shape = vec![1, heads, seq, dim];
    let draw = |salt: u64, std: f32| crate::rand_weights::randn(seed ^ salt, heads * seq * dim, std);
    // Post-norm Q/K have unit RMS; that scale also keeps the top-k decisive.
    let (q, k, v, gate) = (draw(0x51, 1.0), draw(0x4b, 1.0), draw(0x56, 1.0), draw(0x47, 0.5));
    let tensor = |x: &[f32]| CudaTensor::from_vec(x.to_vec(), shape.clone());

    for (name, sparsity, gated) in [("sparse_gated", 0.8, true), ("sparse_ungated", 0.5, false), ("all_tiles_gated", 0.0, true)] {
        let vsa = H3Vsa::new(&layout, heads, dim, H3VsaConfig { sparsity, group: 4 })?;
        let plan = vsa.plan().clone();
        report.note(format!("{name}/plan"), json!({"prefix_tiles": plan.prefix_tiles, "video_tiles": plan.video_tiles, "k_vid": vsa.k_vid(), "rows": seq}));
        let want = attention_host(&q, &k, &v, gated.then_some(&gate[..]), &plan, vsa.k_vid(), heads, dim)?;
        let g = if gated { Some(tensor(&gate)?) } else { None };
        let (got, seconds) = measure(report, name, || Ok(vsa.attend(tensor(&q)?, tensor(&k)?, tensor(&v)?, g)?.host_cow()?.into_owned()))?;
        let d = diff(&got, &want);
        // Prefix rows take the dense splice, video rows the fine kernel: report them apart.
        let split = plan.prefix_rows * dim;
        let part = |lo: usize, hi: usize| -> Vec<f32> { (0..heads).flat_map(|h| got[h * seq * dim + lo..h * seq * dim + hi].to_vec()).collect() };
        let want_part = |lo: usize, hi: usize| -> Vec<f32> { (0..heads).flat_map(|h| want[h * seq * dim + lo..h * seq * dim + hi].to_vec()).collect() };
        report.note(
            format!("{name}/by_segment"),
            json!({"seconds": seconds, "prefix_rows": diff(&part(0, split), &want_part(0, split)).to_json(), "video_rows": diff(&part(split, seq * dim), &want_part(split, seq * dim)).to_json()}),
        );
        report.check(name, d.within(max_rel), d.to_json(), json!({"rel_l2": max_rel}))?;
    }

    // The load-bearing identity: every tile selected and no gate is dense attention.
    let vsa = H3Vsa::new(&layout, heads, dim, H3VsaConfig { sparsity: 0.0, group: 4 })?;
    let got = vsa.attend(tensor(&q)?, tensor(&k)?, tensor(&v)?, None)?.host_cow()?.into_owned();
    let dense = scaled_dot_product_attention(&tensor(&q)?, &tensor(&k)?, &tensor(&v)?, None)?.host_cow()?.into_owned();
    let d = diff(&got, &dense);
    report.check("all_tiles_ungated_is_dense", d.within(max_rel), d.to_json(), json!({"rel_l2": max_rel}))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gen(report: &mut Report, weights: &Path, prompt: &str, seconds: usize, seed: u64, dense: bool, mp4: bool, clip_dir: &Path, device: &str) -> StageResult<()> {
    use fastvideo_cudarc::h3::pipeline::{generate, H3Request};

    report.set("device", crate::gpu::init(device)?);
    let mut request = H3Request::seconds(prompt, seconds, seed).map_err(|e| anyhow::anyhow!(e))?;
    request.dense = dense;
    request.mp4 = mp4;
    report.set("request", json!({"prompt": prompt, "seconds": seconds, "seed": seed, "attention": if dense { "dense, no gate" } else { "vsa-h3 0.8 + to_gate_compress" }, "height": request.height, "width": request.width, "num_frames": request.num_frames}));

    let mem = crate::gpu::PeakMem::start();
    let timer = std::time::Instant::now();
    let out = generate(weights, &request, clip_dir)?;
    let total = timer.elapsed().as_secs_f64();
    let t = &out.timings;
    report.note(
        "timings",
        json!({
            "total_s": total,
            "text_s": t.text_s,
            "refine_s": t.refine_s,
            "load_dit_s": t.load_dit_s,
            "denoise_s": t.denoise_s,
            "step_s": t.step_s,
            "audio_decode_s": t.audio_decode_s,
            "load_vae_s": t.load_vae_s,
            "video_decode_s": t.video_decode_s,
            "write_s": t.write_s,
            "peak_mib": mem.stop(),
        }),
    );
    report.set("output", json!({"frames": out.frames, "text_tokens": out.text_tokens, "sequence_length": out.sequence_length, "mp4": out.mp4, "wav": out.wav, "audio_samples_per_channel": out.geometry.audio_samples()}));
    report.check("frames", out.frames == out.geometry.num_frames, json!({"decoded": out.frames}), json!({"expected": out.geometry.num_frames}))?;

    // A finite, non-constant picture: the cheapest statement that the clip is not garbage.
    let probe = out.frame_paths.get(out.frame_paths.len() / 2).ok_or_else(|| anyhow::anyhow!("no frames were written"))?;
    let img = image::open(probe).with_context(|| format!("open {probe}"))?.to_rgb8();
    let values: Vec<f32> = img.as_raw().iter().map(|&b| f32::from(b) / 255.0).collect();
    let (mean, std) = crate::metrics::mean_std(&values);
    report.check("middle_frame_not_flat", std > 0.02 && mean > 0.02 && mean < 0.98, json!({"mean": mean, "std": std, "frame": probe}), json!({"std_min": 0.02}))?;
    if mp4 {
        report.check("mp4_written", out.mp4.is_some(), json!({"mp4": out.mp4}), json!({"expected": "output.mp4 with an audio track"}))?;
    }
    Ok(())
}
