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
