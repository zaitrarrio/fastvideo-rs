//! LTX-2 stages: each judges one part of the port against the reference dump
//! written by `scripts/gpu/ltx2_oracle.py` (see docs/ports/ltx2.md, section j).
//! Owned by the LTX-2 track; `main.rs` only dispatches here.
//!
//! Every stage feeds **our** code the **oracle's** input for that stage, so an
//! error belongs to the code the stage names and not to whatever ran before
//! it. All stages share one report name (`ltx2`); pass the global
//! `--tag <stage>` to keep their JSON files apart.
//!
//! Precision: the global `--mode exact` (the default) runs every GEMM in
//! float32, which is what the `_f32` oracle variants are compared under. The
//! 19B DiT does not fit 96 GB that way — its stages say so and want
//! `--mode fast` (bf16 weights, what production runs).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use fastvideo_cudarc::llm::DecoderConfig;
use fastvideo_cudarc::ltx2::audio_vae::AudioDecoder;
use fastvideo_cudarc::ltx2::keys::{Keys, Layout};
use fastvideo_cudarc::ltx2::text::{HiddenStack, PaddedPrompt, TextConnectors};
use fastvideo_cudarc::ltx2::vae::VideoDecoder;
use fastvideo_cudarc::ltx2::vocoder::Vocoder;
use fastvideo_cudarc::wan::pipeline::{interleave_audio, write_wav};
use fastvideo_cudarc::CudaTensor;
use fastvideo_cudarc::wan::weights::WeightMap;
use fastvideo_models::ltx2::config::{ltx2_19b_distilled, Ltx2Config};
use serde_json::json;

use crate::metrics::diff;
use crate::model::measure;
use crate::report::{Report, StageResult};
use crate::st::F32Tensor;

#[derive(clap::Subcommand, Debug)]
pub enum Stage {
    /// Print the configuration the port targets.
    Info,
    /// Tokenizer parity, Gemma's 49 hidden states, and both connector outputs.
    Text {
        /// `Lightricks/LTX-2` snapshot: reads `tokenizer/tokenizer.json` and
        /// `text_encoder/model-*-of-00011.safetensors`.
        #[arg(long)]
        weights: PathBuf,
        /// The distilled connectors: `ltx-2-19b-distilled.safetensors`, or a
        /// diffusers folder holding (or being) `connectors/`. The connectors
        /// under `--weights` are the *dev* model's and do not match a
        /// distilled oracle.
        #[arg(long)]
        dit: PathBuf,
        /// `--out` file of `ltx2_oracle.py` (stages `text` and `conn`).
        #[arg(long)]
        oracle: PathBuf,
        /// `--meta` file of `ltx2_oracle.py`, for the prompt. Defaults to the
        /// oracle path with a `.json` extension.
        #[arg(long)]
        meta: Option<PathBuf>,
        /// The prompt, when no meta file is at hand.
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Skip our Gemma forward (47 GB of float32 shards streamed once) and
        /// judge the connectors on the oracle's hidden states only.
        #[arg(long)]
        skip_llm: bool,
        /// Connector outputs against the oracle's float32 rerun.
        #[arg(long, default_value_t = 1e-3)]
        max_rel: f64,
        /// Our Gemma states against the bf16 reference, late layers.
        #[arg(long, default_value_t = 5e-2)]
        max_rel_llm: f64,
        /// Connector outputs when chained on *our* Gemma states (bf16 reference
        /// upstream, so this is a sanity bound, not a parity gate).
        #[arg(long, default_value_t = 0.15)]
        max_rel_e2e: f64,
    },
    /// The audio VAE decoder and the vocoder on the oracle's fixed latent,
    /// float32 on both sides (run with the default `--mode exact`).
    Audio {
        /// A diffusers LTX-2 snapshot: reads `audio_vae/` and `vocoder/`
        /// (identical in `Lightricks/LTX-2` and the distilled conversion).
        #[arg(long)]
        weights: PathBuf,
        /// `--out` file of `ltx2_oracle.py` (stage `audio`).
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Also write our waveform here as a 24 kHz stereo WAV, for ears.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// Log-mel, relative L2.
        #[arg(long, default_value_t = 1e-4)]
        max_rel_mel: f64,
        /// Waveform signal-to-error ratio on the oracle's mel, dB.
        #[arg(long, default_value_t = 60.0)]
        min_snr_db: f64,
        /// The same with our own mel upstream.
        #[arg(long, default_value_t = 50.0)]
        min_snr_db_e2e: f64,
    },
    /// The video VAE decoder on the oracle's fixed latent, float32 on both
    /// sides (default `--mode exact`), whole and streamed in small chunks.
    Vae {
        /// A diffusers LTX-2 snapshot: reads `vae/` (identical in
        /// `Lightricks/LTX-2` and the distilled conversion).
        #[arg(long)]
        weights: PathBuf,
        /// `--out` file of `ltx2_oracle.py` (stage `vae`).
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Pixels live in [-1, 1].
        #[arg(long, default_value_t = 1e-3)]
        max_abs: f64,
        #[arg(long, default_value_t = 1e-4)]
        max_rmse: f64,
        /// Streaming is the same arithmetic in another order of arrival; only
        /// cuDNN picking another algorithm for another shape may show here.
        #[arg(long, default_value_t = 1e-4)]
        max_abs_streamed: f64,
    },
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Info => {
            let c = ltx2_19b_distilled();
            report.set("config", format!("{c:?}"));
            Ok(())
        }
        Stage::Text { weights, dit, oracle, meta, prompt, device, skip_llm, max_rel, max_rel_llm, max_rel_e2e } => text(
            report,
            &TextArgs { weights, dit, oracle, meta: meta.as_deref(), prompt: prompt.as_deref(), device, skip_llm: *skip_llm },
            [*max_rel, *max_rel_llm, *max_rel_e2e],
        ),
        Stage::Audio { weights, oracle, device, wav, max_rel_mel, min_snr_db, min_snr_db_e2e } => {
            audio(report, weights, oracle, device, wav.as_deref(), [*max_rel_mel, *min_snr_db, *min_snr_db_e2e])
        }
        Stage::Vae { weights, oracle, device, max_abs, max_rmse, max_abs_streamed } => {
            vae(report, weights, oracle, device, [*max_abs, *max_rmse, *max_abs_streamed])
        }
    }
}

// ---- oracle file ------------------------------------------------------------

/// The tensors of the oracle file whose names start with one of `prefixes`, as
/// float32. `crate::st` reads float32 only and reads everything; the LTX-2 dump
/// carries int32 ids and masks and, with `--sample`, far more than one stage
/// wants in memory.
fn load_oracle(path: &Path, prefixes: &[&str]) -> anyhow::Result<HashMap<String, F32Tensor>> {
    use safetensors::{Dtype, SafeTensors};
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let file = SafeTensors::deserialize(&bytes).with_context(|| format!("parse safetensors {}", path.display()))?;
    let mut out = HashMap::new();
    for (name, view) in file.tensors() {
        if !prefixes.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        let word = |c: &[u8]| [c[0], c[1], c[2], c[3]];
        let data: Vec<f32> = match view.dtype() {
            Dtype::F32 => view.data().chunks_exact(4).map(|c| f32::from_le_bytes(word(c))).collect(),
            Dtype::I32 => view.data().chunks_exact(4).map(|c| i32::from_le_bytes(word(c)) as f32).collect(),
            other => anyhow::bail!("{}: tensor {name} is {other:?}, expected F32 or I32", path.display()),
        };
        out.insert(name, F32Tensor::new(view.shape().to_vec(), data)?);
    }
    Ok(out)
}

fn take(map: &mut HashMap<String, F32Tensor>, name: &str, path: &Path) -> anyhow::Result<F32Tensor> {
    map.remove(name).with_context(|| {
        format!("{} has no tensor `{name}` — was that ltx2_oracle.py stage skipped?", path.display())
    })
}

fn ints(t: &F32Tensor, name: &str) -> anyhow::Result<Vec<u32>> {
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

fn read_prompt(oracle: &Path, meta: Option<&Path>, prompt: Option<&str>) -> anyhow::Result<String> {
    if let Some(p) = prompt {
        return Ok(p.to_string());
    }
    let path = meta.map_or_else(|| oracle.with_extension("json"), Path::to_path_buf);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("no --prompt, and the oracle meta {} could not be read", path.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    doc.get("prompt")
        .and_then(|p| p.as_str())
        .map(str::to_string)
        .with_context(|| format!("{} has no `prompt`", path.display()))
}

// ---- weights ----------------------------------------------------------------

/// The distilled DiT/connectors: the single file, or a diffusers component
/// folder named `component` under (or at) `path`.
fn open_distilled(path: &Path, component: &str) -> anyhow::Result<(WeightMap, Layout)> {
    let map = if path.is_file() {
        WeightMap::open_files(&[path.to_path_buf()])?
    } else if path.join(component).is_dir() {
        WeightMap::open(&path.join(component))?
    } else if path.is_dir() {
        WeightMap::open(path)?
    } else {
        anyhow::bail!("{} is neither a .safetensors file nor a directory", path.display());
    };
    let layout = Keys::detect(&map);
    Ok((map, layout))
}

fn gib(map: &WeightMap, prefix: &str) -> f64 {
    map.lazy().map_or(0.0, |s| s.bytes_with_prefix(prefix) as f64 / f64::from(1u32 << 30))
}

// ---- text -------------------------------------------------------------------

struct TextArgs<'a> {
    weights: &'a Path,
    dit: &'a Path,
    oracle: &'a Path,
    meta: Option<&'a Path>,
    prompt: Option<&'a str>,
    device: &'a str,
    skip_llm: bool,
}

/// Rows `[total - real, total)` of a `[1, total, width]` reference: where the
/// left-padded pipeline keeps what we compute for the real tokens alone.
fn tail_rows(t: &F32Tensor, real: usize) -> anyhow::Result<&[f32]> {
    let [_, total, width] = t.shape[..] else { anyhow::bail!("expected [1, S, C], got {:?}", t.shape) };
    anyhow::ensure!(real <= total, "{real} real tokens in a reference of {total}");
    Ok(&t.data[(total - real) * width..])
}

fn text(report: &mut Report, a: &TextArgs<'_>, [max_rel, max_rel_llm, max_rel_e2e]: [f64; 3]) -> StageResult<()> {
    report.set("device", crate::gpu::init(a.device)?);
    let cfg: Ltx2Config = ltx2_19b_distilled();
    let max_len = cfg.defaults.max_sequence_length;
    let mut orc = load_oracle(a.oracle, &["text.", "conn."])?;

    // --- tokenizer parity: ours on the prompt vs the pipeline's padded ids ----
    let want_ids = ints(&take(&mut orc, "text.input_ids", a.oracle)?, "text.input_ids")?;
    let want_mask = ints(&take(&mut orc, "text.attention_mask", a.oracle)?, "text.attention_mask")?;
    let prompt = read_prompt(a.oracle, a.meta, a.prompt)?;
    let ours = PaddedPrompt::tokenize(&a.weights.join("tokenizer").join("tokenizer.json"), &prompt, max_len)?;
    let first_diff = ours.ids.iter().zip(&want_ids).position(|(x, y)| x != y);
    report.check(
        "token_ids",
        ours.ids == want_ids && ours.attention_mask() == want_mask,
        json!({"ours_real": ours.real, "ours_len": ours.ids.len(), "first_difference_at": first_diff}),
        json!({"reference_real": want_mask.iter().sum::<u32>(), "reference_len": want_ids.len(), "exact": true}),
    )?;
    // From here on the reference's own ids are used, so a tokenizer difference
    // under --keep-going cannot leak into the arithmetic checks.
    let real = want_mask.iter().sum::<u32>() as usize;
    let reference_prompt = PaddedPrompt::from_ids(&want_ids[want_ids.len() - real..], want_ids.len())?;

    let states = take(&mut orc, "text.hidden_states", a.oracle)?;
    let [n, hidden, depth] = states.shape[..] else {
        return Err(anyhow::anyhow!("text.hidden_states: expected [tokens, hidden, states], got {:?}", states.shape).into());
    };
    if n != real {
        return Err(anyhow::anyhow!("text.hidden_states holds {n} tokens, the mask says {real}").into());
    }
    let oracle_stack = HiddenStack::from_interleaved(&states.data, n, hidden, depth)?;
    drop(states);

    // --- connectors on the oracle's hidden states -----------------------------
    let (map, layout) = open_distilled(a.dit, "connectors")?;
    let keys = Keys::connectors(layout);
    report.set("connectors", json!({"source": a.dit.display().to_string(), "layout": format!("{layout:?}")}));
    let timer = std::time::Instant::now();
    let connectors = TextConnectors::load(&map, &keys, &cfg.connectors)?;
    report.note("load_connectors", json!({"seconds": timer.elapsed().as_secs_f64()}));

    let host = |t: &fastvideo_cudarc::CudaTensor| -> anyhow::Result<Vec<f32>> { Ok(t.host_cow()?.into_owned()) };
    let (got, seconds) = measure(report, "connectors", || {
        let out = connectors.forward(&oracle_stack, max_len)?;
        Ok((host(&out.proj)?, host(&out.video)?, host(&out.audio)?))
    })?;
    report.note("connectors", json!({"seconds": seconds, "tokens": n, "padded_to": max_len}));

    let proj_ref = take(&mut orc, "conn.proj", a.oracle)?;
    let refs: Vec<(&str, F32Tensor, F32Tensor)> = vec![
        ("video", take(&mut orc, "conn.video", a.oracle)?, take(&mut orc, "conn.video_f32", a.oracle)?),
        ("audio", take(&mut orc, "conn.audio", a.oracle)?, take(&mut orc, "conn.audio_f32", a.oracle)?),
    ];
    // Every metric lands before the first gate can stop the stage.
    let d_proj = diff(&got.0, tail_rows(&proj_ref, n)?);
    let pairs: Vec<_> = refs
        .iter()
        .zip([&got.1, &got.2])
        .map(|((name, bf16, f32_), ours)| (*name, diff(ours, &f32_.data), diff(ours, &bf16.data), diff(&bf16.data, &f32_.data)))
        .collect();
    report.set(
        "metrics",
        json!({
            "proj_vs_bf16": d_proj.to_json(),
            "video": {"vs_f32": pairs[0].1.to_json(), "vs_bf16": pairs[0].2.to_json(), "oracle_bf16_vs_f32": pairs[0].3.to_json()},
            "audio": {"vs_f32": pairs[1].1.to_json(), "vs_bf16": pairs[1].2.to_json(), "oracle_bf16_vs_f32": pairs[1].3.to_json()},
        }),
    );
    // The only projection reference is the pipeline's bf16 one, whose
    // normalisation sums ~n·3840 values in bf16: informational.
    report.note("conn.proj_vs_bf16", d_proj.to_json());
    for (name, vs_f32, vs_bf16, floor) in &pairs {
        report.note(format!("conn.{name}_oracle_bf16_vs_f32"), floor.to_json());
        report.note(format!("conn.{name}_vs_bf16"), vs_bf16.to_json());
        report.check(
            format!("conn.{name}_vs_f32"),
            vs_f32.within(max_rel) && vs_f32.cosine >= 0.9999,
            vs_f32.to_json(),
            json!({"rel_l2": max_rel, "cosine_min": 0.9999}),
        )?;
    }
    if a.skip_llm {
        return Ok(());
    }

    // --- Gemma on the reference's ids, then the chain on our own states -------
    let text_dir = a.weights.join("text_encoder");
    let gemma = WeightMap::open(&text_dir)?;
    report.set("text_encoder", json!({"language_model_gib": gib(&gemma, "language_model."), "tokens": real}));
    // The oracle's text stage runs bf16 unless told otherwise; its embedding
    // multiplier is then 62.0 rather than sqrt(3840).
    let llm_cfg = DecoderConfig::gemma3_12b_text().for_bf16_reference();
    let (our_stack, seconds) = measure(report, "gemma", || Ok(HiddenStack::encode(&gemma, &llm_cfg, &reference_prompt)?))?;
    report.note("gemma", json!({"seconds": seconds, "layers": llm_cfg.num_layers(), "tokens": real}));
    let per_state: Vec<_> = our_stack.states.iter().zip(&oracle_stack.states).map(|(o, r)| diff(o, r)).collect();
    report.set("gemma_rel_l2_by_state", per_state.iter().map(|d| d.rel_l2).collect::<Vec<_>>());
    for k in [0usize, 1, 6, 24, 47, 48] {
        let Some(d) = per_state.get(k) else { continue };
        report.check(format!("text.hidden_{k}"), d.within(max_rel_llm), d.to_json(), json!({"rel_l2": max_rel_llm}))?;
    }

    let (e2e, _) = measure(report, "connectors_e2e", || {
        let out = connectors.forward(&our_stack, max_len)?;
        Ok((host(&out.video)?, host(&out.audio)?))
    })?;
    for ((name, bf16, _), ours) in refs.iter().zip([&e2e.0, &e2e.1]) {
        let d = diff(ours, &bf16.data);
        report.check(
            format!("e2e.{name}_vs_bf16"),
            d.within(max_rel_e2e) && d.cosine >= 0.99,
            d.to_json(),
            json!({"rel_l2": max_rel_e2e, "cosine_min": 0.99}),
        )?;
    }
    Ok(())
}

// ---- audio ------------------------------------------------------------------

/// Root-mean-square error and signal-to-error ratio in dB, from a [`diff`].
fn rmse_snr(d: &crate::metrics::Diff, n: usize) -> (f64, f64) {
    let rmse = d.rel_l2 * d.ref_norm / (n.max(1) as f64).sqrt();
    let snr = if d.rel_l2 > 0.0 { -20.0 * d.rel_l2.log10() } else { f64::INFINITY };
    (rmse, snr)
}

fn audio(report: &mut Report, weights: &Path, oracle: &Path, device: &str, wav: Option<&Path>, [max_rel_mel, min_snr, min_snr_e2e]: [f64; 3]) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let cfg = ltx2_19b_distilled();
    let mut orc = load_oracle(oracle, &["audio."])?;
    let latent = take(&mut orc, "audio.latent", oracle)?;
    let want_mel = take(&mut orc, "audio.mel", oracle)?;
    let want_wave = take(&mut orc, "audio.wave", oracle)?;
    drop(orc);

    let timer = std::time::Instant::now();
    let decoder = AudioDecoder::load(&WeightMap::open(&weights.join("audio_vae"))?, &cfg.audio_vae)?;
    let vocoder = Vocoder::load(&WeightMap::open(&weights.join("vocoder"))?, &cfg.vocoder)?;
    report.note("load_audio", json!({"seconds": timer.elapsed().as_secs_f64()}));

    let packed = CudaTensor::from_vec(latent.data, latent.shape.clone())?;
    let (mel, seconds) = measure(report, "audio_vae", || Ok(decoder.decode_packed(&packed)?))?;
    report.note("audio_vae", json!({"seconds": seconds, "latent": latent.shape, "mel": mel.shape}));
    let reference_mel = CudaTensor::from_vec(want_mel.data.clone(), want_mel.shape.clone())?;
    let (wave, seconds) = measure(report, "vocoder", || Ok(vocoder.forward(&reference_mel)?))?;
    report.note("vocoder", json!({"seconds": seconds, "wave": wave.shape, "sample_rate": vocoder.sample_rate()}));
    let (wave_e2e, _) = measure(report, "vocoder_e2e", || Ok(vocoder.forward(&mel)?))?;

    let shape_ok = mel.shape == want_mel.shape && wave.shape == want_wave.shape;
    report.check(
        "audio.shapes",
        shape_ok,
        json!({"mel": mel.shape, "wave": wave.shape}),
        json!({"mel": want_mel.shape, "wave": want_wave.shape}),
    )?;
    let (mel_h, wave_h, e2e_h) = (mel.host_cow()?, wave.host_cow()?, wave_e2e.host_cow()?);
    // Every metric lands before the first gate can stop the stage.
    let d_mel = diff(&mel_h, &want_mel.data);
    let d_wave = diff(&wave_h, &want_wave.data);
    let d_e2e = diff(&e2e_h, &want_wave.data);
    let with = |d: &crate::metrics::Diff, n: usize| {
        let (rmse, snr) = rmse_snr(d, n);
        let mut v = d.to_json();
        v["rmse"] = json!(rmse);
        v["snr_db"] = json!(if snr.is_finite() { snr } else { 999.0 });
        v
    };
    if let Some(path) = wav {
        write_wav(path, &interleave_audio(&e2e_h, cfg.vocoder.out_channels)?, cfg.vocoder.out_channels as u16, cfg.vocoder.output_sampling_rate as u32)?;
        report.set("wav", path.display().to_string());
    }
    report.check("audio.mel", d_mel.within(max_rel_mel), with(&d_mel, mel_h.len()), json!({"rel_l2": max_rel_mel}))?;
    let snr = |d: &crate::metrics::Diff| rmse_snr(d, 1).1;
    report.check("audio.wave", d_wave.non_finite == 0 && snr(&d_wave) >= min_snr, with(&d_wave, wave_h.len()), json!({"snr_db_min": min_snr}))?;
    report.check("e2e.wave", d_e2e.non_finite == 0 && snr(&d_e2e) >= min_snr_e2e, with(&d_e2e, e2e_h.len()), json!({"snr_db_min": min_snr_e2e}))?;
    Ok(())
}

// ---- video vae --------------------------------------------------------------

fn vae(report: &mut Report, weights: &Path, oracle: &Path, device: &str, [max_abs, max_rmse, max_abs_streamed]: [f64; 3]) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let cfg = ltx2_19b_distilled();
    let mut orc = load_oracle(oracle, &["vae."])?;
    let latent = take(&mut orc, "vae.latent", oracle)?;
    let want = take(&mut orc, "vae.video", oracle)?;
    drop(orc);

    let timer = std::time::Instant::now();
    let map = WeightMap::open(&weights.join("vae"))?;
    let decoder = VideoDecoder::load(&map, &cfg.vae)?;
    report.note("load_vae", json!({"seconds": timer.elapsed().as_secs_f64(), "decoder_gib": gib(&map, "decoder.")}));

    let z = CudaTensor::from_vec(latent.data, latent.shape.clone())?;
    // [frames, 3, H, W] runs → one [3, F, H, W] host buffer, the oracle's order.
    let decode = |chunk: usize| -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (mut runs, mut pieces) = (Vec::new(), Vec::new());
        decoder.decode_streaming_chunked(&z, chunk, &mut |_, frames| {
            runs.push(frames.shape[0]);
            pieces.push(frames.clone());
            Ok(())
        })?;
        let all = CudaTensor::cat(&pieces.iter().collect::<Vec<_>>(), 0)?.permute(&[1, 0, 2, 3])?;
        Ok((all.host_cow()?.into_owned(), runs))
    };
    // Warm-up: cuDNN plan search stays out of the timing.
    decode(usize::MAX)?;
    let ((whole, runs), seconds) = measure(report, "vae_decode", || decode(usize::MAX))?;
    report.note("vae_decode", json!({"seconds": seconds, "latent": latent.shape, "runs": runs}));
    let ((streamed, runs), seconds) = measure(report, "vae_decode_streamed", || decode(2))?;
    report.note("vae_decode_streamed", json!({"seconds": seconds, "chunk": 2, "runs": runs}));

    let d = diff(&whole, &want.data);
    let (rmse, _) = rmse_snr(&d, whole.len());
    let mut values = d.to_json();
    values["rmse"] = json!(rmse);
    values["psnr_db"] = json!(crate::metrics::psnr(&whole, &want.data, 2.0).min(999.0));
    let d_stream = diff(&streamed, &whole);
    report.set("metrics", json!({"video": values, "streamed_vs_whole": d_stream.to_json()}));
    report.check(
        "vae.video",
        d.non_finite == 0 && whole.len() == want.data.len() && d.max_abs <= max_abs && rmse <= max_rmse,
        values,
        json!({"max_abs": max_abs, "rmse": max_rmse, "shape": want.shape}),
    )?;
    report.check(
        "vae.streamed_equals_whole",
        d_stream.non_finite == 0 && d_stream.max_abs <= max_abs_streamed,
        d_stream.to_json(),
        json!({"max_abs": max_abs_streamed}),
    )?;
    Ok(())
}
