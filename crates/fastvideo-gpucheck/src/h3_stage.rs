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
        /// `hidden_states[50]`: bf16 on both sides through 50 layers (measured
        /// 1.37e-2). With `--precision fp8` the default gate is 5e-2 instead.
        #[arg(long)]
        max_rel: Option<f64>,
        /// Run the same checks through a RESIDENT encoder at this weight
        /// precision (`native` = the checkpoint's bf16, `fp8` = weight-only
        /// E4M3 rows) instead of the streamed one. For `fp8` the stage also
        /// encodes with our own native path and reports FP8 against it, so the
        /// quantization error is separated from the bf16-reference floor.
        #[arg(long)]
        precision: Option<String>,
        /// `hidden_states[1]`: one layer. Measured 4.35e-3 (cosine 0.99999) on an
        /// RTX PRO 6000 against transformers 5.17, falling to 1.9e-3 by layer 8
        /// before growing slowly to 1.4e-2 at layer 50. An error that shrinks
        /// with depth is first-block bf16 rounding (the reference is bf16 end to
        /// end, norms and softmax included; we keep f32 activations around bf16
        /// GEMMs, and the embedding-scale stream is where that differs most),
        /// not a defect that compounds. The gate is set above that floor; the
        /// number is recorded either way.
        #[arg(long, default_value_t = 1e-2)]
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
        /// Worst pixel (ImageNet-normalized RGB, before the clamp). Calibrated on
        /// the TF32-free, math-SDPA reference: measured 3.2e-5 at rel 1.5e-6.
        /// (Against an unpinned reference it was 8.8e-3: that was the
        /// reference's TF32, not the port.)
        #[arg(long, default_value_t = 2e-4)]
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
        /// Gate on the first `gated_steps` steps, where the comparison still
        /// measures the port. Measured on an RTX PRO 6000 (dense, bf16 reference),
        /// video: 2.4e-4, 2.2e-3, 1.0e-2, 2.2e-2, 4.4e-2, 8.8e-2, 2.0e-1, 4.0e-1;
        /// audio 4.5e-4 .. 2.6e-1. The error roughly doubles per step on both
        /// modalities, and LTX-2 shows the same against its bf16 reference: a
        /// few-step distilled sampler feeds each forward's rounding-level
        /// divergence (1.3e-2 on one forward) back in while the carried-noise
        /// share `sigma'/sigma` shrinks to zero, so two correct bf16
        /// implementations end on different, equally valid samples. A sign,
        /// shift or ratio mistake shows at step 0-2 (and is pinned exactly by
        /// the CPU reference test against diffusers' scheduler); later steps are
        /// recorded, not gated.
        #[arg(long, default_value_t = 3e-2)]
        max_rel: f64,
        #[arg(long, default_value_t = 3)]
        gated_steps: usize,
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
        /// 5 to 15. Ignored when `--num-frames` is given.
        #[arg(long, default_value_t = 5)]
        seconds: usize,
        /// Pixel frames, aligned up to `17 n + 5` and held to 5..15 s at 24
        /// fps (124 is the shortest clip; the GB10 480p cell runs 124).
        #[arg(long)]
        num_frames: Option<usize>,
        /// Canvas height in pixels (default 768). With `--width`: multiples of
        /// 32, at most 768 x 1344 pixels, aspect 1:4 to 4:1 (832x480 is the
        /// 480p cell).
        #[arg(long)]
        height: Option<usize>,
        /// Canvas width in pixels (default 1344).
        #[arg(long)]
        width: Option<usize>,
        /// Refiner and DiT block residency: `auto` (resident when the free
        /// memory covers the planned need, else streamed), `resident`, or
        /// `streamed` (blocks copied from pinned host memory one ahead of the
        /// computing block; decoders loaded for the decode only). Default:
        /// `FASTVIDEO_DIT_OFFLOAD`, else `auto`.
        #[arg(long)]
        dit_offload: Option<String>,
        /// Run as if the card had only this many GiB (e.g. 32 on a 96 GB card
        /// to emulate an RTX 5090): every auto policy decides for that card,
        /// the rest is held back, and a phase peaking above it fails the run.
        /// Default: `FASTVIDEO_DEVICE_BUDGET_GIB`.
        #[arg(long)]
        device_budget_gib: Option<f64>,
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
        /// Memoize the precomputed AdaLN table here (155 MB); a second run then
        /// skips reading 26 GB of projections.
        #[arg(long)]
        adaln_cache: Option<PathBuf>,
        /// Conditioning cache directory. Default: `text-cache` next to the clip
        /// directory (e.g. `/workspace/text-cache`). A prompt seen before then
        /// costs a file read instead of ~10 s of weight streaming.
        #[arg(long)]
        text_cache: Option<PathBuf>,
        #[arg(long)]
        no_text_cache: bool,
        /// Root holding `tokenizer/` + `text_encoder/` when not `--weights`
        /// itself, e.g. the output of `h3 slim-text`.
        #[arg(long)]
        text_weights: Option<PathBuf>,
        /// `auto` (resident-fp8 when >= 85 GB are free before anything loads,
        /// else streamed), `streamed`, `resident-fp8`, `resident-bf16`, or
        /// `recovered-8b` (SearchingMan Qwen3-VL-8B + ARA + adapter).
        #[arg(long, default_value = "auto")]
        text_encoder: String,
        /// With a resident encoder: also encode the prompt by streaming (~10 s,
        /// once, cache bypassed) and report rel_l2 / cosine between the two
        /// conditionings: what quantizing the encoder does to THIS prompt.
        #[arg(long)]
        compare_text_encoders: bool,
        /// Run one untimed generation first, so the reported numbers are a warm
        /// process: weights resident, allocator grown, kernels compiled, the
        /// conditioning cached. The cold run's numbers are recorded beside them.
        #[arg(long)]
        warm: bool,
        /// Directory or `taeh3.safetensors` file. Replaces the official ViT
        /// decoder; the `vae/` snapshot is then unused.
        #[arg(long)]
        taeh3_weights: Option<PathBuf>,
        /// DMD recipe: `8step` / `v2`, `4step-vsa` / `preview-vsa`, `4step-dense`
        /// / `preview-dense`, `sol-h3` (dense on one GPU; `FASTVIDEO_H3_SOL_ATTN=1`
        /// for the engine Sol policy), `sol-h3-spark`, `sol-h3-rtx`. Default:
        /// read `fastvideo_inference.json` or 8-step.
        #[arg(long)]
        h3_recipe: Option<String>,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
    /// CPU only: re-pack the text encoder to exactly what tap 50 reads
    /// (embed_tokens + layers 0..=49, bf16 verbatim, in load order) plus a copy
    /// of `tokenizer/`. The output is a root `--text-weights` accepts.
    SlimText {
        /// Root of the FastH3 snapshot (`tokenizer/`, `text_encoder/`).
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Shard size in GiB; shards break only between layers.
        #[arg(long, default_value_t = 5)]
        shard_gib: u64,
    },
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Info => {
            let c = fastvideo_models::h3::config::H3InferenceContract::fasth3_8step();
            report.set("contract", format!("{c:?}"));
            Ok(())
        }
        Stage::Text {
            weights,
            oracle,
            meta,
            prompt,
            device,
            max_rel,
            max_rel_layer0,
            precision,
        } => {
            let precision = match precision.as_deref() {
                None => None,
                Some("native") => Some(fastvideo_cudarc::llm::WeightPrecision::Native),
                Some("fp8") => Some(fastvideo_cudarc::llm::WeightPrecision::Fp8Rows),
                Some(other) => {
                    return Err(
                        anyhow::anyhow!("unknown --precision '{other}' (native|fp8)").into(),
                    )
                }
            };
            let fp8 = precision == Some(fastvideo_cudarc::llm::WeightPrecision::Fp8Rows);
            let max_rel = max_rel.unwrap_or(if fp8 { 5e-2 } else { 2e-2 });
            // One quantized layer has no 50-layer average to hide in; its own gate scales with the tap's.
            let layer0 = if fp8 {
                max_rel_layer0.max(max_rel)
            } else {
                *max_rel_layer0
            };
            text(
                report,
                weights,
                oracle,
                meta.as_deref(),
                prompt.as_deref(),
                device,
                max_rel,
                layer0,
                precision,
            )
        }
        Stage::AudioVae {
            weights,
            oracle,
            device,
            max_abs,
        } => audio_vae(report, weights, oracle, device, *max_abs),
        Stage::Vae {
            weights,
            oracle,
            device,
            max_abs,
            max_rel,
        } => vae(report, weights, oracle, device, *max_abs, *max_rel),
        Stage::Dit {
            weights,
            oracle,
            device,
            max_rel,
        } => dit(report, weights, oracle, device, *max_rel),
        Stage::Loop {
            weights,
            oracle,
            device,
            max_rel,
            gated_steps,
        } => ladder(report, weights, oracle, device, *max_rel, *gated_steps),
        Stage::Vsa {
            device,
            seed,
            max_rel,
        } => vsa(report, device, *seed, *max_rel),
        Stage::Gen {
            weights,
            prompt,
            seconds,
            num_frames,
            height,
            width,
            dit_offload,
            device_budget_gib,
            seed,
            dense,
            no_mp4,
            clip_dir,
            adaln_cache,
            text_cache,
            no_text_cache,
            text_weights,
            text_encoder,
            compare_text_encoders,
            warm,
            taeh3_weights,
            h3_recipe,
            device,
        } => {
            let text_cache = if *no_text_cache {
                None
            } else {
                Some(text_cache.clone().unwrap_or_else(|| {
                    clip_dir
                        .parent()
                        .unwrap_or(Path::new("."))
                        .join("text-cache")
                }))
            };
            let options = fastvideo_cudarc::h3::pipeline::H3PipelineOptions {
                dense: *dense,
                adaln_cache: adaln_cache.clone(),
                text_root: text_weights.clone(),
                text_cache,
                text_encoder: fastvideo_cudarc::h3::pipeline::TextEncoderChoice::parse(
                    text_encoder,
                )
                .map_err(|e| anyhow::anyhow!(e))?,
                taeh3: taeh3_weights.clone(),
                recipe: h3_recipe.clone(),
                ref2va: false,
                adapter: None,
                reference_image_resize: Default::default(),
                dit_offload: dit_offload
                    .as_deref()
                    .map(fastvideo_cudarc::wan::offload::DitOffload::parse)
                    .transpose()
                    .map_err(|e| anyhow::anyhow!(e))?,
            };
            let canvas = GenCanvas {
                seconds: *seconds,
                num_frames: *num_frames,
                height: *height,
                width: *width,
            };
            gen(
                report,
                weights,
                prompt,
                canvas,
                *seed,
                !*no_mp4,
                clip_dir,
                options,
                *warm,
                *compare_text_encoders,
                device,
                *device_budget_gib,
            )
        }
        Stage::SlimText {
            weights,
            out,
            shard_gib,
        } => slim_text(report, weights, out, *shard_gib),
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
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(Some(
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?,
    ))
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
    precision: Option<fastvideo_cudarc::llm::WeightPrecision>,
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let mut orc = st::load(oracle)?;
    let want_ids = ints(&st::take(&mut orc, "text_ids", oracle)?, "text_ids")?;
    let meta = read_meta(oracle, meta)?;
    if meta
        .as_ref()
        .is_some_and(|m| m.get("text_is_synthetic").is_some())
    {
        return Err(anyhow::anyhow!(
            "{}: `text` is a synthetic stand-in; rerun h3_oracle.py with the text stage",
            oracle.display()
        )
        .into());
    }

    // --- tokenizer parity: ours on the prompt vs the reference's ids ---------
    let prompt = match (
        prompt,
        meta.as_ref()
            .and_then(|m| m.get("prompt"))
            .and_then(|p| p.as_str()),
    ) {
        (Some(p), _) => p.to_string(),
        (None, Some(p)) => p.to_string(),
        (None, None) => {
            return Err(
                anyhow::anyhow!("no prompt: pass --meta <h3_oracle meta json> or --prompt").into(),
            )
        }
    };
    let tokenizer = fastvideo_models::h3::tokenizer::H3Tokenizer::from_file(
        &weights.join("tokenizer").join("tokenizer.json"),
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    let ours: serde_json::Map<String, serde_json::Value> = tokenizer
        .added_special_token_ids()
        .iter()
        .map(|(t, id)| (t.clone(), json!(id)))
        .collect();
    let reference = meta
        .as_ref()
        .and_then(|m| m.get("added_special_token_ids"))
        .cloned();
    let added_ok = match &reference {
        Some(serde_json::Value::Object(r)) => {
            !r.is_empty() && r.iter().all(|(t, id)| ours.get(t) == Some(id))
        }
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
    let tap =
        fastvideo_models::h3::config::H3TextEncoderConfig::fasth3_8step().output_hidden_state_index;
    let named: Vec<(usize, &str, f64)> = [
        (0, "text_h0", 0.0),
        (1, "text_h1", max_rel_layer0),
        (tap, "text", max_rel),
    ]
    .into_iter()
    .filter(|(_, name, _)| orc.contains_key(*name))
    .collect();
    let taps: Vec<usize> = named.iter().map(|(k, _, _)| *k).collect();
    let positions: Vec<u32> = (0..want_ids.len() as u32).collect();
    let attend = vec![true; want_ids.len()];
    let resident = match precision {
        Some(p) => {
            let timer = std::time::Instant::now();
            let decoder = fastvideo_cudarc::llm::ResidentDecoder::load_with(&map, &cfg, tap, p)?;
            report.note("resident_encoder", json!({"precision": format!("{p:?}"), "load_seconds": timer.elapsed().as_secs_f64(), "device_gib": decoder.device_bytes() as f64 / f64::from(1u32 << 30)}));
            Some(decoder)
        }
        None => None,
    };
    let (states, seconds) = measure(report, "text_forward", || {
        let out = match &resident {
            Some(decoder) => decoder.hidden_states(&want_ids, &positions, &attend, &taps)?,
            None => fastvideo_cudarc::llm::hidden_states(
                &map, &cfg, &want_ids, &positions, &attend, &taps,
            )?,
        };
        out.iter()
            .map(|t| Ok(t.host_cow()?.into_owned()))
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    report.note("text_forward", json!({"seconds": seconds, "tokens": want_ids.len(), "layers_run": tap, "encoder": if resident.is_some() { "resident" } else { "streamed" }}));
    // Against OUR native path: for fp8 this is the quantization error alone;
    // for native it must be zero (one layer loop, weights merely left in place).
    if let Some(decoder) = resident {
        drop(decoder);
        let ours = fastvideo_cudarc::llm::hidden_states(
            &map, &cfg, &want_ids, &positions, &attend, &taps,
        )?;
        for (((_, name, _), got), native) in named.iter().zip(&states).zip(&ours) {
            let d = diff(got, &native.host_cow()?);
            report.note(format!("{name}/vs_our_native"), d.to_json());
            if precision == Some(fastvideo_cudarc::llm::WeightPrecision::Native) {
                report.check(
                    format!("{name}/resident_equals_streamed"),
                    d.max_abs == 0.0 && d.non_finite == 0,
                    d.to_json(),
                    json!({"max_abs": 0.0}),
                )?;
            }
        }
    }
    // Every metric lands before the first gate can stop the stage.
    let diffs: Vec<_> = named
        .iter()
        .zip(&states)
        .map(|((_, name, limit), got)| {
            Ok((
                *name,
                *limit,
                diff(got, &st::take(&mut orc, name, oracle)?.data),
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    for (name, limit, d) in &diffs {
        // The embedding gather has nothing to round: bf16 rows read as f32.
        let ok = if *limit == 0.0 {
            d.max_abs == 0.0 && d.non_finite == 0
        } else {
            d.within(*limit) && d.cosine >= 0.999
        };
        report.check(
            *name,
            ok,
            d.to_json(),
            json!({"rel_l2": limit, "cosine_min": 0.999}),
        )?;
    }
    Ok(())
}

fn audio_vae(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    device: &str,
    max_abs: f64,
) -> StageResult<()> {
    use fastvideo_cudarc::h3::audio_vae::H3AudioDecoder;
    use fastvideo_models::h3::config::H3AudioVaeConfig;

    report.set("device", crate::gpu::init(device)?);
    let mut orc = st::load(oracle)?;
    let latent = st::take(&mut orc, "audio_latent", oracle)?;
    let want = st::take(&mut orc, "audio_wave", oracle)?;

    let timer = std::time::Instant::now();
    let map = WeightMap::open(&weights.join("audio_vae"))?;
    // The port loads ONE anti-aliasing filter and uses it everywhere; that is
    // only right if the checkpoint's 254 copies are the same bytes.
    if let Some(lazy) = map.lazy() {
        let keys: Vec<String> = lazy
            .keys_with_prefix("decoder.")
            .into_iter()
            .filter(|k| k.ends_with("filter"))
            .map(|k| k.to_string())
            .collect();
        let first = keys
            .first()
            .map(|k| lazy.view(k).map(|v| v.bytes.to_vec()))
            .transpose()?;
        let mut odd = Vec::new();
        for k in &keys {
            if Some(lazy.view(k)?.bytes) != first.as_deref() {
                odd.push(k.clone());
            }
        }
        report.check(
            "filters_identical",
            !keys.is_empty() && odd.is_empty(),
            json!({"filters": keys.len(), "differing": odd.iter().take(8).collect::<Vec<_>>()}),
            json!({"differing": 0}),
        )?;
    }
    let decoder = H3AudioDecoder::load(H3AudioVaeConfig::fasth3_8step(), &map)?;
    report.note(
        "load_audio_vae",
        json!({"seconds": timer.elapsed().as_secs_f64()}),
    );

    let input = CudaTensor::from_vec(latent.data, latent.shape.clone())?;
    // Intermediates the oracle dumped (`audio_<name>`), diffed as they are
    // produced: the first one that departs names the op that did it.
    let mut stages: Vec<(String, crate::metrics::Diff)> = Vec::new();
    let (wave, seconds) = measure(report, "audio_decode", || {
        let out = decoder.decode_observed(&input, &mut |name, x| {
            if let Some(reference) = orc.get(&format!("audio_{name}")) {
                stages.push((name.to_string(), diff(&x.host_cow()?, &reference.data)));
            }
            Ok(())
        })?;
        Ok((out.shape.clone(), out.host_cow()?.into_owned()))
    })?;
    report.note(
        "audio_decode",
        json!({"seconds": seconds, "latent": latent.shape, "wave": wave.0}),
    );
    for (name, d) in &stages {
        report.note(format!("audio_stage/{name}"), d.to_json());
    }
    report.check(
        "audio_wave_shape",
        wave.0 == want.shape,
        json!({"ours": wave.0}),
        json!({"reference": want.shape}),
    )?;
    let d = diff(&wave.1, &want.data);
    // A clamp to [-1, 1] on both sides can hide an overdriven decode; say how much of it is railed.
    let railed =
        want.data.iter().filter(|v| v.abs() >= 1.0).count() as f64 / want.data.len().max(1) as f64;
    report.note(
        "audio_wave_railed",
        json!({"fraction_of_reference_at_full_scale": railed}),
    );
    report.check(
        "audio_wave",
        d.non_finite == 0 && d.max_abs <= max_abs,
        d.to_json(),
        json!({"max_abs": max_abs}),
    )?;
    Ok(())
}

fn vae(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    device: &str,
    max_abs: f64,
    max_rel: f64,
) -> StageResult<()> {
    use fastvideo_cudarc::h3::vae::H3VideoDecoder;
    use fastvideo_models::h3::config::H3VideoVaeConfig;

    report.set("device", crate::gpu::init(device)?);
    let mut orc = st::load(oracle)?;
    let latent = st::take(&mut orc, "vae_latent", oracle)?;
    let want = st::take(&mut orc, "vae_video_raw", oracle)?;
    drop(orc);

    let timer = std::time::Instant::now();
    let decoder = H3VideoDecoder::load(
        H3VideoVaeConfig::fasth3_8step(),
        &WeightMap::open(&weights.join("vae"))?,
    )?;
    report.note(
        "load_vae",
        json!({"seconds": timer.elapsed().as_secs_f64()}),
    );

    // The reference is [1, 3, F, H, W]; chunks arrive as [3, f, H, W] and are
    // placed by their frame offset.
    let [_, channels, frames, height, width] = want.shape[..] else {
        return Err(anyhow::anyhow!(
            "vae_video_raw has shape {:?}, expected [1, 3, F, H, W]",
            want.shape
        )
        .into());
    };
    let plane = height * width;
    let mut video = vec![f32::NAN; channels * frames * plane];
    let input = CudaTensor::from_vec(latent.data, latent.shape.clone())?;
    let mem = crate::gpu::PeakMem::start();
    let (emitted, seconds) = measure(report, "vae_decode", || {
        Ok(decoder.decode_raw_streaming(&input, &mut |offset, chunk| {
            let [c, f, h, w] = chunk.shape[..] else {
                return Err(fastvideo_cudarc::wan::tensor::TensorError::Message(
                    format!("chunk shape {:?}", chunk.shape),
                ));
            };
            if c != channels || h != height || w != width || offset + f > frames {
                return Err(fastvideo_cudarc::wan::tensor::TensorError::Message(
                    format!(
                        "chunk {:?} at frame {offset} does not fit the reference {:?}",
                        chunk.shape, want.shape
                    ),
                ));
            }
            let host = chunk.host_cow()?;
            for ch in 0..c {
                let dst = (ch * frames + offset) * plane;
                video[dst..dst + f * plane]
                    .copy_from_slice(&host[ch * f * plane..(ch + 1) * f * plane]);
            }
            Ok(())
        })?)
    })?;
    report.note("vae_decode", json!({"seconds": seconds, "frames": emitted, "latent": latent.shape, "peak_mib": mem.stop()}));
    report.check(
        "vae_frames",
        emitted == frames,
        json!({"ours": emitted}),
        json!({"reference": frames}),
    )?;
    let d = diff(&video, &want.data);
    // Where along time the error sits separates a chunk cross-fade bug from a ViT bug.
    let per_frame: Vec<f64> = (0..frames)
        .map(|f| {
            (0..channels)
                .flat_map(|ch| {
                    let base = (ch * frames + f) * plane;
                    video[base..base + plane]
                        .iter()
                        .zip(&want.data[base..base + plane])
                })
                .fold(0.0f64, |m, (a, b)| m.max(f64::from((a - b).abs())))
        })
        .collect();
    report.note("vae_max_abs_per_frame", json!({"values": per_frame}));
    report.check(
        "vae_video_raw",
        d.max_abs <= max_abs && d.within(max_rel),
        d.to_json(),
        json!({"max_abs": max_abs, "rel_l2": max_rel}),
    )?;
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
        return Err(anyhow::anyhow!(
            "video_noise has shape {:?}, expected [1, C, T, H, W]",
            video.shape
        )
        .into());
    };
    if audio.shape.len() != 2 || audio.shape[0] % 2 != 0 || audio.shape[1] != cfg.audio_in_channels
    {
        return Err(anyhow::anyhow!(
            "audio_noise has shape {:?}, expected [2 Na, {}]",
            audio.shape,
            cfg.audio_in_channels
        )
        .into());
    }
    let layout = H3PackedLayout::new(text_tokens, (t, h, w), audio.shape[0] / 2, cfg.patch_size)
        .map_err(|e| anyhow::anyhow!(e))?;

    let want_pos = st::take(orc, "position_ids", oracle)?;
    let ours: Vec<f32> = layout
        .position_ids
        .iter()
        .flatten()
        .map(|&p| p as f32)
        .collect();
    let worst = ours
        .iter()
        .zip(&want_pos.data)
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0.0, f64::max);
    report.check(
        "position_ids",
        ours.len() == want_pos.data.len() && worst == 0.0,
        json!({"rows": layout.sequence_length(), "max_abs": worst}),
        json!({"rows": want_pos.shape.first(), "exact_as_f32": true}),
    )?;
    let want_tags = ints(&st::take(orc, "token_tags", oracle)?, "token_tags")?;
    let tags_ok = want_tags.len() == layout.token_tags.len()
        && want_tags
            .iter()
            .zip(&layout.token_tags)
            .all(|(a, b)| *a == u32::from(*b));
    report.check(
        "token_tags",
        tags_ok,
        json!({"rows": layout.token_tags.len()}),
        json!({"exact": true}),
    )?;

    let schedule = H3JointSchedule::fasth3_8step();
    for (name, ours) in [
        ("video_sigmas", &schedule.video.sigmas),
        ("audio_sigmas", &schedule.audio.sigmas),
        ("video_timesteps", &schedule.video.timesteps),
        ("audio_timesteps", &schedule.audio.timesteps),
    ] {
        let want = st::take(orc, name, oracle)?;
        let same = want.data.len() == ours.len()
            && want
                .data
                .iter()
                .zip(ours.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        report.check(
            name,
            same,
            json!({"ours": ours}),
            json!({"reference": want.data, "bitwise": true}),
        )?;
    }
    Ok(OracleRequest {
        video_rows: patchify(&video.data, [c, t, h, w], cfg.patch_size)
            .map_err(|e| anyhow::anyhow!(e))?,
        audio_rows: audio.data,
        latent_shape: [c, t, h, w],
        layout,
    })
}

fn dit(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    device: &str,
    max_rel: f64,
) -> StageResult<()> {
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
        .find(|&i| {
            schedule.row_timesteps(i).is_ok_and(|r| {
                r.timesteps.len() == want_ts.data.len()
                    && r.timesteps
                        .iter()
                        .zip(&want_ts.data)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            })
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dit_timesteps {:?} match no rung of the FastH3 ladder",
                want_ts.data
            )
        })?;
    let want_index = ints(
        &st::take(&mut orc, "dit_timestep_indices", oracle)?,
        "dit_timestep_indices",
    )?;
    let ours_index = layout.timestep_indices(
        &schedule
            .row_timesteps(step)
            .map_err(|e| anyhow::anyhow!(e))?,
    );
    report.check(
        "timestep_indices",
        want_index.len() == ours_index.len()
            && want_index
                .iter()
                .zip(&ours_index)
                .all(|(a, b)| *a as usize == *b),
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
    // Calibration (RTX PRO 6000, diffusers bf16 reference, 37,745 rows):
    // text_refined 9.75e-3, block_0 1.11e-2, block_24 1.30e-2, block_49 2.93e-2,
    // dit_video 1.30e-2, dit_audio 9.20e-3. About 1e-2 enters at the refiner and
    // then stays FLAT through 24 blocks: entry rounding, not a defect that
    // compounds. Upstream keeps `context_embedder` and the refiner in bf16 (only
    // proj_in/out, the audio heads, time_embedder and rope are fp32-pinned, in
    // both diffusers and FastVideo), norms and softmax included, and their input
    // is a Qwen residual stream with outlier channels, the worst case for an
    // 8-bit mantissa. An f32 path on our side would move us toward the true
    // value, not toward this reference, so the gate is set above the floor.
    let mut results = vec![(
        "text_refined".to_string(),
        2e-2,
        diff(
            &refined.host_cow()?,
            &st::take(&mut orc, "text_refined", oracle)?.data,
        ),
    )];

    // --- load: AdaLN table, then the resident stack (dense: no gates) ---------
    let timer = std::time::Instant::now();
    let model = H3Transformer::load(cfg.clone(), &map, &schedule, false)?;
    report.note(
        "load_dit",
        json!({"seconds": timer.elapsed().as_secs_f64()}),
    );
    let table = model.adaln_table();
    // `temb` rows are the sorted-unique timesteps: video (smaller t) then audio.
    let want_temb = st::take(&mut orc, "temb", oracle)?;
    let ours_temb: Vec<f32> = table.temb[2 * step]
        .iter()
        .chain(&table.temb[2 * step + 1])
        .copied()
        .collect();
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
            ref_adaln.extend_from_slice(
                &want_adaln.data[(p * rows + row) * hidden..(p * rows + row + 1) * hidden],
            );
        }
    }
    results.push(("adaln_block0".into(), 2e-3, diff(&ours_adaln, &ref_adaln)));

    // --- one forward ------------------------------------------------------------
    let stride = orc
        .get("block_0")
        .map(|t| layout.sequence_length().div_ceil(t.shape[1].max(1)))
        .unwrap_or(16);
    let strided: Vec<usize> = (0..layout.sequence_length())
        .step_by(stride.max(1))
        .collect();
    let device_layout = DeviceLayout::new(&cfg, layout.clone())?;
    let video_rows = CudaTensor::from_vec(
        request.video_rows,
        vec![layout.video.len, cfg.video_patch_dim()],
    )?;
    let audio_rows = CudaTensor::from_vec(
        request.audio_rows,
        vec![layout.audio.len, cfg.audio_in_channels],
    )?;
    let wanted: Vec<String> = orc
        .keys()
        .filter(|k| k.starts_with("block_"))
        .cloned()
        .collect();
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
                    let rows = x
                        .reshape(vec![x.shape[1], x.shape[2]])?
                        .index_select_rows(&strided)?;
                    dumps.push((name.to_string(), rows.host_cow()?.into_owned()));
                }
                Ok(())
            }),
            None,
            None,
        )?;
        Ok((
            out.0.host_cow()?.into_owned(),
            out.1.host_cow()?.into_owned(),
        ))
    })?;
    report.note(
        "dit_forward",
        json!({"seconds": seconds, "peak_mib": mem.stop(), "block_row_stride": stride}),
    );

    for (name, got) in &dumps {
        let index: usize = name.trim_start_matches("block_").parse().unwrap_or(0);
        // Divergence is allowed to grow with depth: bf16 on both sides.
        let limit = if index < cfg.num_layers / 2 {
            2e-2
        } else {
            5e-2
        };
        results.push((
            name.clone(),
            limit,
            diff(got, &st::take(&mut orc, name, oracle)?.data),
        ));
    }
    results.push((
        "dit_video".into(),
        max_rel,
        diff(&video_v, &st::take(&mut orc, "dit_video", oracle)?.data),
    ));
    results.push((
        "dit_audio".into(),
        max_rel,
        diff(&audio_v, &st::take(&mut orc, "dit_audio", oracle)?.data),
    ));

    // Every metric lands before the first gate can stop the stage: where the
    // error starts to grow is the diagnosis.
    report.set(
        "metrics",
        results
            .iter()
            .map(|(n, _, d)| (n.clone(), d.to_json()))
            .collect::<serde_json::Map<_, _>>(),
    );
    for (name, limit, d) in &results {
        let cosine_min = if name.starts_with("dit_") { 0.999 } else { 0.0 };
        report.check(
            name.as_str(),
            d.within(*limit) && d.cosine >= cosine_min,
            d.to_json(),
            json!({"rel_l2": limit, "cosine_min": cosine_min}),
        )?;
    }
    Ok(())
}

fn ladder(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    device: &str,
    max_rel: f64,
    gated_steps: usize,
) -> StageResult<()> {
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
    let refined = H3TextRefiner::load(&cfg, &map)?
        .forward(&CudaTensor::from_vec(text.data, text.shape.clone())?)?;
    let timer = std::time::Instant::now();
    let model = H3Transformer::load(cfg.clone(), &map, &schedule, false)?;
    report.note(
        "load_dit",
        json!({"seconds": timer.elapsed().as_secs_f64()}),
    );

    let device_layout = DeviceLayout::new(&cfg, layout.clone())?;
    let video_rows = CudaTensor::from_vec(
        request.video_rows,
        vec![layout.video.len, cfg.video_patch_dim()],
    )?
    .to_device()?;
    let audio_rows = CudaTensor::from_vec(
        request.audio_rows,
        vec![layout.audio.len, cfg.audio_in_channels],
    )?
    .to_device()?;
    let (nv, na) = (
        layout.video.len * cfg.video_patch_dim(),
        layout.audio.len * cfg.audio_in_channels,
    );
    let mut steps = Vec::new();
    let mem = crate::gpu::PeakMem::start();
    let (_, seconds) = measure(report, "loop", || {
        let mut last = std::time::Instant::now();
        denoise(
            &model,
            &device_layout,
            &refined,
            video_rows,
            audio_rows,
            &schedule,
            AttnMode::Dense,
            None,
            None,
            &mut |step, video, audio| {
                let (dv, da) = (
                    diff(
                        &video.host_cow()?,
                        &want_video.data[step * nv..(step + 1) * nv],
                    ),
                    diff(
                        &audio.host_cow()?,
                        &want_audio.data[step * na..(step + 1) * na],
                    ),
                );
                eprintln!(
                    "[INFO] h3 loop step {step}: {:.1}s video rel {:.3e} audio rel {:.3e}",
                    last.elapsed().as_secs_f64(),
                    dv.rel_l2,
                    da.rel_l2
                );
                last = std::time::Instant::now();
                steps.push((dv, da));
                Ok(())
            },
        )?;
        Ok(())
    })?;
    report.note("loop", json!({"seconds": seconds, "peak_mib": mem.stop()}));
    report.set(
        "per_step",
        steps
            .iter()
            .map(|(v, a)| json!({"video": v.to_json(), "audio": a.to_json()}))
            .collect::<Vec<_>>(),
    );
    if steps.len() < gated_steps.max(1) {
        return Err(anyhow::anyhow!(
            "the loop produced {} steps, fewer than the {gated_steps} gated",
            steps.len()
        )
        .into());
    }
    for (index, (v, a)) in steps.iter().enumerate() {
        if index < gated_steps {
            report.check(
                format!("loop_video_step{index}"),
                v.within(max_rel),
                v.to_json(),
                json!({"rel_l2": max_rel}),
            )?;
            report.check(
                format!("loop_audio_step{index}"),
                a.within(max_rel),
                a.to_json(),
                json!({"rel_l2": max_rel}),
            )?;
        } else {
            // Finite is still required: a NaN late in the ladder is a defect at any tolerance.
            report.check(
                format!("loop_step{index}_finite"),
                v.non_finite == 0 && a.non_finite == 0,
                json!({"video_rel_l2": v.rel_l2, "audio_rel_l2": a.rel_l2}),
                json!({"non_finite": 0}),
            )?;
        }
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
    let layout =
        H3PackedLayout::new(150, (9, 12, 20), 50, [1, 2, 2]).map_err(|e| anyhow::anyhow!(e))?;
    // The tensor-core fine kernel is written for head dim 128.
    let (heads, dim, seq) = (4usize, 128usize, layout.sequence_length());
    let shape = vec![1, heads, seq, dim];
    let draw =
        |salt: u64, std: f32| crate::rand_weights::randn(seed ^ salt, heads * seq * dim, std);
    // Post-norm Q/K have unit RMS; that scale also keeps the top-k decisive.
    let (q, k, v, gate) = (
        draw(0x51, 1.0),
        draw(0x4b, 1.0),
        draw(0x56, 1.0),
        draw(0x47, 0.5),
    );
    let tensor = |x: &[f32]| CudaTensor::from_vec(x.to_vec(), shape.clone());

    for (name, sparsity, gated) in [
        ("sparse_gated", 0.8, true),
        ("sparse_ungated", 0.5, false),
        ("all_tiles_gated", 0.0, true),
    ] {
        let vsa = H3Vsa::new(&layout, heads, dim, H3VsaConfig { sparsity, group: 4, tile_size: 64 })?;
        let plan = vsa.plan().clone();
        report.note(format!("{name}/plan"), json!({"prefix_tiles": plan.prefix_tiles, "video_tiles": plan.video_tiles, "k_vid": vsa.k_vid(), "rows": seq}));
        let want = attention_host(
            &q,
            &k,
            &v,
            gated.then_some(&gate[..]),
            &plan,
            vsa.k_vid(),
            heads,
            dim,
        )?;
        let g = if gated { Some(tensor(&gate)?) } else { None };
        let (got, seconds) = measure(report, name, || {
            Ok(vsa
                .attend(tensor(&q)?, tensor(&k)?, tensor(&v)?, g)?
                .host_cow()?
                .into_owned())
        })?;
        let d = diff(&got, &want);
        // Prefix rows take the dense splice, video rows the fine kernel: report them apart.
        let split = plan.prefix_rows * dim;
        let part = |lo: usize, hi: usize| -> Vec<f32> {
            (0..heads)
                .flat_map(|h| got[h * seq * dim + lo..h * seq * dim + hi].to_vec())
                .collect()
        };
        let want_part = |lo: usize, hi: usize| -> Vec<f32> {
            (0..heads)
                .flat_map(|h| want[h * seq * dim + lo..h * seq * dim + hi].to_vec())
                .collect()
        };
        report.note(
            format!("{name}/by_segment"),
            json!({"seconds": seconds, "prefix_rows": diff(&part(0, split), &want_part(0, split)).to_json(), "video_rows": diff(&part(split, seq * dim), &want_part(split, seq * dim)).to_json()}),
        );
        report.check(
            name,
            d.within(max_rel),
            d.to_json(),
            json!({"rel_l2": max_rel}),
        )?;
    }

    // The load-bearing identity: every tile selected and no gate is dense attention.
    let vsa = H3Vsa::new(
        &layout,
        heads,
        dim,
        H3VsaConfig {
            sparsity: 0.0,
            group: 4,
            tile_size: 64,
        },
    )?;
    let got = vsa
        .attend(tensor(&q)?, tensor(&k)?, tensor(&v)?, None)?
        .host_cow()?
        .into_owned();
    let dense = scaled_dot_product_attention(&tensor(&q)?, &tensor(&k)?, &tensor(&v)?, None)?
        .host_cow()?
        .into_owned();
    let d = diff(&got, &dense);
    report.check(
        "all_tiles_ungated_is_dense",
        d.within(max_rel),
        d.to_json(),
        json!({"rel_l2": max_rel}),
    )?;
    Ok(())
}

fn slim_text(report: &mut Report, weights: &Path, out: &Path, shard_gib: u64) -> StageResult<()> {
    let timer = std::time::Instant::now();
    let r = fastvideo_cudarc::h3::slim::write_slim_text_root(weights, out, shard_gib.max(1) << 30)?;
    let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
    report.note(
        "slim_text",
        json!({
            "seconds": timer.elapsed().as_secs_f64(),
            "files": r.files.len(),
            "tensors": r.tensors,
            "bytes": r.bytes,
            "gib": gib(r.bytes),
            "skipped_tensors": r.skipped_tensors,
            "skipped_gib": gib(r.skipped_bytes),
            "out": out,
        }),
    );
    // embed_tokens + 50 layers x 11 tensors, all bf16: the figure docs/ports/h3.md derives from the headers.
    report.check(
        "slim_is_tap_50",
        r.tensors == 551 && r.bytes == 50_315_658_240,
        json!({"tensors": r.tensors, "bytes": r.bytes}),
        json!({"tensors": 551, "bytes": 50_315_658_240u64}),
    )?;
    Ok(())
}

/// The requested H3 geometry: `seconds` at the default 16:9 canvas unless
/// `num_frames` / `height` / `width` say otherwise.
#[derive(Debug, Clone, Copy)]
pub struct GenCanvas {
    pub seconds: usize,
    pub num_frames: Option<usize>,
    pub height: Option<usize>,
    pub width: Option<usize>,
}

impl GenCanvas {
    /// The request, validated against the model's canvas and duration rules
    /// (`H3Geometry::checked`).
    pub fn request(
        self,
        prompt: &str,
        seed: u64,
    ) -> anyhow::Result<fastvideo_cudarc::h3::pipeline::H3Request> {
        use fastvideo_cudarc::h3::pipeline::H3Request;
        let default =
            H3Request::seconds(prompt, self.seconds, seed).map_err(|e| anyhow::anyhow!(e))?;
        if self.num_frames.is_none() && self.height.is_none() && self.width.is_none() {
            return Ok(default);
        }
        if self.height.is_some() != self.width.is_some() {
            anyhow::bail!("h3 gen: pass --height and --width together");
        }
        H3Request::sized(
            prompt,
            self.height.unwrap_or(default.height),
            self.width.unwrap_or(default.width),
            self.num_frames.unwrap_or(default.num_frames),
            seed,
        )
        .map_err(|e| anyhow::anyhow!(e))
    }
}

#[allow(clippy::too_many_arguments)]
fn gen(
    report: &mut Report,
    weights: &Path,
    prompt: &str,
    canvas: GenCanvas,
    seed: u64,
    mp4: bool,
    clip_dir: &Path,
    options: fastvideo_cudarc::h3::pipeline::H3PipelineOptions,
    warm: bool,
    compare_text_encoders: bool,
    device: &str,
    device_budget_gib: Option<f64>,
) -> StageResult<()> {
    use fastvideo_cudarc::h3::pipeline::{H3Output, H3Pipeline};

    report.set(
        "device",
        crate::gpu::init_with_budget(device, device_budget_gib)?,
    );
    report.set(
        "device_budget_gib",
        json!(fastvideo_cudarc::wan::device::device_budget()
            .map(|b| b as f64 / f64::from(1u32 << 30))),
    );
    let seconds = canvas.seconds;
    let mut request = canvas.request(prompt, seed)?;
    request.mp4 = mp4;
    {
        use fastvideo_models::h3::memory::{plan, H3PlanOptions};
        let g = fastvideo_models::h3::config::H3Geometry::new(
            request.height,
            request.width,
            request.num_frames,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
        let phases = |p: &fastvideo_models::h3::memory::H3MemoryPlan| -> Vec<serde_json::Value> {
            p.stages
                .iter()
                .map(|s| json!({"stage": s.stage, "gib": gib(s.total())}))
                .collect()
        };
        let (resident, streamed) = (
            plan(&g, H3PlanOptions::default()),
            plan(&g, H3PlanOptions::streamed()),
        );
        eprintln!(
            "h3 gen {}x{}x{}: planned peak {:.2} GiB resident, {:.2} GiB streamed",
            g.width,
            g.height,
            g.num_frames,
            gib(resident.peak()),
            gib(streamed.peak())
        );
        report.set(
            "memory_plan",
            json!({
                "resident": {"peak_gib": gib(resident.peak()), "stages": phases(&resident)},
                "streamed": {"peak_gib": gib(streamed.peak()), "stages": phases(&streamed)},
            }),
        );
    }
    let attention = {
        use fastvideo_models::h3::lora::{
            is_sol_h3_recipe, is_sol_h3_rtx_recipe, is_sol_h3_spark_recipe,
        };
        use fastvideo_models::h3::sol::{recipe_sol_attn_policy, H3SolAttnPolicy};
        let recipe = options.recipe.as_deref();
        let dense_recipe = recipe.is_some_and(|r| {
            (is_sol_h3_recipe(r) && !is_sol_h3_spark_recipe(r)) || is_sol_h3_rtx_recipe(r)
        });
        match recipe_sol_attn_policy(
            recipe,
            std::env::var("FASTVIDEO_H3_SOL_ATTN").ok().as_deref(),
            options.ref2va,
        )
        .map_err(|e| anyhow::anyhow!(e))?
        {
            H3SolAttnPolicy::Off if options.dense || dense_recipe => "dense, no gate".to_string(),
            H3SolAttnPolicy::Off => "vsa-h3 + to_gate_compress".to_string(),
            policy => format!("sol-attn {policy:?}"),
        }
    };
    report.set(
        "request",
        json!({
            "prompt": prompt, "seconds": seconds, "seed": seed, "warm": warm,
            "attention": attention,
            "height": request.height, "width": request.width, "num_frames": request.num_frames,
            "text_cache": options.text_cache, "text_weights": options.text_root,
            "video_vae": if options.taeh3.is_some() { "taeh3" } else { "official" },
        }),
    );

    let requested_encoder = format!("{:?}", options.text_encoder);
    let free_before = crate::gpu::mem_info().map(|(free, _)| free as f64 / f64::from(1u32 << 30));
    let mem = crate::gpu::PeakMem::start();
    let timer = std::time::Instant::now();
    let pipeline = H3Pipeline::load(weights, options)?;
    let load_s = timer.elapsed().as_secs_f64();
    let l = &pipeline.load_timings;
    let (encoder_kind, encoder_bytes) = pipeline.text_encoder();
    report.note(
        "text_encoder",
        json!({"requested": requested_encoder, "chosen": encoder_kind, "device_gib": encoder_bytes as f64 / f64::from(1u32 << 30), "load_seconds": l.text_encoder_s, "free_gib_before_load": free_before}),
    );
    if compare_text_encoders {
        match pipeline.text_encoder_drift(prompt)? {
            Some((rel_l2, cosine)) => {
                // Same scale as the fp8 gate of `h3 text`: past this the conditioning is a different prompt.
                report.check("text_encoder_drift", rel_l2.is_finite() && rel_l2 <= 5e-2, json!({"rel_l2": rel_l2, "cosine": cosine, "resident": encoder_kind, "against": "streamed"}), json!({"rel_l2": 5e-2}))?;
            }
            None => report.note(
                "text_encoder_drift",
                json!({"skipped": "the encoder is streamed; there is nothing to compare"}),
            ),
        }
    }
    report.note("load", json!({"seconds": load_s, "text_encoder_s": l.text_encoder_s, "refiner_s": l.refiner_s, "dit_s": l.dit_s, "video_vae_s": l.video_vae_s, "audio_vae_s": l.audio_vae_s, "dit_residency": pipeline.residency().as_str()}));

    let timings = |out: &H3Output, total: f64| {
        let t = &out.timings;
        json!({
            "total_s": total,
            "text_s": t.text_s,
            "text_cache": out.text_cache.as_str(),
            "text_encoder": out.text_encoder,
            "refine_s": t.refine_s,
            "denoise_s": t.denoise_s,
            "step_s": t.step_s,
            "audio_decode_s": t.audio_decode_s,
            "video_decode_s": t.video_decode_s,
            "write_s": t.write_s,
            "decode_s": t.audio_decode_s + t.video_decode_s,
        })
    };
    if warm {
        // Untimed in the sense that it is not THE number; it is still recorded,
        // because cold-vs-warm is itself what the user asked about.
        let timer = std::time::Instant::now();
        let cold = pipeline.generate(&request, &clip_dir.join("cold"))?;
        report.note(
            "cold_generation",
            timings(&cold, timer.elapsed().as_secs_f64()),
        );
        // The cold pass would otherwise sit in the same counters as the
        // measured one; dit_phases should describe the timed generate only.
        fastvideo_cudarc::wan::stats::phase_reset();
    }
    let timer = std::time::Instant::now();
    let out = pipeline.generate(&request, clip_dir)?;
    let total = timer.elapsed().as_secs_f64();
    let mut values = timings(&out, total);
    values["warm"] = json!(warm);
    values["load_s"] = json!(load_s);
    values["peak_mib"] = json!(mem.stop());
    report.note("timings", values);
    let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
    report.set(
        "stage_memory",
        json!({
            "dit_residency": out.dit_residency,
            "stages": out.memory.iter().map(|p| json!({
                "stage": p.phase,
                "peak_used_gib": gib(p.peak_used),
                "peak_reserved_gib": gib(p.peak_reserved),
                "end_used_gib": gib(p.end_used),
            })).collect::<Vec<_>>(),
        }),
    );
    if let Some(o) = &out.offload {
        report.set(
            "dit_offload",
            json!({
                "block_copies": o.copies,
                "h2d_gib": gib(o.bytes),
                "h2d_gbs": o.throughput_gbs(),
                "copy_s": o.copy_ms * 1e-3,
                "exposed_s": o.exposed_ms * 1e-3,
                "hidden_fraction": o.hidden_fraction(),
                "worst_stall_ms": o.worst_exposed_ms,
            }),
        );
    }
    // Empty unless FASTVIDEO_PROFILE=1: each phase synchronizes, so this is
    // where time goes, not a number to quote against an unprofiled clip.
    // Nested names overlap (h3_2_attn contains h3_attn_* and vsa_h3_*), so
    // each group's pct is of that group, not of a grand sum.
    let phases = fastvideo_cudarc::wan::stats::phase_report();
    if !phases.is_empty() {
        let row = |n: &str, c: u64, s: f64, total: f64| json!({"phase": n, "calls": c, "seconds": s, "pct": if total > 0.0 { 100.0 * s / total } else { 0.0 }});
        let group = |pred: &dyn Fn(&str) -> bool| {
            let rows: Vec<_> = phases.iter().copied().filter(|(n, _, _)| pred(n)).collect();
            let total: f64 = rows.iter().map(|(_, _, s)| s).sum();
            json!({
                "total_s": total,
                "by_phase": rows.iter().map(|(n, c, s)| row(n, *c, *s, total)).collect::<Vec<_>>(),
            })
        };
        report.set(
            "dit_phases",
            json!({
                "block": group(&|n| n.starts_with("h3_") && !n.starts_with("h3_attn_") && !n.starts_with("h3_ffn_")),
                "attn": group(&|n| n.starts_with("h3_attn_")),
                "ffn": group(&|n| n.starts_with("h3_ffn_")),
                "vsa": group(&|n| n.starts_with("vsa_")),
            }),
        );
    }
    report.set("output", json!({"frames": out.frames, "text_tokens": out.text_tokens, "sequence_length": out.sequence_length, "mp4": out.mp4, "wav": out.wav, "audio_samples_per_channel": out.geometry.audio_samples()}));
    if warm {
        report.check(
            "warm_text_is_a_cache_hit",
            out.text_cache.as_str() != "miss",
            json!({"text_cache": out.text_cache.as_str(), "text_s": out.timings.text_s}),
            json!({"expected": "hit (or disabled)"}),
        )?;
    }
    report.check(
        "frames",
        out.frames == out.geometry.num_frames,
        json!({"decoded": out.frames}),
        json!({"expected": out.geometry.num_frames}),
    )?;
    // A finite, non-constant picture: the cheapest statement that the clip is not garbage.
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
    if mp4 {
        report.check(
            "mp4_written",
            out.mp4.is_some(),
            json!({"mp4": out.mp4}),
            json!({"expected": "output.mp4 with an audio track"}),
        )?;
    }
    Ok(())
}
