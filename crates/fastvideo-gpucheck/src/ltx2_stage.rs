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
use fastvideo_cudarc::ltx2::pipeline::{
    self as ltx2_pipeline, Decoders, Ltx2Paths, Ltx2Pipeline, Ltx2Request, PipelineOptions,
    TextResidency,
};
use fastvideo_cudarc::ltx2::slim::{copy_dir_files, write_slim_decoder, EmbedDtype, SlimOptions};
use fastvideo_cudarc::ltx2::text::{HiddenStack, PaddedPrompt, TextConnectors};
use fastvideo_cudarc::ltx2::transformer::{Ltx2Transformer, Ropes};
use fastvideo_cudarc::ltx2::vae::VideoDecoder;
use fastvideo_cudarc::ltx2::vocoder::Vocoder;
use fastvideo_cudarc::wan::pipeline::{interleave_audio, write_wav};
use fastvideo_cudarc::wan::weights::WeightMap;
use fastvideo_cudarc::CudaTensor;
use fastvideo_models::ltx2::config::{
    ltx2_19b_distilled, ltx2_23_22b_distilled, ltx2_5_22b_distilled, Ltx2Config,
};
use fastvideo_models::ltx2::{Ltx2RopeTables, Ltx2Schedule, SplitRope};
use serde_json::json;

use crate::metrics::diff;
use crate::model::measure;
use crate::report::{Report, StageResult};
use crate::st::F32Tensor;

/// Which distilled LTX checkpoint the port targets (`gen` loads the matching config).
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ModelVersion {
    /// LTX-2.0 19B distilled, deterministic Euler stage 1.
    #[value(name = "2.0")]
    V20,
    /// LTX-2.3 22B distilled, Euler + optional 5+2 / 8+3 two-stage.
    #[value(name = "2.3")]
    V23,
    /// LTX-2.5 22B distilled, ancestral Euler stage 1.
    #[value(name = "2.5")]
    V25,
}

impl ModelVersion {
    pub fn config(self) -> Ltx2Config {
        match self {
            Self::V20 => ltx2_19b_distilled(),
            Self::V23 => ltx2_23_22b_distilled(),
            Self::V25 => ltx2_5_22b_distilled(),
        }
    }
}

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
    /// The rotary tables, then one DiT forward on the oracle's seeded latents
    /// and connector outputs, diffed at three blocks and at both outputs.
    /// Needs the global `--mode fast`: 19B parameters are 38 GB as bf16 and
    /// 76 GB as the float32 that `--mode exact` would load.
    Dit {
        /// The distilled DiT: `ltx-2-19b-distilled.safetensors`, or a
        /// diffusers folder holding (or being) `transformer/`.
        #[arg(long)]
        dit: PathBuf,
        /// `--out` file of `ltx2_oracle.py` (stages `text`, `conn`, `dit`).
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[command(flatten)]
        geometry: Geometry,
        /// Only compare the rotary tables (host arithmetic, no weights read).
        #[arg(long)]
        rope_only: bool,
        /// cos/sin against the reference's tables.
        #[arg(long, default_value_t = 1e-6)]
        max_abs_rope: f64,
        /// Output of block 0 against a bf16 reference.
        #[arg(long, default_value_t = 2e-3)]
        max_rel_first: f64,
        /// Later blocks and both velocities.
        #[arg(long, default_value_t = 2e-2)]
        max_rel: f64,
        /// With a float32 reference in the file (`--dit-dtype both`): ours may
        /// be this many times as far from float32 as the bf16 reference is.
        #[arg(long, default_value_t = 1.25)]
        floor_factor: f64,
    },
    /// The 8-step distilled loop from the oracle's noise on the oracle's
    /// connector outputs, diffed after every step; with `--weights`, the final
    /// latents decoded at full size as well. Needs `--mode fast` (see `dit`).
    Loop {
        /// The distilled DiT (single file or diffusers `transformer/`).
        #[arg(long)]
        dit: PathBuf,
        /// `--out` file of `ltx2_oracle.py --sample`.
        #[arg(long)]
        oracle: PathBuf,
        /// A diffusers snapshot with `vae/`, `audio_vae/`, `vocoder/`: also
        /// compare decoded frames and audio against the oracle's.
        #[arg(long)]
        weights: Option<PathBuf>,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[command(flatten)]
        geometry: Geometry,
        /// Latents after each of the first four steps (sigma 1 → 0.975), where
        /// the trajectory has barely moved and a wrong update rule, sigma or
        /// velocity shows undiluted. Measured on hardware: 6e-4 … 3.4e-3.
        #[arg(long, default_value_t = 5e-3)]
        max_rel_first: f64,
        /// Reference line for the last four steps — *recorded, not gated*
        /// against a bf16 dump. Those steps take 93% of the way to the data, and
        /// a few-step distilled sampler roughly doubles a rounding-level
        /// difference per step (measured 2e-2 → 3e-1 here, the same shape on the
        /// H3 port, both with clean clips): against a reference whose own
        /// residual stream is bf16 that number measures the sampler's
        /// sensitivity, not the port. With `sample32.*` in the file every step
        /// *is* gated, against the measured floor.
        #[arg(long, default_value_t = 5e-2)]
        max_rel: f64,
        /// Our decoders on the *oracle's* final latents (decoder parity at
        /// production size), and on our own (the whole chain).
        #[arg(long, default_value_t = 45.0)]
        min_psnr_db: f64,
        #[arg(long, default_value_t = 35.0)]
        min_psnr_db_e2e: f64,
        /// As for `dit`: the gate when `sample32.*` is in the file.
        #[arg(long, default_value_t = 1.25)]
        floor_factor: f64,
    },
    /// Generate a clip end to end: prompt → mp4 with sound, plus timings and
    /// peak VRAM. Needs `--mode fast`.
    Gen {
        /// Distilled model family: `2.0` (19B, Euler) or `2.5` (22B, ancestral).
        #[arg(long, value_enum, default_value_t = ModelVersion::V20)]
        model_version: ModelVersion,
        /// A diffusers LTX-2 snapshot: `tokenizer/`, `text_encoder/`, `vae/`,
        /// `audio_vae/`, `vocoder/`.
        #[arg(long)]
        weights: PathBuf,
        /// The distilled DiT + connectors: `ltx-2-19b-distilled.safetensors`, or
        /// a diffusers root with `transformer/` and `connectors/`.
        #[arg(long)]
        dit: PathBuf,
        #[arg(long)]
        prompt: String,
        /// Output directory: `output.mp4`, `audio.wav`, `frame-NNN.png`.
        #[arg(long)]
        clip: PathBuf,
        #[arg(long, default_value_t = 10)]
        seed: u64,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[command(flatten)]
        geometry: Geometry,
        /// Frames and WAV only (no ffmpeg needed).
        #[arg(long)]
        no_mp4: bool,
        /// Directory of the text-conditioning cache. Default:
        /// `$FASTVIDEO_CACHE` / `$XDG_CACHE_HOME/fastvideo` / `~/.cache/fastvideo`,
        /// under `ltx2-text`. A hit skips Gemma and the connectors entirely.
        #[arg(long)]
        text_cache: Option<PathBuf>,
        /// Always encode the prompt; neither read nor write the cache.
        #[arg(long)]
        no_text_cache: bool,
        /// Generate once untimed first (cache bypassed, outputs under
        /// `<clip>/warmup`), then the reported run on the loaded pipeline.
        #[arg(long)]
        warm: bool,
        /// Another root for `tokenizer/` + `text_encoder/`: the output of
        /// `ltx2 slim-text`. Default: under `--weights`.
        #[arg(long)]
        text_weights: Option<PathBuf>,
        /// `auto` (resident when the device has room beside the DiT),
        /// `resident` or `streamed`. `FASTVIDEO_LTX2_TEXT` overrides.
        #[arg(long, default_value = "auto")]
        text: String,
        /// Distilled two-stage: half-res stage-1 → spatial ×2 → 3-step stage-2.
        /// Requires `--model-version 2.5` and `latent_upsampler/` under `--weights`.
        #[arg(long, default_value_t = false)]
        two_stage: bool,
        /// DiffVAE diffusion video decoder instead of the conv VAE.
        /// Requires `--model-version 2.5` and `diffusion_decoder/` under `--weights`.
        #[arg(long, default_value_t = false)]
        diff_vae: bool,
        /// First-frame PNG/JPEG for I2V encode (`docs/ports/ltx2.md`).
        #[arg(long)]
        image: Option<PathBuf>,
    },
    /// CPU only: rewrite the text encoder as the language model alone, its
    /// projections narrowed float32 → bf16 once, in load order (47 GB → 25.5 GB,
    /// or 23.5 GB with `--embed bf16`), plus a copy of `tokenizer/`.
    SlimText {
        /// Root holding `text_encoder/` and `tokenizer/`.
        #[arg(long)]
        weights: PathBuf,
        /// Output root: `<slim>/text_encoder/model-slim-*.safetensors`, `<slim>/tokenizer/`.
        #[arg(long)]
        slim: PathBuf,
        /// `f32` keeps the embedding rows bit-identical to the original's;
        /// `bf16` saves 2 GB and matches what a bf16 reference embeds.
        #[arg(long, default_value = "f32")]
        embed: String,
        #[arg(long, default_value_t = 5.0)]
        shard_gib: f64,
    },
}

/// The request geometry the oracle was run with (its own defaults).
#[derive(clap::Args, Debug, Clone, Copy)]
pub struct Geometry {
    #[arg(long, default_value_t = 512)]
    height: usize,
    #[arg(long, default_value_t = 768)]
    width: usize,
    #[arg(long, default_value_t = 121)]
    num_frames: usize,
    #[arg(long, default_value_t = 24.0)]
    frame_rate: f64,
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Info => {
            let c = ltx2_19b_distilled();
            report.set("config", format!("{c:?}"));
            Ok(())
        }
        Stage::Text {
            weights,
            dit,
            oracle,
            meta,
            prompt,
            device,
            skip_llm,
            max_rel,
            max_rel_llm,
            max_rel_e2e,
        } => text(
            report,
            &TextArgs {
                weights,
                dit,
                oracle,
                meta: meta.as_deref(),
                prompt: prompt.as_deref(),
                device,
                skip_llm: *skip_llm,
            },
            [*max_rel, *max_rel_llm, *max_rel_e2e],
        ),
        Stage::Audio {
            weights,
            oracle,
            device,
            wav,
            max_rel_mel,
            min_snr_db,
            min_snr_db_e2e,
        } => audio(
            report,
            weights,
            oracle,
            device,
            wav.as_deref(),
            [*max_rel_mel, *min_snr_db, *min_snr_db_e2e],
        ),
        Stage::Vae {
            weights,
            oracle,
            device,
            max_abs,
            max_rmse,
            max_abs_streamed,
        } => vae(
            report,
            weights,
            oracle,
            device,
            [*max_abs, *max_rmse, *max_abs_streamed],
        ),
        Stage::Dit {
            dit: path,
            oracle,
            device,
            geometry,
            rope_only,
            max_abs_rope,
            max_rel_first,
            max_rel,
            floor_factor,
        } => dit(
            report,
            path,
            oracle,
            device,
            *geometry,
            *rope_only,
            [*max_abs_rope, *max_rel_first, *max_rel, *floor_factor],
        ),
        Stage::Loop {
            dit: path,
            oracle,
            weights,
            device,
            geometry,
            max_rel_first,
            max_rel,
            min_psnr_db,
            min_psnr_db_e2e,
            floor_factor,
        } => sample_loop(
            report,
            path,
            oracle,
            weights.as_deref(),
            device,
            *geometry,
            [
                *max_rel_first,
                *max_rel,
                *min_psnr_db,
                *min_psnr_db_e2e,
                *floor_factor,
            ],
        ),
        Stage::Gen {
            model_version,
            weights,
            dit: path,
            prompt,
            clip,
            seed,
            device,
            geometry,
            no_mp4,
            text_cache,
            no_text_cache,
            warm,
            text_weights,
            text,
            two_stage,
            diff_vae,
            image,
        } => {
            let text_cache = if *no_text_cache {
                None
            } else {
                text_cache
                    .clone()
                    .or_else(fastvideo_cudarc::ltx2::text_cache::default_dir)
            };
            let text_residency = match text.as_str() {
                "auto" => TextResidency::Auto,
                "resident" => TextResidency::Resident,
                "streamed" => TextResidency::Streamed,
                other => {
                    return Err(anyhow::anyhow!(
                        "--text {other}: expected auto, resident or streamed"
                    )
                    .into())
                }
            };
            let paths = Ltx2Paths {
                weights: weights.clone(),
                dit: path.clone(),
                text: text_weights.clone(),
            };
            gen(
                report,
                *model_version,
                &paths,
                &PipelineOptions {
                    text_cache,
                    text_residency,
                },
                prompt,
                clip,
                *seed,
                device,
                *geometry,
                !*no_mp4,
                *warm,
                *two_stage,
                *diff_vae,
                image.as_deref(),
            )
        }
        Stage::SlimText {
            weights,
            slim,
            embed,
            shard_gib,
        } => slim_text(report, weights, slim, embed, *shard_gib),
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
    let file = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse safetensors {}", path.display()))?;
    let mut out = HashMap::new();
    for (name, view) in file.tensors() {
        if !prefixes.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        let word = |c: &[u8]| [c[0], c[1], c[2], c[3]];
        let data: Vec<f32> = match view.dtype() {
            Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(word(c)))
                .collect(),
            Dtype::I32 => view
                .data()
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(word(c)) as f32)
                .collect(),
            other => anyhow::bail!(
                "{}: tensor {name} is {other:?}, expected F32 or I32",
                path.display()
            ),
        };
        out.insert(name, F32Tensor::new(view.shape().to_vec(), data)?);
    }
    Ok(out)
}

fn take(
    map: &mut HashMap<String, F32Tensor>,
    name: &str,
    path: &Path,
) -> anyhow::Result<F32Tensor> {
    map.remove(name).with_context(|| {
        format!(
            "{} has no tensor `{name}` — was that ltx2_oracle.py stage skipped?",
            path.display()
        )
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
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "no --prompt, and the oracle meta {} could not be read",
            path.display()
        )
    })?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    doc.get("prompt")
        .and_then(|p| p.as_str())
        .map(str::to_string)
        .with_context(|| format!("{} has no `prompt`", path.display()))
}

// ---- weights ----------------------------------------------------------------

/// The distilled DiT/connectors: the single file, or a diffusers component
/// folder named `component` under (or at) `path`.
///
/// Also accepts `--dit …/transformer` when looking up `connectors`: the sibling
/// `…/connectors` directory is used (Diffusers split pack).
fn open_distilled(path: &Path, component: &str) -> anyhow::Result<(WeightMap, Layout)> {
    let map = if path.is_file() {
        WeightMap::open_files(&[path.to_path_buf()])?
    } else if path.join(component).is_dir() {
        WeightMap::open(&path.join(component))?
    } else if let Some(sibling) = path
        .parent()
        .map(|p| p.join(component))
        .filter(|p| p.is_dir())
    {
        WeightMap::open(&sibling)?
    } else if path.is_dir() {
        WeightMap::open(path)?
    } else {
        anyhow::bail!(
            "{} is neither a .safetensors file nor a directory",
            path.display()
        );
    };
    let layout = Keys::detect(&map);
    Ok((map, layout))
}

fn gib(map: &WeightMap, prefix: &str) -> f64 {
    map.lazy().map_or(0.0, |s| {
        s.bytes_with_prefix(prefix) as f64 / f64::from(1u32 << 30)
    })
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
    let [_, total, width] = t.shape[..] else {
        anyhow::bail!("expected [1, S, C], got {:?}", t.shape)
    };
    anyhow::ensure!(
        real <= total,
        "{real} real tokens in a reference of {total}"
    );
    Ok(&t.data[(total - real) * width..])
}

fn text(
    report: &mut Report,
    a: &TextArgs<'_>,
    [max_rel, max_rel_llm, max_rel_e2e]: [f64; 3],
) -> StageResult<()> {
    report.set("device", crate::gpu::init(a.device)?);
    let cfg: Ltx2Config = ltx2_19b_distilled();
    let max_len = cfg.defaults.max_sequence_length;
    let mut orc = load_oracle(a.oracle, &["text.", "conn."])?;

    // --- tokenizer parity: ours on the prompt vs the pipeline's padded ids ----
    let want_ids = ints(
        &take(&mut orc, "text.input_ids", a.oracle)?,
        "text.input_ids",
    )?;
    let want_mask = ints(
        &take(&mut orc, "text.attention_mask", a.oracle)?,
        "text.attention_mask",
    )?;
    let prompt = read_prompt(a.oracle, a.meta, a.prompt)?;
    let ours = PaddedPrompt::tokenize(
        &a.weights.join("tokenizer").join("tokenizer.json"),
        &prompt,
        max_len,
    )?;
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
    let reference_prompt =
        PaddedPrompt::from_ids(&want_ids[want_ids.len() - real..], want_ids.len())?;

    let states = take(&mut orc, "text.hidden_states", a.oracle)?;
    let [n, hidden, depth] = states.shape[..] else {
        return Err(anyhow::anyhow!(
            "text.hidden_states: expected [tokens, hidden, states], got {:?}",
            states.shape
        )
        .into());
    };
    if n != real {
        return Err(
            anyhow::anyhow!("text.hidden_states holds {n} tokens, the mask says {real}").into(),
        );
    }
    let oracle_stack = HiddenStack::from_interleaved(&states.data, n, hidden, depth)?;
    drop(states);

    // --- connectors on the oracle's hidden states -----------------------------
    let (map, layout) = open_distilled(a.dit, "connectors")?;
    let keys = Keys::connectors(layout);
    report.set(
        "connectors",
        json!({"source": a.dit.display().to_string(), "layout": format!("{layout:?}")}),
    );
    let timer = std::time::Instant::now();
    let connectors = TextConnectors::load(&map, &keys, &cfg.connectors)?;
    report.note(
        "load_connectors",
        json!({"seconds": timer.elapsed().as_secs_f64()}),
    );

    let host = |t: &fastvideo_cudarc::CudaTensor| -> anyhow::Result<Vec<f32>> {
        Ok(t.host_cow()?.into_owned())
    };
    let (got, seconds) = measure(report, "connectors", || {
        let out = connectors.forward(&oracle_stack, max_len)?;
        Ok((host(&out.proj)?, host(&out.video)?, host(&out.audio)?))
    })?;
    report.note(
        "connectors",
        json!({"seconds": seconds, "tokens": n, "padded_to": max_len}),
    );

    let proj_ref = take(&mut orc, "conn.proj", a.oracle)?;
    let refs: Vec<(&str, F32Tensor, F32Tensor)> = vec![
        (
            "video",
            take(&mut orc, "conn.video", a.oracle)?,
            take(&mut orc, "conn.video_f32", a.oracle)?,
        ),
        (
            "audio",
            take(&mut orc, "conn.audio", a.oracle)?,
            take(&mut orc, "conn.audio_f32", a.oracle)?,
        ),
    ];
    // Every metric lands before the first gate can stop the stage.
    let d_proj = diff(&got.0, tail_rows(&proj_ref, n)?);
    let pairs: Vec<_> = refs
        .iter()
        .zip([&got.1, &got.2])
        .map(|((name, bf16, f32_), ours)| {
            (
                *name,
                diff(ours, &f32_.data),
                diff(ours, &bf16.data),
                diff(&bf16.data, &f32_.data),
            )
        })
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
    report.set(
        "text_encoder",
        json!({"language_model_gib": gib(&gemma, "language_model."), "tokens": real}),
    );
    // The oracle's text stage runs bf16 unless told otherwise; its embedding
    // multiplier is then 62.0 rather than sqrt(3840).
    let llm_cfg = DecoderConfig::gemma3_12b_text().for_bf16_reference();
    let (our_stack, seconds) = measure(report, "gemma", || {
        Ok(HiddenStack::encode(&gemma, &llm_cfg, &reference_prompt)?)
    })?;
    report.note(
        "gemma",
        json!({"seconds": seconds, "layers": llm_cfg.num_layers(), "tokens": real}),
    );
    let per_state: Vec<_> = our_stack
        .states
        .iter()
        .zip(&oracle_stack.states)
        .map(|(o, r)| diff(o, r))
        .collect();
    report.set(
        "gemma_rel_l2_by_state",
        per_state.iter().map(|d| d.rel_l2).collect::<Vec<_>>(),
    );
    for k in [0usize, 1, 6, 24, 47, 48] {
        let Some(d) = per_state.get(k) else { continue };
        report.check(
            format!("text.hidden_{k}"),
            d.within(max_rel_llm),
            d.to_json(),
            json!({"rel_l2": max_rel_llm}),
        )?;
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
    let snr = if d.rel_l2 > 0.0 {
        -20.0 * d.rel_l2.log10()
    } else {
        f64::INFINITY
    };
    (rmse, snr)
}

fn audio(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    device: &str,
    wav: Option<&Path>,
    [max_rel_mel, min_snr, min_snr_e2e]: [f64; 3],
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let cfg = ltx2_19b_distilled();
    let mut orc = load_oracle(oracle, &["audio."])?;
    let latent = take(&mut orc, "audio.latent", oracle)?;
    let want_mel = take(&mut orc, "audio.mel", oracle)?;
    let want_wave = take(&mut orc, "audio.wave", oracle)?;
    drop(orc);

    let timer = std::time::Instant::now();
    let decoder = AudioDecoder::load(
        &WeightMap::open(&weights.join("audio_vae"))?,
        &cfg.audio_vae,
    )?;
    let vocoder = Vocoder::load(&WeightMap::open(&weights.join("vocoder"))?, &cfg.vocoder)?;
    report.note(
        "load_audio",
        json!({"seconds": timer.elapsed().as_secs_f64()}),
    );

    let packed = CudaTensor::from_vec(latent.data, latent.shape.clone())?;
    let (mel, seconds) = measure(report, "audio_vae", || Ok(decoder.decode_packed(&packed)?))?;
    report.note(
        "audio_vae",
        json!({"seconds": seconds, "latent": latent.shape, "mel": mel.shape}),
    );
    let reference_mel = CudaTensor::from_vec(want_mel.data.clone(), want_mel.shape.clone())?;
    let (wave, seconds) = measure(report, "vocoder", || Ok(vocoder.forward(&reference_mel)?))?;
    report.note(
        "vocoder",
        json!({"seconds": seconds, "wave": wave.shape, "sample_rate": vocoder.sample_rate()}),
    );
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
        write_wav(
            path,
            &interleave_audio(&e2e_h, cfg.vocoder.out_channels)?,
            cfg.vocoder.out_channels as u16,
            cfg.vocoder.output_sampling_rate as u32,
        )?;
        report.set("wav", path.display().to_string());
    }
    report.check(
        "audio.mel",
        d_mel.within(max_rel_mel),
        with(&d_mel, mel_h.len()),
        json!({"rel_l2": max_rel_mel}),
    )?;
    let snr = |d: &crate::metrics::Diff| rmse_snr(d, 1).1;
    report.check(
        "audio.wave",
        d_wave.non_finite == 0 && snr(&d_wave) >= min_snr,
        with(&d_wave, wave_h.len()),
        json!({"snr_db_min": min_snr}),
    )?;
    report.check(
        "e2e.wave",
        d_e2e.non_finite == 0 && snr(&d_e2e) >= min_snr_e2e,
        with(&d_e2e, e2e_h.len()),
        json!({"snr_db_min": min_snr_e2e}),
    )?;
    Ok(())
}

// ---- video vae --------------------------------------------------------------

fn vae(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    device: &str,
    [max_abs, max_rmse, max_abs_streamed]: [f64; 3],
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let cfg = ltx2_19b_distilled();
    let mut orc = load_oracle(oracle, &["vae."])?;
    let latent = take(&mut orc, "vae.latent", oracle)?;
    let want = take(&mut orc, "vae.video", oracle)?;
    drop(orc);

    let timer = std::time::Instant::now();
    let map = WeightMap::open(&weights.join("vae"))?;
    let decoder = VideoDecoder::load(&map, &cfg.vae)?;
    report.note(
        "load_vae",
        json!({"seconds": timer.elapsed().as_secs_f64(), "decoder_gib": gib(&map, "decoder.")}),
    );

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
    report.note(
        "vae_decode",
        json!({"seconds": seconds, "latent": latent.shape, "runs": runs}),
    );
    let ((streamed, runs), seconds) = measure(report, "vae_decode_streamed", || decode(2))?;
    report.note(
        "vae_decode_streamed",
        json!({"seconds": seconds, "chunk": 2, "runs": runs}),
    );

    let d = diff(&whole, &want.data);
    let (rmse, _) = rmse_snr(&d, whole.len());
    let mut values = d.to_json();
    values["rmse"] = json!(rmse);
    values["psnr_db"] = json!(crate::metrics::psnr(&whole, &want.data, 2.0).min(999.0));
    let d_stream = diff(&streamed, &whole);
    report.set(
        "metrics",
        json!({"video": values, "streamed_vs_whole": d_stream.to_json()}),
    );
    report.check(
        "vae.video",
        d.non_finite == 0
            && whole.len() == want.data.len()
            && d.max_abs <= max_abs
            && rmse <= max_rmse,
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

// ---- dit --------------------------------------------------------------------

fn host(t: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.host_cow()?.into_owned())
}

fn cuda(t: &F32Tensor) -> anyhow::Result<CudaTensor> {
    Ok(CudaTensor::from_vec(t.data.clone(), t.shape.clone())?)
}

/// One of our tensors against the bf16 reference and, when the oracle ran
/// `--dit-dtype both`, against the float32 one — with the distance *between*
/// the two references as the floor.
///
/// bf16 weights widen to float32 exactly, so the float32 pass is the same
/// function without rounding, and the bf16 pass is "the product". A port is
/// wrong by what separates it from float32 *beyond* what separates the product
/// from float32; against the bf16 dump alone, the reference's own noise is
/// indistinguishable from ours.
struct Judged {
    /// Without a float32 reference: gate, or only record?
    gated: bool,
    name: String,
    vs_bf16: crate::metrics::Diff,
    vs_f32: Option<(crate::metrics::Diff, crate::metrics::Diff)>,
    limit_bf16: f64,
}

impl Judged {
    fn new(
        name: String,
        ours: &[f32],
        bf16: &F32Tensor,
        f32_ref: Option<&F32Tensor>,
        limit_bf16: f64,
    ) -> Self {
        Self {
            gated: true,
            name,
            vs_bf16: diff(ours, &bf16.data),
            vs_f32: f32_ref.map(|r| (diff(ours, &r.data), diff(&bf16.data, &r.data))),
            limit_bf16,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        match &self.vs_f32 {
            Some((ours, floor)) => {
                json!({"vs_bf16": self.vs_bf16.to_json(), "vs_f32": ours.to_json(), "reference_bf16_vs_f32": floor.to_json()})
            }
            None => json!({"vs_bf16": self.vs_bf16.to_json()}),
        }
    }

    /// With a float32 reference: ours must be within `factor` of the floor (or
    /// under an absolute 1e-3, below which the comparison is rounding on both
    /// sides). Without one: the published bf16 limit.
    fn check(&self, report: &mut Report, factor: f64) -> StageResult<()> {
        match &self.vs_f32 {
            Some((ours, floor)) => {
                let limit = (factor * floor.rel_l2).max(1e-3);
                report.check(
                    self.name.clone(),
                    ours.within(limit),
                    json!({"rel_l2_vs_f32": ours.rel_l2, "cosine_vs_f32": ours.cosine, "reference_bf16_vs_f32": floor.rel_l2, "rel_l2_vs_bf16": self.vs_bf16.rel_l2}),
                    json!({"rel_l2_vs_f32": limit, "floor_factor": factor}),
                )
            }
            None if !self.gated => {
                let mut values = self.vs_bf16.to_json();
                values["reference_line"] = json!(self.limit_bf16);
                values["gated"] = json!(false);
                report.note(self.name.clone(), values);
                Ok(())
            }
            None => {
                // A relative error r between near-parallel vectors costs about r²/2
                // of cosine; r² leaves room without admitting a sign or scale slip.
                let cosine_min = 1.0 - self.limit_bf16 * self.limit_bf16;
                report.check(
                    self.name.clone(),
                    self.vs_bf16.within(self.limit_bf16) && self.vs_bf16.cosine >= cosine_min,
                    self.vs_bf16.to_json(),
                    json!({"rel_l2": self.limit_bf16, "cosine_min": cosine_min}),
                )
            }
        }
    }
}

/// Every `stride`-th token of `[1, S, C]`, on the device — the oracle keeps the
/// sub-layer taps of the long (video) stream that way and the short one whole.
fn strided(t: &CudaTensor, stride: usize) -> anyhow::Result<Vec<f32>> {
    let [_, tokens, width] = t.shape[..] else {
        anyhow::bail!("tap of shape {:?}", t.shape)
    };
    if stride <= 1 || tokens <= 1024 {
        return host(t);
    }
    let rows: Vec<usize> = (0..tokens).step_by(stride).collect();
    host(&t.reshape(vec![tokens, width])?.index_select_rows(&rows)?)
}

fn dit(
    report: &mut Report,
    path: &Path,
    oracle: &Path,
    device: &str,
    g: Geometry,
    rope_only: bool,
    [max_abs_rope, max_rel_first, max_rel, floor_factor]: [f64; 4],
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let cfg = ltx2_19b_distilled();
    let t_cfg = &cfg.transformer;
    let mut orc = load_oracle(oracle, &["dit.", "dit32.", "conn.video", "conn.audio"])?;
    let video_in = take(&mut orc, "dit.video_in", oracle)?;
    let audio_in = take(&mut orc, "dit.audio_in", oracle)?;
    let grid = t_cfg.latent_grid(g.num_frames, g.height, g.width);
    let audio_tokens = t_cfg.audio_tokens(g.num_frames, g.frame_rate);
    let geometry_ok = video_in.shape == [1, grid.iter().product::<usize>(), t_cfg.in_channels]
        && audio_in.shape == [1, audio_tokens, t_cfg.audio_in_channels];
    report.check(
        "dit.geometry",
        geometry_ok,
        json!({"video_in": video_in.shape, "audio_in": audio_in.shape}),
        json!({"latent_grid": grid, "audio_tokens": audio_tokens, "hint": "pass the oracle's --height/--width/--num-frames/--frame-rate"}),
    )?;

    // --- rotary tables: pure host arithmetic, judged to the last digits --------
    // Built the way torch builds them on CUDA (scalar division as a multiply by
    // the float32 reciprocal), which is where the oracle runs.
    let tables = Ltx2RopeTables::new(t_cfg, grid, audio_tokens, g.frame_rate as f32);
    let named: [(&str, &SplitRope); 4] = [
        ("video", &tables.video),
        ("audio", &tables.audio),
        ("cross_video", &tables.cross_video),
        ("cross_audio", &tables.cross_audio),
    ];
    let mut rope_diffs = Vec::new();
    for (name, ours) in named {
        for (part, values) in [("cos", &ours.cos), ("sin", &ours.sin)] {
            let want = take(&mut orc, &format!("dit.rope.{name}.{part}"), oracle)?;
            let shape_ok = want.shape == [1, ours.heads, ours.tokens, ours.half];
            rope_diffs.push((
                format!("dit.rope.{name}.{part}"),
                shape_ok,
                want.shape.clone(),
                diff(values, &want.data),
            ));
        }
    }
    for (name, shape_ok, shape, d) in &rope_diffs {
        report.check(
            name.clone(),
            *shape_ok && d.non_finite == 0 && d.max_abs <= max_abs_rope,
            d.to_json(),
            json!({"max_abs": max_abs_rope, "reference_shape": shape}),
        )?;
    }
    if rope_only {
        return Ok(());
    }

    // --- one forward on the oracle's inputs -----------------------------------
    let timestep = take(&mut orc, "dit.timestep", oracle)?
        .data
        .first()
        .copied()
        .context("dit.timestep is empty")?;
    let stride = orc
        .remove("dit.tap_stride")
        .and_then(|t| t.data.first().copied())
        .map_or(1, |s| s as usize);
    let has_f32 = orc.contains_key("dit32.video_out");
    let (map, layout) = open_distilled(path, "transformer")?;
    let keys = Keys::transformer(layout);
    report.set(
        "dit",
        json!({
            "source": path.display().to_string(), "layout": format!("{layout:?}"), "timestep": timestep,
            "video_tokens": video_in.shape[1], "audio_tokens": audio_in.shape[1], "float32_reference": has_f32, "tap_stride": stride,
        }),
    );
    let peak = crate::gpu::PeakMem::start();
    let timer = std::time::Instant::now();
    let model = Ltx2Transformer::load(&map, &keys, t_cfg)?;
    report.note(
        "load_dit",
        json!({"seconds": timer.elapsed().as_secs_f64()}),
    );

    let ropes = Ropes::upload(&tables)?;
    let text = model.project_text(
        &cuda(&take(&mut orc, "conn.video", oracle)?)?,
        &cuda(&take(&mut orc, "conn.audio", oracle)?)?,
    )?;
    let (video, audio) = (cuda(&video_in)?, cuda(&audio_in)?);
    // What the oracle tapped: block outputs `dit.blockNN.{video,audio}` and, inside
    // some blocks and at the heads, sub-layer taps under the names the DiT's probe
    // emits. Only what the file holds is downloaded.
    let wanted: std::collections::HashSet<String> = orc
        .keys()
        .filter_map(|k| k.strip_prefix("dit."))
        .map(str::to_string)
        .collect();
    let mut ours: Vec<(String, Vec<f32>)> = Vec::new();
    let mut probe_err: Option<anyhow::Error> = None;
    let ((v_out, a_out), seconds) = measure(report, "dit_forward", || {
        let mut block_taps: Vec<(String, Vec<f32>)> = Vec::new();
        let mut observe = |i: usize,
                           v: &CudaTensor,
                           a: &CudaTensor|
         -> fastvideo_cudarc::wan::tensor::Result<()> {
            for (stream, t) in [("video", v), ("audio", a)] {
                let name = format!("block{i:02}.{stream}");
                if wanted.contains(&name) {
                    block_taps.push((name, t.host_cow()?.into_owned()));
                }
            }
            Ok(())
        };
        let mut probe = |name: &str, t: &CudaTensor| -> fastvideo_cudarc::wan::tensor::Result<()> {
            if wanted.contains(name) && probe_err.is_none() {
                match strided(t, stride) {
                    Ok(values) => ours.push((name.to_string(), values)),
                    Err(e) => probe_err = Some(e),
                }
            }
            Ok(())
        };
        let (v, a) = model.forward_probed(
            &video,
            &audio,
            &text,
            timestep,
            &ropes,
            Some(&mut observe),
            Some(&mut probe),
        )?;
        ours.extend(block_taps);
        Ok((host(&v)?, host(&a)?))
    })?;
    if let Some(e) = probe_err {
        return Err(e.into());
    }
    report.note("dit_forward", json!({"seconds": seconds, "includes": "tap downloads", "taps": ours.len(), "peak_vram_mib": peak.stop()}));
    ours.push(("video_out".into(), v_out));
    ours.push(("audio_out".into(), a_out));
    // In execution order, so the report reads as the forward does.
    let order = |n: &str| -> (usize, usize) {
        let block = n
            .strip_prefix("block")
            .and_then(|r| r.get(..2))
            .and_then(|b| b.parse().ok())
            .unwrap_or(usize::MAX);
        let step = [
            "attn1_in",
            "attn1_out",
            "attn1_after",
            "attn2_in",
            "attn2_out",
            "attn2_after",
            "av_in",
            "av_out",
            "av_after",
            "ff_in",
            "ff_out",
        ]
        .iter()
        .position(|s| n.ends_with(s))
        .unwrap_or(99);
        (block, step)
    };
    ours.sort_by_key(|(n, _)| (order(n), n.clone()));

    // Every metric lands before the first gate can stop the stage: the first tap
    // at which ours leaves the floor is the diagnosis.
    let mut judged = Vec::new();
    for (name, values) in &ours {
        let bf16 = take(&mut orc, &format!("dit.{name}"), oracle)?;
        let f32_ref = orc.remove(&format!("dit32.{name}"));
        let limit = if name.starts_with("block00") {
            max_rel_first
        } else {
            max_rel
        };
        judged.push(Judged::new(
            format!("dit.{name}"),
            values,
            &bf16,
            f32_ref.as_ref(),
            limit,
        ));
    }
    report.set(
        "metrics",
        judged
            .iter()
            .map(|j| (j.name.clone(), j.to_json()))
            .collect::<serde_json::Map<_, _>>(),
    );
    for j in &judged {
        j.check(report, floor_factor)?;
    }
    Ok(())
}

// ---- sampling loop ----------------------------------------------------------

/// Frames `keep` of a full-size decode, each `[3, H, W]` on the host, without
/// holding the clip: the sink downloads only the frames asked for.
fn decode_frames(
    dec: &Decoders,
    tokens: &CudaTensor,
    grid: [usize; 3],
    keep: &[usize],
) -> anyhow::Result<Vec<Vec<f32>>> {
    let z = fastvideo_cudarc::ltx2::transformer::unpack_video(tokens, grid)?;
    let mut out: Vec<Option<Vec<f32>>> = vec![None; keep.len()];
    dec.video.decode_streaming(&z, &mut |offset, frames| {
        for (slot, &k) in out.iter_mut().zip(keep) {
            if k >= offset && k < offset + frames.shape[0] {
                *slot = Some(frames.narrow(0, k - offset, 1)?.host_cow()?.into_owned());
            }
        }
        Ok(())
    })?;
    out.into_iter()
        .enumerate()
        .map(|(i, f)| f.with_context(|| format!("frame {} was never decoded", keep[i])))
        .collect()
}

fn sample_loop(
    report: &mut Report,
    path: &Path,
    oracle: &Path,
    weights: Option<&Path>,
    device: &str,
    g: Geometry,
    [max_rel_first, max_rel, min_psnr, min_psnr_e2e, floor_factor]: [f64; 5],
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let cfg = ltx2_19b_distilled();
    let t_cfg = &cfg.transformer;
    let mut orc = load_oracle(
        oracle,
        &["sample.", "sample32.", "conn.video", "conn.audio"],
    )?;
    let noise_v = take(&mut orc, "sample.video_noise", oracle)?;
    let noise_a = take(&mut orc, "sample.audio_noise", oracle)?;
    let grid = t_cfg.latent_grid(g.num_frames, g.height, g.width);
    let audio_tokens = t_cfg.audio_tokens(g.num_frames, g.frame_rate);
    report.check(
        "loop.geometry",
        noise_v.shape == [1, grid.iter().product::<usize>(), t_cfg.in_channels]
            && noise_a.shape == [1, audio_tokens, t_cfg.audio_in_channels],
        json!({"video_noise": noise_v.shape, "audio_noise": noise_a.shape}),
        json!({"latent_grid": grid, "audio_tokens": audio_tokens}),
    )?;

    let peak = crate::gpu::PeakMem::start();
    let (map, layout) = open_distilled(path, "transformer")?;
    let timer = std::time::Instant::now();
    let model = Ltx2Transformer::load(&map, &Keys::transformer(layout), t_cfg)?;
    report.note(
        "load_dit",
        json!({"seconds": timer.elapsed().as_secs_f64(), "layout": format!("{layout:?}")}),
    );
    let ropes = Ropes::new(t_cfg, grid, audio_tokens, g.frame_rate as f32)?;
    let text = model.project_text(
        &cuda(&take(&mut orc, "conn.video", oracle)?)?,
        &cuda(&take(&mut orc, "conn.audio", oracle)?)?,
    )?;

    let schedule = Ltx2Schedule::distilled();
    let mut steps: Vec<(Vec<f32>, Vec<f32>, f64)> = Vec::new();
    let ((final_v, final_a), seconds) = measure(report, "denoise", || {
        let mut observe = |_: usize,
                           v: &CudaTensor,
                           a: &CudaTensor,
                           s: f64|
         -> fastvideo_cudarc::wan::pipeline::Result<()> {
            steps.push((v.host_cow()?.into_owned(), a.host_cow()?.into_owned(), s));
            Ok(())
        };
        Ok(ltx2_pipeline::denoise(
            &model,
            &text,
            &ropes,
            &schedule,
            cuda(&noise_v)?,
            cuda(&noise_a)?,
            Some(&mut observe),
        )?)
    })?;
    report.note("denoise", json!({"seconds": seconds, "step_seconds": steps.iter().map(|s| s.2).collect::<Vec<_>>(), "peak_vram_mib": peak.stop()}));
    drop((model, text, ropes));

    // Every step's metric lands before the first gate can stop the stage: where
    // the drift starts is the diagnosis.
    let last = steps.len().saturating_sub(1);
    let mut results: Vec<Judged> = Vec::new();
    let mut oracle_final: Option<(F32Tensor, F32Tensor)> = None;
    for (i, (v, a, _)) in steps.iter().enumerate() {
        // The first half of the schedule covers sigma 1 → 0.975: gated tight.
        let early = i < steps.len() / 2;
        let limit = if early { max_rel_first } else { max_rel };
        let (want_v, want_a) = (
            take(&mut orc, &format!("sample.step{i}.video"), oracle)?,
            take(&mut orc, &format!("sample.step{i}.audio"), oracle)?,
        );
        let (v32, a32) = (
            orc.remove(&format!("sample32.step{i}.video")),
            orc.remove(&format!("sample32.step{i}.audio")),
        );
        for (stream, ours, want, want32) in [
            ("video", v, &want_v, v32.as_ref()),
            ("audio", a, &want_a, a32.as_ref()),
        ] {
            results.push(Judged {
                gated: early,
                ..Judged::new(format!("loop.step{i}.{stream}"), ours, want, want32, limit)
            });
        }
        if i == last {
            oracle_final = Some((want_v, want_a));
        }
    }
    report.set(
        "metrics",
        results
            .iter()
            .map(|j| (j.name.clone(), j.to_json()))
            .collect::<serde_json::Map<_, _>>(),
    );
    for j in &results {
        j.check(report, floor_factor)?;
    }

    // --- decoded output, when the decoders are at hand -------------------------
    let (Some(weights), Some((oracle_v, oracle_a))) = (weights, oracle_final) else {
        return Ok(());
    };
    let Some(frames_ref) = orc.remove("sample.frames") else {
        report.note(
            "loop.decode",
            json!({"skipped": "the oracle file has no sample.frames"}),
        );
        return Ok(());
    };
    let decoders = Decoders::load(weights, &cfg)?;
    let total = cfg.vae.decoded_frames(grid[0]);
    let mut keep = vec![0, total / 2, total - 1];
    keep.dedup();
    let [_, c, kept, h, w] = frames_ref.shape[..] else {
        return Err(anyhow::anyhow!(
            "sample.frames: expected [1, 3, K, H, W], got {:?}",
            frames_ref.shape
        )
        .into());
    };
    if kept != keep.len() || c != 3 {
        return Err(anyhow::anyhow!(
            "sample.frames holds {kept} frames, expected frames {keep:?} of {total}"
        )
        .into());
    }
    // Reference frame k out of the [C, K, H, W] layout, as [3, H, W].
    let plane = h * w;
    let reference = |k: usize| -> Vec<f32> {
        (0..c)
            .flat_map(|ch| {
                frames_ref.data[(ch * kept + k) * plane..(ch * kept + k + 1) * plane]
                    .iter()
                    .copied()
            })
            .collect()
    };
    for (tag, tokens, limit) in [
        ("oracle_latents", cuda(&oracle_v)?, min_psnr),
        ("e2e", final_v.clone(), min_psnr_e2e),
    ] {
        let (ours, seconds) = measure(report, &format!("vae_decode_{tag}"), || {
            decode_frames(&decoders, &tokens, grid, &keep)
        })?;
        let psnr: Vec<f64> = ours
            .iter()
            .enumerate()
            .map(|(k, f)| crate::metrics::psnr(f, &reference(k), 2.0).min(999.0))
            .collect();
        let worst = psnr.iter().copied().fold(f64::INFINITY, f64::min);
        report.check(
            format!("loop.frames_{tag}"),
            worst >= limit,
            json!({"psnr_db": psnr, "frames": keep, "decode_seconds": seconds}),
            json!({"psnr_db_min": limit}),
        )?;
    }
    if let (Some(want_mel), Some(want_wave)) = (orc.remove("sample.mel"), orc.remove("sample.wave"))
    {
        for (tag, tokens, limit) in [
            ("oracle_latents", cuda(&oracle_a)?, 50.0),
            ("e2e", final_a.clone(), 20.0),
        ] {
            let mel = decoders.audio.decode_packed(&tokens)?;
            let wave = decoders.vocoder.forward(&mel)?;
            let (d_mel, d_wave) = (
                diff(&host(&mel)?, &want_mel.data),
                diff(&host(&wave)?, &want_wave.data),
            );
            let snr = rmse_snr(&d_wave, 1).1.min(999.0);
            let mut values = d_wave.to_json();
            values["snr_db"] = json!(snr);
            values["mel_rel_l2"] = json!(d_mel.rel_l2);
            // A vocoder is phase-sensitive: on our own latents the waveform SNR
            // is reported, and only the attributed decode is gated.
            let ok = d_wave.non_finite == 0 && (tag == "e2e" || snr >= limit);
            report.check(format!("loop.wave_{tag}"), ok, values, json!({"snr_db_min": if tag == "e2e" { serde_json::Value::Null } else { json!(limit) }}))?;
        }
    }
    Ok(())
}

// ---- gen --------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn gen(
    report: &mut Report,
    model_version: ModelVersion,
    paths: &Ltx2Paths,
    options: &PipelineOptions,
    prompt: &str,
    clip: &Path,
    seed: u64,
    device: &str,
    g: Geometry,
    mp4: bool,
    warm: bool,
    two_stage: bool,
    diff_vae: bool,
    image: Option<&Path>,
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    report.set(
        "model_version",
        match model_version {
            ModelVersion::V20 => "2.0",
            ModelVersion::V23 => "2.3",
            ModelVersion::V25 => "2.5",
        },
    );
    report.set("two_stage", two_stage);
    report.set("diff_vae", diff_vae);
    let cfg = model_version.config();
    let request = Ltx2Request {
        prompt: prompt.to_string(),
        height: g.height,
        width: g.width,
        num_frames: g.num_frames,
        frame_rate: g.frame_rate,
        seed,
        output_dir: clip.to_path_buf(),
        mp4,
        two_stage,
        diff_vae,
        negative_prompt: String::new(),
        guidance_scale: 1.0,
        audio_guidance_scale: 1.0,
        num_inference_steps: None,
        refine_steps: None,
        image_path: image.map(Path::to_path_buf),
    };
    report.set(
        "request",
        json!({
            "prompt": prompt, "height": g.height, "width": g.width, "num_frames": g.num_frames,
            "frame_rate": g.frame_rate, "seed": seed, "two_stage": two_stage, "diff_vae": diff_vae,
            "image": image.map(|p| p.display().to_string()),
        }),
    );
    let peak = crate::gpu::PeakMem::start();
    let wall = std::time::Instant::now();
    let mut pipeline = Ltx2Pipeline::load(paths, &cfg, options)?;
    if warm {
        // Untimed: first-use costs (cuDNN plan search, allocator growth, page
        // cache) land here. The cache is bypassed so this run cannot turn the
        // timed one into a hit it would not otherwise have been.
        let timer = std::time::Instant::now();
        let warmup = Ltx2Request {
            output_dir: clip.join("warmup"),
            mp4: false,
            ..request.clone()
        };
        pipeline.generate(&warmup, false, None)?;
        report.note("warmup", json!({"seconds": timer.elapsed().as_secs_f64()}));
    }
    // Latent statistics per step: a run that diverges shows here, not in a grey video.
    let mut stats: Vec<serde_json::Value> = Vec::new();
    let mut observe = |i: usize,
                       v: &CudaTensor,
                       a: &CudaTensor,
                       s: f64|
     -> fastvideo_cudarc::wan::pipeline::Result<()> {
        let (vh, ah) = (v.host_cow()?, a.host_cow()?);
        let ((vm, vs), (am, as_)) = (crate::metrics::mean_std(&vh), crate::metrics::mean_std(&ah));
        let bad = crate::metrics::non_finite(&vh) + crate::metrics::non_finite(&ah);
        stats.push(json!({"step": i, "seconds": s, "video_mean": vm, "video_std": vs, "audio_mean": am, "audio_std": as_, "non_finite": bad}));
        Ok(())
    };
    let timed = std::time::Instant::now();
    let out = pipeline.generate(&request, true, Some(&mut observe))?;
    let generate_s = timed.elapsed().as_secs_f64();
    let wall_s = wall.elapsed().as_secs_f64();
    let peak_mib = peak.stop();
    let t = &out.timings;
    report.set(
        "timings",
        json!({
            "warm": warm, "wall_s": wall_s, "generate_s": generate_s, "load_s": pipeline.load_s, "text_s": t.text_s,
            "denoise_s": t.denoise_s, "stage1_s": t.stage1_s, "upsample_s": t.upsample_s, "stage2_s": t.stage2_s,
            "step_s": t.step_s, "decode_audio_s": t.decode_audio_s, "decode_video_s": t.decode_video_s, "write_s": t.write_s,
        }),
    );
    report.set(
        "text",
        json!({
            "cache": out.text.cache.as_str(), "mode": out.text.mode, "seconds": out.text.seconds, "tokens": out.text.tokens,
            "cache_dir": options.text_cache.as_ref().map(|p| p.display().to_string()), "key": out.text.key,
            "text_weights": paths.text.as_ref().map(|p| p.display().to_string()),
        }),
    );
    report.set("peak_vram_mib", peak_mib);
    report.set("steps", &stats);
    report.set("outputs", json!({"mp4": out.mp4, "wav": out.wav, "frames": out.frames.len(), "first_frame": out.frames.first()}));
    report.set(
        "tokens",
        json!({"prompt": out.prompt_tokens, "video": out.video_tokens, "audio": out.audio_tokens}),
    );

    let want_steps = if two_stage { 11 } else { 8 };
    let finite = stats
        .iter()
        .all(|s| s["non_finite"] == json!(0) && s["video_std"].as_f64().is_some_and(|v| v > 1e-4));
    report.check(
        "gen.latents_finite",
        finite && stats.len() == want_steps,
        json!({"steps": stats.len()}),
        json!({"steps": want_steps, "non_finite": 0}),
    )?;
    report.check(
        "gen.frames",
        out.frames.len() == g.num_frames,
        json!({"frames": out.frames.len()}),
        json!({"frames": g.num_frames}),
    )?;
    let wav_bytes = std::fs::metadata(&out.wav).map(|m| m.len()).unwrap_or(0);
    let want_samples = cfg
        .vocoder
        .waveform_samples(cfg.audio_vae.mel_frames(out.audio_tokens));
    report.check(
        "gen.wav",
        wav_bytes == 44 + (want_samples * cfg.vocoder.out_channels * 2) as u64,
        json!({"bytes": wav_bytes}),
        json!({"samples_per_channel": want_samples, "sample_rate": cfg.vocoder.output_sampling_rate, "channels": cfg.vocoder.out_channels}),
    )?;
    if mp4 {
        let size = out
            .mp4
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map_or(0, |m| m.len());
        report.check(
            "gen.mp4",
            size > 0,
            json!({"path": out.mp4, "bytes": size}),
            json!({"bytes_min": 1}),
        )?;
    }
    Ok(())
}

// ---- slim text encoder ------------------------------------------------------

fn slim_text(
    report: &mut Report,
    weights: &Path,
    slim: &Path,
    embed: &str,
    shard_gib: f64,
) -> StageResult<()> {
    let embed = match embed {
        "f32" => EmbedDtype::F32,
        "bf16" => EmbedDtype::Bf16,
        other => return Err(anyhow::anyhow!("--embed {other}: expected f32 or bf16").into()),
    };
    let options = SlimOptions {
        embed,
        shard_bytes: (shard_gib.max(0.001) * f64::from(1u32 << 30)) as u64,
    };
    let cfg = DecoderConfig::gemma3_12b_text();
    let timer = std::time::Instant::now();
    let out = write_slim_decoder(
        &weights.join("text_encoder"),
        &slim.join("text_encoder"),
        &cfg,
        &options,
    )?;
    let copied = copy_dir_files(&weights.join("tokenizer"), &slim.join("tokenizer"))?;
    let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
    report.set(
        "slim",
        json!({
            "seconds": timer.elapsed().as_secs_f64(), "tensors": out.tensors, "narrowed": out.narrowed,
            "bytes_in": out.bytes_in, "bytes_out": out.bytes_out, "gib_in": gib(out.bytes_in), "gib_out": gib(out.bytes_out),
            "files": out.files.iter().map(|(p, n)| json!({"path": p.display().to_string(), "bytes": n})).collect::<Vec<_>>(),
            "tokenizer_files": copied, "use_with": format!("ltx2 gen --text-weights {}", slim.display()),
        }),
    );
    // The written directory must be exactly what the decoder asks for.
    let store = WeightMap::open(&slim.join("text_encoder"))?;
    let lazy = store.lazy().context("slim store is lazy")?;
    let order = fastvideo_cudarc::ltx2::slim::load_order(&cfg);
    let missing = order.iter().filter(|k| !lazy.contains(k)).count();
    report.check(
        "slim.keys",
        missing == 0 && lazy.len() == order.len() && copied > 0,
        json!({"written": lazy.len(), "missing": missing, "tokenizer_files": copied}),
        json!({"expected": order.len()}),
    )?;
    Ok(())
}
