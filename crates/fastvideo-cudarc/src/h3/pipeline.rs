//! FastH3 text-to-audio+video, end to end: prompt, 8 DMD forwards, both
//! decoders, one mp4 with an audio track.
//!
//! **The loop is not the Wan DMD loop.** One forward serves both modalities,
//! each on its own shifted schedule (video 10, audio 3). The model predicts a
//! data-ward velocity, so `x0 = x + sigma * v` — a plus — and the step is the
//! deterministic blend `x' = r x + (1 - r) x0` with `r = sigma' / sigma`; no
//! noise is re-injected between rungs (`eta = 0`). `sigma` for the `x0`
//! estimate is recovered from the float32 timestep while `r` comes off the
//! sigma grid, as the reference keeps them ([`H3Schedule::step_coeffs`]).
//!
//! **What is resident.** [`H3Pipeline`] loads once and generates many times:
//! the DiT (41 GB as bf16 with the VSA gates, 37 GB dense), the text refiner
//! (1.6 GB), the video VAE (4.9 GB as bf16 linears) and the audio VAE (0.3 GB)
//! stay on the device, about 48 GB. A 5 s clip adds ~12 GB of activations
//! (measured peak 53.6 GB with the decoders loaded late), a 15 s clip ~25 GB.
//!
//! **The text encoder is the one component with a choice**
//! ([`TextEncoderChoice`]), because it is 50 GB of weights for milliseconds of
//! compute:
//!
//! * *streamed*: nothing resident, ~10 s of host-to-device traffic
//!   per **new** prompt; the conditioning cache makes a repeated prompt free;
//! * *resident* (what `Auto` picks on a large empty card): pays the load once. On a 96 GB card the arithmetic decides the
//!   precision: DiT 41 + decoders 7 + activations 12 = 60 GB leaves ~36 GB, so
//!   Qwen as FP8-E4M3 (~25 GB) fits at 5 s and bf16 Qwen (50 GB) does not fit at
//!   any length. At 15 s (activations ~25 GB) even FP8 leaves only ~10 GB of
//!   slack, so resident mode is for short clips or for a process that drops it
//!   before denoising. The resident encoder itself lives in [`crate::llm`];
//!   this module only holds the seam ([`HiddenStateEncoder`]).
//!
//! Audio decodes before video: it takes milliseconds and the WAV has to exist
//! before ffmpeg is spawned; the video VAE then streams 17-frame chunks
//! straight into the muxer.

use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_models::h3::config::{
    H3AudioVaeConfig, H3Geometry, H3InferenceContract, H3SigmaSource, H3TransformerConfig,
    H3VideoVaeConfig, H3_AUDIO_CHANNELS, H3_FPS,
};
use fastvideo_models::h3::lora::{is_sol_h3_recipe, sol_h3_forces_ref2va, SolH3AdapterSpec};
use fastvideo_models::h3::packing::{patchify, H3PackedLayout, KeyframeAnchor};
use fastvideo_models::h3::reference::{
    plan_reference_video_canvas, resample_reference_frames, resolve_reference_image_size_with,
    trim_reference_num_frames, validate_references, H3ReferenceSpec, PreparedImageRef,
    PreparedReference, ReferenceImageResize, ReferenceKind,
};
use fastvideo_models::h3::schedule::{H3JointSchedule, H3Schedule};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::audio_vae::H3AudioDecoder;
use super::text::{CacheStatus, HiddenStateEncoder};
use super::transformer::{AttnMode, DeviceLayout, H3SolPolicy, H3TextRefiner, H3Transformer};
use super::vae::H3VideoDecoder;
use super::vae_encoder::H3VideoEncoder;
use super::vsa::H3Vsa;
use crate::wan::pipeline::{
    frames_to_rgb8, interleave_audio, write_wav, PipelineError, Result, VideoWriter,
};
use crate::wan::taehv::{TaeArch, TaeHv};
use crate::wan::tensor::{CudaTensor, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct H3Request {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    /// Aligned up to `17 n + 5`; 5 to 15 seconds at 24 fps.
    pub num_frames: usize,
    /// Write `output.mp4` (needs ffmpeg) next to the PNG frames.
    pub mp4: bool,
    /// FL2VA first-frame image (canvas-fitted, GPU VAE encode).
    pub first_image: Option<std::path::PathBuf>,
    /// FL2VA last-frame image.
    pub last_image: Option<std::path::PathBuf>,
    /// Ordered Ref2VA references (image / video / audio).
    pub references: Vec<H3ReferenceSpec>,
}

impl H3Request {
    /// The default 16:9 canvas (768 x 1344) for a whole number of seconds.
    pub fn seconds(
        prompt: impl Into<String>,
        seconds: usize,
        seed: u64,
    ) -> std::result::Result<Self, String> {
        let g = H3Geometry::default_16x9(seconds)?;
        Ok(Self {
            prompt: prompt.into(),
            seed,
            height: g.height,
            width: g.width,
            num_frames: g.num_frames,
            mp4: true,
            first_image: None,
            last_image: None,
            references: Vec::new(),
        })
    }

    /// Ordered FL2VA anchors implied by the request images.
    pub fn keyframe_anchors(&self) -> Vec<KeyframeAnchor> {
        let mut a = Vec::new();
        if self.first_image.is_some() {
            a.push(KeyframeAnchor::First);
        }
        if self.last_image.is_some() {
            a.push(KeyframeAnchor::Last);
        }
        a
    }

    pub fn is_ref2va(&self) -> bool {
        !self.references.is_empty()
    }
}

/// How prompts are encoded; see the module docs for the memory arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextEncoderChoice {
    /// [`Self::ResidentFp8`] when the card is large and empty enough
    /// ([`AUTO_RESIDENT_FREE_BYTES`] free before anything loads), else streamed.
    #[default]
    Auto,
    /// One decoder layer on the device at a time, per prompt. Nothing resident.
    Streamed,
    /// Layers 0..=49 resident as bf16: 50 GB. Does not fit beside the DiT on a
    /// 96 GB card; for an encode-only process or a larger device.
    ResidentBf16,
    /// Resident with weight-only FP8 rows (24.4 GB): fits beside the DiT at 5 s.
    ResidentFp8,
    /// SearchingMan recovered Qwen3-VL-8B + ARA + 4096→5120 adapter (~11 GiB).
    Recovered8b,
}

/// Free device memory at which `Auto` keeps the encoder resident: DiT 41 +
/// decoders 7 + FP8 Qwen 24.4 + ~12 GB of 5 s activations = 85 GB.
pub const AUTO_RESIDENT_FREE_BYTES: u64 = 85_000_000_000;

impl TextEncoderChoice {
    pub fn parse(name: &str) -> std::result::Result<Self, String> {
        match name {
            "auto" => Ok(Self::Auto),
            "streamed" => Ok(Self::Streamed),
            "resident-bf16" => Ok(Self::ResidentBf16),
            "resident-fp8" => Ok(Self::ResidentFp8),
            "recovered-8b" => Ok(Self::Recovered8b),
            other => Err(format!(
                "unknown text encoder '{other}' (auto|streamed|resident-fp8|resident-bf16|recovered-8b)"
            )),
        }
    }

    /// `Auto` decided by the free device memory measured before any load
    /// (`None`: no device, or it would not say). Explicit choices pass through.
    pub fn resolve(self, free_bytes: Option<u64>) -> Self {
        match self {
            Self::Auto if free_bytes.is_some_and(|f| f >= AUTO_RESIDENT_FREE_BYTES) => {
                Self::ResidentFp8
            }
            Self::Auto => Self::Streamed,
            explicit => explicit,
        }
    }

    fn precision(self) -> Option<crate::llm::WeightPrecision> {
        match self {
            Self::ResidentBf16 => Some(crate::llm::WeightPrecision::Native),
            Self::ResidentFp8 => Some(crate::llm::WeightPrecision::Fp8Rows),
            Self::Auto | Self::Streamed | Self::Recovered8b => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct H3PipelineOptions {
    /// Dense attention without the compression gate: the parity mode the
    /// diffusers oracle judges. The checkpoint was distilled with VSA-H3, so
    /// the default (`false`) is what it should be served with. A dense recipe
    /// forces this to `true`.
    pub dense: bool,
    /// Memoize the precomputed AdaLN table here (155 MB). Building it reads
    /// 26 GB of projections nothing else needs; the file is validated against
    /// the checkpoint and the ladder.
    pub adaln_cache: Option<PathBuf>,
    /// Root holding `tokenizer/` and `text_encoder/` when it is not the
    /// snapshot itself, e.g. the slim re-pack of [`super::slim`].
    pub text_root: Option<PathBuf>,
    /// Conditioning cache directory ([`super::text_cache`]); `None` disables it.
    pub text_cache: Option<PathBuf>,
    pub text_encoder: TextEncoderChoice,
    /// Directory or `taeh3.safetensors` file. When set, the official ViT
    /// decoder is not loaded. `FASTVIDEO_TAEH3_WEIGHTS` is the same switch
    /// when this is `None`.
    pub taeh3: Option<PathBuf>,
    /// Named DMD recipe (`8step`, `4step-vsa`, `4step-dense`, `sol-h3`, `sol-h3-spark`, `sol-h3-rtx`). When unset,
    /// `fastvideo_inference.json` under the weight root (or `transformer/`) is
    /// read if present; otherwise the 8-step V2 contract.
    pub recipe: Option<String>,
    /// Load `transformer_ref/` (Ref2VA) instead of `transformer/` (T2AV/FL2VA).
    pub ref2va: bool,
    /// Sol-H3 adapter file. When unset, `sol-h3` searches beside the weight root.
    pub adapter: Option<PathBuf>,
    /// Ref2VA reference-image fit. `Auto` is 2048 short-edge, and `match` for Sol-H3.
    pub reference_image_resize: ReferenceImageResize,
}

/// Resolve the inference contract: explicit recipe name, then
/// `fastvideo_inference.json`, then the 8-step V2 default.
pub fn resolve_contract(root: &Path, recipe: Option<&str>) -> Result<H3InferenceContract> {
    if let Some(name) = recipe {
        return H3InferenceContract::named(name).map_err(msg);
    }
    for rel in [
        "fastvideo_inference.json",
        "transformer/fastvideo_inference.json",
    ] {
        let path = root.join(rel);
        if path.is_file() {
            return contract_from_inference_json(&path);
        }
    }
    Ok(H3InferenceContract::fasth3_8step())
}

fn contract_from_inference_json(path: &Path) -> Result<H3InferenceContract> {
    let text =
        std::fs::read_to_string(path).map_err(|e| msg(format!("{}: {e}", path.display())))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| msg(format!("{}: {e}", path.display())))?;
    let rungs = v
        .get("dmd_denoising_steps")
        .and_then(|x| x.as_array())
        .ok_or_else(|| msg(format!("{}: missing dmd_denoising_steps", path.display())))?;
    let rungs: Vec<u32> = rungs
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|n| n as u32)
                .ok_or_else(|| msg(format!("{}: non-integer dmd rung", path.display())))
        })
        .collect::<Result<_>>()?;
    let video_shift = v
        .get("flow_shift")
        .or_else(|| v.get("video_scheduler_shift"))
        .and_then(|x| x.as_f64())
        .unwrap_or(10.0);
    let audio_shift = v
        .get("audio_flow_shift")
        .or_else(|| v.get("audio_scheduler_shift"))
        .and_then(|x| x.as_f64())
        .unwrap_or(3.0);
    let vsa = v
        .get("vsa_sparsity")
        .and_then(|x| x.as_f64())
        .unwrap_or(0.8);
    let dense = v
        .get("dense")
        .and_then(|x| x.as_bool())
        .unwrap_or(vsa <= 0.0);
    // Match the published Preview / V2 contracts when the JSON is the usual shape.
    match (rungs.as_slice(), dense) {
        ([999, 874, 749, 624, 500, 375, 250, 125], false) => {
            Ok(H3InferenceContract::fasth3_8step())
        }
        ([999, 749, 500, 250], false) => Ok(H3InferenceContract::fasth3_4step_vsa()),
        ([999, 749, 500, 250], true) => Ok(H3InferenceContract::fasth3_4step_dense()),
        _ => Ok(H3InferenceContract {
            num_inference_steps: rungs.len() + 1,
            transformer_forwards: rungs.len(),
            dmd_denoising_steps: rungs,
            video_scheduler_shift: video_shift,
            audio_scheduler_shift: audio_shift,
            guidance_scale: v
                .get("guidance_scale")
                .and_then(|x| x.as_f64())
                .unwrap_or(1.0),
            vsa_sparsity: vsa,
            vsa_tile_size: 64,
            dense,
            sigma_source: H3SigmaSource::Dmd,
        }),
    }
}

#[derive(Debug, Clone, Default)]
pub struct H3Timings {
    pub text_s: f64,
    pub refine_s: f64,
    pub denoise_s: f64,
    pub step_s: Vec<f64>,
    pub audio_decode_s: f64,
    pub video_decode_s: f64,
    /// The tail of encoding that outlives the decode, not the whole encode.
    pub write_s: f64,
}

#[derive(Debug, Clone)]
pub struct H3Output {
    pub geometry: H3Geometry,
    pub text_tokens: usize,
    pub text_cache: CacheStatus,
    /// `"streamed"`, `"resident-..."`, or `"cache"` when no encoder ran.
    pub text_encoder: &'static str,
    pub sequence_length: usize,
    pub frames: usize,
    pub frame_paths: Vec<String>,
    pub mp4: Option<String>,
    pub wav: PathBuf,
    pub timings: H3Timings,
}

/// The request's starting noise, from one generator in the reference's order:
/// video `[24, T, H, W]` first (then patchified), audio rows `[2 Na, 32]`
/// second, drawn directly in row layout. Torch's CPU sampler is not
/// reproduced, so a seed names a different (equally valid) sample than it does
/// upstream; parity runs inject the oracle's noise instead.
pub fn seeded_noise(
    cfg: &H3TransformerConfig,
    geometry: &H3Geometry,
    seed: u64,
) -> std::result::Result<(Vec<f32>, Vec<f32>), String> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let shape = [
        cfg.in_channels,
        geometry.latent_frames,
        geometry.latent_height,
        geometry.latent_width,
    ];
    let video: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    let audio: Vec<f32> = (0..geometry.audio_rows() * cfg.audio_in_channels)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    Ok((patchify(&video, shape, cfg.patch_size)?, audio))
}

/// One Euler step of `MiniMaxH3Scheduler.step` on device rows, in the
/// reference's order of operations: `x0 = x + sigma_t v`, then
/// `r x + (1 - r) x0`.
pub fn scheduler_step(
    schedule: &H3Schedule,
    step: usize,
    sample: &CudaTensor,
    velocity: &CudaTensor,
) -> Result<CudaTensor> {
    let c = schedule.step_coeffs(step).map_err(msg)?;
    let denoised = CudaTensor::lincomb(&[(1.0, sample), (c.sigma_from_timestep, velocity)])?;
    Ok(CudaTensor::lincomb(&[
        (c.ratio, sample),
        (1.0 - c.ratio, &denoised),
    ])?)
}

/// The 8-forward ladder. `observe(step, video_rows, audio_rows)` sees the
/// state after each step; the last call holds the clean latent rows.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    model: &H3Transformer,
    layout: &DeviceLayout,
    text_refined: &CudaTensor,
    video_rows: CudaTensor,
    audio_rows: CudaTensor,
    schedule: &H3JointSchedule,
    mode: AttnMode<'_>,
    cond_rows: Option<&CudaTensor>,
    cond_audio_rows: Option<&CudaTensor>,
    observe: &mut dyn FnMut(usize, &CudaTensor, &CudaTensor) -> Result<()>,
) -> Result<(CudaTensor, CudaTensor)> {
    let (mut video, mut audio) = (video_rows, audio_rows);
    for step in 0..schedule.num_steps() {
        let (v_video, v_audio) = model.forward(
            step,
            &video,
            &audio,
            text_refined,
            layout,
            mode,
            None,
            cond_rows,
            cond_audio_rows,
        )?;
        video = scheduler_step(&schedule.video, step, &video, &v_video)?;
        audio = scheduler_step(&schedule.audio, step, &audio, &v_audio)?;
        observe(step, &video, &audio)?;
    }
    Ok((video, audio))
}

/// `unpatchify_video_tokens` on the device: rows `[T h w, C pt ph pw]`
/// (channel-major features) to `[1, C, T, H, W]`. The reference's 8-D permute
/// is past the device permute's rank limit; with the singleton temporal patch
/// dropped it is rank 6.
pub fn unpatchify_rows(
    rows: &CudaTensor,
    channels: usize,
    grid: (usize, usize, usize),
    patch: [usize; 3],
) -> std::result::Result<CudaTensor, TensorError> {
    let [pt, ph, pw] = patch;
    let (t, h, w) = grid;
    if pt != 1 {
        return Err(TensorError::Message(format!(
            "unpatchify: temporal patch {pt} is not supported (H3 uses 1)"
        )));
    }
    if rows.shape != [t * h * w, channels * ph * pw] {
        return Err(TensorError::Message(format!(
            "unpatchify: rows {:?} for grid {grid:?}, {channels} channels, patch {patch:?}",
            rows.shape
        )));
    }
    rows.reshape(vec![t, h, w, channels, ph, pw])?
        .permute(&[3, 0, 1, 4, 2, 5])?
        .reshape(vec![1, channels, t, h * ph, w * pw])
}

/// Seconds spent in [`H3Pipeline::load`], by component.
#[derive(Debug, Clone, Default)]
pub struct H3LoadTimings {
    /// Zero when the encoder is streamed.
    pub text_encoder_s: f64,
    pub refiner_s: f64,
    pub dit_s: f64,
    pub video_vae_s: f64,
    pub audio_vae_s: f64,
}

enum VideoDecoder {
    Official(H3VideoDecoder),
    Taeh3(TaeHv),
}

/// VSA gate tensors load only when the contract is sparse (Spark) or an MLX
/// snapshot actually ships them. Dense Sol-H3 / MiniMax-H3 stay ungated.
fn dit_loads_vsa_gate(contract: &H3InferenceContract, mlx_vsa_capable: Option<bool>) -> bool {
    contract.vsa_sparsity > 0.0 && mlx_vsa_capable.unwrap_or(true)
}

fn resolve_taeh3(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    let env = std::env::var("FASTVIDEO_TAEH3_WEIGHTS").unwrap_or_default();
    if env.is_empty() {
        None
    } else {
        Some(PathBuf::from(env))
    }
}

/// Everything but the text encoder, resident: load once, generate many times.
pub struct H3Pipeline {
    root: PathBuf,
    options: H3PipelineOptions,
    cfg: H3TransformerConfig,
    contract: H3InferenceContract,
    schedule: H3JointSchedule,
    refiner: H3TextRefiner,
    model: H3Transformer,
    video_vae: VideoDecoder,
    audio_vae: H3AudioDecoder,
    text_encoder: Option<Box<dyn HiddenStateEncoder>>,
    pub load_timings: H3LoadTimings,
}

impl H3Pipeline {
    /// `root` is the FastH3 snapshot (`transformer/`, `vae/`, `audio_vae/`, and
    /// `tokenizer/` + `text_encoder/` unless `options.text_root` says otherwise).
    pub fn load(root: &Path, options: H3PipelineOptions) -> Result<Self> {
        let mut options = options;
        if options.recipe.as_deref().is_some_and(sol_h3_forces_ref2va) {
            options.ref2va = true;
        }
        if options.recipe.as_deref().is_some_and(is_sol_h3_recipe)
            && options.reference_image_resize == ReferenceImageResize::Auto
        {
            options.reference_image_resize = ReferenceImageResize::Match;
        }
        let contract = resolve_contract(root, options.recipe.as_deref())?;
        let cfg = H3TransformerConfig::fasth3_8step();
        let schedule = H3JointSchedule::from_contract(&contract).map_err(msg)?;
        if contract.dense {
            options.dense = true;
        }
        let mut load_timings = H3LoadTimings::default();
        let timed = |slot: &mut f64, timer: Instant| *slot = timer.elapsed().as_secs_f64();

        // The resident encoder goes first: its FP8 quantization runs on the
        // device with f32 transients (524 MB for the widest matrix), which
        // should happen while the card is otherwise empty.
        let free = crate::wan::device::free_memory().map(|(free, _)| free);
        options.text_encoder = options.text_encoder.resolve(free);
        let timer = Instant::now();
        let text_encoder: Option<Box<dyn HiddenStateEncoder>> = match options.text_encoder {
            TextEncoderChoice::Recovered8b => {
                let text_root = options.text_root.as_deref().unwrap_or(root);
                Some(Box::new(super::recovered_8b::Recovered8bEncoder::load(
                    text_root,
                )?))
            }
            other => match other.precision() {
                Some(precision) => {
                    let text_root = options.text_root.as_deref().unwrap_or(root);
                    Some(Box::new(super::text::load_resident_encoder(
                        text_root, precision,
                    )?))
                }
                None => None,
            },
        };
        if text_encoder.is_some() {
            timed(&mut load_timings.text_encoder_s, timer);
        }

        let (map, mlx) = match super::mlx::find(root) {
            Some(dir) if !options.ref2va => {
                let (map, spec) = super::mlx::open_map(&dir)?;
                crate::wan::log::info(format_args!(
                    "h3 dit=mlx affine int{} g{} ({})",
                    spec.bits,
                    spec.group_size,
                    spec.weights.display()
                ));
                (map, Some(spec))
            }
            _ => {
                let dit = if options.ref2va {
                    let p = root.join("transformer_ref");
                    if !p.is_dir() {
                        return Err(msg(format!(
                            "Ref2VA needs {} (MiniMax-H3 Base Ref2VA partition)",
                            p.display()
                        )));
                    }
                    crate::wan::log::info(format_args!("h3 dit=transformer_ref"));
                    p
                } else {
                    root.join("transformer")
                };
                (WeightMap::open(&dit)?, None)
            }
        };
        // Gate lives on VSA checkpoints and VSA-DataFree replacements. Sol-H3
        // + MiniMax-H3 is Sol-Attn on a dense backbone (`vsa_sparsity == 0`).
        let with_gate = dit_loads_vsa_gate(&contract, mlx.as_ref().map(|s| s.vsa_capable));
        crate::wan::log::info(format_args!(
            "h3 recipe={} steps={} video_shift={} vsa={} dense={}",
            options.recipe.as_deref().unwrap_or("auto"),
            contract.transformer_forwards,
            contract.video_scheduler_shift,
            contract.vsa_sparsity,
            options.dense
        ));
        match fastvideo_models::h3::sol::recipe_sol_attn_policy(
            options.recipe.as_deref(),
            std::env::var("FASTVIDEO_H3_SOL_ATTN").ok().as_deref(),
        ) {
            fastvideo_models::h3::sol::H3SolAttnPolicy::Spark => {
                crate::wan::log::info(format_args!(
                    "h3 sol-attn: stage-1 update 0 dense, later updates layer 0 dense and layers 1-49 sol-attn kernel tau 1/1.25/1.5 (thresh_type=diag)"
                ));
            }
            fastvideo_models::h3::sol::H3SolAttnPolicy::Rtx => {
                crate::wan::log::info(format_args!(
                    "h3 sol-attn: rtx first 10 steps dense, later steps layers 0-1 dense and layers 2-49 sol-attn kernel tau 1.0 (thresh_type=diag)"
                ));
            }
            fastvideo_models::h3::sol::H3SolAttnPolicy::Off => {}
        }
        let teacache = fastvideo_models::h3::sol::teacache_requested(
            std::env::var("FASTVIDEO_H3_SOL_CACHE").ok().as_deref(),
        );
        let mut lora = if options.recipe.as_deref().is_some_and(is_sol_h3_recipe) {
            let spark = options
                .recipe
                .as_deref()
                .is_some_and(fastvideo_models::h3::lora::is_sol_h3_spark_recipe);
            let spec = if options.ref2va {
                SolH3AdapterSpec::ref2va()
            } else if spark {
                fastvideo_models::h3::spark::spark_adapter_spec()
            } else {
                SolH3AdapterSpec::t2v_i2v()
            };
            let path = spec
                .resolve(root, options.adapter.as_deref())
                .map_err(msg)?;
            let fuse = super::lora::H3LoraFuse::open(&map, &path, spec.alpha, spec.scale)?;
            crate::wan::log::info(format_args!(
                "h3 sol-h3: fuse {} pairs rank={} alpha={} scale={} diffs={} ({})",
                fuse.pairs_total,
                fuse.rank,
                fuse.alpha,
                fuse.effective_scale(),
                fuse.diffs_total,
                path.display()
            ));
            if spark {
                crate::wan::log::info(format_args!(
                    "h3 sol-h3-spark: stage-1 VSA {} tile {} BF16 FastH3_VSA_DataFree strength {}; W8A8 FP8 stays off (measured 16–20 dB). Draft {}x{} {}f. H3×2 upscaler and H3-to-LTX adapter run when their checkpoints are set. Joint 3-step LTX refine uses the fixed prompt and cached Gemma",
                    contract.vsa_sparsity,
                    contract.vsa_tile_size,
                    spec.scale,
                    fastvideo_models::h3::sol::SPARK_DRAFT_WIDTH,
                    fastvideo_models::h3::sol::SPARK_DRAFT_HEIGHT,
                    fastvideo_models::h3::sol::SPARK_DRAFT_FRAMES,
                ));
            }
            Some(fuse)
        } else {
            None
        };
        let timer = Instant::now();
        let refiner = H3TextRefiner::load_with(&cfg, &map, &mut lora)?;
        timed(&mut load_timings.refiner_s, timer);
        let timer = Instant::now();
        let model = H3Transformer::load_with(
            cfg.clone(),
            &map,
            &schedule,
            with_gate,
            options.adaln_cache.as_deref(),
            &mut lora,
        )?;
        if teacache {
            model.enable_sol_teacache(schedule.num_steps())?;
        }
        if let Some(fuse) = lora.as_ref() {
            fuse.finish()?;
        }
        timed(&mut load_timings.dit_s, timer);
        let timer = Instant::now();
        let video_vae = match resolve_taeh3(options.taeh3.as_deref()) {
            Some(path) => {
                let tae = TaeHv::load_from_path(&path, TaeArch::H3)
                    .map_err(|e| msg(format!("FASTVIDEO_TAEH3_WEIGHTS={}: {e}", path.display())))?;
                crate::wan::log::info(format_args!("h3 vae=taeh3 ({})", path.display()));
                VideoDecoder::Taeh3(tae)
            }
            None => VideoDecoder::Official(H3VideoDecoder::load(
                H3VideoVaeConfig::fasth3_8step(),
                &WeightMap::open(&root.join("vae"))?,
            )?),
        };
        timed(&mut load_timings.video_vae_s, timer);
        let timer = Instant::now();
        let audio_vae = H3AudioDecoder::load(
            H3AudioVaeConfig::fasth3_8step(),
            &WeightMap::open(&root.join("audio_vae"))?,
        )?;
        timed(&mut load_timings.audio_vae_s, timer);
        Ok(Self {
            root: root.to_path_buf(),
            options,
            cfg,
            contract,
            schedule,
            refiner,
            model,
            video_vae,
            audio_vae,
            text_encoder,
            load_timings,
        })
    }

    /// Replace the text encoder (tests, or an encoder built elsewhere). It is
    /// consulted only on a conditioning-cache miss.
    pub fn with_text_encoder(mut self, encoder: Box<dyn HiddenStateEncoder>) -> Self {
        if !matches!(
            self.options.text_encoder,
            TextEncoderChoice::ResidentBf16
                | TextEncoderChoice::ResidentFp8
                | TextEncoderChoice::Recovered8b
        ) {
            self.options.text_encoder = TextEncoderChoice::ResidentFp8;
        }
        self.text_encoder = Some(encoder);
        self
    }

    /// `(kind, device bytes)` of the encoder a cache miss will use.
    pub fn text_encoder(&self) -> (&'static str, u64) {
        self.text_encoder
            .as_ref()
            .map_or(("streamed", 0), |e| (e.kind(), e.resident_bytes()))
    }

    /// Encode `prompt` with the resident encoder AND by streaming, bypassing
    /// the cache, and return `(rel_l2, cosine)` of resident against streamed.
    /// What matters about a quantized encoder is how far it moves the
    /// conditioning; this is that number for the prompt at hand. `None` when
    /// the encoder is streamed anyway. Costs one streamed encode (~10 s).
    pub fn text_encoder_drift(&self, prompt: &str) -> Result<Option<(f64, f64)>> {
        let Some(resident) = &self.text_encoder else {
            return Ok(None);
        };
        let text_root = self.options.text_root.as_deref().unwrap_or(&self.root);
        let ours =
            super::text::encode_prompt_with(text_root, prompt, None, Some(resident.as_ref()))?;
        let streamed = super::text::encode_prompt_with(text_root, prompt, None, None)?;
        let (a, b) = (ours.hidden.host_cow()?, streamed.hidden.host_cow()?);
        let (mut err, mut aa, mut bb, mut ab) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            let (x, y) = (f64::from(*x), f64::from(*y));
            err += (x - y) * (x - y);
            aa += x * x;
            bb += y * y;
            ab += x * y;
        }
        Ok(Some((
            (err / bb.max(1e-300)).sqrt(),
            ab / (aa * bb).sqrt().max(1e-300),
        )))
    }

    pub fn options(&self) -> &H3PipelineOptions {
        &self.options
    }

    /// Generate one clip into `out_dir`: `frame-NNN.png`, `audio.wav`, and
    /// `output.mp4` with the audio muxed in.
    fn spark_bridge(&self, latents: &CudaTensor) -> Result<Option<CudaTensor>> {
        if !self
            .options
            .recipe
            .as_deref()
            .is_some_and(fastvideo_models::h3::lora::is_sol_h3_spark_recipe)
        {
            return Ok(None);
        }
        match super::spark::SparkBridge::resolve(&self.root).map_err(|e| msg(e.to_string()))? {
            None => {
                crate::wan::log::info(format_args!(
                    "h3 sol-h3-spark: no {} or H3-to-LTX adapter beside {}; H3 decode continues",
                    fastvideo_models::h3::spark::UPSCALER_FILE,
                    self.root.display()
                ));
                Ok(None)
            }
            Some(bridge) => {
                let refined = bridge.forward(latents).map_err(|e| msg(e.to_string()))?;
                if std::env::var_os("FASTVIDEO_LTX2_WEIGHTS").is_none() {
                    crate::wan::log::info(format_args!(
                        "h3 sol-h3-spark: refiner video latent {:?}. Set FASTVIDEO_LTX2_WEIGHTS to run the 3-step LTX refiner. H3 decode continues.",
                        refined.shape
                    ));
                    return Ok(None);
                }
                crate::wan::log::info(format_args!(
                    "h3 sol-h3-spark: refiner video latent {:?}. 3-step LTX refiner follows H3 decode.",
                    refined.shape
                ));
                Ok(Some(refined))
            }
        }
    }

    pub fn generate(&self, request: &H3Request, out_dir: &Path) -> Result<H3Output> {
        let cfg = &self.cfg;
        let geometry =
            H3Geometry::new(request.height, request.width, request.num_frames).map_err(msg)?;
        let mut timings = H3Timings::default();

        // --- text: cache, else resident, else ~50 GB streamed one layer at a time ---
        let timer = Instant::now();
        let resident = match (&self.options.text_encoder, &self.text_encoder) {
            (_, Some(encoder))
                if matches!(
                    self.options.text_encoder,
                    TextEncoderChoice::ResidentBf16
                        | TextEncoderChoice::ResidentFp8
                        | TextEncoderChoice::Recovered8b
                ) =>
            {
                Some(encoder.as_ref())
            }
            (TextEncoderChoice::Streamed | TextEncoderChoice::Auto, _) => None,
            (_, None)
                if matches!(
                    self.options.text_encoder,
                    TextEncoderChoice::ResidentBf16
                        | TextEncoderChoice::ResidentFp8
                        | TextEncoderChoice::Recovered8b
                ) =>
            {
                return Err(msg(format!(
                    "text encoder {:?} was requested but is not loaded",
                    self.options.text_encoder
                )));
            }
            _ => None,
        };
        // Tokenizer always comes from the DiT snapshot (H3 markers). Recovered
        // 8B weights live under text_root and have no tokenizer of their own.
        let (tokenizer_root, encoder_root) =
            if matches!(self.options.text_encoder, TextEncoderChoice::Recovered8b) {
                (
                    self.root.as_path(),
                    self.options.text_root.as_deref().unwrap_or(&self.root),
                )
            } else {
                let r = self.options.text_root.as_deref().unwrap_or(&self.root);
                (r, r)
            };
        let needs_vl = !request.keyframe_anchors().is_empty() || request.is_ref2va();
        let text = if needs_vl {
            if matches!(self.options.text_encoder, TextEncoderChoice::Recovered8b) {
                return Err(msg(
                    "FL2VA/Ref2VA multimodal text needs Qwen3-VL-32B with vision tower; recovered-8b is text-only",
                ));
            }
            encode_request_multimodal(encoder_root, request)?
        } else if matches!(self.options.text_encoder, TextEncoderChoice::Recovered8b) {
            super::text::encode_prompt_recovered(
                tokenizer_root,
                encoder_root,
                &request.prompt,
                self.options.text_cache.as_deref(),
                resident,
            )?
        } else {
            super::text::encode_prompt_with(
                tokenizer_root,
                &request.prompt,
                self.options.text_cache.as_deref(),
                resident,
            )?
        };
        timings.text_s = timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let text_refined = self.refiner.forward(&text.hidden)?;
        timings.refine_s = timer.elapsed().as_secs_f64();

        // --- DiT --------------------------------------------------------------------
        if request.is_ref2va() && !self.options.ref2va {
            return Err(msg(
                "Ref2VA request needs H3PipelineOptions.ref2va (load transformer_ref/)",
            ));
        }
        if !request.is_ref2va() && self.options.ref2va {
            return Err(msg(
                "transformer_ref/ was loaded but the request has no references",
            ));
        }
        if request.is_ref2va() && (request.first_image.is_some() || request.last_image.is_some()) {
            return Err(msg(
                "Ref2VA and FL2VA keyframes cannot be combined in one request",
            ));
        }
        if request.is_ref2va() {
            validate_references(&request.references).map_err(msg)?;
        }

        let anchors = request.keyframe_anchors();
        let (mut layout, cond_rows, cond_audio_rows) = if request.is_ref2va() {
            let encoded = encode_ref2va_conditions(
                &self.root,
                cfg,
                &geometry,
                &request.references,
                request.seed,
                self.options.reference_image_resize,
            )?;
            let layout = H3PackedLayout::with_references(
                text.ids.len(),
                (
                    geometry.latent_frames,
                    geometry.latent_height,
                    geometry.latent_width,
                ),
                geometry.audio_latents,
                cfg.patch_size,
                &encoded.prepared,
            )
            .map_err(msg)?;
            (layout, Some(encoded.video_rows), encoded.audio_rows)
        } else if anchors.is_empty() {
            (
                H3PackedLayout::from_geometry(&geometry, text.ids.len()).map_err(msg)?,
                None,
                None,
            )
        } else {
            let layout =
                H3PackedLayout::from_geometry_with_keyframes(&geometry, text.ids.len(), &anchors)
                    .map_err(msg)?;
            let rows = encode_fl2va_cond_rows(&self.root, cfg, &geometry, request, &anchors)?;
            (layout, Some(rows), None)
        };
        if !text.token_tags.is_empty() {
            layout.set_text_token_tags(&text.token_tags).map_err(msg)?;
        }
        let sequence_length = layout.sequence_length();
        // Interleaved Ref2VA condition audio breaks VSA's contiguous-prefix tiles.
        let force_dense = layout.num_condition_audio_rows > 0;
        let sol_policy = match fastvideo_models::h3::sol::recipe_sol_attn_policy(
            self.options.recipe.as_deref(),
            std::env::var("FASTVIDEO_H3_SOL_ATTN").ok().as_deref(),
        ) {
            fastvideo_models::h3::sol::H3SolAttnPolicy::Off => None,
            kind => Some(H3SolPolicy::from_layout(kind, &layout)),
        };
        match fastvideo_models::h3::sol::sink_layout(
            std::env::var("FASTVIDEO_H3_SOL_SINK").ok().as_deref(),
        ) {
            fastvideo_models::h3::sol::H3SolSinkLayout::Native => {
                crate::wan::log::info(format_args!(
                    "h3 sol sink: native multi-span (FASTVIDEO_H3_SOL_SINK=native)"
                ));
            }
            fastvideo_models::h3::sol::H3SolSinkLayout::Suffix => {
                if sol_policy.is_some() {
                    crate::wan::log::info(format_args!(
                        "h3 sol sink: permute [visual | sinks], single suffix sink"
                    ));
                }
            }
        }
        let vsa = if sol_policy.is_some() || self.options.dense || force_dense {
            if force_dense && !self.options.dense && sol_policy.is_none() {
                crate::wan::log::info(format_args!(
                    "h3 ref2va: dense attention (interleaved condition audio)"
                ));
            }
            None
        } else {
            let vsa_cfg = super::vsa::H3VsaConfig {
                sparsity: self.contract.vsa_sparsity,
                group: crate::wan::envflag::usize_flag("FASTVIDEO_VSA_GROUP", 8).max(1),
            };
            Some(H3Vsa::new(
                &layout,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                vsa_cfg,
            )?)
        };
        let mode = if let Some(ref policy) = sol_policy {
            AttnMode::Sol(policy)
        } else {
            vsa.as_ref().map_or(AttnMode::Dense, AttnMode::Vsa)
        };
        let layout = DeviceLayout::new(cfg, layout)?;
        let (video_noise, audio_noise) = seeded_noise(cfg, &geometry, request.seed).map_err(msg)?;
        let video_rows = CudaTensor::from_vec(
            video_noise,
            vec![geometry.video_rows(), cfg.video_patch_dim()],
        )?
        .to_device()?;
        let audio_rows = CudaTensor::from_vec(
            audio_noise,
            vec![geometry.audio_rows(), cfg.audio_in_channels],
        )?
        .to_device()?;

        let timer = Instant::now();
        let mut last = Instant::now();
        let steps = self.schedule.num_steps();
        let mut step_s = Vec::with_capacity(steps);
        let (video_rows, audio_rows) = denoise(
            &self.model,
            &layout,
            &text_refined,
            video_rows,
            audio_rows,
            &self.schedule,
            mode,
            cond_rows.as_ref(),
            cond_audio_rows.as_ref(),
            &mut |step, _, _| {
                crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
                step_s.push(last.elapsed().as_secs_f64());
                crate::wan::log::info(format_args!(
                    "h3 step {}/{steps}: {:.1}s",
                    step + 1,
                    step_s[step]
                ));
                last = Instant::now();
                Ok(())
            },
        )?;
        timings.denoise_s = timer.elapsed().as_secs_f64();
        timings.step_s = step_s;
        drop((
            vsa,
            sol_policy,
            layout,
            text_refined,
            cond_rows,
            cond_audio_rows,
        ));

        // --- audio first: the WAV must exist before the muxer starts -------------------
        let timer = Instant::now();
        let sample_rate = self.audio_vae.config().sampling_rate as u32;
        let wave = self
            .audio_vae
            .decode_rows(&audio_rows, H3_AUDIO_CHANNELS)?
            .host_cow()?
            .into_owned();
        let wav = out_dir.join("audio.wav");
        write_wav(
            &wav,
            &interleave_audio(&wave, H3_AUDIO_CHANNELS)?,
            H3_AUDIO_CHANNELS as u16,
            sample_rate,
        )?;
        timings.audio_decode_s = timer.elapsed().as_secs_f64();

        // --- video: chunks go to the writer as they decode -------------------------------
        let latents = unpatchify_rows(
            &video_rows,
            cfg.in_channels,
            geometry.token_grid,
            cfg.patch_size,
        )?;
        let refined_video = self.spark_bridge(&latents)?;
        let timer = Instant::now();
        let mut writer =
            VideoWriter::spawn_with_audio(out_dir, H3_FPS as u32, request.mp4, Some(&wav))?;
        // The sink speaks TensorError; carry the writer's own error out beside it.
        let mut writer_error: Option<PipelineError> = None;
        let mut sink = |offset: usize, frames: &CudaTensor| {
            let (h, w) = (frames.shape[2], frames.shape[3]);
            let pushed = frames_to_rgb8(frames).and_then(|rgb| writer.push(offset, h, w, rgb));
            pushed.map_err(|e| {
                let text = e.to_string();
                writer_error = Some(e);
                TensorError::Message(text)
            })
        };
        let decoded = match &self.video_vae {
            VideoDecoder::Official(vae) => vae.decode_streaming(&latents, &mut sink),
            VideoDecoder::Taeh3(tae) => tae
                .decode_streaming(&latents, &mut sink)
                .map(|v| v.shape[2]),
        };
        let frames = match (decoded, writer_error) {
            (_, Some(e)) => return Err(e),
            (r, None) => r?,
        };
        if frames != geometry.num_frames {
            return Err(msg(format!(
                "h3 video decode emitted {frames} frames, geometry wants {}",
                geometry.num_frames
            )));
        }
        timings.video_decode_s = timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let (frame_paths, mp4) = writer.finish()?;
        timings.write_s = timer.elapsed().as_secs_f64();
        if let Some(video) = refined_video {
            let refined = refine_spark(request, &video, &wave, sample_rate, out_dir)?;
            crate::wan::log::info(format_args!(
                "h3 sol-h3-spark: refined {} ({} frames)",
                refined.mp4.as_deref().unwrap_or(refined.wav.as_str()),
                refined.frames.len()
            ));
        }

        Ok(H3Output {
            geometry,
            text_tokens: text.ids.len(),
            text_cache: text.cache,
            text_encoder: text.encoder,
            sequence_length,
            frames,
            frame_paths,
            mp4,
            wav,
            timings,
        })
    }
}

/// Load LTX-2.5 and run the 3-step joint refiner into `out_dir/refined`.
/// The H3 DiT stays resident; this is a second model in the same process.
fn refine_spark(
    request: &H3Request,
    video: &CudaTensor,
    wave: &[f32],
    sample_rate: u32,
    out_dir: &Path,
) -> Result<crate::ltx2::pipeline::Ltx2Output> {
    use crate::ltx2::pipeline::{Ltx2Paths, Ltx2Pipeline, PipelineOptions};

    let weights = match std::env::var("FASTVIDEO_LTX2_WEIGHTS") {
        Ok(raw) => PathBuf::from(raw),
        Err(_) => {
            return Err(msg(
                "h3 sol-h3-spark: FASTVIDEO_LTX2_WEIGHTS was unset before the refiner",
            ))
        }
    };
    if !weights.is_dir() {
        return Err(msg(format!(
            "h3 sol-h3-spark: FASTVIDEO_LTX2_WEIGHTS {} is not a directory",
            weights.display()
        )));
    }
    let dit = std::env::var("FASTVIDEO_LTX2_DIT")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| weights.join("transformer"));
    let cfg = fastvideo_models::ltx2::ltx2_5_22b_distilled();
    let prompt = fastvideo_models::h3::spark::FIXED_PROMPT;
    crate::wan::log::info(format_args!(
        "h3 sol-h3-spark: loading LTX-2.5 from {} (DiT {}) beside the resident H3 model; fixed prompt {prompt:?}; Gemma cache on",
        weights.display(),
        dit.display()
    ));
    let mut pipeline = Ltx2Pipeline::load(
        &Ltx2Paths {
            weights,
            dit,
            text: None,
        },
        &cfg,
        &PipelineOptions {
            text_cache: crate::ltx2::text_cache::default_dir(),
            ..PipelineOptions::default()
        },
    )?;
    pipeline.refine_joint(
        video,
        wave,
        H3_AUDIO_CHANNELS,
        sample_rate,
        prompt,
        request.seed,
        f64::from(H3_FPS as u32),
        &out_dir.join("refined"),
        request.mp4,
    )
}

/// Qwen3-VL multimodal text for FL2VA keyframes / Ref2VA ordered refs.
fn encode_request_multimodal(
    text_root: &Path,
    request: &H3Request,
) -> Result<super::text::TextConditioning> {
    use super::text::{VisionImage, VisionVideo};
    use fastvideo_models::h3::presentation::PresentationRef;

    if request.is_ref2va() {
        let mut images_owned: Vec<(Vec<u8>, usize, usize)> = Vec::new();
        let mut videos_owned: Vec<(Vec<u8>, usize, usize, usize)> = Vec::new();
        let mut refs = Vec::new();
        for spec in &request.references {
            match spec.kind {
                ReferenceKind::Audio => refs.push(PresentationRef::Audio),
                ReferenceKind::Image => {
                    let img = image::open(&spec.path)
                        .map_err(|e| msg(format!("open {}: {e}", spec.path.display())))?
                        .into_rgb8();
                    let (w, h) = (img.width() as usize, img.height() as usize);
                    images_owned.push((img.into_raw(), h, w));
                    refs.push(PresentationRef::Image { token_count: 0 });
                }
                ReferenceKind::Video => {
                    // Cap decode length; Qwen samples at 2 fps from 24 fps.
                    let (frames, w, h, _fps) = super::media::decode_video_rgb(&spec.path, 24 * 16)?;
                    let n = frames.len() / (w * h * 3);
                    videos_owned.push((frames, n, h, w));
                    refs.push(PresentationRef::Video {
                        token_count: 0,
                        block_timestamps: Vec::new(),
                    });
                }
            }
        }
        let images: Vec<VisionImage<'_>> = images_owned
            .iter()
            .map(|(rgb, h, w)| VisionImage {
                rgb,
                height: *h,
                width: *w,
            })
            .collect();
        let videos: Vec<VisionVideo<'_>> = videos_owned
            .iter()
            .map(|(frames, n, h, w)| VisionVideo {
                frames,
                num_frames: *n,
                height: *h,
                width: *w,
            })
            .collect();
        crate::wan::log::info(format_args!(
            "h3 ref2va: Qwen-VL multimodal ({} images, {} videos)",
            images.len(),
            videos.len()
        ));
        return Ok(super::text::encode_ref2va_multimodal(
            text_root,
            &request.prompt,
            &refs,
            &images,
            &videos,
        )?);
    }

    let mut owned = Vec::new();
    for path in [&request.first_image, &request.last_image]
        .into_iter()
        .flatten()
    {
        let img = image::open(path)
            .map_err(|e| msg(format!("open {}: {e}", path.display())))?
            .into_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        owned.push((img.into_raw(), h, w));
    }
    let images: Vec<VisionImage<'_>> = owned
        .iter()
        .map(|(rgb, h, w)| VisionImage {
            rgb,
            height: *h,
            width: *w,
        })
        .collect();
    crate::wan::log::info(format_args!(
        "h3 fl2va: Qwen-VL multimodal ({} images)",
        images.len()
    ));
    Ok(super::text::encode_fl2va_multimodal(
        text_root,
        &request.prompt,
        &images,
    )?)
}

/// GPU-encode FL2VA keyframe images → patchified cond rows (`[Nc, patch_dim]`).
fn encode_fl2va_cond_rows(
    root: &Path,
    cfg: &H3TransformerConfig,
    geometry: &H3Geometry,
    request: &H3Request,
    anchors: &[KeyframeAnchor],
) -> Result<CudaTensor> {
    let encoder = H3VideoEncoder::load(
        H3VideoVaeConfig::fasth3_8step(),
        &WeightMap::open(&root.join("vae"))?,
    )?;
    let mut parts = Vec::with_capacity(anchors.len());
    for (i, anchor) in anchors.iter().enumerate() {
        let path =
            match anchor {
                KeyframeAnchor::First => request.first_image.as_deref().ok_or_else(|| {
                    msg("FL2VA first keyframe requested but first_image is missing")
                })?,
                KeyframeAnchor::Last => request.last_image.as_deref().ok_or_else(|| {
                    msg("FL2VA last keyframe requested but last_image is missing")
                })?,
            };
        crate::wan::log::info(format_args!(
            "h3 fl2va: encode {} ({})",
            anchor.as_str(),
            path.display()
        ));
        // Noise-aug seed distinct from video/audio noise.
        let seed = request.seed.wrapping_add(17 + i as u64);
        let z =
            encoder.encode_keyframe_file(path, request.height, request.width, false, true, seed)?;
        // z: [1, C, 1, h, w] → patchify one latent frame into DiT rows.
        let host = z.host_cow()?.into_owned();
        let shape = [
            cfg.in_channels,
            1,
            geometry.latent_height,
            geometry.latent_width,
        ];
        if host.len() != shape.iter().product::<usize>() {
            return Err(msg(format!(
                "h3 fl2va: encoded latent len {} != {:?}",
                host.len(),
                shape
            )));
        }
        let rows = patchify(&host, shape, cfg.patch_size).map_err(msg)?;
        let n_rows =
            geometry.latent_height / cfg.patch_size[1] * geometry.latent_width / cfg.patch_size[2];
        parts.push(CudaTensor::from_vec(rows, vec![n_rows, cfg.video_patch_dim()])?.to_device()?);
    }
    if parts.len() == 1 {
        Ok(parts.pop().unwrap())
    } else {
        let refs: Vec<&CudaTensor> = parts.iter().collect();
        Ok(CudaTensor::cat(&refs, 0)?)
    }
}

struct Ref2VaEncoded {
    prepared: Vec<PreparedReference>,
    video_rows: CudaTensor,
    audio_rows: Option<CudaTensor>,
}

/// Encode ordered Ref2VA references (images at 2048 short-edge; videos at the
/// clip's own 768-short-edge canvas, trimmed to VAE chunk geometry).
fn encode_ref2va_conditions(
    root: &Path,
    cfg: &H3TransformerConfig,
    geometry: &H3Geometry,
    references: &[H3ReferenceSpec],
    seed: u64,
    resize: ReferenceImageResize,
) -> Result<Ref2VaEncoded> {
    let vae_cfg = H3VideoVaeConfig::fasth3_8step();
    let audio_cfg = H3AudioVaeConfig::fasth3_8step();
    let ratio = vae_cfg.spatial_compression_ratio();
    let sample_rate = audio_cfg.sampling_rate as u32;
    let encoder = H3VideoEncoder::load(vae_cfg, &WeightMap::open(&root.join("vae"))?)?;
    let audio_map = WeightMap::open(&root.join("audio_vae"))?;
    let audio_encoder = super::audio_vae::H3AudioEncoder::load(audio_cfg, &audio_map)?;
    let mut prepared = Vec::with_capacity(references.len());
    let mut video_parts = Vec::new();
    let mut audio_parts = Vec::new();
    let max_audio_samples = geometry.num_frames * sample_rate as usize / H3_FPS;

    for (i, spec) in references.iter().enumerate() {
        match spec.kind {
            ReferenceKind::Audio => {
                crate::wan::log::info(format_args!(
                    "h3 ref2va: encode audio {} ({})",
                    i + 1,
                    spec.path.display()
                ));
                let planar = super::media::decode_audio_stereo_f32(
                    &spec.path,
                    sample_rate,
                    max_audio_samples,
                )?;
                let rows = audio_encoder.encode_stereo_planar(&planar)?;
                let na = rows.shape[0] / H3_AUDIO_CHANNELS;
                audio_parts.push(rows);
                prepared.push(PreparedReference::Audio {
                    num_audio_latents: na,
                });
            }
            ReferenceKind::Image => {
                let img = image::open(&spec.path)
                    .map_err(|e| msg(format!("open {}: {e}", spec.path.display())))?
                    .into_rgb8();
                let (sw, sh) = (img.width() as usize, img.height() as usize);
                let (out_h, out_w) =
                    resolve_reference_image_size_with(sw, sh, resize).map_err(msg)?;
                let prep = PreparedImageRef::from_pixel_size(out_h, out_w, ratio).map_err(msg)?;
                crate::wan::log::info(format_args!(
                    "h3 ref2va: encode image {} {}x{} → {}x{} ({})",
                    i + 1,
                    sw,
                    sh,
                    out_w,
                    out_h,
                    spec.path.display()
                ));
                let z = encoder.encode_keyframe_file(
                    &spec.path,
                    out_h,
                    out_w,
                    true,
                    true,
                    seed.wrapping_add(31 + i as u64),
                )?;
                let host = z.host_cow()?.into_owned();
                let shape = [cfg.in_channels, 1, prep.latent_height, prep.latent_width];
                if host.len() != shape.iter().product::<usize>() {
                    return Err(msg(format!(
                        "h3 ref2va: encoded latent len {} != {:?}",
                        host.len(),
                        shape
                    )));
                }
                let rows = patchify(&host, shape, cfg.patch_size).map_err(msg)?;
                let n_rows = prep.rows_per_frame(cfg.patch_size).map_err(msg)?;
                video_parts.push(
                    CudaTensor::from_vec(rows, vec![n_rows, cfg.video_patch_dim()])?.to_device()?,
                );
                prepared.push(PreparedReference::Image(prep));
            }
            ReferenceKind::Video => {
                let max_src = ((geometry.num_frames as f64) * 2.0).ceil() as usize + 8;
                let max_src = max_src.max(geometry.num_frames + H3_FPS);
                crate::wan::log::info(format_args!(
                    "h3 ref2va: decode video {} ({})",
                    i + 1,
                    spec.path.display()
                ));
                let (raw, src_w, src_h, fps) = super::media::decode_video_rgb(&spec.path, max_src)?;
                let src_frames = raw.len() / (src_w * src_h * 3);
                let (resampled, n24) =
                    resample_reference_frames(&raw, src_frames, src_h, src_w, fps).map_err(msg)?;
                let keep = n24.min(geometry.num_frames);
                let (keep, out_h, out_w) =
                    plan_reference_video_canvas(src_h, src_w, n24, keep).map_err(msg)?;
                let trimmed_n = trim_reference_num_frames(keep).map_err(msg)?.min(keep);
                let trimmed = &resampled[..trimmed_n * src_h * src_w * 3];
                let frames = super::media::resize_rgb_frames(
                    trimmed, trimmed_n, src_h, src_w, out_h, out_w,
                )?;
                crate::wan::log::info(format_args!(
                    "h3 ref2va: encode video {} {} frames @ {:.3}fps → {}x{} / {}f ({})",
                    i + 1,
                    src_frames,
                    fps,
                    out_w,
                    out_h,
                    trimmed_n,
                    spec.path.display()
                ));
                let z = encoder.encode_rgb_frames(
                    &frames,
                    trimmed_n,
                    out_h,
                    out_w,
                    true,
                    seed.wrapping_add(31 + i as u64),
                )?;
                let host = z.host_cow()?.into_owned();
                let lt = z.shape[2];
                let lh = z.shape[3];
                let lw = z.shape[4];
                let shape = [cfg.in_channels, lt, lh, lw];
                if host.len() != shape.iter().product::<usize>() {
                    return Err(msg(format!(
                        "h3 ref2va: video latent len {} != {:?}",
                        host.len(),
                        shape
                    )));
                }
                let rows = patchify(&host, shape, cfg.patch_size).map_err(msg)?;
                let n_rows =
                    (lt / cfg.patch_size[0]) * (lh / cfg.patch_size[1]) * (lw / cfg.patch_size[2]);
                video_parts.push(
                    CudaTensor::from_vec(rows, vec![n_rows, cfg.video_patch_dim()])?.to_device()?,
                );

                let mut num_audio_latents = 0usize;
                if super::media::probe_has_audio(&spec.path).unwrap_or(false) {
                    let planar = super::media::decode_audio_stereo_f32(
                        &spec.path,
                        sample_rate,
                        max_audio_samples,
                    )?;
                    let arows = audio_encoder.encode_stereo_planar(&planar)?;
                    num_audio_latents = arows.shape[0] / H3_AUDIO_CHANNELS;
                    audio_parts.push(arows);
                    crate::wan::log::info(format_args!(
                        "h3 ref2va: encode soundtrack {} → {num_audio_latents} audio latents",
                        i + 1
                    ));
                }
                prepared.push(PreparedReference::Video {
                    num_latent_frames: lt,
                    latent_height: lh,
                    latent_width: lw,
                    num_audio_latents,
                });
            }
        }
    }
    if video_parts.is_empty() {
        return Err(msg(
            "Ref2VA requires at least one visual (image/video) reference",
        ));
    }
    let video_rows = if video_parts.len() == 1 {
        video_parts.pop().unwrap()
    } else {
        let refs: Vec<&CudaTensor> = video_parts.iter().collect();
        CudaTensor::cat(&refs, 0)?
    };
    let audio_rows = if audio_parts.is_empty() {
        None
    } else if audio_parts.len() == 1 {
        Some(audio_parts.pop().unwrap())
    } else {
        let refs: Vec<&CudaTensor> = audio_parts.iter().collect();
        Some(CudaTensor::cat(&refs, 0)?)
    };
    Ok(Ref2VaEncoded {
        prepared,
        video_rows,
        audio_rows,
    })
}

/// Load, generate one clip, drop everything.
pub fn generate(
    root: &Path,
    options: H3PipelineOptions,
    request: &H3Request,
    out_dir: &Path,
) -> Result<H3Output> {
    H3Pipeline::load(root, options)?.generate(request, out_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::h3::packing::unpatchify;

    #[test]
    fn device_unpatchify_is_the_inverse_of_the_host_patchify() {
        let (c, t, h, w) = (3usize, 2usize, 4usize, 6usize);
        let lat: Vec<f32> = (0..c * t * h * w).map(|v| v as f32).collect();
        let rows = patchify(&lat, [c, t, h, w], [1, 2, 2]).unwrap();
        let got = unpatchify_rows(
            &CudaTensor::from_vec(rows.clone(), vec![t * 2 * 3, c * 4]).unwrap(),
            c,
            (t, 2, 3),
            [1, 2, 2],
        )
        .unwrap();
        assert_eq!(got.shape, vec![1, c, t, h, w]);
        assert_eq!(&*got.host_cow().unwrap(), &lat[..]);
        assert_eq!(unpatchify(&rows, [c, t, h, w], [1, 2, 2]).unwrap(), lat);
    }

    #[test]
    fn a_step_is_the_reference_blend_and_the_last_one_lands_on_x0() {
        let schedule = H3JointSchedule::fasth3_8step();
        let x = vec![0.5f32, -1.25, 2.0];
        let v = vec![1.0f32, 0.25, -0.5];
        let (xt, vt) = (
            CudaTensor::from_vec(x.clone(), vec![3, 1]).unwrap(),
            CudaTensor::from_vec(v.clone(), vec![3, 1]).unwrap(),
        );
        for (sched, step) in [
            (&schedule.video, 0usize),
            (&schedule.audio, 4),
            (&schedule.video, 7),
        ] {
            let got = scheduler_step(sched, step, &xt, &vt).unwrap();
            let want = sched.step(step, &x, &v).unwrap();
            for (g, w) in got.host_cow().unwrap().iter().zip(&want) {
                assert!((g - w).abs() <= 1e-6 * w.abs().max(1.0), "{g} vs {w}");
            }
        }
        // sigma[8] = 0: the last step returns x0 = x + sigma * v (note the plus).
        let last = scheduler_step(&schedule.video, 7, &xt, &vt).unwrap();
        let sigma = 1.0f32 - schedule.video.timesteps[7];
        assert!((last.host_cow().unwrap()[0] - (0.5 + sigma * 1.0)).abs() < 1e-6);
    }

    #[test]
    fn auto_keeps_the_encoder_resident_only_on_a_large_empty_card() {
        use TextEncoderChoice::*;
        assert_eq!(Auto.resolve(Some(95_000_000_000)), ResidentFp8);
        assert_eq!(Auto.resolve(Some(AUTO_RESIDENT_FREE_BYTES)), ResidentFp8);
        assert_eq!(
            Auto.resolve(Some(79_000_000_000)),
            Streamed,
            "an 80 GB card streams"
        );
        assert_eq!(
            Auto.resolve(None),
            Streamed,
            "no device, no resident encoder"
        );
        assert_eq!(
            Streamed.resolve(Some(u64::MAX)),
            Streamed,
            "an explicit choice is not second-guessed"
        );
        assert_eq!(ResidentFp8.resolve(Some(0)), ResidentFp8);
        for name in ["auto", "streamed", "resident-fp8", "resident-bf16"] {
            assert!(TextEncoderChoice::parse(name).is_ok());
        }
        assert!(TextEncoderChoice::parse("fp8").is_err());
    }

    #[test]
    fn noise_is_video_first_then_audio_rows_from_one_generator() {
        let cfg = H3TransformerConfig::fasth3_8step();
        let g = H3Geometry::default_16x9(5).unwrap();
        let (video, audio) = seeded_noise(&cfg, &g, 7).unwrap();
        assert_eq!((video.len(), audio.len()), (37_296 * 96, 414 * 32));
        assert_eq!(
            seeded_noise(&cfg, &g, 7).unwrap().1,
            audio,
            "a seed names one sample"
        );
        assert_ne!(seeded_noise(&cfg, &g, 8).unwrap().1, audio);
        let mean = video.iter().map(|&v| f64::from(v)).sum::<f64>() / video.len() as f64;
        let var = video.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / video.len() as f64;
        assert!(
            mean.abs() < 5e-3 && (var - 1.0).abs() < 5e-3,
            "N(0, 1): mean {mean} var {var}"
        );
    }

    #[test]
    fn spark_recipe_is_vsa_and_sol_h3_defaults_to_spark_attn() {
        let spark = resolve_contract(Path::new("/"), Some("sol-h3-spark")).unwrap();
        assert_eq!(spark.vsa_sparsity, 0.9);
        assert_eq!(spark.vsa_tile_size, 64);
        assert!(!spark.dense);
        let sol = resolve_contract(Path::new("/"), Some("sol-h3")).unwrap();
        assert!(!sol.dense);
        assert_eq!(sol.transformer_forwards, 4);
        assert_eq!(sol.vsa_sparsity, 0.0);
        let rtx = resolve_contract(Path::new("/"), Some("sol-h3-rtx")).unwrap();
        assert_eq!(rtx.transformer_forwards, 49);
        let vsa = resolve_contract(Path::new("/"), Some("4step-vsa")).unwrap();
        assert_eq!(vsa.transformer_forwards, 4);
        assert_eq!(vsa.vsa_sparsity, 0.9);
        assert_eq!(
            fastvideo_models::h3::sol::recipe_sol_attn_policy(Some("sol-h3"), None),
            fastvideo_models::h3::sol::H3SolAttnPolicy::Spark
        );
        assert_eq!(
            fastvideo_models::h3::sol::recipe_sol_attn_policy(Some("sol-h3-rtx"), None),
            fastvideo_models::h3::sol::H3SolAttnPolicy::Rtx
        );
        assert_eq!(
            fastvideo_models::h3::sol::recipe_sol_attn_policy(Some("sol-h3-spark"), None),
            fastvideo_models::h3::sol::H3SolAttnPolicy::Off
        );
        assert_eq!(
            fastvideo_models::h3::spark::FIXED_PROMPT,
            "4K, refined, high quality, cinematic detail, clean textures, natural motion."
        );
    }

    #[test]
    fn sol_h3_with_gate_follows_sparsity() {
        let sol = fastvideo_models::h3::config::H3InferenceContract::sol_h3();
        let spark = fastvideo_models::h3::config::H3InferenceContract::sol_h3_spark();
        assert!(!dit_loads_vsa_gate(&sol, None));
        assert!(dit_loads_vsa_gate(&spark, None));
        assert!(!dit_loads_vsa_gate(&spark, Some(false)));
        assert!(!sol.dense && !spark.dense);
    }
}
