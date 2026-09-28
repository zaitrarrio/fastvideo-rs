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
//! On a smaller card ([`H3PipelineOptions::dit_offload`], `auto` when the
//! free memory does not cover [`fastvideo_models::h3::memory::plan`]) the
//! refiner and DiT blocks are **streamed** from pinned host memory through a
//! ring of device slots ([`crate::wan::offload`]) and the video and audio
//! decoders are loaded for the decode only, as the reference's RTX 5090
//! profile runs: only the active component is on the GPU.
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
use fastvideo_models::h3::lora::{
    check_preview_base, is_sol_h3_recipe, preview_lora_strength, sol_h3_forces_ref2va,
    FastH3PreviewVariant, SolH3AdapterSpec, FASTH3_LORA_STRENGTH_ENV,
};
use fastvideo_models::h3::packing::{patchify, H3PackedLayout, KeyframeAnchor};
use fastvideo_models::h3::reference::{
    plan_reference_video_canvas, resample_reference_frames, resolve_reference_image_size_with,
    trim_reference_num_frames, validate_references, H3ReferenceSpec, PreparedImageRef,
    PreparedReference, ReferenceImageResize, ReferenceKind,
};
use fastvideo_models::h3::schedule::{H3JointSchedule, H3Schedule};
use fastvideo_models::h3::sol::H3SolAttnPolicy;
use fastvideo_models::h3::techniques::{H3Attention, H3Techniques};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::audio_vae::H3AudioDecoder;
use super::drain::FrameDrain;
use super::text::{CacheStatus, HiddenStateEncoder};
use super::transformer::{AttnMode, DeviceLayout, H3SolPolicy, H3TextRefiner, H3Transformer};
use super::vae::H3VideoDecoder;
use super::vae_encoder::H3VideoEncoder;
use super::vsa::H3Vsa;
use crate::hooks::{Hooks, Stage};
use crate::wan::offload::{DitOffload, MemoryLog, OffloadStats, PhaseMemory, Residency};
use crate::wan::pipeline::{interleave_audio, write_wav, PipelineError, Result};
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
    /// Aligned up to `17 n + 5`; 4 to 15 seconds at 24 fps (107 to 362
    /// frames). FastVideo-parity callers hold the 5 s floor
    /// (`H3_FASTVIDEO_MIN_DURATION_S`).
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

    /// An explicit canvas and frame count, checked against the model
    /// ([`H3Geometry::checked`]: multiples of 32, at most 768 x 1344 pixels,
    /// 1:4 to 4:1, 5 to 15 s once aligned to `17 n + 5` frames).
    pub fn sized(
        prompt: impl Into<String>,
        height: usize,
        width: usize,
        num_frames: usize,
        seed: u64,
    ) -> std::result::Result<Self, String> {
        let g = H3Geometry::checked(height, width, num_frames)?;
        let mut r = Self::seconds(prompt, 5, seed)?;
        (r.height, r.width, r.num_frames) = (g.height, g.width, g.num_frames);
        Ok(r)
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
    /// A resident `Auto` encoder is released after encoding when the denoise
    /// needs its memory ([`keep_auto_encoder`]); later prompts then stream.
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

/// Free device memory at which `Auto` loads the encoder resident: DiT 41 +
/// decoders 7 + FP8 Qwen 24.4 = 72 GB of weights plus headroom. That covers
/// the load and the first encode, not a denoise: the VSA path at 5 s needs
/// well over the ~12 GB of dense activations (a 96 GB card OOMed with ~27 GiB
/// free), so after encoding `Auto` releases the encoder unless
/// [`keep_auto_encoder`] says the denoise still fits beside it.
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
    /// Root holding `transformer_ref/` (and the Ref2VA turbo adapter) when it
    /// is not the snapshot itself, e.g. `weights/h3-ref2va` beside
    /// `weights/h3-base`. The VAEs and the text encoder still come from the
    /// snapshot (and `text_root`).
    pub ref_root: Option<PathBuf>,
    /// Sol-H3 / FastH3 Preview adapter file. When unset, the recipe's adapter is
    /// searched under and beside the weight root.
    pub adapter: Option<PathBuf>,
    /// Ref2VA reference-image fit. `Auto` is 2048 short-edge, and `match` for Sol-H3.
    pub reference_image_resize: ReferenceImageResize,
    /// Where the refiner and DiT blocks live. `None`: `FASTVIDEO_DIT_OFFLOAD`,
    /// else `Auto` (resident when the free memory covers the planned need).
    /// Streamed also loads the decoders for the decode only.
    pub dit_offload: Option<DitOffload>,
    /// Where the FL2VA / Ref2VA multimodal text encoder lives
    /// ([`I2vEncoderChoice`]).
    pub i2v_encoder: I2vEncoderChoice,
}

/// Where the multimodal (image-conditioned) text encoder's weights live.
///
/// The language model is the T2V text encoder (same checkpoint, same layers)
/// and runs at its precision either way; the vision tower is ~1.2 GB. So
/// `Resident` keeps the vision tower and reuses the resident text encoder, and
/// `Stream` reads both from the volume for every request (the original path:
/// ~50 GB of bf16 or ~24 GB of FP8 language model per image request). The two
/// give the same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum I2vEncoderChoice {
    /// `Resident` when the text encoder is resident and the vision tower fits
    /// beside the planned DiT; else `Stream`.
    #[default]
    Auto,
    Resident,
    Stream,
}

/// What [`I2vEncoderChoice::Auto`] sets aside for the vision tower (bf16
/// weights ~1.15 GB plus its forward's transients) when deciding residency.
pub const I2V_VISION_RESERVE_BYTES: u64 = 2 << 30;

impl I2vEncoderChoice {
    pub fn parse(name: &str) -> std::result::Result<Self, String> {
        match name {
            "auto" | "" => Ok(Self::Auto),
            "resident" => Ok(Self::Resident),
            "stream" | "streamed" => Ok(Self::Stream),
            other => Err(format!(
                "unknown i2v encoder '{other}' (auto|resident|stream)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Resident => "resident",
            Self::Stream => "stream",
        }
    }
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
    /// The whole video decode through a finished mp4: VAE, RGB8 conversion,
    /// copy down, hand-off to the writer (see the split below) and ffmpeg's
    /// exit. PNG frames are not in it (`write_s`).
    pub video_decode_s: f64,
    /// Part of `video_decode_s`: last frame handed over → ffmpeg exited.
    pub video_encode_s: f64,
    /// VAE compute alone (GPU time), the part comparable with FastVideo's
    /// `video_decoding_stage`.
    pub video_vae_s: f64,
    /// RGB8 packing and the device-to-host copy (off the decode thread).
    pub video_rgb_s: f64,
    /// Writer backpressure: time the drain thread waited in `VideoWriter::push`.
    pub video_push_s: f64,
    /// Time the decode thread waited on the frame hand-off or the final drain.
    pub video_wait_s: f64,
    /// `frame-NNN.png` written after the mp4 (see `wan::writer`).
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
    /// Peak device memory of each stage (empty without a device).
    pub memory: Vec<PhaseMemory>,
    /// `"resident"` or `"streamed"`.
    pub dit_residency: &'static str,
    /// Block streaming over the denoise (streamed runs).
    pub offload: Option<OffloadStats>,
}

/// Oracle injection of one modality's starting rows (`FASTVIDEO_INJECT_DIR`).
/// The reference dumps its packed rows: the target noise alone, or
/// `[condition | target]` when the request has condition rows. The target
/// part replaces `target`; the condition part replaces `cond` only with
/// `FASTVIDEO_INJECT_COND=1`. Returns `(target, cond)`.
fn inject_start_rows(
    name: &str,
    target: Vec<f32>,
    cond: Option<CudaTensor>,
) -> Result<(Vec<f32>, Option<CudaTensor>)> {
    let Some((shape, v)) = crate::wan::inject::load(name)? else {
        return Ok((target, cond));
    };
    let cond_n = cond.as_ref().map_or(0, CudaTensor::numel);
    if v.len() == target.len() {
        return Ok((v, cond));
    }
    if cond_n == 0 || v.len() != cond_n + target.len() {
        return Err(msg(format!(
            "FASTVIDEO_INJECT_DIR: {name}: reference shape {shape:?} ({} values); ours needs {} target (+ {cond_n} condition)",
            v.len(),
            target.len()
        )));
    }
    let tail = v[cond_n..].to_vec();
    let cond = match cond {
        Some(c) if std::env::var("FASTVIDEO_INJECT_COND").is_ok_and(|x| x == "1") => {
            crate::wan::log::info(format_args!(
                "inject: {name} condition rows ({cond_n} values) from the reference"
            ));
            Some(CudaTensor::from_vec(v[..cond_n].to_vec(), c.shape.clone())?.to_device()?)
        }
        c => c,
    };
    crate::wan::log::info(format_args!("inject: {name} target rows from the reference"));
    Ok((tail, cond))
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
    // FASTVIDEO_DUMP_DIR: the inputs, each step's velocity and latents, and the
    // first step's block outputs, for a two-run comparison (compare-dumps).
    let dumping = crate::wan::dump::enabled();
    if dumping {
        crate::wan::dump::tensor("text_refined", text_refined)?;
        crate::wan::dump::tensor("video_step00_in", &video)?;
        crate::wan::dump::tensor("audio_step00_in", &audio)?;
        for (tag, s) in [("video", &schedule.video), ("audio", &schedule.audio)] {
            crate::wan::dump::host(&format!("{tag}_sigmas"), &[s.sigmas.len()], &s.sigmas)?;
            crate::wan::dump::host(
                &format!("{tag}_timesteps"),
                &[s.timesteps.len()],
                &s.timesteps,
            )?;
        }
    }
    for step in 0..schedule.num_steps() {
        let mut dump_blocks = |name: &str, x: &CudaTensor| {
            crate::wan::dump::rows_strided(
                &format!("step00_{name}"),
                x,
                crate::wan::dump::BLOCK_ROW_STRIDE,
            )
        };
        let observer: Option<crate::h3::transformer::Observer<'_>> =
            (dumping && step == 0).then_some(&mut dump_blocks as _);
        let (v_video, v_audio) = model.forward(
            step,
            &video,
            &audio,
            text_refined,
            layout,
            mode,
            observer,
            cond_rows,
            cond_audio_rows,
        )?;
        video = scheduler_step(&schedule.video, step, &video, &v_video)?;
        audio = scheduler_step(&schedule.audio, step, &audio, &v_audio)?;
        if dumping {
            crate::wan::dump::tensor(&format!("video_vel_step{:02}", step + 1), &v_video)?;
            crate::wan::dump::tensor(&format!("video_step{:02}", step + 1), &video)?;
            crate::wan::dump::tensor(&format!("audio_vel_step{:02}", step + 1), &v_audio)?;
            crate::wan::dump::tensor(&format!("audio_step{:02}", step + 1), &audio)?;
        }
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
    /// The resident multimodal vision tower (zero when I2V streams).
    pub vision_s: f64,
    pub refiner_s: f64,
    pub dit_s: f64,
    pub video_vae_s: f64,
    pub audio_vae_s: f64,
}

enum VideoDecoder {
    Official(H3VideoDecoder),
    Taeh3(TaeHv),
}

/// VSA runs only with Sol off, a non-dense run, no interleaved condition
/// audio, and a VSA recipe. A zero-sparsity recipe never builds VSA (its DiT
/// is loaded without `to_gate_compress`).
fn uses_vsa(
    sol: H3SolAttnPolicy,
    dense: bool,
    force_dense: bool,
    contract: &H3InferenceContract,
) -> bool {
    sol == H3SolAttnPolicy::Off && !dense && !force_dense && contract.vsa_sparsity > 0.0
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
    // `FASTVIDEO_TAEH3_WEIGHTS`, or a profile's `[techniques.taeh3] weights`.
    let env = fastvideo_models::techniques::settings::var("FASTVIDEO_TAEH3_WEIGHTS")
        .unwrap_or_default();
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
    /// The technique set this pipeline runs (recipe + profile + env flags).
    techniques: H3Techniques,
    /// `options.dense` before a `dense_attention` technique forced it (the
    /// caller's `--dense` or a dense recipe); [`Self::set_arm`] restores it.
    base_dense: bool,
    schedule: H3JointSchedule,
    refiner: H3TextRefiner,
    model: H3Transformer,
    /// `None` when streamed: loaded for each decode and dropped after it.
    video_vae: Option<VideoDecoder>,
    audio_vae: Option<H3AudioDecoder>,
    residency: Residency,
    /// Resident encoder, if any. `Auto` may release it before a denoise that
    /// needs its memory (see [`keep_auto_encoder`]); later prompts then stream.
    text_encoder: std::sync::Mutex<Option<Box<dyn HiddenStateEncoder + Send>>>,
    /// The encoder choice was `Auto` (resolved at load), so it may be released.
    auto_text_encoder: bool,
    /// The resident vision tower + tokenizer of the multimodal text path
    /// (`None`: FL2VA/Ref2VA stream it per request).
    multimodal: Option<super::text::MultimodalEncoder>,
    /// Stream the multimodal encoder even when [`Self::multimodal`] is
    /// loaded ([`Self::set_i2v_encoder`]; the parity check).
    i2v_stream: bool,
    pub load_timings: H3LoadTimings,
    /// What this pipeline keeps on the device, by ledger category.
    booking: std::sync::Mutex<crate::wan::ledger::Booking>,
}

/// Device bytes a denoise needs beside the resident weights for `rows` packed
/// rows: 1 MiB per row (f32 QKVG `[S, 4 x 7168]`, BHSD Q/K/V, the SwiGLU
/// hidden and the VSA gathers are all live around one block) plus 4 GiB of
/// workspace. Conservative on purpose: a 96 GB card with the FP8 encoder,
/// DiT and decoders resident ran out at the first 5 s VSA step with ~27 GiB
/// free. Under bf16 activations (the GPU default) those tensors are half the
/// size: 640 KiB per row, measured at 5 s 1344x768 as ~25 GiB over the live
/// weights (the peak of a run that released the encoder), where the f32 rule
/// asked for 40.9 GiB and released a 22.7 GiB encoder with 40.8 GiB free, so
/// the warm request streamed it (~89 s of text encoding instead of ~0.5 s).
pub fn denoise_reserve_bytes(rows: usize) -> u64 {
    denoise_reserve_bytes_for(rows, crate::wan::tensor::bf16_activations())
}

/// [`denoise_reserve_bytes`] for an explicit activation width.
pub fn denoise_reserve_bytes_for(rows: usize, bf16_act: bool) -> u64 {
    let per_row: u64 = if bf16_act { 640 << 10 } else { 1 << 20 };
    (rows as u64) * per_row + (4u64 << 30)
}

/// Whether an `Auto`-resident encoder may stay on the device through a
/// denoise of `rows` packed rows, given the free bytes measured after the
/// prompt was encoded. Unknown free memory releases it.
pub fn keep_auto_encoder(free_bytes: Option<u64>, rows: usize) -> bool {
    free_bytes.is_some_and(|free| free >= denoise_reserve_bytes(rows))
}

impl H3Pipeline {
    /// `root` is the FastH3 snapshot (`transformer/`, `vae/`, `audio_vae/`, and
    /// `tokenizer/` + `text_encoder/` unless `options.text_root` says otherwise).
    pub fn load(root: &Path, options: H3PipelineOptions) -> Result<Self> {
        crate::wan::tensor::default_bf16_activations();
        let mut options = options;
        // A technique profile names its recipe; the caller's wins.
        let active = fastvideo_models::techniques::settings::active();
        if let Some(profile) = active.profile.as_ref() {
            match (&options.recipe, &profile.recipe) {
                (None, Some(r)) => options.recipe = Some(r.clone()),
                (Some(a), Some(b)) if a != b => crate::wan::log::info(format_args!(
                    "h3 techniques: recipe {a} (command line) overrides profile {}'s {b}",
                    profile.name
                )),
                _ => {}
            }
        }
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
        let techniques =
            H3Techniques::from_process(options.recipe.as_deref(), &contract, options.ref2va)
                .map_err(msg)?;
        let base_dense = options.dense;
        if techniques.forces_dense() {
            options.dense = true;
        }
        if techniques.taeh3.is_some() && resolve_taeh3(options.taeh3.as_deref()).is_none() {
            return Err(msg(
                "techniques.taeh3: no TAEH3 weights (pass --taeh3-weights, set FASTVIDEO_TAEH3_WEIGHTS, or give techniques.taeh3.weights)",
            ));
        }
        if let Some(path) = active.path.as_ref() {
            crate::wan::log::info(format_args!(
                "h3 technique profile: {}",
                path.display()
            ));
            for note in active.profile.iter().flat_map(|p| p.notes.iter()) {
                crate::wan::log::info(format_args!("h3 technique profile note: {note}"));
            }
            for (k, v, source) in active.settings.iter() {
                crate::wan::log::info(format_args!(
                    "h3 technique setting {k}={v} ({source}{})",
                    if std::env::var_os(k).is_some() { "; overridden by env" } else { "" }
                ));
            }
        }
        crate::wan::log::info(format_args!("{}", techniques.describe()));
        let mut load_timings = H3LoadTimings::default();
        let timed = |slot: &mut f64, timer: Instant| *slot = timer.elapsed().as_secs_f64();

        // The resident encoder goes first: its FP8 quantization runs on the
        // device with f32 transients (524 MB for the widest matrix), which
        // should happen while the card is otherwise empty.
        let free = crate::wan::device::free_memory().map(|(free, _)| free);
        let auto_text_encoder = options.text_encoder == TextEncoderChoice::Auto;
        options.text_encoder = options.text_encoder.resolve(free);
        let mut booking = crate::wan::ledger::Booking::default();

        // E12: open every component's shards up front (headers only) and
        // queue their tensors for read-ahead in load order — encoder, refiner,
        // AdaLN projections, blocks, decoders — so the volume stays busy
        // across component boundaries. The loads below read the same mappings.
        let io_base = fastvideo_loader::PrefetchStats::now();
        let io_timer = Instant::now();
        let text_map = match options.text_encoder.precision() {
            Some(precision) => Some(super::text::open_resident_encoder(
                options.text_root.as_deref().unwrap_or(root),
                precision,
            )?),
            None => None,
        };
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
                    let p = options.ref_root.as_deref().unwrap_or(root).join("transformer_ref");
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
        {
            // A fused adapter rebuilds the AdaLN table from the projections;
            // otherwise a valid cache file means they are never read.
            let adapter = options
                .recipe
                .as_deref()
                .is_some_and(|r| FastH3PreviewVariant::from_recipe(r).is_some() || is_sol_h3_recipe(r));
            let skip_adaln = !adapter && options.adaln_cache.as_deref().is_some_and(Path::is_file);
            let adaln = |k: &str| k.contains(".adaln_proj.");
            let refiner = |k: &str| k.starts_with("token_refiner") || k.starts_with("context_embedder") || k.starts_with("refiner.");
            let table = |k: &str| {
                k.starts_with("time_embedder") || k.starts_with("norm_out.linear") || (!skip_adaln && adaln(k))
            };
            let blocks = |k: &str| (k.starts_with("transformer_blocks.") || k.starts_with("blocks.")) && !adaln(k);
            let rest = |k: &str| !adaln(k);
            map.prefetch_groups(&[&refiner, &table, &blocks, &rest]);
        }
        let not_encoder = |k: &str| !k.starts_with("encoder.");
        let mut vae_map = match resolve_taeh3(options.taeh3.as_deref()) {
            Some(_) => None,
            None => WeightMap::open(&root.join("vae")).ok(),
        };
        if let Some(m) = &vae_map {
            m.prefetch_groups(&[&not_encoder]);
        }
        let mut audio_vae_map = WeightMap::open(&root.join("audio_vae")).ok();
        if let Some(m) = &audio_vae_map {
            m.prefetch_groups(&[&not_encoder]);
        }

        let timer = Instant::now();
        let text_encoder = booking.track(
            crate::wan::ledger::TEXT_ENCODER,
            || -> Result<Option<Box<dyn HiddenStateEncoder + Send>>> {
                Ok(match options.text_encoder {
                    TextEncoderChoice::Recovered8b => {
                        let text_root = options.text_root.as_deref().unwrap_or(root);
                        Some(Box::new(super::recovered_8b::Recovered8bEncoder::load(
                            text_root,
                        )?))
                    }
                    other => match (other.precision(), text_map.as_ref()) {
                        (Some(precision), Some(map)) => Some(Box::new(
                            super::text::load_resident_encoder_from(map, precision)?,
                        )),
                        _ => None,
                    },
                })
            },
            |r| match r {
                Ok(Some(e)) => Some(e.resident_bytes()),
                _ => None,
            },
        )?;
        drop(text_map);
        if text_encoder.is_some() {
            timed(&mut load_timings.text_encoder_s, timer);
            crate::wan::weights::log_load_io(
                "h3 text_encoder",
                &io_base,
                io_timer.elapsed().as_secs_f64(),
            );
        }
        // What the DiT placement decides on: the card as the encoder left it.
        let free = if text_encoder.is_some() {
            crate::wan::device::free_memory().map(|(free, _)| free)
        } else {
            free
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
        let preset = techniques
            .sol()
            .and_then(|s| s.preset.as_deref())
            .and_then(fastvideo_models::techniques::methods::SolAttn::preset);
        let named = techniques.sol().is_some_and(|s| preset.as_ref() == Some(s));
        match techniques.sol_policy {
            _ if techniques.sol().is_some() && !named => {
                crate::wan::log::info(format_args!(
                    "h3 sol-attn: {}",
                    techniques.sol().map(|s| s.route.describe()).unwrap_or_default()
                ));
            }
            H3SolAttnPolicy::Engine => {
                crate::wan::log::info(format_args!(
                    "h3 sol-attn: sol-h3 engine policy (opt-in): forward 0 dense, blocks 0-1 dense, blocks 2-49 sol-attn tau 1.0 (thresh_type=diag), prefix sink"
                ));
            }
            H3SolAttnPolicy::Spark => {
                crate::wan::log::info(format_args!(
                    "h3 sol-attn: spark draft ladder (opt-in): update 0 dense, later updates block 0 dense and blocks 1-49 sol-attn tau 1/1.25/1.5 (thresh_type=diag), [visual | text+audio] suffix sink"
                ));
            }
            H3SolAttnPolicy::Rtx => {
                crate::wan::log::info(format_args!(
                    "h3 sol-attn: rtx forwards 0-9 dense, later forwards blocks 0-1 dense and blocks 2-49 sol-attn tau 1.0 (thresh_type=diag), text sink"
                ));
            }
            H3SolAttnPolicy::Off => {}
        }
        let preview = options
            .recipe
            .as_deref()
            .and_then(FastH3PreviewVariant::from_recipe);
        let mut lora = if let Some(variant) = preview {
            // FastH3 Preview v1 = base MiniMax-H3 + one Preview LoRA
            // (`run_fasth3_lora_preview_*_datafree.sh`).
            let has_gate = map.has_tensor("transformer_blocks.0.attn.to_gate_compress.weight");
            let has_json = [
                "fastvideo_inference.json",
                "transformer/fastvideo_inference.json",
            ]
            .iter()
            .any(|rel| root.join(rel).is_file());
            check_preview_base(has_gate, has_json).map_err(msg)?;
            let strength =
                preview_lora_strength(std::env::var(FASTH3_LORA_STRENGTH_ENV).ok().as_deref())
                    .map_err(msg)?;
            let spec = variant.adapter_spec(strength);
            let recipe = options.recipe.as_deref().unwrap_or_default();
            let path = spec.resolve(root, options.adapter.as_deref()).map_err(|e| {
                msg(format!(
                    "recipe {recipe} needs the FastH3 Preview v1 {} adapter (FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA): {e}",
                    variant.subdir()
                ))
            })?;
            let fuse = super::lora::H3LoraFuse::open(&map, &path, spec.alpha, spec.scale)?;
            variant
                .check_adapter(&fuse.replacement_params())
                .map_err(|e| msg(format!("{}: {e}", path.display())))?;
            crate::wan::log::info(format_args!(
                "h3 fasth3-preview {}: fuse {} pairs rank={} strength={} diffs={} set_weight={} ({})",
                variant.subdir(),
                fuse.pairs_total,
                fuse.rank,
                fuse.effective_scale(),
                fuse.diffs_total,
                fuse.replacements_total,
                path.display()
            ));
            Some(fuse)
        } else if options.recipe.as_deref().is_some_and(is_sol_h3_recipe) {
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
            let adapter_root = match (options.ref2va, options.ref_root.as_deref()) {
                (true, Some(r)) => r,
                _ => root,
            };
            let path = spec
                .resolve(adapter_root, options.adapter.as_deref())
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
                    "h3 sol-h3-spark: stage-1 VSA {} tile {} BF16 FastH3_VSA_DataFree strength {}; DiT {} (upstream: W8A8 after the bf16 LoRA merge; FASTVIDEO_H3_QUANT=w8a8). Draft {}x{} {}f. H3×2 upscaler and H3-to-LTX adapter run when their checkpoints are set. Joint 3-step LTX refine uses the fixed prompt and cached Gemma",
                    contract.vsa_sparsity,
                    contract.vsa_tile_size,
                    spec.scale,
                    crate::wan::quant::QuantMode::from_env()
                        .map(|m| m.as_str())
                        .unwrap_or("invalid FASTVIDEO_H3_QUANT"),
                    fastvideo_models::h3::sol::SPARK_DRAFT_WIDTH,
                    fastvideo_models::h3::sol::SPARK_DRAFT_HEIGHT,
                    fastvideo_models::h3::sol::SPARK_DRAFT_FRAMES,
                ));
            }
            Some(fuse)
        } else {
            None
        };
        // The multimodal (FL2VA / Ref2VA) text encoder: the vision tower
        // beside the resident language model, decided before the DiT
        // placement so the placement sees what it leaves.
        let (multimodal, free) = {
            let lm = text_encoder
                .as_deref()
                .and_then(|e| e.decoder())
                .map(|d| d.precision());
            let need = H3Geometry::default_16x9(5)
                .map(|g| {
                    fastvideo_models::h3::memory::plan(
                        &g,
                        fastvideo_models::h3::memory::H3PlanOptions {
                            gate: with_gate,
                            ..Default::default()
                        },
                    )
                    .peak()
                })
                .map_err(msg)?;
            let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
            let (load, why) = match (options.i2v_encoder, lm) {
                (I2vEncoderChoice::Stream, _) => (false, "configured".to_owned()),
                (I2vEncoderChoice::Resident | I2vEncoderChoice::Auto, None) => {
                    if options.i2v_encoder == I2vEncoderChoice::Resident {
                        return Err(msg(format!(
                            "i2v_encoder = resident needs a resident Qwen3-VL text encoder (text_encoder = resident-fp8 | resident-bf16, or auto on a card with {:.0} GB free); it resolved to {:?}",
                            AUTO_RESIDENT_FREE_BYTES as f64 / 1e9,
                            options.text_encoder
                        )));
                    }
                    (false, format!("the text encoder is {:?}, not a resident Qwen3-VL", options.text_encoder))
                }
                (I2vEncoderChoice::Resident, Some(_)) => (true, "configured".to_owned()),
                (I2vEncoderChoice::Auto, Some(_)) => match free {
                    Some(f) if f >= need + I2V_VISION_RESERVE_BYTES => (
                        true,
                        format!(
                            "{:.1} GiB free covers the vision tower ({:.1} GiB) and the resident plan ({:.1} GiB)",
                            gib(f),
                            gib(I2V_VISION_RESERVE_BYTES),
                            gib(need)
                        ),
                    ),
                    Some(f) => (
                        false,
                        format!(
                            "{:.1} GiB free does not cover the vision tower ({:.1} GiB) and the resident plan ({:.1} GiB)",
                            gib(f),
                            gib(I2V_VISION_RESERVE_BYTES),
                            gib(need)
                        ),
                    ),
                    None => (false, "free device memory unknown".to_owned()),
                },
            };
            if load {
                let timer = Instant::now();
                let text_root = options.text_root.as_deref().unwrap_or(root);
                let before = crate::wan::device::free_memory().map(|(f, _)| f);
                let mm = super::text::MultimodalEncoder::load(text_root)?;
                let after = crate::wan::device::free_memory().map(|(f, _)| f);
                let held = before.zip(after).map(|(b, a)| b.saturating_sub(a));
                load_timings.vision_s = timer.elapsed().as_secs_f64();
                crate::wan::log::info(format_args!(
                    "h3 i2v encoder: resident ({why}): vision tower {} GiB loaded in {:.1} s; the language model is the resident text encoder ({:?}), {:.1} GiB free now",
                    held.map_or("?".into(), |b| format!("{:.2}", gib(b))),
                    load_timings.vision_s,
                    options.text_encoder,
                    after.map_or(0.0, gib),
                ));
                (Some(mm), after.or(free))
            } else {
                crate::wan::log::info(format_args!(
                    "h3 i2v encoder: stream ({why}): FL2VA/Ref2VA requests read the vision tower and the {} language model from the volume per request",
                    match options.text_encoder.precision() {
                        Some(crate::llm::WeightPrecision::Fp8Rows) => "fp8",
                        _ => "bf16",
                    }
                ));
                (None, free)
            }
        };
        let residency = {
            let policy = DitOffload::from_env_or(options.dit_offload).map_err(msg)?;
            // What the default request (768p, 5 s) needs with everything resident.
            let need = H3Geometry::default_16x9(5)
                .map(|g| {
                    fastvideo_models::h3::memory::plan(
                        &g,
                        fastvideo_models::h3::memory::H3PlanOptions {
                            gate: with_gate,
                            ..Default::default()
                        },
                    )
                    .peak()
                })
                .map_err(msg)?;
            let residency = policy.resolve(need, free);
            crate::wan::log::info(format_args!(
                "h3 dit offload: {} -> {} (resident plan {:.1} GiB, {} free before load)",
                policy.as_str(),
                residency.as_str(),
                need as f64 / f64::from(1u32 << 30),
                free.map_or("unknown".into(), |f| format!(
                    "{:.1} GiB",
                    f as f64 / f64::from(1u32 << 30)
                )),
            ));
            residency
        };
        let timer = Instant::now();
        let refiner = booking.track(
            crate::wan::ledger::TEXT_REFINER,
            || H3TextRefiner::load_with_residency(&cfg, &map, &mut lora, residency),
            |_| None,
        )?;
        timed(&mut load_timings.refiner_s, timer);
        let timer = Instant::now();
        let model = booking.track(
            crate::wan::ledger::DIT_NONLINEAR,
            || {
                H3Transformer::load_with_residency(
                    cfg.clone(),
                    &map,
                    &schedule,
                    with_gate,
                    options.adaln_cache.as_deref(),
                    &mut lora,
                    residency,
                )
            },
            |_| None,
        )?;
        // The step-output seam: TeaCache over the recipe's forwards unless
        // the technique names its own count.
        if let Some(state) = techniques.teacache_state(schedule.num_steps()).map_err(msg)? {
            if techniques.teacache_is_official() {
                crate::wan::log::info(format_args!(
                    "{}",
                    fastvideo_models::h3::sol::TEACACHE_APPLIED
                ));
            } else {
                crate::wan::log::info(format_args!(
                    "h3 teacache: threshold {} retain {} cooldown {} coefficients {:?}",
                    state.threshold, state.retain_steps, state.cooldown_steps, state.coefficients
                ));
            }
            model.enable_teacache(state)?;
        }
        if let Some(fuse) = lora.as_ref() {
            fuse.finish()?;
            #[cfg(feature = "cuda")]
            {
                use super::transformer::lora_device;
                crate::wan::log::info(format_args!(
                    "h3 lora merge: {}",
                    if lora_device::enabled() { "device" } else { "host" }
                ));
                if lora_device::verify_enabled() {
                    let (ok, bad) = lora_device::verify_counts();
                    crate::wan::log::info(format_args!(
                        "load/verify h3 lora {{\"device_equal\":{ok},\"device_different\":{bad}}}"
                    ));
                }
            }
        }
        timed(&mut load_timings.dit_s, timer);
        drop(map);
        crate::wan::weights::log_load_io("h3 dit", &io_base, io_timer.elapsed().as_secs_f64());
        // Streamed: the decoders are loaded by each decode (the reference makes
        // the full VAE resident only for decoding).
        let (video_vae, audio_vae) = if residency.is_streamed() {
            (None, None)
        } else {
            let timer = Instant::now();
            let video_vae = booking.track(
                crate::wan::ledger::VAE,
                || load_video_decoder_from(root, options.taeh3.as_deref(), vae_map.take()),
                |_| None,
            )?;
            timed(&mut load_timings.video_vae_s, timer);
            let timer = Instant::now();
            let audio_vae = booking.track(
                crate::wan::ledger::AUDIO_VAE,
                || load_audio_decoder_from(root, audio_vae_map.take()),
                |_| None,
            )?;
            timed(&mut load_timings.audio_vae_s, timer);
            (Some(video_vae), Some(audio_vae))
        };
        drop((vae_map, audio_vae_map));
        crate::wan::weights::log_load_io("h3 total", &io_base, io_timer.elapsed().as_secs_f64());
        Ok(Self {
            root: root.to_path_buf(),
            options,
            cfg,
            contract,
            techniques,
            base_dense,
            schedule,
            refiner,
            model,
            video_vae,
            audio_vae,
            residency,
            text_encoder: std::sync::Mutex::new(text_encoder),
            auto_text_encoder,
            multimodal,
            i2v_stream: false,
            load_timings,
            booking: std::sync::Mutex::new(booking),
        })
    }

    /// Replace the text encoder (tests, or an encoder built elsewhere). It is
    /// consulted only on a conditioning-cache miss.
    pub fn with_text_encoder(mut self, encoder: Box<dyn HiddenStateEncoder + Send>) -> Self {
        if !matches!(
            self.options.text_encoder,
            TextEncoderChoice::ResidentBf16
                | TextEncoderChoice::ResidentFp8
                | TextEncoderChoice::Recovered8b
        ) {
            self.options.text_encoder = TextEncoderChoice::ResidentFp8;
        }
        *self.text_encoder.get_mut().expect("h3 text encoder") = Some(encoder);
        self.auto_text_encoder = false;
        self
    }

    /// Where FL2VA / Ref2VA requests encode their multimodal text now:
    /// `Resident` (the vision tower is loaded and not overridden) or `Stream`.
    pub fn i2v_encoder(&self) -> I2vEncoderChoice {
        if self.multimodal.is_some() && !self.i2v_stream {
            I2vEncoderChoice::Resident
        } else {
            I2vEncoderChoice::Stream
        }
    }

    /// Switch the multimodal text path between the resident vision tower and
    /// per-request streaming (same bytes either way; used by the parity
    /// check). `Resident` needs the tower loaded at [`Self::load`].
    pub fn set_i2v_encoder(&mut self, choice: I2vEncoderChoice) -> Result<()> {
        match choice {
            I2vEncoderChoice::Stream => self.i2v_stream = true,
            I2vEncoderChoice::Resident | I2vEncoderChoice::Auto => {
                if self.multimodal.is_none() && choice == I2vEncoderChoice::Resident {
                    return Err(msg(
                        "i2v encoder: the vision tower was not loaded (i2v_encoder resolved to stream at load)",
                    ));
                }
                self.i2v_stream = false;
            }
        }
        Ok(())
    }

    /// `(kind, device bytes)` of the encoder a cache miss will use.
    pub fn text_encoder(&self) -> (&'static str, u64) {
        self.text_encoder
            .lock()
            .expect("h3 text encoder")
            .as_ref()
            .map_or(("streamed", 0), |e| (e.kind(), e.resident_bytes()))
    }

    /// Encode `prompt` with the resident encoder AND by streaming, bypassing
    /// the cache, and return `(rel_l2, cosine)` of resident against streamed.
    /// What matters about a quantized encoder is how far it moves the
    /// conditioning; this is that number for the prompt at hand. `None` when
    /// the encoder is streamed anyway. Costs one streamed encode (~10 s).
    pub fn text_encoder_drift(&self, prompt: &str) -> Result<Option<(f64, f64)>> {
        let guard = self.text_encoder.lock().expect("h3 text encoder");
        let Some(resident) = guard.as_ref() else {
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

    /// The resolved technique set (recipe + profile + env flags).
    pub fn techniques(&self) -> &H3Techniques {
        &self.techniques
    }

    /// Switch the loaded pipeline to another technique profile's runtime
    /// choices (one arm of a multi-arm run: `fv-gpucheck h3 gen --arm`),
    /// without reloading anything. `None` is the process's own profile.
    ///
    /// Only what is decided per request may differ: the attention route
    /// (dense / VSA and its schedule / Sol-Attn and its route), FP8
    /// attention and TeaCache. Everything fixed at load (recipe, linear and
    /// activation precision, kernels, residency, decoder, every installed
    /// `FASTVIDEO_*` setting) must equal the process's, or this is an error
    /// and the arm needs its own process.
    pub fn set_arm(
        &mut self,
        profile: Option<&fastvideo_models::techniques::Profile>,
    ) -> Result<()> {
        let active = fastvideo_models::techniques::settings::active();
        let profile = profile.or(active.profile.as_ref());
        if let Some(p) = profile {
            if let (Some(want), Some(have)) = (p.recipe.as_deref(), self.options.recipe.as_deref())
            {
                if want != have {
                    return Err(msg(format!(
                        "arm {}: recipe {want} differs from the loaded {have}",
                        p.name
                    )));
                }
            }
            let settings = p.settings().map_err(msg)?;
            for (k, v, source) in settings.iter() {
                if active.settings.get(k) != Some(v) {
                    return Err(msg(format!(
                        "arm {}: technique '{source}' installs {k}={v}, the loaded process has {:?}; run it in its own process",
                        p.name,
                        active.settings.get(k)
                    )));
                }
            }
            for (k, v, _) in active.settings.iter() {
                if settings.get(k) != Some(v) {
                    return Err(msg(format!(
                        "arm {}: the loaded process installs {k}={v}, the arm does not; run it in its own process",
                        p.name
                    )));
                }
            }
        }
        let techniques = H3Techniques::resolve(
            self.options.recipe.as_deref(),
            &self.contract,
            self.options.ref2va,
            profile,
            &|k| fastvideo_models::techniques::settings::var(k),
        )
        .map_err(msg)?;
        if techniques.taeh3.is_some() != self.techniques.taeh3.is_some() {
            return Err(msg("arm: the video decoder (taeh3) is fixed at load"));
        }
        // `dense` came from the caller, the contract, or a dense_attention
        // technique; only the last one belongs to the arm.
        let was_forced = self.techniques.forces_dense();
        if was_forced && !techniques.forces_dense() && !self.contract.dense {
            self.options.dense = self.base_dense;
        }
        if techniques.forces_dense() {
            self.options.dense = true;
        }
        self.model.disable_teacache();
        if let Some(state) = techniques
            .teacache_state(self.schedule.num_steps())
            .map_err(msg)?
        {
            self.model.enable_teacache(state)?;
        }
        crate::wan::log::info(format_args!(
            "h3 arm {}: {}",
            profile.map_or("(none)", |p| p.name.as_str()),
            techniques.describe()
        ));
        self.techniques = techniques;
        Ok(())
    }

    pub fn options(&self) -> &H3PipelineOptions {
        &self.options
    }

    /// Where the refiner and DiT blocks live for this pipeline.
    pub fn residency(&self) -> Residency {
        self.residency
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
                    "h3 sol-h3-spark: no upscaler ({}) or H3-to-LTX adapter beside {}; H3 decode continues",
                    fastvideo_models::h3::spark::UPSCALER_FILES.join(" | "),
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
        self.generate_with_hooks(request, out_dir, Hooks::NONE)
    }

    /// [`Self::generate`] with cancellation and progress (serve E1): a stage
    /// event at text / denoise / audio / video, one per denoise step, one per
    /// decoded chunk; a tripped token stops at the next of those with
    /// [`PipelineError::Cancelled`], after the step caches, the streamed
    /// ring and the pool are handed back.
    pub fn generate_with_hooks(
        &self,
        request: &H3Request,
        out_dir: &Path,
        hooks: Hooks<'_>,
    ) -> Result<H3Output> {
        let out = self.generate_hooked(request, out_dir, hooks);
        if matches!(&out, Err(e) if e.is_cancelled()) {
            self.model.end_denoise();
            self.model.release_offload_device();
            crate::wan::device::trim_pool().map_err(|e| msg(e.to_string()))?;
        }
        out
    }

    fn generate_hooked(
        &self,
        request: &H3Request,
        out_dir: &Path,
        hooks: Hooks<'_>,
    ) -> Result<H3Output> {
        let cfg = &self.cfg;
        let geometry =
            H3Geometry::new(request.height, request.width, request.num_frames).map_err(msg)?;
        if hooks.has_sink()
            && self
                .options
                .recipe
                .as_deref()
                .is_some_and(fastvideo_models::h3::lora::is_sol_h3_spark_recipe)
        {
            return Err(msg(
                "h3: a frame sink is not supported with sol-h3-spark (its LTX refiner writes its own clip)",
            ));
        }
        let mut timings = H3Timings::default();
        let mut memory = MemoryLog::start("h3");

        // --- text: cache, else resident, else ~50 GB streamed one layer at a time ---
        hooks.stage(Stage::Text, 0)?;
        let timer = Instant::now();
        let mut encoder_slot = self.text_encoder.lock().expect("h3 text encoder");
        let resident_choice = matches!(
            self.options.text_encoder,
            TextEncoderChoice::ResidentBf16
                | TextEncoderChoice::ResidentFp8
                | TextEncoderChoice::Recovered8b
        );
        let resident = match encoder_slot.as_deref() {
            Some(encoder) if resident_choice => Some(encoder as &dyn HiddenStateEncoder),
            // An `Auto` encoder released before an earlier denoise: stream.
            None if resident_choice && !self.auto_text_encoder => {
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
            let lm = match (self.i2v_stream, self.multimodal.as_ref()) {
                (false, Some(mm)) => match encoder_slot.as_deref().and_then(|e| e.decoder()) {
                    Some(decoder) => Some((mm, decoder)),
                    // An `Auto` encoder released before an earlier denoise.
                    None => None,
                },
                _ => None,
            };
            let precision = self
                .options
                .text_encoder
                .precision()
                .unwrap_or(crate::llm::WeightPrecision::Native);
            let abort = || hooks.is_cancelled();
            let text = encode_request_multimodal(
                encoder_root,
                request,
                lm,
                precision,
                &abort,
                self.options.reference_image_resize,
            );
            // A cancel during the (streamed) multimodal stage stops it
            // between layers; report it as the cancel it is.
            hooks.check()?;
            text?
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
        // Auto kept the encoder resident only because the card was empty at
        // load. The conditioning is encoded (and cached) now; release the
        // encoder when the denoise needs its memory.
        let rows = text.ids.len() + geometry.video_rows() + geometry.audio_rows();
        if self.auto_text_encoder && encoder_slot.is_some() {
            let free = crate::wan::device::free_memory().map(|(free, _)| free);
            if !keep_auto_encoder(free, rows) {
                let released = encoder_slot.take().map_or(0, |e| e.resident_bytes());
                self.booking
                    .lock()
                    .expect("h3 booking")
                    .release(crate::wan::ledger::TEXT_ENCODER);
                crate::wan::log::info(format_args!(
                    "h3 text encoder: auto released the resident encoder ({:.1} GiB) before the denoise ({rows} rows need {:.1} GiB, {:.1} GiB free); later prompts stream",
                    released as f64 / f64::from(1u32 << 30),
                    denoise_reserve_bytes(rows) as f64 / f64::from(1u32 << 30),
                    free.unwrap_or(0) as f64 / f64::from(1u32 << 30),
                ));
            }
        }
        drop(encoder_slot);
        let mut text = text;
        // FASTVIDEO_DUMP_DIR: ours before any injection (the text-encoder
        // parity); FASTVIDEO_INJECT_DIR: then the reference's Qwen hidden
        // states in place of ours (same tokenizer, so the same row count), so
        // a parity run measures the DiT and not the text-encoder precision.
        crate::wan::dump::tensor("text_hidden", &text.hidden)?;
        if crate::wan::inject::text_enabled() {
            match crate::wan::inject::load("text_hidden")? {
                Some((_, v)) if v.len() == text.hidden.numel() => {
                    let shape = text.hidden.shape.clone();
                    text.hidden = CudaTensor::from_vec(v, shape)?.to_device()?;
                }
                Some((shape, _)) => crate::wan::log::info(format_args!(
                    "inject: text_hidden {shape:?} does not fit ours {:?} (a different token count); ours kept",
                    text.hidden.shape
                )),
                None => {}
            }
        }
        memory.mark("text")?;
        let timer = Instant::now();
        let text_refined = self.refiner.forward(&text.hidden)?;
        timings.refine_s = timer.elapsed().as_secs_f64();
        memory.mark("refine")?;

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
        let sol_kind = self.techniques.sol_policy;
        // A Sol route with `dense_backend = "vsa"` runs its dense calls on VSA.
        let sol_dense_vsa = self.techniques.sol().is_some_and(|s| s.dense_vsa)
            && !self.options.dense
            && !force_dense
            && self.contract.vsa_sparsity > 0.0;
        let vsa = if !sol_dense_vsa
            && !uses_vsa(sol_kind, self.options.dense, force_dense, &self.contract)
        {
            if force_dense && !self.options.dense && sol_kind == H3SolAttnPolicy::Off {
                crate::wan::log::info(format_args!(
                    "h3 ref2va: dense attention (interleaved condition audio)"
                ));
            }
            None
        } else {
            // FASTVIDEO_VSA_SPARSITY overrides the profile's, which overrides
            // the recipe's (the oracle's control: 0 keeps every tile,
            // removing the top-k selection while keeping the gated
            // compression branch, as FastVideo's `--vsa-sparsity 0`).
            let sparsity = self.techniques.vsa_sparsity(&self.contract).map_err(msg)?;
            let vsa_cfg = super::vsa::H3VsaConfig {
                sparsity,
                group: self.techniques.vsa_group,
                tile_size: self.contract.vsa_tile_size,
            };
            Some(
                H3Vsa::new(
                    &layout,
                    cfg.num_attention_heads,
                    cfg.attention_head_dim,
                    vsa_cfg,
                )?
                .with_schedule(sparsity, self.techniques.vsa_schedule.clone())
                .with_fp8(self.techniques.fp8_attention.vsa),
            )
        };
        let layout = DeviceLayout::new(cfg, layout)?;
        let sol_policy = match &self.techniques.attention {
            H3Attention::Auto | H3Attention::Dense => None,
            H3Attention::Sol(technique) => {
                let kind = sol_kind;
                let policy = H3SolPolicy::from_technique(technique, &layout)?;
                crate::wan::log::info(format_args!(
                    "{}",
                    fastvideo_models::h3::sol::describe_sink(
                        kind,
                        &fastvideo_models::h3::sol::H3SolSinkSpec {
                            sink: policy.sink,
                            plan: None,
                        }
                    )
                ));
                Some(policy)
            }
        };
        let mode = if let Some(ref policy) = sol_policy {
            match vsa.as_ref() {
                Some(vsa) if sol_dense_vsa => AttnMode::SolVsa { policy, vsa },
                _ => AttnMode::Sol(policy),
            }
        } else {
            vsa.as_ref().map_or(AttnMode::Dense, AttnMode::Vsa)
        };
        let (mut video_noise, mut audio_noise) =
            seeded_noise(cfg, &geometry, request.seed).map_err(msg)?;
        // FASTVIDEO_INJECT_DIR: the reference's packed starting rows (torch's
        // draws, patchified by the reference), in place of our seeded noise.
        // With condition rows (FL2VA / Ref2VA) the reference's rows are
        // `[condition | target]`; only the target part is injected unless
        // FASTVIDEO_INJECT_COND=1, which also takes its condition rows (the
        // control that isolates the DiT from our reference encode).
        let (cond_rows, cond_audio_rows) = {
            let (v, c) = inject_start_rows("video_step00_in", video_noise, cond_rows)?;
            video_noise = v;
            let (a, ca) = inject_start_rows("audio_step00_in", audio_noise, cond_audio_rows)?;
            audio_noise = a;
            (c, ca)
        };
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

        let steps = self.schedule.num_steps();
        hooks.stage(Stage::Denoise, steps)?;
        let mut step_s = Vec::with_capacity(steps);
        // FASTVIDEO_GPU_TRACE: one step's device activity (a no-op when off).
        crate::wan::gpu_trace::pass_begin("h3");
        crate::wan::gpu_trace::step_begin();
        let timer = Instant::now();
        let mut last = Instant::now();
        // fp8_attention (opt-in): dense calls of this denoise on the FP8 kernel;
        // the refiner above and the decoders below stay bf16.
        crate::wan::attn_fp8::set_ops(self.techniques.fp8_attention.dense, false);
        let denoised = denoise(
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
                // The trace window closes before the step-timing sync.
                crate::wan::gpu_trace::step_before_sync();
                crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
                step_s.push(last.elapsed().as_secs_f64());
                crate::wan::gpu_trace::step_end(step_s[step]);
                crate::wan::log::info(format_args!(
                    "h3 step {}/{steps}: {:.1}s",
                    step + 1,
                    step_s[step]
                ));
                if step + 1 < steps {
                    crate::wan::gpu_trace::step_begin();
                }
                last = Instant::now();
                hooks.step(Stage::Denoise, step + 1, steps, None)
            },
        );
        crate::wan::attn_fp8::set_ops(false, false);
        let (video_rows, audio_rows) = denoised?;
        timings.denoise_s = timer.elapsed().as_secs_f64();
        timings.step_s = step_s;
        // TeaCache rows and the step's AdaLN rows go with the denoise.
        self.model.end_denoise();
        drop((
            vsa,
            sol_policy,
            layout,
            text_refined,
            cond_rows,
            cond_audio_rows,
        ));
        let offload = self
            .residency
            .is_streamed()
            .then(|| self.model.report_offload("denoise"));
        // Streamed: the ring's slots go back before the decoders load.
        self.model.release_offload_device();
        crate::wan::device::trim_pool().map_err(|e| msg(e.to_string()))?;
        memory.mark("denoise")?;

        // --- audio first: the WAV must exist before the muxer starts -------------------
        hooks.stage(Stage::AudioDecode, 0)?;
        let timer = Instant::now();
        let mut transient_booking = crate::wan::ledger::Booking::default();
        let transient_audio = match self.audio_vae {
            Some(_) => None,
            None => Some(transient_booking.track(
                crate::wan::ledger::AUDIO_VAE,
                || load_audio_decoder(&self.root),
                |_| None,
            )?),
        };
        let audio_vae = self
            .audio_vae
            .as_ref()
            .or(transient_audio.as_ref())
            .expect("audio decoder");
        let sample_rate = audio_vae.config().sampling_rate as u32;
        let wave = audio_vae
            .decode_rows(&audio_rows, H3_AUDIO_CHANNELS)?
            .host_cow()?
            .into_owned();
        drop(transient_audio);
        transient_booking.release(crate::wan::ledger::AUDIO_VAE);
        let wav = out_dir.join("audio.wav");
        let interleaved = interleave_audio(&wave, H3_AUDIO_CHANNELS)?;
        write_wav(&wav, &interleaved, H3_AUDIO_CHANNELS as u16, sample_rate)?;
        timings.audio_decode_s = timer.elapsed().as_secs_f64();
        hooks.audio(&crate::sink::AudioPcm {
            sample_rate,
            channels: H3_AUDIO_CHANNELS,
            samples: &interleaved,
        })?;
        drop(interleaved);
        crate::wan::device::trim_pool().map_err(|e| msg(e.to_string()))?;
        memory.mark("audio_decode")?;

        // --- video: chunks go to the writer as they decode -------------------------------
        let latents = unpatchify_rows(
            &video_rows,
            cfg.in_channels,
            geometry.token_grid,
            cfg.patch_size,
        )?;
        let refined_video = self.spark_bridge(&latents)?;
        hooks.stage(Stage::VideoDecode, 0)?;
        let timer = Instant::now();
        let transient_video = match self.video_vae {
            Some(_) => None,
            None => {
                let timer = Instant::now();
                let vae = transient_booking.track(
                    crate::wan::ledger::VAE,
                    || load_video_decoder(&self.root, self.options.taeh3.as_deref()),
                    |_| None,
                )?;
                crate::wan::log::info(format_args!(
                    "h3 video vae: resident for the decode ({:.1}s load)",
                    timer.elapsed().as_secs_f64()
                ));
                Some(vae)
            }
        };
        let video_vae = self
            .video_vae
            .as_ref()
            .or(transient_video.as_ref())
            .expect("video decoder");
        // With a frame sink (serve E2) the same writer taps its frames to it
        // and writes no PNGs.
        let writer = hooks.open_writer(out_dir, H3_FPS as f64, request.mp4, Some(&wav))?;
        // Chunks go to the writer through a drain thread, so the decode
        // never waits on a copy down or on PNG / mp4 encoding.
        let mut drain = FrameDrain::new(writer)?;
        // The sink speaks TensorError; carry the writer's own error out beside it.
        let mut writer_error: Option<PipelineError> = None;
        let mut sink = |offset: usize, frames: &CudaTensor| {
            drain
                .push(offset, frames)
                .and_then(|()| hooks.frames(offset + frames.shape[0]))
                .map_err(|e| {
                    let text = e.to_string();
                    writer_error = Some(e);
                    TensorError::Message(text)
                })
        };
        let decoded = match video_vae {
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
        let (mut writer, split) = drain.finish()?;
        let tail = Instant::now();
        let mp4 = writer.finish_video()?;
        if let Some(sent) = hooks.finish_sink()? {
            if sent != frames {
                return Err(msg(format!("h3: frame sink got {sent} of {frames} frames")));
            }
        }
        timings.video_encode_s = tail.elapsed().as_secs_f64();
        timings.video_decode_s = timer.elapsed().as_secs_f64();
        timings.video_vae_s = split.vae_s;
        timings.video_rgb_s = split.rgb_s;
        timings.video_push_s = split.push_s;
        timings.video_wait_s = split.wait_s;
        crate::wan::log::info(format_args!(
            "h3 video decode + mp4 {:.2}s: vae {:.2}s, rgb+copy {:.2}s, writer push {:.2}s, decode waited {:.2}s, mp4 tail {:.2}s",
            timings.video_decode_s, split.vae_s, split.rgb_s, split.push_s, split.wait_s, timings.video_encode_s
        ));
        drop(transient_video);
        drop(transient_booking);
        crate::wan::device::trim_pool().map_err(|e| msg(e.to_string()))?;
        memory.mark("video_decode")?;
        let timer = Instant::now();
        let (frame_paths, _) = writer.finish()?;
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
            memory: memory.phases,
            dit_residency: self.residency.as_str(),
            offload,
        })
    }
}

/// The official ViT video decoder, or TAEH3 when one is configured.
fn load_video_decoder(root: &Path, taeh3: Option<&Path>) -> Result<VideoDecoder> {
    load_video_decoder_from(root, taeh3, None)
}

/// [`load_video_decoder`] reading `vae/` through `opened` when the caller
/// already opened it (and queued its read-ahead).
fn load_video_decoder_from(
    root: &Path,
    taeh3: Option<&Path>,
    opened: Option<WeightMap>,
) -> Result<VideoDecoder> {
    Ok(match resolve_taeh3(taeh3) {
        Some(path) => {
            let tae = TaeHv::load_from_path(&path, TaeArch::H3)
                .map_err(|e| msg(format!("FASTVIDEO_TAEH3_WEIGHTS={}: {e}", path.display())))?;
            crate::wan::log::info(format_args!("h3 vae=taeh3 ({})", path.display()));
            VideoDecoder::Taeh3(tae)
        }
        None => VideoDecoder::Official(H3VideoDecoder::load(
            H3VideoVaeConfig::fasth3_8step(),
            &match opened {
                Some(m) => m,
                None => WeightMap::open(&root.join("vae"))?,
            },
        )?),
    })
}

fn load_audio_decoder(root: &Path) -> Result<H3AudioDecoder> {
    load_audio_decoder_from(root, None)
}

fn load_audio_decoder_from(root: &Path, opened: Option<WeightMap>) -> Result<H3AudioDecoder> {
    Ok(H3AudioDecoder::load(
        H3AudioVaeConfig::fasth3_8step(),
        &match opened {
            Some(m) => m,
            None => WeightMap::open(&root.join("audio_vae"))?,
        },
    )?)
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
    // Spark's refiner is the *dev* transformer with the distilled LoRA fused at
    // 0.8 (`Sol-H3-Spark/configs/checkpoints.json:78`, `stage2_ops/models.py:76`):
    // point FASTVIDEO_LTX2_DIT at ltx-2.5-22b-dev-transformer-bf16 and
    // FASTVIDEO_LTX2_LORA at ltx-2.5-22b-distilled-lora-450-bf16 (or keep it
    // beside the weights); `refine_joint` refuses anything else.
    let cfg = fastvideo_models::ltx2::ltx2_5_22b_dev();
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
///
/// `resident`: the loaded vision tower and the resident language model; else
/// both are read from `text_root` for this request, the language model at
/// `precision` (the resident encoder's, so the bytes match).
fn encode_request_multimodal(
    text_root: &Path,
    request: &H3Request,
    resident: Option<(&super::text::MultimodalEncoder, &crate::llm::ResidentDecoder)>,
    precision: crate::llm::WeightPrecision,
    abort: &dyn Fn() -> bool,
    resize: ReferenceImageResize,
) -> Result<super::text::TextConditioning> {
    use super::text::{MultimodalLm, VisionImage, VisionVideo};
    use fastvideo_models::h3::presentation::PresentationRef;
    let where_ = if resident.is_some() {
        "resident"
    } else {
        "streamed from the volume"
    };
    let encode = |prompt: &str,
                  images: &[VisionImage<'_>],
                  videos: &[VisionVideo<'_>],
                  refs: Option<&[PresentationRef]>|
     -> Result<super::text::TextConditioning> {
        Ok(match resident {
            Some((mm, decoder)) => {
                mm.encode(MultimodalLm::Resident(decoder), prompt, images, videos, refs)?
            }
            None => super::text::encode_multimodal_streamed(
                text_root,
                precision,
                prompt,
                images,
                videos,
                refs,
                Some(abort),
            )?,
        })
    };

    if request.is_ref2va() {
        let mut images_owned: Vec<(Vec<u8>, usize, usize)> = Vec::new();
        let mut videos_owned: Vec<(Vec<u8>, usize, usize, usize)> = Vec::new();
        let mut refs = Vec::new();
        for spec in &request.references {
            match spec.kind {
                ReferenceKind::Audio => refs.push(PresentationRef::Audio),
                ReferenceKind::Image => {
                    let (rgb, h, w, _, _) = load_reference_image(&spec.path, resize)?;
                    images_owned.push((rgb, h, w));
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
            "h3 ref2va: Qwen-VL multimodal ({} images, {} videos, {})",
            images.len(),
            videos.len(),
            where_
        ));
        return encode(&request.prompt, &images, &videos, Some(&refs));
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
        "h3 fl2va: Qwen-VL multimodal ({} images, {})",
        images.len(),
        where_
    ));
    encode(&request.prompt, &images, &[], None)
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

/// A Ref2VA reference image as FastVideo prepares it (`reference.py`
/// `prepare_reference_image`): RGB8 resized with Lanczos to the reference
/// canvas (2048 short edge, or `match`). The VAE encodes this image and
/// Qwen-VL sees the same one, so its vision-token count follows the canvas.
/// Returns `(rgb, height, width, source width, source height)`.
fn load_reference_image(
    path: &Path,
    resize: ReferenceImageResize,
) -> Result<(Vec<u8>, usize, usize, usize, usize)> {
    let img = image::open(path)
        .map_err(|e| msg(format!("open {}: {e}", path.display())))?
        .into_rgb8();
    let (sw, sh) = (img.width() as usize, img.height() as usize);
    let (out_h, out_w) = resolve_reference_image_size_with(sw, sh, resize).map_err(msg)?;
    let img = if (sw, sh) == (out_w, out_h) {
        img
    } else {
        image::imageops::resize(&img, out_w as u32, out_h as u32, image::imageops::FilterType::Lanczos3)
    };
    Ok((img.into_raw(), out_h, out_w, sw, sh))
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
    // Posterior samples, as FastVideo's Ref2VA encode (`_sample_visual_posterior`).
    let encoder = H3VideoEncoder::load(vae_cfg, &WeightMap::open(&root.join("vae"))?)?
        .with_posterior_sample(super::vae_encoder::KEYFRAME_ENCODE_SEED);
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
                let (rgb, out_h, out_w, sw, sh) = load_reference_image(&spec.path, resize)?;
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
                let z = encoder.encode_rgb8(&rgb, out_h, out_w, true, seed.wrapping_add(31 + i as u64))?;
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
    fn i2v_encoder_names() {
        use super::I2vEncoderChoice as C;
        for (name, want) in [
            ("auto", C::Auto),
            ("", C::Auto),
            ("resident", C::Resident),
            ("stream", C::Stream),
            ("streamed", C::Stream),
        ] {
            assert_eq!(C::parse(name), Ok(want), "{name}");
        }
        assert!(C::parse("vram").is_err());
        assert_eq!(C::default(), C::Auto);
        assert_eq!(C::parse(C::Resident.as_str()), Ok(C::Resident));
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
    fn auto_releases_the_encoder_when_the_denoise_needs_the_memory() {
        const GIB: u64 = 1 << 30;
        // 5 s at 1344x768: 37296 video + 414 audio + ~540 text rows.
        let rows = 37_296 + 414 + 540;
        let reserve = denoise_reserve_bytes(rows);
        assert!(reserve > 40 * GIB && reserve < 42 * GIB);
        // The 96 GB card that OOMed: ~27 GiB free with encoder + DiT + VAEs resident.
        assert!(!keep_auto_encoder(Some(27 * GIB), rows));
        // A 141 GB card keeps it (~70 GiB free after the same loads).
        assert!(keep_auto_encoder(Some(70 * GIB), rows));
        assert!(!keep_auto_encoder(None, rows));
        assert!(denoise_reserve_bytes(2 * rows) > reserve);
    }

    #[test]
    fn bf16_activations_reserve_less_for_the_denoise() {
        const GIB: u64 = 1 << 30;
        let rows = 37_756; // 5 s 1344x768, as logged by the FastH3 768p run
        let f32 = denoise_reserve_bytes_for(rows, false);
        let bf16 = denoise_reserve_bytes_for(rows, true);
        assert!(f32 > 40 * GIB && f32 < 42 * GIB);
        assert!(bf16 > 26 * GIB && bf16 < 28 * GIB);
        // That run had 40.8 GiB free with the FP8 encoder resident: the f32
        // rule released it, the bf16 rule keeps it (~25 GiB measured need).
        assert!(40 * GIB < f32);
        assert!(40 * GIB > bf16);
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
    fn spark_recipe_is_vsa_and_one_gpu_sol_h3_is_dense() {
        use fastvideo_models::h3::sol::recipe_sol_attn_policy;
        let spark = resolve_contract(Path::new("/"), Some("sol-h3-spark")).unwrap();
        assert_eq!(spark.vsa_sparsity, 0.9);
        assert_eq!(spark.vsa_tile_size, 64);
        assert!(!spark.dense);
        let sol = resolve_contract(Path::new("/"), Some("sol-h3")).unwrap();
        assert!(sol.dense);
        assert_eq!(sol.transformer_forwards, 4);
        assert_eq!(sol.vsa_sparsity, 0.0);
        let rtx = resolve_contract(Path::new("/"), Some("sol-h3-rtx")).unwrap();
        assert_eq!(rtx.transformer_forwards, 49);
        let vsa = resolve_contract(Path::new("/"), Some("4step-vsa")).unwrap();
        assert_eq!(vsa.transformer_forwards, 4);
        assert_eq!(vsa.vsa_sparsity, 0.9);
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3"), None, false).unwrap(),
            H3SolAttnPolicy::Off
        );
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3-rtx"), None, false).unwrap(),
            H3SolAttnPolicy::Rtx
        );
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3-spark"), None, false).unwrap(),
            H3SolAttnPolicy::Off
        );
        // Dense sol-h3 (FASTVIDEO_H3_SOL_ATTN unset or `off`) never builds
        // VSA, whose gate the zero-sparsity DiT does not load.
        for recipe in ["sol-h3", "sol-h3-rtx"] {
            let c = resolve_contract(Path::new("/"), Some(recipe)).unwrap();
            assert!(!dit_loads_vsa_gate(&c, None));
            assert!(
                !uses_vsa(H3SolAttnPolicy::Off, false, false, &c),
                "{recipe}"
            );
            assert!(
                !uses_vsa(H3SolAttnPolicy::Off, c.dense, false, &c),
                "{recipe}"
            );
        }
        assert!(uses_vsa(H3SolAttnPolicy::Off, false, false, &spark));
        assert!(!uses_vsa(H3SolAttnPolicy::Spark, false, false, &spark));
        assert!(!uses_vsa(H3SolAttnPolicy::Off, false, true, &spark));
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
        assert!(sol.dense && !spark.dense);
    }
}
