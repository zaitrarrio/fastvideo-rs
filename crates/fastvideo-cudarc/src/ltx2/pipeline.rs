//! LTX-2 text → audio + video: one or two denoise stages, then the decoders.
//!
//! The order of work is dictated by memory and by what the muxer needs:
//!
//! 1. **Text.** Gemma (Gemma-3-12B on 2.0 / 2.3, Gemma-4-12B on 2.5) streams
//!    through the device one layer at a time and is gone before anything else
//!    loads; the connectors are loaded, used once and dropped. What remains is
//!    one context per stream.
//! 2. **Denoise.** The DiT loads, lifts the contexts to the stream widths once,
//!    and runs the sampler. On the diffusers-based lines (2.0 / 2.3) latents and
//!    updates are float32. On LTX-2.5 (`ltx_core`) the latent state is bf16
//!    between updates, as the reference stores it ([`LatentState::Bf16`]):
//!    stage 1 is the ancestral sampler (`DistilledPipeline`), stage 2 three
//!    deterministic Euler updates, both one forward per step and unguided.
//! 3. **Decode, audio first.** The audio VAE and vocoder take milliseconds, so
//!    the WAV exists before the first video frame does — which is what lets
//!    ffmpeg be started with the track as an input and then be fed frames as
//!    the video VAE streams them, instead of muxing in a second pass. LTX-2.5
//!    decodes the video in the reference's blended tiles
//!    ([`VideoDecoder::decode_tiled`]).
//!
//! The diffusers reference round-trips each velocity through `x0` and back
//! before the Euler update even with guidance off (`pipeline_ltx2.py:1466-1467`);
//! in float32 that is the identity up to one rounding and is not reproduced. In
//! bf16 (`ltx_core`) it is not the identity, and [`euler_update`] does it.
//! See docs/ports/ltx2.md §a, §f.

use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_models::ltx2::config::{Ltx2Config, Ltx2ModelVersion, Ltx2TextNorm};
use fastvideo_models::ltx2::tiling::TileSizeConfig;
use fastvideo_models::ltx2::{AncestralOpts, Ltx2Schedule};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::llm::{DecoderConfig, ResidentDecoder};
use crate::wan::offload::{DitOffload, Residency};
use crate::wan::pipeline::{
    frames_to_rgb8, interleave_audio, write_wav, PipelineError, Result, VideoWriter,
};
use crate::wan::tensor::{CudaTensor, TensorError};
use crate::wan::weights::WeightMap;

use super::audio_vae::{conform_audio_time, pack_audio_latent, AudioDecoder, AudioEncoder};
use super::diffusion_decoder::DiffusionDecoder;
use super::keys::Keys;
use super::latent_upsampler::LatentUpsampler;
use super::text::{HiddenStack, PaddedPrompt, TextConnectors};
use super::text_cache::{cache_key, weights_identity, CachedContexts, TextCache};
use super::transformer::{
    pack_video, unpack_video, Ltx2Stage2Attn, Ltx2Transformer, Ltx2VideoAttn, Ropes,
    TextConditioning,
};
use super::vae::VideoDecoder;
use super::vocoder::Vocoder;

fn err(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Kernel launches return when queued; a phase's clock stops only once the
/// device has caught up, or the time lands on whoever synchronizes next.
fn sync() -> Result<()> {
    crate::wan::device::synchronize().map_err(|e| err(format!("device synchronize: {e}")))
}

/// Where the weights are.
#[derive(Debug, Clone)]
pub struct Ltx2Paths {
    /// A diffusers LTX-2 snapshot: `tokenizer/`, `text_encoder/`, `vae/`,
    /// `audio_vae/`, `vocoder/`. `Lightricks/LTX-2` and the distilled
    /// conversion carry identical files for all five.
    pub weights: PathBuf,
    /// The *distilled* DiT and connectors: `ltx-2-19b-distilled.safetensors`,
    /// or a diffusers root holding `transformer/` and `connectors/`.
    pub dit: PathBuf,
    /// Another root for `tokenizer/` and `text_encoder/` — a slim rewrite
    /// ([`super::slim`]). `None`: they are under `weights`.
    pub text: Option<PathBuf>,
}

impl Ltx2Paths {
    fn text_root(&self) -> &Path {
        self.text.as_deref().unwrap_or(&self.weights)
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2Request {
    pub prompt: String,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub frame_rate: f64,
    pub seed: u64,
    pub output_dir: PathBuf,
    /// Also mux `output.mp4` (needs ffmpeg on the PATH).
    pub mp4: bool,
    /// Distilled two-stage: half-res stage-1 → spatial ×2 upsampler → stage-2 refine.
    pub two_stage: bool,
    /// DiffVAE: diffusion video decoder instead of the conv VAE (audio unchanged).
    pub diff_vae: bool,
    /// Empty = unconditional branch uses empty tokenization when CFG is on.
    pub negative_prompt: String,
    /// Video CFG (`uncond + scale * (cond - uncond)`). Distilled stays at 1.0.
    pub guidance_scale: f32,
    /// Audio CFG scale. Distilled stays at 1.0; base often 7.0.
    pub audio_guidance_scale: f32,
    /// Base/dev step count (`None` → 40 for 2.0, 30 for 2.3). Distilled: subset
    /// size (`None` → 8); with `--two-stage`, stage-1 steps (5 → implies refine 2).
    pub num_inference_steps: Option<usize>,
    /// Two-stage refine steps (`None` → 3, or 2 when stage-1 is 5). Only 2 or 3.
    pub refine_steps: Option<usize>,
    /// Stage-2 Sol route: layer 0 dense, layers 1..=47 at tau 1.0 / 1.25 / 1.5.
    /// Requires two-stage with 3 refine steps. Video self-attention on Sol
    /// layers runs the Sol-Attn kernel (`thresh_type=diag`).
    pub sol_stage2: bool,
    /// Stage-2 PISA route: layers 0..=1 dense, later video layers piecewise
    /// sparse at 0.9 / block 64. Same refine length. Sparse layers run the
    /// PISA score-route kernel; the midpoint token prune is not applied.
    pub pisa_stage2: bool,
    /// First-frame image for I2V (`None` = T2AV). Uses VAE encode stub until
    /// the full encoder lands.
    pub image_path: Option<PathBuf>,
}

/// Whether stage 2 runs the Sol route when the caller did not choose. The
/// reference's single-GPU LTX-2.5 distilled profile (`RTX5090/gpu_infer.py`,
/// "Sol-Attn is enabled for Stage 2 video self-attention") always installs
/// `LTX25Stage2SolAttention` for the 3-forward refine, so that is the default
/// here too; `dense` is the control run.
pub fn default_sol_stage2(
    cfg: &Ltx2Config,
    two_stage: bool,
    refine_steps: Option<usize>,
    pisa_stage2: bool,
    dense: bool,
) -> bool {
    !dense
        && !pisa_stage2
        && two_stage
        && cfg.version == Ltx2ModelVersion::V25
        && !cfg.scheduler.use_dynamic_shifting
        && refine_steps.unwrap_or(3) == 3
}

impl Ltx2Request {
    pub fn new(
        cfg: &Ltx2Config,
        prompt: impl Into<String>,
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        let d = &cfg.defaults;
        let base = cfg.scheduler.use_dynamic_shifting;
        Self {
            prompt: prompt.into(),
            height: d.height,
            width: d.width,
            num_frames: d.num_frames,
            frame_rate: d.frame_rate,
            seed: 10,
            output_dir: output_dir.into(),
            mp4: true,
            two_stage: false,
            diff_vae: false,
            negative_prompt: String::new(),
            guidance_scale: if base { 4.0 } else { 1.0 },
            audio_guidance_scale: if base { 7.0 } else { 1.0 },
            num_inference_steps: None,
            refine_steps: None,
            sol_stage2: false,
            pisa_stage2: false,
            image_path: None,
        }
    }

    /// The model card's constraints: H and W divisible by 32 (64 when two-stage),
    /// `8k + 1` frames.
    pub fn validate(&self) -> Result<()> {
        let multiple = if self.two_stage { 64 } else { 32 };
        if self.height == 0
            || self.width == 0
            || !self.height.is_multiple_of(multiple)
            || !self.width.is_multiple_of(multiple)
        {
            return Err(err(format!(
                "ltx2: {}x{} — height and width must be positive multiples of {multiple}{}",
                self.width,
                self.height,
                if self.two_stage { " for two-stage" } else { "" }
            )));
        }
        if self.num_frames % 8 != 1 {
            return Err(err(format!(
                "ltx2: {} frames — the frame count must be 8k + 1",
                self.num_frames
            )));
        }
        if self.frame_rate.is_nan() || self.frame_rate <= 0.0 || self.prompt.trim().is_empty() {
            return Err(err(
                "ltx2: needs a positive frame rate and a non-empty prompt",
            ));
        }
        if let Some(n) = self.refine_steps {
            if n != 2 && n != 3 {
                return Err(err(format!("ltx2: refine_steps must be 2 or 3, got {n}")));
            }
        }
        if self.sol_stage2 && self.pisa_stage2 {
            return Err(err("ltx2 stage-2 Sol and PISA are separate routes"));
        }
        if self.sol_stage2 || self.pisa_stage2 {
            if !self.two_stage {
                return Err(err("ltx2 sol/pisa stage-2 requires --two-stage"));
            }
            if self.stage2_steps() != 3 {
                return Err(err("ltx2 sol/pisa stage-2 is the 3-forward refine"));
            }
        }
        Ok(())
    }

    pub fn stage2_attn(&self) -> Ltx2Stage2Attn {
        if self.sol_stage2 {
            Ltx2Stage2Attn::Sol
        } else if self.pisa_stage2 {
            Ltx2Stage2Attn::Pisa
        } else {
            Ltx2Stage2Attn::Off
        }
    }

    /// Stage-1 distilled step count (defaults to 8).
    pub fn stage1_steps(&self) -> usize {
        self.num_inference_steps.unwrap_or(8)
    }

    /// Stage-2 refine step count: explicit, else 2 when stage-1 is 5, else 3.
    pub fn stage2_steps(&self) -> usize {
        self.refine_steps
            .unwrap_or(if self.stage1_steps() == 5 { 2 } else { 3 })
    }
}

/// Wall-clock seconds per phase. `load_s` is the pipeline's one-off load; the
/// rest are what each phase of this generation took (text includes loading the
/// connectors when the text path is streamed).
#[derive(Debug, Clone, Default)]
pub struct Ltx2Timings {
    pub load_s: f64,
    pub text_s: f64,
    pub denoise_s: f64,
    pub stage1_s: f64,
    pub upsample_s: f64,
    pub stage2_s: f64,
    pub step_s: Vec<f64>,
    pub decode_audio_s: f64,
    pub decode_video_s: f64,
    /// Waiting for the frame writer / ffmpeg after the last frame was decoded.
    pub write_s: f64,
}

#[derive(Debug, Clone)]
pub struct Ltx2Output {
    pub frames: Vec<String>,
    pub mp4: Option<String>,
    pub wav: String,
    pub prompt_tokens: usize,
    pub video_tokens: usize,
    pub audio_tokens: usize,
    pub text: TextReport,
    pub timings: Ltx2Timings,
    /// Device-memory peaks per phase, in order (empty without a device).
    pub memory: Vec<PhaseMemory>,
    /// `"resident"` or `"streamed"` DiT blocks.
    pub dit_residency: &'static str,
}

/// One seeded Gaussian stream, standing in for one `torch.Generator`.
///
/// **Choice.** The references draw with torch's CUDA Philox generator
/// (`torch.randn(..., generator=g)`), whose bits depend on the launch geometry
/// torch picks for the device; no stream here can reproduce them, host or
/// device. What *can* be reproduced is the contract: which generator each
/// draw comes from, its seed, the order of draws, their shapes and their
/// dtype. So the draws stay on a seeded host generator (`StdRng`, portable and
/// testable without a GPU), are rounded to bf16 when the reference draws in
/// bf16, and each call — one per sampler step, video and audio together — is
/// written into one pinned buffer and uploaded with a single copy.
pub struct NoiseStream {
    rng: rand::rngs::StdRng,
    bf16: bool,
    #[cfg(feature = "cuda")]
    pinned: Option<cudarc::driver::PinnedHostSlice<f32>>,
}

impl NoiseStream {
    /// `torch.Generator().manual_seed(seed)`; `bf16` rounds every draw the way
    /// `torch.randn(..., dtype=torch.bfloat16)` stores it.
    pub fn new(seed: u64, bf16: bool) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            bf16,
            #[cfg(feature = "cuda")]
            pinned: None,
        }
    }

    fn fill(rng: &mut rand::rngs::StdRng, bf16: bool, out: &mut [f32]) {
        for v in out {
            let x = rng.sample::<f32, _>(StandardNormal);
            *v = if bf16 {
                fastvideo_models::ltx2::schedule::bf16_round(x)
            } else {
                x
            };
        }
    }

    /// One `N(0, 1)` tensor per shape, drawn in order from the stream.
    pub fn draw(&mut self, shapes: &[&[usize]]) -> Result<Vec<CudaTensor>> {
        let sizes: Vec<usize> = shapes.iter().map(|s| s.iter().product()).collect();
        let total: usize = sizes.iter().sum();
        let split = |flat: CudaTensor| -> Result<Vec<CudaTensor>> {
            let mut at = 0usize;
            shapes
                .iter()
                .zip(&sizes)
                .map(|(shape, &n)| {
                    let t = flat.narrow(0, at, n)?.reshape(shape.to_vec())?;
                    at += n;
                    Ok(t)
                })
                .collect()
        };
        #[cfg(feature = "cuda")]
        if total > 0 && crate::wan::stats::device_expected() {
            if let Some(dev) = crate::wan::device::global_device() {
                if self.ensure_pinned(&dev, total) {
                    let flat = self
                        .upload_pinned(&dev, total)
                        .map_err(|e| err(format!("ltx2 noise: upload of {total} draws: {e}")))?;
                    return split(CudaTensor::from_device_slice(flat, vec![total])?);
                }
            }
        }
        let mut host = vec![0f32; total];
        Self::fill(&mut self.rng, self.bf16, &mut host);
        split(CudaTensor::from_vec(host, vec![total])?)
    }

    /// A pinned host buffer of `total` floats; `false` (pageable fallback,
    /// logged) when the driver will not pin one. Draws nothing.
    #[cfg(feature = "cuda")]
    fn ensure_pinned(
        &mut self,
        dev: &std::sync::Arc<crate::wan::device::DeviceContext>,
        total: usize,
    ) -> bool {
        if self.pinned.as_ref().map(|p| p.len()) == Some(total) {
            return true;
        }
        self.pinned = None;
        // Every element is written before the copy reads it.
        match unsafe { dev.ctx.alloc_pinned::<f32>(total) } {
            Ok(p) => {
                self.pinned = Some(p);
                true
            }
            Err(e) => {
                crate::wan::log::info(format_args!(
                    "ltx2 noise: no pinned buffer of {total} floats ({e}); pageable upload"
                ));
                false
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn upload_pinned(
        &mut self,
        dev: &std::sync::Arc<crate::wan::device::DeviceContext>,
        total: usize,
    ) -> std::result::Result<cudarc::driver::CudaSlice<f32>, cudarc::driver::DriverError> {
        let pinned = self.pinned.as_mut().expect("pinned noise buffer");
        // Waits for the previous upload from this buffer before overwriting it.
        Self::fill(&mut self.rng, self.bf16, pinned.as_mut_slice()?);
        let mut flat = unsafe { dev.stream.alloc::<f32>(total) }?;
        dev.stream.memcpy_htod(&*pinned, &mut flat)?;
        crate::wan::stats::record_h2d(total);
        Ok(flat)
    }
}

/// How the sampler stores its latent state between updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentState {
    /// float32 throughout (the diffusers-based LTX-2.0 / 2.3 lines).
    F32,
    /// `ltx_core` (LTX-2.5): the state is a bf16 tensor; each update is
    /// computed in float32 and stored back as bf16
    /// (`diffusion_steps.py:39,106`, `samplers.py:546,558`), the model's
    /// velocity and `x0` are bf16 (`utils.py:36,52`), and noise is drawn in bf16
    /// (`noisers.py:21-27`, `samplers.py:155-157`).
    Bf16,
}

impl LatentState {
    pub fn for_version(version: Ltx2ModelVersion) -> Self {
        if version == Ltx2ModelVersion::V25 {
            Self::Bf16
        } else {
            Self::F32
        }
    }

    /// Round to the state's dtype (a no-op for f32).
    pub fn store(self, t: CudaTensor) -> Result<CudaTensor> {
        match self {
            Self::F32 => Ok(t),
            Self::Bf16 => Ok(t.quantize_bf16()?.to_f32_act()?),
        }
    }
}

/// One deterministic Euler update from `σ_i` to `σ_{i+1}`.
///
/// `F32`: `x + (σ_{i+1} − σ_i)·v`. `Bf16`: the `ltx_core` round trip —
/// `X0Model` turns the (bf16) velocity into `x0 = bf16(x − σ·v)`
/// (`to_denoised`, `ltx_core/utils.py:39-52`), `EulerDiffusionStep` turns it
/// back into `v' = bf16((x − x0)/σ)` (`to_velocity`, `utils.py:21-36`) and
/// stores `bf16(x + v'·dt)` with `dt` the float32 sigma difference
/// (`diffusion_steps.py:31-39`). [`fastvideo_models::ltx2::schedule::ltx_core_euler_step`]
/// is the element-wise host reference.
pub fn euler_update(
    x: &CudaTensor,
    v: &CudaTensor,
    schedule: &Ltx2Schedule,
    i: usize,
    state: LatentState,
) -> Result<CudaTensor> {
    match state {
        LatentState::F32 => Ok(CudaTensor::lincomb(&[
            (1.0, x),
            (schedule.dt(i) as f32, v),
        ])?),
        LatentState::Bf16 => {
            let (sigma, next) = (schedule.sigmas[i] as f32, schedule.sigmas[i + 1] as f32);
            let v = state.store(v.clone())?;
            let x0 = state.store(CudaTensor::lincomb(&[(1.0, x), (-sigma, &v)])?)?;
            let v = state.store(
                CudaTensor::lincomb(&[(1.0, x), (-1.0, &x0)])?.try_mul_scalar(1.0 / sigma)?,
            )?;
            state.store(CudaTensor::lincomb(&[(1.0, x), (next - sigma, &v)])?)
        }
    }
}

/// One `EulerAncestralDiffusionStep` (`diffusion_steps.py:67-106`) with float32
/// coefficients ([`fastvideo_models::ltx2::schedule::ancestral_coeffs_f32`]):
/// `x0 = x − σ·v`, then `r·x + (1−r)·x0`, then `(α'/α_down)·x + s·c·ε`; the
/// terminal step returns `x0`. `noise` must be given when `eta > 0` and
/// `σ_{i+1} > 0`.
pub fn ancestral_update(
    x: &CudaTensor,
    v: &CudaTensor,
    sigma: f64,
    sigma_next: f64,
    opts: AncestralOpts,
    noise: Option<&CudaTensor>,
    state: LatentState,
) -> Result<CudaTensor> {
    let v = state.store(v.clone())?;
    let x0 = state.store(CudaTensor::lincomb(&[(1.0, x), (-(sigma as f32), &v)])?)?;
    let Some(c) = fastvideo_models::ltx2::schedule::ancestral_coeffs_f32(
        sigma as f32,
        sigma_next as f32,
        opts.eta as f32,
        opts.s_noise as f32,
    ) else {
        return Ok(x0);
    };
    let stepped = CudaTensor::lincomb(&[(c.sample, x), (c.denoised, &x0)])?;
    if opts.eta <= 0.0 {
        return state.store(stepped);
    }
    let noise = noise.ok_or_else(|| err("ltx2 ancestral: eta > 0 needs a noise draw"))?;
    state.store(CudaTensor::lincomb(&[
        (c.factor, &stepped),
        (c.noise, noise),
    ])?)
}

/// `GaussianNoiser` at `noise_scale = σ` (`ltx_core/components/noisers.py:29-37`):
/// `torch.lerp(x, ε, σ)` in float32 — for `σ ≥ 0.5` ATen evaluates
/// `ε − (ε − x)·(1 − σ)` — then the full-denoise mask (a no-op) and the store.
pub fn renoise(
    x: &CudaTensor,
    noise: &CudaTensor,
    sigma: f32,
    state: LatentState,
) -> Result<CudaTensor> {
    let out = if sigma >= 0.5 {
        let diff = CudaTensor::lincomb(&[(1.0, noise), (-1.0, x)])?;
        CudaTensor::lincomb(&[(1.0, noise), (-(1.0 - sigma), &diff)])?
    } else {
        let diff = CudaTensor::lincomb(&[(1.0, noise), (-1.0, x)])?;
        CudaTensor::lincomb(&[(1.0, x), (sigma, &diff)])?
    };
    state.store(out)
}

/// Seeded `N(0, 1)` latents, already packed for the DiT: video
/// `[1, F·H·W, 128]` drawn first, then audio `[1, L, 128]`, from one stream —
/// the order every reference pipeline draws them in.
///
/// `ltx_core` (LTX-2.5, [`LatentState::Bf16`]) draws in the *patchified*
/// shapes (`GaussianNoiser` on the patchified state, `blocks.py:214-230`), so
/// token-major. The diffusers pipelines (2.0 / 2.3) draw `[1, C, F, H, W]` and
/// `[1, C, L, M]` and pack. The stream keeps going: LTX-2.5 draws its stage-2
/// renoise from the same generator (`distilled.py:217-218,294-313`).
pub fn initial_noise(
    cfg: &Ltx2Config,
    grid: [usize; 3],
    audio_tokens: usize,
    noise: &mut NoiseStream,
) -> Result<(CudaTensor, CudaTensor)> {
    let c = cfg.transformer.in_channels;
    let [f, h, w] = grid;
    let (ac, bins) = (
        cfg.audio_vae.latent_channels,
        cfg.audio_vae.latent_mel_bins(),
    );
    if LatentState::for_version(cfg.version) == LatentState::Bf16 {
        let mut draws = noise.draw(&[&[1, f * h * w, c], &[1, audio_tokens, ac * bins]])?;
        let audio = draws.pop().expect("audio noise");
        let video = draws.pop().expect("video noise");
        return Ok((video, audio));
    }
    let mut draws = noise.draw(&[&[1, c, f, h, w], &[1, ac, audio_tokens, bins]])?;
    let audio = draws.pop().expect("audio noise");
    let video = draws.pop().expect("video noise");
    // [1, C, L, M] → [1, L, C·M]: feature index = channel · bins + bin.
    let audio = audio
        .permute(&[0, 2, 1, 3])?
        .reshape(vec![1, audio_tokens, ac * bins])?;
    Ok((pack_video(&video)?, audio))
}

/// Called after each step with `(step, video, audio, seconds)`.
pub type StepObserver<'a> = &'a mut dyn FnMut(usize, &CudaTensor, &CudaTensor, f64) -> Result<()>;

/// The distilled Euler loop: `x ← x + (σ_{i+1} - σ_i) · v(x, 1000·σ_i)` for both
/// streams with one forward per step, float32 state. Inputs are packed latents
/// at `σ_0`.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    model: &Ltx2Transformer,
    text: &TextConditioning,
    ropes: &Ropes,
    schedule: &Ltx2Schedule,
    video: CudaTensor,
    audio: CudaTensor,
    observer: Option<StepObserver<'_>>,
    stage2: Ltx2Stage2Attn,
) -> Result<(CudaTensor, CudaTensor)> {
    denoise_with(
        model,
        text,
        ropes,
        schedule,
        video,
        audio,
        observer,
        stage2,
        LatentState::F32,
    )
}

/// [`denoise`] with the state's dtype: one unguided forward per step
/// (`SimpleDenoiser`, `ltx_pipelines/utils/denoisers.py`) and [`euler_update`]
/// (`euler_denoising_loop`, `samplers.py:39-81`). This is LTX-2.5's stage 2
/// and the refiners' loop.
#[allow(clippy::too_many_arguments)]
pub fn denoise_with(
    model: &Ltx2Transformer,
    text: &TextConditioning,
    ropes: &Ropes,
    schedule: &Ltx2Schedule,
    mut video: CudaTensor,
    mut audio: CudaTensor,
    mut observer: Option<StepObserver<'_>>,
    stage2: Ltx2Stage2Attn,
    state: LatentState,
) -> Result<(CudaTensor, CudaTensor)> {
    for i in 0..schedule.num_steps() {
        let timer = Instant::now();
        model.begin_fbcache_step(i);
        model.arm_prune_step(i);
        let (v_video, v_audio) = if let Some(cached) = model.stage1_reuse(i) {
            cached
        } else {
            let out = model.forward_sol(
                &video,
                &audio,
                text,
                schedule.timestep_f32(i),
                ropes,
                None,
                stage2.at(i),
            )?;
            model.stage1_store(&out.0, &out.1);
            out
        };
        video = euler_update(&video, &v_video, schedule, i, state)?;
        audio = euler_update(&audio, &v_audio, schedule, i, state)?;
        sync()?;
        let secs = timer.elapsed().as_secs_f64();
        crate::wan::log::info(format_args!(
            "ltx2 step {}/{} sigma {:.6} ({secs:.2}s)",
            i + 1,
            schedule.num_steps(),
            schedule.sigmas[i]
        ));
        if let Some(obs) = observer.as_mut() {
            obs(i, &video, &audio, secs)?;
        }
    }
    Ok((video, audio))
}

/// Base/dev CFG Euler: two forwards per step, then
/// `v = uncond + scale * (cond - uncond)` separately for video and audio.
pub fn denoise_cfg(
    model: &Ltx2Transformer,
    text_cond: &TextConditioning,
    text_uncond: &TextConditioning,
    ropes: &Ropes,
    schedule: &Ltx2Schedule,
    video_scale: f32,
    audio_scale: f32,
    mut video: CudaTensor,
    mut audio: CudaTensor,
    mut observer: Option<StepObserver<'_>>,
    stage2: Ltx2Stage2Attn,
) -> Result<(CudaTensor, CudaTensor)> {
    for i in 0..schedule.num_steps() {
        let timer = Instant::now();
        let t = schedule.timestep_f32(i);
        let route = stage2.at(i);
        model.begin_fbcache_step(i);
        model.arm_prune_step(i);
        let (v_video, v_audio) = if let Some(cached) = model.stage1_reuse(i) {
            cached
        } else {
            let (vc, ac) = model.forward_sol(&video, &audio, text_cond, t, ropes, None, route)?;
            let (vu, au) = model.forward_sol(&video, &audio, text_uncond, t, ropes, None, route)?;
            let v_video = CudaTensor::lincomb(&[(video_scale, &vc), (1.0 - video_scale, &vu)])?;
            let v_audio = CudaTensor::lincomb(&[(audio_scale, &ac), (1.0 - audio_scale, &au)])?;
            model.stage1_store(&v_video, &v_audio);
            (v_video, v_audio)
        };
        let dt = schedule.dt(i) as f32;
        video = CudaTensor::lincomb(&[(1.0, &video), (dt, &v_video)])?;
        audio = CudaTensor::lincomb(&[(1.0, &audio), (dt, &v_audio)])?;
        sync()?;
        let secs = timer.elapsed().as_secs_f64();
        crate::wan::log::info(format_args!(
            "ltx2 cfg step {}/{} sigma {:.6} v_gs={video_scale} a_gs={audio_scale} ({secs:.2}s)",
            i + 1,
            schedule.num_steps(),
            schedule.sigmas[i]
        ));
        if let Some(obs) = observer.as_mut() {
            obs(i, &video, &audio, secs)?;
        }
    }
    Ok((video, audio))
}

fn velocity_call(
    model: &Ltx2Transformer,
    text: &TextConditioning,
    text_uncond: Option<&TextConditioning>,
    ropes: &Ropes,
    video: &CudaTensor,
    audio: &CudaTensor,
    t: f32,
    video_scale: f32,
    audio_scale: f32,
    route: Ltx2VideoAttn,
    call: usize,
) -> Result<(CudaTensor, CudaTensor)> {
    if let Some(cached) = model.stage1_reuse(call) {
        return Ok(cached);
    }
    let out = if let Some(uncond) = text_uncond {
        let (vc, ac) = model.forward_sol(video, audio, text, t, ropes, None, route)?;
        let (vu, au) = model.forward_sol(video, audio, uncond, t, ropes, None, route)?;
        (
            CudaTensor::lincomb(&[(video_scale, &vc), (1.0 - video_scale, &vu)])?,
            CudaTensor::lincomb(&[(audio_scale, &ac), (1.0 - audio_scale, &au)])?,
        )
    } else {
        model.forward_sol(video, audio, text, t, ropes, None, route)?
    };
    model.stage1_store(&out.0, &out.1);
    Ok(out)
}

fn denoised_from_velocity(
    sample: &CudaTensor,
    velocity: &CudaTensor,
    sigma: f64,
) -> Result<CudaTensor> {
    Ok(CudaTensor::lincomb(&[
        (1.0, sample),
        (-(sigma as f32), velocity),
    ])?)
}

fn res2s_midpoint(
    anchor: &CudaTensor,
    denoised: &CudaTensor,
    h: f64,
    a21: f64,
) -> Result<CudaTensor> {
    let w = (h * a21) as f32;
    Ok(CudaTensor::lincomb(&[(1.0 - w, anchor), (w, denoised)])?)
}

fn res2s_combine(
    anchor: &CudaTensor,
    d1: &CudaTensor,
    d2: &CudaTensor,
    h: f64,
    b1: f64,
    b2: f64,
) -> Result<CudaTensor> {
    let w1 = (h * b1) as f32;
    let w2 = (h * b2) as f32;
    Ok(CudaTensor::lincomb(&[
        (1.0 - w1 - w2, anchor),
        (w1, d1),
        (w2, d2),
    ])?)
}

/// LTX-2.3 ODE res2s (no SDE, no bongmath). Two model calls per step except
/// the last, which snaps to x0 — 15 steps → 29 calls so SCSP 16–28 is live.
pub fn denoise_res2s(
    model: &Ltx2Transformer,
    text: &TextConditioning,
    text_uncond: Option<&TextConditioning>,
    ropes: &Ropes,
    schedule: &Ltx2Schedule,
    video_scale: f32,
    audio_scale: f32,
    mut video: CudaTensor,
    mut audio: CudaTensor,
    mut observer: Option<StepObserver<'_>>,
    stage2: Ltx2Stage2Attn,
) -> Result<(CudaTensor, CudaTensor)> {
    let mut call = 0usize;
    for i in 0..schedule.num_steps() {
        let timer = Instant::now();
        let sigma = schedule.sigmas[i];
        let sigma_next = schedule.sigmas[i + 1];
        let route = stage2.at(i);
        model.begin_fbcache_step(i);
        model.arm_prune_step(i);
        let t = schedule.timestep_f32(i);
        let (v_video, v_audio) = velocity_call(
            model,
            text,
            text_uncond,
            ropes,
            &video,
            &audio,
            t,
            video_scale,
            audio_scale,
            route,
            call,
        )?;
        call += 1;
        let d_video = denoised_from_velocity(&video, &v_video, sigma)?;
        let d_audio = denoised_from_velocity(&audio, &v_audio, sigma)?;
        if sigma_next == 0.0 || i + 1 == schedule.num_steps() {
            video = d_video;
            audio = d_audio;
        } else {
            let h = -(sigma_next / sigma).ln();
            let (a21, b1, b2) = fastvideo_models::ltx2::hq::res2s_coefficients(
                h,
                fastvideo_models::ltx2::hq::RES2S_C2,
            );
            let sub_sigma = (sigma * sigma_next).sqrt();
            let mid_v = res2s_midpoint(&video, &d_video, h, a21)?;
            let mid_a = res2s_midpoint(&audio, &d_audio, h, a21)?;
            let t_mid = (sub_sigma as f32) * (schedule.num_train_timesteps as f32);
            let (v2_v, v2_a) = velocity_call(
                model,
                text,
                text_uncond,
                ropes,
                &mid_v,
                &mid_a,
                t_mid,
                video_scale,
                audio_scale,
                route,
                call,
            )?;
            call += 1;
            let d2_v = denoised_from_velocity(&mid_v, &v2_v, sub_sigma)?;
            let d2_a = denoised_from_velocity(&mid_a, &v2_a, sub_sigma)?;
            video = res2s_combine(&video, &d_video, &d2_v, h, b1, b2)?;
            audio = res2s_combine(&audio, &d_audio, &d2_a, h, b1, b2)?;
        }
        sync()?;
        let secs = timer.elapsed().as_secs_f64();
        crate::wan::log::info(format_args!(
            "ltx2 res2s step {}/{} sigma {:.6} calls {call} ({secs:.2}s)",
            i + 1,
            schedule.num_steps(),
            sigma
        ));
        if let Some(obs) = observer.as_mut() {
            obs(i, &video, &audio, secs)?;
        }
    }
    Ok((video, audio))
}

/// Distilled ancestral loop (LTX-2.5 stage 1: `euler_ancestral_denoising_loop`,
/// `ltx_pipelines/utils/samplers.py:488-563`): one unguided forward per step,
/// then [`ancestral_update`] on each stream. The loop's noise has a generator of
/// its own, seeded `opts.noise_seed` (`seed + 10000`, `distilled.py:62-85,
/// 168-185`), drawn video then audio on every non-terminal step, in the
/// state's dtype.
#[allow(clippy::too_many_arguments)]
pub fn denoise_ancestral(
    model: &Ltx2Transformer,
    text: &TextConditioning,
    ropes: &Ropes,
    schedule: &Ltx2Schedule,
    mut video: CudaTensor,
    mut audio: CudaTensor,
    opts: AncestralOpts,
    mut observer: Option<StepObserver<'_>>,
    stage2: Ltx2Stage2Attn,
    state: LatentState,
) -> Result<(CudaTensor, CudaTensor)> {
    let mut noise = NoiseStream::new(opts.noise_seed, state == LatentState::Bf16);
    for i in 0..schedule.num_steps() {
        let timer = Instant::now();
        let sigma = schedule.sigmas[i];
        let sigma_next = schedule.sigmas[i + 1];
        model.begin_fbcache_step(i);
        model.arm_prune_step(i);
        let (v_video, v_audio) = if let Some(cached) = model.stage1_reuse(i) {
            cached
        } else {
            let out = model.forward_sol(
                &video,
                &audio,
                text,
                schedule.timestep_f32(i),
                ropes,
                None,
                stage2.at(i),
            )?;
            model.stage1_store(&out.0, &out.1);
            out
        };
        // One draw (one upload) per step: video, then audio.
        let draws = if opts.eta > 0.0 && sigma_next != 0.0 {
            noise.draw(&[&video.shape, &audio.shape])?
        } else {
            Vec::new()
        };
        video = ancestral_update(
            &video,
            &v_video,
            sigma,
            sigma_next,
            opts,
            draws.first(),
            state,
        )?;
        audio = ancestral_update(
            &audio,
            &v_audio,
            sigma,
            sigma_next,
            opts,
            draws.get(1),
            state,
        )?;
        sync()?;
        let secs = timer.elapsed().as_secs_f64();
        crate::wan::log::info(format_args!(
            "ltx2 ancestral step {}/{} sigma {:.6} ({secs:.2}s)",
            i + 1,
            schedule.num_steps(),
            sigma
        ));
        if let Some(obs) = observer.as_mut() {
            obs(i, &video, &audio, secs)?;
        }
    }
    Ok((video, audio))
}

/// The three decoders of the output side, plus the optional spatial upsampler.
pub struct Decoders {
    pub video: VideoDecoder,
    pub audio: AudioDecoder,
    pub vocoder: Vocoder,
    pub upsampler: Option<LatentUpsampler>,
}

impl Decoders {
    pub fn load(weights: &Path, cfg: &Ltx2Config) -> Result<Self> {
        let mut decoders = Self::load_without_upsampler(weights, cfg)?;
        decoders.upsampler = load_upsampler(weights, cfg)?;
        Ok(decoders)
    }

    /// The video VAE, audio VAE and vocoder; `upsampler` is left `None` for a
    /// caller that loads it only around its one call ([`load_upsampler`]).
    pub fn load_without_upsampler(weights: &Path, cfg: &Ltx2Config) -> Result<Self> {
        let open = |sub: &str| WeightMap::open(&weights.join(sub));
        Ok(Self {
            video: VideoDecoder::load(&open("vae")?, &cfg.vae)?,
            audio: AudioDecoder::load(&open("audio_vae")?, &cfg.audio_vae)?,
            vocoder: Vocoder::load(&open("vocoder")?, &cfg.vocoder)?,
            upsampler: None,
        })
    }
}

/// The spatial upsampler's folder under `weights`, when the pack has one.
pub fn upsampler_dir(weights: &Path) -> Option<PathBuf> {
    ["latent_upsampler", "spatial_upscaler", "spatial_upsampler"]
        .iter()
        .map(|name| weights.join(name))
        .find(|p| p.is_dir())
}

/// The spatial x2 upsampler, or `None` when the config or the pack has none.
pub fn load_upsampler(weights: &Path, cfg: &Ltx2Config) -> Result<Option<LatentUpsampler>> {
    match (&cfg.latent_upsampler, upsampler_dir(weights)) {
        (Some(ucfg), Some(dir)) => Ok(Some(LatentUpsampler::load(&WeightMap::open(&dir)?, ucfg)?)),
        _ => Ok(None),
    }
}

pub use crate::wan::offload::{MemoryLog, PhaseMemory};

/// Hand what the last phase freed back to the driver, as the reference's
/// blocks do on exit (`AllocatorTrimStrategy.TRIM`: sync + `empty_cache`).
fn trim() -> Result<()> {
    crate::wan::device::trim_pool().map_err(|e| err(format!("device pool trim: {e}")))
}

/// What [`decode_and_write`] left on disk, and how long each part took.
#[derive(Debug, Clone)]
pub struct Written {
    pub frames: Vec<String>,
    pub mp4: Option<String>,
    pub wav: String,
    pub decode_audio_s: f64,
    pub decode_video_s: f64,
    pub write_s: f64,
}

/// Final packed latents → `audio.wav`, `frame-NNN.png` and (when `mp4`)
/// `output.mp4` in `dir`. Audio is decoded and written first so the muxer can
/// take it as an input while frames are still arriving. `tiling`: the
/// reference's blended tile decode ([`VideoDecoder::decode_tiled`]); `None`
/// decodes the whole clip exactly, streamed in time.
#[allow(clippy::too_many_arguments)]
pub fn decode_and_write(
    dec: &Decoders,
    video: &CudaTensor,
    audio: &CudaTensor,
    grid: [usize; 3],
    dir: &Path,
    frame_rate: f64,
    mp4: bool,
    tiling: Option<&TileSizeConfig>,
) -> Result<Written> {
    std::fs::create_dir_all(dir).map_err(|e| err(format!("{}: {e}", dir.display())))?;
    let timer = Instant::now();
    let wave = dec.vocoder.forward(&dec.audio.decode_packed(audio)?)?;
    let channels = wave.shape[1];
    let wav = dir.join("audio.wav");
    let rate = u32::try_from(dec.vocoder.sample_rate())
        .map_err(|_| err("ltx2: vocoder sample rate out of range"))?;
    let channel_count =
        u16::try_from(channels).map_err(|_| err("ltx2: too many audio channels"))?;
    write_wav(
        &wav,
        &interleave_audio(&wave.host_cow()?, channels)?,
        channel_count,
        rate,
    )?;
    let decode_audio_s = timer.elapsed().as_secs_f64();

    let timer = Instant::now();
    let fps = frame_rate.round().max(1.0) as u32;
    let mut writer = VideoWriter::spawn_with_audio(dir, fps, mp4, Some(&wav))?;
    // The decoder's sink speaks tensor errors; carry the writer's own across.
    let mut sink_err: Option<PipelineError> = None;
    let mut sink = |offset: usize, frames: &CudaTensor| -> std::result::Result<(), TensorError> {
        let (h, w) = (frames.shape[2], frames.shape[3]);
        match frames_to_rgb8(frames).and_then(|rgb| writer.push(offset, h, w, rgb)) {
            Ok(()) => Ok(()),
            Err(e) => {
                let text = e.to_string();
                sink_err = Some(e);
                Err(TensorError::Message(text))
            }
        }
    };
    let latents = unpack_video(video, grid)?;
    let decoded = match tiling {
        Some(tiles) => dec.video.decode_tiled(&latents, tiles, &mut sink),
        None => dec.video.decode_streaming(&latents, &mut sink),
    };
    match (decoded, sink_err) {
        (_, Some(e)) => return Err(e),
        (Err(e), None) => return Err(e.into()),
        (Ok(_), None) => {}
    }
    let decode_video_s = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let (frames, mp4_path) = writer.finish()?;
    Ok(Written {
        frames,
        mp4: mp4_path,
        wav: wav.to_string_lossy().into_owned(),
        decode_audio_s,
        decode_video_s,
        write_s: timer.elapsed().as_secs_f64(),
    })
}

/// Like [`decode_and_write`], but video goes through DiffVAE (de-norm → 1-step x0).
pub fn decode_diffvae_and_write(
    dec: &Decoders,
    diffvae: &DiffusionDecoder,
    video: &CudaTensor,
    audio: &CudaTensor,
    grid: [usize; 3],
    dir: &Path,
    frame_rate: f64,
    mp4: bool,
    seed: u64,
) -> Result<Written> {
    std::fs::create_dir_all(dir).map_err(|e| err(format!("{}: {e}", dir.display())))?;
    let timer = Instant::now();
    let wave = dec.vocoder.forward(&dec.audio.decode_packed(audio)?)?;
    let channels = wave.shape[1];
    let wav = dir.join("audio.wav");
    let rate = u32::try_from(dec.vocoder.sample_rate())
        .map_err(|_| err("ltx2: vocoder sample rate out of range"))?;
    let channel_count =
        u16::try_from(channels).map_err(|_| err("ltx2: too many audio channels"))?;
    write_wav(
        &wav,
        &interleave_audio(&wave.host_cow()?, channels)?,
        channel_count,
        rate,
    )?;
    let decode_audio_s = timer.elapsed().as_secs_f64();

    let timer = Instant::now();
    let unpacked = unpack_video(video, grid)?;
    let denorm = dec.video.denormalize(&unpacked)?;
    let rgb = diffvae.decode(&denorm, seed)?; // [1, 3, F, H, W]
    let [_, _, f, h, w] = match rgb.shape[..] {
        [1, 3, f, h, w] => [1, 3, f, h, w],
        _ => {
            return Err(err(format!(
                "DiffVAE expected [1,3,F,H,W], got {:?}",
                rgb.shape
            )))
        }
    };
    // [1,3,F,H,W] → [F,3,H,W] for the writer.
    let frames_nchw = rgb.permute(&[0, 2, 1, 3, 4])?.reshape(vec![f, 3, h, w])?;
    let fps = frame_rate.round().max(1.0) as u32;
    let mut writer = VideoWriter::spawn_with_audio(dir, fps, mp4, Some(&wav))?;
    // Chunk frames to bound peak host RGB buffers.
    const CHUNK: usize = 8;
    let mut offset = 0usize;
    while offset < f {
        let n = (f - offset).min(CHUNK);
        let chunk = frames_nchw.narrow(0, offset, n)?;
        let rgb8 = frames_to_rgb8(&chunk)?;
        writer.push(offset, h, w, rgb8)?;
        offset += n;
    }
    let decode_video_s = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let (frames, mp4_path) = writer.finish()?;
    Ok(Written {
        frames,
        mp4: mp4_path,
        wav: wav.to_string_lossy().into_owned(),
        decode_audio_s,
        decode_video_s,
        write_s: timer.elapsed().as_secs_f64(),
    })
}

/// The distilled DiT/connectors: the single file, or `component` under a
/// diffusers root (or the component folder itself).
///
/// Also accepts `--dit …/transformer` when looking up `connectors`: the sibling
/// `…/connectors` (or `…/text_embedding_projection` on LTX-2.3 packs) is used.
pub fn open_distilled(path: &Path, component: &str) -> Result<WeightMap> {
    let aliases: &[&str] = if component == "connectors" {
        &["connectors", "text_embedding_projection"]
    } else {
        &[component]
    };
    for name in aliases {
        if path.is_file() {
            return Ok(WeightMap::open_files(&[path.to_path_buf()])?);
        }
        if path.join(name).is_dir() {
            return Ok(WeightMap::open(&path.join(name))?);
        }
        if let Some(sibling) = path.parent().map(|p| p.join(name)).filter(|p| p.is_dir()) {
            return Ok(WeightMap::open(&sibling)?);
        }
    }
    if path.is_dir() && component != "connectors" {
        return Ok(WeightMap::open(path)?);
    }
    Err(err(format!(
        "{}: no {component} (tried {})",
        path.display(),
        aliases.join(", ")
    )))
}

/// Whether the conditioning cache answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOutcome {
    /// No cache directory was given, or this call bypassed it.
    Off,
    Hit,
    Miss,
}

impl CacheOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Hit => "hit",
            Self::Miss => "miss",
        }
    }
}

/// What the text phase of one generation did.
#[derive(Debug, Clone)]
pub struct TextReport {
    pub cache: CacheOutcome,
    /// `"cache"` when nothing was computed, else how Gemma ran: `"streamed"` or
    /// `"resident"` (whose first use includes loading it).
    pub mode: &'static str,
    pub tokens: usize,
    pub seconds: f64,
    pub key: Option<String>,
}

/// Where Gemma and the connectors live between prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextResidency {
    /// Resident when the device has room beside what is already loaded, else
    /// streamed. `FASTVIDEO_LTX2_TEXT=resident|streamed` overrides.
    #[default]
    Auto,
    /// One layer on the device at a time; every new prompt re-reads the
    /// checkpoint. The only choice on a card the DiT already fills.
    Streamed,
    /// Gemma (23.5 GB as bf16) and the connectors (2.9 GB) stay loaded: a new
    /// prompt costs one forward.
    Resident,
}

impl TextResidency {
    /// `Auto` resolved against the environment and the device. `free` is what
    /// the device reports *now* — after the DiT and decoders are in place;
    /// `needed` what the text path would hold plus working room.
    pub fn resolve(self, env: Option<&str>, free: Option<u64>, needed: u64) -> Result<bool> {
        let asked = match env.map(str::trim).filter(|v| !v.is_empty()) {
            None => self,
            Some(v) if v.eq_ignore_ascii_case("resident") => Self::Resident,
            Some(v) if v.eq_ignore_ascii_case("streamed") => Self::Streamed,
            Some(v) if v.eq_ignore_ascii_case("auto") => Self::Auto,
            Some(v) => {
                return Err(err(format!(
                    "FASTVIDEO_LTX2_TEXT={v}: expected resident, streamed or auto"
                )))
            }
        };
        Ok(match asked {
            Self::Resident => true,
            Self::Streamed => false,
            // No device to ask (a CPU run): stream, which is what is validated there.
            Self::Auto => free.is_some_and(|f| f >= needed),
        })
    }
}

/// Activations of a text forward and of the denoise that follows, plus
/// allocator slack: what must stay free once the resident encoder is loaded.
const RESIDENT_HEADROOM: u64 = 8 << 30;

struct ResidentText {
    gemma: ResidentDecoder,
    connectors: TextConnectors,
}

/// Prompt → connector contexts, with the conditioning cache in front.
///
/// On a hit neither Gemma nor the connectors are opened: the key is made from
/// the prompt, the tokenizer file and the weight *files'* identities
/// ([`weights_identity`]), which are computed once per encoder.
pub struct TextEncoder {
    paths: Ltx2Paths,
    cfg: Ltx2Config,
    cache: Option<TextCache>,
    residency: TextResidency,
    /// Built on the first cache miss: a run of hits never loads Gemma at all.
    resident: Option<ResidentText>,
    /// `(sha256-able tokenizer bytes, gemma identity, connector identity)`.
    identity: Option<(Vec<u8>, [u8; 32], [u8; 32])>,
}

impl TextEncoder {
    pub fn new(paths: &Ltx2Paths, cfg: &Ltx2Config, options: &PipelineOptions) -> Self {
        Self {
            paths: paths.clone(),
            cfg: cfg.clone(),
            cache: options.text_cache.clone().map(TextCache::new),
            residency: options.text_residency,
            resident: None,
            identity: None,
        }
    }

    fn tokenizer_path(&self) -> PathBuf {
        self.paths
            .text_root()
            .join("tokenizer")
            .join("tokenizer.json")
    }

    fn decoder_config(cfg: &Ltx2Config) -> DecoderConfig {
        // The product runs Gemma in bf16, where the embedding multiplier is 62.0.
        let dc = if cfg.gemma4.is_some() || cfg.version == Ltx2ModelVersion::V25 {
            DecoderConfig::gemma4_12b_text()
        } else {
            DecoderConfig::gemma3_12b_text()
        };
        dc.for_bf16_reference()
    }

    /// Part of the cache key: the encoder *and* the feature normalisation, so a
    /// context normalised one way is never served for the other.
    fn text_encoder_kind(cfg: &Ltx2Config) -> &'static [u8] {
        let gemma4 = cfg.gemma4.is_some() || cfg.version == Ltx2ModelVersion::V25;
        match (gemma4, cfg.connectors.text_norm()) {
            (true, Ltx2TextNorm::PerTokenRms) => b"gemma4-12b/per-token-rms",
            (true, Ltx2TextNorm::MaskedMinMax) => b"gemma4-12b",
            (false, Ltx2TextNorm::PerTokenRms) => b"gemma3-12b/per-token-rms",
            (false, Ltx2TextNorm::MaskedMinMax) => b"gemma3-12b",
        }
    }

    /// Device bytes a resident text path would add, from the shapes alone.
    fn resident_bytes(&self) -> u64 {
        let g = Self::decoder_config(&self.cfg);
        let c = &self.cfg.connectors;
        let width: u64 = if crate::wan::nn::bf16_linears_active() {
            2
        } else {
            4
        };
        let per_layer: u64 = (0..g.num_layers())
            .map(|i| {
                let (hq, hkv, dq, dkv, h) = (
                    g.layer_heads(i),
                    g.layer_kv_heads(i),
                    g.layer_head_dim(i),
                    g.layer_kv_head_dim(i),
                    g.hidden,
                );
                let v = if g.attention_k_eq_v { 0 } else { h * hkv * dkv };
                (h * hq * dq + h * hkv * dkv + v + hq * dq * h + 3 * h * g.intermediate) as u64
            })
            .sum();
        let d = c.inner_dim();
        let connector = (c.video_connector_num_layers + c.audio_connector_num_layers)
            * (4 * d * d + 8 * d * d)
            + c.text_proj_in_features() * c.caption_channels;
        (per_layer + connector as u64) * width
    }

    fn key(&mut self, prompt: &str) -> Result<String> {
        if self.identity.is_none() {
            let tokenizer = std::fs::read(self.tokenizer_path())
                .map_err(|e| err(format!("{}: {e}", self.tokenizer_path().display())))?;
            let text_dir = self.paths.text_root().join("text_encoder");
            // `Lightricks/LTX-2` keeps a stale duplicate shard set next to the real
            // one; when the real one is there, it alone identifies the encoder.
            let has_model_set = std::fs::read_dir(&text_dir)
                .map(|d| {
                    d.filter_map(|e| e.ok())
                        .any(|e| e.file_name().to_string_lossy().starts_with("model-"))
                })
                .unwrap_or(false);
            let gemma = weights_identity(&text_dir, has_model_set.then_some("model-"))?;
            let connectors = if self.paths.dit.is_file() {
                weights_identity(&self.paths.dit, None)?
            } else if self.paths.dit.join("connectors").is_dir() {
                weights_identity(&self.paths.dit.join("connectors"), None)?
            } else if self.paths.dit.join("text_embedding_projection").is_dir() {
                weights_identity(&self.paths.dit.join("text_embedding_projection"), None)?
            } else if let Some(sibling) = self
                .paths
                .dit
                .parent()
                .map(|p| p.join("connectors"))
                .filter(|p| p.is_dir())
            {
                weights_identity(&sibling, None)?
            } else if let Some(sibling) = self
                .paths
                .dit
                .parent()
                .map(|p| p.join("text_embedding_projection"))
                .filter(|p| p.is_dir())
            {
                weights_identity(&sibling, None)?
            } else {
                weights_identity(&self.paths.dit, None)?
            };
            self.identity = Some((tokenizer, gemma, connectors));
        }
        let (tokenizer, gemma, connectors) = self
            .identity
            .as_ref()
            .ok_or_else(|| err("ltx2: text identity missing"))?;
        Ok(cache_key(
            prompt,
            tokenizer,
            self.cfg.defaults.max_sequence_length,
            Self::text_encoder_kind(&self.cfg),
            gemma,
            connectors,
        ))
    }

    fn load_connectors(&self) -> Result<TextConnectors> {
        let map = open_distilled(&self.paths.dit, "connectors")?;
        Ok(TextConnectors::load(
            &map,
            &Keys::connectors(Keys::detect(&map)),
            &self.cfg.connectors,
        )?)
    }

    /// Decide once, on the first prompt that actually has to be encoded, and
    /// load the resident models if that is the answer.
    fn ensure_backend(&mut self) -> Result<()> {
        if self.resident.is_some() || self.residency == TextResidency::Streamed {
            return Ok(());
        }
        let env = std::env::var("FASTVIDEO_LTX2_TEXT").ok();
        let free = crate::wan::device::free_memory().map(|(free, _)| free);
        let needed = self.resident_bytes() + RESIDENT_HEADROOM;
        if !self.residency.resolve(env.as_deref(), free, needed)? {
            crate::wan::log::info(format_args!(
                "ltx2 text: streamed (free {:?} bytes, resident needs {needed})",
                free
            ));
            self.residency = TextResidency::Streamed;
            return Ok(());
        }
        let timer = Instant::now();
        let cfg = Self::decoder_config(&self.cfg);
        let map = WeightMap::open(&self.paths.text_root().join("text_encoder"))?;
        let gemma = ResidentDecoder::load(&map, &cfg, cfg.num_layers())?;
        let connectors = self.load_connectors()?;
        sync()?;
        crate::wan::log::info(format_args!(
            "ltx2 text: resident, {:.1} GiB on device, loaded in {:.1}s",
            gemma.device_bytes() as f64 / f64::from(1u32 << 30),
            timer.elapsed().as_secs_f64()
        ));
        self.resident = Some(ResidentText { gemma, connectors });
        self.residency = TextResidency::Resident;
        Ok(())
    }

    /// Resident: one forward. Streamed: Gemma layer by layer, the connectors
    /// loaded, used and dropped.
    fn compute(&mut self, padded: &PaddedPrompt) -> Result<CachedContexts> {
        self.ensure_backend()?;
        let out = match &self.resident {
            Some(r) => r.connectors.forward(
                &HiddenStack::encode_resident(&r.gemma, padded)?,
                padded.max_len(),
            )?,
            None => {
                let gemma = WeightMap::open(&self.paths.text_root().join("text_encoder"))?;
                let stack = HiddenStack::encode(&gemma, &Self::decoder_config(&self.cfg), padded)?;
                drop(gemma);
                self.load_connectors()?.forward(&stack, padded.max_len())?
            }
        };
        Ok(CachedContexts {
            video: out.video,
            audio: out.audio,
        })
    }

    /// `"resident"`, `"streamed"`, or `"undecided"` while only cache hits have
    /// been served.
    pub fn mode(&self) -> &'static str {
        match (self.resident.is_some(), self.residency) {
            (true, _) => "resident",
            (false, TextResidency::Streamed) => "streamed",
            _ => "undecided",
        }
    }

    /// Two-stage reloads a second fused DiT. Auto would otherwise keep Gemma
    /// resident beside that swap and OOM a 96 GB card. `FASTVIDEO_LTX2_TEXT`
    /// / an explicit residency still win.
    pub fn prefer_streamed_for_two_stage(&mut self) {
        if self.residency != TextResidency::Auto {
            return;
        }
        let env = std::env::var("FASTVIDEO_LTX2_TEXT").ok();
        if env
            .as_deref()
            .is_some_and(|v| v.trim().eq_ignore_ascii_case("resident"))
        {
            return;
        }
        crate::wan::log::info(format_args!(
            "ltx2 text: streamed for two-stage (Gemma is not kept beside DiT reload)"
        ));
        self.residency = TextResidency::Streamed;
    }

    /// Drop a resident Gemma + connectors after contexts are already in hand.
    pub fn unload_resident(&mut self) {
        if self.resident.take().is_some() {
            crate::wan::log::info(format_args!(
                "ltx2 text: dropped resident Gemma (contexts already encoded)"
            ));
        }
    }

    /// `use_cache = false` neither reads nor writes the cache (a warm-up run
    /// must not turn the run it warms up for into a hit).
    pub fn encode(
        &mut self,
        prompt: &str,
        use_cache: bool,
    ) -> Result<(CachedContexts, TextReport)> {
        let timer = Instant::now();
        let padded = PaddedPrompt::tokenize(
            &self.tokenizer_path(),
            prompt,
            self.cfg.defaults.max_sequence_length,
        )?;
        let key = if use_cache && self.cache.is_some() {
            Some(self.key(prompt)?)
        } else {
            None
        };
        // Taken out so the compute closure can borrow the encoder mutably.
        let cache = self.cache.take();
        let result = cached_or(cache.as_ref(), key.as_deref(), &padded, || {
            self.compute(&padded)
        });
        self.cache = cache;
        let (contexts, outcome) = result?;
        sync()?;
        let mode = if outcome == CacheOutcome::Hit {
            "cache"
        } else {
            self.mode()
        };
        Ok((
            contexts,
            TextReport {
                cache: outcome,
                mode,
                tokens: padded.real,
                seconds: timer.elapsed().as_secs_f64(),
                key,
            },
        ))
    }
}

/// The cache protocol, apart from what it caches: look up, else compute and
/// store. A cache that cannot be written is reported by the log, not by failing
/// a generation whose conditioning is already in hand.
fn cached_or(
    cache: Option<&TextCache>,
    key: Option<&str>,
    padded: &PaddedPrompt,
    compute: impl FnOnce() -> Result<CachedContexts>,
) -> Result<(CachedContexts, CacheOutcome)> {
    let (Some(cache), Some(key)) = (cache, key) else {
        return Ok((compute()?, CacheOutcome::Off));
    };
    if let Some(hit) = cache.load(key, padded) {
        return Ok((hit, CacheOutcome::Hit));
    }
    let fresh = compute()?;
    if let Err(e) = cache.store(key, padded, &fresh.video, &fresh.audio) {
        crate::wan::log::info(format_args!("ltx2 text cache: not stored: {e}"));
    }
    Ok((fresh, CacheOutcome::Miss))
}

/// How a pipeline is set up.
#[derive(Debug, Clone, Default)]
pub struct PipelineOptions {
    /// Directory of the conditioning cache; `None` turns it off.
    pub text_cache: Option<PathBuf>,
    pub text_residency: TextResidency,
    /// Where the DiT blocks live. `None`: `FASTVIDEO_DIT_OFFLOAD`, else
    /// `Auto` (resident when the free memory covers the resident plan of the
    /// 4k5s workload). Streamed also keeps the video/audio decoders off the
    /// device except around the calls that read them (`--offload cpu`).
    pub dit_offload: Option<DitOffload>,
}

/// What [`Ltx2Pipeline::generate`] can reproduce. LTX-2.5 generates only
/// through `DistilledPipeline`, which has no guidance at all (`SimpleDenoiser`
/// on both stages, `distilled.py:245-313`); the dev two-stage
/// (`TI2VidTwoStages`) needs the multimodal CFG/STG/modality/rescale guider,
/// which is not ported — plain CFG would silently be a different sampler.
pub fn check_generate_contract(cfg: &Ltx2Config, guided: bool) -> Result<()> {
    if cfg.version != Ltx2ModelVersion::V25 {
        return Ok(());
    }
    if cfg.scheduler.use_dynamic_shifting {
        return Err(err(
            "ltx2: LTX-2.5 dev generation (TI2VidTwoStages, multimodal guider) is not ported; \
the dev transformer runs only as the refiner",
        ));
    }
    if guided {
        return Err(err(
            "ltx2: LTX-2.5 distilled is unguided (DistilledPipeline / SimpleDenoiser); \
set --guidance-scale and --audio-guidance-scale to 1",
        ));
    }
    Ok(())
}

/// A DiT path that names a distilled checkpoint (file, or the directory
/// holding it). The refiners fuse their LoRA onto the *dev* transformer.
fn names_distilled(path: &Path) -> bool {
    let named = |p: Option<&Path>| {
        p.and_then(Path::file_name).is_some_and(|n| {
            n.to_string_lossy()
                .to_ascii_lowercase()
                .contains("distilled")
        })
    };
    named(Some(path)) || named(path.parent())
}

/// Whether the DiT can stay loaded while the VAE decodes. A streamed DiT
/// holds only its block ring on the device (its host copy is what a reload
/// would rebuild), so it always stays. A resident one stays when free device
/// memory covers the planned decode phase plus a 2 GiB margin.
fn keep_dit_through_decode(residency: Residency, cfg: &Ltx2Config, req: &Ltx2Request) -> bool {
    if residency.is_streamed() {
        return true;
    }
    let plan = fastvideo_models::ltx2::memory::plan_distilled_two_stage(
        cfg,
        req.height,
        req.width,
        req.num_frames,
        req.frame_rate,
        fastvideo_models::ltx2::memory::PlanOptions::default(),
    );
    let Some(decode) = plan.phase("decode") else {
        return false;
    };
    let free = crate::wan::device::free_memory().map_or(0, |(f, _)| f);
    free >= decode.total() + (2u64 << 30)
}

fn release_dit_for_decode(model: &mut Option<Ltx2Transformer>, loaded_strength: &mut Option<f32>) {
    *model = None;
    *loaded_strength = None;
}

/// The models of a stage-1 run, loaded once and kept: the DiT (37.8 GB) and the
/// video/audio decoders (~2 GB). A second generation pays for text, denoise and
/// decode only — and for text not even that when the prompt was seen before.
/// `model` is `None` after generate-decode (DiT dropped for VRAM); the next
/// generate reloads it from `dit`.
pub struct Ltx2Pipeline {
    cfg: Ltx2Config,
    dit: PathBuf,
    weights: PathBuf,
    model: Option<Ltx2Transformer>,
    /// Distilled LoRA file, when one is next to the weights or `FASTVIDEO_LTX2_LORA` points at it.
    /// Only resolved for a dev (dynamic-shift) checkpoint; distilled ones never fuse.
    lora: Option<PathBuf>,
    /// Strength the resident DiT was fused at. `Some(0.0)` is the unfused base.
    loaded_strength: Option<f32>,
    /// Video VAE, audio VAE and vocoder; the spatial upsampler is loaded
    /// only around its call in a two-stage run. `None` between the calls
    /// that read them when the DiT is streamed.
    decoders: Option<Decoders>,
    residency: Residency,
    has_upsampler: bool,
    text: TextEncoder,
    /// Seconds [`Self::load`] took.
    pub load_s: f64,
}

impl Ltx2Pipeline {
    pub fn load(paths: &Ltx2Paths, cfg: &Ltx2Config, options: &PipelineOptions) -> Result<Self> {
        let timer = Instant::now();
        // A distilled checkpoint already carries the adapter: the RTX5090
        // distilled two-stage (`run_ltx25_gpu.sh`) passes no LoRA. Only a dev
        // transformer fuses one (two-stage stage 2, the refiners).
        let lora = if cfg.scheduler.use_dynamic_shifting {
            super::lora::resolve(&paths.weights, &paths.dit, cfg.version)?
        } else {
            None
        };
        let residency = {
            let policy = DitOffload::from_env_or(options.dit_offload).map_err(err)?;
            let w = fastvideo_models::ltx2::Rtx5090Workload::DEFAULT;
            let need = fastvideo_models::ltx2::memory::plan_distilled_two_stage(
                cfg,
                w.height(),
                w.width(),
                w.num_frames(),
                w.frame_rate(),
                fastvideo_models::ltx2::memory::PlanOptions::default(),
            )
            .peak();
            let free = crate::wan::device::free_memory().map(|(f, _)| f);
            let residency = policy.resolve(need, free);
            crate::wan::log::info(format_args!(
                "ltx2 dit offload: {} -> {} (resident {} plan {:.1} GiB, {} free before load)",
                policy.as_str(),
                residency.as_str(),
                w.name(),
                need as f64 / f64::from(1u32 << 30),
                free.map_or("unknown".into(), |f| format!(
                    "{:.1} GiB",
                    f as f64 / f64::from(1u32 << 30)
                )),
            ));
            residency
        };
        let model = if lora.is_none() {
            let map = open_distilled(&paths.dit, "transformer")?;
            Some(Ltx2Transformer::load_with_residency(
                &map,
                &Keys::transformer(Keys::detect(&map)),
                &cfg.transformer,
                residency,
            )?)
        } else {
            None
        };
        let decoders = if residency.is_streamed() {
            None
        } else {
            Some(Decoders::load_without_upsampler(&paths.weights, cfg)?)
        };
        let has_upsampler =
            cfg.latent_upsampler.is_some() && upsampler_dir(&paths.weights).is_some();
        sync()?;
        let loaded_strength = model.as_ref().map(|_| 0.0);
        Ok(Self {
            cfg: cfg.clone(),
            dit: paths.dit.clone(),
            weights: paths.weights.clone(),
            model,
            lora,
            loaded_strength,
            decoders,
            residency,
            has_upsampler,
            text: TextEncoder::new(paths, cfg, options),
            load_s: timer.elapsed().as_secs_f64(),
        })
    }

    /// Resident DiT fused at `strength`. `0` is the base checkpoint.
    fn dit_for(&mut self, strength: f32) -> Result<&Ltx2Transformer> {
        let strength = if self.lora.is_some() { strength } else { 0.0 };
        if self.model.is_some() && self.loaded_strength == Some(strength) {
            return Ok(self.model.as_ref().expect("resident dit"));
        }
        if self.model.is_some() && self.lora.is_some() {
            crate::wan::log::info(format_args!(
                "ltx2: set_lora_strength {strength} (resident DiT)"
            ));
            self.model
                .as_mut()
                .expect("resident dit")
                .set_lora_strength(strength)?;
            self.loaded_strength = Some(strength);
            return Ok(self.model.as_ref().expect("resident dit"));
        }
        if self.model.is_some() {
            crate::wan::log::info(format_args!(
                "ltx2: loading DiT at lora strength {strength}"
            ));
            // Drop the old DiT before the next full load. Holding both plus
            // resident Gemma is what OOMed a 96 GB PRO 6000 (~95 GiB).
            self.model = None;
            self.loaded_strength = None;
            sync()?;
        }
        let map = open_distilled(&self.dit, "transformer")?;
        let keys = Keys::transformer(Keys::detect(&map));
        let mut model = if let Some(path) = self.lora.clone() {
            // Strength 0 so `apply_bf16` is a no-op and `attach_linear` snapshots
            // unfused `W0`. Device re-fuse walks the resident linears.
            let guard = super::lora::install(&path, 0.0)?;
            let mut model = Ltx2Transformer::load_with_residency(
                &map,
                &keys,
                &self.cfg.transformer,
                self.residency,
            )?;
            let hits = super::lora::hits();
            if hits == 0 {
                drop(guard);
                return Err(err(format!(
                    "ltx2 lora: no base weight matched {}",
                    path.display()
                )));
            }
            model.set_lora_strength(strength)?;
            drop(guard);
            crate::wan::log::info(format_args!(
                "ltx2 lora: attached {hits} weights, fused at strength {strength} ({})",
                path.display()
            ));
            model
        } else {
            Ltx2Transformer::load_with_residency(
                &map,
                &keys,
                &self.cfg.transformer,
                self.residency,
            )?
        };
        sync()?;
        self.loaded_strength = Some(strength);
        self.model = Some(model);
        Ok(self.model.as_ref().expect("just loaded"))
    }

    /// Where the DiT blocks live for this pipeline.
    pub fn residency(&self) -> Residency {
        self.residency
    }

    /// The decoders, loaded now if a streamed run keeps them off the device.
    fn ensure_decoders(&mut self) -> Result<&Decoders> {
        if self.decoders.is_none() {
            let timer = Instant::now();
            self.decoders = Some(Decoders::load_without_upsampler(&self.weights, &self.cfg)?);
            crate::wan::log::info(format_args!(
                "ltx2 decoders: loaded for this call ({:.1}s)",
                timer.elapsed().as_secs_f64()
            ));
        }
        Ok(self.decoders.as_ref().expect("just loaded"))
    }

    /// Streamed runs: drop the decoders until the next call that reads them.
    fn release_transient_decoders(&mut self) -> Result<()> {
        if self.residency.is_streamed() && self.decoders.take().is_some() {
            trim()?;
        }
        Ok(())
    }

    /// Log the block streaming counters of the stage that just ran.
    fn report_offload(&self, stage: &str) {
        if let Some(model) = self.model.as_ref() {
            if model.residency().is_streamed() {
                model.report_offload(stage);
            }
        }
    }

    fn stage_lora_strengths(&self, two_stage: bool) -> (f32, f32) {
        if !two_stage {
            return (0.0, 0.0);
        }
        // Distilled checkpoints already bake the adapter in. LoRA 0.8 (and
        // the 2.3 0.25/0.5 pair) apply to the dev BF16 DiT only.
        if !self.cfg.scheduler.use_dynamic_shifting {
            return (0.0, 0.0);
        }
        fastvideo_models::ltx2::lora::stage_strengths(self.cfg.version).unwrap_or((0.0, 0.0))
    }

    fn lora_note(&self, two_stage: bool) -> String {
        let Some(path) = &self.lora else {
            return "distilled lora not beside weights".into();
        };
        if !self.cfg.scheduler.use_dynamic_shifting {
            return format!("distilled: no lora fuse {}", path.display());
        }
        let Some((s1, s2)) = fastvideo_models::ltx2::lora::stage_strengths(self.cfg.version) else {
            return format!("lora {}", path.display());
        };
        if two_stage {
            format!("lora {s1}/{s2} {}", path.display())
        } else {
            format!("lora off for single stage {}", path.display())
        }
    }

    /// Turn on the opt-in stage-1 cache / prune the environment asks for, only
    /// where a reference profile runs them; anywhere else they would change the
    /// output of a profile that has none, so the request is refused.
    fn arm_requested(&self, distilled: bool, guided: bool) -> Result<()> {
        let Some(model) = self.model.as_ref() else {
            return Ok(());
        };
        let version = self.cfg.version;
        if fastvideo_models::ltx2::fbcache::requested(
            std::env::var("FASTVIDEO_LTX2_FBCACHE").ok().as_deref(),
        ) {
            fastvideo_models::ltx2::fbcache::scope(version, distilled, guided, 1).map_err(err)?;
            model.enable_fbcache();
        }
        if fastvideo_models::ltx2::pisa::stage1_cache_requested(
            std::env::var("FASTVIDEO_LTX2_STAGE1_CACHE").ok().as_deref(),
        ) {
            fastvideo_models::ltx2::pisa::stage1_cache_scope(version).map_err(err)?;
            model.enable_stage1_cache();
        }
        if fastvideo_models::ltx2::pisa::midpoint_prune_requested(
            std::env::var("FASTVIDEO_LTX2_MIDPOINT_PRUNE")
                .ok()
                .as_deref(),
        ) {
            fastvideo_models::ltx2::pisa::midpoint_prune_scope(version).map_err(err)?;
            model.enable_midpoint_prune();
        }
        Ok(())
    }

    /// One clip. `use_text_cache = false` bypasses the conditioning cache for
    /// this call only.
    pub fn generate(
        &mut self,
        req: &Ltx2Request,
        use_text_cache: bool,
        observer: Option<StepObserver<'_>>,
    ) -> Result<Ltx2Output> {
        req.validate()?;
        let cfg = self.cfg.clone();
        let use_cfg = req.guidance_scale != 1.0 || req.audio_guidance_scale != 1.0;
        check_generate_contract(&cfg, use_cfg)?;
        if fastvideo_models::ltx2::hq::requested(std::env::var("FASTVIDEO_LTX2_HQ").ok().as_deref())
            || (req.two_stage
                && req.stage1_steps() == fastvideo_models::ltx2::hq::STAGE1_STEPS
                && (req.guidance_scale - fastvideo_models::ltx2::hq::GUIDANCE_SCALE).abs() < 1e-6)
        {
            crate::wan::log::info(format_args!(
                "ltx2 hq: stage-1 {} steps, stage-2 sigmas {:?}, guidance {}, {}x{} {}f ({})",
                req.stage1_steps(),
                fastvideo_models::ltx2::hq::STAGE2_SIGMAS,
                req.guidance_scale,
                req.width,
                req.height,
                req.num_frames,
                self.lora_note(req.two_stage),
            ));
        }
        if req.two_stage {
            if !matches!(cfg.version, Ltx2ModelVersion::V23 | Ltx2ModelVersion::V25) {
                return Err(err("ltx2: --two-stage requires model version 2.3 or 2.5"));
            }
            if !self.has_upsampler {
                return Err(err(
                    "ltx2: two-stage needs weights/latent_upsampler|spatial_upscaler (missing beside --weights)",
                ));
            }
            self.text.prefer_streamed_for_two_stage();
        }
        if req.diff_vae {
            if cfg.version != Ltx2ModelVersion::V25 {
                return Err(err("ltx2: --diff-vae requires model version 2.5"));
            }
            if cfg.diffusion_decoder.is_none() {
                return Err(err(
                    "ltx2: --diff-vae needs diffusion_decoder config (2.5 pack)",
                ));
            }
            if !self.weights.join("diffusion_decoder").is_dir() {
                return Err(err("ltx2: --diff-vae needs weights/diffusion_decoder"));
            }
        }
        if self.residency.is_streamed() {
            // A streamed DiT is the small-card profile: Gemma streams too.
            self.text.prefer_streamed_for_two_stage();
        }
        let mut timings = Ltx2Timings::default();
        let mut memory = MemoryLog::start("ltx2");
        let (stage1_h, stage1_w) = if req.two_stage {
            (req.height / 2, req.width / 2)
        } else {
            (req.height, req.width)
        };
        let grid1 = cfg
            .transformer
            .latent_grid(req.num_frames, stage1_h, stage1_w);
        let grid_full = cfg
            .transformer
            .latent_grid(req.num_frames, req.height, req.width);
        let audio_tokens = cfg.transformer.audio_tokens(req.num_frames, req.frame_rate);
        if audio_tokens == 0 {
            return Err(err("ltx2: the clip is too short for a single audio latent"));
        }

        let (contexts, text_report) = self.text.encode(&req.prompt, use_text_cache)?;
        timings.text_s = text_report.seconds;
        let uncond_contexts = if use_cfg {
            let (c, rep) = self.text.encode(&req.negative_prompt, use_text_cache)?;
            timings.text_s += rep.seconds;
            Some(c)
        } else {
            None
        };
        if req.two_stage {
            self.text.unload_resident();
        }
        // The streamed Gemma and the connectors are gone; hand their pool
        // blocks back before the DiT (re)loads and stage 1 allocates.
        trim()?;
        memory.mark("text")?;

        let (s1, s2) = self.stage_lora_strengths(req.two_stage);
        let timer = Instant::now();
        self.dit_for(s1)?;
        let distilled = !cfg.scheduler.use_dynamic_shifting;
        self.arm_requested(distilled, use_cfg)?;
        let state = LatentState::for_version(cfg.version);
        let mut text = {
            let model = self.model.as_ref().expect("dit");
            model.project_text(&contexts.video, &contexts.audio)?
        };
        let mut text_uncond = match &uncond_contexts {
            Some(c) => {
                let model = self.model.as_ref().expect("dit");
                Some(model.project_text(&c.video, &c.audio)?)
            }
            None => None,
        };
        // Stage 2 is one unguided forward per step on every line
        // (`distilled.py:294-313`, `ti2vid_two_stages.py:289-307`): only the
        // conditional context is ever re-projected.
        let reproject = req.two_stage && s1 != s2;
        let mut kept_contexts = reproject.then_some(contexts);
        let ropes = Ropes::new(&cfg.transformer, grid1, audio_tokens, req.frame_rate as f32)?;
        // `generator = torch.Generator().manual_seed(seed)` (`distilled.py:217`):
        // the stage-1 initial noise and, later, the stage-2 renoise.
        let mut noise = NoiseStream::new(req.seed, state == LatentState::Bf16);
        let (mut video, audio) = initial_noise(&cfg, grid1, audio_tokens, &mut noise)?;
        if let Some(ref image_path) = req.image_path {
            let spat = cfg.transformer.vae_scale_factors[1];
            let vae_dir = self.weights.join("vae");
            let first = super::i2v_encode::encode_first_frame(
                image_path,
                stage1_h,
                stage1_w,
                cfg.transformer.in_channels,
                spat,
                Some(vae_dir.as_path()),
            )?;
            video = super::i2v_encode::condition_first_frame(&video, grid1, &first)?;
            crate::wan::log::info(format_args!(
                "ltx2: I2V first-frame encode from {}",
                image_path.display()
            ));
        }
        let mut step_s = Vec::new();
        let mut observer = observer;
        let video_seq_len = grid1[0] * grid1[1] * grid1[2];
        let default_dev_steps = if cfg.version == Ltx2ModelVersion::V23 {
            30
        } else {
            40
        };
        let schedule = if cfg.scheduler.use_dynamic_shifting {
            let steps = req.num_inference_steps.unwrap_or(default_dev_steps);
            Ltx2Schedule::dev(&cfg.scheduler, steps, video_seq_len)
        } else {
            Ltx2Schedule::distilled_subset(req.stage1_steps()).map_err(err)?
        };
        // `DistilledPipeline` samples stage 1 ancestrally from 2.5 on
        // (`distilled.py:62-85`); the dev line never does.
        let ancestral = cfg.version == Ltx2ModelVersion::V25 && distilled;
        let res2s = cfg.version == Ltx2ModelVersion::V23;
        let (mut video, mut audio) = {
            let model = self.model.as_ref().expect("ensure_dit");
            let mut record = |i: usize, v: &CudaTensor, a: &CudaTensor, s: f64| -> Result<()> {
                step_s.push(s);
                match observer.as_mut() {
                    Some(obs) => obs(i, v, a, s),
                    None => Ok(()),
                }
            };
            if ancestral {
                denoise_ancestral(
                    model,
                    &text,
                    &ropes,
                    &schedule,
                    video,
                    audio,
                    AncestralOpts {
                        eta: 1.0,
                        s_noise: 1.0,
                        noise_seed: req.seed + 10_000,
                    },
                    Some(&mut record),
                    Ltx2Stage2Attn::Off,
                    state,
                )?
            } else if res2s {
                denoise_res2s(
                    model,
                    &text,
                    text_uncond.as_ref(),
                    &ropes,
                    &schedule,
                    req.guidance_scale,
                    req.audio_guidance_scale,
                    video,
                    audio,
                    Some(&mut record),
                    Ltx2Stage2Attn::Off,
                )?
            } else if let Some(ref uncond) = text_uncond {
                denoise_cfg(
                    model,
                    &text,
                    uncond,
                    &ropes,
                    &schedule,
                    req.guidance_scale,
                    req.audio_guidance_scale,
                    video,
                    audio,
                    Some(&mut record),
                    Ltx2Stage2Attn::Off,
                )?
            } else {
                denoise(
                    model,
                    &text,
                    &ropes,
                    &schedule,
                    video,
                    audio,
                    Some(&mut record),
                    Ltx2Stage2Attn::Off,
                )?
            }
        };
        timings.stage1_s = timer.elapsed().as_secs_f64();
        self.report_offload("stage1");
        if let Some(model) = self.model.as_ref() {
            model.disarm_fbcache();
            // Stage-1 buffers (cache residuals, the stored velocity) go now,
            // not when stage 2 overwrites them.
            model.clear_step_caches();
        }
        drop(ropes);
        trim()?;
        memory.mark("stage1")?;

        let decode_grid = if req.two_stage {
            let up_timer = Instant::now();
            // Loaded for this call only, like the reference's `VideoUpsampler`
            // block. The DiT stays: stage 2 runs the same weights, and the
            // upsampler's working set is small beside stage 2's.
            {
                // Streamed: the video VAE's latent statistics are what this
                // reads; it is loaded for the call and dropped after it.
                if let Some(model) = self.model.as_ref() {
                    model.release_offload_device();
                }
                let up = load_upsampler(&self.weights, &cfg)?.ok_or_else(|| {
                    err("ltx2: two-stage needs weights/latent_upsampler|spatial_upscaler")
                })?;
                let decoders = self.ensure_decoders()?;
                let upsampled =
                    up.forward(&decoders.video.denormalize(&unpack_video(&video, grid1)?)?)?;
                video = state.store(pack_video(&decoders.video.normalize(&upsampled)?)?)?;
            }
            self.release_transient_decoders()?;
            trim()?;
            timings.upsample_s = up_timer.elapsed().as_secs_f64();
            memory.mark("upsample")?;
            crate::wan::log::info(format_args!(
                "ltx2 upsample {:.2}s → latent grid {:?}",
                timings.upsample_s, grid_full
            ));

            if req.sol_stage2 {
                crate::wan::log::info(format_args!(
                    "ltx2 sol stage-2: taus 1/1.25/1.5, layer 0 dense, layers 1-47 sol-attn kernel (thresh_type=diag), {}",
                    self.lora_note(true)
                ));
            }
            if req.pisa_stage2 {
                crate::wan::log::info(format_args!(
                    "ltx2 pisa stage-2: layers 0-1 dense, layers 2-47 pisa kernel sparsity 0.9 block 64 score-route first-order remainder, feat_norm prune steps 1,2 ratio 0.5 when FASTVIDEO_LTX2_MIDPOINT_PRUNE, {}, stage-1 SCSP res2s calls 16-28 of 29 when FASTVIDEO_LTX2_STAGE1_CACHE",
                    self.lora_note(true)
                ));
            }
            self.dit_for(s2)?;
            if reproject {
                let contexts = kept_contexts.take().expect("stage-2 text context");
                text = {
                    let model = self.model.as_ref().expect("dit");
                    model.project_text(&contexts.video, &contexts.audio)?
                };
            }
            drop(text_uncond.take());
            if let Some(model) = self.model.as_ref() {
                // FBCache is a stage-1 cache only (GB200 `fullopt.toml`).
                model.disarm_fbcache();
                model.clear_step_caches();
                model.set_prune_active(true);
            }
            let schedule2 =
                Ltx2Schedule::distilled_stage_2_steps(req.stage2_steps()).map_err(err)?;
            let sigma = schedule2.sigmas[0] as f32;
            // Stage-2 entry: video then audio re-noised to `σ_0` from the same
            // generator as the initial noise (`distilled.py:294-313`).
            let mut draws = noise.draw(&[&video.shape, &audio.shape])?;
            let (noise_a, noise_v) = (draws.pop().expect("audio"), draws.pop().expect("video"));
            video = renoise(&video, &noise_v, sigma, state)?;
            audio = renoise(&audio, &noise_a, sigma, state)?;
            drop((noise_v, noise_a));

            let s2_timer = Instant::now();
            let ropes2 = Ropes::new(
                &cfg.transformer,
                grid_full,
                audio_tokens,
                req.frame_rate as f32,
            )?;
            let step_offset = step_s.len();
            let (v2, a2) = {
                let model = self.model.as_ref().expect("ensure_dit");
                let mut record2 =
                    |i: usize, v: &CudaTensor, a: &CudaTensor, s: f64| -> Result<()> {
                        step_s.push(s);
                        match observer.as_mut() {
                            Some(obs) => obs(step_offset + i, v, a, s),
                            None => Ok(()),
                        }
                    };
                denoise_with(
                    model,
                    &text,
                    &ropes2,
                    &schedule2,
                    video,
                    audio,
                    Some(&mut record2),
                    req.stage2_attn(),
                    state,
                )?
            };
            if let Some(model) = self.model.as_ref() {
                model.set_prune_active(false);
            }
            video = v2;
            audio = a2;
            timings.stage2_s = s2_timer.elapsed().as_secs_f64();
            self.report_offload("stage2");
            drop(ropes2);
            memory.mark("stage2")?;
            grid_full
        } else {
            grid1
        };
        timings.denoise_s = timings.stage1_s + timings.upsample_s + timings.stage2_s;
        timings.step_s = step_s;
        drop(text);
        drop(text_uncond);

        // Free the resident DiT (~37 GiB) before VAE activations, conv or
        // DiffVAE, when the decode would not fit beside it (the reference's
        // `DiffusionStage` frees its transformer on exit to fit 32 GB). On a
        // card with room it stays, so a warm second generation does not
        // reload 48 blocks from disk.
        if !keep_dit_through_decode(self.residency, &self.cfg, req) {
            release_dit_for_decode(&mut self.model, &mut self.loaded_strength);
            trim()?;
        }
        self.ensure_decoders()?;
        let decoders = self.decoders.as_ref().expect("decoders");

        let written = if req.diff_vae {
            crate::wan::log::info(format_args!("ltx2: DiffVAE decode (DiT dropped)"));
            let dd_cfg = cfg.diffusion_decoder.as_ref().expect("checked above");
            let diffvae = DiffusionDecoder::load(
                &WeightMap::open(&self.weights.join("diffusion_decoder"))?,
                dd_cfg,
            )?;
            decode_diffvae_and_write(
                decoders,
                &diffvae,
                &video,
                &audio,
                decode_grid,
                &req.output_dir,
                req.frame_rate,
                req.mp4,
                req.seed + 40_000,
            )?
        } else {
            // LTX-2.5 decodes with `AUTO_TILING` (Conv VAE: 768/64 long side,
            // 80/24 frames; `helpers.py:60-97`); the diffusers lines decode whole.
            let tiling = if cfg.version == Ltx2ModelVersion::V25 {
                Some(TileSizeConfig::conv_auto(req.height, req.width).map_err(err)?)
            } else {
                None
            };
            decode_and_write(
                decoders,
                &video,
                &audio,
                decode_grid,
                &req.output_dir,
                req.frame_rate,
                req.mp4,
                tiling.as_ref(),
            )?
        };
        timings.decode_audio_s = written.decode_audio_s;
        timings.decode_video_s = written.decode_video_s;
        timings.write_s = written.write_s;
        drop((video, audio));
        self.release_transient_decoders()?;
        trim()?;
        memory.mark("decode")?;
        Ok(Ltx2Output {
            frames: written.frames,
            mp4: written.mp4,
            wav: written.wav,
            prompt_tokens: text_report.tokens,
            video_tokens: decode_grid.iter().product(),
            audio_tokens,
            text: text_report,
            timings,
            memory: memory.phases,
            dit_residency: self.residency.as_str(),
        })
    }

    /// Spark joint refiner: a normalized video latent plus the draft PCM,
    /// three stage-2 updates, then the video VAE muxed with that same PCM.
    pub fn refine_joint(
        &mut self,
        video: &CudaTensor,
        wave_planar: &[f32],
        wave_channels: usize,
        wave_rate: u32,
        prompt: &str,
        seed: u64,
        frame_rate: f64,
        out_dir: &Path,
        mp4: bool,
    ) -> Result<Ltx2Output> {
        let cfg = self.cfg.clone();
        let [b, c, f, h, w] = video.shape[..] else {
            return Err(err(format!(
                "ltx2 refine: video latent must be [1, C, F, H, W], got {:?}",
                video.shape
            )));
        };
        if b != 1 || c != cfg.transformer.in_channels || f == 0 || frame_rate <= 0.0 {
            return Err(err(format!(
                "ltx2 refine: expected [1, {}, F, H, W] at a positive frame rate, got {:?} @ {frame_rate}",
                cfg.transformer.in_channels, video.shape
            )));
        }
        let [st, sh, sw] = cfg.transformer.vae_scale_factors;
        let pixel_frames = (f - 1) * st + 1;
        let grid = [f, h, w];
        if cfg.transformer.latent_grid(pixel_frames, h * sh, w * sw) != grid {
            return Err(err(format!(
                "ltx2 refine: latent {:?} is not the {pixel_frames}x{}x{} grid",
                video.shape,
                h * sh,
                w * sw
            )));
        }
        let audio_tokens = cfg.transformer.audio_tokens(pixel_frames, frame_rate);
        if audio_tokens == 0 {
            return Err(err(
                "ltx2 refine: the clip is too short for a single audio latent",
            ));
        }

        let encoder = AudioEncoder::load(&open_audio_encoder(&self.weights)?, &cfg.audio_vae)?;
        let encoded = encoder.encode_waveform(wave_planar, wave_channels, wave_rate)?;
        drop(encoder);
        let [eb, ec, _, em] = encoded.shape[..] else {
            return Err(err(format!(
                "ltx2 refine: encoded audio {:?}, expected 4 dims",
                encoded.shape
            )));
        };
        if eb != 1 || ec != cfg.audio_vae.latent_channels || em != cfg.audio_vae.latent_mel_bins() {
            return Err(err(format!(
                "ltx2 refine: encoded audio {:?}, expected [1, {}, T, {}]",
                encoded.shape,
                cfg.audio_vae.latent_channels,
                cfg.audio_vae.latent_mel_bins()
            )));
        }
        let state = LatentState::for_version(cfg.version);
        // `encode_source_audio` conforms the AudioVAE latent in bf16
        // (`Sol-H3-Spark/runtime/stage2.py:79-91`).
        let audio = state.store(pack_audio_latent(&conform_audio_time(
            &encoded,
            audio_tokens,
        )?)?)?;
        if audio.shape[2] != cfg.transformer.audio_in_channels {
            return Err(err(format!(
                "ltx2 refine: packed audio {:?}, DiT wants {} features",
                audio.shape, cfg.transformer.audio_in_channels
            )));
        }
        // The adapter's latent reaches the refiner as bf16 (`stage2.py:57-72`).
        let video = state.store(pack_video(video)?)?;

        // The refiners run the *dev* transformer with the distilled LoRA fused
        // at 0.8 (`Sol-H3-Spark/runtime/stage2_ops/models.py:76-80`,
        // `ltx2.5-refiner/GB200/refiner_head_cp.py:336-358`).
        if !cfg.scheduler.use_dynamic_shifting {
            return Err(err(
                "ltx2 refine: the refiner runs the dev transformer config (ltx2_5_22b_dev), not the distilled one",
            ));
        }
        if names_distilled(&self.dit) {
            return Err(err(format!(
                "ltx2 refine: {} names a distilled checkpoint; the refiner fuses the distilled LoRA onto ltx-2.5-22b-dev-transformer",
                self.dit.display()
            )));
        }
        if self.lora.is_none() {
            return Err(err(
                "ltx2 refine: the distilled LoRA (0.8) is required: set FASTVIDEO_LTX2_LORA or put ltx-2.5-22b-distilled-lora-450-bf16.safetensors beside the weights",
            ));
        }
        let strength = fastvideo_models::ltx2::lora::REFINER_STRENGTH;

        let mut timings = Ltx2Timings::default();
        let mut memory = MemoryLog::start("ltx2");
        let (contexts, text_report) = self.text.encode(prompt, true)?;
        timings.text_s = text_report.seconds;
        trim()?;
        memory.mark("text")?;
        let schedule = Ltx2Schedule::distilled_stage_2_steps(3).map_err(err)?;
        crate::wan::log::info(format_args!(
            "ltx2 refine: 3 deterministic steps {:?}, sol stage-2, lora {strength} {}, grid {:?}, {} audio tokens",
            &schedule.sigmas,
            self.lora.as_deref().map(Path::display).map(|d| d.to_string()).unwrap_or_default(),
            grid,
            audio_tokens
        ));
        self.dit_for(strength)?;
        let text = {
            let model = self.model.as_ref().expect("dit");
            model.project_text(&contexts.video, &contexts.audio)?
        };
        let sigma = schedule.sigmas[0] as f32;
        // One generator seeded with the request seed, video then audio
        // (`official_compat_h3_refiner_diagnostic.py:313-342`).
        let mut noise = NoiseStream::new(seed, state == LatentState::Bf16);
        let mut draws = noise.draw(&[&video.shape, &audio.shape])?;
        let (noise_a, noise_v) = (draws.pop().expect("audio"), draws.pop().expect("video"));
        let video = renoise(&video, &noise_v, sigma, state)?;
        let audio = renoise(&audio, &noise_a, sigma, state)?;
        drop((noise_v, noise_a));
        let ropes = Ropes::new(&cfg.transformer, grid, audio_tokens, frame_rate as f32)?;
        let timer = Instant::now();
        let (video, _audio) = {
            let model = self.model.as_ref().expect("dit");
            // No stage-1 cache and no prune on the refiner.
            model.disarm_fbcache();
            model.set_prune_active(false);
            denoise_with(
                model,
                &text,
                &ropes,
                &schedule,
                video,
                audio,
                None,
                Ltx2Stage2Attn::Sol,
                state,
            )?
        };
        timings.stage2_s = timer.elapsed().as_secs_f64();
        timings.denoise_s = timings.stage2_s;
        drop(ropes);
        drop(text);
        self.report_offload("refine");
        if let Some(model) = self.model.as_ref() {
            model.release_offload_device();
        }
        trim()?;
        memory.mark("stage2")?;
        self.ensure_decoders()?;

        std::fs::create_dir_all(out_dir).map_err(|e| err(format!("{}: {e}", out_dir.display())))?;
        let timer = Instant::now();
        let wav = out_dir.join("audio.wav");
        let channel_count = u16::try_from(wave_channels)
            .map_err(|_| err("ltx2 refine: too many audio channels"))?;
        write_wav(
            &wav,
            &interleave_audio(wave_planar, wave_channels)?,
            channel_count,
            wave_rate,
        )?;
        timings.decode_audio_s = timer.elapsed().as_secs_f64();

        let timer = Instant::now();
        let fps = frame_rate.round().max(1.0) as u32;
        let mut writer = VideoWriter::spawn_with_audio(out_dir, fps, mp4, Some(&wav))?;
        let mut sink_err: Option<PipelineError> = None;
        let mut sink =
            |offset: usize, frames: &CudaTensor| -> std::result::Result<(), TensorError> {
                let (h, w) = (frames.shape[2], frames.shape[3]);
                match frames_to_rgb8(frames).and_then(|rgb| writer.push(offset, h, w, rgb)) {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        let text = e.to_string();
                        sink_err = Some(e);
                        Err(TensorError::Message(text))
                    }
                }
            };
        // Spark's explicit Conv tiles: 128/24 frames, 448/64 H, 768/64 W
        // (`stage2_ops/models.py:20-22,119-122`).
        let decoded = self
            .decoders
            .as_ref()
            .expect("decoders")
            .video
            .decode_tiled(
                &unpack_video(&video, grid)?,
                &TileSizeConfig::spark(),
                &mut sink,
            );
        match (decoded, sink_err) {
            (_, Some(e)) => return Err(e),
            (Err(e), None) => return Err(e.into()),
            (Ok(_), None) => {}
        };
        timings.decode_video_s = timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let (frames, mp4_path) = writer.finish()?;
        timings.write_s = timer.elapsed().as_secs_f64();
        self.release_transient_decoders()?;
        memory.mark("decode")?;
        Ok(Ltx2Output {
            frames,
            mp4: mp4_path,
            wav: wav.to_string_lossy().into_owned(),
            prompt_tokens: text_report.tokens,
            video_tokens: grid.iter().product(),
            audio_tokens,
            text: text_report,
            timings,
            memory: memory.phases,
            dit_residency: self.residency.as_str(),
        })
    }
}

fn open_audio_encoder(weights: &Path) -> Result<WeightMap> {
    if let Ok(raw) = std::env::var("FASTVIDEO_LTX2_AUDIO_VAE") {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Ok(WeightMap::open_files(&[path])?);
        }
        if path.is_dir() {
            return Ok(WeightMap::open(&path)?);
        }
        return Err(err(format!(
            "ltx2 audio encode: FASTVIDEO_LTX2_AUDIO_VAE {} is not a file or directory",
            path.display()
        )));
    }
    let dir = weights.join("audio_vae");
    if dir.is_dir() {
        let map = WeightMap::open(&dir)?;
        if map.has_tensor("encoder.conv_in.conv.weight")
            || map.has_tensor("audio_vae.encoder.conv_in.conv.weight")
        {
            return Ok(map);
        }
    }
    let name = "ltx-2.5-audio-vae-bf16.safetensors";
    for path in [
        weights.join("vae").join(name),
        weights.join(name),
        weights.join("audio_vae").join(name),
    ] {
        if path.is_file() {
            return Ok(WeightMap::open_files(&[path])?);
        }
    }
    Err(err(
        "ltx2 audio encode needs audio_vae/ with encoder.* (or ltx-2.5-audio-vae-bf16.safetensors, or FASTVIDEO_LTX2_AUDIO_VAE)",
    ))
}

/// Load, generate one clip, drop everything.
pub fn generate(
    paths: &Ltx2Paths,
    cfg: &Ltx2Config,
    req: &Ltx2Request,
    options: &PipelineOptions,
    observer: Option<StepObserver<'_>>,
) -> Result<Ltx2Output> {
    req.validate()?;
    let mut pipeline = Ltx2Pipeline::load(paths, cfg, options)?;
    let mut out = pipeline.generate(req, true, observer)?;
    out.timings.load_s = pipeline.load_s;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::weights;
    use super::super::keys::Layout;
    use super::*;
    use fastvideo_models::ltx2::config::{
        ltx2_19b_distilled, Ltx2AudioVaeConfig, Ltx2TransformerConfig, Ltx2VideoVaeConfig,
        Ltx2VocoderConfig,
    };

    #[test]
    fn sol_stage2_is_the_25_distilled_two_stage_default() {
        use fastvideo_models::ltx2::config::{ltx2_5_22b_dev, ltx2_5_22b_distilled};
        let d = ltx2_5_22b_distilled();
        assert!(default_sol_stage2(&d, true, None, false, false));
        assert!(default_sol_stage2(&d, true, Some(3), false, false));
        assert!(
            !default_sol_stage2(&d, true, Some(2), false, false),
            "2-step refine"
        );
        assert!(
            !default_sol_stage2(&d, false, None, false, false),
            "one stage"
        );
        assert!(
            !default_sol_stage2(&d, true, None, true, false),
            "PISA chosen"
        );
        assert!(
            !default_sol_stage2(&d, true, None, false, true),
            "dense control"
        );
        assert!(!default_sol_stage2(
            &ltx2_5_22b_dev(),
            true,
            None,
            false,
            false
        ));
        assert!(!default_sol_stage2(
            &ltx2_19b_distilled(),
            true,
            None,
            false,
            false
        ));
    }

    /// A whole LTX-2 in miniature: the latent widths are tied together the way
    /// the real ones are (4 VAE channels = DiT in/out; 2 × 2 audio features).
    fn tiny() -> Ltx2Config {
        Ltx2Config {
            transformer: Ltx2TransformerConfig {
                in_channels: 4,
                out_channels: 4,
                num_attention_heads: 2,
                attention_head_dim: 8,
                cross_attention_dim: 16,
                audio_in_channels: 4,
                audio_out_channels: 4,
                audio_num_attention_heads: 2,
                audio_attention_head_dim: 4,
                audio_cross_attention_dim: 8,
                num_layers: 1,
                caption_channels: 12,
                timestep_proj_dim: 8,
                ..Ltx2TransformerConfig::ltx2_19b()
            },
            vae: Ltx2VideoVaeConfig {
                latent_channels: 4,
                decoder_block_out_channels: vec![16, 32, 64],
                decoder_layers_per_block: vec![1, 1, 1, 1],
                patch_size: 2,
                ..Ltx2VideoVaeConfig::ltx2_19b()
            },
            audio_vae: Ltx2AudioVaeConfig {
                base_channels: 2,
                num_res_blocks: 1,
                latent_channels: 2,
                mel_bins: 8,
                ..Ltx2AudioVaeConfig::ltx2_19b()
            },
            vocoder: Ltx2VocoderConfig {
                in_channels: 16,
                hidden_channels: 64,
                upsample_kernel_sizes: vec![7, 4, 4, 4, 4],
                upsample_factors: vec![3, 2, 2, 2, 2],
                ..Ltx2VocoderConfig::ltx2_19b()
            },
            ..ltx2_19b_distilled()
        }
    }

    fn model_and_inputs(
        cfg: &Ltx2Config,
    ) -> (Ltx2Transformer, TextConditioning, Ropes, [usize; 3], usize) {
        let model = Ltx2Transformer::load(
            &weights(),
            &Keys::transformer(Layout::Diffusers),
            &cfg.transformer,
        )
        .unwrap();
        let ctx = |k: f32| {
            CudaTensor::from_vec(
                (0..3 * 12).map(|i| (i as f32 * k).sin()).collect(),
                vec![1, 3, 12],
            )
            .unwrap()
        };
        let text = model.project_text(&ctx(0.3), &ctx(0.7)).unwrap();
        let (grid, audio_tokens) = ([2usize, 2, 2], 3usize);
        let ropes = Ropes::new(&cfg.transformer, grid, audio_tokens, 24.0).unwrap();
        (model, text, ropes, grid, audio_tokens)
    }

    #[test]
    fn noise_is_seeded_packed_and_video_is_drawn_first() {
        let cfg = tiny();
        let (v, a) = initial_noise(&cfg, [2, 2, 3], 5, &mut NoiseStream::new(7, false)).unwrap();
        assert_eq!(
            (v.shape.clone(), a.shape.clone()),
            (vec![1, 12, 4], vec![1, 5, 4])
        );
        let (v2, a2) = initial_noise(&cfg, [2, 2, 3], 5, &mut NoiseStream::new(7, false)).unwrap();
        assert_eq!(&*v.host_cow().unwrap(), &*v2.host_cow().unwrap());
        assert_eq!(&*a.host_cow().unwrap(), &*a2.host_cow().unwrap());
        assert_ne!(
            &*v.host_cow().unwrap(),
            &*initial_noise(&cfg, [2, 2, 3], 5, &mut NoiseStream::new(8, false))
                .unwrap()
                .0
                .host_cow()
                .unwrap()
        );
        // One generator: the first draw is video channel 0 at token 0, and the
        // audio draws start right after the 4·12 video values.
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let all: Vec<f32> = (0..48 + 20)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        assert_eq!(v.host_cow().unwrap()[0], all[0]);
        // Packed token t, feature c ← unpacked [c, t]: token 1 feature 0 is draw 1.
        assert_eq!(v.host_cow().unwrap()[4], all[1]);
        // Audio [C=2, L=5, M=2] → token 0 = (c0 m0, c0 m1, c1 m0, c1 m1).
        let ah = a.host_cow().unwrap();
        assert_eq!((ah[0], ah[1], ah[2]), (all[48], all[49], all[48 + 10]));
        let mean = all.iter().sum::<f32>() / all.len() as f32;
        assert!(mean.abs() < 0.5);
    }

    #[test]
    fn generate_decode_helper_drops_dit() {
        let cfg = tiny();
        let (model, _, _, _, _) = model_and_inputs(&cfg);
        let mut resident = Some(model);
        let mut loaded = Some(0.0f32);
        release_dit_for_decode(&mut resident, &mut loaded);
        assert!(resident.is_none());
        assert!(loaded.is_none());
    }

    /// The ancestral update against the step written out on the host, with the
    /// model evaluated at `1000·σ_i`: float32 against the f64 reference step,
    /// bf16 against the element-wise `ltx_core` reference.
    #[test]
    fn ancestral_one_step_matches_host_reference() {
        use fastvideo_models::ltx2::schedule::{bf16_round, ltx_core_ancestral_step};
        let cfg = tiny();
        let (model, text, ropes, grid, audio_tokens) = model_and_inputs(&cfg);
        let (video, audio) =
            initial_noise(&cfg, grid, audio_tokens, &mut NoiseStream::new(11, false)).unwrap();
        let schedule = Ltx2Schedule::distilled();
        let opts = AncestralOpts {
            eta: 1.0,
            s_noise: 1.0,
            noise_seed: 99,
        };
        let i = 4usize;
        let (sigma, sigma_next) = (schedule.sigmas[i], schedule.sigmas[i + 1]);
        let (vv, _) = model
            .forward(
                &video,
                &audio,
                &text,
                schedule.timestep_f32(i),
                &ropes,
                None,
            )
            .unwrap();
        let x = video.host_cow().unwrap().into_owned();
        let v = vv.host_cow().unwrap().into_owned();
        let draws = NoiseStream::new(99, false).draw(&[&video.shape]).unwrap();
        let eps = draws[0].host_cow().unwrap().into_owned();

        let mut want = x.clone();
        let den: Vec<f32> = x
            .iter()
            .zip(&v)
            .map(|(x, v)| Ltx2Schedule::denoised_from_velocity(*x, *v, sigma))
            .collect();
        Ltx2Schedule::ancestral_step(&mut want, &den, sigma, sigma_next, 1.0, 1.0, Some(&eps));
        let got = ancestral_update(
            &video,
            &vv,
            sigma,
            sigma_next,
            opts,
            Some(&draws[0]),
            LatentState::F32,
        )
        .unwrap();
        for (a, b) in got.host_cow().unwrap().iter().zip(&want) {
            assert!((a - b).abs() < 1e-4 * (1.0 + b.abs()), "f32: {a} vs {b}");
        }

        let xb = LatentState::Bf16.store(video.clone()).unwrap();
        let eb = LatentState::Bf16.store(draws[0].clone()).unwrap();
        let got = ancestral_update(
            &xb,
            &vv,
            sigma,
            sigma_next,
            opts,
            Some(&eb),
            LatentState::Bf16,
        )
        .unwrap();
        let got = got.host_cow().unwrap();
        for (k, g) in got.iter().enumerate() {
            let w = ltx_core_ancestral_step(
                x[k],
                v[k],
                eps[k],
                sigma as f32,
                sigma_next as f32,
                1.0,
                1.0,
            );
            assert_eq!(*g, bf16_round(*g), "stored as bf16");
            // One bf16 ulp: the device sums may fuse differently from the host's.
            assert!(
                (g - w).abs() <= 1e-2 * (1.0 + w.abs()),
                "bf16[{k}]: {g} vs {w}"
            );
        }
        // The terminal step is x0 itself.
        let last =
            ancestral_update(&xb, &vv, 0.421875, 0.0, opts, None, LatentState::Bf16).unwrap();
        for ((g, x), v) in last.host_cow().unwrap().iter().zip(&x).zip(&v) {
            let w = bf16_round(bf16_round(*x) - bf16_round(*v) * 0.421875);
            assert!((g - w).abs() <= 1e-2 * (1.0 + w.abs()));
        }
    }

    #[test]
    fn bf16_euler_update_is_the_ltx_core_round_trip() {
        use fastvideo_models::ltx2::schedule::{bf16_round, ltx_core_euler_step};
        let schedule = Ltx2Schedule::distilled_stage_2();
        let x: Vec<f32> = (0..64)
            .map(|i| bf16_round((i as f32 * 0.37).sin() * 2.0))
            .collect();
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.91).cos() * 1.5).collect();
        let xt = CudaTensor::from_vec(x.clone(), vec![1, 64]).unwrap();
        let vt = CudaTensor::from_vec(v.clone(), vec![1, 64]).unwrap();
        for i in 0..schedule.num_steps() {
            let got = euler_update(&xt, &vt, &schedule, i, LatentState::Bf16).unwrap();
            let (s, n) = (schedule.sigmas[i] as f32, schedule.sigmas[i + 1] as f32);
            for (k, g) in got.host_cow().unwrap().iter().enumerate() {
                let w = ltx_core_euler_step(x[k], v[k], s, n);
                assert_eq!(*g, bf16_round(*g));
                assert!(
                    (g - w).abs() <= 1e-2 * (1.0 + w.abs()),
                    "step {i}[{k}]: {g} vs {w}"
                );
            }
            // Float32 state is the plain Euler update.
            let f = euler_update(&xt, &vt, &schedule, i, LatentState::F32).unwrap();
            let dt = schedule.dt(i) as f32;
            for (k, g) in f.host_cow().unwrap().iter().enumerate() {
                assert!((g - (x[k] + dt * v[k])).abs() < 1e-6);
            }
        }
    }

    /// `GaussianNoiser`: `lerp(x, ε, σ)` evaluated from the ε end, then bf16.
    #[test]
    fn renoise_is_torch_lerp_then_the_state_dtype() {
        let x = CudaTensor::from_vec(vec![0.5, -1.25, 2.0], vec![1, 3]).unwrap();
        let e = CudaTensor::from_vec(vec![1.0, 0.25, -0.75], vec![1, 3]).unwrap();
        let sigma = 0.909375f32;
        let got = renoise(&x, &e, sigma, LatentState::F32).unwrap();
        for ((g, x), e) in got
            .host_cow()
            .unwrap()
            .iter()
            .zip([0.5f32, -1.25, 2.0])
            .zip([1.0f32, 0.25, -0.75])
        {
            assert!((g - (e - (e - x) * (1.0 - sigma))).abs() < 1e-6);
        }
        let b = renoise(&x, &e, sigma, LatentState::Bf16).unwrap();
        for v in b.host_cow().unwrap().iter() {
            assert_eq!(*v, fastvideo_models::ltx2::schedule::bf16_round(*v));
        }
    }

    /// LTX-2.5 draws in the patchified shapes, in bf16, from one generator that
    /// the stage-2 renoise continues (`distilled.py:217-218,294-313`).
    #[test]
    fn ltx25_noise_is_token_major_bf16_and_one_stream_across_stages() {
        use fastvideo_models::ltx2::schedule::bf16_round;
        let cfg = Ltx2Config {
            version: Ltx2ModelVersion::V25,
            ..tiny()
        };
        let mut stream = NoiseStream::new(7, true);
        let (v, a) = initial_noise(&cfg, [2, 2, 3], 5, &mut stream).unwrap();
        assert_eq!(
            (v.shape.clone(), a.shape.clone()),
            (vec![1, 12, 4], vec![1, 5, 4])
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let all: Vec<f32> = (0..48 + 20 + 48 + 20)
            .map(|_| bf16_round(rng.sample::<f32, _>(StandardNormal)))
            .collect();
        // Token-major: the first four draws are token 0's four channels.
        assert_eq!(&v.host_cow().unwrap()[..], &all[..48]);
        assert_eq!(&a.host_cow().unwrap()[..], &all[48..68]);
        // Stage 2 keeps drawing from the same generator: video, then audio.
        let d = stream.draw(&[&v.shape, &a.shape]).unwrap();
        assert_eq!(&d[0].host_cow().unwrap()[..], &all[68..116]);
        assert_eq!(&d[1].host_cow().unwrap()[..], &all[116..]);
    }

    #[test]
    fn ltx25_generation_is_the_unguided_distilled_pipeline() {
        let distilled = fastvideo_models::ltx2::ltx2_5_22b_distilled();
        assert!(check_generate_contract(&distilled, false).is_ok());
        assert!(check_generate_contract(&distilled, true).is_err());
        assert!(check_generate_contract(&fastvideo_models::ltx2::ltx2_5_22b_dev(), false).is_err());
        // Other lines keep their own (guided) samplers.
        assert!(check_generate_contract(&fastvideo_models::ltx2::ltx2_19b(), true).is_ok());
        assert!(check_generate_contract(&fastvideo_models::ltx2::ltx2_23_22b(), true).is_ok());
        assert_eq!(
            LatentState::for_version(Ltx2ModelVersion::V25),
            LatentState::Bf16
        );
        assert_eq!(
            LatentState::for_version(Ltx2ModelVersion::V23),
            LatentState::F32
        );
    }

    #[test]
    fn the_refiner_refuses_a_distilled_dit_path() {
        assert!(names_distilled(Path::new(
            "/m/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
        )));
        assert!(names_distilled(Path::new(
            "/m/LTX-2.5-Distilled/transformer"
        )));
        assert!(!names_distilled(Path::new(
            "/m/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
        )));
        assert!(!names_distilled(Path::new("/m/LTX-2.5/transformer")));
    }

    #[test]
    fn the_text_cache_key_names_the_feature_normalisation() {
        let v25 = fastvideo_models::ltx2::ltx2_5_22b_distilled();
        let v20 = fastvideo_models::ltx2::ltx2_19b();
        assert_eq!(
            TextEncoder::text_encoder_kind(&v25),
            b"gemma4-12b/per-token-rms"
        );
        assert_eq!(TextEncoder::text_encoder_kind(&v20), b"gemma3-12b");
        assert_eq!(
            TextEncoder::text_encoder_kind(&fastvideo_models::ltx2::ltx2_23_22b()),
            b"gemma3-12b/per-token-rms"
        );
    }

    /// The prune's compensation is per pass: under CFG, the conditional pass of
    /// a prune step is compensated from the conditional pass before it, never
    /// from the unconditional one (`token_prune.py`, keyed by `cache_key`).
    #[test]
    fn prune_compensation_is_kept_per_pass() {
        let cfg = tiny();
        let (model, text, ropes, _, _) = model_and_inputs(&cfg);
        let ctx = |k: f32| {
            CudaTensor::from_vec(
                (0..3 * 12).map(|i| (i as f32 * k).cos()).collect(),
                vec![1, 3, 12],
            )
            .unwrap()
        };
        let uncond = model.project_text(&ctx(0.5), &ctx(0.9)).unwrap();
        let x = |k: f32, n: usize| {
            CudaTensor::from_vec(
                (0..n * 4).map(|i| (i as f32 * k).sin()).collect(),
                vec![1, n, 4],
            )
            .unwrap()
        };
        let (v, a) = (x(0.21, 8), x(0.43, 3));
        let run = |with_uncond: bool| {
            model.enable_midpoint_prune();
            model.set_prune_active(true);
            let mut out = None;
            for step in 0..2 {
                model.arm_prune_step(step);
                let t = 900.0 - 100.0 * step as f32;
                let cond = model.forward(&v, &a, &text, t, &ropes, None).unwrap();
                if with_uncond {
                    model.forward(&v, &a, &uncond, t, &ropes, None).unwrap();
                }
                out = Some(cond.0.host_cow().unwrap().into_owned());
            }
            out.unwrap()
        };
        // Step 1 prunes (`PRUNE_STEPS`); its cond pass must not see the uncond hidden.
        assert!(fastvideo_models::ltx2::prunes_step(1));
        let single = run(false);
        let paired = run(true);
        assert_eq!(single, paired);
    }

    #[test]
    fn res2s_two_steps_stays_finite() {
        let cfg = tiny();
        let (model, text, ropes, grid, audio_tokens) = model_and_inputs(&cfg);
        let (video, audio) =
            initial_noise(&cfg, grid, audio_tokens, &mut NoiseStream::new(3, false)).unwrap();
        let schedule = Ltx2Schedule::from_sigmas(&[1.0, 0.5], 1000);
        let (got_v, got_a) = denoise_res2s(
            &model,
            &text,
            None,
            &ropes,
            &schedule,
            1.0,
            1.0,
            video,
            audio,
            None,
            Ltx2Stage2Attn::Off,
        )
        .unwrap();
        assert!(got_v.host_cow().unwrap().iter().all(|v| v.is_finite()));
        assert!(got_a.host_cow().unwrap().iter().all(|v| v.is_finite()));
        assert_eq!(
            fastvideo_models::ltx2::hq::res2s_num_calls(schedule.num_steps()),
            3
        );
    }

    #[test]
    fn denoise_is_eight_euler_steps_on_the_distilled_sigmas() {
        let cfg = tiny();
        let (model, text, ropes, grid, audio_tokens) = model_and_inputs(&cfg);
        let (video, audio) =
            initial_noise(&cfg, grid, audio_tokens, &mut NoiseStream::new(3, false)).unwrap();
        let schedule = Ltx2Schedule::distilled();
        let mut seen = Vec::new();
        let mut obs = |i: usize, v: &CudaTensor, _: &CudaTensor, _: f64| -> Result<()> {
            seen.push((i, v.host_cow()?.into_owned()));
            Ok(())
        };
        let (got_v, got_a) = denoise(
            &model,
            &text,
            &ropes,
            &schedule,
            video.clone(),
            audio.clone(),
            Some(&mut obs),
            Ltx2Stage2Attn::Off,
        )
        .unwrap();
        assert_eq!(
            seen.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );

        let sigmas = [
            1.0f64, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
        ];
        let (mut xv, mut xa) = (
            video.host_cow().unwrap().into_owned(),
            audio.host_cow().unwrap().into_owned(),
        );
        for i in 0..8 {
            let t = (sigmas[i] as f32) * 1000.0;
            let (vv, va) = model
                .forward(
                    &CudaTensor::from_vec(xv.clone(), video.shape.clone()).unwrap(),
                    &CudaTensor::from_vec(xa.clone(), audio.shape.clone()).unwrap(),
                    &text,
                    t,
                    &ropes,
                    None,
                )
                .unwrap();
            let dt = (sigmas[i + 1] - sigmas[i]) as f32;
            xv.iter_mut()
                .zip(vv.host_cow().unwrap().iter())
                .for_each(|(x, v)| *x += dt * v);
            xa.iter_mut()
                .zip(va.host_cow().unwrap().iter())
                .for_each(|(x, v)| *x += dt * v);
            for (a, b) in seen[i].1.iter().zip(&xv) {
                assert!((a - b).abs() < 1e-5, "step {i}: {a} vs {b}");
            }
        }
        for (a, b) in got_a.host_cow().unwrap().iter().zip(&xa) {
            assert!((a - b).abs() < 1e-5);
        }
        assert_eq!(got_v.shape, video.shape);
    }

    #[test]
    fn decoding_writes_a_wav_first_and_every_frame() {
        let cfg = tiny();
        let map = weights();
        let dec = Decoders {
            video: VideoDecoder::load(&map, &cfg.vae).unwrap(),
            audio: AudioDecoder::load(&map, &cfg.audio_vae).unwrap(),
            vocoder: Vocoder::load(&map, &cfg.vocoder).unwrap(),
            upsampler: None,
        };
        let (video, audio) =
            initial_noise(&cfg, [2, 2, 2], 3, &mut NoiseStream::new(1, false)).unwrap();
        let dir = std::env::temp_dir().join(format!("fv-ltx2-pipeline-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let out =
            decode_and_write(&dec, &video, &audio, [2, 2, 2], &dir, 24.0, false, None).unwrap();
        // 2 latent frames → 9 frames of 32x32 (×8 by the VAE, ×2 by its patch… in this tiny config ×16).
        assert_eq!(out.frames.len(), 9);
        assert!(out.frames.iter().all(|p| Path::new(p).is_file()));
        assert!(out.frames[8].ends_with("frame-008.png"));
        assert_eq!(out.mp4, None);
        // 3 audio latents → 9 mel frames → 9·48 stereo samples of 16-bit PCM after a 44-byte header.
        let wav = std::fs::read(&out.wav).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(wav.len(), 44 + 9 * 48 * 2 * 2);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            24_000
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn requests_outside_the_model_card_are_refused() {
        let cfg = ltx2_19b_distilled();
        let ok = Ltx2Request::new(&cfg, "a cat", "/tmp/x");
        assert!(ok.validate().is_ok());
        assert_eq!(
            (ok.width, ok.height, ok.num_frames, ok.seed),
            (768, 512, 121, 10)
        );
        assert!(Ltx2Request {
            height: 500,
            ..ok.clone()
        }
        .validate()
        .is_err());
        assert!(Ltx2Request {
            num_frames: 120,
            ..ok.clone()
        }
        .validate()
        .is_err());
        assert!(Ltx2Request {
            prompt: "  ".into(),
            ..ok.clone()
        }
        .validate()
        .is_err());
        assert!(Ltx2Request {
            frame_rate: 0.0,
            ..ok.clone()
        }
        .validate()
        .is_err());
        // Two-stage needs multiples of 64 (half-res still lands on the VAE grid).
        assert!(Ltx2Request {
            two_stage: true,
            ..ok.clone()
        }
        .validate()
        .is_ok());
        assert!(Ltx2Request {
            two_stage: true,
            height: 544,
            width: 960,
            ..ok
        }
        .validate()
        .is_err());
    }
    /// Hit, miss and bypass, with the expensive part replaced by a counter.
    #[test]
    fn the_cache_computes_once_per_prompt_and_a_bypass_neither_reads_nor_writes() {
        let dir =
            std::env::temp_dir().join(format!("fv-ltx2-pipeline-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = TextCache::new(&dir);
        let padded = PaddedPrompt::from_ids(&[2, 5, 9], 8).unwrap();
        let calls = std::cell::Cell::new(0usize);
        let compute = || -> Result<CachedContexts> {
            calls.set(calls.get() + 1);
            Ok(CachedContexts {
                video: CudaTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])?,
                audio: CudaTensor::from_vec(vec![5.0, 6.0], vec![1, 2, 1])?,
            })
        };
        // Bypassed: computed, nothing written.
        assert_eq!(
            cached_or(Some(&cache), None, &padded, compute).unwrap().1,
            CacheOutcome::Off
        );
        assert!(!cache.path("k").exists());
        assert_eq!(
            cached_or(Some(&cache), Some("k"), &padded, compute)
                .unwrap()
                .1,
            CacheOutcome::Miss
        );
        let (hit, outcome) = cached_or(Some(&cache), Some("k"), &padded, compute).unwrap();
        assert_eq!(
            (outcome, calls.get()),
            (CacheOutcome::Hit, 2),
            "the third call must not compute"
        );
        assert_eq!(&*hit.video.host_cow().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        // A truncated entry: a miss that recomputes and repairs the file.
        let whole = std::fs::read(cache.path("k")).unwrap();
        std::fs::write(cache.path("k"), &whole[..whole.len() - 5]).unwrap();
        assert_eq!(
            cached_or(Some(&cache), Some("k"), &padded, compute)
                .unwrap()
                .1,
            CacheOutcome::Miss
        );
        assert_eq!(std::fs::read(cache.path("k")).unwrap(), whole);
        // No cache at all.
        assert_eq!(
            cached_or(None, Some("k"), &padded, compute).unwrap().1,
            CacheOutcome::Off
        );
        assert_eq!(calls.get(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn auto_residency_needs_a_device_with_room_and_the_environment_overrides() {
        let gib = |n: u64| n << 30;
        let auto = TextResidency::Auto;
        assert!(
            auto.resolve(None, Some(gib(50)), gib(34)).unwrap(),
            "96 GB card, DiT loaded: room"
        );
        assert!(
            !auto.resolve(None, Some(gib(20)), gib(34)).unwrap(),
            "48 GB card: stream"
        );
        assert!(
            !auto.resolve(None, None, gib(34)).unwrap(),
            "no device: stream"
        );
        assert!(auto.resolve(Some("resident"), Some(0), gib(34)).unwrap());
        assert!(!auto
            .resolve(Some(" Streamed "), Some(gib(90)), gib(34))
            .unwrap());
        assert!(TextResidency::Resident
            .resolve(Some("auto"), Some(gib(90)), gib(34))
            .unwrap());
        assert!(!TextResidency::Streamed
            .resolve(None, Some(gib(90)), gib(34))
            .unwrap());
        assert!(
            TextResidency::Resident
                .resolve(Some(""), None, gib(34))
                .unwrap(),
            "an empty variable is unset"
        );
        assert!(auto.resolve(Some("maybe"), None, 0).is_err());
    }

    #[test]
    fn the_resident_text_path_is_sized_from_the_published_shapes() {
        let paths = Ltx2Paths {
            weights: "/nonexistent".into(),
            dit: "/nonexistent".into(),
            text: None,
        };
        let enc = TextEncoder::new(&paths, &ltx2_19b_distilled(), &PipelineOptions::default());
        // CPU build: float32 widths. Gemma's 48 layers of projections are
        // 10.76 B parameters, the connectors 1.43 B (docs/ports/ltx2.md §g).
        let params = enc.resident_bytes() / 4;
        assert_eq!(
            params,
            48 * 224_133_120 + (4 * 12 * 3840 * 3840 + 188_160 * 3840)
        );
        assert_eq!(enc.mode(), "undecided");
        let mut streamed =
            TextEncoder::new(&paths, &ltx2_19b_distilled(), &PipelineOptions::default());
        streamed.prefer_streamed_for_two_stage();
        assert_eq!(streamed.mode(), "streamed");
        streamed.unload_resident();
        assert_eq!(streamed.mode(), "streamed");
        assert_eq!(paths.text_root(), Path::new("/nonexistent"));
        let slim = Ltx2Paths {
            text: Some("/slim".into()),
            ..paths
        };
        assert_eq!(slim.text_root(), Path::new("/slim"));
    }
}
