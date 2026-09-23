//! Wan inference pipeline: UMT5 → DiT sampling → VAE decode → PNG / optional MP4.
//!
//! Behavioral parity target: Candle `fastvideo_models::wan::pipeline` (MoE, I2V, dual CFG).

use std::path::Path;
use std::process::Command;

use fastvideo_models::schedulers::{
    DmdSchedule, FlowUniPCMultistepScheduler, RcmSchedule, UniPcTerm, FAST_WAN_1_3B_DMD_STEPS,
    RCM_SIGMA_MAX_T2V,
};
use fastvideo_models::wan::{
    i2v_first_frame_mask, moe_expert, MoeExpert, Umt5Config, WanVaeConfig, WanVideoArchConfig,
};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use thiserror::Error;

use super::clip::{ClipVision, ClipVisionConfig};
use super::tensor::{CudaTensor, Result as TensorResult, TensorError};
use super::transformer::WanTransformer3D;
use super::umt5::{pad_prompt_embeds, Umt5Encoder};
use super::vae::AutoencoderKlWan;
use super::weights::WeightMap;

/// TAEHV replaces the Wan VAE decode when `FASTVIDEO_TAEHV_WEIGHTS` names a
/// directory holding `taew2_1.safetensors`.
///
/// A path rather than a boolean, because the weights ship separately from the
/// Wan checkpoint (they live in madebyollin/taehv, not the Diffusers repo) and
/// there is no sensible place to guess. A set-but-unloadable path is an error
/// rather than a silent fall back to the Wan VAE: asking for TAEHV and getting
/// a 3.9s Wan decode would look exactly like TAEHV being slow.
fn load_taehv() -> Result<Option<super::taehv::TaeHv>> {
    let dir = std::env::var("FASTVIDEO_TAEHV_WEIGHTS").unwrap_or_default();
    if dir.is_empty() {
        return Ok(None);
    }
    let map = WeightMap::from_dir(Path::new(&dir))
        .map_err(|e| PipelineError::Message(format!("FASTVIDEO_TAEHV_WEIGHTS={dir}: {e}")))?;
    let tae = super::taehv::TaeHv::load(&map)
        .map_err(|e| PipelineError::Message(format!("FASTVIDEO_TAEHV_WEIGHTS={dir}: {e}")))?;
    super::log::info(format_args!("vae=taehv ({dir})"));
    Ok(Some(tae))
}

/// `FASTVIDEO_DEVICE_STATS=1`: log host↔device transfer counts after `generate()`.
fn log_device_stats_if_enabled() {
    if !super::envflag::bool_flag("FASTVIDEO_DEVICE_STATS", false) {
        return;
    }
    let s = super::stats::snapshot();
    super::log::info(format_args!(
        "device stats: h2d {} ({} MiB) d2h {} ({} MiB) kernel launches {} host fallbacks {:?}",
        s.h2d_count,
        s.h2d_bytes >> 20,
        s.d2h_count,
        s.d2h_bytes >> 20,
        s.launches,
        s.host_fallbacks
    ));
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, PipelineError>;

fn cuda_context_live() -> bool {
    #[cfg(feature = "cuda")]
    {
        super::device::global_device().is_some()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub prompt: String,
    pub negative_prompt: String,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    pub output_dir: String,
    pub tiny: bool,
    pub is_dmd: bool,
    /// TurboWan rCM sampler.
    pub is_rcm: bool,
    pub flow_shift: f64,
    pub dmd_steps: Option<Vec<i32>>,
    /// rCM `sigma_max` (T2V 80 / I2V 200).
    pub rcm_sigma_max: Option<f64>,
    pub tokenizer_path: Option<String>,
    /// First-frame path for I2V (PNG/JPEG).
    pub image_path: Option<String>,
    /// Control / reference video or frame for Fun Control / Lucy (PNG/JPEG).
    pub control_path: Option<String>,
    /// Wan 2.2 low-noise CFG. Falls back to `guidance_scale`.
    pub guidance_scale_2: Option<f32>,
    pub boundary_ratio: Option<f32>,
    /// When true, mux PNG frames to `output.mp4` via ffmpeg if available.
    pub save_video: bool,
    pub fps: u32,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            prompt: "a cat walking".into(),
            negative_prompt: String::new(),
            height: 480,
            width: 832,
            num_frames: 81,
            num_inference_steps: 50,
            guidance_scale: 5.0,
            seed: 42,
            output_dir: "out".into(),
            tiny: false,
            is_dmd: false,
            is_rcm: false,
            flow_shift: 5.0,
            dmd_steps: None,
            rcm_sigma_max: None,
            tokenizer_path: None,
            image_path: None,
            control_path: None,
            guidance_scale_2: None,
            boundary_ratio: None,
            save_video: false,
            fps: 16,
        }
    }
}

/// Which heavy components [`WanPipeline::load_with`] materializes.
#[derive(Debug, Clone, Copy)]
pub struct LoadParts {
    /// UMT5-XXL (~21GB F32 on disk). Skip it when prompt embeddings are
    /// precomputed (see [`WanPipeline::denoise`]): it is by far the largest
    /// component and the only one a validation run on a 24GB card can't hold.
    pub text_encoder: bool,
}

impl Default for LoadParts {
    fn default() -> Self {
        Self { text_encoder: true }
    }
}

/// Per-step progress passed to a [`WanPipeline::denoise`] observer. The
/// observer runs after each step's latent update; returning `Err` aborts the
/// run immediately (fail-fast on NaN, time budget, etc.).
pub struct DenoiseStep<'a> {
    /// 0-based step index.
    pub index: usize,
    pub total: usize,
    pub timestep: f32,
    pub latents: &'a CudaTensor,
}

pub type StepObserver<'o> = dyn FnMut(&DenoiseStep<'_>) -> Result<()> + 'o;

pub struct WanPipeline {
    text: Option<Umt5Encoder>,
    dit: WanTransformer3D,
    dit_2: Option<WanTransformer3D>,
    vae: AutoencoderKlWan,
    /// `FASTVIDEO_TAEHV_WEIGHTS=<dir>`: decode through the tiny autoencoder
    /// instead of the Wan VAE. Loaded alongside rather than instead of it, so
    /// the Wan VAE stays available and the choice is per-decode.
    taehv: Option<super::taehv::TaeHv>,
    clip: Option<ClipVision>,
    tiny: bool,
    boundary_ratio: Option<f32>,
    /// Preset name for clear errors (Fun Control / Lucy).
    preset: String,
}

impl WanPipeline {
    /// Zero-weight tiny graph with eager DiT + VAE decode.
    pub fn tiny() -> Self {
        Self {
            text: Some(Umt5Encoder::zeros(Umt5Config::tiny())),
            dit: WanTransformer3D::zeros(WanVideoArchConfig::tiny()),
            dit_2: None,
            vae: AutoencoderKlWan::zeros(WanVaeConfig::tiny()),
            taehv: None,
            clip: None,
            tiny: true,
            boundary_ratio: None,
            preset: "tiny".into(),
        }
    }

    /// T2V pipeline from already-built components (e.g. seeded random weights
    /// for parity tests). Uses real latent-shape logic, unlike [`Self::tiny`].
    pub fn from_parts(
        text: Option<Umt5Encoder>,
        dit: WanTransformer3D,
        vae: AutoencoderKlWan,
    ) -> Self {
        let boundary_ratio = dit.cfg.boundary_ratio;
        Self {
            text,
            dit,
            dit_2: None,
            vae,
            taehv: None,
            clip: None,
            tiny: false,
            boundary_ratio,
            preset: "custom".into(),
        }
    }

    /// Load Diffusers layout using a registry preset (arch-aware).
    pub fn load(root: &Path, preset: &str) -> Result<Self> {
        Self::load_with(root, preset, LoadParts::default())
    }

    /// [`Self::load`] with control over which components are materialized.
    pub fn load_with(root: &Path, preset: &str, parts: LoadParts) -> Result<Self> {
        let cfg = WanVideoArchConfig::from_preset(preset);
        let dit = WeightMap::from_dir(&root.join("transformer"))?;
        let vae_cfg = if cfg.out_channels == 48 {
            WanVaeConfig {
                z_dim: 48,
                ..WanVaeConfig::wan_2_1()
            }
        } else {
            WanVaeConfig::wan_2_1()
        };
        let vae = WeightMap::from_dir(&root.join("vae"))?;
        let text = if parts.text_encoder {
            let text_dir = if root.join("text_encoder").is_dir() {
                root.join("text_encoder")
            } else {
                root.join("text_encoder_2")
            };
            let map = WeightMap::from_dir(&text_dir)?;
            Some(Umt5Encoder::load(Umt5Config::xxl(), &map)?)
        } else {
            None
        };

        let dit_2 = if root.join("transformer_2").is_dir() {
            let map = WeightMap::from_dir(&root.join("transformer_2"))?;
            Some(WanTransformer3D::load(cfg.clone(), &map)?)
        } else {
            None
        };

        let clip = if root.join("image_encoder").is_dir() && cfg.is_i2v() {
            let map = WeightMap::from_dir(&root.join("image_encoder"))?;
            Some(ClipVision::load(ClipVisionConfig::vit_h_14(), &map)?)
        } else {
            None
        };

        let boundary_ratio = cfg.boundary_ratio;
        let taehv = load_taehv()?;
        Ok(Self {
            text,
            dit: WanTransformer3D::load(cfg, &dit)?,
            dit_2,
            vae: AutoencoderKlWan::load(vae_cfg, &vae)?,
            taehv,
            clip,
            tiny: false,
            boundary_ratio,
            preset: preset.to_string(),
        })
    }

    /// Backward-compatible 1.3B load.
    pub fn load_1_3b(root: &Path) -> Result<Self> {
        Self::load(root, "wan_t2v_1_3b")
    }

    /// `[neg, prompt]` UMT5 embeddings padded to `text_len`: `[2, text_len, 4096]`.
    pub fn encode_prompt_embeds(&self, cfg: &GenerateConfig) -> Result<CudaTensor> {
        let text_len = self.dit.cfg.text_len;
        let text = self.text.as_ref().ok_or_else(|| {
            PipelineError::Message(
                "text encoder not loaded (LoadParts::text_encoder=false); pass precomputed embeddings"
                    .into(),
            )
        })?;
        if let Some(tokenizer) = cfg.tokenizer_path.as_ref() {
            let (prompt_ids, prompt_len) = tokenize_prompt(tokenizer, &cfg.prompt, text_len)?;
            let (neg_ids, neg_len) = tokenize_prompt(tokenizer, &cfg.negative_prompt, text_len)?;
            let prompt_embeds = text.forward(&prompt_ids, 1, prompt_ids.len())?;
            let neg_embeds = text.forward(&neg_ids, 1, neg_ids.len())?;
            let prompt_embeds = pad_prompt_embeds(&prompt_embeds, &[prompt_len], text_len)?;
            let neg_embeds = pad_prompt_embeds(&neg_embeds, &[neg_len], text_len)?;
            return Ok(CudaTensor::cat(&[&neg_embeds, &prompt_embeds], 0)?);
        }
        if self.tiny {
            let seq = text_len.min(8);
            let dummy: Vec<u32> = (0..seq).map(|i| (i % 10) as u32).collect();
            let prompt_embeds = text.forward(&dummy, 1, seq)?;
            let neg_embeds = prompt_embeds.clone();
            let prompt_embeds = pad_prompt_embeds(&prompt_embeds, &[seq], text_len)?;
            let neg_embeds = pad_prompt_embeds(&neg_embeds, &[seq], text_len)?;
            return Ok(CudaTensor::cat(&[&neg_embeds, &prompt_embeds], 0)?);
        }
        Err(PipelineError::Message(
            "real generate needs tokenizer.json next to the Diffusers weights".into(),
        ))
    }

    fn encode_i2v_condition(
        &self,
        image_path: &str,
        height: usize,
        width: usize,
        z_t: usize,
        z_h: usize,
        z_w: usize,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let video = load_rgb_frame(image_path, height, width)?;
        let encoded = self.vae.encode_video(&video)?;
        let encoded = self.vae.normalize_latents(&encoded)?;
        let first = encoded.narrow(2, 0, 1)?;
        let (_b, c, _, eh, ew) = (
            first.shape[0],
            first.shape[1],
            first.shape[2],
            first.shape[3],
            first.shape[4],
        );
        if eh != z_h || ew != z_w {
            return Err(PipelineError::Message(format!(
                "I2V VAE latent spatial {eh}x{ew} does not match expected {z_h}x{z_w}"
            )));
        }
        let rest_t = z_t.saturating_sub(1);
        let cond = if rest_t == 0 {
            first
        } else {
            let rest = CudaTensor::zeros(&[1, c, rest_t, z_h, z_w]);
            CudaTensor::cat(&[&first, &rest], 2)?
        };
        let mask = CudaTensor::from_vec(
            i2v_first_frame_mask(z_t, z_h, z_w),
            vec![1, 4, z_t, z_h, z_w],
        )?;
        Ok((mask, cond))
    }

    pub fn generate(&mut self, cfg: &GenerateConfig) -> Result<Vec<String>> {
        let _gen = super::log::StepTimer::start("generate");
        let needs_control = matches!(
            self.preset.as_str(),
            "wan_fun_1_3b_control" | "lucy_edit_dev"
        ) && !self.tiny;
        if needs_control && cfg.control_path.is_none() && cfg.image_path.is_none() {
            return Err(PipelineError::Message(format!(
                "preset `{}` needs --control <png|jpeg> (or --image) for control/edit \
                 conditioning. Pass a reference frame to enable the Fun Control / Lucy path.",
                self.preset
            )));
        }

        let (z_c, z_t, z_h, z_w) = self.latent_shape(cfg);
        super::log::info(format_args!(
            "generate preset={} tiny={} {}x{} frames={} steps={} dmd={} rcm={} latent=1x{}x{}x{}x{} \
             resident={} cuda={}",
            self.preset,
            self.tiny,
            cfg.width,
            cfg.height,
            cfg.num_frames,
            cfg.num_inference_steps,
            cfg.is_dmd,
            cfg.is_rcm,
            z_c,
            z_t,
            z_h,
            z_w,
            super::resident::residency_enabled(),
            cuda_context_live(),
        ));
        if cfg.is_rcm {
            warn_sla_backend();
        }
        let control_src = cfg
            .control_path
            .as_ref()
            .or(cfg.image_path.as_ref())
            .map(|s| s.as_str());

        let i2v = !self.tiny && self.dit.cfg.in_channels > self.dit.cfg.out_channels;
        if i2v && cfg.image_path.is_none() && control_src.is_none() {
            return Err(PipelineError::Message(
                "I2V generate needs --image <png|jpeg> for 36-channel latent packing".into(),
            ));
        }

        let mut latents = self.initial_latents(cfg)?;

        let i2v_pack = if i2v {
            let image = cfg
                .image_path
                .as_deref()
                .or(control_src)
                .ok_or_else(|| PipelineError::Message("missing --image".into()))?;
            Some(self.encode_i2v_condition(image, cfg.height, cfg.width, z_t, z_h, z_w)?)
        } else if needs_control {
            // Fun Control / Lucy with T2V-width DiT: inject control into first latent frame.
            let path = control_src.ok_or_else(|| {
                PipelineError::Message("control/lucy needs --control or --image".into())
            })?;
            let (_mask, cond) =
                self.encode_i2v_condition(path, cfg.height, cfg.width, z_t, z_h, z_w)?;
            // cond is [1,C,T,H,W] with C==z_c typically after VAE; blend frame 0.
            if cond.shape.get(1).copied() == Some(z_c) {
                let first = cond.narrow(2, 0, 1)?;
                let rest = if z_t > 1 {
                    latents.narrow(2, 1, z_t - 1)?
                } else {
                    CudaTensor::zeros(&[1, z_c, 0, z_h, z_w])
                };
                latents = if z_t > 1 {
                    CudaTensor::cat(&[&first, &rest], 2)?
                } else {
                    first
                };
            }
            None
        } else {
            None
        };

        let clip_tokens = match (&self.clip, cfg.image_path.as_ref()) {
            (Some(clip), Some(path)) => Some(clip.encode_image_file(path)?),
            _ => None,
        };

        let encoder_hs = self.encode_prompt_embeds(cfg)?;
        let latents = self.denoise_inner(
            cfg,
            latents,
            &encoder_hs,
            clip_tokens.as_ref(),
            i2v_pack.as_ref().map(|(m, c)| (m, c)),
            None,
        )?;
        let video = self.decode_latents(&latents)?;
        let paths = write_frames(&video, Path::new(&cfg.output_dir))?;
        if cfg.save_video {
            match mux_mp4(Path::new(&cfg.output_dir), cfg.fps) {
                Ok(p) => super::log::info(format_args!("wrote mp4 {p}")),
                Err(e) => super::log::info(format_args!("mp4 mux skipped: {e}")),
            }
        }
        super::log::info(format_args!(
            "wrote {} png frames → {}",
            paths.len(),
            cfg.output_dir
        ));
        log_device_stats_if_enabled();
        Ok(paths)
    }

    /// `(C, T, H, W)` of the latent for `cfg` (tiny graphs use a fixed shape).
    pub fn latent_shape(&self, cfg: &GenerateConfig) -> (usize, usize, usize, usize) {
        if self.tiny {
            (4, 2, 4, 4)
        } else {
            (
                self.dit.cfg.out_channels,
                (cfg.num_frames.saturating_sub(1)) / 4 + 1,
                cfg.height / 8,
                cfg.width / 8,
            )
        }
    }

    /// Seeded `StdRng` + `StandardNormal` noise, in the same order as the
    /// Candle pipeline, so both backends start from bit-identical latents.
    /// rCM scales by `sigmas[0]` (`x₀ = noise · σ₀`).
    pub fn initial_latents(&self, cfg: &GenerateConfig) -> Result<CudaTensor> {
        let (z_c, z_t, z_h, z_w) = self.latent_shape(cfg);
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let n_el = z_c * z_t * z_h * z_w;
        let noise: Vec<f32> = (0..n_el)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let mut latents = CudaTensor::from_vec(noise, vec![1, z_c, z_t, z_h, z_w])?;
        if cfg.is_rcm {
            let sigma = cfg.rcm_sigma_max.unwrap_or(RCM_SIGMA_MAX_T2V);
            let scale =
                RcmSchedule::new(cfg.num_inference_steps.max(1), sigma).init_noise_scale() as f32;
            latents = latents.mul_scalar(scale);
        }
        Ok(latents)
    }

    /// Text-to-video denoise from precomputed `[neg, prompt]` embeddings
    /// (see [`Self::encode_prompt_embeds`]). `observer` runs after every step.
    pub fn denoise(
        &self,
        cfg: &GenerateConfig,
        latents: CudaTensor,
        encoder_hs: &CudaTensor,
        observer: Option<&mut StepObserver<'_>>,
    ) -> Result<CudaTensor> {
        if self.dit.cfg.in_channels > self.dit.cfg.out_channels {
            return Err(PipelineError::Message(
                "denoise() is text-to-video only; use generate() for I2V".into(),
            ));
        }
        self.denoise_inner(cfg, latents, encoder_hs, None, None, observer)
    }

    fn denoise_inner(
        &self,
        cfg: &GenerateConfig,
        latents: CudaTensor,
        encoder_hs: &CudaTensor,
        image: Option<&CudaTensor>,
        i2v: Option<(&CudaTensor, &CudaTensor)>,
        observer: Option<&mut StepObserver<'_>>,
    ) -> Result<CudaTensor> {
        // Inputs cross to the device once; every step then stays there.
        let latents = latents.to_device()?;
        let encoder_hs = encoder_hs.clone().to_device()?;
        let boundary = cfg.boundary_ratio.or(self.boundary_ratio);
        // DMD / rCM students are distilled for a single conditional pass.
        let (guidance, guidance_2) = if cfg.is_dmd || cfg.is_rcm {
            (1.0, 1.0)
        } else {
            (
                cfg.guidance_scale,
                cfg.guidance_scale_2.unwrap_or(cfg.guidance_scale),
            )
        };
        let n_steps = if cfg.is_rcm {
            RcmSchedule::new(
                cfg.num_inference_steps.max(1),
                cfg.rcm_sigma_max.unwrap_or(RCM_SIGMA_MAX_T2V),
            )
            .num_steps()
        } else if cfg.is_dmd {
            cfg.dmd_steps
                .as_ref()
                .map(|s| s.len())
                .unwrap_or(FAST_WAN_1_3B_DMD_STEPS.len())
        } else {
            cfg.num_inference_steps.max(1)
        };
        let tea_cache = TeaCache::from_env();
        let easy_cache = EasyCacheRuntime::from_env(n_steps)?;
        self.dit.configure_sol_teacache(n_steps)?;
        if let Some(low) = &self.dit_2 {
            low.configure_sol_teacache(n_steps)?;
        }
        if tea_cache.enabled && (easy_cache.is_some() || self.dit.sol_teacache_enabled()) {
            return Err(PipelineError::Message(
                "FASTVIDEO_TEACACHE and FASTVIDEO_WAN_SOL_CACHE both set".into(),
            ));
        }
        let mut ctx = DenoiseCtx {
            high: &self.dit,
            low: self.dit_2.as_ref(),
            boundary_ratio: boundary,
            image,
            i2v,
            guidance,
            guidance_2,
            tea_cache,
            easy_cache,
            sol_tea_step: 0,
        };
        if cfg.is_rcm {
            let sigma = cfg.rcm_sigma_max.unwrap_or(RCM_SIGMA_MAX_T2V);
            let sched = RcmSchedule::new(cfg.num_inference_steps.max(1), sigma);
            super::log::info(format_args!(
                "denoise=rcm steps={} sigma_max={sigma}",
                sched.num_steps()
            ));
            let _denoise = super::log::StepTimer::start(format!("rcm {} steps", sched.num_steps()));
            rcm_denoise(latents, &encoder_hs, &sched, cfg.seed, &mut ctx, observer)
        } else if cfg.is_dmd {
            let steps = cfg
                .dmd_steps
                .clone()
                .unwrap_or_else(|| FAST_WAN_1_3B_DMD_STEPS.to_vec());
            super::log::info(format_args!("denoise=dmd steps={}", steps.len()));
            let sched = DmdSchedule::new(&steps, 1000);
            let _denoise = super::log::StepTimer::start(format!("dmd {} steps", steps.len()));
            dmd_denoise(latents, &encoder_hs, &sched, cfg.seed, &mut ctx, observer)
        } else {
            let mut sched = FlowUniPCMultistepScheduler::new(1000, cfg.flow_shift);
            sched.set_timesteps(cfg.num_inference_steps);
            super::log::info(format_args!(
                "denoise=unipc steps={}",
                cfg.num_inference_steps
            ));
            let _denoise =
                super::log::StepTimer::start(format!("unipc {} steps", cfg.num_inference_steps));
            unipc_denoise(latents, &encoder_hs, &mut sched, &mut ctx, observer)
        }
    }

    /// Un-normalize latents and run the feat-cache VAE decode → `[1, 3, F, H, W]` in `[-1, 1]`.
    pub fn decode_latents(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        self.decode_latents_streaming(latents, &mut |_, _| Ok(()))
    }

    /// `decode_latents`, calling `sink(frame_offset, frames)` with each chunk
    /// of finished frames (`[frames, 3, H, W]` in `[-1, 1]`, in order) while
    /// the decoder is still busy with the next one. Feed a [`VideoWriter`]
    /// from the sink and frame encoding overlaps the decode.
    pub fn decode_latents_streaming(
        &self,
        latents: &CudaTensor,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<CudaTensor> {
        // Errors from the sink are pipeline errors, not tensor errors; carry
        // them across the decoder's own `Result` type.
        let mut sink_err: Option<PipelineError> = None;
        let mut tensor_sink = |offset: usize, frames: &CudaTensor| -> TensorResult<()> {
            match sink(offset, frames) {
                Ok(()) => Ok(()),
                Err(e) => {
                    let text = e.to_string();
                    sink_err = Some(e);
                    Err(TensorError::Message(text))
                }
            }
        };
        // TAEHV takes latents in DiT space — *without* the per-channel
        // un-normalisation the Wan VAE needs. That difference is the whole
        // reason this branch sits above `scale_latents` rather than inside the
        // decoder, and the oracle stage is what established it.
        let decoded = if let Some(tae) = &self.taehv {
            let _vae = super::log::StepTimer::start("taehv.decode");
            tae.decode_streaming(latents, &mut tensor_sink)
        } else {
            let latents = if !self.tiny {
                self.vae.scale_latents(latents)?
            } else {
                latents.clone()
            };
            let _vae = super::log::StepTimer::start("vae.decode");
            self.vae.decode_streaming(&latents, &mut tensor_sink)
        };
        match (decoded, sink_err) {
            (_, Some(e)) => Err(e),
            (Ok(v), None) => Ok(v),
            (Err(e), None) => Err(e.into()),
        }
    }

    pub fn transformer(&self) -> &WanTransformer3D {
        &self.dit
    }

    pub fn vae(&self) -> &AutoencoderKlWan {
        &self.vae
    }

    pub fn clip(&self) -> Option<&ClipVision> {
        self.clip.as_ref()
    }
}

struct DenoiseCtx<'a> {
    high: &'a WanTransformer3D,
    low: Option<&'a WanTransformer3D>,
    boundary_ratio: Option<f32>,
    image: Option<&'a CudaTensor>,
    i2v: Option<(&'a CudaTensor, &'a CudaTensor)>,
    guidance: f32,
    guidance_2: f32,
    tea_cache: TeaCache,
    easy_cache: Option<EasyCacheRuntime>,
    sol_tea_step: usize,
}

/// Simple residual TeaCache with Wan2.1-1.3B poly rescale (upstream TeaCache4Wan2.1).
/// Gate with `FASTVIDEO_TEACACHE=1`. Threshold via `FASTVIDEO_TEACACHE_THRESH`
/// (default 0.08 = upstream "fast" for 1.3B without ret_steps).
struct TeaCache {
    enabled: bool,
    thresh: f32,
    /// Accumulated rescaled relative L1 (upstream policy).
    accumulated: f32,
    step: usize,
    coefficients: [f64; 5],
    prev_modulated: Option<CudaTensor>,
    prev_residual: Option<CudaTensor>,
    cached: Option<CudaTensor>,
}

impl TeaCache {
    fn from_env() -> Self {
        let enabled = std::env::var("FASTVIDEO_TEACACHE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let thresh = std::env::var("FASTVIDEO_TEACACHE_THRESH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.08);
        // Wan2.1 T2V 1.3B coefficients (use_ret_steps=False) from TeaCache4Wan2.1.
        let coefficients = [
            2.39676752e3,
            -1.31110545e3,
            2.01331979e2,
            -8.29855975,
            1.37887774e-1,
        ];
        let ret_steps = std::env::var("FASTVIDEO_TEACACHE_RET")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let coefficients = if ret_steps {
            // use_ret_steps=True coeffs for 1.3B
            [
                -5.21862437e4,
                9.23041404e3,
                -5.28275948e2,
                1.36987616e1,
                -4.99875664e-2,
            ]
        } else {
            coefficients
        };
        Self {
            enabled,
            thresh,
            accumulated: 0.0,
            step: 0,
            coefficients,
            prev_modulated: None,
            prev_residual: None,
            cached: None,
        }
    }

    fn poly_rescale(&self, x: f64) -> f64 {
        let c = &self.coefficients;
        ((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x) + c[4]
    }

    fn maybe_reuse(&mut self, latents: &CudaTensor) -> Option<CudaTensor> {
        if !self.enabled {
            return None;
        }
        let (Some(prev), Some(cached)) = (&self.prev_modulated, &self.cached) else {
            return None;
        };
        if prev.shape != latents.shape {
            return None;
        }
        let Ok(lat_h) = latents.host_cow() else {
            return None;
        };
        let Ok(prev_h) = prev.host_cow() else {
            return None;
        };
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (a, b) in lat_h.iter().zip(prev_h.iter()) {
            num += (f64::from(*a) - f64::from(*b)).abs();
            den += f64::from(*b).abs();
        }
        let rel = num / (den + 1e-6);
        self.accumulated += self.poly_rescale(rel) as f32;
        self.step += 1;
        // Always compute first / last-ish steps; skip when accumulated below thresh.
        if self.step <= 1 {
            return None;
        }
        if self.accumulated < self.thresh {
            if let Some(res) = &self.prev_residual {
                return Some(latents.add(res).unwrap_or_else(|_| cached.clone()));
            }
            return Some(cached.clone());
        }
        self.accumulated = 0.0;
        None
    }

    fn store(&mut self, latents: &CudaTensor, out: &CudaTensor) {
        if !self.enabled {
            return;
        }
        self.prev_modulated = Some(latents.clone());
        self.cached = Some(out.clone());
        if let Ok(res) = out.sub(latents) {
            self.prev_residual = Some(res);
        }
    }
}

/// Sol-engine EasyCache (`WAN22_CACHE_FAMILY=easycache`). Off unless
/// `FASTVIDEO_WAN_SOL_CACHE=easycache`. Cond decides; uncond follows.
struct EasyCacheRuntime {
    state: fastvideo_models::wan::sol_cache::EasyCache,
    step: usize,
    previous_step_input: Option<CudaTensor>,
    last_full_input: Option<CudaTensor>,
    last_full_output: Option<CudaTensor>,
    cond_residual: Option<CudaTensor>,
    uncond_residual: Option<CudaTensor>,
}

fn mean_abs(t: &CudaTensor) -> TensorResult<f64> {
    let host = t.host_cow()?;
    let n = host.len().max(1) as f64;
    Ok(host.iter().map(|v| f64::from(*v).abs()).sum::<f64>() / n)
}

fn mean_abs_delta(a: &CudaTensor, b: &CudaTensor) -> TensorResult<f64> {
    let left = a.host_cow()?;
    let right = b.host_cow()?;
    let n = left.len().max(1) as f64;
    let mut sum = 0.0;
    for (x, y) in left.iter().zip(right.iter()) {
        sum += (f64::from(*x) - f64::from(*y)).abs();
    }
    Ok(sum / n)
}

impl EasyCacheRuntime {
    fn from_env(num_steps: usize) -> Result<Option<Self>> {
        let family = std::env::var("FASTVIDEO_WAN_SOL_CACHE").unwrap_or_default();
        let family = family.trim().to_ascii_lowercase();
        if family.is_empty() || matches!(family.as_str(), "0" | "off" | "none" | "false") {
            return Ok(None);
        }
        if family != "easycache" && family != "teacache" {
            return Err(PipelineError::Message(format!(
                "FASTVIDEO_WAN_SOL_CACHE={family} is not supported (easycache or teacache)"
            )));
        }
        if family == "teacache" {
            return Ok(None);
        }
        let threshold = std::env::var("FASTVIDEO_WAN_EASYCACHE_THRESH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.05);
        let retain = std::env::var("FASTVIDEO_WAN_EASYCACHE_RETAIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(7);
        let cooldown = std::env::var("FASTVIDEO_WAN_EASYCACHE_COOLDOWN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let state = fastvideo_models::wan::sol_cache::EasyCache::new(
            num_steps, threshold, retain, cooldown,
        )
        .map_err(PipelineError::Message)?;
        super::log::info(format_args!(
            "wan sol easycache: threshold {threshold} retain {retain} cooldown {cooldown} steps {num_steps} (cond decides, uncond follows)"
        ));
        Ok(Some(Self {
            state,
            step: 0,
            previous_step_input: None,
            last_full_input: None,
            last_full_output: None,
            cond_residual: None,
            uncond_residual: None,
        }))
    }
}

fn pick_expert<'a>(ctx: &'a DenoiseCtx<'_>, t: f32) -> (&'a WanTransformer3D, f32) {
    if let (Some(ratio), Some(low)) = (ctx.boundary_ratio, ctx.low) {
        match moe_expert(f64::from(t), ratio, 1000) {
            MoeExpert::LowNoise => (low, ctx.guidance_2),
            MoeExpert::HighNoise => (ctx.high, ctx.guidance),
        }
    } else {
        (ctx.high, ctx.guidance)
    }
}

fn pack_dit_input(
    latents: &CudaTensor,
    i2v: Option<(&CudaTensor, &CudaTensor)>,
) -> TensorResult<CudaTensor> {
    if let Some((mask, cond)) = i2v {
        CudaTensor::cat(&[latents, mask, cond], 1)
    } else {
        Ok(latents.clone())
    }
}

fn dit_cfg_easy(
    ctx: &mut DenoiseCtx<'_>,
    latents: &CudaTensor,
    encoder_hs: &CudaTensor,
    t: f32,
) -> TensorResult<CudaTensor> {
    let latent_in = pack_dit_input(latents, ctx.i2v)?;
    let decision = {
        let easy = ctx.easy_cache.as_mut().expect("easycache");
        let step = easy.step;
        easy.step += 1;
        let (change, norm) = if easy.state.needs_cond_signal(step) {
            (
                mean_abs_delta(&latent_in, easy.previous_step_input.as_ref().expect("prev"))?,
                mean_abs(easy.last_full_output.as_ref().expect("last out"))?,
            )
        } else {
            (0.0, 1.0)
        };
        easy.previous_step_input = Some(latent_in.clone());
        easy.state.decide_cond(step, change, norm)
    };
    let rows = encoder_hs.shape[0];
    let cond_hs = if rows > 1 {
        encoder_hs.narrow(0, 1, 1)?
    } else {
        encoder_hs.clone()
    };
    let t1 = CudaTensor::from_vec(vec![t], vec![1])?;
    let (scale, cond) = {
        let (dit, scale) = pick_expert(ctx, t);
        if decision.compute {
            let out = dit.forward_ctx(&latent_in, &t1, &cond_hs, ctx.image)?;
            (scale, Some(out))
        } else {
            (scale, None)
        }
    };
    let cond = if let Some(out) = cond {
        let easy = ctx.easy_cache.as_mut().expect("easycache");
        let (full_in, out_change) = match (&easy.last_full_input, &easy.last_full_output) {
            (Some(prev_in), Some(prev_out)) => (
                mean_abs_delta(&latent_in, prev_in)?,
                mean_abs_delta(&out, prev_out)?,
            ),
            _ => (0.0, 0.0),
        };
        let _ = easy.state.note_cond_computed(full_in, out_change);
        easy.last_full_input = Some(latent_in.clone());
        easy.last_full_output = Some(out.clone());
        easy.cond_residual = Some(out.sub(&latent_in)?);
        out
    } else {
        super::log::debug(format_args!(
            "wan easycache reuse step reason {}",
            decision.reason
        ));
        let res = ctx
            .easy_cache
            .as_ref()
            .expect("easycache")
            .cond_residual
            .as_ref()
            .expect("cond residual");
        latent_in.add(res)?
    };
    if (scale - 1.0).abs() < 1e-6 || rows < 2 {
        return Ok(cond);
    }
    let follow = {
        let easy = ctx.easy_cache.as_ref().expect("easycache");
        easy.state.uncond_compute(easy.uncond_residual.is_some())
    };
    let uncond_hs = encoder_hs.narrow(0, 0, 1)?;
    let uncond = if follow {
        let out = {
            let (dit, _) = pick_expert(ctx, t);
            dit.forward_ctx(&latent_in, &t1, &uncond_hs, ctx.image)?
        };
        let easy = ctx.easy_cache.as_mut().expect("easycache");
        easy.uncond_residual = Some(out.sub(&latent_in)?);
        out
    } else {
        let res = ctx
            .easy_cache
            .as_ref()
            .expect("easycache")
            .uncond_residual
            .as_ref()
            .expect("uncond residual");
        latent_in.add(res)?
    };
    CudaTensor::lincomb(&[(1.0 - scale, &uncond), (scale, &cond)])
}

fn dit_cfg(
    ctx: &mut DenoiseCtx<'_>,
    latents: &CudaTensor,
    encoder_hs: &CudaTensor,
    t: f32,
) -> TensorResult<CudaTensor> {
    if ctx.easy_cache.is_some() {
        return dit_cfg_easy(ctx, latents, encoder_hs, t);
    }
    let tea_step = ctx.sol_tea_step;
    let tea_on =
        ctx.high.sol_teacache_enabled() || ctx.low.is_some_and(|dit| dit.sol_teacache_enabled());
    if tea_on {
        ctx.sol_tea_step += 1;
    }
    if let Some(cached) = ctx.tea_cache.maybe_reuse(latents) {
        static TEA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = TEA.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        super::log::debug(format_args!("teacache hit #{n}"));
        return Ok(cached);
    }
    let (dit, scale) = pick_expert(ctx, t);
    let latent_in = pack_dit_input(latents, ctx.i2v)?;
    // `encoder_hs` is `[negative, prompt]`; a single row is the prompt alone.
    let rows = encoder_hs.shape[0];
    let cond_hs = if rows > 1 {
        encoder_hs.narrow(0, 1, 1)?
    } else {
        encoder_hs.clone()
    };
    let t1 = CudaTensor::from_vec(vec![t], vec![1])?;

    let out = if (scale - 1.0).abs() < 1e-6 {
        if tea_on {
            dit.arm_sol_teacache(true, tea_step);
        }
        dit.forward_ctx(&latent_in, &t1, &cond_hs, ctx.image)?
    } else if !tea_on && ctx.i2v.is_none() && ctx.image.is_none() && rows == 2 {
        // One batch-2 forward for [uncond, cond]: the embeddings are already
        // in that order, so only the latents are duplicated. Sol TeaCache
        // needs one branch per forward, so it takes the split path below.
        let latent_batch = CudaTensor::cat(&[&latent_in, &latent_in], 0)?;
        let t_batch = CudaTensor::from_vec(vec![t, t], vec![2])?;
        let out_batch = dit.forward_ctx(&latent_batch, &t_batch, encoder_hs, None)?;
        let uncond = out_batch.narrow(0, 0, 1)?;
        let cond = out_batch.narrow(0, 1, 1)?;
        // uncond + s·(cond − uncond)
        CudaTensor::lincomb(&[(1.0 - scale, &uncond), (scale, &cond)])?
    } else {
        let uncond_hs = encoder_hs.narrow(0, 0, 1)?;
        if tea_on {
            dit.arm_sol_teacache(true, tea_step);
        }
        let cond = dit.forward_ctx(&latent_in, &t1, &cond_hs, ctx.image)?;
        if tea_on {
            dit.arm_sol_teacache(false, tea_step);
        }
        let uncond = dit.forward_ctx(&latent_in, &t1, &uncond_hs, ctx.image)?;
        CudaTensor::lincomb(&[(1.0 - scale, &uncond), (scale, &cond)])?
    };
    ctx.tea_cache.store(latents, &out);
    Ok(out)
}

fn notify(
    observer: &mut Option<&mut StepObserver<'_>>,
    index: usize,
    total: usize,
    timestep: f32,
    latents: &CudaTensor,
) -> Result<()> {
    match observer.as_deref_mut() {
        Some(obs) => obs(&DenoiseStep {
            index,
            total,
            timestep,
            latents,
        }),
        None => Ok(()),
    }
}

/// Seeded standard-normal noise for DMD step `i` (independent per step).
fn dmd_noise(seed: u64, step: usize, shape: &[usize]) -> Result<CudaTensor> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(
        seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(step as u64 + 1)),
    );
    let n: usize = shape.iter().product();
    let noise: Vec<f32> = (0..n)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    Ok(CudaTensor::from_vec(noise, shape.to_vec())?.to_device()?)
}

/// FastVideo DMD sampler: predict x0 = x − σ_t·v, then re-noise to the next
/// table sigma with fresh noise; the last step returns x0.
fn dmd_denoise(
    mut latents: CudaTensor,
    encoder_hs: &CudaTensor,
    sched: &DmdSchedule,
    seed: u64,
    ctx: &mut DenoiseCtx<'_>,
    mut observer: Option<&mut StepObserver<'_>>,
) -> Result<CudaTensor> {
    let total = sched.num_steps();
    for (i, &t) in sched.train_timesteps.iter().enumerate() {
        let t = t as f32;
        let _step = super::log::StepTimer::start(format!("dmd step {}/{total} t={t:.0}", i + 1));
        let velocity = dit_cfg(ctx, &latents, encoder_hs, t)?;
        let c = sched.step_coeffs(i);
        let x0 = CudaTensor::lincomb(&[(1.0, &latents), (-(c.sigma_t as f32), &velocity)])?;
        latents = match c.sigma_next {
            Some(next) => {
                let noise = dmd_noise(seed, i, &latents.shape)?;
                CudaTensor::lincomb(&[(1.0 - next as f32, &x0), (next as f32, &noise)])?
            }
            None => x0,
        };
        super::device::synchronize().map_err(|e| PipelineError::Message(e.to_string()))?;
        notify(&mut observer, i, total, t, &latents)?;
    }
    Ok(latents)
}

/// TurboDiffusion rCM: same x0 / re-noise shape as DMD, TrigFlow→RF sigmas.
fn rcm_denoise(
    mut latents: CudaTensor,
    encoder_hs: &CudaTensor,
    sched: &RcmSchedule,
    seed: u64,
    ctx: &mut DenoiseCtx<'_>,
    mut observer: Option<&mut StepObserver<'_>>,
) -> Result<CudaTensor> {
    let total = sched.num_steps();
    for i in 0..total {
        let c = sched.step_coeffs(i);
        let t = c.model_timestep;
        let _step = super::log::StepTimer::start(format!("rcm step {}/{total} t={t:.3}", i + 1));
        let velocity = dit_cfg(ctx, &latents, encoder_hs, t)?;
        let x0 = CudaTensor::lincomb(&[(1.0, &latents), (-(c.t_cur as f32), &velocity)])?;
        latents = if c.t_next == 0.0 {
            x0
        } else {
            let noise = dmd_noise(seed, i, &latents.shape)?;
            CudaTensor::lincomb(&[(1.0 - c.t_next as f32, &x0), (c.t_next as f32, &noise)])?
        };
        super::device::synchronize().map_err(|e| PipelineError::Message(e.to_string()))?;
        notify(&mut observer, i, total, t, &latents)?;
    }
    Ok(latents)
}

fn warn_sla_backend() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if super::sla::sla_enabled() {
            let cfg = super::sla::SlaConfig::from_env();
            super::log::info(format_args!(
                "TurboWan: SLA attention (topk={:.2} BLKQ={} BLKK={})",
                cfg.topk_ratio, cfg.blk_q, cfg.blk_k
            ));
        } else {
            super::log::info(format_args!(
                "TurboWan: dense SDPA (set FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN for Sparse-Linear Attention)"
            ));
        }
    });
}

/// Σ coef·term over one UniPC plan combination.
fn unipc_combine(
    terms: &[(UniPcTerm, f64)],
    sample: &CudaTensor,
    last_sample: Option<&CudaTensor>,
    converted: &CudaTensor,
    history: &std::collections::VecDeque<CudaTensor>,
) -> Result<CudaTensor> {
    let mut parts: Vec<(f32, &CudaTensor)> = Vec::with_capacity(terms.len());
    for (term, coef) in terms {
        let t = match term {
            UniPcTerm::Sample => sample,
            UniPcTerm::Converted => converted,
            UniPcTerm::LastSample => last_sample
                .ok_or_else(|| PipelineError::Message("unipc: plan needs a last sample".into()))?,
            UniPcTerm::History(k) => history
                .get(*k)
                .ok_or_else(|| PipelineError::Message(format!("unipc: plan needs history[{k}]")))?,
        };
        parts.push((*coef as f32, t));
    }
    Ok(CudaTensor::lincomb(&parts)?)
}

/// Order-2 bh2 UniPC predictor-corrector (FastVideo defaults) with every
/// update a linear combination of latent-shaped tensors: device kernels on
/// GPU runs, the same math on host for CPU runs.
fn unipc_denoise(
    mut latents: CudaTensor,
    encoder_hs: &CudaTensor,
    sched: &mut FlowUniPCMultistepScheduler,
    ctx: &mut DenoiseCtx<'_>,
    mut observer: Option<&mut StepObserver<'_>>,
) -> Result<CudaTensor> {
    let ts: Vec<f32> = sched
        .inference_timesteps_i64()
        .iter()
        .map(|t| *t as f32)
        .collect();
    let mut history = std::collections::VecDeque::new();
    let mut last_sample: Option<CudaTensor> = None;
    for (i, &t) in ts.iter().enumerate() {
        let _step =
            super::log::StepTimer::start(format!("unipc step {}/{} t={t:.0}", i + 1, ts.len()));
        let velocity = dit_cfg(ctx, &latents, encoder_hs, t)?;
        let plan = sched.plan_step().map_err(PipelineError::Message)?;
        let converted =
            CudaTensor::lincomb(&[(1.0, &latents), (-(plan.convert_scale as f32), &velocity)])?;
        let corrected = match &plan.corrector {
            Some(terms) => {
                unipc_combine(terms, &latents, last_sample.as_ref(), &converted, &history)?
            }
            None => latents.clone(),
        };
        let prev = unipc_combine(
            &plan.predictor,
            &corrected,
            last_sample.as_ref(),
            &converted,
            &history,
        )?;
        history.push_front(converted);
        history.truncate(plan.history_len);
        last_sample = Some(corrected);
        latents = prev;
        super::device::synchronize().map_err(|e| PipelineError::Message(e.to_string()))?;
        notify(&mut observer, i, ts.len(), t, &latents)?;
    }
    Ok(latents)
}

fn tokenize_prompt(path: &str, text: &str, max_len: usize) -> Result<(Vec<u32>, usize)> {
    fastvideo_models::tokenize_prompt(path, text, max_len)
        .map_err(|e| PipelineError::Message(format!("{e}")))
}

fn load_rgb_frame(path: &str, height: usize, width: usize) -> Result<CudaTensor> {
    let img = image::open(path)
        .map_err(|e| PipelineError::Message(format!("open image {path}: {e}")))?
        .into_rgb8();
    let img = image::imageops::resize(
        &img,
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let mut data = vec![0.0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x as u32, y as u32);
            for ch in 0..3 {
                let v = f32::from(p[ch]) / 127.5 - 1.0;
                data[ch * height * width + y * width + x] = v;
            }
        }
    }
    // [1, 3, 1, H, W] single-frame video for VAE encode
    CudaTensor::from_vec(data, vec![1, 3, 1, height, width]).map_err(Into::into)
}

/// Write `[1, C>=3, F, H, W]` video in `[-1, 1]` as `frame-%03d.png`.
pub fn write_frames(video: &CudaTensor, dir: &Path) -> Result<Vec<String>> {
    if video.rank() != 5 || video.shape[0] != 1 {
        return Err(PipelineError::Message(format!(
            "expected 1CTHW video, got {:?}",
            video.shape
        )));
    }
    let (c, t, h, w) = (
        video.shape[1],
        video.shape[2],
        video.shape[3],
        video.shape[4],
    );
    if c < 3 {
        return Err(PipelineError::Message("video needs RGB channels".into()));
    }
    // [1, C, T, H, W] → [T, 3, H, W], the per-frame layout the packer takes.
    let by_frame = video
        .reshape(vec![c, t, h, w])?
        .narrow(0, 0, 3)?
        .permute(&[1, 0, 2, 3])?;
    let rgb = frames_to_rgb8(&by_frame)?;
    let mut writer = VideoWriter::spawn(dir, 0, false)?;
    writer.push(0, h, w, rgb)?;
    Ok(writer.finish()?.0)
}

/// `[frames, 3, H, W]` in `[-1, 1]` → interleaved 8-bit RGB, `[frames, H, W, 3]`.
///
/// On a device tensor this is one kernel plus a 3-byte-per-pixel copy down.
/// The host loop only runs when the tensor has no device buffer, i.e. CPU
/// builds and tests: a GPU run never silently converts on the host.
pub fn frames_to_rgb8(frames: &CudaTensor) -> Result<Vec<u8>> {
    let [f, c, h, w] = frames.shape[..] else {
        return Err(PipelineError::Message(format!(
            "frames_to_rgb8 expects [frames, 3, H, W], got {:?}",
            frames.shape
        )));
    };
    if c != 3 {
        return Err(PipelineError::Message(format!(
            "frames_to_rgb8 needs 3 channels, got {c}"
        )));
    }
    #[cfg(feature = "cuda")]
    {
        let mut on_device = frames.clone();
        on_device.ensure_device()?;
        if let Some(slice) = on_device.device_slice() {
            return Ok(super::ops::pack_rgb_u8_device(
                slice, f, h, w, 127.5, 127.5,
            )?);
        }
    }
    let host = frames.host_cow()?;
    let plane = h * w;
    let mut rgb = vec![0u8; f * plane * 3];
    for (i, px) in rgb.chunks_exact_mut(3).enumerate() {
        let (fi, p) = (i / plane, i % plane);
        for (ch, out) in px.iter_mut().enumerate() {
            let v = host[(fi * 3 + ch) * plane + p];
            *out = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(rgb)
}

/// One chunk of packed frames on its way to disk.
struct FrameBatch {
    offset: usize,
    h: usize,
    w: usize,
    rgb: Vec<u8>,
}

/// Writes frames as they are decoded: PNGs in parallel on a worker thread,
/// and (when `fps > 0` and `mp4` is set) raw rgb24 straight into an ffmpeg
/// process so the mux needs no PNG round trip. Encoding overlaps the decode
/// of the next chunk instead of waiting for the whole video.
pub struct VideoWriter {
    tx: Option<std::sync::mpsc::SyncSender<FrameBatch>>,
    worker: Option<std::thread::JoinHandle<Result<(Vec<String>, Option<String>)>>>,
}

impl VideoWriter {
    /// `fps` and `mp4` decide whether an `output.mp4` is produced next to the
    /// frames; ffmpeg is spawned lazily on the first batch, once the frame
    /// size is known.
    pub fn spawn(dir: &Path, fps: u32, mp4: bool) -> Result<Self> {
        Self::spawn_with_audio(dir, fps, mp4, None)
    }

    /// As [`Self::spawn`], muxing `audio` (a WAV already on disk — see
    /// [`write_wav`]) into the mp4 as AAC. The audio-video models decode their
    /// audio first, which takes milliseconds, so the track exists before the
    /// first frame does and ffmpeg can still be fed frames as they decode.
    pub fn spawn_with_audio(dir: &Path, fps: u32, mp4: bool, audio: Option<&Path>) -> Result<Self> {
        std::fs::create_dir_all(dir).map_err(|e| PipelineError::Message(e.to_string()))?;
        if let Some(a) = audio {
            if !a.is_file() {
                return Err(PipelineError::Message(format!(
                    "audio track {} does not exist",
                    a.display()
                )));
            }
        }
        let audio = audio.map(Path::to_path_buf);
        let dir = dir.to_path_buf();
        // Two chunks of backlog: enough that a slow PNG batch never stalls the
        // decoder, small enough that memory stays bounded.
        let (tx, rx) = std::sync::mpsc::sync_channel::<FrameBatch>(2);
        let worker = std::thread::Builder::new()
            .name("fv-video-writer".into())
            .spawn(move || write_batches(rx, &dir, fps, mp4, audio.as_deref()))
            .map_err(|e| PipelineError::Message(e.to_string()))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
        })
    }

    /// Queue `[frames, h, w, 3]` bytes starting at frame `offset`. Batches must
    /// arrive in frame order; the mp4 is written in the order pushed.
    pub fn push(&mut self, offset: usize, h: usize, w: usize, rgb: Vec<u8>) -> Result<()> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| PipelineError::Message("video writer already finished".into()))?;
        if tx.send(FrameBatch { offset, h, w, rgb }).is_err() {
            // The worker is gone; its error is the one worth reporting.
            return match self.finish() {
                Ok(_) => Err(PipelineError::Message("video writer stopped early".into())),
                Err(e) => Err(e),
            };
        }
        Ok(())
    }

    /// Wait for every queued frame (and ffmpeg) and return the PNG paths in
    /// frame order plus the mp4 path, if one was requested.
    pub fn finish(&mut self) -> Result<(Vec<String>, Option<String>)> {
        drop(self.tx.take());
        match self.worker.take() {
            Some(h) => h
                .join()
                .map_err(|_| PipelineError::Message("video writer thread panicked".into()))?,
            None => Err(PipelineError::Message(
                "video writer already finished".into(),
            )),
        }
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

fn write_batches(
    rx: std::sync::mpsc::Receiver<FrameBatch>,
    dir: &Path,
    fps: u32,
    mp4: bool,
    audio: Option<&Path>,
) -> Result<(Vec<String>, Option<String>)> {
    use rayon::prelude::*;
    use std::io::Write as _;

    let out = dir.join("output.mp4");
    let mut ffmpeg: Option<std::process::Child> = None;
    let mut paths: Vec<String> = Vec::new();
    for batch in rx {
        let FrameBatch { offset, h, w, rgb } = batch;
        let frame_bytes = h * w * 3;
        if frame_bytes == 0 || rgb.len() % frame_bytes != 0 {
            return Err(PipelineError::Message(format!(
                "frame batch of {} bytes is not whole {w}x{h} RGB frames",
                rgb.len()
            )));
        }
        if mp4 && fps > 0 && ffmpeg.is_none() {
            ffmpeg = Some(spawn_ffmpeg_rgb(&out, w, h, fps, audio)?);
        }
        if let Some(child) = ffmpeg.as_mut() {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| PipelineError::Message("ffmpeg stdin closed".into()))?;
            stdin
                .write_all(&rgb)
                .map_err(|e| PipelineError::Message(format!("ffmpeg stdin: {e}")))?;
        }
        let batch_paths: Vec<Result<String>> = rgb
            .par_chunks_exact(frame_bytes)
            .enumerate()
            .map(|(i, frame)| {
                let img = image::RgbImage::from_raw(w as u32, h as u32, frame.to_vec())
                    .ok_or_else(|| PipelineError::Message("rgb buffer size mismatch".into()))?;
                let path = dir.join(format!("frame-{:03}.png", offset + i));
                img.save(&path)
                    .map_err(|e| PipelineError::Message(e.to_string()))?;
                Ok(path.to_string_lossy().into_owned())
            })
            .collect();
        for p in batch_paths {
            paths.push(p?);
        }
    }
    let mp4_path = match ffmpeg {
        Some(mut child) => {
            drop(child.stdin.take());
            let status = child
                .wait()
                .map_err(|e| PipelineError::Message(format!("ffmpeg: {e}")))?;
            if !status.success() {
                return Err(PipelineError::Message(format!(
                    "ffmpeg failed with {status}"
                )));
            }
            Some(out.to_string_lossy().into_owned())
        }
        None => None,
    };
    Ok((paths, mp4_path))
}

fn spawn_ffmpeg_rgb(
    out: &Path,
    w: usize,
    h: usize,
    fps: u32,
    audio: Option<&Path>,
) -> Result<std::process::Child> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
    ])
    .arg(format!("{w}x{h}"))
    .args(["-framerate", &fps.to_string(), "-i", "-"]);
    if let Some(a) = audio {
        // Input 1. Mapped explicitly so the track is never dropped silently,
        // and the mp4 ends with the shorter stream: the two decoders round
        // their lengths differently by a few milliseconds.
        cmd.arg("-i").arg(a).args([
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ]);
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(out)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| PipelineError::Message(format!("ffmpeg not available: {e}")))
}

/// Interleaved `[-1, 1]` samples as a 16-bit PCM WAV. `samples.len()` must be
/// a whole number of `channels`-wide frames. Out-of-range values saturate
/// rather than wrap: a decoder overshooting by 1% must not become a click.
pub fn write_wav(path: &Path, samples: &[f32], channels: u16, sample_rate: u32) -> Result<()> {
    use std::io::Write as _;
    if channels == 0 || samples.len() % usize::from(channels) != 0 {
        return Err(PipelineError::Message(format!(
            "{} samples is not whole {channels}-channel frames",
            samples.len()
        )));
    }
    let data_len = u32::try_from(samples.len() * 2)
        .ok()
        .filter(|n| *n <= u32::MAX - 36)
        .ok_or_else(|| PipelineError::Message("audio too long for a WAV header".into()))?;
    let block = channels * 2;
    let mut buf = Vec::with_capacity(44 + samples.len() * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * u32::from(block)).to_le_bytes());
    buf.extend_from_slice(&block.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &v in samples {
        let q = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        buf.extend_from_slice(&q.to_le_bytes());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| PipelineError::Message(e.to_string()))?;
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&buf))
        .map_err(|e| PipelineError::Message(format!("{}: {e}", path.display())))
}

/// `[channels, samples]` planar (what the audio decoders emit) → interleaved.
pub fn interleave_audio(planar: &[f32], channels: usize) -> Result<Vec<f32>> {
    if channels == 0 || planar.len() % channels != 0 {
        return Err(PipelineError::Message(format!(
            "{} samples over {channels} channels",
            planar.len()
        )));
    }
    let n = planar.len() / channels;
    let mut out = Vec::with_capacity(planar.len());
    for i in 0..n {
        for c in 0..channels {
            out.push(planar[c * n + i]);
        }
    }
    Ok(out)
}

/// Mux `frame-%03d.png` in `dir` into `dir/output.mp4`.
pub fn mux_mp4(dir: &Path, fps: u32) -> Result<String> {
    let out = dir.join("output.mp4");
    let pattern = dir.join("frame-%03d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-nostats",
            "-framerate",
            &fps.to_string(),
            "-i",
            &pattern.to_string_lossy(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            &out.to_string_lossy(),
        ])
        .status()
        .map_err(|e| PipelineError::Message(format!("ffmpeg not available: {e}")))?;
    if !status.success() {
        return Err(PipelineError::Message(format!(
            "ffmpeg failed with {status}"
        )));
    }
    Ok(out.to_string_lossy().into_owned())
}

#[cfg(test)]
mod frame_output_tests {
    use super::*;

    /// The packer's byte mapping is the one the old per-pixel writer used:
    /// trunc(clamp((v + 1) * 127.5, 0, 255)), with -1 → 0 and 1 → 255.
    #[test]
    fn rgb8_mapping_matches_the_previous_writer() {
        let (f, h, w) = (2usize, 2usize, 3usize);
        let vals: Vec<f32> = (0..f * 3 * h * w)
            .map(|i| match i % 5 {
                0 => -1.0,
                1 => 1.0,
                2 => -2.5,
                3 => 0.0,
                _ => 0.25,
            })
            .collect();
        let frames = CudaTensor::from_vec(vals.clone(), vec![f, 3, h, w]).unwrap();
        let rgb = frames_to_rgb8(&frames).unwrap();
        assert_eq!(rgb.len(), f * h * w * 3);
        let plane = h * w;
        for (i, px) in rgb.chunks_exact(3).enumerate() {
            let (fi, p) = (i / plane, i % plane);
            for ch in 0..3 {
                let v = vals[(fi * 3 + ch) * plane + p];
                let want = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                assert_eq!(px[ch], want, "frame {fi} pixel {p} channel {ch}");
            }
        }
    }

    /// Batches pushed in order come out as `frame-%03d.png` in order, whatever
    /// the batch boundaries were.
    #[test]
    fn writer_numbers_frames_across_batches() {
        let dir = std::env::temp_dir().join(format!("fv-writer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (h, w) = (4usize, 6usize);
        let mut writer = VideoWriter::spawn(&dir, 0, false).unwrap();
        writer.push(0, h, w, vec![10u8; 3 * h * w * 3]).unwrap();
        writer.push(3, h, w, vec![200u8; 2 * h * w * 3]).unwrap();
        let (paths, mp4) = writer.finish().unwrap();
        assert!(mp4.is_none());
        assert_eq!(paths.len(), 5);
        for (i, p) in paths.iter().enumerate() {
            assert!(p.ends_with(&format!("frame-{i:03}.png")), "{p}");
        }
        let last = image::open(&paths[4]).unwrap().to_rgb8();
        assert_eq!(last.dimensions(), (w as u32, h as u32));
        assert_eq!(last.get_pixel(0, 0).0, [200, 200, 200]);
        let first = image::open(&paths[0]).unwrap().to_rgb8();
        assert_eq!(first.get_pixel(0, 0).0, [10, 10, 10]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wav_header_and_samples_are_exact() {
        let path = std::env::temp_dir().join(format!("fv-wav-{}.wav", std::process::id()));
        // Stereo, planar in, 3 frames; one sample beyond full scale.
        let inter = interleave_audio(&[0.0, 0.5, -1.0, 1.5, -0.25, 0.0], 2).unwrap();
        assert_eq!(inter, vec![0.0, 1.5, 0.5, -0.25, -1.0, 0.0]);
        write_wav(&path, &inter, 2, 32_000).unwrap();
        let b = std::fs::read(&path).unwrap();
        assert_eq!(&b[..4], b"RIFF");
        assert_eq!(
            u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize,
            b.len() - 8
        );
        assert_eq!(
            u16::from_le_bytes(b[22..24].try_into().unwrap()),
            2,
            "channels"
        );
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 32_000);
        assert_eq!(
            u32::from_le_bytes(b[28..32].try_into().unwrap()),
            32_000 * 4,
            "byte rate"
        );
        assert_eq!(
            u32::from_le_bytes(b[40..44].try_into().unwrap()),
            12,
            "data bytes"
        );
        let pcm: Vec<i16> = b[44..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(
            pcm,
            vec![0, 32767, 16384, -8192, -32767, 0],
            "saturates, never wraps"
        );
        assert!(
            write_wav(&path, &[0.0; 3], 2, 32_000).is_err(),
            "ragged frames"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// End to end through real ffmpeg: frames pushed in batches plus a WAV
    /// come out as one mp4 with a video and an AAC stream. Skipped where ffmpeg
    /// is not installed (it is on every box that generates).
    #[test]
    fn mp4_carries_the_audio_track() {
        let have = |bin: &str| {
            Command::new(bin)
                .arg("-version")
                .output()
                .is_ok_and(|o| o.status.success())
        };
        if !have("ffmpeg") || !have("ffprobe") {
            eprintln!("skip: ffmpeg/ffprobe not installed");
            return;
        }
        let dir = std::env::temp_dir().join(format!("fv-av-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let wav = dir.join("audio.wav");
        let tone: Vec<f32> = (0..32_000)
            .flat_map(|i| {
                let v = (i as f32 * 440.0 * std::f32::consts::TAU / 32_000.0).sin() * 0.3;
                [v, v]
            })
            .collect();
        write_wav(&wav, &tone, 2, 32_000).unwrap();
        let (h, w, fps) = (32usize, 48usize, 24u32);
        let mut writer =
            VideoWriter::spawn_with_audio(&dir.join("frames"), fps, true, Some(&wav)).unwrap();
        for batch in 0..3 {
            writer
                .push(
                    batch * 8,
                    h,
                    w,
                    vec![(60 * batch) as u8 + 40; 8 * h * w * 3],
                )
                .unwrap();
        }
        let (frames, mp4) = writer.finish().unwrap();
        assert_eq!(frames.len(), 24);
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name",
                "-of",
                "csv=p=0",
            ])
            .arg(mp4.expect("an mp4 was requested"))
            .output()
            .unwrap();
        let streams = String::from_utf8_lossy(&probe.stdout);
        assert!(
            streams.contains("h264,video") && streams.contains("aac,audio"),
            "streams: {streams}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_audio_track_is_an_error_before_any_frame() {
        let dir = std::env::temp_dir().join(format!("fv-writer-noaudio-{}", std::process::id()));
        let err =
            VideoWriter::spawn_with_audio(&dir, 24, true, Some(Path::new("/no/such/track.wav")));
        assert!(err
            .err()
            .is_some_and(|e| e.to_string().contains("does not exist")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A batch that is not whole frames is refused, and the error reaches the
    /// caller at `finish` rather than being lost on the worker thread.
    #[test]
    fn writer_reports_worker_errors() {
        let dir = std::env::temp_dir().join(format!("fv-writer-bad-{}", std::process::id()));
        let mut writer = VideoWriter::spawn(&dir, 0, false).unwrap();
        writer.push(0, 4, 4, vec![0u8; 7]).unwrap();
        let err = writer.finish().unwrap_err().to_string();
        assert!(err.contains("not whole"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
