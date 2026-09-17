//! Wan inference pipeline: UMT5 → DiT sampling → VAE decode → PNG / optional MP4.
//!
//! Behavioral parity target: Candle `fastvideo_models::wan::pipeline` (MoE, I2V, dual CFG).

use std::path::Path;
use std::process::Command;

use fastvideo_models::schedulers::{
    DmdSchedule, FlowUniPCMultistepScheduler, FAST_WAN_1_3B_DMD_STEPS,
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

/// `FASTVIDEO_DEVICE_STATS=1`: print per-op device-vs-host dispatch counts
/// (see `tensor::device_path_stats`) after a `generate()` call. Complements
/// `FASTVIDEO_STRICT_DEVICE=1` (which hard-fails on an unexpected host
/// fallback): this is the non-fatal version — a quick "how much of this run
/// actually used the GPU" readout, safe to leave on for any real GPU run
/// without risking a crash if something's a known, accepted host-only case.
fn log_device_path_stats_if_enabled() {
    if !super::envflag::bool_flag("FASTVIDEO_DEVICE_STATS", false) {
        return;
    }
    for (op, hits, misses) in super::tensor::device_path_stats() {
        let total = hits + misses;
        if total == 0 {
            continue;
        }
        super::log::info(format_args!(
            "device-path[{op}]: {hits}/{total} device ({misses} host fallback)"
        ));
    }
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
    pub flow_shift: f64,
    pub dmd_steps: Option<Vec<i32>>,
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
            flow_shift: 5.0,
            dmd_steps: None,
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
        Ok(Self {
            text,
            dit: WanTransformer3D::load(cfg, &dit)?,
            dit_2,
            vae: AutoencoderKlWan::load(vae_cfg, &vae)?,
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
            let (neg_ids, neg_len) =
                tokenize_prompt(tokenizer, &cfg.negative_prompt, text_len)?;
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
        if needs_control
            && cfg.control_path.is_none()
            && cfg.image_path.is_none()
        {
            return Err(PipelineError::Message(format!(
                "preset `{}` needs --control <png|jpeg> (or --image) for control/edit \
                 conditioning. Pass a reference frame to enable the Fun Control / Lucy path.",
                self.preset
            )));
        }

        let (z_c, z_t, z_h, z_w) = self.latent_shape(cfg);
        super::log::info(format_args!(
            "generate preset={} tiny={} {}x{} frames={} steps={} dmd={} latent=1x{}x{}x{}x{} \
             resident={} cuda={}",
            self.preset,
            self.tiny,
            cfg.width,
            cfg.height,
            cfg.num_frames,
            cfg.num_inference_steps,
            cfg.is_dmd,
            z_c,
            z_t,
            z_h,
            z_w,
            super::resident::residency_enabled(),
            cuda_context_live(),
        ));
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
        super::log::info(format_args!("wrote {} png frames → {}", paths.len(), cfg.output_dir));
        log_device_path_stats_if_enabled();
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
    pub fn initial_latents(&self, cfg: &GenerateConfig) -> Result<CudaTensor> {
        let (z_c, z_t, z_h, z_w) = self.latent_shape(cfg);
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let n_el = z_c * z_t * z_h * z_w;
        let noise: Vec<f32> = (0..n_el)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        Ok(CudaTensor::from_vec(noise, vec![1, z_c, z_t, z_h, z_w])?)
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
        let boundary = cfg.boundary_ratio.or(self.boundary_ratio);
        let ctx = DenoiseCtx {
            high: &self.dit,
            low: self.dit_2.as_ref(),
            boundary_ratio: boundary,
            image,
            i2v,
            guidance: cfg.guidance_scale,
            guidance_2: cfg.guidance_scale_2.unwrap_or(cfg.guidance_scale),
            tea_cache: TeaCache::from_env(),
        };
        if cfg.is_dmd {
            let steps = cfg
                .dmd_steps
                .clone()
                .unwrap_or_else(|| FAST_WAN_1_3B_DMD_STEPS.to_vec());
            super::log::info(format_args!("denoise=dmd steps={}", steps.len()));
            let s = DmdSchedule::new(&steps, cfg.flow_shift, 1000);
            let timesteps: Vec<f32> = s.train_timesteps.iter().map(|&t| t as f32).collect();
            let _denoise = super::log::StepTimer::start(format!("dmd {} steps", timesteps.len()));
            euler_denoise(latents, encoder_hs, &timesteps, &s.sigmas, &ctx, observer)
        } else {
            let mut sched = FlowUniPCMultistepScheduler::new(1000, cfg.flow_shift);
            sched.set_timesteps(cfg.num_inference_steps);
            super::log::info(format_args!(
                "denoise=unipc steps={}",
                cfg.num_inference_steps
            ));
            let _denoise =
                super::log::StepTimer::start(format!("unipc {} steps", cfg.num_inference_steps));
            unipc_denoise(latents, encoder_hs, &mut sched, &ctx, observer)
        }
    }

    /// Un-normalize latents and run the feat-cache VAE decode → `[1, 3, F, H, W]` in `[-1, 1]`.
    pub fn decode_latents(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let latents = if !self.tiny {
            self.vae.scale_latents(latents)?
        } else {
            latents.clone()
        };
        // Mark latents as device-resident for the VAE path. The denoise loop
        // already returned device-fresh tensors (dit_cfg returns device_fresh
        // outputs and axpy/add preserve the flag). VAE decode currently uses
        // the host path (causal_conv3d_f32 → cuDNN via host upload), so we
        // don't try to keep the latent device-resident across the VAE call —
        // but we log the residency state so it shows up in bench artifacts.
        let device_fresh_state = {
            #[cfg(feature = "cuda")]
            {
                latents.is_device_fresh()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        super::log::info(format_args!(
            "vae.in device_fresh={} cuda={}",
            device_fresh_state,
            cuda_context_live(),
        ));
        let _vae = super::log::StepTimer::start("vae.decode");
        Ok(self.vae.decode(&latents)?)
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
        let coefficients = [2.39676752e3, -1.31110545e3, 2.01331979e2, -8.29855975, 1.37887774e-1];
        let ret_steps = std::env::var("FASTVIDEO_TEACACHE_RET")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let coefficients = if ret_steps {
            // use_ret_steps=True coeffs for 1.3B
            [-5.21862437e4, 9.23041404e3, -5.28275948e2, 1.36987616e1, -4.99875664e-2]
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

fn dit_cfg(
    ctx: &mut DenoiseCtx<'_>,
    latents: &CudaTensor,
    encoder_hs: &CudaTensor,
    t: f32,
) -> TensorResult<CudaTensor> {
    if let Some(cached) = ctx.tea_cache.maybe_reuse(latents) {
        static TEA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = TEA.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        super::log::debug(format_args!("teacache hit #{n}"));
        return Ok(cached);
    }
    let (dit, scale) = pick_expert(ctx, t);
    let latent_in = pack_dit_input(latents, ctx.i2v)?;
    let cond_hs = encoder_hs.narrow(0, 1, 1)?;

    let out = if (scale - 1.0).abs() < 1e-6 {
        let t_tensor = CudaTensor::from_vec(vec![t], vec![1])?;
        dit.forward_ctx(&latent_in, &t_tensor, &cond_hs, ctx.image)?
    } else if ctx.i2v.is_none() && ctx.image.is_none() {
        // Batch cond+uncond into one forward pass instead of two sequential
        // batch=1 passes: doubles the arithmetic intensity of every GEMM in
        // the step and halves the kernel-launch count for the whole DiT
        // (dozens of blocks × ~15 launches each), instead of paying that
        // twice. Scoped to the no-I2V/no-image-conditioning path: batching
        // those would also need `mask`/`cond`/`image` duplicated to batch=2,
        // and getting that wrong silently mixes cond/uncond image tokens —
        // not worth guessing at without a GPU to check numerics against, so
        // I2V/image conditioning keeps the sequential two-pass path below.
        let uncond_hs = encoder_hs.narrow(0, 0, 1)?;
        let latent_batch = CudaTensor::cat(&[&latent_in, &latent_in], 0)?;
        let hs_batch = CudaTensor::cat(&[&uncond_hs, &cond_hs], 0)?;
        let t_batch = CudaTensor::from_vec(vec![t, t], vec![2])?;
        let out_batch = dit.forward_ctx(&latent_batch, &t_batch, &hs_batch, None)?;
        let uncond = out_batch.narrow(0, 0, 1)?;
        let cond = out_batch.narrow(0, 1, 1)?;
        let delta = cond.sub(&uncond)?;
        uncond.add(&delta.mul_scalar(scale))?
    } else {
        let t_tensor = CudaTensor::from_vec(vec![t], vec![1])?;
        let uncond_hs = encoder_hs.narrow(0, 0, 1)?;
        let cond = dit.forward_ctx(&latent_in, &t_tensor, &cond_hs, ctx.image)?;
        let uncond = dit.forward_ctx(&latent_in, &t_tensor, &uncond_hs, ctx.image)?;
        let delta = cond.sub(&uncond)?;
        uncond.add(&delta.mul_scalar(scale))?
    };
    ctx.tea_cache.store(latents, &out);
    Ok(out)
}

fn euler_denoise(
    mut latents: CudaTensor,
    encoder_hs: &CudaTensor,
    timesteps: &[f32],
    sigmas: &[f64],
    ctx: &DenoiseCtx<'_>,
    mut observer: Option<&mut StepObserver<'_>>,
) -> Result<CudaTensor> {
    let mut ctx = DenoiseCtx {
        high: ctx.high,
        low: ctx.low,
        boundary_ratio: ctx.boundary_ratio,
        image: ctx.image,
        i2v: ctx.i2v,
        guidance: ctx.guidance,
        guidance_2: ctx.guidance_2,
        tea_cache: TeaCache::from_env(),
    };
    for (i, &t) in timesteps.iter().enumerate() {
        let _step = super::log::StepTimer::start(format!(
            "dmd step {}/{} t={t:.0}",
            i + 1,
            timesteps.len()
        ));
        let guided = dit_cfg(&mut ctx, &latents, encoder_hs, t)?;
        let dt = sigmas[i + 1] - sigmas[i];
        let delta = guided.mul_scalar(dt as f32);
        latents = latents.add(&delta)?;
        if let Some(obs) = observer.as_deref_mut() {
            obs(&DenoiseStep {
                index: i,
                total: timesteps.len(),
                timestep: t,
                latents: &latents,
            })?;
        }
    }
    Ok(latents)
}

fn unipc_denoise(
    mut latents: CudaTensor,
    encoder_hs: &CudaTensor,
    sched: &mut FlowUniPCMultistepScheduler,
    ctx: &DenoiseCtx<'_>,
    mut observer: Option<&mut StepObserver<'_>>,
) -> Result<CudaTensor> {
    let mut ctx = DenoiseCtx {
        high: ctx.high,
        low: ctx.low,
        boundary_ratio: ctx.boundary_ratio,
        image: ctx.image,
        i2v: ctx.i2v,
        guidance: ctx.guidance,
        guidance_2: ctx.guidance_2,
        tea_cache: TeaCache::from_env(),
    };
    let device_sched = super::hopper::device_sched_enabled()
        && super::resident::residency_enabled()
        && cuda_context_live()
        && sched.predict_x0;
    if device_sched {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        super::log::info_once(
            &ONCE,
            format_args!("unipc: device order-1 axpy (FASTVIDEO_DEVICE_SCHED)"),
        );
        let ts: Vec<f32> = sched
            .inference_timesteps_i64()
            .iter()
            .map(|t| *t as f32)
            .collect();
        for (i, &t) in ts.iter().enumerate() {
            let _step = super::log::StepTimer::start(format!(
                "unipc-dev step {}/{} t={t:.0}",
                i + 1,
                ts.len()
            ));
            let guided = dit_cfg(&mut ctx, &latents, encoder_hs, t)?;
            let coeffs = sched
                .order1_device_coeffs()
                .map_err(PipelineError::Message)?;
            let converted = latents.add(&guided.mul_scalar(-coeffs.sigma_cur))?;
            latents = latents
                .mul_scalar(coeffs.scale_sample)
                .add(&converted.mul_scalar(coeffs.scale_converted))?;
            if let Some(obs) = observer.as_deref_mut() {
                obs(&DenoiseStep {
                    index: i,
                    total: ts.len(),
                    timestep: t,
                    latents: &latents,
                })?;
            }
        }
        return Ok(latents);
    }
    let shape = latents.shape.clone();
    let ts: Vec<f32> = sched
        .inference_timesteps_i64()
        .iter()
        .map(|t| *t as f32)
        .collect();
    for (i, &t) in ts.iter().enumerate() {
        let _step = super::log::StepTimer::start(format!(
            "unipc step {}/{} t={t:.0}",
            i + 1,
            ts.len()
        ));
        let mut guided = dit_cfg(&mut ctx, &latents, encoder_hs, t)?;
        guided.ensure_host().map_err(|e| PipelineError::Message(e.to_string()))?;
        latents
            .ensure_host()
            .map_err(|e| PipelineError::Message(e.to_string()))?;
        let prev = sched
            .step(&guided.data, &latents.data)
            .map_err(PipelineError::Message)?;
        // Re-upload the updated latents so the next step starts device-fresh.
        // Honors the residency contract through the VAE call.
        let mut fresh = CudaTensor::from_vec(prev, shape.clone())?;
        let _ = fresh.pin_device();
        latents = fresh;
        if let Some(obs) = observer.as_deref_mut() {
            obs(&DenoiseStep {
                index: i,
                total: ts.len(),
                timestep: t,
                latents: &latents,
            })?;
        }
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
    std::fs::create_dir_all(dir).map_err(|e| PipelineError::Message(e.to_string()))?;
    if video.rank() != 5 || video.shape[0] != 1 {
        return Err(PipelineError::Message(format!(
            "expected 1CTHW video, got {:?}",
            video.shape
        )));
    }
    let (_b, c, t, h, w) = (
        video.shape[0],
        video.shape[1],
        video.shape[2],
        video.shape[3],
        video.shape[4],
    );
    if c < 3 {
        return Err(PipelineError::Message("video needs RGB channels".into()));
    }
    let host = video
        .host_cow()
        .map_err(|e| PipelineError::Message(e.to_string()))?;
    let mut paths = Vec::new();
    for ti in 0..t {
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let v = host[(((0 * c + ch) * t + ti) * h + y) * w + x];
                    let byte = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                    rgb[(y * w + x) * 3 + ch] = byte;
                }
            }
        }
        let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
            .ok_or_else(|| PipelineError::Message("rgb buffer size mismatch".into()))?;
        let path = dir.join(format!("frame-{ti:03}.png"));
        img.save(&path)
            .map_err(|e| PipelineError::Message(e.to_string()))?;
        paths.push(path.to_string_lossy().into_owned());
    }
    Ok(paths)
}

/// Mux `frame-%03d.png` in `dir` into `output.mp4` with ffmpeg (libx264).
pub fn mux_mp4(dir: &Path, fps: u32) -> Result<String> {
    let out = dir.join("output.mp4");
    let pattern = dir.join("frame-%03d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
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
