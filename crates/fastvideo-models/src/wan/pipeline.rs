//! Wan inference pipeline: UMT5 → DiT sampling → VAE decode → PNG frames.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::config::WanVideoArchConfig;
use super::family::{i2v_first_frame_mask, moe_expert, MoeExpert};
use super::transformer::WanTransformer3D;
use super::umt5::{pad_prompt_embeds, Umt5Config, Umt5Encoder};
use super::vae::{AutoencoderKlWan, WanVaeConfig};
use crate::schedulers::{DmdSchedule, FlowUniPCMultistepScheduler};

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
    /// First-frame path for I2V (PNG/JPEG). Required when the DiT is 36-channel.
    pub image_path: Option<String>,
    /// Wan 2.2 low-noise CFG. Falls back to `guidance_scale`.
    pub guidance_scale_2: Option<f32>,
    pub boundary_ratio: Option<f32>,
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
            guidance_scale_2: None,
            boundary_ratio: None,
        }
    }
}

pub struct WanPipeline {
    text: Umt5Encoder,
    transformer: WanTransformer3D,
    transformer_2: Option<WanTransformer3D>,
    vae: AutoencoderKlWan,
    device: Device,
    tiny: bool,
    dtype: DType,
    boundary_ratio: Option<f32>,
}

impl WanPipeline {
    pub fn tiny(device: &Device) -> Result<Self> {
        Self::tiny_dtype(device, DType::F32)
    }

    pub fn tiny_dtype(device: &Device, dtype: DType) -> Result<Self> {
        let vb = VarBuilder::zeros(dtype, device);
        Ok(Self {
            text: Umt5Encoder::load(Umt5Config::tiny(), vb.pp("text"))?,
            transformer: WanTransformer3D::load(WanVideoArchConfig::tiny(), vb.pp("dit"))?,
            transformer_2: None,
            vae: AutoencoderKlWan::load(WanVaeConfig::tiny(), vb.pp("vae"))?,
            device: device.clone(),
            tiny: true,
            dtype,
            boundary_ratio: None,
        })
    }

    pub fn load(
        transformer_vb: VarBuilder,
        vae_vb: VarBuilder,
        text_vb: VarBuilder,
        dit_cfg: WanVideoArchConfig,
        vae_cfg: WanVaeConfig,
        text_cfg: Umt5Config,
        device: Device,
        transformer_2_vb: Option<VarBuilder>,
    ) -> Result<Self> {
        let dtype = transformer_vb.dtype();
        let boundary_ratio = dit_cfg.boundary_ratio;
        let transformer_2 = match transformer_2_vb {
            Some(vb) => Some(WanTransformer3D::load(dit_cfg.clone(), vb)?),
            None => None,
        };
        Ok(Self {
            text: Umt5Encoder::load(text_cfg, text_vb)?,
            transformer: WanTransformer3D::load(dit_cfg, transformer_vb)?,
            transformer_2,
            vae: AutoencoderKlWan::load(vae_cfg, vae_vb)?,
            device,
            tiny: false,
            dtype,
            boundary_ratio,
        })
    }

    fn encode_ids(&self, ids: &[u32]) -> Result<Tensor> {
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        self.text.forward(&input, None)
    }

    fn encode_prompt(&self, cfg: &GenerateConfig) -> Result<Tensor> {
        let text_len = self.transformer.cfg.text_len;
        if let Some(tokenizer) = cfg.tokenizer_path.as_ref() {
            let (prompt_ids, prompt_len) = tokenize_prompt(tokenizer, &cfg.prompt, text_len)?;
            let (neg_ids, neg_len) = tokenize_prompt(tokenizer, &cfg.negative_prompt, text_len)?;
            let prompt_embeds =
                pad_prompt_embeds(&self.encode_ids(&prompt_ids)?, &[prompt_len], text_len)?;
            let neg_embeds = pad_prompt_embeds(&self.encode_ids(&neg_ids)?, &[neg_len], text_len)?;
            return Tensor::cat(&[&neg_embeds, &prompt_embeds], 0);
        }
        if self.tiny {
            let seq = text_len.min(8);
            let dummy: Vec<u32> = (0..seq).map(|i| (i % 10) as u32).collect();
            let prompt_embeds = self.encode_ids(&dummy)?;
            let neg_embeds = prompt_embeds.clone();
            let prompt_embeds = pad_prompt_embeds(&prompt_embeds, &[seq], text_len)?;
            let neg_embeds = pad_prompt_embeds(&neg_embeds, &[seq], text_len)?;
            return Tensor::cat(&[&neg_embeds, &prompt_embeds], 0);
        }
        Err(candle_core::Error::Msg(
            "real generate needs tokenizer.json next to the Diffusers weights".into(),
        ))
    }

    pub fn generate(&self, cfg: &GenerateConfig) -> Result<Vec<String>> {
        let (z_t, z_h, z_w, z_c) = if self.tiny {
            (2usize, 4usize, 4usize, 4usize)
        } else {
            (
                (cfg.num_frames.saturating_sub(1)) / 4 + 1,
                cfg.height / 8,
                cfg.width / 8,
                self.transformer.cfg.out_channels,
            )
        };
        let i2v = !self.tiny && self.transformer.cfg.in_channels > self.transformer.cfg.out_channels;
        if i2v && cfg.image_path.is_none() {
            candle_core::bail!("I2V generate needs --image <png|jpeg> for 36-channel latent packing");
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let n_el = z_c * z_t * z_h * z_w;
        let noise: Vec<f32> = (0..n_el)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let mut latents =
            Tensor::from_vec(noise, (1, z_c, z_t, z_h, z_w), &self.device)?.to_dtype(self.dtype)?;

        let i2v_pack = if i2v {
            let image = cfg.image_path.as_deref().unwrap();
            Some(self.encode_i2v_condition(image, cfg.height, cfg.width, z_t, z_h, z_w)?)
        } else {
            None
        };

        let encoder_hs = self.encode_prompt(cfg)?.to_dtype(self.dtype)?;
        let boundary = cfg.boundary_ratio.or(self.boundary_ratio);
        let ctx = DenoiseCtx {
            high: &self.transformer,
            low: self.transformer_2.as_ref(),
            boundary_ratio: boundary,
            image: None,
            i2v: i2v_pack.as_ref().map(|(mask, cond)| (mask, cond)),
            guidance: cfg.guidance_scale,
            guidance_2: cfg.guidance_scale_2.unwrap_or(cfg.guidance_scale),
            dtype: self.dtype,
        };

        if cfg.is_dmd {
            let steps = cfg
                .dmd_steps
                .clone()
                .unwrap_or_else(|| crate::schedulers::FAST_WAN_1_3B_DMD_STEPS.to_vec());
            let s = DmdSchedule::new(&steps, cfg.flow_shift, 1000);
            let timesteps: Vec<f32> = s.train_timesteps.iter().map(|&t| t as f32).collect();
            latents = euler_denoise(latents, &encoder_hs, &timesteps, &s.sigmas, &ctx)?;
        } else {
            let mut sched = FlowUniPCMultistepScheduler::new(1000, cfg.flow_shift);
            sched.set_timesteps(cfg.num_inference_steps);
            latents = unipc_denoise(latents, &encoder_hs, &mut sched, &ctx)?;
        }

        let latents = if self.tiny {
            latents
        } else {
            self.vae.scale_latents(&latents)?
        };
        let video = self.vae.decode(&latents)?;
        write_frames(&video, Path::new(&cfg.output_dir))
    }

    fn encode_i2v_condition(
        &self,
        image_path: &str,
        height: usize,
        width: usize,
        z_t: usize,
        z_h: usize,
        z_w: usize,
    ) -> Result<(Tensor, Tensor)> {
        let video = load_rgb_frame(image_path, height, width, &self.device)?.to_dtype(self.dtype)?;
        let encoded = self.vae.encode_video(&video)?;
        let encoded = self.vae.normalize_latents(&encoded)?;
        let first = encoded.narrow(2, 0, 1)?;
        let (_, c, _, eh, ew) = first.dims5()?;
        if eh != z_h || ew != z_w {
            candle_core::bail!(
                "I2V VAE latent spatial {eh}x{ew} does not match expected {z_h}x{z_w}"
            );
        }
        let rest_t = z_t.saturating_sub(1);
        let cond = if rest_t == 0 {
            first
        } else {
            let rest = Tensor::zeros((1, c, rest_t, z_h, z_w), encoded.dtype(), &self.device)?;
            Tensor::cat(&[&first, &rest], 2)?
        };
        let mask_c = self.transformer.cfg.in_channels.saturating_sub(2 * c).max(1);
        let mask = Tensor::from_vec(
            i2v_first_frame_mask(z_t, z_h, z_w),
            (1, 4, z_t, z_h, z_w),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        let mask = if mask_c == 4 {
            mask
        } else {
            mask.narrow(1, 0, mask_c)?
        };
        Ok((mask, cond.to_dtype(self.dtype)?))
    }
}

struct DenoiseCtx<'a> {
    high: &'a WanTransformer3D,
    low: Option<&'a WanTransformer3D>,
    boundary_ratio: Option<f32>,
    image: Option<&'a Tensor>,
    i2v: Option<(&'a Tensor, &'a Tensor)>,
    guidance: f32,
    guidance_2: f32,
    dtype: DType,
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

fn cfg_guide(uncond: &Tensor, text: &Tensor, scale: f32, dtype: DType) -> Result<Tensor> {
    if (scale - 1.0).abs() < 1e-6 {
        return Ok(text.clone());
    }
    let uncond_f = uncond.to_dtype(DType::F32)?;
    let text_f = text.to_dtype(DType::F32)?;
    (uncond_f.clone() + ((text_f - uncond_f)? * f64::from(scale))?)?.to_dtype(dtype)
}

fn pack_dit_input(latents: &Tensor, i2v: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
    let packed = if let Some((mask, cond)) = i2v {
        Tensor::cat(&[latents, mask, cond], 1)?
    } else {
        latents.clone()
    };
    Tensor::cat(&[&packed, &packed], 0)
}

fn dit_cfg(ctx: &DenoiseCtx, latents: &Tensor, encoder_hs: &Tensor, t: f32) -> Result<Tensor> {
    let (transformer, scale) = pick_expert(ctx, t);
    let device = latents.device();
    let t_tensor = Tensor::from_vec(vec![t, t], (2,), device)?.to_dtype(ctx.dtype)?;
    let latent_in = pack_dit_input(latents, ctx.i2v)?;
    let noise_pred = transformer.forward_ctx(&latent_in, &t_tensor, encoder_hs, ctx.image)?;
    let chunks = noise_pred.chunk(2, 0)?;
    cfg_guide(&chunks[0], &chunks[1], scale, ctx.dtype)
}

fn euler_denoise(
    mut latents: Tensor,
    encoder_hs: &Tensor,
    timesteps: &[f32],
    sigmas: &[f64],
    ctx: &DenoiseCtx,
) -> Result<Tensor> {
    for (i, &t) in timesteps.iter().enumerate() {
        let guided = dit_cfg(ctx, &latents, encoder_hs, t)?;
        let dt = sigmas[i + 1] - sigmas[i];
        let delta = (guided.to_dtype(DType::F32)? * dt)?.to_dtype(ctx.dtype)?;
        latents = (latents + delta)?;
    }
    Ok(latents)
}

fn unipc_denoise(
    mut latents: Tensor,
    encoder_hs: &Tensor,
    sched: &mut FlowUniPCMultistepScheduler,
    ctx: &DenoiseCtx,
) -> Result<Tensor> {
    let device = latents.device().clone();
    let shape = latents.dims().to_vec();
    let ts: Vec<f32> = sched
        .inference_timesteps_i64()
        .iter()
        .map(|t| *t as f32)
        .collect();
    for &t in &ts {
        let guided = dit_cfg(ctx, &latents, encoder_hs, t)?;
        let vel = guided
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let x = latents
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let prev = sched
            .step(&vel, &x)
            .map_err(|e| candle_core::Error::Msg(e))?;
        latents = Tensor::from_vec(prev, shape.clone(), &device)?.to_dtype(ctx.dtype)?;
    }
    Ok(latents)
}

fn load_rgb_frame(path: &str, height: usize, width: usize, device: &Device) -> Result<Tensor> {
    let img = image::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("image load failed: {e}")))?
        .resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb8();
    let mut data = Vec::with_capacity(3 * height * width);
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let p = img.get_pixel(x as u32, y as u32)[c];
                data.push(f32::from(p) / 127.5 - 1.0);
            }
        }
    }
    Tensor::from_vec(data, (1, 3, 1, height, width), device)
}

pub fn tokenize_prompt(path: &str, text: &str, max_len: usize) -> Result<(Vec<u32>, usize)> {
    let tokenizer = tokenizers::Tokenizer::from_file(path)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer load failed: {e}")))?;
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| candle_core::Error::Msg(format!("tokenize failed: {e}")))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    let len = ids.len().max(1);
    Ok((ids, len))
}

fn write_frames(video: &Tensor, dir: &Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let video = video
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .squeeze(0)?;
    let dims = video.dims();
    let (t, h, w) = match dims {
        [_c, t, h, w] => (*t, *h, *w),
        other => candle_core::bail!("expected CTHW video, got {other:?}"),
    };
    let video = ((video + 1.0)? * 127.5)?
        .clamp(0.0, 255.0)?
        .to_dtype(DType::U8)?;
    let mut paths = Vec::new();
    for ti in 0..t {
        let frame = video.narrow(1, ti, 1)?.squeeze(1)?.permute((1, 2, 0))?.contiguous()?;
        let data = frame.flatten_all()?.to_vec1::<u8>()?;
        let img = image::RgbImage::from_raw(w as u32, h as u32, data)
            .ok_or_else(|| candle_core::Error::Msg("rgb buffer size mismatch".into()))?;
        let path = dir.join(format!("frame-{ti:03}.png"));
        img.save(&path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        paths.push(path.to_string_lossy().into_owned());
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_pipeline_writes_png() {
        let pipe = WanPipeline::tiny(&Device::Cpu).unwrap();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.is_dmd = true;
        cfg.flow_shift = 8.0;
        cfg.guidance_scale = 1.0;
        cfg.output_dir = std::env::temp_dir()
            .join("fastvideo-tiny")
            .to_string_lossy()
            .into();
        let paths = pipe.generate(&cfg).unwrap();
        assert!(!paths.is_empty());
        assert!(Path::new(&paths[0]).exists());
    }

    #[test]
    fn umt5_tokenizer_is_not_dummy_ids() {
        let Some(root) = crate::wan::weights::local_wan_t2v_1_3b() else {
            eprintln!("skip: Wan2.1-T2V-1.3B-Diffusers not in HF cache");
            return;
        };
        let tok = root.join("tokenizer/tokenizer.json");
        assert!(tok.is_file(), "{}", tok.display());
        let prompt = "A curious raccoon in a field of sunflowers.";
        let (ids, len) = tokenize_prompt(tok.to_str().unwrap(), prompt, 512).unwrap();
        assert!(len > 4, "expected a real sentencepiece encoding, got {ids:?}");
        let dummy: Vec<u32> = (0..len as u32).collect();
        assert_ne!(ids, dummy, "tokenizer returned dummy sequential ids");
        // UMT5/T5: pad=0, eos=1. Encoding with special tokens ends in eos.
        assert_eq!(*ids.last().unwrap(), 1);
        assert!(ids.iter().any(|&id| id > 10));
    }

    #[test]
    fn load_rgb_frame_is_minus_one_to_one() {
        let dir = std::env::temp_dir().join("fastvideo-i2v-rgb");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cond.png");
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([255, 0, 128]));
        img.save(&path).unwrap();
        let t = load_rgb_frame(path.to_str().unwrap(), 16, 16, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 1, 16, 16]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[16 * 16] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn i2v_pack_concat_is_36_channels() {
        let device = Device::Cpu;
        let noisy = Tensor::zeros((1, 16, 2, 4, 4), DType::F32, &device).unwrap();
        let mask = Tensor::ones((1, 4, 2, 4, 4), DType::F32, &device).unwrap();
        let cond = Tensor::zeros((1, 16, 2, 4, 4), DType::F32, &device).unwrap();
        let packed = pack_dit_input(&noisy, Some((&mask, &cond))).unwrap();
        assert_eq!(packed.dims(), &[2, 36, 2, 4, 4]);
    }
}
