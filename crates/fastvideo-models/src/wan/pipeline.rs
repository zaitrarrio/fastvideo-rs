//! Wan inference pipeline: UMT5 → DiT sampling → VAE decode → PNG frames.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::config::WanVideoArchConfig;
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
        }
    }
}

pub struct WanPipeline {
    text: Umt5Encoder,
    transformer: WanTransformer3D,
    vae: AutoencoderKlWan,
    device: Device,
    tiny: bool,
    dtype: DType,
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
            vae: AutoencoderKlWan::load(WanVaeConfig::tiny(), vb.pp("vae"))?,
            device: device.clone(),
            tiny: true,
            dtype,
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
    ) -> Result<Self> {
        let dtype = transformer_vb.dtype();
        Ok(Self {
            text: Umt5Encoder::load(text_cfg, text_vb)?,
            transformer: WanTransformer3D::load(dit_cfg, transformer_vb)?,
            vae: AutoencoderKlWan::load(vae_cfg, vae_vb)?,
            device,
            tiny: false,
            dtype,
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let n_el = z_c * z_t * z_h * z_w;
        let noise: Vec<f32> = (0..n_el)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let mut latents =
            Tensor::from_vec(noise, (1, z_c, z_t, z_h, z_w), &self.device)?.to_dtype(self.dtype)?;

        let encoder_hs = self.encode_prompt(cfg)?.to_dtype(self.dtype)?;

        if cfg.is_dmd {
            let steps = cfg
                .dmd_steps
                .clone()
                .unwrap_or_else(|| crate::schedulers::FAST_WAN_1_3B_DMD_STEPS.to_vec());
            let s = DmdSchedule::new(&steps, cfg.flow_shift, 1000);
            let timesteps: Vec<f32> = s.train_timesteps.iter().map(|&t| t as f32).collect();
            latents = euler_denoise(
                &self.transformer,
                latents,
                &encoder_hs,
                &timesteps,
                &s.sigmas,
                cfg.guidance_scale,
                self.dtype,
            )?;
        } else {
            let mut sched = FlowUniPCMultistepScheduler::new(1000, cfg.flow_shift);
            sched.set_timesteps(cfg.num_inference_steps);
            latents = unipc_denoise(
                &self.transformer,
                latents,
                &encoder_hs,
                &mut sched,
                cfg.guidance_scale,
                self.dtype,
            )?;
        }

        let latents = if self.tiny {
            latents
        } else {
            self.vae.scale_latents(&latents)?
        };
        let video = self.vae.decode(&latents)?;
        write_frames(&video, Path::new(&cfg.output_dir))
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

fn dit_cfg(
    transformer: &WanTransformer3D,
    latents: &Tensor,
    encoder_hs: &Tensor,
    t: f32,
    guidance_scale: f32,
    dtype: DType,
) -> Result<Tensor> {
    let device = latents.device();
    let t_tensor = Tensor::from_vec(vec![t, t], (2,), device)?.to_dtype(dtype)?;
    let latent_in = Tensor::cat(&[latents, latents], 0)?;
    let noise_pred = transformer.forward(&latent_in, &t_tensor, encoder_hs)?;
    let chunks = noise_pred.chunk(2, 0)?;
    cfg_guide(&chunks[0], &chunks[1], guidance_scale, dtype)
}

fn euler_denoise(
    transformer: &WanTransformer3D,
    mut latents: Tensor,
    encoder_hs: &Tensor,
    timesteps: &[f32],
    sigmas: &[f64],
    guidance_scale: f32,
    dtype: DType,
) -> Result<Tensor> {
    for (i, &t) in timesteps.iter().enumerate() {
        let guided = dit_cfg(transformer, &latents, encoder_hs, t, guidance_scale, dtype)?;
        let dt = sigmas[i + 1] - sigmas[i];
        let delta = (guided.to_dtype(DType::F32)? * dt)?.to_dtype(dtype)?;
        latents = (latents + delta)?;
    }
    Ok(latents)
}

fn unipc_denoise(
    transformer: &WanTransformer3D,
    mut latents: Tensor,
    encoder_hs: &Tensor,
    sched: &mut FlowUniPCMultistepScheduler,
    guidance_scale: f32,
    dtype: DType,
) -> Result<Tensor> {
    let device = latents.device().clone();
    let shape = latents.dims().to_vec();
    let ts: Vec<f32> = sched
        .inference_timesteps_i64()
        .iter()
        .map(|t| *t as f32)
        .collect();
    for &t in &ts {
        let guided = dit_cfg(transformer, &latents, encoder_hs, t, guidance_scale, dtype)?;
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
        latents = Tensor::from_vec(prev, shape.clone(), &device)?.to_dtype(dtype)?;
    }
    Ok(latents)
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
}
