//! FLUX.1 T2I pipeline: CLIP+T5 → packed latents → flow-match Euler → VAE decode.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::flux2::{AutoencoderKlFlux2, Flux2VaeConfig};
use crate::schedulers::FlowMatchEulerDiscreteScheduler;

use super::config::Flux1ArchConfig;
use super::family::{calculate_shift_flux1, packed_hw, unpack_latents_flux1};
use super::text::{
    flux1_dummy_text, flux1_t5_len, pad_token_ids, tokenize_flux1, ClipTextConfig, ClipTextEncoder,
    Flux1TextEncoder, T5Config, T5Encoder,
};
use super::transformer::Flux1Transformer2D;
use super::weights::arch_from_transformer_config;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub prompt: String,
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    pub output_dir: String,
    pub tiny: bool,
    pub clip_tokenizer_path: Option<String>,
    pub t5_tokenizer_path: Option<String>,
    pub embedded_cfg_scale: Option<f32>,
    pub t5_max_len: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            prompt: "a photo of a banana on a wooden table, studio lighting".into(),
            height: 1024,
            width: 1024,
            num_inference_steps: 50,
            guidance_scale: 3.5,
            seed: 0,
            output_dir: "out".into(),
            tiny: false,
            clip_tokenizer_path: None,
            t5_tokenizer_path: None,
            embedded_cfg_scale: Some(3.5),
            t5_max_len: 512,
        }
    }
}

pub struct Flux1Pipeline {
    text: Flux1TextEncoder,
    transformer: Flux1Transformer2D,
    vae: AutoencoderKlFlux2,
    device: Device,
    tiny: bool,
    dtype: DType,
}

impl Flux1Pipeline {
    pub fn tiny(device: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, device);
        let dit_cfg = Flux1ArchConfig::tiny();
        Ok(Self {
            text: Flux1TextEncoder::dummy(
                dit_cfg.joint_attention_dim,
                dit_cfg.pooled_projection_dim,
                device,
                DType::F32,
            ),
            transformer: Flux1Transformer2D::load(dit_cfg, vb.pp("dit"))?,
            vae: AutoencoderKlFlux2::load(Flux2VaeConfig::tiny(), vb.pp("vae"))?,
            device: device.clone(),
            tiny: true,
            dtype: DType::F32,
        })
    }

    pub fn load(
        transformer_vb: VarBuilder,
        vae_vb: VarBuilder,
        clip_vb: Option<VarBuilder>,
        t5_vb: Option<VarBuilder>,
        dit_cfg: Flux1ArchConfig,
        vae_cfg: Flux2VaeConfig,
        device: Device,
        transformer_config_json: Option<&str>,
        t5_max_len: usize,
    ) -> Result<Self> {
        let dit_cfg = match transformer_config_json {
            Some(raw) => arch_from_transformer_config(&dit_cfg, raw).map_err(candle_core::Error::Msg)?,
            None => dit_cfg,
        };
        let dtype = transformer_vb.dtype();
        let text = if flux1_dummy_text() {
            Flux1TextEncoder::dummy(
                dit_cfg.joint_attention_dim,
                dit_cfg.pooled_projection_dim,
                &device,
                dtype,
            )
        } else {
            let clip = clip_vb
                .and_then(|vb| ClipTextEncoder::load(ClipTextConfig::clip_l(), vb).ok());
            let mut t5_cfg = T5Config::xxl();
            t5_cfg.text_len = t5_max_len.max(1);
            let t5 = t5_vb.and_then(|vb| T5Encoder::load(t5_cfg, vb).ok());
            Flux1TextEncoder::load(
                clip,
                t5,
                dit_cfg.joint_attention_dim,
                dit_cfg.pooled_projection_dim,
                device.clone(),
                dtype,
            )
        };
        Ok(Self {
            text,
            transformer: Flux1Transformer2D::load(dit_cfg, transformer_vb)?,
            vae: AutoencoderKlFlux2::load(vae_cfg, vae_vb)?,
            device,
            tiny: false,
            dtype,
        })
    }

    pub fn generate(&self, cfg: &GenerateConfig) -> Result<Vec<String>> {
        let (height, width, steps) = if self.tiny {
            (16usize, 16, 2usize)
        } else {
            (cfg.height.max(16), cfg.width.max(16), cfg.num_inference_steps.max(1))
        };
        let scale = self.vae.cfg.spatial_compression_ratio.max(1);
        let (ph, pw) = if self.tiny {
            (2usize, 2)
        } else {
            packed_hw(height, width, scale)
        };
        let channels = self.transformer.cfg.in_channels;
        let seq = ph * pw;
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let noise: Vec<f32> = (0..channels * seq).map(|_| rng.sample(StandardNormal)).collect();
        let mut latents =
            Tensor::from_vec(noise, (1, channels, 1, ph, pw), &self.device)?.to_dtype(self.dtype)?;
        let (clip_ids, t5_ids) = encode_prompt_ids(&self.text, cfg, self.tiny);
        let (encoder, pooled) = self.text.encode(&clip_ids, &t5_ids)?;
        let mu = calculate_shift_flux1(seq);
        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, 1.0);
        sched.set_timesteps_flux2(steps, Some(mu));
        let guidance = if self.transformer.cfg.guidance_embeds {
            let g = cfg.embedded_cfg_scale.unwrap_or(cfg.guidance_scale);
            Some(Tensor::from_vec(vec![g], (1,), &self.device)?.to_dtype(self.dtype)?)
        } else {
            None
        };
        for t in sched.inference_timesteps().to_vec() {
            let timestep =
                Tensor::from_vec(vec![t as f32 / 1000.0], (1,), &self.device)?.to_dtype(self.dtype)?;
            let vel = self.transformer.forward(
                &latents,
                &encoder,
                &pooled,
                &timestep,
                guidance.as_ref(),
                ph,
                pw,
            )?;
            let x = latents.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let v = vel.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let next = sched.step_euler(&x, &v).map_err(candle_core::Error::Msg)?;
            latents = Tensor::from_vec(next, latents.dims(), &self.device)?.to_dtype(self.dtype)?;
        }
        let unpacked = if self.tiny {
            latents.clone()
        } else {
            unpack_latents(&latents, self.vae.cfg.latent_channels, ph, pw, &self.device, self.dtype)?
        };
        let frames = self.vae.decode(&unpacked)?;
        write_image(&frames, Path::new(&cfg.output_dir))
    }
}

fn encode_prompt_ids(
    text: &Flux1TextEncoder,
    cfg: &GenerateConfig,
    tiny: bool,
) -> (Vec<u32>, Vec<u32>) {
    if tiny {
        return ((0..4).collect(), (0..4).collect());
    }
    let clip_len = text.clip_config().map(|c| c.text_len).unwrap_or(77);
    let t5_default = text.t5_config().map(|c| c.text_len).unwrap_or(cfg.t5_max_len);
    let t5_len = flux1_t5_len(t5_default);
    let clip_pad = text.clip_config().map(|c| c.pad_token_id).unwrap_or(0);
    let t5_pad = text.t5_config().map(|c| c.pad_token_id).unwrap_or(0);
    let clip_raw = if let Some(tok) = &cfg.clip_tokenizer_path {
        tokenize_flux1(tok, &cfg.prompt, clip_len)
            .map(|(ids, _)| ids)
            .unwrap_or_else(|_| hash_ids(&cfg.prompt, clip_len.min(16)))
    } else {
        hash_ids(&cfg.prompt, clip_len.min(16))
    };
    let t5_raw = if let Some(tok) = &cfg.t5_tokenizer_path {
        tokenize_flux1(tok, &cfg.prompt, t5_len)
            .map(|(ids, _)| ids)
            .unwrap_or_else(|_| hash_ids(&cfg.prompt, t5_len.min(32)))
    } else {
        hash_ids(&cfg.prompt, t5_len.min(32))
    };
    let (clip_ids, _) = pad_token_ids(&clip_raw, clip_len, clip_pad);
    let (t5_ids, _) = pad_token_ids(&t5_raw, t5_len, t5_pad);
    (clip_ids, t5_ids)
}

fn hash_ids(prompt: &str, len: usize) -> Vec<u32> {
    prompt
        .bytes()
        .chain(0u8..len as u8)
        .take(len)
        .map(|b| b as u32)
        .collect()
}

fn unpack_latents(
    packed: &Tensor,
    latent_channels: usize,
    ph: usize,
    pw: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let host = packed.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    // Packed storage is NCHW [C*4, ph, pw]; Diffusers unpack wants [seq, C*4].
    let seq = ph * pw;
    let packed_ch = host.len() / seq;
    let mut seq_major = vec![0.0f32; host.len()];
    for s in 0..seq {
        for c in 0..packed_ch {
            seq_major[s * packed_ch + c] = host[c * seq + s];
        }
    }
    let spatial = unpack_latents_flux1(&seq_major, latent_channels, ph, pw)
        .map_err(candle_core::Error::Msg)?;
    Tensor::from_vec(spatial, (1, latent_channels, 1, ph * 2, pw * 2), device)?.to_dtype(dtype)
}

fn write_image(video: &Tensor, dir: &Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let video = video.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.squeeze(0)?;
    let dims = video.dims();
    let (t, h, w) = match dims {
        [_c, t, h, w] => (*t, *h, *w),
        other => candle_core::bail!("expected CTHW image, got {other:?}"),
    };
    let video = ((video + 1.0)? * 127.5)?.clamp(0.0, 255.0)?.to_dtype(DType::U8)?;
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
    fn tiny_writes_png() {
        let pipe = Flux1Pipeline::tiny(&Device::Cpu).unwrap();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.output_dir = std::env::var("FASTVIDEO_ARTIFACT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("flux1-tiny")
            .to_string_lossy()
            .into();
        let paths = pipe.generate(&cfg).unwrap();
        assert!(!paths.is_empty());
        assert!(Path::new(&paths[0]).exists());
    }
}
