//! Flux2 T2I pipeline: text → packed latents → flow-match Euler → VAE decode.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::schedulers::FlowMatchEulerDiscreteScheduler;

use super::config::{Flux2ArchConfig, Flux2VaeConfig};
use super::family::{compute_empirical_mu, packed_hw};
use super::text::{
    flux2_dummy_text, flux2_text_len, format_flux2_prompt, pad_token_ids, tokenize_flux2,
    Flux2TextEncoder, Flux2TextKind, Qwen3Config, Qwen3Encoder,
};
use super::transformer::Flux2Transformer2D;
use super::vae::AutoencoderKlFlux2;
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
    pub tokenizer_path: Option<String>,
    pub embedded_cfg_scale: Option<f32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            prompt: "a photo of a banana on a wooden table, studio lighting".into(),
            height: 1024,
            width: 1024,
            num_inference_steps: 50,
            guidance_scale: 4.0,
            seed: 0,
            output_dir: "out".into(),
            tiny: false,
            tokenizer_path: None,
            embedded_cfg_scale: Some(4.0),
        }
    }
}

pub struct Flux2Pipeline {
    text: Flux2TextEncoder,
    transformer: Flux2Transformer2D,
    vae: AutoencoderKlFlux2,
    device: Device,
    tiny: bool,
    dtype: DType,
}

impl Flux2Pipeline {
    pub fn tiny(device: &Device) -> Result<Self> {
        Self::tiny_kind(device, DType::F32, Flux2TextKind::Mistral3)
    }

    pub fn tiny_klein(device: &Device) -> Result<Self> {
        Self::tiny_kind(device, DType::F32, Flux2TextKind::Qwen3)
    }

    pub fn tiny_kind(device: &Device, dtype: DType, kind: Flux2TextKind) -> Result<Self> {
        let vb = VarBuilder::zeros(dtype, device);
        let dit_cfg = if kind == Flux2TextKind::Qwen3 {
            Flux2ArchConfig::tiny_klein()
        } else {
            Flux2ArchConfig::tiny()
        };
        let text = if kind == Flux2TextKind::Qwen3 {
            let enc = Qwen3Encoder::load(Qwen3Config::tiny(), vb.pp("text"))?;
            Flux2TextEncoder::qwen3(enc, dtype)
        } else {
            Flux2TextEncoder::dummy(kind, dit_cfg.joint_attention_dim, device, dtype)
        };
        Ok(Self {
            text,
            transformer: Flux2Transformer2D::load(dit_cfg, vb.pp("dit"))?,
            vae: AutoencoderKlFlux2::load(Flux2VaeConfig::tiny(), vb.pp("vae"))?,
            device: device.clone(),
            tiny: true,
            dtype,
        })
    }

    pub fn load(
        transformer_vb: VarBuilder,
        vae_vb: VarBuilder,
        text_vb: Option<VarBuilder>,
        dit_cfg: Flux2ArchConfig,
        vae_cfg: Flux2VaeConfig,
        kind: Flux2TextKind,
        device: Device,
        transformer_config_json: Option<&str>,
    ) -> Result<Self> {
        Self::load_with_text_config(
            transformer_vb,
            vae_vb,
            text_vb,
            dit_cfg,
            vae_cfg,
            kind,
            device,
            transformer_config_json,
            None,
        )
    }

    /// Same as [`Self::load`] but honor `text_encoder/config.json` width / depth.
    pub fn load_with_text_config(
        transformer_vb: VarBuilder,
        vae_vb: VarBuilder,
        text_vb: Option<VarBuilder>,
        dit_cfg: Flux2ArchConfig,
        vae_cfg: Flux2VaeConfig,
        kind: Flux2TextKind,
        device: Device,
        transformer_config_json: Option<&str>,
        text_config_json: Option<&str>,
    ) -> Result<Self> {
        let dit_cfg = match transformer_config_json {
            Some(raw) => arch_from_transformer_config(&dit_cfg, raw)
                .map_err(|e| candle_core::Error::Msg(e))?,
            None => dit_cfg,
        };
        let dtype = transformer_vb.dtype();
        let lm_cfg = match text_config_json {
            Some(raw) => Qwen3Config::from_hf_json(kind, raw).map_err(candle_core::Error::Msg)?,
            None => match kind {
                Flux2TextKind::Qwen3 => Qwen3Config::klein_4b(),
                Flux2TextKind::Mistral3 => Qwen3Config::mistral3_24b(),
            },
        };
        let text = if flux2_dummy_text() {
            Flux2TextEncoder::dummy(kind, dit_cfg.joint_attention_dim, &device, dtype)
        } else {
            match (kind, text_vb) {
                (Flux2TextKind::Qwen3, Some(vb)) => {
                    Flux2TextEncoder::qwen3(Qwen3Encoder::load(lm_cfg, vb)?, dtype)
                }
                (Flux2TextKind::Mistral3, Some(vb)) => {
                    Flux2TextEncoder::mistral3(Qwen3Encoder::load(lm_cfg, vb)?, dtype)
                }
                _ => Flux2TextEncoder::dummy(kind, dit_cfg.joint_attention_dim, &device, dtype),
            }
        };
        Ok(Self {
            text,
            transformer: Flux2Transformer2D::load(dit_cfg, transformer_vb)?,
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
        let mut latents = Tensor::from_vec(noise, (1, channels, 1, ph, pw), &self.device)?.to_dtype(self.dtype)?;
        let (ids, valid_len) = encode_prompt_ids(&self.text, cfg, self.tiny, self.transformer.cfg.joint_attention_dim);
        let encoder = self.text.encode_ids_masked(&ids, valid_len)?;
        let mu = compute_empirical_mu(seq, steps);
        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, 1.0);
        sched.set_timesteps_flux2(steps, Some(mu));
        let guidance = if self.transformer.cfg.guidance_embeds {
            let g = cfg.embedded_cfg_scale.unwrap_or(cfg.guidance_scale);
            Some(Tensor::from_vec(vec![g], (1,), &self.device)?.to_dtype(self.dtype)?)
        } else {
            None
        };
        let timesteps: Vec<f64> = sched.inference_timesteps().to_vec();
        for t in timesteps {
            let timestep = Tensor::from_vec(vec![t as f32 / 1000.0], (1,), &self.device)?.to_dtype(self.dtype)?;
            let vel = self.transformer.forward(
                &latents,
                &encoder,
                &timestep,
                guidance.as_ref(),
                ph,
                pw,
            )?;
            let x = latents.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let v = vel.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let next = sched
                .step_euler(&x, &v)
                .map_err(|e| candle_core::Error::Msg(e))?;
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
    text: &Flux2TextEncoder,
    cfg: &GenerateConfig,
    tiny: bool,
    joint_dim: usize,
) -> (Vec<u32>, Option<usize>) {
    if tiny {
        let n = joint_dim.min(8) as u32;
        return ((0..n).collect(), None);
    }
    let default_len = text.lm_config().map(|c| c.text_len).unwrap_or(512);
    let text_len = flux2_text_len(default_len);
    let pad_id = text.lm_config().map(|c| c.pad_token_id).unwrap_or(0);
    let formatted = format_flux2_prompt(text.kind, &cfg.prompt);
    let raw = if let Some(tok) = &cfg.tokenizer_path {
        tokenize_flux2(tok, &formatted, text_len)
            .or_else(|_| crate::wan::tokenize_prompt(tok, &cfg.prompt, text_len))
            .map(|(ids, _)| ids)
            .unwrap_or_else(|_| hash_ids(&cfg.prompt, text_len.min(32)))
    } else {
        hash_ids(&cfg.prompt, text_len.min(32))
    };
    let (ids, valid) = pad_token_ids(&raw, text_len, pad_id);
    (ids, Some(valid))
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
    let host = packed
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let spatial = super::family::unpatchify_2x2(&host, latent_channels, ph, pw)
        .map_err(|e| candle_core::Error::Msg(e))?;
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
    fn tiny_dev_writes_png() {
        let pipe = Flux2Pipeline::tiny(&Device::Cpu).unwrap();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.output_dir = std::env::var("FASTVIDEO_ARTIFACT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("flux2-tiny")
            .to_string_lossy()
            .into();
        let paths = pipe.generate(&cfg).unwrap();
        assert!(!paths.is_empty());
        assert!(Path::new(&paths[0]).exists());
    }

    #[test]
    fn tiny_klein_writes_png() {
        let pipe = Flux2Pipeline::tiny_klein(&Device::Cpu).unwrap();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.embedded_cfg_scale = None;
        cfg.output_dir = std::env::var("FASTVIDEO_ARTIFACT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("flux2-tiny-klein")
            .to_string_lossy()
            .into();
        let paths = pipe.generate(&cfg).unwrap();
        assert!(Path::new(&paths[0]).exists());
    }
}
