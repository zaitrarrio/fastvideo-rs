//! Flux2 T2I generate path on cudarc (`CudaTensor`).

use std::path::Path;

use fastvideo_models::flux2::{
    arch_from_transformer_config, compute_empirical_mu, packed_hw, tokenize_flux2, unpatchify_2x2,
    Flux2ArchConfig, Flux2TextKind, Flux2VaeConfig, Qwen3Config,
};
use fastvideo_models::schedulers::FlowMatchEulerDiscreteScheduler;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use thiserror::Error;

use crate::wan::pipeline::write_frames;
use crate::wan::tensor::{CudaTensor, TensorError};
use crate::wan::weights::WeightMap;

use super::text::{
    flux2_dummy_text, flux2_text_len, format_flux2_prompt, pad_token_ids, Flux2TextEncoder, Qwen3Encoder,
};
use super::transformer::Flux2Transformer2D;
use super::vae::AutoencoderKlFlux2;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, PipelineError>;

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
    pub preset: String,
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
            preset: "flux2_dev".into(),
        }
    }
}

pub struct Flux2Pipeline {
    transformer: Flux2Transformer2D,
    vae: AutoencoderKlFlux2,
    text: Flux2TextEncoder,
    tiny: bool,
    kind: Flux2TextKind,
}

impl Flux2Pipeline {
    pub fn tiny() -> Self {
        Self::tiny_kind(Flux2TextKind::Mistral3)
    }

    pub fn tiny_klein() -> Self {
        Self::tiny_kind(Flux2TextKind::Qwen3)
    }

    pub fn tiny_for_preset(preset: &str) -> Self {
        Self::tiny_kind(Flux2TextKind::from_preset(preset))
    }

    fn tiny_kind(kind: Flux2TextKind) -> Self {
        let cfg = if kind == Flux2TextKind::Qwen3 {
            Flux2ArchConfig::tiny_klein()
        } else {
            Flux2ArchConfig::tiny()
        };
        let text = match kind {
            Flux2TextKind::Qwen3 => Flux2TextEncoder::qwen3(Qwen3Encoder::zeros(Qwen3Config::tiny())),
            Flux2TextKind::Mistral3 => Flux2TextEncoder::dummy(kind, cfg.joint_attention_dim),
        };
        Self {
            transformer: Flux2Transformer2D::zeros(cfg),
            vae: AutoencoderKlFlux2::zeros(Flux2VaeConfig::tiny()),
            text,
            tiny: true,
            kind,
        }
    }

    pub fn load(root: &Path, preset: &str) -> Result<Self> {
        let kind = Flux2TextKind::from_preset(preset);
        let mut cfg = Flux2ArchConfig::from_preset(preset);
        if let Ok(raw) = std::fs::read_to_string(root.join("transformer/config.json")) {
            cfg = arch_from_transformer_config(&cfg, &raw).map_err(PipelineError::Message)?;
        }
        let map = WeightMap::from_dir(&root.join("transformer")).map_err(PipelineError::from)?;
        let transformer = Flux2Transformer2D::load(cfg.clone(), &map)?;
        let vae_dir = root.join("vae");
        let vae = match WeightMap::from_dir(&vae_dir) {
            Ok(vmap) => AutoencoderKlFlux2::load(Flux2VaeConfig::flux2(), &vmap)?,
            Err(e) => {
                return Err(PipelineError::Message(format!(
                    "Flux2 VAE load from {}: {e}",
                    vae_dir.display()
                )))
            }
        };
        let text = load_text_encoder(root, kind, cfg.joint_attention_dim)?;
        Ok(Self {
            transformer,
            vae,
            text,
            tiny: false,
            kind,
        })
    }

    pub fn generate(&mut self, cfg: &GenerateConfig) -> Result<Vec<String>> {
        let (height, width, steps) = if self.tiny {
            (16usize, 16, 2usize)
        } else {
            (cfg.height.max(16), cfg.width.max(16), cfg.num_inference_steps.max(1))
        };
        let (ph, pw) = if self.tiny {
            (2, 2)
        } else {
            packed_hw(height, width, self.vae.cfg.spatial_compression_ratio)
        };
        let channels = self.transformer.cfg.in_channels;
        let seq = ph * pw;
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let noise: Vec<f32> = (0..channels * seq).map(|_| rng.sample(StandardNormal)).collect();
        let mut latents = CudaTensor::from_vec(noise, vec![1, channels, 1, ph, pw])?;
        let (ids, valid_len) = encode_prompt_ids(&self.text, cfg, self.tiny, self.transformer.cfg.joint_attention_dim);
        let encoder = self.text.encode_ids(&ids, valid_len)?;
        let mu = compute_empirical_mu(seq, steps);
        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, 1.0);
        sched.set_timesteps_flux2(steps, Some(mu));
        let guidance = if self.transformer.cfg.guidance_embeds {
            Some(cfg.embedded_cfg_scale.unwrap_or(cfg.guidance_scale))
        } else {
            None
        };
        let timesteps: Vec<f64> = sched.inference_timesteps().to_vec();
        for t in timesteps {
            let vel = self.transformer.forward(
                &latents,
                &encoder,
                t as f32 / 1000.0,
                guidance,
                ph,
                pw,
            )?;
            let x = latents.host_cow()?.to_vec();
            let v = vel.host_cow()?.to_vec();
            let next = sched
                .step_euler(&x, &v)
                .map_err(PipelineError::Message)?;
            latents = CudaTensor::from_vec(next, latents.shape.clone())?;
        }
        let decoded = if self.tiny {
            self.vae.decode(&latents)?
        } else {
            let packed = latents.host_cow()?;
            let spatial = unpatchify_2x2(&packed, self.vae.cfg.latent_channels, ph, pw)
                .map_err(PipelineError::Message)?;
            let unpacked = CudaTensor::from_vec(
                spatial,
                vec![1, self.vae.cfg.latent_channels, 1, ph * 2, pw * 2],
            )?;
            self.vae.decode(&unpacked)?
        };
        write_frames(&decoded, Path::new(&cfg.output_dir)).map_err(|e| PipelineError::Message(e.to_string()))
    }

    pub fn text_kind(&self) -> Flux2TextKind {
        self.kind
    }
}

fn load_text_encoder(root: &Path, kind: Flux2TextKind, joint_dim: usize) -> Result<Flux2TextEncoder> {
    if flux2_dummy_text() {
        return Ok(Flux2TextEncoder::dummy(kind, joint_dim));
    }
    let te = if root.join("text_encoder").is_dir() {
        root.join("text_encoder")
    } else {
        root.join("text_encoder_2")
    };
    let Ok(tmap) = WeightMap::from_dir(&te) else {
        return Ok(Flux2TextEncoder::dummy(kind, joint_dim));
    };
    let mut lm_cfg = match kind {
        Flux2TextKind::Qwen3 => Qwen3Config::klein_4b(),
        Flux2TextKind::Mistral3 => Qwen3Config::mistral3_24b(),
    };
    if let Ok(raw) = std::fs::read_to_string(te.join("config.json")) {
        lm_cfg = Qwen3Config::from_hf_json(kind, &raw).map_err(PipelineError::Message)?;
    }
    let enc = Qwen3Encoder::load(lm_cfg, &tmap)?;
    Ok(match kind {
        Flux2TextKind::Qwen3 => Flux2TextEncoder::qwen3(enc),
        Flux2TextKind::Mistral3 => Flux2TextEncoder::mistral3(enc),
    })
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
            .or_else(|_| fastvideo_models::tokenize_prompt(tok, &cfg.prompt, text_len))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_generate_writes_png() {
        let mut pipe = Flux2Pipeline::tiny();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.output_dir = std::env::temp_dir()
            .join("fastvideo-cudarc-flux2-tiny")
            .to_string_lossy()
            .into();
        let paths = pipe.generate(&cfg).unwrap();
        assert!(Path::new(&paths[0]).exists());
    }

    #[test]
    fn tiny_klein_generate_writes_png() {
        let mut pipe = Flux2Pipeline::tiny_klein();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.embedded_cfg_scale = None;
        cfg.preset = "flux2_klein_4b".into();
        cfg.output_dir = std::env::temp_dir()
            .join("fastvideo-cudarc-flux2-klein-tiny")
            .to_string_lossy()
            .into();
        let paths = pipe.generate(&cfg).unwrap();
        assert!(Path::new(&paths[0]).exists());
        assert_eq!(pipe.text_kind(), Flux2TextKind::Qwen3);
    }
}
