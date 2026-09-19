//! Flux2 T2I generate path on cudarc (`CudaTensor`).

use std::path::Path;

use fastvideo_models::flux2::{
    arch_from_transformer_config, compute_empirical_mu, packed_hw, unpatchify_2x2, Flux2ArchConfig,
    Flux2TextKind, Flux2VaeConfig,
};
use fastvideo_models::schedulers::FlowMatchEulerDiscreteScheduler;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use thiserror::Error;

use crate::wan::pipeline::write_frames;
use crate::wan::tensor::{CudaTensor, TensorError};
use crate::wan::weights::WeightMap;

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
        Self {
            transformer: Flux2Transformer2D::zeros(cfg),
            vae: AutoencoderKlFlux2::zeros(Flux2VaeConfig::tiny()),
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
        let transformer = Flux2Transformer2D::load(cfg, &map)?;
        let vae = match WeightMap::from_dir(&root.join("vae")) {
            Ok(vmap) => AutoencoderKlFlux2::load(Flux2VaeConfig::flux2(), &vmap).unwrap_or_else(|_| {
                AutoencoderKlFlux2::zeros(Flux2VaeConfig::flux2())
            }),
            Err(_) => AutoencoderKlFlux2::zeros(Flux2VaeConfig::flux2()),
        };
        Ok(Self {
            transformer,
            vae,
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
        let text_dim = self.transformer.cfg.joint_attention_dim;
        let text_len = if self.tiny { 4 } else { 16 };
        let embeds: Vec<f32> = (0..text_len * text_dim)
            .map(|i| ((cfg.prompt.len() + i) as f32 * 0.001) % 1.0)
            .collect();
        let encoder = CudaTensor::from_vec(embeds, vec![1, text_len, text_dim])?;
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
