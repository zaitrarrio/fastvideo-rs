//! GLM-Image T2I generate scaffold.

use std::path::{Path, PathBuf};

use fastvideo_models::glm_image::{GlmImageConfig, GlmImagePreset};
use fastvideo_models::vae::AutoencoderKlConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::transformer::GlmImageTransformer;
use crate::vae::AutoencoderKl;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct GlmImageRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub preset: GlmImagePreset,
}

impl GlmImageRequest {
    pub fn base(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = GlmImagePreset::Base;
        Self {
            prompt: prompt.into(),
            negative_prompt: String::new(),
            seed,
            height: preset.default_height(),
            width: preset.default_width(),
            num_steps: preset.default_steps(),
            guidance_scale: preset.guidance_scale(),
            preset,
        }
    }
}

pub struct GlmImagePipeline {
    pub root: PathBuf,
    pub cfg: GlmImageConfig,
    pub preset: GlmImagePreset,
    pub dit: Option<GlmImageTransformer>,
    pub vae: Option<AutoencoderKl>,
}

impl GlmImagePipeline {
    pub fn open(root: impl Into<PathBuf>, preset: GlmImagePreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: GlmImageConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(GlmImageTransformer::load(self.cfg.dit.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = GlmImageConfig::tiny();
        self.dit = Some(GlmImageTransformer::zeros(self.cfg.dit.clone())?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let cfg = if self.cfg.dit.num_layers <= 2 {
            AutoencoderKlConfig::tiny(self.cfg.dit.out_channels)
        } else {
            AutoencoderKlConfig::sd3()
        };
        let vae_dir = self.root.join("vae");
        if vae_dir.is_dir() {
            let map = WeightMap::open(&vae_dir).map_err(|e| msg(e.to_string()))?;
            self.vae = Some(AutoencoderKl::load(cfg, &map)?);
        } else {
            self.vae = Some(AutoencoderKl::zeros(cfg));
        }
        Ok(())
    }

    pub fn load_vae_stub(&mut self) {
        let cfg = if self.cfg.dit.num_layers <= 2 {
            AutoencoderKlConfig::tiny(self.cfg.dit.out_channels)
        } else {
            AutoencoderKlConfig::sd3()
        };
        self.vae = Some(AutoencoderKl::zeros(cfg));
    }

    fn encode_text(&self, prompt: &str) -> Result<CudaTensor> {
        let allow_zeros = self.cfg.dit.num_layers <= 2;
        let dim = self.cfg.dit.text_embed_dim;
        let te = self.root.join("text_encoder");
        crate::text_encode::zeros_or_encode(allow_zeros, &[1, 16, dim], &te, || {
            // Prefer CLIP sequence + channel broadcast; GLM AR tower keys are
            // distinct — if CLIP keys are absent, fail clearly (no zeros).
            let map = WeightMap::open(&te).map_err(|e| msg(e.to_string()))?;
            if map.contains("text_model.embeddings.token_embedding.weight")
                || map.contains("embeddings.token_embedding.weight")
            {
                let emb = crate::text_encode::encode_clip_l_hidden(
                    &self.root,
                    "text_encoder",
                    "tokenizer",
                    prompt,
                )?;
                return crate::text_encode::broadcast_to_dim(&emb, dim);
            }
            Err(msg(format!(
                "glm_image: {} present but GLM AR / CLIP text keys not recognized \
                 (need text_model.embeddings.token_embedding.weight or GLM AR pack)",
                te.display()
            )))
        })
    }

    pub fn generate(&self, request: &GlmImageRequest, out_path: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| msg("GLM-Image: call load_dit()"))?;
        let vae = self.vae.as_ref().ok_or_else(|| msg("GLM-Image: call load_vae() or load_vae_stub()"))?;
        let text = self.encode_text(&request.prompt)?;
        let (lh, lw) = if self.cfg.dit.num_layers <= 2 {
            (8usize, 8usize)
        } else {
            self.cfg.dit.latent_spatial(request.height, request.width)
        };
        let c = self.cfg.dit.out_channels;
        let n = c * lh * lw;
        let mut sched = self.cfg.schedule(request.num_steps);
        let timesteps = sched.inference_timesteps().to_vec();
        let sigmas = sched.inference_sigmas().to_vec();
        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let mut sample: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(&mut rng);
                z * (sigmas[0] as f32)
            })
            .collect();
        for &t in &timesteps {
            let lat = CudaTensor::from_vec(sample.clone(), vec![1, c, lh, lw])?;
            let pred = dit.forward(&lat, &text, t as f32)?;
            let vel = pred.host_cow()?;
            sample = sched.step_euler(&sample, &vel[..n]).map_err(msg)?;
            let _ = (request.guidance_scale, &request.negative_prompt);
        }
        let latents = CudaTensor::from_vec(sample, vec![1, c, lh, lw])?;
        let scaled = vae.scale_latents(&latents)?;
        let pixels = vae.decode(&scaled)?;
        let [_, _, hf, wf] = match pixels.shape[..] {
            [1, 3, hf, wf] => [1, 3, hf, wf],
            _ => return Err(msg(format!("glm_image decode {:?}", pixels.shape))),
        };
        let data = pixels.host_cow()?;
        let mut rgb = vec![0u8; 3 * hf * wf];
        for i in 0..hf * wf {
            for ch in 0..3 {
                rgb[i * 3 + ch] = (data[ch * hf * wf + i] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| msg(e.to_string()))?;
        }
        image::save_buffer(out_path, &rgb, wf as u32, hf as u32, image::ColorType::Rgb8)
            .map_err(|e| msg(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_generate_png() {
        let mut pipe = GlmImagePipeline::open("/tmp/glm-missing", GlmImagePreset::Base).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        pipe.load_vae_stub();
        let mut r = GlmImageRequest::base("test", 1);
        r.height = 64;
        r.width = 64;
        r.num_steps = 2;
        let dir = std::env::temp_dir().join("glm-image-tiny-test.png");
        pipe.generate(&r, &dir).unwrap();
        assert!(dir.is_file());
        let _ = std::fs::remove_file(&dir);
    }
}
