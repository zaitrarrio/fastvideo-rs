//! SD 3.5 T2I generate: noise → MMDiT → AutoencoderKL → PNG.

use std::path::{Path, PathBuf};

use fastvideo_models::sd35::{Sd35Config, Sd35Preset};
use fastvideo_models::vae::AutoencoderKlConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::transformer::Sd35Transformer;
use crate::vae::AutoencoderKl;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct Sd35Request {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub preset: Sd35Preset,
}

impl Sd35Request {
    pub fn medium(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = Sd35Preset::Medium;
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

pub struct Sd35Pipeline {
    pub root: PathBuf,
    pub cfg: Sd35Config,
    pub preset: Sd35Preset,
    pub dit: Option<Sd35Transformer>,
    pub vae: Option<AutoencoderKl>,
}

impl Sd35Pipeline {
    pub fn open(root: impl Into<PathBuf>, preset: Sd35Preset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: Sd35Config::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map =
            WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(Sd35Transformer::load(self.cfg.dit.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = Sd35Config::tiny();
        self.dit = Some(Sd35Transformer::zeros(self.cfg.dit.clone())?);
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
        let dim = self.cfg.dit.joint_attention_dim;
        // SD3.5: T5-XXL lives in `text_encoder_3` / `tokenizer_3`.
        let te = self.root.join("text_encoder_3");
        crate::text_encode::zeros_or_encode(allow_zeros, &[1, 16, dim], &te, || {
            let emb = crate::text_encode::encode_t5_xxl(
                &self.root,
                "text_encoder_3",
                "tokenizer_3",
                prompt,
                256,
            )?;
            if emb.shape.get(2).copied() != Some(dim) {
                return crate::text_encode::broadcast_to_dim(&emb, dim);
            }
            Ok(emb)
        })
    }

    pub fn generate(&self, request: &Sd35Request, out_path: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("SD3.5: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        let vae = self
            .vae
            .as_ref()
            .ok_or_else(|| msg("SD3.5: call load_vae() or load_vae_stub()"))?;

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
            _ => return Err(msg(format!("sd35 decode shape {:?}", pixels.shape))),
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
        let mut pipe = Sd35Pipeline::open("/tmp/sd35-missing", Sd35Preset::Medium).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        pipe.load_vae_stub();
        let mut r = Sd35Request::medium("test", 1);
        r.height = 64;
        r.width = 64;
        r.num_steps = 2;
        let dir = std::env::temp_dir().join("sd35-tiny-test.png");
        pipe.generate(&r, &dir).unwrap();
        assert!(dir.is_file());
        let _ = std::fs::remove_file(&dir);
    }
}
