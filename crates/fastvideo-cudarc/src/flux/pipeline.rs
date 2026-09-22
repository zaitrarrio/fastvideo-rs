//! FLUX.1 T2I generate scaffold.

use std::path::{Path, PathBuf};

use fastvideo_models::flux::{FluxConfig, FluxPreset};
use fastvideo_models::vae::AutoencoderKlConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::transformer::FluxTransformer;
use crate::vae::AutoencoderKl;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct FluxRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub preset: FluxPreset,
}

impl FluxRequest {
    pub fn dev(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = FluxPreset::Dev;
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

pub struct FluxPipeline {
    pub root: PathBuf,
    pub cfg: FluxConfig,
    pub preset: FluxPreset,
    pub dit: Option<FluxTransformer>,
    pub vae: Option<AutoencoderKl>,
}

impl FluxPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: FluxPreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: FluxConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(FluxTransformer::load(self.cfg.dit.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = FluxConfig::tiny();
        self.dit = Some(FluxTransformer::zeros(self.cfg.dit.clone())?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let cfg = if self.cfg.dit.num_layers <= 2 {
            AutoencoderKlConfig::tiny(16)
        } else {
            AutoencoderKlConfig::flux()
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
            AutoencoderKlConfig::tiny(16)
        } else {
            AutoencoderKlConfig::flux()
        };
        self.vae = Some(AutoencoderKl::zeros(cfg));
    }

    fn encode_text(&self, _prompt: &str) -> Result<CudaTensor> {
        Ok(CudaTensor::zeros(&[1, 16, self.cfg.dit.joint_attention_dim]))
    }

    pub fn generate(&self, request: &FluxRequest, out_path: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| msg("FLUX.1: call load_dit()"))?;
        let vae = self.vae.as_ref().ok_or_else(|| msg("FLUX.1: call load_vae() or load_vae_stub()"))?;
        let text = self.encode_text(&request.prompt)?;
        // Tiny path: operate directly in packed/DiT channel space at 8×8.
        let (ph, pw, c) = if self.cfg.dit.num_layers <= 2 {
            (8usize, 8usize, self.cfg.dit.in_channels)
        } else {
            let (ph, pw) = self.cfg.dit.packed_spatial(request.height, request.width);
            (ph, pw, self.cfg.dit.in_channels)
        };
        let n = c * ph * pw;
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
            let lat = CudaTensor::from_vec(sample.clone(), vec![1, c, ph, pw])?;
            let pred = dit.forward(&lat, &text, t as f32, request.guidance_scale)?;
            let vel = pred.host_cow()?;
            sample = sched.step_euler(&sample, &vel[..n]).map_err(msg)?;
        }
        // Unpack to VAE channels for decode: fold packed C→16 by averaging groups.
        let vae_c = vae.cfg.latent_channels;
        let (vh, vw) = if self.cfg.dit.num_layers <= 2 {
            (ph, pw)
        } else {
            (ph * 2, pw * 2)
        };
        let mut unpacked = vec![0f32; vae_c * vh * vw];
        let pack = (c / vae_c).max(1);
        for y in 0..vh.min(if self.cfg.dit.num_layers <= 2 { ph } else { ph * 2 }) {
            for x in 0..vw.min(if self.cfg.dit.num_layers <= 2 { pw } else { pw * 2 }) {
                let (py, px) = if self.cfg.dit.num_layers <= 2 {
                    (y, x)
                } else {
                    (y / 2, x / 2)
                };
                for ch in 0..vae_c {
                    let mut acc = 0f32;
                    for k in 0..pack {
                        let src_c = ch * pack + k;
                        if src_c < c && py < ph && px < pw {
                            acc += sample[(src_c * ph + py) * pw + px];
                        }
                    }
                    unpacked[(ch * vh + y) * vw + x] = acc / pack as f32;
                }
            }
        }
        let latents = CudaTensor::from_vec(unpacked, vec![1, vae_c, vh, vw])?;
        let scaled = vae.scale_latents(&latents)?;
        let pixels = vae.decode(&scaled)?;
        let [_, _, hf, wf] = match pixels.shape[..] {
            [1, 3, hf, wf] => [1, 3, hf, wf],
            _ => return Err(msg(format!("flux decode {:?}", pixels.shape))),
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
        let _ = &request.negative_prompt;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_generate_png() {
        let mut pipe = FluxPipeline::open("/tmp/flux-missing", FluxPreset::Dev).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        pipe.load_vae_stub();
        let mut r = FluxRequest::dev("test", 1);
        r.height = 64;
        r.width = 64;
        r.num_steps = 2;
        let dir = std::env::temp_dir().join("flux-tiny-test.png");
        pipe.generate(&r, &dir).unwrap();
        assert!(dir.is_file());
        let _ = std::fs::remove_file(&dir);
    }
}
