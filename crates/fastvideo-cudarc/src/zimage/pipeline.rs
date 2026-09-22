//! Z-Image T2I generate scaffold: noise → DiT → AutoencoderKL → PNG.

use std::path::{Path, PathBuf};

use fastvideo_models::zimage::{ZImageConfig, ZImagePreset, ZImageTransformerConfig};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::transformer::ZImageTransformer;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct ZImageRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub preset: ZImagePreset,
}

impl ZImageRequest {
    pub fn turbo(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = ZImagePreset::Turbo;
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

/// Minimal SD-style VAE decode stub (identity upsample) until AutoencoderKL lands.
pub struct ZImageVaeStub {
    pub latent_channels: usize,
}

impl ZImageVaeStub {
    pub fn for_cfg(cfg: &ZImageTransformerConfig) -> Self {
        Self {
            latent_channels: cfg.out_channels,
        }
    }

    /// Map latents `[1,C,H,W]` → RGB `[1,3,H*8,W*8]` via nearest upsample + channel fold.
    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let [_, c, h, w] = match latents.shape[..] {
            [1, c, h, w] => [1, c, h, w],
            _ => {
                return Err(msg(format!(
                    "zimage vae want [1,C,H,W], got {:?}",
                    latents.shape
                )))
            }
        };
        let data = latents.host_cow()?;
        let oh = h * 8;
        let ow = w * 8;
        let mut rgb = vec![0f32; 3 * oh * ow];
        for y in 0..oh {
            for x in 0..ow {
                let ly = y / 8;
                let lx = x / 8;
                let mut acc = [0f32; 3];
                for ch in 0..c.min(self.latent_channels) {
                    let v = data[(ch * h + ly) * w + lx];
                    acc[ch % 3] += v;
                }
                for ch in 0..3 {
                    // tanh-ish to [0,1] then caller converts to u8
                    let v = (acc[ch].tanh() * 0.5 + 0.5).clamp(0.0, 1.0);
                    rgb[ch * oh * ow + y * ow + x] = v;
                }
            }
        }
        CudaTensor::from_vec(rgb, vec![1, 3, oh, ow]).map_err(Into::into)
    }
}

pub struct ZImagePipeline {
    pub root: PathBuf,
    pub cfg: ZImageConfig,
    pub preset: ZImagePreset,
    pub dit: Option<ZImageTransformer>,
    pub vae: Option<ZImageVaeStub>,
}

impl ZImagePipeline {
    pub fn open(root: impl Into<PathBuf>, preset: ZImagePreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: ZImageConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(ZImageTransformer::load(self.cfg.dit.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = ZImageConfig::tiny();
        self.dit = Some(ZImageTransformer::zeros(self.cfg.dit.clone())?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        // Full AutoencoderKL load hook: look for `vae/`; until then use stub.
        let vae_dir = self.root.join("vae");
        if vae_dir.is_dir() {
            let _ = WeightMap::open(&vae_dir).map_err(|e| msg(e.to_string()))?;
        }
        self.vae = Some(ZImageVaeStub::for_cfg(&self.cfg.dit));
        Ok(())
    }

    pub fn load_vae_stub(&mut self) {
        self.vae = Some(ZImageVaeStub::for_cfg(&self.cfg.dit));
    }

    fn encode_text(&self, _prompt: &str) -> Result<CudaTensor> {
        // Qwen3 encode hook: zeros until `text_encoder/` + tokenizer present.
        Ok(CudaTensor::zeros(&[1, 16, self.cfg.dit.cap_feat_dim]))
    }

    pub fn generate(&self, request: &ZImageRequest, out_path: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("Z-Image: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("Z-Image: call load_vae() or load_vae_stub()")
        })?;

        let text = self.encode_text(&request.prompt)?;
        let (lh, lw) = self.cfg.dit.latent_spatial(request.height, request.width);
        // For tiny tests, allow non-1024 canvases that still align to patch.
        let (lh, lw) = if self.cfg.dit.dim < 1000 {
            (8usize, 8usize)
        } else {
            (lh, lw)
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
            let _ = request.guidance_scale;
            let _ = &request.negative_prompt;
        }

        let latents = CudaTensor::from_vec(sample, vec![1, c, lh, lw])?;
        let pixels = vae.decode(&latents)?;
        let [_, _, hf, wf] = match pixels.shape[..] {
            [1, 3, hf, wf] => [1, 3, hf, wf],
            _ => {
                return Err(msg(format!(
                    "zimage decode shape {:?} want [1,3,H,W]",
                    pixels.shape
                )))
            }
        };
        let data = pixels.host_cow()?;
        let mut rgb = vec![0u8; 3 * hf * wf];
        for i in 0..hf * wf {
            for ch in 0..3 {
                let v = data[ch * hf * wf + i];
                rgb[i * 3 + ch] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| msg(e.to_string()))?;
        }
        image::save_buffer(
            out_path,
            &rgb,
            wf as u32,
            hf as u32,
            image::ColorType::Rgb8,
        )
        .map_err(|e| msg(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = ZImageRequest::turbo("a garden", 0);
        assert_eq!(r.height, 1024);
        assert_eq!(r.num_steps, 8);
        assert_eq!(r.guidance_scale, 0.0);
    }

    #[test]
    fn tiny_generate_png() {
        let mut pipe = ZImagePipeline::open("/tmp/zimage-missing", ZImagePreset::Turbo).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        pipe.load_vae_stub();
        let mut r = ZImageRequest::turbo("test", 1);
        r.height = 64;
        r.width = 64;
        r.num_steps = 2;
        let dir = std::env::temp_dir().join("zimage-tiny-test.png");
        pipe.generate(&r, &dir).unwrap();
        assert!(dir.is_file());
        let _ = std::fs::remove_file(&dir);
    }
}
