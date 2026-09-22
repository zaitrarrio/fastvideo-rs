//! LingBot generate: Qwen3-VL → DiT Euler → Wan VAE → PNG.

use std::path::{Path, PathBuf};

use fastvideo_models::lingbot::{
    LingBotPreset, LingBotSchedule, LingBotTransformerConfig, PROMPT_CROP_START,
};
use fastvideo_models::wan::WanVaeConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::vae::AutoencoderKlWan;
use crate::wan::weights::WeightMap;

use super::text;
use super::transformer::LingBotTransformer;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct LingBotRequest {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub preset: LingBotPreset,
}

impl LingBotRequest {
    pub fn dense_1_3b(prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            seed,
            height: 480,
            width: 832,
            num_frames: 81,
            num_steps: 40,
            preset: LingBotPreset::Dense13b,
        }
    }
}

pub struct LingBotPipeline {
    pub root: PathBuf,
    pub dit_cfg: LingBotTransformerConfig,
    pub preset: LingBotPreset,
    pub dit: Option<LingBotTransformer>,
    pub vae: Option<AutoencoderKlWan>,
}

impl LingBotPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: LingBotPreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            dit_cfg: LingBotTransformerConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(LingBotTransformer::load(self.dit_cfg.clone(), &map)?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        let mut cfg = WanVaeConfig::wan_2_1();
        cfg.load_encoder = false;
        self.vae = Some(AutoencoderKlWan::load(cfg, &map).map_err(|e| msg(e.to_string()))?);
        Ok(())
    }

    /// Qwen3-VL encode when `text_encoder/` + tokenizer exist; else zero embeds.
    fn encode_text(&self, prompt: &str) -> Result<CudaTensor> {
        let te = self.root.join("text_encoder");
        let tok = self.root.join("tokenizer").join("tokenizer.json");
        if te.is_dir() && tok.is_file() {
            return text::encode_prompt(&self.root, prompt, self.dit_cfg.text_dim)
                .map_err(|e| msg(e.to_string()));
        }
        if te.is_dir() {
            return Err(msg(format!(
                "lingbot text_encoder present but tokenizer/tokenizer.json missing \
                 (crop={PROMPT_CROP_START})"
            )));
        }
        Ok(CudaTensor::zeros(&[1, 16, self.dit_cfg.text_dim]))
    }

    pub fn generate(&self, request: &LingBotRequest, out_dir: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("LingBot: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        let text = self.encode_text(&request.prompt)?;

        let (spat, temp) = (8usize, 4usize);
        let lt = 1 + (request.num_frames.saturating_sub(1)).div_ceil(temp);
        let lh = request.height.div_ceil(spat);
        let lw = request.width.div_ceil(spat);
        let c = self.dit_cfg.in_channels;
        let spatial = lt * lh * lw;

        let mut sched = LingBotSchedule::new(request.num_steps, request.preset);
        let sigmas = sched.sigmas().to_vec();
        let timesteps = sched.timesteps().to_vec();

        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let n = c * spatial;
        let mut sample: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(&mut rng);
                z * (sigmas[0] as f32)
            })
            .collect();

        for &t in &timesteps {
            let lat = CudaTensor::from_vec(sample.clone(), vec![1, c, lt, lh, lw])?;
            let velocity = dit.forward(&lat, &text, t as f32)?;
            let vel = velocity.host_cow()?;
            sample = sched.inner.step_euler(&sample, &vel[..n]).map_err(msg)?;
        }

        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("LingBot: call load_vae() after placing Diffusers `vae/` under --weights")
        })?;
        let latents = CudaTensor::from_vec(sample, vec![1, c, lt, lh, lw])?;
        let scaled = vae.scale_latents(&latents).map_err(|e| msg(e.to_string()))?;
        let pixels = vae.decode(&scaled).map_err(|e| msg(e.to_string()))?;
        let [_, _, tf, hf, wf] = match pixels.shape[..] {
            [1, 3, tf, hf, wf] => [1, 3, tf, hf, wf],
            _ => {
                return Err(msg(format!(
                    "lingbot decode shape {:?} want [1,3,T,H,W]",
                    pixels.shape
                )))
            }
        };
        let by_frame = pixels.reshape(vec![tf, 3, hf, wf])?;
        let rgb = frames_to_rgb8(&by_frame)?;
        std::fs::create_dir_all(out_dir).map_err(|e| msg(e.to_string()))?;
        for i in 0..tf {
            let path = out_dir.join(format!("frame_{i:05}.png"));
            let off = i * 3 * hf * wf;
            image::save_buffer(
                &path,
                &rgb[off..off + 3 * hf * wf],
                wf as u32,
                hf as u32,
                image::ColorType::Rgb8,
            )
            .map_err(|e| msg(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = LingBotRequest::dense_1_3b("hi", 0);
        assert_eq!(r.height, 480);
        assert_eq!(PROMPT_CROP_START, 140);
    }
}
