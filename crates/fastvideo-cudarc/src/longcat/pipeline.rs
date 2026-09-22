//! LongCat generate: UMT5 → DiT Euler → Wan VAE → PNG.

use std::path::{Path, PathBuf};

use fastvideo_models::longcat::{LongCatPreset, LongCatSchedule, LongCatTransformerConfig};
use fastvideo_models::wan::{Umt5Config, WanVaeConfig};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::umt5::Umt5Encoder;
use crate::wan::vae::AutoencoderKlWan;
use crate::wan::weights::WeightMap;

use super::transformer::LongCatTransformer;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct LongCatRequest {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub preset: LongCatPreset,
    pub enable_bsa: bool,
}

impl LongCatRequest {
    pub fn t2v_480p(prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            seed,
            height: 480,
            width: 832,
            num_frames: 93,
            num_steps: 50,
            preset: LongCatPreset::T2v480p,
            enable_bsa: false,
        }
    }
}

pub struct LongCatPipeline {
    pub root: PathBuf,
    pub dit_cfg: LongCatTransformerConfig,
    pub preset: LongCatPreset,
    pub dit: Option<LongCatTransformer>,
    pub vae: Option<AutoencoderKlWan>,
}

impl LongCatPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: LongCatPreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            dit_cfg: LongCatTransformerConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(LongCatTransformer::load(self.dit_cfg.clone(), &map)?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        self.vae = Some(
            AutoencoderKlWan::load(WanVaeConfig::wan_2_1(), &map).map_err(|e| msg(e.to_string()))?,
        );
        Ok(())
    }

    fn encode_text(&self, prompt: &str) -> Result<CudaTensor> {
        let te = self.root.join("text_encoder");
        let tok = self.root.join("tokenizer").join("tokenizer.json");
        if te.is_dir() && tok.is_file() {
            let (ids, len) = fastvideo_models::tokenize_prompt(
                tok.to_str().unwrap_or(""),
                prompt,
                512,
            )
            .map_err(msg)?;
            let mut padded = ids;
            padded.resize(512, 0);
            let map = WeightMap::open(&te).map_err(|e| msg(e.to_string()))?;
            let enc = Umt5Encoder::load(Umt5Config::xxl(), &map).map_err(|e| msg(e.to_string()))?;
            let embeds = enc
                .forward(&padded, 1, 512)
                .map_err(|e| msg(e.to_string()))?;
            // Zero pad like FastVideo umt5_postprocess_text.
            let mut host = embeds.host_cow()?.to_vec();
            let d = Umt5Config::xxl().d_model;
            for i in len..512 {
                for c in 0..d {
                    host[i * d + c] = 0.0;
                }
            }
            return Ok(CudaTensor::from_vec(host, vec![1, 512, d])?);
        }
        Ok(CudaTensor::zeros(&[1, 16, self.dit_cfg.caption_channels]))
    }

    pub fn generate(&self, request: &LongCatRequest, out_dir: &Path) -> Result<()> {
        let enable_bsa = request.enable_bsa || self.preset.enable_bsa();
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("LongCat: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        let text = self.encode_text(&request.prompt)?;

        let (spat, temp) = (8usize, 4usize);
        let lt = 1 + (request.num_frames.saturating_sub(1)).div_ceil(temp);
        let lh = request.height.div_ceil(spat);
        let lw = request.width.div_ceil(spat);
        let c = self.dit_cfg.in_channels;
        let spatial = lt * lh * lw;

        let mut sched = LongCatSchedule::new(request.num_steps, request.preset);
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

        for (i, &t) in timesteps.iter().enumerate() {
            let lat = CudaTensor::from_vec(sample.clone(), vec![1, c, lt, lh, lw])?;
            let velocity = dit.forward_with_bsa(&lat, &text, t as f32, enable_bsa)?;
            let vel = velocity.host_cow()?;
            sample = sched
                .inner
                .step_euler(&sample, &vel[..n])
                .map_err(msg)?;
            let _ = i;
        }

        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("LongCat: call load_vae() after placing Diffusers `vae/` under --weights")
        })?;
        let latents = CudaTensor::from_vec(sample, vec![1, c, lt, lh, lw])?;
        let scaled = vae.scale_latents(&latents).map_err(|e| msg(e.to_string()))?;
        let pixels = vae.decode(&scaled).map_err(|e| msg(e.to_string()))?;
        let [_, _, tf, hf, wf] = match pixels.shape[..] {
            [1, 3, tf, hf, wf] => [1, 3, tf, hf, wf],
            _ => {
                return Err(msg(format!(
                    "longcat decode shape {:?} want [1,3,T,H,W]",
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
    fn request_480p() {
        let r = LongCatRequest::t2v_480p("test", 1);
        assert_eq!(r.height, 480);
        assert_eq!(r.width, 832);
    }
}
