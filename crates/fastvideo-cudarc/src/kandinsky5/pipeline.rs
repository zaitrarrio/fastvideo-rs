//! Kandinsky 5 generate: Qwen + CLIP-L → DiT Euler → HunyuanVideo-16 VAE.

use std::path::{Path, PathBuf};

use fastvideo_models::kandinsky5::{
    Kandinsky5Preset, Kandinsky5Schedule, Kandinsky5TransformerConfig,
};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

use super::text::{self, Kandinsky5TextConditioning};
use super::transformer::Kandinsky5Transformer;
use super::vae::{HunyuanVideo16Vae, HunyuanVideo16VaeConfig};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct Kandinsky5Request {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub preset: Kandinsky5Preset,
}

impl Kandinsky5Request {
    pub fn lite_5s(prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            seed,
            height: 512,
            width: 768,
            num_frames: 121,
            num_steps: 50,
            preset: Kandinsky5Preset::LiteT2v5s,
        }
    }
}

pub struct Kandinsky5Pipeline {
    pub root: PathBuf,
    pub dit_cfg: Kandinsky5TransformerConfig,
    pub preset: Kandinsky5Preset,
    pub dit: Option<Kandinsky5Transformer>,
    pub vae: Option<HunyuanVideo16Vae>,
}

impl Kandinsky5Pipeline {
    pub fn open(root: impl Into<PathBuf>, preset: Kandinsky5Preset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            dit_cfg: Kandinsky5TransformerConfig::lite(),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map =
            WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(Kandinsky5Transformer::load(self.dit_cfg.clone(), &map)?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        self.vae = Some(
            HunyuanVideo16Vae::load(&map, HunyuanVideo16VaeConfig::default_hunyuan())
                .map_err(|e| msg(e.to_string()))?,
        );
        Ok(())
    }

    pub fn schedule(&self, steps: usize) -> Kandinsky5Schedule {
        Kandinsky5Schedule::new(steps, self.preset)
    }

    fn encode_text(&self, prompt: &str) -> Result<Kandinsky5TextConditioning> {
        text::encode_prompt(&self.root, prompt, &self.dit_cfg).map_err(|e| msg(e.to_string()))
    }

    pub fn generate(&self, request: &Kandinsky5Request, out_dir: &Path) -> Result<()> {
        let _ = out_dir;
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("Kandinsky 5: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;

        let text_ok = self.root.join("text_encoder").is_dir()
            && self.root.join("tokenizer").join("tokenizer.json").is_file();
        let cond = if text_ok {
            self.encode_text(&request.prompt)?
        } else {
            return Err(msg(
                "Kandinsky 5: need `tokenizer/tokenizer.json` + `text_encoder/` under --weights",
            ));
        };

        // Classic HunyuanVideo VAE: spatial 8×, temporal 4×, 16 latent ch.
        let (spat, temp) = (8usize, 4usize);
        let lt = 1 + (request.num_frames.saturating_sub(1)).div_ceil(temp);
        let lh = request.height.div_ceil(spat);
        let lw = request.width.div_ceil(spat);
        let c_noise = self.dit_cfg.in_visual_dim;
        let c_in = self.dit_cfg.visual_embed_in_dim();

        let mut sched = self.schedule(request.num_steps);
        let sigmas = sched.sigmas().to_vec();
        let timesteps = sched.timesteps().to_vec();

        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        // Channel-last sample [B,T,H,W,C_noise]
        let n = lt * lh * lw * c_noise;
        let mut sample: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(&mut rng);
                z * (sigmas[0] as f32)
            })
            .collect();

        for &t in &timesteps {
            let lat = pack_visual(&sample, lt, lh, lw, c_noise, c_in)?;
            let velocity = dit.forward(&lat, &cond.qwen, &cond.clip_pooled, t as f32)?;
            let vel = velocity.host_cow()?;
            // Only the noise channels contribute to the Euler update.
            let vel_noise = take_noise_channels(&vel, lt, lh, lw, c_noise, c_in)?;
            sample = sched.inner.step_euler(&sample, &vel_noise).map_err(msg)?;
        }

        // Channel-last → BCTHW for VAE
        let mut cthw = vec![0f32; c_noise * lt * lh * lw];
        for ti in 0..lt {
            for yi in 0..lh {
                for xi in 0..lw {
                    for c in 0..c_noise {
                        let src = ((ti * lh + yi) * lw + xi) * c_noise + c;
                        let dst = c * (lt * lh * lw) + ti * (lh * lw) + yi * lw + xi;
                        cthw[dst] = sample[src];
                    }
                }
            }
        }
        let latents = CudaTensor::from_vec(cthw, vec![1, c_noise, lt, lh, lw])?;
        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("Kandinsky 5: call load_vae() after placing Diffusers `vae/` under --weights")
        })?;
        let scaled = latents.try_mul_scalar(1.0 / vae.scaling_factor())?;
        let pixels = vae.decode(&scaled).map_err(|e| msg(e.to_string()))?;
        let [_, _, tf, hf, wf] = match pixels.shape[..] {
            [1, 3, tf, hf, wf] => [1, 3, tf, hf, wf],
            _ => {
                return Err(msg(format!(
                    "k5 decode shape {:?} want [1,3,T,H,W]",
                    pixels.shape
                )))
            }
        };
        let by_frame = pixels.reshape(vec![tf, 3, hf, wf])?;
        let rgb = crate::wan::pipeline::frames_to_rgb8(&by_frame)?;
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

fn pack_visual(
    sample: &[f32],
    t: usize,
    h: usize,
    w: usize,
    c_noise: usize,
    c_in: usize,
) -> Result<CudaTensor> {
    let want = t * h * w * c_noise;
    if sample.len() != want {
        return Err(msg(format!("k5 pack: {} vs {want}", sample.len())));
    }
    if c_in == c_noise {
        return Ok(CudaTensor::from_vec(
            sample.to_vec(),
            vec![1, t, h, w, c_noise],
        )?);
    }
    // visual_cond: [noise | zeros_cond | zeros_mask]
    let mut packed = vec![0f32; t * h * w * c_in];
    for i in 0..t * h * w {
        for c in 0..c_noise {
            packed[i * c_in + c] = sample[i * c_noise + c];
        }
    }
    Ok(CudaTensor::from_vec(packed, vec![1, t, h, w, c_in])?)
}

fn take_noise_channels(
    vel: &[f32],
    t: usize,
    h: usize,
    w: usize,
    c_noise: usize,
    c_out: usize,
) -> Result<Vec<f32>> {
    // DiT out is out_visual_dim (= c_noise for Lite), shape [1,T,H,W,C]
    if vel.len() != t * h * w * c_out {
        return Err(msg(format!(
            "k5 vel: {} vs {} (T={t} H={h} W={w} C={c_out})",
            vel.len(),
            t * h * w * c_out
        )));
    }
    if c_out == c_noise {
        return Ok(vel.to_vec());
    }
    let mut out = vec![0f32; t * h * w * c_noise];
    for i in 0..t * h * w {
        for c in 0..c_noise {
            out[i * c_noise + c] = vel[i * c_out + c];
        }
    }
    Ok(out)
}
