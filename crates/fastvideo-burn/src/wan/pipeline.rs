//! Wan inference pipeline on Burn ndarray: UMT5 → DiT sampling → VAE decode → PNG frames.

use std::path::Path;

use burn::prelude::*;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use fastvideo_models::schedulers::{DmdSchedule, FlowUniPCMultistepScheduler, FAST_WAN_1_3B_DMD_STEPS};
use fastvideo_models::{Umt5Config, WanVaeConfig, WanVideoArchConfig};

use super::nn::{default_device, to_vec_f32, B, Device};
use super::transformer::WanTransformer3D;
use super::umt5::{pad_prompt_embeds, Umt5Encoder};
use super::vae::AutoencoderKlWan;
use super::weights::WeightMap;
use crate::error::{BurnError, Result};

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub prompt: String,
    pub negative_prompt: String,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    pub output_dir: String,
    pub tiny: bool,
    pub is_dmd: bool,
    pub flow_shift: f64,
    pub dmd_steps: Option<Vec<i32>>,
    pub tokenizer_path: Option<String>,
    pub image_path: Option<String>,
    pub guidance_scale_2: Option<f32>,
    pub boundary_ratio: Option<f32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            prompt: "a cat walking".into(),
            negative_prompt: String::new(),
            height: 480,
            width: 832,
            num_frames: 81,
            num_inference_steps: 50,
            guidance_scale: 5.0,
            seed: 42,
            output_dir: "out".into(),
            tiny: false,
            is_dmd: false,
            flow_shift: 5.0,
            dmd_steps: None,
            tokenizer_path: None,
            image_path: None,
            guidance_scale_2: None,
            boundary_ratio: None,
        }
    }
}

pub struct WanPipeline {
    text: Umt5Encoder,
    transformer: WanTransformer3D,
    vae: AutoencoderKlWan,
    device: Device,
    tiny: bool,
}

impl WanPipeline {
    pub fn tiny() -> Result<Self> {
        Self::tiny_on(&default_device())
    }

    pub fn tiny_on(device: &Device) -> Result<Self> {
        Ok(Self {
            text: Umt5Encoder::zeros(Umt5Config::tiny(), device),
            transformer: WanTransformer3D::zeros(WanVideoArchConfig::tiny(), device),
            vae: AutoencoderKlWan::zeros(WanVaeConfig::tiny(), device),
            device: device.clone(),
            tiny: true,
        })
    }

    pub fn load(root: &Path) -> Result<Self> {
        Self::load_on(root, &default_device())
    }

    pub fn load_on(root: &Path, device: &Device) -> Result<Self> {
        let dit = WeightMap::from_dir(&root.join("transformer"))?;
        let vae = WeightMap::from_dir(&root.join("vae"))?;
        let text_dir = if root.join("text_encoder").is_dir() {
            root.join("text_encoder")
        } else {
            root.join("text_encoder_2")
        };
        let text = WeightMap::from_dir(&text_dir)?;
        Ok(Self {
            text: Umt5Encoder::load(Umt5Config::xxl(), &text, device)?,
            transformer: WanTransformer3D::load(WanVideoArchConfig::wan_t2v_1_3b(), &dit, device)?,
            vae: AutoencoderKlWan::load(WanVaeConfig::wan_2_1(), &vae, device)?,
            device: device.clone(),
            tiny: false,
        })
    }

    fn encode_ids(&self, ids: &[u32]) -> Tensor<B, 3> {
        let input = Tensor::<B, 1, Int>::from_ints(
            ids.iter().map(|&x| x as i32).collect::<Vec<_>>().as_slice(),
            &self.device,
        )
        .unsqueeze::<2>();
        self.text.forward(input, None)
    }

    fn encode_prompt(&self, cfg: &GenerateConfig) -> Result<Tensor<B, 3>> {
        let text_len = self.transformer.cfg.text_len;
        if let Some(tokenizer) = cfg.tokenizer_path.as_ref() {
            let (prompt_ids, prompt_len) = tokenize_prompt(tokenizer, &cfg.prompt, text_len)?;
            let (neg_ids, neg_len) =
                tokenize_prompt(tokenizer, &cfg.negative_prompt, text_len)?;
            let prompt_embeds =
                pad_prompt_embeds(self.encode_ids(&prompt_ids), &[prompt_len], text_len)?;
            let neg_embeds = pad_prompt_embeds(self.encode_ids(&neg_ids), &[neg_len], text_len)?;
            return Ok(Tensor::cat(vec![neg_embeds, prompt_embeds], 0));
        }
        if self.tiny {
            let seq = text_len.min(8);
            let dummy: Vec<u32> = (0..seq).map(|i| (i % 10) as u32).collect();
            let prompt_embeds = self.encode_ids(&dummy);
            let neg_embeds = prompt_embeds.clone();
            let prompt_embeds = pad_prompt_embeds(prompt_embeds, &[seq], text_len)?;
            let neg_embeds = pad_prompt_embeds(neg_embeds, &[seq], text_len)?;
            return Ok(Tensor::cat(vec![neg_embeds, prompt_embeds], 0));
        }
        Err(BurnError::msg(
            "real generate needs tokenizer.json next to the Diffusers weights",
        ))
    }

    pub fn generate(&self, cfg: &GenerateConfig) -> Result<Vec<String>> {
        let (z_t, z_h, z_w, z_c) = if self.tiny {
            (2usize, 4usize, 4usize, 4usize)
        } else {
            (
                (cfg.num_frames.saturating_sub(1)) / 4 + 1,
                cfg.height / 8,
                cfg.width / 8,
                self.transformer.cfg.out_channels,
            )
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let n_el = z_c * z_t * z_h * z_w;
        let noise: Vec<f32> = (0..n_el)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let mut latents =
            Tensor::<B, 1>::from_floats(noise.as_slice(), &self.device).reshape([1, z_c, z_t, z_h, z_w]);

        let encoder_hs = self.encode_prompt(cfg)?;
        let guidance = cfg.guidance_scale;

        if cfg.is_dmd {
            let steps = cfg
                .dmd_steps
                .clone()
                .unwrap_or_else(|| FAST_WAN_1_3B_DMD_STEPS.to_vec());
            let s = DmdSchedule::new(&steps, cfg.flow_shift, 1000);
            let timesteps: Vec<f32> = s.train_timesteps.iter().map(|&t| t as f32).collect();
            latents = euler_denoise(
                latents,
                &encoder_hs,
                &timesteps,
                &s.sigmas,
                &self.transformer,
                guidance,
            )?;
        } else {
            let mut sched = FlowUniPCMultistepScheduler::new(1000, cfg.flow_shift);
            sched.set_timesteps(cfg.num_inference_steps);
            latents = unipc_denoise(latents, &encoder_hs, &mut sched, &self.transformer, guidance)?;
        }

        let latents = if !self.tiny {
            self.vae.scale_latents(latents)
        } else {
            latents
        };
        let video = self.vae.decode(latents);
        write_frames(&video, Path::new(&cfg.output_dir))
    }
}

fn cfg_guide(uncond: Tensor<B, 5>, text: Tensor<B, 5>, scale: f32) -> Tensor<B, 5> {
    if (scale - 1.0).abs() < 1e-6 {
        return text;
    }
    uncond.clone() + (text - uncond).mul_scalar(scale)
}

fn dit_cfg(
    transformer: &WanTransformer3D,
    latents: &Tensor<B, 5>,
    encoder_hs: &Tensor<B, 3>,
    t: f32,
    guidance: f32,
) -> Result<Tensor<B, 5>> {
    let device = &latents.device();
    let t_tensor = Tensor::<B, 1>::from_floats([t], device);
    let cond_hs = encoder_hs.clone().narrow(0, 1, 1);
    let cond = transformer.forward(latents.clone(), t_tensor.clone(), cond_hs);
    if (guidance - 1.0).abs() < 1e-6 {
        return Ok(cond);
    }
    let uncond_hs = encoder_hs.clone().narrow(0, 0, 1);
    let uncond = transformer.forward(latents.clone(), t_tensor, uncond_hs);
    Ok(cfg_guide(uncond, cond, guidance))
}

fn euler_denoise(
    mut latents: Tensor<B, 5>,
    encoder_hs: &Tensor<B, 3>,
    timesteps: &[f32],
    sigmas: &[f64],
    transformer: &WanTransformer3D,
    guidance: f32,
) -> Result<Tensor<B, 5>> {
    for (i, &t) in timesteps.iter().enumerate() {
        let guided = dit_cfg(transformer, &latents, encoder_hs, t, guidance)?;
        let dt = (sigmas[i + 1] - sigmas[i]) as f32;
        latents = latents + guided.mul_scalar(dt);
    }
    Ok(latents)
}

fn unipc_denoise(
    mut latents: Tensor<B, 5>,
    encoder_hs: &Tensor<B, 3>,
    sched: &mut FlowUniPCMultistepScheduler,
    transformer: &WanTransformer3D,
    guidance: f32,
) -> Result<Tensor<B, 5>> {
    let shape = latents.dims();
    let device = latents.device();
    let ts: Vec<f32> = sched
        .inference_timesteps_i64()
        .iter()
        .map(|t| *t as f32)
        .collect();
    for &t in &ts {
        let guided = dit_cfg(transformer, &latents, encoder_hs, t, guidance)?;
        let vel = to_vec_f32(guided.reshape([shape.iter().product::<usize>()]))?;
        let x = to_vec_f32(latents.clone().reshape([shape.iter().product::<usize>()]))?;
        let prev = sched
            .step(&vel, &x)
            .map_err(BurnError::msg)?;
        latents = Tensor::<B, 1>::from_floats(prev.as_slice(), &device).reshape(shape);
    }
    Ok(latents)
}

pub fn tokenize_prompt(path: &str, text: &str, max_len: usize) -> Result<(Vec<u32>, usize)> {
    fastvideo_models::tokenize_prompt(path, text, max_len)
        .map_err(|e| BurnError::msg(format!("{e}")))
}

fn write_frames(video: &Tensor<B, 5>, dir: &Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(dir)?;
    // video: [1, C, T, H, W] → squeeze batch
    let video = video.clone().squeeze::<4>(0);
    let [c, t, h, w] = video.dims();
    if c != 3 {
        return Err(BurnError::msg(format!("expected 3-channel video, got C={c}")));
    }
    // (video + 1) * 127.5, clamp 0..255
    let scaled = (video.add_scalar(1.0)).mul_scalar(127.5).clamp(0.0, 255.0);
    let data = to_vec_f32(scaled)?;
    let mut paths = Vec::new();
    for ti in 0..t {
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    // layout CTHW
                    let idx = ch * (t * h * w) + ti * (h * w) + y * w + x;
                    rgb[(y * w + x) * 3 + ch] = data[idx] as u8;
                }
            }
        }
        let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
            .ok_or_else(|| BurnError::msg("rgb buffer size mismatch"))?;
        let path = dir.join(format!("frame-{ti:03}.png"));
        img.save(&path)?;
        paths.push(path.to_string_lossy().into_owned());
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_generate_writes_png() {
        let pipe = WanPipeline::tiny().unwrap();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.is_dmd = true;
        cfg.flow_shift = 8.0;
        cfg.guidance_scale = 1.0;
        cfg.output_dir = std::env::temp_dir()
            .join("fastvideo-burn-tiny")
            .to_string_lossy()
            .into();
        let paths = pipe.generate(&cfg).unwrap();
        assert!(!paths.is_empty());
        assert!(Path::new(&paths[0]).exists());
    }
}
