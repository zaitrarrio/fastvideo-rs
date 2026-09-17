//! Wan inference pipeline: UMT5 → compiled/eager DiT → compiled/eager VAE.

use std::path::Path;

use fastvideo_models::schedulers::{
    DmdSchedule, FlowUniPCMultistepScheduler, FAST_WAN_1_3B_DMD_STEPS,
};
use fastvideo_models::wan::{Umt5Config, WanVaeConfig, WanVideoArchConfig};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use thiserror::Error;

use super::compiled::{CompiledDitStep, CompiledVaeDecode};
use super::tensor::{NdTensor, Result as TensorResult, TensorError};
use super::transformer::WanTransformer3D;
use super::umt5::{pad_prompt_embeds, Umt5Encoder};
use super::vae::AutoencoderKlWan;
use super::weights::WeightMap;

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
        }
    }
}

pub struct WanPipeline {
    text: Umt5Encoder,
    dit: CompiledDitStep,
    vae: CompiledVaeDecode,
    tiny: bool,
}

impl WanPipeline {
    /// Zero-weight tiny graph with compiled DiT step + VAE decode.
    pub fn tiny() -> Self {
        Self {
            text: Umt5Encoder::zeros(Umt5Config::tiny()),
            dit: CompiledDitStep::tiny(),
            vae: CompiledVaeDecode::tiny(),
            tiny: true,
        }
    }

    pub fn load(root: &Path) -> Result<Self> {
        let dit = WeightMap::from_dir(&root.join("transformer"))?;
        let vae = WeightMap::from_dir(&root.join("vae"))?;
        let text_dir = if root.join("text_encoder").is_dir() {
            root.join("text_encoder")
        } else {
            root.join("text_encoder_2")
        };
        let text = WeightMap::from_dir(&text_dir)?;
        Ok(Self {
            text: Umt5Encoder::load(Umt5Config::xxl(), &text)?,
            dit: CompiledDitStep::from_transformer(WanTransformer3D::load(
                WanVideoArchConfig::wan_t2v_1_3b(),
                &dit,
            )?),
            vae: CompiledVaeDecode::from_vae(AutoencoderKlWan::load(
                WanVaeConfig::wan_2_1(),
                &vae,
            )?),
            tiny: false,
        })
    }

    fn encode_prompt(&self, cfg: &GenerateConfig) -> Result<NdTensor> {
        let text_len = self.dit.transformer().cfg.text_len;
        if let Some(tokenizer) = cfg.tokenizer_path.as_ref() {
            let (prompt_ids, prompt_len) = tokenize_prompt(tokenizer, &cfg.prompt, text_len)?;
            let (neg_ids, neg_len) =
                tokenize_prompt(tokenizer, &cfg.negative_prompt, text_len)?;
            let prompt_embeds = self.text.forward(&prompt_ids, 1, prompt_ids.len())?;
            let neg_embeds = self.text.forward(&neg_ids, 1, neg_ids.len())?;
            let prompt_embeds = pad_prompt_embeds(&prompt_embeds, &[prompt_len], text_len)?;
            let neg_embeds = pad_prompt_embeds(&neg_embeds, &[neg_len], text_len)?;
            return Ok(NdTensor::cat(&[&neg_embeds, &prompt_embeds], 0)?);
        }
        if self.tiny {
            let seq = text_len.min(8);
            let dummy: Vec<u32> = (0..seq).map(|i| (i % 10) as u32).collect();
            let prompt_embeds = self.text.forward(&dummy, 1, seq)?;
            let neg_embeds = prompt_embeds.clone();
            let prompt_embeds = pad_prompt_embeds(&prompt_embeds, &[seq], text_len)?;
            let neg_embeds = pad_prompt_embeds(&neg_embeds, &[seq], text_len)?;
            return Ok(NdTensor::cat(&[&neg_embeds, &prompt_embeds], 0)?);
        }
        Err(PipelineError::Message(
            "real generate needs tokenizer.json next to the Diffusers weights".into(),
        ))
    }

    pub fn generate(&mut self, cfg: &GenerateConfig) -> Result<Vec<String>> {
        let (z_t, z_h, z_w, z_c) = if self.tiny {
            (2usize, 4usize, 4usize, 4usize)
        } else {
            (
                (cfg.num_frames.saturating_sub(1)) / 4 + 1,
                cfg.height / 8,
                cfg.width / 8,
                self.dit.transformer().cfg.out_channels,
            )
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let n_el = z_c * z_t * z_h * z_w;
        let noise: Vec<f32> = (0..n_el)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let mut latents = NdTensor::from_vec(noise, vec![1, z_c, z_t, z_h, z_w])?;
        let encoder_hs = self.encode_prompt(cfg)?;

        if cfg.is_dmd {
            let steps = cfg
                .dmd_steps
                .clone()
                .unwrap_or_else(|| FAST_WAN_1_3B_DMD_STEPS.to_vec());
            let s = DmdSchedule::new(&steps, 1000);
            let timesteps: Vec<f32> = s.train_timesteps.iter().map(|&t| t as f32).collect();
            latents = euler_denoise(
                latents,
                &encoder_hs,
                &timesteps,
                &s.sigmas,
                &mut self.dit,
                cfg.guidance_scale,
            )?;
        } else {
            let mut sched = FlowUniPCMultistepScheduler::new(1000, cfg.flow_shift);
            sched.set_timesteps(cfg.num_inference_steps);
            latents = unipc_denoise(
                latents,
                &encoder_hs,
                &mut sched,
                &mut self.dit,
                cfg.guidance_scale,
            )?;
        }

        let latents = if !self.tiny {
            self.vae.vae().scale_latents(&latents)?
        } else {
            latents
        };
        let video = self.vae.run(&latents)?;
        write_frames(&video, Path::new(&cfg.output_dir))
    }

    pub fn transformer(&self) -> &WanTransformer3D {
        self.dit.transformer()
    }

    pub fn vae(&self) -> &AutoencoderKlWan {
        self.vae.vae()
    }
}

fn tokenize_prompt(path: &str, text: &str, max_len: usize) -> Result<(Vec<u32>, usize)> {
    fastvideo_models::tokenize_prompt(path, text, max_len)
        .map_err(|e| PipelineError::Message(format!("{e}")))
}

fn dit_cfg(
    dit: &mut CompiledDitStep,
    latents: &NdTensor,
    encoder_hs: &NdTensor,
    t: f32,
    guidance: f32,
) -> TensorResult<NdTensor> {
    let t_tensor = NdTensor::from_vec(vec![t], vec![1])?;
    let cond_hs = encoder_hs.narrow(0, 1, 1)?;
    let cond = dit.run(latents, &t_tensor, &cond_hs)?;
    if (guidance - 1.0).abs() < 1e-6 {
        return Ok(cond);
    }
    let uncond_hs = encoder_hs.narrow(0, 0, 1)?;
    let uncond = dit.run(latents, &t_tensor, &uncond_hs)?;
    let delta = cond.sub(&uncond)?;
    uncond.add(&delta.mul_scalar(guidance))
}

fn euler_denoise(
    mut latents: NdTensor,
    encoder_hs: &NdTensor,
    timesteps: &[f32],
    sigmas: &[f64],
    dit: &mut CompiledDitStep,
    guidance: f32,
) -> Result<NdTensor> {
    for (i, &t) in timesteps.iter().enumerate() {
        let guided = dit_cfg(dit, &latents, encoder_hs, t, guidance)?;
        let dt = sigmas[i + 1] - sigmas[i];
        let delta = guided.mul_scalar(dt as f32);
        latents = latents.add(&delta)?;
    }
    Ok(latents)
}

fn unipc_denoise(
    mut latents: NdTensor,
    encoder_hs: &NdTensor,
    sched: &mut FlowUniPCMultistepScheduler,
    dit: &mut CompiledDitStep,
    guidance: f32,
) -> Result<NdTensor> {
    let shape = latents.shape.clone();
    let ts: Vec<f32> = sched
        .inference_timesteps_i64()
        .iter()
        .map(|t| *t as f32)
        .collect();
    for &t in &ts {
        let guided = dit_cfg(dit, &latents, encoder_hs, t, guidance)?;
        let prev = sched
            .step(&guided.data, &latents.data)
            .map_err(PipelineError::Message)?;
        latents = NdTensor::from_vec(prev, shape.clone())?;
    }
    Ok(latents)
}

fn write_frames(video: &NdTensor, dir: &Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(dir).map_err(|e| PipelineError::Message(e.to_string()))?;
    if video.rank() != 5 || video.shape[0] != 1 {
        return Err(PipelineError::Message(format!(
            "expected 1CTHW video, got {:?}",
            video.shape
        )));
    }
    let (_b, c, t, h, w) = (
        video.shape[0],
        video.shape[1],
        video.shape[2],
        video.shape[3],
        video.shape[4],
    );
    if c < 3 {
        return Err(PipelineError::Message("video needs RGB channels".into()));
    }
    let mut paths = Vec::new();
    for ti in 0..t {
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let v = video.data[(((0 * c + ch) * t + ti) * h + y) * w + x];
                    let byte = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                    rgb[(y * w + x) * 3 + ch] = byte;
                }
            }
        }
        let img = image::RgbImage::from_raw(w as u32, h as u32, rgb)
            .ok_or_else(|| PipelineError::Message("rgb buffer size mismatch".into()))?;
        let path = dir.join(format!("frame-{ti:03}.png"));
        img.save(&path)
            .map_err(|e| PipelineError::Message(e.to_string()))?;
        paths.push(path.to_string_lossy().into_owned());
    }
    Ok(paths)
}
