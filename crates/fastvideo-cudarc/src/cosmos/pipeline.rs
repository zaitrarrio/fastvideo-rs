//! Cosmos Predict2 generate: T5 → DiT EDM → Wan VAE → PNG frames.

use std::path::{Path, PathBuf};

use fastvideo_models::cosmos::{CosmosPreset, CosmosSchedule, CosmosTransformerConfig};
use fastvideo_models::wan::WanVaeConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::vae::AutoencoderKlWan;
use crate::wan::weights::WeightMap;

use super::text;
use super::transformer::CosmosTransformer;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Default Diffusers `sigma_conditioning` for Video2World cond frames.
pub const SIGMA_CONDITIONING: f64 = 0.0001;

/// Diffusers `DEFAULT_NEGATIVE_PROMPT` for Cosmos2VideoToWorldPipeline.
pub const DEFAULT_NEGATIVE_PROMPT: &str = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, \
over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, \
underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, \
jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, \
fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. \
Overall, the video is of poor quality.";

#[derive(Debug, Clone)]
pub struct CosmosRequest {
    pub prompt: String,
    /// Empty → Diffusers [`DEFAULT_NEGATIVE_PROMPT`] when CFG is on.
    pub negative_prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub fps: u32,
    pub preset: CosmosPreset,
    /// First-frame / image conditioning for Video2World (`condition_mask`).
    pub image_path: Option<PathBuf>,
    pub sigma_conditioning: f64,
}

impl CosmosRequest {
    pub fn v2w_2b(prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: String::new(),
            seed,
            height: 704,
            width: 1280,
            num_frames: 93,
            num_steps: 35,
            guidance_scale: 7.0,
            fps: 16,
            preset: CosmosPreset::V2w2b,
            image_path: None,
            sigma_conditioning: SIGMA_CONDITIONING,
        }
    }

    pub fn do_classifier_free_guidance(&self) -> bool {
        self.guidance_scale > 1.0
    }

    pub fn resolved_negative_prompt(&self) -> &str {
        if self.negative_prompt.is_empty() {
            DEFAULT_NEGATIVE_PROMPT
        } else {
            &self.negative_prompt
        }
    }
}

pub struct CosmosPipeline {
    pub root: PathBuf,
    pub dit_cfg: CosmosTransformerConfig,
    pub preset: CosmosPreset,
    pub dit: Option<CosmosTransformer>,
    pub vae: Option<AutoencoderKlWan>,
}

impl CosmosPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: CosmosPreset) -> Result<Self> {
        let dit_cfg = match preset {
            CosmosPreset::V2w2b => CosmosTransformerConfig::predict2_2b(),
            CosmosPreset::V2w14b => CosmosTransformerConfig::predict2_14b(),
        };
        Ok(Self {
            root: root.into(),
            dit_cfg,
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map =
            WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(CosmosTransformer::load(self.dit_cfg.clone(), &map)?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        let mut cfg = WanVaeConfig::wan_2_1();
        cfg.load_encoder = true;
        self.vae = Some(AutoencoderKlWan::load(cfg, &map).map_err(|e| msg(e.to_string()))?);
        Ok(())
    }

    pub fn schedule(&self, steps: usize) -> CosmosSchedule {
        CosmosSchedule::new(steps, self.preset)
    }

    /// T5 encode when `text_encoder/` + tokenizer exist; else zero embeds for dry-runs.
    fn encode_text(&self, prompt: &str) -> Result<CudaTensor> {
        let te = self.root.join("text_encoder");
        let tok = self.root.join("tokenizer").join("tokenizer.json");
        if te.is_dir() && tok.is_file() {
            return text::encode_prompt(&self.root, prompt, self.dit_cfg.text_embed_dim)
                .map_err(|e| msg(e.to_string()));
        }
        if te.is_dir() {
            return Err(msg(
                "cosmos text_encoder present but tokenizer/tokenizer.json missing",
            ));
        }
        Ok(CudaTensor::zeros(&[1, 16, self.dit_cfg.text_embed_dim]))
    }

    /// Diffusers `prepare_latents` cond packing: first-frame VAE encode → indicator/mask.
    fn prepare_conditioning(
        &self,
        request: &CosmosRequest,
        lt: usize,
        lh: usize,
        lw: usize,
        c: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let spatial = lt * lh * lw;
        let mut cond_latents = vec![0f32; c * spatial];
        let mut cond_mask = vec![0f32; spatial];
        let Some(path) = request.image_path.as_ref() else {
            return Ok((cond_latents, cond_mask, 0));
        };
        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("Cosmos Video2World image_path set: call load_vae() so Wan VAE can encode")
        })?;
        let video = load_rgb_frame(path, request.height, request.width)?;
        // Single cond frame → 1 latent frame with Wan temporal factor 4.
        let encoded = vae.encode_video(&video).map_err(|e| msg(e.to_string()))?;
        let encoded = vae
            .normalize_latents(&encoded)
            .map_err(|e| msg(e.to_string()))?;
        // sigma_data scale (Predict2 default 1.0).
        let encoded = encoded
            .try_mul_scalar(self.preset.sigma_data() as f32)
            .map_err(|e| msg(e.to_string()))?;
        let eh = encoded.host_cow()?;
        let [_, ec, et, ey, ex] = match encoded.shape[..] {
            [1, ec, et, ey, ex] => [1, ec, et, ey, ex],
            _ => {
                return Err(msg(format!(
                    "cosmos cond encode shape {:?} want [1,C,T,H,W]",
                    encoded.shape
                )))
            }
        };
        if ec != c || ey != lh || ex != lw {
            return Err(msg(format!(
                "cosmos cond latent [{ec},{et},{ey},{ex}] vs want [{c},*,{lh},{lw}]"
            )));
        }
        let num_cond_frames = 1usize;
        let temp = 4usize;
        let num_cond_latent = (num_cond_frames - 1) / temp + 1;
        let num_cond_latent = num_cond_latent.min(lt).min(et);
        for ti in 0..num_cond_latent {
            for y in 0..lh {
                for x in 0..lw {
                    let cell = ti * lh * lw + y * lw + x;
                    cond_mask[cell] = 1.0;
                    for ch in 0..c {
                        let src = ((ch * et + ti) * ey + y) * ex + x;
                        cond_latents[ch * spatial + cell] = eh[src];
                    }
                }
            }
        }
        Ok((cond_latents, cond_mask, num_cond_latent))
    }

    /// Pack noise/cond into DiT input and build per-frame AdaLN timesteps.
    fn pack_step(
        &self,
        sample: &[f32],
        cond_latents: &[f32],
        cond_mask: &[f32],
        c_in: f64,
        current_t: f32,
        t_cond: f32,
        lt: usize,
        lh: usize,
        lw: usize,
        c: usize,
    ) -> Result<(CudaTensor, Vec<f32>)> {
        let spatial = lt * lh * lw;
        let mut packed = vec![0f32; self.dit_cfg.in_channels * spatial];
        let mut frame_ts = vec![current_t; lt];
        for j in 0..spatial {
            let m = cond_mask[j];
            let ti = j / (lh * lw);
            if m > 0.5 {
                frame_ts[ti] = t_cond;
            }
            for ch in 0..c {
                let idx = ch * spatial + j;
                let scaled = sample[idx] * c_in as f32;
                packed[idx] = m * cond_latents[idx] + (1.0 - m) * scaled;
            }
            packed[c * spatial + j] = m;
        }
        let lat = CudaTensor::from_vec(packed, vec![1, self.dit_cfg.in_channels, lt, lh, lw])?;
        Ok((lat, frame_ts))
    }

    fn edm_denoise(
        &self,
        sample: &[f32],
        pred: &CudaTensor,
        cond_latents: &[f32],
        cond_mask: &[f32],
        c_skip: f64,
        c_out: f64,
        n: usize,
        spatial: usize,
        c: usize,
    ) -> Result<Vec<f32>> {
        let ph = pred.host_cow()?;
        if ph.len() < n {
            return Err(msg(format!("cosmos pred {} vs {n}", ph.len())));
        }
        let mut denoised = vec![0f32; n];
        for j in 0..n {
            denoised[j] = c_skip as f32 * sample[j] + c_out as f32 * ph[j];
        }
        for j in 0..spatial {
            let m = cond_mask[j];
            if m > 0.5 {
                for ch in 0..c {
                    let idx = ch * spatial + j;
                    denoised[idx] = cond_latents[idx];
                }
            }
        }
        Ok(denoised)
    }

    pub fn generate(&self, request: &CosmosRequest, out_dir: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("Cosmos: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        let text_pos = self.encode_text(&request.prompt)?;
        let do_cfg = request.do_classifier_free_guidance();
        let text_neg = if do_cfg {
            Some(self.encode_text(request.resolved_negative_prompt())?)
        } else {
            None
        };

        let spat = 8usize;
        let temp = 4usize;
        let lt = 1 + (request.num_frames.saturating_sub(1)).div_ceil(temp);
        let lh = request.height.div_ceil(spat);
        let lw = request.width.div_ceil(spat);
        let c = self.dit_cfg.latent_channels();
        let spatial = lt * lh * lw;

        let (cond_latents, cond_mask, _n_cond) =
            self.prepare_conditioning(request, lt, lh, lw, c)?;

        let sched = self.schedule(request.num_steps);
        let sigmas = sched.inference_sigmas().to_vec();
        let sigma_max = self.preset.sigma_max() as f32;

        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let n = c * spatial;
        let mut sample: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(&mut rng);
                z * sigma_max
            })
            .collect();

        let t_cond = (request.sigma_conditioning / (request.sigma_conditioning + 1.0)) as f32;
        let fps = Some(request.fps as f32);

        for (i, &sigma) in sigmas.iter().enumerate() {
            let (c_in, c_skip, c_out) = CosmosSchedule::edm_coeffs(sigma);
            let current_t = (sigma / (sigma + 1.0)) as f32;
            let (lat, frame_ts) = self.pack_step(
                &sample,
                &cond_latents,
                &cond_mask,
                c_in,
                current_t,
                t_cond,
                lt,
                lh,
                lw,
                c,
            )?;

            let pred_pos = dit.forward(&lat, &text_pos, &frame_ts, fps)?;
            let mut denoised = self.edm_denoise(
                &sample,
                &pred_pos,
                &cond_latents,
                &cond_mask,
                c_skip,
                c_out,
                n,
                spatial,
                c,
            )?;

            if let Some(neg) = text_neg.as_ref() {
                let pred_neg = dit.forward(&lat, neg, &frame_ts, fps)?;
                let denoised_neg = self.edm_denoise(
                    &sample,
                    &pred_neg,
                    &cond_latents,
                    &cond_mask,
                    c_skip,
                    c_out,
                    n,
                    spatial,
                    c,
                )?;
                // Diffusers: noise_pred = pos + guidance_scale * (pos - uncond)
                let g = request.guidance_scale;
                for j in 0..n {
                    denoised[j] = denoised[j] + g * (denoised[j] - denoised_neg[j]);
                }
            }

            let mut deriv = vec![0f32; n];
            for j in 0..n {
                deriv[j] = (sample[j] - denoised[j]) / (sigma as f32).max(1e-6);
            }
            sample = sched.step_euler(&sample, &deriv, i).map_err(msg)?;
            // Keep conditioned latent frames fixed after the Euler step.
            for j in 0..spatial {
                if cond_mask[j] > 0.5 {
                    for ch in 0..c {
                        let idx = ch * spatial + j;
                        sample[idx] = cond_latents[idx];
                    }
                }
            }
        }

        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("Cosmos: call load_vae() after placing Diffusers `vae/` under --weights")
        })?;
        let latents = CudaTensor::from_vec(sample, vec![1, c, lt, lh, lw])?;
        let scaled = vae
            .scale_latents(&latents)
            .map_err(|e| msg(e.to_string()))?;
        let pixels = vae.decode(&scaled).map_err(|e| msg(e.to_string()))?;
        let [_, _, tf, hf, wf] = match pixels.shape[..] {
            [1, 3, tf, hf, wf] => [1, 3, tf, hf, wf],
            _ => {
                return Err(msg(format!(
                    "cosmos decode shape {:?} want [1,3,T,H,W]",
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

fn load_rgb_frame(path: &Path, height: usize, width: usize) -> Result<CudaTensor> {
    let img = image::open(path)
        .map_err(|e| msg(format!("open image {}: {e}", path.display())))?
        .into_rgb8();
    let img = image::imageops::resize(
        &img,
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let mut data = vec![0.0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x as u32, y as u32);
            for ch in 0..3 {
                let v = f32::from(p[ch]) / 127.5 - 1.0;
                data[ch * height * width + y * width + x] = v;
            }
        }
    }
    CudaTensor::from_vec(data, vec![1, 3, 1, height, width]).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = CosmosRequest::v2w_2b("a cat", 0);
        assert!(r.image_path.is_none());
        assert!((r.sigma_conditioning - SIGMA_CONDITIONING).abs() < 1e-12);
        assert!(r.do_classifier_free_guidance());
        assert_eq!(r.resolved_negative_prompt(), DEFAULT_NEGATIVE_PROMPT);
    }

    #[test]
    fn cfg_off_when_scale_one() {
        let mut r = CosmosRequest::v2w_2b("a cat", 0);
        r.guidance_scale = 1.0;
        assert!(!r.do_classifier_free_guidance());
    }
}
