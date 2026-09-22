//! Matrix-Game generate scaffold on Wan DiT / VAE.

use std::path::{Path, PathBuf};

use fastvideo_models::matrixgame::{MatrixGameConfig, MatrixGamePreset};
use fastvideo_models::schedulers::FlowMatchEulerDiscreteScheduler;
use fastvideo_models::wan::WanVaeConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::transformer::WanTransformer3D;
use crate::wan::vae::AutoencoderKlWan;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct MatrixGameRequest {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub preset: MatrixGamePreset,
    pub image_path: Option<PathBuf>,
    /// Keyboard one-hot / multi-hot `[T, keyboard_dim]` (host; optional).
    pub keyboard_cond: Option<Vec<f32>>,
    /// Mouse deltas `[T, mouse_dim]`.
    pub mouse_cond: Option<Vec<f32>>,
}

impl MatrixGameRequest {
    pub fn for_preset(preset: MatrixGamePreset, prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            seed,
            height: preset.default_height(),
            width: preset.default_width(),
            num_frames: preset.default_num_frames(),
            num_steps: preset.default_steps(),
            guidance_scale: 1.0,
            preset,
            image_path: None,
            keyboard_cond: None,
            mouse_cond: None,
        }
    }
}

pub struct MatrixGamePipeline {
    pub root: PathBuf,
    pub cfg: MatrixGameConfig,
    pub preset: MatrixGamePreset,
    pub dit: Option<WanTransformer3D>,
    pub vae: Option<AutoencoderKlWan>,
}

impl MatrixGamePipeline {
    pub fn open(root: impl Into<PathBuf>, preset: MatrixGamePreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: MatrixGameConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(
            WanTransformer3D::load(self.cfg.wan.clone(), &map).map_err(|e| msg(e.to_string()))?,
        );
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = MatrixGameConfig::tiny();
        self.dit = Some(WanTransformer3D::zeros(self.cfg.wan.clone()));
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        let mut cfg = if self.preset.is_mg3() {
            WanVaeConfig::wan_2_2()
        } else {
            WanVaeConfig::wan_2_1()
        };
        cfg.load_encoder = true;
        self.vae = Some(AutoencoderKlWan::load(cfg, &map).map_err(|e| msg(e.to_string()))?);
        Ok(())
    }

    pub fn generate(&self, request: &MatrixGameRequest, out_dir: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("Matrix-Game: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        let _ = (&request.keyboard_cond, &request.mouse_cond, &request.image_path);

        let spat = 8usize;
        let temp = 4usize;
        let lt = 1 + (request.num_frames.saturating_sub(1)).div_ceil(temp);
        let lh = request.height.div_ceil(spat);
        let lw = request.width.div_ceil(spat);
        let c = self.cfg.wan.out_channels;
        let in_c = self.cfg.wan.in_channels;
        let spatial = lt * lh * lw;
        let n = in_c * spatial;

        let sched = FlowMatchEulerDiscreteScheduler::new(
            request.num_steps,
            request.preset.flow_shift(),
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let mut sample: Vec<f32> = (0..n)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        let enc = CudaTensor::zeros(&[1, 16, self.cfg.wan.text_dim.max(16)]);
        for (i, &t) in sched.timesteps.iter().enumerate() {
            let lat = CudaTensor::from_vec(sample.clone(), vec![1, in_c, lt, lh, lw])?;
            let ts = CudaTensor::from_vec(vec![t as f32], vec![1])?;
            let pred = dit.forward(&lat, &ts, &enc).map_err(|e| msg(e.to_string()))?;
            let ph = pred.host_cow()?;
            // Use first `c` channels as velocity when in_c > out_c.
            let mut vel = vec![0f32; c * spatial];
            let pred_c = pred.shape.get(1).copied().unwrap_or(c).min(c);
            for ch in 0..pred_c {
                for j in 0..spatial {
                    vel[ch * spatial + j] = ph[ch * spatial + j];
                }
            }
            // Euler on latent channels only.
            let mut lat_only: Vec<f32> = sample[..c * spatial].to_vec();
            lat_only = sched
                .step(&lat_only, &vel, i)
                .map_err(msg)?;
            sample[..c * spatial].copy_from_slice(&lat_only);
        }

        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("Matrix-Game: call load_vae() after placing Diffusers `vae/` under --weights")
        })?;
        let latents = CudaTensor::from_vec(sample[..c * spatial].to_vec(), vec![1, c, lt, lh, lw])?;
        let scaled = vae
            .scale_latents(&latents)
            .map_err(|e| msg(e.to_string()))?;
        let pixels = vae.decode(&scaled).map_err(|e| msg(e.to_string()))?;
        let [_, _, tf, hf, wf] = match pixels.shape[..] {
            [1, 3, tf, hf, wf] => [1, 3, tf, hf, wf],
            _ => {
                return Err(msg(format!(
                    "matrixgame decode shape {:?} want [1,3,T,H,W]",
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
    fn request_mg3_defaults() {
        let r = MatrixGameRequest::for_preset(
            MatrixGamePreset::Mg3BaseDistilled,
            "",
            0,
        );
        assert_eq!(r.height, 720);
        assert_eq!(r.num_steps, 3);
    }

    #[test]
    fn tiny_dit_forward() {
        let mut pipe =
            MatrixGamePipeline::open("/tmp/mg-missing", MatrixGamePreset::Mg2BaseDistilled)
                .unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        let dit = pipe.dit.as_ref().unwrap();
        let cfg = &pipe.cfg.wan;
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let ts = CudaTensor::from_vec(vec![500f32], vec![1]).unwrap();
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_dim.max(1)]);
        let out = dit.forward(&x, &ts, &enc).unwrap();
        assert_eq!(out.shape[0], 1);
        assert_eq!(out.shape[1], cfg.out_channels);
    }
}
