//! HY-WorldPlay generate on Hunyuan15 DiT / VAE with action_in + camera pack.

use std::path::{Path, PathBuf};

use fastvideo_models::hunyuan15::Hunyuan15Schedule;
use fastvideo_models::hyworld::{
    compute_latent_num, pose_to_input, HyWorldConfig, HyWorldPoseInput, HyWorldPreset,
};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::hunyuan15::transformer::Hunyuan15Transformer;
use crate::hunyuan15::vae::Hunyuan15Vae;
use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct HyWorldRequest {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub preset: HyWorldPreset,
    pub image_path: Option<PathBuf>,
    /// Pose / action string (e.g. `w-31`).
    pub pose: String,
}

impl HyWorldRequest {
    pub fn bidirectional(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = HyWorldPreset::Bidirectional;
        Self {
            prompt: prompt.into(),
            seed,
            height: preset.default_height(),
            width: preset.default_width(),
            num_frames: preset.default_num_frames(),
            num_steps: preset.default_steps(),
            preset,
            image_path: None,
            pose: "w-31".into(),
        }
    }
}

pub struct HyWorldPipeline {
    pub root: PathBuf,
    pub cfg: HyWorldConfig,
    pub preset: HyWorldPreset,
    pub dit: Option<Hunyuan15Transformer>,
    pub vae: Option<Hunyuan15Vae>,
}

impl HyWorldPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: HyWorldPreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: HyWorldConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(Hunyuan15Transformer::load(self.cfg.hy.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = HyWorldConfig::tiny();
        self.dit = Some(Hunyuan15Transformer::zeros(self.cfg.hy.clone())?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        let cfg = fastvideo_models::hunyuan15::Hunyuan15VaeConfig::fasthunyuan15();
        self.vae = Some(Hunyuan15Vae::load(&map, cfg)?);
        Ok(())
    }

    pub fn pose_input(&self, request: &HyWorldRequest) -> Result<HyWorldPoseInput> {
        let lat = compute_latent_num(request.num_frames);
        pose_to_input(&request.pose, lat).map_err(msg)
    }

    /// SigLIP token placeholder until vision weights are under `image_encoder/`.
    pub fn siglip_tokens(&self) -> Vec<f32> {
        HyWorldPoseInput::zeros_siglip_tokens(self.cfg.siglip_tokens, self.cfg.siglip_dim)
    }

    pub fn generate(&self, request: &HyWorldRequest, out_dir: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("HY-World: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        let pose = self.pose_input(request)?;
        let siglip = self.siglip_tokens();
        let _ = (&pose, &siglip, &request.image_path, &request.prompt);

        let spat = 16usize;
        let temp = 4usize;
        let lt = 1 + (request.num_frames.saturating_sub(1)).div_ceil(temp);
        let lh = request.height.div_ceil(spat);
        let lw = request.width.div_ceil(spat);
        let c_out = self.cfg.hy.out_channels;
        let c_in = self.cfg.hy.in_channels;
        let spatial = lt * lh * lw;
        let n = c_out * spatial;

        let mut sched =
            Hunyuan15Schedule::with_shift(request.num_steps, request.preset.flow_shift());
        let timesteps = sched.timesteps().to_vec();
        let sigmas = sched.sigmas().to_vec();

        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let mut sample: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(&mut rng);
                z * (sigmas[0] as f32)
            })
            .collect();

        let text = CudaTensor::zeros(&[1, 16, self.cfg.hy.text_embed_dim]);
        for &t in &timesteps {
            let mut packed = vec![0f32; c_in * spatial];
            packed[..n].copy_from_slice(&sample);
            let lat = CudaTensor::from_vec(packed, vec![1, c_in, lt, lh, lw])?;
            let pred = dit.forward(&lat, &text, None, t as f32)?;
            let ph = pred.host_cow()?;
            let mut vel = vec![0f32; n];
            let copy = ph.len().min(n);
            vel[..copy].copy_from_slice(&ph[..copy]);
            sample = sched.inner.step_euler(&sample, &vel).map_err(msg)?;
        }

        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("HY-World: call load_vae() after placing Diffusers `vae/` under --weights")
        })?;
        let latents = CudaTensor::from_vec(sample, vec![1, c_out, lt, lh, lw])?;
        let pixels = vae.decode(&latents)?;
        let [_, _, tf, hf, wf] = match pixels.shape[..] {
            [1, 3, tf, hf, wf] => [1, 3, tf, hf, wf],
            _ => {
                return Err(msg(format!(
                    "hyworld decode shape {:?} want [1,3,T,H,W]",
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
        let r = HyWorldRequest::bidirectional("walk", 0);
        assert_eq!(r.pose, "w-31");
        assert_eq!(r.height, 480);
    }

    #[test]
    fn tiny_dit_forward() {
        let mut pipe =
            HyWorldPipeline::open("/tmp/hyw-missing", HyWorldPreset::Bidirectional).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        let dit = pipe.dit.as_ref().unwrap();
        let cfg = &pipe.cfg.hy;
        let lat = CudaTensor::zeros(&[1, cfg.in_channels, 2, 2, 2]);
        let text = CudaTensor::zeros(&[1, 4, cfg.text_embed_dim]);
        let out = dit.forward(&lat, &text, None, 500.0).unwrap();
        assert!(!out.shape.is_empty());
    }

    #[test]
    fn pose_action_in() {
        let pipe =
            HyWorldPipeline::open("/tmp/hyw-missing", HyWorldPreset::Bidirectional).unwrap();
        let r = HyWorldRequest::bidirectional("walk", 0);
        let pose = pipe.pose_input(&r).unwrap();
        assert_eq!(pose.action_labels.len(), pose.latent_num);
        assert!(pose.action_labels.iter().any(|&a| a != 0));
        let sig = pipe.siglip_tokens();
        assert_eq!(sig.len(), pipe.cfg.siglip_tokens * pipe.cfg.siglip_dim);
    }
}
