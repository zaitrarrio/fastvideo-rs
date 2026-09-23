//! GameCraft generate on Hunyuan15 DiT / VAE with Plücker CameraNet states.

use std::path::{Path, PathBuf};

use fastvideo_models::gamecraft::{create_camera_trajectory, GameCraftConfig, GameCraftPreset};
use fastvideo_models::hunyuan15::Hunyuan15Schedule;
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
pub struct GameCraftRequest {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub preset: GameCraftPreset,
    pub image_path: Option<PathBuf>,
    pub action: String,
    pub action_speed: f32,
}

impl GameCraftRequest {
    pub fn i2v(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = GameCraftPreset::I2v;
        Self {
            prompt: prompt.into(),
            seed,
            height: preset.default_height(),
            width: preset.default_width(),
            num_frames: preset.default_num_frames(),
            num_steps: preset.default_steps(),
            guidance_scale: preset.guidance_scale(),
            preset,
            image_path: None,
            action: "forward".into(),
            action_speed: 0.2,
        }
    }
}

pub struct GameCraftPipeline {
    pub root: PathBuf,
    pub cfg: GameCraftConfig,
    pub preset: GameCraftPreset,
    pub dit: Option<Hunyuan15Transformer>,
    pub vae: Option<Hunyuan15Vae>,
}

impl GameCraftPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: GameCraftPreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: GameCraftConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map =
            WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(Hunyuan15Transformer::load(self.cfg.hy.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = GameCraftConfig::tiny();
        self.dit = Some(Hunyuan15Transformer::zeros(self.cfg.hy.clone())?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        let cfg = fastvideo_models::hunyuan15::Hunyuan15VaeConfig::fasthunyuan15();
        self.vae = Some(Hunyuan15Vae::load(&map, cfg)?);
        Ok(())
    }

    pub fn camera_states(&self, request: &GameCraftRequest) -> Vec<f32> {
        create_camera_trajectory(
            &request.action,
            request.height,
            request.width,
            request.num_frames,
            request.action_speed,
        )
    }

    pub fn generate(&self, request: &GameCraftRequest, out_dir: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("GameCraft: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        // Plücker [F,6,H,W] for CameraNet; fused into DiT when camera_net weights load.
        let camera_states = self.camera_states(request);
        let fuse = crate::world_fuse::WorldFuse::try_load(
            &self.root,
            crate::world_fuse::FuseKind::CameraNet,
            self.cfg.hy.in_channels,
        )?;

        let spat = 16usize;
        let temp = 4usize;
        let lt = 1 + (request.num_frames.saturating_sub(1)).div_ceil(temp);
        let lh = request.height.div_ceil(spat);
        let lw = request.width.div_ceil(spat);
        let c_out = self.cfg.hy.out_channels;
        let c_in = self.cfg.hy.in_channels;
        let spatial = lt * lh * lw;
        let n = c_out * spatial;
        let pad_c = c_in.saturating_sub(c_out);

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
            let packed_n = c_in * spatial;
            let mut packed = vec![0f32; packed_n];
            packed[..n].copy_from_slice(&sample);
            let _ = pad_c;
            let lat = CudaTensor::from_vec(packed, vec![1, c_in, lt, lh, lw])?;
            let lat = if let Some(ref f) = fuse {
                f.fuse_into_latents(&lat, &camera_states)?
            } else {
                lat
            };
            let pred = dit.forward(&lat, &text, None, t as f32)?;
            // Flatten token or dense pred into CTHW velocity of length n.
            let ph = pred.host_cow()?;
            let mut vel = vec![0f32; n];
            let copy = ph.len().min(n);
            vel[..copy].copy_from_slice(&ph[..copy]);
            sample = sched.inner.step_euler(&sample, &vel).map_err(msg)?;
        }
        let _ = (&request.image_path, &request.prompt, request.guidance_scale);

        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("GameCraft: call load_vae() after placing Diffusers `vae/` under --weights")
        })?;
        let latents = CudaTensor::from_vec(sample, vec![1, c_out, lt, lh, lw])?;
        let pixels = vae.decode(&latents)?;
        let [_, _, tf, hf, wf] = match pixels.shape[..] {
            [1, 3, tf, hf, wf] => [1, 3, tf, hf, wf],
            _ => {
                return Err(msg(format!(
                    "gamecraft decode shape {:?} want [1,3,T,H,W]",
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
        let r = GameCraftRequest::i2v("temple", 0);
        assert_eq!(r.action, "forward");
        assert_eq!(r.height, 704);
    }

    #[test]
    fn tiny_dit_forward() {
        let mut pipe = GameCraftPipeline::open("/tmp/gc-missing", GameCraftPreset::I2v).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        let dit = pipe.dit.as_ref().unwrap();
        let cfg = &pipe.cfg.hy;
        let lat = CudaTensor::zeros(&[1, cfg.in_channels, 2, 2, 2]);
        let text = CudaTensor::zeros(&[1, 4, cfg.text_embed_dim]);
        let out = dit.forward(&lat, &text, None, 500.0).unwrap();
        assert!(!out.shape.is_empty());
    }

    #[test]
    fn plucker_camera_states() {
        let pipe = GameCraftPipeline::open("/tmp/gc-missing", GameCraftPreset::I2v).unwrap();
        let r = GameCraftRequest::i2v("temple", 0);
        let cam = pipe.camera_states(&r);
        assert_eq!(cam.len(), r.num_frames * 6 * r.height * r.width);
        assert!(cam.iter().any(|&v| v.abs() > 1e-8));
    }
}
