//! DreamX generate on Wan 5B DiT / VAE with PRoPE camera pack.

use std::path::{Path, PathBuf};

use fastvideo_models::dreamx::{
    build_dreamx_camera_condition, DreamXCameraCondition, DreamXConfig, DreamXPreset,
};
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
pub struct DreamXRequest {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub preset: DreamXPreset,
    pub image_path: Option<PathBuf>,
    pub action_list: Vec<String>,
    pub action_speed_list: Vec<f32>,
}

impl DreamXRequest {
    pub fn for_preset(preset: DreamXPreset, prompt: impl Into<String>, seed: u64) -> Self {
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
            action_list: vec!["w".into(), "d".into(), "w".into()],
            action_speed_list: vec![4.0, 2.0, 4.0],
        }
    }
}

pub struct DreamXPipeline {
    pub root: PathBuf,
    pub cfg: DreamXConfig,
    pub preset: DreamXPreset,
    pub dit: Option<WanTransformer3D>,
    pub vae: Option<AutoencoderKlWan>,
}

impl DreamXPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: DreamXPreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: DreamXConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(WanTransformer3D::load(self.cfg.wan.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = DreamXConfig::tiny();
        self.dit = Some(WanTransformer3D::zeros(self.cfg.wan.clone()));
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        let mut cfg = WanVaeConfig::wan_2_1();
        cfg.z_dim = 48;
        cfg.load_encoder = true;
        self.vae = Some(AutoencoderKlWan::load(cfg, &map).map_err(|e| msg(e.to_string()))?);
        Ok(())
    }

    pub fn camera_condition(&self, request: &DreamXRequest) -> Result<DreamXCameraCondition> {
        build_dreamx_camera_condition(
            &request.action_list,
            &request.action_speed_list,
            request.num_frames,
        )
        .map_err(msg)
    }

    pub fn generate(&self, request: &DreamXRequest, out_dir: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("DreamX: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        // viewmats + K for PRoPE control adapter (injected when adapter weights load).
        let cam = self.camera_condition(request)?;
        let fuse = crate::world_fuse::WorldFuse::try_load(
            &self.root,
            crate::world_fuse::FuseKind::Prope,
            self.cfg.wan.out_channels,
        )?;
        let mut cam_flat = cam.viewmats.clone();
        cam_flat.extend_from_slice(&cam.intrinsics);

        let spat = 8usize;
        let temp = 4usize;
        let lt = 1 + (request.num_frames.saturating_sub(1)).div_ceil(temp);
        let lh = request.height.div_ceil(spat);
        let lw = request.width.div_ceil(spat);
        let c = self.cfg.wan.out_channels;
        let spatial = lt * lh * lw;
        let n = c * spatial;

        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, request.preset.flow_shift());
        sched.set_timesteps(request.num_steps);
        let timesteps = sched.inference_timesteps().to_vec();
        let sigmas = sched.inference_sigmas().to_vec();

        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let mut sample: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(&mut rng);
                z * (sigmas[0] as f32)
            })
            .collect();

        let enc = CudaTensor::zeros(&[1, 16, self.cfg.wan.text_dim]);
        for &t in &timesteps {
            let lat = CudaTensor::from_vec(sample.clone(), vec![1, c, lt, lh, lw])?;
            let lat = if let Some(ref f) = fuse {
                f.fuse_into_latents(&lat, &cam_flat)?
            } else {
                lat
            };
            let ts = CudaTensor::from_vec(vec![t as f32], vec![1])?;
            let pred = dit.forward(&lat, &ts, &enc)?;
            let vel = pred.host_cow()?;
            sample = sched.step_euler(&sample, &vel[..n]).map_err(msg)?;
        }
        let _ = (&request.image_path, &request.prompt);

        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("DreamX: call load_vae() after placing Diffusers `vae/` under --weights")
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
                    "dreamx decode shape {:?} want [1,3,T,H,W]",
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
    fn request_cam_defaults() {
        let r = DreamXRequest::for_preset(DreamXPreset::Cam5b, "drive", 0);
        assert_eq!(r.height, 480);
        assert_eq!(r.action_list.len(), 3);
    }

    #[test]
    fn tiny_dit_forward() {
        let mut pipe = DreamXPipeline::open("/tmp/dreamx-missing", DreamXPreset::Cam5b).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        let dit = pipe.dit.as_ref().unwrap();
        let cfg = &pipe.cfg.wan;
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let ts = CudaTensor::from_vec(vec![500f32], vec![1]).unwrap();
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_dim]);
        let out = dit.forward(&x, &ts, &enc).unwrap();
        assert_eq!(out.shape[1], cfg.out_channels);
    }

    #[test]
    fn builds_prope_camera() {
        let pipe = DreamXPipeline::open("/tmp/dreamx-missing", DreamXPreset::Cam5b).unwrap();
        let r = DreamXRequest::for_preset(DreamXPreset::Cam5b, "drive", 0);
        let cam = pipe.camera_condition(&r).unwrap();
        assert!(cam.num_latent_frames > 1);
        assert_eq!(cam.viewmats.len(), cam.num_latent_frames * 16);
    }
}
