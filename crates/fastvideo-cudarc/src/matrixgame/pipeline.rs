//! Matrix-Game generate on Wan DiT / VAE with keyboard/mouse action packing.

use std::path::{Path, PathBuf};

use fastvideo_models::matrixgame::{
    create_action_presets, ActionPack, MatrixGameConfig, MatrixGamePreset,
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
    pub keyboard_cond: Option<Vec<f32>>,
    pub mouse_cond: Option<Vec<f32>>,
    /// When action tensors are absent, build a deterministic preset (seeded).
    pub auto_actions: bool,
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
            auto_actions: true,
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
        let map =
            WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(WanTransformer3D::load(self.cfg.wan.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = MatrixGameConfig::tiny();
        self.dit = Some(WanTransformer3D::zeros(self.cfg.wan.clone()));
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        let mut cfg = WanVaeConfig::wan_2_1();
        if self.preset.is_mg3() {
            cfg.z_dim = 48;
        }
        cfg.load_encoder = true;
        self.vae = Some(AutoencoderKlWan::load(cfg, &map).map_err(|e| msg(e.to_string()))?);
        Ok(())
    }

    pub fn resolve_actions(&self, request: &MatrixGameRequest) -> ActionPack {
        if let (Some(kb), Some(mouse)) = (&request.keyboard_cond, &request.mouse_cond) {
            let kb_dim = self.cfg.keyboard_dim;
            let mut pack = ActionPack::zeros(request.num_frames, kb_dim);
            for t in 0..request.num_frames {
                let src = (t * kb_dim).min(kb.len().saturating_sub(kb_dim));
                if src + kb_dim <= kb.len() {
                    pack.keyboard[t * kb_dim..(t + 1) * kb_dim]
                        .copy_from_slice(&kb[src..src + kb_dim]);
                } else if kb.len() >= kb_dim {
                    pack.keyboard[t * kb_dim..(t + 1) * kb_dim].copy_from_slice(&kb[..kb_dim]);
                }
                let ms = if mouse.len() >= (t + 1) * 2 {
                    [mouse[t * 2], mouse[t * 2 + 1]]
                } else if mouse.len() >= 2 {
                    [mouse[0], mouse[1]]
                } else {
                    [0.0, 0.0]
                };
                pack.mouse[t * 2] = ms[0];
                pack.mouse[t * 2 + 1] = ms[1];
            }
            return pack;
        }
        if request.auto_actions {
            return create_action_presets(request.preset, request.num_frames, request.seed);
        }
        ActionPack::zeros(request.num_frames, self.cfg.keyboard_dim)
    }

    pub fn generate(&self, request: &MatrixGameRequest, out_dir: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("Matrix-Game: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;
        // Action tensors packed for DiT action-module injectors.
        let actions = self.resolve_actions(request);
        let fuse = crate::world_fuse::WorldFuse::try_load(
            &self.root,
            crate::world_fuse::FuseKind::Action,
            self.cfg.wan.out_channels,
        )?;
        let mut control = actions.keyboard.clone();
        control.extend_from_slice(&actions.mouse);

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
                f.fuse_into_latents(&lat, &control)?
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
            msg("Matrix-Game: call load_vae() after placing Diffusers `vae/` under --weights")
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
        let r = MatrixGameRequest::for_preset(MatrixGamePreset::Mg3BaseDistilled, "", 0);
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
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_dim]);
        let out = dit.forward(&x, &ts, &enc).unwrap();
        assert_eq!(out.shape[0], 1);
        assert_eq!(out.shape[1], cfg.out_channels);
    }

    #[test]
    fn resolves_auto_actions() {
        let pipe = MatrixGamePipeline::open("/tmp/mg-missing", MatrixGamePreset::Mg2BaseDistilled)
            .unwrap();
        let r = MatrixGameRequest::for_preset(MatrixGamePreset::Mg2BaseDistilled, "", 7);
        let a = pipe.resolve_actions(&r);
        assert_eq!(a.num_frames, r.num_frames);
        assert_eq!(a.keyboard_dim, 4);
        assert!(a.keyboard.iter().any(|&x| x > 0.0));
    }
}
