//! SANA-Video text-to-video: Gemma-2 prompt encode → 50 flow DPM-Solver++
//! steps of the linear-attention DiT with classifier-free guidance → Wan 2.1
//! VAE decode. Spec: docs/ports/sana-video.md.
//!
//! The Sol-Engine optimized arm ([`SanaOptimizations`]) adds EasyCache (the
//! guided output residual `noise_pred − x` is reused while the accumulated
//! drift estimate stays below the threshold), the merged QKV projection and
//! bf16 linear-attention operands.

use std::path::{Path, PathBuf};

use fastvideo_models::sana_video::sol::{EASYCACHE_COOLDOWN_STEPS, EASYCACHE_RETAIN_STEPS};
use fastvideo_models::sana_video::text::{tokenize_pair, with_motion_score};
use fastvideo_models::sana_video::{
    Gemma2TextConfig, SanaDpmSolver, SanaOptimizations, SanaVideoTransformerConfig,
    SANA_FLOW_SHIFT, SANA_MAX_SEQUENCE_LENGTH,
};
use fastvideo_models::wan::sol_cache::EasyCache;
use fastvideo_models::wan::WanVaeConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::llm::DecoderConfig;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::sol_cache::{mean_abs, mean_abs_delta};
use crate::wan::tensor::CudaTensor;
use crate::wan::vae::AutoencoderKlWan;
use crate::wan::weights::WeightMap;

use super::text::{encode_layout, gemma2_decoder};
use super::transformer::SanaVideoTransformer;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| msg(format!("{}: {e}", path.display())))
}

#[derive(Debug, Clone)]
pub struct SanaVideoRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub flow_shift: f64,
    /// Appends `" motion score: {m}."` (the model card's recipe; off by default).
    pub motion_score: Option<u32>,
    /// Raw little-endian f32 `[16, T, H/8, W/8]` initial noise (parity runs
    /// inject the reference's `randn`); `None` draws from `seed`.
    pub noise_path: Option<PathBuf>,
}

impl SanaVideoRequest {
    /// The Sol-Engine speedup canvas: 832x480, 81 frames, 50 steps, cfg 6.
    pub fn published(prompt: impl Into<String>, seed: u64) -> Self {
        use fastvideo_models::sana_video::sol::*;
        Self {
            prompt: prompt.into(),
            negative_prompt: String::new(),
            seed,
            height: PUBLISHED_HEIGHT,
            width: PUBLISHED_WIDTH,
            num_frames: PUBLISHED_FRAMES,
            num_steps: PUBLISHED_STEPS,
            guidance_scale: PUBLISHED_GUIDANCE,
            flow_shift: SANA_FLOW_SHIFT,
            motion_score: None,
            noise_path: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SanaVideoTimings {
    pub text_s: f64,
    pub denoise_s: f64,
    pub step_s: Vec<f64>,
    pub decode_s: f64,
    pub write_s: f64,
    /// Steps whose transformer calls EasyCache skipped.
    pub reused_steps: usize,
}

#[derive(Debug, Clone)]
pub struct SanaVideoOutput {
    pub frames: usize,
    pub frame_paths: Vec<String>,
    pub timings: SanaVideoTimings,
}

pub struct SanaVideoPipeline {
    pub root: PathBuf,
    pub opt: SanaOptimizations,
    dit: SanaVideoTransformer,
    vae: AutoencoderKlWan,
    text_map: WeightMap,
    text_cfg: DecoderConfig,
}

impl SanaVideoPipeline {
    /// Open a Diffusers tree. `vae_dir` overrides `root/vae` (the file is
    /// byte-identical to the Wan 2.1 VAE already on the volume).
    pub fn load(
        root: impl Into<PathBuf>,
        opt: SanaOptimizations,
        vae_dir: Option<&Path>,
    ) -> Result<Self> {
        let root = root.into();
        let dit_cfg =
            SanaVideoTransformerConfig::from_json(&read(&root.join("transformer/config.json"))?)
                .map_err(msg)?;
        let dit_map =
            WeightMap::open(&root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        let dit = SanaVideoTransformer::load(dit_cfg, opt, &dit_map)?;
        let vae_root = vae_dir.map_or_else(|| root.join("vae"), Path::to_path_buf);
        let mut vae_cfg = WanVaeConfig::from_json_str(&read(&vae_root.join("config.json"))?)
            .map_err(msg)?;
        vae_cfg.load_encoder = false;
        let vae_map = WeightMap::open(&vae_root).map_err(|e| msg(e.to_string()))?;
        let vae = AutoencoderKlWan::load(vae_cfg, &vae_map)?;
        let text_cfg = gemma2_decoder(
            &Gemma2TextConfig::from_json(&read(&root.join("text_encoder/config.json"))?)
                .map_err(msg)?,
        );
        let text_map =
            WeightMap::open(&root.join("text_encoder")).map_err(|e| msg(e.to_string()))?;
        Ok(Self {
            root,
            opt,
            dit,
            vae,
            text_map,
            text_cfg,
        })
    }

    pub fn transformer(&self) -> &SanaVideoTransformer {
        &self.dit
    }

    /// `(positive, negative)` caption tokens after the DiT's caption projection.
    pub fn encode(&self, request: &SanaVideoRequest) -> Result<(CudaTensor, CudaTensor)> {
        let prompt = with_motion_score(&request.prompt, request.motion_score);
        let (pos, neg) = tokenize_pair(
            &self.root,
            &prompt,
            &request.negative_prompt,
            SANA_MAX_SEQUENCE_LENGTH,
        )
        .map_err(msg)?;
        let p = encode_layout(&self.text_map, &self.text_cfg, &pos)?;
        let n = encode_layout(&self.text_map, &self.text_cfg, &neg)?;
        Ok((self.dit.project_caption(&p)?, self.dit.project_caption(&n)?))
    }

    fn initial_noise(&self, request: &SanaVideoRequest, shape: &[usize]) -> Result<CudaTensor> {
        let n: usize = shape.iter().product();
        let values = match &request.noise_path {
            Some(p) => {
                let bytes = std::fs::read(p).map_err(|e| msg(format!("{}: {e}", p.display())))?;
                if bytes.len() != 4 * n {
                    return Err(msg(format!(
                        "noise {}: {} bytes, want {} for {shape:?}",
                        p.display(),
                        bytes.len(),
                        4 * n
                    )));
                }
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            }
            None => {
                let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
                (0..n).map(|_| StandardNormal.sample(&mut rng)).collect()
            }
        };
        Ok(CudaTensor::from_vec(values, shape.to_vec())?.to_device()?)
    }

    /// Denoise to latents `[1, 16, T, H/8, W/8]` (DiT space).
    pub fn denoise(
        &self,
        request: &SanaVideoRequest,
        captions: &(CudaTensor, CudaTensor),
        timings: &mut SanaVideoTimings,
    ) -> Result<CudaTensor> {
        let cfg = self.dit.config();
        let (sc, tc) = (
            self.vae.spatial_compression(),
            self.vae.cfg.temporal_compression(),
        );
        if !request.height.is_multiple_of(sc * cfg.patch_size[1])
            || !request.width.is_multiple_of(sc * cfg.patch_size[2])
        {
            return Err(msg(format!(
                "SANA-Video: {}x{} must be a multiple of {}",
                request.width,
                request.height,
                sc * cfg.patch_size[1]
            )));
        }
        let lt = (request.num_frames.saturating_sub(1)) / tc + 1;
        let (lh, lw) = (request.height / sc, request.width / sc);
        let shape = [1, cfg.in_channels, lt, lh, lw];
        let mut x = self.initial_noise(request, &shape)?;
        let rope = self.dit.rope(lt, lh, lw)?;
        let mut solver = SanaDpmSolver::new(request.num_steps, request.flow_shift).map_err(msg)?;
        let timesteps = solver.timesteps().to_vec();
        let mut cache = match self.opt.easycache {
            Some(th) => Some(
                EasyCache::new(
                    timesteps.len(),
                    th,
                    EASYCACHE_RETAIN_STEPS,
                    EASYCACHE_COOLDOWN_STEPS,
                )
                .map_err(msg)?,
            ),
            None => None,
        };
        // EasyCache state: previous step input, last full (input, output),
        // and the reusable residual `output - input`.
        let mut prev_input: Option<CudaTensor> = None;
        let mut last_full: Option<(CudaTensor, CudaTensor)> = None;
        let mut residual: Option<CudaTensor> = None;
        let mut prev_x0: Option<CudaTensor> = None;
        let g = request.guidance_scale;
        let denoise_t = std::time::Instant::now();
        for (step, &t) in timesteps.iter().enumerate() {
            let step_t = std::time::Instant::now();
            let compute = match cache.as_mut() {
                None => true,
                Some(c) => {
                    let (dx, onorm) = if c.needs_cond_signal(step) {
                        let (pi, (_, lo)) = (
                            prev_input.as_ref().expect("signal implies a previous step"),
                            last_full.as_ref().expect("signal implies a full step"),
                        );
                        (mean_abs_delta(&x, pi)?, mean_abs(lo)?)
                    } else {
                        (0.0, 0.0)
                    };
                    c.decide_cond(step, dx, onorm).compute || residual.is_none()
                }
            };
            let noise_pred = if compute {
                let tt = t as f32;
                let cond = self.dit.forward(&x, &captions.0, tt, &rope)?;
                let out = if g > 1.0 {
                    let uncond = self.dit.forward(&x, &captions.1, tt, &rope)?;
                    CudaTensor::lincomb(&[(1.0 - g, &uncond), (g, &cond)])?
                } else {
                    cond
                };
                if let Some(c) = cache.as_mut() {
                    let (din, dout) = match &last_full {
                        Some((li, lo)) => (mean_abs_delta(&x, li)?, mean_abs_delta(&out, lo)?),
                        None => (0.0, 0.0),
                    };
                    c.note_cond_computed(din, dout);
                    residual = Some(out.sub(&x)?);
                    last_full = Some((x.clone(), out.clone()));
                }
                out
            } else {
                timings.reused_steps += 1;
                x.add(residual.as_ref().expect("reuse implies a residual"))?
            };
            if cache.is_some() {
                prev_input = Some(x.clone());
            }
            let plan = solver.plan_step().map_err(msg)?;
            let x0 = CudaTensor::lincomb(&[(1.0, &x), (-plan.sigma, &noise_pred)])?;
            x = match (&prev_x0, plan.order) {
                (Some(p), 2) => CudaTensor::lincomb(&[
                    (plan.sample_coef, &x),
                    (plan.x0_coef, &x0),
                    (plan.prev_x0_coef, p),
                ])?,
                _ => CudaTensor::lincomb(&[(plan.sample_coef, &x), (plan.x0_coef, &x0)])?,
            };
            prev_x0 = Some(x0);
            timings.step_s.push(step_t.elapsed().as_secs_f64());
        }
        timings.denoise_s = denoise_t.elapsed().as_secs_f64();
        Ok(x)
    }

    /// Wan VAE decode of DiT-space latents → `[frames, 3, H, W]` in `[-1, 1]`.
    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let z = self.vae.scale_latents(latents)?;
        let video = self.vae.decode(&z)?;
        let [1, 3, f, h, w] = video.shape[..] else {
            return Err(msg(format!("sana decode: {:?}", video.shape)));
        };
        Ok(video.reshape(vec![3, f, h, w])?.permute(&[1, 0, 2, 3])?)
    }

    pub fn generate(&self, request: &SanaVideoRequest, out_dir: &Path) -> Result<SanaVideoOutput> {
        let mut timings = SanaVideoTimings::default();
        let text_t = std::time::Instant::now();
        let captions = self.encode(request)?;
        timings.text_s = text_t.elapsed().as_secs_f64();
        let latents = self.denoise(request, &captions, &mut timings)?;
        let decode_t = std::time::Instant::now();
        let frames = self.decode(&latents)?;
        timings.decode_s = decode_t.elapsed().as_secs_f64();
        let write_t = std::time::Instant::now();
        let frame_paths = write_frames(&frames, out_dir)?;
        timings.write_s = write_t.elapsed().as_secs_f64();
        Ok(SanaVideoOutput {
            frames: frame_paths.len(),
            frame_paths,
            timings,
        })
    }
}

/// `[frames, 3, H, W]` in `[-1, 1]` → `out_dir/frame_%05d.png`.
pub fn write_frames(frames: &CudaTensor, out_dir: &Path) -> Result<Vec<String>> {
    let [f, 3, h, w] = frames.shape[..] else {
        return Err(msg(format!("write_frames: {:?}", frames.shape)));
    };
    let rgb = crate::wan::pipeline::frames_to_rgb8(frames)?;
    std::fs::create_dir_all(out_dir).map_err(|e| msg(e.to_string()))?;
    let mut paths = Vec::with_capacity(f);
    for i in 0..f {
        let path = out_dir.join(format!("frame_{i:05}.png"));
        image::save_buffer(
            &path,
            &rgb[i * 3 * h * w..(i + 1) * 3 * h * w],
            w as u32,
            h as u32,
            image::ColorType::Rgb8,
        )
        .map_err(|e| msg(e.to_string()))?;
        paths.push(path.display().to_string());
    }
    Ok(paths)
}
