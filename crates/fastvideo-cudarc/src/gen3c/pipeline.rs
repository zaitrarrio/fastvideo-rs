//! GEN3C generate: T5 → Cosmos DiT (GEN3C channels + 3D warp buffers) → Wan VAE.
//!
//! Host warp math is online; MoGe depth is an optional external hook
//! (`depth_path` or [`synthetic_depth`]).

use std::path::{Path, PathBuf};

use fastvideo_models::gen3c::{
    default_intrinsics, generate_camera_trajectory, identity4, pack_rgb_buffers, pack_vae_buffers,
    render_trajectory, synthetic_depth, CameraRotation, Gen3CPreset, Gen3CSchedule,
    Gen3CTransformerConfig, TrajectoryType,
};
use fastvideo_models::wan::WanVaeConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::cosmos::text;
use crate::cosmos::transformer::CosmosTransformer;
use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::vae::AutoencoderKlWan;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// FastVideo GEN3C default negative prompt (matches Cosmos wording).
pub const DEFAULT_NEGATIVE_PROMPT: &str = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, \
over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, \
underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, \
jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special \
effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and \
flickering. Overall, the video is of poor quality.";

#[derive(Debug, Clone)]
pub struct Gen3CRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub fps: u32,
    pub preset: Gen3CPreset,
    pub image_path: Option<PathBuf>,
    pub trajectory: TrajectoryType,
    pub movement_distance: f32,
    pub camera_rotation: CameraRotation,
    pub sigma_conditioning: f64,
    /// Optional MoGe depth map (`H*W` f32 raw or `.npy` not required — grayscale PNG).
    /// When absent, uses constant [`synthetic_depth`] at `center_depth`.
    pub depth_path: Option<PathBuf>,
    pub center_depth: f32,
    /// When true, pack RGB warp stub buffers; when VAE is loaded, prefer VAE-encoded warps.
    pub use_geometry_conditioning: bool,
}

impl Gen3CRequest {
    pub fn cosmos_7b(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = Gen3CPreset::Cosmos7b;
        Self {
            prompt: prompt.into(),
            negative_prompt: String::new(),
            seed,
            height: preset.default_height(),
            width: preset.default_width(),
            num_frames: preset.default_num_frames(),
            num_steps: preset.default_steps(),
            guidance_scale: 1.0,
            fps: preset.default_fps(),
            preset,
            image_path: None,
            trajectory: TrajectoryType::Left,
            movement_distance: 0.3,
            camera_rotation: CameraRotation::CenterFacing,
            sigma_conditioning: preset.sigma_conditional(),
            depth_path: None,
            center_depth: 1.0,
            use_geometry_conditioning: true,
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

pub struct Gen3CPipeline {
    pub root: PathBuf,
    pub gen_cfg: Gen3CTransformerConfig,
    pub preset: Gen3CPreset,
    pub dit: Option<CosmosTransformer>,
    pub vae: Option<AutoencoderKlWan>,
}

impl Gen3CPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: Gen3CPreset) -> Result<Self> {
        let gen_cfg = match preset {
            Gen3CPreset::Cosmos7b => Gen3CTransformerConfig::cosmos_7b(),
        };
        Ok(Self {
            root: root.into(),
            gen_cfg,
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map =
            WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(CosmosTransformer::load(self.gen_cfg.to_cosmos(), &map)?);
        Ok(())
    }

    /// Tiny zero DiT for host unit tests (no Hub weights).
    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.gen_cfg = Gen3CTransformerConfig::tiny();
        self.dit = Some(CosmosTransformer::zeros(self.gen_cfg.to_cosmos())?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        let mut cfg = WanVaeConfig::wan_2_1();
        cfg.load_encoder = true;
        self.vae = Some(AutoencoderKlWan::load(cfg, &map).map_err(|e| msg(e.to_string()))?);
        Ok(())
    }

    pub fn schedule(&self, steps: usize) -> Gen3CSchedule {
        Gen3CSchedule::new(steps, self.preset)
    }

    fn encode_text(&self, prompt: &str) -> Result<CudaTensor> {
        let te = self.root.join("text_encoder");
        let tok = self.root.join("tokenizer").join("tokenizer.json");
        if te.is_dir() && tok.is_file() {
            return text::encode_prompt(&self.root, prompt, self.gen_cfg.text_embed_dim)
                .map_err(|e| msg(e.to_string()));
        }
        if te.is_dir() {
            return Err(msg(
                "gen3c text_encoder present but tokenizer/tokenizer.json missing",
            ));
        }
        Ok(CudaTensor::zeros(&[1, 16, self.gen_cfg.text_embed_dim]))
    }

    /// First-frame VAE encode → condition mask; build 3D warp buffers from depth+trajectory.
    pub(crate) fn prepare_conditioning(
        &self,
        request: &Gen3CRequest,
        lt: usize,
        lh: usize,
        lw: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize)> {
        let c = self.gen_cfg.latent_channels;
        let spatial = lt * lh * lw;
        let mut cond_latents = vec![0f32; c * spatial];
        let mut cond_mask = vec![0f32; spatial];
        let buf_ch = self.gen_cfg.buffer_channels();
        let mut buffers = vec![0f32; buf_ch * spatial];

        let Some(path) = request.image_path.as_ref() else {
            return Ok((cond_latents, cond_mask, buffers, 0));
        };

        let video = load_rgb_frame(path, request.height, request.width)?;
        let rgb = video.host_cow()?;
        // video is [1,3,1,H,W]
        let rgb_chw: Vec<f32> = rgb[0..3 * request.height * request.width].to_vec();

        if let Some(vae) = self.vae.as_ref() {
            let encoded = vae.encode_video(&video).map_err(|e| msg(e.to_string()))?;
            let encoded = vae
                .normalize_latents(&encoded)
                .map_err(|e| msg(e.to_string()))?;
            let encoded = encoded
                .try_mul_scalar(self.preset.sigma_data() as f32)
                .map_err(|e| msg(e.to_string()))?;
            let eh = encoded.host_cow()?;
            let [_, ec, et, ey, ex] = match encoded.shape[..] {
                [1, ec, et, ey, ex] => [1, ec, et, ey, ex],
                _ => {
                    return Err(msg(format!(
                        "gen3c cond encode shape {:?} want [1,C,T,H,W]",
                        encoded.shape
                    )))
                }
            };
            if ec != c || ey != lh || ex != lw {
                return Err(msg(format!(
                    "gen3c cond latent [{ec},{et},{ey},{ex}] vs want [{c},*,{lh},{lw}]"
                )));
            }
            let num_cond_latent = 1usize.min(lt).min(et);
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
        } else {
            // No VAE: still mark first latent frame conditioned (zeros) so mask path works.
            for y in 0..lh {
                for x in 0..lw {
                    cond_mask[y * lw + x] = 1.0;
                }
            }
        }

        if request.use_geometry_conditioning {
            let depth = if let Some(dp) = request.depth_path.as_ref() {
                load_depth_map(dp, request.height, request.width)?
            } else {
                synthetic_depth(request.height, request.width, request.center_depth)
            };
            let (w2cs, ks) = generate_camera_trajectory(
                request.trajectory,
                request.num_frames.min(lt.max(2)),
                request.movement_distance,
                request.camera_rotation,
                request.center_depth,
                request.height,
                request.width,
            );
            let src_w2c = identity4();
            let src_k = default_intrinsics(request.height, request.width);
            // Subsample trajectory to frame_buffer_max warps for the cache.
            let n_buf = self.gen_cfg.frame_buffer_max;
            let step = (w2cs.len() / n_buf).max(1);
            let mut sel_w2c = Vec::new();
            let mut sel_k = Vec::new();
            for i in 0..n_buf {
                let idx = (i * step).min(w2cs.len() - 1);
                sel_w2c.push(w2cs[idx]);
                sel_k.push(ks[idx]);
            }
            let warps = render_trajectory(
                &rgb_chw,
                &depth,
                request.height,
                request.width,
                &src_w2c,
                &src_k,
                &sel_w2c,
                &sel_k,
            );

            if let Some(vae) = self.vae.as_ref() {
                let mut encoded_bufs = Vec::new();
                let mut mask_bufs = Vec::new();
                for (warp_rgb, warp_mask) in &warps {
                    let tensor = CudaTensor::from_vec(
                        warp_rgb.clone(),
                        vec![1, 3, 1, request.height, request.width],
                    )?;
                    let enc = vae.encode_video(&tensor).map_err(|e| msg(e.to_string()))?;
                    let enc = vae
                        .normalize_latents(&enc)
                        .map_err(|e| msg(e.to_string()))?;
                    let eh = enc.host_cow()?;
                    let mut flat = vec![0f32; c * spatial];
                    let [_, ec, et, ey, ex] = match enc.shape[..] {
                        [1, ec, et, ey, ex] => [1, ec, et, ey, ex],
                        _ => return Err(msg(format!("gen3c warp encode {:?}", enc.shape))),
                    };
                    let n_t = et.min(lt);
                    for ch in 0..c.min(ec) {
                        for ti in 0..n_t {
                            for y in 0..lh.min(ey) {
                                for x in 0..lw.min(ex) {
                                    let src = ((ch * et + ti) * ey + y) * ex + x;
                                    let dst = ti * lh * lw + y * lw + x;
                                    flat[ch * spatial + dst] = eh[src];
                                }
                            }
                        }
                    }
                    encoded_bufs.push(flat);
                    // Downsample mask to latent grid.
                    let mut mlat = vec![0f32; spatial];
                    for ti in 0..lt {
                        for y in 0..lh {
                            for x in 0..lw {
                                let sy = (((y as f32 + 0.5) * request.height as f32 / lh as f32)
                                    as usize)
                                    .min(request.height - 1);
                                let sx = (((x as f32 + 0.5) * request.width as f32 / lw as f32)
                                    as usize)
                                    .min(request.width - 1);
                                mlat[ti * lh * lw + y * lw + x] =
                                    warp_mask[sy * request.width + sx];
                            }
                        }
                    }
                    mask_bufs.push(mlat);
                }
                buffers = pack_vae_buffers(
                    &encoded_bufs,
                    &mask_bufs,
                    c,
                    lt,
                    lh,
                    lw,
                    self.gen_cfg.frame_buffer_max,
                    self.gen_cfg.channels_per_buffer,
                );
            } else {
                buffers = pack_rgb_buffers(
                    &warps,
                    request.height,
                    request.width,
                    lt,
                    lh,
                    lw,
                    self.gen_cfg.frame_buffer_max,
                    self.gen_cfg.channels_per_buffer,
                );
            }
        }

        let num_cond = if request.image_path.is_some() { 1 } else { 0 };
        Ok((cond_latents, cond_mask, buffers, num_cond))
    }

    /// Pack noise + mask + 3D warp buffers into DiT input channels.
    fn pack_step(
        &self,
        sample: &[f32],
        cond_latents: &[f32],
        cond_mask: &[f32],
        buffers: &[f32],
        c_in: f64,
        current_t: f32,
        t_cond: f32,
        lt: usize,
        lh: usize,
        lw: usize,
    ) -> Result<(CudaTensor, Vec<f32>)> {
        let c = self.gen_cfg.latent_channels;
        let spatial = lt * lh * lw;
        let in_ch = self.gen_cfg.in_channels();
        let mut packed = vec![0f32; in_ch * spatial];
        let mut frame_ts = vec![current_t; lt];
        let buf_ch = self.gen_cfg.buffer_channels();
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
            for ch in 0..buf_ch {
                let src = ch * spatial + j;
                if src < buffers.len() {
                    packed[(c + 1 + ch) * spatial + j] = buffers[src];
                }
            }
        }
        let lat = CudaTensor::from_vec(packed, vec![1, in_ch, lt, lh, lw])?;
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
            return Err(msg(format!("gen3c pred {} vs {n}", ph.len())));
        }
        let mut denoised = vec![0f32; n];
        for j in 0..n {
            denoised[j] = c_skip as f32 * sample[j] + c_out as f32 * ph[j];
        }
        for j in 0..spatial {
            if cond_mask[j] > 0.5 {
                for ch in 0..c {
                    let idx = ch * spatial + j;
                    denoised[idx] = cond_latents[idx];
                }
            }
        }
        Ok(denoised)
    }

    pub fn generate(&self, request: &Gen3CRequest, out_dir: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("GEN3C: call load_dit() after placing Diffusers `transformer/` under --weights")
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
        let c = self.gen_cfg.latent_channels;
        let spatial = lt * lh * lw;

        let (cond_latents, cond_mask, buffers, _n_cond) =
            self.prepare_conditioning(request, lt, lh, lw)?;

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
            let (c_in, c_skip, c_out) = Gen3CSchedule::edm_coeffs(sigma);
            let current_t = (sigma / (sigma + 1.0)) as f32;
            let (lat, frame_ts) = self.pack_step(
                &sample,
                &cond_latents,
                &cond_mask,
                &buffers,
                c_in,
                current_t,
                t_cond,
                lt,
                lh,
                lw,
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
            msg("GEN3C: call load_vae() after placing Diffusers `vae/` under --weights")
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
                    "gen3c decode shape {:?} want [1,3,T,H,W]",
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

fn load_depth_map(path: &Path, height: usize, width: usize) -> Result<Vec<f32>> {
    let img = image::open(path)
        .map_err(|e| msg(format!("open depth {}: {e}", path.display())))?
        .into_luma8();
    let img = image::imageops::resize(
        &img,
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut depth = vec![0f32; height * width];
    for y in 0..height {
        for x in 0..width {
            // Map 0..255 → 0.1..10.0 meters (MoGe-scale stand-in).
            let v = f32::from(img.get_pixel(x as u32, y as u32)[0]) / 255.0;
            depth[y * width + x] = 0.1 + v * 9.9;
        }
    }
    Ok(depth)
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
        let r = Gen3CRequest::cosmos_7b("a scene", 0);
        assert_eq!(r.height, 704);
        assert_eq!(r.width, 1280);
        assert_eq!(r.num_frames, 121);
        assert!(!r.do_classifier_free_guidance());
        assert_eq!(r.trajectory, TrajectoryType::Left);
    }

    #[test]
    fn tiny_dit_forward_shapes() {
        let mut pipe = Gen3CPipeline::open("/tmp/gen3c-missing", Gen3CPreset::Cosmos7b).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        let dit = pipe.dit.as_ref().unwrap();
        let cfg = &pipe.gen_cfg;
        let x = CudaTensor::zeros(&[1, cfg.in_channels(), 2, 4, 4]);
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_embed_dim]);
        let out = dit.forward(&x, &enc, &[0.5, 0.5], Some(24.0)).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
    }

    #[test]
    fn warp_buffers_without_image() {
        let pipe = Gen3CPipeline::open("/tmp/gen3c-missing", Gen3CPreset::Cosmos7b).unwrap();
        let mut r = Gen3CRequest::cosmos_7b("a scene", 0);
        r.num_frames = 9;
        r.height = 64;
        r.width = 64;
        let (cond, mask, buf, n) = pipe.prepare_conditioning(&r, 3, 8, 8).unwrap();
        assert_eq!(n, 0);
        assert_eq!(cond.len(), pipe.gen_cfg.latent_channels * 3 * 8 * 8);
        assert!(mask.iter().all(|&m| m == 0.0));
        assert_eq!(buf.len(), pipe.gen_cfg.buffer_channels() * 3 * 8 * 8);
    }
}
