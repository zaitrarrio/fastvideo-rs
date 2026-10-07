//! LingBot-Video generate: Qwen3-VL → base DiT (FlowUniPC, CFG) → Wan VAE
//! decode → optional 1080p refiner (bicubic up → Wan VAE encode → noised to
//! `t_thresh` → refiner DiT on the low-noise tail) → decode → mp4/PNG.
//!
//! Mirrors `runner.py` + `pipeline_lingbot_video.py` of the sol-engine
//! vendored LingBot (`models/lingbot_video/baseline`). Residency on one GPU:
//! [`Residency::Both`] keeps base and refiner DiTs loaded (≈120 GB bf16: a
//! B200), [`Residency::Swap`] loads the refiner after dropping the base (one
//! 60 GB DiT at a time: a 96 GB RTX PRO 6000).

use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_models::lingbot::sol::{
    EasyCache, EasyCacheConfig, PisaPolicy, SolArm, OFFICIAL_FPS,
};
use fastvideo_models::lingbot::{
    base_unipc, refiner::{self as host_refiner}, refiner_unipc, transformer_timestep, LingBotPreset,
    LingBotTransformerConfig, DEFAULT_NEGATIVE_PROMPT,
};
use fastvideo_models::schedulers::FlowUniPCMultistepScheduler;
use fastvideo_models::wan::WanVaeConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result, VideoWriter};
use crate::wan::tensor::CudaTensor;
use crate::wan::vae::AutoencoderKlWan;
use crate::wan::vae21_encode::Wan21ChunkedEncoder;
use crate::wan::weights::WeightMap;

use super::text::LingBotTextEncoder;
use super::transformer::{AttnRoute, LingBotTransformer};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Return the pool's cached free memory to the driver between phases
/// ([`crate::wan::device::trim_pool`]).
fn trim() -> Result<()> {
    crate::wan::device::trim_pool().map_err(terr)
}

fn terr(e: impl std::fmt::Display) -> PipelineError {
    PipelineError::Message(e.to_string())
}

/// Refiner stage settings (`--refiner_*` of `runner.py`).
#[derive(Debug, Clone)]
pub struct RefinerRequest {
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub guidance_scale: f32,
    pub shift: f64,
    pub t_thresh: f64,
    pub tail_steps: usize,
}

impl RefinerRequest {
    pub fn official() -> Self {
        use fastvideo_models::lingbot::sol::*;
        Self {
            height: REFINER_HEIGHT,
            width: REFINER_WIDTH,
            steps: REFINER_STEPS,
            guidance_scale: REFINER_GUIDANCE,
            shift: REFINER_SHIFT,
            t_thresh: REFINER_T_THRESH,
            tail_steps: REFINER_SIGMA_TAIL_STEPS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LingBotRequest {
    pub prompt: String,
    /// `None`: `DEFAULT_NEGATIVE_PROMPT` (the official base run).
    pub negative_prompt: Option<String>,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub shift: f64,
    pub fps: u32,
    pub preset: LingBotPreset,
    pub refiner: Option<RefinerRequest>,
    pub sol: SolArm,
}

impl LingBotRequest {
    pub fn dense_1_3b(prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            seed,
            height: 480,
            width: 832,
            num_frames: 81,
            num_steps: 40,
            guidance_scale: 1.0,
            shift: 3.0,
            fps: OFFICIAL_FPS,
            preset: LingBotPreset::Dense13b,
            refiner: None,
            sol: SolArm::BASELINE,
        }
    }

    /// The `models/lingbot_video.toml` contract: 832×480×121, 40 steps,
    /// guidance 3, shift 3, then the 1920×1088 refiner.
    pub fn official(prompt: impl Into<String>) -> Self {
        use fastvideo_models::lingbot::sol::*;
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            seed: OFFICIAL_SEED,
            height: OFFICIAL_HEIGHT,
            width: OFFICIAL_WIDTH,
            num_frames: OFFICIAL_FRAMES,
            num_steps: OFFICIAL_STEPS,
            guidance_scale: OFFICIAL_GUIDANCE,
            shift: OFFICIAL_SHIFT,
            fps: OFFICIAL_FPS,
            preset: LingBotPreset::Moe30b,
            refiner: Some(RefinerRequest::official()),
            sol: SolArm::BASELINE,
        }
    }

    fn check(&self) -> Result<()> {
        if self.num_frames != 1 && (self.num_frames - 1) % 4 != 0 {
            return Err(msg(format!("num_frames must be 1 or 4n+1, got {}", self.num_frames)));
        }
        if self.height % 16 != 0 || self.width % 16 != 0 {
            return Err(msg(format!("height/width must be multiples of 16: {}x{}", self.height, self.width)));
        }
        if let Some(r) = &self.refiner {
            if r.height % 16 != 0 || r.width % 16 != 0 {
                return Err(msg("refiner canvas must be a multiple of 16"));
            }
        }
        Ok(())
    }
}

/// Where the two DiTs live (`FASTVIDEO_LINGBOT_RESIDENCY=both|swap`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    Both,
    Swap,
}

impl Residency {
    pub fn from_env() -> Self {
        match std::env::var("FASTVIDEO_LINGBOT_RESIDENCY").ok().as_deref().map(str::trim) {
            Some("both") | Some("resident") => Self::Both,
            _ => Self::Swap,
        }
    }
}

/// Wall times of one generate, seconds (`lingbot_timing.json`).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LingBotTiming {
    pub arm: String,
    pub residency: String,
    pub load_text_s: f64,
    pub text_encode_s: f64,
    pub load_base_s: f64,
    pub base_denoise_s: f64,
    pub base_decode_s: f64,
    pub base_export_s: f64,
    pub load_refiner_s: f64,
    pub refiner_prepare_s: f64,
    pub refiner_denoise_s: f64,
    pub refiner_decode_s: f64,
    pub refiner_export_s: f64,
    /// Request wall without weight loads: text encode + both stages + exports,
    /// what the published interval times (both models already resident).
    pub request_s: f64,
    pub base_steps_computed: usize,
    pub base_steps_reused: usize,
    pub refiner_steps: usize,
    pub refiner_steps_computed: usize,
    pub refiner_steps_reused: usize,
    pub refiner_sparse_steps: usize,
}

pub struct LingBotPipeline {
    pub root: PathBuf,
    pub dit_cfg: LingBotTransformerConfig,
    pub preset: LingBotPreset,
    pub dit: Option<LingBotTransformer>,
    pub refiner: Option<LingBotTransformer>,
    pub text: Option<LingBotTextEncoder>,
    pub vae: Option<AutoencoderKlWan>,
    pub vae_encoder: Option<Wan21ChunkedEncoder>,
    pub residency: Residency,
}

fn dit_config(dir: &Path, preset: LingBotPreset) -> Result<LingBotTransformerConfig> {
    let path = dir.join("config.json");
    if path.is_file() {
        let text = std::fs::read_to_string(&path).map_err(terr)?;
        return LingBotTransformerConfig::from_hub_json(&text).map_err(msg);
    }
    Ok(LingBotTransformerConfig::for_preset(preset))
}

impl LingBotPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: LingBotPreset) -> Result<Self> {
        let root = root.into();
        let dit_cfg = dit_config(&root.join("transformer"), preset)?;
        Ok(Self {
            root,
            dit_cfg,
            preset,
            dit: None,
            refiner: None,
            text: None,
            vae: None,
            vae_encoder: None,
            residency: Residency::from_env(),
        })
    }

    pub fn load_text(&mut self) -> Result<()> {
        self.text = Some(LingBotTextEncoder::load(&self.root, self.dit_cfg.text_dim).map_err(terr)?);
        Ok(())
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(terr)?;
        self.dit = Some(LingBotTransformer::load(self.dit_cfg.clone(), &map)?);
        Ok(())
    }

    pub fn load_refiner(&mut self) -> Result<()> {
        let dir = self.root.join("refiner");
        let cfg = dit_config(&dir, self.preset)?;
        let map = WeightMap::open(&dir).map_err(terr)?;
        self.refiner = Some(LingBotTransformer::load(cfg, &map)?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(terr)?;
        let mut cfg = WanVaeConfig::wan_2_1();
        cfg.load_encoder = false;
        self.vae = Some(AutoencoderKlWan::load(cfg, &map).map_err(terr)?);
        Ok(())
    }

    pub fn load_vae_encoder(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(terr)?;
        self.vae_encoder = Some(Wan21ChunkedEncoder::load(&map)?);
        Ok(())
    }

    fn latent_shape(&self, frames: usize, height: usize, width: usize) -> [usize; 5] {
        [1, self.dit_cfg.in_channels, (frames - 1) / 4 + 1, height / 8, width / 8]
    }

    /// Run the request; writes `base/` (and `refined/` with a refiner) under
    /// `out_dir`, plus `lingbot_timing.json`.
    pub fn generate(&mut self, request: &LingBotRequest, out_dir: &Path) -> Result<LingBotTiming> {
        request.check()?;
        let mut timing = LingBotTiming {
            arm: request.sol.label().into(),
            residency: format!("{:?}", self.residency).to_lowercase(),
            ..Default::default()
        };
        std::fs::create_dir_all(out_dir).map_err(terr)?;

        // Swap residency: the previous request left the refiner loaded.
        if self.residency == Residency::Swap && self.refiner.take().is_some() {
            trim()?;
        }
        // Text first: under swap the encoder (≈9 GB) leaves before the DiTs arrive.
        let t = Instant::now();
        if self.text.is_none() {
            self.load_text()?;
        }
        timing.load_text_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let enc = self.text.as_ref().expect("loaded");
        let prompt_h = enc.encode(&request.prompt)?;
        let neg = request.negative_prompt.as_deref().unwrap_or(DEFAULT_NEGATIVE_PROMPT);
        let do_cfg = request.guidance_scale > 1.0;
        let neg_h = if do_cfg { Some(enc.encode(neg)?) } else { None };
        timing.text_encode_s = t.elapsed().as_secs_f64();
        if self.residency == Residency::Swap {
            self.text = None;
            trim()?;
        }

        let t = Instant::now();
        if self.dit.is_none() {
            self.load_dit()?;
        }
        if self.vae.is_none() {
            self.load_vae()?;
        }
        let refine = request.refiner.clone();
        if refine.is_some() {
            if self.vae_encoder.is_none() {
                self.load_vae_encoder()?;
            }
        }
        timing.load_base_s = t.elapsed().as_secs_f64();
        if refine.is_some() && self.residency == Residency::Both && self.refiner.is_none() {
            let t = Instant::now();
            self.load_refiner()?;
            timing.load_refiner_s = t.elapsed().as_secs_f64();
        }

        let request_start = Instant::now();
        let mut loads_inside = 0.0f64;

        // ---- base stage ----
        let shape = self.latent_shape(request.num_frames, request.height, request.width);
        let n: usize = shape.iter().product();
        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let t = Instant::now();
        let base = self.dit.as_ref().expect("loaded");
        let text_c = base.embed_text(&prompt_h)?;
        let text_u = match &neg_h {
            Some(h) => Some(base.embed_text(h)?),
            None => None,
        };
        let mut sched = base_unipc(request.num_steps, request.shift);
        let easy = request
            .sol
            .easycache
            .then(|| EasyCacheConfig::for_schedule(sched.inference_timesteps().len()));
        let (latents, stats) = denoise(
            base,
            &mut sched,
            noise,
            shape,
            &text_c,
            text_u.as_ref(),
            request.guidance_scale,
            easy,
            None,
            "base",
        )?;
        timing.base_denoise_s = t.elapsed().as_secs_f64();
        timing.base_steps_computed = stats.computed;
        timing.base_steps_reused = stats.reused;

        let t = Instant::now();
        let vae = self.vae.as_ref().expect("loaded");
        let base_rgb = decode_rgb(vae, latents, shape)?;
        timing.base_decode_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (bf, bh, bw) = (base_rgb.frames, base_rgb.height, base_rgb.width);
        write_video(&out_dir.join("base"), request.fps, &base_rgb)?;
        timing.base_export_s = t.elapsed().as_secs_f64();

        let Some(refine) = refine else {
            timing.request_s = request_start.elapsed().as_secs_f64() + timing.text_encode_s;
            write_timing(out_dir, &timing)?;
            return Ok(timing);
        };

        // ---- refiner stage ----
        if self.residency == Residency::Swap {
            // The base DiT leaves in hundreds of pieces; the pool keeps them
            // (`wan::device` release threshold) and the refiner's weights did
            // not fit between them: CUDA out of memory right after the base
            // stage on a 96 GB card (sol-bench phase B2). Hand them back first.
            // The 1080p VAE encode below also runs before the refiner arrives.
            self.dit = None;
            trim()?;
        }
        let t = Instant::now();
        let (sample, _, _) = host_refiner::training_frame_budget(bf, f64::from(request.fps), request.fps, 4);
        let idx = host_refiner::training_aligned_indices(bf, sample);
        let up = upsample_rgb(&base_rgb.rgb, &idx, bh, bw, refine.height, refine.width);
        drop(base_rgb);
        let video = CudaTensor::from_vec(up, vec![1, 3, sample, refine.height, refine.width])?;
        let encoder = self.vae_encoder.as_ref().ok_or_else(|| msg("vae encoder not loaded"))?;
        let z = encoder.encode_mean(&video)?;
        drop(video);
        trim()?;
        let vae = self.vae.as_ref().expect("loaded");
        let x_up = vae.normalize_latents(&z).map_err(terr)?.host_cow()?.into_owned();
        let rshape = self.latent_shape(sample, refine.height, refine.width);
        if x_up.len() != rshape.iter().product::<usize>() {
            return Err(msg(format!("refiner latent {} vs {:?}", x_up.len(), rshape)));
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let rnoise: Vec<f32> = (0..x_up.len()).map(|_| StandardNormal.sample(&mut rng)).collect();
        let start = host_refiner::noised_start(&x_up, &rnoise, refine.t_thresh as f32);
        drop((x_up, rnoise));
        timing.refiner_prepare_s = t.elapsed().as_secs_f64();
        if self.residency == Residency::Swap {
            let t = Instant::now();
            self.load_refiner()?;
            timing.load_refiner_s = t.elapsed().as_secs_f64();
            loads_inside += timing.load_refiner_s;
        }

        let t = Instant::now();
        let rdit = self.refiner.as_ref().ok_or_else(|| msg("refiner not loaded"))?;
        let rtext_c = rdit.embed_text(&prompt_h)?;
        // `null_cond_clone_zero` (refiner default): a zero Qwen hidden state.
        let rtext_u = if refine.guidance_scale > 1.0 {
            Some(rdit.embed_text(&CudaTensor::zeros(&prompt_h.shape))?)
        } else {
            None
        };
        let mut rsched = refiner_unipc(refine.steps, refine.shift, refine.t_thresh, refine.tail_steps)
            .map_err(msg)?;
        let rsteps = rsched.inference_timesteps().len();
        let reasy = request.sol.easycache.then(|| EasyCacheConfig::for_schedule(rsteps));
        let pisa = (request.sol.pisa).then(PisaPolicy::published).filter(|p| p.refiner);
        let (rlat, rstats) = denoise(
            rdit,
            &mut rsched,
            start,
            rshape,
            &rtext_c,
            rtext_u.as_ref(),
            refine.guidance_scale,
            reasy,
            pisa.as_ref(),
            "refiner",
        )?;
        timing.refiner_denoise_s = t.elapsed().as_secs_f64();
        timing.refiner_steps = rsteps;
        timing.refiner_steps_computed = rstats.computed;
        timing.refiner_steps_reused = rstats.reused;
        timing.refiner_sparse_steps = rstats.sparse;
        // The 1080p decode (121 frames) does not fit beside the 60 GB
        // refiner on a 96 GB card (phase B3 d1: out of memory after the
        // last refiner step). Under swap the refiner leaves before the
        // decode; the next request loads it again either way.
        drop((rtext_c, rtext_u));
        if self.residency == Residency::Swap {
            self.refiner = None;
        }
        trim()?;

        let t = Instant::now();
        let rgb = decode_rgb(self.vae.as_ref().expect("loaded"), rlat, rshape)?;
        timing.refiner_decode_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        write_video(&out_dir.join("refined"), request.fps, &rgb)?;
        timing.refiner_export_s = t.elapsed().as_secs_f64();
        timing.request_s =
            request_start.elapsed().as_secs_f64() - loads_inside + timing.text_encode_s;
        write_timing(out_dir, &timing)?;
        Ok(timing)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DenoiseStats {
    computed: usize,
    reused: usize,
    sparse: usize,
}

/// One stage of CFG FlowUniPC sampling (`LingBotVideoPipeline.__call__`).
#[allow(clippy::too_many_arguments)]
fn denoise(
    dit: &LingBotTransformer,
    sched: &mut FlowUniPCMultistepScheduler,
    mut latents: Vec<f32>,
    shape: [usize; 5],
    text_c: &CudaTensor,
    text_u: Option<&CudaTensor>,
    guidance: f32,
    easy: Option<EasyCacheConfig>,
    pisa: Option<&PisaPolicy>,
    label: &str,
) -> Result<(Vec<f32>, DenoiseStats)> {
    let ts: Vec<i64> = sched.inference_timesteps_i64().to_vec();
    let n_steps = ts.len();
    let mut cache = easy.map(|c| EasyCache::new(c, n_steps));
    let mut last_v: Option<Vec<f32>> = None;
    let mut stats = DenoiseStats::default();
    for (i, &t) in ts.iter().enumerate() {
        let step_t = Instant::now();
        let reuse = match cache.as_mut() {
            Some(c) => last_v.is_some() && c.reuse(i, &latents),
            None => false,
        };
        let v = if reuse {
            stats.reused += 1;
            last_v.clone().expect("cached")
        } else {
            let timestep = transformer_timestep(t, true);
            let route = AttnRoute {
                pisa: pisa.filter(|p| p.sparse_step(i, n_steps)),
            };
            if route.pisa.is_some() {
                stats.sparse += 1;
            }
            let lat = CudaTensor::from_vec(latents.clone(), shape.to_vec())?;
            let cond = dit.forward(&lat, text_c, timestep, route)?;
            let v = match text_u {
                Some(u) if guidance > 1.0 => {
                    let unc = dit.forward(&lat, u, timestep, route)?;
                    unc.add(&cond.sub(&unc)?.try_mul_scalar(guidance)?)?
                }
                _ => cond,
            };
            let v = v.host_cow()?.into_owned();
            if let Some(c) = cache.as_mut() {
                c.computed(&latents);
            }
            stats.computed += 1;
            last_v = Some(v.clone());
            v
        };
        latents = sched.step(&v, &latents).map_err(msg)?;
        crate::wan::log::info(format_args!(
            "lingbot {label} step {}/{n_steps} t={t}{} {:.2}s",
            i + 1,
            if reuse { " (reused)" } else { "" },
            step_t.elapsed().as_secs_f64()
        ));
    }
    Ok((latents, stats))
}

/// Decoded frames as interleaved rgb24 `[F, H, W, 3]`.
struct Rgb {
    rgb: Vec<u8>,
    frames: usize,
    height: usize,
    width: usize,
}

fn decode_rgb(vae: &AutoencoderKlWan, latents: Vec<f32>, shape: [usize; 5]) -> Result<Rgb> {
    let lat = CudaTensor::from_vec(latents, shape.to_vec())?;
    let scaled = vae.scale_latents(&lat).map_err(terr)?;
    let pixels = vae.decode(&scaled).map_err(terr)?;
    let [_, _, f, h, w] = match pixels.shape[..] {
        [1, 3, f, h, w] => [1, 3, f, h, w],
        _ => return Err(msg(format!("lingbot decode shape {:?}", pixels.shape))),
    };
    let by_frame = pixels.permute(&[0, 2, 1, 3, 4])?.reshape(vec![f, 3, h, w])?.clamp(-1.0, 1.0);
    Ok(Rgb {
        rgb: frames_to_rgb8(&by_frame)?,
        frames: f,
        height: h,
        width: w,
    })
}

/// u8 frames (what the mp4 holds) → `[-1, 1]` `[3, F, oh, ow]` after the
/// reference's bicubic resize + clamp.
fn upsample_rgb(rgb: &[u8], idx: &[usize], h: usize, w: usize, oh: usize, ow: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let plane = oh * ow;
    let per_frame: Vec<Vec<f32>> = idx
        .par_iter()
        .map(|&fi| {
            let src = &rgb[fi * h * w * 3..(fi + 1) * h * w * 3];
            let mut chw = vec![0f32; 3 * h * w];
            for p in 0..h * w {
                for c in 0..3 {
                    chw[c * h * w + p] = f32::from(src[p * 3 + c]) / 255.0;
                }
            }
            host_refiner::resize_bicubic_chw(&chw, 3, h, w, oh, ow)
        })
        .collect();
    let f = idx.len();
    let mut out = vec![0f32; 3 * f * plane];
    for (ti, fr) in per_frame.iter().enumerate() {
        for c in 0..3 {
            let dst = &mut out[(c * f + ti) * plane..(c * f + ti + 1) * plane];
            for (d, &s) in dst.iter_mut().zip(&fr[c * plane..(c + 1) * plane]) {
                *d = s * 2.0 - 1.0;
            }
        }
    }
    out
}

fn write_video(dir: &Path, fps: u32, rgb: &Rgb) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(terr)?;
    let mut writer = VideoWriter::spawn(dir, fps, true)?;
    writer.push(0, rgb.height, rgb.width, rgb.rgb.clone())?;
    writer.finish()?;
    Ok(())
}

fn write_timing(out_dir: &Path, timing: &LingBotTiming) -> Result<()> {
    let text = serde_json::to_string_pretty(timing).map_err(terr)?;
    std::fs::write(out_dir.join("lingbot_timing.json"), text).map_err(terr)?;
    crate::wan::log::info(format_args!("lingbot timing {}", serde_json::to_string(timing).map_err(terr)?));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_request_matches_the_profile() {
        let r = LingBotRequest::official("a cat");
        assert_eq!((r.width, r.height, r.num_frames, r.num_steps), (832, 480, 121, 40));
        assert_eq!(r.seed, 42);
        let f = r.refiner.as_ref().unwrap();
        assert_eq!((f.width, f.height, f.steps), (1920, 1088, 8));
        r.check().unwrap();
        let mut bad = r.clone();
        bad.num_frames = 120;
        assert!(bad.check().is_err());
    }

    #[test]
    fn upsample_maps_u8_to_signed_unit_range() {
        // 1 frame 2x2 → 4x4; constant 255 stays 1.0, constant 0 stays -1.
        let rgb = vec![255u8; 2 * 2 * 3];
        let up = upsample_rgb(&rgb, &[0], 2, 2, 4, 4);
        assert_eq!(up.len(), 3 * 16);
        assert!(up.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        let up = upsample_rgb(&vec![0u8; 12], &[0], 2, 2, 4, 4);
        assert!(up.iter().all(|&v| (v + 1.0).abs() < 1e-6));
    }

    #[test]
    fn tiny_two_stage_denoise_runs_on_generated_weights() {
        // Base + refiner sampling loops over a generated tiny MoE DiT, with
        // CFG, EasyCache and the PISA route: the numerics are not checked
        // (random weights), the control flow and shapes are.
        let cfg = LingBotTransformerConfig::tiny_moe();
        let map = WeightMap::generated(|key, shape| {
            let n: usize = shape.iter().product();
            let is_norm = key.contains("norm") && key.ends_with(".weight");
            (0..n)
                .map(|i| if is_norm { 1.0 } else { ((i as f32) * 0.013 + key.len() as f32).sin() * 0.05 })
                .collect()
        });
        let dit = LingBotTransformer::load(cfg.clone(), &map).unwrap();
        let text = dit
            .embed_text(&CudaTensor::from_vec(vec![0.2; 3 * cfg.text_dim], vec![1, 3, cfg.text_dim]).unwrap())
            .unwrap();
        let neg = dit.embed_text(&CudaTensor::zeros(&[1, 3, cfg.text_dim])).unwrap();
        let shape = [1, cfg.in_channels, 2, 4, 4];
        let n: usize = shape.iter().product();
        let mut sched = base_unipc(6, 3.0);
        let (lat, stats) = denoise(
            &dit,
            &mut sched,
            vec![0.5; n],
            shape,
            &text,
            Some(&neg),
            3.0,
            Some(EasyCacheConfig { threshold: 10.0, head_steps: 1, tail_steps: 1, max_reuse: 1 }),
            None,
            "test",
        )
        .unwrap();
        assert_eq!(lat.len(), n);
        assert!(lat.iter().all(|v| v.is_finite()));
        assert_eq!(stats.computed + stats.reused, 6);
        assert!(stats.reused >= 1);

        let mut rsched = refiner_unipc(8, 3.0, 0.85, 2).unwrap();
        let steps = rsched.inference_timesteps().len();
        let policy = PisaPolicy::published();
        let (rl, rs) = denoise(&dit, &mut rsched, lat, shape, &text, Some(&neg), 3.0, None, Some(&policy), "r")
            .unwrap();
        assert_eq!(rl.len(), n);
        assert_eq!(rs.computed, steps);
        assert_eq!(rs.sparse, (0..steps).filter(|&s| policy.sparse_step(s, steps)).count());
    }
}
