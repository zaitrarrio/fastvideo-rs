//! Cosmos3-Super T2V generate (`Cosmos3OmniPipeline.__call__`, text-to-video
//! branch): tokenize cond / uncond prompts → text-tower K/V caches → UniPC
//! (Karras flow sigmas) with CFG over the gen tower → Wan 2.2 VAE decode.
//!
//! The published optimized arm (`models/cosmos3/optimized/env.sh`) adds
//! TeaCache (threshold 1.15, start 10, ≤ 3 consecutive reuses, signal: the
//! step's time embedding) and step-selective NVFP4 linears. TeaCache is here
//! ([`Cosmos3Request::teacache`]); the precision half of the arm is the
//! existing process-wide `FASTVIDEO_FP8` W8A8 (all steps), not NVFP4 (see
//! docs/ports/cosmos3.md).

use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_models::cosmos3::rope::{rope_tables, vision_positions};
use fastvideo_models::cosmos3::{latent_grid, prompt, schedule, Cosmos3TransformerConfig, SolCosmosTea};
use fastvideo_models::wan::WanVaeConfig;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result, VideoWriter};
use crate::wan::tensor::CudaTensor;
use crate::wan::vae::AutoencoderKlWan;
use crate::wan::weights::WeightMap;

use super::transformer::{Cosmos3Transformer, GenCache, UndCache};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

fn terr(e: impl std::fmt::Display) -> PipelineError {
    PipelineError::Message(e.to_string())
}

#[derive(Debug, Clone)]
pub struct Cosmos3Request {
    pub prompt: String,
    /// `None`: the empty string (the pipeline's default).
    pub negative_prompt: Option<String>,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub fps: f64,
    pub num_steps: usize,
    pub guidance_scale: f32,
    /// The optimized arm's TeaCache (`teacache_c115_s10_m3`).
    pub teacache: bool,
}

impl Cosmos3Request {
    /// `models/cosmos3.toml` official config: 1280×720, 189 f, 35 steps,
    /// guidance 6, 24 fps, seed 42.
    pub fn official(prompt: impl Into<String>) -> Self {
        use fastvideo_models::cosmos3::*;
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            seed: 42,
            height: OFFICIAL_HEIGHT,
            width: OFFICIAL_WIDTH,
            num_frames: OFFICIAL_FRAMES,
            fps: f64::from(OFFICIAL_FPS),
            num_steps: OFFICIAL_STEPS,
            guidance_scale: OFFICIAL_GUIDANCE,
            teacache: false,
        }
    }
}

/// Wall times of one generate, seconds (`cosmos3_timing.json`).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Cosmos3Timing {
    pub arm: String,
    pub und_resident: bool,
    pub fp8: bool,
    pub load_s: f64,
    pub text_tower_s: f64,
    pub denoise_s: f64,
    pub decode_s: f64,
    pub export_s: f64,
    /// Text tower + denoise + decode (+ export): the request, weights loaded.
    pub request_s: f64,
    pub steps: usize,
    pub steps_computed: usize,
    pub steps_reused: usize,
    pub und_tokens_cond: usize,
    pub und_tokens_uncond: usize,
    pub gen_tokens: usize,
}

pub struct Cosmos3Pipeline {
    pub root: PathBuf,
    pub cfg: Cosmos3TransformerConfig,
    map: WeightMap,
    pub dit: Option<Cosmos3Transformer>,
    pub vae: Option<AutoencoderKlWan>,
}

fn und_resident() -> bool {
    matches!(
        std::env::var("FASTVIDEO_COSMOS3_UND").ok().as_deref().map(str::trim),
        Some("resident")
    )
}

impl Cosmos3Pipeline {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let dir = root.join("transformer");
        let cfg = match std::fs::read_to_string(dir.join("config.json")) {
            Ok(text) => Cosmos3TransformerConfig::from_hub_json(&text).map_err(msg)?,
            Err(_) => Cosmos3TransformerConfig::super_64b(),
        };
        let map = WeightMap::open(&dir).map_err(terr)?;
        Ok(Self {
            root,
            cfg,
            map,
            dit: None,
            vae: None,
        })
    }

    pub fn load(&mut self) -> Result<()> {
        if self.dit.is_none() {
            self.dit = Some(Cosmos3Transformer::load(self.cfg.clone(), &self.map, und_resident())?);
        }
        if self.vae.is_none() {
            let dir = self.root.join("vae");
            let text = std::fs::read_to_string(dir.join("config.json")).map_err(terr)?;
            let mut vcfg = WanVaeConfig::from_json_str(&text).map_err(msg)?;
            vcfg.load_encoder = false;
            let vmap = WeightMap::open(&dir).map_err(terr)?;
            self.vae = Some(AutoencoderKlWan::load(vcfg, &vmap).map_err(terr)?);
        }
        Ok(())
    }

    fn tokenizer(&self) -> Result<prompt::Tokenizer> {
        let path = ["text_tokenizer/tokenizer.json", "tokenizer.json"]
            .iter()
            .map(|p| self.root.join(p))
            .find(|p| p.is_file())
            .ok_or_else(|| msg(format!("{}: no text_tokenizer/tokenizer.json", self.root.display())))?;
        prompt::Tokenizer::from_file(&path).map_err(|e| msg(format!("{}: {e}", path.display())))
    }

    pub fn generate(&mut self, request: &Cosmos3Request, out_dir: &Path) -> Result<Cosmos3Timing> {
        let t = Instant::now();
        self.load()?;
        let mut timing = Cosmos3Timing {
            arm: if request.teacache { "teacache" } else { "baseline" }.into(),
            und_resident: und_resident(),
            fp8: crate::wan::envflag::bool_flag("FASTVIDEO_FP8", false),
            load_s: t.elapsed().as_secs_f64(),
            ..Default::default()
        };
        std::fs::create_dir_all(out_dir).map_err(terr)?;
        let request_start = Instant::now();
        let dit = self.dit.as_ref().expect("loaded");
        let cfg = &self.cfg;

        // Prompts → text-tower caches.
        let t = Instant::now();
        let tok = self.tokenizer()?;
        let (f, h, w, fps) = (request.num_frames, request.height, request.width, request.fps);
        let cond_ids = prompt::token_ids(&tok, &request.prompt, false, f, h, w, fps).map_err(msg)?;
        let do_cfg = request.guidance_scale > 1.0;
        let neg = request.negative_prompt.as_deref().unwrap_or("");
        let cond = dit.und_cache(&self.map, &cond_ids)?;
        let uncond = if do_cfg {
            let ids = prompt::token_ids(&tok, neg, true, f, h, w, fps).map_err(msg)?;
            Some(dit.und_cache(&self.map, &ids)?)
        } else {
            None
        };
        timing.und_tokens_cond = cond.len;
        timing.und_tokens_uncond = uncond.as_ref().map_or(0, |u| u.len);
        timing.text_tower_s = t.elapsed().as_secs_f64();

        // Latents and per-branch vision rotary tables.
        let (lat, grid) = latent_grid(f, h, w, cfg.latent_patch_size);
        let shape = vec![1, cfg.latent_channel, lat[0], lat[1], lat[2]];
        let n: usize = shape.iter().product();
        timing.gen_tokens = grid.iter().product();
        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let mut latents: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let tables = |und: &UndCache| -> Result<(CudaTensor, CudaTensor)> {
            let offset = (und.len + cfg.temporal_modality_margin) as f32;
            let pos = vision_positions(cfg, grid, offset, Some(fps), 4);
            let (c, s) = rope_tables(cfg, &pos);
            let mut c = CudaTensor::from_vec(c, vec![pos.len(), cfg.head_dim])?;
            let mut s = CudaTensor::from_vec(s, vec![pos.len(), cfg.head_dim])?;
            c.pin_device()?;
            s.pin_device()?;
            Ok((c, s))
        };
        let rope_c = tables(&cond)?;
        let rope_u = match &uncond {
            Some(u) => Some(tables(u)?),
            None => None,
        };

        let t = Instant::now();
        let mut sched = schedule::unipc(request.num_steps);
        let ts: Vec<i64> = sched.inference_timesteps_i64().to_vec();
        timing.steps = ts.len();
        let mut tea = request.teacache.then(SolCosmosTea::official);
        let (mut res_c, mut res_u) = (None, None);
        for (i, &tstep) in ts.iter().enumerate() {
            let step_t = Instant::now();
            let temb = dit.time_embed(schedule::transformer_timestep(tstep, cfg.timestep_scale));
            let compute = tea.as_mut().map_or(true, |c| c.decide(i, &temb));
            let lat_t = CudaTensor::from_vec(latents.clone(), shape.clone())?;
            let tea_on = tea.is_some();
            let vc = dit.forward_gen(&lat_t, &temb, &cond, &rope_c.0, &rope_c.1, mode(tea_on, compute, &mut res_c))?;
            let v = match (&uncond, &rope_u) {
                (Some(u), Some((uc, us))) => {
                    let vu = dit.forward_gen(&lat_t, &temb, u, uc, us, mode(tea_on, compute, &mut res_u))?;
                    vu.add(&vc.sub(&vu)?.try_mul_scalar(request.guidance_scale)?)?
                }
                _ => vc,
            };
            if compute {
                timing.steps_computed += 1;
            } else {
                timing.steps_reused += 1;
            }
            let v = v.host_cow()?.into_owned();
            latents = sched.step(&v, &latents).map_err(msg)?;
            crate::wan::log::info(format_args!(
                "cosmos3 step {}/{} t={tstep}{} {:.2}s",
                i + 1,
                ts.len(),
                if compute { "" } else { " (teacache reuse)" },
                step_t.elapsed().as_secs_f64()
            ));
        }
        timing.denoise_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let vae = self.vae.as_ref().expect("loaded");
        let z = vae
            .scale_latents(&CudaTensor::from_vec(latents, shape)?)
            .map_err(terr)?;
        let pixels = vae.decode(&z).map_err(terr)?;
        let [_, _, fo, ho, wo] = match pixels.shape[..] {
            [1, 3, a, b, c] => [1, 3, a, b, c],
            _ => return Err(msg(format!("cosmos3 decode shape {:?}", pixels.shape))),
        };
        let frames = pixels
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(vec![fo, 3, ho, wo])?
            .clamp(-1.0, 1.0);
        let rgb = frames_to_rgb8(&frames)?;
        timing.decode_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut writer = VideoWriter::spawn(out_dir, request.fps.round() as u32, true)?;
        writer.push(0, ho, wo, rgb)?;
        writer.finish()?;
        timing.export_s = t.elapsed().as_secs_f64();
        timing.request_s = request_start.elapsed().as_secs_f64();
        let text = serde_json::to_string_pretty(&timing).map_err(terr)?;
        std::fs::write(out_dir.join("cosmos3_timing.json"), &text).map_err(terr)?;
        crate::wan::log::info(format_args!("cosmos3 timing {}", serde_json::to_string(&timing).map_err(terr)?));
        Ok(timing)
    }
}

fn mode(tea_on: bool, compute: bool, slot: &mut Option<CudaTensor>) -> GenCache<'_> {
    match (tea_on, compute) {
        (false, _) => GenCache::Off,
        (true, true) => GenCache::Compute(slot),
        (true, false) => GenCache::Reuse(slot),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_request() {
        let r = Cosmos3Request::official("a lake");
        assert_eq!((r.width, r.height, r.num_frames, r.num_steps), (1280, 720, 189, 35));
        assert_eq!(r.guidance_scale, 6.0);
        assert!(!r.teacache);
    }
}
