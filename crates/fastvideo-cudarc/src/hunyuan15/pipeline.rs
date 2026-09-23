//! HunyuanVideo 1.5 generate entry.
//!
//! Flow: optional text encode → FlowMatch Euler denoise on the DiT → VAE
//! decode. Text (Qwen mid-layer + ByT5 zeros) and DiT are wired; VAE decode
//! still refuses until the causal graph lands.

use std::path::{Path, PathBuf};

use fastvideo_models::hunyuan15::{
    Hunyuan15PipelineDefaults, Hunyuan15Preset, Hunyuan15Schedule, Hunyuan15TransformerConfig,
    Hunyuan15VaeConfig,
};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

use super::text::{self, Hunyuan15TextConditioning};
use super::transformer::Hunyuan15Transformer;
use super::vae::{self, Hunyuan15Vae};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct Hunyuan15Request {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub preset: Hunyuan15Preset,
    pub image_path: Option<PathBuf>,
}

impl Hunyuan15Request {
    pub fn t2v_480p(prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            seed,
            height: 480,
            width: 854,
            num_frames: 121,
            num_steps: 50,
            preset: Hunyuan15Preset::T2v480p,
            image_path: None,
        }
    }
}

pub struct Hunyuan15Pipeline {
    pub root: PathBuf,
    pub defaults: Hunyuan15PipelineDefaults,
    pub dit_cfg: Hunyuan15TransformerConfig,
    pub vae_cfg: Hunyuan15VaeConfig,
    pub dit: Option<Hunyuan15Transformer>,
    pub vae: Option<Hunyuan15Vae>,
}

impl Hunyuan15Pipeline {
    pub fn open(root: impl Into<PathBuf>, preset: Hunyuan15Preset) -> Result<Self> {
        let root = root.into();
        let mut dit_cfg = Hunyuan15TransformerConfig::fasthunyuan15();
        if matches!(preset, Hunyuan15Preset::Sr1080p) {
            dit_cfg = dit_cfg.with_meanflow();
        }
        Ok(Self {
            root,
            defaults: Hunyuan15PipelineDefaults::for_preset(preset),
            dit_cfg,
            vae_cfg: Hunyuan15VaeConfig::fasthunyuan15(),
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map =
            WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(Hunyuan15Transformer::load(self.dit_cfg.clone(), &map)?);
        Ok(())
    }

    pub fn load_vae(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("vae")).map_err(|e| msg(e.to_string()))?;
        self.vae = Some(Hunyuan15Vae::load(&map, self.vae_cfg.clone())?);
        Ok(())
    }

    pub fn schedule(&self, steps: usize) -> Hunyuan15Schedule {
        Hunyuan15Schedule::new(steps, self.defaults.preset)
    }

    fn encode_text(&self, prompt: &str) -> Result<Hunyuan15TextConditioning> {
        text::encode_prompt(&self.root, prompt, &self.defaults, &self.dit_cfg)
            .map_err(|e| msg(e.to_string()))
    }

    /// End-to-end generate. Completes denoise when DiT+text are present; VAE
    /// decode still errors with a clear path message.
    pub fn generate(&self, request: &Hunyuan15Request, out_dir: &Path) -> Result<()> {
        let _ = out_dir;
        if request.image_path.is_some() {
            return Err(msg(
                "HunyuanVideo 1.5 I2V: image conditioning not wired yet (needs channel-0 pack + SigLIP)",
            ));
        }
        let dit = self.dit.as_ref().ok_or_else(|| {
            msg("HunyuanVideo 1.5: call load_dit() after placing Diffusers `transformer/` under --weights")
        })?;

        let text_ok = self.root.join("text_encoder").is_dir()
            && self.root.join("tokenizer").join("tokenizer.json").is_file();
        let cond = if text_ok {
            self.encode_text(&request.prompt)?
        } else {
            return Err(msg(
                "HunyuanVideo 1.5: need `tokenizer/tokenizer.json` + `text_encoder/` (Qwen2.5-VL-7B) under --weights",
            ));
        };

        let lt = vae::latent_frames(request.num_frames, self.vae_cfg.temporal_compression_ratio);
        let lh = vae::latent_hw(request.height, self.vae_cfg.spatial_compression_ratio);
        let lw = vae::latent_hw(request.width, self.vae_cfg.spatial_compression_ratio);
        let c_in = self.dit_cfg.in_channels;
        let c_out = self.dit_cfg.out_channels;

        let mut sched = self.schedule(request.num_steps);
        let sigmas = sched.sigmas().to_vec();
        let timesteps = sched.timesteps().to_vec();

        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let n = c_out * lt * lh * lw;
        let mut sample: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(&mut rng);
                z * (sigmas[0] as f32)
            })
            .collect();
        // I2V packs use in_channels=65 (noise 32 + cond 32 + mask 1). T2V still
        // feeds `in_channels` to the DiT — zero-pad cond channels for T2V.
        let pad_c = c_in.saturating_sub(c_out);

        for (step, &t) in timesteps.iter().enumerate() {
            let lat = pack_latents(&sample, c_out, lt, lh, lw, pad_c)?;
            let text2 = Some(&cond.byt5);
            let velocity = dit.forward(&lat, &cond.qwen, text2, t as f32)?;
            // DiT returns `[B, T*H*W, out_channels]` — flatten to CTHW host.
            let vel = tokens_to_cthw(&velocity, c_out, lt, lh, lw)?;
            sample = sched.inner.step_euler(&sample, &vel).map_err(|e| msg(e))?;
            let _ = step;
        }

        let latents = CudaTensor::from_vec(sample, vec![1, c_out, lt, lh, lw])?;
        let vae = self.vae.as_ref().ok_or_else(|| {
            msg("HunyuanVideo 1.5: call load_vae() after placing Diffusers `vae/` under --weights")
        })?;
        let scaled = latents.try_mul_scalar(1.0 / vae.scaling_factor())?;
        let pixels = vae.decode(&scaled)?;
        // [B,3,T,H,W] → [T,3,H,W] for PNG dump
        let [_, _, tf, hf, wf] = match pixels.shape[..] {
            [1, 3, tf, hf, wf] => [1, 3, tf, hf, wf],
            _ => {
                return Err(msg(format!(
                    "hy15 decode shape {:?} want [1,3,T,H,W]",
                    pixels.shape
                )))
            }
        };
        let by_frame = pixels.reshape(vec![tf, 3, hf, wf])?;
        let rgb = crate::wan::pipeline::frames_to_rgb8(&by_frame)?;
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

fn pack_latents(
    sample: &[f32],
    c_out: usize,
    t: usize,
    h: usize,
    w: usize,
    pad_c: usize,
) -> Result<CudaTensor> {
    let want = c_out * t * h * w;
    if sample.len() != want {
        return Err(msg(format!(
            "hy15 pack: sample {} vs {want} (C={c_out} T={t} H={h} W={w})",
            sample.len()
        )));
    }
    if pad_c == 0 {
        return Ok(CudaTensor::from_vec(
            sample.to_vec(),
            vec![1, c_out, t, h, w],
        )?);
    }
    let mut packed = vec![0f32; (c_out + pad_c) * t * h * w];
    // Layout C,T,H,W — copy noise channels, leave cond/mask zero.
    packed[..want].copy_from_slice(sample);
    Ok(CudaTensor::from_vec(
        packed,
        vec![1, c_out + pad_c, t, h, w],
    )?)
}

fn tokens_to_cthw(tokens: &CudaTensor, c: usize, t: usize, h: usize, w: usize) -> Result<Vec<f32>> {
    let seq = t * h * w;
    if tokens.shape != [1, seq, c] {
        return Err(msg(format!(
            "hy15 unpatch: got {:?} want [1,{seq},{c}]",
            tokens.shape
        )));
    }
    let host = tokens.host_cow()?;
    // tokens [1, THW, C] → [C, T, H, W]
    let mut out = vec![0f32; c * seq];
    for s in 0..seq {
        for ch in 0..c {
            out[ch * seq + s] = host[s * c + ch];
        }
    }
    Ok(out)
}
