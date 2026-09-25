//! FLUX.1 T2I generate path on cudarc (`CudaTensor`).

use std::path::Path;

use fastvideo_models::flux::{
    arch_from_transformer_config, calculate_shift_flux1, packed_hw, unpack_latents_flux1, Flux1ArchConfig,
};
use fastvideo_models::flux2::Flux2VaeConfig;
use fastvideo_models::schedulers::FlowMatchEulerDiscreteScheduler;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use thiserror::Error;

use crate::flux2::vae::AutoencoderKlFlux2;
use crate::wan::pipeline::write_frames;
use crate::wan::stats;
use crate::wan::tensor::{CudaTensor, TensorError};
use crate::wan::weights::WeightMap;

use super::text::{dummy_text, pad_token_ids, t5_len, tokenize_flux1, Flux1TextEncoder};
use super::transformer::Flux1Transformer2D;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, PipelineError>;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub prompt: String,
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    pub output_dir: String,
    pub tiny: bool,
    pub clip_tokenizer_path: Option<String>,
    pub t5_tokenizer_path: Option<String>,
    pub embedded_cfg_scale: Option<f32>,
    pub preset: String,
    pub t5_max_len: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            prompt: "a photo of a banana on a wooden table, studio lighting".into(),
            height: 1024,
            width: 1024,
            num_inference_steps: 50,
            guidance_scale: 3.5,
            seed: 0,
            output_dir: "out".into(),
            tiny: false,
            clip_tokenizer_path: None,
            t5_tokenizer_path: None,
            embedded_cfg_scale: Some(3.5),
            preset: "flux1_dev".into(),
            t5_max_len: 512,
        }
    }
}

pub struct Flux1Pipeline {
    transformer: Flux1Transformer2D,
    vae: AutoencoderKlFlux2,
    text: Flux1TextEncoder,
    tiny: bool,
}

impl Flux1Pipeline {
    pub fn tiny() -> Self {
        let cfg = Flux1ArchConfig::tiny();
        let mut vae_cfg = Flux2VaeConfig::tiny();
        // Flux1 tiny DiT is 16-ch packed; Flux2 tiny VAE defaults to 8-ch.
        vae_cfg.latent_channels = cfg.in_channels;
        Self {
            transformer: Flux1Transformer2D::zeros(cfg.clone()),
            vae: AutoencoderKlFlux2::zeros(vae_cfg),
            text: Flux1TextEncoder::dummy(cfg.joint_attention_dim, cfg.pooled_projection_dim),
            tiny: true,
        }
    }

    pub fn load(root: &Path, preset: &str) -> Result<Self> {
        let mut cfg = Flux1ArchConfig::from_preset(preset);
        if let Ok(raw) = std::fs::read_to_string(root.join("transformer/config.json")) {
            cfg = arch_from_transformer_config(&cfg, &raw).map_err(PipelineError::Message)?;
        }
        let map = WeightMap::from_dir(&root.join("transformer")).map_err(PipelineError::from)?;
        let transformer = Flux1Transformer2D::load(cfg.clone(), &map)?;
        let vae_dir = root.join("vae");
        let vae = match WeightMap::from_dir(&vae_dir) {
            Ok(vmap) => AutoencoderKlFlux2::load(Flux2VaeConfig::flux1(), &vmap)?,
            Err(e) => {
                return Err(PipelineError::Message(format!(
                    "FLUX.1 VAE load from {}: {e}",
                    vae_dir.display()
                )))
            }
        };
        let t5_max = if preset.contains("schnell") { 256 } else { 512 };
        let text = Flux1TextEncoder::load(
            root,
            cfg.joint_attention_dim,
            cfg.pooled_projection_dim,
            t5_max,
        )?;
        Ok(Self {
            transformer,
            vae,
            text,
            tiny: false,
        })
    }

    pub fn generate(&mut self, cfg: &GenerateConfig) -> Result<Vec<String>> {
        let _gen = crate::wan::log::StepTimer::start("flux1.generate");
        let xfer0 = stats::snapshot();
        let (height, width, steps) = if self.tiny {
            (16usize, 16, 2usize)
        } else {
            (cfg.height.max(16), cfg.width.max(16), cfg.num_inference_steps.max(1))
        };
        let (ph, pw) = if self.tiny {
            (2, 2)
        } else {
            packed_hw(height, width, self.vae.cfg.spatial_compression_ratio)
        };
        let channels = self.transformer.cfg.in_channels;
        let seq = ph * pw;
        crate::wan::log::info(format_args!(
            "flux1 generate preset={} tiny={} {}x{} packed={}x{} seq={} steps={} text={}",
            cfg.preset,
            self.tiny,
            width,
            height,
            ph,
            pw,
            seq,
            steps,
            if dummy_text() { "dummy" } else { "clip+t5" },
        ));
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let noise: Vec<f32> = (0..channels * seq).map(|_| rng.sample(StandardNormal)).collect();
        let mut latents = CudaTensor::from_vec(noise, vec![1, channels, 1, ph, pw])?;
        let (clip_ids, t5_ids) = encode_prompt_ids(&self.text, cfg, self.tiny);
        let t_text = std::time::Instant::now();
        let (encoder, pooled) = {
            let _t = crate::wan::log::StepTimer::start("flux1.text.encode");
            self.text.encode(&clip_ids, &t5_ids)?
        };
        let text_encode_ms = t_text.elapsed().as_millis();
        let mu = calculate_shift_flux1(seq);
        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, 1.0);
        sched.set_timesteps_flux2(steps, Some(mu));
        let guidance = if self.transformer.cfg.guidance_embeds {
            Some(cfg.embedded_cfg_scale.unwrap_or(cfg.guidance_scale))
        } else {
            None
        };
        let timesteps: Vec<f64> = sched.inference_timesteps().to_vec();
        let mut steps_ms = Vec::with_capacity(timesteps.len());
        let mut euler_host_ms = 0u128;
        let t_denoise = std::time::Instant::now();
        {
            let _denoise = crate::wan::log::StepTimer::start(format!("flux1.denoise {} steps", timesteps.len()));
            for (i, t) in timesteps.iter().copied().enumerate() {
                let t_step = std::time::Instant::now();
                let _step = crate::wan::log::StepTimer::start(format!(
                    "flux1.dit step {}/{} t={t:.1}",
                    i + 1,
                    timesteps.len()
                ));
                let vel = self.transformer.forward(
                    &latents,
                    &encoder,
                    &pooled,
                    t as f32 / 1000.0,
                    guidance,
                    ph,
                    pw,
                )?;
                let t_euler = std::time::Instant::now();
                let dt = sched.take_euler_dt().map_err(PipelineError::Message)?;
                latents = CudaTensor::lincomb(&[(1.0, &latents), (dt, &vel)])?;
                euler_host_ms += t_euler.elapsed().as_millis();
                steps_ms.push(t_step.elapsed().as_millis());
            }
        }
        let denoise_ms = t_denoise.elapsed().as_millis();
        let t_unpack = std::time::Instant::now();
        let unpacked = if self.tiny {
            None
        } else {
            Some(unpack_nchw(
                &latents,
                self.vae.cfg.latent_channels,
                ph,
                pw,
            )?)
        };
        let unpack_ms = t_unpack.elapsed().as_millis();
        let t_vae = std::time::Instant::now();
        let decoded = {
            let _vae = crate::wan::log::StepTimer::start("flux1.vae.decode");
            match &unpacked {
                Some(u) => self.vae.decode(u)?,
                None => self.vae.decode(&latents)?,
            }
        };
        let vae_decode_ms = t_vae.elapsed().as_millis();
        let t_write = std::time::Instant::now();
        let paths = {
            let _w = crate::wan::log::StepTimer::start("flux1.write_frames");
            write_frames(&decoded, Path::new(&cfg.output_dir)).map_err(|e| PipelineError::Message(e.to_string()))?
        };
        let write_frames_ms = t_write.elapsed().as_millis();
        let xfer = stats::snapshot().since(&xfer0);
        let profile = serde_json::json!({
            "preset": cfg.preset,
            "tiny": self.tiny,
            "family": "flux1",
            "text": if dummy_text() { "dummy" } else { "clip+t5" },
            "height": height,
            "width": width,
            "packed_h": ph,
            "packed_w": pw,
            "seq": seq,
            "steps": steps,
            "shift_mu": mu,
            "text_encode_ms": text_encode_ms,
            "denoise_ms": denoise_ms,
            "steps_ms": steps_ms,
            "euler_host_ms": euler_host_ms,
            "unpack_ms": unpack_ms,
            "vae_decode_ms": vae_decode_ms,
            "write_frames_ms": write_frames_ms,
            "h2d_count": xfer.h2d_count,
            "h2d_mib": xfer.h2d_bytes >> 20,
            "d2h_count": xfer.d2h_count,
            "d2h_mib": xfer.d2h_bytes >> 20,
            "host_fallbacks": xfer.host_fallbacks,
            "sdpa": crate::wan::nn::sdpa_backend(),
            "profile_run": "this generate (last timed run when used from bench --runs)",
        });
        crate::wan::log::info(format_args!(
            "flux1.profile text_ms={text_encode_ms} denoise_ms={denoise_ms} vae_ms={vae_decode_ms} \
             write_ms={write_frames_ms} euler_host_ms={euler_host_ms} h2d={} ({} MiB) d2h={} ({} MiB)",
            xfer.h2d_count,
            xfer.h2d_bytes >> 20,
            xfer.d2h_count,
            xfer.d2h_bytes >> 20,
        ));
        write_profile(Path::new(&cfg.output_dir), &profile)?;
        Ok(paths)
    }
}

fn unpack_nchw(packed: &CudaTensor, latent_channels: usize, ph: usize, pw: usize) -> Result<CudaTensor> {
    let host = packed.host_cow()?;
    let seq = ph * pw;
    let packed_ch = host.len() / seq;
    let mut seq_major = vec![0.0f32; host.len()];
    for s in 0..seq {
        for c in 0..packed_ch {
            seq_major[s * packed_ch + c] = host[c * seq + s];
        }
    }
    let spatial = unpack_latents_flux1(&seq_major, latent_channels, ph, pw).map_err(PipelineError::Message)?;
    Ok(CudaTensor::from_vec(
        spatial,
        vec![1, latent_channels, 1, ph * 2, pw * 2],
    )?)
}

fn encode_prompt_ids(text: &Flux1TextEncoder, cfg: &GenerateConfig, tiny: bool) -> (Vec<u32>, Vec<u32>) {
    if tiny {
        return ((0..4).collect(), (0..4).collect());
    }
    let clip_len = text.clip_len();
    let t5_default = text.t5_len().max(cfg.t5_max_len);
    let t5_seq = t5_len(t5_default);
    let clip_raw = if let Some(tok) = &cfg.clip_tokenizer_path {
        tokenize_flux1(tok, &cfg.prompt, clip_len)
            .map(|(ids, _)| ids)
            .unwrap_or_else(|_| hash_ids(&cfg.prompt, clip_len.min(16)))
    } else {
        hash_ids(&cfg.prompt, clip_len.min(16))
    };
    let t5_raw = if let Some(tok) = &cfg.t5_tokenizer_path {
        tokenize_flux1(tok, &cfg.prompt, t5_seq)
            .map(|(ids, _)| ids)
            .unwrap_or_else(|_| hash_ids(&cfg.prompt, t5_seq.min(32)))
    } else {
        hash_ids(&cfg.prompt, t5_seq.min(32))
    };
    let (clip_ids, _) = pad_token_ids(&clip_raw, clip_len, 0);
    let (t5_ids, _) = pad_token_ids(&t5_raw, t5_seq, 0);
    (clip_ids, t5_ids)
}

fn write_profile(dir: &Path, profile: &serde_json::Value) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| PipelineError::Message(e.to_string()))?;
    let path = dir.join("profile.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(profile).map_err(|e| PipelineError::Message(e.to_string()))?,
    )
    .map_err(|e| PipelineError::Message(e.to_string()))?;
    crate::wan::log::info(format_args!("wrote {}", path.display()));
    Ok(())
}

fn hash_ids(prompt: &str, len: usize) -> Vec<u32> {
    prompt
        .bytes()
        .chain(0u8..len as u8)
        .take(len)
        .map(|b| b as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_generate_writes_png() {
        let mut pipe = Flux1Pipeline::tiny();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.output_dir = std::env::temp_dir()
            .join("fastvideo-cudarc-flux1-tiny")
            .to_string_lossy()
            .into();
        let paths = pipe.generate(&cfg).unwrap();
        assert!(Path::new(&paths[0]).exists());
        let profile = Path::new(&cfg.output_dir).join("profile.json");
        assert!(profile.exists(), "expected {}", profile.display());
        let body = std::fs::read_to_string(&profile).unwrap();
        assert!(body.contains("text_encode_ms"), "{body}");
        assert!(body.contains("\"family\": \"flux1\""), "{body}");
        assert!(body.contains("shift_mu"), "{body}");
    }
}
