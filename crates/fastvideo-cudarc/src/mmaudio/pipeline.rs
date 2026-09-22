//! MMAudio generate scaffold (T2A / V2A with optional video path).

use std::path::{Path, PathBuf};

use fastvideo_models::mmaudio::{MmAudioConfig, MmAudioPreset};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::transformer::MmAudioTransformer;
use crate::stable_audio::pipeline::AudioVaeStub;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct MmAudioRequest {
    pub prompt: String,
    pub seed: u64,
    pub duration_s: f32,
    pub num_steps: usize,
    pub sample_rate: u32,
    pub preset: MmAudioPreset,
    /// Optional video/frames root for V2A (zeros visual cond if absent).
    pub video_path: Option<PathBuf>,
}

impl MmAudioRequest {
    pub fn t2a(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = MmAudioPreset::Large44kV2;
        Self {
            prompt: prompt.into(),
            seed,
            duration_s: preset.duration_s(),
            num_steps: preset.default_steps(),
            sample_rate: preset.sample_rate(),
            preset,
            video_path: None,
        }
    }
}

pub struct MmAudioPipeline {
    pub root: PathBuf,
    pub cfg: MmAudioConfig,
    pub preset: MmAudioPreset,
    pub dit: Option<MmAudioTransformer>,
    pub vae: Option<AudioVaeStub>,
}

impl MmAudioPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: MmAudioPreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: MmAudioConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    /// Prefer `MMAUDIO_MODEL_PATH` when Hub pack is missing.
    pub fn resolve_root(explicit: impl Into<PathBuf>) -> PathBuf {
        std::env::var_os("MMAUDIO_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| explicit.into())
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(MmAudioTransformer::load(self.cfg.dit.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = MmAudioConfig::tiny();
        self.dit = Some(MmAudioTransformer::zeros(self.cfg.dit.clone())?);
        Ok(())
    }

    pub fn load_vae_stub(&mut self) {
        self.vae = Some(AudioVaeStub {
            latent_channels: self.cfg.dit.out_channels,
            hop_length: if self.cfg.dit.sample_size <= 32 { 8 } else { self.preset.hop_length() },
            audio_channels: self.cfg.audio_channels,
        });
    }

    fn encode_text(&self, prompt: &str) -> Result<CudaTensor> {
        let allow_zeros = self.cfg.dit.num_layers <= 2;
        let dim = self.cfg.dit.text_dim;
        let te = self.root.join("text_encoder");
        crate::text_encode::zeros_or_encode(allow_zeros, &[1, 16, dim], &te, || {
            let map = WeightMap::open(&te).map_err(|e| msg(e.to_string()))?;
            if map.contains("encoder.block.0.layer.0.SelfAttention.q.weight")
                || map.contains("shared.weight")
                || map.contains("encoder.embed_tokens.weight")
            {
                let emb = crate::text_encode::encode_t5_11b(
                    &self.root,
                    "text_encoder",
                    "tokenizer",
                    prompt,
                    64,
                )?;
                return crate::text_encode::broadcast_to_dim(&emb, dim);
            }
            let emb = crate::text_encode::encode_clip_l_hidden(
                &self.root,
                "text_encoder",
                "tokenizer",
                prompt,
            )?;
            crate::text_encode::broadcast_to_dim(&emb, dim)
        })
    }

    fn encode_visual(&self, path: Option<&Path>) -> Result<Option<CudaTensor>> {
        let Some(p) = path else {
            return Ok(None);
        };
        let allow_zeros = self.cfg.dit.num_layers <= 2;
        let dim = self.cfg.dit.visual_dim;
        let ve = self.root.join("image_encoder");
        if !ve.is_dir() && !allow_zeros {
            return Err(msg(format!(
                "MMAudio V2A: video path {} set but missing {} (Synchformer/CLIP visual)",
                p.display(),
                ve.display()
            )));
        }
        if !ve.is_dir() {
            return Ok(Some(CudaTensor::zeros(&[1, 16, dim])));
        }
        // Visual graph: reuse CLIP sequence path when weights look like CLIP;
        // otherwise clear error (no silent zeros with weights present).
        let map = WeightMap::open(&ve).map_err(|e| msg(e.to_string()))?;
        if map.contains("text_model.embeddings.token_embedding.weight")
            || map.contains("vision_model.embeddings.patch_embedding.weight")
            || map.contains("embeddings.patch_embedding.weight")
        {
            // Frame path reserved for Synchformer; until wired, CLIP text tower
            // on the prompt is not correct for visual — error clearly.
            return Err(msg(format!(
                "MMAudio: {} present (visual keys detected) but Synchformer frame encode \
                 is not wired yet; refuse zeros with weights present",
                ve.display()
            )));
        }
        Err(msg(format!(
            "MMAudio: {} present but no recognized Synchformer/CLIP visual keys",
            ve.display()
        )))
    }

    pub fn generate(&self, request: &MmAudioRequest, out_path: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| msg("MMAudio: call load_dit()"))?;
        let vae = self.vae.as_ref().ok_or_else(|| msg("MMAudio: call load_vae_stub()"))?;
        let text = self.encode_text(&request.prompt)?;
        let visual = self.encode_visual(request.video_path.as_deref())?;
        let tlen = if self.cfg.dit.num_layers <= 2 {
            self.cfg.dit.sample_size
        } else {
            let want = ((request.duration_s * request.sample_rate as f32) as usize)
                .div_ceil(self.preset.hop_length())
                .max(1);
            want.min(self.cfg.dit.sample_size)
        };
        let c = self.cfg.dit.in_channels;
        let n = c * tlen;
        let mut sched = self.cfg.schedule(request.num_steps);
        let timesteps = sched.inference_timesteps().to_vec();
        let sigmas = sched.inference_sigmas().to_vec();
        let mut rng = rand::rngs::StdRng::seed_from_u64(request.seed);
        let mut sample: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(&mut rng);
                z * (sigmas[0] as f32)
            })
            .collect();
        for &t in &timesteps {
            let lat = CudaTensor::from_vec(sample.clone(), vec![1, c, tlen])?;
            let pred = dit.forward(&lat, &text, visual.as_ref(), t as f32)?;
            let vel = pred.host_cow()?;
            sample = sched.step_euler(&sample, &vel[..n]).map_err(msg)?;
        }
        let latents = CudaTensor::from_vec(sample, vec![1, c, tlen])?;
        let pcm = vae.decode(&latents)?;
        crate::stable_audio::pipeline::write_wav_public(
            out_path,
            &pcm,
            request.sample_rate,
            self.cfg.audio_channels as u16,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_generate_wav() {
        let mut pipe = MmAudioPipeline::open("/tmp/mmaudio-missing", MmAudioPreset::Large44kV2)
            .unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        pipe.load_vae_stub();
        let mut r = MmAudioRequest::t2a("rain", 1);
        r.duration_s = 0.5;
        r.num_steps = 2;
        r.sample_rate = 16_000;
        let dir = std::env::temp_dir().join("mmaudio-tiny-test.wav");
        pipe.generate(&r, &dir).unwrap();
        assert!(dir.is_file());
        let _ = std::fs::remove_file(&dir);
    }
}
