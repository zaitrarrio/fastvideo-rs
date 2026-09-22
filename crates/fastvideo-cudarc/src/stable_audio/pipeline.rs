//! Stable Audio T2A generate scaffold: noise → DiT → stub VAE → WAV.

use std::path::{Path, PathBuf};

use fastvideo_models::stable_audio::{StableAudioConfig, StableAudioPreset};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::transformer::StableAudioTransformer;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct StableAudioRequest {
    pub prompt: String,
    pub seed: u64,
    pub duration_s: f32,
    pub num_steps: usize,
    pub sample_rate: u32,
    pub preset: StableAudioPreset,
}

impl StableAudioRequest {
    pub fn for_preset(preset: StableAudioPreset, prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            seed,
            duration_s: preset.duration_s(),
            num_steps: preset.default_steps(),
            sample_rate: preset.sample_rate(),
            preset,
        }
    }
}

/// Identity/nearest audio VAE stub: latents → PCM via channel fold + upsample.
pub struct AudioVaeStub {
    pub latent_channels: usize,
    pub hop_length: usize,
    pub audio_channels: usize,
}

impl AudioVaeStub {
    pub fn decode(&self, latents: &CudaTensor) -> Result<Vec<f32>> {
        let [_, c, t] = match latents.shape[..] {
            [1, c, t] => [1, c, t],
            _ => return Err(msg(format!("audio vae want [1,C,T], got {:?}", latents.shape))),
        };
        let data = latents.host_cow()?;
        let samples = t * self.hop_length;
        let mut pcm = vec![0f32; samples * self.audio_channels];
        for i in 0..samples {
            let ti = i / self.hop_length;
            let mut acc = [0f32; 2];
            for ch in 0..c.min(self.latent_channels) {
                acc[ch % self.audio_channels] += data[ch * t + ti.min(t - 1)];
            }
            for a in 0..self.audio_channels {
                pcm[i * self.audio_channels + a] = acc[a].tanh() * 0.2;
            }
        }
        Ok(pcm)
    }
}

pub struct StableAudioPipeline {
    pub root: PathBuf,
    pub cfg: StableAudioConfig,
    pub preset: StableAudioPreset,
    pub dit: Option<StableAudioTransformer>,
    pub vae: Option<AudioVaeStub>,
}

impl StableAudioPipeline {
    pub fn open(root: impl Into<PathBuf>, preset: StableAudioPreset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            cfg: StableAudioConfig::for_preset(preset),
            preset,
            dit: None,
            vae: None,
        })
    }

    pub fn load_dit(&mut self) -> Result<()> {
        let map = WeightMap::open(&self.root.join("transformer")).map_err(|e| msg(e.to_string()))?;
        self.dit = Some(StableAudioTransformer::load(self.cfg.dit.clone(), &map)?);
        Ok(())
    }

    pub fn load_dit_zeros_tiny(&mut self) -> Result<()> {
        self.cfg = StableAudioConfig::tiny();
        self.dit = Some(StableAudioTransformer::zeros(self.cfg.dit.clone())?);
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
        let dim = self.cfg.dit.cross_attention_dim;
        let te = self.root.join("text_encoder");
        crate::text_encode::zeros_or_encode(allow_zeros, &[1, 16, dim], &te, || {
            // Stable Audio Open: T5 projected to cross_attention_dim (often 768).
            // Prefer T5-11B when d_model matches after broadcast; else CLIP-L.
            let map = WeightMap::open(&te).map_err(|e| msg(e.to_string()))?;
            if map.contains("encoder.block.0.layer.0.SelfAttention.q.weight")
                || map.contains("encoder.embed_tokens.weight")
                || map.contains("shared.weight")
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

    pub fn generate(&self, request: &StableAudioRequest, out_path: &Path) -> Result<()> {
        let dit = self.dit.as_ref().ok_or_else(|| msg("Stable Audio: call load_dit()"))?;
        let vae = self.vae.as_ref().ok_or_else(|| msg("Stable Audio: call load_vae_stub()"))?;
        let text = self.encode_text(&request.prompt)?;
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
            let pred = dit.forward(&lat, &text, t as f32)?;
            let vel = pred.host_cow()?;
            sample = sched.step_euler(&sample, &vel[..n]).map_err(msg)?;
        }
        let latents = CudaTensor::from_vec(sample, vec![1, c, tlen])?;
        let pcm = vae.decode(&latents)?;
        write_wav(out_path, &pcm, request.sample_rate, self.cfg.audio_channels as u16)?;
        Ok(())
    }
}

pub(crate) fn write_wav_public(
    path: &Path,
    pcm: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    write_wav(path, pcm, sample_rate, channels)
}

fn write_wav(path: &Path, pcm: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| msg(e.to_string()))?;
    }
    let frames = pcm.len() / channels as usize;
    let data_bytes = frames * channels as usize * 2;
    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_bytes);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_bytes as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * channels as u32 * 2;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&(channels * 2).to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, buf).map_err(|e| msg(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_generate_wav() {
        let mut pipe = StableAudioPipeline::open("/tmp/sa-missing", StableAudioPreset::OpenSmall).unwrap();
        pipe.load_dit_zeros_tiny().unwrap();
        pipe.load_vae_stub();
        let mut r = StableAudioRequest::for_preset(StableAudioPreset::OpenSmall, "drums", 1);
        r.duration_s = 0.5;
        r.num_steps = 2;
        r.sample_rate = 16_000;
        let dir = std::env::temp_dir().join("stable-audio-tiny-test.wav");
        pipe.generate(&r, &dir).unwrap();
        assert!(dir.is_file());
        let _ = std::fs::remove_file(&dir);
    }
}
