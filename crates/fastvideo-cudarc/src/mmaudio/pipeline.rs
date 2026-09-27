//! MMAudio V2A / T2A generation (`eval_utils.generate`): features, Euler
//! flow matching with classifier-free guidance, VAE decode, BigVGAN.
//!
//! The sampler follows the bf16 network upstream runs: the noise, every
//! intermediate latent, the timestep and each guided flow are rounded to
//! bf16 exactly where torch's bf16 tensors round them (`x = x + dt * flow`
//! is two roundings; `cfg * c + (1 - cfg) * u` three). The step itself is a
//! few thousand values and runs on the host.
//!
//! Oracle hooks (docs/oracle.md): `FASTVIDEO_DUMP_DIR` writes `mm_*`;
//! `FASTVIDEO_INJECT_DIR` reads the reference's `mm_x0` (noise) and, with
//! `FASTVIDEO_MMAUDIO_INJECT=pixels|features`, its preprocessed frames or its
//! encoder outputs.

use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_models::mmaudio::frames::{clip_pixels, select_frames, sync_pixels, CLIP_SIZE, SYNC_SIZE};
use fastvideo_models::mmaudio::{bf16_round, euler_times, MmAudioConfig, MmAudioPreset, SequenceConfig};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::bigvgan::BigVgan;
use super::clip::{ClipText, ClipVisual};
use super::synchformer::Synchformer;
use super::transformer::{Conditions, MmAudioTransformer};
use super::vae::MmAudioVaeDecoder;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;
use crate::wan::{dump, inject};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Decoded rgb24 frames of a clip.
#[derive(Debug, Clone)]
pub struct VideoFrames {
    pub height: usize,
    pub width: usize,
    /// Timestamps (s) of each frame.
    pub times: Vec<f64>,
    pub frames: Vec<Vec<u8>>,
}

impl VideoFrames {
    pub fn constant_rate(height: usize, width: usize, fps: f64, frames: Vec<Vec<u8>>) -> Self {
        let times = (0..frames.len()).map(|i| i as f64 / fps).collect();
        Self {
            height,
            width,
            times,
            frames,
        }
    }

    /// Decode a video file with ffmpeg (rgb24) at its own rate.
    pub fn from_file(path: &Path) -> Result<Self> {
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=p=0",
            ])
            .arg(path)
            .output()
            .map_err(|e| msg(format!("ffprobe: {e}")))?;
        let text = String::from_utf8_lossy(&probe.stdout);
        let parts: Vec<&str> = text.trim().split(',').collect();
        if parts.len() < 3 {
            return Err(msg(format!("ffprobe {}: {text:?}", path.display())));
        }
        let w: usize = parts[0].parse().map_err(|_| msg("ffprobe width"))?;
        let h: usize = parts[1].parse().map_err(|_| msg("ffprobe height"))?;
        let (num, den) = parts[2].split_once('/').unwrap_or((parts[2], "1"));
        let (num, den): (f64, f64) = (
            num.parse().map_err(|_| msg("ffprobe rate"))?,
            den.parse().map_err(|_| msg("ffprobe rate"))?,
        );
        let out = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .output()
            .map_err(|e| msg(format!("ffmpeg decode: {e}")))?;
        if !out.status.success() {
            return Err(msg(format!("ffmpeg decode failed: {}", String::from_utf8_lossy(&out.stderr))));
        }
        let fsz = h * w * 3;
        let frames: Vec<Vec<u8>> = out.stdout.chunks_exact(fsz).map(<[u8]>::to_vec).collect();
        let times = (0..frames.len()).map(|i| i as f64 * den / num).collect();
        Ok(Self {
            height: h,
            width: w,
            times,
            frames,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MmAudioRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: u64,
    pub duration_s: f64,
    pub num_steps: usize,
    pub cfg_strength: f32,
    pub video: Option<VideoFrames>,
}

impl MmAudioRequest {
    pub fn t2a(prompt: impl Into<String>, seed: u64) -> Self {
        let p = MmAudioPreset::Large44kV2;
        Self {
            prompt: prompt.into(),
            negative_prompt: String::new(),
            seed,
            duration_s: f64::from(p.duration_s()),
            num_steps: p.default_steps(),
            cfg_strength: p.default_cfg(),
            video: None,
        }
    }
}

/// Seconds per stage.
#[derive(Debug, Clone, Default)]
pub struct MmAudioTimings {
    pub preprocess_s: f64,
    pub clip_s: f64,
    pub sync_s: f64,
    pub text_s: f64,
    pub dit_s: f64,
    pub vae_s: f64,
    pub vocoder_s: f64,
    pub total_s: f64,
}

#[derive(Debug, Clone)]
pub struct MmAudioOutput {
    /// Mono waveform in `[-1, 1]`.
    pub waveform: Vec<f32>,
    pub sample_rate: u32,
    /// The duration `load_video` settled on (sequence lengths follow it).
    pub duration_s: f64,
    pub timings: MmAudioTimings,
}

/// Upstream file layout under the weight root (`scripts/gpu/fetch-mmaudio.py`).
pub struct MmAudioPaths {
    pub root: PathBuf,
}

impl MmAudioPaths {
    fn st(&self, name: &str) -> PathBuf {
        self.root.join("safetensors").join(format!("{name}.safetensors"))
    }
    fn map(&self, name: &str) -> Result<WeightMap> {
        let p = self.st(name);
        if !p.is_file() {
            return Err(msg(format!("MMAudio: missing {}", p.display())));
        }
        WeightMap::open_files(&[p]).map_err(|e| msg(e.to_string()))
    }
    fn tokenizer(&self) -> PathBuf {
        self.root.join("DFN5B-CLIP-ViT-H-14-384").join("tokenizer.json")
    }
}

pub struct MmAudioPipeline {
    pub paths: MmAudioPaths,
    pub cfg: MmAudioConfig,
    pub preset: MmAudioPreset,
    pub dit: MmAudioTransformer,
    vae: MmAudioVaeDecoder,
    vocoder: BigVgan,
    clip_visual: ClipVisual,
    clip_text: ClipText,
    synchformer: Synchformer,
}

fn sync() -> Result<()> {
    crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))
}

fn dump_host(name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
    dump::host(name, shape, data).map_err(Into::into)
}

/// `FASTVIDEO_MMAUDIO_INJECT`: comma-separated stages whose reference inputs
/// replace ours: `pixels` (the preprocessed frames), `features` (CLIP, sync
/// and text features), `x1` (the sampled latent, for the VAE) and `mel`
/// (the reference's f32 decode `mm_mel_f32`, for the vocoder).
fn inject_on(stage: &str) -> bool {
    inject::enabled()
        && std::env::var("FASTVIDEO_MMAUDIO_INJECT")
            .unwrap_or_default()
            .split(',')
            .any(|s| s.trim() == stage)
}

impl MmAudioPipeline {
    /// `MMAUDIO_MODEL_PATH` wins over `explicit`.
    pub fn resolve_root(explicit: impl Into<PathBuf>) -> PathBuf {
        std::env::var_os("MMAUDIO_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| explicit.into())
    }

    pub fn load(root: impl Into<PathBuf>, preset: MmAudioPreset) -> Result<Self> {
        let paths = MmAudioPaths { root: root.into() };
        let cfg = MmAudioConfig::for_preset(preset);
        let dit = MmAudioTransformer::load(cfg.dit.clone(), &paths.map("mmaudio_large_44k_v2")?)?;
        let vae = MmAudioVaeDecoder::load(cfg.vae.clone(), &paths.map("vae_44k")?)?;
        let vocoder = BigVgan::load(cfg.vocoder.clone(), &paths.map("bigvgan_v2_44k")?)?;
        let clip_map = paths.map("clip_dfn5b_h14_384")?;
        let clip_visual = ClipVisual::load(ClipVisual::vit_h14(), &clip_map, 14, 27, 1024)?;
        let clip_text = ClipText::load(ClipText::h14_text(), &clip_map, 49408, 77)?;
        let synchformer = Synchformer::load(&paths.map("synchformer")?)?;
        Ok(Self {
            paths,
            cfg,
            preset,
            dit,
            vae,
            vocoder,
            clip_visual,
            clip_text,
            synchformer,
        })
    }

    /// `[1, 77, 1024]` text features.
    pub fn encode_text(&self, text: &str) -> Result<CudaTensor> {
        let ids = fastvideo_models::mmaudio::tokenize::tokenize(&self.paths.tokenizer(), text).map_err(msg)?;
        self.clip_text.encode(&ids).map_err(Into::into)
    }

    fn injected(name: &str, shape: Vec<usize>) -> Result<Option<CudaTensor>> {
        let n: usize = shape.iter().product();
        match inject::load_numel(name, n)? {
            Some(v) => Ok(Some(CudaTensor::from_vec(v, shape)?)),
            None => Ok(None),
        }
    }

    pub fn generate(&self, req: &MmAudioRequest) -> Result<MmAudioOutput> {
        let t_all = Instant::now();
        let mut tm = MmAudioTimings::default();
        let dcfg = &self.cfg.dit;

        // Frames -> the two pixel streams (host).
        let t0 = Instant::now();
        let mut duration = req.duration_s;
        let mut pixels = None;
        if let Some(v) = &req.video {
            let sel = select_frames(&v.times, req.duration_s);
            duration = sel.duration_sec;
            let pick = |idx: &[usize]| -> Vec<&[u8]> { idx.iter().map(|&i| v.frames[i].as_slice()).collect() };
            let cp = clip_pixels(&pick(&sel.clip_indices), v.height, v.width, true);
            let sp = sync_pixels(&pick(&sel.sync_indices), v.height, v.width);
            pixels = Some((sel.clip_indices.len(), cp, sel.sync_indices.len(), sp));
        }
        let seq = SequenceConfig::config_44k(duration);
        let (nl, nc, ns) = (seq.latent_seq_len(), seq.clip_seq_len(), seq.sync_seq_len());
        tm.preprocess_s = t0.elapsed().as_secs_f64();

        // Features.
        let (clip_f, sync_f) = match &pixels {
            Some((tc, cp, ts, sp)) => {
                let (cs, ss) = (CLIP_SIZE, SYNC_SIZE);
                dump_host("mm_clip_pixels", &[*tc, 3, cs, cs], cp)?;
                dump_host("mm_sync_pixels", &[*ts, 3, ss, ss], sp)?;
                let mut cpx = CudaTensor::from_vec(cp.clone(), vec![*tc, 3, cs, cs])?;
                let mut spx = CudaTensor::from_vec(sp.clone(), vec![*ts, 3, ss, ss])?;
                if inject_on("pixels") {
                    if let Some(t) = Self::injected("mm_clip_pixels", vec![*tc, 3, cs, cs])? {
                        cpx = t;
                    }
                    if let Some(t) = Self::injected("mm_sync_pixels", vec![*ts, 3, ss, ss])? {
                        spx = t;
                    }
                }
                let t1 = Instant::now();
                let clip = self.clip_visual.encode(&cpx)?.unsqueeze(0)?; // [1, T, 1024]
                sync()?;
                tm.clip_s = t1.elapsed().as_secs_f64();
                let t2 = Instant::now();
                let syncf = self.synchformer.encode(&spx)?;
                sync()?;
                tm.sync_s = t2.elapsed().as_secs_f64();
                dump::tensor("mm_clip_f", &clip)?;
                dump::tensor("mm_sync_f", &syncf)?;
                // generate() uses the model's sequence lengths.
                let clip = clip.narrow(1, 0, nc.min(clip.shape[1]))?;
                let syncf = syncf.narrow(1, 0, ns.min(syncf.shape[1]))?;
                (clip, syncf)
            }
            None => (self.dit.empty_clip(nc)?, self.dit.empty_sync(ns)?),
        };
        if clip_f.shape[1] != nc || sync_f.shape[1] != ns {
            return Err(msg(format!(
                "MMAudio: features {:?}/{:?} for clip {nc} / sync {ns}",
                clip_f.shape, sync_f.shape
            )));
        }
        let t3 = Instant::now();
        let mut text_f = self.encode_text(&req.prompt)?;
        let mut neg_f = self.encode_text(&req.negative_prompt)?;
        sync()?;
        tm.text_s = t3.elapsed().as_secs_f64();
        dump::tensor("mm_text_f", &text_f)?;
        dump::tensor("mm_neg_text_f", &neg_f)?;
        let (mut clip_f, mut sync_f) = (clip_f, sync_f);
        if inject_on("features") {
            let td = dcfg.text_dim;
            if let Some(t) = Self::injected("mm_clip_f", vec![1, nc, dcfg.clip_dim])? {
                clip_f = t;
            }
            if let Some(t) = Self::injected("mm_sync_f", vec![1, ns, dcfg.sync_dim])? {
                sync_f = t;
            }
            if let Some(t) = Self::injected("mm_text_f", vec![1, 77, td])? {
                text_f = t;
            }
            if let Some(t) = Self::injected("mm_neg_text_f", vec![1, 77, td])? {
                neg_f = t;
            }
        }

        // Sampler.
        let t4 = Instant::now();
        let cond = self.dit.preprocess(&clip_f, &sync_f, &text_f, nl)?;
        let empty = self
            .dit
            .preprocess(&self.dit.empty_clip(nc)?, &self.dit.empty_sync(ns)?, &neg_f, nl)?;
        let rot = self.dit.rotations(nl, nc)?;
        let ld = dcfg.latent_dim;
        let n = nl * ld;
        let mut x: Vec<f32> = match inject::load_numel("mm_x0", n)? {
            Some(v) => v.into_iter().map(bf16_round).collect(),
            None => {
                let mut rng = rand::rngs::StdRng::seed_from_u64(req.seed);
                (0..n)
                    .map(|_| {
                        let z: f32 = StandardNormal.sample(&mut rng);
                        bf16_round(z)
                    })
                    .collect()
            }
        };
        dump_host("mm_x0", &[1, nl, ld], &x)?;
        let times = euler_times(req.num_steps);
        let cfg = req.cfg_strength;
        for i in 0..req.num_steps {
            let t = times[i];
            let tb = bf16_round(t);
            let lat = CudaTensor::from_vec(x.clone(), vec![1, nl, ld])?;
            dump::set_blocks(i == 0);
            let fc = self.predict(&lat, tb, &cond, &rot)?;
            dump::set_blocks(false);
            let flow: Vec<f32> = if cfg < 1.0 {
                fc.iter().map(|&v| bf16_round(v)).collect()
            } else {
                let fu = self.predict(&lat, tb, &empty, &rot)?;
                if i == 0 {
                    dump_host("mm_flow_u_step00", &[1, nl, ld], &fu)?;
                }
                fc.iter()
                    .zip(&fu)
                    .map(|(&c, &u)| {
                        let a = bf16_round(cfg * bf16_round(c));
                        let b = bf16_round((1.0 - cfg) * bf16_round(u));
                        bf16_round(a + b)
                    })
                    .collect()
            };
            if i == 0 {
                dump_host("mm_flow_c_step00", &[1, nl, ld], &fc)?;
            }
            let dt = times[i + 1] - t;
            for (xv, fv) in x.iter_mut().zip(&flow) {
                *xv = bf16_round(*xv + bf16_round(dt * fv));
            }
            dump_host(&format!("mm_x_step{i:02}"), &[1, nl, ld], &x)?;
        }
        // unnormalize (bf16, in place upstream).
        let mut x1: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let c = i % ld;
                bf16_round(bf16_round(v * self.dit.latent_std[c]) + self.dit.latent_mean[c])
            })
            .collect();
        dump_host("mm_x1", &[1, nl, ld], &x1)?;
        if inject_on("x1") {
            if let Some(v) = inject::load_numel("mm_x1", n)? {
                x1 = v;
            }
        }
        tm.dit_s = t4.elapsed().as_secs_f64();

        let t5 = Instant::now();
        let z = CudaTensor::from_vec(x1, vec![1, nl, ld])?.transpose(1, 2)?;
        let mut mel = self.vae.decode(&z)?;
        sync()?;
        tm.vae_s = t5.elapsed().as_secs_f64();
        dump::tensor("mm_mel", &mel)?;
        if inject_on("mel") {
            if let Some(t) = Self::injected("mm_mel_f32", mel.shape.clone())? {
                mel = t;
            }
        }
        let t6 = Instant::now();
        let wav = self.vocoder.forward(&mel)?;
        let waveform = wav.host_cow()?.into_owned();
        tm.vocoder_s = t6.elapsed().as_secs_f64();
        dump_host("mm_wave", &[1, waveform.len()], &waveform)?;
        tm.total_s = t_all.elapsed().as_secs_f64();
        Ok(MmAudioOutput {
            waveform,
            sample_rate: self.cfg.sample_rate,
            duration_s: duration,
            timings: tm,
        })
    }

    fn predict(
        &self,
        lat: &CudaTensor,
        t: f32,
        cond: &Conditions,
        rot: &[(CudaTensor, CudaTensor); 2],
    ) -> Result<Vec<f32>> {
        Ok(self.dit.predict_flow(lat, t, cond, rot)?.host_cow()?.into_owned())
    }
}

/// 16-bit PCM mono WAV.
pub fn write_wav(path: &Path, pcm: &[f32], sample_rate: u32) -> Result<()> {
    let mut b = Vec::with_capacity(44 + pcm.len() * 2);
    let data_len = (pcm.len() * 2) as u32;
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| msg(format!("{}: {e}", path.display())))
}
