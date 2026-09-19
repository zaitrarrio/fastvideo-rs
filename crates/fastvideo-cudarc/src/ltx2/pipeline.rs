//! LTX-2 distilled text → audio + video, stage 1: eight Euler steps on a fixed
//! sigma list, one joint DiT forward per step, no guidance of any kind.
//!
//! The order of work is dictated by memory and by what the muxer needs:
//!
//! 1. **Text.** Gemma-3-12B streams through the device one layer at a time and
//!    is gone before anything else loads; the connectors (2.9 GB) are loaded,
//!    used once and dropped. What remains is two `[1, 1024, 3840]` contexts.
//! 2. **Denoise.** The DiT (37.8 GB bf16) loads, lifts the contexts to the
//!    stream widths once, and runs the eight steps. Latents and the Euler
//!    update are float32, as in the reference.
//! 3. **Decode, audio first.** The audio VAE and vocoder take milliseconds, so
//!    the WAV exists before the first video frame does — which is what lets
//!    ffmpeg be started with the track as an input and then be fed frames as
//!    the video VAE streams them, instead of muxing in a second pass.
//!
//! The reference round-trips each velocity through `x0` and back before the
//! Euler update even with guidance off (`pipeline_ltx2.py:1466-1467`); that is
//! the identity up to one float32 rounding and is not reproduced.
//! See docs/ports/ltx2.md §a, §f.

use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_models::ltx2::config::Ltx2Config;
use fastvideo_models::ltx2::Ltx2Schedule;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::llm::DecoderConfig;
use crate::wan::pipeline::{frames_to_rgb8, interleave_audio, write_wav, PipelineError, Result, VideoWriter};
use crate::wan::tensor::{CudaTensor, TensorError};
use crate::wan::weights::WeightMap;

use super::audio_vae::AudioDecoder;
use super::keys::Keys;
use super::text::{HiddenStack, PaddedPrompt, TextConnectors};
use super::transformer::{pack_video, unpack_video, Ltx2Transformer, Ropes, TextConditioning};
use super::vae::VideoDecoder;
use super::vocoder::Vocoder;

fn err(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Kernel launches return when queued; a phase's clock stops only once the
/// device has caught up, or the time lands on whoever synchronizes next.
fn sync() -> Result<()> {
    crate::wan::device::synchronize().map_err(|e| err(format!("device synchronize: {e}")))
}

/// Where the weights are.
#[derive(Debug, Clone)]
pub struct Ltx2Paths {
    /// A diffusers LTX-2 snapshot: `tokenizer/`, `text_encoder/`, `vae/`,
    /// `audio_vae/`, `vocoder/`. `Lightricks/LTX-2` and the distilled
    /// conversion carry identical files for all five.
    pub weights: PathBuf,
    /// The *distilled* DiT and connectors: `ltx-2-19b-distilled.safetensors`,
    /// or a diffusers root holding `transformer/` and `connectors/`.
    pub dit: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Ltx2Request {
    pub prompt: String,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub frame_rate: f64,
    pub seed: u64,
    pub output_dir: PathBuf,
    /// Also mux `output.mp4` (needs ffmpeg on the PATH).
    pub mp4: bool,
}

impl Ltx2Request {
    pub fn new(cfg: &Ltx2Config, prompt: impl Into<String>, output_dir: impl Into<PathBuf>) -> Self {
        let d = &cfg.defaults;
        Self {
            prompt: prompt.into(),
            height: d.height,
            width: d.width,
            num_frames: d.num_frames,
            frame_rate: d.frame_rate,
            seed: 10,
            output_dir: output_dir.into(),
            mp4: true,
        }
    }

    /// The model card's constraints: H and W divisible by 32, `8k + 1` frames.
    pub fn validate(&self) -> Result<()> {
        if self.height == 0 || self.width == 0 || !self.height.is_multiple_of(32) || !self.width.is_multiple_of(32) {
            return Err(err(format!("ltx2: {}x{} — height and width must be positive multiples of 32", self.width, self.height)));
        }
        if self.num_frames % 8 != 1 {
            return Err(err(format!("ltx2: {} frames — the frame count must be 8k + 1", self.num_frames)));
        }
        if self.frame_rate.is_nan() || self.frame_rate <= 0.0 || self.prompt.trim().is_empty() {
            return Err(err("ltx2: needs a positive frame rate and a non-empty prompt"));
        }
        Ok(())
    }
}

/// Wall-clock seconds per phase. Loads are summed; the rest are what the phase
/// itself took.
#[derive(Debug, Clone, Default)]
pub struct Ltx2Timings {
    pub load_s: f64,
    pub text_s: f64,
    pub denoise_s: f64,
    pub step_s: Vec<f64>,
    pub decode_audio_s: f64,
    pub decode_video_s: f64,
    /// Waiting for the frame writer / ffmpeg after the last frame was decoded.
    pub write_s: f64,
}

#[derive(Debug, Clone)]
pub struct Ltx2Output {
    pub frames: Vec<String>,
    pub mp4: Option<String>,
    pub wav: String,
    pub prompt_tokens: usize,
    pub video_tokens: usize,
    pub audio_tokens: usize,
    pub timings: Ltx2Timings,
}

/// Seeded float32 `N(0, 1)` latents, already packed for the DiT: video
/// `[1, F·H·W, 128]` drawn first, then audio `[1, L, 128]`, from one generator
/// — the order the reference pipeline draws them in. (Its draws come from
/// torch's generator; ours cannot reproduce those bits, only the contract.)
pub fn initial_noise(cfg: &Ltx2Config, grid: [usize; 3], audio_tokens: usize, seed: u64) -> Result<(CudaTensor, CudaTensor)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut draw = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.sample::<f32, _>(StandardNormal)).collect() };
    let c = cfg.transformer.in_channels;
    let [f, h, w] = grid;
    let video = CudaTensor::from_vec(draw(c * f * h * w), vec![1, c, f, h, w])?;
    let (ac, bins) = (cfg.audio_vae.latent_channels, cfg.audio_vae.latent_mel_bins());
    // [1, C, L, M] → [1, L, C·M]: feature index = channel · bins + bin.
    let audio = CudaTensor::from_vec(draw(ac * audio_tokens * bins), vec![1, ac, audio_tokens, bins])?;
    let audio = audio.permute(&[0, 2, 1, 3])?.reshape(vec![1, audio_tokens, ac * bins])?;
    Ok((pack_video(&video)?, audio))
}

/// Called after each step with `(step, video, audio, seconds)`.
pub type StepObserver<'a> = &'a mut dyn FnMut(usize, &CudaTensor, &CudaTensor, f64) -> Result<()>;

/// The distilled Euler loop: `x ← x + (σ_{i+1} - σ_i) · v(x, 1000·σ_i)` for both
/// streams with one forward per step. Inputs are packed latents at `σ_0`.
pub fn denoise(
    model: &Ltx2Transformer,
    text: &TextConditioning,
    ropes: &Ropes,
    schedule: &Ltx2Schedule,
    mut video: CudaTensor,
    mut audio: CudaTensor,
    mut observer: Option<StepObserver<'_>>,
) -> Result<(CudaTensor, CudaTensor)> {
    for i in 0..schedule.num_steps() {
        let timer = Instant::now();
        let (v_video, v_audio) = model.forward(&video, &audio, text, schedule.timestep_f32(i), ropes, None)?;
        let dt = schedule.dt(i) as f32;
        video = CudaTensor::lincomb(&[(1.0, &video), (dt, &v_video)])?;
        audio = CudaTensor::lincomb(&[(1.0, &audio), (dt, &v_audio)])?;
        sync()?;
        let secs = timer.elapsed().as_secs_f64();
        crate::wan::log::info(format_args!("ltx2 step {}/{} sigma {:.6} ({secs:.2}s)", i + 1, schedule.num_steps(), schedule.sigmas[i]));
        if let Some(obs) = observer.as_mut() {
            obs(i, &video, &audio, secs)?;
        }
    }
    Ok((video, audio))
}

/// The three decoders of the output side.
pub struct Decoders {
    pub video: VideoDecoder,
    pub audio: AudioDecoder,
    pub vocoder: Vocoder,
}

impl Decoders {
    pub fn load(weights: &Path, cfg: &Ltx2Config) -> Result<Self> {
        let open = |sub: &str| WeightMap::open(&weights.join(sub));
        Ok(Self {
            video: VideoDecoder::load(&open("vae")?, &cfg.vae)?,
            audio: AudioDecoder::load(&open("audio_vae")?, &cfg.audio_vae)?,
            vocoder: Vocoder::load(&open("vocoder")?, &cfg.vocoder)?,
        })
    }
}

/// What [`decode_and_write`] left on disk, and how long each part took.
#[derive(Debug, Clone)]
pub struct Written {
    pub frames: Vec<String>,
    pub mp4: Option<String>,
    pub wav: String,
    pub decode_audio_s: f64,
    pub decode_video_s: f64,
    pub write_s: f64,
}

/// Final packed latents → `audio.wav`, `frame-NNN.png` and (when `mp4`)
/// `output.mp4` in `dir`. Audio is decoded and written first so the muxer can
/// take it as an input while frames are still arriving.
pub fn decode_and_write(dec: &Decoders, video: &CudaTensor, audio: &CudaTensor, grid: [usize; 3], dir: &Path, frame_rate: f64, mp4: bool) -> Result<Written> {
    std::fs::create_dir_all(dir).map_err(|e| err(format!("{}: {e}", dir.display())))?;
    let timer = Instant::now();
    let wave = dec.vocoder.forward(&dec.audio.decode_packed(audio)?)?;
    let channels = wave.shape[1];
    let wav = dir.join("audio.wav");
    let rate = u32::try_from(dec.vocoder.sample_rate()).map_err(|_| err("ltx2: vocoder sample rate out of range"))?;
    let channel_count = u16::try_from(channels).map_err(|_| err("ltx2: too many audio channels"))?;
    write_wav(&wav, &interleave_audio(&wave.host_cow()?, channels)?, channel_count, rate)?;
    let decode_audio_s = timer.elapsed().as_secs_f64();

    let timer = Instant::now();
    let fps = frame_rate.round().max(1.0) as u32;
    let mut writer = VideoWriter::spawn_with_audio(dir, fps, mp4, Some(&wav))?;
    // The decoder's sink speaks tensor errors; carry the writer's own across.
    let mut sink_err: Option<PipelineError> = None;
    let mut sink = |offset: usize, frames: &CudaTensor| -> std::result::Result<(), TensorError> {
        let (h, w) = (frames.shape[2], frames.shape[3]);
        match frames_to_rgb8(frames).and_then(|rgb| writer.push(offset, h, w, rgb)) {
            Ok(()) => Ok(()),
            Err(e) => {
                let text = e.to_string();
                sink_err = Some(e);
                Err(TensorError::Message(text))
            }
        }
    };
    let decoded = dec.video.decode_streaming(&unpack_video(video, grid)?, &mut sink);
    match (decoded, sink_err) {
        (_, Some(e)) => return Err(e),
        (Err(e), None) => return Err(e.into()),
        (Ok(_), None) => {}
    }
    let decode_video_s = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let (frames, mp4_path) = writer.finish()?;
    Ok(Written {
        frames,
        mp4: mp4_path,
        wav: wav.to_string_lossy().into_owned(),
        decode_audio_s,
        decode_video_s,
        write_s: timer.elapsed().as_secs_f64(),
    })
}

/// The distilled DiT/connectors: the single file, or `component` under a
/// diffusers root (or the component folder itself).
pub fn open_distilled(path: &Path, component: &str) -> Result<WeightMap> {
    let map = if path.is_file() {
        WeightMap::open_files(&[path.to_path_buf()])?
    } else if path.join(component).is_dir() {
        WeightMap::open(&path.join(component))?
    } else if path.is_dir() {
        WeightMap::open(path)?
    } else {
        return Err(err(format!("{} is neither a .safetensors file nor a directory", path.display())));
    };
    Ok(map)
}

/// Prompt → the two connector contexts. Gemma is streamed and the connectors
/// are dropped on return, so nothing of the text path stays on the device but
/// the contexts. Returns `(contexts, real token count, load seconds)`.
pub fn encode_prompt(paths: &Ltx2Paths, cfg: &Ltx2Config, prompt: &str) -> Result<(super::text::TextContexts, usize, f64)> {
    let max_len = cfg.defaults.max_sequence_length;
    let padded = PaddedPrompt::tokenize(&paths.weights.join("tokenizer").join("tokenizer.json"), prompt, max_len)?;
    let gemma = WeightMap::open(&paths.weights.join("text_encoder"))?;
    // The product runs Gemma in bf16, where the embedding multiplier is 62.0.
    let stack = HiddenStack::encode(&gemma, &DecoderConfig::gemma3_12b_text().for_bf16_reference(), &padded)?;
    drop(gemma);
    let timer = Instant::now();
    let map = open_distilled(&paths.dit, "connectors")?;
    let connectors = TextConnectors::load(&map, &Keys::connectors(Keys::detect(&map)), &cfg.connectors)?;
    let load_s = timer.elapsed().as_secs_f64();
    Ok((connectors.forward(&stack, max_len)?, padded.real, load_s))
}

/// The whole stage-1 run.
pub fn generate(paths: &Ltx2Paths, cfg: &Ltx2Config, req: &Ltx2Request, observer: Option<StepObserver<'_>>) -> Result<Ltx2Output> {
    req.validate()?;
    let mut timings = Ltx2Timings::default();
    let grid = cfg.transformer.latent_grid(req.num_frames, req.height, req.width);
    let audio_tokens = cfg.transformer.audio_tokens(req.num_frames, req.frame_rate);
    if audio_tokens == 0 {
        return Err(err("ltx2: the clip is too short for a single audio latent"));
    }

    let timer = Instant::now();
    let (contexts, prompt_tokens, connector_load_s) = encode_prompt(paths, cfg, &req.prompt)?;
    sync()?;
    timings.text_s = timer.elapsed().as_secs_f64() - connector_load_s;
    timings.load_s += connector_load_s;

    let timer = Instant::now();
    let map = open_distilled(&paths.dit, "transformer")?;
    let model = Ltx2Transformer::load(&map, &Keys::transformer(Keys::detect(&map)), &cfg.transformer)?;
    sync()?;
    timings.load_s += timer.elapsed().as_secs_f64();

    let timer = Instant::now();
    let text = model.project_text(&contexts.video, &contexts.audio)?;
    drop(contexts);
    let ropes = Ropes::new(&cfg.transformer, grid, audio_tokens, req.frame_rate as f32)?;
    let (video, audio) = initial_noise(cfg, grid, audio_tokens, req.seed)?;
    let mut step_s = Vec::new();
    let mut observer = observer;
    let mut record = |i: usize, v: &CudaTensor, a: &CudaTensor, s: f64| -> Result<()> {
        step_s.push(s);
        match observer.as_mut() {
            Some(obs) => obs(i, v, a, s),
            None => Ok(()),
        }
    };
    let (video, audio) = denoise(&model, &text, &ropes, &Ltx2Schedule::distilled(), video, audio, Some(&mut record))?;
    timings.denoise_s = timer.elapsed().as_secs_f64();
    timings.step_s = step_s;
    // The decoders need ~3 GB; the DiT's 38 GB are not needed again.
    drop((model, text, ropes));

    let timer = Instant::now();
    let decoders = Decoders::load(&paths.weights, cfg)?;
    timings.load_s += timer.elapsed().as_secs_f64();
    let written = decode_and_write(&decoders, &video, &audio, grid, &req.output_dir, req.frame_rate, req.mp4)?;
    timings.decode_audio_s = written.decode_audio_s;
    timings.decode_video_s = written.decode_video_s;
    timings.write_s = written.write_s;
    Ok(Ltx2Output {
        frames: written.frames,
        mp4: written.mp4,
        wav: written.wav,
        prompt_tokens,
        video_tokens: grid.iter().product(),
        audio_tokens,
        timings,
    })
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::weights;
    use super::super::keys::Layout;
    use super::*;
    use fastvideo_models::ltx2::config::{ltx2_19b_distilled, Ltx2AudioVaeConfig, Ltx2TransformerConfig, Ltx2VideoVaeConfig, Ltx2VocoderConfig};

    /// A whole LTX-2 in miniature: the latent widths are tied together the way
    /// the real ones are (4 VAE channels = DiT in/out; 2 × 2 audio features).
    fn tiny() -> Ltx2Config {
        Ltx2Config {
            transformer: Ltx2TransformerConfig {
                in_channels: 4,
                out_channels: 4,
                num_attention_heads: 2,
                attention_head_dim: 8,
                cross_attention_dim: 16,
                audio_in_channels: 4,
                audio_out_channels: 4,
                audio_num_attention_heads: 2,
                audio_attention_head_dim: 4,
                audio_cross_attention_dim: 8,
                num_layers: 1,
                caption_channels: 12,
                timestep_proj_dim: 8,
                ..Ltx2TransformerConfig::ltx2_19b()
            },
            vae: Ltx2VideoVaeConfig {
                latent_channels: 4,
                decoder_block_out_channels: [16, 32, 64],
                decoder_layers_per_block: [1, 1, 1, 1],
                patch_size: 2,
                ..Ltx2VideoVaeConfig::ltx2_19b()
            },
            audio_vae: Ltx2AudioVaeConfig { base_channels: 2, num_res_blocks: 1, latent_channels: 2, mel_bins: 8, ..Ltx2AudioVaeConfig::ltx2_19b() },
            vocoder: Ltx2VocoderConfig {
                in_channels: 16,
                hidden_channels: 64,
                upsample_kernel_sizes: [7, 4, 4, 4, 4],
                upsample_factors: [3, 2, 2, 2, 2],
                ..Ltx2VocoderConfig::ltx2_19b()
            },
            ..ltx2_19b_distilled()
        }
    }

    fn model_and_inputs(cfg: &Ltx2Config) -> (Ltx2Transformer, TextConditioning, Ropes, [usize; 3], usize) {
        let model = Ltx2Transformer::load(&weights(), &Keys::transformer(Layout::Diffusers), &cfg.transformer).unwrap();
        let ctx = |k: f32| CudaTensor::from_vec((0..3 * 12).map(|i| (i as f32 * k).sin()).collect(), vec![1, 3, 12]).unwrap();
        let text = model.project_text(&ctx(0.3), &ctx(0.7)).unwrap();
        let (grid, audio_tokens) = ([2usize, 2, 2], 3usize);
        let ropes = Ropes::new(&cfg.transformer, grid, audio_tokens, 24.0).unwrap();
        (model, text, ropes, grid, audio_tokens)
    }

    #[test]
    fn noise_is_seeded_packed_and_video_is_drawn_first() {
        let cfg = tiny();
        let (v, a) = initial_noise(&cfg, [2, 2, 3], 5, 7).unwrap();
        assert_eq!((v.shape.clone(), a.shape.clone()), (vec![1, 12, 4], vec![1, 5, 4]));
        let (v2, a2) = initial_noise(&cfg, [2, 2, 3], 5, 7).unwrap();
        assert_eq!(&*v.host_cow().unwrap(), &*v2.host_cow().unwrap());
        assert_eq!(&*a.host_cow().unwrap(), &*a2.host_cow().unwrap());
        assert_ne!(&*v.host_cow().unwrap(), &*initial_noise(&cfg, [2, 2, 3], 5, 8).unwrap().0.host_cow().unwrap());
        // One generator: the first draw is video channel 0 at token 0, and the
        // audio draws start right after the 4·12 video values.
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let all: Vec<f32> = (0..48 + 20).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
        assert_eq!(v.host_cow().unwrap()[0], all[0]);
        // Packed token t, feature c ← unpacked [c, t]: token 1 feature 0 is draw 1.
        assert_eq!(v.host_cow().unwrap()[4], all[1]);
        // Audio [C=2, L=5, M=2] → token 0 = (c0 m0, c0 m1, c1 m0, c1 m1).
        let ah = a.host_cow().unwrap();
        assert_eq!((ah[0], ah[1], ah[2]), (all[48], all[49], all[48 + 10]));
        let mean = all.iter().sum::<f32>() / all.len() as f32;
        assert!(mean.abs() < 0.5);
    }

    /// The loop against the update written out on the host, step by step, with
    /// the model evaluated at `1000·σ_i`.
    #[test]
    fn denoise_is_eight_euler_steps_on_the_distilled_sigmas() {
        let cfg = tiny();
        let (model, text, ropes, grid, audio_tokens) = model_and_inputs(&cfg);
        let (video, audio) = initial_noise(&cfg, grid, audio_tokens, 3).unwrap();
        let schedule = Ltx2Schedule::distilled();
        let mut seen = Vec::new();
        let mut obs = |i: usize, v: &CudaTensor, _: &CudaTensor, _: f64| -> Result<()> {
            seen.push((i, v.host_cow()?.into_owned()));
            Ok(())
        };
        let (got_v, got_a) = denoise(&model, &text, &ropes, &schedule, video.clone(), audio.clone(), Some(&mut obs)).unwrap();
        assert_eq!(seen.iter().map(|(i, _)| *i).collect::<Vec<_>>(), (0..8).collect::<Vec<_>>());

        let sigmas = [1.0f64, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];
        let (mut xv, mut xa) = (video.host_cow().unwrap().into_owned(), audio.host_cow().unwrap().into_owned());
        for i in 0..8 {
            let t = (sigmas[i] as f32) * 1000.0;
            let (vv, va) = model
                .forward(&CudaTensor::from_vec(xv.clone(), video.shape.clone()).unwrap(), &CudaTensor::from_vec(xa.clone(), audio.shape.clone()).unwrap(), &text, t, &ropes, None)
                .unwrap();
            let dt = (sigmas[i + 1] - sigmas[i]) as f32;
            xv.iter_mut().zip(vv.host_cow().unwrap().iter()).for_each(|(x, v)| *x += dt * v);
            xa.iter_mut().zip(va.host_cow().unwrap().iter()).for_each(|(x, v)| *x += dt * v);
            for (a, b) in seen[i].1.iter().zip(&xv) {
                assert!((a - b).abs() < 1e-5, "step {i}: {a} vs {b}");
            }
        }
        for (a, b) in got_a.host_cow().unwrap().iter().zip(&xa) {
            assert!((a - b).abs() < 1e-5);
        }
        assert_eq!(got_v.shape, video.shape);
    }

    #[test]
    fn decoding_writes_a_wav_first_and_every_frame() {
        let cfg = tiny();
        let map = weights();
        let dec = Decoders {
            video: VideoDecoder::load(&map, &cfg.vae).unwrap(),
            audio: AudioDecoder::load(&map, &cfg.audio_vae).unwrap(),
            vocoder: Vocoder::load(&map, &cfg.vocoder).unwrap(),
        };
        let (video, audio) = initial_noise(&cfg, [2, 2, 2], 3, 1).unwrap();
        let dir = std::env::temp_dir().join(format!("fv-ltx2-pipeline-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let out = decode_and_write(&dec, &video, &audio, [2, 2, 2], &dir, 24.0, false).unwrap();
        // 2 latent frames → 9 frames of 32x32 (×8 by the VAE, ×2 by its patch… in this tiny config ×16).
        assert_eq!(out.frames.len(), 9);
        assert!(out.frames.iter().all(|p| Path::new(p).is_file()));
        assert!(out.frames[8].ends_with("frame-008.png"));
        assert_eq!(out.mp4, None);
        // 3 audio latents → 9 mel frames → 9·48 stereo samples of 16-bit PCM after a 44-byte header.
        let wav = std::fs::read(&out.wav).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(wav.len(), 44 + 9 * 48 * 2 * 2);
        assert_eq!(u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]), 24_000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn requests_outside_the_model_card_are_refused() {
        let cfg = ltx2_19b_distilled();
        let ok = Ltx2Request::new(&cfg, "a cat", "/tmp/x");
        assert!(ok.validate().is_ok());
        assert_eq!((ok.width, ok.height, ok.num_frames, ok.seed), (768, 512, 121, 10));
        assert!(Ltx2Request { height: 500, ..ok.clone() }.validate().is_err());
        assert!(Ltx2Request { num_frames: 120, ..ok.clone() }.validate().is_err());
        assert!(Ltx2Request { prompt: "  ".into(), ..ok.clone() }.validate().is_err());
        assert!(Ltx2Request { frame_rate: 0.0, ..ok }.validate().is_err());
    }
}
