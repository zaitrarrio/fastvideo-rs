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

use fastvideo_models::ltx2::config::{Ltx2Config, Ltx2ModelVersion};
use fastvideo_models::ltx2::{AncestralOpts, Ltx2Schedule};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::llm::{DecoderConfig, ResidentDecoder};
use crate::wan::pipeline::{frames_to_rgb8, interleave_audio, write_wav, PipelineError, Result, VideoWriter};
use crate::wan::tensor::{CudaTensor, TensorError};
use crate::wan::weights::WeightMap;

use super::audio_vae::AudioDecoder;
use super::keys::Keys;
use super::text::{HiddenStack, PaddedPrompt, TextConnectors};
use super::text_cache::{cache_key, weights_identity, CachedContexts, TextCache};
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
    /// Another root for `tokenizer/` and `text_encoder/` — a slim rewrite
    /// ([`super::slim`]). `None`: they are under `weights`.
    pub text: Option<PathBuf>,
}

impl Ltx2Paths {
    fn text_root(&self) -> &Path {
        self.text.as_deref().unwrap_or(&self.weights)
    }
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

/// Wall-clock seconds per phase. `load_s` is the pipeline's one-off load; the
/// rest are what each phase of this generation took (text includes loading the
/// connectors when the text path is streamed).
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
    pub text: TextReport,
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

fn apply_ancestral(
    sample: &CudaTensor,
    velocity: &CudaTensor,
    sigma: f64,
    sigma_next: f64,
    opts: AncestralOpts,
    rng: &mut rand::rngs::StdRng,
) -> Result<CudaTensor> {
    let mut x = sample.host_cow()?.into_owned();
    let v = velocity.host_cow()?;
    let denoised: Vec<f32> = x.iter().zip(v.iter()).map(|(s, vel)| Ltx2Schedule::denoised_from_velocity(*s, *vel, sigma)).collect();
    let noise = if opts.eta > 0.0 {
        Some((0..x.len()).map(|_| rng.sample::<f32, _>(StandardNormal)).collect::<Vec<f32>>())
    } else {
        None
    };
    Ltx2Schedule::ancestral_step(&mut x, &denoised, sigma, sigma_next, opts.eta, opts.s_noise, noise.as_deref());
    Ok(CudaTensor::from_vec(x, sample.shape.clone())?)
}

/// Distilled ancestral loop (LTX-2.5): velocity → `x0`, then
/// `EulerAncestralDiffusionStep` at each sigma.
pub fn denoise_ancestral(
    model: &Ltx2Transformer,
    text: &TextConditioning,
    ropes: &Ropes,
    schedule: &Ltx2Schedule,
    mut video: CudaTensor,
    mut audio: CudaTensor,
    opts: AncestralOpts,
    mut observer: Option<StepObserver<'_>>,
) -> Result<(CudaTensor, CudaTensor)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(opts.noise_seed);
    for i in 0..schedule.num_steps() {
        let timer = Instant::now();
        let sigma = schedule.sigmas[i];
        let sigma_next = schedule.sigmas[i + 1];
        let (v_video, v_audio) = model.forward(&video, &audio, text, schedule.timestep_f32(i), ropes, None)?;
        video = apply_ancestral(&video, &v_video, sigma, sigma_next, opts, &mut rng)?;
        audio = apply_ancestral(&audio, &v_audio, sigma, sigma_next, opts, &mut rng)?;
        sync()?;
        let secs = timer.elapsed().as_secs_f64();
        crate::wan::log::info(format_args!("ltx2 ancestral step {}/{} sigma {:.6} ({secs:.2}s)", i + 1, schedule.num_steps(), sigma));
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
///
/// Also accepts `--dit …/transformer` when looking up `connectors`: the sibling
/// `…/connectors` directory is used (Diffusers split pack).
pub fn open_distilled(path: &Path, component: &str) -> Result<WeightMap> {
    let map = if path.is_file() {
        WeightMap::open_files(&[path.to_path_buf()])?
    } else if path.join(component).is_dir() {
        WeightMap::open(&path.join(component))?
    } else if let Some(sibling) = path.parent().map(|p| p.join(component)).filter(|p| p.is_dir()) {
        // `path` is already a component dir (e.g. `…/transformer`); open the sibling.
        WeightMap::open(&sibling)?
    } else if path.is_dir() {
        WeightMap::open(path)?
    } else {
        return Err(err(format!("{} is neither a .safetensors file nor a directory", path.display())));
    };
    Ok(map)
}

/// Whether the conditioning cache answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOutcome {
    /// No cache directory was given, or this call bypassed it.
    Off,
    Hit,
    Miss,
}

impl CacheOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Hit => "hit",
            Self::Miss => "miss",
        }
    }
}

/// What the text phase of one generation did.
#[derive(Debug, Clone)]
pub struct TextReport {
    pub cache: CacheOutcome,
    /// `"cache"` when nothing was computed, else how Gemma ran: `"streamed"` or
    /// `"resident"` (whose first use includes loading it).
    pub mode: &'static str,
    pub tokens: usize,
    pub seconds: f64,
    pub key: Option<String>,
}

/// Where Gemma and the connectors live between prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextResidency {
    /// Resident when the device has room beside what is already loaded, else
    /// streamed. `FASTVIDEO_LTX2_TEXT=resident|streamed` overrides.
    #[default]
    Auto,
    /// One layer on the device at a time; every new prompt re-reads the
    /// checkpoint. The only choice on a card the DiT already fills.
    Streamed,
    /// Gemma (23.5 GB as bf16) and the connectors (2.9 GB) stay loaded: a new
    /// prompt costs one forward.
    Resident,
}

impl TextResidency {
    /// `Auto` resolved against the environment and the device. `free` is what
    /// the device reports *now* — after the DiT and decoders are in place;
    /// `needed` what the text path would hold plus working room.
    pub fn resolve(self, env: Option<&str>, free: Option<u64>, needed: u64) -> Result<bool> {
        let asked = match env.map(str::trim).filter(|v| !v.is_empty()) {
            None => self,
            Some(v) if v.eq_ignore_ascii_case("resident") => Self::Resident,
            Some(v) if v.eq_ignore_ascii_case("streamed") => Self::Streamed,
            Some(v) if v.eq_ignore_ascii_case("auto") => Self::Auto,
            Some(v) => return Err(err(format!("FASTVIDEO_LTX2_TEXT={v}: expected resident, streamed or auto"))),
        };
        Ok(match asked {
            Self::Resident => true,
            Self::Streamed => false,
            // No device to ask (a CPU run): stream, which is what is validated there.
            Self::Auto => free.is_some_and(|f| f >= needed),
        })
    }
}

/// Activations of a text forward and of the denoise that follows, plus
/// allocator slack: what must stay free once the resident encoder is loaded.
const RESIDENT_HEADROOM: u64 = 8 << 30;

struct ResidentText {
    gemma: ResidentDecoder,
    connectors: TextConnectors,
}

/// Prompt → connector contexts, with the conditioning cache in front.
///
/// On a hit neither Gemma nor the connectors are opened: the key is made from
/// the prompt, the tokenizer file and the weight *files'* identities
/// ([`weights_identity`]), which are computed once per encoder.
pub struct TextEncoder {
    paths: Ltx2Paths,
    cfg: Ltx2Config,
    cache: Option<TextCache>,
    residency: TextResidency,
    /// Built on the first cache miss: a run of hits never loads Gemma at all.
    resident: Option<ResidentText>,
    /// `(sha256-able tokenizer bytes, gemma identity, connector identity)`.
    identity: Option<(Vec<u8>, [u8; 32], [u8; 32])>,
}

impl TextEncoder {
    pub fn new(paths: &Ltx2Paths, cfg: &Ltx2Config, options: &PipelineOptions) -> Self {
        Self {
            paths: paths.clone(),
            cfg: cfg.clone(),
            cache: options.text_cache.clone().map(TextCache::new),
            residency: options.text_residency,
            resident: None,
            identity: None,
        }
    }

    fn tokenizer_path(&self) -> PathBuf {
        self.paths.text_root().join("tokenizer").join("tokenizer.json")
    }

    fn decoder_config(cfg: &Ltx2Config) -> DecoderConfig {
        // The product runs Gemma in bf16, where the embedding multiplier is 62.0.
        let dc = if cfg.gemma4.is_some() || cfg.version == Ltx2ModelVersion::V25 {
            DecoderConfig::gemma4_12b_text()
        } else {
            DecoderConfig::gemma3_12b_text()
        };
        dc.for_bf16_reference()
    }

    fn text_encoder_kind(cfg: &Ltx2Config) -> &'static [u8] {
        if cfg.gemma4.is_some() || cfg.version == Ltx2ModelVersion::V25 {
            b"gemma4-12b"
        } else {
            b"gemma3-12b"
        }
    }

    /// Device bytes a resident text path would add, from the shapes alone.
    fn resident_bytes(&self) -> u64 {
        let g = Self::decoder_config(&self.cfg);
        let c = &self.cfg.connectors;
        let width: u64 = if crate::wan::nn::bf16_linears_active() { 2 } else { 4 };
        let per_layer: u64 = (0..g.num_layers())
            .map(|i| {
                let (hq, hkv, dq, dkv, h) = (g.layer_heads(i), g.layer_kv_heads(i), g.layer_head_dim(i), g.layer_kv_head_dim(i), g.hidden);
                let v = if g.attention_k_eq_v { 0 } else { h * hkv * dkv };
                (h * hq * dq + h * hkv * dkv + v + hq * dq * h + 3 * h * g.intermediate) as u64
            })
            .sum();
        let d = c.inner_dim();
        let connector = (c.video_connector_num_layers + c.audio_connector_num_layers) * (4 * d * d + 8 * d * d) + c.text_proj_in_features() * c.caption_channels;
        (per_layer + connector as u64) * width
    }

    fn key(&mut self, prompt: &str) -> Result<String> {
        if self.identity.is_none() {
            let tokenizer = std::fs::read(self.tokenizer_path()).map_err(|e| err(format!("{}: {e}", self.tokenizer_path().display())))?;
            let text_dir = self.paths.text_root().join("text_encoder");
            // `Lightricks/LTX-2` keeps a stale duplicate shard set next to the real
            // one; when the real one is there, it alone identifies the encoder.
            let has_model_set = std::fs::read_dir(&text_dir)
                .map(|d| d.filter_map(|e| e.ok()).any(|e| e.file_name().to_string_lossy().starts_with("model-")))
                .unwrap_or(false);
            let gemma = weights_identity(&text_dir, has_model_set.then_some("model-"))?;
            let connectors = if self.paths.dit.is_file() {
                weights_identity(&self.paths.dit, None)?
            } else if self.paths.dit.join("connectors").is_dir() {
                weights_identity(&self.paths.dit.join("connectors"), None)?
            } else if let Some(sibling) = self.paths.dit.parent().map(|p| p.join("connectors")).filter(|p| p.is_dir()) {
                weights_identity(&sibling, None)?
            } else {
                weights_identity(&self.paths.dit, None)?
            };
            self.identity = Some((tokenizer, gemma, connectors));
        }
        let (tokenizer, gemma, connectors) = self.identity.as_ref().ok_or_else(|| err("ltx2: text identity missing"))?;
        Ok(cache_key(
            prompt,
            tokenizer,
            self.cfg.defaults.max_sequence_length,
            Self::text_encoder_kind(&self.cfg),
            gemma,
            connectors,
        ))
    }

    fn load_connectors(&self) -> Result<TextConnectors> {
        let map = open_distilled(&self.paths.dit, "connectors")?;
        Ok(TextConnectors::load(&map, &Keys::connectors(Keys::detect(&map)), &self.cfg.connectors)?)
    }

    /// Decide once, on the first prompt that actually has to be encoded, and
    /// load the resident models if that is the answer.
    fn ensure_backend(&mut self) -> Result<()> {
        if self.resident.is_some() || self.residency == TextResidency::Streamed {
            return Ok(());
        }
        let env = std::env::var("FASTVIDEO_LTX2_TEXT").ok();
        let free = crate::wan::device::free_memory().map(|(free, _)| free);
        let needed = self.resident_bytes() + RESIDENT_HEADROOM;
        if !self.residency.resolve(env.as_deref(), free, needed)? {
            crate::wan::log::info(format_args!("ltx2 text: streamed (free {:?} bytes, resident needs {needed})", free));
            self.residency = TextResidency::Streamed;
            return Ok(());
        }
        let timer = Instant::now();
        let cfg = Self::decoder_config(&self.cfg);
        let map = WeightMap::open(&self.paths.text_root().join("text_encoder"))?;
        let gemma = ResidentDecoder::load(&map, &cfg, cfg.num_layers())?;
        let connectors = self.load_connectors()?;
        sync()?;
        crate::wan::log::info(format_args!("ltx2 text: resident, {:.1} GiB on device, loaded in {:.1}s", gemma.device_bytes() as f64 / f64::from(1u32 << 30), timer.elapsed().as_secs_f64()));
        self.resident = Some(ResidentText { gemma, connectors });
        self.residency = TextResidency::Resident;
        Ok(())
    }

    /// Resident: one forward. Streamed: Gemma layer by layer, the connectors
    /// loaded, used and dropped.
    fn compute(&mut self, padded: &PaddedPrompt) -> Result<CachedContexts> {
        self.ensure_backend()?;
        let out = match &self.resident {
            Some(r) => r.connectors.forward(&HiddenStack::encode_resident(&r.gemma, padded)?, padded.max_len())?,
            None => {
                let gemma = WeightMap::open(&self.paths.text_root().join("text_encoder"))?;
                let stack = HiddenStack::encode(&gemma, &Self::decoder_config(&self.cfg), padded)?;
                drop(gemma);
                self.load_connectors()?.forward(&stack, padded.max_len())?
            }
        };
        Ok(CachedContexts { video: out.video, audio: out.audio })
    }

    /// `"resident"`, `"streamed"`, or `"undecided"` while only cache hits have
    /// been served.
    pub fn mode(&self) -> &'static str {
        match (self.resident.is_some(), self.residency) {
            (true, _) => "resident",
            (false, TextResidency::Streamed) => "streamed",
            _ => "undecided",
        }
    }

    /// `use_cache = false` neither reads nor writes the cache (a warm-up run
    /// must not turn the run it warms up for into a hit).
    pub fn encode(&mut self, prompt: &str, use_cache: bool) -> Result<(CachedContexts, TextReport)> {
        let timer = Instant::now();
        let padded = PaddedPrompt::tokenize(&self.tokenizer_path(), prompt, self.cfg.defaults.max_sequence_length)?;
        let key = if use_cache && self.cache.is_some() { Some(self.key(prompt)?) } else { None };
        // Taken out so the compute closure can borrow the encoder mutably.
        let cache = self.cache.take();
        let result = cached_or(cache.as_ref(), key.as_deref(), &padded, || self.compute(&padded));
        self.cache = cache;
        let (contexts, outcome) = result?;
        sync()?;
        let mode = if outcome == CacheOutcome::Hit { "cache" } else { self.mode() };
        Ok((contexts, TextReport { cache: outcome, mode, tokens: padded.real, seconds: timer.elapsed().as_secs_f64(), key }))
    }
}

/// The cache protocol, apart from what it caches: look up, else compute and
/// store. A cache that cannot be written is reported by the log, not by failing
/// a generation whose conditioning is already in hand.
fn cached_or(
    cache: Option<&TextCache>,
    key: Option<&str>,
    padded: &PaddedPrompt,
    compute: impl FnOnce() -> Result<CachedContexts>,
) -> Result<(CachedContexts, CacheOutcome)> {
    let (Some(cache), Some(key)) = (cache, key) else { return Ok((compute()?, CacheOutcome::Off)) };
    if let Some(hit) = cache.load(key, padded) {
        return Ok((hit, CacheOutcome::Hit));
    }
    let fresh = compute()?;
    if let Err(e) = cache.store(key, padded, &fresh.video, &fresh.audio) {
        crate::wan::log::info(format_args!("ltx2 text cache: not stored: {e}"));
    }
    Ok((fresh, CacheOutcome::Miss))
}

/// How a pipeline is set up.
#[derive(Debug, Clone, Default)]
pub struct PipelineOptions {
    /// Directory of the conditioning cache; `None` turns it off.
    pub text_cache: Option<PathBuf>,
    pub text_residency: TextResidency,
}

/// The models of a stage-1 run, loaded once and kept: the DiT (37.8 GB) and the
/// three decoders (~3 GB). A second generation pays for text, denoise and
/// decode only — and for text not even that when the prompt was seen before.
pub struct Ltx2Pipeline {
    cfg: Ltx2Config,
    model: Ltx2Transformer,
    decoders: Decoders,
    text: TextEncoder,
    /// Seconds [`Self::load`] took.
    pub load_s: f64,
}

impl Ltx2Pipeline {
    pub fn load(paths: &Ltx2Paths, cfg: &Ltx2Config, options: &PipelineOptions) -> Result<Self> {
        let timer = Instant::now();
        let map = open_distilled(&paths.dit, "transformer")?;
        let model = Ltx2Transformer::load(&map, &Keys::transformer(Keys::detect(&map)), &cfg.transformer)?;
        let decoders = Decoders::load(&paths.weights, cfg)?;
        sync()?;
        Ok(Self { cfg: cfg.clone(), model, decoders, text: TextEncoder::new(paths, cfg, options), load_s: timer.elapsed().as_secs_f64() })
    }

    /// One clip. `use_text_cache = false` bypasses the conditioning cache for
    /// this call only.
    pub fn generate(&mut self, req: &Ltx2Request, use_text_cache: bool, observer: Option<StepObserver<'_>>) -> Result<Ltx2Output> {
        req.validate()?;
        let cfg = &self.cfg;
        let mut timings = Ltx2Timings::default();
        let grid = cfg.transformer.latent_grid(req.num_frames, req.height, req.width);
        let audio_tokens = cfg.transformer.audio_tokens(req.num_frames, req.frame_rate);
        if audio_tokens == 0 {
            return Err(err("ltx2: the clip is too short for a single audio latent"));
        }

        let (contexts, text_report) = self.text.encode(&req.prompt, use_text_cache)?;
        timings.text_s = text_report.seconds;

        let timer = Instant::now();
        let text = self.model.project_text(&contexts.video, &contexts.audio)?;
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
        let schedule = Ltx2Schedule::distilled();
        let (video, audio) = if cfg.version == Ltx2ModelVersion::V25 {
            denoise_ancestral(
                &self.model,
                &text,
                &ropes,
                &schedule,
                video,
                audio,
                AncestralOpts { eta: 1.0, s_noise: 1.0, noise_seed: req.seed + 10_000 },
                Some(&mut record),
            )?
        } else {
            denoise(&self.model, &text, &ropes, &schedule, video, audio, Some(&mut record))?
        };
        timings.denoise_s = timer.elapsed().as_secs_f64();
        timings.step_s = step_s;
        drop((text, ropes));

        let written = decode_and_write(&self.decoders, &video, &audio, grid, &req.output_dir, req.frame_rate, req.mp4)?;
        timings.decode_audio_s = written.decode_audio_s;
        timings.decode_video_s = written.decode_video_s;
        timings.write_s = written.write_s;
        Ok(Ltx2Output {
            frames: written.frames,
            mp4: written.mp4,
            wav: written.wav,
            prompt_tokens: text_report.tokens,
            video_tokens: grid.iter().product(),
            audio_tokens,
            text: text_report,
            timings,
        })
    }
}

/// Load, generate one clip, drop everything.
pub fn generate(paths: &Ltx2Paths, cfg: &Ltx2Config, req: &Ltx2Request, options: &PipelineOptions, observer: Option<StepObserver<'_>>) -> Result<Ltx2Output> {
    req.validate()?;
    let mut pipeline = Ltx2Pipeline::load(paths, cfg, options)?;
    let mut out = pipeline.generate(req, true, observer)?;
    out.timings.load_s = pipeline.load_s;
    Ok(out)
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
                decoder_block_out_channels: vec![16, 32, 64],
                decoder_layers_per_block: vec![1, 1, 1, 1],
                patch_size: 2,
                ..Ltx2VideoVaeConfig::ltx2_19b()
            },
            audio_vae: Ltx2AudioVaeConfig { base_channels: 2, num_res_blocks: 1, latent_channels: 2, mel_bins: 8, ..Ltx2AudioVaeConfig::ltx2_19b() },
            vocoder: Ltx2VocoderConfig {
                in_channels: 16,
                hidden_channels: 64,
                upsample_kernel_sizes: vec![7, 4, 4, 4, 4],
                upsample_factors: vec![3, 2, 2, 2, 2],
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
    fn ancestral_one_step_matches_host_reference() {
        let cfg = tiny();
        let (model, text, ropes, grid, audio_tokens) = model_and_inputs(&cfg);
        let (video, audio) = initial_noise(&cfg, grid, audio_tokens, 11).unwrap();
        let schedule = Ltx2Schedule::distilled();
        let opts = AncestralOpts { eta: 1.0, s_noise: 1.0, noise_seed: 99 };
        let mut rng = rand::rngs::StdRng::seed_from_u64(opts.noise_seed);
        let i = 4usize;
        let sigma = schedule.sigmas[i];
        let sigma_next = schedule.sigmas[i + 1];
        let t = schedule.timestep_f32(i);
        let (vv, va) = model.forward(&video, &audio, &text, t, &ropes, None).unwrap();
        let mut want_v = video.host_cow().unwrap().into_owned();
        let mut want_a = audio.host_cow().unwrap().into_owned();
        let den_v: Vec<f32> = want_v
            .iter()
            .zip(vv.host_cow().unwrap().iter())
            .map(|(x, v)| Ltx2Schedule::denoised_from_velocity(*x, *v, sigma))
            .collect();
        let den_a: Vec<f32> = want_a
            .iter()
            .zip(va.host_cow().unwrap().iter())
            .map(|(x, v)| Ltx2Schedule::denoised_from_velocity(*x, *v, sigma))
            .collect();
        let noise_v: Vec<f32> = (0..want_v.len()).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
        let noise_a: Vec<f32> = (0..want_a.len()).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
        Ltx2Schedule::ancestral_step(&mut want_v, &den_v, sigma, sigma_next, opts.eta, opts.s_noise, Some(&noise_v));
        Ltx2Schedule::ancestral_step(&mut want_a, &den_a, sigma, sigma_next, opts.eta, opts.s_noise, Some(&noise_a));
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(opts.noise_seed);
        let got_v = apply_ancestral(&video, &vv, sigma, sigma_next, opts, &mut rng2).unwrap();
        let got_a = apply_ancestral(&audio, &va, sigma, sigma_next, opts, &mut rng2).unwrap();
        for (a, b) in got_v.host_cow().unwrap().iter().zip(&want_v) {
            assert!((a - b).abs() < 1e-5, "video: {a} vs {b}");
        }
        for (a, b) in got_a.host_cow().unwrap().iter().zip(&want_a) {
            assert!((a - b).abs() < 1e-5, "audio: {a} vs {b}");
        }
    }

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
    /// Hit, miss and bypass, with the expensive part replaced by a counter.
    #[test]
    fn the_cache_computes_once_per_prompt_and_a_bypass_neither_reads_nor_writes() {
        let dir = std::env::temp_dir().join(format!("fv-ltx2-pipeline-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = TextCache::new(&dir);
        let padded = PaddedPrompt::from_ids(&[2, 5, 9], 8).unwrap();
        let calls = std::cell::Cell::new(0usize);
        let compute = || -> Result<CachedContexts> {
            calls.set(calls.get() + 1);
            Ok(CachedContexts {
                video: CudaTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])?,
                audio: CudaTensor::from_vec(vec![5.0, 6.0], vec![1, 2, 1])?,
            })
        };
        // Bypassed: computed, nothing written.
        assert_eq!(cached_or(Some(&cache), None, &padded, compute).unwrap().1, CacheOutcome::Off);
        assert!(!cache.path("k").exists());
        assert_eq!(cached_or(Some(&cache), Some("k"), &padded, compute).unwrap().1, CacheOutcome::Miss);
        let (hit, outcome) = cached_or(Some(&cache), Some("k"), &padded, compute).unwrap();
        assert_eq!((outcome, calls.get()), (CacheOutcome::Hit, 2), "the third call must not compute");
        assert_eq!(&*hit.video.host_cow().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        // A truncated entry: a miss that recomputes and repairs the file.
        let whole = std::fs::read(cache.path("k")).unwrap();
        std::fs::write(cache.path("k"), &whole[..whole.len() - 5]).unwrap();
        assert_eq!(cached_or(Some(&cache), Some("k"), &padded, compute).unwrap().1, CacheOutcome::Miss);
        assert_eq!(std::fs::read(cache.path("k")).unwrap(), whole);
        // No cache at all.
        assert_eq!(cached_or(None, Some("k"), &padded, compute).unwrap().1, CacheOutcome::Off);
        assert_eq!(calls.get(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn auto_residency_needs_a_device_with_room_and_the_environment_overrides() {
        let gib = |n: u64| n << 30;
        let auto = TextResidency::Auto;
        assert!(auto.resolve(None, Some(gib(50)), gib(34)).unwrap(), "96 GB card, DiT loaded: room");
        assert!(!auto.resolve(None, Some(gib(20)), gib(34)).unwrap(), "48 GB card: stream");
        assert!(!auto.resolve(None, None, gib(34)).unwrap(), "no device: stream");
        assert!(auto.resolve(Some("resident"), Some(0), gib(34)).unwrap());
        assert!(!auto.resolve(Some(" Streamed "), Some(gib(90)), gib(34)).unwrap());
        assert!(TextResidency::Resident.resolve(Some("auto"), Some(gib(90)), gib(34)).unwrap());
        assert!(!TextResidency::Streamed.resolve(None, Some(gib(90)), gib(34)).unwrap());
        assert!(TextResidency::Resident.resolve(Some(""), None, gib(34)).unwrap(), "an empty variable is unset");
        assert!(auto.resolve(Some("maybe"), None, 0).is_err());
    }

    #[test]
    fn the_resident_text_path_is_sized_from_the_published_shapes() {
        let paths = Ltx2Paths { weights: "/nonexistent".into(), dit: "/nonexistent".into(), text: None };
        let enc = TextEncoder::new(&paths, &ltx2_19b_distilled(), &PipelineOptions::default());
        // CPU build: float32 widths. Gemma's 48 layers of projections are
        // 10.76 B parameters, the connectors 1.43 B (docs/ports/ltx2.md §g).
        let params = enc.resident_bytes() / 4;
        assert_eq!(params, 48 * 224_133_120 + (4 * 12 * 3840 * 3840 + 188_160 * 3840));
        assert_eq!(enc.mode(), "undecided");
        assert_eq!(paths.text_root(), Path::new("/nonexistent"));
        let slim = Ltx2Paths { text: Some("/slim".into()), ..paths };
        assert_eq!(slim.text_root(), Path::new("/slim"));
    }
}
