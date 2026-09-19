//! FastH3 text-to-audio+video, end to end: prompt, 8 DMD forwards, both
//! decoders, one mp4 with an audio track.
//!
//! **The loop is not the Wan DMD loop.** One forward serves both modalities,
//! each on its own shifted schedule (video 10, audio 3). The model predicts a
//! data-ward velocity, so `x0 = x + sigma * v` — a plus — and the step is the
//! deterministic blend `x' = r x + (1 - r) x0` with `r = sigma' / sigma`; no
//! noise is re-injected between rungs (`eta = 0`). `sigma` for the `x0`
//! estimate is recovered from the float32 timestep while `r` comes off the
//! sigma grid, as the reference keeps them ([`H3Schedule::step_coeffs`]).
//!
//! **What is resident.** [`H3Pipeline`] loads once and generates many times:
//! the DiT (41 GB as bf16 with the VSA gates, 37 GB dense), the text refiner
//! (1.6 GB), the video VAE (4.9 GB as bf16 linears) and the audio VAE (0.3 GB)
//! stay on the device, about 48 GB. A 5 s clip adds ~12 GB of activations
//! (measured peak 53.6 GB with the decoders loaded late), a 15 s clip ~25 GB.
//!
//! **The text encoder is the one component with a choice**
//! ([`TextEncoderChoice`]), because it is 50 GB of weights for milliseconds of
//! compute:
//!
//! * *streamed*: nothing resident, ~10 s of host-to-device traffic
//!   per **new** prompt; the conditioning cache makes a repeated prompt free;
//! * *resident* (what `Auto` picks on a large empty card): pays the load once. On a 96 GB card the arithmetic decides the
//!   precision: DiT 41 + decoders 7 + activations 12 = 60 GB leaves ~36 GB, so
//!   Qwen as FP8-E4M3 (~25 GB) fits at 5 s and bf16 Qwen (50 GB) does not fit at
//!   any length. At 15 s (activations ~25 GB) even FP8 leaves only ~10 GB of
//!   slack, so resident mode is for short clips or for a process that drops it
//!   before denoising. The resident encoder itself lives in [`crate::llm`];
//!   this module only holds the seam ([`HiddenStateEncoder`]).
//!
//! Audio decodes before video: it takes milliseconds and the WAV has to exist
//! before ffmpeg is spawned; the video VAE then streams 17-frame chunks
//! straight into the muxer.

use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_models::h3::config::{
    H3AudioVaeConfig, H3Geometry, H3TransformerConfig, H3VideoVaeConfig, H3_AUDIO_CHANNELS, H3_FPS,
};
use fastvideo_models::h3::packing::{patchify, H3PackedLayout};
use fastvideo_models::h3::schedule::{H3JointSchedule, H3Schedule};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::audio_vae::H3AudioDecoder;
use super::text::{CacheStatus, HiddenStateEncoder};
use super::transformer::{AttnMode, DeviceLayout, H3TextRefiner, H3Transformer};
use super::vae::H3VideoDecoder;
use super::vsa::H3Vsa;
use crate::wan::pipeline::{frames_to_rgb8, interleave_audio, write_wav, PipelineError, Result, VideoWriter};
use crate::wan::tensor::{CudaTensor, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct H3Request {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    /// Aligned up to `17 n + 5`; 5 to 15 seconds at 24 fps.
    pub num_frames: usize,
    /// Write `output.mp4` (needs ffmpeg) next to the PNG frames.
    pub mp4: bool,
}

impl H3Request {
    /// The default 16:9 canvas (768 x 1344) for a whole number of seconds.
    pub fn seconds(prompt: impl Into<String>, seconds: usize, seed: u64) -> std::result::Result<Self, String> {
        let g = H3Geometry::default_16x9(seconds)?;
        Ok(Self { prompt: prompt.into(), seed, height: g.height, width: g.width, num_frames: g.num_frames, mp4: true })
    }
}

/// How prompts are encoded; see the module docs for the memory arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextEncoderChoice {
    /// [`Self::ResidentFp8`] when the card is large and empty enough
    /// ([`AUTO_RESIDENT_FREE_BYTES`] free before anything loads), else streamed.
    #[default]
    Auto,
    /// One decoder layer on the device at a time, per prompt. Nothing resident.
    Streamed,
    /// Layers 0..=49 resident as bf16: 50 GB. Does not fit beside the DiT on a
    /// 96 GB card; for an encode-only process or a larger device.
    ResidentBf16,
    /// Resident with weight-only FP8 rows (24.4 GB): fits beside the DiT at 5 s.
    ResidentFp8,
}

/// Free device memory at which `Auto` keeps the encoder resident: DiT 41 +
/// decoders 7 + FP8 Qwen 24.4 + ~12 GB of 5 s activations = 85 GB.
pub const AUTO_RESIDENT_FREE_BYTES: u64 = 85_000_000_000;

impl TextEncoderChoice {
    pub fn parse(name: &str) -> std::result::Result<Self, String> {
        match name {
            "auto" => Ok(Self::Auto),
            "streamed" => Ok(Self::Streamed),
            "resident-bf16" => Ok(Self::ResidentBf16),
            "resident-fp8" => Ok(Self::ResidentFp8),
            other => Err(format!("unknown text encoder '{other}' (auto|streamed|resident-fp8|resident-bf16)")),
        }
    }

    /// `Auto` decided by the free device memory measured before any load
    /// (`None`: no device, or it would not say). Explicit choices pass through.
    pub fn resolve(self, free_bytes: Option<u64>) -> Self {
        match self {
            Self::Auto if free_bytes.is_some_and(|f| f >= AUTO_RESIDENT_FREE_BYTES) => Self::ResidentFp8,
            Self::Auto => Self::Streamed,
            explicit => explicit,
        }
    }

    fn precision(self) -> Option<crate::llm::WeightPrecision> {
        match self {
            Self::ResidentBf16 => Some(crate::llm::WeightPrecision::Native),
            Self::ResidentFp8 => Some(crate::llm::WeightPrecision::Fp8Rows),
            Self::Auto | Self::Streamed => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct H3PipelineOptions {
    /// Dense attention without the compression gate: the parity mode the
    /// diffusers oracle judges. The checkpoint was distilled with VSA-H3, so
    /// the default (`false`) is what it should be served with.
    pub dense: bool,
    /// Memoize the precomputed AdaLN table here (155 MB). Building it reads
    /// 26 GB of projections nothing else needs; the file is validated against
    /// the checkpoint and the ladder.
    pub adaln_cache: Option<PathBuf>,
    /// Root holding `tokenizer/` and `text_encoder/` when it is not the
    /// snapshot itself, e.g. the slim re-pack of [`super::slim`].
    pub text_root: Option<PathBuf>,
    /// Conditioning cache directory ([`super::text_cache`]); `None` disables it.
    pub text_cache: Option<PathBuf>,
    pub text_encoder: TextEncoderChoice,
}

#[derive(Debug, Clone, Default)]
pub struct H3Timings {
    pub text_s: f64,
    pub refine_s: f64,
    pub denoise_s: f64,
    pub step_s: Vec<f64>,
    pub audio_decode_s: f64,
    pub video_decode_s: f64,
    /// The tail of encoding that outlives the decode, not the whole encode.
    pub write_s: f64,
}

#[derive(Debug, Clone)]
pub struct H3Output {
    pub geometry: H3Geometry,
    pub text_tokens: usize,
    pub text_cache: CacheStatus,
    /// `"streamed"`, `"resident-..."`, or `"cache"` when no encoder ran.
    pub text_encoder: &'static str,
    pub sequence_length: usize,
    pub frames: usize,
    pub frame_paths: Vec<String>,
    pub mp4: Option<String>,
    pub wav: PathBuf,
    pub timings: H3Timings,
}

/// The request's starting noise, from one generator in the reference's order:
/// video `[24, T, H, W]` first (then patchified), audio rows `[2 Na, 32]`
/// second, drawn directly in row layout. Torch's CPU sampler is not
/// reproduced, so a seed names a different (equally valid) sample than it does
/// upstream; parity runs inject the oracle's noise instead.
pub fn seeded_noise(cfg: &H3TransformerConfig, geometry: &H3Geometry, seed: u64) -> std::result::Result<(Vec<f32>, Vec<f32>), String> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let shape = [cfg.in_channels, geometry.latent_frames, geometry.latent_height, geometry.latent_width];
    let video: Vec<f32> = (0..shape.iter().product::<usize>()).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
    let audio: Vec<f32> = (0..geometry.audio_rows() * cfg.audio_in_channels).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
    Ok((patchify(&video, shape, cfg.patch_size)?, audio))
}

/// One Euler step of `MiniMaxH3Scheduler.step` on device rows, in the
/// reference's order of operations: `x0 = x + sigma_t v`, then
/// `r x + (1 - r) x0`.
pub fn scheduler_step(schedule: &H3Schedule, step: usize, sample: &CudaTensor, velocity: &CudaTensor) -> Result<CudaTensor> {
    let c = schedule.step_coeffs(step).map_err(msg)?;
    let denoised = CudaTensor::lincomb(&[(1.0, sample), (c.sigma_from_timestep, velocity)])?;
    Ok(CudaTensor::lincomb(&[(c.ratio, sample), (1.0 - c.ratio, &denoised)])?)
}

/// The 8-forward ladder. `observe(step, video_rows, audio_rows)` sees the
/// state after each step; the last call holds the clean latent rows.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    model: &H3Transformer,
    layout: &DeviceLayout,
    text_refined: &CudaTensor,
    video_rows: CudaTensor,
    audio_rows: CudaTensor,
    schedule: &H3JointSchedule,
    mode: AttnMode<'_>,
    observe: &mut dyn FnMut(usize, &CudaTensor, &CudaTensor) -> Result<()>,
) -> Result<(CudaTensor, CudaTensor)> {
    let (mut video, mut audio) = (video_rows, audio_rows);
    for step in 0..schedule.num_steps() {
        let (v_video, v_audio) = model.forward(step, &video, &audio, text_refined, layout, mode, None)?;
        video = scheduler_step(&schedule.video, step, &video, &v_video)?;
        audio = scheduler_step(&schedule.audio, step, &audio, &v_audio)?;
        observe(step, &video, &audio)?;
    }
    Ok((video, audio))
}

/// `unpatchify_video_tokens` on the device: rows `[T h w, C pt ph pw]`
/// (channel-major features) to `[1, C, T, H, W]`. The reference's 8-D permute
/// is past the device permute's rank limit; with the singleton temporal patch
/// dropped it is rank 6.
pub fn unpatchify_rows(rows: &CudaTensor, channels: usize, grid: (usize, usize, usize), patch: [usize; 3]) -> std::result::Result<CudaTensor, TensorError> {
    let [pt, ph, pw] = patch;
    let (t, h, w) = grid;
    if pt != 1 {
        return Err(TensorError::Message(format!("unpatchify: temporal patch {pt} is not supported (H3 uses 1)")));
    }
    if rows.shape != [t * h * w, channels * ph * pw] {
        return Err(TensorError::Message(format!("unpatchify: rows {:?} for grid {grid:?}, {channels} channels, patch {patch:?}", rows.shape)));
    }
    rows.reshape(vec![t, h, w, channels, ph, pw])?.permute(&[3, 0, 1, 4, 2, 5])?.reshape(vec![1, channels, t, h * ph, w * pw])
}

/// Seconds spent in [`H3Pipeline::load`], by component.
#[derive(Debug, Clone, Default)]
pub struct H3LoadTimings {
    /// Zero when the encoder is streamed.
    pub text_encoder_s: f64,
    pub refiner_s: f64,
    pub dit_s: f64,
    pub video_vae_s: f64,
    pub audio_vae_s: f64,
}

/// Everything but the text encoder, resident: load once, generate many times.
pub struct H3Pipeline {
    root: PathBuf,
    options: H3PipelineOptions,
    cfg: H3TransformerConfig,
    schedule: H3JointSchedule,
    refiner: H3TextRefiner,
    model: H3Transformer,
    video_vae: H3VideoDecoder,
    audio_vae: H3AudioDecoder,
    text_encoder: Option<Box<dyn HiddenStateEncoder>>,
    pub load_timings: H3LoadTimings,
}

impl H3Pipeline {
    /// `root` is the FastH3 snapshot (`transformer/`, `vae/`, `audio_vae/`, and
    /// `tokenizer/` + `text_encoder/` unless `options.text_root` says otherwise).
    pub fn load(root: &Path, options: H3PipelineOptions) -> Result<Self> {
        let cfg = H3TransformerConfig::fasth3_8step();
        let schedule = H3JointSchedule::fasth3_8step();
        let mut load_timings = H3LoadTimings::default();
        let timed = |slot: &mut f64, timer: Instant| *slot = timer.elapsed().as_secs_f64();

        // The resident encoder goes first: its FP8 quantization runs on the
        // device with f32 transients (524 MB for the widest matrix), which
        // should happen while the card is otherwise empty.
        let free = crate::wan::device::free_memory().map(|(free, _)| free);
        let mut options = options;
        options.text_encoder = options.text_encoder.resolve(free);
        let timer = Instant::now();
        let text_encoder: Option<Box<dyn HiddenStateEncoder>> = match options.text_encoder.precision() {
            Some(precision) => {
                let text_root = options.text_root.as_deref().unwrap_or(root);
                Some(Box::new(super::text::load_resident_encoder(text_root, precision)?))
            }
            None => None,
        };
        if text_encoder.is_some() {
            timed(&mut load_timings.text_encoder_s, timer);
        }

        let map = WeightMap::open(&root.join("transformer"))?;
        let timer = Instant::now();
        let refiner = H3TextRefiner::load(&cfg, &map)?;
        timed(&mut load_timings.refiner_s, timer);
        let timer = Instant::now();
        let model = H3Transformer::load_cached(cfg.clone(), &map, &schedule, !options.dense, options.adaln_cache.as_deref())?;
        timed(&mut load_timings.dit_s, timer);
        let timer = Instant::now();
        let video_vae = H3VideoDecoder::load(H3VideoVaeConfig::fasth3_8step(), &WeightMap::open(&root.join("vae"))?)?;
        timed(&mut load_timings.video_vae_s, timer);
        let timer = Instant::now();
        let audio_vae = H3AudioDecoder::load(H3AudioVaeConfig::fasth3_8step(), &WeightMap::open(&root.join("audio_vae"))?)?;
        timed(&mut load_timings.audio_vae_s, timer);
        Ok(Self { root: root.to_path_buf(), options, cfg, schedule, refiner, model, video_vae, audio_vae, text_encoder, load_timings })
    }

    /// Replace the text encoder (tests, or an encoder built elsewhere). It is
    /// consulted only on a conditioning-cache miss.
    pub fn with_text_encoder(mut self, encoder: Box<dyn HiddenStateEncoder>) -> Self {
        if !matches!(self.options.text_encoder, TextEncoderChoice::ResidentBf16 | TextEncoderChoice::ResidentFp8) {
            self.options.text_encoder = TextEncoderChoice::ResidentFp8;
        }
        self.text_encoder = Some(encoder);
        self
    }

    /// `(kind, device bytes)` of the encoder a cache miss will use.
    pub fn text_encoder(&self) -> (&'static str, u64) {
        self.text_encoder.as_ref().map_or(("streamed", 0), |e| (e.kind(), e.resident_bytes()))
    }

    /// Encode `prompt` with the resident encoder AND by streaming, bypassing
    /// the cache, and return `(rel_l2, cosine)` of resident against streamed.
    /// What matters about a quantized encoder is how far it moves the
    /// conditioning; this is that number for the prompt at hand. `None` when
    /// the encoder is streamed anyway. Costs one streamed encode (~10 s).
    pub fn text_encoder_drift(&self, prompt: &str) -> Result<Option<(f64, f64)>> {
        let Some(resident) = &self.text_encoder else { return Ok(None) };
        let text_root = self.options.text_root.as_deref().unwrap_or(&self.root);
        let ours = super::text::encode_prompt_with(text_root, prompt, None, Some(resident.as_ref()))?;
        let streamed = super::text::encode_prompt_with(text_root, prompt, None, None)?;
        let (a, b) = (ours.hidden.host_cow()?, streamed.hidden.host_cow()?);
        let (mut err, mut aa, mut bb, mut ab) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            let (x, y) = (f64::from(*x), f64::from(*y));
            err += (x - y) * (x - y);
            aa += x * x;
            bb += y * y;
            ab += x * y;
        }
        Ok(Some(((err / bb.max(1e-300)).sqrt(), ab / (aa * bb).sqrt().max(1e-300))))
    }

    pub fn options(&self) -> &H3PipelineOptions {
        &self.options
    }

    /// Generate one clip into `out_dir`: `frame-NNN.png`, `audio.wav`, and
    /// `output.mp4` with the audio muxed in.
    pub fn generate(&self, request: &H3Request, out_dir: &Path) -> Result<H3Output> {
        let cfg = &self.cfg;
        let geometry = H3Geometry::new(request.height, request.width, request.num_frames).map_err(msg)?;
        let mut timings = H3Timings::default();

        // --- text: cache, else resident, else ~50 GB streamed one layer at a time ---
        let timer = Instant::now();
        let resident = match (self.options.text_encoder.precision(), &self.text_encoder) {
            (None, _) => None,
            (Some(_), Some(encoder)) => Some(encoder.as_ref()),
            (Some(_), None) => return Err(msg(format!("text encoder {:?} was requested but is not loaded", self.options.text_encoder))),
        };
        let text_root = self.options.text_root.as_deref().unwrap_or(&self.root);
        let text = super::text::encode_prompt_with(text_root, &request.prompt, self.options.text_cache.as_deref(), resident)?;
        timings.text_s = timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let text_refined = self.refiner.forward(&text.hidden)?;
        timings.refine_s = timer.elapsed().as_secs_f64();

        // --- DiT --------------------------------------------------------------------
        let layout = H3PackedLayout::from_geometry(&geometry, text.ids.len()).map_err(msg)?;
        let sequence_length = layout.sequence_length();
        let vsa = if self.options.dense { None } else { Some(H3Vsa::new(&layout, cfg.num_attention_heads, cfg.attention_head_dim, super::vsa::H3VsaConfig::fasth3_8step())?) };
        let mode = vsa.as_ref().map_or(AttnMode::Dense, AttnMode::Vsa);
        let layout = DeviceLayout::new(cfg, layout)?;
        let (video_noise, audio_noise) = seeded_noise(cfg, &geometry, request.seed).map_err(msg)?;
        let video_rows = CudaTensor::from_vec(video_noise, vec![geometry.video_rows(), cfg.video_patch_dim()])?.to_device()?;
        let audio_rows = CudaTensor::from_vec(audio_noise, vec![geometry.audio_rows(), cfg.audio_in_channels])?.to_device()?;

        let timer = Instant::now();
        let mut last = Instant::now();
        let steps = self.schedule.num_steps();
        let mut step_s = Vec::with_capacity(steps);
        let (video_rows, audio_rows) = denoise(&self.model, &layout, &text_refined, video_rows, audio_rows, &self.schedule, mode, &mut |step, _, _| {
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            step_s.push(last.elapsed().as_secs_f64());
            crate::wan::log::info(format_args!("h3 step {}/{steps}: {:.1}s", step + 1, step_s[step]));
            last = Instant::now();
            Ok(())
        })?;
        timings.denoise_s = timer.elapsed().as_secs_f64();
        timings.step_s = step_s;
        drop((vsa, layout, text_refined));

        // --- audio first: the WAV must exist before the muxer starts -------------------
        let timer = Instant::now();
        let sample_rate = self.audio_vae.config().sampling_rate as u32;
        let wave = self.audio_vae.decode_rows(&audio_rows, H3_AUDIO_CHANNELS)?.host_cow()?.into_owned();
        let wav = out_dir.join("audio.wav");
        write_wav(&wav, &interleave_audio(&wave, H3_AUDIO_CHANNELS)?, H3_AUDIO_CHANNELS as u16, sample_rate)?;
        timings.audio_decode_s = timer.elapsed().as_secs_f64();

        // --- video: chunks go to the writer as they decode -------------------------------
        let latents = unpatchify_rows(&video_rows, cfg.in_channels, geometry.token_grid, cfg.patch_size)?;
        let timer = Instant::now();
        let mut writer = VideoWriter::spawn_with_audio(out_dir, H3_FPS as u32, request.mp4, Some(&wav))?;
        // The sink speaks TensorError; carry the writer's own error out beside it.
        let mut writer_error: Option<PipelineError> = None;
        let decoded = self.video_vae.decode_streaming(&latents, &mut |offset, frames| {
            let (h, w) = (frames.shape[2], frames.shape[3]);
            let pushed = frames_to_rgb8(frames).and_then(|rgb| writer.push(offset, h, w, rgb));
            pushed.map_err(|e| {
                let text = e.to_string();
                writer_error = Some(e);
                TensorError::Message(text)
            })
        });
        let frames = match (decoded, writer_error) {
            (_, Some(e)) => return Err(e),
            (r, None) => r?,
        };
        timings.video_decode_s = timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let (frame_paths, mp4) = writer.finish()?;
        timings.write_s = timer.elapsed().as_secs_f64();

        Ok(H3Output {
            geometry,
            text_tokens: text.ids.len(),
            text_cache: text.cache,
            text_encoder: text.encoder,
            sequence_length,
            frames,
            frame_paths,
            mp4,
            wav,
            timings,
        })
    }
}

/// Load, generate one clip, drop everything.
pub fn generate(root: &Path, options: H3PipelineOptions, request: &H3Request, out_dir: &Path) -> Result<H3Output> {
    H3Pipeline::load(root, options)?.generate(request, out_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::h3::packing::unpatchify;

    #[test]
    fn device_unpatchify_is_the_inverse_of_the_host_patchify() {
        let (c, t, h, w) = (3usize, 2usize, 4usize, 6usize);
        let lat: Vec<f32> = (0..c * t * h * w).map(|v| v as f32).collect();
        let rows = patchify(&lat, [c, t, h, w], [1, 2, 2]).unwrap();
        let got = unpatchify_rows(&CudaTensor::from_vec(rows.clone(), vec![t * 2 * 3, c * 4]).unwrap(), c, (t, 2, 3), [1, 2, 2]).unwrap();
        assert_eq!(got.shape, vec![1, c, t, h, w]);
        assert_eq!(&*got.host_cow().unwrap(), &lat[..]);
        assert_eq!(unpatchify(&rows, [c, t, h, w], [1, 2, 2]).unwrap(), lat);
    }

    #[test]
    fn a_step_is_the_reference_blend_and_the_last_one_lands_on_x0() {
        let schedule = H3JointSchedule::fasth3_8step();
        let x = vec![0.5f32, -1.25, 2.0];
        let v = vec![1.0f32, 0.25, -0.5];
        let (xt, vt) = (CudaTensor::from_vec(x.clone(), vec![3, 1]).unwrap(), CudaTensor::from_vec(v.clone(), vec![3, 1]).unwrap());
        for (sched, step) in [(&schedule.video, 0usize), (&schedule.audio, 4), (&schedule.video, 7)] {
            let got = scheduler_step(sched, step, &xt, &vt).unwrap();
            let want = sched.step(step, &x, &v).unwrap();
            for (g, w) in got.host_cow().unwrap().iter().zip(&want) {
                assert!((g - w).abs() <= 1e-6 * w.abs().max(1.0), "{g} vs {w}");
            }
        }
        // sigma[8] = 0: the last step returns x0 = x + sigma * v (note the plus).
        let last = scheduler_step(&schedule.video, 7, &xt, &vt).unwrap();
        let sigma = 1.0f32 - schedule.video.timesteps[7];
        assert!((last.host_cow().unwrap()[0] - (0.5 + sigma * 1.0)).abs() < 1e-6);
    }

    #[test]
    fn auto_keeps_the_encoder_resident_only_on_a_large_empty_card() {
        use TextEncoderChoice::*;
        assert_eq!(Auto.resolve(Some(95_000_000_000)), ResidentFp8);
        assert_eq!(Auto.resolve(Some(AUTO_RESIDENT_FREE_BYTES)), ResidentFp8);
        assert_eq!(Auto.resolve(Some(79_000_000_000)), Streamed, "an 80 GB card streams");
        assert_eq!(Auto.resolve(None), Streamed, "no device, no resident encoder");
        assert_eq!(Streamed.resolve(Some(u64::MAX)), Streamed, "an explicit choice is not second-guessed");
        assert_eq!(ResidentFp8.resolve(Some(0)), ResidentFp8);
        for name in ["auto", "streamed", "resident-fp8", "resident-bf16"] {
            assert!(TextEncoderChoice::parse(name).is_ok());
        }
        assert!(TextEncoderChoice::parse("fp8").is_err());
    }

    #[test]
    fn noise_is_video_first_then_audio_rows_from_one_generator() {
        let cfg = H3TransformerConfig::fasth3_8step();
        let g = H3Geometry::default_16x9(5).unwrap();
        let (video, audio) = seeded_noise(&cfg, &g, 7).unwrap();
        assert_eq!((video.len(), audio.len()), (37_296 * 96, 414 * 32));
        assert_eq!(seeded_noise(&cfg, &g, 7).unwrap().1, audio, "a seed names one sample");
        assert_ne!(seeded_noise(&cfg, &g, 8).unwrap().1, audio);
        let mean = video.iter().map(|&v| f64::from(v)).sum::<f64>() / video.len() as f64;
        let var = video.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / video.len() as f64;
        assert!(mean.abs() < 5e-3 && (var - 1.0).abs() < 5e-3, "N(0, 1): mean {mean} var {var}");
    }
}
