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
//! **Stages never share the device.** The text encoder streams through and is
//! gone; the refiner runs once and is dropped; the DiT is dropped before the
//! decoders load. Audio decodes first — it takes milliseconds and the WAV has
//! to exist before ffmpeg is spawned — and the video VAE then streams 17-frame
//! chunks straight into the muxer.

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
    /// Dense attention without the compression gate: the parity mode the
    /// diffusers oracle judges. The checkpoint was distilled with VSA-H3, so
    /// the default (`false`) is what it should be served with.
    pub dense: bool,
    /// Write `output.mp4` (needs ffmpeg) next to the PNG frames.
    pub mp4: bool,
    /// Where to memoize the precomputed AdaLN table (155 MB). Building it reads
    /// 26 GB of projections nothing else needs, so a warm start is much
    /// shorter; the file is validated against the checkpoint and the ladder.
    pub adaln_cache: Option<PathBuf>,
}

impl H3Request {
    /// The default 16:9 canvas (768 x 1344) for a whole number of seconds.
    pub fn seconds(prompt: impl Into<String>, seconds: usize, seed: u64) -> std::result::Result<Self, String> {
        let g = H3Geometry::default_16x9(seconds)?;
        Ok(Self { prompt: prompt.into(), seed, height: g.height, width: g.width, num_frames: g.num_frames, dense: false, mp4: true, adaln_cache: None })
    }
}

#[derive(Debug, Clone, Default)]
pub struct H3Timings {
    pub text_s: f64,
    pub refine_s: f64,
    pub load_dit_s: f64,
    pub denoise_s: f64,
    pub step_s: Vec<f64>,
    pub audio_decode_s: f64,
    pub load_vae_s: f64,
    pub video_decode_s: f64,
    /// The tail of encoding that outlives the decode, not the whole encode.
    pub write_s: f64,
}

#[derive(Debug, Clone)]
pub struct H3Output {
    pub geometry: H3Geometry,
    pub text_tokens: usize,
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

/// Generate one clip into `out_dir`: `frame-NNN.png`, `audio.wav`, and
/// `output.mp4` with the audio muxed in. `root` is the FastH3 snapshot.
pub fn generate(root: &Path, request: &H3Request, out_dir: &Path) -> Result<H3Output> {
    let cfg = H3TransformerConfig::fasth3_8step();
    let geometry = H3Geometry::new(request.height, request.width, request.num_frames).map_err(msg)?;
    let schedule = H3JointSchedule::fasth3_8step();
    let mut timings = H3Timings::default();

    // --- text: ~50 GB streams through one layer at a time ----------------------
    let timer = Instant::now();
    let (ids, text) = super::text::encode_prompt(root, &request.prompt)?;
    timings.text_s = timer.elapsed().as_secs_f64();

    let map = WeightMap::open(&root.join("transformer"))?;
    let timer = Instant::now();
    let text_refined = H3TextRefiner::load(&cfg, &map)?.forward(&text)?;
    drop(text);
    timings.refine_s = timer.elapsed().as_secs_f64();

    // --- DiT ---------------------------------------------------------------------
    let timer = Instant::now();
    let model = H3Transformer::load_cached(cfg.clone(), &map, &schedule, !request.dense, request.adaln_cache.as_deref())?;
    timings.load_dit_s = timer.elapsed().as_secs_f64();

    let layout = H3PackedLayout::from_geometry(&geometry, ids.len()).map_err(msg)?;
    let sequence_length = layout.sequence_length();
    let vsa = if request.dense { None } else { Some(H3Vsa::new(&layout, cfg.num_attention_heads, cfg.attention_head_dim, super::vsa::H3VsaConfig::fasth3_8step())?) };
    let mode = vsa.as_ref().map_or(AttnMode::Dense, AttnMode::Vsa);
    let layout = DeviceLayout::new(&cfg, layout)?;
    let (video_noise, audio_noise) = seeded_noise(&cfg, &geometry, request.seed).map_err(msg)?;
    let video_rows = CudaTensor::from_vec(video_noise, vec![geometry.video_rows(), cfg.video_patch_dim()])?.to_device()?;
    let audio_rows = CudaTensor::from_vec(audio_noise, vec![geometry.audio_rows(), cfg.audio_in_channels])?.to_device()?;

    let timer = Instant::now();
    let mut last = Instant::now();
    let mut step_s = Vec::with_capacity(schedule.num_steps());
    let (video_rows, audio_rows) = denoise(&model, &layout, &text_refined, video_rows, audio_rows, &schedule, mode, &mut |step, _, _| {
        crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
        step_s.push(last.elapsed().as_secs_f64());
        crate::wan::log::info(format_args!("h3 step {}/{}: {:.1}s", step + 1, schedule.num_steps(), step_s[step]));
        last = Instant::now();
        Ok(())
    })?;
    timings.denoise_s = timer.elapsed().as_secs_f64();
    timings.step_s = step_s;
    // 41 GiB the decoders have no use for.
    drop((model, vsa, layout, text_refined));

    // --- audio first: the WAV must exist before the muxer starts -------------------
    let timer = Instant::now();
    let audio_cfg = H3AudioVaeConfig::fasth3_8step();
    let sample_rate = audio_cfg.sampling_rate as u32;
    let wave = {
        let decoder = H3AudioDecoder::load(audio_cfg, &WeightMap::open(&root.join("audio_vae"))?)?;
        decoder.decode_rows(&audio_rows, H3_AUDIO_CHANNELS)?.host_cow()?.into_owned()
    };
    let wav = out_dir.join("audio.wav");
    write_wav(&wav, &interleave_audio(&wave, H3_AUDIO_CHANNELS)?, H3_AUDIO_CHANNELS as u16, sample_rate)?;
    timings.audio_decode_s = timer.elapsed().as_secs_f64();

    // --- video: chunks go to the writer as they decode -------------------------------
    let timer = Instant::now();
    let decoder = H3VideoDecoder::load(H3VideoVaeConfig::fasth3_8step(), &WeightMap::open(&root.join("vae"))?)?;
    timings.load_vae_s = timer.elapsed().as_secs_f64();
    let latents = unpatchify_rows(&video_rows, cfg.in_channels, geometry.token_grid, cfg.patch_size)?;
    let timer = Instant::now();
    let mut writer = VideoWriter::spawn_with_audio(out_dir, H3_FPS as u32, request.mp4, Some(&wav))?;
    // The sink speaks TensorError; carry the writer's own error out beside it.
    let mut writer_error: Option<PipelineError> = None;
    let decoded = decoder.decode_streaming(&latents, &mut |offset, frames| {
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

    Ok(H3Output { geometry, text_tokens: ids.len(), sequence_length, frames, frame_paths, mp4, wav, timings })
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
