//! LTX-2 video editing on the distilled pipeline: **retake** (regenerate a
//! time window of a video, its picture, its sound or both) and **extend**
//! (continue a video after its end or before its start).
//!
//! References (see [`fastvideo_models::ltx2::edit`] for the token maths):
//!
//! * `ltx_pipelines/retake.py` (Lightricks/LTX-2 `fd4ded7`, `RetakePipeline`)
//!   and LTX-Desktop's `ltx_retake_pipeline.py` (Lightricks/LTX-Desktop
//!   `68cd86c`, the same flow plus extend and the 2.5 sampler):
//!   * `video_latent_from_file`: the source decoded frame by frame
//!     (`decode_video_from_file`, RGB24, no rotation applied), each frame
//!     `resize_and_center_crop`ped to the generation size and normalized
//!     to bf16 ([`super::i2v_encode::image_pixels`]), then
//!     `VideoEncoder.tiled_encode` with `TileSizeConfig.default()` (80 / 24
//!     frames, 768 / 64 pixels; [`EncodePlan`]) and conformed to the clip's
//!     latent frames;
//!   * `audio_latent_from_file`: the source's first audio stream
//!     (`decode_audio_from_file(…, max_duration = frames / fps)`), the audio
//!     VAE encoder, conformed (cut or zero-padded) to the clip's audio
//!     latents; no audio stream: no audio conditioning (the audio is
//!     generated over the whole clip);
//!   * extend: both latents zero-padded by the new latent frames at the front
//!     or the back;
//!   * one `DiffusionStage` at the generation size with the distilled sigmas
//!     and, for 2.5, the ancestral sampler (`distilled_stage_sampler_kwargs`:
//!     "Distilled 2.5+ checkpoints need the ancestral sampler"; `retake.py`
//!     itself does not pass it), and the `TemporalRegionMask` on each
//!     regenerated stream: kept tokens are pinned to their clean latent
//!     (mask 0, [`StageConditioning::pinned`]), window tokens start from
//!     noise. A stream that is not regenerated is frozen (mask 0 everywhere
//!     and its `Modality.sigma` 0: [`super::transformer::Ltx2Transformer::set_audio_frozen`],
//!     [`super::transformer::Ltx2Transformer::set_video_frozen`]);
//!   * the whole clip decoded (video VAE with `AUTO_TILING`, audio VAE +
//!     vocoder): the kept frames are VAE round trips, as upstream.
//!
//! Additions of this server, outside the references: a **context window**
//! (`window_start`, `window_frames`: the source frames the model sees, for an
//! extension of a long source; the serve layer copies the rest to the output),
//! and a **dub** (a new audio track for a retake window, spliced into the
//! source's waveform at the source's rate, then frozen as conditioning).
//!
//! Oracle dumps: `v2v_video_latent` (`[1, F·H·W, 128]`, the window's encoded
//! latent before padding), `v2v_audio_wave` (`[C, N]`) and `v2v_audio_latent`
//! (`[1, L, 128]`, before padding); with FASTVIDEO_INJECT_DIR the reference's
//! latents replace ours (`FASTVIDEO_INJECT_COND=0` keeps ours) and its
//! waveform is encoded (`FASTVIDEO_INJECT_PIXELS=0` keeps ours).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use fastvideo_models::ltx2::config::Ltx2Config;
use fastvideo_models::ltx2::edit::{self as e, ExtendAt};
use fastvideo_models::ltx2::tiling::{EncodePlan, TileSizeConfig, VIDEO_SCALE};
use image::RgbImage;

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;

use super::a2v::{self, Waveform, ENCODER_CHANNELS};
use super::audio_vae::{conform_audio_time, pack_audio_latent, AudioEncoder};
use super::i2v_encode::{image_pixels, CondSegment, StageConditioning};
use super::pipeline::LatentState;
use super::transformer::pack_video;
use super::vae_encoder::VideoEncoder;

fn err(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// What an edit does, relative to its source window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditKind {
    /// Regenerate `[start_s, end_s)` of the window (`regenerate_video`,
    /// `regenerate_audio`).
    Retake {
        start_s: f64,
        end_s: f64,
        video: bool,
        audio: bool,
    },
    /// `frames` new frames (a multiple of 8) at `at`.
    Extend { frames: usize, at: ExtendAt },
}

/// A retake or extend of a source video (`Ltx2Request::edit`).
#[derive(Debug, Clone, PartialEq)]
pub struct VideoEdit {
    /// The source video file.
    pub source: PathBuf,
    /// Its exact frame rate: the model's positions and the window's times use
    /// it (`Ltx2Request::frame_rate` must equal it).
    pub source_fps: f64,
    /// The first source frame the model sees.
    pub window_start: usize,
    /// How many (`8k + 1`).
    pub window_frames: usize,
    pub kind: EditKind,
    /// Whether the source has an audio stream to condition on.
    pub source_audio: bool,
    /// Retake only: a new audio track for the window (spliced in, then the
    /// audio is frozen).
    pub dub: Option<PathBuf>,
}

impl VideoEdit {
    /// The generated clip's frame count: the window plus any extension.
    pub fn target_frames(&self) -> usize {
        match self.kind {
            EditKind::Retake { .. } => self.window_frames,
            EditKind::Extend { frames, .. } => self.window_frames + frames,
        }
    }

    /// The regenerated span in seconds of the generated clip.
    pub fn region(&self) -> (f64, f64) {
        match self.kind {
            EditKind::Retake { start_s, end_s, .. } => (start_s, end_s),
            EditKind::Extend { frames, at } => e::extend_window(self.window_frames, frames, at, self.source_fps),
        }
    }

    /// `(regenerate_video, regenerate_audio)`.
    pub fn regenerates(&self) -> (bool, bool) {
        match self.kind {
            EditKind::Retake { video, audio, .. } => (video, audio),
            EditKind::Extend { .. } => (true, true),
        }
    }

    /// The request's shape against the edit (`num_frames`, `frame_rate`,
    /// one stage).
    pub fn validate(&self, num_frames: usize, frame_rate: f64, two_stage: bool) -> Result<()> {
        if two_stage {
            return Err(err("ltx2 retake/extend runs one stage at the source size (RetakePipeline)"));
        }
        if e::frames_8k1_floor(self.window_frames) != Some(self.window_frames) {
            return Err(err(format!("ltx2 edit: the window is {} frames, not 8k+1 (at least 9)", self.window_frames)));
        }
        if num_frames != self.target_frames() {
            return Err(err(format!(
                "ltx2 edit: the request is {num_frames} frames, the edit makes {}",
                self.target_frames()
            )));
        }
        if !(self.source_fps.is_finite() && self.source_fps > 0.0) || (frame_rate - self.source_fps).abs() > 1e-9 {
            return Err(err(format!(
                "ltx2 edit: frame rate {frame_rate} must be the source's {}",
                self.source_fps
            )));
        }
        match self.kind {
            EditKind::Retake { start_s, end_s, video, audio } => {
                let clip = self.window_frames as f64 / self.source_fps;
                if !(0.0..clip).contains(&start_s) || end_s <= start_s {
                    return Err(err(format!("ltx2 retake window [{start_s}, {end_s}) is outside the {clip:.3} s clip")));
                }
                if !video && !audio {
                    return Err(err("ltx2 retake regenerates the video, the audio or both"));
                }
                if self.dub.is_some() && (!video || audio) {
                    return Err(err("ltx2 retake: new window audio goes with regenerating the video only"));
                }
            }
            EditKind::Extend { frames, .. } => {
                if frames == 0 || frames % e::TIME_FACTOR != 0 {
                    return Err(err(format!("ltx2 extend: {frames} new frames is not a positive multiple of 8")));
                }
                if self.dub.is_some() {
                    return Err(err("ltx2 extend takes no new audio"));
                }
            }
        }
        Ok(())
    }
}

/// `(frames, fps)` of the first video stream (ffprobe `nb_frames` and
/// `avg_frame_rate`; the frames are counted by decoding when the container
/// has no count).
pub fn probe_video(path: &Path) -> Result<(usize, f64)> {
    let run = |count: bool| -> Result<String> {
        let mut c = Command::new("ffprobe");
        c.args(["-v", "error", "-select_streams", "v:0"]);
        if count {
            c.args(["-count_frames", "-show_entries", "stream=nb_read_frames,avg_frame_rate"]);
        } else {
            c.args(["-show_entries", "stream=nb_frames,avg_frame_rate"]);
        }
        let out = c
            .args(["-of", "default=noprint_wrappers=1"])
            .arg(path)
            .output()
            .map_err(|x| err(format!("ffprobe not available: {x}")))?;
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let parse = |text: &str| -> (Option<usize>, Option<f64>) {
        let mut frames = None;
        let mut fps = None;
        for line in text.lines() {
            let (k, v) = line.split_once('=').unwrap_or((line, ""));
            match k.trim() {
                "nb_frames" | "nb_read_frames" => frames = v.trim().parse::<usize>().ok().filter(|n| *n > 0),
                "avg_frame_rate" => {
                    let (n, d) = v.trim().split_once('/').unwrap_or((v.trim(), "1"));
                    fps = match (n.parse::<f64>(), d.parse::<f64>()) {
                        (Ok(n), Ok(d)) if n > 0.0 && d > 0.0 => Some(n / d),
                        _ => None,
                    };
                }
                _ => {}
            }
        }
        (frames, fps)
    };
    let (mut frames, fps) = parse(&run(false)?);
    if frames.is_none() {
        frames = parse(&run(true)?).0;
    }
    match (frames, fps) {
        (Some(n), Some(f)) => Ok((n, f)),
        _ => Err(err(format!("ffprobe {}: no video frame count or rate", path.display()))),
    }
}

impl VideoEdit {
    /// The oracle's edit spec (`scripts/gpu/upstream/ltx25_edit.py`):
    /// `retake:SOURCE:START:END:av|v|a` (the regenerated streams) or
    /// `extend:SOURCE:FRAMES:start|end`. The window is the whole source cut
    /// to `8k + 1` frames (LTX-Desktop `correct_frame_count`), its rate and
    /// audio probed.
    pub fn from_spec(spec: &str) -> Result<Self> {
        let bad = || err(format!("edit spec {spec}: retake:SRC:START:END:av|v|a or extend:SRC:FRAMES:start|end"));
        let (kind, rest) = spec.split_once(':').ok_or_else(bad)?;
        let (source, kind) = match kind {
            "retake" => {
                let mut it = rest.rsplitn(4, ':');
                let (mode, end, start, src) = (it.next(), it.next(), it.next(), it.next());
                let (Some(mode), Some(end), Some(start), Some(src)) = (mode, end, start, src) else { return Err(bad()) };
                let start_s: f64 = start.parse().map_err(|_| bad())?;
                let end_s: f64 = end.parse().map_err(|_| bad())?;
                if !matches!(mode, "av" | "v" | "a") {
                    return Err(bad());
                }
                (src, EditKind::Retake { start_s, end_s, video: mode.contains('v'), audio: mode.contains('a') })
            }
            "extend" => {
                let mut it = rest.rsplitn(3, ':');
                let (at, frames, src) = (it.next(), it.next(), it.next());
                let (Some(at), Some(frames), Some(src)) = (at, frames, src) else { return Err(bad()) };
                let at = match at {
                    "start" => ExtendAt::Start,
                    "end" => ExtendAt::End,
                    _ => return Err(bad()),
                };
                (src, EditKind::Extend { frames: frames.parse().map_err(|_| bad())?, at })
            }
            _ => return Err(bad()),
        };
        let source = PathBuf::from(source);
        let (frames, fps) = probe_video(&source)?;
        let window = e::frames_8k1_floor(frames)
            .ok_or_else(|| err(format!("{}: {frames} frames, the editor needs at least 9", source.display())))?;
        let source_audio = a2v::probe(&source).is_ok();
        Ok(Self { source, source_fps: fps, window_start: 0, window_frames: window, kind, source_audio, dub: None })
    }
}

/// `(width, height)` of the first video stream (ffprobe).
fn probe_size(path: &Path) -> Result<(usize, usize)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .map_err(|x| err(format!("ffprobe not available: {x}")))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut it = text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").split(',');
    let w = it.next().and_then(|v| v.trim().parse::<usize>().ok()).filter(|v| *v > 0);
    let h = it.next().and_then(|v| v.trim().parse::<usize>().ok()).filter(|v| *v > 0);
    match (w, h) {
        (Some(w), Some(h)) if out.status.success() => Ok((w, h)),
        _ => Err(err(format!(
            "ffprobe {}: no video stream size ({})",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ))),
    }
}

/// The window's frames, preprocessed: `[3, F, H, W]` bf16 values kept as
/// their bit patterns (the reference casts `normalize_images` to bf16).
pub struct SourcePixels {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    /// Frame-major: `frames` planes of `[3, H, W]`.
    bits: Vec<u16>,
}

impl SourcePixels {
    /// `[3, t, y, x]` of a tile as f32, channel-major (the encoder's layout).
    pub fn tile(&self, t: std::ops::Range<usize>, y: std::ops::Range<usize>, x: std::ops::Range<usize>) -> Vec<f32> {
        let (h, w) = (self.height, self.width);
        let (nt, ny, nx) = (t.len(), y.len(), x.len());
        let mut out = vec![0f32; 3 * nt * ny * nx];
        for c in 0..3 {
            for (ti, f) in t.clone().enumerate() {
                let plane = &self.bits[(f * 3 + c) * h * w..(f * 3 + c + 1) * h * w];
                for (yi, yy) in y.clone().enumerate() {
                    let row = &plane[yy * w + x.start..yy * w + x.end];
                    let dst = &mut out[((c * nt + ti) * ny + yi) * nx..((c * nt + ti) * ny + yi + 1) * nx];
                    for (d, b) in dst.iter_mut().zip(row) {
                        *d = f32::from_bits(u32::from(*b) << 16);
                    }
                }
            }
        }
        out
    }

    /// Frame `f` as `[3, H, W]` f32.
    pub fn frame(&self, f: usize) -> Vec<f32> {
        self.tile(f..f + 1, 0..self.height, 0..self.width)
    }
}

/// `decode_video_from_file` + `video_preprocess` for source frames
/// `start .. start + count`: ffmpeg RGB24 (no autorotation, no frame-rate
/// conversion), each frame resized and center-cropped to `height x width`.
pub fn decode_source(path: &Path, start: usize, count: usize, height: usize, width: usize) -> Result<SourcePixels> {
    let (sw, sh) = probe_size(path)?;
    let trim = format!("trim=start_frame={start}:end_frame={}", start + count);
    let run = |sync: &[&str]| -> Result<std::process::Child> {
        Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-noautorotate", "-i"])
            .arg(path)
            .args(["-map", "0:v:0", "-vf", &trim])
            .args(sync)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|x| err(format!("ffmpeg not available: {x}")))
    };
    let frame_bytes = sw * sh * 3;
    let attempt = |sync: &[&str]| -> Result<(Vec<u16>, usize, String, bool)> {
        let mut child = run(sync)?;
        let mut stdout = child.stdout.take().ok_or_else(|| err("ffmpeg stdout closed"))?;
        let mut buf = vec![0u8; frame_bytes];
        let mut bits: Vec<u16> = Vec::with_capacity(count * 3 * height * width);
        let mut n = 0usize;
        while n < count {
            match stdout.read_exact(&mut buf) {
                Ok(()) => {}
                Err(x) if x.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(x) => return Err(err(format!("ffmpeg stdout: {x}"))),
            }
            let img = RgbImage::from_raw(sw as u32, sh as u32, buf.clone()).ok_or_else(|| err("ffmpeg frame size"))?;
            bits.extend(image_pixels(&img, height, width).into_iter().map(|v| (v.to_bits() >> 16) as u16));
            n += 1;
        }
        drop(stdout);
        let mut stderr = String::new();
        if let Some(mut x) = child.stderr.take() {
            let _ = x.read_to_string(&mut stderr);
        }
        let ok = child.wait().map_err(|x| err(format!("ffmpeg: {x}")))?.success();
        Ok((bits, n, stderr, ok))
    };
    let (mut bits, mut n, mut stderr, mut ok) = attempt(&["-fps_mode", "passthrough"])?;
    if !ok && n == 0 && stderr.contains("fps_mode") {
        // ffmpeg before 5.1.
        (bits, n, stderr, ok) = attempt(&["-vsync", "passthrough"])?;
    }
    if n < count {
        return Err(err(format!(
            "ltx2 edit: {} gave {n} frames from frame {start}, the window needs {count}{}",
            path.display(),
            if ok { String::new() } else { format!(" (ffmpeg: {})", stderr.trim()) }
        )));
    }
    Ok(SourcePixels { frames: n, height, width, bits })
}

/// `VideoEncoder.tiled_encode`: each tile of the plan encoded on its own,
/// weighted by its separable masks and summed; divided by the summed
/// weights unless they are all 1. Returns `[1, C, F', H', W']` on the host
/// (accumulated in f32; the reference accumulates in the model's bf16).
pub fn tiled_encode(encoder: &VideoEncoder, px: &SourcePixels, cfg: &TileSizeConfig) -> Result<CudaTensor> {
    let plan = EncodePlan::new(px.frames, px.height, px.width, cfg, VIDEO_SCALE).map_err(err)?;
    let [lf, lh, lw] = plan.latent;
    let c = encoder.latent_channels();
    let mut acc = vec![0f32; c * lf * lh * lw];
    let mut wsum = (!plan.complementary).then(|| vec![0f32; lf * lh * lw]);
    for tt in &plan.time {
        for th in &plan.height {
            for tw in &plan.width {
                let pixels = px.tile(tt.input.clone(), th.input.clone(), tw.input.clone());
                let lat = encoder.encode_video(&pixels, tt.input.len(), th.input.len(), tw.input.len())?;
                let (nf, nh, nw) = (tt.latent.len(), th.latent.len(), tw.latent.len());
                if lat.shape != [1, c, nf, nh, nw] {
                    return Err(err(format!("ltx2 tiled encode: tile latent {:?}, expected [1, {c}, {nf}, {nh}, {nw}]", lat.shape)));
                }
                let host = lat.host_cow()?;
                let m = |mask: &Option<Vec<f32>>, i: usize| mask.as_ref().map_or(1.0, |m| m[i]);
                for (fi, f) in tt.latent.clone().enumerate() {
                    for (yi, y) in th.latent.clone().enumerate() {
                        for (xi, x) in tw.latent.clone().enumerate() {
                            let w = m(&tt.mask, fi) * m(&th.mask, yi) * m(&tw.mask, xi);
                            let at = (f * lh + y) * lw + x;
                            for ch in 0..c {
                                acc[ch * lf * lh * lw + at] += host[((ch * nf + fi) * nh + yi) * nw + xi] * w;
                            }
                            if let Some(ws) = wsum.as_mut() {
                                ws[at] += w;
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(ws) = wsum {
        let n = lf * lh * lw;
        for ch in 0..c {
            for i in 0..n {
                acc[ch * n + i] /= ws[i].max(1e-8);
            }
        }
    }
    Ok(CudaTensor::from_vec(acc, vec![1, c, lf, lh, lw])?)
}

/// `[1, n, C]` zero rows.
fn zero_rows(n: usize, c: usize) -> Result<CudaTensor> {
    Ok(CudaTensor::zeros(&[1, n, c]).to_device()?)
}

/// `_pad_latent_frames` on packed rows: `pad` zero rows before (`start`) or
/// after (`end`).
fn pad_rows(x: CudaTensor, pad: usize, at: ExtendAt) -> Result<CudaTensor> {
    if pad == 0 {
        return Ok(x);
    }
    let z = zero_rows(pad, x.shape[2])?;
    Ok(match at {
        ExtendAt::Start => CudaTensor::cat(&[&z, &x], 1)?,
        ExtendAt::End => CudaTensor::cat(&[&x, &z], 1)?,
    })
}

/// Replace `t` with the reference's dump of the same size when injecting
/// (`FASTVIDEO_INJECT_COND`, default on).
fn inject(name: &str, t: CudaTensor) -> Result<CudaTensor> {
    if crate::wan::inject::enabled() && std::env::var("FASTVIDEO_INJECT_COND").map_or(true, |v| v != "0") {
        if let Some(v) = crate::wan::inject::load_numel(name, t.numel())? {
            crate::wan::log::info(format_args!("inject: {name} (reference)"));
            return Ok(CudaTensor::from_vec(v, t.shape.clone())?.to_device()?);
        }
    }
    Ok(t)
}

/// The kept runs of a stream as pinned segments over `clean`'s rows
/// (`frame_tokens` rows per latent frame).
fn pinned_segments(mask: &[bool], frame_tokens: usize, clean: &CudaTensor) -> Result<Vec<CondSegment>> {
    e::kept_runs(mask)
        .into_iter()
        .map(|(start, len)| {
            let (s, n) = (start * frame_tokens, len * frame_tokens);
            Ok(CondSegment { start: s, len: n, mask: 0.0, clean: clean.narrow(1, s, n)? })
        })
        .collect()
}

/// The conditioning of an edit's one stage.
pub struct EditConditioning {
    /// Video runs pinned (and the audio's: [`StageConditioning::audio_cond`]
    /// or frozen: [`StageConditioning::frozen_audio`]).
    pub cond: StageConditioning,
    /// Retake `replace_audio`: the video is frozen.
    pub video_frozen: bool,
    /// Seconds spent decoding and encoding the video / the audio.
    pub video_s: f64,
    pub audio_s: f64,
}

/// Encode the window and build the stage's conditioning. `grid` is the
/// generated clip's latent grid, `audio_tokens` its audio latent count.
pub fn prepare(
    weights: &Path,
    cfg: &Ltx2Config,
    edit: &VideoEdit,
    size: (usize, usize),
    grid: [usize; 3],
    audio_tokens: usize,
) -> Result<EditConditioning> {
    let (height, width) = size;
    let state = LatentState::for_version(cfg.version);
    let target = edit.target_frames();
    let fps = edit.source_fps;
    let [lf, lh, lw] = grid;
    let hw = lh * lw;
    if lf != e::video_latent_frames(target) {
        return Err(err(format!("ltx2 edit: latent grid {grid:?} for {target} frames")));
    }
    let (regen_v, regen_a) = edit.regenerates();
    let (r0, r1) = edit.region();

    // ---- video
    let t = Instant::now();
    let px = decode_source(&edit.source, edit.window_start, edit.window_frames, height, width)?;
    crate::wan::dump::host("v2v_frame0_pixels", &[3, height, width], &px.frame(0))?;
    let encoder = VideoEncoder::load(&weights.join("vae"), cfg.vae.patch_size, cfg.vae.pixel_norm_eps)?;
    let latent = tiled_encode(&encoder, &px, &TileSizeConfig::encode_default())?;
    drop((encoder, px));
    let window_latent = e::video_latent_frames(edit.window_frames);
    if latent.shape[2] != window_latent || latent.shape[3] != lh || latent.shape[4] != lw {
        return Err(err(format!("ltx2 edit: encoded {:?}, the window is [{window_latent}, {lh}, {lw}]", latent.shape)));
    }
    let packed = state.store(pack_video(&latent.to_device()?)?)?;
    crate::wan::dump::tensor("v2v_video_latent", &packed)?;
    crate::wan::dump::digest("v2v_video_latent", &packed)?;
    let packed = inject("v2v_video_latent", packed)?;
    let clean_v = match edit.kind {
        EditKind::Extend { frames, at } => pad_rows(packed, frames / e::TIME_FACTOR * hw, at)?,
        EditKind::Retake { .. } => packed,
    };
    if clean_v.shape[1] != lf * hw {
        return Err(err(format!("ltx2 edit: clean video {:?} for {} rows", clean_v.shape, lf * hw)));
    }
    let video_s = t.elapsed().as_secs_f64();

    let video_mask = if regen_v {
        e::in_region(&e::video_latent_spans(lf, fps), r0, r1)
    } else {
        vec![false; lf]
    };
    let segs = pinned_segments(&video_mask, hw, &clean_v)?;
    let mut cond = StageConditioning::pinned(lf * hw, hw, segs)?;
    crate::wan::log::info(format_args!(
        "ltx2 edit: video {}x{} {target} frames at {fps:.3} fps, window {}+{} of {}, region [{r0:.3}, {r1:.3}) s, {}/{lf} latent frames regenerated, encoded in {video_s:.2}s",
        width,
        height,
        edit.window_start,
        edit.window_frames,
        edit.source.display(),
        video_mask.iter().filter(|b| **b).count(),
    ));

    // ---- audio
    let t = Instant::now();
    let window_s = edit.window_frames as f64 / fps;
    let window_tokens = cfg.transformer.audio_tokens(edit.window_frames, fps);
    let wave = if edit.source_audio {
        Some(a2v::decode_range(&edit.source, edit.window_start as f64 / fps, Some(window_s), None)?)
    } else {
        None
    };
    let wave = match (&edit.dub, wave, edit.kind) {
        (Some(dub), w, EditKind::Retake { start_s, end_s, .. }) => {
            let mut base = match w {
                Some(w) => w,
                None => {
                    let (rate, _) = a2v::probe(dub)?;
                    Waveform::silence(ENCODER_CHANNELS, a2v::max_samples(window_s, rate), rate)
                }
            };
            let new = a2v::decode_range(dub, 0.0, Some(end_s - start_s), Some(base.rate))?;
            base.splice(a2v::max_samples(start_s, base.rate), &new)?;
            Some(base)
        }
        (_, w, _) => w,
    };
    let clean_a = match wave {
        None => None,
        Some(mut wave) => {
            let wname = "v2v_audio_wave";
            crate::wan::dump::host(wname, &[wave.channels, wave.samples()], &wave.planar)?;
            if crate::wan::inject::enabled() && std::env::var("FASTVIDEO_INJECT_PIXELS").map_or(true, |v| v != "0") {
                if let Some(v) = crate::wan::inject::load_numel(wname, wave.planar.len())? {
                    crate::wan::log::info(format_args!("inject: {wname} (reference)"));
                    wave.planar = v;
                }
            }
            let encoder = AudioEncoder::load(&super::pipeline::open_audio_encoder(weights)?, &cfg.audio_vae)?;
            let encoded = encoder.encode_waveform(&wave.planar, wave.channels, wave.rate)?;
            drop(encoder);
            // `_conform_latent_length`: cut, or zero-padded when the audio is short.
            let packed = state.store(pack_audio_latent(&conform_audio_time(&encoded, window_tokens)?.to_device()?)?)?;
            crate::wan::dump::tensor("v2v_audio_latent", &packed)?;
            crate::wan::dump::digest("v2v_audio_latent", &packed)?;
            let packed = inject("v2v_audio_latent", packed)?;
            let padded = match edit.kind {
                EditKind::Extend { at, .. } => pad_rows(packed, audio_tokens.saturating_sub(window_tokens), at)?,
                EditKind::Retake { .. } => packed,
            };
            if padded.shape[1] != audio_tokens {
                return Err(err(format!("ltx2 edit: clean audio {:?} for {audio_tokens} tokens", padded.shape)));
            }
            crate::wan::log::info(format_args!(
                "ltx2 edit: audio {} Hz, {:.3}s{} → {window_tokens} of {audio_tokens} tokens",
                wave.rate,
                wave.duration_s(),
                if edit.dub.is_some() { " (window dubbed)" } else { "" }
            ));
            Some(padded)
        }
    };
    let audio_s = t.elapsed().as_secs_f64();
    if let Some(clean) = clean_a {
        if regen_a {
            let mask = e::in_region(&e::audio_latent_spans(audio_tokens), r0, r1);
            let segs = pinned_segments(&mask, 1, &clean)?;
            cond = cond.with_audio(StageConditioning::pinned(audio_tokens, 1, segs)?);
        } else {
            cond = cond.with_frozen_audio(clean)?;
        }
    }
    Ok(EditConditioning { cond, video_frozen: !regen_v, video_s, audio_s })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(kind: EditKind) -> VideoEdit {
        VideoEdit {
            source: PathBuf::from("/x.mp4"),
            source_fps: 24.0,
            window_start: 0,
            window_frames: 97,
            kind,
            source_audio: true,
            dub: None,
        }
    }

    #[test]
    fn edits_check_their_request() {
        let r = edit(EditKind::Retake { start_s: 1.0, end_s: 3.0, video: true, audio: true });
        assert_eq!(r.target_frames(), 97);
        assert!(r.validate(97, 24.0, false).is_ok());
        assert!(r.validate(97, 24.0, true).is_err());
        assert!(r.validate(105, 24.0, false).is_err());
        assert!(r.validate(97, 25.0, false).is_err());
        let x = edit(EditKind::Extend { frames: 48, at: ExtendAt::End });
        assert_eq!(x.target_frames(), 145);
        assert!(x.validate(145, 24.0, false).is_ok());
        assert_eq!(x.region(), (85.0 / 24.0, 145.0 / 24.0));
        assert!(edit(EditKind::Extend { frames: 44, at: ExtendAt::End }).validate(141, 24.0, false).is_err());
        let mut d = edit(EditKind::Retake { start_s: 1.0, end_s: 3.0, video: true, audio: true });
        d.dub = Some(PathBuf::from("/d.wav"));
        assert!(d.validate(97, 24.0, false).is_err());
        d.kind = EditKind::Retake { start_s: 1.0, end_s: 3.0, video: true, audio: false };
        assert!(d.validate(97, 24.0, false).is_ok());
    }

    #[test]
    fn source_pixels_tiles_are_channel_major() {
        // 2 frames of 2x3, value = f·100 + c·10 + y·3 + x (as bf16 bits of small ints).
        let (h, w) = (2usize, 3usize);
        let mut bits = Vec::new();
        for f in 0..2 {
            for c in 0..3 {
                for i in 0..h * w {
                    let v = (f * 100 + c * 10 + i) as f32;
                    bits.push((v.to_bits() >> 16) as u16);
                }
            }
        }
        let px = SourcePixels { frames: 2, height: h, width: w, bits };
        let t = px.tile(1..2, 1..2, 1..3);
        // c 0: frame 1, row 1, x 1..3 → 104, 105; c 1: 114, 115; c 2: 124, 125.
        assert_eq!(t, vec![104.0, 105.0, 114.0, 115.0, 124.0, 125.0]);
        assert_eq!(px.frame(0)[..3], [0.0, 1.0, 2.0]);
    }
}
