//! The V2A sidecar for the video-only Wan family: a finished clip in, the
//! same clip with an AAC soundtrack out (strobe `scripts/batch/sidecar-audio.py`,
//! done in-process on our MMAudio port).
//!
//! Opt-in: `FASTVIDEO_WAN_AUDIO=mmaudio` (or `--audio mmaudio`). Weights:
//! `FASTVIDEO_MMAUDIO_WEIGHTS` / `MMAUDIO_MODEL_PATH` (default
//! `$FV_WEIGHTS/mmaudio-44k-v2`, `/workspace/weights/mmaudio-44k-v2`).
//! `FASTVIDEO_WAN_AUDIO_PROMPT` replaces the video prompt as the text
//! condition; `FASTVIDEO_WAN_AUDIO_STEPS` / `_CFG` / `_SEED` the sampler knobs
//! (defaults 25 / 4.5 / the video seed).
//!
//! The clip is read back from the mp4 exactly as upstream reads it (PyAV
//! there, ffmpeg here): the encoded frames are what both sides condition on.
//! The duration is `frames / fps`, as the sidecar passes `clip_s`.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use fastvideo_models::mmaudio::MmAudioPreset;

use super::pipeline::{write_wav, MmAudioOutput, MmAudioPipeline, MmAudioRequest, VideoFrames};
use crate::wan::pipeline::{PipelineError, Result};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// `FASTVIDEO_WAN_AUDIO`: `mmaudio` turns the sidecar on.
pub fn requested() -> bool {
    matches!(
        std::env::var("FASTVIDEO_WAN_AUDIO").unwrap_or_default().trim(),
        "mmaudio" | "1"
    )
}

pub fn weights_root() -> PathBuf {
    if let Some(p) = std::env::var_os("FASTVIDEO_MMAUDIO_WEIGHTS").or_else(|| std::env::var_os("MMAUDIO_MODEL_PATH")) {
        return PathBuf::from(p);
    }
    let base = std::env::var_os("FV_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/workspace/weights"));
    base.join("mmaudio-44k-v2")
}

/// One resident pipeline per process (loaded on first use).
pub fn shared() -> Result<&'static Mutex<MmAudioPipeline>> {
    static PIPE: OnceLock<Mutex<MmAudioPipeline>> = OnceLock::new();
    if let Some(p) = PIPE.get() {
        return Ok(p);
    }
    let p = MmAudioPipeline::load(weights_root(), MmAudioPreset::Large44kV2)?;
    Ok(PIPE.get_or_init(|| Mutex::new(p)))
}

#[derive(Debug, Clone)]
pub struct SidecarReport {
    pub wav: PathBuf,
    pub mp4: PathBuf,
    pub clip_s: f64,
    /// MMAudio generate (features through vocoder), device-synchronized.
    pub audio_s: f64,
    /// Read-back of the clip plus the mux.
    pub io_s: f64,
    pub audio: MmAudioOutput,
}

impl SidecarReport {
    pub fn audio_rtf(&self) -> f64 {
        self.audio_s / self.clip_s
    }
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// V2A on `mp4`, then mux the soundtrack into it in place (video stream
/// copied, AAC audio). `num_frames`/`fps` define the clip duration.
pub fn run_on_mp4(mp4: &Path, prompt: &str, seed: u64, num_frames: usize, fps: u32) -> Result<SidecarReport> {
    let t_io = Instant::now();
    let video = VideoFrames::from_file(mp4)?;
    let mut io_s = t_io.elapsed().as_secs_f64();
    let clip_s = num_frames as f64 / f64::from(fps.max(1));
    let preset = MmAudioPreset::Large44kV2;
    let req = MmAudioRequest {
        prompt: std::env::var("FASTVIDEO_WAN_AUDIO_PROMPT").unwrap_or_else(|_| prompt.to_string()),
        negative_prompt: String::new(),
        seed: env_or("FASTVIDEO_WAN_AUDIO_SEED", seed),
        duration_s: clip_s,
        num_steps: env_or("FASTVIDEO_WAN_AUDIO_STEPS", preset.default_steps()),
        cfg_strength: env_or("FASTVIDEO_WAN_AUDIO_CFG", preset.default_cfg()),
        video: Some(video),
    };
    let pipe = shared()?.lock().map_err(|_| msg("mmaudio lock"))?;
    let t = Instant::now();
    let audio = pipe.generate(&req)?;
    let audio_s = t.elapsed().as_secs_f64();
    drop(pipe);
    let t_mux = Instant::now();
    let wav = mp4.with_extension("wav");
    write_wav(&wav, &audio.waveform, audio.sample_rate)?;
    mux(mp4, &wav)?;
    io_s += t_mux.elapsed().as_secs_f64();
    Ok(SidecarReport {
        wav,
        mp4: mp4.to_path_buf(),
        clip_s,
        audio_s,
        io_s,
        audio,
    })
}

/// `ffmpeg -i video -i wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest`,
/// replacing `mp4`, then refuse a result without an audio stream.
pub fn mux(mp4: &Path, wav: &Path) -> Result<()> {
    let tmp = mp4.with_extension("av.mp4");
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(mp4)
        .arg("-i")
        .arg(wav)
        .args(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest"])
        .arg(&tmp)
        .status()
        .map_err(|e| msg(format!("ffmpeg mux: {e}")))?;
    if !st.success() {
        return Err(msg(format!("ffmpeg mux failed: {st}")));
    }
    let probe = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0"])
        .arg(&tmp)
        .output()
        .map_err(|e| msg(format!("ffprobe: {e}")))?;
    if !String::from_utf8_lossy(&probe.stdout).contains("audio") {
        return Err(msg(format!("muxed {} has no audio stream", tmp.display())));
    }
    std::fs::rename(&tmp, mp4).map_err(|e| msg(e.to_string()))
}
