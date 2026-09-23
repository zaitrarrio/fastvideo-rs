//! Host-side ffmpeg helpers for Ref2VA reference decode (RGB frames + PCM).
//!
//! Decode stays off the device graph; the GPU encoders consume the prepared
//! buffers. Needs `ffmpeg` / `ffprobe` on PATH.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::wan::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Probe width, height, and average frame rate of the first video stream.
pub fn probe_video(path: &Path) -> Result<(usize, usize, f64)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "csv=p=0",
            path.to_str()
                .ok_or_else(|| msg("video path is not utf-8"))?,
        ])
        .output()
        .map_err(|e| msg(format!("ffprobe not available: {e}")))?;
    if !out.status.success() {
        return Err(msg(format!(
            "ffprobe failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = line.trim().split(',').collect();
    if parts.len() < 3 {
        return Err(msg(format!("ffprobe: unexpected video metadata `{line}`")));
    }
    let width: usize = parts[0]
        .parse()
        .map_err(|_| msg(format!("ffprobe width: {}", parts[0])))?;
    let height: usize = parts[1]
        .parse()
        .map_err(|_| msg(format!("ffprobe height: {}", parts[1])))?;
    let fps = parse_rate(parts[2])
        .or_else(|| parts.get(3).and_then(|p| parse_rate(p)))
        .ok_or_else(|| msg(format!("ffprobe: no frame rate in `{line}`")))?;
    if width == 0 || height == 0 || fps <= 0.0 {
        return Err(msg(format!(
            "ffprobe: bad geometry {width}x{height} @ {fps}"
        )));
    }
    Ok((width, height, fps))
}

fn parse_rate(s: &str) -> Option<f64> {
    if let Some((a, b)) = s.split_once('/') {
        let n: f64 = a.parse().ok()?;
        let d: f64 = b.parse().ok()?;
        if d == 0.0 {
            return None;
        }
        Some(n / d)
    } else {
        s.parse().ok()
    }
}

/// Whether the container has at least one audio stream.
pub fn probe_has_audio(path: &Path) -> Result<bool> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path.to_str().ok_or_else(|| msg("path is not utf-8"))?,
        ])
        .output()
        .map_err(|e| msg(format!("ffprobe not available: {e}")))?;
    Ok(out.status.success() && !out.stdout.is_empty())
}

/// Decode up to `max_frames` RGB24 frames at the stream's native resolution.
/// Returns `(frames_hwc, width, height, fps)`.
pub fn decode_video_rgb(path: &Path, max_frames: usize) -> Result<(Vec<u8>, usize, usize, f64)> {
    let (width, height, fps) = probe_video(path)?;
    let frame_bytes = width * height * 3;
    let mut child = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostats",
            "-i",
            path.to_str()
                .ok_or_else(|| msg("video path is not utf-8"))?,
            "-an",
            "-frames:v",
            &max_frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| msg(format!("ffmpeg not available: {e}")))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| msg("ffmpeg stdout closed"))?;
    let mut buf = Vec::new();
    stdout
        .read_to_end(&mut buf)
        .map_err(|e| msg(format!("ffmpeg stdout: {e}")))?;
    let status = child.wait().map_err(|e| msg(format!("ffmpeg: {e}")))?;
    if !status.success() {
        return Err(msg(format!("ffmpeg decode failed with {status}")));
    }
    if buf.is_empty() || buf.len() % frame_bytes != 0 {
        return Err(msg(format!(
            "ffmpeg returned {} bytes for {width}x{height} frames from {}",
            buf.len(),
            path.display()
        )));
    }
    Ok((buf, width, height, fps))
}

/// Decode the first audio stream to stereo f32le at `sample_rate`, truncated to
/// `max_samples` per channel. Returns planar `[L...][R...]` of equal length.
pub fn decode_audio_stereo_f32(
    path: &Path,
    sample_rate: u32,
    max_samples: usize,
) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostats",
            "-i",
            path.to_str()
                .ok_or_else(|| msg("audio path is not utf-8"))?,
            "-vn",
            "-ac",
            "2",
            "-ar",
            &sample_rate.to_string(),
            "-f",
            "f32le",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| msg(format!("ffmpeg not available: {e}")))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| msg("ffmpeg stdout closed"))?;
    let mut raw = Vec::new();
    stdout
        .read_to_end(&mut raw)
        .map_err(|e| msg(format!("ffmpeg stdout: {e}")))?;
    let status = child.wait().map_err(|e| msg(format!("ffmpeg: {e}")))?;
    if !status.success() {
        return Err(msg(format!("ffmpeg audio decode failed with {status}")));
    }
    if raw.len() % 8 != 0 {
        return Err(msg(format!(
            "ffmpeg audio: {} bytes not stereo f32",
            raw.len()
        )));
    }
    let interleaved: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let total = interleaved.len() / 2;
    let keep = total.min(max_samples);
    if keep == 0 {
        return Err(msg(format!("no audio samples in {}", path.display())));
    }
    // Interleaved LR → planar [L|R].
    let mut planar = vec![0f32; keep * 2];
    for i in 0..keep {
        planar[i] = interleaved[i * 2];
        planar[keep + i] = interleaved[i * 2 + 1];
    }
    Ok(planar)
}

/// Resize packed HWC RGB frames with LANCZOS (via `image`).
pub fn resize_rgb_frames(
    frames: &[u8],
    num_frames: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Result<Vec<u8>> {
    let src_bytes = src_h * src_w * 3;
    let dst_bytes = dst_h * dst_w * 3;
    if frames.len() != num_frames * src_bytes {
        return Err(msg(format!(
            "resize: {} bytes for {num_frames}x{src_h}x{src_w}",
            frames.len()
        )));
    }
    if src_h == dst_h && src_w == dst_w {
        return Ok(frames.to_vec());
    }
    let mut out = Vec::with_capacity(num_frames * dst_bytes);
    for t in 0..num_frames {
        let start = t * src_bytes;
        let img = image::RgbImage::from_raw(
            src_w as u32,
            src_h as u32,
            frames[start..start + src_bytes].to_vec(),
        )
        .ok_or_else(|| msg("resize: invalid rgb buffer"))?;
        let resized = image::imageops::resize(
            &img,
            dst_w as u32,
            dst_h as u32,
            image::imageops::FilterType::Lanczos3,
        );
        out.extend_from_slice(resized.as_raw());
    }
    Ok(out)
}
