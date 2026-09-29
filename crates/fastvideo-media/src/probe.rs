//! Input probing and decoding (design §3.2 `StagedMedia.probe`, WP-05
//! ingestion).
//!
//! [`probe`] tries, in order: the `image` crate (PNG/JPEG, no subprocess),
//! ffprobe, and for MP4 files the built-in box reader when ffprobe is absent.
//! `MediaProbe` carries the `w,h,duration_s,fps,audio_rate` facts design §3.2
//! names, plus codecs and channel count.

use std::path::Path;
use std::process::Command;

use bytes::Bytes;

use crate::av::{f32le_to_f32, Pcm, RgbFrame};
use crate::error::{MediaError, Result};
use crate::tools;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct MediaProbe {
    pub kind: Option<MediaKind>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_s: Option<f64>,
    pub fps: Option<f64>,
    pub audio_rate: Option<u32>,
    pub audio_channels: Option<u8>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    /// e.g. `png`, `jpeg`, `mov,mp4,m4a,3gp,3g2,mj2`.
    pub format: String,
    /// Codec profiles as ffprobe names them (`Constrained Baseline`, `LC`).
    pub video_profile: Option<String>,
    pub audio_profile: Option<String>,
    /// The video stream's frame count (`nb_frames`, the MP4 sample count).
    #[serde(default)]
    pub frames: Option<u32>,
}

impl From<&MediaProbe> for fastvideo_protocol::MediaProbe {
    /// The subset `StagedMedia.probe` carries (design §3.2).
    fn from(p: &MediaProbe) -> Self {
        Self { width: p.width, height: p.height, duration_s: p.duration_s, fps: p.fps, audio_rate: p.audio_rate, frames: p.frames }
    }
}

/// Probe a staged file.
pub fn probe(path: &Path) -> Result<MediaProbe> {
    if let Some(p) = probe_image(path)? {
        return Ok(p);
    }
    if tools::ffprobe_available() {
        return ffprobe(path);
    }
    if let Ok(info) = crate::mp4::inspect(path) {
        return Ok(from_mp4(&info));
    }
    Err(MediaError::Unsupported(format!("cannot probe {} without ffprobe", path.display())))
}

/// PNG/JPEG via the image crate; `None` if the file is not an image it knows.
pub fn probe_image(path: &Path) -> Result<Option<MediaProbe>> {
    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let Some(fmt) = reader.format() else { return Ok(None) };
    let (w, h) = reader.into_dimensions().map_err(|e| MediaError::parse(Some(path), e.to_string()))?;
    Ok(Some(MediaProbe {
        kind: Some(MediaKind::Image),
        width: Some(w),
        height: Some(h),
        format: format!("{fmt:?}").to_lowercase(),
        ..Default::default()
    }))
}

fn parse_rate(s: &str) -> Option<f64> {
    let (n, d) = s.split_once('/').unwrap_or((s, "1"));
    let n: f64 = n.parse().ok()?;
    let d: f64 = d.parse().ok()?;
    if d == 0.0 || n == 0.0 { None } else { Some(n / d) }
}

/// Parse `ffprobe -print_format json -show_format -show_streams` output.
pub fn parse_ffprobe_json(json: &str) -> Result<MediaProbe> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| MediaError::parse(None, e.to_string()))?;
    let mut p = MediaProbe {
        format: v["format"]["format_name"].as_str().unwrap_or_default().to_string(),
        duration_s: v["format"]["duration"].as_str().and_then(|s| s.parse().ok()),
        ..Default::default()
    };
    let streams = v["streams"].as_array().cloned().unwrap_or_default();
    let mut has_video = false;
    let mut still = false;
    for s in &streams {
        match s["codec_type"].as_str() {
            Some("video") if !has_video => {
                has_video = true;
                p.width = s["width"].as_u64().map(|x| x as u32);
                p.height = s["height"].as_u64().map(|x| x as u32);
                p.video_codec = s["codec_name"].as_str().map(str::to_string);
                p.video_profile = s["profile"].as_str().map(str::to_string);
                p.fps = s["avg_frame_rate"].as_str().and_then(parse_rate).or_else(|| s["r_frame_rate"].as_str().and_then(parse_rate));
                p.frames = s["nb_frames"].as_str().and_then(|n| n.parse().ok()).filter(|n: &u32| *n > 0);
                still = matches!(p.video_codec.as_deref(), Some("png" | "mjpeg" | "webp" | "bmp" | "gif"))
                    && s["nb_frames"].as_str().map(|n| n == "1").unwrap_or(true)
                    && p.format.contains("pipe");
            }
            Some("audio") if p.audio_codec.is_none() => {
                p.audio_codec = s["codec_name"].as_str().map(str::to_string);
                p.audio_profile = s["profile"].as_str().map(str::to_string);
                p.audio_rate = s["sample_rate"].as_str().and_then(|r| r.parse().ok());
                p.audio_channels = s["channels"].as_u64().map(|c| c as u8);
            }
            _ => {}
        }
    }
    p.kind = if still {
        Some(MediaKind::Image)
    } else if has_video {
        Some(MediaKind::Video)
    } else if p.audio_codec.is_some() {
        Some(MediaKind::Audio)
    } else {
        None
    };
    Ok(p)
}

pub fn ffprobe(path: &Path) -> Result<MediaProbe> {
    let mut cmd = Command::new(tools::ffprobe_bin());
    cmd.args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams"]).arg(path);
    let out = tools::run_checked(cmd, "ffprobe")?;
    parse_ffprobe_json(&String::from_utf8_lossy(&out.stdout))
}

fn from_mp4(i: &crate::mp4::Mp4Info) -> MediaProbe {
    let v = i.video();
    let a = i.audio();
    MediaProbe {
        kind: Some(if v.is_some() { MediaKind::Video } else { MediaKind::Audio }),
        width: v.and_then(|t| t.width),
        height: v.and_then(|t| t.height),
        duration_s: Some(i.duration_s),
        fps: v.and_then(|t| t.fps),
        audio_rate: a.and_then(|t| t.audio.as_ref().map(|x| x.sample_rate)),
        audio_channels: a.and_then(|t| t.audio.as_ref().map(|x| x.channels as u8)),
        video_codec: v.map(|t| if t.codec.starts_with("avc") { "h264".into() } else { t.codec.clone() }),
        audio_codec: a.map(|t| if t.codec == "mp4a" { "aac".into() } else { t.codec.clone() }),
        format: "mov,mp4,m4a,3gp,3g2,mj2".into(),
        video_profile: None,
        audio_profile: None,
        frames: v.and_then(|t| u32::try_from(t.samples).ok()).filter(|n| *n > 0),
    }
}

/// Decode an image (PNG/JPEG) to RGB24.
pub fn decode_image_rgb(path: &Path) -> Result<RgbFrame> {
    let img = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()
        .map_err(|e| MediaError::Decode(format!("{}: {e}", path.display())))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    Ok(RgbFrame { width: w, height: h, data: Bytes::from(img.into_raw()), index: 0 })
}

/// Decode the audio of any file ffmpeg reads to interleaved f32 at `rate`/`channels`.
pub fn decode_audio(path: &Path, rate: u32, channels: u8) -> Result<Pcm> {
    let mut cmd = tools::ffmpeg_command();
    cmd.arg("-i")
        .arg(path)
        .args(["-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ar", &rate.to_string(), "-ac", &channels.to_string(), "pipe:1"]);
    let out = tools::run_checked(cmd, "ffmpeg")?;
    let mut s = f32le_to_f32(&out.stdout);
    s.truncate(s.len() - s.len() % channels as usize);
    crate::av::pcm(rate, channels, s)
}

/// Decode a video's frames to RGB24 at its own size (optionally resampled to
/// `fps`), up to `max_frames`.
pub fn decode_video_rgb(path: &Path, fps: Option<u32>, max_frames: Option<u32>) -> Result<Vec<RgbFrame>> {
    let p = probe(path)?;
    let (w, h) = match (p.width, p.height) {
        (Some(w), Some(h)) => (w, h),
        _ => return Err(MediaError::Decode(format!("{} has no video stream", path.display()))),
    };
    let mut cmd = tools::ffmpeg_command();
    cmd.arg("-i").arg(path).arg("-an");
    if let Some(f) = fps {
        cmd.args(["-vf", &format!("fps={f}")]);
    }
    if let Some(n) = max_frames {
        cmd.args(["-frames:v", &n.to_string()]);
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]);
    let out = tools::run_checked(cmd, "ffmpeg")?;
    let size = w as usize * h as usize * 3;
    Ok(out
        .stdout
        .chunks_exact(size)
        .enumerate()
        .map(|(i, c)| RgbFrame { width: w, height: h, data: Bytes::copy_from_slice(c), index: i as u64 })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_probe_and_decode() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.png");
        let img = image::RgbImage::from_fn(6, 4, |x, y| image::Rgb([x as u8 * 10, y as u8 * 20, 7]));
        img.save(&p).unwrap();
        let pr = probe(&p).unwrap();
        assert_eq!(pr.kind, Some(MediaKind::Image));
        assert_eq!((pr.width, pr.height), (Some(6), Some(4)));
        assert_eq!(pr.format, "png");
        let f = decode_image_rgb(&p).unwrap();
        assert_eq!((f.width, f.height), (6, 4));
        assert_eq!(&f.data[3..6], &[10, 0, 7]);
        // A non-image is not claimed by the image path.
        let q = dir.path().join("y.bin");
        std::fs::write(&q, b"not an image at all").unwrap();
        assert!(probe_image(&q).unwrap().is_none());
    }

    #[test]
    fn ffprobe_json_golden() {
        // Shape of ffprobe 7 output for a hosted-H3-like MP4 (trimmed).
        let j = r#"{"streams":[
          {"index":0,"codec_name":"h264","profile":"High","codec_type":"video","width":1344,"height":768,
           "r_frame_rate":"24/1","avg_frame_rate":"24/1","nb_frames":"124"},
          {"index":1,"codec_name":"aac","profile":"LC","codec_type":"audio","sample_rate":"32000","channels":2}],
          "format":{"format_name":"mov,mp4,m4a,3gp,3g2,mj2","duration":"5.184000"}}"#;
        let p = parse_ffprobe_json(j).unwrap();
        assert_eq!(p.kind, Some(MediaKind::Video));
        assert_eq!((p.width, p.height, p.fps), (Some(1344), Some(768), Some(24.0)));
        assert_eq!((p.audio_rate, p.audio_channels), (Some(32_000), Some(2)));
        assert_eq!(p.audio_profile.as_deref(), Some("LC"));
        assert_eq!(p.duration_s, Some(5.184));
        let staged = fastvideo_protocol::MediaProbe::from(&p);
        assert_eq!(staged.dims(), Some((1344, 768)));
        assert_eq!(staged.audio_rate, Some(32_000));
        let a = parse_ffprobe_json(
            r#"{"streams":[{"codec_type":"audio","codec_name":"mp3","sample_rate":"44100","channels":1}],"format":{"format_name":"mp3","duration":"1.0"}}"#,
        )
        .unwrap();
        assert_eq!(a.kind, Some(MediaKind::Audio));
        assert_eq!(parse_rate("30000/1001").map(|r| (r * 1000.0).round()), Some(29970.0));
        assert_eq!(parse_rate("0/0"), None);
    }
}
