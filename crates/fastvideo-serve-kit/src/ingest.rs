//! Media ingestion: stage every [`MediaRef`] of a request into the job's input
//! dir, enforce per-API limits, check the type and probe (design §2.1, §4.5).
//!
//! - `http(s)://` URLs are fetched (feature `fetch`) through the SSRF guard
//!   ([`crate::net`]): the host is resolved once, every address must be public,
//!   the connection is pinned to the checked address, redirects are refused or
//!   re-checked hop by hop, and size and time limits apply while streaming.
//! - `data:` URIs must be base64; the encoded payload and the decoded bytes
//!   are both capped.
//! - Upload ids resolve through [`UploadStore`].
//! - Provider files (`mm_file://`, OpenAI `file_id`) are
//!   `Unsupported(ProviderFiles)`.
//!
//! The type comes from the file's magic bytes when recognizable, else from the
//! declared type (`Content-Type` or the data URI), and must match the expected
//! kind (image/video/audio) or the request is refused with `UnsupportedMedia`.
//! Probing goes through the [`Prober`] hook ([`DefaultProber`]: the `image`
//! crate for PNG/JPEG, `ffprobe` for everything else when installed).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use fastvideo_protocol::{
    ApiError, GapId, GenerationRequest, MediaKind, MediaProbe, MediaRef, StagedInputs,
    StagedMedia,
};
use time::OffsetDateTime;

use crate::net::TargetPolicy;
use crate::uploads::UploadStore;

const MB: u64 = 1024 * 1024;

/// Limits for one media kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindLimits {
    /// Largest staged file (fetched, uploaded or decoded).
    pub max_bytes: u64,
    /// Largest base64 payload of a data URI.
    pub data_uri_max_encoded: u64,
    /// Whole-fetch timeout (all redirect hops and the body).
    pub timeout: Duration,
}

/// Per-API ingestion policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IngestPolicy {
    pub image: KindLimits,
    pub video: KindLimits,
    pub audio: KindLimits,
    pub target: TargetPolicy,
    /// Follow redirects (each hop re-checked by the SSRF guard).
    pub follow_redirects: bool,
    pub max_redirects: u8,
}

impl Default for IngestPolicy {
    /// fal, MiniMax, FastVideo: http(s), redirects followed (5), private
    /// targets refused; images 30 MB / 20 s, video 100 MB / 60 s, audio
    /// 50 MB / 30 s; data URIs up to the 64 MB body limit.
    fn default() -> Self {
        Self {
            image: KindLimits { max_bytes: 30 * MB, data_uri_max_encoded: 40 * MB, timeout: Duration::from_secs(20) },
            video: KindLimits { max_bytes: 100 * MB, data_uri_max_encoded: 64 * MB, timeout: Duration::from_secs(60) },
            audio: KindLimits { max_bytes: 50 * MB, data_uri_max_encoded: 64 * MB, timeout: Duration::from_secs(30) },
            target: TargetPolicy::default(),
            follow_redirects: true,
            max_redirects: 5,
        }
    }
}

impl IngestPolicy {
    /// LTX §2.0: HTTPS only, no IP hosts, no redirects; images 15 MB / 10 s,
    /// video and audio 32 MB / 30 s; data URIs 7 MB (image) and 15 MB encoded.
    pub fn ltx() -> Self {
        Self {
            image: KindLimits { max_bytes: 15 * MB, data_uri_max_encoded: 7 * MB, timeout: Duration::from_secs(10) },
            video: KindLimits { max_bytes: 32 * MB, data_uri_max_encoded: 15 * MB, timeout: Duration::from_secs(30) },
            audio: KindLimits { max_bytes: 32 * MB, data_uri_max_encoded: 15 * MB, timeout: Duration::from_secs(30) },
            target: TargetPolicy { https_only: true, allow_ip_literals: false, allow_private: false },
            follow_redirects: false,
            max_redirects: 0,
        }
    }

    pub fn limits(&self, k: MediaKind) -> &KindLimits {
        match k {
            MediaKind::Image => &self.image,
            MediaKind::Video => &self.video,
            MediaKind::Audio => &self.audio,
        }
    }
}

/// Probing hook (image decode, ffprobe, or `fastvideo-media`'s prober).
#[async_trait::async_trait]
pub trait Prober: Send + Sync + 'static {
    /// Facts about a staged file whose type already matched `kind`. An error
    /// refuses the input (`UnsupportedMedia` for an undecodable file).
    async fn probe(&self, path: &Path, kind: MediaKind, mime: &str) -> Result<MediaProbe, ApiError>;
}

/// PNG/JPEG via the `image` crate; other types via `ffprobe` when present
/// (missing ffprobe yields an empty probe, never an error).
#[derive(Clone, Debug)]
pub struct DefaultProber {
    pub ffprobe: Option<PathBuf>,
}

impl Default for DefaultProber {
    fn default() -> Self {
        Self { ffprobe: Some(PathBuf::from("ffprobe")) }
    }
}

#[async_trait::async_trait]
impl Prober for DefaultProber {
    async fn probe(&self, path: &Path, kind: MediaKind, mime: &str) -> Result<MediaProbe, ApiError> {
        if kind == MediaKind::Image && matches!(mime, "image/png" | "image/jpeg") {
            let p = path.to_owned();
            let dims = tokio::task::spawn_blocking(move || {
                image::ImageReader::open(&p)?
                    .with_guessed_format()?
                    .into_dimensions()
                    .map_err(std::io::Error::other)
            })
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?;
            return match dims {
                Ok((w, h)) if w > 0 && h > 0 => Ok(MediaProbe { width: Some(w), height: Some(h), ..Default::default() }),
                _ => Err(ApiError::unsupported_media("the image could not be decoded")),
            };
        }
        // MP4 / MOV video: the box reader gives the exact sample count, size,
        // rate and audio track without a subprocess (retake / extend need all
        // four); other containers go to ffprobe.
        if kind == MediaKind::Video {
            let p = path.to_owned();
            let boxed = tokio::task::spawn_blocking(move || fastvideo_media::mp4::inspect(&p).ok())
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
            if let Some(probe) = boxed.as_ref().and_then(mp4_probe) {
                return Ok(probe);
            }
        }
        let mut p = match &self.ffprobe {
            Some(bin) => ffprobe(bin, path).await.unwrap_or_default(),
            None => MediaProbe::default(),
        };
        // A WAV's rate and length are in its header: read them when ffprobe
        // is missing (audio-to-video needs both).
        if kind == MediaKind::Audio && (p.audio_rate.is_none() || p.duration_s.is_none()) {
            let head = tokio::fs::read(path).await.ok().map(|mut b| {
                b.truncate(WAV_HEAD_MAX);
                b
            });
            let size = tokio::fs::metadata(path).await.map(|m| m.len()).unwrap_or(0);
            if let Some((rate, secs)) = head.as_deref().and_then(|h| parse_wav_header(h, size)) {
                p.audio_rate.get_or_insert(rate);
                p.duration_s.get_or_insert(secs);
            }
        }
        Ok(p)
    }
}

/// An MP4 / MOV with a sized video track: its size, sample count and rate
/// (the frame count over the track's duration when `stts` has several
/// deltas), the file's duration and the first audio track's rate.
pub fn mp4_probe(i: &fastvideo_media::mp4::Mp4Info) -> Option<MediaProbe> {
    let v = i.video()?;
    let (w, h) = (v.width.filter(|w| *w > 0)?, v.height.filter(|h| *h > 0)?);
    let frames = u32::try_from(v.samples).ok().filter(|n| *n > 0)?;
    let dur = v.duration_s();
    let fps = v.fps.or_else(|| (dur > 0.0).then(|| f64::from(frames) / dur))?;
    Some(MediaProbe {
        width: Some(w),
        height: Some(h),
        duration_s: Some(if dur > 0.0 { dur } else { i.duration_s }),
        fps: Some(fps),
        audio_rate: i.audio().and_then(|a| a.audio.as_ref().map(|x| x.sample_rate)).filter(|r| *r > 0),
        frames: Some(frames),
    })
}

/// How much of a WAV file [`parse_wav_header`] looks at.
const WAV_HEAD_MAX: usize = 1 << 16;

/// `(sample_rate, seconds)` from a RIFF/WAVE header: the `fmt ` chunk's rate
/// and block size and the `data` chunk's length (clamped to the file for
/// streamed WAVs whose size field is 0 or 0xFFFFFFFF). `None` for anything
/// else.
pub fn parse_wav_header(head: &[u8], file_len: u64) -> Option<(u32, f64)> {
    let u16_at = |i: usize| head.get(i..i + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |i: usize| head.get(i..i + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    if head.get(0..4)? != b"RIFF" || head.get(8..12)? != b"WAVE" {
        return None;
    }
    let (mut rate, mut block) = (None, None);
    let mut at = 12usize;
    while at + 8 <= head.len() {
        let id = &head[at..at + 4];
        let len = u32_at(at + 4)? as usize;
        let body = at + 8;
        match id {
            b"fmt " => {
                rate = u32_at(body + 4).filter(|&r| r > 0);
                block = u16_at(body + 12).filter(|&b| b > 0);
            }
            b"data" => {
                let (rate, block) = (rate?, u64::from(block?));
                let avail = file_len.saturating_sub(body as u64);
                let bytes = if len == 0 || len == u32::MAX as usize { avail } else { (len as u64).min(avail) };
                return Some((rate, (bytes / block) as f64 / f64::from(rate)));
            }
            _ => {}
        }
        at = body.checked_add(len + (len & 1))?;
    }
    None
}

/// Runs `ffprobe -show_streams -show_format` and reads the first video and
/// audio stream. `None` when ffprobe is missing or fails.
pub async fn ffprobe(bin: &Path, path: &Path) -> Option<MediaProbe> {
    let out = tokio::process::Command::new(bin)
        .args(["-v", "error", "-print_format", "json", "-show_streams", "-show_format"])
        .arg(path)
        .kill_on_drop(true)
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_ffprobe(&serde_json::from_slice(&out.stdout).ok()?)
}

/// Reads ffprobe JSON (`streams[]`, `format.duration`).
pub fn parse_ffprobe(v: &serde_json::Value) -> Option<MediaProbe> {
    let streams = v.get("streams")?.as_array()?;
    let mut p = MediaProbe::default();
    let num = |x: Option<&serde_json::Value>| -> Option<f64> {
        let x = x?;
        x.as_f64().or_else(|| x.as_str()?.parse().ok())
    };
    for s in streams {
        match s.get("codec_type").and_then(|c| c.as_str()) {
            Some("video") if p.width.is_none() => {
                p.width = s.get("width").and_then(|w| w.as_u64()).map(|w| w as u32);
                p.height = s.get("height").and_then(|w| w.as_u64()).map(|w| w as u32);
                p.fps = s
                    .get("avg_frame_rate")
                    .or(s.get("r_frame_rate"))
                    .and_then(|r| r.as_str())
                    .and_then(|r| {
                        let (a, b) = r.split_once('/')?;
                        let (a, b): (f64, f64) = (a.parse().ok()?, b.parse().ok()?);
                        (b > 0.0 && a > 0.0).then(|| a / b)
                    });
                if p.duration_s.is_none() {
                    p.duration_s = num(s.get("duration"));
                }
                p.frames = num(s.get("nb_frames")).filter(|n| *n >= 1.0).map(|n| n as u32);
            }
            Some("audio") if p.audio_rate.is_none() => {
                p.audio_rate = num(s.get("sample_rate")).map(|r| r as u32);
                if p.duration_s.is_none() {
                    p.duration_s = num(s.get("duration"));
                }
            }
            _ => {}
        }
    }
    if let Some(d) = num(v.get("format").and_then(|f| f.get("duration"))) {
        p.duration_s = Some(d);
    }
    Some(p)
}

/// The MIME type recognized from magic bytes, if any.
pub fn sniff_mime(head: &[u8]) -> Option<&'static str> {
    let h = head;
    let at = |i: usize, s: &[u8]| h.len() >= i + s.len() && &h[i..i + s.len()] == s;
    if at(0, b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if at(0, b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if at(0, b"GIF87a") || at(0, b"GIF89a") {
        Some("image/gif")
    } else if at(0, b"RIFF") && at(8, b"WEBP") {
        Some("image/webp")
    } else if at(0, b"RIFF") && at(8, b"WAVE") {
        Some("audio/wav")
    } else if at(4, b"ftyp") {
        if at(8, b"qt  ") {
            Some("video/quicktime")
        } else if at(8, b"M4A ") || at(8, b"M4B ") {
            Some("audio/mp4")
        } else if at(8, b"avif") || at(8, b"avis") {
            Some("image/avif")
        } else if at(8, b"heic") || at(8, b"heix") || at(8, b"mif1") {
            Some("image/heic")
        } else {
            Some("video/mp4")
        }
    } else if at(0, b"\x1a\x45\xdf\xa3") {
        Some("video/webm")
    } else if at(0, b"OggS") {
        Some("audio/ogg")
    } else if at(0, b"fLaC") {
        Some("audio/flac")
    } else if at(0, b"ID3") {
        Some("audio/mpeg")
    } else if h.len() >= 2 && h[0] == 0xff && (h[1] & 0xf6) == 0xf0 {
        Some("audio/aac")
    } else if h.len() >= 2 && h[0] == 0xff && (h[1] & 0xe0) == 0xe0 {
        Some("audio/mpeg")
    } else {
        None
    }
}

/// The media kind of a MIME type.
pub fn kind_of(mime: &str) -> Option<MediaKind> {
    let top = mime.split('/').next()?;
    match top {
        "image" => Some(MediaKind::Image),
        "video" => Some(MediaKind::Video),
        "audio" => Some(MediaKind::Audio),
        _ => None,
    }
}

fn kind_word(k: MediaKind) -> &'static str {
    match k {
        MediaKind::Image => "image",
        MediaKind::Video => "video",
        MediaKind::Audio => "audio",
    }
}

fn ext_for(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "video/mp4" => "mp4",
        "video/quicktime" => "mov",
        "video/webm" => "webm",
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        "audio/mpeg" => "mp3",
        "audio/mp4" => "m4a",
        "audio/ogg" => "ogg",
        "audio/flac" => "flac",
        "audio/aac" => "aac",
        _ => "bin",
    }
}

/// A parsed `data:` URI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataUri<'a> {
    pub mime: Option<&'a str>,
    pub base64: bool,
    pub payload: &'a str,
}

/// Splits `data:[<mime>][;params][;base64],<payload>`.
pub fn parse_data_uri(s: &str) -> Option<DataUri<'_>> {
    let rest = s.strip_prefix("data:")?;
    let (head, payload) = rest.split_once(',')?;
    let mut parts = head.split(';');
    let mime = parts.next().filter(|m| !m.is_empty());
    let base64 = parts.any(|p| p.eq_ignore_ascii_case("base64"));
    Some(DataUri { mime, base64, payload })
}

/// Decodes a base64 data URI payload (standard or URL-safe alphabet,
/// whitespace ignored).
fn decode_b64(p: &str) -> Option<Vec<u8>> {
    let clean: String = p.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    use base64::engine::general_purpose as g;
    let pad = base64::engine::GeneralPurposeConfig::new()
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent);
    let std = base64::engine::GeneralPurpose::new(&base64::alphabet::STANDARD, pad);
    let url = base64::engine::GeneralPurpose::new(&base64::alphabet::URL_SAFE, pad);
    std.decode(&clean)
        .or_else(|_| url.decode(&clean))
        .or_else(|_| g::STANDARD.decode(&clean))
        .ok()
}

/// Stages media for requests.
#[derive(Clone)]
pub struct Ingestor {
    uploads: Option<Arc<UploadStore>>,
    prober: Arc<dyn Prober>,
}

impl std::fmt::Debug for Ingestor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ingestor").finish_non_exhaustive()
    }
}

impl Default for Ingestor {
    fn default() -> Self {
        Self::new(None, Arc::new(DefaultProber::default()))
    }
}

impl Ingestor {
    pub fn new(uploads: Option<Arc<UploadStore>>, prober: Arc<dyn Prober>) -> Self {
        Self { uploads, prober }
    }

    /// Stages every media ref of `req` into `dir` (created), in request order:
    /// keyframes, references, then audio. On error, files staged so far stay
    /// in `dir` for the caller to remove.
    pub async fn stage(
        &self,
        req: &GenerationRequest,
        policy: &IngestPolicy,
        dir: &Path,
        now: OffsetDateTime,
    ) -> Result<StagedInputs, ApiError> {
        if req.media_refs().next().is_none() {
            return Ok(StagedInputs::default());
        }
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(|e| ApiError::internal(format!("creating input dir: {e}")))?;
        let mut out = StagedInputs::default();
        for (i, k) in req.keyframes.iter().enumerate() {
            let param = format!("keyframes[{i}]");
            let m = self.stage_one(&k.image, MediaKind::Image, policy, dir, &param, now).await?;
            out.keyframes.push((k.at, m));
        }
        for (i, r) in req.references.iter().enumerate() {
            let param = format!("references[{i}]");
            let m = self.stage_one(&r.media, r.kind, policy, dir, &param, now).await?;
            out.references.push((r.kind, m));
        }
        if let Some(a) = &req.audio_in {
            out.audio_in = Some(self.stage_one(&a.media, MediaKind::Audio, policy, dir, "audio", now).await?);
        }
        if let Some(e) = &req.edit {
            out.video_in = Some(self.stage_one(&e.video, MediaKind::Video, policy, dir, "video_url", now).await?);
        }
        Ok(out)
    }

    /// Stages one ref as `<dir>/<n>.<ext>`.
    pub async fn stage_one(
        &self,
        r: &MediaRef,
        kind: MediaKind,
        policy: &IngestPolicy,
        dir: &Path,
        param: &str,
        now: OffsetDateTime,
    ) -> Result<StagedMedia, ApiError> {
        let lim = *policy.limits(kind);
        let tmp = dir.join(format!(".in-{}", crate::random_token()));
        let io = |e: std::io::Error| ApiError::internal(format!("staging input: {e}"));
        let (bytes, declared): (u64, Option<String>) = match r {
            MediaRef::ProviderFile(_) => {
                return Err(ApiError::unsupported(GapId::ProviderFiles).with_param(param));
            }
            MediaRef::DataUri(s) => {
                let d = parse_data_uri(s)
                    .ok_or_else(|| ApiError::invalid_param(param, format!("`{param}` is not a valid data URI")))?;
                if !d.base64 {
                    return Err(ApiError::invalid_param(param, "data URIs must be base64 encoded"));
                }
                if d.payload.len() as u64 > lim.data_uri_max_encoded {
                    return Err(ApiError::payload_too_large(format!(
                        "`{param}` data URI exceeds {} bytes encoded",
                        lim.data_uri_max_encoded
                    ))
                    .with_param(param));
                }
                let data = decode_b64(d.payload)
                    .ok_or_else(|| ApiError::invalid_param(param, format!("`{param}` has invalid base64")))?;
                if data.len() as u64 > lim.max_bytes {
                    return Err(too_large(param, lim.max_bytes));
                }
                tokio::fs::write(&tmp, &data).await.map_err(io)?;
                (data.len() as u64, d.mime.map(str::to_owned))
            }
            MediaRef::Upload(id) => {
                let f = self
                    .uploads
                    .as_ref()
                    .and_then(|u| u.resolve(id, now))
                    .ok_or_else(|| ApiError::invalid_param(param, format!("upload `{id}` was not found or has expired")))?;
                if f.bytes > lim.max_bytes {
                    return Err(too_large(param, lim.max_bytes));
                }
                tokio::fs::copy(&f.path, &tmp).await.map_err(io)?;
                (f.bytes, f.mime)
            }
            MediaRef::Http(url) => fetch::fetch_to(url, &lim, policy, &tmp, param).await?,
        };
        let result = self.finish(&tmp, bytes, declared, kind, dir, param).await;
        if result.is_err() {
            let _ = tokio::fs::remove_file(&tmp).await;
        }
        result
    }

    async fn finish(
        &self,
        tmp: &Path,
        bytes: u64,
        declared: Option<String>,
        kind: MediaKind,
        dir: &Path,
        param: &str,
    ) -> Result<StagedMedia, ApiError> {
        let mut head = [0u8; 32];
        let n = {
            use tokio::io::AsyncReadExt;
            let mut f = tokio::fs::File::open(tmp)
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
            let mut n = 0;
            while n < head.len() {
                let k = f.read(&mut head[n..]).await.map_err(|e| ApiError::internal(e.to_string()))?;
                if k == 0 {
                    break;
                }
                n += k;
            }
            n
        };
        if n == 0 {
            return Err(ApiError::invalid_param(param, format!("`{param}` is empty")));
        }
        let declared = declared
            .map(|m| m.split(';').next().unwrap_or(&m).trim().to_ascii_lowercase())
            .filter(|m| !m.is_empty() && m != "application/octet-stream" && m != "binary/octet-stream");
        let mut mime = match sniff_mime(&head[..n]) {
            Some(s) => s.to_owned(),
            None => declared.ok_or_else(|| {
                ApiError::unsupported_media(format!("`{param}`: unrecognized {} format", kind_word(kind))).with_param(param)
            })?,
        };
        if kind_of(&mime) != Some(kind) {
            return Err(ApiError::unsupported_media(format!(
                "`{param}`: expected {} input, got `{mime}`",
                kind_word(kind)
            ))
            .with_param(param));
        }
        let mut path = dir.join(format!("{}.{}", crate::random_token(), ext_for(&mime)));
        tokio::fs::rename(tmp, &path)
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let mut bytes = bytes;
        if kind == MediaKind::Image && matches!(mime.as_str(), "image/png" | "image/jpeg") {
            let (src, out) = (path.clone(), dir.join(format!("{}.png", crate::random_token())));
            let done = tokio::task::spawn_blocking(move || apply_exif_orientation(&src, &out).map(|b| b.map(|b| (out, b))))
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
            match done {
                Ok(Some((out, b))) => {
                    let _ = tokio::fs::remove_file(&path).await;
                    (path, mime, bytes) = (out, "image/png".to_owned(), b);
                }
                Ok(None) => {}
                Err(e) => {
                    let _ = tokio::fs::remove_file(&path).await;
                    return Err(ApiError::unsupported_media(format!("`{param}`: the image could not be decoded: {e}")).with_param(param));
                }
            }
        }
        let probe = match self.prober.probe(&path, kind, &mime).await {
            Ok(p) => p,
            Err(e) => {
                let _ = tokio::fs::remove_file(&path).await;
                return Err(if e.param.is_none() { e.with_param(param) } else { e });
            }
        };
        Ok(StagedMedia { path, mime, bytes, probe })
    }
}

/// Bakes a JPEG/PNG's EXIF orientation into its pixels: when the image
/// carries an orientation other than "as stored", it is decoded, rotated or
/// flipped upright and written losslessly to `out` as PNG (without EXIF);
/// returns the new file's size. `None` when there is nothing to apply (only
/// the headers are read). Every consumer downstream (the canvas choice, the
/// gateway's copy to a worker, each pipeline's decode) then sees the image
/// the way a viewer displays it.
pub fn apply_exif_orientation(path: &Path, out: &Path) -> std::io::Result<Option<u64>> {
    use image::metadata::Orientation;
    use image::ImageDecoder as _;
    let mut decoder = image::ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .map_err(std::io::Error::other)?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    if orientation == Orientation::NoTransforms {
        return Ok(None);
    }
    let mut img = image::DynamicImage::from_decoder(decoder).map_err(std::io::Error::other)?;
    img.apply_orientation(orientation);
    img.save_with_format(out, image::ImageFormat::Png).map_err(std::io::Error::other)?;
    Ok(Some(std::fs::metadata(out)?.len()))
}

/// Fetches a public `url` into `dest` under `policy` (the ingestion SSRF
/// guard, redirect rules, and `kind`'s size and time limits); returns the
/// byte count. A gateway worker uses it for inputs the client gave as a
/// URL (docs/serve/gateway.md §3).
pub async fn fetch_public(url: &url::Url, kind: MediaKind, policy: &IngestPolicy, dest: &Path, param: &str) -> Result<u64, ApiError> {
    fetch::fetch_to(url, policy.limits(kind), policy, dest, param).await.map(|(n, _)| n)
}

fn too_large(param: &str, max: u64) -> ApiError {
    ApiError::payload_too_large(format!("`{param}` exceeds {max} bytes")).with_param(param)
}

#[cfg(feature = "fetch")]
mod fetch {
    use super::*;
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    /// Fetches `url` into `dest`, returning `(bytes, content_type)`.
    pub async fn fetch_to(
        url: &url::Url,
        lim: &KindLimits,
        policy: &IngestPolicy,
        dest: &Path,
        param: &str,
    ) -> Result<(u64, Option<String>), ApiError> {
        match tokio::time::timeout(lim.timeout, fetch_inner(url, lim, policy, dest, param)).await {
            Ok(r) => r,
            Err(_) => {
                let _ = tokio::fs::remove_file(dest).await;
                Err(ApiError::invalid_param(
                    param,
                    format!("fetching `{param}` timed out after {} s", lim.timeout.as_secs()),
                ))
            }
        }
    }

    async fn fetch_inner(
        url: &url::Url,
        lim: &KindLimits,
        policy: &IngestPolicy,
        dest: &Path,
        param: &str,
    ) -> Result<(u64, Option<String>), ApiError> {
        let bad = |m: String| ApiError::invalid_param(param, m);
        let mut url = url.clone();
        let mut hops = 0u8;
        let resp = loop {
            let addrs = crate::net::resolve_target(&url, &policy.target)
                .await
                .map_err(|e| bad(format!("`{param}`: {e}")))?;
            let mut b = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy();
            if let Some(url::Host::Domain(d)) = url.host() {
                b = b.resolve_to_addrs(d, &addrs);
            }
            let client = b.build().map_err(|e| ApiError::internal(e.to_string()))?;
            let resp = client
                .get(url.clone())
                .send()
                .await
                .map_err(|e| bad(format!("fetching `{param}` failed: {}", without_url(&e))))?;
            if resp.status().is_redirection() {
                if !policy.follow_redirects || hops >= policy.max_redirects {
                    return Err(bad(format!("`{param}`: redirects are not allowed")));
                }
                let loc = resp
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|l| url.join(l).ok())
                    .ok_or_else(|| bad(format!("`{param}`: bad redirect")))?;
                url = loc;
                hops += 1;
                continue;
            }
            break resp;
        };
        if !resp.status().is_success() {
            return Err(bad(format!("fetching `{param}` returned HTTP {}", resp.status().as_u16())));
        }
        if resp.content_length().is_some_and(|n| n > lim.max_bytes) {
            return Err(too_large(param, lim.max_bytes));
        }
        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let io = |e: std::io::Error| ApiError::internal(format!("staging input: {e}"));
        let mut f = tokio::fs::File::create(dest).await.map_err(io)?;
        let mut n = 0u64;
        let mut body = resp.bytes_stream();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|e| bad(format!("fetching `{param}` failed: {}", without_url(&e))))?;
            n += chunk.len() as u64;
            if n > lim.max_bytes {
                drop(f);
                let _ = tokio::fs::remove_file(dest).await;
                return Err(too_large(param, lim.max_bytes));
            }
            f.write_all(&chunk).await.map_err(io)?;
        }
        f.flush().await.map_err(io)?;
        Ok((n, ct))
    }

    fn without_url(e: &reqwest::Error) -> String {
        let mut s = e.to_string();
        if let Some(u) = e.url() {
            s = s.replace(u.as_str(), "<url>");
        }
        s
    }
}

#[cfg(not(feature = "fetch"))]
mod fetch {
    use super::*;
    pub async fn fetch_to(
        _url: &url::Url,
        _lim: &KindLimits,
        _policy: &IngestPolicy,
        _dest: &Path,
        param: &str,
    ) -> Result<(u64, Option<String>), ApiError> {
        Err(ApiError::invalid_param(
            param,
            "remote media URLs are not enabled on this server (use a data URI or an upload)",
        ))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn wav(rate: u32, channels: u16, frames: u32) -> Vec<u8> {
        let block = 2 * channels;
        let data = frames * u32::from(block);
        let mut v = b"RIFF".to_vec();
        v.extend((36 + data).to_le_bytes());
        v.extend(b"WAVEfmt ");
        v.extend(16u32.to_le_bytes());
        v.extend(1u16.to_le_bytes());
        v.extend(channels.to_le_bytes());
        v.extend(rate.to_le_bytes());
        v.extend((rate * u32::from(block)).to_le_bytes());
        v.extend(block.to_le_bytes());
        v.extend(16u16.to_le_bytes());
        // An extra chunk before `data` (LIST), odd-sized to test padding.
        v.extend(b"LIST");
        v.extend(3u32.to_le_bytes());
        v.extend([1, 2, 3, 0]);
        v.extend(b"data");
        v.extend(data.to_le_bytes());
        v.extend(vec![0u8; data as usize]);
        v
    }

    #[test]
    fn wav_headers_give_rate_and_length() {
        let w = wav(48_000, 2, 24_000);
        let (rate, secs) = parse_wav_header(&w, w.len() as u64).unwrap();
        assert_eq!(rate, 48_000);
        assert!((secs - 0.5).abs() < 1e-9);
        let w = wav(16_000, 1, 48_000);
        assert_eq!(parse_wav_header(&w, w.len() as u64), Some((16_000, 3.0)));
        assert_eq!(parse_wav_header(b"RIFF\0\0\0\0AVI LIST", 16), None);
        assert_eq!(parse_wav_header(b"ID3", 3), None);
    }
    use fastvideo_protocol::{Anchor, ErrorKind, Keyframe, ProtocolId, Reference, UploadId};

    pub fn png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([10, 20, 30]));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    /// A `w`x`h` JPEG (as stored) carrying EXIF `orientation` (1..=8); the
    /// left half is dark and the right half bright, so a rotation shows.
    pub fn jpeg_exif(w: u32, h: u32, orientation: u16) -> Vec<u8> {
        use image::ImageEncoder as _;
        let img = image::RgbImage::from_fn(w, h, |x, _| if x < w / 2 { image::Rgb([0, 0, 0]) } else { image::Rgb([255, 255, 255]) });
        // A little-endian TIFF header with one IFD entry: 0x0112 SHORT = orientation.
        let mut exif = vec![0x49, 0x49, 0x2a, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0];
        exif.extend_from_slice(&orientation.to_le_bytes());
        exif.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        let mut out = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95);
        enc.set_exif_metadata(exif).unwrap();
        enc.write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8).unwrap();
        out
    }

    #[tokio::test]
    async fn exif_orientation_is_baked_in() {
        let dir = tmpdir();
        let ing = Ingestor::new(None, Arc::new(DefaultProber { ffprobe: None }));
        let now = OffsetDateTime::now_utc();
        // Stored 64x32 landscape, orientation 6 (rotate 90 CW to display):
        // the viewer sees 32x64 portrait with the bright half at the bottom.
        let s = ing
            .stage(&i2v(MediaRef::DataUri(data_uri("image/jpeg", &jpeg_exif(64, 32, 6)))), &IngestPolicy::default(), &dir, now)
            .await
            .unwrap();
        let m = &s.keyframes[0].1;
        assert_eq!(m.probe.dims(), Some((32, 64)));
        assert_eq!(m.mime, "image/png");
        assert_eq!(m.path.extension().unwrap(), "png");
        assert_eq!(m.bytes, std::fs::metadata(&m.path).unwrap().len());
        let img = image::open(&m.path).unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (32, 64));
        assert!(img.get_pixel(16, 4)[0] < 64 && img.get_pixel(16, 60)[0] > 192, "rotated clockwise");
        // Only the upright file stays staged.
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        // 8 (rotate 90 CCW) and 3 (180) too; 1 is left untouched.
        for (o, dims, mime) in [(8, (32, 64), "image/png"), (3, (64, 32), "image/png"), (1, (64, 32), "image/jpeg")] {
            let s = ing
                .stage(&i2v(MediaRef::DataUri(data_uri("image/jpeg", &jpeg_exif(64, 32, o)))), &IngestPolicy::default(), &dir, now)
                .await
                .unwrap();
            assert_eq!((s.keyframes[0].1.probe.dims(), s.keyframes[0].1.mime.as_str()), (Some(dims), mime), "orientation {o}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    pub fn data_uri(mime: &str, b: &[u8]) -> String {
        format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(b))
    }

    fn tmpdir() -> PathBuf {
        std::env::temp_dir().join(format!("fvkit-ing-{}", crate::random_token()))
    }

    fn i2v(r: MediaRef) -> GenerationRequest {
        let mut g = GenerationRequest::text(ProtocolId::Fal, "m", "p");
        g.keyframes.push(Keyframe { at: Anchor::First, image: r });
        g
    }

    #[test]
    fn sniffing() {
        assert_eq!(sniff_mime(&png(1, 1)), Some("image/png"));
        assert_eq!(sniff_mime(b"\xff\xd8\xff\xe0...."), Some("image/jpeg"));
        assert_eq!(sniff_mime(b"\0\0\0\x20ftypisom\0\0"), Some("video/mp4"));
        assert_eq!(sniff_mime(b"\0\0\0\x20ftypqt  \0\0"), Some("video/quicktime"));
        assert_eq!(sniff_mime(b"\0\0\0\x20ftypM4A \0\0"), Some("audio/mp4"));
        assert_eq!(sniff_mime(b"RIFF\0\0\0\0WAVEfmt "), Some("audio/wav"));
        assert_eq!(sniff_mime(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));
        assert_eq!(sniff_mime(b"ID3\x04"), Some("audio/mpeg"));
        assert_eq!(sniff_mime(b"<html>"), None);
    }

    #[test]
    fn data_uri_parse() {
        let d = parse_data_uri("data:image/png;base64,AAAA").unwrap();
        assert_eq!((d.mime, d.base64, d.payload), (Some("image/png"), true, "AAAA"));
        let d = parse_data_uri("data:,hello").unwrap();
        assert_eq!((d.mime, d.base64), (None, false));
        assert!(parse_data_uri("data:nocomma").is_none());
        assert_eq!(decode_b64("aGk"), Some(b"hi".to_vec()));
        assert_eq!(decode_b64("a Gk=\n"), Some(b"hi".to_vec()));
    }

    #[test]
    fn ffprobe_json() {
        let v = serde_json::json!({
            "streams": [
                {"codec_type":"video","width":1344,"height":768,"avg_frame_rate":"24/1","duration":"5.166667"},
                {"codec_type":"audio","sample_rate":"32000"}
            ],
            "format": {"duration":"5.2"}
        });
        let p = parse_ffprobe(&v).unwrap();
        assert_eq!((p.width, p.height, p.audio_rate), (Some(1344), Some(768), Some(32000)));
        assert_eq!(p.fps, Some(24.0));
        assert_eq!(p.duration_s, Some(5.2));
    }

    #[tokio::test]
    async fn data_uri_staging_and_limits() {
        let dir = tmpdir();
        let ing = Ingestor::default();
        let now = OffsetDateTime::now_utc();
        let img = png(64, 48);
        let s = ing
            .stage(&i2v(MediaRef::DataUri(data_uri("image/png", &img))), &IngestPolicy::default(), &dir, now)
            .await
            .unwrap();
        let m = &s.keyframes[0].1;
        assert_eq!((m.mime.as_str(), m.bytes), ("image/png", img.len() as u64));
        assert_eq!(m.probe.dims(), Some((64, 48)));
        assert!(m.path.starts_with(&dir) && m.path.extension().unwrap() == "png");

        // Declared type lies: magic bytes win.
        let s = ing
            .stage(&i2v(MediaRef::DataUri(data_uri("image/jpeg", &img))), &IngestPolicy::default(), &dir, now)
            .await
            .unwrap();
        assert_eq!(s.keyframes[0].1.mime, "image/png");

        let mut tight = IngestPolicy::ltx();
        tight.image.data_uri_max_encoded = 16;
        let e = ing.stage(&i2v(MediaRef::DataUri(data_uri("image/png", &img))), &tight, &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::PayloadTooLarge);
        assert_eq!(e.param.as_deref(), Some("keyframes[0]"));
        let mut tight = IngestPolicy::ltx();
        tight.image.max_bytes = 16;
        let e = ing.stage(&i2v(MediaRef::DataUri(data_uri("image/png", &img))), &tight, &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::PayloadTooLarge);

        // Wrong kind, unknown bytes, not base64, bad base64, corrupt png.
        let mp4 = b"\0\0\0\x20ftypisom\0\0\0\0isomavc1";
        let e = ing.stage(&i2v(MediaRef::DataUri(data_uri("video/mp4", mp4))), &IngestPolicy::default(), &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::UnsupportedMedia);
        let e = ing.stage(&i2v(MediaRef::DataUri(data_uri("application/octet-stream", b"????"))), &IngestPolicy::default(), &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::UnsupportedMedia);
        let e = ing.stage(&i2v(MediaRef::DataUri("data:image/png,abc".into())), &IngestPolicy::default(), &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidRequest);
        let e = ing.stage(&i2v(MediaRef::DataUri("data:image/png;base64,!!!".into())), &IngestPolicy::default(), &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidRequest);
        let mut broken = img.clone();
        broken.truncate(12);
        let e = ing.stage(&i2v(MediaRef::DataUri(data_uri("image/png", &broken))), &IngestPolicy::default(), &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::UnsupportedMedia);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn provider_files_and_uploads() {
        let dir = tmpdir();
        let now = OffsetDateTime::now_utc();
        let urls = crate::artifacts::LocalUrls {
            public_base: url::Url::parse("http://h").unwrap(),
            key: crate::artifacts::UrlKey::new("k"),
        };
        let up = Arc::new(UploadStore::new(dir.join("up"), urls).unwrap());
        let t = up.create(None, None, now).unwrap();
        let ing = Ingestor::new(Some(up.clone()), Arc::new(DefaultProber { ffprobe: None }));
        let e = ing.stage(&i2v(MediaRef::ProviderFile("mm_file://1".into())), &IngestPolicy::default(), &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::Unsupported(GapId::ProviderFiles));
        let e = ing.stage(&i2v(MediaRef::Upload(UploadId(t.token.clone()))), &IngestPolicy::default(), &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidRequest, "not uploaded yet");
        up.put(&t.token, None, axum::body::Body::from(png(8, 8)), now).await.unwrap();
        let s = ing.stage(&i2v(MediaRef::Upload(UploadId(t.token.clone()))), &IngestPolicy::default(), &dir, now).await.unwrap();
        assert_eq!(s.keyframes[0].1.probe.dims(), Some((8, 8)));
        let mut tight = IngestPolicy::ltx();
        tight.image.max_bytes = 10;
        let e = ing.stage(&i2v(MediaRef::Upload(UploadId(t.token.clone()))), &tight, &dir, now).await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::PayloadTooLarge);

        // Reference kinds are checked (audio ref given an image).
        let mut g = GenerationRequest::text(ProtocolId::MiniMaxV2, "m", "p");
        g.references.push(Reference { kind: MediaKind::Audio, media: MediaRef::DataUri(data_uri("image/png", &png(2, 2))) });
        let e = ing.stage(&g, &IngestPolicy::default(), &dir, now).await.unwrap_err();
        assert_eq!((e.kind, e.param.as_deref()), (ErrorKind::UnsupportedMedia, Some("references[0]")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(feature = "fetch"))]
    #[tokio::test]
    async fn http_refused_without_fetch() {
        let dir = tmpdir();
        let e = Ingestor::default()
            .stage(&i2v(MediaRef::Http(url::Url::parse("https://example.com/a.png").unwrap())), &IngestPolicy::default(), &dir, OffsetDateTime::now_utc())
            .await
            .unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidRequest);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// HTTP fetches against a local axum server (feature `fetch`).
    #[cfg(feature = "fetch")]
    mod http {
        use super::*;
        use axum::routing::get;
        use axum::Router;

        async fn server() -> std::net::SocketAddr {
            let img = png(32, 16);
            let big = vec![0u8; 64 * 1024];
            let app = Router::new()
                .route("/img.png", get({
                    let img = img.clone();
                    move || async move { ([("content-type", "image/png")], img) }
                }))
                .route("/noctype", get({
                    let img = img.clone();
                    move || async move { img }
                }))
                .route("/big.png", get(move || async move {
                    let mut b = png(1, 1);
                    b.extend_from_slice(&big);
                    ([("content-type", "image/png")], b)
                }))
                .route("/chunked.png", get(|| async {
                    let chunks = futures::stream::iter((0..64).map(|i| {
                        let mut c = if i == 0 { png(1, 1) } else { Vec::new() };
                        c.extend_from_slice(&[0u8; 1024]);
                        Ok::<_, std::io::Error>(bytes::Bytes::from(c))
                    }));
                    axum::body::Body::from_stream(chunks)
                }))
                .route("/slow.png", get(|| async {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    png(1, 1)
                }))
                .route("/redirect", get(|| async {
                    (axum::http::StatusCode::FOUND, [("location", "/img.png")])
                }))
                .route("/page", get(|| async { ([("content-type", "text/html")], "<html></html>") }))
                .route("/missing", get(|| async { axum::http::StatusCode::NOT_FOUND }));
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = l.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
            addr
        }

        fn open_policy() -> IngestPolicy {
            let mut p = IngestPolicy::default();
            p.target.allow_private = true;
            p
        }

        async fn stage(url: &str, p: &IngestPolicy) -> Result<StagedInputs, ApiError> {
            let dir = tmpdir();
            let r = Ingestor::default()
                .stage(&i2v(MediaRef::Http(url::Url::parse(url).unwrap())), p, &dir, OffsetDateTime::now_utc())
                .await;
            let _ = std::fs::remove_dir_all(&dir);
            r
        }

        #[tokio::test]
        async fn fetch_rules() {
            let a = server().await;
            let base = format!("http://127.0.0.1:{}", a.port());
            let s = stage(&format!("{base}/img.png"), &open_policy()).await.unwrap();
            assert_eq!(s.keyframes[0].1.probe.dims(), Some((32, 16)));
            // No content type: sniffed.
            assert!(stage(&format!("{base}/noctype"), &open_policy()).await.is_ok());

            // SSRF guard: loopback refused by default.
            let e = stage(&format!("{base}/img.png"), &IngestPolicy::default()).await.unwrap_err();
            assert_eq!(e.kind, ErrorKind::InvalidRequest);
            assert!(e.message.contains("non-public"), "{}", e.message);
            let e = stage(&format!("http://localhost:{}/img.png", a.port()), &IngestPolicy::default()).await.unwrap_err();
            assert_eq!(e.kind, ErrorKind::InvalidRequest);

            // Size: declared and streamed.
            let mut p = open_policy();
            p.image.max_bytes = 16 * 1024;
            assert_eq!(stage(&format!("{base}/big.png"), &p).await.unwrap_err().kind, ErrorKind::PayloadTooLarge);
            assert_eq!(stage(&format!("{base}/chunked.png"), &p).await.unwrap_err().kind, ErrorKind::PayloadTooLarge);

            // Timeout.
            let mut p = open_policy();
            p.image.timeout = Duration::from_millis(300);
            let e = stage(&format!("{base}/slow.png"), &p).await.unwrap_err();
            assert!(e.message.contains("timed out"), "{}", e.message);

            // Redirects: followed by default, refused under LTX rules.
            assert!(stage(&format!("{base}/redirect"), &open_policy()).await.is_ok());
            let mut ltxish = IngestPolicy::ltx();
            ltxish.target = TargetPolicy { https_only: false, allow_ip_literals: false, allow_private: true };
            let host = format!("http://localhost:{}", a.port());
            assert!(stage(&format!("{host}/img.png"), &ltxish).await.is_ok(), "domain name allowed");
            let e = stage(&format!("{host}/redirect"), &ltxish).await.unwrap_err();
            assert!(e.message.contains("redirects"), "{}", e.message);
            let e = stage(&format!("{base}/img.png"), &ltxish).await.unwrap_err();
            assert!(e.message.contains("IP address"), "{}", e.message);
            let e = stage(&format!("{host}/img.png"), &IngestPolicy::ltx()).await.unwrap_err();
            assert!(e.message.contains("https"), "{}", e.message);

            // Type and status.
            assert_eq!(stage(&format!("{base}/page"), &open_policy()).await.unwrap_err().kind, ErrorKind::UnsupportedMedia);
            let e = stage(&format!("{base}/missing"), &open_policy()).await.unwrap_err();
            assert!(e.message.contains("404"));
        }
    }
}
