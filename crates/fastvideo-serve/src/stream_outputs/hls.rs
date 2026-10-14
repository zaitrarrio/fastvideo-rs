//! HLS read-back for the live Cloudflare checks (the `outputs_live` test and
//! `examples/cf_live_check.rs`): what an HLS playlist serves, and whether
//! its audio is really there.
//!
//! Cloudflare Stream serves the audio of an RTMPS / SRT input as its own
//! rendition: the master playlist lists `#EXT-X-MEDIA:TYPE=AUDIO,
//! GROUP-ID="group_audio",…,URI="…"` and every video variant names the group
//! (`AUDIO="group_audio"`), while the video segments carry no audio
//! (docs/serve/e2e/h3-test-round-2026-10-14.md, finding 4). A reader that
//! decodes only a video variant's segments sees no audio at all. So:
//!
//! - [`MasterPlaylist::parse`] reads the variants and the `EXT-X-MEDIA`
//!   renditions; [`MasterPlaylist::audio_for`] says where the chosen
//!   variant's audio is: a separate rendition (its media playlist URI), muxed
//!   into the variant's own segments (no `AUDIO` group, or a group whose
//!   rendition has no `URI`, which RFC 8216 §4.3.4.1 defines as "in the
//!   variant"), or a group the playlist does not list yet.
//! - [`MediaPlaylist::parse`] reads a media playlist's segments, init map
//!   and media sequence.
//! - [`AudioLevels`] measures decoded PCM (RMS per second and peak, dBFS)
//!   and [`AudioLevels::audible`] applies the same bar as the WHEP check
//!   (`scripts/serve/e2e/whep_audio.py`: seconds above -60 dBFS RMS).
//! - `read_audio` (feature `http-client`) follows the audio of a live master
//!   playlist from the oldest segment it lists, accumulates segments for a
//!   window, decodes them with ffmpeg and measures them.
//!
//! Pure parsing; no network outside `read_audio`.

/// A second whose RMS is above this is "not silent" (the WHEP check's bar).
pub const AUDIBLE_RMS_DBFS: f64 = -60.0;
/// The loudest sample must reach this.
pub const AUDIBLE_PEAK_DBFS: f64 = -40.0;
/// How many seconds above [`AUDIBLE_RMS_DBFS`] a read needs.
pub const AUDIBLE_MIN_SECONDS: usize = 2;
/// What digital silence reads as (instead of -inf).
pub const FLOOR_DBFS: f64 = -120.0;

/// One `#EXT-X-STREAM-INF` variant.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Variant {
    pub bandwidth: u64,
    pub resolution: Option<(u32, u32)>,
    pub codecs: Option<String>,
    /// `AUDIO` group id.
    pub audio_group: Option<String>,
    pub uri: String,
    /// The `#EXT-X-STREAM-INF` line as written.
    pub info: String,
}

/// One `#EXT-X-MEDIA` rendition.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Rendition {
    /// `AUDIO`, `VIDEO`, `SUBTITLES`, `CLOSED-CAPTIONS`.
    pub kind: String,
    pub group_id: String,
    pub name: Option<String>,
    pub language: Option<String>,
    pub default: bool,
    pub autoselect: bool,
    /// None: the rendition is in the variant's own segments.
    pub uri: Option<String>,
}

/// A master playlist.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct MasterPlaylist {
    pub variants: Vec<Variant>,
    pub media: Vec<Rendition>,
}

/// Where a variant's audio is.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioSource<'a> {
    /// A separate rendition with its own media playlist.
    Rendition(&'a Rendition),
    /// In the variant's own segments (no `AUDIO` group, or the group's
    /// rendition has no `URI`).
    Muxed,
    /// The variant names a group the master lists no audio rendition of
    /// (yet: Cloudflare lists it some time after the video).
    MissingGroup(String),
}

/// The attributes of a tag (`KEY=value,KEY="quoted, with commas"`), quotes
/// removed, in order.
pub fn attributes(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = s.trim();
    while !rest.is_empty() {
        let Some(eq) = rest.find('=') else { break };
        let key = rest[..eq].trim().trim_start_matches(',').trim().to_owned();
        rest = &rest[eq + 1..];
        let value;
        if let Some(q) = rest.strip_prefix('"') {
            let end = q.find('"').unwrap_or(q.len());
            value = q[..end].to_owned();
            rest = q.get(end + 1..).unwrap_or("");
        } else {
            let end = rest.find(',').unwrap_or(rest.len());
            value = rest[..end].trim().to_owned();
            rest = &rest[end..];
        }
        rest = rest.trim_start().trim_start_matches(',');
        out.push((key, value));
    }
    out
}

fn attr<'a>(a: &'a [(String, String)], key: &str) -> Option<&'a str> {
    a.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

impl MasterPlaylist {
    /// Parses a master playlist. A media playlist (segments, no
    /// `#EXT-X-STREAM-INF`) parses to no variants: see [`is_media_playlist`].
    pub fn parse(text: &str) -> Self {
        let mut m = MasterPlaylist::default();
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        while let Some(l) = lines.next() {
            if let Some(a) = l.strip_prefix("#EXT-X-STREAM-INF:") {
                let a = attributes(a);
                // The URI is the next line that is not a tag or comment.
                let uri = lines.by_ref().find(|u| !u.starts_with('#'));
                let Some(uri) = uri else { break };
                m.variants.push(Variant {
                    bandwidth: attr(&a, "BANDWIDTH")
                        .and_then(|b| b.parse().ok())
                        .unwrap_or(0),
                    resolution: attr(&a, "RESOLUTION").and_then(|r| {
                        let (w, h) = r.split_once(['x', 'X'])?;
                        Some((w.parse().ok()?, h.parse().ok()?))
                    }),
                    codecs: attr(&a, "CODECS").map(str::to_owned),
                    audio_group: attr(&a, "AUDIO").map(str::to_owned),
                    uri: uri.to_owned(),
                    info: l.to_owned(),
                });
            } else if let Some(a) = l.strip_prefix("#EXT-X-MEDIA:") {
                let a = attributes(a);
                let yes = |k: &str| attr(&a, k).is_some_and(|v| v.eq_ignore_ascii_case("YES"));
                m.media.push(Rendition {
                    kind: attr(&a, "TYPE").unwrap_or_default().to_owned(),
                    group_id: attr(&a, "GROUP-ID").unwrap_or_default().to_owned(),
                    name: attr(&a, "NAME").map(str::to_owned),
                    language: attr(&a, "LANGUAGE").map(str::to_owned),
                    default: yes("DEFAULT"),
                    autoselect: yes("AUTOSELECT"),
                    uri: attr(&a, "URI").map(str::to_owned),
                });
            }
        }
        m
    }

    /// The variant with the highest bandwidth (the reader's video choice).
    pub fn best_variant(&self) -> Option<&Variant> {
        self.variants.iter().max_by_key(|v| v.bandwidth)
    }

    /// The variant with the lowest bandwidth (cheapest to fetch for muxed
    /// audio).
    pub fn lightest_variant(&self) -> Option<&Variant> {
        self.variants.iter().min_by_key(|v| v.bandwidth)
    }

    /// Where `v`'s audio is: the `DEFAULT=YES` audio rendition of its group,
    /// else the `AUTOSELECT=YES` one, else the first one listed.
    pub fn audio_for(&self, v: &Variant) -> AudioSource<'_> {
        let Some(g) = &v.audio_group else {
            return AudioSource::Muxed;
        };
        let group: Vec<&Rendition> = self
            .media
            .iter()
            .filter(|r| r.kind.eq_ignore_ascii_case("AUDIO") && &r.group_id == g)
            .collect();
        let pick = group
            .iter()
            .find(|r| r.default)
            .or_else(|| group.iter().find(|r| r.autoselect))
            .or(group.first());
        match pick {
            None => AudioSource::MissingGroup(g.clone()),
            Some(r) if r.uri.is_none() => AudioSource::Muxed,
            Some(r) => AudioSource::Rendition(r),
        }
    }
}

/// A media playlist rather than a master (segments, no variants).
pub fn is_media_playlist(text: &str) -> bool {
    text.lines().any(|l| l.starts_with("#EXTINF"))
        && !text.lines().any(|l| l.starts_with("#EXT-X-STREAM-INF"))
}

/// One media segment.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Segment {
    /// Media sequence number (`EXT-X-MEDIA-SEQUENCE` + position).
    pub sequence: u64,
    pub duration: f64,
    pub uri: String,
}

/// A media playlist.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct MediaPlaylist {
    pub target_duration: Option<f64>,
    pub media_sequence: u64,
    /// `EXT-X-MAP` URI (fMP4 init segment).
    pub init: Option<String>,
    pub segments: Vec<Segment>,
    pub ended: bool,
}

impl MediaPlaylist {
    pub fn parse(text: &str) -> Self {
        let mut p = MediaPlaylist::default();
        let mut dur = 0.0;
        for l in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if let Some(t) = l.strip_prefix("#EXT-X-TARGETDURATION:") {
                p.target_duration = t.trim().parse().ok();
            } else if let Some(t) = l.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
                p.media_sequence = t.trim().parse().unwrap_or(0);
            } else if let Some(m) = l.strip_prefix("#EXT-X-MAP:") {
                p.init = attr(&attributes(m), "URI").map(str::to_owned);
            } else if let Some(d) = l.strip_prefix("#EXTINF:") {
                dur = d
                    .split(',')
                    .next()
                    .and_then(|x| x.trim().parse().ok())
                    .unwrap_or(0.0);
            } else if l.starts_with("#EXT-X-ENDLIST") {
                p.ended = true;
            } else if !l.starts_with('#') {
                let sequence = p.media_sequence + p.segments.len() as u64;
                p.segments.push(Segment {
                    sequence,
                    duration: dur,
                    uri: l.to_owned(),
                });
                dur = 0.0;
            }
        }
        p
    }

    /// Seconds of media listed.
    pub fn seconds(&self) -> f64 {
        self.segments.iter().map(|s| s.duration).sum()
    }
}

/// Loudness of decoded audio.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct AudioLevels {
    pub seconds: f64,
    /// RMS of the whole read, dBFS.
    pub rms_dbfs: f64,
    /// Loudest sample, dBFS.
    pub peak_dbfs: f64,
    /// RMS of each whole or partial second, dBFS.
    pub rms_dbfs_per_second: Vec<f64>,
    /// Seconds above [`AUDIBLE_RMS_DBFS`].
    pub audible_seconds: usize,
}

fn dbfs(x: f64) -> f64 {
    if x > 0.0 {
        (20.0 * x.log10()).max(FLOOR_DBFS)
    } else {
        FLOOR_DBFS
    }
}

impl AudioLevels {
    /// From mono samples in [-1, 1] at `rate` Hz.
    pub fn from_samples(samples: &[f32], rate: u32) -> Self {
        let rate = rate.max(1) as usize;
        let mut per = Vec::new();
        let (mut sum, mut peak) = (0.0f64, 0.0f64);
        for chunk in samples.chunks(rate) {
            let s: f64 = chunk.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            sum += s;
            peak = chunk.iter().fold(peak, |p, &x| p.max(f64::from(x).abs()));
            per.push(round1(dbfs((s / chunk.len() as f64).sqrt())));
        }
        let rms = if samples.is_empty() {
            0.0
        } else {
            (sum / samples.len() as f64).sqrt()
        };
        Self {
            seconds: samples.len() as f64 / rate as f64,
            rms_dbfs: round1(dbfs(rms)),
            peak_dbfs: round1(dbfs(peak)),
            audible_seconds: per.iter().filter(|&&d| d > AUDIBLE_RMS_DBFS).count(),
            rms_dbfs_per_second: per,
        }
    }

    /// From signed 16-bit little-endian mono PCM.
    pub fn from_s16le(pcm: &[u8], rate: u32) -> Self {
        let s: Vec<f32> = pcm
            .chunks_exact(2)
            .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
            .collect();
        Self::from_samples(&s, rate)
    }

    /// Present and not silent: at least [`AUDIBLE_MIN_SECONDS`] seconds above
    /// [`AUDIBLE_RMS_DBFS`] RMS and a peak above [`AUDIBLE_PEAK_DBFS`].
    pub fn audible(&self) -> bool {
        self.audible_seconds >= AUDIBLE_MIN_SECONDS && self.peak_dbfs > AUDIBLE_PEAK_DBFS
    }
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

#[cfg(feature = "http-client")]
pub use readback::{read_audio, AudioReadback};

#[cfg(feature = "http-client")]
mod readback {
    use std::collections::BTreeSet;
    use std::path::Path;
    use std::process::Command;
    use std::time::{Duration, Instant};

    use super::{is_media_playlist, AudioLevels, AudioSource, MasterPlaylist, MediaPlaylist};

    /// What `read_audio` found.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct AudioReadback {
        /// `rendition` (separate `EXT-X-MEDIA TYPE=AUDIO` playlist) or
        /// `muxed` (the variant's own segments).
        pub source: &'static str,
        pub group: Option<String>,
        pub name: Option<String>,
        pub language: Option<String>,
        /// The variant whose audio was followed (`#EXT-X-STREAM-INF` line).
        pub variant: Option<String>,
        /// The media playlist read (relative to the master).
        pub playlist_uri: String,
        /// From the call to the first poll that listed a segment.
        pub listed_after_ms: u64,
        /// The first poll: media sequence of its oldest segment and how
        /// many it listed (a sequence of 0 is the start of the stream).
        pub first_sequence: u64,
        pub first_listed: usize,
        pub polls: usize,
        pub segments: usize,
        pub segment_seconds: f64,
        /// Segments that left the playlist before a poll saw them.
        pub missed_segments: u64,
        pub codec: Option<String>,
        pub sample_rate: Option<u32>,
        pub channels: Option<u32>,
        pub levels: Option<AudioLevels>,
        pub audible: bool,
        /// Raw master and media playlist text of the first read (fixtures).
        #[serde(skip)]
        pub master_text: String,
        #[serde(skip)]
        pub media_text: String,
    }

    async fn get(http: &reqwest::Client, u: &reqwest::Url) -> Result<Vec<u8>, String> {
        let r = http
            .get(u.clone())
            .send()
            .await
            .map_err(|e| e.without_url().to_string())?;
        if !r.status().is_success() {
            return Err(format!("HTTP {} for {}", r.status(), u.path()));
        }
        r.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| e.without_url().to_string())
    }

    async fn text(http: &reqwest::Client, u: &reqwest::Url) -> Result<String, String> {
        Ok(String::from_utf8_lossy(&get(http, u).await?).into_owned())
    }

    /// Follows the audio of the HLS master playlist `master` until
    /// `deadline`: the best (highest-bandwidth) variant's `AUDIO` rendition
    /// when it has one, else the variant's own segments (the lightest
    /// variant: muxed audio is the same in each). Polls until a playlist
    /// lists a segment, then takes every listed segment from the oldest
    /// (the start of the stream while the playlist still holds it) and keeps
    /// polling until `window` seconds of segments are in, the playlist ends
    /// or `deadline` passes. The segments (after the `EXT-X-MAP` init) are
    /// joined into `out`, ffprobe names the audio stream and ffmpeg decodes
    /// it to 48 kHz mono for [`AudioLevels`].
    pub async fn read_audio(
        http: &reqwest::Client,
        master: &str,
        window: Duration,
        deadline: Instant,
        out: &Path,
    ) -> Result<AudioReadback, String> {
        let t0 = Instant::now();
        let base = reqwest::Url::parse(master).map_err(|e| e.to_string())?;
        let mut last = String::from("never polled");
        // 1. The playlist to follow, once it lists a segment.
        let (mut r, media_url, mut media) = loop {
            if Instant::now() >= deadline {
                return Err(format!(
                    "no audio playlist with segments before the deadline (last: {last})"
                ));
            }
            match find(http, &base).await {
                Ok(Some(x)) => break x,
                Ok(None) => last = "the audio playlist lists no segment yet".into(),
                Err(e) => last = e,
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        };
        r.listed_after_ms = t0.elapsed().as_millis() as u64;
        r.first_sequence = media.segments.first().map_or(0, |s| s.sequence);
        r.first_listed = media.segments.len();
        // 2. Accumulate segments from the oldest listed.
        let mut seen = BTreeSet::new();
        let mut file = Vec::new();
        let mut next_seq = None::<u64>;
        if let Some(i) = &media.init {
            file.extend_from_slice(
                &get(http, &media_url.join(i).map_err(|e| e.to_string())?).await?,
            );
        }
        loop {
            r.polls += 1;
            for s in &media.segments {
                if r.segment_seconds >= window.as_secs_f64() || !seen.insert(s.sequence) {
                    continue;
                }
                if let Some(n) = next_seq {
                    r.missed_segments += s.sequence.saturating_sub(n);
                }
                next_seq = Some(s.sequence + 1);
                file.extend_from_slice(
                    &get(http, &media_url.join(&s.uri).map_err(|e| e.to_string())?).await?,
                );
                r.segments += 1;
                r.segment_seconds += s.duration;
            }
            if r.segment_seconds >= window.as_secs_f64()
                || media.ended
                || Instant::now() >= deadline
            {
                break;
            }
            let wait = media.target_duration.unwrap_or(3.0).clamp(1.0, 6.0) / 2.0;
            tokio::time::sleep(Duration::from_secs_f64(wait)).await;
            match text(http, &media_url).await {
                Ok(t) => media = MediaPlaylist::parse(&t),
                Err(e) => tracing::debug!(error = %e, "HLS audio poll"),
            }
        }
        r.segment_seconds = (r.segment_seconds * 1000.0).round() / 1000.0;
        std::fs::write(out, &file).map_err(|e| e.to_string())?;
        let f = out.to_string_lossy().into_owned();
        let probe = Command::new(fastvideo_media::tools::ffprobe_bin())
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-of",
                "json",
                &f,
            ])
            .output()
            .map_err(|e| format!("ffprobe: {e}"))?;
        let v: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap_or_default();
        let s = &v["streams"][0];
        r.codec = s["codec_name"].as_str().map(str::to_owned);
        r.sample_rate = s["sample_rate"].as_str().and_then(|x| x.parse().ok());
        r.channels = s["channels"].as_u64().map(|c| c as u32);
        if r.codec.is_some() {
            let pcm = Command::new(fastvideo_media::tools::ffmpeg_bin())
                .args([
                    "-v", "error", "-nostdin", "-i", &f, "-map", "0:a:0", "-ac", "1", "-ar",
                    "48000", "-f", "s16le", "-",
                ])
                .output()
                .map_err(|e| format!("ffmpeg: {e}"))?;
            if !pcm.stdout.is_empty() {
                r.levels = Some(AudioLevels::from_s16le(&pcm.stdout, 48_000));
            }
        }
        r.audible = r.levels.as_ref().is_some_and(AudioLevels::audible);
        Ok(r)
    }

    /// The master (or a bare media playlist) → the audio playlist to follow
    /// and its first listing, or None while it lists no segment.
    async fn find(
        http: &reqwest::Client,
        base: &reqwest::Url,
    ) -> Result<Option<(AudioReadback, reqwest::Url, MediaPlaylist)>, String> {
        let master_text = text(http, base).await?;
        let mut r = AudioReadback {
            source: "muxed",
            group: None,
            name: None,
            language: None,
            variant: None,
            playlist_uri: String::new(),
            listed_after_ms: 0,
            first_sequence: 0,
            first_listed: 0,
            polls: 0,
            segments: 0,
            segment_seconds: 0.0,
            missed_segments: 0,
            codec: None,
            sample_rate: None,
            channels: None,
            levels: None,
            audible: false,
            master_text: master_text.clone(),
            media_text: String::new(),
        };
        let (uri, media_text) = if is_media_playlist(&master_text) {
            (base.as_str().to_owned(), master_text.clone())
        } else {
            let m = MasterPlaylist::parse(&master_text);
            let best = m
                .best_variant()
                .ok_or("no variant in the master playlist yet")?;
            let uri = match m.audio_for(best) {
                AudioSource::Rendition(a) => {
                    r.source = "rendition";
                    r.group = Some(a.group_id.clone());
                    r.name = a.name.clone();
                    r.language = a.language.clone();
                    r.variant = Some(best.info.clone());
                    a.uri.clone().unwrap_or_default()
                }
                AudioSource::Muxed => {
                    let v = m.lightest_variant().unwrap_or(best);
                    r.variant = Some(v.info.clone());
                    v.uri.clone()
                }
                AudioSource::MissingGroup(g) => {
                    return Err(format!("the variant names audio group {g:?} but the master lists no rendition of it yet"))
                }
            };
            let u = base.join(&uri).map_err(|e| e.to_string())?;
            (uri, text(http, &u).await?)
        };
        let media_url = base.join(&uri).map_err(|e| e.to_string())?;
        let media = MediaPlaylist::parse(&media_text);
        if media.segments.is_empty() {
            return Ok(None);
        }
        r.playlist_uri = uri;
        r.media_text = media_text;
        Ok(Some((r, media_url, media)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CF_MASTER: &str =
        include_str!("../../tests/fixtures/hls/cloudflare-master-audio-rendition.m3u8");
    const CF_AUDIO: &str = include_str!("../../tests/fixtures/hls/cloudflare-audio-media.m3u8");
    const MUXED_MASTER: &str = include_str!("../../tests/fixtures/hls/muxed-master.m3u8");
    const MUXED_NO_URI: &str =
        include_str!("../../tests/fixtures/hls/muxed-group-without-uri.m3u8");
    const MEDIA_TS: &str = include_str!("../../tests/fixtures/hls/muxed-media.m3u8");

    #[test]
    fn attributes_keep_quoted_commas() {
        let a = attributes(
            r#"RESOLUTION=1260x720,CODECS="avc1.64001f,mp4a.40.2",BANDWIDTH=6573313,AUDIO="group_audio""#,
        );
        assert_eq!(attr(&a, "CODECS"), Some("avc1.64001f,mp4a.40.2"));
        assert_eq!(attr(&a, "BANDWIDTH"), Some("6573313"));
        assert_eq!(attr(&a, "AUDIO"), Some("group_audio"));
        assert_eq!(attr(&a, "RESOLUTION"), Some("1260x720"));
        assert!(attributes("").is_empty());
    }

    #[test]
    fn cloudflare_master_has_a_separate_audio_rendition() {
        let m = MasterPlaylist::parse(CF_MASTER);
        assert!(!is_media_playlist(CF_MASTER));
        assert_eq!(m.variants.len(), 4);
        // Cloudflare does not list the variants by bandwidth (480p first here).
        assert_eq!(m.variants[0].resolution, Some((840, 480)));
        let best = m.best_variant().unwrap();
        assert_eq!(best.resolution, Some((1260, 720)));
        assert_eq!(best.audio_group.as_deref(), Some("group_audio"));
        assert_eq!(m.lightest_variant().unwrap().resolution, Some((420, 240)));
        match m.audio_for(best) {
            AudioSource::Rendition(r) => {
                assert_eq!(r.kind, "AUDIO");
                assert_eq!(r.group_id, "group_audio");
                assert!(r.default);
                let uri = r.uri.as_deref().unwrap();
                assert!(uri.contains(".m3u8"), "{uri}");
                assert!(
                    !m.variants.iter().any(|v| v.uri == uri),
                    "the audio playlist is not a video variant"
                );
            }
            other => panic!("want a rendition, got {other:?}"),
        }
        // Every variant shares the group.
        assert!(m
            .variants
            .iter()
            .all(|v| matches!(m.audio_for(v), AudioSource::Rendition(_))));
    }

    #[test]
    fn cloudflare_audio_media_playlist() {
        let p = MediaPlaylist::parse(CF_AUDIO);
        assert!(is_media_playlist(CF_AUDIO));
        // A live window (not the start of the stream): sequence 5 onwards,
        // 2 s fMP4 segments under the recording's audio path.
        assert_eq!(p.media_sequence, 5);
        assert_eq!(p.segments.len(), 8);
        assert_eq!(p.target_duration, Some(3.0));
        assert!((p.seconds() - 16.0).abs() < 1e-9);
        assert!(p
            .init
            .as_deref()
            .is_some_and(|i| i.starts_with("../../") && i.contains("/audio/128/init.mp4?")));
        assert!(p.segments[0].uri.contains("/audio/128/seg_5.mp4?"));
        assert_eq!(p.segments[0].sequence, p.media_sequence);
        let last = p.segments.last().unwrap();
        assert_eq!(
            last.sequence,
            p.media_sequence + p.segments.len() as u64 - 1
        );
        assert!(p
            .segments
            .iter()
            .all(|s| s.duration > 0.0 && !s.uri.starts_with('#')));
        assert!(!p.ended);
        assert!(p.seconds() > 0.0);
    }

    #[test]
    fn muxed_master_without_an_audio_group() {
        let m = MasterPlaylist::parse(MUXED_MASTER);
        assert_eq!(m.variants.len(), 2);
        assert!(m.media.is_empty());
        let best = m.best_variant().unwrap();
        assert_eq!(best.uri, "hi/index.m3u8");
        assert_eq!(m.audio_for(best), AudioSource::Muxed);
        assert_eq!(m.lightest_variant().unwrap().uri, "lo/index.m3u8");
    }

    #[test]
    fn a_group_rendition_without_uri_is_muxed() {
        let m = MasterPlaylist::parse(MUXED_NO_URI);
        let best = m.best_variant().unwrap();
        assert_eq!(best.audio_group.as_deref(), Some("aud"));
        assert_eq!(m.audio_for(best), AudioSource::Muxed);
    }

    #[test]
    fn a_group_not_listed_yet_is_missing() {
        let m = MasterPlaylist::parse(
            "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=10,AUDIO=\"group_audio\"\nv.m3u8\n",
        );
        assert_eq!(
            m.audio_for(&m.variants[0]),
            AudioSource::MissingGroup("group_audio".into())
        );
    }

    #[test]
    fn the_default_rendition_wins() {
        let m = MasterPlaylist::parse(concat!(
            "#EXTM3U\n",
            "#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"commentary\",DEFAULT=NO,AUTOSELECT=YES,URI=\"c.m3u8\"\n",
            "#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"main\",DEFAULT=YES,URI=\"m.m3u8\"\n",
            "#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"a\",NAME=\"subs\",DEFAULT=YES,URI=\"s.m3u8\"\n",
            "#EXT-X-STREAM-INF:BANDWIDTH=10,AUDIO=\"a\"\n",
            "v.m3u8\n",
        ));
        match m.audio_for(&m.variants[0]) {
            AudioSource::Rendition(r) => assert_eq!(r.uri.as_deref(), Some("m.m3u8")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn muxed_media_playlist_with_ts_segments() {
        assert!(is_media_playlist(MEDIA_TS));
        assert_eq!(MasterPlaylist::parse(MEDIA_TS).variants.len(), 0);
        let p = MediaPlaylist::parse(MEDIA_TS);
        assert_eq!(p.media_sequence, 41);
        assert_eq!(p.init, None);
        assert_eq!(p.segments.len(), 3);
        assert_eq!(
            p.segments[2],
            Segment {
                sequence: 43,
                duration: 3.0,
                uri: "seg43.ts".into()
            }
        );
        assert!((p.seconds() - 8.5).abs() < 1e-9);
        assert!(p.ended);
    }

    #[test]
    fn fmp4_init_map() {
        let p = MediaPlaylist::parse("#EXTM3U\n#EXT-X-TARGETDURATION:3\n#EXT-X-MAP:URI=\"init.mp4?x=1\"\n#EXTINF:3.000,\nseg0.mp4\n");
        assert_eq!(p.init.as_deref(), Some("init.mp4?x=1"));
        assert_eq!(p.target_duration, Some(3.0));
        assert_eq!(p.segments[0].sequence, 0);
    }

    #[test]
    fn levels_of_a_tone_and_of_silence() {
        let rate = 48_000;
        // 3 s of a 0.25 sine (the fake engine's clip audio), then 3 s of zeros.
        let mut s: Vec<f32> = (0..3 * rate)
            .map(|i| 0.25 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin())
            .collect();
        s.extend(std::iter::repeat_n(0.0, 3 * rate));
        let l = AudioLevels::from_samples(&s, rate as u32);
        assert_eq!(l.rms_dbfs_per_second.len(), 6);
        // RMS of a 0.25 sine: 0.25 / sqrt(2) -> -15.1 dBFS.
        assert!((l.rms_dbfs_per_second[0] - -15.1).abs() < 0.2, "{l:?}");
        assert_eq!(l.rms_dbfs_per_second[5], FLOOR_DBFS);
        assert_eq!(l.audible_seconds, 3);
        assert!((l.peak_dbfs - -12.0).abs() < 0.2);
        assert!(l.audible());

        let silent = AudioLevels::from_samples(&vec![0.0; 2 * rate], rate as u32);
        assert!(!silent.audible());
        assert_eq!(silent.peak_dbfs, FLOOR_DBFS);
        // AAC's "digital silence" (about -91 dB) is not audible either.
        let hiss = AudioLevels::from_samples(&vec![2.8e-5; 4 * rate], rate as u32);
        assert!(!hiss.audible(), "{hiss:?}");
        // One loud second is not enough.
        let mut blip = vec![0.0f32; 4 * rate];
        blip[..rate].iter_mut().for_each(|x| *x = 0.5);
        assert!(!AudioLevels::from_samples(&blip, rate as u32).audible());
        assert!(!AudioLevels::from_samples(&[], rate as u32).audible());
    }

    #[test]
    fn levels_from_s16le() {
        let pcm: Vec<u8> = (0..96_000)
            .flat_map(|i| if i % 2 == 0 { 16384i16 } else { -16384i16 }.to_le_bytes())
            .collect();
        let l = AudioLevels::from_s16le(&pcm, 48_000);
        assert_eq!(l.rms_dbfs_per_second, vec![-6.0, -6.0]);
        assert_eq!(l.peak_dbfs, -6.0);
        assert!(l.audible());
    }
}
