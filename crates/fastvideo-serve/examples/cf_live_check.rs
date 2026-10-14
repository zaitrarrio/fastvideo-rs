//! Stream outputs against the real Cloudflare Stream from a **running**
//! fv-serve (a GPU-served model), for the GPU checks (docs/serve/e2e/
//! h3-test-round-2026-10-14.md). The same measurements as the gated
//! `outputs_live` test's Cloudflare case, which runs the fake engine
//! in-process, but over HTTP against a server on this machine:
//!
//! ```text
//! cf_live_check streams --base http://127.0.0.1:8000 --model fasth3 \
//!   --cases whip-h264,whip-vp8,whip-vp9,rtmps,srt --seconds 60 --out /e2e/out/cf \
//!   [--clips 7] [--audio-seconds 60] [--audio-deadline 300]
//! cf_live_check whep --url <whep url> --codec h264 --seconds 40 --out /e2e/out/cf/copy
//! cf_live_check cleanup --uids <uid,uid,...> [--recordings-of <deleted input uid,...>]
//! cf_live_check list
//! ```
//!
//! `streams`: one `POST /fv/v1/streams` per case (output `cloudflare` with
//! `codec`, `cloudflare-rtmps`, `cloudflare-srt`); WHIP cases are read back
//! over WHEP with a str0m recvonly peer (video access units decoded by
//! ffprobe for codec / resolution, Opus packets counted), RTMPS / SRT over
//! HLS: the top variant's segments decoded (video), and the audio followed
//! through the master playlist (`fastvideo_serve::stream_outputs::hls`):
//! Cloudflare serves it as a separate rendition (`#EXT-X-MEDIA TYPE=AUDIO`,
//! the group the top variant names); a muxed playlist is read from the
//! variant's own segments. The audio playlist is read from the oldest
//! segment it lists for `--audio-seconds` (60) of segments, at most until
//! `--audio-deadline` (300) seconds after the POST, decoded and measured
//! (RMS per second, peak): the case passes only when the audio is present
//! and not silent (`AudioLevels::audible`: 2 s above -60 dBFS RMS, peak above
//! -40 dBFS, the WHEP check's bar). A clip stream plays each clip once and
//! then holds the last frame in silence, so the HLS cases queue `--clips` (7)
//! more clips after the prompt (a new clip every clip build, as the
//! 2026-10-14 round's pod C did) and run for up to `--audio-deadline` + 60 s.
//! Meanwhile `--ingest-seconds` (45) of the input's RTMPS playback (what
//! Cloudflare ingested, passed through) are read from `--ingest-after` (20) s.
//! SRT is judged on that ingest (audible) plus an AAC rendition in the HLS:
//! Cloudflare's HLS of an SRT input lags and skips most of the input
//! (docs/serve/stream-outputs.md "Cloudflare SRT HLS"); RTMPS on the HLS audio.
//! Time to ready is from the POST to the first decoded WHEP frame / the first
//! HLS segment that decodes. Each stream is then deleted and its live input
//! cleaned up as in `cleanup`.
//!
//! `cleanup`: for each uid this run created, GET the live input; only an
//! input whose name starts with `fv-test-` is touched (anything else is
//! reported and left alone): its recordings are deleted, then the input,
//! then a GET must answer 404.
//!
//! `list`: the account's live inputs that start with `fv-test-` (uid and
//! name) and how many other inputs exist (never their names).
//!
//! Env: `FV_KEY` (fv-serve API key), `FV_CF_ACCOUNT_ID`,
//! `FV_CF_STREAM_API_TOKEN` (never printed). Results: one `RESULT {json}`
//! line per case on stdout and `<out>/<case>.json` (HLS cases also keep the
//! first master / audio playlists read, `<case>-master.m3u8` and
//! `<case>-audio.m3u8`, and the joined audio segments `<case>-audio.seg`);
//! the exit status is 1 when a case failed.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use fastvideo_serve::stream_outputs::hls;
use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::Direction;
use fastvideo_webrtc::writer::{TrackKind, VideoCodec};
use serde_json::{json, Value};

const PREFIX: &str = "fv-test-";
const CF_API: &str = "https://api.cloudflare.com/client/v4";

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn codec_of(s: &str) -> VideoCodec {
    match s {
        "vp8" => VideoCodec::Vp8,
        "vp9" => VideoCodec::Vp9,
        _ => VideoCodec::H264,
    }
}

/// (codec, decoded frames, keyframes, width, height of the last frame).
fn decode(path: &str, extra: &[&str]) -> Option<(String, usize, usize, u64, u64)> {
    let out = Command::new(fastvideo_media::tools::ffprobe_bin())
        .args(["-v", "error"])
        .args(extra)
        .args(["-select_streams", "v:0", "-show_entries", "stream=codec_name:frame=width,height,key_frame", "-show_frames", "-of", "json", path])
        .output()
        .ok()?;
    let v: Value = serde_json::from_slice(&out.stdout).ok()?;
    let codec = v["streams"][0]["codec_name"].as_str()?.to_owned();
    let frames = v["frames"].as_array()?;
    let keys = frames.iter().filter(|f| f["key_frame"] == 1).count();
    let last = frames.last()?;
    Some((codec, frames.len(), keys, last["width"].as_u64()?, last["height"].as_u64()?))
}

/// The first audio stream of a media file: codec, sample rate, channels and
/// the mean / max volume (ffmpeg volumedetect), or null without audio.
fn audio_probe(path: &str) -> Value {
    let out = Command::new(fastvideo_media::tools::ffprobe_bin())
        .args(["-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name,sample_rate,channels", "-of", "json", path])
        .output();
    let Ok(out) = out else { return Value::Null };
    let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    let Some(s) = v["streams"].as_array().and_then(|a| a.first()).cloned() else { return Value::Null };
    let vol = Command::new(fastvideo_media::tools::ffmpeg_bin())
        .args(["-v", "info", "-nostdin", "-i", path, "-vn", "-af", "volumedetect", "-f", "null", "-"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stderr).into_owned())
        .unwrap_or_default();
    let grab = |k: &str| {
        vol.lines()
            .find_map(|l| l.split(k).nth(1))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|x| x.parse::<f64>().ok())
    };
    json!({"codec": s["codec_name"], "sample_rate": s["sample_rate"], "channels": s["channels"],
           "mean_volume_db": grab("mean_volume:"), "max_volume_db": grab("max_volume:")})
}

/// Mean luma and per-frame deviation of the decoded video (64x36 gray);
/// (frames, mean, deviation, black frames).
fn luma(path: &str, extra: &[&str]) -> Option<(usize, f64, f64, usize)> {
    let out = Command::new(fastvideo_media::tools::ffmpeg_bin())
        .args(["-v", "error", "-nostdin"])
        .args(extra)
        .args(["-i", path, "-an", "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "-"])
        .output()
        .ok()?;
    let px = 64 * 36;
    let n = out.stdout.len() / px;
    if n == 0 {
        return None;
    }
    let (mut ms, mut ds, mut black) = (0.0, 0.0, 0);
    for f in out.stdout.chunks_exact(px) {
        let m = f.iter().map(|&b| f64::from(b)).sum::<f64>() / px as f64;
        let d = (f.iter().map(|&b| (f64::from(b) - m).powi(2)).sum::<f64>() / px as f64).sqrt();
        ms += m;
        ds += d;
        if m < 20.0 && d < 4.0 {
            black += 1;
        }
    }
    Some((n, ms / n as f64, ds / n as f64, black))
}

fn es_file(codec: VideoCodec, frames: &[bytes::Bytes]) -> Vec<u8> {
    let mut file = Vec::new();
    match codec {
        VideoCodec::H264 => frames.iter().for_each(|f| file.extend_from_slice(f)),
        _ => {
            file.extend_from_slice(b"DKIF\0\0\x20\0");
            file.extend_from_slice(if codec == VideoCodec::Vp9 { b"VP90" } else { b"VP80" });
            file.extend_from_slice(&[0u8; 4]);
            file.extend_from_slice(&90_000u32.to_le_bytes());
            file.extend_from_slice(&1u32.to_le_bytes());
            file.extend_from_slice(&(frames.len() as u32).to_le_bytes());
            file.extend_from_slice(&[0u8; 4]);
            for (i, f) in frames.iter().enumerate() {
                file.extend_from_slice(&(f.len() as u32).to_le_bytes());
                file.extend_from_slice(&(i as u64).to_le_bytes());
                file.extend_from_slice(f);
            }
        }
    }
    file
}

struct Viewer {
    frames: Vec<bytes::Bytes>,
    audio_packets: usize,
    audio_bytes: usize,
    /// From the WHEP POST to the first video frame.
    first_ms: Option<u128>,
}

async fn whep_view(whep_url: &str, codec: VideoCodec, secs: u64) -> Result<Viewer, String> {
    let host = RtcHost::bind(HostConfig::default()).await.map_err(|e| e.to_string())?;
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
            channels: vec![],
            srflx: host.gather_srflx(&[fastvideo_webrtc::ice::IceServer::default_stun()], Duration::from_secs(2)).await,
            h264_level: fastvideo_webrtc::profile::H264Level::L4_0,
            video_codecs: vec![codec],
        })
        .await
        .map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let resp = reqwest::Client::new()
        .post(whep_url)
        .header("content-type", "application/sdp")
        .body(offer)
        .send()
        .await
        .map_err(|e| e.without_url().to_string())?;
    let status = resp.status();
    let answer = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("WHEP HTTP {status}: {}", answer.chars().take(160).collect::<String>()));
    }
    let mut peer = pending.accept_answer(&answer).await.map_err(|e| e.to_string())?;
    let mut v = Viewer { frames: Vec::new(), audio_packets: 0, audio_bytes: 0, first_ms: None };
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        match tokio::time::timeout(Duration::from_millis(500), peer.next_event()).await {
            Ok(Some(PeerEvent::Media { kind: TrackKind::Video, data, .. })) => {
                v.first_ms.get_or_insert(t0.elapsed().as_millis());
                v.frames.push(data);
            }
            Ok(Some(PeerEvent::Media { kind: TrackKind::Audio, data, .. })) => {
                v.audio_packets += 1;
                v.audio_bytes += data.len();
            }
            Ok(Some(PeerEvent::Closed(r))) => {
                if v.frames.is_empty() {
                    return Err(format!("WHEP peer closed: {r:?}"));
                }
                break;
            }
            Ok(None) => break,
            _ => {}
        }
    }
    peer.close();
    host.shutdown().await;
    Ok(v)
}

/// WHEP read-back of one URL, retried every 2 s until video arrives or
/// `deadline` passes; then decoded. Returns the result object.
async fn whep_case(whep: &str, codec: VideoCodec, t0: Instant, deadline: Duration, watch_s: u64, out: &Path, name: &str) -> Value {
    let mut last_err = String::from("never tried");
    while t0.elapsed() < deadline {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let t_try = t0.elapsed();
        match whep_view(whep, codec, watch_s).await {
            Ok(w) if !w.frames.is_empty() => {
                let file = out.join(format!("{name}.{}", if codec == VideoCodec::H264 { "h264" } else { "ivf" }));
                std::fs::write(&file, es_file(codec, &w.frames)).ok();
                let fmt: &[&str] = if codec == VideoCodec::H264 { &["-f", "h264"] } else { &["-f", "ivf"] };
                let f = file.to_string_lossy().into_owned();
                let d = decode(&f, fmt);
                let l = luma(&f, fmt);
                let ready = t_try.as_millis() + w.first_ms.unwrap_or(0);
                return json!({
                    "transport": "whep", "ok": d.as_ref().is_some_and(|d| d.1 > 0),
                    "time_to_ready_ms": ready, "first_frame_after_whep_post_ms": w.first_ms,
                    "received_frames": w.frames.len(),
                    "decoded": d.map(|d| json!({"codec": d.0, "frames": d.1, "keyframes": d.2, "width": d.3, "height": d.4})),
                    "luma": l.map(|l| json!({"frames": l.0, "mean": l.1, "deviation": l.2, "black": l.3})),
                    "audio": {"present": w.audio_packets > 0, "opus_packets": w.audio_packets,
                              "mean_packet_bytes": if w.audio_packets > 0 { w.audio_bytes as f64 / w.audio_packets as f64 } else { 0.0 }},
                    "watched_s": watch_s,
                });
            }
            Ok(_) => last_err = "WHEP connected but no video frames".into(),
            Err(e) => last_err = e,
        }
    }
    json!({"transport": "whep", "ok": false, "error": last_err})
}

/// The top HLS variant's last `n` segments (with the fMP4 init segment) in
/// one file; the master's variant lines.
async fn hls_grab(http: &reqwest::Client, master: &str, n: usize, out: &Path) -> Result<Vec<String>, String> {
    let get = |u: reqwest::Url| async move {
        let r = http.get(u.clone()).send().await.map_err(|e| e.without_url().to_string())?;
        if !r.status().is_success() {
            return Err(format!("HTTP {}", r.status()));
        }
        r.bytes().await.map_err(|e| e.without_url().to_string())
    };
    let base = reqwest::Url::parse(master).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&get(base.clone()).await?).into_owned();
    let mut vars: Vec<(u64, String, String)> = Vec::new();
    let mut lines = text.lines();
    while let Some(l) = lines.next() {
        if l.starts_with("#EXT-X-STREAM-INF") {
            let bw = l.split("BANDWIDTH=").nth(1).and_then(|r| r.split(',').next()).and_then(|b| b.parse().ok()).unwrap_or(0);
            if let Some(u) = lines.next() {
                vars.push((bw, l.to_owned(), u.trim().to_owned()));
            }
        }
    }
    vars.sort_by_key(|v| std::cmp::Reverse(v.0));
    let (_, _, uri) = vars.first().cloned().ok_or("no variant in the master playlist yet")?;
    let media_url = base.join(&uri).map_err(|e| e.to_string())?;
    let media = String::from_utf8_lossy(&get(media_url.clone()).await?).into_owned();
    let mut init = None;
    let mut segs = Vec::new();
    for l in media.lines() {
        if let Some(m) = l.strip_prefix("#EXT-X-MAP:") {
            init = m.split("URI=\"").nth(1).and_then(|r| r.split('"').next()).map(str::to_owned);
        } else if !l.starts_with('#') && !l.trim().is_empty() {
            segs.push(l.trim().to_owned());
        }
    }
    if segs.is_empty() {
        return Err("the media playlist has no segment yet".into());
    }
    let mut file = Vec::new();
    if let Some(i) = &init {
        file.extend_from_slice(&get(media_url.join(i).map_err(|e| e.to_string())?).await?);
    }
    let take = segs.len().min(n);
    for s in &segs[segs.len() - take..] {
        file.extend_from_slice(&get(media_url.join(s).map_err(|e| e.to_string())?).await?);
    }
    std::fs::write(out, &file).map_err(|e| e.to_string())?;
    Ok(vars.into_iter().map(|v| v.1).collect())
}

struct Cf {
    http: reqwest::Client,
    api: String,
    token: String,
}

impl Cf {
    fn from_env() -> Self {
        Self {
            http: reqwest::Client::new(),
            api: format!("{CF_API}/accounts/{}/stream", env("FV_CF_ACCOUNT_ID")),
            token: env("FV_CF_STREAM_API_TOKEN"),
        }
    }

    async fn get(&self, path: &str) -> (u16, Value) {
        match self.http.get(format!("{}{path}", self.api)).bearer_auth(&self.token).send().await {
            Ok(r) => {
                let st = r.status().as_u16();
                (st, r.json().await.unwrap_or(Value::Null))
            }
            Err(e) => (0, json!({"error": e.without_url().to_string()})),
        }
    }

    async fn delete(&self, path: &str) -> u16 {
        match self.http.delete(format!("{}{path}", self.api)).bearer_auth(&self.token).send().await {
            Ok(r) => r.status().as_u16(),
            Err(_) => 0,
        }
    }

    async fn input_state(&self, uid: &str) -> Option<String> {
        let (st, v) = self.get(&format!("/live_inputs/{uid}")).await;
        (st == 200).then(|| v["result"]["status"]["current"]["state"].as_str().unwrap_or("").to_owned())
    }

    /// Deletes one input this run created (only an `fv-test-` input) and its
    /// recordings; then checks it answers 404.
    async fn cleanup(&self, uid: &str) -> Value {
        let (st, v) = self.get(&format!("/live_inputs/{uid}")).await;
        if st == 404 {
            return json!({"uid": uid, "before": 404, "gone": true, "note": "already deleted (on_stop)"});
        }
        if st != 200 {
            return json!({"uid": uid, "before": st, "gone": false, "note": "GET failed"});
        }
        let name = v["result"]["meta"]["name"].as_str().unwrap_or("").to_owned();
        if !name.starts_with(PREFIX) {
            return json!({"uid": uid, "before": 200, "gone": false, "note": "not an fv-test- input: left alone"});
        }
        let enabled = v["result"]["enabled"].clone();
        let (_, vids) = self.get(&format!("/live_inputs/{uid}/videos")).await;
        let mut recs = Vec::new();
        for r in vids["result"].as_array().into_iter().flatten() {
            if let Some(vid) = r["uid"].as_str() {
                let t = Instant::now();
                let status = loop {
                    let s = self.delete(&format!("/{vid}")).await;
                    if s != 409 || t.elapsed() > Duration::from_secs(180) {
                        break s;
                    }
                    tokio::time::sleep(Duration::from_secs(5)).await;
                };
                recs.push(json!({"video": vid, "duration_s": r["duration"], "delete": status}));
            }
        }
        // A recording that could not be deleted keeps its input, so a later
        // `cleanup` finds it again (deleting the input would orphan it).
        if recs.iter().any(|r| !matches!(r["delete"].as_u64(), Some(200 | 404))) {
            return json!({"uid": uid, "name": name, "before": 200, "recordings": recs, "gone": false,
                          "note": "a recording was not deleted: input kept, run cleanup again"});
        }
        let del = self.delete(&format!("/live_inputs/{uid}")).await;
        tokio::time::sleep(Duration::from_secs(2)).await;
        let (after, _) = self.get(&format!("/live_inputs/{uid}")).await;
        json!({"uid": uid, "name": name, "before": 200, "enabled_before": enabled, "recordings": recs,
               "delete": del, "after": after, "gone": after == 404})
    }

    /// Deletes the recordings of the given (already deleted) inputs this
    /// run created: only videos whose `liveInput` is one of `inputs`; then
    /// checks each answers 404.
    async fn cleanup_recordings(&self, inputs: &[&str]) -> Value {
        let (_, vids) = self.get("").await;
        let mut out = Vec::new();
        for v in vids["result"].as_array().into_iter().flatten() {
            let (Some(vid), Some(li)) = (v["uid"].as_str(), v["liveInput"].as_str()) else { continue };
            if !inputs.contains(&li) {
                continue;
            }
            let del = self.delete(&format!("/{vid}")).await;
            tokio::time::sleep(Duration::from_secs(2)).await;
            let (after, _) = self.get(&format!("/{vid}")).await;
            out.push(json!({"video": vid, "live_input": li, "delete": del, "after": after, "gone": after == 404}));
        }
        json!(out)
    }

    async fn list(&self) -> Value {
        let (st, v) = self.get("/live_inputs").await;
        let all = v["result"].as_array().cloned().unwrap_or_default();
        let ours: Vec<Value> = all
            .iter()
            .filter(|i| i["meta"]["name"].as_str().is_some_and(|n| n.starts_with(PREFIX)))
            .map(|i| json!({"uid": i["uid"], "name": i["meta"]["name"], "created": i["created"]}))
            .collect();
        // Recordings whose live input is gone (an fv-test- input deleted
        // before its recording was): uid, input, created; nothing else.
        let live: Vec<&str> = all.iter().filter_map(|i| i["uid"].as_str()).collect();
        let (_, vids) = self.get("").await;
        let orphans: Vec<Value> = vids["result"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|v| v["liveInput"].as_str().is_some_and(|l| !l.is_empty() && !live.contains(&l)))
            .map(|v| json!({"uid": v["uid"], "live_input": v["liveInput"], "created": v["created"]}))
            .collect();
        json!({"http": st, "fv_test_inputs": ours, "other_inputs": all.len() - ours.len(),
               "recordings_without_input": orphans, "videos": vids["result"].as_array().map_or(0, Vec::len),
               "preexisting_divine_firefly_present": all.iter().any(|i| i["meta"]["name"] == "divine-firefly-3665")})
    }
}

async fn streams(args: &[String]) {
    let base = arg(args, "--base").unwrap_or_else(|| "http://127.0.0.1:8000".into());
    let model = arg(args, "--model").expect("--model");
    let prompt = arg(args, "--prompt").unwrap_or_else(|| {
        "A lighthouse keeper climbs the spiral stairs at dusk and lights the great lamp; waves crash below, gulls cry, \
         wind howls against the glass."
            .into()
    });
    let secs: u64 = arg(args, "--seconds").and_then(|s| s.parse().ok()).unwrap_or(60);
    let cases = arg(args, "--cases").unwrap_or_else(|| "whip-h264,whip-vp8,whip-vp9,rtmps,srt".into());
    let clips: usize = arg(args, "--clips").and_then(|s| s.parse().ok()).unwrap_or(7);
    let clip_prompt = arg(args, "--clip-prompt").unwrap_or_else(|| {
        "Storm waves crash against the lighthouse rocks at night, thunder rolls, rain lashes the windows, the lamp beam \
         sweeps the sea."
            .into()
    });
    let audio_window: u64 = arg(args, "--audio-seconds").and_then(|s| s.parse().ok()).unwrap_or(60);
    let audio_deadline: u64 = arg(args, "--audio-deadline").and_then(|s| s.parse().ok()).unwrap_or(300);
    let ingest_secs: u64 = arg(args, "--ingest-seconds").and_then(|s| s.parse().ok()).unwrap_or(45);
    let ingest_after: u64 = arg(args, "--ingest-after").and_then(|s| s.parse().ok()).unwrap_or(20);
    let mut failed = Vec::new();
    let out = PathBuf::from(arg(args, "--out").unwrap_or_else(|| "cf-out".into()));
    std::fs::create_dir_all(&out).unwrap();
    let key = env("FV_KEY");
    let cf = Cf::from_env();
    let http = reqwest::Client::new();
    let mut created = Vec::new();
    for name in cases.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (output, codec) = match name {
            "whip-h264" => (json!({"kind": "cloudflare"}), Some(VideoCodec::H264)),
            "whip-vp8" => (json!({"kind": "cloudflare", "codec": "vp8"}), Some(VideoCodec::Vp8)),
            "whip-vp9" => (json!({"kind": "cloudflare", "codec": "vp9"}), Some(VideoCodec::Vp9)),
            "rtmps" => (json!({"kind": "cloudflare-rtmps"}), None),
            "srt" => (json!({"kind": "cloudflare-srt"}), None),
            other => panic!("unknown case {other}"),
        };
        let body = if codec.is_some() {
            json!({"model": model, "prompt": prompt, "output": output, "max_seconds": secs + 60})
        } else {
            // HLS: keep clips (and their audio) coming while the audio is read.
            let more: Vec<Value> = (0..clips).map(|_| json!({"prompt": clip_prompt})).collect();
            json!({"model": model, "prompt": prompt, "clips": more, "output": output, "max_seconds": secs.max(audio_deadline) + 60})
        };
        let t0 = Instant::now();
        let r = http.post(format!("{base}/fv/v1/streams")).bearer_auth(&key).json(&body).send().await;
        let (st, v) = match r {
            Ok(r) => {
                let st = r.status().as_u16();
                (st, r.json::<Value>().await.unwrap_or(Value::Null))
            }
            Err(e) => (0, json!({"error": e.without_url().to_string()})),
        };
        let create_ms = t0.elapsed().as_millis();
        if st != 201 {
            let res = json!({"case": name, "ok": false, "create_http": st, "error": v["error"].clone()});
            failed.push(name.to_owned());
            println!("RESULT {res}");
            std::fs::write(out.join(format!("{name}.json")), serde_json::to_vec_pretty(&res).unwrap()).ok();
            continue;
        }
        let id = v["id"].as_str().unwrap_or_default().to_owned();
        let pb = v["playback"].clone();
        let uid = pb["live_input_id"].as_str().unwrap_or_default().to_owned();
        if !uid.is_empty() {
            created.push(uid.clone());
        }
        println!("{name}: stream {id} created in {create_ms} ms, live input {uid}");
        let read = match codec {
            Some(c) => {
                let whep = pb["whep_url"].as_str().unwrap_or_default().to_owned();
                whep_case(&whep, c, t0, Duration::from_secs(secs + 120), 40, &out, name).await
            }
            None => {
                let hls = pb["hls_url"].as_str().unwrap_or_default().to_owned();
                let file = out.join(format!("{name}.seg"));
                let mut connected: Option<f64> = None;
                // What Cloudflare ingested, apart from its HLS: `--ingest-seconds`
                // (45) of the input's RTMPS playback (a pass-through) from
                // `--ingest-after` (20) s after the create, while clips play (an
                // h3-turbo clip plays ~26-31 s apart, for 5 s).
                let mut ingest_task = Some({
                    let (uid, out, name) = (uid.clone(), out.clone(), name.to_owned());
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_secs(ingest_after)).await;
                        let (_, li) = Cf::from_env().get(&format!("/live_inputs/{uid}")).await;
                        let pb = &li["result"]["rtmpsPlayback"];
                        match (pb["url"].as_str(), pb["streamKey"].as_str()) {
                            (Some(u), Some(k)) => {
                                let f = out.join(format!("{name}-ingest.flv"));
                                serde_json::to_value(hls::read_rtmps_playback(u, k, ingest_secs, &f).await).unwrap_or(Value::Null)
                            }
                            _ => json!({"audible": false, "error": "the live input lists no rtmpsPlayback"}),
                        }
                    })
                });
                let mut ready: Option<f64> = None;
                let mut last = String::new();
                while t0.elapsed() < Duration::from_secs(secs.max(150)) {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    if connected.is_none() && cf.input_state(&uid).await.as_deref() == Some("connected") {
                        connected = Some(t0.elapsed().as_secs_f64());
                    }
                    match hls_grab(&http, &hls, 1, &file).await {
                        Ok(_) if decode(&file.to_string_lossy(), &[]).is_some_and(|d| d.1 > 0) => {
                            ready = Some(t0.elapsed().as_secs_f64());
                            break;
                        }
                        Ok(_) => last = "a segment that does not decode yet".into(),
                        Err(e) => last = e,
                    }
                }
                match ready {
                    None => json!({"transport": "hls", "ok": false, "error": last, "connected_after_s": connected}),
                    Some(rd) => {
                        tokio::time::sleep(Duration::from_secs(12)).await;
                        let vars = hls_grab(&http, &hls, 3, &file).await;
                        let f = file.to_string_lossy().into_owned();
                        let d = decode(&f, &[]);
                        let video_ok = d.as_ref().is_some_and(|d| d.1 > 0);
                        // The audio, through the master playlist (a separate rendition on Cloudflare).
                        let deadline = t0 + Duration::from_secs(audio_deadline);
                        let afile = out.join(format!("{name}-audio.seg"));
                        let ingest = async { ingest_task.take().expect("one HLS read per case").await.unwrap_or(Value::Null) };
                        let hls_audio = async {
                            match hls::read_audio(&http, &hls, Duration::from_secs(audio_window), deadline, &afile).await {
                                Ok(a) => {
                                    std::fs::write(out.join(format!("{name}-master.m3u8")), &a.master_text).ok();
                                    std::fs::write(out.join(format!("{name}-audio.m3u8")), &a.media_text).ok();
                                    let mut j = serde_json::to_value(&a).unwrap_or(Value::Null);
                                    j["read_after_post_s"] = json!(t0.elapsed().as_secs_f64());
                                    j
                                }
                                Err(e) => json!({"audible": false, "error": e}),
                            }
                        };
                        let (audio, ingest) = tokio::join!(hls_audio, ingest);
                        // RTMPS: the HLS audio rendition must be audible. SRT: Cloudflare's SRT
                        // HLS lags and skips most of the input (docs/serve/stream-outputs.md
                        // "Cloudflare SRT HLS"), so there the rendition must be listed and decode
                        // as AAC, and the audio must be audible in what Cloudflare ingested.
                        let audio_ok = if name == "srt" {
                            audio["codec"] == "aac" && ingest["audible"] == true
                        } else {
                            audio["audible"] == true
                        };
                        json!({
                            "transport": "hls", "ok": video_ok && audio_ok, "video_ok": video_ok, "audio_ok": audio_ok,
                            "time_to_ready_ms": (rd * 1000.0) as u64, "connected_after_s": connected,
                            "variants": vars.unwrap_or_default(),
                            "decoded": d.map(|d| json!({"codec": d.0, "frames": d.1, "keyframes": d.2, "width": d.3, "height": d.4})),
                            "luma": luma(&f, &[]).map(|l| json!({"frames": l.0, "mean": l.1, "deviation": l.2, "black": l.3})),
                            "audio": audio,
                            "ingest_audio": ingest,
                            "video_segment_audio": audio_probe(&f),
                        })
                    }
                }
            }
        };
        if t0.elapsed() < Duration::from_secs(secs) {
            tokio::time::sleep(Duration::from_secs(secs) - t0.elapsed()).await;
        }
        let stv: Value = match http.get(format!("{base}/fv/v1/streams/{id}")).bearer_auth(&key).send().await {
            Ok(r) => r.json().await.unwrap_or(Value::Null),
            Err(_) => Value::Null,
        };
        let del = http.delete(format!("{base}/fv/v1/streams/{id}")).bearer_auth(&key).send().await.map(|r| r.status().as_u16()).unwrap_or(0);
        tokio::time::sleep(Duration::from_secs(3)).await;
        let clean = if uid.is_empty() { Value::Null } else { cf.cleanup(&uid).await };
        let s = &stv["status"];
        if read["ok"] != true {
            failed.push(name.to_owned());
        }
        let res = json!({
            "case": name, "model": model, "create_ms": create_ms, "stream_id": id, "live_input": uid,
            "ok": read["ok"], "read": read,
            "server": {"output_state": s["output_state"], "output_error": s["output_error"], "encoder": s["encoder"],
                       "video_codec": s["video_codec"], "width": s["width"], "height": s["height"], "fps": s["fps"],
                       "frames_sent": s["frames_sent"], "keyframes_sent": s["keyframes_sent"],
                       "audio_packets_sent": s["audio_packets_sent"], "ticks_dropped": s["ticks_dropped"],
                       "ttff": stv["ttff"].clone(), "end_reason": stv["end_reason"].clone()},
            "stream_delete_http": del, "cleanup": clean,
        });
        println!("RESULT {res}");
        std::fs::write(out.join(format!("{name}.json")), serde_json::to_vec_pretty(&res).unwrap()).ok();
    }
    println!("CREATED {}", created.join(","));
    println!("LIST {}", cf.list().await);
    if !failed.is_empty() {
        println!("FAILED {}", failed.join(","));
        std::process::exit(1);
    }
}

/// The live edge of both HLS playlists now: the best video variant's and
/// the audio rendition's media sequence and listed seconds.
async fn edges(http: &reqwest::Client, master: &str) -> Value {
    async fn get(http: &reqwest::Client, u: reqwest::Url) -> Option<String> {
        let r = http.get(u).send().await.ok()?;
        if !r.status().is_success() {
            return None;
        }
        r.text().await.ok()
    }
    fn edge(p: Option<hls::MediaPlaylist>) -> Value {
        match p {
            Some(p) => json!({"first_seq": p.media_sequence, "last_seq": p.segments.last().map(|s| s.sequence),
                              "listed": p.segments.len(), "listed_s": p.seconds()}),
            None => Value::Null,
        }
    }
    let Ok(base) = reqwest::Url::parse(master) else { return Value::Null };
    let Some(text) = get(http, base.clone()).await else { return json!({"master": "not served"}) };
    let m = hls::MasterPlaylist::parse(&text);
    let Some(best) = m.best_variant() else { return json!({"master": "no variant"}) };
    let video = match base.join(&best.uri) {
        Ok(u) => get(http, u).await.map(|t| hls::MediaPlaylist::parse(&t)),
        Err(_) => None,
    };
    let audio = match m.audio_for(best) {
        hls::AudioSource::Rendition(r) => match base.join(r.uri.as_deref().unwrap_or_default()) {
            Ok(u) => get(http, u).await.map(|t| hls::MediaPlaylist::parse(&t)),
            Err(_) => None,
        },
        _ => None,
    };
    json!({"video": edge(video), "audio": edge(audio)})
}

/// One synthetic tick: a moving 1344x768 pattern and 1/24 s of the
/// time-coded tone (0.25, stereo): 200 + 100 * floor(t / 10) Hz at stream
/// time t, so a decoded second says which part of the stream it is.
fn synthetic_tick(i: u64, w: u32, h: u32, per: usize) -> (fastvideo_media::RgbFrame, Vec<f32>) {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for (j, px) in rgb.chunks_exact_mut(3).enumerate() {
        let (x, y) = ((j as u64) % u64::from(w), (j as u64) / u64::from(w));
        px[0] = ((x + i * 4) % 256) as u8;
        px[1] = ((y + i * 2) % 256) as u8;
        px[2] = (((x + y) / 2 + i) % 256) as u8;
    }
    let frame = fastvideo_media::av::frame(w, h, rgb, i).expect("frame size");
    let audio = (0..per)
        .flat_map(|k| {
            let n = i as usize * per + k;
            let t = n as f64 / 48_000.0;
            let v = (0.25 * (2.0 * std::f64::consts::PI * (200.0 + 100.0 * (t / 10.0).floor()) * t).sin()) as f32;
            [v, v]
        })
        .collect();
    (frame, audio)
}

/// The stream time each decoded second of `file` carries, from the
/// time-coded tone (`synthetic_tick`, the lavfi `aevalsrc`): the frequency
/// by zero crossings, 200 + 100 * k Hz -> 10 * k s. Null for a silent second.
fn tone_clock(file: &Path) -> Vec<Option<u64>> {
    let pcm = Command::new(fastvideo_media::tools::ffmpeg_bin())
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(file)
        .args(["-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "s16le", "-"])
        .output()
        .map(|o| o.stdout)
        .unwrap_or_default();
    let s: Vec<i16> = pcm.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
    s.chunks(48_000)
        .filter(|c| c.len() >= 24_000)
        .map(|c| {
            let peak = c.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
            if peak < 300 {
                return None;
            }
            let crossings = c.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count() as f64;
            let f = crossings / 2.0 * 48_000.0 / c.len() as f64;
            Some((((f - 200.0) / 100.0).round().max(0.0) as u64) * 10)
        })
        .collect()
}

/// `srt://host:port` with the host replaced by its first IPv4 address.
async fn pin_host(u: &str) -> String {
    let Ok(mut url) = reqwest::Url::parse(u) else { return u.to_owned() };
    let (Some(host), Some(port)) = (url.host_str().map(str::to_owned), url.port()) else { return u.to_owned() };
    let ip = tokio::net::lookup_host((host.as_str(), port)).await.ok().and_then(|mut a| a.find(|a| a.is_ipv4()));
    match ip {
        Some(a) if url.set_ip_host(a.ip()).is_ok() => url.to_string().trim_end_matches('/').to_owned(),
        _ => u.to_owned(),
    }
}

/// `push`: Cloudflare's ingest without fv-serve. Creates one
/// `fv-test-push-…` input (recording, so HLS exists), publishes a synthetic
/// stream to it for `--seconds` (150) and reads the HLS audio back as
/// `streams` does, sampling the live edge of both playlists every 10 s.
/// `--kind srt|rtmps`; `--source sink` (fastvideo's ffmpeg sink with the
/// arguments fv-serve uses, libx264 test encoder, fed frames and a tone in
/// real time; SRT only) or `lavfi` (a plain ffmpeg: `testsrc2` + `sine`,
/// x264 CBR, 2 s closed GOP, AAC-LC 128k); `--latency-ms` (SRT, 1000);
/// `--video-kbps` (lavfi, 6000). The URL, stream id, passphrase and key
/// never print (ffmpeg's stderr is scrubbed).
async fn push(args: &[String]) {
    use fastvideo_serve::stream_outputs::cloudflare::{CfSettings, CfStream};
    use fastvideo_serve::stream_outputs::{OnStop, Recording};
    use std::sync::atomic::{AtomicBool, Ordering};
    let kind = arg(args, "--kind").unwrap_or_else(|| "srt".into());
    let source = arg(args, "--source").unwrap_or_else(|| "lavfi".into());
    let secs: u64 = arg(args, "--seconds").and_then(|s| s.parse().ok()).unwrap_or(150);
    let latency: u32 = arg(args, "--latency-ms").and_then(|s| s.parse().ok()).unwrap_or(1000);
    let kbps: u32 = arg(args, "--video-kbps").and_then(|s| s.parse().ok()).unwrap_or(6000);
    let audio_window: u64 = arg(args, "--audio-seconds").and_then(|s| s.parse().ok()).unwrap_or(60);
    let out = PathBuf::from(arg(args, "--out").unwrap_or_else(|| "cf-out".into()));
    std::fs::create_dir_all(&out).unwrap();
    if source == "sink" && kind != "srt" {
        eprintln!("--source sink is SRT only (RTMPS goes through fv-serve's TLS relay)");
        return;
    }
    let cfs = CfStream::new(CfSettings {
        account_id: env("FV_CF_ACCOUNT_ID"),
        api_token: env("FV_CF_STREAM_API_TOKEN"),
        api_base: CF_API.into(),
        recording: Recording::Automatic,
        on_stop: OnStop::Auto,
        delete_recording_after_days: None,
        customer_domain: None,
        timeout: Duration::from_secs(30),
    })
    .expect("cloudflare client");
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let name = format!("{PREFIX}push-{kind}-{source}-{stamp}");
    let li = match cfs.create(&name, true).await {
        Ok(li) => li,
        Err(e) => {
            println!("RESULT {}", json!({"ok": false, "error": e.message}));
            return;
        }
    };
    println!("CREATED {} ({name})", li.uid);
    let hls_url = li.playback(None, false)["hls_url"].as_str().unwrap_or_default().to_owned();
    let url = match kind.as_str() {
        "rtmps" => format!("{}{}", li.rtmps_url.clone().unwrap_or_default(), li.rtmps_key.clone().unwrap_or_default()),
        _ => fastvideo_media::sink::srt_url(
            // A fully static ffmpeg (FV_FFMPEG) cannot resolve host names:
            // the address is pinned as fv-serve's SRT output does.
            &pin_host(li.srt_url.as_deref().unwrap_or_default()).await,
            li.srt_stream_id.as_deref(),
            li.srt_passphrase.as_deref(),
            latency,
        ),
    };
    // `--srt-extra maxbw=…&…`: more libsrt options on the URL.
    let url = match (kind.as_str(), arg(args, "--srt-extra")) {
        ("srt", Some(x)) => format!("{url}&{x}"),
        _ => url,
    };
    let mut secrets: Vec<String> = [&li.srt_stream_id, &li.srt_passphrase, &li.rtmps_key].into_iter().flatten().cloned().collect();
    secrets.push(url.clone());
    let scrub = |t: &str| secrets.iter().filter(|x| !x.is_empty()).fold(t.to_owned(), |t, x| t.replace(x.as_str(), "<redacted>"));
    let t0 = Instant::now();
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let producer: std::thread::JoinHandle<String> = if source == "sink" {
        use fastvideo_media::sink::{FfmpegSink, SinkAudioIn, SinkConfig, SinkTarget};
        let stop = stop.clone();
        let target = SinkTarget::Srt { url: url.clone() };
        std::thread::spawn(move || {
            let (w, h, fps) = (1344u32, 768u32, 24u32);
            let mut cfg = SinkConfig::new(target, w, h, fps, Some(SinkAudioIn { rate: 48_000, channels: 2 }));
            cfg.encoder = fastvideo_media::video::FfmpegH264::Libx264CpuTest;
            let mut sink = match FfmpegSink::start(cfg) {
                Ok(s) => s,
                Err(e) => return format!("sink start: {e}"),
            };
            let start = Instant::now();
            let mut i: u64 = 0;
            let mut errors = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let due = start + Duration::from_secs_f64(i as f64 / f64::from(fps));
                if let Some(d) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(d);
                }
                let (frame, audio) = synthetic_tick(i, w, h, 48_000 / fps as usize);
                let _ = sink.send(&frame, Some(&audio));
                if let Some(e) = sink.exit_error() {
                    errors.push(e);
                }
                i += 1;
            }
            let st = sink.stats();
            let fin = sink.finish(Duration::from_secs(5)).err().map(|e| e.to_string());
            format!(
                "sink: {} frames, {} audio ticks, dropped {} / {}, restarts {}, ran {:.1}s, exits {errors:?}, finish {fin:?}",
                st.frames_sent,
                st.audio_ticks_sent,
                st.dropped_video,
                st.dropped_audio,
                st.restarts,
                start.elapsed().as_secs_f64()
            )
        })
    } else {
        let stop = stop.clone();
        let fmt = if kind == "rtmps" { "flv" } else { "mpegts" };
        let rate = format!("{kbps}k");
        let mut cmd = Command::new(fastvideo_media::tools::ffmpeg_bin());
        cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin", "-re", "-f", "lavfi", "-i", "testsrc2=size=1344x768:rate=24"])
            .args(["-re", "-f", "lavfi", "-i", "aevalsrc=0.25*sin(2*PI*(200+100*floor(t/10))*t):s=48000:c=stereo", "-map", "0:v", "-map", "1:a"])
            .args(["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p"])
            .args(["-b:v", &rate, "-maxrate", &rate, "-minrate", &rate, "-bufsize", &rate])
            .args(["-g", "48", "-keyint_min", "48", "-sc_threshold", "0", "-bf", "0", "-flags", "+cgop", "-x264-params", "nal-hrd=cbr"])
            .args(["-c:a", "aac", "-profile:a", "aac_low", "-b:a", "128k", "-ar", "48000", "-ac", "2"])
            .args(["-t", &secs.to_string(), "-f", fmt, &url])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());
        std::thread::spawn(move || {
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => return format!("ffmpeg: {e}"),
            };
            let start = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            let ran = start.elapsed().as_secs_f64();
            let _ = child.kill();
            let o = child.wait_with_output().map(|o| String::from_utf8_lossy(&o.stderr).into_owned()).unwrap_or_default();
            format!("ffmpeg ran {ran:.1}s; stderr: {}", o.lines().take(8).collect::<Vec<_>>().join(" | "))
        })
    };
    let http = reqwest::Client::new();
    let (hls2, http2, stop2) = (hls_url.clone(), http.clone(), stop.clone());
    let sampler = tokio::spawn(async move {
        let mut v = Vec::new();
        while !stop2.load(Ordering::Relaxed) {
            v.push(json!({"t_s": t0.elapsed().as_secs(), "edge": edges(&http2, &hls2).await}));
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
        v
    });
    // What Cloudflare ingested, apart from its HLS: 30 s of the input's
    // RTMPS playback (a TCP pass-through, no transcode) from 40 s in.
    let playback = {
        let (cfs, uid, out) = (cfs.clone(), li.uid.clone(), out.clone());
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(40)).await;
            let v = cfs.get(&uid).await.unwrap_or(Value::Null);
            let (Some(u), Some(k)) = (v["rtmpsPlayback"]["url"].as_str(), v["rtmpsPlayback"]["streamKey"].as_str()) else {
                return json!({"error": "no rtmpsPlayback in the live input"});
            };
            let file = out.join("rtmps-playback.flv");
            let mut j = serde_json::to_value(hls::read_rtmps_playback(u, k, 30, &file).await).unwrap_or(Value::Null);
            j["stream_time_per_second"] = json!(tone_clock(&file));
            j
        })
    };
    let afile = out.join(format!("push-{kind}-{source}-audio.seg"));
    let audio = match hls::read_audio(&http, &hls_url, Duration::from_secs(audio_window), t0 + Duration::from_secs(secs), &afile).await {
        Ok(a) => {
            std::fs::write(out.join(format!("push-{kind}-{source}-audio.m3u8")), &a.media_text).ok();
            let mut j = serde_json::to_value(&a).unwrap_or(Value::Null);
            j["stream_time_per_second"] = json!(tone_clock(&afile));
            j
        }
        Err(e) => json!({"audible": false, "error": e}),
    };
    if t0.elapsed() < Duration::from_secs(secs) {
        tokio::time::sleep(Duration::from_secs(secs) - t0.elapsed()).await;
    }
    let state = cfs.get(&li.uid).await.map(|v| v["status"].clone()).unwrap_or(Value::Null);
    stop.store(true, Ordering::Relaxed);
    let edges_seen = sampler.await.unwrap_or_default();
    let prod = tokio::task::spawn_blocking(move || producer.join().unwrap_or_default()).await.unwrap_or_default();
    tokio::time::sleep(Duration::from_secs(3)).await;
    let clean = Cf::from_env().cleanup(&li.uid).await;
    let res = json!({"kind": kind, "source": source, "seconds": secs, "latency_ms": latency, "video_kbps": kbps, "srt_extra": arg(args, "--srt-extra"), "ffmpeg": fastvideo_media::tools::ffmpeg_bin(),
                     "audio": audio, "rtmps_playback": playback.await.unwrap_or(Value::Null), "edges": edges_seen, "input_status": state, "producer": scrub(&prod), "cleanup": clean});
    println!("RESULT {res}");
    std::fs::write(out.join(format!("push-{kind}-{source}.json")), serde_json::to_vec_pretty(&res).unwrap()).ok();
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("streams") => streams(&args).await,
        Some("whep") => {
            let url = arg(&args, "--url").expect("--url");
            let codec = codec_of(&arg(&args, "--codec").unwrap_or_default());
            let secs: u64 = arg(&args, "--seconds").and_then(|s| s.parse().ok()).unwrap_or(40);
            let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "cf-out".into()));
            std::fs::create_dir_all(&out).unwrap();
            let r = whep_case(&url, codec, Instant::now(), Duration::from_secs(secs + 60), secs, &out, "whep").await;
            println!("RESULT {r}");
            std::fs::write(out.join("whep.json"), serde_json::to_vec_pretty(&r).unwrap()).ok();
        }
        Some("cleanup") => {
            let cf = Cf::from_env();
            for uid in arg(&args, "--uids").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
                println!("CLEANUP {}", cf.cleanup(uid).await);
            }
            // `--recordings-of <uid,...>`: recordings left by inputs this run
            // created and already deleted.
            let of = arg(&args, "--recordings-of").unwrap_or_default();
            let of: Vec<&str> = of.split(',').filter(|s| !s.is_empty()).collect();
            if !of.is_empty() {
                println!("RECORDINGS {}", cf.cleanup_recordings(&of).await);
            }
            println!("LIST {}", cf.list().await);
        }
        Some("list") => println!("LIST {}", Cf::from_env().list().await),
        Some("push") => push(&args).await,
        _ => eprintln!("usage: cf_live_check streams|whep|cleanup|list (see the source header)"),
    }
}
