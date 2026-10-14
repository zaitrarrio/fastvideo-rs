//! Stream outputs against real destinations, on the fake engine (gated; both
//! skip without their variable). docs/serve/stream-outputs.md "Testing".
//!
//! - `FV_TEST_OUTPUTS_MEDIAMTX=1`: a local MediaMTX (`FV_TEST_MEDIAMTX_BIN`,
//!   else v1.15.1 downloaded into the target dir) takes one stream per kind
//!   from fv-serve: WHIP (H.264, VP8, VP9), RTMP, RTMPS (a self-signed
//!   certificate from the `openssl` CLI) and SRT (with a passphrase the
//!   path requires). MediaMTX's API must show each path ready with a video
//!   track, and ffprobe reads the resolution back over RTSP. Kinds the
//!   local ffmpeg cannot publish (no `rtmps` / `srt` protocol) are reported
//!   and skipped. `ffmpeg -protocols` is printed.
//! - `FV_TEST_OUTPUTS_CLOUDFLARE=1` with `FV_CF_ACCOUNT_ID` and
//!   `FV_CF_STREAM_API_TOKEN`: one stream per kind (WHIP H.264 at 1344x768,
//!   WHIP VP8, RTMPS, SRT) against Cloudflare Stream for
//!   `FV_TEST_CF_SECONDS` (60), live inputs named `fv-test-…`. WHEP
//!   playback is checked with a str0m viewer that decodes what Cloudflare
//!   delivers (resolution recorded), HLS by ffprobe on the manifest; every
//!   input is deleted and checked gone.
//!
//! - `FV_TEST_OUTPUTS_MEDIAMTX=1` also runs `mediamtx_director_copy_carries_audio`:
//!   a director session (fake H3, audio on) with `x_fv_output` over WHIP and
//!   over RTMP; MediaMTX must list an audio track beside H.264 (Opus /
//!   MPEG-4 Audio) and ffprobe must find it over RTSP. Run it without
//!   `encoders` too (the serve images' feature set): `--features fake,http-client`.
//!
//!   cargo test -p fastvideo-serve --features full,fake --test outputs_live -- --nocapture --test-threads=1

#![cfg(all(feature = "webrtc", feature = "http-client"))]

use std::collections::BTreeMap;
use std::path::{Path as FsPath, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::Direction;
use fastvideo_webrtc::writer::{TrackKind, VideoCodec};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-outputs-live";
const MTX_VERSION: &str = "v1.15.1";
const SRT_PASS: &str = "fv-test-passphrase-123";

fn on(var: &str) -> bool {
    std::env::var(var).is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes"))
}

async fn app_with(extra: &[(&str, String)]) -> App {
    app_cfg(extra, |_| {}).await
}

async fn app_cfg(extra: &[(&str, String)], f: impl FnOnce(&mut Config)) -> App {
    let dir = tempfile::Builder::new().prefix("fv-serve-outputs-live-").tempdir().unwrap().keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("FV_STREAM_ENCODER".to_owned(), "x264-test".to_owned());
    for (k, v) in extra {
        env.insert((*k).to_owned(), v.clone());
    }
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    f(&mut c);
    c.validate().unwrap();
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut b = Request::builder().method(method).uri(uri).header("authorization", format!("Bearer {KEY}"));
    let req = match body {
        Some(v) => {
            b = b.header("content-type", "application/json");
            b.body(Body::from(v.to_string())).unwrap()
        }
        None => b.body(Body::empty()).unwrap(),
    };
    let r = app.clone().oneshot(req).await.unwrap();
    let s = r.status();
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 24).await.unwrap();
    (s, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn ffmpeg_protocols() -> String {
    Command::new(fastvideo_media::tools::ffmpeg_bin())
        .args(["-hide_banner", "-protocols"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

fn has_protocol(list: &str, p: &str) -> bool {
    list.lines().any(|l| l.trim() == p)
}

/// Decodes `secs` of `url`'s video with ffprobe: (codec, decoded frames,
/// keyframes, width, height of the last frame).
fn decode(url: &str, extra: &[&str], secs: Option<u32>) -> Option<(String, usize, usize, u64, u64)> {
    let mut cmd = Command::new(fastvideo_media::tools::ffprobe_bin());
    cmd.args(["-v", "error"]).args(extra).args(["-select_streams", "v:0"]);
    if let Some(s) = secs {
        cmd.args(["-read_intervals", &format!("%+{s}")]);
    }
    let out = cmd
        .args(["-show_entries", "stream=codec_name:frame=width,height,key_frame", "-show_frames", "-of", "json", url])
        .output()
        .ok()?;
    let v: Value = serde_json::from_slice(&out.stdout).ok()?;
    if v["frames"].as_array().is_none_or(|f| f.is_empty()) {
        let err = String::from_utf8_lossy(&out.stderr);
        println!("decode {url}: no frames; ffprobe: {}", err.lines().take(6).collect::<Vec<_>>().join(" | "));
    }
    let codec = v["streams"][0]["codec_name"].as_str()?.to_owned();
    let frames = v["frames"].as_array()?;
    let keys = frames.iter().filter(|f| f["key_frame"] == 1).count();
    let last = frames.last()?;
    Some((codec, frames.len(), keys, last["width"].as_u64()?, last["height"].as_u64()?))
}

/// ffprobe's first video stream of `url`: (codec, width, height).
fn probe(url: &str, extra: &[&str]) -> Option<(String, u64, u64)> {
    let out = Command::new(fastvideo_media::tools::ffprobe_bin())
        .args(["-v", "error"])
        .args(extra)
        .args(["-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "json", url])
        .output()
        .ok()?;
    let v: Value = serde_json::from_slice(&out.stdout).ok()?;
    let s = &v["streams"][0];
    Some((s["codec_name"].as_str()?.to_owned(), s["width"].as_u64()?, s["height"].as_u64()?))
}

// ---- MediaMTX --------------------------------------------------------------------

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

async fn mediamtx_bin(work: &FsPath) -> PathBuf {
    if let Ok(p) = std::env::var("FV_TEST_MEDIAMTX_BIN") {
        return PathBuf::from(p);
    }
    let bin = work.join("mediamtx");
    if bin.exists() {
        return bin;
    }
    let url = format!("https://github.com/bluenviron/mediamtx/releases/download/{MTX_VERSION}/mediamtx_{MTX_VERSION}_linux_amd64.tar.gz");
    let bytes = reqwest::get(&url).await.unwrap().error_for_status().unwrap().bytes().await.unwrap();
    let tgz = work.join("mediamtx.tar.gz");
    std::fs::write(&tgz, &bytes).unwrap();
    let ok = Command::new("tar").arg("xzf").arg(&tgz).arg("-C").arg(work).arg("mediamtx").status().unwrap().success();
    assert!(ok, "extracting MediaMTX");
    bin
}

struct Mtx {
    child: Child,
    api: u16,
    rtsp: u16,
    rtmp: u16,
    rtmps: u16,
    srt: u16,
    webrtc: u16,
    log: PathBuf,
}

impl Drop for Mtx {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn start_mediamtx() -> Mtx {
    let work = std::env::var("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir()).join("fv-mediamtx");
    std::fs::create_dir_all(&work).unwrap();
    let bin = mediamtx_bin(&work).await;
    // RTMPS: a self-signed certificate.
    let (key, crt) = (work.join("server.key"), work.join("server.crt"));
    let tls = key.exists()
        || Command::new("openssl")
            .args(["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2", "-subj", "/CN=127.0.0.1"])
            .arg("-keyout")
            .arg(&key)
            .arg("-out")
            .arg(&crt)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
    let cfg = work.join("mediamtx.yml");
    // Settings from MTX_* env; every path accepts publishers.
    std::fs::write(&cfg, "paths:\n  all_others:\n").unwrap();
    let (api, rtsp, rtmp, rtmps, srt, webrtc, udp) = (free_port(), free_port(), free_port(), free_port(), free_port(), free_port(), free_port());
    let log = work.join(format!("mediamtx-{api}.log"));
    let mut cmd = Command::new(&bin);
    cmd.arg(&cfg)
        .env("MTX_API", "yes")
        .env("MTX_APIADDRESS", format!("127.0.0.1:{api}"))
        .env("MTX_RTSPADDRESS", format!("127.0.0.1:{rtsp}"))
        .env("MTX_RTSPTRANSPORTS", "tcp")
        .env("MTX_RTMPADDRESS", format!("127.0.0.1:{rtmp}"))
        .env("MTX_RTMPENCRYPTION", if tls { "optional" } else { "no" })
        .env("MTX_RTMPSADDRESS", format!("127.0.0.1:{rtmps}"))
        .env("MTX_RTMPSERVERKEY", &key)
        .env("MTX_RTMPSERVERCERT", &crt)
        .env("MTX_HLS", "no")
        .env("MTX_SRTADDRESS", format!("127.0.0.1:{srt}"))
        .env("MTX_WEBRTCADDRESS", format!("127.0.0.1:{webrtc}"))
        .env("MTX_WEBRTCLOCALUDPADDRESS", format!("127.0.0.1:{udp}"))
        .env("MTX_WEBRTCADDITIONALHOSTS", "127.0.0.1")
        .env("MTX_PATHDEFAULTS_SRTPUBLISHPASSPHRASE", SRT_PASS)
        .stdout(std::fs::File::create(&log).unwrap())
        .stderr(Stdio::inherit());
    let child = cmd.spawn().expect("starting MediaMTX");
    let m = Mtx { child, api, rtsp, rtmp, rtmps: if tls { rtmps } else { 0 }, srt, webrtc, log };
    let t0 = Instant::now();
    while reqwest::get(format!("http://127.0.0.1:{api}/v3/paths/list")).await.is_err() {
        assert!(t0.elapsed() < Duration::from_secs(15), "MediaMTX API did not come up");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    m
}

/// Waits until MediaMTX shows `path` ready with a video track; `Err` with
/// the last API answer.
async fn mtx_ready(api: u16, path: &str) -> Result<Value, Value> {
    let t0 = Instant::now();
    let mut last = Value::Null;
    loop {
        if let Ok(r) = reqwest::get(format!("http://127.0.0.1:{api}/v3/paths/get/{path}")).await {
            if let Ok(v) = r.json::<Value>().await {
                let tracks = v["tracks"].to_string();
                let video = ["H264", "VP8", "VP9"].iter().any(|c| tracks.contains(c));
                if v["ready"] == true && video && v["bytesReceived"].as_u64().unwrap_or(0) > 50_000 {
                    return Ok(v);
                }
                last = v;
            }
        }
        if t0.elapsed() > Duration::from_secs(45) {
            return Err(last);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn tail(p: &FsPath, n: usize) -> String {
    let s = std::fs::read_to_string(p).unwrap_or_default();
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mediamtx_whip_rtmp_rtmps_srt() {
    if !on("FV_TEST_OUTPUTS_MEDIAMTX") {
        eprintln!("skipping: FV_TEST_OUTPUTS_MEDIAMTX not set");
        return;
    }
    std::env::set_var("FV_STREAM_STUN", "none");
    let protocols = ffmpeg_protocols();
    println!("ffmpeg -protocols: {}", protocols.split_whitespace().collect::<Vec<_>>().join(" "));
    let m = start_mediamtx().await;
    let a = app_with(&[("FV_OUTPUTS_ALLOW_PRIVATE", "1".into())]).await;
    let mut kinds: Vec<(&str, Value)> = vec![
        ("whip-h264", json!({"kind": "whip", "url": format!("http://127.0.0.1:{}/fvwhip/whip", m.webrtc)})),
        ("rtmp", json!({"kind": "rtmps", "url": format!("rtmp://127.0.0.1:{}/live", m.rtmp), "stream_key": "fvrtmp"})),
    ];
    if fastvideo_media::vp8::libvpx_available() {
        kinds.push(("whip-vp8", json!({"kind": "whip", "url": format!("http://127.0.0.1:{}/fvvp8/whip", m.webrtc), "codec": "vp8"})));
    }
    if fastvideo_media::vp8::libvpx_vp9_available() {
        kinds.push(("whip-vp9", json!({"kind": "whip", "url": format!("http://127.0.0.1:{}/fvvp9/whip", m.webrtc), "codec": "vp9"})));
    }
    if m.rtmps != 0 && has_protocol(&protocols, "rtmps") {
        kinds.push(("rtmps", json!({"kind": "rtmps", "url": format!("rtmps://127.0.0.1:{}/live", m.rtmps), "stream_key": "fvrtmps"})));
    } else {
        println!("RESULT rtmps: skipped (ffmpeg rtmps: {}, openssl cert: {})", has_protocol(&protocols, "rtmps"), m.rtmps != 0);
    }
    if has_protocol(&protocols, "srt") || has_protocol(&protocols, "libsrt") {
        kinds.push(("srt", json!({"kind": "srt", "url": format!("srt://127.0.0.1:{}", m.srt), "stream_id": "publish:fvsrt", "passphrase": SRT_PASS})));
    } else {
        println!("RESULT srt: skipped (this ffmpeg has no libsrt)");
    }
    for (name, output) in kinds {
        let path = match name {
            "whip-h264" => "fvwhip",
            "whip-vp8" => "fvvp8",
            "whip-vp9" => "fvvp9",
            "rtmp" => "live/fvrtmp",
            "rtmps" => "live/fvrtmps",
            _ => "fvsrt",
        };
        // fake-h3-turbo: H3's native 1344x768 with audio.
        let body = json!({"model": "fake-h3-turbo", "prompt": "a lighthouse", "output": output});
        let (s, v) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
        assert_eq!(s, StatusCode::CREATED, "{name}: {v}");
        let id = v["id"].as_str().unwrap().to_owned();
        let t0 = Instant::now();
        let info = match mtx_ready(m.api, path).await {
            Ok(v) => v,
            Err(last) => {
                let (_, st) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
                panic!("{name}: MediaMTX path {path} never became ready: {last}\nstream: {}\nMediaMTX log:\n{}", st["status"], tail(&m.log, 30));
            }
        };
        let ready_s = t0.elapsed().as_secs_f64();
        let rtsp = format!("rtsp://127.0.0.1:{}/{path}", m.rtsp);
        let mut probed = decode(&rtsp, &["-rtsp_transport", "tcp"], Some(4));
        if let Some(codec) = match name {
            "whip-h264" => Some(VideoCodec::H264),
            "whip-vp8" => Some(VideoCodec::Vp8),
            "whip-vp9" => Some(VideoCodec::Vp9),
            _ => None,
        } {
            // WebRTC read-back too: MediaMTX (pion) → a str0m viewer → ffmpeg.
            let file = std::env::temp_dir().join(format!("fv-whep-{name}.bin"));
            let (n, d) = whep_decode(&format!("http://127.0.0.1:{}/{path}/whep", m.webrtc), codec, 4, &file).await;
            println!("RESULT {name}: WHEP read-back from MediaMTX: {n} frames received, decoded (codec, frames, keyframes, w, h) {d:?}");
            if probed.is_none() {
                // ffmpeg's RTSP VP9 depacketizer: rely on the WHEP read-back.
                probed = d;
            }
        }
        let (_, st) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
        println!(
            "RESULT {name}: MediaMTX ready after {ready_s:.1}s, tracks {}, bytes {}, decoded over RTSP (codec, frames, keyframes, w, h) {probed:?}, fv-serve sent {} frames ({} {}), output_error {}",
            info["tracks"], info["bytesReceived"], st["status"]["frames_sent"], st["status"]["video_codec"], st["status"]["encoder"], st["status"]["output_error"]
        );
        // fake-h3-turbo has audio: WHIP carries Opus, RTMP/SRT AAC.
        let audio_track = if name.starts_with("whip") { "Opus" } else { "MPEG-4 Audio" };
        assert!(info["tracks"].to_string().contains(audio_track), "{name}: no {audio_track} track: {}", info["tracks"]);
        let (codec, frames, _, w, h) = probed.unwrap_or_else(|| panic!("{name}: ffprobe could not decode {rtsp}"));
        assert!(frames >= 24, "{name}: {frames} frames decoded in 4 s");
        assert_eq!((w, h), (1344, 768), "{name}: native resolution");
        assert!(["h264", "vp8", "vp9"].contains(&codec.as_str()), "{name}: {codec}");
        let (s, _) = call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
        assert_eq!(s, StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    println!("MediaMTX log: {}", m.log.display());
}

/// ffprobe's audio streams of `url` (codec names; empty: none).
#[cfg(feature = "fal")]
fn audio_codecs(url: &str, extra: &[&str]) -> Vec<String> {
    let Ok(out) = Command::new(fastvideo_media::tools::ffprobe_bin())
        .args(["-v", "error"])
        .args(extra)
        .args(["-select_streams", "a", "-show_entries", "stream=codec_name,sample_rate,channels", "-of", "json", url])
        .output()
    else {
        return Vec::new();
    };
    let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(Value::Null);
    println!("ffprobe audio {url}: {}", v["streams"]);
    v["streams"].as_array().map(|a| a.iter().filter_map(|s| s["codec_name"].as_str().map(str::to_owned)).collect()).unwrap_or_default()
}

/// The next control message of `type` on a director peer (others skipped).
#[cfg(feature = "fal")]
async fn next_msg(peer: &mut fastvideo_webrtc::host::Peer, ty: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(90), async {
        loop {
            match peer.next_event().await {
                Some(PeerEvent::Message(m)) => {
                    let v: Value = serde_json::from_str(m.as_text().unwrap()).unwrap();
                    if v["type"] == ty {
                        return v;
                    }
                    if v["type"] == "output_error" || v["type"] == "error" {
                        println!("director: {v}");
                    }
                }
                Some(PeerEvent::Closed(r)) => panic!("closed waiting for {ty}: {r:?}"),
                Some(_) => {}
                None => panic!("peer gone waiting for {ty}"),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("no {ty}"))
}

/// Director output copies carry the session's audio (gpu-check-2026-10-13
/// bug 3: the WHIP copy reached MediaMTX as `1 track (H264)`): a fake H3
/// director session (audio on) copies to MediaMTX over WHIP (Opus) and RTMP
/// (AAC); MediaMTX lists both tracks and ffprobe reads the audio back.
#[cfg(feature = "fal")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mediamtx_director_copy_carries_audio() {
    if !on("FV_TEST_OUTPUTS_MEDIAMTX") {
        eprintln!("skipping: FV_TEST_OUTPUTS_MEDIAMTX not set");
        return;
    }
    std::env::set_var("FV_STREAM_STUN", "none");
    let m = start_mediamtx().await;
    let a = app_cfg(&[("FV_OUTPUTS_ALLOW_PRIVATE", "1".into())], |c| c.webrtc.public_ip = "127.0.0.1".into()).await;
    let copies = [
        ("whip", "fvdirector", json!({"kind": "whip", "url": format!("http://127.0.0.1:{}/fvdirector/whip", m.webrtc), "target": "mediamtx"}), "Opus", "opus"),
        ("rtmp", "live/fvdirrtmp", json!({"kind": "rtmps", "url": format!("rtmp://127.0.0.1:{}/live", m.rtmp), "stream_key": "fvdirrtmp"}), "MPEG-4 Audio", "aac"),
    ];
    for (name, path, out, mtx_track, codec) in copies {
        let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
        let (pending, offer) = host
            .offer(OfferOptions {
                video: Some(Direction::RecvOnly),
                audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
                channels: vec!["control".into()],
                video_codecs: vec![VideoCodec::H264, VideoCodec::Vp8],
                ..Default::default()
            })
            .await
            .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/wma/session")
            .header("authorization", format!("Key {KEY}"))
            .header("content-type", "application/json")
            .body(Body::from(json!({"app_id": "minimax/h3-max/director", "sdp": offer, "type": "offer"}).to_string()))
            .unwrap();
        let r = a.router.clone().oneshot(req).await.unwrap();
        assert_eq!(r.status(), StatusCode::OK, "{name}: /wma/session");
        let v: Value = serde_json::from_slice(&axum::body::to_bytes(r.into_body(), 1 << 24).await.unwrap()).unwrap();
        let mut peer = pending.accept_answer(v["sdp"].as_str().unwrap()).await.unwrap();
        next_msg(&mut peer, "session_info").await;
        let ctl = |m: Value| fastvideo_webrtc::channel::ChannelMessage::text("control", m.to_string());
        peer.send_message(ctl(json!({"type": "configure", "prompt_version": 1, "prompt": "a lighthouse", "x_fv_output": out}))).await.unwrap();
        let c = next_msg(&mut peer, "configured").await;
        assert!(c["x_fv_output"].is_object(), "{name}: {c}");
        next_msg(&mut peer, "output_ready").await;
        let info = match mtx_ready(m.api, path).await {
            Ok(v) => v,
            Err(last) => panic!("{name}: MediaMTX path {path} never became ready: {last}\nMediaMTX log:\n{}", tail(&m.log, 30)),
        };
        println!("RESULT director copy {name}: MediaMTX tracks {}, bytes {}", info["tracks"], info["bytesReceived"]);
        let tracks: Vec<String> = info["tracks"].as_array().unwrap().iter().filter_map(|t| t.as_str().map(str::to_owned)).collect();
        assert!(tracks.iter().any(|t| t == "H264"), "{name}: tracks {tracks:?}");
        assert!(tracks.iter().any(|t| t == mtx_track), "{name}: the copy has no audio track ({tracks:?}); MediaMTX log:\n{}", tail(&m.log, 20));
        let rtsp = format!("rtsp://127.0.0.1:{}/{path}", m.rtsp);
        let audio = audio_codecs(&rtsp, &["-rtsp_transport", "tcp"]);
        assert!(audio.iter().any(|a| a == codec), "{name}: ffprobe audio streams {audio:?}, want {codec}");
        peer.send_message(ctl(json!({"type": "stop"}))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        peer.close();
        host.shutdown().await;
    }
}

/// Plays `whep_url` for `secs`, writes what arrived to a file (IVF for
/// VP8/VP9, Annex-B for H.264) and decodes it with ffprobe: (frames
/// received, (codec, decoded frames, keyframes, w, h)).
async fn whep_decode(whep_url: &str, codec: VideoCodec, secs: u64, out: &FsPath) -> (usize, Option<(String, usize, usize, u64, u64)>) {
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
            channels: vec![],
            srflx: vec![],
            h264_level: fastvideo_webrtc::profile::H264Level::L4_0,
            video_codecs: vec![codec],
        })
        .await
        .unwrap();
    let resp = reqwest::Client::new().post(whep_url).header("content-type", "application/sdp").body(offer).send().await.unwrap();
    assert!(resp.status().is_success(), "WHEP {whep_url}: {}", resp.status());
    let answer = resp.text().await.unwrap();
    let mut peer = pending.accept_answer(&answer).await.unwrap();
    let mut frames: Vec<bytes::Bytes> = Vec::new();
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        if let Ok(Some(PeerEvent::Media { kind: TrackKind::Video, data, .. })) =
            tokio::time::timeout(Duration::from_millis(300), peer.next_event()).await
        {
            frames.push(data);
        }
    }
    peer.close();
    host.shutdown().await;
    let mut file = Vec::new();
    match codec {
        VideoCodec::H264 => frames.iter().for_each(|f| file.extend_from_slice(f)),
        _ => {
            file.extend_from_slice(b"DKIF\0\0\x20\0");
            file.extend_from_slice(if codec == VideoCodec::Vp9 { b"VP90" } else { b"VP80" });
            // width, height (unknown: 0), rate 90000 / scale 1, frames, unused.
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
    std::fs::write(out, &file).unwrap();
    let fmt: &[&str] = if codec == VideoCodec::H264 { &["-f", "h264"] } else { &["-f", "ivf"] };
    (frames.len(), decode(out.to_str().unwrap(), fmt, None))
}

// ---- Cloudflare (live) ---------------------------------------------------------------

struct Viewer {
    frames: usize,
    keyframes: usize,
    size: Option<(u32, u32)>,
    audio: usize,
    first_ms: Option<u128>,
}

/// Plays `whep_url` for `secs` with a str0m recvonly peer; returns what
/// arrived (resolution from the first keyframe's SPS / VP8 header).
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
        return Err(format!("WHEP HTTP {status}: {}", answer.chars().take(200).collect::<String>()));
    }
    let mut peer = pending.accept_answer(&answer).await.map_err(|e| e.to_string())?;
    let mut v = Viewer { frames: 0, keyframes: 0, size: None, audio: 0, first_ms: None };
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        match tokio::time::timeout(Duration::from_millis(500), peer.next_event()).await {
            Ok(Some(PeerEvent::Media { kind: TrackKind::Video, data, .. })) => {
                v.frames += 1;
                v.first_ms.get_or_insert(t0.elapsed().as_millis());
                let size = match codec {
                    VideoCodec::H264 => fastvideo_media::h264::find_sps(&data).map(|s| (s.width, s.height)),
                    _ => fastvideo_media::decode::vp8_keyframe_size(&data),
                };
                if let Some(s) = size {
                    v.keyframes += 1;
                    v.size = Some(s);
                }
            }
            Ok(Some(PeerEvent::Media { kind: TrackKind::Audio, .. })) => v.audio += 1,
            Ok(Some(PeerEvent::Closed(r))) => return Err(format!("WHEP peer closed: {r:?}")),
            Ok(None) => break,
            _ => {}
        }
    }
    peer.close();
    host.shutdown().await;
    Ok(v)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cloudflare_live_each_kind() {
    if !on("FV_TEST_OUTPUTS_CLOUDFLARE") {
        eprintln!("skipping: FV_TEST_OUTPUTS_CLOUDFLARE not set");
        return;
    }
    let account = std::env::var("FV_CF_ACCOUNT_ID").expect("FV_CF_ACCOUNT_ID");
    let token = std::env::var("FV_CF_STREAM_API_TOKEN").expect("FV_CF_STREAM_API_TOKEN");
    let secs: u64 = std::env::var("FV_TEST_CF_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let a = app_with(&[
        ("FV_CF_ACCOUNT_ID", account.clone()),
        ("FV_CF_STREAM_API_TOKEN", token.clone()),
        ("FV_CF_STREAM_NAME_PREFIX", "fv-test-".into()),
    ])
    .await;
    let http = reqwest::Client::new();
    let api = format!("https://api.cloudflare.com/client/v4/accounts/{account}/stream/live_inputs");
    let mut uids = Vec::new();
    for (name, output, codec) in [
        ("whip-h264", json!({"kind": "cloudflare"}), Some(VideoCodec::H264)),
        ("whip-vp8", json!({"kind": "cloudflare", "codec": "vp8"}), Some(VideoCodec::Vp8)),
        ("rtmps", json!({"kind": "cloudflare-rtmps"}), None),
        ("srt", json!({"kind": "cloudflare-srt"}), None),
    ] {
        let body = json!({"model": "fake-h3-turbo", "prompt": "a lighthouse at dusk", "output": output});
        let (s, v) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
        assert_eq!(s, StatusCode::CREATED, "{name}: {v}");
        let id = v["id"].as_str().unwrap().to_owned();
        let pb = v["playback"].clone();
        uids.push(pb["live_input_id"].as_str().unwrap().to_owned());
        println!("{name}: playback {pb}");
        let t0 = Instant::now();
        let result = match codec {
            Some(c) => {
                // Let the publisher connect, then watch over WHEP.
                tokio::time::sleep(Duration::from_secs(5)).await;
                match whep_view(pb["whep_url"].as_str().unwrap(), c, secs.saturating_sub(10).max(10)).await {
                    Ok(w) => format!(
                        "WHEP: {} frames, {} keyframes, delivered {:?}, {} audio packets, first frame after {:?} ms",
                        w.frames, w.keyframes, w.size, w.audio, w.first_ms
                    ),
                    Err(e) => format!("WHEP failed: {e}"),
                }
            }
            None => {
                // HLS appears a few segments after the broadcast starts.
                let hls = pb["hls_url"].as_str().unwrap().to_owned();
                let mut got = None;
                while t0.elapsed() < Duration::from_secs(secs) {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    if let Some(p) = probe(&hls, &[]) {
                        got = Some(p);
                        break;
                    }
                }
                format!("HLS ffprobe: {got:?} after {:.0}s", t0.elapsed().as_secs_f64())
            }
        };
        if t0.elapsed() < Duration::from_secs(secs) {
            tokio::time::sleep(Duration::from_secs(secs) - t0.elapsed()).await;
        }
        let (_, st) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
        println!(
            "RESULT {name}: {result}; sent {} frames ({} {}x{}), output_error {}",
            st["status"]["frames_sent"], st["status"]["video_codec"], st["status"]["width"], st["status"]["height"], st["status"]["output_error"]
        );
        call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
    }
    // Every fv-test- input is gone (or disabled when it recorded).
    tokio::time::sleep(Duration::from_secs(3)).await;
    for uid in uids {
        let r = http.get(format!("{api}/{uid}")).bearer_auth(&token).send().await.unwrap();
        let st = r.status();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        let recorded = v["result"]["recording"]["mode"] == "automatic";
        println!("cleanup {uid}: HTTP {st}, enabled {}, recording {}", v["result"]["enabled"], v["result"]["recording"]["mode"]);
        if recorded {
            // `on_stop = auto` disabled it to keep the recording: delete the
            // recordings (stored minutes cost money) and the input here.
            let videos: Value = http
                .get(format!("{api}/{uid}/videos"))
                .bearer_auth(&token)
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap_or(Value::Null);
            let stream_api = api.trim_end_matches("/live_inputs");
            for v in videos["result"].as_array().into_iter().flatten() {
                if let Some(vid) = v["uid"].as_str() {
                    let d = http.delete(format!("{stream_api}/{vid}")).bearer_auth(&token).send().await.unwrap();
                    println!("cleanup recording {vid}: HTTP {}", d.status());
                }
            }
            let d = http.delete(format!("{api}/{uid}")).bearer_auth(&token).send().await.unwrap();
            assert!(d.status().is_success(), "deleting the recorded test input {uid}");
        } else {
            assert_eq!(st, reqwest::StatusCode::NOT_FOUND, "{uid} should be deleted");
        }
    }
}
