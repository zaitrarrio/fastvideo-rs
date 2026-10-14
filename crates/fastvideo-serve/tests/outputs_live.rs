//! Stream outputs against real destinations, on the fake engine (gated; both
//! skip without their variable). docs/serve/stream-outputs.md "Testing".
//!
//! - `FV_TEST_OUTPUTS_MEDIAMTX=1`: a local MediaMTX (`FV_TEST_MEDIAMTX_BIN`,
//!   else v1.15.1 downloaded into the target dir) takes one stream per kind
//!   from fv-serve: WHIP (H.264, VP8, VP9), RTMP, RTMPS (a self-signed
//!   certificate from the `openssl` CLI) and SRT (with a passphrase the
//!   path requires). MediaMTX's API must show each path ready with a video
//!   track, and ffprobe reads the resolution back over RTSP. RTMPS uses a
//!   certificate from a test CA: refused by default (the output fails with
//!   the verification error), accepted with `FV_OUTPUTS_RTMPS_CA_FILE` and
//!   with `insecure_skip_verify`. SRT is skipped when the local ffmpeg has
//!   no libsrt. `ffmpeg -protocols` is printed.
//! - `FV_TEST_OUTPUTS_CLOUDFLARE=1` with `FV_CF_ACCOUNT_ID` and
//!   `FV_CF_STREAM_API_TOKEN`: one stream per kind (WHIP H.264 at 1344x768,
//!   WHIP VP8, WHIP VP9, RTMPS, SRT; `FV_TEST_CF_CASES=whip-vp8,srt` picks
//!   some) against Cloudflare Stream for `FV_TEST_CF_SECONDS` (60), live
//!   inputs named `fv-test-…`. WHEP playback: a str0m viewer records what
//!   Cloudflare delivers and ffmpeg decodes it (resolution, codec, luma, so
//!   a black picture shows). RTMPS/SRT: the HLS manifest and segments and
//!   the DASH manifest decode, with the HLS live-edge latency; the audio is
//!   followed through the master playlist (Cloudflare's separate
//!   `#EXT-X-MEDIA TYPE=AUDIO` rendition, or the variant's own segments when
//!   muxed), read from the oldest segment listed, decoded and measured: it
//!   must be present and not silent (`stream_outputs::hls::AudioLevels::
//!   audible`). These streams queue 19 more clips after the prompt so the
//!   audio keeps playing (a clip plays once, then the stream holds in
//!   silence). Each case prints ingest-to-playable time. Every input and its
//!   recordings are deleted.
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
use fastvideo_serve::stream_outputs::hls;
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

fn pem(label: &str, der: &[u8]) -> String {
    use base64::Engine as _;
    let b = base64::engine::general_purpose::STANDARD.encode(der);
    let lines: Vec<&str> = b.as_bytes().chunks(64).map(|c| std::str::from_utf8(c).unwrap()).collect();
    format!("-----BEGIN {label}-----\n{}\n-----END {label}-----\n", lines.join("\n"))
}

/// Writes `ca.pem` (a fresh test CA) and `server.crt` / `server.key` (a
/// certificate for 127.0.0.1 and localhost it issued) into `dir`.
fn test_ca(dir: &FsPath) {
    let ca_key = rcgen::KeyPair::generate().unwrap();
    let mut ca_params = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    ca_params.distinguished_name.push(rcgen::DnType::CommonName, "fastvideo test CA");
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let key = rcgen::KeyPair::generate().unwrap();
    let leaf = rcgen::CertificateParams::new(vec!["127.0.0.1".to_owned(), "localhost".to_owned()])
        .unwrap()
        .signed_by(&key, &rcgen::Issuer::new(ca_params, ca_key))
        .unwrap();
    std::fs::write(dir.join("ca.pem"), pem("CERTIFICATE", ca.der())).unwrap();
    std::fs::write(dir.join("server.crt"), pem("CERTIFICATE", leaf.der())).unwrap();
    std::fs::write(dir.join("server.key"), pem("PRIVATE KEY", &key.serialize_der())).unwrap();
}

struct Mtx {
    ca: PathBuf,
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
    // RTMPS: a certificate for 127.0.0.1 issued by a test CA no system
    // trusts (verification refuses it unless given `ca.pem`).
    let (key, crt) = (work.join("server.key"), work.join("server.crt"));
    test_ca(&work);
    let tls = true;
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
    let m = Mtx { ca: work.join("ca.pem"), child, api, rtsp, rtmp, rtmps: if tls { rtmps } else { 0 }, srt, webrtc, log };
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
    // The same server trusting the test CA (`[outputs] rtmps_ca_file`).
    let a_ca = app_with(&[("FV_OUTPUTS_ALLOW_PRIVATE", "1".into()), ("FV_OUTPUTS_RTMPS_CA_FILE", m.ca.display().to_string())]).await;
    if m.rtmps != 0 {
        rtmps_refuses_an_unknown_ca(&a, &m).await;
    }
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
    if m.rtmps != 0 {
        // fastvideo's TLS relay: ffmpeg needs only plain rtmp.
        kinds.push(("rtmps", json!({"kind": "rtmps", "url": format!("rtmps://127.0.0.1:{}/live", m.rtmps), "stream_key": "fvrtmps"})));
        kinds.push((
            "rtmps-insecure",
            json!({"kind": "rtmps", "url": format!("rtmps://localhost:{}/live", m.rtmps), "stream_key": "fvrtmpsi", "insecure_skip_verify": true}),
        ));
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
            "rtmps-insecure" => "live/fvrtmpsi",
            _ => "fvsrt",
        };
        // fake-h3-turbo: H3's native 1344x768 with audio.
        let body = json!({"model": "fake-h3-turbo", "prompt": "a lighthouse", "output": output});
        // RTMPS (verified) on the server that trusts the test CA.
        let a = if name == "rtmps" { &a_ca } else { &a };
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

/// RTMPS to a server whose certificate no trusted CA issued: the output
/// fails with a verification error, MediaMTX never sees a publisher, and
/// nothing is retried unverified.
async fn rtmps_refuses_an_unknown_ca(a: &App, m: &Mtx) {
    let output = json!({"kind": "rtmps", "url": format!("rtmps://127.0.0.1:{}/live", m.rtmps), "stream_key": "fvrefused"});
    let (s, v) = call(&a.router, "POST", "/fv/v1/streams", Some(json!({"model": "fake-h3-turbo", "prompt": "x", "output": output}))).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    let t0 = Instant::now();
    let st = loop {
        let (_, st) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
        if st["status"]["output_state"] == "failed" || t0.elapsed() > Duration::from_secs(30) {
            break st;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    let err = st["status"]["output_error"].as_str().unwrap_or_default().to_owned();
    println!("RESULT rtmps-unknown-ca: output_state {} after {:.1}s, output_error {err:?}", st["status"]["output_state"], t0.elapsed().as_secs_f64());
    assert_eq!(st["status"]["output_state"], "failed", "{st}");
    assert!(err.contains("failed verification") && err.contains("UnknownIssuer"), "{err}");
    assert!(!err.contains("fvrefused"), "the key is scrubbed: {err}");
    // Nothing reached MediaMTX on that path.
    let r: Value = reqwest::get(format!("http://127.0.0.1:{}/v3/paths/get/live/fvrefused", m.api)).await.unwrap().json().await.unwrap_or(Value::Null);
    assert_ne!(r["ready"], true, "{r}");
    call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
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
    let file = es_file(codec, &frames);
    std::fs::write(out, &file).unwrap();
    let fmt: &[&str] = if codec == VideoCodec::H264 { &["-f", "h264"] } else { &["-f", "ivf"] };
    (frames.len(), decode(out.to_str().unwrap(), fmt, None))
}

/// What a WHEP viewer received as one file ffmpeg reads: IVF for VP8/VP9,
/// Annex-B for H.264.
fn es_file(codec: VideoCodec, frames: &[bytes::Bytes]) -> Vec<u8> {
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
    file
}

// ---- Cloudflare (live) ---------------------------------------------------------------

struct Viewer {
    frames: Vec<bytes::Bytes>,
    keyframes: usize,
    size: Option<(u32, u32)>,
    audio: usize,
    /// From the WHEP POST to the first video frame.
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
    let mut v = Viewer { frames: Vec::new(), keyframes: 0, size: None, audio: 0, first_ms: None };
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        match tokio::time::timeout(Duration::from_millis(500), peer.next_event()).await {
            Ok(Some(PeerEvent::Media { kind: TrackKind::Video, data, .. })) => {
                v.first_ms.get_or_insert(t0.elapsed().as_millis());
                let size = match codec {
                    VideoCodec::H264 => fastvideo_media::h264::find_sps(&data).map(|s| (s.width, s.height)),
                    VideoCodec::Vp8 => fastvideo_media::decode::vp8_keyframe_size(&data),
                    VideoCodec::Vp9 => None,
                };
                if let Some(s) = size {
                    v.keyframes += 1;
                    v.size = Some(s);
                }
                v.frames.push(data);
            }
            Ok(Some(PeerEvent::Media { kind: TrackKind::Audio, .. })) => v.audio += 1,
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

/// Luma of what ffmpeg decodes from `input` (scaled to 64x36 gray): (frames,
/// mean luma, mean per-frame standard deviation, frames that look black:
/// mean < 20 and deviation < 4). A black or flat stream has a deviation near 0.
fn luma(input: &str, extra: &[&str], secs: Option<u32>) -> Option<(usize, f64, f64, usize)> {
    let mut cmd = Command::new(fastvideo_media::tools::ffmpeg_bin());
    cmd.args(["-v", "error"]).args(extra).args(["-i", input]);
    if let Some(s) = secs {
        cmd.args(["-t", &s.to_string()]);
    }
    let out = cmd.args(["-an", "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "-"]).output().ok()?;
    let px = 64 * 36;
    let n = out.stdout.len() / px;
    if n == 0 {
        return None;
    }
    let (mut mean_sum, mut dev_sum, mut black) = (0.0, 0.0, 0);
    for f in out.stdout.chunks_exact(px) {
        let m = f.iter().map(|&b| f64::from(b)).sum::<f64>() / px as f64;
        let d = (f.iter().map(|&b| (f64::from(b) - m).powi(2)).sum::<f64>() / px as f64).sqrt();
        mean_sum += m;
        dev_sum += d;
        if m < 20.0 && d < 4.0 {
            black += 1;
        }
    }
    Some((n, mean_sum / n as f64, dev_sum / n as f64, black))
}

/// Unix ms of an ISO-8601 UTC time like `2026-10-13T22:40:01.123Z` (or `+00:00`).
fn iso_ms(s: &str) -> Option<i64> {
    let s = s.trim();
    let (date, time) = s.split_once('T')?;
    let mut d = date.split('-').map(|x| x.parse::<i64>());
    let (y, mo, da) = (d.next()?.ok()?, d.next()?.ok()?, d.next()?.ok()?);
    let time = time.trim_end_matches('Z');
    let time = time.split(['+']).next()?;
    let mut t = time.split(':');
    let (h, mi) = (t.next()?.parse::<i64>().ok()?, t.next()?.parse::<i64>().ok()?);
    let sec: f64 = t.next()?.parse().ok()?;
    // Days from civil (Howard Hinnant).
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + da - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(((days * 86_400 + h * 3600 + mi * 60) as f64 * 1000.0 + sec * 1000.0) as i64)
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

/// What the HLS manifest serves now, fetched over HTTPS by the test (the
/// build pod's static ffmpeg cannot resolve host names): the master's
/// variants (`#EXT-X-STREAM-INF` lines, highest first), the live-edge
/// latency (now - the end of the last segment by EXT-X-PROGRAM-DATE-TIME,
/// when Cloudflare writes it), and the last `segments` segments of the
/// best video variant joined into one local file ffmpeg decodes (with the
/// EXT-X-MAP init segment for fMP4).
struct HlsGrab {
    variants: Vec<String>,
    latency_ms: Option<i64>,
    file: PathBuf,
    segments: usize,
    target_duration: Option<f64>,
}

async fn hls_grab(http: &reqwest::Client, master: &str, segments: usize, out: &FsPath) -> Result<HlsGrab, String> {
    let get = |u: reqwest::Url| async move {
        let r = http.get(u.clone()).send().await.map_err(|e| e.without_url().to_string())?;
        if !r.status().is_success() {
            return Err(format!("HTTP {} for {}", r.status(), u.path()));
        }
        r.bytes().await.map_err(|e| e.without_url().to_string())
    };
    let base = reqwest::Url::parse(master).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&get(base.clone()).await?).into_owned();
    // (bandwidth, info line, uri) of each video variant.
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
    let (_, _, uri) = vars.first().cloned().ok_or_else(|| format!("no variant in the master playlist: {}", text.lines().take(4).collect::<Vec<_>>().join(" | ")))?;
    let media_url = base.join(&uri).map_err(|e| e.to_string())?;
    let media = String::from_utf8_lossy(&get(media_url.clone()).await?).into_owned();
    let (mut pdt, mut dur, mut last_end, mut target) = (None::<i64>, 0.0f64, None::<i64>, None);
    let mut init = None;
    let mut segs = Vec::new();
    for l in media.lines() {
        if let Some(t) = l.strip_prefix("#EXT-X-PROGRAM-DATE-TIME:") {
            pdt = iso_ms(t);
        } else if let Some(t) = l.strip_prefix("#EXT-X-TARGETDURATION:") {
            target = t.trim().parse().ok();
        } else if let Some(m) = l.strip_prefix("#EXT-X-MAP:") {
            init = m.split("URI=\"").nth(1).and_then(|r| r.split('"').next()).map(str::to_owned);
        } else if let Some(d) = l.strip_prefix("#EXTINF:") {
            dur = d.split(',').next().and_then(|x| x.parse().ok()).unwrap_or(0.0);
        } else if !l.starts_with('#') && !l.trim().is_empty() {
            segs.push(l.trim().to_owned());
            if let Some(p) = pdt {
                let end = p + (dur * 1000.0) as i64;
                last_end = Some(end);
                pdt = Some(end);
            }
        }
    }
    if segs.is_empty() {
        return Err("the media playlist has no segment yet".into());
    }
    let mut file = Vec::new();
    if let Some(i) = &init {
        file.extend_from_slice(&get(media_url.join(i).map_err(|e| e.to_string())?).await?);
    }
    let take = segs.len().min(segments);
    for s in &segs[segs.len() - take..] {
        file.extend_from_slice(&get(media_url.join(s).map_err(|e| e.to_string())?).await?);
    }
    std::fs::write(out, &file).map_err(|e| e.to_string())?;
    Ok(HlsGrab {
        variants: vars.into_iter().map(|v| v.1).collect(),
        latency_ms: last_end.map(|e| now_ms() - e),
        file: out.to_owned(),
        segments: take,
        target_duration: target,
    })
}

/// The DASH manifest: HTTP status and its video representations
/// (`width`x`height` codecs).
async fn dash_manifest(http: &reqwest::Client, mpd: &str) -> String {
    let r = match http.get(mpd).send().await {
        Ok(r) => r,
        Err(e) => return format!("DASH: {}", e.without_url()),
    };
    let st = r.status();
    let text = r.text().await.unwrap_or_default();
    let reps: Vec<String> = text
        .split("<Representation")
        .skip(1)
        .filter_map(|r| {
            let attr = |n: &str| r.split(&format!(" {n}=\"")).nth(1).and_then(|x| x.split('"').next()).map(str::to_owned);
            Some(format!("{}x{} {}", attr("width")?, attr("height")?, attr("codecs").unwrap_or_default()))
        })
        .collect();
    format!("DASH HTTP {st}, video representations {reps:?}")
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
    // FV_TEST_CF_CASES=whip-vp8,srt runs a subset.
    let only: Option<Vec<String>> = std::env::var("FV_TEST_CF_CASES").ok().map(|s| s.split(',').map(|x| x.trim().to_owned()).collect());
    let a = app_with(&[
        ("FV_CF_ACCOUNT_ID", account.clone()),
        ("FV_CF_STREAM_API_TOKEN", token.clone()),
        ("FV_CF_STREAM_NAME_PREFIX", "fv-test-".into()),
    ])
    .await;
    let protocols = ffmpeg_protocols();
    println!("ffmpeg {:?}: rtmps {}, srt {}", fastvideo_media::tools::ffmpeg_bin(), has_protocol(&protocols, "rtmps"), has_protocol(&protocols, "libsrt") || has_protocol(&protocols, "srt"));
    let http = reqwest::Client::new();
    let api = format!("https://api.cloudflare.com/client/v4/accounts/{account}/stream/live_inputs");
    let mut uids = Vec::new();
    let mut failures = Vec::new();
    for (name, output, codec) in [
        ("whip-h264", json!({"kind": "cloudflare"}), Some(VideoCodec::H264)),
        ("whip-vp8", json!({"kind": "cloudflare", "codec": "vp8"}), Some(VideoCodec::Vp8)),
        ("whip-vp9", json!({"kind": "cloudflare", "codec": "vp9"}), Some(VideoCodec::Vp9)),
        ("rtmps", json!({"kind": "cloudflare-rtmps"}), None),
        ("srt", json!({"kind": "cloudflare-srt"}), None),
    ] {
        if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == name)) {
            continue;
        }
        let body = if codec.is_some() {
            json!({"model": "fake-h3-turbo", "prompt": "a lighthouse at dusk", "output": output})
        } else {
            // HLS: the audio rendition is listed well after the video; keep
            // clips (and their audio) playing for ~100 s (20 is the queue's
            // generation capacity).
            let more: Vec<Value> = (0..19).map(|i| json!({"prompt": format!("a lighthouse at dusk, shot {i}")})).collect();
            json!({"model": "fake-h3-turbo", "prompt": "a lighthouse at dusk", "clips": more, "output": output})
        };
        let t0 = Instant::now();
        let (s, v) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
        if s != StatusCode::CREATED {
            println!("RESULT {name}: create refused {s}: {v}");
            failures.push(name);
            continue;
        }
        let create_ms = t0.elapsed().as_millis();
        let id = v["id"].as_str().unwrap().to_owned();
        let pb = v["playback"].clone();
        if let Some(u) = pb["live_input_id"].as_str() {
            uids.push(u.to_owned());
        }
        println!("{name}: created in {create_ms} ms; playback {pb}");
        let result = match codec {
            Some(c) => {
                // Retry WHEP until the publisher is up (Cloudflare answers 409 /
                // 404 before), then watch for the rest of the case.
                let whep = pb["whep_url"].as_str().unwrap_or_default().to_owned();
                let mut outcome = Err("never tried".to_owned());
                let mut first_try = None;
                while t0.elapsed() < Duration::from_secs(secs.saturating_sub(10)) {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    let left = secs.saturating_sub(t0.elapsed().as_secs()).max(10);
                    let t_try = t0.elapsed();
                    match whep_view(&whep, c, left).await {
                        Ok(w) if !w.frames.is_empty() => {
                            first_try = Some(t_try);
                            outcome = Ok(w);
                            break;
                        }
                        Ok(_) => outcome = Err("WHEP connected but no video frames".into()),
                        Err(e) => outcome = Err(e),
                    }
                }
                match outcome {
                    Ok(w) => {
                        let file = std::env::temp_dir().join(format!("fv-cf-{name}.bin"));
                        std::fs::write(&file, es_file(c, &w.frames)).unwrap();
                        let fmt: &[&str] = if c == VideoCodec::H264 { &["-f", "h264"] } else { &["-f", "ivf"] };
                        let d = decode(file.to_str().unwrap(), fmt, None);
                        let l = luma(file.to_str().unwrap(), fmt, None);
                        let playable = first_try.map(|t| t.as_millis() + w.first_ms.unwrap_or(0));
                        if d.as_ref().is_none_or(|d| d.1 == 0) {
                            failures.push(name);
                        }
                        format!(
                            "WHEP: {} frames received ({} keyframes with a size, header size {:?}), {} audio packets; ingest-to-playable {:?} ms (create -> first WHEP video frame; first frame {:?} ms after the WHEP POST); decoded (codec, frames, keyframes, w, h) {d:?}; luma (frames, mean, deviation, black frames) {l:?}",
                            w.frames.len(), w.keyframes, w.size, w.audio, playable, w.first_ms
                        )
                    }
                    Err(e) => {
                        failures.push(name);
                        format!("WHEP failed: {e}")
                    }
                }
            }
            None => {
                // HLS appears a few segments after the broadcast starts.
                let hls = pb["hls_url"].as_str().unwrap_or_default().to_owned();
                let dash = pb["dash_url"].as_str().unwrap_or_default().to_owned();
                let file = std::env::temp_dir().join(format!("fv-cf-{name}.seg"));
                let mut playable = None;
                let mut last = String::new();
                // When Cloudflare reports the input connected (ingest), apart
                // from when HLS serves a decodable segment.
                let uid = pb["live_input_id"].as_str().unwrap_or_default().to_owned();
                let mut connected: Option<Duration> = None;
                // SRT's first HLS segment took 68-72 s on 2026-10-13, 95 s
                // and more than 120 s on 2026-10-14.
                while t0.elapsed() < Duration::from_secs(secs.max(180)) {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    if connected.is_none() {
                        if let Ok(r) = http.get(format!("{api}/{uid}")).bearer_auth(&token).send().await {
                            let v: Value = r.json().await.unwrap_or(Value::Null);
                            if v["result"]["status"]["current"]["state"] == "connected" {
                                connected = Some(t0.elapsed());
                            }
                        }
                    }
                    match hls_grab(&http, &hls, 1, &file).await {
                        Ok(g) if decode(g.file.to_str().unwrap(), &[], None).is_some_and(|d| d.1 > 0) => {
                            playable = Some(t0.elapsed());
                            break;
                        }
                        Ok(_) => last = "a segment that does not decode yet".into(),
                        Err(e) => last = e,
                    }
                }
                match playable {
                    None => {
                        failures.push(name);
                        format!("HLS never playable after {:.0}s (last: {last}; input connected after {connected:?})", t0.elapsed().as_secs_f64())
                    }
                    Some(p) => {
                        // Let a few more segments land, then decode and measure.
                        tokio::time::sleep(Duration::from_secs(12)).await;
                        let g = hls_grab(&http, &hls, 3, &file).await;
                        let (d, l, info) = match &g {
                            Ok(g) => (
                                decode(g.file.to_str().unwrap(), &[], None),
                                luma(g.file.to_str().unwrap(), &[], None),
                                format!("variants {:?}; live-edge latency {:?} ms (target duration {:?} s); {} segments", g.variants, g.latency_ms, g.target_duration, g.segments),
                            ),
                            Err(e) => (None, None, format!("grab failed: {e}")),
                        };
                        let dash_info = dash_manifest(&http, &dash).await;
                        if d.as_ref().is_none_or(|d| d.1 == 0) {
                            failures.push(name);
                        }
                        // The audio: Cloudflare's separate rendition (or muxed), from the oldest segment listed.
                        let afile = std::env::temp_dir().join(format!("fv-cf-{name}-audio.seg"));
                        let deadline = t0 + Duration::from_secs(secs.max(180) + 120);
                        let audio = match hls::read_audio(&http, &hls, Duration::from_secs(30), deadline, &afile).await {
                            Ok(a) => {
                                if !a.audible {
                                    failures.push(name);
                                }
                                format!("audio {} after {:.1}s: {}", if a.audible { "audible" } else { "SILENT OR MISSING" }, t0.elapsed().as_secs_f64(), serde_json::to_string(&a).unwrap_or_default())
                            }
                            Err(e) => {
                                failures.push(name);
                                format!("audio not read: {e}")
                            }
                        };
                        format!(
                            "HLS: ingest-to-playable {:.1}s (create -> first segment that decodes; Cloudflare showed the input connected after {:?}); {info}; decoded (codec, frames, keyframes, w, h) {d:?}; luma {l:?}; {dash_info}; {audio}",
                            p.as_secs_f64(),
                            connected.map(|c| format!("{:.1}s", c.as_secs_f64()))
                        )
                    }
                }
            }
        };
        if t0.elapsed() < Duration::from_secs(secs) {
            tokio::time::sleep(Duration::from_secs(secs) - t0.elapsed()).await;
        }
        let (_, st) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
        println!(
            "RESULT {name}: {result}; fv-serve sent {} frames ({} {}x{} @ {}), output_state {}, output_error {}",
            st["status"]["frames_sent"],
            st["status"]["video_codec"],
            st["status"]["width"],
            st["status"]["height"],
            st["status"]["fps"],
            st["status"]["output_state"],
            st["status"]["output_error"]
        );
        call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
    }
    // Every fv-test- input is gone (or disabled when it recorded).
    tokio::time::sleep(Duration::from_secs(3)).await;
    let stream_api = api.trim_end_matches("/live_inputs").to_owned();
    for uid in uids {
        let r = http.get(format!("{api}/{uid}")).bearer_auth(&token).send().await.unwrap();
        let st = r.status();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        println!("cleanup {uid}: HTTP {st}, enabled {}, recording {}", v["result"]["enabled"], v["result"]["recording"]["mode"]);
        if st.is_success() {
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
            for v in videos["result"].as_array().into_iter().flatten() {
                if let Some(vid) = v["uid"].as_str() {
                    // A recording still being processed answers 409: retry.
                    let t = Instant::now();
                    let status = loop {
                        let d = http.delete(format!("{stream_api}/{vid}")).bearer_auth(&token).send().await.unwrap();
                        if d.status() != reqwest::StatusCode::CONFLICT || t.elapsed() > Duration::from_secs(180) {
                            break d.status();
                        }
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    };
                    println!("cleanup recording {vid} ({}s): HTTP {status} after {:.0}s", v["duration"], t.elapsed().as_secs_f64());
                    assert!(status.is_success() || status == reqwest::StatusCode::NOT_FOUND, "deleting recording {vid}: {status}");
                }
            }
            let d = http.delete(format!("{api}/{uid}")).bearer_auth(&token).send().await.unwrap();
            assert!(d.status().is_success(), "deleting the test input {uid}");
        }
    }
    assert!(failures.is_empty(), "cases without playback or audio: {failures:?}");
}
