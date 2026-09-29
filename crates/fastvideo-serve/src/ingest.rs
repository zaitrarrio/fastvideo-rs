//! Native WHIP ingest: `/fv/v1/streams/ingest` (design §5.11).
//!
//! A client publishes its camera and microphone to a **duplex** model with
//! one WHIP request (RFC 9725 shape): `POST` an SDP offer, get `201
//! Created`, the answer and a `Location`. When the offer's m-lines are
//! `sendrecv`, the model's output comes back on the same peer (the browser
//! sees it at once, no second WHEP request); a `sendonly` offer (OBS, a
//! WHIP encoder) only feeds the model.
//!
//! | Route | Behaviour |
//! |---|---|
//! | `POST /fv/v1/streams/ingest?model=…` (`application/sdp`) | Admission (one session per executor: 429 + `Retry-After` when busy; 503 while loading), a non-trickle answer that receives VP8/H.264 + Opus (capped with `b=AS` at the model's `max_bitrate_kbps`) and sends the model's output (H.264 when an encoder is available, else VP8; Opus stereo when the model has audio). Query: `max_seconds` (the causal session-length rule, design §5.2), `scene`, `persona` (the session context), `width`/`height`/`fps` (output canvas) |
//! | `GET /fv/v1/streams/ingest` | Live and recently ended ingest streams |
//! | `GET /fv/v1/streams/ingest/{id}` | State, the model's duplex state, the ingest counters (decoded, dropped, refused), peer stats |
//! | `POST /fv/v1/streams/ingest/{id}/commands` | `{"type": "set_paused", "data": {"paused": true}}`, `{"type": "get_state"}` |
//! | `DELETE /fv/v1/streams/ingest/{id}` | Stop (the WHIP resource) |
//! | `PATCH /fv/v1/streams/ingest/{id}` | 405: answers are complete, trickle is not needed |
//!
//! Every route needs the native API key (`Authorization: Bearer`). Limits
//! come from the model's input caps (size, fps, bitrate) and the session
//! length; the ingest refuses larger pictures and drops media over the cap
//! (counted in the stream's `ingest` object).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_engine_service::{DuplexCommand, DuplexControl, EngineService, TickReceiver};
use fastvideo_media::decode::DecoderPool;
use fastvideo_media::pacer::VideoOut;
use fastvideo_protocol::{
    canvas_for_aspect, ApiError, CausalLimits, Continuity, DuplexSpec, ErrorKind, InputVideoCodec, ModelCaps,
    ProtocolId, RgbFrame, SessionContext, SessionSpec, StreamCaps, TrackSet,
};
use fastvideo_serve_kit::ServeCtx;
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout, PeerEvent, PeerHandle, ReceiveOptions, RtcHost};
use fastvideo_webrtc::ingest::{Ingest, IngestConfig, IngestStats};
use fastvideo_webrtc::writer::{AudioPacket, VideoCodec, VideoFrame};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Notify;

use crate::gate::ServiceGate;

/// Ingest settings (from `[webrtc]` / `[streams]`).
#[derive(Clone, Debug)]
pub struct IngestSettings {
    /// H.264 encoder for the output (`auto` | `nvenc` | `openh264` |
    /// `x264-test`), as `[webrtc] encoder`.
    pub encoder: String,
    /// The session-length rule (design §5.2).
    pub limits: CausalLimits,
    /// Ended streams kept for `GET`.
    pub keep_ended: usize,
}

impl IngestSettings {
    pub fn from_config(c: &crate::config::Config) -> Self {
        Self { encoder: c.webrtc.encoder.clone(), limits: c.streams.causal_limits(), keep_ended: 32 }
    }
}

/// Query of `POST /fv/v1/streams/ingest`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestQuery {
    pub model: String,
    #[serde(default)]
    pub max_seconds: Option<u32>,
    #[serde(default)]
    pub scene: Option<String>,
    #[serde(default)]
    pub persona: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub fps: Option<u32>,
}

#[derive(Default)]
struct Counters {
    frames_sent: AtomicU64,
    keyframes_sent: AtomicU64,
    audio_packets_sent: AtomicU64,
    keyframe_requests: AtomicU64,
    encode_errors: AtomicU64,
    /// Pictures dropped because the video encoder was behind.
    frames_dropped: AtomicU64,
}

struct Entry {
    id: String,
    model: String,
    created_ms: u64,
    max_seconds: u32,
    canvas: (u32, u32),
    fps: u32,
    state: Mutex<&'static str>,
    end_reason: Mutex<Option<String>>,
    video_codec: Mutex<Option<String>>,
    control: Mutex<Option<DuplexControl>>,
    ingest: Mutex<Option<IngestStats>>,
    peer: Mutex<Option<PeerHandle>>,
    counters: Counters,
    stop: Notify,
}

impl Entry {
    fn view(&self) -> Value {
        let g = |m: &Mutex<Option<String>>| m.lock().ok().and_then(|g| g.clone());
        let duplex = self.control.lock().ok().and_then(|c| c.as_ref().map(|c| c.state()));
        let ingest = self.ingest.lock().ok().and_then(|i| i.clone());
        let peer = self.peer.lock().ok().and_then(|p| p.as_ref().map(|p| p.stats()));
        let c = &self.counters;
        json!({
            "id": self.id,
            "object": "fv.ingest_stream",
            "model": self.model,
            "created_at": self.created_ms / 1000,
            "state": *self.state.lock().map(|s| *s).as_ref().unwrap_or(&"closed"),
            "end_reason": g(&self.end_reason),
            "max_seconds": self.max_seconds,
            "output": {
                "width": self.canvas.0, "height": self.canvas.1, "fps": self.fps,
                "video_codec": g(&self.video_codec),
                "frames_sent": c.frames_sent.load(Ordering::Relaxed),
                "keyframes_sent": c.keyframes_sent.load(Ordering::Relaxed),
                "audio_packets_sent": c.audio_packets_sent.load(Ordering::Relaxed),
                "keyframe_requests": c.keyframe_requests.load(Ordering::Relaxed),
                "encode_errors": c.encode_errors.load(Ordering::Relaxed),
                "frames_dropped": c.frames_dropped.load(Ordering::Relaxed),
            },
            "session": duplex,
            "ingest": ingest,
            "peer": peer,
        })
    }

    fn set_state(&self, s: &'static str) {
        if let Ok(mut g) = self.state.lock() {
            *g = s;
        }
    }
}

/// The ingest registry.
pub struct IngestStreams {
    engine: EngineService,
    host: RtcHost,
    cfg: IngestSettings,
    map: Mutex<HashMap<String, Arc<Entry>>>,
    order: Mutex<VecDeque<String>>,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn err_reply(e: &ApiError) -> Response {
    let status = match e.kind {
        ErrorKind::Conflict => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
    };
    let kind = serde_json::to_value(e.kind).unwrap_or(Value::Null);
    let mut r = (status, Json(json!({"error": {"kind": kind, "message": e.message, "param": e.param}}))).into_response();
    if let Some(s) = e.retry_after_s {
        if let Ok(v) = HeaderValue::from_str(&s.to_string()) {
            r.headers_mut().insert("retry-after", v);
        }
    }
    r
}

impl IngestStreams {
    pub fn new(engine: EngineService, host: RtcHost, cfg: IngestSettings) -> Self {
        Self { engine, host, cfg, map: Mutex::new(HashMap::new()), order: Mutex::new(VecDeque::new()) }
    }

    fn get(&self, id: &str) -> Option<Arc<Entry>> {
        self.map.lock().ok()?.get(id).cloned()
    }

    fn insert(&self, e: Arc<Entry>) {
        let id = e.id.clone();
        if let (Ok(mut m), Ok(mut o)) = (self.map.lock(), self.order.lock()) {
            m.insert(id.clone(), e);
            o.push_back(id);
            // Forget the oldest ended streams beyond `keep_ended`.
            let ended: Vec<String> = o
                .iter()
                .filter(|id| m.get(*id).is_none_or(|e| e.state.lock().map(|s| *s == "closed").unwrap_or(true)))
                .cloned()
                .collect();
            for id in ended.iter().take(ended.len().saturating_sub(self.cfg.keep_ended)) {
                m.remove(id);
                o.retain(|x| x != id);
            }
        }
    }

    fn list(&self) -> Vec<Value> {
        let ids: Vec<String> = self.order.lock().map(|o| o.iter().rev().cloned().collect()).unwrap_or_default();
        ids.iter().filter_map(|id| self.get(id)).map(|e| e.view()).collect()
    }
}

/// The duplex session spec for a query (pure; exposed for tests): output
/// canvas and fps from the query or the model, stereo audio when the model
/// sends audio, the session-length rule.
pub fn duplex_spec(caps: &ModelCaps, q: &IngestQuery, limits: &CausalLimits) -> Result<DuplexSpec, ApiError> {
    let Some(d) = caps.stream.as_ref().and_then(StreamCaps::duplex) else {
        return Err(ApiError::invalid_param("model", format!("`{}` does not take client input (not a duplex model)", caps.id)));
    };
    let fps = q.fps.unwrap_or(d.target_fps);
    if !caps.fps.allows(fps) {
        return Err(ApiError::invalid_param("fps", format!("fps {fps} is not supported by `{}`", caps.id)));
    }
    let canvas = match (q.width, q.height) {
        (Some(w), Some(h)) => (w, h),
        (None, None) => {
            let short = caps.canvas.short_edges.first().copied().unwrap_or(360);
            canvas_for_aspect(&caps.canvas, 16.0 / 9.0, short)
        }
        _ => return Err(ApiError::invalid_param("width", "give both width and height, or neither")),
    };
    if canvas.0 % 2 != 0 || canvas.1 % 2 != 0 || canvas.0 < 16 || canvas.1 < 16 {
        return Err(ApiError::invalid_param("width", "the canvas needs even sides of at least 16"));
    }
    if u64::from(canvas.0) * u64::from(canvas.1) > caps.canvas.max_area {
        return Err(ApiError::invalid_param("width", format!("{}x{} is larger than the model's canvas", canvas.0, canvas.1)));
    }
    let tracks = TrackSet::for_model(caps, canvas, fps, ("video", "audio"), 2, false);
    let tracks = if d.audio_out { tracks } else { TrackSet { audio: None, ..tracks } };
    let context = SessionContext { scene: q.scene.clone(), persona: q.persona.clone() };
    context.validate()?;
    Ok(DuplexSpec {
        session: SessionSpec {
            model: caps.id.clone(),
            tracks,
            canvas,
            fps,
            continuity: Continuity::HardCut,
            max_seconds: Some(limits.resolve(q.max_seconds)?),
            seed: None,
        },
        context,
    })
}

/// What the answer sends, in preference order: H.264 when an H.264
/// encoder is usable here, VP8 when ffmpeg has libvpx.
fn send_codecs(encoder: &str) -> Vec<VideoCodec> {
    let mut v = Vec::new();
    if h264_backend(encoder).is_some() {
        v.push(VideoCodec::H264);
    }
    if fastvideo_media::vp8::libvpx_available() {
        v.push(VideoCodec::Vp8);
    }
    if v.is_empty() {
        v.push(VideoCodec::H264);
    }
    v
}

fn h264_backend(choice: &str) -> Option<fastvideo_media::video::EncoderBackend> {
    use fastvideo_media::video::EncoderBackend;
    match choice {
        "nvenc" => Some(EncoderBackend::Nvenc),
        "openh264" => EncoderBackend::OpenH264.compiled().then_some(EncoderBackend::OpenH264),
        "x264-test" => Some(EncoderBackend::CpuTestX264),
        _ => {
            // `auto` is resolved at startup (`encoders::resolve`); a router
        // built without `App::build` probes here (blocking: callers are on
        // a blocking thread).
        let (_, sel) = crate::encoders::auto_selection();
            if sel.streams == "auto" {
                return None;
            }
            h264_backend(&sel.streams)
        }
    }
}

fn receive_options(caps: &ModelCaps) -> ReceiveOptions {
    let input = caps.stream.as_ref().and_then(StreamCaps::duplex).map(|d| d.input.clone());
    let Some(input) = input else { return ReceiveOptions::default() };
    ReceiveOptions {
        video_codecs: input
            .video
            .iter()
            .flat_map(|v| v.codecs.iter())
            .map(|c| match c {
                InputVideoCodec::Vp8 => VideoCodec::Vp8,
                InputVideoCodec::H264 => VideoCodec::H264,
            })
            .collect(),
        audio: input.audio.is_some(),
        max_bitrate_kbps: input.max_bitrate_kbps,
        ..ReceiveOptions::default()
    }
}

async fn create(st: &Arc<IngestStreams>, q: IngestQuery, offer: &str) -> Result<(Arc<Entry>, String), ApiError> {
    let caps = st
        .engine
        .caps()
        .get(&fastvideo_protocol::ModelId::new(&q.model))
        .or_else(|| st.engine.caps().resolve(&q.model))
        .cloned()
        .ok_or_else(|| ApiError::invalid_param("model", format!("model `{}` is not served here", q.model)))?;
    let spec = duplex_spec(&caps, &q, &st.cfg.limits)?;
    let duplex = caps.stream.as_ref().and_then(StreamCaps::duplex).cloned().expect("checked by duplex_spec");
    // Decoders for every accepted codec start now, before the first frame.
    let icfg = IngestConfig { max_seconds: spec.session.max_seconds, ..IngestConfig::new(duplex.input.video.clone(), duplex.input.audio.clone()) }
        .prewarm(DecoderPool::per_session());
    let session = st.engine.open_duplex_session(spec.clone()).await?;
    let audio = spec.session.tracks.has_audio();
    // Codec probes (ffmpeg) stay off the async runtime.
    let enc = st.cfg.encoder.clone();
    let video_codecs = tokio::task::spawn_blocking(move || send_codecs(&enc))
        .await
        .map_err(|e| ApiError::internal(format!("codec probe: {e}")))?;
    let opts = AnswerOptions {
        video: true,
        video_codecs,
        audio: audio.then_some(AudioLayout::Stereo),
        channels: fastvideo_webrtc::channel::ChannelPolicy::none(),
        receive: Some(receive_options(&caps)),
        ..AnswerOptions::default()
    };
    let (mut peer, answer) = st
        .host
        .answer(offer, opts)
        .await
        .map_err(|e| ApiError::new(ErrorKind::InvalidRequest, format!("offer: {e}")))?;
    let inbound = peer.take_inbound();
    let (handle, events) = peer.split();
    let started = Ingest::start(icfg, session.input().clone(), handle.clone())
        .map_err(|e| ApiError::internal(format!("ingest: {e}")))
        .and_then(|i| session.start().map(|(c, p)| (i, c, p)));
    let (ingest, control, paced) = match started {
        Ok(v) => v,
        Err(e) => {
            handle.close();
            return Err(e);
        }
    };
    control.set_paused(true);
    let entry = Arc::new(Entry {
        id: format!("fvingest_{}", fastvideo_serve_kit::random_token()),
        model: caps.id.to_string(),
        created_ms: now_ms(),
        max_seconds: spec.session.max_seconds.unwrap_or_default(),
        canvas: spec.session.canvas,
        fps: spec.session.fps,
        state: Mutex::new("connecting"),
        end_reason: Mutex::new(None),
        video_codec: Mutex::new(handle.video_codec().map(|c| format!("{c:?}").to_lowercase())),
        control: Mutex::new(Some(control.clone())),
        ingest: Mutex::new(None),
        peer: Mutex::new(Some(handle.clone())),
        counters: Counters::default(),
        stop: Notify::new(),
    });
    st.insert(entry.clone());
    let force = Arc::new(AtomicBool::new(true));
    let out = spawn_output(
        entry.clone(),
        paced.ticks,
        handle.clone(),
        spec.session.canvas,
        spec.session.fps,
        audio,
        force.clone(),
        st.cfg.encoder.clone(),
    );
    let e2 = entry.clone();
    let pace = paced.stats.clone();
    tokio::spawn(async move {
        let reason = run(&e2, events, inbound, &ingest, &control, &force, &pace).await;
        tracing::info!(stream = %e2.id, %reason, "ingest stream ended");
        control.close();
        handle.close();
        if let Ok(mut i) = e2.ingest.lock() {
            *i = Some(ingest.stats());
        }
        drop(ingest);
        let _ = out.await;
        if let Ok(mut r) = e2.end_reason.lock() {
            r.get_or_insert(reason);
        }
        e2.set_state("closed");
    });
    Ok((entry, answer))
}

/// Peer events and inbound media until the stream ends; returns why.
async fn run(
    e: &Arc<Entry>,
    mut events: tokio::sync::mpsc::UnboundedReceiver<PeerEvent>,
    mut inbound: Option<tokio::sync::mpsc::Receiver<fastvideo_webrtc::host::InboundMedia>>,
    ingest: &Ingest,
    control: &DuplexControl,
    force: &AtomicBool,
    pace: &tokio::sync::watch::Receiver<fastvideo_engine_service::PaceStats>,
) -> String {
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    loop {
        tokio::select! {
            _ = e.stop.notified() => return "stopped".into(),
            ev = events.recv() => match ev {
                Some(PeerEvent::Connected) => {
                    e.set_state("streaming");
                    control.set_paused(false);
                }
                Some(PeerEvent::KeyframeRequest { .. }) => {
                    e.counters.keyframe_requests.fetch_add(1, Ordering::Relaxed);
                    force.store(true, Ordering::Relaxed);
                }
                Some(PeerEvent::Closed(r)) => return format!("peer closed: {r:?}"),
                Some(_) => {}
                None => return "peer gone".into(),
            },
            m = async { match inbound.as_mut() { Some(r) => r.recv().await, None => std::future::pending().await } } => match m {
                Some(m) => ingest.push(m),
                None => inbound = None,
            },
            _ = tick.tick() => {
                if let Ok(mut i) = e.ingest.lock() {
                    *i = Some(ingest.stats());
                }
                if control.is_closed() {
                    return "session ended".into();
                }
                // The model ended on its own (`max_seconds` of output).
                if let Some(end) = pace.borrow().ended.clone() {
                    return serde_json::to_value(end)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_owned))
                        .unwrap_or_else(|| "ended".into());
                }
            }
        }
    }
}

/// The output encoder: H.264 (NVENC, OpenH264, x264) or VP8 (libvpx).
enum OutEncoder {
    H264(Box<dyn fastvideo_media::video::VideoEncoder>),
    Vp8(Box<fastvideo_media::vp8::Vp8Encoder>),
}

impl OutEncoder {
    fn new(codec: VideoCodec, encoder: &str, (w, h): (u32, u32), fps: u32) -> Result<Self, String> {
        match codec {
            VideoCodec::H264 => {
                let b = h264_backend(encoder).ok_or("no H.264 encoder here")?;
                fastvideo_media::video::create_encoder(b, fastvideo_media::video::H264Config::new(w, h, fps))
                    .map(OutEncoder::H264)
                    .map_err(|e| e.to_string())
            }
            VideoCodec::Vp8 => fastvideo_media::vp8::Vp8Encoder::with_pool(
                fastvideo_media::vp8::Vp8Config::new(w, h, fps),
                Some(fastvideo_media::pipe::SparePool::per_session()),
            )
            .map(|e| OutEncoder::Vp8(Box::new(e)))
            .map_err(|e| e.to_string()),
        }
    }

    fn encode(&mut self, f: &RgbFrame, key: bool) -> Result<Vec<fastvideo_media::video::EncodedFrame>, String> {
        match self {
            OutEncoder::H264(e) => {
                if key {
                    e.force_idr();
                }
                e.encode(f).map_err(|e| e.to_string())
            }
            OutEncoder::Vp8(e) => {
                if key {
                    e.force_keyframe();
                }
                e.encode(f).map_err(|e| e.to_string())
            }
        }
    }

    fn poll(&mut self) -> Vec<fastvideo_media::video::EncodedFrame> {
        match self {
            OutEncoder::H264(e) => e.poll().unwrap_or_default(),
            OutEncoder::Vp8(e) => e.poll().unwrap_or_default(),
        }
    }
}

/// The model's output to the peer: the tick thread sends Opus (stereo,
/// 20 ms) and hands pictures to the video encoder thread through a 2-frame
/// queue; when the encoder is behind (an ffmpeg start on a loaded host) a
/// picture is dropped and the next one is a keyframe, so audio never
/// waits for video. Fresh pictures are encoded; a repeat only when a
/// keyframe is due.
#[allow(clippy::too_many_arguments)]
fn spawn_output(
    e: Arc<Entry>,
    mut ticks: TickReceiver,
    peer: PeerHandle,
    canvas: (u32, u32),
    fps: u32,
    audio: bool,
    force: Arc<AtomicBool>,
    encoder: String,
) -> tokio::task::JoinHandle<()> {
    let rt = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let codec = peer.video_codec();
        if codec.is_none() {
            tracing::info!(stream = %e.id, "the offer receives no video: no video out");
        }
        let (vtx, vrx) = std::sync::mpsc::sync_channel::<(RgbFrame, u32, bool)>(2);
        let video = codec.map(|c| {
            let (e, peer, force, encoder) = (e.clone(), peer.clone(), force.clone(), encoder.clone());
            std::thread::Builder::new()
                .name("fv-ingest-video-out".into())
                .spawn(move || video_out(e, vrx, peer, c, canvas, fps, force, encoder))
        });
        let mut opus = audio
            .then(|| fastvideo_media::opus::OpusEncoder::new(fastvideo_media::opus::OpusConfig::whip(), 0).ok())
            .flatten();
        while let Some(t) = rt.block_on(ticks.recv()) {
            if let (Some(op), Some(a)) = (opus.as_mut(), t.audio.as_ref()) {
                if let Ok(pkts) = op.push(a) {
                    for p in pkts {
                        if peer.send_audio(AudioPacket::new(p.data, u64::from(p.rtp_ts))).is_ok() {
                            e.counters.audio_packets_sent.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
            if codec.is_none() {
                continue;
            }
            let key = force.swap(false, Ordering::Relaxed) || ticks.take_dropped();
            let pic = match t.video {
                VideoOut::Fresh(f) => Some(f),
                VideoOut::Repeat(f) if key => Some(f),
                _ => None,
            };
            let Some(f) = pic else {
                if key {
                    force.store(true, Ordering::Relaxed);
                }
                continue;
            };
            if vtx.try_send((f, t.video_rtp, key)).is_err() {
                // Behind: this picture is dropped, the next is a keyframe.
                e.counters.frames_dropped.fetch_add(1, Ordering::Relaxed);
                force.store(true, Ordering::Relaxed);
            }
        }
        drop(vtx);
        if let Some(Ok(h)) = video {
            let _ = h.join();
        }
    })
}

/// The video encoder thread of [`spawn_output`].
#[allow(clippy::too_many_arguments)]
fn video_out(
    e: Arc<Entry>,
    rx: std::sync::mpsc::Receiver<(RgbFrame, u32, bool)>,
    peer: PeerHandle,
    codec: VideoCodec,
    canvas: (u32, u32),
    fps: u32,
    force: Arc<AtomicBool>,
    encoder: String,
) {
    let mut enc = match OutEncoder::new(codec, &encoder, canvas, fps) {
        Ok(x) => Some(x),
        Err(err) => {
            tracing::warn!(stream = %e.id, error = %err, "no output encoder");
            None
        }
    };
    let mut rtps: VecDeque<u32> = VecDeque::new();
    let send = |frames: Vec<fastvideo_media::video::EncodedFrame>, rtps: &mut VecDeque<u32>| {
        for f in frames {
            let rtp = rtps.pop_front().unwrap_or_default();
            if peer.send_video(VideoFrame::new(f.data, u64::from(rtp))).is_ok() {
                e.counters.frames_sent.fetch_add(1, Ordering::Relaxed);
                if f.keyframe {
                    e.counters.keyframes_sent.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    };
    loop {
        // Pipe encoders hand frames back asynchronously: collect them
        // whether or not a picture arrives.
        let next = rx.recv_timeout(Duration::from_millis(20));
        if let Some(x) = enc.as_mut() {
            send(x.poll(), &mut rtps);
        }
        let (f, rtp, key) = match next {
            Ok(v) => v,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if enc.is_none() {
            enc = OutEncoder::new(codec, &encoder, canvas, fps).ok();
        }
        let Some(x) = enc.as_mut() else { continue };
        rtps.push_back(rtp);
        match x.encode(&f, key) {
            Ok(frames) => send(frames, &mut rtps),
            Err(err) => {
                e.counters.encode_errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(stream = %e.id, error = %err, "output encode failed");
                rtps.clear();
                enc = None;
                force.store(true, Ordering::Relaxed);
            }
        }
    }
}

fn auth(ctx: &ServeCtx, h: &HeaderMap) -> Result<(), Box<Response>> {
    ctx.auth().authenticate(ProtocolId::Native, h).map(|_| ()).map_err(|e| Box::new(err_reply(&e)))
}

async fn post_offer(ctx: ServeCtx, st: Arc<IngestStreams>, h: HeaderMap, q: IngestQuery, body: Bytes) -> Response {
    if let Err(r) = auth(&ctx, &h) {
        return *r;
    }
    let ct = h.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or_default();
    if !ct.starts_with("application/sdp") {
        return err_reply(&ApiError::new(ErrorKind::UnsupportedMedia, "WHIP offers are application/sdp"));
    }
    let Ok(offer) = std::str::from_utf8(&body) else {
        return err_reply(&ApiError::invalid_param("body", "the offer is not UTF-8"));
    };
    match create(&st, q, offer).await {
        Ok((e, answer)) => {
            let mut r = (StatusCode::CREATED, answer).into_response();
            let h = r.headers_mut();
            h.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/sdp"));
            if let Ok(v) = HeaderValue::from_str(&format!("/fv/v1/streams/ingest/{}", e.id)) {
                h.insert(header::LOCATION, v);
            }
            if let Ok(v) = HeaderValue::from_str(&e.id) {
                h.insert("x-fv-stream-id", v);
            }
            r
        }
        Err(e) => err_reply(&e),
    }
}

fn not_found(id: &str) -> Response {
    err_reply(&ApiError::new(ErrorKind::NotFound, format!("no ingest stream {id}")))
}

/// The `/fv/v1/streams/ingest` routes over `host` (the process's WebRTC host).
pub fn routes(gate: Arc<ServiceGate>, host: RtcHost, cfg: IngestSettings) -> Router<ServeCtx> {
    let st = Arc::new(IngestStreams::new(gate.engine().clone(), host, cfg));
    let (s1, s2, s3, s4, s5, s6) = (st.clone(), st.clone(), st.clone(), st.clone(), st.clone(), st);
    Router::new()
        .route(
            "/fv/v1/streams/ingest",
            post(move |State(ctx): State<ServeCtx>, h: HeaderMap, Query(q): Query<IngestQuery>, body: Bytes| {
                let st = s1.clone();
                async move { post_offer(ctx, st, h, q, body).await }
            })
            .get(move |State(ctx): State<ServeCtx>, h: HeaderMap| {
                let st = s2.clone();
                async move {
                    if let Err(r) = auth(&ctx, &h) {
                        return *r;
                    }
                    Json(json!({"object": "list", "data": st.list()})).into_response()
                }
            }),
        )
        .route(
            "/fv/v1/streams/ingest/{id}",
            get(move |State(ctx): State<ServeCtx>, h: HeaderMap, Path(id): Path<String>| {
                let st = s3.clone();
                async move {
                    if let Err(r) = auth(&ctx, &h) {
                        return *r;
                    }
                    match st.get(&id) {
                        Some(e) => Json(e.view()).into_response(),
                        None => not_found(&id),
                    }
                }
            })
            .delete(move |State(ctx): State<ServeCtx>, h: HeaderMap, Path(id): Path<String>| {
                let st = s4.clone();
                async move {
                    if let Err(r) = auth(&ctx, &h) {
                        return *r;
                    }
                    let Some(e) = st.get(&id) else { return not_found(&id) };
                    e.stop.notify_one();
                    for _ in 0..200 {
                        if e.state.lock().map(|s| *s == "closed").unwrap_or(true) {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(25)).await;
                    }
                    Json(e.view()).into_response()
                }
            })
            .patch(move |State(ctx): State<ServeCtx>, h: HeaderMap, Path(id): Path<String>| {
                let st = s5.clone();
                async move {
                    if let Err(r) = auth(&ctx, &h) {
                        return *r;
                    }
                    if st.get(&id).is_none() {
                        return not_found(&id);
                    }
                    let mut r = (StatusCode::METHOD_NOT_ALLOWED, "the answer is complete: no trickle ICE").into_response();
                    r.headers_mut().insert(header::ALLOW, HeaderValue::from_static("GET, DELETE"));
                    r
                }
            }),
        )
        .route(
            "/fv/v1/streams/ingest/{id}/commands",
            post(move |State(ctx): State<ServeCtx>, h: HeaderMap, Path(id): Path<String>, Json(body): Json<Value>| {
                let st = s6.clone();
                async move {
                    if let Err(r) = auth(&ctx, &h) {
                        return *r;
                    }
                    let Some(e) = st.get(&id) else { return not_found(&id) };
                    let ctl = e.control.lock().ok().and_then(|c| c.clone());
                    let Some(ctl) = ctl.filter(|c| !c.is_closed()) else {
                        return err_reply(&ApiError::conflict("the stream has ended"));
                    };
                    match serde_json::from_value::<DuplexCommand>(body) {
                        Ok(cmd) => Json(json!({"reply": ctl.apply(cmd)})).into_response(),
                        Err(err) => err_reply(&ApiError::invalid_param("type", err.to_string())),
                    }
                }
            }),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(model: &str) -> IngestQuery {
        IngestQuery { model: model.into(), ..IngestQuery::default() }
    }

    #[test]
    fn spec_follows_the_duplex_caps() {
        let echo = fastvideo_engine_service::echo_caps();
        let l = CausalLimits::default();
        let s = duplex_spec(&echo, &q("fv-echo"), &l).unwrap();
        assert_eq!((s.session.canvas, s.session.fps), ((640, 360), 30));
        assert_eq!(s.session.tracks.audio.as_ref().map(|a| a.channels), Some(2));
        assert_eq!(s.session.max_seconds, Some(120));
        let s = duplex_spec(&echo, &IngestQuery { max_seconds: Some(900), scene: Some("desk".into()), ..q("fv-echo") }, &l).unwrap();
        assert_eq!(s.session.max_seconds, Some(300));
        assert_eq!(s.context.scene.as_deref(), Some("desk"));
        assert_eq!(duplex_spec(&echo, &IngestQuery { fps: Some(7), ..q("fv-echo") }, &l).unwrap_err().param.as_deref(), Some("fps"));
        assert_eq!(
            duplex_spec(&echo, &IngestQuery { width: Some(3840), height: Some(2160), ..q("fv-echo") }, &l).unwrap_err().param.as_deref(),
            Some("width")
        );
        let h3 = fastvideo_engine_service::FakeModel::h3_max().caps;
        assert_eq!(duplex_spec(&h3, &q("h3"), &l).unwrap_err().param.as_deref(), Some("model"));
        let r = receive_options(&echo);
        assert_eq!(r.video_codecs, vec![VideoCodec::Vp8, VideoCodec::H264]);
        assert_eq!(r.max_bitrate_kbps, 4000);
    }
}
