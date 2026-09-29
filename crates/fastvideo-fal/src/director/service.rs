//! The director service: configuration, admission, the session registry
//! (heartbeats) and session creation over the shared WebRTC host.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_media::video::EncoderBackend;
use fastvideo_protocol::{
    canvas_for_aspect, h3_1080p_canvas, ApiError, Continuity, ErrorKind, Family, ModelCaps, ModelId, SessionSpec, StreamCaps, Task, TrackSet,
};
use fastvideo_serve_kit::{IngestPolicy, ServeCtx};
use fastvideo_webrtc::channel::ChannelPolicy;
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout, RtcHost};
use fastvideo_webrtc::ice::IceServer;
use fastvideo_webrtc::writer::VideoCodec as WVideoCodec;
use serde_json::Value;

use super::control::Limits;
use super::engine::{frames_for, DirectorEngine};
use super::info::{app_name, director_info, InfoFacts};
use super::media::VideoCodec;
use super::messages::{Aspect, Resolution};
use super::session::{self, SessionHandle, SessionInit};
use crate::FalApp;

/// `[fal] director` settings.
#[derive(Clone, Debug)]
pub struct DirectorConfig {
    /// fal apps whose `{app}/director` endpoint we serve (`app_id` of
    /// `/wma/session`); the first one also answers the runner routes.
    pub apps: Vec<FalApp>,
    /// ICE servers handed to clients by `/ice` (none of ours is TURN: str0m
    /// has no TURN client, risk R2; the browser may bring its own).
    pub ice_servers: Vec<IceServer>,
    /// Default chunk duration (`default_chunk_duration`), clamped to the model.
    pub chunk_seconds: f64,
    /// `max_session_seconds` in emitted video time; `None` = unlimited.
    pub max_session_seconds: Option<u64>,
    /// Three missed 5 s heartbeats (design §5.6).
    pub heartbeat_timeout: Duration,
    /// `configure` must arrive this soon after the control channel opens.
    pub configure_timeout: Duration,
    /// Built chunks queued behind the one playing (host RAM: one 10 s 768p
    /// chunk is ~750 MB of RGB).
    pub buffer_chunks: usize,
    /// H.264 backend (NVENC in production, design §0.1).
    pub h264: EncoderBackend,
    /// Video bitrate; `None`: by canvas (§5.1).
    pub video_bitrate: Option<u32>,
    /// Answer VP8 (ffmpeg `libvpx`; intra-only libwebp when ffmpeg has no
    /// libvpx) to offers without H.264; real
    /// Chrome/Safari/Firefox offer H.264 and get it. Also what every offer
    /// gets when `h264` is not in this build (see [`video_codecs`]).
    pub vp8_fallback: bool,
    /// Audio crossfade at chunk joins (design §5.5).
    pub crossfade_ms: u16,
    /// Limits for `image_url` / `end_image_url` fetches.
    pub ingest: IngestPolicy,
    /// Session directories (anchor frames, staged images).
    pub work_dir: PathBuf,
    /// Concurrent sessions (`one_session_per_machine`).
    pub max_sessions: usize,
}

impl Default for DirectorConfig {
    fn default() -> Self {
        Self {
            apps: vec![FalApp::h3(fastvideo_protocol::Tier::Max)],
            ice_servers: vec![IceServer::default_stun()],
            chunk_seconds: 10.0,
            max_session_seconds: None,
            heartbeat_timeout: Duration::from_secs(15),
            configure_timeout: Duration::from_secs(60),
            buffer_chunks: 1,
            h264: EncoderBackend::Nvenc,
            video_bitrate: None,
            vp8_fallback: true,
            crossfade_ms: 20,
            ingest: IngestPolicy::default(),
            work_dir: std::env::temp_dir().join("fv-director"),
            max_sessions: 1,
        }
    }
}

/// The video codecs the director answers, in preference order: H.264, then
/// VP8 with `vp8_fallback`. When the H.264 backend is not in this build, VP8
/// comes first: an `auto` encoder on a GPU without NVENC (H100, A100 have no
/// NVENC hardware) resolves to OpenH264, which images built without the
/// `openh264` feature lack, and a browser (which offers both) must get the
/// VP8 we can encode, not an H.264 that fails the session at its first
/// frame ("built without the `openh264` feature"). H.264 stays in the list
/// so an H.264-only offer is still answered as before.
pub fn video_codecs(h264_compiled: bool, vp8_fallback: bool) -> Vec<WVideoCodec> {
    match (h264_compiled, vp8_fallback) {
        (true, true) => vec![WVideoCodec::H264, WVideoCodec::Vp8],
        (false, true) => vec![WVideoCodec::Vp8, WVideoCodec::H264],
        (_, false) => vec![WVideoCodec::H264],
    }
}

/// `(width, height)` of `res` at `aspect` on the model's canvas rules.
/// The H3 1080P tier streams the generation canvas (1920x1088 at 16:9):
/// chunks are not cropped.
pub fn canvas_for(caps: &ModelCaps, res: Resolution, aspect: Aspect) -> (u32, u32) {
    if caps.family == Family::H3 && caps.canvas.is_hd(res.short_edge()) {
        let (w, h, _) = h3_1080p_canvas(caps, aspect.ratio(), 1.0);
        return (w, h);
    }
    canvas_for_aspect(&caps.canvas, aspect.ratio(), res.short_edge())
}

/// Resolutions a model serves: its canvas tiers among 480p / 768p / 1080p
/// (1080p only with the opt-in H3 1080P tier: each chunk takes about 2.5x
/// as long to build as at 768p).
pub fn served_resolutions(caps: &ModelCaps) -> Vec<Resolution> {
    [Resolution::R480, Resolution::R768, Resolution::R1080]
        .into_iter()
        .filter(|r| caps.canvas.short_edges.contains(&r.short_edge()))
        .collect()
}

/// The control limits for a model.
pub fn limits_for(cfg: &DirectorConfig, caps: &ModelCaps) -> Limits {
    let fps = caps.fps.default;
    let (min_s, max_s) = match caps.stream {
        Some(StreamCaps::Clip { min_s, max_s }) => (f64::from(min_s), f64::from(max_s)),
        _ => (5.0, 15.0),
    };
    let max = max_s.min(15.0);
    let min = min_s.max(5.0).min(max);
    Limits {
        fps,
        chunk_seconds: cfg.chunk_seconds.clamp(min, max),
        min_chunk_seconds: min,
        max_chunk_seconds: max,
        resolutions: served_resolutions(caps),
        ..Limits::default()
    }
}

/// One media stream for every track of the answer: browsers group tracks
/// by `msid` stream id, so the video and audio tracks arrive in a single
/// `MediaStream` (one `onMedia` call, one `<video>` element with sound).
pub fn unify_msid(answer: &str, stream: &str) -> String {
    let mut out = String::with_capacity(answer.len());
    for line in answer.split_inclusive('\n') {
        let body = line.trim_end_matches(['\r', '\n']);
        let eol = &line[body.len()..];
        if let Some(rest) = body.strip_prefix("a=msid:") {
            if let Some((_, track)) = rest.split_once(' ') {
                out.push_str(&format!("a=msid:{stream} {track}{eol}"));
                continue;
            }
        }
        if let Some(rest) = body.strip_prefix("a=ssrc:") {
            if let Some((ssrc, attr)) = rest.split_once(" msid:") {
                if let Some((_, track)) = attr.split_once(' ') {
                    out.push_str(&format!("a=ssrc:{ssrc} msid:{stream} {track}{eol}"));
                    continue;
                }
            }
        }
        out.push_str(line);
    }
    out
}

/// A new session's answer.
#[derive(Debug)]
pub struct Created {
    pub session_id: String,
    pub sdp: String,
    pub handle: Arc<SessionHandle>,
    pub caps: ModelCaps,
}

/// The director: config, the shared WebRTC host, the engine seam and the
/// open sessions.
pub struct DirectorService {
    cfg: Arc<DirectorConfig>,
    host: RtcHost,
    engine: Arc<dyn DirectorEngine>,
    sessions: Mutex<HashMap<String, Arc<SessionHandle>>>,
}

impl std::fmt::Debug for DirectorService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectorService").field("host", &self.host).field("apps", &self.cfg.apps).finish()
    }
}

impl DirectorService {
    pub fn new(cfg: DirectorConfig, host: RtcHost, engine: Arc<dyn DirectorEngine>) -> Arc<Self> {
        Arc::new(Self { cfg: Arc::new(cfg), host, engine, sessions: Mutex::new(HashMap::new()) })
    }

    pub fn config(&self) -> &DirectorConfig {
        &self.cfg
    }

    pub fn host(&self) -> &RtcHost {
        &self.host
    }

    /// The fal app of a director endpoint id (`minimax/h3-max/director`).
    pub fn app_for(&self, app_id: &str) -> Option<&FalApp> {
        let id = app_id.trim().trim_matches('/');
        let base = id.strip_suffix("/director")?;
        self.cfg.apps.iter().find(|a| a.id == base)
    }

    /// The app answering the runner routes (`/start-session`, `/info`).
    pub fn runner_app(&self) -> Option<&FalApp> {
        self.cfg.apps.first()
    }

    pub fn session(&self, id: &str) -> Option<Arc<SessionHandle>> {
        self.sessions.lock().expect("sessions").get(id).cloned()
    }

    /// Sessions not yet closed.
    pub fn open_sessions(&self) -> usize {
        self.sessions.lock().expect("sessions").values().filter(|s| s.is_open()).count()
    }

    /// The app's model caps (clip streaming required). Causal (SF-Wan)
    /// models are refused here, so the causal session limits of design §5.2
    /// (`[streams] causal_*_max_s`) never apply to the director; its own
    /// `max_session_seconds` bounds clip sessions.
    pub fn caps_for(&self, ctx: &ServeCtx, app: &FalApp) -> Result<ModelCaps, ApiError> {
        let id = crate::queue::resolve_app_model(ctx, app, crate::Endpoint::TextToVideo)?;
        let caps = ctx
            .engine()
            .models()
            .into_iter()
            .find(|c| c.id.0 == id)
            .ok_or_else(|| ApiError::not_found(format!("Application \"{}/director\" not found", app.id)))?;
        if !matches!(caps.stream, Some(StreamCaps::Clip { .. })) {
            return Err(ApiError::invalid(format!("model `{}` does not support clip streaming", caps.id)));
        }
        Ok(caps)
    }

    /// The per-session facts `session_info` / `/info` report.
    pub fn facts(&self, app: &FalApp, caps: &ModelCaps, audio: bool) -> InfoFacts {
        let limits = limits_for(&self.cfg, caps);
        InfoFacts {
            app: app_name(&app.id),
            default_chunk_frames: frames_for(caps, limits.fps, limits.chunk_seconds).unwrap_or(caps.frames.default),
            limits,
            max_session_seconds: self.cfg.max_session_seconds,
            audio,
        }
    }

    /// `DirectorInfo` of the runner app.
    pub fn info(&self, ctx: &ServeCtx) -> Result<Value, ApiError> {
        let app = self.runner_app().ok_or_else(|| ApiError::not_found("no director app is configured"))?;
        let caps = self.caps_for(ctx, app)?;
        Ok(director_info(&self.facts(app, &caps, caps.has_native_audio())))
    }

    /// Admission, engine session, WebRTC answer, then the session task.
    pub async fn create(
        self: &Arc<Self>,
        ctx: &ServeCtx,
        app: &FalApp,
        offer: &str,
        session_id: Option<String>,
        heartbeats: bool,
    ) -> Result<Created, ApiError> {
        if self.open_sessions() >= self.cfg.max_sessions.max(1) {
            return Err(ApiError::new(ErrorKind::Conflict, "a director session is already running on this machine").with_retry_after(5));
        }
        let caps = self.caps_for(ctx, app)?;
        let fps = caps.fps.default;
        let facts_probe = limits_for(&self.cfg, &caps);
        let res = if facts_probe.resolutions.contains(&Resolution::R768) {
            Resolution::R768
        } else {
            *facts_probe.resolutions.last().ok_or_else(|| ApiError::invalid(format!("model `{}` serves neither 480p nor 768p", caps.id)))?
        };
        let canvas = canvas_for(&caps, res, Aspect::Landscape);
        let tracks = TrackSet::for_model(&caps, canvas, fps, ("video", "audio"), 2, false);
        let audio = tracks.has_audio();
        let continuity = if caps.supports(Task::I2V) {
            Continuity::AnchorLastFrame { crossfade_ms: self.cfg.crossfade_ms }
        } else {
            Continuity::Crossfade { ms: self.cfg.crossfade_ms }
        };
        let spec = SessionSpec {
            model: ModelId(caps.id.0.clone()),
            tracks,
            canvas,
            fps,
            continuity,
            max_seconds: self.cfg.max_session_seconds.map(|s| s as u32),
            seed: None,
        };
        // Starting counts as busy: the engine session is held from here.
        let clips = self.engine.open(spec).await?;
        let opts = AnswerOptions {
            video: true,
            audio: audio.then_some(AudioLayout::Stereo),
            channels: ChannelPolicy::fal(),
            video_codecs: video_codecs(self.cfg.h264.compiled(), self.cfg.vp8_fallback),
            ..AnswerOptions::default()
        };
        let (peer, answer) = self.host.answer(offer, opts).await.map_err(|e| match e {
            fastvideo_webrtc::WebrtcError::Sdp(e) => ApiError::invalid(format!("invalid offer: {e}")),
            fastvideo_webrtc::WebrtcError::PeerLimit(n) => ApiError::conflict(format!("peer limit ({n}) reached")).with_retry_after(5),
            other => ApiError::internal(format!("webrtc: {other}")),
        })?;
        let codec = match peer.video_codec() {
            Some(WVideoCodec::H264) => VideoCodec::H264,
            Some(WVideoCodec::Vp8) => VideoCodec::Vp8,
            None => {
                peer.close();
                return Err(ApiError::invalid("the offer has no receivable video we can send (H.264 or VP8)"));
            }
        };
        let id = session_id.filter(|s| !s.is_empty()).unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let answer = unify_msid(&answer, &format!("fv-director-{}", &id[..id.len().min(8)]));
        let handle = SessionHandle::new(id.clone(), heartbeats);
        self.sessions.lock().expect("sessions").insert(id.clone(), handle.clone());
        let (peer_handle, events) = peer.split();
        let init = SessionInit {
            cfg: self.cfg.clone(),
            ctx: ctx.clone(),
            clips,
            peer: peer_handle,
            events,
            codec,
            handle: handle.clone(),
            facts: self.facts(app, &caps, audio),
            continuity,
        };
        let svc = self.clone();
        let sid = id.clone();
        tracing::info!(session = %id, app = %app.id, model = %caps.id, ?codec, audio, "director session starting");
        tokio::spawn(async move {
            session::run(init).await;
            svc.sessions.lock().expect("sessions").remove(&sid);
        });
        Ok(Created { session_id: id, sdp: answer, handle, caps })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_vp8_when_h264_is_not_in_the_build() {
        use WVideoCodec::{Vp8, H264};
        assert_eq!(video_codecs(true, true), vec![H264, Vp8]);
        assert_eq!(video_codecs(true, false), vec![H264]);
        assert_eq!(video_codecs(false, true), vec![Vp8, H264]);
        assert_eq!(video_codecs(false, false), vec![H264]);
    }

    #[test]
    fn one_stream_for_all_tracks() {
        let a = "v=0\r\na=msid:s1 v1\r\na=ssrc:11 msid:s1 v1\r\na=ssrc:11 cname:x\r\na=msid:s2 a1\r\n";
        assert_eq!(
            unify_msid(a, "fv"),
            "v=0\r\na=msid:fv v1\r\na=ssrc:11 msid:fv v1\r\na=ssrc:11 cname:x\r\na=msid:fv a1\r\n"
        );
    }

    #[test]
    fn the_h3_1080p_tier_serves_director_1080p() {
        let mut caps = ModelCaps::h3("h3", false);
        caps.canvas.short_edges.push(480);
        assert_eq!(served_resolutions(&caps), [Resolution::R480, Resolution::R768]);
        caps.canvas = caps.canvas.clone().with_h3_1080p();
        assert_eq!(served_resolutions(&caps), [Resolution::R480, Resolution::R768, Resolution::R1080]);
        assert_eq!(canvas_for(&caps, Resolution::R768, Aspect::Landscape), (1344, 768));
        assert_eq!(canvas_for(&caps, Resolution::R1080, Aspect::Landscape), (1920, 1088));
        assert_eq!(canvas_for(&caps, Resolution::R1080, Aspect::Portrait), (1088, 1920));
        assert_eq!(canvas_for(&caps, Resolution::R1080, Aspect::Square), (1088, 1088));
    }
}
