//! The director service: configuration, admission, the session registry
//! (heartbeats) and session creation over the shared WebRTC host.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_media::video::EncoderBackend;
use fastvideo_protocol::{
    canvas_for_aspect, h3_1080p_canvas, resolve_canvas, ApiError, CanvasSpec, Continuity, ErrorKind, Family, ModelCaps, ModelId, ResolvedCanvas,
    SessionSpec, StreamCaps, Task, TrackSet,
};
use fastvideo_serve_kit::{IngestPolicy, ServeCtx};
use fastvideo_webrtc::channel::ChannelPolicy;
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout, RtcHost};
use fastvideo_webrtc::ice::IceServer;
use fastvideo_webrtc::writer::VideoCodec as WVideoCodec;
use serde_json::Value;

use super::control::{CausalLimits, Limits};
use super::engine::{frames_for, DirectorEngine};
use super::info::{app_name, director_info, InfoFacts};
pub use super::info::{relative_chunk_cost, served_resolutions};
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
    /// Causal models: blocks per director chunk (`chunk` message; 4 blocks
    /// of 12 frames = 3 s at 16 fps).
    pub causal_chunk_blocks: u32,
    /// Causal models: video kept queued for playout; above it the rollout
    /// pauses at the next block boundary, so a prompt update reaches the
    /// picture within about this lead.
    pub causal_lead_seconds: f64,
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
            causal_chunk_blocks: 4,
            causal_lead_seconds: 2.0,
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

/// The canvas of `res` at `aspect` on the model's canvas rules:
/// `(width, height, crop)`, where `(width, height)` is what the engine
/// generates and `crop` the delivered size when that differs.
///
/// - The H3 1080P tier streams the generation canvas (1920x1088 at 16:9):
///   chunks are not cropped.
/// - A pad-and-crop model (LTX two-stage: sides multiples of 64) takes the
///   exact size the batch APIs give the same tier and aspect (even sides,
///   the tier on the short side) and generates it padded up to the multiple:
///   480p 16:9 is 854x480 from 896x512, 720p 1280x720 from 1280x768, 768p
///   1366x768 from 1408x768, 1080p 1920x1080 from 1920x1088. The session
///   centre-crops every frame; the next chunk's anchor is the uncropped
///   last frame, so continuity does not zoom.
/// - Any other model: the canvas-for-aspect rule (sides snapped to the
///   multiple), not cropped.
pub fn canvas_for(caps: &ModelCaps, res: Resolution, aspect: Aspect) -> ResolvedCanvas {
    if caps.family == Family::H3 && caps.canvas.is_hd(res.short_edge()) {
        let (w, h, _) = h3_1080p_canvas(caps, aspect.ratio(), 1.0);
        return (w, h, None);
    }
    if caps.canvas.pad_and_crop {
        let spec = CanvasSpec::Aspect { ratio: aspect.as_ratio(), short_edge: res.short_edge() };
        match resolve_canvas(&spec, caps, None) {
            Ok(c) => return c,
            // Not reached for a served tier (the dimension sweep checks every
            // one); the snapped canvas below is the engine-valid fallback.
            Err(e) => tracing::warn!(model = %caps.id, res = res.as_str(), aspect = aspect.as_str(), error = %e.message, "director canvas"),
        }
    }
    let (w, h) = canvas_for_aspect(&caps.canvas, aspect.ratio(), res.short_edge());
    (w, h, None)
}

/// The control limits for a model.
pub fn limits_for(cfg: &DirectorConfig, caps: &ModelCaps) -> Limits {
    let fps = caps.fps.default;
    if let Some(StreamCaps::Causal { block_frames, context, .. }) = caps.stream {
        let block_seconds = f64::from(block_frames) / f64::from(fps.max(1));
        let chunk_blocks = cfg.causal_chunk_blocks.max(1);
        let chunk = block_seconds * f64::from(chunk_blocks);
        return Limits {
            fps,
            chunk_seconds: chunk,
            min_chunk_seconds: chunk,
            max_chunk_seconds: chunk,
            resolutions: served_resolutions(caps),
            script_max_end_images: 0,
            causal: Some(CausalLimits { block_frames, block_seconds, chunk_blocks, context }),
            ..Limits::default()
        };
    }
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
        // The H3 1080P tier's clip cap (5 s; 10 s with `h3_1080p_long`).
        hd_max_chunk_seconds: caps
            .canvas
            .hd
            .filter(|t| t.short_edge == Resolution::R1080.short_edge())
            .and_then(|t| t.max_frames)
            .map(|n| (f64::from(n) / f64::from(fps.max(1))).floor()),
        ..Limits::default()
    }
}

/// One media stream for every track of the answer (moved to
/// `fastvideo_webrtc::sdp`, shared with the Reactor avatar mode).
pub use fastvideo_webrtc::sdp::unify_msid;

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

    /// The app's model caps: clip streaming, or a causal rollout (LongLive,
    /// SF-Wan; docs/serve/director-causal.md). The director's own
    /// `max_session_seconds` bounds both kinds of session (the Reactor's
    /// `[streams] causal_*_max_s` do not apply here).
    pub fn caps_for(&self, ctx: &ServeCtx, app: &FalApp) -> Result<ModelCaps, ApiError> {
        let id = crate::queue::resolve_app_model(ctx, app, crate::Endpoint::TextToVideo)?;
        let caps = ctx
            .engine()
            .models()
            .into_iter()
            .find(|c| c.id.0 == id)
            .ok_or_else(|| ApiError::not_found(format!("Application \"{}/director\" not found", app.id)))?;
        if !super::info::director_capable(&caps) {
            return Err(ApiError::invalid(format!("model `{}` supports neither clip streaming nor a causal rollout", caps.id)));
        }
        Ok(caps)
    }

    /// The per-session facts `session_info` / `/info` report.
    pub fn facts(&self, app: &FalApp, caps: &ModelCaps, audio: bool) -> InfoFacts {
        let limits = limits_for(&self.cfg, caps);
        let default_chunk_frames = match &limits.causal {
            Some(c) => c.block_frames * c.chunk_blocks,
            None => frames_for(caps, limits.fps, limits.chunk_seconds).unwrap_or(caps.frames.default),
        };
        InfoFacts {
            app: app_name(&app.id),
            default_chunk_frames,
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
        let causal = matches!(caps.stream, Some(StreamCaps::Causal { .. }));
        let facts_probe = limits_for(&self.cfg, &caps);
        let res = if facts_probe.resolutions.contains(&Resolution::R768) && !causal {
            Resolution::R768
        } else {
            *facts_probe.resolutions.last().ok_or_else(|| ApiError::invalid(format!("model `{}` serves neither 480p nor 768p", caps.id)))?
        };
        // The engine session generates `canvas`; the tracks carry the
        // delivered (cropped) size.
        let (w, h, crop) = canvas_for(&caps, res, Aspect::Landscape);
        let canvas = (w, h);
        let tracks = TrackSet::for_model(&caps, crop.unwrap_or(canvas), fps, ("video", "audio"), 2, false);
        let audio = tracks.has_audio();
        let continuity = if causal {
            // One rollout: the KV cache carries continuity.
            Continuity::HardCut
        } else if caps.supports(Task::I2V) {
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
        let (clips, stream) = if causal {
            let st = self.engine.open_stream(spec).await?;
            let clips: Arc<dyn super::engine::DirectorClips> = Arc::new(session::NoClips(st.caps().clone(), st.spec().clone()));
            (clips, Some(st))
        } else {
            (self.engine.open(spec).await?, None)
        };
        // Release a causal lease on any refusal below (a clip session goes
        // with its last reference).
        let release = |stream: &Option<Arc<dyn super::engine::DirectorStream>>| {
            if let Some(st) = stream.clone() {
                tokio::spawn(async move { st.close().await });
            }
        };
        let opts = AnswerOptions {
            video: true,
            audio: audio.then_some(AudioLayout::Stereo),
            channels: ChannelPolicy::fal(),
            video_codecs: video_codecs(self.cfg.h264.compiled(), self.cfg.vp8_fallback),
            ..AnswerOptions::default()
        };
        let (peer, answer) = match self.host.answer(offer, opts).await {
            Ok(x) => x,
            Err(e) => {
                release(&stream);
                return Err(match e {
                    fastvideo_webrtc::WebrtcError::Sdp(e) => ApiError::invalid(format!("invalid offer: {e}")),
                    fastvideo_webrtc::WebrtcError::PeerLimit(n) => ApiError::conflict(format!("peer limit ({n}) reached")).with_retry_after(5),
                    other => ApiError::internal(format!("webrtc: {other}")),
                });
            }
        };
        let codec = match peer.video_codec() {
            Some(WVideoCodec::H264) => VideoCodec::H264,
            Some(WVideoCodec::Vp8) => VideoCodec::Vp8,
            None => {
                peer.close();
                release(&stream);
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
            stream,
            peer: peer_handle,
            events,
            codec,
            handle: handle.clone(),
            facts: self.facts(app, &caps, audio),
            continuity,
        };
        let svc = self.clone();
        let sid = id.clone();
        tracing::info!(session = %id, app = %app.id, model = %caps.id, ?codec, audio, causal, "director session starting");
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
        assert_eq!(canvas_for(&caps, Resolution::R768, Aspect::Landscape), (1344, 768, None));
        assert_eq!(canvas_for(&caps, Resolution::R480, Aspect::Landscape), (832, 480, None));
        assert_eq!(canvas_for(&caps, Resolution::R1080, Aspect::Landscape), (1920, 1088, None));
        assert_eq!(canvas_for(&caps, Resolution::R1080, Aspect::Portrait), (1088, 1920, None));
        assert_eq!(canvas_for(&caps, Resolution::R1080, Aspect::Square), (1088, 1088, None));
        assert_eq!(relative_chunk_cost(&caps, Resolution::R1080), Some(2.5));
        assert_eq!(relative_chunk_cost(&caps, Resolution::R480), None);
        // 1080p chunks within the tier's clip cap: 5 s, 10 s with the flag.
        let l = limits_for(&DirectorConfig::default(), &caps);
        assert_eq!(l.hd_max_chunk_seconds, Some(5.0));
        let at = l.at(Resolution::R1080);
        assert_eq!((at.min_chunk_seconds, at.chunk_seconds, at.max_chunk_seconds), (5.0, 5.0, 5.0));
        assert_eq!(l.at(Resolution::R768), l);
        fastvideo_protocol::apply_feature_flags(&mut caps, &|_| true);
        let at = limits_for(&DirectorConfig::default(), &caps).at(Resolution::R1080);
        assert_eq!((at.chunk_seconds, at.max_chunk_seconds), (10.0, 10.0));
    }

    /// LTX two-stage caps as the CUDA catalog declares them (sides multiples
    /// of 64, pad-and-crop, the 480/720/768/1080 tiers among others).
    fn ltx_caps() -> ModelCaps {
        let mut caps = ModelCaps::h3("ltx", true);
        caps.family = Family::Ltx2;
        caps.canvas = fastvideo_protocol::CanvasCaps {
            multiple: 64,
            max_area: 3840 * 2176,
            aspect: (0.25, 4.0),
            short_edges: vec![1080, 480, 720, 768, 1440, 2160],
            pad_and_crop: true,
            hd: None,
        };
        caps
    }

    #[test]
    fn ltx_serves_480_720_768_and_1080_padded_and_cropped() {
        let caps = ltx_caps();
        assert_eq!(served_resolutions(&caps), Resolution::ALL);
        use Aspect::{Landscape as L, Portrait as P, Square as S};
        // (resolution, aspect) -> generated, delivered.
        let want = [
            (Resolution::R480, L, (896, 512), (854, 480)),
            (Resolution::R480, P, (512, 896), (480, 854)),
            (Resolution::R480, S, (512, 512), (480, 480)),
            (Resolution::R720, L, (1280, 768), (1280, 720)),
            (Resolution::R720, P, (768, 1280), (720, 1280)),
            (Resolution::R720, S, (768, 768), (720, 720)),
            (Resolution::R768, L, (1408, 768), (1366, 768)),
            (Resolution::R768, P, (768, 1408), (768, 1366)),
            (Resolution::R768, S, (768, 768), (768, 768)),
            (Resolution::R1080, L, (1920, 1088), (1920, 1080)),
            (Resolution::R1080, P, (1088, 1920), (1080, 1920)),
            (Resolution::R1080, S, (1088, 1088), (1080, 1080)),
        ];
        for (r, a, gen, out) in want {
            let (w, h, crop) = canvas_for(&caps, r, a);
            assert_eq!(((w, h), crop.unwrap_or((w, h))), (gen, out), "{} {}", r.as_str(), a.as_str());
            // Two-stage: stage 1 runs at half size on the latent grid (/32).
            assert_eq!((w % 64, h % 64), (0, 0), "{} {}", r.as_str(), a.as_str());
            assert_eq!(crop.is_some(), gen != out);
        }
        // No 1080p clip cap on LTX; every tier keeps the model's chunk range.
        let l = limits_for(&DirectorConfig::default(), &caps);
        assert_eq!(l.hd_max_chunk_seconds, None);
        for r in Resolution::ALL {
            assert_eq!(l.at(r), l);
        }
        // Labels: chunk cost next to 768p.
        assert_eq!(relative_chunk_cost(&caps, Resolution::R768), None);
        assert!(relative_chunk_cost(&caps, Resolution::R480).unwrap() < 1.0);
        assert!(relative_chunk_cost(&caps, Resolution::R720).unwrap() <= 1.0);
        assert!(relative_chunk_cost(&caps, Resolution::R1080).unwrap() > 1.0);
    }

    #[test]
    fn ltx_director_form_lists_every_tier_with_labels() {
        let form = crate::catalog::director_schema(&ltx_caps());
        let p = &form["properties"]["resolution"];
        assert_eq!(p["enum"], serde_json::json!(["480p", "720p", "768p", "1080p"]));
        assert_eq!(p["default"], "768p");
        assert!(p["x-fv-labels"]["1080p"].as_str().unwrap().contains("slower"), "{p}");
        assert!(p["x-fv-labels"].get("768p").is_none());
        // H3 without the 1080P tier: no labels, 768p default.
        let form = crate::catalog::director_schema(&ModelCaps::h3("h3", false));
        assert_eq!(form["properties"]["resolution"]["enum"], serde_json::json!(["768p"]));
        assert_eq!(form["properties"]["resolution"]["x-fv-labels"], serde_json::json!({}));
    }
}
