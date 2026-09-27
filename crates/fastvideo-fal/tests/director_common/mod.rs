//! Shared fixture for the director tests: the real `EngineService` over the
//! fake backend, a `ServeCtx` with fal keys, the director service on a
//! loopback WebRTC host, and the fal router with the director mounted.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use fastvideo_engine_service::{
    ClipBuild, ClipSession, EngineBackend, EngineConfig, EngineService, FakeBackend, FakeConfig, FakeModel, FakeTiming,
    Mp4Mode, Readiness,
};
use fastvideo_fal::director::{ChunkBuild, ChunkOutput, DirectorClips, DirectorConfig, DirectorEngine, DirectorService};
use fastvideo_fal::{router_with, FalApp, FalConfig};
use fastvideo_media::video::EncoderBackend;
use fastvideo_protocol::{ApiError, Job, JobId, ModelCaps, SessionSpec, Tier};
use fastvideo_serve_kit::{Auth, AuthMode, EngineGate, KeyRing, ServeConfig, ServeCtx, UrlKey};
use fastvideo_webrtc::host::{HostConfig, RtcHost};
use tower::ServiceExt;
use url::Url;

pub const KEY: &str = "fal-test-key";
pub const T: Duration = Duration::from_secs(60);

/// The A/V model: fake H3 max with the 480p tier added (E3), so tests run
/// at 832x480.
pub fn h3_av() -> FakeModel {
    let mut m = FakeModel::h3_max();
    m.caps.canvas.short_edges = vec![768, 480];
    m
}

/// A video-only 24 fps clip model on the H3 grid.
pub fn h3_silent() -> FakeModel {
    let mut m = FakeModel::h3_max();
    m.caps.id = fastvideo_protocol::ModelId::new("fake-h3-silent");
    m.caps.served_names = vec!["fake-h3-silent".into()];
    m.caps.audio = None;
    m.caps.tier = None;
    m.caps.recipe = None;
    m.caps.canvas.short_edges = vec![768, 480];
    m
}

/// The silent model's fal app (`fv/h3-silent/director`).
pub fn silent_app() -> FalApp {
    FalApp { id: "fv/h3-silent".into(), model: "fake-h3-silent".into(), tier: None }
}

/// `DirectorEngine` over `EngineService` (what fv-serve does).
pub struct EngineDirector(pub EngineService);

/// The clip session, released by `close` (the build future may outlive it).
pub struct Clips {
    caps: ModelCaps,
    spec: SessionSpec,
    session: tokio::sync::RwLock<Option<ClipSession>>,
}

#[async_trait::async_trait]
impl DirectorClips for Clips {
    fn caps(&self) -> &ModelCaps {
        &self.caps
    }
    fn spec(&self) -> &SessionSpec {
        &self.spec
    }
    async fn close(&self) {
        if let Some(s) = self.session.write().await.take() {
            s.close();
        }
    }
    async fn build(&self, c: ChunkBuild) -> Result<ChunkOutput, ApiError> {
        let b = ClipBuild {
            prompt: c.prompt,
            negative_prompt: None,
            seed: c.seed,
            seconds: Some(c.seconds),
            frames: None,
            canvas: c.canvas,
            first_frame: c.first_frame,
            last_frame: c.last_frame,
        };
        let job = {
            let g = self.session.read().await;
            let s = g.as_ref().ok_or_else(|| ApiError::internal("the clip session is closed"))?;
            s.build(b).await?
        };
        let out = job.wait().await?;
        Ok(ChunkOutput { frames: out.frames.unwrap_or_default(), audio: out.audio })
    }
}

#[async_trait::async_trait]
impl DirectorEngine for EngineDirector {
    async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn DirectorClips>, ApiError> {
        let s = self.0.open_clip_session(spec).await?;
        Ok(Arc::new(Clips { caps: s.caps().clone(), spec: s.spec().clone(), session: tokio::sync::RwLock::new(Some(s)) }))
    }
}

/// Model table + aliases; batch submit is not used here.
struct Gate(EngineService);

#[async_trait::async_trait]
impl EngineGate for Gate {
    fn models(&self) -> Vec<ModelCaps> {
        self.0.caps().models().cloned().collect()
    }
    fn alias(&self, name: &str) -> Option<String> {
        self.0.caps().resolve(name).map(|c| c.id.0.clone())
    }
    async fn submit(&self, _job: &Job) -> Result<(), ApiError> {
        Err(ApiError::internal("batch jobs are not part of the director tests"))
    }
    async fn cancel(&self, _id: JobId) -> bool {
        false
    }
}

/// An ffmpeg with the given encoder (`FV_FFMPEG`, else `ffmpeg` on PATH).
pub fn ffmpeg_with(encoder: &str) -> Option<PathBuf> {
    let p = std::env::var_os("FV_FFMPEG").map(PathBuf::from).unwrap_or_else(|| "ffmpeg".into());
    let out = std::process::Command::new(&p).args(["-hide_banner", "-encoders"]).output().ok()?;
    String::from_utf8_lossy(&out.stdout).contains(encoder).then_some(p)
}

/// The H.264 backend the tests can run: x264 through ffmpeg when present,
/// else OpenH264 when compiled in.
pub fn h264_backend() -> Option<EncoderBackend> {
    if ffmpeg_with("libx264").is_some() {
        return Some(EncoderBackend::CpuTestX264);
    }
    EncoderBackend::OpenH264.compiled().then_some(EncoderBackend::OpenH264)
}

pub struct Fixture {
    pub ctx: ServeCtx,
    pub engine: EngineService,
    pub svc: Arc<DirectorService>,
    pub app: Router,
    pub dir: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub struct Opts {
    pub step: Duration,
    pub heartbeat_timeout: Duration,
    pub chunk_seconds: f64,
    pub max_session_seconds: Option<u64>,
    pub h264: EncoderBackend,
    /// Bind the WebRTC host on the default interface (browsers do not use
    /// loopback candidates) instead of 127.0.0.1.
    pub lan: bool,
    pub public_base: String,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            step: Duration::from_millis(1),
            heartbeat_timeout: Duration::from_secs(15),
            chunk_seconds: 5.0,
            max_session_seconds: None,
            h264: EncoderBackend::CpuTestX264,
            lan: false,
            public_base: "https://fal.fv.test".into(),
        }
    }
}

pub async fn fixture(o: Opts) -> Fixture {
    let dir = std::env::temp_dir().join(format!("fv-director-{}", uuid::Uuid::new_v4().simple()));
    let fake = FakeConfig {
        models: vec![h3_av(), h3_silent()],
        timing: FakeTiming { step: o.step, ..FakeTiming::default() },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    };
    let cfg = EngineConfig { output_dir: dir.join("engine"), ..EngineConfig::default() };
    let backends: Vec<Box<dyn EngineBackend>> = vec![Box::new(FakeBackend::new(fake))];
    let engine = EngineService::start(cfg, backends).expect("engine starts");
    assert_eq!(tokio::time::timeout(T, engine.wait_ready()).await.unwrap(), Readiness::Ready);

    let scfg = ServeConfig::new(Url::parse(&o.public_base).unwrap(), &dir);
    let ctx = ServeCtx::builder(scfg, Arc::new(Gate(engine.clone())))
        .auth(Auth::new(AuthMode::Keys, KeyRing::from_plain([KEY])))
        .url_key(UrlKey::new("director-test-url-key"))
        .build()
        .await
        .unwrap();

    let hcfg = if o.lan {
        HostConfig { udp_bind: Some("0.0.0.0:0".parse().unwrap()), ice_servers: Vec::new(), ..HostConfig::default() }
    } else {
        HostConfig::loopback(true, false)
    };
    let host = RtcHost::bind(hcfg).await.unwrap();
    let dcfg = DirectorConfig {
        apps: vec![FalApp::h3(Tier::Max), silent_app()],
        ice_servers: Vec::new(),
        chunk_seconds: o.chunk_seconds,
        max_session_seconds: o.max_session_seconds,
        heartbeat_timeout: o.heartbeat_timeout,
        h264: o.h264,
        work_dir: dir.join("director"),
        ..DirectorConfig::default()
    };
    let svc = DirectorService::new(dcfg, host, Arc::new(EngineDirector(engine.clone())));
    let fal = FalConfig { apps: vec![FalApp::h3(Tier::Max), silent_app()], ..FalConfig::default() };
    let app = router_with(ctx.clone(), fal, fastvideo_fal::director::routes(svc.clone())).merge(ctx.routes().with_state(()));
    Fixture { ctx, engine, svc, app, dir }
}

pub struct Resp {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: bytes::Bytes,
}

impl Resp {
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or(serde_json::Value::Null)
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

pub async fn send(app: &Router, req: Request<Body>) -> Resp {
    let r = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = r.into_parts();
    let body = axum::body::to_bytes(body, 1 << 30).await.unwrap();
    Resp { status: parts.status, headers: parts.headers, body }
}

pub fn req(method: &str, uri: &str, key: Option<&str>, body: Option<String>) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(k) = key {
        b = b.header("authorization", format!("Key {k}"));
    }
    match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v)).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

pub async fn post(app: &Router, uri: &str, body: serde_json::Value) -> Resp {
    send(app, req("POST", uri, Some(KEY), Some(body.to_string()))).await
}
