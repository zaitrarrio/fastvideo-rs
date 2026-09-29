//! Shared test harness: the MiniMax router over serve-kit and the fake
//! engine, a real local HTTP callback receiver, and golden-file helpers.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use base64::Engine as _;
use fastvideo_engine_service::{
    CancelOutcome, EngineBackend, EngineConfig, EngineEvent, EngineService, FakeBackend,
    FakeConfig, FakeModel, Mp4Mode, Priority, Readiness, Recipe,
};
use fastvideo_minimax::{callback_renderer, MiniMax, MiniMaxConfig};
use fastvideo_protocol::{
    ApiError, AudioPlan, Job, JobId, MediaKind, MediaProbe, ModelCaps, ProtocolId, Tier,
};
use fastvideo_serve_kit::artifacts::ArtifactMeta;
use fastvideo_serve_kit::auth::{Auth, AuthMode, KeyRing};
use fastvideo_serve_kit::callback::{CallbackSender, CallbackTransport, PostError, PostReply};
use fastvideo_serve_kit::ctx::{EngineGate, ServeConfig};
use fastvideo_serve_kit::events::{apply_event, FinishedOutput, JobEvent};
use fastvideo_serve_kit::ingest::{DefaultProber, Prober};
use fastvideo_serve_kit::net::TargetPolicy;
use fastvideo_serve_kit::{ServeCtx, UrlKey};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;
use url::Url;

pub const KEY: &str = "sk-test-a";
pub const KEY_B: &str = "sk-test-b";
pub const T: Duration = Duration::from_secs(30);

// ------------------------------------------------------------------ engine

/// H3 fakes: max (ref2va, 768 + 480 tiers), turbo, draft.
pub fn fake_models() -> Vec<FakeModel> {
    let mut max = FakeModel::h3_max();
    max.caps.canvas.short_edges = vec![768, 480];
    let mut turbo = FakeModel::h3_turbo();
    turbo.caps.canvas.short_edges = vec![768, 480];
    let draft = FakeModel {
        caps: ModelCaps::h3("fake-h3-draft", false).with_tier(Tier::Draft, "4step-vsa-480p-tiny-vae"),
        recipe: Recipe {
            name: "4step-vsa-480p-tiny-vae".into(),
            steps: Some(4),
            attention: "vsa".into(),
            vae: "tiny".into(),
            summary: "fake: below the quality gate".into(),
            ..Recipe::default()
        },
    };
    vec![max, turbo, draft]
}

/// `EngineGate` over the real `EngineService` with the fake backend. Jobs
/// run on a 64x32 canvas (the stored job keeps its real canvas) so debug
/// builds stay fast. With `hold` set, submitted jobs stay queued.
pub struct FakeGate {
    pub engine: EngineService,
    pub ctx: OnceLock<ServeCtx>,
    pub hold: AtomicBool,
    pub held: Mutex<Vec<JobId>>,
}

impl FakeGate {
    pub async fn start() -> Arc<Self> {
        let cfg = FakeConfig { models: fake_models(), mp4: Mp4Mode::Off, ..FakeConfig::default() };
        let backends: Vec<Box<dyn EngineBackend>> = vec![Box::new(FakeBackend::new(cfg))];
        let ecfg = EngineConfig {
            output_dir: std::env::temp_dir().join(format!("fvmm-eng-{}", fastvideo_serve_kit::random_token())),
            ..EngineConfig::default()
        };
        let engine = EngineService::start(ecfg, backends).expect("engine starts");
        assert_eq!(tokio::time::timeout(T, engine.wait_ready()).await.unwrap(), Readiness::Ready);
        Arc::new(Self { engine, ctx: OnceLock::new(), hold: AtomicBool::new(false), held: Mutex::new(Vec::new()) })
    }
}

#[async_trait::async_trait]
impl EngineGate for FakeGate {
    fn models(&self) -> Vec<ModelCaps> {
        self.engine.caps().models().cloned().collect()
    }
    fn alias(&self, name: &str) -> Option<String> {
        self.engine
            .caps()
            .tier_aliases()
            .into_iter()
            .find(|(a, _)| a == name)
            .map(|(_, m)| m.0)
    }
    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        if self.hold.load(Ordering::SeqCst) {
            self.held.lock().unwrap().push(job.id);
            return Ok(());
        }
        let mut r = job.resolved.clone();
        r.width = 64;
        r.height = 32;
        let mut h = self.engine.submit(job.id, r, Priority::Batch).await?;
        let ctx = self.ctx.get().expect("ctx set").clone();
        let (id, resolved) = (job.id, job.resolved.clone());
        tokio::spawn(async move {
            while let Some(ev) = h.events.recv().await {
                let ev = match ev {
                    EngineEvent::Queued { position } => JobEvent::Queued { position },
                    EngineEvent::Started => JobEvent::Started,
                    EngineEvent::Stage { name } => JobEvent::Stage { name: name.into() },
                    EngineEvent::Progress { step, total } => JobEvent::Progress { step, total },
                    EngineEvent::Log(l) => JobEvent::Log(l),
                    EngineEvent::Failed(e) => JobEvent::Failed(e),
                    EngineEvent::Cancelled => JobEvent::Cancelled,
                    EngineEvent::Finished(out) => {
                        let file = match out.mp4 {
                            Some(p) => p,
                            None => {
                                let dir = ctx.outputs_dir(id);
                                tokio::fs::create_dir_all(&dir).await.unwrap();
                                let p = dir.join("output.mp4");
                                tokio::fs::write(&p, b"\0\0\0\x18ftypisom-fake-mp4").await.unwrap();
                                p
                            }
                        };
                        let (w, hh) = resolved.output_size();
                        let audio = match resolved.audio {
                            AudioPlan::Native { rate, channels } => Some((rate, channels)),
                            _ => None,
                        };
                        let meta = ArtifactMeta { file_name: "output.mp4".into(), mime: "video/mp4".into(), width: w, height: hh, frames: resolved.num_frames, fps: resolved.fps, audio };
                        JobEvent::Finished(FinishedOutput { file, meta, metrics: out.metrics })
                    }
                };
                let _ = apply_event(&ctx, id, ev).await;
            }
        });
        Ok(())
    }
    async fn cancel(&self, id: JobId) -> bool {
        let mut held = self.held.lock().unwrap();
        if let Some(i) = held.iter().position(|j| *j == id) {
            held.remove(i);
            return true;
        }
        self.engine.cancel(id) != CancelOutcome::Unknown
    }
}

/// Images decode for real; video and audio report fixed facts (3.2 s video,
/// 6 s audio) so `usage` can be checked without ffprobe.
pub struct TestProber;

#[async_trait::async_trait]
impl Prober for TestProber {
    async fn probe(&self, path: &Path, kind: MediaKind, mime: &str) -> Result<MediaProbe, ApiError> {
        match kind {
            MediaKind::Image => DefaultProber { ffprobe: None }.probe(path, kind, mime).await,
            MediaKind::Video => Ok(MediaProbe { width: Some(1280), height: Some(720), duration_s: Some(3.2), fps: Some(24.0), audio_rate: None, frames: None }),
            MediaKind::Audio => Ok(MediaProbe { duration_s: Some(6.0), audio_rate: Some(48000), ..MediaProbe::default() }),
        }
    }
}

// ---------------------------------------------------------------- callbacks

/// A minimal HTTP/1.1 POST client over tokio TCP (no reqwest needed).
pub struct TcpTransport;

#[async_trait::async_trait]
impl CallbackTransport for TcpTransport {
    async fn post(&self, url: &Url, headers: &[(String, String)], body: bytes::Bytes, _t: &TargetPolicy) -> Result<PostReply, PostError> {
        let err = |e: std::io::Error| PostError::Transport(e.to_string());
        let host = url.host_str().unwrap_or("127.0.0.1").to_owned();
        let port = url.port_or_known_default().unwrap_or(80);
        let mut s = tokio::net::TcpStream::connect((host.as_str(), port)).await.map_err(err)?;
        let mut req = format!("POST {} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nContent-Length: {}\r\n", url.path(), body.len());
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        s.write_all(req.as_bytes()).await.map_err(err)?;
        s.write_all(&body).await.map_err(err)?;
        let mut out = Vec::new();
        s.read_to_end(&mut out).await.map_err(err)?;
        let text = String::from_utf8_lossy(&out).into_owned();
        let status: u16 = text.split(' ').nth(1).and_then(|c| c.parse().ok()).ok_or_else(|| PostError::Transport("bad response".into()))?;
        let body = text.split_once("\r\n\r\n").map(|x| x.1).unwrap_or_default().to_owned();
        Ok(PostReply { status, body: body.into() })
    }
}

/// A real local receiver: `/cb` echoes challenges, `/bad` answers a wrong
/// echo. Every POST body is recorded with its path.
#[derive(Clone, Default)]
pub struct Receiver {
    pub seen: Arc<Mutex<Vec<(String, serde_json::Value)>>>,
    pub port: u16,
}

impl Receiver {
    pub async fn start() -> Self {
        let seen: Arc<Mutex<Vec<(String, serde_json::Value)>>> = Arc::default();
        let (s1, s2) = (seen.clone(), seen.clone());
        let app = Router::new()
            .route("/cb", axum::routing::post(move |axum::Json(v): axum::Json<serde_json::Value>| {
                let s = s1.clone();
                async move {
                    s.lock().unwrap().push(("/cb".into(), v.clone()));
                    match v.get("challenge") {
                        Some(c) => axum::Json(serde_json::json!({ "challenge": c })),
                        None => axum::Json(serde_json::json!({})),
                    }
                }
            }))
            .route("/bad", axum::routing::post(move |axum::Json(v): axum::Json<serde_json::Value>| {
                let s = s2.clone();
                async move {
                    s.lock().unwrap().push(("/bad".into(), v));
                    axum::Json(serde_json::json!({ "challenge": "wrong" }))
                }
            }));
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        Self { seen, port }
    }
    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }
    /// Bodies received on `path`, in order.
    pub fn bodies(&self, path: &str) -> Vec<serde_json::Value> {
        self.seen.lock().unwrap().iter().filter(|(p, _)| p == path).map(|(_, v)| v.clone()).collect()
    }
    /// Waits until `path` has received a task body with a terminal status.
    pub async fn wait_terminal(&self, path: &str) -> Vec<serde_json::Value> {
        let deadline = tokio::time::Instant::now() + T;
        loop {
            let b = self.bodies(path);
            if b.iter().any(|v| matches!(v["task"]["status"].as_str(), Some("succeeded" | "failed" | "cancelled"))) {
                return b;
            }
            assert!(tokio::time::Instant::now() < deadline, "no terminal callback on {path}: {b:?}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

// ------------------------------------------------------------------ fixture

pub struct Fixture {
    pub ctx: ServeCtx,
    pub gate: Arc<FakeGate>,
    pub app: Router,
    pub rx: Receiver,
    pub dir: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub async fn fixture() -> Fixture {
    fixture_with(MiniMaxConfig::default()).await
}

pub async fn fixture_with(cfg: MiniMaxConfig) -> Fixture {
    let dir = std::env::temp_dir().join(format!("fvmm-{}", fastvideo_serve_kit::random_token()));
    let gate = FakeGate::start().await;
    let mm = Arc::new(MiniMax::new(cfg));
    let mut sender = CallbackSender::new(Some(Arc::new(TcpTransport)), None);
    sender.target.allow_private = true;
    let scfg = ServeConfig::new(Url::parse("http://fv.test").unwrap(), &dir);
    let ctx = ServeCtx::builder(scfg, gate.clone())
        .auth(Auth::new(AuthMode::Keys, KeyRing::from_plain([KEY, KEY_B])))
        .url_key(UrlKey::new("test-key"))
        .callbacks(Arc::new(sender))
        .renderer(ProtocolId::MiniMaxV2, callback_renderer(&mm))
        .prober(Arc::new(TestProber))
        .build()
        .await
        .unwrap();
    gate.ctx.set(ctx.clone()).ok().unwrap();
    let app = mm.router().merge(ctx.routes()).with_state(ctx.clone());
    let rx = Receiver::start().await;
    Fixture { ctx, gate, app, rx, dir }
}

pub async fn call(app: &Router, method: &str, uri: &str, key: Option<&str>, body: Option<serde_json::Value>) -> (StatusCode, HeaderMap, serde_json::Value) {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(k) = key {
        b = b.header("authorization", format!("Bearer {k}"));
    }
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let r = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = r.into_parts();
    let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
    let v = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (parts.status, parts.headers, v)
}

/// POST /v2/video_generation with the test key; returns the task id.
pub async fn create_ok(f: &Fixture, body: serde_json::Value) -> String {
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(body)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v.as_object().unwrap().len(), 1, "only task_id: {v}");
    v["task_id"].as_str().unwrap().to_owned()
}

pub async fn query(f: &Fixture, id: &str) -> serde_json::Value {
    let (s, _, v) = call(&f.app, "GET", &format!("/v2/query/video_generation/{id}"), Some(KEY), None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v["task"].clone()
}

/// Polls the query route until the task is terminal.
pub async fn wait_done(f: &Fixture, id: &str) -> serde_json::Value {
    let deadline = tokio::time::Instant::now() + T;
    loop {
        let t = query(f, id).await;
        if matches!(t["status"].as_str(), Some("succeeded" | "failed" | "cancelled")) {
            return t;
        }
        assert!(tokio::time::Instant::now() < deadline, "task {id} did not finish: {t}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

pub fn t2v(model: &str, duration: u64) -> serde_json::Value {
    serde_json::json!({"model": model, "content": [{"type": "text", "text": "a red fox in snow"}],
        "resolution": "768P", "duration": duration, "ratio": "16:9"})
}

/// A `w`x`h` PNG as a data URI.
pub fn png_uri(w: u32, h: u32) -> String {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([200, 30, 30]));
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img).write_to(&mut buf, image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf.into_inner()))
}

pub fn mp4_uri() -> String {
    format!("data:video/mp4;base64,{}", base64::engine::general_purpose::STANDARD.encode(b"\0\0\0\x18ftypisom\0\0\0\0isomfake"))
}

pub fn wav_uri() -> String {
    format!("data:audio/wav;base64,{}", base64::engine::general_purpose::STANDARD.encode(b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0"))
}

// ------------------------------------------------------------------- golden

pub fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name)
}

/// Compares `actual` to `tests/golden/<name>`; `FV_UPDATE_GOLDEN=1` rewrites it.
pub fn assert_golden(name: &str, actual: &serde_json::Value) {
    let p = golden_path(name);
    if std::env::var("FV_UPDATE_GOLDEN").is_ok_and(|v| v == "1") {
        std::fs::write(&p, serde_json::to_string_pretty(actual).unwrap() + "\n").unwrap();
    }
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap();
    assert_eq!(actual, &want, "golden {name}:\n{}", serde_json::to_string_pretty(actual).unwrap());
}

pub fn load_golden(name: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(golden_path(name)).unwrap()).unwrap()
}
