//! Shared fixture for the fal queue tests: the real `EngineService` over the
//! fake backend, a `ServeCtx` with fal keys, a webhook signer and a
//! recording webhook transport, and the fal router.
#![allow(dead_code)]

use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use fastvideo_engine_service::{
    EngineBackend, EngineConfig, EngineEvent, EngineService, FakeBackend, FakeConfig, FakeModel,
    FakeTiming, Mp4Mode, Priority, Readiness,
};
use fastvideo_fal::{router, FalConfig, FalWebhook};
use fastvideo_protocol::{ApiError, AudioPlan, Job, JobId, JobMetrics, ModelCaps, ProtocolId};
use fastvideo_serve_kit::artifacts::ArtifactMeta;
use fastvideo_serve_kit::callback::{CallbackTransport, PostError, PostReply};
use fastvideo_serve_kit::events::{apply_event, FinishedOutput, JobEvent};
use fastvideo_serve_kit::net::TargetPolicy;
use fastvideo_serve_kit::{
    Auth, AuthMode, CallbackSender, EngineGate, KeyRing, ServeConfig, ServeCtx, UrlKey,
    WebhookSigner,
};
use tower::ServiceExt;
use url::Url;

pub const KEY: &str = "fal-test-key";
pub const OTHER_KEY: &str = "fal-other-key";
pub const T: Duration = Duration::from_secs(60);

/// The ffmpeg the fake engine writes MP4s with: `FV_FFMPEG`, else `ffmpeg`
/// on PATH, else none (placeholder outputs).
pub fn ffmpeg() -> Option<PathBuf> {
    let p = std::env::var_os("FV_FFMPEG").map(PathBuf::from).unwrap_or_else(|| "ffmpeg".into());
    std::process::Command::new(&p)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()
        .filter(|s| s.success())
        .map(|_| p)
}

/// Which engine backs the fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// The real `EngineService` over `FakeBackend` (renders every frame; a
    /// 1344x768x124 job takes seconds in a debug build).
    Fake,
    /// An instant simulation with the same caps: one job at a time, 4 steps,
    /// queue positions, cancel at the next step, a placeholder output.
    Sim,
}

/// Simulated engine state.
pub struct Sim {
    step: Duration,
    exec: tokio::sync::Mutex<()>,
    waiting: AtomicU32,
    cancels: Mutex<HashMap<JobId, Arc<AtomicBool>>>,
}

pub enum Backend {
    Fake(EngineService),
    Sim(Sim),
}

/// `EngineGate` over the engine (what the fv-serve binary does, WP-10):
/// submits resolved jobs and pumps their events into the store.
pub struct Gate {
    pub backend: Backend,
    pub caps: Vec<ModelCaps>,
    pub ctx: OnceLock<ServeCtx>,
    /// Canned admission error (e.g. `Loading`).
    pub refuse: Mutex<Option<ApiError>>,
}

async fn placeholder(ctx: &ServeCtx, id: JobId) -> PathBuf {
    let dir = ctx.outputs_dir(id);
    tokio::fs::create_dir_all(&dir).await.unwrap();
    let p = dir.join("output.mp4");
    tokio::fs::write(&p, b"\0\0\0\x18ftypisom\0\0\x02\0isomiso2avc1mp41").await.unwrap();
    p
}

fn finished(job: &Job, file: PathBuf, metrics: JobMetrics) -> JobEvent {
    let r = &job.resolved;
    let (w, h) = r.output_size();
    let audio = match r.audio {
        AudioPlan::Native { rate, channels } => Some((rate, channels)),
        _ => None,
    };
    let meta = ArtifactMeta {
        file_name: fastvideo_fal::output_file_name(job),
        mime: "video/mp4".into(),
        width: w,
        height: h,
        frames: r.num_frames,
        fps: r.fps,
        audio,
    };
    JobEvent::Finished(FinishedOutput { file, meta, metrics })
}

async fn run_sim(gate: Arc<Gate>, ctx: ServeCtx, job: Job, position: u32, flag: Arc<AtomicBool>) {
    let Backend::Sim(sim) = &gate.backend else { unreachable!() };
    let id = job.id;
    let ev = |e: JobEvent| {
        let ctx = ctx.clone();
        async move {
            let _ = apply_event(&ctx, id, e).await;
        }
    };
    ev(JobEvent::Queued { position }).await;
    let _exec = sim.exec.lock().await;
    sim.waiting.fetch_sub(1, Ordering::SeqCst);
    if flag.load(Ordering::SeqCst) {
        return ev(JobEvent::Cancelled).await;
    }
    ev(JobEvent::Started).await;
    ev(JobEvent::Stage { name: "text_encode".into() }).await;
    ev(JobEvent::Stage { name: "denoise".into() }).await;
    let t0 = tokio::time::Instant::now();
    for s in 1..=4 {
        tokio::time::sleep(sim.step).await;
        if flag.load(Ordering::SeqCst) {
            return ev(JobEvent::Cancelled).await;
        }
        if job.resolved.prompt.contains("[fake:fail]") {
            return ev(JobEvent::Failed(ApiError::engine_failed(format!("injected failure at step {s}/4")))).await;
        }
        ev(JobEvent::Progress { step: s, total: 4 }).await;
    }
    let metrics = JobMetrics { inference_s: Some(t0.elapsed().as_secs_f64()), ..Default::default() };
    let file = placeholder(&ctx, id).await;
    ev(finished(&job, file, metrics)).await;
    sim.cancels.lock().unwrap().remove(&id);
}

pub struct GateRef(pub Arc<Gate>);

#[async_trait::async_trait]
impl EngineGate for GateRef {
    fn models(&self) -> Vec<ModelCaps> {
        self.0.caps.clone()
    }
    fn alias(&self, name: &str) -> Option<String> {
        match &self.0.backend {
            Backend::Fake(e) => e.caps().resolve(name).map(|c| c.id.0.clone()),
            Backend::Sim(_) => None,
        }
    }
    fn admit(&self) -> Result<(), ApiError> {
        match self.0.refuse.lock().unwrap().clone() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        let ctx = self.0.ctx.get().expect("ctx set").clone();
        let job = job.clone();
        let engine = match &self.0.backend {
            Backend::Sim(sim) => {
                let flag = Arc::new(AtomicBool::new(false));
                sim.cancels.lock().unwrap().insert(job.id, flag.clone());
                let position = sim.waiting.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(run_sim(self.0.clone(), ctx, job, position, flag));
                return Ok(());
            }
            Backend::Fake(e) => e,
        };
        let mut h = engine.submit(job.id, job.resolved.clone(), Priority::Batch).await?;
        tokio::spawn(async move {
            while let Some(ev) = h.events.recv().await {
                let terminal = ev.is_terminal();
                let jev = match ev {
                    EngineEvent::Queued { position } => JobEvent::Queued { position },
                    EngineEvent::Started => JobEvent::Started,
                    EngineEvent::Stage { name } => JobEvent::Stage { name: name.to_owned() },
                    EngineEvent::Progress { step, total } => JobEvent::Progress { step, total },
                    EngineEvent::Log(l) => JobEvent::Log(l),
                    EngineEvent::Failed(e) => JobEvent::Failed(e),
                    EngineEvent::Cancelled => JobEvent::Cancelled,
                    EngineEvent::Finished(out) => {
                        // No ffmpeg: a placeholder so the job still has an artifact.
                        let file = match out.mp4 {
                            Some(p) => p,
                            None => placeholder(&ctx, job.id).await,
                        };
                        finished(&job, file, out.metrics)
                    }
                };
                let _ = apply_event(&ctx, job.id, jev).await;
                if terminal {
                    break;
                }
            }
        });
        Ok(())
    }
    async fn cancel(&self, id: JobId) -> bool {
        match &self.0.backend {
            Backend::Fake(e) => !matches!(e.cancel(id), fastvideo_engine_service::CancelOutcome::Unknown),
            Backend::Sim(sim) => match sim.cancels.lock().unwrap().get(&id) {
                Some(f) => {
                    f.store(true, Ordering::SeqCst);
                    true
                }
                None => false,
            },
        }
    }
}

/// One recorded POST: target, headers, body.
pub type Post = (Url, Vec<(String, String)>, bytes::Bytes);

/// Records webhook POSTs.
#[derive(Default)]
pub struct Hooks(pub Mutex<Vec<Post>>);

#[async_trait::async_trait]
impl CallbackTransport for Hooks {
    async fn post(&self, u: &Url, h: &[(String, String)], body: bytes::Bytes, _t: &TargetPolicy) -> Result<PostReply, PostError> {
        self.0.lock().unwrap().push((u.clone(), h.to_vec(), body));
        Ok(PostReply { status: 200, body: bytes::Bytes::from_static(b"{}") })
    }
}

pub struct Fixture {
    pub ctx: ServeCtx,
    pub gate: Arc<Gate>,
    pub hooks: Arc<Hooks>,
    pub signer: WebhookSigner,
    pub app: Router,
    pub dir: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub struct Opts {
    pub engine: Engine,
    pub public_base: String,
    /// Time per denoise step of the fake engine.
    pub step: Duration,
    pub models: Vec<FakeModel>,
    pub mp4: bool,
    pub queue_max: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            engine: Engine::Sim,
            public_base: "https://fal.fv.test".into(),
            step: Duration::from_millis(1),
            models: vec![FakeModel::h3_max(), FakeModel::h3_turbo()],
            mp4: false,
            queue_max: 32,
        }
    }
}

pub async fn fixture(o: Opts) -> Fixture {
    let dir = std::env::temp_dir().join(format!("fv-fal-{}", uuid::Uuid::new_v4().simple()));
    let mut fake = FakeConfig {
        models: o.models.clone(),
        timing: FakeTiming { step: o.step, ..FakeTiming::default() },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    };
    if o.mp4 {
        if let Some(f) = ffmpeg() {
            fake.mp4 = Mp4Mode::Auto;
            fake.ffmpeg = f;
        }
    }
    let caps: Vec<ModelCaps> = o.models.iter().map(|m| m.caps.clone()).collect();
    let backend = match o.engine {
        Engine::Fake => {
            let cfg = EngineConfig { queue_max: o.queue_max, output_dir: dir.join("engine"), ..EngineConfig::default() };
            let backends: Vec<Box<dyn EngineBackend>> = vec![Box::new(FakeBackend::new(fake))];
            let engine = EngineService::start(cfg, backends).expect("engine starts");
            assert_eq!(tokio::time::timeout(T, engine.wait_ready()).await.unwrap(), Readiness::Ready);
            Backend::Fake(engine)
        }
        Engine::Sim => Backend::Sim(Sim {
            step: o.step,
            exec: tokio::sync::Mutex::new(()),
            waiting: AtomicU32::new(0),
            cancels: Mutex::new(HashMap::new()),
        }),
    };
    let gate = Arc::new(Gate { backend, caps, ctx: OnceLock::new(), refuse: Mutex::new(None) });
    let hooks = Arc::new(Hooks::default());
    let signer = WebhookSigner::from_seed([7u8; 32], "fv-test-user");
    let cb = Arc::new(CallbackSender::new(Some(hooks.clone()), Some(signer.clone())));
    let scfg = ServeConfig::new(Url::parse(&o.public_base).unwrap(), &dir);
    let ctx = ServeCtx::builder(scfg, Arc::new(GateRef(gate.clone())))
        .auth(Auth::new(AuthMode::Keys, KeyRing::from_plain([KEY, OTHER_KEY])))
        .url_key(UrlKey::new("fal-test-url-key"))
        .callbacks(cb)
        .renderer(ProtocolId::Fal, Arc::new(FalWebhook::default()))
        .build()
        .await
        .unwrap();
    gate.ctx.set(ctx.clone()).ok().unwrap();
    let app = router(ctx.clone(), FalConfig::default()).merge(ctx.routes().with_state(()));
    Fixture { ctx, gate, hooks, signer, app, dir }
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
    pub fn header(&self, k: &str) -> Option<&str> {
        self.headers.get(k).and_then(|v| v.to_str().ok())
    }
}

pub async fn send(app: &Router, req: Request<Body>) -> Resp {
    let r = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = r.into_parts();
    let body = axum::body::to_bytes(body, 1 << 30).await.unwrap();
    Resp { status: parts.status, headers: parts.headers, body }
}

pub fn req(method: &str, uri: &str, key: Option<&str>, body: Option<serde_json::Value>) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(k) = key {
        b = b.header("authorization", format!("Key {k}"));
    }
    match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

pub async fn call(app: &Router, method: &str, uri: &str, body: Option<serde_json::Value>) -> Resp {
    send(app, req(method, uri, Some(KEY), body)).await
}

/// The path (with query) of an absolute URL we handed out.
pub fn path_of(u: &str) -> String {
    let u = Url::parse(u).unwrap();
    match u.query() {
        Some(q) => format!("{}?{q}", u.path()),
        None => u.path().to_owned(),
    }
}

/// Polls status until `COMPLETED`.
pub async fn wait_completed(app: &Router, status_path: &str) -> serde_json::Value {
    let deadline = tokio::time::Instant::now() + T;
    loop {
        let r = call(app, "GET", status_path, None).await;
        assert_eq!(r.status, StatusCode::OK, "{:?}", r.json());
        let v = r.json();
        if v["status"] == "COMPLETED" {
            return v;
        }
        assert!(tokio::time::Instant::now() < deadline, "timed out: {v}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
