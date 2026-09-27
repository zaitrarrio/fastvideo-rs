//! Generic handler, events, SSE and callback wiring over a toy API and a fake
//! engine.

use std::sync::{Mutex, OnceLock};

use axum::http::Request;
use axum::Router;
use fastvideo_protocol::{
    Anchor, CallbackSpec, CanvasSpec, ErrorKind, JobStatus, Keyframe, MediaRef, ModelCaps,
    ProtocolId, SseEvent, SseFollow, SseSpec, Task, UrlSigner, ViewCtx,
};
use tower::ServiceExt;
use url::Url;

use super::*;
use crate::artifacts::{ArtifactMeta, UrlKey};
use crate::auth::{Auth, AuthMode, KeyRing};
use crate::callback::{CallbackRender, CallbackSender, CallbackTransport, PostError, PostReply};
use crate::ctx::{EngineGate, ServeConfig};
use crate::events::{apply_event, cancel_job, FinishedOutput, JobEvent};
use crate::net::TargetPolicy;

// ------------------------------------------------------------- toy protocol

struct Toy;
impl BatchProtocol for Toy {
    fn id(&self) -> ProtocolId {
        ProtocolId::MiniMaxV2
    }
    fn new_external_id(&self, job: JobId) -> String {
        format!("toy_{}", job.0.simple())
    }
    fn render_error(&self, err: &ApiError, cx: &ErrorCtx) -> HttpReply {
        HttpReply::json(
            err.http_status(),
            serde_json::json!({"error": err.kind.code(), "message": err.message, "param": err.param, "request_id": cx.request_id}),
        )
    }
}

#[derive(serde::Deserialize)]
struct ToyBody {
    prompt: String,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    callback_url: Option<String>,
}

struct T2v;
impl SubmitEndpoint for T2v {
    type Body = ToyBody;
    fn normalize(&self, b: ToyBody, _cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError> {
        let mut r = GenerationRequest::text(ProtocolId::MiniMaxV2, "MiniMax-H3", b.prompt);
        if let Some(i) = b.image {
            r.task = Task::I2V;
            r.canvas = CanvasSpec::FollowImage { short_edge: 768 };
            r.keyframes.push(Keyframe { at: Anchor::First, image: MediaRef::parse(&i, "image")? });
        }
        if let Some(u) = b.callback_url {
            r.callback = Some(CallbackSpec::MiniMax { url: Url::parse(&u).map_err(|e| ApiError::invalid(e.to_string()))? });
        }
        Ok(r)
    }
    fn submit_reply(&self, job: &Job, _cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, serde_json::json!({"task_id": job.external_id}))
    }
}

struct View;
impl JobView for View {
    fn status_reply(&self, job: &Job, _cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, serde_json::json!({"status": job.status().as_str(), "progress": job.progress}))
    }
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        match job.artifacts.first() {
            Some(a) => HttpReply::json(200, serde_json::json!({"url": cx.urls.url_for(a, Duration::from_secs(60)).as_str()}))
                .with_header("x-toy", "1"),
            None => HttpReply::json(400, serde_json::json!({"status": job.status().as_str()})),
        }
    }
}

impl CallbackRender for View {
    fn callback_body(&self, job: &Job, _cx: &ViewCtx) -> Option<serde_json::Value> {
        Some(serde_json::json!({"task": {"task_id": job.external_id, "status": job.status().as_str()}}))
    }
}

// ------------------------------------------------------------- fake engine

#[derive(Default)]
struct Engine {
    submitted: Mutex<Vec<JobId>>,
    cancelled: Mutex<Vec<JobId>>,
    loading: bool,
    /// Run jobs to completion on submit.
    auto: OnceLock<ServeCtx>,
}

#[async_trait::async_trait]
impl EngineGate for Engine {
    fn models(&self) -> Vec<ModelCaps> {
        vec![ModelCaps::h3("fasth3", false)]
    }
    fn alias(&self, name: &str) -> Option<String> {
        (name == "MiniMax-H3").then(|| "fasth3".to_owned())
    }
    fn admit(&self) -> Result<(), ApiError> {
        if self.loading {
            Err(ApiError::loading("models are loading"))
        } else {
            Ok(())
        }
    }
    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        self.submitted.lock().unwrap().push(job.id);
        if let Some(ctx) = self.auto.get().cloned() {
            let id = job.id;
            tokio::spawn(async move { run_fake(&ctx, id).await });
        }
        Ok(())
    }
    async fn cancel(&self, id: JobId) -> bool {
        self.cancelled.lock().unwrap().push(id);
        true
    }
}

async fn run_fake(ctx: &ServeCtx, id: JobId) -> Job {
    apply_event(ctx, id, JobEvent::Queued { position: 0 }).await.unwrap();
    apply_event(ctx, id, JobEvent::Started).await.unwrap();
    apply_event(ctx, id, JobEvent::Progress { step: 2, total: 4 }).await.unwrap();
    let dir = ctx.outputs_dir(id);
    tokio::fs::create_dir_all(&dir).await.unwrap();
    let file = dir.join("out.mp4");
    tokio::fs::write(&file, b"\0\0\0\x18ftypisom-fake-mp4").await.unwrap();
    let meta = ArtifactMeta { file_name: "out.mp4".into(), mime: "video/mp4".into(), width: 1344, height: 768, frames: 124, fps: 24, audio: Some((32000, 2)) };
    apply_event(ctx, id, JobEvent::Finished(FinishedOutput { file, meta, metrics: Default::default() })).await.unwrap()
}

/// Records callback POSTs and echoes challenges.
#[derive(Default)]
struct Rx(Mutex<Vec<serde_json::Value>>);
#[async_trait::async_trait]
impl CallbackTransport for Rx {
    async fn post(&self, _u: &Url, _h: &[(String, String)], body: bytes::Bytes, _t: &TargetPolicy) -> Result<PostReply, PostError> {
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        self.0.lock().unwrap().push(v.clone());
        let out = match v.get("challenge") {
            Some(c) => serde_json::json!({"challenge": c}),
            None => serde_json::json!({}),
        };
        Ok(PostReply { status: 200, body: serde_json::to_vec(&out).unwrap().into() })
    }
}

struct Fixture {
    ctx: ServeCtx,
    engine: Arc<Engine>,
    rx: Arc<Rx>,
    app: Router,
    dir: std::path::PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

async fn fixture(engine: Engine) -> Fixture {
    let dir = std::env::temp_dir().join(format!("fvkit-h-{}", crate::random_token()));
    let engine = Arc::new(engine);
    let rx = Arc::new(Rx::default());
    let cb = Arc::new(CallbackSender::new(Some(rx.clone()), None));
    let cfg = ServeConfig::new(Url::parse("http://fv.test").unwrap(), &dir);
    let ctx = ServeCtx::builder(cfg, engine.clone())
        .auth(Auth::new(AuthMode::Keys, KeyRing::from_plain(["sk-a", "sk-b"])))
        .url_key(UrlKey::new("k"))
        .callbacks(cb)
        .renderer(ProtocolId::MiniMaxV2, Arc::new(View))
        .build()
        .await
        .unwrap();
    let (p, v) = (Arc::new(Toy), Arc::new(View));
    let app = Router::new()
        .route("/submit", submit(p.clone(), Arc::new(T2v), v.clone(), SubmitOpts::new(IngestPolicy::default())))
        .route("/sync", submit(p.clone(), Arc::new(T2v), v.clone(), SubmitOpts::new(IngestPolicy::default()).sync(Duration::from_secs(5))))
        .route("/jobs/{id}", status(p.clone(), v.clone(), "id"))
        .route("/jobs/{id}/result", result(p, v, "id"))
        .merge(ctx.routes())
        .with_state(ctx.clone());
    Fixture { ctx, engine, rx, app, dir }
}

async fn call(app: &Router, method: &str, uri: &str, key: Option<&str>, body: Option<serde_json::Value>) -> (StatusCode, HeaderMap, serde_json::Value) {
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

async fn settle(f: &Fixture) {
    for _ in 0..200 {
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        if f.rx.0.lock().unwrap().iter().any(|v| v["task"]["status"] == "succeeded") {
            return;
        }
    }
}

// ------------------------------------------------------------- tests

#[tokio::test]
async fn submit_auth_store_engine() {
    let f = fixture(Engine::default()).await;
    let (s, _, v) = call(&f.app, "POST", "/submit", None, Some(serde_json::json!({"prompt": "a"}))).await;
    assert_eq!((s, v["error"].as_str()), (StatusCode::UNAUTHORIZED, Some("unauthorized")));
    assert_eq!(v["request_id"].as_str().unwrap().len(), 32);
    assert!(f.engine.submitted.lock().unwrap().is_empty());

    let (s, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a cat"}))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let ext = v["task_id"].as_str().unwrap().to_owned();
    let job = f.ctx.jobs().by_external(ProtocolId::MiniMaxV2, &ext).await.unwrap();
    assert_eq!(f.engine.submitted.lock().unwrap().as_slice(), &[job.id]);
    assert_eq!(job.resolved.model.0, "fasth3");
    assert_eq!((job.resolved.width, job.resolved.height, job.resolved.num_frames), (1344, 768, 124));
    assert_eq!(job.request_echo["prompt"], "a cat");
    assert!(job.owner.is_some());
    assert_eq!(job.expires_at - job.created_at, time::Duration::days(7), "MiniMax retention");

    // Status: owner only.
    let (s, _, v) = call(&f.app, "GET", &format!("/jobs/{ext}"), Some("sk-a"), None).await;
    assert_eq!((s, v["status"].as_str()), (StatusCode::OK, Some("queued")));
    let (s, _, _) = call(&f.app, "GET", &format!("/jobs/{ext}"), Some("sk-b"), None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, _, v) = call(&f.app, "GET", "/jobs/nope", Some("sk-a"), None).await;
    assert_eq!((s, v["error"].as_str()), (StatusCode::NOT_FOUND, Some("not_found")));
}

#[tokio::test]
async fn submit_errors_render_through_adapter() {
    let f = fixture(Engine::default()).await;
    let bad = |b: &str| Request::post("/submit").header("authorization", "Bearer sk-a").body(Body::from(b.to_owned())).unwrap();
    let r = f.app.clone().oneshot(bad("{not json")).await.unwrap();
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    let r = f.app.clone().oneshot(bad(r#"{"no_prompt":1}"#)).await.unwrap();
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    // Unsupported media from ingestion keeps the param.
    let (s, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a", "image": "data:image/png;base64,AAAA"}))).await;
    assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE, "{v}");
    assert_eq!(v["param"], "keyframes[0]");
    assert!(f.ctx.jobs().list(Default::default()).await.items.is_empty(), "no job on refusal");
    assert!(!f.dir.join("inputs").exists() || std::fs::read_dir(f.dir.join("inputs")).unwrap().next().is_none(), "inputs cleaned");
    // Provider files are a gap.
    let (_, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a", "image": "mm_file://x"}))).await;
    assert_eq!(v["error"], "unsupported");

    let f = fixture(Engine { loading: true, ..Default::default() }).await;
    let (s, h, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a"}))).await;
    assert_eq!((s, v["error"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("loading")));
    assert_eq!(h["retry-after"], "1");
}

#[tokio::test]
async fn i2v_data_uri_is_staged_and_followed() {
    let f = fixture(Engine::default()).await;
    let img = crate::ingest::tests::png(64, 36);
    let uri = crate::ingest::tests::data_uri("image/png", &img);
    let (s, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a", "image": uri}))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let j = f.ctx.jobs().by_external(ProtocolId::MiniMaxV2, v["task_id"].as_str().unwrap()).await.unwrap();
    assert_eq!(j.resolved.keyframes.len(), 1);
    assert!(j.resolved.keyframes[0].1.starts_with(f.ctx.inputs_dir(j.id)));
    assert!(j.resolved.width > j.resolved.height, "canvas follows the 16:9 image");
}

#[tokio::test]
async fn events_artifacts_urls_and_callbacks() {
    let f = fixture(Engine::default()).await;
    let (_, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a", "callback_url": "https://hooks.example.com/cb"}))).await;
    let ext = v["task_id"].as_str().unwrap().to_owned();
    let id = f.ctx.jobs().by_external(ProtocolId::MiniMaxV2, &ext).await.unwrap().id;
    let (s, _, _) = call(&f.app, "GET", &format!("/jobs/{ext}/result"), Some("sk-a"), None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "not finished");

    let j = run_fake(&f.ctx, id).await;
    assert_eq!(j.status(), JobStatus::Succeeded);
    assert_eq!(j.artifacts[0].bytes, 21);
    assert!(j.logs.is_empty());

    let (s, h, v) = call(&f.app, "GET", &format!("/jobs/{ext}/result"), Some("sk-a"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h["x-toy"], "1");
    let u = Url::parse(v["url"].as_str().unwrap()).unwrap();
    assert_eq!(u.host_str(), Some("fv.test"));
    // The signed URL serves the file without auth.
    let r = f.app.clone().oneshot(Request::get(format!("{}?{}", u.path(), u.query().unwrap())).body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(r.headers()["access-control-allow-origin"], "*");

    // Late events are ignored.
    let j = apply_event(&f.ctx, id, JobEvent::Failed(ApiError::engine_failed("late"))).await.unwrap();
    assert_eq!(j.status(), JobStatus::Succeeded);

    settle(&f).await;
    let seen = f.rx.0.lock().unwrap().clone();
    assert!(seen[0].get("challenge").is_some());
    let statuses: Vec<&str> = seen[1..].iter().map(|v| v["task"]["status"].as_str().unwrap()).collect();
    assert_eq!(statuses, ["queued", "running", "succeeded"]);

    // Removal deletes the artifact.
    let path = match &j.artifacts[0].location {
        fastvideo_protocol::ArtifactLocation::Local(p) => p.clone(),
        _ => unreachable!(),
    };
    assert!(path.exists());
    f.ctx.jobs().remove(id).await.unwrap();
    assert!(!path.exists());
}

#[tokio::test]
async fn sync_submit_waits_for_result() {
    let f = fixture(Engine::default()).await;
    f.engine.auto.set(f.ctx.clone()).ok().unwrap();
    let (s, _, v) = call(&f.app, "POST", "/sync", Some("sk-a"), Some(serde_json::json!({"prompt": "a"}))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(v["url"].as_str().unwrap().contains("/files/"));
}

#[tokio::test]
async fn cancel_semantics() {
    let f = fixture(Engine::default()).await;
    let (_, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a"}))).await;
    let id = f.ctx.jobs().by_external(ProtocolId::MiniMaxV2, v["task_id"].as_str().unwrap()).await.unwrap().id;
    let j = cancel_job(&f.ctx, id).await.unwrap();
    assert_eq!(j.status(), JobStatus::Cancelled);
    assert_eq!(f.engine.cancelled.lock().unwrap().as_slice(), &[id]);
    assert_eq!(cancel_job(&f.ctx, id).await.unwrap_err().kind, ErrorKind::AlreadyCompleted);

    // Running: flag + engine token; the engine's Cancelled event finishes it.
    let (_, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "b"}))).await;
    let id = f.ctx.jobs().by_external(ProtocolId::MiniMaxV2, v["task_id"].as_str().unwrap()).await.unwrap().id;
    apply_event(&f.ctx, id, JobEvent::Started).await.unwrap();
    let j = cancel_job(&f.ctx, id).await.unwrap();
    assert_eq!((j.status(), j.cancel_requested), (JobStatus::Running, true));
    let j = apply_event(&f.ctx, id, JobEvent::Cancelled).await.unwrap();
    assert_eq!(j.status(), JobStatus::Cancelled);
}

#[tokio::test]
async fn sse_follows_job_until_terminal() {
    let f = fixture(Engine::default()).await;
    let (_, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a"}))).await;
    let id = f.ctx.jobs().by_external(ProtocolId::MiniMaxV2, v["task_id"].as_str().unwrap()).await.unwrap().id;
    let spec = SseSpec {
        initial: vec![SseEvent::data(r#"{"status":"queued"}"#)],
        follow: Some(SseFollow::JobStatus { job: id, close_on_terminal: true, with_logs: false }),
        keepalive: Some(Duration::from_secs(15)),
    };
    let resp = into_response(HttpReply::sse(spec), &f.ctx, Some(Arc::new(View))).await;
    assert_eq!(resp.headers()["content-type"], "text/event-stream");
    let ctx = f.ctx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        apply_event(&ctx, id, JobEvent::Started).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        apply_event(&ctx, id, JobEvent::Failed(ApiError::engine_failed("boom"))).await.unwrap();
    });
    let body = tokio::time::timeout(Duration::from_secs(5), axum::body::to_bytes(resp.into_body(), 1 << 20))
        .await
        .expect("stream closes after the terminal status")
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    let datas: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    assert_eq!(datas.first(), Some(&r#"{"status":"queued"}"#));
    assert!(datas.last().unwrap().contains("\"failed\""), "{text}");
}

/// `with_logs` reaches the follower's `ViewCtx`.
#[tokio::test]
async fn sse_follow_renders_with_logs() {
    struct LogsFlag;
    impl JobView for LogsFlag {
        fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
            HttpReply::json(200, serde_json::json!({"status": job.status().as_str(), "with_logs": cx.with_logs}))
        }
        fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
            self.status_reply(job, cx)
        }
    }
    let f = fixture(Engine::default()).await;
    let (_, _, v) = call(&f.app, "POST", "/submit", Some("sk-a"), Some(serde_json::json!({"prompt": "a"}))).await;
    let id = f.ctx.jobs().by_external(ProtocolId::MiniMaxV2, v["task_id"].as_str().unwrap()).await.unwrap().id;
    let spec = SseSpec {
        initial: vec![],
        follow: Some(SseFollow::JobStatus { job: id, close_on_terminal: true, with_logs: true }),
        keepalive: None,
    };
    let resp = into_response(HttpReply::sse(spec), &f.ctx, Some(Arc::new(LogsFlag))).await;
    let ctx = f.ctx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        apply_event(&ctx, id, JobEvent::Failed(ApiError::engine_failed("boom"))).await.unwrap();
    });
    let body = tokio::time::timeout(Duration::from_secs(5), axum::body::to_bytes(resp.into_body(), 1 << 20))
        .await
        .unwrap()
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains(r#""with_logs":true"#), "{text}");
}

/// A job removed while its output is being stored does not leak the
/// stored artifact.
#[tokio::test]
async fn finished_output_is_not_leaked_when_the_job_vanishes() {
    struct Vanishing(crate::store::MemJobStore);
    #[async_trait::async_trait]
    impl fastvideo_protocol::JobStore for Vanishing {
        async fn insert(&self, job: Job) -> Result<(), fastvideo_protocol::StoreError> {
            self.0.insert(job).await
        }
        async fn get(&self, id: JobId) -> Option<Job> {
            self.0.get(id).await
        }
        async fn by_external(&self, p: ProtocolId, e: &str) -> Option<Job> {
            self.0.by_external(p, e).await
        }
        async fn update(&self, id: JobId, _f: fastvideo_protocol::JobUpdate) -> Result<Job, fastvideo_protocol::StoreError> {
            // Removed between `get` and `update`.
            Err(fastvideo_protocol::StoreError::NotFound(id))
        }
        async fn list(&self, q: fastvideo_protocol::ListQuery) -> fastvideo_protocol::Page<Job> {
            self.0.list(q).await
        }
        async fn remove(&self, id: JobId) -> Option<Job> {
            self.0.remove(id).await
        }
        fn watch(&self, id: JobId) -> Option<tokio::sync::watch::Receiver<fastvideo_protocol::JobSnapshot>> {
            self.0.watch(id)
        }
        async fn sweep_expired(&self, now: time::OffsetDateTime) -> usize {
            self.0.sweep_expired(now).await
        }
    }
    use fastvideo_protocol::JobStore as _;
    let dir = std::env::temp_dir().join(format!("fvkit-leak-{}", crate::random_token()));
    let store = Arc::new(Vanishing(crate::store::MemJobStore::memory()));
    let cfg = ServeConfig::new(Url::parse("http://fv.test").unwrap(), &dir);
    let ctx = ServeCtx::builder(cfg, Arc::new(Engine::default())).jobs(store.clone()).build().await.unwrap();
    let job = crate::store::tests::job(ProtocolId::MiniMaxV2, "gone", time::OffsetDateTime::now_utc());
    let id = job.id;
    store.0.insert(job).await.unwrap();
    let src = dir.join("out.mp4");
    std::fs::write(&src, b"mp4").unwrap();
    let out = FinishedOutput {
        file: src,
        meta: ArtifactMeta { file_name: "out.mp4".into(), mime: "video/mp4".into(), ..Default::default() },
        metrics: Default::default(),
    };
    let e = apply_event(&ctx, id, JobEvent::Finished(out)).await.unwrap_err();
    assert_eq!(e.kind, ErrorKind::NotFound);
    let left: Vec<_> = std::fs::read_dir(dir.join("artifacts")).unwrap().collect();
    assert!(left.is_empty(), "stored artifact leaked: {left:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn reply_conversion() {
    let f = fixture(Engine::default()).await;
    let r = into_response(HttpReply::bytes(201, "video/mp4", &b"abc"[..]).with_header("x-request-id", "r1"), &f.ctx, None).await;
    assert_eq!(r.status(), StatusCode::CREATED);
    assert_eq!(r.headers()["content-type"], "video/mp4");
    assert_eq!(r.headers()["x-request-id"], "r1");
    let p = f.dir.join("f.bin");
    std::fs::write(&p, b"12345").unwrap();
    let r = into_response(HttpReply::file(200, &p, "video/mp4"), &f.ctx, None).await;
    assert_eq!(r.headers()["content-type"], "video/mp4");
    assert_eq!(axum::body::to_bytes(r.into_body(), 100).await.unwrap().as_ref(), b"12345");
    let r = into_response(HttpReply::empty(204), &f.ctx, None).await;
    assert_eq!(r.status(), StatusCode::NO_CONTENT);
    // `url_signer` is the artifact store's.
    let _: &dyn UrlSigner = f.ctx.url_signer();
}
