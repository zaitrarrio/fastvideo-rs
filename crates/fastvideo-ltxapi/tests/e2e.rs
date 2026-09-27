//! End to end over HTTP: the LTX router + serve-kit + `EngineService` running
//! the `FakeBackend` (design §7.3). Covers the v2 lifecycle, v1 sync bytes and
//! timeouts, uploads and `ltx://` refs, auth, stubs, gaps and model binding.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use fastvideo_engine_service::{
    CancelOutcome, EngineConfig, EngineEvent, EngineService, FakeBackend, FakeConfig, Mp4Mode,
    Priority,
};
use fastvideo_ltxapi::{router, LtxConfig};
use fastvideo_protocol::{ApiError, AudioPlan, Job, JobId, ModelCaps, ProtocolId};
use fastvideo_serve_kit::artifacts::ArtifactMeta;
use fastvideo_serve_kit::auth::{Auth, AuthMode, KeyRing};
use fastvideo_serve_kit::ctx::{EngineGate, ServeConfig};
use fastvideo_serve_kit::events::{apply_event, FinishedOutput, JobEvent};
use fastvideo_serve_kit::{ServeCtx, UrlKey};
use serde_json::{json, Value};
use tower::ServiceExt;
use url::Url;

/// A 1x1 RGBA PNG.
const PNG: [u8; 70] = [
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 218, 99, 100, 96, 248, 95, 15, 0, 2,
    135, 1, 128, 235, 71, 186, 146, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
/// What the gate stores when the fake wrote no MP4 (no ffmpeg on the host).
const STUB_MP4: &[u8] = b"\0\0\0\x18ftypisom\0\0\x02\0isomiso2fake-ltx-output";

/// `EngineGate` over the real engine service with the fake backend: engine
/// events are pumped into the job store through `apply_event`.
struct Gate {
    engine: EngineService,
    ctx: OnceLock<ServeCtx>,
    /// Expose the engine's tier aliases (`ltx-pro`, …) through `alias`.
    aliases: bool,
}

#[async_trait::async_trait]
impl EngineGate for Gate {
    fn models(&self) -> Vec<ModelCaps> {
        self.engine.caps().models().cloned().collect()
    }
    fn alias(&self, name: &str) -> Option<String> {
        if !self.aliases {
            return None;
        }
        self.engine
            .caps()
            .tier_aliases()
            .into_iter()
            .find(|(a, _)| a == name)
            .map(|(_, m)| m.0)
    }
    fn admit(&self) -> Result<(), ApiError> {
        if self.engine.readiness().is_ready() {
            Ok(())
        } else {
            Err(ApiError::loading("models are loading"))
        }
    }
    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        let mut h = self.engine.submit(job.id, job.resolved.clone(), Priority::Batch).await?;
        let ctx = self.ctx.get().expect("ctx").clone();
        let (id, resolved) = (job.id, job.resolved.clone());
        tokio::spawn(async move {
            while let Some(ev) = h.events.recv().await {
                let ev = match ev {
                    EngineEvent::Queued { position } => JobEvent::Queued { position },
                    EngineEvent::Started => JobEvent::Started,
                    EngineEvent::Stage { name } => JobEvent::Stage { name: name.to_owned() },
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
                                tokio::fs::write(&p, STUB_MP4).await.unwrap();
                                p
                            }
                        };
                        let (w, h) = resolved.output_size();
                        let audio = match resolved.audio {
                            AudioPlan::Native { rate, channels } => Some((rate, channels)),
                            _ => None,
                        };
                        let meta = ArtifactMeta {
                            file_name: "output.mp4".into(),
                            mime: "video/mp4".into(),
                            width: w,
                            height: h,
                            frames: resolved.num_frames,
                            fps: resolved.fps,
                            audio,
                        };
                        JobEvent::Finished(FinishedOutput { file, meta, metrics: out.metrics })
                    }
                };
                let _ = apply_event(&ctx, id, ev).await;
            }
        });
        Ok(())
    }
    async fn cancel(&self, id: JobId) -> bool {
        self.engine.cancel(id) != CancelOutcome::Unknown
    }
}

struct Fx {
    app: Router,
    ctx: ServeCtx,
    dir: std::path::PathBuf,
}

impl Drop for Fx {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

async fn fixture(cfg: LtxConfig, aliases: bool) -> Fx {
    let dir = std::env::temp_dir().join(format!("fv-ltxapi-{}", fastvideo_serve_kit::random_token()));
    let fake = FakeConfig { mp4: Mp4Mode::Off, ..FakeConfig::default() }.with_models(&["fake-ltx-pro", "fake-ltx-turbo"]);
    let engine = EngineService::start(
        EngineConfig { output_dir: dir.join("engine"), ..EngineConfig::default() },
        vec![Box::new(FakeBackend::new(fake))],
    )
    .unwrap();
    assert!(engine.wait_ready().await.is_ready());
    let gate = Arc::new(Gate { engine, ctx: OnceLock::new(), aliases });
    let scfg = ServeConfig::new(Url::parse("http://fv.test").unwrap(), &dir);
    let ctx = ServeCtx::builder(scfg, gate.clone())
        .auth(Auth::new(AuthMode::Keys, KeyRing::from_plain(["sk-a", "sk-b"])))
        .url_key(UrlKey::new("k"))
        .build()
        .await
        .unwrap();
    let _ = gate.ctx.set(ctx.clone());
    let app = router(cfg).merge(ctx.routes()).with_state(ctx.clone());
    Fx { app, ctx, dir }
}

struct Resp {
    status: StatusCode,
    headers: HeaderMap,
    bytes: Vec<u8>,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.bytes).unwrap_or(Value::Null)
    }
    fn header(&self, k: &str) -> Option<&str> {
        self.headers.get(k).and_then(|v| v.to_str().ok())
    }
    fn err_type(&self) -> String {
        let v = self.json();
        assert_eq!(v["type"], "error", "not an error envelope: {v}");
        v["error"]["type"].as_str().unwrap().to_owned()
    }
}

async fn send(app: &Router, req: Request<Body>) -> Resp {
    let r = app.clone().oneshot(req).await.unwrap();
    let status = r.status();
    let headers = r.headers().clone();
    let bytes = axum::body::to_bytes(r.into_body(), usize::MAX).await.unwrap().to_vec();
    Resp { status, headers, bytes }
}

async fn call(app: &Router, method: &str, uri: &str, key: Option<&str>, body: Option<Value>) -> Resp {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(k) = key {
        b = b.header("authorization", format!("Bearer {k}"));
    }
    let body = match body {
        Some(v) => {
            b = b.header("content-type", "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    send(app, b.body(body).unwrap()).await
}

fn assert_rid(r: &Resp) {
    let rid = r.header("x-request-id").expect("x-request-id");
    assert_eq!(rid.len(), 32, "{rid}");
    assert!(rid.chars().all(|c| c.is_ascii_hexdigit()));
}

fn t2v(model: &str, res: &str, duration: u32) -> Value {
    json!({"prompt": "a red fox in the snow", "model": model, "duration": duration, "resolution": res})
}

async fn poll_terminal(app: &Router, ep: &str, id: &str) -> Resp {
    for _ in 0..2400 {
        let r = call(app, "GET", &format!("/v2/{ep}/{id}"), Some("sk-a"), None).await;
        assert_eq!(r.status, StatusCode::OK, "{:?}", r.json());
        let s = r.json()["status"].as_str().unwrap().to_owned();
        assert!(["pending", "processing", "completed", "failed"].contains(&s.as_str()));
        if s == "completed" || s == "failed" {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("job {id} did not finish");
}

fn path_and_query(u: &str) -> String {
    let u = Url::parse(u).unwrap();
    format!("{}?{}", u.path(), u.query().unwrap_or(""))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v2_lifecycle() {
    let fx = fixture(LtxConfig::default(), true).await;
    let r = call(&fx.app, "POST", "/v2/text-to-video", Some("sk-a"), Some(t2v("ltx-2-5-fast", "1280x720", 8))).await;
    assert_eq!(r.status, StatusCode::ACCEPTED, "{:?}", r.json());
    assert_rid(&r);
    let body = r.json();
    let id = body["id"].as_str().unwrap().to_owned();
    assert_eq!(body.as_object().unwrap().len(), 2);
    assert!(uuid::Uuid::parse_str(&id).is_ok());
    assert!(body["created_at"].as_str().unwrap().ends_with('Z'));
    assert_eq!(r.header("x-fv-tier"), Some("turbo"));

    // Geometry went through pad-and-crop.
    let job = fx.ctx.jobs().by_external(ProtocolId::LtxV2, &id).await.unwrap();
    assert_eq!(job.resolved.model.0, "fake-ltx-turbo");
    assert_eq!((job.resolved.width, job.resolved.height, job.resolved.num_frames), (1280, 768, 193));
    assert_eq!(job.resolved.post.crop, Some((1280, 720)));
    assert_eq!(job.requested_model(), "ltx-2-5-fast");

    let done = poll_terminal(&fx.app, "text-to-video", &id).await;
    let v = done.json();
    assert_eq!(v["status"], "completed", "{v}");
    assert_eq!(v["id"], id);
    assert!(v["completed_at"].is_string());
    assert_rid(&done);
    let url = v["result"]["video_url"].as_str().unwrap();
    // Download needs no auth.
    let file = call(&fx.app, "GET", &path_and_query(url), None, None).await;
    assert_eq!(file.status, StatusCode::OK);
    assert_eq!(file.bytes, STUB_MP4);

    // Wrong endpoint segment, wrong surface, unknown id, other owner: 404.
    for (uri, key) in [
        (format!("/v2/image-to-video/{id}"), "sk-a"),
        (format!("/v2/retake/{id}"), "sk-a"),
        ("/v2/text-to-video/00000000-0000-0000-0000-000000000000".to_owned(), "sk-a"),
        (format!("/v2/text-to-video/{id}"), "sk-b"),
    ] {
        let r = call(&fx.app, "GET", &uri, Some(key), None).await;
        assert_eq!(r.status, StatusCode::NOT_FOUND, "{uri}");
        assert_eq!(r.err_type(), "not_found_error");
        assert_rid(&r);
    }
    let r = call(&fx.app, "GET", &format!("/v2/text-to-video/{id}"), None, None).await;
    assert_eq!((r.status, r.err_type()), (StatusCode::UNAUTHORIZED, "authentication_error".into()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v2_engine_failure() {
    let fx = fixture(LtxConfig::default(), true).await;
    let mut b = t2v("ltx-2-5-pro", "1280x720", 6);
    b["prompt"] = json!("please [fake:fail] now");
    let r = call(&fx.app, "POST", "/v2/text-to-video", Some("sk-a"), Some(b)).await;
    assert_eq!(r.status, StatusCode::ACCEPTED);
    assert_eq!(r.header("x-fv-tier"), Some("max"));
    let id = r.json()["id"].as_str().unwrap().to_owned();
    let v = poll_terminal(&fx.app, "text-to-video", &id).await.json();
    assert_eq!(v["status"], "failed");
    assert_eq!(v["error"]["type"], "api_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("injected failure"));
    assert!(v.get("result").is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v1_sync_returns_mp4_bytes() {
    let fx = fixture(LtxConfig::default(), false).await;
    let r = call(&fx.app, "POST", "/v1/text-to-video", Some("sk-a"), Some(t2v("ltx-turbo", "720x1280", 6))).await;
    assert_eq!(r.status, StatusCode::OK, "{}", String::from_utf8_lossy(&r.bytes));
    assert_eq!(r.header("content-type"), Some("video/mp4"));
    assert_rid(&r);
    assert_eq!(r.header("x-fv-tier"), Some("turbo"));
    assert_eq!(r.bytes, STUB_MP4);
    // A v1 job is not visible through the v2 status route.
    let jobs = fx.ctx.jobs().list(Default::default()).await;
    let j = jobs.items.iter().find(|j| j.protocol == ProtocolId::LtxV1).unwrap();
    assert_eq!((j.resolved.width, j.resolved.height, j.resolved.post.crop), (768, 1280, Some((720, 1280))));
    let r = call(&fx.app, "GET", &format!("/v2/text-to-video/{}", j.external_id), Some("sk-a"), None).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);

    // Failures come back on the same request.
    let mut b = t2v("ltx-turbo", "1280x720", 6);
    b["prompt"] = json!("[fake:fail]");
    let r = call(&fx.app, "POST", "/v1/text-to-video", Some("sk-a"), Some(b)).await;
    assert_eq!((r.status, r.err_type()), (StatusCode::INTERNAL_SERVER_ERROR, "api_error".into()));
    assert_rid(&r);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v1_timeout_is_504_and_cancels() {
    let cfg = LtxConfig { sync_timeout: Some(Duration::from_millis(1)), ..LtxConfig::default() };
    let fx = fixture(cfg, true).await;
    let r = call(&fx.app, "POST", "/v1/text-to-video", Some("sk-a"), Some(t2v("ltx-2-5-fast", "1280x720", 6))).await;
    assert_eq!(r.status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(r.err_type(), "api_error");
    assert_rid(&r);
    let jobs = fx.ctx.jobs().list(Default::default()).await;
    let j = &jobs.items[0];
    assert!(j.cancel_requested || j.is_terminal());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v1_concurrency_limit() {
    let cfg = LtxConfig { v1_concurrency: Some(1), ..LtxConfig::default() };
    let fx = fixture(cfg, true).await;
    let a = call(&fx.app, "POST", "/v1/text-to-video", Some("sk-a"), Some(t2v("ltx-turbo", "1280x720", 6)));
    let b = async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        call(&fx.app, "POST", "/v1/text-to-video", Some("sk-a"), Some(t2v("ltx-turbo", "1280x720", 6))).await
    };
    let c = async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        call(&fx.app, "POST", "/v1/upload", Some("sk-a"), None).await
    };
    let (a, b, c) = tokio::join!(a, b, c);
    assert_eq!(a.status, StatusCode::OK);
    assert_eq!(b.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(b.err_type(), "concurrency_limit_error");
    assert_eq!(b.header("retry-after"), Some("5"));
    // Uploads are not generations.
    assert_eq!(c.status, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upload_then_image_to_video() {
    let fx = fixture(LtxConfig::default(), true).await;
    let r = call(&fx.app, "POST", "/v1/upload", Some("sk-a"), None).await;
    assert_eq!(r.status, StatusCode::OK);
    assert_rid(&r);
    let u = r.json();
    assert_eq!(u["required_headers"], json!({}));
    let storage = u["storage_uri"].as_str().unwrap().to_owned();
    assert!(storage.starts_with("ltx://uploads/"));
    assert!(u["expires_at"].as_str().unwrap().ends_with('Z'));
    let put_url = Url::parse(u["upload_url"].as_str().unwrap()).unwrap();
    assert_eq!(put_url.path(), format!("/uploads/{}", &storage["ltx://uploads/".len()..]));
    // Clients copy upstream's x-goog-* headers; they are accepted and ignored.
    let put = Request::put(put_url.path())
        .header("content-type", "image/png")
        .header("x-goog-content-length-range", "0,209715200")
        .header("x-goog-if-generation-match", "0")
        .body(Body::from(PNG.to_vec()))
        .unwrap();
    assert_eq!(send(&fx.app, put).await.status, StatusCode::OK);

    let mut b = t2v("ltx-2-3-pro", "720x1280", 6);
    b["image_uri"] = json!(storage);
    let r = call(&fx.app, "POST", "/v2/image-to-video", Some("sk-a"), Some(b)).await;
    assert_eq!(r.status, StatusCode::ACCEPTED, "{:?}", r.json());
    let id = r.json()["id"].as_str().unwrap().to_owned();
    let job = fx.ctx.jobs().by_external(ProtocolId::LtxV2, &id).await.unwrap();
    assert_eq!(job.resolved.keyframes.len(), 1);
    assert_eq!((job.resolved.width, job.resolved.height), (768, 1280));
    let v = poll_terminal(&fx.app, "image-to-video", &id).await.json();
    assert_eq!(v["status"], "completed", "{v}");
    // The job is not visible under text-to-video.
    let r = call(&fx.app, "GET", &format!("/v2/text-to-video/{id}"), Some("sk-a"), None).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);

    // Data URIs work too; unknown upload tokens and plain http are refused.
    let mut b = t2v("ltx-2-3-pro", "1280x720", 6);
    b["image_uri"] = json!(format!("data:image/png;base64,{PNG_B64}"));
    let r = call(&fx.app, "POST", "/v2/image-to-video", Some("sk-a"), Some(b.clone())).await;
    assert_eq!(r.status, StatusCode::ACCEPTED, "{:?}", r.json());
    for bad in ["ltx://uploads/0123456789abcdef0123456789abcdef", "http://example.com/a.png"] {
        b["image_uri"] = json!(bad);
        let r = call(&fx.app, "POST", "/v2/image-to-video", Some("sk-a"), Some(b.clone())).await;
        assert_eq!((r.status, r.err_type()), (StatusCode::BAD_REQUEST, "invalid_request_error".into()), "{bad}");
    }
    // Upload needs auth.
    let r = call(&fx.app, "POST", "/v1/upload", None, None).await;
    assert_eq!((r.status, r.err_type()), (StatusCode::UNAUTHORIZED, "authentication_error".into()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn auth_stubs_and_gaps() {
    let fx = fixture(LtxConfig::default(), false).await;
    let app = &fx.app;
    let body = t2v("ltx-2-5-fast", "1920x1080", 8);

    // Auth: missing, wrong scheme, unknown key.
    for key in [None, Some("nope")] {
        let r = call(app, "POST", "/v2/text-to-video", key, Some(body.clone())).await;
        assert_eq!((r.status, r.err_type()), (StatusCode::UNAUTHORIZED, "authentication_error".into()));
        assert_rid(&r);
    }
    let req = Request::post("/v2/text-to-video")
        .header("authorization", "Key sk-a")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    assert_eq!(send(app, req).await.status, StatusCode::UNAUTHORIZED);

    // Stubs: 403 after auth on both surfaces.
    for api in ["v1", "v2"] {
        for ep in fastvideo_ltxapi::stubs::STUB_ENDPOINTS {
            let uri = format!("/{api}/{ep}");
            let r = call(app, "POST", &uri, Some("sk-a"), Some(json!({"video_uri": "https://example.com/v.mp4"}))).await;
            assert_eq!((r.status, r.err_type()), (StatusCode::FORBIDDEN, "permission_error".into()), "{uri}");
            assert_eq!(r.json()["error"]["message"], "endpoint not available for the account");
            assert_rid(&r);
            let r = call(app, "POST", &uri, None, None).await;
            assert_eq!(r.status, StatusCode::UNAUTHORIZED, "{uri}");
        }
    }

    // Gaps and validation: 400 invalid_request_error; never approximated.
    let with = |k: &str, v: Value| {
        let mut b = body.clone();
        b[k] = v;
        b
    };
    let cases = [
        (with("fps", json!(30)), "fps must be one of"),
        (with("duration", Value::Null), "automatic duration"),
        (with("camera_motion", json!("dolly_in")), "camera_motion"),
        (with("model", json!("ltx-2-pro")), "removed"),
        (with("model", json!("ltx-9")), "model must be one of"),
        (with("duration", json!(12)).tap_res("3840x2160"), "allowed: 6, 8, 10"),
        (with("resolution", json!("1920x1088")), "resolution must be one of"),
        (json!({"model": "ltx-2-5-fast", "resolution": "1920x1080", "prompt": "p"}), "duration is required"),
    ];
    for (b, needle) in cases {
        let r = call(app, "POST", "/v2/text-to-video", Some("sk-a"), Some(b.clone())).await;
        assert_eq!((r.status, r.err_type()), (StatusCode::BAD_REQUEST, "invalid_request_error".into()), "{b}");
        let msg = r.json()["error"]["message"].as_str().unwrap().to_owned();
        assert!(msg.contains(needle), "{msg} !~ {needle}");
    }
    let mut kf = with("image_uri", json!("https://example.com/a.png"));
    kf["last_frame_uri"] = json!("https://example.com/b.png");
    let r = call(app, "POST", "/v2/image-to-video", Some("sk-a"), Some(kf)).await;
    assert_eq!((r.status, r.err_type()), (StatusCode::BAD_REQUEST, "invalid_request_error".into()));
    assert!(r.json()["error"]["message"].as_str().unwrap().contains("last-frame"));
    let r = call(app, "POST", "/v2/text-to-video", Some("sk-a"), Some(json!("nope"))).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    let req = Request::post("/v2/text-to-video")
        .header("authorization", "Bearer sk-a")
        .body(Body::from("{not json"))
        .unwrap();
    assert_eq!(send(app, req).await.status, StatusCode::BAD_REQUEST);

    // A documented id whose tier is not served: 403 (no draft model here).
    let r = call(app, "POST", "/v2/text-to-video", Some("sk-a"), Some(with("model", json!("ltx-draft")))).await;
    assert_eq!((r.status, r.err_type()), (StatusCode::FORBIDDEN, "permission_error".into()));
    // Without gate aliases the tier still binds (resolve_tier).
    let r = call(app, "POST", "/v2/text-to-video", Some("sk-a"), Some(body.clone())).await;
    assert_eq!(r.status, StatusCode::ACCEPTED, "{:?}", r.json());
    // No job was stored for any refused request.
    assert_eq!(fx.ctx.jobs().list(Default::default()).await.items.len(), 1);
}

trait TapRes {
    fn tap_res(self, res: &str) -> Value;
}
impl TapRes for Value {
    fn tap_res(mut self, res: &str) -> Value {
        self["resolution"] = json!(res);
        self
    }
}
