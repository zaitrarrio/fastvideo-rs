//! Shared test helpers: an `EngineGate` over the real `EngineService` with the
//! fake backend (what `fv-serve --features fake` wires), a router fixture,
//! and HTTP call helpers.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use fastvideo_engine_service::{
    CancelOutcome, EngineBackend, EngineConfig, EngineEvent, EngineService, FakeBackend,
    FakeConfig, FakeModel, FakeTiming, JobHandle, Mp4Mode, Priority, Readiness, Recipe,
};
use fastvideo_openai_videos::{fastwan, router, VideosConfig};
use fastvideo_protocol::{
    ApiError, AudioPlan, CanvasCaps, FpsCaps, FrameGrid, Job, JobId, KnobCaps, ModelCaps, ModelId,
    RefLimits, StreamCaps, Task, Tier,
};
use fastvideo_serve_kit::auth::{Auth, AuthMode, KeyRing};
use fastvideo_serve_kit::{
    apply_event, ArtifactMeta, EngineGate, FinishedOutput, JobEvent, ServeConfig, ServeCtx, UrlKey,
};
use tower::ServiceExt;

pub const T: Duration = Duration::from_secs(30);

/// A FastWan-style Wan clip model: 4k+1 frames 49..=121, video-only, any
/// of 16/24 fps (container rate), canvases up to 1280x704.
pub fn fastwan_model() -> FakeModel {
    let grid = FrameGrid::new(4, 1, 49, 121, 81);
    FakeModel {
        caps: ModelCaps {
            id: ModelId::new("fake-fastwan"),
            family: fastvideo_protocol::Family::Wan,
            served_names: vec!["fastwan-5b".into()],
            tasks: [Task::T2V].into_iter().collect(),
            audio: None,
            fps: FpsCaps {
                allowed: vec![16, 24],
                default: 24,
                container_only: true,
            },
            stream: Some(StreamCaps::Clip {
                min_s: 49.0 / 24.0,
                max_s: 121.0 / 24.0,
            }),
            frames: grid,
            canvas: CanvasCaps {
                multiple: 16,
                max_area: 1280 * 704,
                aspect: (0.25, 4.0),
                short_edges: vec![704, 480],
                pad_and_crop: false,
            },
            refs: RefLimits::none(),
            knobs: KnobCaps {
                guidance_2: false,
                ..KnobCaps::all()
            },
            resident: true,
            tier: None,
            recipe: None,
        }
        .with_tier(Tier::Turbo, "dmd-3step"),
        recipe: Recipe {
            name: "dmd-3step".into(),
            steps: Some(3),
            attention: "dense".into(),
            vae: "full".into(),
            summary: "fake: FastWan 3-step".into(),
            ..Recipe::default()
        },
    }
}

/// The engine as `fv-serve` wires it: submit -> `JobHandle` -> events applied
/// to the job store through serve-kit's `apply_event`.
pub struct Bridge {
    pub engine: EngineService,
    pub ctx: OnceLock<ServeCtx>,
    pub loading: bool,
}

#[async_trait::async_trait]
impl EngineGate for Bridge {
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
    fn admit(&self) -> Result<(), ApiError> {
        if self.loading {
            return Err(ApiError::loading("models are loading"));
        }
        match self.engine.readiness() {
            Readiness::Ready => Ok(()),
            Readiness::Failed(e) => Err(ApiError::engine_failed(e)),
            _ => Err(ApiError::loading("models are loading")),
        }
    }
    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        let h = self
            .engine
            .submit(job.id, job.resolved.clone(), Priority::Batch)
            .await?;
        let ctx = self.ctx.get().expect("ctx set").clone();
        let job = job.clone();
        tokio::spawn(forward(ctx, job, h));
        Ok(())
    }
    async fn cancel(&self, id: JobId) -> bool {
        self.engine.cancel(id) != CancelOutcome::Unknown
    }
}

async fn forward(ctx: ServeCtx, job: Job, mut h: JobHandle) {
    while let Some(ev) = h.events.recv().await {
        let ev = match ev {
            EngineEvent::Queued { position } => JobEvent::Queued { position },
            EngineEvent::Started => JobEvent::Started,
            EngineEvent::Stage { name } => JobEvent::Stage {
                name: name.to_owned(),
            },
            EngineEvent::Progress { step, total } => JobEvent::Progress { step, total },
            EngineEvent::Log(l) => JobEvent::Log(l),
            EngineEvent::Failed(e) => JobEvent::Failed(e),
            EngineEvent::Cancelled => JobEvent::Cancelled,
            EngineEvent::Finished(out) => {
                // Without ffmpeg the fake writes no MP4: stand in a tiny file.
                let file = match out.mp4 {
                    Some(p) => p,
                    None => {
                        let dir = ctx.outputs_dir(job.id);
                        tokio::fs::create_dir_all(&dir).await.unwrap();
                        let p = dir.join("output.mp4");
                        tokio::fs::write(&p, fake_mp4(&job)).await.unwrap();
                        p
                    }
                };
                let r = &job.resolved;
                let (w, h) = r.output_size();
                let audio = match r.audio {
                    AudioPlan::Native { rate, channels } => Some((rate, channels)),
                    _ => None,
                };
                let meta = ArtifactMeta {
                    file_name: "video.mp4".into(),
                    mime: "video/mp4".into(),
                    width: w,
                    height: h,
                    frames: r.num_frames,
                    fps: r.fps,
                    audio,
                };
                JobEvent::Finished(FinishedOutput {
                    file,
                    meta,
                    metrics: out.metrics,
                })
            }
        };
        let terminal = matches!(
            ev,
            JobEvent::Finished(_) | JobEvent::Failed(_) | JobEvent::Cancelled
        );
        let _ = apply_event(&ctx, job.id, ev).await;
        if terminal {
            break;
        }
    }
}

/// The stand-in MP4 bytes: an `ftyp` box plus the job's shape.
pub fn fake_mp4(job: &Job) -> Vec<u8> {
    let r = &job.resolved;
    let mut b = b"\0\0\0\x18ftypisom\0\0\x02\0isomiso2".to_vec();
    b.extend_from_slice(format!("{}x{}x{}@{}", r.width, r.height, r.num_frames, r.fps).as_bytes());
    b
}

pub struct Fixture {
    pub ctx: ServeCtx,
    pub bridge: Arc<Bridge>,
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
    pub loading: bool,
    pub keys: bool,
    pub models: Vec<FakeModel>,
    pub cfg: VideosConfig,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            step: Duration::from_millis(2),
            loading: false,
            keys: false,
            models: vec![
                FakeModel::h3_turbo(),
                FakeModel::h3_max(),
                FakeModel::wan(),
                fastwan_model(),
            ],
            cfg: VideosConfig {
                created: 1_700_000_000,
                ..VideosConfig::default()
            },
        }
    }
}

pub async fn fixture(o: Opts) -> Fixture {
    let dir = std::env::temp_dir().join(format!("fv-oaiv-{}", fastvideo_serve_kit::random_token()));
    std::fs::create_dir_all(&dir).unwrap();
    let fake = FakeConfig {
        models: o.models,
        timing: FakeTiming {
            step: o.step,
            ..FakeTiming::default()
        },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    };
    let ecfg = EngineConfig {
        output_dir: dir.join("engine"),
        ..EngineConfig::default()
    };
    let backends: Vec<Box<dyn EngineBackend>> = vec![Box::new(FakeBackend::new(fake))];
    let engine = EngineService::start(ecfg, backends).unwrap();
    assert_eq!(
        tokio::time::timeout(T, engine.wait_ready()).await.unwrap(),
        Readiness::Ready
    );
    let bridge = Arc::new(Bridge {
        engine,
        ctx: OnceLock::new(),
        loading: o.loading,
    });
    let cfg = ServeConfig::new(url::Url::parse("http://fv.test").unwrap(), &dir);
    let auth = if o.keys {
        Auth::new(AuthMode::Keys, KeyRing::from_plain(["sk-a", "sk-b"]))
    } else {
        Auth::default()
    };
    let ctx = ServeCtx::builder(cfg, bridge.clone())
        .auth(auth)
        .url_key(UrlKey::new("k"))
        .build()
        .await
        .unwrap();
    bridge.ctx.set(ctx.clone()).ok().unwrap();
    let app = router(&ctx, o.cfg.clone())
        .merge(fastwan::service_routes(&ctx, &o.cfg))
        .merge(ctx.routes())
        .with_state(ctx.clone());
    Fixture {
        ctx,
        bridge,
        app,
        dir,
    }
}

pub struct Reply {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub bytes: bytes::Bytes,
}

impl Reply {
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.bytes).unwrap_or(serde_json::Value::Null)
    }
}

pub async fn send(app: &Router, req: Request<Body>) -> Reply {
    let r = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = r.into_parts();
    let bytes = axum::body::to_bytes(body, 1 << 26).await.unwrap();
    Reply {
        status: parts.status,
        headers: parts.headers,
        bytes,
    }
}

pub async fn call(app: &Router, method: &str, uri: &str, body: Option<serde_json::Value>) -> Reply {
    call_key(app, method, uri, None, body).await
}

pub async fn call_key(
    app: &Router,
    method: &str,
    uri: &str,
    key: Option<&str>,
    body: Option<serde_json::Value>,
) -> Reply {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(k) = key {
        b = b.header("authorization", format!("Bearer {k}"));
    }
    let req = match body {
        Some(v) => b
            .header("content-type", "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    send(app, req).await
}

/// Polls `uri` until `done(json)` or the timeout.
pub async fn poll(
    app: &Router,
    uri: &str,
    done: impl Fn(&serde_json::Value) -> bool,
) -> serde_json::Value {
    let deadline = tokio::time::Instant::now() + T;
    loop {
        let r = call(app, "GET", uri, None).await;
        assert_eq!(
            r.status,
            StatusCode::OK,
            "{uri}: {}",
            String::from_utf8_lossy(&r.bytes)
        );
        let v = r.json();
        if done(&v) {
            return v;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out polling {uri}: {v}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A small valid PNG.
pub fn png(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
    });
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}
