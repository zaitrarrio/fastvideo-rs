//! `ServeCtx`: the state every adapter route shares (design §2.1).
//!
//! The engine is reached through [`EngineGate`], a narrow seam the binary
//! (WP-10) implements over `fastvideo-engine-service::EngineService`: the
//! generic handlers only need the model table, admission, submit and cancel.
//! Engine progress flows back through [`crate::events::apply_event`].

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use fastvideo_protocol::{
    ApiError, GenerationRequest, Job, JobId, JobStore, ModelCaps, ProtocolId, UrlSigner, ViewCtx,
};
use time::OffsetDateTime;
use url::Url;

use crate::artifacts::{files_router, ArtifactStore, FileServer, LocalArtifactStore, LocalUrls, UrlKey};
use crate::auth::Auth;
use crate::callback::{CallbackRender, CallbackSender};
use crate::ingest::{DefaultProber, Ingestor, Prober};
use crate::store::MemJobStore;
use crate::uploads::{uploads_router_with_clock, UploadStore};

/// The engine as the HTTP layer sees it.
#[async_trait::async_trait]
pub trait EngineGate: Send + Sync + 'static {
    /// Every servable model's caps.
    fn models(&self) -> Vec<ModelCaps>;
    /// `ServeConfig.aliases` lookup (e.g. `MiniMax-H3-Max` -> `fasth3`).
    fn alias(&self, _name: &str) -> Option<String> {
        None
    }
    /// Admission before any work: `Loading` while warming, `QueueFull`.
    fn admit(&self) -> Result<(), ApiError> {
        Ok(())
    }
    /// Queues a stored `Queued` job. Progress comes back through
    /// [`crate::events::apply_event`]. An error removes the job again.
    async fn submit(&self, job: &Job) -> Result<(), ApiError>;
    /// Trips the job's cancel token (queued or running). Returns whether the
    /// engine knew the job.
    async fn cancel(&self, id: JobId) -> bool;
}

/// Moderation hook (design §10 R8). The default accepts everything.
pub trait SafetyFilter: Send + Sync + 'static {
    /// `ContentFiltered` refuses the request.
    fn check_request(&self, _req: &GenerationRequest) -> Result<(), ApiError> {
        Ok(())
    }
}

/// Accepts everything.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoSafetyFilter;
impl SafetyFilter for NoSafetyFilter {}

/// Server-wide settings serve-kit needs.
#[derive(Clone, Debug)]
pub struct ServeConfig {
    /// Base for every URL we hand out (`server.public_base_url`).
    pub public_base: Url,
    /// `server.state_dir`: `jobs/`, `artifacts/`, `uploads/`, `inputs/`.
    pub state_dir: PathBuf,
    /// Default signed-URL lifetime (`artifacts.url_ttl_s`, 24 h).
    pub url_ttl: Duration,
    /// Per-API retention overrides (defaults: `ProtocolId::default_retention`).
    pub retention: BTreeMap<ProtocolId, Duration>,
    /// Request body cap (`limits.body_max_mb`, 64 MB).
    pub body_max_bytes: usize,
    /// How long sync endpoints wait for a result.
    pub sync_timeout: Duration,
}

impl ServeConfig {
    pub fn new(public_base: Url, state_dir: impl Into<PathBuf>) -> Self {
        Self {
            public_base,
            state_dir: state_dir.into(),
            url_ttl: Duration::from_secs(24 * 3600),
            retention: BTreeMap::new(),
            body_max_bytes: 64 * 1024 * 1024,
            sync_timeout: Duration::from_secs(600),
        }
    }
    pub fn retention(&self, p: ProtocolId) -> Duration {
        self.retention
            .get(&p)
            .copied()
            .unwrap_or_else(|| p.default_retention())
    }
}

struct Inner {
    cfg: ServeConfig,
    auth: Auth,
    jobs: Arc<dyn JobStore>,
    artifacts: Arc<dyn ArtifactStore>,
    uploads: Arc<UploadStore>,
    ingest: Ingestor,
    engine: Arc<dyn EngineGate>,
    callbacks: Arc<CallbackSender>,
    renderers: HashMap<ProtocolId, Arc<dyn CallbackRender>>,
    safety: Arc<dyn SafetyFilter>,
    files: Arc<FileServer>,
    now: Arc<dyn Fn() -> OffsetDateTime + Send + Sync>,
}

/// Shared state for every route. Cheap to clone.
#[derive(Clone)]
pub struct ServeCtx {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for ServeCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServeCtx")
            .field("public_base", &self.inner.cfg.public_base.as_str())
            .field("state_dir", &self.inner.cfg.state_dir)
            .finish_non_exhaustive()
    }
}

/// Builds a [`ServeCtx`]; unset parts get local defaults under `state_dir`.
pub struct ServeCtxBuilder {
    cfg: ServeConfig,
    engine: Arc<dyn EngineGate>,
    auth: Auth,
    url_key: Option<UrlKey>,
    jobs: Option<Arc<dyn JobStore>>,
    artifacts: Option<Arc<dyn ArtifactStore>>,
    callbacks: Option<Arc<CallbackSender>>,
    renderers: HashMap<ProtocolId, Arc<dyn CallbackRender>>,
    safety: Arc<dyn SafetyFilter>,
    prober: Arc<dyn Prober>,
    now: Arc<dyn Fn() -> OffsetDateTime + Send + Sync>,
    artifacts_root: Option<PathBuf>,
}

impl ServeCtxBuilder {
    /// Where local artifacts live and `/files` serves them from (default
    /// `state_dir/artifacts`); processes on one host can share it.
    pub fn artifacts_root(mut self, p: impl Into<PathBuf>) -> Self {
        self.artifacts_root = Some(p.into());
        self
    }
    pub fn auth(mut self, a: Auth) -> Self {
        self.auth = a;
        self
    }
    /// `FV_URL_SIGNING_KEY` (default: random per process).
    pub fn url_key(mut self, k: UrlKey) -> Self {
        self.url_key = Some(k);
        self
    }
    /// Job store (default: a durable `MemJobStore` under `state_dir/jobs`).
    pub fn jobs(mut self, s: Arc<dyn JobStore>) -> Self {
        self.jobs = Some(s);
        self
    }
    /// Artifact store (default: local under `state_dir/artifacts`).
    pub fn artifacts(mut self, a: Arc<dyn ArtifactStore>) -> Self {
        self.artifacts = Some(a);
        self
    }
    pub fn callbacks(mut self, c: Arc<CallbackSender>) -> Self {
        self.callbacks = Some(c);
        self
    }
    /// Callback body renderer for one API (MiniMax, fal).
    pub fn renderer(mut self, p: ProtocolId, r: Arc<dyn CallbackRender>) -> Self {
        self.renderers.insert(p, r);
        self
    }
    pub fn safety(mut self, s: Arc<dyn SafetyFilter>) -> Self {
        self.safety = s;
        self
    }
    pub fn prober(mut self, p: Arc<dyn Prober>) -> Self {
        self.prober = p;
        self
    }
    /// Clock override (tests).
    pub fn clock(mut self, now: Arc<dyn Fn() -> OffsetDateTime + Send + Sync>) -> Self {
        self.now = now;
        self
    }

    pub async fn build(self) -> std::io::Result<ServeCtx> {
        let dir = &self.cfg.state_dir;
        let key = self.url_key.unwrap_or_else(UrlKey::random);
        let urls = LocalUrls { public_base: self.cfg.public_base.clone(), key: key.clone() };
        let art_root = self.artifacts_root.clone().unwrap_or_else(|| dir.join("artifacts"));
        let up_root = dir.join("uploads");
        tokio::fs::create_dir_all(&art_root).await?;
        let artifacts: Arc<dyn ArtifactStore> = match self.artifacts {
            Some(a) => a,
            None => Arc::new(LocalArtifactStore::new(&art_root, urls.clone())),
        };
        let jobs: Arc<dyn JobStore> = match self.jobs {
            Some(j) => j,
            None => Arc::new(
                MemJobStore::open(dir.join("jobs"), (self.now)())
                    .await?
                    .with_artifacts(artifacts.clone())
                    .with_inputs_root(dir.join("inputs")),
            ),
        };
        let uploads = Arc::new(UploadStore::new(&up_root, urls)?);
        let callbacks = self.callbacks.unwrap_or_else(|| {
            Arc::new(CallbackSender::new(CallbackSender::default_transport(), None))
        });
        Ok(ServeCtx {
            inner: Arc::new(Inner {
                files: Arc::new(FileServer { key, roots: vec![art_root, up_root] }),
                ingest: Ingestor::new(Some(uploads.clone()), self.prober),
                cfg: self.cfg,
                auth: self.auth,
                jobs,
                artifacts,
                uploads,
                engine: self.engine,
                callbacks,
                renderers: self.renderers,
                safety: self.safety,
                now: self.now,
            }),
        })
    }
}

impl ServeCtx {
    pub fn builder(cfg: ServeConfig, engine: Arc<dyn EngineGate>) -> ServeCtxBuilder {
        ServeCtxBuilder {
            cfg,
            engine,
            auth: Auth::default(),
            url_key: None,
            jobs: None,
            artifacts: None,
            callbacks: None,
            renderers: HashMap::new(),
            safety: Arc::new(NoSafetyFilter),
            prober: Arc::new(DefaultProber::default()),
            now: Arc::new(OffsetDateTime::now_utc),
            artifacts_root: None,
        }
    }

    pub fn config(&self) -> &ServeConfig {
        &self.inner.cfg
    }
    pub fn auth(&self) -> &Auth {
        &self.inner.auth
    }
    pub fn jobs(&self) -> &Arc<dyn JobStore> {
        &self.inner.jobs
    }
    pub fn artifacts(&self) -> &Arc<dyn ArtifactStore> {
        &self.inner.artifacts
    }
    pub fn uploads(&self) -> &Arc<UploadStore> {
        &self.inner.uploads
    }
    pub fn ingestor(&self) -> &Ingestor {
        &self.inner.ingest
    }
    pub fn engine(&self) -> &Arc<dyn EngineGate> {
        &self.inner.engine
    }
    pub fn callbacks(&self) -> &Arc<CallbackSender> {
        &self.inner.callbacks
    }
    pub fn safety(&self) -> &Arc<dyn SafetyFilter> {
        &self.inner.safety
    }
    pub fn now(&self) -> OffsetDateTime {
        (self.inner.now)()
    }
    /// Staged inputs of job `id`.
    pub fn inputs_dir(&self, id: JobId) -> PathBuf {
        self.inner.cfg.state_dir.join("inputs").join(id.to_string())
    }
    /// A scratch dir for engine outputs of job `id`.
    pub fn outputs_dir(&self, id: JobId) -> PathBuf {
        self.inner.cfg.state_dir.join("outputs").join(id.to_string())
    }
    pub fn url_signer(&self) -> &dyn UrlSigner {
        self.inner.artifacts.signer()
    }

    /// A [`ViewCtx`] for rendering at the current time.
    pub fn view_ctx(&self, with_logs: bool) -> ViewCtx<'_> {
        ViewCtx {
            now: self.now(),
            urls: self.url_signer(),
            public_base: &self.inner.cfg.public_base,
            with_logs,
        }
    }

    /// `GET /files/{id}/{name}` and `PUT /uploads/{token}` (design §9).
    pub fn routes<S: Clone + Send + Sync + 'static>(&self) -> Router<S> {
        files_router(self.inner.files.clone())
            .merge(uploads_router_with_clock(self.inner.uploads.clone(), self.inner.now.clone()))
    }

    /// Sends the job's callback for its current state, if it has one and its
    /// API registered a renderer.
    pub fn notify(&self, job: &Job) {
        if job.callback.is_none() {
            return;
        }
        let Some(r) = self.inner.renderers.get(&job.protocol) else {
            return;
        };
        if let Some(body) = r.callback_body(job, &self.view_ctx(false)) {
            self.inner.callbacks.dispatch(job.clone(), body);
        }
    }
}
