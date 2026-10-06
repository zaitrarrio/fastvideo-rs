//! The edge on the host (docs/serve/edge-control-plane.md): the family
//! objects (the pure scheduler of `fastvideo-dispatch-proto`, as the
//! Durable Object hosts it), the registry and the public front in one
//! process. The production edge is the Cloudflare Worker in
//! `crates/fastvideo-edge`; this host runs the same routing decisions
//! (`fastvideo_dispatch_proto::front`) for the integration tests, the compat
//! suites (`FV_COMPAT_EDGE=1`) and local use (`fv-edge-local`).
//!
//! Routes:
//!
//! | route | what |
//! |---|---|
//! | `/families/{f}/connect` (ws), `…/enqueue`, `…/cancel/{job}`, `…/status`, `…/metrics`, `…/sessions[/{id}/{renew,release}]`, `…/jobs/{id}/wait` | the family objects (internal token; status and metrics also the admin token) |
//! | `GET /registry` | every family's workers and load (internal token) |
//! | everything else | the public front: classify, authenticate, apply quotas, forward to a front (or answer here) |
//!
//! Keys are serve-kit's [`KeyStore`] (D1 or memory): the same table, the
//! same `/fv/v1/admin/keys` shapes as fv-serve's.

// Handler helpers return a ready `Response` as their error (early return).
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_dispatch_proto::front::{
    self, classify, is_submit, plan_forward, reactor_owner, scan_model, Class, DirectorOp, EdgeRoute, Quotas, ReactorOp, Registry, Reply, Scan, SessionBinding,
    StreamOp, Target, Verdict, EDGE_AUTH_HEADER, REQUEST_ID_HEADER,
};
use fastvideo_dispatch_proto::sched::{Admit, Cfg, JobRec, Out, Sched, SessionRec, UploadRec, WorkerRec};
use fastvideo_dispatch_proto::{self as proto, DoMsg, EnqueueReq, JobWait, Part, PartUrl, SessionGrant, SessionReq, WorkerMsg};
use fastvideo_serve_kit::{AdminToken, KeyRing, KeyStore};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, watch};

/// Where direct uploads go (the Worker's R2 binding in production; an S3
/// mock in tests). Without one, workers store outputs through their own
/// artifact store.
#[async_trait::async_trait]
pub trait Outputs: Send + Sync + 'static {
    fn bucket(&self) -> String;
    fn part_url(&self, key: &str, upload_id: &str, n: u16) -> String;
    async fn create(&self, key: &str) -> Result<String, String>;
    async fn complete(&self, key: &str, upload_id: &str, parts: &[Part]) -> Result<u64, String>;
    async fn abort(&self, key: &str, upload_id: &str);
}

/// The host's settings.
#[derive(Clone)]
pub struct EdgeHostCfg {
    /// Workers' and fronts' shared secret (`FV_INTERNAL_TOKEN`).
    pub internal_token: String,
    /// The cluster's admin token (`/fv/v1/admin/*`, the families view).
    pub admin_token: Option<String>,
    /// `FV_API_KEYS` (SHA-256 list).
    pub static_keys: KeyRing,
    /// Minted keys.
    pub keys: Arc<KeyStore>,
    /// `auth: none` (a demo cluster): every caller is anonymous and allowed.
    pub auth_none: bool,
    pub sched: Cfg,
    pub quotas: Quotas,
    /// The cluster's Reactor model when several fronts have one.
    pub reactor_model: Option<String>,
    /// Lease of an admitted session without a renew.
    pub session_ttl_ms: i64,
    /// WHIP ingest offers: answer 307 to the admitted worker with a session
    /// capability (`FV_EDGE_WHIP=redirect`) instead of proxying the offer.
    pub whip_redirect: bool,
    pub outputs: Option<Arc<dyn Outputs>>,
}

impl std::fmt::Debug for EdgeHostCfg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeHostCfg").field("auth_none", &self.auth_none).finish_non_exhaustive()
    }
}

impl EdgeHostCfg {
    pub fn new(internal_token: impl Into<String>, keys: Arc<KeyStore>) -> Self {
        Self {
            internal_token: internal_token.into(),
            admin_token: None,
            static_keys: KeyRing::default(),
            keys,
            auth_none: false,
            sched: Cfg::default(),
            quotas: Quotas::default(),
            reactor_model: None,
            session_ttl_ms: 1_800_000,
            whip_redirect: false,
            outputs: None,
        }
    }
}

#[derive(Default)]
struct Objects {
    scheds: HashMap<String, Sched>,
    conns: HashMap<(String, String), (u64, mpsc::UnboundedSender<String>)>,
    next_conn: u64,
    waiters: HashMap<String, oneshot::Sender<Result<SessionGrant, String>>>,
}

/// The edge on the host (see the module docs).
pub struct EdgeHost {
    cfg: EdgeHostCfg,
    objects: Mutex<Objects>,
    /// Bumped by a "deploy": every worker socket drops.
    deploy: watch::Sender<u64>,
    /// Bumped on every job change (job waits).
    jobs: watch::Sender<u64>,
    key_epoch: AtomicU64,
    bindings: Mutex<HashMap<String, SessionBinding>>,
    /// Fixed one-minute windows: key or address → (window, count).
    rate: Mutex<HashMap<String, (i64, u32)>>,
    http: reqwest::Client,
    admin: Option<AdminToken>,
}

impl std::fmt::Debug for EdgeHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeHost").finish_non_exhaustive()
    }
}

fn now_ms() -> i64 {
    (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64
}

fn err(status: u16, kind: &str, message: impl Into<String>) -> Response {
    (StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(json!({"error": {"kind": kind, "message": message.into()}}))).into_response()
}

fn reply(r: Reply) -> Response {
    let mut resp = (StatusCode::from_u16(r.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(r.body())).into_response();
    if let Some(s) = r.retry_after {
        resp.headers_mut().insert("retry-after", HeaderValue::from(s));
    }
    resp
}

/// Request headers never forwarded to a front.
const DROP_REQ: &[&str] = &["host", "connection", "keep-alive", "transfer-encoding", "content-length", "upgrade", "proxy-connection", "te", "trailer"];
/// Response headers not copied back.
const DROP_RESP: &[&str] = &["connection", "keep-alive", "transfer-encoding", "content-length", "upgrade"];

impl EdgeHost {
    /// Serves the host on `listener`; returns it and its base URL.
    pub async fn start(cfg: EdgeHostCfg, listener: tokio::net::TcpListener) -> (Arc<Self>, String) {
        let base = format!("http://{}", listener.local_addr().expect("bound"));
        let (deploy, _) = watch::channel(0);
        let (jobs, _) = watch::channel(0);
        let http = reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none()).connect_timeout(Duration::from_secs(5)).build().unwrap_or_default();
        let admin = cfg.admin_token.as_deref().filter(|t| !t.is_empty()).map(AdminToken::from_secret);
        let h = Arc::new(Self {
            cfg,
            objects: Mutex::default(),
            deploy,
            jobs,
            key_epoch: AtomicU64::new(0),
            bindings: Mutex::default(),
            rate: Mutex::default(),
            http,
            admin,
        });
        let app = Router::new()
            .route("/families/{f}/connect", get(connect))
            .route("/families/{f}/enqueue", post(enqueue))
            .route("/families/{f}/cancel/{job}", post(cancel))
            .route("/families/{f}/status", get(status))
            .route("/families/{f}/metrics", get(metrics_route))
            .route("/families/{f}/sessions", post(admit_route))
            .route("/families/{f}/sessions/{id}/{op}", post(session_op))
            .route("/families/{f}/jobs/{id}/wait", get(job_wait))
            .route("/registry", get(registry_route))
            .fallback(public)
            .layer(axum::extract::DefaultBodyLimit::disable())
            .with_state(h.clone());
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        // The alarm.
        let t = h.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(50)).await;
                let now = now_ms();
                let objs: Vec<String> = t.lock().scheds.keys().cloned().collect();
                for o in objs {
                    let out = {
                        let mut g = t.lock();
                        let Some(s) = g.scheds.get_mut(&o) else { continue };
                        if s.next_wake(now).is_some_and(|w| w <= now) {
                            s.tick(now)
                        } else {
                            continue;
                        }
                    };
                    t.apply(&o, out);
                }
            }
        });
        (h, base)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Objects> {
        self.objects.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A Worker deploy: every socket drops, every object reloads from its rows.
    pub fn redeploy(&self) {
        let now = now_ms();
        let mut g = self.lock();
        let objs: Vec<String> = g.scheds.keys().cloned().collect();
        for o in objs {
            let s = g.scheds.remove(&o).expect("listed");
            let jobs: Vec<JobRec> = s.jobs().cloned().collect();
            let workers: Vec<WorkerRec> = s.workers().cloned().collect();
            let sessions: Vec<SessionRec> = s.sessions().cloned().collect();
            let uploads: Vec<UploadRec> = s.uploads().cloned().collect();
            let mut s = Sched::restore_all(o.clone(), self.cfg.sched.clone(), jobs, workers, sessions, uploads);
            let ids: Vec<String> = s.workers().filter(|w| w.connected).map(|w| w.worker_id.clone()).collect();
            for id in ids {
                s.disconnect(&id, now);
            }
            g.scheds.insert(o, s);
        }
        g.conns.clear();
        drop(g);
        self.deploy.send_modify(|v| *v += 1);
    }

    /// Runs `f` on object `obj`'s scheduler, then its effects.
    pub fn with<T>(self: &Arc<Self>, obj: &str, f: impl FnOnce(&mut Sched) -> (T, Vec<Out>)) -> T {
        let (t, out) = {
            let mut g = self.lock();
            let cfg = self.cfg.sched.clone();
            let s = g.scheds.entry(obj.to_owned()).or_insert_with(|| Sched::new(obj, cfg));
            f(s)
        };
        self.apply(obj, out);
        t
    }

    fn send(&self, obj: &str, worker: &str, msg: &DoMsg) {
        let g = self.lock();
        if let Some((_, tx)) = g.conns.get(&(obj.to_owned(), worker.to_owned())) {
            let _ = tx.send(serde_json::to_string(msg).unwrap_or_default());
        }
    }

    fn apply(self: &Arc<Self>, obj: &str, out: Vec<Out>) {
        for o in out {
            match o {
                Out::Send { worker, msg } => self.send(obj, &worker, &msg),
                Out::Close { worker } => {
                    self.lock().conns.remove(&(obj.to_owned(), worker));
                }
                Out::PushSpilled { .. } => {}
                Out::SessionReady { session_id, result } => {
                    if let Some(w) = self.lock().waiters.remove(&session_id) {
                        let _ = w.send(result);
                    }
                }
                Out::Grant { worker, req, job_id, key, upload_id, from, count, expires_ms } => {
                    let msg = match &self.cfg.outputs {
                        Some(o) => DoMsg::UploadGrant {
                            req,
                            job_id,
                            part_urls: (from..from.saturating_add(count)).map(|n| PartUrl { n, url: o.part_url(&key, &upload_id, n) }).collect(),
                            upload_id,
                            key,
                            bucket: o.bucket(),
                            expires_ms,
                            error: None,
                        },
                        None => DoMsg::UploadGrant {
                            req,
                            job_id,
                            upload_id: String::new(),
                            key: String::new(),
                            bucket: String::new(),
                            part_urls: Vec::new(),
                            expires_ms: 0,
                            error: Some("no outputs store on this edge".into()),
                        },
                    };
                    self.send(obj, &worker, &msg);
                }
                Out::CreateUpload { req, key, parts, .. } => {
                    let (h, obj) = (self.clone(), obj.to_owned());
                    tokio::spawn(async move {
                        let r = match &h.cfg.outputs {
                            Some(o) => o.create(&key).await,
                            None => Err("no outputs store on this edge".to_owned()),
                        };
                        h.with(&obj, |s| ((), s.upload_created(&key, req, parts, r, now_ms())));
                    });
                }
                Out::CompleteUpload { req, key, upload_id, parts, .. } => {
                    let (h, obj) = (self.clone(), obj.to_owned());
                    tokio::spawn(async move {
                        let r = match &h.cfg.outputs {
                            Some(o) => o.complete(&key, &upload_id, &parts).await,
                            None => Err("no outputs store on this edge".to_owned()),
                        };
                        h.with(&obj, |s| ((), s.upload_completed(&key, req, r, now_ms())));
                    });
                }
                Out::AbortUpload { key, upload_id } => {
                    if let Some(o) = self.cfg.outputs.clone() {
                        tokio::spawn(async move { o.abort(&key, &upload_id).await });
                    }
                }
            }
        }
        let changed = {
            let mut g = self.lock();
            g.scheds.get_mut(obj).map(|s| !s.take_dirty().jobs.is_empty()).unwrap_or(false)
        };
        if changed {
            self.jobs.send_modify(|v| *v += 1);
        }
    }

    /// Every family's view.
    pub fn registry(self: &Arc<Self>) -> Registry {
        let now = now_ms();
        let statuses: Vec<(String, proto::PoolStatus)> = {
            let g = self.lock();
            g.scheds.iter().filter_map(|(o, s)| o.strip_prefix("family:").map(|f| (f.to_owned(), s.status(now)))).collect()
        };
        Registry::from_statuses(statuses, self.key_epoch.load(Ordering::SeqCst), now)
    }

    fn internal(&self, h: &HeaderMap) -> bool {
        h.get(proto::TOKEN_HEADER).is_some_and(|v| front::ct_eq(v.as_bytes(), self.cfg.internal_token.as_bytes()))
    }

    fn is_admin(&self, h: &HeaderMap) -> bool {
        self.admin.as_ref().is_some_and(|a| a.check_headers(h))
    }

    /// The identity verdict for a request's `Authorization`.
    fn verdict(&self, h: &HeaderMap) -> Verdict {
        let mut v = Verdict { v: 1, ..Verdict::default() };
        if self.cfg.auth_none {
            return v;
        }
        let Some((scheme, key)) = h.get("authorization").and_then(|v| v.to_str().ok()).and_then(front::parse_authorization) else {
            return v;
        };
        v.presented = true;
        v.scheme = Some(scheme.to_owned());
        // Never the admin token as a client key.
        if self.admin.as_ref().is_some_and(|a| a.check(key)) {
            return v;
        }
        let owner = self.cfg.static_keys.check(key).or_else(|| fastvideo_serve_kit::keys::KeyCheck::check(self.cfg.keys.as_ref(), key));
        if let Some(o) = owner {
            v.valid = true;
            v.key = Some(o.0);
        }
        v
    }

    /// Counts one request in `who`'s minute; whether it is over `limit`.
    fn over(&self, who: &str, limit: u32) -> bool {
        if limit == 0 {
            return false;
        }
        let win = now_ms() / 60_000;
        let mut g = self.rate.lock().unwrap_or_else(|p| p.into_inner());
        let e = g.entry(who.to_owned()).or_insert((win, 0));
        if e.0 != win {
            *e = (win, 0);
        }
        e.1 += 1;
        e.1 > limit
    }

    /// Forwards a request to a front.
    #[allow(clippy::too_many_arguments)]
    async fn forward(&self, url: &str, method: &Method, path_q: &str, headers: &HeaderMap, body: reqwest::Body, verdict: &Verdict, timeout: Option<Duration>) -> Response {
        let m = reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET);
        let mut r = self.http.request(m, format!("{}{path_q}", url.trim_end_matches('/')));
        for (k, v) in headers {
            let n = k.as_str();
            if DROP_REQ.contains(&n) || n.starts_with("x-fv-") || n == "authorization" {
                continue;
            }
            r = r.header(n, v.as_bytes());
        }
        // The admin token goes on (the front checks it for its admin routes).
        if self.is_admin(headers) {
            if let Some(a) = headers.get("authorization") {
                r = r.header("authorization", a.as_bytes());
            }
        }
        r = r.header(proto::TOKEN_HEADER, &self.cfg.internal_token).header(EDGE_AUTH_HEADER, verdict.header());
        if let Some(t) = timeout {
            r = r.timeout(t);
        }
        match r.body(body).send().await {
            Ok(resp) => {
                let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
                let mut hs = HeaderMap::new();
                for (k, v) in resp.headers() {
                    if DROP_RESP.contains(&k.as_str()) {
                        continue;
                    }
                    if let (Ok(n), Ok(val)) = (HeaderName::from_bytes(k.as_str().as_bytes()), HeaderValue::from_bytes(v.as_bytes())) {
                        hs.append(n, val);
                    }
                }
                let mut out = Response::new(Body::from_stream(resp.bytes_stream()));
                *out.status_mut() = status;
                *out.headers_mut() = hs;
                out
            }
            Err(e) => {
                let e = e.without_url().to_string();
                tracing::warn!(url, error = %e, "edge: the front did not answer");
                err(502, "loading", format!("the worker did not answer: {e}"))
            }
        }
    }

    async fn forward_json(&self, url: &str, path: &str, headers: &HeaderMap, body: Bytes, verdict: &Verdict) -> Result<(u16, HeaderMap, Value, Bytes), Response> {
        let resp = self.forward(url, &Method::POST, path, headers, body.into(), verdict, Some(Duration::from_secs(60))).await;
        let status = resp.status().as_u16();
        let hs = resp.headers().clone();
        let bytes = axum::body::to_bytes(resp.into_body(), 64 << 20).await.map_err(|e| err(502, "loading", e.to_string()))?;
        let v = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        Ok((status, hs, v, bytes))
    }

    /// A family object's admission (`kind`, `owner`).
    async fn admit(self: &Arc<Self>, family: &str, kind: &str, model: Option<String>, owner: Option<String>) -> Result<SessionGrant, (u16, String)> {
        let obj = format!("family:{family}");
        let req = SessionReq { session_id: None, model, kind: kind.to_owned(), owner, ttl_ms: self.cfg.session_ttl_ms };
        let (tx, rx) = oneshot::channel();
        let a = {
            let (a, out) = {
                let mut g = self.lock();
                let cfg = self.cfg.sched.clone();
                let (a, out) = g.scheds.entry(obj.clone()).or_insert_with(|| Sched::new(&obj, cfg)).admit(req, now_ms());
                if let Admit::Pending(id) = &a {
                    g.waiters.insert(id.clone(), tx);
                }
                (a, out)
            };
            self.apply(&obj, out);
            a
        };
        match a {
            Admit::Granted(g) => Ok(g),
            Admit::Refused(m) => Err((429, m)),
            Admit::Pending(id) => match tokio::time::timeout(Duration::from_secs(20), rx).await {
                Ok(Ok(Ok(g))) => Ok(g),
                Ok(Ok(Err(m))) => Err((429, m)),
                _ => {
                    self.with(&obj, |s| s.release(&id, now_ms()));
                    Err((503, "no worker answered the session offer in time".into()))
                }
            },
        }
    }

    /// Director sessions end over their data channel, not HTTP: before
    /// refusing a session for lack of a GPU, ask the workers whether the
    /// director sessions bound in `family` still live, and release the
    /// ended ones. Whether any was released.
    async fn reclaim(self: &Arc<Self>, family: &str) -> bool {
        let held: Vec<(String, SessionBinding)> = self
            .bindings
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|(k, b)| b.family == family && k.starts_with("director:"))
            .map(|(k, b)| (k.clone(), b.clone()))
            .collect();
        let mut freed = false;
        for (alias, b) in held {
            let sid = alias.trim_start_matches("director:");
            let r = self
                .http
                .post(format!("{}/wma/session/heartbeat", b.endpoint.trim_end_matches('/')))
                .header(proto::TOKEN_HEADER, &self.cfg.internal_token)
                .header(EDGE_AUTH_HEADER, Verdict { v: 1, key: b.owner.clone(), presented: true, valid: b.owner.is_some(), scheme: Some("key".into()), deny: None }.header())
                .json(&json!({"session_id": sid}))
                .timeout(Duration::from_secs(5))
                .send()
                .await;
            let alive = match r {
                Ok(r) => r.json::<Value>().await.ok().and_then(|v| v.get("alive").and_then(Value::as_bool)).unwrap_or(true),
                Err(_) => false,
            };
            if !alive {
                self.unbind(&alias);
                self.release(&b);
                freed = true;
            }
        }
        freed
    }

    /// [`EdgeHost::admit`], reclaiming ended director sessions once when
    /// no GPU has room.
    async fn admit_or_reclaim(self: &Arc<Self>, family: &str, kind: &str, model: Option<String>, owner: Option<String>) -> Result<SessionGrant, (u16, String)> {
        match self.admit(family, kind, model.clone(), owner.clone()).await {
            Err((429, m)) => {
                if !self.reclaim(family).await {
                    return Err((429, m));
                }
                // The worker reports its freed slot in its next `slots` frame.
                let mut last = Err((429, m));
                for _ in 0..25 {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    last = self.admit(family, kind, model.clone(), owner.clone()).await;
                    if !matches!(last, Err((429, _))) {
                        break;
                    }
                }
                last
            }
            r => r,
        }
    }

    fn release(self: &Arc<Self>, b: &SessionBinding) {
        let obj = format!("family:{}", b.family);
        self.with(&obj, |s| s.release(&b.session_id, now_ms()));
    }

    fn renew(self: &Arc<Self>, b: &SessionBinding) -> bool {
        let obj = format!("family:{}", b.family);
        self.with(&obj, |s| (s.renew(&b.session_id, 0, now_ms()), Vec::new())).is_some()
    }

    fn bind(&self, alias: &str, b: SessionBinding) {
        self.bindings.lock().unwrap_or_else(|p| p.into_inner()).insert(alias.to_owned(), b);
    }

    fn binding(&self, alias: &str) -> Option<SessionBinding> {
        self.bindings.lock().unwrap_or_else(|p| p.into_inner()).get(alias).cloned()
    }

    fn unbind(&self, alias: &str) -> Option<SessionBinding> {
        self.bindings.lock().unwrap_or_else(|p| p.into_inner()).remove(alias)
    }

    /// One front per family, `GET path` with the verdict, the JSON answers.
    async fn fan_out(&self, fronts: Vec<String>, path: &str, headers: &HeaderMap, verdict: &Verdict) -> Vec<Value> {
        let calls = fronts.into_iter().map(|url| async move {
            let r = self.forward(&url, &Method::GET, path, headers, reqwest::Body::from(Vec::new()), verdict, Some(Duration::from_secs(15))).await;
            if !r.status().is_success() {
                return None;
            }
            let b = axum::body::to_bytes(r.into_body(), 16 << 20).await.ok()?;
            serde_json::from_slice::<Value>(&b).ok()
        });
        futures::future::join_all(calls).await.into_iter().flatten().collect()
    }
}

// ------------------------------------------------------------- family objects

fn object(f: &str) -> Result<String, Response> {
    if proto::valid_id(f) {
        Ok(format!("family:{f}"))
    } else {
        Err(err(400, "invalid_request", "invalid family id"))
    }
}

async fn connect(State(h): State<Arc<EdgeHost>>, Path(f): Path<String>, hs: HeaderMap, ws: WebSocketUpgrade) -> Response {
    if !h.internal(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    let obj = match object(&f) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let Some(worker) = hs.get(proto::WORKER_HEADER).and_then(|v| v.to_str().ok()).filter(|w| proto::valid_id(w)).map(str::to_owned) else {
        return err(400, "invalid_request", "x-fv-worker-id is missing or invalid");
    };
    ws.on_upgrade(move |socket| run_socket(h, obj, worker, socket))
}

async fn run_socket(h: Arc<EdgeHost>, obj: String, worker: String, socket: WebSocket) {
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let conn = {
        let mut g = h.lock();
        g.next_conn += 1;
        let c = g.next_conn;
        g.conns.insert((obj.clone(), worker.clone()), (c, tx));
        c
    };
    let mut deploy = h.deploy.subscribe();
    loop {
        tokio::select! {
            _ = deploy.changed() => return,
            m = rx.recv() => {
                let Some(m) = m else { return };
                if sink.send(Message::Text(m.into())).await.is_err() {
                    break;
                }
            }
            f = stream.next() => {
                let Some(Ok(Message::Text(t))) = f else { break };
                if t.as_str() == proto::PING {
                    let _ = sink.send(Message::Text(proto::PONG.into())).await;
                    continue;
                }
                let Ok(msg) = serde_json::from_str::<WorkerMsg>(t.as_str()) else { continue };
                h.with(&obj, |s| ((), s.on_msg(&worker, msg, now_ms())));
            }
        }
    }
    let gone = {
        let mut g = h.lock();
        let key = (obj.clone(), worker.clone());
        let mine = g.conns.get(&key).is_some_and(|(c, _)| *c == conn);
        if mine {
            g.conns.remove(&key);
        }
        mine
    };
    if gone {
        h.with(&obj, |s| ((), s.disconnect(&worker, now_ms())));
    }
}

async fn enqueue(State(h): State<Arc<EdgeHost>>, Path(f): Path<String>, hs: HeaderMap, Json(req): Json<EnqueueReq>) -> Response {
    if !h.internal(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    let obj = match object(&f) {
        Ok(o) => o,
        Err(r) => return r,
    };
    if !proto::valid_id(&req.job_id) {
        return err(400, "invalid_request", "invalid job id");
    }
    let resp = h.with(&obj, |s| s.enqueue(req, now_ms()));
    (StatusCode::ACCEPTED, Json(resp)).into_response()
}

async fn cancel(State(h): State<Arc<EdgeHost>>, Path((f, job)): Path<(String, String)>, hs: HeaderMap) -> Response {
    if !h.internal(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    let obj = match object(&f) {
        Ok(o) => o,
        Err(r) => return r,
    };
    match h.with(&obj, |s| s.cancel(&job, now_ms())) {
        Some(st) => Json(json!({"job_id": job, "state": st})).into_response(),
        None => err(404, "not_found", "unknown job"),
    }
}

async fn status(State(h): State<Arc<EdgeHost>>, Path(f): Path<String>, hs: HeaderMap) -> Response {
    if !h.internal(&hs) && !h.is_admin(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    let obj = match object(&f) {
        Ok(o) => o,
        Err(r) => return r,
    };
    Json(h.with(&obj, |s| (s.status(now_ms()), Vec::new()))).into_response()
}

async fn metrics_route(State(h): State<Arc<EdgeHost>>, Path(f): Path<String>, hs: HeaderMap) -> Response {
    if !h.internal(&hs) && !h.is_admin(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    let obj = match object(&f) {
        Ok(o) => o,
        Err(r) => return r,
    };
    Json(h.with(&obj, |s| (s.metrics(now_ms()), Vec::new()))).into_response()
}

async fn admit_route(State(h): State<Arc<EdgeHost>>, Path(f): Path<String>, hs: HeaderMap, Json(req): Json<SessionReq>) -> Response {
    if !h.internal(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    if !proto::valid_id(&f) {
        return err(400, "invalid_request", "invalid family id");
    }
    match h.admit(&f, &req.kind, req.model, req.owner).await {
        Ok(g) => Json(g).into_response(),
        Err((st, m)) => err(st, if st == 429 { "queue_full" } else { "loading" }, m),
    }
}

async fn session_op(State(h): State<Arc<EdgeHost>>, Path((f, id, op)): Path<(String, String, String)>, hs: HeaderMap) -> Response {
    if !h.internal(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    let obj = match object(&f) {
        Ok(o) => o,
        Err(r) => return r,
    };
    match op.as_str() {
        "renew" => match h.with(&obj, |s| (s.renew(&id, 0, now_ms()), Vec::new())) {
            Some(g) => Json(g).into_response(),
            None => err(404, "not_found", "no live session"),
        },
        "release" => match h.with(&obj, |s| s.release(&id, now_ms())) {
            Some(st) => Json(json!({"session_id": id, "state": st})).into_response(),
            None => err(404, "not_found", "unknown session"),
        },
        _ => err(404, "not_found", "no such route"),
    }
}

#[derive(serde::Deserialize)]
struct WaitQ {
    #[serde(default)]
    since: String,
    #[serde(default)]
    wait_ms: i64,
}

async fn job_wait(State(h): State<Arc<EdgeHost>>, Path((f, id)): Path<(String, String)>, hs: HeaderMap, Query(q): Query<WaitQ>) -> Response {
    if !h.internal(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    let obj = match object(&f) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let phase = |h: &EdgeHost| h.lock().scheds.get(&obj).map_or("unknown", |s| s.job_phase(&id)).to_owned();
    let mut rx = h.jobs.subscribe();
    let limit = Duration::from_millis(q.wait_ms.clamp(0, proto::JOB_WAIT_MAX_MS) as u64);
    let _ = tokio::time::timeout(limit, async {
        loop {
            if phase(&h) != q.since {
                return;
            }
            if rx.changed().await.is_err() {
                return;
            }
        }
    })
    .await;
    Json(JobWait { job_id: id.clone(), phase: phase(&h) }).into_response()
}

async fn registry_route(State(h): State<Arc<EdgeHost>>, hs: HeaderMap) -> Response {
    if !h.internal(&hs) && !h.is_admin(&hs) {
        return err(401, "unauthorized", "a valid token is required");
    }
    Json(h.registry()).into_response()
}

// ------------------------------------------------------------- public front

fn client_addr(hs: &HeaderMap) -> String {
    hs.get("cf-connecting-ip")
        .or_else(|| hs.get("x-forwarded-for"))
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(|v| v.trim().to_owned())
        .unwrap_or_else(|| "local".into())
}

/// The protocol-shaped error of a session route the edge refuses itself.
fn session_error(protocol: &str, status: u16, message: String, retry: Option<u32>) -> Response {
    let body = match protocol {
        "fal_director" => json!({"error": message}),
        "reactor" => json!({"detail": message}),
        _ => json!({"error": {"kind": if status == 429 { "queue_full" } else { "loading" }, "message": message}}),
    };
    let mut r = (StatusCode::from_u16(status).unwrap_or(StatusCode::SERVICE_UNAVAILABLE), Json(body)).into_response();
    if let Some(s) = retry {
        r.headers_mut().insert("retry-after", HeaderValue::from(s));
    }
    r
}

async fn public(State(h): State<Arc<EdgeHost>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let method = parts.method.clone();
    let path = parts.uri.path().to_owned();
    let query = parts.uri.query().unwrap_or("").to_owned();
    let path_q = parts.uri.path_and_query().map(|p| p.as_str().to_owned()).unwrap_or_else(|| path.clone());
    let mut headers = parts.headers.clone();
    if !headers.contains_key(REQUEST_ID_HEADER) {
        if let Ok(v) = HeaderValue::from_str(&format!("req_{}", uuid::Uuid::new_v4().simple())) {
            headers.insert(REQUEST_ID_HEADER, v);
        }
    }
    if let Ok(v) = HeaderValue::from_str(&client_addr(&parts.headers)) {
        headers.insert("x-forwarded-for", v);
    }
    let mut class = classify(method.as_str(), &path);
    // fal's proxy: route the request it stands for (admission, bindings and
    // all), straight to a front's own route.
    let (path, query, path_q) = match (&class.target, headers.get("x-fal-target-url").and_then(|v| v.to_str().ok()).and_then(front::unproxy)) {
        (Target::FalProxy, Some(inner)) => {
            let (p, q) = inner.split_once('?').map(|(p, q)| (p.to_owned(), q.to_owned())).unwrap_or((inner.clone(), String::new()));
            class = classify(method.as_str(), &p);
            if class.target == Target::FalProxy {
                return err(400, "invalid_request", "a proxied request cannot target the proxy");
            }
            headers.remove("x-fal-target-url");
            (p, q, inner)
        }
        _ => (path, query, path_q),
    };
    let t0 = std::time::Instant::now();
    let resp = route(&h, &class, &method, &path, &query, &path_q, &headers, body).await;
    tracing::debug!(method = %method, path = %path, protocol = class.protocol, status = resp.status().as_u16(), ms = t0.elapsed().as_millis() as u64, "edge: request");
    resp
}

#[allow(clippy::too_many_arguments)]
async fn route(h: &Arc<EdgeHost>, class: &Class, method: &Method, path: &str, query: &str, path_q: &str, headers: &HeaderMap, body: Body) -> Response {
    if let Target::Edge(r) = &class.target {
        return edge_route(h, r, method, path, headers, body).await;
    }
    if class.target == Target::NotFound {
        return err(404, "not_found", "no such route");
    }
    // The internal upload fetch: fronts only.
    if class.protocol == "internal" && !h.internal(headers) {
        return err(401, "unauthorized", "a valid token is required");
    }
    let mut verdict = h.verdict(headers);
    // Quotas (docs/serve/edge-control-plane.md §2.2).
    if verdict.presented && !verdict.valid && h.over(&format!("bad:{}", client_addr(headers)), h.cfg.quotas.invalid_key_rpm) {
        return reply(Reply { status: 429, kind: "rate_limited", message: "too many requests with an invalid key".into(), retry_after: Some(60) });
    }
    if let Some(k) = verdict.key.clone().filter(|_| is_submit(method.as_str(), class)) {
        if h.over(&k, h.cfg.quotas.key_rpm) {
            verdict.deny = Some(front::Deny { kind: "rate_limited".into(), message: "this key's submit rate limit is reached".into(), retry_after: Some(10) });
        } else if h.cfg.quotas.key_in_flight > 0 {
            let n = h.registry().owners.get(&k).copied().unwrap_or(0);
            if n >= h.cfg.quotas.key_in_flight {
                verdict.deny = Some(front::Deny { kind: "rate_limited".into(), message: format!("this key has {n} unfinished jobs (limit {})", h.cfg.quotas.key_in_flight), retry_after: Some(10) });
            }
        }
    }
    match &class.target {
        Target::Director(op) => return director(h, *op, method, path_q, headers, body, &verdict).await,
        Target::Reactor(op) => return reactor(h, op, method, path_q, headers, body, &verdict).await,
        Target::Stream { ingest, op } => return stream(h, *ingest, op, method, path_q, headers, body, &verdict).await,
        _ => {}
    }
    let reg = h.registry();
    // The body's model (a submit): read it whole on the host.
    let (bytes, model, body) = if class.target == Target::Body {
        let b = match axum::body::to_bytes(body, 256 << 20).await {
            Ok(b) => b,
            Err(e) => return err(413, "payload_too_large", e.to_string()),
        };
        let ct = headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
        let m = match scan_model(ct, &b, true) {
            Scan::Found(m) => Some(m),
            _ => None,
        };
        (Some(b), m, None)
    } else {
        (None, None, Some(body))
    };
    let fal_target = headers.get("x-fal-target-url").and_then(|v| v.to_str().ok()).and_then(|u| url::Url::parse(u).ok()).map(|u| u.path().to_owned());
    let fwd = match plan_forward(class, &reg, path, query, model.as_deref(), fal_target.as_deref()) {
        Ok(f) => f,
        Err(r) => return reply(r),
    };
    if fwd.deny.is_some() && verdict.deny.is_none() {
        verdict.deny = fwd.deny.clone();
    }
    let body: reqwest::Body = match (bytes, body) {
        (Some(b), _) => b.into(),
        (None, Some(body)) => reqwest::Body::wrap_stream(body.into_data_stream()),
        (None, None) => reqwest::Body::from(Vec::new()),
    };
    // Long waits (sync endpoints, SSE) are bounded by the client, not here.
    h.forward(&fwd.url, method, path_q, headers, body, &verdict, None).await
}

async fn edge_route(h: &Arc<EdgeHost>, r: &EdgeRoute, method: &Method, path: &str, headers: &HeaderMap, body: Body) -> Response {
    let reg = h.registry();
    let ready = reg.any_ready();
    let version = json!({"sha": crate::build_info::BuildInfo::current().git_sha_short, "channel": null});
    let need_admin = || {
        if h.is_admin(headers) {
            None
        } else {
            Some(err(401, "unauthorized", "the admin token is required"))
        }
    };
    // The caller's identity rides on the fan-outs (the APIs' own auth).
    let verdict = h.verdict(headers);
    match r {
        // FastWan's `GET /` names its model: the FastWan front answers.
        EdgeRoute::Root => match reg.fronts().find(|x| x.info.defaults.contains_key("fastwan")) {
            Some(x) => h.forward(&x.info.url, method, path, headers, reqwest::Body::from(Vec::new()), &verdict, Some(Duration::from_secs(15))).await,
            None => Json(json!({"server": "fv-edge", "role": "edge", "version": env!("CARGO_PKG_VERSION"), "ready": ready})).into_response(),
        },
        EdgeRoute::Ping => {
            if ready {
                Json(json!({"status": "healthy"})).into_response()
            } else {
                (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "unavailable"}))).into_response()
            }
        }
        EdgeRoute::Health => {
            let (c, status, state) = if ready { (StatusCode::OK, "ok", "AVAILABLE") } else { (StatusCode::SERVICE_UNAVAILABLE, "unavailable", "UNAVAILABLE") };
            (c, Json(json!({"status": status, "model_loaded": ready, "state": state, "edge": true, "version": env!("CARGO_PKG_VERSION")}))).into_response()
        }
        EdgeRoute::Healthz => {
            let st = front::status_body(&reg, version, now_ms());
            let c = if ready { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
            (c, Json(json!({"state": if ready { "ready" } else { "unavailable" }, "edge": true, "pools": st["pools"], "version": env!("CARGO_PKG_VERSION")}))).into_response()
        }
        EdgeRoute::Status => Json(front::status_body(&reg, version, now_ms())).into_response(),
        EdgeRoute::Capabilities => {
            let fronts = reg.one_per_family("native").into_iter().map(|x| x.info.url.clone()).collect();
            let bodies = h.fan_out(fronts, "/fv/v1/capabilities", headers, &verdict).await;
            if bodies.is_empty() && !verdict.valid && !h.cfg.auth_none {
                return err(401, "unauthorized", if verdict.presented { "invalid credentials" } else { "missing credentials" });
            }
            let mut body = front::merge_capabilities(&bodies, &reg, now_ms());
            // The cluster's mode, not the fronts' (`trust-edge`).
            if let Some(a) = body.get_mut("auth").and_then(Value::as_object_mut) {
                a.insert("mode".into(), json!(if h.cfg.auth_none { "none" } else { "keys" }));
            }
            Json(body).into_response()
        }
        EdgeRoute::Models => {
            let fronts = reg.one_per_family("openai_videos").into_iter().map(|x| x.info.url.clone()).collect();
            let bodies = h.fan_out(fronts, "/v1/models", headers, &verdict).await;
            Json(json!({"object": "list", "data": front::merge_list(&bodies, "data", "/id")})).into_response()
        }
        EdgeRoute::Model(m) => match reg.family_of_name(m).and_then(|f| reg.pick(Some(f), "openai_videos", None, None)).or_else(|| reg.pick(None, "openai_videos", None, None)) {
            Some(x) => h.forward(&x.info.url, method, path, headers, reqwest::Body::from(Vec::new()), &verdict, Some(Duration::from_secs(15))).await,
            None => err(503, "loading", "no worker serves this API right now"),
        },
        EdgeRoute::FalSchema => {
            let fronts = reg.one_per_family("fal").into_iter().map(|x| x.info.url.clone()).collect();
            let bodies = h.fan_out(fronts, "/fal/schema", headers, &verdict).await;
            Json(front::merge_fal_schema(&bodies)).into_response()
        }
        EdgeRoute::Keys | EdgeRoute::KeyRevoke(_) => {
            if let Some(r) = need_admin() {
                return r;
            }
            // serve-kit's own admin routes over the shared key store.
            let admin = h.admin.clone().expect("checked");
            let router: Router = fastvideo_serve_kit::admin_routes(h.cfg.keys.clone(), Arc::new(admin));
            let mut req = Request::new(body);
            *req.method_mut() = method.clone();
            *req.uri_mut() = path.parse().unwrap_or_default();
            *req.headers_mut() = headers.clone();
            let resp = tower::ServiceExt::oneshot(router, req).await.unwrap_or_else(|e| match e {});
            if matches!(r, EdgeRoute::KeyRevoke(_)) && resp.status().is_success() {
                h.key_epoch.fetch_add(1, Ordering::SeqCst);
            }
            resp
        }
        EdgeRoute::KeysInvalidate => {
            if let Some(r) = need_admin() {
                return r;
            }
            if let Err(e) = h.cfg.keys.refresh().await {
                return err(503, "loading", format!("reloading the keys: {e}"));
            }
            let e = h.key_epoch.fetch_add(1, Ordering::SeqCst) + 1;
            Json(json!({"key_epoch": e})).into_response()
        }
        EdgeRoute::Families => {
            if let Some(r) = need_admin() {
                return r;
            }
            let now = now_ms();
            let (statuses, metrics): (serde_json::Map<String, Value>, serde_json::Map<String, Value>) = {
                let g = h.lock();
                let st = g.scheds.iter().filter_map(|(o, s)| o.strip_prefix("family:").map(|f| (f.to_owned(), serde_json::to_value(s.status(now)).unwrap_or(Value::Null)))).collect();
                let me = g.scheds.iter().filter_map(|(o, s)| o.strip_prefix("family:").map(|f| (f.to_owned(), serde_json::to_value(s.metrics(now)).unwrap_or(Value::Null)))).collect();
                (st, me)
            };
            Json(json!({"object": "fv.edge.families", "families": statuses, "metrics": metrics, "key_epoch": reg.key_epoch})).into_response()
        }
        EdgeRoute::Metrics => {
            if let Some(r) = need_admin() {
                return r;
            }
            let now = now_ms();
            let ms: Vec<proto::FamilyMetrics> = h.lock().scheds.iter().filter(|(o, _)| o.starts_with("family:")).map(|(_, s)| s.metrics(now)).collect();
            ([("content-type", "text/plain; version=0.0.4")], front::prometheus(&ms)).into_response()
        }
        EdgeRoute::Internal => err(404, "not_found", "no such route"),
        EdgeRoute::Moved(why) => err(404, "not_found", *why),
    }
}

async fn read(body: Body) -> Result<Bytes, Response> {
    axum::body::to_bytes(body, 64 << 20).await.map_err(|e| err(413, "payload_too_large", e.to_string()))
}

#[allow(clippy::too_many_arguments)]
async fn director(h: &Arc<EdgeHost>, op: DirectorOp, method: &Method, path_q: &str, headers: &HeaderMap, body: Body, verdict: &Verdict) -> Response {
    let reg = h.registry();
    let raw = match read(body).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let v: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    // The app's family (else the first director front's).
    let app_family = v.get("app_id").and_then(Value::as_str).and_then(|a| reg.fal_app(&format!("/{}", a.trim_matches('/'))).map(|(_, f)| f.to_owned()));
    let family = app_family.or_else(|| reg.pick(None, "fal_director", None, None).map(|x| x.family.to_owned()));
    let Some(family) = family else {
        return session_error("fal_director", 503, "no worker serves the director right now".into(), Some(10));
    };
    match op {
        DirectorOp::Ice | DirectorOp::Info => match reg.pick(Some(&family), "fal_director", None, None) {
            Some(x) => h.forward(&x.info.url, method, path_q, headers, raw.into(), verdict, Some(Duration::from_secs(30))).await,
            None => session_error("fal_director", 503, "no worker serves the director right now".into(), Some(10)),
        },
        DirectorOp::Session => {
            let g = match h.admit_or_reclaim(&family, "director", None, verdict.key.clone()).await {
                Ok(g) => g,
                Err((st, m)) => return session_error("fal_director", st, m, Some(10)),
            };
            let b = SessionBinding { family: family.clone(), session_id: g.session_id.clone(), lease: g.lease, endpoint: g.endpoint.clone(), kind: "director".into(), owner: verdict.key.clone(), expires_ms: g.expires_ms };
            let (status, hs, v, bytes) = match h.forward_json(&g.endpoint, path_q, headers, raw, verdict).await {
                Ok(x) => x,
                Err(r) => {
                    h.release(&b);
                    return r;
                }
            };
            match v.get("session_id").and_then(Value::as_str).filter(|_| (200..300).contains(&status)) {
                Some(sid) => h.bind(&format!("director:{sid}"), b),
                None => h.release(&b),
            }
            let mut r = Response::new(Body::from(bytes));
            *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
            for (k, val) in hs.iter() {
                if !DROP_RESP.contains(&k.as_str()) {
                    r.headers_mut().insert(k.clone(), val.clone());
                }
            }
            r
        }
        DirectorOp::Heartbeat => {
            let sid = v.get("session_id").and_then(Value::as_str).unwrap_or_default();
            let Some(b) = h.binding(&format!("director:{sid}")) else {
                return Json(json!({"alive": false})).into_response();
            };
            let (status, _, v, bytes) = match h.forward_json(&b.endpoint, path_q, headers, raw, verdict).await {
                Ok(x) => x,
                Err(r) => return r,
            };
            if v.get("alive") == Some(&Value::Bool(false)) {
                h.unbind(&format!("director:{sid}"));
                h.release(&b);
            } else {
                h.renew(&b);
            }
            let mut r = Response::new(Body::from(bytes));
            *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
            r.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
            r
        }
        DirectorOp::Start => {
            let g = match h.admit_or_reclaim(&family, "director", None, verdict.key.clone()).await {
                Ok(g) => g,
                Err((st, m)) => return session_error("fal_director", st, m, Some(10)),
            };
            let b = SessionBinding { family, session_id: g.session_id.clone(), lease: g.lease, endpoint: g.endpoint.clone(), kind: "director".into(), owner: verdict.key.clone(), expires_ms: g.expires_ms };
            let resp = h.forward(&g.endpoint, method, path_q, headers, raw.into(), verdict, None).await;
            if !resp.status().is_success() {
                h.release(&b);
                return resp;
            }
            hold_while_streaming(h.clone(), b, resp)
        }
    }
}

/// An SSE session: renewed while the stream is open, released when it closes.
fn hold_while_streaming(h: Arc<EdgeHost>, b: SessionBinding, resp: Response) -> Response {
    struct Guard {
        h: Arc<EdgeHost>,
        b: SessionBinding,
        renew: tokio::task::JoinHandle<()>,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            self.renew.abort();
            self.h.release(&self.b);
        }
    }
    let renew = {
        let (h, b) = (h.clone(), b.clone());
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(15)).await;
                if !h.renew(&b) {
                    break;
                }
            }
        })
    };
    let guard = Guard { h, b, renew };
    let (parts, body) = resp.into_parts();
    let stream = body.into_data_stream().map(move |c| {
        let _ = &guard;
        c
    });
    Response::from_parts(parts, Body::from_stream(stream))
}

#[allow(clippy::too_many_arguments)]
async fn reactor(h: &Arc<EdgeHost>, op: &ReactorOp, method: &Method, path_q: &str, headers: &HeaderMap, body: Body, verdict: &Verdict) -> Response {
    let reg = h.registry();
    let owner = reactor_owner(verdict.key.as_deref().filter(|_| verdict.valid), Some(&client_addr(headers)));
    let raw = match read(body).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(family) = reg.reactor_family(h.cfg.reactor_model.as_deref()).map(str::to_owned) else {
        return session_error("reactor", 503, "no worker serves the Reactor runtime right now".into(), Some(10));
    };
    let held = h.binding(&owner);
    match op {
        ReactorOp::Start => {
            if let Some(b) = held {
                // One session per caller: the worker answers (busy or idempotent).
                return h.forward(&b.endpoint, method, path_q, headers, raw.into(), verdict, Some(Duration::from_secs(60))).await;
            }
            let g = match h.admit_or_reclaim(&family, "reactor", None, verdict.key.clone()).await {
                Ok(g) => g,
                Err((st, m)) => return session_error("reactor", if st == 429 { 503 } else { st }, m, Some(10)),
            };
            let b = SessionBinding { family, session_id: g.session_id.clone(), lease: g.lease, endpoint: g.endpoint.clone(), kind: "reactor".into(), owner: verdict.key.clone(), expires_ms: g.expires_ms };
            let resp = h.forward(&g.endpoint, method, path_q, headers, raw.into(), verdict, Some(Duration::from_secs(60))).await;
            if resp.status().is_success() {
                h.bind(&owner, b);
            } else {
                h.release(&b);
            }
            resp
        }
        ReactorOp::Stop => {
            let Some(b) = held else {
                return session_error("reactor", 404, "no active session".into(), None);
            };
            let resp = h.forward(&b.endpoint, method, path_q, headers, raw.into(), verdict, Some(Duration::from_secs(60))).await;
            if resp.status().is_success() {
                h.unbind(&owner);
                h.release(&b);
            }
            resp
        }
        ReactorOp::Follow | ReactorOp::Sid(_) => {
            let target = match held {
                Some(b) => {
                    h.renew(&b);
                    b.endpoint
                }
                // No session: a ready worker answers the descriptor, schema, journal.
                None => match reg.pick(Some(&family), "reactor", None, None) {
                    Some(x) => x.info.url.clone(),
                    None => return session_error("reactor", 503, "no worker serves the Reactor runtime right now".into(), Some(10)),
                },
            };
            let timeout = (!path_q.starts_with("/events")).then_some(Duration::from_secs(60));
            h.forward(&target, method, path_q, headers, raw.into(), verdict, timeout).await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn stream(h: &Arc<EdgeHost>, ingest: bool, op: &StreamOp, method: &Method, path_q: &str, headers: &HeaderMap, body: Body, verdict: &Verdict) -> Response {
    let reg = h.registry();
    let kind = if ingest { "ingest" } else { "stream" };
    let raw = match read(body).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let owner_ok = |b: &SessionBinding| b.owner.is_none() || b.owner == verdict.key;
    match op {
        StreamOp::Create => {
            // The model: the body's (streams) or the query's (ingest).
            let model = if ingest {
                path_q.split_once('?').and_then(|(_, q)| front::query_param(q, "model"))
            } else {
                serde_json::from_slice::<Value>(&raw).ok().and_then(|v| v.get("model").and_then(Value::as_str).map(str::to_owned))
            };
            let family = model.as_deref().and_then(|m| reg.family_of_name(m)).map(str::to_owned).or_else(|| reg.default_family("native").map(str::to_owned));
            let Some(family) = family else {
                return session_error("native", 503, "no worker serves this model right now".into(), Some(10));
            };
            let g = match h.admit_or_reclaim(&family, kind, model.clone(), verdict.key.clone()).await {
                Ok(g) => g,
                Err((st, m)) => return session_error("native", st, m, Some(10)),
            };
            let b = SessionBinding { family, session_id: g.session_id.clone(), lease: g.lease, endpoint: g.endpoint.clone(), kind: kind.into(), owner: verdict.key.clone(), expires_ms: g.expires_ms };
            if ingest && h.cfg.whip_redirect {
                // The 307 hand-off: the client re-sends its offer to the
                // worker with a capability; the lease runs out on its TTL
                // (or when the worker ends the session).
                let cap = front::sign_cap(&h.cfg.internal_token, &front::SessionCap { verdict: verdict.clone(), exp_ms: now_ms() + front::CAP_TTL_MS });
                let to = front::with_param(&format!("{}{path_q}", g.endpoint.trim_end_matches('/')), front::CAP_PARAM, &cap);
                return (StatusCode::TEMPORARY_REDIRECT, [("location", to)]).into_response();
            }
            let resp = h.forward(&g.endpoint, method, path_q, headers, raw.into(), verdict, Some(Duration::from_secs(60))).await;
            if !resp.status().is_success() {
                h.release(&b);
                return resp;
            }
            // The id: the JSON answer's (streams) or the Location's (ingest).
            let (parts, body) = resp.into_parts();
            let bytes = axum::body::to_bytes(body, 16 << 20).await.unwrap_or_default();
            let id = if ingest {
                parts.headers.get("location").and_then(|v| v.to_str().ok()).and_then(|l| l.rsplit('/').next()).map(str::to_owned)
            } else {
                serde_json::from_slice::<Value>(&bytes).ok().and_then(|v| v.get("id").and_then(Value::as_str).map(str::to_owned))
            };
            match id {
                Some(id) => h.bind(&format!("{kind}:{id}"), b),
                None => h.release(&b),
            }
            let mut parts = parts;
            // A Location on the worker's host points back at the edge.
            if let Some(l) = parts.headers.get("location").and_then(|v| v.to_str().ok()).map(str::to_owned) {
                if let Some(rest) = l.strip_prefix(g.endpoint.trim_end_matches('/')) {
                    if let Ok(v) = HeaderValue::from_str(rest) {
                        parts.headers.insert("location", v);
                    }
                }
            }
            parts.headers.remove("content-length");
            Response::from_parts(parts, Body::from(bytes))
        }
        StreamOp::List => {
            let fronts = reg.distinct("native").into_iter().map(|x| x.info.url.clone()).collect();
            let path = if ingest { "/fv/v1/streams/ingest" } else { "/fv/v1/streams" };
            let bodies = h.fan_out(fronts, path, headers, verdict).await;
            Json(json!({"object": "list", "data": front::merge_list(&bodies, "data", "/id")})).into_response()
        }
        StreamOp::Follow(id) | StreamOp::Delete(id) => {
            let alias = format!("{kind}:{id}");
            let Some(b) = h.binding(&alias).filter(owner_ok) else {
                return session_error("native", 404, format!("stream `{id}` was not found"), None);
            };
            let del = matches!(op, StreamOp::Delete(_));
            if !del {
                h.renew(&b);
            }
            let resp = h.forward(&b.endpoint, method, path_q, headers, raw.into(), verdict, Some(Duration::from_secs(60))).await;
            if del && resp.status().is_success() {
                h.unbind(&alias);
                h.release(&b);
            }
            resp
        }
    }
}
