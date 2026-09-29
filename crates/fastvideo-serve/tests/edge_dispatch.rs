//! Push dispatch through a pool Durable Object (docs/serve/gateway-cloudflare.md)
//! on the fake engine: a gateway whose pod pool has `dispatch =
//! "durable-object"`, fake-engine workers holding a socket to the pool's
//! dispatcher (`[dispatch] do_url`), all sharing one D1 (the SQLite mock).
//!
//! The dispatcher is, by default, [`native_do`]: the shared scheduler
//! (`fastvideo_dispatch_proto::sched`) behind the same routes as the
//! Worker, on axum. With `FV_EDGE_URL` (and `FV_EDGE_TOKEN`, the Worker's
//! `FV_INTERNAL_TOKEN`) the tests use a real one instead: `wrangler dev`
//! (scripts/serve/cf-edge.sh dev) or the staging Worker.
//!
//! - `do_path_runs_jobs_cancels_and_redispatches_after_a_loss`
//! - `do_path_survives_a_redeploy_during_jobs`: the dispatcher restarts
//!   (every socket drops, the state is reloaded from storage) while jobs
//!   run; the workers reconnect and reconcile, and every job runs exactly
//!   once. Against an external Worker the test waits for the operator to
//!   redeploy it (`FV_EDGE_REDEPLOY=1`, see the doc).
//! - `queue_time_gateway_vs_durable_object` (`FV_EDGE_BENCH=1`, a few
//!   minutes): submit → worker start, one job at a time and 5 at once, on
//!   both paths (printed; `--nocapture`). `FV_EDGE_BENCH_D1_MS` (250) is
//!   the simulated D1 round trip, `FV_EDGE_BENCH_JOBS` (5) the serial jobs.

#![cfg(all(feature = "http-client", feature = "openai-videos", feature = "minimax", feature = "fal", feature = "ltxapi"))]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::PoolStatus;
use fastvideo_serve::config::{Config, DispatchMode, EngineBackendKind, JobBackend, KeyStoreBackend, PoolCfg, PoolKind, Role};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::client::{D1Error, D1Transport, RawReply};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-edge-user";

fn init_log() {
    if std::env::var("RUST_LOG").is_ok() {
        let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_test_writer().try_init();
    }
}
const ADMIN: &str = "fvadm_edge_test";

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fv-edge-{tag}-{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()))
}

/// The native stand-in for the Durable Object: the same routes, the same
/// scheduler, a restartable "deploy".
mod native_do {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
    use axum::extract::{Path, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use fastvideo_dispatch_proto::sched::{Cfg, JobRec, Out, Sched, WorkerRec};
    use fastvideo_dispatch_proto::{self as proto, EnqueueReq, WorkerMsg};
    use futures::{SinkExt, StreamExt};
    use tokio::sync::{mpsc, watch};

    fn now() -> i64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
    }

    struct Inner {
        /// Workers refused (as by a network partition) until then (ms).
        blocked: HashMap<String, i64>,
        scheds: HashMap<String, Sched>,
        /// (pool, worker) → (connection nonce, sender).
        conns: HashMap<(String, String), (u64, mpsc::UnboundedSender<String>)>,
        next_conn: u64,
    }

    #[derive(Clone)]
    pub struct NativeDo {
        inner: Arc<Mutex<Inner>>,
        token: String,
        cfg: Cfg,
        /// Bumped by a redeploy: every socket task ends.
        epoch: watch::Sender<u64>,
        pub version: Arc<Mutex<String>>,
    }

    impl NativeDo {
        pub async fn start(token: &str, cfg: Cfg) -> (Self, String) {
            let (epoch, _) = watch::channel(0);
            let d = Self {
                inner: Arc::new(Mutex::new(Inner { blocked: HashMap::new(), scheds: HashMap::new(), conns: HashMap::new(), next_conn: 0 })),
                token: token.to_owned(),
                cfg,
                epoch,
                version: Arc::new(Mutex::new("native-1".into())),
            };
            let app = Router::new()
                .route("/pools/{pool}/connect", get(connect))
                .route("/pools/{pool}/enqueue", post(enqueue))
                .route("/pools/{pool}/cancel/{job}", post(cancel))
                .route("/pools/{pool}/status", get(status))
                .with_state(d.clone());
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", l.local_addr().unwrap());
            tokio::spawn(async move {
                let _ = axum::serve(l, app).await;
            });
            // The alarm.
            let t = d.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    let now = now();
                    let mut g = t.inner.lock().unwrap();
                    let pools: Vec<String> = g.scheds.keys().cloned().collect();
                    for p in pools {
                        let due = g.scheds.get(&p).and_then(|s| s.next_wake(now)).is_some_and(|w| w <= now);
                        if due {
                            let out = g.scheds.get_mut(&p).unwrap().tick(now);
                            apply(&mut g, &p, out);
                        }
                    }
                }
            });
            (d, base)
        }

        /// A deploy: every socket drops and each pool's scheduler is rebuilt
        /// from its persisted rows (as a restarted Durable Object is).
        pub fn redeploy(&self) {
            let now = now();
            let mut g = self.inner.lock().unwrap();
            let pools: Vec<String> = g.scheds.keys().cloned().collect();
            for p in pools {
                let s = g.scheds.remove(&p).unwrap();
                let jobs: Vec<JobRec> = s.jobs().cloned().collect();
                let workers: Vec<WorkerRec> = s.workers().cloned().collect();
                let mut s = Sched::restore(p.clone(), self.cfg.clone(), jobs, workers);
                let ids: Vec<String> = s.workers().filter(|w| w.connected).map(|w| w.worker_id.clone()).collect();
                for id in ids {
                    s.disconnect(&id, now);
                }
                g.scheds.insert(p, s);
            }
            g.conns.clear();
            let v = *self.epoch.borrow() + 1;
            let _ = self.epoch.send(v);
            let mut ver = self.version.lock().unwrap();
            *ver = format!("native-{}", v + 1);
        }

        /// A network partition: `worker`'s socket drops and it cannot
        /// reconnect for `for_ms`; the worker itself keeps running.
        pub fn partition(&self, worker: &str, for_ms: i64) {
            let mut g = self.inner.lock().unwrap();
            let t = now();
            g.blocked.insert(worker.to_owned(), t + for_ms);
            g.conns.retain(|(_, w), _| w != worker);
            for s in g.scheds.values_mut() {
                s.disconnect(worker, t);
            }
        }

        fn authed(&self, h: &HeaderMap) -> bool {
            h.get(proto::TOKEN_HEADER).and_then(|v| v.to_str().ok()) == Some(self.token.as_str())
        }
    }

    fn apply(g: &mut Inner, pool: &str, out: Vec<Out>) {
        for o in out {
            match o {
                Out::Send { worker, msg } => {
                    if let Some((_, tx)) = g.conns.get(&(pool.to_owned(), worker)) {
                        let _ = tx.send(serde_json::to_string(&msg).unwrap());
                    }
                }
                Out::Close { worker } => {
                    g.conns.remove(&(pool.to_owned(), worker));
                }
                // No spill here (only the Worker has an R2 bucket).
                Out::PushSpilled { .. } => {}
            }
        }
        if let Some(s) = g.scheds.get_mut(pool) {
            let _ = s.take_dirty();
        }
    }

    fn sched<'a>(g: &'a mut Inner, pool: &str, cfg: &Cfg) -> &'a mut Sched {
        g.scheds.entry(pool.to_owned()).or_insert_with(|| Sched::new(pool, cfg.clone()))
    }

    async fn connect(State(d): State<NativeDo>, Path(pool): Path<String>, h: HeaderMap, ws: WebSocketUpgrade) -> Response {
        if !d.authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let Some(worker) = h.get(proto::WORKER_HEADER).and_then(|v| v.to_str().ok()).map(str::to_owned) else {
            return StatusCode::BAD_REQUEST.into_response();
        };
        if d.inner.lock().unwrap().blocked.get(&worker).is_some_and(|t| *t > now()) {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        ws.on_upgrade(move |socket| run(d, pool, worker, socket))
    }

    async fn run(d: NativeDo, pool: String, worker: String, socket: WebSocket) {
        let (mut sink, mut stream) = socket.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let conn = {
            let mut g = d.inner.lock().unwrap();
            g.next_conn += 1;
            let c = g.next_conn;
            g.conns.insert((pool.clone(), worker.clone()), (c, tx));
            c
        };
        let mut epoch = d.epoch.subscribe();
        loop {
            tokio::select! {
                _ = epoch.changed() => return,
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
                    let mut g = d.inner.lock().unwrap();
                    let out = sched(&mut g, &pool, &d.cfg).on_msg(&worker, msg, now());
                    apply(&mut g, &pool, out);
                }
            }
        }
        let mut g = d.inner.lock().unwrap();
        let key = (pool.clone(), worker.clone());
        if g.conns.get(&key).is_some_and(|(c, _)| *c == conn) {
            g.conns.remove(&key);
            let out = sched(&mut g, &pool, &d.cfg).disconnect(&worker, now());
            apply(&mut g, &pool, out);
        }
    }

    async fn enqueue(State(d): State<NativeDo>, Path(pool): Path<String>, h: HeaderMap, Json(req): Json<EnqueueReq>) -> Response {
        if !d.authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let mut g = d.inner.lock().unwrap();
        let (resp, out) = sched(&mut g, &pool, &d.cfg).enqueue(req, now());
        apply(&mut g, &pool, out);
        (StatusCode::ACCEPTED, Json(resp)).into_response()
    }

    async fn cancel(State(d): State<NativeDo>, Path((pool, job)): Path<(String, String)>, h: HeaderMap) -> Response {
        if !d.authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let mut g = d.inner.lock().unwrap();
        let (st, out) = sched(&mut g, &pool, &d.cfg).cancel(&job, now());
        apply(&mut g, &pool, out);
        match st {
            Some(s) => Json(serde_json::json!({"job_id": job, "state": s})).into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    }

    async fn status(State(d): State<NativeDo>, Path(pool): Path<String>, h: HeaderMap) -> Response {
        if !d.authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let mut g = d.inner.lock().unwrap();
        let mut st = sched(&mut g, &pool, &d.cfg).status(now());
        st.dispatcher = d.version.lock().unwrap().clone();
        Json(st).into_response()
    }
}

/// The dispatcher under test: native, or `FV_EDGE_URL`.
struct Dispatcher {
    base: String,
    token: String,
    native: Option<native_do::NativeDo>,
}

impl Dispatcher {
    async fn new() -> Self {
        match std::env::var("FV_EDGE_URL").ok().filter(|s| !s.is_empty()) {
            Some(base) => {
                let token = std::env::var("FV_EDGE_TOKEN").expect("FV_EDGE_TOKEN (the Worker's FV_INTERNAL_TOKEN) with FV_EDGE_URL");
                Self { base: base.trim_end_matches('/').to_owned(), token, native: None }
            }
            None => {
                let token = "edge-internal-token".to_owned();
                let cfg = fastvideo_dispatch_proto::sched::Cfg { reconnect_grace_ms: 1_500, stale_after_ms: 5_000, redispatch_wait_ms: 3_000, ..Default::default() };
                let (d, base) = native_do::NativeDo::start(&token, cfg).await;
                Self { base, token, native: Some(d) }
            }
        }
    }
    fn external(&self) -> bool {
        self.native.is_none()
    }
    async fn status(&self, http: &Http, pool: &str) -> PoolStatus {
        let r = http.0.get(format!("{}/pools/{pool}/status", self.base)).header("x-fv-internal-token", &self.token).send().await.unwrap();
        assert!(r.status().is_success(), "status: {}", r.status());
        r.json().await.unwrap()
    }
}

/// A unique pool id per run (an external dispatcher keeps its state).
fn pool_id(tag: &str) -> String {
    format!("{tag}-{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() % 0xff_ffff)
}

#[derive(Clone)]
struct Cuttable {
    mock: MockD1,
    dead: Arc<AtomicBool>,
    delay: Duration,
}

#[async_trait::async_trait]
impl D1Transport for Cuttable {
    async fn post(&self, body: &Value) -> Result<RawReply, D1Error> {
        if self.dead.load(Ordering::SeqCst) {
            return Err(D1Error::Transport("worker is gone".into()));
        }
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        self.mock.post(body).await
    }
}

struct Shared {
    mock: MockD1,
    arts: PathBuf,
    d1_delay: Duration,
    token: String,
}

impl Shared {
    fn new(token: &str) -> Self {
        let arts = tmp("arts");
        std::fs::create_dir_all(&arts).unwrap();
        Self { mock: MockD1::new(), arts, d1_delay: Duration::ZERO, token: token.to_owned() }
    }
    fn d1(&self, dead: &Arc<AtomicBool>) -> D1Client {
        D1Client::new(Arc::new(Cuttable { mock: self.mock.clone(), dead: dead.clone(), delay: self.d1_delay }))
            .with_retry(fastvideo_serve_kit::d1::RetryPolicy::immediate(1))
    }
    fn job_row(&self, external_id: &str) -> serde_json::Map<String, Value> {
        self.mock.sql("SELECT worker, job FROM jobs WHERE external_id = ?", &[json!(external_id)]).unwrap().remove(0)
    }
    fn job(&self, external_id: &str) -> fastvideo_protocol::Job {
        serde_json::from_str(self.job_row(external_id)["job"].as_str().unwrap()).unwrap()
    }
}

fn base_config(tag: &str, sh: &Shared) -> Config {
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "shared-signing-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), tmp(tag).display().to_string());
    env.insert("FV_INTERNAL_TOKEN".to_owned(), sh.token.clone());
    env.insert("FV_ARTIFACTS_DIR".to_owned(), sh.arts.display().to_string());
    env.insert("FV_ADMIN_TOKEN".to_owned(), ADMIN.to_owned());
    env.insert("FV_CF_ACCOUNT_ID".to_owned(), "acct".to_owned());
    env.insert("FV_CF_API_TOKEN".to_owned(), "tok".to_owned());
    env.insert("FV_D1_DATABASE_ID".to_owned(), "db".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::D1;
    c.jobs.progress_interval_ms = 100;
    c.jobs.heartbeat_s = 1;
    c.auth.key_store = KeyStoreBackend::Memory;
    c.engine.fake.step_ms = 5;
    c.protocols.fastwan = false;
    c
}

struct Running {
    app: App,
    base: String,
    dead: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl Running {
    /// Gone: no HTTP, no D1, no dispatcher socket (dropping the app ends it).
    fn kill(self) {
        self.dead.store(true, Ordering::SeqCst);
        self.task.abort();
        drop(self.app);
    }
}

async fn serve(c: Config, sh: &Shared, ov: Overrides) -> Running {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let mut c = c;
    if c.engine.backend == EngineBackendKind::Remote {
        c.server.public_base_url = Some(base.clone());
    }
    c.validate().unwrap();
    let dead = Arc::new(AtomicBool::new(false));
    let ov = Overrides { d1: Some(sh.d1(&dead)), ..ov };
    let app = App::build(c, ov).await.unwrap();
    app.gate.engine().wait_ready().await;
    let router = app.router.clone();
    let task = tokio::spawn(async move {
        let _ = axum::serve(l, router).await;
    });
    Running { app, base, dead, task }
}

/// A fake-engine worker of `pool`: with `do_url`, connected to the
/// dispatcher only (no public URL, no gateway registration); without, a
/// plain pod worker (the gateway path).
async fn worker(sh: &Shared, tag: &str, pool: &str, do_url: Option<&str>, step_ms: u64, capacity: u32) -> Running {
    let mut c = base_config(tag, sh);
    c.server.role = Role::Worker;
    c.server.worker_id = Some(format!("worker-{tag}"));
    c.gateway.pool = Some(pool.to_owned());
    c.engine.fake.models = vec!["fake-h3-turbo".into()];
    c.engine.fake.step_ms = step_ms;
    c.dispatch.do_url = do_url.map(str::to_owned);
    c.dispatch.capacity = capacity;
    c.dispatch.status_s = 2;
    serve(c, sh, Overrides::default()).await
}

async fn gateway(sh: &Shared, pool: &str, dispatch: DispatchMode, do_url: Option<&str>, urls: &[&str]) -> Running {
    let mut c = base_config("gw", sh);
    c.engine.backend = EngineBackendKind::Remote;
    c.pools = vec![PoolCfg {
        id: pool.into(),
        kind: PoolKind::Pod,
        urls: urls.iter().map(|u| u.to_string()).collect(),
        fake_models: vec!["fake-h3-turbo".into()],
        stale_after_s: 3,
        retries: 1,
        dispatch,
        do_url: do_url.map(str::to_owned),
        ..PoolCfg::default()
    }];
    c.gateway.tick_s = 1;
    c.gateway.watch_poll_ms = 100;
    serve(c, sh, Overrides::default()).await
}

struct Http(reqwest::Client);

impl Http {
    fn new() -> Self {
        Self(reqwest::Client::builder().no_proxy().build().unwrap())
    }
    async fn call(&self, method: &str, url: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.0.request(reqwest::Method::from_bytes(method.as_bytes()).unwrap(), url).header("authorization", format!("Bearer {KEY}"));
        if let Some(b) = body {
            r = r.json(&b);
        }
        let resp = r.send().await.unwrap();
        let s = resp.status().as_u16();
        let bytes = resp.bytes().await.unwrap();
        (s, serde_json::from_slice(&bytes).unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into_owned())))
    }
    async fn submit(&self, g: &str, prompt: &str) -> String {
        let (s, v) = self.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-h3-turbo", "prompt": prompt}))).await;
        assert_eq!(s, 202, "{v}");
        v["id"].as_str().unwrap().to_owned()
    }
    async fn wait(&self, g: &str, id: &str, done: impl Fn(&Value) -> bool, limit: Duration) -> Value {
        let t0 = Instant::now();
        loop {
            let (s, v) = self.call("GET", &format!("{g}/fv/v1/jobs/{id}"), None).await;
            if s == 200 && done(&v) {
                return v;
            }
            assert!(t0.elapsed() < limit, "job {id} never got there: {v}");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
    async fn finished(&self, g: &str, id: &str) -> Value {
        self.wait(g, id, |v| matches!(v["status"].as_str(), Some("succeeded" | "failed" | "cancelled")), Duration::from_secs(180)).await
    }
}

/// Waits until `n` workers are connected to the dispatcher's `pool`.
async fn connected(d: &Dispatcher, http: &Http, pool: &str, n: usize) -> PoolStatus {
    let t0 = Instant::now();
    loop {
        let st = d.status(http, pool).await;
        if st.usable_workers() >= n {
            return st;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "workers never connected: {st:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Waits until the gateway sees `pool` available (its tick read the DO).
async fn available(http: &Http, g: &str) {
    let t0 = Instant::now();
    loop {
        let (_, v) = http.call("GET", &format!("{g}/fv/v1/status"), None).await;
        if v["pools"].as_array().is_some_and(|a| a.iter().any(|p| p["available"] == true)) {
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "pool never available: {v}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn do_path_runs_jobs_cancels_and_redispatches_after_a_loss() {
    init_log();
    let d = Dispatcher::new().await;
    let sh = Shared::new(&d.token);
    let pool = pool_id("h3");
    let http = Http::new();
    let a = worker(&sh, "a", &pool, Some(&d.base), 150, 1).await;
    let b = worker(&sh, "b", &pool, Some(&d.base), 150, 1).await;
    connected(&d, &http, &pool, 2).await;
    let gw = gateway(&sh, &pool, DispatchMode::DurableObject, Some(&d.base), &[]).await;
    let g = gw.base.clone();
    available(&http, &g).await;

    // Caps and workers come from the dispatcher (the workers' hellos).
    let pools: Value = http.0.get(format!("{g}/fv/v1/gateway/pools")).bearer_auth(ADMIN).send().await.unwrap().json().await.unwrap();
    assert!(pools.to_string().contains("do:worker-a"), "{pools}");

    // A job runs, pushed to a worker; the dispatch row says so.
    let id = http.submit(&g, "pushed").await;
    let done = http.finished(&g, &id).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    let row = sh.mock.sql("SELECT d.kind AS kind, d.state AS state FROM gw_dispatch d JOIN jobs j ON j.id = d.job_id WHERE j.external_id = ?", &[json!(id)]).unwrap();
    assert_eq!(row[0]["kind"], "durable-object");
    let job = sh.job(&id);
    assert!(job.dispatched_at.is_some() && job.started_at.is_some());

    // Cancel while running goes through the dispatcher to the worker.
    let cid = http.submit(&g, "to cancel").await;
    http.wait(&g, &cid, |v| v["status"] == "running", Duration::from_secs(30)).await;
    let (s, v) = http.call("DELETE", &format!("{g}/fv/v1/jobs/{cid}"), None).await;
    assert_eq!(s, 200, "{v}");
    let c = http.finished(&g, &cid).await;
    assert_eq!(c["status"], "cancelled", "{c}");

    // Worker loss: its socket closes and stays closed; after the grace
    // period the dispatcher re-dispatches the job (takeover) to the other.
    let lid = http.submit(&g, "survives a lost worker").await;
    http.wait(&g, &lid, |v| v["status"] == "running", Duration::from_secs(30)).await;
    let holder = sh.job_row(&lid)["worker"].as_str().unwrap().to_owned();
    let (victim, survivor, keep) = if holder == "worker-a" { (a, "worker-b", b) } else { (b, "worker-a", a) };
    victim.kill();
    let done = http.finished(&g, &lid).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(sh.job_row(&lid)["worker"], survivor);
    let st = d.status(&http, &pool).await;
    assert!(st.failed.is_empty(), "{st:?}");

    // The last worker gone too: re-dispatched once, then failed by the
    // dispatcher, and the gateway fails the D1 row.
    if !d.external() {
        let fid = http.submit(&g, "no worker left").await;
        http.wait(&g, &fid, |v| v["status"] == "running", Duration::from_secs(30)).await;
        keep.kill();
        let f = http.wait(&g, &fid, |v| v["status"] == "failed", Duration::from_secs(60)).await;
        assert!(f.to_string().contains("lost"), "{f}");
    }
    drop(gw);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn do_path_survives_a_redeploy_during_jobs() {
    init_log();
    let d = Dispatcher::new().await;
    if d.external() && std::env::var("FV_EDGE_REDEPLOY").ok().as_deref() != Some("1") {
        eprintln!("skipped: an external dispatcher is redeployed only with FV_EDGE_REDEPLOY=1");
        return;
    }
    let sh = Shared::new(&d.token);
    let pool = pool_id("rd");
    let http = Http::new();
    // Jobs long enough to span a deploy (external: tens of seconds).
    let step = if d.external() { 1500 } else { 200 };
    let a = worker(&sh, "ra", &pool, Some(&d.base), step, 2).await;
    let b = worker(&sh, "rb", &pool, Some(&d.base), step, 2).await;
    connected(&d, &http, &pool, 2).await;
    let gw = gateway(&sh, &pool, DispatchMode::DurableObject, Some(&d.base), &[]).await;
    let g = gw.base.clone();
    available(&http, &g).await;
    let mut ids = Vec::new();
    for i in 0..4 {
        ids.push(http.submit(&g, &format!("across a redeploy {i}")).await);
    }
    for id in &ids {
        http.wait(&g, id, |v| v["status"] == "running" || v["status"] == "queued", Duration::from_secs(30)).await;
    }
    // Who holds what before the deploy.
    let before: BTreeMap<String, String> = {
        let t0 = Instant::now();
        loop {
            let m: BTreeMap<String, String> = ids.iter().filter_map(|id| sh.job_row(id)["worker"].as_str().map(|w| (id.clone(), w.to_owned()))).collect();
            if m.len() == ids.len() || t0.elapsed() > Duration::from_secs(20) {
                break m;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    assert_eq!(before.len(), ids.len(), "every job adopted before the deploy");
    let v0 = d.status(&http, &pool).await.dispatcher;
    match &d.native {
        Some(n) => n.redeploy(),
        None => {
            println!("FV-EDGE: redeploy the Worker now (dispatcher version {v0}); waiting for a new version");
            let t0 = Instant::now();
            loop {
                let v = http.0.get(format!("{}/pools/{pool}/status", d.base)).header("x-fv-internal-token", &d.token).send().await;
                if let Ok(r) = v {
                    if let Ok(st) = r.json::<PoolStatus>().await {
                        if st.dispatcher != v0 {
                            println!("FV-EDGE: dispatcher now {} after {:.1} s", st.dispatcher, t0.elapsed().as_secs_f64());
                            break;
                        }
                    }
                }
                assert!(t0.elapsed() < Duration::from_secs(600), "no redeploy within 10 min");
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        }
    }
    let t_re = Instant::now();
    let st = connected(&d, &http, &pool, 2).await;
    println!("FV-EDGE: both workers reconnected {:.2} s after the new version answered; {:?}", t_re.elapsed().as_secs_f64(), st.workers.iter().map(|w| (&w.worker_id, w.held)).collect::<Vec<_>>());
    // Every job finishes, once, on the worker that held it.
    for id in &ids {
        let v = http.finished(&g, id).await;
        assert_eq!(v["status"], "succeeded", "{v}");
        let job = sh.job(id);
        assert!(!job.logs.iter().any(|l| l.message.contains("dispatching again")), "{id} was re-dispatched: {:?}", job.logs);
        assert_eq!(sh.job_row(id)["worker"].as_str(), before.get(id).map(String::as_str), "{id} changed worker");
    }
    // Each worker adopted exactly the jobs it held before (a second take
    // after the reconnect would be a second adopt).
    for (w, name) in [(&a, "worker-ra"), (&b, "worker-rb")] {
        let mine = before.values().filter(|v| v.as_str() == name).count();
        assert_eq!(w.app.d1.as_ref().unwrap().cached(), mine, "{name} adopted other jobs");
    }
    let st = d.status(&http, &pool).await;
    assert!(st.failed.is_empty(), "{st:?}");
    assert_eq!((st.queued, st.pushed), (0, 0), "{st:?}");
    drop((gw, a, b));
}

/// One job's times, seconds: created → dispatched_at, created → started_at.
#[derive(Clone, Copy, Debug, Default)]
struct Times {
    dispatch: f64,
    queue: f64,
    submit: f64,
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let mean = s.iter().sum::<f64>() / s.len().max(1) as f64;
    (mean, s[s.len() / 2], s[s.len() - 1])
}

/// Runs `n` jobs, `at_once` at a time, and returns each job's times.
async fn measure(sh: &Shared, g: &str, http: &Http, n: usize, at_once: usize) -> Vec<Times> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < n {
        let batch = at_once.min(n - k);
        let subs = (0..batch).map(|i| {
            let prompt = format!("bench {k} {i}");
            async move {
                let t0 = Instant::now();
                let id = http.submit(g, &prompt).await;
                (id, t0.elapsed().as_secs_f64())
            }
        });
        let ids = futures::future::join_all(subs).await;
        for (id, submit) in ids {
            let v = http.finished(g, &id).await;
            assert_eq!(v["status"], "succeeded", "{v}");
            let j = sh.job(&id);
            let secs = |a: time::OffsetDateTime, b: time::OffsetDateTime| (b - a).as_seconds_f64();
            out.push(Times { dispatch: secs(j.created_at, j.dispatched_at.unwrap()), queue: secs(j.created_at, j.started_at.unwrap()), submit });
        }
        k += batch;
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn queue_time_gateway_vs_durable_object() {
    if std::env::var("FV_EDGE_BENCH").ok().as_deref() != Some("1") {
        eprintln!("skipped: set FV_EDGE_BENCH=1 to measure queue times");
        return;
    }
    init_log();
    let d1_ms: u64 = std::env::var("FV_EDGE_BENCH_D1_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
    let n: usize = std::env::var("FV_EDGE_BENCH_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let d = Dispatcher::new().await;
    let http = Http::new();
    let mut rows = Vec::new();
    for path in ["gateway", "durable-object"] {
        let mut sh = Shared::new(&d.token);
        sh.d1_delay = Duration::from_millis(d1_ms);
        let pool = pool_id("bench");
        let edge = path == "durable-object";
        let do_url = edge.then_some(d.base.as_str());
        // Five workers, so five jobs at once start at once on both paths.
        let mut ws = Vec::new();
        for i in 0..5 {
            ws.push(worker(&sh, &format!("{path}-{i}"), &pool, do_url, 5, 1).await);
        }
        let urls: Vec<String> = if edge { Vec::new() } else { ws.iter().map(|w| w.base.clone()).collect() };
        let url_refs: Vec<&str> = urls.iter().map(String::as_str).collect();
        if edge {
            connected(&d, &http, &pool, 5).await;
        }
        let gw = gateway(&sh, &pool, if edge { DispatchMode::DurableObject } else { DispatchMode::Gateway }, do_url, &url_refs).await;
        let g = gw.base.clone();
        available(&http, &g).await;
        // Warm-up (connections, caches).
        measure(&sh, &g, &http, 1, 1).await;
        let one = measure(&sh, &g, &http, n, 1).await;
        let five = measure(&sh, &g, &http, 5, 5).await;
        if edge {
            let st = d.status(&http, &pool).await;
            println!("dispatcher timings ({}): {:?}", st.dispatcher, st.timings);
        }
        rows.push((path, one, five));
        drop(gw);
        for w in ws {
            w.kill();
        }
    }
    let where_ = if d.external() { d.base.clone() } else { "native (in-process) dispatcher".into() };
    println!("\nqueue time, fake engine, D1 {d1_ms} ms per call, dispatcher: {where_}");
    println!("{:<16} {:<9} {:>8} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9}", "path", "jobs", "queue_m", "queue_50", "queue_mx", "disp_m", "disp_50", "disp_mx", "submit_m");
    for (path, one, five) in &rows {
        for (label, v) in [(format!("{n} x 1"), one), ("5 at once".to_owned(), five)] {
            let (qm, q50, qmx) = stats(&v.iter().map(|t| t.queue).collect::<Vec<_>>());
            let (dm, d50, dmx) = stats(&v.iter().map(|t| t.dispatch).collect::<Vec<_>>());
            let (sm, _, _) = stats(&v.iter().map(|t| t.submit).collect::<Vec<_>>());
            println!("{path:<16} {label:<9} {qm:>8.3} {q50:>8.3} {qmx:>8.3} {dm:>9.3} {d50:>9.3} {dmx:>9.3} {sm:>9.3}");
        }
    }
}

/// Phase 2 fencing: a worker cut off from the dispatcher (a partition, the
/// worker keeps running) loses its job to another worker after the grace
/// period; when it comes back its copy is cancelled, and none of its writes
/// replace the new holder's row.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_partitioned_worker_is_fenced_out() {
    init_log();
    let d = Dispatcher::new().await;
    let Some(native) = d.native.clone() else {
        eprintln!("skipped: needs the native dispatcher (a partition is simulated there)");
        return;
    };
    let sh = Shared::new(&d.token);
    let pool = pool_id("fence");
    let http = Http::new();
    let a = worker(&sh, "fa", &pool, Some(&d.base), 150, 1).await;
    let b = worker(&sh, "fb", &pool, Some(&d.base), 150, 1).await;
    connected(&d, &http, &pool, 2).await;
    let gw = gateway(&sh, &pool, DispatchMode::DurableObject, Some(&d.base), &[]).await;
    let g = gw.base.clone();
    available(&http, &g).await;
    let id = http.submit(&g, "across a partition").await;
    http.wait(&g, &id, |v| v["status"] == "running", Duration::from_secs(30)).await;
    let holder = sh.job_row(&id)["worker"].as_str().unwrap().to_owned();
    let (stale, survivor) = if holder == "worker-fa" { (&a, "worker-fb") } else { (&b, "worker-fa") };
    // Cut the holder off for longer than the grace period (1.5 s here).
    native.partition(&holder, 4_000);
    let done = http.finished(&g, &id).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    let r = sh.mock.sql("SELECT worker, lease FROM jobs WHERE external_id = ?", &[json!(id)]).unwrap().remove(0);
    assert_eq!(r["worker"], survivor, "{r:?}");
    assert_eq!(r["lease"].as_f64(), Some(2.0), "{r:?}");
    // The stale holder is fenced: its own copy was stopped, not written.
    let jid = sh.job(&id).id;
    let t0 = Instant::now();
    while !stale.app.d1.as_ref().unwrap().is_fenced(jid) {
        assert!(t0.elapsed() < Duration::from_secs(20), "the stale holder was never fenced");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    let r = sh.mock.sql("SELECT worker, status FROM jobs WHERE external_id = ?", &[json!(id)]).unwrap().remove(0);
    assert_eq!((r["worker"].as_str(), r["status"].as_str()), (Some(survivor), Some("succeeded")), "{r:?}");
    drop(gw);
}

/// Phase 2 restage: a worker that cannot fetch a job's input nacks 424;
/// the gateway sends the job again with its inputs in the store, and it
/// runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_424_restages_inputs_through_the_store() {
    use fastvideo_dispatch_proto::{DoMsg, WorkerMsg};
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::Message;
    init_log();
    let d = Dispatcher::new().await;
    let sh = Shared::new(&d.token);
    let pool = pool_id("rst");
    let http = Http::new();
    // A stand-in worker that answers every push with 424.
    let url = format!("{}/pools/{pool}/connect", d.base.replacen("http", "ws", 1));
    let mut req = url.into_client_request().unwrap();
    req.headers_mut().insert("x-fv-internal-token", d.token.parse().unwrap());
    req.headers_mut().insert("x-fv-worker-id", "flaky".parse().unwrap());
    let (ws, _) = tokio_tungstenite::connect_async(req).await.unwrap();
    let (mut sink, mut stream) = ws.split();
    let hello = WorkerMsg::Hello(fastvideo_dispatch_proto::Hello { worker_id: "flaky".into(), pool: pool.clone(), capacity: 1, ..Default::default() });
    sink.send(Message::text(serde_json::to_string(&hello).unwrap())).await.unwrap();
    connected(&d, &http, &pool, 1).await;
    let gw = gateway(&sh, &pool, DispatchMode::DurableObject, Some(&d.base), &[]).await;
    let g = gw.base.clone();
    available(&http, &g).await;
    let (s, v) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-h3-turbo", "prompt": "a fox", "image_url": noise_png(64, 48)}))).await;
    assert_eq!(s, 202, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    // The push arrives inline; answer 424, then leave.
    let first = loop {
        let Some(Ok(Message::Text(t))) = stream.next().await else { panic!("socket closed") };
        if let Ok(DoMsg::Job { job_id, attempt, envelope, .. }) = serde_json::from_str::<DoMsg>(t.as_str()) {
            break (job_id, attempt, envelope);
        }
    };
    assert!(first.2.to_string().contains("\"inline\""), "the first push carries the input inline");
    let nack = WorkerMsg::Nack { job_id: first.0.clone(), attempt: first.1, retry: true, code: 424, message: "cannot fetch".into() };
    sink.send(Message::text(serde_json::to_string(&nack).unwrap())).await.unwrap();
    drop((sink, stream));
    // A real worker joins; the gateway restages and the job runs there.
    let w = worker(&sh, "rw", &pool, Some(&d.base), 5, 1).await;
    let done = http.wait(&g, &id, |v| v["status"] == "succeeded", Duration::from_secs(60)).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    let st = d.status(&http, &pool).await;
    assert!(st.restage.is_empty() && st.failed.is_empty(), "{st:?}");
    drop((gw, w));
}

/// A PNG data URI of noise.
fn noise_png(w: u32, h: u32) -> String {
    use base64::Engine as _;
    let mut x: u32 = 0x9e37_79b9;
    let img = image::RgbImage::from_fn(w, h, |_, _| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        image::Rgb([x as u8, (x >> 8) as u8, (x >> 16) as u8])
    });
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(out.into_inner()))
}
