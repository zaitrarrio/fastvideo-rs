//! Per-job overhead on the fake engine (docs/serve/gateway.md §3.5): one
//! job end to end on a single server and through the gateway with one pod
//! worker, with D1 mocked at 0 ms and at 250 ms per call.
//!
//! - `overhead_budget_on_the_fake_engine` (always runs): at 0 ms D1 the job
//!   is visible as finished to a client (status poll, SSE, webhook) within
//!   the simulated work plus a small margin, on both paths.
//! - `waterfall` (`--ignored --nocapture`): prints the timeline of every
//!   instrumented point (the `debug` events of the submit pipeline, gateway,
//!   worker, engine pump, output, store; every D1 call; the R2 mock; the
//!   client's poll, SSE and webhook) and the gaps over 100 ms.

#![cfg(feature = "http-client")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use fastvideo_serve::config::{Config, EngineBackendKind, JobBackend, KeyStoreBackend, PoolCfg, PoolKind, Role};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::client::{D1Error, D1Transport, RawReply};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-overhead-user";
const TOKEN: &str = "overhead-internal-token";
/// Fake step time: 4 steps of `fake-h3-turbo` = 1 s of simulated work.
const STEP_MS: u64 = 250;
const STEPS: u64 = 4;

// ---- the timeline --------------------------------------------------------

struct Mark {
    at: Instant,
    what: String,
}

fn marks() -> &'static Mutex<Option<Vec<Mark>>> {
    static M: OnceLock<Mutex<Option<Vec<Mark>>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(None))
}

fn mark(what: impl Into<String>) {
    if let Some(v) = marks().lock().unwrap().as_mut() {
        v.push(Mark { at: Instant::now(), what: what.into() });
    }
}

fn start_recording() {
    install_layer();
    *marks().lock().unwrap() = Some(Vec::new());
}

fn stop_recording() -> Vec<Mark> {
    let mut v = marks().lock().unwrap().take().unwrap_or_default();
    v.sort_by_key(|m| m.at);
    v
}

/// Records the `debug`+ events of the fastvideo crates as marks.
struct Rec;

#[derive(Default)]
struct Msg(String);

impl tracing::field::Visit for Msg {
    fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
        if f.name() == "message" {
            self.0 = format!("{v:?}");
        }
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Rec {
    fn on_event(&self, ev: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        if !ev.metadata().target().starts_with("fastvideo") {
            return;
        }
        let mut m = Msg::default();
        ev.record(&mut m);
        mark(m.0);
    }
}

fn install_layer() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::Layer;
        let sub = tracing_subscriber::registry().with(Rec.with_filter(tracing_subscriber::filter::LevelFilter::DEBUG));
        let _ = tracing::subscriber::set_global_default(sub);
    });
}

// ---- mocks -----------------------------------------------------------------

/// The D1 mock behind a round-trip latency; every call is a mark.
#[derive(Clone)]
struct TimedD1 {
    mock: MockD1,
    delay: Duration,
}

fn sql_label(body: &Value) -> String {
    let one = |s: &Value| -> String {
        let sql = s["sql"].as_str().unwrap_or("?").split_whitespace().collect::<Vec<_>>().join(" ");
        sql.chars().take(48).collect()
    };
    match body.get("batch").and_then(Value::as_array) {
        Some(b) => format!("batch[{}] {}", b.len(), b.first().map(one).unwrap_or_default()),
        None => one(body),
    }
}

#[async_trait::async_trait]
impl D1Transport for TimedD1 {
    async fn post(&self, body: &Value) -> Result<RawReply, D1Error> {
        let l = sql_label(body);
        mark(format!("d1 > {l}"));
        tokio::time::sleep(self.delay).await;
        let r = self.mock.post(body).await;
        mark(format!("d1 < {l}"));
        r
    }
}

/// The shared artifacts directory with an R2-like PUT latency; marks.
struct R2Mock {
    inner: fastvideo_serve_kit::LocalArtifactStore,
    put: Duration,
}

impl fastvideo_protocol::UrlSigner for R2Mock {
    fn url_for(&self, a: &fastvideo_protocol::Artifact, ttl: Duration) -> url::Url {
        fastvideo_protocol::UrlSigner::url_for(&self.inner, a, ttl)
    }
}

#[async_trait::async_trait]
impl fastvideo_serve_kit::ArtifactStore for R2Mock {
    async fn put(&self, src: &std::path::Path, meta: fastvideo_serve_kit::ArtifactMeta) -> Result<fastvideo_protocol::Artifact, fastvideo_protocol::ApiError> {
        mark("r2 > PUT");
        tokio::time::sleep(self.put).await;
        let r = self.inner.put(src, meta).await;
        mark("r2 < PUT");
        r
    }
    async fn delete(&self, a: &fastvideo_protocol::Artifact) {
        self.inner.delete(a).await
    }
    fn signer(&self) -> &dyn fastvideo_protocol::UrlSigner {
        self
    }
    async fn open(&self, a: &fastvideo_protocol::Artifact) -> Result<fastvideo_serve_kit::ArtifactBody, fastvideo_protocol::ApiError> {
        self.inner.open(a).await
    }
}

// ---- servers ---------------------------------------------------------------

/// One measured setup.
#[derive(Clone, Debug)]
struct Scn {
    name: &'static str,
    gateway: bool,
    d1_ms: u64,
    r2_put_ms: u64,
    /// The fake engine writes its MP4 through ffmpeg/libx264 (when present).
    x264: bool,
    /// `jobs.progress_interval_ms`, `jobs.heartbeat_s`, `gateway.watch_poll_ms`
    /// (the burst test's settings were 100 / 1 / 100; defaults 1000 / 60 / 1000).
    progress_ms: u64,
    heartbeat_s: u64,
    watch_poll_ms: u64,
    /// The client's status poll interval.
    poll_ms: u64,
    /// Fake step time (4 steps).
    step_ms: u64,
}

impl Scn {
    fn new(name: &'static str, gateway: bool, d1_ms: u64) -> Self {
        Self { name, gateway, d1_ms, r2_put_ms: 0, x264: false, progress_ms: 1000, heartbeat_s: 60, watch_poll_ms: 1000, poll_ms: 20, step_ms: STEP_MS }
    }
    fn work_ms(&self) -> f64 {
        (self.step_ms * STEPS) as f64
    }
}

struct Shared {
    mock: MockD1,
    arts: PathBuf,
    d1_delay: Duration,
    r2_put: Duration,
}

fn tmp(tag: &str) -> PathBuf {
    tempfile::Builder::new().prefix(&format!("fv-overhead-{tag}-")).tempdir().unwrap().keep()
}

fn base_config(tag: &str, sh: &Shared, s: &Scn) -> Config {
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "shared-signing-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), tmp(tag).display().to_string());
    env.insert("FV_INTERNAL_TOKEN".to_owned(), TOKEN.to_owned());
    env.insert("FV_ARTIFACTS_DIR".to_owned(), sh.arts.display().to_string());
    env.insert("FV_CF_ACCOUNT_ID".to_owned(), "acct".to_owned());
    env.insert("FV_CF_API_TOKEN".to_owned(), "tok".to_owned());
    env.insert("FV_D1_DATABASE_ID".to_owned(), "db".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::D1;
    c.jobs.progress_interval_ms = s.progress_ms;
    c.jobs.heartbeat_s = s.heartbeat_s;
    c.auth.key_store = KeyStoreBackend::Memory;
    c.server.callbacks_allow_private = true;
    c.protocols.fastwan = false;
    c
}

fn engine_config(c: &mut Config, s: &Scn) {
    c.engine.fake.models = vec!["fake-h3-turbo".into()];
    c.engine.fake.step_ms = s.step_ms;
    c.engine.fake.mp4 = s.x264;
}

struct Running {
    _app: App,
    base: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Running {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn listen() -> (tokio::net::TcpListener, String) {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    (l, base)
}

async fn serve(c: Config, sh: &Shared, listener: tokio::net::TcpListener, base: String) -> Running {
    let d1 = D1Client::new(Arc::new(TimedD1 { mock: sh.mock.clone(), delay: sh.d1_delay })).with_retry(fastvideo_serve_kit::d1::RetryPolicy::immediate(1));
    let urls = fastvideo_serve_kit::artifacts::LocalUrls { public_base: base.parse().unwrap(), key: fastvideo_serve_kit::UrlKey::new("shared-signing-key") };
    let arts: Arc<dyn fastvideo_serve_kit::ArtifactStore> = Arc::new(R2Mock { inner: fastvideo_serve_kit::LocalArtifactStore::new(sh.arts.clone(), urls), put: sh.r2_put });
    let app = App::build(c, Overrides { d1: Some(d1), artifacts: Some(arts), ..Default::default() }).await.unwrap();
    app.gate.engine().wait_ready().await;
    let router = app.router.clone();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    Running { _app: app, base, task }
}

/// The server the client talks to, and what stands behind it.
struct Setup {
    entry: Running,
    _worker: Option<Running>,
}

async fn setup(s: &Scn) -> Setup {
    let sh = Shared {
        mock: MockD1::new(),
        arts: tmp("arts"),
        d1_delay: Duration::from_millis(s.d1_ms),
        r2_put: Duration::from_millis(s.r2_put_ms),
    };
    std::fs::create_dir_all(&sh.arts).unwrap();
    if !s.gateway {
        let (l, base) = listen().await;
        let mut c = base_config("single", &sh, s);
        c.server.public_base_url = Some(base.clone());
        engine_config(&mut c, s);
        c.validate().unwrap();
        return Setup { entry: serve(c, &sh, l, base).await, _worker: None };
    }
    let (wl, wbase) = listen().await;
    let mut wc = base_config("worker", &sh, s);
    wc.server.role = Role::Worker;
    wc.server.public_base_url = Some(wbase.clone());
    wc.server.worker_id = Some("worker-overhead".into());
    engine_config(&mut wc, s);
    wc.validate().unwrap();
    let worker = serve(wc, &sh, wl, wbase.clone()).await;
    let (gl, gbase) = listen().await;
    let mut gc = base_config("gw", &sh, s);
    gc.engine.backend = EngineBackendKind::Remote;
    gc.server.public_base_url = Some(gbase.clone());
    gc.pools = vec![PoolCfg {
        id: "h3".into(),
        kind: PoolKind::Pod,
        urls: vec![wbase],
        fake_models: vec!["fake-h3-turbo".into()],
        stale_after_s: 30,
        retries: 1,
        ..PoolCfg::default()
    }];
    gc.gateway.tick_s = 1;
    gc.gateway.watch_poll_ms = s.watch_poll_ms;
    gc.gateway.runpod_api_base = "http://127.0.0.1:9/v2".into();
    gc.gateway.runpod_api_key = fastvideo_serve::config::Secret("rp-key".into());
    gc.validate().unwrap();
    let entry = serve(gc, &sh, gl, gbase).await;
    // Let a probe see the worker.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    Setup { entry, _worker: Some(worker) }
}

// ---- one job -----------------------------------------------------------------

/// Client-side milestones (ms after the submit call started).
#[derive(Clone, Debug, Default)]
struct Seen {
    accepted: f64,
    poll_running: Option<f64>,
    poll_done: f64,
    sse_done: Option<f64>,
    webhook: Option<f64>,
}

async fn webhook_sink() -> (String, tokio::sync::mpsc::UnboundedReceiver<Instant>, tokio::task::JoinHandle<()>) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let (l, base) = listen().await;
    let app = axum::Router::new().route(
        "/hook",
        axum::routing::post(move |body: axum::body::Bytes| {
            let tx = tx.clone();
            async move {
                let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                mark(format!("client: webhook received ({})", v["status"].as_str().unwrap_or("?")));
                let _ = tx.send(Instant::now());
                "ok"
            }
        }),
    );
    let t = tokio::spawn(async move {
        let _ = axum::serve(l, app).await;
    });
    (format!("{base}/hook"), rx, t)
}

fn http() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

/// Runs one fal job (`minimax/h3-turbo`) with a webhook, a status poll and
/// an SSE stream; returns the client milestones and the timeline.
async fn one_job(s: &Scn) -> (Seen, Vec<Mark>) {
    let st = setup(s).await;
    let (hook, mut hooks, _ht) = webhook_sink().await;
    let base = st.entry.base.clone();
    let c = http();
    let auth = format!("Key {KEY}");
    start_recording();
    let t0 = Instant::now();
    let ms = |t: Instant| (t - t0).as_secs_f64() * 1e3;
    mark("client: submit");
    let r = c
        .post(format!("{base}/minimax/h3-turbo/text-to-video?fal_webhook={hook}"))
        .header("authorization", &auth)
        .json(&json!({"prompt": format!("overhead {}", s.name), "seed": 1}))
        .send()
        .await
        .unwrap();
    let code = r.status().as_u16();
    let v: Value = r.json().await.unwrap();
    assert_eq!(code, 200, "{v}");
    mark("client: submit answered");
    let mut seen = Seen { accepted: ms(Instant::now()), ..Seen::default() };
    let rid = v["request_id"].as_str().unwrap().to_owned();
    // SSE: fal's status stream.
    let sse = {
        let (c, auth, url) = (c.clone(), auth.clone(), format!("{base}/minimax/h3-turbo/requests/{rid}/status/stream"));
        tokio::spawn(async move {
            let mut r = c.get(url).header("authorization", auth).send().await.unwrap();
            let mut buf = String::new();
            while let Ok(Some(chunk)) = r.chunk().await {
                buf.push_str(&String::from_utf8_lossy(&chunk));
                if buf.contains("COMPLETED") {
                    mark("client: SSE COMPLETED");
                    return Some(Instant::now());
                }
            }
            None
        })
    };
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let r = c.get(format!("{base}/minimax/h3-turbo/requests/{rid}/status")).header("authorization", &auth).send().await.unwrap();
        let v: Value = r.json().await.unwrap_or(Value::Null);
        match v["status"].as_str() {
            Some("IN_PROGRESS") if seen.poll_running.is_none() => {
                mark("client: poll IN_PROGRESS");
                seen.poll_running = Some(ms(Instant::now()));
            }
            Some("COMPLETED") => {
                mark("client: poll COMPLETED");
                seen.poll_done = ms(Instant::now());
                assert!(v.get("error").is_none_or(Value::is_null), "{v}");
                break;
            }
            _ => {}
        }
        assert!(Instant::now() < deadline, "the job never finished: {v}");
        tokio::time::sleep(Duration::from_millis(s.poll_ms)).await;
    }
    seen.sse_done = tokio::time::timeout(Duration::from_secs(10), sse).await.ok().and_then(|r| r.ok().flatten()).map(ms);
    seen.webhook = tokio::time::timeout(Duration::from_secs(10), hooks.recv()).await.ok().flatten().map(ms);
    // Late background writes (the terminal flush on a slow D1).
    tokio::time::sleep(Duration::from_millis(300)).await;
    let tl = stop_recording();
    let r = c.get(format!("{base}/minimax/h3-turbo/requests/{rid}")).header("authorization", &auth).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    drop(st);
    (seen, tl)
}

fn print_waterfall(s: &Scn, seen: &Seen, tl: &[Mark]) {
    let Some(t0) = tl.first().map(|m| m.at) else { return };
    println!("\n=== {} (gateway: {}, D1 {} ms, R2 PUT {} ms, x264: {}, progress {} ms, heartbeat {} s, watch poll {} ms, client poll {} ms, steps 4 x {} ms)",
        s.name, s.gateway, s.d1_ms, s.r2_put_ms, s.x264, s.progress_ms, s.heartbeat_s, s.watch_poll_ms, s.poll_ms, s.step_ms);
    let mut prev = t0;
    let mut prev_key = t0;
    let mut gaps = Vec::new();
    let mut d1_calls = 0;
    for m in tl {
        let t = (m.at - t0).as_secs_f64() * 1e3;
        let d = (m.at - prev).as_secs_f64() * 1e3;
        let is_d1 = m.what.starts_with("d1 ");
        if m.what.starts_with("d1 >") {
            d1_calls += 1;
        }
        // Gaps between the milestones (D1 call marks aside: they overlap).
        if !is_d1 {
            let g = (m.at - prev_key).as_secs_f64() * 1e3;
            if g > 100.0 {
                gaps.push((g, m.what.clone()));
            }
            prev_key = m.at;
        }
        println!("{t:>8.1} {d:>+8.1}  {}", m.what);
        prev = m.at;
    }
    println!("-- D1 calls: {d1_calls}; gaps over 100 ms before: {}", gaps.iter().map(|(g, w)| format!("[{g:.0} ms] {w}")).collect::<Vec<_>>().join("; "));
    println!(
        "-- client: accepted {:.0} ms, poll IN_PROGRESS {:?}, poll COMPLETED {:.0} ms, SSE {:?}, webhook {:?} (simulated work {} ms)",
        seen.accepted, seen.poll_running.map(|x| x.round()), seen.poll_done, seen.sse_done.map(|x| x.round()), seen.webhook.map(|x| x.round()), s.work_ms()
    );
}

/// The waterfall of one job per setup (prints; `--ignored --nocapture`).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "measurement: prints the per-job waterfall"]
async fn waterfall() {
    let mut out = Vec::new();
    let burst_like = Scn { progress_ms: 100, heartbeat_s: 1, watch_poll_ms: 100, poll_ms: 100, x264: true, ..Scn::new("gateway, burst-test settings, x264", true, 250) };
    let scns = [
        Scn::new("single, D1 0 ms", false, 0),
        Scn::new("single, D1 250 ms", false, 250),
        Scn::new("gateway, D1 0 ms", true, 0),
        Scn::new("gateway, D1 250 ms", true, 250),
        Scn { step_ms: 300, ..Scn::new("gateway, D1 0 ms, 4 x 300 ms", true, 0) },
        Scn { step_ms: 300, ..Scn::new("gateway, D1 250 ms, 4 x 300 ms", true, 250) },
        Scn { r2_put_ms: 1000, ..Scn::new("gateway, D1 250 ms, R2 PUT 1 s", true, 250) },
        Scn { x264: true, ..Scn::new("single, D1 0 ms, x264", false, 0) },
        burst_like,
    ];
    for s in scns {
        let (seen, tl) = one_job(&s).await;
        print_waterfall(&s, &seen, &tl);
        out.push((s, seen));
    }
    println!("\n{:<44} {:>9} {:>9} {:>9} {:>9}", "setup", "poll", "SSE", "webhook", "overhead");
    for (s, seen) in &out {
        let work = s.work_ms();
        println!("{:<44} {:>9.0} {:>9.0} {:>9.0} {:>9.0}", s.name, seen.poll_done, seen.sse_done.unwrap_or(f64::NAN), seen.webhook.unwrap_or(f64::NAN), seen.poll_done - work);
    }
}

/// The overhead budget: a job's completion reaches the client (status
/// poll, SSE, webhook) within the simulated work plus a margin (submit,
/// pump, output, store writes, the client's own 20 ms poll) at 0 ms D1, on
/// a single server and through the gateway; with D1 at 250 ms, within the
/// round trips a job needs (the gateway's insert, the worker's adopt, the
/// terminal write) plus the same margin. The steps (4 x 300 ms) end between
/// two ticks of the gateway's 1 s watch poll, which a job used to wait for.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn overhead_budget_on_the_fake_engine() {
    // Debug build on a loaded CI host: generous next to the ~1 s poll tick
    // and the serial writes this guards against.
    let margin = 400.0;
    for (s, rtts) in [
        (Scn { step_ms: 300, ..Scn::new("single, D1 0 ms", false, 0) }, 0.0),
        (Scn { step_ms: 300, ..Scn::new("gateway, D1 0 ms", true, 0) }, 0.0),
        (Scn { step_ms: 300, ..Scn::new("gateway, D1 250 ms", true, 250) }, 3.0),
    ] {
        let (seen, tl) = one_job(&s).await;
        print_waterfall(&s, &seen, &tl);
        let budget = s.work_ms() + rtts * s.d1_ms as f64 + margin;
        let sse = seen.sse_done.expect("SSE saw the job finish");
        let hook = seen.webhook.expect("the webhook arrived");
        for (what, v) in [("poll", seen.poll_done), ("SSE", sse), ("webhook", hook)] {
            assert!(v <= budget, "{}: {what} saw the job finish {v:.0} ms after submit (budget {budget:.0} ms)", s.name);
        }
    }
}
