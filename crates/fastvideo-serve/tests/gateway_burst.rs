//! Burst placement on the gateway path (docs/serve/gateway.md §3.3): the
//! gateway reserves a slot on a pod worker when it picks it, before the
//! dispatch call, so the concurrent submits of a burst spread over the
//! pool instead of all landing on the worker that looked emptiest at the
//! last probe (docs/serve/gateway-cloudflare.md §9.4 found 5 jobs on 5
//! idle workers waiting 5-11 s on one GPU).
//!
//! Covered on the fake engine (fv-serve workers on loopback TCP, one shared
//! D1 mock with a round-trip latency):
//! - a burst of N jobs over M workers spreads evenly and runs in parallel
//!   (queue ≈ one job time per round of ⌈N/M⌉);
//! - a worker lost mid-burst (its jobs run again on the others);
//! - reservations released when the dispatch call fails (refusal, 5xx,
//!   connection error);
//! - two gateway replicas sharing the pool and D1 (one dispatch row per
//!   job, a lost job re-dispatched once);
//! - the before/after measurement: 5 jobs on 1 worker and on 3 workers.

#![cfg(feature = "http-client")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_serve::config::{Config, EngineBackendKind, JobBackend, KeyStoreBackend, PoolCfg, PoolKind, Role};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::client::{D1Error, D1Transport, RawReply};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-burst-user";
const TOKEN: &str = "burst-internal-token";
/// Fake step time (4 steps of `fake-h3-turbo`; its D1 writes add to it).
const STEP_MS: u64 = 250;

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fv-burst-{tag}-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ))
}

/// A D1 transport that can be cut (a lost worker stops writing), with a
/// round-trip latency (50 ms; the measurement uses the real D1 API's 250 ms).
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
        tokio::time::sleep(self.delay).await;
        self.mock.post(body).await
    }
}

struct Shared {
    mock: MockD1,
    arts: PathBuf,
    d1_delay: Duration,
}

impl Shared {
    fn new() -> Self {
        let arts = tmp("arts");
        std::fs::create_dir_all(&arts).unwrap();
        Self { mock: MockD1::new(), arts, d1_delay: Duration::from_millis(50) }
    }
    fn d1(&self, dead: &Arc<AtomicBool>) -> D1Client {
        D1Client::new(Arc::new(Cuttable { mock: self.mock.clone(), dead: dead.clone(), delay: self.d1_delay }))
            .with_retry(fastvideo_serve_kit::d1::RetryPolicy::immediate(1))
    }
    fn rows(&self, sql: &str) -> Vec<serde_json::Map<String, Value>> {
        self.mock.sql(sql, &[]).unwrap()
    }
}

fn base_config(tag: &str, sh: &Shared) -> Config {
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
    c.jobs.progress_interval_ms = 100;
    c.jobs.heartbeat_s = 1;
    c.auth.key_store = KeyStoreBackend::Memory;
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
    /// Stops answering HTTP and writing D1 (a lost worker).
    fn kill(&self) {
        self.dead.store(true, Ordering::SeqCst);
        self.task.abort();
    }
}

async fn listen() -> (tokio::net::TcpListener, String) {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    (l, base)
}

async fn serve(c: Config, sh: &Shared, listener: tokio::net::TcpListener, base: String) -> Running {
    let dead = Arc::new(AtomicBool::new(false));
    let app = App::build(c, Overrides { d1: Some(sh.d1(&dead)), ..Default::default() }).await.unwrap();
    app.gate.engine().wait_ready().await;
    let router = app.router.clone();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    Running { app, base, dead, task }
}

/// A fake-engine pod worker serving `fake-h3-turbo` (one executor).
async fn pod_worker(sh: &Shared, tag: &str) -> Running {
    let (l, base) = listen().await;
    let mut c = base_config(tag, sh);
    c.server.role = Role::Worker;
    c.server.public_base_url = Some(base.clone());
    c.server.worker_id = Some(format!("worker-{tag}"));
    c.engine.fake.models = vec!["fake-h3-turbo".into()];
    c.engine.fake.step_ms = STEP_MS;
    c.validate().unwrap();
    serve(c, sh, l, base).await
}

fn pod_pool(urls: &[String]) -> PoolCfg {
    PoolCfg {
        id: "h3".into(),
        kind: PoolKind::Pod,
        urls: urls.to_vec(),
        fake_models: vec!["fake-h3-turbo".into()],
        stale_after_s: 3,
        retries: 1,
        ..PoolCfg::default()
    }
}

async fn gateway(sh: &Shared, tag: &str, urls: &[String]) -> Running {
    let (l, base) = listen().await;
    let mut c = base_config(tag, sh);
    c.engine.backend = EngineBackendKind::Remote;
    c.server.public_base_url = Some(base.clone());
    c.pools = vec![pod_pool(urls)];
    c.gateway.tick_s = 1;
    c.gateway.watch_poll_ms = 100;
    c.gateway.runpod_api_base = "http://127.0.0.1:9/v2".into();
    c.gateway.runpod_api_key = fastvideo_serve::config::Secret("rp-key".into());
    c.validate().unwrap();
    serve(c, sh, l, base).await
}

fn http() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

/// Submits `n` jobs at once through `gw`; returns their ids (the native API's).
async fn burst(gws: &[&str], n: usize, tag: &str) -> Vec<String> {
    let c = http();
    let calls = (0..n).map(|i| {
        let c = c.clone();
        let g = gws[i % gws.len()].to_owned();
        async move {
            let r = c
                .post(format!("{g}/fv/v1/jobs"))
                .bearer_auth(KEY)
                .json(&json!({"model": "fake-h3-turbo", "prompt": format!("{tag} {i}")}))
                .send()
                .await
                .unwrap();
            let s = r.status().as_u16();
            let v: Value = r.json().await.unwrap();
            assert_eq!(s, 202, "{v}");
            v["id"].as_str().unwrap().to_owned()
        }
    });
    futures::future::join_all(calls).await
}

/// One finished job as D1 has it.
#[derive(Debug)]
struct Done {
    worker: String,
    status: String,
    /// `created_at` → `started_at` (fal `timings.queue`).
    queue: f64,
    started: time::OffsetDateTime,
    completed: time::OffsetDateTime,
}

/// Waits until every job in `ids` is finished; returns them.
async fn wait_all(sh: &Shared, ids: &[String]) -> Vec<Done> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let mut out = Vec::new();
        for id in ids {
            let r = sh.mock.sql("SELECT job, worker FROM jobs WHERE external_id = ?", &[json!(id)]).unwrap();
            let Some(r) = r.first() else { break };
            let job: fastvideo_protocol::Job = serde_json::from_str(r["job"].as_str().unwrap()).unwrap();
            if !job.is_terminal() {
                break;
            }
            let (Some(st), Some(done)) = (job.started_at, job.completed_at) else {
                out.push(Done { worker: String::new(), status: job.status().as_str().into(), queue: f64::NAN, started: job.created_at, completed: job.created_at });
                continue;
            };
            out.push(Done {
                worker: r["worker"].as_str().unwrap_or_default().to_owned(),
                status: job.status().as_str().into(),
                queue: (st - job.created_at).as_seconds_f64(),
                started: st,
                completed: done,
            });
        }
        if out.len() == ids.len() {
            return out;
        }
        assert!(Instant::now() < deadline, "jobs never finished");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn per_worker(done: &[Done]) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for d in done {
        *m.entry(d.worker.clone()).or_insert(0) += 1;
    }
    m
}

/// Mean run time (`started_at` → `completed_at`): the job time.
fn run_time(done: &[Done]) -> f64 {
    done.iter().map(|d| (d.completed - d.started).as_seconds_f64()).sum::<f64>() / done.len() as f64
}

fn stats(done: &[Done]) -> (f64, f64) {
    let q: Vec<f64> = done.iter().map(|d| d.queue).collect();
    (q.iter().sum::<f64>() / q.len() as f64, q.iter().cloned().fold(0.0, f64::max))
}

/// Whether some pair of jobs on different workers ran at the same time.
fn overlapped(done: &[Done]) -> bool {
    done.iter().any(|a| done.iter().any(|b| a.worker != b.worker && a.started < b.completed && b.started < a.completed))
}

/// Reservations and placements left on the gateway's view of the pool.
fn leftovers(gw: &Running) -> Vec<(String, u32)> {
    let g = gw.app.gateway.as_ref().unwrap();
    let st = g.pools[0].lock();
    st.workers.values().filter(|w| w.reserved > 0).map(|w| (w.url.clone(), w.reserved)).collect()
}

/// N jobs over M workers: ⌈N/M⌉ per worker, running in parallel, the
/// k-th round waiting about k-1 job times.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_burst_spreads_evenly_and_runs_in_parallel() {
    let sh = Shared::new();
    let ws = [pod_worker(&sh, "sa").await, pod_worker(&sh, "sb").await, pod_worker(&sh, "sc").await];
    let urls: Vec<String> = ws.iter().map(|w| w.base.clone()).collect();
    let gw = gateway(&sh, "s-gw", &urls).await;
    // Let a probe see the workers (capacity, queue_max) first.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let ids = burst(&[gw.base.as_str()], 6, "spread").await;
    let done = wait_all(&sh, &ids).await;
    assert!(done.iter().all(|d| d.status == "succeeded"), "{done:?}");
    let pw = per_worker(&done);
    assert_eq!(pw.len(), 3, "every worker took part: {pw:?}");
    assert!(pw.values().all(|n| *n == 2), "⌈6/3⌉ = 2 jobs per worker: {pw:?}");
    assert!(overlapped(&done), "the workers ran jobs in parallel: {done:?}");
    let (mean, max) = stats(&done);
    // Round 1 waits for the dispatch (≈ 3 D1 round trips), round 2 one job
    // time more (a job's time here includes its D1 progress writes).
    let job = run_time(&done);
    println!("spread: queue mean {mean:.2} s, max {max:.2} s; job time {job:.2} s");
    assert!(max < 1.5 * job + 1.0, "queue max {max:.2} s (mean {mean:.2} s) for a 2-round burst of {job:.2} s jobs");
    assert!(leftovers(&gw).is_empty(), "{:?}", leftovers(&gw));
    // Nothing unreported survives the next probe.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let g = gw.app.gateway.as_ref().unwrap();
    assert!(g.pools[0].lock().workers.values().all(|w| w.unreported == 0 && w.load() == 0 && w.capacity == 1 && w.queue_max == 32));
    drop((gw, ws));
}

/// A worker disappears while the burst runs: its jobs are dispatched again
/// on the others (not on it), everything finishes, nothing stays reserved.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_worker_lost_mid_burst() {
    let sh = Shared::new();
    let ws = [pod_worker(&sh, "la").await, pod_worker(&sh, "lb").await, pod_worker(&sh, "lc").await];
    let urls: Vec<String> = ws.iter().map(|w| w.base.clone()).collect();
    let gw = gateway(&sh, "l-gw", &urls).await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let ids = burst(&[gw.base.as_str()], 6, "loss").await;
    // Kill the worker of the first job as soon as it runs.
    let victim = loop {
        let r = sh.mock.sql("SELECT worker FROM jobs WHERE external_id = ? AND status = 'running'", &[json!(ids[0])]).unwrap();
        if let Some(w) = r.first().and_then(|r| r["worker"].as_str()) {
            break w.to_owned();
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let vi = ws.iter().position(|w| w.app.config.server.worker_id.as_deref() == Some(victim.as_str())).unwrap();
    ws[vi].kill();
    let done = wait_all(&sh, &ids).await;
    assert!(done.iter().all(|d| d.status == "succeeded"), "{done:?}");
    assert!(done.iter().all(|d| d.worker != victim), "nothing finished on the lost worker: {done:?}");
    let d = sh.rows("SELECT attempt FROM gw_dispatch");
    let attempts: Vec<i64> = d.iter().map(|r| r["attempt"].as_f64().unwrap() as i64).collect();
    assert!(attempts.contains(&2) && attempts.iter().all(|a| *a <= 2), "{attempts:?}");
    assert!(leftovers(&gw).is_empty(), "{:?}", leftovers(&gw));
    drop((gw, ws));
}

/// A stand-in pod worker: answers the probe as a ready, idle worker and
/// every dispatch after `delay` with `status` (or drops the connection).
async fn failing_worker(status: u16, delay: Duration) -> (String, tokio::task::JoinHandle<()>) {
    use axum::routing::{get, post};
    let (l, base) = listen().await;
    let app = axum::Router::new()
        .route(
            "/fv/v1/internal/status",
            get(|| async {
                axum::Json(json!({"worker_id": "stand-in", "readiness": "ready", "draining": false, "capacity": 1, "queue_max": 4,
                    "stats": {"queued_batch": 0, "queued_stream": 0, "running": 0, "sessions": 0}, "models": []}))
            }),
        )
        .route(
            "/fv/v1/internal/jobs",
            post(move || async move {
                tokio::time::sleep(delay).await;
                let code = axum::http::StatusCode::from_u16(status).unwrap();
                let kind = if status < 500 { "invalid_request" } else { "internal" };
                (code, axum::Json(json!({"error": {"kind": kind, "message": "refused by the stand-in"}})))
            }),
        );
    let t = tokio::spawn(async move {
        let _ = axum::serve(l, app).await;
    });
    (base, t)
}

/// Every failed dispatch call gives its reservation back: while the calls
/// are in progress the burst's reservations are spread over the workers;
/// once they failed (5xx, 4xx, a connection error) none is left.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn reservations_are_released_when_the_dispatch_fails() {
    let sh = Shared::new();
    let (a, _ta) = failing_worker(500, Duration::from_millis(800)).await;
    let (b, _tb) = failing_worker(503, Duration::from_millis(800)).await;
    let gw = gateway(&sh, "f-gw", &[a.clone(), b.clone()]).await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let g = gw.app.gateway.clone().unwrap();
    let c = http();
    let submit = |i: usize| {
        let c = c.clone();
        let base = gw.base.clone();
        async move {
            let r = c.post(format!("{base}/fv/v1/jobs")).bearer_auth(KEY).json(&json!({"model": "fake-h3-turbo", "prompt": format!("fail {i}")})).send().await.unwrap();
            r.status().as_u16()
        }
    };
    let calls = tokio::spawn(futures::future::join_all((0..4).map(submit)));
    // While the calls hang on the workers: 4 reservations, 2 per worker.
    let mut seen = BTreeMap::new();
    for _ in 0..40 {
        seen = g.pools[0].lock().workers.values().map(|w| (w.url.clone(), w.reserved)).collect::<BTreeMap<_, _>>();
        if seen.values().sum::<u32>() == 4 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(seen.get(&a), Some(&2), "{seen:?}");
    assert_eq!(seen.get(&b), Some(&2), "{seen:?}");
    let codes = calls.await.unwrap();
    assert!(codes.iter().all(|c| *c == 503), "no worker took them: {codes:?}");
    assert!(leftovers(&gw).is_empty(), "{:?}", leftovers(&gw));
    assert!(g.pools[0].lock().workers.values().all(|w| w.placed.is_empty()));

    // A refusal of the job itself (4xx): released too.
    let (c4, _t4) = failing_worker(400, Duration::ZERO).await;
    let gw4 = gateway(&sh, "f4-gw", std::slice::from_ref(&c4)).await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let r = c.post(format!("{}/fv/v1/jobs", gw4.base)).bearer_auth(KEY).json(&json!({"model": "fake-h3-turbo", "prompt": "bad"})).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 400);
    assert!(leftovers(&gw4).is_empty(), "{:?}", leftovers(&gw4));

    // A worker that went away between probes (connection refused).
    let (l, gone) = listen().await;
    let (ok, _tok) = failing_worker(500, Duration::ZERO).await;
    let gwc = gateway(&sh, "fc-gw", &[gone.clone(), ok.clone()]).await;
    drop(l);
    let r = c.post(format!("{}/fv/v1/jobs", gwc.base)).bearer_auth(KEY).json(&json!({"model": "fake-h3-turbo", "prompt": "gone"})).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 503);
    assert!(leftovers(&gwc).is_empty(), "{:?}", leftovers(&gwc));
    drop((gw, gw4, gwc));
}

/// Two gateway replicas in front of one pool, sharing D1: each spreads its
/// share of the burst; when a worker is lost, both reapers see its jobs and
/// the dispatch-row claim lets exactly one re-dispatch each (attempt 2,
/// never 3); every job has one dispatch row.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn two_gateway_replicas_share_the_pool() {
    let sh = Shared::new();
    let ws = [pod_worker(&sh, "ra").await, pod_worker(&sh, "rb").await, pod_worker(&sh, "rc").await];
    let urls: Vec<String> = ws.iter().map(|w| w.base.clone()).collect();
    let g1 = gateway(&sh, "r-gw1", &urls).await;
    let g2 = gateway(&sh, "r-gw2", &urls).await;
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let ids = burst(&[g1.base.as_str(), g2.base.as_str()], 6, "replicas").await;
    let done = wait_all(&sh, &ids).await;
    assert!(done.iter().all(|d| d.status == "succeeded"), "{done:?}");
    let pw = per_worker(&done);
    assert_eq!(pw.len(), 3, "{pw:?}");
    assert!(pw.values().all(|n| *n <= 3), "{pw:?}");
    let (mean, max) = stats(&done);
    let job = run_time(&done);
    assert!(max < 2.5 * job + 1.0, "queue max {max:.2} s (mean {mean:.2} s), job time {job:.2} s");

    // A second burst; a worker is lost while it runs.
    let ids2 = burst(&[g1.base.as_str(), g2.base.as_str()], 6, "replicas-loss").await;
    let victim = loop {
        let r = sh.mock.sql("SELECT worker FROM jobs WHERE external_id = ? AND status = 'running'", &[json!(ids2[0])]).unwrap();
        if let Some(w) = r.first().and_then(|r| r["worker"].as_str()) {
            break w.to_owned();
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let vi = ws.iter().position(|w| w.app.config.server.worker_id.as_deref() == Some(victim.as_str())).unwrap();
    ws[vi].kill();
    let done2 = wait_all(&sh, &ids2).await;
    assert!(done2.iter().all(|d| d.status == "succeeded" && d.worker != victim), "{done2:?}");
    let rows = sh.rows("SELECT d.job_id AS id, d.attempt AS attempt FROM gw_dispatch d");
    assert_eq!(rows.len(), 12, "one dispatch row per job: {rows:?}");
    let attempts: Vec<i64> = rows.iter().map(|r| r["attempt"].as_f64().unwrap() as i64).collect();
    assert!(attempts.contains(&2) && attempts.iter().all(|a| *a <= 2), "each lost job re-dispatched once: {attempts:?}");
    assert!(leftovers(&g1).is_empty() && leftovers(&g2).is_empty());
    drop((g1, g2, ws));
}

/// The measurement (docs/serve/gateway.md §3.3): 5 jobs at once through the
/// gateway on 1 worker and on 3 workers (fake jobs of 4 x 250 ms steps, D1 at 250 ms).
/// Prints the queue times (`created_at` → `started_at`). On 3 workers the
/// burst takes 2 rounds (⌈5/3⌉), so no job waits two job times.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn burst_queue_times_one_worker_vs_three() {
    let mut out = Vec::new();
    for m in [1usize, 3] {
        let mut sh = Shared::new();
        sh.d1_delay = Duration::from_millis(250);
        let mut ws = Vec::new();
        for i in 0..m {
            ws.push(pod_worker(&sh, &format!("m{m}-{i}")).await);
        }
        let urls: Vec<String> = ws.iter().map(|w| w.base.clone()).collect();
        let gw = gateway(&sh, &format!("m{m}-gw"), &urls).await;
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let t0 = Instant::now();
        let ids = burst(&[gw.base.as_str()], 5, &format!("measure-{m}")).await;
        let done = wait_all(&sh, &ids).await;
        let wall = t0.elapsed().as_secs_f64();
        assert!(done.iter().all(|d| d.status == "succeeded"), "{done:?}");
        let (mean, max) = stats(&done);
        let mut q: Vec<f64> = done.iter().map(|d| d.queue).collect();
        q.sort_by(f64::total_cmp);
        out.push((m, mean, max, wall, run_time(&done), per_worker(&done), q));
        drop((gw, ws));
    }
    println!("fake-engine gateway burst: 5 jobs at once, {STEP_MS} ms x 4 steps per job, D1 0.25 s per call");
    println!("{:<8} {:>7} {:>7} {:>7} {:>7}  per worker / queue times (s)", "workers", "mean_q", "max_q", "wall_s", "job_s");
    for (m, mean, max, wall, job, pw, q) in &out {
        println!("{m:<8} {mean:>7.2} {max:>7.2} {wall:>7.2} {job:>7.2}  {:?} {:?}", pw.values().collect::<Vec<_>>(), q.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
    }
    let (_, mean3, max3, _, job3, pw3, _) = &out[1];
    let (_, mean1, _, _, _, _, _) = &out[0];
    assert_eq!(pw3.len(), 3, "{pw3:?}");
    assert!(pw3.values().all(|n| *n <= 2), "⌈5/3⌉: {pw3:?}");
    assert!(*max3 < 1.5 * job3 + 1.0, "3 workers: max queue {max3:.2} s, job time {job3:.2} s");
    assert!(*mean3 < mean1 / 2.0, "3 workers wait less than half as long as 1: {mean3:.2} vs {mean1:.2}");
}
