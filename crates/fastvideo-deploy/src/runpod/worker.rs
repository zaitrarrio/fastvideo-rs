//! The worker loop: take, ping, stop, per-job progress/stream/done.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::{mpsc, watch, Notify};

use super::{Cancel, InResp, JobCtx, JobError, JobHandler, JobInput, Method, OutReq, RawJob, RunpodEnv, Transport, Update};

/// Timing knobs (defaults follow `runpod-python`).
#[derive(Clone, Debug)]
pub struct WorkerOptions {
    /// Jobs run at once (1: one GPU, one job).
    pub concurrency: usize,
    /// Client timeout of a job-take / job-stop long poll (SDK: 90 s).
    pub poll_timeout: Duration,
    /// Back-off after a 429 (SDK: 5 s).
    pub backoff_429: Duration,
    /// Back-off after a transport error or unexpected status.
    pub error_backoff: Duration,
    /// Pause after an empty take (204/400) in case the server does not
    /// long-poll.
    pub idle_backoff: Duration,
    /// job-done attempts (SDK: 3, Fibonacci back-off in `backoff_unit`s).
    pub done_attempts: u32,
    pub backoff_unit: Duration,
    /// Time a cancelled job gets to wind down before it is dropped.
    pub cancel_grace: Duration,
    /// On shutdown: time running jobs get to finish before cancellation.
    pub shutdown_grace: Duration,
    /// `runpod_version`.
    pub version: String,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            concurrency: 1,
            poll_timeout: Duration::from_secs(90),
            backoff_429: Duration::from_secs(5),
            error_backoff: Duration::from_secs(1),
            idle_backoff: Duration::from_millis(200),
            done_attempts: 3,
            backoff_unit: Duration::from_secs(1),
            cancel_grace: Duration::from_secs(10),
            shutdown_grace: Duration::from_secs(25),
            version: super::version(),
        }
    }
}

/// What the loop did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkerReport {
    pub taken: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub cancelled: u64,
    /// job-done posts that never got a 2xx.
    pub undelivered: u64,
}

#[derive(Default)]
struct Counters {
    taken: AtomicU64,
    succeeded: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    undelivered: AtomicU64,
}

struct Shared<T, H> {
    env: RunpodEnv,
    opts: WorkerOptions,
    transport: T,
    handler: H,
    inflight: Mutex<BTreeMap<String, Cancel>>,
    changed: Notify,
    counters: Counters,
    /// Why running jobs were cancelled by the worker itself (shutdown).
    shutting_down: watch::Sender<bool>,
}

/// The Runpod queue worker.
pub struct Worker<T: Transport, H: JobHandler> {
    shared: Arc<Shared<T, H>>,
}

fn fib_backoff(unit: Duration, attempt: u32) -> Duration {
    let (mut a, mut b) = (1u32, 1u32);
    for _ in 0..attempt {
        (a, b) = (b, a.saturating_add(b));
    }
    unit * a
}

impl<T: Transport, H: JobHandler> Worker<T, H> {
    pub fn new(env: RunpodEnv, transport: T, handler: H, opts: WorkerOptions) -> Self {
        Self {
            shared: Arc::new(Shared {
                env,
                opts,
                transport,
                handler,
                inflight: Mutex::new(BTreeMap::new()),
                changed: Notify::new(),
                counters: Counters::default(),
                shutting_down: watch::channel(false).0,
            }),
        }
    }

    /// Runs until `shutdown` resolves, then stops taking, lets running jobs
    /// finish within `shutdown_grace`, cancels the rest (they report
    /// `{"error":…}`), and returns.
    pub async fn run(self, shutdown: impl Future<Output = ()> + Send) -> WorkerReport {
        let s = self.shared.clone();
        let (stop_tx, stop_rx) = watch::channel(false);
        let ping = tokio::spawn(ping_loop(s.clone(), stop_rx.clone()));
        let stopper = tokio::spawn(stop_loop(s.clone(), stop_rx.clone()));
        tracing::info!(worker = %s.env.worker_id, concurrency = s.opts.concurrency, "runpod worker: taking jobs");
        tokio::pin!(shutdown);
        tokio::select! {
            _ = take_loop(s.clone()) => {}
            _ = &mut shutdown => tracing::info!("runpod worker: shutdown requested, no more takes"),
        }
        s.shutting_down.send_replace(true);
        // Let running jobs finish.
        let drained = tokio::time::timeout(s.opts.shutdown_grace, wait_idle(&s)).await.is_ok();
        if !drained {
            let ids: Vec<String> = {
                let m = s.inflight.lock().unwrap_or_else(|p| p.into_inner());
                m.iter().map(|(id, c)| {
                    c.cancel();
                    id.clone()
                }).collect()
            };
            tracing::warn!(jobs = ?ids, "runpod worker: cancelling jobs still running after the grace period");
            let _ = tokio::time::timeout(s.opts.cancel_grace + Duration::from_secs(30), wait_idle(&s)).await;
        }
        stop_tx.send_replace(true);
        let _ = stopper.await;
        let _ = ping.await;
        report(&s)
    }

    /// Startup failed (CUDA init, weights): claim one job, fail it with
    /// `reason`, and return (the caller exits 1), like the SDK's prestart
    /// path. `wait` bounds how long to wait for a job to fail.
    pub async fn fail_one(self, reason: &str, wait: Duration) -> WorkerReport {
        let s = self.shared.clone();
        let deadline = tokio::time::Instant::now() + wait;
        while tokio::time::Instant::now() < deadline {
            match take_once(&s, false).await {
                Taken::Jobs(jobs) => {
                    for j in jobs {
                        s.counters.taken.fetch_add(1, Ordering::Relaxed);
                        s.counters.failed.fetch_add(1, Ordering::Relaxed);
                        let body = error_body(&s, &JobError::new("WorkerStartupError", reason));
                        post_done(&s, &j.id, &body, false).await;
                    }
                    break;
                }
                Taken::Wait(d) => tokio::time::sleep(d).await,
            }
        }
        report(&s)
    }
}

fn report<T: Transport, H: JobHandler>(s: &Shared<T, H>) -> WorkerReport {
    let c = &s.counters;
    WorkerReport {
        taken: c.taken.load(Ordering::Relaxed),
        succeeded: c.succeeded.load(Ordering::Relaxed),
        failed: c.failed.load(Ordering::Relaxed),
        cancelled: c.cancelled.load(Ordering::Relaxed),
        undelivered: c.undelivered.load(Ordering::Relaxed),
    }
}

fn inflight_ids<T: Transport, H: JobHandler>(s: &Shared<T, H>) -> Vec<String> {
    s.inflight.lock().unwrap_or_else(|p| p.into_inner()).keys().cloned().collect()
}

async fn wait_idle<T: Transport, H: JobHandler>(s: &Shared<T, H>) {
    loop {
        let n = s.changed.notified();
        if s.inflight.lock().unwrap_or_else(|p| p.into_inner()).is_empty() {
            return;
        }
        n.await;
    }
}

fn auth_headers<T: Transport, H: JobHandler>(s: &Shared<T, H>) -> Vec<(String, String)> {
    vec![("Authorization".into(), s.env.api_key.clone())]
}

enum Taken {
    Jobs(Vec<RawJob>),
    Wait(Duration),
}

/// One job-take request.
async fn take_once<T: Transport, H: JobHandler>(s: &Shared<T, H>, in_progress: bool) -> Taken {
    let free = s.opts.concurrency.saturating_sub(inflight_ids(s).len()).max(1);
    let url = if free > 1 { s.env.take_batch_url(in_progress, free) } else { s.env.take_url(in_progress) };
    let req = OutReq { method: Method::Get, url, headers: auth_headers(s), body: None, timeout: s.opts.poll_timeout };
    match s.transport.send(req).await {
        Ok(InResp { status: 200, body }) => {
            let v: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "job-take: unparseable 200 body");
                    return Taken::Wait(s.opts.error_backoff);
                }
            };
            let items = match v {
                Value::Array(a) => a,
                Value::Null => Vec::new(),
                o => vec![o],
            };
            let jobs: Vec<RawJob> = items
                .into_iter()
                .filter_map(|j| match serde_json::from_value::<RawJob>(j) {
                    Ok(j) => Some(j),
                    Err(e) => {
                        tracing::warn!(error = %e, "job-take: job without id/input");
                        None
                    }
                })
                .collect();
            if jobs.is_empty() {
                Taken::Wait(s.opts.idle_backoff)
            } else {
                Taken::Jobs(jobs)
            }
        }
        // 204: no job; 400: expected under FlashBoot.
        Ok(InResp { status: 204 | 400, .. }) => Taken::Wait(s.opts.idle_backoff),
        Ok(InResp { status: 429, .. }) => Taken::Wait(s.opts.backoff_429),
        Ok(InResp { status, .. }) => {
            tracing::warn!(status, "job-take: unexpected status");
            Taken::Wait(s.opts.error_backoff)
        }
        Err(e) => {
            tracing::warn!(error = %e, "job-take failed");
            Taken::Wait(s.opts.error_backoff)
        }
    }
}

async fn take_loop<T: Transport, H: JobHandler>(s: Arc<Shared<T, H>>) {
    loop {
        // Wait for a free slot.
        loop {
            let n = s.changed.notified();
            if inflight_ids(&s).len() < s.opts.concurrency {
                break;
            }
            n.await;
        }
        let busy = !inflight_ids(&s).is_empty();
        match take_once(&s, busy).await {
            Taken::Jobs(jobs) => {
                for j in jobs {
                    start_job(&s, j);
                }
            }
            Taken::Wait(d) => tokio::time::sleep(d).await,
        }
    }
}

fn start_job<T: Transport, H: JobHandler>(s: &Arc<Shared<T, H>>, job: RawJob) {
    let cancel = Cancel::default();
    {
        let mut m = s.inflight.lock().unwrap_or_else(|p| p.into_inner());
        if m.contains_key(&job.id) {
            tracing::warn!(job = %job.id, "job-take returned a job that is already running; ignored");
            return;
        }
        m.insert(job.id.clone(), cancel.clone());
    }
    s.counters.taken.fetch_add(1, Ordering::Relaxed);
    let s = s.clone();
    tokio::spawn(async move {
        let id = job.id.clone();
        run_job(&s, job, cancel).await;
        s.inflight.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
        s.changed.notify_waiters();
    });
}

async fn run_job<T: Transport, H: JobHandler>(s: &Arc<Shared<T, H>>, job: RawJob, cancel: Cancel) {
    let id = job.id.clone();
    let input = match JobInput::parse(&job.input) {
        Ok(i) => i,
        Err(e) => {
            s.counters.failed.fetch_add(1, Ordering::Relaxed);
            let body = error_body(s, &JobError::new("InvalidInput", e));
            post_done(s, &id, &body, false).await;
            return;
        }
    };
    tracing::info!(job = %id, kind = input.kind(), "job started");
    let t0 = std::time::Instant::now();
    let (tx, rx) = mpsc::unbounded_channel();
    let forwarder = tokio::spawn(forward_updates(s.clone(), id.clone(), rx));
    let cx = JobCtx::new(id.clone(), cancel.clone(), tx);
    let res = {
        let handled = s.handler.handle(input, cx);
        tokio::pin!(handled);
        tokio::select! {
            r = &mut handled => r,
            _ = async { cancel.cancelled().await; tokio::time::sleep(s.opts.cancel_grace).await; } => {
                Err(JobError::new("Cancelled", "the job was cancelled and did not stop within the grace period"))
            }
        }
    };
    // The handler's context (and sender) is gone: the forwarder drains and ends.
    let streamed = forwarder.await.unwrap_or(false);
    let res = match res {
        Err(e) if cancel.is_cancelled() && e.kind != "Cancelled" => {
            let why = if *s.shutting_down.borrow() { "the worker is shutting down" } else { "the job was cancelled" };
            Err(JobError { kind: "Cancelled".into(), message: format!("{why}: {}", e.message), output: e.output })
        }
        Ok(_) if cancel.is_cancelled() && *s.shutting_down.borrow() => Err(JobError::new("Cancelled", "the worker is shutting down")),
        r => r,
    };
    let body = match &res {
        Ok(v) => {
            s.counters.succeeded.fetch_add(1, Ordering::Relaxed);
            json!({"output": v})
        }
        Err(e) => {
            if e.kind == "Cancelled" {
                s.counters.cancelled.fetch_add(1, Ordering::Relaxed);
            } else {
                s.counters.failed.fetch_add(1, Ordering::Relaxed);
            }
            error_body(s, e)
        }
    };
    tracing::info!(job = %id, ok = res.is_ok(), secs = t0.elapsed().as_secs_f64(), "job finished");
    post_done(s, &id, &body, streamed).await;
}

/// `{"error": "<json>"}` with the SDK's report fields (plus `output` if any).
fn error_body<T: Transport, H: JobHandler>(s: &Shared<T, H>, e: &JobError) -> Value {
    let report = json!({
        "error_type": e.kind,
        "error_message": e.message,
        "hostname": s.env.hostname,
        "worker_id": s.env.worker_id,
        "runpod_version": s.opts.version,
    });
    let mut body = json!({"error": report.to_string()});
    if let Some(o) = &e.output {
        body["output"] = o.clone();
    }
    body
}

/// Posts progress and stream chunks in order; returns whether any stream
/// chunk was sent (the final job-done then carries `isStream=true`).
async fn forward_updates<T: Transport, H: JobHandler>(s: Arc<Shared<T, H>>, id: String, mut rx: mpsc::UnboundedReceiver<Update>) -> bool {
    let mut streamed = false;
    while let Some(u) = rx.recv().await {
        let (url, body) = match u {
            Update::Progress(p) => (s.env.done_url(&id, false), json!({"status": "IN_PROGRESS", "output": p})),
            Update::Stream(chunk) => match s.env.stream_url(&id) {
                Some(u) => {
                    streamed = true;
                    (u, json!({"output": chunk}))
                }
                None => {
                    tracing::warn!(job = %id, "stream chunk dropped: RUNPOD_WEBHOOK_POST_STREAM is not set");
                    continue;
                }
            },
        };
        // Progress is best effort: one attempt.
        if let Err(e) = post_json(&s, &id, url, &body).await {
            tracing::warn!(job = %id, error = %e, "progress/stream post failed");
        }
    }
    streamed
}

async fn post_json<T: Transport, H: JobHandler>(s: &Shared<T, H>, id: &str, url: String, body: &Value) -> Result<(), String> {
    let mut headers = auth_headers(s);
    headers.push(("Content-Type".into(), "application/x-www-form-urlencoded".into()));
    headers.push(("charset".into(), "utf-8".into()));
    headers.push(("X-Request-ID".into(), id.to_owned()));
    let req = OutReq { method: Method::Post, url, headers, body: Some(body.to_string().into_bytes()), timeout: Duration::from_secs(60) };
    match s.transport.send(req).await {
        Ok(r) if (200..300).contains(&r.status) => Ok(()),
        Ok(r) => Err(format!("HTTP {}", r.status)),
        Err(e) => Err(e),
    }
}

/// job-done with `done_attempts` tries and Fibonacci back-off.
async fn post_done<T: Transport, H: JobHandler>(s: &Shared<T, H>, id: &str, body: &Value, is_stream: bool) {
    let attempts = s.opts.done_attempts.max(1);
    for attempt in 0..attempts {
        match post_json(s, id, s.env.done_url(id, is_stream), body).await {
            Ok(()) => return,
            Err(e) => {
                tracing::warn!(job = %id, attempt = attempt + 1, error = %e, "job-done failed");
                if attempt + 1 < attempts {
                    tokio::time::sleep(fib_backoff(s.opts.backoff_unit, attempt)).await;
                }
            }
        }
    }
    s.counters.undelivered.fetch_add(1, Ordering::Relaxed);
    tracing::error!(job = %id, "job-done was never accepted; the platform will time the job out");
}

async fn ping_loop<T: Transport, H: JobHandler>(s: Arc<Shared<T, H>>, mut stop: watch::Receiver<bool>) {
    loop {
        if let Some(url) = s.env.ping_url(&inflight_ids(&s), &s.opts.version) {
            let req = OutReq { method: Method::Get, url, headers: auth_headers(&s), body: None, timeout: s.env.ping_interval * 2 };
            match s.transport.send(req).await {
                Ok(r) if (200..300).contains(&r.status) => {}
                Ok(r) => tracing::debug!(status = r.status, "ping"),
                Err(e) => tracing::debug!(error = %e, "ping failed"),
            }
        }
        tokio::select! {
            _ = tokio::time::sleep(s.env.ping_interval) => {}
            _ = stop.wait_for(|v| *v) => return,
        }
    }
}

async fn stop_loop<T: Transport, H: JobHandler>(s: Arc<Shared<T, H>>, mut stop: watch::Receiver<bool>) {
    let Some(url) = s.env.stop_url() else { return };
    loop {
        let req = OutReq { method: Method::Get, url: url.clone(), headers: auth_headers(&s), body: None, timeout: s.opts.poll_timeout };
        let pause = tokio::select! {
            r = s.transport.send(req) => match r {
                Ok(InResp { status: 200, body }) => {
                    let ids: Vec<String> = serde_json::from_slice::<Value>(&body)
                        .ok()
                        .and_then(|v| v.get("jobsToStop").cloned())
                        .and_then(|v| serde_json::from_value(v).ok())
                        .unwrap_or_default();
                    let m = s.inflight.lock().unwrap_or_else(|p| p.into_inner());
                    for id in ids {
                        if let Some(c) = m.get(&id) {
                            tracing::info!(job = %id, "job-stop: cancelling");
                            c.cancel();
                        }
                    }
                    Duration::ZERO
                }
                Ok(InResp { status: 204, .. }) => s.opts.idle_backoff,
                Ok(InResp { status: 429, .. }) => s.opts.backoff_429,
                // Endpoints without the stop channel.
                Ok(InResp { status: 404 | 405, .. }) => Duration::from_secs(60),
                Ok(_) | Err(_) => s.opts.error_backoff.max(Duration::from_millis(500)),
            },
            _ = stop.wait_for(|v| *v) => return,
        };
        tokio::select! {
            _ = tokio::time::sleep(pause) => {}
            _ = stop.wait_for(|v| *v) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci() {
        let u = Duration::from_secs(1);
        let v: Vec<u64> = (0..5).map(|i| fib_backoff(u, i).as_secs()).collect();
        assert_eq!(v, [1, 1, 2, 3, 5]);
    }
}
