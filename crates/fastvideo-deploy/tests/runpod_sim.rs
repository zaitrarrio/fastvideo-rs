//! The Runpod queue worker against the local simulator (WP-16 acceptance):
//! take, ping, stop, progress, stream, done and its retries, 429/400
//! back-off, `kind: http` (with `wait`), `kind: stream`, `kind: info`,
//! shutdown and the prestart failure path.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_deploy::dispatch::{DispatchOpts, RouterHandler};
use fastvideo_deploy::runpod::sim::{Sim, SimStatus};
use fastvideo_deploy::runpod::{
    JobCtx, JobError, JobHandler, JobInput, RouterTransport, RunpodEnv, Transport, Worker, WorkerOptions, WorkerReport,
};
use serde_json::{json, Value};
use tokio::sync::oneshot;

const KEY: &str = "rp-test-key";

fn opts() -> WorkerOptions {
    WorkerOptions {
        poll_timeout: Duration::from_secs(2),
        backoff_429: Duration::from_millis(10),
        error_backoff: Duration::from_millis(10),
        idle_backoff: Duration::from_millis(5),
        backoff_unit: Duration::from_millis(10),
        cancel_grace: Duration::from_millis(300),
        shutdown_grace: Duration::from_millis(300),
        ..WorkerOptions::default()
    }
}

fn env_for(sim: &Sim, base: &str) -> RunpodEnv {
    let m: BTreeMap<String, String> = sim.env(base, "w-test", 100).into_iter().collect();
    RunpodEnv::from_lookup(|k| m.get(k).cloned()).unwrap()
}

/// A fake API: capabilities, an OpenAI-style video job that finishes on
/// the third poll, a MiniMax-style task, a stuck job, and native streams.
fn api() -> Router {
    #[derive(Clone, Default)]
    struct St {
        polls: Arc<AtomicUsize>,
        stream_polls: Arc<AtomicUsize>,
        deleted: Arc<AtomicUsize>,
    }
    let st = St::default();
    Router::new()
        .route("/fv/v1/capabilities", get(|| async { Json(json!({"object": "fv.capabilities", "models": [{"id": "fake-wan"}]})) }))
        .route("/v1/videos", post(|Json(b): Json<Value>| async move { Json(json!({"id": "vid_1", "status": "queued", "model": b["model"]})) }))
        .route(
            "/v1/videos/{id}",
            get(|State(s): State<St>, Path(id): Path<String>| async move {
                let n = s.polls.fetch_add(1, Ordering::SeqCst);
                let status = if n >= 2 { "completed" } else { "in_progress" };
                Json(json!({"id": id, "status": status, "progress": n * 50}))
            }),
        )
        .route("/v2/video_generation", post(|| async { Json(json!({"task_id": "106916112212032"})) }))
        .route(
            "/v2/query/video_generation",
            get(|Query(q): Query<BTreeMap<String, String>>| async move {
                Json(json!({"task_id": q["task_id"], "status": "Success", "file_id": "f1"}))
            }),
        )
        .route("/stuck", post(|| async { Json(json!({"id": "s1", "status": "queued"})) }))
        .route("/stuck/{id}", get(|| async { Json(json!({"status": "running"})) }))
        .route("/teapot", get(|| async { (StatusCode::IM_A_TEAPOT, "short and stout") }))
        .route("/fv/v1/streams", post(|Json(b): Json<Value>| async move {
            if b["whip_url"].as_str().unwrap_or_default().is_empty() {
                return (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "whip_url required"}})));
            }
            (StatusCode::CREATED, Json(json!({"id": "str_1", "status": "live", "model": b["model"], "has_token": b["whip_token"].is_string()})))
        }))
        .route(
            "/fv/v1/streams/{id}",
            get(|State(s): State<St>| async move {
                let n = s.stream_polls.fetch_add(1, Ordering::SeqCst);
                if n >= 2 {
                    Json(json!({"id": "str_1", "status": "ended", "frames": 480, "reason": "duration"}))
                } else {
                    Json(json!({"id": "str_1", "status": "live", "frames": n * 16}))
                }
            })
            .delete(|State(s): State<St>| async move {
                s.deleted.fetch_add(1, Ordering::SeqCst);
                Json(json!({"id": "str_1", "status": "stopped", "frames": 7}))
            }),
        )
        .with_state(st)
}

fn handler() -> RouterHandler {
    RouterHandler::new(api()).with_opts(DispatchOpts {
        poll_interval: Duration::from_millis(10),
        stream_poll: Duration::from_millis(10),
        ..DispatchOpts::default()
    })
}

struct Running {
    stop: oneshot::Sender<()>,
    done: tokio::task::JoinHandle<WorkerReport>,
}

impl Running {
    async fn finish(self) -> WorkerReport {
        let _ = self.stop.send(());
        self.done.await.unwrap()
    }
}

fn start<T: Transport, H: JobHandler>(env: RunpodEnv, t: T, h: H, o: WorkerOptions) -> Running {
    let (stop, rx) = oneshot::channel();
    let w = Worker::new(env, t, h, o);
    let done = tokio::spawn(w.run(async move {
        let _ = rx.await;
    }));
    Running { stop, done }
}

fn sim() -> Sim {
    Sim::new(KEY).with_long_poll(Duration::from_millis(20))
}

fn start_sim<H: JobHandler>(sim: &Sim, h: H, o: WorkerOptions) -> Running {
    start(env_for(sim, "http://sim.local"), RouterTransport(sim.router()), h, o)
}

const T: Duration = Duration::from_secs(10);

#[tokio::test]
async fn http_job_take_done_and_ping() {
    let sim = sim();
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "http", "method": "GET", "path": "/fv/v1/capabilities"}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Completed, "{j:?}");
    let out = j.output.clone().unwrap();
    assert_eq!(out["status"], 200);
    assert_eq!(out["body"]["models"][0]["id"], "fake-wan");
    assert_eq!(out["headers"]["content-type"], "application/json");
    assert_eq!(j.final_is_stream, Some(false));
    assert!(j.x_request_ids.iter().all(|r| r == &id));
    // Heartbeats carry the version; the first take was not "in progress".
    tokio::time::sleep(Duration::from_millis(250)).await;
    let pings = sim.pings();
    assert!(!pings.is_empty());
    assert!(pings.iter().all(|p| p.worker == "w-test" && p.version.starts_with("fv-rs/")));
    assert_eq!(sim.takes().first(), Some(&false));
    assert_eq!(sim.unauthorized(), 0);
    let r = w.finish().await;
    assert_eq!((r.taken, r.succeeded, r.failed), (1, 1, 0));
}

#[tokio::test]
async fn ping_lists_the_running_job() {
    let sim = sim();
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "http", "path": "/stuck", "wait": true, "timeout_s": 1}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    // WaitTimeout: failed, with the last status kept as output.
    assert_eq!(j.status, SimStatus::Failed);
    assert_eq!(j.error_report().unwrap()["error_type"], "WaitTimeout");
    assert_eq!(j.output.unwrap()["body"]["status"], "running");
    assert!(sim.pings().iter().any(|p| p.job_ids == vec![id.clone()]), "{:?}", sim.pings());
    w.finish().await;
}

#[tokio::test]
async fn wait_polls_to_terminal_with_progress() {
    let sim = sim();
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "http", "path": "/v1/videos", "headers": {"authorization": "Bearer k"},
        "body": {"model": "fake-wan", "prompt": "fox"}, "wait": true}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Completed, "{j:?}");
    let out = j.output.unwrap();
    assert_eq!(out["body"]["status"], "completed");
    assert_eq!(out["submit"]["id"], "vid_1");
    assert_eq!(out["poll_path"], "/v1/videos/vid_1");
    let states: Vec<&str> = j.progress.iter().filter_map(|p| p["state"].as_str()).collect();
    assert_eq!(states, ["in_progress", "completed"]);
    // MiniMax: task_id + the query route.
    let id = sim.submit(json!({"path": "/v2/video_generation", "body": {"model": "MiniMax-H3"}, "wait": true}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    let out = j.output.unwrap();
    assert_eq!(out["poll_path"], "/v2/query/video_generation?task_id=106916112212032");
    assert_eq!(out["body"]["status"], "Success");
    w.finish().await;
}

#[tokio::test]
async fn non_2xx_and_text_bodies_are_returned() {
    let sim = sim();
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "http", "method": "GET", "path": "/teapot", "wait": true}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    let out = j.output.unwrap();
    assert_eq!(out["status"], 418);
    assert_eq!(out["body"], "short and stout");
    w.finish().await;
}

#[tokio::test]
async fn done_retries_then_succeeds() {
    let sim = sim();
    sim.fail_next_done(2);
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "http", "method": "GET", "path": "/fv/v1/capabilities"}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Completed);
    assert_eq!(j.done_posts, 3, "two 500s, then accepted");
    let r = w.finish().await;
    assert_eq!(r.undelivered, 0);
}

#[tokio::test]
async fn done_retries_are_bounded() {
    let sim = sim();
    sim.fail_next_done(3);
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "http", "method": "GET", "path": "/fv/v1/capabilities"}));
    tokio::time::sleep(Duration::from_millis(500)).await;
    let j = sim.job(&id).unwrap();
    assert_eq!(j.done_posts, 3);
    assert_eq!(j.status, SimStatus::InProgress, "never accepted");
    let r = w.finish().await;
    assert_eq!(r.undelivered, 1);
}

#[tokio::test]
async fn take_backs_off_on_429_400_and_errors() {
    let sim = sim();
    sim.script_take(&[429, 400, 500, 204]);
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "info"}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Completed);
    assert!(j.output.unwrap()["version"].as_str().unwrap().starts_with("fv-rs/"));
    assert!(sim.takes().len() >= 5);
    w.finish().await;
}

#[tokio::test]
async fn stop_channel_cancels_the_job() {
    let sim = sim();
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "http", "path": "/stuck", "wait": true}));
    // Let it start.
    for _ in 0..200 {
        if sim.job(&id).unwrap().status == SimStatus::InProgress {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    sim.cancel(&id);
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Cancelled);
    let e = j.error_report().unwrap();
    assert_eq!(e["error_type"], "Cancelled");
    assert_eq!(e["worker_id"], "w-test");
    let r = w.finish().await;
    assert_eq!(r.cancelled, 1);
}

#[tokio::test]
async fn stream_job_goes_live_and_returns_stats() {
    let sim = sim();
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "stream", "model": "wan-sf", "prompt": "a fox", "whip_url": "https://whip.example/pub",
        "whip_token": "tok", "duration_s": 30}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Completed, "{j:?}");
    assert_eq!(j.progress[0]["state"], "live");
    assert_eq!(j.progress[0]["stream_id"], "str_1");
    assert_eq!(j.progress[0]["stream"]["has_token"], true);
    let out = j.output.unwrap();
    assert_eq!(out["status"], "ended");
    assert_eq!(out["frames"], 480);
    // A refused stream fails the job with the server's message.
    let id = sim.submit(json!({"kind": "stream", "model": "wan-sf", "whip_url": ""}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Failed);
    let e = j.error_report().unwrap();
    assert_eq!(e["error_type"], "StreamRefused");
    assert_eq!(e["error_message"], "whip_url required");
    w.finish().await;
}

/// Emits stream chunks, then a result.
struct Chunky;

#[async_trait::async_trait]
impl JobHandler for Chunky {
    async fn handle(&self, input: JobInput, cx: JobCtx) -> Result<Value, JobError> {
        if let JobInput::Info(_) = input {
            for i in 0..3 {
                cx.stream(json!({"chunk": i}));
            }
            cx.progress(json!({"state": "almost"}));
            return Ok(json!({"chunks": 3}));
        }
        // Everything else: run until cancelled (shutdown test).
        cx.cancel.cancelled().await;
        Err(JobError::new("Interrupted", "stopped"))
    }
}

#[tokio::test]
async fn stream_chunks_and_final_is_stream() {
    let sim = sim();
    let w = start_sim(&sim, Chunky, opts());
    let id = sim.submit(json!({"kind": "info"}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.stream, vec![json!({"chunk": 0}), json!({"chunk": 1}), json!({"chunk": 2})]);
    assert_eq!(j.progress, vec![json!({"state": "almost"})]);
    assert_eq!(j.final_is_stream, Some(true));
    assert_eq!(j.output.unwrap()["chunks"], 3);
    w.finish().await;
}

#[tokio::test]
async fn invalid_input_fails_the_job() {
    let sim = sim();
    let w = start_sim(&sim, handler(), opts());
    let id = sim.submit(json!({"kind": "shell", "cmd": "rm -rf /"}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Failed);
    let e = j.error_report().unwrap();
    assert_eq!(e["error_type"], "InvalidInput");
    assert!(e["error_message"].as_str().unwrap().contains("unknown job kind"));
    let id = sim.submit(json!({"kind": "http", "path": "/fv/v1/forward"}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.error_report().unwrap()["error_type"], "InvalidInput");
    w.finish().await;
}

#[tokio::test]
async fn shutdown_cancels_running_jobs_and_reports() {
    let sim = sim();
    let w = start_sim(&sim, Chunky, opts());
    let id = sim.submit(json!({"kind": "http", "path": "/whatever"}));
    for _ in 0..200 {
        if sim.job(&id).unwrap().status == SimStatus::InProgress {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let r = w.finish().await;
    assert_eq!(r.cancelled, 1, "{r:?}");
    let j = sim.job(&id).unwrap();
    assert_eq!(j.status, SimStatus::Failed);
    let e = j.error_report().unwrap();
    assert_eq!(e["error_type"], "Cancelled");
    assert!(e["error_message"].as_str().unwrap().contains("shutting down"), "{e}");
    // No takes after shutdown: a new job stays queued.
    let id2 = sim.submit(json!({"kind": "info"}));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(sim.job(&id2).unwrap().status, SimStatus::InQueue);
}

#[tokio::test]
async fn concurrency_two_uses_batch_take() {
    let sim = sim();
    let w = start_sim(&sim, handler(), WorkerOptions { concurrency: 2, ..opts() });
    let a = sim.submit(json!({"kind": "http", "path": "/v1/videos", "body": {"model": "m"}, "wait": true}));
    let b = sim.submit(json!({"kind": "info"}));
    assert_eq!(sim.wait_terminal(&a, T).await.unwrap().status, SimStatus::Completed);
    assert_eq!(sim.wait_terminal(&b, T).await.unwrap().status, SimStatus::Completed);
    let r = w.finish().await;
    assert_eq!(r.succeeded, 2);
}

#[tokio::test]
async fn prestart_failure_fails_one_job() {
    let sim = sim();
    let w = Worker::new(env_for(&sim, "http://sim.local"), RouterTransport(sim.router()), handler(), opts());
    let id = sim.submit(json!({"kind": "info"}));
    let id2 = sim.submit(json!({"kind": "info"}));
    let r = w.fail_one("CUDA init failed: no device", Duration::from_secs(5)).await;
    assert_eq!((r.taken, r.failed), (1, 1));
    let j = sim.job(&id).unwrap();
    assert_eq!(j.status, SimStatus::Failed);
    let e = j.error_report().unwrap();
    assert_eq!(e["error_type"], "WorkerStartupError");
    assert_eq!(e["error_message"], "CUDA init failed: no device");
    assert_eq!(sim.job(&id2).unwrap().status, SimStatus::InQueue);
}

#[tokio::test]
async fn wrong_key_is_rejected_by_the_simulator() {
    let sim = sim();
    let mut env = env_for(&sim, "http://sim.local");
    env.api_key = "wrong".into();
    let w = start(env, RouterTransport(sim.router()), handler(), opts());
    let id = sim.submit(json!({"kind": "info"}));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(sim.job(&id).unwrap().status, SimStatus::InQueue);
    assert!(sim.unauthorized() > 0);
    w.finish().await;
}

/// The same loop over real HTTP (reqwest) against the simulator on a TCP port.
#[cfg(feature = "runpod")]
#[tokio::test]
async fn reqwest_transport_over_tcp() {
    use fastvideo_deploy::runpod::ReqwestTransport;
    let sim = sim();
    let (base, server) = sim.serve("127.0.0.1:0").await.unwrap();
    let w = start(env_for(&sim, &base), ReqwestTransport::new().unwrap(), handler(), opts());
    let id = sim.submit(json!({"kind": "http", "path": "/v1/videos", "body": {"model": "m"}, "wait": true}));
    let j = sim.wait_terminal(&id, T).await.unwrap();
    assert_eq!(j.status, SimStatus::Completed);
    assert_eq!(j.output.unwrap()["body"]["status"], "completed");
    // Public client API: /run and /status.
    let c = reqwest::Client::new();
    let r: Value = c.post(format!("{base}/v2/sim-ep/run")).json(&json!({"input": {"kind": "info"}})).send().await.unwrap().json().await.unwrap();
    let id = r["id"].as_str().unwrap().to_owned();
    sim.wait_terminal(&id, T).await.unwrap();
    let s: Value = c.get(format!("{base}/v2/sim-ep/status/{id}")).send().await.unwrap().json().await.unwrap();
    assert_eq!(s["status"], "COMPLETED");
    w.finish().await;
    server.abort();
}
