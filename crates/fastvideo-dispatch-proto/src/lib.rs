//! Push dispatch between GPU workers and a pool dispatcher
//! (docs/serve/gateway-cloudflare.md §3.1-§3.3).
//!
//! A GPU worker (fv-serve, `server.role = "worker"`) keeps one outbound
//! WebSocket to its pool's dispatcher (`GET {dispatcher}/pools/{pool}/connect`,
//! [`TOKEN_HEADER`] and [`WORKER_HEADER`] on the upgrade), announces itself
//! with [`WorkerMsg::Hello`], and receives jobs pushed as [`DoMsg::Job`]. Its
//! [`WorkerMsg::Ack`] means the job is taken (inputs in place, D1 row
//! adopted, submitted to the engine). Progress and results keep going
//! through D1/R2 as on the gateway path; [`WorkerMsg::Done`] only frees the
//! slot.
//!
//! The dispatcher is a Cloudflare Durable Object (`crates/fastvideo-edge`)
//! today; the protocol and the scheduler ([`sched`]) are plain Rust with no
//! runtime, so fv-serve can host the same push model natively.
//!
//! Everything is JSON text frames, tagged by `"t"`. The dispatch envelope
//! (`{"job", "inputs", "attempt", "pool"}`, gateway.md §3) rides as an
//! opaque JSON value: the dispatcher never looks inside it.

pub mod sched;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Protocol version announced in [`Hello::proto`].
pub const PROTO_VERSION: u32 = 1;

/// Header carrying the internal token (the same secret as the gateway's
/// `x-fv-internal-token`).
pub const TOKEN_HEADER: &str = "x-fv-internal-token";
/// Header naming the connecting worker (`[A-Za-z0-9._-]`, ≤ 128 chars).
pub const WORKER_HEADER: &str = "x-fv-worker-id";

/// Keep-alive text frame a worker sends; the dispatcher answers
/// [`PONG`] without waking up (Durable Object auto-response).
pub const PING: &str = "ping";
pub const PONG: &str = "pong";

/// The dispatcher's routes under a pool, relative to its base URL.
pub fn connect_path(pool: &str) -> String {
    format!("/pools/{pool}/connect")
}
pub fn enqueue_path(pool: &str) -> String {
    format!("/pools/{pool}/enqueue")
}
pub fn cancel_path(pool: &str, job: &str) -> String {
    format!("/pools/{pool}/cancel/{job}")
}
pub fn status_path(pool: &str) -> String {
    format!("/pools/{pool}/status")
}

/// Whether `s` is a usable pool, worker or job id (path and tag safe).
pub fn valid_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// A job a worker holds or recently finished, as it reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Held {
    pub job_id: String,
    pub attempt: u32,
    /// The lease (fencing token) it was pushed with.
    #[serde(default)]
    pub lease: u64,
    /// `queued` | `running` (unfinished) or `succeeded` | `failed` |
    /// `cancelled` (finished).
    pub state: String,
}

impl Held {
    pub fn finished(&self) -> bool {
        matches!(self.state.as_str(), "succeeded" | "failed" | "cancelled")
    }
}

/// First frame of a worker on a new socket.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    pub worker_id: String,
    pub pool: String,
    #[serde(default)]
    pub proto: u32,
    /// fv-serve version and short git sha.
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub sha: String,
    /// Jobs it takes at once (running + waiting on its GPU queue).
    #[serde(default = "one")]
    pub capacity: u32,
    #[serde(default)]
    pub draining: bool,
    /// Model ids it serves (empty: any model of the pool).
    #[serde(default)]
    pub models: Vec<String>,
    /// Its caps as its `/fv/v1/internal/status` reports them (`models`), for
    /// the gateway's live caps; opaque here.
    #[serde(default)]
    pub caps: Value,
    /// Jobs it took and has not reported [`WorkerMsg::Done`] for, plus
    /// jobs finished while it was disconnected (reconcile, §3.3).
    #[serde(default)]
    pub jobs: Vec<Held>,
}

fn one() -> u32 {
    1
}

/// Worker → dispatcher.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum WorkerMsg {
    Hello(Hello),
    /// The job is taken (the adopt). `worker_ms`: time from receiving the
    /// push to this ack on the worker (input fetch, adopt, engine submit).
    Ack { job_id: String, attempt: u32, #[serde(default)] lease: u64, #[serde(default)] worker_ms: u64 },
    /// The job was not taken. `retry`: another worker (or later) may take
    /// it (busy, draining, held elsewhere); otherwise it is failed with
    /// `message`. `code` 424: a client URL it could not fetch; the job
    /// waits for the gateway to send it again with the input in the store
    /// ([`PoolStatus::restage`], [`EnqueueReq::replace`]).
    Nack { job_id: String, attempt: u32, retry: bool, #[serde(default)] code: u16, #[serde(default)] message: String },
    /// A taken job finished (its D1 row is already terminal).
    Done { job_id: String, attempt: u32, state: String },
    /// Load report (also the liveness heartbeat), every few seconds.
    Status { #[serde(default)] running: u32, #[serde(default)] draining: bool, #[serde(default = "one")] capacity: u32 },
}

/// Dispatcher → worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum DoMsg {
    /// Answer to [`WorkerMsg::Hello`]: the jobs it reported that it must
    /// drop (the dispatcher gave them to another worker meanwhile).
    Welcome { worker_id: String, pool: String, #[serde(default)] cancel: Vec<String> },
    /// A job to take: ack or nack. `lease` is its fencing token: the worker
    /// writes the job's row only while no newer lease holds it, so a worker
    /// the dispatcher gave up on cannot overwrite its successor.
    /// `takeover`: a re-dispatch after a worker loss.
    Job { job_id: String, attempt: u32, envelope: Value, #[serde(default)] lease: u64, #[serde(default)] takeover: bool },
    /// Stop a job (client cancel).
    Cancel { job_id: String },
    /// Take no new jobs (running ones finish).
    Drain { on: bool },
}

/// `POST /pools/{pool}/enqueue`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnqueueReq {
    pub job_id: String,
    /// The dispatch envelope, as the worker's `POST /fv/v1/internal/jobs`
    /// would take it.
    pub envelope: Value,
    /// Re-dispatches after a worker loss before the job is failed.
    #[serde(default = "one")]
    pub retries: u32,
    /// The model (for workers that announce a model list).
    #[serde(default)]
    pub model: Option<String>,
    /// Replace the envelope of a job waiting for a restage (its inputs now
    /// in the store) and queue it again.
    #[serde(default)]
    pub replace: bool,
}

/// Answer to an enqueue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnqueueResp {
    pub job_id: String,
    /// `pushed` (sent to a worker now) | `queued` | an existing job's state
    /// for a duplicate enqueue.
    pub state: String,
    #[serde(default)]
    pub worker: Option<String>,
    /// Jobs ahead of it in the dispatcher's queue.
    #[serde(default)]
    pub position: u32,
    #[serde(default)]
    pub duplicate: bool,
}

/// A worker as the dispatcher sees it (`GET /pools/{pool}/status`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub connected: bool,
    pub draining: bool,
    pub capacity: u32,
    /// Jobs pushed or running there.
    pub held: u32,
    pub version: String,
    pub sha: String,
    pub last_seen_ms: i64,
    #[serde(default)]
    pub caps: Value,
}

/// A job the dispatcher failed itself (worker lost after its retries, or
/// refused by a worker for good): the gateway marks its D1 row failed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FailedJob {
    pub job_id: String,
    pub attempt: u32,
    pub error: String,
    pub at_ms: i64,
}

/// Timings of the last dispatched jobs, milliseconds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DispatchTimings {
    pub count: u32,
    /// Enqueue received → pushed on a socket.
    pub queue_p50_ms: f64,
    pub queue_max_ms: f64,
    /// Pushed → ack received (network both ways plus the worker's take).
    pub ack_p50_ms: f64,
    pub ack_max_ms: f64,
    /// Of which on the worker (its `worker_ms`).
    pub worker_p50_ms: f64,
}

/// `GET /pools/{pool}/status`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PoolStatus {
    pub pool: String,
    pub now_ms: i64,
    pub queued: u32,
    pub pushed: u32,
    pub running: u32,
    pub workers: Vec<WorkerInfo>,
    /// Failed by the dispatcher in the last hour.
    #[serde(default)]
    pub failed: Vec<FailedJob>,
    #[serde(default)]
    pub timings: DispatchTimings,
    /// Jobs waiting for the gateway to put a client-URL input in the store
    /// (a worker could not fetch it) and enqueue them again with `replace`.
    #[serde(default)]
    pub restage: Vec<String>,
    /// Worker build of the dispatcher (the Worker script version).
    #[serde(default)]
    pub dispatcher: String,
}

impl PoolStatus {
    /// Connected workers that take new jobs.
    pub fn usable_workers(&self) -> usize {
        self.workers.iter().filter(|w| w.connected && !w.draining).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn frames_round_trip() {
        let m = WorkerMsg::Ack { job_id: "j".into(), attempt: 2, lease: 3, worker_ms: 7 };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(s, r#"{"t":"ack","job_id":"j","attempt":2,"lease":3,"worker_ms":7}"#);
        assert_eq!(serde_json::from_str::<WorkerMsg>(&s).unwrap(), m);
        let hello: WorkerMsg = serde_json::from_value(json!({"t": "hello", "worker_id": "w1", "pool": "p"})).unwrap();
        match hello {
            WorkerMsg::Hello(h) => {
                assert_eq!(h.capacity, 1);
                assert!(h.jobs.is_empty());
            }
            other => panic!("{other:?}"),
        }
        let job = DoMsg::Job { job_id: "j".into(), attempt: 1, envelope: json!({"job": {}}), lease: 1, takeover: false };
        let v = serde_json::to_value(&job).unwrap();
        assert_eq!(v["t"], "job");
        assert_eq!(serde_json::from_value::<DoMsg>(v).unwrap(), job);
    }

    #[test]
    fn ids() {
        assert!(valid_id("h3-turbo"));
        assert!(valid_id("w_1.a"));
        assert!(!valid_id(""));
        assert!(!valid_id("a/b"));
        assert!(!valid_id(&"x".repeat(129)));
    }
}
