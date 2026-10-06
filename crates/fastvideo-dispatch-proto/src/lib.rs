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
//!
//! **Protocol 2** (docs/serve/dispatch-do-family.md): one dispatcher per
//! model *family* (`/families/{family}/…`, object name `family:{family}`),
//! credit-based offers ([`WorkerMsg::Slots`]), direct output uploads through
//! URLs the dispatcher mints ([`WorkerMsg::UploadInit`] …, [`presign`]) and
//! streaming-session admission ([`SessionReq`], [`DoMsg::SessionOffer`]).
//! Protocol-1 workers keep working against either kind of object.

pub mod presign;
pub mod sched;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Protocol version announced in [`Hello::proto`].
pub const PROTO_VERSION: u32 = 2;

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

/// Where a dispatcher lives: a pool (protocol 1 routes) or a model family.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    Pool(String),
    Family(String),
}

impl Scope {
    /// Route prefix: `/pools/{pool}` or `/families/{family}`.
    pub fn base_path(&self) -> String {
        match self {
            Scope::Pool(p) => format!("/pools/{p}"),
            Scope::Family(f) => format!("/families/{f}"),
        }
    }
    /// The Durable Object's name (`{pool}` or `family:{family}`).
    pub fn object_name(&self) -> String {
        match self {
            Scope::Pool(p) => p.clone(),
            Scope::Family(f) => format!("family:{f}"),
        }
    }
    /// The bare pool or family id.
    pub fn id(&self) -> &str {
        match self {
            Scope::Pool(p) | Scope::Family(p) => p,
        }
    }
    pub fn is_family(&self) -> bool {
        matches!(self, Scope::Family(_))
    }
    pub fn connect_path(&self) -> String {
        format!("{}/connect", self.base_path())
    }
    pub fn enqueue_path(&self) -> String {
        format!("{}/enqueue", self.base_path())
    }
    pub fn cancel_path(&self, job: &str) -> String {
        format!("{}/cancel/{job}", self.base_path())
    }
    pub fn status_path(&self) -> String {
        format!("{}/status", self.base_path())
    }
    pub fn metrics_path(&self) -> String {
        format!("{}/metrics", self.base_path())
    }
    pub fn sessions_path(&self) -> String {
        format!("{}/sessions", self.base_path())
    }
    pub fn session_renew_path(&self, id: &str) -> String {
        format!("{}/sessions/{id}/renew", self.base_path())
    }
    pub fn session_release_path(&self, id: &str) -> String {
        format!("{}/sessions/{id}/release", self.base_path())
    }
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
    /// Protocol 2: sessions it holds from this dispatcher (re-announced
    /// after a reconnect).
    #[serde(default)]
    pub sessions: Vec<HeldSession>,
    /// Protocol 2: its public base URL (signalling for sessions goes there
    /// directly; media flows client ↔ GPU).
    #[serde(default)]
    pub endpoint: String,
    /// Protocol 2: its credits at connect time (absent: protocol 1, the
    /// dispatcher places by `capacity`).
    #[serde(default)]
    pub slots: Option<Slots>,
}

/// A session a worker holds, as it re-announces it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeldSession {
    pub session_id: String,
    pub lease: u64,
}

/// A worker's credits for one dispatcher (docs/serve/dispatch-do-family.md §6.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slots {
    /// Jobs its arbiter would take now.
    pub free: u32,
    /// Sessions its arbiter would take now.
    #[serde(default)]
    pub session_free: u32,
    /// Offers ([`DoMsg::Job`]) received from this dispatcher on this socket
    /// when `free` was computed.
    #[serde(default)]
    pub offers_seen: u64,
}

/// One uploaded part (S3 `CompleteMultipartUpload`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Part {
    pub n: u16,
    pub etag: String,
}

/// A part URL of a grant.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartUrl {
    pub n: u16,
    pub url: String,
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
    /// Protocol 2: credits (sent on every change of the worker's arbiter).
    Slots(Slots),
    /// Protocol 2: open a multipart upload for a held job's output; the
    /// answer is a [`DoMsg::UploadGrant`] with `req`.
    UploadInit { req: u64, job_id: String, attempt: u32, lease: u64, name: String, #[serde(default)] content_type: String, #[serde(default = "one_u16")] parts: u16 },
    /// More (or fresh) part URLs: parts `from .. from + count`.
    UploadMore { req: u64, job_id: String, upload_id: String, from: u16, count: u16 },
    /// Every part is uploaded: complete it. Answer: [`DoMsg::UploadCommitted`].
    UploadDone { req: u64, job_id: String, attempt: u32, lease: u64, upload_id: String, parts: Vec<Part>, bytes: u64, sha256: String },
    /// Give the upload up.
    UploadAbort { job_id: String, upload_id: String },
    /// The arbiter reserved the GPU for the offered session.
    SessionAck { session_id: String, lease: u64, #[serde(default)] endpoint: String },
    /// The offered session was refused (busy, draining).
    SessionNack { session_id: String, lease: u64, #[serde(default)] code: u16, #[serde(default)] message: String },
    /// A session ended on the worker.
    SessionEnd { session_id: String, lease: u64 },
}

fn one_u16() -> u16 {
    1
}

/// Dispatcher → worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum DoMsg {
    /// Answer to [`WorkerMsg::Hello`]: the jobs it reported that it must
    /// drop (the dispatcher gave them to another worker meanwhile).
    Welcome {
        worker_id: String,
        pool: String,
        #[serde(default)]
        cancel: Vec<String>,
        /// Protocol 2: re-announced sessions the dispatcher ended meanwhile.
        #[serde(default)]
        end_sessions: Vec<String>,
    },
    /// A job to take: ack or nack. `lease` is its fencing token: the worker
    /// writes the job's row only while no newer lease holds it, so a worker
    /// the dispatcher gave up on cannot overwrite its successor.
    /// `takeover`: a re-dispatch after a worker loss.
    Job { job_id: String, attempt: u32, envelope: Value, #[serde(default)] lease: u64, #[serde(default)] takeover: bool },
    /// Stop a job (client cancel).
    Cancel { job_id: String },
    /// Take no new jobs (running ones finish).
    Drain { on: bool },
    /// Answer to [`WorkerMsg::UploadInit`] / [`WorkerMsg::UploadMore`]:
    /// `error` set means no upload (fenced, unknown job, storage error).
    UploadGrant {
        req: u64,
        job_id: String,
        #[serde(default)]
        upload_id: String,
        #[serde(default)]
        key: String,
        #[serde(default)]
        bucket: String,
        #[serde(default)]
        part_urls: Vec<PartUrl>,
        #[serde(default)]
        expires_ms: i64,
        #[serde(default)]
        error: Option<String>,
    },
    /// Answer to [`WorkerMsg::UploadDone`].
    UploadCommitted {
        req: u64,
        job_id: String,
        key: String,
        #[serde(default)]
        bucket: String,
        bytes: u64,
        ok: bool,
        #[serde(default)]
        error: Option<String>,
    },
    /// Reserve the GPU for a streaming session (ack or nack).
    SessionOffer { session_id: String, lease: u64, #[serde(default)] model: Option<String>, #[serde(default)] kind: String, ttl_ms: i64 },
    /// The session's lease ended (released, expired, given to another).
    SessionRevoke { session_id: String, #[serde(default)] reason: String },
}

/// `POST {scope}/sessions`: admit a streaming session.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionReq {
    /// Idempotency: the same id answers the same live session.
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// `director` | `reactor` | `stream` | …
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub owner: Option<String>,
    /// Lease length without a renew (0: the dispatcher's default).
    #[serde(default)]
    pub ttl_ms: i64,
}

/// An admitted session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionGrant {
    pub session_id: String,
    pub lease: u64,
    pub worker_id: String,
    /// The GPU worker's public base URL: signalling goes there.
    pub endpoint: String,
    pub expires_ms: i64,
}

/// `GET {scope}/metrics`: the family's demand signal (fv-control, autoscaler).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FamilyMetrics {
    pub family: String,
    pub now_ms: i64,
    pub queued: u32,
    /// Age of the oldest queued job (0: none).
    pub oldest_queued_ms: i64,
    pub pushed: u32,
    pub running: u32,
    /// Connected workers.
    pub workers: u32,
    /// Sum of connected workers' capacities, and the dispatcher's estimate
    /// of their free job slots.
    pub slots_total: u32,
    pub slots_free: u32,
    pub sessions_live: u32,
    /// Connected workers' free session slots.
    pub session_capacity: u32,
    pub failed_1h: u32,
    pub queue_p50_ms: f64,
    pub ack_p50_ms: f64,
    /// Open output uploads.
    #[serde(default)]
    pub uploads_open: u32,
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
    /// Protocol 2: the dispatcher's estimate of its free job slots, its
    /// free session slots and its public endpoint.
    #[serde(default)]
    pub free: u32,
    #[serde(default)]
    pub session_free: u32,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub proto: u32,
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
    /// Protocol 2: live and offered sessions.
    #[serde(default)]
    pub sessions: Vec<SessionInfo>,
}

/// A session as `status` lists it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub state: String,
    pub worker: Option<String>,
    pub lease: u64,
    pub kind: String,
    pub expires_ms: i64,
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
