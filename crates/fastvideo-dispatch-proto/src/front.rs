//! The edge's routing decisions (docs/serve/edge-control-plane.md §2): pure
//! functions the wasm Worker (`crates/fastvideo-edge`) and the native host
//! (`fastvideo_serve::edge_host`) share, so both route the same way.
//!
//! - [`FrontInfo`]: what a worker that is also an API front announces in its
//!   hello (the names that route to it, its fal apps, the protocols it
//!   mounts, its URL).
//! - [`classify`]: `(method, path)` → which protocol and how to find the
//!   front: by fal app, by the body's `model`, by job id (sticky), by
//!   session, any front, or answered by the edge itself.
//! - [`Registry`]: every family's workers (from the family objects'
//!   status), with name → family resolution and the front choice
//!   ([`Registry::pick`]; rendezvous hashing for sticky ids).
//! - [`Verdict`]: the identity decision the edge forwards in
//!   [`EDGE_AUTH_HEADER`]; the front's `auth.mode = trust-edge` applies the
//!   API's policy to it and renders any refusal in the API's own shape.
//! - [`scan_model`]: the `model` of a JSON or multipart body, read from as
//!   few bytes as possible.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::WorkerInfo;

/// The identity verdict the edge forwards to a front (JSON, [`Verdict`]).
pub const EDGE_AUTH_HEADER: &str = "x-fv-edge-auth";
/// The request id the edge mints (or keeps from the client).
pub const REQUEST_ID_HEADER: &str = "x-request-id";

fn yes() -> bool {
    true
}

/// A worker's API front, as its hello announces it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontInfo {
    /// Its own base URL (the Runpod proxy host), where the edge forwards.
    pub url: String,
    /// The protocols it mounts (`fal`, `fal_director`, `minimax`, `ltx`,
    /// `openai_videos`, `fastwan`, `native`, `reactor`, `console`).
    #[serde(default)]
    pub protocols: Vec<String>,
    /// Every name a request may give for one of its models (model ids,
    /// served names, aliases, tier aliases) → the model id.
    #[serde(default)]
    pub names: BTreeMap<String, String>,
    /// The fal app ids it mounts (`minimax/h3-turbo`, `lightricks/ltx-2.5`).
    #[serde(default)]
    pub fal_apps: Vec<String>,
    /// The model each protocol uses when a request names none.
    #[serde(default)]
    pub defaults: BTreeMap<String, String>,
    /// The model its Reactor runtime streams (`None`: no Reactor here).
    #[serde(default)]
    pub reactor: Option<String>,
    /// Models it failed to load, and why.
    #[serde(default)]
    pub failed_models: BTreeMap<String, String>,
    /// Its models are loaded.
    #[serde(default = "yes")]
    pub ready: bool,
    /// The routing tag of the upload tokens it issues (`{tag}.{random}`).
    #[serde(default)]
    pub tag: String,
}

/// The routing tag of a worker id: 8 hex characters of its SHA-256.
pub fn worker_tag(worker_id: &str) -> String {
    hex(&Sha256::digest(worker_id.as_bytes())[..4])
}

/// The tag of an upload token issued by a front (`{tag}.{random}`).
pub fn token_tag(token: &str) -> Option<&str> {
    let (tag, rest) = token.split_once('.')?;
    (tag.len() == 8 && tag.bytes().all(|b| b.is_ascii_hexdigit()) && !rest.is_empty()).then_some(tag)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ------------------------------------------------------------------ identity

/// Why the edge refuses a request it still forwards (so the front renders
/// the refusal in the API's shape).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deny {
    /// `rate_limited` | `unavailable` | `unknown_model` | `queue_full`.
    pub kind: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u32>,
}

/// The edge's identity decision (the [`EDGE_AUTH_HEADER`] value).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub v: u32,
    /// The key's owner id (`key_<12 hex>`) when it is valid.
    #[serde(default)]
    pub key: Option<String>,
    /// An `Authorization` with a key was presented.
    #[serde(default)]
    pub presented: bool,
    /// The presented key is valid (not revoked).
    #[serde(default)]
    pub valid: bool,
    /// `key` (fal) or `bearer`.
    #[serde(default)]
    pub scheme: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deny: Option<Deny>,
}

impl Verdict {
    pub fn header(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
    pub fn parse(s: &str) -> Option<Self> {
        serde_json::from_str(s).ok()
    }
}

/// `Bearer <k>` / `Key <k>` (scheme case-insensitive) → `(scheme, key)`,
/// the scheme as `bearer` or `key` (as serve-kit's `parse_authorization`).
pub fn parse_authorization(v: &str) -> Option<(&'static str, &str)> {
    let v = v.trim();
    let (scheme, rest) = v.split_once(char::is_whitespace)?;
    let key = rest.trim();
    if key.is_empty() {
        return None;
    }
    let scheme = if scheme.eq_ignore_ascii_case("bearer") {
        "bearer"
    } else if scheme.eq_ignore_ascii_case("key") {
        "key"
    } else {
        return None;
    };
    Some((scheme, key))
}

/// SHA-256 hex of a key (the `api_keys.digest` / `FV_API_KEYS` form).
pub fn key_digest(key: &str) -> String {
    hex(&Sha256::digest(key.as_bytes()))
}

/// The owner id of a key digest (`key_<first 12 hex>`).
pub fn key_id(digest_hex: &str) -> String {
    format!("key_{}", &digest_hex[..12.min(digest_hex.len())])
}

/// Constant-time equality.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ------------------------------------------------------------------ classify

/// What the edge itself answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EdgeRoute {
    Root,
    Ping,
    Health,
    Healthz,
    /// `GET /fv/v1/status`.
    Status,
    /// `GET /fv/v1/capabilities` (merged across families).
    Capabilities,
    /// `GET /metrics` (admin token).
    Metrics,
    /// `GET /v1/models` (merged across families).
    Models,
    /// `GET /v1/models/{model}`.
    Model(String),
    /// `GET /fal/schema` (merged across families).
    FalSchema,
    /// `GET|POST /fv/v1/admin/keys`.
    Keys,
    /// `DELETE /fv/v1/admin/keys/{id}`.
    KeyRevoke(String),
    /// `POST /fv/v1/admin/keys/invalidate`: drop every cached key now.
    KeysInvalidate,
    /// `GET /fv/v1/edge/families` (and the old `/fv/v1/gateway/pools`).
    Families,
    /// `/families/…`, `/pools/…`, `/registry…`, `/up`, `/dl`: the
    /// dispatcher's own routes (internal token).
    Internal,
    /// Routes of the retired gateway that moved to fv-control.
    Moved(&'static str),
}

/// A director (fal realtime) signalling call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectorOp {
    /// `/wma/ice`: any director front (no state).
    Ice,
    /// `/wma/session`: admit, proxy, bind the answer's `session_id`.
    Session,
    /// `/wma/session/heartbeat`: by the body's `session_id`; renews.
    Heartbeat,
    /// `/start-session` (SSE): admit, proxy streamed, release at the end.
    Start,
    /// `/info`: any director front.
    Info,
}

/// A Reactor local-runtime call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReactorOp {
    /// `/start_session`: admit, lease to the caller.
    Start,
    /// `/session`, `/schema`, `/events`: the caller's lease.
    Follow,
    /// `/stop_session`: the caller's lease, then release.
    Stop,
    /// `/sessions/{sid}/…`.
    Sid(String),
}

/// A native stream (`/fv/v1/streams`) or WHIP ingest call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamOp {
    /// `POST`: admit (model in the body or query), proxy, bind the answer's id.
    Create,
    /// `GET` of the collection: every front's list, merged.
    List,
    /// By id (`GET`, `POST …/commands`, `PATCH` trickle ICE).
    Follow(String),
    /// `DELETE` by id: proxy, then release.
    Delete(String),
}

/// How the edge finds the front for a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Edge(EdgeRoute),
    /// The fal app at `path[skip..]`; `id`: a request id (sticky).
    FalApp { skip: usize, id: Option<String> },
    /// fal's proxy: the app is in `x-fal-target-url`.
    FalProxy,
    /// The `model` of the body (a submit).
    Body,
    /// A job id from the path, or the query parameter named (sticky).
    Job(String),
    JobQuery(&'static str),
    /// Any ready front that mounts the protocol.
    Any,
    /// An upload token (`PUT /uploads/{t}`, `GET /files/{t}/…`, the
    /// internal upload fetch): the front whose tag it carries, else any.
    Upload(String),
    Director(DirectorOp),
    Reactor(ReactorOp),
    /// Native streams (`ingest = false`) or WHIP ingest (`true`).
    Stream { ingest: bool, op: StreamOp },
    /// No such route.
    NotFound,
}

/// A classified request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Class {
    /// The protocol (as [`FrontInfo::protocols`] names it), `edge` for the
    /// edge's own routes.
    pub protocol: &'static str,
    pub target: Target,
}

fn c(protocol: &'static str, target: Target) -> Class {
    Class { protocol, target }
}

/// LTX endpoints (generation and 403 stubs).
const LTX_ENDPOINTS: &[&str] = &["text-to-video", "image-to-video", "audio-to-video", "retake", "extend", "video-to-video-hdr", "video-to-video-reframe"];

/// Classifies one request (see the module docs). `path` has no query.
pub fn classify(method: &str, path: &str) -> Class {
    let m = method.to_ascii_uppercase();
    let m = m.as_str();
    let segs: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    let s = |i: usize| segs.get(i).copied().unwrap_or("");
    let n = if path == "/" { 0 } else { segs.len() };
    use EdgeRoute as E;
    use Target as T;
    match (m, path) {
        (_, "/") => return c("edge", T::Edge(E::Root)),
        (_, "/ping") => return c("edge", T::Edge(E::Ping)),
        (_, "/health") => return c("edge", T::Edge(E::Health)),
        (_, "/healthz") => return c("edge", T::Edge(E::Healthz)),
        (_, "/metrics") => return c("edge", T::Edge(E::Metrics)),
        ("GET", "/fv/v1/status") => return c("edge", T::Edge(E::Status)),
        ("GET", "/fv/v1/capabilities") => return c("edge", T::Edge(E::Capabilities)),
        ("GET", "/v1/models") => return c("edge", T::Edge(E::Models)),
        ("GET", "/fal/schema") => return c("edge", T::Edge(E::FalSchema)),
        (_, "/fv/v1/admin/keys") => return c("edge", T::Edge(E::Keys)),
        ("POST", "/fv/v1/admin/keys/invalidate") => return c("edge", T::Edge(E::KeysInvalidate)),
        (_, "/fv/v1/edge/families" | "/fv/v1/gateway/pools") => return c("edge", T::Edge(E::Families)),
        (_, "/up" | "/dl") => return c("edge", T::Edge(E::Internal)),
        _ => {}
    }
    match s(0) {
        "families" | "pools" | "registry" => return c("edge", T::Edge(E::Internal)),
        "console" => return c("console", T::Any),
        _ => {}
    }
    if m == "GET" && n == 3 && s(0) == "v1" && s(1) == "models" {
        return c("edge", T::Edge(E::Model(s(2).to_owned())));
    }
    if m == "DELETE" && n == 5 && path.starts_with("/fv/v1/admin/keys/") {
        return c("edge", T::Edge(E::KeyRevoke(s(4).to_owned())));
    }
    if path.starts_with("/fv/v1/admin/releases") || path == "/fv/v1/admin/deployments" {
        return c("edge", T::Edge(E::Moved("releases and deployments are on fv-control (/api/releases)")));
    }
    if path == "/fv/v1/admin/token/sealed" {
        return c("edge", T::Edge(E::Moved("the admin token is held by fv-control")));
    }
    if path == "/fv/v1/admin/flags" || path.starts_with("/fv/v1/admin/flags/") {
        return c("native", T::Any);
    }
    if n == 5 && path.starts_with("/fv/v1/internal/uploads/") {
        return c("internal", T::Upload(s(4).to_owned()));
    }
    if s(0) == "fv" && s(1) == "v1" && s(2) == "internal" {
        return c("edge", T::NotFound);
    }
    // Uploads and files.
    if n == 2 && s(0) == "uploads" && m == "PUT" {
        return c("serve", T::Upload(s(1).to_owned()));
    }
    if n == 3 && s(0) == "files" && matches!(m, "GET" | "HEAD") {
        return c("serve", T::Upload(s(1).to_owned()));
    }
    // OpenAI-style videos.
    match (m, n, s(0), s(1)) {
        ("POST", 2, "v1", "videos") | ("POST", 3, "v1", "videos") if n == 2 || matches!(s(2), "generations" | "sync") => {
            return c("openai_videos", T::Body);
        }
        ("GET", 2, "v1", "videos") => return c("openai_videos", T::Any),
        ("GET" | "DELETE", 3, "v1", "videos") => return c("openai_videos", T::Job(s(2).to_owned())),
        ("GET", 4, "v1", "videos") if s(3) == "content" => return c("openai_videos", T::Job(s(2).to_owned())),
        (_, 2, "v1", "model_info") => return c("openai_videos", T::Any),
        _ => {}
    }
    // FastWan.
    match (m, n, s(0)) {
        ("POST", 1, "generate") => return c("fastwan", T::Body),
        ("GET", 2, "status") | ("GET" | "DELETE", 2, "video") => return c("fastwan", T::Job(s(1).to_owned())),
        _ => {}
    }
    // MiniMax.
    if s(0) == "v2" {
        match (m, path) {
            ("POST", "/v2/video_generation") => return c("minimax", T::Body),
            ("GET", "/v2/query/video_generation") => return c("minimax", T::JobQuery("task_id")),
            (_, "/v2/h3_context_ir" | "/v2/video_regeneration") => return c("minimax", T::Any),
            _ => {}
        }
        if n == 3 && s(1) == "video_generation" && m == "DELETE" {
            return c("minimax", T::Job(s(2).to_owned()));
        }
        if n == 4 && s(1) == "query" && s(2) == "video_generation" && m == "GET" {
            return c("minimax", T::Job(s(3).to_owned()));
        }
    }
    // LTX.
    if (s(0) == "v1" || s(0) == "v2") && LTX_ENDPOINTS.contains(&s(1)) {
        if m == "POST" && n == 2 {
            return c("ltx", T::Body);
        }
        if m == "GET" && n == 3 && s(0) == "v2" {
            return c("ltx", T::Job(s(2).to_owned()));
        }
    }
    if path == "/v1/upload" {
        return c("ltx", T::Any);
    }
    // Native jobs and streams.
    if s(0) == "fv" && s(1) == "v1" {
        match (m, n, s(2)) {
            ("POST", 3, "jobs") => return c("native", T::Body),
            ("GET", 3, "jobs") => return c("native", T::Any),
            ("GET" | "DELETE", 4, "jobs") => return c("native", T::Job(s(3).to_owned())),
            ("GET", 5, "jobs") if s(4) == "content" => return c("native", T::Job(s(3).to_owned())),
            _ => {}
        }
        if s(2) == "streams" {
            let ingest = s(3) == "ingest";
            let base = if ingest { 4 } else { 3 };
            let id = s(base);
            let op = match (m, n - base) {
                ("POST", 0) => StreamOp::Create,
                ("GET", 0) => StreamOp::List,
                ("DELETE", 1) => StreamOp::Delete(id.to_owned()),
                (_, 1) | (_, 2) => StreamOp::Follow(id.to_owned()),
                _ => return c("native", T::NotFound),
            };
            return c("native", T::Stream { ingest, op });
        }
        return c("native", T::NotFound);
    }
    // fal shared routes, director, Reactor.
    match (m, path) {
        (_, "/fal/proxy") => return c("fal", T::FalProxy),
        ("POST", "/storage/upload/initiate") => return c("fal", T::Any),
        ("GET", "/.well-known/jwks.json") => return c("fal", T::Any),
        ("POST", "/wma/ice") => return c("fal_director", T::Director(DirectorOp::Ice)),
        ("POST", "/wma/session") => return c("fal_director", T::Director(DirectorOp::Session)),
        ("POST", "/wma/session/heartbeat") => return c("fal_director", T::Director(DirectorOp::Heartbeat)),
        ("POST", "/start-session") => return c("fal_director", T::Director(DirectorOp::Start)),
        (_, "/info") => return c("fal_director", T::Director(DirectorOp::Info)),
        ("POST", "/start_session") => return c("reactor", T::Reactor(ReactorOp::Start)),
        ("POST", "/stop_session") => return c("reactor", T::Reactor(ReactorOp::Stop)),
        ("GET", "/session" | "/schema" | "/events") => return c("reactor", T::Reactor(ReactorOp::Follow)),
        _ => {}
    }
    if s(0) == "sessions" && n >= 3 {
        return c("reactor", T::Reactor(ReactorOp::Sid(s(1).to_owned())));
    }
    if path.starts_with("/fal/schema/") {
        return c("fal", T::FalApp { skip: "/fal/schema".len(), id: None });
    }
    // Everything else is a fal app path: `/{owner}/{app}[/sub][/requests/{id}[/…]]`,
    // or the same under `/run`.
    if n >= 2 {
        let skip = if s(0) == "run" { 4 } else { 0 };
        let id = segs.iter().position(|x| *x == "requests").and_then(|i| segs.get(i + 1)).map(|x| (*x).to_owned());
        return c("fal", T::FalApp { skip, id });
    }
    c("edge", T::NotFound)
}

/// The value of query parameter `name` (form-decoded minimally: `+` and
/// `%XX`).
pub fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (k == name).then(|| percent_decode(v))
    })
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex2 = || b.get(i + 1..i + 3).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u8::from_str_radix(h, 16).ok());
        match b[i] {
            b'+' => out.push(b' '),
            b'%' => match hex2() {
                Some(x) => {
                    out.push(x);
                    i += 2;
                }
                None => out.push(b'%'),
            },
            x => out.push(x),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ------------------------------------------------------------------ registry

/// Every family's workers, as the family objects report them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Registry {
    /// Family id → its workers (connected or within their grace period).
    pub families: BTreeMap<String, Vec<WorkerInfo>>,
    /// Unfinished jobs per owner, over every family.
    #[serde(default)]
    pub owners: BTreeMap<String, u32>,
    /// Bumped when a key is revoked: edge isolates drop their key caches.
    #[serde(default)]
    pub key_epoch: u64,
    #[serde(default)]
    pub at_ms: i64,
    /// Per family: queued and running jobs, live sessions.
    #[serde(default)]
    pub counts: BTreeMap<String, Counts>,
}

/// A family's load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    pub queued: u32,
    pub running: u32,
    pub sessions: u32,
}

/// A front the edge may forward to.
#[derive(Clone, Copy, Debug)]
pub struct Front<'a> {
    pub family: &'a str,
    pub worker: &'a WorkerInfo,
    pub info: &'a FrontInfo,
}

impl Registry {
    /// From each family object's status (`family`, its `status`).
    pub fn from_statuses(statuses: Vec<(String, crate::PoolStatus)>, key_epoch: u64, now_ms: i64) -> Self {
        let mut r = Registry { key_epoch, at_ms: now_ms, ..Registry::default() };
        for (f, st) in statuses {
            for (o, n) in &st.owners {
                *r.owners.entry(o.clone()).or_default() += n;
            }
            let sessions = st.sessions.iter().filter(|x| x.state == "live").count() as u32;
            r.counts.insert(f.clone(), Counts { queued: st.queued, running: st.running + st.pushed, sessions });
            r.families.insert(f, st.workers);
        }
        r
    }

    /// Ready, connected, non-draining fronts.
    pub fn fronts(&self) -> impl Iterator<Item = Front<'_>> {
        self.families.iter().flat_map(|(f, ws)| {
            ws.iter().filter_map(move |w| {
                let info = w.front.as_ref()?;
                (w.connected && !w.draining && w.ready && info.ready && !info.url.is_empty()).then_some(Front { family: f, worker: w, info })
            })
        })
    }

    /// The family serving the model named `name` (any name of
    /// [`FrontInfo::names`]), with every front that knows the name.
    pub fn family_of_name(&self, name: &str) -> Option<&str> {
        self.fronts().find(|x| x.info.names.contains_key(name)).map(|x| x.family)
    }

    /// The fal app `path` starts with (the longest match) and its family.
    pub fn fal_app(&self, path: &str) -> Option<(String, &str)> {
        let mut best: Option<(String, &str)> = None;
        for x in self.fronts() {
            for a in &x.info.fal_apps {
                let a = a.trim_matches('/');
                let p = format!("/{a}");
                let hit = path == p || path.starts_with(&format!("{p}/"));
                if hit && best.as_ref().is_none_or(|(b, _)| a.len() > b.len()) {
                    best = Some((a.to_owned(), x.family));
                }
            }
        }
        best
    }

    /// The family of the model `protocol` uses when a request names none.
    pub fn default_family(&self, protocol: &str) -> Option<&str> {
        let name = self.fronts().find_map(|x| x.info.defaults.get(protocol))?;
        self.family_of_name(name)
    }

    /// The family of the Reactor runtime (`preferred`: the cluster's
    /// Reactor model when several fronts have one).
    pub fn reactor_family(&self, preferred: Option<&str>) -> Option<&str> {
        if let Some(p) = preferred {
            if let Some(x) = self.fronts().find(|x| x.info.reactor.as_deref() == Some(p)) {
                return Some(x.family);
            }
        }
        self.fronts().find(|x| x.info.reactor.is_some()).map(|x| x.family)
    }

    /// Picks a front: of `family` (any family when `None`), mounting
    /// `protocol` (unless empty), knowing `name` (unless `None`). With
    /// `sticky`, the rendezvous choice for that id; otherwise the least
    /// held (ties: the fewest held of the GPU, then the id).
    pub fn pick(&self, family: Option<&str>, protocol: &str, name: Option<&str>, sticky: Option<&str>) -> Option<Front<'_>> {
        let ok = |x: &Front<'_>| {
            family.is_none_or(|f| x.family == f)
                && (protocol.is_empty() || x.info.protocols.iter().any(|p| p == protocol))
                && name.is_none_or(|n| x.info.names.contains_key(n))
        };
        let mut c: Vec<Front<'_>> = self.fronts().filter(ok).collect();
        // One worker can be a front in several families: once is enough.
        c.sort_by(|a, b| a.worker.worker_id.cmp(&b.worker.worker_id));
        c.dedup_by(|a, b| a.worker.worker_id == b.worker.worker_id);
        match sticky {
            Some(id) => c.into_iter().max_by_key(|x| rendezvous(id, &x.worker.worker_id)),
            None => c.into_iter().min_by_key(|x| (x.worker.held, u32::MAX - x.worker.free, x.worker.worker_id.clone())),
        }
    }

    /// The front whose upload tag is `tag`.
    pub fn by_tag(&self, tag: &str) -> Option<Front<'_>> {
        self.fronts().find(|x| x.info.tag == tag)
    }

    /// Every distinct front (one per worker), for fan-outs.
    pub fn distinct(&self, protocol: &str) -> Vec<Front<'_>> {
        let mut c: Vec<Front<'_>> = self.fronts().filter(|x| protocol.is_empty() || x.info.protocols.iter().any(|p| p == protocol)).collect();
        c.sort_by(|a, b| a.worker.worker_id.cmp(&b.worker.worker_id));
        c.dedup_by(|a, b| a.worker.worker_id == b.worker.worker_id);
        c
    }

    /// One front per family (for merges of per-family views).
    pub fn one_per_family(&self, protocol: &str) -> Vec<Front<'_>> {
        let mut out: Vec<Front<'_>> = Vec::new();
        for x in self.fronts() {
            if (protocol.is_empty() || x.info.protocols.iter().any(|p| p == protocol)) && !out.iter().any(|o| o.family == x.family) {
                out.push(x);
            }
        }
        out
    }

    /// Whether any front is ready.
    pub fn any_ready(&self) -> bool {
        self.fronts().next().is_some()
    }
}

/// Rendezvous (highest random weight) hash of `(id, worker)`: FNV-1a 64.
pub fn rendezvous(id: &str, worker: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in id.bytes().chain([0u8]).chain(worker.bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Finalizer (FNV alone is weak on short, similar keys).
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}

// ------------------------------------------------------------------ views

/// A worker's public state word (`crate::status` of fv-serve).
fn worker_state(w: &WorkerInfo) -> &'static str {
    let failed = w.front.as_ref().is_some_and(|f| !f.failed_models.is_empty() && f.names.is_empty());
    if !w.connected {
        "down"
    } else if w.draining {
        "draining"
    } else if failed {
        "failed"
    } else if !w.ready {
        "loading"
    } else if w.held > 0 {
        "busy"
    } else {
        "ready"
    }
}

fn state_rank(s: &str) -> u8 {
    match s {
        "ready" => 0,
        "busy" => 1,
        "loading" => 2,
        "scaled_to_zero" => 3,
        "draining" => 4,
        "unhealthy" => 5,
        "down" => 6,
        _ => 7,
    }
}

/// `GET /fv/v1/status` at the edge: fv-serve's `fv.status` shape, one pool
/// per family (workers labelled `w1`, `w2`, …; no URLs or ids).
pub fn status_body(reg: &Registry, version: serde_json::Value, now_ms: i64) -> serde_json::Value {
    use serde_json::json;
    let mut pools = Vec::new();
    let mut models: BTreeMap<String, (u8, Vec<String>, Option<String>)> = BTreeMap::new();
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    let mut overall = 7u8;
    let mut mixed_any = false;
    for (f, ws) in &reg.families {
        let mut workers = Vec::new();
        let mut fam_models: Vec<String> = Vec::new();
        let mut failed: BTreeMap<String, String> = BTreeMap::new();
        let mut versions: BTreeMap<String, u32> = BTreeMap::new();
        let mut last_seen: Option<i64> = None;
        for (i, w) in ws.iter().enumerate() {
            let st = worker_state(w);
            workers.push(json!({
                "label": format!("w{}", i + 1),
                "state": st,
                "last_seen_s": ((now_ms - w.last_seen_ms).max(0) as f64 / 100.0).round() / 10.0,
                "running": w.held,
                "queued": 0,
                "sessions": 0,
            }));
            last_seen = Some(last_seen.map_or(w.last_seen_ms, |l| l.max(w.last_seen_ms)));
            if w.connected {
                *versions.entry(if w.sha.is_empty() { "unknown".into() } else { w.sha.clone() }).or_default() += 1;
            }
            if let Some(fi) = &w.front {
                for (n, m) in &fi.names {
                    names.entry(n.clone()).or_insert_with(|| m.clone());
                    if !fam_models.contains(m) {
                        fam_models.push(m.clone());
                    }
                }
                failed.extend(fi.failed_models.clone());
            }
        }
        let state = workers.iter().map(|w| w["state"].as_str().unwrap_or("down")).min_by_key(|s| state_rank(s)).unwrap_or("scaled_to_zero").to_owned();
        overall = overall.min(state_rank(&state));
        let mixed = versions.len() > 1;
        mixed_any |= mixed;
        let available = reg.fronts().any(|x| x.family == f);
        let c = reg.counts.get(f).copied().unwrap_or_default();
        for m in fam_models.iter().chain(failed.keys()) {
            let (st, why) = match failed.get(m) {
                Some(w) => (7u8, Some(w.clone())),
                None => (state_rank(&state), None),
            };
            let e = models.entry(m.clone()).or_insert((st, Vec::new(), None));
            e.0 = e.0.min(st);
            if e.2.is_none() {
                e.2 = why;
            }
            if !e.1.contains(f) {
                e.1.push(f.clone());
            }
        }
        let mut models_list: Vec<String> = fam_models.clone();
        models_list.sort();
        pools.push(json!({
            "id": f,
            "kind": "family",
            "state": state,
            "available": available,
            "models": models_list,
            "queued": c.queued,
            "running": c.running,
            "last_seen_s": last_seen.map(|l| ((now_ms - l).max(0) as f64 / 100.0).round() / 10.0),
            "workers": workers,
            "versions": versions.iter().map(|(sha, n)| json!({"sha": sha, "channel": null, "workers": n})).collect::<Vec<_>>(),
            "mixed_versions": mixed,
            "failed_models": failed,
        }));
    }
    let word = |r: u8| match r {
        0 => "ready",
        1 => "busy",
        2 => "loading",
        3 => "scaled_to_zero",
        4 => "draining",
        5 => "unhealthy",
        6 => "down",
        _ => "failed",
    };
    let models: BTreeMap<String, serde_json::Value> = models
        .into_iter()
        .map(|(m, (r, p, why))| {
            let mut v = json!({"state": word(r), "pools": p});
            if let (7, Some(w)) = (r, why) {
                v["reason"] = serde_json::Value::String(w);
            }
            (m, v)
        })
        .collect();
    json!({
        "object": "fv.status",
        "gateway": false,
        "edge": true,
        "state": if reg.families.is_empty() { "down" } else { word(overall) },
        "version": version,
        "mixed_versions": mixed_any,
        "pools": pools,
        "models": models,
        "names": names,
    })
}

/// Merges per-family bodies' `key` arrays, keeping the first item of each
/// `id` (a JSON pointer into the item).
pub fn merge_list(bodies: &[serde_json::Value], key: &str, id: &str) -> Vec<serde_json::Value> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for b in bodies {
        for item in b.get(key).and_then(|v| v.as_array()).into_iter().flatten() {
            let k = item.pointer(id).map(|v| v.to_string()).unwrap_or_default();
            if seen.insert(k) {
                out.push(item.clone());
            }
        }
    }
    out
}

/// `/fv/v1/capabilities` at the edge: one front's per family, merged
/// (`models`, `tiers`, `aliases`), with the families as `pools`.
pub fn merge_capabilities(bodies: &[serde_json::Value], reg: &Registry, now_ms: i64) -> serde_json::Value {
    let mut aliases = serde_json::Map::new();
    for b in bodies {
        if let Some(m) = b.get("aliases").and_then(|v| v.as_object()) {
            for (k, v) in m {
                aliases.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    let status = status_body(reg, serde_json::Value::Null, now_ms);
    serde_json::json!({
        "object": "fv.capabilities",
        "models": merge_list(bodies, "models", "/caps/id"),
        "tiers": merge_list(bodies, "tiers", "/alias"),
        "aliases": aliases,
        "readiness": if reg.any_ready() { "ready" } else { "unavailable" },
        "gateway": false,
        "edge": true,
        "auth": bodies.iter().find_map(|b| b.get("auth").cloned()).unwrap_or(serde_json::Value::Null),
        "pools": status["pools"].clone(),
    })
}

/// `/fal/schema` at the edge: apps merged by id, their endpoints by
/// `endpoint_id`.
pub fn merge_fal_schema(bodies: &[serde_json::Value]) -> serde_json::Value {
    let mut apps: Vec<serde_json::Value> = Vec::new();
    for b in bodies {
        for a in b.get("apps").and_then(|v| v.as_array()).into_iter().flatten() {
            match apps.iter_mut().find(|x| x.get("id") == a.get("id")) {
                Some(x) => {
                    let mut eps = x.get("endpoints").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                    for e in a.get("endpoints").and_then(|v| v.as_array()).into_iter().flatten() {
                        if !eps.iter().any(|o| o.get("endpoint_id") == e.get("endpoint_id")) {
                            eps.push(e.clone());
                        }
                    }
                    x["endpoints"] = serde_json::Value::Array(eps);
                }
                None => apps.push(a.clone()),
            }
        }
    }
    serde_json::json!({"apps": apps})
}

/// The families' demand gauges in the Prometheus text format (the edge's
/// `/metrics`; the gateway's `fv_pool_*` per family).
pub fn prometheus(ms: &[crate::FamilyMetrics]) -> String {
    let mut out = String::new();
    let gauges: [(&str, &str, fn(&crate::FamilyMetrics) -> f64); 9] = [
        ("fv_family_queued", "jobs queued in the family object", |m| f64::from(m.queued)),
        ("fv_family_oldest_queued_seconds", "age of the oldest queued job", |m| m.oldest_queued_ms as f64 / 1e3),
        ("fv_family_running", "jobs pushed or running", |m| f64::from(m.pushed + m.running)),
        ("fv_family_workers", "connected workers", |m| f64::from(m.workers)),
        ("fv_family_slots_total", "job slots of connected workers", |m| f64::from(m.slots_total)),
        ("fv_family_slots_free", "free job slots (estimate)", |m| f64::from(m.slots_free)),
        ("fv_family_sessions_live", "live streaming sessions", |m| f64::from(m.sessions_live)),
        ("fv_family_failed_1h", "jobs the dispatcher failed in the last hour", |m| f64::from(m.failed_1h)),
        ("fv_family_queue_p50_seconds", "enqueue to push, p50", |m| m.queue_p50_ms / 1e3),
    ];
    for (name, help, f) in gauges {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} gauge\n"));
        for m in ms {
            out.push_str(&format!("{name}{{family=\"{}\"}} {}\n", m.family, f(m)));
        }
    }
    out
}

// ------------------------------------------------------------------ plans

/// The edge's quotas (docs/serve/edge-control-plane.md §2.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quotas {
    /// Requests per minute per key (0: none).
    pub key_rpm: u32,
    /// Unfinished jobs per key over every family (0: none).
    pub key_in_flight: u32,
    /// Requests per minute per client address with an invalid key.
    pub invalid_key_rpm: u32,
}

impl Default for Quotas {
    fn default() -> Self {
        Self { key_rpm: 300, key_in_flight: 30, invalid_key_rpm: 60 }
    }
}

/// Where a stateless request goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Forward {
    pub url: String,
    pub family: String,
    pub worker_id: String,
    /// Forwarded with the verdict: the front renders it.
    pub deny: Option<Deny>,
}

/// The edge's own answer (no front can render it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    pub retry_after: Option<u32>,
}

impl Reply {
    pub fn body(&self) -> serde_json::Value {
        serde_json::json!({"error": {"kind": self.kind, "message": self.message}})
    }
}

fn unavailable(msg: impl Into<String>) -> Reply {
    Reply { status: 503, kind: "loading", message: msg.into(), retry_after: Some(10) }
}

/// Whether a request starts a job (a submit: the in-flight quota applies).
pub fn is_submit(method: &str, class: &Class) -> bool {
    method.eq_ignore_ascii_case("POST") && matches!(class.target, Target::Body | Target::FalApp { id: None, .. })
}

/// The front of a stateless request (every [`Target`] but the edge's own
/// routes and sessions). `model`: the body's model for [`Target::Body`]
/// (`None`: absent); `fal_target`: the path of `x-fal-target-url` for
/// [`Target::FalProxy`]; `query`: the raw query.
pub fn plan_forward(class: &Class, reg: &Registry, path: &str, query: &str, model: Option<&str>, fal_target: Option<&str>) -> Result<Forward, Reply> {
    let fwd = |x: Front<'_>, deny: Option<Deny>| Forward { url: x.info.url.clone(), family: x.family.to_owned(), worker_id: x.worker.worker_id.clone(), deny };
    let p = class.protocol;
    // Any front of the protocol, rendering `deny` (a front of another family
    // still speaks the API).
    let any_with = |deny: Deny| reg.pick(None, p, None, None).map(|x| fwd(x, Some(deny)));
    match &class.target {
        Target::FalApp { skip, id } => {
            let rest = path.get(*skip..).unwrap_or("");
            match reg.fal_app(rest) {
                Some((_, family)) => {
                    let family = family.to_owned();
                    reg.pick(Some(&family), "fal", None, id.as_deref()).map(|x| fwd(x, None)).ok_or_else(|| unavailable(format!("no worker serves this fal app right now (family `{family}`)")))
                }
                // No front mounts the app: any fal front answers its 404.
                None => reg.pick(None, "fal", None, None).map(|x| fwd(x, None)).ok_or_else(|| unavailable("no fal front is up")),
            }
        }
        Target::FalProxy => {
            let target = fal_target.unwrap_or("");
            let fam = reg.fal_app(target).map(|(_, f)| f.to_owned());
            reg.pick(fam.as_deref(), "fal", None, None).map(|x| fwd(x, None)).ok_or_else(|| unavailable("no fal front is up"))
        }
        Target::Body => {
            let family = match model {
                Some(m) => reg.family_of_name(m).map(str::to_owned),
                None => reg.default_family(p).map(str::to_owned),
            };
            match family {
                Some(f) => match reg.pick(Some(&f), p, None, None) {
                    Some(x) => Ok(fwd(x, None)),
                    None => any_with(Deny { kind: "unavailable".into(), message: format!("no worker serves this model right now (family `{f}`)"), retry_after: Some(10) })
                        .ok_or_else(|| unavailable(format!("no worker serves this model right now (family `{f}`)"))),
                },
                // A name no front announces (an API-specific model name):
                // a front of the API resolves it itself.
                None => reg.pick(None, p, None, None).map(|x| fwd(x, None)).ok_or_else(|| unavailable("no worker serves this API right now")),
            }
        }
        Target::Job(id) => reg.pick(None, p, None, Some(id)).map(|x| fwd(x, None)).ok_or_else(|| unavailable("no worker serves this API right now")),
        Target::JobQuery(name) => {
            let id = query_param(query, name);
            reg.pick(None, p, None, id.as_deref()).map(|x| fwd(x, None)).ok_or_else(|| unavailable("no worker serves this API right now"))
        }
        Target::Any => reg.pick(None, p, None, None).map(|x| fwd(x, None)).ok_or_else(|| unavailable("no worker serves this API right now")),
        Target::Upload(token) => {
            let tagged = token_tag(token).and_then(|t| reg.by_tag(t));
            match (tagged, token_tag(token)) {
                (Some(x), _) => Ok(fwd(x, None)),
                // Tagged by a front that is gone: its uploads went with it.
                (None, Some(_)) => Err(Reply { status: 404, kind: "not_found", message: "the upload is gone (its worker left)".into(), retry_after: None }),
                (None, None) => reg.pick(None, "", None, Some(token)).map(|x| fwd(x, None)).ok_or_else(|| unavailable("no worker is up")),
            }
        }
        _ => Err(Reply { status: 404, kind: "not_found", message: "no such route".into(), retry_after: None }),
    }
}

// ------------------------------------------------------------------ sessions

/// A session the edge admitted, under the id clients use for it (the
/// worker's director session id, the Reactor owner, the stream id).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionBinding {
    pub family: String,
    /// The family object's session id (renew, release).
    pub session_id: String,
    pub lease: u64,
    /// The worker's base URL.
    pub endpoint: String,
    pub kind: String,
    #[serde(default)]
    pub owner: Option<String>,
    pub expires_ms: i64,
}

/// The binding key of a Reactor caller (its key, else its address).
pub fn reactor_owner(key: Option<&str>, addr: Option<&str>) -> String {
    match key {
        Some(k) => format!("reactor:{k}"),
        None => format!("reactor:ip:{}", addr.unwrap_or("unknown")),
    }
}

// ------------------------------------------------------------------ bodies

/// What [`scan_model`] found so far.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scan {
    /// The value of the top-level `model`.
    Found(String),
    /// The body is complete and has no `model`.
    Absent,
    /// Read more of the body.
    More,
}

/// The `model` of a body prefix: JSON (`content_type` contains `json`, or
/// anything else that starts with `{`) or `multipart/form-data` (a part
/// named `model`). `complete`: `body` is the whole body.
pub fn scan_model(content_type: &str, body: &[u8], complete: bool) -> Scan {
    if let Some(b) = content_type.split(';').find_map(|p| p.trim().strip_prefix("boundary=")) {
        return scan_multipart(b.trim_matches('"'), body, complete);
    }
    scan_json(body, complete)
}

fn scan_multipart(boundary: &str, body: &[u8], complete: bool) -> Scan {
    let needle = b"name=\"model\"";
    let Some(i) = find(body, needle) else {
        return if complete { Scan::Absent } else { Scan::More };
    };
    let rest = &body[i..];
    let Some(h) = find(rest, b"\r\n\r\n") else {
        return if complete { Scan::Absent } else { Scan::More };
    };
    let val = &rest[h + 4..];
    let end_marker = format!("\r\n--{boundary}");
    match find(val, end_marker.as_bytes()) {
        Some(e) => Scan::Found(String::from_utf8_lossy(&val[..e]).trim().to_owned()),
        None if complete => Scan::Absent,
        None => Scan::More,
    }
}

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    h.windows(n.len()).position(|w| w == n)
}

/// A JSON tokenizer just deep enough to find the top-level `"model":
/// "<string>"` on a prefix.
fn scan_json(body: &[u8], complete: bool) -> Scan {
    let more = || if complete { Scan::Absent } else { Scan::More };
    let mut i = 0;
    let n = body.len();
    let ws = |i: &mut usize| {
        while *i < n && body[*i].is_ascii_whitespace() {
            *i += 1;
        }
    };
    ws(&mut i);
    if i >= n {
        return more();
    }
    if body[i] != b'{' {
        return Scan::Absent;
    }
    i += 1;
    // Reads a string starting at body[*i] == '"'; returns its raw bytes
    // (escapes kept) or None when the prefix ends inside it.
    fn string(body: &[u8], i: &mut usize) -> Option<(usize, usize)> {
        let start = *i + 1;
        let mut j = start;
        while j < body.len() {
            match body[j] {
                b'\\' => j += 2,
                b'"' => {
                    *i = j + 1;
                    return Some((start, j));
                }
                _ => j += 1,
            }
        }
        None
    }
    // Skips any value at depth > 0 (or a scalar); None: the prefix ended.
    fn skip(body: &[u8], i: &mut usize) -> Option<()> {
        let n = body.len();
        if *i >= n {
            return None;
        }
        match body[*i] {
            b'"' => string(body, i).map(|_| ()),
            b'{' | b'[' => {
                let mut depth = 0i32;
                while *i < n {
                    match body[*i] {
                        b'"' => {
                            string(body, i)?;
                            continue;
                        }
                        b'{' | b'[' => depth += 1,
                        b'}' | b']' => {
                            depth -= 1;
                            if depth == 0 {
                                *i += 1;
                                return Some(());
                            }
                        }
                        _ => {}
                    }
                    *i += 1;
                }
                None
            }
            _ => {
                while *i < n && !matches!(body[*i], b',' | b'}' | b']') && !body[*i].is_ascii_whitespace() {
                    *i += 1;
                }
                (*i < n).then_some(())
            }
        }
    }
    loop {
        ws(&mut i);
        if i >= n {
            return more();
        }
        match body[i] {
            b'}' => return Scan::Absent,
            b',' => {
                i += 1;
                continue;
            }
            b'"' => {}
            _ => return Scan::Absent,
        }
        let Some((ks, ke)) = string(body, &mut i) else { return more() };
        let is_model = &body[ks..ke] == b"model";
        ws(&mut i);
        if i >= n {
            return more();
        }
        if body[i] != b':' {
            return Scan::Absent;
        }
        i += 1;
        ws(&mut i);
        if i >= n {
            return more();
        }
        if is_model {
            if body[i] != b'"' {
                return Scan::Absent;
            }
            let Some((vs, ve)) = string(body, &mut i) else { return more() };
            let raw = &body[vs..ve];
            // Unescape through serde (rare: names have no escapes).
            let quoted = [b"\"".as_slice(), raw, b"\""].concat();
            return match serde_json::from_slice::<String>(&quoted) {
                Ok(s) => Scan::Found(s),
                Err(_) => Scan::Absent,
            };
        }
        if skip(body, &mut i).is_none() {
            return more();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn front(url: &str, names: &[&str], apps: &[&str], protocols: &[&str]) -> FrontInfo {
        FrontInfo {
            url: url.into(),
            protocols: protocols.iter().map(|s| s.to_string()).collect(),
            names: names.iter().map(|s| (s.to_string(), names[0].to_string())).collect(),
            fal_apps: apps.iter().map(|s| s.to_string()).collect(),
            ready: true,
            ..FrontInfo::default()
        }
    }

    fn w(id: &str, held: u32, f: FrontInfo) -> WorkerInfo {
        WorkerInfo { worker_id: id.into(), connected: true, held, ready: true, front: Some(f), ..WorkerInfo::default() }
    }

    fn reg() -> Registry {
        let mut r = Registry::default();
        let h3 = front("https://a", &["fasth3", "h3-turbo", "MiniMax-H3"], &["minimax/h3-turbo", "minimax/h3"], &["fal", "minimax", "native"]);
        let ltx = front("https://b", &["ltx25-distill-sol", "ltx-turbo"], &["lightricks/ltx-2.5"], &["fal", "ltx", "native"]);
        r.families.insert("h3".into(), vec![w("wa", 1, h3.clone()), w("wa2", 0, h3)]);
        r.families.insert("ltx".into(), vec![w("wb", 0, ltx)]);
        r
    }

    #[test]
    fn classifies_every_api() {
        assert_eq!(classify("POST", "/v1/videos").target, Target::Body);
        assert_eq!(classify("POST", "/v1/videos/sync").target, Target::Body);
        assert_eq!(classify("GET", "/v1/videos/video_1").target, Target::Job("video_1".into()));
        assert_eq!(classify("GET", "/v1/videos/video_1/content").target, Target::Job("video_1".into()));
        assert_eq!(classify("GET", "/v1/videos").target, Target::Any);
        assert_eq!(classify("POST", "/v2/video_generation").protocol, "minimax");
        assert_eq!(classify("GET", "/v2/query/video_generation").target, Target::JobQuery("task_id"));
        assert_eq!(classify("POST", "/v2/text-to-video").target, Target::Body);
        assert_eq!(classify("GET", "/v2/text-to-video/abc").target, Target::Job("abc".into()));
        assert_eq!(classify("POST", "/v1/image-to-video").protocol, "ltx");
        assert_eq!(classify("POST", "/fv/v1/jobs").target, Target::Body);
        assert_eq!(classify("GET", "/fv/v1/jobs/j1/content").target, Target::Job("j1".into()));
        assert_eq!(classify("POST", "/fv/v1/streams").target, Target::Stream { ingest: false, op: StreamOp::Create });
        assert_eq!(classify("DELETE", "/fv/v1/streams/s1").target, Target::Stream { ingest: false, op: StreamOp::Delete("s1".into()) });
        assert_eq!(classify("POST", "/fv/v1/streams/ingest").target, Target::Stream { ingest: true, op: StreamOp::Create });
        assert_eq!(classify("POST", "/fv/v1/streams/ingest/i1/commands").target, Target::Stream { ingest: true, op: StreamOp::Follow("i1".into()) });
        assert_eq!(classify("POST", "/minimax/h3-turbo/text-to-video").target, Target::FalApp { skip: 0, id: None });
        assert_eq!(classify("GET", "/minimax/h3-turbo/requests/r1/status").target, Target::FalApp { skip: 0, id: Some("r1".into()) });
        assert_eq!(classify("POST", "/run/minimax/h3-turbo/text-to-video").target, Target::FalApp { skip: 4, id: None });
        assert_eq!(classify("PUT", "/uploads/abcd1234.xyz").target, Target::Upload("abcd1234.xyz".into()));
        assert_eq!(classify("GET", "/files/t/x.png").target, Target::Upload("t".into()));
        assert_eq!(classify("POST", "/wma/session").target, Target::Director(DirectorOp::Session));
        assert_eq!(classify("POST", "/start_session").target, Target::Reactor(ReactorOp::Start));
        assert_eq!(classify("POST", "/sessions/s9/uploads").target, Target::Reactor(ReactorOp::Sid("s9".into())));
        assert_eq!(classify("GET", "/fv/v1/status").target, Target::Edge(EdgeRoute::Status));
        assert_eq!(classify("DELETE", "/fv/v1/admin/keys/key_1").target, Target::Edge(EdgeRoute::KeyRevoke("key_1".into())));
        assert_eq!(classify("GET", "/fv/v1/internal/status").target, Target::NotFound);
        assert_eq!(classify("GET", "/console/assets/app.js").protocol, "console");
        assert_eq!(classify("GET", "/families/h3/status").target, Target::Edge(EdgeRoute::Internal));
    }

    #[test]
    fn registry_routes_names_apps_and_sticky_ids() {
        let r = reg();
        assert_eq!(r.family_of_name("MiniMax-H3"), Some("h3"));
        assert_eq!(r.family_of_name("ltx-turbo"), Some("ltx"));
        assert_eq!(r.family_of_name("nope"), None);
        assert_eq!(r.fal_app("/minimax/h3-turbo/text-to-video").map(|x| x.0), Some("minimax/h3-turbo".to_owned()));
        assert_eq!(r.fal_app("/minimax/h3/requests/x").map(|x| x.1), Some("h3"));
        assert_eq!(r.fal_app("/minimax/h3x/requests/x"), None);
        // Least held first.
        assert_eq!(r.pick(Some("h3"), "fal", None, None).unwrap().worker.worker_id, "wa2");
        // Sticky: the same id always lands on the same front.
        let a = r.pick(Some("h3"), "fal", None, Some("req-1")).unwrap().worker.worker_id.clone();
        for _ in 0..5 {
            assert_eq!(r.pick(Some("h3"), "fal", None, Some("req-1")).unwrap().worker.worker_id, a);
        }
        // Sticky ids spread over both fronts.
        let n = (0..200).filter(|i| r.pick(Some("h3"), "fal", None, Some(&format!("id-{i}"))).unwrap().worker.worker_id == "wa").count();
        assert!((60..140).contains(&n), "{n}");
        // A protocol only one family mounts.
        assert_eq!(r.pick(None, "ltx", None, None).unwrap().family, "ltx");
        assert_eq!(r.one_per_family("fal").len(), 2);
    }

    #[test]
    fn plans_forwards() {
        let mut r = reg();
        r.families.get_mut("ltx").unwrap()[0].front.as_mut().unwrap().tag = worker_tag("wb");
        let p = |m: &str, path: &str, model: Option<&str>| plan_forward(&classify(m, path), &r, path, "", model, None);
        assert_eq!(p("POST", "/v2/video_generation", Some("MiniMax-H3")).unwrap().family, "h3");
        assert_eq!(p("POST", "/v2/text-to-video", Some("ltx-turbo")).unwrap().url, "https://b");
        // A name nobody announces: a front of the API resolves it.
        assert_eq!(p("POST", "/v2/text-to-video", Some("ltx-2-fast")).unwrap().url, "https://b");
        assert_eq!(p("POST", "/minimax/h3-turbo/text-to-video", None).unwrap().family, "h3");
        assert_eq!(p("GET", "/lightricks/ltx-2.5/requests/r1/status", None).unwrap().family, "ltx");
        let up = format!("{}.abc", worker_tag("wb"));
        assert_eq!(p("PUT", &format!("/uploads/{up}"), None).unwrap().url, "https://b");
        assert_eq!(p("GET", &format!("/files/{up}/x.png"), None).unwrap().url, "https://b");
        let gone = format!("{}.abc", worker_tag("left"));
        assert_eq!(p("PUT", &format!("/uploads/{gone}"), None).unwrap_err().status, 404);
        assert_eq!(p("GET", "/fv/v1/jobs/j1", None).unwrap().family.is_empty(), false);
        assert!(is_submit("POST", &classify("POST", "/fv/v1/jobs")));
        assert!(!is_submit("GET", &classify("GET", "/fv/v1/jobs/x")));
        let mut empty = Registry::default();
        empty.families.insert("h3".into(), Vec::new());
        assert_eq!(plan_forward(&classify("POST", "/fv/v1/jobs"), &empty, "/fv/v1/jobs", "", Some("x"), None).unwrap_err().status, 503);
    }

    #[test]
    fn draining_and_loading_workers_are_not_fronts() {
        let mut r = reg();
        for w in r.families.get_mut("ltx").unwrap() {
            w.ready = false;
        }
        assert!(r.pick(Some("ltx"), "", None, None).is_none());
        for w in r.families.get_mut("h3").unwrap() {
            w.draining = true;
        }
        assert!(!r.any_ready());
    }

    #[test]
    fn finds_the_model_of_a_prefix() {
        let j = br#"{"prompt": "a \"model\" here", "nested": {"model": "no"}, "list": [1, {"model": "x"}], "model": "fasth3", "image": "data:..."}"#;
        assert_eq!(scan_model("application/json", j, true), Scan::Found("fasth3".into()));
        let k = br#"{"model":"MiniMax-H3","prompt":"#;
        assert_eq!(scan_model("application/json", k, false), Scan::Found("MiniMax-H3".into()));
        assert_eq!(scan_model("application/json", br#"{"prompt": "abc"#, false), Scan::More);
        assert_eq!(scan_model("application/json", br#"{"prompt": "abc"}"#, true), Scan::Absent);
        assert_eq!(scan_model("application/json", br#"[1]"#, true), Scan::Absent);
        let mp = b"--XY\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhi\r\n--XY\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nsora-2\r\n--XY--\r\n";
        assert_eq!(scan_model("multipart/form-data; boundary=XY", mp, true), Scan::Found("sora-2".into()));
        assert_eq!(scan_model("multipart/form-data; boundary=XY", &mp[..60], false), Scan::More);
    }

    #[test]
    fn verdicts_and_keys() {
        assert_eq!(parse_authorization("Key abc"), Some(("key", "abc")));
        assert_eq!(parse_authorization("bearer  x "), Some(("bearer", "x")));
        assert_eq!(parse_authorization("Basic x"), None);
        let d = key_digest("sk-good");
        assert_eq!(d.len(), 64);
        assert_eq!(key_id(&d), format!("key_{}", &d[..12]));
        let v = Verdict { v: 1, key: Some("key_1".into()), presented: true, valid: true, scheme: Some("key".into()), deny: None };
        assert_eq!(Verdict::parse(&v.header()), Some(v));
        let t = worker_tag("pod-1");
        assert_eq!(t.len(), 8);
        assert_eq!(token_tag(&format!("{t}.abc")), Some(t.as_str()));
        assert_eq!(token_tag("plain"), None);
        assert_eq!(query_param("a=1&task_id=12%2034&b", "task_id").as_deref(), Some("12 34"));
    }
}
