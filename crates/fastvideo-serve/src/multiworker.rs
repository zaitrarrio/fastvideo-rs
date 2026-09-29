//! Multi-worker mode: which routes a replica may serve when a load balancer
//! spreads requests over several workers (design §6.2, Runpod serverless
//! load balancer with `workers.max > 1`).
//!
//! A Runpod load-balancer endpoint sends each request to any worker, and
//! clients cannot pin one (`X-Runpod-Worker-Id` is not honoured). A route is
//! safe there only if every worker can answer it. [`scope`] sorts every
//! route of the §9 table ([`route_table`]) into:
//!
//! | [`Scope`] | Served when `server.workers_max > 1` | Routes |
//! |---|---|---|
//! | `Local` | always | health, `/metrics`, `/console`, catalogs (`/v1/models`, `/fv/v1/capabilities`, `/fal/schema`, JWKS), LTX v1 sync (the reply is the MP4 bytes), LTX and MiniMax 4xx stubs |
//! | `Sync` | artifacts shared (`artifacts.backend = s3`, R2) | `POST /v1/videos/sync`, fal `/run/{app}/…`: one request, but the reply links the output |
//! | `Jobs` | jobs shared (`jobs.backend = d1`) **and** artifacts shared | async submit, status, result, list of every API: any worker reads the job from D1 |
//! | `Keys` | minted keys shared (`auth.key_store = d1`) | `/fv/v1/admin/keys` (other workers pick changes up within 30 s) |
//! | `Pinned` | never | cancel and delete (the running job is authoritative on its worker, whose next write would undo a cancel written elsewhere), uploads (`/uploads`, `/v1/upload`, fal storage) and `/files` (worker-local disk), fal `status/stream`, `/fal/proxy`, `/fv/v1/streams*`, the fal director and the Reactor runtime (WebRTC sessions live on one worker) |
//!
//! [`layer`] enforces it: a request whose route is not served answers
//! `404` with a JSON reason and `x-fv-multi-worker: not-served`; a request
//! that matches no route of the table is refused the same way (fail
//! closed). `OPTIONS` (CORS preflight) passes. With `workers_max = 1`
//! nothing is filtered.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderValue, Method, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use serde::Serialize;
use serde_json::json;

use crate::config::{ArtifactBackend, Config, KeyStoreBackend};
use crate::router::{route_table, Owner, RouteSpec, LTX_STUBS};

/// What a route needs to be answered by any worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Answered entirely by the worker it reaches.
    Local,
    /// One request, but the reply links an artifact: needs shared artifacts.
    Sync,
    /// Async jobs: needs a shared job store and shared artifacts.
    Jobs,
    /// Minted API keys: needs a shared key store.
    Keys,
    /// Needs the worker that holds the job, upload or session.
    Pinned,
}

/// Which state the workers share.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Policy {
    /// `server.workers_max`; 1 means a single worker (nothing filtered).
    pub workers_max: u32,
    /// Jobs are in D1.
    pub jobs: bool,
    /// Artifacts are in S3/R2 (presigned URLs any worker can mint and read).
    pub artifacts: bool,
    /// Minted keys are in D1.
    pub keys: bool,
    /// This is the gateway (docs/serve/gateway.md §6): cancel/delete go
    /// through `gw_dispatch`, SSE polls D1 and session signalling follows
    /// D1 leases, so those routes are served by any replica; only uploads,
    /// `/fal/proxy` and (without R2) `/files` stay pinned.
    pub gateway: bool,
}

impl Policy {
    /// From the config and the job store actually built (`jobs_kind`, as
    /// `/healthz` reports it: only `d1` is shared).
    pub fn from_config(c: &Config, jobs_kind: &str) -> Self {
        Self {
            workers_max: c.server.workers_max.max(1),
            jobs: jobs_kind == "d1",
            artifacts: c.artifact_backend() == ArtifactBackend::S3,
            keys: c.key_store_backend() == KeyStoreBackend::D1,
            gateway: c.engine.backend == crate::config::EngineBackendKind::Remote,
        }
    }

    /// Whether the route at `path` (of `scope`) is served, with the
    /// gateway's wider set.
    pub fn serves_route(&self, scope: Scope, path: &str) -> bool {
        if self.serves(scope) {
            return true;
        }
        self.gateway && scope == Scope::Pinned && self.jobs && !gateway_pinned(path, self.artifacts)
    }

    pub fn multi(&self) -> bool {
        self.workers_max > 1
    }

    /// Whether a route of `scope` is served.
    pub fn serves(&self, scope: Scope) -> bool {
        if !self.multi() {
            return true;
        }
        match scope {
            Scope::Local => true,
            Scope::Sync => self.artifacts,
            Scope::Jobs => self.jobs && self.artifacts,
            Scope::Keys => self.keys,
            Scope::Pinned => false,
        }
    }

    /// Why `scope` is not served (`None` when it is).
    pub fn refusal(&self, scope: Scope) -> Option<&'static str> {
        if self.serves(scope) {
            return None;
        }
        Some(match scope {
            Scope::Local => unreachable!("local routes are always served"),
            Scope::Sync => "its reply links the output, and artifacts are worker-local (set artifacts.backend = s3 / the FV_R2_* variables)",
            Scope::Jobs => "async jobs need a job store and artifacts every worker shares (jobs.backend = d1 and artifacts.backend = s3)",
            Scope::Keys => "minted API keys need a key store every worker shares (auth.key_store = d1)",
            Scope::Pinned => "it needs the worker that holds the job, upload or session, and a load balancer cannot route to it",
        })
    }
}

/// Routes a gateway replica still cannot serve for another one: its upload
/// store is local disk (and `/files` without R2); `/fal/proxy` re-enters the
/// router past this filter.
fn gateway_pinned(path: &str, artifacts_shared: bool) -> bool {
    path.starts_with("/uploads/")
        || path == "/v1/upload"
        || path == "/storage/upload/initiate"
        || path == "/fal/proxy"
        || (path.starts_with("/files/") && !artifacts_shared)
}

/// The scope of one §9 route.
pub fn scope(spec: &RouteSpec) -> Scope {
    let (m, p) = (spec.method, spec.path.as_str());
    match spec.owner {
        Owner::Serve | Owner::Console | Owner::Gateway => Scope::Local,
        Owner::ServeKit => Scope::Pinned, // /files, /uploads: worker-local disk
        Owner::FalDirector | Owner::Reactor => Scope::Pinned,
        Owner::OpenAiVideos => match (m, p) {
            (_, "/v1/models" | "/v1/models/{model}" | "/v1/model_info") => Scope::Local,
            ("POST", "/v1/videos/sync") => Scope::Sync,
            ("DELETE", _) => Scope::Pinned,
            _ => Scope::Jobs,
        },
        Owner::MiniMax => match (m, p) {
            (_, "/v2/h3_context_ir" | "/v2/video_regeneration") => Scope::Local,
            ("DELETE", _) => Scope::Pinned,
            _ => Scope::Jobs,
        },
        Owner::Ltx => {
            if p == "/v1/upload" {
                Scope::Pinned
            } else if LTX_STUBS.iter().any(|s| p.split('/').nth(2) == Some(s)) || p.starts_with("/v1/") {
                // 403/404 stubs, and v1 sync whose reply is the video itself.
                Scope::Local
            } else {
                Scope::Jobs
            }
        }
        Owner::Native => match (m, p) {
            (_, "/fv/v1/capabilities") => Scope::Local,
            (_, p) if p.starts_with("/fv/v1/admin/keys") => Scope::Keys,
            (_, p) if p.starts_with("/fv/v1/streams") => Scope::Pinned,
            ("DELETE", _) => Scope::Pinned,
            _ => Scope::Jobs,
        },
        Owner::Fal => {
            if p.starts_with("/fal/schema") || p == "/.well-known/jwks.json" {
                Scope::Local
            } else if p == "/fal/proxy" || p == "/storage/upload/initiate" {
                // The proxy re-enters the fal router (bypassing this filter).
                Scope::Pinned
            } else if p.ends_with("/cancel") || p.ends_with("/status/stream") {
                Scope::Pinned
            } else if p.starts_with("/run/") {
                Scope::Sync
            } else {
                Scope::Jobs
            }
        }
    }
}

#[derive(Debug)]
struct Entry {
    method: &'static str,
    segs: Vec<Option<String>>,
    scope: Scope,
}

/// The route table as a request matcher.
#[derive(Debug)]
pub struct Matcher {
    entries: Vec<Entry>,
}

impl Matcher {
    pub fn new(fal_apps: &[String]) -> Self {
        let entries = route_table(fal_apps)
            .iter()
            .map(|s| Entry {
                method: s.method,
                segs: s
                    .path
                    .split('/')
                    .skip(1)
                    .map(|seg| (!seg.starts_with('{')).then(|| seg.to_owned()))
                    .collect(),
                scope: scope(s),
            })
            .collect();
        Self { entries }
    }

    /// The scope of the route `method path` dispatches to (static segments
    /// win over parameters, as in the router), or `None` when no route of
    /// the table matches.
    pub fn scope_of(&self, method: &Method, path: &str) -> Option<Scope> {
        let method = if method == Method::HEAD { "GET" } else { method.as_str() };
        let segs: Vec<&str> = path.split('/').skip(1).collect();
        self.entries
            .iter()
            .filter(|e| e.method == method && e.segs.len() == segs.len())
            .filter(|e| {
                e.segs.iter().zip(&segs).all(|(t, s)| match t {
                    Some(lit) => lit == s,
                    None => !s.is_empty(),
                })
            })
            // Most specific: static segments from the left.
            .max_by_key(|e| e.segs.iter().map(|t| t.is_some()).collect::<Vec<_>>())
            .map(|e| e.scope)
    }
}

/// Wraps `router` so that only the routes `policy` serves are answered.
/// A no-op with a single worker.
pub fn layer(router: Router, policy: Policy, fal_apps: &[String]) -> Router {
    if !policy.multi() {
        return router;
    }
    let matcher = Arc::new(Matcher::new(fal_apps));
    router.layer(axum::middleware::from_fn(move |req: Request<Body>, next: Next| {
        let matcher = matcher.clone();
        async move {
            if req.method() == Method::OPTIONS {
                return next.run(req).await;
            }
            let found = matcher.scope_of(req.method(), req.uri().path());
            let why = match found {
                Some(s) if policy.serves_route(s, req.uri().path()) => None,
                Some(s) => policy.refusal(s),
                None => Some("it is not a route of this server's table"),
            };
            match why {
                None => next.run(req).await,
                Some(why) => refuse(&policy, req.method(), req.uri().path(), found, why),
            }
        }
    }))
}

fn refuse(policy: &Policy, method: &Method, path: &str, scope: Option<Scope>, why: &str) -> Response {
    let body = json!({
        "error": {
            "type": "not_served_by_multi_worker_deployment",
            "message": format!(
                "{method} {path} is not served by this deployment ({} workers behind a load balancer): {why}",
                policy.workers_max
            ),
            "scope": scope,
        }
    });
    let mut r = (StatusCode::NOT_FOUND, Json(body)).into_response();
    r.headers_mut().insert("x-fv-multi-worker", HeaderValue::from_static("not-served"));
    r
}

/// A startup summary of what is and is not served.
pub fn summary(policy: &Policy) -> serde_json::Value {
    let all = [Scope::Local, Scope::Sync, Scope::Jobs, Scope::Keys, Scope::Pinned];
    json!({
        "workers_max": policy.workers_max,
        "served": all.iter().filter(|s| policy.serves(**s)).collect::<Vec<_>>(),
        "not_served": all.iter().filter(|s| !policy.serves(**s)).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apps() -> Vec<String> {
        crate::config::ProtocolsCfg::default().fal_apps
    }

    #[test]
    fn every_route_has_the_expected_scope() {
        let m = Matcher::new(&apps());
        let cases: &[(&str, &str, Scope)] = &[
            ("GET", "/health", Scope::Local),
            ("GET", "/ping", Scope::Local),
            ("GET", "/healthz", Scope::Local),
            ("GET", "/metrics", Scope::Local),
            ("GET", "/console/models/minimax/h3-max/text-to-video", Scope::Local),
            ("GET", "/v1/models", Scope::Local),
            ("GET", "/fv/v1/capabilities", Scope::Local),
            ("GET", "/fv/v1/status", Scope::Local),
            ("GET", "/fal/schema/minimax/h3-max/text-to-video", Scope::Local),
            ("GET", "/.well-known/jwks.json", Scope::Local),
            ("POST", "/v1/text-to-video", Scope::Local),
            ("POST", "/v2/retake", Scope::Local),
            ("GET", "/v2/retake/abc", Scope::Local),
            ("POST", "/v2/h3_context_ir", Scope::Local),
            ("POST", "/v1/videos/sync", Scope::Sync),
            ("POST", "/run/minimax/h3-max/text-to-video", Scope::Sync),
            ("POST", "/v1/videos", Scope::Jobs),
            ("GET", "/v1/videos", Scope::Jobs),
            ("GET", "/v1/videos/video_abc", Scope::Jobs),
            ("GET", "/v1/videos/video_abc/content", Scope::Jobs),
            ("HEAD", "/v1/videos/video_abc/content", Scope::Jobs),
            ("POST", "/generate", Scope::Jobs),
            ("GET", "/status/abc", Scope::Jobs),
            ("GET", "/video/abc", Scope::Jobs),
            ("POST", "/v2/video_generation", Scope::Jobs),
            ("GET", "/v2/query/video_generation/123", Scope::Jobs),
            ("GET", "/v2/query/video_generation", Scope::Jobs),
            ("POST", "/v2/text-to-video", Scope::Jobs),
            ("GET", "/v2/image-to-video/abc", Scope::Jobs),
            ("POST", "/fv/v1/jobs", Scope::Jobs),
            ("GET", "/fv/v1/jobs/fvjob_1/content", Scope::Jobs),
            ("POST", "/minimax/h3-max/text-to-video", Scope::Jobs),
            ("GET", "/minimax/h3-max/requests/abc/status", Scope::Jobs),
            ("GET", "/minimax/h3-max/text-to-video/requests/abc/response", Scope::Jobs),
            ("GET", "/fv/v1/admin/keys", Scope::Keys),
            ("DELETE", "/fv/v1/admin/keys/key_1", Scope::Keys),
            ("DELETE", "/v1/videos/video_abc", Scope::Pinned),
            ("DELETE", "/video/abc", Scope::Pinned),
            ("DELETE", "/v2/video_generation/123", Scope::Pinned),
            ("DELETE", "/fv/v1/jobs/fvjob_1", Scope::Pinned),
            ("PUT", "/minimax/h3-max/requests/abc/cancel", Scope::Pinned),
            ("GET", "/minimax/h3-max/requests/abc/status/stream", Scope::Pinned),
            ("GET", "/files/a/b.mp4", Scope::Pinned),
            ("PUT", "/uploads/tok", Scope::Pinned),
            ("POST", "/v1/upload", Scope::Pinned),
            ("POST", "/storage/upload/initiate", Scope::Pinned),
            ("POST", "/fal/proxy", Scope::Pinned),
            ("POST", "/fv/v1/streams", Scope::Pinned),
            ("POST", "/fv/v1/streams/s1/commands", Scope::Pinned),
            ("POST", "/wma/session", Scope::Pinned),
            ("POST", "/minimax/h3-max/director/ice", Scope::Pinned),
            ("POST", "/start_session", Scope::Pinned),
            ("GET", "/sessions/s/transport/webrtc/ice_servers", Scope::Pinned),
        ];
        for (meth, path, want) in cases {
            let got = m.scope_of(&Method::from_bytes(meth.as_bytes()).unwrap(), path);
            assert_eq!(got, Some(*want), "{meth} {path}");
        }
        for (meth, path) in [("GET", "/nope"), ("POST", "/minimax/h9/text-to-video"), ("PATCH", "/v1/videos"), ("GET", "/v1/videos/")] {
            assert_eq!(m.scope_of(&Method::from_bytes(meth.as_bytes()).unwrap(), path), None, "{meth} {path}");
        }
    }

    #[test]
    fn the_whole_table_is_classified_and_matchable() {
        let m = Matcher::new(&apps());
        for s in route_table(&apps()) {
            let uri = s.path.split('/').map(|seg| if seg.starts_with('{') { "x" } else { seg }).collect::<Vec<_>>().join("/");
            let got = m.scope_of(&Method::from_bytes(s.method.as_bytes()).unwrap(), &uri);
            assert_eq!(got, Some(scope(&s)), "{} {}", s.method, s.path);
        }
    }

    #[test]
    fn policy_per_shared_state() {
        let single = Policy { workers_max: 1, ..Default::default() };
        assert!(single.serves(Scope::Pinned));
        let bare = Policy { workers_max: 3, ..Default::default() };
        assert!(bare.serves(Scope::Local));
        for s in [Scope::Sync, Scope::Jobs, Scope::Keys, Scope::Pinned] {
            assert!(!bare.serves(s), "{s:?}");
        }
        let jobs_only = Policy { jobs: true, ..bare };
        assert!(!jobs_only.serves(Scope::Jobs), "D1 jobs with local artifacts");
        let shared = Policy { jobs: true, artifacts: true, keys: true, ..bare };
        for s in [Scope::Local, Scope::Sync, Scope::Jobs, Scope::Keys] {
            assert!(shared.serves(s), "{s:?}");
        }
        assert!(!shared.serves(Scope::Pinned));
        assert!(shared.refusal(Scope::Pinned).unwrap().contains("worker"));
    }
}
