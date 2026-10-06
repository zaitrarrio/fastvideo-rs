//! Family Durable Objects (docs/serve/dispatch-do-family.md) on the fake
//! engine: fake-engine workers that serve several families through one
//! arbiter, a gateway whose pools name a family, and a native stand-in for
//! the family objects (the shared scheduler behind the Worker's routes, with
//! session admission and direct uploads against a local S3 mock that checks
//! SigV4 presigned URLs).
//!
//! - `multi_family_worker_never_holds_two_jobs`: a burst on two families;
//!   two workers with one slot each; no worker ever runs two jobs at once.
//! - `an_arbiter_nack_is_offered_elsewhere_at_once`: a stale credit makes a
//!   family object offer a job to a busy GPU; its 429 sends the job to the
//!   other worker with no backoff.
//! - `a_missed_ack_requeues_to_another_worker`: a worker that never acks;
//!   the ack deadline puts the job back and another worker runs it.
//! - `a_deploy_reconnects_and_reannounces_without_duplicates`: the objects
//!   restart (sockets drop, state reloads) while jobs run; every job runs
//!   once, on the worker that held it.
//! - `direct_upload_commits_through_the_family_object`: the output goes to
//!   the S3 mock through the object's part URLs; the job's artifact is the
//!   object key; the sha256 matches.
//! - `tail_upload_overlaps_the_write_and_resends_only_changed_parts`: the
//!   upload thread sends parts while the file grows; a header patched at the
//!   end costs part 1 only.
//! - `a_session_is_admitted_through_the_family_object`: a Reactor session
//!   through the gateway reserves the GPU on the worker (its batch job waits),
//!   the signalling goes to the worker's endpoint, stop frees the GPU.
//!
//! With `FV_EDGE_URL` + `FV_EDGE_TOKEN` (the Worker's internal token) the
//! burst, direct-upload and session tests run against a real dispatcher
//! (staging) instead of the stand-in, under per-run family names; the
//! direct-upload test then reads the object back through `/dl` with
//! `FV_EDGE_UPLOAD_KEY`. The tests that need the stand-in's internals skip.

#![cfg(all(feature = "http-client", feature = "openai-videos", feature = "minimax", feature = "fal", feature = "ltxapi"))]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::{FamilyMetrics, PoolStatus};
use fastvideo_serve::config::{Config, DispatchMode, EngineBackendKind, JobBackend, KeyStoreBackend, PoolCfg, PoolKind, Role};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::client::{D1Error, D1Transport, RawReply};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-family-user";
const ADMIN: &str = "fvadm_family_test";
const TOKEN: &str = "family-internal-token";
const BUCKET: &str = "fv-test-outputs";

fn init_log() {
    if std::env::var("RUST_LOG").is_ok() {
        let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_test_writer().try_init();
    }
}

fn tmp(tag: &str) -> PathBuf {
    tempfile::Builder::new().prefix(&format!("fv-family-{tag}-")).tempdir().unwrap().keep()
}

fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

/// An S3 mock: multipart uploads and objects, every request checked
/// against its SigV4 query signature.
mod s3 {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::{Arc, Mutex};

    use axum::body::Bytes;
    use axum::extract::{Path, RawQuery, State};
    use axum::http::{HeaderMap, Method, StatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::any;
    use axum::Router;
    use fastvideo_dispatch_proto::presign::{parse_query, S3Presign};

    #[derive(Default)]
    pub struct Store {
        /// upload id → (key, parts).
        pub uploads: HashMap<String, (String, BTreeMap<u16, Vec<u8>>)>,
        pub objects: HashMap<String, Vec<u8>>,
        /// (upload id, part) → PUTs.
        pub part_puts: HashMap<(String, u16), u32>,
        pub aborted: Vec<String>,
        pub bad_signatures: u32,
        next: u64,
    }

    #[derive(Clone)]
    pub struct Mock {
        pub store: Arc<Mutex<Store>>,
        pub signer: S3Presign,
    }

    /// `YYYYMMDDTHHMMSSZ` → Unix seconds.
    fn unix(ts: &str) -> Option<i64> {
        let n = |a: usize, b: usize| ts.get(a..b)?.parse::<i64>().ok();
        let (y, m, d, hh, mm, ss) = (n(0, 4)?, n(4, 6)?, n(6, 8)?, n(9, 11)?, n(11, 13)?, n(13, 15)?);
        let y2 = if m <= 2 { y - 1 } else { y };
        let era = y2.div_euclid(400);
        let yoe = y2 - era * 400;
        let mp = (m + 9) % 12;
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        Some((era * 146_097 + doe - 719_468) * 86_400 + hh * 3600 + mm * 60 + ss)
    }

    impl Mock {
        pub async fn start(access: &str, secret: &str) -> (Self, String) {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", l.local_addr().unwrap());
            let signer = S3Presign {
                endpoint: base.clone(),
                region: "auto".into(),
                bucket: super::BUCKET.into(),
                access_key: access.into(),
                secret_key: secret.into(),
                path_style: true,
            };
            let m = Self { store: Arc::default(), signer };
            let app = Router::new().route("/{bucket}/{*key}", any(handle)).with_state(m.clone());
            tokio::spawn(async move {
                let _ = axum::serve(l, app).await;
            });
            (m, base)
        }

        /// Recomputes the signature of a presigned request.
        fn signed(&self, method: &str, key: &str, query: &str) -> bool {
            let q = parse_query(query);
            let (Some(sig), Some(ts), Some(exp)) = (q.get("X-Amz-Signature"), q.get("X-Amz-Date"), q.get("X-Amz-Expires")) else { return false };
            let Some(t) = unix(ts) else { return false };
            let exp: i64 = exp.parse().unwrap_or(0);
            if t + exp < super::now() / 1000 {
                return false;
            }
            let extra: Vec<(&str, &str)> = q.iter().filter(|(k, _)| !k.starts_with("X-Amz-")).map(|(k, v)| (k.as_str(), v.as_str())).collect();
            let url = self.signer.presign(method, key, &extra, exp as u64, t);
            let want = parse_query(url.split_once('?').unwrap().1).remove("X-Amz-Signature").unwrap_or_default();
            &want == sig
        }
    }

    fn xml(status: StatusCode, body: String) -> Response {
        (status, [("content-type", "application/xml")], body).into_response()
    }

    async fn handle(State(m): State<Mock>, Path((bucket, key)): Path<(String, String)>, method: Method, RawQuery(q): RawQuery, _h: HeaderMap, body: Bytes) -> Response {
        let q = q.unwrap_or_default();
        if bucket != super::BUCKET || !m.signed(method.as_str(), &key, &q) {
            m.store.lock().unwrap().bad_signatures += 1;
            return xml(StatusCode::FORBIDDEN, "<Error><Code>SignatureDoesNotMatch</Code></Error>".into());
        }
        let p = parse_query(&q);
        let mut st = m.store.lock().unwrap();
        match (method.as_str(), p.get("uploadId"), p.get("partNumber")) {
            ("POST", None, None) if p.contains_key("uploads") => {
                st.next += 1;
                let id = format!("mpu-{}", st.next);
                st.uploads.insert(id.clone(), (key.clone(), BTreeMap::new()));
                xml(StatusCode::OK, format!("<InitiateMultipartUploadResult><Bucket>{bucket}</Bucket><Key>{key}</Key><UploadId>{id}</UploadId></InitiateMultipartUploadResult>"))
            }
            ("PUT", Some(id), Some(n)) => {
                let n: u16 = n.parse().unwrap_or(0);
                let Some((_, parts)) = st.uploads.get_mut(id) else { return xml(StatusCode::NOT_FOUND, "<Error><Code>NoSuchUpload</Code></Error>".into()) };
                let etag = format!("\"{}\"", fastvideo_dispatch_proto::presign::sha256_hex(&body));
                parts.insert(n, body.to_vec());
                *st.part_puts.entry((id.clone(), n)).or_default() += 1;
                (StatusCode::OK, [("etag", etag)]).into_response()
            }
            ("POST", Some(id), None) => {
                let Some((k, parts)) = st.uploads.remove(id) else { return xml(StatusCode::NOT_FOUND, "<Error><Code>NoSuchUpload</Code></Error>".into()) };
                // The listed parts, in order, with matching ETags.
                let text = String::from_utf8_lossy(&body).into_owned();
                let mut data = Vec::new();
                for chunk in text.split("<Part>").skip(1) {
                    let n: u16 = chunk.split("<PartNumber>").nth(1).and_then(|s| s.split('<').next()).and_then(|s| s.parse().ok()).unwrap_or(0);
                    let etag = chunk.split("<ETag>").nth(1).and_then(|s| s.split('<').next()).unwrap_or_default().trim_matches('"').to_owned();
                    let Some(b) = parts.get(&n) else { return xml(StatusCode::BAD_REQUEST, "<Error><Code>InvalidPart</Code></Error>".into()) };
                    if fastvideo_dispatch_proto::presign::sha256_hex(b) != etag {
                        return xml(StatusCode::BAD_REQUEST, "<Error><Code>InvalidPart</Code></Error>".into());
                    }
                    data.extend_from_slice(b);
                }
                st.objects.insert(k.clone(), data);
                xml(StatusCode::OK, format!("<CompleteMultipartUploadResult><Key>{k}</Key></CompleteMultipartUploadResult>"))
            }
            ("DELETE", Some(id), None) => {
                st.uploads.remove(id);
                st.aborted.push(id.clone());
                StatusCode::NO_CONTENT.into_response()
            }
            ("HEAD", None, None) | ("GET", None, None) => match st.objects.get(&key) {
                Some(b) if method == Method::HEAD => (StatusCode::OK, [("content-length", b.len().to_string())]).into_response(),
                Some(b) => (StatusCode::OK, b.clone()).into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
            _ => xml(StatusCode::BAD_REQUEST, "<Error><Code>InvalidRequest</Code></Error>".into()),
        }
    }

    /// The object operations a family object performs on its bucket.
    #[derive(Clone)]
    pub struct Client {
        pub signer: S3Presign,
        pub http: reqwest::Client,
    }

    impl Client {
        fn url(&self, method: &str, key: &str, extra: &[(&str, &str)]) -> String {
            self.signer.presign(method, key, extra, 300, super::now() / 1000)
        }
        pub async fn create(&self, key: &str) -> Result<String, String> {
            let r = self.http.post(self.url("POST", key, &[("uploads", "")])).send().await.map_err(|e| e.to_string())?;
            let t = r.text().await.map_err(|e| e.to_string())?;
            t.split("<UploadId>").nth(1).and_then(|s| s.split('<').next()).map(str::to_owned).ok_or(t)
        }
        pub fn part_url(&self, key: &str, upload: &str, n: u16) -> String {
            self.signer.part_url(key, upload, n, 600, super::now() / 1000)
        }
        pub async fn complete(&self, key: &str, upload: &str, parts: &[fastvideo_dispatch_proto::Part]) -> Result<u64, String> {
            let mut body = String::from("<CompleteMultipartUpload>");
            for p in parts {
                body.push_str(&format!("<Part><PartNumber>{}</PartNumber><ETag>{}</ETag></Part>", p.n, p.etag));
            }
            body.push_str("</CompleteMultipartUpload>");
            let r = self.http.post(self.url("POST", key, &[("uploadId", upload)])).body(body).send().await.map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Err(format!("complete: HTTP {}: {}", r.status(), r.text().await.unwrap_or_default()));
            }
            let h = self.http.head(self.url("HEAD", key, &[])).send().await.map_err(|e| e.to_string())?;
            h.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).ok_or_else(|| "no size".to_owned())
        }
        pub async fn abort(&self, key: &str, upload: &str) {
            let _ = self.http.delete(self.url("DELETE", key, &[("uploadId", upload)])).send().await;
        }
    }
}

/// The native stand-in for the family objects: the Worker's family routes,
/// the shared scheduler per object, session admission that waits for the
/// worker, and uploads in S3 mode against the mock.
mod family_do {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
    use axum::extract::{Path, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use fastvideo_dispatch_proto::sched::{Admit, Cfg, JobRec, Out, Sched, SessionRec, UploadRec, WorkerRec};
    use fastvideo_dispatch_proto::{self as proto, DoMsg, EnqueueReq, PartUrl, SessionGrant, SessionReq, WorkerMsg};
    use futures::{SinkExt, StreamExt};
    use tokio::sync::{mpsc, oneshot, watch};

    use super::now;
    use super::s3::Client;

    #[derive(Default)]
    pub struct Inner {
        scheds: HashMap<String, Sched>,
        conns: HashMap<(String, String), (u64, mpsc::UnboundedSender<String>)>,
        next_conn: u64,
        waiters: HashMap<String, oneshot::Sender<Result<SessionGrant, String>>>,
        /// Frames received: (object, worker, frame).
        pub frames: Vec<(String, String, WorkerMsg)>,
        /// Offers sent per job.
        pub offers: HashMap<String, u32>,
    }

    #[derive(Clone)]
    pub struct FamilyDo {
        pub inner: Arc<Mutex<Inner>>,
        cfg: Cfg,
        s3: Client,
        epoch: watch::Sender<u64>,
    }

    impl FamilyDo {
        pub async fn start(cfg: Cfg, s3: Client) -> (Self, String) {
            let (epoch, _) = watch::channel(0);
            let d = Self { inner: Arc::default(), cfg, s3, epoch };
            let app = Router::new()
                .route("/families/{f}/connect", get(connect))
                .route("/families/{f}/enqueue", post(enqueue))
                .route("/families/{f}/cancel/{job}", post(cancel))
                .route("/families/{f}/status", get(status))
                .route("/families/{f}/metrics", get(metrics))
                .route("/families/{f}/sessions", post(admit))
                .route("/families/{f}/sessions/{id}/{op}", post(session_op))
                .layer(axum::extract::DefaultBodyLimit::max(64 << 20))
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
                    let objs: Vec<String> = t.inner.lock().unwrap().scheds.keys().cloned().collect();
                    for o in objs {
                        let out = {
                            let mut g = t.inner.lock().unwrap();
                            let s = g.scheds.get_mut(&o).unwrap();
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
            (d, base)
        }

        /// A deploy: every socket drops, every object reloads from its rows.
        pub fn redeploy(&self) {
            let now = now();
            let mut g = self.inner.lock().unwrap();
            let objs: Vec<String> = g.scheds.keys().cloned().collect();
            for o in objs {
                let s = g.scheds.remove(&o).unwrap();
                let jobs: Vec<JobRec> = s.jobs().cloned().collect();
                let workers: Vec<WorkerRec> = s.workers().cloned().collect();
                let sessions: Vec<SessionRec> = s.sessions().cloned().collect();
                let uploads: Vec<UploadRec> = s.uploads().cloned().collect();
                let mut s = Sched::restore_all(o.clone(), self.cfg.clone(), jobs, workers, sessions, uploads);
                let ids: Vec<String> = s.workers().filter(|w| w.connected).map(|w| w.worker_id.clone()).collect();
                for id in ids {
                    s.disconnect(&id, now);
                }
                g.scheds.insert(o, s);
            }
            g.conns.clear();
            let v = *self.epoch.borrow() + 1;
            let _ = self.epoch.send(v);
        }

        /// Runs `f` on object `obj`'s scheduler, then its effects.
        pub fn with<T>(&self, obj: &str, f: impl FnOnce(&mut Sched) -> (T, Vec<Out>)) -> T {
            let (t, out) = {
                let mut g = self.inner.lock().unwrap();
                let cfg = self.cfg.clone();
                let s = g.scheds.entry(obj.to_owned()).or_insert_with(|| Sched::new(obj, cfg));
                f(s)
            };
            self.apply(obj, out);
            t
        }

        fn send(&self, obj: &str, worker: &str, msg: &DoMsg) {
            let g = self.inner.lock().unwrap();
            if let Some((_, tx)) = g.conns.get(&(obj.to_owned(), worker.to_owned())) {
                let _ = tx.send(serde_json::to_string(msg).unwrap());
            }
        }

        fn apply(&self, obj: &str, out: Vec<Out>) {
            for o in out {
                match o {
                    Out::Send { worker, msg } => {
                        if let DoMsg::Job { job_id, .. } = &msg {
                            *self.inner.lock().unwrap().offers.entry(job_id.clone()).or_default() += 1;
                        }
                        self.send(obj, &worker, &msg);
                    }
                    Out::Close { worker } => {
                        self.inner.lock().unwrap().conns.remove(&(obj.to_owned(), worker));
                    }
                    Out::PushSpilled { .. } => {}
                    Out::SessionReady { session_id, result } => {
                        if let Some(w) = self.inner.lock().unwrap().waiters.remove(&session_id) {
                            let _ = w.send(result);
                        }
                    }
                    Out::Grant { worker, req, job_id, key, upload_id, from, count, expires_ms } => {
                        let part_urls = (from..from.saturating_add(count)).map(|n| PartUrl { n, url: self.s3.part_url(&key, &upload_id, n) }).collect();
                        self.send(obj, &worker, &DoMsg::UploadGrant { req, job_id, upload_id, key, bucket: super::BUCKET.into(), part_urls, expires_ms, error: None });
                    }
                    Out::CreateUpload { req, key, parts, .. } => {
                        let (d, obj) = (self.clone(), obj.to_owned());
                        tokio::spawn(async move {
                            let r = d.s3.create(&key).await;
                            d.with(&obj, |s| ((), s.upload_created(&key, req, parts, r, now())));
                        });
                    }
                    Out::CompleteUpload { req, key, upload_id, parts, .. } => {
                        let (d, obj) = (self.clone(), obj.to_owned());
                        tokio::spawn(async move {
                            let r = d.s3.complete(&key, &upload_id, &parts).await;
                            d.with(&obj, |s| ((), s.upload_completed(&key, req, r, now())));
                        });
                    }
                    Out::AbortUpload { key, upload_id } => {
                        let s3 = self.s3.clone();
                        tokio::spawn(async move { s3.abort(&key, &upload_id).await });
                    }
                }
            }
            let mut g = self.inner.lock().unwrap();
            if let Some(s) = g.scheds.get_mut(obj) {
                let _ = s.take_dirty();
            }
        }

        /// Injects a frame as if `worker` had sent it on its socket.
        pub fn inject(&self, family: &str, worker: &str, msg: WorkerMsg) {
            self.with(&format!("family:{family}"), |s| ((), s.on_msg(worker, msg, now())));
        }

        pub fn nacks(&self, worker: &str, code: u16) -> usize {
            self.inner.lock().unwrap().frames.iter().filter(|(_, w, m)| w == worker && matches!(m, WorkerMsg::Nack { code: c, .. } if *c == code)).count()
        }
    }

    fn authed(h: &HeaderMap) -> bool {
        h.get(proto::TOKEN_HEADER).and_then(|v| v.to_str().ok()) == Some(super::TOKEN) || h.get("authorization").and_then(|v| v.to_str().ok()) == Some(&format!("Bearer {}", super::ADMIN))
    }

    async fn connect(State(d): State<FamilyDo>, Path(f): Path<String>, h: HeaderMap, ws: WebSocketUpgrade) -> Response {
        if !authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let Some(worker) = h.get(proto::WORKER_HEADER).and_then(|v| v.to_str().ok()).map(str::to_owned) else {
            return StatusCode::BAD_REQUEST.into_response();
        };
        ws.on_upgrade(move |socket| run(d, format!("family:{f}"), worker, socket))
    }

    async fn run(d: FamilyDo, obj: String, worker: String, socket: WebSocket) {
        let (mut sink, mut stream) = socket.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let conn = {
            let mut g = d.inner.lock().unwrap();
            g.next_conn += 1;
            let c = g.next_conn;
            g.conns.insert((obj.clone(), worker.clone()), (c, tx));
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
                    d.inner.lock().unwrap().frames.push((obj.clone(), worker.clone(), msg.clone()));
                    d.with(&obj, |s| ((), s.on_msg(&worker, msg, now())));
                }
            }
        }
        let gone = {
            let mut g = d.inner.lock().unwrap();
            let key = (obj.clone(), worker.clone());
            let mine = g.conns.get(&key).is_some_and(|(c, _)| *c == conn);
            if mine {
                g.conns.remove(&key);
            }
            mine
        };
        if gone {
            d.with(&obj, |s| ((), s.disconnect(&worker, now())));
        }
    }

    async fn enqueue(State(d): State<FamilyDo>, Path(f): Path<String>, h: HeaderMap, Json(req): Json<EnqueueReq>) -> Response {
        if !authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let resp = d.with(&format!("family:{f}"), |s| s.enqueue(req, now()));
        (StatusCode::ACCEPTED, Json(resp)).into_response()
    }

    async fn cancel(State(d): State<FamilyDo>, Path((f, job)): Path<(String, String)>, h: HeaderMap) -> Response {
        if !authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        match d.with(&format!("family:{f}"), |s| s.cancel(&job, now())) {
            Some(st) => Json(serde_json::json!({"job_id": job, "state": st})).into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    }

    async fn status(State(d): State<FamilyDo>, Path(f): Path<String>, h: HeaderMap) -> Response {
        if !authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let st = d.with(&format!("family:{f}"), |s| (s.status(now()), Vec::new()));
        Json(st).into_response()
    }

    async fn metrics(State(d): State<FamilyDo>, Path(f): Path<String>, h: HeaderMap) -> Response {
        if !authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        Json(d.with(&format!("family:{f}"), |s| (s.metrics(now()), Vec::new()))).into_response()
    }

    async fn admit(State(d): State<FamilyDo>, Path(f): Path<String>, h: HeaderMap, Json(req): Json<SessionReq>) -> Response {
        if !authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let obj = format!("family:{f}");
        let (tx, rx) = oneshot::channel();
        let a = {
            let (a, out) = {
                let mut g = d.inner.lock().unwrap();
                let cfg = d.cfg.clone();
                let (a, out) = g.scheds.entry(obj.clone()).or_insert_with(|| Sched::new(&obj, cfg)).admit(req, now());
                if let Admit::Pending(id) = &a {
                    g.waiters.insert(id.clone(), tx);
                }
                (a, out)
            };
            d.apply(&obj, out);
            a
        };
        let err = |m: String| (StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({"error": {"kind": "queue_full", "message": m}}))).into_response();
        match a {
            Admit::Granted(g) => Json(g).into_response(),
            Admit::Refused(m) => err(m),
            Admit::Pending(_) => match tokio::time::timeout(Duration::from_secs(20), rx).await {
                Ok(Ok(Ok(g))) => Json(g).into_response(),
                Ok(Ok(Err(m))) => err(m),
                _ => StatusCode::SERVICE_UNAVAILABLE.into_response(),
            },
        }
    }

    async fn session_op(State(d): State<FamilyDo>, Path((f, id, op)): Path<(String, String, String)>, h: HeaderMap) -> Response {
        if !authed(&h) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let obj = format!("family:{f}");
        match op.as_str() {
            "renew" => match d.with(&obj, |s| (s.renew(&id, 0, now()), Vec::new())) {
                Some(g) => Json(g).into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
            "release" => match d.with(&obj, |s| s.release(&id, now())) {
                Some(st) => Json(serde_json::json!({"session_id": id, "state": st})).into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
            _ => StatusCode::NOT_FOUND.into_response(),
        }
    }
}

// ------------------------------------------------------------- harness

#[derive(Clone)]
struct Cuttable {
    mock: MockD1,
    dead: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl D1Transport for Cuttable {
    async fn post(&self, body: &Value) -> Result<RawReply, D1Error> {
        if self.dead.load(Ordering::SeqCst) {
            return Err(D1Error::Transport("worker is gone".into()));
        }
        self.mock.post(body).await
    }
}

struct Env {
    mock: MockD1,
    arts: PathBuf,
    s3: s3::Mock,
    /// The stand-in (`None`: an external dispatcher at `base`).
    native: Option<family_do::FamilyDo>,
    base: String,
    token: String,
    run: String,
    http: Http,
}

impl Env {
    async fn new() -> Self {
        Self::with_cfg(fastvideo_dispatch_proto::sched::Cfg { reconnect_grace_ms: 3_000, stale_after_ms: 8_000, ack_timeout_ms: 1_000, ..Default::default() }).await
    }
    async fn with_cfg(cfg: fastvideo_dispatch_proto::sched::Cfg) -> Self {
        let arts = tmp("arts");
        std::fs::create_dir_all(&arts).unwrap();
        let (s3, _) = s3::Mock::start("AKTEST", "test-secret").await;
        let client = s3::Client { signer: s3.signer.clone(), http: reqwest::Client::builder().no_proxy().build().unwrap() };
        let run = format!("{:x}", now() % 0xff_ffff);
        if let Some(base) = std::env::var("FV_EDGE_URL").ok().filter(|s| !s.is_empty()) {
            let token = std::env::var("FV_EDGE_TOKEN").expect("FV_EDGE_TOKEN with FV_EDGE_URL");
            return Self { mock: MockD1::new(), arts, s3, native: None, base: base.trim_end_matches('/').to_owned(), token, run, http: Http::new() };
        }
        let (d, base) = family_do::FamilyDo::start(cfg, client).await;
        Self { mock: MockD1::new(), arts, s3, native: Some(d), base, token: TOKEN.to_owned(), run, http: Http::new() }
    }
    fn external(&self) -> bool {
        self.native.is_none()
    }
    /// The stand-in (tests that need it skip on an external dispatcher).
    fn d(&self) -> &family_do::FamilyDo {
        self.native.as_ref().expect("the native stand-in")
    }
    /// A family's name for this run (unique on an external dispatcher, whose
    /// objects keep their state).
    fn fam(&self, f: &str) -> String {
        if self.external() {
            format!("{f}-{}", self.run)
        } else {
            f.to_owned()
        }
    }
    fn d1(&self, dead: &Arc<AtomicBool>) -> D1Client {
        D1Client::new(Arc::new(Cuttable { mock: self.mock.clone(), dead: dead.clone() })).with_retry(fastvideo_serve_kit::d1::RetryPolicy::immediate(1))
    }
    fn job(&self, external_id: &str) -> fastvideo_protocol::Job {
        let r = self.mock.sql("SELECT job FROM jobs WHERE external_id = ?", &[json!(external_id)]).unwrap().remove(0);
        serde_json::from_str(r["job"].as_str().unwrap()).unwrap()
    }
    fn worker_of(&self, external_id: &str) -> String {
        let r = self.mock.sql("SELECT worker FROM jobs WHERE external_id = ?", &[json!(external_id)]).unwrap().remove(0);
        r["worker"].as_str().unwrap_or_default().to_owned()
    }
    async fn status(&self, family: &str) -> PoolStatus {
        let r = self.http.0.get(format!("{}/families/{}/status", self.base, self.fam(family))).header("x-fv-internal-token", &self.token).send().await.unwrap();
        r.json().await.unwrap()
    }
    async fn metrics(&self, family: &str) -> FamilyMetrics {
        let r = self.http.0.get(format!("{}/families/{}/metrics", self.base, self.fam(family))).header("x-fv-internal-token", &self.token).send().await.unwrap();
        r.json().await.unwrap()
    }
    /// Waits until `n` workers are connected to `family`'s object.
    async fn connected(&self, family: &str, n: usize) {
        let t0 = Instant::now();
        loop {
            let st = self.status(family).await;
            if st.usable_workers() >= n {
                return;
            }
            assert!(t0.elapsed() < Duration::from_secs(30), "workers never connected to {family}: {st:?}");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn base_config(&self, tag: &str) -> Config {
        let mut env = BTreeMap::new();
        env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
        env.insert("FV_URL_SIGNING_KEY".to_owned(), "shared-signing-key".to_owned());
        env.insert("FV_STATE_DIR".to_owned(), tmp(tag).display().to_string());
        env.insert("FV_INTERNAL_TOKEN".to_owned(), self.token.clone());
        env.insert("FV_ARTIFACTS_DIR".to_owned(), self.arts.display().to_string());
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

    async fn serve(&self, mut c: Config) -> Running {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        c.server.public_base_url = Some(base.clone());
        c.validate().unwrap();
        let dead = Arc::new(AtomicBool::new(false));
        let app = App::build(c, Overrides { d1: Some(self.d1(&dead)), ..Overrides::default() }).await.unwrap();
        app.gate.engine().wait_ready().await;
        let router = app.router.clone();
        let task = tokio::spawn(async move {
            let _ = axum::serve(l, router).await;
        });
        Running { app, base, dead, task }
    }

    /// A fake-engine worker serving `families` through one arbiter.
    async fn worker(&self, tag: &str, families: &[&str], models: &[&str], step_ms: u64, direct: bool) -> Running {
        let mut c = self.base_config(tag);
        c.server.role = Role::Worker;
        c.server.worker_id = Some(format!("worker-{tag}"));
        c.engine.fake.models = models.iter().map(|m| m.to_string()).collect();
        c.engine.fake.step_ms = step_ms;
        c.dispatch.do_url = Some(self.base.clone());
        c.dispatch.families = families.iter().map(|f| self.fam(f)).collect();
        c.dispatch.capacity = 1;
        c.dispatch.sessions = 1;
        c.dispatch.status_s = 2;
        c.dispatch.direct_upload = direct;
        self.serve(c).await
    }

    /// A gateway with one durable-object pool per (pool, family, models).
    async fn gateway(&self, pools: &[(&str, &str, &[&str])]) -> Running {
        let mut c = self.base_config("gw");
        c.engine.backend = EngineBackendKind::Remote;
        c.pools = pools
            .iter()
            .map(|(id, family, models)| PoolCfg {
                id: (*id).into(),
                kind: PoolKind::Pod,
                fake_models: models.iter().map(|m| m.to_string()).collect(),
                stale_after_s: 3,
                retries: 1,
                dispatch: DispatchMode::DurableObject,
                do_url: Some(self.base.clone()),
                family: Some(self.fam(family)),
                ..PoolCfg::default()
            })
            .collect();
        c.gateway.tick_s = 1;
        c.gateway.watch_poll_ms = 100;
        self.serve(c).await
    }
}

struct Running {
    app: App,
    base: String,
    dead: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl Running {
    #[allow(dead_code)]
    fn kill(self) {
        self.dead.store(true, Ordering::SeqCst);
        self.task.abort();
        drop(self.app);
    }
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
    async fn submit(&self, g: &str, model: &str, prompt: &str) -> String {
        let (s, v) = self.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": model, "prompt": prompt}))).await;
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
        self.wait(g, id, |v| matches!(v["status"].as_str(), Some("succeeded" | "failed" | "cancelled")), Duration::from_secs(120)).await
    }
}

/// Waits until the gateway sees every pool available.
async fn available(http: &Http, g: &str, n: usize) {
    let t0 = Instant::now();
    loop {
        let (_, v) = http.call("GET", &format!("{g}/fv/v1/status"), None).await;
        if v["pools"].as_array().is_some_and(|a| a.iter().filter(|p| p["available"] == true).count() >= n) {
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "pools never available: {v}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ----------------------------------------------------------------- tests

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn multi_family_worker_never_holds_two_jobs() {
    init_log();
    let e = Env::new().await;
    let models: &[&str] = &["fake-h3-turbo", "fake-ltx-turbo"];
    let a = e.worker("a", &["h3", "ltx"], models, 20, false).await;
    let b = e.worker("b", &["h3", "ltx"], models, 20, false).await;
    e.connected("h3", 2).await;
    e.connected("ltx", 2).await;
    let gw = e.gateway(&[("p-h3", "h3", &["fake-h3-turbo"]), ("p-ltx", "ltx", &["fake-ltx-turbo"])]).await;
    let g = gw.base.clone();
    available(&e.http, &g, 2).await;

    // A burst on both families at once.
    let subs = (0..12).map(|i| {
        let (http, g) = (&e.http, g.clone());
        async move { http.submit(&g, if i % 2 == 0 { "fake-h3-turbo" } else { "fake-ltx-turbo" }, &format!("burst {i}")).await }
    });
    let ids = futures::future::join_all(subs).await;
    let mut spans: BTreeMap<String, Vec<(i128, i128)>> = BTreeMap::new();
    for (i, id) in ids.iter().enumerate() {
        let v = e.http.finished(&g, id).await;
        assert_eq!(v["status"], "succeeded", "{v}");
        let j = e.job(id);
        let (s, c) = (j.started_at.unwrap().unix_timestamp_nanos(), j.completed_at.unwrap().unix_timestamp_nanos());
        spans.entry(e.worker_of(id)).or_default().push((s, c));
        // Each job went through its own family's pool (and object).
        let want = if i % 2 == 0 { "p-h3" } else { "p-ltx" };
        let row = e.mock.sql("SELECT d.pool AS pool FROM gw_dispatch d JOIN jobs j ON j.id = d.job_id WHERE j.external_id = ?", &[json!(id)]).unwrap();
        assert_eq!(row[0]["pool"], want, "{v}");
    }
    // Each worker ran its jobs one after another: its one slot was never
    // given to two families at once.
    for (w, mut s) in spans.clone() {
        s.sort();
        for p in s.windows(2) {
            assert!(p[0].1 <= p[1].0, "worker {w} ran two jobs at once: {s:?}");
        }
    }
    assert_eq!(spans.len(), 2, "both workers took jobs: {spans:?}");
    if let Some(d) = &e.native {
        let offers: u32 = d.inner.lock().unwrap().offers.values().sum();
        eprintln!("FAMILY-BURST: 12 jobs, {offers} offers, 429 nacks: a {} b {}", d.nacks("worker-a", 429), d.nacks("worker-b", 429));
    }
    let q: Vec<f64> = ids.iter().map(|id| { let j = e.job(id); (j.started_at.unwrap() - j.created_at).as_seconds_f64() }).collect();
    eprintln!("FAMILY-BURST: queue s {:?}", q.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    let m = e.metrics("h3").await;
    assert_eq!((m.queued, m.running, m.workers), (0, 0, 2), "{m:?}");
    drop((a, b, gw));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn an_arbiter_nack_is_offered_elsewhere_at_once() {
    init_log();
    let e = Env::new().await;
    if e.external() {
        eprintln!("skipped: needs the native stand-in");
        return;
    }
    // a serves h3 and ltx; b serves ltx only.
    let a = e.worker("a", &["h3", "ltx"], &["fake-h3-turbo", "fake-ltx-turbo"], 400, false).await;
    let b = e.worker("b", &["ltx"], &["fake-ltx-turbo"], 20, false).await;
    e.connected("h3", 1).await;
    e.connected("ltx", 2).await;
    let gw = e.gateway(&[("p-h3", "h3", &["fake-h3-turbo"]), ("p-ltx", "ltx", &["fake-ltx-turbo"])]).await;
    let g = gw.base.clone();
    available(&e.http, &g, 2).await;
    // a's GPU is busy with an h3 job.
    let h = e.http.submit(&g, "fake-h3-turbo", "long").await;
    e.http.wait(&g, &h, |v| v["status"] == "running", Duration::from_secs(30)).await;
    // A stale credit from a (as if a `slots` frame crossed the h3 offer):
    // the ltx object believes a has a free slot, and prefers it (worker-a < worker-b).
    e.d().inject("ltx", "worker-a", fastvideo_dispatch_proto::WorkerMsg::Slots(fastvideo_dispatch_proto::Slots { free: 1, session_free: 0, offers_seen: 1_000 }));
    let t0 = Instant::now();
    let l = e.http.submit(&g, "fake-ltx-turbo", "re-offered").await;
    let v = e.http.finished(&g, &l).await;
    assert_eq!(v["status"], "succeeded", "{v}");
    assert_eq!(e.worker_of(&l), "worker-b", "the job ran on the free GPU");
    assert!(e.d().nacks("worker-a", 429) >= 1, "a refused the offer");
    let j = e.job(&l);
    let queue = (j.started_at.unwrap() - j.created_at).as_seconds_f64();
    eprintln!("FAMILY-NACK: re-offered after a 429; queue {queue:.3} s, end-to-end {:.3} s", t0.elapsed().as_secs_f64());
    assert!(queue < 1.0, "no backoff after an arbiter nack: {queue}");
    let hv = e.http.finished(&g, &h).await;
    assert_eq!(hv["status"], "succeeded");
    drop((a, b, gw));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_missed_ack_requeues_to_another_worker() {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::Message;
    init_log();
    let e = Env::new().await;
    if e.external() {
        eprintln!("skipped: needs the native stand-in");
        return;
    }
    // A "worker" that announces a free slot and never answers an offer.
    let mut req = format!("{}/families/wan/connect", e.base.replace("http://", "ws://")).into_client_request().unwrap();
    req.headers_mut().insert("x-fv-internal-token", TOKEN.parse().unwrap());
    req.headers_mut().insert("x-fv-worker-id", "aaa-silent".parse().unwrap());
    let (ws, _) = tokio_tungstenite::connect_async(req).await.unwrap();
    let (mut sink, mut stream) = ws.split();
    let hello = json!({"t": "hello", "worker_id": "aaa-silent", "pool": "family:wan", "proto": 2, "capacity": 1, "slots": {"free": 1}});
    sink.send(Message::text(hello.to_string())).await.unwrap();
    let offers = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let o2 = offers.clone();
    let silent = tokio::spawn(async move {
        let _sink = sink;
        while let Some(Ok(m)) = stream.next().await {
            if m.to_text().is_ok_and(|t| t.contains("\"t\":\"job\"")) {
                o2.fetch_add(1, Ordering::SeqCst);
            }
        }
    });
    e.connected("wan", 1).await;
    let gw = e.gateway(&[("p-wan", "wan", &["fake-wan"])]).await;
    let g = gw.base.clone();
    available(&e.http, &g, 1).await;
    let id = e.http.submit(&g, "fake-wan", "acked late").await;
    // The silent worker got the offer; nobody else is there yet.
    let t0 = Instant::now();
    while offers.load(Ordering::SeqCst) == 0 {
        assert!(t0.elapsed() < Duration::from_secs(10), "no offer to the silent worker");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let b = e.worker("b", &["wan"], &["fake-wan"], 5, false).await;
    let v = e.http.finished(&g, &id).await;
    assert_eq!(v["status"], "succeeded", "{v}");
    assert_eq!(e.worker_of(&id), "worker-b");
    let st = e.status("wan").await;
    eprintln!("FAMILY-ACK-TIMEOUT: requeued after the ack deadline; workers {:?}", st.workers.iter().map(|w| (&w.worker_id, w.held)).collect::<Vec<_>>());
    assert_eq!(e.d().inner.lock().unwrap().offers[&e.job(&id).id.to_string()], 2, "offered twice (the second under a new lease)");
    silent.abort();
    drop((b, gw));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_deploy_reconnects_and_reannounces_without_duplicates() {
    init_log();
    let e = Env::new().await;
    if e.external() {
        eprintln!("skipped: needs the native stand-in");
        return;
    }
    let a = e.worker("a", &["wan"], &["fake-wan"], 150, false).await;
    let b = e.worker("b", &["wan"], &["fake-wan"], 150, false).await;
    e.connected("wan", 2).await;
    let gw = e.gateway(&[("p-wan", "wan", &["fake-wan"])]).await;
    let g = gw.base.clone();
    available(&e.http, &g, 1).await;
    let mut ids = Vec::new();
    for i in 0..4 {
        ids.push(e.http.submit(&g, "fake-wan", &format!("deploy {i}")).await);
    }
    // Two run, two wait in the object.
    for id in &ids[..2] {
        e.http.wait(&g, id, |v| v["status"] == "running", Duration::from_secs(30)).await;
    }
    let before: BTreeMap<String, String> = ids[..2].iter().map(|id| (id.clone(), e.worker_of(id))).collect();
    let t0 = Instant::now();
    e.d().redeploy();
    e.connected("wan", 2).await;
    eprintln!("FAMILY-REDEPLOY: workers back in {:.3} s", t0.elapsed().as_secs_f64());
    for id in &ids {
        let v = e.http.finished(&g, id).await;
        assert_eq!(v["status"], "succeeded", "{v}");
    }
    for (id, w) in &before {
        assert_eq!(&e.worker_of(id), w, "{id} finished where it ran before the deploy");
    }
    let offers = e.d().inner.lock().unwrap().offers.clone();
    for id in &ids {
        let jid = e.job(id).id.to_string();
        assert_eq!(offers.get(&jid).copied().unwrap_or(0), 1, "{id} was offered once: {offers:?}");
    }
    let st = e.status("wan").await;
    assert!(st.failed.is_empty(), "{:?}", st.failed);
    drop((a, b, gw));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn direct_upload_commits_through_the_family_object() {
    init_log();
    let e = Env::new().await;
    let w = e.worker("up", &["wan"], &["fake-wan"], 5, true).await;
    e.connected("wan", 1).await;
    let gw = e.gateway(&[("p-wan", "wan", &["fake-wan"])]).await;
    let g = gw.base.clone();
    available(&e.http, &g, 1).await;
    let id = e.http.submit(&g, "fake-wan", "direct upload").await;
    let v = e.http.finished(&g, &id).await;
    assert_eq!(v["status"], "succeeded", "{v}");
    let job = e.job(&id);
    let art = &job.artifacts[0];
    let fastvideo_protocol::ArtifactLocation::Object { bucket, key } = &art.location else { panic!("not an object: {art:?}") };
    assert!(key.starts_with(&format!("outputs/{}/{}/1-1/", e.fam("wan"), job.id)), "{key}");
    if e.external() {
        // Read it back through the Worker (`/dl`, signed with the upload key).
        let secret = std::env::var("FV_EDGE_UPLOAD_KEY").expect("FV_EDGE_UPLOAD_KEY with FV_EDGE_URL");
        let cap = fastvideo_dispatch_proto::presign::Cap::Get { key: key.clone() };
        let url = fastvideo_dispatch_proto::presign::cap_url(&e.base, secret.as_bytes(), &cap, now() + 60_000);
        let body = e.http.0.get(url).send().await.unwrap().bytes().await.unwrap();
        assert_eq!(body.len() as u64, art.bytes, "the object in R2");
        eprintln!("FAMILY-UPLOAD: {bucket}/{key} {} bytes, sha256 {}", body.len(), fastvideo_dispatch_proto::presign::sha256_hex(&body));
        drop((w, gw));
        return;
    }
    assert_eq!(bucket, BUCKET);
    {
        let st = e.s3.store.lock().unwrap();
        let obj = st.objects.get(key).expect("the object is in the bucket");
        assert_eq!(obj.len() as u64, art.bytes);
        assert_eq!(st.bad_signatures, 0);
        assert!(st.uploads.is_empty(), "no upload left open");
    }
    // The family object recorded the result (key, bytes, sha256).
    let r = e.d().with("family:wan", |s| (s.job(&job.id.to_string()).and_then(|j| j.result.clone()), Vec::new())).expect("a result");
    assert_eq!((r.key.as_str(), r.bytes), (key.as_str(), art.bytes));
    assert_eq!(r.sha256, fastvideo_dispatch_proto::presign::sha256_hex(e.s3.store.lock().unwrap().objects.get(key).unwrap()));
    // A cancelled job's upload is aborted.
    let c = e.http.submit(&g, "fake-wan", "cancel me").await;
    let _ = e.http.call("DELETE", &format!("{g}/fv/v1/jobs/{c}"), None).await;
    let cv = e.http.finished(&g, &c).await;
    assert!(matches!(cv["status"].as_str(), Some("cancelled" | "succeeded")), "{cv}");
    let t0 = Instant::now();
    while !e.s3.store.lock().unwrap().uploads.is_empty() {
        assert!(t0.elapsed() < Duration::from_secs(10), "an upload stayed open");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    drop((w, gw));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tail_upload_overlaps_the_write_and_resends_only_changed_parts() {
    use std::io::{Seek, SeekFrom, Write};

    use fastvideo_dispatch_proto::{DoMsg, PartUrl, Scope, WorkerMsg};
    use fastvideo_serve::edge_link::LinkHandle;
    use fastvideo_serve::upload::{UploadSpec, Uploads};
    init_log();
    let e = Env::new().await;
    let client = s3::Client { signer: e.s3.signer.clone(), http: reqwest::Client::builder().no_proxy().build().unwrap() };
    // A family object reduced to the upload frames.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<WorkerMsg>();
    let link = LinkHandle::new(Scope::Family("wan".into()), tx);
    let l2 = link.clone();
    tokio::spawn(async move {
        let mut key = String::new();
        while let Some(m) = rx.recv().await {
            let answer = match m {
                WorkerMsg::UploadInit { req, job_id, name, .. } => {
                    key = format!("outputs/wan/{job_id}/1-1/{name}");
                    let id = client.create(&key).await.unwrap();
                    let part_urls = (1..=4).map(|n| PartUrl { n, url: client.part_url(&key, &id, n) }).collect();
                    DoMsg::UploadGrant { req, job_id, upload_id: id, key: key.clone(), bucket: BUCKET.into(), part_urls, expires_ms: now() + 600_000, error: None }
                }
                WorkerMsg::UploadMore { req, job_id, upload_id, from, count } => {
                    let part_urls = (from..from + count).map(|n| PartUrl { n, url: client.part_url(&key, &upload_id, n) }).collect();
                    DoMsg::UploadGrant { req, job_id, upload_id, key: key.clone(), bucket: BUCKET.into(), part_urls, expires_ms: now() + 600_000, error: None }
                }
                WorkerMsg::UploadDone { req, job_id, upload_id, parts, bytes, .. } => {
                    let size = client.complete(&key, &upload_id, &parts).await.unwrap();
                    DoMsg::UploadCommitted { req, job_id, key: key.clone(), bucket: BUCKET.into(), bytes: size, ok: size == bytes, error: None }
                }
                _ => continue,
            };
            l2.resolve(answer);
        }
    });
    let part = 64 * 1024u64;
    let uploads = Uploads::new(part);
    let dir = tmp("tail");
    let job = fastvideo_protocol::JobId::new();
    let file = dir.join(job.to_string()).join("output.mp4");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, b"").unwrap();
    uploads.start(UploadSpec { job, attempt: 1, lease: 1, tail: file.clone(), name: "output.mp4".into(), content_type: "video/mp4".into(), link });
    // The "encoder": six full parts and a short tail, 120 ms apart, then a
    // patch of the first bytes (as a muxer fixing its header).
    let mut f = std::fs::OpenOptions::new().append(true).open(&file).unwrap();
    let mut all = Vec::new();
    for i in 0..6u8 {
        let chunk = vec![i.wrapping_mul(31).wrapping_add(7); part as usize];
        f.write_all(&chunk).unwrap();
        f.flush().unwrap();
        all.extend_from_slice(&chunk);
        tokio::time::sleep(Duration::from_millis(120)).await;
    }
    let tail = vec![0xAB; 1234];
    f.write_all(&tail).unwrap();
    all.extend_from_slice(&tail);
    drop(f);
    let mut f = std::fs::OpenOptions::new().write(true).open(&file).unwrap();
    f.seek(SeekFrom::Start(4)).unwrap();
    f.write_all(b"moov").unwrap();
    drop(f);
    all[4..8].copy_from_slice(b"moov");
    let c = uploads.finish(job, &file).await.expect("a direct upload").expect("committed");
    eprintln!("FAMILY-TAIL: {:?}", c.stats);
    assert_eq!(c.bytes, all.len() as u64);
    assert_eq!(c.sha256, fastvideo_dispatch_proto::presign::sha256_hex(&all));
    let st = e.s3.store.lock().unwrap();
    assert_eq!(st.objects.get(&c.key).map(Vec::as_slice), Some(&all[..]));
    let puts = |n: u16| st.part_puts.iter().filter(|((_, p), _)| *p == n).map(|(_, c)| *c).sum::<u32>();
    assert_eq!(puts(1), 2, "the patched first part went twice");
    for n in 2..=7 {
        assert_eq!(puts(n), 1, "part {n} went once");
    }
    assert!(c.stats.bytes_early >= 5 * part, "most parts went while the file grew: {:?}", c.stats);
    assert_eq!(c.stats.bytes_resent, part);
}

#[cfg(feature = "reactor")]
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_session_is_admitted_through_the_family_object() {
    init_log();
    let e = Env::new().await;
    // One GPU: Reactor sessions (sfwan) and batch jobs (wan) share its slot.
    let w = e.worker("rt", &["sfwan", "wan"], &["fake-sfwan", "fake-wan"], 5, false).await;
    e.connected("sfwan", 1).await;
    e.connected("wan", 1).await;
    let gw = e.gateway(&[("sfwan-live", "sfwan", &["fake-sfwan"]), ("p-wan", "wan", &["fake-wan"])]).await;
    let g = gw.base.clone();
    available(&e.http, &g, 2).await;

    let t0 = Instant::now();
    let (s, v) = e.http.call("POST", &format!("{g}/start_session"), Some(json!({}))).await;
    assert_eq!(s, 200, "{v}");
    eprintln!("FAMILY-SESSION: admitted and started in {:.3} s", t0.elapsed().as_secs_f64());
    let st = e.status("sfwan").await;
    assert_eq!(st.sessions.len(), 1, "{st:?}");
    assert_eq!((st.sessions[0].state.as_str(), st.sessions[0].worker.as_deref()), ("live", Some("worker-rt")));
    let rows = e.mock.sql("SELECT target, body, state FROM gw_sessions WHERE kind = 'reactor'", &[]).unwrap();
    assert_eq!(rows[0]["target"], w.base.as_str(), "signalling goes to the worker's endpoint");
    assert!(rows[0]["body"].as_str().unwrap().contains(&st.sessions[0].session_id));
    // While the session holds the GPU, a batch job waits.
    let j = e.http.submit(&g, "fake-wan", "after the session").await;
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let (_, jv) = e.http.call("GET", &format!("{g}/fv/v1/jobs/{j}"), None).await;
    assert_eq!(jv["status"], "queued", "the GPU is the session's: {jv}");
    // A second session finds no slot.
    let sec = e.http.0.post(format!("{}/families/{}/sessions", e.base, e.fam("sfwan"))).header("x-fv-internal-token", &e.token).json(&json!({"kind": "reactor"})).send().await.unwrap();
    assert_eq!(sec.status().as_u16(), 429);
    // The gateway renews the lease from its tick (TTL 60 s here; just check it lives on).
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    assert_eq!(e.status("sfwan").await.sessions.len(), 1);
    // Stop: the lease ends on both sides and the job runs.
    let (s, _) = e.http.call("POST", &format!("{g}/stop_session"), Some(json!({"reason": "done"}))).await;
    assert_eq!(s, 200);
    let jv = e.http.finished(&g, &j).await;
    assert_eq!(jv["status"], "succeeded", "{jv}");
    assert!(e.status("sfwan").await.sessions.is_empty());
    drop((w, gw));
}
