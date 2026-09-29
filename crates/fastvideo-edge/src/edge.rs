//! The Worker front and the `PoolScheduler` Durable Object (see the crate
//! docs). wasm32 only.

use std::cell::{Cell, RefCell};

use fastvideo_dispatch_proto as proto;
use proto::sched::{Cfg, Dirty, JobRec, Out, Sched, WorkerRec};
use proto::{EnqueueReq, WorkerMsg};
use serde::{Deserialize, Serialize};
use serde_json::json;
use wasm_bindgen::JsValue;
use worker::*;

const DO_BINDING: &str = "POOL_SCHEDULER";
const D1_BINDING: &str = "DB";
/// R2 bucket for large envelopes (optional; without it everything stays in SQLite).
const R2_BINDING: &str = "ENVELOPES";
/// Envelopes larger than this (JSON bytes) are spilled to R2 (`SPILL_BYTES`).
const SPILL_BYTES: usize = 1 << 20;
/// Envelope chunk size in SQLite (a DO row / string is at most 2 MB).
const CHUNK: usize = 1_000_000;

fn now_ms() -> i64 {
    Date::now().as_millis() as i64
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn version(env: &Env) -> String {
    env.var("FV_EDGE_VERSION").map(|v| v.to_string()).unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_owned())
}

/// The presented token: `x-fv-internal-token`, else `Authorization: Bearer`.
fn presented(req: &Request) -> Option<String> {
    let h = req.headers();
    if let Ok(Some(t)) = h.get(proto::TOKEN_HEADER) {
        return Some(t);
    }
    h.get("authorization").ok().flatten().and_then(|v| v.strip_prefix("Bearer ").map(str::to_owned))
}

enum Auth {
    Ok,
    Denied,
    NotConfigured,
}

/// Checks the token against the secrets named (any of them).
fn check(env: &Env, req: &Request, secrets: &[&str]) -> Auth {
    let mut configured = false;
    let Some(got) = presented(req) else {
        return if secrets.iter().any(|s| env.secret(s).is_ok()) { Auth::Denied } else { Auth::NotConfigured };
    };
    for s in secrets {
        if let Ok(want) = env.secret(s) {
            let want = want.to_string();
            if want.is_empty() {
                continue;
            }
            configured = true;
            if ct_eq(got.as_bytes(), want.as_bytes()) {
                return Auth::Ok;
            }
        }
    }
    if configured {
        Auth::Denied
    } else {
        Auth::NotConfigured
    }
}

fn json_err(status: u16, kind: &str, message: &str) -> Result<Response> {
    Ok(Response::from_json(&json!({"error": {"kind": kind, "message": message}}))?.with_status(status))
}

/// `/pools/{pool}/{action}[/{job}]`.
fn split(path: &str) -> Option<(String, String, Option<String>)> {
    let mut it = path.trim_start_matches('/').split('/');
    if it.next()? != "pools" {
        return None;
    }
    let pool = it.next()?.to_owned();
    let action = it.next()?.to_owned();
    let arg = it.next().map(str::to_owned);
    if it.next().is_some() || !proto::valid_id(&pool) || arg.as_deref().is_some_and(|a| !proto::valid_id(a)) {
        return None;
    }
    Some((pool, action, arg))
}

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let path = req.path();
    if matches!(path.as_str(), "/" | "/healthz") {
        return Response::from_json(&json!({"object": "fv.edge", "version": version(&env)}));
    }
    let Some((pool, action, _)) = split(&path) else {
        return json_err(404, "not_found", "no such route");
    };
    let secrets: &[&str] = match action.as_str() {
        "connect" | "enqueue" | "cancel" => &["FV_INTERNAL_TOKEN"],
        "status" => &["FV_INTERNAL_TOKEN", "FV_ADMIN_TOKEN"],
        _ => return json_err(404, "not_found", "no such route"),
    };
    match check(&env, &req, secrets) {
        Auth::Ok => {}
        Auth::Denied => return json_err(401, "unauthorized", "a valid token is required"),
        Auth::NotConfigured => return json_err(503, "loading", "the dispatcher has no token configured"),
    }
    let ns = env.durable_object(DO_BINDING)?;
    // `POOL_LOCATIONS`: {"h3-turbo": "weur", …} (near the pool's pods).
    let hint = env
        .var("POOL_LOCATIONS")
        .ok()
        .and_then(|v| serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&v.to_string()).ok())
        .and_then(|m| m.get(&pool).and_then(|h| h.as_str()).map(str::to_owned));
    let stub = match hint {
        Some(h) => ns.get_by_name_with_location_hint(&pool, &h)?,
        None => ns.get_by_name(&pool)?,
    };
    stub.fetch_with_request(req).await
}

/// What a worker socket carries across hibernation.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Attach {
    worker_id: String,
    pool: String,
    /// Connection nonce: a replaced socket closing later does not
    /// disconnect the worker's new one.
    conn: String,
}

#[derive(Deserialize)]
struct RecRow {
    rec: String,
}

#[derive(Deserialize)]
struct EnvRow {
    job_id: String,
    data: String,
}

#[derive(Deserialize)]
struct MetaRow {
    v: String,
}

/// One pool's scheduler.
#[durable_object]
pub struct PoolScheduler {
    state: State,
    env: Env,
    sched: RefCell<Option<Sched>>,
    /// The alarm last set (ms), to skip redundant `setAlarm` calls.
    alarm_at: Cell<Option<i64>>,
}

impl PoolScheduler {
    fn sql(&self) -> worker::SqlStorage {
        self.state.storage().sql()
    }

    fn init_tables(&self) -> Result<()> {
        let sql = self.sql();
        sql.exec("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, rec TEXT NOT NULL)", None)?;
        sql.exec(
            "CREATE TABLE IF NOT EXISTS envelopes (job_id TEXT NOT NULL, part INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY (job_id, part))",
            None,
        )?;
        sql.exec("CREATE TABLE IF NOT EXISTS workers (worker_id TEXT PRIMARY KEY, rec TEXT NOT NULL)", None)?;
        sql.exec("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)", None)?;
        Ok(())
    }

    fn cfg(&self) -> Cfg {
        let mut c = Cfg::default();
        let n = |k: &str| self.env.var(k).ok().and_then(|v| v.to_string().parse::<i64>().ok()).filter(|v| *v > 0);
        if let Some(v) = n("ACK_TIMEOUT_MS") {
            c.ack_timeout_ms = v;
        }
        if let Some(v) = n("RECONNECT_GRACE_MS") {
            c.reconnect_grace_ms = v;
        }
        if let Some(v) = n("STALE_AFTER_MS") {
            c.stale_after_ms = v;
        }
        if let Some(v) = n("REDISPATCH_WAIT_MS") {
            c.redispatch_wait_ms = v;
        }
        c
    }

    /// The pool this object serves (stored on first use).
    fn pool_name(&self, from_path: Option<&str>) -> Result<String> {
        self.init_tables()?;
        let sql = self.sql();
        if let Some(p) = from_path {
            sql.exec("INSERT OR IGNORE INTO meta (k, v) VALUES ('pool', ?)", vec![p.into()])?;
            return Ok(p.to_owned());
        }
        let rows: Vec<MetaRow> = sql.exec("SELECT v FROM meta WHERE k = 'pool'", None)?.to_array()?;
        rows.into_iter().next().map(|r| r.v).ok_or_else(|| Error::RustError("pool name unknown".into()))
    }

    /// Loads the scheduler from SQLite when this instance has none yet
    /// (first event after a start, an eviction or a deploy).
    fn ensure(&self, pool: &str) -> Result<()> {
        if self.sched.borrow().is_some() {
            return Ok(());
        }
        self.init_tables()?;
        let sql = self.sql();
        let rows: Vec<RecRow> = sql.exec("SELECT rec FROM jobs", None)?.to_array()?;
        let mut jobs: Vec<JobRec> = rows.iter().filter_map(|r| serde_json::from_str(&r.rec).ok()).collect();
        let parts: Vec<EnvRow> = sql.exec("SELECT job_id, data FROM envelopes ORDER BY job_id, part", None)?.to_array()?;
        let mut envs: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
        for p in parts {
            envs.entry(p.job_id).or_default().push_str(&p.data);
        }
        for j in &mut jobs {
            if let Some(e) = envs.get(&j.job_id) {
                j.envelope = serde_json::from_str(e).ok();
            }
        }
        let rows: Vec<RecRow> = sql.exec("SELECT rec FROM workers", None)?.to_array()?;
        let workers: Vec<WorkerRec> = rows.iter().filter_map(|r| serde_json::from_str(&r.rec).ok()).collect();
        let mut s = Sched::restore(pool, self.cfg(), jobs, workers);
        // Sockets survive hibernation, not deploys: a worker recorded as
        // connected without a live socket starts its grace period now.
        let now = now_ms();
        let ids: Vec<String> = s.workers().filter(|w| w.connected).map(|w| w.worker_id.clone()).collect();
        for id in ids {
            if self.state.get_websockets_with_tag(&id).is_empty() {
                s.disconnect(&id, now);
            }
        }
        *self.sched.borrow_mut() = Some(s);
        Ok(())
    }

    fn with<T>(&self, f: impl FnOnce(&mut Sched) -> T) -> Result<T> {
        let mut g = self.sched.borrow_mut();
        let s = g.as_mut().ok_or_else(|| Error::RustError("scheduler not loaded".into()))?;
        Ok(f(s))
    }

    /// Sends the effects, persists what changed, re-arms the alarm, and
    /// queues the D1 record writes.
    async fn apply(&self, out: Vec<Out>) -> Result<()> {
        for o in out {
            match o {
                Out::Send { worker, msg } => {
                    let text = serde_json::to_string(&msg).map_err(|e| Error::RustError(e.to_string()))?;
                    for ws in self.state.get_websockets_with_tag(&worker) {
                        if let Err(e) = ws.send_with_str(&text) {
                            console_warn!("send to {worker} failed: {e:?}");
                        }
                    }
                }
                Out::Close { worker } => {
                    for ws in self.state.get_websockets_with_tag(&worker) {
                        let _ = ws.close(Some(4001), Some("no heartbeat"));
                    }
                }
                Out::PushSpilled { worker, key, msg } => {
                    if let Err(e) = self.push_spilled(&worker, &key, msg).await {
                        // The ack timeout pushes it again.
                        console_warn!("spilled push to {worker} failed: {e:?}");
                    }
                }
            }
        }
        let now = now_ms();
        let (dirty, pool, wake) = self.with(|s| (s.take_dirty(), s.pool.clone(), s.next_wake(now).map(|t| t.max(now + 1))))?;
        self.persist(&dirty)?;
        self.record(&pool, &dirty);
        self.drop_spills(&dirty);
        // Only ever move the alarm earlier: a later deadline is picked up by
        // the tick that runs anyway (it recomputes), so most events (every
        // enqueue moves the next ack deadline) cost no alarm write.
        if let Some(t) = wake {
            let set = self.alarm_at.get();
            if set.is_none_or(|a| t < a || a <= now) {
                self.state.storage().set_alarm(ScheduledTime::new(js_sys::Date::new(&JsValue::from_f64(t as f64)))).await?;
                self.alarm_at.set(Some(t));
            }
        }
        Ok(())
    }

    /// Loads a spilled envelope and sends the push.
    async fn push_spilled(&self, worker: &str, key: &str, mut msg: proto::DoMsg) -> Result<()> {
        let bucket = self.env.bucket(R2_BINDING)?;
        let Some(obj) = bucket.get(key).execute().await? else {
            console_warn!("spilled envelope {key} is missing");
            return Ok(());
        };
        let text = match obj.body() {
            Some(b) => b.text().await?,
            None => return Ok(()),
        };
        if let proto::DoMsg::Job { envelope, .. } = &mut msg {
            *envelope = serde_json::from_str(&text).map_err(|e| Error::RustError(e.to_string()))?;
        }
        let text = serde_json::to_string(&msg).map_err(|e| Error::RustError(e.to_string()))?;
        for ws in self.state.get_websockets_with_tag(worker) {
            if let Err(e) = ws.send_with_str(&text) {
                console_warn!("send to {worker} failed: {e:?}");
            }
        }
        Ok(())
    }

    /// Deletes spilled envelopes no job needs any more (behind the response).
    fn drop_spills(&self, d: &Dirty) {
        if d.dropped_spills.is_empty() {
            return;
        }
        let Ok(bucket) = self.env.bucket(R2_BINDING) else { return };
        let keys = d.dropped_spills.clone();
        self.state.wait_until(async move {
            for k in keys {
                if let Err(e) = bucket.delete(k.as_str()).await {
                    console_warn!("deleting spilled envelope {k}: {e:?}");
                }
            }
        });
    }

    /// Enqueues, spilling a large envelope to R2 when the bucket is bound.
    async fn enqueue(&self, pool: &str, body: EnqueueReq, now: i64) -> Result<(proto::EnqueueResp, Vec<Out>)> {
        let limit = self.env.var("SPILL_BYTES").ok().and_then(|v| v.to_string().parse::<usize>().ok()).unwrap_or(SPILL_BYTES);
        let text = serde_json::to_string(&body.envelope).map_err(|e| Error::RustError(e.to_string()))?;
        if text.len() > limit {
            if let Ok(bucket) = self.env.bucket(R2_BINDING) {
                let key = format!("env/{pool}/{}/{:x}", body.job_id, (js_sys::Math::random() * 1e15) as u64);
                bucket.put(key.as_str(), text).execute().await?;
                let req = EnqueueReq { envelope: serde_json::Value::Null, ..body };
                return self.with(|s| s.enqueue_spilled(req, Some(key), now));
            }
        }
        drop(text);
        self.with(|s| s.enqueue(body, now))
    }

    fn persist(&self, d: &Dirty) -> Result<()> {
        if d.is_empty() {
            return Ok(());
        }
        let sql = self.sql();
        for j in &d.jobs {
            let rec = serde_json::to_string(j).map_err(|e| Error::RustError(e.to_string()))?;
            sql.exec("INSERT OR REPLACE INTO jobs (job_id, rec) VALUES (?, ?)", vec![j.job_id.as_str().into(), rec.into()])?;
        }
        for (id, env) in &d.new_envelopes {
            let text = serde_json::to_string(env).map_err(|e| Error::RustError(e.to_string()))?;
            sql.exec("DELETE FROM envelopes WHERE job_id = ?", vec![id.as_str().into()])?;
            for (i, part) in chunks(&text, CHUNK).into_iter().enumerate() {
                sql.exec("INSERT INTO envelopes (job_id, part, data) VALUES (?, ?, ?)", vec![id.as_str().into(), (i as i64).into(), part.into()])?;
            }
        }
        for id in &d.dropped_envelopes {
            sql.exec("DELETE FROM envelopes WHERE job_id = ?", vec![id.as_str().into()])?;
        }
        for id in &d.removed_jobs {
            sql.exec("DELETE FROM jobs WHERE job_id = ?", vec![id.as_str().into()])?;
        }
        for w in &d.workers {
            let rec = serde_json::to_string(w).map_err(|e| Error::RustError(e.to_string()))?;
            sql.exec("INSERT OR REPLACE INTO workers (worker_id, rec) VALUES (?, ?)", vec![w.worker_id.as_str().into(), rec.into()])?;
        }
        for id in &d.removed_workers {
            sql.exec("DELETE FROM workers WHERE worker_id = ?", vec![id.as_str().into()])?;
        }
        Ok(())
    }

    /// The durable record in D1 (`edge_jobs`), written behind the response.
    fn record(&self, pool: &str, d: &Dirty) {
        if d.jobs.is_empty() {
            return;
        }
        let Ok(db) = self.env.d1(D1_BINDING) else { return };
        let now = now_ms() as f64;
        let opt = |v: Option<i64>| v.map_or(JsValue::NULL, |x| JsValue::from_f64(x as f64));
        let mut stmts = vec![db.prepare(
            "CREATE TABLE IF NOT EXISTS edge_jobs (job_id TEXT PRIMARY KEY, pool TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL, \
             worker TEXT, enqueued_at INTEGER, pushed_at INTEGER, acked_at INTEGER, finished_at INTEGER, error TEXT, updated_at INTEGER NOT NULL)",
        )];
        for j in &d.jobs {
            let args = [
                JsValue::from_str(&j.job_id),
                JsValue::from_str(pool),
                JsValue::from_str(j.phase.as_str()),
                JsValue::from_f64(j.attempt as f64),
                j.worker.as_deref().map_or(JsValue::NULL, JsValue::from_str),
                JsValue::from_f64(j.enqueued_at as f64),
                opt(j.pushed_at),
                opt(j.acked_at),
                opt(j.finished_at),
                j.error.as_deref().map_or(JsValue::NULL, JsValue::from_str),
                JsValue::from_f64(now),
            ];
            match db.prepare("INSERT OR REPLACE INTO edge_jobs (job_id, pool, state, attempt, worker, enqueued_at, pushed_at, acked_at, finished_at, error, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)").bind(&args) {
                Ok(s) => stmts.push(s),
                Err(e) => console_warn!("edge_jobs bind: {e:?}"),
            }
        }
        self.state.wait_until(async move {
            if let Err(e) = db.batch(stmts).await {
                console_warn!("edge_jobs write-behind failed: {e:?}");
            }
        });
    }

    async fn accept_worker(&self, req: Request, pool: &str) -> Result<Response> {
        if !req.headers().get("upgrade")?.is_some_and(|u| u.eq_ignore_ascii_case("websocket")) {
            return json_err(426, "invalid_request", "expected a WebSocket upgrade");
        }
        let Some(worker) = req.headers().get(proto::WORKER_HEADER)?.filter(|w| proto::valid_id(w)) else {
            return json_err(400, "invalid_request", "x-fv-worker-id is missing or invalid");
        };
        let conn = format!("{:x}", (js_sys::Math::random() * 1e15) as u64);
        // A worker reconnecting without closing its old socket: replace it.
        for old in self.state.get_websockets_with_tag(&worker) {
            let _ = old.close(Some(4000), Some("replaced by a new connection"));
        }
        let pair = WebSocketPair::new()?;
        self.state.accept_websocket_with_tags(&pair.server, &[worker.as_str()]);
        pair.server.serialize_attachment(Attach { worker_id: worker.clone(), pool: pool.to_owned(), conn })?;
        Response::from_websocket(pair.client)
    }

    fn attach(ws: &WebSocket) -> Option<Attach> {
        ws.deserialize_attachment::<Attach>().ok().flatten()
    }

    async fn closed(&self, ws: WebSocket) -> Result<()> {
        let Some(a) = Self::attach(&ws) else { return Ok(()) };
        // Another live socket of the same worker (it reconnected): keep it.
        let others = self.state.get_websockets_with_tag(&a.worker_id).iter().filter_map(Self::attach).any(|o| o.conn != a.conn);
        if others {
            return Ok(());
        }
        self.ensure(&a.pool)?;
        let out = self.with(|s| s.disconnect(&a.worker_id, now_ms()))?;
        self.apply(out).await
    }
}

/// Splits `s` into pieces of at most `max` bytes on char boundaries.
fn chunks(s: &str, max: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = s;
    while rest.len() > max {
        let mut cut = max;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push(&rest[..cut]);
        rest = &rest[cut..];
    }
    out.push(rest);
    out
}

impl DurableObject for PoolScheduler {
    fn new(state: State, env: Env) -> Self {
        // Keep-alive pings are answered without waking the object.
        if let Ok(pair) = WebSocketRequestResponsePair::new(proto::PING, proto::PONG) {
            state.set_websocket_auto_response(&pair);
        }
        Self { state, env, sched: RefCell::new(None), alarm_at: Cell::new(None) }
    }

    async fn fetch(&self, mut req: Request) -> Result<Response> {
        let Some((pool, action, arg)) = split(&req.path()) else {
            return json_err(404, "not_found", "no such route");
        };
        let pool = self.pool_name(Some(&pool))?;
        self.ensure(&pool)?;
        let now = now_ms();
        match (req.method(), action.as_str()) {
            (Method::Get, "connect") => self.accept_worker(req, &pool).await,
            (Method::Post, "enqueue") => {
                let body: EnqueueReq = match req.json().await {
                    Ok(b) => b,
                    Err(e) => return json_err(400, "invalid_request", &format!("enqueue body: {e}")),
                };
                if !proto::valid_id(&body.job_id) {
                    return json_err(400, "invalid_request", "invalid job id");
                }
                let (resp, out) = self.enqueue(&pool, body, now).await?;
                self.apply(out).await?;
                Ok(Response::from_json(&resp)?.with_status(202))
            }
            (Method::Post, "cancel") => {
                let Some(job) = arg else { return json_err(404, "not_found", "no job id") };
                let (st, out) = self.with(|s| s.cancel(&job, now))?;
                self.apply(out).await?;
                match st {
                    Some(st) => Response::from_json(&json!({"job_id": job, "state": st})),
                    None => json_err(404, "not_found", "unknown job"),
                }
            }
            (Method::Get, "status") => {
                let mut st = self.with(|s| s.status(now))?;
                st.dispatcher = version(&self.env);
                Response::from_json(&st)
            }
            _ => json_err(405, "invalid_request", "method not allowed"),
        }
    }

    async fn alarm(&self) -> Result<Response> {
        self.alarm_at.set(None);
        // (A spurious early alarm is harmless: the tick recomputes.)
        let pool = self.pool_name(None)?;
        self.ensure(&pool)?;
        let out = self.with(|s| s.tick(now_ms()))?;
        self.apply(out).await?;
        Response::ok("")
    }

    async fn websocket_message(&self, ws: WebSocket, message: WebSocketIncomingMessage) -> Result<()> {
        let WebSocketIncomingMessage::String(text) = message else { return Ok(()) };
        if text == proto::PING {
            return ws.send_with_str(proto::PONG);
        }
        let Some(a) = Self::attach(&ws) else {
            let _ = ws.close(Some(4002), Some("unknown socket"));
            return Ok(());
        };
        let msg: WorkerMsg = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                console_warn!("bad frame from {}: {e}", a.worker_id);
                return Ok(());
            }
        };
        if let WorkerMsg::Hello(h) = &msg {
            if h.worker_id != a.worker_id || h.pool != a.pool {
                let _ = ws.close(Some(4003), Some("hello does not match the connection"));
                return Ok(());
            }
        }
        self.ensure(&a.pool)?;
        let out = self.with(|s| s.on_msg(&a.worker_id, msg, now_ms()))?;
        self.apply(out).await
    }

    async fn websocket_close(&self, ws: WebSocket, _code: usize, _reason: String, _was_clean: bool) -> Result<()> {
        self.closed(ws).await
    }

    async fn websocket_error(&self, ws: WebSocket, _error: Error) -> Result<()> {
        self.closed(ws).await
    }
}
