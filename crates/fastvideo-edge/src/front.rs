//! The public front of the edge Worker (docs/serve/edge-control-plane.md
//! §2): every API request that is not one of the dispatcher's own routes.
//! The routing decisions are `fastvideo_dispatch_proto::front`'s (shared
//! with the native host, `fastvideo_serve::edge_host`); this module does the
//! I/O: the registry snapshot (cached per isolate), key checks against D1
//! (cached, dropped when the registry's key epoch moves), quotas, the
//! forward to a front over its Runpod proxy host, session admission through
//! the family objects and the session bindings in the registry object.
//!
//! Worker bindings and settings: `DB` (the D1 with `api_keys`), `REGISTRY`
//! and `POOL_SCHEDULER` (Durable Objects), secrets `FV_INTERNAL_TOKEN`,
//! `FV_ADMIN_TOKEN`, `FV_API_KEYS` (SHA-256 list, optional); vars
//! `FV_EDGE_AUTH` (`keys` | `none`), `FV_REACTOR_MODEL`, `FV_EDGE_KEY_RPM`,
//! `FV_EDGE_KEY_IN_FLIGHT`, `FV_EDGE_SESSION_TTL_MS`, `FV_EDGE_WHIP`
//! (`proxy` | `redirect`: WHIP ingest offers proxied, or a 307 to the
//! admitted worker with a session capability).

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;

use fastvideo_dispatch_proto::front::{
    self as f, classify, is_submit, plan_forward, reactor_owner, scan_model, Class, DirectorOp, EdgeRoute, Quotas, ReactorOp, Registry, Reply, Scan, SessionBinding, StreamOp, Target,
    Verdict, EDGE_AUTH_HEADER, REQUEST_ID_HEADER,
};
use fastvideo_dispatch_proto::{self as proto, FamilyMetrics, PoolStatus, SessionGrant, SessionReq};
use serde_json::{json, Value};
use wasm_bindgen::JsValue;
use worker::*;

use crate::edge::{ct_eq, family_stub, json_err, now_ms, rust_err, version, D1_BINDING, REGISTRY_BINDING};

/// How long an isolate serves its registry snapshot.
const REGISTRY_TTL_MS: i64 = 2_000;
/// Key cache: valid keys, unknown keys.
const KEY_HIT_MS: i64 = 15_000;
const KEY_MISS_MS: i64 = 5_000;
/// Largest body read whole (a submit whose `model` comes late).
const BODY_MAX: usize = 100 << 20;

/// Request headers never forwarded to a front.
const DROP_REQ: &[&str] = &["host", "connection", "keep-alive", "transfer-encoding", "content-length", "upgrade", "te", "trailer", "authorization"];

#[derive(Default)]
struct Cache {
    registry: Option<(i64, Registry)>,
    /// Key digest → (owner, cached at).
    keys: HashMap<String, (Option<String>, i64)>,
    epoch: u64,
    /// Fixed one-minute windows per key / address (per isolate).
    rate: HashMap<String, (i64, u32)>,
}

thread_local! {
    static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
}

fn secret(env: &Env, name: &str) -> Option<String> {
    env.secret(name).ok().map(|s| s.to_string()).filter(|s| !s.is_empty())
}

fn var(env: &Env, name: &str) -> Option<String> {
    env.var(name).ok().map(|s| s.to_string()).filter(|s| !s.is_empty())
}

fn quotas(env: &Env) -> Quotas {
    let d = Quotas::default();
    let n = |k: &str, v: u32| var(env, k).and_then(|x| x.parse().ok()).unwrap_or(v);
    Quotas { key_rpm: n("FV_EDGE_KEY_RPM", d.key_rpm), key_in_flight: n("FV_EDGE_KEY_IN_FLIGHT", d.key_in_flight), invalid_key_rpm: d.invalid_key_rpm }
}

fn reply(r: Reply) -> Result<Response> {
    let mut resp = Response::from_json(&r.body())?.with_status(r.status);
    if let Some(s) = r.retry_after {
        resp.headers_mut().set("retry-after", &s.to_string())?;
    }
    Ok(resp)
}

fn is_admin(env: &Env, h: &Headers) -> bool {
    let Some(want) = secret(env, "FV_ADMIN_TOKEN") else { return false };
    let got = h.get("authorization").ok().flatten().and_then(|v| v.strip_prefix("Bearer ").map(str::to_owned));
    got.is_some_and(|g| ct_eq(g.as_bytes(), want.as_bytes()))
}

fn is_internal(env: &Env, h: &Headers) -> bool {
    let Some(want) = secret(env, "FV_INTERNAL_TOKEN") else { return false };
    h.get(proto::TOKEN_HEADER).ok().flatten().is_some_and(|g| ct_eq(g.as_bytes(), want.as_bytes()))
}

fn client_addr(h: &Headers) -> String {
    h.get("cf-connecting-ip").ok().flatten().unwrap_or_else(|| "unknown".into())
}

/// A request to a Durable Object or a front.
fn request(url: &str, method: Method, headers: Headers, body: Option<JsValue>) -> Result<Request> {
    let mut init = RequestInit::new();
    init.with_method(method).with_headers(headers).with_body(body).with_redirect(RequestRedirect::Manual);
    Request::new_with_init(url, &init)
}

fn token_headers(env: &Env) -> Result<Headers> {
    let h = Headers::new();
    h.set(proto::TOKEN_HEADER, &secret(env, "FV_INTERNAL_TOKEN").unwrap_or_default())?;
    h.set("content-type", "application/json")?;
    Ok(h)
}

pub(crate) async fn registry_call(env: &Env, method: Method, path: &str, body: Option<String>) -> Result<Response> {
    let stub = env.durable_object(REGISTRY_BINDING)?.get_by_name("registry")?;
    let req = request(&format!("https://registry{path}"), method, token_headers(env)?, body.map(|b| JsValue::from_str(&b)))?;
    stub.fetch_with_request(req).await
}

/// The registry snapshot (cached per isolate); a moved key epoch drops the
/// key cache.
async fn registry(env: &Env) -> Result<Registry> {
    let now = now_ms();
    if let Some(r) = CACHE.with(|c| c.borrow().registry.as_ref().filter(|(at, _)| now - at < REGISTRY_TTL_MS).map(|(_, r)| r.clone())) {
        return Ok(r);
    }
    let mut resp = registry_call(env, Method::Get, "/registry", None).await?;
    let r: Registry = resp.json().await?;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.epoch != r.key_epoch {
            c.keys.clear();
            c.epoch = r.key_epoch;
        }
        c.registry = Some((now, r.clone()));
    });
    Ok(r)
}

/// Drops this isolate's key cache and registry snapshot (a revoke or an
/// invalidate here; other isolates follow the registry's key epoch).
pub(crate) fn drop_key_cache() {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.keys.clear();
        c.registry = None;
    });
}

/// The owner of a key: `FV_API_KEYS`, else the D1 `api_keys` table (cached).
async fn key_owner(env: &Env, key: &str) -> Option<String> {
    let digest = f::key_digest(key);
    if let Some(list) = secret(env, "FV_API_KEYS") {
        if list.split([',', ' ', '\n', '\t']).any(|d| ct_eq(d.trim().as_bytes(), digest.as_bytes())) {
            return Some(f::key_id(&digest));
        }
    }
    let now = now_ms();
    let hit = CACHE.with(|c| {
        c.borrow().keys.get(&digest).and_then(|(o, at)| {
            let ttl = if o.is_some() { KEY_HIT_MS } else { KEY_MISS_MS };
            (now - at < ttl).then(|| o.clone())
        })
    });
    if let Some(o) = hit {
        return o;
    }
    let db = env.d1(D1_BINDING).ok()?;
    let row: Option<Value> = match db.prepare("SELECT id FROM api_keys WHERE digest = ? AND revoked_at IS NULL").bind(&[JsValue::from_str(&digest)]) {
        Ok(s) => s.first(None).await.ok().flatten(),
        Err(_) => None,
    };
    let owner = row.and_then(|r| r.get("id").and_then(Value::as_str).map(str::to_owned));
    CACHE.with(|c| {
        c.borrow_mut().keys.insert(digest, (owner.clone(), now));
    });
    owner
}

async fn verdict(env: &Env, h: &Headers) -> Verdict {
    let mut v = Verdict { v: 1, ..Verdict::default() };
    if var(env, "FV_EDGE_AUTH").as_deref() == Some("none") {
        return v;
    }
    let Some(auth) = h.get("authorization").ok().flatten() else { return v };
    let Some((scheme, key)) = f::parse_authorization(&auth) else { return v };
    v.presented = true;
    v.scheme = Some(scheme.to_owned());
    if secret(env, "FV_ADMIN_TOKEN").is_some_and(|a| ct_eq(a.as_bytes(), key.as_bytes())) {
        return v;
    }
    if let Some(o) = key_owner(env, key).await {
        v.valid = true;
        v.key = Some(o);
    }
    v
}

fn over(who: &str, limit: u32) -> bool {
    if limit == 0 {
        return false;
    }
    let win = now_ms() / 60_000;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let e = c.rate.entry(who.to_owned()).or_insert((win, 0));
        if e.0 != win {
            *e = (win, 0);
        }
        e.1 += 1;
        e.1 > limit
    })
}

/// Forwards to a front (`url` its base) with the internal token and the
/// verdict; the response streams back.
async fn forward(env: &Env, url: &str, method: Method, path_q: &str, h: &Headers, body: Option<JsValue>, verdict: &Verdict) -> Result<Response> {
    let out = Headers::new();
    for (k, v) in h.entries() {
        // Client `x-fv-*` headers never pass (the verdict is the edge's),
        // except the tracing opt-in (docs/serve/tracing.md).
        if DROP_REQ.contains(&k.as_str()) || (k.starts_with("x-fv-") && k != proto::front::TRACE_OPT_IN_HEADER) {
            continue;
        }
        out.append(&k, &v)?;
    }
    // The admin token goes on (the front checks it for its admin routes).
    if is_admin(env, h) {
        if let Some(a) = h.get("authorization")? {
            out.set("authorization", &a)?;
        }
    }
    out.set(proto::TOKEN_HEADER, &secret(env, "FV_INTERNAL_TOKEN").unwrap_or_default())?;
    out.set(EDGE_AUTH_HEADER, &verdict.header())?;
    out.set("x-forwarded-for", &client_addr(h))?;
    let req = request(&format!("{}{path_q}", url.trim_end_matches('/')), method, out, body)?;
    match Fetch::Request(req).send().await {
        Ok(r) => Ok(r),
        Err(e) => json_err(502, "loading", &format!("the worker did not answer: {e}")),
    }
}

async fn forward_json(env: &Env, url: &str, path_q: &str, h: &Headers, body: Vec<u8>, verdict: &Verdict) -> Result<(u16, Value, Vec<u8>, Headers)> {
    let mut r = forward(env, url, Method::Post, path_q, h, Some(js_sys::Uint8Array::from(&body[..]).into()), verdict).await?;
    let status = r.status_code();
    let hs = r.headers().clone();
    let bytes = r.bytes().await.unwrap_or_default();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Ok((status, v, bytes, hs))
}

fn bytes_body(b: &[u8]) -> Option<JsValue> {
    Some(js_sys::Uint8Array::from(b).into())
}

/// The protocol-shaped error of a session route the edge refuses itself.
fn session_error(protocol: &str, status: u16, message: &str) -> Result<Response> {
    let body = match protocol {
        "fal_director" => json!({"error": message}),
        "reactor" => json!({"detail": message}),
        _ => json!({"error": {"kind": if status == 429 { "queue_full" } else { "loading" }, "message": message}}),
    };
    let mut r = Response::from_json(&body)?.with_status(status);
    r.headers_mut().set("retry-after", "10")?;
    Ok(r)
}

/// The Worker's public front (see the module docs).
pub async fn handle(mut req: Request, env: Env, ctx: Context) -> Result<Response> {
    let url = req.url()?;
    let mut path = url.path().to_owned();
    let mut query = url.query().unwrap_or("").to_owned();
    let mut path_q = if query.is_empty() { path.clone() } else { format!("{path}?{query}") };
    let method = req.method();
    let h = req.headers().clone();
    if h.get(REQUEST_ID_HEADER)?.is_none() {
        h.set(REQUEST_ID_HEADER, &format!("req_{:x}{:x}", now_ms(), (js_sys::Math::random() * 1e12) as u64))?;
    }
    let mut class = classify(method.as_ref(), &path);
    // fal's proxy: route the request it stands for.
    if class.target == Target::FalProxy {
        if let Some(inner) = h.get("x-fal-target-url")?.and_then(|t| f::unproxy(&t)) {
            let (p, q) = inner.split_once('?').map(|(p, q)| (p.to_owned(), q.to_owned())).unwrap_or((inner.clone(), String::new()));
            class = classify(method.as_ref(), &p);
            if class.target == Target::FalProxy {
                return json_err(400, "invalid_request", "a proxied request cannot target the proxy");
            }
            h.delete("x-fal-target-url")?;
            (path, query, path_q) = (p, q, inner);
        }
    }
    let t0 = now_ms();
    // docs/serve/tracing.md: an opted-in request's edge steps.
    let tr = crate::trace::EdgeTrace::from_request(&env, &h, &query);
    let resp = route(&mut req, &env, &ctx, &class, method.clone(), &path, &query, &path_q, &h, tr.as_ref()).await;
    console_log!(
        "{}",
        json!({"edge": "request", "method": method.as_ref(), "path": path, "protocol": class.protocol, "status": resp.as_ref().map(|r| r.status_code()).unwrap_or(500), "ms": now_ms() - t0})
    );
    match (tr, resp) {
        (Some(tr), Ok(r)) => tr.finish(r, t0, &env, &ctx),
        (_, resp) => resp,
    }
}

#[allow(clippy::too_many_arguments)]
async fn route(
    req: &mut Request,
    env: &Env,
    ctx: &Context,
    class: &Class,
    method: Method,
    path: &str,
    query: &str,
    path_q: &str,
    h: &Headers,
    tr: Option<&crate::trace::EdgeTrace>,
) -> Result<Response> {
    if let Target::Edge(r) = &class.target {
        return edge_route(req, env, r, method, path, h).await;
    }
    if class.target == Target::NotFound {
        return json_err(404, "not_found", "no such route");
    }
    if class.protocol == "internal" && !is_internal(env, h) {
        return json_err(401, "unauthorized", "a valid token is required");
    }
    let t_auth = now_ms();
    let mut v = verdict(env, h).await;
    if let Some(tr) = tr {
        tr.span("auth", t_auth);
    }
    let q = quotas(env);
    if v.presented && !v.valid && over(&format!("bad:{}", client_addr(h)), q.invalid_key_rpm) {
        return reply(Reply { status: 429, kind: "rate_limited", message: "too many requests with an invalid key".into(), retry_after: Some(60) });
    }
    let t_reg = now_ms();
    let reg = registry(env).await?;
    if let Some(tr) = tr {
        tr.span("registry", t_reg);
    }
    if let Some(k) = v.key.clone().filter(|_| is_submit(method.as_ref(), class)) {
        if over(&k, q.key_rpm) {
            v.deny = Some(f::Deny { kind: "rate_limited".into(), message: "this key's submit rate limit is reached".into(), retry_after: Some(10) });
        } else if q.key_in_flight > 0 {
            let n = reg.owners.get(&k).copied().unwrap_or(0);
            if n >= q.key_in_flight {
                v.deny = Some(f::Deny { kind: "rate_limited".into(), message: format!("this key has {n} unfinished jobs (limit {})", q.key_in_flight), retry_after: Some(10) });
            }
        }
    }
    match &class.target {
        Target::Director(op) => return director(req, env, ctx, &reg, *op, method, path_q, h, &v).await,
        Target::Reactor(op) => return reactor(req, env, &reg, op, method, path_q, h, &v).await,
        Target::Stream { ingest, op } => return stream(req, env, &reg, *ingest, op, method, path_q, h, &v).await,
        _ => {}
    }
    let t_body = now_ms();
    let (body, model) = if class.target == Target::Body {
        let b = req.bytes().await?;
        if let Some(tr) = tr {
            tr.span("body_read", t_body);
        }
        if b.len() > BODY_MAX {
            return json_err(413, "payload_too_large", "the body is too large");
        }
        let ct = h.get("content-type")?.unwrap_or_default();
        let m = match scan_model(&ct, &b, true) {
            Scan::Found(m) => Some(m),
            _ => None,
        };
        (bytes_body(&b), m)
    } else {
        (req.inner().body().map(JsValue::from), None)
    };
    let fwd = match plan_forward(class, &reg, path, query, model.as_deref(), None) {
        Ok(x) => x,
        Err(r) => return reply(r),
    };
    if v.deny.is_none() {
        v.deny = fwd.deny.clone();
    }
    let body = if matches!(method, Method::Get | Method::Head) { None } else { body };
    let t_fwd = now_ms();
    let r = forward(env, &fwd.url, method, path_q, h, body, &v).await;
    if let (Some(tr), Ok(resp)) = (tr, r.as_ref()) {
        tr.forwarded(&fwd.url, t_fwd, resp);
    }
    r
}

/// Every family object's status and metrics (registry families).
async fn families(env: &Env, reg: &Registry) -> (serde_json::Map<String, Value>, Vec<FamilyMetrics>) {
    let mut st = serde_json::Map::new();
    let mut ms = Vec::new();
    for fam in reg.families.keys() {
        let Ok(stub) = family_stub(env, &format!("family:{fam}")) else { continue };
        for what in ["status", "metrics"] {
            let Ok(h) = token_headers(env) else { continue };
            let Ok(r) = request(&format!("https://do/families/{fam}/{what}"), Method::Get, h, None) else { continue };
            if let Ok(mut resp) = stub.fetch_with_request(r).await {
                if what == "status" {
                    if let Ok(s) = resp.json::<PoolStatus>().await {
                        st.insert(fam.clone(), serde_json::to_value(s).unwrap_or(Value::Null));
                    }
                } else if let Ok(m) = resp.json::<FamilyMetrics>().await {
                    ms.push(m);
                }
            }
        }
    }
    (st, ms)
}

/// One front per family, `GET path` with the caller's verdict; the JSON
/// answers.
async fn fan_out(env: &Env, fronts: Vec<String>, path: &str, h: &Headers, v: &Verdict) -> Vec<Value> {
    let calls = fronts.into_iter().map(|u| async move {
        let mut r = forward(env, &u, Method::Get, path, h, None, v).await.ok()?;
        if r.status_code() / 100 != 2 {
            return None;
        }
        r.json::<Value>().await.ok()
    });
    futures::future::join_all(calls).await.into_iter().flatten().collect()
}

async fn edge_route(req: &mut Request, env: &Env, r: &EdgeRoute, method: Method, path: &str, h: &Headers) -> Result<Response> {
    let reg = registry(env).await?;
    let ready = reg.any_ready();
    let v = verdict(env, h).await;
    let ver = json!({"sha": version(env), "channel": null});
    match r {
        EdgeRoute::Root => match reg.fronts().find(|x| x.info.defaults.contains_key("fastwan")) {
            Some(x) => forward(env, &x.info.url, method, path, h, None, &v).await,
            None => Response::from_json(&json!({"server": "fv-edge", "role": "edge", "version": version(env), "ready": ready})),
        },
        EdgeRoute::Ping => Ok(Response::from_json(&json!({"status": if ready { "healthy" } else { "unavailable" }}))?.with_status(if ready { 200 } else { 503 })),
        EdgeRoute::Health => {
            let body = json!({"status": if ready { "ok" } else { "unavailable" }, "model_loaded": ready, "state": if ready { "AVAILABLE" } else { "UNAVAILABLE" }, "edge": true, "version": version(env)});
            Ok(Response::from_json(&body)?.with_status(if ready { 200 } else { 503 }))
        }
        EdgeRoute::Healthz => {
            let st = f::status_body(&reg, ver, now_ms());
            let body = json!({"state": if ready { "ready" } else { "unavailable" }, "edge": true, "pools": st["pools"], "version": version(env)});
            Ok(Response::from_json(&body)?.with_status(if ready { 200 } else { 503 }))
        }
        EdgeRoute::Status => Response::from_json(&f::status_body(&reg, ver, now_ms())),
        EdgeRoute::Capabilities => {
            let fronts = reg.one_per_family("native").into_iter().map(|x| x.info.url.clone()).collect();
            let bodies = fan_out(env, fronts, "/fv/v1/capabilities", h, &v).await;
            let none = var(env, "FV_EDGE_AUTH").as_deref() == Some("none");
            if bodies.is_empty() && !v.valid && !none {
                return json_err(401, "unauthorized", if v.presented { "invalid credentials" } else { "missing credentials" });
            }
            let mut body = f::merge_capabilities(&bodies, &reg, now_ms());
            if let Some(a) = body.get_mut("auth").and_then(Value::as_object_mut) {
                a.insert("mode".into(), json!(if none { "none" } else { "keys" }));
            }
            Response::from_json(&body)
        }
        EdgeRoute::Models => {
            let fronts = reg.one_per_family("openai_videos").into_iter().map(|x| x.info.url.clone()).collect();
            let bodies = fan_out(env, fronts, "/v1/models", h, &v).await;
            Response::from_json(&json!({"object": "list", "data": f::merge_list(&bodies, "data", "/id")}))
        }
        EdgeRoute::Model(m) => match reg.family_of_name(m).and_then(|fam| reg.pick(Some(fam), "openai_videos", None, None)).or_else(|| reg.pick(None, "openai_videos", None, None)) {
            Some(x) => forward(env, &x.info.url, method, path, h, None, &v).await,
            None => json_err(503, "loading", "no worker serves this API right now"),
        },
        EdgeRoute::FalSchema => {
            let fronts = reg.one_per_family("fal").into_iter().map(|x| x.info.url.clone()).collect();
            let bodies = fan_out(env, fronts, "/fal/schema", h, &v).await;
            Response::from_json(&f::merge_fal_schema(&bodies))
        }
        EdgeRoute::Keys | EdgeRoute::KeyRevoke(_) | EdgeRoute::KeysInvalidate => {
            if !is_admin(env, h) {
                return json_err(401, "unauthorized", "the admin token is required");
            }
            crate::keys::admin(req, env, r, method).await
        }
        EdgeRoute::Families => {
            if !is_admin(env, h) {
                return json_err(401, "unauthorized", "the admin token is required");
            }
            let (st, ms) = families(env, &reg).await;
            let metrics: serde_json::Map<String, Value> = ms.into_iter().map(|m| (m.family.clone(), serde_json::to_value(m).unwrap_or(Value::Null))).collect();
            Response::from_json(&json!({"object": "fv.edge.families", "families": st, "metrics": metrics, "key_epoch": reg.key_epoch}))
        }
        EdgeRoute::Metrics => {
            if !is_admin(env, h) {
                return json_err(401, "unauthorized", "the admin token is required");
            }
            let (_, ms) = families(env, &reg).await;
            let mut resp = Response::ok(f::prometheus(&ms))?;
            resp.headers_mut().set("content-type", "text/plain; version=0.0.4")?;
            Ok(resp)
        }
        EdgeRoute::Internal => json_err(404, "not_found", "no such route"),
        EdgeRoute::Moved(why) => json_err(404, "not_found", why),
    }
}

// ------------------------------------------------------------- sessions

async fn admit(env: &Env, family: &str, kind: &str, model: Option<String>, owner: Option<String>) -> std::result::Result<SessionGrant, (u16, String)> {
    let ttl = var(env, "FV_EDGE_SESSION_TTL_MS").and_then(|v| v.parse().ok()).unwrap_or(1_800_000);
    let body = SessionReq { session_id: None, model, kind: kind.to_owned(), owner, ttl_ms: ttl };
    let call = async {
        let stub = family_stub(env, &format!("family:{family}"))?;
        let r = request(&format!("https://do/families/{family}/sessions"), Method::Post, token_headers(env)?, Some(JsValue::from_str(&serde_json::to_string(&body).map_err(rust_err)?)))?;
        stub.fetch_with_request(r).await
    };
    match call.await {
        Ok(mut r) if r.status_code() == 200 => r.json::<SessionGrant>().await.map_err(|e| (503, e.to_string())),
        Ok(mut r) => {
            let st = r.status_code();
            let v: Value = r.json().await.unwrap_or(Value::Null);
            Err((st, v.pointer("/error/message").and_then(Value::as_str).unwrap_or("no worker has a free session slot").to_owned()))
        }
        Err(e) => Err((503, e.to_string())),
    }
}

async fn session_op(env: &Env, b: &SessionBinding, op: &str) -> bool {
    let call = async {
        let stub = family_stub(env, &format!("family:{}", b.family))?;
        let r = request(&format!("https://do/families/{}/sessions/{}/{op}", b.family, b.session_id), Method::Post, token_headers(env)?, None)?;
        stub.fetch_with_request(r).await
    };
    matches!(call.await, Ok(r) if r.status_code() == 200)
}

async fn bind(env: &Env, alias: &str, b: &SessionBinding) {
    if let Ok(body) = serde_json::to_string(b) {
        let _ = registry_call(env, Method::Put, &format!("/registry/bind/{alias}"), Some(body)).await;
    }
}

async fn binding(env: &Env, alias: &str) -> Option<SessionBinding> {
    let mut r = registry_call(env, Method::Get, &format!("/registry/bind/{alias}"), None).await.ok()?;
    if r.status_code() != 200 {
        return None;
    }
    r.json().await.ok()
}

async fn unbind(env: &Env, alias: &str) {
    let _ = registry_call(env, Method::Delete, &format!("/registry/bind/{alias}"), None).await;
}

/// Director sessions end over their data channel: before refusing a
/// session, release the ended ones of `family` (asking their workers).
async fn reclaim(env: &Env, family: &str) -> bool {
    let Ok(mut r) = registry_call(env, Method::Get, &format!("/registry/binds/{family}/director:"), None).await else { return false };
    let held: HashMap<String, SessionBinding> = r.json().await.unwrap_or_default();
    let mut freed = false;
    for (alias, b) in held {
        let sid = alias.trim_start_matches("director:");
        let v = Verdict { v: 1, key: b.owner.clone(), presented: true, valid: b.owner.is_some(), scheme: Some("key".into()), deny: None };
        let Ok(hs) = token_headers(env) else { continue };
        let _ = hs.set(EDGE_AUTH_HEADER, &v.header());
        let alive = match request(&format!("{}/wma/session/heartbeat", b.endpoint.trim_end_matches('/')), Method::Post, hs, Some(JsValue::from_str(&json!({"session_id": sid}).to_string()))) {
            Ok(rq) => match Fetch::Request(rq).send().await {
                Ok(mut resp) => resp.json::<Value>().await.ok().and_then(|x| x.get("alive").and_then(Value::as_bool)).unwrap_or(true),
                Err(_) => false,
            },
            Err(_) => true,
        };
        if !alive {
            unbind(env, &alias).await;
            session_op(env, &b, "release").await;
            freed = true;
        }
    }
    freed
}

async fn admit_or_reclaim(env: &Env, family: &str, kind: &str, model: Option<String>, owner: Option<String>) -> std::result::Result<SessionGrant, (u16, String)> {
    match admit(env, family, kind, model.clone(), owner.clone()).await {
        Err((429, m)) => {
            if !reclaim(env, family).await {
                return Err((429, m));
            }
            let mut last = Err((429, m));
            for _ in 0..25 {
                Delay::from(Duration::from_millis(200)).await;
                last = admit(env, family, kind, model.clone(), owner.clone()).await;
                if !matches!(last, Err((429, _))) {
                    break;
                }
            }
            last
        }
        r => r,
    }
}

fn binding_of(family: &str, g: &SessionGrant, kind: &str, owner: Option<String>) -> SessionBinding {
    SessionBinding { family: family.to_owned(), session_id: g.session_id.clone(), lease: g.lease, endpoint: g.endpoint.clone(), kind: kind.into(), owner, expires_ms: g.expires_ms }
}

fn respond(status: u16, bytes: Vec<u8>, hs: &Headers) -> Result<Response> {
    let mut r = Response::from_bytes(bytes)?.with_status(status);
    for (k, v) in hs.entries() {
        if !matches!(k.as_str(), "content-length" | "transfer-encoding" | "connection") {
            r.headers_mut().set(&k, &v)?;
        }
    }
    Ok(r)
}

#[allow(clippy::too_many_arguments)]
async fn director(req: &mut Request, env: &Env, ctx: &Context, reg: &Registry, op: DirectorOp, method: Method, path_q: &str, h: &Headers, v: &Verdict) -> Result<Response> {
    let raw = req.bytes().await.unwrap_or_default();
    let body: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    let app_family = body.get("app_id").and_then(Value::as_str).and_then(|a| reg.fal_app(&format!("/{}", a.trim_matches('/'))).map(|(_, fam)| fam.to_owned()));
    let Some(family) = app_family.or_else(|| reg.pick(None, "fal_director", None, None).map(|x| x.family.to_owned())) else {
        return session_error("fal_director", 503, "no worker serves the director right now");
    };
    match op {
        DirectorOp::Ice | DirectorOp::Info => match reg.pick(Some(&family), "fal_director", None, None) {
            Some(x) => {
                let body = bytes_body(&raw).filter(|_| method != Method::Get);
                forward(env, &x.info.url, method, path_q, h, body, v).await
            }
            None => session_error("fal_director", 503, "no worker serves the director right now"),
        },
        DirectorOp::Session => {
            let g = match admit_or_reclaim(env, &family, "director", None, v.key.clone()).await {
                Ok(g) => g,
                Err((st, m)) => return session_error("fal_director", st, &m),
            };
            let b = binding_of(&family, &g, "director", v.key.clone());
            let (status, out, bytes, hs) = match forward_json(env, &g.endpoint, path_q, h, raw, v).await {
                Ok(x) => x,
                Err(e) => {
                    session_op(env, &b, "release").await;
                    return Err(e);
                }
            };
            match out.get("session_id").and_then(Value::as_str).filter(|_| (200..300).contains(&status)) {
                Some(sid) => bind(env, &format!("director:{sid}"), &b).await,
                None => {
                    session_op(env, &b, "release").await;
                }
            }
            respond(status, bytes, &hs)
        }
        DirectorOp::Heartbeat => {
            let sid = body.get("session_id").and_then(Value::as_str).unwrap_or_default().to_owned();
            let Some(b) = binding(env, &format!("director:{sid}")).await else {
                return Response::from_json(&json!({"alive": false}));
            };
            let (status, out, bytes, hs) = forward_json(env, &b.endpoint, path_q, h, raw, v).await?;
            if out.get("alive") == Some(&Value::Bool(false)) {
                unbind(env, &format!("director:{sid}")).await;
                session_op(env, &b, "release").await;
            } else {
                session_op(env, &b, "renew").await;
            }
            respond(status, bytes, &hs)
        }
        DirectorOp::Start => {
            let g = match admit_or_reclaim(env, &family, "director", None, v.key.clone()).await {
                Ok(g) => g,
                Err((st, m)) => return session_error("fal_director", st, &m),
            };
            let b = binding_of(&family, &g, "director", v.key.clone());
            let resp = forward(env, &g.endpoint, method, path_q, h, bytes_body(&raw), v).await?;
            if resp.status_code() / 100 != 2 {
                session_op(env, &b, "release").await;
                return Ok(resp);
            }
            // SSE for the session's life: release when the stream ends.
            let ResponseBody::Stream(stream) = resp.body() else { return Ok(resp) };
            let stream = stream.clone();
            let ts = web_sys::TransformStream::new()?;
            let readable = ts.readable();
            let pipe = worker::wasm_bindgen_futures::JsFuture::from(stream_into(stream, ts.writable()));
            let env2 = env.clone();
            ctx.wait_until(async move {
                let _ = pipe.await;
                session_op(&env2, &b, "release").await;
            });
            let out = Response::builder().with_status(resp.status_code()).with_headers(resp.headers().clone()).stream(readable);
            Ok(out)
        }
    }
}

/// Pipes a worker byte stream into a writable (the promise of the pipe).
fn stream_into(rs: web_sys::ReadableStream, w: web_sys::WritableStream) -> js_sys::Promise {
    rs.pipe_to(&w)
}

#[allow(clippy::too_many_arguments)]
async fn reactor(req: &mut Request, env: &Env, reg: &Registry, op: &ReactorOp, method: Method, path_q: &str, h: &Headers, v: &Verdict) -> Result<Response> {
    let owner = reactor_owner(v.key.as_deref().filter(|_| v.valid), Some(&client_addr(h)));
    let raw = if matches!(method, Method::Get | Method::Head) { Vec::new() } else { req.bytes().await.unwrap_or_default() };
    let body = || (!raw.is_empty()).then(|| bytes_body(&raw)).flatten();
    let pref = var(env, "FV_REACTOR_MODEL");
    let Some(family) = reg.reactor_family(pref.as_deref()).map(str::to_owned) else {
        return session_error("reactor", 503, "no worker serves the Reactor runtime right now");
    };
    let held = binding(env, &owner).await;
    match op {
        ReactorOp::Start => {
            if let Some(b) = held {
                return forward(env, &b.endpoint, method, path_q, h, body(), v).await;
            }
            let g = match admit_or_reclaim(env, &family, "reactor", None, v.key.clone()).await {
                Ok(g) => g,
                Err((st, m)) => return session_error("reactor", if st == 429 { 503 } else { st }, &m),
            };
            let b = binding_of(&family, &g, "reactor", v.key.clone());
            let resp = forward(env, &g.endpoint, method, path_q, h, body(), v).await?;
            if resp.status_code() / 100 == 2 {
                bind(env, &owner, &b).await;
            } else {
                session_op(env, &b, "release").await;
            }
            Ok(resp)
        }
        ReactorOp::Stop => {
            let Some(b) = held else { return session_error("reactor", 404, "no active session") };
            let resp = forward(env, &b.endpoint, method, path_q, h, body(), v).await?;
            if resp.status_code() / 100 == 2 {
                unbind(env, &owner).await;
                session_op(env, &b, "release").await;
            }
            Ok(resp)
        }
        ReactorOp::Follow | ReactorOp::Sid(_) => {
            let target = match held {
                Some(b) => {
                    session_op(env, &b, "renew").await;
                    b.endpoint
                }
                None => match reg.pick(Some(&family), "reactor", None, None) {
                    Some(x) => x.info.url.clone(),
                    None => return session_error("reactor", 503, "no worker serves the Reactor runtime right now"),
                },
            };
            forward(env, &target, method, path_q, h, body(), v).await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn stream(req: &mut Request, env: &Env, reg: &Registry, ingest: bool, op: &StreamOp, method: Method, path_q: &str, h: &Headers, v: &Verdict) -> Result<Response> {
    let kind = if ingest { "ingest" } else { "stream" };
    let raw = if matches!(method, Method::Get | Method::Head) { Vec::new() } else { req.bytes().await.unwrap_or_default() };
    let body = || (!raw.is_empty()).then(|| bytes_body(&raw)).flatten();
    match op {
        StreamOp::Create => {
            let model = if ingest {
                path_q.split_once('?').and_then(|(_, q)| f::query_param(q, "model"))
            } else {
                serde_json::from_slice::<Value>(&raw).ok().and_then(|x| x.get("model").and_then(Value::as_str).map(str::to_owned))
            };
            let Some(family) = model.as_deref().and_then(|m| reg.family_of_name(m)).or_else(|| reg.default_family("native")).map(str::to_owned) else {
                return session_error("native", 503, "no worker serves this model right now");
            };
            let g = match admit_or_reclaim(env, &family, kind, model.clone(), v.key.clone()).await {
                Ok(g) => g,
                Err((st, m)) => return session_error("native", st, &m),
            };
            let b = binding_of(&family, &g, kind, v.key.clone());
            if ingest && var(env, "FV_EDGE_WHIP").as_deref() == Some("redirect") {
                // The 307 hand-off: the client re-sends its offer to the
                // worker with a capability; the lease runs out on its TTL
                // (or when the worker ends the session).
                let key = secret(env, "FV_INTERNAL_TOKEN").unwrap_or_default();
                let cap = f::sign_cap(&key, &f::SessionCap { verdict: v.clone(), exp_ms: now_ms() + f::CAP_TTL_MS });
                let to = f::with_param(&format!("{}{path_q}", g.endpoint.trim_end_matches('/')), f::CAP_PARAM, &cap);
                let hs = Headers::new();
                hs.set("location", &to)?;
                return Ok(Response::empty()?.with_status(307).with_headers(hs));
            }
            let mut resp = forward(env, &g.endpoint, method, path_q, h, body(), v).await?;
            if resp.status_code() / 100 != 2 {
                session_op(env, &b, "release").await;
                return Ok(resp);
            }
            let status = resp.status_code();
            let hs = resp.headers().clone();
            let bytes = resp.bytes().await.unwrap_or_default();
            let id = if ingest {
                hs.get("location")?.and_then(|l| l.rsplit('/').next().map(str::to_owned))
            } else {
                serde_json::from_slice::<Value>(&bytes).ok().and_then(|x| x.get("id").and_then(Value::as_str).map(str::to_owned))
            };
            match id {
                Some(id) => bind(env, &format!("{kind}:{id}"), &b).await,
                None => {
                    session_op(env, &b, "release").await;
                }
            }
            // A Location on the worker's host points back at the edge.
            if let Some(l) = hs.get("location")? {
                if let Some(rest) = l.strip_prefix(g.endpoint.trim_end_matches('/')) {
                    hs.set("location", rest)?;
                }
            }
            respond(status, bytes, &hs)
        }
        StreamOp::List => {
            let fronts = reg.distinct("native").into_iter().map(|x| x.info.url.clone()).collect();
            let p = if ingest { "/fv/v1/streams/ingest" } else { "/fv/v1/streams" };
            let bodies = fan_out(env, fronts, p, h, v).await;
            Response::from_json(&json!({"object": "list", "data": f::merge_list(&bodies, "data", "/id")}))
        }
        StreamOp::Follow(id) | StreamOp::Delete(id) => {
            let alias = format!("{kind}:{id}");
            let Some(b) = binding(env, &alias).await.filter(|b| b.owner.is_none() || b.owner == v.key) else {
                return session_error("native", 404, &format!("stream `{id}` was not found"));
            };
            let del = matches!(op, StreamOp::Delete(_));
            if !del {
                session_op(env, &b, "renew").await;
            }
            let resp = forward(env, &b.endpoint, method, path_q, h, body(), v).await?;
            if del && resp.status_code() / 100 == 2 {
                unbind(env, &alias).await;
                session_op(env, &b, "release").await;
            }
            Ok(resp)
        }
    }
}
