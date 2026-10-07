//! Request tracing at the edge (docs/serve/tracing.md).
//!
//! A request that opts in (`x-fv-trace: 1` or `?fv_trace=1`; the Worker
//! var `FV_TRACE=off` disables it) gets its edge steps collected in memory
//! (receive, key check, registry, body read, the forward to the front with
//! the front's clock sample). The response carries
//! `x-fv-edge-t: <recv ms>;<send ms>` (the edge's clock, for the client's
//! alignment) and `server-timing`; the events go to the front that served
//! the request (`POST /fv/v1/traces/{id}/events`) in `ctx.wait_until`,
//! after the response is sent.
//!
//! The Workers clock (`Date.now()`) only advances across I/O, so edge spans
//! measure the I/O they wait on (D1, the registry object, the front) with
//! 1 ms resolution; CPU time inside the isolate reads as 0.

use std::cell::RefCell;

use fastvideo_dispatch_proto as proto;
use serde_json::{json, Value};
use worker::*;

use crate::edge::now_ms;

pub(crate) struct EdgeTrace {
    /// 32 hex.
    pub id: String,
    /// This hop's span (16 hex), the parent of the front's.
    span: String,
    events: RefCell<Vec<Value>>,
    /// The front the request went to.
    front: RefCell<Option<String>>,
}

fn truthy(v: &str) -> bool {
    matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes")
}

fn hex(n: usize) -> String {
    (0..n).map(|_| format!("{:x}", (js_sys::Math::random() * 16.0) as u8 & 15)).collect()
}

impl EdgeTrace {
    /// `None` unless the request opted in and the Worker allows tracing.
    pub fn from_request(env: &Env, h: &Headers, query: &str) -> Option<Self> {
        let mode = env.var("FV_TRACE").map(|v| v.to_string()).unwrap_or_default().to_ascii_lowercase();
        if matches!(mode.as_str(), "off" | "0" | "false" | "no" | "none") {
            return None;
        }
        let opted = h.get("x-fv-trace").ok().flatten().is_some_and(|v| truthy(&v))
            || query.split('&').any(|kv| matches!(kv, "fv_trace=1" | "fv_trace=true"));
        if !opted && !matches!(mode.as_str(), "all" | "always") {
            return None;
        }
        let id = h
            .get("traceparent")
            .ok()
            .flatten()
            .and_then(|tp| tp.split('-').nth(1).map(str::to_ascii_lowercase))
            .filter(|id| id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()) && id.bytes().any(|b| b != b'0'))
            .unwrap_or_else(|| hex(32));
        Some(Self { id, span: hex(16), events: RefCell::new(Vec::new()), front: RefCell::new(None) })
    }

    /// The `traceparent` the front gets (this hop as its parent).
    pub fn traceparent(&self) -> String {
        format!("00-{}-{}-01", self.id, self.span)
    }

    fn push(&self, name: &str, start_ms: i64, dur_ms: i64, attrs: Option<Value>) {
        let mut e = json!({
            "trace": self.id, "host": "edge", "comp": "edge", "name": name, "clock": "host",
            "t_wall_ns": start_ms * 1_000_000, "dur_ns": dur_ms.max(0) * 1_000_000,
        });
        if let Some(a) = attrs {
            e["attrs"] = a;
        }
        self.events.borrow_mut().push(e);
    }

    /// A span from `start_ms` to now.
    pub fn span(&self, name: &str, start_ms: i64) {
        let now = now_ms();
        self.push(name, start_ms, now - start_ms, None);
    }

    /// The forward to `front`: a span plus the NTP-style sample from the
    /// front's `x-fv-trace-t` (`<recv ns>;<send ns>` on its clock).
    pub fn forwarded(&self, front: &str, start_ms: i64, resp: &Response) {
        let end = now_ms();
        let pod = resp.headers().get(proto::front::TRACE_TIME_HEADER).ok().flatten();
        let sample = pod.as_deref().and_then(|v| v.split_once(';')).and_then(|(a, b)| Some((a.trim().parse::<i64>().ok()?, b.trim().parse::<i64>().ok()?)));
        let attrs = sample.map(|(t1, t2)| json!({"sync": {"peer": "front", "t0": start_ms * 1_000_000, "t1": t1, "t2": t2, "t3": end * 1_000_000}}));
        self.push("forward", start_ms, end - start_ms, Some(json!({"status": resp.status_code(), "front": front, "sync": attrs.map(|a| a["sync"].clone())})));
        *self.front.borrow_mut() = Some(front.to_owned());
    }

    /// Adds `x-fv-edge-t` and `server-timing` to the response and ships the
    /// events after it is sent.
    pub fn finish(self, resp: Response, recv_ms: i64, env: &Env, ctx: &Context) -> Result<Response> {
        let send = now_ms();
        self.push("request", recv_ms, send - recv_ms, Some(json!({"status": resp.status_code()})));
        let hs = Headers::new();
        for (k, v) in resp.headers().entries() {
            hs.append(&k, &v)?;
        }
        hs.set("x-fv-edge-t", &format!("{recv_ms};{send}"))?;
        hs.append("server-timing", &format!("fv-edge;dur={}", send - recv_ms))?;
        hs.set("x-fv-trace-id", &self.id)?;
        let resp = resp.with_headers(hs);
        if let Some(front) = self.front.borrow().clone() {
            let token = env.secret("FV_INTERNAL_TOKEN").map(|s| s.to_string()).unwrap_or_default();
            let url = format!("{}/fv/v1/traces/{}/events", front.trim_end_matches('/'), self.id);
            let body = json!({"events": self.events.borrow().clone()}).to_string();
            ctx.wait_until(async move {
                let h = Headers::new();
                let _ = h.set(proto::TOKEN_HEADER, &token);
                let _ = h.set("content-type", "application/json");
                let mut init = RequestInit::new();
                init.with_method(Method::Post).with_headers(h).with_body(Some(body.into()));
                if let Ok(req) = Request::new_with_init(&url, &init) {
                    let _ = Fetch::Request(req).send().await;
                }
            });
        }
        Ok(resp)
    }
}
