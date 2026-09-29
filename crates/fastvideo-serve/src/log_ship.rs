//! Log shipping to fv-control (docs/control/README.md "Logs"): a tracing
//! layer that turns every event into one JSON line (`ts`, `level`,
//! `target`, `fields` with the message and the fields of the enclosing
//! spans, so `job_id` / `session_id` ride along) and a background task that
//! POSTs them in batches to the controller's ingest endpoint.
//!
//! Off unless `FV_LOG_SHIP_URL` is set; configured by env only:
//!
//! | env | default | |
//! |---|---|---|
//! | `FV_LOG_SHIP_URL` | – | `https://<controller>/ingest/v1/logs` |
//! | `FV_LOG_SHIP_TOKEN` | – | the cluster's ingest token (Bearer; never logged) |
//! | `FV_LOG_SHIP_LEVEL` | `info` | the most verbose level shipped (`RUST_LOG` still filters first) |
//! | `FV_LOG_SHIP_BATCH` | `200` | lines per POST |
//! | `FV_LOG_SHIP_INTERVAL_MS` | `2000` | flush interval |
//! | `FV_LOG_SHIP_MAX_PER_MIN` | `6000` | lines per minute; the rest are dropped and counted |
//! | `FV_LOG_SHIP_POD` | `RUNPOD_POD_ID`, else `FV_WORKER_ID` | the pod id the lines are filed under |
//!
//! Never blocks the caller: the layer `try_send`s into a bounded queue
//! (10 000 lines) and drops when it is full. The HTTP client's own events
//! (hyper, reqwest, rustls, h2) are not shipped, so shipping cannot feed itself.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

const QUEUE: usize = 10_000;
static DROPPED: AtomicU64 = AtomicU64::new(0);
type Rx = tokio::sync::mpsc::Receiver<Value>;
static RX: OnceLock<Mutex<Option<(Rx, ShipCfg)>>> = OnceLock::new();

/// The shipper's settings (from the env).
#[derive(Clone, Debug)]
pub struct ShipCfg {
    pub url: String,
    pub token: String,
    pub level: Level,
    pub batch: usize,
    pub interval_ms: u64,
    pub max_per_min: u64,
    pub pod: String,
}

impl ShipCfg {
    /// `None` unless `FV_LOG_SHIP_URL` and `FV_LOG_SHIP_TOKEN` are set.
    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let url = get("FV_LOG_SHIP_URL").filter(|v| !v.is_empty())?;
        let token = get("FV_LOG_SHIP_TOKEN").filter(|v| !v.is_empty())?;
        let num = |k: &str, d: u64| get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
        let level = match get("FV_LOG_SHIP_LEVEL").unwrap_or_default().to_ascii_lowercase().as_str() {
            "trace" => Level::TRACE,
            "debug" => Level::DEBUG,
            "warn" => Level::WARN,
            "error" => Level::ERROR,
            _ => Level::INFO,
        };
        let pod = get("FV_LOG_SHIP_POD").or_else(|| get("RUNPOD_POD_ID")).or_else(|| get("FV_WORKER_ID")).unwrap_or_else(|| "local".into());
        Some(Self {
            url,
            token,
            level,
            batch: num("FV_LOG_SHIP_BATCH", 200).clamp(1, 2000) as usize,
            interval_ms: num("FV_LOG_SHIP_INTERVAL_MS", 2000).clamp(200, 60_000),
            max_per_min: num("FV_LOG_SHIP_MAX_PER_MIN", 6000).max(1),
            pod,
        })
    }
}

/// The tracing layer (see the module docs).
pub struct ShipLayer {
    tx: tokio::sync::mpsc::Sender<Value>,
    level: Level,
}

/// The layer when shipping is configured; `spawn` starts the sender later,
/// inside the runtime. Call once, before the subscriber is installed.
pub fn layer() -> Option<ShipLayer> {
    let cfg = ShipCfg::from_env(|k| std::env::var(k).ok())?;
    if !cfg!(feature = "http-client") {
        eprintln!("fv-serve: FV_LOG_SHIP_URL is set but this build has no HTTP client; logs are not shipped");
        return None;
    }
    let (tx, rx) = tokio::sync::mpsc::channel(QUEUE);
    let level = cfg.level;
    RX.get_or_init(|| Mutex::new(None)).lock().ok()?.replace((rx, cfg));
    Some(ShipLayer { tx, level })
}

/// Lines dropped so far (queue full or over the per-minute budget).
pub fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

#[derive(Default)]
struct JsonVisitor(Map<String, Value>);
impl Visit for JsonVisitor {
    fn record_debug(&mut self, f: &Field, v: &dyn std::fmt::Debug) {
        self.0.insert(f.name().to_owned(), Value::String(format!("{v:?}")));
    }
    fn record_str(&mut self, f: &Field, v: &str) {
        self.0.insert(f.name().to_owned(), Value::String(v.to_owned()));
    }
    fn record_i64(&mut self, f: &Field, v: i64) {
        self.0.insert(f.name().to_owned(), v.into());
    }
    fn record_u64(&mut self, f: &Field, v: u64) {
        self.0.insert(f.name().to_owned(), v.into());
    }
    fn record_f64(&mut self, f: &Field, v: f64) {
        self.0.insert(f.name().to_owned(), v.into());
    }
    fn record_bool(&mut self, f: &Field, v: bool) {
        self.0.insert(f.name().to_owned(), v.into());
    }
}
/// A span's fields, kept in its extensions for the events inside it.
struct SpanFields(Map<String, Value>);

fn skipped(target: &str) -> bool {
    ["hyper", "reqwest", "rustls", "h2", "tower", "fastvideo_serve::log_ship"].iter().any(|p| target.starts_with(p))
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for ShipLayer {
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &tracing::span::Id, ctx: Context<'_, S>) {
        let mut v = JsonVisitor::default();
        attrs.record(&mut v);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanFields(v.0));
        }
    }
    fn on_record(&self, id: &tracing::span::Id, values: &tracing::span::Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut v = JsonVisitor::default();
            values.record(&mut v);
            let mut ext = span.extensions_mut();
            if let Some(f) = ext.get_mut::<SpanFields>() {
                f.0.extend(v.0);
            }
        }
    }
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let meta = event.metadata();
        if *meta.level() > self.level || skipped(meta.target()) {
            return;
        }
        let mut fields = Map::new();
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                if let Some(f) = span.extensions().get::<SpanFields>() {
                    for (k, v) in &f.0 {
                        fields.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        let mut v = JsonVisitor(fields);
        event.record(&mut v);
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
        let line = serde_json::json!({"ts": ts, "level": meta.level().as_str(), "target": meta.target(), "fields": Value::Object(v.0)});
        if self.tx.try_send(line).is_err() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Starts the sender task (inside the tokio runtime). No-op when shipping is off.
#[cfg(feature = "http-client")]
pub fn spawn() {
    let Some((rx, cfg)) = RX.get().and_then(|m| m.lock().ok()).and_then(|mut g| g.take()) else {
        return;
    };
    tokio::spawn(run(rx, cfg));
}
#[cfg(not(feature = "http-client"))]
pub fn spawn() {}

#[cfg(feature = "http-client")]
async fn run(mut rx: Rx, cfg: ShipCfg) {
    use std::time::{Duration, Instant};
    let http = match reqwest::Client::builder().timeout(Duration::from_secs(15)).build() {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut window = Instant::now();
    let mut in_window = 0u64;
    let mut reported_drops = 0u64;
    loop {
        let mut batch = Vec::with_capacity(cfg.batch);
        let deadline = tokio::time::sleep(Duration::from_millis(cfg.interval_ms));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                m = rx.recv() => match m {
                    Some(v) => { batch.push(v); if batch.len() >= cfg.batch { break; } }
                    None => { if batch.is_empty() { return; } break; }
                },
                _ = &mut deadline => break,
            }
        }
        if window.elapsed() >= Duration::from_secs(60) {
            window = Instant::now();
            in_window = 0;
        }
        let room = cfg.max_per_min.saturating_sub(in_window) as usize;
        if batch.len() > room {
            DROPPED.fetch_add((batch.len() - room) as u64, Ordering::Relaxed);
            batch.truncate(room);
        }
        let drops = dropped();
        if drops > reported_drops {
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
            batch.push(serde_json::json!({"ts": ts, "level": "WARN", "target": "fv_serve::log_ship", "fields": {"message": "log lines dropped (queue full or over FV_LOG_SHIP_MAX_PER_MIN)", "dropped_total": drops}}));
            reported_drops = drops;
        }
        if batch.is_empty() {
            continue;
        }
        in_window += batch.len() as u64;
        let body = serde_json::json!({"pod": cfg.pod, "lines": batch});
        for attempt in 0..3u32 {
            let r = http.post(&cfg.url).bearer_auth(&cfg.token).json(&body).send().await;
            match r {
                Ok(r) if r.status().is_success() => break,
                // Auth or request problems do not get better with a retry.
                Ok(r) if r.status().is_client_error() && r.status().as_u16() != 429 => break,
                _ => tokio::time::sleep(Duration::from_millis(500 * 2u64.pow(attempt))).await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::prelude::*;

    #[test]
    fn config_from_env() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(a, _)| *a == k).map(|(_, b)| (*b).to_owned());
        assert!(ShipCfg::from_env(env(&[])).is_none());
        assert!(ShipCfg::from_env(env(&[("FV_LOG_SHIP_URL", "https://x")])).is_none(), "a token is required");
        let c = ShipCfg::from_env(env(&[("FV_LOG_SHIP_URL", "https://x"), ("FV_LOG_SHIP_TOKEN", "t"), ("RUNPOD_POD_ID", "abc"), ("FV_LOG_SHIP_LEVEL", "WARN"), ("FV_LOG_SHIP_BATCH", "99999")])).unwrap();
        assert_eq!(c.pod, "abc");
        assert_eq!(c.level, Level::WARN);
        assert_eq!(c.batch, 2000);
        assert_eq!(c.interval_ms, 2000);
    }

    #[test]
    fn events_become_json_lines_with_span_fields() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let sub = tracing_subscriber::registry().with(ShipLayer { tx, level: Level::INFO });
        tracing::subscriber::with_default(sub, || {
            let span = tracing::info_span!("job", job_id = "job_1");
            let _g = span.enter();
            tracing::info!(step = 3, "denoise");
            tracing::debug!("not shipped at info");
            tracing::warn!(target: "hyper::client", "not shipped: the HTTP client");
        });
        let a = rx.try_recv().unwrap();
        assert_eq!(a["level"], "INFO");
        assert_eq!(a["fields"]["message"], "denoise");
        assert_eq!(a["fields"]["job_id"], "job_1");
        assert_eq!(a["fields"]["step"], 3);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn a_full_queue_drops_instead_of_blocking() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let before = dropped();
        let sub = tracing_subscriber::registry().with(ShipLayer { tx, level: Level::INFO });
        tracing::subscriber::with_default(sub, || {
            for i in 0..5 {
                tracing::info!(i, "x");
            }
        });
        assert!(dropped() >= before + 4);
    }
}
