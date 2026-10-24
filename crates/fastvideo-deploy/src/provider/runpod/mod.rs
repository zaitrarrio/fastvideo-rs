//! Runpod (docs/adr/0018-fv-deploy.md §4): [`Pod`] (REST v1 `POST /pods`) and
//! [`Serverless`] (REST v1 `POST /templates` + `POST /endpoints`), on the
//! shared [`RunpodApi`] client: REST v1, GraphQL (balance, price quotes), the
//! console's log endpoint (`hapi`, as fv-control's `runpod.ts` reads it), the
//! serverless queue API and the pods' HTTPS proxy.
//!
//! The payloads are byte-for-byte what `scripts/serve/runpod-pod.sh` and
//! `scripts/serve/runpod-endpoint.sh` send for the same inputs
//! (`fixtures/deploy/*.json`, regenerated from the scripts by
//! `scripts/serve/tests/fv-deploy-goldens.sh`; `tests/payload_goldens.rs`).
//!
//! The API key is only ever an `Authorization` header; error bodies are
//! scrubbed of it and `Debug` never prints it.

mod pod;
mod serverless;

pub use pod::{create_body as pod_create_body, Pod, PodListing, PodReach};
pub use serverless::{
    endpoint_body, template_body, QueueReach, Serverless, SERVERLESS_DELETE_ATTEMPTS,
};

use std::time::Duration;

use serde_json::{json, Value};

use super::{scrub, Account, CloudType, Offer, ProviderError, ProviderKind};
use crate::boot::env::BootEnv;
use crate::boot::sys::{Clock, Http, Method, Request};

/// REST v1.
pub const REST: &str = "https://rest.runpod.io/v1";
/// The serverless queue API.
pub const QUEUE: &str = "https://api.runpod.ai/v2";
pub const GRAPHQL: &str = "https://api.runpod.io/graphql";
/// The console's log endpoint (not in the public docs; it answers with the
/// API key: docs/control/README.md "Logs").
pub const HAPI: &str = "https://hapi.runpod.net/v1";
/// A pod's port 8000 behind Runpod's HTTPS proxy (`{id}` is the pod id).
pub const PROXY: &str = "https://{id}-8000.proxy.runpod.net";

/// The API key's variable.
pub const API_KEY_VAR: &str = "RUNPOD_API_KEY";

/// Reply texts that mean "no capacity for this GPU type here" (the scripts'
/// and the autoscaler's list).
pub const NO_CAPACITY: [&str; 3] = [
    "no instances currently available",
    "no longer any instances",
    "does not have the resources",
];

/// Where the Runpod APIs are (overridable for tests and fakes with the
/// scripts' variables).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunpodUrls {
    pub rest: String,
    pub queue: String,
    pub graphql: String,
    pub hapi: String,
    /// `{id}` stands for the pod id.
    pub proxy: String,
}

impl Default for RunpodUrls {
    fn default() -> Self {
        Self {
            rest: REST.into(),
            queue: QUEUE.into(),
            graphql: GRAPHQL.into(),
            hapi: HAPI.into(),
            proxy: PROXY.into(),
        }
    }
}

impl RunpodUrls {
    /// `RUNPOD_API_BASE`, `RUNPOD_QUEUE_API`, `RUNPOD_GRAPHQL` (the scripts'
    /// names), `RUNPOD_HAPI`, `FV_RUNPOD_PROXY`; the defaults otherwise.
    pub fn from_env(env: &BootEnv) -> Self {
        let d = Self::default();
        let get = |k: &str, dflt: String| {
            env.get(k)
                .map(|v| v.trim_end_matches('/').to_owned())
                .unwrap_or(dflt)
        };
        Self {
            rest: get("RUNPOD_API_BASE", d.rest),
            queue: get("RUNPOD_QUEUE_API", d.queue),
            graphql: get("RUNPOD_GRAPHQL", d.graphql),
            hapi: get("RUNPOD_HAPI", d.hapi),
            proxy: get("FV_RUNPOD_PROXY", d.proxy),
        }
    }

    /// A pod's proxy base URL.
    pub fn pod_url(&self, pod: &str) -> String {
        self.proxy.replace("{id}", pod)
    }
}

/// The Runpod client: the HTTP and clock seams, the key, the URLs.
pub struct RunpodApi<'a> {
    http: &'a dyn Http,
    clock: &'a dyn Clock,
    key: String,
    pub urls: RunpodUrls,
}

impl std::fmt::Debug for RunpodApi<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunpodApi")
            .field("urls", &self.urls)
            .field(
                "key",
                &if self.key.is_empty() {
                    "<unset>"
                } else {
                    "<redacted>"
                },
            )
            .finish()
    }
}

/// An answer with its status and raw body.
pub(crate) struct Raw {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Raw {
    fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

impl<'a> RunpodApi<'a> {
    pub fn new(
        http: &'a dyn Http,
        clock: &'a dyn Clock,
        key: impl Into<String>,
        urls: RunpodUrls,
    ) -> Self {
        Self {
            http,
            clock,
            key: key.into(),
            urls,
        }
    }

    /// From the environment: `RUNPOD_API_KEY` (required) and the URL
    /// overrides. The key never leaves this struct but as a header.
    pub fn from_env(
        http: &'a dyn Http,
        clock: &'a dyn Clock,
        env: &BootEnv,
    ) -> Result<Self, ProviderError> {
        let key = env
            .get(API_KEY_VAR)
            .ok_or(ProviderError::MissingKey(API_KEY_VAR))?;
        Ok(Self::new(http, clock, key, RunpodUrls::from_env(env)))
    }

    pub fn clock(&self) -> &'a dyn Clock {
        self.clock
    }

    pub fn http(&self) -> &'a dyn Http {
        self.http
    }

    /// Whether `text` contains the key (a launch's env must not).
    pub fn leaks_key(&self, text: &str) -> bool {
        !self.key.is_empty() && text.contains(&self.key)
    }

    fn scrub(&self, text: &str) -> String {
        scrub(text, &self.key, 400)
    }

    /// One request with the key as a bearer header.
    pub(crate) fn raw(
        &self,
        method: Method,
        url: &str,
        body: Option<&Value>,
        timeout: Duration,
        what: &str,
    ) -> Result<Raw, ProviderError> {
        let mut req = Request::new(method, url, timeout).bearer(&self.key);
        if let Some(b) = body {
            req = req.json(b.to_string());
        } else {
            req = req.header("content-type", "application/json");
        }
        let r = self.http.send(&req).map_err(|e| ProviderError::Transport {
            what: what.to_owned(),
            why: self.scrub(&e.to_string()),
        })?;
        Ok(Raw {
            status: r.status,
            body: r.body,
        })
    }

    /// A REST v1 call whose non-2xx answer is an error (no capacity
    /// recognized); an empty body is `Null`.
    pub(crate) fn rest(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        what: &str,
    ) -> Result<Value, ProviderError> {
        let url = format!("{}{path}", self.urls.rest);
        let r = self.raw(method, &url, body, Duration::from_secs(60), what)?;
        self.json_or_error(r, what)
    }

    pub(crate) fn json_or_error(&self, r: Raw, what: &str) -> Result<Value, ProviderError> {
        let text = String::from_utf8_lossy(&r.body).into_owned();
        if !r.ok() {
            let snippet = self.scrub(&text);
            if NO_CAPACITY.iter().any(|m| snippet.contains(m)) {
                return Err(ProviderError::NoCapacity(format!("{what}: {snippet}")));
            }
            return Err(ProviderError::Api {
                what: what.to_owned(),
                status: r.status,
                body: snippet,
            });
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| ProviderError::Decode {
            what: what.to_owned(),
            why: e.to_string(),
        })
    }

    /// A GraphQL query's `data` (errors are an error).
    pub fn graphql(&self, query: &str, what: &str) -> Result<Value, ProviderError> {
        let body = json!({ "query": query });
        let r = self.raw(
            Method::Post,
            &self.urls.graphql,
            Some(&body),
            Duration::from_secs(30),
            what,
        )?;
        let v = self.json_or_error(r, what)?;
        if let Some(errs) = v.get("errors").and_then(Value::as_array) {
            if !errs.is_empty() {
                let msg: Vec<&str> = errs
                    .iter()
                    .filter_map(|e| e.get("message").and_then(Value::as_str))
                    .collect();
                return Err(ProviderError::Api {
                    what: what.to_owned(),
                    status: 200,
                    body: self.scrub(&msg.join("; ")),
                });
            }
        }
        Ok(v.get("data").cloned().unwrap_or(Value::Null))
    }

    /// The account balance and spend (`myself { clientBalance currentSpendPerHr }`).
    pub fn account(&self, kind: ProviderKind) -> Result<Account, ProviderError> {
        let d = self.graphql(BALANCE_QUERY, "GraphQL balance")?;
        let balance = d
            .pointer("/myself/clientBalance")
            .and_then(Value::as_f64)
            .ok_or_else(|| ProviderError::Decode {
                what: "GraphQL balance".into(),
                why: "no myself.clientBalance".into(),
            })?;
        Ok(Account {
            provider: kind,
            balance_usd: balance,
            spend_per_hr_usd: d
                .pointer("/myself/currentSpendPerHr")
                .and_then(Value::as_f64),
        })
    }

    /// The quote for one GPU type (`scripts/gpu/runpod-price.sh`): the
    /// cloud's price and, with a data centre, its lowest uninterruptible
    /// price; the higher counts; none (or no positive number) is `None`.
    pub fn quote(
        &self,
        kind: ProviderKind,
        gpu: &str,
        cloud: CloudType,
        dc: Option<&str>,
    ) -> Result<Offer, ProviderError> {
        let d = self.graphql(&price_query(gpu, cloud, dc), "GraphQL price")?;
        Ok(Offer {
            provider: kind,
            gpu_type: gpu.to_owned(),
            data_center: dc.map(str::to_owned),
            cloud,
            usd_per_hr: quoted_price(&d, cloud),
            stock: None,
        })
    }
}

/// The balance query (`runpod-pod.sh check_balance`, fv-control `runpod.account`).
pub const BALANCE_QUERY: &str = "{ myself { clientBalance currentSpendPerHr } }";

/// `fv_runpod_gpu_price`'s query, spelled as the script spells it.
pub fn price_query(gpu: &str, cloud: CloudType, dc: Option<&str>) -> String {
    let lit = |s: &str| serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into());
    let lowest = match dc {
        Some(dc) => format!(
            "lowestPrice(input: {{gpuCount: 1, secureCloud: {}, dataCenterId: {}}}) {{ uninterruptablePrice }}",
            cloud == CloudType::Secure,
            lit(dc)
        ),
        None => String::new(),
    };
    format!(
        "query {{ gpuTypes(input: {{id: {}}}) {{ id securePrice communityPrice {lowest} }} }}",
        lit(gpu)
    )
}

/// The script's jq: the cloud's price and the lowest uninterruptible one,
/// positive numbers only, the larger.
pub fn quoted_price(data: &Value, cloud: CloudType) -> Option<f64> {
    let g = data.pointer("/gpuTypes/0")?;
    let base = match cloud {
        CloudType::Secure => g.get("securePrice"),
        CloudType::Community => g.get("communityPrice"),
    };
    [base, g.pointer("/lowestPrice/uninterruptablePrice")]
        .into_iter()
        .flatten()
        .filter_map(Value::as_f64)
        .filter(|p| *p > 0.0)
        .reduce(f64::max)
}

/// Parses Runpod's `"2026-09-27 20:31:01.321 +0000 UTC"` (and RFC 3339
/// `2026-09-27T20:31:01Z`) into unix seconds (UTC offsets other than zero
/// are not used by Runpod and not handled).
pub fn parse_runpod_time(s: &str) -> Option<f64> {
    let s = s.trim();
    let (date, rest) = s.split_at_checked(10)?;
    let mut d = date.split('-').map(|x| x.parse::<i64>().ok());
    let (y, mo, da) = (d.next()??, d.next()??, d.next()??);
    let time = rest.trim_start_matches([' ', 'T']);
    let hms: String = time
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == ':' || *c == '.')
        .collect();
    let mut t = hms.split(':');
    let (h, mi) = (
        t.next()?.parse::<i64>().ok()?,
        t.next()?.parse::<i64>().ok()?,
    );
    let sec: f64 = t.next()?.parse().ok()?;
    // Days from civil (Howard Hinnant).
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let doy = (153 * (mo + if mo > 2 { -3 } else { 9 }) + 2) / 5 + da - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some((days * 86_400 + h * 3600 + mi * 60) as f64 + sec)
}
