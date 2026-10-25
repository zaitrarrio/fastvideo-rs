//! A Vast instance (docs/adr/0018-fv-deploy.md §7.2): the offer search
//! (REST `POST /bundles/`) and the create (`PUT /asks/<offer>/`) that
//! `scripts/serve/vast.sh` sends, field for field
//! (`fixtures/deploy/vast-{offers,create}.json`, written by that script's own
//! `plan`; `tests/payload_goldens.rs`), reached at the instance's public
//! address and mapped port 8000.
//!
//! What the repository's scripts already use (and so is not invented here):
//! `POST /bundles/` with the filter object and `.offers[].{id,dph_total}`,
//! `PUT /asks/<id>/` answering `.new_contract`, `GET /instances/<id>/`
//! answering `.instances.{actual_status,public_ipaddr,ports["8000/tcp"][0].HostPort}`,
//! `DELETE /instances/<id>/`, and the account's `credit` (`vastai show user`,
//! `scripts/gpu/lib.sh`). **UNVERIFIED** (not in a script or a document read
//! for C3b; marked where used): the account's REST path `GET /users/current/`,
//! the instance list `GET /instances/` answering `.instances[]` with `label`,
//! the offers' `gpu_name` and `gpu_name: {"in": [...]}` filter, a 404 for an
//! instance that is gone, and a 2xx `{"success": false}` meaning a refused
//! create. The live gate (the owner's) checks them.
//!
//! The API key is only ever an `Authorization: Bearer` header (as `vast.sh`);
//! error bodies are scrubbed of it and `Debug` never prints it.

use std::time::Duration;

use serde_json::{json, Map, Value};

use super::runpod::PodReach;
use super::{
    scrub, Account, ApiCall, CloudType, Cursor, Instance, InstanceId, InstanceState, Launch,
    LogPage, Offer, OfferQuery, Probe, Provider, ProviderError, ProviderKind, Reach, ReachOptions,
};
use crate::boot::sys::{Clock, Http, Method, Request};
use crate::vast::{env_dict, onstart, port_of, Port};

/// The REST base (`vast.sh` `VAST_API_BASE` default).
pub const API: &str = "https://console.vast.ai/api/v0";
/// The API key's variable.
pub const API_KEY_VAR: &str = "VAST_API_KEY";
/// Vast refuses an onstart of 4048 characters or more (`vast.sh validate`).
pub const MAX_ONSTART: usize = 4048;
/// The offer filter's floors (`vast.sh offer_query`): a CUDA 13 driver,
/// reliability 0.95, at most 20 offers, cheapest first.
pub const MIN_CUDA: f64 = 13.0;
pub const MIN_RELIABILITY: f64 = 0.95;
pub const OFFER_LIMIT: u32 = 20;
/// How many offers a quote reads (one quote per GPU name among them).
pub const QUOTE_LIMIT: u32 = 64;
/// The script's offer placeholder in `plan`.
pub const OFFER_PLACEHOLDER: &str = "<offer id>";

/// The Vast client: the HTTP and clock seams, the key, the base URL.
pub struct VastApi<'a> {
    http: &'a dyn Http,
    clock: &'a dyn Clock,
    key: String,
    pub base: String,
}

impl std::fmt::Debug for VastApi<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VastApi")
            .field("base", &self.base)
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

impl<'a> VastApi<'a> {
    pub fn new(
        http: &'a dyn Http,
        clock: &'a dyn Clock,
        key: impl Into<String>,
        base: impl Into<String>,
    ) -> Self {
        Self {
            http,
            clock,
            key: key.into(),
            base: base.into().trim_end_matches('/').to_owned(),
        }
    }

    /// From the environment: `VAST_API_KEY` (required) and `VAST_API_BASE`.
    pub fn from_env(
        http: &'a dyn Http,
        clock: &'a dyn Clock,
        env: &crate::boot::env::BootEnv,
    ) -> Result<Self, ProviderError> {
        let key = env
            .get(API_KEY_VAR)
            .ok_or(ProviderError::MissingKey(API_KEY_VAR))?;
        Ok(Self::new(http, clock, key, base_from(env)))
    }

    pub fn clock(&self) -> &'a dyn Clock {
        self.clock
    }

    /// Whether `text` contains the key (a launch must not).
    pub fn leaks_key(&self, text: &str) -> bool {
        !self.key.is_empty() && text.contains(&self.key)
    }

    /// One call; a non-2xx answer is an error (a 4xx: a refusal), an empty
    /// body `Null`. `gone_ok`: a 404 answers `Ok(None)`.
    fn call(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        what: &str,
        gone_ok: bool,
    ) -> Result<Option<Value>, ProviderError> {
        let mut req = Request::new(
            method,
            format!("{}{path}", self.base),
            Duration::from_secs(60),
        )
        .bearer(&self.key)
        .header("content-type", "application/json");
        if let Some(b) = body {
            req.body = Some(b.to_string().into_bytes());
        }
        let r = self.http.send(&req).map_err(|e| ProviderError::Transport {
            what: what.to_owned(),
            why: scrub(&e.to_string(), &self.key, 400),
        })?;
        let text = String::from_utf8_lossy(&r.body).into_owned();
        if r.status == 404 && gone_ok {
            return Ok(None);
        }
        if !(200..300).contains(&r.status) {
            return Err(ProviderError::Api {
                what: what.to_owned(),
                status: r.status,
                body: scrub(&text, &self.key, 400),
            });
        }
        if text.trim().is_empty() {
            return Ok(Some(Value::Null));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Decode {
            what: what.to_owned(),
            why: e.to_string(),
        })?;
        // UNVERIFIED: a 2xx `{"success": false, "msg": …}` is a refusal (the
        // offer is gone, the arguments are bad): nothing was created.
        if v.get("success") == Some(&Value::Bool(false)) {
            let msg = v
                .get("msg")
                .or_else(|| v.get("error"))
                .map(|m| m.as_str().map_or_else(|| m.to_string(), str::to_owned))
                .unwrap_or_default();
            return Err(ProviderError::NoCapacity(format!(
                "{what}: {}",
                scrub(&msg, &self.key, 300)
            )));
        }
        Ok(Some(v))
    }

    fn json(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        what: &str,
    ) -> Result<Value, ProviderError> {
        Ok(self
            .call(method, path, body, what, false)?
            .unwrap_or(Value::Null))
    }

    /// `POST /bundles/` with `query`: the offers, cheapest first.
    pub fn search(&self, query: &Value) -> Result<Vec<Value>, ProviderError> {
        let v = self.json(Method::Post, "/bundles/", Some(query), "POST bundles")?;
        Ok(v.get("offers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default())
    }

    /// `GET /instances/<id>/`: the instance object; `None` when gone (404).
    pub fn instance(&self, id: &str) -> Result<Option<Value>, ProviderError> {
        let v = self.call(
            Method::Get,
            &format!("/instances/{id}/"),
            None,
            "GET instance",
            true,
        )?;
        // `vast.sh`: `.instances // .`.
        Ok(v.map(|v| match v.get("instances") {
            Some(i) => i.clone(),
            None => v,
        }))
    }
}

/// `VAST_API_BASE` or [`API`].
pub fn base_from(env: &crate::boot::env::BootEnv) -> String {
    env.get("VAST_API_BASE")
        .map(|v| v.trim_end_matches('/').to_owned())
        .unwrap_or_else(|| API.to_owned())
}

/// A Vast search filter (`vast.sh offer_query`). `disk` and `max_dph` are the
/// launch's; a quote leaves them out (the price cap is then checked on the
/// quote, and the create's own search applies both).
pub fn search_query(
    gpu_names: &[String],
    gpus: u32,
    disk: Option<u32>,
    max_dph: Option<f64>,
    limit: u32,
) -> Value {
    let mut q = Map::new();
    q.insert("rentable".into(), json!({"eq": true}));
    q.insert("verified".into(), json!({"eq": true}));
    q.insert("num_gpus".into(), json!({"eq": gpus}));
    q.insert("direct_port_count".into(), json!({"gte": 1}));
    q.insert("cuda_max_good".into(), json!({"gte": MIN_CUDA}));
    q.insert("reliability2".into(), json!({"gte": MIN_RELIABILITY}));
    if let Some(d) = disk {
        q.insert("disk_space".into(), json!({"gte": d}));
    }
    if let Some(p) = max_dph {
        q.insert("dph_total".into(), json!({"lte": p}));
    }
    if !gpu_names.is_empty() {
        // UNVERIFIED: the `in` operator on `gpu_name` (the script filters no
        // GPU name: any GPU with a CUDA 13 driver and a direct port).
        q.insert("gpu_name".into(), json!({"in": gpu_names}));
    }
    q.insert("type".into(), json!("on-demand"));
    q.insert("order".into(), json!([["dph_total", "asc"]]));
    q.insert("limit".into(), json!(limit));
    Value::Object(q)
}

/// The offer search of a launch (`vast.sh offer_query`,
/// `fixtures/deploy/vast-offers.json`).
pub fn offer_query(l: &Launch) -> Value {
    search_query(
        &l.gpu.types,
        l.gpu.count,
        Some(l.disk.container_gb),
        Some(l.guards.max_usd_per_hr),
        OFFER_LIMIT,
    )
}

/// The ports a launch publishes on Vast (`-p` keys of the env object).
fn ports(l: &Launch) -> Result<Vec<Port>, ProviderError> {
    l.ports
        .iter()
        .map(|p| {
            port_of(p).ok_or_else(|| ProviderError::Unsupported(format!("Vast: bad port {p:?}")))
        })
        .collect()
}

/// The create body of a launch (`vast.sh create_payload`,
/// `fixtures/deploy/vast-create.json`): the image, the label (the launch
/// name), the disk, `ssh_direct` with an onstart that starts fv-boot with the
/// launch's arguments, and the env **object** (published ports as `-p` keys,
/// then the plain settings). Refused: a secret name in the env (Vast injects
/// secrets as account env vars), an entrypoint other than fv-boot (the
/// onstart starts it), an onstart over Vast's 4048 characters, no published
/// port.
pub fn create_body(l: &Launch) -> Result<Value, ProviderError> {
    if !(l.entrypoint.is_empty() || l.entrypoint == [crate::launch::FV_BOOT_BIN]) {
        return Err(ProviderError::Unsupported(format!(
            "Vast: the onstart starts fv-boot; entrypoint {:?} cannot be sent",
            l.entrypoint
        )));
    }
    let ports = ports(l)?;
    if ports.is_empty() {
        return Err(ProviderError::Unsupported(
            "Vast: no published port (fv-serve's 8000 is unreachable without one)".into(),
        ));
    }
    let vars: Vec<(&str, &str)> = l
        .env
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let env = env_dict(&ports, &vars).map_err(ProviderError::Unsupported)?;
    // `$(onstart)` in the script drops the heredoc's last newline.
    let start = onstart(&l.args.join(" ")).trim_end_matches('\n').to_owned();
    if start.len() >= MAX_ONSTART {
        return Err(ProviderError::Unsupported(format!(
            "Vast: onstart is {} characters (Vast caps it at {MAX_ONSTART})",
            start.len()
        )));
    }
    Ok(json!({
        "client_id": "me",
        "image": l.image,
        "label": l.name,
        "disk": l.disk.container_gb,
        "runtype": "ssh_direct",
        "target_state": "running",
        "cancel_unavail": true,
        "onstart": start,
        "env": env,
    }))
}

/// `actual_status` → [`InstanceState`] (`vast.sh`: `running`; `exited`,
/// `offline`, `unknown` are a bad host).
pub fn instance_state(actual: &str) -> InstanceState {
    match actual {
        "running" => InstanceState::Running,
        "created" | "loading" | "scheduling" | "starting" => InstanceState::Pending,
        "exited" | "stopped" | "offline" => InstanceState::Exited,
        _ => InstanceState::Unknown,
    }
}

/// `http://<public_ipaddr>:<host port of 8000/tcp>` once Vast reports both.
pub fn instance_url(inst: &Value) -> Option<String> {
    let ip = inst
        .get("public_ipaddr")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())?;
    let port = inst.pointer("/ports/8000~1tcp/0/HostPort").and_then(|p| {
        p.as_str()
            .map(str::to_owned)
            .or_else(|| p.as_u64().map(|n| n.to_string()))
    })?;
    Some(format!("http://{ip}:{port}"))
}

fn offer_id(o: &Value) -> Option<String> {
    o.get("id").and_then(|i| {
        i.as_u64()
            .map(|n| n.to_string())
            .or_else(|| i.as_str().map(str::to_owned))
    })
}

/// Vast instances.
#[derive(Debug)]
pub struct Vast<'a> {
    pub api: VastApi<'a>,
}

impl<'a> Vast<'a> {
    pub fn new(api: VastApi<'a>) -> Self {
        Self { api }
    }
}

impl Provider for Vast<'_> {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Vast
    }

    /// The cheapest offer of each GPU name among the matching offers (any
    /// GPU when the query names none), its `dph_total` the quote.
    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>, ProviderError> {
        let found = self
            .api
            .search(&search_query(&q.gpu_types, 1, None, None, QUOTE_LIMIT))?;
        let mut out: Vec<Offer> = Vec::new();
        for o in &found {
            // UNVERIFIED: `gpu_name` on an offer.
            let name = o
                .get("gpu_name")
                .and_then(Value::as_str)
                .unwrap_or("any")
                .to_owned();
            if out.iter().any(|x| x.gpu_type == name) {
                continue;
            }
            out.push(Offer {
                provider: self.kind(),
                gpu_type: name,
                data_center: None,
                cloud: CloudType::Community,
                usd_per_hr: super::parse_usd(o.get("dph_total")),
                stock: None,
            });
        }
        // A named type without an offer is quoted as none (never created).
        for g in &q.gpu_types {
            if !out.iter().any(|x| &x.gpu_type == g) {
                out.push(Offer {
                    provider: self.kind(),
                    gpu_type: g.clone(),
                    data_center: None,
                    cloud: CloudType::Community,
                    usd_per_hr: None,
                    stock: Some("none".into()),
                });
            }
        }
        Ok(out)
    }

    fn payload(&self, launch: &Launch) -> Result<Vec<ApiCall>, ProviderError> {
        Ok(vec![
            ApiCall {
                method: "POST".into(),
                url: format!("{}/bundles/", self.api.base),
                body: offer_query(launch),
            },
            ApiCall {
                method: "PUT".into(),
                url: format!("{}/asks/{OFFER_PLACEHOLDER}/", self.api.base),
                body: create_body(launch)?,
            },
        ])
    }

    /// The script's create: search with the launch's filter (price cap and
    /// disk included), rent the cheapest offer. The billed price is that
    /// offer's `dph_total`.
    fn create(&self, launch: &Launch) -> Result<Instance, ProviderError> {
        let body = create_body(launch)?;
        let offers = self.api.search(&offer_query(launch))?;
        let Some(o) = offers.first() else {
            return Err(ProviderError::NoCapacity(format!(
                "no Vast offer under ${}/hr with a direct port and CUDA >= {MIN_CUDA} ({})",
                launch.guards.max_usd_per_hr,
                if launch.gpu.types.is_empty() {
                    "any GPU".to_owned()
                } else {
                    launch.gpu.types.join(", ")
                }
            )));
        };
        let offer = offer_id(o).ok_or_else(|| ProviderError::Decode {
            what: "POST bundles".into(),
            why: "an offer without an id".into(),
        })?;
        let v = self.api.json(
            Method::Put,
            &format!("/asks/{offer}/"),
            Some(&body),
            "PUT ask",
        )?;
        let id = v
            .get("new_contract")
            .and_then(|i| {
                i.as_u64()
                    .map(|n| n.to_string())
                    .or_else(|| i.as_str().map(str::to_owned))
            })
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ProviderError::Decode {
                what: "PUT ask".into(),
                why: format!("no new_contract in {}", scrub(&v.to_string(), "", 300)),
            })?;
        Ok(Instance {
            id: InstanceId::new(id),
            provider: self.kind(),
            name: launch.name.clone(),
            state: InstanceState::Pending,
            url: None,
            gpu: o
                .get("gpu_name")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| launch.gpu.types.first().cloned()),
            data_center: o
                .get("geolocation")
                .and_then(Value::as_str)
                .map(str::to_owned),
            usd_per_hr: super::parse_usd(o.get("dph_total")),
            created_at: Some(self.api.clock().now()),
        })
    }

    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError> {
        Ok(match self.api.instance(id.as_str())? {
            None => InstanceState::Terminated,
            Some(v) => match v.get("actual_status").and_then(Value::as_str) {
                Some(s) => instance_state(s),
                None => InstanceState::Unknown,
            },
        })
    }

    /// Not in C3b: the scripts read no Vast log API (fv-serve logs to
    /// `/var/log/fv-serve.log` in the instance; `vastai logs` / ssh).
    fn logs(&self, id: &InstanceId, _since: Option<&Cursor>) -> Result<LogPage, ProviderError> {
        Err(ProviderError::Unsupported(format!(
            "Vast {id}: fv-deploy reads no Vast log API (UNVERIFIED); ssh in and read {}",
            crate::vast::MODEL_LOG
        )))
    }

    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError> {
        self.api
            .call(
                Method::Delete,
                &format!("/instances/{id}/"),
                None,
                "DELETE instance",
                true,
            )
            .map(|_| ())
    }

    /// UNVERIFIED path: `GET /users/current/` (what `vastai show user`
    /// reads); the balance is `credit` (`scripts/gpu/lib.sh vast_balance`:
    /// `.credit // .balance`).
    fn account(&self) -> Result<Account, ProviderError> {
        let v = self
            .api
            .json(Method::Get, "/users/current/", None, "GET user")?;
        let balance = super::parse_usd(v.get("credit"))
            .or_else(|| super::parse_usd(v.get("balance")))
            .ok_or_else(|| ProviderError::Decode {
                what: "GET user".into(),
                why: "no credit / balance".into(),
            })?;
        Ok(Account {
            provider: self.kind(),
            balance_usd: balance,
            spend_per_hr_usd: None,
        })
    }

    fn reach<'b>(
        &'b self,
        instance: &Instance,
        opts: &ReachOptions<'_>,
    ) -> Result<Box<dyn Reach + 'b>, ProviderError> {
        Ok(Box::new(VastReach {
            api: &self.api,
            id: instance.id.clone(),
            key: opts.key.map(str::to_owned),
            inner: instance.url.clone().map(|base| PodReach {
                http: self.api.http,
                base,
                key: opts.key.map(str::to_owned),
            }),
        }))
    }

    /// UNVERIFIED listing: `GET /instances/` answering `.instances[]` with
    /// each one's `id` and `label` (the launch name).
    fn sweep(
        &self,
        name: &str,
        keep: Option<&InstanceId>,
    ) -> Result<Vec<InstanceId>, ProviderError> {
        let v = self
            .api
            .json(Method::Get, "/instances/", None, "GET instances")?;
        let mut gone = Vec::new();
        let mut last = None;
        for i in v
            .get("instances")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let Some(id) = offer_id(i) else { continue };
            if i.get("label").and_then(Value::as_str) != Some(name)
                || keep.is_some_and(|k| k.as_str() == id)
            {
                continue;
            }
            let id = InstanceId::new(id);
            match self.destroy(&id) {
                Ok(()) => gone.push(id),
                Err(e) => last = Some(e),
            }
        }
        match last {
            Some(e) => Err(e),
            None => Ok(gone),
        }
    }

    /// The rented offer's `dph_total` is the billed price.
    fn bills_at_create(&self) -> bool {
        true
    }
}

/// A Vast instance's fv-serve at its public address: until Vast reports the
/// address and the mapped port, a probe asks for them; then as a pod's.
pub struct VastReach<'a> {
    api: &'a VastApi<'a>,
    id: InstanceId,
    key: Option<String>,
    inner: Option<PodReach<'a>>,
}

impl VastReach<'_> {
    fn resolve(&mut self) -> Result<(), String> {
        if self.inner.is_some() {
            return Ok(());
        }
        let inst = match self.api.instance(self.id.as_str()) {
            Ok(Some(v)) => v,
            Ok(None) => return Err(format!("{} is gone", self.id)),
            Err(e) => return Err(e.to_string()),
        };
        let st = inst
            .get("actual_status")
            .and_then(Value::as_str)
            .unwrap_or("?");
        match (st, instance_url(&inst)) {
            ("running", Some(base)) => {
                self.inner = Some(PodReach {
                    http: self.api.http,
                    base,
                    key: self.key.clone(),
                });
                Ok(())
            }
            _ => Err(format!("instance {st}, no address yet")),
        }
    }
}

impl Reach for VastReach<'_> {
    fn ping(&mut self) -> Probe {
        if let Err(w) = self.resolve() {
            return Probe::Down(w);
        }
        match self.inner.as_mut() {
            Some(r) => r.ping(),
            None => Probe::Down("no address".into()),
        }
    }

    fn call(
        &mut self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        stop: &dyn Fn() -> bool,
    ) -> Result<(u16, Value), ProviderError> {
        self.resolve().map_err(|why| ProviderError::Transport {
            what: format!("{} {path}", method.as_str()),
            why,
        })?;
        match self.inner.as_mut() {
            Some(r) => r.call(method, path, body, stop),
            None => Err(ProviderError::Unsupported("no address".into())),
        }
    }
}
