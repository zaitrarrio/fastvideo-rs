//! A Runpod serverless queue endpoint: a template (REST v2 `POST /templates`)
//! and an endpoint on it (REST v2 `POST /serverless`, v1's `/endpoints`), the
//! v2 form of the bodies `runpod-endpoint.sh`'s `template_payload` /
//! `endpoint_payload` build (ADR-0018 §11); reached through the queue API,
//! one native job per call (the envelope of `crate::dispatch`). The instance
//! id is `<endpoint>/<template>`: fv-deploy always creates its own template
//! and deletes it with the endpoint.

use std::time::Duration;

use serde_json::{json, Map, Value};

use super::RunpodApi;
use crate::boot::sys::Method;
use crate::provider::{
    Account, ApiCall, Cursor, Instance, InstanceId, InstanceState, Launch, LogPage, Offer,
    OfferQuery, Probe, Provider, ProviderError, ProviderKind, Reach, ReachOptions,
};

/// DELETE attempts on an endpoint (`runpod-endpoint.sh down`: 6, `i * 5` s apart).
pub const SERVERLESS_DELETE_ATTEMPTS: u32 = 6;

/// The serverless template of a launch (`template_payload`,
/// `fixtures/deploy/runpod-template.json`) in v2's shape: `imageName` →
/// `image`, `isServerless` → `serverless`, `containerDiskInGb` → `disk`,
/// `dockerEntrypoint` / `dockerStartCmd` → `entrypoint` / `cmd`,
/// `volumeInGb` → `mounts.persistent {size, path}` only when not 0 (v2 has
/// no "no disk" value, [`super::pod::mounts_of`]).
pub fn template_body(l: &Launch) -> Value {
    let mut m = Map::new();
    m.insert("name".into(), json!(l.name));
    m.insert("image".into(), json!(l.image));
    m.insert("serverless".into(), json!(true));
    m.insert("disk".into(), json!(l.disk.container_gb));
    if l.disk.volume_gb > 0 {
        m.insert(
            "mounts".into(),
            json!({"persistent": {"size": l.disk.volume_gb, "path": crate::launch::POD_VOLUME_MOUNT}}),
        );
    }
    if !l.entrypoint.is_empty() {
        m.insert("entrypoint".into(), json!(l.entrypoint));
    }
    if !l.args.is_empty() {
        m.insert("cmd".into(), json!(l.args));
    }
    m.insert("env".into(), json!(l.env));
    Value::Object(m)
}

/// A v2 endpoint's GPU selection. v1's `gpuTypeIds` named GPU **types**
/// (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, the ids the price quotes
/// use); v2's `gpu.pools` names serverless **pools** (`ADA_24`,
/// `BLACKWELL_96`: `GET /catalog/gpus` `pool`) and narrows one with
/// `excludedTypes`, with no inclusive allowlist. A launch's types are
/// resolved against the catalog ([`EndpointGpu::resolve`]): a type is its
/// pool with the pool's other types excluded; a name that is itself a pool
/// is the whole pool.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EndpointGpu {
    pub pools: Vec<String>,
    pub excluded_types: Vec<String>,
}

impl EndpointGpu {
    /// The selection a launch's GPU names make over the catalog's GPU types
    /// (`GET /catalog/gpus` `gpus[] {id, pool}`): a name equal to a pool id
    /// selects the pool whole; a name equal to a type id selects its pool
    /// and excludes the pool's other types (unless they are named too); a
    /// name that is neither, or a type outside every serverless pool, is
    /// refused (`Unsupported`), never silently widened.
    pub fn resolve(names: &[String], catalog: &[Value]) -> Result<Self, ProviderError> {
        fn pool_of(v: &Value) -> Option<&str> {
            v.get("pool")
                .and_then(Value::as_str)
                .filter(|p| !p.is_empty())
        }
        fn id_of(v: &Value) -> Option<&str> {
            v.get("id").and_then(Value::as_str)
        }
        let mut pools: Vec<String> = Vec::new();
        // Pools named as such: every type of theirs stays.
        let mut whole: Vec<String> = Vec::new();
        for n in names {
            if catalog.iter().any(|g| pool_of(g) == Some(n.as_str())) {
                if !pools.contains(n) {
                    pools.push(n.clone());
                }
                whole.push(n.clone());
                continue;
            }
            let Some(g) = catalog.iter().find(|g| id_of(g) == Some(n.as_str())) else {
                return Err(ProviderError::Unsupported(format!(
                    "serverless GPU {n}: neither a GPU type nor a pool of GET /catalog/gpus"
                )));
            };
            let Some(p) = pool_of(g) else {
                return Err(ProviderError::Unsupported(format!(
                    "serverless GPU {n}: not in any serverless pool (GET /catalog/gpus pool is null)"
                )));
            };
            if !pools.iter().any(|x| x == p) {
                pools.push(p.to_owned());
            }
        }
        let mut excluded: Vec<String> = catalog
            .iter()
            .filter(|g| {
                pool_of(g)
                    .is_some_and(|p| pools.iter().any(|x| x == p) && !whole.iter().any(|w| w == p))
            })
            .filter_map(id_of)
            .filter(|id| !names.iter().any(|n| n == id))
            .map(str::to_owned)
            .collect();
        excluded.sort();
        excluded.dedup();
        Ok(Self {
            pools,
            excluded_types: excluded,
        })
    }

    /// The selection a plan shows without the catalog (`fv-deploy plan`
    /// makes no request): `runpod-endpoint.sh gpu_pools_json`'s fallback,
    /// the `ADA_24` pool, so the plan's body is the script's. A create never
    /// uses it: [`Self::resolve`] over the catalog, or a refusal.
    pub fn unresolved(_names: &[String]) -> Self {
        Self {
            pools: vec!["ADA_24".into()],
            excluded_types: Vec::new(),
        }
    }
}

/// The queue endpoint of a launch on template `template_id`
/// (`endpoint_payload`, `fixtures/deploy/runpod-endpoint.json`) in v2's
/// shape: `type: QUEUE` (required), `gpuTypeIds` + `gpuCount` +
/// `minCudaVersion` + `allowedCudaVersions` → `gpu: {pools, excludedTypes,
/// count, minCudaVersion, allowedCudaVersions}` (`gpu`: [`EndpointGpu`]),
/// `networkVolumeId` → `networkVolumes: [id]`, `workersMin` / `workersMax` /
/// `idleTimeout` → `workers: {min, max, idleTimeout}` (`idleTimeout` only
/// with `QUEUE_DELAY`: v2 refuses it with `REQUEST_COUNT`), `scalerType` +
/// `scalerValue` → `scaling: {type, queueDelay | requestCount}`, `flashboot`
/// `false` / `true` → `"OFF"` / `"FLASHBOOT"`, `executionTimeoutMs` →
/// `timeout` (ms); `computeType` gone.
pub fn endpoint_body(l: &Launch, template_id: &str, gpu: &EndpointGpu) -> Value {
    let s = l
        .serverless
        .clone()
        .unwrap_or(crate::launch::default_serverless());
    let mut m = Map::new();
    m.insert("name".into(), json!(l.name));
    m.insert("templateId".into(), json!(template_id));
    m.insert("type".into(), json!("QUEUE"));
    let mut g = Map::new();
    g.insert("pools".into(), json!(gpu.pools));
    if !gpu.excluded_types.is_empty() {
        g.insert("excludedTypes".into(), json!(gpu.excluded_types));
    }
    g.insert("count".into(), json!(l.gpu.count));
    if !l.gpu.allowed_cuda.is_empty() {
        g.insert("allowedCudaVersions".into(), json!(l.gpu.allowed_cuda));
    } else if let Some(c) = &l.gpu.min_cuda {
        g.insert("minCudaVersion".into(), json!(c));
    }
    m.insert("gpu".into(), Value::Object(g));
    if let Some(v) = &l.placement.volume {
        m.insert("networkVolumes".into(), json!([v.id]));
    }
    if !l.placement.data_centers.is_empty() {
        m.insert("dataCenterIds".into(), json!(l.placement.data_centers));
    }
    let request_count = s.scaler_type == "REQUEST_COUNT";
    let mut w = Map::new();
    w.insert("min".into(), json!(s.workers_min));
    w.insert("max".into(), json!(s.workers_max));
    if !request_count {
        w.insert("idleTimeout".into(), json!(s.idle_timeout_s));
    }
    m.insert("workers".into(), Value::Object(w));
    m.insert(
        "scaling".into(),
        if request_count {
            json!({"type": "REQUEST_COUNT", "requestCount": s.scaler_value})
        } else {
            json!({"type": s.scaler_type, "queueDelay": s.scaler_value})
        },
    );
    m.insert(
        "flashboot".into(),
        json!(if s.flashboot { "FLASHBOOT" } else { "OFF" }),
    );
    m.insert("timeout".into(), json!(s.execution_timeout_ms));
    Value::Object(m)
}

/// `<endpoint>/<template>` → (endpoint, template).
pub fn split_id(id: &InstanceId) -> (&str, Option<&str>) {
    match id.as_str().split_once('/') {
        Some((e, t)) if !t.is_empty() => (e, Some(t)),
        Some((e, _)) => (e, None),
        None => (id.as_str(), None),
    }
}

fn id_of(v: &Value, what: &str) -> Result<String, ProviderError> {
    v.get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| ProviderError::Decode {
            what: what.into(),
            why: "no id in the answer".into(),
        })
}

/// Runpod serverless queue endpoints.
#[derive(Debug)]
pub struct Serverless<'a> {
    pub api: RunpodApi<'a>,
}

impl<'a> Serverless<'a> {
    pub fn new(api: RunpodApi<'a>) -> Self {
        Self { api }
    }

    fn delete_template(&self, tpl: &str) -> Result<(), ProviderError> {
        let url = format!("{}/templates/{tpl}", self.api.urls.rest);
        let r = self.api.raw(
            Method::Delete,
            &url,
            None,
            Duration::from_secs(60),
            "DELETE template",
        )?;
        if r.status == 404 {
            return Ok(());
        }
        self.api.json_or_error(r, "DELETE template").map(|_| ())
    }
}

impl Provider for Serverless<'_> {
    fn kind(&self) -> ProviderKind {
        ProviderKind::RunpodServerless
    }

    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>, ProviderError> {
        q.gpu_types
            .iter()
            .map(|g| {
                self.api
                    .quote(self.kind(), g, q.cloud, q.data_center.as_deref())
            })
            .collect()
    }

    fn payload(&self, launch: &Launch) -> Result<Vec<ApiCall>, ProviderError> {
        Ok(vec![
            ApiCall {
                method: "POST".into(),
                url: format!("{}/templates", self.api.urls.rest),
                body: template_body(launch),
            },
            ApiCall {
                method: "POST".into(),
                url: format!("{}/serverless", self.api.urls.rest),
                body: endpoint_body(
                    launch,
                    "<template id>",
                    &EndpointGpu::unresolved(&launch.gpu.types),
                ),
            },
        ])
    }

    fn create(&self, launch: &Launch) -> Result<Instance, ProviderError> {
        // The GPU types' pools, before anything is created: an unknown type
        // refuses the launch with nothing to sweep.
        let catalog = self.api.list("/catalog/gpus", "gpus", "GET catalog gpus")?;
        let gpu = EndpointGpu::resolve(&launch.gpu.types, &catalog)?;
        let t = self.api.rest(
            Method::Post,
            "/templates",
            Some(&template_body(launch)),
            "POST templates",
        )?;
        let tpl = id_of(&t, "POST templates")?;
        let e = match self.api.rest(
            Method::Post,
            "/serverless",
            Some(&endpoint_body(launch, &tpl, &gpu)),
            "POST serverless",
        ) {
            Ok(e) => e,
            Err(err) => {
                // Refused for sure: the template alone costs nothing, but it is
                // ours, remove it. Otherwise the endpoint may exist: the
                // caller's sweep by name removes both.
                if err.is_definite_refusal() {
                    let _ = self.delete_template(&tpl);
                }
                return Err(err);
            }
        };
        // A 2xx without an id: the endpoint may exist (the caller sweeps).
        let ep = id_of(&e, "POST serverless")?;
        Ok(Instance {
            id: InstanceId::new(format!("{ep}/{tpl}")),
            provider: self.kind(),
            name: launch.name.clone(),
            state: InstanceState::Running,
            url: Some(format!("{}/{ep}", self.api.urls.queue)),
            gpu: launch.gpu.types.first().cloned(),
            data_center: launch.placement.data_centers.first().cloned(),
            usd_per_hr: None,
            created_at: Some(self.api.clock().now()),
            memory_gb: None,
            vcpus: None,
        })
    }

    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError> {
        let (ep, _) = split_id(id);
        let url = format!("{}/serverless/{ep}", self.api.urls.rest);
        let r = self.api.raw(
            Method::Get,
            &url,
            None,
            Duration::from_secs(30),
            "GET endpoint",
        )?;
        // Only a 404 is "gone" (the backstop must not stand down on a 400 or
        // an answer without an id).
        if r.status == 404 {
            return Ok(InstanceState::Terminated);
        }
        if r.status == 400 {
            return Ok(InstanceState::Unknown);
        }
        let v = self.api.json_or_error(r, "GET endpoint")?;
        Ok(if v.get("id").and_then(Value::as_str).is_some() {
            InstanceState::Running
        } else {
            InstanceState::Unknown
        })
    }

    fn logs(&self, id: &InstanceId, _since: Option<&Cursor>) -> Result<LogPage, ProviderError> {
        let (ep, _) = split_id(id);
        let url = format!("{}/{ep}/health", self.api.urls.queue);
        let r = self.api.raw(
            Method::Get,
            &url,
            None,
            Duration::from_secs(30),
            "GET health",
        )?;
        let v = self.api.json_or_error(r, "GET health")?;
        Ok(LogPage {
            lines: vec![format!("[health] {v}")],
            next: None,
        })
    }

    /// `runpod-endpoint.sh down`: workers to 0 (v2 `workers: {min, max}`),
    /// then DELETE the endpoint (6 attempts, `i * 5` s apart; gone is fine),
    /// then our template.
    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError> {
        let (ep, tpl) = split_id(id);
        let url = format!("{}/serverless/{ep}", self.api.urls.rest);
        let _ = self.api.raw(
            Method::Patch,
            &url,
            Some(&json!({"workers": {"min": 0, "max": 0}})),
            Duration::from_secs(60),
            "PATCH endpoint",
        );
        let mut last = None;
        for i in 1..=SERVERLESS_DELETE_ATTEMPTS {
            match self.api.raw(
                Method::Delete,
                &url,
                None,
                Duration::from_secs(60),
                "DELETE endpoint",
            ) {
                Ok(r) if r.status == 404 || (200..300).contains(&r.status) => {
                    last = None;
                    break;
                }
                Ok(r) => last = Some(self.api.json_or_error(r, "DELETE endpoint").err()),
                Err(e) => last = Some(Some(e)),
            }
            if i < SERVERLESS_DELETE_ATTEMPTS {
                self.api
                    .clock()
                    .sleep(Duration::from_secs(u64::from(i) * 5));
            }
        }
        if let Some(Some(e)) = last {
            return Err(e);
        }
        match tpl {
            Some(t) => self.delete_template(t),
            None => Ok(()),
        }
    }

    fn account(&self) -> Result<Account, ProviderError> {
        self.api.account(self.kind())
    }

    /// Endpoints named `name` (with their templates), then templates named
    /// `name` that no endpoint uses any more (a create whose endpoint answer
    /// was lost). A v2 endpoint does not name its template (v1's
    /// `templateId`): fv-deploy's template carries the endpoint's name, so
    /// the second pass removes it.
    fn sweep(
        &self,
        name: &str,
        keep: Option<&InstanceId>,
    ) -> Result<Vec<InstanceId>, ProviderError> {
        let (keep_ep, keep_tpl) = keep.map(split_id).unwrap_or(("", None));
        let mut gone = Vec::new();
        let mut last = None;
        let eps = self
            .api
            .list("/serverless", "endpoints", "GET serverless")?;
        for e in &eps {
            let Some(ep) = e.get("id").and_then(Value::as_str) else {
                continue;
            };
            if e.get("name").and_then(Value::as_str) != Some(name) || ep == keep_ep {
                continue;
            }
            let id = match e.get("templateId").and_then(Value::as_str) {
                Some(t) if !t.is_empty() && Some(t) != keep_tpl => {
                    InstanceId::new(format!("{ep}/{t}"))
                }
                _ => InstanceId::new(ep),
            };
            match self.destroy(&id) {
                Ok(()) => gone.push(id),
                Err(err) => last = Some(err),
            }
        }
        let tpls = self.api.list("/templates", "templates", "GET templates")?;
        for t in &tpls {
            let Some(tid) = t.get("id").and_then(Value::as_str) else {
                continue;
            };
            if t.get("name").and_then(Value::as_str) != Some(name) || Some(tid) == keep_tpl {
                continue;
            }
            match self.delete_template(tid) {
                Ok(()) => gone.push(InstanceId::new(format!("template:{tid}"))),
                Err(err) => last = Some(err),
            }
        }
        match last {
            Some(e) => Err(e),
            None => Ok(gone),
        }
    }

    /// The endpoint's price is per worker-second, not reported at create:
    /// the price cap checks the quote only (docs/adr/0018-fv-deploy.md §5.3).
    fn bills_at_create(&self) -> bool {
        false
    }

    fn reach<'b>(
        &'b self,
        instance: &Instance,
        opts: &ReachOptions<'_>,
    ) -> Result<Box<dyn Reach + 'b>, ProviderError> {
        let (ep, _) = split_id(&instance.id);
        Ok(Box::new(QueueReach {
            api: &self.api,
            endpoint: ep.to_owned(),
            execution_timeout_ms: u64::try_from(opts.call_timeout.as_millis()).unwrap_or(u64::MAX),
            info_job: None,
            poll: Duration::from_secs(2),
        }))
    }
}

/// A queue endpoint's fv-serve, one native job per call.
pub struct QueueReach<'a> {
    pub api: &'a RunpodApi<'a>,
    pub endpoint: String,
    pub execution_timeout_ms: u64,
    /// The cold `info` job the readiness poll waits on.
    pub info_job: Option<String>,
    pub poll: Duration,
}

const TERMINAL: [&str; 4] = ["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"];

impl QueueReach<'_> {
    fn submit(&self, input: Value) -> Result<String, ProviderError> {
        let url = format!("{}/{}/run", self.api.urls.queue, self.endpoint);
        let body =
            json!({"input": input, "policy": {"executionTimeout": self.execution_timeout_ms}});
        let r = self.api.raw(
            Method::Post,
            &url,
            Some(&body),
            Duration::from_secs(60),
            "queue run",
        )?;
        let v = self.api.json_or_error(r, "queue run")?;
        id_of(&v, "queue run")
    }

    fn status(&self, job: &str) -> Result<Value, ProviderError> {
        let url = format!("{}/{}/status/{job}", self.api.urls.queue, self.endpoint);
        let r = self.api.raw(
            Method::Get,
            &url,
            None,
            Duration::from_secs(60),
            "queue status",
        )?;
        self.api.json_or_error(r, "queue status")
    }
}

fn job_status(v: &Value) -> &str {
    v.get("status").and_then(Value::as_str).unwrap_or("?")
}

impl Reach for QueueReach<'_> {
    /// Ready once a cold `info` job has completed (a queue endpoint with
    /// `workers.min` 0 starts a worker only for a job).
    fn ping(&mut self) -> Probe {
        let job = match &self.info_job {
            Some(j) => j.clone(),
            None => match self.submit(json!({"kind": "info"})) {
                Ok(j) => {
                    self.info_job = Some(j);
                    return Probe::Loading;
                }
                Err(e) => return Probe::Down(e.to_string()),
            },
        };
        match self.status(&job) {
            Ok(v) => match job_status(&v) {
                "COMPLETED" => Probe::Ready,
                s if TERMINAL.contains(&s) => {
                    self.info_job = None;
                    Probe::Down(format!("info job {job}: {s}"))
                }
                _ => Probe::Loading,
            },
            Err(e) => Probe::Down(e.to_string()),
        }
    }

    fn call(
        &mut self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        stop: &dyn Fn() -> bool,
    ) -> Result<(u16, Value), ProviderError> {
        let mut input = json!({"kind": "http", "method": method.as_str(), "path": path});
        if let Some(b) = body {
            input["body"] = b.clone();
        }
        if method == Method::Post {
            input["wait"] = json!(true);
        }
        let job = self.submit(input)?;
        let clock = self.api.clock();
        let deadline = clock.now() + self.execution_timeout_ms as f64 / 1000.0;
        loop {
            let v = self.status(&job)?;
            let s = job_status(&v);
            if s == "COMPLETED" {
                let out = v.get("output").cloned().unwrap_or(Value::Null);
                let status = out
                    .get("status")
                    .and_then(Value::as_u64)
                    .and_then(|x| u16::try_from(x).ok())
                    .unwrap_or(200);
                return Ok((status, out.get("body").cloned().unwrap_or(Value::Null)));
            }
            if TERMINAL.contains(&s) {
                return Err(ProviderError::Api {
                    what: format!("queue job {job} ({} {path})", method.as_str()),
                    status: 0,
                    body: format!("{s}: {}", v.get("error").cloned().unwrap_or(Value::Null)),
                });
            }
            if stop() {
                return Err(ProviderError::Unsupported(format!(
                    "queue job {job} ({} {path}): interrupted",
                    method.as_str()
                )));
            }
            if clock.now() > deadline {
                return Err(ProviderError::Api {
                    what: format!("queue job {job} ({} {path})", method.as_str()),
                    status: 0,
                    body: format!("still {s} at the execution timeout"),
                });
            }
            clock.sleep(self.poll);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Vec<Value> {
        vec![
            json!({"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "pool": "BLACKWELL_96"}),
            json!({"id": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "pool": "BLACKWELL_96"}),
            json!({"id": "NVIDIA L40S", "pool": "ADA_48_PRO"}),
            json!({"id": "NVIDIA L40", "pool": "ADA_48_PRO"}),
            json!({"id": "NVIDIA H200", "pool": "H200"}),
            json!({"id": "NVIDIA A2", "pool": null}),
        ]
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn gpu_types_resolve_to_their_pools_with_the_other_types_excluded() {
        let g = EndpointGpu::resolve(
            &names(&[
                "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "NVIDIA H200",
            ]),
            &catalog(),
        )
        .unwrap_or_default();
        assert_eq!(g.pools, names(&["BLACKWELL_96", "H200"]));
        assert_eq!(
            g.excluded_types,
            names(&["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"])
        );
        // Both types of a pool: nothing excluded; a pool by name: whole.
        let g = EndpointGpu::resolve(&names(&["NVIDIA L40S", "NVIDIA L40"]), &catalog())
            .unwrap_or_default();
        assert_eq!(
            (g.pools, g.excluded_types),
            (names(&["ADA_48_PRO"]), vec![])
        );
        let g = EndpointGpu::resolve(&names(&["ADA_48_PRO"]), &catalog()).unwrap_or_default();
        assert_eq!(
            (g.pools, g.excluded_types),
            (names(&["ADA_48_PRO"]), vec![])
        );
        // A type outside every pool, or unknown: refused.
        assert!(EndpointGpu::resolve(&names(&["NVIDIA A2"]), &catalog()).is_err());
        assert!(EndpointGpu::resolve(&names(&["NVIDIA B300"]), &catalog()).is_err());
        assert_eq!(
            EndpointGpu::unresolved(&names(&["NVIDIA H200"])).pools,
            names(&["ADA_24"])
        );
    }

    #[test]
    fn the_endpoint_body_is_v2s_shape() {
        let mut l = crate::launch::script_endpoint(&crate::launch::ScriptEndpoint {
            name: "n",
            image: "img",
            gpus: &["NVIDIA H200"],
            config: "/etc/fv/x.toml",
            inline: None,
            volume: "vol",
            data_center: "EU-RO-1",
            guards: crate::guard::default_guards(),
        });
        let gpu = EndpointGpu {
            pools: names(&["H200"]),
            excluded_types: vec![],
        };
        let b = endpoint_body(&l, "tpl", &gpu);
        assert_eq!(b["type"], "QUEUE");
        assert_eq!(
            b["gpu"],
            json!({"pools": ["H200"], "count": 1, "minCudaVersion": "13.0"})
        );
        assert_eq!(b["networkVolumes"], json!(["vol"]));
        assert_eq!(b["workers"], json!({"min": 0, "max": 1, "idleTimeout": 5}));
        assert_eq!(
            b["scaling"],
            json!({"type": "QUEUE_DELAY", "queueDelay": 1})
        );
        assert_eq!(
            (b["flashboot"].as_str(), b["timeout"].as_u64()),
            (Some("OFF"), Some(1_800_000))
        );
        assert!(b.get("computeType").is_none() && b.get("gpuTypeIds").is_none());
        let mut s = crate::launch::default_serverless();
        s.scaler_type = "REQUEST_COUNT".into();
        s.scaler_value = 4;
        s.flashboot = true;
        l.serverless = Some(s);
        let b = endpoint_body(&l, "tpl", &gpu);
        assert_eq!(
            b["scaling"],
            json!({"type": "REQUEST_COUNT", "requestCount": 4})
        );
        assert_eq!(b["workers"], json!({"min": 0, "max": 1}));
        assert_eq!(b["flashboot"], "FLASHBOOT");
        // An exact CUDA set replaces the floor (mutually exclusive in v2).
        l.gpu.allowed_cuda = vec!["12.8".into()];
        let b = endpoint_body(&l, "tpl", &gpu);
        assert_eq!(b["gpu"]["allowedCudaVersions"], json!(["12.8"]));
        assert!(b["gpu"].get("minCudaVersion").is_none());
    }
}
