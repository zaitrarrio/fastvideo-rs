//! A Runpod serverless queue endpoint: a template (REST v1 `POST /templates`)
//! and an endpoint on it (REST v1 `POST /endpoints`), the bodies
//! `runpod-endpoint.sh`'s `template_payload` / `endpoint_payload` build;
//! reached through the queue API, one native job per call (the envelope of
//! `crate::dispatch`). The instance id is `<endpoint>/<template>`: fv-deploy
//! always creates its own template and deletes it with the endpoint.

use std::time::Duration;

use serde_json::{json, Map, Value};

use super::RunpodApi;
use crate::boot::sys::Method;
use crate::provider::{
    Account, ApiCall, Cursor, Instance, InstanceId, InstanceState, Launch, LogPage, Offer,
    OfferQuery, Probe, Provider, ProviderError, ProviderKind, Reach,
};

/// DELETE attempts on an endpoint (`runpod-endpoint.sh down`: 6, `i * 5` s apart).
pub const SERVERLESS_DELETE_ATTEMPTS: u32 = 6;

/// The serverless template of a launch (`template_payload`,
/// `fixtures/deploy/runpod-template.json`).
pub fn template_body(l: &Launch) -> Value {
    let mut m = Map::new();
    m.insert("name".into(), json!(l.name));
    m.insert("imageName".into(), json!(l.image));
    m.insert("isServerless".into(), json!(true));
    m.insert("containerDiskInGb".into(), json!(l.disk.container_gb));
    m.insert("volumeInGb".into(), json!(l.disk.volume_gb));
    if !l.entrypoint.is_empty() {
        m.insert("dockerEntrypoint".into(), json!(l.entrypoint));
    }
    if !l.args.is_empty() {
        m.insert("dockerStartCmd".into(), json!(l.args));
    }
    m.insert("env".into(), json!(l.env));
    Value::Object(m)
}

/// The queue endpoint of a launch on template `template_id`
/// (`endpoint_payload`, `fixtures/deploy/runpod-endpoint.json`).
pub fn endpoint_body(l: &Launch, template_id: &str) -> Value {
    let s = l
        .serverless
        .clone()
        .unwrap_or(crate::launch::default_serverless());
    let mut m = Map::new();
    m.insert("name".into(), json!(l.name));
    m.insert("templateId".into(), json!(template_id));
    m.insert("computeType".into(), json!("GPU"));
    m.insert("gpuTypeIds".into(), json!(l.gpu.types));
    m.insert("gpuCount".into(), json!(l.gpu.count));
    if let Some(v) = &l.placement.volume {
        m.insert("networkVolumeId".into(), json!(v.id));
    }
    if !l.placement.data_centers.is_empty() {
        m.insert("dataCenterIds".into(), json!(l.placement.data_centers));
    }
    m.insert("workersMin".into(), json!(s.workers_min));
    m.insert("workersMax".into(), json!(s.workers_max));
    m.insert("idleTimeout".into(), json!(s.idle_timeout_s));
    m.insert("flashboot".into(), json!(s.flashboot));
    m.insert("executionTimeoutMs".into(), json!(s.execution_timeout_ms));
    m.insert("scalerType".into(), json!(s.scaler_type));
    m.insert("scalerValue".into(), json!(s.scaler_value));
    if let Some(c) = &l.gpu.min_cuda {
        m.insert("minCudaVersion".into(), json!(c));
    }
    if !l.gpu.allowed_cuda.is_empty() {
        m.insert("allowedCudaVersions".into(), json!(l.gpu.allowed_cuda));
    }
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
                url: format!("{}/endpoints", self.api.urls.rest),
                body: endpoint_body(launch, "<template id>"),
            },
        ])
    }

    fn create(&self, launch: &Launch) -> Result<Instance, ProviderError> {
        let t = self.api.rest(
            Method::Post,
            "/templates",
            Some(&template_body(launch)),
            "POST templates",
        )?;
        let tpl = id_of(&t, "POST templates")?;
        let e = match self.api.rest(
            Method::Post,
            "/endpoints",
            Some(&endpoint_body(launch, &tpl)),
            "POST endpoints",
        ) {
            Ok(e) => e,
            Err(err) => {
                // The template alone costs nothing, but it is ours: remove it.
                let _ = self.delete_template(&tpl);
                return Err(err);
            }
        };
        let ep = match id_of(&e, "POST endpoints") {
            Ok(ep) => ep,
            Err(err) => {
                let _ = self.delete_template(&tpl);
                return Err(err);
            }
        };
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
        })
    }

    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError> {
        let (ep, _) = split_id(id);
        let url = format!("{}/endpoints/{ep}", self.api.urls.rest);
        let r = self.api.raw(
            Method::Get,
            &url,
            None,
            Duration::from_secs(30),
            "GET endpoint",
        )?;
        if r.status == 404 || r.status == 400 {
            return Ok(InstanceState::Terminated);
        }
        let v = self.api.json_or_error(r, "GET endpoint")?;
        Ok(if v.get("id").and_then(Value::as_str).is_some() {
            InstanceState::Running
        } else {
            InstanceState::Terminated
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

    /// `runpod-endpoint.sh down`: workers to 0, then DELETE the endpoint
    /// (6 attempts, `i * 5` s apart; gone is fine), then our template.
    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError> {
        let (ep, tpl) = split_id(id);
        let url = format!("{}/endpoints/{ep}", self.api.urls.rest);
        let _ = self.api.raw(
            Method::Patch,
            &url,
            Some(&json!({"workersMin": 0, "workersMax": 0})),
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

    fn reach<'b>(
        &'b self,
        instance: &Instance,
        _key: Option<&str>,
    ) -> Result<Box<dyn Reach + 'b>, ProviderError> {
        let (ep, _) = split_id(&instance.id);
        Ok(Box::new(QueueReach {
            api: &self.api,
            endpoint: ep.to_owned(),
            execution_timeout_ms: 1_800_000,
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
    /// `workersMin` 0 starts a worker only for a job).
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
