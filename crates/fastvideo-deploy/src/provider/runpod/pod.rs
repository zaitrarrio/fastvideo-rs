//! A Runpod GPU pod: REST v1 `POST /pods` with the body `runpod-pod.sh`'s
//! `payload()` builds, reached through `https://<id>-8000.proxy.runpod.net`.

use std::time::Duration;

use serde_json::{json, Map, Value};

use super::RunpodApi;
use crate::boot::sys::{Method, Request};
use crate::provider::{
    body_json, parse_usd, scrub, Account, ApiCall, Cursor, Instance, InstanceId, InstanceState,
    Launch, LogPage, Offer, OfferQuery, Probe, Provider, ProviderError, ProviderKind, Reach,
    ReachOptions,
};

/// The `PodCreateInput` of a launch: `runpod-pod.sh payload()` field for
/// field (`fixtures/deploy/runpod-pod.json`), fv-control's
/// `workerCreatePayload` for the placement fields.
pub fn create_body(l: &Launch) -> Value {
    let mut m = Map::new();
    m.insert("name".into(), json!(l.name));
    m.insert("imageName".into(), json!(l.image));
    m.insert("cloudType".into(), json!(l.placement.cloud.as_str()));
    m.insert("computeType".into(), json!("GPU"));
    m.insert("gpuTypeIds".into(), json!(l.gpu.types));
    m.insert("gpuCount".into(), json!(l.gpu.count));
    m.insert("containerDiskInGb".into(), json!(l.disk.container_gb));
    m.insert("volumeInGb".into(), json!(l.disk.volume_gb));
    if !l.ports.is_empty() {
        m.insert("ports".into(), json!(l.ports));
    }
    if !l.entrypoint.is_empty() {
        m.insert("dockerEntrypoint".into(), json!(l.entrypoint));
    }
    if !l.args.is_empty() {
        m.insert("dockerStartCmd".into(), json!(l.args));
    }
    m.insert("env".into(), json!(l.env));
    if let Some(v) = &l.placement.volume {
        m.insert("networkVolumeId".into(), json!(v.id));
        m.insert("volumeMountPath".into(), json!(v.mount_path));
    }
    if !l.placement.data_centers.is_empty() {
        m.insert("dataCenterIds".into(), json!(l.placement.data_centers));
    }
    if !l.gpu.allowed_cuda.is_empty() {
        m.insert("allowedCudaVersions".into(), json!(l.gpu.allowed_cuda));
    }
    Value::Object(m)
}

/// Runpod pods.
#[derive(Debug)]
pub struct Pod<'a> {
    pub api: RunpodApi<'a>,
}

impl<'a> Pod<'a> {
    pub fn new(api: RunpodApi<'a>) -> Self {
        Self { api }
    }
}

/// `desiredStatus` → [`InstanceState`].
pub fn pod_state(desired: &str) -> InstanceState {
    match desired {
        "RUNNING" => InstanceState::Running,
        "CREATED" | "PENDING" | "STARTING" => InstanceState::Pending,
        "EXITED" | "STOPPED" => InstanceState::Exited,
        "TERMINATED" | "TERMINATING" => InstanceState::Terminated,
        _ => InstanceState::Unknown,
    }
}

/// `"<container lines>:<system lines>"` already returned.
fn parse_cursor(c: Option<&Cursor>) -> (usize, usize) {
    let Some(c) = c else { return (0, 0) };
    let mut it = c.0.split(':').map(|x| x.parse::<usize>().unwrap_or(0));
    (it.next().unwrap_or(0), it.next().unwrap_or(0))
}

fn str_lines(v: &Value, k: &str) -> Vec<String> {
    v.get(k)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|x| x.as_str().map_or_else(|| x.to_string(), str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

impl Provider for Pod<'_> {
    fn kind(&self) -> ProviderKind {
        ProviderKind::RunpodPod
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
        Ok(vec![ApiCall {
            method: "POST".into(),
            url: format!("{}/pods", self.api.urls.rest),
            body: create_body(launch),
        }])
    }

    fn create(&self, launch: &Launch) -> Result<Instance, ProviderError> {
        let v = self.api.rest(
            Method::Post,
            "/pods",
            Some(&create_body(launch)),
            "POST pods",
        )?;
        let id = v
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ProviderError::Decode {
                what: "POST pods".into(),
                why: format!("no id in {}", scrub(&v.to_string(), "", 300)),
            })?;
        let dc = v
            .pointer("/machine/dataCenterId")
            .or_else(|| v.get("dataCenterId"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| launch.placement.data_centers.first().cloned());
        Ok(Instance {
            id: InstanceId::new(id),
            provider: self.kind(),
            name: launch.name.clone(),
            state: InstanceState::Pending,
            url: Some(self.api.urls.pod_url(id)),
            gpu: launch.gpu.types.first().cloned(),
            data_center: dc,
            // A number or a numeric string; missing or unparseable is `None`
            // and fails the billed-price check.
            usd_per_hr: parse_usd(v.get("costPerHr")),
            created_at: Some(self.api.clock().now()),
        })
    }

    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError> {
        let url = format!("{}/pods/{}", self.api.urls.rest, id);
        let r = self
            .api
            .raw(Method::Get, &url, None, Duration::from_secs(30), "GET pod")?;
        // Only a 404 is "gone"; a 400 or an answer without an id is not
        // proof (the backstop must not stand down on it).
        if r.status == 404 {
            return Ok(InstanceState::Terminated);
        }
        if r.status == 400 {
            return Ok(InstanceState::Unknown);
        }
        let v = self.api.json_or_error(r, "GET pod")?;
        if v.get("id").and_then(Value::as_str).is_none() {
            return Ok(InstanceState::Unknown);
        }
        Ok(pod_state(
            v.get("desiredStatus").and_then(Value::as_str).unwrap_or(""),
        ))
    }

    fn logs(&self, id: &InstanceId, since: Option<&Cursor>) -> Result<LogPage, ProviderError> {
        let url = format!("{}/pod/{}/logs", self.api.urls.hapi, id);
        let r = self.api.raw(
            Method::Get,
            &url,
            None,
            Duration::from_secs(30),
            "GET pod logs",
        )?;
        let v = self.api.json_or_error(r, "GET pod logs")?;
        let (c0, s0) = parse_cursor(since);
        let container = str_lines(&v, "container");
        let system = str_lines(&v, "system");
        // A restarted container starts its log again: then everything is new.
        let c0 = if c0 > container.len() { 0 } else { c0 };
        let s0 = if s0 > system.len() { 0 } else { s0 };
        let mut lines: Vec<String> = system[s0..]
            .iter()
            .map(|l| format!("[system] {l}"))
            .collect();
        lines.extend(container[c0..].iter().cloned());
        Ok(LogPage {
            lines,
            next: Some(Cursor(format!("{}:{}", container.len(), system.len()))),
        })
    }

    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError> {
        let url = format!("{}/pods/{}", self.api.urls.rest, id);
        let r = self.api.raw(
            Method::Delete,
            &url,
            None,
            Duration::from_secs(60),
            "DELETE pod",
        )?;
        if r.status == 404 {
            return Ok(());
        }
        self.api.json_or_error(r, "DELETE pod").map(|_| ())
    }

    fn account(&self) -> Result<Account, ProviderError> {
        self.api.account(self.kind())
    }

    fn sweep(
        &self,
        name: &str,
        keep: Option<&InstanceId>,
    ) -> Result<Vec<InstanceId>, ProviderError> {
        let v = self.api.rest(Method::Get, "/pods", None, "GET pods")?;
        let mut gone = Vec::new();
        let mut last = None;
        for p in v.as_array().map(Vec::as_slice).unwrap_or_default() {
            let Some(id) = p.get("id").and_then(Value::as_str) else {
                continue;
            };
            let named = p.get("name").and_then(Value::as_str) == Some(name);
            let dead = p.get("desiredStatus").and_then(Value::as_str) == Some("TERMINATED");
            if !named || dead || keep.is_some_and(|k| k.as_str() == id) {
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

    fn bills_at_create(&self) -> bool {
        true
    }

    fn reach<'b>(
        &'b self,
        instance: &Instance,
        opts: &ReachOptions<'_>,
    ) -> Result<Box<dyn Reach + 'b>, ProviderError> {
        let base = instance
            .url
            .clone()
            .unwrap_or_else(|| self.api.urls.pod_url(instance.id.as_str()));
        Ok(Box::new(PodReach {
            http: self.api.http(),
            base,
            key: opts.key.map(str::to_owned),
        }))
    }
}

/// A pod's fv-serve through the HTTPS proxy, with the run's API key.
pub struct PodReach<'a> {
    pub http: &'a dyn crate::boot::sys::Http,
    pub base: String,
    pub key: Option<String>,
}

impl Reach for PodReach<'_> {
    fn ping(&mut self) -> Probe {
        let req = Request::new(
            Method::Get,
            format!("{}/ping", self.base),
            Duration::from_secs(15),
        );
        match self.http.send(&req) {
            Ok(r) if r.status == 200 => Probe::Ready,
            Ok(r) if r.status == 204 => Probe::Loading,
            Ok(r) => Probe::Down(format!("/ping {}", r.status)),
            Err(e) => Probe::Down(scrub(&e.to_string(), "", 200)),
        }
    }

    fn call(
        &mut self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        _stop: &dyn Fn() -> bool,
    ) -> Result<(u16, Value), ProviderError> {
        let mut req = Request::new(
            method,
            format!("{}{path}", self.base),
            Duration::from_secs(120),
        );
        if let Some(k) = &self.key {
            req = req.bearer(k);
        }
        if let Some(b) = body {
            req = req.json(b.to_string());
        }
        let r = self.http.send(&req).map_err(|e| ProviderError::Transport {
            what: format!("{} {path}", method.as_str()),
            why: self
                .key
                .as_deref()
                .map_or_else(|| e.to_string(), |k| scrub(&e.to_string(), k, 300)),
        })?;
        Ok((r.status, body_json(&r.body)))
    }
}
