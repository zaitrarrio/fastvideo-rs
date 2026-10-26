//! fv-deploy's Runpod providers against a fake Runpod ([`FakeRunpod`]: REST
//! v1, GraphQL, the log endpoint, the queue API and the pods' proxy with a
//! fake fv-serve behind it), and the whole `up` lifecycle on [`Pod`] and
//! [`Serverless`] over it.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use serde_json::{json, Value};

use super::runpod::{
    parse_runpod_time, price_query, quoted_price, Pod, RunpodApi, RunpodUrls, Serverless,
};
use super::*;
use crate::boot::sys::{Clock, Http, LogSink, Method, Request, Response, SysError};
use crate::guard::{self, Backstop, GuardError};
use crate::launch::{script_endpoint, script_pod, ScriptEndpoint, ScriptPod};
use crate::lifecycle::{DeployError, Deployer, SmokeOptions, UpOptions};

pub(crate) const GPU: &str = "NVIDIA RTX PRO 6000 Blackwell Server Edition";
pub(crate) const IMAGE: &str =
    "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:1111111111111111111111111111111111111111111111111111111111111111";
pub(crate) const KEY: &str = "rpa_TESTKEY0123456789";

pub(crate) fn urls() -> RunpodUrls {
    RunpodUrls {
        rest: "https://rest.test/v1".into(),
        queue: "https://queue.test/v2".into(),
        graphql: "https://gql.test/graphql".into(),
        hapi: "https://hapi.test/v1".into(),
        proxy: "https://{id}-8000.proxy.test".into(),
    }
}

/// A clock that only moves when slept on.
#[derive(Default)]
pub(crate) struct FakeClock {
    pub now: Cell<f64>,
}

impl FakeClock {
    pub fn at(t: f64) -> Self {
        Self { now: Cell::new(t) }
    }
}

impl Clock for FakeClock {
    fn now(&self) -> f64 {
        self.now.get()
    }
    fn sleep(&self, d: Duration) {
        self.now.set(self.now.get() + d.as_secs_f64());
    }
}

#[derive(Default)]
pub(crate) struct Lines(pub RefCell<Vec<String>>);

impl LogSink for Lines {
    fn line(&self, s: &str) {
        self.0.borrow_mut().push(s.to_owned());
    }
}

/// A backstop that records what it was asked to guard.
#[derive(Default)]
pub(crate) struct RecBackstop(pub RefCell<Vec<(ProviderKind, InstanceId, f64)>>);

impl Backstop for RecBackstop {
    fn arm(&self, kind: ProviderKind, id: &InstanceId, deadline: f64) -> Result<u32, GuardError> {
        self.0.borrow_mut().push((kind, id.clone(), deadline));
        Ok(4242)
    }
}

struct FakePod {
    name: String,
    desired: String,
    /// `/ping` answers in order; the last one repeats.
    pings: Vec<u16>,
    pinged: usize,
}

/// A fake Runpod: state behind `RefCell`, every request recorded.
pub(crate) struct FakeRunpod {
    pub balance: Cell<f64>,
    /// GPU type → secure price (absent: no quote).
    pub prices: RefCell<BTreeMap<String, f64>>,
    /// GPU types with no stock.
    pub no_stock: RefCell<Vec<String>>,
    /// The `costPerHr` the create answers (default: the quote as a number).
    pub billed: RefCell<Option<Value>>,
    /// `currentSpendPerHr` of the account.
    pub spend: Cell<f64>,
    /// The create is carried out but its answer is lost (a transport error).
    pub lose_create_answer: Cell<bool>,
    /// Every DELETE of a pod answers 500.
    pub delete_fails: Cell<bool>,
    /// The `/ping` sequence of each new pod.
    pub pings: RefCell<Vec<u16>>,
    pods: RefCell<BTreeMap<String, FakePod>>,
    /// endpoint id → (template id, name).
    endpoints: RefCell<BTreeMap<String, (String, String)>>,
    /// (template id, name).
    templates: RefCell<Vec<(String, String)>>,
    jobs: RefCell<BTreeMap<String, Value>>,
    job_polls: Cell<u32>,
    seq: Cell<u32>,
    pub log: RefCell<Vec<Request>>,
}

impl FakeRunpod {
    pub fn new() -> Self {
        let mut prices = BTreeMap::new();
        prices.insert(GPU.to_owned(), 1.84);
        Self {
            balance: Cell::new(20.0),
            prices: RefCell::new(prices),
            no_stock: RefCell::new(Vec::new()),
            billed: RefCell::new(None),
            spend: Cell::new(0.5),
            lose_create_answer: Cell::new(false),
            delete_fails: Cell::new(false),
            pings: RefCell::new(vec![502, 204, 200]),
            pods: RefCell::new(BTreeMap::new()),
            endpoints: RefCell::new(BTreeMap::new()),
            templates: RefCell::new(Vec::new()),
            jobs: RefCell::new(BTreeMap::new()),
            job_polls: Cell::new(0),
            seq: Cell::new(0),
            log: RefCell::new(Vec::new()),
        }
    }

    fn next(&self, p: &str) -> String {
        self.seq.set(self.seq.get() + 1);
        format!("{p}{}", self.seq.get())
    }

    pub fn calls(&self, method: Method, needle: &str) -> Vec<Request> {
        self.log
            .borrow()
            .iter()
            .filter(|r| r.method == method && r.url.contains(needle))
            .cloned()
            .collect()
    }

    pub fn live_pods(&self) -> Vec<String> {
        self.pods.borrow().keys().cloned().collect()
    }

    pub fn live_endpoints(&self) -> usize {
        self.endpoints.borrow().len()
    }

    pub fn live_templates(&self) -> usize {
        self.templates.borrow().len()
    }

    fn body(req: &Request) -> Value {
        req.body
            .as_ref()
            .and_then(|b| serde_json::from_slice(b).ok())
            .unwrap_or(Value::Null)
    }

    /// The fake fv-serve behind a pod's proxy or a queue job.
    fn serve(&self, method: &str, path: &str, body: &Value) -> (u16, Value) {
        match (method, path) {
            ("GET", "/healthz") => (
                200,
                json!({"state": "ready", "loaded": ["fasth3"], "loading": []}),
            ),
            ("GET", "/fv/v1/capabilities") => (200, json!({"models": [{"id": "fasth3"}]})),
            ("POST", "/fv/v1/jobs") => {
                let id = self.next("job-");
                self.jobs.borrow_mut().insert(
                    id.clone(),
                    json!({"id": id, "model": body["model"], "status": "queued"}),
                );
                (200, json!({"id": id, "status": "queued"}))
            }
            ("GET", p) if p.starts_with("/fv/v1/jobs/") => {
                let id = &p["/fv/v1/jobs/".len()..];
                self.job_polls.set(self.job_polls.get() + 1);
                let st = if self.job_polls.get() < 3 {
                    "running"
                } else {
                    "succeeded"
                };
                match self.jobs.borrow().get(id) {
                    Some(j) => (200, json!({"id": id, "model": j["model"], "status": st})),
                    None => (404, json!({"error": "no such job"})),
                }
            }
            _ => (404, json!({"error": "not found"})),
        }
    }

    fn ok(v: Value) -> Response {
        Response {
            status: 200,
            body: v.to_string().into_bytes(),
        }
    }

    fn status(code: u16, v: Value) -> Response {
        Response {
            status: code,
            body: v.to_string().into_bytes(),
        }
    }

    fn graphql(&self, q: &str) -> Response {
        if q.contains("clientBalance") {
            return Self::ok(
                json!({"data": {"myself": {"clientBalance": self.balance.get(), "currentSpendPerHr": self.spend.get()}}}),
            );
        }
        if q.contains("gpuTypes") {
            let prices = self.prices.borrow();
            let hit = prices.iter().find(|(g, _)| q.contains(&format!("{:?}", g)));
            return match hit {
                Some((g, p)) => Self::ok(
                    json!({"data": {"gpuTypes": [{"id": g, "securePrice": p, "communityPrice": p * 0.7,
                    "lowestPrice": {"uninterruptablePrice": p - 0.1}}]}}),
                ),
                None => Self::ok(json!({"data": {"gpuTypes": []}})),
            };
        }
        Self::ok(json!({"errors": [{"message": "unknown query"}]}))
    }

    fn rest(&self, req: &Request, path: &str) -> Response {
        let m = req.method.as_str();
        let body = Self::body(req);
        match (m, path) {
            ("POST", "/pods") => {
                let gpu = body["gpuTypeIds"][0].as_str().unwrap_or("").to_owned();
                if self.no_stock.borrow().contains(&gpu) {
                    return Self::status(
                        500,
                        json!({"error": "There are no instances currently available"}),
                    );
                }
                let id = self.next("pod");
                let price = self
                    .billed
                    .borrow()
                    .clone()
                    .or_else(|| self.prices.borrow().get(&gpu).map(|p| json!(p)))
                    .unwrap_or(json!(9.0));
                self.pods.borrow_mut().insert(
                    id.clone(),
                    FakePod {
                        name: body["name"].as_str().unwrap_or("").to_owned(),
                        desired: "RUNNING".into(),
                        pings: self.pings.borrow().clone(),
                        pinged: 0,
                    },
                );
                Self::ok(
                    json!({"id": id, "costPerHr": price, "machine": {"dataCenterId": "EU-RO-1"}}),
                )
            }
            ("GET", "/pods") => Self::ok(Value::Array(
                self.pods
                    .borrow()
                    .iter()
                    .map(|(id, p)| json!({"id": id, "name": p.name, "desiredStatus": p.desired}))
                    .collect(),
            )),
            ("GET", "/endpoints") => Self::ok(Value::Array(
                self.endpoints
                    .borrow()
                    .iter()
                    .map(|(id, (t, n))| json!({"id": id, "name": n, "templateId": t}))
                    .collect(),
            )),
            ("GET", "/templates") => Self::ok(Value::Array(
                self.templates
                    .borrow()
                    .iter()
                    .map(|(id, n)| json!({"id": id, "name": n}))
                    .collect(),
            )),
            ("DELETE", p) if p.starts_with("/pods/") && self.delete_fails.get() => {
                Self::status(500, json!({"error": "internal"}))
            }
            ("GET", p) if p.starts_with("/pods/") => match self.pods.borrow().get(&p[6..]) {
                Some(pod) => Self::ok(json!({"id": &p[6..], "desiredStatus": pod.desired})),
                None => Self::status(404, json!({"error": "pod not found"})),
            },
            ("DELETE", p) if p.starts_with("/pods/") => {
                match self.pods.borrow_mut().remove(&p[6..]) {
                    Some(_) => Self::ok(Value::Null),
                    None => Self::status(404, json!({"error": "pod not found"})),
                }
            }
            ("POST", "/templates") => {
                let id = self.next("tpl");
                self.templates
                    .borrow_mut()
                    .push((id.clone(), body["name"].as_str().unwrap_or("").to_owned()));
                Self::ok(json!({"id": id, "name": body["name"]}))
            }
            ("DELETE", p) if p.starts_with("/templates/") => {
                let id = &p["/templates/".len()..];
                let mut t = self.templates.borrow_mut();
                let n = t.len();
                t.retain(|(x, _)| x != id);
                if t.len() < n {
                    Self::ok(Value::Null)
                } else {
                    Self::status(404, json!({"error": "no template"}))
                }
            }
            ("POST", "/endpoints") => {
                let id = self.next("ep");
                self.endpoints.borrow_mut().insert(
                    id.clone(),
                    (
                        body["templateId"].as_str().unwrap_or("").to_owned(),
                        body["name"].as_str().unwrap_or("").to_owned(),
                    ),
                );
                Self::ok(json!({"id": id, "templateId": body["templateId"]}))
            }
            ("PATCH", p) if p.starts_with("/endpoints/") => Self::ok(json!({"id": &p[11..]})),
            ("GET", p) if p.starts_with("/endpoints/") => {
                match self.endpoints.borrow().get(&p[11..]) {
                    Some((t, _)) => Self::ok(json!({"id": &p[11..], "templateId": t})),
                    None => Self::status(404, json!({"error": "no endpoint"})),
                }
            }
            ("DELETE", p) if p.starts_with("/endpoints/") => {
                match self.endpoints.borrow_mut().remove(&p[11..]) {
                    Some(_) => Self::ok(Value::Null),
                    None => Self::status(404, json!({"error": "no endpoint"})),
                }
            }
            _ => Self::status(404, json!({"error": format!("fake: no route {m} {path}")})),
        }
    }

    fn queue(&self, req: &Request, path: &str) -> Response {
        // /<ep>/run, /<ep>/status/<job>, /<ep>/health
        let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        match (req.method, parts.as_slice()) {
            (Method::Post, [_, "run"]) => {
                let input = Self::body(req)["input"].clone();
                let id = self.next("q-");
                let out = match input["kind"].as_str() {
                    Some("info") => json!({"gpu": "fake"}),
                    _ => {
                        let (s, b) = self.serve(
                            input["method"].as_str().unwrap_or("GET"),
                            input["path"].as_str().unwrap_or("/"),
                            &input["body"],
                        );
                        json!({"status": s, "body": b})
                    }
                };
                self.jobs.borrow_mut().insert(id.clone(), out);
                Self::ok(json!({"id": id, "status": "IN_QUEUE"}))
            }
            (Method::Get, [_, "status", job]) => match self.jobs.borrow().get(*job) {
                Some(out) => Self::ok(json!({"id": job, "status": "COMPLETED", "output": out})),
                None => Self::status(404, json!({"error": "no job"})),
            },
            (Method::Get, [_, "health"]) => Self::ok(json!({"workers": {"idle": 1}})),
            _ => Self::status(404, json!({"error": "fake queue: no route"})),
        }
    }

    fn proxy(&self, req: &Request, pod: &str, path: &str) -> Response {
        if path == "/ping" {
            let mut pods = self.pods.borrow_mut();
            let Some(p) = pods.get_mut(pod) else {
                return Self::status(404, json!({}));
            };
            let i = p.pinged.min(p.pings.len().saturating_sub(1));
            p.pinged += 1;
            let code = p.pings.get(i).copied().unwrap_or(502);
            return Response {
                status: code,
                body: Vec::new(),
            };
        }
        let auth = req
            .headers
            .iter()
            .any(|(k, v)| k == "authorization" && v == "Bearer fvk-run");
        if !auth {
            return Self::status(401, json!({"error": "no key"}));
        }
        let (s, v) = self.serve(req.method.as_str(), path, &Self::body(req));
        Self::status(s, v)
    }
}

impl Http for FakeRunpod {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        self.log.borrow_mut().push(req.clone());
        let u = &req.url;
        let authed = req
            .headers
            .iter()
            .any(|(k, v)| k == "authorization" && v == &format!("Bearer {KEY}"));
        if let Some(p) = u.strip_prefix("https://rest.test/v1") {
            assert!(authed, "REST call without the key: {u}");
            let r = self.rest(req, p);
            if self.lose_create_answer.get()
                && req.method == Method::Post
                && (p == "/pods" || p == "/endpoints")
            {
                return Err(SysError::Http {
                    url: u.clone(),
                    why: "connection reset after the request was sent".into(),
                });
            }
            return Ok(r);
        }
        if u.starts_with("https://gql.test/graphql") {
            assert!(authed);
            let q = Self::body(req)["query"].as_str().unwrap_or("").to_owned();
            return Ok(self.graphql(&q));
        }
        if let Some(p) = u.strip_prefix("https://queue.test/v2") {
            assert!(authed);
            return Ok(self.queue(req, p));
        }
        if let Some(p) = u.strip_prefix("https://hapi.test/v1/pod/") {
            let id = p.trim_end_matches("/logs");
            return Ok(if self.pods.borrow().contains_key(id) {
                Self::ok(
                    json!({"container": ["[fv-boot] start", "FV-SERVE READY"], "system": ["pulling image"]}),
                )
            } else {
                Self::status(404, json!({}))
            });
        }
        if let Some(rest) = u.strip_prefix("https://") {
            if let Some((host, path)) = rest.split_once('/') {
                if let Some(pod) = host.strip_suffix("-8000.proxy.test") {
                    return Ok(self.proxy(req, pod, &format!("/{path}")));
                }
            }
        }
        Err(SysError::Http {
            url: u.clone(),
            why: "fake: unknown host".into(),
        })
    }

    fn download(&self, url: &str, _dest: &Path, _t: Duration) -> Result<u64, SysError> {
        Err(SysError::Http {
            url: url.into(),
            why: "fake: no downloads".into(),
        })
    }
}

pub(crate) fn guards() -> Guards {
    guard::default_guards()
}

pub(crate) fn eu_pod_launch() -> Launch {
    let mut l = script_pod(&ScriptPod {
        name: "fv-deploy-test",
        image: IMAGE,
        gpu: GPU,
        key_hash: "hash",
        config: "/etc/fv/runpod.toml",
        inline: None,
        volume: Some(("bz0ofkfx45", "EU-RO-1")),
        guards: guards(),
    });
    l.placement.region = Some("eu".into());
    l
}

fn api<'a>(rp: &'a FakeRunpod, clock: &'a FakeClock) -> RunpodApi<'a> {
    RunpodApi::new(rp, clock, KEY, urls())
}

fn never() -> bool {
    false
}

fn leaks(s: &str) -> bool {
    s.contains(KEY)
}

// ------------------------------------------------------------------ the API

#[test]
fn price_query_is_the_scripts() {
    assert_eq!(
        price_query(GPU, CloudType::Secure, Some("EU-RO-1")),
        "query { gpuTypes(input: {id: \"NVIDIA RTX PRO 6000 Blackwell Server Edition\"}) { id securePrice communityPrice lowestPrice(input: {gpuCount: 1, secureCloud: true, dataCenterId: \"EU-RO-1\"}) { uninterruptablePrice } } }"
    );
    assert_eq!(
        price_query("X", CloudType::Community, None),
        "query { gpuTypes(input: {id: \"X\"}) { id securePrice communityPrice  } }"
    );
    let d = json!({"gpuTypes": [{"securePrice": 1.79, "communityPrice": 0.0, "lowestPrice": {"uninterruptablePrice": 1.99}}]});
    assert_eq!(quoted_price(&d, CloudType::Secure), Some(1.99));
    assert_eq!(quoted_price(&d, CloudType::Community), Some(1.99));
    assert_eq!(
        quoted_price(
            &json!({"gpuTypes": [{"securePrice": 0}]}),
            CloudType::Secure
        ),
        None
    );
    assert_eq!(
        quoted_price(&json!({"gpuTypes": []}), CloudType::Secure),
        None
    );
}

#[test]
fn runpod_times() {
    assert_eq!(
        parse_runpod_time("2026-09-28 00:00:00.000 +0000 UTC"),
        Some(1_790_553_600.0)
    );
    assert_eq!(parse_runpod_time("garbage"), None);
}

#[test]
fn key_is_a_header_only_and_never_in_errors_or_debug() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(1000.0);
    let pod = Pod::new(api(&rp, &clock));
    assert!(!format!("{pod:?}").contains(KEY));
    // An error body that echoes the key is scrubbed.
    let e = pod
        .api
        .json_or_error(
            super::runpod::Raw {
                status: 500,
                body: format!("bad key {KEY}").into_bytes(),
            },
            "x",
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("<redacted>") && !e.contains(KEY), "{e}");
    let a = pod.account().ok();
    assert_eq!(a.map(|a| a.balance_usd), Some(20.0));
    for r in rp.log.borrow().iter() {
        assert!(!r.url.contains(KEY));
        let body = r
            .body
            .as_ref()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        assert!(!body.contains(KEY));
    }
}

#[test]
fn pod_create_status_logs_destroy() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(1000.0);
    let pod = Pod::new(api(&rp, &clock));
    let inst = pod.create(&eu_pod_launch()).expect("create");
    assert_eq!(inst.id.as_str(), "pod1");
    assert_eq!(inst.url.as_deref(), Some("https://pod1-8000.proxy.test"));
    assert_eq!(inst.usd_per_hr, Some(1.84));
    assert_eq!(inst.data_center.as_deref(), Some("EU-RO-1"));
    assert_eq!(pod.status(&inst.id).ok(), Some(InstanceState::Running));
    let page = pod.logs(&inst.id, None).expect("logs");
    assert_eq!(
        page.lines,
        [
            "[system] pulling image",
            "[fv-boot] start",
            "FV-SERVE READY"
        ]
    );
    let again = pod.logs(&inst.id, page.next.as_ref()).expect("logs");
    assert!(again.lines.is_empty());
    pod.destroy(&inst.id).expect("destroy");
    assert_eq!(pod.status(&inst.id).ok(), Some(InstanceState::Terminated));
    // Gone is not an error.
    pod.destroy(&inst.id).expect("destroy twice");
    // What went out is the payload builder's body.
    let post = rp.calls(Method::Post, "/pods");
    let sent: Value =
        serde_json::from_slice(post[0].body.as_deref().unwrap_or_default()).unwrap_or_default();
    assert_eq!(sent, runpod::pod_create_body(&eu_pod_launch()));
}

#[test]
fn no_stock_is_no_capacity() {
    let rp = FakeRunpod::new();
    rp.no_stock.borrow_mut().push(GPU.into());
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    assert!(matches!(
        pod.create(&eu_pod_launch()),
        Err(ProviderError::NoCapacity(_))
    ));
}

#[test]
fn serverless_create_and_destroy_like_the_script() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(0.0);
    let sls = Serverless::new(api(&rp, &clock));
    let l = script_endpoint(&ScriptEndpoint {
        name: "fv-deploy-q",
        image: IMAGE,
        gpus: &[GPU],
        config: "/etc/fv/runpod.toml",
        inline: None,
        volume: "bz0ofkfx45",
        data_center: "EU-RO-1",
        guards: guards(),
    });
    let inst = sls.create(&l).expect("create");
    assert_eq!(inst.id.as_str(), "ep2/tpl1");
    assert_eq!((rp.live_endpoints(), rp.live_templates()), (1, 1));
    let ep: Value = serde_json::from_slice(
        rp.calls(Method::Post, "/endpoints")[0]
            .body
            .as_deref()
            .unwrap_or_default(),
    )
    .unwrap_or_default();
    assert_eq!(ep["templateId"], "tpl1");
    assert_eq!(sls.status(&inst.id).ok(), Some(InstanceState::Running));
    // Ready once the cold info job completes; one native call through the queue.
    let mut r = sls.reach(&inst, &ReachOptions::default()).expect("reach");
    assert_eq!(r.ping(), Probe::Loading);
    assert_eq!(r.ping(), Probe::Ready);
    let (s, caps) = r
        .call(Method::Get, "/fv/v1/capabilities", None, &never)
        .expect("call");
    assert_eq!((s, caps["models"][0]["id"].as_str()), (200, Some("fasth3")));
    drop(r);
    sls.destroy(&inst.id).expect("destroy");
    assert_eq!((rp.live_endpoints(), rp.live_templates()), (0, 0));
    // Workers to zero first, as `runpod-endpoint.sh down`.
    let order: Vec<String> = rp
        .log
        .borrow()
        .iter()
        .filter(|r| r.url.contains("/endpoints/") || r.url.contains("/templates/"))
        .map(|r| {
            format!(
                "{} {}",
                r.method.as_str(),
                r.url.rsplit('/').nth(1).unwrap_or("")
            )
        })
        .collect();
    assert_eq!(
        order,
        [
            "GET endpoints",
            "PATCH endpoints",
            "DELETE endpoints",
            "DELETE templates"
        ]
    );
}

// ------------------------------------------------------------------ up on Runpod

fn deployer<'a>(
    p: &'a dyn Provider,
    clock: &'a FakeClock,
    log: &'a Lines,
    bs: &'a RecBackstop,
) -> Deployer<'a> {
    Deployer {
        provider: p,
        clock,
        log,
        backstop: bs,
        ledger: None,
        recorder: None,
        interrupted: &never,
        leaks: &leaks,
        poll: Duration::from_secs(5),
    }
}

fn up_opts() -> UpOptions {
    UpOptions {
        run_key: Some("fvk-run".into()),
        smoke: Some(SmokeOptions::default()),
        keep: false,
    }
}

#[test]
fn pod_up_ready_smoke_down() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(1_000.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let r = deployer(&pod, &clock, &log, &bs)
        .up(&eu_pod_launch(), &up_opts())
        .expect("up");
    assert!(r.ok, "{r:?}");
    assert_eq!(r.cycles.len(), 1);
    assert_eq!(r.cycles[0].ready_after_s, Some(10.0));
    let s = r.smoke.expect("smoke");
    assert_eq!(s.job.status, "succeeded");
    assert_eq!(s.job.model.as_deref(), Some("fasth3"));
    assert_eq!(s.healthz["state"], "ready");
    assert!(r.deleted);
    assert!(rp.live_pods().is_empty());
    // The backstop guards the pod until the cap.
    assert_eq!(
        bs.0.borrow()[0],
        (
            ProviderKind::RunpodPod,
            InstanceId::new("pod1"),
            1_000.0 + 1800.0
        )
    );
    // The quote came before the create.
    let order: Vec<&str> = rp
        .log
        .borrow()
        .iter()
        .filter_map(|r| {
            if r.url.contains("gql") {
                let q = String::from_utf8_lossy(r.body.as_deref().unwrap_or_default()).into_owned();
                Some(if q.contains("gpuTypes") {
                    "quote"
                } else {
                    "balance"
                })
            } else if r.method == Method::Post && r.url.ends_with("/pods") {
                Some("create")
            } else {
                None
            }
        })
        .collect();
    assert_eq!(order, ["balance", "quote", "create"]);
}

#[test]
fn serverless_up_ready_smoke_down() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(0.0);
    let sls = Serverless::new(api(&rp, &clock));
    let mut l = script_endpoint(&ScriptEndpoint {
        name: "fv-deploy-q",
        image: IMAGE,
        gpus: &[GPU],
        config: "/etc/fv/runpod.toml",
        inline: None,
        volume: "bz0ofkfx45",
        data_center: "EU-RO-1",
        guards: guards(),
    });
    l.placement.region = Some("eu".into());
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut o = up_opts();
    o.run_key = None;
    let r = deployer(&sls, &clock, &log, &bs).up(&l, &o).expect("up");
    assert!(r.ok, "{r:?}");
    assert_eq!(r.smoke.map(|s| s.job.status), Some("succeeded".into()));
    assert_eq!((rp.live_endpoints(), rp.live_templates()), (0, 0));
}

#[test]
fn pod_never_ready_is_replaced_three_times_then_given_up() {
    let rp = FakeRunpod::new();
    *rp.pings.borrow_mut() = vec![502];
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let e = deployer(&pod, &clock, &log, &bs).up(&eu_pod_launch(), &up_opts());
    match e {
        Err(DeployError::NotReady {
            cycles, timeout_s, ..
        }) => assert_eq!((cycles, timeout_s), (3, 600)),
        other => panic!("{other:?}"),
    }
    assert_eq!(rp.calls(Method::Post, "/pods").len(), 3);
    assert_eq!(rp.calls(Method::Delete, "/pods/").len(), 3);
    assert!(rp.live_pods().is_empty());
    assert_eq!(bs.0.borrow().len(), 3);
    // Each pod got its 600 s (and no more than one poll beyond).
    assert!(
        clock.now() >= 1800.0 && clock.now() < 1800.0 + 3.0 * 5.0 + 1.0,
        "{}",
        clock.now()
    );
}

#[test]
fn balance_below_floor_refuses_before_any_create() {
    let rp = FakeRunpod::new();
    rp.balance.set(7.99);
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let e = deployer(&pod, &clock, &log, &bs).up(&eu_pod_launch(), &up_opts());
    assert!(
        matches!(
            e,
            Err(DeployError::Guard(GuardError::BalanceBelowFloor { .. }))
        ),
        "{e:?}"
    );
    assert!(rp.calls(Method::Post, "/pods").is_empty());
    assert!(bs.0.borrow().is_empty());
}

#[test]
fn over_cap_or_unquoted_gpu_is_never_created() {
    let rp = FakeRunpod::new();
    rp.prices.borrow_mut().insert(GPU.into(), 4.0);
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut l = eu_pod_launch();
    l.gpu.types = vec![GPU.into(), "NVIDIA H200".into()];
    let e = deployer(&pod, &clock, &log, &bs).up(&l, &up_opts());
    assert!(matches!(e, Err(DeployError::NoCapacity(_))), "{e:?}");
    assert!(rp.calls(Method::Post, "/pods").is_empty());
    // Billed over the cap (the quote passed): deleted at once.
    rp.prices.borrow_mut().insert(GPU.into(), 1.5);
    *rp.billed.borrow_mut() = Some(json!(3.0));
    let e = deployer(&pod, &clock, &log, &bs).up(&l, &up_opts());
    assert!(matches!(e, Err(DeployError::NoCapacity(_))), "{e:?}");
    assert_eq!(rp.calls(Method::Post, "/pods").len(), 1);
    assert!(rp.live_pods().is_empty());
}

#[test]
fn falls_back_to_the_next_gpu_type_without_stock() {
    let rp = FakeRunpod::new();
    rp.prices.borrow_mut().insert("NVIDIA H200".into(), 2.2);
    rp.no_stock.borrow_mut().push(GPU.into());
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut l = eu_pod_launch();
    l.gpu.types = vec![GPU.into(), "NVIDIA H200".into()];
    let r = deployer(&pod, &clock, &log, &bs)
        .up(&l, &up_opts())
        .expect("up");
    assert_eq!(
        r.instance.and_then(|i| i.gpu).as_deref(),
        Some("NVIDIA H200")
    );
}

#[test]
fn keep_leaves_the_pod_for_the_backstop() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut o = up_opts();
    o.keep = true;
    o.smoke = None;
    let r = deployer(&pod, &clock, &log, &bs)
        .up(&eu_pod_launch(), &o)
        .expect("up");
    assert!(!r.deleted && r.ok);
    assert_eq!(rp.live_pods(), ["pod1"]);
    // The backstop's own run deletes it at the deadline.
    let out = guard::run_backstop(&pod, &InstanceId::new("pod1"), 1800.0, &clock, &log)
        .expect("backstop");
    assert_eq!(out, guard::BackstopOutcome::Deleted);
    assert!(rp.live_pods().is_empty());
    assert!(clock.now() >= 1800.0);
}

#[test]
fn backstop_exits_early_when_the_pod_is_gone() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let inst = pod.create(&eu_pod_launch()).expect("create");
    pod.destroy(&inst.id).expect("destroy");
    let log = Lines::default();
    let out = guard::run_backstop(&pod, &inst.id, 1800.0, &clock, &log).expect("backstop");
    assert_eq!(out, guard::BackstopOutcome::AlreadyGone);
    // Two "gone" readings a minute apart, never one.
    assert!(
        clock.now() >= 120.0 && clock.now() <= 121.0,
        "{}",
        clock.now()
    );
    assert!(rp.calls(Method::Delete, "/pods/").len() == 1);
}

#[test]
fn smoke_failure_still_deletes() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut o = up_opts();
    o.run_key = Some("wrong-key".into());
    let r = deployer(&pod, &clock, &log, &bs)
        .up(&eu_pod_launch(), &o)
        .expect("up");
    assert!(!r.ok);
    assert!(r
        .smoke
        .and_then(|s| s.why)
        .unwrap_or_default()
        .contains("401"));
    assert!(r.deleted && rp.live_pods().is_empty());
}

#[test]
fn interrupted_wait_deletes_the_pod() {
    let rp = FakeRunpod::new();
    *rp.pings.borrow_mut() = vec![502];
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let stop = || clock.now() > 30.0;
    let mut d = deployer(&pod, &clock, &log, &bs);
    d.interrupted = &stop;
    assert!(matches!(
        d.up(&eu_pod_launch(), &up_opts()),
        Err(DeployError::Interrupted)
    ));
    assert!(rp.live_pods().is_empty());
}

#[test]
fn launch_with_the_key_in_its_env_is_refused() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut l = eu_pod_launch();
    l.env.insert("FV_BACKSTOP_API_KEY".into(), KEY.into());
    let e = deployer(&pod, &clock, &log, &bs).up(&l, &up_opts());
    assert!(
        matches!(
            e,
            Err(DeployError::Guard(GuardError::SecretInLaunch("env")))
        ),
        "{e:?}"
    );
    assert!(rp.log.borrow().is_empty());
}

// ------------------------------------------------------------------ review fixes

/// A backstop that refuses to arm.
struct NoArm;

impl Backstop for NoArm {
    fn arm(&self, _k: ProviderKind, _id: &InstanceId, _d: f64) -> Result<u32, GuardError> {
        Err(GuardError::Backstop("the child exited at start".into()))
    }
}

fn ledger_at(tag: &str) -> crate::lifecycle::Ledger {
    let dir = std::env::temp_dir().join(format!("fv-deploy-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    crate::lifecycle::Ledger {
        path: dir.join("ledger.tsv"),
    }
}

#[test]
fn a_lost_create_answer_stops_and_sweeps_by_name() {
    let rp = FakeRunpod::new();
    rp.lose_create_answer.set(true);
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut l = eu_pod_launch();
    l.gpu.types = vec![GPU.into(), "NVIDIA H200".into()];
    rp.prices.borrow_mut().insert("NVIDIA H200".into(), 2.0);
    let e = deployer(&pod, &clock, &log, &bs).up(&l, &up_opts());
    match e {
        Err(DeployError::CreateUncertain { swept, .. }) => assert_eq!(swept, "pod1"),
        other => panic!("{other:?}"),
    }
    // No fallback create on the next GPU type; the pod the lost answer
    // created is gone.
    assert_eq!(rp.calls(Method::Post, "/pods").len(), 1);
    assert!(rp.live_pods().is_empty());
}

#[test]
fn a_lost_endpoint_answer_sweeps_the_endpoint_and_its_template() {
    let rp = FakeRunpod::new();
    rp.lose_create_answer.set(true);
    let clock = FakeClock::at(0.0);
    let sls = Serverless::new(api(&rp, &clock));
    let mut l = script_endpoint(&ScriptEndpoint {
        name: "fv-deploy-q-lost",
        image: IMAGE,
        gpus: &[GPU],
        config: "/etc/fv/runpod.toml",
        inline: None,
        volume: "bz0ofkfx45",
        data_center: "EU-RO-1",
        guards: guards(),
    });
    l.placement.region = Some("eu".into());
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut o = up_opts();
    o.run_key = None;
    let e = deployer(&sls, &clock, &log, &bs).up(&l, &o);
    assert!(
        matches!(e, Err(DeployError::CreateUncertain { .. })),
        "{e:?}"
    );
    assert_eq!((rp.live_endpoints(), rp.live_templates()), (0, 0));
}

#[test]
fn the_exit_sweep_removes_a_stray_instance_of_the_run() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    // A pod of this run's name that `up` never heard of (a lost answer of
    // an earlier attempt in the same run).
    let stray = pod.create(&eu_pod_launch()).expect("create");
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let r = deployer(&pod, &clock, &log, &bs)
        .up(&eu_pod_launch(), &up_opts())
        .expect("up");
    assert!(r.ok);
    assert!(rp.live_pods().is_empty(), "{:?}", rp.live_pods());
    assert!(log
        .0
        .borrow()
        .iter()
        .any(|l| l.contains(&format!("swept {}", stray.id))));
    // With --keep, the kept pod survives the sweep.
    let mut o = up_opts();
    o.keep = true;
    let r = deployer(&pod, &clock, &log, &bs)
        .up(&eu_pod_launch(), &o)
        .expect("up");
    let kept = r.instance.map(|i| i.id.to_string()).unwrap_or_default();
    assert_eq!(rp.live_pods(), [kept]);
}

#[test]
fn billed_price_as_a_string_passes_and_unreadable_fails() {
    let rp = FakeRunpod::new();
    *rp.billed.borrow_mut() = Some(json!("0.74"));
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let inst = pod.create(&eu_pod_launch()).expect("create");
    assert_eq!(inst.usd_per_hr, Some(0.74));
    pod.destroy(&inst.id).expect("destroy");
    for bad in [json!("n/a"), json!(null), json!(-1)] {
        *rp.billed.borrow_mut() = Some(bad.clone());
        let (log, bs) = (Lines::default(), RecBackstop::default());
        let e = deployer(&pod, &clock, &log, &bs).up(&eu_pod_launch(), &up_opts());
        assert!(
            matches!(e, Err(DeployError::NoCapacity(ref m)) if m.contains("no (or no readable) billed price")),
            "{bad}: {e:?}"
        );
        // Backstopped before the check, deleted after it.
        assert_eq!(bs.0.borrow().len(), 1, "{bad}");
        assert!(rp.live_pods().is_empty(), "{bad}");
    }
}

#[test]
fn over_cap_pod_whose_delete_fails_stops_with_its_backstop_armed() {
    let rp = FakeRunpod::new();
    *rp.billed.borrow_mut() = Some(json!(3.0));
    rp.delete_fails.set(true);
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let ledger = ledger_at("overcap");
    let mut d = deployer(&pod, &clock, &log, &bs);
    d.ledger = Some(&ledger);
    let e = d.up(&eu_pod_launch(), &up_opts());
    assert!(
        matches!(e, Err(DeployError::DeleteFailed { ref id, .. }) if id.as_str() == "pod1"),
        "{e:?}"
    );
    assert_eq!(bs.0.borrow().len(), 1);
    // Three delete attempts, 5 s apart (plus the exit sweep's own attempt).
    assert!(rp.calls(Method::Delete, "/pods/pod1").len() >= 3);
    let t = std::fs::read_to_string(&ledger.path).unwrap_or_default();
    assert!(
        t.contains("delete-failed pod1") && !t.contains("deleted pod1"),
        "{t}"
    );
}

#[test]
fn not_ready_pod_whose_delete_fails_is_not_replaced() {
    let rp = FakeRunpod::new();
    *rp.pings.borrow_mut() = vec![502];
    rp.delete_fails.set(true);
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let e = deployer(&pod, &clock, &log, &bs).up(&eu_pod_launch(), &up_opts());
    assert!(matches!(e, Err(DeployError::DeleteFailed { .. })), "{e:?}");
    assert_eq!(rp.calls(Method::Post, "/pods").len(), 1);
}

#[test]
fn no_backstop_deletes_with_retries_and_says_so() {
    let rp = FakeRunpod::new();
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let log = Lines::default();
    let d = Deployer {
        provider: &pod,
        clock: &clock,
        log: &log,
        backstop: &NoArm,
        ledger: None,
        recorder: None,
        interrupted: &never,
        leaks: &leaks,
        poll: Duration::from_secs(5),
    };
    match d.up(&eu_pod_launch(), &up_opts()) {
        Err(DeployError::NoBackstop { id, deleted, .. }) => {
            assert_eq!((id.as_str(), deleted), ("pod1", true))
        }
        other => panic!("{other:?}"),
    }
    assert!(rp.live_pods().is_empty());
    assert!(!log
        .0
        .borrow()
        .iter()
        .any(|l| l.contains("the backstop retries")));
    // When the delete fails too, the error says so (and names the id).
    rp.delete_fails.set(true);
    let e = d.up(&eu_pod_launch(), &up_opts());
    match &e {
        Err(DeployError::NoBackstop {
            id, deleted: false, ..
        }) => {
            assert_eq!(id.as_str(), "pod2");
            assert!(e
                .as_ref()
                .err()
                .map(|x| x.to_string())
                .unwrap_or_default()
                .contains("delete it by hand"));
        }
        other => panic!("{other:?}"),
    }
    assert!(rp.calls(Method::Delete, "/pods/pod2").len() >= 3);
}

#[test]
fn the_floor_holds_over_the_whole_cap() {
    let rp = FakeRunpod::new();
    // 9 − (1.84 + 0.5) × 1800 / 3600 = 7.83 < 8: refused before any create.
    rp.balance.set(9.0);
    let clock = FakeClock::at(0.0);
    let pod = Pod::new(api(&rp, &clock));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let e = deployer(&pod, &clock, &log, &bs).up(&eu_pod_launch(), &up_opts());
    assert!(
        matches!(
            e,
            Err(DeployError::Guard(
                GuardError::BalanceWouldDropBelowFloor { .. }
            ))
        ),
        "{e:?}"
    );
    assert!(rp.calls(Method::Post, "/pods").is_empty());
    // With no other spend it fits: 9 − 1.84 × 0.5 = 8.08.
    rp.spend.set(0.0);
    assert!(deployer(&pod, &clock, &log, &bs)
        .up(&eu_pod_launch(), &up_opts())
        .is_ok());
}

#[test]
fn serverless_with_more_than_one_worker_is_refused() {
    let mut l = script_endpoint(&ScriptEndpoint {
        name: "fv-deploy-q2",
        image: IMAGE,
        gpus: &[GPU],
        config: "/etc/fv/runpod.toml",
        inline: None,
        volume: "bz0ofkfx45",
        data_center: "EU-RO-1",
        guards: guards(),
    });
    if let Some(s) = l.serverless.as_mut() {
        s.workers_max = 2;
    }
    assert!(matches!(
        guard::check_serverless(&l),
        Err(GuardError::Invalid(_))
    ));
}

#[test]
fn prices_parse_from_numbers_and_strings() {
    assert_eq!(parse_usd(Some(&json!(0.74))), Some(0.74));
    assert_eq!(parse_usd(Some(&json!("0.74"))), Some(0.74));
    assert_eq!(parse_usd(Some(&json!(" 1.5 "))), Some(1.5));
    assert_eq!(parse_usd(Some(&json!("n/a"))), None);
    assert_eq!(parse_usd(Some(&json!(-2))), None);
    assert_eq!(parse_usd(Some(&json!(null))), None);
    assert_eq!(parse_usd(None), None);
}

#[test]
fn create_refusals_are_told_from_lost_answers() {
    let api4 = ProviderError::Api {
        what: "x".into(),
        status: 422,
        body: String::new(),
    };
    let api5 = ProviderError::Api {
        what: "x".into(),
        status: 502,
        body: String::new(),
    };
    assert!(api4.is_definite_refusal());
    assert!(ProviderError::NoCapacity("x".into()).is_definite_refusal());
    assert!(!api5.is_definite_refusal());
    assert!(!ProviderError::Transport {
        what: "x".into(),
        why: "reset".into()
    }
    .is_definite_refusal());
    assert!(!ProviderError::Decode {
        what: "x".into(),
        why: "no id".into()
    }
    .is_definite_refusal());
}
