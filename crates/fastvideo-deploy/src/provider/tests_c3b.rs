//! fv-deploy's Vast and GCP providers (C3b) against a fake Vast
//! ([`FakeVast`]) and a fake Compute Engine ([`FakeGcp`]), each with a fake
//! fv-serve at the instance's address, and the whole shared `up` lifecycle
//! on both: the same guards, backstop, replace rule, smoke and sweeps as on
//! Runpod (`provider/tests.rs`). No network: every request is recorded.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use base64::Engine as _;
use serde_json::{json, Value};

use super::gcp::{GcpApi, GcpAssets, GcpOptions, GcpPrices, GcpVm, PriceRow, StaticToken};
use super::tests::{FakeClock, Lines, RecBackstop, IMAGE};
use super::vast::{Vast, VastApi};
use super::*;
use crate::boot::sys::{Clock, Http, Method, Request, Response, SysError};
use crate::guard::{self, GuardError};
use crate::launch::{script_gcp, script_vast, ScriptGcp, ScriptVast};
use crate::lifecycle::{DeployError, Deployer, SmokeOptions, UpOptions};

const VAST_KEY: &str = "vast_SENTINEL_KEY_7";
const GCP_TOKEN: &str = "ya29.SENTINEL_TOKEN_9";
const RUN_KEY: &str = "fvk-run";

fn resp(status: u16, v: Value) -> Response {
    Response {
        status,
        body: v.to_string().into_bytes(),
    }
}

fn body_of(req: &Request) -> Value {
    req.body
        .as_ref()
        .and_then(|b| serde_json::from_slice(b).ok())
        .unwrap_or(Value::Null)
}

fn bearer(req: &Request) -> Option<String> {
    req.headers
        .iter()
        .find(|(k, _)| k == "authorization")
        .and_then(|(_, v)| v.strip_prefix("Bearer ").map(str::to_owned))
}

/// A fake fv-serve: `/ping` answers `pings` in order (the last repeats),
/// then `/healthz`, the capabilities and one job that succeeds on its
/// second poll; every call but `/ping` needs the run key.
#[derive(Default)]
struct FakeServe {
    pings: RefCell<Vec<u16>>,
    pinged: Cell<usize>,
    polls: Cell<u32>,
}

impl FakeServe {
    fn new(pings: &[u16]) -> Self {
        Self {
            pings: RefCell::new(pings.to_vec()),
            ..Self::default()
        }
    }

    fn answer(&self, req: &Request, path: &str) -> Response {
        if path == "/ping" {
            let p = self.pings.borrow();
            let i = self.pinged.get().min(p.len().saturating_sub(1));
            self.pinged.set(self.pinged.get() + 1);
            return Response {
                status: p.get(i).copied().unwrap_or(502),
                body: Vec::new(),
            };
        }
        if bearer(req).as_deref() != Some(RUN_KEY) {
            return resp(401, json!({"error": "no key"}));
        }
        match (req.method, path) {
            (Method::Get, "/healthz") => resp(200, json!({"state": "ready", "loaded": ["fake"]})),
            (Method::Get, "/fv/v1/capabilities") => resp(200, json!({"models": [{"id": "fake"}]})),
            (Method::Post, "/fv/v1/jobs") => resp(200, json!({"id": "j1", "status": "queued"})),
            (Method::Get, "/fv/v1/jobs/j1") => {
                self.polls.set(self.polls.get() + 1);
                let st = if self.polls.get() < 2 {
                    "running"
                } else {
                    "succeeded"
                };
                resp(200, json!({"id": "j1", "status": st}))
            }
            _ => resp(404, json!({"error": "not found"})),
        }
    }
}

/// Numbers compared as numbers (the fixtures are written by jq).
fn num(v: &Value) -> Value {
    match v {
        Value::Number(n) => json!(n.as_f64().unwrap_or(f64::NAN)),
        Value::Array(a) => Value::Array(a.iter().map(num).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), num(v))).collect()),
        other => other.clone(),
    }
}

fn never() -> bool {
    false
}

fn deployer<'a>(
    p: &'a dyn Provider,
    clock: &'a FakeClock,
    log: &'a Lines,
    bs: &'a RecBackstop,
    leaks: &'a dyn Fn(&str) -> bool,
) -> Deployer<'a> {
    Deployer {
        provider: p,
        clock,
        log,
        backstop: bs,
        ledger: None,
        interrupted: &never,
        leaks,
        poll: Duration::from_secs(5),
    }
}

fn up_opts() -> UpOptions {
    UpOptions {
        run_key: Some(RUN_KEY.into()),
        smoke: Some(SmokeOptions::default()),
        keep: false,
    }
}

// ------------------------------------------------------------------ Vast

const VAST: &str = "https://vast.test/api/v0";
const VAST_IP: &str = "203.0.113.5";

/// A fake Vast REST API.
struct FakeVast {
    /// (offer id, GPU name, $/hr).
    offers: RefCell<Vec<(u64, String, f64)>>,
    /// id → (label, actual_status).
    instances: RefCell<BTreeMap<String, (String, String)>>,
    credit: Cell<f64>,
    /// The create is carried out but its answer is lost.
    lose_create: Cell<bool>,
    /// Every ask answers `{"success": false}`.
    refuse_asks: Cell<bool>,
    /// The instance's own `dph_total` (absent: none reported).
    inst_dph: Cell<Option<f64>>,
    /// This many listings leave out every instance (a create not listed yet).
    hide_listings: Cell<u32>,
    /// The listing's entries carry no `label`.
    unlabelled: Cell<bool>,
    /// Every DELETE answers `{"success": false}`.
    refuse_deletes: Cell<bool>,
    seq: Cell<u64>,
    serve: FakeServe,
    log: RefCell<Vec<Request>>,
}

impl FakeVast {
    fn new() -> Self {
        Self {
            offers: RefCell::new(vec![
                (11, "RTX 4090".into(), 0.42),
                (12, "RTX 3090".into(), 0.21),
                (13, "H100 SXM".into(), 2.10),
            ]),
            instances: RefCell::new(BTreeMap::new()),
            credit: Cell::new(25.0),
            lose_create: Cell::new(false),
            refuse_asks: Cell::new(false),
            inst_dph: Cell::new(None),
            hide_listings: Cell::new(0),
            unlabelled: Cell::new(false),
            refuse_deletes: Cell::new(false),
            seq: Cell::new(0),
            serve: FakeServe::new(&[502, 204, 200]),
            log: RefCell::new(Vec::new()),
        }
    }

    fn live(&self) -> Vec<String> {
        self.instances.borrow().keys().cloned().collect()
    }

    fn rest(&self, req: &Request, path: &str) -> Response {
        let b = body_of(req);
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        match (req.method, parts.as_slice()) {
            (Method::Post, ["bundles"]) => {
                let cap = b.pointer("/dph_total/lte").and_then(Value::as_f64);
                let names: Option<Vec<String>> = b
                    .pointer("/gpu_name/in")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_owned))
                            .collect()
                    });
                let mut o: Vec<(u64, String, f64)> = self
                    .offers
                    .borrow()
                    .iter()
                    .filter(|(_, g, p)| {
                        cap.is_none_or(|c| *p <= c) && names.as_ref().is_none_or(|n| n.contains(g))
                    })
                    .cloned()
                    .collect();
                o.sort_by(|a, b| a.2.total_cmp(&b.2));
                resp(
                    200,
                    json!({"offers": o.iter().map(|(id, g, p)| json!({"id": id, "gpu_name": g, "dph_total": p})).collect::<Vec<_>>()}),
                )
            }
            (Method::Put, ["asks", offer]) => {
                if self.refuse_asks.get()
                    || !self
                        .offers
                        .borrow()
                        .iter()
                        .any(|(id, _, _)| id.to_string() == *offer)
                {
                    return resp(200, json!({"success": false, "msg": "no_such_ask"}));
                }
                self.seq.set(self.seq.get() + 1);
                let id = (9000 + self.seq.get()).to_string();
                let label = b["label"].as_str().unwrap_or("").to_owned();
                self.instances
                    .borrow_mut()
                    .insert(id.clone(), (label, "loading".into()));
                resp(
                    200,
                    json!({"success": true, "new_contract": id.parse::<u64>().unwrap_or(0)}),
                )
            }
            (Method::Get, ["instances", id]) => {
                let mut inst = self.instances.borrow_mut();
                let Some(e) = inst.get_mut(*id) else {
                    return resp(404, json!({"error": "no such instance"}));
                };
                let mut out = json!({"instances": {
                    "id": id, "label": e.0, "actual_status": e.1,
                    "public_ipaddr": VAST_IP,
                    "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41000"}]}
                }});
                if let Some(d) = self.inst_dph.get() {
                    out["instances"]["dph_total"] = json!(d);
                }
                e.1 = "running".into();
                resp(200, out)
            }
            (Method::Get, ["instances"]) => {
                if self.hide_listings.get() > 0 {
                    self.hide_listings.set(self.hide_listings.get() - 1);
                    return resp(200, json!({"instances": []}));
                }
                let unl = self.unlabelled.get();
                resp(
                    200,
                    json!({"instances": self.instances.borrow().iter().map(|(id, (l, s))| {
                        let mut e = json!({"id": id.parse::<u64>().unwrap_or(0), "label": l, "actual_status": s});
                        if unl { if let Some(o) = e.as_object_mut() { o.remove("label"); } }
                        e
                    }).collect::<Vec<_>>()}),
                )
            }
            (Method::Delete, ["instances", _]) if self.refuse_deletes.get() => {
                resp(200, json!({"success": false, "msg": "no_such_instance"}))
            }
            (Method::Delete, ["instances", id]) => match self.instances.borrow_mut().remove(*id) {
                Some(_) => resp(200, json!({"success": true})),
                None => resp(404, json!({"error": "no such instance"})),
            },
            (Method::Get, ["users", "current"]) => {
                resp(200, json!({"id": 1, "credit": self.credit.get()}))
            }
            _ => resp(404, json!({"error": format!("fake vast: {path}")})),
        }
    }
}

impl Http for FakeVast {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        self.log.borrow_mut().push(req.clone());
        if let Some(p) = req.url.strip_prefix(VAST) {
            if bearer(req).as_deref() != Some(VAST_KEY) {
                return Ok(resp(401, json!({"error": "bad key"})));
            }
            let r = self.rest(req, p.split('?').next().unwrap_or(p));
            if req.method == Method::Put && self.lose_create.get() {
                return Err(SysError::Http {
                    url: req.url.clone(),
                    why: "connection reset".into(),
                });
            }
            return Ok(r);
        }
        if let Some(p) = req.url.strip_prefix(&format!("http://{VAST_IP}:41000")) {
            return Ok(self.serve.answer(req, p));
        }
        Err(SysError::Http {
            url: req.url.clone(),
            why: "fake vast: unknown host".into(),
        })
    }

    fn download(&self, url: &str, _d: &Path, _t: Duration) -> Result<u64, SysError> {
        Err(SysError::Http {
            url: url.into(),
            why: "no downloads".into(),
        })
    }
}

fn vast_launch() -> Launch {
    let mut g = guard::default_guards();
    g.max_usd_per_hr = guard::default_max_usd_per_hr(ProviderKind::Vast);
    let mut l = script_vast(&ScriptVast {
        name: "fv-deploy-fake-1025120000",
        image: IMAGE,
        key_hash: "hash",
        config: "/etc/fv/runpod-fake.toml",
        disk_gb: 40,
        guards: g,
    });
    l.args.clear();
    l.env
        .insert("FV_WORKER_TOML_B64".into(), "W3NlcnZlcl0K".into());
    l
}

fn vast_leaks(s: &str) -> bool {
    s.contains(VAST_KEY)
}

#[test]
fn vast_up_rents_the_cheapest_offer_under_the_cap_then_smokes_and_deletes() {
    let fv = FakeVast::new();
    let clock = FakeClock::at(1_000.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let r = deployer(&p, &clock, &log, &bs, &vast_leaks)
        .up(&vast_launch(), &up_opts())
        .unwrap_or_else(|e| panic!("{e}: {:?}", log.0.borrow()));
    assert!(r.ok, "{r:?}");
    let inst = r.instance.clone().unwrap_or_else(|| panic!("no instance"));
    assert_eq!(inst.gpu.as_deref(), Some("RTX 3090"));
    assert_eq!(inst.usd_per_hr, Some(0.21));
    assert_eq!(
        r.smoke.as_ref().map(|s| s.job.status.as_str()),
        Some("succeeded")
    );
    assert!(r.deleted && fv.live().is_empty());
    assert_eq!(
        bs.0.borrow()[0],
        (ProviderKind::Vast, inst.id.clone(), 1_000.0 + 1800.0)
    );
    // Balance first, then the quote, then the create (the script's PUT on
    // the cheapest offer, with the script's body).
    let calls: Vec<String> = fv
        .log
        .borrow()
        .iter()
        .filter(|r| r.url.starts_with(VAST))
        .map(|r| format!("{} {}", r.method.as_str(), &r.url[VAST.len()..]))
        .collect();
    assert_eq!(
        &calls[..4],
        [
            "GET /users/current/",
            "POST /bundles/",
            "POST /bundles/",
            "PUT /asks/12/"
        ]
    );
    let put = fv
        .log
        .borrow()
        .iter()
        .find(|r| r.method == Method::Put)
        .map(body_of)
        .unwrap_or(Value::Null);
    assert_eq!(put["label"], "fv-deploy-fake-1025120000");
    assert_eq!(put["env"]["-p 8000:8000"], "1");
    // The create's search is the script's, exactly (no `gpu_name`: no
    // --gpu; the UNVERIFIED filter is never sent unasked).
    let search = fv
        .log
        .borrow()
        .iter()
        .filter(|r| r.url.ends_with("/bundles/"))
        .map(body_of)
        .nth(1)
        .unwrap_or(Value::Null);
    assert!(search.get("gpu_name").is_none(), "{search}");
    let golden: Value =
        serde_json::from_str(include_str!("../../fixtures/deploy/vast-offers.json"))
            .unwrap_or(Value::Null);
    assert_eq!(num(&search), num(&golden));
    // The key is a header only.
    for r in fv.log.borrow().iter() {
        assert!(!r.url.contains(VAST_KEY));
        let b = String::from_utf8_lossy(r.body.as_deref().unwrap_or_default()).into_owned();
        assert!(!b.contains(VAST_KEY));
    }
    assert!(!format!("{p:?}").contains(VAST_KEY));
}

#[test]
fn vast_over_cap_offers_are_never_rented() {
    let fv = FakeVast::new();
    fv.offers.borrow_mut().retain(|o| o.2 > 1.0);
    let clock = FakeClock::at(0.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let e = deployer(&p, &clock, &log, &bs, &vast_leaks)
        .up(&vast_launch(), &up_opts())
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("over the cap"), "{e}");
    assert!(fv.log.borrow().iter().all(|r| r.method != Method::Put));
}

#[test]
fn vast_below_the_floor_rents_nothing() {
    let fv = FakeVast::new();
    fv.credit.set(7.5);
    let clock = FakeClock::at(0.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let e = deployer(&p, &clock, &log, &bs, &vast_leaks).up(&vast_launch(), &up_opts());
    assert!(
        matches!(
            e,
            Err(DeployError::Guard(GuardError::BalanceBelowFloor { .. }))
        ),
        "{e:?}"
    );
    assert!(fv.log.borrow().iter().all(|r| r.method != Method::Put));
}

#[test]
fn vast_lost_create_answer_is_swept_by_label() {
    let fv = FakeVast::new();
    fv.lose_create.set(true);
    let clock = FakeClock::at(0.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let e = deployer(&p, &clock, &log, &bs, &vast_leaks).up(&vast_launch(), &up_opts());
    assert!(
        matches!(e, Err(DeployError::CreateUncertain { .. })),
        "{e:?}"
    );
    assert!(fv.live().is_empty(), "{:?}", fv.live());
}

#[test]
fn vast_status_logs_and_refusals() {
    let fv = FakeVast::new();
    let clock = FakeClock::at(0.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    // Gone (404) is Terminated, and its delete is fine.
    assert_eq!(
        p.status(&InstanceId::new("1"))
            .unwrap_or(InstanceState::Unknown),
        InstanceState::Terminated
    );
    assert!(p.destroy(&InstanceId::new("1")).is_ok());
    // A refused ask (`{"success": false}`, UNVERIFIED shape) is a definite
    // refusal: nothing was created, the lifecycle tries the next type.
    fv.refuse_asks.set(true);
    let e = p.create(&vast_launch()).err();
    assert!(
        e.as_ref().is_some_and(ProviderError::is_definite_refusal),
        "{e:?}"
    );
    assert!(fv.live().is_empty());
    fv.refuse_asks.set(false);
    // Logs are not read on Vast (no API in the scripts): a clear refusal.
    assert!(p.logs(&InstanceId::new("1"), None).is_err());
    // A secret name in the env is refused before any request.
    let mut s = vast_launch();
    s.env.insert("FV_CF_API_TOKEN".into(), "x".into());
    let n = fv.log.borrow().len();
    let e = p
        .create(&s)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("Vast account env var"), "{e}");
    assert_eq!(fv.log.borrow().len(), n);
    // An instance that reports no address yet is down, then reachable.
    let inst = p.create(&vast_launch()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(inst.usd_per_hr, Some(0.21));
    // (The create read the instance's price; it is loading again.)
    if let Some(e) = fv.instances.borrow_mut().get_mut(inst.id.as_str()) {
        e.1 = "loading".into();
    }
    let mut r = p
        .reach(
            &inst,
            &ReachOptions {
                key: Some(RUN_KEY),
                ..ReachOptions::default()
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(r.ping(), Probe::Down(w) if w.contains("loading")));
    assert_eq!(r.ping(), Probe::Down("/ping 502".into()));
    assert_eq!(p.account().map(|a| a.balance_usd).unwrap_or(0.0), 25.0);
}

// ------------------------------------------------------------------ GCP

const COMPUTE: &str = "https://compute.test/compute/v1";
const GCP_IP: &str = "203.0.113.9";
const ZONE: &str = "europe-west4-b";

/// A fake Compute Engine: instances, firewall rules, the weights disk,
/// operations that are done at the first wait, the serial port.
struct FakeGcp {
    instances: RefCell<BTreeMap<String, Value>>,
    firewalls: RefCell<BTreeMap<String, Value>>,
    disk: Cell<bool>,
    /// The instance insert's operation ends in an error (no capacity).
    insert_fails: Cell<bool>,
    /// The insert is carried out but its answer is lost.
    lose_insert: Cell<bool>,
    /// `GET` of an instance answers 403 (a lookup refused after the insert).
    get_forbidden: Cell<bool>,
    ops: Cell<u32>,
    serve: FakeServe,
    log: RefCell<Vec<Request>>,
}

impl FakeGcp {
    fn new() -> Self {
        Self {
            instances: RefCell::new(BTreeMap::new()),
            firewalls: RefCell::new(BTreeMap::new()),
            disk: Cell::new(true),
            insert_fails: Cell::new(false),
            lose_insert: Cell::new(false),
            get_forbidden: Cell::new(false),
            ops: Cell::new(0),
            serve: FakeServe::new(&[502, 200]),
            log: RefCell::new(Vec::new()),
        }
    }

    fn op(&self, error: bool) -> Response {
        self.ops.set(self.ops.get() + 1);
        let name = if error {
            format!("op-err-{}", self.ops.get())
        } else {
            format!("op-{}", self.ops.get())
        };
        resp(
            200,
            json!({"name": name, "status": "RUNNING", "selfLink": format!("{COMPUTE}/projects/p/zones/{ZONE}/operations/{name}")}),
        )
    }

    fn api(&self, req: &Request, path: &str) -> Response {
        let b = body_of(req);
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        match (req.method, parts.as_slice()) {
            (Method::Post, ["projects", "p", "zones", _, "operations", op, "wait"]) => {
                if op.starts_with("op-err") {
                    resp(
                        200,
                        json!({"status": "DONE", "error": {"errors": [{"code": "ZONE_RESOURCE_POOL_EXHAUSTED"}]}}),
                    )
                } else {
                    resp(200, json!({"status": "DONE"}))
                }
            }
            (Method::Get, ["projects", "p", "aggregated", "instances"]) => resp(
                200,
                json!({"items": {format!("zones/{ZONE}"): {"instances": self.instances.borrow().values().cloned().collect::<Vec<_>>()}}}),
            ),
            (Method::Get, ["projects", "p", "zones", _, "disks", d]) => {
                if self.disk.get() && d.starts_with("fv-weights-hdml-") {
                    resp(
                        200,
                        json!({"name": d, "labels": {"fv-owner": "fastvideo-rs"}}),
                    )
                } else {
                    resp(404, json!({"error": {"message": "disk not found"}}))
                }
            }
            (Method::Post, ["projects", "p", "global", "firewalls"]) => {
                let n = b["name"].as_str().unwrap_or("").to_owned();
                self.firewalls.borrow_mut().insert(n, b);
                self.op(false)
            }
            (Method::Get, ["projects", "p", "global", "firewalls", n]) => {
                match self.firewalls.borrow().get(*n) {
                    Some(f) => resp(200, f.clone()),
                    None => resp(404, json!({"error": {"message": "not found"}})),
                }
            }
            (Method::Delete, ["projects", "p", "global", "firewalls", n]) => {
                self.firewalls.borrow_mut().remove(*n);
                self.op(false)
            }
            (Method::Post, ["projects", "p", "zones", _, "instances"]) => {
                if self.insert_fails.get() {
                    return self.op(true);
                }
                let n = b["name"].as_str().unwrap_or("").to_owned();
                let mut inst = b.clone();
                inst["status"] = json!("RUNNING");
                inst["zone"] = json!(format!("{COMPUTE}/projects/p/zones/{ZONE}"));
                inst["networkInterfaces"][0]["accessConfigs"][0]["natIP"] = json!(GCP_IP);
                self.instances.borrow_mut().insert(n, inst);
                self.op(false)
            }
            (Method::Get, ["projects", "p", "zones", _, "instances", _])
                if self.get_forbidden.get() =>
            {
                resp(403, json!({"error": {"message": "forbidden"}}))
            }
            (Method::Get, ["projects", "p", "zones", _, "instances", n]) => {
                match self.instances.borrow().get(*n) {
                    Some(i) => resp(200, i.clone()),
                    None => resp(404, json!({"error": {"message": "not found"}})),
                }
            }
            (Method::Delete, ["projects", "p", "zones", _, "instances", n]) => {
                match self.instances.borrow_mut().remove(*n) {
                    Some(_) => self.op(false),
                    None => resp(404, json!({"error": {"message": "not found"}})),
                }
            }
            (Method::Get, ["projects", "p", "zones", _, "instances", _, "serialPort"]) => resp(
                200,
                json!({"contents": "FV-GCP STARTED\nfv-serve: admin token fvadm_s3cr3t-Tok ok\n", "next": "4242"}),
            ),
            _ => resp(
                404,
                json!({"error": {"message": format!("fake gcp: {path}")}}),
            ),
        }
    }
}

impl Http for FakeGcp {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        self.log.borrow_mut().push(req.clone());
        if let Some(p) = req.url.strip_prefix(COMPUTE) {
            if bearer(req).as_deref() != Some(GCP_TOKEN) {
                return Ok(resp(401, json!({"error": {"message": "no token"}})));
            }
            let path = p.split('?').next().unwrap_or(p);
            let r = self.api(req, path);
            if req.method == Method::Post && path.ends_with("/instances") && self.lose_insert.get()
            {
                return Err(SysError::Http {
                    url: req.url.clone(),
                    why: "connection reset".into(),
                });
            }
            return Ok(r);
        }
        if let Some(p) = req.url.strip_prefix(&format!("http://{GCP_IP}:8000")) {
            return Ok(self.serve.answer(req, p));
        }
        Err(SysError::Http {
            url: req.url.clone(),
            why: "fake gcp: unknown host".into(),
        })
    }

    fn download(&self, url: &str, _d: &Path, _t: Duration) -> Result<u64, SysError> {
        Err(SysError::Http {
            url: url.into(),
            why: "no downloads".into(),
        })
    }
}

fn gcp_launch(fake: bool) -> Launch {
    script_gcp(&ScriptGcp {
        name: "fv-deploy-h3-turbo-1025120000",
        image: IMAGE,
        key_hash: "hash",
        config_text: "[server]\n",
        machine: "g4-standard-48",
        zone: ZONE,
        fake,
        guards: guard::default_guards(),
    })
}

fn gcp_opts() -> GcpOptions {
    GcpOptions {
        family: "h3-turbo".into(),
        run: "1025120000".into(),
        verify_cells: vec!["fasth3-4step-vsa".into(), "aux".into()],
        spot: false,
        idle_s: 1800,
        source_ranges: vec!["198.51.100.7/32".into()],
        secret_manager: false,
    }
}

fn gcp_vm<'a>(
    fg: &'a FakeGcp,
    clock: &'a FakeClock,
    tok: &'a StaticToken,
    budget: Option<f64>,
) -> GcpVm<'a> {
    let mut api = GcpApi::new(fg, clock, tok, "p", ZONE);
    api.compute = COMPUTE.into();
    api.budget_usd = budget;
    GcpVm::new(
        api,
        gcp_opts(),
        GcpAssets {
            startup_script: "#!/bin/bash\n".into(),
            ..GcpAssets::default()
        },
        GcpPrices(vec![PriceRow {
            machine: "g4-standard-48".into(),
            region: "europe-west4".into(),
            on_demand: 4.9499,
            spot: 2.2091,
        }]),
    )
}

fn gcp_leaks(s: &str) -> bool {
    s.contains(GCP_TOKEN)
}

fn gcp_guards() -> Guards {
    let mut g = guard::default_guards();
    g.max_usd_per_hr = guard::default_max_usd_per_hr(ProviderKind::Gcp);
    g
}

#[test]
fn gcp_up_creates_rule_and_vm_smokes_and_deletes_both() {
    let fg = FakeGcp::new();
    let clock = FakeClock::at(2_000.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, Some(20.0));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut l = gcp_launch(false);
    l.guards = gcp_guards();
    let r = deployer(&vm, &clock, &log, &bs, &gcp_leaks)
        .up(&l, &up_opts())
        .unwrap_or_else(|e| panic!("{e}: {:?}", log.0.borrow()));
    assert!(r.ok, "{r:?}");
    let inst = r.instance.clone().unwrap_or_else(|| panic!("no instance"));
    assert_eq!(inst.url.as_deref(), Some("http://203.0.113.9:8000"));
    assert_eq!(inst.usd_per_hr, Some(4.9499));
    assert!(r.deleted);
    assert!(fg.instances.borrow().is_empty() && fg.firewalls.borrow().is_empty());
    assert_eq!(
        bs.0.borrow()[0],
        (
            ProviderKind::Gcp,
            InstanceId::new(&l.name),
            2_000.0 + 1800.0
        )
    );
    // The disk check, then the rule, then the VM; each operation waited on.
    let calls: Vec<String> = fg
        .log
        .borrow()
        .iter()
        .filter(|r| r.url.starts_with(COMPUTE) && !r.url.ends_with("/wait"))
        .map(|r| {
            let p = &r.url[COMPUTE.len()..];
            format!(
                "{} {}",
                r.method.as_str(),
                p.rsplit('/').next().unwrap_or(p)
            )
        })
        .collect();
    // The account (budget less committed, our live VMs' spend) first.
    assert_eq!(
        &calls[..5],
        [
            "GET instances?filter=labels.fv-owner%3Dfastvideo-rs",
            "GET fv-weights-hdml-europe-west4-b",
            "POST firewalls",
            "POST instances",
            "GET fv-deploy-h3-turbo-1025120000"
        ]
    );
    // The token is a header only, never in a URL, a body or `Debug`.
    for r in fg.log.borrow().iter() {
        assert!(!r.url.contains(GCP_TOKEN));
        let b = String::from_utf8_lossy(r.body.as_deref().unwrap_or_default()).into_owned();
        assert!(!b.contains(GCP_TOKEN));
    }
    assert!(!format!("{vm:?}").contains(GCP_TOKEN));
}

#[test]
fn gcp_without_a_budget_creates_nothing() {
    let fg = FakeGcp::new();
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, None);
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut l = gcp_launch(false);
    l.guards = gcp_guards();
    let e = deployer(&vm, &clock, &log, &bs, &gcp_leaks)
        .up(&l, &up_opts())
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("FV_GCP_BUDGET_USD"), "{e}");
    assert!(fg.log.borrow().iter().all(|r| r.method != Method::Post));
}

#[test]
fn gcp_budget_floor_and_price_cap_hold() {
    let fg = FakeGcp::new();
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    // $10 − $4.95 × 0.5 h < $8: refused before any create.
    let vm = gcp_vm(&fg, &clock, &tok, Some(10.0));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut l = gcp_launch(false);
    l.guards = gcp_guards();
    let e = deployer(&vm, &clock, &log, &bs, &gcp_leaks).up(&l, &up_opts());
    assert!(
        matches!(
            e,
            Err(DeployError::Guard(
                GuardError::BalanceWouldDropBelowFloor { .. }
            ))
        ),
        "{e:?}"
    );
    // A machine over the cap, and one the table does not list, are never created.
    let vm = gcp_vm(&fg, &clock, &tok, Some(100.0));
    l.guards.max_usd_per_hr = 4.0;
    let e = deployer(&vm, &clock, &log, &bs, &gcp_leaks)
        .up(&l, &up_opts())
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("over the cap"), "{e}");
    l.guards = gcp_guards();
    l.gpu.types = vec!["a2-unlisted-1g".into()];
    let e = deployer(&vm, &clock, &log, &bs, &gcp_leaks)
        .up(&l, &up_opts())
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("no quote"), "{e}");
    assert!(fg.log.borrow().iter().all(|r| r.method != Method::Post));
}

#[test]
fn gcp_failed_insert_removes_its_rule_and_a_missing_disk_creates_nothing() {
    let fg = FakeGcp::new();
    fg.insert_fails.set(true);
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, Some(50.0));
    let e = vm.create(&gcp_launch(false)).err();
    assert!(
        e.as_ref().is_some_and(ProviderError::is_definite_refusal),
        "{e:?}"
    );
    assert!(fg.firewalls.borrow().is_empty() && fg.instances.borrow().is_empty());
    fg.insert_fails.set(false);
    fg.disk.set(false);
    let n = fg.log.borrow().len();
    let e = vm
        .create(&gcp_launch(false))
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("weights disk"), "{e}");
    assert!(fg.log.borrow()[n..].iter().all(|r| r.method == Method::Get));
    // The fake engine mounts no disk: no disk check.
    assert!(vm.create(&gcp_launch(true)).is_ok());
}

#[test]
fn gcp_lost_insert_answer_is_swept_by_name() {
    let fg = FakeGcp::new();
    fg.lose_insert.set(true);
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, Some(50.0));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let mut l = gcp_launch(false);
    l.guards = gcp_guards();
    let e = deployer(&vm, &clock, &log, &bs, &gcp_leaks).up(&l, &up_opts());
    assert!(
        matches!(e, Err(DeployError::CreateUncertain { .. })),
        "{e:?}"
    );
    assert!(fg.instances.borrow().is_empty() && fg.firewalls.borrow().is_empty());
}

#[test]
fn gcp_never_deletes_what_is_not_ours_and_reads_the_serial_log() {
    let fg = FakeGcp::new();
    fg.instances.borrow_mut().insert(
        "someone-else".into(),
        json!({"name": "someone-else", "status": "RUNNING", "labels": {"team": "other"}}),
    );
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, Some(50.0));
    let e = vm
        .destroy(&InstanceId::new("someone-else"))
        .err()
        .map(|e| e.to_string());
    assert!(e.is_some_and(|e| e.contains("refusing")));
    assert!(fg.instances.borrow().contains_key("someone-else"));
    assert!(fg.log.borrow().iter().all(|r| r.method != Method::Delete));
    // Gone is fine, and is what the backstop's early exit reads.
    assert!(vm.destroy(&InstanceId::new("gone")).is_ok());
    assert_eq!(
        vm.status(&InstanceId::new("gone"))
            .unwrap_or(InstanceState::Unknown),
        InstanceState::Terminated
    );
    assert_eq!(
        vm.status(&InstanceId::new("someone-else"))
            .unwrap_or(InstanceState::Unknown),
        InstanceState::Running
    );
    let page = vm
        .logs(&InstanceId::new("someone-else"), None)
        .unwrap_or_default();
    assert_eq!(page.next, Some(Cursor("4242".into())));
    assert!(
        page.lines.iter().any(|l| l.contains("fvadm_<masked> ok")),
        "{:?}",
        page.lines
    );
    assert!(page.lines.iter().all(|l| !l.contains("s3cr3t")));
}

#[test]
fn gcp_maps_states_and_names_its_resources() {
    use super::gcp::{boot_disk_type, instance_state, provisioning, region_of};
    assert_eq!(instance_state("TERMINATED"), InstanceState::Exited);
    assert_eq!(instance_state("STAGING"), InstanceState::Pending);
    assert_eq!(region_of("europe-west4-b"), "europe-west4");
    assert_eq!(boot_disk_type("g4-standard-48"), "hyperdisk-balanced");
    assert_eq!(boot_disk_type("g2-standard-8"), "pd-balanced");
    assert_eq!(provisioning("a3-highgpu-1g", false), "SPOT");
    assert_eq!(provisioning("g4-standard-48", false), "STANDARD");
    assert_eq!(provisioning("g4-standard-48", true), "SPOT");
    // A US zone is refused (EU only), the wrong disk too.
    let mut l = gcp_launch(false);
    l.placement.data_centers = vec!["us-central1-a".into()];
    assert!(guard::check_region(&l).is_err());
    let mut l = gcp_launch(false);
    if let Some(v) = l.placement.volume.as_mut() {
        v.id = "someone-elses-disk".into();
    }
    assert!(guard::check_region(&l).is_err());
    // The config rides as fv-config, never as FV_WORKER_TOML_B64 in fv-env.
    let fg = FakeGcp::new();
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, Some(50.0));
    let calls = vm.payload(&gcp_launch(false)).unwrap_or_default();
    let items = calls[1].body["metadata"]["items"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let get = |k: &str| {
        items
            .iter()
            .find(|i| i["key"] == k)
            .map(|i| i["value"].clone())
    };
    assert_eq!(get("fv-config"), Some(json!("[server]\n")));
    assert!(!get("fv-env")
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
        .contains("FV_WORKER_TOML_B64"));
    assert_eq!(
        base64::engine::general_purpose::STANDARD.encode("[server]\n"),
        gcp_launch(false).env["FV_WORKER_TOML_B64"]
    );
    // A launch without an inline config cannot be sent.
    let mut l = gcp_launch(false);
    l.env.remove("FV_WORKER_TOML_B64");
    assert!(vm.payload(&l).is_err());
    let mut shown = calls[1].body.clone();
    super::gcp::redact_for_display(&mut shown);
    assert!(shown.to_string().contains("<12 chars: #!/bin/bash...>"));
}

#[test]
fn every_kind_parses_and_names_itself() {
    for k in ProviderKind::ALL {
        assert_eq!(ProviderKind::parse(k.as_str()), Some(k));
    }
    assert_eq!(ProviderKind::parse("vast"), Some(ProviderKind::Vast));
    assert_eq!(ProviderKind::parse("gcp"), Some(ProviderKind::Gcp));
}

/// The VM exists once its insert is done: an address lookup refused right
/// after is not a failed create (that would read as a definite refusal and
/// leave the VM to the exit sweep); the probe looks the address up later.
#[test]
fn gcp_a_refused_address_lookup_after_the_insert_is_not_a_failed_create() {
    let fg = FakeGcp::new();
    fg.get_forbidden.set(true);
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, Some(50.0));
    let inst = vm
        .create(&gcp_launch(true))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(inst.url, None);
    assert!(fg.instances.borrow().contains_key(&inst.name));
}

// ------------------------------------------------------------------ review of #193

/// Review 3: an unexpected listing is an error, never "nothing left" (the
/// sweep is Vast's only protection after a lost create answer).
#[test]
fn vast_sweep_refuses_a_listing_it_cannot_read() {
    let fv = FakeVast::new();
    let clock = FakeClock::at(0.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    p.create(&vast_launch()).unwrap_or_else(|e| panic!("{e}"));
    fv.unlabelled.set(true);
    let e = p.sweep("fv-deploy-fake-1025120000", None);
    assert!(matches!(e, Err(ProviderError::Decode { .. })), "{e:?}");
    assert_eq!(fv.live().len(), 1);
}

/// Review 3: a lost answer whose instance shows in the listing only later
/// is still found: the sweep is retried (6 times, 18 s apart).
#[test]
fn vast_lost_answer_listed_late_is_swept_by_a_later_sweep() {
    let fv = FakeVast::new();
    fv.lose_create.set(true);
    fv.hide_listings.set(3);
    let clock = FakeClock::at(0.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let e = deployer(&p, &clock, &log, &bs, &vast_leaks).up(&vast_launch(), &up_opts());
    match e {
        Err(DeployError::CreateUncertain { swept, .. }) => assert!(swept.contains("90"), "{swept}"),
        other => panic!("{other:?}"),
    }
    assert!(fv.live().is_empty(), "{:?}", fv.live());
    assert!(clock.now() >= 3.0 * 18.0, "{}", clock.now());
}

/// Review 4: `--gpu` names narrow the search (the UNVERIFIED filter is
/// sent only then).
#[test]
fn vast_names_gpus_only_when_asked() {
    let fv = FakeVast::new();
    let clock = FakeClock::at(0.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    let any = p.offers(&OfferQuery::default()).unwrap_or_default();
    assert_eq!(any.len(), 1);
    assert_eq!(
        (any[0].gpu_type.as_str(), any[0].usd_per_hr),
        (super::vast::ANY_GPU, Some(0.21))
    );
    let mut l = vast_launch();
    l.gpu.types = vec!["RTX 4090".into()];
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let r = deployer(&p, &clock, &log, &bs, &vast_leaks)
        .up(&l, &up_opts())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(r.instance.and_then(|i| i.gpu).as_deref(), Some("RTX 4090"));
    let search = fv
        .log
        .borrow()
        .iter()
        .filter(|r| r.url.ends_with("/bundles/"))
        .map(body_of)
        .next_back()
        .unwrap_or(Value::Null);
    assert_eq!(search["gpu_name"]["in"], json!(["RTX 4090"]));
}

/// Review 9, 10, 11: the instance's own price when reported; a debt fails
/// the floor; a refused delete of a gone instance is done, of a live one an
/// error.
#[test]
fn vast_billed_price_debt_and_refused_deletes() {
    let fv = FakeVast::new();
    fv.inst_dph.set(Some(0.27));
    let clock = FakeClock::at(0.0);
    let p = Vast::new(VastApi::new(&fv, &clock, VAST_KEY, VAST));
    let inst = p.create(&vast_launch()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(inst.usd_per_hr, Some(0.27));
    fv.credit.set(-3.0);
    let a = p.account().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a.balance_usd, -3.0);
    assert!(guard::check_balance(&a, 8.0).is_err());
    fv.refuse_deletes.set(true);
    assert!(
        p.destroy(&inst.id).is_err(),
        "a live instance is not deleted"
    );
    assert!(p.destroy(&InstanceId::new("424242")).is_ok(), "gone");
}

/// Review 2: the budget is per run and shrinks by what each create
/// committed (price × cap): the second VM of a $28 run at a 4 h cap is
/// refused by the floor (28 − 19.8 = 8.2 left, less 19.8 more < 8).
#[test]
fn gcp_budget_shrinks_with_each_create() {
    let fg = FakeGcp::new();
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, Some(28.0));
    let mut l = gcp_launch(true);
    l.guards = gcp_guards();
    l.guards.cap_s = 14_400;
    let a0 = vm.account().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((a0.balance_usd, a0.spend_per_hr_usd), (28.0, Some(0.0)));
    assert!(guard::check_projection(&a0, 4.9499, l.guards.cap_s, 8.0).is_ok());
    vm.create(&l).unwrap_or_else(|e| panic!("{e}"));
    let a1 = vm.account().unwrap_or_else(|e| panic!("{e}"));
    assert!(
        (a1.balance_usd - (28.0 - 4.9499 * 4.0)).abs() < 1e-9,
        "{a1:?}"
    );
    // The live VM is the account's current spend.
    assert_eq!(a1.spend_per_hr_usd, Some(4.9499));
    assert!(guard::check_projection(&a1, 4.9499, l.guards.cap_s, 8.0).is_err());
}

/// Review 1: `down` of an instance the provider does not know (a wrong
/// zone) is "not found", never "deleted".
#[test]
fn down_of_an_unknown_instance_is_not_found() {
    let fg = FakeGcp::new();
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let vm = gcp_vm(&fg, &clock, &tok, Some(50.0));
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let d = deployer(&vm, &clock, &log, &bs, &gcp_leaks);
    let e = d.down(&InstanceId::new("fv-elsewhere"));
    assert!(matches!(e, Err(DeployError::NotFound(_))), "{e:?}");
    let inst = vm
        .create(&gcp_launch(true))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(d.down(&inst.id).is_ok());
    assert!(fg.instances.borrow().is_empty());
}

/// Review 5, 7: NaN never passes a floor or a cap; an empty source range is
/// refused before any request.
#[test]
fn nan_fails_closed_and_an_empty_source_range_is_refused() {
    let a = Account {
        provider: ProviderKind::Gcp,
        balance_usd: f64::NAN,
        spend_per_hr_usd: None,
    };
    assert!(guard::check_balance(&a, 8.0).is_err());
    let ok = Account {
        balance_usd: 100.0,
        ..a.clone()
    };
    assert!(guard::check_projection(&ok, f64::NAN, 1800, 8.0).is_err());
    assert!(guard::check_projection(&a, 1.0, 1800, 8.0).is_err());
    let o = Offer {
        provider: ProviderKind::Gcp,
        gpu_type: "m".into(),
        data_center: None,
        cloud: CloudType::Secure,
        usd_per_hr: Some(f64::NAN),
        stock: None,
    };
    assert!(guard::check_quote(&o, 6.0).is_err());
    let fg = FakeGcp::new();
    let clock = FakeClock::at(0.0);
    let tok = StaticToken(GCP_TOKEN.into());
    let mut vm = gcp_vm(&fg, &clock, &tok, Some(50.0));
    vm.options.source_ranges = vec![" ".into()];
    assert!(vm.create(&gcp_launch(true)).is_err());
    assert!(vm.payload(&gcp_launch(true)).is_err());
    assert!(fg.log.borrow().is_empty());
}
