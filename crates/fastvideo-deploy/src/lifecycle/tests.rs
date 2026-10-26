//! The shared guards and the lifecycle against a provider-neutral fake
//! ([`FakeProvider`]): nothing here knows Runpod, so every provider that
//! implements the trait gets the same behaviour.

use std::cell::{Cell, RefCell};
use std::time::Duration;

use serde_json::{json, Value};

use super::*;
use crate::guard::{check_region, run_backstop, BackstopOutcome, GuardError};
use crate::provider::tests::{eu_pod_launch, FakeClock, Lines, RecBackstop};
use crate::provider::{
    Account, ApiCall, Cursor, Instance, InstanceId, InstanceState, Launch, LogPage, Offer,
    OfferQuery, Probe, Provider, ProviderError, ProviderKind, Reach, ReachOptions, VolumeMount,
};

/// A provider with programmable money and readiness.
struct FakeProvider<'c> {
    clock: &'c FakeClock,
    balance: Cell<f64>,
    price: Cell<Option<f64>>,
    /// Seconds after creation when an instance answers `/ping` 200 (`None`: never).
    ready_after: Cell<Option<f64>>,
    /// The first N instances never get ready.
    bad_first: Cell<usize>,
    created: RefCell<Vec<(InstanceId, f64)>>,
    destroyed: RefCell<Vec<InstanceId>>,
    calls: RefCell<Vec<String>>,
    /// The billed price `create` reports (default: the quote).
    billed: Cell<Option<Option<f64>>>,
    /// `status` alternates gone / running (a flapping API).
    flap: Cell<bool>,
    flips: Cell<u32>,
    /// The job never finishes.
    stuck_job: Cell<bool>,
}

impl<'c> FakeProvider<'c> {
    fn new(clock: &'c FakeClock) -> Self {
        Self {
            clock,
            balance: Cell::new(50.0),
            price: Cell::new(Some(1.0)),
            ready_after: Cell::new(Some(30.0)),
            bad_first: Cell::new(0),
            created: RefCell::new(Vec::new()),
            destroyed: RefCell::new(Vec::new()),
            calls: RefCell::new(Vec::new()),
            billed: Cell::new(None),
            flap: Cell::new(false),
            flips: Cell::new(0),
            stuck_job: Cell::new(false),
        }
    }

    fn live(&self) -> usize {
        self.created.borrow().len() - self.destroyed.borrow().len()
    }
}

struct FakeReach<'a> {
    p: &'a FakeProvider<'a>,
    born: f64,
    /// 1-based creation order.
    nth: usize,
    polls: u32,
}

impl Reach for FakeReach<'_> {
    fn ping(&mut self) -> Probe {
        if self.nth <= self.p.bad_first.get() {
            return Probe::Down("/ping 502".into());
        }
        match self.p.ready_after.get() {
            Some(t) if self.p.clock.now() - self.born >= t => Probe::Ready,
            Some(_) => Probe::Loading,
            None => Probe::Down("/ping 502".into()),
        }
    }

    fn call(
        &mut self,
        method: crate::boot::sys::Method,
        path: &str,
        _body: Option<&Value>,
        _stop: &dyn Fn() -> bool,
    ) -> Result<(u16, Value), ProviderError> {
        self.p
            .calls
            .borrow_mut()
            .push(format!("{} {path}", method.as_str()));
        Ok(match path {
            "/healthz" => (200, json!({"state": "ready"})),
            "/fv/v1/capabilities" => (
                200,
                json!({"models": [{"caps": {"id": "ltx25-distill-sol"}}]}),
            ),
            "/fv/v1/jobs" => (202, json!({"id": "j1"})),
            _ => {
                self.polls += 1;
                let done = self.polls > 1 && !self.p.stuck_job.get();
                (
                    200,
                    json!({"id": "j1", "status": if done { "succeeded" } else { "running" }}),
                )
            }
        })
    }
}

impl Provider for FakeProvider<'_> {
    fn kind(&self) -> ProviderKind {
        ProviderKind::RunpodPod
    }
    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>, ProviderError> {
        self.calls.borrow_mut().push("offers".into());
        Ok(q.gpu_types
            .iter()
            .map(|g| Offer {
                provider: self.kind(),
                gpu_type: g.clone(),
                data_center: q.data_center.clone(),
                cloud: q.cloud,
                usd_per_hr: self.price.get(),
                stock: None,
            })
            .collect())
    }
    fn payload(&self, _l: &Launch) -> Result<Vec<ApiCall>, ProviderError> {
        Ok(Vec::new())
    }
    fn create(&self, l: &Launch) -> Result<Instance, ProviderError> {
        self.calls.borrow_mut().push("create".into());
        let id = InstanceId::new(format!("i{}", self.created.borrow().len() + 1));
        let now = self.clock.now();
        self.created.borrow_mut().push((id.clone(), now));
        Ok(Instance {
            id,
            provider: self.kind(),
            name: l.name.clone(),
            state: InstanceState::Pending,
            url: None,
            gpu: l.gpu.types.first().cloned(),
            data_center: None,
            usd_per_hr: self.billed.get().unwrap_or(self.price.get()),
            created_at: Some(now),
            memory_gb: None,
            vcpus: None,
        })
    }
    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError> {
        if self.flap.get() {
            self.flips.set(self.flips.get() + 1);
            return Ok(if self.flips.get() % 2 == 1 {
                InstanceState::Terminated
            } else {
                InstanceState::Running
            });
        }
        Ok(if self.destroyed.borrow().contains(id) {
            InstanceState::Terminated
        } else {
            InstanceState::Running
        })
    }
    fn logs(&self, _id: &InstanceId, _c: Option<&Cursor>) -> Result<LogPage, ProviderError> {
        Ok(LogPage::default())
    }
    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError> {
        self.calls.borrow_mut().push(format!("destroy {id}"));
        if !self.destroyed.borrow().contains(id) {
            self.destroyed.borrow_mut().push(id.clone());
        }
        Ok(())
    }
    fn account(&self) -> Result<Account, ProviderError> {
        self.calls.borrow_mut().push("account".into());
        Ok(Account {
            provider: self.kind(),
            balance_usd: self.balance.get(),
            spend_per_hr_usd: None,
        })
    }
    fn sweep(
        &self,
        name: &str,
        keep: Option<&InstanceId>,
    ) -> Result<Vec<InstanceId>, ProviderError> {
        self.calls.borrow_mut().push(format!("sweep {name}"));
        let live: Vec<InstanceId> = self
            .created
            .borrow()
            .iter()
            .map(|(id, _)| id.clone())
            .filter(|id| !self.destroyed.borrow().contains(id) && Some(id) != keep)
            .collect();
        for id in &live {
            self.destroy(id)?;
        }
        Ok(live)
    }
    fn bills_at_create(&self) -> bool {
        true
    }
    fn reach<'a>(
        &'a self,
        i: &Instance,
        _opts: &ReachOptions<'_>,
    ) -> Result<Box<dyn Reach + 'a>, ProviderError> {
        Ok(Box::new(FakeReach {
            p: self,
            born: i.created_at.unwrap_or(0.0),
            nth: self
                .created
                .borrow()
                .iter()
                .position(|(id, _)| *id == i.id)
                .map_or(0, |n| n + 1),
            polls: 0,
        }))
    }
}

fn never() -> bool {
    false
}
fn no_leak(_: &str) -> bool {
    false
}

type Armed = Vec<(ProviderKind, InstanceId, f64)>;

fn run(
    p: &FakeProvider<'_>,
    clock: &FakeClock,
    l: &Launch,
    o: &UpOptions,
) -> (Result<UpReport, DeployError>, Armed) {
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let d = Deployer {
        provider: p,
        clock,
        log: &log,
        backstop: &bs,
        ledger: None,
        recorder: None,
        interrupted: &never,
        leaks: &no_leak,
        poll: Duration::from_secs(5),
    };
    let r = d.up(l, o);
    let armed = bs.0.borrow().clone();
    (r, armed)
}

fn smoke_up() -> UpOptions {
    UpOptions {
        run_key: None,
        smoke: Some(SmokeOptions::default()),
        keep: false,
    }
}

#[test]
fn create_ready_smoke_down() {
    let clock = FakeClock::at(100.0);
    let p = FakeProvider::new(&clock);
    let (r, armed) = run(&p, &clock, &eu_pod_launch(), &smoke_up());
    let r = r.expect("up");
    assert!(r.ok && r.deleted);
    assert_eq!(r.cycles[0].ready_after_s, Some(30.0));
    let s = r.smoke.expect("smoke");
    assert_eq!(
        (s.job.model.as_deref(), s.job.status.as_str()),
        (Some("ltx25-distill-sol"), "succeeded")
    );
    assert_eq!(
        armed,
        [(
            ProviderKind::RunpodPod,
            InstanceId::new("i1"),
            100.0 + 1800.0
        )]
    );
    assert_eq!(p.live(), 0);
    let calls = p.calls.borrow().clone();
    assert_eq!(&calls[..3], ["account", "offers", "create"]);
    assert_eq!(
        &calls[3..],
        [
            "GET /healthz",
            "GET /fv/v1/capabilities",
            "POST /fv/v1/jobs",
            "GET /fv/v1/jobs/j1",
            "GET /fv/v1/jobs/j1",
            "destroy i1",
            "sweep fv-deploy-test"
        ]
    );
}

#[test]
fn not_ready_is_replaced_after_600_s_and_given_up_after_3() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    p.ready_after.set(None);
    let (r, armed) = run(&p, &clock, &eu_pod_launch(), &smoke_up());
    assert!(
        matches!(
            r,
            Err(DeployError::NotReady {
                cycles: 3,
                timeout_s: 600,
                ..
            })
        ),
        "{r:?}"
    );
    let created: Vec<f64> = p.created.borrow().iter().map(|(_, t)| *t).collect();
    assert_eq!(created, [0.0, 600.0, 1200.0]);
    assert_eq!(p.live(), 0);
    assert_eq!(armed.len(), 3);
}

#[test]
fn ready_on_the_second_cycle() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    p.bad_first.set(1);
    let (r, armed) = run(&p, &clock, &eu_pod_launch(), &smoke_up());
    let r = r.expect("up");
    assert_eq!(r.cycles.len(), 2);
    assert!(
        r.cycles[0].outcome.starts_with("replaced: not ready 600 s"),
        "{:?}",
        r.cycles[0]
    );
    assert_eq!(r.cycles[1].ready_after_s, Some(30.0));
    let created: Vec<f64> = p.created.borrow().iter().map(|(_, t)| *t).collect();
    assert_eq!(created, [0.0, 600.0]);
    assert_eq!(armed.len(), 2);
    assert_eq!(p.live(), 0);
}

#[test]
fn balance_floor_refuses_before_create() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    p.balance.set(7.5);
    let (r, armed) = run(&p, &clock, &eu_pod_launch(), &smoke_up());
    assert!(matches!(
        r,
        Err(DeployError::Guard(GuardError::BalanceBelowFloor { .. }))
    ));
    // Nothing created: the floor came first (the exit sweep only lists).
    assert_eq!(
        p.calls.borrow().clone(),
        ["account", "sweep fv-deploy-test"]
    );
    assert!(armed.is_empty());
}

#[test]
fn price_cap_and_unknown_price_refuse_before_create() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    p.price.set(Some(9.0));
    let (r, _) = run(&p, &clock, &eu_pod_launch(), &smoke_up());
    assert!(
        matches!(r, Err(DeployError::NoCapacity(ref m)) if m.contains("over the cap")),
        "{r:?}"
    );
    p.price.set(None);
    let (r, _) = run(&p, &clock, &eu_pod_launch(), &smoke_up());
    assert!(
        matches!(r, Err(DeployError::NoCapacity(ref m)) if m.contains("unknown price")),
        "{r:?}"
    );
    assert!(p.created.borrow().is_empty());
}

#[test]
fn guard_numbers_are_checked() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    let mut l = eu_pod_launch();
    l.guards.min_balance_usd = 5.0;
    let (r, _) = run(&p, &clock, &l, &smoke_up());
    assert!(matches!(r, Err(DeployError::Guard(GuardError::Invalid(_)))));
    assert!(p.calls.borrow().is_empty());
}

#[test]
fn region_guard_eu_volume_only() {
    let ok = eu_pod_launch();
    assert_eq!(check_region(&ok), Ok(()));
    let mut us = ok.clone();
    us.placement.volume = Some(VolumeMount {
        id: "s2k01690bi".into(),
        mount_path: "/workspace".into(),
    });
    assert!(matches!(check_region(&us), Err(GuardError::UsGone(_))));
    let mut by_name = ok.clone();
    by_name.placement.volume = Some(VolumeMount {
        id: "fv-weights-b200-us".into(),
        mount_path: "/workspace".into(),
    });
    assert!(matches!(check_region(&by_name), Err(GuardError::UsGone(_))));
    let mut region_us = ok.clone();
    region_us.placement.region = Some("us".into());
    assert!(matches!(
        check_region(&region_us),
        Err(GuardError::UsGone(_))
    ));
    let mut other = ok.clone();
    other.placement.volume = Some(VolumeMount {
        id: "jg48s6o1w0".into(),
        mount_path: "/workspace".into(),
    });
    assert!(matches!(check_region(&other), Err(GuardError::Region(_))));
    let mut dc = ok.clone();
    dc.placement.data_centers = vec!["US-CA-2".into()];
    assert!(matches!(check_region(&dc), Err(GuardError::Region(_))));
    let mut none = ok.clone();
    none.placement.volume = None;
    assert!(matches!(check_region(&none), Err(GuardError::Region(_))));
    let mut r2 = ok;
    r2.weights = crate::provider::WeightsSource::R2;
    assert!(matches!(check_region(&r2), Err(GuardError::Region(_))));
}

#[test]
fn backstop_fires_at_the_deadline_through_the_provider() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    let l = eu_pod_launch();
    let i = p.create(&l).expect("create");
    let log = Lines::default();
    let out = run_backstop(&p, &i.id, 1800.0, &clock, &log).expect("backstop");
    assert_eq!(out, BackstopOutcome::Deleted);
    assert_eq!(clock.now(), 1800.0);
    assert_eq!(p.destroyed.borrow().as_slice(), std::slice::from_ref(&i.id));
    assert!(log
        .0
        .borrow()
        .iter()
        .any(|l| l.contains("deadline: deleted i1")));
}

#[test]
fn the_backstop_command_is_detached_and_keyless() {
    let b =
        crate::guard::DetachedBackstop::new("/opt/fv-deploy".into(), None, "/state/armed".into());
    let c = b.command(
        ProviderKind::RunpodServerless,
        &InstanceId::new("ep1/tpl1"),
        1234.2,
    );
    let args: Vec<String> = c
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        args,
        [
            "backstop",
            "--kind",
            "runpod-serverless",
            "--id",
            "ep1/tpl1",
            "--deadline",
            "1235",
            "--armed-file",
            "/state/armed/armed-ep1_tpl1-PID"
        ]
        .map(|a| a.replace("PID", &std::process::id().to_string()))
    );
    assert_eq!(c.get_program(), "/opt/fv-deploy");
    // The key is inherited from the environment, never set on the command.
    assert!(c.get_envs().next().is_none());
}

#[test]
fn smoke_body_and_first_model() {
    let b = smoke_job_body("fasth3", Some(&json!({"seed": 7, "num_frames": 9})));
    assert_eq!(
        b,
        json!({"model": "fasth3", "prompt": "a red fox trotting through fresh snow", "seed": 7, "short_edge": 480, "aspect_ratio": "16:9", "num_frames": 9})
    );
    assert_eq!(
        first_model(&json!({"models": [{"id": "a"}, {"id": "b"}]})).as_deref(),
        Some("a")
    );
    assert_eq!(
        first_model(&json!({"models": [{"caps": {"id": "c"}}]})).as_deref(),
        Some("c")
    );
    assert_eq!(first_model(&json!({"models": []})), None);
}

#[test]
fn ledger_appends_tsv_lines() {
    let dir = std::env::temp_dir().join(format!("fv-deploy-ledger-{}", std::process::id()));
    let l = Ledger {
        path: dir.join("ledger.tsv"),
    };
    l.record(1_790_553_600.0, "runpod-pod-created pod1 x");
    l.record(1_790_553_601.0, "runpod-pod-deleted pod1 done");
    let t = std::fs::read_to_string(&l.path).unwrap_or_default();
    assert_eq!(
        t,
        "2026-09-28T00:00:00Z\trunpod-pod-created pod1 x\n2026-09-28T00:00:01Z\trunpod-pod-deleted pod1 done\n"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn guard_upper_bounds_unless_forced() {
    let mut g = crate::guard::default_guards();
    assert!(crate::guard::check_guards(&g).is_ok());
    for (cap, cycles, timeout) in [(14_401, 3, 600), (1800, 4, 600), (1800, 3, 601)] {
        g.cap_s = Some(cap);
        g.ready_cycles = Some(cycles);
        g.ready_timeout_s = Some(timeout);
        assert!(
            matches!(crate::guard::check_guards(&g), Err(GuardError::Invalid(_))),
            "{cap} {cycles} {timeout}"
        );
        g.force = true;
        assert!(crate::guard::check_guards(&g).is_ok());
        g.force = false;
    }
    // Never the floor.
    g = crate::guard::default_guards();
    g.force = true;
    g.min_balance_usd = 7.0;
    assert!(crate::guard::check_guards(&g).is_err());
}

#[test]
fn every_create_counts_against_the_cycles() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    // Every create is billed over the cap: each one is deleted, and a GPU-type
    // fallback is a create too.
    p.billed.set(Some(Some(9.0)));
    let mut l = eu_pod_launch();
    l.gpu.types = vec!["A".into(), "B".into(), "C".into(), "D".into()];
    l.guards.ready_cycles = Some(2);
    let (r, armed) = run(&p, &clock, &l, &smoke_up());
    assert!(r.is_err());
    assert_eq!(p.created.borrow().len(), 2);
    assert_eq!(armed.len(), 2);
    assert_eq!(p.live(), 0);
}

#[test]
fn unknown_billed_price_fails_the_cap() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    p.billed.set(Some(None));
    let (r, armed) = run(&p, &clock, &eu_pod_launch(), &smoke_up());
    assert!(
        matches!(r, Err(DeployError::NoCapacity(ref m)) if m.contains("no (or no readable) billed price")),
        "{r:?}"
    );
    assert_eq!(armed.len(), 1);
    assert_eq!(p.live(), 0);
}

#[test]
fn the_backstop_does_not_stand_down_on_a_flapping_gone() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    let i = p.create(&eu_pod_launch()).expect("create");
    p.flap.set(true);
    let log = Lines::default();
    let out = run_backstop(&p, &i.id, 600.0, &clock, &log).expect("backstop");
    assert_eq!(out, BackstopOutcome::Deleted);
    assert_eq!(clock.now(), 600.0);
}

#[test]
fn an_interrupt_during_the_smoke_deletes_the_instance() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    p.stuck_job.set(true);
    let (log, bs) = (Lines::default(), RecBackstop::default());
    let stop = || clock.now() > 100.0;
    let d = Deployer {
        provider: &p,
        clock: &clock,
        log: &log,
        backstop: &bs,
        ledger: None,
        recorder: None,
        interrupted: &stop,
        leaks: &no_leak,
        poll: Duration::from_secs(5),
    };
    let r = d.up(&eu_pod_launch(), &smoke_up());
    assert!(matches!(r, Err(DeployError::Interrupted)), "{r:?}");
    assert_eq!(p.live(), 0);
    // Stopped within a poll of the signal, not at the 600 s job wait.
    assert!(clock.now() < 110.0, "{}", clock.now());
}

/// C3d (ADR-0018 §10.4): a guard `null` ("not checked by the launcher that
/// wrote this launch") is refused before anything is created, and a ledger
/// shows it as "not checked"; launches written before (every number) read
/// the same.
#[test]
fn not_checked_guards_are_refused_and_shown() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    let mut l = eu_pod_launch();
    l.guards.max_usd_per_hr = None;
    l.guards.ready_timeout_s = None;
    let (r, armed) = run(&p, &clock, &l, &smoke_up());
    match r {
        Err(DeployError::Guard(GuardError::NotChecked(f))) => {
            assert_eq!(f, ["max_usd_per_hr", "ready_timeout_s"]);
        }
        other => panic!("{other:?}"),
    }
    assert!(p.created.borrow().is_empty() && armed.is_empty());
    assert_eq!(
        l.guards.ledger_view(),
        json!({"max_usd_per_hr": "not checked", "min_balance_usd": 8.0, "cap_s": 1800,
               "ready_timeout_s": "not checked", "ready_cycles": 3})
    );
    assert_eq!(
        l.guards.ledger_line(),
        "max_usd_per_hr=not-checked min_balance_usd=8 cap_s=1800 ready_timeout_s=not-checked ready_cycles=3"
    );
    let v = serde_json::to_value(l.guards).unwrap_or(Value::Null);
    assert_eq!(v["max_usd_per_hr"], Value::Null);
    let old: crate::provider::Guards = serde_json::from_value(json!({
        "max_usd_per_hr": 2.5, "min_balance_usd": 8, "cap_s": 1800, "ready_timeout_s": 600, "ready_cycles": 3
    }))
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(old, crate::guard::default_guards());
    assert!(crate::guard::check_guards(&old).is_ok());
    // A guard key left out is an error at parse (not "not checked"): every
    // launcher states each guard, a number or null.
    let missing = serde_json::from_value::<crate::provider::Guards>(json!({
        "max_usd_per_hr": 2.5, "min_balance_usd": 8, "ready_timeout_s": 600, "ready_cycles": 3
    }));
    assert!(
        missing
            .as_ref()
            .is_err_and(|e| e.to_string().contains("cap_s")),
        "{missing:?}"
    );
    let mut launch = serde_json::to_value(eu_pod_launch()).unwrap_or(Value::Null);
    if let Some(g) = launch["guards"].as_object_mut() {
        g.remove("cap_s");
    }
    assert!(serde_json::from_value::<Launch>(launch).is_err());
    let nulls: crate::provider::Guards = serde_json::from_value(json!({
        "max_usd_per_hr": null, "min_balance_usd": 8, "cap_s": null, "ready_timeout_s": null, "ready_cycles": null
    }))
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        nulls.not_checked(),
        ["max_usd_per_hr", "cap_s", "ready_timeout_s", "ready_cycles"]
    );
}

/// Records what the D1 ledger would be told.
#[derive(Default)]
struct RecRecorder(RefCell<Vec<String>>);

impl Recorder for RecRecorder {
    fn created(&self, _l: &Launch, i: &Instance) -> Result<(), String> {
        self.0.borrow_mut().push(format!("created {}", i.id));
        Ok(())
    }
    fn ready(&self, i: &Instance) -> Result<(), String> {
        self.0.borrow_mut().push(format!("ready {}", i.id));
        Err("D1 down".into())
    }
    fn deleted(&self, _k: ProviderKind, id: &InstanceId, status: &str) -> Result<(), String> {
        self.0.borrow_mut().push(format!("{status} {id}"));
        Ok(())
    }
}

/// C3d (§10.3): `up` tells the deployments ledger of every create, ready and
/// delete; a ledger failure is a warning, never a failed deploy.
#[test]
fn the_recorder_sees_create_ready_and_delete() {
    let clock = FakeClock::at(100.0);
    let p = FakeProvider::new(&clock);
    let (log, bs, rec) = (
        Lines::default(),
        RecBackstop::default(),
        RecRecorder::default(),
    );
    let d = Deployer {
        provider: &p,
        clock: &clock,
        log: &log,
        backstop: &bs,
        ledger: None,
        recorder: Some(&rec),
        interrupted: &never,
        leaks: &no_leak,
        poll: Duration::from_secs(5),
    };
    let r = d
        .up(&eu_pod_launch(), &smoke_up())
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(r.ok && r.deleted);
    let id = r.instance.map(|i| i.id.to_string()).unwrap_or_default();
    assert_eq!(
        *rec.0.borrow(),
        [
            format!("created {id}"),
            format!("ready {id}"),
            format!("deleted {id}")
        ]
    );
    assert!(log
        .0
        .borrow()
        .iter()
        .any(|l| l.contains("could not record") && l.contains("D1 down")));
}

/// Records whether the backstop was armed for an instance when the ledger
/// was told of its creation.
struct OrderRecorder<'a> {
    bs: &'a RecBackstop,
    events: RefCell<Vec<String>>,
}

impl Recorder for OrderRecorder<'_> {
    fn created(&self, _l: &Launch, i: &Instance) -> Result<(), String> {
        let armed = self.bs.0.borrow().iter().any(|(_, id, _)| *id == i.id);
        self.events
            .borrow_mut()
            .push(format!("created {} armed={armed}", i.id));
        Ok(())
    }
    fn ready(&self, i: &Instance) -> Result<(), String> {
        self.events.borrow_mut().push(format!("ready {}", i.id));
        Ok(())
    }
    fn deleted(&self, _k: ProviderKind, id: &InstanceId, status: &str) -> Result<(), String> {
        self.events.borrow_mut().push(format!("{status} {id}"));
        Ok(())
    }
}

/// Review fix (#199): the D1 write (minutes at worst) comes after the
/// backstop is armed and the billed price is checked, as `runpod-pod.sh`
/// orders them; an instance deleted at once for its billed price gets no row.
#[test]
fn the_ledger_hears_of_a_create_only_after_its_backstop_and_price_check() {
    for over_cap in [false, true] {
        let clock = FakeClock::at(100.0);
        let p = FakeProvider::new(&clock);
        if over_cap {
            p.billed.set(Some(Some(9.0)));
        }
        let (log, bs) = (Lines::default(), RecBackstop::default());
        let rec = OrderRecorder {
            bs: &bs,
            events: RefCell::new(Vec::new()),
        };
        let d = Deployer {
            provider: &p,
            clock: &clock,
            log: &log,
            backstop: &bs,
            ledger: None,
            recorder: Some(&rec),
            interrupted: &never,
            leaks: &no_leak,
            poll: Duration::from_secs(5),
        };
        let r = d.up(&eu_pod_launch(), &smoke_up());
        let ev = rec.events.borrow().clone();
        if over_cap {
            assert!(r.is_err());
            assert!(!ev.iter().any(|e| e.starts_with("created")), "{ev:?}");
            assert!(ev.iter().any(|e| e.starts_with("deleted")), "{ev:?}");
        } else {
            let id = r
                .ok()
                .and_then(|r| r.instance)
                .map(|i| i.id.to_string())
                .unwrap_or_default();
            assert_eq!(ev[0], format!("created {id} armed=true"), "{ev:?}");
        }
    }
}

/// Review fix (#199): a delete by the detached backstop at the deadline is
/// recorded in the deployments ledger too (warn-only); an instance already
/// gone is not recorded again.
#[test]
fn the_backstop_records_its_delete() {
    let clock = FakeClock::at(0.0);
    let p = FakeProvider::new(&clock);
    let i = p.create(&eu_pod_launch()).expect("create");
    let (log, rec) = (Lines::default(), RecRecorder::default());
    let out = crate::guard::run_backstop_recorded(&p, &i.id, 1800.0, &clock, &log, Some(&rec))
        .expect("backstop");
    assert_eq!(out, BackstopOutcome::Deleted);
    assert_eq!(*rec.0.borrow(), [format!("deleted {}", i.id)]);
}
