//! `fv-deploy up | smoke | down` over any [`Provider`]
//! (docs/adr/0018-fv-deploy.md §5, §6): the guards around the create, the
//! detached backstop, the readiness poll with the replace rule, the smoke
//! (`scripts/gpu/fv-boot-smoke-pod.sh`'s sequence: `/ping`, `/healthz`, one
//! native job) and the delete, with a local ledger. Every step runs on the
//! seams (provider, clock, log, backstop), so the tests drive the whole
//! lifecycle against a fake provider in milliseconds.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

use crate::boot::sys::{Clock, LogSink, Method};
use crate::guard::{self, Backstop, GuardError};
use crate::launch::utc_iso;
use crate::provider::{
    Account, Instance, InstanceId, InstanceState, Launch, OfferQuery, Probe, Provider,
    ProviderError, ProviderKind, Reach, ReachOptions,
};

/// Why `up` (or `smoke`, `down`) failed.
#[derive(Debug, thiserror::Error)]
pub enum DeployError {
    #[error(transparent)]
    Guard(#[from] GuardError),
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("no instance became ready in {cycles} cycle(s) of {timeout_s} s: {last}")]
    NotReady {
        cycles: u32,
        timeout_s: u64,
        last: String,
    },
    #[error("no GPU type could be created: {0}")]
    NoCapacity(String),
    #[error("interrupted")]
    Interrupted,
    #[error("smoke failed: {0}")]
    Smoke(String),
    #[error("no backstop for {id} ({why}); {}", if *deleted { "it was deleted" } else { "its delete FAILED too: delete it by hand (`fv-deploy down`)" })]
    NoBackstop {
        id: InstanceId,
        deleted: bool,
        why: String,
    },
    #[error("could not delete {id} ({why}); stopped instead of creating another (its backstop deletes it at the cap; `fv-deploy down {id}`)")]
    DeleteFailed { id: InstanceId, why: String },
    #[error("{0} not found (nothing deleted): check the id, --kind and, on GCP, --zone")]
    NotFound(InstanceId),
    #[error("the create's answer was lost ({why}); stopped; swept by name: {swept}")]
    CreateUncertain { why: String, swept: String },
}

/// The append-only local ledger (`artifacts/fv-deploy/ledger.tsv`, the
/// scripts' `ledger.tsv` format: `<UTC time>\t<event> <id> k=v …`). The D1
/// `deployments` ledger is a [`Recorder`] (C3d, docs/adr/0018-fv-deploy.md §10.3).
#[derive(Clone, Debug)]
pub struct Ledger {
    pub path: PathBuf,
}

impl Ledger {
    pub fn record(&self, now: f64, line: &str) {
        if let Some(d) = self.path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = writeln!(f, "{}\t{line}", utc_iso(now));
        }
    }
}

/// Where `up`, `down` and the sweeps record what they create, find ready and
/// delete, besides the local [`Ledger`]: the D1 `deployments` table that
/// fv-control and `release.sh` read (`crate::release::registry::D1Recorder`,
/// docs/adr/0018-fv-deploy.md §10.3). Best effort, as the scripts'
/// `registry.sh`: a failure is logged and never fails the deploy.
pub trait Recorder {
    fn created(&self, launch: &Launch, instance: &Instance) -> Result<(), String>;
    fn ready(&self, instance: &Instance) -> Result<(), String>;
    /// `status`: `deleted` (or `gone` for a reconcile).
    fn deleted(&self, kind: ProviderKind, id: &InstanceId, status: &str) -> Result<(), String>;
}

/// What the smoke does.
#[derive(Clone, Debug, PartialEq)]
pub struct SmokeOptions {
    /// The job's model (default: the first model of `/fv/v1/capabilities`).
    pub model: Option<String>,
    /// JSON merged into the job body (`FV_SMOKE_PARAMS`).
    pub params: Option<Value>,
    /// How long to wait for the job (`FV_SMOKE_JOB_WAIT_S`, 600 s).
    pub job_wait_s: u64,
}

impl Default for SmokeOptions {
    fn default() -> Self {
        Self {
            model: None,
            params: None,
            job_wait_s: 600,
        }
    }
}

/// The smoke's job body (fv-boot-smoke-pod.sh).
pub fn smoke_job_body(model: &str, params: Option<&Value>) -> Value {
    let mut b = json!({
        "model": model,
        "prompt": "a red fox trotting through fresh snow",
        "seed": 1,
        "short_edge": 480,
        "aspect_ratio": "16:9",
    });
    if let (Some(Value::Object(p)), Value::Object(m)) = (params, &mut b) {
        for (k, v) in p {
            m.insert(k.clone(), v.clone());
        }
    }
    b
}

/// The first model of `/fv/v1/capabilities` (`models[0].id`, else
/// `models[0].caps.id`).
pub fn first_model(caps: &Value) -> Option<String> {
    let m = caps.get("models")?.as_array()?.first()?;
    m.get("id")
        .or_else(|| m.pointer("/caps/id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// The smoke's job outcome.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SmokeJob {
    pub id: Option<String>,
    pub model: Option<String>,
    pub status: String,
    pub error: Option<Value>,
    pub wall_s: Option<f64>,
}

/// What the smoke saw (the summary of fv-boot-smoke-pod.sh).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SmokeReport {
    pub ok: bool,
    pub ping: String,
    pub healthz_status: Option<u16>,
    /// `/healthz`'s `state`, `loaded`, `loading`.
    pub healthz: Value,
    pub job: SmokeJob,
    pub why: Option<String>,
}

/// How long the smoke waits for `/ping` 200 on an instance `up` found ready.
pub const PING_WAIT: Duration = Duration::from_secs(120);

const JOB_TERMINAL: [&str; 3] = ["succeeded", "failed", "cancelled"];

/// The smoke against a ready instance: `/ping` 200, `/healthz`, the first
/// model of `/fv/v1/capabilities` (or `opts.model`), one job submitted and
/// polled to its end. `ok` when the job succeeded. `interrupted` is polled in
/// every wait (the report then says why).
pub fn smoke(
    reach: &mut dyn Reach,
    opts: &SmokeOptions,
    clock: &dyn Clock,
    log: &dyn LogSink,
    interrupted: &dyn Fn() -> bool,
) -> SmokeReport {
    let mut r = SmokeReport {
        healthz: Value::Null,
        ..SmokeReport::default()
    };
    let stopped = |r: &mut SmokeReport| {
        r.why = Some("interrupted".into());
    };
    // A queue endpoint answers its first probe with a job in flight: give
    // the instance up to `PING_WAIT` to say ready (a ready pod: at once).
    let t = clock.now();
    let mut p = reach.ping();
    while p != Probe::Ready && clock.now() - t < PING_WAIT.as_secs_f64() {
        if interrupted() {
            stopped(&mut r);
            return r;
        }
        clock.sleep(Duration::from_secs(2));
        p = reach.ping();
    }
    r.ping = match &p {
        Probe::Ready => "200".into(),
        Probe::Loading => "204".into(),
        Probe::Down(w) => w.clone(),
    };
    if p != Probe::Ready {
        r.why = Some(format!("/ping is not 200 ({})", r.ping));
        return r;
    }
    match reach.call(Method::Get, "/healthz", None, interrupted) {
        Ok((s, v)) => {
            r.healthz_status = Some(s);
            r.healthz = json!({
                "state": v.get("state"), "loaded": v.get("loaded"), "loading": v.get("loading")
            });
            log.line(&format!("[fv-deploy] /healthz {s}"));
        }
        Err(e) => {
            r.why = Some(format!("/healthz: {e}"));
            return r;
        }
    }
    let model = match &opts.model {
        Some(m) => m.clone(),
        None => match reach.call(Method::Get, "/fv/v1/capabilities", None, interrupted) {
            Ok((200, caps)) => match first_model(&caps) {
                Some(m) => m,
                None => {
                    r.why = Some("capabilities name no model".into());
                    return r;
                }
            },
            Ok((s, v)) => {
                r.why = Some(format!(
                    "capabilities: HTTP {s}: {}",
                    crate::provider::scrub(&v.to_string(), "", 200)
                ));
                return r;
            }
            Err(e) => {
                r.why = Some(format!("capabilities: {e}"));
                return r;
            }
        },
    };
    r.job.model = Some(model.clone());
    let body = smoke_job_body(&model, opts.params.as_ref());
    log.line(&format!("[fv-deploy] job: {body}"));
    let t0 = clock.now();
    let id = match reach.call(Method::Post, "/fv/v1/jobs", Some(&body), interrupted) {
        Ok((s, v)) if (200..300).contains(&s) => {
            v.get("id").and_then(Value::as_str).map(str::to_owned)
        }
        Ok((s, v)) => {
            r.job.status = "submit_failed".into();
            r.why = Some(format!(
                "job submit: HTTP {s}: {}",
                crate::provider::scrub(&v.to_string(), "", 300)
            ));
            return r;
        }
        Err(e) => {
            r.job.status = "submit_failed".into();
            r.why = Some(format!("job submit: {e}"));
            return r;
        }
    };
    let Some(id) = id else {
        r.job.status = "submit_failed".into();
        r.why = Some("job submit: no id".into());
        return r;
    };
    r.job.id = Some(id.clone());
    let deadline = t0 + opts.job_wait_s as f64;
    loop {
        if interrupted() {
            stopped(&mut r);
            r.job.wall_s = Some(clock.now() - t0);
            return r;
        }
        match reach.call(Method::Get, &format!("/fv/v1/jobs/{id}"), None, interrupted) {
            Ok((_, v)) => {
                let st = v
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                if JOB_TERMINAL.contains(&st.as_str()) {
                    r.job.error = v.get("error").filter(|e| !e.is_null()).cloned();
                    r.job.status = st;
                    break;
                }
                r.job.status = st;
            }
            Err(e) => r.why = Some(format!("job poll: {e}")),
        }
        if clock.now() >= deadline {
            r.why = Some(format!(
                "job {id} not finished after {} s (last {})",
                opts.job_wait_s, r.job.status
            ));
            r.job.wall_s = Some(clock.now() - t0);
            return r;
        }
        clock.sleep(Duration::from_secs(2));
    }
    r.job.wall_s = Some(clock.now() - t0);
    log.line(&format!("[fv-deploy] job {id}: {}", r.job.status));
    r.ok = r.job.status == "succeeded";
    if !r.ok {
        r.why = Some(format!("job {id} ended {}", r.job.status));
    } else {
        r.why = None;
    }
    r
}

/// What `up` should do after the instance is ready.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UpOptions {
    /// The run's API key for fv-serve (its hash is the launch's `FV_API_KEYS`).
    pub run_key: Option<String>,
    /// Run the smoke (`None`: skip it).
    pub smoke: Option<SmokeOptions>,
    /// Leave the instance running (the backstop still deletes it at the cap).
    pub keep: bool,
}

/// One create of the replace rule.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cycle {
    pub n: u32,
    pub instance: Option<InstanceId>,
    pub gpu: Option<String>,
    pub usd_per_hr: Option<f64>,
    pub ready_after_s: Option<f64>,
    pub outcome: String,
}

/// What `up` did.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct UpReport {
    pub name: String,
    pub provider: ProviderKind,
    pub instance: Option<Instance>,
    pub cycles: Vec<Cycle>,
    pub smoke: Option<SmokeReport>,
    pub deleted: bool,
    pub ok: bool,
}

enum Ready {
    Yes(f64),
    No(String),
}

/// Delete attempts before fv-deploy gives up on an instance (fv-control's
/// watchdog: 3, 5 s apart).
pub const DELETE_ATTEMPTS: u32 = 3;
/// The pause between them.
pub const DELETE_RETRY: Duration = Duration::from_secs(5);
/// Sweeps after a lost create answer (until one finds the instance) …
pub const LOST_SWEEPS: u32 = 6;
/// … this far apart (1.5 min in all).
pub const LOST_SWEEP_GAP: Duration = Duration::from_secs(18);

/// The lifecycle on one provider.
pub struct Deployer<'a> {
    pub provider: &'a dyn Provider,
    pub clock: &'a dyn Clock,
    pub log: &'a dyn LogSink,
    pub backstop: &'a dyn Backstop,
    pub ledger: Option<&'a Ledger>,
    /// The D1 `deployments` ledger (C3d), when configured.
    pub recorder: Option<&'a dyn Recorder>,
    /// Polled in every wait; true stops `up` (the instance is deleted).
    pub interrupted: &'a dyn Fn() -> bool,
    /// Whether a text contains the provider's API key (the env guard).
    pub leaks: &'a dyn Fn(&str) -> bool,
    /// The readiness poll's interval (the scripts': 5 s).
    pub poll: Duration,
}

impl Deployer<'_> {
    fn say(&self, s: &str) {
        self.log.line(&format!("[fv-deploy] {s}"));
    }

    fn record(&self, s: &str) {
        if let Some(l) = self.ledger {
            l.record(self.clock.now(), s);
        }
    }

    fn recorded(&self, what: &str, r: Result<(), String>) {
        if let Err(e) = r {
            self.say(&format!(
                "WARNING: the deployments ledger (D1) could not record {what}: {e}"
            ));
        }
    }

    fn record_deleted(&self, id: &InstanceId) {
        if let Some(r) = self.recorder {
            self.recorded(
                &format!("the delete of {id}"),
                r.deleted(self.provider.kind(), id, "deleted"),
            );
        }
    }

    /// Every guard that needs no network.
    pub fn preflight(&self, launch: &Launch) -> Result<(), DeployError> {
        guard::check_guards(&launch.guards)?;
        guard::check_region(launch)?;
        guard::check_serverless(launch)?;
        guard::check_env(launch, self.leaks)?;
        if launch.provider != self.provider.kind() {
            return Err(GuardError::Invalid(format!(
                "the launch is for {}, the provider is {}",
                launch.provider,
                self.provider.kind()
            ))
            .into());
        }
        Ok(())
    }

    /// The balance floor, read now; the account for the projection.
    pub fn check_balance(&self, floor: f64) -> Result<Account, DeployError> {
        let a = self.provider.account()?;
        guard::check_balance(&a, floor)?;
        self.say(&format!(
            "balance ${:.2} (floor ${floor:.2})",
            a.balance_usd
        ));
        Ok(a)
    }

    /// Deletes with retries (3 attempts, 5 s apart); the ledger says
    /// "deleted" only when it was.
    pub fn destroy_retry(&self, id: &InstanceId, why: &str) -> Result<(), ProviderError> {
        let mut last = None;
        for i in 1..=DELETE_ATTEMPTS {
            match self.provider.destroy(id) {
                Ok(()) => {
                    self.say(&format!("deleted {id} ({why})"));
                    self.record(&format!("{}-deleted {id} {why}", self.provider.kind()));
                    self.record_deleted(id);
                    return Ok(());
                }
                Err(e) => {
                    self.say(&format!("delete of {id} failed (attempt {i}): {e}"));
                    last = Some(e);
                }
            }
            if i < DELETE_ATTEMPTS {
                self.clock.sleep(DELETE_RETRY);
            }
        }
        self.record(&format!(
            "{}-delete-failed {id} {why}",
            self.provider.kind()
        ));
        Err(last.unwrap_or_else(|| ProviderError::Unsupported("no delete attempt".into())))
    }

    /// Arms the backstop of a just-created instance; when that fails the
    /// instance is deleted (with retries) and `up` stops.
    fn arm_or_delete(&self, inst: &Instance, cap_s: u64) -> Result<(), DeployError> {
        let deadline = inst.created_at.unwrap_or_else(|| self.clock.now()) + cap_s as f64;
        match self.backstop.arm(inst.provider, &inst.id, deadline) {
            Ok(pid) => {
                self.say(&format!(
                    "backstop armed for {} (pid {pid}): deleted at the latest {cap_s} s after create",
                    inst.id
                ));
                Ok(())
            }
            Err(e) => {
                self.say(&format!(
                    "no backstop for {}: {e}; deleting it now",
                    inst.id
                ));
                let deleted = self.destroy_retry(&inst.id, "no-backstop").is_ok();
                Err(DeployError::NoBackstop {
                    id: inst.id.clone(),
                    deleted,
                    why: e.to_string(),
                })
            }
        }
    }

    /// One create behind the price cap and the projected balance floor: a
    /// pod tries each GPU type in order (quote first, never create over the
    /// cap or at an unknown price); every create is backstopped at once, then
    /// a billed price over the cap (or unreadable) deletes it. A serverless
    /// endpoint gets every type whose quote passes. `budget` counts the
    /// instances `up` may still create. A create whose answer was lost stops
    /// everything and sweeps by name.
    pub fn create_guarded(
        &self,
        launch: &Launch,
        account: &Account,
        budget: &mut u32,
    ) -> Result<Instance, DeployError> {
        let g = guard::check_guards(&launch.guards)?;
        let cap = g.max_usd_per_hr;
        let offers = self.provider.offers(&OfferQuery {
            gpu_types: launch.gpu.types.clone(),
            data_center: launch.placement.data_centers.first().cloned(),
            cloud: launch.placement.cloud,
        })?;
        let workers = launch
            .serverless
            .as_ref()
            .map_or(1, |s| s.workers_max.max(1));
        let mut notes = Vec::new();
        let mut ok_types = Vec::new();
        let mut floor_err = None;
        for o in &offers {
            let p = match guard::check_quote(o, cap) {
                Ok(p) => p,
                Err(e) => {
                    self.say(&e.to_string());
                    notes.push(e.to_string());
                    continue;
                }
            };
            match guard::check_projection(
                account,
                p * f64::from(workers),
                g.cap_s,
                g.min_balance_usd,
            ) {
                Ok(()) => ok_types.push(o.gpu_type.clone()),
                Err(e) => {
                    self.say(&e.to_string());
                    notes.push(e.to_string());
                    floor_err.get_or_insert(e);
                }
            }
        }
        if ok_types.is_empty() {
            return Err(match floor_err {
                Some(e) => e.into(),
                None => DeployError::NoCapacity(notes.join("; ")),
            });
        }
        let attempts: Vec<Launch> = if launch.gpu.types.is_empty() {
            // The launch names no GPU type (Vast: any GPU): the provider
            // quoted "any" and creates it as it is, unnarrowed.
            vec![launch.clone()]
        } else if launch.provider == ProviderKind::RunpodServerless {
            let mut l = launch.clone();
            l.gpu.types = ok_types;
            vec![l]
        } else {
            ok_types.iter().map(|g| launch.for_gpu(g)).collect()
        };
        for l in attempts {
            if *budget == 0 {
                notes.push("instance budget (ready_cycles) used up".into());
                break;
            }
            let label = l.gpu.types.join(", ");
            match self.provider.create(&l) {
                Ok(inst) => {
                    *budget -= 1;
                    self.record(&format!(
                        "{}-created {} {} gpu={label} image={} {}",
                        inst.provider,
                        inst.id,
                        inst.name,
                        l.image,
                        l.guards.ledger_line()
                    ));
                    // The backstop first: nothing below may leave it unguarded
                    // (the D1 ledger's write can take minutes: after it).
                    self.arm_or_delete(&inst, g.cap_s)?;
                    if self.provider.bills_at_create() {
                        if let Err(e) = guard::check_billed(&inst, cap) {
                            self.say(&format!("{e}: deleting"));
                            if let Err(d) = self.destroy_retry(&inst.id, "over-cap") {
                                return Err(DeployError::DeleteFailed {
                                    id: inst.id.clone(),
                                    why: d.to_string(),
                                });
                            }
                            notes.push(e.to_string());
                            continue;
                        }
                    }
                    // The D1 row once the instance is backstopped and within
                    // the cap (as runpod-pod.sh: create, backstop, row). An
                    // instance deleted at once (no backstop, billed over the
                    // cap) gets no row; its delete's update matches none.
                    if let Some(r) = self.recorder {
                        self.recorded(&inst.id.to_string(), r.created(&l, &inst));
                    }
                    self.say(&format!(
                        "{} {} on {label}{}",
                        inst.provider,
                        inst.id,
                        inst.usd_per_hr
                            .map(|p| format!(" at ${p:.3}/hr"))
                            .unwrap_or_default()
                    ));
                    return Ok(inst);
                }
                Err(e) if e.is_definite_refusal() => {
                    self.say(&format!("no instance on {label}: {e}"));
                    notes.push(format!("{label}: {e}"));
                }
                Err(e) => {
                    // The request may have created something: never create
                    // another; find it by its (unique) name and delete it.
                    self.say(&format!(
                        "the create's answer was lost on {label}: {e}; sweeping by name"
                    ));
                    let swept = self.sweep_lost(&launch.name);
                    return Err(DeployError::CreateUncertain {
                        why: e.to_string(),
                        swept,
                    });
                }
            }
        }
        Err(DeployError::NoCapacity(notes.join("; ")))
    }

    /// After a lost create answer: sweeps by name until it finds something,
    /// [`LOST_SWEEPS`] times [`LOST_SWEEP_GAP`] apart (an instance may only
    /// show in the provider's listing a little later; on Vast this sweep is
    /// the only protection: no id, so no backstop, and no cloud-side cap).
    fn sweep_lost(&self, name: &str) -> String {
        let mut last = String::new();
        for i in 1..=LOST_SWEEPS {
            last = self.sweep(name, None);
            // Something swept: done. Nothing, or a failed sweep: again.
            if last != "nothing left" && !last.starts_with("FAILED") {
                return last;
            }
            if i < LOST_SWEEPS {
                self.clock.sleep(LOST_SWEEP_GAP);
            }
        }
        format!(
            "{last} after {LOST_SWEEPS} sweeps over {} s",
            (LOST_SWEEPS - 1) as u64 * LOST_SWEEP_GAP.as_secs()
        )
    }

    /// Deletes everything named `name` but `keep`; a summary for the error
    /// or the log.
    fn sweep(&self, name: &str, keep: Option<&InstanceId>) -> String {
        match self.provider.sweep(name, keep) {
            Ok(ids) if ids.is_empty() => "nothing left".into(),
            Ok(ids) => {
                for id in &ids {
                    self.say(&format!("swept {id} (named {name})"));
                    self.record(&format!("{}-deleted {id} sweep", self.provider.kind()));
                    self.record_deleted(id);
                }
                ids.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            }
            Err(e) => {
                self.say(&format!(
                    "WARNING: the sweep of {name} failed: {e}; check `fv-deploy` / the console"
                ));
                format!("FAILED: {e}")
            }
        }
    }

    fn wait_ready(
        &self,
        inst: &Instance,
        key: Option<&str>,
        timeout_s: u64,
    ) -> Result<Ready, DeployError> {
        let t0 = inst.created_at.unwrap_or_else(|| self.clock.now());
        let deadline = t0 + timeout_s as f64;
        let mut reach = self.provider.reach(
            inst,
            &ReachOptions {
                key,
                ..ReachOptions::default()
            },
        )?;
        let mut last: String;
        let mut first_answer = false;
        let mut next_status = self.clock.now() + 60.0;
        loop {
            if (self.interrupted)() {
                return Err(DeployError::Interrupted);
            }
            match reach.ping() {
                Probe::Ready => return Ok(Ready::Yes(self.clock.now() - t0)),
                Probe::Loading => {
                    if !first_answer {
                        self.say(&format!(
                            "{} answers /ping (loading) {:.0} s after create",
                            inst.id,
                            self.clock.now() - t0
                        ));
                        first_answer = true;
                    }
                    last = "/ping 204 (loading)".into();
                }
                Probe::Down(w) => last = w,
            }
            if self.clock.now() >= next_status {
                next_status = self.clock.now() + 60.0;
                if let Ok(s @ (InstanceState::Terminated | InstanceState::Exited)) =
                    self.provider.status(&inst.id)
                {
                    return Ok(Ready::No(format!("{} is {s:?}", inst.id)));
                }
            }
            if self.clock.now() >= deadline {
                return Ok(Ready::No(format!(
                    "not ready {timeout_s} s after creation (last: {last})"
                )));
            }
            self.clock.sleep(self.poll);
        }
    }

    /// `fv-deploy up`: guards, then at most `ready_cycles` creates, each
    /// backstopped at once and replaced when not ready within
    /// `ready_timeout_s`; the smoke; the delete unless `keep`. Every exit
    /// sweeps the launch's name (all but a kept instance).
    pub fn up(&self, launch: &Launch, opts: &UpOptions) -> Result<UpReport, DeployError> {
        self.preflight(launch)?;
        let r = self.up_inner(launch, opts);
        let keep = match &r {
            Ok(rep) if !rep.deleted => rep.instance.as_ref().map(|i| i.id.clone()),
            _ => None,
        };
        let swept = self.sweep(&launch.name, keep.as_ref());
        if swept != "nothing left" {
            self.say(&format!("exit sweep of {}: {swept}", launch.name));
        }
        r
    }

    fn up_inner(&self, launch: &Launch, opts: &UpOptions) -> Result<UpReport, DeployError> {
        let g = guard::check_guards(&launch.guards)?;
        let mut cycles = Vec::new();
        let mut ready: Option<Instance> = None;
        let mut budget = g.ready_cycles;
        let mut n = 0;
        while budget > 0 {
            n += 1;
            if (self.interrupted)() {
                return Err(DeployError::Interrupted);
            }
            // The floor before every create (a replacement included).
            let account = self.check_balance(g.min_balance_usd)?;
            let inst = self.create_guarded(launch, &account, &mut budget)?;
            let mut cycle = Cycle {
                n,
                instance: Some(inst.id.clone()),
                gpu: inst.gpu.clone(),
                usd_per_hr: inst.usd_per_hr,
                ready_after_s: None,
                outcome: String::new(),
            };
            match self.wait_ready(&inst, opts.run_key.as_deref(), g.ready_timeout_s) {
                Ok(Ready::Yes(t)) => {
                    self.say(&format!("{} ready {t:.1} s after create", inst.id));
                    self.record(&format!(
                        "{}-ready {} after_s={t:.1}",
                        inst.provider, inst.id
                    ));
                    if let Some(r) = self.recorder {
                        self.recorded(&format!("{} ready", inst.id), r.ready(&inst));
                    }
                    cycle.ready_after_s = Some(t);
                    cycle.outcome = "ready".into();
                    cycles.push(cycle);
                    let mut i = inst;
                    i.state = InstanceState::Running;
                    ready = Some(i);
                    break;
                }
                Ok(Ready::No(why)) => {
                    self.say(&format!(
                        "{} {why}: replacing ({budget} instance(s) left of {})",
                        inst.id, g.ready_cycles
                    ));
                    if let Err(e) = self.destroy_retry(&inst.id, "not-ready") {
                        return Err(DeployError::DeleteFailed {
                            id: inst.id,
                            why: e.to_string(),
                        });
                    }
                    cycle.outcome = format!("replaced: {why}");
                    cycles.push(cycle);
                }
                Err(e) => {
                    let _ = self.destroy_retry(&inst.id, "interrupted");
                    return Err(e);
                }
            }
        }
        let Some(inst) = ready else {
            return Err(DeployError::NotReady {
                cycles: n,
                timeout_s: g.ready_timeout_s,
                last: cycles.last().map(|c| c.outcome.clone()).unwrap_or_default(),
            });
        };
        let smoke = match &opts.smoke {
            Some(so) => {
                let ro = ReachOptions {
                    key: opts.run_key.as_deref(),
                    call_timeout: Duration::from_secs(so.job_wait_s),
                };
                match self.provider.reach(&inst, &ro) {
                    Ok(mut reach) => Some(smoke(
                        reach.as_mut(),
                        so,
                        self.clock,
                        self.log,
                        self.interrupted,
                    )),
                    Err(e) => Some(SmokeReport {
                        why: Some(e.to_string()),
                        ..SmokeReport::default()
                    }),
                }
            }
            None => None,
        };
        if (self.interrupted)() {
            let _ = self.destroy_retry(&inst.id, "interrupted");
            return Err(DeployError::Interrupted);
        }
        let smoke_ok = smoke.as_ref().is_none_or(|s| s.ok);
        let deleted = if opts.keep {
            self.say(&format!(
                "keeping {} (the backstop deletes it {} s after create; `fv-deploy down {}` now)",
                inst.id, g.cap_s, inst.id
            ));
            false
        } else {
            let why = if smoke_ok { "done" } else { "smoke-failed" };
            if let Err(e) = self.destroy_retry(&inst.id, why) {
                return Err(DeployError::DeleteFailed {
                    id: inst.id,
                    why: e.to_string(),
                });
            }
            true
        };
        Ok(UpReport {
            name: launch.name.clone(),
            provider: launch.provider,
            instance: Some(inst),
            cycles,
            smoke,
            deleted,
            ok: smoke_ok,
        })
    }

    /// `fv-deploy down`: an instance the provider does not know (a wrong
    /// id, kind or zone) is "not found", never "deleted".
    pub fn down(&self, id: &InstanceId) -> Result<(), DeployError> {
        if self.provider.status(id)? == InstanceState::Terminated {
            return Err(DeployError::NotFound(id.clone()));
        }
        self.provider.destroy(id)?;
        self.record(&format!("{}-deleted {id} down", self.provider.kind()));
        self.record_deleted(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
