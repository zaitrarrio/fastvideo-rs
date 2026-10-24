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
    Instance, InstanceId, InstanceState, Launch, OfferQuery, Probe, Provider, ProviderError,
    ProviderKind, Reach,
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
}

/// The append-only local ledger (`artifacts/fv-deploy/ledger.tsv`, the
/// scripts' `ledger.tsv` format: `<UTC time>\t<event> <id> k=v …`). The D1
/// `deployments` ledger is plan C3d.
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
/// polled to its end. `ok` when the job succeeded.
pub fn smoke(
    reach: &mut dyn Reach,
    opts: &SmokeOptions,
    clock: &dyn Clock,
    log: &dyn LogSink,
) -> SmokeReport {
    let mut r = SmokeReport {
        healthz: Value::Null,
        ..SmokeReport::default()
    };
    // A queue endpoint answers its first probe with a job in flight: give
    // the instance up to `PING_WAIT` to say ready (a ready pod: at once).
    let t = clock.now();
    let mut p = reach.ping();
    while p != Probe::Ready && clock.now() - t < PING_WAIT.as_secs_f64() {
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
    match reach.call(Method::Get, "/healthz", None) {
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
        None => match reach.call(Method::Get, "/fv/v1/capabilities", None) {
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
    let id = match reach.call(Method::Post, "/fv/v1/jobs", Some(&body)) {
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
        match reach.call(Method::Get, &format!("/fv/v1/jobs/{id}"), None) {
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

/// The lifecycle on one provider.
pub struct Deployer<'a> {
    pub provider: &'a dyn Provider,
    pub clock: &'a dyn Clock,
    pub log: &'a dyn LogSink,
    pub backstop: &'a dyn Backstop,
    pub ledger: Option<&'a Ledger>,
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

    /// Every guard that needs no network.
    pub fn preflight(&self, launch: &Launch) -> Result<(), DeployError> {
        guard::check_guards(&launch.guards)?;
        guard::check_region(launch)?;
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

    /// The balance floor, read now.
    pub fn check_balance(&self, floor: f64) -> Result<(), DeployError> {
        let a = self.provider.account()?;
        guard::check_balance(&a, floor)?;
        self.say(&format!(
            "balance ${:.2} (floor ${floor:.2})",
            a.balance_usd
        ));
        Ok(())
    }

    /// One create behind the price cap: a pod tries each GPU type in order
    /// (quote first, never create over the cap or at an unknown price; a
    /// billed price over the cap deletes it at once); a serverless endpoint
    /// gets every type whose quote passes.
    pub fn create_guarded(&self, launch: &Launch) -> Result<Instance, DeployError> {
        let cap = launch.guards.max_usd_per_hr;
        let offers = self.provider.offers(&OfferQuery {
            gpu_types: launch.gpu.types.clone(),
            data_center: launch.placement.data_centers.first().cloned(),
            cloud: launch.placement.cloud,
        })?;
        let mut notes = Vec::new();
        let mut ok_types = Vec::new();
        for o in &offers {
            match guard::check_quote(o, cap) {
                Ok(p) => ok_types.push((o.gpu_type.clone(), p)),
                Err(e) => {
                    self.say(&e.to_string());
                    notes.push(e.to_string());
                }
            }
        }
        if ok_types.is_empty() {
            return Err(DeployError::NoCapacity(notes.join("; ")));
        }
        let attempts: Vec<Launch> = if launch.provider == ProviderKind::RunpodServerless {
            let mut l = launch.clone();
            l.gpu.types = ok_types.iter().map(|(g, _)| g.clone()).collect();
            vec![l]
        } else {
            ok_types.iter().map(|(g, _)| launch.for_gpu(g)).collect()
        };
        for l in attempts {
            let label = l.gpu.types.join(", ");
            match self.provider.create(&l) {
                Ok(inst) => {
                    self.record(&format!(
                        "{}-created {} {} gpu={label} image={}",
                        inst.provider, inst.id, inst.name, l.image
                    ));
                    if let Err(e) = guard::check_billed(&inst, cap) {
                        self.say(&format!("{e}: deleting"));
                        let _ = self.provider.destroy(&inst.id);
                        self.record(&format!("{}-deleted {} over-cap", inst.provider, inst.id));
                        notes.push(e.to_string());
                        continue;
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
                Err(e) => {
                    self.say(&format!("no instance on {label}: {e}"));
                    notes.push(format!("{label}: {e}"));
                }
            }
        }
        Err(DeployError::NoCapacity(notes.join("; ")))
    }

    fn wait_ready(
        &self,
        inst: &Instance,
        key: Option<&str>,
        timeout_s: u64,
    ) -> Result<Ready, DeployError> {
        let t0 = inst.created_at.unwrap_or_else(|| self.clock.now());
        let deadline = t0 + timeout_s as f64;
        let mut reach = self.provider.reach(inst, key)?;
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

    fn destroy_logged(&self, id: &InstanceId, why: &str) -> bool {
        match self.provider.destroy(id) {
            Ok(()) => {
                self.say(&format!("deleted {id} ({why})"));
                self.record(&format!("{}-deleted {id} {why}", self.provider.kind()));
                true
            }
            Err(e) => {
                self.say(&format!(
                    "WARNING: delete of {id} failed ({e}); the backstop retries at the cap"
                ));
                false
            }
        }
    }

    /// `fv-deploy up`: guards, then at most `ready_cycles` creates, each
    /// with its backstop and replaced when not ready within
    /// `ready_timeout_s`; the smoke; the delete unless `keep`.
    pub fn up(&self, launch: &Launch, opts: &UpOptions) -> Result<UpReport, DeployError> {
        self.preflight(launch)?;
        let g = launch.guards;
        let mut cycles = Vec::new();
        let mut ready: Option<Instance> = None;
        for n in 1..=g.ready_cycles {
            if (self.interrupted)() {
                return Err(DeployError::Interrupted);
            }
            // The floor before every create (a replacement included).
            self.check_balance(g.min_balance_usd)?;
            let inst = self.create_guarded(launch)?;
            let deadline = inst.created_at.unwrap_or_else(|| self.clock.now()) + g.cap_s as f64;
            match self.backstop.arm(inst.provider, &inst.id, deadline) {
                Ok(pid) => self.say(&format!(
                    "backstop armed for {} (pid {pid}): deleted at the latest {} s after create",
                    inst.id, g.cap_s
                )),
                Err(e) => {
                    self.destroy_logged(&inst.id, "no backstop");
                    return Err(e.into());
                }
            }
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
                        "{} {why}: replacing (cycle {n} of {})",
                        inst.id, g.ready_cycles
                    ));
                    self.destroy_logged(&inst.id, "not-ready");
                    cycle.outcome = format!("replaced: {why}");
                    cycles.push(cycle);
                }
                Err(e) => {
                    self.destroy_logged(&inst.id, "interrupted");
                    return Err(e);
                }
            }
        }
        let Some(inst) = ready else {
            return Err(DeployError::NotReady {
                cycles: g.ready_cycles,
                timeout_s: g.ready_timeout_s,
                last: cycles.last().map(|c| c.outcome.clone()).unwrap_or_default(),
            });
        };
        let smoke = match &opts.smoke {
            Some(so) => match self.provider.reach(&inst, opts.run_key.as_deref()) {
                Ok(mut reach) => Some(smoke(reach.as_mut(), so, self.clock, self.log)),
                Err(e) => Some(SmokeReport {
                    why: Some(e.to_string()),
                    ..SmokeReport::default()
                }),
            },
            None => None,
        };
        let smoke_ok = smoke.as_ref().is_none_or(|s| s.ok);
        let deleted = if opts.keep {
            self.say(&format!(
                "keeping {} (the backstop deletes it {} s after create; `fv-deploy down {}` now)",
                inst.id, g.cap_s, inst.id
            ));
            false
        } else {
            self.destroy_logged(&inst.id, if smoke_ok { "done" } else { "smoke-failed" })
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

    /// `fv-deploy down`.
    pub fn down(&self, id: &InstanceId) -> Result<(), DeployError> {
        self.provider.destroy(id)?;
        self.record(&format!("{}-deleted {id} down", self.provider.kind()));
        Ok(())
    }
}

#[cfg(test)]
mod tests;
