//! The shared guards of fv-deploy (docs/adr/0018-fv-deploy.md §5), written
//! once for every provider: the region guard, the balance floor, the price
//! cap, the env check and the detached wall-clock backstop. The readiness
//! rule (replace after 600 s, at most 3 cycles) is applied by
//! `crate::lifecycle` with the numbers of [`crate::provider::Guards`].

use std::path::PathBuf;
use std::time::Duration;

use fastvideo_manifest::deploy::{EU_DATA_CENTER, EU_REGION, EU_VOLUME_ID};

use crate::boot::sys::{Clock, LogSink};
use crate::provider::{
    Account, Guards, Instance, InstanceId, InstanceState, Launch, Offer, Provider, ProviderError,
    ProviderKind, WeightsSource,
};

/// CLAUDE.md "GPU pods": stop before the balance would drop below $8.
pub const MIN_BALANCE_USD: f64 = 8.0;
/// The owner's readiness rule (plan §1): not ready 10 minutes after creation
/// → delete and acquire a new one.
pub const READY_TIMEOUT_S: u64 = 600;
/// … at most three cycles, then stop and report.
pub const READY_CYCLES: u32 = 3;
/// The scripts' wall-clock cap (`FV_POD_CAP_S`, `FV_ENDPOINT_CAP_S`).
pub const CAP_S: u64 = 1800;
/// The default price cap for one GPU ($/hr). Runpod's quote is checked
/// before every create; the RTX PRO 6000 Secure quote is below it (the
/// number itself is a guard choice, not a price fact).
pub const MAX_USD_PER_HR: f64 = 2.5;
/// The deleted US volume (`scripts/gpu/volumes.sh` `FV_US_DELETED_VOLUME_ID`).
pub const US_DELETED_VOLUME_ID: &str = "s2k01690bi";
/// … and its name.
pub const US_DELETED_VOLUME_NAME: &str = "fv-weights-b200-us";
/// `FV_US_GONE`.
pub const US_GONE: &str =
    "US weights volume deleted 2026-10; EU only, see docs/ops/runpod-volumes.md";

/// The default guards.
pub fn default_guards() -> Guards {
    Guards {
        max_usd_per_hr: MAX_USD_PER_HR,
        min_balance_usd: MIN_BALANCE_USD,
        cap_s: CAP_S,
        ready_timeout_s: READY_TIMEOUT_S,
        ready_cycles: READY_CYCLES,
    }
}

/// A guard that refused.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GuardError {
    #[error("{0}")]
    UsGone(String),
    #[error("region guard: {0}")]
    Region(String),
    #[error("balance ${balance:.2} is below the floor ${floor:.2}: not creating anything")]
    BalanceBelowFloor { balance: f64, floor: f64 },
    #[error("price check: no quote for {gpu:?}; not creating at an unknown price")]
    NoQuote { gpu: String },
    #[error("price check: {gpu:?} is quoted at ${usd:.3}/hr, over the cap ${cap:.3}/hr")]
    OverCap { gpu: String, usd: f64, cap: f64 },
    #[error("{id} is billed ${usd:.3}/hr, over the cap ${cap:.3}/hr")]
    BilledOverCap { id: String, usd: f64, cap: f64 },
    #[error(
        "the launch's {0} contains the API key: refused (secrets are references, never values)"
    )]
    SecretInLaunch(&'static str),
    #[error("guards: {0}")]
    Invalid(String),
    #[error("backstop: {0}")]
    Backstop(String),
}

/// Runpod GPU instances only with the EU weights volume `bz0ofkfx45` in
/// `EU-RO-1` (CLAUDE.md; plan §1, owner instruction 2026-10-23); the deleted
/// US volume is refused by id and by name; volume weights only (C3a).
pub fn check_region(l: &Launch) -> Result<(), GuardError> {
    if let Some(r) = &l.placement.region {
        if r == "us" {
            return Err(GuardError::UsGone(US_GONE.into()));
        }
        if r != EU_REGION {
            return Err(GuardError::Region(format!("unknown region {r} (eu)")));
        }
    }
    if !l.provider.is_runpod() {
        return Ok(());
    }
    let Some(v) = &l.placement.volume else {
        return Err(GuardError::Region(format!(
            "a Runpod GPU instance mounts the EU weights volume {EU_VOLUME_ID} (owner instruction 2026-10-23); this launch mounts none"
        )));
    };
    if v.id == US_DELETED_VOLUME_ID || v.id == US_DELETED_VOLUME_NAME {
        return Err(GuardError::UsGone(US_GONE.into()));
    }
    if v.id != EU_VOLUME_ID {
        return Err(GuardError::Region(format!(
            "volume {}: the only weights volume is {EU_VOLUME_ID} in {EU_DATA_CENTER} (CLAUDE.md, EU only)",
            v.id
        )));
    }
    if l.placement.data_centers != [EU_DATA_CENTER] {
        return Err(GuardError::Region(format!(
            "data centres {:?}: the volume {EU_VOLUME_ID} is in {EU_DATA_CENTER} only",
            l.placement.data_centers
        )));
    }
    if l.weights != WeightsSource::Volume {
        return Err(GuardError::Region(format!(
            "weights source {}: fv-deploy launches volume weights only (C3a)",
            l.weights.as_str()
        )));
    }
    Ok(())
}

/// The guard numbers themselves.
pub fn check_guards(g: &Guards) -> Result<(), GuardError> {
    let bad = |m: &str| Err(GuardError::Invalid(m.to_owned()));
    if !(g.max_usd_per_hr.is_finite() && g.max_usd_per_hr > 0.0) {
        return bad("the price cap must be a positive number");
    }
    if !(g.min_balance_usd.is_finite() && g.min_balance_usd >= MIN_BALANCE_USD) {
        return bad("the balance floor is at least $8 (CLAUDE.md)");
    }
    if g.cap_s == 0 {
        return bad("the wall-clock cap must be positive");
    }
    if g.ready_timeout_s == 0 || g.ready_cycles == 0 {
        return bad("the readiness timeout and cycles must be positive");
    }
    Ok(())
}

/// No env value, argument or image may contain the provider's API key.
pub fn check_env(l: &Launch, leaks: impl Fn(&str) -> bool) -> Result<(), GuardError> {
    if l.env.iter().any(|(k, v)| leaks(k) || leaks(v)) {
        return Err(GuardError::SecretInLaunch("env"));
    }
    if l.args.iter().chain(&l.entrypoint).any(|a| leaks(a)) {
        return Err(GuardError::SecretInLaunch("command"));
    }
    if leaks(&l.image) || leaks(&l.name) {
        return Err(GuardError::SecretInLaunch("image or name"));
    }
    Ok(())
}

/// The balance floor.
pub fn check_balance(a: &Account, floor: f64) -> Result<(), GuardError> {
    if a.balance_usd < floor {
        return Err(GuardError::BalanceBelowFloor {
            balance: a.balance_usd,
            floor,
        });
    }
    Ok(())
}

/// The price cap on a quote, before a create (`runpod-price.sh`): no quote
/// or a quote above the cap refuses.
pub fn check_quote(o: &Offer, cap: f64) -> Result<f64, GuardError> {
    match o.usd_per_hr {
        None => Err(GuardError::NoQuote {
            gpu: o.gpu_type.clone(),
        }),
        Some(p) if p > cap => Err(GuardError::OverCap {
            gpu: o.gpu_type.clone(),
            usd: p,
            cap,
        }),
        Some(p) => Ok(p),
    }
}

/// The price cap on what the created instance is billed (only when it
/// differs from the quote).
pub fn check_billed(i: &Instance, cap: f64) -> Result<(), GuardError> {
    match i.usd_per_hr {
        Some(p) if p > cap => Err(GuardError::BilledOverCap {
            id: i.id.to_string(),
            usd: p,
            cap,
        }),
        _ => Ok(()),
    }
}

// ------------------------------------------------------------------ backstop

/// Arms the wall-clock backstop of one instance: something that deletes it
/// at `deadline` (unix s) whatever happens to this process.
pub trait Backstop {
    /// Returns the backstop's pid (0 when not a process).
    fn arm(&self, kind: ProviderKind, id: &InstanceId, deadline: f64) -> Result<u32, GuardError>;
}

/// `fv-deploy backstop` arguments.
pub fn backstop_args(kind: ProviderKind, id: &InstanceId, deadline: f64) -> Vec<String> {
    vec![
        "backstop".into(),
        "--kind".into(),
        kind.as_str().into(),
        "--id".into(),
        id.to_string(),
        "--deadline".into(),
        format!("{:.0}", deadline.ceil()),
    ]
}

/// The detached backstop (docs/adr/0018-fv-deploy.md §5.4): this binary
/// again, as `fv-deploy backstop --kind K --id ID --deadline T`, in a process
/// group of its own (a Ctrl-C to the terminal's foreground group does not
/// reach it), stdin closed, output to a log file; it ignores SIGHUP (as
/// `nohup` did) and inherits the environment, so the API key is never on a
/// command line. It is what `runpod-pod.sh` / `runpod-http.sh` do with
/// `nohup bash -c 'sleep $CAP; curl -X DELETE …' &`, plus an early exit once
/// the instance is gone.
#[derive(Clone, Debug)]
pub struct DetachedBackstop {
    pub exe: PathBuf,
    /// Appended to; `None`: `/dev/null`.
    pub log: Option<PathBuf>,
}

impl DetachedBackstop {
    pub fn command(
        &self,
        kind: ProviderKind,
        id: &InstanceId,
        deadline: f64,
    ) -> std::process::Command {
        use std::os::unix::process::CommandExt as _;
        let mut c = std::process::Command::new(&self.exe);
        c.args(backstop_args(kind, id, deadline));
        c.process_group(0);
        c
    }
}

impl Backstop for DetachedBackstop {
    fn arm(&self, kind: ProviderKind, id: &InstanceId, deadline: f64) -> Result<u32, GuardError> {
        use std::process::Stdio;
        let out = || -> Result<Stdio, GuardError> {
            Ok(match &self.log {
                Some(p) => Stdio::from(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(p)
                        .map_err(|e| GuardError::Backstop(format!("{}: {e}", p.display())))?,
                ),
                None => Stdio::null(),
            })
        };
        let child = self
            .command(kind, id, deadline)
            .stdin(Stdio::null())
            .stdout(out()?)
            .stderr(out()?)
            .spawn()
            .map_err(|e| GuardError::Backstop(format!("{}: {e}", self.exe.display())))?;
        Ok(child.id())
    }
}

/// What a backstop run did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackstopOutcome {
    /// The deadline passed and the instance was deleted.
    Deleted,
    /// The instance was gone before the deadline (deleted by `up` / `down`).
    AlreadyGone,
}

/// How often the backstop asks whether the instance still exists.
pub const BACKSTOP_CHECK: Duration = Duration::from_secs(60);
/// Delete attempts at the deadline (fv-control's watchdog: 3, 5 s apart).
pub const BACKSTOP_ATTEMPTS: u32 = 3;

/// The backstop process's body: wait for the deadline (checking every
/// minute whether the instance is already gone), then delete it.
pub fn run_backstop(
    p: &dyn Provider,
    id: &InstanceId,
    deadline: f64,
    clock: &dyn Clock,
    log: &dyn LogSink,
) -> Result<BackstopOutcome, ProviderError> {
    log.line(&format!(
        "[backstop] {} {id}: deleting at {deadline:.0} (in {:.0} s)",
        p.kind(),
        (deadline - clock.now()).max(0.0)
    ));
    loop {
        let left = deadline - clock.now();
        if left <= 0.0 {
            break;
        }
        clock.sleep(Duration::from_secs_f64(
            left.min(BACKSTOP_CHECK.as_secs_f64()),
        ));
        if deadline - clock.now() > 0.0 {
            if let Ok(InstanceState::Terminated) = p.status(id) {
                log.line(&format!("[backstop] {id} is gone; nothing to do"));
                return Ok(BackstopOutcome::AlreadyGone);
            }
        }
    }
    let mut last = None;
    for i in 1..=BACKSTOP_ATTEMPTS {
        match p.destroy(id) {
            Ok(()) => {
                log.line(&format!("[backstop] deadline: deleted {id}"));
                return Ok(BackstopOutcome::Deleted);
            }
            Err(e) => {
                log.line(&format!("[backstop] delete {id} (attempt {i}): {e}"));
                last = Some(e);
            }
        }
        if i < BACKSTOP_ATTEMPTS {
            clock.sleep(Duration::from_secs(5));
        }
    }
    Err(last.unwrap_or_else(|| ProviderError::Unsupported("backstop: no attempt made".into())))
}
