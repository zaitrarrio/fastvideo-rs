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
    Account, CheckedGuards, Guards, Instance, InstanceId, InstanceState, Launch, Offer, Provider,
    ProviderError, ProviderKind, WeightsSource,
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
/// The longest wall-clock cap without `--force-guards` (4 h).
pub const MAX_CAP_S: u64 = 4 * 3600;
/// How long `arm` waits for the backstop process to say it is armed.
pub const ARM_TIMEOUT: Duration = Duration::from_secs(15);
/// The line the backstop writes to its armed file once it can delete.
pub const ARMED: &str = "armed";
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

/// `x >= floor`, false when either is NaN: every floor fails closed.
pub fn at_least(x: f64, floor: f64) -> bool {
    matches!(
        x.partial_cmp(&floor),
        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    )
}

/// `x <= cap`, false when either is NaN: every cap fails closed.
pub fn at_most(x: f64, cap: f64) -> bool {
    matches!(
        x.partial_cmp(&cap),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    )
}

/// The default guards.
pub fn default_guards() -> Guards {
    Guards::all(
        MAX_USD_PER_HR,
        MIN_BALANCE_USD,
        CAP_S,
        READY_TIMEOUT_S,
        READY_CYCLES,
    )
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
        "{id}: the provider reported no (or no readable) billed price; refused as over the cap"
    )]
    BilledUnknown { id: String },
    #[error("balance ${balance:.2} would drop to ${projected:.2} (< the floor ${floor:.2}) over the {cap_s} s cap at ${usd_per_hr:.3}/hr plus the account's current ${spend_per_hr:.3}/hr")]
    BalanceWouldDropBelowFloor {
        balance: f64,
        projected: f64,
        floor: f64,
        usd_per_hr: f64,
        spend_per_hr: f64,
        cap_s: u64,
    },
    #[error(
        "the launch's {0} contains the API key: refused (secrets are references, never values)"
    )]
    SecretInLaunch(&'static str),
    #[error("guards: {0}")]
    Invalid(String),
    #[error("guards: {} not checked by the launcher that wrote this launch (null); fv-deploy checks every guard: set them", .0.join(", "))]
    NotChecked(Vec<&'static str>),
    #[error("backstop: {0}")]
    Backstop(String),
}

/// Runpod GPU instances only with the EU weights volume `bz0ofkfx45` in
/// `EU-RO-1` (CLAUDE.md; plan §1, owner instruction 2026-10-23); the deleted
/// US volume is refused by id and by name; volume weights only (C3a). Vast
/// and GCP have their own weights rules (C3b, §8).
pub fn check_region(l: &Launch) -> Result<(), GuardError> {
    if let Some(r) = &l.placement.region {
        if r == "us" {
            return Err(GuardError::UsGone(US_GONE.into()));
        }
        if r != EU_REGION {
            return Err(GuardError::Region(format!("unknown region {r} (eu)")));
        }
    }
    match l.provider {
        ProviderKind::RunpodPod | ProviderKind::RunpodServerless => {}
        ProviderKind::Vast => return check_vast(l),
        ProviderKind::Gcp => return check_gcp(l),
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

/// Vast (C3b, docs/adr/0018-fv-deploy.md §8.2): no weights volume, and r2
/// weights are not in fv-deploy yet: weights `none` (the fake-engine smoke).
fn check_vast(l: &Launch) -> Result<(), GuardError> {
    if l.placement.volume.is_some() || l.weights != WeightsSource::None {
        return Err(GuardError::Region(format!(
            "Vast: weights source {} with {}: Vast mounts no weights volume and fv-deploy does not fetch r2 weights yet (only weights none)",
            l.weights.as_str(),
            l.placement
                .volume
                .as_ref()
                .map_or_else(|| "no volume".to_owned(), |v| format!("volume {}", v.id))
        )));
    }
    Ok(())
}

/// GCP (C3b, §8.3): the weights are the zone's read-only weights disk
/// (`fv-weights-hdml-<zone>`) or none (the fake engine); the VM runs in an
/// EU zone (`europe-*`, CLAUDE.md: EU only).
fn check_gcp(l: &Launch) -> Result<(), GuardError> {
    let Some(zone) = l.placement.data_centers.first() else {
        return Err(GuardError::Region("GCP: no zone".into()));
    };
    if !zone.starts_with("europe-") {
        return Err(GuardError::Region(format!(
            "GCP zone {zone}: EU zones only (europe-*; CLAUDE.md, the family Durable Objects in weur)"
        )));
    }
    match (&l.placement.volume, l.weights) {
        (Some(v), WeightsSource::Volume) => {
            let want = crate::provider::gcp::weights_disk(zone);
            if v.id != want {
                return Err(GuardError::Region(format!(
                    "GCP: weights disk {}: the zone's weights disk is {want} (read-only)",
                    v.id
                )));
            }
            Ok(())
        }
        (None, WeightsSource::None) => Ok(()),
        (v, w) => Err(GuardError::Region(format!(
            "GCP: weights source {} with {}: the weights disk (volume) or none",
            w.as_str(),
            v.as_ref()
                .map_or_else(|| "no disk".to_owned(), |v| format!("disk {}", v.id))
        ))),
    }
}

/// The default price cap of a provider ($/hr): Runpod's for one GPU
/// ([`MAX_USD_PER_HR`]), the scripts' for Vast (`VAST_MAX_DPH` 0.60) and for
/// a whole GCP VM (`FV_GCP_MAX_DPH` 6.0).
pub fn default_max_usd_per_hr(kind: ProviderKind) -> f64 {
    match kind {
        ProviderKind::RunpodPod | ProviderKind::RunpodServerless => MAX_USD_PER_HR,
        ProviderKind::Vast => VAST_MAX_USD_PER_HR,
        ProviderKind::Gcp => crate::provider::gcp::MAX_USD_PER_HR,
    }
}

/// `vast.sh` `VAST_MAX_DPH`.
pub const VAST_MAX_USD_PER_HR: f64 = 0.6;

/// The guard numbers themselves: every one present (a `null`, "not checked
/// by this launcher", is refused: fv-deploy checks them all) and within the
/// owner's bounds. The numbers `up` runs on.
pub fn check_guards(g: &Guards) -> Result<CheckedGuards, GuardError> {
    let g = g.checked().map_err(GuardError::NotChecked)?;
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
    // The owner's rules are upper bounds too (plan §1); `--force-guards`
    // lifts them (never the $8 floor).
    if !g.force {
        if g.cap_s > MAX_CAP_S {
            return bad("the wall-clock cap is at most 4 h (14400 s) without --force-guards");
        }
        if g.ready_cycles > READY_CYCLES {
            return bad("at most 3 readiness cycles (the owner's rule) without --force-guards");
        }
        if g.ready_timeout_s > READY_TIMEOUT_S {
            return bad(
                "the readiness timeout is at most 600 s (the owner's rule) without --force-guards",
            );
        }
    }
    Ok(g)
}

/// A serverless endpoint's price cap is checked on the GPU's pod quote for one
/// worker (docs/adr/0018-fv-deploy.md §5.3: no serverless quote is read), so
/// more than one worker is refused.
pub fn check_serverless(l: &Launch) -> Result<(), GuardError> {
    if let Some(s) = &l.serverless {
        if s.workers_max > 1 {
            return Err(GuardError::Invalid(format!(
                "workers_max {}: the price cap covers one worker (ADR-0018 §5.3)",
                s.workers_max
            )));
        }
    }
    Ok(())
}

/// A host-RAM floor (`GpuSpec::min_ram_gb`, Runpod `minRAMPerGPU`) is a
/// positive number of GB per GPU, on a Runpod pod only: no other create body
/// carries it, so elsewhere it would be dropped unsaid (docs/adr/0019-fv-lab.md
/// §16.10).
pub fn check_min_ram(l: &Launch) -> Result<(), GuardError> {
    match l.gpu.min_ram_gb {
        None => Ok(()),
        Some(0) => Err(GuardError::Invalid(
            "min_ram_gb 0: the host-RAM floor is a positive number of GB per GPU (Runpod minRAMPerGPU)"
                .into(),
        )),
        Some(gb) if l.provider != ProviderKind::RunpodPod => Err(GuardError::Invalid(format!(
            "min_ram_gb {gb}: a host-RAM floor is a Runpod pod's minRAMPerGPU; a {} launch has none",
            l.provider
        ))),
        Some(_) => Ok(()),
    }
}

/// The floor over the whole cap (CLAUDE.md: stop before the balance *would*
/// drop below $8): `balance − (usd_per_hr + current spend) × cap_s / 3600 ≥ floor`.
pub fn check_projection(
    a: &Account,
    usd_per_hr: f64,
    cap_s: u64,
    floor: f64,
) -> Result<(), GuardError> {
    let spend = a
        .spend_per_hr_usd
        .filter(|x| x.is_finite() && *x > 0.0)
        .unwrap_or(0.0);
    let projected = a.balance_usd - (usd_per_hr + spend) * cap_s as f64 / 3600.0;
    if !at_least(projected, floor) {
        return Err(GuardError::BalanceWouldDropBelowFloor {
            balance: a.balance_usd,
            projected,
            floor,
            usd_per_hr,
            spend_per_hr: spend,
            cap_s,
        });
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
    if !at_least(a.balance_usd, floor) {
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
        Some(p) if !at_most(p, cap) => Err(GuardError::OverCap {
            gpu: o.gpu_type.clone(),
            usd: p,
            cap,
        }),
        Some(p) => Ok(p),
    }
}

/// The price cap on what the created instance is billed (a provider that
/// bills at create): over the cap, or no readable price, fails.
pub fn check_billed(i: &Instance, cap: f64) -> Result<(), GuardError> {
    match i.usd_per_hr {
        Some(p) if !at_most(p, cap) => Err(GuardError::BilledOverCap {
            id: i.id.to_string(),
            usd: p,
            cap,
        }),
        Some(_) => Ok(()),
        None => Err(GuardError::BilledUnknown {
            id: i.id.to_string(),
        }),
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
pub fn backstop_args(
    kind: ProviderKind,
    id: &InstanceId,
    deadline: f64,
    armed: &std::path::Path,
) -> Vec<String> {
    vec![
        "backstop".into(),
        "--kind".into(),
        kind.as_str().into(),
        "--id".into(),
        id.to_string(),
        "--deadline".into(),
        format!("{:.0}", deadline.ceil()),
        "--armed-file".into(),
        armed.display().to_string(),
    ]
}

/// The detached backstop (docs/adr/0018-fv-deploy.md §5.4): this binary
/// again, as `fv-deploy backstop --kind K --id ID --deadline T --armed-file F`,
/// in a process group of its own (a Ctrl-C to the terminal's foreground group
/// does not reach it), stdin closed, output to a log file; it ignores SIGHUP
/// (as `nohup` did) and inherits the environment, so the API key is never on
/// a command line. It is what `runpod-pod.sh` / `runpod-http.sh` do with
/// `nohup bash -c 'sleep $CAP; curl -X DELETE …' &`, plus a handshake (the
/// child writes [`ARMED`] to `F` once its client is built and its signal
/// handlers are in place; `arm` fails unless that happens within
/// `ready_timeout`) and an early exit once the instance is gone.
#[derive(Clone, Debug)]
pub struct DetachedBackstop {
    pub exe: PathBuf,
    /// Appended to; `None`: `/dev/null`.
    pub log: Option<PathBuf>,
    /// Where the armed files go (a private directory: the state dir).
    pub armed_dir: PathBuf,
    pub ready_timeout: Duration,
    /// Set on the child (tests point it at a fake API).
    pub env: Vec<(String, String)>,
    /// Removed from the child's environment.
    pub env_remove: Vec<String>,
}

impl DetachedBackstop {
    pub fn new(exe: PathBuf, log: Option<PathBuf>, armed_dir: PathBuf) -> Self {
        Self {
            exe,
            log,
            armed_dir,
            ready_timeout: ARM_TIMEOUT,
            env: Vec::new(),
            env_remove: Vec::new(),
        }
    }

    pub fn armed_file(&self, id: &InstanceId) -> PathBuf {
        let safe: String = id
            .as_str()
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.armed_dir
            .join(format!("armed-{safe}-{}", std::process::id()))
    }

    pub fn command(
        &self,
        kind: ProviderKind,
        id: &InstanceId,
        deadline: f64,
    ) -> std::process::Command {
        use std::os::unix::process::CommandExt as _;
        let mut c = std::process::Command::new(&self.exe);
        c.args(backstop_args(kind, id, deadline, &self.armed_file(id)));
        for k in &self.env_remove {
            c.env_remove(k);
        }
        for (k, v) in &self.env {
            c.env(k, v);
        }
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
        std::fs::create_dir_all(&self.armed_dir)
            .map_err(|e| GuardError::Backstop(format!("{}: {e}", self.armed_dir.display())))?;
        let armed = self.armed_file(id);
        let _ = std::fs::remove_file(&armed);
        let mut child = self
            .command(kind, id, deadline)
            .stdin(Stdio::null())
            .stdout(out()?)
            .stderr(out()?)
            .spawn()
            .map_err(|e| GuardError::Backstop(format!("{}: {e}", self.exe.display())))?;
        let t0 = std::time::Instant::now();
        loop {
            if std::fs::read_to_string(&armed).is_ok_and(|t| t.trim() == ARMED) {
                let _ = std::fs::remove_file(&armed);
                return Ok(child.id());
            }
            if let Ok(Some(st)) = child.try_wait() {
                return Err(GuardError::Backstop(format!(
                    "the backstop for {id} exited at start ({st}); see its log"
                )));
            }
            if t0.elapsed() >= self.ready_timeout {
                let _ = child.kill();
                let _ = child.wait();
                return Err(GuardError::Backstop(format!(
                    "the backstop for {id} did not report armed within {} s",
                    self.ready_timeout.as_secs()
                )));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// The backstop's side of the handshake.
pub fn report_armed(path: &std::path::Path) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, format!("{ARMED}\n"))?;
    std::fs::rename(&tmp, path)
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

/// The backstop process's body: wait for the deadline, then delete it. It
/// stands down early only after two "gone" readings (a 404) at least a minute
/// apart, never on a single answer.
pub fn run_backstop(
    p: &dyn Provider,
    id: &InstanceId,
    deadline: f64,
    clock: &dyn Clock,
    log: &dyn LogSink,
) -> Result<BackstopOutcome, ProviderError> {
    run_backstop_recorded(p, id, deadline, clock, log, None)
}

/// [`run_backstop`], telling the deployments ledger (D1, C3d) when it
/// deleted the instance at the deadline (warn-only, as every ledger write:
/// a failure is a log line). An instance already gone was recorded by
/// whoever deleted it.
pub fn run_backstop_recorded(
    p: &dyn Provider,
    id: &InstanceId,
    deadline: f64,
    clock: &dyn Clock,
    log: &dyn LogSink,
    recorder: Option<&dyn crate::lifecycle::Recorder>,
) -> Result<BackstopOutcome, ProviderError> {
    let out = backstop_body(p, id, deadline, clock, log)?;
    if let (BackstopOutcome::Deleted, Some(r)) = (&out, recorder) {
        if let Err(e) = r.deleted(p.kind(), id, "deleted") {
            log.line(&format!(
                "[backstop] WARNING: the deployments ledger (D1) could not record the delete of {id}: {e}"
            ));
        }
    }
    Ok(out)
}

fn backstop_body(
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
    let mut gone_since: Option<f64> = None;
    loop {
        let left = deadline - clock.now();
        if left <= 0.0 {
            break;
        }
        clock.sleep(Duration::from_secs_f64(
            left.min(BACKSTOP_CHECK.as_secs_f64()),
        ));
        if deadline - clock.now() > 0.0 {
            match p.status(id) {
                Ok(InstanceState::Terminated) => match gone_since {
                    Some(t) if clock.now() - t >= BACKSTOP_CHECK.as_secs_f64() => {
                        log.line(&format!("[backstop] {id} is gone; nothing to do"));
                        return Ok(BackstopOutcome::AlreadyGone);
                    }
                    Some(_) => {}
                    None => gone_since = Some(clock.now()),
                },
                _ => gone_since = None,
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
