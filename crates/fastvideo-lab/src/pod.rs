//! `fv-lab pod {up,status,fetch,down,reap}` (docs/adr/0019-fv-lab.md §6) on
//! fv-deploy's `Provider` trait and lifecycle: the price cap, the balance
//! floor, the EU-volume region guard, the detached backstop and the
//! 600 s / 3-cycle readiness rule are `fastvideo_deploy`'s, reused, not
//! copied. What is fv-lab's:
//!
//! - [`launch`]: the `Launch` of a tier for a commit: the runtime image
//!   `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>`, the tier's GPU, disk
//!   and caps, the EU volume, and [`start_command`], which is
//!   `scripts/gpu/runpod-http.sh`'s start command with `runpod-matrix.sh`
//!   replaced by `fv-lab run` (the payload golden,
//!   `tests/payload_golden.rs`, holds the rest equal);
//! - [`LabPod`]: the pod provider whose readiness is the run directory's
//!   HTTP server answering (`GET /<family>/`, what `runpod-http.sh wait_up`
//!   polls), since a lab pod serves files, not fv-serve's `/ping`;
//! - [`fetch_run`] / [`wait_done`]: `runpod-http.sh fetch_results` and
//!   `wait_done`, into `artifacts/runpod/<family>/<tag>/`;
//! - [`reap_recorded`] / [`reap_account`]: which fv-lab pods are past their
//!   cap (this machine's records by default; the full name pattern with
//!   `--account`); [`judge_run`]: whether a fetched run succeeded.

use std::path::{Path, PathBuf};
use std::time::Duration;

use fastvideo_deploy::boot::sys::{Clock, Http, LogSink, Method, Request};
use fastvideo_deploy::guard;
use fastvideo_deploy::launch::{utc_parts, POD_VOLUME_MOUNT};
use fastvideo_deploy::provider::runpod::{Pod, PodListing};
use fastvideo_deploy::provider::{
    Account, ApiCall, Cursor, Disk, GpuSpec, Guards, Instance, InstanceId, InstanceState, Launch,
    LogPage, Offer, OfferQuery, Placement, Probe, Provider, ProviderError, ProviderKind, Reach,
    ReachOptions, VolumeMount, LAUNCH_SCHEMA,
};
use fastvideo_manifest::deploy::{EU_DATA_CENTER, EU_VOLUME_ID};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::plan::Tier;

/// Every fv-lab pod's name starts with this (`reap` deletes only these).
pub const POD_PREFIX: &str = "fv-lab-";
/// `fv-lab` in the runtime image.
pub const FV_LAB_BIN: &str = "/opt/fastvideo-rs/target/release/fv-lab";
/// The template of the pod's start command: `runpod-http.sh start_cmd` for
/// `run`, expanded with its defaults, `@…@` marking what varies.
const START_TEMPLATE: &str = include_str!("pod-start.sh.in");
/// What `reap` adds to a pod's cap before it deletes it (the backstop
/// deletes at the cap; reap catches a pod whose backstop died with its host).
pub const REAP_GRACE_S: u64 = 600;

/// Why a pod command failed.
#[derive(thiserror::Error, Debug)]
pub enum PodError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("{url}: {why}")]
    Fetch { url: String, why: String },
    #[error("{path}: {why}")]
    Io { path: PathBuf, why: String },
}

/// What the start command varies (`runpod-http.sh`'s environment, read at
/// plan time).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartOptions {
    pub family: String,
    pub tag: String,
    /// `verify-weights.sh` cells for `weights.log`.
    pub preflight: Vec<String>,
    /// `FV_GEN_TIMEOUT_S`.
    pub cell_timeout_s: u64,
    /// `FV_EXTRA_ENV`: space-separated `K=V` for the run.
    pub extra_env: Vec<String>,
    /// `FV_CELLS`: only these arms.
    pub cells: Vec<String>,
    /// `FV_LPIPS=1`.
    pub lpips: bool,
    /// `FV_IDLE_GPU_MIN` (0: no idle watchdog), `FV_IDLE_GPU_GRACE_MIN`,
    /// `FV_IDLE_NET_BYTES`, `FV_MIN_DRIVER`.
    pub idle_min: u32,
    pub idle_grace_min: u32,
    pub idle_net_bytes: u64,
    pub min_driver: u32,
    /// `FV_SKIP_TAE=1` (the tier's `tae = false`).
    pub skip_tae: bool,
    /// The runner (`fv-lab` in the image).
    pub runner: String,
    /// The pod's own wall cap (`FV_POD_CAP_S`, the tier's `wall_cap_s`): the
    /// start command deletes the pod this many seconds after the container
    /// starts, with the pod-scoped key, whatever the host that made it does
    /// (docs/adr/0023-lab-report-and-ci-bridge.md §4).
    pub cap_s: u64,
}

impl StartOptions {
    /// `runpod-http.sh`'s defaults for a tier.
    pub fn for_tier(tier: &Tier, tag: &str) -> Self {
        Self {
            family: tier.name.clone(),
            tag: tag.to_owned(),
            preflight: tier.preflight.clone(),
            cell_timeout_s: tier.cell_timeout_s,
            extra_env: Vec::new(),
            cells: Vec::new(),
            lpips: false,
            idle_min: 10,
            idle_grace_min: 30,
            idle_net_bytes: 262_144,
            min_driver: 580,
            skip_tae: !tier.tae,
            runner: format!("{FV_LAB_BIN} run"),
            cap_s: tier.wall_cap_s,
        }
    }

    fn check(&self) -> Result<(), PodError> {
        if self.cap_s == 0 {
            return Err(PodError::Invalid(
                "a pod without a wall cap (cap_s 0): every fv-lab pod deletes itself at its cap"
                    .into(),
            ));
        }
        let safe = |s: &str| {
            !s.is_empty()
                && s.bytes().all(|b| {
                    b.is_ascii_alphanumeric()
                        || matches!(b, b'.' | b'_' | b'-' | b'/' | b':' | b',' | b'=')
                })
        };
        for (what, v) in [("family", &self.family), ("tag", &self.tag)] {
            if !safe(v) || v.contains('=') {
                return Err(PodError::Invalid(format!(
                    "{what} `{v}` is not a plain name"
                )));
            }
        }
        for c in self.preflight.iter().chain(&self.cells) {
            if !safe(c) || c.contains('=') {
                return Err(PodError::Invalid(format!("`{c}` is not a plain name")));
            }
        }
        for e in &self.extra_env {
            let ok = e.split_once('=').is_some_and(|(k, _)| {
                !k.is_empty()
                    && k.bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
            }) && safe(e);
            if !ok {
                return Err(PodError::Invalid(format!(
                    "FV_EXTRA_ENV entry `{e}` is not K=V with a plain value"
                )));
            }
        }
        Ok(())
    }
}

/// The pod's start command (`dockerStartCmd: ["/bin/bash", "-c", <this>]`).
pub fn start_command(o: &StartOptions) -> Result<String, PodError> {
    o.check()?;
    let mut s = START_TEMPLATE.trim_end_matches('\n').to_owned();
    if o.idle_min == 0 {
        // `idle_watchdog` prints nothing when FV_IDLE_GPU_MIN=0.
        s = s
            .split_once("set -u\n")
            .map(|(_, rest)| format!("set -u\n{rest}"))
            .unwrap_or(s);
    }
    let lpips = if o.lpips { "FV_LPIPS=1" } else { "" };
    for (k, v) in [
        ("@FAMILY@", o.family.clone()),
        ("@TAG@", o.tag.clone()),
        ("@PREFLIGHT@", o.preflight.join(" ")),
        ("@GEN_TIMEOUT@", o.cell_timeout_s.to_string()),
        ("@EXTRA_ENV@", o.extra_env.join(" ")),
        ("@FV_CELLS@", o.cells.join(" ")),
        ("@PROMPTS@", String::new()),
        ("@LPIPS@", lpips.to_owned()),
        ("@IDLE_MIN@", o.idle_min.to_string()),
        ("@GRACE_MIN@", o.idle_grace_min.to_string()),
        ("@NET_BYTES@", o.idle_net_bytes.to_string()),
        ("@MIN_DRIVER@", o.min_driver.to_string()),
        ("@SKIP_TAE@", if o.skip_tae { "1" } else { "0" }.to_owned()),
        ("@RUNNER@", o.runner.clone()),
        ("@CAP_S@", o.cap_s.to_string()),
    ] {
        s = s.replace(k, &v);
    }
    Ok(s)
}

/// `pod up --cells`: every name must be an arm of the tier (a typo or a
/// flag-looking word would otherwise run nothing, silently, or worse).
pub fn check_cells<'a>(
    tier: &str,
    arms: impl IntoIterator<Item = &'a str>,
    cells: &[String],
) -> Result<(), PodError> {
    let arms: Vec<&str> = arms.into_iter().collect();
    match cells.iter().find(|c| !arms.contains(&c.as_str())) {
        Some(c) => Err(PodError::Invalid(format!(
            "--cells `{c}` is not an arm of tier `{tier}`"
        ))),
        None => Ok(()),
    }
}

/// The pods a fv-deploy ledger (`<state>/ledger.tsv`, lines `<UTC>\t<kind>-created
/// <id> …` and `<UTC>\t<kind>-deleted <id> …`) records as created and not
/// since deleted, in creation order: what `fv-lab pod down --from-ledger`
/// deletes. The ledger line is written right after the create, before the
/// backstop and readiness, so a run interrupted at any point after the create
/// leaves its pod here (docs/adr/0023-lab-report-and-ci-bridge.md §4). Only
/// `runpod-pod` lines: fv-lab makes no other kind. A `delete-failed` line
/// keeps the pod listed.
pub fn undeleted_in_ledger(ledger: &str) -> Vec<String> {
    let mut live: Vec<String> = Vec::new();
    for line in ledger.lines() {
        let event = line.split_once('\t').map_or(line, |(_, e)| e);
        let mut words = event.split_whitespace();
        let (Some(what), Some(id)) = (words.next(), words.next()) else {
            continue;
        };
        match what {
            "runpod-pod-created" => {
                if !live.iter().any(|x| x == id) {
                    live.push(id.to_owned());
                }
            }
            "runpod-pod-deleted" => live.retain(|x| x != id),
            _ => {}
        }
    }
    live
}

/// A 7-hex-digit commit (`--sha`).
pub fn check_sha7(sha: &str) -> Result<(), PodError> {
    check_sha7_named("--sha", sha)
}

/// A 7-hex-digit commit given as `what` (a flag or variable, for the error).
pub fn check_sha7_named(what: &str, sha: &str) -> Result<(), PodError> {
    if sha.len() == 7
        && sha
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(PodError::Invalid(format!(
            "{what} `{sha}`: 7 lower-case hex digits"
        )))
    }
}

/// The baseline commit of a pod's run (ADR-0019 §9.2): `--baseline-sha`
/// (`flag`) or one `FV_BASELINE_SHA=<7>` entry of `--extra-env`, not both;
/// 7 hex digits (an empty value is refused); never the run's own commit
/// `sha7` (a byte-identity gate of a build against itself passes whatever it
/// does). A flag is appended to `extra_env`. `needed`: the tier has
/// `bin = "baseline"` arms, so a baseline commit is required.
pub fn baseline_for_pod(
    sha7: &str,
    flag: Option<&str>,
    extra_env: &mut Vec<String>,
    needed: bool,
) -> Result<Option<String>, PodError> {
    const KEY: &str = "FV_BASELINE_SHA=";
    let from_env: Vec<&str> = extra_env
        .iter()
        .filter_map(|e| e.strip_prefix(KEY))
        .collect();
    if from_env.len() > 1 || (flag.is_some() && !from_env.is_empty()) {
        return Err(PodError::Invalid(
            "the baseline commit given twice (--baseline-sha and/or FV_BASELINE_SHA in --extra-env)".into(),
        ));
    }
    let chosen = match (flag, from_env.first()) {
        (Some(f), _) => {
            check_sha7_named("--baseline-sha", f)?;
            Some(f.to_owned())
        }
        (None, Some(v)) => {
            check_sha7_named("--extra-env FV_BASELINE_SHA", v)?;
            Some((*v).to_owned())
        }
        (None, None) => None,
    };
    if let Some(b) = &chosen {
        if b == sha7 {
            return Err(PodError::Invalid(format!(
                "the baseline commit {b} is the run's own commit: a byte-identity gate against itself proves nothing"
            )));
        }
        if flag.is_some() {
            extra_env.push(format!("{KEY}{b}"));
        }
    } else if needed {
        return Err(PodError::Invalid(
            "the tier runs `bin = \"baseline\"` arms: pass --baseline-sha <7>, the baseline commit"
                .into(),
        ));
    }
    Ok(chosen)
}

/// fv-lab's run tag: `<sha>-<MMDDHHMMSS>-<4 hex>` (UTC, and a random
/// suffix). `runpod-http.sh`'s `<sha>-<MMDDHHMM>` repeats within a minute,
/// and two runs sharing a pod name would sweep each other's pod (fv-deploy's
/// exit sweep deletes by exact name); `pod up` also refuses a name that is
/// already live ([`name_taken`]).
pub fn run_tag(sha7: &str, unix: f64, suffix: u16) -> String {
    let (_, m, d, h, mi, sec) = utc_parts(unix);
    format!("{sha7}-{m:02}{d:02}{h:02}{mi:02}{sec:02}-{suffix:04x}")
}

/// Two random bytes from `/dev/urandom`, for [`run_tag`].
pub fn random_suffix() -> Result<u16, PodError> {
    use std::io::Read as _;
    let mut b = [0u8; 2];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| PodError::Io {
            path: PathBuf::from("/dev/urandom"),
            why: e.to_string(),
        })?;
    Ok(u16::from_be_bytes(b))
}

/// `fv-lab-<family>-<tag>` (`runpod-http.sh` with `FV_POD_PREFIX=fv-lab`).
pub fn pod_name(family: &str, tag: &str) -> String {
    format!("{POD_PREFIX}{family}-{tag}")
}

/// The `Launch` of `tier` at `sha7`: one pod with the start command, the
/// EU volume at `/workspace`, the tier's caps under fv-deploy's guards.
pub fn launch(tier: &Tier, sha7: &str, start: &StartOptions) -> Result<Launch, PodError> {
    check_sha7(sha7)?;
    let cmd = start_command(start)?;
    Ok(Launch {
        schema: LAUNCH_SCHEMA,
        name: pod_name(&tier.name, &start.tag),
        provider: ProviderKind::RunpodPod,
        source: None,
        image: tier.image.image(sha7),
        entrypoint: Vec::new(),
        args: vec!["/bin/bash".into(), "-c".into(), cmd],
        env: Default::default(),
        ports: vec!["8000/http".into()],
        gpu: GpuSpec {
            types: vec![tier.gpu.clone()],
            count: tier.gpu_count,
            allowed_cuda: Vec::new(),
            min_cuda: None,
        },
        placement: Placement {
            region: Some(tier.region.clone()),
            data_centers: vec![EU_DATA_CENTER.into()],
            volume: Some(VolumeMount {
                id: EU_VOLUME_ID.into(),
                mount_path: POD_VOLUME_MOUNT.into(),
            }),
            cloud: tier.cloud,
        },
        disk: Disk {
            container_gb: tier.container_disk_gb,
            volume_gb: 0,
        },
        weights: tier.weights,
        guards: Guards {
            max_usd_per_hr: tier.max_usd_per_hr,
            min_balance_usd: guard::MIN_BALANCE_USD,
            cap_s: tier.wall_cap_s,
            ready_timeout_s: guard::READY_TIMEOUT_S,
            ready_cycles: guard::READY_CYCLES,
            force: false,
        },
        serverless: None,
    })
}

// ------------------------------------------------------------------ provider

/// Runpod pods whose readiness is the run directory's server: everything is
/// [`Pod`]'s but [`Provider::reach`].
#[derive(Debug)]
pub struct LabPod<'a> {
    pub inner: Pod<'a>,
    pub family: String,
}

impl Provider for LabPod<'_> {
    fn kind(&self) -> ProviderKind {
        self.inner.kind()
    }
    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>, ProviderError> {
        self.inner.offers(q)
    }
    fn payload(&self, launch: &Launch) -> Result<Vec<ApiCall>, ProviderError> {
        self.inner.payload(launch)
    }
    fn create(&self, launch: &Launch) -> Result<Instance, ProviderError> {
        self.inner.create(launch)
    }
    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError> {
        self.inner.status(id)
    }
    fn logs(&self, id: &InstanceId, since: Option<&Cursor>) -> Result<LogPage, ProviderError> {
        self.inner.logs(id, since)
    }
    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError> {
        self.inner.destroy(id)
    }
    fn account(&self) -> Result<Account, ProviderError> {
        self.inner.account()
    }
    fn reach<'b>(
        &'b self,
        instance: &Instance,
        _opts: &ReachOptions<'_>,
    ) -> Result<Box<dyn Reach + 'b>, ProviderError> {
        let base = instance
            .url
            .clone()
            .unwrap_or_else(|| self.inner.api.urls.pod_url(instance.id.as_str()));
        Ok(Box::new(LabReach {
            http: self.inner.api.http(),
            base,
            family: self.family.clone(),
        }))
    }
    fn sweep(
        &self,
        name: &str,
        keep: Option<&InstanceId>,
    ) -> Result<Vec<InstanceId>, ProviderError> {
        self.inner.sweep(name, keep)
    }
    fn bills_at_create(&self) -> bool {
        self.inner.bills_at_create()
    }
}

/// A lab pod's file server through the HTTPS proxy.
pub struct LabReach<'a> {
    pub http: &'a dyn Http,
    pub base: String,
    pub family: String,
}

impl Reach for LabReach<'_> {
    /// Ready when `GET /<family>/` answers 200 (the start command made the
    /// directory before it started the server).
    fn ping(&mut self) -> Probe {
        let url = format!("{}/{}/", self.base, self.family);
        match self
            .http
            .send(&Request::new(Method::Get, &url, Duration::from_secs(30)))
        {
            Ok(r) if r.status == 200 => Probe::Ready,
            Ok(r) => Probe::Down(format!("/{}/ {}", self.family, r.status)),
            Err(e) => Probe::Down(e.to_string()),
        }
    }

    fn call(
        &mut self,
        method: Method,
        path: &str,
        _body: Option<&Value>,
        _stop: &dyn Fn() -> bool,
    ) -> Result<(u16, Value), ProviderError> {
        let url = format!("{}{path}", self.base);
        let r = self
            .http
            .send(&Request::new(method, &url, Duration::from_secs(30)))
            .map_err(|e| ProviderError::Transport {
                what: url.clone(),
                why: e.to_string(),
            })?;
        Ok((r.status, fastvideo_deploy::provider::body_json(&r.body)))
    }
}

// ------------------------------------------------------------------ records

/// What `up` records about a pod (`artifacts/fv-lab/pods/<id>.json`, and
/// `pod.json` in the run directory).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PodRecord {
    pub id: String,
    pub name: String,
    pub family: String,
    pub tag: String,
    pub sha: String,
    pub image: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usd_per_hr: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<f64>,
    pub cap_s: u64,
}

/// `artifacts/runpod/<family>/<tag>` under `root`: where `runpod-http.sh`
/// puts a run, so existing summaries and compares read fv-lab runs.
pub fn run_dir(root: &Path, family: &str, tag: &str) -> PathBuf {
    root.join("artifacts/runpod").join(family).join(tag)
}

// ------------------------------------------------------------------ reap

/// An fv-lab pod name, `fv-lab-<tier>-<sha7>-<MMDDHHMMSS>-<4 hex>`, taken apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PodName<'a> {
    pub tier: &'a str,
    pub sha: &'a str,
    pub stamp: &'a str,
    pub suffix: &'a str,
}

/// The full fv-lab pattern, or `None`. A `runpod-http.sh` pod made with
/// `FV_POD_PREFIX=fv-lab` (`…-<sha>-<MMDDHHMM>`) does not match.
pub fn parse_pod_name(name: &str) -> Option<PodName<'_>> {
    let rest = name.strip_prefix(POD_PREFIX)?;
    let mut parts = rest.rsplitn(4, '-');
    let suffix = parts.next()?;
    let stamp = parts.next()?;
    let sha = parts.next()?;
    let tier = parts.next()?;
    let hex = |s: &str, n: usize| {
        s.len() == n
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    let ok = hex(suffix, 4)
        && stamp.len() == 10
        && stamp.bytes().all(|b| b.is_ascii_digit())
        && hex(sha, 7)
        && crate::plan::is_name(tier);
    ok.then_some(PodName {
        tier,
        sha,
        stamp,
        suffix,
    })
}

/// Whether a live pod (not terminated) already has this exact name.
pub fn name_taken(pods: &[PodListing], name: &str) -> bool {
    pods.iter()
        .any(|p| p.name == name && p.state != InstanceState::Terminated)
}

fn past_cap(p: &PodListing, created: Option<f64>, cap_s: u64, now: f64) -> bool {
    p.state != InstanceState::Terminated
        && created.is_some_and(|t| now - t > (cap_s + REAP_GRACE_S) as f64)
}

/// `pod reap` (the default): the pods this machine's `pod up` recorded
/// (`artifacts/fv-lab/pods/<id>.json`) that are still live and older than
/// their recorded cap plus [`REAP_GRACE_S`]. Nothing else is touched
/// (CLAUDE.md: only pods you created).
pub fn reap_recorded<'a>(
    pods: &'a [PodListing],
    now: f64,
    records: &[PodRecord],
) -> Vec<&'a PodListing> {
    pods.iter()
        .filter(|p| {
            records
                .iter()
                .find(|r| r.id == p.id.as_str())
                .is_some_and(|r| past_cap(p, p.created_at.or(r.created_at), r.cap_s, now))
        })
        .collect()
}

/// `pod reap --account`: every pod whose name is the full fv-lab pattern of a
/// known tier ([`parse_pod_name`]; `caps` maps tier name to its wall cap),
/// older than that cap plus [`REAP_GRACE_S`]. A pod without a readable
/// creation time is never taken.
pub fn reap_account<'a>(
    pods: &'a [PodListing],
    now: f64,
    caps: &[(&str, u64)],
) -> Vec<&'a PodListing> {
    pods.iter()
        .filter(|p| {
            parse_pod_name(&p.name)
                .and_then(|n| caps.iter().find(|(t, _)| *t == n.tier))
                .is_some_and(|(_, cap)| past_cap(p, p.created_at, *cap, now))
        })
        .collect()
}

/// What makes a fetched run a success (`runpod-http.sh run`'s verdicts, made
/// strict): `DONE` says `done <tag>`, `matrix.out` holds `matrix_exit=0` and
/// no other exit, and the host's driver was not too old. The problems; empty
/// when none.
pub fn judge_run(dir: &Path, tag: &str) -> Vec<String> {
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
    let mut bad = Vec::new();
    match read("DONE") {
        Some(t) if t.lines().next() == Some(format!("done {tag}").as_str()) => {}
        Some(t) => bad.push(format!("DONE says {:?}, not `done {tag}`", t.trim())),
        None => bad.push("no DONE: the run did not finish (or was not fetched)".into()),
    }
    match read("matrix.out") {
        None => bad.push("no matrix.out fetched".into()),
        Some(t) => {
            let exits: Vec<&str> = t
                .lines()
                .filter(|l| l.starts_with("matrix_exit="))
                .collect();
            if exits.is_empty() {
                bad.push("matrix.out has no matrix_exit line".into());
            } else if exits.iter().any(|l| *l != "matrix_exit=0") {
                bad.push(format!(
                    "the runner exited abnormally: {}",
                    exits.join(", ")
                ));
            }
        }
    }
    if let Some(l) = read("box.txt")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("driver_too_old"))
    {
        bad.push(format!("host driver too old ({l}); rerun"));
    }
    bad
}

// ------------------------------------------------------------------ fetch

/// The `href="…"` targets of a directory listing.
pub fn hrefs(listing: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = listing;
    while let Some(i) = rest.find("href=\"") {
        let after = &rest[i + 6..];
        let Some(j) = after.find('"') else { break };
        out.push(after[..j].to_owned());
        rest = &after[j + 1..];
    }
    out
}

/// One proxy GET (`curl --max-time 30 --fail`).
fn get(http: &dyn Http, url: &str, timeout: Duration) -> Result<Vec<u8>, PodError> {
    let r = http
        .send(&Request::new(Method::Get, url, timeout))
        .map_err(|e| PodError::Fetch {
            url: url.to_owned(),
            why: e.to_string(),
        })?;
    if !r.ok() {
        return Err(PodError::Fetch {
            url: url.to_owned(),
            why: format!("HTTP {}", r.status),
        });
    }
    Ok(r.body)
}

fn get_text(http: &dyn Http, url: &str) -> Option<String> {
    get(http, url, Duration::from_secs(30))
        .ok()
        .map(|b| String::from_utf8_lossy(&b).into_owned())
}

fn save(http: &dyn Http, url: &str, dest: &Path) -> bool {
    match get(http, url, Duration::from_secs(900)) {
        Ok(b) => std::fs::write(dest, b).is_ok(),
        Err(_) => {
            let _ = std::fs::remove_file(dest);
            false
        }
    }
}

/// The run-level files `fetch_results` takes.
pub const RUN_FILES: [&str; 10] = [
    "box.txt",
    "tree.txt",
    "weights.log",
    "matrix.out",
    "live.log",
    "kernels.out",
    "kernels.json",
    "lpips-fetch.log",
    "sysinfo.txt",
    "sampler.log",
];

/// `runpod-http.sh fetch_results`: the run files, then per directory of the
/// run its `summary.json`, `stderr.log`, `stdout.log`, `gpucheck-out/*.json`
/// and every JSON directly in it (benchmark.json, `frames.sha256.json`,
/// compare/, gate/), plus `DONE`. The
/// listing is retried (one failed read once lost a whole run). Returns the
/// directories fetched.
pub fn fetch_run(
    http: &dyn Http,
    base: &str,
    family: &str,
    tag: &str,
    dest: &Path,
    clock: &dyn Clock,
    log: &dyn LogSink,
) -> Result<Vec<String>, PodError> {
    let ioe = |p: &Path, e: std::io::Error| PodError::Io {
        path: p.to_path_buf(),
        why: e.to_string(),
    };
    std::fs::create_dir_all(dest).map_err(|e| ioe(dest, e))?;
    let run = format!("{base}/{family}/{tag}");
    for f in RUN_FILES {
        save(http, &format!("{run}/{f}"), &dest.join(f));
    }
    // The finish marker, which `judge_run` reads (runpod-http.sh only polls it).
    save(http, &format!("{run}/DONE"), &dest.join("DONE"));
    let mut cells = Vec::new();
    for attempt in 1..=5 {
        if let Some(l) = get_text(http, &format!("{run}/")) {
            cells = hrefs(&l)
                .into_iter()
                .filter_map(|h| h.strip_suffix('/').map(str::to_owned))
                .filter(|h| !h.contains('/') && !h.is_empty() && h != "..")
                .collect();
        }
        if !cells.is_empty() {
            break;
        }
        log.line(&format!(
            "[fv-lab] cell listing empty (attempt {attempt}); retrying"
        ));
        clock.sleep(Duration::from_secs(10));
    }
    for cell in &cells {
        let d = dest.join(cell);
        std::fs::create_dir_all(&d).map_err(|e| ioe(&d, e))?;
        for f in ["summary.json", "stderr.log", "stdout.log"] {
            save(http, &format!("{run}/{cell}/{f}"), &d.join(f));
        }
        if let Some(l) = get_text(http, &format!("{run}/{cell}/gpucheck-out/")) {
            for f in hrefs(&l)
                .into_iter()
                .filter(|h| h.ends_with(".json") && !h.contains('/'))
            {
                let g = d.join("gpucheck-out");
                std::fs::create_dir_all(&g).map_err(|e| ioe(&g, e))?;
                save(http, &format!("{run}/{cell}/gpucheck-out/{f}"), &g.join(&f));
            }
        }
        if let Some(l) = get_text(http, &format!("{run}/{cell}/")) {
            for f in hrefs(&l)
                .into_iter()
                .filter(|h| h.ends_with(".json") && !h.contains('/'))
            {
                save(http, &format!("{run}/{cell}/{f}"), &d.join(&f));
            }
        }
    }
    log.line(&format!("[fv-lab] results -> {}", dest.display()));
    Ok(cells)
}

/// Why [`wait_done`] stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Done {
    /// `DONE` says `done <tag>`.
    Finished,
    /// The pod is gone or stopped before finishing.
    Gone(String),
    /// The wall-clock limit passed.
    Timeout,
    /// Interrupted.
    Interrupted,
}

/// `runpod-http.sh wait_done`: poll `DONE` (it carries the tag, so a proxy
/// error page never reads as finished), print new `live.log` lines, stop
/// when the pod is gone or `limit_s` passed.
#[allow(clippy::too_many_arguments)]
pub fn wait_done(
    http: &dyn Http,
    provider: &dyn Provider,
    id: &InstanceId,
    base: &str,
    family: &str,
    tag: &str,
    limit_s: u64,
    clock: &dyn Clock,
    log: &dyn LogSink,
    interrupted: &dyn Fn() -> bool,
) -> Done {
    let t0 = clock.now();
    let run = format!("{base}/{family}/{tag}");
    let mut last = String::new();
    loop {
        if interrupted() {
            return Done::Interrupted;
        }
        if get_text(http, &format!("{run}/DONE"))
            .is_some_and(|t| t.lines().next() == Some(format!("done {tag}").as_str()))
        {
            return Done::Finished;
        }
        if let Ok(s @ (InstanceState::Terminated | InstanceState::Exited)) = provider.status(id) {
            return Done::Gone(format!("{id} is {s:?} before finishing"));
        }
        if let Some(cur) = get_text(http, &format!("{run}/live.log"))
            .and_then(|t| t.lines().last().map(str::to_owned))
        {
            if !cur.is_empty() && cur != last {
                log.line(&format!("[fv-lab] pod: {cur}"));
                last = cur;
            }
        }
        if clock.now() - t0 >= limit_s as f64 {
            return Done::Timeout;
        }
        clock.sleep(Duration::from_secs(30));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_baseline_commit_of_a_pod_fails_closed() {
        let mut env = vec!["FASTVIDEO_TEST_X=1".to_owned()];
        assert_eq!(
            baseline_for_pod("1234567", Some("7654321"), &mut env, true).unwrap(),
            Some("7654321".to_owned())
        );
        assert_eq!(env, ["FASTVIDEO_TEST_X=1", "FV_BASELINE_SHA=7654321"]);
        let mut via_env = vec!["FV_BASELINE_SHA=7654321".to_owned()];
        assert!(baseline_for_pod("1234567", None, &mut via_env, true).is_ok());
        assert_eq!(via_env.len(), 1);
        let err = |sha: &str, flag: Option<&str>, env: &[&str], needed: bool| {
            let mut e: Vec<String> = env.iter().map(|s| (*s).to_owned()).collect();
            baseline_for_pod(sha, flag, &mut e, needed)
                .unwrap_err()
                .to_string()
        };
        assert!(err("1234567", Some("1234567"), &[], true).contains("own commit"));
        assert!(err("1234567", None, &["FV_BASELINE_SHA=1234567"], true).contains("own commit"));
        assert!(err("1234567", None, &["FV_BASELINE_SHA="], true)
            .contains("--extra-env FV_BASELINE_SHA ``"));
        assert!(err("1234567", None, &["FV_BASELINE_SHA=main"], false).contains("7 lower-case hex"));
        assert!(err("1234567", Some("xyz"), &[], true).contains("--baseline-sha `xyz`"));
        assert!(err(
            "1234567",
            Some("7654321"),
            &["FV_BASELINE_SHA=7654321"],
            true
        )
        .contains("twice"));
        assert!(err("1234567", None, &[], true).contains("--baseline-sha <7>"));
        assert_eq!(
            baseline_for_pod("1234567", None, &mut Vec::new(), false).unwrap(),
            None
        );
    }
    use crate::plan::Tiers;

    const TIERS: &str = r#"
schema = 1
[[tier]]
name = "rtx6000"
description = "d"
provider = "runpod-pod"
gpu = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
gpu_count = 1
region = "eu"
image = "runtime"
max_usd_per_hr = 2.5
wall_cap_s = 14400
cell_timeout_s = 3600
container_disk_gb = 120
weights = "volume"
preflight = ["sol-h3"]
tae = true
cells = ["c"]
"#;

    fn tier() -> Tier {
        Tiers::parse(TIERS, "t").unwrap().tiers.remove(0)
    }

    #[test]
    fn the_launch_passes_fv_deploys_guards() {
        let t = tier();
        let o = StartOptions::for_tier(&t, "abcdef1-10241200");
        let l = launch(&t, "abcdef1", &o).unwrap();
        assert_eq!(l.name, "fv-lab-rtx6000-abcdef1-10241200");
        assert_eq!(
            l.image,
            "ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-abcdef1"
        );
        assert_eq!(guard::check_guards(&l.guards), Ok(()));
        assert_eq!(guard::check_region(&l), Ok(()));
        assert!(l.args[2]
            .contains("/opt/fastvideo-rs/target/release/fv-lab run rtx6000 >\"$OUT/matrix.out\""));
        assert!(l.args[2].contains("verify-weights.sh sol-h3 >"));
        assert!(launch(&t, "ABCDEF1", &o).is_err());
        assert!(launch(&t, "abcdef", &o).is_err());
    }

    #[test]
    fn start_options_refuse_shell_text() {
        let t = tier();
        let mut o = StartOptions::for_tier(&t, "abcdef1-1");
        o.extra_env = vec!["FASTVIDEO_TEST_X=1".into()];
        assert!(start_command(&o)
            .unwrap()
            .contains("env FASTVIDEO_TEST_X=1 FV_WORK=/workspace"));
        for bad in ["X=$(id)", "x=1", "X=a b", "X"] {
            o.extra_env = vec![bad.into()];
            assert!(start_command(&o).is_err(), "{bad}");
        }
        o.extra_env.clear();
        o.cells = vec!["a\"b".into()];
        assert!(start_command(&o).is_err());
        o.cells.clear();
        o.idle_min = 0;
        assert!(start_command(&o).unwrap().starts_with("set -u\n"));
    }

    #[test]
    fn every_pod_deletes_itself_at_its_wall_cap() {
        let t = tier();
        let mut o = StartOptions::for_tier(&t, "abcdef1-1");
        assert_eq!(o.cap_s, t.wall_cap_s);
        // The self-delete is GraphQL podTerminate with the pod-scoped key: Runpod
        // refuses that key a REST DELETE (HTTP 403, ADR-0023 §4). Success is an
        // HTTP 200 without "errors"; anything else is retried.
        let graphql = r#"-X POST https://api.runpod.io/graphql"#;
        let mutation = r#"--data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}""#;
        let success = r#"[ "$code" = 200 ] && ! grep -q '"errors"' "$1.resp""#;
        for idle in [10, 0] {
            o.idle_min = idle;
            let s = start_command(&o).unwrap();
            let cap = format!("    sleep {}\n    while :; do", t.wall_cap_s);
            let at = s.find(&cap).expect("the cap watchdog");
            // Started before anything that can block (the server, the run).
            assert!(at < s.find("fv-gpucheck serve").unwrap());
            assert!(
                s.contains("      fv_self_delete /tmp/cap-watchdog.log && break\n      sleep 60\n")
            );
            // One definition per watchdog (the idle one only when it runs).
            let defs = if idle == 0 { 1 } else { 2 };
            assert_eq!(s.matches(graphql).count(), defs, "idle {idle}");
            assert_eq!(s.matches(mutation).count(), defs, "idle {idle}");
            assert_eq!(s.matches(success).count(), defs, "idle {idle}");
            if idle != 0 {
                assert!(s.contains("      fv_self_delete /tmp/idle-watchdog.log\n"));
            }
            // No REST delete anywhere (it never worked with the pod's key).
            assert!(!s.contains("rest.runpod.io"));
            assert!(!s.contains("-X DELETE"));
            assert!(!s.contains("@CAP_S@"));
        }
        o.cap_s = 0;
        assert!(start_command(&o).is_err());
    }

    #[test]
    fn cells_must_be_arms_of_the_tier() {
        let arms = ["sol-h3", "ltx25-distill-sol"];
        assert!(check_cells("gold", arms, &["sol-h3".into()]).is_ok());
        assert!(check_cells("gold", arms, &[]).is_ok());
        for bad in ["--state-dir", "x", "sol-h3 "] {
            assert!(check_cells("gold", arms, &[bad.into()]).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_ledger_lists_pods_created_and_not_deleted() {
        let ledger = "2026-10-24T21:00:00Z\trunpod-pod-created aaa fv-lab-gold-x gpu=G image=I\n\
            2026-10-24T21:00:01Z\trunpod-pod-created bbb fv-lab-gold-y gpu=G image=I\n\
            2026-10-24T21:00:02Z\trunpod-pod-delete-failed bbb over-cap\n\
            2026-10-24T21:00:03Z\trunpod-pod-created ccc fv-lab-gold-z gpu=G image=I\n\
            2026-10-24T21:00:04Z\trunpod-pod-deleted aaa down\n\
            2026-10-24T21:00:05Z\trunpod-serverless-created ddd ep gpu=G image=I\n\
            garbage\n";
        assert_eq!(undeleted_in_ledger(ledger), ["bbb", "ccc"]);
        assert!(undeleted_in_ledger("").is_empty());
        // Created again after a delete (an id is never reused, but the order holds).
        let again =
            "t\trunpod-pod-created a n\nt\trunpod-pod-deleted a x\nt\trunpod-pod-created a n\n";
        assert_eq!(undeleted_in_ledger(again), ["a"]);
    }

    #[test]
    fn tags_are_unique_per_second_and_suffix() {
        // 2026-10-24 12:05:09 UTC.
        let t = 1_792_843_509.0;
        assert_eq!(run_tag("abcdef1", t, 0x00af), "abcdef1-1024120509-00af");
        assert_ne!(run_tag("abcdef1", t, 1), run_tag("abcdef1", t, 2));
        assert_ne!(run_tag("abcdef1", t, 1), run_tag("abcdef1", t + 1.0, 1));
        let n = pod_name("sol-h3x", &run_tag("abcdef1", t, 0xbeef));
        assert_eq!(
            parse_pod_name(&n),
            Some(PodName {
                tier: "sol-h3x",
                sha: "abcdef1",
                stamp: "1024120509",
                suffix: "beef"
            })
        );
        // runpod-http.sh's names (FV_POD_PREFIX=fv-lab) and others never parse.
        for other in [
            "fv-lab-rtx6000-abcdef1-10241205",
            "fv-rtx6000-abcdef1-1024120509-beef",
            "fv-lab-rtx6000-ABCDEF1-1024120509-beef",
            "fv-lab-rtx6000-abcdef1-1024120509-xyz1",
            "fv-lab--abcdef1-1024120509-beef",
        ] {
            assert_eq!(parse_pod_name(other), None, "{other}");
        }
        assert!(random_suffix().is_ok());
    }

    fn listing(id: &str, name: &str, created: Option<f64>, state: InstanceState) -> PodListing {
        PodListing {
            id: InstanceId::new(id),
            name: name.into(),
            state,
            created_at: created,
            usd_per_hr: None,
        }
    }

    #[test]
    fn a_live_pod_with_the_name_blocks_a_create() {
        let pods = vec![
            listing("a", "fv-lab-x", Some(0.0), InstanceState::Running),
            listing("b", "fv-lab-y", Some(0.0), InstanceState::Terminated),
        ];
        assert!(name_taken(&pods, "fv-lab-x"));
        assert!(!name_taken(&pods, "fv-lab-y"));
        assert!(!name_taken(&pods, "fv-lab-z"));
    }

    fn record(id: &str, cap_s: u64, created: Option<f64>) -> PodRecord {
        PodRecord {
            id: id.into(),
            name: format!("fv-lab-rtx6000-{id}"),
            family: "rtx6000".into(),
            tag: "t".into(),
            sha: "abcdef1".into(),
            image: "i".into(),
            url: "u".into(),
            gpu: None,
            usd_per_hr: None,
            created_at: created,
            cap_s,
        }
    }

    #[test]
    fn reap_by_default_takes_only_recorded_pods_past_their_cap() {
        let now = 100_000.0;
        let old = Some(now - 14_400.0 - 601.0);
        let n = |i: u16| pod_name("rtx6000", &format!("abcdef1-1024120509-{i:04x}"));
        let pods = vec![
            listing("mine-old", &n(1), old, InstanceState::Running),
            listing(
                "mine-young",
                &n(2),
                Some(now - 14_400.0),
                InstanceState::Running,
            ),
            listing("theirs-old", &n(3), old, InstanceState::Running),
            listing("mine-no-time", &n(4), None, InstanceState::Running),
            listing("mine-gone", &n(5), old, InstanceState::Terminated),
        ];
        let records = vec![
            record("mine-old", 14_400, None),
            record("mine-young", 14_400, None),
            record("mine-no-time", 14_400, old),
            record("mine-gone", 14_400, None),
        ];
        let r: Vec<&str> = reap_recorded(&pods, now, &records)
            .iter()
            .map(|p| p.id.as_str())
            .collect();
        // Another session's pod is never taken; the record's own creation time
        // stands in for a missing createdAt.
        assert_eq!(r, ["mine-old", "mine-no-time"]);
    }

    #[test]
    fn reap_account_needs_the_full_pattern_of_a_known_tier() {
        let now = 100_000.0;
        let old = Some(now - 20_000.0);
        let pods = vec![
            listing(
                "a",
                "fv-lab-rtx6000-abcdef1-1024120509-0001",
                old,
                InstanceState::Running,
            ),
            // runpod-http.sh with FV_POD_PREFIX=fv-lab: it may carry a cap over 4 h.
            listing(
                "b",
                "fv-lab-rtx6000-abcdef1-10241205",
                old,
                InstanceState::Running,
            ),
            listing(
                "c",
                "fv-lab-unknown-abcdef1-1024120509-0001",
                old,
                InstanceState::Running,
            ),
            listing("d", "fv-serve-x", old, InstanceState::Running),
            listing(
                "e",
                "fv-lab-rtx6000-abcdef1-1024120509-0002",
                None,
                InstanceState::Running,
            ),
            listing(
                "f",
                "fv-lab-rtx6000-abcdef1-1024120509-0003",
                Some(now - 100.0),
                InstanceState::Running,
            ),
        ];
        let r: Vec<&str> = reap_account(&pods, now, &[("rtx6000", 14_400)])
            .iter()
            .map(|p| p.id.as_str())
            .collect();
        assert_eq!(r, ["a"]);
    }

    #[test]
    fn a_run_is_good_only_with_done_and_matrix_exit_zero() {
        let d = std::env::temp_dir().join(format!("fv-lab-judge-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let w = |f: &str, t: &str| std::fs::write(d.join(f), t).unwrap();
        assert_eq!(judge_run(&d, "t1").len(), 2, "nothing fetched");
        w("DONE", "done t1\n");
        assert!(judge_run(&d, "t1")[0].contains("no matrix.out"));
        w("matrix.out", "log\n");
        assert!(judge_run(&d, "t1")[0].contains("no matrix_exit"));
        w("matrix.out", "log\nmatrix_exit=2\n");
        assert!(judge_run(&d, "t1")[0].contains("abnormally"));
        w("matrix.out", "log\nmatrix_exit=0\n");
        assert!(judge_run(&d, "t1").is_empty());
        w("DONE", "done other\n");
        assert!(judge_run(&d, "t1")[0].contains("not `done t1`"));
        w("DONE", "done t1\n");
        w("box.txt", "x\ndriver_too_old=570\n");
        assert!(judge_run(&d, "t1")[0].contains("driver too old"));
        let _ = std::fs::remove_dir_all(&d);
    }

    struct Files(std::collections::BTreeMap<String, String>);

    impl Http for Files {
        fn send(
            &self,
            req: &Request,
        ) -> Result<fastvideo_deploy::boot::sys::Response, fastvideo_deploy::boot::sys::SysError>
        {
            Ok(match self.0.get(&req.url) {
                Some(b) => fastvideo_deploy::boot::sys::Response {
                    status: 200,
                    body: b.clone().into_bytes(),
                },
                None => fastvideo_deploy::boot::sys::Response {
                    status: 404,
                    body: Vec::new(),
                },
            })
        }
        fn download(
            &self,
            url: &str,
            _: &Path,
            _: Duration,
        ) -> Result<u64, fastvideo_deploy::boot::sys::SysError> {
            Err(fastvideo_deploy::boot::sys::SysError::Http {
                url: url.into(),
                why: "no".into(),
            })
        }
    }

    struct Still;
    impl Clock for Still {
        fn now(&self) -> f64 {
            0.0
        }
        fn sleep(&self, _: Duration) {}
    }
    struct Quiet;
    impl LogSink for Quiet {
        fn line(&self, _: &str) {}
    }

    #[test]
    fn fetch_takes_done_summaries_and_the_frames_digest_but_not_frames() {
        let run = "https://p/rtx6000/t1";
        let files: std::collections::BTreeMap<String, String> = [
            (format!("{run}/DONE"), "done t1\n"),
            (format!("{run}/matrix.out"), "matrix_exit=0\n"),
            (format!("{run}/"), "<a href=\"a/\">a/</a><a href=\"live.log\">l</a>"),
            (
                format!("{run}/a/"),
                "<a href=\"summary.json\">s</a><a href=\"frames.sha256.json\">f</a><a href=\"benchmark.json\">b</a><a href=\"frames/\">d</a>",
            ),
            (format!("{run}/a/summary.json"), "{\"exit\":0}"),
            (format!("{run}/a/frames.sha256.json"), "{\"arm\":\"a\"}"),
            (format!("{run}/a/benchmark.json"), "{}"),
            (format!("{run}/a/frames/frame-000.png"), "png"),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_owned()))
        .collect();
        let dest = std::env::temp_dir().join(format!("fv-lab-fetch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dest);
        let cells = fetch_run(
            &Files(files),
            "https://p",
            "rtx6000",
            "t1",
            &dest,
            &Still,
            &Quiet,
        )
        .unwrap();
        assert_eq!(cells, ["a"]);
        for f in [
            "DONE",
            "matrix.out",
            "a/summary.json",
            "a/frames.sha256.json",
            "a/benchmark.json",
        ] {
            assert!(dest.join(f).is_file(), "{f}");
        }
        assert!(!dest.join("a/frames").exists());
        assert!(judge_run(&dest, "t1").is_empty());
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn hrefs_of_a_listing() {
        let l = "<li><a href=\"a/\">a/</a></li><li><a href=\"b.json\">b</a></li>";
        assert_eq!(hrefs(l), ["a/", "b.json"]);
    }
}
