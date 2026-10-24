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
//! - [`reap_candidates`]: which `fv-lab-*` pods are past their cap.

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
/// The cap `reap` assumes for an fv-lab pod whose tier it does not know
/// (fv-deploy's 4 h bound, the longest any tier may ask for).
pub const MAX_REAP_CAP_S: u64 = guard::MAX_CAP_S;

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
        }
    }

    fn check(&self) -> Result<(), PodError> {
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
    ] {
        s = s.replace(k, &v);
    }
    Ok(s)
}

/// A 7-hex-digit commit.
pub fn check_sha7(sha: &str) -> Result<(), PodError> {
    if sha.len() == 7
        && sha
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(PodError::Invalid(format!(
            "--sha `{sha}`: 7 lower-case hex digits"
        )))
    }
}

/// `runpod-http.sh`'s run tag: `<sha>-<MMDDHHMM>` (UTC).
pub fn run_tag(sha7: &str, unix: f64) -> String {
    let (_, m, d, h, mi, _) = utc_parts(unix);
    format!("{sha7}-{m:02}{d:02}{h:02}{mi:02}")
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

/// The `fv-lab-*` pods older than their cap plus [`REAP_GRACE_S`] (`cap_of`
/// maps a pod name to its tier's wall cap). A pod without a readable
/// creation time is never reaped (it is listed by `reap` instead).
pub fn reap_candidates(
    pods: &[PodListing],
    now: f64,
    cap_of: impl Fn(&str) -> u64,
) -> Vec<&PodListing> {
    pods.iter()
        .filter(|p| p.name.starts_with(POD_PREFIX))
        .filter(|p| p.state != InstanceState::Terminated)
        .filter(|p| {
            p.created_at
                .is_some_and(|t| now - t > (cap_of(&p.name) + REAP_GRACE_S) as f64)
        })
        .collect()
}

/// The tier of an fv-lab pod name (`fv-lab-<family>-<sha7>-<MMDDHHMM>`).
pub fn family_of(name: &str) -> Option<&str> {
    let rest = name.strip_prefix(POD_PREFIX)?;
    let mut parts = rest.rsplitn(3, '-');
    let _stamp = parts.next()?;
    let _sha = parts.next()?;
    parts.next()
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
/// and every JSON directly in it (benchmark.json, compare/, gate/). The
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
        o.extra_env = vec!["FASTVIDEO_X=1".into()];
        assert!(start_command(&o)
            .unwrap()
            .contains("env FASTVIDEO_X=1 FV_WORK=/workspace"));
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
    fn tags_names_and_families() {
        // 2026-10-24 12:05:09 UTC.
        assert_eq!(run_tag("abcdef1", 1_792_843_509.0), "abcdef1-10241205");
        assert_eq!(
            family_of("fv-lab-rtx6000-abcdef1-10241205"),
            Some("rtx6000")
        );
        assert_eq!(
            family_of("fv-lab-sol-h3x-abcdef1-10241205"),
            Some("sol-h3x")
        );
        assert_eq!(family_of("fv-rtx6000-x"), None);
    }

    #[test]
    fn reap_takes_only_old_fv_lab_pods() {
        let p = |name: &str, created: Option<f64>, state| PodListing {
            id: InstanceId::new(name),
            name: name.into(),
            state,
            created_at: created,
            usd_per_hr: None,
        };
        let now = 100_000.0;
        let pods = vec![
            p(
                "fv-lab-rtx6000-a-1",
                Some(now - 14_400.0 - 601.0),
                InstanceState::Running,
            ),
            p(
                "fv-lab-rtx6000-b-1",
                Some(now - 14_400.0),
                InstanceState::Running,
            ),
            p("fv-lab-rtx6000-c-1", None, InstanceState::Running),
            p("fv-lab-rtx6000-d-1", Some(0.0), InstanceState::Terminated),
            p("fv-serve-x", Some(0.0), InstanceState::Running),
        ];
        let r: Vec<&str> = reap_candidates(&pods, now, |_| 14_400)
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(r, ["fv-lab-rtx6000-a-1"]);
    }

    #[test]
    fn hrefs_of_a_listing() {
        let l = "<li><a href=\"a/\">a/</a></li><li><a href=\"b.json\">b</a></li>";
        assert_eq!(hrefs(l), ["a/", "b.json"]);
    }
}
