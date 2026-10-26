//! `fv-lab oracle` (docs/adr/0019-fv-lab.md §19): the driver of an oracle
//! run, what `scripts/gpu/oracle.sh` and §15.7's hand-run commands do, on
//! fv-deploy's guards.
//!
//! - [`Stacks`]: `lab/upstream.toml`, the upstream reference stacks (one
//!   pod per upstream image) and, per stack, the oracle targets it makes
//!   and the weight steps each needs. Every target is the `[reference]
//!   target` of one cell ([`Stacks::check_against`]).
//! - [`upstream_start_command`] / [`upstream_launch`]: the upstream pod,
//!   `runpod-http.sh upstream`'s (`src/upstream-start.sh.in`, generated from
//!   the script's `upstream_start_cmd` with its variables as markers) plus
//!   the pod's own wall-cap watchdog that every fv-lab pod carries; the
//!   payload golden (`tests/oracle_driver.rs`) holds the rest equal to
//!   `runpod-http.sh plan-upstream`.
//! - [`plan_oracle`]: the run's plan from the selected arms, with every
//!   refusal made before anything is created.
//! - [`drive`]: the orchestration on an [`OracleOps`] (the binary's real
//!   one, the tests' fake): the upstream pods up, the references waited for,
//!   the runtime tier's `fv-lab pod up --wait` with `FV_ORACLE_URL`, the
//!   keepalive, and the release of the upstream pods on `ORACLE_REFS_DONE`,
//!   on the stock wait, and on every failure.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use fastvideo_deploy::guard;
use fastvideo_deploy::launch::POD_VOLUME_MOUNT;
use fastvideo_deploy::provider::{
    CloudType, Disk, GpuSpec, Guards, Launch, Placement, ProviderKind, VolumeMount, WeightsSource,
    LAUNCH_SCHEMA,
};
use fastvideo_manifest::deploy::{EU_DATA_CENTER, EU_VOLUME_ID};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::plan::{is_name, Lab, LabError, Tier, MAX_WALL_CAP_S};
use crate::pod::{check_sha7, pod_name};

/// The only `lab/upstream.toml` schema this crate reads.
pub const UPSTREAM_SCHEMA: u32 = 1;
/// Where the stacks live, relative to the repository (or image) root.
pub const UPSTREAM_PATH: &str = "lab/upstream.toml";
/// `image = "<target>"` boots `<repo>-<target>:<tag>` (docker/upstream.Dockerfile,
/// .github/workflows/upstream-images.yml).
pub const UPSTREAM_IMAGE_REPO: &str = "ghcr.io/zaitrarrio/fastvideo-rs-upstream";
/// The upstream pods' family: their run directory `/root/runs/upstream/<tag>`
/// (served as `/upstream/<tag>`), their pod name `fv-lab-upstream-<tag>` and
/// `artifacts/runpod/upstream/<tag>`, as `runpod-http.sh upstream`'s.
pub const UPSTREAM_FAMILY: &str = "upstream";
/// What `runpod-http.sh` fetches of an upstream run: everything but the
/// dumps and clips (`oracle.sh`'s `FV_FETCH_SKIP`).
pub const UPSTREAM_FETCH_SKIP: &str = r"oracle-dump\.tar|/dump/|\.mp4$";
/// The tier `fv-lab oracle` runs by default.
pub const ORACLE_TIER: &str = "oracle";
/// `scripts/gpu/oracle.sh`'s `FV_RUNTIME_STOCK_WAIT_S` is 900 s; fv-lab's is
/// fv-deploy's whole readiness budget (3 cycles of 600 s) and 5 minutes more
/// (a pod ready late in its third cycle, then recorded), since a create
/// without stock fails at once (`no capacity`) and the wait only bounds the
/// runtime pod's readiness.
pub const STOCK_WAIT_S: u64 = guard::READY_TIMEOUT_S * guard::READY_CYCLES as u64 + 300;
/// What the upstream pods keep for the runtime pod's reference fetch after
/// its readiness (s): the wait for the references leaves this and the stock
/// wait before the upstream pods' cap; and how long after the runtime pod is
/// ready the driver waits for its `ORACLE_REFS_DONE` before it releases the
/// upstream pods all the same (the runner's preflight, about 5 min, then the
/// downloads: a 1.3 GB H3 reference took under a minute in §15.7's run;
/// the rest is margin for the larger LTX dumps and a slow proxy).
pub const REF_FETCH_RESERVE_S: u64 = 1800;
/// The driver's poll period (s): `oracle.sh`'s keepalive and release loop.
pub const POLL_S: u64 = 60;
const UPSTREAM_START_TEMPLATE: &str = include_str!("upstream-start.sh.in");
/// A `hub_token` stack's pod env: `HF_TOKEN` (what `scripts/gpu/upstream/pod.sh`
/// and `reconstruct.py` read) filled by Runpod from the owner's secret
/// `fv_hf_token` (docs/ops/runpod-secrets.md), as `runpod-http.sh upstream`
/// passes it for the LTX weight steps.
pub const HUB_TOKEN_ENV: (&str, &str) = ("HF_TOKEN", "fv_hf_token");

/// A `pod.sh` step that reads a gated Hub repository: `weights:ltx25*`, for
/// which `runpod-http.sh upstream` passes the token (`upstream_hub_token`);
/// a stack with one must set `hub_token`, so the two paths cannot drift.
pub fn hub_step(step: &str) -> bool {
    step.starts_with("weights:ltx25")
}

/// An env name that would put a secret's value in the repository.
pub fn secret_like(key: &str) -> bool {
    ["TOKEN", "SECRET", "KEY"].iter().any(|w| key.contains(w))
}

/// `{{ RUNPOD_SECRET_fv_hf_token }}`: the reference, never the token.
pub fn hub_token_ref() -> String {
    format!("{{{{ RUNPOD_SECRET_{} }}}}", HUB_TOKEN_ENV.1)
}

// ------------------------------------------------------------------ stacks

/// `lab/upstream.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Stacks {
    /// Always 1.
    pub schema: u32,
    /// The upstream stacks, in the order their pods come up.
    #[serde(rename = "stack")]
    pub stacks: Vec<Stack>,
}

/// One upstream image and the pod that runs its references.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Stack {
    pub name: String,
    pub description: String,
    /// The docker/upstream.Dockerfile target: the image
    /// `ghcr.io/zaitrarrio/fastvideo-rs-upstream-<image>:<tag>`.
    pub image: String,
    pub gpu: String,
    pub gpu_count: u32,
    #[serde(default)]
    pub cloud: CloudType,
    /// `eu` only (the EU weights volume, CLAUDE.md).
    pub region: String,
    pub max_usd_per_hr: f64,
    /// The backstop's deadline and the pod's own cap (s), at most 4 h.
    pub wall_cap_s: u64,
    pub container_disk_gb: u32,
    /// The pod's host-RAM floor, whole GB per GPU (Runpod `minRAMPerGPU`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub min_ram_gb: Option<u32>,
    /// `UP_CELL_TIMEOUT_S`: each reference's cap on the pod (s).
    pub cell_timeout_s: u64,
    /// `FV_EXTRA_ENV` of the pod: `K=V` for every reference.
    #[serde(default)]
    pub env: Vec<String>,
    /// The pod's `HF_TOKEN` as the Runpod secret reference
    /// `{{ RUNPOD_SECRET_fv_hf_token }}` (the owner's secret), never a value:
    /// for a stack whose weight steps read gated Hub repositories (sol-ltx25:
    /// `Lightricks/LTX-2.5`). Default false: the pod env has no entry.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hub_token: bool,
    /// The oracle targets this stack makes.
    #[serde(rename = "target")]
    pub targets: Vec<StackTarget>,
}

/// One oracle target of a stack: `oracle-<name>` on the pod, the
/// `[reference] target` of one cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StackTarget {
    pub name: String,
    /// The `scripts/gpu/upstream/pod.sh` steps its reference needs before
    /// `oracle:` (`weights:<view>`), in order.
    #[serde(default)]
    pub steps: Vec<String>,
    /// Weight trees its reference reads from the R2 weights store
    /// (`scripts/gpu/weights-r2.tsv` dests; ADR-0019 §21): each a `pod.sh`
    /// step `r2:<tree>` after `steps` (`r2-weights.py plan --pin`, then
    /// `r2-fetch.py --root $UW`, onto the container disk, never the volume).
    /// A selected target with one gives the pod the R2 key as Runpod secret
    /// references (`weights::r2_env_refs`), which `pod.sh` hands to that
    /// step alone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub r2: Vec<String>,
    /// The container disk its reference needs (GB), when more than the
    /// stack's: the pod gets the largest of the stack's and its selected
    /// targets', so a run without this target keeps the stack's disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub container_disk_gb: Option<u32>,
}

/// The `pod.sh` step that fetches an R2 tree (`r2:<tree>`).
pub fn r2_step(tree: &str) -> String {
    format!("r2:{tree}")
}

/// The first line of the start command of a pod that has the R2 key
/// (`weights::r2_env_refs`): the two values copied into the shell's own
/// variables and removed from its environment before anything starts, so
/// the watchdogs, the file server and the clone inherit none of it
/// (`runpod-http.sh`'s, ADR-0019 §21.2). What it cannot do: the values stay
/// in the container's initial environment (PID 1's `/proc/<pid>/environ`),
/// readable by root on the pod.
pub const R2_KEY_CAPTURE: &str = "_fvr2_id=\"${FV_WEIGHTS_R2_ACCESS_KEY_ID:-}\" _fvr2_secret=\"${FV_WEIGHTS_R2_SECRET_ACCESS_KEY:-}\"; unset FV_WEIGHTS_R2_ACCESS_KEY_ID FV_WEIGHTS_R2_SECRET_ACCESS_KEY";

/// The env prefix that hands the captured key to the one command that needs
/// it (`pod.sh`, which hands it to its plan step alone).
pub const R2_KEY_HANDOFF: &str =
    "FV_WEIGHTS_R2_ACCESS_KEY_ID=\"$_fvr2_id\" FV_WEIGHTS_R2_SECRET_ACCESS_KEY=\"$_fvr2_secret\" ";

fn sfield(file: &str, f: String, msg: impl Into<String>) -> LabError {
    LabError::Field {
        file: file.to_owned(),
        field: f,
        msg: msg.into(),
    }
}

/// A `K=V` the start command can carry unquoted: an upper-case key, a value of
/// letters, digits and `._-/:,=`.
pub fn plain_env(e: &str) -> bool {
    e.split_once('=').is_some_and(|(k, v)| {
        !k.is_empty()
            && k.bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
            && v.bytes().all(|b| {
                b.is_ascii_alphanumeric()
                    || matches!(b, b'.' | b'_' | b'-' | b'/' | b':' | b',' | b'=')
            })
    })
}

impl Stacks {
    pub fn parse(text: &str, file: &str) -> Result<Stacks, LabError> {
        let s: Stacks = toml::from_str(text).map_err(|e| LabError::Syntax {
            file: file.to_owned(),
            msg: e.to_string(),
        })?;
        s.validate(file)?;
        Ok(s)
    }

    /// `<root>/lab/upstream.toml`, parsed and validated.
    pub fn load(root: &Path) -> Result<Stacks, LabError> {
        let p = root.join(UPSTREAM_PATH);
        let text = std::fs::read_to_string(&p).map_err(|e| LabError::Io {
            path: p.display().to_string(),
            msg: e.to_string(),
        })?;
        Stacks::parse(&text, UPSTREAM_PATH)
    }

    pub fn validate(&self, file: &str) -> Result<(), LabError> {
        if self.schema != UPSTREAM_SCHEMA {
            return Err(LabError::Schema {
                file: file.to_owned(),
                got: self.schema,
                want: UPSTREAM_SCHEMA,
            });
        }
        let mut names = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for (i, s) in self.stacks.iter().enumerate() {
            let at = |f: &str| format!("stack[{i}].{f}");
            if !is_name(&s.name) || !names.insert(s.name.as_str()) {
                return Err(sfield(
                    file,
                    at("name"),
                    format!("`{}`: [a-z0-9][a-z0-9._-]*, once", s.name),
                ));
            }
            if !is_name(&s.image) {
                return Err(sfield(
                    file,
                    at("image"),
                    format!("`{}` is not an image target", s.image),
                ));
            }
            if s.gpu.trim().is_empty() {
                return Err(sfield(file, at("gpu"), "empty"));
            }
            if s.gpu_count != 1 {
                return Err(sfield(
                    file,
                    at("gpu_count"),
                    "the references run on one GPU (1)",
                ));
            }
            if s.region != fastvideo_manifest::deploy::EU_REGION {
                return Err(sfield(
                    file,
                    at("region"),
                    "only `eu` (the EU weights volume, CLAUDE.md)",
                ));
            }
            if !(s.max_usd_per_hr.is_finite() && s.max_usd_per_hr > 0.0) {
                return Err(sfield(file, at("max_usd_per_hr"), "a positive number"));
            }
            if s.wall_cap_s == 0 || s.wall_cap_s > MAX_WALL_CAP_S {
                return Err(sfield(
                    file,
                    at("wall_cap_s"),
                    format!("1..={MAX_WALL_CAP_S} (fv-deploy's 4 h bound)"),
                ));
            }
            if s.container_disk_gb == 0 {
                return Err(sfield(file, at("container_disk_gb"), "positive"));
            }
            if s.min_ram_gb == Some(0) {
                return Err(sfield(
                    file,
                    at("min_ram_gb"),
                    "a positive whole number of GB per GPU (Runpod minRAMPerGPU), or leave it out",
                ));
            }
            if s.cell_timeout_s == 0 {
                return Err(sfield(
                    file,
                    at("cell_timeout_s"),
                    "positive (UP_CELL_TIMEOUT_S)",
                ));
            }
            if let Some(e) = s.env.iter().find(|e| !plain_env(e)) {
                return Err(sfield(
                    file,
                    at("env"),
                    format!("`{e}` is not K=V with a plain value"),
                ));
            }
            if let Some(k) = s
                .env
                .iter()
                .filter_map(|e| e.split_once('=').map(|(k, _)| k))
                .find(|k| secret_like(k))
            {
                return Err(sfield(
                    file,
                    at("env"),
                    format!(
                        "`{k}` looks like a secret: a stack's env is in the repository (the Hub token: `hub_token = true`, a Runpod secret reference)"
                    ),
                ));
            }
            if !s.hub_token {
                if let Some(st) = s
                    .targets
                    .iter()
                    .flat_map(|t| &t.steps)
                    .find(|st| hub_step(st))
                {
                    return Err(sfield(
                        file,
                        at("hub_token"),
                        format!(
                            "`{st}` reads the gated Lightricks/LTX-2.5: set `hub_token = true` (runpod-http.sh upstream passes the token for it too)"
                        ),
                    ));
                }
            }
            if s.targets.is_empty() {
                return Err(sfield(file, at("target"), "empty"));
            }
            for (j, t) in s.targets.iter().enumerate() {
                let tat = |f: &str| format!("stack[{i}].target[{j}].{f}");
                if !is_name(&t.name) || !targets.insert(t.name.as_str()) {
                    return Err(sfield(
                        file,
                        tat("name"),
                        format!("`{}`: a target name, in one stack only", t.name),
                    ));
                }
                if let Some(st) = t
                    .steps
                    .iter()
                    .find(|st| st.strip_prefix("weights:").is_none_or(|v| !is_name(v)))
                {
                    return Err(sfield(
                        file,
                        tat("steps"),
                        format!(
                            "`{st}` is not a `weights:<view>` step of scripts/gpu/upstream/pod.sh"
                        ),
                    ));
                }
                let mut trees = BTreeSet::new();
                if let Some(tree) = t
                    .r2
                    .iter()
                    .find(|tree| !crate::weights::safe_tree(tree) || !trees.insert(tree.as_str()))
                {
                    return Err(sfield(
                        file,
                        tat("r2"),
                        format!("`{tree}`: a weights-r2.tsv tree name ([A-Za-z0-9._/-], relative, no `.` or `..`), once"),
                    ));
                }
                if t.container_disk_gb == Some(0) {
                    return Err(sfield(
                        file,
                        tat("container_disk_gb"),
                        "positive, or leave it out (the stack's)",
                    ));
                }
            }
        }
        Ok(())
    }

    /// The stack that makes `target`, and the target's entry.
    pub fn for_target(&self, target: &str) -> Option<(&Stack, &StackTarget)> {
        self.stacks
            .iter()
            .find_map(|s| s.targets.iter().find(|t| t.name == target).map(|t| (s, t)))
    }

    /// The stacks against the run plan: every stack target is the
    /// `[reference] target` of a cell, and every such cell's target is made
    /// by a stack; every `r2` tree is published (`lab.root`'s
    /// `scripts/gpu/weights-r2.tsv`).
    pub fn check_against(&self, lab: &Lab) -> Result<(), LabError> {
        if self
            .stacks
            .iter()
            .flat_map(|s| &s.targets)
            .any(|t| !t.r2.is_empty())
        {
            let p = lab.root.join(crate::weights::R2_TABLE);
            let table = std::fs::read_to_string(&p)
                .map_err(|e| LabError::Io {
                    path: p.display().to_string(),
                    msg: e.to_string(),
                })
                .and_then(|t| {
                    crate::weights::parse_r2_table(&t).map_err(|m| LabError::Syntax {
                        file: crate::weights::R2_TABLE.to_owned(),
                        msg: m,
                    })
                })?;
            for (i, s) in self.stacks.iter().enumerate() {
                for (j, t) in s.targets.iter().enumerate() {
                    if let Some(tree) = t.r2.iter().find(|tree| !table.contains_key(*tree)) {
                        return Err(sfield(
                            UPSTREAM_PATH,
                            format!("stack[{i}].target[{j}].r2"),
                            format!("`{tree}` is not a dest of {}", crate::weights::R2_TABLE),
                        ));
                    }
                }
            }
        }
        let refs: BTreeSet<&str> = lab
            .cells
            .values()
            .filter_map(|c| c.reference.as_ref().map(|r| r.target.as_str()))
            .collect();
        for (i, s) in self.stacks.iter().enumerate() {
            for (j, t) in s.targets.iter().enumerate() {
                if !refs.contains(t.name.as_str()) {
                    return Err(sfield(
                        UPSTREAM_PATH,
                        format!("stack[{i}].target[{j}].name"),
                        format!(
                            "`{}` is the [reference] target of no cell (lab/cells/)",
                            t.name
                        ),
                    ));
                }
            }
        }
        if let Some(t) = refs.iter().find(|t| self.for_target(t).is_none()) {
            return Err(sfield(
                UPSTREAM_PATH,
                "stack".into(),
                format!("no stack makes the reference `{t}` a cell reads"),
            ));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ the upstream pod

/// What the upstream pod's start command varies (`runpod-http.sh upstream`'s
/// environment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamStart {
    pub tag: String,
    /// The full commit the pod clones for its runner scripts (`UP_SHA`).
    pub up_sha: String,
    /// `UP_STEPS`.
    pub up_steps: String,
    /// `UP_CELL_TIMEOUT_S`.
    pub cell_timeout_s: u64,
    /// `FV_EXTRA_ENV`.
    pub extra_env: Vec<String>,
    /// `FV_IDLE_GPU_MIN` (`net`: a minute the pod sent data counts as busy),
    /// `FV_IDLE_GPU_GRACE_MIN`, `FV_IDLE_NET_BYTES`.
    pub idle_min: u32,
    pub idle_grace_min: u32,
    pub idle_net_bytes: u64,
    /// The pod's own wall cap (s).
    pub cap_s: u64,
}

/// A 40-hex commit (`UP_SHA`, what `git fetch --depth 1 origin <sha>` takes).
pub fn check_full_sha(sha: &str) -> Result<(), String> {
    if sha.len() == 40
        && sha
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(format!("upstream commit `{sha}`: 40 lower-case hex digits"))
    }
}

/// An upstream image tag: `latest` (CI pushes it on main) or `sha-<7 hex>`.
pub fn check_image_tag(tag: &str) -> Result<(), String> {
    match tag.strip_prefix("sha-") {
        None if tag == "latest" => Ok(()),
        Some(s) if check_sha7(s).is_ok() => Ok(()),
        _ => Err(format!(
            "upstream image tag `{tag}`: `latest` or `sha-<7 hex>` (.github/workflows/upstream-images.yml)"
        )),
    }
}

impl UpstreamStart {
    /// Whether `UP_STEPS` has an `r2:<tree>` step: the pod then has the R2
    /// key, which its start command takes out of its environment on its
    /// first line and hands to `pod.sh` alone (`runpod-http.sh`'s
    /// `upstream_r2_key`).
    pub fn has_r2_step(&self) -> bool {
        self.up_steps.split(' ').any(|s| s.starts_with("r2:"))
    }

    /// `runpod-http.sh`'s defaults for a stack and its steps.
    pub fn for_stack(stack: &Stack, tag: &str, up_sha: &str, up_steps: &str) -> Self {
        UpstreamStart {
            tag: tag.to_owned(),
            up_sha: up_sha.to_owned(),
            up_steps: up_steps.to_owned(),
            cell_timeout_s: stack.cell_timeout_s,
            extra_env: stack.env.clone(),
            idle_min: 10,
            idle_grace_min: 30,
            idle_net_bytes: 262_144,
            cap_s: stack.wall_cap_s,
        }
    }

    fn check(&self) -> Result<(), String> {
        if self.cap_s == 0 {
            return Err("an upstream pod without a wall cap (cap_s 0)".into());
        }
        if self.idle_min == 0 {
            return Err("an upstream pod without its idle watchdog (idle_min 0)".into());
        }
        let plain = |s: &str| {
            !s.is_empty()
                && s.bytes().all(|b| {
                    b.is_ascii_alphanumeric()
                        || matches!(b, b'.' | b'_' | b'-' | b':' | b',' | b' ')
                })
        };
        if !is_name(&self.tag) {
            return Err(format!("tag `{}` is not a plain name", self.tag));
        }
        check_full_sha(&self.up_sha)?;
        if !plain(&self.up_steps) {
            return Err(format!(
                "UP_STEPS `{}` is not plain step names",
                self.up_steps
            ));
        }
        if let Some(e) = self.extra_env.iter().find(|e| !plain_env(e)) {
            return Err(format!(
                "FV_EXTRA_ENV entry `{e}` is not K=V with a plain value"
            ));
        }
        Ok(())
    }
}

/// The upstream pod's start command (`dockerStartCmd: ["/bin/bash", "-c", <this>]`).
pub fn upstream_start_command(o: &UpstreamStart) -> Result<String, String> {
    o.check()?;
    let mut s = UPSTREAM_START_TEMPLATE.trim_end_matches('\n').to_owned();
    let (r2_head, r2_env) = if o.has_r2_step() {
        (format!("{R2_KEY_CAPTURE}\n"), R2_KEY_HANDOFF.to_owned())
    } else {
        (String::new(), String::new())
    };
    for (k, v) in [
        ("@R2_HEAD@", r2_head),
        ("@R2_ENV@", r2_env),
        ("@TAG@", o.tag.clone()),
        ("@UP_SHA@", o.up_sha.clone()),
        ("@UP_STEPS@", o.up_steps.clone()),
        ("@UP_CELL_TIMEOUT@", o.cell_timeout_s.to_string()),
        ("@EXTRA_ENV@", o.extra_env.join(" ")),
        ("@IDLE_MIN@", o.idle_min.to_string()),
        ("@GRACE_MIN@", o.idle_grace_min.to_string()),
        ("@NET_BYTES@", o.idle_net_bytes.to_string()),
        ("@CAP_S@", o.cap_s.to_string()),
    ] {
        s = s.replace(k, &v);
    }
    Ok(s)
}

/// `UP_STEPS` for some targets of a stack: `info:box`, the targets' weight
/// steps then their `r2:<tree>` steps (each once, in the stack's target
/// order), `oracle:<targets>`.
pub fn up_steps(stack: &Stack, targets: &[String]) -> String {
    let mut steps = vec!["info:box".to_owned()];
    for t in &stack.targets {
        if targets.contains(&t.name) {
            for s in t
                .steps
                .iter()
                .cloned()
                .chain(t.r2.iter().map(|x| r2_step(x)))
            {
                if !steps.contains(&s) {
                    steps.push(s);
                }
            }
        }
    }
    steps.push(format!("oracle:{}", targets.join(",")));
    steps.join(" ")
}

/// The `Launch` of an upstream pod making `targets` of `stack`: the stack's
/// image at `image_tag`, the EU volume at `/workspace` (the references read
/// `/workspace/weights`), the stack's GPU, RAM floor and caps under
/// fv-deploy's guards, the largest of the stack's and the targets'
/// container disks; with `hub_token`, `HF_TOKEN` as a Runpod secret
/// reference; when a target has an `r2` tree, the R2 key as Runpod secret
/// references (`weights::r2_env_refs`), never values.
pub fn upstream_launch(
    stack: &Stack,
    targets: &[String],
    image_tag: &str,
    start: &UpstreamStart,
) -> Result<Launch, String> {
    check_image_tag(image_tag)?;
    let cmd = upstream_start_command(start)?;
    let selected: Vec<&StackTarget> = stack
        .targets
        .iter()
        .filter(|t| targets.contains(&t.name))
        .collect();
    let mut env = std::collections::BTreeMap::new();
    if stack.hub_token {
        env.insert(HUB_TOKEN_ENV.0.to_owned(), hub_token_ref());
    }
    let r2 = selected.iter().any(|t| !t.r2.is_empty());
    if r2 != start.has_r2_step() {
        return Err(format!(
            "the targets' R2 trees ({}) and UP_STEPS `{}` disagree: the key is passed for r2:<tree> steps only",
            if r2 { "some" } else { "none" },
            start.up_steps
        ));
    }
    if r2 {
        env.extend(crate::weights::r2_env_refs());
    }
    let disk = selected
        .iter()
        .filter_map(|t| t.container_disk_gb)
        .fold(stack.container_disk_gb, u32::max);
    Ok(Launch {
        schema: LAUNCH_SCHEMA,
        name: pod_name(UPSTREAM_FAMILY, &start.tag),
        provider: ProviderKind::RunpodPod,
        source: None,
        image: format!("{UPSTREAM_IMAGE_REPO}-{}:{image_tag}", stack.image),
        entrypoint: Vec::new(),
        args: vec!["/bin/bash".into(), "-c".into(), cmd],
        env,
        ports: vec!["8000/http".into()],
        gpu: GpuSpec {
            types: vec![stack.gpu.clone()],
            count: stack.gpu_count,
            allowed_cuda: Vec::new(),
            min_cuda: None,
            min_ram_gb: stack.min_ram_gb,
        },
        placement: Placement {
            region: Some(stack.region.clone()),
            data_centers: vec![EU_DATA_CENTER.into()],
            volume: Some(VolumeMount {
                id: EU_VOLUME_ID.into(),
                mount_path: POD_VOLUME_MOUNT.into(),
            }),
            cloud: stack.cloud,
        },
        disk: Disk {
            container_gb: disk,
            volume_gb: 0,
        },
        weights: WeightsSource::Volume,
        guards: Guards::all(
            stack.max_usd_per_hr,
            guard::MIN_BALANCE_USD,
            start.cap_s,
            guard::READY_TIMEOUT_S,
            guard::READY_CYCLES,
        ),
        serverless: None,
    })
}

// ------------------------------------------------------------------ the plan

/// What `fv-lab oracle` was asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleRequest {
    /// The runtime tier (`oracle`).
    pub tier: String,
    /// The runtime image's commit (7 hex).
    pub sha: String,
    /// The commit the upstream pods clone (40 hex).
    pub up_sha: String,
    /// `latest` or `sha-<7>`.
    pub image_tag: String,
    /// The arms to run (`--cells`).
    pub cells: Vec<String>,
    /// More `K=V` for the runtime run (`--extra-env`; not `FV_ORACLE_URL`).
    pub extra_env: Vec<String>,
    /// The runtime pod's tag, and each stack's upstream pod's (in stack order).
    pub runtime_tag: String,
    pub upstream_tags: Vec<String>,
    /// Lower caps (s): the runtime pod's (`--wall-cap-s`), the upstream pods'.
    pub runtime_cap_s: Option<u64>,
    pub upstream_cap_s: Option<u64>,
    /// How long the runtime pod may take to be ready (s).
    pub stock_wait_s: u64,
    /// How long the references may take, counted from the first upstream
    /// create (s); `None`: what the upstream cap leaves
    /// ([`REF_FETCH_RESERVE_S`], the stock wait).
    pub refs_wait_s: Option<u64>,
    /// `--hold-upstream`: the upstream pods stay (after the runtime pod has
    /// every reference, or after the runtime side succeeded) until the caller
    /// lifts the hold, for another reader of the same references (the GPU
    /// check's script side, §19.7); the stock wait, a failed start, a failed
    /// runtime side and an interrupt release at once.
    pub hold: bool,
}

/// One upstream pod of the plan.
#[derive(Debug, Clone, PartialEq)]
pub struct UpstreamPlan {
    pub stack: String,
    pub targets: Vec<String>,
    pub start: UpstreamStart,
    pub launch: Launch,
}

/// The run `fv-lab oracle` makes.
#[derive(Debug, Clone, PartialEq)]
pub struct OraclePlan {
    pub tier: String,
    pub sha: String,
    pub runtime_tag: String,
    /// The selected arms, in the tier's order.
    pub cells: Vec<String>,
    /// Each selected target and its selected arms, in the tier's order.
    pub arms_by_target: Vec<(String, Vec<String>)>,
    pub extra_env: Vec<String>,
    pub runtime_cap_s: u64,
    pub upstream: Vec<UpstreamPlan>,
    pub stock_wait_s: u64,
    pub refs_wait_s: u64,
    /// [`OracleRequest::hold`], and how long after the upstream pods came up
    /// a hold may keep them (s): their cap less the fetch's reserve.
    pub hold: bool,
    pub hold_limit_s: u64,
    /// The money every pod may cost at its price cap over its wall cap, plus
    /// the two readiness cycles fv-deploy may replace per pod (2 × 600 s at
    /// the same price) ($): the balance must stay above fv-deploy's floor
    /// after all of them.
    pub projection_usd: f64,
    /// More `pod up` arguments for the runtime side's fetch (`--fetch-tree
    /// [--fetch-skip ERE] [--fetch-max-mb N]`; ADR-0019 §20.6: the clip-metric
    /// targets' `metrics.txt` is a file of the tree, not of the summaries).
    /// Empty: `pod up`'s default fetch.
    pub runtime_fetch: Vec<String>,
}

impl OraclePlan {
    /// The arms of the targets in `served`, in the tier's order.
    pub fn arms_for(&self, served: &[String]) -> Vec<String> {
        self.arms_by_target
            .iter()
            .filter(|(t, _)| served.contains(t))
            .flat_map(|(_, a)| a.clone())
            .collect()
    }
}

/// The plan of an oracle run, every refusal before anything exists: the tier
/// reads references; `--cells` are arms of it; each selected target is made
/// by a stack; `FV_ORACLE_URL` is the driver's; the caps are only lowered;
/// the waits fit the upstream pods' cap; both launches pass fv-deploy's
/// guards and region check.
pub fn plan_oracle(lab: &Lab, stacks: &Stacks, r: &OracleRequest) -> Result<OraclePlan, String> {
    check_sha7(&r.sha).map_err(|e| e.to_string())?;
    check_full_sha(&r.up_sha)?;
    check_image_tag(&r.image_tag)?;
    stacks.check_against(lab).map_err(|e| e.to_string())?;
    let tier: &Tier = lab
        .tier(&r.tier)
        .ok_or_else(|| format!("no tier `{}` in lab/tiers.toml", r.tier))?;
    // Every cell of the tier, its extra cells too (ADR-0019 §20.3): a
    // clip-metric target is selected by its arms like any other.
    let cells: Vec<_> = tier.all_cells().filter_map(|c| lab.cells.get(c)).collect();
    if !cells.iter().any(|c| c.reference.is_some()) {
        return Err(format!(
            "tier `{}` reads no upstream reference: `fv-lab pod up --family {}` runs it",
            tier.name, tier.name
        ));
    }
    if let Some(e) = r
        .extra_env
        .iter()
        .flat_map(|e| e.split_whitespace())
        .find(|e| e.starts_with("FV_ORACLE_URL="))
    {
        return Err(format!(
            "--extra-env `{e}`: fv-lab oracle sets FV_ORACLE_URL to the upstream pods it brings up"
        ));
    }
    let wanted: Vec<String> = r.cells.clone();
    if wanted.is_empty() {
        return Err(format!(
            "--cells: the arms to run (e.g. every arm of `oracle-fasth3-4step-vsa`); tier `{}` has {} arms",
            tier.name,
            cells.iter().map(|c| c.arms.len()).sum::<usize>()
        ));
    }
    let all: Vec<&str> = cells
        .iter()
        .flat_map(|c| c.arms.iter().map(|a| a.name.as_str()))
        .collect();
    if let Some(c) = wanted.iter().find(|c| !all.contains(&c.as_str())) {
        return Err(format!(
            "--cells `{c}` is not an arm of tier `{}`",
            tier.name
        ));
    }
    let mut arms_by_target: Vec<(String, Vec<String>)> = Vec::new();
    let mut selected = Vec::new();
    for c in &cells {
        let arms: Vec<String> = c
            .arms
            .iter()
            .filter(|a| wanted.contains(&a.name))
            .map(|a| a.name.clone())
            .collect();
        if arms.is_empty() {
            continue;
        }
        selected.extend(arms.iter().cloned());
        match &c.reference {
            Some(rf) => arms_by_target.push((rf.target.clone(), arms)),
            None => {
                return Err(format!(
                    "--cells: cell `{}` reads no upstream reference (run it with `fv-lab pod up`)",
                    c.name
                ))
            }
        }
    }
    let runtime_cap_s = match r.runtime_cap_s {
        None => tier.wall_cap_s,
        Some(c) if c == 0 || c > tier.wall_cap_s => {
            return Err(format!(
                "--wall-cap-s {c}: the runtime tier's cap can only be lowered (1..={} s)",
                tier.wall_cap_s
            ))
        }
        Some(c) => c,
    };
    if r.stock_wait_s == 0 || r.stock_wait_s >= runtime_cap_s {
        return Err(format!(
            "--stock-wait-s {}: positive and below the runtime pod's cap ({runtime_cap_s} s)",
            r.stock_wait_s
        ));
    }
    // One pod per stack that makes a selected target, in the stacks' order.
    let mut upstream = Vec::new();
    // Each pod at its price cap over its wall cap, and the readiness cycles
    // fv-deploy may replace (up to 2 more pods of 600 s each).
    let replaced = u64::from(guard::READY_CYCLES - 1) * guard::READY_TIMEOUT_S;
    let mut projection = tier.max_usd_per_hr * (runtime_cap_s + replaced) as f64 / 3600.0;
    let mut tags = r.upstream_tags.iter();
    let mut min_cap = u64::MAX;
    for s in &stacks.stacks {
        let targets: Vec<String> = s
            .targets
            .iter()
            .filter(|t| arms_by_target.iter().any(|(x, _)| *x == t.name))
            .map(|t| t.name.clone())
            .collect();
        if targets.is_empty() {
            continue;
        }
        let cap = match r.upstream_cap_s {
            None => s.wall_cap_s,
            Some(c) if c == 0 || c > s.wall_cap_s => {
                return Err(format!(
                    "--upstream-wall-cap-s {c}: stack `{}`'s cap can only be lowered (1..={} s)",
                    s.name, s.wall_cap_s
                ))
            }
            Some(c) => c,
        };
        min_cap = min_cap.min(cap);
        let tag = tags
            .next()
            .ok_or_else(|| format!("no upstream tag for stack `{}`", s.name))?;
        let mut start = UpstreamStart::for_stack(s, tag, &r.up_sha, &up_steps(s, &targets));
        start.cap_s = cap;
        let l = upstream_launch(s, &targets, &r.image_tag, &start)?;
        guard::check_guards(&l.guards).map_err(|e| e.to_string())?;
        guard::check_region(&l).map_err(|e| e.to_string())?;
        guard::check_min_ram(&l).map_err(|e| e.to_string())?;
        projection += s.max_usd_per_hr * (cap + replaced) as f64 / 3600.0;
        upstream.push(UpstreamPlan {
            stack: s.name.clone(),
            targets,
            start,
            launch: l,
        });
    }
    if let Some((t, _)) = arms_by_target
        .iter()
        .find(|(t, _)| stacks.for_target(t).is_none())
    {
        return Err(format!(
            "no stack in {UPSTREAM_PATH} makes the reference `{t}`"
        ));
    }
    let room = min_cap.saturating_sub(r.stock_wait_s + REF_FETCH_RESERVE_S);
    let refs_wait_s = match r.refs_wait_s {
        None if room > 0 => room,
        Some(w) if w > 0 && w <= room => w,
        other => {
            return Err(format!(
                "--refs-wait-s {}: the upstream pods (cap {min_cap} s) must live through the wait, the runtime pod's readiness (--stock-wait-s {}) and its fetch ({REF_FETCH_RESERVE_S} s): at most {room} s",
                other.map_or("(default)".into(), |w| w.to_string()),
                r.stock_wait_s
            ))
        }
    };
    let mut extra_env = r.extra_env.clone();
    crate::knobs::check_extra_env(&tier.name, &extra_env)?;
    extra_env.retain(|e| !e.is_empty());
    Ok(OraclePlan {
        tier: tier.name.clone(),
        sha: r.sha.clone(),
        runtime_tag: r.runtime_tag.clone(),
        cells: selected,
        arms_by_target,
        extra_env,
        runtime_cap_s,
        upstream,
        stock_wait_s: r.stock_wait_s,
        refs_wait_s,
        hold: r.hold,
        hold_limit_s: min_cap.saturating_sub(crate::pod::FETCH_RESERVE_S),
        projection_usd: projection,
        runtime_fetch: Vec::new(),
    })
}

/// The base URL the runtime run reads a pod's references from
/// (`FV_ORACLE_URL`): `<pod url>/upstream/<tag>`.
pub fn reference_base(pod_url: &str, tag: &str) -> String {
    format!("{}/{UPSTREAM_FAMILY}/{tag}", pod_url.trim_end_matches('/'))
}

/// `fv-lab pod up`'s arguments for the runtime side: the tier, the arms of
/// the served targets, `FV_ORACLE_URL` (the served pods' bases, comma
/// separated), the plan's tag and cap, `--yes --wait`.
pub fn runtime_args(plan: &OraclePlan, urls: &[String], cells: &[String]) -> Vec<String> {
    let mut a = vec![
        "pod".to_owned(),
        "up".into(),
        "--sha".into(),
        plan.sha.clone(),
        "--family".into(),
        plan.tier.clone(),
        "--tag".into(),
        plan.runtime_tag.clone(),
        "--cells".into(),
        cells.join(","),
        "--extra-env".into(),
        format!("FV_ORACLE_URL={}", urls.join(",")),
    ];
    a.extend(plan.extra_env.iter().cloned());
    a.extend(["--wall-cap-s".into(), plan.runtime_cap_s.to_string()]);
    a.extend(plan.runtime_fetch.iter().cloned());
    a.extend(["--yes".into(), "--wait".into()]);
    a
}

// ------------------------------------------------------------------ the driver

/// An upstream pod once up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpPod {
    pub stack: String,
    pub id: String,
    /// `https://<id>-8000.proxy.runpod.net`.
    pub url: String,
    pub tag: String,
    pub targets: Vec<String>,
}

impl UpPod {
    /// `FV_ORACLE_URL`'s entry for this pod.
    pub fn base(&self) -> String {
        reference_base(&self.url, &self.tag)
    }
}

/// The runtime pod, once `pod up` recorded it (`pod.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimePod {
    pub id: String,
    pub url: String,
    pub tag: String,
}

/// An upstream run as the driver sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpState {
    Running,
    /// `DONE` says `done <tag>`: every step ended (a reference not served by
    /// then failed).
    Finished,
    /// Terminated or exited.
    Gone,
}

/// One reference as the driver sees it (`<base>/oracle-<target>/`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefState {
    /// No `ORACLE_DONE` yet (or the pod could not be asked: polled again).
    Pending,
    /// `ORACLE_DONE` and `oracle-dump.tar`: a dump the runtime side can fetch.
    Served,
    /// `ORACLE_DONE` without `oracle-dump.tar`: the reference's cell ended
    /// without a dump (`scripts/gpu/upstream/oracle.sh` writes `ORACLE_DONE`
    /// either way); why, with its `cell.json` exit when the pod serves it.
    NoDump(String),
}

/// What a reference's directory holds once it is served
/// (`scripts/gpu/upstream/oracle.sh`: written whole, then renamed).
pub const ORACLE_DUMP_TAR: &str = "oracle-dump.tar";

/// How long a stopped runtime side gets to delete what it made (s): past it,
/// it is killed and its pod deleted by the driver.
pub const ABORT_WAIT_S: u64 = 600;

/// What the driver does to the world; [`crate::oracle_live::LiveOps`] talks
/// to Runpod and the pods, the tests' fake records calls. Errors are messages.
pub trait OracleOps {
    fn now(&self) -> f64;
    fn sleep(&mut self, d: Duration);
    fn log(&mut self, line: &str);
    fn interrupted(&self) -> bool;
    /// The upstream pod up and serving its run directory (fv-deploy's `up`,
    /// kept), recorded. An `Err` leaves no pod behind: one created before a
    /// later step failed is deleted first.
    fn upstream_up(&mut self, p: &UpstreamPlan) -> Result<UpPod, String>;
    /// The reference's state: `<base>/oracle-<target>/ORACLE_DONE` answers
    /// 200 and the directory lists [`ORACLE_DUMP_TAR`] ([`RefState::Served`]);
    /// `ORACLE_DONE` without the tar is [`RefState::NoDump`].
    fn reference(&mut self, pod: &UpPod, target: &str) -> RefState;
    fn upstream_state(&mut self, pod: &UpPod) -> UpState;
    /// One fetch of the pod's `keepalive.bin`: its idle watchdog counts the
    /// traffic as busy.
    fn keepalive(&mut self, pod: &UpPod);
    /// Start the runtime side (`fv-lab pod up … --yes --wait` in the
    /// background).
    fn runtime_start(&mut self, args: &[String]) -> Result<(), String>;
    /// The runtime pod, once ready and recorded.
    fn runtime_pod(&mut self) -> Option<RuntimePod>;
    /// `<runtime run>/ORACLE_REFS_DONE` answers 200.
    fn runtime_refs_done(&mut self, rt: &RuntimePod) -> bool;
    /// The runtime side ended: `Some(success)`. A failed one whose pod is
    /// still up has that pod deleted.
    fn runtime_exit(&mut self) -> Option<bool>;
    /// The runtime side's exit code, once it ended with one.
    fn runtime_exit_code(&self) -> Option<i32> {
        None
    }
    /// Ask the runtime side to stop (SIGTERM: it deletes what it made); does
    /// not wait.
    fn runtime_stop(&mut self);
    /// Wait up to `limit` for the runtime side to end; past it, kill it and
    /// delete its pod. Its exit, as [`OracleOps::runtime_exit`].
    fn runtime_wait(&mut self, limit: Duration) -> Option<bool>;
    /// The served pods' bases (`FV_ORACLE_URL`), once known (`--url-file`).
    fn publish_urls(&mut self, urls: &[String]);
    /// Whether the caller lifted `--hold-upstream` (true without a hold).
    fn hold_released(&mut self) -> bool;
    /// Fetch the upstream run (its tree but the dumps and clips).
    fn upstream_fetch(&mut self, pod: &UpPod) -> Result<(), String>;
    /// Delete the upstream pod (fv-deploy's retries and ledger).
    fn upstream_down(&mut self, pod: &UpPod) -> Result<(), String>;
}

/// What happened to one upstream pod.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpstreamOutcome {
    pub stack: String,
    pub id: String,
    pub tag: String,
    /// Why it was released.
    pub released: String,
    /// `Err("not fetched (…)")` when the release skipped the fetch.
    pub fetched: Result<(), String>,
    pub deleted: Result<(), String>,
}

/// The driver's report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct DriveReport {
    pub upstream: Vec<UpstreamOutcome>,
    /// Targets whose reference was served, and those that never were.
    pub served: Vec<String>,
    pub missing: Vec<String>,
    /// The arms the runtime side ran (the served targets' selected arms).
    pub runtime_cells: Vec<String>,
    pub runtime_started: bool,
    /// The runtime run's tag, and its pod once it was ready.
    pub runtime_tag: String,
    pub runtime_pod: Option<RuntimePod>,
    /// The runtime side's exit: `Some(true)` a fetched, judged success.
    pub runtime_ok: Option<bool>,
    pub runtime_exit_code: Option<i32>,
    /// Why the run failed (empty: it succeeded).
    pub problems: Vec<String>,
}

impl DriveReport {
    pub fn ok(&self) -> bool {
        self.problems.is_empty()
    }
}

struct Live {
    pod: UpPod,
    released: bool,
}

/// How often a reference still pending is asked for once its pod's run has
/// ended or the pod is gone, [`FINAL_ASK_GAP_S`] apart, before its target
/// fails: `ORACLE_DONE` and `DONE` can land in one poll window, and one
/// proxy error on the listing must not drop a finished reference.
pub const FINAL_ASKS: u32 = 3;
pub const FINAL_ASK_GAP_S: u64 = 5;

/// One answer of [`OracleOps::reference`] for `t`, noted (and logged).
#[allow(clippy::too_many_arguments)]
fn note_ref<O: OracleOps>(
    ops: &mut O,
    pod: &UpPod,
    t: &str,
    st: RefState,
    served: &mut Vec<String>,
    failed: &mut Vec<String>,
    why: &mut Vec<(String, String)>,
) {
    match st {
        RefState::Pending => {}
        RefState::Served => {
            ops.log(&format!(
                "[fv-lab oracle] reference {t} served by {}",
                pod.id
            ));
            served.push(t.to_owned());
        }
        RefState::NoDump(w) => {
            ops.log(&format!(
                "[fv-lab oracle] reference {t} not served by {}: {w}",
                pod.id
            ));
            failed.push(t.to_owned());
            why.push((t.to_owned(), w));
        }
    }
}

/// Fetch (unless `fetch` is false: an interrupt deletes at once, a fetch may
/// take 900 s a file), then delete, every pod not yet released.
fn release<O: OracleOps>(
    ops: &mut O,
    live: &mut [Live],
    why: &str,
    fetch: bool,
    report: &mut DriveReport,
) {
    for l in live.iter_mut().filter(|l| !l.released) {
        l.released = true;
        ops.log(&format!(
            "[fv-lab oracle] release upstream pod {} ({}): {why}",
            l.pod.id, l.pod.stack
        ));
        let fetched = if fetch {
            ops.upstream_fetch(&l.pod)
        } else {
            Err(format!("not fetched ({why})"))
        };
        if let (true, Err(e)) = (fetch, &fetched) {
            ops.log(&format!(
                "[fv-lab oracle] fetch of upstream {}: {e}",
                l.pod.id
            ));
        }
        let deleted = ops.upstream_down(&l.pod);
        if let Err(e) = &deleted {
            ops.log(&format!(
                "[fv-lab oracle] could not delete upstream pod {}: {e} (its backstop deletes it at the cap)",
                l.pod.id
            ));
            report
                .problems
                .push(format!("upstream pod {} not deleted: {e}", l.pod.id));
        }
        report.upstream.push(UpstreamOutcome {
            stack: l.pod.stack.clone(),
            id: l.pod.id.clone(),
            tag: l.pod.tag.clone(),
            released: why.to_owned(),
            fetched,
            deleted,
        });
    }
}

/// The oracle run (ADR-0019 §19): every path ends with every upstream pod
/// this run created deleted (fetched first, but on an interrupt).
///
/// 1. The upstream pods up, one per stack, in order (a failure releases
///    those already up).
/// 2. The references waited for (every [`POLL_S`], keepalive meanwhile):
///    until each selected target is served (`ORACLE_DONE` and its
///    `oracle-dump.tar`), ended without a dump (`ORACLE_DONE` alone: missing,
///    with the reason), its pod finished without it or is gone (asked
///    [`FINAL_ASKS`] times more first, [`FINAL_ASK_GAP_S`] apart), or
///    `refs_wait_s` passed since the first create. No reference: release
///    without a runtime pod, fail. A pod none of whose targets was served is
///    released at once.
/// 3. The runtime side started on the served targets' arms with their pods'
///    bases as `FV_ORACLE_URL`.
/// 4. Until it ends, keepalive; the upstream pods are released once its pod
///    published `ORACLE_REFS_DONE`, or [`REF_FETCH_RESERVE_S`] after its pod
///    was ready without it; no runtime pod within `stock_wait_s`: the runtime
///    side is asked to stop, the upstream pods are released, then it is
///    waited for ([`ABORT_WAIT_S`]).
/// 5. The upstream pods still up are released when the runtime side ends.
///
/// An interrupt stops the runtime side, deletes the upstream pods without
/// fetching them, then waits for the runtime side. With [`OraclePlan::hold`]
/// the releases of 4 and of 5 after a successful runtime side also wait for
/// the caller to lift the hold (keepalive meanwhile), at most until
/// `hold_limit_s` after the first create; the stock wait, a failed start, a
/// failed runtime side and an interrupt release at once.
pub fn drive<O: OracleOps>(ops: &mut O, plan: &OraclePlan) -> DriveReport {
    let mut report = DriveReport {
        runtime_tag: plan.runtime_tag.clone(),
        ..DriveReport::default()
    };
    let mut live: Vec<Live> = Vec::new();
    let tick = Duration::from_secs(POLL_S);
    let abort = Duration::from_secs(ABORT_WAIT_S);
    // Before the first create: what the references' deadline and a hold's
    // limit count from (the pods' caps count from their creates, later).
    let t_up = ops.now();
    // 1.
    for p in &plan.upstream {
        if ops.interrupted() {
            report.problems.push("interrupted".into());
            release(ops, &mut live, "interrupted", false, &mut report);
            return report;
        }
        ops.log(&format!(
            "[fv-lab oracle] upstream pod ({}): {} on {}",
            p.stack, p.start.up_steps, p.launch.image
        ));
        match ops.upstream_up(p) {
            Ok(pod) => {
                ops.log(&format!(
                    "[fv-lab oracle] upstream {} pod {} tag {}: {}",
                    p.stack,
                    pod.id,
                    pod.tag,
                    pod.base()
                ));
                live.push(Live {
                    pod,
                    released: false,
                });
            }
            Err(e) => {
                report
                    .problems
                    .push(format!("upstream pod ({}) did not come up: {e}", p.stack));
                release(
                    ops,
                    &mut live,
                    "another upstream pod did not come up",
                    true,
                    &mut report,
                );
                report.missing = plan.arms_by_target.iter().map(|(t, _)| t.clone()).collect();
                return report;
            }
        }
    }
    // 2.
    let mut served: Vec<String> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    // Why a failed target was not served (a target without an entry: the
    // general reasons, logged where they happen).
    let mut why: Vec<(String, String)> = Vec::new();
    loop {
        if ops.interrupted() {
            report.problems.push("interrupted".into());
            release(ops, &mut live, "interrupted", false, &mut report);
            return report;
        }
        for l in &live {
            for t in &l.pod.targets {
                if !served.contains(t) && !failed.contains(t) {
                    let st = ops.reference(&l.pod, t);
                    note_ref(ops, &l.pod, t, st, &mut served, &mut failed, &mut why);
                }
            }
        }
        for l in &live {
            let pending = |served: &[String], failed: &[String]| -> Vec<String> {
                l.pod
                    .targets
                    .iter()
                    .filter(|t| !served.contains(t) && !failed.contains(t))
                    .cloned()
                    .collect()
            };
            if pending(&served, &failed).is_empty() {
                continue;
            }
            let st = ops.upstream_state(&l.pod);
            if st != UpState::Running {
                // Its run ended (or the pod went) in this poll window: a
                // reference written just before may have met a transient
                // proxy error above. Ask again, a few times, before failing.
                for ask in 0..FINAL_ASKS {
                    let left = pending(&served, &failed);
                    if left.is_empty() {
                        break;
                    }
                    if ask > 0 {
                        ops.sleep(Duration::from_secs(FINAL_ASK_GAP_S));
                    }
                    for t in &left {
                        let r = ops.reference(&l.pod, t);
                        note_ref(ops, &l.pod, t, r, &mut served, &mut failed, &mut why);
                    }
                }
                let left = pending(&served, &failed);
                if !left.is_empty() {
                    ops.log(&format!(
                        "[fv-lab oracle] upstream pod {} is {st:?} without {}",
                        l.pod.id,
                        left.join(", ")
                    ));
                    failed.extend(left);
                }
            }
        }
        if served.len() + failed.len() >= plan.arms_by_target.len() {
            break;
        }
        if ops.now() - t_up >= plan.refs_wait_s as f64 {
            for (t, _) in &plan.arms_by_target {
                if !served.contains(t) && !failed.contains(t) {
                    ops.log(&format!(
                        "[fv-lab oracle] reference {t} not served {} s after the first upstream create",
                        plan.refs_wait_s
                    ));
                    failed.push(t.clone());
                }
            }
            break;
        }
        for l in &live {
            ops.keepalive(&l.pod);
        }
        ops.sleep(tick);
    }
    // In the plan's order.
    report.served = plan
        .arms_by_target
        .iter()
        .map(|(t, _)| t.clone())
        .filter(|t| served.contains(t))
        .collect();
    report.missing = plan
        .arms_by_target
        .iter()
        .map(|(t, _)| t.clone())
        .filter(|t| !served.contains(t))
        .collect();
    if !report.missing.is_empty() {
        let named: Vec<String> = report
            .missing
            .iter()
            .map(|t| match why.iter().find(|(x, _)| x == t) {
                Some((_, w)) => format!("{t} ({w})"),
                None => t.clone(),
            })
            .collect();
        report
            .problems
            .push(format!("references not served: {}", named.join(", ")));
    }
    if report.served.is_empty() {
        release(ops, &mut live, "no reference was served", true, &mut report);
        return report;
    }
    // A pod that serves nothing is no use to the runtime side.
    let (idle, mut busy): (Vec<Live>, Vec<Live>) = live
        .into_iter()
        .partition(|l| !l.pod.targets.iter().any(|t| served.contains(t)));
    let mut idle = idle;
    release(
        ops,
        &mut idle,
        "none of its references was served",
        true,
        &mut report,
    );
    // 3.
    let cells = plan.arms_for(&report.served);
    let urls: Vec<String> = busy.iter().map(|l| l.pod.base()).collect();
    report.runtime_cells = cells.clone();
    let args = runtime_args(plan, &urls, &cells);
    ops.publish_urls(&urls);
    if let Err(e) = ops.runtime_start(&args) {
        report
            .problems
            .push(format!("runtime side did not start: {e}"));
        release(
            ops,
            &mut busy,
            "the runtime side did not start",
            true,
            &mut report,
        );
        return report;
    }
    report.runtime_started = true;
    // 4.
    let t1 = ops.now();
    let mut released = false;
    // Why the upstream pods may go (once the hold, if any, is lifted).
    let mut trigger: Option<String> = None;
    let mut ready_at: Option<f64> = None;
    // The hold lifted (or none), or its limit passed.
    let lifted = |ops: &mut O| -> Option<&'static str> {
        if !plan.hold || ops.hold_released() {
            Some(if plan.hold {
                "; the hold is lifted"
            } else {
                ""
            })
        } else if ops.now() - t_up >= plan.hold_limit_s as f64 {
            Some("; the hold's limit passed")
        } else {
            None
        }
    };
    let ok = loop {
        if ops.interrupted() {
            report.problems.push("interrupted".into());
            ops.runtime_stop();
            release(ops, &mut busy, "interrupted", false, &mut report);
            break ops.runtime_wait(abort);
        }
        if let Some(ok) = ops.runtime_exit() {
            break Some(ok);
        }
        if !released {
            for l in &busy {
                ops.keepalive(&l.pod);
            }
            if trigger.is_none() {
                match ops.runtime_pod() {
                    None if ops.now() - t1 >= plan.stock_wait_s as f64 => {
                        let why = format!("no runtime pod after {} s", plan.stock_wait_s);
                        report.problems.push(why.clone());
                        ops.runtime_stop();
                        release(ops, &mut busy, &why, true, &mut report);
                        break ops.runtime_wait(abort);
                    }
                    None => {}
                    Some(rt) => {
                        let at = *ready_at.get_or_insert(ops.now());
                        let done = ops.runtime_refs_done(&rt);
                        report.runtime_pod = Some(rt);
                        if done {
                            trigger = Some("the runtime pod has every reference".into());
                        } else if ops.now() - at >= REF_FETCH_RESERVE_S as f64 {
                            let why = format!(
                                "no ORACLE_REFS_DONE {REF_FETCH_RESERVE_S} s after the runtime pod was ready"
                            );
                            ops.log(&format!("[fv-lab oracle] {why}"));
                            trigger = Some(why);
                        }
                    }
                }
            }
            if let Some(t) = &trigger {
                if let Some(how) = lifted(ops) {
                    let why = format!("{t}{how}");
                    release(ops, &mut busy, &why, true, &mut report);
                    released = true;
                }
            }
        }
        ops.sleep(tick);
    };
    report.runtime_ok = ok;
    report.runtime_exit_code = ops.runtime_exit_code();
    match ok {
        Some(true) => {}
        Some(false) => report
            .problems
            .push("the runtime side failed (fv-lab pod up --wait)".into()),
        None => report.problems.push(format!(
            "the runtime side did not end {ABORT_WAIT_S} s after it was stopped (killed, its pod deleted)"
        )),
    }
    // 5. After a success a hold is waited for; after a failure, not.
    let (how, fetch) = if ok == Some(true) && busy.iter().any(|l| !l.released) {
        loop {
            if ops.interrupted() {
                report.problems.push("interrupted".into());
                break ("; interrupted while held", false);
            }
            if let Some(how) = lifted(ops) {
                break (how, true);
            }
            for l in &busy {
                ops.keepalive(&l.pod);
            }
            ops.sleep(tick);
        }
    } else {
        ("", true)
    };
    release(
        ops,
        &mut busy,
        &format!("the runtime side ended{how}"),
        fetch,
        &mut report,
    );
    report
}

/// The run's money guard before anything is created: the account's balance,
/// less what its pods already spend over the longest cap of the run, less
/// every pod of the run at its price cap over its wall cap
/// ([`OraclePlan::projection_usd`]), stays at fv-deploy's floor ($8,
/// CLAUDE.md) or above. fv-deploy checks each create on its own as well.
pub fn check_balance(
    account: &fastvideo_deploy::provider::Account,
    plan: &OraclePlan,
) -> Result<(), String> {
    let cap = plan
        .upstream
        .iter()
        .map(|u| u.start.cap_s)
        .chain([plan.runtime_cap_s])
        .max()
        .unwrap_or(0);
    let spend = account
        .spend_per_hr_usd
        .filter(|x| x.is_finite() && *x > 0.0)
        .unwrap_or(0.0);
    let left = account.balance_usd - spend * cap as f64 / 3600.0 - plan.projection_usd;
    if left.is_finite() && left >= guard::MIN_BALANCE_USD {
        Ok(())
    } else {
        Err(format!(
            "the balance ${:.2} (spending ${spend:.2}/hr) would drop to ${left:.2}, below the ${} floor, if the run's {} pod(s) ran to their caps (${:.2}); lower --wall-cap-s / --upstream-wall-cap-s or select fewer targets",
            account.balance_usd,
            guard::MIN_BALANCE_USD,
            plan.upstream.len() + 1,
            plan.projection_usd
        ))
    }
}
