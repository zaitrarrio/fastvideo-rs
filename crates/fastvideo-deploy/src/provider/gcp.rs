//! A Google Compute Engine GPU VM (docs/adr/0018-fv-deploy.md §8.3): the
//! firewall rule and the instance insert that `scripts/gcp/vm.sh up` sends,
//! field for field (`fixtures/deploy/gcp-{firewall,instance}.json`, written
//! from that script's own `firewall_payload` / `instance_payload`;
//! `tests/payload_goldens.rs`), reached at the VM's external address on port
//! 8000 with the run's key.
//!
//! The VM boots the Ubuntu accelerator image and `scripts/gcp/startup.sh`
//! (shipped as the `startup-script` metadata item), which runs the serve
//! image with the worker config from the `fv-config` item and the container
//! environment from `fv-env` (ADR-0011: the config is sent at boot; a
//! launch's `FV_WORKER_TOML_B64` becomes `fv-config`).
//!
//! Money guards that differ from Runpod's (§8.3): the quote is the price
//! table of `scripts/gcp/lib.sh` (`GCP_PRICES`, read from the file, not a
//! live price: Compute Engine has no price API the scripts use); there is no
//! account balance, so the floor is checked on the run's budget
//! (`FV_GCP_BUDGET_USD`) and `up` refuses without one; the wall clock is
//! held twice, by fv-deploy's detached backstop and by Compute Engine itself
//! (`scheduling.maxRunDuration` = the cap, `instanceTerminationAction:
//! DELETE`), plus the VM's own idle stop and the `fv-deadline` label `vm.sh
//! reap` reads.
//!
//! Auth: an OAuth access token from a service-account key, minted by the
//! reviewed `scripts/gcp/auth.sh` (openssl RS256; no RSA code here, no new
//! dependency) through [`ScriptToken`], or given as `GCP_ACCESS_TOKEN`. The
//! token is only ever an `Authorization: Bearer` header.

use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine as _;
use serde_json::{json, Value};

use super::runpod::PodReach;
use super::{
    scrub, Account, ApiCall, CloudType, Cursor, Instance, InstanceId, InstanceState, Launch,
    LogPage, Offer, OfferQuery, Provider, ProviderError, ProviderKind, Reach, ReachOptions,
};
use crate::boot::env::BootEnv;
use crate::boot::sys::{Clock, Http, Method, Request};

/// The Compute Engine REST base (`lib.sh` `COMPUTE`).
pub const COMPUTE: &str = "https://compute.googleapis.com/compute/v1";
/// Every resource fv-deploy creates carries `fv-owner=fastvideo-rs`; nothing
/// without it is ever deleted.
pub const OWNER_LABEL: &str = "fastvideo-rs";
/// The boot image (`lib.sh` `GCP_IMAGE`: the NVIDIA R580 driver).
pub const BOOT_IMAGE: &str = "projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accelerator-2404-amd64-with-nvidia-580";
/// `lib.sh` defaults: EU (CLAUDE.md, the family Durable Objects in `weur`).
pub const DEFAULT_ZONE: &str = "europe-west4-b";
pub const NETWORK: &str = "default";
/// ICE-UDP (`vm.sh` `UDP_PORTS`).
pub const UDP_PORTS: &str = "40010";
pub const CADDY_IMAGE: &str = "caddy:2";
/// The boot disk (`vm.sh`: `diskSizeGb: "100"`).
pub const BOOT_DISK_GB: u32 = 100;
/// `fv-idle-s` default (`vm.sh` `FV_GCP_IDLE_S`).
pub const IDLE_S: u64 = 1800;
/// The script's price cap (`FV_GCP_MAX_DPH`): a whole G4 VM is about $5/hr.
pub const MAX_USD_PER_HR: f64 = 6.0;
/// The run's budget (the balance floor's "balance" on GCP).
pub const BUDGET_VAR: &str = "FV_GCP_BUDGET_USD";
/// A ready access token instead of minting one.
pub const TOKEN_VAR: &str = "GCP_ACCESS_TOKEN";
/// The secrets `vm.sh` ships by name (`SECRET_NAMES`).
pub const SECRET_NAMES: [&str; 11] = [
    "FV_CF_ACCOUNT_ID",
    "FV_CF_API_TOKEN",
    "FV_D1_DATABASE_ID",
    "FV_R2_BUCKET",
    "FV_R2_ENDPOINT",
    "FV_R2_ACCESS_KEY_ID",
    "FV_R2_SECRET_ACCESS_KEY",
    "FV_WEBHOOK_ED25519_KEY",
    "FV_URL_SIGNING_KEY",
    "FV_INTERNAL_TOKEN",
    "FV_ADMIN_TOKEN",
];
/// Metadata items that carry a repository file (shortened by
/// [`redact_for_display`], as `lib.sh gcp_redact` does).
pub const FILE_ITEMS: [&str; 6] = [
    "startup-script",
    "fv-config",
    "fv-script-verify-weights",
    "fv-script-verify-safetensors",
    "fv-manifest",
    "fv-sha256",
];
/// How many times an operation is waited on (`gce_wait`: 90 × `POST …/wait`).
pub const OP_WAITS: u32 = 90;

/// The weights disk of a zone (`vm.sh` `FV_GCP_WEIGHTS_DISK` default),
/// mounted read-only; never modified or deleted.
pub fn weights_disk(zone: &str) -> String {
    format!("fv-weights-hdml-{zone}")
}

/// `europe-west4-b` → `europe-west4`.
pub fn region_of(zone: &str) -> &str {
    zone.rsplit_once('-').map_or(zone, |(r, _)| r)
}

/// `gcp_boot_disk_type`: G4 / A3 / A4 / C4 boot from Hyperdisk Balanced.
pub fn boot_disk_type(machine: &str) -> &'static str {
    if ["g4-", "a3-", "a4-", "c4-"]
        .iter()
        .any(|p| machine.starts_with(p))
    {
        "hyperdisk-balanced"
    } else {
        "pd-balanced"
    }
}

/// `gcp_provisioning`: `a3-highgpu-{1,2,4}g` exist only as Spot; else Spot
/// when asked.
pub fn provisioning(machine: &str, spot: bool) -> &'static str {
    if matches!(machine, "a3-highgpu-1g" | "a3-highgpu-2g" | "a3-highgpu-4g") || spot {
        "SPOT"
    } else {
        "STANDARD"
    }
}

/// A VM's own settings (the provider constants above aside). The machine
/// type is the launch's `gpu.types[0]`, the zone `placement.data_centers[0]`,
/// the read-only weights disk `placement.volume`, the boot disk
/// `disk.container_gb`, the cloud-side `maxRunDuration` `guards.cap_s`.
/// Not in the `Launch` schema (C3b changes it only by the two provider
/// kinds): `fv-deploy` builds these with the launch
/// (`crate::launch::gcp_options`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GcpOptions {
    /// `fv-family`: the `deploy/deploy.toml` `[[gcp]]` row name.
    pub family: String,
    /// `fv-run`: the launch's stamp.
    pub run: String,
    /// `fv-verify-cells` (`verify-weights.sh` cells checked on the disk).
    pub verify_cells: Vec<String>,
    /// Spot instead of on-demand.
    pub spot: bool,
    /// `fv-idle-s` (0: off).
    pub idle_s: u64,
    /// The firewall rule's `sourceRanges`.
    pub source_ranges: Vec<String>,
    /// `fv-secrets-mode secret-manager` with every [`SECRET_NAMES`] (else
    /// `none`). The script's third mode, secret values in the metadata, is
    /// not offered: a secret value never rides in a request fv-deploy builds.
    pub secret_manager: bool,
}

/// The repository files a VM receives as metadata (`vm.sh metadata_items`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GcpAssets {
    pub startup_script: String,
    pub verify_weights: String,
    pub verify_safetensors: String,
    pub manifest: String,
    pub sha256: String,
}

impl GcpAssets {
    /// Reads them under the repository root.
    pub fn load(root: &Path) -> Result<GcpAssets, ProviderError> {
        let read = |rel: &str| {
            std::fs::read_to_string(root.join(rel)).map_err(|e| {
                ProviderError::Unsupported(format!("{rel}: {e} (the GCP VM's metadata)"))
            })
        };
        Ok(GcpAssets {
            startup_script: read("scripts/gcp/startup.sh")?,
            verify_weights: read("scripts/gpu/verify-weights.sh")?,
            verify_safetensors: read("scripts/gpu/verify-safetensors.sh")?,
            manifest: read("scripts/gpu/weights-manifest.tsv")?,
            sha256: read("scripts/gpu/weights-sha256.tsv")?,
        })
    }
}

/// One row of `lib.sh` `GCP_PRICES`: the whole machine's $/hr.
#[derive(Clone, Debug, PartialEq)]
pub struct PriceRow {
    pub machine: String,
    pub region: String,
    pub on_demand: f64,
    pub spot: f64,
}

/// The price table of `scripts/gcp/lib.sh` (`GCP_PRICES`), read from the
/// file so the two never differ.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GcpPrices(pub Vec<PriceRow>);

impl GcpPrices {
    pub fn parse(lib_sh: &str) -> GcpPrices {
        let Some(start) = lib_sh.find("GCP_PRICES=\"") else {
            return GcpPrices::default();
        };
        let body = lib_sh[start + "GCP_PRICES=\"".len()..]
            .split('"')
            .next()
            .unwrap_or("");
        GcpPrices(
            body.lines()
                .filter_map(|l| {
                    let f: Vec<&str> = l.split('\t').collect();
                    Some(PriceRow {
                        machine: (*f.first()?).trim().to_owned(),
                        region: (*f.get(1)?).trim().to_owned(),
                        on_demand: f.get(2)?.trim().parse().ok()?,
                        spot: f.get(3)?.trim().parse().ok()?,
                    })
                })
                .collect(),
        )
    }

    pub fn load(root: &Path) -> Result<GcpPrices, ProviderError> {
        let p = root.join("scripts/gcp/lib.sh");
        let t = std::fs::read_to_string(&p)
            .map_err(|e| ProviderError::Unsupported(format!("{}: {e}", p.display())))?;
        Ok(Self::parse(&t))
    }

    /// `gcp_price`: the region's row, else the highest listed row for the
    /// machine (conservative); `None` when the machine is not listed.
    pub fn price(&self, machine: &str, spot: bool, region: &str) -> Option<f64> {
        let col = |r: &PriceRow| if spot { r.spot } else { r.on_demand };
        let rows: Vec<&PriceRow> = self.0.iter().filter(|r| r.machine == machine).collect();
        rows.iter()
            .find(|r| r.region == region)
            .map(|r| col(r))
            .or_else(|| rows.iter().map(|r| col(r)).reduce(f64::max))
    }
}

/// Where an access token comes from (asked before every request: a token
/// lives an hour, the backstop may wait four).
pub trait TokenSource {
    fn token(&self) -> Result<String, ProviderError>;
}

/// A fixed token (`GCP_ACCESS_TOKEN`; tests).
pub struct StaticToken(pub String);

impl TokenSource for StaticToken {
    fn token(&self) -> Result<String, ProviderError> {
        if self.0.is_empty() {
            return Err(ProviderError::MissingKey(TOKEN_VAR));
        }
        Ok(self.0.clone())
    }
}

/// A token minted by `scripts/gcp/auth.sh` (`gcp_token`: from
/// `GCP_SA_KEY_JSON` / `GCP_SA_KEY_FILE`, cached mode 600 until 5 minutes
/// before it expires). The key never passes through this process.
pub struct ScriptToken {
    pub root: PathBuf,
}

impl TokenSource for ScriptToken {
    fn token(&self) -> Result<String, ProviderError> {
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg("source \"$0/scripts/gcp/auth.sh\" && gcp_token")
            .arg(&self.root)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| ProviderError::Transport {
                what: "scripts/gcp/auth.sh gcp_token".into(),
                why: e.to_string(),
            })?;
        let tok = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        if !out.status.success() || tok.is_empty() {
            // Its stderr names the problem (never the key or the token).
            return Err(ProviderError::MissingKey(
                "a GCP token (GCP_SA_KEY_JSON / GCP_SA_KEY_FILE for scripts/gcp/auth.sh, or GCP_ACCESS_TOKEN)",
            ));
        }
        Ok(tok)
    }
}

/// `GCP_PROJECT`, else the service-account key's `project_id`
/// (`GCP_SA_KEY_JSON` as JSON or base64, `GCP_SA_KEY_FILE`, else
/// `/root/.config/fv/gcp_sa_key.json`), as `auth.sh gcp_project` reads it.
pub fn project_from_env(env: &BootEnv) -> Option<String> {
    if let Some(p) = env.get("GCP_PROJECT").filter(|p| !p.is_empty()) {
        return Some(p.to_owned());
    }
    let raw = match env.get("GCP_SA_KEY_JSON").filter(|k| !k.is_empty()) {
        Some(k) => k.to_owned(),
        None => {
            let f = env
                .get("GCP_SA_KEY_FILE")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/root/.config/fv/gcp_sa_key.json"));
            std::fs::read_to_string(f).ok()?
        }
    };
    let text = if raw.trim_start().starts_with('{') {
        raw
    } else {
        String::from_utf8(
            base64::engine::general_purpose::STANDARD
                .decode(raw.trim())
                .ok()?,
        )
        .ok()?
    };
    serde_json::from_str::<Value>(&text)
        .ok()?
        .get("project_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// The Compute Engine client.
pub struct GcpApi<'a> {
    http: &'a dyn Http,
    clock: &'a dyn Clock,
    token: &'a dyn TokenSource,
    pub project: String,
    /// The zone of id-based calls (status, delete, logs).
    pub zone: String,
    pub compute: String,
    /// The run's budget (`FV_GCP_BUDGET_USD`): GCP reports no balance.
    pub budget_usd: Option<f64>,
}

impl std::fmt::Debug for GcpApi<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcpApi")
            .field("project", &self.project)
            .field("zone", &self.zone)
            .field("compute", &self.compute)
            .field("budget_usd", &self.budget_usd)
            .finish_non_exhaustive()
    }
}

impl<'a> GcpApi<'a> {
    pub fn new(
        http: &'a dyn Http,
        clock: &'a dyn Clock,
        token: &'a dyn TokenSource,
        project: impl Into<String>,
        zone: impl Into<String>,
    ) -> Self {
        Self {
            http,
            clock,
            token,
            project: project.into(),
            zone: zone.into(),
            compute: COMPUTE.into(),
            budget_usd: None,
        }
    }

    /// `FV_GCP_COMPUTE_API` (the scripts' test override), `GCP_ZONE`,
    /// `FV_GCP_BUDGET_USD`; the project from [`project_from_env`] (`plan`
    /// without one shows `dry-run-project`, as `vm.sh plan`).
    pub fn from_env(
        http: &'a dyn Http,
        clock: &'a dyn Clock,
        token: &'a dyn TokenSource,
        env: &BootEnv,
    ) -> Self {
        let mut a = Self::new(
            http,
            clock,
            token,
            project_from_env(env).unwrap_or_else(|| "dry-run-project".into()),
            env.get("GCP_ZONE").unwrap_or(DEFAULT_ZONE),
        );
        if let Some(c) = env.get("FV_GCP_COMPUTE_API") {
            a.compute = c.trim_end_matches('/').to_owned();
        }
        a.budget_usd = env
            .get(BUDGET_VAR)
            .and_then(|b| b.trim().parse::<f64>().ok())
            .filter(|b| b.is_finite() && *b >= 0.0);
        a
    }

    pub fn clock(&self) -> &'a dyn Clock {
        self.clock
    }

    fn zone_url(&self, zone: &str, rest: &str) -> String {
        format!(
            "{}/projects/{}/zones/{zone}{rest}",
            self.compute, self.project
        )
    }

    fn firewalls_url(&self) -> String {
        format!(
            "{}/projects/{}/global/firewalls",
            self.compute, self.project
        )
    }

    /// One call: `(status, JSON)`; transport errors are errors, any status
    /// is returned (the caller decides what a 404 means).
    fn send(
        &self,
        method: Method,
        url: &str,
        body: Option<&Value>,
        what: &str,
    ) -> Result<(u16, Value), ProviderError> {
        let tok = self.token.token()?;
        let mut req = Request::new(method, url, Duration::from_secs(120)).bearer(&tok);
        if let Some(b) = body {
            req = req.json(b.to_string());
        }
        let r = self.http.send(&req).map_err(|e| ProviderError::Transport {
            what: what.to_owned(),
            why: scrub(&e.to_string(), &tok, 400),
        })?;
        let v = super::body_json(&r.body);
        if let Value::String(s) = &v {
            if s.contains(&tok) {
                return Ok((r.status, Value::String(scrub(s, &tok, 400))));
            }
        }
        Ok((r.status, v))
    }

    /// A 2xx answer's JSON; anything else an error (`.error.message`).
    fn ok(
        &self,
        method: Method,
        url: &str,
        body: Option<&Value>,
        what: &str,
    ) -> Result<Value, ProviderError> {
        let (s, v) = self.send(method, url, body, what)?;
        if (200..300).contains(&s) {
            return Ok(v);
        }
        Err(api_error(what, s, &v))
    }

    /// `gce_wait`: `POST <selfLink>/wait` until `DONE`; an operation error
    /// is a refusal (the resource was not made).
    fn wait(&self, op: &Value, what: &str) -> Result<(), ProviderError> {
        let link =
            op.get("selfLink")
                .and_then(Value::as_str)
                .ok_or_else(|| ProviderError::Decode {
                    what: what.to_owned(),
                    why: "not an operation (no selfLink)".into(),
                })?;
        let mut cur = op.clone();
        for _ in 0..OP_WAITS {
            if cur.get("status").and_then(Value::as_str) == Some("DONE") {
                if let Some(e) = cur.get("error") {
                    let text = e.to_string();
                    return Err(ProviderError::NoCapacity(format!(
                        "{what}: operation failed: {}",
                        scrub(&text, "", 400)
                    )));
                }
                return Ok(());
            }
            cur = self.ok(Method::Post, &format!("{link}/wait"), None, what)?;
        }
        Err(ProviderError::Transport {
            what: what.to_owned(),
            why: "the operation is still running after the wait budget".into(),
        })
    }

    /// The instance, `None` when there is none (404).
    pub fn instance(&self, zone: &str, name: &str) -> Result<Option<Value>, ProviderError> {
        let (s, v) = self.send(
            Method::Get,
            &self.zone_url(zone, &format!("/instances/{name}")),
            None,
            "GET instance",
        )?;
        match s {
            404 => Ok(None),
            200..=299 => Ok(Some(v)),
            _ => Err(api_error("GET instance", s, &v)),
        }
    }

    /// Deletes our firewall rule of a VM (absent: nothing to do; not ours:
    /// left alone, `firewall_delete`). Whether one was deleted.
    fn delete_firewall(&self, vm: &str) -> Result<bool, ProviderError> {
        let url = format!("{}/fv-serve-{vm}", self.firewalls_url());
        let (s, v) = self.send(Method::Get, &url, None, "GET firewall")?;
        if s == 404 {
            return Ok(false);
        }
        if !(200..300).contains(&s) {
            return Err(api_error("GET firewall", s, &v));
        }
        if !is_ours(&v) {
            return Ok(false);
        }
        let op = self.ok(Method::Delete, &url, None, "DELETE firewall")?;
        self.wait(&op, "DELETE firewall")?;
        Ok(true)
    }

    /// `vm_down`: the instance (ours only) and its firewall rule; whether
    /// anything was deleted. A VM without `fv-owner=fastvideo-rs` is refused.
    pub fn delete_vm(&self, zone: &str, name: &str) -> Result<bool, ProviderError> {
        let mut any = false;
        if let Some(inst) = self.instance(zone, name)? {
            if !is_ours(&inst) {
                return Err(ProviderError::Unsupported(format!(
                    "instance {name} has no fv-owner={OWNER_LABEL} label: refusing to delete"
                )));
            }
            let url = self.zone_url(zone, &format!("/instances/{name}"));
            let (s, op) = self.send(Method::Delete, &url, None, "DELETE instance")?;
            if s != 404 {
                if !(200..300).contains(&s) {
                    return Err(api_error("DELETE instance", s, &op));
                }
                self.wait(&op, "DELETE instance")?;
                any = true;
            }
        }
        any |= self.delete_firewall(name)?;
        Ok(any)
    }
}

fn api_error(what: &str, status: u16, v: &Value) -> ProviderError {
    let msg = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .map_or_else(|| v.to_string(), str::to_owned);
    ProviderError::Api {
        what: what.to_owned(),
        status,
        body: scrub(&msg, "", 400),
    }
}

/// `gcp_is_ours`: labelled `fv-owner=fastvideo-rs`, or (a firewall rule,
/// which has no labels) so described.
pub fn is_ours(v: &Value) -> bool {
    v.pointer("/labels/fv-owner").and_then(Value::as_str) == Some(OWNER_LABEL)
        || v.get("description")
            .and_then(Value::as_str)
            .is_some_and(|d| d.contains(&format!("fv-owner={OWNER_LABEL}")))
}

/// Compute Engine `status` → [`InstanceState`] (its `TERMINATED` is a
/// stopped VM that still exists; a deleted one answers 404).
pub fn instance_state(status: &str) -> InstanceState {
    match status {
        "RUNNING" => InstanceState::Running,
        "PROVISIONING" | "STAGING" | "REPAIRING" => InstanceState::Pending,
        "STOPPING" | "STOPPED" | "SUSPENDING" | "SUSPENDED" | "TERMINATED" => InstanceState::Exited,
        _ => InstanceState::Unknown,
    }
}

/// `gcp_labels <kind> <run> [extra]`.
fn labels(kind: &str, run: &str, extra: &[(&str, &str)]) -> Value {
    let mut m = serde_json::Map::new();
    m.insert("fv-owner".into(), json!(OWNER_LABEL));
    m.insert("fv-kind".into(), json!(kind));
    if !run.is_empty() {
        m.insert("fv-run".into(), json!(run));
    }
    for (k, v) in extra {
        m.insert((*k).to_owned(), json!(v));
    }
    Value::Object(m)
}

fn machine_of(l: &Launch) -> Result<&str, ProviderError> {
    l.gpu
        .types
        .first()
        .map(String::as_str)
        .ok_or_else(|| ProviderError::Unsupported("GCP: no machine type (gpu.types)".into()))
}

fn zone_of(l: &Launch) -> &str {
    l.placement
        .data_centers
        .first()
        .map_or(DEFAULT_ZONE, String::as_str)
}

/// The worker config's text (`fv-config`) from the launch's
/// `FV_WORKER_TOML_B64`, and the container env (`fv-env`) without it.
fn config_and_env(l: &Launch) -> Result<(String, Value), ProviderError> {
    let b64 = l.env.get("FV_WORKER_TOML_B64").ok_or_else(|| {
        ProviderError::Unsupported(
            "GCP: the worker config rides as the fv-config metadata item; the launch has no FV_WORKER_TOML_B64".into(),
        )
    })?;
    let text = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .ok()
        .and_then(|b| String::from_utf8(b).ok())
        .ok_or_else(|| {
            ProviderError::Unsupported("GCP: FV_WORKER_TOML_B64 does not decode".into())
        })?;
    let env: serde_json::Map<String, Value> = l
        .env
        .iter()
        .filter(|(k, _)| k.as_str() != "FV_WORKER_TOML_B64")
        .map(|(k, v)| (k.clone(), json!(v)))
        .collect();
    Ok((text, Value::Object(env)))
}

/// The instance insert of a launch (`vm.sh instance_payload`,
/// `fixtures/deploy/gcp-instance.json`); `deadline` is the `fv-deadline`
/// label (unix seconds: creation + the cap).
pub fn instance_body(
    l: &Launch,
    o: &GcpOptions,
    assets: &GcpAssets,
    project: &str,
    deadline: u64,
) -> Result<Value, ProviderError> {
    if !l.entrypoint.is_empty() || !l.args.is_empty() {
        return Err(ProviderError::Unsupported(
            "GCP: startup.sh runs the image's own entrypoint with --config <fv-config>; a launch entrypoint or arguments cannot be sent".into(),
        ));
    }
    let machine = machine_of(l)?;
    let zone = zone_of(l);
    let (config, env) = config_and_env(l)?;
    let mut disks = vec![json!({
        "boot": true, "autoDelete": true, "deviceName": "boot",
        "initializeParams": {
            "sourceImage": BOOT_IMAGE,
            "diskSizeGb": l.disk.container_gb.to_string(),
            "diskType": format!("zones/{zone}/diskTypes/{}", boot_disk_type(machine)),
            "labels": labels("serve", &o.run, &[]),
        }
    })];
    if let Some(v) = &l.placement.volume {
        disks.push(json!({
            "source": format!("projects/{project}/zones/{zone}/disks/{}", v.id),
            "mode": "READ_ONLY", "autoDelete": false, "deviceName": "fv-weights",
        }));
    }
    let (mode, names) = if o.secret_manager {
        ("secret-manager", SECRET_NAMES.join(" "))
    } else {
        ("none", String::new())
    };
    let item = |k: &str, v: &str| json!({"key": k, "value": v});
    let items = vec![
        item("startup-script", &assets.startup_script),
        item("fv-config", &config),
        item("fv-role", "serve"),
        item("fv-image", &l.image),
        item("fv-env", &env.to_string()),
        item("fv-verify-cells", &o.verify_cells.join(" ")),
        item("fv-secrets-mode", mode),
        item("fv-secret-names", &names),
        item("fv-encode-bench", "0"),
        item("fv-idle-s", &o.idle_s.to_string()),
        item("fv-tls", "none"),
        item("fv-caddy-image", CADDY_IMAGE),
        item("fv-script-verify-weights", &assets.verify_weights),
        item("fv-script-verify-safetensors", &assets.verify_safetensors),
        item("fv-manifest", &assets.manifest),
        item("fv-sha256", &assets.sha256),
    ];
    let deadline = deadline.to_string();
    Ok(json!({
        "name": l.name,
        "description": format!("fastvideo-rs fv-serve (fv-owner={OWNER_LABEL})"),
        "machineType": format!("zones/{zone}/machineTypes/{machine}"),
        "labels": labels("serve", &o.run, &[("fv-family", &o.family), ("fv-role", "serve"), ("fv-deadline", &deadline)]),
        "tags": {"items": ["fv-serve", l.name]},
        "scheduling": {
            "onHostMaintenance": "TERMINATE", "automaticRestart": false,
            "provisioningModel": provisioning(machine, o.spot),
            "instanceTerminationAction": "DELETE",
            "maxRunDuration": {"seconds": l.guards.cap_s.to_string()},
        },
        "disks": disks,
        "networkInterfaces": [{
            "network": format!("projects/{project}/global/networks/{NETWORK}"),
            "nicType": "GVNIC", "stackType": "IPV4_ONLY",
            "accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT", "networkTier": "PREMIUM"}],
        }],
        "serviceAccounts": [{
            "email": format!("fv-vm@{project}.iam.gserviceaccount.com"),
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        }],
        "shieldedInstanceConfig": {"enableSecureBoot": false, "enableVtpm": true, "enableIntegrityMonitoring": true},
        "metadata": {"items": items},
    }))
}

/// The VM's firewall rule (`vm.sh firewall_payload`,
/// `fixtures/deploy/gcp-firewall.json`): fv-serve's 8000 and ICE-TCP 40000,
/// ICE-UDP 40010, from `source_ranges` only, to this VM's tag.
pub fn firewall_body(name: &str, o: &GcpOptions, project: &str) -> Value {
    json!({
        "name": format!("fv-serve-{name}"),
        "network": format!("projects/{project}/global/networks/{NETWORK}"),
        "direction": "INGRESS", "priority": 1000,
        "description": format!("fv-owner={OWNER_LABEL} fv-run={} fv-serve TCP 8000,40000, ICE-UDP {UDP_PORTS}", o.run),
        "targetTags": [name],
        "sourceRanges": o.source_ranges,
        "allowed": [{"IPProtocol": "tcp", "ports": ["8000", "40000"]}, {"IPProtocol": "udp", "ports": [UDP_PORTS]}],
    })
}

/// For printing a plan: file items shortened to their length and first line
/// (`lib.sh gcp_redact`); the request itself is unchanged.
pub fn redact_for_display(body: &mut Value) {
    if let Some(items) = body
        .pointer_mut("/metadata/items")
        .and_then(Value::as_array_mut)
    {
        for it in items {
            let file = it
                .get("key")
                .and_then(Value::as_str)
                .is_some_and(|k| FILE_ITEMS.contains(&k));
            if let (true, Some(v)) = (file, it.get("value").and_then(Value::as_str)) {
                let first: String = v.lines().next().unwrap_or("").chars().take(60).collect();
                it["value"] = json!(format!("<{} chars: {first}...>", v.chars().count()));
            }
        }
    }
}

/// `mask_tokens`: the one-time admin token in the serial log.
pub fn mask_tokens(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(i) = rest.find("fvadm_") {
        out.push_str(&rest[..i]);
        out.push_str("fvadm_<masked>");
        let tail = &rest[i + "fvadm_".len()..];
        let n = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            .unwrap_or(tail.len());
        rest = &tail[n..];
    }
    out.push_str(rest);
    out
}

/// Compute Engine VMs.
pub struct GcpVm<'a> {
    pub api: GcpApi<'a>,
    pub options: GcpOptions,
    pub assets: GcpAssets,
    pub prices: GcpPrices,
}

impl std::fmt::Debug for GcpVm<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcpVm")
            .field("api", &self.api)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl<'a> GcpVm<'a> {
    pub fn new(api: GcpApi<'a>, options: GcpOptions, assets: GcpAssets, prices: GcpPrices) -> Self {
        Self {
            api,
            options,
            assets,
            prices,
        }
    }

    fn deadline(&self, l: &Launch) -> u64 {
        (self.api.clock.now() + l.guards.cap_s as f64)
            .ceil()
            .max(0.0) as u64
    }
}

impl Provider for GcpVm<'_> {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Gcp
    }

    /// The table price of each machine type in the zone's region (the whole
    /// VM, on-demand or Spot); a machine the table does not list has no
    /// quote and is never created.
    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>, ProviderError> {
        let zone = q.data_center.as_deref().unwrap_or(&self.api.zone);
        Ok(q.gpu_types
            .iter()
            .map(|m| {
                let spot = provisioning(m, self.options.spot) == "SPOT";
                Offer {
                    provider: self.kind(),
                    gpu_type: m.clone(),
                    data_center: Some(zone.to_owned()),
                    cloud: CloudType::Secure,
                    usd_per_hr: self.prices.price(m, spot, region_of(zone)),
                    stock: Some("price table (scripts/gcp/lib.sh)".into()),
                }
            })
            .collect())
    }

    fn payload(&self, launch: &Launch) -> Result<Vec<ApiCall>, ProviderError> {
        let project = &self.api.project;
        let zone = zone_of(launch);
        Ok(vec![
            ApiCall {
                method: "POST".into(),
                url: self.api.firewalls_url(),
                body: firewall_body(&launch.name, &self.options, project),
            },
            ApiCall {
                method: "POST".into(),
                url: self.api.zone_url(zone, "/instances"),
                body: instance_body(
                    launch,
                    &self.options,
                    &self.assets,
                    project,
                    self.deadline(launch),
                )?,
            },
        ])
    }

    /// `vm_up`: the weights disk must exist; the firewall rule, then the
    /// instance (each operation waited on; a failed instance removes its
    /// rule); its external address is the URL.
    fn create(&self, launch: &Launch) -> Result<Instance, ProviderError> {
        let zone = zone_of(launch);
        let body = instance_body(
            launch,
            &self.options,
            &self.assets,
            &self.api.project,
            self.deadline(launch),
        )?;
        let machine = machine_of(launch)?.to_owned();
        if let Some(v) = &launch.placement.volume {
            let url = self.api.zone_url(zone, &format!("/disks/{}", v.id));
            let (s, d) = self.api.send(Method::Get, &url, None, "GET weights disk")?;
            if !(200..300).contains(&s) {
                return Err(api_error(
                    &format!(
                        "weights disk {} in {zone} (scripts/gcp/weights.sh disk-up)",
                        v.id
                    ),
                    s,
                    &d,
                ));
            }
        }
        let op = self.api.ok(
            Method::Post,
            &self.api.firewalls_url(),
            Some(&firewall_body(
                &launch.name,
                &self.options,
                &self.api.project,
            )),
            "POST firewall",
        )?;
        self.api.wait(&op, "POST firewall")?;
        let created = self
            .api
            .ok(
                Method::Post,
                &self.api.zone_url(zone, "/instances"),
                Some(&body),
                "POST instance",
            )
            .and_then(|op| self.api.wait(&op, "POST instance"));
        if let Err(e) = created {
            // Refused for sure: the rule is ours and alone, remove it. Else
            // the VM may exist: the caller's sweep (by name) removes both.
            if e.is_definite_refusal() {
                let _ = self.api.delete_firewall(&launch.name);
            }
            return Err(e);
        }
        let ip = self.api.instance(zone, &launch.name)?.and_then(|v| {
            v.pointer("/networkInterfaces/0/accessConfigs/0/natIP")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
        let spot = provisioning(&machine, self.options.spot) == "SPOT";
        Ok(Instance {
            id: InstanceId::new(launch.name.clone()),
            provider: self.kind(),
            name: launch.name.clone(),
            state: InstanceState::Pending,
            url: ip.map(|ip| format!("http://{ip}:8000")),
            gpu: Some(machine.clone()),
            data_center: Some(zone.to_owned()),
            usd_per_hr: self.prices.price(&machine, spot, region_of(zone)),
            created_at: Some(self.api.clock.now()),
        })
    }

    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError> {
        Ok(match self.api.instance(&self.api.zone, id.as_str())? {
            None => InstanceState::Terminated,
            Some(v) => instance_state(v.get("status").and_then(Value::as_str).unwrap_or("")),
        })
    }

    /// The serial console (`vm.sh ssh-free-logs`): startup progress and the
    /// container log, admin tokens masked; the cursor is the console offset.
    fn logs(&self, id: &InstanceId, since: Option<&Cursor>) -> Result<LogPage, ProviderError> {
        let start = since.map_or("0", |c| c.0.as_str());
        let url = self.api.zone_url(
            &self.api.zone,
            &format!("/instances/{id}/serialPort?port=1&start={start}"),
        );
        let v = self.api.ok(Method::Get, &url, None, "GET serial port")?;
        let lines = v
            .get("contents")
            .and_then(Value::as_str)
            .unwrap_or("")
            .lines()
            .map(mask_tokens)
            .collect();
        let next = v.get("next").and_then(|n| {
            n.as_str()
                .map(str::to_owned)
                .or_else(|| n.as_u64().map(|x| x.to_string()))
        });
        Ok(LogPage {
            lines,
            next: next.map(Cursor),
        })
    }

    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError> {
        self.api.delete_vm(&self.api.zone, id.as_str()).map(|_| ())
    }

    /// GCP bills an account with no prepaid balance: the floor is checked
    /// on the run's budget (`FV_GCP_BUDGET_USD`); without one, nothing is
    /// created.
    fn account(&self) -> Result<Account, ProviderError> {
        let b = self
            .api
            .budget_usd
            .ok_or(ProviderError::MissingKey(BUDGET_VAR))?;
        Ok(Account {
            provider: self.kind(),
            balance_usd: b,
            spend_per_hr_usd: None,
        })
    }

    fn reach<'b>(
        &'b self,
        instance: &Instance,
        opts: &ReachOptions<'_>,
    ) -> Result<Box<dyn Reach + 'b>, ProviderError> {
        let base = match &instance.url {
            Some(u) => u.clone(),
            None => {
                let ip = self
                    .api
                    .instance(&self.api.zone, instance.id.as_str())?
                    .and_then(|v| {
                        v.pointer("/networkInterfaces/0/accessConfigs/0/natIP")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .ok_or_else(|| ProviderError::Decode {
                        what: "GET instance".into(),
                        why: format!("{} has no external address", instance.id),
                    })?;
                format!("http://{ip}:8000")
            }
        };
        Ok(Box::new(PodReach {
            http: self.api.http,
            base,
            key: opts.key.map(str::to_owned),
        }))
    }

    /// Instance names are unique in a zone: the sweep of a name is the
    /// delete of that VM (ours only) and its firewall rule.
    fn sweep(
        &self,
        name: &str,
        keep: Option<&InstanceId>,
    ) -> Result<Vec<InstanceId>, ProviderError> {
        if keep.is_some_and(|k| k.as_str() == name) {
            return Ok(Vec::new());
        }
        Ok(if self.api.delete_vm(&self.api.zone, name)? {
            vec![InstanceId::new(name)]
        } else {
            Vec::new()
        })
    }

    /// Compute Engine reports no price at create: the cap is the table
    /// quote's.
    fn bills_at_create(&self) -> bool {
        false
    }
}
