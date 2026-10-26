//! `fv-deploy`: one instance of a deployment preset on a provider, with the
//! shared guards (docs/adr/0018-fv-deploy.md, docs/ops/fv-deploy.md).
//!
//! ```text
//! fv-deploy plan  --family F --preset P [--region eu] [--kind pod|serverless|vast|gcp] [options]
//!                 the Launch and the exact provider requests; no network
//! fv-deploy up    … --yes [--no-smoke] [--keep]
//!                 balance floor, price cap, create, backstop, ready (replace
//!                 after 600 s, at most 3 cycles), smoke, delete (unless --keep);
//!                 without --yes it is `plan`
//! fv-deploy status <id> | logs <id> [--follow] | smoke <id> | down <id>  [--kind …]
//! fv-deploy schema launch|instance|offer|account
//! fv-deploy templates sync <image> <ref> | sync-all <file> | list | id <image> <sls|pod> | plan <image> <ref>
//!                 the Runpod serve templates (scripts/serve/runpod-templates.sh)
//! fv-deploy release promote <sha|digest|tag> [channel] | rollback [channel] [--to ID] | resolve <target>
//!                 release channels (scripts/serve/release.sh): GHCR channel
//!                 tags, the templates, the D1 history; dispatches release.yml
//!                 unless --local (or in GitHub Actions); --dry-run plans
//! fv-deploy reconcile [--dry-run] [--adopt] [--fix]
//!                 the account against the D1 deployments ledger
//! fv-deploy backstop --kind K --id ID --deadline T --armed-file F
//!                 (started by `up`, detached; writes F once armed)
//! ```
//!
//! The provider's key comes from `RUNPOD_API_KEY` (Runpod), `VAST_API_KEY`
//! (Vast) or a GCP access token (`GCP_ACCESS_TOKEN`, else minted by
//! `scripts/gcp/auth.sh` from `GCP_SA_KEY_JSON` / `GCP_SA_KEY_FILE`), and is
//! never printed. GCP has no balance: `FV_GCP_BUDGET_USD` (or
//! `--budget-usd`) is the run's budget the floor is checked on. `up` and
//! `down` record Runpod pods and endpoints in D1's `deployments` when a
//! Cloudflare token is set (`FV_CF_API_TOKEN`, as the scripts; C3d).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};

use fastvideo_deploy::boot::env::BootEnv;
use fastvideo_deploy::boot::http::ReqwestHttp;
use fastvideo_deploy::boot::sys::{Clock, Http, LogSink, RealClock, RealSpawner, StderrLog};
use fastvideo_deploy::guard::{self, run_backstop, DetachedBackstop};
use fastvideo_deploy::launch::{self, PresetRequest, Repo, KEY_HASH_PLACEHOLDER};
use fastvideo_deploy::lifecycle::{smoke, Deployer, Ledger, Recorder, SmokeOptions, UpOptions};
use fastvideo_deploy::provider::gcp::{
    self, GcpApi, GcpAssets, GcpOptions, GcpPrices, GcpVm, ScriptToken, StaticToken, TokenSource,
};
use fastvideo_deploy::provider::runpod::Flavour;
use fastvideo_deploy::provider::runpod::{Pod, RunpodApi, RunpodUrls, Serverless};
use fastvideo_deploy::provider::vast::{Vast, VastApi};
use fastvideo_deploy::provider::{
    schema, Cursor, Guards, Instance, InstanceId, Launch, Provider, ProviderKind, ReachOptions,
};
use fastvideo_deploy::release::d1::D1;
use fastvideo_deploy::release::registry::{D1Recorder, Registry};
use fastvideo_deploy::release::{self, promote, reconcile, templates, ReleaseError};

#[derive(Parser)]
#[command(
    name = "fv-deploy",
    version,
    about = "Deploy a preset on a provider with the shared money guards (docs/ops/fv-deploy.md)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the Launch and the provider's exact requests (no network).
    Plan(LaunchArgs),
    /// Create, wait for ready, smoke and delete (`--yes`; else as `plan`).
    Up(UpArgs),
    /// The provider's state of an instance and one /ping.
    Status(IdArgs),
    /// The instance's logs.
    Logs {
        #[command(flatten)]
        id: IdArgs,
        /// Keep reading every 10 s.
        #[arg(long)]
        follow: bool,
    },
    /// The smoke against a running instance.
    Smoke {
        #[command(flatten)]
        id: IdArgs,
        #[command(flatten)]
        smoke: SmokeArgs,
    },
    /// Delete an instance.
    Down(IdArgs),
    /// Print a JSON Schema (launch, instance, offer, account).
    Schema { name: String },
    /// The Runpod serve templates (fv-serve-<image>-sls / -pod).
    Templates {
        #[command(subcommand)]
        cmd: TemplatesCmd,
        /// The repository root (default: the nearest parent with deploy/deploy.toml).
        #[arg(long, global = true)]
        root: Option<PathBuf>,
    },
    /// Release channels: promote a build, roll a channel back, resolve a build.
    Release {
        #[command(subcommand)]
        cmd: ReleaseCmd,
        #[arg(long, global = true)]
        root: Option<PathBuf>,
    },
    /// Match the account's pods / endpoints / templates to the D1 deployments ledger.
    Reconcile {
        /// Show, change nothing.
        #[arg(long)]
        dry_run: bool,
        /// Record unknown fv-serve resources.
        #[arg(long)]
        adopt: bool,
        /// Record the live image of drifted ones.
        #[arg(long)]
        fix: bool,
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// The detached backstop `up` starts (not for direct use).
    Backstop {
        #[arg(long)]
        kind: String,
        #[arg(long)]
        id: String,
        /// Unix seconds.
        #[arg(long)]
        deadline: f64,
        /// Written (`armed`) once the backstop can delete: `up` waits for it.
        #[arg(long)]
        armed_file: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum TemplatesCmd {
    /// Create or update the image's templates to boot <reference> (a digest).
    Sync {
        image: String,
        reference: String,
        /// The release channel the templates name (default FV_RELEASE_CHANNEL).
        #[arg(long)]
        channel: Option<String>,
    },
    /// `sync` for every "<image> <reference>" line of a file.
    SyncAll {
        file: PathBuf,
        #[arg(long)]
        channel: Option<String>,
    },
    /// The fv-serve-* templates: id, name, image.
    List,
    /// One template's id.
    Id { image: String, flavour: String },
    /// The create bodies (no request).
    Plan {
        image: String,
        reference: String,
        #[arg(long)]
        channel: Option<String>,
    },
}

#[derive(Args, Clone, Default)]
struct ReleaseFlags {
    /// Print the plan; change nothing anywhere.
    #[arg(long)]
    dry_run: bool,
    /// Run here (retag, templates, D1) instead of dispatching release.yml.
    #[arg(long, conflicts_with = "dispatch")]
    local: bool,
    /// Dispatch release.yml (the default outside GitHub Actions).
    #[arg(long)]
    dispatch: bool,
    /// A note for the release history.
    #[arg(long)]
    notes: Option<String>,
    /// Leave the Runpod templates as they are.
    #[arg(long)]
    no_templates: bool,
    /// Promote a build missing some serve images (they keep their tags).
    #[arg(long)]
    allow_partial: bool,
    /// Re-apply a release the channel already has.
    #[arg(long)]
    force: bool,
    /// The channel (when not given as the positional argument; default stable).
    #[arg(long = "channel")]
    channel_flag: Option<String>,
}

#[derive(Subcommand)]
enum ReleaseCmd {
    /// Point the channel's tags at a build (no rebuild), its templates when
    /// the channel is theirs, and record the release.
    Promote {
        /// A git sha (7+ hex), an image digest (sha256:…) or a tag.
        target: String,
        channel: Option<String>,
        #[command(flatten)]
        flags: ReleaseFlags,
    },
    /// Re-promote the channel's previous release (or --to ID).
    Rollback {
        channel: Option<String>,
        /// A release id instead of the previous one.
        #[arg(long)]
        to: Option<i64>,
        #[command(flatten)]
        flags: ReleaseFlags,
    },
    /// The image set of one build (JSON; no change).
    Resolve { target: String },
}

#[derive(Args, Clone)]
struct LaunchArgs {
    /// The model family (families/<family>.toml).
    #[arg(long)]
    family: String,
    /// A preset id of deploy/deploy.toml.
    #[arg(long)]
    preset: String,
    #[arg(long, default_value = "eu")]
    region: String,
    /// pod, serverless, vast or gcp.
    #[arg(long, default_value = "pod")]
    kind: String,
    /// An image reference instead of the preset's channel tag.
    #[arg(long)]
    image: Option<String>,
    /// The release channel of the preset's image (stable or latest).
    #[arg(long, default_value = "stable")]
    channel: String,
    /// GPU types in fallback order (default: the region's).
    #[arg(long = "gpu")]
    gpus: Vec<String>,
    /// A volume id (only the EU volume is accepted).
    #[arg(long)]
    volume: Option<String>,
    /// Wall-clock cap (s): the backstop deletes the instance then.
    #[arg(long, default_value_t = guard::CAP_S)]
    cap_s: u64,
    /// The price cap ($/hr; default per provider: Runpod 2.50 for one GPU,
    /// Vast 0.60, GCP 6.00 for the whole VM).
    #[arg(long)]
    max_usd_per_hr: Option<f64>,
    #[arg(long, default_value_t = guard::MIN_BALANCE_USD)]
    min_balance: f64,
    #[arg(long, default_value_t = guard::READY_TIMEOUT_S)]
    ready_timeout_s: u64,
    #[arg(long, default_value_t = guard::READY_CYCLES)]
    ready_cycles: u32,
    /// Lift the upper bounds (cap 4 h, 3 cycles, 600 s readiness); never the $8 floor.
    #[arg(long)]
    force_guards: bool,
    /// The repository root (default: the nearest parent with deploy/deploy.toml).
    #[arg(long)]
    root: Option<PathBuf>,
    #[command(flatten)]
    gcp: GcpArgs,
}

/// GCP only (`scripts/gcp/vm.sh`'s settings).
#[derive(Args, Clone)]
struct GcpArgs {
    /// Who may reach the VM (CIDRs; default FV_GCP_SOURCE_CIDR, else this
    /// machine's public addresses as /32).
    #[arg(long = "source-cidr")]
    source_cidrs: Vec<String>,
    /// A Spot VM instead of on-demand.
    #[arg(long)]
    spot: bool,
    /// The VM deletes itself after this long at 0 % GPU (0: off).
    #[arg(long, default_value_t = gcp::IDLE_S)]
    idle_s: u64,
    /// The VM reads the FV_* secrets from Secret Manager by name (else none).
    #[arg(long)]
    secret_manager: bool,
    /// The run's budget (USD) the balance floor is checked on (GCP has no
    /// balance); default FV_GCP_BUDGET_USD.
    #[arg(long)]
    budget_usd: Option<f64>,
}

#[derive(Args, Clone)]
struct SmokeArgs {
    /// The smoke job's model (default: the first of /fv/v1/capabilities).
    #[arg(long)]
    model: Option<String>,
    /// JSON merged into the job body.
    #[arg(long)]
    params: Option<String>,
    #[arg(long, default_value_t = 600)]
    job_wait_s: u64,
}

#[derive(Args, Clone)]
struct UpArgs {
    #[command(flatten)]
    launch: LaunchArgs,
    #[command(flatten)]
    smoke: SmokeArgs,
    /// Actually create (without it, `up` prints the plan).
    #[arg(long)]
    yes: bool,
    /// Skip the smoke.
    #[arg(long)]
    no_smoke: bool,
    /// Leave the instance running (the backstop still deletes it at the cap).
    #[arg(long)]
    keep: bool,
    /// Ledger, run keys, reports (default: <root>/artifacts/fv-deploy).
    #[arg(long)]
    state_dir: Option<PathBuf>,
}

#[derive(Args, Clone)]
struct IdArgs {
    id: String,
    /// pod, serverless, vast or gcp (default: what `up` recorded).
    #[arg(long)]
    kind: Option<String>,
    #[arg(long)]
    state_dir: Option<PathBuf>,
    #[arg(long)]
    root: Option<PathBuf>,
    /// GCP: the VM's zone (default: what `up` recorded; required without a
    /// record, never GCP_ZONE: a wrong zone would read a live VM as gone).
    #[arg(long)]
    zone: Option<String>,
}

fn fail(msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("[fv-deploy] FAILED: {msg}");
    ExitCode::FAILURE
}

fn kind_of(s: &str) -> Result<ProviderKind, String> {
    ProviderKind::parse(s).ok_or_else(|| format!("unknown kind {s} (pod, serverless, vast, gcp)"))
}

fn root_of(arg: Option<&PathBuf>) -> Result<PathBuf, String> {
    if let Some(r) = arg {
        return Ok(r.clone());
    }
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    Repo::find_root(&cwd)
        .ok_or_else(|| "no deploy/deploy.toml above the current directory (--root)".into())
}

fn state_dir(arg: Option<&PathBuf>, root: Option<&PathBuf>) -> PathBuf {
    arg.cloned().unwrap_or_else(|| {
        root_of(root)
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("artifacts/fv-deploy")
    })
}

fn file_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '~'
            }
        })
        .collect()
}

fn print_json(v: &impl serde::Serialize) {
    match serde_json::to_string_pretty(v) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("[fv-deploy] json: {e}"),
    }
}

/// The longest cap with a fixed `GCP_ACCESS_TOKEN` (a token lives an hour;
/// the backstop deletes at the cap with the token it inherited).
const STATIC_TOKEN_MAX_CAP_S: u64 = 50 * 60;

fn request(a: &LaunchArgs, key_hash: &str) -> Result<PresetRequest, String> {
    let kind = kind_of(&a.kind)?;
    if let Some(b) = a.gcp.budget_usd {
        if !(b.is_finite() && b >= 0.0) {
            return Err(format!(
                "--budget-usd {b}: a finite amount of dollars, at least 0"
            ));
        }
    }
    Ok(PresetRequest {
        family: a.family.clone(),
        preset: a.preset.clone(),
        region: a.region.clone(),
        kind,
        image: a.image.clone(),
        channel: a.channel.clone(),
        gpu_types: a.gpus.clone(),
        volume: a.volume.clone(),
        guards: Guards {
            force: a.force_guards,
            ..Guards::all(
                a.max_usd_per_hr
                    .unwrap_or_else(|| guard::default_max_usd_per_hr(kind)),
                a.min_balance,
                a.cap_s,
                a.ready_timeout_s,
                a.ready_cycles,
            )
        },
        key_hash: key_hash.to_owned(),
    })
}

/// The repository root a detached backstop is told (GCP: its token script).
const ROOT_VAR: &str = "FV_DEPLOY_ROOT";

/// What a provider is built from: the HTTP and clock seams, the
/// environment, the GCP token source, the repository root (GCP's metadata
/// files and price table).
struct Ctx {
    env: BootEnv,
    http: Box<dyn Http>,
    clock: RealClock,
    token: Box<dyn TokenSource>,
    root: Option<PathBuf>,
}

impl Ctx {
    /// The real one (`offline`: `plan`, whose HTTP makes no request and
    /// whose providers need no key).
    fn new(root: Option<PathBuf>, offline: bool) -> Result<Ctx, String> {
        let env = BootEnv::from_process();
        let http: Box<dyn Http> = if offline {
            Box::new(NoHttp)
        } else {
            Box::new(ReqwestHttp::new().map_err(|e| e.to_string())?)
        };
        let root = root
            .or_else(|| env.get(ROOT_VAR).map(PathBuf::from))
            .or_else(|| root_of(None).ok());
        let token: Box<dyn TokenSource> = match (offline, env.get(gcp::TOKEN_VAR)) {
            (true, _) => Box::new(StaticToken(String::new())),
            (false, Some(t)) => Box::new(StaticToken(t.to_owned())),
            (false, None) => Box::new(ScriptToken {
                root: root.clone().unwrap_or_else(|| PathBuf::from(".")),
            }),
        };
        Ok(Ctx {
            env,
            http,
            clock: RealClock,
            token,
            root,
        })
    }

    /// The provider's API key variable's value ("" when unset).
    fn key(&self, kind: ProviderKind) -> String {
        let var = match kind {
            ProviderKind::RunpodPod | ProviderKind::RunpodServerless => {
                fastvideo_deploy::provider::runpod::API_KEY_VAR
            }
            ProviderKind::Vast => fastvideo_deploy::provider::vast::API_KEY_VAR,
            ProviderKind::Gcp => gcp::TOKEN_VAR,
        };
        self.env.get(var).unwrap_or("").to_owned()
    }

    /// The provider of `kind`; `keyed`: its key must be set (every command
    /// but `plan`). `gcp`: the VM's settings (defaults for id commands).
    fn provider(
        &self,
        kind: ProviderKind,
        keyed: bool,
        gcp_opts: Option<GcpOptions>,
        budget: Option<f64>,
        zone: Option<&str>,
    ) -> Result<Box<dyn Provider + '_>, String> {
        let http = self.http.as_ref();
        Ok(match kind {
            ProviderKind::RunpodPod | ProviderKind::RunpodServerless => {
                let api = if keyed {
                    RunpodApi::from_env(http, &self.clock, &self.env).map_err(|e| e.to_string())?
                } else {
                    RunpodApi::new(http, &self.clock, "", RunpodUrls::from_env(&self.env))
                };
                if kind == ProviderKind::RunpodPod {
                    Box::new(Pod::new(api))
                } else {
                    Box::new(Serverless::new(api))
                }
            }
            ProviderKind::Vast => {
                let api = if keyed {
                    VastApi::from_env(http, &self.clock, &self.env).map_err(|e| e.to_string())?
                } else {
                    VastApi::new(
                        http,
                        &self.clock,
                        "",
                        fastvideo_deploy::provider::vast::base_from(&self.env),
                    )
                };
                Box::new(Vast::new(api))
            }
            ProviderKind::Gcp => {
                let mut api = GcpApi::from_env(http, &self.clock, self.token.as_ref(), &self.env);
                if let Some(b) = budget {
                    api.budget_usd = Some(b);
                }
                // The launch's zone, not GCP_ZONE: status and delete must look
                // where the VM was created.
                if let Some(z) = zone {
                    api.zone = z.to_owned();
                }
                let (assets, prices) = match &self.root {
                    Some(r) => (
                        GcpAssets::load(r).map_err(|e| e.to_string())?,
                        GcpPrices::load(r).map_err(|e| e.to_string())?,
                    ),
                    None => (GcpAssets::default(), GcpPrices::default()),
                };
                let opts = gcp_opts.unwrap_or(GcpOptions {
                    family: String::new(),
                    run: String::new(),
                    verify_cells: Vec::new(),
                    spot: false,
                    idle_s: gcp::IDLE_S,
                    source_ranges: Vec::new(),
                    secret_manager: false,
                });
                Box::new(GcpVm::new(api, opts, assets, prices))
            }
        })
    }
}

/// The GCP VM settings of a launch (`None` for the other kinds).
fn gcp_options(
    a: &LaunchArgs,
    repo: &Repo,
    req: &PresetRequest,
    stamp: &str,
    cidrs: Vec<String>,
) -> Result<Option<GcpOptions>, String> {
    if req.kind != ProviderKind::Gcp {
        return Ok(None);
    }
    launch::gcp_options(
        repo,
        req,
        stamp,
        cidrs,
        a.gcp.spot,
        a.gcp.idle_s,
        a.gcp.secret_manager,
    )
    .map(Some)
    .map_err(|e| e.to_string())
}

/// `--source-cidr`, else `FV_GCP_SOURCE_CIDR` (comma or space separated);
/// `discover`: else this machine's public addresses as /32
/// (`gcp_source_cidrs`, api.ipify.org over HTTPS and HTTP); `plan`: the
/// script's dry-run placeholder.
fn source_cidrs(
    a: &LaunchArgs,
    env: &BootEnv,
    http: Option<&dyn Http>,
) -> Result<Vec<String>, String> {
    let split = |s: &str| -> Vec<String> {
        s.split([',', ' '])
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(str::to_owned)
            .collect()
    };
    let given: Vec<String> = a.gcp.source_cidrs.iter().flat_map(|s| split(s)).collect();
    if !given.is_empty() {
        return Ok(given);
    }
    // Empty or separators only: unset (as lib.sh), never an empty list.
    if let Some(e) = env
        .get("FV_GCP_SOURCE_CIDR")
        .map(split)
        .filter(|v| !v.is_empty())
    {
        return Ok(e);
    }
    let Some(http) = http else {
        return Ok(vec!["198.51.100.7/32".into()]);
    };
    let mut ips: Vec<String> = Vec::new();
    for url in ["https://api.ipify.org", "http://api.ipify.org"] {
        let req = fastvideo_deploy::boot::sys::Request::new(
            fastvideo_deploy::boot::sys::Method::Get,
            url,
            Duration::from_secs(10),
        );
        if let Ok(r) = http.send(&req) {
            let ip = String::from_utf8_lossy(&r.body).trim().to_owned();
            let v4 = ip.split('.').count() == 4 && ip.split('.').all(|o| o.parse::<u8>().is_ok());
            if r.ok() && v4 && !ips.contains(&ip) {
                ips.push(ip);
            }
        }
    }
    if ips.is_empty() {
        return Err("GCP: no source CIDR (pass --source-cidr or set FV_GCP_SOURCE_CIDR)".into());
    }
    Ok(ips.into_iter().map(|ip| format!("{ip}/32")).collect())
}

/// The first SIGINT / SIGTERM sets the flag (the lifecycle then deletes
/// what it created); a second one ends the process at once (exit 130: the
/// backstops still delete at their caps).
fn interrupt_flag() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let f = Arc::clone(&flag);
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return;
        };
        rt.block_on(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let Ok(mut term) = signal(SignalKind::terminate()) else {
                return;
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!(
                "[fv-deploy] interrupted: deleting what this run created (again to quit now)"
            );
            f.store(true, Ordering::SeqCst);
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!("[fv-deploy] second signal: quitting; the backstop deletes at the cap");
            std::process::exit(130);
        });
    });
    flag
}

fn smoke_options(s: &SmokeArgs) -> Result<SmokeOptions, String> {
    let params = match &s.params {
        Some(p) => Some(serde_json::from_str::<Value>(p).map_err(|e| format!("--params: {e}"))?),
        None => None,
    };
    Ok(SmokeOptions {
        model: s.model.clone(),
        params,
        job_wait_s: s.job_wait_s,
    })
}

fn cmd_plan(a: &LaunchArgs) -> Result<(), String> {
    let root = root_of(a.root.as_ref())?;
    let repo = Repo::load(&root).map_err(|e| e.to_string())?;
    // No key, no network: the payload is the provider code's own builder.
    let ctx = Ctx::new(Some(root), true)?;
    let stamp = launch::utc_stamp(ctx.clock.now());
    let req = request(a, KEY_HASH_PLACEHOLDER)?;
    let l = launch::from_preset(&repo, &req, &stamp).map_err(|e| e.to_string())?;
    guard::check_guards(&l.guards).map_err(|e| e.to_string())?;
    guard::check_region(&l).map_err(|e| e.to_string())?;
    let cidrs = source_cidrs(a, &ctx.env, None)?;
    let opts = gcp_options(a, &repo, &req, &stamp, cidrs)?;
    let zone = l.placement.data_centers.first().map(String::as_str);
    let p = ctx.provider(l.provider, false, opts, a.gcp.budget_usd, zone)?;
    let mut calls = p.payload(&l).map_err(|e| e.to_string())?;
    if l.provider == ProviderKind::Gcp {
        for c in &mut calls {
            gcp::redact_for_display(&mut c.body);
        }
    }
    print_json(&json!({
        "launch": l,
        "requests": calls,
        "note": "the image is pinned to its digest by `up`; FV_API_KEYS is the run key's SHA-256",
    }));
    Ok(())
}

/// The plan's HTTP: never called.
struct NoHttp;

impl Http for NoHttp {
    fn send(
        &self,
        req: &fastvideo_deploy::boot::sys::Request,
    ) -> Result<fastvideo_deploy::boot::sys::Response, fastvideo_deploy::boot::sys::SysError> {
        Err(fastvideo_deploy::boot::sys::SysError::Http {
            url: req.url.clone(),
            why: "plan makes no request".into(),
        })
    }
    fn download(
        &self,
        url: &str,
        _dest: &Path,
        _t: Duration,
    ) -> Result<u64, fastvideo_deploy::boot::sys::SysError> {
        Err(fastvideo_deploy::boot::sys::SysError::Http {
            url: url.to_owned(),
            why: "plan makes no request".into(),
        })
    }
}

fn write_private(p: &Path, text: &str) -> Result<(), String> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt as _;
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(0o600)
        .open(p)
        .map_err(|e| format!("{}: {e}", p.display()))?;
    f.write_all(text.as_bytes()).map_err(|e| e.to_string())
}

fn cmd_up(u: &UpArgs) -> Result<bool, String> {
    if !u.yes {
        eprintln!("[fv-deploy] dry run (add --yes to create): the plan follows");
        cmd_plan(&u.launch)?;
        return Ok(true);
    }
    let root = root_of(u.launch.root.as_ref())?;
    let repo = Repo::load(&root).map_err(|e| e.to_string())?;
    let state = state_dir(u.state_dir.as_ref(), Some(&root));
    let ctx = Ctx::new(Some(root.clone()), false)?;
    let clock = &ctx.clock;
    let log = StderrLog;
    let key = launch::run_key().map_err(|e| e.to_string())?;
    let stamp = launch::utc_stamp(clock.now());
    let req = request(&u.launch, &launch::key_hash(&key))?;
    let mut l = launch::from_preset(&repo, &req, &stamp).map_err(|e| e.to_string())?;
    if l.provider == ProviderKind::Gcp
        && ctx.env.get(gcp::TOKEN_VAR).is_some()
        && l.guards.cap_s.unwrap_or(0) > STATIC_TOKEN_MAX_CAP_S
    {
        return Err(format!(
            "--cap-s {}: with a fixed {} the backstop's delete at the cap would outlive the token (about an hour); at most {STATIC_TOKEN_MAX_CAP_S} s, or let scripts/gcp/auth.sh mint tokens (GCP_SA_KEY_JSON / GCP_SA_KEY_FILE, {} unset)",
            l.guards.cap_s.unwrap_or(0),
            gcp::TOKEN_VAR,
            gcp::TOKEN_VAR
        ));
    }
    let opts = if l.provider == ProviderKind::Gcp {
        let cidrs = source_cidrs(&u.launch, &ctx.env, Some(ctx.http.as_ref()))?;
        gcp_options(&u.launch, &repo, &req, &stamp, cidrs)?
    } else {
        None
    };
    let pinned = launch::ghcr_resolve(ctx.http.as_ref(), &l.image).map_err(|e| e.to_string())?;
    launch::pin_image(&mut l, &pinned);
    log.line(&format!(
        "[fv-deploy] {} {} on {}: image {pinned}",
        l.provider,
        l.name,
        l.gpu.types.join(", ")
    ));
    // The env guard's test: never send the key in a launch (it is only a header).
    let api_key = ctx.key(l.provider);
    let leaks = |s: &str| !api_key.is_empty() && s.contains(api_key.as_str());
    let zone = l.placement.data_centers.first().map(String::as_str);
    let p = ctx.provider(l.provider, true, opts, u.launch.gcp.budget_usd, zone)?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut backstop = DetachedBackstop::new(
        exe,
        Some(state.join(format!("backstop-{stamp}.log"))),
        state.join("armed"),
    );
    backstop.env = backstop_env(&ctx, &l, u.launch.gcp.budget_usd);
    std::fs::create_dir_all(&state).map_err(|e| format!("{}: {e}", state.display()))?;
    let ledger = Ledger {
        path: state.join("ledger.tsv"),
    };
    let d1 = deployments_d1(&ctx, &root, l.provider, &log);
    let reg = d1.as_ref().map(|d| Registry::new(d, clock, &ctx.env));
    let rec = reg.as_ref().map(|r| D1Recorder { registry: r });
    let flag = interrupt_flag();
    let interrupted = || flag.load(Ordering::SeqCst);
    let d = Deployer {
        provider: p.as_ref(),
        clock,
        log: &log,
        backstop: &backstop,
        ledger: Some(&ledger),
        recorder: rec.as_ref().map(|r| r as &dyn Recorder),
        interrupted: &interrupted,
        leaks: &leaks,
        poll: Duration::from_secs(5),
    };
    let opts = UpOptions {
        // fv-serve checks the run key everywhere but behind Runpod's queue.
        run_key: (l.provider != ProviderKind::RunpodServerless).then(|| key.clone()),
        smoke: if u.no_smoke {
            None
        } else {
            Some(smoke_options(&u.smoke)?)
        },
        keep: u.keep,
    };
    let report = d.up(&l, &opts).map_err(|e| e.to_string())?;
    if let Some(inst) = &report.instance {
        let rec = json!({"instance": inst, "launch": l.name});
        let f = state.join(format!("instances/{}.json", file_id(inst.id.as_str())));
        let _ = write_private(&f, &rec.to_string());
        if u.keep {
            write_private(
                &state.join(format!("keys/{}.key", file_id(inst.id.as_str()))),
                &key,
            )?;
            log.line(&format!(
                "[fv-deploy] run key in {}",
                state.join("keys").display()
            ));
        }
    }
    let _ = write_private(
        &state.join(format!("up-{stamp}.json")),
        &serde_json::to_string_pretty(&report).unwrap_or_default(),
    );
    print_json(&report);
    Ok(report.ok)
}

/// The D1 client of the `deployments` ledger, for a Runpod kind with a
/// Cloudflare token set (else `None`, with a note: `registry.sh` does the same).
fn deployments_d1<'a>(
    ctx: &'a Ctx,
    root: &Path,
    kind: ProviderKind,
    log: &dyn LogSink,
) -> Option<D1<'a>> {
    if !kind.is_runpod() || ctx.env.raw("FV_REGISTRY") == Some("0") {
        return None;
    }
    let sql = std::fs::read_to_string(root.join("deploy/d1/registry.sql")).ok();
    let d = D1::from_env(ctx.http.as_ref(), &ctx.env, sql);
    if d.is_none() {
        log.line("[fv-deploy] no D1 token (FV_CF_API_TOKEN): the deployments ledger is not written (docs/serve/releases.md)");
    }
    d
}

fn recorded(state: &Path, id: &str) -> Option<Instance> {
    let t = std::fs::read_to_string(state.join(format!("instances/{}.json", file_id(id)))).ok()?;
    let v: Value = serde_json::from_str(&t).ok()?;
    serde_json::from_value(v.get("instance")?.clone()).ok()
}

/// The kind of an id: `--kind`, else what `up` recorded, else a serverless
/// id (`<endpoint>/<template>`); a bare id with neither is refused (a pod
/// `down` of an endpoint id would "succeed" on its 404).
fn id_kind(
    flag: Option<&str>,
    recorded: Option<ProviderKind>,
    id: &str,
) -> Result<ProviderKind, String> {
    match (flag, recorded) {
        (Some(k), _) => kind_of(k),
        (None, Some(k)) => Ok(k),
        (None, None) if id.contains('/') => Ok(ProviderKind::RunpodServerless),
        (None, None) => Err(format!(
            "no local record of {id}: pass --kind pod or --kind serverless"
        )),
    }
}

/// What the detached backstop of a launch needs besides the inherited
/// environment: for GCP the zone, the project, the budget and the repository
/// root (its token script), which `up` may have taken from elsewhere.
fn backstop_env(ctx: &Ctx, l: &Launch, budget: Option<f64>) -> Vec<(String, String)> {
    if l.provider != ProviderKind::Gcp {
        return Vec::new();
    }
    let mut v = vec![(
        "GCP_ZONE".to_owned(),
        l.placement
            .data_centers
            .first()
            .cloned()
            .unwrap_or_else(|| gcp::DEFAULT_ZONE.into()),
    )];
    if let Some(p) = gcp::project_from_env(&ctx.env) {
        v.push(("GCP_PROJECT".into(), p));
    }
    if let Some(b) = budget {
        v.push((gcp::BUDGET_VAR.into(), b.to_string()));
    }
    if let Some(r) = &ctx.root {
        v.push((ROOT_VAR.into(), r.display().to_string()));
    }
    v
}

fn with_provider<T>(
    a: &IdArgs,
    f: impl FnOnce(&dyn Provider, Instance, &Path, &RealClock) -> Result<T, String>,
) -> Result<T, String> {
    let state = state_dir(a.state_dir.as_ref(), a.root.as_ref());
    let rec = recorded(&state, &a.id);
    let kind = id_kind(a.kind.as_deref(), rec.as_ref().map(|r| r.provider), &a.id)?;
    let ctx = Ctx::new(a.root.clone(), false)?;
    let clock = &ctx.clock;
    let url = match kind {
        ProviderKind::RunpodPod => Some(RunpodUrls::from_env(&ctx.env).pod_url(&a.id)),
        ProviderKind::RunpodServerless | ProviderKind::Vast | ProviderKind::Gcp => None,
    };
    let inst = rec.unwrap_or(Instance {
        id: InstanceId::new(a.id.clone()),
        provider: kind,
        name: a.id.clone(),
        state: fastvideo_deploy::provider::InstanceState::Unknown,
        url,
        gpu: None,
        data_center: None,
        usd_per_hr: None,
        created_at: None,
    });
    // GCP: the zone `up` recorded, else --zone; never a default.
    let zone = match kind {
        ProviderKind::Gcp => Some(
            a.zone
                .clone()
                .or_else(|| inst.data_center.clone())
                .ok_or_else(|| {
                    format!(
                        "no local record of the GCP VM {}: pass --zone (the zone it runs in)",
                        a.id
                    )
                })?,
        ),
        _ => None,
    };
    let p = ctx.provider(kind, true, None, None, zone.as_deref())?;
    f(p.as_ref(), inst, &state, clock)
}

fn run() -> Result<bool, String> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Plan(a) => cmd_plan(&a).map(|()| true),
        Cmd::Up(u) => cmd_up(&u),
        Cmd::Status(a) => with_provider(&a, |p, inst, state, _| {
            let st = p.status(&inst.id).map_err(|e| e.to_string())?;
            let key =
                std::fs::read_to_string(state.join(format!("keys/{}.key", file_id(&a.id)))).ok();
            let ro = ReachOptions {
                key: key.as_deref().map(str::trim),
                ..ReachOptions::default()
            };
            let ping = p
                .reach(&inst, &ro)
                .map(|mut r| format!("{:?}", r.ping()))
                .unwrap_or_else(|e| e.to_string());
            print_json(
                &json!({"id": inst.id, "provider": inst.provider, "state": st, "ping": ping}),
            );
            Ok(true)
        }),
        Cmd::Logs { id, follow } => with_provider(&id, |p, inst, _, clock| {
            let mut cur: Option<Cursor> = None;
            loop {
                let page = p.logs(&inst.id, cur.as_ref()).map_err(|e| e.to_string())?;
                for l in &page.lines {
                    println!("{l}");
                }
                if !follow || page.next.is_none() {
                    return Ok(true);
                }
                cur = page.next;
                clock.sleep(Duration::from_secs(10));
            }
        }),
        Cmd::Smoke { id, smoke: s } => with_provider(&id, |p, inst, state, clock| {
            let key =
                std::fs::read_to_string(state.join(format!("keys/{}.key", file_id(&id.id)))).ok();
            let so = smoke_options(&s)?;
            let ro = ReachOptions {
                key: key.as_deref().map(str::trim),
                call_timeout: Duration::from_secs(so.job_wait_s),
            };
            let mut reach = p.reach(&inst, &ro).map_err(|e| e.to_string())?;
            let flag = interrupt_flag();
            let stop = || flag.load(Ordering::SeqCst);
            let r = smoke(reach.as_mut(), &so, clock, &StderrLog, &stop);
            print_json(&r);
            Ok(r.ok)
        }),
        Cmd::Down(a) => with_provider(&a, |p, inst, state, clock| {
            let ledger = Ledger {
                path: state.join("ledger.tsv"),
            };
            let never = || false;
            let leaks = |_: &str| false;
            let none = DetachedBackstop::new(PathBuf::new(), None, state.join("armed"));
            let ctx = Ctx::new(a.root.clone(), false)?;
            let root = ctx.root.clone().unwrap_or_else(|| PathBuf::from("."));
            let d1 = deployments_d1(&ctx, &root, p.kind(), &StderrLog);
            let reg = d1.as_ref().map(|d| Registry::new(d, clock, &ctx.env));
            let rec = reg.as_ref().map(|r| D1Recorder { registry: r });
            let d = Deployer {
                provider: p,
                clock,
                log: &StderrLog,
                backstop: &none,
                ledger: Some(&ledger),
                recorder: rec.as_ref().map(|r| r as &dyn Recorder),
                interrupted: &never,
                leaks: &leaks,
                poll: Duration::from_secs(5),
            };
            d.down(&inst.id).map_err(|e| e.to_string())?;
            println!("deleted {}", inst.id);
            Ok(true)
        }),
        Cmd::Templates { cmd, root } => cmd_templates(cmd, root.as_ref()),
        Cmd::Release { cmd, root } => cmd_release(cmd, root.as_ref()),
        Cmd::Reconcile {
            dry_run,
            adopt,
            fix,
            root,
        } => with_release(root.as_ref(), false, |ctx| {
            reconcile::reconcile(
                ctx,
                &reconcile::Options {
                    dry_run,
                    adopt,
                    fix,
                },
            )
            .map(|_| true)
        }),
        Cmd::Schema { name } => {
            let s = schema::by_name(&name)
                .ok_or_else(|| format!("no schema {name} (launch, instance, offer, account)"))?;
            print!("{}", schema::text(&s));
            Ok(true)
        }
        Cmd::Backstop {
            kind,
            id,
            deadline,
            armed_file,
        } => {
            // As nohup: a hangup (or a Ctrl-C to a group it is not in) never
            // stops it; the handlers stay installed for the process's life.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            let (hup, int) = rt.block_on(async {
                use tokio::signal::unix::{signal, SignalKind};
                (
                    signal(SignalKind::hangup()),
                    signal(SignalKind::interrupt()),
                )
            });
            let _guards = (
                hup.map_err(|e| format!("SIGHUP handler: {e}"))?,
                int.map_err(|e| format!("SIGINT handler: {e}"))?,
            );
            let kind = kind_of(&kind)?;
            let ctx = Ctx::new(None, false)?;
            let clock = &ctx.clock;
            let p = ctx.provider(kind, true, None, None, None)?;
            // GCP: a token must come now, not first at the deadline.
            if kind == ProviderKind::Gcp {
                p.status(&InstanceId::new(id.clone()))
                    .map_err(|e| format!("GCP backstop: {e}"))?;
            }
            // Armed: the client exists, the key is there, the handlers are in
            // place. `up` waits for this line before it trusts the backstop.
            if let Some(f) = &armed_file {
                guard::report_armed(f).map_err(|e| format!("{}: {e}", f.display()))?;
            }
            eprintln!("[backstop] armed (pid {})", std::process::id());
            run_backstop(
                p.as_ref(),
                &InstanceId::new(id),
                deadline,
                clock,
                &StderrLog,
            )
            .map(|_| true)
            .map_err(|e| e.to_string())
        }
    }
}

/// The result lines (the scripts' stdout).
struct StdoutLog;

impl LogSink for StdoutLog {
    fn line(&self, s: &str) {
        println!("{s}");
    }
}

/// Runs a templates / release / reconcile command on the real seams
/// (`offline`: no request is made).
fn with_release(
    root: Option<&PathBuf>,
    offline: bool,
    f: impl FnOnce(&release::Ctx<'_>) -> Result<bool, ReleaseError>,
) -> Result<bool, String> {
    let root = root_of(root)?;
    let repo = Repo::load(&root).map_err(|e| e.to_string())?;
    let env = BootEnv::from_process();
    let http: Box<dyn Http> = if offline {
        Box::new(NoHttp)
    } else {
        Box::new(ReqwestHttp::new().map_err(|e| e.to_string())?)
    };
    let clock = RealClock;
    let ctx = release::Ctx::new(
        &repo,
        http.as_ref(),
        &RealSpawner,
        &clock,
        &env,
        &StderrLog,
        &StdoutLog,
    );
    f(&ctx).map_err(|e| e.to_string())
}

/// The channel a template sync names: `--channel`, else `FV_RELEASE_CHANNEL`.
fn template_channel(flag: Option<String>, ctx: &release::Ctx<'_>) -> Option<String> {
    flag.or_else(|| ctx.env.get("FV_RELEASE_CHANNEL").map(str::to_owned))
        .filter(|c| !c.is_empty())
}

fn cmd_templates(cmd: TemplatesCmd, root: Option<&PathBuf>) -> Result<bool, String> {
    let offline = matches!(cmd, TemplatesCmd::Plan { .. });
    with_release(root, offline, |ctx| match cmd {
        TemplatesCmd::Sync {
            image,
            reference,
            channel,
        } => {
            let api = ctx.runpod()?;
            let ch = template_channel(channel, ctx);
            templates::sync(ctx, &api, &image, &reference, ch.as_deref()).map(|()| true)
        }
        TemplatesCmd::SyncAll { file, channel } => {
            let text = std::fs::read_to_string(&file)
                .map_err(|e| ReleaseError::Refused(format!("{}: {e}", file.display())))?;
            let api = ctx.runpod()?;
            let ch = template_channel(channel, ctx);
            templates::sync_all(ctx, &api, &text, ch.as_deref()).map(|()| true)
        }
        TemplatesCmd::List => {
            for l in templates::list(&ctx.runpod()?)? {
                println!("{l}");
            }
            Ok(true)
        }
        TemplatesCmd::Id { image, flavour } => {
            let f = Flavour::parse(&flavour)
                .ok_or_else(|| ReleaseError::Refused(format!("flavour {flavour}: sls or pod")))?;
            let api = ctx.runpod()?;
            let t = templates::by_name(
                &api,
                &fastvideo_deploy::provider::runpod::serve_template_name(&image, f),
            )?;
            if let Some(id) = t.as_ref().and_then(|t| t.get("id")).and_then(Value::as_str) {
                println!("{id}");
            }
            Ok(true)
        }
        TemplatesCmd::Plan {
            image,
            reference,
            channel,
        } => {
            let img = ctx
                .image(&image)
                .cloned()
                .ok_or_else(|| ReleaseError::Refused(format!("unknown serve image {image}")))?;
            let ch = template_channel(channel, ctx);
            for &f in img.flavours() {
                print_json(&templates::body(
                    ctx,
                    &image,
                    f,
                    &reference,
                    ch.as_deref(),
                    true,
                )?);
            }
            Ok(true)
        }
    })
}

fn release_options(f: &ReleaseFlags, to: Option<i64>) -> promote::Options {
    promote::Options {
        dry_run: f.dry_run,
        mode: if f.local {
            Some(promote::Mode::Local)
        } else if f.dispatch {
            Some(promote::Mode::Dispatch)
        } else {
            None
        },
        notes: f.notes.clone(),
        templates: !f.no_templates,
        allow_partial: f.allow_partial,
        force: f.force,
        to,
    }
}

fn cmd_release(cmd: ReleaseCmd, root: Option<&PathBuf>) -> Result<bool, String> {
    with_release(root, false, |ctx| match cmd {
        ReleaseCmd::Promote {
            target,
            channel,
            flags,
        } => {
            let ch = channel
                .or_else(|| flags.channel_flag.clone())
                .unwrap_or_else(|| "stable".into());
            promote::promote(ctx, &target, &ch, &release_options(&flags, None)).map(|()| true)
        }
        ReleaseCmd::Rollback { channel, to, flags } => {
            let ch = channel
                .or_else(|| flags.channel_flag.clone())
                .unwrap_or_else(|| "stable".into());
            promote::rollback(ctx, &ch, &release_options(&flags, to)).map(|()| true)
        }
        ReleaseCmd::Resolve { target } => {
            print_json(&promote::resolve(ctx, &target)?.to_json());
            Ok(true)
        }
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => fail(e),
    }
}
