//! `fv-deploy`: one instance of a deployment preset on a provider, with the
//! shared guards (docs/adr/0018-fv-deploy.md, docs/ops/fv-deploy.md).
//!
//! ```text
//! fv-deploy plan  --family F --preset P [--region eu] [--kind pod|serverless] [options]
//!                 the Launch and the exact provider requests; no network
//! fv-deploy up    … --yes [--no-smoke] [--keep]
//!                 balance floor, price cap, create, backstop, ready (replace
//!                 after 600 s, at most 3 cycles), smoke, delete (unless --keep);
//!                 without --yes it is `plan`
//! fv-deploy status <id> | logs <id> [--follow] | smoke <id> | down <id>  [--kind …]
//! fv-deploy schema launch|instance|offer|account
//! fv-deploy backstop --kind K --id ID --deadline T --armed-file F
//!                 (started by `up`, detached; writes F once armed)
//! ```
//!
//! The provider's key comes from `RUNPOD_API_KEY` and is never printed.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};

use fastvideo_deploy::boot::env::BootEnv;
use fastvideo_deploy::boot::http::ReqwestHttp;
use fastvideo_deploy::boot::sys::{Clock, Http, LogSink, RealClock, StderrLog};
use fastvideo_deploy::guard::{self, run_backstop, DetachedBackstop};
use fastvideo_deploy::launch::{self, PresetRequest, Repo, KEY_HASH_PLACEHOLDER};
use fastvideo_deploy::lifecycle::{smoke, Deployer, Ledger, SmokeOptions, UpOptions};
use fastvideo_deploy::provider::runpod::{Pod, RunpodApi, Serverless};
use fastvideo_deploy::provider::{
    schema, Cursor, Guards, Instance, InstanceId, Provider, ProviderKind, ReachOptions,
};

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
    /// pod or serverless.
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
    #[arg(long, default_value_t = guard::MAX_USD_PER_HR)]
    max_usd_per_hr: f64,
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
    /// pod or serverless (default: what `up` recorded, else pod).
    #[arg(long)]
    kind: Option<String>,
    #[arg(long)]
    state_dir: Option<PathBuf>,
    #[arg(long)]
    root: Option<PathBuf>,
}

fn fail(msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("[fv-deploy] FAILED: {msg}");
    ExitCode::FAILURE
}

fn kind_of(s: &str) -> Result<ProviderKind, String> {
    ProviderKind::parse(s).ok_or_else(|| format!("unknown kind {s} (pod, serverless)"))
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

fn request(a: &LaunchArgs, key_hash: &str) -> Result<PresetRequest, String> {
    Ok(PresetRequest {
        family: a.family.clone(),
        preset: a.preset.clone(),
        region: a.region.clone(),
        kind: kind_of(&a.kind)?,
        image: a.image.clone(),
        channel: a.channel.clone(),
        gpu_types: a.gpus.clone(),
        volume: a.volume.clone(),
        guards: Guards {
            max_usd_per_hr: a.max_usd_per_hr,
            min_balance_usd: a.min_balance,
            cap_s: a.cap_s,
            ready_timeout_s: a.ready_timeout_s,
            ready_cycles: a.ready_cycles,
            force: a.force_guards,
        },
        key_hash: key_hash.to_owned(),
    })
}

fn provider<'a>(kind: ProviderKind, api: RunpodApi<'a>) -> Box<dyn Provider + 'a> {
    match kind {
        ProviderKind::RunpodPod => Box::new(Pod::new(api)),
        ProviderKind::RunpodServerless => Box::new(Serverless::new(api)),
    }
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
    let clock = RealClock;
    let l = launch::from_preset(
        &repo,
        &request(a, KEY_HASH_PLACEHOLDER)?,
        &launch::utc_stamp(clock.now()),
    )
    .map_err(|e| e.to_string())?;
    guard::check_guards(&l.guards).map_err(|e| e.to_string())?;
    guard::check_region(&l).map_err(|e| e.to_string())?;
    // No key, no network: the payload is the provider code's own builder.
    let http = NoHttp;
    let api = RunpodApi::new(
        &http,
        &clock,
        "",
        fastvideo_deploy::provider::runpod::RunpodUrls::from_env(&BootEnv::from_process()),
    );
    let p = provider(l.provider, api);
    let calls = p.payload(&l).map_err(|e| e.to_string())?;
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
    let env = BootEnv::from_process();
    let http = ReqwestHttp::new().map_err(|e| e.to_string())?;
    let clock = RealClock;
    let log = StderrLog;
    let api = RunpodApi::from_env(&http, &clock, &env).map_err(|e| e.to_string())?;
    let key = launch::run_key().map_err(|e| e.to_string())?;
    let stamp = launch::utc_stamp(clock.now());
    let mut l = launch::from_preset(&repo, &request(&u.launch, &launch::key_hash(&key))?, &stamp)
        .map_err(|e| e.to_string())?;
    let pinned = launch::ghcr_resolve(&http, &l.image).map_err(|e| e.to_string())?;
    launch::pin_image(&mut l, &pinned);
    log.line(&format!(
        "[fv-deploy] {} {} on {}: image {pinned}",
        l.provider,
        l.name,
        l.gpu.types.join(", ")
    ));
    // The env guard's test: never send the key in a launch (it is only a header).
    let api_key = env
        .get(fastvideo_deploy::provider::runpod::API_KEY_VAR)
        .unwrap_or("")
        .to_owned();
    let leaks = |s: &str| !api_key.is_empty() && s.contains(api_key.as_str());
    let p = provider(l.provider, api);
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let backstop = DetachedBackstop::new(
        exe,
        Some(state.join(format!("backstop-{stamp}.log"))),
        state.join("armed"),
    );
    std::fs::create_dir_all(&state).map_err(|e| format!("{}: {e}", state.display()))?;
    let ledger = Ledger {
        path: state.join("ledger.tsv"),
    };
    let flag = interrupt_flag();
    let interrupted = || flag.load(Ordering::SeqCst);
    let d = Deployer {
        provider: p.as_ref(),
        clock: &clock,
        log: &log,
        backstop: &backstop,
        ledger: Some(&ledger),
        interrupted: &interrupted,
        leaks: &leaks,
        poll: Duration::from_secs(5),
    };
    let opts = UpOptions {
        run_key: (l.provider == ProviderKind::RunpodPod).then(|| key.clone()),
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

fn with_provider<T>(
    a: &IdArgs,
    f: impl FnOnce(&dyn Provider, Instance, &Path, &RealClock) -> Result<T, String>,
) -> Result<T, String> {
    let state = state_dir(a.state_dir.as_ref(), a.root.as_ref());
    let rec = recorded(&state, &a.id);
    let kind = id_kind(a.kind.as_deref(), rec.as_ref().map(|r| r.provider), &a.id)?;
    let env = BootEnv::from_process();
    let http = ReqwestHttp::new().map_err(|e| e.to_string())?;
    let clock = RealClock;
    let api = RunpodApi::from_env(&http, &clock, &env).map_err(|e| e.to_string())?;
    let url = match kind {
        ProviderKind::RunpodPod => Some(api.urls.pod_url(&a.id)),
        ProviderKind::RunpodServerless => None,
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
    let p = provider(kind, api);
    f(p.as_ref(), inst, &state, &clock)
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
            let d = Deployer {
                provider: p,
                clock,
                log: &StderrLog,
                backstop: &none,
                ledger: Some(&ledger),
                interrupted: &never,
                leaks: &leaks,
                poll: Duration::from_secs(5),
            };
            d.down(&inst.id).map_err(|e| e.to_string())?;
            println!("deleted {}", inst.id);
            Ok(true)
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
            let env = BootEnv::from_process();
            let http = ReqwestHttp::new().map_err(|e| e.to_string())?;
            let clock = RealClock;
            let api = RunpodApi::from_env(&http, &clock, &env).map_err(|e| e.to_string())?;
            let p = provider(kind, api);
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
                &clock,
                &StderrLog,
            )
            .map(|_| true)
            .map_err(|e| e.to_string())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => fail(e),
    }
}
