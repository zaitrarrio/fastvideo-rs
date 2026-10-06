//! Deploy glue (WP-16, design §6.2-§6.4): the Runpod queue mode, the Vast
//! forwarder route and the diagnostics `info` job, over `fastvideo-deploy`.
//!
//! - `server.mode = runpod-queue` ([`run_runpod_queue`]): no HTTP listener;
//!   the Rust worker loop takes jobs from `RUNPOD_WEBHOOK_GET_JOB` and
//!   dispatches their native envelope into the same router the HTTP mode
//!   serves. Models load before the first take; a load failure fails one job
//!   with the reason and exits 1 (design §6.2). SIGTERM stops taking, lets
//!   the running job finish within `shutdown_grace_s`, then drains.
//!   Without `RUNPOD_WEBHOOK_GET_JOB` (local), `FV_RUNPOD_TEST_INPUT`
//!   (a JSON envelope, or `@file`) runs once and prints the output.
//! - `server.forward = true` mounts `POST /fv/v1/forward` (Vast PyWorker).

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context};
use axum::Router;
use fastvideo_deploy::dispatch::{InfoFn, RouterHandler};
use fastvideo_deploy::env::Discovery;
use fastvideo_deploy::runpod::{InfoJob, JobCtx, JobHandler, JobInput};
use fastvideo_engine_service::Readiness;
use serde_json::{json, Value};

use crate::app::App;
use crate::config::Config;
use crate::gate::ServiceGate;

/// When this process started (for cold-start reports).
#[derive(Clone, Copy, Debug)]
pub struct Boot {
    pub at: Instant,
    pub unix: f64,
}

impl Boot {
    pub fn now() -> Self {
        let unix = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        Self { at: Instant::now(), unix }
    }
}

/// Where pods mount the weight volume; its HF-cache trees link absolutely
/// under this path.
pub const POD_WEIGHTS: &str = "/workspace/weights";

/// Links `alias` → `root` when the weights root is elsewhere (a Runpod
/// serverless worker mounts the volume at `/runpod-volume`) and `alias` does
/// not exist, so the volume's absolute links into `/workspace/weights`
/// resolve. Returns whether it created the link.
pub fn link_weights_alias(root: &Path, alias: &Path) -> std::io::Result<bool> {
    if root == alias || !root.is_dir() || alias.symlink_metadata().is_ok() {
        return Ok(false);
    }
    if let Some(parent) = alias.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(root, alias)?;
        Ok(true)
    }
    #[cfg(not(unix))]
    Ok(false)
}

/// [`link_weights_alias`] for `FV_WEIGHTS` and [`POD_WEIGHTS`] (logged).
pub fn link_pod_weights() {
    let Some(root) = std::env::var("FV_WEIGHTS").ok().filter(|s| !s.is_empty()) else { return };
    match link_weights_alias(Path::new(&root), Path::new(POD_WEIGHTS)) {
        Ok(true) => tracing::info!(%root, "linked {POD_WEIGHTS} to the weights root (absolute volume links)"),
        Ok(false) => {}
        Err(e) => tracing::warn!(%root, error = %e, "could not link {POD_WEIGHTS} to the weights root"),
    }
}

fn http_port(c: &Config) -> u16 {
    c.bind_addr().map(|a| a.port()).unwrap_or(8000)
}

fn weights_summary() -> Value {
    let Some(root) = std::env::var("FV_WEIGHTS").ok().filter(|s| !s.is_empty()) else {
        return json!({"root": null});
    };
    let p = Path::new(&root);
    let entries: Vec<String> = std::fs::read_dir(p)
        .map(|rd| {
            let mut v: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
            v.sort();
            v.truncate(64);
            v
        })
        .unwrap_or_default();
    json!({"root": root, "exists": p.is_dir(), "entries": entries})
}

/// Encodes a few black frames with `h264_nvenc` (the check
/// `fastvideo_media::video::nvenc_available` makes) and keeps ffmpeg's
/// error text when it fails, so a host without a working NVENC says why.
fn nvenc_probe() -> Value {
    let o = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x144:r=24:d=0.2"])
        .args(["-c:v", "h264_nvenc", "-f", "null", "-"])
        .stdin(std::process::Stdio::null())
        .output();
    match o {
        Ok(o) if o.status.success() => json!({"ok": true}),
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let tail: Vec<&str> = err.lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect();
            let libs: Vec<String> = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/nvidia/lib64"]
                .iter()
                .filter_map(|d| std::fs::read_dir(d).ok())
                .flat_map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()))
                .filter(|n| n.starts_with("libnvidia-encode") || n.starts_with("libnvcuvid"))
                .collect();
            json!({"ok": false, "stderr": tail, "driver_libs": libs})
        }
        Err(e) => json!({"ok": false, "stderr": [e.to_string()]}),
    }
}

fn cmd_line(prog: &str, args: &[&str]) -> Option<String> {
    let o = std::process::Command::new(prog).args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}

/// The startup NVENC probe behind `auto`, if it ran.
fn startup_probe() -> Value {
    match fastvideo_media::video::auto_encoder_if_probed() {
        Some(a) => json!({"backend": a.backend, "attempts": a.attempts, "nvenc_error": a.nvenc_error}),
        None => Value::Null,
    }
}

/// The `info` job / diagnostics: version, platform discovery, readiness,
/// weights root, GPU, NVENC (`nvenc: true` also encodes a few frames).
pub fn info_fn(config: &Config, gate: Arc<ServiceGate>, boot: Boot, ready_after: Arc<std::sync::OnceLock<f64>>) -> InfoFn {
    let port = http_port(config);
    let (udp, tcp) = (config.webrtc.udp_port, config.webrtc.tcp_port);
    let engine = format!("{:?}", config.engine.backend).to_ascii_lowercase();
    let jobs = format!("{:?}", config.job_backend()).to_ascii_lowercase();
    let artifacts = format!("{:?}", config.artifact_backend()).to_ascii_lowercase();
    let webhook_key = !config.webhook_key.is_empty();
    // The encoders in use (after `auto` was resolved at startup) and, when
    // any setting was `auto`, what the startup probe found.
    let encoders = json!({
        "director": config.director.encoder,
        "reactor": config.reactor.h264,
        "streams": config.webrtc.encoder,
        "post": config.engine.post_encoder,
    });
    Arc::new(move |job: InfoJob| {
        let gate = gate.clone();
        let (engine, jobs, artifacts) = (engine.clone(), jobs.clone(), artifacts.clone());
        let ready_after = ready_after.clone();
        let encoders = encoders.clone();
        Box::pin(async move {
            let blocking = tokio::task::spawn_blocking(move || {
                let encoders = cmd_line("ffmpeg", &["-hide_banner", "-encoders"]).unwrap_or_default();
                let gpu = cmd_line("nvidia-smi", &["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]);
                let nvenc_encode = job.nvenc.then(nvenc_probe);
                (encoders.contains("h264_nvenc"), gpu, nvenc_encode, weights_summary())
            })
            .await;
            let (nvenc_built, gpu, nvenc_encode, weights) = blocking.unwrap_or((false, None, None, Value::Null));
            let nvenc_ok = nvenc_encode.as_ref().map(|v| v["ok"] == true);
            let readiness = match gate.engine().readiness() {
                Readiness::Ready => json!("ready"),
                Readiness::Loading { done, total } => json!({"loading": [done, total]}),
                Readiness::Failed(e) => json!({"failed": e}),
            };
            let models: Vec<String> = gate.engine().caps().models().map(|m| m.id.0.clone()).collect();
            json!({
                "server": "fv-serve",
                "version": crate::build_info::VERSION,
                "runtime": fastvideo_deploy::runpod::version(),
                "process_start_unix": boot.unix,
                "uptime_s": boot.at.elapsed().as_secs_f64(),
                "ready_after_s": ready_after.get(),
                "readiness": readiness,
                "models": models,
                "engine": engine,
                "jobs_backend": jobs,
                "artifacts_backend": artifacts,
                "webhook_key_configured": webhook_key,
                "deploy": Discovery::from_process().summary(port, udp, tcp),
                "weights": weights,
                "gpu": gpu,
                "ffmpeg_h264_nvenc": nvenc_built,
                "nvenc_encode_ok": nvenc_ok,
                "nvenc_probe": nvenc_encode,
                "encoders": encoders,
                "encoder_startup_probe": startup_probe(),
                "nvidia_driver_capabilities": std::env::var("NVIDIA_DRIVER_CAPABILITIES").ok(),
            })
        })
    })
}

/// The dispatcher over the app's router.
pub fn handler(app: &App, boot: Boot, ready_after: Arc<std::sync::OnceLock<f64>>) -> RouterHandler {
    let h = RouterHandler::new(app.router.clone()).with_info(info_fn(&app.config, app.gate.clone(), boot, ready_after));
    // A queue worker behind the gateway: queue jobs are authenticated by
    // the platform, so the dispatcher adds the internal token itself (it
    // never travels in the job input, docs/serve/gateway.md §3).
    if app.config.server.role == crate::config::Role::Worker {
        h.with_header("x-fv-internal-token", app.config.gateway.internal_token.expose())
    } else {
        h
    }
}

/// Adds `POST /fv/v1/forward` (Vast PyWorker target) to a built router.
pub fn with_forward(router: Router, handler: RouterHandler) -> Router {
    router.merge(fastvideo_deploy::dispatch::forward_routes::<()>(handler))
}

fn test_input() -> anyhow::Result<Option<Value>> {
    let Some(v) = std::env::var("FV_RUNPOD_TEST_INPUT").ok().filter(|s| !s.trim().is_empty()) else { return Ok(None) };
    let text = match v.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?,
        None => v,
    };
    let j: Value = serde_json::from_str(&text).context("FV_RUNPOD_TEST_INPUT is not JSON")?;
    Ok(Some(j.get("input").cloned().unwrap_or(j)))
}

/// Runs `server.mode = runpod-queue` until `stop` resolves.
pub async fn run_runpod_queue(app: App, boot: Boot, stop: impl std::future::Future<Output = ()> + Send + 'static) -> anyhow::Result<()> {
    let ready_after = Arc::new(std::sync::OnceLock::new());
    let h = handler(&app, boot, ready_after.clone());
    let gate = app.gate.clone();
    let grace = app.config.shutdown_grace();
    let d1 = app.d1.clone();
    let env = match fastvideo_deploy::runpod::RunpodEnv::from_process() {
        Ok(e) => e,
        Err(why) => {
            // Local mode: one job from FV_RUNPOD_TEST_INPUT.
            let Some(input) = test_input()? else {
                return Err(anyhow!("server.mode = runpod-queue: {why}; set FV_RUNPOD_TEST_INPUT to run one job locally"));
            };
            let r = gate.engine().wait_ready().await;
            if let Readiness::Failed(e) = r {
                return Err(anyhow!("model loading failed: {e}"));
            }
            let (cx, _rx) = JobCtx::detached("local-test");
            let input = JobInput::parse(&input).map_err(|e| anyhow!(e))?;
            let out = h.handle(input, cx).await;
            let v = match out {
                Ok(v) => json!({"output": v}),
                Err(e) => json!({"error": e.to_string(), "output": e.output}),
            };
            println!("{}", serde_json::to_string_pretty(&v)?);
            crate::app::drain(&gate, grace, d1.as_deref()).await;
            return Ok(());
        }
    };
    tracing::info!(worker = %env.worker_id, "runpod-queue: waiting for models before taking jobs");
    let run = run_worker(env, h, gate.clone(), grace, boot, ready_after, stop).await;
    crate::app::drain(&gate, grace, d1.as_deref()).await;
    run
}

#[cfg(feature = "http-client")]
async fn run_worker(
    env: fastvideo_deploy::runpod::RunpodEnv,
    h: RouterHandler,
    gate: Arc<ServiceGate>,
    grace: Duration,
    boot: Boot,
    ready_after: Arc<std::sync::OnceLock<f64>>,
    stop: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    use fastvideo_deploy::runpod::{ReqwestTransport, Worker, WorkerOptions};
    let transport = ReqwestTransport::new().map_err(|e| anyhow!("HTTP client: {e}"))?;
    let opts = WorkerOptions { shutdown_grace: grace, ..WorkerOptions::default() };
    let worker = Worker::new(env, transport, h, opts);
    match gate.engine().wait_ready().await {
        Readiness::Failed(e) => {
            tracing::error!(error = %e, "model loading failed: failing one job, then exiting");
            let r = worker.fail_one(&format!("model loading failed: {e}"), Duration::from_secs(120)).await;
            tracing::info!(?r, "prestart failure reported");
            Err(anyhow!("model loading failed: {e}"))
        }
        _ => {
            let secs = boot.at.elapsed().as_secs_f64();
            let _ = ready_after.set(secs);
            let ids: Vec<String> = gate.engine().caps().models().map(|m| m.id.0.clone()).collect();
            println!("FV-SERVE READY models={}", ids.join(","));
            tracing::info!(ready_after_s = secs, "runpod worker: ready");
            let r = worker.run(stop).await;
            tracing::info!(?r, "runpod worker stopped");
            Ok(())
        }
    }
}

#[cfg(not(feature = "http-client"))]
async fn run_worker(
    _env: fastvideo_deploy::runpod::RunpodEnv,
    _h: RouterHandler,
    _gate: Arc<ServiceGate>,
    _grace: Duration,
    _boot: Boot,
    _ready_after: Arc<std::sync::OnceLock<f64>>,
    _stop: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    Err(anyhow!("server.mode = runpod-queue needs fv-serve built with `http-client`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn weights_alias_links_once() {
        let base = tempfile::Builder::new().prefix("fv-alias-").tempdir().unwrap().keep();
        let root = base.join("runpod-volume/weights");
        std::fs::create_dir_all(root.join("h3-base")).unwrap();
        let alias = base.join("workspace/weights");
        assert!(link_weights_alias(&root, &alias).unwrap());
        assert!(alias.join("h3-base").is_dir());
        // Existing alias, same path, or a missing root: nothing to do.
        assert!(!link_weights_alias(&root, &alias).unwrap());
        assert!(!link_weights_alias(&root, &root).unwrap());
        assert!(!link_weights_alias(&base.join("missing"), &base.join("other")).unwrap());
        std::fs::remove_dir_all(&base).unwrap();
    }
}
