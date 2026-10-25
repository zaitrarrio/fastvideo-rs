//! `fv-serve` binary (design §6.1): `fv-serve [--config /etc/fv/serve.toml]`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{CommandFactory, FromArgMatches, Parser};
use fastvideo_serve::build_info::BuildInfo;
use fastvideo_serve::config::{Config, ProcessEnv};
use fastvideo_serve::{App, Overrides};

#[derive(Debug, Parser)]
#[command(
    name = "fv-serve",
    version,
    about = "fastvideo-rs serving: FastVideo, MiniMax, fal, LTX and Reactor APIs"
)]
struct Args {
    /// TOML config (docs/serve/design.md §6.1, configs/serve/).
    #[arg(long, env = "FV_CONFIG")]
    config: Option<PathBuf>,
    /// Print the effective config (secrets redacted) and exit.
    #[arg(long)]
    print_config: bool,
}

/// The subscriber: the `[log]` filter, log shipping, the OpenTelemetry
/// layer (feature `otel`, `[otel] enabled`) and the stderr formatter. `Err`
/// is an `[otel]` exporter that could not be built.
fn init_tracing(c: &Config) -> Result<(), String> {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_new(&c.log.filter).unwrap_or_else(|_| EnvFilter::new("info"));
    // Log shipping to fv-control (FV_LOG_SHIP_URL; off by default, src/log_ship.rs).
    let ship = fastvideo_serve::log_ship::layer();
    // OTLP export of the request traces (src/otel.rs; docs/serve/tracing.md
    // "OpenTelemetry export"): built here, before the runtime, and `None`
    // unless `[otel] enabled`.
    #[cfg(feature = "otel")]
    let otel = fastvideo_serve::otel::init(&c.otel).map_err(|e| e.to_string())?;
    let fmt = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
    let reg = tracing_subscriber::registry().with(filter).with(ship);
    #[cfg(feature = "otel")]
    let reg = reg.with(otel);
    if c.log.format == "json" {
        reg.with(fmt.json()).init();
    } else {
        reg.with(fmt).init();
    }
    #[cfg(not(feature = "otel"))]
    {
        if c.otel.enabled {
            tracing::warn!(
                "[otel] enabled = true, but this build has no `otel` feature: no trace export"
            );
        }
    }
    Ok(())
}

/// `--version` prints the build identity (git sha, build time, variant,
/// image; docs/serve/releases.md), `-V` the package version only.
fn parse_args() -> Args {
    let matches = Args::command()
        .version(fastvideo_serve::build_info::VERSION)
        .long_version(BuildInfo::current_long_version())
        .get_matches();
    Args::from_arg_matches(&matches).unwrap_or_else(|e| e.exit())
}

fn main() -> ExitCode {
    let args = parse_args();
    let config = match Config::load(args.config.as_deref(), &ProcessEnv) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fv-serve: config: {e}");
            return ExitCode::from(2);
        }
    };
    if args.print_config {
        println!(
            "{}",
            serde_json::to_string_pretty(&config.redacted()).unwrap_or_default()
        );
        return ExitCode::SUCCESS;
    }
    if let Err(e) = init_tracing(&config) {
        eprintln!("fv-serve: {e}");
        return ExitCode::from(2);
    }
    for note in &config.deprecations {
        tracing::warn!("config: {note}");
    }
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("fv-serve: tokio runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let res = rt.block_on(async move {
        tracing::info!(config = %config.redacted(), build = %BuildInfo::current().json(), "fv-serve {}", BuildInfo::current().summary());
        fastvideo_serve::log_ship::spawn();
        let boot = fastvideo_serve::deploy::Boot::now();
        fastvideo_serve::deploy::link_pod_weights();
        let queue = config.server.mode == fastvideo_serve::config::Mode::RunpodQueue;
        let addr = config.bind_addr()?;
        if queue {
            let app = App::build(config, Overrides::default()).await?;
            return fastvideo_serve::deploy::run_runpod_queue(app, boot, fastvideo_serve::shutdown::signal()).await;
        }
        // Bind before building, so `/ping` answers 204 (initializing) from
        // the first second instead of the port being closed while the
        // encoder probe, the stores and the engine start.
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!(%addr, "listening (starting)");
        fastvideo_serve::app::serve_while_building(
            listener,
            App::build(config, Overrides::default()),
            fastvideo_serve::shutdown::signal(),
        )
        .await
    });
    // The last spans leave before the process does.
    #[cfg(feature = "otel")]
    fastvideo_serve::otel::shutdown();
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %format!("{e:#}"), "fv-serve failed");
            eprintln!("fv-serve: {e:#}");
            ExitCode::FAILURE
        }
    }
}
