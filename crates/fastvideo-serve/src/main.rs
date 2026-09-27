//! `fv-serve` binary (design §6.1): `fv-serve [--config /etc/fv/serve.toml]`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use fastvideo_serve::config::{Config, ProcessEnv};
use fastvideo_serve::{App, Overrides};

#[derive(Debug, Parser)]
#[command(name = "fv-serve", version, about = "fastvideo-rs serving: FastVideo, MiniMax, fal, LTX and Reactor APIs")]
struct Args {
    /// TOML config (docs/serve/design.md §6.1, configs/serve/).
    #[arg(long, env = "FV_CONFIG")]
    config: Option<PathBuf>,
    /// Print the effective config (secrets redacted) and exit.
    #[arg(long)]
    print_config: bool,
}

fn init_tracing(c: &Config) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_new(&c.log.filter).unwrap_or_else(|_| EnvFilter::new("info"));
    let b = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr);
    if c.log.format == "json" {
        b.json().init();
    } else {
        b.init();
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config = match Config::load(args.config.as_deref(), &ProcessEnv) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fv-serve: config: {e}");
            return ExitCode::from(2);
        }
    };
    if args.print_config {
        println!("{}", serde_json::to_string_pretty(&config.redacted()).unwrap_or_default());
        return ExitCode::SUCCESS;
    }
    init_tracing(&config);
    let rt = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("fv-serve: tokio runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let res = rt.block_on(async move {
        tracing::info!(config = %config.redacted(), "fv-serve {}", env!("CARGO_PKG_VERSION"));
        let addr = config.bind_addr()?;
        let app = App::build(config, Overrides::default()).await?;
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!(%addr, "listening");
        app.serve(listener, fastvideo_serve::shutdown::signal()).await
    });
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %format!("{e:#}"), "fv-serve failed");
            eprintln!("fv-serve: {e:#}");
            ExitCode::FAILURE
        }
    }
}
