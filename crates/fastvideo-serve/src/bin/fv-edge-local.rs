//! `fv-edge-local`: the edge on the host (`fastvideo_serve::edge_host`) for
//! local runs and the compat suites (`FV_COMPAT_EDGE=1`). Configured from
//! the environment:
//!
//! | variable | meaning |
//! |---|---|
//! | `FV_EDGE_BIND` | listen address (default `127.0.0.1:8787`) |
//! | `FV_INTERNAL_TOKEN` | the workers' shared secret (required) |
//! | `FV_ADMIN_TOKEN` | the admin token |
//! | `FV_API_KEYS` | SHA-256 list of static keys |
//! | `FV_KEY_STORE` + the D1 set (`FV_CF_ACCOUNT_ID`, `FV_CF_API_TOKEN`, `FV_D1_DATABASE_ID`, `FV_D1_API_BASE`) | minted keys, as fv-serve reads them |
//! | `FV_EDGE_AUTH` (`keys` \| `none`) | `none`: every caller anonymous |
//! | `FV_REACTOR_MODEL` | the Reactor model when several fronts have one |
//! | `FV_EDGE_WHIP` (`proxy` \| `redirect`) | WHIP ingest offers: proxied (default) or a 307 to the worker with a session capability |
//! | `FV_EDGE_KEY_RPM`, `FV_EDGE_KEY_IN_FLIGHT` | quotas (0: none) |
//!
//! Prints `FV-EDGE READY <base URL>` once listening.


use fastvideo_dispatch_proto::front::Quotas;
use fastvideo_serve::config::{Config, ProcessEnv};
use fastvideo_serve::edge_host::{EdgeHost, EdgeHostCfg};
use fastvideo_serve_kit::KeyRing;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).init();
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let mut c = Config::default();
    c.apply_env(&ProcessEnv).map_err(|e| anyhow::anyhow!("{e}"))?;
    let token = var("FV_INTERNAL_TOKEN").ok_or_else(|| anyhow::anyhow!("FV_INTERNAL_TOKEN is required"))?;
    let keys = fastvideo_serve::storage::build_key_store(&c, None).await.map_err(|e| anyhow::anyhow!(e))?;
    let mut cfg = EdgeHostCfg::new(token, keys);
    cfg.admin_token = var("FV_ADMIN_TOKEN");
    if let Some(k) = var("FV_API_KEYS") {
        cfg.static_keys = KeyRing::from_hash_list(&k).map_err(|e| anyhow::anyhow!(e))?;
    }
    cfg.auth_none = var("FV_EDGE_AUTH").as_deref() == Some("none");
    cfg.reactor_model = var("FV_REACTOR_MODEL");
    cfg.whip_redirect = var("FV_EDGE_WHIP").as_deref() == Some("redirect");
    let n = |k: &str, d: u32| var(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let q = Quotas::default();
    cfg.quotas = Quotas { key_rpm: n("FV_EDGE_KEY_RPM", q.key_rpm), key_in_flight: n("FV_EDGE_KEY_IN_FLIGHT", q.key_in_flight), invalid_key_rpm: q.invalid_key_rpm };
    let bind = var("FV_EDGE_BIND").unwrap_or_else(|| "127.0.0.1:8787".into());
    let l = tokio::net::TcpListener::bind(&bind).await?;
    let (_h, base) = EdgeHost::start(cfg, l).await;
    println!("FV-EDGE READY {base}");
    tokio::signal::ctrl_c().await?;
    Ok(())
}
