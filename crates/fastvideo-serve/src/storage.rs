//! Job and artifact stores from config (design §0 decision 7, §3.4, §6.2).
//!
//! - Jobs: `memory` | `file` (serve-kit `MemJobStore` with manifests) | `d1`
//!   (`D1JobStore` over the Cloudflare D1 HTTP API; needs `http-client`).
//! - Artifacts: `local` (signed `/files` URLs) | `s3` (Cloudflare R2 or any
//!   S3-compatible bucket, presigned URLs; uploads need `http-client`).

use std::sync::Arc;
use std::time::Duration;

use fastvideo_protocol::JobStore;
use fastvideo_serve_kit::artifacts::{LocalUrls, S3Config};
use fastvideo_serve_kit::{ArtifactStore, D1Config, D1Options, LocalArtifactStore, MemJobStore, S3ArtifactStore, UrlKey};
use time::OffsetDateTime;
use url::Url;

use crate::config::{ArtifactBackend, Config, JobBackend};

/// What `build_jobs` produced (the D1 store is kept for flushing at shutdown).
pub struct Jobs {
    pub store: Arc<dyn JobStore>,
    pub d1: Option<Arc<fastvideo_serve_kit::D1JobStore>>,
    pub kind: JobBackend,
}

impl std::fmt::Debug for Jobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jobs").field("kind", &self.kind).finish_non_exhaustive()
    }
}

/// The `[artifacts.s3]` block as a serve-kit `S3Config`.
pub fn s3_config(c: &Config) -> Result<S3Config, String> {
    let s = &c.artifacts.s3;
    let endpoint = s.endpoint.as_deref().ok_or("artifacts.s3.endpoint is not set")?;
    let endpoint = Url::parse(endpoint).map_err(|e| format!("artifacts.s3.endpoint: {e}"))?;
    Ok(S3Config {
        endpoint,
        region: s.region.clone().unwrap_or_else(|| "auto".into()),
        bucket: s.bucket.clone().ok_or("artifacts.s3.bucket is not set")?,
        access_key: s.access_key_id.expose().to_owned(),
        secret_key: s.secret_access_key.expose().to_owned(),
        path_style: s.path_style.unwrap_or(true),
        prefix: s.prefix.clone().unwrap_or_else(|| "fv-serve/".into()),
    })
}

/// The artifact store (and the URL key used for `/files`).
pub fn build_artifacts(c: &Config, public_base: &Url, key: &UrlKey) -> Result<Arc<dyn ArtifactStore>, String> {
    Ok(match c.artifact_backend() {
        ArtifactBackend::S3 => {
            if !cfg!(feature = "http-client") {
                return Err("S3/R2 artifacts need fv-serve built with `http-client`".into());
            }
            Arc::new(S3ArtifactStore::new(s3_config(c)?))
        }
        _ => Arc::new(LocalArtifactStore::new(
            c.server.state_dir.join("artifacts"),
            LocalUrls { public_base: public_base.clone(), key: key.clone() },
        )),
    })
}

/// The D1 connection settings.
pub fn d1_config(c: &Config) -> Result<D1Config, String> {
    let d = &c.jobs.d1;
    let mut cfg = D1Config::new(
        d.account_id.clone().ok_or("jobs.d1.account_id (FV_CF_ACCOUNT_ID) is not set")?,
        d.api_token.expose(),
        d.database_id.clone().ok_or("jobs.d1.database_id (FV_D1_DATABASE_ID) is not set")?,
    );
    if let Some(b) = &d.api_base {
        cfg.api_base = b.clone();
    }
    Ok(cfg)
}

/// `D1Options` from config.
pub fn d1_options(c: &Config, worker: &str) -> D1Options {
    let mut o = D1Options::new(worker);
    o.progress_interval = Duration::from_millis(c.jobs.progress_interval_ms.max(1));
    o.stale_after = (c.jobs.stale_after_s > 0).then(|| Duration::from_secs(c.jobs.stale_after_s));
    o
}

/// The job store.
pub async fn build_jobs(c: &Config, worker: &str, artifacts: Arc<dyn ArtifactStore>) -> Result<Jobs, String> {
    let kind = c.job_backend();
    let inputs = c.server.state_dir.join("inputs");
    match kind {
        JobBackend::Memory => Ok(Jobs {
            store: Arc::new(MemJobStore::memory().with_artifacts(artifacts).with_inputs_root(inputs)),
            d1: None,
            kind,
        }),
        JobBackend::D1 => {
            #[cfg(feature = "http-client")]
            {
                let client = fastvideo_serve_kit::D1Client::http(d1_config(c)?).map_err(|e| e.to_string())?;
                let store = fastvideo_serve_kit::D1JobStore::new(client, d1_options(c, worker))
                    .with_artifacts(artifacts)
                    .with_inputs_root(inputs)
                    .open(OffsetDateTime::now_utc())
                    .await
                    .map_err(|e| format!("opening the D1 job store: {e}"))?;
                Ok(Jobs { store: store.clone(), d1: Some(store), kind })
            }
            #[cfg(not(feature = "http-client"))]
            {
                let _ = (worker, artifacts, inputs);
                Err("jobs.backend = d1 needs fv-serve built with `http-client`".into())
            }
        }
        JobBackend::File | JobBackend::Auto => {
            let s = MemJobStore::open(c.server.state_dir.join("jobs"), OffsetDateTime::now_utc())
                .await
                .map_err(|e| format!("opening the job store: {e}"))?;
            Ok(Jobs {
                store: Arc::new(s.with_artifacts(artifacts).with_inputs_root(inputs)),
                d1: None,
                kind: JobBackend::File,
            })
        }
    }
}
