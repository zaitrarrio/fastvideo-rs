//! `fv-deploy templates | release | reconcile` (plan C3d,
//! docs/adr/0018-fv-deploy.md §10): what `scripts/serve/runpod-templates.sh`
//! and `scripts/serve/release.sh` do, on the same seams as the rest of
//! fv-deploy (`Http`, `Spawner`, `Clock`), so every flow runs against fakes.
//!
//! - [`templates`]: the Runpod serve templates (`fv-serve-<image>-sls` /
//!   `-pod`), built by the Runpod provider's template builder
//!   ([`crate::provider::runpod::serve_template_body`]).
//! - [`promote`]: `release promote | rollback | resolve`: channel tags in
//!   GHCR, the templates when the channel is theirs, the release row in D1.
//! - [`reconcile`]: the account's pods / endpoints / templates against the
//!   D1 `deployments` ledger.
//! - [`registry`] / [`d1`] / [`ghcr`]: the ledger (`registry.sh`, `d1.sh`)
//!   and the registry (`ghcr.sh`).
//!
//! The scripts stay until the owner's live gate (§10.6); the request goldens
//! (`fixtures/deploy/release.json`, `tests/release_goldens.rs`) hold the two
//! to the same requests.

pub mod d1;
pub mod ghcr;
pub mod promote;
pub mod reconcile;
pub mod registry;
pub mod templates;

use std::path::PathBuf;

use fastvideo_manifest::deploy::Compute;

use crate::boot::env::BootEnv;
use crate::boot::sys::{Clock, Http, LogSink, Spawner};
use crate::launch::Repo;
use crate::provider::runpod::{Flavour, RunpodApi};
use crate::provider::ProviderError;

/// Why a templates / release / reconcile command stopped (the scripts' `die`).
#[derive(Debug, thiserror::Error)]
pub enum ReleaseError {
    /// Refused or failed, with the reason (the scripts' message).
    #[error("{0}")]
    Refused(String),
    #[error(transparent)]
    D1(#[from] d1::D1Error),
    #[error(transparent)]
    Ghcr(#[from] ghcr::GhcrError),
    #[error(transparent)]
    Provider(#[from] ProviderError),
}

fn refused(s: impl Into<String>) -> ReleaseError {
    ReleaseError::Refused(s.into())
}

/// One serve image of `deploy/deploy.toml` `[[images]]` (`variants.sh`
/// `FV_IMAGES`): its id and whether it is the CPU image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServeImage {
    pub id: String,
    pub cpu: bool,
}

impl ServeImage {
    /// Its templates (`fv_image_flavours`).
    pub fn flavours(&self) -> &'static [Flavour] {
        Flavour::of(self.cpu)
    }
}

/// The serve images, in `deploy/deploy.toml` order.
pub fn serve_images(repo: &Repo) -> Vec<ServeImage> {
    repo.deploy
        .images
        .iter()
        .map(|i| ServeImage {
            id: i.id.clone(),
            cpu: i.compute == Compute::Cpu,
        })
        .collect()
}

/// The release keys (`FV_RELEASE_KEYS`): `debug` and every serve image.
pub fn release_keys(images: &[ServeImage]) -> Vec<String> {
    std::iter::once(ghcr::DEBUG_KEY.to_owned())
        .chain(images.iter().map(|i| i.id.clone()))
        .collect()
}

/// What every command runs on.
pub struct Ctx<'a> {
    pub http: &'a dyn Http,
    pub spawner: &'a dyn Spawner,
    pub clock: &'a dyn Clock,
    pub env: &'a BootEnv,
    /// The serve images (`deploy/deploy.toml`).
    pub images: Vec<ServeImage>,
    /// `deploy/d1/registry.sql` (`FV_REGISTRY_SQL`), applied before the
    /// first D1 write.
    pub registry_sql: Option<PathBuf>,
    /// Progress and warnings (the scripts' `log`, stderr).
    pub log: &'a dyn LogSink,
    /// The result lines (the scripts' stdout).
    pub out: &'a dyn LogSink,
}

impl<'a> Ctx<'a> {
    /// From the repository: its images and `deploy/d1/registry.sql`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        repo: &Repo,
        http: &'a dyn Http,
        spawner: &'a dyn Spawner,
        clock: &'a dyn Clock,
        env: &'a BootEnv,
        log: &'a dyn LogSink,
        out: &'a dyn LogSink,
    ) -> Self {
        let registry_sql = env
            .get("FV_REGISTRY_SQL")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo.root.join("deploy/d1/registry.sql"));
        Self {
            http,
            spawner,
            clock,
            env,
            images: serve_images(repo),
            registry_sql: Some(registry_sql),
            log,
            out,
        }
    }

    pub fn say(&self, s: &str) {
        self.log.line(&format!("[fv-deploy] {s}"));
    }

    pub fn print(&self, s: &str) {
        self.out.line(s);
    }

    pub fn image(&self, id: &str) -> Option<&ServeImage> {
        self.images.iter().find(|i| i.id == id)
    }

    /// The D1 client when a token is set (`fv_d1_available`).
    pub fn d1(&self) -> Option<d1::D1<'a>> {
        let sql = self
            .registry_sql
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok());
        d1::D1::from_env(self.http, self.env, sql)
    }

    /// The Runpod client (`RUNPOD_API_KEY` required).
    pub fn runpod(&self) -> Result<RunpodApi<'a>, ReleaseError> {
        RunpodApi::from_env(self.http, self.clock, self.env).map_err(ReleaseError::from)
    }

    /// The channel the Runpod templates follow (`FV_TEMPLATE_CHANNEL`, default `stable`).
    pub fn template_channel(&self) -> String {
        self.env
            .get("FV_TEMPLATE_CHANNEL")
            .unwrap_or("stable")
            .to_owned()
    }
}

/// The first 7 characters of a sha (`short`).
pub fn short(sha: &str) -> &str {
    sha.get(..7).unwrap_or(sha)
}

/// The 12 hex digits after `sha256:` of a reference (`dshort`).
pub fn dshort(reference: &str) -> String {
    let d = reference.rsplit_once('@').map_or(reference, |(_, d)| d);
    d.chars().skip(7).take(12).collect()
}

/// A channel name (`check_channel`): lower-case, not a build tag, not an image.
pub fn check_channel(c: &str, images: &[ServeImage]) -> Result<(), ReleaseError> {
    let mut chars = c.chars();
    let ok = c.len() >= 2
        && c.len() <= 31
        && chars.next().is_some_and(|x| x.is_ascii_lowercase())
        && chars.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '-');
    if !ok {
        return Err(refused(format!(
            "channel '{c}': lower-case letters, digits and '-'"
        )));
    }
    if c.starts_with("sha-") || c.starts_with("buildcache") || c.contains("-sha-") {
        return Err(refused(format!(
            "channel '{c}' collides with the build tags"
        )));
    }
    if images.iter().any(|i| i.id == c) {
        return Err(refused(format!("channel '{c}' is a serve image name")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imgs() -> Vec<ServeImage> {
        vec![
            ServeImage {
                id: "cuda".into(),
                cpu: false,
            },
            ServeImage {
                id: "cpu".into(),
                cpu: true,
            },
        ]
    }

    #[test]
    fn channels_follow_check_channel() {
        for ok in ["stable", "latest", "canary-2"] {
            assert!(check_channel(ok, &imgs()).is_ok(), "{ok}");
        }
        for bad in [
            "Stable",
            "s",
            "sha-abc",
            "buildcache",
            "x-sha-1",
            "cuda",
            "1abc",
            "a_b",
        ] {
            assert!(check_channel(bad, &imgs()).is_err(), "{bad}");
        }
        assert_eq!(release_keys(&imgs()), ["debug", "cuda", "cpu"]);
    }

    #[test]
    fn short_forms() {
        assert_eq!(short("abcdef0123"), "abcdef0");
        assert_eq!(short("ab"), "ab");
        assert_eq!(dshort("r@sha256:0123456789abcdef"), "0123456789ab");
        assert_eq!(dshort("-"), "");
    }
}
