//! The fv-serve images in GHCR (`scripts/serve/lib/ghcr.sh`): tags, digests,
//! labels and the retag a promotion is (`docker buildx imagetools create`:
//! no rebuild, the same digest under another tag).
//!
//! Release keys: `debug` (the all-in-one image: `:sha-<s>`, `:latest`,
//! `:<channel>`) and every serve image of `deploy/deploy.toml` (`cuda`,
//! `cpu`: `:<i>-sha-<s>`, `:<i>`, `:<i>-<channel>`).

use std::cell::RefCell;
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::boot::env::BootEnv;
use crate::boot::sys::{Cmd, Http, Io, Method, Request, Spawner};
use crate::launch::{MANIFEST_ACCEPT, SERVE_REPO};

use super::registry::Digests;

/// The registry API (`FV_REGISTRY_API`; the tests use a mock).
pub const REGISTRY_API: &str = "https://ghcr.io";
/// The debug image's release key.
pub const DEBUG_KEY: &str = "debug";

/// A GHCR call that failed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("GHCR: {0}")]
pub struct GhcrError(pub String);

/// The serve repository in a registry, read anonymously (a public package).
pub struct Ghcr<'a> {
    http: &'a dyn Http,
    /// `https://ghcr.io`.
    pub api: String,
    /// `ghcr.io/zaitrarrio/fastvideo-rs-serve` (`FV_SERVE_REPO`).
    pub repo: String,
    token: RefCell<Option<String>>,
}

impl std::fmt::Debug for Ghcr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ghcr")
            .field("api", &self.api)
            .field("repo", &self.repo)
            .finish_non_exhaustive()
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// `sha-<s>` for the debug image, `<i>-sha-<s>` for a serve image
/// (`fv_key_sha_tag`).
pub fn key_sha_tag(key: &str, short: &str) -> String {
    if key == DEBUG_KEY {
        format!("sha-{short}")
    } else {
        format!("{key}-sha-{short}")
    }
}

/// The tags a promotion to `channel` points at the image
/// (`fv_key_channel_tags`): debug `:<channel>`; a serve image
/// `:<i>-<channel>`, and for `latest` also the plain `:<i>`.
pub fn key_channel_tags(key: &str, channel: &str) -> Vec<String> {
    if key == DEBUG_KEY {
        vec![channel.to_owned()]
    } else if channel == "latest" {
        vec![format!("{key}-latest"), key.to_owned()]
    } else {
        vec![format!("{key}-{channel}")]
    }
}

impl<'a> Ghcr<'a> {
    pub fn from_env(http: &'a dyn Http, env: &BootEnv) -> Self {
        Self {
            http,
            api: env
                .get("FV_REGISTRY_API")
                .unwrap_or(REGISTRY_API)
                .trim_end_matches('/')
                .to_owned(),
            repo: env.get("FV_SERVE_REPO").unwrap_or(SERVE_REPO).to_owned(),
            token: RefCell::new(None),
        }
    }

    /// The repository's path in the registry (`zaitrarrio/fastvideo-rs-serve`).
    pub fn path(&self) -> &str {
        self.repo
            .split_once('/')
            .map_or(self.repo.as_str(), |(_, p)| p)
    }

    fn get(
        &self,
        url: &str,
        token: Option<&str>,
        accept: Option<&str>,
    ) -> Result<(u16, Vec<u8>), GhcrError> {
        let mut req = Request::new(Method::Get, url, Duration::from_secs(60));
        if let Some(t) = token {
            req = req.bearer(t);
        }
        if let Some(a) = accept {
            req = req.header("accept", a);
        }
        let r = self.http.send(&req).map_err(|e| GhcrError(e.to_string()))?;
        Ok((r.status, r.body))
    }

    /// The anonymous pull token (`fv_ghcr_init`), once.
    pub fn token(&self) -> Result<String, GhcrError> {
        if let Some(t) = self.token.borrow().as_ref() {
            return Ok(t.clone());
        }
        let (_, body) = self.get(
            &format!("{}/token?scope=repository:{}:pull", self.api, self.path()),
            None,
            None,
        )?;
        let t = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| v.get("token").and_then(Value::as_str).map(str::to_owned))
            .filter(|t| !t.is_empty())
            .ok_or_else(|| GhcrError(format!("no pull token for {}", self.repo)))?;
        *self.token.borrow_mut() = Some(t.clone());
        Ok(t)
    }

    fn manifest(&self, reference: &str) -> Result<Option<Vec<u8>>, GhcrError> {
        let tok = self.token()?;
        let (status, body) = self.get(
            &format!("{}/v2/{}/manifests/{reference}", self.api, self.path()),
            Some(&tok),
            Some(MANIFEST_ACCEPT),
        )?;
        Ok(((200..300).contains(&status) && !body.is_empty()).then_some(body))
    }

    /// A tag's (or a digest's) digest, `sha256:…`: the SHA-256 of the
    /// manifest bytes the registry serves (its `docker-content-digest`);
    /// `None` when there is no such tag (`fv_ghcr_digest`).
    pub fn digest(&self, reference: &str) -> Result<Option<String>, GhcrError> {
        Ok(self
            .manifest(reference)?
            .map(|b| format!("sha256:{}", hex(&Sha256::digest(&b)))))
    }

    /// The image config's labels (`fv_ghcr_labels`; an index resolves to its
    /// amd64 manifest).
    pub fn labels(&self, digest: &str) -> Result<Value, GhcrError> {
        let parse = |b: Vec<u8>| -> Result<Value, GhcrError> {
            serde_json::from_slice(&b).map_err(|e| GhcrError(format!("manifest {digest}: {e}")))
        };
        let missing = || GhcrError(format!("no manifest {digest}"));
        let mut m = parse(self.manifest(digest)?.ok_or_else(missing)?)?;
        if let Some(list) = m.get("manifests").and_then(Value::as_array) {
            let d = list
                .iter()
                .find(|x| {
                    x.pointer("/platform/architecture").and_then(Value::as_str) == Some("amd64")
                })
                .and_then(|x| x.get("digest").and_then(Value::as_str))
                .ok_or_else(|| GhcrError(format!("{digest}: no amd64 manifest")))?
                .to_owned();
            m = parse(self.manifest(&d)?.ok_or_else(missing)?)?;
        }
        let cfg = m
            .pointer("/config/digest")
            .and_then(Value::as_str)
            .ok_or_else(|| GhcrError(format!("{digest}: no config")))?;
        let tok = self.token()?;
        let (status, body) = self.get(
            &format!("{}/v2/{}/blobs/{cfg}", self.api, self.path()),
            Some(&tok),
            None,
        )?;
        if !(200..300).contains(&status) {
            return Err(GhcrError(format!("{digest}: config blob HTTP {status}")));
        }
        let c: Value =
            serde_json::from_slice(&body).map_err(|e| GhcrError(format!("config {cfg}: {e}")))?;
        Ok(c.pointer("/config/Labels")
            .cloned()
            .unwrap_or(Value::Object(Default::default())))
    }

    /// The git revision label of a digest (`org.opencontainers.image.revision`).
    pub fn revision(&self, digest: &str) -> Result<Option<String>, GhcrError> {
        Ok(self
            .labels(digest)?
            .get("org.opencontainers.image.revision")
            .and_then(Value::as_str)
            .map(str::to_owned))
    }

    /// `key -> repo@digest` of the build's tags that exist, in key order
    /// (`fv_ghcr_set_for_sha`).
    pub fn set_for_sha(&self, short: &str, keys: &[String]) -> Result<Digests, GhcrError> {
        let mut out = Vec::new();
        for k in keys {
            if let Some(d) = self.digest(&key_sha_tag(k, short))? {
                out.push((k.clone(), format!("{}@{d}", self.repo)));
            }
        }
        Ok(Digests(out))
    }

    /// The retag command (`docker buildx imagetools create --tag <repo>:<tag> <src>`).
    pub fn retag_cmd(&self, docker: &str, src: &str, tag: &str) -> Cmd {
        Cmd::new(docker)
            .args(["buildx", "imagetools", "create", "--tag"])
            .arg(format!("{}:{tag}", self.repo))
            .arg(src)
            .stdout(Io::ToStderr)
            .stderr(Io::Inherit)
    }

    /// Points `:<tag>` at `src` (`repo@sha256:…`) and checks the tag then
    /// resolves to the same digest (`fv_ghcr_retag`).
    pub fn retag(
        &self,
        spawner: &dyn Spawner,
        docker: &str,
        src: &str,
        tag: &str,
    ) -> Result<(), GhcrError> {
        let want = src.rsplit_once('@').map_or(src, |(_, d)| d);
        let code = spawner
            .status(&self.retag_cmd(docker, src, tag))
            .map_err(|e| GhcrError(format!("retag {tag}: {e}")))?;
        if code != Some(0) {
            return Err(GhcrError(format!("retag {tag}: imagetools create failed")));
        }
        match self.digest(tag)? {
            Some(got) if got == want => Ok(()),
            Some(got) => Err(GhcrError(format!(
                "retag {tag}: resolves to {got}, not {want}"
            ))),
            None => Err(GhcrError(format!("retag {tag}: the tag does not resolve"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_follow_ghcr_sh() {
        assert_eq!(key_sha_tag("debug", "abc1234"), "sha-abc1234");
        assert_eq!(key_sha_tag("cuda", "abc1234"), "cuda-sha-abc1234");
        assert_eq!(key_channel_tags("debug", "stable"), ["stable"]);
        assert_eq!(key_channel_tags("cuda", "stable"), ["cuda-stable"]);
        assert_eq!(key_channel_tags("cpu", "latest"), ["cpu-latest", "cpu"]);
    }
}
