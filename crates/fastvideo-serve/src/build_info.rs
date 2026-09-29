//! Build and deployment identity (docs/serve/releases.md): what `fv-serve
//! --version` prints and what `/health`, `/healthz` and `/` report under
//! `build`.
//!
//! Two halves:
//!
//! - compiled in by `build.rs`: the package version, the git sha, the build
//!   time (the commit time), the CI build id, the enabled features and the
//!   cargo profile;
//! - read from the environment at start, because one binary serves several
//!   images (the CUDA variants share it) and an image cannot know its own
//!   digest: `FV_VARIANT` (baked per variant image), `FV_IMAGE_REF`,
//!   `FV_IMAGE_TAG`, `FV_IMAGE_DIGEST` and `FV_RELEASE_CHANNEL` (set by the
//!   Runpod templates and the deploy scripts, scripts/serve/runpod-templates.sh).
//!
//! Nothing here is secret.

use std::sync::OnceLock;

use serde::Serialize;
use serde_json::Value;

use crate::config::{Env, ProcessEnv};

/// The package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Full git sha of the build, or `unknown`.
pub const GIT_SHA: &str = env!("FV_BUILD_GIT_SHA");
/// Commit time of the build (`YYYY-MM-DDTHH:MM:SSZ`), or empty.
pub const BUILD_TIME: &str = env!("FV_BUILD_TIME");
/// CI build id (scripts/gpu/docker.sh build-id), or empty.
pub const BUILD_ID: &str = env!("FV_BUILD_ID");
/// Variant fixed at compile time (`FV_BUILD_VARIANT`), or empty.
pub const BUILD_VARIANT: &str = env!("FV_BUILD_VARIANT");
/// Enabled crate features, comma separated.
pub const FEATURES: &str = env!("FV_BUILD_FEATURES");
/// Cargo profile (`release`, `debug`).
pub const PROFILE: &str = env!("FV_BUILD_PROFILE");

/// The image this process runs from, as the deployment tells it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ImageInfo {
    /// `FV_IMAGE_REF`: the reference the pod or template was created with.
    #[serde(rename = "ref")]
    pub reference: Option<String>,
    /// `FV_IMAGE_TAG`, or the tag of `FV_IMAGE_REF`.
    pub tag: Option<String>,
    /// `FV_IMAGE_DIGEST`, or the digest of `FV_IMAGE_REF` (`sha256:…`).
    pub digest: Option<String>,
}

/// Everything `--version` and the health routes report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BuildInfo {
    pub version: String,
    pub git_sha: String,
    pub git_sha_short: String,
    /// Commit time of the build, or `None` when unknown.
    pub build_time: Option<String>,
    pub build_id: Option<String>,
    pub features: Vec<String>,
    pub profile: String,
    /// The image variant (`h3-turbo`, `gateway`, …; `FV_VARIANT`).
    pub variant: Option<String>,
    pub image: ImageInfo,
    /// The release channel the deployment followed (`stable`, `latest`, …).
    pub channel: Option<String>,
}

fn non_empty(s: &str) -> Option<String> {
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_owned())
}

/// `sha256:<64 hex>` or nothing.
fn valid_digest(d: &str) -> Option<String> {
    let d = d.trim();
    let hex = d.strip_prefix("sha256:")?;
    (hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit())).then(|| d.to_ascii_lowercase())
}

/// Splits `repo[:tag][@digest]` into (tag, digest).
fn split_ref(r: &str) -> (Option<String>, Option<String>) {
    let (name, digest) = match r.split_once('@') {
        Some((n, d)) => (n, valid_digest(d)),
        None => (r, None),
    };
    // The tag is after the last ':' that follows the last '/' (a registry
    // host may carry a port).
    let last = name.rsplit('/').next().unwrap_or(name);
    let tag = last.split_once(':').and_then(|(_, t)| non_empty(t));
    (tag, digest)
}

impl BuildInfo {
    /// The compiled-in half plus the deployment's environment.
    pub fn from_env(env: &dyn Env) -> Self {
        let reference = env.var("FV_IMAGE_REF").and_then(|r| non_empty(&r));
        let (ref_tag, ref_digest) = reference.as_deref().map(split_ref).unwrap_or_default();
        let git_sha = non_empty(GIT_SHA).unwrap_or_else(|| "unknown".into());
        let git_sha_short = if git_sha == "unknown" { git_sha.clone() } else { git_sha.chars().take(7).collect() };
        BuildInfo {
            version: VERSION.to_owned(),
            git_sha,
            git_sha_short,
            build_time: non_empty(BUILD_TIME),
            build_id: non_empty(BUILD_ID),
            features: FEATURES.split(',').filter_map(non_empty).collect(),
            profile: PROFILE.to_owned(),
            variant: env.var("FV_VARIANT").and_then(|v| non_empty(&v)).or_else(|| non_empty(BUILD_VARIANT)),
            image: ImageInfo {
                tag: env.var("FV_IMAGE_TAG").and_then(|t| non_empty(&t)).or(ref_tag),
                digest: env.var("FV_IMAGE_DIGEST").and_then(|d| valid_digest(&d)).or(ref_digest),
                reference,
            },
            channel: env.var("FV_RELEASE_CHANNEL").and_then(|c| non_empty(&c)),
        }
    }

    /// The process-wide value (read once from the process environment).
    pub fn current() -> &'static BuildInfo {
        static CURRENT: OnceLock<BuildInfo> = OnceLock::new();
        CURRENT.get_or_init(|| BuildInfo::from_env(&ProcessEnv))
    }

    /// [`long_version`](Self::long_version) of [`current`](Self::current),
    /// for clap (which takes a `&'static str`).
    pub fn current_long_version() -> &'static str {
        static TEXT: OnceLock<String> = OnceLock::new();
        TEXT.get_or_init(|| Self::current().long_version())
    }

    /// The JSON object the health routes embed under `build`.
    pub fn json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// One line: `0.1.0 (abc1234 2026-09-29T10:00:00Z) h3-turbo stable sha256:0123456789ab`.
    pub fn summary(&self) -> String {
        let mut s = format!("{} ({}", self.version, self.git_sha_short);
        if let Some(t) = &self.build_time {
            s.push(' ');
            s.push_str(t);
        }
        s.push(')');
        for part in [&self.variant, &self.channel].into_iter().flatten() {
            s.push(' ');
            s.push_str(part);
        }
        if let Some(d) = &self.image.digest {
            s.push(' ');
            s.push_str(&d[..d.len().min("sha256:".len() + 12)]);
        }
        s
    }

    /// `fv-serve --version`: the version line, then one `key: value` line
    /// per known field.
    pub fn long_version(&self) -> String {
        let mut lines = vec![self.version.clone()];
        let mut kv = |k: &str, v: Option<&str>| {
            if let Some(v) = v.filter(|v| !v.is_empty()) {
                lines.push(format!("{k:<9} {v}"));
            }
        };
        kv("git:", Some(&self.git_sha));
        kv("built:", self.build_time.as_deref());
        kv("build-id:", self.build_id.as_deref());
        kv("profile:", Some(&self.profile));
        kv("features:", Some(&self.features.join(",")));
        kv("variant:", self.variant.as_deref());
        kv("channel:", self.channel.as_deref());
        kv("image:", self.image.reference.as_deref());
        kv("tag:", self.image.tag.as_deref());
        kv("digest:", self.image.digest.as_deref());
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    const D: &str = "sha256:c782eb378f3f5e41139096010c4942244793cde70b5e5bba97877a11ecfd6045";

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn compiled_half_is_present() {
        let b = BuildInfo::from_env(&env(&[]));
        assert_eq!(b.version, env!("CARGO_PKG_VERSION"));
        assert!(!b.git_sha.is_empty());
        assert!(b.git_sha == "unknown" || b.git_sha.chars().all(|c| c.is_ascii_hexdigit()), "{}", b.git_sha);
        assert!(b.git_sha_short.len() <= 7 || b.git_sha_short == "unknown");
        assert!(b.git_sha.starts_with(&b.git_sha_short));
        assert!(!b.profile.is_empty());
        assert!(!b.features.iter().any(|f| f == "default" || f.is_empty()));
        assert_eq!(b.image, ImageInfo::default());
        assert_eq!(b.channel, None);
    }

    #[test]
    fn deployment_env_fills_variant_image_and_channel() {
        let b = BuildInfo::from_env(&env(&[
            ("FV_VARIANT", "h3-turbo"),
            ("FV_IMAGE_REF", &format!("ghcr.io/zaitrarrio/fastvideo-rs-serve@{D}")),
            ("FV_IMAGE_TAG", "h3-turbo-sha-2cd1ba0"),
            ("FV_RELEASE_CHANNEL", "stable"),
        ]));
        assert_eq!(b.variant.as_deref(), Some("h3-turbo"));
        assert_eq!(b.image.digest.as_deref(), Some(D));
        assert_eq!(b.image.tag.as_deref(), Some("h3-turbo-sha-2cd1ba0"));
        assert_eq!(b.channel.as_deref(), Some("stable"));
        let j = b.json();
        assert_eq!(j["image"]["ref"], format!("ghcr.io/zaitrarrio/fastvideo-rs-serve@{D}"));
        assert_eq!(j["image"]["digest"], D);
        assert_eq!(j["variant"], "h3-turbo");
        assert_eq!(j["channel"], "stable");
        assert!(j["git_sha"].is_string() && j["features"].is_array());
        let s = b.summary();
        assert!(s.contains("h3-turbo stable sha256:c782eb378f3f"), "{s}");
        let v = b.long_version();
        assert!(v.starts_with(env!("CARGO_PKG_VERSION")), "{v}");
        assert!(v.contains(&format!("digest:   {D}")), "{v}");
        assert!(v.contains("variant:  h3-turbo"), "{v}");
    }

    #[test]
    fn tag_and_digest_come_from_the_ref() {
        let b = BuildInfo::from_env(&env(&[("FV_IMAGE_REF", "localhost:5000/fv/serve:stable")]));
        assert_eq!(b.image.tag.as_deref(), Some("stable"));
        assert_eq!(b.image.digest, None);
        let b = BuildInfo::from_env(&env(&[("FV_IMAGE_REF", &format!("ghcr.io/o/r:h3-max@{D}"))]));
        assert_eq!(b.image.tag.as_deref(), Some("h3-max"));
        assert_eq!(b.image.digest.as_deref(), Some(D));
        // An explicit digest wins; a malformed one is dropped.
        let b = BuildInfo::from_env(&env(&[("FV_IMAGE_REF", "ghcr.io/o/r@sha256:abc"), ("FV_IMAGE_DIGEST", "latest")]));
        assert_eq!(b.image.digest, None);
        let b = BuildInfo::from_env(&env(&[("FV_IMAGE_DIGEST", &D.to_ascii_uppercase().replace("SHA256", "sha256"))]));
        assert_eq!(b.image.digest.as_deref(), Some(D));
    }

    #[test]
    fn current_is_stable() {
        assert_eq!(BuildInfo::current() as *const _, BuildInfo::current() as *const _);
    }
}
