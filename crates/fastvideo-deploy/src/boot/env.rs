//! The boot's environment contract (docs/adr/0014-fv-boot.md §3): a snapshot
//! of the process environment, the kind of host it describes, and the fixed
//! paths of the image (overridable in tests).

use std::collections::BTreeMap;
use std::path::PathBuf;

/// The process environment, as the boot read it at start. Every value the
/// boot reads goes through [`BootEnv::get`] (an empty value counts as unset,
/// as `${VAR:-}` does in the shell boots it replaces).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BootEnv {
    vars: BTreeMap<String, String>,
}

impl BootEnv {
    pub fn from_process() -> Self {
        Self {
            vars: std::env::vars_os()
                .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
                .collect(),
        }
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            vars: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    /// The value, `None` when unset or empty.
    pub fn get(&self, k: &str) -> Option<&str> {
        self.vars
            .get(k)
            .map(String::as_str)
            .filter(|v| !v.is_empty())
    }

    /// Set (even to the empty string): `${VAR+x}`.
    pub fn is_set(&self, k: &str) -> bool {
        self.vars.contains_key(k)
    }

    /// The raw value, empty included.
    pub fn raw(&self, k: &str) -> Option<&str> {
        self.vars.get(k).map(String::as_str)
    }

    pub fn get_or<'a>(&'a self, k: &str, default: &'a str) -> &'a str {
        self.get(k).unwrap_or(default)
    }

    /// `VAR=1`.
    pub fn flag(&self, k: &str) -> bool {
        self.get(k) == Some("1")
    }

    /// A number of seconds, else `default`.
    pub fn secs(&self, k: &str, default: f64) -> f64 {
        self.get(k)
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(default)
    }

    pub fn set(&mut self, k: impl Into<String>, v: impl Into<String>) {
        self.vars.insert(k.into(), v.into());
    }

    pub fn remove(&mut self, k: &str) {
        self.vars.remove(k);
    }

    pub fn vars(&self) -> &BTreeMap<String, String> {
        &self.vars
    }

    /// The kind of host this environment describes.
    pub fn kind(&self) -> Kind {
        if let Some(p) = self.get("FV_PROVIDER").filter(|p| *p != "runpod") {
            return Kind::Provider(Provider::parse(p));
        }
        // Runpod's own serverless marker (or fv-control's endpoint name) first: a
        // serverless worker given the worker role (scripts/serve/runpod-endpoint.sh's
        // role env) is still a serverless worker.
        if self.get("RUNPOD_ENDPOINT_ID").is_some() || self.get("FV_CONTROL_ENDPOINT").is_some() {
            return Kind::Serverless;
        }
        // A cluster worker is a Runpod *pod* with the worker role; GCP and CloudRift
        // workers (also FV_SERVE_ROLE=worker, no RUNPOD_POD_ID) are the image on its own.
        if self.get("FV_SERVE_ROLE") == Some("worker") && self.get("RUNPOD_POD_ID").is_some() {
            return Kind::ClusterWorker;
        }
        Kind::Image
    }
}

/// A provider fv-control launches pods on outside Runpod (`FV_PROVIDER`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provider {
    Gmi,
    Brev,
    Other(String),
}

impl Provider {
    pub fn parse(s: &str) -> Self {
        match s {
            "gmi" => Self::Gmi,
            "brev" => Self::Brev,
            o => Self::Other(o.to_owned()),
        }
    }
    pub fn name(&self) -> &str {
        match self {
            Self::Gmi => "gmi",
            Self::Brev => "brev",
            Self::Other(s) => s,
        }
    }
}

/// What started the container, as the environment tells it. It picks the
/// steps the legacy boots differed in (ADR-0014 §2): the config handling of a
/// cluster worker, the tunnel and fail sleep of a provider pod.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A Runpod pod fv-control placed in a cluster or as a standalone pod
    /// (`FV_SERVE_ROLE=worker` and `RUNPOD_POD_ID`, not serverless; the legacy
    /// `WORKER_BOOT`, `EDGE_WORKER_BOOT`, `WATCHDOG_WORKER_BOOT`).
    ClusterWorker,
    /// A GMI / Brev pod (`FV_PROVIDER`; the legacy `PROVIDER_BOOT`).
    Provider(Provider),
    /// A Runpod serverless worker (`RUNPOD_ENDPOINT_ID` or fv-control's
    /// `FV_CONTROL_ENDPOINT`; the legacy `SLS_BOOT`, `SLS_R2_BOOT`,
    /// `SLS_HUB_BOOT`).
    Serverless,
    /// The image started on its own (the legacy `fv-entry.sh`: GCP, CloudRift,
    /// Vast, the scripts, a serverless template without a boot override).
    Image,
}

impl Kind {
    pub fn label(&self) -> String {
        match self {
            Self::ClusterWorker => "cluster worker".into(),
            Self::Provider(p) => format!("{} pod", p.name()),
            Self::Serverless => "serverless worker".into(),
            Self::Image => "image".into(),
        }
    }
    /// fv-control's own config handling (`/fv-worker.toml`, the gateway off).
    pub fn controller_worker(&self) -> bool {
        matches!(self, Self::ClusterWorker | Self::Provider(_))
    }
}

/// The image's fixed paths. [`Paths::image`] is what the serve images use;
/// tests point every one into a temporary directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paths {
    /// Where an inline or copied worker config is written (`/fv-worker.toml`).
    pub worker_toml: PathBuf,
    /// A serverless worker's volume mount (`/runpod-volume/weights`).
    pub runpod_volume_weights: PathBuf,
    /// Where pods mount the volume (`/workspace/weights`).
    pub workspace_weights: PathBuf,
    /// The serve binary unless `FV_SERVE_BIN` names another.
    pub serve_bin: PathBuf,
    /// The prompt upsampler sidecar's launcher.
    pub upsampler: PathBuf,
    /// The image's copy of the R2 fetcher unless `FV_R2_FETCH_IMAGE` names another.
    pub r2_fetch_image: PathBuf,
    /// The tools directory unless `FV_R2_TOOLS` names another (`/opt/fv`).
    pub tools: PathBuf,
    /// The default state directory unless `FV_STATE_DIR` names another.
    pub state: PathBuf,
    /// Where cloudflared is installed.
    pub cloudflared: PathBuf,
    /// The pinned cloudflared release and its SHA-256.
    pub cloudflared_url: String,
    pub cloudflared_sha256: String,
    /// This binary (re-run for the watchdog and the fetch guard).
    pub self_exe: PathBuf,
}

impl Paths {
    pub fn image() -> Self {
        Self {
            worker_toml: "/fv-worker.toml".into(),
            runpod_volume_weights: "/runpod-volume/weights".into(),
            workspace_weights: "/workspace/weights".into(),
            serve_bin: "/opt/fastvideo-rs/bin/fv-serve".into(),
            upsampler: "/opt/fastvideo-rs/bin/fv-upsampler".into(),
            r2_fetch_image: "/opt/fastvideo-rs/bin/r2-fetch.py".into(),
            tools: "/opt/fv".into(),
            state: "/fvstate".into(),
            cloudflared: "/usr/local/bin/cloudflared".into(),
            cloudflared_url: super::tunnel::CLOUDFLARED_URL.into(),
            cloudflared_sha256: super::tunnel::CLOUDFLARED_SHA256.into(),
            self_exe: std::env::current_exe()
                .unwrap_or_else(|_| PathBuf::from("/opt/fastvideo-rs/bin/fv-boot")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_values_are_unset() {
        let e = BootEnv::from_pairs([("A", ""), ("B", "x")]);
        assert_eq!(e.get("A"), None);
        assert!(e.is_set("A"));
        assert_eq!(e.get("B"), Some("x"));
        assert!(!e.is_set("C"));
    }

    #[test]
    fn kind_from_the_env() {
        let k = |p: &[(&str, &str)]| BootEnv::from_pairs(p.iter().copied()).kind();
        assert_eq!(
            k(&[("FV_PROVIDER", "gmi"), ("FV_SERVE_ROLE", "worker")]),
            Kind::Provider(Provider::Gmi)
        );
        assert_eq!(
            k(&[
                ("FV_PROVIDER", "runpod"),
                ("FV_SERVE_ROLE", "worker"),
                ("RUNPOD_POD_ID", "p")
            ]),
            Kind::ClusterWorker
        );
        assert_eq!(
            k(&[("FV_SERVE_ROLE", "worker"), ("RUNPOD_POD_ID", "p")]),
            Kind::ClusterWorker
        );
        // GCP / CloudRift workers: the role without a Runpod pod.
        assert_eq!(k(&[("FV_SERVE_ROLE", "worker")]), Kind::Image);
        // runpod-endpoint.sh's role endpoints: serverless first.
        assert_eq!(
            k(&[
                ("FV_SERVE_ROLE", "worker"),
                ("RUNPOD_POD_ID", "w"),
                ("RUNPOD_ENDPOINT_ID", "ep")
            ]),
            Kind::Serverless
        );
        assert_eq!(k(&[("RUNPOD_ENDPOINT_ID", "ep")]), Kind::Serverless);
        assert_eq!(k(&[("FV_CONTROL_ENDPOINT", "x")]), Kind::Serverless);
        assert_eq!(k(&[("RUNPOD_POD_ID", "p")]), Kind::Image);
        assert!(Kind::Provider(Provider::Brev).controller_worker());
        assert!(!Kind::Serverless.controller_worker());
    }

    #[test]
    fn secs_parse_or_default() {
        let e = BootEnv::from_pairs([("S", "12.5"), ("B", "x"), ("N", "-1")]);
        assert_eq!(e.secs("S", 1.0), 12.5);
        assert_eq!(e.secs("B", 1.0), 1.0);
        assert_eq!(e.secs("N", 1.0), 1.0);
        assert_eq!(e.secs("U", 3.0), 3.0);
    }
}
