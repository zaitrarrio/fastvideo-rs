//! Config: TOML file plus environment, with secret redaction (design §6.1,
//! §6.3, §0 decision 7).
//!
//! Precedence: defaults < `--config` TOML < environment. Secrets come only
//! from the environment (or the TOML for local development) and are held in
//! [`Secret`], whose `Debug`/`Serialize` never print the value.
//!
//! Environment (upper-case `FV_*`; the Cloudflare/R2 names are also accepted
//! in the lower-case Runpod-secret spelling, e.g. `fv_cf_account_id`):
//!
//! | Variable | Sets |
//! |---|---|
//! | `FV_SERVE_MODE` | `server.mode` (`http` \| `runpod-queue`) |
//! | `PORT` | port of `server.bind` (Runpod load balancer) |
//! | `FV_BIND`, `FV_PUBLIC_BASE_URL`, `FV_STATE_DIR`, `FV_WORKER_ID` | `server.*` |
//! | `FV_WEIGHTS` | substituted for `${FV_WEIGHTS}` in `models[].weights` |
//! | `FV_AUTH_MODE`, `FV_API_KEYS` (SHA-256 hex list) | `auth.*` |
//! | `FV_URL_SIGNING_KEY`, `FV_WEBHOOK_ED25519_KEY` | signing keys |
//! | `FV_ENGINE` (`fake` \| `cuda`) | `engine.backend` |
//! | `FV_JOB_STORE` (`auto` \| `memory` \| `file` \| `d1`) | `jobs.backend` |
//! | `FV_CF_ACCOUNT_ID`, `FV_CF_API_TOKEN`, `FV_D1_DATABASE_ID` | `jobs.d1.*` |
//! | `FV_ARTIFACTS` (`auto` \| `local` \| `s3`) | `artifacts.backend` |
//! | `FV_R2_BUCKET`, `FV_R2_ENDPOINT`, `FV_R2_ACCESS_KEY_ID`, `FV_R2_SECRET_ACCESS_KEY` | `artifacts.s3.*` (R2: region `auto`) |
//! | `FV_S3_ENDPOINT`, `FV_S3_REGION`, `FV_S3_BUCKET`, `FV_S3_ACCESS_KEY_ID`, `FV_S3_SECRET_ACCESS_KEY` | `artifacts.s3.*` (generic S3) |
//! | `FV_LOG_FORMAT` (`text` \| `json`), `RUST_LOG` | logging |
//!
//! `auto` picks D1 when the three D1 values are set (else the file store)
//! and S3/R2 when bucket, endpoint and both keys are set (else local).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use fastvideo_serve_kit::AuthMode;
use serde::{Deserialize, Serialize, Serializer};

/// A secret string: never printed.
#[derive(Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct Secret(pub String);

impl Secret {
    pub fn expose(&self) -> &str {
        &self.0
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_empty() { "<unset>" } else { "<redacted>" })
    }
}

impl Serialize for Secret {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(if self.0.is_empty() { "<unset>" } else { "<redacted>" })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    #[default]
    Http,
    RunpodQueue,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerCfg {
    pub bind: String,
    pub public_base_url: Option<String>,
    pub state_dir: PathBuf,
    pub mode: Mode,
    /// Worker identity for the D1 `worker` column (default: `RUNPOD_POD_ID`,
    /// `HOSTNAME`, else a random id).
    pub worker_id: Option<String>,
    /// SIGTERM grace for the running generation (design §6.3: 25 s).
    pub shutdown_grace_s: u64,
    /// Sync endpoints' wait (`/v1/videos/sync`, LTX v1, fal `/run`).
    pub sync_timeout_s: u64,
}

impl Default for ServerCfg {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8000".into(),
            public_base_url: None,
            state_dir: PathBuf::from("/workspace/fv-state"),
            mode: Mode::Http,
            worker_id: None,
            shutdown_grace_s: 25,
            sync_timeout_s: 600,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthCfg {
    pub mode: AuthMode,
    /// SHA-256 hex hashes of the accepted keys (normally `FV_API_KEYS`).
    pub keys: Secret,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactBackend {
    #[default]
    Auto,
    Local,
    /// S3-compatible (Cloudflare R2, Runpod S3, AWS).
    #[serde(alias = "r2")]
    S3,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct S3Cfg {
    /// R2: `https://<account>.r2.cloudflarestorage.com`.
    pub endpoint: Option<String>,
    /// R2: `auto`.
    pub region: Option<String>,
    pub bucket: Option<String>,
    pub access_key_id: Secret,
    pub secret_access_key: Secret,
    /// Path-style URLs (default true: works for R2 and Runpod S3).
    pub path_style: Option<bool>,
    /// Key prefix (default `fv-serve/`).
    pub prefix: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ArtifactsCfg {
    pub backend: ArtifactBackend,
    pub url_ttl_s: u64,
    /// HMAC key for `/files` URLs (normally `FV_URL_SIGNING_KEY`).
    pub signing_key: Secret,
    pub s3: S3Cfg,
}

impl Default for ArtifactsCfg {
    fn default() -> Self {
        Self {
            backend: ArtifactBackend::Auto,
            url_ttl_s: 86_400,
            signing_key: Secret::default(),
            s3: S3Cfg::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobBackend {
    #[default]
    Auto,
    /// In memory only (tests; Runpod LB sync-only).
    Memory,
    /// In memory plus JSON manifests under `state_dir/jobs` (MemJobStore).
    File,
    /// Cloudflare D1 (design §0 decision 7).
    D1,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct D1Cfg {
    pub account_id: Option<String>,
    pub api_token: Secret,
    pub database_id: Option<String>,
    /// Override the API base (tests: the local mock).
    pub api_base: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JobsCfg {
    pub backend: JobBackend,
    pub d1: D1Cfg,
    /// ≤ 1 progress write per job per this interval (D1).
    pub progress_interval_ms: u64,
    /// Other workers' unfinished jobs without a heartbeat for this long are
    /// failed (D1); 0 disables.
    pub stale_after_s: u64,
    /// Expiry sweep period.
    pub sweep_interval_s: u64,
    /// Per-API retention overrides in seconds, keyed by `ProtocolId` name.
    pub retention_s: BTreeMap<String, u64>,
}

impl Default for JobsCfg {
    fn default() -> Self {
        Self {
            backend: JobBackend::Auto,
            d1: D1Cfg::default(),
            progress_interval_ms: 1000,
            stale_after_s: 900,
            sweep_interval_s: 300,
            retention_s: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineBackendKind {
    /// FakeBackend (CPU CI, `--features fake` smoke runs).
    #[default]
    Fake,
    /// CudaBackend (WP-11; `--features cuda`).
    Cuda,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FakeCfg {
    /// Model ids from the fake default set (empty: all).
    pub models: Vec<String>,
    pub step_ms: u64,
    pub load_ms: u64,
    /// Build time / clip time (overrides `step_ms`).
    pub rtf: Option<f64>,
    /// Mark every fake model resident (no swapping needed).
    pub all_resident: bool,
    /// Without ffmpeg the fake writes no MP4; store a small placeholder
    /// file instead so the job still succeeds (CI).
    pub placeholder_output: bool,
}

impl Default for FakeCfg {
    fn default() -> Self {
        Self { models: Vec::new(), step_ms: 2, load_ms: 0, rtf: None, all_resident: true, placeholder_output: true }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineCfg {
    pub backend: EngineBackendKind,
    /// Load non-resident models on demand (risk R18).
    pub swap: bool,
    /// Tier alias (`h3-max`, `ltx-turbo`, ...) -> model id.
    pub tier_overrides: BTreeMap<String, String>,
    pub fake: FakeCfg,
    /// Post-processing H.264 encoder for crops: `nvenc` (deployed) or
    /// `cpu-test-x264` (CPU CI only).
    pub post_encoder: String,
}

impl Default for EngineCfg {
    fn default() -> Self {
        Self {
            backend: EngineBackendKind::Fake,
            swap: false,
            tier_overrides: BTreeMap::new(),
            fake: FakeCfg::default(),
            post_encoder: "nvenc".into(),
        }
    }
}

/// One `[[models]]` entry (consumed by the CUDA backend, WP-11).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCfg {
    pub id: String,
    pub family: String,
    #[serde(default)]
    pub weights: Option<String>,
    #[serde(default)]
    pub recipe: Option<String>,
    #[serde(default)]
    pub resident: bool,
    #[serde(default)]
    pub served_names: Vec<String>,
    /// Anything else the backend reads (`continuity`, `profile`, ...).
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

/// Which APIs are mounted (design §6.1 `[protocols]`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProtocolsCfg {
    pub openai_videos: bool,
    pub fastwan: bool,
    pub minimax: bool,
    pub fal: bool,
    pub fal_director: bool,
    pub ltx: bool,
    pub reactor: bool,
    /// Native `/fv/v1/*`.
    pub native: bool,
    /// fal apps (static prefixes, design §4.4).
    pub fal_apps: Vec<String>,
}

impl Default for ProtocolsCfg {
    fn default() -> Self {
        Self {
            openai_videos: true,
            fastwan: false,
            minimax: true,
            fal: true,
            fal_director: true,
            ltx: true,
            reactor: true,
            native: true,
            fal_apps: vec!["minimax/h3-max".into(), "minimax/h3-turbo".into(), "minimax/h3-draft".into()],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LimitsCfg {
    pub queue_max: usize,
    pub body_max_mb: usize,
}

impl Default for LimitsCfg {
    fn default() -> Self {
        Self { queue_max: 32, body_max_mb: 64 }
    }
}

/// `[ltx]` (design §4.5).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LtxCfg {
    /// `/v1/*` wait; default `server.sync_timeout_s`.
    pub sync_timeout_s: Option<u64>,
    /// Concurrent `/v1/*` generations per key (upstream: 2); 0 disables.
    pub v1_concurrency: usize,
    /// Lifetime of `result.video_url`; default `artifacts.url_ttl_s`.
    pub url_ttl_s: Option<u64>,
}

impl Default for LtxCfg {
    fn default() -> Self {
        Self { sync_timeout_s: None, v1_concurrency: 2, url_ttl_s: None }
    }
}

/// `[webrtc]`: consumed by the streaming packages (WP-13/14/15).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WebrtcCfg {
    /// Note: design §6.1 uses 70010/70000, above the 65535 port range;
    /// kept as u32 so that config parses, validated by the streaming packages.
    pub udp_port: u32,
    pub tcp_port: u32,
    pub public_ip: String,
    pub ice_servers: Vec<toml::Value>,
    /// Default WHIP target when a stream does not say (`cloudflare` \| `mediamtx`).
    pub whip_target: Option<String>,
    pub whip_token: Secret,
}

impl Default for WebrtcCfg {
    fn default() -> Self {
        Self {
            udp_port: 70010,
            tcp_port: 70000,
            public_ip: "auto".into(),
            ice_servers: Vec::new(),
            whip_target: None,
            whip_token: Secret::default(),
        }
    }
}

/// `[reactor]`: the Reactor local runtime (design §5.7, WP-13).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReactorCfg {
    /// Model id/served name to stream; unset: the first resident model with
    /// stream caps.
    pub model: Option<String>,
    /// Canvas short edge (default: the model's default tier).
    pub short_edge: Option<u32>,
    /// Initial aspect (`16:9`, `1:1`, `9:16`, `4:3`).
    pub aspect: String,
    /// Session seed; unset: `/start_session` `seed`, else drawn.
    pub seed: Option<u64>,
    /// RT `ORPHAN_TIMEOUT_SECONDS`.
    pub orphan_timeout_s: u64,
    /// RT `WEBRTC_CLIENT_PING_TIMEOUT_SECONDS`.
    pub ping_timeout_s: u64,
    pub max_connections: usize,
    /// H.264 encoder for H.264 peers: `nvenc` | `openh264` | `off` (VP8
    /// peers, such as the Python SDK, always get intra-only VP8).
    pub h264: String,
    pub h264_bitrate_bps: Option<u32>,
}

impl Default for ReactorCfg {
    fn default() -> Self {
        Self {
            model: None,
            short_edge: None,
            aspect: "16:9".into(),
            seed: None,
            orphan_timeout_s: 60,
            ping_timeout_s: 20,
            max_connections: 64,
            h264: "nvenc".into(),
            h264_bitrate_bps: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogCfg {
    /// `text` or `json`.
    pub format: String,
    /// `RUST_LOG`-style filter.
    pub filter: String,
}

impl Default for LogCfg {
    fn default() -> Self {
        Self { format: "text".into(), filter: "info".into() }
    }
}

/// The whole `serve.toml`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerCfg,
    pub auth: AuthCfg,
    pub artifacts: ArtifactsCfg,
    pub jobs: JobsCfg,
    pub engine: EngineCfg,
    pub models: Vec<ModelCfg>,
    pub aliases: BTreeMap<String, String>,
    pub protocols: ProtocolsCfg,
    pub ltx: LtxCfg,
    pub limits: LimitsCfg,
    pub webrtc: WebrtcCfg,
    pub reactor: ReactorCfg,
    pub log: LogCfg,
    /// Ed25519 seed for fal webhooks (normally `FV_WEBHOOK_ED25519_KEY`).
    pub webhook_key: Secret,
}

/// Why a config was refused.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {0}: {1}")]
    Read(PathBuf, std::io::Error),
    #[error("parsing {0}: {1}")]
    Parse(String, toml::de::Error),
    #[error("{0}")]
    Invalid(String),
}

/// Environment lookup (tests pass a map).
pub trait Env {
    fn var(&self, name: &str) -> Option<String>;
}

/// The process environment.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessEnv;

impl Env for ProcessEnv {
    fn var(&self, name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.trim().is_empty())
    }
}

impl Env for BTreeMap<String, String> {
    fn var(&self, name: &str) -> Option<String> {
        self.get(name).cloned().filter(|v| !v.trim().is_empty())
    }
}

/// `NAME` or its lower-case Runpod-secret spelling.
fn var2(env: &dyn Env, name: &str) -> Option<String> {
    env.var(name).or_else(|| env.var(&name.to_ascii_lowercase()))
}

fn parse_enum<T: for<'de> Deserialize<'de>>(name: &str, v: &str) -> Result<T, ConfigError> {
    T::deserialize(serde::de::value::StrDeserializer::<serde::de::value::Error>::new(v.trim()))
        .map_err(|e| ConfigError::Invalid(format!("{name}={v}: {e}")))
}

impl Config {
    /// Parses TOML text.
    pub fn from_toml(text: &str, origin: &str) -> Result<Self, ConfigError> {
        toml::from_str(text).map_err(|e| ConfigError::Parse(origin.to_owned(), e))
    }

    /// Defaults < file < environment, then validation.
    pub fn load(path: Option<&Path>, env: &dyn Env) -> Result<Self, ConfigError> {
        let mut c = match path {
            Some(p) => {
                let text = std::fs::read_to_string(p).map_err(|e| ConfigError::Read(p.to_owned(), e))?;
                Self::from_toml(&text, &p.display().to_string())?
            }
            None => Self::default(),
        };
        c.apply_env(env)?;
        c.validate()?;
        Ok(c)
    }

    /// Applies the environment overrides (module docs).
    pub fn apply_env(&mut self, env: &dyn Env) -> Result<(), ConfigError> {
        if let Some(v) = env.var("FV_SERVE_MODE") {
            self.server.mode = parse_enum("FV_SERVE_MODE", &v)?;
        }
        if let Some(v) = env.var("FV_BIND") {
            self.server.bind = v;
        }
        if let Some(p) = env.var("PORT") {
            let port: u16 = p.trim().parse().map_err(|_| ConfigError::Invalid(format!("PORT={p} is not a port")))?;
            let host = self.server.bind.rsplit_once(':').map_or("0.0.0.0", |(h, _)| h).to_owned();
            self.server.bind = format!("{host}:{port}");
        }
        if let Some(v) = env.var("FV_PUBLIC_BASE_URL") {
            self.server.public_base_url = Some(v);
        }
        if let Some(v) = env.var("FV_STATE_DIR") {
            self.server.state_dir = v.into();
        }
        if let Some(v) = env.var("FV_WORKER_ID") {
            self.server.worker_id = Some(v);
        }
        if self.server.worker_id.is_none() {
            self.server.worker_id = env.var("RUNPOD_POD_ID").or_else(|| env.var("HOSTNAME"));
        }
        if let Some(w) = env.var("FV_WEIGHTS") {
            for m in &mut self.models {
                if let Some(p) = &mut m.weights {
                    *p = p.replace("${FV_WEIGHTS}", &w);
                }
            }
        }
        if let Some(v) = env.var("FV_AUTH_MODE") {
            self.auth.mode = parse_enum("FV_AUTH_MODE", &v)?;
        }
        if let Some(v) = env.var("FV_API_KEYS") {
            self.auth.keys = Secret(v);
        }
        if let Some(v) = env.var("FV_URL_SIGNING_KEY") {
            self.artifacts.signing_key = Secret(v);
        }
        if let Some(v) = env.var("FV_WEBHOOK_ED25519_KEY") {
            self.webhook_key = Secret(v);
        }
        if let Some(v) = env.var("FV_WHIP_TOKEN") {
            self.webrtc.whip_token = Secret(v);
        }
        // RT's own environment names (reactor §3.3).
        for (k, slot) in [
            ("ORPHAN_TIMEOUT_SECONDS", &mut self.reactor.orphan_timeout_s),
            ("WEBRTC_CLIENT_PING_TIMEOUT_SECONDS", &mut self.reactor.ping_timeout_s),
        ] {
            if let Some(v) = env.var(k) {
                *slot = v
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .map(|s| s.ceil() as u64)
                    .ok_or_else(|| ConfigError::Invalid(format!("{k}={v} is not a positive number of seconds")))?;
            }
        }
        if let Some(v) = env.var("FV_REACTOR_MODEL") {
            self.reactor.model = Some(v);
        }
        if let Some(v) = env.var("FV_ENGINE") {
            self.engine.backend = parse_enum("FV_ENGINE", &v)?;
        }
        if let Some(v) = env.var("FV_JOB_STORE") {
            self.jobs.backend = parse_enum("FV_JOB_STORE", &v)?;
        }
        if let Some(v) = var2(env, "FV_CF_ACCOUNT_ID") {
            self.jobs.d1.account_id = Some(v);
        }
        if let Some(v) = var2(env, "FV_CF_API_TOKEN") {
            self.jobs.d1.api_token = Secret(v);
        }
        if let Some(v) = var2(env, "FV_D1_DATABASE_ID") {
            self.jobs.d1.database_id = Some(v);
        }
        if let Some(v) = env.var("FV_D1_API_BASE") {
            self.jobs.d1.api_base = Some(v);
        }
        if let Some(v) = env.var("FV_ARTIFACTS") {
            self.artifacts.backend = parse_enum("FV_ARTIFACTS", &v)?;
        }
        let s3 = &mut self.artifacts.s3;
        // Generic S3 first, then R2 (the provisioned default) wins.
        for (key, r2) in [("S3", false), ("R2", true)] {
            if let Some(v) = var2(env, &format!("FV_{key}_ENDPOINT")) {
                s3.endpoint = Some(v);
                if r2 {
                    s3.region = Some("auto".into());
                }
            }
            if let Some(v) = var2(env, &format!("FV_{key}_BUCKET")) {
                s3.bucket = Some(v);
            }
            if let Some(v) = var2(env, &format!("FV_{key}_ACCESS_KEY_ID")) {
                s3.access_key_id = Secret(v);
            }
            if let Some(v) = var2(env, &format!("FV_{key}_SECRET_ACCESS_KEY")) {
                s3.secret_access_key = Secret(v);
            }
        }
        if let Some(v) = env.var("FV_S3_REGION") {
            s3.region = Some(v);
        }
        if let Some(v) = env.var("FV_LOG_FORMAT") {
            self.log.format = v;
        }
        if let Some(v) = env.var("RUST_LOG") {
            self.log.filter = v;
        }
        Ok(())
    }

    /// Consistency checks that do not need I/O.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.bind_addr()?;
        if let Some(u) = &self.server.public_base_url {
            url::Url::parse(u).map_err(|e| ConfigError::Invalid(format!("server.public_base_url: {e}")))?;
        }
        if self.auth.mode == AuthMode::Keys && !self.auth.keys.is_empty() {
            fastvideo_serve_kit::KeyRing::from_hash_list(self.auth.keys.expose()).map_err(ConfigError::Invalid)?;
        }
        if self.jobs.backend == JobBackend::D1 && !self.d1_configured() {
            return Err(ConfigError::Invalid(
                "jobs.backend = d1 needs account id, API token and database id (FV_CF_ACCOUNT_ID, FV_CF_API_TOKEN, FV_D1_DATABASE_ID)".into(),
            ));
        }
        if self.artifacts.backend == ArtifactBackend::S3 && !self.s3_configured() {
            return Err(ConfigError::Invalid(
                "artifacts.backend = s3 needs endpoint, bucket and both keys (FV_R2_* or FV_S3_*)".into(),
            ));
        }
        for p in self.jobs.retention_s.keys() {
            if !fastvideo_protocol::ProtocolId::ALL.iter().any(|x| x.as_str() == p) {
                return Err(ConfigError::Invalid(format!("jobs.retention_s: unknown API `{p}`")));
            }
        }
        match self.engine.post_encoder.as_str() {
            "nvenc" | "cpu-test-x264" => {}
            other => return Err(ConfigError::Invalid(format!("engine.post_encoder: unknown `{other}`"))),
        }
        Ok(())
    }

    pub fn bind_addr(&self) -> Result<SocketAddr, ConfigError> {
        self.server
            .bind
            .parse()
            .map_err(|e| ConfigError::Invalid(format!("server.bind `{}`: {e}", self.server.bind)))
    }

    pub fn d1_configured(&self) -> bool {
        let d = &self.jobs.d1;
        d.account_id.is_some() && d.database_id.is_some() && !d.api_token.is_empty()
    }

    pub fn s3_configured(&self) -> bool {
        let s = &self.artifacts.s3;
        s.endpoint.is_some() && s.bucket.is_some() && !s.access_key_id.is_empty() && !s.secret_access_key.is_empty()
    }

    /// The job store `auto` resolves to.
    pub fn job_backend(&self) -> JobBackend {
        match self.jobs.backend {
            JobBackend::Auto if self.d1_configured() => JobBackend::D1,
            JobBackend::Auto => JobBackend::File,
            b => b,
        }
    }

    /// The artifact store `auto` resolves to.
    pub fn artifact_backend(&self) -> ArtifactBackend {
        match self.artifacts.backend {
            ArtifactBackend::Auto if self.s3_configured() => ArtifactBackend::S3,
            ArtifactBackend::Auto => ArtifactBackend::Local,
            b => b,
        }
    }

    pub fn shutdown_grace(&self) -> Duration {
        Duration::from_secs(self.server.shutdown_grace_s)
    }

    /// The config as JSON with every secret redacted (startup log).
    pub fn redacted(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn design_sketch_parses() {
        let text = r#"
[server]
bind = "0.0.0.0:8000"
public_base_url = "https://pod-8000.proxy.runpod.net"
state_dir = "/workspace/fv-state"
[auth]
mode = "keys"
[artifacts]
backend = "local"
url_ttl_s = 86400
[[models]]
id = "fasth3"
family = "h3"
weights = "${FV_WEIGHTS}/FastH3"
recipe = "4step-vsa"
resident = true
served_names = ["fasth3"]
continuity = "anchor-last-frame"
[aliases]
"MiniMax-H3" = "fasth3"
"MiniMax-H3-Max" = "fasth3"
[protocols]
openai_videos = true
fastwan = false
minimax = true
fal = true
fal_director = true
ltx = false
reactor = true
[webrtc]
udp_port = 70010
tcp_port = 70000
public_ip = "auto"
ice_servers = [{urls=["stun:stun.l.google.com:19302"]}]
[limits]
queue_max = 32
body_max_mb = 64
"#;
        let mut c = Config::from_toml(text, "sketch").unwrap();
        c.apply_env(&env(&[("FV_WEIGHTS", "/runpod-volume/weights"), ("PORT", "8080")])).unwrap();
        c.validate().unwrap();
        assert_eq!(c.models[0].weights.as_deref(), Some("/runpod-volume/weights/FastH3"));
        assert_eq!(c.models[0].extra["continuity"].as_str(), Some("anchor-last-frame"));
        assert_eq!(c.server.bind, "0.0.0.0:8080");
        assert_eq!(c.aliases["MiniMax-H3-Max"], "fasth3");
        assert!(!c.protocols.ltx && c.protocols.minimax);
        assert_eq!(c.job_backend(), JobBackend::File);
        assert_eq!(c.artifact_backend(), ArtifactBackend::Local);
    }

    #[test]
    fn unknown_keys_are_refused() {
        let e = Config::from_toml("[server]\nbindd = \"x\"\n", "t").unwrap_err();
        assert!(e.to_string().contains("bindd"), "{e}");
    }

    #[test]
    fn runpod_secrets_select_d1_and_r2_and_are_redacted() {
        let mut c = Config::default();
        c.apply_env(&env(&[
            ("fv_cf_account_id", "acct"),
            ("fv_cf_api_token", "cf-token-value"),
            ("FV_D1_DATABASE_ID", "1796e295-a7f0-4402-bbed-ec94ccb27c15"),
            ("fv_r2_bucket", "fv-media"),
            ("fv_r2_endpoint", "https://acct.r2.cloudflarestorage.com"),
            ("fv_r2_access_key_id", "AKID"),
            ("fv_r2_secret_access_key", "r2-secret-value"),
            ("FV_API_KEYS", &fastvideo_serve_kit::KeyRing::hash_hex("k")),
            ("FV_SERVE_MODE", "runpod-queue"),
            ("RUNPOD_POD_ID", "pod-123"),
        ]))
        .unwrap();
        c.validate().unwrap();
        assert_eq!(c.job_backend(), JobBackend::D1);
        assert_eq!(c.artifact_backend(), ArtifactBackend::S3);
        assert_eq!(c.artifacts.s3.region.as_deref(), Some("auto"));
        assert_eq!(c.server.mode, Mode::RunpodQueue);
        assert_eq!(c.server.worker_id.as_deref(), Some("pod-123"));
        let shown = format!("{} {:?}", c.redacted(), c);
        for secret in ["cf-token-value", "r2-secret-value", "AKID"] {
            assert!(!shown.contains(secret), "{secret} leaked");
        }
        assert!(shown.contains("fv-media"));
    }

    #[test]
    fn bad_values_are_errors() {
        let mut c = Config::default();
        assert!(c.apply_env(&env(&[("FV_SERVE_MODE", "lambda")])).is_err());
        assert!(c.apply_env(&env(&[("PORT", "http")])).is_err());
        let mut c = Config::default();
        c.jobs.backend = JobBackend::D1;
        assert!(c.validate().is_err(), "d1 without credentials");
        let mut c = Config::default();
        c.auth.keys = Secret("nothex".into());
        assert!(c.validate().is_err());
        let mut c = Config::default();
        c.jobs.retention_s.insert("minimax_v2".into(), 3600);
        c.validate().unwrap();
        c.jobs.retention_s.insert("nope".into(), 1);
        assert!(c.validate().is_err());
    }
}
