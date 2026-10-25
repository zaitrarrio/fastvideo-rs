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
//! | `FV_WORKERS_MAX` | `server.workers_max` (load-balanced replicas; above 1 only routes any worker can answer are served) |
//! | `FV_SERVE_FORWARD` (`1`) | `server.forward` (Vast PyWorker route) |
//! | `FV_CALLBACKS_ALLOW_PRIVATE` (`1`) | `server.callbacks_allow_private` (tests only: webhooks to loopback/private hosts) |
//! | `FV_WEIGHTS` | substituted for `${FV_WEIGHTS}` in `models[].weights` |
//! | `FV_AUTH_MODE`, `FV_API_KEYS` (SHA-256 hex list) | `auth.*` |
//! | `FV_ADMIN_TOKEN` | `auth.admin_token` (else made once and kept in `<state_dir>/admin_token`, mode 600) |
//! | `FV_ADMIN_TOKEN_RECIPIENT` | `auth.admin_token_recipient` (X25519 public key: the token sealed at `/fv/v1/admin/token/sealed`) |
//! | `FV_KEY_STORE` (`auto` \| `memory` \| `file` \| `d1`) | `auth.key_store` (minted API keys) |
//! | `FV_CONSOLE` (`0` \| `1`) | `server.console` (the `/console` pages) |
//! | `FV_CORS_ORIGINS` (`*` \| comma-separated origins \| `none`) | `server.cors_origins` |
//! | `FV_URL_SIGNING_KEY`, `FV_WEBHOOK_ED25519_KEY` | signing keys |
//! | `FV_ENGINE` (`fake` \| `cuda`) | `engine.backend` |
//! | `FV_ECHO_MODEL` (`0` \| `1`) | `engine.echo_model` (the `fv-echo` loopback duplex model: client camera/microphone in, overlaid echo out) |
//! | `FV_FAKE_H3_1080P` (`0` \| `1`), `FV_FAKE_DEVICE` (`a100`, `h100`, `sm80:80`, …) | `engine.fake.h3_1080p`, `engine.fake.device` |
//! | `FV_JOB_STORE` (`auto` \| `memory` \| `file` \| `d1`) | `jobs.backend` |
//! | `FV_CF_ACCOUNT_ID`, `FV_CF_API_TOKEN`, `FV_D1_DATABASE_ID` | `jobs.d1.*` |
//! | `FV_ARTIFACTS` (`auto` \| `local` \| `s3`) | `artifacts.backend` |
//! | `FV_R2_BUCKET`, `FV_R2_ENDPOINT`, `FV_R2_ACCESS_KEY_ID`, `FV_R2_SECRET_ACCESS_KEY` | `artifacts.s3.*` (R2: region `auto`) |
//! | `FV_S3_ENDPOINT`, `FV_S3_REGION`, `FV_S3_BUCKET`, `FV_S3_ACCESS_KEY_ID`, `FV_S3_SECRET_ACCESS_KEY` | `artifacts.s3.*` (generic S3) |
//! | `FV_LOG_FORMAT` (`text` \| `json`), `RUST_LOG` | logging |
//! | `FV_OTEL` (`0` \| `1`), `FV_OTEL_ENDPOINT`, `FV_OTEL_PROTOCOL` (`http/protobuf`), `FV_OTEL_SERVICE_NAME`, `FV_OTEL_SAMPLE_RATIO` (0..=1), `FV_OTEL_TIMEOUT_MS` | `[otel]`: OpenTelemetry (OTLP) export of the request traces; needs the `otel` cargo feature, off by default (docs/serve/tracing.md "OpenTelemetry export") |
//! | `FV_SERVE_ROLE` (`standalone` \| `worker`) | `server.role` (a worker of a cluster: an edge front, a direct worker or an executor; docs/serve/edge-control-plane.md) |
//! | `FV_INTERNAL_TOKEN` | `cluster.internal_token` (edge ↔ worker; never a user key) |
//! | `FV_GATEWAY_POOL` | `cluster.pool` (a worker's pool: its registration row in D1 `gw_workers`, the per-pool object socket; the variable keeps the retired gateway's name) |
//! | `FV_WORKER_DIRECT` (`0` \| `1`) | `cluster.direct` (clients call the worker with API keys; docs/control/gateway-less-auth.md) |
//! | `FV_DISPATCH_DO_URL`, `FV_DISPATCH_CAPACITY` | `dispatch.do_url` / `dispatch.capacity` (a worker's socket to its pool's Durable Object) |
//! | `FV_DISPATCH_FAMILIES`, `FV_DISPATCH_SESSIONS`, `FV_DISPATCH_DIRECT_UPLOAD`, `FV_DISPATCH_UPLOAD_PART_MIB` | `dispatch.families` / `sessions` / `direct_upload` / `upload_part_mib` (family Durable Objects, docs/serve/dispatch-do-family.md) |
//! | `FV_DISPATCH_FRONT` (`0` \| `1`), `FV_DISPATCH_ENDPOINT`, `FV_DISPATCH_MODEL_FAMILIES` (`model=family,…`), `FV_DISPATCH_MAX_QUEUED` | `dispatch.front` / `endpoint` / `model_families` / `max_queued` (and `FV_DISPATCH_STATUS_S`: `dispatch.status_s`): an API front behind the edge Worker (docs/serve/edge-control-plane.md) |
//! | `FV_MP4_FRAGMENTED` (`0` \| `1`) | `engine.mp4_fragmented` (append-only MP4 for overlapped uploads) |
//! | `FV_JOB_DEADLINE_S` | `engine.job_deadline_s` (wall-clock limit of one job's generation; 0 = none) |
//! | `FV_ARTIFACTS_DIR` | `artifacts.local_dir` (local artifacts shared by processes on one host) |
//! | `FV_JOBS_HEARTBEAT_S` | `jobs.heartbeat_s` (D1 heartbeat of unfinished jobs) |
//! | `FV_UPSAMPLER_BASE_URL`, `FV_UPSAMPLER_MODEL`, `FV_UPSAMPLER_TIMEOUT_MS`, `FV_UPSAMPLER_HEDGE`, `FV_UPSAMPLER_DECK_SIZE`, `FV_UPSAMPLER_DEFAULT_ON`, `FV_UPSAMPLER_API_KEY_ENV` | `[upsampler]` (a base URL alone creates it; docs/serve/prompt-upsampling.md) |
//! | `FV_UPSAMPLER_LOCAL` (`0` \| `1`) | `upsampler.local` (the LLM shares this GPU; unset: a loopback `base_url` is local). Unset `timeout_ms` / `hedge` / `deck_size` of a local server default to 60000 / off / 2 |
//! | `FV_UPSAMPLER_API_KEY` (or the variable `api_key_env` names) | `upsampler.api_key` (secret) |
//! | `FV_CAUSAL_DEFAULT_MAX_S`, `FV_CAUSAL_HARD_MAX_S` | `streams.causal_default_max_s` / `causal_hard_max_s` (live SF-Wan session length, 120 / 300 s of video) |
//! | `FV_SEGMENTS_MODEL`, `FV_SEGMENTS_IDLE_TTL_S`, `FV_SEGMENTS_ENCODER` | `segments.model` / `idle_ttl_s` / `encoder` (the `/v1/segments` adapter, docs/serve/segments.md) |
//! | `FV_CF_STREAM_API_TOKEN` (secret; Account > Stream > Edit), `FV_CF_ACCOUNT_ID` (shared with D1), `FV_CF_STREAM_API_BASE`, `FV_CF_STREAM_NAME_PREFIX`, `FV_CF_STREAM_CUSTOMER_DOMAIN` | `outputs.cloudflare.*`: the Cloudflare Stream default output of live streams (docs/serve/stream-outputs.md) |
//! | `FV_OUTPUTS_ALLOW_PRIVATE` (`1`), `FV_OUTPUTS_SRT_LATENCY_MS`, `FV_OUTPUTS_RTMPS_CA_FILE` | `outputs.allow_private` (tests / LAN relays: loopback and private destinations), `outputs.srt_latency_ms`, `outputs.rtmps_ca_file` (extra CAs for RTMPS, PEM) |
//! | `FV_DIRECTOR_ALLOW_OUTPUT` (`1`) | `director.allow_output` (accept `configure.output`, not only `x_fv_output`) |
//! | `FV_MODERATION_NCII` (`off` \| `keyframe` \| `all`), `FV_MODERATION_NCII_DIR`, `FV_MODERATION_NCII_THRESHOLD` | `[moderation]` (the NCII prompt guard; docs/ports/h3-turbo-loras.md) |
//!
//! `auto` picks D1 when the three D1 values are set (else the file store)
//! and S3/R2 when bucket, endpoint and both keys are set (else local).
//!
//! `[cluster]` was named `[gateway]` until plan F3b (2026-10-25): a file
//! that still says `[gateway]` is read as `[cluster]` and noted in
//! [`Config::deprecations`]; a file with both is refused. The configs
//! fv-control and the other launchers send keep `[gateway]` until every
//! image they can boot reads `[cluster]` (docs/adr/0011-deploy-section.md,
//! "Renaming a worker-config section").

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
        f.write_str(if self.0.is_empty() {
            "<unset>"
        } else {
            "<redacted>"
        })
    }
}

impl Serialize for Secret {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(if self.0.is_empty() {
            "<unset>"
        } else {
            "<redacted>"
        })
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
    /// Serve the `/console` pages (docs/serve/console.md).
    pub console: bool,
    /// Mount `POST /fv/v1/forward` (the Vast PyWorker target, WP-16).
    pub forward: bool,
    /// Let webhooks/callbacks (fal `fal_webhook`, MiniMax `callback_url`)
    /// reach loopback and private hosts. Off in production (the SSRF
    /// guard); the client-compat suites turn it on to receive them locally.
    pub callbacks_allow_private: bool,
    /// Workers behind the load balancer (Runpod serverless LB `workers.max`,
    /// design §6.2). Above 1, only routes any worker can answer are served
    /// (`crate::multiworker`): async jobs need D1 jobs and S3/R2 artifacts;
    /// cancel, uploads and streaming sessions are never served.
    pub workers_max: u32,
    /// Browser origins allowed to call the API cross-origin (CORS on every
    /// route, preflights included: fal-style uploads from another page,
    /// the console opened on another host). `["*"]` (the default, as fal's
    /// own endpoints) allows any origin; otherwise exact origins such as
    /// `https://app.example.com`; `[]` turns CORS off. Credentials
    /// (cookies) are never allowed: the APIs authenticate with headers.
    pub cors_origins: Vec<String>,
    /// `worker`: a GPU worker of a cluster (docs/serve/edge-control-plane.md):
    /// mounts `/fv/v1/internal/*`; its `[cluster]` table says how clients
    /// reach it (behind the edge, or `direct`).
    pub role: Role,
}

/// What this process is in a cluster.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Role {
    /// A server of its own.
    #[default]
    Standalone,
    /// A worker of a cluster (`[cluster]`).
    Worker,
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
            console: true,
            forward: false,
            callbacks_allow_private: false,
            workers_max: 1,
            cors_origins: vec!["*".into()],
            role: Role::Standalone,
        }
    }
}

/// Where minted API keys live (`auth.key_store`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyStoreBackend {
    /// D1 when D1 is configured (and built), else `file`.
    #[default]
    Auto,
    Memory,
    /// `state_dir/api_keys.json`.
    File,
    /// Cloudflare D1 table `api_keys`.
    D1,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthCfg {
    pub mode: AuthMode,
    /// SHA-256 hex hashes of the accepted keys (normally `FV_API_KEYS`).
    pub keys: Secret,
    /// Admin token for `/fv/v1/admin/*` (normally `FV_ADMIN_TOKEN`); when
    /// unset one is made on the first start and kept in
    /// `<state_dir>/admin_token` (mode 600; `crate::admin_token`).
    pub admin_token: Secret,
    /// X25519 public key (base64) the admin token is sealed to at
    /// `GET /fv/v1/admin/token/sealed` (`FV_ADMIN_TOKEN_RECIPIENT`).
    pub admin_token_recipient: Option<String>,
    /// Minted-key store.
    pub key_store: KeyStoreBackend,
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
    /// Local artifacts directory (default `state_dir/artifacts`): processes
    /// on one host (a gateway and its workers in tests) can share it.
    pub local_dir: Option<PathBuf>,
}

impl Default for ArtifactsCfg {
    fn default() -> Self {
        Self {
            backend: ArtifactBackend::Auto,
            url_ttl_s: 86_400,
            signing_key: Secret::default(),
            s3: S3Cfg::default(),
            local_dir: None,
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
    /// D1 heartbeat of this worker's unfinished jobs (`updated_at`); the
    /// gateway declares a worker lost after its pool's `stale_after_s`.
    pub heartbeat_s: u64,
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
            heartbeat_s: 60,
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
    /// Write the MP4 through `ffmpeg` (libx264) when it is on `PATH`
    /// (`false`: never; the placeholder stands in). The encode of a
    /// full-size clip takes seconds in a debug build: timing tests turn it
    /// off to measure the serving path alone.
    pub mp4: bool,
    /// Give the fake H3 max / turbo models the native 1080P tier, as an
    /// 80 GB-class GPU serves them (`FV_FAKE_H3_1080P`).
    pub h3_1080p: bool,
    /// A simulated GPU for the startup capability check (`a100`, `l40s`,
    /// `h100`, `b200`, `rtx-pro-6000`, `sm<NN>[:<GiB>]`; `FV_FAKE_DEVICE`):
    /// a model it cannot run is failed with the reason (the fake H3 models
    /// need FP8, sm89+). Unset: no check.
    pub device: Option<String>,
}

impl Default for FakeCfg {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            step_ms: 2,
            load_ms: 0,
            rtf: None,
            all_resident: true,
            placeholder_output: true,
            mp4: true,
            h3_1080p: false,
            device: None,
        }
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
    /// Post-processing H.264 encoder for crops: `auto` (NVENC when the
    /// startup probe encodes, else `cpu-test-x264`), `nvenc` (deployed) or
    /// `cpu-test-x264` (CPU CI only).
    pub post_encoder: String,
    /// Serve the `fv-echo` loopback duplex model next to the backend's
    /// models (design §5.11): it re-shows the client's camera with an
    /// overlay, so WebRTC ingest is testable end to end without a GPU.
    pub echo_model: bool,
    /// Write batch MP4s fragmented (append-only, `+frag_keyframe+empty_moov`)
    /// instead of faststart, so a direct upload can follow the file while it
    /// is written (docs/serve/dispatch-do-family.md §7.2).
    pub mp4_fragmented: bool,
    /// Wall-clock limit, in seconds, of one batch job's generation (from the
    /// start of `generate`, after any on-demand load; `FV_JOB_DEADLINE_S`).
    /// Past it the engine cancels the job at its next denoise step and it
    /// fails with `timeout` (HTTP 504 on a sync route). 0: no limit.
    /// Streaming sessions are not bounded by it. At most a day.
    pub job_deadline_s: u64,
}

impl Default for EngineCfg {
    fn default() -> Self {
        Self {
            backend: EngineBackendKind::Fake,
            swap: false,
            tier_overrides: BTreeMap::new(),
            fake: FakeCfg::default(),
            post_encoder: "auto".into(),
            echo_model: false,
            mp4_fragmented: false,
            job_deadline_s: 0,
        }
    }
}

impl EngineCfg {
    /// The longest `job_deadline_s` accepted (a day): a larger value is a
    /// unit mistake (milliseconds), not a deadline.
    pub const JOB_DEADLINE_MAX_S: u64 = 86_400;

    /// `job_deadline_s` as the engine takes it (`None` for 0).
    pub fn job_deadline(&self) -> Option<std::time::Duration> {
        (self.job_deadline_s > 0).then(|| std::time::Duration::from_secs(self.job_deadline_s))
    }
}

/// One `[[models]]` entry (consumed by the CUDA backend, WP-11): a variant
/// of the family manifests (`families/*.toml`, plan A3) and the
/// deployment's choices about it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCfg {
    /// The family-file variant: its id (`fasth3-4step-vsa`), one of its
    /// served names (`fasth3`) or a tier alias (`h3-turbo`).
    pub variant: String,
    /// The id this deployment serves the model under (default: the id of
    /// the variant `variant` resolves to, so `variant = "h3-max"` serves
    /// `sol-h3`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The weight directory when it is not the variant's tree under
    /// `FV_WEIGHTS`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
    #[serde(default)]
    pub resident: bool,
    /// Extra names this deployment serves the model under (the variant's
    /// own served names always apply).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub served_names: Vec<String>,
    /// Anything else the backend reads (`continuity`, `warmup`, ...).
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

impl ModelCfg {
    /// The entry's name in messages: its `id`, else the variant as written.
    pub fn label(&self) -> &str {
        self.id.as_deref().unwrap_or(&self.variant)
    }

    /// The keys an entry no longer takes (plan A3): the family and the
    /// recipe are the variant's, the public names come from the family file.
    const RETIRED_KEYS: [(&'static str, &'static str); 2] = [
        ("family", "the variant's family is in families/*.toml"),
        (
            "recipe",
            "name the variant (its id, a served name or a tier alias) in `variant`",
        ),
    ];

    fn validate(&self) -> Result<(), ConfigError> {
        if self.variant.trim().is_empty() {
            return Err(ConfigError::Invalid(format!(
                "[[models]] `{}`: `variant` is empty (a variant id, a served name or a tier alias of families/*.toml)",
                self.label()
            )));
        }
        for (k, why) in Self::RETIRED_KEYS {
            if self.extra.contains_key(k) {
                return Err(ConfigError::Invalid(format!(
                    "[[models]] `{}`: `{k}` is no longer a key: {why}",
                    self.label()
                )));
            }
        }
        Ok(())
    }
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
            // = fastvideo_fal::DEFAULT_APPS (the H3 tiers plus fal's own
            // `minimax/h3-max-turbo` and base `minimax/h3` ids).
            fal_apps: [
                "minimax/h3-max",
                "minimax/h3-turbo",
                "minimax/h3-draft",
                "minimax/h3-max-turbo",
                "minimax/h3",
            ]
            .map(String::from)
            .to_vec(),
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
        Self {
            queue_max: 32,
            body_max_mb: 64,
        }
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
        Self {
            sync_timeout_s: None,
            v1_concurrency: 2,
            url_ttl_s: None,
        }
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
    /// H.264 encoder for native `/fv/v1/streams` WHIP publishing
    /// (`FV_STREAM_ENCODER`): `auto` (NVENC when the startup probe encodes,
    /// else OpenH264) | `nvenc` | `openh264` | `x264-test`.
    pub encoder: String,
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
            encoder: "auto".into(),
        }
    }
}

/// `[streams]`: limits shared by every front-end that opens a live stream
/// (design §5.2). Causal (SF-Wan) sessions get `causal_default_max_s` of
/// video unless the request asks for another length, never more than
/// `causal_hard_max_s`; a `reset` restarts the clock, up to
/// `causal_hard_max_s` for the whole session.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StreamsCfg {
    /// `FV_CAUSAL_DEFAULT_MAX_S` (120, the R12 study's unattended limit).
    pub causal_default_max_s: u32,
    /// `FV_CAUSAL_HARD_MAX_S` (300).
    pub causal_hard_max_s: u32,
}

impl Default for StreamsCfg {
    fn default() -> Self {
        let l = fastvideo_protocol::CausalLimits::default();
        Self {
            causal_default_max_s: l.default_max_s,
            causal_hard_max_s: l.hard_max_s,
        }
    }
}

impl StreamsCfg {
    pub fn causal_limits(&self) -> fastvideo_protocol::CausalLimits {
        fastvideo_protocol::CausalLimits {
            default_max_s: self.causal_default_max_s,
            hard_max_s: self.causal_hard_max_s,
        }
    }
}

/// `[segments]`: the `/v1/segments` adapter (docs/serve/segments.md): a
/// control plane pulls an infinite stream segment by segment out of one
/// causal session per `stream_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SegmentsCfg {
    /// Mount the routes (with `protocols.native`).
    pub enabled: bool,
    /// The causal model a request without `model` uses (`FV_SEGMENTS_MODEL`);
    /// empty: the first causal model the engine serves.
    pub model: Option<String>,
    /// Close a stream's session after this long without a segment request
    /// (`FV_SEGMENTS_IDLE_TTL_S`, 120).
    pub idle_ttl_s: u64,
    /// Session length in video seconds; `None`: `[streams]
    /// causal_default_max_s`, never more than `causal_hard_max_s`.
    pub max_seconds: Option<u32>,
    /// Signed `video_url` lifetime (3600).
    pub url_ttl_s: u64,
    /// `nvenc` (production) | `libx264` (CPU tests) (`FV_SEGMENTS_ENCODER`).
    pub encoder: String,
    /// Finished segments kept for `GET` (256).
    pub keep: usize,
    /// Segments a stream may have waiting (8).
    pub queue: usize,
}

impl Default for SegmentsCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            model: None,
            idle_ttl_s: 120,
            max_seconds: None,
            url_ttl_s: 3600,
            encoder: "nvenc".into(),
            keep: 256,
            queue: 8,
        }
    }
}

impl SegmentsCfg {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !matches!(self.encoder.as_str(), "nvenc" | "libx264") {
            return Err(ConfigError::Invalid(format!(
                "[segments] encoder `{}`: nvenc or libx264",
                self.encoder
            )));
        }
        if self.max_seconds == Some(0) {
            return Err(ConfigError::Invalid(
                "[segments] max_seconds must be 1 or more".into(),
            ));
        }
        Ok(())
    }
}

/// `[director]`: the fal WMA director (design §5.6, WP-14). The WebRTC host
/// comes from `[webrtc]`; the apps from `[protocols] fal_apps` (each gets
/// `{app}/director`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DirectorCfg {
    /// `default_chunk_duration`: the chunk length of sessions that send no
    /// `configure.chunk_duration` (5 or 10 s; default 5). Snapped to the
    /// nearest length the model serves (a model whose clips stop short of
    /// 10 s runs 5 s), else clamped to its clip range.
    pub chunk_seconds: f64,
    /// `max_session_seconds` in video time; 0 = unlimited.
    pub max_session_seconds: u64,
    /// H.264 encoder: `auto` (NVENC when the startup probe encodes, else
    /// OpenH264), `nvenc` or `openh264`.
    pub encoder: String,
    /// Answer VP8 to offers without H.264 (ffmpeg `libvpx`, else
    /// intra-only libwebp).
    pub vp8_fallback: bool,
    /// Built chunks queued behind the one playing; 0 (the default): no
    /// cap, the director builds the next chunk as soon as the last one is
    /// built, until the session stops. Queued chunks are host RAM (RGB at
    /// the delivered size: ~370 MB per 5 s 768p chunk).
    pub buffer_chunks: usize,
    /// The memory guard in MiB: generation waits for playout while this
    /// much video is queued (logged). 0 (the default): half the host's
    /// available memory at startup, at least 2 GiB.
    pub buffer_max_mb: u64,
    /// Video bitrate in bit/s; 0 = by canvas.
    pub video_bitrate: u32,
    /// Causal models (LongLive, SF-Wan): blocks per director `chunk`
    /// (4 blocks of 12 frames = 3 s at 16 fps).
    pub causal_chunk_blocks: u32,
    /// Causal models: 0 (the default): the rollout never pauses (only the
    /// memory guard holds it). Above 0: seconds of video kept queued for
    /// playout; above it the rollout pauses at the next block boundary
    /// (shorter prompt latency).
    pub causal_lead_seconds: f64,
    /// Accept `configure.output` (fal's schema forbids extra fields, so off
    /// by default: `x_fv_output` is always accepted; docs/serve/stream-outputs.md).
    pub allow_output: bool,
    /// Upsampling sessions: a chunk whose prompt expansion is not ready
    /// waits for it (at most `[upsampler] timeout`). Off (the default): it
    /// builds with the unexpanded prompt at once.
    pub upsample_wait: bool,
}

impl Default for DirectorCfg {
    fn default() -> Self {
        Self {
            chunk_seconds: 5.0,
            max_session_seconds: 0,
            encoder: "auto".into(),
            vp8_fallback: true,
            buffer_chunks: 0,
            buffer_max_mb: 0,
            video_bitrate: 0,
            causal_chunk_blocks: 4,
            causal_lead_seconds: 0.0,
            allow_output: false,
            upsample_wait: false,
        }
    }
}

/// `[outputs]`: where live streams publish (docs/serve/stream-outputs.md).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputsCfg {
    /// Loopback / private / link-local destinations allowed (MediaMTX
    /// tests, LAN relays). Off: such URLs are refused, also when a name
    /// resolves to one.
    pub allow_private: bool,
    /// SRT receiver latency (ffmpeg `latency`), 20..8000 ms.
    pub srt_latency_ms: u32,
    /// RTMPS: a PEM file of extra CA certificates trusted besides the
    /// system's (`FV_OUTPUTS_RTMPS_CA_FILE`; a private CA, or a test CA).
    /// RTMPS always verifies the server's certificate and name
    /// (docs/serve/stream-outputs.md "RTMPS TLS").
    pub rtmps_ca_file: Option<PathBuf>,
    /// The default output: a Cloudflare Stream live input per stream.
    pub cloudflare: CloudflareOutputCfg,
}

impl Default for OutputsCfg {
    fn default() -> Self {
        Self {
            allow_private: false,
            srt_latency_ms: 1000,
            rtmps_ca_file: None,
            cloudflare: CloudflareOutputCfg::default(),
        }
    }
}

/// `[outputs.cloudflare]`: set `account_id` and `api_token` to make
/// Cloudflare Stream the default output.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CloudflareOutputCfg {
    /// `FV_CF_ACCOUNT_ID`.
    pub account_id: String,
    /// `FV_CF_STREAM_API_TOKEN` (Account > Stream > Edit).
    pub api_token: Secret,
    /// `FV_CF_STREAM_API_BASE` (tests point it at a fake server).
    pub api_base: String,
    /// `auto` (WebRTC off, RTMPS/SRT automatic: HLS needs it) | `off` | `automatic`.
    pub recording: crate::stream_outputs::Recording,
    /// `auto` (delete, or disable when it recorded) | `delete` | `disable` | `keep`.
    pub on_stop: crate::stream_outputs::OnStop,
    /// `deleteRecordingAfterDays` of recording inputs (Cloudflare's minimum is 30).
    pub delete_recording_after_days: Option<u32>,
    /// `customer-<code>.cloudflarestream.com` when the API does not name it.
    pub customer_domain: Option<String>,
    /// Live inputs are named `<name_prefix><stream id>` (`fv-`; tests use
    /// `fv-test-`, `FV_CF_STREAM_NAME_PREFIX`).
    pub name_prefix: String,
}

impl Default for CloudflareOutputCfg {
    fn default() -> Self {
        Self {
            account_id: String::new(),
            api_token: Secret::default(),
            api_base: "https://api.cloudflare.com/client/v4".into(),
            recording: crate::stream_outputs::Recording::Auto,
            on_stop: crate::stream_outputs::OnStop::Auto,
            delete_recording_after_days: None,
            customer_domain: None,
            name_prefix: "fv-".into(),
        }
    }
}

/// `[upsampler]`: the prompt upsampler (docs/serve/prompt-upsampling.md).
/// Any OpenAI-compatible `/chat/completions` server. Absent: no prompt is
/// ever rewritten and every expansion flag stays an accepted no-op.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpsamplerCfg {
    /// The API root, e.g. `https://api.openai.com/v1` (`/chat/completions`
    /// is appended).
    pub base_url: String,
    /// The chat model.
    pub model: String,
    /// The environment variable holding the API key (default
    /// `FV_UPSAMPLER_API_KEY`); unset or empty: no `authorization` header.
    pub api_key_env: String,
    /// The key itself (local development only; normally from
    /// `api_key_env`). Never printed.
    pub api_key: Secret,
    /// One call, every hedged attempt included (default 10000; 60000 for a
    /// [`local`](Self::local) server). Read it with [`Self::timeout_ms`].
    pub timeout_ms: Option<u64>,
    /// Two concurrent attempts per call, first success wins (default on;
    /// off for a local server). Read it with [`Self::hedge`].
    pub hedge: Option<bool>,
    /// The LLM runs next to fv-serve on the same GPU (the llama-server
    /// sidecar, docs/serve/upsampler-local.md; `FV_UPSAMPLER_LOCAL`).
    /// Unset: a loopback `base_url` (127.0.0.0/8, ::1, localhost) is local.
    /// A local server decodes 2-3x slower while a chunk denoises
    /// (docs/serve/e2e/gpu-check-2026-10-13.md), so its unset `timeout_ms`,
    /// `hedge` and `deck_size` default to 60000, off and 2.
    pub local: Option<bool>,
    /// Director: prior segments the LLM sees when `configure.memory` is
    /// absent (default 12, 1..50).
    pub memory: u32,
    /// Clients that omit the flag get their protocol's documented default
    /// (fal `balanced`, MiniMax H3-Max `balanced`, fal ingredient and LTX
    /// on, director on). Off (default): upsampling is opt-in per request.
    pub default_on: bool,
    /// Prompt budgets per model family, in characters (used while the
    /// family's tokenizer is not available; else its token limit).
    pub max_chars: MaxCharsCfg,
    /// Director: the LLM's context in tokens (default 28000: a local
    /// Qwen3-4B / Qwen3-VL-4B at 32k, with room for the answer).
    pub context_tokens: usize,
    /// Director: segments planned ahead (fal's `prompt_deck_size`, default
    /// 6; 2 for a local server). Read it with [`Self::deck_size`].
    pub deck_size: Option<usize>,
    /// The LLM is a vision model (Qwen3-VL behind llama-server `--mmproj`):
    /// the director sends the first frame and end frames as `image_url`
    /// parts (`FV_UPSAMPLER_VISION`).
    pub vision: bool,
    /// Ask the LLM server's `/tokenize` (llama-server, vLLM) for exact
    /// premise counts; without it, or off, tokens are estimated.
    pub tokenize: bool,
    /// Text-encoder `tokenizer.json` per family (default: under the first
    /// `[[models]]` entry's weights of that family).
    pub tokenizers: TokenizersCfg,
}

/// `[upsampler.tokenizers]`: paths to `tokenizer.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TokenizersCfg {
    pub h3: Option<String>,
    pub wan: Option<String>,
    pub ltx: Option<String>,
}

impl Default for UpsamplerCfg {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: String::new(),
            api_key_env: "FV_UPSAMPLER_API_KEY".into(),
            api_key: Secret::default(),
            timeout_ms: None,
            hedge: None,
            local: None,
            memory: 12,
            default_on: false,
            max_chars: MaxCharsCfg::default(),
            context_tokens: fastvideo_upsampler::director::DEFAULT_CONTEXT_TOKENS,
            deck_size: None,
            vision: false,
            tokenize: true,
            tokenizers: TokenizersCfg::default(),
        }
    }
}

/// `[upsampler.max_chars]`: each family's text-encoder budget (see
/// docs/serve/prompt-upsampling.md §Limits).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MaxCharsCfg {
    /// Wan, FastWan, SF-Wan, LongLive: umT5 512 tokens.
    pub wan: usize,
    /// LTX-2.x: Gemma 3, 1024 tokens.
    pub ltx: usize,
    /// H3: 1024 tokens, refused beyond.
    pub h3: usize,
    pub other: usize,
}

impl Default for MaxCharsCfg {
    fn default() -> Self {
        let d = fastvideo_upsampler::MaxChars::default();
        Self {
            wan: d.wan,
            ltx: d.ltx,
            h3: d.h3,
            other: d.other,
        }
    }
}

/// `[upsampler]` defaults for an LLM elsewhere / a local sidecar on this GPU.
pub const UPSAMPLER_TIMEOUT_MS: u64 = 10_000;
pub const UPSAMPLER_LOCAL_TIMEOUT_MS: u64 = 60_000;
pub const UPSAMPLER_DECK_SIZE: usize = 6;
pub const UPSAMPLER_LOCAL_DECK_SIZE: usize = 2;

impl UpsamplerCfg {
    /// Whether the LLM shares this host's GPU: `local`, else a loopback
    /// `base_url`.
    pub fn is_local(&self) -> bool {
        self.local.unwrap_or_else(|| {
            url::Url::parse(self.base_url.trim()).is_ok_and(|u| match u.host() {
                Some(url::Host::Ipv4(a)) => a.is_loopback(),
                Some(url::Host::Ipv6(a)) => a.is_loopback(),
                Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
                None => false,
            })
        })
    }

    /// `timeout_ms`, else 60000 for a local server (a ~1000-token segment
    /// at the 32-38 tok/s a local 4B decodes beside a denoising chunk takes
    /// ~30 s), else 10000.
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms.unwrap_or(if self.is_local() {
            UPSAMPLER_LOCAL_TIMEOUT_MS
        } else {
            UPSAMPLER_TIMEOUT_MS
        })
    }

    /// `hedge`, else off for a local server (a hedge doubles its load), else on.
    pub fn hedge(&self) -> bool {
        self.hedge.unwrap_or(!self.is_local())
    }

    /// `deck_size`, else 2 for a local server (every planned segment is an
    /// LLM call competing with the video model for the GPU), else 6.
    pub fn deck_size(&self) -> usize {
        self.deck_size.unwrap_or(if self.is_local() {
            UPSAMPLER_LOCAL_DECK_SIZE
        } else {
            UPSAMPLER_DECK_SIZE
        })
    }

    /// The service settings (the transport is built from `base_url`,
    /// `model` and the key).
    pub fn settings(&self) -> fastvideo_upsampler::Settings {
        let m = self.max_chars;
        fastvideo_upsampler::Settings {
            model: self.model.clone(),
            timeout: Duration::from_millis(self.timeout_ms()),
            hedge: self.hedge(),
            memory: self.memory as usize,
            max_chars: fastvideo_upsampler::MaxChars {
                wan: m.wan,
                ltx: m.ltx,
                h3: m.h3,
                other: m.other,
            },
            max_tokens: fastvideo_upsampler::MaxTokens::default(),
            default_on: self.default_on,
            context_tokens: self.context_tokens,
            deck_size: self.deck_size(),
            vision: self.vision,
            tokenize: self.tokenize,
        }
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: String| Err(ConfigError::Invalid(format!("[upsampler]: {m}")));
        let u = url::Url::parse(self.base_url.trim())
            .map_err(|e| ConfigError::Invalid(format!("[upsampler] base_url: {e}")))?;
        if !matches!(u.scheme(), "http" | "https") {
            return bad(format!("base_url must be http(s), got {}", u.scheme()));
        }
        if self.model.trim().is_empty() {
            return bad("model is required (FV_UPSAMPLER_MODEL)".into());
        }
        if !(100..=120_000).contains(&self.timeout_ms()) {
            return bad(format!(
                "timeout_ms must be 100..120000, got {}",
                self.timeout_ms()
            ));
        }
        if !(1..=50).contains(&self.memory) {
            return bad(format!("memory must be 1..50, got {}", self.memory));
        }
        let m = self.max_chars;
        if [m.wan, m.ltx, m.h3, m.other]
            .iter()
            .any(|&n| !(200..=20_000).contains(&n))
        {
            return bad("max_chars values must be 200..20000".into());
        }
        if !(4_096..=1_000_000).contains(&self.context_tokens) {
            return bad(format!(
                "context_tokens must be 4096..1000000, got {}",
                self.context_tokens
            ));
        }
        if !(1..=12).contains(&self.deck_size()) {
            return bad(format!("deck_size must be 1..12, got {}", self.deck_size()));
        }
        Ok(())
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
    /// H.264 encoder for H.264 peers: `auto` (NVENC when the startup probe
    /// encodes, else OpenH264) | `nvenc` | `openh264` | `off` (VP8
    /// peers, such as the Python SDK, always get VP8: inter-frame through
    /// ffmpeg `libvpx`, intra-only libwebp when ffmpeg has no libvpx).
    pub h264: String,
    pub h264_bitrate_bps: Option<u32>,
    /// Command set (`FV_REACTOR_MODE`): `auto` (the model's own: clip or
    /// causal) | `avatar` (the script avatar, Reactor `ltx`, on an
    /// image-to-video model with audio such as LTX-2.5).
    pub mode: String,
    /// Avatar: the longest generation window, seconds of speech.
    pub avatar_window_s: f64,
    /// Avatar: delivered canvas `WxH` (Reactor `ltx`: 640x352; LTX
    /// generates it padded to its multiple and it is centre-cropped).
    pub avatar_size: String,
    /// Avatar: session ceiling in video seconds (design §5.2).
    pub avatar_session_max_s: u32,
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
            h264: "auto".into(),
            h264_bitrate_bps: None,
            mode: "auto".into(),
            avatar_window_s: 10.0,
            avatar_size: "640x352".into(),
            avatar_session_max_s: 1800,
        }
    }
}

/// `[cluster]`: a worker's cluster settings (`server.role = "worker"`).
/// Named `[gateway]` until plan F3b, after the retired gateway
/// (docs/serve/edge-control-plane.md §9); that name is still read
/// ([`Config::from_toml`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClusterCfg {
    /// The cluster's shared secret (`FV_INTERNAL_TOKEN`): the edge's and
    /// fv-control's calls to the worker.
    pub internal_token: Secret,
    /// Worker: its pool (`FV_GATEWAY_POOL`); pod workers with a public base
    /// URL upsert their row in D1 `gw_workers` (the table keeps the retired
    /// gateway's name: fv-autoscale's pod pools read it, and images of
    /// every age write it).
    pub pool: Option<String>,
    /// Worker: register in `gw_workers` (pods; needs D1). fv-control's
    /// workers turn it off at boot (fv-boot, the shell boots).
    pub register: bool,
    /// A direct worker (fv-control `control_plane: "direct"`,
    /// `FV_WORKER_DIRECT`): clients call it directly, so `auth.mode`
    /// applies as on a standalone server (API keys, the shared minted-key
    /// store, `FV_ADMIN_TOKEN` for the admin routes) and the internal token
    /// guards only `/fv/v1/internal/*` (drain and status for the controller).
    pub direct: bool,
    /// D1 poll for SSE and sync waits.
    pub watch_poll_ms: u64,
}

impl Default for ClusterCfg {
    fn default() -> Self {
        Self {
            internal_token: Secret::default(),
            pool: None,
            register: true,
            direct: false,
            watch_poll_ms: 1000,
        }
    }
}

/// `[dispatch]` of a worker: the socket to its pool's Durable Object, or
/// one socket per model family (docs/serve/dispatch-do-family.md).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DispatchCfg {
    /// The fv-edge Worker's base URL (`https://…`); unset: no socket.
    pub do_url: Option<String>,
    /// Jobs this worker takes at once (running + waiting on its GPU),
    /// across every family it serves (the arbiter's budget).
    pub capacity: u32,
    /// Status (heartbeat) frame interval.
    pub status_s: u64,
    /// The families whose Durable Objects this worker serves (protocol 2:
    /// credits, the arbiter, sessions, direct uploads); empty: the single
    /// per-pool socket of `cluster.pool`.
    pub families: Vec<String>,
    /// Streaming sessions this worker admits through the family objects
    /// (0: none).
    pub sessions: u32,
    /// A session holds the whole GPU (no batch job while it is live).
    pub session_exclusive: bool,
    /// Upload outputs straight to R2 through part URLs the family object
    /// mints, overlapped with the encode (no R2 credentials needed here).
    pub direct_upload: bool,
    /// Part size of direct uploads, MiB (S3: ≥ 5 except the last).
    pub upload_part_mib: u32,
    /// Also an API front behind the edge Worker (docs/serve/edge-control-plane.md
    /// §2.4): `auth.mode` becomes `trust-edge`, and a submit is enqueued on
    /// the model's family object instead of the local engine.
    pub front: bool,
    /// This worker's own base URL (the edge forwards here; sessions'
    /// signalling goes here). Default: `server.public_base_url`.
    pub endpoint: Option<String>,
    /// Model id → the family whose object queues its jobs. A model not
    /// listed goes to the only family (with one), else is not fronted.
    pub model_families: BTreeMap<String, String>,
    /// Front: refuse a submit (429) when this many jobs of its model wait
    /// in its family object (0: no limit; the pool's `max_queued`).
    pub max_queued: u32,
    /// Front: re-dispatches after a worker loss before a job fails.
    pub retries: u32,
    /// Front: inputs up to this many bytes per job ride inside the
    /// envelope (as the gateway's `inline_inputs_max_bytes`).
    pub inline_inputs_max_bytes: u64,
    /// Front: large video/audio given as a public URL is fetched by the
    /// executing worker (as the gateway's `input_passthrough`).
    pub input_passthrough: bool,
}

impl Default for DispatchCfg {
    fn default() -> Self {
        Self {
            do_url: None,
            capacity: 2,
            status_s: 10,
            families: Vec::new(),
            sessions: 1,
            session_exclusive: true,
            direct_upload: false,
            upload_part_mib: 8,
            front: false,
            endpoint: None,
            model_families: BTreeMap::new(),
            max_queued: 0,
            retries: 1,
            inline_inputs_max_bytes: 8 << 20,
            input_passthrough: true,
        }
    }
}

impl DispatchCfg {
    /// The family whose object queues `model`'s jobs.
    pub fn family_of(&self, model: &str) -> Option<&str> {
        if let Some(f) = self.model_families.get(model) {
            return Some(f);
        }
        match self.families.as_slice() {
            [one] => Some(one),
            _ => None,
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
        Self {
            format: "text".into(),
            filter: "info".into(),
        }
    }
}

/// `[otel]`: OpenTelemetry export of the request traces (docs/serve/tracing.md
/// "OpenTelemetry export"). Off by default; the export code is compiled in by
/// the `otel` cargo feature (a build without it accepts the section, warns
/// when `enabled` and exports nothing).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OtelCfg {
    /// Export spans (`FV_OTEL=1`).
    pub enabled: bool,
    /// The collector's OTLP/HTTP traces URL, used as given (the SDK appends
    /// nothing): `http://127.0.0.1:4318/v1/traces` by default.
    pub endpoint: String,
    /// `http/protobuf` (the only protocol built; gRPC is not).
    pub protocol: String,
    /// `service.name` of the exported resource.
    pub service_name: String,
    /// Head sampling by trace id, 0..=1 (1: every traced request).
    pub sample_ratio: f64,
    /// Export request timeout.
    pub timeout_ms: u64,
}

impl OtelCfg {
    /// The one protocol the exporter speaks.
    pub const PROTOCOL: &'static str = "http/protobuf";

    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.protocol != Self::PROTOCOL {
            return Err(ConfigError::Invalid(format!(
                "[otel] protocol = `{}`: only `{}` is built",
                self.protocol,
                Self::PROTOCOL
            )));
        }
        if !(0.0..=1.0).contains(&self.sample_ratio) {
            return Err(ConfigError::Invalid(format!(
                "[otel] sample_ratio = {}: expected 0..=1",
                self.sample_ratio
            )));
        }
        if self.timeout_ms == 0 {
            return Err(ConfigError::Invalid(
                "[otel] timeout_ms must be 1 or more".into(),
            ));
        }
        if self.service_name.trim().is_empty() {
            return Err(ConfigError::Invalid(
                "[otel] service_name must not be empty".into(),
            ));
        }
        let u = url::Url::parse(&self.endpoint)
            .map_err(|e| ConfigError::Invalid(format!("[otel] endpoint: {e}")))?;
        if !matches!(u.scheme(), "http" | "https") || u.host_str().is_none() {
            return Err(ConfigError::Invalid(format!(
                "[otel] endpoint `{}`: an http(s) URL with a host",
                self.endpoint
            )));
        }
        Ok(())
    }
}

impl Default for OtelCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://127.0.0.1:4318/v1/traces".into(),
            protocol: Self::PROTOCOL.into(),
            service_name: "fv-serve".into(),
            sample_ratio: 1.0,
            timeout_ms: 10_000,
        }
    }
}

/// `[moderation]`: optional prompt admission checks (design §10 R8;
/// docs/ports/h3-turbo-loras.md "NCII guard"). Off by default.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModerationCfg {
    /// The NCII prompt guard (`hfmlsoc/ncii-light-guard-v01`, MIT, CPU):
    /// `off` (default), `keyframe` (requests with an image or video input,
    /// as the MiniMax-H3-Turbo-Lora Space runs it) or `all`.
    pub ncii: String,
    /// Its directory (`config.json`, `model.safetensors`, `tokenizer.json`).
    /// Default `$FV_WEIGHTS/auxiliary/ncii-light-guard`
    /// (`/workspace/weights/...`).
    pub ncii_dir: Option<PathBuf>,
    /// Refuse when the `ncii` probability reaches this (default 0.5: the
    /// pipeline's top label, what the Space refuses on).
    pub ncii_threshold: f32,
    /// Prompts classified at once (CPU threads are shared with the engine's
    /// host work; default 2).
    pub ncii_concurrency: usize,
}

impl Default for ModerationCfg {
    fn default() -> Self {
        Self {
            ncii: "off".into(),
            ncii_dir: None,
            ncii_threshold: 0.5,
            ncii_concurrency: 2,
        }
    }
}

impl ModerationCfg {
    pub fn scope(&self) -> Result<fastvideo_protocol::ModerationScope, ConfigError> {
        fastvideo_protocol::ModerationScope::parse(&self.ncii)
            .map_err(|e| ConfigError::Invalid(format!("[moderation] ncii: {e}")))
    }

    /// `ncii_dir`, else `$FV_WEIGHTS/auxiliary/ncii-light-guard`.
    pub fn ncii_dir(&self, env: &dyn Env) -> PathBuf {
        self.ncii_dir.clone().unwrap_or_else(|| {
            PathBuf::from(
                env.var("FV_WEIGHTS")
                    .unwrap_or_else(|| "/workspace/weights".into()),
            )
            .join(fastvideo_models::ncii::NCII_TREE)
        })
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.scope()?;
        if !(self.ncii_threshold > 0.0 && self.ncii_threshold <= 1.0) {
            return Err(ConfigError::Invalid(format!(
                "[moderation] ncii_threshold = {}: expected (0, 1]",
                self.ncii_threshold
            )));
        }
        if self.ncii_concurrency == 0 {
            return Err(ConfigError::Invalid(
                "[moderation] ncii_concurrency must be 1 or more".into(),
            ));
        }
        Ok(())
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
    pub protocols: ProtocolsCfg,
    pub ltx: LtxCfg,
    pub limits: LimitsCfg,
    pub webrtc: WebrtcCfg,
    pub streams: StreamsCfg,
    pub segments: SegmentsCfg,
    pub outputs: OutputsCfg,
    pub director: DirectorCfg,
    pub reactor: ReactorCfg,
    /// `None`: no prompt upsampling.
    pub upsampler: Option<UpsamplerCfg>,
    /// Prompt admission checks (the NCII guard); off by default.
    pub moderation: ModerationCfg,
    pub log: LogCfg,
    /// OpenTelemetry export of the request traces; off by default.
    pub otel: OtelCfg,
    /// Ed25519 seed for fal webhooks (normally `FV_WEBHOOK_ED25519_KEY`).
    pub webhook_key: Secret,
    /// `[cluster]` (read from `[gateway]` too, its name before plan F3b).
    #[serde(alias = "gateway")]
    pub cluster: ClusterCfg,
    /// A worker's socket to its pool's Durable Object (docs/serve/gateway-cloudflare.md).
    pub dispatch: DispatchCfg,
    /// The deprecated forms the file used, one note each (`[gateway]`):
    /// fv-serve logs them once tracing is up. Not part of the config.
    #[serde(skip)]
    pub deprecations: Vec<String>,
}

/// The `[cluster]` section's name before plan F3b.
const CLUSTER_OLD_NAME: &str = "gateway";

/// `[cluster]` and its deprecated name `[gateway]`: a file with both is
/// refused; one with the old name only gets a note (`Some`).
fn cluster_section(raw: &toml::Table) -> Result<Option<String>, String> {
    match (raw.contains_key("cluster"), raw.contains_key(CLUSTER_OLD_NAME)) {
        (true, true) => Err("[cluster] and [gateway] are both set: [gateway] is the deprecated name of [cluster] (plan F3b); keep only [cluster]".to_owned()),
        (false, true) => Ok(Some("[gateway] is read as [cluster], its new name (plan F3b): the launchers keep sending [gateway] until every image reads [cluster] (docs/adr/0011-deploy-section.md, \"Renaming a worker-config section\"); a config only for this image or newer can say [cluster]".to_owned())),
        _ => Ok(None),
    }
}

/// The config forms plan A3 retired, each with what replaces it: the
/// `[aliases]` table (the variants' `public_names`) and a `[[models]]` entry
/// without `variant` or with `family` / `recipe`.
fn retired_forms(raw: &toml::Table) -> Result<(), String> {
    if raw.contains_key("aliases") {
        return Err("[aliases] is no longer a table: the public API names (MiniMax-H3-Turbo, ...) are each \
             variant's `public_names` in families/*.toml, and the tier aliases are the engine's"
            .to_owned());
    }
    let Some(toml::Value::Array(models)) = raw.get("models") else {
        return Ok(());
    };
    for (i, m) in models.iter().enumerate() {
        let Some(m) = m.as_table() else { continue };
        let label = m
            .get("id")
            .or_else(|| m.get("variant"))
            .and_then(toml::Value::as_str)
            .map_or_else(|| format!("#{i}"), |s| format!("`{s}`"));
        for (k, why) in ModelCfg::RETIRED_KEYS {
            if m.contains_key(k) {
                return Err(format!(
                    "[[models]] {label}: `{k}` is no longer a key: {why}"
                ));
            }
        }
        if !m.contains_key("variant") {
            return Err(format!(
                "[[models]] {label}: `variant` is required (a variant id, a served name or a tier alias of families/*.toml)"
            ));
        }
    }
    Ok(())
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
    env.var(name)
        .or_else(|| env.var(&name.to_ascii_lowercase()))
}

fn parse_enum<T: for<'de> Deserialize<'de>>(name: &str, v: &str) -> Result<T, ConfigError> {
    T::deserialize(serde::de::value::StrDeserializer::<serde::de::value::Error>::new(v.trim()))
        .map_err(|e| ConfigError::Invalid(format!("{name}={v}: {e}")))
}

impl Config {
    /// Parses TOML text.
    pub fn from_toml(text: &str, origin: &str) -> Result<Self, ConfigError> {
        // The pre-A3 forms get a message naming their replacement before
        // serde's generic "unknown field" / "missing field" could.
        let mut deprecations = Vec::new();
        if let Ok(raw) = text.parse::<toml::Table>() {
            retired_forms(&raw).map_err(|m| ConfigError::Invalid(format!("{origin}: {m}")))?;
            let note = cluster_section(&raw)
                .map_err(|m| ConfigError::Invalid(format!("{origin}: {m}")))?;
            deprecations.extend(note.map(|n| format!("{origin}: {n}")));
        }
        let mut c: Self =
            toml::from_str(text).map_err(|e| ConfigError::Parse(origin.to_owned(), e))?;
        c.deprecations = deprecations;
        Ok(c)
    }

    /// Defaults < file < environment, then validation.
    pub fn load(path: Option<&Path>, env: &dyn Env) -> Result<Self, ConfigError> {
        let mut c = match path {
            Some(p) => {
                let text =
                    std::fs::read_to_string(p).map_err(|e| ConfigError::Read(p.to_owned(), e))?;
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
            let port: u16 = p
                .trim()
                .parse()
                .map_err(|_| ConfigError::Invalid(format!("PORT={p} is not a port")))?;
            let host = self
                .server
                .bind
                .rsplit_once(':')
                .map_or("0.0.0.0", |(h, _)| h)
                .to_owned();
            self.server.bind = format!("{host}:{port}");
        }
        if let Some(v) = env.var("FV_PUBLIC_BASE_URL") {
            self.server.public_base_url = Some(v);
        }
        if let Some(v) = env.var("FV_STATE_DIR") {
            self.server.state_dir = v.into();
        }
        if let Some(v) = env.var("FV_SERVE_FORWARD") {
            self.server.forward = matches!(v.trim(), "1" | "true" | "yes");
        }
        if let Some(v) = env.var("FV_CALLBACKS_ALLOW_PRIVATE") {
            self.server.callbacks_allow_private = matches!(v.trim(), "1" | "true" | "yes");
        }
        if let Some(v) = env.var("FV_WORKERS_MAX") {
            self.server.workers_max =
                v.trim().parse().ok().filter(|n| *n >= 1).ok_or_else(|| {
                    ConfigError::Invalid(format!(
                        "FV_WORKERS_MAX={v} is not a worker count (1 or more)"
                    ))
                })?;
        }
        if let Some(v) = env.var("FV_CORS_ORIGINS") {
            self.server.cors_origins = match v.trim() {
                "" | "none" | "off" => Vec::new(),
                list => list
                    .split(',')
                    .map(|o| o.trim().to_owned())
                    .filter(|o| !o.is_empty())
                    .collect(),
            };
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
        if let Some(v) = env.var("FV_ADMIN_TOKEN") {
            self.auth.admin_token = Secret(v.trim().to_owned());
        }
        if let Some(v) = env.var("FV_ADMIN_TOKEN_RECIPIENT") {
            self.auth.admin_token_recipient = Some(v.trim().to_owned()).filter(|v| !v.is_empty());
        }
        if let Some(v) = env.var("FV_KEY_STORE") {
            self.auth.key_store = parse_enum("FV_KEY_STORE", &v)?;
        }
        if let Some(v) = env.var("FV_CONSOLE") {
            self.server.console = !matches!(v.trim(), "0" | "false" | "off" | "no");
        }
        if let Some(v) = env.var("FV_ECHO_MODEL") {
            self.engine.echo_model = matches!(v.trim(), "1" | "true" | "on" | "yes");
        }
        if let Some(v) = env.var("FV_MP4_FRAGMENTED") {
            self.engine.mp4_fragmented = matches!(v.trim(), "1" | "true" | "on" | "yes");
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
        if let Some(v) = env.var("FV_STREAM_ENCODER") {
            self.webrtc.encoder = v;
        }
        // RT's own environment names (reactor §3.3).
        for (k, slot) in [
            ("ORPHAN_TIMEOUT_SECONDS", &mut self.reactor.orphan_timeout_s),
            (
                "WEBRTC_CLIENT_PING_TIMEOUT_SECONDS",
                &mut self.reactor.ping_timeout_s,
            ),
        ] {
            if let Some(v) = env.var(k) {
                *slot = v
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .map(|s| s.ceil() as u64)
                    .ok_or_else(|| {
                        ConfigError::Invalid(format!("{k}={v} is not a positive number of seconds"))
                    })?;
            }
        }
        for (k, slot) in [
            (
                "FV_CAUSAL_DEFAULT_MAX_S",
                &mut self.streams.causal_default_max_s,
            ),
            ("FV_CAUSAL_HARD_MAX_S", &mut self.streams.causal_hard_max_s),
        ] {
            if let Some(v) = env.var(k) {
                *slot = v.trim().parse().ok().filter(|n| *n >= 1).ok_or_else(|| {
                    ConfigError::Invalid(format!("{k}={v} is not a number of seconds (1 or more)"))
                })?;
            }
        }
        if let Some(v) = env.var("FV_SEGMENTS_MODEL") {
            self.segments.model = Some(v.trim().to_owned()).filter(|m| !m.is_empty());
        }
        if let Some(v) = env.var("FV_SEGMENTS_ENCODER") {
            self.segments.encoder = v.trim().to_owned();
        }
        if let Some(v) = env.var("FV_SEGMENTS_IDLE_TTL_S") {
            self.segments.idle_ttl_s =
                v.trim().parse().ok().filter(|n| *n >= 1).ok_or_else(|| {
                    ConfigError::Invalid(format!(
                        "FV_SEGMENTS_IDLE_TTL_S={v} is not a number of seconds (1 or more)"
                    ))
                })?;
        }
        if let Some(v) = env.var("FV_REACTOR_MODEL") {
            self.reactor.model = Some(v);
        }
        if let Some(v) = env.var("FV_REACTOR_MODE") {
            self.reactor.mode = v;
        }
        if let Some(v) = env.var("FV_ENGINE") {
            self.engine.backend = parse_enum("FV_ENGINE", &v)?;
        }
        if let Some(v) = env.var("FV_FAKE_H3_1080P") {
            self.engine.fake.h3_1080p = matches!(v.trim(), "1" | "true" | "on" | "yes");
        }
        if let Some(v) = env.var("FV_JOB_DEADLINE_S") {
            self.engine.job_deadline_s = v.trim().parse().map_err(|_| {
                ConfigError::Invalid(format!(
                    "FV_JOB_DEADLINE_S={v} is not a number of seconds (0 = none)"
                ))
            })?;
        }
        if let Some(v) = env.var("FV_FAKE_DEVICE") {
            let v = v.trim();
            self.engine.fake.device = (!v.is_empty()).then(|| v.to_owned());
        }
        if let Some(v) = env.var("FV_JOB_STORE") {
            self.jobs.backend = parse_enum("FV_JOB_STORE", &v)?;
        }
        if let Some(v) = var2(env, "FV_CF_ACCOUNT_ID") {
            self.outputs.cloudflare.account_id = v.trim().to_owned();
            self.jobs.d1.account_id = Some(v);
        }
        if let Some(v) = var2(env, "FV_CF_STREAM_API_TOKEN") {
            self.outputs.cloudflare.api_token = Secret(v.trim().to_owned());
        }
        if let Some(v) = env.var("FV_CF_STREAM_API_BASE") {
            self.outputs.cloudflare.api_base = v.trim().to_owned();
        }
        if let Some(v) = env.var("FV_CF_STREAM_NAME_PREFIX") {
            self.outputs.cloudflare.name_prefix = v.trim().to_owned();
        }
        if let Some(v) = env.var("FV_CF_STREAM_CUSTOMER_DOMAIN") {
            self.outputs.cloudflare.customer_domain = Some(v.trim().to_owned());
        }
        if let Some(v) = env.var("FV_OUTPUTS_ALLOW_PRIVATE") {
            self.outputs.allow_private = matches!(v.trim(), "1" | "true" | "yes" | "on");
        }
        if let Some(v) = env.var("FV_OUTPUTS_SRT_LATENCY_MS") {
            self.outputs.srt_latency_ms = v
                .trim()
                .parse()
                .ok()
                .filter(|n| (20..=8000).contains(n))
                .ok_or_else(|| {
                    ConfigError::Invalid(format!("FV_OUTPUTS_SRT_LATENCY_MS={v} is not 20..8000"))
                })?;
        }
        if let Some(v) = env.var("FV_OUTPUTS_RTMPS_CA_FILE") {
            self.outputs.rtmps_ca_file =
                Some(PathBuf::from(v.trim())).filter(|p| !p.as_os_str().is_empty());
        }
        if let Some(v) = env.var("FV_DIRECTOR_ALLOW_OUTPUT") {
            self.director.allow_output = matches!(v.trim(), "1" | "true" | "yes" | "on");
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
        if let Some(v) = env.var("FV_JOBS_HEARTBEAT_S") {
            self.jobs.heartbeat_s = v.trim().parse().ok().filter(|n| *n >= 1).ok_or_else(|| {
                ConfigError::Invalid(format!(
                    "FV_JOBS_HEARTBEAT_S={v} is not a number of seconds"
                ))
            })?;
        }
        if let Some(v) = env.var("FV_ARTIFACTS_DIR") {
            self.artifacts.local_dir = Some(v.into());
        }
        if let Some(v) = env.var("FV_SERVE_ROLE") {
            self.server.role = parse_enum("FV_SERVE_ROLE", &v)?;
        }
        if let Some(v) = env.var("FV_INTERNAL_TOKEN") {
            self.cluster.internal_token = Secret(v.trim().to_owned());
        }
        if let Some(v) = env.var("FV_GATEWAY_POOL") {
            self.cluster.pool = Some(v);
        }
        if let Some(v) = env.var("FV_WORKER_DIRECT") {
            self.cluster.direct = matches!(v.trim(), "1" | "true" | "on" | "yes");
        }
        if let Some(v) = env.var("FV_DISPATCH_DO_URL") {
            self.dispatch.do_url = Some(v.trim().to_owned()).filter(|s| !s.is_empty());
        }
        if let Some(v) = env
            .var("FV_DISPATCH_CAPACITY")
            .and_then(|v| v.trim().parse().ok())
        {
            self.dispatch.capacity = v;
        }
        if let Some(v) = env.var("FV_DISPATCH_FAMILIES") {
            self.dispatch.families = v
                .split([',', ' '])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
        }
        if let Some(v) = env
            .var("FV_DISPATCH_SESSIONS")
            .and_then(|v| v.trim().parse().ok())
        {
            self.dispatch.sessions = v;
        }
        if let Some(v) = env.var("FV_DISPATCH_DIRECT_UPLOAD") {
            self.dispatch.direct_upload = matches!(v.trim(), "1" | "true" | "on" | "yes");
        }
        if let Some(v) = env
            .var("FV_DISPATCH_UPLOAD_PART_MIB")
            .and_then(|v| v.trim().parse().ok())
        {
            self.dispatch.upload_part_mib = v;
        }
        if let Some(v) = env.var("FV_DISPATCH_FRONT") {
            self.dispatch.front = matches!(v.trim(), "1" | "true" | "on" | "yes");
        }
        if let Some(v) = env.var("FV_DISPATCH_ENDPOINT") {
            self.dispatch.endpoint = Some(v.trim().to_owned()).filter(|s| !s.is_empty());
        }
        if let Some(v) = env.var("FV_DISPATCH_MODEL_FAMILIES") {
            // `model=family,model=family`
            self.dispatch.model_families = v
                .split([',', ' '])
                .filter_map(|kv| kv.trim().split_once('='))
                .map(|(m, f)| (m.trim().to_owned(), f.trim().to_owned()))
                .filter(|(m, f)| !m.is_empty() && !f.is_empty())
                .collect();
        }
        if let Some(v) = env
            .var("FV_DISPATCH_MAX_QUEUED")
            .and_then(|v| v.trim().parse().ok())
        {
            self.dispatch.max_queued = v;
        }
        if let Some(v) = env
            .var("FV_DISPATCH_STATUS_S")
            .and_then(|v| v.trim().parse().ok())
        {
            self.dispatch.status_s = v;
        }
        self.apply_upsampler_env(env)?;
        if let Some(v) = env.var("FV_MODERATION_NCII") {
            self.moderation.ncii = v.trim().to_owned();
        }
        if let Some(v) = env
            .var("FV_MODERATION_NCII_DIR")
            .filter(|v| !v.trim().is_empty())
        {
            self.moderation.ncii_dir = Some(PathBuf::from(v.trim()));
        }
        if let Some(v) = env.var("FV_MODERATION_NCII_THRESHOLD") {
            self.moderation.ncii_threshold = v.trim().parse().map_err(|_| {
                ConfigError::Invalid(format!("FV_MODERATION_NCII_THRESHOLD={v} is not a number"))
            })?;
        }
        if let Some(v) = env.var("FV_LOG_FORMAT") {
            self.log.format = v;
        }
        if let Some(v) = env.var("RUST_LOG") {
            self.log.filter = v;
        }
        self.apply_otel_env(env)?;
        Ok(())
    }

    /// `FV_OTEL*` → `[otel]`.
    fn apply_otel_env(&mut self, env: &dyn Env) -> Result<(), ConfigError> {
        if let Some(v) = env.var("FV_OTEL") {
            self.otel.enabled = matches!(v.trim(), "1" | "true" | "on" | "yes");
        }
        if let Some(v) = env.var("FV_OTEL_ENDPOINT") {
            self.otel.endpoint = v.trim().to_owned();
        }
        if let Some(v) = env.var("FV_OTEL_PROTOCOL") {
            self.otel.protocol = v.trim().to_owned();
        }
        if let Some(v) = env.var("FV_OTEL_SERVICE_NAME") {
            self.otel.service_name = v.trim().to_owned();
        }
        if let Some(v) = env.var("FV_OTEL_SAMPLE_RATIO") {
            self.otel.sample_ratio = v.trim().parse().map_err(|_| {
                ConfigError::Invalid(format!("FV_OTEL_SAMPLE_RATIO={v} is not a number"))
            })?;
        }
        if let Some(v) = env.var("FV_OTEL_TIMEOUT_MS") {
            self.otel.timeout_ms = v.trim().parse().map_err(|_| {
                ConfigError::Invalid(format!(
                    "FV_OTEL_TIMEOUT_MS={v} is not a number of milliseconds"
                ))
            })?;
        }
        Ok(())
    }

    /// `FV_UPSAMPLER_*` (a base URL alone creates the section), then the
    /// key from `api_key_env`.
    fn apply_upsampler_env(&mut self, env: &dyn Env) -> Result<(), ConfigError> {
        if let Some(v) = env.var("FV_UPSAMPLER_BASE_URL") {
            self.upsampler
                .get_or_insert_with(UpsamplerCfg::default)
                .base_url = v.trim().to_owned();
        }
        let Some(u) = self.upsampler.as_mut() else {
            return Ok(());
        };
        if let Some(v) = env.var("FV_UPSAMPLER_MODEL") {
            u.model = v.trim().to_owned();
        }
        if let Some(v) = env.var("FV_UPSAMPLER_DEFAULT_ON") {
            u.default_on = matches!(v.trim(), "1" | "true" | "on" | "yes");
        }
        if let Some(v) = env.var("FV_UPSAMPLER_HEDGE") {
            u.hedge = Some(matches!(v.trim(), "1" | "true" | "on" | "yes"));
        }
        if let Some(v) = env.var("FV_UPSAMPLER_LOCAL") {
            u.local = Some(matches!(v.trim(), "1" | "true" | "on" | "yes"));
        }
        if let Some(v) = env.var("FV_UPSAMPLER_VISION") {
            u.vision = matches!(v.trim(), "1" | "true" | "on" | "yes");
        }
        if let Some(v) = env.var("FV_UPSAMPLER_CONTEXT_TOKENS") {
            u.context_tokens = v.trim().parse().map_err(|_| {
                ConfigError::Invalid(format!("FV_UPSAMPLER_CONTEXT_TOKENS={v} is not a number"))
            })?;
        }
        if let Some(v) = env.var("FV_UPSAMPLER_DECK_SIZE") {
            u.deck_size = Some(v.trim().parse().map_err(|_| {
                ConfigError::Invalid(format!("FV_UPSAMPLER_DECK_SIZE={v} is not a number"))
            })?);
        }
        if let Some(v) = env.var("FV_UPSAMPLER_TIMEOUT_MS") {
            u.timeout_ms = Some(v.trim().parse().map_err(|_| {
                ConfigError::Invalid(format!(
                    "FV_UPSAMPLER_TIMEOUT_MS={v} is not a number of milliseconds"
                ))
            })?);
        }
        if let Some(v) = env.var("FV_UPSAMPLER_API_KEY_ENV") {
            u.api_key_env = v.trim().to_owned();
        }
        let name = u.api_key_env.trim().to_owned();
        if !name.is_empty() {
            if let Some(k) = var2(env, &name) {
                u.api_key = Secret(k.trim().to_owned());
            }
        }
        Ok(())
    }

    /// Consistency checks that do not need I/O.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.bind_addr()?;
        for m in &self.models {
            m.validate()?;
        }
        if let Some(u) = &self.upsampler {
            u.validate()?;
        }
        self.moderation.validate()?;
        self.otel.validate()?;
        if self.server.workers_max == 0 {
            return Err(ConfigError::Invalid(
                "server.workers_max must be 1 or more".into(),
            ));
        }
        self.streams
            .causal_limits()
            .check()
            .map_err(|e| ConfigError::Invalid(format!("[streams]: {e}")))?;
        self.segments.validate()?;
        if let Some(u) = &self.server.public_base_url {
            url::Url::parse(u)
                .map_err(|e| ConfigError::Invalid(format!("server.public_base_url: {e}")))?;
        }
        for o in &self.server.cors_origins {
            if o == "*" {
                if self.server.cors_origins.len() > 1 {
                    return Err(ConfigError::Invalid(
                        "server.cors_origins: `*` cannot be combined with origins".into(),
                    ));
                }
                continue;
            }
            let ok = url::Url::parse(o)
                .map(|u| {
                    matches!(u.scheme(), "http" | "https")
                        && u.host().is_some()
                        && u.origin().ascii_serialization() == *o
                })
                .unwrap_or(false);
            if !ok {
                return Err(ConfigError::Invalid(format!(
                    "server.cors_origins: `{o}` is not an origin (scheme://host[:port], no path or trailing slash)"
                )));
            }
        }
        if self.auth.mode == AuthMode::Keys && !self.auth.keys.is_empty() {
            fastvideo_serve_kit::KeyRing::from_hash_list(self.auth.keys.expose())
                .map_err(ConfigError::Invalid)?;
        }
        if self.auth.key_store == KeyStoreBackend::D1 && !self.d1_configured() {
            return Err(ConfigError::Invalid("auth.key_store = d1 needs the D1 settings (FV_CF_ACCOUNT_ID, FV_CF_API_TOKEN, FV_D1_DATABASE_ID)".into()));
        }
        if self.jobs.backend == JobBackend::D1 && !self.d1_configured() {
            return Err(ConfigError::Invalid(
                "jobs.backend = d1 needs account id, API token and database id (FV_CF_ACCOUNT_ID, FV_CF_API_TOKEN, FV_D1_DATABASE_ID)".into(),
            ));
        }
        if self.artifacts.backend == ArtifactBackend::S3 && !self.s3_configured() {
            return Err(ConfigError::Invalid(
                "artifacts.backend = s3 needs endpoint, bucket and both keys (FV_R2_* or FV_S3_*)"
                    .into(),
            ));
        }
        for p in self.jobs.retention_s.keys() {
            if !fastvideo_protocol::ProtocolId::ALL
                .iter()
                .any(|x| x.as_str() == p)
            {
                return Err(ConfigError::Invalid(format!(
                    "jobs.retention_s: unknown API `{p}`"
                )));
            }
        }
        match self.engine.post_encoder.as_str() {
            "auto" | "nvenc" | "cpu-test-x264" => {}
            other => {
                return Err(ConfigError::Invalid(format!(
                    "engine.post_encoder: unknown `{other}`"
                )))
            }
        }
        if self.engine.job_deadline_s > EngineCfg::JOB_DEADLINE_MAX_S {
            return Err(ConfigError::Invalid(format!(
                "engine.job_deadline_s: {} is more than a day ({}); seconds, 0 = none",
                self.engine.job_deadline_s,
                EngineCfg::JOB_DEADLINE_MAX_S
            )));
        }
        match self.director.encoder.as_str() {
            "auto" | "nvenc" | "openh264" | "cpu-test-x264" => {}
            other => {
                return Err(ConfigError::Invalid(format!(
                    "director.encoder: unknown `{other}` (auto | nvenc | openh264)"
                )))
            }
        }
        match self.reactor.mode.as_str() {
            "auto" | "avatar" => {}
            other => {
                return Err(ConfigError::Invalid(format!(
                    "reactor.mode: unknown `{other}` (auto | avatar)"
                )))
            }
        }
        if parse_size(&self.reactor.avatar_size).is_none() {
            return Err(ConfigError::Invalid(format!(
                "reactor.avatar_size: `{}` is not WxH",
                self.reactor.avatar_size
            )));
        }
        if !(1.0..=20.0).contains(&self.reactor.avatar_window_s) {
            return Err(ConfigError::Invalid(
                "reactor.avatar_window_s: 1..=20 seconds".into(),
            ));
        }
        if self.reactor.avatar_session_max_s == 0 {
            return Err(ConfigError::Invalid(
                "reactor.avatar_session_max_s: at least 1".into(),
            ));
        }
        match self.reactor.h264.as_str() {
            "auto" | "nvenc" | "openh264" | "off" => {}
            other => {
                return Err(ConfigError::Invalid(format!(
                    "reactor.h264: unknown `{other}` (auto | nvenc | openh264 | off)"
                )))
            }
        }
        match self.webrtc.encoder.as_str() {
            "auto" | "nvenc" | "openh264" | "x264-test" => {}
            other => {
                return Err(ConfigError::Invalid(format!(
                    "webrtc.encoder: unknown `{other}` (auto | nvenc | openh264)"
                )))
            }
        }
        if !(self.director.chunk_seconds.is_finite() && self.director.chunk_seconds > 0.0) {
            return Err(ConfigError::Invalid(
                "director.chunk_seconds must be positive".into(),
            ));
        }
        self.validate_cluster()
    }

    /// `[cluster]`, `[dispatch]` and `server.role` checks.
    fn validate_cluster(&self) -> Result<(), ConfigError> {
        let bad = |m: String| Err(ConfigError::Invalid(m));
        if self.cluster.direct {
            if self.server.role != Role::Worker {
                return bad("cluster.direct (FV_WORKER_DIRECT) is for server.role = worker".into());
            }
            if matches!(self.auth.mode, AuthMode::TrustGateway | AuthMode::TrustEdge) {
                return bad("cluster.direct: nothing in front authenticates the clients, so auth.mode must be keys or none (not trust-gateway or trust-edge)".into());
            }
            if self.auth.admin_token.is_empty() {
                return bad("cluster.direct needs auth.admin_token (FV_ADMIN_TOKEN): every worker of the cluster shares it".into());
            }
        }
        if self.server.role == Role::Worker {
            if self.cluster.internal_token.is_empty() {
                return bad(
                    "server.role = worker needs cluster.internal_token (FV_INTERNAL_TOKEN)".into(),
                );
            }
            if let Some(u) = &self.dispatch.do_url {
                url::Url::parse(u)
                    .map_err(|e| ConfigError::Invalid(format!("dispatch.do_url `{u}`: {e}")))?;
                if self.dispatch.families.is_empty()
                    && self.cluster.pool.as_deref().unwrap_or("").is_empty()
                {
                    return bad("dispatch.do_url needs cluster.pool (FV_GATEWAY_POOL, the pool whose Durable Object this worker connects to) or dispatch.families (FV_DISPATCH_FAMILIES)".into());
                }
            }
            for f in &self.dispatch.families {
                if !fastvideo_dispatch_proto::valid_id(f) {
                    return bad(format!(
                        "dispatch.families: `{f}` is not a valid family id ([A-Za-z0-9._-])"
                    ));
                }
            }
            if !self.dispatch.families.is_empty() && self.dispatch.do_url.is_none() {
                return bad("dispatch.families needs dispatch.do_url (FV_DISPATCH_DO_URL)".into());
            }
            if self.dispatch.front {
                if self.dispatch.families.is_empty() || self.dispatch.do_url.is_none() {
                    return bad("dispatch.front (FV_DISPATCH_FRONT) needs dispatch.do_url and dispatch.families: a front enqueues on the family objects".into());
                }
                if self.cluster.direct {
                    return bad("dispatch.front and cluster.direct exclude each other: behind the edge, the edge authenticates clients".into());
                }
                for (m, f) in &self.dispatch.model_families {
                    if !self.dispatch.families.contains(f) {
                        return bad(format!("dispatch.model_families: `{m}` maps to `{f}`, which is not in dispatch.families"));
                    }
                }
                if let Some(e) = &self.dispatch.endpoint {
                    url::Url::parse(e).map_err(|err| {
                        ConfigError::Invalid(format!("dispatch.endpoint `{e}`: {err}"))
                    })?;
                }
            }
            if self.dispatch.direct_upload && self.dispatch.families.is_empty() {
                return bad("dispatch.direct_upload needs dispatch.families: upload URLs come from the family Durable Objects".into());
            }
            if self.dispatch.direct_upload && !(5..=1024).contains(&self.dispatch.upload_part_mib) {
                return bad(
                    "dispatch.upload_part_mib must be 5..=1024 (S3 multipart parts)".into(),
                );
            }
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
        s.endpoint.is_some()
            && s.bucket.is_some()
            && !s.access_key_id.is_empty()
            && !s.secret_access_key.is_empty()
    }

    /// The job store `auto` resolves to.
    pub fn job_backend(&self) -> JobBackend {
        match self.jobs.backend {
            JobBackend::Auto if self.d1_configured() => JobBackend::D1,
            JobBackend::Auto => JobBackend::File,
            b => b,
        }
    }

    /// The minted-key store `auto` resolves to (D1 only in builds with
    /// `http-client`).
    pub fn key_store_backend(&self) -> KeyStoreBackend {
        match self.auth.key_store {
            KeyStoreBackend::Auto if self.d1_configured() && cfg!(feature = "http-client") => {
                KeyStoreBackend::D1
            }
            KeyStoreBackend::Auto => KeyStoreBackend::File,
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

/// `WxH` (even sides, 64..=4096).
pub fn parse_size(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.trim().split_once(['x', 'X'])?;
    let (w, h): (u32, u32) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    let ok = |v: u32| (64..=4096).contains(&v) && v.is_multiple_of(2);
    (ok(w) && ok(h)).then_some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn otel_default_off_env_file_and_validation() {
        let c = Config::default();
        assert!(!c.otel.enabled, "off by default");
        assert_eq!(c.otel.endpoint, "http://127.0.0.1:4318/v1/traces");
        assert_eq!(c.otel.protocol, OtelCfg::PROTOCOL);
        assert_eq!(c.otel.service_name, "fv-serve");
        assert_eq!(c.otel.sample_ratio, 1.0);
        assert_eq!(c.otel.timeout(), std::time::Duration::from_secs(10));
        c.validate().unwrap();
        // An absent section parses to the defaults; the section round-trips.
        let t: Config = toml::from_str("[server]\nbind = \"127.0.0.1:1\"\n").unwrap();
        assert_eq!(t.otel, OtelCfg::default());
        let text = toml::to_string(&c).unwrap();
        assert!(text.contains("[otel]"), "{text}");
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back.otel, c.otel);
        let t: Config = toml::from_str(
            "[otel]\nenabled = true\nendpoint = \"https://otel.example/v1/traces\"\nsample_ratio = 0.5\n",
        )
        .unwrap();
        assert!(t.otel.enabled);
        assert_eq!(t.otel.endpoint, "https://otel.example/v1/traces");
        assert_eq!(t.otel.sample_ratio, 0.5);
        t.validate().unwrap();
        // Environment over the file.
        let mut c = t;
        c.apply_env(&env(&[
            ("FV_OTEL", "0"),
            ("FV_OTEL_ENDPOINT", "http://collector:4318/v1/traces"),
            ("FV_OTEL_SERVICE_NAME", "fv-serve-eu"),
            ("FV_OTEL_SAMPLE_RATIO", "0.25"),
            ("FV_OTEL_TIMEOUT_MS", "2500"),
        ]))
        .unwrap();
        assert!(!c.otel.enabled);
        assert_eq!(c.otel.endpoint, "http://collector:4318/v1/traces");
        assert_eq!(c.otel.service_name, "fv-serve-eu");
        assert_eq!(c.otel.sample_ratio, 0.25);
        assert_eq!(c.otel.timeout_ms, 2500);
        c.validate().unwrap();
        c.apply_env(&env(&[("FV_OTEL", "1")])).unwrap();
        assert!(c.otel.enabled);
        assert!(c
            .apply_env(&env(&[("FV_OTEL_SAMPLE_RATIO", "half")]))
            .is_err());
        assert!(c.apply_env(&env(&[("FV_OTEL_TIMEOUT_MS", "x")])).is_err());
        // Validation: the one protocol, the ratio range, the URL.
        for (text, needle) in [
            ("[otel]\nprotocol = \"grpc\"\n", "protocol"),
            ("[otel]\nsample_ratio = 1.5\n", "sample_ratio"),
            ("[otel]\nsample_ratio = -0.1\n", "sample_ratio"),
            ("[otel]\ntimeout_ms = 0\n", "timeout_ms"),
            ("[otel]\nservice_name = \" \"\n", "service_name"),
            ("[otel]\nendpoint = \"collector:4318\"\n", "endpoint"),
            ("[otel]\nendpoint = \"ftp://x/v1/traces\"\n", "endpoint"),
        ] {
            let t: Config = toml::from_str(text).unwrap();
            let err = t.validate().unwrap_err().to_string();
            assert!(err.contains(needle), "{text}: {err}");
        }
        // Unknown keys are refused like in every section.
        assert!(toml::from_str::<Config>("[otel]\nheaders = {}\n").is_err());
        // The printed config shows the section (nothing in it is secret).
        let v = Config::default().redacted();
        assert_eq!(v["otel"]["enabled"], serde_json::json!(false));
    }

    #[test]
    fn job_deadline_default_env_and_validation() {
        let mut c = Config::default();
        assert_eq!(c.engine.job_deadline_s, 0, "off by default");
        assert_eq!(c.engine.job_deadline(), None);
        c.validate().unwrap();
        c.apply_env(&env(&[("FV_JOB_DEADLINE_S", "90")])).unwrap();
        assert_eq!(
            c.engine.job_deadline(),
            Some(std::time::Duration::from_secs(90))
        );
        c.validate().unwrap();
        c.apply_env(&env(&[("FV_JOB_DEADLINE_S", "0")])).unwrap();
        assert_eq!(c.engine.job_deadline(), None);
        assert!(c
            .apply_env(&env(&[("FV_JOB_DEADLINE_S", "ninety")]))
            .is_err());
        assert!(c.apply_env(&env(&[("FV_JOB_DEADLINE_S", "-1")])).is_err());
        let t: Config = toml::from_str("[engine]\njob_deadline_s = 600\n").unwrap();
        assert_eq!(
            t.engine.job_deadline(),
            Some(std::time::Duration::from_secs(600))
        );
        t.validate().unwrap();
        let t: Config = toml::from_str("[engine]\njob_deadline_s = 600000\n").unwrap();
        let err = t.validate().unwrap_err().to_string();
        assert!(err.contains("job_deadline_s"), "{err}");
    }

    #[test]
    fn cors_origins_default_env_and_validation() {
        let mut c = Config::default();
        assert_eq!(c.server.cors_origins, vec!["*"]);
        c.apply_env(&env(&[(
            "FV_CORS_ORIGINS",
            "https://a.example, http://127.0.0.1:3000",
        )]))
        .unwrap();
        assert_eq!(
            c.server.cors_origins,
            vec!["https://a.example", "http://127.0.0.1:3000"]
        );
        c.validate().unwrap();
        c.apply_env(&env(&[("FV_CORS_ORIGINS", "none")])).unwrap();
        assert!(c.server.cors_origins.is_empty());
        c.validate().unwrap();
        for bad in [
            "https://a.example/",
            "a.example",
            "https://a.example/path",
            "*,https://a.example",
        ] {
            c.apply_env(&env(&[("FV_CORS_ORIGINS", bad)])).unwrap();
            assert!(c.validate().is_err(), "{bad}");
        }
    }

    #[test]
    fn causal_stream_limits_default_file_env_and_validation() {
        let mut c = Config::default();
        assert_eq!(
            c.streams.causal_limits(),
            fastvideo_protocol::CausalLimits {
                default_max_s: 120,
                hard_max_s: 300
            }
        );
        let mut c2 = Config::from_toml(
            "[streams]\ncausal_default_max_s = 60\ncausal_hard_max_s = 90\n",
            "t",
        )
        .unwrap();
        c2.validate().unwrap();
        assert_eq!(
            (
                c2.streams.causal_default_max_s,
                c2.streams.causal_hard_max_s
            ),
            (60, 90)
        );
        c2.apply_env(&env(&[
            ("FV_CAUSAL_DEFAULT_MAX_S", "30"),
            ("FV_CAUSAL_HARD_MAX_S", "45"),
        ]))
        .unwrap();
        assert_eq!(
            c2.streams.causal_limits(),
            fastvideo_protocol::CausalLimits {
                default_max_s: 30,
                hard_max_s: 45
            }
        );
        assert!(c.apply_env(&env(&[("FV_CAUSAL_HARD_MAX_S", "0")])).is_err());
        c.apply_env(&env(&[("FV_CAUSAL_DEFAULT_MAX_S", "400")]))
            .unwrap();
        assert!(c.validate().is_err(), "default above the hard ceiling");
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
variant = "h3-turbo"
id = "fasth3"
weights = "${FV_WEIGHTS}/FastH3"
resident = true
continuity = "anchor-last-frame"
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
        c.apply_env(&env(&[
            ("FV_WEIGHTS", "/runpod-volume/weights"),
            ("PORT", "8080"),
        ]))
        .unwrap();
        c.validate().unwrap();
        assert_eq!(
            c.models[0].weights.as_deref(),
            Some("/runpod-volume/weights/FastH3")
        );
        assert_eq!(
            c.models[0].extra["continuity"].as_str(),
            Some("anchor-last-frame")
        );
        assert_eq!(c.server.bind, "0.0.0.0:8080");
        assert_eq!(
            (c.models[0].variant.as_str(), c.models[0].label()),
            ("h3-turbo", "fasth3")
        );
        assert!(!c.protocols.ltx && c.protocols.minimax);
        assert_eq!(c.job_backend(), JobBackend::File);
        assert_eq!(c.artifact_backend(), ArtifactBackend::Local);
    }

    #[test]
    fn unknown_keys_are_refused() {
        let e = Config::from_toml("[server]\nbindd = \"x\"\n", "t").unwrap_err();
        assert!(e.to_string().contains("bindd"), "{e}");
        // The pre-A3 model keys and the `[aliases]` table name their replacement.
        let e = Config::from_toml("[aliases]\n\"MiniMax-H3\" = \"fasth3\"\n", "t").unwrap_err();
        assert!(e.to_string().contains("public_names"), "{e}");
        for old in ["family = \"h3\"", "recipe = \"h3-turbo\""] {
            let e = Config::from_toml(&format!("[[models]]\nvariant = \"h3-turbo\"\n{old}\n"), "t")
                .unwrap_err()
                .to_string();
            assert!(e.contains("no longer a key"), "{old}: {e}");
        }
        // A whole pre-A3 block: the first retired key is named, not serde's "missing field".
        let e = Config::from_toml(
            "[[models]]\nid = \"fasth3\"\nfamily = \"h3\"\nrecipe = \"h3-turbo\"\n",
            "t",
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("`fasth3`: `family` is no longer a key"), "{e}");
        let e = Config::from_toml("[[models]]\nid = \"fasth3\"\n", "t")
            .unwrap_err()
            .to_string();
        assert!(e.contains("`variant` is required"), "{e}");
        let c = Config::from_toml("[[models]]\nvariant = \"\"\n", "t").unwrap();
        assert!(c.validate().is_err());
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
            ("FV_ADMIN_TOKEN", "fvadm_admin-secret-value"),
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
        for secret in [
            "cf-token-value",
            "r2-secret-value",
            "AKID",
            "admin-secret-value",
        ] {
            assert!(!shown.contains(secret), "{secret} leaked");
        }
        assert!(shown.contains("fv-media"));
    }

    #[test]
    fn key_store_and_console_env() {
        let mut c = Config::default();
        assert!(c.server.console);
        assert_eq!(c.key_store_backend(), KeyStoreBackend::File);
        c.apply_env(&env(&[
            ("FV_KEY_STORE", "memory"),
            ("FV_CONSOLE", "0"),
            ("FV_ADMIN_TOKEN", " t "),
        ]))
        .unwrap();
        assert_eq!(c.key_store_backend(), KeyStoreBackend::Memory);
        assert!(!c.server.console);
        assert_eq!(c.auth.admin_token.expose(), "t");
        c.auth.key_store = KeyStoreBackend::D1;
        assert!(c.validate().is_err(), "d1 key store without credentials");
        assert!(c.apply_env(&env(&[("FV_KEY_STORE", "redis")])).is_err());
    }

    #[test]
    fn callbacks_allow_private_is_opt_in() {
        let mut c = Config::default();
        assert!(
            !c.server.callbacks_allow_private,
            "the SSRF guard is on by default"
        );
        c.apply_env(&env(&[("FV_CALLBACKS_ALLOW_PRIVATE", "1")]))
            .unwrap();
        assert!(c.server.callbacks_allow_private);
        c.apply_env(&env(&[("FV_CALLBACKS_ALLOW_PRIVATE", "0")]))
            .unwrap();
        assert!(!c.server.callbacks_allow_private);
        let c = Config::from_toml("[server]\ncallbacks_allow_private = true\n", "t").unwrap();
        assert!(c.server.callbacks_allow_private);
    }

    #[test]
    fn workers_max_env() {
        let mut c = Config::default();
        assert_eq!(c.server.workers_max, 1);
        c.apply_env(&env(&[("FV_WORKERS_MAX", "3")])).unwrap();
        assert_eq!(c.server.workers_max, 3);
        c.validate().unwrap();
        for bad in ["0", "-1", "many"] {
            assert!(
                Config::default()
                    .apply_env(&env(&[("FV_WORKERS_MAX", bad)]))
                    .is_err(),
                "{bad}"
            );
        }
        let c = Config::from_toml("[server]\nworkers_max = 0\n", "t").unwrap();
        assert!(c.validate().is_err());
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

    #[test]
    fn a_workers_object_socket_needs_its_pool_or_families() {
        let mut w = Config::default();
        w.server.role = Role::Worker;
        w.cluster.internal_token = Secret("t".into());
        w.apply_env(&env(&[
            ("FV_DISPATCH_DO_URL", "https://edge.example"),
            ("FV_DISPATCH_CAPACITY", "3"),
        ]))
        .unwrap();
        assert_eq!(
            (w.dispatch.do_url.as_deref(), w.dispatch.capacity),
            (Some("https://edge.example"), 3)
        );
        assert!(w.validate().is_err(), "a worker's socket needs its pool");
        w.cluster.pool = Some("h3".into());
        w.validate().unwrap();
    }

    /// A direct worker (no gateway): a worker, its own client auth, and the
    /// cluster's admin token.
    #[test]
    fn direct_worker_needs_the_worker_role_client_auth_and_an_admin_token() {
        let mut w = Config::default();
        w.apply_env(&env(&[
            ("FV_SERVE_ROLE", "worker"),
            ("FV_INTERNAL_TOKEN", "t"),
            ("FV_WORKER_DIRECT", "1"),
        ]))
        .unwrap();
        assert!(w.cluster.direct);
        let e = w.validate().unwrap_err();
        assert!(e.to_string().contains("FV_ADMIN_TOKEN"), "{e}");
        w.apply_env(&env(&[("FV_ADMIN_TOKEN", "fvadm_x")])).unwrap();
        w.validate().unwrap();
        w.auth.mode = AuthMode::TrustGateway;
        assert!(w
            .validate()
            .unwrap_err()
            .to_string()
            .contains("trust-gateway"));
        w.auth.mode = AuthMode::Keys;
        w.server.role = Role::Standalone;
        assert!(w
            .validate()
            .unwrap_err()
            .to_string()
            .contains("server.role = worker"));
        let mut d = Config::default();
        d.apply_env(&env(&[("FV_WORKER_DIRECT", "0")])).unwrap();
        assert!(!d.cluster.direct);
    }

    /// Serializes `c` and parses the text back. A secret serializes as a
    /// placeholder, never its value, so the placeholders of unset secrets
    /// are dropped first (a file without secrets has none).
    fn round_trip(c: &Config, origin: &str) -> Config {
        let text = toml::to_string(c).unwrap_or_else(|e| panic!("{origin}: serialize: {e}"));
        let text: String = text
            .lines()
            .filter(|l| !l.ends_with("= \"<unset>\""))
            .map(|l| format!("{l}\n"))
            .collect();
        assert!(
            !text.contains("[gateway]"),
            "{origin}: serialized under the old name\n{text}"
        );
        Config::from_toml(&text, origin)
            .unwrap_or_else(|e| panic!("{origin}: re-parse: {e}\n{text}"))
    }

    fn value(c: &Config) -> toml::Value {
        toml::Value::try_from(c).unwrap()
    }

    /// `[cluster]` and its old name `[gateway]` (plan F3b): both parse to
    /// the same config, which serializes as `[cluster]` and parses back
    /// equal; only the old name leaves a deprecation note.
    #[test]
    fn cluster_and_its_old_name_gateway_round_trip() {
        let new = "[server]\nrole = \"worker\"\n\n[cluster]\npool = \"h3\"\nregister = false\ndirect = false\nwatch_poll_ms = 250\n";
        let old = new.replace("[cluster]", "[gateway]");
        let n = Config::from_toml(new, "new").unwrap();
        let o = Config::from_toml(&old, "old").unwrap();
        assert!(n.deprecations.is_empty(), "{:?}", n.deprecations);
        assert_eq!(o.deprecations.len(), 1, "{:?}", o.deprecations);
        assert!(
            o.deprecations[0].starts_with("old: [gateway] is read as [cluster]"),
            "{}",
            o.deprecations[0]
        );
        assert_eq!(value(&o), value(&n));
        assert_eq!(
            (
                n.cluster.pool.as_deref(),
                n.cluster.register,
                n.cluster.watch_poll_ms
            ),
            (Some("h3"), false, 250)
        );
        for (c, origin) in [(&n, "new"), (&o, "old")] {
            let back = round_trip(c, origin);
            assert_eq!(value(&back), value(c), "{origin}");
            assert!(
                back.deprecations.is_empty(),
                "{origin}: {:?}",
                back.deprecations
            );
        }
        // Any spelling of the old header (spaces, quotes, a trailing comment).
        for text in [
            "[ gateway ] # a comment\npool = \"h3\"\n",
            "[\"gateway\"]\npool = \"h3\"\n",
            "gateway.pool = \"h3\"\n",
        ] {
            let c = Config::from_toml(text, "t").unwrap();
            assert_eq!(
                (c.cluster.pool.as_deref(), c.deprecations.len()),
                (Some("h3"), 1),
                "{text}"
            );
        }
        // The defaults round-trip too (no section at all).
        let d = Config::default();
        assert_eq!(value(&round_trip(&d, "default")), value(&d));
    }

    /// Every key of the old table is read, and its schema is unchanged:
    /// an unknown key is refused under either name.
    #[test]
    fn the_old_name_reads_every_key_and_refuses_unknown_ones() {
        let c = Config::from_toml(
            "[gateway]\ninternal_token = \"t\"\npool = \"p\"\nregister = false\ndirect = true\nwatch_poll_ms = 7\n",
            "t",
        )
        .unwrap();
        assert_eq!(
            (
                c.cluster.internal_token.expose(),
                c.cluster.pool.as_deref(),
                c.cluster.register,
                c.cluster.direct,
                c.cluster.watch_poll_ms
            ),
            ("t", Some("p"), false, true, 7)
        );
        for name in ["cluster", "gateway"] {
            let e = Config::from_toml(&format!("[{name}]\nreactor_model = \"x\"\n"), "t")
                .unwrap_err()
                .to_string();
            assert!(e.contains("reactor_model"), "{name}: {e}");
        }
    }

    /// A file with both names is refused (whichever spelling sets them).
    #[test]
    fn cluster_and_gateway_together_are_refused() {
        for text in [
            "[cluster]\npool = \"a\"\n\n[gateway]\npool = \"b\"\n",
            "[gateway]\nregister = false\n\n[cluster]\n",
            "gateway.pool = \"a\"\n\n[cluster]\nregister = false\n",
        ] {
            let e = Config::from_toml(text, "t").unwrap_err();
            assert!(
                matches!(&e, ConfigError::Invalid(m) if m.contains("[cluster] and [gateway] are both set")),
                "{text}: {e}"
            );
        }
    }

    /// Every shipped config (configs/serve) round-trips, and reads the same
    /// with its `[gateway]` renamed to `[cluster]`. Stage 1 of the rename
    /// (docs/adr/0011-deploy-section.md, "Renaming a worker-config section"):
    /// they are sent to images built before it, whose fv-serve refuses an
    /// unknown `[cluster]`, so none says `[cluster]` yet.
    #[test]
    fn shipped_configs_round_trip_under_both_names() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/serve");
        let (mut n, mut with_old_name) = (0, 0);
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("toml"))
            .collect();
        entries.sort();
        for p in entries {
            let f = p.file_name().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).unwrap();
            let raw: toml::Table = text.parse().unwrap();
            assert!(
                !raw.contains_key("cluster"),
                "{f}: says [cluster]; the shipped configs keep [gateway] until every image reads [cluster] (ADR-0011, \"Renaming a worker-config section\")"
            );
            let c = Config::from_toml(&text, &f).unwrap();
            assert_eq!(value(&round_trip(&c, &f)), value(&c), "{f}");
            // Renamed as TOML reads the file (any header spelling: spaces,
            // quotes, a trailing comment, dotted keys), not by text.
            let mut moved = raw.clone();
            if let Some(t) = moved.remove(CLUSTER_OLD_NAME) {
                moved.insert("cluster".to_owned(), t);
                with_old_name += 1;
            }
            let renamed = toml::to_string(&moved).unwrap();
            let r = Config::from_toml(&renamed, &f).unwrap();
            assert!(r.deprecations.is_empty(), "{f}: {:?}", r.deprecations);
            assert_eq!(value(&r), value(&c), "{f}: [cluster] vs [gateway]");
            n += 1;
        }
        assert!(n >= 2, "configs/serve has {n} configs");
        assert!(with_old_name > 0, "no shipped config has a [gateway] table");
    }
}
