//! The fv-deploy data (docs/adr/0018-fv-deploy.md §3): what is launched
//! ([`Launch`]), what came of it ([`Instance`], [`InstanceState`]), what a
//! provider offers ([`Offer`]) and what the account holds ([`Account`]).
//!
//! Every type is serde + schemars; `schemas/{launch,instance,offer,account}.v1.json`
//! are their committed JSON Schemas (`tests/schema.rs` checks the drift), the
//! shared schema fv-control's generated TypeScript types come from (plan C3c).
//!
//! A [`Launch`] never carries a secret value: its `env` holds plain settings
//! and Runpod secret references (`{{ RUNPOD_SECRET_… }}`); the provider's API
//! key lives only in the client, and `guard::check_env` refuses a launch whose
//! env contains it.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The only `Launch` schema this crate reads and writes.
pub const LAUNCH_SCHEMA: u32 = 1;

/// Which provider, and which kind of instance on it (C3a: the two Runpod
/// kinds; C3b: `vast` and `gcp`, docs/adr/0018-fv-deploy.md §8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    /// A Runpod GPU pod (REST v1 `POST /pods`).
    RunpodPod,
    /// A Runpod serverless queue endpoint (REST v1 `POST /templates` +
    /// `POST /endpoints`).
    RunpodServerless,
    /// A Vast instance (REST `POST /bundles/` offer search, `PUT
    /// /asks/<offer>/` create; `scripts/serve/vast.sh`).
    Vast,
    /// A Google Compute Engine GPU VM (Compute REST v1 firewall + instance
    /// insert; `scripts/gcp/vm.sh`).
    Gcp,
}

impl ProviderKind {
    /// Every kind, in a fixed order (`--kind` help, tests).
    pub const ALL: [ProviderKind; 4] = [
        ProviderKind::RunpodPod,
        ProviderKind::RunpodServerless,
        ProviderKind::Vast,
        ProviderKind::Gcp,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::RunpodPod => "runpod-pod",
            ProviderKind::RunpodServerless => "runpod-serverless",
            ProviderKind::Vast => "vast",
            ProviderKind::Gcp => "gcp",
        }
    }

    pub fn parse(s: &str) -> Option<ProviderKind> {
        match s {
            "runpod-pod" | "pod" => Some(ProviderKind::RunpodPod),
            "runpod-serverless" | "serverless" => Some(ProviderKind::RunpodServerless),
            "vast" => Some(ProviderKind::Vast),
            "gcp" => Some(ProviderKind::Gcp),
            _ => None,
        }
    }

    /// Whether this kind runs on Runpod (the region guard's EU-volume rule).
    pub fn is_runpod(self) -> bool {
        matches!(
            self,
            ProviderKind::RunpodPod | ProviderKind::RunpodServerless
        )
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Runpod's cloud tier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CloudType {
    #[default]
    Secure,
    Community,
}

impl CloudType {
    pub fn as_str(self) -> &'static str {
        match self {
            CloudType::Secure => "SECURE",
            CloudType::Community => "COMMUNITY",
        }
    }
}

/// Where the weights come from (fv-boot's `FV_WEIGHTS_SOURCE`, ADR-0014 §3.3).
/// C3a launches `volume` only (the region guard's rule); the other sources
/// are typed so the schema does not change when they are allowed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum WeightsSource {
    #[default]
    Volume,
    R2,
    Hub,
    None,
    Registry,
}

impl WeightsSource {
    pub fn as_str(self) -> &'static str {
        match self {
            WeightsSource::Volume => "volume",
            WeightsSource::R2 => "r2",
            WeightsSource::Hub => "hub",
            WeightsSource::None => "none",
            WeightsSource::Registry => "registry",
        }
    }
}

/// Where a launch came from in the deployment data (`deploy/deploy.toml`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LaunchSource {
    /// The model family (`families/<family>.toml`).
    pub family: String,
    /// The preset id (a `[[presets]]` or `[[family_presets]]` id).
    pub preset: String,
    /// The serve image (`[[images]]` id: the CUDA image).
    pub image: String,
    /// The worker config (a `configs/serve` file name), sent inline as
    /// `FV_WORKER_TOML_B64` (the image bakes none).
    pub config: String,
}

/// The GPU request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GpuSpec {
    /// Provider GPU type ids in fallback order (one is tried at a time for a
    /// pod; a serverless endpoint gets the whole list).
    pub types: Vec<String>,
    pub count: u32,
    /// Runpod `allowedCudaVersions` (exact labels); empty = no filter (the
    /// default: REST cannot name the 13.1 / 13.2 hosts,
    /// `scripts/gpu/runpod-cuda.sh`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_cuda: Vec<String>,
    /// Runpod serverless `minCudaVersion`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_cuda: Option<String>,
}

/// A network volume and where it is mounted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VolumeMount {
    pub id: String,
    /// `/workspace` on a pod (a serverless worker always sees `/runpod-volume`).
    pub mount_path: String,
}

/// Where the instance runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    /// fv-control's region id (`eu`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Provider data centre ids in order (`EU-RO-1`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data_centers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<VolumeMount>,
    #[serde(default)]
    pub cloud: CloudType,
}

/// Disks, in GB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Disk {
    pub container_gb: u32,
    /// A pod's own volume (0: none; the network volume is separate).
    pub volume_gb: u32,
}

/// The money and readiness guards of one launch (docs/adr/0018-fv-deploy.md §5).
///
/// `null` (Rust `None`) in `max_usd_per_hr`, `cap_s`, `ready_timeout_s` or
/// `ready_cycles` means **not checked by the launcher that wrote this
/// launch** (§10.4): fv-control's serverless endpoints check no serverless
/// price and no worker readiness, a Runpod template launches nothing. It is
/// never a cap or a timeout: fv-deploy's own `up` refuses a launch with any
/// of them `null` ([`Guards::checked`]), and a ledger shows them as "not
/// checked". The balance floor is always checked (CLAUDE.md: $8). Every
/// field stays required in the schema: a launcher states each guard, a
/// number or `null`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Guards {
    /// Price cap ($/hr): an offer quoted above it is never created; a created
    /// instance billed above it is deleted at once. `null`: not checked by
    /// this launcher.
    #[schemars(required, schema_with = "nullable_double")]
    #[serde(deserialize_with = "present")]
    pub max_usd_per_hr: Option<f64>,
    /// Balance floor: nothing is created while the account balance is below
    /// it (CLAUDE.md: $8).
    pub min_balance_usd: f64,
    /// Wall-clock cap (s): the detached backstop deletes the instance this
    /// many seconds after its creation, whatever happens to the CLI. `null`:
    /// not checked by this launcher.
    #[schemars(required, schema_with = "nullable_uint64")]
    #[serde(deserialize_with = "present")]
    pub cap_s: Option<u64>,
    /// An instance not ready (`/ping` 200) this long (s) after its creation
    /// is deleted and replaced (the owner's rule: 600 s). `null`: not checked
    /// by this launcher.
    #[schemars(required, schema_with = "nullable_uint64")]
    #[serde(deserialize_with = "present")]
    pub ready_timeout_s: Option<u64>,
    /// At most this many instances are created for one `up` (the owner's
    /// rule: 3), then it gives up. Every create counts, a GPU-type fallback
    /// after a billed-over-cap delete included. `null`: not checked by this
    /// launcher.
    #[schemars(required, schema_with = "nullable_uint32")]
    #[serde(deserialize_with = "present")]
    pub ready_cycles: Option<u32>,
    /// Lifts the upper bounds (cap ≤ 4 h, ≤ 3 cycles, ≤ 600 s readiness;
    /// `--force-guards`). The $8 floor is never lifted.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub force: bool,
}

/// A guard: the key must be there, its value a number or `null` (serde would
/// read a missing `Option` field as `None`, which here means "not checked":
/// a launcher that forgets a guard must not pass as one that states it).
fn present<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d)
}

// A guard's schema: required, a number or `null` (schemars would make an
// `Option` field optional, and `required` alone drops the `null`).
fn nullable_double(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({"type": ["number", "null"], "format": "double"})
}

fn nullable_uint64(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({"type": ["integer", "null"], "format": "uint64", "minimum": 0})
}

fn nullable_uint32(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({"type": ["integer", "null"], "format": "uint32", "minimum": 0})
}

/// The guards with every number present: what fv-deploy's `up` runs on
/// ([`Guards::checked`]). Not serialized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CheckedGuards {
    pub max_usd_per_hr: f64,
    pub min_balance_usd: f64,
    pub cap_s: u64,
    pub ready_timeout_s: u64,
    pub ready_cycles: u32,
    pub force: bool,
}

/// What a ledger writes for a guard `null` in the launch.
pub const NOT_CHECKED: &str = "not checked";

impl Guards {
    /// Every guard set (what fv-deploy writes).
    pub fn all(
        max_usd_per_hr: f64,
        min_balance_usd: f64,
        cap_s: u64,
        ready_timeout_s: u64,
        ready_cycles: u32,
    ) -> Self {
        Self {
            max_usd_per_hr: Some(max_usd_per_hr),
            min_balance_usd,
            cap_s: Some(cap_s),
            ready_timeout_s: Some(ready_timeout_s),
            ready_cycles: Some(ready_cycles),
            force: false,
        }
    }

    /// The names of the guards this launch leaves unchecked (`null`), in
    /// schema order.
    pub fn not_checked(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.max_usd_per_hr.is_none() {
            v.push("max_usd_per_hr");
        }
        if self.cap_s.is_none() {
            v.push("cap_s");
        }
        if self.ready_timeout_s.is_none() {
            v.push("ready_timeout_s");
        }
        if self.ready_cycles.is_none() {
            v.push("ready_cycles");
        }
        v
    }

    /// Every number, or the names of the ones that are `null` (fv-deploy's
    /// `up` checks every guard: a launch another launcher wrote with "not
    /// checked" guards is refused, never run with a made-up value).
    pub fn checked(&self) -> Result<CheckedGuards, Vec<&'static str>> {
        match (
            self.max_usd_per_hr,
            self.cap_s,
            self.ready_timeout_s,
            self.ready_cycles,
        ) {
            (Some(max_usd_per_hr), Some(cap_s), Some(ready_timeout_s), Some(ready_cycles)) => {
                Ok(CheckedGuards {
                    max_usd_per_hr,
                    min_balance_usd: self.min_balance_usd,
                    cap_s,
                    ready_timeout_s,
                    ready_cycles,
                    force: self.force,
                })
            }
            _ => Err(self.not_checked()),
        }
    }

    /// The guards as a ledger shows them: each number, or "not checked"
    /// (`null` in the launch), in schema order.
    pub fn ledger_view(&self) -> serde_json::Value {
        fn num<T: Into<serde_json::Value>>(v: Option<T>) -> serde_json::Value {
            v.map_or_else(|| serde_json::Value::from(NOT_CHECKED), Into::into)
        }
        let mut m = serde_json::Map::new();
        m.insert("max_usd_per_hr".into(), num(self.max_usd_per_hr));
        m.insert("min_balance_usd".into(), self.min_balance_usd.into());
        m.insert("cap_s".into(), num(self.cap_s));
        m.insert("ready_timeout_s".into(), num(self.ready_timeout_s));
        m.insert("ready_cycles".into(), num(self.ready_cycles));
        if self.force {
            m.insert("force".into(), true.into());
        }
        serde_json::Value::Object(m)
    }

    /// One line for the local ledger: `max_usd_per_hr=2.5 min_balance_usd=8
    /// cap_s=1800 ready_timeout_s=600 ready_cycles=3`, "not-checked" for a
    /// `null`.
    pub fn ledger_line(&self) -> String {
        fn word<T: std::fmt::Display>(v: Option<T>) -> String {
            v.map_or_else(|| "not-checked".to_owned(), |x| x.to_string())
        }
        format!(
            "max_usd_per_hr={} min_balance_usd={} cap_s={} ready_timeout_s={} ready_cycles={}",
            word(self.max_usd_per_hr),
            self.min_balance_usd,
            word(self.cap_s),
            word(self.ready_timeout_s),
            word(self.ready_cycles)
        )
    }
}

/// A Runpod serverless queue endpoint's scaling fields.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ServerlessSpec {
    pub workers_min: u32,
    pub workers_max: u32,
    pub idle_timeout_s: u32,
    pub flashboot: bool,
    pub execution_timeout_ms: u64,
    pub scaler_type: String,
    pub scaler_value: u32,
}

/// One instance to create: the provider-neutral request. Built from a family
/// preset plus a provider section (`crate::launch`), turned into the
/// provider's exact payload by [`crate::provider::Provider::payload`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    /// Always 1.
    pub schema: u32,
    /// The instance name (`fv-deploy-<preset>-<MMDDHHMMSS>`).
    pub name: String,
    pub provider: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<LaunchSource>,
    /// A digest-pinned image reference (`…@sha256:…`) when launched; `plan`
    /// may show a tag.
    pub image: String,
    /// The container entrypoint (`/opt/fastvideo-rs/bin/fv-boot`); empty: the
    /// image's own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entrypoint: Vec<String>,
    /// Its arguments (Runpod `dockerStartCmd`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// The container environment: plain values and provider secret
    /// references only, never a secret value.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Published ports (`8000/http`, `70000/tcp`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<String>,
    pub gpu: GpuSpec,
    pub placement: Placement,
    pub disk: Disk,
    #[serde(default)]
    pub weights: WeightsSource,
    pub guards: Guards,
    /// Serverless endpoints only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serverless: Option<ServerlessSpec>,
}

impl Launch {
    /// The same launch narrowed to one GPU type (a pod is created on one
    /// type at a time, each behind its own price check).
    pub fn for_gpu(&self, gpu: &str) -> Launch {
        let mut l = self.clone();
        l.gpu.types = vec![gpu.to_owned()];
        l
    }
}

/// A provider's instance id (a Runpod pod id; a serverless endpoint id,
/// `<endpoint>/<template>` when fv-deploy created the template too).
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct InstanceId(pub String);

impl InstanceId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for InstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the provider says about an instance (readiness is not the
/// provider's: it is the instance's own `/ping`, `crate::provider::Probe`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InstanceState {
    /// Created; no container running yet.
    Pending,
    /// The container (or the endpoint) is up.
    Running,
    /// Stopped; it still exists (and may still bill its disk).
    Exited,
    /// Deleted, or unknown to the provider.
    Terminated,
    /// A provider state fv-deploy does not map.
    Unknown,
}

/// A created instance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Instance {
    pub id: InstanceId,
    pub provider: ProviderKind,
    pub name: String,
    pub state: InstanceState,
    /// The instance's HTTP base (a pod's `https://<id>-8000.proxy.runpod.net`;
    /// a serverless endpoint's queue API base).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_center: Option<String>,
    /// Billed $/hr when the provider says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usd_per_hr: Option<f64>,
    /// Unix seconds (fv-deploy's clock when it created it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<f64>,
}

/// What [`Offer`]s to quote.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OfferQuery {
    pub gpu_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_center: Option<String>,
    #[serde(default)]
    pub cloud: CloudType,
}

/// One quote: a GPU type (in a data centre) at a price.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Offer {
    pub provider: ProviderKind,
    pub gpu_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_center: Option<String>,
    pub cloud: CloudType,
    /// $/hr for one GPU; `None`: no quote (never created at an unknown price).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usd_per_hr: Option<f64>,
    /// The provider's stock word (`High`, `Low`, …) when it gives one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stock: Option<String>,
}

/// The account, for the balance floor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Account {
    pub provider: ProviderKind,
    pub balance_usd: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spend_per_hr_usd: Option<f64>,
}

/// Where a log read resumes (opaque to callers).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct Cursor(pub String);

/// Log lines since a cursor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LogPage {
    pub lines: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<Cursor>,
}

/// One request a provider would send (what `plan` prints; no secret: the
/// API key is a header the client adds).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ApiCall {
    pub method: String,
    pub url: String,
    pub body: serde_json::Value,
}
