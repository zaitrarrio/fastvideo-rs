//! How a [`Launch`] is built (docs/adr/0018-fv-deploy.md §3.2): a preset of
//! the deployment data (`deploy/deploy.toml`, ADR-0011, with the family
//! manifests it names) plus a provider section (the region's data centre,
//! volume and GPU types, the image channel, the guards).
//!
//! - [`Repo::load`] reads the repository's `deploy/deploy.toml`,
//!   `families/*.toml` and the EU volume's tree list
//!   (`scripts/gpu/volumes.sh` `FV_EU_VOLUME_TREES`).
//! - [`from_preset`] builds the launch fv-deploy creates: fv-boot as the
//!   entrypoint (ADR-0014), the preset's config baked (`--config /etc/fv/…`)
//!   or inline (`FV_WORKER_TOML_B64`), volume weights on the EU volume.
//! - [`script_pod`] / [`script_endpoint`] build what `runpod-pod.sh` /
//!   `runpod-endpoint.sh` send, so the payload goldens pin the provider code
//!   to the scripts it replaces.
//!
//! Pure but for [`Repo::load`] (file reads), [`ghcr_resolve`] (HTTP through
//! the seam) and [`run_key`] (`/dev/urandom`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine as _;
use fastvideo_manifest::deploy::{DeployManifest, EU_DATA_CENTER, EU_REGION, EU_VOLUME_ID};
use fastvideo_manifest::family::{tree_covers, FamilySet};
use sha2::{Digest, Sha256};

use crate::boot::sys::{Http, Method, Request};
use crate::guard;
use crate::provider::{
    CloudType, Disk, GpuSpec, Guards, Launch, LaunchSource, Placement, ProviderKind,
    ServerlessSpec, VolumeMount, WeightsSource, LAUNCH_SCHEMA,
};

/// The serve images' one GHCR package (`scripts/serve/variants.sh` `FV_SERVE_REPO`).
pub const SERVE_REPO: &str = "ghcr.io/zaitrarrio/fastvideo-rs-serve";
/// fv-boot in the serve images (ADR-0014; `fixtures/boot-protocol.json`).
pub const FV_BOOT_BIN: &str = "/opt/fastvideo-rs/bin/fv-boot";
/// fv-serve in the serve images.
pub const FV_SERVE_BIN: &str = "/opt/fastvideo-rs/bin/fv-serve";
/// Where the images carry their worker configs.
pub const IMAGE_CONFIG_DIR: &str = "/etc/fv";
/// The EU region's GPU types (`scripts/gpu/volumes.sh` `FV_EU_GPUS`; plan §1:
/// RTX PRO 6000 by default).
pub const EU_GPUS: [&str; 1] = ["NVIDIA RTX PRO 6000 Blackwell Server Edition"];
/// A pod's network-volume mount.
pub const POD_VOLUME_MOUNT: &str = "/workspace";
/// The serverless workers' minimum CUDA (`scripts/gpu/runpod-cuda.sh` `FV_MIN_CUDA`).
pub const MIN_CUDA: &str = "13.0";
/// `plan`'s stand-in for the run key's hash (`runpod-pod.sh plan`).
pub const KEY_HASH_PLACEHOLDER: &str = "<sha256 of the run key>";
/// A pod's container disk when the preset does not size it (fv-control
/// `workerCreatePayload`: 40 GB with the volume).
pub const POD_DISK_GB: u32 = 40;
/// A serverless template's container disk (`runpod-endpoint.sh`).
pub const TEMPLATE_DISK_GB: u32 = 20;

/// Why no launch could be built.
#[derive(Debug, thiserror::Error)]
pub enum LaunchError {
    #[error("{path}: {why}")]
    Io { path: PathBuf, why: String },
    #[error("deployment data: {0}")]
    Deploy(String),
    #[error("{0}")]
    Preset(String),
    #[error("region: {0}")]
    Region(String),
    #[error("image {image}: {why}")]
    Image { image: String, why: String },
}

/// The Runpod secret references every serve container gets (values live in
/// Runpod, never here; `runpod-pod.sh` `SECRET_ENV_JSON`, fv-control
/// `SECRET_ENV_REFS`).
pub fn secret_env_refs() -> BTreeMap<String, String> {
    [
        ("FV_CF_ACCOUNT_ID", "fv_cf_account_id"),
        ("FV_CF_API_TOKEN", "fv_cf_api_token"),
        ("FV_D1_DATABASE_ID", "fv_d1_database_id"),
        ("FV_R2_BUCKET", "fv_r2_bucket"),
        ("FV_R2_ENDPOINT", "fv_r2_endpoint"),
        ("FV_R2_ACCESS_KEY_ID", "fv_r2_access_key_id"),
        ("FV_R2_SECRET_ACCESS_KEY", "fv_r2_secret_access_key"),
        ("FV_WEBHOOK_ED25519_KEY", "fv_webhook_ed25519_key"),
    ]
    .into_iter()
    .map(|(k, s)| (k.to_owned(), format!("{{{{ RUNPOD_SECRET_{s} }}}}")))
    .collect()
}

/// `FV_IMAGE_REF` / `FV_IMAGE_DIGEST` / `FV_RELEASE_CHANNEL`
/// (`variants.sh fv_image_env_json`, fv-control `imageIdentEnv`).
pub fn image_ident_env(image: &str, channel: Option<&str>) -> BTreeMap<String, String> {
    let mut e = BTreeMap::new();
    e.insert("FV_IMAGE_REF".to_owned(), image.to_owned());
    if let Some((_, d)) = image.split_once('@') {
        if image.contains("@sha256:") {
            e.insert("FV_IMAGE_DIGEST".to_owned(), d.to_owned());
        }
    }
    if let Some(c) = channel.filter(|c| !c.is_empty()) {
        e.insert("FV_RELEASE_CHANNEL".to_owned(), c.to_owned());
    }
    e
}

/// Points a launch at a (resolved) image and refreshes its identity env.
pub fn pin_image(launch: &mut Launch, image: &str) {
    launch.image = image.to_owned();
    let channel = launch.env.get("FV_RELEASE_CHANNEL").cloned();
    launch.env.remove("FV_IMAGE_DIGEST");
    launch
        .env
        .extend(image_ident_env(image, channel.as_deref()));
}

/// `runpod-endpoint.sh`'s endpoint scaling: workers 0..1, idle 5 s, FlashBoot
/// off, 30 min execution timeout, `QUEUE_DELAY` 1.
pub fn default_serverless() -> ServerlessSpec {
    ServerlessSpec {
        workers_min: 0,
        workers_max: 1,
        idle_timeout_s: 5,
        flashboot: false,
        execution_timeout_ms: 1_800_000,
        scaler_type: "QUEUE_DELAY".into(),
        scaler_value: 1,
    }
}

/// A region's provider section: its data centre, weights volume and GPU types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionDefaults {
    pub region: String,
    pub data_center: String,
    pub volume: String,
    pub gpus: Vec<String>,
}

/// `eu` only: the US volume was deleted 2026-10 and is not rebuilt
/// (CLAUDE.md, `scripts/gpu/volumes.sh` `FV_US_GONE`).
pub fn region_defaults(region: &str) -> Result<RegionDefaults, LaunchError> {
    match region {
        EU_REGION => Ok(RegionDefaults {
            region: EU_REGION.into(),
            data_center: EU_DATA_CENTER.into(),
            volume: EU_VOLUME_ID.into(),
            gpus: EU_GPUS.iter().map(|s| (*s).to_owned()).collect(),
        }),
        "us" => Err(LaunchError::Region(guard::US_GONE.into())),
        other => Err(LaunchError::Region(format!("unknown region {other} (eu)"))),
    }
}

/// The trees of `FV_EU_VOLUME_TREES` in `scripts/gpu/volumes.sh` (the same
/// reading as fv-control's `serverless-presets.test.ts`).
pub fn parse_eu_trees(volumes_sh: &str) -> Vec<String> {
    let Some(start) = volumes_sh.find("FV_EU_VOLUME_TREES=\"") else {
        return Vec::new();
    };
    let rest = &volumes_sh[start + "FV_EU_VOLUME_TREES=\"".len()..];
    let body = rest.split('"').next().unwrap_or("");
    body.replace("\\\n", " ")
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// The repository's deployment data.
#[derive(Clone, Debug)]
pub struct Repo {
    pub root: PathBuf,
    pub deploy: DeployManifest,
    pub families: FamilySet,
    /// `FV_EU_VOLUME_TREES`.
    pub eu_trees: Vec<String>,
}

fn read(p: &Path) -> Result<String, LaunchError> {
    std::fs::read_to_string(p).map_err(|e| LaunchError::Io {
        path: p.to_owned(),
        why: e.to_string(),
    })
}

impl Repo {
    /// Reads `deploy/deploy.toml` (checked against `families/`) and the EU
    /// tree list under `root`.
    pub fn load(root: &Path) -> Result<Repo, LaunchError> {
        let families = fastvideo_manifest::family::load_dir(&root.join("families"))
            .map_err(|e| LaunchError::Deploy(e.to_string()))?;
        let deploy =
            DeployManifest::parse_with(&read(&root.join("deploy/deploy.toml"))?, &families)
                .map_err(|e| LaunchError::Deploy(e.to_string()))?;
        let eu_trees = parse_eu_trees(&read(&root.join("scripts/gpu/volumes.sh"))?);
        Ok(Repo {
            root: root.to_owned(),
            deploy,
            families,
            eu_trees,
        })
    }

    /// The nearest ancestor of `start` holding `deploy/deploy.toml`.
    pub fn find_root(start: &Path) -> Option<PathBuf> {
        start
            .ancestors()
            .find(|d| d.join("deploy/deploy.toml").is_file() && d.join("families").is_dir())
            .map(Path::to_owned)
    }
}

/// What to launch, from the command line.
#[derive(Clone, Debug, PartialEq)]
pub struct PresetRequest {
    pub family: String,
    pub preset: String,
    pub region: String,
    pub kind: ProviderKind,
    /// An image reference that replaces the preset's channel tag.
    pub image: Option<String>,
    /// `stable` or `latest` (`docs/serve/releases.md`).
    pub channel: String,
    /// GPU types in order (default: the region's).
    pub gpu_types: Vec<String>,
    /// A volume id to mount instead of the region's (the guard refuses any
    /// other than the EU volume).
    pub volume: Option<String>,
    pub guards: Guards,
    /// `FV_API_KEYS` (the run key's SHA-256; `plan` shows the placeholder).
    pub key_hash: String,
}

/// The resolved preset: image variant, config, trees, disk.
struct Resolved {
    image_variant: String,
    config: String,
    trees: Vec<String>,
    container_disk_gb: Option<u32>,
}

fn resolve_preset(repo: &Repo, family: &str, preset: &str) -> Result<Resolved, LaunchError> {
    let d = &repo.deploy;
    let (image_variant, config, variants, disk) = if let Some(p) = d.preset(preset) {
        (
            p.image.clone(),
            p.config.clone(),
            d.preset_variants(&repo.families, p),
            p.container_disk_gb,
        )
    } else if let Some(fp) = d.family_preset(preset) {
        let disk = fp
            .members
            .iter()
            .filter_map(|m| d.preset(m).and_then(|p| p.container_disk_gb))
            .max();
        (
            fp.image.clone(),
            fp.config.clone(),
            d.family_preset_variants(&repo.families, fp),
            disk,
        )
    } else {
        let ids: Vec<&str> = d
            .presets
            .iter()
            .filter(|p| {
                d.preset_variants(&repo.families, p)
                    .iter()
                    .all(|(f, _)| f.family.id.as_str() == family)
            })
            .map(|p| p.id.as_str())
            .collect();
        return Err(LaunchError::Preset(format!(
            "no preset {preset} in deploy/deploy.toml ({family} presets: {})",
            ids.join(", ")
        )));
    };
    if variants.is_empty() {
        return Err(LaunchError::Preset(format!(
            "preset {preset} serves no variant"
        )));
    }
    if let Some((f, v)) = variants
        .iter()
        .find(|(f, _)| f.family.id.as_str() != family)
    {
        return Err(LaunchError::Preset(format!(
            "preset {preset} serves {} of family {}, not {family}",
            v.id,
            f.family.id.as_str()
        )));
    }
    let mut trees: Vec<String> = variants
        .iter()
        .flat_map(|(_, v)| v.trees.iter().cloned())
        .collect();
    trees.sort();
    trees.dedup();
    Ok(Resolved {
        image_variant,
        config,
        trees,
        container_disk_gb: disk,
    })
}

/// The launch fv-deploy creates for a preset (docs/adr/0018-fv-deploy.md
/// §3.2). `stamp` names it (`MMDDHHMMSS`).
pub fn from_preset(repo: &Repo, req: &PresetRequest, stamp: &str) -> Result<Launch, LaunchError> {
    let r = resolve_preset(repo, &req.family, &req.preset)?;
    let region = region_defaults(&req.region)?;
    let volume = req.volume.clone().unwrap_or(region.volume.clone());
    // Volume weights: every tree the preset reads must be on the EU volume.
    let missing: Vec<&str> = r
        .trees
        .iter()
        .filter(|t| !repo.eu_trees.iter().any(|e| tree_covers(e, t)))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        return Err(LaunchError::Preset(format!(
            "preset {}: trees not on the EU volume {EU_VOLUME_ID} ({}); weights_source r2 / hub is not in fv-deploy yet (scripts/gpu/volumes.sh FV_EU_VOLUME_TREES)",
            req.preset,
            missing.join(", ")
        )));
    }
    let image_def = repo.deploy.image(&r.image_variant).ok_or_else(|| {
        LaunchError::Deploy(format!(
            "no image {} in deploy/deploy.toml",
            r.image_variant
        ))
    })?;
    let baked = image_def.carries(&r.config);
    let (image, channel) = match &req.image {
        Some(i) => (i.clone(), None),
        None => (
            format!("{SERVE_REPO}:{}-{}", r.image_variant, req.channel),
            Some(req.channel.as_str()),
        ),
    };
    let mut env = secret_env_refs();
    env.extend(image_ident_env(&image, channel));
    let mut args = Vec::new();
    if baked {
        args = vec![
            "--config".to_owned(),
            format!("{IMAGE_CONFIG_DIR}/{}", r.config),
        ];
    } else {
        let p = repo.root.join("configs/serve").join(&r.config);
        let text = read(&p)?;
        env.insert(
            "FV_WORKER_TOML_B64".into(),
            base64::engine::general_purpose::STANDARD.encode(text.as_bytes()),
        );
    }
    env.insert(
        "FV_WEIGHTS_SOURCE".into(),
        WeightsSource::Volume.as_str().into(),
    );
    env.insert("FV_STATE_DIR".into(), "/fvstate".into());
    env.insert("RUST_LOG".into(), "info".into());
    let gpus = if req.gpu_types.is_empty() {
        region.gpus.clone()
    } else {
        req.gpu_types.clone()
    };
    let name = format!("fv-deploy-{}-{stamp}", req.preset);
    let source = Some(LaunchSource {
        family: req.family.clone(),
        preset: req.preset.clone(),
        image_variant: r.image_variant.clone(),
        config: r.config.clone(),
        config_baked: baked,
    });
    let placement = Placement {
        region: Some(region.region.clone()),
        data_centers: vec![region.data_center.clone()],
        volume: Some(VolumeMount {
            id: volume,
            mount_path: POD_VOLUME_MOUNT.into(),
        }),
        cloud: CloudType::Secure,
    };
    Ok(match req.kind {
        ProviderKind::RunpodPod => {
            env.insert("FV_SERVE_MODE".into(), "http".into());
            env.insert("FV_WEIGHTS".into(), "/workspace/weights".into());
            env.insert("FV_API_KEYS".into(), req.key_hash.clone());
            env.insert("FV_SERVE_FORWARD".into(), "1".into());
            Launch {
                schema: LAUNCH_SCHEMA,
                name,
                provider: req.kind,
                source,
                image,
                entrypoint: vec![FV_BOOT_BIN.into()],
                args,
                env,
                ports: vec!["8000/http".into(), "70000/tcp".into()],
                gpu: GpuSpec {
                    types: gpus,
                    count: 1,
                    allowed_cuda: Vec::new(),
                    min_cuda: None,
                },
                placement,
                disk: Disk {
                    container_gb: r.container_disk_gb.unwrap_or(POD_DISK_GB),
                    volume_gb: 0,
                },
                weights: WeightsSource::Volume,
                guards: req.guards,
                serverless: None,
            }
        }
        ProviderKind::RunpodServerless => {
            env.insert("FV_SERVE_MODE".into(), "runpod-queue".into());
            env.insert("FV_AUTH_MODE".into(), "trust-gateway".into());
            env.insert("FV_WEIGHTS".into(), "/runpod-volume/weights".into());
            env.insert("FV_CACHE_DIR".into(), "/fvstate/cache".into());
            Launch {
                schema: LAUNCH_SCHEMA,
                name,
                provider: req.kind,
                source,
                image,
                entrypoint: vec![FV_BOOT_BIN.into()],
                args,
                env,
                ports: Vec::new(),
                gpu: GpuSpec {
                    types: gpus,
                    count: 1,
                    allowed_cuda: Vec::new(),
                    min_cuda: Some(MIN_CUDA.into()),
                },
                placement,
                disk: Disk {
                    container_gb: r.container_disk_gb.unwrap_or(TEMPLATE_DISK_GB),
                    volume_gb: 0,
                },
                weights: WeightsSource::Volume,
                guards: req.guards,
                serverless: Some(default_serverless()),
            }
        }
    })
}

/// What `scripts/serve/runpod-pod.sh` creates (its `payload()` inputs):
/// fv-serve as the entrypoint with `--config <config>`, the run key's hash,
/// a volume at `/workspace` in its data centre, no CUDA filter.
pub struct ScriptPod<'a> {
    pub name: &'a str,
    pub image: &'a str,
    pub gpu: &'a str,
    pub key_hash: &'a str,
    pub config: &'a str,
    /// `(volume id, its data centre)`.
    pub volume: Option<(&'a str, &'a str)>,
    pub guards: Guards,
}

pub fn script_pod(p: &ScriptPod<'_>) -> Launch {
    let mut env = secret_env_refs();
    env.extend(image_ident_env(p.image, None));
    for (k, v) in [
        ("FV_SERVE_MODE", "http"),
        ("FV_STATE_DIR", "/fvstate"),
        ("FV_WEIGHTS", "/workspace/weights"),
        ("FV_API_KEYS", p.key_hash),
        ("FV_SERVE_FORWARD", "1"),
        ("RUST_LOG", "info"),
    ] {
        env.insert(k.into(), v.into());
    }
    Launch {
        schema: LAUNCH_SCHEMA,
        name: p.name.into(),
        provider: ProviderKind::RunpodPod,
        source: None,
        image: p.image.into(),
        entrypoint: vec![FV_SERVE_BIN.into()],
        args: vec!["--config".into(), p.config.into()],
        env,
        ports: vec!["8000/http".into(), "70000/tcp".into()],
        gpu: GpuSpec {
            types: vec![p.gpu.into()],
            count: 1,
            allowed_cuda: Vec::new(),
            min_cuda: None,
        },
        placement: Placement {
            region: None,
            data_centers: p.volume.map(|(_, dc)| vec![dc.into()]).unwrap_or_default(),
            volume: p.volume.map(|(id, _)| VolumeMount {
                id: id.into(),
                mount_path: POD_VOLUME_MOUNT.into(),
            }),
            cloud: CloudType::Secure,
        },
        disk: Disk {
            container_gb: 30,
            volume_gb: 0,
        },
        weights: WeightsSource::Volume,
        guards: p.guards,
        serverless: None,
    }
}

/// The serverless entrypoint `runpod-endpoint.sh` sends (links the volume's
/// weights where a pod sees them, then fv-serve; fv-boot does the same).
pub const SCRIPT_LINK_WEIGHTS: &str = "if [ -d /runpod-volume/weights ] && [ ! -e /workspace/weights ]; then mkdir -p /workspace && ln -s /runpod-volume/weights /workspace/weights; fi; exec /opt/fastvideo-rs/bin/fv-serve \"$@\"";

/// What `scripts/serve/runpod-endpoint.sh` creates for a queue endpoint
/// (`template_payload` + `endpoint_payload` inputs).
pub struct ScriptEndpoint<'a> {
    pub name: &'a str,
    pub image: &'a str,
    pub gpus: &'a [&'a str],
    pub config: &'a str,
    pub volume: &'a str,
    pub data_center: &'a str,
    pub guards: Guards,
}

pub fn script_endpoint(p: &ScriptEndpoint<'_>) -> Launch {
    let mut env = secret_env_refs();
    env.extend(image_ident_env(p.image, None));
    for (k, v) in [
        ("FV_SERVE_MODE", "runpod-queue"),
        ("FV_AUTH_MODE", "trust-gateway"),
        ("FV_STATE_DIR", "/fvstate"),
        ("FV_WEIGHTS", "/runpod-volume/weights"),
        ("FV_CACHE_DIR", "/fvstate/cache"),
        ("RUST_LOG", "info"),
    ] {
        env.insert(k.into(), v.into());
    }
    Launch {
        schema: LAUNCH_SCHEMA,
        name: p.name.into(),
        provider: ProviderKind::RunpodServerless,
        source: None,
        image: p.image.into(),
        entrypoint: vec![
            "/bin/sh".into(),
            "-c".into(),
            SCRIPT_LINK_WEIGHTS.into(),
            "fv-serve".into(),
        ],
        args: vec!["--config".into(), p.config.into()],
        env,
        ports: Vec::new(),
        gpu: GpuSpec {
            types: p.gpus.iter().map(|g| (*g).to_owned()).collect(),
            count: 1,
            allowed_cuda: Vec::new(),
            min_cuda: Some(MIN_CUDA.into()),
        },
        placement: Placement {
            region: None,
            data_centers: vec![p.data_center.into()],
            volume: Some(VolumeMount {
                id: p.volume.into(),
                mount_path: "/runpod-volume".into(),
            }),
            cloud: CloudType::Secure,
        },
        disk: Disk {
            container_gb: TEMPLATE_DISK_GB,
            volume_gb: 0,
        },
        weights: WeightsSource::Volume,
        guards: p.guards,
        serverless: Some(default_serverless()),
    }
}

/// A fresh run key (`fvk-<32 hex>`, as `runpod-pod.sh`'s `openssl rand -hex
/// 16`), from `/dev/urandom`.
pub fn run_key() -> Result<String, LaunchError> {
    use std::io::Read as _;
    let mut b = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| LaunchError::Io {
            path: "/dev/urandom".into(),
            why: e.to_string(),
        })?;
    Ok(format!("fvk-{}", hex(&b)))
}

/// SHA-256 of a run key, lower-case hex (`FV_API_KEYS`).
pub fn key_hash(key: &str) -> String {
    hex(&Sha256::digest(key.as_bytes()))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// `(year, month, day, hour, minute, second)` (UTC) of a unix time.
pub fn utc_parts(unix: f64) -> (i64, i64, i64, i64, i64, i64) {
    let secs = if unix.is_finite() && unix > 0.0 {
        unix as i64
    } else {
        0
    };
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    // Civil from days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d, tod / 3600, tod / 60 % 60, tod % 60)
}

/// `MMDDHHMMSS` (UTC) of a unix time (the scripts' `date -u +%m%d%H%M%S`).
pub fn utc_stamp(unix: f64) -> String {
    let (_, m, d, h, mi, s) = utc_parts(unix);
    format!("{m:02}{d:02}{h:02}{mi:02}{s:02}")
}

/// `YYYY-MM-DDTHH:MM:SSZ` (the ledger's `date -u +%FT%TZ`).
pub fn utc_iso(unix: f64) -> String {
    let (y, m, d, h, mi, s) = utc_parts(unix);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// The `Accept` header the scripts' `resolve_digest` sends.
pub const MANIFEST_ACCEPT: &str = "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json";

/// A `ghcr.io/<repo>:<tag>` reference pinned to its digest
/// (`runpod-pod.sh resolve_digest`): an anonymous pull token, then the
/// manifest; the digest is the SHA-256 of the manifest bytes (what the
/// registry's `Docker-Content-Digest` header carries). A digest reference is
/// returned as is.
pub fn ghcr_resolve(http: &dyn Http, image: &str) -> Result<String, LaunchError> {
    if image.contains("@sha256:") {
        return Ok(image.to_owned());
    }
    let err = |why: String| LaunchError::Image {
        image: image.to_owned(),
        why,
    };
    let repo_tag = image
        .strip_prefix("ghcr.io/")
        .ok_or_else(|| err("only ghcr.io tags are resolved; pass a digest".into()))?;
    let (repo, tag) = match repo_tag.rsplit_once(':') {
        Some((r, t)) if !t.contains('/') => (r, t),
        _ => (repo_tag, "latest"),
    };
    let tok_url = format!("https://ghcr.io/token?scope=repository:{repo}:pull");
    let r = http
        .send(&Request::new(
            Method::Get,
            &tok_url,
            Duration::from_secs(30),
        ))
        .map_err(|e| err(e.to_string()))?;
    let token = serde_json::from_slice::<serde_json::Value>(&r.body)
        .ok()
        .and_then(|v| v.get("token").and_then(|t| t.as_str()).map(str::to_owned))
        .ok_or_else(|| err(format!("no pull token (HTTP {})", r.status)))?;
    let url = format!("https://ghcr.io/v2/{repo}/manifests/{tag}");
    let m = http
        .send(
            &Request::new(Method::Get, &url, Duration::from_secs(30))
                .bearer(&token)
                .header("accept", MANIFEST_ACCEPT),
        )
        .map_err(|e| err(e.to_string()))?;
    if !m.ok() || m.body.is_empty() {
        return Err(err(format!(
            "manifest HTTP {} (private package, or no such tag)",
            m.status
        )));
    }
    Ok(format!(
        "ghcr.io/{repo}@sha256:{}",
        hex(&Sha256::digest(&m.body))
    ))
}
