//! How a [`Launch`] is built (docs/adr/0018-fv-deploy.md §3.2): a preset of
//! the deployment data (`deploy/deploy.toml`, ADR-0011, with the family
//! manifests it names) plus a provider section (the region's data centre,
//! volume and GPU types, the image channel, the guards).
//!
//! - [`Repo::load`] reads the repository's `deploy/deploy.toml`,
//!   `families/*.toml` and the EU volume's tree list
//!   (`scripts/gpu/volumes.sh` `FV_EU_VOLUME_TREES`).
//! - [`from_preset`] builds the launch fv-deploy creates: fv-boot as the
//!   entrypoint (ADR-0014) on the one CUDA image, the preset's config inline
//!   (`FV_WORKER_TOML_B64`: the image bakes none, ADR-0011), volume weights
//!   on the EU volume.
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
/// The largest `FV_WORKER_TOML_B64` a launch sends (bytes of base64). Runpod
/// documents no limit on one env value (only 50 variables per pod, and a
/// 1 MiB body on the v2 API); this is a conservative guard, about three
/// times the largest committed config (`runpod-family-h3.toml`, 5600 bytes
/// encoded), and fv-control's `MAX_WORKER_TOML_B64` is the same.
pub const MAX_WORKER_TOML_B64: usize = 16 * 1024;
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

/// The resolved preset: image, config, trees, disk.
struct Resolved {
    image: String,
    config: String,
    trees: Vec<String>,
    container_disk_gb: Option<u32>,
}

fn resolve_preset(repo: &Repo, family: &str, preset: &str) -> Result<Resolved, LaunchError> {
    let d = &repo.deploy;
    let (image, config, variants, disk) = if let Some(p) = d.preset(preset) {
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
        image,
        config,
        trees,
        container_disk_gb: disk,
    })
}

/// The launch fv-deploy creates for a preset (docs/adr/0018-fv-deploy.md
/// §3.2, §8). `stamp` names it (`MMDDHHMMSS`). Runpod: a `[[presets]]` or
/// `[[family_presets]]` id on the EU volume; GCP: a `[[gcp]]` row
/// ([`gcp_launch`]); Vast: the fake-engine smoke only ([`vast_launch`]).
pub fn from_preset(repo: &Repo, req: &PresetRequest, stamp: &str) -> Result<Launch, LaunchError> {
    match req.kind {
        ProviderKind::RunpodPod | ProviderKind::RunpodServerless => runpod_launch(repo, req, stamp),
        ProviderKind::Vast => vast_launch(repo, req, stamp),
        ProviderKind::Gcp => gcp_launch(repo, req, stamp),
    }
}

fn runpod_launch(repo: &Repo, req: &PresetRequest, stamp: &str) -> Result<Launch, LaunchError> {
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
    if repo.deploy.image(&r.image).is_none() {
        return Err(LaunchError::Deploy(format!(
            "no image {} in deploy/deploy.toml",
            r.image
        )));
    }
    let (image, channel) = match &req.image {
        Some(i) => (i.clone(), None),
        None => (
            format!("{SERVE_REPO}:{}-{}", r.image, req.channel),
            Some(req.channel.as_str()),
        ),
    };
    let mut env = secret_env_refs();
    env.extend(image_ident_env(&image, channel));
    // The image bakes no config: the preset's rides inline (fv-boot refuses
    // a boot without one).
    let p = repo.root.join("configs/serve").join(&r.config);
    let (k, v) = inline_config(&read(&p)?)
        .map_err(|why| LaunchError::Preset(format!("{}: {why}", r.config)))?;
    env.insert(k, v);
    let args = Vec::new();
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
        image: r.image.clone(),
        config: r.config.clone(),
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
                    min_ram_gb: None,
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
                    min_ram_gb: None,
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
        ProviderKind::Vast | ProviderKind::Gcp => {
            return Err(LaunchError::Preset(format!(
                "{} is not a Runpod kind",
                req.kind
            )))
        }
    })
}

/// The preset id of the fake-engine smoke (`deploy.toml`'s `[[gcp]]` row
/// `fake`, the config the images carry: `runpod-fake.toml`); its family is
/// `fake` too.
pub const FAKE_PRESET: &str = "fake";
/// A Vast instance's disk (`vast.sh` `VAST_DISK_GB`).
pub const VAST_DISK_GB: u32 = 40;
/// Vast's published ports (`vast.sh env_dict`): HTTP, ICE-UDP and ICE-TCP,
/// symmetric.
pub const VAST_PORTS: [&str; 3] = ["8000/tcp", "70010/udp", "70000/tcp"];

/// The image of a launch outside Runpod's presets: `--image`, else the
/// deployment data's one GPU image on the channel.
fn channel_image(
    repo: &Repo,
    req: &PresetRequest,
) -> Result<(String, Option<String>), LaunchError> {
    if let Some(i) = &req.image {
        return Ok((i.clone(), None));
    }
    let img = repo
        .deploy
        .gpu_image()
        .ok_or_else(|| LaunchError::Deploy("no GPU image in deploy/deploy.toml".into()))?;
    Ok((
        format!("{SERVE_REPO}:{}-{}", img.id, req.channel),
        Some(req.channel.clone()),
    ))
}

fn read_config(repo: &Repo, config: &str) -> Result<(String, String), LaunchError> {
    let p = repo.root.join("configs/serve").join(config);
    inline_config(&read(&p)?).map_err(|why| LaunchError::Preset(format!("{config}: {why}")))
}

/// A Vast launch (docs/adr/0018-fv-deploy.md §8.2): what `vast.sh smoke`
/// rents, from fv-deploy. Vast has no weights volume and r2 weights are not
/// in fv-deploy yet, so the only preset is the fake-engine smoke
/// ([`FAKE_PRESET`], family `fake`): the CUDA image's channel tag, fv-boot
/// started by the onstart with the config inline (`FV_WORKER_TOML_B64`, the
/// image bakes none; ADR-0011), the script's env, ports and disk; any GPU
/// with a CUDA 13 driver unless `--gpu` names Vast GPU names. Secrets are
/// Vast account env vars and never in the launch.
pub fn vast_launch(repo: &Repo, req: &PresetRequest, stamp: &str) -> Result<Launch, LaunchError> {
    if req.preset != FAKE_PRESET || req.family != FAKE_PRESET {
        return Err(LaunchError::Preset(format!(
            "Vast: only --family {FAKE_PRESET} --preset {FAKE_PRESET} (the fake-engine smoke) launches: Vast has no weights volume and r2 weights are not in fv-deploy yet (docs/adr/0018-fv-deploy.md §8.2); got {} / {}",
            req.family, req.preset
        )));
    }
    let (image, channel) = channel_image(repo, req)?;
    let mut env = image_ident_env(&image, channel.as_deref());
    for (k, v) in [
        ("FV_SERVE_MODE", "http"),
        ("FV_STATE_DIR", "/fvstate"),
        ("FV_API_KEYS", req.key_hash.as_str()),
        ("FV_SERVE_FORWARD", "1"),
        ("RUST_LOG", "info"),
        ("FV_WEIGHTS_SOURCE", WeightsSource::None.as_str()),
    ] {
        env.insert(k.into(), v.into());
    }
    let config = fastvideo_manifest::deploy::SMOKE_CONFIG;
    let (k, v) = read_config(repo, config)?;
    env.insert(k, v);
    Ok(Launch {
        schema: LAUNCH_SCHEMA,
        name: format!("fv-deploy-{}-{stamp}", req.preset),
        provider: ProviderKind::Vast,
        source: Some(LaunchSource {
            family: req.family.clone(),
            preset: req.preset.clone(),
            image: repo
                .deploy
                .gpu_image()
                .map(|i| i.id.clone())
                .unwrap_or_default(),
            config: config.into(),
        }),
        image,
        entrypoint: vec![FV_BOOT_BIN.into()],
        args: Vec::new(),
        env,
        ports: VAST_PORTS.iter().map(|p| (*p).to_owned()).collect(),
        gpu: GpuSpec {
            types: req.gpu_types.clone(),
            count: 1,
            allowed_cuda: Vec::new(),
            min_cuda: None,
            min_ram_gb: None,
        },
        placement: Placement {
            region: None,
            data_centers: Vec::new(),
            volume: None,
            cloud: CloudType::Community,
        },
        disk: Disk {
            container_gb: VAST_DISK_GB,
            volume_gb: 0,
        },
        weights: WeightsSource::None,
        guards: req.guards,
        serverless: None,
    })
}

/// What `scripts/serve/vast.sh` creates (its `create_payload` inputs): the
/// onstart starts fv-boot with `--config <config>` (a file in the image),
/// its env object, no image identity, the script's offer filter.
pub struct ScriptVast<'a> {
    pub name: &'a str,
    pub image: &'a str,
    pub key_hash: &'a str,
    /// `FV_SERVE_CONFIG` (default `/etc/fv/runpod-fake.toml`).
    pub config: &'a str,
    pub disk_gb: u32,
    /// `VAST_MAX_DPH` in the guards' price cap.
    pub guards: Guards,
}

pub fn script_vast(p: &ScriptVast<'_>) -> Launch {
    let mut env = BTreeMap::new();
    for (k, v) in [
        ("FV_SERVE_MODE", "http"),
        ("FV_STATE_DIR", "/fvstate"),
        ("FV_API_KEYS", p.key_hash),
        ("FV_SERVE_FORWARD", "1"),
        ("RUST_LOG", "info"),
    ] {
        env.insert(k.to_owned(), v.to_owned());
    }
    Launch {
        schema: LAUNCH_SCHEMA,
        name: p.name.into(),
        provider: ProviderKind::Vast,
        source: None,
        image: p.image.into(),
        entrypoint: vec![FV_BOOT_BIN.into()],
        args: vec!["--config".into(), p.config.into()],
        env,
        ports: VAST_PORTS.iter().map(|x| (*x).to_owned()).collect(),
        gpu: GpuSpec {
            types: Vec::new(),
            count: 1,
            allowed_cuda: Vec::new(),
            min_cuda: None,
            min_ram_gb: None,
        },
        placement: Placement {
            region: None,
            data_centers: Vec::new(),
            volume: None,
            cloud: CloudType::Community,
        },
        disk: Disk {
            container_gb: p.disk_gb,
            volume_gb: 0,
        },
        weights: WeightsSource::None,
        guards: p.guards,
        serverless: None,
    }
}

/// The `[[gcp]]` row of a request: its name is the preset; its dispatch
/// family must be the request's family (a model family through
/// `[dispatch]`, or the dispatch family itself); the `fake` row's family is
/// `fake`.
fn gcp_row<'a>(
    repo: &'a Repo,
    req: &PresetRequest,
) -> Result<&'a fastvideo_manifest::deploy::GcpRow, LaunchError> {
    let d = &repo.deploy;
    let row = d.gcp.iter().find(|r| r.name == req.preset).ok_or_else(|| {
        LaunchError::Preset(format!(
            "no [[gcp]] row {} in deploy/deploy.toml (rows: {})",
            req.preset,
            d.gcp
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;
    let ok = match &row.dispatch_family {
        Some(df) => &req.family == df || d.dispatch.get(&req.family) == Some(df),
        None => req.family == FAKE_PRESET,
    };
    if !ok {
        return Err(LaunchError::Preset(format!(
            "[[gcp]] row {} serves dispatch family {}, not {}",
            row.name,
            row.dispatch_family.as_deref().unwrap_or(FAKE_PRESET),
            req.family
        )));
    }
    Ok(row)
}

/// A GCP launch (docs/adr/0018-fv-deploy.md §8.3): what `vm.sh up <row>`
/// creates, from fv-deploy: the `[[gcp]]` row's config inline (it becomes
/// the `fv-config` metadata item) and machine type (or `--gpu` machine
/// types, tried in order), the zone of the region (`eu`:
/// `europe-west4-b`), the read-only weights disk of that zone (none for the
/// `fake` row), the script's container env plus the image identity, the
/// channel and the weights source. The VM's own settings are
/// [`gcp_options`].
pub fn gcp_launch(repo: &Repo, req: &PresetRequest, stamp: &str) -> Result<Launch, LaunchError> {
    let row = gcp_row(repo, req)?;
    let zone = gcp_zone(&req.region)?;
    let (image, channel) = channel_image(repo, req)?;
    let fake = row.name == FAKE_PRESET;
    let weights = if fake {
        WeightsSource::None
    } else {
        WeightsSource::Volume
    };
    let mut env = gcp_env(&image, &req.key_hash);
    if let Some(c) = channel {
        env.insert("FV_RELEASE_CHANNEL".into(), c);
    }
    env.insert("FV_WEIGHTS_SOURCE".into(), weights.as_str().into());
    let (k, v) = read_config(repo, &row.config)?;
    env.insert(k, v);
    let machines = if req.gpu_types.is_empty() {
        vec![row.machine.clone()]
    } else {
        req.gpu_types.clone()
    };
    Ok(Launch {
        schema: LAUNCH_SCHEMA,
        name: format!("fv-deploy-{}-{stamp}", req.preset),
        provider: ProviderKind::Gcp,
        source: Some(LaunchSource {
            family: req.family.clone(),
            preset: row.name.clone(),
            image: repo
                .deploy
                .gpu_image()
                .map(|i| i.id.clone())
                .unwrap_or_default(),
            config: row.config.clone(),
        }),
        image,
        entrypoint: Vec::new(),
        args: Vec::new(),
        env,
        ports: GCP_PORTS.iter().map(|p| (*p).to_owned()).collect(),
        gpu: GpuSpec {
            types: machines,
            count: 1,
            allowed_cuda: Vec::new(),
            min_cuda: None,
            min_ram_gb: None,
        },
        placement: gcp_placement(&zone, !fake, Some(req.region.clone())),
        disk: Disk {
            container_gb: crate::provider::gcp::BOOT_DISK_GB,
            volume_gb: 0,
        },
        weights,
        guards: req.guards,
        serverless: None,
    })
}

/// The ports a GCP VM's firewall rule opens (informational: the rule is
/// `crate::provider::gcp::firewall_body`).
pub const GCP_PORTS: [&str; 3] = ["8000/tcp", "40000/tcp", "40010/udp"];

/// `eu` → `europe-west4-b` (`scripts/gcp/lib.sh`); the US is refused as for
/// Runpod (EU only, CLAUDE.md).
pub fn gcp_zone(region: &str) -> Result<String, LaunchError> {
    match region {
        EU_REGION => Ok(crate::provider::gcp::DEFAULT_ZONE.into()),
        "us" => Err(LaunchError::Region(guard::US_GONE.into())),
        other => Err(LaunchError::Region(format!("unknown region {other} (eu)"))),
    }
}

fn gcp_placement(zone: &str, disk: bool, region: Option<String>) -> Placement {
    Placement {
        region,
        data_centers: vec![zone.to_owned()],
        volume: disk.then(|| VolumeMount {
            id: crate::provider::gcp::weights_disk(zone),
            mount_path: POD_VOLUME_MOUNT.into(),
        }),
        cloud: CloudType::Secure,
    }
}

/// `vm.sh env_json` for the serve role.
fn gcp_env(image: &str, key_hash: &str) -> BTreeMap<String, String> {
    let mut env = image_ident_env(image, None);
    for (k, v) in [
        ("FV_SERVE_MODE", "http"),
        ("FV_WEIGHTS", "/workspace/weights"),
        ("FV_STATE_DIR", "/fvstate"),
        ("RUST_LOG", "info"),
        ("FV_GCP_UDP_PORTS", crate::provider::gcp::UDP_PORTS),
        ("FV_API_KEYS", key_hash),
        ("FV_SERVE_FORWARD", "1"),
    ] {
        env.insert(k.into(), v.into());
    }
    env
}

/// The VM's own settings for a `[[gcp]]` launch: the row's labels and
/// verify cells, the run stamp, and what the command line chose.
pub fn gcp_options(
    repo: &Repo,
    req: &PresetRequest,
    stamp: &str,
    source_ranges: Vec<String>,
    spot: bool,
    idle_s: u64,
    secret_manager: bool,
) -> Result<crate::provider::gcp::GcpOptions, LaunchError> {
    let row = gcp_row(repo, req)?;
    Ok(crate::provider::gcp::GcpOptions {
        family: row.name.clone(),
        run: stamp.to_owned(),
        verify_cells: row.verify_cells.clone(),
        spot,
        idle_s,
        source_ranges,
        secret_manager,
    })
}

/// What `scripts/gcp/vm.sh up <family>` creates (`instance_payload` inputs
/// for the serve role, no secrets set): the row's config and machine, the
/// weights disk unless `fake`, the script's env.
pub struct ScriptGcp<'a> {
    pub name: &'a str,
    pub image: &'a str,
    pub key_hash: &'a str,
    /// `configs/serve/<row config>`'s text.
    pub config_text: &'a str,
    pub machine: &'a str,
    pub zone: &'a str,
    pub fake: bool,
    /// `FV_GCP_CAP_S` in `cap_s`.
    pub guards: Guards,
}

pub fn script_gcp(p: &ScriptGcp<'_>) -> Launch {
    let mut env = gcp_env(p.image, p.key_hash);
    env.insert(
        "FV_WORKER_TOML_B64".into(),
        base64::engine::general_purpose::STANDARD.encode(p.config_text.as_bytes()),
    );
    Launch {
        schema: LAUNCH_SCHEMA,
        name: p.name.into(),
        provider: ProviderKind::Gcp,
        source: None,
        image: p.image.into(),
        entrypoint: Vec::new(),
        args: Vec::new(),
        env,
        ports: GCP_PORTS.iter().map(|x| (*x).to_owned()).collect(),
        gpu: GpuSpec {
            types: vec![p.machine.into()],
            count: 1,
            allowed_cuda: Vec::new(),
            min_cuda: None,
            min_ram_gb: None,
        },
        placement: gcp_placement(p.zone, !p.fake, None),
        disk: Disk {
            container_gb: crate::provider::gcp::BOOT_DISK_GB,
            volume_gb: 0,
        },
        weights: if p.fake {
            WeightsSource::None
        } else {
            WeightsSource::Volume
        },
        guards: p.guards,
        serverless: None,
    }
}

/// `FV_WORKER_TOML_B64` for a worker config's text, or why it is too large
/// for a launch's env ([`MAX_WORKER_TOML_B64`]).
pub fn inline_config(toml: &str) -> Result<(String, String), String> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(toml.as_bytes());
    if b64.len() > MAX_WORKER_TOML_B64 {
        return Err(format!(
            "the worker config is {} bytes as FV_WORKER_TOML_B64, over the {MAX_WORKER_TOML_B64}-byte guard",
            b64.len()
        ));
    }
    Ok(("FV_WORKER_TOML_B64".to_owned(), b64))
}

/// The env a script launch sends its config in (`variants.sh
/// fv_config_env_json`): the TOML inline when the script read it from
/// `configs/serve`, else `FV_WORKER_CONFIG` naming the file the image
/// carries (`/etc/fv/runpod-fake.toml`).
fn script_config_env(config: &str, inline: Option<&str>) -> (String, String) {
    match inline {
        Some(t) => (
            "FV_WORKER_TOML_B64".to_owned(),
            base64::engine::general_purpose::STANDARD.encode(t.as_bytes()),
        ),
        None => ("FV_WORKER_CONFIG".to_owned(), config.to_owned()),
    }
}

/// What `scripts/serve/runpod-pod.sh` creates (its `payload()` inputs):
/// fv-boot (the image's entrypoint) with the config in the env, the run
/// key's hash, a volume at `/workspace` in its data centre, no CUDA filter.
pub struct ScriptPod<'a> {
    pub name: &'a str,
    pub image: &'a str,
    pub gpu: &'a str,
    pub key_hash: &'a str,
    /// `FV_SERVE_CONFIG` as the script names it.
    pub config: &'a str,
    /// The config's text when the image does not carry it (the script sends
    /// it inline); `None`: `FV_WORKER_CONFIG` names `config` in the image.
    pub inline: Option<&'a str>,
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
    let (k, v) = script_config_env(p.config, p.inline);
    env.insert(k, v);
    Launch {
        schema: LAUNCH_SCHEMA,
        name: p.name.into(),
        provider: ProviderKind::RunpodPod,
        source: None,
        image: p.image.into(),
        entrypoint: vec![FV_BOOT_BIN.into()],
        args: Vec::new(),
        env,
        ports: vec!["8000/http".into(), "70000/tcp".into()],
        gpu: GpuSpec {
            types: vec![p.gpu.into()],
            count: 1,
            allowed_cuda: Vec::new(),
            min_cuda: None,
            min_ram_gb: None,
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

/// What `scripts/serve/runpod-endpoint.sh` creates for a queue endpoint
/// (`template_payload` + `endpoint_payload` inputs): fv-boot (it links the
/// volume's weights where a pod sees them) with the config in the env.
pub struct ScriptEndpoint<'a> {
    pub name: &'a str,
    pub image: &'a str,
    pub gpus: &'a [&'a str],
    /// `FV_SERVE_CONFIG` as the script names it.
    pub config: &'a str,
    /// The config's text when the image does not carry it (see [`ScriptPod::inline`]).
    pub inline: Option<&'a str>,
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
    let (k, v) = script_config_env(p.config, p.inline);
    env.insert(k, v);
    Launch {
        schema: LAUNCH_SCHEMA,
        name: p.name.into(),
        provider: ProviderKind::RunpodServerless,
        source: None,
        image: p.image.into(),
        entrypoint: vec![FV_BOOT_BIN.into()],
        args: Vec::new(),
        env,
        ports: Vec::new(),
        gpu: GpuSpec {
            types: p.gpus.iter().map(|g| (*g).to_owned()).collect(),
            count: 1,
            allowed_cuda: Vec::new(),
            min_cuda: Some(MIN_CUDA.into()),
            min_ram_gb: None,
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
