//! A Runpod pod created from a template (REST v1 `POST /pods` with a
//! `templateId`): the autoscaler's pod pools (`fastvideo-autoscale`
//! `RunpodPods`, docs/adr/0018-fv-deploy.md §9). Written here, beside
//! fv-deploy's own pod body ([`super::pod_create_body`]), so there is one
//! place that builds a Runpod pod create; the autoscaler sends it on its
//! async client.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use super::template_body;
use crate::launch::{image_ident_env, secret_env_refs, POD_VOLUME_MOUNT};
use crate::provider::{
    Disk, GpuSpec, Guards, Launch, Placement, ProviderKind, WeightsSource, LAUNCH_SCHEMA,
};

/// One pod of a pool: a template on one GPU type in one placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemplatePod<'a> {
    pub name: &'a str,
    pub template_id: &'a str,
    /// `SECURE` / `COMMUNITY`.
    pub cloud_type: &'a str,
    pub gpu: &'a str,
    pub volume_id: &'a str,
    pub data_center: &'a str,
    /// The pod's `env`, when it sends one ([`with_worker_config`]); `None`:
    /// the template's env applies as it is.
    pub env: Option<&'a Map<String, Value>>,
}

/// The `POST /pods` body: one GPU, the network volume at `/workspace` in its
/// data centre, no `allowedCudaVersions` (REST cannot name the 13.1 / 13.2
/// hosts, `scripts/gpu/runpod-cuda.sh`).
pub fn template_pod_body(p: &TemplatePod<'_>) -> Value {
    let mut body = json!({
        "name": p.name,
        "templateId": p.template_id,
        "computeType": "GPU",
        "cloudType": p.cloud_type,
        "gpuTypeIds": [p.gpu],
        "gpuCount": 1,
        "networkVolumeId": p.volume_id,
        "volumeMountPath": "/workspace",
        "dataCenterIds": [p.data_center],
    });
    if let Some(env) = p.env {
        body["env"] = Value::Object(env.clone());
    }
    body
}

/// The `env` of a REST v1 template object, as strings (a non-string value
/// is skipped; an absent `env` is empty).
pub fn template_env_of(v: &Value) -> Map<String, Value> {
    v.get("env")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter(|(_, x)| x.is_string())
                .map(|(k, x)| (k.clone(), x.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// The pod env that sends a worker config (ADR-0011: the image bakes none):
/// the template's own env (the secret references, the image identity) with
/// `FV_WORKER_TOML_B64` on top. REST v1 `POST /pods` documents no rule for a
/// body `env` next to a `templateId` (the v2 create documents a per-key
/// merge, body values winning), so the body carries the template's env too:
/// the same pod whichever rule v1 applies.
pub fn with_worker_config(template_env: &Map<String, Value>, b64: &str) -> Map<String, Value> {
    let mut env = template_env.clone();
    env.insert("FV_WORKER_TOML_B64".into(), Value::String(b64.to_owned()));
    env
}

// ------------------------------------------------------------- serve templates

/// A serve image's Runpod template kind: the serverless queue worker or the
/// HTTP pod (`runpod-templates.sh` `fv_image_flavours`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavour {
    Sls,
    Pod,
}

impl Flavour {
    pub fn as_str(self) -> &'static str {
        match self {
            Flavour::Sls => "sls",
            Flavour::Pod => "pod",
        }
    }

    pub fn parse(s: &str) -> Option<Flavour> {
        match s {
            "sls" => Some(Flavour::Sls),
            "pod" => Some(Flavour::Pod),
            _ => None,
        }
    }

    /// A GPU image has both; the CPU image a pod template only.
    pub fn of(cpu: bool) -> &'static [Flavour] {
        if cpu {
            &[Flavour::Pod]
        } else {
            &[Flavour::Sls, Flavour::Pod]
        }
    }
}

/// `fv-serve-<image>-<flavour>` (`fv_template_name`).
pub fn serve_template_name(image_id: &str, f: Flavour) -> String {
    format!("fv-serve-{image_id}-{}", f.as_str())
}

/// The serve template of one image as a [`Launch`] (what
/// `runpod-templates.sh` publishes): the image (a digest), the secret
/// references, its identity env (`FV_IMAGE_REF`, `FV_IMAGE_DIGEST`, the
/// channel when given) and the flavour's serve env; the queue worker's disk
/// 20 GB, the pod's 30 GB with port 8000 (and 70000/tcp on a GPU image). No
/// entrypoint (the image's: fv-boot) and no worker config (a launch from the
/// template sends one, ADR-0011). A template creates nothing, so its guards
/// are "not checked" (`null`) but the floor; a CPU image is a launch with no
/// GPU (`count` 0), never sent as a create (§7.3: CPU is outside Launch v1).
pub fn serve_template_launch(
    image_id: &str,
    f: Flavour,
    cpu: bool,
    image: &str,
    channel: Option<&str>,
) -> Launch {
    let mut env: BTreeMap<String, String> = secret_env_refs();
    env.extend(image_ident_env(image, channel));
    let kv = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    };
    match f {
        Flavour::Sls => env.extend(kv(&[
            ("FV_SERVE_MODE", "runpod-queue"),
            ("FV_AUTH_MODE", "trust-gateway"),
            ("FV_STATE_DIR", "/fvstate"),
            ("FV_WEIGHTS", "/runpod-volume/weights"),
            ("FV_CACHE_DIR", "/fvstate/cache"),
            ("RUST_LOG", "info"),
        ])),
        Flavour::Pod => {
            env.extend(kv(&[
                ("FV_SERVE_MODE", "http"),
                ("FV_STATE_DIR", "/fvstate"),
                ("RUST_LOG", "info"),
            ]));
            if !cpu {
                env.extend(kv(&[
                    ("FV_WEIGHTS", "/workspace/weights"),
                    ("FV_SERVE_FORWARD", "1"),
                ]));
            }
        }
    }
    let ports = match (f, cpu) {
        (Flavour::Sls, _) => vec![],
        (Flavour::Pod, true) => vec!["8000/http".to_owned()],
        (Flavour::Pod, false) => vec!["8000/http".to_owned(), "70000/tcp".to_owned()],
    };
    Launch {
        schema: LAUNCH_SCHEMA,
        name: serve_template_name(image_id, f),
        provider: match f {
            Flavour::Sls => ProviderKind::RunpodServerless,
            Flavour::Pod => ProviderKind::RunpodPod,
        },
        source: None,
        image: image.to_owned(),
        entrypoint: Vec::new(),
        args: Vec::new(),
        env,
        ports,
        gpu: GpuSpec {
            types: Vec::new(),
            count: u32::from(!cpu),
            allowed_cuda: Vec::new(),
            min_cuda: None,
            min_ram_gb: None,
        },
        placement: Placement {
            region: None,
            data_centers: Vec::new(),
            volume: None,
            cloud: Default::default(),
        },
        disk: Disk {
            container_gb: match f {
                Flavour::Sls => 20,
                Flavour::Pod => 30,
            },
            volume_gb: 0,
        },
        weights: if cpu {
            WeightsSource::None
        } else {
            WeightsSource::Volume
        },
        guards: Guards {
            max_usd_per_hr: None,
            min_balance_usd: crate::guard::MIN_BALANCE_USD,
            cap_s: None,
            ready_timeout_s: None,
            ready_cycles: None,
            force: false,
        },
        serverless: None,
    }
}

/// The REST v1 body of a serve template (`runpod-templates.sh payload`): the
/// serverless template body of its launch ([`template_body`]) with the
/// template's own fields: the readme, a pod's volume mount and ports, the
/// registry auth when set; `create` adds what the update API does not take
/// (`isServerless`, `category`).
pub fn serve_template_body(
    l: &Launch,
    image_id: &str,
    f: Flavour,
    cpu: bool,
    registry_auth: Option<&str>,
    create: bool,
) -> Value {
    let mut b = template_body(l);
    if let Value::Object(m) = &mut b {
        m.remove("isServerless");
        m.insert(
            "readme".into(),
            json!(format!(
                "fv-serve {image_id} ({}), published by CI; no worker config: a launch sends one (FV_WORKER_TOML_B64); docs/serve/images.md",
                f.as_str()
            )),
        );
        if f == Flavour::Pod {
            m.insert("volumeMountPath".into(), json!(POD_VOLUME_MOUNT));
            m.insert("ports".into(), json!(l.ports));
        }
        if let Some(a) = registry_auth.filter(|a| !a.is_empty()) {
            m.insert("containerRegistryAuthId".into(), json!(a));
        }
        if create {
            m.insert("isServerless".into(), json!(f == Flavour::Sls));
            m.insert("category".into(), json!(if cpu { "CPU" } else { "NVIDIA" }));
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_template_pod_sends_the_template_env_with_the_config_on_top() {
        let tpl = template_env_of(&json!({"env": {
            "FV_CF_ACCOUNT_ID": "{{ RUNPOD_SECRET_fv_cf_account_id }}",
            "FV_WORKER_TOML_B64": "b2xk",
            "N": 3
        }}));
        let env = with_worker_config(&tpl, "W3NlcnZlcl0K");
        let b = template_pod_body(&TemplatePod {
            name: "fv-as-x-1",
            template_id: "tpl",
            cloud_type: "SECURE",
            gpu: "NVIDIA H200",
            volume_id: "bz0ofkfx45",
            data_center: "EU-RO-1",
            env: Some(&env),
        });
        assert_eq!(b["env"]["FV_WORKER_TOML_B64"], "W3NlcnZlcl0K");
        assert_eq!(
            b["env"]["FV_CF_ACCOUNT_ID"],
            "{{ RUNPOD_SECRET_fv_cf_account_id }}"
        );
        assert!(b["env"].get("N").is_none());
        assert_eq!(b["gpuTypeIds"], json!(["NVIDIA H200"]));
        assert!(b.get("allowedCudaVersions").is_none());
        let bare = template_pod_body(&TemplatePod {
            name: "n",
            template_id: "t",
            cloud_type: "SECURE",
            gpu: "g",
            volume_id: "v",
            data_center: "d",
            env: None,
        });
        assert!(bare.get("env").is_none());
    }
}
