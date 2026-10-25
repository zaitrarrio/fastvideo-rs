//! A Runpod pod created from a template (REST v1 `POST /pods` with a
//! `templateId`): the autoscaler's pod pools (`fastvideo-autoscale`
//! `RunpodPods`, docs/adr/0018-fv-deploy.md §9). Written here, beside
//! fv-deploy's own pod body ([`super::pod_create_body`]), so there is one
//! place that builds a Runpod pod create; the autoscaler sends it on its
//! async client.

use serde_json::{json, Map, Value};

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
