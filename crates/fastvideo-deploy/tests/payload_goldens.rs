//! The payload goldens (docs/adr/0018-fv-deploy.md §4): the Runpod provider
//! code builds exactly what `scripts/serve/runpod-pod.sh` and
//! `scripts/serve/runpod-endpoint.sh` send for the same inputs
//! (`fixtures/deploy/*.json`, written by those scripts' own `plan` through
//! `scripts/serve/tests/fv-deploy-goldens.sh`), and a preset launch differs
//! from the script pod only where ADR-0018 says it does.

use std::path::{Path, PathBuf};

use base64::Engine as _;
use serde_json::{json, Value};

use fastvideo_deploy::guard;
use fastvideo_deploy::launch::{
    self, from_preset, pin_image, script_endpoint, script_pod, PresetRequest, Repo, ScriptEndpoint,
    ScriptPod, KEY_HASH_PLACEHOLDER,
};
use fastvideo_deploy::provider::runpod::{endpoint_body, pod_create_body, template_body};
use fastvideo_deploy::provider::ProviderKind;

// The inputs of fv-deploy-goldens.sh.
const IMAGE: &str =
    "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:1111111111111111111111111111111111111111111111111111111111111111";
const GPU: &str = "NVIDIA RTX PRO 6000 Blackwell Server Edition";
const CONFIG: &str = "/etc/fv/runpod.toml";

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    crate_root().join("../..")
}

fn fixture(name: &str) -> Value {
    let p = crate_root().join("fixtures/deploy").join(name);
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&t).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn script_pod_launch() -> fastvideo_deploy::provider::Launch {
    script_pod(&ScriptPod {
        name: "fv-serve-plan",
        image: IMAGE,
        gpu: GPU,
        key_hash: KEY_HASH_PLACEHOLDER,
        config: CONFIG,
        volume: Some(("bz0ofkfx45", "EU-RO-1")),
        guards: guard::default_guards(),
    })
}

#[test]
fn pod_create_equals_runpod_pod_sh() {
    assert_eq!(
        pod_create_body(&script_pod_launch()),
        fixture("runpod-pod.json")
    );
}

#[test]
fn template_and_endpoint_equal_runpod_endpoint_sh() {
    let l = script_endpoint(&ScriptEndpoint {
        name: "fv-serve-plan",
        image: IMAGE,
        gpus: &[GPU],
        config: CONFIG,
        volume: "bz0ofkfx45",
        data_center: "EU-RO-1",
        guards: guard::default_guards(),
    });
    assert_eq!(template_body(&l), fixture("runpod-template.json"));
    assert_eq!(
        endpoint_body(&l, "<template id>"),
        fixture("runpod-endpoint.json")
    );
    assert_eq!(guard::check_region(&l), Ok(()));
}

fn request(preset: &str, kind: ProviderKind) -> PresetRequest {
    PresetRequest {
        family: "h3".into(),
        preset: preset.into(),
        region: "eu".into(),
        kind,
        image: None,
        channel: "stable".into(),
        gpu_types: Vec::new(),
        volume: None,
        guards: guard::default_guards(),
        key_hash: KEY_HASH_PLACEHOLDER.into(),
    }
}

fn repo() -> Repo {
    Repo::load(&repo_root()).unwrap_or_else(|e| panic!("{e}"))
}

/// `fv-deploy up --family h3 --preset h3-turbo --region eu` (the live gate):
/// the script pod with fv-boot as the entrypoint, the volume weights source,
/// the channel, fv-control's 40 GB container disk; nothing else differs.
#[test]
fn preset_pod_is_the_script_pod_plus_fv_boot() {
    let r = repo();
    let mut l = from_preset(
        &r,
        &request("h3-turbo", ProviderKind::RunpodPod),
        "1024120000",
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        l.image,
        "ghcr.io/zaitrarrio/fastvideo-rs-serve:h3-turbo-stable"
    );
    assert_eq!(guard::check_region(&l), Ok(()));
    pin_image(&mut l, IMAGE);
    let got = pod_create_body(&l);
    let mut want = fixture("runpod-pod.json");
    want["name"] = json!("fv-deploy-h3-turbo-1024120000");
    want["dockerEntrypoint"] = json!(["/opt/fastvideo-rs/bin/fv-boot"]);
    want["containerDiskInGb"] = json!(40);
    want["env"]["FV_WEIGHTS_SOURCE"] = json!("volume");
    want["env"]["FV_RELEASE_CHANNEL"] = json!("stable");
    assert_eq!(got, want);
    let src = l.source.unwrap_or_else(|| panic!("no source"));
    assert_eq!(
        (
            src.image_variant.as_str(),
            src.config.as_str(),
            src.config_baked
        ),
        ("h3-turbo", "runpod.toml", true)
    );
}

/// A preset serverless endpoint: the script's template with fv-boot as the
/// entrypoint (it links the volume itself) and the endpoint unchanged.
#[test]
fn preset_endpoint_is_the_script_endpoint_plus_fv_boot() {
    let r = repo();
    let mut l = from_preset(
        &r,
        &request("h3-turbo", ProviderKind::RunpodServerless),
        "1024120000",
    )
    .unwrap_or_else(|e| panic!("{e}"));
    pin_image(&mut l, IMAGE);
    let mut want_t = fixture("runpod-template.json");
    want_t["name"] = json!("fv-deploy-h3-turbo-1024120000");
    want_t["dockerEntrypoint"] = json!(["/opt/fastvideo-rs/bin/fv-boot"]);
    want_t["env"]["FV_WEIGHTS_SOURCE"] = json!("volume");
    want_t["env"]["FV_RELEASE_CHANNEL"] = json!("stable");
    assert_eq!(template_body(&l), want_t);
    let mut want_e = fixture("runpod-endpoint.json");
    want_e["name"] = json!("fv-deploy-h3-turbo-1024120000");
    assert_eq!(endpoint_body(&l, "<template id>"), want_e);
}

/// Every preset either builds (its trees are on the EU volume; its config
/// baked as `--config`, or inline as `FV_WORKER_TOML_B64` equal to the
/// committed file) or is refused for its trees; H3 and LTX presets build.
#[test]
fn every_preset_builds_or_is_refused_for_its_trees() {
    let r = repo();
    let mut built = Vec::new();
    for p in &r.deploy.presets {
        let fam = r
            .deploy
            .preset_variants(&r.families, p)
            .first()
            .map(|(f, _)| f.family.id.as_str().to_owned())
            .unwrap_or_default();
        let mut req = request(&p.id, ProviderKind::RunpodPod);
        req.family = fam.clone();
        match from_preset(&r, &req, "1") {
            Ok(l) => {
                assert_eq!(guard::check_region(&l), Ok(()), "{}", p.id);
                let src = l.source.clone().unwrap_or_else(|| panic!("no source"));
                if src.config_baked {
                    assert_eq!(
                        l.args,
                        ["--config".to_owned(), format!("/etc/fv/{}", p.config)]
                    );
                    assert!(!l.env.contains_key("FV_WORKER_TOML_B64"));
                } else {
                    assert!(l.args.is_empty(), "{}", p.id);
                    let b64 = l.env.get("FV_WORKER_TOML_B64").cloned().unwrap_or_default();
                    let text = base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .unwrap_or_default();
                    let file = std::fs::read(repo_root().join("configs/serve").join(&p.config))
                        .unwrap_or_default();
                    assert_eq!(text, file, "{}", p.id);
                }
                built.push(p.id.clone());
            }
            Err(e) => assert!(
                e.to_string().contains("not on the EU volume"),
                "{}: {e}",
                p.id
            ),
        }
    }
    for want in ["h3-turbo", "h3-max", "ltx", "ltx-pro"] {
        assert!(
            built.iter().any(|b| b == want),
            "{want} did not build: {built:?}"
        );
    }
    assert!(
        !built.iter().any(|b| b == "wan"),
        "wan is not on the EU volume: {built:?}"
    );
}

#[test]
fn wrong_family_unknown_preset_and_us_are_refused() {
    let r = repo();
    let mut req = request("ltx", ProviderKind::RunpodPod);
    let e = from_preset(&r, &req, "1")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("not h3"), "{e}");
    req.preset = "no-such".into();
    let e = from_preset(&r, &req, "1")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        e.contains("no preset no-such") && e.contains("h3-turbo"),
        "{e}"
    );
    let mut us = request("h3-turbo", ProviderKind::RunpodPod);
    us.region = "us".into();
    let e = from_preset(&r, &us, "1")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("US weights volume deleted"), "{e}");
    // A volume override is built, then refused by the region guard.
    let mut vol = request("h3-turbo", ProviderKind::RunpodPod);
    vol.volume = Some("s2k01690bi".into());
    let l = from_preset(&r, &vol, "1").unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        guard::check_region(&l),
        Err(guard::GuardError::UsGone(_))
    ));
}

#[test]
fn eu_trees_are_read_from_volumes_sh() {
    let t = std::fs::read_to_string(repo_root().join("scripts/gpu/volumes.sh")).unwrap_or_default();
    let trees = launch::parse_eu_trees(&t);
    assert!(trees.iter().any(|x| x == "h3-base"), "{trees:?}");
    assert!(trees.iter().any(|x| x == "ltx25"), "{trees:?}");
    assert!(trees.iter().any(|x| x == "auxiliary/lpips"), "{trees:?}");
    assert!(Path::new(&repo_root()).join("deploy/deploy.toml").is_file());
    assert_eq!(
        Repo::find_root(&crate_root()).map(|p| p.join("families").is_dir()),
        Some(true)
    );
}

#[test]
fn stamps_and_keys() {
    assert_eq!(launch::utc_stamp(1_790_553_600.0), "0928000000");
    assert_eq!(launch::utc_iso(1_790_553_601.0), "2026-09-28T00:00:01Z");
    assert_eq!(launch::utc_iso(951_782_400.0), "2000-02-29T00:00:00Z");
    let k = launch::run_key().unwrap_or_default();
    assert!(k.starts_with("fvk-") && k.len() == 36, "{k}");
    assert_eq!(
        launch::key_hash("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
