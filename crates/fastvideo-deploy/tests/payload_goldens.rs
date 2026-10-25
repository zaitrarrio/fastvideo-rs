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

/// The text the scripts send inline for `CONFIG` (the image carries only
/// `runpod-fake.toml`): `configs/serve/runpod.toml`.
fn config_text() -> String {
    let p = repo_root().join("configs/serve/runpod.toml");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn script_pod_launch() -> fastvideo_deploy::provider::Launch {
    let text = config_text();
    script_pod(&ScriptPod {
        name: "fv-serve-plan",
        image: IMAGE,
        gpu: GPU,
        key_hash: KEY_HASH_PLACEHOLDER,
        config: CONFIG,
        inline: Some(&text),
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
    let text = config_text();
    let l = script_endpoint(&ScriptEndpoint {
        name: "fv-serve-plan",
        image: IMAGE,
        gpus: &[GPU],
        config: CONFIG,
        inline: Some(&text),
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
/// the script pod (fv-boot, the config inline: the one image bakes none) on
/// the CUDA image's channel tag, with the volume weights source, the channel
/// and fv-control's 40 GB container disk; nothing else differs.
#[test]
fn preset_pod_is_the_script_pod_plus_fv_boot() {
    let r = repo();
    let mut l = from_preset(
        &r,
        &request("h3-turbo", ProviderKind::RunpodPod),
        "1024120000",
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(l.image, "ghcr.io/zaitrarrio/fastvideo-rs-serve:cuda-stable");
    assert_eq!(guard::check_region(&l), Ok(()));
    pin_image(&mut l, IMAGE);
    let got = pod_create_body(&l);
    let mut want = fixture("runpod-pod.json");
    want["name"] = json!("fv-deploy-h3-turbo-1024120000");
    want["containerDiskInGb"] = json!(40);
    want["env"]["FV_WEIGHTS_SOURCE"] = json!("volume");
    want["env"]["FV_RELEASE_CHANNEL"] = json!("stable");
    assert_eq!(got, want);
    let src = l.source.unwrap_or_else(|| panic!("no source"));
    assert_eq!(
        (src.image.as_str(), src.config.as_str()),
        ("cuda", "runpod.toml")
    );
}

/// A preset serverless endpoint: the script's template (fv-boot links the
/// volume itself, the config inline) and the endpoint unchanged.
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
    want_t["env"]["FV_WEIGHTS_SOURCE"] = json!("volume");
    want_t["env"]["FV_RELEASE_CHANNEL"] = json!("stable");
    assert_eq!(template_body(&l), want_t);
    let mut want_e = fixture("runpod-endpoint.json");
    want_e["name"] = json!("fv-deploy-h3-turbo-1024120000");
    assert_eq!(endpoint_body(&l, "<template id>"), want_e);
}

/// Every preset either builds (its trees are on the EU volume; its config
/// inline as `FV_WORKER_TOML_B64` equal to the committed file, under the
/// env size guard, never as `--config`) or is refused for its trees; H3 and
/// LTX presets build.
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
                assert_eq!(src.image, "cuda", "{}", p.id);
                assert!(l.args.is_empty(), "{}", p.id);
                assert!(!l.env.contains_key("FV_WORKER_CONFIG"), "{}", p.id);
                let b64 = l.env.get("FV_WORKER_TOML_B64").cloned().unwrap_or_default();
                assert!(b64.len() <= launch::MAX_WORKER_TOML_B64, "{}", p.id);
                let text = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .unwrap_or_default();
                let file = std::fs::read(repo_root().join("configs/serve").join(&p.config))
                    .unwrap_or_default();
                assert_eq!(text, file, "{}", p.id);
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

/// The template env size guard (ADR-0011 "What it needs" (3)): Runpod
/// documents no limit on one env value, so a launch refuses a config whose
/// `FV_WORKER_TOML_B64` exceeds `MAX_WORKER_TOML_B64`, and every committed
/// worker config fits under it with room (the largest, the H3 family's, is
/// about a third of it).
#[test]
fn every_worker_config_fits_the_env_size_guard() {
    let dir = repo_root().join("configs/serve");
    let mut largest = (0usize, String::new());
    for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.unwrap_or_else(|e| panic!("{e}")).path();
        if p.extension().is_none_or(|x| x != "toml") {
            continue;
        }
        let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let (k, v) =
            launch::inline_config(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert_eq!(k, "FV_WORKER_TOML_B64");
        if v.len() > largest.0 {
            largest = (v.len(), p.display().to_string());
        }
    }
    assert!(largest.0 * 2 < launch::MAX_WORKER_TOML_B64, "{largest:?}");
    let big = "#".repeat(launch::MAX_WORKER_TOML_B64);
    let e = launch::inline_config(&big).err().unwrap_or_default();
    assert!(e.contains("over the 16384-byte guard"), "{e}");
}

// ------------------------------------------------------------ Vast and GCP (C3b)

use fastvideo_deploy::launch::{script_gcp, script_vast, ScriptGcp, ScriptVast};
use fastvideo_deploy::provider::gcp::{
    firewall_body, instance_body, GcpAssets, GcpOptions, GcpPrices, FILE_ITEMS,
};
use fastvideo_deploy::provider::vast::{create_body as vast_create_body, offer_query};

/// Numbers compared as numbers: jq keeps the script's literal `13.0` /
/// `0.60`, serde_json reads `13` and `13.0` as different `Value`s.
fn num(v: &Value) -> Value {
    match v {
        Value::Number(n) => json!(n.as_f64().unwrap_or(f64::NAN)),
        Value::Array(a) => Value::Array(a.iter().map(num).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), num(v))).collect()),
        other => other.clone(),
    }
}

fn vast_guards() -> fastvideo_deploy::provider::Guards {
    let mut g = guard::default_guards();
    g.max_usd_per_hr = guard::default_max_usd_per_hr(ProviderKind::Vast);
    g
}

fn script_vast_launch() -> fastvideo_deploy::provider::Launch {
    script_vast(&ScriptVast {
        name: "fvgpu-serve-plan",
        image: IMAGE,
        key_hash: KEY_HASH_PLACEHOLDER,
        config: "/etc/fv/runpod-fake.toml",
        disk_gb: 40,
        guards: vast_guards(),
    })
}

#[test]
fn vast_search_and_create_equal_vast_sh() {
    let l = script_vast_launch();
    assert_eq!(num(&offer_query(&l)), num(&fixture("vast-offers.json")));
    let body = vast_create_body(&l).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(num(&body), num(&fixture("vast-create.json")));
}

fn vast_request() -> PresetRequest {
    let mut r = request("fake", ProviderKind::Vast);
    r.family = "fake".into();
    r.guards = vast_guards();
    r
}

/// `fv-deploy up --kind vast --family fake --preset fake`: the script's
/// instance with the config inline (the one image bakes none, ADR-0011), so
/// the onstart starts fv-boot without `--config`; plus the image identity,
/// the channel and the weights source; the search is the script's.
#[test]
fn preset_vast_is_the_script_instance_plus_the_inline_config() {
    let r = repo();
    let mut l = from_preset(&r, &vast_request(), "1025120000").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(guard::check_region(&l), Ok(()));
    pin_image(&mut l, IMAGE);
    assert_eq!(num(&offer_query(&l)), num(&fixture("vast-offers.json")));
    let got = vast_create_body(&l).unwrap_or_else(|e| panic!("{e}"));
    let mut want = fixture("vast-create.json");
    want["label"] = json!("fv-deploy-fake-1025120000");
    want["onstart"] = json!(want["onstart"]
        .as_str()
        .unwrap_or_default()
        .replace(" --config /etc/fv/runpod-fake.toml", ""));
    let fake =
        std::fs::read(repo_root().join("configs/serve/runpod-fake.toml")).unwrap_or_default();
    for (k, v) in [
        ("FV_IMAGE_REF", IMAGE.to_owned()),
        (
            "FV_IMAGE_DIGEST",
            IMAGE.split_once('@').map(|x| x.1).unwrap_or("").to_owned(),
        ),
        ("FV_RELEASE_CHANNEL", "stable".to_owned()),
        ("FV_WEIGHTS_SOURCE", "none".to_owned()),
        (
            "FV_WORKER_TOML_B64",
            base64::engine::general_purpose::STANDARD.encode(&fake),
        ),
    ] {
        want["env"][k] = json!(v);
    }
    assert_eq!(num(&got), num(&want));
}

#[test]
fn vast_refuses_presets_with_weights_and_secrets() {
    let r = repo();
    let mut req = vast_request();
    req.family = "h3".into();
    req.preset = "h3-turbo".into();
    let e = from_preset(&r, &req, "1")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("only --family fake --preset fake"), "{e}");
    let mut l = script_vast_launch();
    l.env.insert("FV_R2_SECRET_ACCESS_KEY".into(), "x".into());
    let e = vast_create_body(&l)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("Vast account env var"), "{e}");
    let mut l = script_vast_launch();
    l.weights = fastvideo_deploy::provider::WeightsSource::R2;
    assert!(guard::check_region(&l).is_err());
}

const GCP_RUN: &str = "1025120000";
const GCP_DEADLINE: u64 = 1_790_000_000;

fn gcp_options(family: &str, cells: &[&str]) -> GcpOptions {
    GcpOptions {
        family: family.into(),
        run: GCP_RUN.into(),
        verify_cells: cells.iter().map(|c| (*c).to_owned()).collect(),
        spot: false,
        idle_s: 1800,
        source_ranges: vec!["198.51.100.7/32".into()],
        secret_manager: false,
    }
}

/// The instance as the fixture holds it: the file items as `<file>`, after
/// checking each is the repository file it names; `fv-env` as its object.
fn gcp_normalized(body: &Value, assets: &GcpAssets, config: &str) -> Value {
    let mut b = body.clone();
    let files = [
        ("startup-script", assets.startup_script.as_str()),
        ("fv-config", config),
        ("fv-script-verify-weights", assets.verify_weights.as_str()),
        (
            "fv-script-verify-safetensors",
            assets.verify_safetensors.as_str(),
        ),
        ("fv-manifest", assets.manifest.as_str()),
        ("fv-sha256", assets.sha256.as_str()),
    ];
    for it in b["metadata"]["items"]
        .as_array_mut()
        .unwrap_or(&mut Vec::new())
    {
        let k = it["key"].as_str().unwrap_or_default().to_owned();
        if let Some((_, text)) = files.iter().find(|(f, _)| *f == k) {
            assert!(FILE_ITEMS.contains(&k.as_str()));
            assert_eq!(it["value"].as_str(), Some(*text), "{k}");
            assert!(!text.is_empty(), "{k}");
            it["value"] = json!("<file>");
        } else if k == "fv-env" {
            let env: Value =
                serde_json::from_str(it["value"].as_str().unwrap_or("null")).unwrap_or(Value::Null);
            it["value"] = env;
        }
    }
    b
}

fn gcp_assets() -> GcpAssets {
    GcpAssets::load(&repo_root()).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn gcp_firewall_and_instance_equal_vm_sh() {
    let cfg = std::fs::read_to_string(repo_root().join("configs/serve/gcp-h3-turbo.toml"))
        .unwrap_or_else(|e| panic!("{e}"));
    let mut g = guard::default_guards();
    g.cap_s = 1800;
    let l = script_gcp(&ScriptGcp {
        name: "fv-h3-turbo-1025120000",
        image: IMAGE,
        key_hash: KEY_HASH_PLACEHOLDER,
        config_text: &cfg,
        machine: "g4-standard-48",
        zone: "europe-west4-b",
        fake: false,
        guards: g,
    });
    let o = gcp_options("h3-turbo", &["fasth3-4step-vsa", "aux"]);
    let assets = gcp_assets();
    let body = instance_body(&l, &o, &assets, "golden-project", GCP_DEADLINE)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        gcp_normalized(&body, &assets, &cfg),
        fixture("gcp-instance.json")
    );
    assert_eq!(
        firewall_body(&l.name, &o, "golden-project"),
        fixture("gcp-firewall.json")
    );
    assert_eq!(guard::check_region(&l), Ok(()));
}

/// `fv-deploy up --kind gcp --family h3 --preset h3-turbo`: `vm.sh up
/// h3-turbo` with fv-deploy's name, and the channel and the weights source
/// in the container env; nothing else differs.
#[test]
fn preset_gcp_is_the_script_vm_plus_the_channel() {
    let r = repo();
    let mut req = request("h3-turbo", ProviderKind::Gcp);
    req.guards.cap_s = 1800;
    let mut l = from_preset(&r, &req, GCP_RUN).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(guard::check_region(&l), Ok(()));
    pin_image(&mut l, IMAGE);
    let o = launch::gcp_options(
        &r,
        &req,
        GCP_RUN,
        vec!["198.51.100.7/32".into()],
        false,
        1800,
        false,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(o, gcp_options("h3-turbo", &["fasth3-4step-vsa", "aux"]));
    let cfg = std::fs::read_to_string(repo_root().join("configs/serve/gcp-h3-turbo.toml"))
        .unwrap_or_else(|e| panic!("{e}"));
    let assets = gcp_assets();
    let body = instance_body(&l, &o, &assets, "golden-project", GCP_DEADLINE)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut want = fixture("gcp-instance.json");
    let name = "fv-deploy-h3-turbo-1025120000";
    want["name"] = json!(name);
    want["tags"]["items"] = json!(["fv-serve", name]);
    for it in want["metadata"]["items"]
        .as_array_mut()
        .unwrap_or(&mut Vec::new())
    {
        if it["key"] == "fv-env" {
            it["value"]["FV_RELEASE_CHANNEL"] = json!("stable");
            it["value"]["FV_WEIGHTS_SOURCE"] = json!("volume");
        }
    }
    assert_eq!(gcp_normalized(&body, &assets, &cfg), want);
    let mut fw = fixture("gcp-firewall.json");
    fw["name"] = json!(format!("fv-serve-{name}"));
    fw["targetTags"] = json!([name]);
    assert_eq!(firewall_body(&l.name, &o, "golden-project"), fw);
}

/// Every `[[gcp]]` row builds for its family (a model family through
/// `[dispatch]`, or `fake`), passes the GCP guard and sends its config;
/// only `fake` mounts no weights disk.
#[test]
fn every_gcp_row_builds() {
    let r = repo();
    let assets = gcp_assets();
    for row in &r.deploy.gcp {
        let family = match &row.dispatch_family {
            Some(df) => r
                .deploy
                .dispatch
                .iter()
                .find(|(_, d)| *d == df)
                .map(|(m, _)| m.clone())
                .unwrap_or_else(|| df.clone()),
            None => "fake".into(),
        };
        let mut req = request(&row.name, ProviderKind::Gcp);
        req.family = family.clone();
        let l = from_preset(&r, &req, "1").unwrap_or_else(|e| panic!("{}: {e}", row.name));
        assert_eq!(guard::check_region(&l), Ok(()), "{}", row.name);
        assert_eq!(
            l.placement.volume.is_none(),
            row.name == "fake",
            "{}",
            row.name
        );
        assert_eq!(l.gpu.types, std::slice::from_ref(&row.machine), "{}", row.name);
        let o = launch::gcp_options(
            &r,
            &req,
            "1",
            vec!["198.51.100.7/32".into()],
            false,
            1800,
            false,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let body = instance_body(&l, &o, &assets, "p", 1).unwrap_or_else(|e| panic!("{e}"));
        let file = std::fs::read_to_string(repo_root().join("configs/serve").join(&row.config))
            .unwrap_or_default();
        let items = body["metadata"]["items"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let cfg = items
            .iter()
            .find(|i| i["key"] == "fv-config")
            .map(|i| i["value"].clone());
        assert_eq!(cfg, Some(json!(file)), "{}", row.name);
        let env: Value = items
            .iter()
            .find(|i| i["key"] == "fv-env")
            .and_then(|i| i["value"].as_str())
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null);
        assert!(env.get("FV_WORKER_TOML_B64").is_none(), "{}", row.name);
        let mut wrong = req.clone();
        wrong.family = "no-such-family".into();
        assert!(from_preset(&r, &wrong, "1").is_err(), "{}", row.name);
    }
    let mut us = request("h3-turbo", ProviderKind::Gcp);
    us.region = "us".into();
    let e = from_preset(&r, &us, "1")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("US weights volume deleted"), "{e}");
}

#[test]
fn gcp_prices_are_read_from_lib_sh() {
    let p = GcpPrices::load(&repo_root()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        p.price("g4-standard-48", false, "europe-west4"),
        Some(4.9499)
    );
    assert_eq!(
        p.price("g4-standard-48", true, "europe-west4"),
        Some(2.2091)
    );
    // Not listed in the region: the highest listed row (conservative).
    assert_eq!(
        p.price("g2-standard-16", false, "europe-north1"),
        Some(1.2058)
    );
    assert_eq!(p.price("no-such-machine", false, "europe-west4"), None);
    assert!(p.0.len() >= 10, "{:?}", p.0.len());
}
