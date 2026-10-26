//! The payload golden (docs/adr/0019-fv-lab.md §6, D1b): for every tier, the
//! pod `fv-lab pod up --family <tier>` creates is the one `FV_FAMILY=<tier>
//! runpod-http.sh run` creates (`fixtures/runpod-http-<tier>.json`, from
//! `runpod-http.sh plan` through
//! `scripts/gpu/tests/fv-lab-goldens.sh`), apart from the listed differences:
//!
//! 1. the start command runs `fv-lab run` where the script ran
//!    `bash …/runpod-matrix.sh` (the one changed line);
//! 2. the body carries `env: {}` (fv-deploy's `create_body` always sends it;
//!    the script sends `env` only with FV_POD_FILES, which fv-lab has not);
//!    a `weights = "volume+r2"` tier's (`gold`, ADR-0019 §16) holds the R2
//!    key as Runpod secret references and nothing else
//!    (`weights::r2_env_refs`), which the script has no way to send;
//! 3. (`up` only, not in a plan) the image is pinned to its digest.
//!
//! The name is equal because the golden is made with `FV_POD_PREFIX=fv-lab`.
//! A tier's host-RAM floor (`min_ram_gb`, ADR-0019 §16.10) is no difference:
//! the golden passes it as the script's `FV_MIN_RAM_GB`, and both bodies
//! carry `minRAMPerGPU`; a tier without one has the field in neither.
//!
//! For the tiers of `scripts/gpu/fv-lab-retired.sh` the fixture is frozen
//! (ADR-0019 §10): `runpod-http.sh run` hands them to `fv-lab pod up`, so the
//! body is the one the script sent before its branch went, and fv-lab's pod
//! must stay it unless a PR changes both on purpose.
//!
//! A restored mode (ADR-0019 §12, `FV_LAB_RESTORED`) is the body
//! `FV_FAMILY=<family> FV_EXTRA_ENV=<K=V> runpod-http.sh plan` sent at
//! 5987e6a4, read with the family's name as the tier's (pod name, run
//! directory, runner) and without the mode's `FV_EXTRA_ENV` (`common`).

mod common;

use std::path::PathBuf;

use serde_json::{json, Value};

use fastvideo_deploy::provider::runpod::pod_create_body;
use fastvideo_lab::plan::{Lab, TierWeights};
use fastvideo_lab::pod::{launch, StartOptions};
use fastvideo_lab::weights::r2_env_refs;

const SHA: &str = "1234567";
const TAG: &str = "1234567-10241200";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(family: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(format!("runpod-http-{family}.json"));
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&t).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn pod_body_equals_runpod_http_sh_but_the_runner() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    assert!(lab.tiers.tiers.len() >= 10);
    let mut restored = 0;
    let mut r2_tiers = 0;
    let mut floors = 0;
    for tier in &lab.tiers.tiers {
        let family = tier.name.as_str();
        let l = launch(tier, SHA, &StartOptions::for_tier(tier, TAG)).expect("launch");
        let mut ours = pod_create_body(&l);
        // Difference 2.
        let env = match tier.weights {
            TierWeights::Volume => json!({}),
            TierWeights::VolumeR2 => {
                let refs = r2_env_refs();
                assert!(refs.values().all(|v| v.starts_with("{{ RUNPOD_SECRET_")));
                json!(refs)
            }
        };
        assert_eq!(
            ours.as_object_mut().and_then(|m| m.remove("env")),
            Some(env),
            "{family}"
        );
        r2_tiers += usize::from(tier.weights == TierWeights::VolumeR2);
        assert_eq!(
            ours.get("minRAMPerGPU").and_then(Value::as_u64),
            tier.min_ram_gb.map(u64::from),
            "{family}: minRAMPerGPU is the tier's min_ram_gb"
        );
        floors += usize::from(tier.min_ram_gb.is_some());
        // Difference 1.
        let mut theirs = fixture(family);
        let mut cmd = theirs["dockerStartCmd"][2]
            .as_str()
            .expect("a start command")
            .to_owned();
        if let Some(r) = common::restored_tier(family) {
            cmd = r.start_command(&cmd);
            let name = theirs["name"].as_str().expect("a pod name");
            theirs["name"] = Value::String(r.pod_name(name));
        }
        let script = format!(
            "bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh {family} >\"$OUT/matrix.out\" 2>&1"
        );
        let lab_run = format!(
            "/opt/fastvideo-rs/target/release/fv-lab run {family} >\"$OUT/matrix.out\" 2>&1"
        );
        assert_eq!(
            cmd.matches(&script).count(),
            1,
            "{family}: the runner line moved"
        );
        theirs["dockerStartCmd"][2] = Value::String(cmd.replace(&script, &lab_run));
        restored += usize::from(common::restored_tier(family).is_some());
        if ours != theirs {
            let a = ours["dockerStartCmd"][2].as_str().unwrap_or("");
            let b = theirs["dockerStartCmd"][2].as_str().unwrap_or("");
            for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
                assert_eq!(x, y, "{family}: start command line {i}");
            }
            assert_eq!(ours, theirs, "{family}");
        }
    }
    assert_eq!(
        restored,
        common::restored().len(),
        "every restored mode is a tier"
    );
    // Only the gold tier reads R2 trees; every other tier's pod is unchanged.
    assert_eq!(r2_tiers, 1, "one volume+r2 tier (gold)");
    assert_eq!(
        lab.tier("gold").map(|t| t.weights),
        Some(TierWeights::VolumeR2)
    );
    // Only the gold tier asks for a host-RAM floor (its 14B arm's host load,
    // lab/tiers.toml); every other tier's pod has Runpod's default.
    assert_eq!(floors, 1, "one tier with min_ram_gb (gold)");
    assert_eq!(lab.tier("gold").and_then(|t| t.min_ram_gb), Some(192));
}
