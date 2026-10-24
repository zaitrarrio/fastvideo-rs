//! The payload golden (docs/adr/0019-fv-lab.md §6): the pod `fv-lab pod up
//! --family rtx6000` creates is the one `runpod-http.sh run` creates
//! (`fixtures/runpod-http-rtx6000.json`, from `runpod-http.sh plan` through
//! `scripts/gpu/tests/fv-lab-goldens.sh`), apart from the listed differences:
//!
//! 1. the start command runs `fv-lab run` where the script ran
//!    `bash …/runpod-matrix.sh` (the one changed line);
//! 2. the body carries `env: {}` (fv-deploy's `create_body` always sends it;
//!    the script sends `env` only with FV_POD_FILES, which fv-lab has not);
//! 3. (`up` only, not in a plan) the image is pinned to its digest.
//!
//! The name is equal because the golden is made with `FV_POD_PREFIX=fv-lab`.

use std::path::PathBuf;

use serde_json::{json, Value};

use fastvideo_deploy::provider::runpod::pod_create_body;
use fastvideo_lab::plan::Lab;
use fastvideo_lab::pod::{launch, StartOptions};

const SHA: &str = "1234567";
const TAG: &str = "1234567-10241200";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/runpod-http-rtx6000.json");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&t).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn pod_body_equals_runpod_http_sh_but_the_runner() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let tier = lab.tier("rtx6000").expect("tier rtx6000");
    let l = launch(tier, SHA, &StartOptions::for_tier(tier, TAG)).expect("launch");
    let mut ours = pod_create_body(&l);
    // Difference 2.
    assert_eq!(
        ours.as_object_mut().and_then(|m| m.remove("env")),
        Some(json!({}))
    );
    // Difference 1.
    let mut theirs = fixture();
    let cmd = theirs["dockerStartCmd"][2]
        .as_str()
        .expect("a start command")
        .to_owned();
    let script =
        "bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh rtx6000 >\"$OUT/matrix.out\" 2>&1";
    let lab_run = "/opt/fastvideo-rs/target/release/fv-lab run rtx6000 >\"$OUT/matrix.out\" 2>&1";
    assert_eq!(cmd.matches(script).count(), 1, "the runner line moved");
    theirs["dockerStartCmd"][2] = Value::String(cmd.replace(script, lab_run));
    if ours != theirs {
        let a = ours["dockerStartCmd"][2].as_str().unwrap_or("");
        let b = theirs["dockerStartCmd"][2].as_str().unwrap_or("");
        for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
            assert_eq!(x, y, "start command line {i}");
        }
        assert_eq!(ours, theirs);
    }
}
