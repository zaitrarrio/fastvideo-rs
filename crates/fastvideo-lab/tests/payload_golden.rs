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
    // Only the gold, oracle-r2, serve-engine, sfstream, mmaudio and wan tiers
    // read R2 trees (ADR-0019 §16, §21, §22 S3, S4, S5, S6); every other
    // tier's pod is unchanged.
    assert_eq!(
        r2_tiers, 6,
        "six volume+r2 tiers (gold, oracle-r2, serve-engine, sfstream, mmaudio, wan)"
    );
    for t in [
        "gold",
        "oracle-r2",
        "serve-engine",
        "sfstream",
        "mmaudio",
        "wan",
    ] {
        assert_eq!(
            lab.tier(t).map(|t| t.weights),
            Some(TierWeights::VolumeR2),
            "{t}"
        );
    }
    // oracle-r2's pod is the oracle tier's but for the disk (150 GB: its
    // 28.93 GB R2 tree and the 120 GB) and the idle limit (r2_idle_min of
    // its 5400 s cell cap: 115 min), as runpod-http.sh plan's with those.
    let r2 = fixture("oracle-r2");
    let vol = fixture("oracle");
    assert_eq!(r2["containerDiskInGb"], json!(150));
    assert_eq!(vol["containerDiskInGb"], json!(120));
    let cmd = r2["dockerStartCmd"][2].as_str().unwrap_or("");
    assert!(
        cmd.contains("lim=115; [ \"$used\" = 1 ] && lim=115"),
        "the idle limit"
    );
    assert_eq!(fastvideo_lab::weights::r2_idle_min(5400), 115);
    // serve-engine's pod (ADR-0019 §22, S3) is the rtx6000 tier's but for the
    // disk (150 GB: fastwan21-1.3b's 29.21 GB and the 120 GB) and the idle
    // limit (r2_idle_min of its 3600 s cell cap: 85 min).
    let se = fixture("serve-engine");
    assert_eq!(se["containerDiskInGb"], json!(150));
    let cmd = se["dockerStartCmd"][2].as_str().unwrap_or("");
    assert!(
        cmd.contains("lim=85; [ \"$used\" = 1 ] && lim=85"),
        "the idle limit"
    );
    assert_eq!(fastvideo_lab::weights::r2_idle_min(3600), 85);
    // sfstream's (ADR-0019 §22, S4) likewise: 150 GB (sfwan21-1.3b's
    // 28.93 GB and the 120 GB), 85 min, and the TAE fetch at start (`tae =
    // true`: its arms read `{tae}` without a TAE gate, Addition 16).
    let sf = fixture("sfstream");
    assert_eq!(sf["containerDiskInGb"], json!(150));
    let cmd = sf["dockerStartCmd"][2].as_str().unwrap_or("");
    assert!(
        cmd.contains("lim=85; [ \"$used\" = 1 ] && lim=85"),
        "the idle limit"
    );
    assert!(
        cmd.contains(
            "if [ \"0\" != 1 ] && [ -f /opt/fastvideo-rs/scripts/gpu/fetch-tae.sh ]; then"
        ),
        "the TAE fetch runs (FV_SKIP_TAE 0)"
    );
    assert!(!cmd.contains("if [ \"1\" != 1 ]"), "not skipped");
    // mmaudio's (ADR-0019 §22, S5) likewise: 180 GB (at most fastwan21-1.3b
    // and mmaudio-44k-v2 at once, 50.68 GB, and the 120 GB), 85 min, and the
    // TAE fetch at start (its Wan arms read `{tae}` without a TAE gate).
    let mm = fixture("mmaudio");
    assert_eq!(mm["containerDiskInGb"], json!(180));
    let cmd = mm["dockerStartCmd"][2].as_str().unwrap_or("");
    assert!(
        cmd.contains("lim=85; [ \"$used\" = 1 ] && lim=85"),
        "the idle limit"
    );
    assert!(
        cmd.contains(
            "if [ \"0\" != 1 ] && [ -f /opt/fastvideo-rs/scripts/gpu/fetch-tae.sh ]; then"
        ),
        "the TAE fetch runs (FV_SKIP_TAE 0)"
    );
    // wan's (ADR-0019 §22, S6): 210 GB (wan21-t2v-14b's 80.41 GB, the most
    // fv-lab's drop-then-fetch plan holds at once, and the 120 GB), 85 min,
    // the TAE fetch at start (its 45 gated arms read `{tae}`, Addition 16).
    let wan = fixture("wan");
    assert_eq!(wan["containerDiskInGb"], json!(210));
    let cmd = wan["dockerStartCmd"][2].as_str().unwrap_or("");
    assert!(
        cmd.contains("lim=85; [ \"$used\" = 1 ] && lim=85"),
        "the idle limit"
    );
    assert!(
        cmd.contains(
            "if [ \"0\" != 1 ] && [ -f /opt/fastvideo-rs/scripts/gpu/fetch-tae.sh ]; then"
        ),
        "the TAE fetch runs (FV_SKIP_TAE 0)"
    );
    // Two tiers ask for a host-RAM floor, both for a Wan 14B arm's host load
    // (lab/tiers.toml): gold (256 GB since the 7e580e5 gold run measured its
    // plug arm's peak, ADR-0019 §16.10) and wan (its 14B arms, as gold's,
    // §22, S6); every other tier's pod has Runpod's default.
    assert_eq!(floors, 2, "two tiers with min_ram_gb (gold, wan)");
    assert_eq!(lab.tier("gold").and_then(|t| t.min_ram_gb), Some(256));
    assert_eq!(lab.tier("wan").and_then(|t| t.min_ram_gb), Some(256));
    assert_eq!(wan["minRAMPerGPU"], json!(256));
}

/// The script runner's pod for a `volume+r2` tier's GPU check (ADR-0019
/// §21.8): `FV_FAMILY=<family> FV_R2_TREES=<tree> runpod-http.sh plan` reads
/// its weights as fv-lab's pod of the tier does: the same Runpod secret
/// references (`weights::r2_env_refs`), the tier's disk and idle limit,
/// fv-lab's link roots (`TreeSources::link_roots`) and the pinned revision for
/// `r2-stage.sh`, and the matrix's `$W` the merged root. Without
/// `FV_R2_TREES` the body has no `env` (`fv-lab-goldens.sh`). Four: the
/// `oracle` family's `sfwan13` for `oracle-r2`
/// (`fixtures/runpod-http-oracle-sfwan13-r2.json`, §21.7), `serve-engine`'s
/// wan group (`fixtures/runpod-http-serve-engine-r2.json`, §22.12),
/// `sfstream` (§22.14), `mmaudio`'s two trees (§22.16) and `wan`'s two
/// (§22.18), one test each. The script side's disk is the tier's, but for
/// `wan` (`script_side_r2_pod_on`: it stages the check's two trees at once
/// where the tier is sized for its whole drop-then-fetch plan).
fn script_side_r2_pod(tier: &str, golden: &str, plain: &str, trees: &[&str]) {
    let disk = Lab::load(&repo_root())
        .expect("lab/ parses")
        .tier(tier)
        .expect("tier")
        .container_disk_gb;
    script_side_r2_pod_on(tier, golden, plain, trees, disk);
}

fn script_side_r2_pod_on(tier: &str, golden: &str, plain: &str, trees: &[&str], disk: u32) {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let sources = fastvideo_lab::weights::TreeSources::load(&repo_root()).expect("trees");
    let t = lab.tier(tier).expect("tier");
    let body = fixture(golden);
    let refs: serde_json::Map<String, Value> = r2_env_refs()
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    assert_eq!(body["env"], Value::Object(refs), "{tier}");
    assert_eq!(
        body["containerDiskInGb"].as_u64(),
        Some(u64::from(disk)),
        "{tier}"
    );
    let cmd = body["dockerStartCmd"][2].as_str().expect("a start command");
    let idle = fastvideo_lab::weights::r2_idle_min(t.cell_timeout_s);
    assert!(
        cmd.contains(&format!("lim={idle}; [ \"$used\" = 1 ] && lim={idle}")),
        "{tier}"
    );
    assert!(
        cmd.starts_with("_fvr2_id=\"${FV_WEIGHTS_R2_ACCESS_KEY_ID:-}\""),
        "{tier}"
    );
    let specs: Vec<String> = trees
        .iter()
        .map(|tree| {
            let rev = sources
                .r2
                .get(*tree)
                .map(|t| t.rev.clone())
                .unwrap_or_else(|| panic!("{tree} in weights-r2.tsv"));
            format!("{tree}@{rev}")
        })
        .collect();
    let want = format!(
        "$SCRATCH/weights $SCRATCH/r2 \"$OUT\" {} /workspace/weights \"{}\" {} ",
        t.cell_timeout_s,
        sources.link_roots().join(" "),
        specs.join(" ")
    );
    assert!(cmd.contains(&want), "{tier}: {want}");
    assert!(
        cmd.contains("env  FV_WORK=$SCRATCH FV_SCRATCH=$SCRATCH "),
        "{tier}"
    );
    // Without FV_R2_TREES: no env, the volume's root.
    let plain = fixture(plain);
    assert!(plain.get("env").is_none(), "{tier}");
    let pcmd = plain["dockerStartCmd"][2].as_str().unwrap_or("");
    assert!(
        pcmd.contains("env  FV_WORK=/workspace FV_SCRATCH=$SCRATCH "),
        "{tier}"
    );
}

#[test]
fn the_script_side_r2_pod_reads_its_trees_as_the_oracle_r2_tier() {
    script_side_r2_pod(
        "oracle-r2",
        "oracle-sfwan13-r2",
        "oracle",
        &["sfwan21-1.3b"],
    );
    // The oracle tier's 120 GB body plus the tree.
    assert_eq!(fixture("oracle")["containerDiskInGb"].as_u64(), Some(120));
}

#[test]
fn the_script_side_r2_pod_reads_its_trees_as_the_serve_engine_tier() {
    script_side_r2_pod(
        "serve-engine",
        "serve-engine-r2",
        "serve-engine",
        &["fastwan21-1.3b"],
    );
    // The tier's own body already has its 150 GB.
    assert_eq!(
        fixture("serve-engine")["containerDiskInGb"].as_u64(),
        Some(150)
    );
}

#[test]
fn the_script_side_r2_pod_reads_its_trees_as_the_sfstream_tier() {
    script_side_r2_pod("sfstream", "sfstream-r2", "sfstream", &["sfwan21-1.3b"]);
    assert_eq!(fixture("sfstream")["containerDiskInGb"].as_u64(), Some(150));
}

/// `mmaudio`'s GPU check (ADR-0019 §22, S5, §22.16): the script side stages
/// both trees its two checked arms read (`wan13-audio`: fastwan21-1.3b and
/// mmaudio-44k-v2; `ours-v2a`: mmaudio-44k-v2) before the matrix, on the
/// tier's 180 GB, which holds them (51 GB) and the 120 GB; the whole family
/// with sfwan21-1.3b too would need 200 GB on the script side, which stages
/// every tree at once where fv-lab drops a tree no later arm reads.
#[test]
fn the_script_side_r2_pod_reads_its_trees_as_the_mmaudio_tier() {
    script_side_r2_pod(
        "mmaudio",
        "mmaudio-r2",
        "mmaudio",
        &["fastwan21-1.3b", "mmaudio-44k-v2"],
    );
    assert_eq!(fixture("mmaudio")["containerDiskInGb"].as_u64(), Some(180));
    let sources = fastvideo_lab::weights::TreeSources::load(&repo_root()).expect("trees");
    let bytes: u64 = ["fastwan21-1.3b", "mmaudio-44k-v2"]
        .iter()
        .map(|t| sources.r2.get(*t).map(|t| t.bytes).expect("an R2 tree"))
        .sum();
    assert_eq!(bytes, 50_676_290_876);
    assert!(bytes.div_ceil(1_000_000_000) + 120 <= 180);
}

/// `wan`'s GPU check (ADR-0019 §22, S6, §22.18): the script side stages both
/// trees its checked arms read (`wan13-dmd`, `wan13-dmd-dense`:
/// fastwan21-1.3b; `cli-fw22`, `engine-fw22`: fastwan22-ti2v-5b;
/// `kernels-wan`: none) before the matrix, on 174 GB (their 53.41 GB and the
/// 120 GB), with the tier's 256 GB host-RAM floor; the tier's own body has 210
/// GB, for fv-lab's plan of the whole family.
#[test]
fn the_script_side_r2_pod_reads_its_trees_as_the_wan_tier() {
    let trees = ["fastwan21-1.3b", "fastwan22-ti2v-5b"];
    script_side_r2_pod_on("wan", "wan-r2", "wan", &trees, 174);
    assert_eq!(fixture("wan")["containerDiskInGb"].as_u64(), Some(210));
    assert_eq!(fixture("wan-r2")["minRAMPerGPU"].as_u64(), Some(256));
    let sources = fastvideo_lab::weights::TreeSources::load(&repo_root()).expect("trees");
    let bytes: u64 = trees
        .iter()
        .map(|t| sources.r2.get(*t).map(|t| t.bytes).expect("an R2 tree"))
        .sum();
    assert_eq!(bytes, 53_413_890_791);
    assert_eq!(bytes.div_ceil(1_000_000_000) + 120, 174);
}
