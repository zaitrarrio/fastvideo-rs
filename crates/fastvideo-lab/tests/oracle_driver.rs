//! `fv-lab oracle` (docs/adr/0019-fv-lab.md §19): the upstream stacks of
//! `lab/upstream.toml`, the upstream pod's body against `runpod-http.sh
//! plan-upstream` (the payload golden), the plan's refusals, the money
//! guard, and the driver on a fake world: its order, the release of the
//! upstream pods on `ORACLE_REFS_DONE`, on the stock wait, on a timeout and
//! on every error.
//!
//! The payload golden's intended differences (`fixtures/runpod-http-upstream-<stack>.json`,
//! `scripts/gpu/tests/fv-lab-goldens.sh`):
//!
//! 1. the start command carries the pod's own wall-cap watchdog right after
//!    `set -u`, the block every fv-lab pod has (`src/pod-start.sh.in`,
//!    docs/adr/0023-lab-report-and-ci-bridge.md §4); `runpod-http.sh
//!    upstream`'s pod has none (only the driver's `nohup sleep; curl`);
//! 2. the body carries `env: {}` (fv-deploy's `create_body` always sends it);
//! 3. (`up` only, not in a plan) the image is pinned to its digest.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};

use fastvideo_deploy::provider::runpod::pod_create_body;
use fastvideo_deploy::provider::{Account, ProviderKind};
use fastvideo_lab::oracle::{
    check_balance, check_image_tag, drive, plan_oracle, runtime_args, up_steps, upstream_launch,
    upstream_start_command, OracleOps, OraclePlan, OracleRequest, RuntimePod, Stacks, UpPod,
    UpState, UpstreamPlan, UpstreamStart, STOCK_WAIT_S,
};
use fastvideo_lab::plan::Lab;

const SHA: &str = "1234567";
const TAG: &str = "1234567-10241200";
const UP_SHA: &str = "1234567890abcdef1234567890abcdef12345678";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lab() -> (Lab, Stacks) {
    let root = repo_root();
    let lab = Lab::load(&root).expect("lab/ parses");
    let stacks = Stacks::load(&root).expect("lab/upstream.toml parses");
    (lab, stacks)
}

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&t).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The cap watchdog block of the runtime pod's start command (between
/// `set -u` and the run's own lines): what difference 1 inserts.
fn cap_block() -> String {
    let t = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/pod-start.sh.in"),
    )
    .expect("pod-start.sh.in");
    let a = t.find("set -u\n").expect("set -u") + "set -u\n".len();
    let b = t
        .find("# All writes on the container disk")
        .expect("the run's lines");
    t[a..b].to_owned()
}

// ------------------------------------------------------------------ the golden

#[test]
fn the_upstream_pod_body_equals_runpod_http_sh_plan_upstream_but_the_cap_watchdog() {
    let (_, stacks) = lab();
    // The selections fv-lab-goldens.sh passes (oracle.sh's start_up settings,
    // UP_STEPS in §15.7's form).
    let cases: [(&str, &[&str]); 2] = [
        ("fastvideo", &["fasth3-4step-vsa"]),
        ("sol-ltx25", &["ltx25-512p", "ltx25-512p-dense"]),
    ];
    for (name, targets) in cases {
        let stack = stacks
            .stacks
            .iter()
            .find(|s| s.name == name)
            .expect("the stack");
        let targets: Vec<String> = targets.iter().map(|t| (*t).to_owned()).collect();
        let start = UpstreamStart::for_stack(stack, TAG, UP_SHA, &up_steps(stack, &targets));
        let l = upstream_launch(stack, "latest", &start).expect("launch");
        let mut ours = pod_create_body(&l);
        // Difference 2.
        assert_eq!(
            ours.as_object_mut().and_then(|m| m.remove("env")),
            Some(json!({})),
            "{name}"
        );
        // Difference 1.
        let cmd = ours["dockerStartCmd"][2]
            .as_str()
            .expect("a command")
            .to_owned();
        let block = cap_block();
        let block = block.replace("@CAP_S@", &stack.wall_cap_s.to_string());
        assert_eq!(cmd.matches(&block).count(), 1, "{name}: one cap watchdog");
        assert!(cmd.contains(&format!("set -u\n{block}OUT=/root/runs/upstream/{TAG}\n")));
        ours["dockerStartCmd"][2] = Value::String(cmd.replace(&block, ""));
        let theirs = fixture(&format!("runpod-http-upstream-{name}.json"));
        if ours != theirs {
            let a = ours["dockerStartCmd"][2].as_str().unwrap_or("");
            let b = theirs["dockerStartCmd"][2].as_str().unwrap_or("");
            for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
                assert_eq!(x, y, "{name}: start command line {i}");
            }
            assert_eq!(ours, theirs, "{name}");
        }
        assert_eq!(
            ours["minRAMPerGPU"].as_u64(),
            stack.min_ram_gb.map(u64::from),
            "{name}"
        );
    }
    // Every stack has a golden.
    assert_eq!(stacks.stacks.len(), cases.len());
}

#[test]
fn the_upstream_start_command_is_the_scripts_with_its_values() {
    let (_, stacks) = lab();
    let s = &stacks.stacks[0];
    let t = vec!["fasth3-8step".to_owned(), "fasth3-4step-vsa".to_owned()];
    // The weight steps once each, in the stack's order; the targets as selected.
    assert_eq!(
        up_steps(s, &t),
        "info:box weights:fasth3-8step weights:h3-diffusers oracle:fasth3-8step,fasth3-4step-vsa"
    );
    assert_eq!(
        up_steps(s, &["h3-ref2va-4step".to_owned()]),
        "info:box oracle:h3-ref2va-4step"
    );
    let mut o = UpstreamStart::for_stack(s, TAG, UP_SHA, "info:box oracle:fasth3-4step-vsa");
    let cmd = upstream_start_command(&o).expect("command");
    assert!(cmd.contains(&format!("fetch -q --depth 1 origin {UP_SHA} ")));
    assert!(cmd.contains(
        "env FASTVIDEO_DUMP_OPS=0,1,24,47 UP_STEPS=\"info:box oracle:fasth3-4step-vsa\" UP_STEPS_BG=\"\" UP_CELLS=\"\"   UP_CELL_TIMEOUT_S=7200 UP_LOCAL=1"
    ));
    assert!(
        cmd.contains("[ \"net\" = net ] && tx0=$(txb)"),
        "the idle watchdog counts traffic"
    );
    assert!(cmd.contains(&format!("    sleep {}\n", s.wall_cap_s)));
    for m in [
        "@TAG@",
        "@UP_SHA@",
        "@UP_STEPS@",
        "@EXTRA_ENV@",
        "@CAP_S@",
        "@IDLE_MIN@",
    ] {
        assert!(!cmd.contains(m), "{m}");
    }
    // Shell text is refused.
    o.up_steps = "info:box $(id)".into();
    assert!(upstream_start_command(&o).is_err());
    o.up_steps = "info:box".into();
    o.up_sha = "1234567".into();
    assert!(upstream_start_command(&o).is_err());
    o.up_sha = UP_SHA.into();
    o.extra_env = vec!["X=a b".into()];
    assert!(upstream_start_command(&o).is_err());
    o.extra_env.clear();
    o.cap_s = 0;
    assert!(upstream_start_command(&o).is_err());
    for (tag, ok) in [
        ("latest", true),
        ("sha-1234567", true),
        ("sha-123", false),
        ("main", false),
    ] {
        assert_eq!(check_image_tag(tag).is_ok(), ok, "{tag}");
    }
}

// ------------------------------------------------------------------ the stacks file

#[test]
fn every_oracle_cell_has_a_stack_and_every_stack_target_a_cell() {
    let (lab, stacks) = lab();
    stacks.check_against(&lab).expect("stacks and cells agree");
    let refs: Vec<String> = lab
        .tier("oracle")
        .expect("oracle tier")
        .cells
        .iter()
        .filter_map(|c| lab.cells.get(c))
        .filter_map(|c| c.reference.as_ref().map(|r| r.target.clone()))
        .collect();
    assert_eq!(refs.len(), 9);
    for t in &refs {
        assert!(stacks.for_target(t).is_some(), "{t}");
    }
    // The H3 LoRA references need the host-RAM floor (docs/oracle.md).
    assert_eq!(
        stacks
            .for_target("fasth3-4step-vsa")
            .map(|(s, _)| s.min_ram_gb),
        Some(Some(192))
    );
}

#[test]
fn the_stacks_file_refuses_what_it_cannot_run() {
    let text = std::fs::read_to_string(repo_root().join("lab/upstream.toml")).unwrap();
    assert!(Stacks::parse(&text, "u").is_ok());
    for (from, to) in [
        ("schema = 1", "schema = 2"),
        ("region = \"eu\"", "region = \"us\""),
        ("gpu_count = 1", "gpu_count = 2"),
        ("wall_cap_s = 14400", "wall_cap_s = 14401"),
        ("min_ram_gb = 192", "min_ram_gb = 0"),
        ("cell_timeout_s = 7200", "cell_timeout_s = 0"),
        ("container_disk_gb = 150", "container_disk_gb = 0"),
        ("max_usd_per_hr = 2.5", "max_usd_per_hr = 0"),
        ("image = \"fastvideo\"", "image = \"Fast Video\""),
        (
            "env = [\"FASTVIDEO_DUMP_OPS=0,1,24,47\"]",
            "env = [\"FASTVIDEO_DUMP_OPS=$(id)\"]",
        ),
        (
            "steps = [\"weights:h3-diffusers\"]",
            "steps = [\"oracle:x\"]",
        ),
        (
            "name = \"fasth3-4step-dense\"",
            "name = \"fasth3-4step-vsa\"",
        ),
        ("name = \"sol-ltx25\"", "name = \"fastvideo\""),
        (
            "cell_timeout_s = 7200",
            "cell_timeout_s = 7200\nunknown = 1",
        ),
    ] {
        assert!(text.contains(from), "{from}");
        let bad = text.replacen(from, to, 1);
        assert!(Stacks::parse(&bad, "u").is_err(), "{from} -> {to}");
    }
    // A target no cell reads, or a cell reference no stack makes.
    let (lab, _) = lab();
    let extra = Stacks::parse(
        &text.replacen("name = \"ltx25-4k-dense\"", "name = \"ltx25-i2v\"", 1),
        "u",
    )
    .unwrap();
    assert!(extra.check_against(&lab).is_err());
}

// ------------------------------------------------------------------ the plan

fn request(cells: &[&str]) -> OracleRequest {
    OracleRequest {
        tier: "oracle".into(),
        sha: SHA.into(),
        up_sha: UP_SHA.into(),
        image_tag: "latest".into(),
        cells: cells.iter().map(|c| (*c).to_owned()).collect(),
        extra_env: Vec::new(),
        runtime_tag: TAG.into(),
        upstream_tags: vec![
            "1234567-1024120001-aaaa".into(),
            "1234567-1024120002-bbbb".into(),
        ],
        runtime_cap_s: None,
        upstream_cap_s: None,
        stock_wait_s: STOCK_WAIT_S,
        refs_wait_s: None,
        hold: false,
    }
}

const VSA: [&str; 5] = [
    "oracle-fasth3-4step-vsa",
    "oracle-fasth3-4step-vsa-diff",
    "oracle-fasth3-4step-vsa-f32",
    "oracle-fasth3-4step-vsa-f32-diff",
    "oracle-fasth3-4step-vsa-bf16-vs-f32",
];

#[test]
fn the_plan_brings_up_one_pod_per_stack_of_the_selected_targets() {
    let (lab, stacks) = lab();
    let p = plan_oracle(&lab, &stacks, &request(&VSA)).expect("plan");
    assert_eq!(p.upstream.len(), 1);
    let u = &p.upstream[0];
    assert_eq!(u.stack, "fastvideo");
    assert_eq!(u.targets, ["fasth3-4step-vsa"]);
    assert_eq!(
        u.start.up_steps,
        "info:box weights:h3-diffusers oracle:fasth3-4step-vsa"
    );
    assert_eq!(u.launch.name, "fv-lab-upstream-1234567-1024120001-aaaa");
    assert_eq!(
        u.launch.image,
        "ghcr.io/zaitrarrio/fastvideo-rs-upstream-fastvideo:latest"
    );
    assert_eq!(u.launch.gpu.min_ram_gb, Some(192));
    assert_eq!(p.cells, VSA);
    assert_eq!(
        p.arms_by_target,
        [(
            "fasth3-4step-vsa".to_owned(),
            VSA.map(String::from).to_vec()
        )]
    );
    // The refs wait is what the upstream cap leaves: 14400 - 1800 - 1800.
    assert_eq!((p.stock_wait_s, p.refs_wait_s), (1800, 10800));
    // Two pods at $2.50/hr for 4 h each.
    assert!(
        (p.projection_usd - 20.0).abs() < 1e-9,
        "{}",
        p.projection_usd
    );
    // An LTX arm too: a second stack, its own tag, in lab/upstream.toml's order.
    let mut r = request(&["oracle-ltx25-512p", "oracle-ltx25-512p-diff", VSA[0]]);
    r.upstream_cap_s = Some(7200);
    r.runtime_cap_s = Some(5400);
    let p = plan_oracle(&lab, &stacks, &r).expect("plan");
    assert_eq!(
        p.upstream
            .iter()
            .map(|u| u.stack.as_str())
            .collect::<Vec<_>>(),
        ["fastvideo", "sol-ltx25"]
    );
    assert_eq!(
        p.upstream[1].launch.name,
        "fv-lab-upstream-1234567-1024120002-bbbb"
    );
    assert_eq!(
        p.upstream[1].start.up_steps,
        "info:box weights:ltx25 oracle:ltx25-512p"
    );
    assert_eq!(p.upstream[1].launch.disk.container_gb, 250);
    assert_eq!(p.upstream[1].launch.gpu.min_ram_gb, None);
    assert!(p
        .upstream
        .iter()
        .all(|u| u.start.cap_s == 7200 && u.launch.guards.cap_s == Some(7200)));
    assert_eq!(p.refs_wait_s, 7200 - 1800 - 1800);
    assert!((p.projection_usd - 2.5 * (1.5 + 2.0 + 2.0)).abs() < 1e-9);
    // The tier's order, whatever --cells' order.
    assert_eq!(
        p.cells,
        [VSA[0], "oracle-ltx25-512p", "oracle-ltx25-512p-diff"]
    );
    // The runtime side's command: the served pods' bases, the arms, the cap.
    let a = runtime_args(
        &p,
        &["https://a-8000.proxy.runpod.net/upstream/t1".into()],
        &[VSA[0].into()],
    );
    assert_eq!(
        a,
        [
            "pod",
            "up",
            "--sha",
            SHA,
            "--family",
            "oracle",
            "--tag",
            TAG,
            "--cells",
            VSA[0],
            "--extra-env",
            "FV_ORACLE_URL=https://a-8000.proxy.runpod.net/upstream/t1",
            "--wall-cap-s",
            "5400",
            "--yes",
            "--wait"
        ]
    );
}

#[test]
fn the_plan_refuses_before_anything_exists() {
    let (lab, stacks) = lab();
    let refused = |r: OracleRequest, why: &str| {
        let e = plan_oracle(&lab, &stacks, &r).expect_err(why);
        assert!(e.contains(why), "{e} lacks {why}");
    };
    refused(request(&[]), "--cells");
    refused(request(&["nope"]), "is not an arm of tier `oracle`");
    let mut r = request(&VSA);
    r.tier = "rtx6000".into();
    refused(r, "reads no upstream reference");
    let mut r = request(&VSA);
    r.extra_env = vec!["FV_ORACLE_URL=https://x".into()];
    refused(r, "sets FV_ORACLE_URL");
    let mut r = request(&VSA);
    r.extra_env = vec!["FV_ORACLE_TARGETS=fasth3-4step-vsa".into()];
    refused(r, "FV_ORACLE_TARGETS");
    let mut r = request(&VSA);
    r.sha = "abc".into();
    refused(r, "7 lower-case hex");
    let mut r = request(&VSA);
    r.up_sha = SHA.into();
    refused(r, "40 lower-case hex");
    let mut r = request(&VSA);
    r.image_tag = "main".into();
    refused(r, "`latest` or `sha-<7 hex>`");
    let mut r = request(&VSA);
    r.runtime_cap_s = Some(14_401);
    refused(r, "can only be lowered");
    let mut r = request(&VSA);
    r.upstream_cap_s = Some(20_000);
    refused(r, "can only be lowered");
    let mut r = request(&VSA);
    r.stock_wait_s = 0;
    refused(r, "--stock-wait-s 0");
    let mut r = request(&VSA);
    r.refs_wait_s = Some(10_801);
    refused(r, "at most 10800 s");
    let mut r = request(&VSA);
    r.upstream_cap_s = Some(3600);
    refused(r, "at most 0 s");
    let mut r = request(&VSA);
    r.upstream_tags.clear();
    refused(r, "no upstream tag");
}

#[test]
fn the_balance_must_cover_every_pod_of_the_run() {
    let (lab, stacks) = lab();
    let p = plan_oracle(&lab, &stacks, &request(&VSA)).unwrap();
    let acct = |b: f64, s: Option<f64>| Account {
        provider: ProviderKind::RunpodPod,
        balance_usd: b,
        spend_per_hr_usd: s,
    };
    // $20 for the two pods at their caps, $8 floor.
    assert!(check_balance(&acct(28.0, None), &p).is_ok());
    assert!(check_balance(&acct(27.99, None), &p).is_err());
    // What the account already spends counts over the run's longest cap.
    assert!(check_balance(&acct(32.0, Some(1.0)), &p).is_ok());
    assert!(check_balance(&acct(31.99, Some(1.0)), &p).is_err());
}

// ------------------------------------------------------------------ the driver

/// A scripted world. Times are seconds since the driver started (each sleep
/// adds its duration).
#[derive(Default)]
struct Fake {
    t: f64,
    events: Vec<String>,
    /// A stack whose pod does not come up.
    up_fails: Option<&'static str>,
    /// When each target's reference is served (absent: never).
    served_at: BTreeMap<&'static str, f64>,
    /// When a stack's upstream run writes DONE / its pod is gone.
    finished_at: BTreeMap<&'static str, f64>,
    gone_at: BTreeMap<&'static str, f64>,
    start_fails: bool,
    /// Relative to the runtime start: its pod recorded, ORACLE_REFS_DONE, its exit.
    started: Option<f64>,
    pod_after: Option<f64>,
    refs_after: Option<f64>,
    exit_after: Option<(f64, bool)>,
    aborted: bool,
    interrupt_at: Option<f64>,
    fetch_fails: bool,
    down_fails: bool,
    ups: usize,
    /// When the caller lifts the hold (absent: never).
    hold_at: Option<f64>,
    urls: Vec<String>,
}

impl Fake {
    fn rt(&self, after: Option<f64>) -> bool {
        matches!((self.started, after), (Some(s), Some(a)) if self.t >= s + a)
    }
    fn count(&self, prefix: &str) -> usize {
        self.events.iter().filter(|e| e.starts_with(prefix)).count()
    }
    fn index(&self, e: &str) -> usize {
        self.events
            .iter()
            .position(|x| x == e)
            .unwrap_or_else(|| panic!("no event `{e}` in {:?}", self.events))
    }
}

impl OracleOps for Fake {
    fn now(&self) -> f64 {
        self.t
    }
    fn sleep(&mut self, d: Duration) {
        self.t += d.as_secs_f64();
    }
    fn log(&mut self, _line: &str) {}
    fn interrupted(&self) -> bool {
        self.interrupt_at.is_some_and(|t| self.t >= t)
    }
    fn upstream_up(&mut self, p: &UpstreamPlan) -> Result<UpPod, String> {
        self.events.push(format!("up {}", p.stack));
        if self.up_fails == Some(p.stack.as_str()) {
            return Err("no capacity".into());
        }
        self.ups += 1;
        let id = format!("pod{}", self.ups);
        Ok(UpPod {
            stack: p.stack.clone(),
            url: format!("https://{id}-8000.proxy.runpod.net"),
            id,
            tag: p.start.tag.clone(),
            targets: p.targets.clone(),
        })
    }
    fn reference_served(&mut self, _pod: &UpPod, target: &str) -> bool {
        self.served_at.get(target).is_some_and(|t| self.t >= *t)
    }
    fn upstream_state(&mut self, pod: &UpPod) -> UpState {
        let s = pod.stack.as_str();
        if self.gone_at.get(s).is_some_and(|t| self.t >= *t) {
            UpState::Gone
        } else if self.finished_at.get(s).is_some_and(|t| self.t >= *t) {
            UpState::Finished
        } else {
            UpState::Running
        }
    }
    fn keepalive(&mut self, pod: &UpPod) {
        self.events.push(format!("keepalive {}", pod.id));
    }
    fn runtime_start(&mut self, args: &[String]) -> Result<(), String> {
        self.events.push(format!("start {}", args.join(" ")));
        if self.start_fails {
            return Err("spawn failed".into());
        }
        self.started = Some(self.t);
        Ok(())
    }
    fn runtime_pod(&mut self) -> Option<RuntimePod> {
        self.rt(self.pod_after).then(|| RuntimePod {
            id: "rt".into(),
            url: "https://rt-8000.proxy.runpod.net".into(),
            tag: TAG.into(),
        })
    }
    fn runtime_refs_done(&mut self, _rt: &RuntimePod) -> bool {
        self.rt(self.refs_after)
    }
    fn runtime_exit(&mut self) -> Option<bool> {
        if self.aborted {
            return Some(false);
        }
        let (after, ok) = self.exit_after?;
        self.rt(Some(after)).then_some(ok)
    }
    fn runtime_abort(&mut self) {
        self.events.push("abort".into());
        self.aborted = true;
    }
    fn publish_urls(&mut self, urls: &[String]) {
        self.events.push(format!("urls {}", urls.join(",")));
        self.urls = urls.to_vec();
    }
    fn hold_released(&mut self) -> bool {
        self.hold_at.is_some_and(|t| self.t >= t)
    }
    fn upstream_fetch(&mut self, pod: &UpPod) -> Result<(), String> {
        self.events.push(format!("fetch {}", pod.id));
        if self.fetch_fails {
            Err("proxy 502".into())
        } else {
            Ok(())
        }
    }
    fn upstream_down(&mut self, pod: &UpPod) -> Result<(), String> {
        self.events.push(format!("down {}", pod.id));
        if self.down_fails {
            Err("HTTP 500".into())
        } else {
            Ok(())
        }
    }
}

fn plan(cells: &[&str]) -> OraclePlan {
    let (lab, stacks) = lab();
    plan_oracle(&lab, &stacks, &request(cells)).expect("plan")
}

/// Every pod that came up was fetched and deleted exactly once, fetch first.
fn all_released(f: &Fake) {
    for i in 1..=f.ups {
        let id = format!("pod{i}");
        assert_eq!(f.count(&format!("fetch {id}")), 1, "{id}: {:?}", f.events);
        assert_eq!(f.count(&format!("down {id}")), 1, "{id}: {:?}", f.events);
        assert!(f.index(&format!("fetch {id}")) < f.index(&format!("down {id}")));
    }
}

#[test]
fn the_driver_waits_for_the_reference_then_runs_and_releases_on_refs_done() {
    let p = plan(&VSA);
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 600.0)].into(),
        pod_after: Some(300.0),
        refs_after: Some(600.0),
        exit_after: Some((1500.0, true)),
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert!(r.ok(), "{:?}", r.problems);
    assert_eq!(r.served, ["fasth3-4step-vsa"]);
    assert_eq!(r.runtime_ok, Some(true));
    assert_eq!(r.upstream.len(), 1);
    assert_eq!(
        r.upstream[0].released,
        "the runtime pod has every reference"
    );
    // Order: up, keepalives while the reference is made, the runtime side
    // with the pod's base, keepalives, release, and nothing after the release.
    assert_eq!(f.events[0], "up fastvideo");
    let start = f
        .events
        .iter()
        .position(|e| e.starts_with("start "))
        .expect("started");
    // The served bases are published (`--url-file`) right before the start.
    assert_eq!(
        f.events[start - 1],
        "urls https://pod1-8000.proxy.runpod.net/upstream/1234567-1024120001-aaaa"
    );
    assert!(f.events[1..start - 1].iter().all(|e| e == "keepalive pod1"));
    assert_eq!(
        start - 2,
        10,
        "a keepalive a minute for the 600 s of the reference"
    );
    assert!(f.events[start].contains(
        "--extra-env FV_ORACLE_URL=https://pod1-8000.proxy.runpod.net/upstream/1234567-1024120001-aaaa"
    ));
    assert!(f.events[start].contains(&format!("--cells {}", VSA.join(","))));
    let fetch = f.index("fetch pod1");
    assert!(start < fetch && fetch + 1 == f.index("down pod1"));
    assert!(f.events[start + 1..fetch]
        .iter()
        .all(|e| e == "keepalive pod1"));
    assert_eq!(
        f.events.len(),
        fetch + 2,
        "nothing after the release: {:?}",
        f.events
    );
    all_released(&f);
    // The run took until the runtime side ended, not the upstream pod's life.
    assert!(f.t >= 600.0 + 1500.0);
}

#[test]
fn the_upstream_pods_go_when_the_runtime_side_ends_without_refs_done() {
    let p = plan(&VSA);
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        pod_after: Some(300.0),
        exit_after: Some((900.0, false)),
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert!(!r.ok());
    assert_eq!(r.runtime_ok, Some(false));
    assert_eq!(r.upstream[0].released, "the runtime side ended");
    assert!(r.problems.iter().any(|x| x.contains("runtime side failed")));
    all_released(&f);
}

#[test]
fn no_runtime_pod_within_the_stock_wait_stops_it_and_releases() {
    let p = plan(&VSA);
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        pod_after: None,
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert!(!r.ok());
    assert!(
        r.problems
            .iter()
            .any(|x| x == "no runtime pod after 1800 s"),
        "{:?}",
        r.problems
    );
    assert_eq!(r.upstream[0].released, "no runtime pod after 1800 s");
    assert!(f.index("abort") < f.index("fetch pod1"));
    all_released(&f);
    let start_t = 0.0;
    assert!(
        f.t >= start_t + 1800.0 && f.t < start_t + 1800.0 + 61.0,
        "{}",
        f.t
    );
}

#[test]
fn a_reference_never_served_releases_without_a_runtime_pod() {
    let mut p = plan(&VSA);
    p.refs_wait_s = 3600;
    let mut f = Fake::default();
    let r = drive(&mut f, &p);
    assert!(!r.ok());
    assert_eq!(r.missing, ["fasth3-4step-vsa"]);
    assert!(!r.runtime_started);
    assert_eq!(f.count("start "), 0);
    assert_eq!(r.upstream[0].released, "no reference was served");
    all_released(&f);
    assert!(f.t >= 3600.0 && f.t < 3661.0);
}

#[test]
fn an_upstream_run_that_ends_without_its_reference_is_not_waited_for() {
    // Two stacks: the LTX reference is served, the H3 run ends without its.
    let p = plan(&[
        VSA[0],
        VSA[1],
        "oracle-ltx25-512p",
        "oracle-ltx25-512p-diff",
    ]);
    let mut f = Fake {
        served_at: [("ltx25-512p", 120.0)].into(),
        finished_at: [("fastvideo", 300.0)].into(),
        pod_after: Some(60.0),
        refs_after: Some(120.0),
        exit_after: Some((600.0, true)),
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert_eq!(r.served, ["ltx25-512p"]);
    assert_eq!(r.missing, ["fasth3-4step-vsa"]);
    assert!(!r.ok(), "a missing reference fails the run");
    // The H3 pod went before the runtime side started; the runtime side ran
    // the LTX arms only, against the LTX pod alone.
    let start = f
        .events
        .iter()
        .position(|e| e.starts_with("start "))
        .unwrap();
    assert!(f.index("down pod1") < start);
    assert!(f.events[start].contains("--cells oracle-ltx25-512p,oracle-ltx25-512p-diff "));
    assert!(f.events[start].contains("FV_ORACLE_URL=https://pod2-8000.proxy.runpod.net/upstream/"));
    assert!(!f.events[start].contains("pod1"));
    assert_eq!(r.upstream[0].released, "none of its references was served");
    assert_eq!(
        r.upstream[1].released,
        "the runtime pod has every reference"
    );
    all_released(&f);
    // Waited until the H3 run ended (300 s), not the refs wait.
    assert!(f.t < 300.0 + 120.0 + 600.0 + 120.0);
}

#[test]
fn a_gone_upstream_pod_drops_its_targets() {
    let p = plan(&VSA);
    let mut f = Fake {
        gone_at: [("fastvideo", 120.0)].into(),
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert_eq!(r.missing, ["fasth3-4step-vsa"]);
    assert_eq!(f.count("start "), 0);
    all_released(&f);
    assert!(f.t < 200.0);
}

#[test]
fn a_second_upstream_pod_that_fails_releases_the_first() {
    let p = plan(&[VSA[0], "oracle-ltx25-512p"]);
    let mut f = Fake {
        up_fails: Some("sol-ltx25"),
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert!(!r.ok());
    assert!(r.problems[0].contains("upstream pod (sol-ltx25) did not come up: no capacity"));
    assert_eq!(
        f.events,
        ["up fastvideo", "up sol-ltx25", "fetch pod1", "down pod1"]
    );
    assert!(!r.runtime_started);
    // The first stack failing creates nothing more.
    let mut f = Fake {
        up_fails: Some("fastvideo"),
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert!(!r.ok());
    assert_eq!(f.events, ["up fastvideo"]);
}

#[test]
fn a_runtime_side_that_does_not_start_releases() {
    let p = plan(&VSA);
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        start_fails: true,
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert!(!r.ok());
    assert!(r
        .problems
        .iter()
        .any(|x| x.contains("did not start: spawn failed")));
    assert_eq!(r.upstream[0].released, "the runtime side did not start");
    all_released(&f);
}

#[test]
fn an_interrupt_stops_the_runtime_side_and_releases() {
    let p = plan(&VSA);
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        pod_after: Some(60.0),
        exit_after: Some((10_000.0, true)),
        interrupt_at: Some(600.0),
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert!(!r.ok());
    assert!(f.index("abort") < f.index("fetch pod1"));
    assert_eq!(r.upstream[0].released, "interrupted");
    all_released(&f);
    // Interrupted while the reference is made: no runtime side at all.
    let mut f = Fake {
        interrupt_at: Some(120.0),
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert!(!r.ok() && !r.runtime_started);
    all_released(&f);
}

#[test]
fn a_failed_upstream_delete_fails_the_run_and_a_failed_fetch_does_not_stop_the_delete() {
    let p = plan(&VSA);
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        pod_after: Some(0.0),
        refs_after: Some(60.0),
        exit_after: Some((120.0, true)),
        fetch_fails: true,
        down_fails: true,
        ..Fake::default()
    };
    let r = drive(&mut f, &p);
    assert_eq!(r.runtime_ok, Some(true));
    assert!(!r.ok());
    assert!(r
        .problems
        .iter()
        .any(|x| x.contains("not deleted: HTTP 500")));
    assert_eq!(r.upstream[0].fetched, Err("proxy 502".into()));
    all_released(&f);
}

#[test]
fn a_hold_keeps_the_upstream_pods_for_another_reader_until_lifted() {
    let (lab, stacks) = lab();
    let mut r = request(&VSA);
    r.hold = true;
    let p = plan_oracle(&lab, &stacks, &r).expect("plan");
    assert!(p.hold);
    assert_eq!(p.hold_limit_s, 14_400 - 900);
    // The runtime pod has every reference at 600 s; the hold is lifted at
    // 1200 s (the other reader done): the release waits for it.
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        pod_after: Some(300.0),
        refs_after: Some(600.0),
        exit_after: Some((3000.0, true)),
        hold_at: Some(1200.0),
        ..Fake::default()
    };
    let rep = drive(&mut f, &p);
    assert!(rep.ok(), "{:?}", rep.problems);
    assert_eq!(
        rep.upstream[0].released,
        "the runtime pod has every reference; the hold is lifted"
    );
    // The URL was published before the runtime side started.
    let urls = f.index("urls https://pod1-8000.proxy.runpod.net/upstream/1234567-1024120001-aaaa");
    let start = f
        .events
        .iter()
        .position(|e| e.starts_with("start "))
        .unwrap();
    assert!(urls < start);
    // Released at the first tick at or after 1200 s, keepalives until then.
    let fetch = f.index("fetch pod1");
    assert_eq!(
        f.events[start + 1..fetch]
            .iter()
            .filter(|e| *e == "keepalive pod1")
            .count(),
        21,
        "a keepalive a minute from 0 s to 1200 s: {:?}",
        f.events
    );
    all_released(&f);
    // The runtime side ends first: the release still waits for the hold.
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        pod_after: Some(60.0),
        exit_after: Some((600.0, true)),
        hold_at: Some(2400.0),
        ..Fake::default()
    };
    let rep = drive(&mut f, &p);
    assert_eq!(
        rep.upstream[0].released,
        "the runtime side ended; the hold is lifted"
    );
    assert!(f.t >= 2400.0 && f.t < 2461.0, "{}", f.t);
    all_released(&f);
    // Never lifted: the hold's limit releases them.
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        pod_after: Some(60.0),
        exit_after: Some((600.0, true)),
        ..Fake::default()
    };
    let rep = drive(&mut f, &p);
    assert_eq!(
        rep.upstream[0].released,
        "the runtime side ended; the hold's limit passed"
    );
    assert!(f.t >= 13_500.0 && f.t < 13_561.0, "{}", f.t);
    all_released(&f);
    // The stock wait does not wait for a hold.
    let mut f = Fake {
        served_at: [("fasth3-4step-vsa", 0.0)].into(),
        ..Fake::default()
    };
    let rep = drive(&mut f, &p);
    assert_eq!(rep.upstream[0].released, "no runtime pod after 1800 s");
    all_released(&f);
}
