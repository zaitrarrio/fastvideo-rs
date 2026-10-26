//! The `fv-deploy` binary against a local fake Runpod (a TCP server on
//! 127.0.0.1): the detached backstop deletes the pod by itself after its
//! parent is gone, `up` refuses below the balance floor before any create,
//! `plan` (and `up` without `--yes`) make no request and never print the key.

#![cfg(feature = "deploy")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fastvideo_deploy::guard::{Backstop, DetachedBackstop};
use fastvideo_deploy::provider::{InstanceId, ProviderKind};

const BIN: &str = env!("CARGO_BIN_EXE_fv-deploy");
const KEY: &str = "rpa_SENTINEL_KEY_42";

/// `"<METHOD> <path> <authorization>"` per request.
type Seen = Arc<Mutex<Vec<String>>>;

/// A fake Runpod: GraphQL answers the balance `balance`, `GET /v1/pods/<id>`
/// says RUNNING, `DELETE` succeeds; every request is recorded.
fn fake_runpod(balance: f64) -> (String, Seen) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| panic!("bind: {e}"));
    let base = format!(
        "http://{}",
        l.local_addr().map(|a| a.to_string()).unwrap_or_default()
    );
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let s = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in l.incoming().flatten() {
            let mut rd = BufReader::new(match stream.try_clone() {
                Ok(c) => c,
                Err(_) => continue,
            });
            let mut line = String::new();
            if rd.read_line(&mut line).is_err() {
                continue;
            }
            let (mut len, mut auth) = (0usize, String::new());
            loop {
                let mut h = String::new();
                if rd.read_line(&mut h).is_err() || h.trim().is_empty() {
                    break;
                }
                let lower = h.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap_or(0);
                }
                if lower.starts_with("authorization:") {
                    auth = h["authorization:".len()..].trim().to_owned();
                }
            }
            let mut body = vec![0u8; len];
            let _ = rd.read_exact(&mut body);
            let mut parts = line.split_whitespace();
            let (m, p) = (
                parts.next().unwrap_or("").to_owned(),
                parts.next().unwrap_or("").to_owned(),
            );
            if let Ok(mut g) = s.lock() {
                g.push(format!("{m} {p} {auth}"));
            }
            let reply = if p.starts_with("/graphql") {
                format!("{{\"data\":{{\"myself\":{{\"clientBalance\":{balance},\"currentSpendPerHr\":0}}}}}}")
            } else if m == "GET" && p.starts_with("/v1/pods/") {
                format!("{{\"id\":\"{}\",\"desiredStatus\":\"RUNNING\"}}", &p[9..])
            } else {
                "{}".to_owned()
            };
            let mut out = stream;
            let _ = write!(
                out,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                reply.len()
            );
        }
    });
    (base, seen)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fake_env(c: &mut Command, base: &str) {
    for k in [
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ] {
        c.env_remove(k);
    }
    c.env("NO_PROXY", "127.0.0.1")
        .env("no_proxy", "127.0.0.1")
        .env("RUNPOD_API_KEY", KEY)
        .env("RUNPOD_API_BASE", format!("{base}/v1"))
        .env("RUNPOD_GRAPHQL", format!("{base}/graphql"));
}

fn backstop_with_fake_env(base: &str, tag: &str) -> DetachedBackstop {
    let dir = std::env::temp_dir().join(format!("fv-deploy-armed-{tag}-{}", std::process::id()));
    let mut b = DetachedBackstop::new(PathBuf::from(BIN), None, dir);
    b.ready_timeout = Duration::from_secs(20);
    b.env_remove = [
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    b.env = vec![
        ("NO_PROXY".into(), "127.0.0.1".into()),
        ("no_proxy".into(), "127.0.0.1".into()),
        ("RUNPOD_API_KEY".into(), KEY.into()),
        ("RUNPOD_API_BASE".into(), format!("{base}/v1")),
        ("RUNPOD_GRAPHQL".into(), format!("{base}/graphql")),
    ];
    b
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[test]
fn detached_backstop_deletes_the_pod_on_its_own() {
    let (base, seen) = fake_runpod(20.0);
    let mut b = backstop_with_fake_env(&base, "bs1");
    // The handshake: arm returns once the child wrote its armed file.
    let pid = b
        .arm(
            ProviderKind::RunpodPod,
            &InstanceId::new("podbs1"),
            now() + 3.0,
        )
        .unwrap_or_else(|e| panic!("arm: {e}"));
    assert!(pid > 0);
    // A child that cannot arm (no key) makes arm fail, not succeed.
    b.env_remove.push("RUNPOD_API_KEY".into());
    b.env.retain(|(k, _)| k != "RUNPOD_API_KEY");
    let e = b.arm(
        ProviderKind::RunpodPod,
        &InstanceId::new("podbs2"),
        now() + 3.0,
    );
    assert!(e.is_err(), "{e:?}");
    let t0 = Instant::now();
    let deleted = loop {
        let hit = seen
            .lock()
            .map(|g| {
                g.iter()
                    .find(|l| l.starts_with("DELETE /v1/pods/podbs1"))
                    .cloned()
            })
            .unwrap_or(None);
        if let Some(h) = hit {
            break h;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(60),
            "no DELETE within 60 s: {:?}",
            seen.lock().map(|g| g.clone())
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(deleted.ends_with(&format!("Bearer {KEY}")), "{deleted}");
    assert!(
        t0.elapsed() >= Duration::from_secs(1),
        "deleted before the deadline"
    );
}

#[test]
fn up_refuses_below_the_balance_floor_before_any_create() {
    let (base, seen) = fake_runpod(7.0);
    let mut c = Command::new(BIN);
    c.args([
        "up", "--family", "h3", "--preset", "h3-turbo", "--region", "eu", "--yes", "--cap-s", "1800",
        "--image",
        "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:1111111111111111111111111111111111111111111111111111111111111111",
    ])
    .arg("--root")
    .arg(repo_root())
    .arg("--state-dir")
    .arg(std::env::temp_dir().join(format!("fv-deploy-bin-{}", std::process::id())));
    fake_env(&mut c, &base);
    let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.contains("below the floor"), "{err}");
    assert!(!err.contains(KEY));
    let reqs = seen.lock().map(|g| g.clone()).unwrap_or_default();
    assert!(
        reqs.iter().all(|r| !r.starts_with("POST /v1/pods")),
        "{reqs:?}"
    );
    assert!(
        reqs.iter().any(|r| r.starts_with("POST /graphql")),
        "{reqs:?}"
    );
}

#[test]
fn plan_and_dry_up_make_no_request_and_hide_the_key() {
    let (base, seen) = fake_runpod(20.0);
    for args in [
        vec!["plan", "--family", "h3", "--preset", "h3-turbo"],
        vec![
            "up", "--family", "h3", "--preset", "h3-turbo", "--region", "eu", "--cap-s", "1800",
        ],
    ] {
        let mut c = Command::new(BIN);
        c.args(&args).arg("--root").arg(repo_root());
        fake_env(&mut c, &base);
        let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
        let (so, se) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(out.status.success(), "{args:?}: {se}");
        assert!(!so.contains(KEY) && !se.contains(KEY));
        let v: serde_json::Value =
            serde_json::from_str(&so).unwrap_or_else(|e| panic!("{e}: {so}"));
        let body = &v["requests"][0]["body"];
        assert_eq!(body["networkVolumeId"], "bz0ofkfx45");
        assert_eq!(body["dataCenterIds"][0], "EU-RO-1");
        assert_eq!(body["dockerEntrypoint"][0], "/opt/fastvideo-rs/bin/fv-boot");
        assert_eq!(body["env"]["FV_API_KEYS"], "<sha256 of the run key>");
        assert_eq!(v["launch"]["guards"]["ready_timeout_s"], 600);
    }
    assert!(seen.lock().map(|g| g.is_empty()).unwrap_or(false));
}

#[test]
fn plan_refuses_the_us_volume() {
    let mut c = Command::new(BIN);
    c.args([
        "plan",
        "--family",
        "h3",
        "--preset",
        "h3-turbo",
        "--volume",
        "s2k01690bi",
    ])
    .arg("--root")
    .arg(repo_root());
    let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("US weights volume deleted"));
}

#[test]
fn down_needs_a_kind_for_a_bare_id_and_infers_serverless_from_a_slash() {
    let (base, seen) = fake_runpod(20.0);
    let state = std::env::temp_dir().join(format!("fv-deploy-down-{}", std::process::id()));
    let mut c = Command::new(BIN);
    c.args(["down", "abc123"])
        .arg("--state-dir")
        .arg(&state)
        .arg("--root")
        .arg(repo_root());
    fake_env(&mut c, &base);
    let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("pass --kind"));
    assert!(seen.lock().map(|g| g.is_empty()).unwrap_or(false));
    let mut c = Command::new(BIN);
    c.args(["down", "ep9/tpl9"])
        .arg("--state-dir")
        .arg(&state)
        .arg("--root")
        .arg(repo_root());
    fake_env(&mut c, &base);
    let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let reqs = seen.lock().map(|g| g.clone()).unwrap_or_default();
    assert!(
        reqs.iter()
            .any(|r| r.starts_with("DELETE /v1/endpoints/ep9 ")),
        "{reqs:?}"
    );
    assert!(
        reqs.iter()
            .any(|r| r.starts_with("DELETE /v1/templates/tpl9 ")),
        "{reqs:?}"
    );
    assert!(!reqs.iter().any(|r| r.contains("/pods/")), "{reqs:?}");
}

const VAST_KEY: &str = "vast_SENTINEL_KEY_43";
const GCP_TOKEN: &str = "ya29.SENTINEL_TOKEN_44";

/// `plan --kind vast|gcp`: no request (not even a token mint), no key
/// printed; Vast's search and create, GCP's firewall rule and instance with
/// the file items shortened for display.
#[test]
fn plan_vast_and_gcp_make_no_request_and_hide_the_keys() {
    let (base, seen) = fake_runpod(20.0);
    for (kind, family, preset) in [("vast", "fake", "fake"), ("gcp", "h3", "h3-turbo")] {
        let mut c = Command::new(BIN);
        c.args([
            "plan", "--kind", kind, "--family", family, "--preset", preset,
        ])
        .arg("--root")
        .arg(repo_root());
        fake_env(&mut c, &base);
        c.env("VAST_API_KEY", VAST_KEY)
            .env("VAST_API_BASE", format!("{base}/vast"))
            .env("GCP_ACCESS_TOKEN", GCP_TOKEN)
            .env("GCP_PROJECT", "plan-project")
            .env("FV_GCP_COMPUTE_API", format!("{base}/compute/v1"))
            .env_remove("FV_GCP_SOURCE_CIDR");
        let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
        let (so, se) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(out.status.success(), "{kind}: {se}");
        for k in [KEY, VAST_KEY, GCP_TOKEN] {
            assert!(!so.contains(k) && !se.contains(k), "{kind}");
        }
        let v: serde_json::Value =
            serde_json::from_str(&so).unwrap_or_else(|e| panic!("{e}: {so}"));
        assert_eq!(v["launch"]["provider"], kind);
        let reqs = v["requests"].as_array().cloned().unwrap_or_default();
        assert_eq!(reqs.len(), 2, "{kind}");
        if kind == "vast" {
            assert!(reqs[0]["url"]
                .as_str()
                .unwrap_or("")
                .ends_with("/vast/bundles/"));
            assert_eq!(reqs[0]["body"]["dph_total"]["lte"], 0.6);
            assert_eq!(reqs[1]["method"], "PUT");
            assert_eq!(reqs[1]["body"]["runtype"], "ssh_direct");
            assert_eq!(reqs[1]["body"]["env"]["FV_WEIGHTS_SOURCE"], "none");
        } else {
            assert_eq!(reqs[0]["body"]["sourceRanges"][0], "198.51.100.7/32");
            let inst = &reqs[1]["body"];
            assert_eq!(inst["scheduling"]["maxRunDuration"]["seconds"], "1800");
            assert_eq!(inst["disks"][1]["mode"], "READ_ONLY");
            let items = inst["metadata"]["items"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let start = items
                .iter()
                .find(|i| i["key"] == "startup-script")
                .and_then(|i| i["value"].as_str())
                .unwrap_or_default()
                .to_owned();
            assert!(start.starts_with('<') && start.contains("chars"), "{start}");
            assert!(reqs[1]["url"]
                .as_str()
                .unwrap_or("")
                .contains("/projects/plan-project/zones/europe-west4-b/instances"));
        }
    }
    assert!(seen.lock().map(|g| g.is_empty()).unwrap_or(false));
}

/// `down --kind vast` deletes with the Vast key as a bearer header; `down
/// --kind gcp` refuses a VM without the `fv-owner` label (the fake answers
/// `{}`) and deletes nothing.
#[test]
fn down_on_vast_and_a_foreign_gcp_vm() {
    let (base, seen) = fake_runpod(20.0);
    let state = std::env::temp_dir().join(format!("fv-deploy-down-c3b-{}", std::process::id()));
    let mut c = Command::new(BIN);
    c.args(["down", "4242", "--kind", "vast"])
        .arg("--state-dir")
        .arg(&state)
        .arg("--root")
        .arg(repo_root());
    fake_env(&mut c, &base);
    c.env("VAST_API_KEY", VAST_KEY)
        .env("VAST_API_BASE", format!("{base}/vast"));
    let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let gcp_down = |id: &str, zone: Option<&str>| {
        let mut c = Command::new(BIN);
        c.args(["down", id, "--kind", "gcp"])
            .arg("--state-dir")
            .arg(&state)
            .arg("--root")
            .arg(repo_root());
        if let Some(z) = zone {
            c.args(["--zone", z]);
        }
        fake_env(&mut c, &base);
        c.env("GCP_ACCESS_TOKEN", GCP_TOKEN)
            .env("GCP_PROJECT", "p")
            .env("GCP_ZONE", "europe-west4-b")
            .env("FV_GCP_COMPUTE_API", format!("{base}/compute/v1"));
        let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    // No record, no --zone: refused before any request (never GCP_ZONE).
    let (ok, se) = gcp_down("fv-other", None);
    assert!(!ok && se.contains("pass --zone"), "{se}");
    assert!(!seen
        .lock()
        .map(|g| g.iter().any(|r| r.contains("/compute/")))
        .unwrap_or(true));
    // The zone `up` recorded wins over GCP_ZONE.
    std::fs::create_dir_all(state.join("instances")).unwrap_or_else(|e| panic!("{e}"));
    std::fs::write(
        state.join("instances/fv-other.json"),
        r#"{"instance":{"id":"fv-other","provider":"gcp","name":"fv-other","state":"running","data_center":"europe-west1-c"},"launch":"fv-other"}"#,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let (ok, se) = gcp_down("fv-other", None);
    // The fake answers `{}`: no fv-owner label, so the delete is refused.
    assert!(!ok && se.contains("refusing"), "{se}");
    assert!(!se.contains(GCP_TOKEN));
    // --zone without a record.
    let (ok, _) = gcp_down("fv-third", Some("europe-west2-a"));
    assert!(!ok);
    let reqs = seen.lock().map(|g| g.clone()).unwrap_or_default();
    assert!(
        reqs.iter()
            .any(|r| r == &format!("DELETE /vast/instances/4242/ Bearer {VAST_KEY}")),
        "{reqs:?}"
    );
    for (zone, vm) in [
        ("europe-west1-c", "fv-other"),
        ("europe-west2-a", "fv-third"),
    ] {
        let get =
            format!("GET /compute/v1/projects/p/zones/{zone}/instances/{vm} Bearer {GCP_TOKEN}");
        assert!(reqs.contains(&get), "{reqs:?}");
    }
    assert!(
        !reqs.iter().any(|r| r.contains("/zones/europe-west4-b/")),
        "{reqs:?}"
    );
    assert!(
        !reqs.iter().any(|r| r.starts_with("DELETE /compute")),
        "{reqs:?}"
    );
    let _ = std::fs::remove_dir_all(&state);
}

/// Review 5, 7, 8: a NaN budget is refused; a separator-only
/// FV_GCP_SOURCE_CIDR is unset (never an empty source list); a fixed
/// GCP_ACCESS_TOKEN refuses a cap the token would not outlive. No request.
#[test]
fn gcp_cli_refuses_nan_budgets_empty_ranges_and_long_static_tokens() {
    let (base, seen) = fake_runpod(20.0);
    let run = |args: &[&str], cidr: &str| {
        let mut c = Command::new(BIN);
        c.args(args).arg("--root").arg(repo_root());
        fake_env(&mut c, &base);
        c.env("GCP_ACCESS_TOKEN", GCP_TOKEN)
            .env("GCP_PROJECT", "p")
            .env("FV_GCP_SOURCE_CIDR", cidr)
            .env("FV_GCP_COMPUTE_API", format!("{base}/compute/v1"));
        let out = c.output().unwrap_or_else(|e| panic!("run: {e}"));
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let gcp = ["--kind", "gcp", "--family", "h3", "--preset", "h3-turbo"];
    let mut nan = vec!["plan"];
    nan.extend(gcp);
    nan.extend(["--budget-usd", "nan"]);
    let (ok, _, se) = run(&nan, "203.0.113.1/32");
    assert!(!ok && se.contains("--budget-usd NaN"), "{se}");
    let mut plan = vec!["plan"];
    plan.extend(gcp);
    let (ok, so, se) = run(&plan, " , ");
    assert!(ok, "{se}");
    let v: serde_json::Value = serde_json::from_str(&so).unwrap_or_else(|e| panic!("{e}: {so}"));
    assert_eq!(
        v["requests"][0]["body"]["sourceRanges"],
        serde_json::json!(["198.51.100.7/32"])
    );
    let state = std::env::temp_dir().join(format!("fv-deploy-tok-{}", std::process::id()));
    let state_s = state.display().to_string();
    let image = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:1111111111111111111111111111111111111111111111111111111111111111";
    let mut up = vec!["up"];
    up.extend(gcp);
    up.extend([
        "--yes",
        "--cap-s",
        "3600",
        "--image",
        image,
        "--state-dir",
        state_s.as_str(),
        "--budget-usd",
        "50",
    ]);
    let (ok, _, se) = run(&up, "203.0.113.1/32");
    assert!(!ok && se.contains("outlive the token"), "{se}");
    assert!(!se.contains(GCP_TOKEN));
    assert!(seen.lock().map(|g| g.is_empty()).unwrap_or(false));
    let _ = std::fs::remove_dir_all(&state);
}

/// C3d (ADR-0018 §10): `templates plan` prints the create bodies with no
/// request (no key needed); `reconcile` and `release rollback --local`
/// without a D1 token refuse before any request; `release promote` outside
/// GitHub Actions without a GitHub token refuses to dispatch; a bad channel
/// is refused.
#[test]
fn templates_plan_and_release_refusals_make_no_request() {
    let (base, seen) = fake_runpod(100.0);
    let image = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:1111111111111111111111111111111111111111111111111111111111111111";
    let run = |args: &[&str], env: &[(&str, &str)]| {
        let mut c = Command::new(BIN);
        c.args(args)
            .arg("--root")
            .arg(repo_root())
            .env_remove("RUNPOD_API_KEY")
            .env_remove("FV_CF_API_TOKEN")
            .env_remove("CLOUDFLARE_API_TOKEN")
            .env_remove("CLOUDFLARE_API_KEY")
            .env_remove("GH_TOKEN")
            .env_remove("GITHUB_TOKEN")
            .env_remove("GITHUB_ACTIONS")
            .env("RUNPOD_API_BASE", format!("{base}/v1"))
            .env("FV_REGISTRY_API", &base)
            .env("FV_GITHUB_API", &base)
            .env("FV_D1_API_BASE", format!("{base}/client/v4"));
        for (k, v) in env {
            c.env(k, v);
        }
        c.output().unwrap_or_else(|e| panic!("{e}"))
    };
    let o = run(
        &["templates", "plan", "cuda", image, "--channel", "stable"],
        &[],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let out = String::from_utf8_lossy(&o.stdout);
    let bodies: Vec<serde_json::Value> = serde_json::Deserializer::from_str(&out)
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap_or_else(|e| panic!("{e}: {out}"));
    let names: Vec<&str> = bodies.iter().filter_map(|b| b["name"].as_str()).collect();
    assert_eq!(names, ["fv-serve-cuda-sls", "fv-serve-cuda-pod"]);
    assert_eq!(bodies[0]["env"]["FV_RELEASE_CHANNEL"], "stable");
    let o = run(&["templates", "plan", "gpu", image], &[]);
    assert!(!o.status.success());

    let o = run(&["reconcile", "--dry-run"], &[("RUNPOD_API_KEY", KEY)]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("D1 is the release history"));
    let o = run(&["release", "rollback", "stable", "--local"], &[]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("D1 is the release history"));
    let o = run(&["release", "promote", "abcdef0", "stable"], &[]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("needs GH_TOKEN"));
    let o = run(&["release", "promote", "abcdef0", "Stable", "--local"], &[]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("lower-case letters"));
    let all = format!("{:?}", seen.lock().map(|g| g.clone()).unwrap_or_default());
    assert_eq!(all, "[]", "no request");
}
