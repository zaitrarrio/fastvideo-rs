//! The `fv-deploy` binary against a local fake Runpod (a TCP server on
//! 127.0.0.1): the detached backstop deletes the pod by itself after its
//! parent is gone, `up` refuses below the balance floor before any create,
//! `plan` (and `up` without `--yes`) make no request and never print the key.

#![cfg(feature = "deploy")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fastvideo_deploy::guard::DetachedBackstop;
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

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[test]
fn detached_backstop_deletes_the_pod_on_its_own() {
    let (base, seen) = fake_runpod(20.0);
    let b = DetachedBackstop {
        exe: PathBuf::from(BIN),
        log: None,
    };
    let mut c = b.command(
        ProviderKind::RunpodPod,
        &InstanceId::new("podbs1"),
        now() + 2.0,
    );
    fake_env(&mut c, &base);
    // Started and forgotten: nothing here waits on it or keeps it alive.
    let child = c
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    let pid = child
        .map(|ch| ch.id())
        .unwrap_or_else(|e| panic!("spawn: {e}"));
    assert!(pid > 0);
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
