//! The request goldens of `fv-deploy templates | release | reconcile` and the
//! D1 `deployments` ledger (docs/adr/0018-fv-deploy.md §10):
//! `fixtures/deploy/release.json` records what `release.sh`,
//! `runpod-templates.sh` and `registry.sh` send for fixed scenarios against
//! `scripts/serve/tests/mock_api.py` (written by
//! `scripts/serve/tests/fv-deploy-goldens.sh`). This test runs the same
//! scenarios, in the same order, with fv-deploy's code against the same mock
//! and asserts the same retags, Runpod writes, D1 writes, dispatches and
//! stdout, normalized the same way (digests as symbols, times as `<now>`).
//! The deployments rows of `fv-deploy up` differ from the scripts' only
//! where §10.3 says. Skipped without python3 (as `boot_r2.rs`), except in CI
//! (`CI` set), where a missing python3 fails.

use std::cell::RefCell;
use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::{json, Value};

use fastvideo_deploy::boot::env::BootEnv;
use fastvideo_deploy::boot::sys::{
    which, Clock, Cmd, Http, LogSink, Method, Output, Request, Response, Spawner, SysError,
};
use fastvideo_deploy::guard;
use fastvideo_deploy::launch::Repo;
use fastvideo_deploy::lifecycle::Recorder;
use fastvideo_deploy::provider::{
    Instance, InstanceId, InstanceState, Launch, LaunchSource, ProviderKind,
};
use fastvideo_deploy::release::registry::{D1Recorder, Registry};
use fastvideo_deploy::release::{self, promote, reconcile, templates};

const REPO: &str = "ghcr.io/zaitrarrio/fastvideo-rs-serve";
const A: &str = "aaaaaaa111111111111111111111111111111111";
const B: &str = "bbbbbbb222222222222222222222222222222222";
const C: &str = "ccccccc333333333333333333333333333333333";
const GPU: &str = "NVIDIA RTX PRO 6000 Blackwell Server Edition";

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    crate_root().join("../..")
}

// ------------------------------------------------------------------ seams

/// HTTP/1.1 over a plain socket (the mock is local, no TLS), one connection
/// per request.
struct PlainHttp;

impl Http for PlainHttp {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        let err = |why: String| SysError::Http {
            url: req.url.clone(),
            why,
        };
        let rest = req
            .url
            .strip_prefix("http://")
            .ok_or_else(|| err("only http:// URLs".into()))?;
        let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
        let mut s = TcpStream::connect(host).map_err(|e| err(e.to_string()))?;
        s.set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| err(e.to_string()))?;
        let body = req.body.clone().unwrap_or_default();
        let mut head = format!(
            "{} /{path} HTTP/1.1\r\nhost: {host}\r\nconnection: close\r\ncontent-length: {}\r\n",
            req.method.as_str(),
            body.len()
        );
        for (k, v) in &req.headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        head.push_str("\r\n");
        s.write_all(head.as_bytes())
            .map_err(|e| err(e.to_string()))?;
        s.write_all(&body).map_err(|e| err(e.to_string()))?;
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).map_err(|e| err(e.to_string()))?;
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| err("no header end".into()))?;
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .ok_or_else(|| err(format!("bad status line {head:?}")))?;
        Ok(Response {
            status,
            body: raw[split + 4..].to_vec(),
        })
    }

    fn download(&self, url: &str, _dest: &Path, _t: Duration) -> Result<u64, SysError> {
        Err(SysError::Http {
            url: url.into(),
            why: "not used".into(),
        })
    }
}

/// `docker buildx imagetools create --tag <repo>:<tag> <repo>@<digest>`:
/// moves the mock's tag (the script goldens' stub does the same), and
/// records the tag and the source.
struct FakeDocker<'a> {
    base: String,
    http: &'a PlainHttp,
    log: RefCell<Vec<Value>>,
}

impl Spawner for FakeDocker<'_> {
    fn output(&self, cmd: &Cmd) -> Result<Output, SysError> {
        Ok(Output {
            code: self.status(cmd)?,
            ..Output::default()
        })
    }
    fn status(&self, cmd: &Cmd) -> Result<Option<i32>, SysError> {
        let a = &cmd.args;
        assert_eq!(a[..4], ["buildx", "imagetools", "create", "--tag"], "{a:?}");
        let (tag, src) = (a[4].clone(), a[5].clone());
        self.log.borrow_mut().push(json!([tag, src]));
        let body = json!({
            "tag": tag.rsplit_once(':').map_or(tag.as_str(), |(_, t)| t),
            "digest": src.rsplit_once('@').map_or(src.as_str(), |(_, d)| d),
        });
        let r = self.http.send(
            &Request::new(
                Method::Post,
                format!("{}/__tag", self.base),
                Duration::from_secs(10),
            )
            .json(body.to_string()),
        )?;
        Ok(Some(if r.ok() { 0 } else { 1 }))
    }
    fn tee(&self, cmd: &Cmd, _log: &Path) -> Result<Option<i32>, SysError> {
        self.status(cmd)
    }
    fn spawn(&self, _cmd: &Cmd) -> Result<u32, SysError> {
        Ok(0)
    }
}

struct Wall;

impl Clock for Wall {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

#[derive(Default)]
struct Lines(RefCell<Vec<String>>);

impl LogSink for Lines {
    fn line(&self, s: &str) {
        self.0.borrow_mut().push(s.to_owned());
    }
}

// ------------------------------------------------------------------ the mock

struct Mock {
    child: Child,
    base: String,
    dir: PathBuf,
}

impl Drop for Mock {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn start_mock() -> Mock {
    let dir = std::env::temp_dir().join(format!("fv-release-goldens-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
    let port = dir.join("port");
    let mut child = Command::new("python3")
        .arg(repo_root().join("scripts/serve/tests/mock_api.py"))
        .arg(&port)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("python3: {e}"));
    let mut base = None;
    for _ in 0..200 {
        if let Ok(p) = std::fs::read_to_string(&port) {
            if !p.trim().is_empty() {
                base = Some(format!("http://127.0.0.1:{}", p.trim()));
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let Some(base) = base else {
        let _ = child.kill();
        let _ = child.wait();
        panic!("the mock did not start");
    };
    Mock { child, base, dir }
}

fn call(base: &str, method: Method, path: &str, body: Option<Value>) -> Value {
    let mut req = Request::new(method, format!("{base}{path}"), Duration::from_secs(30));
    if let Some(b) = body {
        req = req.json(b.to_string());
    }
    if path.starts_with("/v1/") {
        req = req.bearer("test-runpod-key");
    }
    let r = PlainHttp.send(&req).unwrap_or_else(|e| panic!("{e}"));
    serde_json::from_slice(&r.body).unwrap_or(Value::Null)
}

// ------------------------------------------------------------------ normalizing

/// `digest -> symbol` of the builds, longest form first.
fn symbols(builds: &[(&str, &Value)]) -> Vec<(String, String)> {
    let mut full = Vec::new();
    let mut short = Vec::new();
    for (n, d) in builds {
        if let Some(m) = d.as_object() {
            for (k, v) in m {
                let hex = v
                    .as_str()
                    .unwrap_or("")
                    .trim_start_matches("sha256:")
                    .to_owned();
                full.push((format!("sha256:{hex}"), format!("<{n}.{k}>")));
                short.push((hex[..12].to_owned(), format!("<{n}.{k}:12>")));
            }
        }
    }
    full.extend(short);
    full
}

fn normalize(v: Value, syms: &[(String, String)]) -> Value {
    match v {
        Value::String(mut s) => {
            for (d, n) in syms {
                s = s.replace(d, n);
            }
            Value::String(s)
        }
        Value::Number(n) if n.as_f64().is_some_and(|x| x >= 1e12) => json!("<now>"),
        Value::Array(a) => Value::Array(a.into_iter().map(|x| normalize(x, syms)).collect()),
        Value::Object(m) => Value::Object(
            m.into_iter()
                .map(|(k, x)| (k, normalize(x, syms)))
                .collect(),
        ),
        x => x,
    }
}

fn collapse(sql: &str) -> String {
    sql.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One scenario's effects, as fv-deploy-release-goldens.sh records them.
fn effects(
    base: &str,
    since: u64,
    ok: bool,
    stdout: &[String],
    retags: Vec<Value>,
    syms: &[(String, String)],
) -> Value {
    let log = call(base, Method::Get, &format!("/__log?since={since}"), None);
    let reqs = log["requests"].as_array().cloned().unwrap_or_default();
    let p = |r: &Value| r["p"].as_str().unwrap_or("").to_owned();
    let runpod: Vec<Value> = reqs
        .iter()
        .filter(|r| p(r).starts_with("/v1/") && r["m"] != "GET")
        .map(|r| json!([r["m"], p(r), r["body"]]))
        .collect();
    let d1: Vec<Value> = reqs
        .iter()
        .filter(|r| p(r).ends_with("/query"))
        .filter_map(|r| {
            let sql = collapse(r["body"]["sql"].as_str().unwrap_or(""));
            (sql.starts_with("INSERT ") || sql.starts_with("UPDATE "))
                .then(|| json!([sql, r["body"].get("params").cloned().unwrap_or(json!([]))]))
        })
        .collect();
    let dispatch: Vec<Value> = reqs
        .iter()
        .filter(|r| p(r).ends_with("/dispatches"))
        .map(|r| r["body"].clone())
        .collect();
    // One global sequence (the mock logs the docker stub's retags too): the
    // order of retags, Runpod writes, D1 writes and dispatches.
    let order: Vec<String> = reqs
        .iter()
        .filter_map(|r| {
            let path = p(r).split('?').next().unwrap_or("").to_owned();
            let m = r["m"].as_str().unwrap_or("");
            if m == "RETAG" {
                Some(format!("retag {}", r["body"]["tag"].as_str().unwrap_or("")))
            } else if path.starts_with("/v1/") && m != "GET" {
                Some(format!("{m} {path}"))
            } else if path.ends_with("/query") {
                let sql = collapse(r["body"]["sql"].as_str().unwrap_or(""));
                let w: Vec<&str> = sql.split(' ').collect();
                match w.first() {
                    Some(&"INSERT") => Some(format!("d1 INSERT {}", w.get(2).unwrap_or(&""))),
                    Some(&"UPDATE") => Some(format!("d1 UPDATE {}", w.get(1).unwrap_or(&""))),
                    _ => None,
                }
            } else if path.ends_with("/dispatches") {
                Some("dispatch".into())
            } else {
                None
            }
        })
        .collect();
    normalize(
        json!({
            "ok": ok,
            "stdout": stdout.iter().filter(|l| !l.is_empty()).collect::<Vec<_>>(),
            "retags": retags,
            "runpod": runpod,
            "d1": d1,
            "dispatch": dispatch,
            "order": order,
        }),
        syms,
    )
}

fn log_next(base: &str) -> u64 {
    call(base, Method::Get, "/__log", None)["next"]
        .as_u64()
        .unwrap_or(0)
}

// ------------------------------------------------------------------ the test

fn base_env(base: &str) -> Vec<(String, String)> {
    [
        ("FV_D1_API_BASE", format!("{base}/client/v4")),
        ("FV_CF_API_TOKEN", "test-cf-token".into()),
        ("RUNPOD_API_BASE", format!("{base}/v1")),
        ("RUNPOD_GRAPHQL", format!("{base}/graphql")),
        ("RUNPOD_API_KEY", "test-runpod-key".into()),
        ("FV_REGISTRY_API", base.to_owned()),
        ("FV_GITHUB_API", base.to_owned()),
        ("GH_TOKEN", "test-gh-token".into()),
        ("FV_DEPLOYED_BY", "test".into()),
        ("FV_DOCKER", "docker".into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}

/// What one scenario ran: its stdout and whether it succeeded.
type Ran = (bool, Vec<String>);

#[test]
fn release_templates_and_reconcile_send_what_the_scripts_send() {
    if which("python3", std::env::var("PATH").ok().as_deref()).is_none() {
        // In CI a missing python3 must fail, not pass by skipping.
        assert!(
            std::env::var_os("CI").is_none(),
            "no python3 on PATH in CI: the release goldens cannot run"
        );
        eprintln!("skipped: no python3 on PATH");
        return;
    }
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(crate_root().join("fixtures/deploy/release.json"))
            .unwrap_or_else(|e| panic!("{e}")),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let want = fixture["scenarios"].as_array().cloned().unwrap_or_default();
    let mock = start_mock();
    let base = mock.base.clone();
    let http = PlainHttp;
    let repo = Repo::load(&repo_root()).unwrap_or_else(|e| panic!("{e}"));
    let docker = FakeDocker {
        base: base.clone(),
        http: &http,
        log: RefCell::new(Vec::new()),
    };
    let clock = Wall;

    let build = |sha: &str, keys: &[&str]| {
        call(
            &base,
            Method::Post,
            "/__build",
            Some(json!({"sha": sha, "keys": keys})),
        )
    };
    let da = build(A, &["debug", "cuda", "cpu"]);
    let db = build(B, &["debug", "cuda", "cpu"]);
    let dc = build(C, &["debug", "cuda"]);
    let mut tpls = serde_json::Map::new();
    for t in ["cuda-sls", "cuda-pod", "cpu-pod"] {
        tpls.insert(
            format!("tpl-{t}"),
            json!({"id": format!("tpl-{t}"), "name": format!("fv-serve-{t}"), "imageName": format!("{REPO}@sha256:0ld"), "env": {}}),
        );
    }
    call(
        &base,
        Method::Post,
        "/__seed",
        Some(json!({"templates": tpls})),
    );
    let syms = symbols(&[("A", &da), ("B", &db), ("C", &dc)]);
    let reff = |d: &Value, k: &str| format!("{REPO}@{}", d[k].as_str().unwrap_or(""));

    let mut got: Vec<(String, Value)> = Vec::new();
    // Runs one scenario with extra env; records its effects.
    let mut scenario =
        |name: &str,
         extra: &[(&str, &str)],
         f: &dyn Fn(&release::Ctx<'_>) -> Result<(), release::ReleaseError>| {
            let since = log_next(&base);
            let n = docker.log.borrow().len();
            let mut pairs = base_env(&base);
            pairs.extend(
                extra
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned())),
            );
            let env = BootEnv::from_pairs(pairs);
            let (log, out) = (Lines::default(), Lines::default());
            let ctx = release::Ctx::new(&repo, &http, &docker, &clock, &env, &log, &out);
            let r = f(&ctx);
            if let Err(e) = &r {
                eprintln!("{name}: {e}");
            }
            let ran: Ran = (r.is_ok(), out.0.borrow().clone());
            let retags = docker.log.borrow()[n..].to_vec();
            got.push((
                name.to_owned(),
                effects(&base, since, ran.0, &ran.1, retags, &syms),
            ));
        };
    let local = promote::Options {
        mode: Some(promote::Mode::Local),
        ..promote::Options::default()
    };
    macro_rules! sc {
        ($name:expr, $extra:expr, |$c:ident| $body:expr) => {
            scenario($name, $extra, &|$c: &release::Ctx<'_>| $body)
        };
    }
    type O = promote::Options;

    // ---- release promote / rollback (the order of fv-deploy-release-goldens.sh)
    sc!("promote-dry-run", &[], |c| promote::promote(
        c,
        "aaaaaaa",
        "stable",
        &O {
            dry_run: true,
            ..local.clone()
        }
    ));
    sc!("promote-dispatch", &[], |c| promote::promote(
        c,
        "aaaaaaa",
        "stable",
        &O {
            notes: Some("first stable".into()),
            ..O::default()
        }
    ));
    sc!("promote", &[], |c| promote::promote(
        c,
        "aaaaaaa",
        "stable",
        &O {
            notes: Some("first stable".into()),
            ..local.clone()
        }
    ));
    sc!("promote-again", &[], |c| promote::promote(
        c, "aaaaaaa", "stable", &local
    ));
    sc!("promote-latest", &[], |c| promote::promote(
        c, "bbbbbbb", "latest", &local
    ));
    sc!("promote-missing-image", &[], |c| promote::promote(
        c, "ccccccc", "stable", &local
    ));
    let b_cuda = reff(&db, "cuda");
    sc!("promote-digest", &[], |c| promote::promote(
        c,
        &b_cuda,
        "stable",
        &O {
            templates: false,
            ..local.clone()
        }
    ));
    sc!("rollback-dry-run", &[], |c| promote::rollback(
        c,
        "stable",
        &O {
            dry_run: true,
            ..local.clone()
        }
    ));
    sc!("rollback", &[], |c| promote::rollback(c, "stable", &local));
    sc!("rollback-again", &[], |c| promote::rollback(
        c, "stable", &local
    ));
    sc!("rollback-to", &[], |c| promote::rollback(
        c,
        "stable",
        &O {
            to: Some(3),
            ..local.clone()
        }
    ));
    sc!("rollback-dispatch", &[], |c| promote::rollback(
        c,
        "stable",
        &O {
            to: Some(1),
            ..O::default()
        }
    ));
    // A release from before the one CUDA image (variant keys, no `cuda`), in jq's key order.
    let old = format!(
        "{{\"debug\":\"{}\",\"h3-turbo\":\"{}\",\"cpu\":\"{}\"}}",
        reff(&da, "debug"),
        reff(&da, "cuda"),
        reff(&da, "cpu")
    );
    call(
        &base,
        Method::Post,
        "/__sql",
        Some(
            json!({"sql": format!("INSERT INTO releases (channel, git_sha, digests, action, promoted_at, promoted_by) VALUES ('stable', '0000000aaaa', '{old}', 'promote', 1, 'test')")}),
        ),
    );
    let old_id = call(
        &base,
        Method::Post,
        "/__sql",
        Some(json!({"sql": "SELECT MAX(id) AS n FROM releases"})),
    )[0]["n"]
        .as_i64()
        .unwrap_or(0);
    sc!("rollback-partial-refused", &[], |c| promote::rollback(
        c,
        "stable",
        &O {
            to: Some(old_id),
            ..local.clone()
        }
    ));
    sc!("rollback-partial", &[], |c| promote::rollback(
        c,
        "stable",
        &O {
            to: Some(old_id),
            allow_partial: true,
            ..local.clone()
        }
    ));
    sc!("rollback-back", &[], |c| promote::rollback(
        c,
        "stable",
        &O {
            to: Some(3),
            ..local.clone()
        }
    ));

    // ---- templates
    fn sync(
        c: &release::Ctx<'_>,
        id: &str,
        image: &str,
        ch: &str,
    ) -> Result<(), release::ReleaseError> {
        let api = c.runpod()?;
        templates::sync(c, &api, id, image, Some(ch))
    }
    sc!("templates-same", &[], |c| sync(
        c, "cuda", &b_cuda, "stable"
    ));
    call(&base, Method::Delete, "/v1/templates/tpl-cpu-pod", None);
    let b_cpu = reff(&db, "cpu");
    sc!("templates-create", &[], |c| sync(
        c, "cpu", &b_cpu, "stable"
    ));
    let a_cuda = reff(&da, "cuda");
    let auth = [("RUNPOD_REGISTRY_AUTH_ID", "auth-golden")];
    sc!("templates-update", &auth, |c| sync(
        c, "cuda", &a_cuda, "latest"
    ));
    sc!("templates-unknown", &auth, |c| sync(
        c, "gpu", &a_cuda, "latest"
    ));
    let tag = format!("{REPO}:cuda-stable");
    sc!("templates-tag", &auth, |c| sync(c, "cuda", &tag, "latest"));

    // ---- reconcile
    call(
        &base,
        Method::Post,
        "/__seed",
        Some(json!({
        "pods": {
            "stray1": {"id": "stray1", "name": "fv-serve-smoke-x", "imageName": reff(&db, "cuda"),
                "desiredStatus": "RUNNING", "createdAt": "2026-09-29 01:22:30.836 +0000 UTC", "costPerHr": 0.5,
                "machine": {"gpuDisplayName": "RTX PRO 6000", "dataCenterId": "EU-RO-1"}, "env": {"FV_ADMIN_TOKEN": "never-print-me"}},
            "other1": {"id": "other1", "name": "someone-else", "imageName": "daydreamlive/scope", "desiredStatus": "RUNNING"}},
        "endpoints": {"ep1": {"id": "ep1", "name": "fv-serve-ep-x", "templateId": "tpl-cuda-sls",
            "createdAt": "2026-09-30 10:00:00.000 +0000 UTC", "gpuTypeIds": [GPU], "dataCenterIds": ["EU-RO-1"]}}})),
    );
    call(
        &base,
        Method::Post,
        "/__sql",
        Some(
            json!({"sql": "INSERT INTO deployments (id, kind, runpod_id, created_at, updated_at, created_by, status) VALUES ('pod:ghost', 'pod', 'ghost', 1, 1, 'test', 'ready')"}),
        ),
    );
    type R = reconcile::Options;
    sc!("reconcile-dry-run", &[], |c| reconcile::reconcile(
        c,
        &R {
            dry_run: true,
            ..R::default()
        }
    )
    .map(|_| ()));
    sc!("reconcile-adopt", &[], |c| reconcile::reconcile(
        c,
        &R {
            adopt: true,
            ..R::default()
        }
    )
    .map(|_| ()));
    call(
        &base,
        Method::Post,
        "/__seed",
        Some(
            json!({"pods": {"stray1": {"id": "stray1", "name": "fv-serve-smoke-x", "imageName": reff(&da, "cuda"), "desiredStatus": "RUNNING"}}}),
        ),
    );
    sc!("reconcile-drift", &[], |c| reconcile::reconcile(
        c,
        &R::default()
    )
    .map(|_| ()));
    sc!("reconcile-fix", &[], |c| reconcile::reconcile(
        c,
        &R {
            fix: true,
            ..R::default()
        }
    )
    .map(|_| ()));

    // ---- the deployments rows of fv-deploy up (create, ready, delete)
    fn record(
        c: &release::Ctx<'_>,
        launch: &Launch,
        inst: &Instance,
    ) -> Result<(), release::ReleaseError> {
        let d1 = c
            .d1()
            .ok_or_else(|| release::ReleaseError::Refused("no D1".into()))?;
        let reg = Registry::new(&d1, c.clock, c.env);
        let r = D1Recorder { registry: &reg };
        r.created(launch, inst)
            .map_err(release::ReleaseError::Refused)?;
        r.ready(inst).map_err(release::ReleaseError::Refused)?;
        r.deleted(inst.provider, &inst.id, "deleted")
            .map_err(release::ReleaseError::Refused)
    }
    let (pod_l, pod_i) = up_instance(
        ProviderKind::RunpodPod,
        &b_cuda,
        "fv-serve-smoke-1025120000",
        "pod-golden-1",
    );
    sc!("registry-pod", &[], |c| record(c, &pod_l, &pod_i));
    let (ep_l, ep_i) = up_instance(
        ProviderKind::RunpodServerless,
        &b_cuda,
        "fv-serve-ep-1025120000",
        "ep-golden-1/tpl-golden-1",
    );
    sc!("registry-endpoint", &[], |c| record(c, &ep_l, &ep_i));

    // ---- partial failures (fv-deploy-release-goldens.sh's order)
    call(
        &base,
        Method::Post,
        "/__fail",
        Some(json!({"m": "PATCH", "p": "/v1/templates/", "status": 500, "times": 1})),
    );
    sc!("promote-template-fails", &[], |c| promote::promote(
        c, "aaaaaaa", "stable", &local
    ));
    call(
        &base,
        Method::Post,
        "/__fail",
        Some(
            json!({"m": "POST", "p": "/client/v4/", "sql": "INSERT INTO releases", "status": 500, "times": 1}),
        ),
    );
    sc!("promote-insert-fails", &[], |c| promote::promote(
        c, "aaaaaaa", "canary", &local
    ));

    // ---- compare
    let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
    let want_names: Vec<&str> = want.iter().filter_map(|s| s["name"].as_str()).collect();
    assert_eq!(
        names, want_names,
        "the scenarios of fv-deploy-release-goldens.sh, in order"
    );
    for ((name, g), w) in got.iter().zip(&want) {
        let mut w = w.clone();
        if let Some(m) = w.as_object_mut() {
            for k in ["name", "prog", "args", "env"] {
                m.remove(k);
            }
        }
        let g = if name.starts_with("registry-") {
            up_rows_as_script_rows(g.clone(), &w)
        } else {
            g.clone()
        };
        let g = scripts_program(g);
        assert_eq!(
            g, w,
            "scenario {name}: fv-deploy (left) vs the script (right)"
        );
    }
    // Review fix (#199): the row names the channel the launch follows
    // (`up --channel`, its FV_RELEASE_CHANNEL), not this process's.
    {
        let env = BootEnv::from_pairs(base_env(&base));
        let (log, out) = (Lines::default(), Lines::default());
        let ctx = release::Ctx::new(&repo, &http, &docker, &clock, &env, &log, &out);
        let (mut l, i) = up_instance(
            ProviderKind::RunpodPod,
            &b_cuda,
            "fv-deploy-h3-turbo-1",
            "pod-chan-1",
        );
        l.env.insert("FV_RELEASE_CHANNEL".into(), "canary".into());
        let d1 = ctx.d1().unwrap_or_else(|| panic!("no D1"));
        let reg = Registry::new(&d1, &clock, &env);
        D1Recorder { registry: &reg }
            .created(&l, &i)
            .unwrap_or_else(|e| panic!("{e}"));
        let row = call(
            &base,
            Method::Post,
            "/__sql",
            Some(
                json!({"sql": "SELECT channel, created_by FROM deployments WHERE id = 'pod:pod-chan-1'"}),
            ),
        );
        assert_eq!(row[0]["channel"], "canary", "{row}");
        assert_eq!(row[0]["created_by"], "test/fv-deploy", "{row}");
    }
    let all = got.iter().map(|(_, v)| v.to_string()).collect::<String>();
    for s in [
        "test-cf-token",
        "test-runpod-key",
        "test-gh-token",
        "never-print-me",
    ] {
        assert!(!all.contains(s), "{s} in the requests");
    }
}

/// `created_by` is `<actor>/fv-deploy`; the scripts write their own name
/// (`release.sh`, `runpod-pod.sh`, `runpod-endpoint.sh`): the one intended
/// difference in every deployments row (ADR-0018 §10.3).
fn scripts_program(v: Value) -> Value {
    match v {
        Value::String(s) if s.ends_with("/fv-deploy") => {
            Value::String(s.replace("/fv-deploy", "/release.sh"))
        }
        Value::Array(a) => Value::Array(a.into_iter().map(scripts_program).collect()),
        Value::Object(m) => Value::Object(
            m.into_iter()
                .map(|(k, x)| (k, scripts_program(x)))
                .collect(),
        ),
        x => x,
    }
}

/// The created row of `fv-deploy up` as the script's: the program word as
/// the script's, the `region` column dropped (fv-deploy adds it: `eu`) and
/// `meta` as the script's (fv-deploy's has the same keys plus `guards`, the
/// launch's guards with "not checked" for a `null`). Asserts those
/// differences are exactly what fv-deploy adds.
fn up_rows_as_script_rows(mut g: Value, w: &Value) -> Value {
    let script_by = w["d1"][0][1][5].as_str().unwrap_or("").to_owned();
    let Some(insert) = g["d1"].get_mut(0) else {
        return g;
    };
    let sql = insert[0].as_str().unwrap_or("").to_owned();
    let cols: Vec<String> = sql
        .split_once('(')
        .and_then(|(_, r)| r.split_once(')'))
        .map(|(c, _)| c.split(", ").map(str::to_owned).collect())
        .unwrap_or_default();
    let params = insert[1].as_array().cloned().unwrap_or_default();
    let mut keep_cols = Vec::new();
    let mut keep_params = Vec::new();
    for (c, p) in cols.iter().zip(params) {
        match c.as_str() {
            "region" => assert_eq!(p, "eu", "fv-deploy's added region"),
            "created_by" => {
                assert!(p.as_str().is_some_and(|s| s.ends_with("/fv-deploy")), "{p}");
                keep_cols.push(c.clone());
                keep_params.push(json!(script_by));
            }
            "meta" => {
                let mut m: Value =
                    serde_json::from_str(p.as_str().unwrap_or("")).unwrap_or(Value::Null);
                let guards = m.as_object_mut().and_then(|o| o.remove("guards"));
                assert_eq!(
                    guards,
                    Some(
                        json!({"max_usd_per_hr": 2.5, "min_balance_usd": 8.0, "cap_s": 1800, "ready_timeout_s": 600, "ready_cycles": 3})
                    ),
                    "the launch's guards in meta"
                );
                // The script's meta text, if it is the same object.
                let ws = w["d1"][0][1]
                    .as_array()
                    .and_then(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .find(|s| s.starts_with('{'))
                    })
                    .unwrap_or("")
                    .to_owned();
                let wm: Value = serde_json::from_str(&ws).unwrap_or(Value::Null);
                assert_eq!(m, wm, "meta (without guards) as the script's");
                keep_cols.push(c.clone());
                keep_params.push(json!(ws));
            }
            _ => {
                keep_cols.push(c.clone());
                keep_params.push(p);
            }
        }
    }
    let set = sql
        .split_once(" DO UPDATE SET ")
        .map(|(_, s)| s)
        .unwrap_or("");
    let set: Vec<&str> = set
        .split(", ")
        .filter(|x| !x.starts_with("region = "))
        .collect();
    let n = keep_cols.len();
    *insert = json!([
        format!(
            "INSERT INTO deployments ({}) VALUES ({}) ON CONFLICT (id) DO UPDATE SET {}",
            keep_cols.join(", "),
            vec!["?"; n].join(", "),
            set.join(", ")
        ),
        keep_params
    ]);
    g
}

/// The launch and the created instance of `fv-deploy up` matching the
/// inputs of the script goldens' registry calls.
fn up_instance(kind: ProviderKind, image: &str, name: &str, id: &str) -> (Launch, Instance) {
    let mut l = fastvideo_deploy::launch::script_pod(&fastvideo_deploy::launch::ScriptPod {
        name,
        image,
        gpu: GPU,
        key_hash: "h",
        config: "/etc/fv/runpod.toml",
        inline: None,
        volume: Some(("bz0ofkfx45", "EU-RO-1")),
        guards: guard::default_guards(),
    });
    l.provider = kind;
    l.placement.region = Some("eu".into());
    l.source = Some(LaunchSource {
        family: "h3".into(),
        preset: "h3-turbo".into(),
        image: "cuda".into(),
        config: "/etc/fv/runpod.toml".into(),
    });
    let inst = Instance {
        id: InstanceId::new(id),
        provider: kind,
        name: name.to_owned(),
        state: InstanceState::Running,
        url: None,
        gpu: (kind == ProviderKind::RunpodPod).then(|| GPU.to_owned()),
        data_center: Some("EU-RO-1".into()),
        usd_per_hr: (kind == ProviderKind::RunpodPod).then_some(1.84),
        created_at: None,
    };
    (l, inst)
}
