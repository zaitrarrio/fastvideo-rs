//! `fv-lab oracle`'s world, [`LiveOps`] (docs/adr/0019-fv-lab.md §19), on
//! fakes: a provider (fv-deploy's `Provider`, its creates and deletes
//! recorded), an HTTP of fixed answers and a fake `fv-lab` executable for
//! the runtime side. What the review of #216 asked to hold: a pod whose
//! records cannot be written is deleted; a stopped runtime side gets SIGTERM
//! and is waited for; a failed one's pod is deleted; a reused tag is refused;
//! the URL and hold files; a reference served only with its dump.
#![cfg(unix)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use fastvideo_deploy::boot::sys::{Clock, Http, LogSink, RealClock, Request, Response, SysError};
use fastvideo_deploy::guard::{Backstop, GuardError};
use fastvideo_deploy::provider::{
    Account, ApiCall, Cursor, Instance, InstanceId, InstanceState, Launch, LogPage, Offer,
    OfferQuery, Probe, Provider, ProviderError, ProviderKind, Reach, ReachOptions,
};
use fastvideo_lab::oracle::{
    plan_oracle, OracleOps, OraclePlan, OracleRequest, RefState, Stacks, UpPod, STOCK_WAIT_S,
};
use fastvideo_lab::oracle_live::LiveOps;
use fastvideo_lab::plan::Lab;

const UP_SHA: &str = "1234567890abcdef1234567890abcdef12345678";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("oracle-live-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn plan(tag: &str) -> OraclePlan {
    let root = repo_root();
    let lab = Lab::load(&root).unwrap();
    let stacks = Stacks::load(&root).unwrap();
    plan_oracle(
        &lab,
        &stacks,
        &OracleRequest {
            tier: "oracle".into(),
            sha: "1234567".into(),
            up_sha: UP_SHA.into(),
            image_tag: "latest".into(),
            cells: vec![
                "oracle-fasth3-4step-vsa".into(),
                "oracle-fasth3-4step-vsa-diff".into(),
            ],
            extra_env: Vec::new(),
            runtime_tag: tag.into(),
            upstream_tags: vec![format!("{tag}-up")],
            runtime_cap_s: None,
            upstream_cap_s: None,
            stock_wait_s: STOCK_WAIT_S,
            refs_wait_s: None,
            hold: false,
        },
    )
    .unwrap()
}

// ------------------------------------------------------------------ fakes

#[derive(Default)]
struct FakeProvider {
    created: RefCell<Vec<String>>,
    destroyed: RefCell<Vec<String>>,
}

struct Ready;
impl Reach for Ready {
    fn ping(&mut self) -> Probe {
        Probe::Ready
    }
    fn call(
        &mut self,
        _m: fastvideo_deploy::boot::sys::Method,
        _p: &str,
        _b: Option<&Value>,
        _s: &dyn Fn() -> bool,
    ) -> Result<(u16, Value), ProviderError> {
        Ok((200, Value::Null))
    }
}

impl Provider for FakeProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::RunpodPod
    }
    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>, ProviderError> {
        Ok(q.gpu_types
            .iter()
            .map(|g| Offer {
                provider: self.kind(),
                gpu_type: g.clone(),
                data_center: q.data_center.clone(),
                cloud: q.cloud,
                usd_per_hr: Some(2.0),
                stock: None,
            })
            .collect())
    }
    fn payload(&self, _l: &Launch) -> Result<Vec<ApiCall>, ProviderError> {
        Ok(Vec::new())
    }
    fn create(&self, l: &Launch) -> Result<Instance, ProviderError> {
        let id = format!("up{}", self.created.borrow().len() + 1);
        self.created.borrow_mut().push(id.clone());
        Ok(Instance {
            id: InstanceId::new(id),
            provider: self.kind(),
            name: l.name.clone(),
            state: InstanceState::Pending,
            url: None,
            gpu: l.gpu.types.first().cloned(),
            data_center: None,
            usd_per_hr: Some(2.0),
            created_at: Some(RealClock.now()),
            memory_gb: Some(283.0),
            vcpus: Some(28.0),
        })
    }
    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError> {
        Ok(
            if self.destroyed.borrow().iter().any(|d| d == id.as_str()) {
                InstanceState::Terminated
            } else {
                InstanceState::Running
            },
        )
    }
    fn logs(&self, _id: &InstanceId, _c: Option<&Cursor>) -> Result<LogPage, ProviderError> {
        Ok(LogPage::default())
    }
    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError> {
        self.destroyed.borrow_mut().push(id.to_string());
        Ok(())
    }
    fn account(&self) -> Result<Account, ProviderError> {
        Ok(Account {
            provider: self.kind(),
            balance_usd: 100.0,
            spend_per_hr_usd: None,
        })
    }
    fn reach<'b>(
        &'b self,
        _i: &Instance,
        _o: &ReachOptions<'_>,
    ) -> Result<Box<dyn Reach + 'b>, ProviderError> {
        Ok(Box::new(Ready))
    }
    fn sweep(
        &self,
        _name: &str,
        _keep: Option<&InstanceId>,
    ) -> Result<Vec<InstanceId>, ProviderError> {
        Ok(Vec::new())
    }
    fn bills_at_create(&self) -> bool {
        true
    }
}

struct NoBackstop;
impl Backstop for NoBackstop {
    fn arm(&self, _k: ProviderKind, _id: &InstanceId, _d: f64) -> Result<u32, GuardError> {
        Ok(0)
    }
}

/// Fixed answers by URL (404 otherwise); requests recorded.
#[derive(Default)]
struct FakeHttp {
    answers: BTreeMap<String, (u16, String)>,
    asked: RefCell<Vec<String>>,
}
impl Http for FakeHttp {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        self.asked.borrow_mut().push(req.url.clone());
        let (status, body) = self
            .answers
            .get(&req.url)
            .cloned()
            .unwrap_or((404, String::new()));
        Ok(Response {
            status,
            body: body.into_bytes(),
        })
    }
    fn download(&self, url: &str, dest: &Path, _t: Duration) -> Result<u64, SysError> {
        match self.answers.get(url) {
            Some((200, b)) => {
                std::fs::write(dest, b).map_err(|e| SysError::io(dest, e))?;
                Ok(b.len() as u64)
            }
            _ => Err(SysError::Status {
                url: url.into(),
                status: 404,
            }),
        }
    }
}

struct Quiet;
impl LogSink for Quiet {
    fn line(&self, _s: &str) {}
}

/// A fake `fv-lab`: records its arguments, writes the runtime run's
/// `pod.json` (as `pod up` does once its pod is ready), then exits with
/// `exit` at once, or (`exit` None) waits for SIGTERM and exits 7.
fn fake_fv_lab(dir: &Path, run_dir: &Path, exit: Option<i32>) -> PathBuf {
    let exe = dir.join("fv-lab");
    let pod = r#"{"id":"rt1","name":"fv-lab-oracle-x","family":"oracle","tag":"t","sha":"1234567","image":"i","url":"https://rt1-8000.proxy.runpod.net","cap_s":14400}"#;
    let tail = match exit {
        Some(c) => format!("exit {c}\n"),
        None => "while :; do sleep 0.1; done\n".to_owned(),
    };
    std::fs::write(
        &exe,
        format!(
            "#!/bin/bash\ntrap 'echo term >>{d}/mark; exit 7' TERM\necho \"$@\" >{d}/args\nmkdir -p {r}\nprintf '%s' '{pod}' >{r}/pod.json\n{tail}",
            d = dir.display(),
            r = run_dir.display(),
        ),
    )
    .unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    exe
}

/// Run `f` on a LiveOps over the fakes.
#[allow(clippy::too_many_arguments)]
fn with_ops<R>(
    root: &Path,
    plan: &OraclePlan,
    provider: &FakeProvider,
    http: &FakeHttp,
    exe: PathBuf,
    url_file: Option<PathBuf>,
    hold: Option<PathBuf>,
    f: impl FnOnce(&mut LiveOps<'_>) -> R,
) -> R {
    let pod_url = |id: &str| format!("https://{id}-8000.proxy.runpod.net");
    let list = || -> Result<Vec<fastvideo_deploy::provider::runpod::PodListing>, String> {
        Ok(Vec::new())
    };
    let resolve = |image: &str| -> Result<String, String> { Ok(format!("{image}@sha256:00")) };
    let never = || false;
    let leaks = |_: &str| false;
    let mut ops = LiveOps {
        root: root.to_path_buf(),
        state: root.join("state"),
        plan,
        http,
        provider,
        pod_url: &pod_url,
        list: &list,
        resolve: &resolve,
        backstop: &NoBackstop,
        clock: &RealClock,
        log: &Quiet,
        interrupted: &never,
        leaks: &leaks,
        exe,
        url_file,
        hold,
        poll: Duration::from_millis(10),
        child: None,
        exit: None,
        code: None,
    };
    f(&mut ops)
}

// ------------------------------------------------------------------ tests

#[test]
fn an_upstream_pod_is_recorded_and_returned() {
    let root = scratch("up-ok");
    let p = plan("1234567-1026000001-0001");
    let prov = FakeProvider::default();
    let up = with_ops(
        &root,
        &p,
        &prov,
        &FakeHttp::default(),
        root.join("x"),
        None,
        None,
        |o| o.upstream_up(&p.upstream[0]),
    )
    .expect("up");
    assert_eq!(up.id, "up1");
    assert_eq!(up.url, "https://up1-8000.proxy.runpod.net");
    assert!(prov.destroyed.borrow().is_empty());
    let rec = std::fs::read_to_string(
        root.join("artifacts/runpod/upstream/1234567-1026000001-0001-up/pod.json"),
    )
    .unwrap();
    assert!(
        rec.contains("\"min_ram_gb\": 192") && rec.contains("@sha256:00"),
        "{rec}"
    );
    assert!(root.join("state/pods/up1.json").is_file());
    assert!(std::fs::read_to_string(root.join("state/ledger.tsv"))
        .unwrap()
        .contains("runpod-pod-created up1"));
}

/// B2 of the review: the pod exists once `up` returned; a record that cannot
/// be written must not leave it to the backstop.
#[test]
fn an_upstream_pod_whose_record_cannot_be_written_is_deleted() {
    let root = scratch("up-record");
    let p = plan("1234567-1026000002-0002");
    // A file where the run directory goes: its pod.json cannot be written.
    let dir = root.join("artifacts/runpod/upstream");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("1234567-1026000002-0002-up"), "in the way").unwrap();
    let prov = FakeProvider::default();
    let e = with_ops(
        &root,
        &p,
        &prov,
        &FakeHttp::default(),
        root.join("x"),
        None,
        None,
        |o| o.upstream_up(&p.upstream[0]),
    )
    .unwrap_err();
    assert_eq!(*prov.created.borrow(), ["up1"]);
    assert_eq!(*prov.destroyed.borrow(), ["up1"], "{e}");
    assert!(e.contains("(pod up1 deleted)"), "{e}");
}

/// SF3 / SF9 of the review: `runtime_stop` sends SIGTERM (the child, `pod
/// up`, deletes what it made on it) and `runtime_wait` waits for it; a
/// failed runtime side whose pod is still up has it deleted (SF4).
#[test]
fn a_stopped_runtime_side_gets_sigterm_and_its_pod_is_deleted() {
    let root = scratch("rt-stop");
    let tag = "1234567-1026000003-0003";
    let p = plan(tag);
    let run = root.join("artifacts/runpod/oracle").join(tag);
    let exe = fake_fv_lab(&root, &run, None);
    let prov = FakeProvider::default();
    let (ok, code, exit_again) = with_ops(
        &root,
        &p,
        &prov,
        &FakeHttp::default(),
        exe,
        None,
        None,
        |o| {
            o.runtime_start(&["pod".into(), "up".into()]).unwrap();
            let t0 = RealClock.now();
            while o.runtime_pod().is_none() && RealClock.now() - t0 < 10.0 {
                RealClock.sleep(Duration::from_millis(50));
            }
            assert_eq!(o.runtime_pod().map(|r| r.id), Some("rt1".into()));
            assert_eq!(o.runtime_exit(), None, "still running");
            o.runtime_stop();
            let ok = o.runtime_wait(Duration::from_secs(10));
            (ok, o.runtime_exit_code(), o.runtime_exit())
        },
    );
    let mark = std::fs::read_to_string(root.join("mark")).unwrap_or_default();
    assert_eq!(mark.trim(), "term", "the child got SIGTERM");
    assert_eq!((ok, code, exit_again), (Some(false), Some(7), Some(false)));
    // The child's own pod, still live after its failure: deleted.
    assert_eq!(*prov.destroyed.borrow(), ["rt1"]);
    // It ran with this run's root and state.
    let args = std::fs::read_to_string(root.join("args")).unwrap();
    assert!(
        args.starts_with("pod up --root ") && args.contains(" --state-dir "),
        "{args}"
    );
}

#[test]
fn a_runtime_side_that_ends_well_is_left_alone() {
    let root = scratch("rt-ok");
    let tag = "1234567-1026000004-0004";
    let p = plan(tag);
    let run = root.join("artifacts/runpod/oracle").join(tag);
    let exe = fake_fv_lab(&root, &run, Some(0));
    let prov = FakeProvider::default();
    let (ok, code) = with_ops(
        &root,
        &p,
        &prov,
        &FakeHttp::default(),
        exe,
        None,
        None,
        |o| {
            o.runtime_start(&["pod".into()]).unwrap();
            (
                o.runtime_wait(Duration::from_secs(10)),
                o.runtime_exit_code(),
            )
        },
    );
    assert_eq!((ok, code), (Some(true), Some(0)));
    assert!(prov.destroyed.borrow().is_empty());
    // A failure (exit 3) with its pod up: the pod is deleted.
    let root = scratch("rt-fail");
    let run = root.join("artifacts/runpod/oracle").join(tag);
    let exe = fake_fv_lab(&root, &run, Some(3));
    let prov = FakeProvider::default();
    let ok = with_ops(
        &root,
        &p,
        &prov,
        &FakeHttp::default(),
        exe,
        None,
        None,
        |o| {
            o.runtime_start(&["pod".into()]).unwrap();
            o.runtime_wait(Duration::from_secs(10))
        },
    );
    assert_eq!(ok, Some(false));
    assert_eq!(*prov.destroyed.borrow(), ["rt1"]);
}

#[test]
fn a_reused_runtime_tag_is_refused() {
    let root = scratch("rt-reused");
    let tag = "1234567-1026000005-0005";
    let p = plan(tag);
    std::fs::create_dir_all(root.join("artifacts/runpod/oracle").join(tag)).unwrap();
    let prov = FakeProvider::default();
    let e = with_ops(
        &root,
        &p,
        &prov,
        &FakeHttp::default(),
        root.join("none"),
        None,
        None,
        |o| o.runtime_start(&["pod".into()]),
    )
    .unwrap_err();
    assert!(e.contains("was used before"), "{e}");
}

#[test]
fn the_url_file_and_the_hold() {
    let root = scratch("files");
    let p = plan("1234567-1026000006-0006");
    let urls = root.join("urls");
    let hold = root.join("hold");
    let prov = FakeProvider::default();
    with_ops(
        &root,
        &p,
        &prov,
        &FakeHttp::default(),
        root.join("none"),
        Some(urls.clone()),
        Some(hold.clone()),
        |o| {
            o.publish_urls(&[
                "https://a/upstream/t1".into(),
                "https://b/upstream/t2".into(),
            ]);
            assert!(!o.hold_released());
            std::fs::write(&hold, "").unwrap();
            assert!(o.hold_released());
        },
    );
    assert_eq!(
        std::fs::read_to_string(&urls).unwrap(),
        "https://a/upstream/t1,https://b/upstream/t2\n"
    );
    assert!(!root.join("urls.fv-lab-part").exists());
    // Without a hold, it is always lifted; without a URL file, nothing is written.
    with_ops(
        &root,
        &p,
        &prov,
        &FakeHttp::default(),
        root.join("none"),
        None,
        None,
        |o| {
            assert!(o.hold_released());
            o.publish_urls(&["x".into()]);
        },
    );
}

#[test]
fn references_and_the_upstream_run_through_the_proxy() {
    let root = scratch("http");
    let p = plan("1234567-1026000007-0007");
    let pod = UpPod {
        stack: "fastvideo".into(),
        id: "up1".into(),
        url: "https://up1-8000.proxy.runpod.net".into(),
        tag: "t1".into(),
        targets: vec!["fasth3-4step-vsa".into()],
    };
    let base = "https://up1-8000.proxy.runpod.net/upstream/t1";
    let mut http = FakeHttp::default();
    // fasth3-4step-vsa: done with its dump (fv-gpucheck serve's listing).
    http.answers.insert(
        format!("{base}/oracle-fasth3-4step-vsa/ORACLE_DONE"),
        (200, "done\n".into()),
    );
    http.answers.insert(
        format!("{base}/oracle-fasth3-4step-vsa/"),
        (
            200,
            listing(&[
                "ORACLE_DONE",
                "cell.json",
                "dump.ls",
                "oracle-dump.tar",
                "oracle_meta.json",
            ]),
        ),
    );
    // ltx25-retake: ORACLE_DONE without a dump (the 2026-10-26 run, its
    // cell exited 2): missing, with the cell's exit.
    http.answers.insert(
        format!("{base}/oracle-ltx25-retake/ORACLE_DONE"),
        (200, "done\n".into()),
    );
    http.answers.insert(
        format!("{base}/oracle-ltx25-retake/"),
        (
            200,
            listing(&["ORACLE_DONE", "cell.json", "mem.log", "stderr.log"]),
        ),
    );
    http.answers.insert(
        format!("{base}/oracle-ltx25-retake/cell.json"),
        (
            200,
            r#"{"cell":"oracle-ltx25-retake","exit":2,"seconds":3}"#.into(),
        ),
    );
    // ltx25-a2v-i2v: no dump and no cell.json; a partial name is no tar.
    http.answers.insert(
        format!("{base}/oracle-ltx25-a2v-i2v/ORACLE_DONE"),
        (200, "done\n".into()),
    );
    http.answers.insert(
        format!("{base}/oracle-ltx25-a2v-i2v/"),
        (200, listing(&["ORACLE_DONE", "oracle-dump.tar.part"])),
    );
    // ltx25-i2v: no dump, cell.json not JSON; ltx25-kf: no dump, its exit
    // null (a cell killed before it wrote one).
    for (t, cell) in [
        ("ltx25-i2v", "{\"cell\":"),
        ("ltx25-kf", r#"{"cell":"oracle-ltx25-kf","exit":null}"#),
    ] {
        http.answers.insert(
            format!("{base}/oracle-{t}/ORACLE_DONE"),
            (200, "done\n".into()),
        );
        http.answers.insert(
            format!("{base}/oracle-{t}/"),
            (200, listing(&["ORACLE_DONE", "cell.json"])),
        );
        http.answers
            .insert(format!("{base}/oracle-{t}/cell.json"), (200, cell.into()));
    }
    // ltx25-512p: ORACLE_DONE but the listing not answered (a proxy 502):
    // asked again at the next poll.
    http.answers.insert(
        format!("{base}/oracle-ltx25-512p/ORACLE_DONE"),
        (200, "done\n".into()),
    );
    http.answers
        .insert(format!("{base}/oracle-ltx25-512p/"), (502, String::new()));
    http.answers
        .insert(format!("{base}/DONE"), (200, "done t1\n".into()));
    let prov = FakeProvider::default();
    with_ops(
        &root,
        &p,
        &prov,
        &http,
        root.join("none"),
        None,
        None,
        |o| {
            assert_eq!(o.reference(&pod, "fasth3-4step-vsa"), RefState::Served);
            assert_eq!(o.reference(&pod, "fasth3-8step"), RefState::Pending);
            assert_eq!(
                o.reference(&pod, "ltx25-retake"),
                RefState::NoDump("ORACLE_DONE without oracle-dump.tar; cell exit 2".into())
            );
            assert_eq!(
                o.reference(&pod, "ltx25-a2v-i2v"),
                RefState::NoDump("ORACLE_DONE without oracle-dump.tar; no cell.json".into())
            );
            assert_eq!(o.reference(&pod, "ltx25-512p"), RefState::Pending);
            assert_eq!(
                o.reference(&pod, "ltx25-i2v"),
                RefState::NoDump("ORACLE_DONE without oracle-dump.tar; cell.json malformed".into())
            );
            assert_eq!(
                o.reference(&pod, "ltx25-kf"),
                RefState::NoDump(
                    "ORACLE_DONE without oracle-dump.tar; cell.json has no exit".into()
                )
            );
            assert_eq!(
                o.upstream_state(&pod),
                fastvideo_lab::oracle::UpState::Finished
            );
            o.keepalive(&pod);
        },
    );
    let asked = http.asked.borrow();
    assert!(asked.contains(&"https://up1-8000.proxy.runpod.net/keepalive.bin".to_owned()));
    // The dump is looked up in the directory's listing, never downloaded.
    assert!(!asked.iter().any(|u| u.ends_with("/oracle-dump.tar")));
    // A target without ORACLE_DONE is not asked further.
    assert!(!asked.iter().any(|u| u.ends_with("/oracle-fasth3-8step/")));
}

/// A directory listing as `fv-gpucheck serve` writes it.
fn listing(names: &[&str]) -> String {
    let mut s = String::from("<!DOCTYPE html>\n<html><body><ul>\n");
    for n in names {
        s.push_str(&format!("<li><a href=\"{n}\">{n}</a></li>\n"));
    }
    s.push_str("</ul></body></html>\n");
    s
}
