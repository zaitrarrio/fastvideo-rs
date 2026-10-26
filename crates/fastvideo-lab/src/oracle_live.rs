//! The world of `fv-lab oracle` (docs/adr/0019-fv-lab.md §19): [`LiveOps`],
//! the [`OracleOps`] of the binary. The upstream pods through fv-deploy's
//! `Deployer` on a [`Provider`] (Runpod's in the binary), the pods' files
//! through an [`Http`], the runtime side as a child process (`fv-lab pod up
//! --yes --wait`). Every seam is a parameter, so the tests run it against a
//! fake provider, a fake HTTP and a fake executable (`tests/oracle_live.rs`).

use std::os::unix::process::CommandExt as _;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use fastvideo_deploy::boot::sys::{Clock, Http, LogSink, Method, Request};
use fastvideo_deploy::guard::Backstop;
use fastvideo_deploy::lifecycle::{Deployer, Ledger, UpOptions};
use fastvideo_deploy::provider::runpod::PodListing;
use fastvideo_deploy::provider::{InstanceId, InstanceState, Provider};

use crate::oracle::{
    OracleOps, OraclePlan, RefState, RuntimePod, UpPod, UpState, UpstreamPlan, ORACLE_DUMP_TAR,
    UPSTREAM_FAMILY, UPSTREAM_FETCH_SKIP,
};
use crate::pod::{
    fetch_tree, hrefs, name_taken, or_delete, run_dir, write_record, FetchSkip, PodRecord,
    TreeFetch,
};

/// The binary's [`OracleOps`]. Fields are the seams; `child`, `exit` and
/// `code` start empty.
pub struct LiveOps<'a> {
    /// The repository root (`artifacts/…` under it) and fv-lab's state
    /// directory (ledger, pod records, backstop logs).
    pub root: PathBuf,
    pub state: PathBuf,
    pub plan: &'a OraclePlan,
    pub http: &'a dyn Http,
    /// Creates, watches and deletes pods (Runpod's in the binary; its
    /// readiness the upstream run directory's server).
    pub provider: &'a dyn Provider,
    /// A pod's proxy base URL when the create answer has none.
    pub pod_url: &'a dyn Fn(&str) -> String,
    /// The live pods, for the name check before a create.
    pub list: &'a dyn Fn() -> Result<Vec<PodListing>, String>,
    /// An image tag to its digest (`ghcr_resolve`).
    pub resolve: &'a dyn Fn(&str) -> Result<String, String>,
    pub backstop: &'a dyn Backstop,
    pub clock: &'a dyn Clock,
    pub log: &'a dyn LogSink,
    pub interrupted: &'a dyn Fn() -> bool,
    /// Whether a text holds the provider's API key (fv-deploy's env guard).
    pub leaks: &'a dyn Fn(&str) -> bool,
    /// The `fv-lab` the runtime side runs (this binary).
    pub exe: PathBuf,
    /// `--url-file`, `--hold-upstream`.
    pub url_file: Option<PathBuf>,
    pub hold: Option<PathBuf>,
    /// fv-deploy's readiness poll.
    pub poll: Duration,
    pub child: Option<Child>,
    pub exit: Option<bool>,
    pub code: Option<i32>,
}

impl LiveOps<'_> {
    fn deployer<'b>(&'b self, ledger: &'b Ledger) -> Deployer<'b> {
        Deployer {
            provider: self.provider,
            clock: self.clock,
            log: self.log,
            backstop: self.backstop,
            ledger: Some(ledger),
            recorder: None,
            interrupted: self.interrupted,
            leaks: self.leaks,
            poll: self.poll,
        }
    }

    fn ledger(&self) -> Ledger {
        Ledger {
            path: self.state.join("ledger.tsv"),
        }
    }

    fn get(&self, url: &str, secs: u64) -> Option<(u16, Vec<u8>)> {
        self.http
            .send(&Request::new(Method::Get, url, Duration::from_secs(secs)))
            .ok()
            .map(|r| (r.status, r.body))
    }

    fn runtime_dir(&self) -> PathBuf {
        run_dir(&self.root, &self.plan.tier, &self.plan.runtime_tag)
    }

    /// A runtime side that failed may leave its pod up (killed before its
    /// own delete): the pod its `pod.json` names, if still live, is deleted.
    /// The run directory did not exist before this child started
    /// ([`OracleOps::runtime_start`] refuses a reused tag), so the record is
    /// this child's.
    fn delete_child_pod(&mut self) {
        let Some(rt) = self.runtime_pod() else {
            return;
        };
        let id = InstanceId::new(rt.id.clone());
        match self.provider.status(&id) {
            Ok(InstanceState::Terminated | InstanceState::Exited) => {}
            _ => {
                let ledger = self.ledger();
                let r = self
                    .deployer(&ledger)
                    .destroy_retry(&id, "runtime-side-failed");
                self.log.line(&format!(
                    "[fv-lab oracle] the runtime side failed with its pod {} up: {}",
                    rt.id,
                    match r {
                        Ok(()) => "deleted".to_owned(),
                        Err(e) => format!("not deleted: {e} (its backstop deletes it at the cap)"),
                    }
                ));
            }
        }
    }

    fn ended(&mut self, ok: bool, code: Option<i32>) -> Option<bool> {
        self.exit = Some(ok);
        self.code = code;
        if !ok {
            self.delete_child_pod();
        }
        self.exit
    }

    fn signal(&self, sig: &str) {
        if let Some(c) = &self.child {
            let _ = Command::new("kill")
                .args([sig, &c.id().to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

impl OracleOps for LiveOps<'_> {
    fn now(&self) -> f64 {
        self.clock.now()
    }

    fn sleep(&mut self, d: Duration) {
        // In short steps, so an interrupt is seen within seconds.
        let end = self.clock.now() + d.as_secs_f64();
        while self.clock.now() < end && !(self.interrupted)() {
            self.clock.sleep(Duration::from_secs(1));
        }
    }

    fn log(&mut self, line: &str) {
        self.log.line(line);
    }

    fn interrupted(&self) -> bool {
        (self.interrupted)()
    }

    fn upstream_up(&mut self, p: &UpstreamPlan) -> Result<UpPod, String> {
        let mut l = p.launch.clone();
        if name_taken(&(self.list)()?, &l.name) {
            return Err(format!(
                "a live pod is already named {}: not creating another",
                l.name
            ));
        }
        let tag_image = l.image.clone();
        l.image = (self.resolve)(&tag_image)?;
        self.log.line(&format!(
            "[fv-lab oracle] {} on {}: image {} ({tag_image}){}",
            l.name,
            l.gpu.types.join(", "),
            l.image,
            l.gpu
                .min_ram_gb
                .map(|g| format!(", host RAM floor {g} GB per GPU (minRAMPerGPU)"))
                .unwrap_or_default()
        ));
        let ledger = self.ledger();
        let d = self.deployer(&ledger);
        let report = d
            .up(
                &l,
                &UpOptions {
                    run_key: None,
                    smoke: None,
                    keep: true,
                },
            )
            .map_err(|e| e.to_string())?;
        // No instance kept: `up`'s exit sweep deleted every pod of the name.
        let inst = report.instance.ok_or("up returned no pod")?;
        let url = inst
            .url
            .clone()
            .unwrap_or_else(|| (self.pod_url)(inst.id.as_str()));
        let rec = PodRecord {
            id: inst.id.to_string(),
            name: l.name.clone(),
            family: UPSTREAM_FAMILY.into(),
            tag: p.start.tag.clone(),
            sha: p.start.up_sha.get(..7).unwrap_or_default().to_owned(),
            image: l.image.clone(),
            url: url.clone(),
            gpu: inst.gpu.clone(),
            usd_per_hr: inst.usd_per_hr,
            created_at: inst.created_at,
            cap_s: p.start.cap_s,
            min_ram_gb: l.gpu.min_ram_gb,
            memory_gb: inst.memory_gb,
            vcpus: inst.vcpus,
        };
        // The pod exists from here on: a record that cannot be written
        // deletes it before the error is returned (review of #216).
        let written = write_record(
            &self.state.join("pods").join(format!("{}.json", rec.id)),
            &rec,
        )
        .and_then(|()| {
            write_record(
                &run_dir(&self.root, UPSTREAM_FAMILY, &rec.tag).join("pod.json"),
                &rec,
            )
        });
        or_delete(written, &rec.id, || {
            d.destroy_retry(&inst.id, "record-failed")
                .map_err(|e| e.to_string())
        })?;
        Ok(UpPod {
            stack: p.stack.clone(),
            id: rec.id,
            url,
            tag: rec.tag,
            targets: p.targets.clone(),
        })
    }

    fn reference(&mut self, pod: &UpPod, target: &str) -> RefState {
        let dir = format!("{}/oracle-{target}/", pod.base());
        if !self
            .get(&format!("{dir}ORACLE_DONE"), 30)
            .is_some_and(|(s, _)| s == 200)
        {
            return RefState::Pending;
        }
        // `ORACLE_DONE` is written after the tar's rename, so the listing is
        // final; one that cannot be read is asked again at the next poll.
        let Some(listing) = self
            .get(&dir, 30)
            .filter(|(s, _)| *s == 200)
            .map(|(_, b)| String::from_utf8_lossy(&b).into_owned())
        else {
            return RefState::Pending;
        };
        if hrefs(&listing).iter().any(|h| h == ORACLE_DUMP_TAR) {
            return RefState::Served;
        }
        let exit = self
            .get(&format!("{dir}cell.json"), 30)
            .filter(|(s, _)| *s == 200)
            .and_then(|(_, b)| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .map(|v| match &v["exit"] {
                serde_json::Value::Null => "cell.json has no exit".to_owned(),
                e => format!("cell exit {e}"),
            })
            .unwrap_or_else(|| "no cell.json".to_owned());
        RefState::NoDump(format!("ORACLE_DONE without {ORACLE_DUMP_TAR}; {exit}"))
    }

    fn upstream_state(&mut self, pod: &UpPod) -> UpState {
        let done = format!("done {}", pod.tag);
        if self
            .get(&format!("{}/DONE", pod.base()), 30)
            .filter(|(s, _)| *s == 200)
            .is_some_and(|(_, b)| String::from_utf8_lossy(&b).lines().next() == Some(done.as_str()))
        {
            return UpState::Finished;
        }
        match self.provider.status(&InstanceId::new(pod.id.clone())) {
            Ok(InstanceState::Terminated | InstanceState::Exited) => UpState::Gone,
            _ => UpState::Running,
        }
    }

    fn keepalive(&mut self, pod: &UpPod) {
        let _ = self.get(
            &format!("{}/keepalive.bin", pod.url.trim_end_matches('/')),
            50,
        );
    }

    fn runtime_start(&mut self, args: &[String]) -> Result<(), String> {
        // A reused tag would leave another run's pod.json for this one.
        let dir = self.runtime_dir();
        if dir.exists() {
            return Err(format!(
                "{} exists: the runtime tag {} was used before (pick another --tag)",
                dir.display(),
                self.plan.runtime_tag
            ));
        }
        let child = Command::new(&self.exe)
            .args(args)
            .arg("--root")
            .arg(&self.root)
            .arg("--state-dir")
            .arg(&self.state)
            .stdin(Stdio::null())
            // Its JSON report goes with the logs: stdout is this run's report.
            .stdout(std::io::stderr())
            // Its own process group: a terminal's Ctrl-C reaches the driver
            // only, which stops the child once (SIGTERM: it deletes what it
            // made); a second signal would make it quit without deleting.
            .process_group(0)
            .spawn()
            .map_err(|e| format!("{}: {e}", self.exe.display()))?;
        self.log.line(&format!(
            "[fv-lab oracle] runtime side (pid {}): fv-lab {}",
            child.id(),
            args.join(" ")
        ));
        self.child = Some(child);
        Ok(())
    }

    fn runtime_pod(&mut self) -> Option<RuntimePod> {
        let p = self.runtime_dir().join("pod.json");
        let rec: PodRecord = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
        Some(RuntimePod {
            id: rec.id,
            url: rec.url,
            tag: rec.tag,
        })
    }

    fn runtime_refs_done(&mut self, rt: &RuntimePod) -> bool {
        self.get(
            &format!(
                "{}/{}/{}/ORACLE_REFS_DONE",
                rt.url.trim_end_matches('/'),
                self.plan.tier,
                rt.tag
            ),
            30,
        )
        .is_some_and(|(s, _)| s == 200)
    }

    fn runtime_exit(&mut self) -> Option<bool> {
        if self.exit.is_some() {
            return self.exit;
        }
        let status = self.child.as_mut()?.try_wait();
        match status {
            Ok(Some(st)) => self.ended(st.success(), st.code()),
            Ok(None) => None,
            Err(e) => {
                self.log.line(&format!("[fv-lab oracle] runtime side: {e}"));
                self.ended(false, None)
            }
        }
    }

    fn runtime_exit_code(&self) -> Option<i32> {
        self.code
    }

    fn runtime_stop(&mut self) {
        if self.exit.is_none() && self.child.is_some() {
            self.signal("-TERM");
            self.log
                .line("[fv-lab oracle] runtime side asked to stop (SIGTERM)");
        }
    }

    fn runtime_wait(&mut self, limit: Duration) -> Option<bool> {
        self.child.as_ref()?;
        let end = self.clock.now() + limit.as_secs_f64();
        loop {
            if let Some(ok) = self.runtime_exit() {
                return Some(ok);
            }
            if self.clock.now() >= end {
                break;
            }
            self.clock.sleep(Duration::from_millis(200));
        }
        self.log.line(&format!(
            "[fv-lab oracle] the runtime side did not end within {} s: killed",
            limit.as_secs()
        ));
        self.signal("-KILL");
        if let Some(c) = self.child.as_mut() {
            let _ = c.wait();
        }
        self.ended(false, None);
        None
    }

    fn publish_urls(&mut self, urls: &[String]) {
        let Some(f) = self.url_file.clone() else {
            return;
        };
        // Written under a temporary name and renamed: a reader never sees half.
        let part = f.with_extension("fv-lab-part");
        let r = std::fs::write(&part, format!("{}\n", urls.join(",")))
            .and_then(|()| std::fs::rename(&part, &f));
        self.log.line(&match r {
            Ok(()) => format!(
                "[fv-lab oracle] FV_ORACLE_URL={} -> {}",
                urls.join(","),
                f.display()
            ),
            Err(e) => format!("[fv-lab oracle] {}: {e}", f.display()),
        });
    }

    fn hold_released(&mut self) -> bool {
        self.hold.as_ref().is_none_or(|f| f.exists())
    }

    fn upstream_fetch(&mut self, pod: &UpPod) -> Result<(), String> {
        let opts = TreeFetch {
            skip: Some(FetchSkip::parse(UPSTREAM_FETCH_SKIP)?),
            max_mb: 400,
        };
        fetch_tree(
            self.http,
            &pod.url,
            UPSTREAM_FAMILY,
            &pod.tag,
            &run_dir(&self.root, UPSTREAM_FAMILY, &pod.tag),
            &opts,
            self.clock,
            self.log,
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    }

    fn upstream_down(&mut self, pod: &UpPod) -> Result<(), String> {
        let ledger = self.ledger();
        self.deployer(&ledger)
            .destroy_retry(&InstanceId::new(pod.id.clone()), "down")
            .map_err(|e| e.to_string())
    }
}
