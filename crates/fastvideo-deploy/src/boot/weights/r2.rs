//! `weights_source r2` (docs/ops/r2-weights.md): the trees come from the R2
//! weights store through `scripts/gpu/r2-fetch.py` (the image's copy when its
//! SHA-256 is the one fv-control pins, `FV_R2_FETCH_SHA256`, else fv-control's
//! `FV_R2_FETCH_URL`), on a plan fv-control presigns for this pod
//! (`FV_WEIGHTS_PLAN_URL` with the pod's token). No R2 key ever reaches the pod.
//!
//! What the shell boots did, in one place (control r2weights.ts `R2_FETCH_SH`,
//! `R2_RUNPOD_SH`, serverless/weights.ts `SLS_R2_SH`):
//!
//! - an R2 cache volume (`FV_R2_CACHE_ROOT`, serverless): the trees go there
//!   add-only and `FV_WEIGHTS` links to it;
//! - trees this host never loads are skipped (`FV_R2_SKIP_TREES`, default:
//!   the MXFP8 DiTs below sm_100, and the aux trees on an add-only cache);
//! - the serverless cache check (`FV_R2_TREE_REVS`): every tree complete at its
//!   revision asks fv-control for nothing;
//! - **streaming** (`FV_R2_STREAM`, default on; a fv-serve carrying
//!   [`STREAMING_FETCH_MARKER`](crate::boot::STREAMING_FETCH_MARKER); not on an
//!   add-only cache): fv-serve starts once the files its loader does not gate
//!   are in place and loads while the rest arrives. The fetch then runs under
//!   `fv-boot fetch-guard` ([`guard`]), a process of its own that outlives the
//!   `exec`: it times the fetch when it ends and, if it fails after fv-serve
//!   started, stops fv-serve (the boot's pid) so a partial tree is never served;
//! - the aux rows, by SHA-256.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::super::error::BootError;
use super::super::plan;
use super::super::sys::SysError;
use super::super::{Ctx, STREAMING_FETCH_MARKER};
use super::{
    apt_install, aux_files, aux_rows, has_python, sha256_file, tree_present, Prepared, SourceKind,
    Streaming, WeightsSource,
};

/// The MXFP8 pre-quantized DiTs (control recipe-trees.ts `MXFP8_TREE_GLOB`):
/// of no use below sm_100, where the engine runs W8A8.
pub const MXFP8_TREE_GLOB: &str = "*_mxfp8";

pub const STREAM_READY: &str = "r2-stream-ready";
pub const FETCH_RC: &str = "r2-fetch.rc";
pub const SERVE_START: &str = "serve-start";
pub const FETCH_LOG: &str = "r2-fetch.log";
pub const FETCH_STATUS: &str = "r2-fetch.json";

/// `weights_source r2`.
pub struct R2;

/// Whether a file contains `needle` (`grep -qaF`), read in chunks.
pub fn file_contains(p: &Path, needle: &[u8]) -> bool {
    let Ok(mut f) = std::fs::File::open(p) else {
        return false;
    };
    if needle.is_empty() {
        return false;
    }
    let mut buf = vec![0u8; 1 << 20];
    let mut carry: Vec<u8> = Vec::new();
    loop {
        let n = match f.read(&mut buf) {
            Ok(0) | Err(_) => return false,
            Ok(n) => n,
        };
        carry.extend_from_slice(&buf[..n]);
        if crate::boot::driver::contains(&carry, needle) {
            return true;
        }
        let keep = needle.len().saturating_sub(1).min(carry.len());
        carry.drain(..carry.len() - keep);
    }
}

/// The last line of the fetch log that says FAILED (quotes as `'`, at most 240 chars).
pub fn failed_line(log: &Path) -> String {
    std::fs::read_to_string(log)
        .ok()
        .and_then(|t| {
            t.lines()
                .rev()
                .find(|l| l.contains("FAILED"))
                .map(str::to_owned)
        })
        .unwrap_or_default()
        .replace('"', "'")
        .chars()
        .take(240)
        .collect()
}

/// The fetcher's summary (`weights ready: …`, at most 200 chars).
pub fn ready_line(log: &Path) -> String {
    std::fs::read_to_string(log)
        .ok()
        .and_then(|t| {
            t.lines()
                .rev()
                .find_map(|l| l.find("weights ready: ").map(|i| l[i..].to_owned()))
        })
        .unwrap_or_default()
        .chars()
        .take(200)
        .collect()
}

fn state_of(c: &Ctx<'_>) -> PathBuf {
    c.get("FV_R2_STATE")
        .map(PathBuf::from)
        .unwrap_or_else(|| c.state_dir())
}

fn mkdir(p: &Path) -> Result<(), BootError> {
    std::fs::create_dir_all(p).map_err(|e| BootError::Sys(SysError::io(p, e)))
}

/// `FV_R2_SKIP_TREES` as given (even empty), else the host's default.
fn skip_trees(c: &Ctx<'_>, add_only: bool) -> String {
    if c.env.is_set("FV_R2_SKIP_TREES") {
        return c.env.raw("FV_R2_SKIP_TREES").unwrap_or_default().to_owned();
    }
    let cc = c
        .deps
        .spawner
        .output(
            &c.cmd("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"]),
        )
        .ok()
        .filter(|o| o.ok())
        .and_then(|o| o.stdout_text().lines().next().map(|l| l.replace(' ', "")))
        .unwrap_or_default();
    let mut s = String::new();
    let b = cc.as_bytes();
    if b.len() >= 2 && (b'1'..=b'9').contains(&b[0]) && b[1] == b'.' {
        s.push_str(MXFP8_TREE_GLOB);
    }
    if add_only {
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str("auxiliary/*");
    }
    s
}

/// The fetcher to run: the image's copy when its SHA-256 is the pinned one, else fv-control's.
fn fetcher(c: &Ctx<'_>, tools: &Path, sha: &str) -> Result<PathBuf, BootError> {
    let t = c.now();
    let img = c
        .get("FV_R2_FETCH_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|| c.paths.r2_fetch_image.clone());
    if img.is_file() && sha256_file(&img).map(|(h, _)| h == sha).unwrap_or(false) {
        c.step("fetcher", t, "from the image");
        return Ok(img);
    }
    let url = c.get("FV_R2_FETCH_URL").ok_or_else(|| {
        BootError::Weights("r2-fetch.py download from fv-control: no FV_R2_FETCH_URL".into())
    })?;
    let f = tools.join("r2-fetch.py");
    let mut ok = false;
    for i in 0..5 {
        if c.deps
            .http
            .download(url, &f, Duration::from_secs(30))
            .is_ok()
        {
            ok = true;
            break;
        }
        if i < 4 {
            c.deps.clock.sleep(Duration::from_secs(1));
        }
    }
    if !ok {
        return Err(BootError::Weights(
            "r2-fetch.py download from fv-control".into(),
        ));
    }
    if !sha256_file(&f).map(|(h, _)| h == sha).unwrap_or(false) {
        return Err(BootError::Weights("r2-fetch.py sha256".into()));
    }
    c.step("fetcher", t, "downloaded from fv-control");
    Ok(f)
}

impl WeightsSource for R2 {
    fn kind(&self) -> SourceKind {
        SourceKind::R2
    }

    fn prepare(&self, c: &mut Ctx<'_>) -> Result<Prepared, BootError> {
        let plan_url = c
            .get("FV_WEIGHTS_PLAN_URL")
            .map(str::to_owned)
            .ok_or_else(|| BootError::Env("weights_source r2: no FV_WEIGHTS_PLAN_URL".into()))?;
        let sha = c
            .get("FV_R2_FETCH_SHA256")
            .map(str::to_ascii_lowercase)
            .ok_or_else(|| BootError::Env("weights_source r2: no FV_R2_FETCH_SHA256".into()))?;
        if c.get("FV_POD_ID").is_none() && c.kind != super::super::Kind::Serverless {
            if let Some(id) = c.env.get("RUNPOD_POD_ID").map(str::to_owned) {
                c.export("FV_POD_ID", id);
            }
        }
        c.say(&format!("weights: R2 weights store (plan: {plan_url})"));
        let boot_note = format!("container started; {}", c.driver);
        c.sls_report("boot", &boot_note);
        let s = state_of(c);
        let tools = c
            .get("FV_R2_TOOLS")
            .map(PathBuf::from)
            .unwrap_or_else(|| c.paths.tools.clone());
        let weights = c.weights_dir();
        let (root, add_only) = match c.get("FV_R2_CACHE_ROOT").map(PathBuf::from) {
            Some(r) if r.parent().is_some_and(Path::is_dir) => (r, true),
            _ => (weights.clone(), false),
        };
        mkdir(&root)?;
        mkdir(&s)?;
        mkdir(&tools)?;
        if root != weights && std::fs::symlink_metadata(&weights).is_err() {
            if let Some(p) = weights.parent() {
                mkdir(p)?;
            }
            std::os::unix::fs::symlink(&root, &weights).map_err(|e| SysError::io(&weights, e))?;
        }
        let skip = skip_trees(c, add_only);
        // Exported only when it says something (or was given): an unset
        // FV_R2_SKIP_TREES stays unset for the fetcher and fv-serve.
        if !skip.is_empty() || c.env.is_set("FV_R2_SKIP_TREES") {
            c.export("FV_R2_SKIP_TREES", skip.clone());
        }

        let revs = plan::tree_revs(c.get("FV_R2_TREE_REVS"));
        let fetch = if revs.is_empty() {
            true
        } else {
            let miss: Vec<&str> = revs
                .iter()
                .filter(|(d, _)| !plan::any_glob(&skip, d))
                .filter(|(d, r)| !tree_present(&root, d, r))
                .map(|(d, _)| d.as_str())
                .collect();
            if miss.is_empty() {
                c.say(&format!(
                    "weights: cache hit (every tree at its revision under {}): no fetch",
                    root.display()
                ));
                c.sls_report("cache", "hit");
                false
            } else {
                let skipping = if skip.is_empty() {
                    String::new()
                } else {
                    format!(" (skipping {skip})")
                };
                c.say(&format!(
                    "weights: cache miss: {}: fetching from the R2 weights store{skipping}",
                    miss.join(" ")
                ));
                c.sls_report("cache", &format!("miss: {}", miss.join(" ")));
                true
            }
        };

        let mut streaming = None;
        if fetch {
            let t = c.now();
            if has_python(c) {
                c.step("tools", t, "python3 in the image");
            } else {
                if !apt_install(
                    c,
                    &["python3", "curl", "ca-certificates"],
                    &s.join("apt-r2.log"),
                ) {
                    return Err(BootError::Weights("apt python3".into()));
                }
                c.step("tools", t, "apt-get python3: the image has no python3");
            }
            let f = fetcher(c, &tools, &sha)?;
            let token_env = if c.get("FV_WEIGHTS_TOKEN").is_some() {
                "FV_WEIGHTS_TOKEN"
            } else {
                "FV_ENDPOINT_REPORT_TOKEN"
            };
            let pod = match c.rep.protocol() {
                Some(super::super::report::Protocol::Serverless) => {
                    c.rep.pod().unwrap_or_default().to_owned()
                }
                _ => c
                    .get("FV_POD_ID")
                    .or_else(|| c.get("RUNPOD_POD_ID"))
                    .unwrap_or_default()
                    .to_owned(),
            };
            let mut args = vec![
                f.display().to_string(),
                "--root".into(),
                root.display().to_string(),
                "--plan-url".into(),
                plan_url.clone(),
                "--token-env".into(),
                token_env.into(),
                "--pod".into(),
                pod,
                "--status".into(),
                s.join(FETCH_STATUS).display().to_string(),
            ];
            if let Some(u) = c
                .get("FV_SLS_REPORT_URL")
                .or_else(|| c.get("FV_ENDPOINT_REPORT_URL"))
            {
                args.extend(["--report-url".into(), u.to_owned()]);
            }
            let bin = c.serve_bin();
            let stream = c.get_or("FV_R2_STREAM", "1") == "1" && !add_only;
            let stream = stream && {
                let can = file_contains(&bin, STREAMING_FETCH_MARKER.as_bytes());
                if !can {
                    c.say(&format!(
                        "weights: {} cannot load while the weights arrive: fetching first",
                        bin.display()
                    ));
                }
                can
            };
            if add_only {
                args.push("--add-only".into());
            }
            let log = s.join(FETCH_LOG);
            let t = c.now();
            if stream {
                for n in [STREAM_READY, FETCH_RC, SERVE_START] {
                    let _ = std::fs::remove_file(s.join(n));
                }
                args.extend([
                    "--stream".into(),
                    "--stream-ready".into(),
                    s.join(STREAM_READY).display().to_string(),
                ]);
                let guard = c
                    .cmd(c.paths.self_exe.display().to_string())
                    .args([
                        "fetch-guard",
                        "--boot-pid",
                        &c.pid.to_string(),
                        "--t0",
                        &format!("{:.6}", c.log.t0()),
                        "--state",
                        &s.display().to_string(),
                        "--",
                    ])
                    .arg("python3")
                    .args(args);
                c.deps.spawner.spawn(&guard)?;
                while !s.join(STREAM_READY).exists() && !s.join(FETCH_RC).exists() {
                    c.deps.clock.sleep(Duration::from_millis(200));
                }
                if !s.join(STREAM_READY).exists() {
                    return Err(BootError::Weights(format!(
                        "r2 weights: {}",
                        failed_line(&log)
                    )));
                }
                c.step("stream_ready", t, "the files fv-serve reads unguarded are in place; the rest streams in while it loads");
                c.export("FASTVIDEO_STREAMING_FETCH", "1");
                c.export(
                    "FASTVIDEO_STREAMING_FETCH_STATUS",
                    s.join(FETCH_STATUS).display().to_string(),
                );
                // The fetcher first: this process, and fv-serve after the exec, yield the CPUs to it.
                let nice = c.get_or("FV_SERVE_NICE", "5").to_owned();
                let _ = c.deps.spawner.output(&c.cmd("renice").args([
                    "-n",
                    &nice,
                    "-p",
                    &c.pid.to_string(),
                ]));
                streaming = Some(Streaming { state: s.clone() });
            } else {
                let rc = c.deps.spawner.tee(&c.cmd("python3").args(args), &log)?;
                if rc != Some(0) {
                    return Err(BootError::Weights(format!(
                        "r2 weights: {}",
                        failed_line(&log)
                    )));
                }
                c.step("fetch", t, &ready_line(&log));
            }
        }

        // A GMI / Brev pod (pod protocol): PROVIDER_BOOT ran its weights loop
        // after the fetch: leftovers cleaned, `weights checking` reported, then
        // the aux rows.
        if c.rep.protocol() == Some(super::super::report::Protocol::Pod) && !add_only {
            super::clean_leftovers(&root);
            c.pod_report("weights", "", "checking");
        }
        let rows = aux_rows(c)?;
        if !rows.is_empty() {
            let t = c.now();
            let (n, got) = aux_files(c, &root, &rows, add_only)?;
            c.step(
                "aux",
                t,
                &format!("{n} files checked, {got} from their pinned URLs"),
            );
        }
        Ok(Prepared {
            root: Some(root),
            streaming,
        })
    }
}

/// Just before the exec: a fetch still running gets fv-serve's start time
/// (its `load_overlap` step). `true` while it runs.
pub fn before_serve(c: &Ctx<'_>, s: &Streaming) -> bool {
    if s.state.join(FETCH_RC).exists() {
        return false;
    }
    let _ = std::fs::write(s.state.join(SERVE_START), format!("{:.6}", c.now()));
    true
}

/// `fv-boot fetch-guard`: runs the streaming fetch to its end (stdout and
/// stderr into the fetch log and the container log), times it, and on a
/// failure after fv-serve started stops fv-serve (`boot_pid`: TERM, then
/// KILL after `FV_STREAM_KILL_GRACE_S`). Writes the exit code to `r2-fetch.rc`.
pub fn guard(c: &Ctx<'_>, boot_pid: u32, state: &Path, program: &str, args: &[String]) -> i32 {
    let t = c.now();
    let log = state.join(FETCH_LOG);
    let rc = c
        .deps
        .spawner
        .tee(&c.cmd(program).args(args.iter().cloned()), &log)
        .ok()
        .flatten();
    if rc == Some(0) {
        c.step("fetch", t, &ready_line(&log));
        if let Some(since) = std::fs::read_to_string(state.join(SERVE_START))
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
        {
            c.step(
                "load_overlap",
                since,
                "fv-serve loading while the fetch ran",
            );
        }
    } else if state.join(STREAM_READY).exists() {
        let why: String = failed_line(&log).chars().take(200).collect();
        c.say(&format!(
            "FAILED: r2 weights while fv-serve was loading: {why}: stopping fv-serve"
        ));
        c.rep
            .failed(c.deps.http, c.deps.clock, &format!("r2 weights: {why}"));
        let _ = c
            .deps
            .spawner
            .output(&super::super::sys::kill_cmd("TERM", boot_pid));
        c.deps.clock.sleep(Duration::from_secs_f64(
            c.env.secs("FV_STREAM_KILL_GRACE_S", 20.0),
        ));
        let _ = c
            .deps
            .spawner
            .output(&super::super::sys::kill_cmd("KILL", boot_pid));
    }
    let code = rc.unwrap_or(1);
    let _ = std::fs::write(state.join(FETCH_RC), code.to_string());
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fvb-r2-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn log_lines() {
        let d = tmp("lines");
        let l = d.join("x.log");
        std::fs::write(&l, "a\nr2-fetch: FAILED: tree \"x\" part 3\nweights ready: 1 tree(s), 2.00 GB fetched in 3.0s (666 MB/s); retries 0\nb\n").unwrap();
        assert_eq!(failed_line(&l), "r2-fetch: FAILED: tree 'x' part 3");
        assert_eq!(
            ready_line(&l),
            "weights ready: 1 tree(s), 2.00 GB fetched in 3.0s (666 MB/s); retries 0"
        );
        assert_eq!(failed_line(&d.join("none")), "");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn marker_across_chunks() {
        let d = tmp("marker");
        let f = d.join("bin");
        let mut b = vec![0u8; (1 << 20) - 5];
        b.extend_from_slice(STREAMING_FETCH_MARKER.as_bytes());
        b.extend_from_slice(&[0u8; 10]);
        std::fs::write(&f, &b).unwrap();
        assert!(file_contains(&f, STREAMING_FETCH_MARKER.as_bytes()));
        std::fs::write(&f, b"\x7fELF no marker").unwrap();
        assert!(!file_contains(&f, STREAMING_FETCH_MARKER.as_bytes()));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
