//! The `fv-boot` binary started exactly as the serve-image smoke's refusal check
//! starts the image (`docker run --rm -e FV_FAIL_SLEEP_S=0 <image>`: no
//! arguments, the image's own environment, no worker config): it must refuse
//! and exit non-zero at once, without starting fv-serve, sending a report or
//! pausing (ADR-0011, one image).
//!
//! serve-image run 286 hung on that check for hours: its images carried an
//! fv-boot from before ADR-0011 (a stale tools release, scripts/ci/prebuilt.sh),
//! which booted fv-serve's default config and served forever. The stand-in
//! fv-serve here sleeps when it is exec'd to serve, so such a boot fails this
//! test by its deadline instead of passing.

#![cfg(feature = "boot")]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_fv-boot");

/// Far below the 30 s fail pause an image boot takes without `FV_FAIL_SLEEP_S`.
const DEADLINE: Duration = Duration::from_secs(20);

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Self {
        let d = std::env::temp_dir().join(format!("fvb-bin-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("mkdir {}: {e}", d.display()));
        Self(d)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A stand-in fv-serve: its informational flags answer; started to serve, it
/// leaves a marker and blocks, as a real server does.
fn stand_in_serve(dir: &Path) -> (PathBuf, PathBuf) {
    let bin = dir.join("fv-serve");
    let served = dir.join("served");
    let script = format!(
        "#!/bin/sh\nfor a in \"$@\"; do case \"$a\" in --version) echo 'fv-serve stand-in'; exit 0;; --print-config) echo '{{}}'; exit 0;; esac; done\n: > '{}'\nexec sleep 600\n",
        served.display()
    );
    std::fs::write(&bin, script).unwrap_or_else(|e| panic!("write {}: {e}", bin.display()));
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755))
        .unwrap_or_else(|e| panic!("chmod {}: {e}", bin.display()));
    (bin, served)
}

/// Runs fv-boot with exactly `env` (and no arguments); kills it at the deadline.
/// Returns (exit code, stdout + stderr, elapsed), or panics naming the hang.
fn boot(env: &[(&str, String)]) -> (Option<i32>, String, Duration) {
    let t0 = Instant::now();
    let mut child = Command::new(BIN)
        .env_clear()
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {BIN}: {e}"));
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if t0.elapsed() > DEADLINE => {
                let _ = child.kill();
                let out = child.wait_with_output().ok();
                let text = out
                    .map(|o| {
                        format!(
                            "{}{}",
                            String::from_utf8_lossy(&o.stdout),
                            String::from_utf8_lossy(&o.stderr)
                        )
                    })
                    .unwrap_or_default();
                panic!(
                    "fv-boot without a config still ran after {DEADLINE:?} (it must refuse at once; \
                     the serve-image smoke hung on exactly this):\n{text}"
                );
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => panic!("wait: {e}"),
        }
    }
    let o = child
        .wait_with_output()
        .unwrap_or_else(|e| panic!("output: {e}"));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    (o.status.code(), text, t0.elapsed())
}

/// The image's environment (docker/gpucheck.Dockerfile serve-cpu / serve-cuda
/// `ENV`) plus the smoke's `FV_FAIL_SLEEP_S=0`; fv-serve and the state
/// directory pointed into `dir`.
fn image_env(dir: &Path, serve: &Path, variant: &str) -> Vec<(&'static str, String)> {
    let mut env = vec![
        ("PATH", "/usr/local/bin:/usr/bin:/bin".to_owned()),
        ("FV_STATE_DIR", dir.join("fvstate").display().to_string()),
        ("RUST_LOG", "info".to_owned()),
        ("FV_VARIANT", variant.to_owned()),
        ("FV_FAIL_SLEEP_S", "0".to_owned()),
        ("FV_SERVE_BIN", serve.display().to_string()),
    ];
    if variant == "cuda" {
        env.push(("NVIDIA_VISIBLE_DEVICES", "all".to_owned()));
        env.push((
            "NVIDIA_DRIVER_CAPABILITIES",
            "compute,utility,video".to_owned(),
        ));
    }
    env
}

#[test]
fn the_image_started_without_a_config_refuses_and_exits_at_once() {
    for variant in ["cpu", "cuda"] {
        let dir = Dir::new(variant);
        let (serve, served) = stand_in_serve(&dir.0);
        let (code, out, took) = boot(&image_env(&dir.0, &serve, variant));
        assert_eq!(code, Some(1), "{variant}: exit code; output:\n{out}");
        assert!(
            out.contains("FAILED: no worker config: the image serves no default"),
            "{variant}: no refusal line:\n{out}"
        );
        assert!(
            !served.exists(),
            "{variant}: fv-serve was started to serve:\n{out}"
        );
        assert!(
            !out.contains("exec "),
            "{variant}: the boot went on to exec fv-serve:\n{out}"
        );
        // FV_FAIL_SLEEP_S=0 and no report sink: nothing to wait for.
        assert!(
            took < Duration::from_secs(10),
            "{variant}: took {took:?}:\n{out}"
        );
    }
}

/// An old image's baked `FV_CONFIG` is not a config either: same refusal, and
/// it says why.
#[test]
fn a_baked_fv_config_is_refused_the_same_way() {
    let dir = Dir::new("baked");
    let (serve, served) = stand_in_serve(&dir.0);
    let mut env = image_env(&dir.0, &serve, "cpu");
    env.push(("FV_CONFIG", "/etc/fv/runpod.toml".to_owned()));
    let (code, out, _) = boot(&env);
    assert_eq!(code, Some(1), "{out}");
    assert!(
        out.contains("FV_CONFIG, the old images' baked config, is no longer read"),
        "{out}"
    );
    assert!(!served.exists(), "{out}");
}
