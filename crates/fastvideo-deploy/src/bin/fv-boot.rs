//! `fv-boot`: the serve images' entrypoint (docs/adr/0014-fv-boot.md).
//!
//! ```text
//! fv-boot [fv-serve args…]   the boot sequence, then exec fv-serve
//! fv-boot watchdog           the deadline / balance backstop (started by the boot)
//! fv-boot fetch-guard --boot-pid P --t0 T --state DIR -- <program> <args…>
//!                            a streaming R2 fetch run to its end (started by the boot)
//! fv-boot version            this binary's version
//! fv-boot --version | --help | --print-config [args…]
//!                            fv-serve's own flags: exec fv-serve directly, no boot
//! ```
//!
//! Every input is the environment (ADR-0014 §3); arguments other than the
//! subcommands above are fv-serve's (`--config <path>`, as the image
//! entrypoint passed them before).

use std::os::unix::process::CommandExt;
use std::process::{Command, ExitCode};

use fastvideo_deploy::boot::http::ReqwestHttp;
use fastvideo_deploy::boot::sys::{RealClock, RealSpawner, StderrLog};
use fastvideo_deploy::boot::{self, watchdog, weights, BootEnv, BootError, Ctx, Deps, Paths};

fn guard_args(args: &[String]) -> Option<(u32, f64, String, String, Vec<String>)> {
    let mut pid = None;
    let mut t0 = None;
    let mut state = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--boot-pid" => pid = it.next()?.parse().ok(),
            "--t0" => t0 = it.next()?.parse().ok(),
            "--state" => state = it.next().cloned(),
            "--" => {
                let program = it.next()?.clone();
                let rest: Vec<String> = it.cloned().collect();
                return Some((pid?, t0?, state?, program, rest));
            }
            _ => return None,
        }
    }
    None
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let http = match ReqwestHttp::new() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[fv-boot] FAILED: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (sp, clock, log) = (RealSpawner, RealClock, StderrLog);
    let deps = Deps {
        spawner: &sp,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let env = BootEnv::from_process();
    let pid = std::process::id();
    match args.first().map(String::as_str) {
        Some("version") => {
            println!("fv-boot {}", boot::VERSION);
            ExitCode::SUCCESS
        }
        Some("watchdog") => {
            let c = Ctx::new(env, deps, Paths::image(), pid);
            match watchdog::mode(&c.env) {
                Some(m) => {
                    watchdog::run(&c, &m, None);
                    ExitCode::SUCCESS
                }
                None => {
                    c.say("[watchdog] nothing to guard in this environment");
                    ExitCode::SUCCESS
                }
            }
        }
        Some("fetch-guard") => {
            let Some((boot_pid, t0, state, program, rest)) = guard_args(&args[1..]) else {
                eprintln!(
                    "fv-boot fetch-guard --boot-pid P --t0 T --state DIR -- <program> <args…>"
                );
                return ExitCode::from(2);
            };
            let c = Ctx::with_t0(env, deps, Paths::image(), pid, t0);
            let code =
                weights::r2::guard(&c, boot_pid, std::path::Path::new(&state), &program, &rest);
            ExitCode::from(u8::try_from(code.clamp(0, 255)).unwrap_or(1))
        }
        _ if boot::passthrough(&args) => {
            let bin = env
                .get("FV_SERVE_BIN")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| Paths::image().serve_bin);
            let e = Command::new(&bin).args(&args).exec();
            eprintln!("[fv-boot] FAILED: exec {}: {e}", bin.display());
            ExitCode::from(127)
        }
        _ => {
            let mut c = Ctx::new(env, deps, Paths::image(), pid);
            match boot::run(&mut c, &args) {
                Ok(plan) => {
                    let e = Command::new(&plan.program)
                        .args(&plan.args)
                        .env_clear()
                        .envs(&plan.env)
                        .exec();
                    let err = BootError::Config(format!("exec {}: {e}", plan.program.display()));
                    ExitCode::from(u8::try_from(c.fail(&err)).unwrap_or(1))
                }
                Err(e) => ExitCode::from(u8::try_from(c.fail(&e)).unwrap_or(1)),
            }
        }
    }
}
