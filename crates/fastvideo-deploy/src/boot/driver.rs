//! The host's NVIDIA driver against what the image needs (control
//! driver-check.ts, docs/ops/r2-bench-2026-10-12.md finding 2): the images
//! ship the CUDA 13.4 runtime and fv-serve binds the CUDA 13 driver API, so a
//! host whose `libcuda.so.1` predates CUDA 13.0 (driver < 580) would panic at
//! the first call (`undefined symbol: cuCtxGetDevice_v2`). The check fails
//! fast instead, with a line fv-control's boot diagnosis treats as fatal for
//! the pod (podlogs.ts: `driver_too_old: `). A host with neither nvidia-smi
//! nor a libcuda (a CPU worker) passes.

use std::path::{Path, PathBuf};

use super::env::BootEnv;
use super::sys::{Cmd, Spawner};

/// The least driver branch the images run on (CUDA 13.0's).
pub const MIN_DRIVER_MAJOR: u32 = 580;
/// A driver-API symbol of CUDA 13.0 that cudarc binds and older drivers lack.
pub const CUDA13_SYMBOL: &str = "cuCtxGetDevice_v2";
/// The word the failure line starts with (podlogs.ts matches `driver_too_old: `).
pub const DRIVER_TOO_OLD: &str = "driver_too_old";

/// Where the shell check looked for libcuda when `ldconfig -p` knew none.
pub const LIBCUDA_FALLBACKS: [&str; 3] = [
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/local/nvidia/lib64/libcuda.so.1",
];

/// What the host says.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    /// `nvidia-smi --query-gpu=driver_version` (first GPU), if it ran.
    pub driver: Option<String>,
    /// The libcuda the loader would open, if any.
    pub libcuda: Option<PathBuf>,
    /// Whether that file carries [`CUDA13_SYMBOL`] (`None`: no libcuda).
    pub has_symbol: Option<bool>,
}

/// `Ok("driver <v> ok")`, or `Err("driver_too_old: …")`.
pub fn verdict(f: &Facts) -> Result<String, String> {
    let dv = f.driver.as_deref();
    if let Some(major) = dv.and_then(|v| v.split('.').next()?.parse::<u32>().ok()) {
        if major < MIN_DRIVER_MAJOR {
            return Err(format!(
                "{DRIVER_TOO_OLD}: host driver {} < {MIN_DRIVER_MAJOR} (the image needs CUDA 13)",
                dv.unwrap_or_default()
            ));
        }
    }
    if let (Some(lc), Some(false)) = (&f.libcuda, f.has_symbol) {
        return Err(format!(
            "{DRIVER_TOO_OLD}: {} (driver {}) has no {CUDA13_SYMBOL} (the image needs a CUDA 13 driver, >= {MIN_DRIVER_MAJOR})",
            lc.display(),
            dv.unwrap_or("unknown")
        ));
    }
    Ok(format!("driver {} ok", dv.unwrap_or("unknown")))
}

/// `ldconfig -p`'s `libcuda.so.1` path (the first line naming it, last field).
pub fn libcuda_from_ldconfig(out: &str) -> Option<PathBuf> {
    out.lines()
        .find(|l| l.contains("libcuda.so.1 "))
        .and_then(|l| l.split_whitespace().last())
        .map(PathBuf::from)
}

/// Whether `bytes` contain `needle` (`grep -a`).
pub fn contains(bytes: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && bytes.windows(needle.len()).any(|w| w == needle)
}

/// Reads the facts from the host.
pub fn facts(env: &BootEnv, sp: &dyn Spawner, fallbacks: &[&Path]) -> Facts {
    let driver = sp
        .output(
            &Cmd::new("nvidia-smi").args(["--query-gpu=driver_version", "--format=csv,noheader"]),
        )
        .ok()
        .filter(|o| o.ok())
        .and_then(|o| {
            o.stdout_text()
                .lines()
                .next()
                .map(|l| l.replace(' ', ""))
                .filter(|l| !l.is_empty())
        });
    let libcuda = env.get("FV_LIBCUDA").map(PathBuf::from).or_else(|| {
        sp.output(&Cmd::new("ldconfig").arg("-p"))
            .ok()
            .and_then(|o| libcuda_from_ldconfig(&o.stdout_text()))
            .or_else(|| {
                fallbacks
                    .iter()
                    .find(|p| p.exists())
                    .map(|p| p.to_path_buf())
            })
    });
    let has_symbol = libcuda.as_ref().map(|p| {
        std::fs::read(p)
            .map(|b| contains(&b, CUDA13_SYMBOL.as_bytes()))
            .unwrap_or(false)
    });
    Facts {
        driver,
        libcuda,
        has_symbol,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts() {
        let ok = Facts {
            driver: Some("595.91.07".into()),
            libcuda: Some("/usr/lib/x86_64-linux-gnu/libcuda.so.1".into()),
            has_symbol: Some(true),
        };
        assert_eq!(verdict(&ok).as_deref(), Ok("driver 595.91.07 ok"));
        let old = Facts {
            driver: Some("550.54.15".into()),
            ..ok.clone()
        };
        assert_eq!(
            verdict(&old).unwrap_err(),
            "driver_too_old: host driver 550.54.15 < 580 (the image needs CUDA 13)"
        );
        let nosym = Facts {
            driver: None,
            has_symbol: Some(false),
            ..ok.clone()
        };
        assert_eq!(
            verdict(&nosym).unwrap_err(),
            "driver_too_old: /usr/lib/x86_64-linux-gnu/libcuda.so.1 (driver unknown) has no cuCtxGetDevice_v2 (the image needs a CUDA 13 driver, >= 580)"
        );
        // A CPU host: neither.
        assert_eq!(
            verdict(&Facts::default()).as_deref(),
            Ok("driver unknown ok")
        );
        // An unparsable version is not judged (the shell's `-lt` failed quietly).
        let odd = Facts {
            driver: Some("n/a".into()),
            ..Facts::default()
        };
        assert!(verdict(&odd).is_ok());
    }

    #[test]
    fn ldconfig_parse() {
        let out = "1234 libs found in cache `/etc/ld.so.cache'\n\tlibcudart.so.13 (libc6,x86-64) => /usr/local/cuda/lib64/libcudart.so.13\n\tlibcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1\n";
        assert_eq!(
            libcuda_from_ldconfig(out),
            Some(PathBuf::from("/usr/lib/x86_64-linux-gnu/libcuda.so.1"))
        );
        assert_eq!(libcuda_from_ldconfig("nothing\n"), None);
    }

    #[test]
    fn grep_a() {
        assert!(contains(
            b"\x00\x01cuCtxGetDevice_v2\x00",
            CUDA13_SYMBOL.as_bytes()
        ));
        assert!(!contains(b"cuCtxGetDevice\x00", CUDA13_SYMBOL.as_bytes()));
    }
}
