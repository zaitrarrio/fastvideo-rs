//! Ahead-of-time kernel compilation.
//!
//! When an `nvcc` is reachable (`NVCC=/path` or on `PATH`) and the `cuda`
//! feature is on, every SM in `FV_CUBIN_SMS` (default: Turing through
//! Blackwell) gets a cubin and a PTX compiled from `src/wan/kernels.cu`, and a
//! generated `aot.rs` embeds them. The runtime then loads real SASS for the
//! device it finds — no NVRTC on the box, no driver JIT, and the target list is
//! decided here where CI can check it, not at runtime where a missing mapping
//! once compiled every guarded kernel body out on Blackwell.
//!
//! Without `nvcc` the table is empty and the runtime falls back to NVRTC from
//! the identical source string. That keeps `cargo check`/`cargo test` working
//! on a machine with no toolkit; a warning says the binary will JIT.
//!
//! An `nvcc` that is present but fails is a hard build error: a toolkit that
//! cannot compile our kernels is a bug to see, not a reason to quietly ship a
//! JIT-only binary.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SRC: &str = "src/wan/kernels.cu";
const DEFAULT_SMS: &str = "75,80,86,89,90,100,120";

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("NVCC") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|d| d.join("nvcc"))
        .find(|p| p.is_file())
        .or_else(|| {
            let p = Path::new("/usr/local/cuda/bin/nvcc");
            p.is_file().then(|| p.to_path_buf())
        })
}

fn main() {
    println!("cargo:rerun-if-changed={SRC}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=FV_CUBIN_SMS");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let table = out.join("aot.rs");
    let cuda_on = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let nvcc = if cuda_on { find_nvcc() } else { None };

    let Some(nvcc) = nvcc else {
        if cuda_on {
            println!("cargo:warning=fastvideo-cudarc: no nvcc found; kernels will be NVRTC-compiled at run time (set NVCC=... for ahead-of-time cubins)");
        }
        fs::write(&table, "pub static AOT: &[AotKernel] = &[];\n").expect("write aot.rs");
        return;
    };

    let sms: Vec<u32> = env::var("FV_CUBIN_SMS")
        .unwrap_or_else(|_| DEFAULT_SMS.to_string())
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("FV_CUBIN_SMS entries are integers like 89"))
        .collect();

    let mut entries = String::new();
    for sm in &sms {
        let cubin = out.join(format!("kernels_sm{sm}.cubin"));
        let ptx = out.join(format!("kernels_compute{sm}.ptx"));
        // Same options NVRTC gets: fast math (which implies ftz and fmad).
        for (kind, arch, dest) in [("-cubin", format!("sm_{sm}"), &cubin), ("-ptx", format!("compute_{sm}"), &ptx)] {
            let status = Command::new(&nvcc)
                .args([kind, "-arch", &arch, "-O3", "--use_fast_math", "-o"])
                .arg(dest)
                .arg(SRC)
                .status()
                .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
            assert!(status.success(), "nvcc {kind} -arch={arch} failed for {SRC}");
        }
        entries.push_str(&format!(
            "    AotKernel {{ sm: {sm}, cubin: include_bytes!({:?}), ptx: include_str!({:?}) }},\n",
            cubin.display(),
            ptx.display()
        ));
    }
    fs::write(&table, format!("pub static AOT: &[AotKernel] = &[\n{entries}];\n")).expect("write aot.rs");
    println!("cargo:warning=fastvideo-cudarc: embedded cubins for sm {:?} via {}", sms, nvcc.display());
}
