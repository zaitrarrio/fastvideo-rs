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
//! Tile-IR NVFP4 GEMM cubins from `fv-oxide-aot` (crates/fastvideo-oxide-kernels,
//! cutile-rs compiled ahead of time through `tileiras`) are embedded beside the
//! nvcc cubins when `manifest.tsv` exists under `artifacts/oxide/` (or
//! `FV_OXIDE_CUBIN_DIR`). Missing artifacts are skipped so `cargo test` on a
//! Mac still works; `FV_REQUIRE_OXIDE=100,120` (the image build) makes a
//! missing SM a hard error instead.
//!
//! Without `nvcc` the nvcc table is empty and the runtime falls back to NVRTC
//! from the identical source string. That keeps `cargo check`/`cargo test`
//! working on a machine with no toolkit; a warning says the binary will JIT.
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

fn oxide_cubin_dir() -> PathBuf {
    if let Ok(p) = env::var("FV_OXIDE_CUBIN_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/oxide")
}

/// One row of `manifest.tsv` written by `fv-oxide-aot`
/// (`sm  out  bm  bn  bk  entry  file`).
struct OxideRow {
    sm: u32,
    out: String,
    bm: u32,
    bn: u32,
    bk: u32,
    entry: String,
    path: PathBuf,
}

fn find_oxide_cubins() -> Vec<OxideRow> {
    let dir = oxide_cubin_dir();
    let manifest = dir.join("manifest.tsv");
    println!("cargo:rerun-if-changed={}", dir.display());
    println!("cargo:rerun-if-changed={}", manifest.display());
    let Ok(text) = fs::read_to_string(&manifest) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let f: Vec<&str> = line.split('\t').collect();
        assert!(f.len() == 7, "{}: bad row {line:?}", manifest.display());
        let num = |s: &str| -> u32 {
            s.parse()
                .unwrap_or_else(|_| panic!("{}: bad number {s:?}", manifest.display()))
        };
        let path = dir.join(f[6]);
        println!("cargo:rerun-if-changed={}", path.display());
        assert!(
            path.is_file(),
            "{} lists {} but it is missing",
            manifest.display(),
            path.display()
        );
        rows.push(OxideRow {
            sm: num(f[0]),
            out: f[1].to_string(),
            bm: num(f[2]),
            bn: num(f[3]),
            bk: num(f[4]),
            entry: f[5].to_string(),
            path,
        });
    }
    rows
}

/// `FV_REQUIRE_OXIDE=100,120`: fail the build unless a Tile-IR cubin is
/// embedded for every listed SM. Set by the image build, so an image can
/// never ship without the oxide kernels because a stage silently skipped.
fn require_oxide(rows: &[OxideRow]) {
    let Ok(req) = env::var("FV_REQUIRE_OXIDE") else {
        return;
    };
    for sm in req.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let sm: u32 = sm
            .parse()
            .expect("FV_REQUIRE_OXIDE entries are integers like 120");
        assert!(
            rows.iter().any(|r| r.sm == sm),
            "FV_REQUIRE_OXIDE: no oxide Tile-IR cubin for sm_{sm} in {} (run fv-oxide-aot)",
            oxide_cubin_dir().display()
        );
    }
}

fn oxide_table(rows: &[OxideRow]) -> String {
    let mut entries = String::new();
    for r in rows {
        entries.push_str(&format!(
            "    OxideCubin {{ sm: {}, out: {:?}, bm: {}, bn: {}, bk: {}, entry: {:?}, cubin: include_bytes!({:?}) }},\n",
            r.sm,
            r.out,
            r.bm,
            r.bn,
            r.bk,
            r.entry,
            r.path.display()
        ));
    }
    format!("pub static OXIDE_AOT: &[OxideCubin] = &[\n{entries}];\n")
}

fn write_aot(table: &Path, nvcc_entries: &str, oxide: &str) {
    fs::write(
        table,
        format!("pub static AOT: &[AotKernel] = &[\n{nvcc_entries}];\n{oxide}"),
    )
    .expect("write aot.rs");
}

fn main() {
    println!("cargo:rerun-if-changed={SRC}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=FV_CUBIN_SMS");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=FV_OXIDE_CUBIN_DIR");
    println!("cargo:rerun-if-env-changed=FV_REQUIRE_OXIDE");

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let table = out.join("aot.rs");
    let cuda_on = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let nvcc = if cuda_on { find_nvcc() } else { None };
    let oxide_rows = find_oxide_cubins();
    require_oxide(&oxide_rows);
    if !oxide_rows.is_empty() {
        println!(
            "cargo:warning=fastvideo-cudarc: embedded {} oxide Tile-IR cubins for sm {:?}",
            oxide_rows.len(),
            {
                let mut sms: Vec<u32> = oxide_rows.iter().map(|r| r.sm).collect();
                sms.dedup();
                sms
            }
        );
    }
    let oxide = oxide_table(&oxide_rows);

    let Some(nvcc) = nvcc else {
        if cuda_on {
            println!("cargo:warning=fastvideo-cudarc: no nvcc found; kernels will be NVRTC-compiled at run time (set NVCC=... for ahead-of-time cubins)");
        }
        write_aot(&table, "", &oxide);
        return;
    };

    let sms: Vec<u32> = env::var("FV_CUBIN_SMS")
        .unwrap_or_else(|_| DEFAULT_SMS.to_string())
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse()
                .expect("FV_CUBIN_SMS entries are integers like 89")
        })
        .collect();

    let mut entries = String::new();
    for sm in &sms {
        let cubin = out.join(format!("kernels_sm{sm}.cubin"));
        let ptx = out.join(format!("kernels_compute{sm}.ptx"));
        // Same options NVRTC gets: IEEE division/sqrt, no denormal flush, FMA
        // contraction on — how PyTorch's own CUDA kernels are compiled, so
        // elementwise ops round like the reference (no --use_fast_math).
        for (kind, arch, dest) in [
            ("-cubin", format!("sm_{sm}"), &cubin),
            ("-ptx", format!("compute_{sm}"), &ptx),
        ] {
            let status = Command::new(&nvcc)
                .args([
                    kind,
                    "-arch",
                    &arch,
                    "-O3",
                    "--fmad=true",
                    "--prec-div=true",
                    "--prec-sqrt=true",
                    "--ftz=false",
                    "-o",
                ])
                .arg(dest)
                .arg(SRC)
                .status()
                .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
            assert!(
                status.success(),
                "nvcc {kind} -arch={arch} failed for {SRC}"
            );
        }
        entries.push_str(&format!(
            "    AotKernel {{ sm: {sm}, cubin: include_bytes!({:?}), ptx: include_str!({:?}) }},\n",
            cubin.display(),
            ptx.display()
        ));
    }
    write_aot(&table, &entries, &oxide);
    println!(
        "cargo:warning=fastvideo-cudarc: embedded cubins for sm {:?} via {}",
        sms,
        nvcc.display()
    );
}
