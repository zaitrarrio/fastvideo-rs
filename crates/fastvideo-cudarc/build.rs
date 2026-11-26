//! Ahead-of-time kernel compilation.
//!
//! When an `nvcc` is reachable (`NVCC=/path` or on `PATH`) and the `cuda`
//! feature is on, every SM in `FV_CUBIN_SMS` (default: Turing through
//! Blackwell) gets a cubin and a PTX compiled from the kernel source (the
//! region files of [`KERNEL_PARTS`] concatenated into `$OUT_DIR/kernels.cu`),
//! and a generated `aot.rs` embeds them. The runtime then loads real SASS for the
//! device it finds — no NVRTC on the box, no driver JIT, and the target list is
//! decided here where CI can check it, not at runtime where a missing mapping
//! once compiled every guarded kernel body out on Blackwell.
//!
//! With the `oxide` feature, Tile-IR NVFP4 GEMM cubins from `fv-oxide-aot`
//! (crates/fastvideo-oxide-kernels, cutile-rs compiled ahead of time through
//! `tileiras`) are embedded beside the nvcc cubins when `manifest.tsv` exists
//! under `artifacts/oxide/` (or `FV_OXIDE_CUBIN_DIR`). Missing artifacts are skipped so `cargo test` on a
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
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The MoE module's NVRTC option strings, shared with the runtime
/// (`rt::moe_nvrtc_opts`).
#[path = "src/rt/moe_nvrtc_opts.rs"]
mod moe_nvrtc_opts;

/// The NVRTC / nvcc kernel source, one region per file, in this order. The
/// concatenation is byte for byte the former `src/wan/kernels.cu` (sha256
/// `c9b61cd2b2677292eba5a83b6ec93ba744c69f51de3bec06accbaae9087cc0d3`, checked
/// by `rt::kernel_src`'s tests), and both consumers see only the concatenation:
/// nvcc compiles `$OUT_DIR/kernels.cu` written here, NVRTC gets the same bytes
/// as `concat!(include_str!(..), ..)` from the generated `kernel_src.rs`. Adding
/// a region means appending a file here and nowhere else.
const KERNEL_PARTS: &[&str] = &[
    "src/rt/kernels/core.cu",
    "src/rt/kernels/sol.cu",
    "src/rt/kernels/attn2.cu",
    "src/rt/kernels/attn3.cu",
    "src/rt/kernels/moe.cu",
    "src/rt/kernels/nvfp4.cu",
    "src/ltx2/kernels/dit.cu",
    "src/rt/kernels/act16.cu",
    "src/h3/kernels/block_ops.cu",
    "src/h3/kernels/video_vae.cu",
    "src/ltx2/kernels/video_vae.cu",
    "src/rt/kernels/dit_fusions.cu",
    "src/wan/kernels/block.cu",
    "src/rt/kernels/lora_fuse.cu",
    "src/rt/kernels/conv_igemm.cu",
];
/// The FP8 attention module (`rt::attn_fp8`): its own cubins, sm_89 and newer.
const SRC_FP8: &str = "src/rt/attn_fp8.cu";
/// The SageAttention2-style module (`rt::attn_sage`): same options, sm_89+.
const SRC_SAGE: &str = "src/rt/attn_sage.cu";
/// The VDN-H3 linear-branch module (`h3::vdn`): IEEE math (no fast math), sm_80+.
const SRC_VDN: &str = "src/h3/vdn.cu";
/// Datacenter attention kernels (tcgen05 / wgmma): arch-specific targets only.
const DC_SRC: &str = "src/rt/attn_dc.cu";
/// Sparse FP4 VSA attention (`rt::attn_fp4`, block-scaled `mma.sync` FP4):
/// sm_120a only, built when 120 is in FV_CUBIN_SMS.
const SRC_FP4: &str = "src/rt/attn_fp4.cu";
/// (SM, nvcc arch) pairs attn_dc.cu is built for, when their SM is in
/// FV_CUBIN_SMS. sm_103a (B300) rides on 100: it is the same tcgen05 code,
/// built so a B300 can opt in (`FASTVIDEO_DC_SM103=1`; never run on one).
const DC_ARCHS: &[(u32, &str)] = &[(90, "sm_90a"), (100, "sm_100a"), (103, "sm_103a")];

/// Whether attn_dc.cu is built for `sm` given the FV_CUBIN_SMS list.
fn dc_wanted(sm: u32, sms: &[u32]) -> bool {
    sms.contains(&sm) || (sm == 103 && sms.contains(&100))
}
const DEFAULT_SMS: &str = "75,80,86,89,90,100,120";
/// What `rt::ops::moe_kernel_src()` puts before the MoE region; the
/// `moe_aot_source_is_moe_kernel_src` test keeps the two in step.
const MOE_PREFIX: &str =
    "#ifndef IDX\n#define IDX() ((long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x)\n#endif\n";
const MOE_BEGIN: &str = "// ==== region: moe ====";
const MOE_END: &str = "// ==== end region: moe ====";

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

/// `rerun-if-changed` on `path`, or on its nearest existing ancestor.
fn watch_existing(path: &Path) {
    let mut p = path;
    while !p.exists() {
        match p.parent() {
            Some(q) if !q.as_os_str().is_empty() => p = q,
            _ => return,
        }
    }
    println!("cargo:rerun-if-changed={}", p.display());
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
    // Cargo treats a missing rerun-if-changed path as changed on every build,
    // which re-ran this script (all the nvcc cubins) on each no-op `cargo
    // build --features cuda` when artifacts/oxide does not exist. Watch the
    // nearest existing ancestor instead: its mtime changes when the dir appears.
    watch_existing(&dir);
    if manifest.exists() {
        println!("cargo:rerun-if-changed={}", manifest.display());
    }
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

fn write_aot(table: &Path, nvcc_entries: &str, oxide: &str, dc_entries: &str) {
    fs::write(
        table,
        format!(
            "pub static AOT: &[AotKernel] = &[\n{nvcc_entries}];\n{oxide}\
             pub static DC_AOT: &[(u32, &[u8])] = &[\n{dc_entries}];\n"
        ),
    )
    .expect("write aot.rs");
}

/// `link-static`: cudarc's `static-linking` build asks the linker for
/// `libcudnn.a` (`static:+whole-archive=cudnn`), which NVIDIA's licence does
/// not let us ship (ADR-0024), and for `static:+whole-archive=stdc++`. rustc
/// resolves both when it builds cudarc's rlib, from cudarc's own search
/// list, so the build (docker/gpucheck.Dockerfile `static-build`,
/// scripts/dev/release-artifacts-pod.sh) points `CUDA_TOOLKIT_ROOT_DIR` at a
/// root whose `lib64` holds an empty `libcudnn.a` (the ar magic alone is a
/// valid archive: GNU ld, lld and mold accept it) and a link to the C++
/// compiler's `libstdc++.a`. This crate then links the shared cuDNN by its
/// exact file name (`-l:libcudnn.so.9`; a plain `-lcudnn` would find the
/// empty shim first), so every cuDNN symbol resolves from the library the
/// image carries: a `cuda-static` fv-serve links cuBLAS, cuBLASLt, cudart,
/// NVRTC and CUPTI statically and loads libcudnn.so.9 as before (its own
/// cuBLASLt / NVRTC needs stay with the shared libraries until ADR-0024
/// step 3 removes cuDNN). The build host needs libcudnn.so.9 on the linker's
/// path (the cuDNN runtime package).
fn cudnn_shared() {
    println!("cargo:rustc-link-lib=dylib:+verbatim=libcudnn.so.9");
    match env::var("CUDA_TOOLKIT_ROOT_DIR") {
        Ok(root) if Path::new(&root).join("lib64/libcudnn.a").is_file() => {}
        _ => println!(
            "cargo:warning=fastvideo-cudarc link-static: CUDA_TOOLKIT_ROOT_DIR does not name a root with lib64/libcudnn.a (the empty shim) and lib64/libstdc++.a; cudarc's rlib will not link (docs/serve/images.md, the cuda-static image)"
        ),
    }
}

fn main() {
    for part in KERNEL_PARTS {
        println!("cargo:rerun-if-changed={part}");
    }
    println!("cargo:rerun-if-changed={SRC_FP8}");
    println!("cargo:rerun-if-changed={SRC_SAGE}");
    println!("cargo:rerun-if-changed={SRC_VDN}");
    println!("cargo:rerun-if-changed={DC_SRC}");
    println!("cargo:rerun-if-changed={SRC_FP4}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=FV_CUBIN_SMS");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LINK_STATIC");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
    if env::var_os("CARGO_FEATURE_LINK_STATIC").is_some() {
        cudnn_shared();
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OXIDE");
    println!("cargo:rerun-if-env-changed=FV_OXIDE_CUBIN_DIR");
    println!("cargo:rerun-if-env-changed=FV_REQUIRE_OXIDE");
    println!("cargo:rerun-if-env-changed=FV_DC_PTXAS_VERBOSE");

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let kernel_cu = write_kernel_src(&out);
    let table = out.join("aot.rs");
    let fp8_table = out.join("aot_attn_fp8.rs");
    let sage_table = out.join("aot_attn_sage.rs");
    let vdn_table = out.join("aot_h3_vdn.rs");
    let fp4_table = out.join("aot_attn_fp4.rs");
    let moe_table = out.join("aot_moe.rs");
    let moe_cu = write_moe_src(&out, &kernel_cu);
    let cuda_on = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let nvcc = if cuda_on { find_nvcc() } else { None };
    // The oxide Tile-IR cubins are embedded with the `oxide` feature only
    // (ADR-0015); without it FV_REQUIRE_OXIDE has nothing to check.
    let oxide_on = env::var_os("CARGO_FEATURE_OXIDE").is_some();
    let oxide_rows = if oxide_on {
        find_oxide_cubins()
    } else {
        Vec::new()
    };
    if oxide_on {
        require_oxide(&oxide_rows);
    } else if env::var_os("FV_REQUIRE_OXIDE").is_some() {
        println!(
            "cargo:warning=fastvideo-cudarc: FV_REQUIRE_OXIDE ignored: built without the `oxide` feature (no Tile-IR cubins embedded)"
        );
    }
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
        write_fp8(&fp8_table, "");
        write_module_table(&sage_table, "AOT_ATTN_SAGE", "");
        write_module_table(&vdn_table, "AOT_H3_VDN", "");
        write_cubin_table(&fp4_table, "AOT_ATTN_FP4", "");
        write_moe_table(&moe_table, "", None, "");
        write_aot(&table, "", &oxide, "");
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
                .arg(&kernel_cu)
                .status()
                .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
            assert!(
                status.success(),
                "nvcc {kind} -arch={arch} failed for {}",
                kernel_cu.display()
            );
        }
        entries.push_str(&format!(
            "    AotKernel {{ sm: {sm}, cubin: include_bytes!({:?}), ptx: include_str!({:?}) }},\n",
            cubin.display(),
            ptx.display()
        ));
    }
    // attn_dc.cu: one cubin per arch-specific target whose SM is listed. No
    // PTX: sm_90a / sm_100a code runs on exactly that SM, so there is nothing
    // to fall forward to.
    let mut dc_entries = String::new();
    for &(sm, arch) in DC_ARCHS.iter().filter(|(sm, _)| dc_wanted(*sm, &sms)) {
        let cubin = out.join(format!("attn_dc_{arch}.cubin"));
        // FV_DC_PTXAS_VERBOSE=1: ptxas register / spill / smem report per
        // kernel, surfaced as cargo warnings (kernel work on the build pod).
        let verbose = env::var("FV_DC_PTXAS_VERBOSE").is_ok_and(|v| v == "1");
        let mut cmd = Command::new(&nvcc);
        cmd.args([
            "-cubin",
            "-arch",
            arch,
            "-O3",
            "--fmad=true",
            "--prec-div=true",
            "--prec-sqrt=true",
            "--ftz=false",
        ]);
        if verbose {
            cmd.arg("-Xptxas=-v");
        }
        let res = cmd
            .arg("-o")
            .arg(&cubin)
            .arg(DC_SRC)
            .output()
            .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
        let text = String::from_utf8_lossy(&res.stderr);
        if verbose || !res.status.success() {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                println!("cargo:warning=attn_dc {arch}: {line}");
            }
        }
        assert!(
            res.status.success(),
            "nvcc -cubin -arch={arch} failed for {DC_SRC}:\n{text}"
        );
        dc_entries.push_str(&format!(
            "    ({sm}, include_bytes!({:?})),\n",
            cubin.display()
        ));
    }
    write_aot(&table, &entries, &oxide, &dc_entries);
    write_fp8(&fp8_table, &fp8_entries(&nvcc, &out, &sms));
    write_module_table(
        &sage_table,
        "AOT_ATTN_SAGE",
        &module_entries(&nvcc, &out, &sms, SRC_SAGE, "attn_sage"),
    );
    write_module_table(
        &vdn_table,
        "AOT_H3_VDN",
        &precise_module_entries(&nvcc, &out, &sms, SRC_VDN, "h3_vdn"),
    );
    write_cubin_table(&fp4_table, "AOT_ATTN_FP4", &fp4_entries(&nvcc, &out, &sms));
    let nvrtc = Nvrtc::load(&nvcc);
    let moe_nvrtc = nvrtc_version(&nvrtc);
    write_moe_table(
        &moe_table,
        &moe_entries(&nvrtc, &out, &sms, &moe_cu),
        Some(&sms),
        &moe_nvrtc,
    );
    println!(
        "cargo:warning=fastvideo-cudarc: embedded cubins for sm {:?} via {}",
        sms,
        nvcc.display()
    );
}

/// Concatenate [`KERNEL_PARTS`] into `$OUT_DIR/kernels.cu` (nvcc's input) and
/// write `$OUT_DIR/kernel_src.rs`, which spells the same concatenation as a
/// `concat!` of `include_str!`s for the NVRTC fallback (`rt::kernel_src`).
/// One list, two consumers, so the two paths cannot drift apart.
fn write_kernel_src(out: &Path) -> PathBuf {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let mut cat = Vec::new();
    let mut rs = String::from(
        "/// The kernel source: `build.rs`'s `KERNEL_PARTS` concatenated in order.\n\
         #[cfg_attr(not(feature = \"cuda\"), allow(dead_code))]\n\
         pub(crate) const KERNEL_SRC: &str = concat!(\n",
    );
    for part in KERNEL_PARTS {
        let path = root.join(part);
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        cat.extend_from_slice(&bytes);
        rs.push_str(&format!("    include_str!({:?}),\n", path.display()));
    }
    rs.push_str(");\n");
    let cu = out.join("kernels.cu");
    fs::write(&cu, cat).unwrap_or_else(|e| panic!("write {}: {e}", cu.display()));
    let gen = out.join("kernel_src.rs");
    fs::write(&gen, rs).unwrap_or_else(|e| panic!("write {}: {e}", gen.display()));
    cu
}

/// `$OUT_DIR/moe_aot.cu`: the MoE region of `kernels.cu` with the prefix
/// `rt::ops::moe_kernel_src()` adds, i.e. exactly the string NVRTC compiles.
fn write_moe_src(out: &Path, kernel_cu: &Path) -> PathBuf {
    let all = fs::read_to_string(kernel_cu)
        .unwrap_or_else(|e| panic!("read {}: {e}", kernel_cu.display()));
    let start = all.find(MOE_BEGIN).expect("kernels.cu moe region");
    let end = all.find(MOE_END).expect("kernels.cu moe end");
    let cu = out.join("moe_aot.cu");
    fs::write(&cu, format!("{MOE_PREFIX}{}", &all[start..end]))
        .unwrap_or_else(|e| panic!("write {}: {e}", cu.display()));
    cu
}

/// `$OUT_DIR/aot_moe.rs`: `AOT_MOE` (sm, cubin, ptx), whether it was built
/// (only where nvcc, i.e. the toolkit, was found), the SMs it was asked for
/// (the coverage test's input) and the libnvrtc that compiled it.
fn write_moe_table(table: &Path, entries: &str, sms: Option<&[u32]>, nvrtc: &str) {
    let list = sms
        .unwrap_or(&[])
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(
        table,
        format!(
            "pub static AOT_MOE: &[(u32, &[u8], &str)] = &[\n{entries}];\n\
             pub const AOT_MOE_BUILT: bool = {};\n\
             pub const AOT_MOE_SMS: &[u32] = &[{list}];\n\
             pub const AOT_MOE_NVRTC: &str = {nvrtc:?};\n",
            sms.is_some()
        ),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", table.display()));
}

/// The LingBot MoE module (`moe_aot.cu`) for every SM in the list, cubin +
/// PTX, compiled by NVRTC itself (the pinned libnvrtc of the build image, the
/// same 13.4 the runtime image ships) with exactly the option strings the
/// runtime NVRTC path passes ([`moe_nvrtc_opts::MOE_NVRTC_OPTIONS`]): owner
/// decision FVID-2026-11-12-moe-aot-build-time-nvrtc, ADR-0024 step 1a. The
/// cubin is then the bytes the runtime would have compiled, so the served
/// outputs cannot move; fv-gpucheck's `moe_aot` group checks it bit for bit.
/// (nvcc `--use_fast_math --ftz=true` did not: 4 of 12 checks differed on an
/// RTX PRO 6000, because cudarc's `use_fast_math` only adds `--fmad=true`.)
fn moe_entries(nvrtc: &Nvrtc, out: &Path, sms: &[u32], src: &Path) -> String {
    let text = fs::read_to_string(src).unwrap_or_else(|e| panic!("read {}: {e}", src.display()));
    let mut entries = String::new();
    for sm in sms {
        let cubin = out.join(format!("moe_sm{sm}.cubin"));
        let ptx = out.join(format!("moe_compute{sm}.ptx"));
        let bytes = nvrtc_compile(nvrtc, &text, &format!("sm_{sm}"), true);
        fs::write(&cubin, bytes).unwrap_or_else(|e| panic!("write {}: {e}", cubin.display()));
        let bytes = nvrtc_compile(nvrtc, &text, &format!("compute_{sm}"), false);
        fs::write(&ptx, bytes).unwrap_or_else(|e| panic!("write {}: {e}", ptx.display()));
        entries.push_str(&format!(
            "    ({sm}, include_bytes!({:?}), include_str!({:?})),\n",
            cubin.display(),
            ptx.display()
        ));
    }
    entries
}

/// libnvrtc at build time, dlopened (`libloading`) from the toolkit `nvcc`
/// belongs to (`<nvcc>/../../lib64`, `.../targets/x86_64-linux/lib`), else
/// `CUDA_HOME` / `CUDA_ROOT` / `CUDA_PATH`, `/usr/local/cuda`, else the
/// loader's own search path. Only the entry points the MoE table needs.
struct Nvrtc {
    _lib: libloading::Library,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
    create: unsafe extern "C" fn(
        *mut *mut c_void,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    compile: unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int,
    log_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    log: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    cubin_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    cubin: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    ptx_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    ptx: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    destroy: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
}

impl Nvrtc {
    fn load(nvcc: &Path) -> Self {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(root) = nvcc.parent().and_then(Path::parent) {
            candidates.push(root.join("lib64/libnvrtc.so"));
            candidates.push(root.join("targets/x86_64-linux/lib/libnvrtc.so"));
        }
        for root in [
            env::var("CUDA_HOME"),
            env::var("CUDA_ROOT"),
            env::var("CUDA_PATH"),
        ]
        .into_iter()
        .flatten()
        {
            candidates.push(Path::new(&root).join("lib64/libnvrtc.so"));
        }
        candidates.push(PathBuf::from("/usr/local/cuda/lib64/libnvrtc.so"));
        let mut tried = Vec::new();
        let lib = candidates
            .iter()
            .filter(|p| p.is_file())
            .find_map(|p| {
                // SAFETY: libnvrtc runs no unsound initialisers.
                match unsafe { libloading::Library::new(p) } {
                    Ok(l) => Some(l),
                    Err(e) => {
                        tried.push(format!("{}: {e}", p.display()));
                        None
                    }
                }
            })
            .or_else(|| unsafe { libloading::Library::new("libnvrtc.so") }.ok())
            .unwrap_or_else(|| {
                panic!(
                    "the MoE AOT table needs libnvrtc at build time (nvcc was found at {}, so the \
                     CUDA toolkit is expected): libnvrtc.so did not load (tried {:?}, {tried:?})",
                    nvcc.display(),
                    candidates
                )
            });
        macro_rules! sym {
            ($name:literal) => {
                // SAFETY: the signatures above are libnvrtc's (nvrtc.h).
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                    .unwrap_or_else(|e| panic!("libnvrtc: no {}: {e}", $name))
            };
        }
        Self {
            version: sym!("nvrtcVersion"),
            create: sym!("nvrtcCreateProgram"),
            compile: sym!("nvrtcCompileProgram"),
            log_size: sym!("nvrtcGetProgramLogSize"),
            log: sym!("nvrtcGetProgramLog"),
            cubin_size: sym!("nvrtcGetCUBINSize"),
            cubin: sym!("nvrtcGetCUBIN"),
            ptx_size: sym!("nvrtcGetPTXSize"),
            ptx: sym!("nvrtcGetPTX"),
            destroy: sym!("nvrtcDestroyProgram"),
            error_string: sym!("nvrtcGetErrorString"),
            _lib: lib,
        }
    }

    fn err(&self, r: c_int) -> String {
        // SAFETY: nvrtcGetErrorString returns a static C string for any code.
        unsafe { CStr::from_ptr((self.error_string)(r)) }
            .to_string_lossy()
            .into_owned()
    }

    /// A `(size getter, bytes getter)` pair as the bytes (the trailing NUL of
    /// a text result, PTX or the log, dropped).
    fn bytes(
        &self,
        prog: *mut c_void,
        size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
        get: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
        what: &str,
        text: bool,
    ) -> Vec<u8> {
        let mut n = 0usize;
        // SAFETY: `prog` is a live program, `n` a valid out-pointer, and the
        // buffer is as large as the size getter asked for.
        unsafe {
            let r = size(prog, &mut n);
            assert!(r == 0, "{what} size: {}", self.err(r));
            let mut buf = vec![0u8; n];
            if n > 0 {
                let r = get(prog, buf.as_mut_ptr().cast());
                assert!(r == 0, "{what}: {}", self.err(r));
            }
            if text && buf.last() == Some(&0) {
                buf.pop();
            }
            buf
        }
    }
}

/// The libnvrtc version the MoE table was compiled with, as `major.minor`.
fn nvrtc_version(nvrtc: &Nvrtc) -> String {
    let (mut major, mut minor) = (0 as c_int, 0 as c_int);
    // SAFETY: two valid out-pointers.
    let r = unsafe { (nvrtc.version)(&mut major, &mut minor) };
    assert!(r == 0, "nvrtcVersion: {}", nvrtc.err(r));
    format!("{major}.{minor}")
}

/// NVRTC `src` for `arch` with the MoE option strings: a cubin for a real
/// `sm_XY` (`sass`), PTX for a `compute_XY`. Any failure is a build error.
fn nvrtc_compile(nvrtc: &Nvrtc, src: &str, arch: &str, sass: bool) -> Vec<u8> {
    let c_src = CString::new(src).expect("moe source has no NUL");
    let options: Vec<CString> = moe_nvrtc_opts::MOE_NVRTC_OPTIONS
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once(format!("--gpu-architecture={arch}")))
        .map(|s| CString::new(s).expect("nvrtc option has no NUL"))
        .collect();
    let opt_ptrs: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();
    let mut prog: *mut c_void = std::ptr::null_mut();
    // SAFETY: `c_src` and `options` outlive the program; `prog` comes from
    // nvrtcCreateProgram and is destroyed exactly once at the end.
    let out = unsafe {
        let r = (nvrtc.create)(
            &mut prog,
            c_src.as_ptr(),
            c"moe.cu".as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        assert!(r == 0, "nvrtc create (moe {arch}): {}", nvrtc.err(r));
        let r = (nvrtc.compile)(prog, opt_ptrs.len() as c_int, opt_ptrs.as_ptr());
        if r != 0 {
            let log =
                String::from_utf8_lossy(&nvrtc.bytes(prog, nvrtc.log_size, nvrtc.log, "log", true))
                    .into_owned();
            panic!(
                "nvrtc moe {arch} {:?}: {}\n{log}",
                moe_nvrtc_opts::MOE_NVRTC_OPTIONS,
                nvrtc.err(r)
            );
        }
        let bytes = if sass {
            let b = nvrtc.bytes(
                prog,
                nvrtc.cubin_size,
                nvrtc.cubin,
                &format!("nvrtcGetCUBIN moe {arch}"),
                false,
            );
            assert!(!b.is_empty(), "nvrtc moe {arch}: empty cubin");
            b
        } else {
            nvrtc.bytes(
                prog,
                nvrtc.ptx_size,
                nvrtc.ptx,
                &format!("nvrtc ptx moe {arch}"),
                true,
            )
        };
        let _ = (nvrtc.destroy)(&mut prog);
        bytes
    };
    out
}

fn write_fp8(table: &Path, entries: &str) {
    write_module_table(table, "AOT_ATTN_FP8", entries);
}

fn write_cubin_table(table: &Path, name: &str, entries: &str) {
    fs::write(
        table,
        format!("pub static {name}: &[(u32, &[u8])] = &[\n{entries}];\n"),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", table.display()));
}

/// `attn_fp4.cu` as an sm_120a cubin when 120 is listed (the block-scaled FP4
/// `mma.sync` is arch-specific: no PTX to fall forward to). Fast math as the
/// other attention modules; `FV_DC_PTXAS_VERBOSE=1` reports registers / spills.
fn fp4_entries(nvcc: &Path, out: &Path, sms: &[u32]) -> String {
    if !sms.contains(&120) {
        return String::new();
    }
    let cubin = out.join("attn_fp4_sm_120a.cubin");
    let verbose = env::var("FV_DC_PTXAS_VERBOSE").is_ok_and(|v| v == "1");
    let mut cmd = Command::new(nvcc);
    cmd.args([
        "-cubin",
        "-arch",
        "sm_120a",
        "-O3",
        "-std=c++17",
        "--use_fast_math",
    ]);
    if verbose {
        cmd.arg("-Xptxas=-v");
    }
    let res = cmd
        .arg("-o")
        .arg(&cubin)
        .arg(SRC_FP4)
        .output()
        .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
    let text = String::from_utf8_lossy(&res.stderr);
    if verbose || !res.status.success() {
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            println!("cargo:warning=attn_fp4 sm_120a: {line}");
        }
    }
    assert!(
        res.status.success(),
        "nvcc -cubin -arch=sm_120a failed for {SRC_FP4}:\n{text}"
    );
    format!("    (120, include_bytes!({:?})),\n", cubin.display())
}

fn write_module_table(table: &Path, name: &str, entries: &str) {
    fs::write(
        table,
        format!("pub static {name}: &[(u32, &[u8], &str)] = &[\n{entries}];\n"),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", table.display()));
}

/// `attn_fp8.cu` for every SM in the list with FP8 `mma.sync` (sm_89+), with
/// the options its NVRTC fallback uses (fast math, FTZ, C++17). A driver
/// older than the box's NVRTC refuses NVRTC's PTX, so the cubin is the path.
fn fp8_entries(nvcc: &Path, out: &Path, sms: &[u32]) -> String {
    module_entries(nvcc, out, sms, SRC_FP8, "attn_fp8")
}

/// An sm_80+ module (`src`) compiled like `kernels.cu` (IEEE division and
/// square root, no denormal flush, FMA contraction on), cubin + PTX per SM.
fn precise_module_entries(nvcc: &Path, out: &Path, sms: &[u32], src: &str, stem: &str) -> String {
    let mut entries = String::new();
    for sm in sms.iter().filter(|&&s| s >= 80) {
        let cubin = out.join(format!("{stem}_sm{sm}.cubin"));
        let ptx = out.join(format!("{stem}_compute{sm}.ptx"));
        for (kind, arch, dest) in [
            ("-cubin", format!("sm_{sm}"), &cubin),
            ("-ptx", format!("compute_{sm}"), &ptx),
        ] {
            let status = Command::new(nvcc)
                .args([
                    kind,
                    "-arch",
                    &arch,
                    "-O3",
                    "-std=c++17",
                    "--fmad=true",
                    "--prec-div=true",
                    "--prec-sqrt=true",
                    "--ftz=false",
                    "-o",
                ])
                .arg(dest)
                .arg(src)
                .status()
                .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
            assert!(
                status.success(),
                "nvcc {kind} -arch={arch} failed for {src}"
            );
        }
        entries.push_str(&format!(
            "    ({sm}, include_bytes!({:?}), include_str!({:?})),\n",
            cubin.display(),
            ptx.display()
        ));
    }
    entries
}

/// An sm_89+ attention module (`src`) as cubin + PTX per SM, named `stem`.
fn module_entries(nvcc: &Path, out: &Path, sms: &[u32], src: &str, stem: &str) -> String {
    let mut entries = String::new();
    for sm in sms.iter().filter(|&&s| s >= 89) {
        let cubin = out.join(format!("{stem}_sm{sm}.cubin"));
        let ptx = out.join(format!("{stem}_compute{sm}.ptx"));
        for (kind, arch, dest) in [
            ("-cubin", format!("sm_{sm}"), &cubin),
            ("-ptx", format!("compute_{sm}"), &ptx),
        ] {
            let status = Command::new(nvcc)
                .args([
                    kind,
                    "-arch",
                    &arch,
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-o",
                ])
                .arg(dest)
                .arg(src)
                .status()
                .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
            assert!(
                status.success(),
                "nvcc {kind} -arch={arch} failed for {src}"
            );
        }
        entries.push_str(&format!(
            "    ({sm}, include_bytes!({:?}), include_str!({:?})),\n",
            cubin.display(),
            ptx.display()
        ));
    }
    entries
}
