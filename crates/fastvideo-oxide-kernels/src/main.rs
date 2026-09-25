//! `fv-oxide-aot OUT_DIR [--sm 100,120]` — compile the Tile-IR NVFP4 GEMM to
//! cubins without a GPU. Needs `tileiras` (CUDA >= 13.2) reachable through
//! `CUTILE_TILEIRAS_PATH`, `CUDA_TOOLKIT_PATH`/`CUDA_HOME` or `PATH`.
//!
//! Writes, per SM and [`fastvideo_oxide_kernels::VARIANTS`] entry,
//! `<stem>.cubin` and `<stem>.tileir`, plus `manifest.tsv`
//! (`sm  out  bm  bn  bk  entry  file`) that fastvideo-cudarc's build.rs reads.
//! Any failure exits non-zero: CI must not ship an image without the cubins.

use std::fmt::Write as _;
use std::path::PathBuf;

use cutile::compile_api::KernelCompiler;
use cutile::cutile_compiler::cuda_tile_runtime_utils::{
    run_tileiras, tileiras_fingerprint, TileirasOptions,
};
use cutile::cutile_compiler::specialization::{DivHint, SpecializationBits};
use fastvideo_oxide_kernels::{nvfp4_w4a4, Variant, SMS, VARIANTS};

/// Divisibility facts every launch guarantees (see [`Variant`]): dims and
/// the leading stride divisible by 16 elements, unit inner stride, 16-byte
/// aligned base pointers.
fn spec16() -> SpecializationBits {
    let d16 = DivHint {
        divisor: 16,
        max: 16,
    };
    let d1 = DivHint {
        divisor: 1,
        max: 16,
    };
    SpecializationBits {
        shape_div: vec![d16, d16],
        stride_div: vec![d16, d1],
        stride_one: vec![false, true],
        base_ptr_div: d16,
        elements_disjoint: true,
    }
}

/// `entry @name(` from the Tile IR text.
fn entry_name(ir: &str) -> Option<String> {
    let at = ir.find("entry @")? + "entry @".len();
    let rest = &ir[at..];
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    Some(rest[..end].to_string())
}

fn compile_one(v: &Variant, sm: u32) -> Result<(String, String, String, Vec<u8>), String> {
    let stem = v.stem(sm);
    let names = ["z", "x", "y", "x_scales", "y_scales"];
    let strides: Vec<(&str, &[i32])> = names.iter().map(|n| (*n, &[-1, 1][..])).collect();
    let specs: Vec<(&str, SpecializationBits)> = names.iter().map(|n| (*n, spec16())).collect();
    let artifacts = KernelCompiler::new(
        nvfp4_w4a4::__module_ast_self,
        "nvfp4_w4a4",
        "nvfp4_oxide_w4a4_gemm",
    )
    .generics(v.generics())
    .strides(&strides)
    .spec_args(&specs)
    .target(&format!("sm_{sm}"))
    .compile()
    .map_err(|e| format!("{stem}: Tile IR compile: {e}"))?;
    // cutile moves bounds checks it can state on launch extents out of the
    // kernel and runs them on the host before each JIT launch. The cudarc
    // launcher enforces the stronger `Variant` preconditions instead; the
    // hoisted predicates go into `<stem>.checks` for review.
    let checks: String = artifacts
        .launch_checks()
        .iter()
        .map(|c| format!("{}\t{:?}\n", c.cause, c.predicate))
        .collect();
    let ir = artifacts.ir_text();
    let entry = entry_name(&ir).ok_or_else(|| format!("{stem}: no entry in IR"))?;
    let bc = artifacts
        .bytecode()
        .map_err(|e| format!("{stem}: bytecode: {e}"))?;
    let cubin = run_tileiras(&bc, &format!("sm_{sm}"), &TileirasOptions::default())
        .map_err(|e| format!("{stem}: tileiras: {e}"))?;
    Ok((entry, ir, checks, cubin))
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(
        args.next()
            .ok_or("usage: fv-oxide-aot OUT_DIR [--sm 100,120]")?,
    );
    let mut sms: Vec<u32> = SMS.to_vec();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--sm" => {
                sms = args
                    .next()
                    .ok_or("--sm needs a list")?
                    .split(',')
                    .map(|s| s.trim().trim_start_matches("sm_").parse::<u32>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("--sm: {e}"))?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;
    println!("fv-oxide-aot: tileiras {}", tileiras_fingerprint());
    let mut manifest = String::from("# sm\tout\tbm\tbn\tbk\tentry\tfile\n");
    for &sm in &sms {
        for v in VARIANTS {
            let (entry, ir, checks, cubin) = compile_one(v, sm)?;
            let stem = v.stem(sm);
            let file = format!("{stem}.cubin");
            std::fs::write(out.join(&file), &cubin).map_err(|e| format!("write {file}: {e}"))?;
            std::fs::write(out.join(format!("{stem}.tileir")), &ir)
                .map_err(|e| format!("write {stem}.tileir: {e}"))?;
            std::fs::write(out.join(format!("{stem}.checks")), &checks)
                .map_err(|e| format!("write {stem}.checks: {e}"))?;
            println!("fv-oxide-aot: {file} entry={entry} {} bytes", cubin.len());
            let _ = writeln!(
                manifest,
                "{sm}\t{}\t{}\t{}\t{}\t{entry}\t{file}",
                v.out, v.bm, v.bn, v.bk
            );
        }
    }
    std::fs::write(out.join("manifest.tsv"), manifest).map_err(|e| format!("manifest: {e}"))?;
    Ok(())
}

fn main() {
    // The Tile IR compiler recurses deeply; cutile's own tests run it on a
    // large stack for the same reason.
    let h = std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(run)
        .expect("spawn compile thread");
    match h.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("fv-oxide-aot: {e}");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("fv-oxide-aot: compiler panicked");
            std::process::exit(1);
        }
    }
}
