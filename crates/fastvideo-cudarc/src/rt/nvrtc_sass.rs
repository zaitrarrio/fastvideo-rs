//! Run-time NVRTC that loads on any driver new enough for the device.
//!
//! `compile_ptx_with_opts` hands the driver PTX to JIT. PTX carries the ISA
//! version of the NVRTC that wrote it, and a driver older than that NVRTC
//! refuses it: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. The runtime image ships
//! NVRTC 13.4, while pods run whatever driver the host has (595.91 in sol-bench
//! phase B, where every LingBot MoE call died on its first kernel).
//!
//! So NVRTC compiles to SASS for the device's own SM first (`-arch=sm_XY`,
//! `nvrtcGetCUBIN`). A cubin needs no JIT and so no PTX ISA support from the
//! driver, only support for the SM, which any driver that created the context
//! has. PTX for the `compute_*` targets of [`super::hopper::nvrtc_arches`]
//! stays as the fallback for an NVRTC too old to know the SM (libnvrtc 12.x
//! on Blackwell): there the driver is the newer one and JIT works.
//! `FASTVIDEO_NVRTC_PTX=1` skips the SASS attempt.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::{result, sys, CompileOptions};

/// Real SM target (`sm_XY`) NVRTC can emit SASS for, or `None` for an SM this
/// table does not know (the PTX fallback then serves it).
pub fn sass_arch(sm_major: i32, sm_minor: i32) -> Option<&'static str> {
    Some(match (sm_major, sm_minor) {
        (7, 5) => "sm_75",
        (8, 0) => "sm_80",
        (8, 6) => "sm_86",
        (8, 7) => "sm_87",
        (8, 9) => "sm_89",
        (9, 0) => "sm_90",
        (10, 0) => "sm_100",
        (10, 3) => "sm_103",
        (11, 0) => "sm_110",
        (12, 0) => "sm_120",
        (12, 1) => "sm_121",
        _ => return None,
    })
}

/// The option strings cudarc 0.17's `CompileOptions::build` passes (it is
/// crate-private), with `arch` as given. Same strings, so a SASS module is
/// compiled with exactly the flags the PTX path used.
fn option_strings(opts: &CompileOptions, arch: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(v) = opts.ftz {
        out.push(format!("--ftz={v}"));
    }
    if let Some(v) = opts.prec_sqrt {
        out.push(format!("--prec-sqrt={v}"));
    }
    if let Some(v) = opts.prec_div {
        out.push(format!("--prec-div={v}"));
    }
    if let Some(v) = opts.fmad {
        out.push(format!("--fmad={v}"));
    }
    if opts.use_fast_math == Some(true) {
        out.push("--fmad=true".into());
    }
    if let Some(n) = opts.maxrregcount {
        out.push(format!("--maxrregcount={n}"));
    }
    for p in &opts.include_paths {
        out.push(format!("--include-path={p}"));
    }
    out.push(format!("--gpu-architecture={arch}"));
    out.extend(opts.options.iter().cloned());
    out
}

/// NVRTC `src` to a cubin for the real target `arch` (`sm_120`, `sm_90a`, ...).
pub fn compile_cubin(src: &str, opts: &CompileOptions, arch: &str) -> Result<Vec<u8>, String> {
    let c_src = CString::new(src).map_err(|e| format!("nvrtc source: {e}"))?;
    let prog = result::create_program(&c_src, None).map_err(|e| format!("nvrtc create: {e:?}"))?;
    let options = option_strings(opts, arch);
    // SAFETY: `prog` comes from `create_program`, `c_src` outlives it, and it
    // is destroyed exactly once below whatever happens in between.
    let out = unsafe {
        match result::compile_program(prog, &options) {
            Err(e) => {
                let log = result::get_program_log(prog)
                    .ok()
                    .map(|l| CStr::from_ptr(l.as_ptr()).to_string_lossy().into_owned())
                    .unwrap_or_default();
                Err(format!("nvrtc {arch} {options:?}: {e:?}\n{log}"))
            }
            Ok(()) => {
                let mut size = 0usize;
                match sys::nvrtcGetCUBINSize(prog, &mut size).result() {
                    Err(e) => Err(format!("nvrtcGetCUBINSize {arch}: {e:?}")),
                    Ok(()) if size == 0 => Err(format!("nvrtc {arch}: empty cubin")),
                    Ok(()) => {
                        let mut buf = vec![0u8; size];
                        sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr().cast())
                            .result()
                            .map(|()| buf)
                            .map_err(|e| format!("nvrtcGetCUBIN {arch}: {e:?}"))
                    }
                }
            }
        }
    };
    // SAFETY: created above, not destroyed yet.
    let _ = unsafe { result::destroy_program(prog) };
    out
}

/// Load cubin bytes. `cuModuleLoad` takes a path, as for the embedded cubins.
pub fn load_cubin(
    ctx: &Arc<CudaContext>,
    cubin: &[u8],
    stem: &str,
) -> Result<Arc<CudaModule>, String> {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "fv-nvrtc-{stem}-{}-{}.cubin",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&path, cubin).map_err(|e| format!("write {}: {e}", path.display()))?;
    let loaded = ctx.load_module(cudarc::nvrtc::Ptx::from_file(&path));
    let _ = std::fs::remove_file(&path);
    loaded.map_err(|e| format!("load {stem} cubin: {e}"))
}

/// What [`load`] ended up loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin {
    /// SASS from `nvrtcGetCUBIN` (`true`) or PTX the driver JIT-compiled.
    pub sass: bool,
    /// The NVRTC `--gpu-architecture`.
    pub arch: &'static str,
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.sass { "sass" } else { "ptx" };
        write!(f, "nvrtc {kind} {}", self.arch)
    }
}

fn ptx_forced() -> bool {
    super::envflag::bool_flag("FASTVIDEO_NVRTC_PTX", false)
}

/// Compile `src` at run time and load it: SASS for `sass` (the device's real
/// target, see [`sass_arch`]) first, then PTX for each of `ptx_arches` in
/// order. `opts.arch` is ignored. Returns the module and where it came from
/// (shown as `nvrtc sass sm_120` / `nvrtc ptx compute_90` in the banners).
pub fn load(
    ctx: &Arc<CudaContext>,
    src: &str,
    opts: &CompileOptions,
    sass: Option<&'static str>,
    ptx_arches: &[&'static str],
    stem: &str,
) -> Result<(Arc<CudaModule>, Origin), String> {
    let mut errors = Vec::new();
    if let (Some(arch), false) = (sass, ptx_forced()) {
        match compile_cubin(src, opts, arch).and_then(|c| load_cubin(ctx, &c, stem)) {
            Ok(m) => return Ok((m, Origin { sass: true, arch })),
            Err(e) => errors.push(e),
        }
    }
    for &arch in ptx_arches {
        let o = CompileOptions {
            arch: Some(arch),
            ..opts.clone()
        };
        match cudarc::nvrtc::compile_ptx_with_opts(src, o) {
            Ok(ptx) => match ctx.load_module(ptx) {
                Ok(m) => return Ok((m, Origin { sass: false, arch })),
                Err(e) => errors.push(format!("load {stem} ptx {arch}: {e}")),
            },
            Err(e) => errors.push(format!("nvrtc {arch}: {e}")),
        }
    }
    Err(if errors.is_empty() {
        format!("{stem}: no candidate arch")
    } else {
        format!("{stem}: {}", errors.join("; "))
    })
}

/// [`load`] for the live device's SM.
pub fn load_for_device(
    ctx: &Arc<CudaContext>,
    sm_major: i32,
    sm_minor: i32,
    src: &str,
    opts: &CompileOptions,
    stem: &str,
) -> Result<(Arc<CudaModule>, Origin), String> {
    load(
        ctx,
        src,
        opts,
        sass_arch(sm_major, sm_minor),
        super::hopper::nvrtc_arches(sm_major, sm_minor),
        stem,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_nvrtc_target_has_a_real_sm() {
        for (maj, min) in [
            (7, 5),
            (8, 0),
            (8, 6),
            (8, 9),
            (9, 0),
            (10, 0),
            (10, 3),
            (12, 0),
            (12, 1),
        ] {
            let arch = sass_arch(maj, min).expect("sass arch");
            assert_eq!(arch, format!("sm_{maj}{min}"));
        }
        assert_eq!(sass_arch(7, 0), None);
    }

    /// With the toolkit's libnvrtc (no GPU needed; `FV_NVRTC_GATE=1`): the
    /// LingBot MoE region and `kernels.cu` compile to SASS (an ELF cubin)
    /// for the RTX PRO 6000's sm_120, the module phase B could not load as PTX.
    #[test]
    fn moe_and_kernels_compile_to_sass_for_sm120() {
        if std::env::var_os("FV_NVRTC_GATE").is_none() {
            eprintln!("skip: set FV_NVRTC_GATE=1 where libnvrtc is installed");
            return;
        }
        let fast = CompileOptions {
            use_fast_math: Some(true),
            ftz: Some(true),
            ..Default::default()
        };
        let moe = compile_cubin(&super::super::ops::moe_kernel_src(), &fast, "sm_120").unwrap();
        assert!(moe.starts_with(b"\x7fELF"), "moe: not a cubin");
        let ieee = CompileOptions {
            use_fast_math: Some(false),
            ftz: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            fmad: Some(true),
            ..Default::default()
        };
        let all = compile_cubin(super::super::kernels::KERNEL_SRC, &ieee, "sm_120").unwrap();
        assert!(all.starts_with(b"\x7fELF"), "kernels.cu: not a cubin");
    }

    /// Same option strings as cudarc's PTX path, with the real target.
    #[test]
    fn options_match_cudarcs_build() {
        let fast = CompileOptions {
            use_fast_math: Some(true),
            ftz: Some(true),
            options: vec!["-std=c++17".into()],
            ..Default::default()
        };
        assert_eq!(
            option_strings(&fast, "sm_120"),
            [
                "--ftz=true",
                "--fmad=true",
                "--gpu-architecture=sm_120",
                "-std=c++17"
            ]
        );
        let ieee = CompileOptions {
            use_fast_math: Some(false),
            ftz: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            fmad: Some(true),
            ..Default::default()
        };
        assert_eq!(
            option_strings(&ieee, "sm_90a"),
            [
                "--ftz=false",
                "--prec-sqrt=true",
                "--prec-div=true",
                "--fmad=true",
                "--gpu-architecture=sm_90a"
            ]
        );
    }
}
