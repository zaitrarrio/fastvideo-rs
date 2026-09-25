//! NVRTC-compiled CUDA kernels.
//!
//! Compiled once into [`crate::wan::device::DeviceContext`] when CUDA is initialized.
//! Every kernel has a plain-Rust twin in [`super::ops`] (the CPU path) and a
//! parity check in `fv-gpucheck kernels`. Indices are `long` so no tensor this
//! crate sees can overflow them.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaFunction, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::device::{DeviceError, Result};

/// The NVRTC / nvcc source: one file, two consumers. build.rs compiles it
/// ahead of time with nvcc when a toolkit is present (per-SM cubins embedded
/// in the binary); the runtime falls back to NVRTC from this same string
/// when it is not. Keeping them one file is what makes the two paths
/// provably the same code.
const KERNEL_SRC: &str = include_str!("kernels.cu");

/// Declares [`KernelFns`] and [`KERNEL_NAMES`] from one list, so a kernel
/// can't be compiled but not loaded (or vice versa).
macro_rules! kernel_fns {
    ($($name:ident),+ $(,)?) => {
        pub struct KernelFns {
            $(pub $name: CudaFunction,)+
            /// Tile-IR W4A4 GEMM from `scripts/oxide.sh`. None unless a cubin
            /// was embedded for this SM. Launch is still gated off by
            /// `FASTVIDEO_NVFP4_OXIDE_GEMM`.
            pub nvfp4_oxide_w4a4_gemm: Option<CudaFunction>,
        }

        /// Every `__global__` entry point in [`KERNEL_SRC`], in load order.
        pub const KERNEL_NAMES: &[&str] = &[$(stringify!($name)),+];

        impl KernelFns {
            fn load(module: &Arc<cudarc::driver::CudaModule>) -> Result<Self> {
                Ok(Self {
                    $($name: module.load_function(stringify!($name))?,)+
                    nvfp4_oxide_w4a4_gemm: None,
                })
            }
        }
    };
}

kernel_fns!(
    fp8_row_scales,
    fp8_rows_quantize,
    fp8_rows_dequant_bf16,
    affine_quantize,
    affine_dequant,
    affine_w16_gemm,
    nvfp4_w4a4_gemm,
    nvfp4_kv_dequant,
    nvfp4_reconstruct,
    ln_adaln_e_rope_half,
    pad_axis,
    group_norm_stats,
    group_norm_apply,
    gelu_erf,
    leaky_relu,
    snake_beta,
    rope_half,
    apply_rotary_bshd,
    repeat_kv,
    pack_rgb_u8,
    vsa_mma_attn,
    vsa_mma_attn_tma,
    vsa_tile_qkv,
    tanh_scaled,
    quantize_e4m3,
    dequantize_e4m3,
    amax_abs,
    abs_diff_sum,
    e4m3_scale_from_amax,
    elem_add,
    elem_mul,
    elem_sub,
    mul_scalar,
    add_scalar,
    silu,
    swiglu_value_first,
    gelu_tanh,
    clamp_f,
    fill_f,
    lincomb3,
    bcast_binary,
    add_bias_inplace,
    bias_gelu_inplace,
    cast_f32_bf16,
    cast_bf16_f32_bias_act,
    residual_gate_add_e,
    softmax_last,
    softmax_last_bf16,
    rms_norm_last,
    layer_norm_last,
    ln_adaln_e,
    qk_norm_rope_bhsd,
    split_heads_bhsd,
    merge_heads,
    gather_nd,
    block_copy,
    upsample_nearest,
    rms_norm_channels,
    temporal_unfold,
    index_select_rows,
    flash_attn_f32,
    vsa_tile_mean,
    vsa_topk,
    vsa_gather_kv,
    vsa_gather_q,
    vsa_fused_attn,
    vsa_mask_pad,
    vsa_combine,
    sol_tile_sum,
    sol_diag_threshold,
    sol_exact_lists,
    sol_fine_partials,
    sol_coarse_partials,
    sol_lse_combine,
    sol_lse_merge,
    sol_normalize_partials,
    sol_global_h_bar,
    sol_pisa_first_order,
    sol_mma_attn_partials,
);

/// NVRTC-compile the kernel module for `sm_major.sm_minor` without touching a
/// GPU. NVRTC only needs `libnvrtc`, so this runs on any Linux box with the
/// CUDA runtime libraries — a free gate before renting hardware.
/// NVRTC-compile the source for a device. Tries the device's native arch
/// first; an NVRTC too old to know it (libnvrtc 12.x on Blackwell) gets the
/// forward-compatible fallback instead of an error, and the arch actually used
/// is returned so the banner can say so.
pub fn compile_ptx(sm_major: i32, sm_minor: i32) -> Result<(cudarc::nvrtc::Ptx, &'static str)> {
    let mut last = None;
    for arch in super::hopper::nvrtc_arches(sm_major, sm_minor) {
        let opts = CompileOptions {
            arch: Some(arch),
            use_fast_math: Some(true),
            ftz: Some(true),
            // Do not also set `fmad`: use_fast_math already injects --fmad=true.
            ..Default::default()
        };
        match compile_ptx_with_opts(KERNEL_SRC, opts) {
            Ok(ptx) => return Ok((ptx, arch)),
            Err(e) => last = Some(format!("arch={arch}: {e}")),
        }
    }
    Err(DeviceError::Message(format!(
        "nvrtc compile failed for sm_{sm_major}{sm_minor} ({})",
        last.unwrap_or_else(|| "no candidate arch".into())
    )))
}

/// One ahead-of-time compiled target, embedded by build.rs.
pub struct AotKernel {
    /// `sm_major * 10 + sm_minor`, e.g. 89 for Ada, 120 for consumer Blackwell.
    pub sm: u32,
    /// SASS for exactly that SM — no JIT.
    pub cubin: &'static [u8],
    /// PTX for `compute_{sm}`: a forward-compatible fallback for a newer SM
    /// with the same major.
    pub ptx: &'static str,
}

/// Tile-IR cubin from `scripts/oxide.sh` (`cargo oxide build --arch sm_100,sm_120`).
pub struct OxideCubin {
    pub sm: u32,
    pub cubin: &'static [u8],
}

mod aot {
    use super::{AotKernel, OxideCubin};
    include!(concat!(env!("OUT_DIR"), "/aot.rs"));
}

fn load_oxide_gemm(ctx: &Arc<cudarc::driver::CudaContext>, want: u32) -> Option<CudaFunction> {
    let k = aot::OXIDE_AOT.iter().find(|k| k.sm == want)?;
    let path = std::env::temp_dir().join(format!("fv-oxide-{}-sm{want}.cubin", std::process::id()));
    std::fs::write(&path, k.cubin).ok()?;
    let loaded = ctx.load_module(cudarc::nvrtc::Ptx::from_file(&path));
    let _ = std::fs::remove_file(&path);
    let module = loaded.ok()?;
    module
        .load_function("nvfp4_oxide_w4a4_gemm")
        .or_else(|_| module.load_function("linear_tile"))
        .ok()
}

/// Where the loaded kernels came from — reported in the device banner so a
/// run says whether it ran SASS, forward-compatible PTX, or a runtime compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOrigin {
    /// Embedded cubin for exactly this SM.
    Cubin(u32),
    /// Embedded PTX for `compute_{n}`, JIT-compiled by the driver.
    Ptx(u32),
    /// NVRTC at run time, for the given NVRTC arch.
    Nvrtc(&'static str),
}

/// SMs with an embedded cubin, for the compile gate to report.
pub fn aot_sms() -> Vec<u32> {
    aot::AOT.iter().map(|k| k.sm).collect()
}

impl KernelFns {
    /// Load the kernels for a device: exact-SM cubin if one is embedded, else
    /// the highest same-major embedded PTX at or below the device, else NVRTC
    /// from source. `FASTVIDEO_KERNELS=nvrtc` forces the last, so the source
    /// path stays exercised on hardware even once every SM ships a cubin.
    pub fn load_for(
        ctx: &Arc<cudarc::driver::CudaContext>,
        sm_major: i32,
        sm_minor: i32,
    ) -> Result<(Self, KernelOrigin)> {
        let want = (sm_major * 10 + sm_minor) as u32;
        let force_nvrtc = super::envflag::string_flag("FASTVIDEO_KERNELS", "auto") == "nvrtc";
        if !force_nvrtc {
            if let Some(k) = aot::AOT.iter().find(|k| k.sm == want) {
                // cuModuleLoad takes a path; the bytes go through a per-process
                // temp file that is removed once the module is resident.
                let path = std::env::temp_dir()
                    .join(format!("fv-gpucheck-{}-sm{want}.cubin", std::process::id()));
                std::fs::write(&path, k.cubin)
                    .map_err(|e| DeviceError::Message(format!("write {}: {e}", path.display())))?;
                let loaded = ctx.load_module(cudarc::nvrtc::Ptx::from_file(&path));
                let _ = std::fs::remove_file(&path);
                let module = loaded?;
                let mut fns = Self::load(&module)?;
                fns.nvfp4_oxide_w4a4_gemm = load_oxide_gemm(ctx, want);
                return Ok((fns, KernelOrigin::Cubin(want)));
            }
            if let Some(k) = aot::AOT
                .iter()
                .filter(|k| k.sm / 10 == sm_major as u32 && k.sm <= want)
                .max_by_key(|k| k.sm)
            {
                let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(k.ptx))?;
                let mut fns = Self::load(&module)?;
                fns.nvfp4_oxide_w4a4_gemm = load_oxide_gemm(ctx, want);
                return Ok((fns, KernelOrigin::Ptx(k.sm)));
            }
        }
        let (ptx, arch) = compile_ptx(sm_major, sm_minor)?;
        let module = ctx.load_module(ptx)?;
        let mut fns = Self::load(&module)?;
        fns.nvfp4_oxide_w4a4_gemm = load_oxide_gemm(ctx, want);
        Ok((fns, KernelOrigin::Nvrtc(arch)))
    }

    /// NVRTC only — what the compile gate exercises for each named arch.
    pub fn compile(
        ctx: &Arc<cudarc::driver::CudaContext>,
        sm_major: i32,
        sm_minor: i32,
    ) -> Result<Self> {
        let (ptx, _) = compile_ptx(sm_major, sm_minor)?;
        let module = ctx.load_module(ptx)?;
        Self::load(&module)
    }
}

/// One thread per element, 1024 threads per block.
pub fn cfg_n(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n.max(1) as u32)
}

/// One block per row, `ROW_BLOCK_THREADS` threads/block, with dynamic shared
/// memory for the tree reduction. 256 is a power of two (the reduction needs
/// one) and a good fit for the hidden widths this crate sees.
pub const ROW_BLOCK_THREADS: u32 = 256;

pub fn cfg_rows(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.max(1) as u32, 1, 1),
        block_dim: (ROW_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: ROW_BLOCK_THREADS * std::mem::size_of::<f32>() as u32,
    }
}

/// Reduction scratch plus one row of `width` floats (AdaLN+RoPE fusion).
pub fn cfg_rows_with_row(rows: usize, width: usize) -> LaunchConfig {
    let mut cfg = cfg_rows(rows);
    cfg.shared_mem_bytes += width.max(1) as u32 * std::mem::size_of::<f32>() as u32;
    cfg
}

/// Tiled flash attention: grid = bh*sq blocks, block = d threads.
pub fn cfg_flash(bh: usize, sq: usize, d: usize) -> LaunchConfig {
    let d_u = d as u32;
    let smem_bytes = (2 * 32 * d_u + 32 * (d_u / 32)) * std::mem::size_of::<f32>() as u32;
    LaunchConfig {
        grid_dim: ((bh * sq).max(1) as u32, 1, 1),
        block_dim: (d_u, 1, 1),
        shared_mem_bytes: smem_bytes,
    }
}

/// Launch `$f` on `$stream` with `$cfg`, pushing each argument in order.
/// Scalars are passed by reference (`&n`), buffers as `&slice`/`&mut slice`.
macro_rules! launch {
    ($stream:expr, $f:expr, $cfg:expr; $($arg:expr),+ $(,)?) => {
        ({
            use cudarc::driver::PushKernelArg as _;
            super::stats::record_launch();
            let mut builder = $stream.launch_builder($f);
            $( builder.arg($arg); )+
            unsafe { builder.launch($cfg) }.map(|_| ()).map_err(super::device::DeviceError::from)
        })
    };
}
pub(crate) use launch;
