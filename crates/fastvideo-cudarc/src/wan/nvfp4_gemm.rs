//! NVFP4 (E2M1 + E4M3 16-block scales) GEMM backends on Blackwell, at kernel
//! level only — no model path calls these yet.
//!
//! - **cuBLASLt block-scaled FP4** ([`LtNvfp4Plan`]): `CUDA_R_4F_E2M1` A/B
//!   with `CUBLASLT_MATMUL_DESC_{A,B}_SCALE_MODE = VEC16_UE4M3`, FP32
//!   accumulate, bf16 out. cuBLASLt wants the scales in its 128x4 tiled
//!   layout ([`swizzle_scales_device`]); the tensor-level decode factors ride
//!   in `alpha`.
//! - **oxide Tile-IR** ([`oxide_gemm`]): the cutile-rs kernel in
//!   `crates/fastvideo-oxide-kernels`, compiled ahead of time to cubins by
//!   `fv-oxide-aot` and embedded by build.rs. It reads the plain row-major
//!   scales, so it needs no swizzle.
//!
//! Operand layout for both (the host recipe in `fastvideo_models::nvfp4`):
//! activations `x[M, K]` and weight `w[N, K]` packed two codes per byte, low
//! nibble first; scales `[rows, K/16]`; `C[M, N] = x @ wᵀ`.

use cudarc::cublaslt::sys as lt;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use fastvideo_models::nvfp4::ScaleRule;

use super::device::{self, DeviceContext};
use super::fp8::{check, set_attr, Desc, Layout, LtContext, Pref};
use super::kernels::{cfg_n, launch, OxideGemm};
use super::tensor::{Result, TensorError};

fn err(e: impl std::fmt::Display) -> TensorError {
    TensorError::Message(e.to_string())
}

fn dev() -> Result<std::sync::Arc<DeviceContext>> {
    device::global_device().ok_or_else(|| err("no global CUDA device context"))
}

/// A packed NVFP4 tensor on the device.
pub struct Nvfp4Dev {
    /// `[rows, cols/2]`, low nibble first.
    pub packed: CudaSlice<u8>,
    /// `[rows, cols/16]` E4M3 codes, row-major.
    pub scales: CudaSlice<u8>,
    /// One-element tensor amax.
    pub amax: CudaSlice<f32>,
    pub rows: usize,
    pub cols: usize,
    pub rule: ScaleRule,
}

impl Nvfp4Dev {
    /// `amax / (e2m1_max * e4m3_max)` (reads the device amax back).
    pub fn decode(&self) -> Result<f32> {
        let amax = dev()?.stream.memcpy_dtov(&self.amax).map_err(err)?[0];
        Ok(fastvideo_models::nvfp4::dequant_factor(amax, self.rule))
    }
}

/// Device twin of `fastvideo_models::nvfp4::quantize`: tensor amax, then per
/// 16-block E4M3 scale and E2M1 codes. Bit-identical to the host.
pub fn quantize_device(
    x: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
    rule: ScaleRule,
) -> Result<Nvfp4Dev> {
    let d = dev()?;
    if cols == 0 || !cols.is_multiple_of(16) || x.len() != rows * cols {
        return Err(err(format!(
            "nvfp4 quantize: {} elements for [{rows}, {cols}] (cols % 16 == 0)",
            x.len()
        )));
    }
    let n = x.len() as i64;
    let mut amax = d.stream.alloc_zeros::<f32>(1).map_err(err)?;
    let cfg_amax = LaunchConfig {
        grid_dim: (x.len().div_ceil(256).clamp(1, 1024) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    launch!(d.stream, &d.kernels.amax_abs, cfg_amax; x, &mut amax, &n).map_err(err)?;
    let mut packed = unsafe { d.stream.alloc::<u8>((rows * cols / 2).max(1)) }.map_err(err)?;
    let mut scales = unsafe { d.stream.alloc::<u8>((rows * cols / 16).max(1)) }.map_err(err)?;
    let rule_i = match rule {
        ScaleRule::Static6 => 0i32,
        ScaleRule::Static4 => 1,
        ScaleRule::Mse => 2,
    };
    let (rr, cc) = (rows as i64, cols as i64);
    launch!(
        d.stream, &d.kernels.nvfp4_quantize_pack, cfg_n(rows * cols / 16);
        x, &mut packed, &mut scales, &amax, &rr, &cc, &rule_i
    )
    .map_err(err)?;
    Ok(Nvfp4Dev {
        packed,
        scales,
        amax,
        rows,
        cols,
        rule,
    })
}

/// Row-major `[rows, cols/16]` scales → cuBLASLt `VEC16_UE4M3` tiled layout
/// (rows padded to 128, scale columns to 4, zero-filled).
pub fn swizzle_scales_device(
    scales: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
) -> Result<CudaSlice<u8>> {
    let d = dev()?;
    let sc = cols / 16;
    if scales.len() != rows * sc {
        return Err(err(format!(
            "nvfp4 swizzle: {} scales for [{rows}, {sc}]",
            scales.len()
        )));
    }
    let (rp, cp) = (rows.div_ceil(128) * 128, sc.div_ceil(4) * 4);
    let mut out = unsafe { d.stream.alloc::<u8>((rp * cp).max(1)) }.map_err(err)?;
    let (r, s, rp_, cp_) = (rows as i64, sc as i64, rp as i64, cp as i64);
    launch!(
        d.stream, &d.kernels.nvfp4_scales_swizzle, cfg_n(rp * cp);
        scales, &mut out, &r, &s, &rp_, &cp_
    )
    .map_err(err)?;
    Ok(out)
}

/// Host twin of [`swizzle_scales_device`], for the parity check.
pub fn swizzle_scales_host(scales: &[u8], rows: usize, sc: usize) -> Vec<u8> {
    let (rp, cp) = (rows.div_ceil(128) * 128, sc.div_ceil(4) * 4);
    let mut out = vec![0u8; rp * cp];
    for r in 0..rows {
        for c in 0..sc {
            let tile = (r / 128) * (cp / 4) + c / 4;
            let rr = r % 128;
            out[tile * 512 + (rr % 32) * 16 + (rr / 32) * 4 + c % 4] = scales[r * sc + c];
        }
    }
    out
}

/// Deterministic uniform `[-scale, scale)` fill (benchmark operands).
pub fn fill_uniform_device(n: usize, seed: u64, scale: f32) -> Result<CudaSlice<f32>> {
    let d = dev()?;
    let mut out = unsafe { d.stream.alloc::<f32>(n.max(1)) }.map_err(err)?;
    let nn = n as i64;
    launch!(d.stream, &d.kernels.fill_hash_uniform, cfg_n(n); &mut out, &nn, &seed, &scale)
        .map_err(err)?;
    Ok(out)
}

// ---- oxide Tile-IR -----------------------------------------------------------

/// Loaded oxide GEMMs with output type `out` (`"bf16"` / `"f32"`).
pub fn oxide_variants(out: &str) -> Result<Vec<OxideGemm>> {
    let d = dev()?;
    if let Some(e) = &d.kernels.oxide_error {
        return Err(err(format!("oxide cubins failed to load: {e}")));
    }
    Ok(d.kernels
        .oxide_gemms
        .iter()
        .filter(|g| g.out == out)
        .cloned()
        .collect())
}

/// Whether `g` can run `[m, k] @ [n, k]ᵀ` (the divisibility the cubin was
/// compiled to assume; see `fastvideo_oxide_kernels::Variant`).
pub fn oxide_fits(g: &OxideGemm, m: usize, n: usize, k: usize) -> std::result::Result<(), String> {
    let (bm, bn, bk) = (g.bm as usize, g.bn as usize, g.bk as usize);
    if m == 0
        || n == 0
        || !m.is_multiple_of(bm)
        || !n.is_multiple_of(bn)
        || !k.is_multiple_of(bk)
        || !k.is_multiple_of(256)
    {
        return Err(format!(
            "oxide {}: needs m%{bm}, n%{bn}, k%{} == 0 (m={m} n={n} k={k})",
            g.label(),
            bk.max(256)
        ));
    }
    // cutile passes extents and strides as i32; keep every element offset
    // inside that range too.
    if m.saturating_mul(n) > i32::MAX as usize || m.max(n).saturating_mul(k / 2) > i32::MAX as usize
    {
        return Err(format!("oxide {}: extents overflow i32", g.label()));
    }
    Ok(())
}

/// `z[m, n] = alpha * x[m, k] @ y[n, k]ᵀ` through an oxide cubin. `z` must
/// hold `m * n` elements of the variant's output type.
///
/// # Safety
/// All pointers are live device allocations of the implied sizes, 16-byte
/// aligned, and [`oxide_fits`] holds.
#[allow(clippy::too_many_arguments)]
pub unsafe fn oxide_gemm(
    g: &OxideGemm,
    z: CUdeviceptr,
    x: CUdeviceptr,
    y: CUdeviceptr,
    x_scales: CUdeviceptr,
    y_scales: CUdeviceptr,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) -> Result<()> {
    oxide_fits(g, m, n, k).map_err(err)?;
    let d = dev()?;
    let (mi, ni, kp, ks) = (m as i32, n as i32, (k / 2) as i32, (k / 16) as i32);
    let (bm, bn) = (g.bm as i32, g.bn as i32);
    let one = 1i32;
    let mut b = d.stream.launch_builder(&g.func);
    // z: ptr, dims, strides, partition dims, partition strides.
    b.arg(&z).arg(&mi).arg(&ni).arg(&ni).arg(&one);
    b.arg(&bm).arg(&bn).arg(&ni).arg(&one);
    // x, y: packed [rows, k/2].
    b.arg(&x).arg(&mi).arg(&kp).arg(&kp).arg(&one);
    b.arg(&y).arg(&ni).arg(&kp).arg(&kp).arg(&one);
    // scales [rows, k/16].
    b.arg(&x_scales).arg(&mi).arg(&ks).arg(&ks).arg(&one);
    b.arg(&y_scales).arg(&ni).arg(&ks).arg(&ks).arg(&one);
    b.arg(&alpha);
    super::stats::record_launch();
    b.launch(LaunchConfig {
        grid_dim: ((m / g.bm as usize) as u32, (n / g.bn as usize) as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    })
    .map_err(err)?;
    Ok(())
}

// ---- cuBLASLt block-scaled FP4 --------------------------------------------------

/// One cuBLASLt NVFP4 problem: descriptors bound to a pair of swizzled scale
/// buffers, plus the heuristic's candidate algorithms.
pub struct LtNvfp4Plan {
    desc: Desc,
    la: Layout,
    lb: Layout,
    ld: Layout,
    algos: Vec<lt::cublasLtMatmulHeuristicResult_t>,
    alpha: f32,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl LtNvfp4Plan {
    /// `C[m, n] (bf16) = alpha * x[m, k] @ w[n, k]ᵀ`.
    ///
    /// In cuBLAS's column-major terms this is the TN problem
    /// `D[n, m] = W[k, n]ᵀ · X[k, m]` — the only layout FP4 supports — with
    /// the weight as A and the activations as B, which is how both are
    /// already stored.
    ///
    /// # Safety
    /// `w_scales` / `x_scales` are swizzled scale buffers
    /// ([`swizzle_scales_device`]) that outlive the plan.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        ltc: &LtContext,
        m: usize,
        n: usize,
        k: usize,
        w_scales: CUdeviceptr,
        x_scales: CUdeviceptr,
        alpha: f32,
        max_algos: usize,
    ) -> Result<Self> {
        if !m.is_multiple_of(16) || !n.is_multiple_of(16) || !k.is_multiple_of(32) {
            return Err(err(format!(
                "cuBLASLt NVFP4 needs m, n % 16 and k % 32 (m={m} n={n} k={k})"
            )));
        }
        let fp4 = lt::cudaDataType_t::CUDA_R_4F_E2M1;
        let bf16 = lt::cudaDataType_t::CUDA_R_16BF;
        let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        check(
            lt::cublasLtMatmulDescCreate(
                &mut desc,
                lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt::cudaDataType_t::CUDA_R_32F,
            ),
            "cublasLtMatmulDescCreate",
        )?;
        let desc = Desc(desc);
        use cudarc::cublas::sys::cublasOperation_t;
        let (op_t, op_n) = (
            cublasOperation_t::CUBLAS_OP_T as i32,
            cublasOperation_t::CUBLAS_OP_N as i32,
        );
        type A = lt::cublasLtMatmulDescAttributes_t;
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_TRANSA, &op_t)?;
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_TRANSB, &op_n)?;
        let vec16 =
            lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 as u32;
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16)?;
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16)?;
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scales)?;
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &x_scales)?;

        let layout = |ty, rows: usize, cols: usize, ld: usize| -> Result<Layout> {
            let mut l: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
            check(
                lt::cublasLtMatrixLayoutCreate(&mut l, ty, rows as u64, cols as u64, ld as i64),
                "cublasLtMatrixLayoutCreate",
            )?;
            Ok(Layout(l))
        };
        let la = layout(fp4, k, n, k)?;
        let lb = layout(fp4, k, m, k)?;
        let ld = layout(bf16, n, m, n)?;

        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        check(lt::cublasLtMatmulPreferenceCreate(&mut pref), "preference")?;
        let pref = Pref(pref);
        let ws = ltc.workspace_bytes;
        check(
            lt::cublasLtMatmulPreferenceSetAttribute(
                pref.0,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws as *const usize).cast(),
                std::mem::size_of::<usize>(),
            ),
            "preference workspace",
        )?;
        let want = max_algos.max(1);
        let mut results: Vec<lt::cublasLtMatmulHeuristicResult_t> = vec![std::mem::zeroed(); want];
        let mut found: i32 = 0;
        check(
            lt::cublasLtMatmulAlgoGetHeuristic(
                ltc.handle,
                desc.0,
                la.0,
                lb.0,
                ld.0,
                ld.0,
                pref.0,
                want as i32,
                results.as_mut_ptr(),
                &mut found,
            ),
            "cublasLtMatmulAlgoGetHeuristic",
        )?;
        results.truncate(found.max(0) as usize);
        if results.is_empty() {
            return Err(err(format!(
                "cuBLASLt has no NVFP4 (VEC16_UE4M3) algorithm for m={m} n={n} k={k}"
            )));
        }
        Ok(Self {
            desc,
            la,
            lb,
            ld,
            algos: results,
            alpha,
            m,
            n,
            k,
        })
    }

    pub fn algo_count(&self) -> usize {
        self.algos.len()
    }

    /// Run with heuristic candidate `algo`.
    ///
    /// # Safety
    /// `w` is packed `[n, k/2]`, `x` packed `[m, k/2]`, `c` holds `m * n` bf16.
    pub unsafe fn run(
        &self,
        ltc: &LtContext,
        algo: usize,
        w: CUdeviceptr,
        x: CUdeviceptr,
        c: CUdeviceptr,
    ) -> Result<()> {
        let d = dev()?;
        let a = self
            .algos
            .get(algo)
            .ok_or_else(|| err(format!("cuBLASLt NVFP4: no algorithm #{algo}")))?;
        let beta = 0.0f32;
        let (ws_ptr, _g) = ltc.workspace.device_ptr(&d.stream);
        check(
            lt::cublasLtMatmul(
                ltc.handle,
                self.desc.0,
                (&self.alpha as *const f32).cast(),
                w as *const _,
                self.la.0,
                x as *const _,
                self.lb.0,
                (&beta as *const f32).cast(),
                c as *const _,
                self.ld.0,
                c as *mut _,
                self.ld.0,
                &a.algo,
                ws_ptr as *mut _,
                ltc.workspace_bytes,
                d.stream.cu_stream() as *mut _,
            ),
            "cublasLtMatmul (NVFP4)",
        )
    }
}
