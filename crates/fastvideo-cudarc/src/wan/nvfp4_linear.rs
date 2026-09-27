//! NVFP4 linear on the model path: sol-engine `transforms/nvfp4_ffn.py`
//! (TransformerEngine NVFP4 GEMM on the LTX video FFN, RHT and stochastic
//! rounding off, rows padded to 16).
//!
//! The weight is quantized once at load and the bf16 activation on every
//! call, both with the TE `NVFP4BlockScaling` rule (`static_6`: one f32
//! tensor amax, E4M3 scales per 16 elements, bit-identical to
//! `fastvideo_models::nvfp4::quantize`) straight into cuBLASLt's operand
//! layout: packed E2M1 rows plus scales already in the `VEC16_UE4M3` 128x4
//! tiled layout ("to_blocked"), so no swizzle pass runs. The quantizer can
//! read `gelu_tanh(x)` instead of `x` (the FFN's GELU is fused into the down
//! projection's quantizer). `alpha = decode(x) · decode(w)` is computed on
//! the device and read by cuBLASLt through a device pointer, so a call never
//! synchronizes. bf16 output, bias in the epilogue.
//!
//! The GEMM is `D[n, m] = W[k, n]ᵀ · X[k, m]` in cuBLAS's column-major terms
//! (the TN form FP4 requires): row-major `y[m, n] = x[m, k] · w[n, k]ᵀ`.

use cudarc::cublaslt::sys as lt;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig};
use fastvideo_models::nvfp4::ScaleRule;

use super::device::{self, DeviceContext};
use super::fp8::{self, check, set_attr, Desc, Layout, LtContext, Pref};
use super::kernels::launch;
use super::tensor::{Result, TensorError};
// `launch!` names `super::stats` from its call site.
use super::stats;

fn err(e: impl std::fmt::Display) -> TensorError {
    TensorError::Message(e.to_string())
}

fn dev() -> Result<std::sync::Arc<DeviceContext>> {
    device::global_device().ok_or_else(|| err("no global CUDA device context"))
}

/// An NVFP4 GEMM operand in cuBLASLt's layout.
pub struct Nvfp4Operand {
    /// `[rows_pack, k/2]`, low nibble first; rows past `rows` are zero.
    pub packed: CudaSlice<u8>,
    /// Swizzled `VEC16_UE4M3` scales for `rows` padded to 128 (pad zero).
    pub scales: CudaSlice<u8>,
    /// One-element tensor amax.
    pub amax: CudaSlice<f32>,
    pub rows: usize,
    /// `rows` rounded up to 16 (sol-engine `pad_m`).
    pub rows_pack: usize,
    pub k: usize,
}

impl Nvfp4Operand {
    pub fn bytes(&self) -> usize {
        self.packed.len() + self.scales.len() + 4
    }
}

fn rule_code(rule: ScaleRule) -> Result<i32> {
    match rule {
        ScaleRule::Static6 => Ok(0),
        ScaleRule::Static4 => Ok(1),
        ScaleRule::Mse => Err(err(
            "nvfp4 GEMM: the FourOverSix MSE rule has no cuBLASLt form (use static_6)",
        )),
    }
}

/// Whether the device has NVFP4 tensor cores (sm_100 / sm_120 and later).
pub fn tensor_cores() -> bool {
    device::global_device().is_some_and(|d| d.sm_major >= 10)
}

/// Quantize bf16 `x[rows, k]` (device pointer; `bf16(gelu_tanh(x))` when
/// `gelu`) into an [`Nvfp4Operand`]: an amax pass, then one pass that writes
/// codes and swizzled scales. `k % 64 == 0`.
pub fn quantize_bf16_operand(
    x_bf16: CUdeviceptr,
    rows: usize,
    k: usize,
    gelu: bool,
    rule: ScaleRule,
) -> Result<Nvfp4Operand> {
    let d = dev()?;
    if k == 0 || !k.is_multiple_of(64) || rows == 0 {
        return Err(err(format!("nvfp4 operand: [{rows}, {k}] (k % 64 == 0)")));
    }
    let code = rule_code(rule)?;
    let n = rows * k;
    let mut amax = d.stream.alloc_zeros::<f32>(1).map_err(err)?;
    let g = i32::from(gelu);
    let nn = n as i64;
    let cfg_amax = LaunchConfig {
        grid_dim: ((n / 8).div_ceil(256).clamp(1, 4096) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    launch!(d.stream, &d.kernels.nvfp4_amax_bf16, cfg_amax; &x_bf16, &g, &mut amax, &nn)
        .map_err(err)?;
    let rows_pack = rows.next_multiple_of(16);
    let rows_scale = rows.next_multiple_of(128);
    let sc = k / 16;
    let mut packed = unsafe { d.stream.alloc::<u8>(rows_pack * k / 2) }.map_err(err)?;
    let mut scales = unsafe { d.stream.alloc::<u8>(rows_scale * sc) }.map_err(err)?;
    let threads = rows_scale * sc;
    let cfg = LaunchConfig {
        grid_dim: (threads.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r, rp, rs, kk) = (rows as i64, rows_pack as i64, rows_scale as i64, k as i64);
    launch!(
        d.stream, &d.kernels.nvfp4_quant_bf16_sw, cfg;
        &x_bf16, &g, &amax, &mut packed, &mut scales, &r, &rp, &rs, &kk, &code
    )
    .map_err(err)?;
    Ok(Nvfp4Operand {
        packed,
        scales,
        amax,
        rows,
        rows_pack,
        k,
    })
}

/// A cached cuBLASLt problem: descriptors and the chosen algorithm. The
/// scale and bias pointers are rebound on the descriptor per call.
struct LtLinearPlan {
    desc: Desc,
    la: Layout,
    lb: Layout,
    ld: Layout,
    algo: lt::cublasLtMatmulAlgo_t,
    /// alpha / beta read from device memory (else host values, one sync).
    device_alpha: bool,
    /// Bias in the epilogue (else the caller adds it).
    bias: bool,
    /// Columns the GEMM runs: `m`, or `m` padded to 16 when cuBLASLt has no
    /// algorithm for the unpadded count.
    run_m: usize,
}
// Raw cuBLASLt handles, only used under the cache's lock.
unsafe impl Send for LtLinearPlan {}

type PlanKey = (usize, usize, usize, bool);
type PlanMap = std::collections::HashMap<PlanKey, LtLinearPlan>;
static LT_PLANS: std::sync::Mutex<Option<PlanMap>> = std::sync::Mutex::new(None);

/// Pointers one call binds on the descriptor.
struct LtPtrs {
    w_scales: CUdeviceptr,
    x_scales: CUdeviceptr,
    bias: CUdeviceptr,
}

/// # Safety
/// The pointers are live device allocations.
unsafe fn bind(desc: &Desc, bias: bool, p: &LtPtrs) -> Result<()> {
    type A = lt::cublasLtMatmulDescAttributes_t;
    set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p.w_scales)?;
    set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p.x_scales)?;
    if bias {
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &p.bias)?;
    }
    Ok(())
}

/// A plan for `D[n, run_m] = W[k, n]ᵀ · X[k, run_m]`; `None` when the
/// heuristic has no algorithm for this combination.
///
/// # Safety
/// The pointers are live device allocations.
unsafe fn build_plan(
    ltc: &LtContext,
    run_m: usize,
    n: usize,
    k: usize,
    device_alpha: bool,
    bias: bool,
    p: &LtPtrs,
) -> Result<Option<LtLinearPlan>> {
    let fp4 = lt::cudaDataType_t::CUDA_R_4F_E2M1;
    let bf16t = lt::cudaDataType_t::CUDA_R_16BF;
    let mut raw: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
    check(
        lt::cublasLtMatmulDescCreate(
            &mut raw,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType_t::CUDA_R_32F,
        ),
        "cublasLtMatmulDescCreate",
    )?;
    let desc = Desc(raw);
    use cudarc::cublas::sys::cublasOperation_t;
    type A = lt::cublasLtMatmulDescAttributes_t;
    let (op_t, op_n) = (
        cublasOperation_t::CUBLAS_OP_T as i32,
        cublasOperation_t::CUBLAS_OP_N as i32,
    );
    set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_TRANSA, &op_t)?;
    set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_TRANSB, &op_n)?;
    let vec16 = lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 as u32;
    set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16)?;
    set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16)?;
    let mode = if device_alpha {
        lt::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE as i32
    } else {
        lt::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST as i32
    };
    set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode)?;
    if bias {
        let epi = lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS as u32;
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_EPILOGUE, &epi)?;
        set_attr(desc.0, A::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &(bf16t as i32))?;
    }
    bind(&desc, bias, p)?;
    let layout = |ty, rows: usize, cols: usize, ld: usize| -> Result<Layout> {
        let mut l: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check(
            lt::cublasLtMatrixLayoutCreate(&mut l, ty, rows as u64, cols as u64, ld as i64),
            "cublasLtMatrixLayoutCreate",
        )?;
        Ok(Layout(l))
    };
    let la = layout(fp4, k, n, k)?;
    let lb = layout(fp4, k, run_m, k)?;
    let ld = layout(bf16t, n, run_m, n)?;
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
    let mut result: lt::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
    let mut found: i32 = 0;
    let st = lt::cublasLtMatmulAlgoGetHeuristic(
        ltc.handle, desc.0, la.0, lb.0, ld.0, ld.0, pref.0, 1, &mut result, &mut found,
    );
    if st != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || found == 0 {
        return Ok(None);
    }
    Ok(Some(LtLinearPlan {
        desc,
        la,
        lb,
        ld,
        algo: result.algo,
        device_alpha,
        bias,
        run_m,
    }))
}

/// A linear whose GEMM runs on NVFP4 tensor cores: `y = x · Wᵀ + b`, bf16 in
/// and out, `W` quantized once ([`Self::from_bf16`]), `x` on every call.
pub struct Nvfp4Linear {
    w: Nvfp4Operand,
    bias: Option<std::sync::Arc<CudaSlice<half::bf16>>>,
    pub n: usize,
    pub k: usize,
    pub rule: ScaleRule,
}

impl Nvfp4Linear {
    /// Quantize a device bf16 `[n, k]` weight (pointer) with `rule`. `bias`
    /// is bf16 `[n]`.
    pub fn from_bf16(
        w_bf16: CUdeviceptr,
        n: usize,
        k: usize,
        bias: Option<std::sync::Arc<CudaSlice<half::bf16>>>,
        rule: ScaleRule,
    ) -> Result<Self> {
        if !n.is_multiple_of(16) {
            return Err(err(format!("nvfp4 linear: out features {n} % 16 != 0")));
        }
        if bias.as_ref().is_some_and(|b| b.len() != n) {
            return Err(err("nvfp4 linear: bias length"));
        }
        let w = quantize_bf16_operand(w_bf16, n, k, false, rule)?;
        Ok(Self {
            w,
            bias,
            n,
            k,
            rule,
        })
    }

    /// Device bytes held (codes, scales, bias).
    pub fn held_bytes(&self) -> usize {
        self.w.bytes() + self.bias.as_ref().map_or(0, |b| b.len() * 2)
    }

    /// The quantized weight (parity checks).
    pub fn weight(&self) -> &Nvfp4Operand {
        &self.w
    }

    /// The bias (bf16 `[n]`).
    pub fn bias(&self) -> Option<&CudaSlice<half::bf16>> {
        self.bias.as_deref()
    }

    /// `x[m, k]` bf16 (device pointer; `gelu_tanh(x)` when `gelu_in`) → bf16
    /// `[m, n]`, and whether the bias is in it (else the caller adds
    /// [`Self::bias`]).
    pub fn forward_bf16(
        &self,
        x_bf16: CUdeviceptr,
        m: usize,
        gelu_in: bool,
    ) -> Result<(CudaSlice<half::bf16>, bool)> {
        let a = quantize_bf16_operand(x_bf16, m, self.k, gelu_in, self.rule)?;
        self.gemm(&a)
    }

    /// The GEMM on an already-quantized activation.
    #[allow(clippy::map_entry)]
    pub fn gemm(&self, a: &Nvfp4Operand) -> Result<(CudaSlice<half::bf16>, bool)> {
        let d = dev()?;
        if a.k != self.k {
            return Err(err(format!(
                "nvfp4 linear: activation k {} != {}",
                a.k, self.k
            )));
        }
        let (m, n, k) = (a.rows, self.n, self.k);
        let mut ab = d.stream.alloc_zeros::<f32>(2).map_err(err)?;
        let e2 = self.rule.e2m1_max();
        let one = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(d.stream, &d.kernels.nvfp4_alpha, one; &a.amax, &self.w.amax, &e2, &mut ab)
            .map_err(err)?;
        let ltc = fp8::lt_context(&d)?;
        let (wsp, _g1) = self.w.scales.device_ptr(&d.stream);
        let (xsp, _g2) = a.scales.device_ptr(&d.stream);
        let (wp, _g3) = self.w.packed.device_ptr(&d.stream);
        let (xp, _g4) = a.packed.device_ptr(&d.stream);
        let bias_ptr = match &self.bias {
            Some(b) => b.device_ptr(&d.stream).0,
            None => 0,
        };
        let ptrs = LtPtrs {
            w_scales: wsp,
            x_scales: xsp,
            bias: bias_ptr,
        };
        let has_bias = self.bias.is_some();
        let mut guard = LT_PLANS
            .lock()
            .map_err(|_| err("nvfp4 plan cache poisoned"))?;
        let plans = guard.get_or_insert_with(Default::default);
        let key = (m, n, k, has_bias);
        if !plans.contains_key(&key) {
            let mut built = None;
            let biases: &[bool] = if has_bias { &[true, false] } else { &[false] };
            'search: for run_m in [m, a.rows_pack] {
                for device_alpha in [true, false] {
                    for &bias in biases {
                        let p = unsafe { build_plan(&ltc, run_m, n, k, device_alpha, bias, &ptrs)? };
                        if let Some(p) = p {
                            built = Some(p);
                            break 'search;
                        }
                    }
                }
            }
            let plan = built.ok_or_else(|| {
                err(format!(
                    "cuBLASLt has no NVFP4 (VEC16_UE4M3) algorithm for m={m} n={n} k={k} on sm_{}{}",
                    d.sm_major, d.sm_minor
                ))
            })?;
            if plan.run_m != m || !plan.device_alpha || plan.bias != has_bias {
                super::log::info(format_args!(
                    "nvfp4 linear m={m} n={n} k={k}: runs {} rows, device alpha {}, bias epilogue {}",
                    plan.run_m, plan.device_alpha, plan.bias
                ));
            }
            plans.insert(key, plan);
        }
        let plan = &plans[&key];
        unsafe { bind(&plan.desc, plan.bias, &ptrs)? };
        let run_m = plan.run_m;
        let mut c = unsafe { d.stream.alloc::<half::bf16>(run_m * n) }.map_err(err)?;
        let host_ab: Vec<f32>;
        let (ab_ptr, _g5) = ab.device_ptr(&d.stream);
        let (alpha_p, beta_p): (*const std::ffi::c_void, *const std::ffi::c_void) =
            if plan.device_alpha {
                (ab_ptr as *const _, (ab_ptr + 4) as *const _)
            } else {
                host_ab = d.stream.memcpy_dtov(&ab).map_err(err)?;
                (
                    (&host_ab[0] as *const f32).cast(),
                    (&host_ab[1] as *const f32).cast(),
                )
            };
        {
            let (cp, _g6) = c.device_ptr_mut(&d.stream);
            let (ws_ptr, _g7) = ltc.workspace.device_ptr(&d.stream);
            stats::record_launch();
            unsafe {
                check(
                    lt::cublasLtMatmul(
                        ltc.handle,
                        plan.desc.0,
                        alpha_p,
                        wp as *const _,
                        plan.la.0,
                        xp as *const _,
                        plan.lb.0,
                        beta_p,
                        cp as *const _,
                        plan.ld.0,
                        cp as *mut _,
                        plan.ld.0,
                        &plan.algo,
                        ws_ptr as *mut _,
                        ltc.workspace_bytes,
                        d.stream.cu_stream() as *mut _,
                    ),
                    "cublasLtMatmul (NVFP4 linear)",
                )?;
            }
        }
        let bias_in = plan.bias;
        drop(guard);
        if run_m != m {
            let mut out = unsafe { d.stream.alloc::<half::bf16>(m * n) }.map_err(err)?;
            let src = c.slice(0..m * n);
            d.stream.memcpy_dtod(&src, &mut out).map_err(err)?;
            return Ok((out, bias_in));
        }
        Ok((c, bias_in))
    }
}
