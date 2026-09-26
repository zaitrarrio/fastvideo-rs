//! Dense SDPA through cuDNN's fused attention engine (backend API).
//!
//! The operation graph is cudnn-frontend's SDPA forward for inference
//! (`node/scaled_dot_product_flash_attention.h` + `node/softmax.h`), built
//! by hand because cudarc exposes the backend API but not the frontend:
//!
//! ```text
//! S  = matmul(Q, K^T)            f32, virtual        ("bmm1")
//! S' = S * scale                 by-value scalar     ("attn_scale")
//! M  = reduce_max(S', axis=-1)   [.., sq, 1]         (softmax "Max")
//! E  = exp(S' - M)                                   ("sub", "exp")
//! Z  = reduce_add(E, axis=-1)                        ("sum")
//! P  = E / Z                                         ("div")
//! O  = matmul(P, V)              bf16                ("bmm2")
//! ```
//!
//! Q/K/V/O are bf16 BHSD `[1, bh, s, d]`; the plan is built once per shape
//! and cached, and a shape cuDNN offers no engine for falls back to
//! `flash_mma_fwd2`.
//!
//! Status (RTX PRO 6000, cuDNN 9.26.0, 2026-09-26): cuDNN rejects this
//! composite softmax graph on every shape, per its own log
//! (`CUDNN_LOGLEVEL_DBG=2`): "non-flash composite MHA fprop is no longer
//! supported (removed with the xmma512 engine) at: !is_flash_fprop". From
//! cuDNN 9.21 cudnn-frontend lowers softmax to the single unified
//! `OPERATION_SOFTMAX` backend node (`node/softmax.h:386-390`), which the
//! sm_120 flash engines match; cudarc 0.17.8's cuDNN bindings predate that
//! descriptor. So `FASTVIDEO_FLASH_KERNEL=cudnn` currently always runs
//! `flash_mma_fwd2`. Measured through torch 2.13 / cuDNN on the same GPU,
//! cuDNN's fused SDPA is 5-6% faster than `flash_mma_fwd2` at the H3 / LTX
//! shapes (392-397 vs 371-373 TFLOPS): the next step is the unified node.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cudnn::sys::{
    self, cudnnBackendAttributeName_t as A, cudnnBackendAttributeType_t as T,
    cudnnBackendDescriptorType_t as DT, cudnnBackendDescriptor_t, cudnnDataType_t,
    cudnnStatus_t,
};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn ok(st: cudnnStatus_t, what: &str) -> Result<()> {
    if st == cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(msg(format!("cudnn sdpa: {what}: {st:?}")))
    }
}

/// An owned backend descriptor.
struct Desc(cudnnBackendDescriptor_t);

// The descriptors are only touched under the plan cache's mutex or while
// executing on the one stream they were built for.
unsafe impl Send for Desc {}
unsafe impl Sync for Desc {}

impl Drop for Desc {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnBackendDestroyDescriptor(self.0);
        }
    }
}

impl Desc {
    fn new(ty: DT) -> Result<Self> {
        let mut d: cudnnBackendDescriptor_t = std::ptr::null_mut();
        ok(unsafe { sys::cudnnBackendCreateDescriptor(ty, &mut d) }, "create")?;
        Ok(Self(d))
    }
    fn set<V>(&self, name: A, ty: T, vals: &[V]) -> Result<()> {
        ok(
            unsafe {
                sys::cudnnBackendSetAttribute(
                    self.0,
                    name,
                    ty,
                    vals.len() as i64,
                    vals.as_ptr() as *const c_void,
                )
            },
            &format!("set {name:?}"),
        )
    }
    fn set_desc(&self, name: A, d: &[&Desc]) -> Result<()> {
        let ptrs: Vec<cudnnBackendDescriptor_t> = d.iter().map(|x| x.0).collect();
        self.set(name, T::CUDNN_TYPE_BACKEND_DESCRIPTOR, &ptrs)
    }
    fn finalize(self, what: &str) -> Result<Self> {
        ok(unsafe { sys::cudnnBackendFinalize(self.0) }, what)?;
        Ok(self)
    }
}

struct Handle(sys::cudnnHandle_t);
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

/// One cuDNN handle bound to the device stream, for the backend API.
fn handle(dev: &super::device::DeviceContext) -> Result<sys::cudnnHandle_t> {
    static H: OnceLock<std::result::Result<Handle, String>> = OnceLock::new();
    let h = H.get_or_init(|| unsafe {
        let mut h: sys::cudnnHandle_t = std::ptr::null_mut();
        let st = sys::cudnnCreate(&mut h);
        if st != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(format!("cudnnCreate: {st:?}"));
        }
        let st = sys::cudnnSetStream(h, dev.stream.cu_stream() as sys::cudaStream_t);
        if st != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(format!("cudnnSetStream: {st:?}"));
        }
        Ok(Handle(h))
    });
    match h {
        Ok(h) => Ok(h.0),
        Err(e) => Err(msg(e.clone())),
    }
}

const UID_Q: i64 = 1;
const UID_K: i64 = 2;
const UID_V: i64 = 3;
const UID_O: i64 = 4;
const UID_SCALE: i64 = 5;

fn tensor(
    uid: i64,
    dtype: cudnnDataType_t,
    dims: [i64; 4],
    strides: [i64; 4],
    virt: bool,
    by_value: bool,
) -> Result<Desc> {
    let t = Desc::new(DT::CUDNN_BACKEND_TENSOR_DESCRIPTOR)?;
    t.set(A::CUDNN_ATTR_TENSOR_DATA_TYPE, T::CUDNN_TYPE_DATA_TYPE, &[dtype])?;
    t.set(A::CUDNN_ATTR_TENSOR_DIMENSIONS, T::CUDNN_TYPE_INT64, &dims)?;
    t.set(A::CUDNN_ATTR_TENSOR_STRIDES, T::CUDNN_TYPE_INT64, &strides)?;
    t.set(A::CUDNN_ATTR_TENSOR_UNIQUE_ID, T::CUDNN_TYPE_INT64, &[uid])?;
    t.set(A::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, T::CUDNN_TYPE_INT64, &[16i64])?;
    t.set(A::CUDNN_ATTR_TENSOR_IS_VIRTUAL, T::CUDNN_TYPE_BOOLEAN, &[virt])?;
    if by_value {
        t.set(A::CUDNN_ATTR_TENSOR_IS_BY_VALUE, T::CUDNN_TYPE_BOOLEAN, &[true])?;
    }
    t.finalize("tensor")
}

fn pointwise_op(
    mode: sys::cudnnPointwiseMode_t,
    x: &Desc,
    b: Option<&Desc>,
    y: &Desc,
) -> Result<(Desc, Desc)> {
    let pw = Desc::new(DT::CUDNN_BACKEND_POINTWISE_DESCRIPTOR)?;
    pw.set(A::CUDNN_ATTR_POINTWISE_MODE, T::CUDNN_TYPE_POINTWISE_MODE, &[mode])?;
    pw.set(
        A::CUDNN_ATTR_POINTWISE_MATH_PREC,
        T::CUDNN_TYPE_DATA_TYPE,
        &[cudnnDataType_t::CUDNN_DATA_FLOAT],
    )?;
    let pw = pw.finalize("pointwise desc")?;
    let op = Desc::new(DT::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, &[&pw])?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_POINTWISE_XDESC, &[x])?;
    if let Some(b) = b {
        op.set_desc(A::CUDNN_ATTR_OPERATION_POINTWISE_BDESC, &[b])?;
    }
    op.set_desc(A::CUDNN_ATTR_OPERATION_POINTWISE_YDESC, &[y])?;
    Ok((pw, op.finalize("pointwise op")?))
}

fn reduction_op(
    op_ty: sys::cudnnReduceTensorOp_t,
    x: &Desc,
    y: &Desc,
) -> Result<(Desc, Desc)> {
    let r = Desc::new(DT::CUDNN_BACKEND_REDUCTION_DESCRIPTOR)?;
    r.set(
        A::CUDNN_ATTR_REDUCTION_OPERATOR,
        T::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
        &[op_ty],
    )?;
    r.set(
        A::CUDNN_ATTR_REDUCTION_COMP_TYPE,
        T::CUDNN_TYPE_DATA_TYPE,
        &[cudnnDataType_t::CUDNN_DATA_FLOAT],
    )?;
    let r = r.finalize("reduction desc")?;
    let op = Desc::new(DT::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_REDUCTION_DESC, &[&r])?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_REDUCTION_XDESC, &[x])?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_REDUCTION_YDESC, &[y])?;
    Ok((r, op.finalize("reduction op")?))
}

fn matmul_op(a: &Desc, b: &Desc, c: &Desc) -> Result<(Desc, Desc)> {
    let m = Desc::new(DT::CUDNN_BACKEND_MATMUL_DESCRIPTOR)?;
    m.set(
        A::CUDNN_ATTR_MATMUL_COMP_TYPE,
        T::CUDNN_TYPE_DATA_TYPE,
        &[cudnnDataType_t::CUDNN_DATA_FLOAT],
    )?;
    let m = m.finalize("matmul desc")?;
    let op = Desc::new(DT::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_MATMUL_ADESC, &[a])?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_MATMUL_BDESC, &[b])?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_MATMUL_CDESC, &[c])?;
    op.set_desc(A::CUDNN_ATTR_OPERATION_MATMUL_DESC, &[&m])?;
    Ok((m, op.finalize("matmul op")?))
}

/// A finalized execution plan (the descriptors it was built from stay alive
/// with it) and its workspace size.
pub struct SdpaPlan {
    plan: Desc,
    workspace: usize,
    engine: String,
    _keep: Vec<Desc>,
}

fn build(
    dev: &super::device::DeviceContext,
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Result<SdpaPlan> {
    use cudnnDataType_t::{CUDNN_DATA_BFLOAT16 as BF16, CUDNN_DATA_FLOAT as F32};
    let h = handle(dev)?;
    let (bh, sq, sk, d) = (bh as i64, sq as i64, sk as i64, d as i64);
    let q = tensor(UID_Q, BF16, [1, bh, sq, d], [bh * sq * d, sq * d, d, 1], false, false)?;
    // K^T as a [.., d, sk] view of row-major K.
    let kt = tensor(UID_K, BF16, [1, bh, d, sk], [bh * sk * d, sk * d, 1, d], false, false)?;
    let v = tensor(UID_V, BF16, [1, bh, sk, d], [bh * sk * d, sk * d, d, 1], false, false)?;
    let o = tensor(UID_O, BF16, [1, bh, sq, d], [bh * sq * d, sq * d, d, 1], false, false)?;
    let scale = tensor(UID_SCALE, F32, [1, 1, 1, 1], [1, 1, 1, 1], false, true)?;
    let full = |uid| {
        tensor(uid, F32, [1, bh, sq, sk], [bh * sq * sk, sq * sk, sk, 1], true, false)
    };
    let row = |uid| tensor(uid, F32, [1, bh, sq, 1], [bh * sq, sq, 1, 1], true, false);
    let (s, s2, mx, sub, ex, sum, p) = (
        full(100)?,
        full(101)?,
        row(102)?,
        full(103)?,
        full(104)?,
        row(105)?,
        full(106)?,
    );
    use sys::cudnnPointwiseMode_t as PM;
    use sys::cudnnReduceTensorOp_t as RO;
    let (d1, bmm1) = matmul_op(&q, &kt, &s)?;
    let (d2, mul) = pointwise_op(PM::CUDNN_POINTWISE_MUL, &s, Some(&scale), &s2)?;
    let (d3, rmax) = reduction_op(RO::CUDNN_REDUCE_TENSOR_MAX, &s2, &mx)?;
    let (d4, psub) = pointwise_op(PM::CUDNN_POINTWISE_SUB, &s2, Some(&mx), &sub)?;
    let (d5, pexp) = pointwise_op(PM::CUDNN_POINTWISE_EXP, &sub, None, &ex)?;
    let (d6, rsum) = reduction_op(RO::CUDNN_REDUCE_TENSOR_ADD, &ex, &sum)?;
    let (d7, pdiv) = pointwise_op(PM::CUDNN_POINTWISE_DIV, &ex, Some(&sum), &p)?;
    let (d8, bmm2) = matmul_op(&p, &v, &o)?;

    let graph = Desc::new(DT::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)?;
    graph.set_desc(
        A::CUDNN_ATTR_OPERATIONGRAPH_OPS,
        &[&bmm1, &mul, &rmax, &psub, &pexp, &rsum, &pdiv, &bmm2],
    )?;
    graph.set(A::CUDNN_ATTR_OPERATIONGRAPH_HANDLE, T::CUDNN_TYPE_HANDLE, &[h])?;
    let graph = graph.finalize("operation graph")?;

    // Heuristic modes in cudnn-frontend's order: A, then B, then FALLBACK.
    use sys::cudnnBackendHeurMode_t as HM;
    let mut last = String::from("no engine config");
    let mut keep = vec![
        q, kt, v, o, scale, s, s2, mx, sub, ex, sum, p, d1, bmm1, d2, mul, d3, rmax, d4, psub, d5,
        pexp, d6, rsum, d7, pdiv, d8, bmm2,
    ];
    for mode in [HM::CUDNN_HEUR_MODE_A, HM::CUDNN_HEUR_MODE_B, HM::CUDNN_HEUR_MODE_FALLBACK] {
        let heur = Desc::new(DT::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)?;
        heur.set_desc(A::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, &[&graph])?;
        heur.set(A::CUDNN_ATTR_ENGINEHEUR_MODE, T::CUDNN_TYPE_HEUR_MODE, &[mode])?;
        let heur = match heur.finalize("engine heuristics") {
            Ok(h) => h,
            Err(e) => {
                last = format!("{mode:?}: {e}");
                continue;
            }
        };
        const MAX_CFG: usize = 16;
        let cfgs: Vec<Desc> = (0..MAX_CFG)
            .map(|_| Desc::new(DT::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR))
            .collect::<Result<_>>()?;
        let mut raw: Vec<cudnnBackendDescriptor_t> = cfgs.iter().map(|c| c.0).collect();
        let mut count = 0i64;
        if let Err(e) = ok(
            unsafe {
                sys::cudnnBackendGetAttribute(
                    heur.0,
                    A::CUDNN_ATTR_ENGINEHEUR_RESULTS,
                    T::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    MAX_CFG as i64,
                    &mut count,
                    raw.as_mut_ptr() as *mut c_void,
                )
            },
            "heuristic results",
        ) {
            last = format!("{mode:?}: {e}");
            continue;
        }
        if count <= 0 {
            last = format!("{mode:?}: 0 engine configs");
            continue;
        }
        for (i, cfg) in cfgs.iter().take(count as usize).enumerate() {
            let plan = Desc::new(DT::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)?;
            plan.set(A::CUDNN_ATTR_EXECUTION_PLAN_HANDLE, T::CUDNN_TYPE_HANDLE, &[h])?;
            plan.set_desc(A::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, &[cfg])?;
            match plan.finalize("execution plan") {
                Ok(plan) => {
                    let mut ws = 0i64;
                    let mut n = 0i64;
                    ok(
                        unsafe {
                            sys::cudnnBackendGetAttribute(
                                plan.0,
                                A::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                T::CUDNN_TYPE_INT64,
                                1,
                                &mut n,
                                &mut ws as *mut i64 as *mut c_void,
                            )
                        },
                        "workspace size",
                    )?;
                    keep.push(graph);
                    keep.push(heur);
                    return Ok(SdpaPlan {
                        plan,
                        workspace: ws.max(0) as usize,
                        engine: format!("{mode:?} config {i} of {count}"),
                        _keep: keep,
                    });
                }
                Err(e) => last = format!("{mode:?}: {e}"),
            }
        }
    }
    Err(msg(format!("cudnn sdpa: no executable plan ({last})")))
}

type Key = (usize, usize, usize, usize);

fn plans() -> &'static Mutex<HashMap<Key, Option<Arc<SdpaPlan>>>> {
    static P: OnceLock<Mutex<HashMap<Key, Option<Arc<SdpaPlan>>>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The cached plan for a shape, built on first use; `None` (also cached)
/// when cuDNN offers no engine for it.
fn plan_for(
    dev: &super::device::DeviceContext,
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Option<Arc<SdpaPlan>> {
    let key = (bh, sq, sk, d);
    let mut map = plans().lock().expect("cudnn sdpa plans");
    if let Some(p) = map.get(&key) {
        return p.clone();
    }
    let p = match build(dev, bh, sq, sk, d) {
        Ok(p) => {
            super::log::info_once(
                &LOGGED,
                format_args!(
                    "sdpa: cuDNN fused attention (cuDNN {}, {}, workspace {} B)",
                    unsafe { sys::cudnnGetVersion() },
                    p.engine,
                    p.workspace
                ),
            );
            Some(Arc::new(p))
        }
        Err(e) => {
            super::log::info_once(
                &FAILED,
                format_args!("sdpa: cuDNN fused attention unavailable, using flash_mma: {e}"),
            );
            None
        }
    };
    map.insert(key, p.clone());
    p
}

static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `out = softmax(scale * q k^T) v` on bf16 BHSD `[bh, sq, d]` x `[bh, sk, d]`
/// through cuDNN. `Ok(false)` when cuDNN has no engine for the shape (the
/// caller runs its own kernel).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_bf16(
    q: &CudaSlice<half::bf16>,
    k: &CudaSlice<half::bf16>,
    v: &CudaSlice<half::bf16>,
    out: &mut CudaSlice<half::bf16>,
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
    scale: f32,
) -> Result<bool> {
    let dev = super::device::global_device().ok_or_else(|| msg("cudnn sdpa: no device"))?;
    let Some(plan) = plan_for(&dev, bh, sq, sk, d) else {
        return Ok(false);
    };
    let h = handle(&dev)?;
    let ws = if plan.workspace > 0 {
        Some(unsafe { dev.stream.alloc::<u8>(plan.workspace) }.map_err(|e| msg(e.to_string()))?)
    } else {
        None
    };
    let stream = &dev.stream;
    let (qp, _g1) = q.device_ptr(stream);
    let (kp, _g2) = k.device_ptr(stream);
    let (vp, _g3) = v.device_ptr(stream);
    let (op, _g4) = out.device_ptr_mut(stream);
    let wsp = ws.as_ref().map(|w| w.device_ptr(stream));
    let scale_host = scale;
    let uids = [UID_Q, UID_K, UID_V, UID_O, UID_SCALE];
    let ptrs: [*mut c_void; 5] = [
        qp as *mut c_void,
        kp as *mut c_void,
        vp as *mut c_void,
        op as *mut c_void,
        &scale_host as *const f32 as *mut c_void,
    ];
    let pack = Desc::new(DT::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)?;
    pack.set(A::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, T::CUDNN_TYPE_INT64, &uids)?;
    pack.set(A::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, T::CUDNN_TYPE_VOID_PTR, &ptrs)?;
    let wptr: *mut c_void = wsp
        .as_ref()
        .map_or(std::ptr::null_mut(), |(p, _)| *p as *mut c_void);
    pack.set(A::CUDNN_ATTR_VARIANT_PACK_WORKSPACE, T::CUDNN_TYPE_VOID_PTR, &[wptr])?;
    let pack = pack.finalize("variant pack")?;
    ok(
        unsafe { sys::cudnnBackendExecute(h, plan.plan.0, pack.0) },
        "execute",
    )?;
    Ok(true)
}
