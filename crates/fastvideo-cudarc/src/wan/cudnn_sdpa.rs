//! Dense SDPA through cuDNN's fused attention engines (backend API).
//!
//! cudarc 0.17.8 exposes cuDNN's backend API but predates the SDPA
//! descriptors of cuDNN >= 9.13 / 9.21, so the ones this module needs are
//! declared here ([`raw`]) with their cuDNN 9.26 header values
//! (`cudnn_graph_v9.h`), and every backend call goes through cudarc's loaded
//! library with the enum arguments widened to `u32` (the enums are
//! `repr(u32)`, so the ABI is the same) and statuses read as plain `i32`
//! (a newer cuDNN may return status codes cudarc's enum does not list).
//!
//! Three operation graphs (`auto` tries unified, then composite; softmax only by name):
//!
//! * [`SdpaGraph::Unified`]: one `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR`
//!   (Q, K, V, O and a by-value scale), which is what cudnn-frontend's
//!   `UnifiedSDPANode` builds for a plain bf16 inference SDPA on cuDNN >= 9.13.1
//!   (`node/scaled_dot_product_flash_attention.h`, `AttentionImplementation_t::AUTO`
//!   picks UNIFIED first, `graph_properties.h` `_auto_select_implementation`).
//!   K is the literal K (not K^T).
//! * [`SdpaGraph::Softmax`]: the composite graph with cuDNN >= 9.21's single
//!   `CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR` node, as cudnn-frontend's
//!   `CompositeSDPANode` lowers it (`node/softmax.h`, `UnifiedSoftmaxNode`):
//!   `S = Q K^T`, `S' = S * scale`, `P = softmax(S')`, `O = P V`.
//! * [`SdpaGraph::Composite`]: the pre-9.21 composite softmax (reduce max,
//!   sub, exp, reduce sum, div). cuDNN 9.26 rejects it on every shape: "non-flash
//!   composite MHA fprop is no longer supported (removed with the xmma512
//!   engine)". Kept for older runtimes.
//!
//! Q/K/V/O are bf16 BHSD `[1, bh, s, d]`; the plan is built once per shape,
//! graph and engine-config index and cached, and a shape cuDNN offers no
//! engine for falls back to `flash_mma_fwd2` (in `attn.rs`).

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cudnn::sys::{
    self, cudnnBackendAttributeName_t as A, cudnnBackendAttributeType_t as T,
    cudnnBackendDescriptorType_t as DT, cudnnBackendDescriptor_t, cudnnDataType_t,
};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Raw backend FFI: the descriptor types and attribute names cudarc lacks,
/// and the backend entry points with `u32` enums and `i32` statuses.
pub mod raw {
    use super::*;

    /// `cudnnBackendDescriptorType_t` (cuDNN 9.26 `cudnn_graph_v9.h`).
    pub const OPERATION_SDPA_FWD_DESCRIPTOR: u32 = 41; // @since 9.13.0
    pub const OPERATION_SOFTMAX_DESCRIPTOR: u32 = 45; // @since 9.20.0

    /// `cudnnBackendAttributeName_t`.
    pub const ATTR_OPERATION_SDPA_FWD_QDESC: u32 = 2800;
    pub const ATTR_OPERATION_SDPA_FWD_KDESC: u32 = 2801;
    pub const ATTR_OPERATION_SDPA_FWD_VDESC: u32 = 2802;
    pub const ATTR_OPERATION_SDPA_FWD_ODESC: u32 = 2803;
    pub const ATTR_OPERATION_SDPA_FWD_STATSDESC: u32 = 2804;
    pub const ATTR_OPERATION_SDPA_FWD_SCALEDESC: u32 = 2805;
    pub const ATTR_OPERATION_SOFTMAX_XDESC: u32 = 3100;
    pub const ATTR_OPERATION_SOFTMAX_YDESC: u32 = 3101;

    pub const STATUS_SUCCESS: i32 = 0;

    type CreateFn = unsafe extern "C" fn(u32, *mut cudnnBackendDescriptor_t) -> i32;
    type DestroyFn = unsafe extern "C" fn(cudnnBackendDescriptor_t) -> i32;
    type SetFn = unsafe extern "C" fn(cudnnBackendDescriptor_t, u32, u32, i64, *const c_void) -> i32;
    type GetFn =
        unsafe extern "C" fn(cudnnBackendDescriptor_t, u32, u32, i64, *mut i64, *mut c_void) -> i32;
    type FinalizeFn = unsafe extern "C" fn(cudnnBackendDescriptor_t) -> i32;
    type ExecuteFn =
        unsafe extern "C" fn(sys::cudnnHandle_t, cudnnBackendDescriptor_t, cudnnBackendDescriptor_t) -> i32;

    // The transmutes only change enum parameters to their `repr(u32)` integer
    // and the `repr(u32)` status return to `i32`: same size, same ABI.
    pub unsafe fn create(ty: u32, d: *mut cudnnBackendDescriptor_t) -> i32 {
        let f: CreateFn = std::mem::transmute(sys::culib().cudnnBackendCreateDescriptor);
        f(ty, d)
    }
    pub unsafe fn destroy(d: cudnnBackendDescriptor_t) -> i32 {
        let f: DestroyFn = std::mem::transmute(sys::culib().cudnnBackendDestroyDescriptor);
        f(d)
    }
    pub unsafe fn set(d: cudnnBackendDescriptor_t, name: u32, ty: u32, n: i64, v: *const c_void) -> i32 {
        let f: SetFn = std::mem::transmute(sys::culib().cudnnBackendSetAttribute);
        f(d, name, ty, n, v)
    }
    pub unsafe fn get(
        d: cudnnBackendDescriptor_t,
        name: u32,
        ty: u32,
        cap: i64,
        n: *mut i64,
        v: *mut c_void,
    ) -> i32 {
        let f: GetFn = std::mem::transmute(sys::culib().cudnnBackendGetAttribute);
        f(d, name, ty, cap, n, v)
    }
    pub unsafe fn finalize(d: cudnnBackendDescriptor_t) -> i32 {
        let f: FinalizeFn = std::mem::transmute(sys::culib().cudnnBackendFinalize);
        f(d)
    }
    pub unsafe fn execute(
        h: sys::cudnnHandle_t,
        plan: cudnnBackendDescriptor_t,
        pack: cudnnBackendDescriptor_t,
    ) -> i32 {
        let f: ExecuteFn = std::mem::transmute(sys::culib().cudnnBackendExecute);
        f(h, plan, pack)
    }
    /// cuDNN's last error message (thread-local in the library).
    pub fn last_error() -> String {
        let mut buf = vec![0 as std::ffi::c_char; 1024];
        unsafe { (sys::culib().cudnnGetLastErrorString)(buf.as_mut_ptr(), buf.len()) };
        let s = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
        s.to_string_lossy().trim().to_string()
    }
}

fn ok(st: i32, what: &str) -> Result<()> {
    if st == raw::STATUS_SUCCESS {
        Ok(())
    } else {
        let e = raw::last_error();
        Err(msg(format!("cudnn sdpa: {what}: status {st}{}{e}", if e.is_empty() { "" } else { ": " })))
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
            raw::destroy(self.0);
        }
    }
}

impl Desc {
    fn new(ty: DT) -> Result<Self> {
        Self::new_raw(ty as u32)
    }
    fn new_raw(ty: u32) -> Result<Self> {
        let mut d: cudnnBackendDescriptor_t = std::ptr::null_mut();
        ok(unsafe { raw::create(ty, &mut d) }, &format!("create descriptor type {ty}"))?;
        Ok(Self(d))
    }
    fn set_raw<V>(&self, name: u32, ty: T, vals: &[V]) -> Result<()> {
        ok(
            unsafe {
                raw::set(self.0, name, ty as u32, vals.len() as i64, vals.as_ptr() as *const c_void)
            },
            &format!("set attribute {name}"),
        )
    }
    fn set<V>(&self, name: A, ty: T, vals: &[V]) -> Result<()> {
        self.set_raw(name as u32, ty, vals)
    }
    fn set_desc_raw(&self, name: u32, d: &[&Desc]) -> Result<()> {
        let ptrs: Vec<cudnnBackendDescriptor_t> = d.iter().map(|x| x.0).collect();
        self.set_raw(name, T::CUDNN_TYPE_BACKEND_DESCRIPTOR, &ptrs)
    }
    fn set_desc(&self, name: A, d: &[&Desc]) -> Result<()> {
        self.set_desc_raw(name as u32, d)
    }
    fn finalize(self, what: &str) -> Result<Self> {
        ok(unsafe { raw::finalize(self.0) }, what)?;
        Ok(self)
    }
    fn get_i64(&self, name: A) -> Result<i64> {
        let (mut v, mut n) = (0i64, 0i64);
        ok(
            unsafe {
                raw::get(self.0, name as u32, T::CUDNN_TYPE_INT64 as u32, 1, &mut n, &mut v as *mut i64 as *mut c_void)
            },
            &format!("get {name:?}"),
        )?;
        Ok(v)
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
        if st != sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(format!("cudnnCreate: {st:?}"));
        }
        let st = sys::cudnnSetStream(h, dev.stream.cu_stream() as sys::cudaStream_t);
        if st != sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(format!("cudnnSetStream: {st:?}"));
        }
        Ok(Handle(h))
    });
    match h {
        Ok(h) => {
            // The handle is shared; the stream is the caller's device's (a
            // thread may run on another stream, e.g. a CUDA-graph capture).
            let st = unsafe { sys::cudnnSetStream(h.0, dev.stream.cu_stream() as sys::cudaStream_t) };
            if st != sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(msg(format!("cudnnSetStream: {st:?}")));
            }
            Ok(h.0)
        }
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

/// Which operation graph describes the attention to cuDNN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdpaGraph {
    /// One `OPERATION_SDPA_FWD` node (cuDNN >= 9.13.1).
    Unified,
    /// bmm, scale, one `OPERATION_SOFTMAX` node, bmm (cuDNN >= 9.21).
    Softmax,
    /// bmm, scale, max / sub / exp / sum / div, bmm (pre-9.21 composite).
    Composite,
}

impl SdpaGraph {
    pub const ALL: [SdpaGraph; 3] = [SdpaGraph::Unified, SdpaGraph::Softmax, SdpaGraph::Composite];
    pub fn name(self) -> &'static str {
        match self {
            SdpaGraph::Unified => "unified",
            SdpaGraph::Softmax => "softmax",
            SdpaGraph::Composite => "composite",
        }
    }
}

/// `FASTVIDEO_CUDNN_SDPA_GRAPH=auto|unified|softmax|composite`: the graphs to
/// try, in order (`auto`: unified, then composite).
fn graphs_requested() -> Vec<SdpaGraph> {
    match super::envflag::string_flag("FASTVIDEO_CUDNN_SDPA_GRAPH", "auto").as_str() {
        "unified" => vec![SdpaGraph::Unified],
        "softmax" => vec![SdpaGraph::Softmax],
        "composite" => vec![SdpaGraph::Composite],
        // Not Softmax: on cuDNN 9.26 / sm_120 it builds a plan whose output is
        // inf / NaN (attn3_parity), so it runs only when asked for by name.
        _ => vec![SdpaGraph::Unified, SdpaGraph::Composite],
    }
}

/// The finalized operation graph and the descriptors it references.
struct Graph {
    graph: Desc,
    _keep: Vec<Desc>,
}

fn build_graph(h: sys::cudnnHandle_t, kind: SdpaGraph, bh: i64, sq: i64, sk: i64, d: i64) -> Result<Graph> {
    use cudnnDataType_t::{CUDNN_DATA_BFLOAT16 as BF16, CUDNN_DATA_FLOAT as F32};
    use sys::cudnnPointwiseMode_t as PM;
    use sys::cudnnReduceTensorOp_t as RO;
    let q = tensor(UID_Q, BF16, [1, bh, sq, d], [bh * sq * d, sq * d, d, 1], false, false)?;
    let v = tensor(UID_V, BF16, [1, bh, sk, d], [bh * sk * d, sk * d, d, 1], false, false)?;
    let o = tensor(UID_O, BF16, [1, bh, sq, d], [bh * sq * d, sq * d, d, 1], false, false)?;
    let scale = tensor(UID_SCALE, F32, [1, 1, 1, 1], [1, 1, 1, 1], false, true)?;
    let full = |uid| tensor(uid, F32, [1, bh, sq, sk], [bh * sq * sk, sq * sk, sk, 1], true, false);
    let row = |uid| tensor(uid, F32, [1, bh, sq, 1], [bh * sq, sq, 1, 1], true, false);
    let (ops, mut keep): (Vec<Desc>, Vec<Desc>) = match kind {
        SdpaGraph::Unified => {
            // The literal K, [.., sk, d].
            let k = tensor(UID_K, BF16, [1, bh, sk, d], [bh * sk * d, sk * d, d, 1], false, false)?;
            let op = Desc::new_raw(raw::OPERATION_SDPA_FWD_DESCRIPTOR)?;
            op.set_desc_raw(raw::ATTR_OPERATION_SDPA_FWD_QDESC, &[&q])?;
            op.set_desc_raw(raw::ATTR_OPERATION_SDPA_FWD_KDESC, &[&k])?;
            op.set_desc_raw(raw::ATTR_OPERATION_SDPA_FWD_VDESC, &[&v])?;
            op.set_desc_raw(raw::ATTR_OPERATION_SDPA_FWD_ODESC, &[&o])?;
            op.set_desc_raw(raw::ATTR_OPERATION_SDPA_FWD_SCALEDESC, &[&scale])?;
            let op = op.finalize("sdpa_fwd op")?;
            (vec![op], vec![k])
        }
        SdpaGraph::Softmax | SdpaGraph::Composite => {
            // K^T as a [.., d, sk] view of row-major K.
            let kt = tensor(UID_K, BF16, [1, bh, d, sk], [bh * sk * d, sk * d, 1, d], false, false)?;
            let (s, s2, p) = (full(100)?, full(101)?, full(106)?);
            let (d1, bmm1) = matmul_op(&q, &kt, &s)?;
            let (d2, mul) = pointwise_op(PM::CUDNN_POINTWISE_MUL, &s, Some(&scale), &s2)?;
            let (d8, bmm2) = matmul_op(&p, &v, &o)?;
            if kind == SdpaGraph::Softmax {
                let sm = Desc::new_raw(raw::OPERATION_SOFTMAX_DESCRIPTOR)?;
                sm.set_desc_raw(raw::ATTR_OPERATION_SOFTMAX_XDESC, &[&s2])?;
                sm.set_desc_raw(raw::ATTR_OPERATION_SOFTMAX_YDESC, &[&p])?;
                let sm = sm.finalize("softmax op")?;
                (vec![bmm1, mul, sm, bmm2], vec![kt, s, s2, p, d1, d2, d8])
            } else {
                let (mx, sub, ex, sum) = (row(102)?, full(103)?, full(104)?, row(105)?);
                let (d3, rmax) = reduction_op(RO::CUDNN_REDUCE_TENSOR_MAX, &s2, &mx)?;
                let (d4, psub) = pointwise_op(PM::CUDNN_POINTWISE_SUB, &s2, Some(&mx), &sub)?;
                let (d5, pexp) = pointwise_op(PM::CUDNN_POINTWISE_EXP, &sub, None, &ex)?;
                let (d6, rsum) = reduction_op(RO::CUDNN_REDUCE_TENSOR_ADD, &ex, &sum)?;
                let (d7, pdiv) = pointwise_op(PM::CUDNN_POINTWISE_DIV, &ex, Some(&sum), &p)?;
                (
                    vec![bmm1, mul, rmax, psub, pexp, rsum, pdiv, bmm2],
                    vec![kt, s, s2, p, mx, sub, ex, sum, d1, d2, d3, d4, d5, d6, d7, d8],
                )
            }
        }
    };
    let graph = Desc::new(DT::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)?;
    graph.set_desc(A::CUDNN_ATTR_OPERATIONGRAPH_OPS, &ops.iter().collect::<Vec<_>>())?;
    graph.set(A::CUDNN_ATTR_OPERATIONGRAPH_HANDLE, T::CUDNN_TYPE_HANDLE, &[h])?;
    let graph = graph.finalize("operation graph")?;
    keep.extend([q, v, o, scale]);
    keep.extend(ops);
    Ok(Graph { graph, _keep: keep })
}

/// A finalized execution plan (the descriptors it was built from stay alive
/// with it) and its workspace size.
pub struct SdpaPlan {
    plan: Desc,
    workspace: usize,
    /// Graph, heuristic mode, config index, engine global index, and the
    /// plan's JSON (engine name and knobs), for logs and the bench.
    pub info: String,
    /// Engine configs the heuristic offered in the mode that produced the plan.
    pub configs: usize,
    _keep: Vec<Desc>,
}

fn engine_index(cfg: &Desc) -> Option<i64> {
    let eng = Desc::new(DT::CUDNN_BACKEND_ENGINE_DESCRIPTOR).ok()?;
    let mut n = 0i64;
    let mut p = eng.0;
    let st = unsafe {
        raw::get(
            cfg.0,
            A::CUDNN_ATTR_ENGINECFG_ENGINE as u32,
            T::CUDNN_TYPE_BACKEND_DESCRIPTOR as u32,
            1,
            &mut n,
            &mut p as *mut cudnnBackendDescriptor_t as *mut c_void,
        )
    };
    if st != raw::STATUS_SUCCESS {
        return None;
    }
    eng.get_i64(A::CUDNN_ATTR_ENGINE_GLOBAL_INDEX).ok()
}

fn plan_json(plan: &Desc) -> String {
    let mut buf = vec![0u8; 4096];
    let mut n = 0i64;
    let st = unsafe {
        raw::get(
            plan.0,
            A::CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION as u32,
            T::CUDNN_TYPE_CHAR as u32,
            buf.len() as i64,
            &mut n,
            buf.as_mut_ptr() as *mut c_void,
        )
    };
    if st != raw::STATUS_SUCCESS {
        return String::new();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

fn build(
    dev: &super::device::DeviceContext,
    kind: SdpaGraph,
    want_cfg: usize,
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Result<SdpaPlan> {
    let h = handle(dev)?;
    let g = build_graph(h, kind, bh as i64, sq as i64, sk as i64, d as i64)?;

    // Heuristic modes in cudnn-frontend's order: A, then B, then FALLBACK.
    use sys::cudnnBackendHeurMode_t as HM;
    let mut last = String::from("no engine config");
    for mode in [HM::CUDNN_HEUR_MODE_A, HM::CUDNN_HEUR_MODE_B, HM::CUDNN_HEUR_MODE_FALLBACK] {
        let heur = Desc::new(DT::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)?;
        heur.set_desc(A::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, &[&g.graph])?;
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
        let mut ptrs: Vec<cudnnBackendDescriptor_t> = cfgs.iter().map(|c| c.0).collect();
        let mut count = 0i64;
        if let Err(e) = ok(
            unsafe {
                raw::get(
                    heur.0,
                    A::CUDNN_ATTR_ENGINEHEUR_RESULTS as u32,
                    T::CUDNN_TYPE_BACKEND_DESCRIPTOR as u32,
                    MAX_CFG as i64,
                    &mut count,
                    ptrs.as_mut_ptr() as *mut c_void,
                )
            },
            "heuristic results",
        ) {
            last = format!("{mode:?}: {e}");
            continue;
        }
        let count = (count.max(0) as usize).min(MAX_CFG);
        if count == 0 {
            last = format!("{mode:?}: 0 engine configs");
            continue;
        }
        // Configs in heuristic order; `want_cfg` skips that many that finalize.
        let mut skipped = 0usize;
        for (i, cfg) in cfgs.iter().take(count).enumerate() {
            let plan = Desc::new(DT::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)?;
            plan.set(A::CUDNN_ATTR_EXECUTION_PLAN_HANDLE, T::CUDNN_TYPE_HANDLE, &[h])?;
            plan.set_desc(A::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, &[cfg])?;
            match plan.finalize("execution plan") {
                Ok(plan) => {
                    if skipped < want_cfg {
                        skipped += 1;
                        continue;
                    }
                    let ws = plan.get_i64(A::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)?;
                    let json = plan_json(&plan);
                    let info = format!(
                        "graph {} {mode:?} config {i} of {count}, engine {}, {}",
                        kind.name(),
                        engine_index(cfg).map_or("?".into(), |e| e.to_string()),
                        json.chars().take(400).collect::<String>()
                    );
                    let mut keep = vec![g.graph, heur];
                    keep.extend(g._keep);
                    keep.extend(cfgs);
                    return Ok(SdpaPlan {
                        plan,
                        workspace: ws.max(0) as usize,
                        info,
                        configs: count,
                        _keep: keep,
                    });
                }
                Err(e) => last = format!("{mode:?} config {i}: {e}"),
            }
        }
        if skipped > 0 {
            last = format!("{mode:?}: only {skipped} executable configs (asked for index {want_cfg})");
        }
    }
    Err(msg(format!("cudnn sdpa ({}): no executable plan ({last})", kind.name())))
}

type Key = (SdpaGraph, usize, usize, usize, usize, usize);

type PlanMap = HashMap<Key, std::result::Result<Arc<SdpaPlan>, String>>;

fn plans() -> &'static Mutex<PlanMap> {
    static P: OnceLock<Mutex<PlanMap>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The cached plan for a shape on one graph and config index, built on first
/// use; the error (also cached) when cuDNN offers no engine for it.
pub fn plan_for_graph(
    kind: SdpaGraph,
    cfg: usize,
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> std::result::Result<Arc<SdpaPlan>, String> {
    let dev = super::device::global_device().ok_or_else(|| "cudnn sdpa: no device".to_string())?;
    let key = (kind, cfg, bh, sq, sk, d);
    let mut map = plans().lock().expect("cudnn sdpa plans");
    if let Some(p) = map.get(&key) {
        return p.clone();
    }
    let p = build(&dev, kind, cfg, bh, sq, sk, d).map(Arc::new).map_err(|e| e.to_string());
    map.insert(key, p.clone());
    p
}

/// The plan `FASTVIDEO_CUDNN_SDPA_GRAPH` / `FASTVIDEO_CUDNN_SDPA_CFG` select:
/// the first graph that yields one. `None` when no graph does.
fn plan_for(bh: usize, sq: usize, sk: usize, d: usize) -> Option<Arc<SdpaPlan>> {
    let cfg = super::envflag::usize_flag("FASTVIDEO_CUDNN_SDPA_CFG", 0);
    let mut errs = Vec::new();
    for kind in graphs_requested() {
        match plan_for_graph(kind, cfg, bh, sq, sk, d) {
            Ok(p) => {
                super::log::info_once(
                    &LOGGED,
                    format_args!(
                        "sdpa: cuDNN fused attention (cuDNN {}, {}, workspace {} B)",
                        unsafe { sys::cudnnGetVersion() },
                        p.info,
                        p.workspace
                    ),
                );
                return Some(p);
            }
            Err(e) => errs.push(e),
        }
    }
    super::log::info_once(
        &FAILED,
        format_args!("sdpa: cuDNN fused attention unavailable, using flash_mma: {}", errs.join("; ")),
    );
    None
}

/// Whether [`sdpa_bf16`] would run cuDNN for this shape (builds and caches
/// the plan on first use).
pub fn has_plan(bh: usize, sq: usize, sk: usize, d: usize) -> bool {
    plan_for(bh, sq, sk, d).is_some()
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
    let Some(plan) = plan_for(bh, sq, sk, d) else {
        return Ok(false);
    };
    execute(&plan, q, k, v, out, scale)?;
    Ok(true)
}

/// Run a plan from [`plan_for_graph`] (the bench and parity checks pick
/// graph and config explicitly).
pub fn execute(
    plan: &SdpaPlan,
    q: &CudaSlice<half::bf16>,
    k: &CudaSlice<half::bf16>,
    v: &CudaSlice<half::bf16>,
    out: &mut CudaSlice<half::bf16>,
    scale: f32,
) -> Result<()> {
    let dev = super::device::global_device().ok_or_else(|| msg("cudnn sdpa: no device"))?;
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
    ok(unsafe { raw::execute(h, plan.plan.0, pack.0) }, "execute")
}
