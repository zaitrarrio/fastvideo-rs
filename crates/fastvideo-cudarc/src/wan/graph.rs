//! CUDA graphs for the causal SF-Wan block loop (serve design §8, E7).
//!
//! A block of the open-ended rollout ([`super::stream`]) is five DiT
//! forwards (the Self-Forcing steps and the clean-context pass) of a few
//! thousand kernel launches each. Captured once and replayed, the host
//! enqueues one graph launch where it enqueued every kernel.
//!
//! **Stream.** The device's own stream is the legacy default stream, which
//! cannot be captured. [`GraphStream`] is a second [`DeviceContext`] on the
//! same CUDA context with a non-blocking stream of its own, its own cuBLAS
//! handle (with an explicit workspace, so cuBLAS allocates nothing while
//! capturing) and its own cuDNN handle; every op of this crate reads its
//! device through [`super::device::global_device`], so
//! [`GraphStream::scope`] points the calling thread at it and every launch,
//! allocation and free of the scope lands on that stream. The legacy stream
//! is synchronized on entry and the graph stream before leaving, so work on
//! either side of the scope is ordered.
//!
//! **Capture.** [`GraphStream::capture`] runs a closure under
//! `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: a host synchronization or a
//! pageable copy inside it invalidates the capture and comes back as an
//! error, and the caller falls back to running the closure eagerly.
//! Allocations inside become graph memory nodes; every one must be freed
//! inside the graph too ([`Graph::escaped`] counts those that are not),
//! otherwise a replay would hand the same addresses out again. Inputs and
//! outputs are therefore persistent buffers the caller refills in place
//! ([`assign`]) before a launch.

use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// `FASTVIDEO_WAN_GRAPH` (default on): the SF-Wan rollout replays CUDA
/// graphs of its block forwards; `=0` runs every block eagerly on the
/// device's own stream (the pre-E7 path).
pub fn graphs_enabled() -> bool {
    static FLAG: super::envflag::CachedBool = super::envflag::CachedBool::new();
    FLAG.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_WAN_GRAPH", true))
}

/// A copy of `t` in a buffer of its own on the device (a host tensor is
/// uploaded), same dtype: a persistent slot for [`assign`].
pub(crate) fn duplicate(t: &CudaTensor) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    if t.is_device_fresh() {
        let out =
            super::act16::OutBuf::new(t.numel(), t.is_bf16())?.into_tensor(t.shape.clone())?;
        assign(&out, t)?;
        return Ok(out);
    }
    let mut c = t.clone();
    // The device copy only: a slot refilled in place must not keep a host
    // copy that would go stale.
    c.pin_device()?;
    Ok(c)
}

/// Overwrite `dst`'s device storage with `src`'s values (converted to
/// `dst`'s dtype; the same dtype copies bits), in place: `dst` keeps its
/// address. Both must hold the same number of elements.
pub(crate) fn assign(dst: &CudaTensor, src: &CudaTensor) -> Result<()> {
    let n = dst.numel();
    if src.numel() != n {
        return Err(msg(format!(
            "graph assign: {:?} into {:?}",
            src.shape, dst.shape
        )));
    }
    #[cfg(feature = "cuda")]
    if dst.is_device_fresh() {
        return super::act16::copy_rows_into(src, dst, 1, n, n, n, 0, 0);
    }
    Err(msg("graph assign: the destination is not a device buffer"))
}

#[cfg(feature = "cuda")]
pub use imp::{Graph, GraphStream};

#[cfg(feature = "cuda")]
mod imp {
    use std::sync::Arc;

    use cudarc::driver::sys;

    use super::super::device::{self, DeviceContext};
    use super::{msg, Result};

    fn check(r: sys::CUresult, what: &str) -> Result<()> {
        if r == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(msg(format!("{what}: {r:?}")))
        }
    }

    /// `FASTVIDEO_WAN_GRAPH_CUBLAS_WS_MIB` (default 32, cuBLAS's size for
    /// Hopper): the graph stream's explicit cuBLAS workspace.
    fn cublas_workspace_bytes() -> usize {
        super::super::envflag::usize_flag("FASTVIDEO_WAN_GRAPH_CUBLAS_WS_MIB", 32) << 20
    }

    struct Inner {
        dev: Arc<DeviceContext>,
        _cublas_ws: cudarc::driver::CudaSlice<u8>,
    }

    /// A capture-capable stream over the global device (see the module
    /// docs). Cheap to clone.
    #[derive(Clone)]
    pub struct GraphStream(Arc<Inner>);

    impl GraphStream {
        /// A new stream over the global device.
        pub fn new() -> Result<Self> {
            let base =
                device::global_device().ok_or_else(|| msg("graph stream: no CUDA device"))?;
            let dev = base.on_new_stream().map_err(|e| msg(e.to_string()))?;
            let bytes = cublas_workspace_bytes();
            let ws = dev
                .stream
                .alloc_zeros::<u8>(bytes.max(1))
                .map_err(|e| msg(e.to_string()))?;
            {
                let (p, _g) = cudarc::driver::DevicePtr::device_ptr(&ws, &dev.stream);
                let st = unsafe {
                    cudarc::cublas::sys::cublasSetWorkspace_v2(
                        *dev.cublas.handle(),
                        p as *mut std::ffi::c_void,
                        bytes,
                    )
                };
                if st != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(msg(format!("cublasSetWorkspace: {st:?}")));
                }
            }
            dev.synchronize().map_err(|e| msg(e.to_string()))?;
            Ok(Self(Arc::new(Inner {
                dev: Arc::new(dev),
                _cublas_ws: ws,
            })))
        }

        pub fn device(&self) -> &Arc<DeviceContext> {
            &self.0.dev
        }

        /// Run `f` with this thread's device on this stream: the legacy
        /// stream is drained first, and the previous device is put back
        /// after (also on a panic). Work `f` queues is not waited for.
        pub fn scope<R>(&self, f: impl FnOnce() -> R) -> R {
            if let Some(d) = device::global_device() {
                if !Arc::ptr_eq(&d, &self.0.dev) {
                    if let Err(e) = d.synchronize() {
                        super::super::log::info(format_args!("graph scope: legacy sync: {e}"));
                    }
                }
            }
            struct Restore(Option<Option<Arc<DeviceContext>>>);
            impl Drop for Restore {
                fn drop(&mut self) {
                    if let Some(prev) = self.0.take() {
                        device::replace_thread_device(prev);
                    }
                }
            }
            let _restore = Restore(Some(device::replace_thread_device(Some(
                self.0.dev.clone(),
            ))));
            f()
        }

        pub fn synchronize(&self) -> Result<()> {
            self.0.dev.synchronize().map_err(|e| msg(e.to_string()))
        }

        /// Capture what `f` queues on this stream (call inside
        /// [`Self::scope`]) into an instantiated graph. Nothing runs: launch
        /// it with [`Graph::launch`]. An error from `f` or from the capture
        /// (a host synchronization inside, say) ends the capture and is
        /// returned; the host state `f` changed is the caller's to undo.
        pub fn capture(&self, f: impl FnOnce() -> Result<()>) -> Result<Graph> {
            let dev = &self.0.dev;
            dev.ctx.bind_to_thread().map_err(|e| msg(e.to_string()))?;
            // Surface (and clear) any error recorded before: what the
            // capture records afterwards is the capture's.
            if let Err(e) = dev.ctx.check_err() {
                return Err(msg(format!("graph capture: pending CUDA error {e}")));
            }
            let stream = dev.stream.cu_stream();
            check(
                unsafe {
                    sys::cuStreamBeginCapture_v2(
                        stream,
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )
                },
                "cuStreamBeginCapture",
            )?;
            let body = f();
            let mut graph: sys::CUgraph = std::ptr::null_mut();
            let end = unsafe { sys::cuStreamEndCapture(stream, &mut graph) };
            let recorded = dev.ctx.check_err();
            let destroy = |g: sys::CUgraph| {
                if !g.is_null() {
                    unsafe { sys::cuGraphDestroy(g) };
                }
            };
            if let Err(e) = body {
                destroy(graph);
                return Err(msg(format!("graph capture: {e} (end capture: {end:?})")));
            }
            if let Err(e) = check(end, "cuStreamEndCapture") {
                destroy(graph);
                return Err(e);
            }
            if let Err(e) = recorded {
                destroy(graph);
                return Err(msg(format!("graph capture: CUDA error inside: {e}")));
            }
            if graph.is_null() {
                return Err(msg("graph capture: empty graph"));
            }
            let stats = match node_stats(graph) {
                Ok(s) => s,
                Err(e) => {
                    destroy(graph);
                    return Err(e);
                }
            };
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            if let Err(e) = check(
                unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) },
                "cuGraphInstantiate",
            ) {
                destroy(graph);
                return Err(e);
            }
            Ok(Graph {
                graph,
                exec,
                dev: dev.clone(),
                stats,
            })
        }

        /// Hand the graph memory pool's unused physical memory back.
        pub fn trim(&self) {
            let _ = self.synchronize();
            unsafe { sys::cuDeviceGraphMemTrim(self.0.dev.ctx.cu_device()) };
        }
    }

    /// Node counts of a captured graph.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct GraphStats {
        pub nodes: usize,
        pub kernels: usize,
        pub allocs: usize,
        pub frees: usize,
        /// Allocations with no free in the graph: memory that outlives it.
        pub escaped: usize,
        pub alloc_bytes: usize,
    }

    fn node_stats(graph: sys::CUgraph) -> Result<GraphStats> {
        let mut n = 0usize;
        check(
            unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n) },
            "cuGraphGetNodes",
        )?;
        let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
        check(
            unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n) },
            "cuGraphGetNodes",
        )?;
        nodes.truncate(n);
        let mut s = GraphStats {
            nodes: n,
            ..GraphStats::default()
        };
        let mut allocated = std::collections::HashSet::new();
        let mut freed = std::collections::HashSet::new();
        for &node in &nodes {
            let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            check(
                unsafe { sys::cuGraphNodeGetType(node, &mut ty) },
                "cuGraphNodeGetType",
            )?;
            match ty {
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => s.kernels += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => {
                    let mut p: sys::CUDA_MEM_ALLOC_NODE_PARAMS = unsafe { std::mem::zeroed() };
                    check(
                        unsafe { sys::cuGraphMemAllocNodeGetParams(node, &mut p) },
                        "cuGraphMemAllocNodeGetParams",
                    )?;
                    s.allocs += 1;
                    s.alloc_bytes += p.bytesize;
                    allocated.insert(p.dptr);
                }
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => {
                    let mut p: sys::CUdeviceptr = 0;
                    check(
                        unsafe { sys::cuGraphMemFreeNodeGetParams(node, &mut p) },
                        "cuGraphMemFreeNodeGetParams",
                    )?;
                    s.frees += 1;
                    freed.insert(p);
                }
                _ => {}
            }
        }
        s.escaped = allocated.difference(&freed).count();
        Ok(s)
    }

    /// An instantiated graph, launched on the stream it was captured on.
    pub struct Graph {
        graph: sys::CUgraph,
        exec: sys::CUgraphExec,
        dev: Arc<DeviceContext>,
        pub stats: GraphStats,
    }

    // The raw handles are used from one thread at a time (the rollout's
    // `&mut self`); CUDA graph objects need no more than that.
    unsafe impl Send for Graph {}
    unsafe impl Sync for Graph {}

    impl Graph {
        /// Allocations that outlive the graph (see [`GraphStats::escaped`]).
        pub fn escaped(&self) -> usize {
            self.stats.escaped
        }

        pub fn launch(&self) -> Result<()> {
            self.dev
                .ctx
                .bind_to_thread()
                .map_err(|e| msg(e.to_string()))?;
            check(
                unsafe { sys::cuGraphLaunch(self.exec, self.dev.stream.cu_stream()) },
                "cuGraphLaunch",
            )
        }
    }

    impl Drop for Graph {
        fn drop(&mut self) {
            if self.dev.ctx.bind_to_thread().is_err() {
                return;
            }
            unsafe {
                if !self.exec.is_null() {
                    sys::cuGraphExecDestroy(self.exec);
                }
                if !self.graph.is_null() {
                    sys::cuGraphDestroy(self.graph);
                }
            }
        }
    }

    pub use GraphStats as Stats;
}

#[cfg(feature = "cuda")]
pub use imp::Stats as GraphStats;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_needs_equal_sizes_and_a_device_buffer() {
        let a = CudaTensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = CudaTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(assign(&a, &b).is_err());
        // A host tensor has no storage to overwrite in place.
        assert!(assign(&a, &a.clone()).is_err());
    }
}
