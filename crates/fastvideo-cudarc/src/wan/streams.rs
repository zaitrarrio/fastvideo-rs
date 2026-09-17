//! CUDA stream pool for overlapping QKV/attention with MLP within a DiT block.
//!
//! Two streams (QKV/attention on stream 0, MLP/FFN on stream 1) overlap
//! compute and let the GEMM scheduler hide softmax / RMS-norm latencies.
//! `FASTVIDEO_TWO_STREAMS=1` is the default on sm_90 (Hopper).

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use super::device::{global_device, DeviceError, Result};
use super::envflag::CachedBool;

static TWO_STREAMS_CACHE: CachedBool = CachedBool::new();
static CUGRAPH_CACHE: CachedBool = CachedBool::new();

/// Whether to enable the two-stream DiT path (`FASTVIDEO_TWO_STREAMS=1` default on).
/// Cached: consulted once per DiT block per step.
pub fn two_streams_enabled() -> bool {
    TWO_STREAMS_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_TWO_STREAMS", true))
}

/// Whether to enable cuGraph capture for DiT blocks (`FASTVIDEO_CUGRAPH=1`).
/// Defaults **off**: as of this crate's `layer_norm`/modulate fixes the DiT
/// block body no longer forces D2H/H2D round trips mid-block, so capture is
/// now *safe* to attempt, but the denoising loop does not yet feed captured
/// graphs stable, pre-allocated input buffers on replay (see the scratch-pool
/// module) — enabling this without that would replay against stale pointers.
/// Set explicitly to `1` only once that wiring lands; until then this stays
/// opt-in scaffolding, not a silent default flip.
pub fn cugraph_enabled() -> bool {
    CUGRAPH_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_CUGRAPH", false))
}

/// Pool of side streams created on demand. Each entry is a `CudaStream` that
/// shares the global `CudaContext`. The primary (default) stream is still
/// owned by the `DeviceContext`; this pool holds auxiliary streams.
#[derive(Default)]
pub struct StreamPool {
    streams: Vec<Arc<CudaStream>>,
}

impl StreamPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocate `count` additional streams on the active context.
    pub fn ensure(&mut self, count: usize) -> Result<()> {
        if self.streams.len() >= count {
            return Ok(());
        }
        let dev = global_device().ok_or_else(|| {
            DeviceError::Message("no global CUDA device context".into())
        })?;
        while self.streams.len() < count {
            let stream = CudaContext::new_stream(&dev.ctx)?;
            self.streams.push(stream);
        }
        Ok(())
    }

    pub fn get(&self, idx: usize) -> Option<Arc<CudaStream>> {
        self.streams.get(idx).cloned()
    }

    pub fn len(&self) -> usize {
        self.streams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}