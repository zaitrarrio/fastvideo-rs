//! Device-memory scratch-buffer pool for eliminating per-op `cuMemAlloc` calls.
//!
//! The CUDA driver serialises all allocations through a global context lock, so
//! every `alloc_zeros` inside a tight GEMM/elementwise loop adds latency even
//! when the GPU is otherwise idle.  This module provides a size-class recycling
//! pool that amortises that cost: a released buffer is returned to its size
//! class instead of freed, and the next `acquire` of the same size re-uses it.
//!
//! ## Size classes
//!
//! Buffers are grouped into power-of-two size classes (number of `f32` elements).
//! A request for `n` elements is rounded up to the next power of two.  Classes
//! below 1 KiB elements are collapsed into the 1 KiB class to avoid unbounded
//! fragmentation at tiny sizes.  The pool is bounded per-class
//! (`MAX_FREE_PER_CLASS`); surplus buffers are dropped (freed) immediately.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let buf = ScratchPool::global().acquire(1024)?;   // CudaSlice<f32>
//! // ... use buf ...
//! ScratchPool::global().release(buf);
//! ```
//!
//! Buffers MUST NOT be released across CUDA context boundaries.  The pool is
//! keyed to the `global_device` context and will not accept buffers from a
//! different context.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use cudarc::driver::CudaSlice;

use super::device::{DeviceError, Result};

/// Maximum free buffers to keep per size class before dropping surplus.
const MAX_FREE_PER_CLASS: usize = 4;

/// Minimum size class: all requests below this are rounded up here.
const MIN_CLASS_ELEMS: usize = 1024;

type BufVec = Vec<CudaSlice<f32>>;

pub struct ScratchPool {
    /// (size_class_log2 → free list).  Access is infrequent (once per GEMM),
    /// so a single mutex for the whole table is fine.
    free: Mutex<std::collections::HashMap<u32, BufVec>>,
}

impl ScratchPool {
    fn new() -> Self {
        Self {
            free: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Process-wide singleton (one pool per process; safe across threads).
    pub fn global() -> &'static Self {
        static POOL: OnceLock<ScratchPool> = OnceLock::new();
        POOL.get_or_init(Self::new)
    }

    /// Acquire a `CudaSlice<f32>` of at least `n` elements.
    /// Returns a fresh (`alloc_zeros`) slice when the free list is empty.
    pub fn acquire(&self, n: usize) -> Result<CudaSlice<f32>> {
        let cls = size_class(n);
        let cls_elems = 1usize << cls;
        {
            let mut guard = self.free.lock().expect("scratch pool lock");
            if let Some(list) = guard.get_mut(&cls) {
                if let Some(buf) = list.pop() {
                    return Ok(buf);
                }
            }
        }
        let dev = super::device::global_device().ok_or_else(|| {
            DeviceError::Message("ScratchPool::acquire: no CUDA device context".into())
        })?;
        Ok(dev
            .stream
            .alloc_zeros::<f32>(cls_elems)
            .map_err(|e| DeviceError::Message(e.to_string()))?)
    }

    /// Return a buffer to the pool for reuse.  Drops the buffer (frees GPU
    /// memory) when the free list for its class is already full.
    pub fn release(&self, buf: CudaSlice<f32>) {
        let cls = size_class(buf.len());
        let mut guard = self.free.lock().expect("scratch pool lock");
        let list = guard.entry(cls).or_insert_with(Vec::new);
        if list.len() < MAX_FREE_PER_CLASS {
            list.push(buf);
        }
        // else: drop buf → cuMemFree
    }

    /// Drain all free lists (useful before context teardown).
    pub fn clear(&self) {
        let mut guard = self.free.lock().expect("scratch pool lock");
        guard.clear();
    }
}

fn size_class(n: usize) -> u32 {
    let n = n.max(MIN_CLASS_ELEMS);
    // Round up to next power of two.
    let p = n.next_power_of_two();
    p.trailing_zeros()
}
