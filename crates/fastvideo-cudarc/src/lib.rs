//! cudarc CUDA adapter.
//!
//! The device runtime shared by every family (tensors, device context, kernels,
//! ops, weights, offload) lives in [`rt`]. Native Wan T2V lives in [`wan`]: UMT5 →
//! DiT → feat-cache VAE on host `CudaTensor` with optional cuBLAS matmul when the
//! `cuda` feature is enabled and a global device context is set.

#![allow(unsafe_code)]
// Host reference code indexes tensors the way the Python it is checked against
// does; the iterator forms clippy suggests (`.iter().enumerate().take(n)`) hide
// the shape check an out-of-range index gives.
#![allow(clippy::needless_range_loop)]

pub mod cosmos;
pub mod cosmos3;
pub mod dreamx;
pub mod flux;
pub mod flux2;
pub mod gamecraft;
pub mod gen3c;
pub mod glm_image;
pub mod h3;
pub mod hooks;
pub mod hub_keys;
pub mod hunyuan15;
pub mod hyworld;
pub mod kandinsky5;
pub mod lingbot;
pub mod lingbotworld;
pub mod llm;
pub mod longcat;
pub mod ltx2;
pub mod matrixgame;
pub mod mmaudio;
pub mod pisa_attn;
pub mod rt;
pub mod sana_video;
pub mod sd35;
pub mod sink;
pub mod sol_attn;
pub mod stable_audio;
pub mod text_encode;
pub mod timing;
pub mod vae;
pub mod wan;
pub mod world_fuse;
pub mod zimage;

pub use hooks::{CancelToken, Hooks, Progress, Stage};
pub use rt::device::{resolve_device, DeviceError};
pub use rt::tensor::CudaTensor;
pub use sink::{AudioPcm, CollectFrames, FrameSink, SinkPort, VideoFrames};
pub use wan::{
    ClipVision, ClipVisionConfig, DenoiseStep, GenerateConfig, LoadParts, StepObserver, WanPipeline,
};

/// Host-side accounting for tests: counts the bytes of live heap blocks whose
/// size falls in a window, on the calling thread only, while armed. The
/// device path allocates one buffer where the host path allocates one `Vec`,
/// so the live count of activation-sized blocks during a host forward is the
/// live set the device pool would hold (score buffers, which the device
/// chunks, fall outside the window by construction of the test geometry).
#[cfg(test)]
pub(crate) mod alloc_track {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static WINDOW: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
        static LIVE: Cell<i64> = const { Cell::new(0) };
        static PEAK: Cell<i64> = const { Cell::new(0) };
    }

    pub struct Tracking;

    fn note(size: usize, add: bool) {
        let _ = WINDOW.try_with(|w| {
            let (lo, hi) = w.get();
            if hi == 0 || size < lo || size > hi {
                return;
            }
            let _ = LIVE.try_with(|l| {
                let v = l.get() + if add { size as i64 } else { -(size as i64) };
                l.set(v);
                let _ = PEAK.try_with(|p| p.set(p.get().max(v)));
            });
        });
    }

    unsafe impl GlobalAlloc for Tracking {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            note(l.size(), true);
            System.alloc(l)
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            note(l.size(), true);
            System.alloc_zeroed(l)
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            note(l.size(), false);
            System.dealloc(p, l)
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            note(l.size(), false);
            note(new, true);
            System.realloc(p, l, new)
        }
    }

    #[global_allocator]
    static GLOBAL: Tracking = Tracking;

    /// Run `f` and return its result with the highest net growth of live
    /// blocks of `lo..=hi` bytes during it (freeing a block that was alive
    /// before `f` counts as negative growth).
    pub fn peak_live<R>(lo: usize, hi: usize, f: impl FnOnce() -> R) -> (R, i64) {
        LIVE.with(|l| l.set(0));
        PEAK.with(|p| p.set(0));
        WINDOW.with(|w| w.set((lo, hi)));
        let out = f();
        WINDOW.with(|w| w.set((0, 0)));
        (out, PEAK.with(Cell::get))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_cuda_errors_without_feature() {
        if cfg!(feature = "cuda") {
            return;
        }
        let err = resolve_device("cuda").unwrap_err();
        assert!(err.to_string().contains("without `--features cuda`"));
    }
}
