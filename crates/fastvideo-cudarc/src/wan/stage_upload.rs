//! Weight upload through pinned staging (E12).
//!
//! The plain path for a bf16 weight is: allocate a host `Vec`, copy the
//! mapped bytes into it, `memcpy_stod` from that pageable buffer (the driver
//! bounces it through its own staging memory and the call returns only when
//! done). Here the mapped bytes are copied straight into one of two
//! page-locked buffers (in parallel, which is also where the mapping's pages
//! are faulted in) and the DMA of one chunk runs while the next is filled. The
//! device receives the same bytes; only the route differs.
//!
//! `FASTVIDEO_STAGED_UPLOAD=0` turns it off.

use std::cell::RefCell;
use std::time::Instant;

use cudarc::driver::{CudaEvent, CudaSlice, PinnedHostSlice};
use half::bf16;
use rayon::prelude::*;

use super::stats;
use super::tensor::{Result, TensorError};
use fastvideo_loader::prefetch::{add_consumer_time, ConsumerTime};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Bytes per staging chunk (a multiple of every element size used).
const CHUNK: usize = 64 << 20;

struct Stage {
    bufs: [PinnedHostSlice<u8>; 2],
    pending: [Option<CudaEvent>; 2],
    next: usize,
}

thread_local! {
    static STAGE: RefCell<Option<Stage>> = const { RefCell::new(None) };
}

static DEFAULT_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Make staging default to off for the rest of the process unless
/// `FASTVIDEO_STAGED_UPLOAD` is set explicitly (see
/// [`fastvideo_loader::prefetch::default_off`]).
pub fn default_off() {
    DEFAULT_OFF.store(true, std::sync::atomic::Ordering::Release);
}

pub fn enabled() -> bool {
    static EXPLICIT: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let explicit = EXPLICIT.get_or_init(|| {
        std::env::var_os("FASTVIDEO_STAGED_UPLOAD")
            .map(|_| super::envflag::bool_flag("FASTVIDEO_STAGED_UPLOAD", true))
    });
    explicit.unwrap_or_else(|| !DEFAULT_OFF.load(std::sync::atomic::Ordering::Acquire))
}

/// Upload little-endian bf16 `bytes` (e.g. a view into a mapped shard) as a
/// new device buffer on the global stream. `None` when staging is off or its
/// pinned buffers cannot be had; the caller then takes its plain path.
pub fn upload_bf16_bytes(bytes: &[u8]) -> Result<Option<CudaSlice<bf16>>> {
    upload_parts::<bf16>(&[bytes])
}

/// [`upload_bf16_bytes`] of the concatenation of `parts` (stacked rows of a
/// fused projection), without building the concatenation on the host.
pub fn upload_bf16_parts(parts: &[&[u8]]) -> Result<Option<CudaSlice<bf16>>> {
    upload_parts::<bf16>(parts)
}

/// A host bf16 buffer (e.g. a weight fused on the host) through the stage.
pub fn upload_bf16_values(values: &[bf16]) -> Result<Option<CudaSlice<bf16>>> {
    // SAFETY: bf16 is two plain bytes.
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) };
    upload_parts::<bf16>(&[bytes])
}

/// Raw bytes (FP8 codes) through the stage.
pub fn upload_u8(bytes: &[u8]) -> Result<Option<CudaSlice<u8>>> {
    upload_parts::<u8>(&[bytes])
}

/// The concatenation of `parts` (little-endian `T`s) as a new `CudaSlice<T>`.
fn upload_parts<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    parts: &[&[u8]],
) -> Result<Option<CudaSlice<T>>> {
    let size = std::mem::size_of::<T>();
    if !enabled() || parts.iter().any(|p| p.len() % size != 0) {
        return Ok(None);
    }
    let Some(dev) = super::device::global_device() else {
        return Ok(None);
    };
    let n: usize = parts.iter().map(|p| p.len() / size).sum();
    STAGE.with(|slot| -> Result<Option<CudaSlice<T>>> {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            // SAFETY: every byte is written before it is read.
            let made = (|| -> std::result::Result<_, cudarc::driver::DriverError> {
                let a = unsafe { dev.ctx.alloc_pinned::<u8>(CHUNK) }?;
                let b = unsafe { dev.ctx.alloc_pinned::<u8>(CHUNK) }?;
                Ok([a, b])
            })();
            match made {
                Ok(bufs) => {
                    *slot = Some(Stage {
                        bufs,
                        pending: [None, None],
                        next: 0,
                    })
                }
                Err(e) => {
                    crate::wan::log::info(format_args!("staged upload off: {e}"));
                    return Ok(None);
                }
            }
        }
        let stage = slot.as_mut().expect("just made");
        let err = |e: cudarc::driver::DriverError| msg(format!("staged upload: {e}"));
        // SAFETY: fully written by the copies below before any kernel reads it
        // (they are ordered on the same stream).
        let mut dst = unsafe { dev.stream.alloc::<T>(n) }.map_err(err)?;
        let mut at = 0;
        for bytes in parts {
            let mut off = 0;
            while off < bytes.len() {
                let len = (bytes.len() - off).min(CHUNK);
                let k = stage.next;
                stage.next ^= 1;
                if let Some(ev) = stage.pending[k].take() {
                    let t = Instant::now();
                    ev.synchronize().map_err(err)?;
                    add_consumer_time(ConsumerTime::H2dWait, t.elapsed());
                }
                let t = Instant::now();
                let host = stage.bufs[k].as_mut_slice().map_err(err)?;
                host[..len]
                    .par_chunks_mut(4 << 20)
                    .zip(bytes[off..off + len].par_chunks(4 << 20))
                    .for_each(|(d, s)| d.copy_from_slice(s));
                add_consumer_time(ConsumerTime::Fill, t.elapsed());
                // SAFETY: pinned memory is page-aligned and `len` is a whole
                // number of `T`s; `T` is plain data.
                let typed: &[T] =
                    unsafe { std::slice::from_raw_parts(host.as_ptr().cast::<T>(), len / size) };
                let first = (at + off) / size;
                let mut view = dst.slice_mut(first..first + len / size);
                dev.stream.memcpy_htod(typed, &mut view).map_err(err)?;
                stage.pending[k] = Some(dev.stream.record_event(None).map_err(err)?);
                off += len;
            }
            at += bytes.len();
        }
        stats::record_h2d(n * size / 4);
        Ok(Some(dst))
    })
}

/// `FASTVIDEO_VERIFY_UPLOAD=1`: loaders also build the plain upload and
/// compare it with the staged one.
pub fn verify_enabled() -> bool {
    static FLAG: super::envflag::CachedBool = super::envflag::CachedBool::new();
    FLAG.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_VERIFY_UPLOAD", false))
}

static VERIFY: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Book one comparison; the first few mismatches are logged by name.
pub fn record_verify(what: &str, equal: bool) {
    let i = usize::from(!equal);
    let n = VERIFY[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !equal && n < 5 {
        crate::wan::log::info(format_args!("staged upload MISMATCH: {what}"));
    }
}

/// `(equal, different)` comparisons so far.
pub fn verify_counts() -> (u64, u64) {
    (
        VERIFY[0].load(std::sync::atomic::Ordering::Relaxed),
        VERIFY[1].load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Wait for every staged copy this thread queued (before its pinned buffers
/// could be dropped, or for a timing that must include the transfers).
pub fn drain() -> Result<()> {
    STAGE.with(|slot| {
        if let Some(stage) = slot.borrow_mut().as_mut() {
            for ev in stage.pending.iter_mut().filter_map(Option::take) {
                ev.synchronize()
                    .map_err(|e| msg(format!("staged upload: {e}")))?;
            }
        }
        Ok(())
    })
}
