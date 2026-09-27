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

/// Elements per staging chunk (64 MiB of bf16).
const CHUNK: usize = 32 << 20;

struct Stage {
    bufs: [PinnedHostSlice<bf16>; 2],
    pending: [Option<CudaEvent>; 2],
    next: usize,
}

thread_local! {
    static STAGE: RefCell<Option<Stage>> = const { RefCell::new(None) };
}

pub fn enabled() -> bool {
    static FLAG: super::envflag::CachedBool = super::envflag::CachedBool::new();
    FLAG.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_STAGED_UPLOAD", true))
}

/// Upload little-endian bf16 `bytes` (e.g. a view into a mapped shard) as a
/// new device buffer on the global stream. `None` when staging is off or its
/// pinned buffers cannot be had; the caller then takes its plain path.
pub fn upload_bf16_bytes(bytes: &[u8]) -> Result<Option<CudaSlice<bf16>>> {
    upload_bf16_parts(&[bytes])
}

/// [`upload_bf16_bytes`] of the concatenation of `parts` (stacked rows of a
/// fused projection), without building the concatenation on the host.
pub fn upload_bf16_parts(parts: &[&[u8]]) -> Result<Option<CudaSlice<bf16>>> {
    if !enabled() || parts.iter().any(|p| p.len() % 2 != 0) {
        return Ok(None);
    }
    let Some(dev) = super::device::global_device() else {
        return Ok(None);
    };
    let n: usize = parts.iter().map(|p| p.len() / 2).sum();
    STAGE.with(|slot| -> Result<Option<CudaSlice<bf16>>> {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            // SAFETY: every element is written before it is read.
            let made = (|| -> std::result::Result<_, cudarc::driver::DriverError> {
                let a = unsafe { dev.ctx.alloc_pinned::<bf16>(CHUNK) }?;
                let b = unsafe { dev.ctx.alloc_pinned::<bf16>(CHUNK) }?;
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
        let mut dst = unsafe { dev.stream.alloc::<bf16>(n) }.map_err(err)?;
        let mut at = 0;
        for bytes in parts {
            let part_n = bytes.len() / 2;
            let mut off = 0;
            while off < part_n {
                let len = (part_n - off).min(CHUNK);
                let k = stage.next;
                stage.next ^= 1;
                if let Some(ev) = stage.pending[k].take() {
                    let t = Instant::now();
                    ev.synchronize().map_err(err)?;
                    add_consumer_time(ConsumerTime::H2dWait, t.elapsed());
                }
                let t = Instant::now();
                let host = stage.bufs[k].as_mut_slice().map_err(err)?;
                // SAFETY: bf16 is two plain bytes; the pinned buffer is aligned.
                let dst_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), len * 2)
                };
                dst_bytes
                    .par_chunks_mut(4 << 20)
                    .zip(bytes[off * 2..(off + len) * 2].par_chunks(4 << 20))
                    .for_each(|(d, s)| d.copy_from_slice(s));
                add_consumer_time(ConsumerTime::Fill, t.elapsed());
                let mut view = dst.slice_mut(at + off..at + off + len);
                dev.stream
                    .memcpy_htod(&host[..len], &mut view)
                    .map_err(err)?;
                stage.pending[k] = Some(dev.stream.record_event(None).map_err(err)?);
                off += len;
            }
            at += part_n;
        }
        stats::record_h2d(n / 2);
        Ok(Some(dst))
    })
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
