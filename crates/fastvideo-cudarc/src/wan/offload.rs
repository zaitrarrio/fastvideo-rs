//! Block weight residency: a DiT's blocks either live on the device for the
//! whole run ([`Residency::Resident`]) or are **streamed**: their weights sit
//! in page-locked host memory and are copied into a small ring of device slots
//! just ahead of the block that reads them.
//!
//! Streaming is the layerwise offload of the reference's 32 GiB profiles
//! (SGLang `LayerwiseOffloadManager`, `dit_offload_prefetch_size=1`): while
//! block `i` computes on the compute stream, block `i + 1` (or the next
//! [`lookahead`] blocks) is copied host-to-device on a second stream.
//!
//! * **Same numbers.** Only a linear's weight moves; the kernels, the weight
//!   bits and every other tensor are those of the resident path. A block is
//!   rebuilt for one forward by cloning its *skeleton* (norms, tables, biases
//!   and whatever cannot stream, all still resident) and installing the slot's
//!   buffers into its linears, in the fixed order
//!   [`OffloadBlock::for_each_linear_mut`] visits them.
//! * **Explicit synchronization** (this crate runs cudarc with event tracking
//!   off, as the text-encoder prefetcher does, see `llm/prefetch.rs`): the
//!   copy of a slot waits on the event the compute stream recorded after the
//!   slot's previous block ran; the compute stream waits on the event the copy
//!   stream recorded after the slot was filled. No host synchronization.
//! * **Recycled buffers.** A slot keeps its device buffers for the whole run
//!   (one allocation per streamed linear, sized by the first block it holds);
//!   blocks of one model share their shapes, so nothing is allocated after the
//!   first pass. The ring holds `lookahead + 1` slots and wraps: the last block
//!   of a forward prefetches the first block of the next.
//! * **What it costs.** `slots x block bytes` on the device, the whole stack in
//!   pinned host memory, and whatever part of each copy the block before it
//!   cannot hide. [`BlockWeights::report`] logs the copy throughput and the
//!   hidden fraction measured with timing events.
//!
//! Only plain bf16 linears stream (and, on host runs, plain f32 ones); a
//! quantized, LoRA-carrying or nvfp4 linear stays in the skeleton, resident.
//!
//! `FASTVIDEO_DIT_OFFLOAD=auto|resident|streamed` picks the policy when the
//! caller does not; `FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD` (default 1) the depth.

use std::sync::Mutex;

use super::nn::{Linear, StreamedWeight};
use super::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

pub const ENV: &str = "FASTVIDEO_DIT_OFFLOAD";
pub const LOOKAHEAD_ENV: &str = "FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD";
const GIB: f64 = (1u64 << 30) as f64;

/// The requested policy. `Auto` keeps the blocks resident when the free
/// device memory covers the planned need, and streams them otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DitOffload {
    #[default]
    Auto,
    Resident,
    Streamed,
}

impl DitOffload {
    pub fn parse(name: &str) -> std::result::Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Self::Auto),
            "resident" | "off" | "0" | "none" => Ok(Self::Resident),
            "streamed" | "stream" | "layerwise" | "on" | "1" => Ok(Self::Streamed),
            other => Err(format!(
                "unknown DiT offload '{other}' (auto|resident|streamed)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Resident => "resident",
            Self::Streamed => "streamed",
        }
    }

    /// `explicit` (a CLI flag) wins, then [`ENV`], then `Auto`.
    pub fn from_env_or(explicit: Option<Self>) -> std::result::Result<Self, String> {
        if let Some(p) = explicit {
            return Ok(p);
        }
        match std::env::var(ENV) {
            Ok(v) => Self::parse(&v).map_err(|e| format!("{ENV}: {e}")),
            Err(_) => Ok(Self::Auto),
        }
    }

    /// `need`: device bytes of the run with the blocks resident. `free`: what
    /// the device reports before the model loads (`None`: no device, or it
    /// would not say; nothing is gained by streaming then).
    pub fn resolve(self, need: u64, free: Option<u64>) -> Residency {
        match self {
            Self::Resident => Residency::Resident,
            Self::Streamed => Residency::Streamed,
            Self::Auto => match free {
                Some(free) if free < need => Residency::Streamed,
                _ => Residency::Resident,
            },
        }
    }
}

/// Where a model's blocks live for a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    Resident,
    Streamed,
}

impl Residency {
    pub fn is_streamed(self) -> bool {
        self == Self::Streamed
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Streamed => "streamed",
        }
    }
}

/// Blocks copied ahead of the one computing (default 1, at most 8). The ring
/// has one more slot than this.
pub fn lookahead() -> usize {
    super::envflag::usize_flag(LOOKAHEAD_ENV, 1).clamp(1, 8)
}

/// A block whose linears can be streamed. `for_each_linear_mut` must visit
/// the same linears in the same order on every call and on every clone.
pub trait OffloadBlock: Clone {
    fn for_each_linear_mut(&mut self, f: &mut dyn FnMut(&mut Linear) -> Result<()>) -> Result<()>;
}

/// Copy throughput and overlap of a streamed model since the last report.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OffloadStats {
    /// Block copies issued.
    pub copies: u64,
    pub bytes: u64,
    /// Copies whose timing was read back (timing events are resolved lazily).
    pub timed: u64,
    /// Device time of the timed copies.
    pub copy_ms: f64,
    /// Part of `copy_ms` the compute stream waited for: the copy was not done
    /// when the block was needed.
    pub exposed_ms: f64,
    /// Longest single stall.
    pub worst_exposed_ms: f64,
}

impl OffloadStats {
    /// GB/s of the timed copies.
    pub fn throughput_gbs(&self) -> f64 {
        if self.copy_ms <= 0.0 || self.copies == 0 {
            return 0.0;
        }
        let timed_bytes = self.bytes as f64 * self.timed as f64 / self.copies as f64;
        timed_bytes / (self.copy_ms * 1e-3) / 1e9
    }

    /// Fraction of copy time hidden behind compute (1 = fully overlapped).
    pub fn hidden_fraction(&self) -> f64 {
        if self.copy_ms <= 0.0 {
            return 1.0;
        }
        (1.0 - self.exposed_ms / self.copy_ms).clamp(0.0, 1.0)
    }
}

/// One linear's weight while it is off the device.
enum HostPart {
    /// Elements `[offset, offset + len)` of the block's pinned buffer.
    #[cfg(feature = "cuda")]
    Pinned { offset: usize, len: usize },
    /// Host runs: the tensor itself; a forward installs a clone.
    Tensor(super::tensor::CudaTensor),
}

struct HostBlock {
    /// One entry per linear in visit order; `None` stays in the skeleton.
    parts: Vec<Option<HostPart>>,
    #[cfg(feature = "cuda")]
    pinned: Option<cudarc::driver::PinnedHostSlice<half::bf16>>,
    bytes: u64,
}

/// A model's blocks with their residency.
pub struct BlockWeights<B> {
    /// Resident: the blocks. Streamed: the skeletons.
    blocks: Vec<B>,
    host: Vec<HostBlock>,
    residency: Residency,
    lookahead: usize,
    label: &'static str,
    #[cfg(feature = "cuda")]
    ring: Mutex<Option<ring::DeviceRing>>,
    stats: Mutex<OffloadStats>,
}

impl<B: OffloadBlock> BlockWeights<B> {
    pub fn new(label: &'static str, residency: Residency) -> Self {
        Self {
            blocks: Vec::new(),
            host: Vec::new(),
            residency,
            lookahead: lookahead(),
            label,
            #[cfg(feature = "cuda")]
            ring: Mutex::new(None),
            stats: Mutex::new(OffloadStats::default()),
        }
    }

    pub fn resident(label: &'static str, blocks: Vec<B>) -> Self {
        let mut w = Self::new(label, Residency::Resident);
        w.blocks = blocks;
        w
    }

    /// Copies kept ahead of the computing block (streamed runs).
    pub fn with_lookahead(mut self, lookahead: usize) -> Self {
        self.lookahead = lookahead.clamp(1, 8);
        self
    }

    pub fn residency(&self) -> Residency {
        self.residency
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Block `i` as loaded (resident) or without its streamed weights. Fine
    /// for anything but its linears: norms, tables, flags.
    pub fn skeleton(&self, i: usize) -> &B {
        &self.blocks[i]
    }

    /// Every block as loaded (resident) or its skeleton (streamed): for
    /// changes to what stays on the device, such as a LoRA re-fuse.
    pub fn skeletons_mut(&mut self) -> impl Iterator<Item = &mut B> {
        self.blocks.iter_mut()
    }

    /// Bytes of streamed weight kept in host memory.
    pub fn host_bytes(&self) -> u64 {
        self.host.iter().map(|h| h.bytes).sum()
    }

    /// Bytes of one block's streamed weight (the largest block).
    pub fn block_bytes(&self) -> u64 {
        self.host.iter().map(|h| h.bytes).max().unwrap_or(0)
    }

    /// Device slots the ring holds.
    pub fn slots(&self) -> usize {
        self.lookahead + 1
    }

    /// Add the next block. Streamed: its linears' weights leave the device
    /// now (copied into one pinned host buffer), so loading holds one block
    /// on the device at a time.
    pub fn push(&mut self, mut block: B) -> Result<()> {
        if self.residency == Residency::Resident {
            self.blocks.push(block);
            return Ok(());
        }
        let mut taken: Vec<Option<StreamedWeight>> = Vec::new();
        block.for_each_linear_mut(&mut |lin| {
            taken.push(lin.take_streamed_weight());
            Ok(())
        })?;
        let host = HostBlock::from_taken(taken)?;
        self.host.push(host);
        self.blocks.push(block);
        Ok(())
    }

    /// Run `f` on block `index` with its weights on the device. Streamed:
    /// waits (on the device) for the block's copy, and queues the copies of
    /// the next [`Self::slots`]` - 1` blocks (wrapping to the start) first.
    pub fn with<R>(&self, index: usize, f: impl FnOnce(&B) -> Result<R>) -> Result<R> {
        if index >= self.blocks.len() {
            return Err(msg(format!(
                "{}: block {index} of {}",
                self.label,
                self.blocks.len()
            )));
        }
        if self.residency == Residency::Resident {
            return f(&self.blocks[index]);
        }
        #[cfg(feature = "cuda")]
        if self.host[index].pinned.is_some() {
            return self.with_device(index, f);
        }
        // Host run: the parts are the tensors themselves.
        let mut block = self.blocks[index].clone();
        let parts = &self.host[index].parts;
        let mut k = 0;
        block.for_each_linear_mut(&mut |lin| {
            if let Some(Some(HostPart::Tensor(t))) = parts.get(k) {
                lin.put_streamed_weight(StreamedWeight::Tensor(t.clone()));
            }
            k += 1;
            Ok(())
        })?;
        f(&block)
    }

    #[cfg(feature = "cuda")]
    fn with_device<R>(&self, index: usize, f: impl FnOnce(&B) -> Result<R>) -> Result<R> {
        let n = self.blocks.len();
        let (block, slot) = {
            let mut guard = self.ring.lock().expect("offload ring");
            if guard.is_none() {
                *guard = Some(ring::DeviceRing::new(self.slots())?);
            }
            let ring = guard.as_mut().expect("ring");
            let window: Vec<usize> = (0..self.slots().min(n)).map(|k| (index + k) % n).collect();
            let mut stats = self.stats.lock().expect("offload stats");
            for &j in &window {
                ring.ensure(j, &self.host[j], &window, &mut stats)?;
            }
            ring.resolve_timings(&mut stats, false);
            let slot = ring.acquire(index)?;
            let mut block = self.blocks[index].clone();
            let parts = &self.host[index].parts;
            let mut installs = ring.weights(slot).into_iter();
            let mut k = 0;
            block.for_each_linear_mut(&mut |lin| {
                if matches!(parts.get(k), Some(Some(HostPart::Pinned { .. }))) {
                    let w = installs
                        .next()
                        .ok_or_else(|| msg("offload: slot has fewer parts than the block"))?;
                    lin.put_streamed_weight(StreamedWeight::Bf16(w));
                }
                k += 1;
                Ok(())
            })?;
            (block, slot)
        };
        let out = f(&block);
        drop(block);
        let mut guard = self.ring.lock().expect("offload ring");
        guard.as_mut().expect("ring").release(slot)?;
        out
    }

    /// Log the copy throughput and overlap since the last report, then reset
    /// the counters. Resolves every outstanding timing event (a host wait for
    /// work that is normally long finished).
    pub fn report(&self, what: &str) -> OffloadStats {
        // Lock order everywhere: ring, then stats.
        #[cfg(feature = "cuda")]
        let mut ring = self.ring.lock().expect("offload ring");
        let mut stats = self.stats.lock().expect("offload stats");
        #[cfg(feature = "cuda")]
        if let Some(ring) = ring.as_mut() {
            ring.resolve_timings(&mut stats, true);
        }
        let s = *stats;
        *stats = OffloadStats::default();
        if self.residency == Residency::Streamed && s.copies > 0 {
            super::log::info(format_args!(
                "{} offload {what}: {} block copies, {:.1} GiB H2D, {:.1} GB/s, copy {:.2}s of which {:.1}% hidden behind compute (exposed {:.2}s, worst stall {:.1} ms)",
                self.label,
                s.copies,
                s.bytes as f64 / GIB,
                s.throughput_gbs(),
                s.copy_ms * 1e-3,
                100.0 * s.hidden_fraction(),
                s.exposed_ms * 1e-3,
                s.worst_exposed_ms,
            ));
        }
        s
    }

    /// Free the device ring (streamed runs): its slots go back to the pool
    /// until the next forward builds a new one. Waits for the kernels that
    /// read them.
    pub fn release_device(&self) {
        #[cfg(feature = "cuda")]
        {
            let mut guard = self.ring.lock().expect("offload ring");
            if let Some(mut ring) = guard.take() {
                ring.resolve_timings(&mut self.stats.lock().expect("offload stats"), true);
                drop(ring);
            }
        }
    }

    /// One line describing the residency, for load logs.
    pub fn describe(&self) -> String {
        match self.residency {
            Residency::Resident => format!("{}: {} blocks resident", self.label, self.len()),
            Residency::Streamed => format!(
                "{}: {} blocks streamed ({:.2} GiB pinned host, {} device slots x {:.2} GiB, lookahead {})",
                self.label,
                self.len(),
                self.host_bytes() as f64 / GIB,
                self.slots(),
                self.block_bytes() as f64 / GIB,
                self.lookahead
            ),
        }
    }
}

impl HostBlock {
    fn from_taken(taken: Vec<Option<StreamedWeight>>) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let elems: usize = taken
                .iter()
                .map(|t| match t {
                    Some(StreamedWeight::Bf16(w)) => w.len(),
                    _ => 0,
                })
                .sum();
            if elems > 0 {
                return Self::pin(taken, elems);
            }
        }
        let mut bytes = 0u64;
        let parts = taken
            .into_iter()
            .map(|t| match t {
                Some(StreamedWeight::Tensor(t)) => {
                    bytes += t.numel() as u64 * 4;
                    Some(HostPart::Tensor(t))
                }
                #[cfg(feature = "cuda")]
                Some(StreamedWeight::Bf16(_)) => unreachable!("counted above"),
                None => None,
            })
            .collect();
        Ok(Self {
            parts,
            #[cfg(feature = "cuda")]
            pinned: None,
            bytes,
        })
    }

    /// Copy every bf16 weight of the block into one pinned buffer and let the
    /// device copies go.
    #[cfg(feature = "cuda")]
    fn pin(taken: Vec<Option<StreamedWeight>>, elems: usize) -> Result<Self> {
        let dev = super::device::global_device().ok_or_else(|| msg("offload: no device"))?;
        let err = |e: cudarc::driver::DriverError| {
            msg(format!(
                "offload: pinning {:.2} GiB of block weights: {e}",
                (elems * 2) as f64 / GIB
            ))
        };
        // Every element is written by the copies below before it is read.
        let mut pinned = unsafe { dev.ctx.alloc_pinned::<half::bf16>(elems) }.map_err(err)?;
        let mut parts = Vec::with_capacity(taken.len());
        let mut device = Vec::new();
        {
            let host = pinned.as_mut_slice().map_err(err)?;
            let mut at = 0;
            for t in taken {
                match t {
                    Some(StreamedWeight::Bf16(w)) => {
                        let len = w.len();
                        dev.stream
                            .memcpy_dtoh(&*w, &mut host[at..at + len])
                            .map_err(err)?;
                        super::stats::record_d2h(len / 2);
                        parts.push(Some(HostPart::Pinned { offset: at, len }));
                        at += len;
                        device.push(w);
                    }
                    Some(StreamedWeight::Tensor(_)) => {
                        return Err(msg(
                            "offload: a block mixes device and host weights; cannot stream it",
                        ))
                    }
                    None => parts.push(None),
                }
            }
            dev.stream.synchronize().map_err(err)?;
        }
        drop(device);
        Ok(Self {
            parts,
            pinned: Some(pinned),
            bytes: elems as u64 * 2,
        })
    }
}

#[cfg(feature = "cuda")]
mod ring {
    //! The device side: slots, the copy stream and the events between them.

    use super::{msg, HostBlock, HostPart, OffloadStats};
    use crate::wan::tensor::Result;
    use cudarc::driver::sys::CUevent_flags;
    use cudarc::driver::{CudaEvent, CudaSlice, CudaStream};
    use half::bf16;
    use std::sync::Arc;

    struct Slot {
        /// One buffer per streamed linear of the block it holds.
        parts: Vec<Arc<CudaSlice<bf16>>>,
        block: Option<usize>,
        /// Copy stream, after the slot's copies.
        ready: Option<CudaEvent>,
        /// Compute stream, after the last kernel that read the slot.
        freed: Option<CudaEvent>,
        in_use: bool,
        last_use: u64,
    }

    /// A copy whose timing is not read back yet.
    struct Pending {
        start: CudaEvent,
        end: CudaEvent,
        /// Compute stream, when the block was asked for (`None`: never used).
        wanted: Option<CudaEvent>,
    }

    pub(super) struct DeviceRing {
        copy: Arc<CudaStream>,
        compute: Arc<CudaStream>,
        slots: Vec<Slot>,
        tick: u64,
        /// Per slot: the timing of its latest copy.
        pending: Vec<Option<Pending>>,
        done: Vec<Pending>,
    }

    fn timed(stream: &CudaStream) -> Result<CudaEvent> {
        stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| msg(format!("offload event: {e}")))
    }

    impl DeviceRing {
        pub(super) fn new(slots: usize) -> Result<Self> {
            let dev =
                crate::wan::device::global_device().ok_or_else(|| msg("offload: no device"))?;
            let copy = dev
                .ctx
                .new_stream()
                .map_err(|e| msg(format!("offload copy stream: {e}")))?;
            Ok(Self {
                copy,
                compute: dev.stream.clone(),
                slots: (0..slots)
                    .map(|_| Slot {
                        parts: Vec::new(),
                        block: None,
                        ready: None,
                        freed: None,
                        in_use: false,
                        last_use: 0,
                    })
                    .collect(),
                tick: 0,
                pending: (0..slots).map(|_| None).collect(),
                done: Vec::new(),
            })
        }

        /// Make sure block `j` is in a slot or on its way there. A slot is
        /// reused only when it holds none of `window` and is not in use; its
        /// copy then waits for the kernels that last read it.
        pub(super) fn ensure(
            &mut self,
            j: usize,
            host: &HostBlock,
            window: &[usize],
            stats: &mut OffloadStats,
        ) -> Result<()> {
            if self.slots.iter().any(|s| s.block == Some(j)) {
                return Ok(());
            }
            let s = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.in_use && s.block.is_none_or(|b| !window.contains(&b)))
                .min_by_key(|(_, s)| (s.block.is_some(), s.last_use))
                .map(|(i, _)| i)
                .ok_or_else(|| msg("offload: every slot is busy"))?;
            let err = |e: cudarc::driver::DriverError| msg(format!("offload, block {j}: {e}"));
            let pinned = host
                .pinned
                .as_ref()
                .ok_or_else(|| msg("offload: block has no pinned weights"))?
                .as_slice()
                .map_err(err)?;
            let lens: Vec<(usize, usize)> = host
                .parts
                .iter()
                .filter_map(|p| match p {
                    Some(HostPart::Pinned { offset, len }) => Some((*offset, *len)),
                    _ => None,
                })
                .collect();
            let slot = &mut self.slots[s];
            if let Some(freed) = slot.freed.take() {
                self.copy.wait(&freed).map_err(err)?;
            }
            let fits = slot.parts.len() == lens.len()
                && slot
                    .parts
                    .iter()
                    .zip(&lens)
                    .all(|(p, (_, l))| p.len() == *l);
            if !fits {
                slot.parts.clear();
                for (_, len) in &lens {
                    // Written by the copy below before any kernel reads it.
                    let buf = unsafe { self.copy.alloc::<bf16>(*len) }.map_err(err)?;
                    slot.parts.push(Arc::new(buf));
                }
            }
            let start = timed(&self.copy)?;
            let mut bytes = 0u64;
            for (part, (offset, len)) in slot.parts.iter_mut().zip(&lens) {
                let dst = Arc::get_mut(part)
                    .ok_or_else(|| msg("offload: a slot buffer is still referenced"))?;
                // A plain slice of the pinned buffer: cudarc adds no
                // synchronization; the `ready` event is what orders it.
                self.copy
                    .memcpy_htod(&pinned[*offset..*offset + *len], dst)
                    .map_err(err)?;
                crate::wan::stats::record_h2d(*len / 2);
                bytes += *len as u64 * 2;
            }
            let end = timed(&self.copy)?;
            slot.ready = Some(self.copy.record_event(None).map_err(err)?);
            slot.block = Some(j);
            self.tick += 1;
            slot.last_use = self.tick;
            stats.copies += 1;
            stats.bytes += bytes;
            if let Some(old) = self.pending[s].replace(Pending {
                start,
                end,
                wanted: None,
            }) {
                self.done.push(old);
            }
            Ok(())
        }

        /// The compute stream waits for block `index`'s copy.
        pub(super) fn acquire(&mut self, index: usize) -> Result<usize> {
            let s = self
                .slots
                .iter()
                .position(|s| s.block == Some(index))
                .ok_or_else(|| msg(format!("offload: block {index} was never queued")))?;
            let err = |e: cudarc::driver::DriverError| msg(format!("offload, block {index}: {e}"));
            if let Some(p) = self.pending[s].as_mut() {
                if p.wanted.is_none() {
                    p.wanted = Some(timed(&self.compute)?);
                }
            }
            if let Some(ready) = self.slots[s].ready.as_ref() {
                self.compute.wait(ready).map_err(err)?;
            }
            self.tick += 1;
            let slot = &mut self.slots[s];
            slot.in_use = true;
            slot.last_use = self.tick;
            Ok(s)
        }

        pub(super) fn weights(&self, s: usize) -> Vec<Arc<CudaSlice<bf16>>> {
            self.slots[s].parts.clone()
        }

        /// The block's kernels are queued: record when they are done.
        pub(super) fn release(&mut self, s: usize) -> Result<()> {
            let slot = &mut self.slots[s];
            slot.in_use = false;
            slot.freed = Some(
                self.compute
                    .record_event(None)
                    .map_err(|e| msg(format!("offload release: {e}")))?,
            );
            Ok(())
        }

        /// Fold finished copies into `stats`. `wait` also waits for the ones
        /// still running (a report).
        pub(super) fn resolve_timings(&mut self, stats: &mut OffloadStats, wait: bool) {
            if wait {
                for p in self.pending.iter_mut() {
                    if p.as_ref().is_some_and(|p| p.wanted.is_some()) {
                        self.done.extend(p.take());
                    }
                }
            }
            let mut keep = Vec::new();
            for p in self.done.drain(..) {
                let ready =
                    p.end.is_complete() && p.wanted.as_ref().is_none_or(CudaEvent::is_complete);
                if !ready && !wait {
                    keep.push(p);
                    continue;
                }
                let Ok(copy) = p.start.elapsed_ms(&p.end) else {
                    continue;
                };
                stats.timed += 1;
                stats.copy_ms += f64::from(copy);
                if let Some(wanted) = &p.wanted {
                    // Positive: the copy ended after the block was wanted.
                    if let Ok(stall) = wanted.elapsed_ms(&p.end) {
                        let stall = f64::from(stall.max(0.0)).min(f64::from(copy));
                        stats.exposed_ms += stall;
                        stats.worst_exposed_ms = stats.worst_exposed_ms.max(stall);
                    }
                }
            }
            self.done = keep;
        }
    }

    impl Drop for DeviceRing {
        fn drop(&mut self) {
            // Slot buffers are freed on the copy stream; nothing may still read them.
            let _ = self.compute.synchronize();
            let _ = self.copy.synchronize();
        }
    }
}

/// Device memory of one pipeline stage, from the allocator pool's high-water
/// marks (`torch.cuda.max_memory_allocated` / `_reserved`).
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseMemory {
    pub phase: &'static str,
    /// Highest bytes in live allocations during the phase.
    pub peak_used: u64,
    /// Highest bytes the pool held from the driver during the phase.
    pub peak_reserved: u64,
    /// Live bytes when the phase ended (after its frees).
    pub end_used: u64,
}

/// Per-stage peaks: [`Self::mark`] closes the current stage and opens the
/// next. Without a device (CPU runs) every call records nothing.
#[derive(Debug, Default)]
pub struct MemoryLog {
    pub model: &'static str,
    pub phases: Vec<PhaseMemory>,
}

impl MemoryLog {
    pub fn start(model: &'static str) -> Self {
        super::device::reset_pool_peaks();
        Self {
            model,
            phases: Vec::new(),
        }
    }

    /// Close `phase`: wait for its kernels, record the pool's peaks, log
    /// them, and restart the marks for the next phase.
    pub fn mark(&mut self, phase: &'static str) -> Result<()> {
        super::device::synchronize().map_err(|e| msg(e.to_string()))?;
        let Some(u) = super::device::pool_usage() else {
            return Ok(());
        };
        super::log::info(format_args!(
            "{} memory {phase}: peak {:.2} GiB used, {:.2} GiB reserved; {:.2} GiB live after",
            self.model,
            u.used_high as f64 / GIB,
            u.reserved_high as f64 / GIB,
            u.used as f64 / GIB
        ));
        self.phases.push(PhaseMemory {
            phase,
            peak_used: u.used_high,
            peak_reserved: u.reserved_high,
            end_used: u.used,
        });
        super::device::reset_pool_peaks();
        Ok(())
    }

    pub fn peak_used(&self) -> u64 {
        self.phases.iter().map(|p| p.peak_used).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wan::tensor::CudaTensor;

    #[derive(Clone)]
    struct Toy {
        a: Linear,
        b: Linear,
        norm: CudaTensor,
    }

    impl OffloadBlock for Toy {
        fn for_each_linear_mut(
            &mut self,
            f: &mut dyn FnMut(&mut Linear) -> Result<()>,
        ) -> Result<()> {
            f(&mut self.a)?;
            f(&mut self.b)
        }
    }

    impl Toy {
        fn new(seed: f32) -> Self {
            let w = |n: usize, k: f32| (0..n).map(|i| (i as f32 * 0.37 + k).sin()).collect();
            Self {
                a: Linear::from_tensors(
                    CudaTensor::from_vec(w(12, seed), vec![4, 3]).unwrap(),
                    Some(CudaTensor::from_vec(w(4, seed + 1.0), vec![4]).unwrap()),
                )
                .unwrap(),
                b: Linear::from_tensors(
                    CudaTensor::from_vec(w(8, seed + 2.0), vec![2, 4]).unwrap(),
                    None,
                )
                .unwrap(),
                norm: CudaTensor::from_vec(w(4, seed + 3.0), vec![4]).unwrap(),
            }
        }

        fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
            let h = self.a.forward(x)?.mul(&self.norm)?;
            self.b.forward(&h)
        }
    }

    #[test]
    fn policy_parses_and_resolves() {
        assert_eq!(DitOffload::parse("auto").unwrap(), DitOffload::Auto);
        assert_eq!(DitOffload::parse("Streamed").unwrap(), DitOffload::Streamed);
        assert_eq!(DitOffload::parse("resident").unwrap(), DitOffload::Resident);
        assert!(DitOffload::parse("sometimes").is_err());
        let gib = 1u64 << 30;
        // Auto: resident when the card has room, streamed when it does not.
        assert_eq!(
            DitOffload::Auto.resolve(60 * gib, Some(90 * gib)),
            Residency::Resident
        );
        assert_eq!(
            DitOffload::Auto.resolve(60 * gib, Some(30 * gib)),
            Residency::Streamed
        );
        // No device: nothing to save.
        assert_eq!(
            DitOffload::Auto.resolve(60 * gib, None),
            Residency::Resident
        );
        // Overrides ignore the memory.
        assert_eq!(
            DitOffload::Streamed.resolve(1, Some(90 * gib)),
            Residency::Streamed
        );
        assert_eq!(
            DitOffload::Resident.resolve(60 * gib, Some(gib)),
            Residency::Resident
        );
        assert_eq!(
            DitOffload::from_env_or(Some(DitOffload::Streamed)).unwrap(),
            DitOffload::Streamed
        );
    }

    #[test]
    fn streamed_blocks_are_bit_identical_to_resident() {
        let blocks: Vec<Toy> = (0..5).map(|i| Toy::new(i as f32)).collect();
        let resident = BlockWeights::resident("toy", blocks.clone());
        let mut streamed = BlockWeights::new("toy", Residency::Streamed).with_lookahead(2);
        for b in blocks {
            streamed.push(b).unwrap();
        }
        // The skeleton holds no linear weight; the norms stay.
        assert_eq!(streamed.skeleton(0).a.weight.numel(), 0);
        assert_eq!(streamed.skeleton(0).norm.numel(), 4);
        assert_eq!(streamed.host_bytes(), 5 * (12 + 8) * 4);
        let x0 = CudaTensor::from_vec((0..6).map(|i| i as f32 * 0.1 - 0.2).collect(), vec![2, 3])
            .unwrap();
        // Two passes, with an early exit in the first (a cache skip), and
        // an out-of-order access.
        let run = |w: &BlockWeights<Toy>, order: &[usize]| -> Vec<f32> {
            let mut outs = Vec::new();
            for &i in order {
                let y = w.with(i, |b| b.forward(&x0)).unwrap();
                outs.extend(y.host_cow().unwrap().iter().copied());
            }
            outs
        };
        let order = [0, 1, 0, 1, 2, 3, 4, 3];
        let a = run(&resident, &order);
        let b = run(&streamed, &order);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        // The skeleton is left untouched by a forward.
        assert_eq!(streamed.skeleton(1).b.weight.numel(), 0);
        assert!(streamed.with(9, |_| Ok(())).is_err());
    }

    #[test]
    fn stats_report_hidden_fraction_and_throughput() {
        let s = OffloadStats {
            copies: 10,
            bytes: 10_000_000_000,
            timed: 10,
            copy_ms: 400.0,
            exposed_ms: 100.0,
            worst_exposed_ms: 30.0,
        };
        assert!((s.hidden_fraction() - 0.75).abs() < 1e-12);
        assert!((s.throughput_gbs() - 25.0).abs() < 1e-9);
        assert_eq!(OffloadStats::default().hidden_fraction(), 1.0);
    }
}
