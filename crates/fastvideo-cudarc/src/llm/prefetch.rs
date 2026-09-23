//! The streamed encoder's layer pipeline: while layer `i` computes, a worker
//! reads layer `i + 1` from the mapped shards into page-locked memory and
//! uploads it on a second stream.
//!
//! Without this the streamed forward is a strict sequence per layer — page the
//! weights in, convert them, copy them from pageable memory (which the driver
//! stages through its own bounce buffer), and only then launch — and the device
//! idles through the first three. Here the host side of a layer costs the
//! compute stream nothing unless the disk is slower than the GPU.
//!
//! Synchronization is explicit (this crate runs cudarc with event tracking off):
//!
//! * The worker finishes a layer's copies (`synchronize` on the copy stream)
//!   *before* it hands the layer over, so the compute stream never sees a
//!   half-written weight and needs no wait of its own.
//! * The weights are allocated on the copy stream, so they are freed on it.
//!   Before a used layer is dropped, the copy stream is made to wait for the
//!   compute stream's work up to that point; the stream-ordered free then
//!   cannot hand the memory to the next upload while kernels still read it.
//!
//! Device memory holds up to three layers (computing, staged, in the channel)
//! instead of one; the host holds one layer of pinned memory. The numbers are
//! those of the plain streamed path bit for bit: same bf16 conversion
//! ([`fill_bf16`]), same norm path, same layer assembly.
//!
//! `FASTVIDEO_LLM_PREFETCH=0` turns it off.

use super::{
    linear_specs, max_layer_linear_elems, norm_from, norm_specs, DecoderConfig, Layer, LayerSource,
};
use crate::wan::nn::Linear;
use crate::wan::stats;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{fill_bf16, WeightMap};
use cudarc::driver::{CudaSlice, CudaStream, PinnedHostSlice};
use fastvideo_loader::LazyStore;
use half::bf16;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn enabled() -> bool {
    std::env::var("FASTVIDEO_LLM_PREFETCH").map_or(true, |v| v != "0")
        && crate::wan::nn::bf16_linears_active()
}

/// One layer, uploaded and complete on the device.
struct StagedLayer {
    index: usize,
    /// In [`linear_specs`] order.
    linears: Vec<CudaSlice<bf16>>,
    /// In [`norm_specs`] order: host values, turned into tensors by the compute
    /// thread exactly as the unprefetched path does.
    norms: Vec<Vec<f32>>,
    /// Host time spent reading and converting / waiting for the copies.
    fill: Duration,
    copy: Duration,
}

/// What a prefetched run needs before it starts: the copy stream and the
/// pinned buffer. `None` (with a log line) when either cannot be had, and the
/// caller streams the plain way.
pub(super) struct Stage<'a> {
    lazy: &'a LazyStore,
    copy: Arc<CudaStream>,
    compute: Arc<CudaStream>,
    pinned: PinnedHostSlice<bf16>,
}

impl<'a> Stage<'a> {
    pub(super) fn new(lazy: &'a LazyStore, cfg: &DecoderConfig) -> Option<Self> {
        if !enabled() {
            return None;
        }
        let dev = crate::wan::device::global_device()?;
        let elems = max_layer_linear_elems(cfg);
        let made = dev.ctx.new_stream().and_then(|copy| {
            // Every element is written before it is read.
            let pinned = unsafe { dev.ctx.alloc_pinned::<bf16>(elems) }?;
            Ok((copy, pinned))
        });
        match made {
            Ok((copy, pinned)) => Some(Self {
                lazy,
                copy,
                compute: dev.stream.clone(),
                pinned,
            }),
            Err(e) => {
                crate::wan::log::info(format_args!(
                    "llm prefetch off: {e} ({} MiB of pinned memory asked for)",
                    (elems * 2) >> 20
                ));
                None
            }
        }
    }

    /// Run `f` over a source that serves layers `0..last` in order.
    pub(super) fn run<R>(
        self,
        map: &WeightMap,
        cfg: &DecoderConfig,
        last: usize,
        f: impl FnOnce(&mut Prefetched<'_>) -> Result<R>,
    ) -> Result<R> {
        let Stage {
            lazy,
            copy,
            compute,
            pinned,
        } = self;
        std::thread::scope(|scope| {
            // One layer waits in the channel while the worker stages the next.
            let (tx, rx) = sync_channel::<Result<StagedLayer>>(1);
            let worker_copy = copy.clone();
            scope.spawn(move || worker(lazy, cfg, last, &worker_copy, pinned, &tx));
            let mut source = Prefetched {
                map,
                cfg,
                rx,
                copy,
                compute,
                next: 0,
                starved: Duration::ZERO,
                fill: Duration::ZERO,
                upload: Duration::ZERO,
            };
            let out = f(&mut source);
            if out.is_ok() {
                source.report();
            }
            // Dropping the receiver is what stops a worker that is still ahead.
            drop(source);
            out
        })
    }
}

fn worker(
    lazy: &LazyStore,
    cfg: &DecoderConfig,
    last: usize,
    copy: &Arc<CudaStream>,
    mut pinned: PinnedHostSlice<bf16>,
    tx: &SyncSender<Result<StagedLayer>>,
) {
    for index in 0..last {
        let staged = stage_layer(lazy, cfg, index, copy, &mut pinned);
        let failed = staged.is_err();
        if tx.send(staged).is_err() || failed {
            break;
        }
    }
    // The buffer must not be unpinned under a copy that is still running.
    let _ = copy.synchronize();
}

fn stage_layer(
    lazy: &LazyStore,
    cfg: &DecoderConfig,
    index: usize,
    copy: &Arc<CudaStream>,
    pinned: &mut PinnedHostSlice<bf16>,
) -> Result<StagedLayer> {
    let err = |e: cudarc::driver::DriverError| msg(format!("llm prefetch, layer {index}: {e}"));
    let p = format!("{}.{index}", cfg.layer_prefix);
    // The previous layer's copies were synchronized before it was sent, so the
    // buffer is free to overwrite.
    let host = pinned.as_mut_slice().map_err(err)?;
    let mut fill = Duration::ZERO;
    let mut linears = Vec::with_capacity(7);
    let mut at = 0;
    for (name, i, o) in linear_specs(cfg, index) {
        if cfg.attention_k_eq_v && name == "self_attn.v_proj" {
            continue;
        }
        let key = format!("{p}.{name}.weight");
        let shape = lazy
            .shape(&key)
            .ok_or_else(|| msg(format!("key {key}: not in the checkpoint")))?;
        if shape != [o, i] {
            return Err(msg(format!(
                "key {key}: shape {shape:?} != expected {:?}",
                [o, i]
            )));
        }
        let region = &mut host[at..at + i * o];
        at += i * o;
        let t = Instant::now();
        fill_bf16(lazy, &key, region)?;
        fill += t.elapsed();
        // Queued, not waited for: the next tensor is read while this one moves.
        // A plain slice of the pinned buffer, so cudarc adds no synchronization
        // of its own; the `synchronize` below is what makes it safe.
        let mut dst = unsafe { copy.alloc::<bf16>(i * o) }.map_err(err)?;
        copy.memcpy_htod(&*region, &mut dst).map_err(err)?;
        stats::record_h2d(i * o / 2);
        linears.push(dst);
    }
    let t = Instant::now();
    let mut norms = Vec::new();
    for (name, width) in norm_specs(cfg, index) {
        let key = format!("{p}.{name}.weight");
        let (shape, values) = lazy.to_f32(&key).map_err(|e| msg(e.to_string()))?;
        if shape != [width] {
            return Err(msg(format!(
                "key {key}: shape {shape:?} != expected {:?}",
                [width]
            )));
        }
        norms.push(values);
    }
    fill += t.elapsed();
    let t = Instant::now();
    copy.synchronize().map_err(err)?;
    Ok(StagedLayer {
        index,
        linears,
        norms,
        fill,
        copy: t.elapsed(),
    })
}

pub(super) struct Prefetched<'a> {
    map: &'a WeightMap,
    cfg: &'a DecoderConfig,
    rx: Receiver<Result<StagedLayer>>,
    copy: Arc<CudaStream>,
    compute: Arc<CudaStream>,
    next: usize,
    /// How long the compute thread waited for a layer that was not ready.
    starved: Duration,
    fill: Duration,
    upload: Duration,
}

impl Prefetched<'_> {
    fn report(&self) {
        crate::wan::log::info(format_args!(
            "llm prefetch: {} layers, host read+convert {:.2}s, upload wait {:.2}s, compute starved {:.2}s",
            self.next,
            self.fill.as_secs_f64(),
            self.upload.as_secs_f64(),
            self.starved.as_secs_f64(),
        ));
    }
}

impl LayerSource for Prefetched<'_> {
    fn with_layer<R>(&mut self, index: usize, f: impl FnOnce(&Layer) -> Result<R>) -> Result<R> {
        if index != self.next {
            return Err(msg(format!(
                "llm prefetch: layer {index} asked for, {} staged",
                self.next
            )));
        }
        let t = Instant::now();
        let staged = self
            .rx
            .recv()
            .map_err(|_| msg("llm prefetch: the staging thread stopped"))??;
        self.starved += t.elapsed();
        if staged.index != index {
            return Err(msg(format!(
                "llm prefetch: got layer {}, wanted {index}",
                staged.index
            )));
        }
        self.next += 1;
        self.fill += staged.fill;
        self.upload += staged.copy;

        let cfg = self.cfg;
        let mut linears = staged.linears.into_iter();
        let mut norms: Vec<(&str, Option<Vec<f32>>)> = norm_specs(cfg, index)
            .into_iter()
            .map(|(n, _)| n)
            .zip(staged.norms.into_iter().map(Some))
            .collect();
        let layer = Layer::assemble(
            cfg,
            index,
            &mut |name, i, o| {
                let w = linears
                    .next()
                    .ok_or_else(|| msg(format!("llm prefetch: no weight staged for {name}")))?;
                Linear::from_device_bf16(w, i, o)
            },
            // `assemble` asks for norms in its own order; serve them by name.
            &mut |name, width| {
                let values = norms
                    .iter_mut()
                    .find(|(n, _)| *n == name)
                    .and_then(|(_, v)| v.take())
                    .ok_or_else(|| msg(format!("llm prefetch: no norm staged for {name}")))?;
                norm_from(CudaTensor::from_vec(values, vec![width])?, cfg.norm_offset)
            },
        );
        let out = layer.and_then(|layer| {
            let out = f(&layer);
            // The layer's memory belongs to the copy stream. Hold its free back
            // until the kernels queued above have run.
            let done = self
                .compute
                .record_event(None)
                .and_then(|e| self.copy.wait(&e));
            drop(layer);
            done.map_err(|e| msg(format!("llm prefetch, layer {index}: {e}")))?;
            out
        });
        out
    }

    fn final_norm(&mut self) -> Result<CudaTensor> {
        super::norm_weight(
            self.map,
            &self.cfg.final_norm_key,
            self.cfg.hidden,
            self.cfg.norm_offset,
        )
    }
}
