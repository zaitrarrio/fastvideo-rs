//! The H3 denoise loop's step timing (docs/perf/h3-denoise-graphs.md).
//!
//! **Lazy step clock.** The loop used to synchronize the device after every
//! step only to time it. That drained the queue at each step boundary: the
//! next step's host preparation (its bf16 AdaLN table, ~50 ms of host work
//! before its first block) then ran with the GPU idle. [`StepClock`] records a
//! CUDA event after each step instead and reads the events lazily: a step's
//! log line is printed once its event has completed (polled, never waited
//! for, at the next step boundary), and the device is synchronized once,
//! after the last step. Per-step seconds are then GPU time between the
//! events. No kernel, buffer or launch order changes, so the output bits do
//! not either.
//!
//! `FASTVIDEO_H3_STEP_SYNC=1` restores the per-step synchronize (and
//! host-clock step seconds). `FASTVIDEO_GPU_TRACE` implies it: the CUPTI
//! trace window of one step needs the queue drained around that step.

use std::time::Instant;

use crate::wan::pipeline::{PipelineError, Result};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// `FASTVIDEO_H3_STEP_SYNC=1` (default off), or a GPU trace: synchronize
/// after every denoise step.
pub fn step_sync() -> bool {
    crate::wan::envflag::bool_flag("FASTVIDEO_H3_STEP_SYNC", false)
        || crate::wan::gpu_trace::enabled()
        || !cfg!(feature = "cuda")
}

/// Per-step timing of one denoise ([`step_sync`] or lazy CUDA events).
pub struct StepClock {
    steps: usize,
    sync: bool,
    last: Instant,
    step_s: Vec<f64>,
    /// Lazy mode: the start event, then one per queued step.
    #[cfg(feature = "cuda")]
    events: Vec<cudarc::driver::CudaEvent>,
}

impl StepClock {
    /// Before the first step is queued.
    pub fn start(steps: usize) -> Result<Self> {
        let sync = step_sync();
        let mut clock = Self {
            steps,
            sync,
            last: Instant::now(),
            step_s: Vec::with_capacity(steps),
            #[cfg(feature = "cuda")]
            events: Vec::with_capacity(steps + 1),
        };
        if sync {
            // FASTVIDEO_GPU_TRACE: one step's device activity (a no-op when off).
            crate::wan::gpu_trace::pass_begin("h3");
            crate::wan::gpu_trace::step_begin();
        } else {
            clock.record()?;
        }
        Ok(clock)
    }

    /// Whether this clock synchronizes per step.
    pub fn is_sync(&self) -> bool {
        self.sync
    }

    #[cfg(feature = "cuda")]
    fn record(&mut self) -> Result<()> {
        let dev = crate::wan::device::global_device()
            .ok_or_else(|| msg("h3 step clock: no CUDA device"))?;
        let event = dev
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| msg(format!("h3 step clock: {e}")))?;
        self.events.push(event);
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn record(&mut self) -> Result<()> {
        Ok(())
    }

    /// Step `step`'s work is queued (on this thread's device stream).
    pub fn step_queued(&mut self, step: usize) -> Result<()> {
        if self.sync {
            // The trace window closes before the step-timing sync.
            crate::wan::gpu_trace::step_before_sync();
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            let s = self.last.elapsed().as_secs_f64();
            self.step_s.push(s);
            crate::wan::gpu_trace::step_end(s);
            self.log(step, s);
            if step + 1 < self.steps {
                crate::wan::gpu_trace::step_begin();
            }
            self.last = Instant::now();
            return Ok(());
        }
        self.record()?;
        self.collect(false)
    }

    fn log(&self, step: usize, s: f64) {
        crate::wan::log::info(format_args!("h3 step {}/{}: {s:.1}s", step + 1, self.steps));
    }

    /// Lazy mode: time and log every step whose event has completed (all of
    /// them with `wait`, after a synchronize).
    #[cfg(feature = "cuda")]
    fn collect(&mut self, wait: bool) -> Result<()> {
        while self.step_s.len() + 1 < self.events.len() {
            let i = self.step_s.len();
            if !wait && !self.events[i + 1].is_complete() {
                break;
            }
            let ms = self.events[i]
                .elapsed_ms(&self.events[i + 1])
                .map_err(|e| msg(format!("h3 step clock: {e}")))?;
            let s = f64::from(ms) * 1e-3;
            self.step_s.push(s);
            self.log(i, s);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn collect(&mut self, _wait: bool) -> Result<()> {
        Ok(())
    }

    /// After the last step is queued: wait for the device (lazy mode) and
    /// return the per-step seconds.
    pub fn finish(mut self) -> Result<Vec<f64>> {
        if !self.sync {
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            self.collect(true)?;
        }
        Ok(self.step_s)
    }
}

// ---- FASTVIDEO_H3_GRAPHS: CUDA graphs of the denoise steps ----------------------

/// `FASTVIDEO_H3_GRAPHS=1` (default off): replay a CUDA graph of each denoise
/// step (forward + scheduler step) once a request shape has been seen twice.
pub fn graphs_enabled() -> bool {
    crate::wan::envflag::bool_flag("FASTVIDEO_H3_GRAPHS", false)
}

/// The VSA settings a captured step bakes in (k_vid per block, the fine
/// stage's kernel, the tile grouping).
#[derive(Debug, Clone, PartialEq)]
pub struct VsaKey {
    pub sparsity_bits: u64,
    pub schedule: fastvideo_models::techniques::VsaSchedule,
    pub fp8: bool,
    pub fp4: bool,
    pub group: usize,
    pub tile_size: usize,
}

/// What a captured step depends on beside its inputs' values: the packed
/// layout (canvas, frames, text length and tags), the attention route and the
/// dense FP8 attention switch. Two requests with an equal key run the same
/// kernels with the same arguments, so a graph of one replays for the other.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphKey {
    pub layout: fastvideo_models::h3::packing::H3PackedLayout,
    pub vsa: Option<VsaKey>,
    pub fp8_dense: bool,
}

#[cfg(feature = "cuda")]
pub use imp::H3Graphs;

#[cfg(feature = "cuda")]
mod imp {
    use fastvideo_models::h3::schedule::H3JointSchedule;

    use super::{GraphKey, Result};
    use crate::h3::pipeline::scheduler_step;
    use crate::h3::transformer::{AttnMode, DeviceLayout, H3Transformer};
    use crate::h3::vsa::H3Vsa;
    use crate::wan::graph::{assign, duplicate, Graph, GraphStream};
    use crate::wan::tensor::{CudaTensor, TensorError};

    /// Request shapes kept (each holds a graph per step and its slots). With
    /// four (2 x 480p + 2 x 768p V2-NVFP4, 32 graphs) the fourth shape's
    /// capture ran out of graph memory at step 2 on a 96 GB card with 45 GB
    /// in use (docs/perf/h3-denoise-graphs.md); three fitted.
    const MAX_KEYS: usize = 2;

    /// The persistent buffers a step graph reads and writes: the text
    /// conditioning and the latent rows, refilled in place per request; each
    /// step reads the rows and overwrites them with the next step's.
    struct Slots {
        text: CudaTensor,
        video: CudaTensor,
        audio: CudaTensor,
    }

    struct Entry {
        key: GraphKey,
        layout: DeviceLayout,
        vsa: Option<H3Vsa>,
        slots: Option<Slots>,
        graphs: Vec<Option<Graph>>,
        runs: usize,
    }

    /// The graph cache of one pipeline (see the module docs).
    pub struct H3Graphs {
        gs: GraphStream,
        entries: Vec<Entry>,
        /// Every step's bf16 AdaLN table, read by every captured graph.
        adaln: Option<Vec<CudaTensor>>,
        /// Why graphs were given up for this pipeline, if they were.
        failed: Option<String>,
    }

    fn tensor_err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(e.to_string())
    }

    fn same_slot(slot: &CudaTensor, src: &CudaTensor) -> bool {
        slot.shape == src.shape && slot.dtype() == src.dtype()
    }

    /// One step on the slots: the forward, the scheduler step, the new rows
    /// written back into the slots. What a step graph captures.
    fn step_body(
        model: &H3Transformer,
        layout: &DeviceLayout,
        mode: AttnMode<'_>,
        slots: &Slots,
        schedule: &H3JointSchedule,
        step: usize,
    ) -> crate::wan::tensor::Result<()> {
        let (v_video, v_audio) = model.forward(
            step,
            &slots.video,
            &slots.audio,
            &slots.text,
            layout,
            mode,
            None,
            None,
            None,
        )?;
        let video =
            scheduler_step(&schedule.video, step, &slots.video, &v_video).map_err(tensor_err)?;
        let audio =
            scheduler_step(&schedule.audio, step, &slots.audio, &v_audio).map_err(tensor_err)?;
        if !same_slot(&slots.video, &video) || !same_slot(&slots.audio, &audio) {
            return Err(tensor_err(format!(
                "h3 graphs: step {step} rows {:?}/{:?} do not fit the slots {:?}/{:?}",
                (&video.shape, video.dtype()),
                (&audio.shape, audio.dtype()),
                (&slots.video.shape, slots.video.dtype()),
                (&slots.audio.shape, slots.audio.dtype()),
            )));
        }
        assign(&slots.video, &video)?;
        assign(&slots.audio, &audio)
    }

    /// The graph memory pool's high-water mark, GiB.
    fn graph_mem_high_gib(gs: &GraphStream) -> f64 {
        use cudarc::driver::sys;
        let mut v: u64 = 0;
        let r = unsafe {
            sys::cuDeviceGetGraphMemAttribute(
                gs.device().ctx.cu_device(),
                sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH,
                (&mut v as *mut u64).cast(),
            )
        };
        if r == sys::CUresult::CUDA_SUCCESS {
            v as f64 / f64::from(1u32 << 30)
        } else {
            f64::NAN
        }
    }

    impl H3Graphs {
        pub fn new() -> Result<Self> {
            Ok(Self {
                gs: GraphStream::new()?,
                entries: Vec::new(),
                adaln: None,
                failed: None,
            })
        }

        fn give_up(&mut self, why: String) {
            crate::wan::log::info(format_args!("h3 graphs: {why}; eager from here"));
            self.failed.get_or_insert(why);
        }

        /// [`crate::h3::pipeline::denoise`] (no condition rows, no fresh
        /// noise) through the graph cache. The first request of a key runs
        /// that function as it is, which warms every lazy setup up; the
        /// second captures each step on the graph stream and launches it;
        /// later ones replay. `make` builds the layout and VSA of a key not
        /// yet cached. `observe` runs after each step is queued.
        #[allow(clippy::too_many_arguments)]
        pub fn denoise(
            &mut self,
            model: &H3Transformer,
            key: GraphKey,
            make: impl FnOnce() -> Result<(DeviceLayout, Option<H3Vsa>)>,
            text_refined: &CudaTensor,
            video_rows: CudaTensor,
            audio_rows: CudaTensor,
            schedule: &H3JointSchedule,
            seed: u64,
            observe: &mut dyn FnMut(usize, &CudaTensor, &CudaTensor) -> Result<()>,
        ) -> Result<(CudaTensor, CudaTensor)> {
            let steps = schedule.num_steps();
            let index = match self.entries.iter().position(|e| e.key == key) {
                Some(i) => i,
                None => {
                    if self.entries.len() >= MAX_KEYS {
                        // The oldest shape goes (its graphs and slots).
                        self.gs.synchronize()?;
                        self.entries.remove(0);
                    }
                    let (layout, vsa) = make()?;
                    self.entries.push(Entry {
                        key,
                        layout,
                        vsa,
                        slots: None,
                        graphs: (0..steps).map(|_| None).collect(),
                        runs: 0,
                    });
                    self.entries.len() - 1
                }
            };
            let first = self.entries[index].runs == 0;
            self.entries[index].runs += 1;
            if first || self.failed.is_some() {
                let e = &self.entries[index];
                let mode = e.vsa.as_ref().map_or(AttnMode::Dense, AttnMode::Vsa);
                return crate::h3::pipeline::denoise(
                    model,
                    &e.layout,
                    text_refined,
                    video_rows,
                    audio_rows,
                    schedule,
                    seed,
                    mode,
                    None,
                    None,
                    observe,
                );
            }
            if self.adaln.is_none() {
                self.adaln = Some(model.adaln_rows16_all()?);
            }
            let gs = self.gs.clone();
            let out = gs.scope(|| {
                self.run_graphs(
                    model,
                    index,
                    text_refined,
                    &video_rows,
                    &audio_rows,
                    schedule,
                    observe,
                )
            });
            if out.is_err() {
                // Whatever was queued finishes before the caller frees.
                let _ = self.gs.synchronize();
            }
            out
        }

        #[allow(clippy::too_many_arguments)]
        fn run_graphs(
            &mut self,
            model: &H3Transformer,
            index: usize,
            text_refined: &CudaTensor,
            video_rows: &CudaTensor,
            audio_rows: &CudaTensor,
            schedule: &H3JointSchedule,
            observe: &mut dyn FnMut(usize, &CudaTensor, &CudaTensor) -> Result<()>,
        ) -> Result<(CudaTensor, CudaTensor)> {
            let steps = schedule.num_steps();
            // Refill the slots (allocated the first time, outside any capture).
            {
                let e = &mut self.entries[index];
                match &e.slots {
                    Some(s)
                        if same_slot(&s.text, text_refined)
                            && same_slot(&s.video, video_rows)
                            && same_slot(&s.audio, audio_rows) =>
                    {
                        assign(&s.text, text_refined)?;
                        assign(&s.video, video_rows)?;
                        assign(&s.audio, audio_rows)?;
                    }
                    _ => {
                        e.graphs.iter_mut().for_each(|g| *g = None);
                        e.slots = Some(Slots {
                            text: duplicate(text_refined)?,
                            video: duplicate(video_rows)?,
                            audio: duplicate(audio_rows)?,
                        });
                    }
                }
            }
            let adaln = self.adaln.clone().expect("h3 graphs: adaln tables");
            let (mut replayed, mut captured, mut eager) = (0usize, 0usize, 0usize);
            for step in 0..steps {
                let replay = self.entries[index].graphs[step].is_some();
                if replay {
                    self.entries[index].graphs[step]
                        .as_ref()
                        .expect("h3 graphs: graph")
                        .launch()?;
                    replayed += 1;
                } else {
                    let capture = if self.failed.is_some() {
                        None
                    } else {
                        let e = &self.entries[index];
                        let slots = e.slots.as_ref().expect("h3 graphs: slots");
                        let mode = e.vsa.as_ref().map_or(AttnMode::Dense, AttnMode::Vsa);
                        Some(model.with_pinned_adaln(&adaln, || {
                            self.gs.capture(|| {
                                step_body(model, &e.layout, mode, slots, schedule, step)
                            })
                        }))
                    };
                    let run_eager = match capture {
                        Some(Ok(graph)) => {
                            // The launch is this step.
                            graph.launch()?;
                            crate::wan::log::info(format_args!(
                                "h3 graphs: step {step} captured: {:?}, graph pool high {:.2} GiB",
                                graph.stats,
                                graph_mem_high_gib(&self.gs)
                            ));
                            captured += 1;
                            if graph.escaped() == 0 {
                                self.entries[index].graphs[step] = Some(graph);
                            } else {
                                // Its memory is live and may be referenced:
                                // never replayed, never destroyed (wan::stream).
                                let why = format!(
                                    "step {step} graph: {} allocations outlive it",
                                    graph.escaped()
                                );
                                std::mem::forget(graph);
                                self.give_up(why);
                            }
                            false
                        }
                        Some(Err(err)) => {
                            // Nothing ran: the step runs eagerly on the same slots.
                            self.give_up(format!("step {step} capture: {err}"));
                            true
                        }
                        None => true,
                    };
                    if run_eager {
                        let e = &self.entries[index];
                        let slots = e.slots.as_ref().expect("h3 graphs: slots");
                        let mode = e.vsa.as_ref().map_or(AttnMode::Dense, AttnMode::Vsa);
                        model.with_pinned_adaln(&adaln, || {
                            step_body(model, &e.layout, mode, slots, schedule, step)
                        })?;
                        eager += 1;
                    }
                }
                let slots = self.entries[index]
                    .slots
                    .as_ref()
                    .expect("h3 graphs: slots");
                observe(step, &slots.video, &slots.audio)?;
            }
            crate::wan::log::info(format_args!(
                "h3 graphs: {steps} steps: {replayed} replayed, {captured} captured, {eager} eager"
            ));
            let slots = self.entries[index]
                .slots
                .as_ref()
                .expect("h3 graphs: slots");
            let out = (duplicate(&slots.video)?, duplicate(&slots.audio)?);
            self.gs.synchronize()?;
            Ok(out)
        }
    }

    impl Drop for H3Graphs {
        fn drop(&mut self) {
            let _ = self.gs.synchronize();
            self.entries.clear();
            self.gs.trim();
        }
    }
}
