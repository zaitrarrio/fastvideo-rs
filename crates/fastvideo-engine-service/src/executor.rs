//! One executor thread per `EngineBackend` (one per GPU). Never async: it
//! owns the backend (and so the CUDA context), pulls work from the
//! scheduler, and reports through channels that never block it (design §3.6,
//! §5.1).
//!
//! Loop: warm-load the resident models, then repeatedly
//! 1. release causal sessions their owners closed (`causal_close`);
//! 2. take the next dispatch (a job, or one block of the leased causal
//!    session); with none, one background warm-up run when a resident model
//!    still wants one (fast boot B), else park on the work condvar;
//! 3. run it.
//!
//! Background warm-up never shares the GPU with a job: it runs on this
//! thread, only while nothing is queued and no session holds the executor,
//! and any arriving job or session trips its cancel token
//! ([`crate::service::State::yield_warmups`]), so it stops at its next step
//! and the job runs; the cancelled run is retried once the executor is idle
//! again.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_protocol::{ApiError, ModelId};

use crate::backend::{ClipSink, CollectSink, EngineBackend, LoadEvent, NullSink, SessionId};
use crate::cancel::{is_cancel, OutputMode, StepControl, StepEvent};
use crate::cancel::CancelToken;
use crate::pool::{Residency, Warmup};
use crate::scheduler::{Dispatch, QueueItem};
use crate::service::{EngineEvent, Shared};
use crate::stream::causal::CausalShared;

pub(crate) fn run(sh: Arc<Shared>, idx: usize, mut backend: Box<dyn EngineBackend>) {
    let warm: Vec<ModelId> = {
        let st = sh.lock();
        st.pool
            .entries()
            .filter(|e| e.executor == idx && e.warm)
            .map(|e| e.model.clone())
            .collect()
    };
    for m in &warm {
        // A failure is recorded in the pool (readiness `Failed`).
        let _ = load(&sh, idx, backend.as_mut(), m, true);
    }
    let mut warmups = Warmups::new(&sh, idx, backend.as_ref(), &warm);
    loop {
        let (closing, work, warm_run) = {
            let mut st = sh.lock();
            loop {
                if st.shutdown {
                    break (std::mem::take(&mut st.to_close[idx]), None, None);
                }
                let closing = std::mem::take(&mut st.to_close[idx]);
                if !closing.is_empty() {
                    break (closing, None, None);
                }
                if let Some(d) = st.sched.next_for(idx) {
                    if let Dispatch::Job(it) = &d {
                        if let Some(e) = st.jobs.get_mut(&it.job) {
                            e.running = true;
                            e.last_pos = None;
                        }
                        st.publish_positions();
                        sh.changed(&st);
                    }
                    break (Vec::new(), Some(d), None);
                }
                // Idle: a background warm-up run, unless a session holds
                // this executor or the engine is draining.
                if !st.draining && st.sched.exec(idx).session.is_none() {
                    if let Some((model, name)) = warmups.next(&st.pool, idx) {
                        let token = CancelToken::new();
                        st.warmup_cancel[idx] = Some(token.clone());
                        st.pool.set_warmup(idx, &model, Warmup::Running);
                        sh.changed(&st);
                        break (Vec::new(), None, Some((model, name, token)));
                    }
                }
                st = sh.work_cv.wait(st).unwrap_or_else(|p| p.into_inner());
            }
        };
        for s in closing {
            backend.causal_close(s);
        }
        match (work, warm_run) {
            (Some(Dispatch::Job(it)), _) => run_job(&sh, idx, backend.as_mut(), it),
            (Some(Dispatch::Causal(sid)), _) => causal_turn(&sh, backend.as_mut(), sid),
            (None, Some((model, name, token))) => {
                warmups.run(&sh, idx, backend.as_mut(), &model, &name, &token)
            }
            (None, None) => {
                if sh.lock().shutdown {
                    break;
                }
            }
        }
    }
    let mut st = sh.lock();
    st.alive -= 1;
    sh.changed(&st);
}

/// The background warm-up runs left on this executor (fast boot B).
struct Warmups {
    /// Per model, in declaration order: the runs not done yet.
    left: Vec<(ModelId, std::collections::VecDeque<String>)>,
    /// Per model: when its first run started, time on the GPU, runs done.
    started: std::collections::BTreeMap<ModelId, (Instant, f64, Vec<String>, u32)>,
}

impl Warmups {
    fn new(sh: &Shared, idx: usize, backend: &dyn EngineBackend, models: &[ModelId]) -> Self {
        let mut st = sh.lock();
        let mut left = Vec::new();
        for m in models {
            if st.pool.state(idx, m) != Some(&Residency::Resident) {
                continue;
            }
            let runs = backend.warmup_pending(m);
            if !runs.is_empty() {
                st.pool.set_warmup(idx, m, Warmup::Pending);
                left.push((m.clone(), runs.into_iter().collect()));
            }
        }
        sh.changed(&st);
        Self {
            left,
            started: Default::default(),
        }
    }

    /// The next run: the first model, still resident, with runs left.
    fn next(&mut self, pool: &crate::pool::ModelPool, idx: usize) -> Option<(ModelId, String)> {
        // A model that left the GPU (swap mode) drops its warm-up.
        self.left
            .retain(|(m, runs)| !runs.is_empty() && pool.state(idx, m) == Some(&Residency::Resident));
        self.left
            .first()
            .and_then(|(m, runs)| runs.front().map(|r| (m.clone(), r.clone())))
    }

    fn run(
        &mut self,
        sh: &Shared,
        idx: usize,
        backend: &mut dyn EngineBackend,
        model: &ModelId,
        name: &str,
        token: &CancelToken,
    ) {
        let entry = self.started.entry(model.clone()).or_insert_with(|| {
            let runs: Vec<String> = self
                .left
                .iter()
                .find(|(m, _)| m == model)
                .map(|(_, r)| r.iter().cloned().collect())
                .unwrap_or_default();
            tracing::info!(model = %model, runs = %runs.join(", "), "warmup started (background)");
            (Instant::now(), 0.0, Vec::new(), 0)
        });
        let t = Instant::now();
        let r = backend.warmup_run(model, name, token);
        let took = t.elapsed().as_secs_f64();
        entry.1 += took;
        let mut st = sh.lock();
        st.warmup_cancel[idx] = None;
        let runs = self.left.iter_mut().find(|(m, _)| m == model).map(|(_, r)| r);
        let state = match r {
            Ok(what) => {
                entry.2.push(what);
                if let Some(runs) = runs {
                    runs.pop_front();
                    if runs.is_empty() {
                        tracing::info!(
                            model = %model,
                            seconds = entry.1,
                            elapsed_s = entry.0.elapsed().as_secs_f64(),
                            yielded = entry.3,
                            runs = %entry.2.join(", "),
                            "warmup done (background)"
                        );
                        Warmup::Done
                    } else {
                        Warmup::Pending
                    }
                } else {
                    Warmup::Done
                }
            }
            Err(e) if is_cancel(&e) => {
                entry.3 += 1;
                tracing::info!(model = %model, run = name, after_s = took, "warmup yielded to a job (background)");
                Warmup::Pending
            }
            Err(e) => {
                tracing::warn!(model = %model, run = name, error = %e, "warmup failed; serving without it");
                if let Some(runs) = runs {
                    runs.clear();
                }
                Warmup::Failed
            }
        };
        st.pool.set_warmup(idx, model, state);
        sh.changed(&st);
    }
}

/// Loads `model` on this executor, tracking residency in the pool.
/// `boot`: a resident model loaded at start. Its background warm-up runs
/// are marked pending in the same critical section that makes it resident,
/// so nobody can see it ready with its warm-up not yet known (the readiness
/// probe and `warmup()` read the pool under the same lock).
fn load(sh: &Shared, idx: usize, backend: &mut dyn EngineBackend, model: &ModelId, boot: bool) -> Result<(), ApiError> {
    {
        let mut st = sh.lock();
        st.pool.set(
            idx,
            model,
            Residency::Loading {
                stage: None,
                done: 0,
                total: 0,
            },
        );
        sh.changed(&st);
    }
    let mut stage: Option<String> = None;
    let mut obs = |ev: LoadEvent| {
        let mut st = sh.lock();
        let (done, total) = match (ev, st.pool.state(idx, model)) {
            (LoadEvent::Stage(s), Some(Residency::Loading { done, total, .. })) => {
                stage = Some(s.to_owned());
                (*done, *total)
            }
            (LoadEvent::Progress { done, total }, _) => (done, total),
            _ => (0, 0),
        };
        st.pool.set(
            idx,
            model,
            Residency::Loading {
                stage: stage.clone(),
                done,
                total,
            },
        );
        sh.changed(&st);
    };
    let r = backend.load(model, &mut obs);
    let mut st = sh.lock();
    match &r {
        Ok(()) => {
            st.pool.set(idx, model, Residency::Resident);
            st.sched.exec_mut(idx).resident.insert(model.clone());
            if boot && !backend.warmup_pending(model).is_empty() {
                st.pool.set_warmup(idx, model, Warmup::Pending);
            }
        }
        Err(e) => {
            tracing::error!(executor = idx, model = %model, error = %e, "model load failed");
            st.pool.set(idx, model, Residency::Failed(e.clone()));
            st.sched.exec_mut(idx).resident.remove(model);
        }
    }
    sh.changed(&st);
    r
}

/// Swap mode: evicts every other resident model, then loads `model`.
fn swap_in(sh: &Shared, idx: usize, backend: &mut dyn EngineBackend, model: &ModelId) -> Result<(), ApiError> {
    let others: Vec<ModelId> = {
        let st = sh.lock();
        st.sched
            .exec(idx)
            .resident
            .iter()
            .filter(|m| *m != model)
            .cloned()
            .collect()
    };
    for m in others {
        backend.unload(&m);
        let mut st = sh.lock();
        st.sched.exec_mut(idx).resident.remove(&m);
        st.pool.set(idx, &m, Residency::Unloaded);
        sh.changed(&st);
    }
    load(sh, idx, backend, model, false)
}

/// Marks per traced run: stage boundaries, every denoise step, a few more.
const TIMELINE_MARKS: usize = 256;

fn run_job(sh: &Arc<Shared>, idx: usize, backend: &mut dyn EngineBackend, it: QueueItem) {
    let Some((tx, cancel, job, mode, trace, queued_ns)) = ({
        let st = sh.lock();
        st.jobs
            .get(&it.job)
            .map(|e| (e.tx.clone(), e.cancel.clone(), e.job.clone(), e.mode.clone(), e.trace, e.queued_ns))
    }) else {
        sh.lock().sched.finish(idx);
        return;
    };
    // docs/serve/tracing.md: queue wait (submit to dequeue) and the run.
    let t_run = trace.map(|t| {
        t.span_since(fastvideo_trace::Comp::Queue, "wait", queued_ns, idx as i64);
        fastvideo_trace::now_ns()
    });
    let _ = tx.send(EngineEvent::Started);
    let t0 = Instant::now();
    let result = (|| {
        cancel.check()?;
        let resident = sh.lock().sched.exec(idx).resident.contains(&job.model);
        if !resident {
            let _ = tx.send(EngineEvent::Stage { name: "load" });
            swap_in(sh, idx, backend, &job.model)?;
        }
        let etx = tx.clone();
        let mut ctl = StepControl::new(cancel.clone(), mode.clone(), move |ev| {
            let _ = etx.send(match ev {
                StepEvent::Stage(name) => EngineEvent::Stage { name },
                StepEvent::Progress { step, total } => EngineEvent::Progress { step, total },
                StepEvent::Log(l) => EngineEvent::Log(l),
            });
        });
        if let Some(t) = trace {
            // Marks storage is allocated here, before the run.
            let pool = backend.marks(TIMELINE_MARKS).unwrap_or_else(|| Box::new(fastvideo_trace::HostMarks));
            ctl = ctl.with_timeline(fastvideo_trace::Timeline::new(t, pool, TIMELINE_MARKS));
        }
        let mut collect = CollectSink::default();
        let mut null = NullSink;
        let sink: &mut dyn ClipSink = match mode {
            OutputMode::Frames => &mut collect,
            OutputMode::File { .. } => &mut null,
        };
        let generated = backend.generate(&job, sink, &ctl);
        // Device marks resolve on the trace drain thread, not here.
        ctl.finish_timeline();
        let mut out = generated?;
        if mode == OutputMode::Frames {
            if out.frames.is_none() {
                out.frames = Some(std::mem::take(&mut collect.frames));
            }
            if out.audio.is_none() {
                out.audio = collect.joined_audio();
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        if out.metrics.inference_s.is_none() {
            out.metrics.inference_s = Some(wall);
        }
        if out.metrics.build_rtf.is_none() && job.duration_s() > 0.0 {
            out.metrics.build_rtf = Some(out.metrics.inference_s.unwrap_or(wall) / job.duration_s());
        }
        Ok(out)
    })();
    if let (Some(t), Some(s)) = (trace, t_run) {
        t.span_since(fastvideo_trace::Comp::Engine, "run", s, i64::from(result.is_ok()));
    }
    let ev = match result {
        Ok(out) => EngineEvent::Finished(out),
        Err(e) if is_cancel(&e) => EngineEvent::Cancelled,
        Err(e) => EngineEvent::Failed(e),
    };
    let mut st = sh.lock();
    st.sched.finish(idx);
    st.jobs.remove(&it.job);
    let _ = tx.send(ev);
    sh.changed(&st);
}

/// One causal turn: open on first use, then one block (or a short park while
/// paused).
fn causal_turn(sh: &Arc<Shared>, backend: &mut dyn EngineBackend, sid: SessionId) {
    let cs: Option<Arc<CausalShared>> = sh.lock().causal.get(&sid).cloned();
    let Some(cs) = cs else {
        // Lease without a session object (should not happen): release it.
        let mut st = sh.lock();
        st.sched.close_session(sid);
        sh.changed(&st);
        return;
    };
    if cs.is_closed() {
        sh.close_causal_session(sid);
        return;
    }
    if !cs.is_opened() {
        let spec = cs.backend_spec();
        match backend.causal_open(sid, &spec) {
            Ok(()) => cs.set_opened(),
            Err(e) => {
                cs.fail(e);
                sh.close_causal_session(sid);
                return;
            }
        }
    }
    let Some(input) = cs.next_input(Duration::from_millis(50)) else {
        return;
    };
    let ctl = StepControl::detached(cs.cancel.clone(), OutputMode::Frames);
    let mut sink = CollectSink::default();
    let t0 = Instant::now();
    match backend.causal_block(sid, &input, &mut sink, &ctl) {
        Ok(mut stats) => {
            if stats.block_ms == 0.0 {
                stats.block_ms = t0.elapsed().as_secs_f64() * 1e3;
            }
            let audio = sink.joined_audio();
            cs.deliver(input, sink.frames, audio, stats);
        }
        Err(e) if is_cancel(&e) => {}
        Err(e) => {
            cs.fail(e);
            sh.close_causal_session(sid);
        }
    }
}
