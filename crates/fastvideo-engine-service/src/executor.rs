//! One executor thread per `EngineBackend` (one per GPU). Never async: it
//! owns the backend (and so the CUDA context), pulls work from the
//! scheduler, and reports through channels that never block it (design §3.6,
//! §5.1).
//!
//! Loop: warm-load the resident models, then repeatedly
//! 1. release causal sessions their owners closed (`causal_close`);
//! 2. take the next dispatch (a job, or one block of the leased causal
//!    session), parking on the work condvar when there is none;
//! 3. run it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_protocol::{ApiError, ModelId};

use crate::backend::{ClipSink, CollectSink, EngineBackend, LoadEvent, NullSink, SessionId};
use crate::cancel::{is_cancel, OutputMode, StepControl, StepEvent};
use crate::pool::Residency;
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
    for m in warm {
        // A failure is recorded in the pool (readiness `Failed`).
        let _ = load(&sh, idx, backend.as_mut(), &m);
    }
    loop {
        let (closing, work) = {
            let mut st = sh.lock();
            loop {
                if st.shutdown {
                    break (std::mem::take(&mut st.to_close[idx]), None);
                }
                let closing = std::mem::take(&mut st.to_close[idx]);
                if !closing.is_empty() {
                    break (closing, None);
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
                    break (Vec::new(), Some(d));
                }
                st = sh.work_cv.wait(st).unwrap_or_else(|p| p.into_inner());
            }
        };
        for s in closing {
            backend.causal_close(s);
        }
        match work {
            Some(Dispatch::Job(it)) => run_job(&sh, idx, backend.as_mut(), it),
            Some(Dispatch::Causal(sid)) => causal_turn(&sh, backend.as_mut(), sid),
            None => {
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

/// Loads `model` on this executor, tracking residency in the pool.
fn load(sh: &Shared, idx: usize, backend: &mut dyn EngineBackend, model: &ModelId) -> Result<(), ApiError> {
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
    load(sh, idx, backend, model)
}

fn run_job(sh: &Arc<Shared>, idx: usize, backend: &mut dyn EngineBackend, it: QueueItem) {
    let Some((tx, cancel, job, mode)) = ({
        let st = sh.lock();
        st.jobs
            .get(&it.job)
            .map(|e| (e.tx.clone(), e.cancel.clone(), e.job.clone(), e.mode.clone()))
    }) else {
        sh.lock().sched.finish(idx);
        return;
    };
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
        let ctl = StepControl::new(cancel.clone(), mode.clone(), move |ev| {
            let _ = etx.send(match ev {
                StepEvent::Stage(name) => EngineEvent::Stage { name },
                StepEvent::Progress { step, total } => EngineEvent::Progress { step, total },
                StepEvent::Log(l) => EngineEvent::Log(l),
            });
        });
        let mut collect = CollectSink::default();
        let mut null = NullSink;
        let sink: &mut dyn ClipSink = match mode {
            OutputMode::Frames => &mut collect,
            OutputMode::File { .. } => &mut null,
        };
        let mut out = backend.generate(&job, sink, &ctl)?;
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
