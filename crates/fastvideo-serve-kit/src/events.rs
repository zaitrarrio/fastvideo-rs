//! Applies engine progress to stored jobs, stores outputs as artifacts, and
//! fires callbacks on status changes.
//!
//! [`JobEvent`] mirrors `EngineEvent` (design §3.6) without depending on the
//! engine crate's types; the binary maps one to the other and calls
//! [`apply_event`] for every event of a job's `JobHandle`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_protocol::{ApiError, Job, JobId, JobMetrics, LogLine};

use crate::artifacts::ArtifactMeta;
use crate::ctx::ServeCtx;

/// One engine event for a job.
#[derive(Clone, Debug, PartialEq)]
pub enum JobEvent {
    Queued { position: u32 },
    Started,
    Stage { name: String },
    Progress { step: u32, total: u32 },
    Log(LogLine),
    Finished(FinishedOutput),
    Failed(ApiError),
    Cancelled,
}

/// A finished generation: the output file (moved into the artifact store).
#[derive(Clone, Debug, PartialEq)]
pub struct FinishedOutput {
    pub file: PathBuf,
    pub meta: ArtifactMeta,
    pub metrics: JobMetrics,
}

/// Applies `ev` to job `id`. Illegal transitions (an event after the job
/// already ended, e.g. `Finished` after a cancel) leave the job unchanged and
/// discard any output. Status changes fire the job's callback.
pub async fn apply_event(ctx: &ServeCtx, id: JobId, ev: JobEvent) -> Result<Job, ApiError> {
    let now = ctx.now();
    let current = ctx
        .jobs()
        .get(id)
        .await
        .ok_or_else(|| ApiError::not_found(format!("job {id} not found")))?;
    let before = current.status();
    // docs/serve/tracing.md: the output path of a traced job.
    let trace = matches!(ev, JobEvent::Finished(_))
        .then(|| current.trace.as_deref().and_then(fastvideo_trace::Trace::from_traceparent).or_else(fastvideo_trace::current))
        .flatten();
    drop(current);
    let job = match ev {
        JobEvent::Finished(out) => {
            tracing::debug!(job = %id, "job: storing the output");
            let t_put = trace.map(|_| fastvideo_trace::now_ns());
            let bytes = if trace.is_some() { std::fs::metadata(&out.file).map(|m| m.len() as i64).unwrap_or(0) } else { 0 };
            let art = ctx.artifacts().put(&out.file, out.meta).await;
            if let (Some(t), Some(s)) = (trace, t_put) {
                t.span_since(fastvideo_trace::Comp::Upload, "artifact_put", s, bytes);
            }
            tracing::debug!(job = %id, "job: output stored");
            let art = match art {
                Ok(a) => a,
                Err(e) => {
                    let e2 = e.clone();
                    let j = ctx
                        .jobs()
                        .update(id, Box::new(move |j| {
                            let _ = j.mark_failed(now, e2);
                        }))
                        .await?;
                    tracing::warn!(job = %id, error = %e, "storing output failed");
                    finish(ctx, before, &j);
                    return Ok(j);
                }
            };
            let accepted = Arc::new(Mutex::new(false));
            let acc = accepted.clone();
            let a2 = art.clone();
            let t_store = trace.map(|_| fastvideo_trace::now_ns());
            let j = match ctx
                .jobs()
                .update(id, Box::new(move |j| {
                    *acc.lock().unwrap_or_else(|p| p.into_inner()) =
                        j.mark_succeeded(now, vec![a2], out.metrics).is_ok();
                }))
                .await
            {
                Ok(j) => j,
                Err(e) => {
                    // The job went away (removed / expired) while the output
                    // was being stored: do not leak the stored artifact.
                    ctx.artifacts().delete(&art).await;
                    return Err(e.into());
                }
            };
            if let (Some(t), Some(s)) = (trace, t_store) {
                t.span_since(fastvideo_trace::Comp::Store, "terminal_write", s, 0);
            }
            if !*accepted.lock().unwrap_or_else(|p| p.into_inner()) {
                ctx.artifacts().delete(&art).await;
            }
            j
        }
        ev => {
            ctx.jobs()
                .update(id, Box::new(move |j| match ev {
                    JobEvent::Queued { position } => {
                        if !j.is_terminal() && j.started_at.is_none() {
                            j.queue_position = Some(position);
                        }
                    }
                    JobEvent::Started => {
                        let _ = j.mark_running(now);
                    }
                    JobEvent::Stage { name } => {
                        if !j.is_terminal() {
                            j.logs.push(LogLine::info(format!("stage: {name}"), now));
                        }
                    }
                    JobEvent::Progress { step, total } => j.set_step(step, total),
                    JobEvent::Log(l) => {
                        if !j.is_terminal() {
                            j.logs.push(l);
                        }
                    }
                    JobEvent::Failed(e) => {
                        let _ = j.mark_failed(now, e);
                    }
                    JobEvent::Cancelled => {
                        let _ = j.mark_cancelled(now);
                    }
                    JobEvent::Finished(_) => unreachable!("handled above"),
                }))
                .await?
        }
    };
    finish(ctx, before, &job);
    if job.is_terminal() && before != job.status() {
        tracing::debug!(job = %id, status = job.status().as_str(), "job: terminal state recorded");
    }
    Ok(job)
}

fn finish(ctx: &ServeCtx, before: fastvideo_protocol::JobStatus, job: &Job) {
    if job.status() != before {
        ctx.notify(job);
    }
}

/// Cancels a job: a queued job is cancelled at once; a running one gets
/// `cancel_requested` and the engine's token is tripped (the engine then
/// reports `Cancelled`). A finished job is `AlreadyCompleted`.
pub async fn cancel_job(ctx: &ServeCtx, id: JobId) -> Result<Job, ApiError> {
    let job = ctx
        .jobs()
        .get(id)
        .await
        .ok_or_else(|| ApiError::not_found(format!("job {id} not found")))?;
    if job.is_terminal() {
        return Err(ApiError::already_completed("the job already finished"));
    }
    ctx.engine().cancel(id).await;
    let now = ctx.now();
    let before = job.status();
    let job = ctx
        .jobs()
        .update(id, Box::new(move |j| {
            if j.status() == fastvideo_protocol::JobStatus::Queued {
                let _ = j.mark_cancelled(now);
            } else if !j.is_terminal() {
                j.cancel_requested = true;
            }
        }))
        .await?;
    finish(ctx, before, &job);
    Ok(job)
}

/// Waits until job `id` is terminal or `timeout` passes; returns the latest job.
pub async fn wait_terminal(ctx: &ServeCtx, id: JobId, timeout: Duration) -> Option<Job> {
    let mut rx = ctx.jobs().watch(id)?;
    let _ = tokio::time::timeout(timeout, async {
        loop {
            if rx.borrow_and_update().state.is_terminal() {
                return;
            }
            if rx.changed().await.is_err() {
                return;
            }
        }
    })
    .await;
    ctx.jobs().get(id).await
}
