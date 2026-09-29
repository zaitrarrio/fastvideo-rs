//! The engine seam: serve-kit's [`EngineGate`] over the real
//! [`EngineService`], and the pump that turns each job's `EngineEvent`
//! stream into serve-kit [`JobEvent`]s (applied to the store, which fires
//! callbacks). serve-kit's WP-05 notes leave this glue to WP-10.
//!
//! On `Finished` the MP4 goes through `fastvideo-media::mp4::finalize`
//! (faststart remux, `-an`, crop; design §4) when the job needs it or
//! ffmpeg is present, then into the artifact store. The fake engine writes
//! no MP4 without ffmpeg; with `placeholder_output` a small stand-in file is
//! stored so CPU CI still sees a finished job with an artifact.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use fastvideo_engine_service::{CancelOutcome, ClipOutput, EngineEvent, EngineService, Priority, Readiness};
use fastvideo_media::mp4;
use fastvideo_media::video::FfmpegH264;
use fastvideo_protocol::{ApiError, AudioPlan, Job, JobId, ModelCaps, ResolvedJob};
use fastvideo_serve_kit::events::apply_event;
use fastvideo_serve_kit::{ArtifactMeta, EngineGate, FinishedOutput, JobEvent, ServeCtx};

/// How finished outputs are post-processed.
#[derive(Clone, Debug)]
pub struct OutputPolicy {
    /// Store a placeholder when the engine produced no file (fake engine
    /// without ffmpeg).
    pub placeholder: bool,
    /// Encoder for crop re-encodes.
    pub encoder: FfmpegH264,
    /// Where finalized files are staged before the artifact store takes them.
    pub scratch: PathBuf,
}

/// [`EngineGate`] over [`EngineService`].
pub struct ServiceGate {
    engine: EngineService,
    aliases: BTreeMap<String, String>,
    ctx: OnceLock<ServeCtx>,
    admitting: AtomicBool,
    output: OutputPolicy,
}

impl std::fmt::Debug for ServiceGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceGate").field("engine", &self.engine).finish_non_exhaustive()
    }
}

impl ServiceGate {
    /// `aliases`: the config's `[aliases]`; the engine's tier aliases
    /// (`h3-max`, `ltx-turbo`, ...) are added unless the config names them.
    pub fn new(engine: EngineService, mut aliases: BTreeMap<String, String>, output: OutputPolicy) -> Arc<Self> {
        for (alias, model) in engine.caps().tier_aliases() {
            aliases.entry(alias).or_insert(model.0);
        }
        Arc::new(Self { engine, aliases, ctx: OnceLock::new(), admitting: AtomicBool::new(true), output })
    }

    /// Connects the gate to the context built around it (once).
    pub fn attach(&self, ctx: ServeCtx) {
        let _ = self.ctx.set(ctx);
    }

    pub fn engine(&self) -> &EngineService {
        &self.engine
    }

    pub fn aliases(&self) -> &BTreeMap<String, String> {
        &self.aliases
    }

    /// Stops admission (shutdown step 1): new submits answer `Loading` (503).
    pub fn stop_admission(&self) {
        self.admitting.store(false, Ordering::SeqCst);
    }

    pub fn admitting(&self) -> bool {
        self.admitting.load(Ordering::SeqCst)
    }

    fn ctx(&self) -> Result<&ServeCtx, ApiError> {
        self.ctx.get().ok_or_else(|| ApiError::internal("engine gate is not attached"))
    }
}

#[async_trait::async_trait]
impl EngineGate for ServiceGate {
    fn models(&self) -> Vec<ModelCaps> {
        self.engine.caps().models().cloned().collect()
    }

    fn alias(&self, name: &str) -> Option<String> {
        self.aliases.get(name).cloned()
    }

    fn admit(&self) -> Result<(), ApiError> {
        if !self.admitting() {
            return Err(ApiError::loading("the server is shutting down"));
        }
        match self.engine.readiness() {
            Readiness::Ready => Ok(()),
            Readiness::Loading { done, total } => {
                Err(ApiError::loading(format!("models are loading ({done}/{total})")))
            }
            Readiness::Failed(m) => Err(ApiError::engine_failed(format!("model loading failed: {m}"))),
        }
    }

    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        let ctx = self.ctx()?.clone();
        let handle = self.engine.submit(job.id, job.resolved.clone(), Priority::Batch).await?;
        metrics::counter!("fv_jobs_submitted_total", "api" => job.protocol.as_str()).increment(1);
        let output = self.output.clone();
        let (id, api, resolved) = (job.id, job.protocol.as_str(), job.resolved.clone());
        let name = crate::adapters::artifact_file_name(job);
        tokio::spawn(pump(ctx, id, api, resolved, name, handle.events, output));
        Ok(())
    }

    async fn cancel(&self, id: JobId) -> bool {
        self.engine.cancel(id) != CancelOutcome::Unknown
    }
}

/// Maps a job's engine events onto the store until its terminal event.
async fn pump(
    ctx: ServeCtx,
    id: JobId,
    api: &'static str,
    resolved: ResolvedJob,
    file_name: String,
    mut events: tokio::sync::mpsc::UnboundedReceiver<EngineEvent>,
    output: OutputPolicy,
) {
    let started = std::time::Instant::now();
    let mut ended = false;
    while let Some(ev) = events.recv().await {
        let ev = match ev {
            EngineEvent::Queued { position } => JobEvent::Queued { position },
            EngineEvent::Started => JobEvent::Started,
            EngineEvent::Stage { name } => JobEvent::Stage { name: name.to_owned() },
            EngineEvent::Progress { step, total } => JobEvent::Progress { step, total },
            EngineEvent::Log(l) => JobEvent::Log(l),
            EngineEvent::Failed(e) => JobEvent::Failed(e),
            EngineEvent::Cancelled => JobEvent::Cancelled,
            EngineEvent::Finished(out) => match finish(id, &resolved, &file_name, out, &output).await {
                Ok(f) => JobEvent::Finished(f),
                Err(e) => JobEvent::Failed(e),
            },
        };
        let terminal = matches!(ev, JobEvent::Finished(_) | JobEvent::Failed(_) | JobEvent::Cancelled);
        let status = match &ev {
            JobEvent::Finished(_) => "succeeded",
            JobEvent::Failed(_) => "failed",
            JobEvent::Cancelled => "cancelled",
            _ => "",
        };
        if let Err(e) = apply_event(&ctx, id, ev).await {
            tracing::warn!(job = %id, error = %e, "applying engine event failed");
        }
        if terminal {
            metrics::counter!("fv_jobs_finished_total", "api" => api, "status" => status).increment(1);
            metrics::histogram!("fv_job_duration_seconds", "api" => api).record(started.elapsed().as_secs_f64());
            ended = true;
            break;
        }
    }
    if !ended {
        let e = ApiError::internal("the engine dropped the job");
        let _ = apply_event(&ctx, id, JobEvent::Failed(e)).await;
    }
    let _ = tokio::fs::remove_dir_all(output.scratch.join(id.to_string())).await;
}

/// The engine's output -> a finalized file plus its artifact facts.
async fn finish(id: JobId, r: &ResolvedJob, file_name: &str, out: ClipOutput, policy: &OutputPolicy) -> Result<FinishedOutput, ApiError> {
    let dir = policy.scratch.join(id.to_string());
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| ApiError::internal(format!("output dir: {e}")))?;
    // Pad-and-crop canvases (LTX 1280x720 -> 1280x768 -> crop) report the
    // cropped size; `-an` outputs carry no audio.
    let (width, height) = r.output_size();
    let mut audio = match r.audio {
        AudioPlan::Native { rate, channels } => Some((rate, channels)),
        _ => None,
    };
    if r.post.drop_audio {
        audio = None;
    }
    let file = match out.mp4 {
        Some(src) => {
            let post: mp4::PostProcess = (&r.post).into();
            let needs_post = post.crop.is_some() || post.drop_audio;
            if needs_post || fastvideo_media::tools::ffmpeg_available() {
                let dst = dir.join("final.mp4");
                let (src2, dst2, enc) = (src.clone(), dst.clone(), policy.encoder);
                tokio::task::spawn_blocking(move || mp4::finalize_with(&src2, &dst2, &post, enc))
                    .await
                    .map_err(|e| ApiError::internal(format!("post-processing task: {e}")))?
                    .map_err(|e| ApiError::engine_failed(format!("post-processing: {e}")))?;
                let _ = tokio::fs::remove_file(&src).await;
                dst
            } else {
                src
            }
        }
        None if policy.placeholder => write_placeholder(&dir, id, r).await?,
        None => return Err(ApiError::engine_failed("the engine produced no output file")),
    };
    Ok(FinishedOutput {
        file,
        meta: ArtifactMeta {
            file_name: file_name.to_owned(),
            mime: "video/mp4".into(),
            width,
            height,
            // An edit's output adds the source frames outside its window.
            frames: r.output_frames(),
            fps: r.fps,
            audio,
        },
        metrics: out.metrics,
    })
}

async fn write_placeholder(dir: &Path, id: JobId, r: &ResolvedJob) -> Result<PathBuf, ApiError> {
    let p = dir.join("placeholder.mp4");
    let body = format!(
        "fv-serve placeholder output (fake engine without ffmpeg)\njob={id} model={} {}x{} frames={} fps={} seed={}\n",
        r.model, r.width, r.height, r.num_frames, r.fps, r.seed
    );
    tokio::fs::write(&p, body)
        .await
        .map_err(|e| ApiError::internal(format!("writing placeholder: {e}")))?;
    Ok(p)
}
