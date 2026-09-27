//! `fv-gpucheck engine`: one model through the serve engine (WP-11), end to
//! end on the GPU: `CudaBackend` loads it resident, `EngineService::submit`
//! runs a request negotiated from the model's caps, and the job's MP4 and
//! frames are checked. With `--reference <clip dir>` the job's PNG frames are
//! compared byte for byte with a CLI run (`h3|ltx2|wan gen`) of the same
//! recipe and seed. With `--cancel-after-step N` a second job is cancelled
//! once step N is reported, and must end `Cancelled` within one step.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fastvideo_engine_service::cuda::{
    self, CudaBackend, CudaBackendConfig, CudaRecipe, Mp4Encoder, WeightLayout,
};
use fastvideo_engine_service::{EngineConfig, EngineEvent, EngineService, Priority, Readiness};
use fastvideo_protocol::{
    negotiate, CanvasSpec, GenerationRequest, JobId, Length, ProtocolId, Snap, StagedInputs,
    TimingSpec,
};
use serde_json::json;

use crate::report::{Report, StageResult};

pub struct Args<'a> {
    pub model: &'a str,
    pub weights_root: &'a Path,
    pub tae_dir: Option<&'a Path>,
    pub prompt: &'a str,
    pub seed: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_frames: Option<u32>,
    pub out: &'a Path,
    pub text_encoder: Option<&'a str>,
    pub adaln_cache: Option<&'a Path>,
    pub ltx_text: Option<&'a str>,
    pub reference: Option<&'a Path>,
    pub cancel_after_step: Option<u32>,
    pub encoder: &'a str,
}

/// `frame-*.png` under `dir`, sorted.
fn frames_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("frame-") && n.ends_with(".png"))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// Per-frame SHA-256 digests and the digest of their concatenation (the
/// `frames_sha256` of `wan gen`'s benchmark.json).
fn digests(frames: &[PathBuf]) -> anyhow::Result<(Vec<[u8; 32]>, String)> {
    let mut per = Vec::with_capacity(frames.len());
    let mut all = Vec::new();
    for f in frames {
        let d = crate::benchmark::sha256(&std::fs::read(f)?);
        all.extend_from_slice(&d);
        per.push(d);
    }
    Ok((per, crate::benchmark::sha256_hex_bytes(&all)))
}

pub fn run(report: &mut Report, a: &Args<'_>) -> StageResult<()> {
    let mut layout = WeightLayout::new(a.weights_root);
    if let Some(t) = a.tae_dir {
        layout = layout.with_tae_dir(t);
    }
    let cat = cuda::catalog(&layout);
    let mut model =
        cuda::find(&cat, a.model).ok_or_else(|| anyhow::anyhow!("unknown model {}", a.model))?;
    match &mut model.recipe {
        CudaRecipe::H3(r) => {
            if let Some(t) = a.text_encoder {
                r.text_encoder = t.to_owned();
            }
            r.adaln_cache = a.adaln_cache.map(Path::to_path_buf);
        }
        CudaRecipe::Ltx2(r) => {
            if let Some(t) = a.ltx_text {
                r.text = t.to_owned();
            }
        }
        _ => {}
    }
    let encoder = match a.encoder {
        "nvenc" => Mp4Encoder::Nvenc,
        "x264" => Mp4Encoder::Libx264CpuTest,
        _ if fastvideo_media::video::nvenc_available() => Mp4Encoder::Nvenc,
        _ => Mp4Encoder::Libx264CpuTest,
    };
    report.set("model", &model);
    report.set("recipe", model.describe());
    report.set("mp4_encoder", format!("{encoder:?}"));
    let mut cfg = CudaBackendConfig::new(0, vec![model.clone()]);
    cfg.keep_frames = true;
    cfg.encoder = encoder;
    cfg.work_dir = a.out.join("work");
    let backend = CudaBackend::new(cfg).map_err(|e| anyhow::anyhow!("backend: {e}"))?;
    report.set("process_plan", json!({"profile": backend.plan().profile, "settings": backend.plan().settings, "env": backend.plan().env}));
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let t0 = Instant::now();
    let engine = EngineService::start(
        EngineConfig {
            output_dir: a.out.to_path_buf(),
            ..EngineConfig::default()
        },
        vec![Box::new(backend)],
    )
    .map_err(|e| anyhow::anyhow!("engine start: {e}"))?;
    let ready = rt.block_on(engine.wait_ready());
    let load_s = t0.elapsed().as_secs_f64();
    report.note("load", json!({"seconds": load_s, "readiness": format!("{ready:?}"), "pool": format!("{:?}", engine.pool())}));
    report.check(
        "ready",
        ready == Readiness::Ready,
        json!({"readiness": format!("{ready:?}")}),
        json!({"want": "Ready"}),
    )?;

    let caps = engine
        .caps()
        .get(&model.id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("model not in the capability table"))?;
    report.set("caps", &caps);
    let mut req = GenerationRequest::text(ProtocolId::Native, model.id.as_str(), a.prompt);
    req.seed = Some(a.seed);
    if let (Some(w), Some(h)) = (a.width, a.height) {
        req.canvas = CanvasSpec::Exact {
            width: w,
            height: h,
        };
    }
    req.timing = TimingSpec {
        length: match a.num_frames {
            Some(n) => Length::Frames {
                value: n,
                snap: Snap::Exact,
            },
            None => Length::ModelDefault,
        },
        fps: None,
    };
    let job = negotiate(&req, &caps, &StagedInputs::default())
        .map_err(|e| anyhow::anyhow!("negotiate: {e:?}"))?;
    report.set("job", &job);

    // ---- one job, to an MP4 ----
    let id = JobId::new();
    let t1 = Instant::now();
    let mut handle = rt
        .block_on(engine.submit(id, job.clone(), Priority::Batch))
        .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
    let (mut steps, mut stages, mut first_step_s, mut out) = (Vec::new(), Vec::new(), None, None);
    rt.block_on(async {
        while let Some(ev) = handle.events.recv().await {
            match ev {
                EngineEvent::Progress { step, total } => {
                    first_step_s.get_or_insert(t1.elapsed().as_secs_f64());
                    steps.push((step, total));
                }
                EngineEvent::Stage { name } => stages.push((name, t1.elapsed().as_secs_f64())),
                EngineEvent::Finished(o) => {
                    out = Some(Ok(o));
                    break;
                }
                EngineEvent::Failed(e) => {
                    out = Some(Err(format!("{e:?}")));
                    break;
                }
                EngineEvent::Cancelled => {
                    out = Some(Err("cancelled".into()));
                    break;
                }
                _ => {}
            }
        }
    });
    let wall_s = t1.elapsed().as_secs_f64();
    let out = out
        .unwrap_or_else(|| Err("no terminal event".into()))
        .map_err(|e| anyhow::anyhow!("job failed: {e}"))?;
    report.note(
        "timings",
        json!({
            "submit_to_mp4_s": wall_s,
            "first_step_s": first_step_s,
            "stages": stages.iter().map(|(n, t)| json!({"stage": n, "at_s": t})).collect::<Vec<_>>(),
            "metrics": out.metrics,
            "load_s": load_s,
        }),
    );
    let want_steps = model.describe().steps.unwrap_or(0);
    report.check(
        "progress",
        steps.last().is_some_and(|(s, t)| s == t) && steps.len() as u32 == want_steps,
        json!({"events": steps.len(), "last": steps.last()}),
        json!({"steps": want_steps}),
    )?;
    let mp4 = out.mp4.clone().ok_or_else(|| anyhow::anyhow!("no mp4"))?;
    let bytes = std::fs::metadata(&mp4).map(|m| m.len()).unwrap_or(0);
    let probe =
        fastvideo_media::probe::ffprobe(&mp4).map_err(|e| anyhow::anyhow!("ffprobe: {e}"))?;
    let (ow, oh) = job.output_size();
    let frames_mp4 = probe
        .duration_s
        .zip(probe.fps)
        .map(|(d, f)| (d * f).round() as u32);
    let audio_ok = job.audio.has_audio() == probe.audio_rate.is_some();
    report.check(
        "mp4",
        bytes > 0 && probe.width == Some(ow) && probe.height == Some(oh) && audio_ok
            && frames_mp4.is_some_and(|n| n.abs_diff(job.num_frames) <= 1),
        json!({"path": mp4, "bytes": bytes, "width": probe.width, "height": probe.height, "fps": probe.fps,
               "duration_s": probe.duration_s, "frames": frames_mp4, "audio_rate": probe.audio_rate}),
        json!({"width": ow, "height": oh, "frames": job.num_frames, "audio": job.audio.has_audio()}),
    )?;
    let job_dir = mp4.parent().unwrap_or(a.out).to_path_buf();
    let frames = frames_in(&job_dir.join("frames"));
    let (per, digest) = digests(&frames)?;
    report.check(
        "frames",
        frames.len() as u32 == job.num_frames,
        json!({"pngs": frames.len(), "frames_sha256": digest}),
        json!({"frames": job.num_frames}),
    )?;
    if let Some(r) = a.reference {
        let rf = frames_in(r);
        let (rper, rdigest) = digests(&rf)?;
        let first_diff = per.iter().zip(&rper).position(|(x, y)| x != y);
        let same = per.len() == rper.len() && first_diff.is_none();
        report.check(
            "identical_to_cli",
            same,
            json!({"engine_frames": per.len(), "cli_frames": rper.len(), "engine_sha256": digest,
                   "cli_sha256": rdigest, "first_differing_frame": first_diff, "cli_dir": r}),
            json!({"identical": true}),
        )?;
    }

    // ---- cancel mid-run ----
    if let Some(at) = a.cancel_after_step {
        let used_before = fastvideo_cudarc::wan::device::pool_usage().map(|u| u.used);
        let id = JobId::new();
        let t2 = Instant::now();
        let mut h = rt
            .block_on(engine.submit(id, job.clone(), Priority::Batch))
            .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
        let (mut cancel_at, mut after_cancel, mut end) = (None::<(u32, f64)>, 0u32, None);
        rt.block_on(async {
            loop {
                let ev =
                    match tokio::time::timeout(Duration::from_secs(3600), h.events.recv()).await {
                        Ok(Some(ev)) => ev,
                        _ => break,
                    };
                match ev {
                    EngineEvent::Progress { step, .. } => {
                        if cancel_at.is_some() {
                            after_cancel += 1;
                        } else if step >= at {
                            cancel_at = Some((step, t2.elapsed().as_secs_f64()));
                            engine.cancel(id);
                        }
                    }
                    EngineEvent::Cancelled => {
                        end = Some(("cancelled", t2.elapsed().as_secs_f64()));
                        break;
                    }
                    EngineEvent::Finished(_) => {
                        end = Some(("finished", t2.elapsed().as_secs_f64()));
                        break;
                    }
                    EngineEvent::Failed(_) => {
                        end = Some(("failed", t2.elapsed().as_secs_f64()));
                        break;
                    }
                    _ => {}
                }
            }
        });
        // The executor thread has handed the pool back by the time the
        // terminal event is sent; let the stream settle.
        std::thread::sleep(Duration::from_millis(500));
        let used_after = fastvideo_cudarc::wan::device::pool_usage().map(|u| u.used);
        let leftover = job_dir
            .parent()
            .map(|p| p.join(id.to_string()).join("frames"))
            .filter(|p| p.exists());
        report.check(
            "cancel",
            end.is_some_and(|(k, _)| k == "cancelled") && after_cancel <= 1,
            json!({"cancelled_at_step": cancel_at.map(|c| c.0), "cancel_at_s": cancel_at.map(|c| c.1),
                   "end": end.map(|e| e.0), "end_s": end.map(|e| e.1),
                   "cancel_to_end_s": cancel_at.zip(end).map(|(c, e)| e.1 - c.1),
                   "steps_after_cancel": after_cancel, "pool_used_before": used_before,
                   "pool_used_after": used_after, "leftover_frames_dir": leftover}),
            json!({"end": "cancelled", "steps_after_cancel_max": 1}),
        )?;
    }
    rt.block_on(engine.drain(Duration::from_secs(5)));
    Ok(())
}
