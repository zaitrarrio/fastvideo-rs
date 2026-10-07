//! EngineService over the FakeBackend: job lifecycle, progress, scheduling,
//! cancellation, readiness, sessions, tiers and drain (design §3.6, §7.3).

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use common::*;
use fastvideo_engine_service::fake::{audio_len, decode_frame_index};
use fastvideo_engine_service::{
    CancelOutcome, CausalBlock, CausalSession, ClipBuild, EngineConfig, EngineEvent, FakeConfig, FakeFaults, FakeModel,
    FakeTiming, FakeWarmup, ManualClock, Mp4Mode, Priority, Readiness, Residency, Tier, Warmup,
    SOL_H3_4STEP_PROFILE,
};
use fastvideo_protocol::{
    AudioPlan, AudioTrack, Continuity, ErrorKind, Family, JobId, ModelId, SessionSpec, TrackSet,
    VideoTrack,
};

fn h3_audio() -> AudioPlan {
    AudioPlan::Native {
        rate: 32_000,
        channels: 2,
    }
}

fn fast_fake() -> FakeConfig {
    FakeConfig {
        timing: FakeTiming {
            step: Duration::from_millis(1),
            ..FakeTiming::default()
        },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    }
}

fn spec(model: &str, fps: u32, audio: bool) -> SessionSpec {
    SessionSpec {
        model: model.into(),
        tracks: TrackSet {
            video: VideoTrack {
                name: "main_video".into(),
                width: 64,
                height: 32,
                fps,
            },
            audio: audio.then(|| AudioTrack {
                name: "main_audio".into(),
                rate: 48_000,
                channels: 2,
            }),
        },
        canvas: (64, 32),
        fps,
        continuity: Continuity::HardCut,
        max_seconds: None,
        seed: Some(7),
    }
}

#[tokio::test]
async fn job_lifecycle_events_and_fake_output() {
    let e = ready(EngineConfig::default(), fast_fake()).await;
    let j = job("fake-h3-turbo", 124, 24, h3_audio(), "a cat");
    let mut h = e
        .submit_frames(JobId::new(), j.clone(), Priority::Batch)
        .await
        .unwrap();
    let evs = until_terminal(&mut h).await;
    assert_eq!(evs[0], EngineEvent::Queued { position: 0 });
    assert_eq!(evs[1], EngineEvent::Started);
    let stages: Vec<&str> = evs
        .iter()
        .filter_map(|e| match e {
            EngineEvent::Stage { name } => Some(*name),
            _ => None,
        })
        .collect();
    assert_eq!(stages, ["text_encode", "denoise", "decode"]);
    assert_eq!(progress_steps(&evs), [1, 2, 3, 4], "turbo recipe = 4 steps");
    let EngineEvent::Finished(out) = evs.last().unwrap().clone() else {
        panic!("not finished: {evs:?}")
    };
    let frames = out.frames.clone().unwrap();
    assert_eq!(frames.len(), 124);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (64, 32));
        assert_eq!(decode_frame_index(f), Some(i as u32), "order, no drops");
    }
    let a = out.audio.clone().unwrap();
    assert_eq!((a.rate, a.channels), (32_000, 2));
    assert_eq!(a.frames(), audio_len(124, 24, 32_000));
    assert_eq!(a.samples[0], 1.0, "click at clip start");
    assert!(out.metrics.inference_s.is_some() && out.metrics.build_rtf.is_some());
    assert!(out.mp4.is_none());

    // Deterministic: the same job renders the same A/V.
    let again = e
        .submit_frames(JobId::new(), j, Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(again.frames.unwrap(), frames);
    assert_eq!(again.audio.unwrap(), a);

    // Video-only model: no audio.
    let out = e
        .submit_frames(JobId::new(), wan("x"), Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert!(out.audio.is_none());
    assert_eq!(out.frames.unwrap().len(), 49);
}

#[tokio::test]
async fn batch_file_mode_writes_mp4_when_ffmpeg_exists() {
    let dir = std::env::temp_dir().join(format!("fv-engine-test-{}", uuid::Uuid::new_v4()));
    let cfg = EngineConfig {
        output_dir: dir.clone(),
        ..EngineConfig::default()
    };
    let fake = FakeConfig {
        mp4: Mp4Mode::Auto,
        ..fast_fake()
    };
    let e = ready(cfg, fake).await;
    let id = JobId::new();
    let out = e
        .submit(id, job("fake-h3-turbo", 124, 24, h3_audio(), "p"), Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert!(out.frames.is_none(), "batch output is a file, not frames");
    let ffmpeg = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success());
    if ffmpeg {
        let p = out.mp4.expect("mp4 written");
        assert_eq!(p, dir.join(id.to_string()).join("output.mp4"));
        assert!(std::fs::metadata(&p).unwrap().len() > 0);
    } else {
        eprintln!("ffmpeg not found: MP4 write skipped");
        assert!(out.mp4.is_none());
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn stream_priority_runs_before_queued_batch() {
    let clock = Arc::new(ManualClock::new());
    let e = ready(EngineConfig::default(), manual_fake(&clock)).await;
    let mut a = e.submit(JobId::new(), wan("a"), Priority::Batch).await.unwrap();
    until_started(&mut a).await;
    sleepers(&clock, 1).await;

    let mut b = e.submit(JobId::new(), wan("b"), Priority::Batch).await.unwrap();
    let mut c = e.submit(JobId::new(), wan("c"), Priority::Batch).await.unwrap();
    let mut d = e.submit(JobId::new(), wan("d"), Priority::Stream).await.unwrap();
    assert_eq!(
        pending(&mut b),
        [EngineEvent::Queued { position: 0 }, EngineEvent::Queued { position: 1 }]
    );
    assert_eq!(
        pending(&mut c),
        [EngineEvent::Queued { position: 1 }, EngineEvent::Queued { position: 2 }]
    );
    assert_eq!(pending(&mut d), [EngineEvent::Queued { position: 0 }]);
    assert_eq!(e.queue_position(d.id), Some(0));
    assert_eq!(e.queue_position(c.id), Some(2));
    let s = e.stats();
    assert_eq!((s.queued_batch, s.queued_stream, s.running), (2, 1, 1));

    step_to(&clock, &mut a, 3).await;
    assert!(matches!(until_terminal(&mut a).await.last(), Some(EngineEvent::Finished(_))));
    // The executor took D next and is parked on D's first step: B still waits.
    until_started(&mut d).await;
    let bv = pending(&mut b);
    assert!(!has_started(&bv), "batch started before the stream build: {bv:?}");
    let _drive = Driver::new(&clock);
    let dv = until_terminal(&mut d).await;
    assert!(matches!(dv.last(), Some(EngineEvent::Finished(_))));
    assert!(until_terminal(&mut b).await.contains(&EngineEvent::Started));
    let cv = until_terminal(&mut c).await;
    assert!(cv.contains(&EngineEvent::Queued { position: 0 }));
    assert!(matches!(cv.last(), Some(EngineEvent::Finished(_))));
}

#[tokio::test]
async fn cancel_queued_job_dequeues_and_repositions() {
    let clock = Arc::new(ManualClock::new());
    let e = ready(EngineConfig::default(), manual_fake(&clock)).await;
    let mut a = e.submit(JobId::new(), wan("a"), Priority::Batch).await.unwrap();
    until_started(&mut a).await;
    let mut b = e.submit(JobId::new(), wan("b"), Priority::Batch).await.unwrap();
    let mut c = e.submit(JobId::new(), wan("c"), Priority::Batch).await.unwrap();
    pending(&mut c);

    b.cancel.cancel();
    assert_eq!(
        until_terminal(&mut b).await,
        [EngineEvent::Queued { position: 0 }, EngineEvent::Cancelled]
    );
    assert_eq!(ev(&mut c).await, EngineEvent::Queued { position: 0 });
    assert_eq!(e.cancel(c.id), CancelOutcome::Dequeued);
    assert_eq!(ev(&mut c).await, EngineEvent::Cancelled);
    assert_eq!(e.cancel(c.id), CancelOutcome::Unknown);
    assert_eq!(e.cancel(JobId::new()), CancelOutcome::Unknown);
    assert_eq!(e.stats().queued_batch, 0);

    // The running job is untouched.
    let _drive = Driver::new(&clock);
    assert!(matches!(until_terminal(&mut a).await.last(), Some(EngineEvent::Finished(_))));
}

#[tokio::test]
async fn cancel_running_job_stops_within_one_step() {
    let clock = Arc::new(ManualClock::new());
    let e = ready(EngineConfig::default(), manual_fake(&clock)).await;
    let mut a = e.submit(JobId::new(), wan("a"), Priority::Batch).await.unwrap();
    until_started(&mut a).await;
    sleepers(&clock, 1).await;
    clock.advance(Duration::from_secs(1));
    assert_eq!(ev(&mut a).await, EngineEvent::Stage { name: "text_encode" });
    assert_eq!(ev(&mut a).await, EngineEvent::Stage { name: "denoise" });
    assert_eq!(ev(&mut a).await, EngineEvent::Progress { step: 1, total: 3 });
    sleepers(&clock, 1).await;

    assert_eq!(e.cancel(a.id), CancelOutcome::Requested);
    clock.advance(Duration::from_secs(1));
    let rest = until_terminal(&mut a).await;
    assert_eq!(
        rest,
        [EngineEvent::Progress { step: 2, total: 3 }, EngineEvent::Cancelled],
        "the step in flight reports, then the job unwinds"
    );

    // The executor is free again.
    let _drive = Driver::new(&clock);
    let out = e
        .submit(JobId::new(), wan("next"), Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await;
    assert!(out.is_ok());
}

#[tokio::test]
async fn injected_failure_fails_the_job() {
    let e = ready(EngineConfig::default(), fast_fake()).await;
    let err = e
        .submit(JobId::new(), wan("boom [fake:fail]"), Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::EngineFailed);
    assert!(err.message.contains("step 1"), "{}", err.message);
    // And the engine keeps working.
    assert!(e
        .submit(JobId::new(), wan("fine"), Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await
        .is_ok());
}

#[tokio::test]
async fn queue_full_and_admission_errors() {
    let clock = Arc::new(ManualClock::new());
    let cfg = EngineConfig {
        queue_max: 1,
        ..EngineConfig::default()
    };
    let e = ready(cfg, manual_fake(&clock)).await;
    let mut a = e.submit(JobId::new(), wan("a"), Priority::Batch).await.unwrap();
    until_started(&mut a).await;
    let _b = e.submit(JobId::new(), wan("b"), Priority::Batch).await.unwrap();
    let err = e
        .submit(JobId::new(), wan("c"), Priority::Batch)
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::QueueFull);
    assert!(err.retry_after_s.is_some());
    // Stream builds are not bounded by queue_max.
    let _s = e.submit(JobId::new(), wan("s"), Priority::Stream).await.unwrap();

    let err = e
        .submit(JobId::new(), job("nope", 49, 16, AudioPlan::None, "x"), Priority::Batch)
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidRequest);
    assert_eq!(err.param.as_deref(), Some("model"));

    let id = JobId::new();
    let _d = e.submit(id, wan("d"), Priority::Stream).await.unwrap();
    let err = e.submit(id, wan("d"), Priority::Stream).await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::Conflict);
}

#[tokio::test]
async fn readiness_tracks_loading() {
    let clock = Arc::new(ManualClock::new());
    let fake = FakeConfig {
        timing: FakeTiming {
            load: Duration::from_secs(4),
            step: Duration::from_secs(1),
            ..FakeTiming::default()
        },
        ..manual_fake(&clock)
    }
    .with_models(&["fake-wan", "fake-h3-turbo"]);
    let e = start(EngineConfig::default(), vec![fake]);
    assert_eq!(e.readiness(), Readiness::Loading { done: 0, total: 2 });
    let err = e
        .submit(JobId::new(), wan("early"), Priority::Batch)
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Loading);
    assert_eq!(err.retry_after_s, Some(1));
    sleepers(&clock, 1).await;
    assert!(matches!(
        e.pool().model_state(&"fake-h3-turbo".into()),
        Some(Residency::Loading { .. })
    ));
    let mut rx = e.watch_readiness();
    let _drive = Driver::new(&clock);
    // Progress is monotone (updates may coalesce, so `done: 1` can be skipped).
    tokio::time::timeout(
        T,
        rx.wait_for(|r| matches!(r, Readiness::Loading { done: 1, total: 2 } | Readiness::Ready)),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(tokio::time::timeout(T, e.wait_ready()).await.unwrap(), Readiness::Ready);
    assert_eq!(
        e.pool().resident_models(),
        vec![ModelId::new("fake-h3-turbo"), ModelId::new("fake-wan")]
    );
}

#[tokio::test]
async fn load_failure_is_reported() {
    let mut faults = FakeFaults::default();
    faults.load_fail.insert("fake-wan".into());
    let fake = FakeConfig {
        faults,
        ..fast_fake()
    };
    let e = start(EngineConfig::default(), vec![fake]);
    let r = tokio::time::timeout(T, e.wait_ready()).await.unwrap();
    assert!(matches!(&r, Readiness::Failed(m) if m.contains("fake-wan")), "{r:?}");
    let err = e
        .submit(JobId::new(), wan("x"), Priority::Batch)
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::EngineFailed);
    // Other models still serve.
    assert!(e
        .submit_frames(
            JobId::new(),
            job("fake-h3-turbo", 124, 24, h3_audio(), "ok"),
            Priority::Batch
        )
        .await
        .unwrap()
        .wait()
        .await
        .is_ok());
}

async fn next_ok(s: &mut CausalSession) -> CausalBlock {
    tokio::time::timeout(T, s.next_block())
        .await
        .expect("block timeout")
        .expect("session ended")
        .expect("block failed")
}

/// Advances one step at a time until `h` reports progress `k`.
async fn step_to(clock: &Arc<ManualClock>, h: &mut fastvideo_engine_service::JobHandle, k: u32) {
    loop {
        sleepers(clock, 1).await;
        clock.advance(Duration::from_secs(1));
        loop {
            match ev(h).await {
                EngineEvent::Progress { step, .. } if step == k => return,
                EngineEvent::Progress { .. } => break,
                e => assert!(!e.is_terminal(), "{e:?}"),
            }
        }
    }
}

#[tokio::test]
async fn causal_session_holds_an_exclusive_lease() {
    let e = ready(EngineConfig::default(), fast_fake()).await;
    let mut s = e
        .open_causal_session(spec("fake-sfwan", 16, false))
        .await
        .unwrap();
    // One session per executor.
    let err = e
        .open_causal_session(spec("fake-sfwan", 16, false))
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Conflict);
    let err = e
        .open_clip_session(spec("fake-h3-turbo", 24, true))
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Conflict);
    // Wrong streaming shape.
    assert!(e.open_causal_session(spec("fake-wan", 16, false)).await.is_err());

    assert_eq!(s.set_prompt("a"), 1);
    for want in 0..2u64 {
        let b = next_ok(&mut s).await;
        assert_eq!(b.index, want);
        assert_eq!(b.prompt_version, 1);
        assert_eq!(b.frames.len(), 12);
        for (i, f) in b.frames.iter().enumerate() {
            assert_eq!(decode_frame_index(f), Some((want * 12 + i as u64) as u32));
        }
        assert!(b.audio.is_none());
    }

    // Batch work for the same executor waits while the lease is held.
    let mut batch = e.submit(JobId::new(), wan("batch"), Priority::Batch).await.unwrap();
    for _ in 0..6 {
        next_ok(&mut s).await;
    }
    let bv = pending(&mut batch);
    assert_eq!(bv, [EngineEvent::Queued { position: 0 }]);

    // A prompt change lands at a block boundary.
    assert_eq!(s.set_prompt("b"), 2);
    let mut seen = None;
    for _ in 0..10 {
        let b = next_ok(&mut s).await;
        if b.prompt_version == 2 {
            seen = Some(b);
            break;
        }
    }
    let b = seen.expect("prompt v2 applied");
    assert!(b.index > 0 && !b.reset);

    // Reset restarts at block 0.
    s.reset();
    let mut reset = None;
    for _ in 0..10 {
        let b = next_ok(&mut s).await;
        if b.reset {
            reset = Some(b);
            break;
        }
    }
    let b = reset.expect("reset applied");
    assert_eq!(b.index, 0);
    assert_eq!(decode_frame_index(&b.frames[0]), Some(0));
    assert!(s.stats().blocks >= 10);

    // Closing frees the executor: the batch job runs.
    s.close();
    let bv = until_terminal(&mut batch).await;
    assert!(matches!(bv.last(), Some(EngineEvent::Finished(_))), "{bv:?}");
    // And a new session can open.
    let s2 = e
        .open_causal_session(spec("fake-sfwan", 16, false))
        .await
        .unwrap();
    drop(s2);
}

#[tokio::test]
async fn causal_session_pauses_and_reports_block_failures() {
    let e = ready(EngineConfig::default(), fast_fake()).await;
    let mut s = e
        .open_causal_session(spec("fake-sfwan", 16, false))
        .await
        .unwrap();
    // No prompt yet: nothing is generated.
    assert!(tokio::time::timeout(Duration::from_millis(100), s.next_block())
        .await
        .is_err());
    s.set_paused(true);
    s.set_prompt("p");
    assert!(tokio::time::timeout(Duration::from_millis(100), s.next_block())
        .await
        .is_err());
    s.set_paused(false);
    let b = tokio::time::timeout(T, s.next_block()).await.unwrap().unwrap().unwrap();
    assert_eq!(b.index, 0);

    s.set_prompt("bad [fake:fail]");
    let mut failed = false;
    for _ in 0..10 {
        match tokio::time::timeout(T, s.next_block()).await.unwrap() {
            Some(Ok(_)) => continue,
            Some(Err(err)) => {
                assert_eq!(err.kind, ErrorKind::EngineFailed);
                failed = true;
                break;
            }
            None => break,
        }
    }
    assert!(failed, "block failure delivered");
    assert!(tokio::time::timeout(T, s.next_block()).await.unwrap().is_none());
    // The lease was released by the failure.
    assert!(e
        .submit(JobId::new(), wan("after"), Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await
        .is_ok());
}

#[tokio::test]
async fn clip_session_builds_run_as_stream_jobs_in_memory() {
    let clock = Arc::new(ManualClock::new());
    let e = ready(EngineConfig::default(), manual_fake(&clock)).await;
    // A video+audio session needs 48000 % fps == 0 and a supported fps.
    let err = e
        .open_clip_session(spec("fake-h3-turbo", 25, true))
        .await
        .unwrap_err();
    assert_eq!(err.param.as_deref(), Some("fps"));

    let session = e.open_clip_session(spec("fake-h3-turbo", 24, true)).await.unwrap();
    assert_eq!(session.frames_for(Some(5.0)).unwrap(), 124);
    assert_eq!(session.frames_for(None).unwrap(), 124);
    assert_eq!(session.frames_for(Some(100.0)).unwrap_err().param.as_deref(), Some("seconds"));
    let r = session
        .resolve(&ClipBuild {
            prompt: "x".into(),
            seconds: Some(6.0),
            ..ClipBuild::default()
        })
        .unwrap();
    assert_eq!((r.num_frames, r.seed, r.width, r.height), (158, 7, 64, 32));
    assert_eq!(r.audio, h3_audio());
    assert_eq!(r.tier, Some(Tier::Turbo));
    assert_eq!(r.recipe.as_deref(), Some("4step-vsa"));

    let mut a = e.submit(JobId::new(), wan("a"), Priority::Batch).await.unwrap();
    until_started(&mut a).await;
    let mut b = e.submit(JobId::new(), wan("b"), Priority::Batch).await.unwrap();
    let mut build = session
        .build(ClipBuild {
            prompt: "clip".into(),
            ..ClipBuild::default()
        })
        .await
        .unwrap();
    assert_eq!(pending(&mut build), [EngineEvent::Queued { position: 0 }]);
    assert_eq!(
        pending(&mut b),
        [EngineEvent::Queued { position: 0 }, EngineEvent::Queued { position: 1 }]
    );

    step_to(&clock, &mut a, 3).await;
    until_terminal(&mut a).await;
    until_started(&mut build).await;
    assert!(!has_started(&pending(&mut b)), "clip build ran first");
    let _drive = Driver::new(&clock);
    let bv = until_terminal(&mut build).await;
    let Some(EngineEvent::Finished(out)) = bv.last() else {
        panic!("{bv:?}")
    };
    let frames = out.frames.as_ref().unwrap();
    assert_eq!(frames.len(), 124);
    assert_eq!(decode_frame_index(&frames[123]), Some(123));
    assert_eq!(out.audio.as_ref().unwrap().frames(), audio_len(124, 24, 32_000));
    assert!(out.mp4.is_none());
    until_terminal(&mut b).await;

    drop(session);
    let s = e.open_clip_session(spec("fake-h3-turbo", 24, true)).await;
    assert!(s.is_ok(), "slot released on drop");
}

#[tokio::test]
async fn executors_run_in_parallel() {
    let clock = Arc::new(ManualClock::new());
    let mut f1 = manual_fake(&clock).with_models(&["fake-wan"]);
    f1.device.index = 0;
    let mut f2 = manual_fake(&clock).with_models(&["fake-wan", "fake-h3-turbo"]);
    f2.device.index = 1;
    let e = start(EngineConfig::default(), vec![f1, f2]);
    assert_eq!(tokio::time::timeout(T, e.wait_ready()).await.unwrap(), Readiness::Ready);
    assert_eq!(
        e.caps().entry(&"fake-wan".into()).unwrap().executors,
        vec![0, 1]
    );
    let mut a = e.submit(JobId::new(), wan("a"), Priority::Batch).await.unwrap();
    let mut b = e.submit(JobId::new(), wan("b"), Priority::Batch).await.unwrap();
    until_started(&mut a).await;
    until_started(&mut b).await;
    sleepers(&clock, 2).await;
    assert_eq!(e.stats().running, 2);
    // An H3 job can only go to executor 1; it waits for it.
    let mut h = e
        .submit_frames(
            JobId::new(),
            job("fake-h3-turbo", 124, 24, h3_audio(), "h"),
            Priority::Batch,
        )
        .await
        .unwrap();
    let _drive = Driver::new(&clock);
    for hd in [&mut a, &mut b, &mut h] {
        assert!(matches!(until_terminal(hd).await.last(), Some(EngineEvent::Finished(_))));
    }
}

#[tokio::test]
async fn swap_mode_loads_on_demand() {
    let mut fake = fast_fake();
    fake.models = vec![FakeModel::wan(), {
        let mut m = FakeModel::h3_turbo();
        m.caps.resident = false;
        m
    }];
    let e = ready(EngineConfig::default(), fake.clone()).await;
    let err = e
        .submit(JobId::new(), job("fake-h3-turbo", 124, 24, h3_audio(), "x"), Priority::Batch)
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidRequest);

    let cfg = EngineConfig {
        swap: true,
        ..EngineConfig::default()
    };
    let e = ready(cfg, fake).await;
    let mut h = e
        .submit(JobId::new(), job("fake-h3-turbo", 124, 24, h3_audio(), "x"), Priority::Batch)
        .await
        .unwrap();
    let evs = until_terminal(&mut h).await;
    assert!(evs.contains(&EngineEvent::Stage { name: "load" }));
    assert!(matches!(evs.last(), Some(EngineEvent::Finished(_))));
    let pool = e.pool();
    assert_eq!(pool.model_state(&"fake-h3-turbo".into()), Some(Residency::Resident));
    assert_eq!(pool.model_state(&"fake-wan".into()), Some(Residency::Unloaded));
    // And back.
    assert!(e
        .submit(JobId::new(), wan("w"), Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await
        .is_ok());
    assert_eq!(e.pool().model_state(&"fake-h3-turbo".into()), Some(Residency::Unloaded));
}

#[tokio::test]
async fn tiers_and_recipes_in_the_capability_table() {
    let e = ready(EngineConfig::default(), fast_fake()).await;
    let caps = e.caps();
    let id = |f, t| caps.tier(f, t).map(|m| m.as_str().to_owned());
    assert_eq!(id(Family::H3, Tier::Max).as_deref(), Some("fake-h3-max"));
    assert_eq!(id(Family::H3, Tier::Turbo).as_deref(), Some("fake-h3-turbo"));
    assert_eq!(id(Family::Ltx2, Tier::Max).as_deref(), Some("fake-ltx-pro"));
    assert_eq!(id(Family::Ltx2, Tier::Turbo).as_deref(), Some("fake-ltx-turbo"));
    assert_eq!(caps.resolve("h3-turbo").unwrap().id.as_str(), "fake-h3-turbo");
    assert_eq!(caps.resolve("ltx-pro").unwrap().id.as_str(), "fake-ltx-pro");
    let aliases = caps.tier_aliases();
    assert!(aliases.contains(&("h3-max".to_owned(), ModelId::new("fake-h3-max"))));
    for f in [Family::H3, Family::Ltx2] {
        for t in [Tier::Max, Tier::Turbo] {
            let via_protocol = fastvideo_protocol::resolve_tier(f, t, caps.models()).unwrap();
            assert_eq!(Some(&via_protocol.id), caps.tier(f, t));
        }
    }
    let max = caps.recipe(&"fake-h3-max".into()).unwrap();
    let turbo = caps.recipe(&"fake-h3-turbo".into()).unwrap();
    assert!(max.steps > turbo.steps, "max runs the full step count");
    // §0.5: Sol-H3 4-step serves the tau-ladder profile.
    let sol = caps.recipe(&"fake-sol-h3".into()).unwrap();
    assert_eq!(sol.profile.as_deref(), Some(SOL_H3_4STEP_PROFILE));
    assert_eq!(caps.get(&"fake-sol-h3".into()).unwrap().recipe.as_deref(), Some("sol-h3"));

    // Overrides rebind a tier.
    let mut ov = BTreeMap::new();
    ov.insert("h3-turbo".to_owned(), ModelId::new("fake-sol-h3"));
    let cfg = EngineConfig {
        tier_overrides: ov,
        ..EngineConfig::default()
    };
    let e = start(cfg, vec![fast_fake()]);
    assert_eq!(e.caps().resolve("h3-turbo").unwrap().id.as_str(), "fake-sol-h3");
    assert_eq!(e.caps().get(&"fake-h3-turbo".into()).unwrap().tier, None);

    // A bad override is a start error.
    let mut ov = BTreeMap::new();
    ov.insert("h3-turbo".to_owned(), ModelId::new("fake-wan"));
    let bad = fastvideo_engine_service::EngineService::start(
        EngineConfig {
            tier_overrides: ov,
            ..EngineConfig::default()
        },
        vec![Box::new(fastvideo_engine_service::FakeBackend::new(fast_fake()))],
    );
    assert!(bad.is_err());
}

#[tokio::test]
async fn drain_cancels_queued_and_lets_running_finish() {
    let fake = FakeConfig {
        timing: FakeTiming {
            step: Duration::from_millis(30),
            ..FakeTiming::default()
        },
        ..fast_fake()
    };
    let e = ready(EngineConfig::default(), fake).await;
    let mut a = e.submit(JobId::new(), wan("a"), Priority::Batch).await.unwrap();
    until_started(&mut a).await;
    let mut b = e.submit(JobId::new(), wan("b"), Priority::Batch).await.unwrap();
    tokio::time::timeout(T, e.drain(Duration::from_secs(10)))
        .await
        .unwrap();
    assert!(matches!(until_terminal(&mut a).await.last(), Some(EngineEvent::Finished(_))));
    assert_eq!(until_terminal(&mut b).await.last(), Some(&EngineEvent::Cancelled));
    let err = e
        .submit(JobId::new(), wan("late"), Priority::Batch)
        .await
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Loading);
    assert!(e.stats().draining);
}

#[tokio::test]
async fn drain_cancels_running_after_grace() {
    let clock = Arc::new(ManualClock::new());
    let e = ready(EngineConfig::default(), manual_fake(&clock)).await;
    let mut a = e.submit(JobId::new(), wan("a"), Priority::Batch).await.unwrap();
    until_started(&mut a).await;
    let mut s = e
        .open_causal_session(spec("fake-sfwan", 16, false))
        .await
        .unwrap();
    s.set_prompt("p");
    sleepers(&clock, 1).await;
    let clock2 = clock.clone();
    let tick = tokio::spawn(async move {
        // Nothing advances during the grace; afterwards the step in flight
        // completes and sees the cancel.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _d = Driver::new(&clock2);
        tokio::time::sleep(Duration::from_secs(3)).await;
    });
    tokio::time::timeout(T, e.drain(Duration::from_millis(50)))
        .await
        .unwrap();
    let av = until_terminal(&mut a).await;
    assert_eq!(av.last(), Some(&EngineEvent::Cancelled), "{av:?}");
    assert!(tokio::time::timeout(T, s.next_block()).await.unwrap().is_none());
    tick.abort();
}

/// Fast boot B: with a background warm-up the engine reports ready as soon
/// as the weights are resident; a job that arrives during the warm-up
/// cancels the run in flight at its next step and runs first; the warm-up
/// never shares the executor with a job; it resumes when the executor is
/// idle; and the first job's output equals the same job's output after the
/// warm-up (same seed, same bytes).
#[tokio::test]
async fn background_warmup_is_ready_first_and_yields_to_jobs() {
    let clock = Arc::new(ManualClock::new());
    let journal = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let fake = FakeConfig {
        warmup: Some(FakeWarmup {
            runs: vec!["i2v".into(), "t2v".into()],
            steps: 10,
            step: Duration::from_secs(1),
        }),
        journal: Some(journal.clone()),
        ..manual_fake(&clock)
    }
    .with_models(&["fake-wan"]);
    let e = start(EngineConfig::default(), vec![fake]);
    assert_eq!(tokio::time::timeout(T, e.wait_ready()).await.unwrap(), Readiness::Ready);
    assert_eq!(e.warmup(), "warming", "ready before the warm-up ran");

    // The warm-up's first run is on the executor (parked on its first step).
    sleepers(&clock, 1).await;
    assert_eq!(
        e.pool().entries().map(|p| p.warmup).collect::<Vec<_>>(),
        vec![Warmup::Running]
    );
    let mut first = e.submit_frames(JobId::new(), wan("first"), Priority::Batch).await.unwrap();
    // The run in flight finishes its step, sees the cancel and yields.
    clock.advance(Duration::from_secs(1));
    let drive = Driver::new(&clock);
    let first = until_terminal(&mut first).await;
    let EngineEvent::Finished(first) = first.last().unwrap().clone() else {
        panic!("first job: {first:?}")
    };
    // Idle again: the warm-up resumes and completes.
    let warm = tokio::time::timeout(T, async {
        while e.warmup() != "warm" {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(warm.is_ok(), "the warm-up never completed");
    assert_eq!(e.readiness(), Readiness::Ready);
    let after = e
        .submit_frames(JobId::new(), wan("first"), Priority::Batch)
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    drop(drive);
    assert_eq!(after.frames, first.frames, "a job during warm-up renders what a warm job renders");
    assert_eq!(after.audio, first.audio);

    let j = journal.lock().unwrap().clone();
    // Never two things on the executor at once: strict begin/end pairs.
    for pair in j.chunks(2) {
        let (b, en) = (&pair[0], &pair[1]);
        assert!(b.starts_with("begin ") && en.starts_with("end ") && b[6..] == en[4..], "overlap in {j:?}");
    }
    let order: Vec<&str> = j.iter().filter(|l| l.starts_with("begin ")).map(|l| &l[6..]).collect();
    assert_eq!(
        order,
        [
            "warmup fake-wan i2v",    // cancelled by the job
            "generate fake-wan first", // the job runs first
            "warmup fake-wan i2v",    // retried while idle
            "warmup fake-wan t2v",
            "generate fake-wan first",
        ]
    );
}

/// Without a configured warm-up nothing runs after the load and the state is `off`.
#[tokio::test]
async fn no_warmup_reports_off() {
    let e = ready(EngineConfig::default(), fast_fake().with_models(&["fake-wan"])).await;
    assert_eq!(e.warmup(), "off");
}
