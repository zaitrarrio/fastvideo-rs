//! WP-15 acceptance on the fake engine: the causal block loop under an
//! exclusive lease, prompt switches at block boundaries, the Reactor causal
//! command set, the adaptive pacer and TTFF phases (design §5.4, §5.7).

use std::time::Duration;

use fastvideo_engine_service::stream::{
    spawn_causal_pacer, CausalCommand, CausalPacerConfig, CausalReply,
};
use fastvideo_engine_service::{
    EngineConfig, EngineEvent, EngineService, FakeBackend, FakeConfig, FakeModel, FakeTiming,
    Mp4Mode, Priority, Readiness,
};
use fastvideo_media::pacer::VideoOut;
use fastvideo_protocol::{
    AudioPlan, Continuity, EndReason, JobId, ModelId, PostProcess, ResolvedJob, SamplingOverrides,
    SessionSpec, Task, TrackSet,
};

const T: Duration = Duration::from_secs(20);

async fn engine(step_ms: u64) -> EngineService {
    let cfg = FakeConfig {
        timing: FakeTiming {
            step: Duration::from_millis(step_ms),
            ..FakeTiming::default()
        },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    };
    let e = EngineService::start(EngineConfig::default(), vec![Box::new(FakeBackend::new(cfg))]).unwrap();
    assert_eq!(tokio::time::timeout(T, e.wait_ready()).await.unwrap(), Readiness::Ready);
    e
}

fn spec() -> SessionSpec {
    let caps = FakeModel::sf_wan().caps;
    SessionSpec {
        model: ModelId::new("fake-sfwan"),
        tracks: TrackSet::for_model(&caps, (64, 32), 16, ("main_video", "main_audio"), 1, false),
        canvas: (64, 32),
        fps: 16,
        continuity: Continuity::HardCut,
        max_seconds: None,
        seed: Some(7),
    }
}

fn wan_job() -> ResolvedJob {
    ResolvedJob {
        model: "fake-wan".into(),
        task: Task::T2V,
        prompt: "batch".into(),
        negative_prompt: String::new(),
        seed: 1,
        width: 64,
        height: 32,
        num_frames: 49,
        fps: 16,
        keyframes: vec![],
        references: vec![],
        audio_in: None,
        audio: AudioPlan::None,
        post: PostProcess::default(),
        sampling: SamplingOverrides::default(),
        tier: None,
        recipe: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_reactor_causal_command_set() {
    let e = engine(1).await;
    let s = e.open_causal_session(spec()).await.unwrap();
    let c = s.control();
    let CausalReply::StateUpdate(st) = c.apply(CausalCommand::GetState) else { panic!() };
    assert_eq!((st.prompt.as_str(), st.paused, st.seed, st.block_index), ("", false, 7, 0));
    assert_eq!(
        c.apply(CausalCommand::SetPrompt { prompt: "  ".into() }),
        CausalReply::CommandError {
            command: "set_prompt".into(),
            reason: "The prompt is empty.".into()
        }
    );
    let CausalReply::StateUpdate(st) = c.apply(CausalCommand::SetPrompt { prompt: "a fox".into() }) else { panic!() };
    assert_eq!(st.prompt, "a fox");
    let CausalReply::StateUpdate(st) = c.apply(CausalCommand::SetSeed { seed: 9 }) else { panic!() };
    assert_eq!(st.seed, 9);
    let CausalReply::StateUpdate(st) = c.apply(CausalCommand::SetPaused { paused: true }) else { panic!() };
    assert!(st.paused);
    // Wire shape: {"type":"state_update","data":{prompt,paused,seed,block_index,unique_fps}}.
    let v = serde_json::to_value(c.apply(CausalCommand::GetState)).unwrap();
    assert_eq!(v["type"], "state_update");
    let mut keys: Vec<String> = v["data"].as_object().unwrap().keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["block_index", "paused", "prompt", "seed", "unique_fps"]);
    let cmd: CausalCommand = serde_json::from_value(serde_json::json!({"type": "set_prompt", "data": {"prompt": "x"}})).unwrap();
    assert_eq!(cmd.name(), "set_prompt");
    s.close();
    assert!(matches!(c.apply(CausalCommand::GetState), CausalReply::CommandError { .. }));
}

#[tokio::test(flavor = "multi_thread")]
async fn prompt_switches_land_at_block_boundaries_and_reset_restarts() {
    let e = engine(2).await;
    let mut s = e.open_causal_session(spec()).await.unwrap();
    assert_eq!(s.set_prompt("one"), 1);
    let mut seen = Vec::new();
    for _ in 0..3 {
        let b = tokio::time::timeout(T, s.next_block()).await.unwrap().unwrap().unwrap();
        seen.push((b.index, b.prompt_version));
    }
    assert_eq!(s.set_prompt("two"), 2);
    // Blocks already generated or in flight keep version 1; every later one is 2.
    let mut switched = false;
    for _ in 0..8 {
        let b = tokio::time::timeout(T, s.next_block()).await.unwrap().unwrap().unwrap();
        assert!(b.prompt_version >= if switched { 2 } else { 1 });
        switched |= b.prompt_version == 2;
        assert_eq!(b.frames.len(), 12);
        seen.push((b.index, b.prompt_version));
    }
    assert!(switched);
    let idx: Vec<u64> = seen.iter().map(|x| x.0).collect();
    assert!(idx.windows(2).all(|w| w[1] == w[0] + 1), "{idx:?}");
    s.reset();
    loop {
        let b = tokio::time::timeout(T, s.next_block()).await.unwrap().unwrap().unwrap();
        if b.reset {
            assert_eq!(b.index, 0);
            break;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_lease_is_exclusive_and_batch_waits() {
    let e = engine(2).await;
    let s = e.open_causal_session(spec()).await.unwrap();
    s.set_prompt("x");
    // One stream session per executor, Starting counts busy.
    assert!(e.open_causal_session(spec()).await.is_err());
    let mut h = e.submit(JobId::new(), wan_job(), Priority::Batch).await.unwrap();
    let first = tokio::time::timeout(T, h.events.recv()).await.unwrap().unwrap();
    assert!(matches!(first, EngineEvent::Queued { .. }));
    tokio::time::sleep(Duration::from_millis(200)).await;
    while let Ok(ev) = h.events.try_recv() {
        assert!(!matches!(ev, EngineEvent::Started), "batch ran under the causal lease");
    }
    s.close();
    let out = tokio::time::timeout(T, h.wait()).await.unwrap();
    assert!(out.is_ok());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_causal_pacer_ticks_and_reports_ttff_and_unique_fps() {
    // 4 steps x 20 ms = 80 ms per 12-frame block: ~150 frames/s generated,
    // so the pacer plays at the 16 fps ceiling and drops the excess.
    let e = engine(20).await;
    let s = e.open_causal_session(spec()).await.unwrap();
    let c = s.control();
    c.set_prompt("a lighthouse");
    let mut paced = spawn_causal_pacer(s, CausalPacerConfig::for_spec(&spec())).unwrap();
    let mut rtp = Vec::new();
    for _ in 0..24 {
        let t = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().unwrap();
        assert!(t.audio.is_none(), "SF-Wan is video-only");
        assert!(!matches!(t.video, VideoOut::Nothing));
        rtp.push(t.video_rtp);
    }
    c.mark_first_frame_sent();
    assert!(*paced.first_frame.borrow());
    assert!(rtp.windows(2).all(|w| w[1] > w[0]));
    let ttff = c.ttff();
    assert!(ttff.load_ms.is_some() && ttff.first_block_ms.is_some() && ttff.transport_ms.is_some());
    assert!(ttff.total_ms.unwrap() >= ttff.first_block_ms.unwrap());
    let st = paced.stats.borrow().clone();
    assert!(st.effective_fps > 4.0 && st.effective_fps <= 16.0, "{st:?}");
    assert!(c.stats().unique_fps > 0.0);
    assert!(c.state().block_index > 0);
    // Closing the control ends the pacer (the session ends).
    c.close();
    let end = tokio::time::timeout(T, paced.task).await;
    assert!(end.is_ok());
    assert_eq!(paced.stats.borrow().ended, Some(EndReason::Stopped));
}

#[tokio::test(flavor = "multi_thread")]
async fn max_seconds_ends_the_causal_stream_in_video_time() {
    let e = engine(1).await;
    let mut sp = spec();
    sp.max_seconds = Some(1);
    let s = e.open_causal_session(sp.clone()).await.unwrap();
    s.set_prompt("x");
    let mut paced = spawn_causal_pacer(s, CausalPacerConfig::for_spec(&sp)).unwrap();
    let mut n = 0;
    while tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().is_some() {
        n += 1;
    }
    // Video time follows the adaptive rate (the RTP clock): playout eases in
    // from the 4 fps floor, so 1 s of video is at most 16 ticks.
    assert!((4..=17).contains(&n), "{n} ticks for 1 s of video");
    let vs = paced.stats.borrow().video_seconds;
    assert!((1.0..1.3).contains(&vs), "{vs}");
    assert_eq!(paced.stats.borrow().ended, Some(EndReason::SessionLimit));
    // The lease is released: a new session opens.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(e.open_causal_session(spec()).await.is_ok());
}
