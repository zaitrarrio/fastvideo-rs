//! WP-12 acceptance: the fast-h3 queue-and-playout contract on the fake
//! engine (design §5.5). The scenarios mirror infinite-livestream's
//! `fast-h3/tests/test_fasth3.py` "playout loop" block: tiny clips (6 frames
//! = 0.25 s at 24 fps) played in real time.

mod common;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_engine_service::stream::{
    spawn_clip_pacer, ClipCommand, ClipEvent, ClipOutputs, ClipPacerConfig, ClipPlayer,
    ClipPlayerConfig, IdlePolicy, MediaItem, PlayOutcome, QueueName,
};
use fastvideo_engine_service::{
    BlockInput, BlockStats, CausalSpec, ClipOutput, ClipSink, DeviceInfo, EngineBackend,
    EngineConfig, EngineService, FakeBackend, FakeConfig, FakeModel, FakeTiming, LoadEvent,
    Mp4Mode, Readiness, Recipe, SessionId, StepControl,
};
use fastvideo_media::pacer::VideoOut;
use fastvideo_protocol::{
    canvas_for_aspect, Anchor, ApiError, Continuity, ErrorKind, FrameGrid, ModelCaps, ModelId,
    ResolvedJob, SessionSpec, StreamCaps, TrackSet,
};

const T: Duration = Duration::from_secs(20);
const FRAMES: u32 = 6;

/// H3-shaped (24 fps, 32 kHz stereo, t2v/i2v) with a tiny 6..48 grid.
fn tiny_h3() -> FakeModel {
    let mut m = FakeModel::h3_turbo();
    m.caps.id = ModelId::new("tiny-h3");
    m.caps.served_names = vec!["tiny-h3".into()];
    m.caps.frames = FrameGrid::new(1, 0, FRAMES, 48, FRAMES);
    m.caps.stream = Some(StreamCaps::Clip {
        min_s: FRAMES as f32 / 24.0,
        max_s: 2.0,
    });
    m.caps.tier = None;
    m
}

/// Video-only T2V clip model (AnchorLastFrame is invalid on it).
fn tiny_wan() -> FakeModel {
    let mut m = FakeModel::wan();
    m.caps.id = ModelId::new("tiny-wan");
    m.caps.served_names = vec!["tiny-wan".into()];
    m.caps.frames = FrameGrid::new(1, 0, 4, 32, 8);
    m.caps.stream = Some(StreamCaps::Clip { min_s: 0.25, max_s: 2.0 });
    m
}

fn fake(step_ms: u64) -> FakeConfig {
    FakeConfig {
        models: vec![tiny_h3(), tiny_wan(), FakeModel::sf_wan()],
        timing: FakeTiming {
            step: Duration::from_millis(step_ms),
            ..FakeTiming::default()
        },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    }
}

async fn engine_with(backend: Box<dyn EngineBackend>) -> EngineService {
    let e = EngineService::start(EngineConfig::default(), vec![backend]).unwrap();
    assert_eq!(tokio::time::timeout(T, e.wait_ready()).await.unwrap(), Readiness::Ready);
    e
}

async fn engine(step_ms: u64) -> EngineService {
    engine_with(Box::new(FakeBackend::new(fake(step_ms)))).await
}

fn spec(model: &str, continuity: Continuity, channels: Option<u8>) -> SessionSpec {
    let m = if model == "tiny-wan" { tiny_wan() } else { tiny_h3() };
    let fps = m.caps.fps.default;
    let mut tracks = TrackSet::for_model(&m.caps, (64, 32), fps, ("main_video", "main_audio"), 1, false);
    if let (Some(a), Some(ch)) = (tracks.audio.as_mut(), channels) {
        a.channels = ch;
    }
    if channels.is_none() {
        tracks.audio = None;
    }
    SessionSpec {
        model: ModelId::new(model),
        tracks,
        canvas: (64, 32),
        fps,
        continuity,
        max_seconds: None,
        seed: Some(1000),
    }
}

/// A player with its outputs split: events and media are drained into logs
/// by background tasks.
struct Rig {
    player: ClipPlayer,
    events: Arc<Mutex<Vec<ClipEvent>>>,
    media: Arc<Mutex<Vec<MediaItem>>>,
    _engine: EngineService,
}

impl Rig {
    async fn new(e: EngineService, spec: SessionSpec, cfg: ClipPlayerConfig) -> Self {
        let s = e.open_clip_session(spec).await.unwrap();
        let (player, out) = s.into_player(cfg).unwrap();
        let (events, media) = drain(out);
        Self {
            player,
            events,
            media,
            _engine: e,
        }
    }

    async fn cmd(&self, c: ClipCommand) -> Option<ClipEvent> {
        self.player.command(c).await.unwrap()
    }

    async fn enqueue(&self, prompt: &str) -> ClipEvent {
        self.cmd(ClipCommand::Enqueue {
            prompt: prompt.into(),
            metadata: "tag".into(),
            seed: None,
            seconds: None,
            position: None,
        })
        .await
        .expect("clip_queued")
    }

    fn names(&self) -> Vec<&'static str> {
        self.events.lock().unwrap().iter().map(|e| e.type_name()).collect()
    }

    fn fasth3_names(&self) -> Vec<&'static str> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.is_fasth3())
            .map(|e| e.type_name())
            .collect()
    }

    fn count(&self, name: &str) -> usize {
        self.names().iter().filter(|n| **n == name).count()
    }

    async fn until(&self, what: &str, f: impl Fn(&Rig) -> bool) {
        let t0 = std::time::Instant::now();
        while !f(self) {
            assert!(t0.elapsed() < T, "timed out waiting for {what}; events: {:?}", self.names());
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn slices(&self) -> Vec<(u32, usize, Option<Vec<f32>>)> {
        self.media
            .lock()
            .unwrap()
            .iter()
            .filter_map(|m| match m {
                MediaItem::Slice(s) => Some((s.first_frame, s.frames.len(), s.audio.clone())),
                _ => None,
            })
            .collect()
    }
}

type Log<T> = Arc<Mutex<Vec<T>>>;

fn drain(mut out: ClipOutputs) -> (Log<ClipEvent>, Log<MediaItem>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let media = Arc::new(Mutex::new(Vec::new()));
    let (e2, m2) = (events.clone(), media.clone());
    tokio::spawn(async move {
        while let Some(e) = out.events.recv().await {
            e2.lock().unwrap().push(e);
        }
    });
    tokio::spawn(async move {
        while let Some(m) = out.media.recv().await {
            m2.lock().unwrap().push(m);
        }
    });
    (events, media)
}

fn clip_id(ev: &ClipEvent) -> String {
    match ev {
        ClipEvent::ClipQueued { clip } => clip.clip_id.to_string(),
        other => panic!("not clip_queued: {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn scripted_log_matches_the_fasth3_message_sequence() {
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, Some(1)), ClipPlayerConfig::default()).await;

    // enqueue -> reply clip_queued; broadcasts queue_update, state_update.
    let q = r.enqueue("a").await;
    let ClipEvent::ClipQueued { clip } = &q else { panic!() };
    assert_eq!((clip.frames, clip.seed, clip.ready, clip.metadata.as_str()), (FRAMES, 1000, false, "tag"));
    r.until("clip_generated", |r| r.count("clip_generated") == 1).await;
    // play -> bodyless; queue_update, state_update, clip_started, state_update,
    // then clip_finished, state_update.
    assert_eq!(r.cmd(ClipCommand::Play { clip_id: String::new() }).await, None);
    r.until("clip_finished", |r| r.count("clip_finished") == 1).await;
    // Nothing else starts on its own (no autoplay).
    tokio::time::sleep(Duration::from_millis(150)).await;

    assert_eq!(
        r.fasth3_names(),
        [
            "queue_update",
            "state_update",
            "clip_generated",
            "queue_update",
            "state_update",
            "queue_update",
            "state_update",
            "clip_started",
            "state_update",
            "clip_finished",
            "state_update",
        ]
    );
    assert_eq!(r.names().iter().filter(|n| **n == "build_started").count(), 1);

    // The whole clip went out, in 3-frame lockstep slices.
    let slices = r.slices();
    assert_eq!(slices.iter().map(|s| s.1).sum::<usize>(), FRAMES as usize);
    for (_, n, audio) in &slices {
        assert_eq!(audio.as_ref().unwrap().len(), n * 2000, "48 kHz mono, 2000 samples per frame");
    }
    let ev = r.events.lock().unwrap().clone();
    let started = ev.iter().find_map(|e| match e {
        ClipEvent::ClipStarted { clip } => Some(clip.clone()),
        _ => None,
    });
    let finished = ev.iter().find_map(|e| match e {
        ClipEvent::ClipFinished { clip, seconds_sent } => Some((clip.clone(), *seconds_sent)),
        _ => None,
    });
    let started = started.unwrap();
    let (finished, sent) = finished.unwrap();
    assert!(started.ready);
    assert_eq!(started.metadata, "tag");
    assert_eq!(finished.clip_id, started.clip_id);
    assert_eq!(sent, 0.25);
    // The clip end is announced to the pacer with nothing armed (black/hold).
    let ends: Vec<_> = r
        .media
        .lock()
        .unwrap()
        .iter()
        .filter_map(|m| match m {
            MediaItem::ClipEnd { outcome, armed, .. } => Some((*outcome, *armed)),
            _ => None,
        })
        .collect();
    assert_eq!(ends, [(PlayOutcome::Finished, false)]);
    let st = r.player.state();
    assert!(!st.playing);
    assert_eq!((st.clips_played, st.playout_queued, st.generation_queued), (1, 0, 0));
    assert!(st.valid_commands.contains(&"set_canvas".to_string()));
    r.player.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn refusals_are_broadcast_and_replies_stay_bodyless() {
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, Some(1)), ClipPlayerConfig::default()).await;
    assert_eq!(r.cmd(ClipCommand::Play { clip_id: String::new() }).await, None);
    assert_eq!(r.cmd(ClipCommand::Stop).await, None);
    assert_eq!(r.cmd(ClipCommand::Pop { clip_id: "nope".into() }).await, None);
    assert_eq!(r.cmd(ClipCommand::Pop { clip_id: String::new() }).await, None);
    assert_eq!(
        r.cmd(ClipCommand::Enqueue {
            prompt: "   ".into(),
            metadata: String::new(),
            seed: None,
            seconds: None,
            position: None
        })
        .await,
        None
    );
    assert_eq!(r.cmd(ClipCommand::SetClipSeconds(99.0)).await, None);
    r.until("six refusals", |r| r.count("command_error") == 6).await;
    let errs: Vec<(String, String)> = r
        .events
        .lock()
        .unwrap()
        .iter()
        .map(|e| match e {
            ClipEvent::CommandError { command, reason } => (command.clone(), reason.clone()),
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(
        errs,
        [
            ("play".to_string(), "The playout queue is empty; `enqueue` a clip and wait for `clip_generated`.".to_string()),
            ("stop".to_string(), "No clip is playing.".to_string()),
            ("pop".to_string(), "No queued clip has id 'nope'.".to_string()),
            ("pop".to_string(), "Pass the `clip_id` of the queued clip to remove.".to_string()),
            ("enqueue".to_string(), "The prompt is empty; a clip needs one.".to_string()),
            ("set_clip_seconds".to_string(), "The clip length must be between 0.25 and 2 seconds, got 99.".to_string()),
        ]
    );
    // Reads and settings reply with their message.
    assert!(matches!(r.cmd(ClipCommand::GetState).await, Some(ClipEvent::StateUpdate(_))));
    assert!(matches!(r.cmd(ClipCommand::GetQueue).await, Some(ClipEvent::QueueUpdate { .. })));
    assert_eq!(r.cmd(ClipCommand::SetSeed(7)).await, Some(ClipEvent::SeedAccepted { seed: 7 }));
    assert_eq!(
        r.cmd(ClipCommand::SetClipSeconds(1.0)).await,
        Some(ClipEvent::ClipLengthAccepted { clip_seconds: 1.0, frames: 24 })
    );
    assert_eq!(r.player.state().seed, 7);
}

#[tokio::test(flavor = "multi_thread")]
async fn seeds_advance_and_explicit_values_apply_to_one_clip() {
    // Audience absent: nothing builds, so the queue is inspectable.
    let cfg = ClipPlayerConfig {
        audience: false,
        ..ClipPlayerConfig::default()
    };
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, Some(1)), cfg).await;
    let a = r.enqueue("a").await;
    let b = r
        .cmd(ClipCommand::Enqueue {
            prompt: "b".into(),
            metadata: String::new(),
            seed: Some(5),
            seconds: Some(1.0),
            position: Some(0),
        })
        .await
        .unwrap();
    let c = r.enqueue("c").await;
    let info = |e: &ClipEvent| match e {
        ClipEvent::ClipQueued { clip } => (clip.seed, clip.frames),
        _ => panic!(),
    };
    assert_eq!(info(&a), (1000, FRAMES));
    assert_eq!(info(&b), (5, 24));
    assert_eq!(info(&c), (1001, FRAMES));
    let Some(ClipEvent::QueueUpdate { generation, playout }) = r.cmd(ClipCommand::GetQueue).await else { panic!() };
    assert_eq!(generation.iter().map(|c| c.prompt.as_str()).collect::<Vec<_>>(), ["b", "a", "c"]);
    assert!(playout.is_empty());
    // move within the generation queue
    let Some(ClipEvent::ClipMoved { queue, position, .. }) = r
        .cmd(ClipCommand::Move { clip_id: clip_id(&c), position: 0 })
        .await
    else {
        panic!()
    };
    assert_eq!((queue, position), (QueueName::Generation, 0));
    // No build without an audience.
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(r.count("build_started"), 0);
    r.player.set_audience(true).await.unwrap();
    r.until("three builds", |r| r.count("clip_generated") == 3).await;
    let built: Vec<String> = r
        .events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| match e {
            ClipEvent::ClipGenerated { clip, build } => {
                assert!(build.build_s > 0.0 && build.clip_s > 0.0);
                Some(clip.prompt.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(built, ["c", "b", "a"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_cuts_the_clip_and_keeps_the_queue() {
    let cfg = ClipPlayerConfig {
        clip_seconds: Some(2.0),
        ..ClipPlayerConfig::default()
    };
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, Some(2)), cfg).await;
    r.enqueue("a").await;
    r.enqueue("b").await;
    r.until("two built", |r| r.player.state().playout_queued == 2).await;
    r.cmd(ClipCommand::Play { clip_id: String::new() }).await;
    r.until("first slice", |r| !r.slices().is_empty()).await;
    assert_eq!(r.cmd(ClipCommand::Stop).await, None);
    r.until("clip_stopped", |r| r.count("clip_stopped") == 1).await;
    assert_eq!(r.count("clip_finished"), 0);
    let sent: usize = r.slices().iter().map(|s| s.1).sum();
    assert!(sent < 48, "the cut clip went out only partially ({sent})");
    assert!(r.media.lock().unwrap().contains(&MediaItem::Clear));
    let st = r.player.state();
    assert_eq!((st.playout_queued, st.playing, st.clips_played), (1, false, 1));
    // Stereo slices carry 2 channels.
    for (_, n, a) in r.slices() {
        assert_eq!(a.unwrap().len(), n * 2000 * 2);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn autoplay_chains_ready_clips_without_play() {
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, Some(1)), ClipPlayerConfig::default()).await;
    assert_eq!(
        r.cmd(ClipCommand::SetAutoplay(true)).await,
        Some(ClipEvent::AutoplayAccepted { enabled: true })
    );
    r.enqueue("a").await;
    r.enqueue("b").await;
    r.until("two finished", |r| r.count("clip_finished") == 2).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    let started: Vec<String> = r
        .events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| match e {
            ClipEvent::ClipStarted { clip } => Some(clip.prompt.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(started, ["a", "b"]);
    assert_eq!(r.slices().iter().map(|s| s.1).sum::<usize>(), 2 * FRAMES as usize);
    assert_eq!(r.count("clip_started"), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_playout_reservation_pauses_builds() {
    let cfg = ClipPlayerConfig {
        playout_capacity: 1,
        ..ClipPlayerConfig::default()
    };
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, None), cfg).await;
    for p in ["a", "b", "c"] {
        r.enqueue(p).await;
    }
    r.until("one built", |r| r.count("clip_generated") == 1).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(r.count("build_started"), 1, "no build while the playout queue is full");
    assert!(r.player.state().valid_commands.contains(&"play".to_string()));
    // Playing frees the slot; the next build starts while the clip plays.
    r.cmd(ClipCommand::Play { clip_id: String::new() }).await;
    r.until("second built", |r| r.count("clip_generated") == 2).await;
    // Video-only session: slices carry no audio.
    r.until("finished", |r| r.count("clip_finished") == 1).await;
    assert!(r.slices().iter().all(|s| s.2.is_none()));
}

#[tokio::test(flavor = "multi_thread")]
async fn pop_cancels_the_build_in_flight_and_discards_it() {
    // 4 steps x 150 ms: the pop lands mid-build.
    let r = Rig::new(engine(150).await, spec("tiny-h3", Continuity::HardCut, Some(1)), ClipPlayerConfig::default()).await;
    let a = r.enqueue("a").await;
    r.until("build_started", |r| r.count("build_started") == 1).await;
    let popped = r.cmd(ClipCommand::Pop { clip_id: clip_id(&a) }).await;
    assert!(matches!(popped, Some(ClipEvent::ClipPopped { .. })));
    r.enqueue("b").await;
    r.until("b built", |r| r.count("clip_generated") == 1).await;
    let gen: Vec<String> = r
        .events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| match e {
            ClipEvent::ClipGenerated { clip, .. } => Some(clip.prompt.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(gen, ["b"]);
    assert_eq!(r.count("clip_failed"), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failing_build_reports_and_the_queue_moves_on() {
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, Some(1)), ClipPlayerConfig::default()).await;
    r.enqueue("a [fake:fail]").await;
    r.enqueue("b").await;
    r.until("b built", |r| r.count("clip_generated") == 1).await;
    let failed = r
        .events
        .lock()
        .unwrap()
        .iter()
        .find_map(|e| match e {
            ClipEvent::ClipFailed { clip, reason } => Some((clip.prompt.clone(), reason.clone())),
            _ => None,
        })
        .unwrap();
    assert_eq!(failed.0, "a [fake:fail]");
    assert!(failed.1.contains("injected failure"), "{}", failed.1);
    let Some(ClipEvent::QueueUpdate { playout, .. }) = r.cmd(ClipCommand::GetQueue).await else { panic!() };
    assert_eq!(playout.iter().map(|c| c.prompt.as_str()).collect::<Vec<_>>(), ["b"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn reset_cuts_playout_drops_queues_and_restores_defaults() {
    let cfg = ClipPlayerConfig {
        clip_seconds: Some(2.0),
        ..ClipPlayerConfig::default()
    };
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, Some(1)), cfg).await;
    r.enqueue("a").await;
    r.enqueue("b").await;
    r.until("two built", |r| r.player.state().playout_queued == 2).await;
    r.cmd(ClipCommand::SetSeed(42)).await;
    r.cmd(ClipCommand::SetAutoplay(true)).await;
    r.until("playing", |r| !r.slices().is_empty()).await;
    r.enqueue("c").await;
    let reply = r.cmd(ClipCommand::Reset).await;
    assert_eq!(
        reply,
        Some(ClipEvent::SessionReset {
            cleared_clips: 2,
            was_playing: true
        })
    );
    r.until("clip_stopped", |r| r.count("clip_stopped") == 1).await;
    let st = r.player.state();
    assert_eq!(
        (st.playing, st.generation_queued, st.playout_queued, st.seed, st.autoplay),
        (false, 0, 0, 1000, false)
    );
    assert_eq!(st.clip_seconds, 2.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_canvas_is_locked_while_clips_exist() {
    let cfg = ClipPlayerConfig {
        audience: false,
        ..ClipPlayerConfig::default()
    };
    let r = Rig::new(engine(1).await, spec("tiny-h3", Continuity::HardCut, Some(1)), cfg).await;
    assert_eq!(r.player.state().aspect, "2:1");
    let want = canvas_for_aspect(&tiny_h3().caps.canvas, 1.0, 32);
    assert_eq!(
        r.cmd(ClipCommand::SetCanvas("1:1".into())).await,
        Some(ClipEvent::CanvasAccepted {
            aspect: "1:1".into(),
            width: want.0,
            height: want.1
        })
    );
    r.enqueue("a").await;
    assert_eq!(r.cmd(ClipCommand::SetCanvas("16:9".into())).await, None);
    r.until("command_error", |r| r.count("command_error") == 1).await;
    assert_eq!(r.player.state().aspect, "1:1");
}

/// Wraps the fake and records every job it builds.
struct Recording {
    inner: FakeBackend,
    jobs: Arc<Mutex<Vec<ResolvedJob>>>,
}

impl EngineBackend for Recording {
    fn device(&self) -> DeviceInfo {
        self.inner.device()
    }
    fn caps(&self) -> Vec<ModelCaps> {
        self.inner.caps()
    }
    fn recipe(&self, m: &ModelId) -> Recipe {
        self.inner.recipe(m)
    }
    fn load(&mut self, m: &ModelId, obs: &mut dyn FnMut(LoadEvent)) -> Result<(), ApiError> {
        self.inner.load(m, obs)
    }
    fn generate(&mut self, job: &ResolvedJob, out: &mut dyn ClipSink, ctl: &StepControl) -> Result<ClipOutput, ApiError> {
        // The anchor must exist when the build reads it.
        for (_, p) in &job.keyframes {
            assert!(p.is_file(), "keyframe {} missing", p.display());
        }
        self.jobs.lock().unwrap().push(job.clone());
        self.inner.generate(job, out, ctl)
    }
    fn causal_open(&mut self, s: SessionId, spec: &CausalSpec) -> Result<(), ApiError> {
        self.inner.causal_open(s, spec)
    }
    fn causal_block(&mut self, s: SessionId, i: &BlockInput, out: &mut dyn ClipSink, ctl: &StepControl) -> Result<BlockStats, ApiError> {
        self.inner.causal_block(s, i, out, ctl)
    }
    fn causal_close(&mut self, s: SessionId) {
        self.inner.causal_close(s)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn anchor_last_frame_builds_from_the_previous_last_frame() {
    let jobs = Arc::new(Mutex::new(Vec::new()));
    let e = engine_with(Box::new(Recording {
        inner: FakeBackend::new(fake(1)),
        jobs: jobs.clone(),
    }))
    .await;
    let dir = std::env::temp_dir().join(format!("fv-anchor-test-{}", std::process::id()));
    let cfg = ClipPlayerConfig {
        session_dir: Some(dir.clone()),
        autoplay: true,
        ..ClipPlayerConfig::default()
    };
    let r = Rig::new(e, spec("tiny-h3", Continuity::AnchorLastFrame { crossfade_ms: 20 }, Some(1)), cfg).await;
    r.enqueue("a").await;
    r.enqueue("b").await;
    r.enqueue("c").await;
    r.until("three built", |r| r.count("clip_generated") == 3).await;
    let jobs = jobs.lock().unwrap().clone();
    assert_eq!(jobs.len(), 3);
    assert!(jobs[0].keyframes.is_empty(), "the first clip has nothing to anchor on");
    for j in &jobs[1..] {
        assert_eq!(j.keyframes.len(), 1);
        assert_eq!(j.keyframes[0].0, Anchor::First);
        assert!(j.keyframes[0].1.starts_with(&dir));
    }
    assert_ne!(jobs[1].keyframes[0].1, jobs[2].keyframes[0].1);
    // The PNG is the previous clip's last frame.
    let png: PathBuf = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|x| x == "png"))
        .expect("an anchor png");
    let img = image::open(&png).unwrap().to_rgb8();
    assert_eq!(img.dimensions(), (64, 32));
    r.player.close().await;
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn anchor_needs_an_image_to_video_model() {
    let e = engine(1).await;
    let s = e
        .open_clip_session(spec("tiny-wan", Continuity::AnchorLastFrame { crossfade_ms: 20 }, None))
        .await
        .unwrap();
    let err = s.into_player(ClipPlayerConfig::default()).unwrap_err();
    assert_eq!(err.param.as_deref(), Some("continuity"));
}

/// First 20 ms of each clip's audio (the fake puts a full-scale click at
/// every clip start).
async fn clip_heads(continuity: Continuity) -> Vec<f32> {
    let r = Rig::new(engine(1).await, spec("tiny-h3", continuity, Some(1)), ClipPlayerConfig {
        autoplay: true,
        ..ClipPlayerConfig::default()
    })
    .await;
    r.enqueue("a").await;
    r.enqueue("b").await;
    r.until("two finished", |r| r.count("clip_finished") == 2).await;
    r.slices()
        .iter()
        .filter(|s| s.0 == 0)
        .map(|s| s.2.as_ref().unwrap()[..960].iter().fold(0f32, |m, x| m.max(x.abs())))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn crossfade_fades_clip_edges_and_hard_cut_does_not() {
    let hard = clip_heads(Continuity::HardCut).await;
    let soft = clip_heads(Continuity::Crossfade { ms: 20 }).await;
    assert_eq!(hard.len(), 2);
    assert!(hard[1] > 0.5, "hard cut keeps the click: {hard:?}");
    // The first clip is never faded in; the second is.
    assert!(soft[0] > 0.5, "{soft:?}");
    assert!(soft[1] < hard[1] * 0.9, "{soft:?} vs {hard:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_clip_pacer_ticks_one_frame_and_48000_over_fps_samples() {
    let e = engine(1).await;
    let sp = spec("tiny-h3", Continuity::HardCut, Some(2));
    let s = e.open_clip_session(sp.clone()).await.unwrap();
    let (player, out) = s.into_player(ClipPlayerConfig::default()).unwrap();
    let mut events = out.events;
    tokio::spawn(async move { while events.recv().await.is_some() {} });
    let mut paced = spawn_clip_pacer(
        out.media,
        ClipPacerConfig {
            idle: IdlePolicy::Black,
            ..ClipPacerConfig::for_spec(&sp)
        },
    )
    .unwrap();
    assert!(!*paced.first_frame.borrow());
    player
        .command(ClipCommand::Enqueue {
            prompt: "a".into(),
            metadata: String::new(),
            seed: None,
            seconds: None,
            position: None,
        })
        .await
        .unwrap();
    while player.state().playout_queued == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    player.command(ClipCommand::Play { clip_id: String::new() }).await.unwrap();
    let mut fresh = 0;
    let mut black_after = false;
    for i in 0..(FRAMES as u64 + 12) {
        let t = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().unwrap();
        assert_eq!(t.index, i);
        assert_eq!(t.video_rtp as u64, i * 90_000 / 24);
        assert_eq!(t.audio.as_ref().unwrap().len(), 2000 * 2);
        match &t.video {
            VideoOut::Fresh(_) => fresh += 1,
            VideoOut::Repeat(f) => {
                if fresh == FRAMES && f.data.iter().all(|b| *b == 0) {
                    black_after = true;
                }
            }
            VideoOut::Nothing => panic!("ticks start at the first frame"),
        }
    }
    assert_eq!(fresh, FRAMES);
    assert!(black_after, "IdlePolicy::Black flushes to black after the clip");
    assert!(*paced.first_frame.borrow());
    let st = paced.stats.borrow().clone();
    assert_eq!(st.fresh_frames, FRAMES as u64);
    player.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn admission_one_session_per_executor_and_resident_models_only() {
    let e = engine(1).await;
    let s = e.open_clip_session(spec("tiny-h3", Continuity::HardCut, Some(1))).await.unwrap();
    let busy = e.open_clip_session(spec("tiny-h3", Continuity::HardCut, Some(1))).await.unwrap_err();
    assert_eq!(busy.kind, ErrorKind::Conflict, "{busy:?}");
    let (player, _out) = s.into_player(ClipPlayerConfig::default()).unwrap();
    player.close().await;
    // Closing the player frees the executor.
    let again = e.open_clip_session(spec("tiny-h3", Continuity::HardCut, Some(1))).await;
    assert!(again.is_ok());
    drop(again);

    // A model still loading: 503 with Retry-After.
    let slow = FakeConfig {
        timing: FakeTiming {
            load: Duration::from_millis(800),
            ..FakeTiming::default()
        },
        ..fake(1)
    };
    let e2 = EngineService::start(EngineConfig::default(), vec![Box::new(FakeBackend::new(slow))]).unwrap();
    let err = e2.open_clip_session(spec("tiny-h3", Continuity::HardCut, Some(1))).await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::Loading);
    assert_eq!(err.retry_after_s, Some(1));
}
