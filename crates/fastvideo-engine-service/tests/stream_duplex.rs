//! Duplex sessions on the loopback echo (design §5.11): admission (duplex
//! models only, one session at a time, the context checked), input frames
//! from the rings shown with the overlay at the session canvas and newest
//! first, the microphone played back, pause, the waiting card before
//! input, and the session limit.

use std::time::Duration;

use fastvideo_engine_service::stream::{
    read_echo_index, DuplexCommand, DuplexControl, DuplexReply, EchoBackend, PacedStream, Tick, ECHO_BORDER,
    ECHO_MODEL,
};
use fastvideo_engine_service::{EngineConfig, EngineService, FakeBackend, FakeConfig, Mp4Mode, Readiness};
use fastvideo_media::pacer::VideoOut;
use fastvideo_protocol::{
    Continuity, DuplexSpec, EndReason, ErrorKind, InputAudio, InputFrame, ModelId, Pcm, RgbFrame, SessionContext,
    SessionSpec, TrackSet,
};

const T: Duration = Duration::from_secs(20);

async fn engine() -> EngineService {
    let fake = FakeBackend::new(FakeConfig { mp4: Mp4Mode::Off, ..FakeConfig::default() });
    let e = EngineService::start(EngineConfig::default(), vec![Box::new(fake), Box::new(EchoBackend)]).unwrap();
    assert_eq!(tokio::time::timeout(T, e.wait_ready()).await.unwrap(), Readiness::Ready);
    e
}

fn spec(model: &str, channels: Option<u8>, max_seconds: Option<u32>) -> DuplexSpec {
    let caps = fastvideo_engine_service::echo_caps();
    let tracks = TrackSet::for_model(&caps, (320, 180), 30, ("video", "audio"), channels.unwrap_or(1), false);
    let tracks = if channels.is_none() { TrackSet { audio: None, ..tracks } } else { tracks };
    DuplexSpec {
        session: SessionSpec {
            model: ModelId::new(model),
            tracks,
            canvas: (320, 180),
            fps: 30,
            continuity: Continuity::HardCut,
            max_seconds,
            seed: None,
        },
        context: SessionContext { scene: Some("a kitchen".into()), persona: None },
    }
}

fn input(rgb: [u8; 3], pts: u64) -> InputFrame {
    InputFrame { frame: RgbFrame::solid(640, 360, rgb, 0), pts_us: pts, source: (1280, 720) }
}

/// Samples of the 0.5 test tone in a tick's audio.
fn tone_samples(t: &Tick) -> usize {
    t.audio.as_ref().unwrap().iter().filter(|x| (**x - 0.5).abs() < 1e-6).count()
}

/// The centre pixel of a fresh frame (`None` for a repeat).
fn fresh_centre(t: &Tick) -> Option<[u8; 3]> {
    match &t.video {
        VideoOut::Fresh(f) => f.pixel(160, 120),
        _ => None,
    }
}

/// Pauses the session and waits until the pause has taken hold: the worker
/// reads the flag once per tick, so the tick in flight when it was set
/// (index `frames_out` at that moment) may still take an input frame; every
/// later one cannot. Returns the ticks received up to and including that
/// one, so the caller can push input that must not be shown yet.
async fn pause(ctl: &DuplexControl, paced: &mut PacedStream) -> Vec<Tick> {
    let DuplexReply::StateUpdate(p) = ctl.apply(DuplexCommand::SetPaused { paused: true }) else { panic!() };
    assert!(p.paused);
    let mut seen = Vec::new();
    loop {
        let t = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().expect("the session ended");
        let done = t.index >= p.frames_out;
        seen.push(t);
        if done {
            return seen;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn echo_shows_the_newest_input_with_the_overlay_and_plays_the_mic_back() {
    let e = engine().await;
    let s = e.open_duplex_session(spec(ECHO_MODEL, Some(2), None)).await.unwrap();
    assert_eq!(s.duplex_caps().input.video.as_ref().unwrap().width, 640);
    // One session per executor.
    let busy = e.open_duplex_session(spec(ECHO_MODEL, Some(2), None)).await.unwrap_err();
    assert_eq!(busy.kind, ErrorKind::Conflict);
    let (ctl, mut paced) = s.start().unwrap();
    // Before any input: the grey waiting card, with the overlay.
    let t = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().unwrap();
    let VideoOut::Fresh(f) = &t.video else { panic!("{:?}", t.video) };
    assert_eq!((f.width, f.height), (320, 180));
    assert_eq!(f.pixel(1, 1), Some(ECHO_BORDER));
    assert_eq!(f.pixel(160, 120), Some([64, 64, 64]));
    assert_eq!(t.audio.as_ref().map(Vec::len), Some(1600 * 2), "48000/30 stereo samples");
    let rings = ctl.input().clone();
    // The mic comes back: 100 ms of a mono tone at 0.5, 4800 samples upmixed
    // to stereo, in consecutive ticks once the worker has drained it (the
    // ticks already queued before the push carry silence).
    rings.audio.push(0, InputAudio { pcm: Pcm::new(48_000, 1, vec![0.5f32; 4800]), pts_us: 0 });
    let mut audio_hits = 0usize;
    for _ in 0..60 {
        let t = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().unwrap();
        let n = tone_samples(&t);
        if n == 0 && audio_hits > 0 {
            break;
        }
        audio_hits += n;
    }
    assert_eq!(audio_hits, 9600, "ticks dropped: {}", paced.ticks.dropped());
    // Three frames arrive at once: only the newest is shown. The three
    // pushes are separate ring operations and the worker takes the newest
    // frame on every tick, so the burst is pushed while the session is
    // paused (the worker leaves the ring alone) and shown on resume.
    for t in pause(&ctl, &mut paced).await {
        assert_eq!(fresh_centre(&t), Some([64, 64, 64]), "still the waiting card");
    }
    rings.video.push(1, input([255, 0, 0], 1));
    rings.video.push(2, input([0, 255, 0], 2));
    rings.video.push(3, input([10, 20, 250], 3));
    let DuplexReply::StateUpdate(p) = ctl.apply(DuplexCommand::SetPaused { paused: false }) else { panic!() };
    assert!(!p.paused);
    let mut shown = None;
    for _ in 0..30 {
        let t = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().unwrap();
        if let VideoOut::Fresh(f) = &t.video {
            if f.pixel(160, 120) != Some([64, 64, 64]) {
                shown = Some((t.index, f.clone()));
                break;
            }
        }
    }
    let (index, f) = shown.expect("an input frame was shown");
    assert_eq!(f.pixel(160, 120), Some([10, 20, 250]), "the newest frame, scaled to the canvas");
    assert_eq!(f.pixel(0, 90), Some(ECHO_BORDER));
    assert_eq!(read_echo_index(&f), Some(index));
    let st = ctl.state();
    assert_eq!((st.input_frames, st.input_skipped), (1, 2));
    assert!(st.has_input && st.input_latency_ms.is_some());
    assert_eq!(st.context.scene.as_deref(), Some("a kitchen"));
    // Nothing new: the last frame repeats.
    let t = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().unwrap();
    assert!(matches!(t.video, VideoOut::Repeat(_)));
    // Paused: input is not shown.
    for t in pause(&ctl, &mut paced).await {
        assert!(matches!(t.video, VideoOut::Repeat(_)), "nothing new before the pause");
    }
    rings.video.push(4, input([200, 200, 0], 4));
    for _ in 0..5 {
        let t = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().unwrap();
        assert!(matches!(t.video, VideoOut::Repeat(_)), "paused");
    }
    assert_eq!(ctl.state().input_frames, 1, "the frame pushed while paused was not taken");
    ctl.apply(DuplexCommand::SetPaused { paused: false });
    ctl.close();
    while tokio::time::timeout(T, paced.ticks.recv()).await.unwrap().is_some() {}
    assert_eq!(paced.stats.borrow().ended, Some(EndReason::Stopped));
    // Released: a new session opens.
    tokio::time::timeout(T, async {
        loop {
            if let Ok(s) = e.open_duplex_session(spec(ECHO_MODEL, None, None)).await {
                return s;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn admission_and_the_session_limit() {
    let e = engine().await;
    // A clip model is not duplex, a duplex model is not clip.
    let err = e.open_duplex_session(spec("fake-h3-max", None, None)).await.unwrap_err();
    assert_eq!(err.param.as_deref(), Some("model"));
    let mut bad = spec(ECHO_MODEL, None, None);
    bad.context.persona = Some("x".repeat(fastvideo_protocol::CONTEXT_MAX_CHARS + 1));
    assert_eq!(e.open_duplex_session(bad).await.unwrap_err().param.as_deref(), Some("context.persona"));
    let err = e.open_clip_session(spec(ECHO_MODEL, None, None).session).await.unwrap_err();
    assert_eq!(err.param.as_deref(), Some("model"));
    // 1 s of video, then SessionLimit; video-only: no audio in the ticks.
    let s = e.open_duplex_session(spec(ECHO_MODEL, None, Some(1))).await.unwrap();
    let (_ctl, mut paced) = s.start().unwrap();
    let mut n = 0;
    while let Some(t) = tokio::time::timeout(T, paced.ticks.recv()).await.unwrap() {
        assert!(t.audio.is_none());
        n += 1;
    }
    assert_eq!(n, 30);
    assert_eq!(paced.stats.borrow().ended, Some(EndReason::SessionLimit));
}
