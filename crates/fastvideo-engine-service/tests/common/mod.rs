//! Shared helpers for the engine-service integration tests.
#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fastvideo_engine_service::{
    EngineBackend, EngineConfig, EngineEvent, EngineService, FakeBackend, FakeConfig, FakeTiming,
    JobHandle, ManualClock, Mp4Mode, Readiness,
};
use fastvideo_protocol::{
    AudioPlan, PostProcess, ResolvedJob, SamplingOverrides, Task,
};

pub const T: Duration = Duration::from_secs(20);

/// A 64x32 job (small canvas keeps tests fast).
pub fn job(model: &str, frames: u32, fps: u32, audio: AudioPlan, prompt: &str) -> ResolvedJob {
    ResolvedJob {
        model: model.into(),
        task: Task::T2V,
        prompt: prompt.into(),
        negative_prompt: String::new(),
        seed: 42,
        width: 64,
        height: 32,
        num_frames: frames,
        fps,
        keyframes: vec![],
        references: vec![],
        audio_in: None,
        audio,
        post: PostProcess::default(),
        sampling: SamplingOverrides::default(),
        tier: None,
        recipe: None,
    }
}

/// A 49-frame video-only Wan job (3 steps).
pub fn wan(prompt: &str) -> ResolvedJob {
    job("fake-wan", 49, 16, AudioPlan::None, prompt)
}

/// A fake config on a manual clock, 1 s per step, no MP4.
pub fn manual_fake(clock: &Arc<ManualClock>) -> FakeConfig {
    FakeConfig {
        clock: clock.clone(),
        timing: FakeTiming {
            step: Duration::from_secs(1),
            ..FakeTiming::default()
        },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    }
}

pub fn start(cfg: EngineConfig, fakes: Vec<FakeConfig>) -> EngineService {
    let backends: Vec<Box<dyn EngineBackend>> = fakes
        .into_iter()
        .map(|c| Box::new(FakeBackend::new(c)) as Box<dyn EngineBackend>)
        .collect();
    EngineService::start(cfg, backends).expect("engine starts")
}

/// Starts one fake backend and waits until ready.
pub async fn ready(cfg: EngineConfig, fake: FakeConfig) -> EngineService {
    let e = start(cfg, vec![fake]);
    assert_eq!(tokio::time::timeout(T, e.wait_ready()).await.unwrap(), Readiness::Ready);
    e
}

pub async fn ev(h: &mut JobHandle) -> EngineEvent {
    tokio::time::timeout(T, h.events.recv())
        .await
        .expect("event timeout")
        .expect("event stream closed")
}

pub async fn until_terminal(h: &mut JobHandle) -> Vec<EngineEvent> {
    let mut v = Vec::new();
    loop {
        let e = ev(h).await;
        let t = e.is_terminal();
        v.push(e);
        if t {
            return v;
        }
    }
}

/// Waits for `Started`, returning what came before it.
pub async fn until_started(h: &mut JobHandle) -> Vec<EngineEvent> {
    let mut v = Vec::new();
    loop {
        let e = ev(h).await;
        assert!(!e.is_terminal(), "terminal before start: {e:?}");
        if e == EngineEvent::Started {
            return v;
        }
        v.push(e);
    }
}

/// Events already delivered, without waiting.
pub fn pending(h: &mut JobHandle) -> Vec<EngineEvent> {
    let mut v = Vec::new();
    while let Ok(e) = h.events.try_recv() {
        v.push(e);
    }
    v
}

pub async fn sleepers(clock: &Arc<ManualClock>, n: usize) {
    let c = clock.clone();
    let ok = tokio::task::spawn_blocking(move || c.wait_for_sleepers(n, T))
        .await
        .unwrap();
    assert!(ok, "executor never parked on the clock");
}

/// Advances the manual clock by 1 s whenever someone sleeps on it, until
/// dropped.
pub struct Driver {
    stop: Arc<AtomicBool>,
    t: Option<std::thread::JoinHandle<()>>,
}

impl Driver {
    pub fn new(clock: &Arc<ManualClock>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let (s, c) = (stop.clone(), clock.clone());
        let t = std::thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                if c.sleepers() > 0 {
                    c.advance(Duration::from_secs(1));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        Self { stop, t: Some(t) }
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.t.take() {
            let _ = t.join();
        }
    }
}

pub fn progress_steps(evs: &[EngineEvent]) -> Vec<u32> {
    evs.iter()
        .filter_map(|e| match e {
            EngineEvent::Progress { step, .. } => Some(*step),
            _ => None,
        })
        .collect()
}

pub fn has_started(evs: &[EngineEvent]) -> bool {
    evs.contains(&EngineEvent::Started)
}
