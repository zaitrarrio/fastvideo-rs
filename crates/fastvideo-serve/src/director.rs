//! The fal WMA director mount (design §5.6, WP-14): the `DirectorEngine`
//! seam over `EngineService` clip sessions and causal sessions
//! (docs/serve/director-causal.md), and `DirectorConfig` from
//! `[director]` + `[protocols]`, on the process's shared WebRTC host
//! (`crate::rtc`, from `[webrtc]`).
//!
//! Built with features `fal` + `webrtc`; its routes are merged into the
//! fal router (so `/fal/proxy` reaches `/wma/*` too).

use std::sync::Arc;
use std::time::Duration;

use fastvideo_engine_service::{CancelToken, CausalControl, CausalSession, ClipBuild, ClipSession, EngineService};
use fastvideo_fal::director::{
    ChunkBuild, ChunkOutput, DirectorClips, DirectorConfig, DirectorEngine, DirectorService, DirectorStream, StreamBlock,
};
use fastvideo_media::video::EncoderBackend;
use fastvideo_protocol::{ApiError, JobId, ModelCaps, SessionSpec};
#[cfg(test)]
use fastvideo_webrtc::host::HostConfig;
use fastvideo_webrtc::host::RtcHost;

use crate::adapters::MountCfg;
use crate::config::Config;

/// `DirectorEngine` over the engine's clip sessions.
pub struct EngineDirector(pub EngineService);

/// One clip session; `close` releases it even while a build is in flight
/// and cancels the builds it queued (a queued one is dropped, a running one
/// ends at its next denoise step), so the next session on the executor does
/// not wait behind a stopped session's chunks.
pub struct Clips {
    caps: ModelCaps,
    spec: SessionSpec,
    session: tokio::sync::RwLock<Option<ClipSession>>,
    inflight: std::sync::Mutex<Vec<(JobId, CancelToken)>>,
}

impl Clips {
    fn new(s: ClipSession) -> Self {
        Self {
            caps: s.caps().clone(),
            spec: s.spec().clone(),
            session: tokio::sync::RwLock::new(Some(s)),
            inflight: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn inflight(&self) -> std::sync::MutexGuard<'_, Vec<(JobId, CancelToken)>> {
        self.inflight.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[async_trait::async_trait]
impl DirectorClips for Clips {
    fn caps(&self) -> &ModelCaps {
        &self.caps
    }
    fn spec(&self) -> &SessionSpec {
        &self.spec
    }
    async fn build(&self, c: ChunkBuild) -> Result<ChunkOutput, ApiError> {
        let b = ClipBuild {
            prompt: c.prompt,
            negative_prompt: None,
            seed: c.seed,
            seconds: Some(c.seconds),
            frames: None,
            canvas: c.canvas,
            first_frame: c.first_frame,
            last_frame: c.last_frame,
            audio_drive: None,
        };
        let job = {
            let g = self.session.read().await;
            let s = g.as_ref().ok_or_else(|| ApiError::internal("the clip session is closed"))?;
            s.build(b).await?
        };
        let id = job.id;
        self.inflight().push((id, job.cancel.clone()));
        let out = job.wait().await;
        self.inflight().retain(|(j, _)| *j != id);
        let out = out?;
        Ok(ChunkOutput { frames: out.frames.unwrap_or_default(), audio: out.audio })
    }
    async fn close(&self) {
        let tokens: Vec<CancelToken> = self.inflight().drain(..).map(|(_, t)| t).collect();
        for t in tokens {
            t.cancel();
        }
        if let Some(s) = self.session.write().await.take() {
            s.close();
        }
    }
}

/// A causal rollout (LongLive / SF-Wan) under the engine's exclusive
/// executor lease (docs/serve/director-causal.md). `close` ends it through
/// the control handle, so it works while `next_block` holds the session.
pub struct Stream {
    caps: ModelCaps,
    spec: SessionSpec,
    control: CausalControl,
    session: tokio::sync::Mutex<Option<CausalSession>>,
}

impl Stream {
    pub fn new(s: CausalSession) -> Self {
        Self { caps: s.caps().clone(), spec: s.spec().clone(), control: s.control(), session: tokio::sync::Mutex::new(Some(s)) }
    }
}

#[async_trait::async_trait]
impl DirectorStream for Stream {
    fn caps(&self) -> &ModelCaps {
        &self.caps
    }
    fn spec(&self) -> &SessionSpec {
        &self.spec
    }
    fn set_prompt(&self, prompt: &str) -> u64 {
        self.control.set_prompt(prompt)
    }
    fn set_seed(&self, seed: u64) {
        // The rollout draws its noise per reset: restart it with the seed.
        self.control.set_seed(seed);
        self.control.reset();
    }
    fn set_paused(&self, paused: bool) {
        self.control.set_paused(paused);
    }
    async fn next_block(&self) -> Option<Result<StreamBlock, ApiError>> {
        let mut g = self.session.lock().await;
        let r = g.as_mut()?.next_block().await;
        if r.is_none() {
            // Ended: release the session (and its lease) now.
            g.take();
        }
        Some(r?.map(|b| StreamBlock {
            index: b.index,
            prompt_version: b.prompt_version,
            frames: b.frames,
            block_ms: b.stats.block_ms,
            recache_ms: b.stats.recache_ms,
        }))
    }
    async fn close(&self) {
        self.control.close();
    }
}

#[async_trait::async_trait]
impl DirectorEngine for EngineDirector {
    async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn DirectorClips>, ApiError> {
        let s = self.0.open_clip_session(spec).await?;
        Ok(Arc::new(Clips::new(s)))
    }
    async fn open_stream(&self, spec: SessionSpec) -> Result<Arc<dyn DirectorStream>, ApiError> {
        let s = self.0.open_causal_session(spec).await?;
        Ok(Arc::new(Stream::new(s)))
    }
}

/// `[director] encoder` → backend (`auto`, normally resolved at startup by
/// [`crate::encoders::resolve`], probes here).
fn h264_backend(s: &str) -> EncoderBackend {
    match s {
        "openh264" => EncoderBackend::OpenH264,
        "cpu-test-x264" => EncoderBackend::CpuTestX264,
        "auto" => h264_backend(&crate::encoders::auto_selection().1.director),
        _ => EncoderBackend::Nvenc,
    }
}

/// `DirectorConfig` from the server config.
pub fn director_config(c: &Config, m: &MountCfg, host: &RtcHost) -> DirectorConfig {
    let d = &c.director;
    DirectorConfig {
        // The H3-schema apps (`minimax/h3-*`, and `owner/alias` apps such as
        // `fastvideo/ltx-turbo` or `fastvideo/longlive`); fal's LTX and Wan
        // family apps have no director. A causal model's app runs the causal
        // director (docs/serve/director-causal.md).
        apps: crate::adapters::fal_config(m).apps.into_iter().filter(|a| a.kind().director()).collect(),
        ice_servers: host.ice_servers().to_vec(),
        chunk_seconds: d.chunk_seconds,
        max_session_seconds: (d.max_session_seconds > 0).then_some(d.max_session_seconds),
        buffer_chunks: d.buffer_chunks.max(1),
        h264: h264_backend(&d.encoder),
        video_bitrate: (d.video_bitrate > 0).then_some(d.video_bitrate),
        vp8_fallback: d.vp8_fallback,
        ingest: fastvideo_fal::FalConfig::default().ingest,
        work_dir: c.server.state_dir.join("director"),
        heartbeat_timeout: Duration::from_secs(15),
        causal_chunk_blocks: d.causal_chunk_blocks.max(1),
        causal_lead_seconds: d.causal_lead_seconds.max(0.5),
        ..DirectorConfig::default()
    }
}

/// The director service on the shared WebRTC host (`crate::rtc`).
pub fn build(c: &Config, m: &MountCfg, engine: EngineService, host: RtcHost) -> Arc<DirectorService> {
    let cfg = director_config(c, m, &host);
    tracing::info!(apps = cfg.apps.len(), udp = ?host.udp_addr(), tcp = ?host.tcp_addr(), "fal director mounted");
    DirectorService::new(cfg, host, Arc::new(EngineDirector(engine)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stopped session's in-flight chunk must not keep the executor busy:
    /// `close` cancels it (WP-18 GPU E2E: a stopped director session's
    /// anchored chunk delayed the next session's first chunk by ~70 s).
    #[test]
    fn close_cancels_the_inflight_build() {
        use fastvideo_protocol::{AudioTrack, Continuity, TrackSet, VideoTrack};
        let mut c = Config::default();
        c.engine.fake.step_ms = 2_000;
        let engine = crate::app::build_engine(&c).unwrap();
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let spec = SessionSpec {
                model: "fake-h3-turbo".into(),
                tracks: TrackSet {
                    video: VideoTrack { name: "main_video".into(), width: 64, height: 32, fps: 24 },
                    audio: Some(AudioTrack { name: "main_audio".into(), rate: 48_000, channels: 2 }),
                },
                canvas: (64, 32),
                fps: 24,
                continuity: Continuity::HardCut,
                max_seconds: None,
                seed: Some(7),
            };
            let d = EngineDirector(engine);
            let clips = loop {
                match d.open(spec.clone()).await {
                    Ok(c) => break c,
                    Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
                }
            };
            let c2 = clips.clone();
            let build = tokio::spawn(async move {
                c2.build(ChunkBuild { prompt: "p".into(), seconds: 5.0, ..ChunkBuild::default() }).await
            });
            tokio::time::sleep(Duration::from_millis(300)).await;
            let t0 = std::time::Instant::now();
            clips.close().await;
            let r = tokio::time::timeout(Duration::from_secs(5), build).await.expect("build ends").unwrap();
            let e = r.expect_err("the build is cancelled");
            assert!(fastvideo_engine_service::cancel::is_cancel(&e), "{e:?}");
            assert!(t0.elapsed() < Duration::from_secs(5));
        });
    }

    #[test]
    fn config_maps() {
        let mut c = Config::default();
        c.director.max_session_seconds = 600;
        c.director.encoder = "openh264".into();
        let m = crate::app::mount_cfg(&c);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
            let d = director_config(&c, &m, &host);
            assert_eq!(d.apps.len(), fastvideo_fal::DEFAULT_APPS.len());
            assert_eq!(d.max_session_seconds, Some(600));
            assert_eq!(d.h264, EncoderBackend::OpenH264);
            assert_eq!(d.chunk_seconds, 10.0);
            assert!(d.vp8_fallback);
            host.shutdown().await;
        });
    }
}
