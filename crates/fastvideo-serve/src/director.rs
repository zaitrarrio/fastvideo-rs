//! The fal WMA director mount (design §5.6, WP-14): the `DirectorEngine`
//! seam over `EngineService` clip sessions and `DirectorConfig` from
//! `[director]` + `[protocols]`, on the process's shared WebRTC host
//! (`crate::rtc`, from `[webrtc]`).
//!
//! Built with features `fal` + `webrtc`; its routes are merged into the
//! fal router (so `/fal/proxy` reaches `/wma/*` too).

use std::sync::Arc;
use std::time::Duration;

use fastvideo_engine_service::{ClipBuild, ClipSession, EngineService};
use fastvideo_fal::director::{ChunkBuild, ChunkOutput, DirectorClips, DirectorConfig, DirectorEngine, DirectorService};
use fastvideo_media::video::EncoderBackend;
use fastvideo_protocol::{ApiError, ModelCaps, SessionSpec};
#[cfg(test)]
use fastvideo_webrtc::host::HostConfig;
use fastvideo_webrtc::host::RtcHost;

use crate::adapters::MountCfg;
use crate::config::Config;

/// `DirectorEngine` over the engine's clip sessions.
pub struct EngineDirector(pub EngineService);

/// One clip session; `close` releases it even while a build is in flight.
pub struct Clips {
    caps: ModelCaps,
    spec: SessionSpec,
    session: tokio::sync::RwLock<Option<ClipSession>>,
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
        };
        let job = {
            let g = self.session.read().await;
            let s = g.as_ref().ok_or_else(|| ApiError::internal("the clip session is closed"))?;
            s.build(b).await?
        };
        let out = job.wait().await?;
        Ok(ChunkOutput { frames: out.frames.unwrap_or_default(), audio: out.audio })
    }
    async fn close(&self) {
        if let Some(s) = self.session.write().await.take() {
            s.close();
        }
    }
}

#[async_trait::async_trait]
impl DirectorEngine for EngineDirector {
    async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn DirectorClips>, ApiError> {
        let s = self.0.open_clip_session(spec).await?;
        Ok(Arc::new(Clips { caps: s.caps().clone(), spec: s.spec().clone(), session: tokio::sync::RwLock::new(Some(s)) }))
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
        apps: crate::adapters::fal_config(m).apps,
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
            assert_eq!(d.apps.len(), 3);
            assert_eq!(d.max_session_seconds, Some(600));
            assert_eq!(d.h264, EncoderBackend::OpenH264);
            assert_eq!(d.chunk_seconds, 10.0);
            assert!(d.vp8_fallback);
            host.shutdown().await;
        });
    }
}
