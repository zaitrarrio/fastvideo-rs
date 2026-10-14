//! The output publisher (docs/serve/stream-outputs.md "Pipeline"): paced
//! ticks → one encode → the destination.
//!
//! | destination | encode | transport |
//! |---|---|---|
//! | WHIP (Cloudflare default, any WHIP URL) | H.264 Constrained Baseline (`[webrtc] encoder`), or VP8 / VP9 profile 0 (ffmpeg libvpx) by `codec`; Opus 48 kHz stereo | str0m peer, complete offer (no trickle) |
//! | RTMP(S), SRT (Cloudflare or any) | ffmpeg sink: H.264 CBR closed 2 s GOP (NVENC; libx264 on CPU builds) + AAC-LC 48 kHz stereo | FLV / MPEG-TS over ffmpeg's rtmp(s)/libsrt |
//!
//! Both take the native canvas and fps. Native streams feed the pacer's
//! ticks straight in ([`Source::Paced`]); director and Reactor output copies
//! feed a [`TapReceiver`] ([`Source::Tap`]) through [`Opener`]. Credentials
//! never leave [`OutputSpec`] / [`Provisioned`]: status, events and logs
//! carry redacted URLs, and transport errors are scrubbed of the
//! destination URL and its key.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_engine_service::stream::PacedStream;
use fastvideo_media::output::{OutputCopy, OutputMedia, OutputOpener, OutputTap, TapReceiver};
use fastvideo_media::pacer::VideoOut;
use fastvideo_media::sink::{srt_url, FfmpegSink, SinkAudioIn, SinkConfig, SinkTarget};
use fastvideo_media::video::{create_encoder, EncodedFrame, EncoderBackend, FfmpegH264, KeyframePolicy, VideoEncoder};
use fastvideo_media::vp8::{Vp8Config, Vp8Encoder};
use fastvideo_media::RgbFrame;
use fastvideo_protocol::{ApiError, ErrorKind};
use fastvideo_webrtc::host::{AudioLayout, HostConfig, PeerEvent, RtcHost};
use fastvideo_webrtc::ice::IceServer;
use fastvideo_webrtc::whip::{WhipAuth, WhipConfig, WhipPublishOptions, WhipPublisher, WhipTarget};
use fastvideo_webrtc::writer::{video_rtp_time, VideoCodec, VideoFrame};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Notify};
use url::Url;

use super::cloudflare::LiveInput;
use super::spec::{looks_like_key, redact_url, scrub, OutputKind, OutputSpec};
use super::{OutputState, Outputs};

/// Where the encoded media goes, credentials included (never printed).
pub(crate) enum Dest {
    Whip(WhipConfig),
    Sink(SinkTarget),
}

/// A destination ready to publish to (a Cloudflare live input created).
pub struct Provisioned {
    pub(crate) dest: Dest,
    pub(crate) live: Option<LiveInput>,
    pub(crate) kind: OutputKind,
    /// The public playback block (`whep_url`, `hls_url`, …; `{}` when the
    /// destination has no known player).
    pub playback: Value,
    /// Strings scrubbed from every transport error.
    secrets: Vec<String>,
}

impl std::fmt::Debug for Provisioned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Provisioned({}, {})", self.kind.as_str(), self.playback)
    }
}

impl Provisioned {
    fn scrub(&self, text: &str) -> String {
        let s: Vec<&str> = self.secrets.iter().map(String::as_str).collect();
        scrub(text, &s)
    }
}

/// The secret parts of a URL: itself, userinfo, query values, key-like
/// path segments.
fn url_secrets(u: &str) -> Vec<String> {
    let mut out = vec![u.to_owned()];
    if let Ok(p) = Url::parse(u) {
        if !p.username().is_empty() {
            out.push(p.username().to_owned());
        }
        if let Some(pw) = p.password() {
            out.push(pw.to_owned());
        }
        out.extend(p.query_pairs().map(|(_, v)| v.into_owned()));
        if let Some(segs) = p.path_segments() {
            out.extend(segs.filter(|s| looks_like_key(s)).map(str::to_owned));
        }
    }
    out
}

impl Outputs {
    /// Creates the Cloudflare live input (Cloudflare kinds) or checks the
    /// destination's DNS (others), and builds the publish target.
    pub async fn provision(&self, spec: &OutputSpec, label: &str) -> Result<Provisioned, ApiError> {
        let mut secrets: Vec<String> = [&spec.token, &spec.stream_key, &spec.stream_id, &spec.passphrase]
            .into_iter()
            .flatten()
            .cloned()
            .collect();
        if let Some(u) = &spec.url {
            secrets.extend(url_secrets(u.as_str()));
        }
        if spec.kind.is_cloudflare() {
            let cf = self.cloudflare().ok_or_else(|| {
                ApiError::invalid_param("output.kind", "this server has no Cloudflare Stream account ([outputs.cloudflare] account_id and FV_CF_STREAM_API_TOKEN)")
            })?;
            let whip = spec.kind == OutputKind::Cloudflare;
            let prefix = self.cfg.cloudflare.as_ref().map_or("fv-", |a| a.name_prefix.as_str());
            let name = format!("{prefix}{label}");
            let live = cf.create(&name, cf.records(whip)).await?;
            let playback = live.playback(cf.settings().customer_domain.as_deref(), whip);
            tracing::info!(live_input = %live.uid, kind = spec.kind.as_str(), recording = live.recording, "cloudflare live input created");
            let missing = |what: &str| ApiError::internal(format!("the Cloudflare live input has no {what} ingest"));
            let dest = match spec.kind {
                OutputKind::Cloudflare => {
                    let u = live.webrtc_url.clone().ok_or_else(|| missing("WebRTC"))?;
                    let url = Url::parse(&u).map_err(|e| ApiError::internal(format!("Cloudflare WebRTC URL: {e}")))?;
                    Dest::Whip(WhipConfig::new(url, WhipAuth::None).with_target(WhipTarget::Cloudflare).with_codec(spec.codec))
                }
                OutputKind::CloudflareRtmps => {
                    let (u, k) = live.rtmps_url.clone().zip(live.rtmps_key.clone()).ok_or_else(|| missing("RTMPS"))?;
                    Dest::Sink(SinkTarget::Rtmp { url: format!("{}/{}", u.trim_end_matches('/'), k) })
                }
                _ => {
                    let u = live.srt_url.clone().ok_or_else(|| missing("SRT"))?;
                    let url = srt_url(&u, live.srt_stream_id.as_deref(), live.srt_passphrase.as_deref(), self.cfg.srt_latency_ms);
                    Dest::Sink(SinkTarget::Srt { url })
                }
            };
            for s in [&live.webrtc_url, &live.rtmps_key, &live.srt_stream_id, &live.srt_passphrase].into_iter().flatten() {
                secrets.extend(url_secrets(s));
            }
            secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
            return Ok(Provisioned { dest, live: Some(live), kind: spec.kind, playback, secrets });
        }
        // Your own destination: every address its name resolves to must be
        // public unless `[outputs] allow_private`.
        if let Some((host, port)) = spec.host() {
            let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host.as_str(), port))
                .await
                .map_err(|_| ApiError::invalid_param("output.url", "the destination host does not resolve"))?
                .collect();
            for a in &addrs {
                super::spec::check_ip(a.ip(), self.cfg.policy)?;
            }
        }
        let url = spec.url.clone().ok_or_else(|| ApiError::invalid_param("output.url", "this kind needs `url`"))?;
        let dest = match spec.kind {
            OutputKind::Whip => {
                let token = spec.token.clone().or_else(|| self.cfg.whip_token.clone()).unwrap_or_default();
                if let Some(t) = &self.cfg.whip_token {
                    secrets.push(t.clone());
                }
                let auth = match &spec.user {
                    Some(u) => WhipAuth::from_user_token(u, &token),
                    None if token.is_empty() => WhipAuth::None,
                    None => WhipAuth::Bearer(token),
                };
                let target = match spec.whip_target {
                    Some(t) => t,
                    None => match &self.cfg.whip_target {
                        Some(t) => t.parse::<WhipTarget>().map_err(|e| ApiError::invalid_param("output.target", e.to_string()))?,
                        None => WhipTarget::guess(&url),
                    },
                };
                Dest::Whip(WhipConfig::new(url, auth).with_target(target).with_codec(spec.codec))
            }
            OutputKind::Rtmps => Dest::Sink(SinkTarget::Rtmp { url: spec.rtmp_publish_url().unwrap_or_default() }),
            _ => Dest::Sink(SinkTarget::Srt {
                url: srt_url(url.as_str(), spec.stream_id.as_deref(), spec.passphrase.as_deref(), self.cfg.srt_latency_ms),
            }),
        };
        secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
        Ok(Provisioned { dest, live: None, kind: spec.kind, playback: json!({}), secrets })
    }

    /// Deletes / disables the Cloudflare live input per `on_stop` (best
    /// effort, logged).
    pub async fn release(&self, p: &Provisioned) {
        let (Some(live), Some(cf)) = (&p.live, self.cloudflare()) else { return };
        match cf.release(live).await {
            Ok(a) => tracing::info!(live_input = %live.uid, action = ?a, "cloudflare live input released"),
            Err(e) => tracing::warn!(live_input = %live.uid, error = %e.message, "cloudflare live input release failed"),
        }
    }
}

/// One tick for the publisher.
pub(crate) struct PubTick {
    pub frame: Option<RgbFrame>,
    pub audio: Option<Vec<f32>>,
    pub rtp: u64,
    /// Ticks were dropped before this one (ask for a keyframe).
    pub gap: bool,
}

/// Where ticks come from.
pub(crate) enum Source<'a> {
    /// A native stream's pacer.
    Paced(&'a mut PacedStream),
    /// A session's output tap.
    Tap { rx: TapReceiver, fps: u32 },
}

impl Source<'_> {
    async fn next(&mut self) -> Option<PubTick> {
        match self {
            Source::Paced(p) => {
                let t = p.ticks.recv().await?;
                let gap = p.ticks.take_dropped();
                let frame = match t.video {
                    VideoOut::Fresh(f) | VideoOut::Repeat(f) => Some(f),
                    VideoOut::Nothing => None,
                };
                Some(PubTick { frame, audio: t.audio, rtp: u64::from(t.video_rtp), gap })
            }
            Source::Tap { rx, fps } => {
                let t = rx.recv().await?;
                let gap = rx.take_dropped();
                Some(PubTick { frame: t.frame, audio: t.audio, rtp: video_rtp_time(t.index, *fps), gap })
            }
        }
    }
}

/// Phase changes a native stream mirrors into its own state.
pub(crate) type PhaseHook<'a> = Box<dyn Fn(&'static str) + Send + Sync + 'a>;

/// Everything a publish run needs.
pub(crate) struct Ctx<'a> {
    pub outs: &'a Outputs,
    pub state: &'a OutputState,
    pub stop: &'a Notify,
    pub fps: u32,
    /// Interleaved channels of the tick audio (`None`: video only).
    pub audio_channels: Option<u8>,
    pub events: Option<mpsc::UnboundedSender<Value>>,
    /// Called once the first video frame went out.
    pub on_first_sent: Option<Box<dyn FnOnce() + Send + 'a>>,
    pub on_phase: Option<PhaseHook<'a>>,
}

impl Ctx<'_> {
    fn phase(&self, p: &'static str) {
        self.state.set(|s| s.output_state = p.into());
        if let Some(h) = &self.on_phase {
            h(p);
        }
    }

    fn event(&self, v: Value) {
        if let Some(tx) = &self.events {
            let _ = tx.send(v);
        }
    }

    fn ready(&self) {
        self.event(json!({"type": "output_ready", "output": self.state.view, "playback": self.state.playback()}));
    }

    fn first_sent(&mut self) {
        if let Some(f) = self.on_first_sent.take() {
            f();
        }
    }
}

fn err(kind: ErrorKind, p: &Provisioned, msg: impl std::fmt::Display) -> ApiError {
    ApiError::new(kind, p.scrub(&msg.to_string()))
}

/// The encoder backend of `[webrtc] encoder` (`auto` resolved).
fn pick_encoder(choice: &str) -> EncoderBackend {
    match choice {
        "nvenc" => EncoderBackend::Nvenc,
        "openh264" => EncoderBackend::OpenH264,
        "x264-test" => EncoderBackend::CpuTestX264,
        _ => {
            let (_, sel) = crate::encoders::auto_selection();
            if sel.streams == "auto" { EncoderBackend::OpenH264 } else { pick_encoder(&sel.streams) }
        }
    }
}

/// Publishes `src` to `prov` until the source ends or `stop`. Returns the
/// end reason; an `Err` is a destination failure (scrubbed).
pub(crate) async fn publish(ctx: &mut Ctx<'_>, prov: &Provisioned, src: &mut Source<'_>) -> Result<String, ApiError> {
    ctx.phase("waiting");
    // Never offer an empty track (design §5.2): start at the first frame.
    let first = loop {
        tokio::select! {
            _ = ctx.stop.notified() => return Ok("stopped".into()),
            t = src.next() => match t {
                None => return Ok("ended".into()),
                Some(t) if t.frame.is_some() => break t,
                Some(_) => {}
            },
        }
    };
    match &prov.dest {
        Dest::Whip(w) => publish_whip(ctx, prov, w.clone(), src, first).await,
        Dest::Sink(t) => publish_sink(ctx, prov, t.clone(), src, first).await,
    }
}

enum VEnc {
    H264(Box<dyn VideoEncoder>),
    Vpx(Box<Vp8Encoder>),
}

impl VEnc {
    fn new(codec: VideoCodec, target: WhipTarget, backend: EncoderBackend, w: u32, h: u32, fps: u32) -> Result<(Self, u32, String), String> {
        match codec {
            VideoCodec::H264 => {
                let c = crate::whip::whip_h264(target, w, h, fps);
                let gop = c.gop_frames();
                let e = create_encoder(backend, c).map_err(|e| format!("encoder: {e}"))?;
                Ok((VEnc::H264(e), gop, format!("{backend:?}")))
            }
            VideoCodec::Vp8 | VideoCodec::Vp9 => {
                let vp9 = codec == VideoCodec::Vp9;
                let ok = if vp9 { fastvideo_media::vp8::libvpx_vp9_available() } else { fastvideo_media::vp8::libvpx_available() };
                if !ok {
                    return Err(format!("{} needs ffmpeg with {}", codec.sdp_name(), if vp9 { "libvpx-vp9" } else { "libvpx" }));
                }
                let mut c = if vp9 { Vp8Config::vp9(w, h, fps) } else { Vp8Config::new(w, h, fps) };
                if target == WhipTarget::Cloudflare {
                    c.gop_seconds = 1.0;
                }
                let gop = c.gop_frames();
                let name = c.encoder_name().to_owned();
                let e = Vp8Encoder::new(c).map_err(|e| format!("encoder: {e}"))?;
                Ok((VEnc::Vpx(Box::new(e)), gop, name))
            }
        }
    }

    fn dims(&self) -> (u32, u32) {
        match self {
            VEnc::H264(e) => (e.config().input_width, e.config().input_height),
            VEnc::Vpx(e) => (e.config().width, e.config().height),
        }
    }

    fn encode(&mut self, f: &RgbFrame, force: bool) -> Result<Vec<EncodedFrame>, String> {
        match self {
            VEnc::H264(e) => {
                if force {
                    e.force_idr();
                }
                e.encode(f).map_err(|e| format!("encode: {e}"))
            }
            VEnc::Vpx(e) => {
                if force {
                    e.force_keyframe();
                }
                e.encode(f).map_err(|e| format!("encode: {e}"))
            }
        }
    }

    fn finish(&mut self) {
        let _ = match self {
            VEnc::H264(e) => e.finish(),
            VEnc::Vpx(e) => e.finish(),
        };
    }
}

/// Mono tick audio duplicated to stereo for Opus.
fn stereo(a: &[f32], channels: Option<u8>) -> std::borrow::Cow<'_, [f32]> {
    if channels == Some(1) {
        std::borrow::Cow::Owned(a.iter().flat_map(|&s| [s, s]).collect())
    } else {
        std::borrow::Cow::Borrowed(a)
    }
}

async fn publish_whip(ctx: &mut Ctx<'_>, prov: &Provisioned, wcfg: WhipConfig, src: &mut Source<'_>, first: PubTick) -> Result<String, ApiError> {
    let cfg = &ctx.outs.cfg;
    let target = wcfg.target;
    let codec = wcfg.codec;
    let audio = ctx.audio_channels.is_some();
    ctx.phase("connecting");
    let stun: Vec<IceServer> = cfg.stun.iter().map(IceServer::stun).collect();
    let host = RtcHost::bind(HostConfig { udp_bind: Some(cfg.udp_bind), ice_servers: stun.clone(), ..HostConfig::default() })
        .await
        .map_err(|e| err(ErrorKind::Internal, prov, e))?;
    let opts = WhipPublishOptions { audio: audio.then_some(AudioLayout::Stereo), stun, stun_timeout: Duration::from_secs(2) };
    let mut publisher = match WhipPublisher::publish(&host, wcfg, opts).await {
        Ok(p) => p,
        Err(e) => {
            host.shutdown().await;
            return Err(err(ErrorKind::Internal, prov, format!("whip: {e}")));
        }
    };
    let resource = publisher.resource_url().map(|u| redact_url(u, OutputKind::Whip));
    let negotiated = publisher.negotiated_video_codec().map(str::to_owned);
    ctx.state.set(|s| {
        s.whip_resource = resource;
        s.video_codec = negotiated;
    });
    let r = whip_loop(ctx, prov, &mut publisher, src, first, target, codec).await;
    let _ = publisher.teardown().await;
    host.shutdown().await;
    r
}

async fn whip_loop(
    ctx: &mut Ctx<'_>,
    prov: &Provisioned,
    publisher: &mut WhipPublisher,
    src: &mut Source<'_>,
    first: PubTick,
    target: WhipTarget,
    codec: VideoCodec,
) -> Result<String, ApiError> {
    let handle = publisher.peer().handle().clone();
    // ICE + DTLS.
    let deadline = tokio::time::Instant::now() + ctx.outs.cfg.connect_timeout;
    loop {
        tokio::select! {
            ev = publisher.peer().next_event() => match ev {
                Some(PeerEvent::Connected) => break,
                Some(PeerEvent::Closed(r)) => return Err(err(ErrorKind::Internal, prov, format!("whip peer closed: {r:?}"))),
                Some(_) => {}
                None => return Err(err(ErrorKind::Internal, prov, "whip peer ended")),
            },
            _ = tokio::time::sleep_until(deadline) => return Err(err(ErrorKind::Timeout, prov, "whip ICE/DTLS did not connect")),
            _ = ctx.stop.notified() => return Ok("stopped".into()),
        }
    }
    let backend = {
        let c = ctx.outs.cfg.encoder.clone();
        tokio::task::spawn_blocking(move || pick_encoder(&c)).await.map_err(|e| ApiError::internal(e.to_string()))?
    };
    let fps = ctx.fps;
    let mut enc: Option<VEnc> = None;
    let mut idr = KeyframePolicy::new(1.0, 1);
    // Opus comes with `webrtc` (this module's gate), not `encoders`: the
    // serve images build without `encoders`, and a gate on it left the
    // offer's audio m-line without packets, so MediaMTX published the copy
    // video-only (docs/serve/e2e/gpu-check-2026-10-13.md, bug 3).
    let mut opus = if ctx.audio_channels.is_some() {
        Some(
            fastvideo_media::opus::OpusEncoder::new(fastvideo_media::opus::OpusConfig::whip(), 0)
                .map_err(|e| ApiError::internal(format!("opus: {e}")))?,
        )
    } else {
        None
    };
    ctx.phase("publishing");
    ctx.ready();
    let t0 = Instant::now();
    let mut rtps: VecDeque<u64> = VecDeque::new();
    let mut pending = Some(first);
    let reason = loop {
        let t = match pending.take() {
            Some(t) => t,
            None => tokio::select! {
                _ = ctx.stop.notified() => break "stopped".to_owned(),
                ev = publisher.peer().next_event() => {
                    match ev {
                        Some(PeerEvent::KeyframeRequest { .. }) => {
                            idr.request(t0.elapsed().as_secs_f64());
                            ctx.state.set(|s| s.keyframe_requests += 1);
                        }
                        Some(PeerEvent::Closed(r)) => return Err(err(ErrorKind::Internal, prov, format!("whip peer closed: {r:?}"))),
                        Some(_) => {}
                        None => return Err(err(ErrorKind::Internal, prov, "whip peer ended")),
                    }
                    continue;
                }
                t = src.next() => match t {
                    Some(t) => t,
                    None => break "ended".to_owned(),
                },
            },
        };
        if t.gap {
            idr.request(t0.elapsed().as_secs_f64());
            ctx.state.set(|s| s.ticks_dropped += 1);
        }
        if let (Some(op), Some(a)) = (opus.as_mut(), t.audio.as_ref()) {
            match op.push(&stereo(a, ctx.audio_channels)) {
                Ok(pkts) => {
                    let mut sent = 0;
                    for p in pkts {
                        if handle.send_audio(fastvideo_webrtc::writer::AudioPacket::new(p.data, u64::from(p.rtp_ts))).is_ok() {
                            sent += 1;
                        }
                    }
                    ctx.state.set(|s| s.audio_packets_sent += sent);
                }
                Err(e) => tracing::warn!(error = %e, "opus encode failed"),
            }
        }
        let Some(frame) = t.frame else { continue };
        // One encoder per canvas (a size change restarts it on a keyframe).
        if enc.as_ref().is_none_or(|e| e.dims() != (frame.width, frame.height)) {
            let (w, h) = (frame.width, frame.height);
            if let Some(mut old) = enc.take() {
                let _ = tokio::task::spawn_blocking(move || old.finish()).await;
            }
            let (e, gop, name) = tokio::task::spawn_blocking(move || VEnc::new(codec, target, backend, w, h, fps))
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?
                .map_err(|e| err(ErrorKind::Internal, prov, e))?;
            idr = KeyframePolicy::new(1.0, gop);
            rtps.clear();
            enc = Some(e);
            ctx.state.set(|s| {
                s.encoder = Some(name);
                s.width = Some(w);
                s.height = Some(h);
            });
        }
        let force = idr.next_frame(t0.elapsed().as_secs_f64());
        rtps.push_back(t.rtp);
        let mut e = enc.take().expect("encoder present");
        let (e, out) = tokio::task::spawn_blocking(move || {
            let r = e.encode(&frame, force);
            (e, r)
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
        enc = Some(e);
        if force {
            ctx.state.set(|s| s.forced_idrs += 1);
        }
        let covered = idr.covered();
        ctx.state.set(|s| s.keyframe_requests_covered = covered);
        let aus = out.map_err(|e| err(ErrorKind::Internal, prov, e))?;
        for au in aus {
            let rtp = rtps.pop_front().unwrap_or(t.rtp);
            let key = au.keyframe;
            if key {
                idr.keyframe_out(t0.elapsed().as_secs_f64());
            }
            match handle.send_video(VideoFrame::new(au.data, rtp)) {
                Ok(()) => {
                    ctx.state.set(|s| {
                        s.frames_sent += 1;
                        if key {
                            s.keyframes_sent += 1;
                        }
                    });
                    ctx.first_sent();
                }
                Err(_) => {
                    ctx.state.set(|s| s.video_send_errors += 1);
                    idr.request(t0.elapsed().as_secs_f64());
                }
            }
        }
    };
    if let Some(mut e) = enc.take() {
        let _ = tokio::task::spawn_blocking(move || e.finish()).await;
    }
    Ok(reason)
}

async fn publish_sink(ctx: &mut Ctx<'_>, prov: &Provisioned, target: SinkTarget, src: &mut Source<'_>, first: PubTick) -> Result<String, ApiError> {
    ctx.phase("connecting");
    let backend = {
        let c = ctx.outs.cfg.encoder.clone();
        tokio::task::spawn_blocking(move || pick_encoder(&c)).await.map_err(|e| ApiError::internal(e.to_string()))?
    };
    // NVENC in production; the CPU builds (fake engine, tests) run the
    // libx264 test path, the only ffmpeg H.264 encoder they have.
    let encoder = if backend == EncoderBackend::Nvenc { FfmpegH264::Nvenc } else { FfmpegH264::Libx264CpuTest };
    let f0 = first.frame.as_ref().expect("first tick has a frame");
    let (w, h) = (f0.width, f0.height);
    let mut scfg = SinkConfig::new(target, w, h, ctx.fps, ctx.audio_channels.map(|c| SinkAudioIn { rate: 48_000, channels: c }));
    scfg.encoder = encoder;
    let mut sink = tokio::task::spawn_blocking(move || FfmpegSink::start(scfg))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(|e| err(ErrorKind::Internal, prov, format!("sink: {e}")))?;
    ctx.state.set(|s| {
        s.video_codec = Some("H264".into());
        s.encoder = Some(format!("ffmpeg {}", if encoder == FfmpegH264::Nvenc { "h264_nvenc" } else { "libx264" }));
        s.width = Some(w);
        s.height = Some(h);
    });
    ctx.phase("publishing");
    ctx.ready();
    let mut pending = Some(first);
    let mut last_check = Instant::now();
    let mut reported: Option<String> = None;
    let reason = loop {
        let t = match pending.take() {
            Some(t) => t,
            None => tokio::select! {
                _ = ctx.stop.notified() => break "stopped".to_owned(),
                t = src.next() => match t {
                    Some(t) => t,
                    None => break "ended".to_owned(),
                },
            },
        };
        if t.gap {
            ctx.state.set(|s| s.ticks_dropped += 1);
        }
        let Some(frame) = t.frame else { continue };
        if (frame.width, frame.height) != (w, h) {
            continue;
        }
        let audio = match (ctx.audio_channels, t.audio.as_deref()) {
            (Some(_), Some(a)) => Some(a),
            _ => None,
        };
        if let Err(e) = sink.send(&frame, audio) {
            tracing::warn!(error = %prov.scrub(&e.to_string()), "sink send failed");
        }
        let st = sink.stats();
        ctx.state.set(|s| {
            s.frames_sent = st.frames_sent;
            s.sink_restarts = st.restarts;
        });
        ctx.first_sent();
        // About once a second: a dead ffmpeg (refused connection, bad key)
        // is reported, not fatal; the sink restarts it every 2 s.
        if last_check.elapsed() >= Duration::from_secs(1) {
            last_check = Instant::now();
            if !sink.is_running() {
                let msg = prov.scrub(&sink.last_error().unwrap_or_else(|| "ffmpeg exited".into()));
                if reported.as_deref() != Some(msg.as_str()) {
                    tracing::warn!(output = %ctx.state.view, error = %msg, "output sink failed; restarting");
                    ctx.state.set(|s| s.output_error = Some(msg.clone()));
                    ctx.event(json!({"type": "output_error", "output": ctx.state.view, "message": msg, "fatal": false}));
                    reported = Some(msg);
                }
            }
        }
    };
    let r = tokio::task::spawn_blocking(move || sink.finish(Duration::from_secs(5))).await.map_err(|e| ApiError::internal(e.to_string()))?;
    match r {
        Ok(st) => ctx.state.set(|s| s.frames_sent = st.frames_sent),
        Err(e) => tracing::info!(error = %prov.scrub(&e.to_string()), "sink finished with an error"),
    }
    Ok(reason)
}

/// The [`OutputOpener`] of director and Reactor sessions.
pub struct Opener(pub Arc<Outputs>);

impl OutputOpener for Opener {
    fn open(&self, request: &Value, media: OutputMedia) -> Result<OutputCopy, ApiError> {
        let spec = self.0.resolve(Some(request), None)?;
        let ticks = (media.fps.max(1) as usize) * 2;
        let (tap, rx) = OutputTap::channel(ticks);
        let (tx, events) = mpsc::unbounded_channel();
        let view = spec.view();
        let outs = self.0.clone();
        tokio::spawn(run_copy(outs, spec, media, rx, tx));
        Ok(OutputCopy { tap, events, view })
    }

    fn info(&self) -> Value {
        self.0.info()
    }
}

/// A session's output copy, to completion. Failures become events.
async fn run_copy(outs: Arc<Outputs>, spec: OutputSpec, media: OutputMedia, rx: TapReceiver, tx: mpsc::UnboundedSender<Value>) {
    let state = OutputState::new(spec.view());
    let view = state.view.clone();
    let prov = match outs.provision(&spec, &media.label).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(output = %view, error = %e.message, "output copy could not start");
            let _ = tx.send(json!({"type": "output_error", "output": view, "message": e.message, "fatal": true}));
            let _ = tx.send(json!({"type": "output_ended", "output": view, "reason": "error"}));
            return;
        }
    };
    if let Ok(mut p) = state.playback.lock() {
        *p = prov.playback.clone();
    }
    let stop = Notify::new();
    let mut src = Source::Tap { rx, fps: media.fps.max(1) };
    let mut ctx = Ctx {
        outs: &outs,
        state: &state,
        stop: &stop,
        fps: media.fps.max(1),
        audio_channels: media.audio_channels,
        events: Some(tx.clone()),
        on_first_sent: None,
        on_phase: None,
    };
    let r = publish(&mut ctx, &prov, &mut src).await;
    drop(src);
    outs.release(&prov).await;
    let reason = match r {
        Ok(reason) => reason,
        Err(e) => {
            tracing::warn!(output = %view, error = %e.message, "output copy failed");
            let _ = tx.send(json!({"type": "output_error", "output": view, "message": e.message, "fatal": true}));
            "error".into()
        }
    };
    let st = state.status();
    tracing::info!(output = %view, reason, frames = st.frames_sent, "output copy ended");
    let _ = tx.send(json!({"type": "output_ended", "output": view, "reason": reason, "frames_sent": st.frames_sent}));
}

