//! Per-connection gateway: wire-version latch, watchdog, pause gate and
//! message routing (reactor §2.1, §4.3, §5.2, §5.4; design §5.7).
//!
//! - The first inbound frame on either channel latches v0/v1
//!   ([`crate::wire::sniff`]). Outbound messages wait for the latch (at most
//!   `latch_grace`, then the offer header's seed is used), so a v1 SDK never
//!   sees a v0 greeting.
//! - **Any** inbound message resets the watchdog; a connection silent for
//!   `ping_timeout` (20 s, polled every 2 s) is lost.
//! - Outbound tracks start paused (the host answers with every send m-line
//!   gated); `ResumeTrack{name}` opens that track for this connection and
//!   forces a keyframe of the current picture (black at the start of the
//!   connection), `PauseTrack` closes it again.
//! - `RequestSchema` → `model_schema`; `RequestClip`/`RequestRecording` →
//!   `clip_failed{reason:"recording disabled"}`; `PublishTrack` → error
//!   `publish_refused` (no IN tracks) — in duplex mode it claims the input
//!   track's publisher slot (first come, first served) and this
//!   connection's inbound camera/microphone media then feeds the session's
//!   input rings through an [`Ingest`]; `UnpublishTrack` and closing
//!   release it; commands are validated against the
//!   mode's table (`invalid_command`) and run by the session driver; the
//!   reply is a correlated `ModelMessage` or a bodyless ack (v1 only).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use fastvideo_webrtc::host::{InboundMedia, Peer, PeerEvent, PeerHandle};
use fastvideo_webrtc::ingest::Ingest;
use fastvideo_webrtc::writer::TrackKind;
use serde_json::json;
use tokio::sync::mpsc;

use crate::driver::Outcome;
use crate::session::{Live, Reactor};
use crate::wire::{self, ClientMsg, ServerMsg, WireVersion};

/// RT's clip refusal reason (design §5.7).
pub const RECORDING_DISABLED: &str = "recording disabled";

struct Gw {
    rt: Reactor,
    live: Arc<Live>,
    conn: u32,
    handle: PeerHandle,
    mapping: HashMap<String, String>,
    version: Option<WireVersion>,
    seed: WireVersion,
    pending: Vec<ServerMsg>,
    /// Validated commands, run one at a time in arrival order by this
    /// connection's command worker (RT runs a model's handlers serially).
    commands: mpsc::UnboundedSender<(Option<String>, String, serde_json::Map<String, serde_json::Value>)>,
    /// Duplex: this connection's decode pipeline (from its first publish).
    ingest: Option<Ingest>,
    /// The last refusal reported to the client (`input_rejected`).
    rejected: Option<String>,
}

/// Runs one connection until its peer closes.
pub(crate) fn spawn(
    rt: Reactor,
    live: Arc<Live>,
    conn: u32,
    peer: Peer,
    seed: WireVersion,
    mapping: HashMap<String, String>,
) {
    tokio::spawn(async move {
        let mut peer = peer;
        let inbound = peer.take_inbound();
        let (handle, events) = peer.split();
        let out = live.out.register(conn);
        let (rtx, rrx) = mpsc::unbounded_channel();
        let (ctx, mut crx) = mpsc::unbounded_channel::<(Option<String>, String, serde_json::Map<String, serde_json::Value>)>();
        let driver = live.driver.clone();
        tokio::spawn(async move {
            while let Some((request_id, name, args)) = crx.recv().await {
                let o = driver.command(conn, &name, args).await;
                let m = match (o, request_id) {
                    (Outcome::Reply(kind, data), r) => ServerMsg::Model { request_id: r, kind, data },
                    (Outcome::Ack, Some(r)) => ServerMsg::CommandAck { request_id: r },
                    (Outcome::Error { code, message }, Some(r)) => ServerMsg::CommandError { request_id: r, code, message },
                    (_, None) => continue,
                };
                if rtx.send(m).is_err() {
                    break;
                }
            }
        });
        let mut gw = Gw {
            rt,
            live,
            conn,
            handle,
            mapping,
            version: None,
            seed,
            pending: Vec::new(),
            commands: ctx,
            ingest: None,
            rejected: None,
        };
        gw.run(events, out, rrx, inbound).await;
    });
}

impl Gw {
    async fn run(
        &mut self,
        mut events: mpsc::UnboundedReceiver<PeerEvent>,
        mut out: mpsc::Receiver<ServerMsg>,
        mut replies: mpsc::UnboundedReceiver<ServerMsg>,
        mut inbound: Option<mpsc::Receiver<InboundMedia>>,
    ) {
        let cfg = self.rt.config().clone();
        let mut watchdog = tokio::time::interval(cfg.watchdog_interval);
        watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut last_rx = Instant::now();
        let mut connected_at: Option<Instant> = None;
        let mut opened = false;
        loop {
            tokio::select! {
                ev = events.recv() => {
                    let Some(ev) = ev else { break };
                    match ev {
                        PeerEvent::Connected => {
                            last_rx = Instant::now();
                            connected_at = Some(last_rx);
                            if !opened {
                                opened = true;
                                self.live.media.add_peer(self.handle.clone());
                                self.rt.connection_opened(&self.live, self.conn);
                                self.live.driver.greet(self.conn);
                            }
                        }
                        PeerEvent::ChannelOpen { label } => {
                            tracing::debug!(conn = self.conn, %label, "channel open");
                        }
                        PeerEvent::ChannelClose { .. } => {}
                        PeerEvent::Message(m) => {
                            last_rx = Instant::now();
                            let v = *self.version.get_or_insert_with(|| wire::sniff(&m));
                            self.flush_pending();
                            match wire::decode(v, &m) {
                                Ok(msg) => self.route(msg),
                                Err(e) => tracing::warn!(conn = self.conn, error = %e, "undecodable frame dropped"),
                            }
                        }
                        PeerEvent::KeyframeRequest { .. } => {
                            if let Some(c) = self.handle.video_codec() {
                                self.live.media.request_keyframe(c);
                            }
                        }
                        PeerEvent::Media { .. } => {}
                        PeerEvent::Closed(reason) => {
                            tracing::info!(conn = self.conn, ?reason, "reactor connection closed");
                            break;
                        }
                    }
                }
                m = out.recv() => {
                    // `None`: the outbox dropped us (overflow) or the session ended.
                    let Some(m) = m else { break };
                    self.send(m);
                }
                Some(m) = replies.recv() => self.send(m),
                m = async { match inbound.as_mut() { Some(r) => r.recv().await, None => std::future::pending().await } } => {
                    match m {
                        // Media counts as liveness, like any inbound frame.
                        Some(m) => {
                            last_rx = Instant::now();
                            if let Some(i) = &self.ingest {
                                i.push(m);
                            }
                        }
                        None => inbound = None,
                    }
                }
                _ = watchdog.tick() => {
                    self.report_ingest();
                    let now = Instant::now();
                    if self.version.is_none() && connected_at.is_some_and(|t| now - t >= cfg.latch_grace) {
                        // No inbound frame yet: fall back to the header's seed.
                        self.version = Some(self.seed);
                        self.flush_pending();
                    }
                    if connected_at.is_some() && now - last_rx >= cfg.ping_timeout {
                        tracing::info!(conn = self.conn, "no message for {:?}: connection lost", cfg.ping_timeout);
                        break;
                    }
                }
            }
        }
        self.handle.close();
        if let Some(i) = &self.live.inputs {
            let released = i.release_all(self.conn);
            if !released.is_empty() {
                tracing::info!(conn = self.conn, tracks = ?released, "input tracks released (connection closed)");
            }
        }
        // Stops its decode thread (without waiting for it).
        drop(self.ingest.take());
        self.live.out.unregister(self.conn);
        self.live.media.remove_peer(self.handle.id());
        if opened {
            self.rt.connection_closed(&self.live, self.conn);
        }
        // Forget the peer (a PUT re-offer attaches a new one).
        let mut conns = crate::session::lock(&self.live.conns);
        if let Some(c) = conns.get_mut(&self.conn) {
            if c.peer.as_ref().is_some_and(|p| p.id() == self.handle.id()) {
                c.peer = None;
            }
        }
    }

    fn send(&mut self, m: ServerMsg) {
        match self.version {
            None => self.pending.push(m),
            Some(v) => {
                if let Some(cm) = wire::encode(v, &m) {
                    if let Err(e) = self.handle.post_message(cm) {
                        tracing::debug!(conn = self.conn, error = %e, "send failed");
                    }
                }
            }
        }
    }

    fn flush_pending(&mut self) {
        for m in std::mem::take(&mut self.pending) {
            self.send(m);
        }
    }

    /// The m-line of a track name: the offer's `track_mapping`, else the
    /// first m-line of the track's kind.
    fn set_track_paused(&self, name: &str, paused: bool) {
        let kind = if name == self.live.tracks.video.name {
            Some(TrackKind::Video)
        } else if self.live.tracks.audio.as_ref().is_some_and(|a| a.name == name) {
            Some(TrackKind::Audio)
        } else {
            None
        };
        let Some(kind) = kind else {
            tracing::debug!(conn = self.conn, name, "pause/resume of an unknown track ignored");
            return;
        };
        let r = match self.mapping.get(name) {
            Some(mid) => self.handle.set_paused(mid, paused),
            None => self.handle.set_kind_paused(kind, paused),
        };
        if let Err(e) = r {
            tracing::debug!(conn = self.conn, error = %e, "pause gate");
        }
        if !paused && kind == TrackKind::Video {
            if let Some(c) = self.handle.video_codec() {
                self.live.media.kick(c);
            }
        }
    }

    fn route(&mut self, msg: ClientMsg) {
        match msg {
            ClientMsg::Ping => {}
            ClientMsg::RequestSchema { request_id } => {
                self.send(ServerMsg::ModelSchema { request_id, openapi: self.live.openapi.clone() })
            }
            ClientMsg::FileUploaded { upload } => {
                tracing::debug!(conn = self.conn, upload = %upload.upload_id, "file_uploaded ignored (no upload hook)");
            }
            ClientMsg::RequestClip { request_id, .. } | ClientMsg::RequestRecording { request_id } => {
                let request_id = if request_id.is_empty() { new_request_id() } else { request_id };
                self.send(ServerMsg::ClipFailed { request_id, reason: RECORDING_DISABLED.into() })
            }
            ClientMsg::PublishTrack { request_id, name } => self.publish(request_id, name),
            ClientMsg::PauseTrack { name } => self.set_track_paused(&name, true),
            ClientMsg::ResumeTrack { name } => self.set_track_paused(&name, false),
            ClientMsg::UnpublishTrack { name } => self.unpublish(&name),
            ClientMsg::Error { code, message } => {
                tracing::debug!(conn = self.conn, %code, %message, "client error payload");
            }
            ClientMsg::Command { request_id, name, mut data, uploads } => {
                // `Command.uploads[param]` fills that parameter with the
                // upload reference (reactor §3.5); the driver resolves it.
                for (param, u) in uploads {
                    data.insert(
                        param,
                        serde_json::json!({"upload_id": u.upload_id, "name": u.name, "mime_type": u.mime_type, "size": u.size}),
                    );
                }
                // RT mints an id for an uncorrelated v1 command.
                let v1 = self.version == Some(WireVersion::V1);
                let request_id = match request_id {
                    Some(r) => Some(r),
                    None if v1 => Some(new_request_id()),
                    None => None,
                };
                let args = match self.live.table.validate(&name, &data) {
                    Ok(a) => a,
                    Err(e) => {
                        tracing::info!(conn = self.conn, command = %name, error = %e, "invalid command");
                        if let Some(r) = request_id {
                            self.send(ServerMsg::CommandError { request_id: r, code: "invalid_command".into(), message: e });
                        }
                        return;
                    }
                };
                let _ = self.commands.send((request_id, name, args));
            }
        }
    }
}

impl Gw {
    /// `PublishTrack`: claim the input track's slot and open this
    /// connection's media for it.
    fn publish(&mut self, request_id: String, name: String) {
        let refuse = |message: String| ServerMsg::PublishTrackError {
            request_id: request_id.clone(),
            code: "publish_refused".into(),
            message,
        };
        let Some(inputs) = self.live.inputs.clone() else {
            self.send(refuse(format!("the model declares no input track `{name}`")));
            return;
        };
        let kind = match inputs.claim(&name, self.conn) {
            Ok(k) => k,
            Err(m) => {
                tracing::info!(conn = self.conn, track = %name, reason = %m, "publish refused");
                self.send(refuse(m));
                return;
            }
        };
        if self.ingest.is_none() {
            match Ingest::start(inputs.ingest_config(), inputs.buffers.clone(), self.handle.clone()) {
                Ok(i) => {
                    i.set_enabled(TrackKind::Video, false);
                    i.set_enabled(TrackKind::Audio, false);
                    self.ingest = Some(i);
                }
                Err(e) => {
                    inputs.release(&name, self.conn);
                    self.send(refuse(format!("cannot start the ingest: {e}")));
                    return;
                }
            }
        }
        if let Some(i) = &self.ingest {
            // The offer's `track_mapping` names the m-line; else any of the kind.
            i.set_mid(kind, self.mapping.get(&name).cloned());
            i.set_enabled(kind, true);
        }
        tracing::info!(conn = self.conn, track = %name, "input track published");
        // Ask for a keyframe now: decoding starts at one.
        if kind == TrackKind::Video {
            if let Some(mid) = self.mapping.get(&name) {
                let _ = self.handle.request_keyframe(mid);
            }
        }
        self.send(ServerMsg::PublishTrackOk { request_id });
    }

    /// `UnpublishTrack`: release the slot if this connection holds it.
    fn unpublish(&mut self, name: &str) {
        let Some(inputs) = self.live.inputs.clone() else { return };
        if let Some(kind) = inputs.release(name, self.conn) {
            if let Some(i) = &self.ingest {
                i.set_enabled(kind, false);
            }
            tracing::info!(conn = self.conn, track = %name, "input track unpublished");
        }
    }

    /// Publishes this connection's ingest counters to the session (when it
    /// publishes anything) and tells the client about a new refusal.
    fn report_ingest(&mut self) {
        let (Some(inputs), Some(ingest)) = (self.live.inputs.clone(), &self.ingest) else { return };
        let st = ingest.stats();
        if inputs.publishers().values().any(|c| *c == self.conn) {
            inputs.set_stats(st.clone());
        }
        if st.rejected.is_some() && st.rejected != self.rejected {
            self.rejected = st.rejected.clone();
            let track = inputs.video.clone().unwrap_or_default();
            self.send(ServerMsg::broadcast("input_rejected", json!({"track": track, "reason": st.rejected})));
        }
    }
}

fn new_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}
