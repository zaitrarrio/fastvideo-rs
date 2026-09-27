//! str0m peer host: one UDP socket and one ICE-TCP listener shared by every
//! peer (design §5.8; WP-04).
//!
//! str0m is sans-IO. [`RtcHost`] owns the sockets and runs one tokio task
//! that demultiplexes inbound packets to peers with `Rtc::accepts`, feeds
//! timeouts, and drains `poll_output` after **every** mutation (the str0m
//! single-mutation invariant). Everything else talks to that task through
//! channels:
//!
//! - [`RtcHost::answer`]: accept a full offer and return a **non-trickle**
//!   answer with every host candidate and `a=end-of-candidates` (Reactor
//!   `sdp_params`, fal WMA `/session`). Client trickle is still accepted
//!   through [`PeerHandle::add_remote_candidate`] (Reactor
//!   `ice_candidates`).
//! - [`RtcHost::offer`] + [`PendingOffer::accept_answer`]: the offering side
//!   (the WHIP publisher, and loopback tests).
//! - [`PeerHandle`]: pre-encoded H.264/Opus writes, data-channel sends, the
//!   per-mid pause gate, close. [`Peer`] also yields [`PeerEvent`]s.
//!
//! ICE-TCP (RFC 6544) is passive only: browsers connect to our listener and
//! every packet is RFC 4571 framed. That is the only inbound path on a
//! Runpod pod (no UDP, deploy §2). Candidates are advertised at the public
//! addresses of the [`CandidatePlan`]; inbound packets are attributed to the
//! advertised address (1:1 NAT) so str0m's ICE agent recognises them.
//!
//! The server never runs TURN (str0m has none; risk R2) and never trickles
//! its own candidates.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use str0m::change::{SdpAnswer, SdpOffer, SdpPendingOffer};
use str0m::channel::ChannelId;
use str0m::format::Codec;
use str0m::media::{Direction as RtcDirection, Frequency, MediaKind as RtcMediaKind, MediaTime, Mid, Pt};
use str0m::net::{Protocol, Receive, TcpType};
use str0m::{Candidate, Event, IceConnectionState, IceCreds, Input, Output, Rtc};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::{mpsc, oneshot, watch};

use crate::channel::{ChannelMessage, ChannelPolicy, PendingMessages};
use crate::framing;
use crate::ice::{default_interface_ip, CandidatePlan, IceServer, PublicAddrs, Transport};
use crate::sdp::{self, Direction, MediaKind, OfferRequirements, Sdp};
use crate::stun;
use crate::writer::{AudioPacket, TrackKind, VideoFrame, WallclockMap, AUDIO_CLOCK_HZ, VIDEO_CLOCK_HZ};
use crate::WebrtcError;

/// Host configuration (design §6.1 `[webrtc]`).
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// UDP mux socket. `None` disables UDP (Runpod pods have none).
    pub udp_bind: Option<SocketAddr>,
    /// ICE-TCP passive listener. `None` disables ICE-TCP.
    pub tcp_bind: Option<SocketAddr>,
    /// Public addresses to advertise (from [`crate::ice::resolve_ports`]).
    pub public: PublicAddrs,
    /// Extra host-candidate addresses (e.g. a LAN IP next to a public one).
    pub extra_udp: Vec<SocketAddr>,
    pub extra_tcp: Vec<SocketAddr>,
    /// ICE servers **for clients** (fal `/ice`, Reactor `ice_servers`), and
    /// the STUN servers the WHIP publisher probes for its srflx candidate.
    pub ice_servers: Vec<IceServer>,
    /// A peer must reach ICE+DTLS connected within this (Reactor: 30 s).
    pub negotiation_timeout: Duration,
    /// A connected peer that sends nothing for this long is dropped. str0m
    /// sends consent checks every few seconds, which live peers answer
    /// (design §5.10: 20 s).
    pub idle_timeout: Duration,
    /// How long ICE may stay `Disconnected` before the peer is closed.
    pub disconnect_grace: Duration,
    /// Concurrent peers (Reactor: 64 connections).
    pub max_peers: usize,
    /// Concurrent inbound ICE-TCP connections.
    pub max_tcp_connections: usize,
    /// str0m stats cadence; also the [`PeerStats`] refresh rate.
    pub stats_interval: Duration,
}

impl Default for HostConfig {
    fn default() -> Self {
        HostConfig {
            udp_bind: Some("0.0.0.0:0".parse().expect("static addr")),
            tcp_bind: None,
            public: PublicAddrs::default(),
            extra_udp: Vec::new(),
            extra_tcp: Vec::new(),
            ice_servers: vec![IceServer::default_stun()],
            negotiation_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(20),
            disconnect_grace: Duration::from_secs(5),
            max_peers: 64,
            max_tcp_connections: 256,
            stats_interval: Duration::from_secs(1),
        }
    }
}

impl HostConfig {
    /// Loopback-only config for tests and same-host clients.
    pub fn loopback(udp: bool, tcp: bool) -> Self {
        let lo: SocketAddr = "127.0.0.1:0".parse().expect("static addr");
        HostConfig {
            udp_bind: udp.then_some(lo),
            tcp_bind: tcp.then_some(lo),
            ice_servers: Vec::new(),
            ..HostConfig::default()
        }
    }
}

/// Audio channel layout on the wire (always 48 kHz Opus). Reactor is mono,
/// fal WMA and WHIP stereo (design §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioLayout {
    Mono,
    Stereo,
}

/// How to answer an offer.
#[derive(Debug, Clone)]
pub struct AnswerOptions {
    /// Send video on the offered video m-line(s). Requires H.264 CB pm=1.
    pub video: bool,
    /// Send audio. `None`: the session is video-only and every audio m-line
    /// is answered `a=inactive` (design §5.3, fal WMA row).
    pub audio: Option<AudioLayout>,
    /// Accepted client-created data-channel labels.
    pub channels: ChannelPolicy,
    /// Start every send m-line paused (Reactor pause gate: nothing is sent
    /// until `ResumeTrack`, reactor §4.3).
    pub start_paused: bool,
    /// Local ICE credentials (Reactor `ice_credentials` for relaying
    /// front-ends). `None`: random.
    pub ice_credentials: Option<(String, String)>,
}

impl Default for AnswerOptions {
    fn default() -> Self {
        AnswerOptions {
            video: true,
            audio: Some(AudioLayout::Stereo),
            channels: ChannelPolicy::Any,
            start_paused: false,
            ice_credentials: None,
        }
    }
}

/// How to build an offer (we are the offerer, e.g. WHIP).
#[derive(Debug, Clone)]
pub struct OfferOptions {
    /// Video m-line direction (WHIP: `SendOnly`). `None`: no video m-line.
    pub video: Option<Direction>,
    /// Audio m-line direction and layout. `None`: no audio m-line — the
    /// video-only WHIP case (design §5.3).
    pub audio: Option<(Direction, AudioLayout)>,
    /// Data channels we create (in-band negotiated, reliable, ordered).
    pub channels: Vec<String>,
    /// Server-reflexive addresses to advertise for the UDP socket (from a
    /// STUN probe, see [`RtcHost::stun_probe`]).
    pub srflx: Vec<SocketAddr>,
}

impl Default for OfferOptions {
    fn default() -> Self {
        OfferOptions {
            video: Some(Direction::SendOnly),
            audio: Some((Direction::SendOnly, AudioLayout::Stereo)),
            channels: Vec::new(),
            srflx: Vec::new(),
        }
    }
}

/// One negotiated audio/video m-line, from our point of view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaInfo {
    pub mid: String,
    pub kind: TrackKind,
    /// Our direction for this m-line after negotiation.
    pub direction: Direction,
}

impl MediaInfo {
    pub fn sends(&self) -> bool {
        self.direction.is_sending()
    }
}

/// Why a peer closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloseReason {
    /// [`PeerHandle::close`], or the [`Peer`] was dropped.
    Local,
    /// Not connected within `negotiation_timeout`.
    NegotiationTimeout,
    /// Connected, but nothing received for `idle_timeout`.
    IdleTimeout,
    /// ICE stayed disconnected for `disconnect_grace`.
    IceDisconnected,
    /// The remote closed DTLS/SCTP.
    RemoteClosed,
    HostShutdown,
    Error(String),
}

/// Events from one peer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerEvent {
    /// ICE and DTLS are up; media can flow. Also a good moment to force an
    /// IDR (a [`PeerEvent::KeyframeRequest`] follows for each video mid).
    Connected,
    ChannelOpen { label: String },
    ChannelClose { label: String },
    Message(ChannelMessage),
    /// PLI/FIR from the remote (or a new viewer): force an IDR.
    KeyframeRequest { mid: String },
    /// Inbound media (loopback/tests; our server peers only send).
    Media { mid: String, kind: TrackKind, rtp_time: u64, keyframe: bool, data: Bytes },
    Closed(CloseReason),
}

/// Per-lane counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct LaneStats {
    /// Frames/packets handed to str0m.
    pub written: u64,
    /// Dropped because the m-line was paused or inactive.
    pub dropped_paused: u64,
    /// Dropped because the peer was not connected yet.
    pub dropped_not_connected: u64,
    pub write_errors: u64,
    /// From str0m egress stats.
    pub bytes: u64,
    pub packets: u64,
    pub nacks: u64,
    pub plis: u64,
    pub firs: u64,
}

/// Snapshot of one peer, refreshed every `stats_interval`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct PeerStats {
    pub connected: bool,
    pub bytes_rx: u64,
    pub bytes_tx: u64,
    pub rtt_ms: Option<f64>,
    /// `"udp"` or `"tcp"` for the selected candidate pair.
    pub transport: Option<String>,
    pub video: LaneStats,
    pub audio: LaneStats,
    pub messages_in: u64,
    pub messages_out: u64,
    pub keyframe_requests: u64,
}

/// Host-wide counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct HostStats {
    pub peers: usize,
    pub tcp_connections: usize,
    pub udp_rx_packets: u64,
    pub udp_tx_packets: u64,
    pub tcp_rx_packets: u64,
    pub tcp_tx_packets: u64,
    /// Inbound packets no peer accepted.
    pub unmatched_packets: u64,
    /// Outbound packets dropped (socket busy, TCP stream gone).
    pub dropped_tx_packets: u64,
}

type PeerId = u64;

enum Cmd {
    Answer { offer: String, opts: AnswerOptions, reply: oneshot::Sender<Result<Peer, WebrtcError>>, answer_tx: oneshot::Sender<String> },
    Offer { opts: OfferOptions, reply: oneshot::Sender<Result<(PendingOffer, String), WebrtcError>> },
    AcceptAnswer { peer: PeerId, answer: String, reply: oneshot::Sender<Result<Vec<MediaInfo>, WebrtcError>> },
    Video { peer: PeerId, frame: VideoFrame },
    Audio { peer: PeerId, packet: AudioPacket },
    Data { peer: PeerId, msg: ChannelMessage, reply: Option<oneshot::Sender<Result<(), WebrtcError>>> },
    SetPaused { peer: PeerId, mid: Option<String>, kind: Option<TrackKind>, paused: bool },
    AddCandidate { peer: PeerId, candidate: String, reply: oneshot::Sender<Result<bool, WebrtcError>> },
    Close { peer: PeerId },
    StunProbe { server: SocketAddr, reply: oneshot::Sender<SocketAddr> },
    Shutdown { reply: oneshot::Sender<()> },
}

struct Shared {
    plan: CandidatePlan,
    udp_local: Option<SocketAddr>,
    tcp_local: Option<SocketAddr>,
    ice_servers: Vec<IceServer>,
    stats: watch::Receiver<HostStats>,
}

/// The shared WebRTC host. Cheap to clone.
#[derive(Clone)]
pub struct RtcHost {
    tx: mpsc::Sender<Cmd>,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for RtcHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtcHost")
            .field("udp", &self.shared.udp_local)
            .field("tcp", &self.shared.tcp_local)
            .field("plan", &self.shared.plan)
            .finish()
    }
}

impl RtcHost {
    /// Bind the sockets and start the host task on the current tokio runtime.
    pub async fn bind(cfg: HostConfig) -> Result<Self, WebrtcError> {
        if cfg.udp_bind.is_none() && cfg.tcp_bind.is_none() {
            return Err(WebrtcError::Config("neither udp_bind nor tcp_bind is set".into()));
        }
        let udp = match cfg.udp_bind {
            Some(a) => Some(Arc::new(UdpSocket::bind(a).await?)),
            None => None,
        };
        let listener = match cfg.tcp_bind {
            Some(a) => Some(TcpListener::bind(a).await?),
            None => None,
        };
        let udp_local = udp.as_ref().map(|s| s.local_addr()).transpose()?;
        let tcp_local = listener.as_ref().map(|s| s.local_addr()).transpose()?;
        let needs_default = udp_local.is_some_and(|a| a.ip().is_unspecified())
            || tcp_local.is_some_and(|a| a.ip().is_unspecified());
        let default_ip = if needs_default {
            let v6 = udp_local.or(tcp_local).is_some_and(|a| a.is_ipv6());
            default_interface_ip(v6)
        } else {
            None
        };
        let plan = CandidatePlan::build(udp_local, tcp_local, &cfg.public, &cfg.extra_udp, &cfg.extra_tcp, default_ip);
        if plan.udp.is_empty() && plan.tcp.is_empty() {
            return Err(WebrtcError::Config("no usable candidate address (bound to 0.0.0.0 without a route; set public/extra addresses)".into()));
        }
        tracing::info!(?udp_local, ?tcp_local, candidates = ?plan.candidate_lines(), "webrtc host bound");
        let (tx, rx) = mpsc::channel(4096);
        let (stats_tx, stats_rx) = watch::channel(HostStats::default());
        let shared = Arc::new(Shared { plan: plan.clone(), udp_local, tcp_local, ice_servers: cfg.ice_servers.clone(), stats: stats_rx });
        let (tcp_in_tx, tcp_in_rx) = mpsc::channel(4096);
        let host = HostLoop {
            cfg,
            plan,
            udp,
            listener,
            tcp_out: HashMap::new(),
            tcp_in_tx,
            tcp_in_rx,
            cmd_rx: rx,
            cmd_tx: tx.downgrade(),
            peers: HashMap::new(),
            next_id: 1,
            stun_waiters: HashMap::new(),
            stats: HostStats::default(),
            stats_tx,
        };
        tokio::spawn(host.run());
        Ok(RtcHost { tx, shared })
    }

    pub fn udp_addr(&self) -> Option<SocketAddr> {
        self.shared.udp_local
    }

    pub fn tcp_addr(&self) -> Option<SocketAddr> {
        self.shared.tcp_local
    }

    /// The addresses advertised as host candidates.
    pub fn candidate_plan(&self) -> &CandidatePlan {
        &self.shared.plan
    }

    /// ICE servers for clients (fal `/wma/ice`, Reactor `ice_servers`).
    pub fn ice_servers(&self) -> &[IceServer] {
        &self.shared.ice_servers
    }

    pub fn stats(&self) -> HostStats {
        self.shared.stats.borrow().clone()
    }

    /// Answer a complete offer. Returns the peer and the non-trickle answer.
    pub async fn answer(&self, offer: &str, opts: AnswerOptions) -> Result<(Peer, String), WebrtcError> {
        let (reply, rx) = oneshot::channel();
        let (answer_tx, answer_rx) = oneshot::channel();
        self.send(Cmd::Answer { offer: offer.to_string(), opts, reply, answer_tx }).await?;
        let peer = rx.await.map_err(|_| WebrtcError::HostGone)??;
        let answer = answer_rx.await.map_err(|_| WebrtcError::HostGone)?;
        Ok((peer, answer))
    }

    /// Create a complete (non-trickle) offer. Apply the remote answer with
    /// [`PendingOffer::accept_answer`].
    pub async fn offer(&self, opts: OfferOptions) -> Result<(PendingOffer, String), WebrtcError> {
        let (reply, rx) = oneshot::channel();
        self.send(Cmd::Offer { opts, reply }).await?;
        rx.await.map_err(|_| WebrtcError::HostGone)?
    }

    /// Ask a STUN server for this host's public UDP address (the mapped
    /// address of the shared socket).
    pub async fn stun_probe(&self, server: SocketAddr, timeout: Duration) -> Result<SocketAddr, WebrtcError> {
        let (reply, rx) = oneshot::channel();
        self.send(Cmd::StunProbe { server, reply }).await?;
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(a)) => Ok(a),
            Ok(Err(_)) => Err(WebrtcError::Stun(format!("no UDP socket to probe {server} from"))),
            Err(_) => Err(WebrtcError::Stun(format!("no STUN response from {server}"))),
        }
    }

    /// Resolve and probe the configured `stun:` servers; returns every
    /// distinct mapped address (usually one). Failures are logged.
    pub async fn gather_srflx(&self, servers: &[IceServer], timeout: Duration) -> Vec<SocketAddr> {
        let mut out = Vec::new();
        let Some(local) = self.udp_addr() else { return out };
        for (host, port) in servers.iter().flat_map(IceServer::stun_targets) {
            let addrs = match tokio::net::lookup_host((host.as_str(), port)).await {
                Ok(a) => a.filter(|a| a.is_ipv4() == local.is_ipv4()).collect::<Vec<_>>(),
                Err(e) => {
                    tracing::warn!(%host, error = %e, "stun host lookup failed");
                    continue;
                }
            };
            let Some(server) = addrs.first().copied() else { continue };
            match self.stun_probe(server, timeout).await {
                Ok(mapped) if !out.contains(&mapped) => out.push(mapped),
                Ok(_) => {}
                Err(e) => tracing::warn!(%server, error = %e, "stun probe failed"),
            }
        }
        out
    }

    /// Close every peer (they get [`CloseReason::HostShutdown`]) and stop.
    pub async fn shutdown(&self) {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Cmd::Shutdown { reply }).await.is_ok() {
            let _ = rx.await;
        }
    }

    async fn send(&self, cmd: Cmd) -> Result<(), WebrtcError> {
        self.tx.send(cmd).await.map_err(|_| WebrtcError::HostGone)
    }
}

/// Handle to one peer: writes, sends, pause gate, close. Cheap to clone;
/// all clones address the same peer.
#[derive(Clone)]
pub struct PeerHandle {
    id: PeerId,
    tx: mpsc::Sender<Cmd>,
    media: Arc<Vec<MediaInfo>>,
    stats: watch::Receiver<PeerStats>,
}

impl std::fmt::Debug for PeerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerHandle").field("id", &self.id).field("media", &self.media).finish()
    }
}

impl PeerHandle {
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Negotiated audio/video m-lines (mid, kind, our direction).
    pub fn media(&self) -> &[MediaInfo] {
        &self.media
    }

    /// True when some m-line of `kind` sends.
    pub fn sends(&self, kind: TrackKind) -> bool {
        self.media.iter().any(|m| m.kind == kind && m.sends())
    }

    pub fn stats(&self) -> PeerStats {
        self.stats.borrow().clone()
    }

    /// Queue one H.264 access unit to every unpaused video m-line. Never
    /// blocks: when the host is saturated the frame is dropped and
    /// [`WebrtcError::Backpressure`] is returned (force an IDR later).
    pub fn send_video(&self, frame: VideoFrame) -> Result<(), WebrtcError> {
        self.try_send(Cmd::Video { peer: self.id, frame })
    }

    /// Queue one Opus packet to every unpaused audio m-line (never blocks).
    pub fn send_audio(&self, packet: AudioPacket) -> Result<(), WebrtcError> {
        self.try_send(Cmd::Audio { peer: self.id, packet })
    }

    /// Send a data-channel message. Messages to a channel that has not
    /// opened yet are queued (at most 64 per peer) and flushed on open.
    pub async fn send_message(&self, msg: ChannelMessage) -> Result<(), WebrtcError> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Cmd::Data { peer: self.id, msg, reply: Some(reply) }).await.map_err(|_| WebrtcError::HostGone)?;
        rx.await.map_err(|_| WebrtcError::PeerGone)?
    }

    /// Fire-and-forget variant of [`send_message`](Self::send_message).
    pub fn post_message(&self, msg: ChannelMessage) -> Result<(), WebrtcError> {
        self.try_send(Cmd::Data { peer: self.id, msg, reply: None })
    }

    /// Pause or resume one m-line (the Reactor pause gate). A resumed video
    /// m-line emits a [`PeerEvent::KeyframeRequest`].
    pub fn set_paused(&self, mid: &str, paused: bool) -> Result<(), WebrtcError> {
        self.try_send(Cmd::SetPaused { peer: self.id, mid: Some(mid.to_string()), kind: None, paused })
    }

    /// Pause or resume every m-line of one kind.
    pub fn set_kind_paused(&self, kind: TrackKind, paused: bool) -> Result<(), WebrtcError> {
        self.try_send(Cmd::SetPaused { peer: self.id, mid: None, kind: Some(kind), paused })
    }

    /// Add a trickled remote candidate (`candidate:…`, with or without
    /// `a=`). Returns `false` when it was ignored: end-of-candidates (empty),
    /// mDNS hostnames, or unparsable input.
    pub async fn add_remote_candidate(&self, candidate: &str) -> Result<bool, WebrtcError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Cmd::AddCandidate { peer: self.id, candidate: candidate.to_string(), reply })
            .await
            .map_err(|_| WebrtcError::HostGone)?;
        rx.await.map_err(|_| WebrtcError::PeerGone)?
    }

    /// Close the peer. Idempotent.
    pub fn close(&self) {
        let _ = self.tx.try_send(Cmd::Close { peer: self.id });
    }

    fn try_send(&self, cmd: Cmd) -> Result<(), WebrtcError> {
        self.tx.try_send(cmd).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => WebrtcError::Backpressure,
            mpsc::error::TrySendError::Closed(_) => WebrtcError::HostGone,
        })
    }
}

/// A peer plus its event stream. Dropping the event stream closes the peer.
pub struct Peer {
    handle: PeerHandle,
    events: mpsc::UnboundedReceiver<PeerEvent>,
}

impl std::fmt::Debug for Peer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.handle.fmt(f)
    }
}

impl Peer {
    pub fn handle(&self) -> &PeerHandle {
        &self.handle
    }

    /// The next event; `None` after [`PeerEvent::Closed`] was delivered.
    pub async fn next_event(&mut self) -> Option<PeerEvent> {
        self.events.recv().await
    }

    pub fn try_next_event(&mut self) -> Option<PeerEvent> {
        self.events.try_recv().ok()
    }

    pub fn split(self) -> (PeerHandle, mpsc::UnboundedReceiver<PeerEvent>) {
        (self.handle, self.events)
    }
}

impl std::ops::Deref for Peer {
    type Target = PeerHandle;
    fn deref(&self) -> &PeerHandle {
        &self.handle
    }
}

/// An offer waiting for its answer. Dropping it closes the peer.
pub struct PendingOffer {
    handle: Option<PeerHandle>,
    events: Option<mpsc::UnboundedReceiver<PeerEvent>>,
}

impl std::fmt::Debug for PendingOffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingOffer").field("peer", &self.handle.as_ref().map(|h| h.id)).finish()
    }
}

impl PendingOffer {
    /// Apply the remote answer and return the live peer.
    pub async fn accept_answer(mut self, answer: &str) -> Result<Peer, WebrtcError> {
        let handle = self.handle.take().ok_or(WebrtcError::PeerGone)?;
        let events = self.events.take().ok_or(WebrtcError::PeerGone)?;
        let (reply, rx) = oneshot::channel();
        handle
            .tx
            .send(Cmd::AcceptAnswer { peer: handle.id, answer: answer.to_string(), reply })
            .await
            .map_err(|_| WebrtcError::HostGone)?;
        match rx.await.map_err(|_| WebrtcError::PeerGone)? {
            Ok(media) => Ok(Peer { handle: PeerHandle { media: Arc::new(media), ..handle }, events }),
            Err(e) => {
                handle.close();
                Err(e)
            }
        }
    }
}

impl Drop for PendingOffer {
    fn drop(&mut self) {
        if let Some(h) = &self.handle {
            h.close();
        }
    }
}

// ---------------------------------------------------------------------------
// The host task.

enum TcpIn {
    Packet(SocketAddr, Vec<u8>),
    Closed(SocketAddr),
}

struct HostLoop {
    cfg: HostConfig,
    plan: CandidatePlan,
    udp: Option<Arc<UdpSocket>>,
    listener: Option<TcpListener>,
    tcp_out: HashMap<SocketAddr, mpsc::Sender<Vec<u8>>>,
    tcp_in_tx: mpsc::Sender<TcpIn>,
    tcp_in_rx: mpsc::Receiver<TcpIn>,
    cmd_rx: mpsc::Receiver<Cmd>,
    cmd_tx: mpsc::WeakSender<Cmd>,
    peers: HashMap<PeerId, PeerSlot>,
    next_id: PeerId,
    stun_waiters: HashMap<stun::TransactionId, (Instant, oneshot::Sender<SocketAddr>)>,
    stats: HostStats,
    stats_tx: watch::Sender<HostStats>,
}

struct Io<'a> {
    udp: Option<&'a UdpSocket>,
    tcp_out: &'a HashMap<SocketAddr, mpsc::Sender<Vec<u8>>>,
    stats: &'a mut HostStats,
}

impl Io<'_> {
    fn transmit(&mut self, t: str0m::net::Transmit) {
        match t.proto {
            Protocol::Udp => match self.udp {
                Some(s) => match s.try_send_to(&t.contents, t.destination) {
                    Ok(_) => self.stats.udp_tx_packets += 1,
                    Err(_) => self.stats.dropped_tx_packets += 1,
                },
                None => self.stats.dropped_tx_packets += 1,
            },
            Protocol::Tcp => {
                let framed = framing::frame(&t.contents);
                match (self.tcp_out.get(&t.destination), framed) {
                    (Some(w), Some(f)) if w.try_send(f).is_ok() => self.stats.tcp_tx_packets += 1,
                    _ => self.stats.dropped_tx_packets += 1,
                }
            }
            _ => self.stats.dropped_tx_packets += 1,
        }
    }
}

enum SlotState {
    Offering(Option<SdpPendingOffer>),
    Active,
}

struct MidState {
    info: MediaInfo,
    mid: Mid,
    pt: Option<Pt>,
    wall: WallclockMap,
    paused: bool,
}

struct PeerSlot {
    rtc: Rtc,
    state: SlotState,
    events: mpsc::UnboundedSender<PeerEvent>,
    policy: ChannelPolicy,
    mids: Vec<MidState>,
    channels: HashMap<ChannelId, String>,
    labels: HashMap<String, ChannelId>,
    pending: PendingMessages,
    created: Instant,
    connected: bool,
    last_rx: Instant,
    disconnected_since: Option<Instant>,
    next_timeout: Instant,
    stats: PeerStats,
    stats_tx: watch::Sender<PeerStats>,
    closed: Option<CloseReason>,
}

impl PeerSlot {
    fn emit(&mut self, e: PeerEvent) {
        if self.events.send(e).is_err() && self.closed.is_none() {
            // Nobody listens any more: the owner dropped the Peer.
            self.closed = Some(CloseReason::Local);
        }
    }

    fn close(&mut self, reason: CloseReason) {
        if self.closed.is_none() {
            self.closed = Some(reason);
        }
    }

    fn lane(&mut self, kind: TrackKind) -> &mut LaneStats {
        match kind {
            TrackKind::Video => &mut self.stats.video,
            TrackKind::Audio => &mut self.stats.audio,
        }
    }
}

impl HostLoop {
    async fn run(mut self) {
        let mut buf = vec![0u8; 2048];
        let mut housekeeping = tokio::time::interval(Duration::from_millis(200));
        housekeeping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let deadline = self
                .peers
                .values()
                .map(|p| p.next_timeout)
                .min()
                .unwrap_or_else(|| Instant::now() + Duration::from_secs(1));
            let udp = self.udp.clone();
            enum Wake {
                Cmd(Option<Cmd>),
                Udp(std::io::Result<(usize, SocketAddr)>),
                Accept(std::io::Result<(TcpStream, SocketAddr)>),
                Tcp(Option<TcpIn>),
                Timeout,
                Housekeeping,
            }
            let wake = tokio::select! {
                c = self.cmd_rx.recv() => Wake::Cmd(c),
                r = async { match &udp { Some(s) => s.recv_from(&mut buf).await, None => std::future::pending().await } } => Wake::Udp(r),
                a = async { match &self.listener { Some(l) => l.accept().await, None => std::future::pending().await } } => Wake::Accept(a),
                t = self.tcp_in_rx.recv() => Wake::Tcp(t),
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => Wake::Timeout,
                _ = housekeeping.tick() => Wake::Housekeeping,
            };
            let now = Instant::now();
            match wake {
                Wake::Cmd(None) => break,
                Wake::Cmd(Some(Cmd::Shutdown { reply })) => {
                    for p in self.peers.values_mut() {
                        p.close(CloseReason::HostShutdown);
                    }
                    self.reap(now);
                    let _ = reply.send(());
                    break;
                }
                Wake::Cmd(Some(cmd)) => self.command(cmd, now),
                Wake::Udp(Ok((n, source))) => {
                    self.stats.udp_rx_packets += 1;
                    let data = buf[..n].to_vec();
                    self.packet(Transport::Udp, source, &data, now);
                }
                Wake::Udp(Err(e)) => {
                    // ICMP port unreachable surfaces here on some platforms; not fatal.
                    tracing::debug!(error = %e, "udp recv error");
                }
                Wake::Accept(Ok((stream, remote))) => self.accept_tcp(stream, remote),
                Wake::Accept(Err(e)) => tracing::warn!(error = %e, "ice-tcp accept failed"),
                Wake::Tcp(Some(TcpIn::Packet(remote, data))) => {
                    self.stats.tcp_rx_packets += 1;
                    self.packet(Transport::TcpPassive, remote, &data, now);
                }
                Wake::Tcp(Some(TcpIn::Closed(remote))) => {
                    self.tcp_out.remove(&remote);
                }
                Wake::Tcp(None) => {}
                Wake::Timeout => {
                    let ids: Vec<PeerId> = self.peers.iter().filter(|(_, p)| p.next_timeout <= now).map(|(id, _)| *id).collect();
                    for id in ids {
                        self.with_peer(id, now, |slot, _| {
                            if let Err(e) = slot.rtc.handle_input(Input::Timeout(now)) {
                                slot.close(CloseReason::Error(e.to_string()));
                            }
                        });
                    }
                }
                Wake::Housekeeping => self.housekeeping(now),
            }
            self.reap(now);
        }
        tracing::debug!("webrtc host stopped");
    }

    /// Run `f` on a peer, then drain its output (the str0m invariant).
    fn with_peer(&mut self, id: PeerId, now: Instant, f: impl FnOnce(&mut PeerSlot, &mut Io)) {
        let Some(slot) = self.peers.get_mut(&id) else { return };
        let mut io = Io { udp: self.udp.as_deref(), tcp_out: &self.tcp_out, stats: &mut self.stats };
        f(slot, &mut io);
        drain(slot, &mut io, now);
    }

    fn accept_tcp(&mut self, stream: TcpStream, remote: SocketAddr) {
        if self.tcp_out.len() >= self.cfg.max_tcp_connections {
            tracing::warn!(%remote, "ice-tcp connection limit reached");
            return;
        }
        let _ = stream.set_nodelay(true);
        let (mut rd, mut wr) = stream.into_split();
        let (wtx, mut wrx) = mpsc::channel::<Vec<u8>>(1024);
        self.tcp_out.insert(remote, wtx);
        tokio::spawn(async move {
            while let Some(b) = wrx.recv().await {
                if wr.write_all(&b).await.is_err() {
                    break;
                }
            }
        });
        let in_tx = self.tcp_in_tx.clone();
        let idle = self.cfg.idle_timeout.max(Duration::from_secs(5)) + self.cfg.negotiation_timeout;
        tokio::spawn(async move {
            let mut dec = framing::Decoder::new();
            let mut buf = vec![0u8; 4096];
            loop {
                let n = match tokio::time::timeout(idle, rd.read(&mut buf)).await {
                    Ok(Ok(0)) | Ok(Err(_)) | Err(_) => break,
                    Ok(Ok(n)) => n,
                };
                match dec.push(&buf[..n]) {
                    Ok(frames) => {
                        for f in frames {
                            if in_tx.send(TcpIn::Packet(remote, f)).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(%remote, error = %e, "not an ice-tcp stream");
                        break;
                    }
                }
            }
            let _ = in_tx.send(TcpIn::Closed(remote)).await;
        });
    }

    fn packet(&mut self, t: Transport, source: SocketAddr, data: &[u8], now: Instant) {
        if t == Transport::Udp {
            if let Some((tx, mapped)) = stun::parse_binding_success(data) {
                if let Some((_, w)) = self.stun_waiters.remove(&tx) {
                    let _ = w.send(mapped);
                    return;
                }
            }
        }
        let Some(destination) = self.plan.destination_for(t, source) else {
            self.stats.unmatched_packets += 1;
            return;
        };
        let proto = match t {
            Transport::Udp => Protocol::Udp,
            Transport::TcpPassive => Protocol::Tcp,
        };
        let Ok(recv) = Receive::new(proto, source, destination, data) else {
            self.stats.unmatched_packets += 1;
            return;
        };
        let input = Input::Receive(now, recv);
        let Some(id) = self.peers.iter().find(|(_, p)| p.closed.is_none() && p.rtc.accepts(&input)).map(|(id, _)| *id) else {
            self.stats.unmatched_packets += 1;
            return;
        };
        self.with_peer(id, now, |slot, _| {
            slot.last_rx = now;
            if let Err(e) = slot.rtc.handle_input(input) {
                tracing::debug!(error = %e, "rtc rejected input");
            }
        });
    }

    fn housekeeping(&mut self, now: Instant) {
        let cfg = &self.cfg;
        for slot in self.peers.values_mut() {
            if slot.closed.is_some() {
                continue;
            }
            if slot.events.is_closed() {
                slot.close(CloseReason::Local);
            } else if !slot.connected && now.duration_since(slot.created) > cfg.negotiation_timeout {
                slot.close(CloseReason::NegotiationTimeout);
            } else if slot.connected && now.duration_since(slot.last_rx) > cfg.idle_timeout {
                slot.close(CloseReason::IdleTimeout);
            } else if slot.disconnected_since.is_some_and(|d| now.duration_since(d) > cfg.disconnect_grace) {
                slot.close(CloseReason::IceDisconnected);
            } else if !slot.rtc.is_alive() {
                slot.close(CloseReason::RemoteClosed);
            }
        }
        self.stun_waiters.retain(|_, (t, w)| now.duration_since(*t) < Duration::from_secs(30) && !w.is_closed());
        self.stats.peers = self.peers.len();
        self.stats.tcp_connections = self.tcp_out.len();
        self.stats_tx.send_if_modified(|s| {
            let changed = *s != self.stats;
            if changed {
                *s = self.stats.clone();
            }
            changed
        });
    }

    fn reap(&mut self, now: Instant) {
        let closed: Vec<PeerId> = self.peers.iter().filter(|(_, p)| p.closed.is_some()).map(|(id, _)| *id).collect();
        for id in closed {
            self.with_peer(id, now, |slot, _| slot.rtc.disconnect());
            if let Some(mut slot) = self.peers.remove(&id) {
                let reason = slot.closed.take().unwrap_or(CloseReason::Local);
                tracing::debug!(peer = id, ?reason, "peer closed");
                slot.stats.connected = false;
                let _ = slot.stats_tx.send(slot.stats.clone());
                let _ = slot.events.send(PeerEvent::Closed(reason));
            }
        }
    }

    fn command(&mut self, cmd: Cmd, now: Instant) {
        match cmd {
            Cmd::Answer { offer, opts, reply, answer_tx } => {
                let r = self.answer(&offer, opts, now);
                match r {
                    Ok((peer, answer)) => {
                        let _ = answer_tx.send(answer);
                        if let Err(Ok(peer)) = reply.send(Ok(peer)) {
                            peer.close();
                        }
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                    }
                }
            }
            Cmd::Offer { opts, reply } => {
                let r = self.offer(opts, now);
                let _ = reply.send(r);
            }
            Cmd::AcceptAnswer { peer, answer, reply } => {
                let mut result = Err(WebrtcError::PeerGone);
                self.with_peer(peer, now, |slot, _| result = accept_answer(slot, &answer));
                let _ = reply.send(result);
            }
            Cmd::Video { peer, frame } => {
                self.with_peer(peer, now, |slot, io| {
                    write_media(slot, io, now, TrackKind::Video, frame.rtp_time, &frame.data);
                });
            }
            Cmd::Audio { peer, packet } => {
                self.with_peer(peer, now, |slot, io| {
                    write_media(slot, io, now, TrackKind::Audio, packet.rtp_time, &packet.data);
                });
            }
            Cmd::Data { peer, msg, reply } => {
                let mut result = Err(WebrtcError::PeerGone);
                self.with_peer(peer, now, |slot, _| result = send_message(slot, msg));
                if let Some(r) = reply {
                    let _ = r.send(result);
                }
            }
            Cmd::SetPaused { peer, mid, kind, paused } => {
                self.with_peer(peer, now, |slot, _| {
                    let mut resumed_video = Vec::new();
                    for m in slot.mids.iter_mut() {
                        let hit = mid.as_deref().is_some_and(|x| x == m.info.mid) || kind.is_some_and(|k| k == m.info.kind);
                        if hit {
                            if m.paused && !paused && m.info.kind == TrackKind::Video && m.info.sends() {
                                resumed_video.push(m.info.mid.clone());
                            }
                            m.paused = paused;
                        }
                    }
                    if slot.connected {
                        for mid in resumed_video {
                            slot.emit(PeerEvent::KeyframeRequest { mid });
                        }
                    }
                });
            }
            Cmd::AddCandidate { peer, candidate, reply } => {
                let mut result = Err(WebrtcError::PeerGone);
                self.with_peer(peer, now, |slot, _| {
                    let c = candidate.trim();
                    let c = c.strip_prefix("a=").unwrap_or(c);
                    if c.is_empty() {
                        result = Ok(false);
                        return;
                    }
                    let c = if c.starts_with("candidate:") { c.to_string() } else { format!("candidate:{c}") };
                    result = match Candidate::from_sdp_string(&c) {
                        Ok(cand) => {
                            slot.rtc.add_remote_candidate(cand);
                            Ok(true)
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "ignoring remote candidate");
                            Ok(false)
                        }
                    };
                });
                let _ = reply.send(result);
            }
            Cmd::Close { peer } => {
                if let Some(p) = self.peers.get_mut(&peer) {
                    p.close(CloseReason::Local);
                }
            }
            Cmd::StunProbe { server, reply } => {
                let Some(udp) = &self.udp else { return };
                let tx = stun::new_transaction_id();
                if udp.try_send_to(&stun::binding_request(&tx), server).is_ok() {
                    self.stun_waiters.insert(tx, (now, reply));
                }
            }
            Cmd::Shutdown { .. } => unreachable!("handled in run"),
        }
    }

    fn new_rtc(&self, now: Instant, creds: Option<(String, String)>) -> Rtc {
        let mut cfg = Rtc::builder()
            .clear_codecs()
            .enable_opus(true, false)
            .set_stats_interval(Some(self.cfg.stats_interval));
        // Constrained Baseline first (what OpenH264 emits, §5.9), then
        // Baseline as a fallback match; both FU-A (packetization-mode=1).
        cfg.codec_config().add_h264(108.into(), Some(109.into()), true, 0x42e01f);
        cfg.codec_config().add_h264(127.into(), Some(121.into()), true, 0x42001f);
        if let Some((ufrag, pass)) = creds {
            cfg = cfg.set_local_ice_credentials(IceCreds { ufrag, pass });
        }
        cfg.build(now)
    }

    fn add_candidates(&self, rtc: &mut Rtc, srflx: &[SocketAddr]) {
        for a in &self.plan.udp {
            match Candidate::host(*a, "udp") {
                Ok(c) => {
                    rtc.add_local_candidate(c);
                }
                Err(e) => tracing::warn!(addr = %a, error = %e, "bad udp candidate"),
            }
        }
        for a in &self.plan.tcp {
            match Candidate::builder().tcp().host(*a).tcptype(TcpType::Passive).build() {
                Ok(c) => {
                    rtc.add_local_candidate(c);
                }
                Err(e) => tracing::warn!(addr = %a, error = %e, "bad tcp candidate"),
            }
        }
        for s in srflx {
            let Some(base) = self.plan.udp.iter().find(|b| b.is_ipv4() == s.is_ipv4() && !b.ip().is_loopback()).or(self.plan.udp.first()) else {
                continue;
            };
            match Candidate::server_reflexive(*s, *base, "udp") {
                Ok(c) => {
                    rtc.add_local_candidate(c);
                }
                Err(e) => tracing::warn!(addr = %s, error = %e, "bad srflx candidate"),
            }
        }
    }

    fn new_slot(&mut self, rtc: Rtc, state: SlotState, policy: ChannelPolicy, mids: Vec<MidState>, now: Instant) -> (PeerId, PeerHandle, mpsc::UnboundedReceiver<PeerEvent>) {
        let id = self.next_id;
        self.next_id += 1;
        let (etx, erx) = mpsc::unbounded_channel();
        let (stats_tx, stats_rx) = watch::channel(PeerStats::default());
        let media = Arc::new(mids.iter().map(|m| m.info.clone()).collect());
        let slot = PeerSlot {
            rtc,
            state,
            events: etx,
            policy,
            mids,
            channels: HashMap::new(),
            labels: HashMap::new(),
            pending: PendingMessages::default(),
            created: now,
            connected: false,
            last_rx: now,
            disconnected_since: None,
            next_timeout: now,
            stats: PeerStats::default(),
            stats_tx,
            closed: None,
        };
        self.peers.insert(id, slot);
        let tx = self.cmd_tx.upgrade().expect("host holds its own sender while running");
        (id, PeerHandle { id, tx, media, stats: stats_rx }, erx)
    }

    fn answer(&mut self, offer: &str, opts: AnswerOptions, now: Instant) -> Result<(Peer, String), WebrtcError> {
        if self.peers.len() >= self.cfg.max_peers {
            return Err(WebrtcError::PeerLimit(self.cfg.max_peers));
        }
        let mut sdp = Sdp::parse(offer)?;
        let summary = sdp::validate_offer(&sdp, OfferRequirements::default())?;
        let wants_video = opts.video && summary.media.iter().any(|m| m.kind == MediaKind::Video && m.offerer_receives());
        let wants_audio = opts.audio.is_some() && summary.media.iter().any(|m| m.kind == MediaKind::Audio && m.offerer_receives());
        sdp::validate_offer(&sdp, OfferRequirements { video: wants_video, audio: wants_audio })?;
        sdp.strip_unresolvable_candidates();
        if opts.audio.is_none() {
            sdp.set_direction_for_kind(MediaKind::Audio, Direction::Inactive);
        }
        if !opts.video {
            sdp.set_direction_for_kind(MediaKind::Video, Direction::Inactive);
        }
        // We never mirror the Reactor RXMT trailer (reactor §4.2).
        sdp.remove_session_attr("x-reactor-frame-metadata");

        let mut rtc = self.new_rtc(now, opts.ice_credentials.clone());
        self.add_candidates(&mut rtc, &[]);
        let offer = SdpOffer::from_sdp_string(&sdp.to_string()).map_err(|e| WebrtcError::Rtc(e.to_string()))?;
        let answer = rtc.sdp_api().accept_offer(offer).map_err(|e| WebrtcError::Rtc(e.to_string()))?;

        let mut answer_sdp = Sdp::parse(&answer.to_sdp_string())?;
        answer_sdp.finish_candidates();
        if let Some(layout) = opts.audio {
            answer_sdp.set_opus_stereo(layout == AudioLayout::Stereo);
        }

        let mids = summary
            .media
            .iter()
            .filter(|m| !m.rejected)
            .filter_map(|m| {
                let kind = match m.kind {
                    MediaKind::Video => TrackKind::Video,
                    MediaKind::Audio => TrackKind::Audio,
                    _ => return None,
                };
                let mid = Mid::from(m.mid.as_str());
                let dir = rtc.media(mid).map(|x| from_rtc_dir(x.direction())).unwrap_or(Direction::Inactive);
                Some(mid_state(mid, kind, dir, opts.start_paused))
            })
            .collect();
        let (id, handle, events) = self.new_slot(rtc, SlotState::Active, opts.channels, mids, now);
        self.with_peer(id, now, |_, _| {});
        Ok((Peer { handle, events }, answer_sdp.to_string()))
    }

    fn offer(&mut self, opts: OfferOptions, now: Instant) -> Result<(PendingOffer, String), WebrtcError> {
        if self.peers.len() >= self.cfg.max_peers {
            return Err(WebrtcError::PeerLimit(self.cfg.max_peers));
        }
        let mut rtc = self.new_rtc(now, None);
        self.add_candidates(&mut rtc, &opts.srflx);
        let mut api = rtc.sdp_api();
        let mut mids = Vec::new();
        if let Some(dir) = opts.video {
            let mid = api.add_media(RtcMediaKind::Video, to_rtc_dir(dir), None, None, None);
            mids.push(mid_state(mid, TrackKind::Video, dir, false));
        }
        if let Some((dir, _)) = opts.audio {
            let mid = api.add_media(RtcMediaKind::Audio, to_rtc_dir(dir), None, None, None);
            mids.push(mid_state(mid, TrackKind::Audio, dir, false));
        }
        for label in &opts.channels {
            api.add_channel(label.clone());
        }
        let (offer, pending) = api.apply().ok_or_else(|| WebrtcError::Config("offer has no m-lines (no media and no channels)".into()))?;
        let mut sdp = Sdp::parse(&offer.to_sdp_string())?;
        sdp.finish_candidates();
        sdp.prefer_h264();
        if let Some((_, layout)) = opts.audio {
            sdp.set_opus_stereo(layout == AudioLayout::Stereo);
        }
        let policy = ChannelPolicy::Only(opts.channels.clone());
        let (id, handle, events) = self.new_slot(rtc, SlotState::Offering(Some(pending)), policy, mids, now);
        self.with_peer(id, now, |_, _| {});
        Ok((PendingOffer { handle: Some(handle), events: Some(events) }, sdp.to_string()))
    }
}

fn mid_state(mid: Mid, kind: TrackKind, direction: Direction, paused: bool) -> MidState {
    let clock = match kind {
        TrackKind::Video => VIDEO_CLOCK_HZ,
        TrackKind::Audio => AUDIO_CLOCK_HZ,
    };
    MidState { info: MediaInfo { mid: mid.to_string(), kind, direction }, mid, pt: None, wall: WallclockMap::new(clock), paused }
}

fn to_rtc_dir(d: Direction) -> RtcDirection {
    match d {
        Direction::SendRecv => RtcDirection::SendRecv,
        Direction::SendOnly => RtcDirection::SendOnly,
        Direction::RecvOnly => RtcDirection::RecvOnly,
        Direction::Inactive => RtcDirection::Inactive,
    }
}

fn from_rtc_dir(d: RtcDirection) -> Direction {
    match d {
        RtcDirection::SendRecv => Direction::SendRecv,
        RtcDirection::SendOnly => Direction::SendOnly,
        RtcDirection::RecvOnly => Direction::RecvOnly,
        RtcDirection::Inactive => Direction::Inactive,
    }
}

fn accept_answer(slot: &mut PeerSlot, answer: &str) -> Result<Vec<MediaInfo>, WebrtcError> {
    let SlotState::Offering(pending) = &mut slot.state else {
        return Err(WebrtcError::Rtc("peer is not waiting for an answer".into()));
    };
    let pending = pending.take().ok_or_else(|| WebrtcError::Rtc("answer already applied".into()))?;
    let parsed = Sdp::parse(answer)?;
    if parsed.media.is_empty() {
        return Err(sdp::SdpError::NoMedia.into());
    }
    let ans = SdpAnswer::from_sdp_string(&parsed.to_string()).map_err(|e| WebrtcError::Rtc(e.to_string()))?;
    slot.rtc.sdp_api().accept_answer(pending, ans).map_err(|e| WebrtcError::Rtc(e.to_string()))?;
    slot.state = SlotState::Active;
    for m in slot.mids.iter_mut() {
        if let Some(media) = slot.rtc.media(m.mid) {
            m.info.direction = from_rtc_dir(media.direction());
        }
    }
    Ok(slot.mids.iter().map(|m| m.info.clone()).collect())
}

fn send_message(slot: &mut PeerSlot, msg: ChannelMessage) -> Result<(), WebrtcError> {
    let Some(id) = slot.labels.get(&msg.label).copied() else {
        return slot.pending.push(msg).map_err(|_| WebrtcError::Backpressure);
    };
    let Some(mut ch) = slot.rtc.channel(id) else {
        return Err(WebrtcError::ChannelClosed(msg.label));
    };
    ch.write(msg.binary, &msg.data).map_err(|e| WebrtcError::Rtc(e.to_string()))?;
    slot.stats.messages_out += 1;
    Ok(())
}

fn write_media(slot: &mut PeerSlot, io: &mut Io, now: Instant, kind: TrackKind, rtp_time: u64, data: &Bytes) {
    let targets: Vec<usize> = slot.mids.iter().enumerate().filter(|(_, m)| m.info.kind == kind).map(|(i, _)| i).collect();
    if targets.is_empty() {
        slot.lane(kind).dropped_paused += 1;
        return;
    }
    for i in targets {
        let (sends, paused) = (slot.mids[i].info.sends(), slot.mids[i].paused);
        if !sends || paused {
            slot.lane(kind).dropped_paused += 1;
            continue;
        }
        if !slot.connected {
            slot.lane(kind).dropped_not_connected += 1;
            continue;
        }
        let mid = slot.mids[i].mid;
        let pt = match slot.mids[i].pt {
            Some(pt) => Some(pt),
            None => {
                let pt = slot.rtc.writer(mid).and_then(|w| pick_pt(&w, kind));
                slot.mids[i].pt = pt;
                pt
            }
        };
        let Some(pt) = pt else {
            slot.lane(kind).write_errors += 1;
            continue;
        };
        let wall = slot.mids[i].wall.wallclock(now, rtp_time);
        let freq = match kind {
            TrackKind::Video => Frequency::NINETY_KHZ,
            TrackKind::Audio => Frequency::FORTY_EIGHT_KHZ,
        };
        let res = slot.rtc.writer(mid).map(|w| w.write(pt, wall, MediaTime::new(rtp_time, freq), Arc::<[u8]>::from(&data[..])));
        match res {
            Some(Ok(())) => slot.lane(kind).written += 1,
            None => slot.lane(kind).write_errors += 1,
            Some(Err(e)) => {
                tracing::debug!(error = %e, "media write failed");
                slot.lane(kind).write_errors += 1;
            }
        }
        // One mutation, one drain.
        drain(slot, io, now);
    }
}

fn pick_pt(w: &str0m::media::Writer, kind: TrackKind) -> Option<Pt> {
    let params: Vec<_> = w.payload_params().cloned().collect();
    match kind {
        TrackKind::Video => {
            let h264: Vec<_> = params.iter().filter(|p| p.spec().codec == Codec::H264).collect();
            h264.iter()
                .find(|p| p.spec().format.profile_level_id == Some(0x42e01f) && p.spec().format.packetization_mode == Some(1))
                .or_else(|| h264.iter().find(|p| p.spec().format.packetization_mode == Some(1)))
                .or_else(|| h264.first())
                .map(|p| p.pt())
        }
        TrackKind::Audio => params.iter().find(|p| p.spec().codec == Codec::Opus).map(|p| p.pt()),
    }
}

/// Drain `poll_output` until `Timeout`, dispatching transmits and events.
fn drain(slot: &mut PeerSlot, io: &mut Io, now: Instant) {
    loop {
        let out = match slot.rtc.poll_output() {
            Ok(o) => o,
            Err(e) => {
                slot.close(CloseReason::Error(e.to_string()));
                slot.next_timeout = now + Duration::from_secs(1);
                return;
            }
        };
        match out {
            Output::Timeout(t) => {
                slot.next_timeout = t;
                return;
            }
            Output::Transmit(t) => io.transmit(t),
            Output::Event(e) => event(slot, e, now),
        }
    }
}

fn event(slot: &mut PeerSlot, e: Event, now: Instant) {
    match e {
        Event::Connected => {
            slot.connected = true;
            slot.stats.connected = true;
            slot.disconnected_since = None;
            slot.emit(PeerEvent::Connected);
            let video: Vec<String> = slot
                .mids
                .iter()
                .filter(|m| m.info.kind == TrackKind::Video && m.info.sends() && !m.paused)
                .map(|m| m.info.mid.clone())
                .collect();
            for mid in video {
                slot.emit(PeerEvent::KeyframeRequest { mid });
            }
            let _ = slot.stats_tx.send(slot.stats.clone());
        }
        Event::IceConnectionStateChange(s) => match s {
            IceConnectionState::Disconnected => {
                slot.disconnected_since.get_or_insert(now);
            }
            IceConnectionState::Connected | IceConnectionState::Completed => slot.disconnected_since = None,
            _ => {}
        },
        Event::ChannelOpen(id, label) => {
            if !slot.policy.accepts(&label) {
                tracing::debug!(%label, "closing data channel with an unexpected label");
                slot.rtc.direct_api().close_data_channel(id);
                return;
            }
            slot.channels.insert(id, label.clone());
            slot.labels.insert(label.clone(), id);
            for msg in slot.pending.take_label(&label) {
                if let Err(e) = send_message(slot, msg) {
                    tracing::debug!(error = %e, "queued message dropped");
                }
            }
            slot.emit(PeerEvent::ChannelOpen { label });
        }
        Event::ChannelData(d) => {
            let Some(label) = slot.channels.get(&d.id).cloned() else { return };
            slot.stats.messages_in += 1;
            slot.emit(PeerEvent::Message(ChannelMessage { label, binary: d.binary, data: Bytes::from(d.data) }));
        }
        Event::ChannelClose(id) => {
            if let Some(label) = slot.channels.remove(&id) {
                slot.labels.remove(&label);
                slot.emit(PeerEvent::ChannelClose { label });
            }
        }
        Event::KeyframeRequest(k) => {
            slot.stats.keyframe_requests += 1;
            slot.emit(PeerEvent::KeyframeRequest { mid: k.mid.to_string() });
        }
        Event::MediaData(d) => {
            let kind = if d.params.spec().codec.is_audio() { TrackKind::Audio } else { TrackKind::Video };
            let keyframe = d.is_keyframe();
            slot.emit(PeerEvent::Media { mid: d.mid.to_string(), kind, rtp_time: d.time.numer(), keyframe, data: Bytes::copy_from_slice(&d.data) });
        }
        Event::PeerStats(s) => {
            slot.stats.bytes_rx = s.bytes_rx;
            slot.stats.bytes_tx = s.bytes_tx;
            slot.stats.rtt_ms = s.rtt.map(|r| r.as_secs_f64() * 1000.0);
            slot.stats.transport = s.selected_candidate_pair.map(|p| match p.protocol {
                Protocol::Udp => "udp".to_string(),
                _ => "tcp".to_string(),
            });
            let _ = slot.stats_tx.send(slot.stats.clone());
        }
        Event::MediaEgressStats(s) => {
            let kind = slot.mids.iter().find(|m| m.mid == s.mid).map(|m| m.info.kind);
            if let Some(kind) = kind {
                let lane = slot.lane(kind);
                lane.bytes = s.bytes;
                lane.packets = s.packets;
                lane.nacks = s.nacks;
                lane.plis = s.plis;
                lane.firs = s.firs;
            }
        }
        Event::Closed => slot.close(CloseReason::RemoteClosed),
        _ => {}
    }
}

#[allow(dead_code)]
fn _assert_send() {
    fn is_send<T: Send>() {}
    is_send::<RtcHost>();
    is_send::<Peer>();
    is_send::<PeerHandle>();
}
