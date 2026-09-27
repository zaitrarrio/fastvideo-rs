//! The fast-h3 queue-and-playout contract (reactor §4bis, design §5.5) for
//! clip models (H3, LTX, FastWan), over [`ClipEngine::build`].
//!
//! - A clip moves **generation queue** (cap 20) → **build** (one at a time,
//!   front first, only while a peer is connected, only while the playout
//!   queue has room: submit-time reservation) → **playout queue** (cap 10)
//!   → **playing** (in neither queue). A build whose entry was popped is
//!   discarded.
//! - Nothing plays unless `play` arms a clip or autoplay is on (a standing
//!   `play`). After a clip ends with nothing armed, the stream holds on one
//!   black frame (fast-h3's `output.flush()`).
//! - Playback emits 3-frame slices, each with exactly `3·48000/fps` samples
//!   (6000 at 24 fps), through the session pacer: A/V lockstep.
//! - Refusals are broadcast `command_error{command, reason}` plus a bodyless
//!   ack, never raised (older v0 clients never see failure frames).
//!
//! This is an actor: one task owns the state, commands arrive through a
//! channel, builds and playback run as tasks that report back. When WP-12's
//! engine-side `ClipSession` queue lands, this file is where it plugs in.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use fastvideo_protocol::{canvas_for_aspect, ApiError, CanvasCaps, WIRE_AUDIO_RATE};
use serde::Serialize;
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::commands::{ClipBounds, ASPECTS};
use crate::driver::{Driver, Outbox, Outcome};
use crate::engine::{BuildRequest, ClipEngine, WireClip};
use crate::media::MediaPipeline;
use crate::wire::ServerMsg;

pub const GENERATION_CAPACITY: usize = 20;
pub const PLAYOUT_CAPACITY: usize = 10;
/// Frames per emit slice (fast-h3 `_emit_clip`).
pub const SLICE_FRAMES: usize = 3;

/// `ClipInfo`, embedded whole in every clip message.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClipInfo {
    pub clip_id: Uuid,
    pub prompt: String,
    pub metadata: String,
    pub frames: u32,
    pub seconds: f64,
    pub seed: u64,
    pub ready: bool,
}

impl ClipInfo {
    fn json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// Static inputs of a clip session.
#[derive(Clone, Debug)]
pub struct ClipDriverConfig {
    pub fps: u32,
    pub bounds: ClipBounds,
    pub seed: u64,
    pub canvas_caps: CanvasCaps,
    /// Short edge the aspects resolve at.
    pub short_edge: u32,
    pub aspect: String,
    pub canvas: (u32, u32),
    pub has_audio: bool,
}

/// `(width, height)` for one of [`ASPECTS`].
pub fn aspect_canvas(caps: &CanvasCaps, aspect: &str, short_edge: u32) -> Option<(u32, u32)> {
    let (w, h) = aspect.split_once(':')?;
    let (w, h): (f64, f64) = (w.parse().ok()?, h.parse().ok()?);
    (w > 0.0 && h > 0.0).then(|| canvas_for_aspect(caps, w / h, short_edge))
}

enum Msg {
    Command { name: String, args: Map<String, Value>, reply: oneshot::Sender<Outcome> },
    Greet(u32),
    Peers(usize),
    BuildDone { id: Uuid, res: Result<WireClip, ApiError> },
    PlayDone { id: Uuid },
    Close(oneshot::Sender<()>),
}

/// The clip-mode [`Driver`].
pub struct ClipDriver {
    tx: mpsc::UnboundedSender<Msg>,
}

impl ClipDriver {
    pub fn start(
        cfg: ClipDriverConfig,
        engine: Arc<dyn ClipEngine>,
        media: Arc<MediaPipeline>,
        out: Outbox,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let clip_seconds = cfg.bounds.default_s;
        let actor = Actor {
            clip_seconds,
            seed: cfg.seed,
            aspect: cfg.aspect.clone(),
            canvas: cfg.canvas,
            cfg,
            engine,
            media,
            out,
            tx: tx.clone(),
            gen: VecDeque::new(),
            building: None,
            playout: VecDeque::new(),
            playing: None,
            armed: None,
            autoplay: false,
            clips_played: 0,
            seconds_sent: 0.0,
            peers: 0,
            closed: false,
        };
        tokio::spawn(actor.run(rx));
        Self { tx }
    }
}

#[async_trait]
impl Driver for ClipDriver {
    async fn command(&self, _conn: u32, name: &str, args: Map<String, Value>) -> Outcome {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Msg::Command { name: name.to_owned(), args, reply }).is_err() {
            return Outcome::Error { code: "internal_error".into(), message: "session closed".into() };
        }
        rx.await.unwrap_or(Outcome::Error { code: "internal_error".into(), message: "session closed".into() })
    }
    fn greet(&self, conn: u32) {
        let _ = self.tx.send(Msg::Greet(conn));
    }
    fn peers_changed(&self, connected: usize) {
        let _ = self.tx.send(Msg::Peers(connected));
    }
    async fn close(&self) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Msg::Close(tx)).is_ok() {
            let _ = rx.await;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Armed {
    Front,
    Id(Uuid),
}

struct Actor {
    cfg: ClipDriverConfig,
    engine: Arc<dyn ClipEngine>,
    media: Arc<MediaPipeline>,
    out: Outbox,
    tx: mpsc::UnboundedSender<Msg>,
    gen: VecDeque<ClipInfo>,
    building: Option<(ClipInfo, (u32, u32), JoinHandle<()>)>,
    playout: VecDeque<(ClipInfo, Arc<WireClip>)>,
    playing: Option<(ClipInfo, JoinHandle<()>)>,
    armed: Option<Armed>,
    autoplay: bool,
    clip_seconds: f64,
    seed: u64,
    aspect: String,
    canvas: (u32, u32),
    clips_played: u64,
    seconds_sent: f64,
    peers: usize,
    closed: bool,
}

fn refuse(out: &Outbox, command: &str, reason: impl Into<String>) -> Outcome {
    out.broadcast(ServerMsg::broadcast(
        "command_error",
        json!({"command": command, "reason": reason.into()}),
    ));
    Outcome::Ack
}

fn reply(t: &str, data: Value) -> Outcome {
    Outcome::Reply(t.to_owned(), data)
}

impl Actor {
    async fn run(mut self, mut rx: mpsc::UnboundedReceiver<Msg>) {
        while let Some(m) = rx.recv().await {
            match m {
                Msg::Command { name, args, reply } => {
                    let o = if self.closed {
                        Outcome::Error { code: "internal_error".into(), message: "session closed".into() }
                    } else {
                        self.command(&name, &args).await
                    };
                    let _ = reply.send(o);
                }
                Msg::Greet(conn) => {
                    self.out.to(conn, self.state_msg());
                    self.out.to(conn, self.queue_msg());
                }
                Msg::Peers(n) => {
                    self.peers = n;
                    self.pump();
                }
                Msg::BuildDone { id, res } => self.build_done(id, res),
                Msg::PlayDone { id } => self.play_done(id).await,
                Msg::Close(done) => {
                    self.closed = true;
                    self.abort_all();
                    self.engine.close();
                    let _ = done.send(());
                    break;
                }
            }
        }
        self.abort_all();
    }

    fn abort_all(&mut self) {
        if let Some((_, _, h)) = self.building.take() {
            h.abort();
        }
        if let Some((_, h)) = self.playing.take() {
            h.abort();
        }
    }

    fn fps(&self) -> u32 {
        self.cfg.fps.max(1)
    }

    fn has_queued(&self) -> bool {
        !self.gen.is_empty() || self.building.is_some() || !self.playout.is_empty()
    }

    fn valid_commands(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        let gen_len = self.gen.len() + usize::from(self.building.is_some());
        if gen_len < GENERATION_CAPACITY {
            v.push("enqueue");
        }
        if self.playing.is_none() && self.has_queued() {
            v.push("play");
        }
        if self.has_queued() {
            v.push("pop");
            v.push("move");
        }
        if self.playing.is_some() {
            v.push("stop");
        }
        v.extend(["get_queue", "get_state", "set_clip_seconds", "set_seed", "set_autoplay"]);
        if !self.has_queued() && self.playing.is_none() {
            v.push("set_canvas");
        }
        v.push("reset");
        v
    }

    fn state_json(&self) -> Value {
        json!({
            "clip_seconds": self.clip_seconds,
            "clip_seconds_min": self.cfg.bounds.min_s,
            "clip_seconds_max": self.cfg.bounds.max_s,
            "seed": self.seed,
            "autoplay": self.autoplay,
            "aspect": self.aspect,
            "width": self.canvas.0,
            "height": self.canvas.1,
            "playing": self.playing.is_some(),
            "playing_clip_id": self.playing.as_ref().map(|(c, _)| c.clip_id),
            "generation_queued": self.gen.len() + usize::from(self.building.is_some()),
            "generation_capacity": GENERATION_CAPACITY,
            "playout_queued": self.playout.len(),
            "playout_capacity": PLAYOUT_CAPACITY,
            "clips_played": self.clips_played,
            "seconds_sent": self.seconds_sent,
            "valid_commands": self.valid_commands(),
        })
    }

    fn queue_json(&self) -> Value {
        let generation: Vec<Value> = self
            .building
            .iter()
            .map(|(c, _, _)| c.json())
            .chain(self.gen.iter().map(ClipInfo::json))
            .collect();
        json!({
            "generation": generation,
            "playout": self.playout.iter().map(|(c, _)| c.json()).collect::<Vec<_>>(),
            "playing": self.playing.as_ref().map(|(c, _)| c.json()),
        })
    }

    fn state_msg(&self) -> ServerMsg {
        ServerMsg::broadcast("state_update", self.state_json())
    }

    fn queue_msg(&self) -> ServerMsg {
        ServerMsg::broadcast("queue_update", self.queue_json())
    }

    fn broadcast_state(&self) {
        self.out.broadcast(self.state_msg());
    }

    fn broadcast_queue(&self) {
        self.out.broadcast(self.queue_msg());
    }

    fn snap(&self, seconds: f64) -> Result<(u32, f64), String> {
        let frames = self.engine.frames_for(Some(seconds)).map_err(|e| e.message)?;
        Ok((frames, f64::from(frames) / f64::from(self.fps())))
    }

    async fn command(&mut self, name: &str, a: &Map<String, Value>) -> Outcome {
        let str_arg = |k: &str| a.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
        let u64_arg = |k: &str| a.get(k).and_then(Value::as_u64);
        match name {
            "enqueue" => {
                if self.gen.len() + usize::from(self.building.is_some()) >= GENERATION_CAPACITY {
                    return refuse(&self.out, name, "the generation queue is full");
                }
                let secs = a.get("seconds").and_then(Value::as_f64).unwrap_or(self.clip_seconds);
                let (frames, seconds) = match self.snap(secs) {
                    Ok(x) => x,
                    Err(e) => return refuse(&self.out, name, e),
                };
                let clip = ClipInfo {
                    clip_id: Uuid::new_v4(),
                    prompt: str_arg("prompt"),
                    metadata: str_arg("metadata"),
                    frames,
                    seconds,
                    seed: u64_arg("seed").unwrap_or(self.seed),
                    ready: false,
                };
                let pos = u64_arg("position").map_or(self.gen.len(), |p| (p as usize).min(self.gen.len()));
                self.gen.insert(pos, clip.clone());
                self.broadcast_queue();
                self.broadcast_state();
                self.pump();
                reply("clip_queued", json!({"clip": clip.json()}))
            }
            "play" => {
                let id = str_arg("clip_id");
                let target = if id.trim().is_empty() {
                    if !self.has_queued() {
                        return refuse(&self.out, name, "nothing is queued");
                    }
                    Armed::Front
                } else {
                    let Ok(u) = Uuid::parse_str(id.trim()) else {
                        return refuse(&self.out, name, format!("unknown clip {id}"));
                    };
                    let known = self.gen.iter().any(|c| c.clip_id == u)
                        || self.building.as_ref().is_some_and(|(c, _, _)| c.clip_id == u)
                        || self.playout.iter().any(|(c, _)| c.clip_id == u);
                    if !known {
                        return refuse(&self.out, name, format!("unknown clip {id}"));
                    }
                    Armed::Id(u)
                };
                self.armed = Some(target);
                self.try_start();
                self.broadcast_queue();
                self.broadcast_state();
                Outcome::Ack
            }
            "pop" => {
                let Some(u) = Uuid::parse_str(str_arg("clip_id").trim()).ok() else {
                    return refuse(&self.out, name, "unknown clip");
                };
                let popped = if let Some(i) = self.gen.iter().position(|c| c.clip_id == u) {
                    self.gen.remove(i)
                } else if self.building.as_ref().is_some_and(|(c, _, _)| c.clip_id == u) {
                    // Discard the build.
                    self.building.take().map(|(c, _, h)| {
                        h.abort();
                        c
                    })
                } else if let Some(i) = self.playout.iter().position(|(c, _)| c.clip_id == u) {
                    self.playout.remove(i).map(|(c, _)| c)
                } else if self.playing.as_ref().is_some_and(|(c, _)| c.clip_id == u) {
                    return refuse(&self.out, name, "the clip is playing; stop it instead");
                } else {
                    return refuse(&self.out, name, "unknown clip");
                };
                if self.armed == Some(Armed::Id(u)) {
                    self.armed = None;
                }
                self.pump();
                self.broadcast_queue();
                self.broadcast_state();
                reply("clip_popped", json!({"clip": popped.map(|c| c.json())}))
            }
            "move" => {
                let Some(u) = Uuid::parse_str(str_arg("clip_id").trim()).ok() else {
                    return refuse(&self.out, name, "unknown clip");
                };
                let pos = u64_arg("position").unwrap_or(0) as usize;
                let (clip, queue, at) = if let Some(i) = self.gen.iter().position(|c| c.clip_id == u) {
                    let c = self.gen.remove(i).expect("index from position");
                    let at = pos.min(self.gen.len());
                    self.gen.insert(at, c.clone());
                    (c, "generation", at)
                } else if let Some(i) = self.playout.iter().position(|(c, _)| c.clip_id == u) {
                    let c = self.playout.remove(i).expect("index from position");
                    let at = pos.min(self.playout.len());
                    let info = c.0.clone();
                    self.playout.insert(at, c);
                    (info, "playout", at)
                } else {
                    return refuse(&self.out, name, "the clip is not queued");
                };
                self.broadcast_queue();
                let queue_view = self.queue_json();
                reply("clip_moved", json!({"clip": clip.json(), "queue": queue, "position": at, "queues": queue_view}))
            }
            "stop" => {
                let Some((clip, h)) = self.playing.take() else {
                    return refuse(&self.out, name, "nothing is playing");
                };
                h.abort();
                self.armed = None;
                self.media.flush();
                self.out.broadcast(ServerMsg::broadcast("clip_stopped", json!({"clip": clip.json()})));
                self.broadcast_state();
                Outcome::Ack
            }
            "get_queue" => reply("queue_update", self.queue_json()),
            "get_state" => reply("state_update", self.state_json()),
            "set_clip_seconds" => {
                let secs = a.get("seconds").and_then(Value::as_f64).unwrap_or(self.clip_seconds);
                match self.snap(secs) {
                    Ok((frames, seconds)) => {
                        self.clip_seconds = seconds;
                        self.broadcast_state();
                        reply("clip_length_accepted", json!({"clip_seconds": seconds, "frames": frames}))
                    }
                    Err(e) => refuse(&self.out, name, e),
                }
            }
            "set_seed" => {
                self.seed = u64_arg("seed").unwrap_or(self.seed);
                self.broadcast_state();
                reply("seed_accepted", json!({"seed": self.seed}))
            }
            "set_autoplay" => {
                self.autoplay = a.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                self.try_start();
                self.broadcast_state();
                reply("autoplay_accepted", json!({"enabled": self.autoplay}))
            }
            "set_canvas" => {
                if self.has_queued() || self.playing.is_some() {
                    return refuse(&self.out, name, "set_canvas needs empty queues and nothing playing");
                }
                let aspect = str_arg("aspect");
                if !ASPECTS.contains(&aspect.as_str()) {
                    return refuse(&self.out, name, format!("unknown aspect {aspect}"));
                }
                let Some(canvas) = aspect_canvas(&self.cfg.canvas_caps, &aspect, self.cfg.short_edge) else {
                    return refuse(&self.out, name, format!("unknown aspect {aspect}"));
                };
                self.aspect = aspect.clone();
                self.canvas = canvas;
                self.broadcast_state();
                reply("canvas_accepted", json!({"aspect": aspect, "width": canvas.0, "height": canvas.1}))
            }
            "reset" => {
                let cleared = self.gen.len() + usize::from(self.building.is_some()) + self.playout.len();
                let was_playing = self.playing.is_some();
                self.abort_all();
                self.gen.clear();
                self.playout.clear();
                self.armed = None;
                if was_playing {
                    self.media.flush();
                }
                self.broadcast_queue();
                self.broadcast_state();
                reply("session_reset", json!({"cleared_clips": cleared, "was_playing": was_playing}))
            }
            other => Outcome::Error { code: "invalid_command".into(), message: format!("unknown command `{other}`") },
        }
    }

    /// Starts the next build when allowed.
    fn pump(&mut self) {
        if self.closed || self.building.is_some() || self.peers == 0 {
            return;
        }
        // Submit-time reservation: the build's output needs a playout slot.
        if self.playout.len() >= PLAYOUT_CAPACITY {
            return;
        }
        let Some(clip) = self.gen.pop_front() else { return };
        let req = BuildRequest {
            prompt: clip.prompt.clone(),
            seed: clip.seed,
            seconds: clip.seconds,
            canvas: self.canvas,
        };
        let engine = self.engine.clone();
        let tx = self.tx.clone();
        let id = clip.clip_id;
        let h = tokio::spawn(async move {
            let res = engine.build(req).await;
            let _ = tx.send(Msg::BuildDone { id, res });
        });
        self.building = Some((clip, self.canvas, h));
    }

    fn build_done(&mut self, id: Uuid, res: Result<WireClip, ApiError>) {
        let Some((clip, _, _)) = self.building.take_if(|(c, _, _)| c.clip_id == id) else {
            // Popped or reset while building: discarded.
            return;
        };
        match res {
            Ok(w) => {
                let mut c = clip;
                c.ready = true;
                c.frames = w.frames.len() as u32;
                self.playout.push_back((c.clone(), Arc::new(w)));
                self.out.broadcast(ServerMsg::broadcast("clip_generated", json!({"clip": c.json()})));
                self.try_start();
            }
            Err(e) => {
                tracing::warn!(clip = %id, error = %e.message, "clip build failed");
                self.out.broadcast(ServerMsg::broadcast(
                    "clip_failed",
                    json!({"clip": clip.json(), "reason": e.message}),
                ));
                if self.armed == Some(Armed::Id(id)) {
                    self.armed = None;
                }
            }
        }
        self.broadcast_queue();
        self.broadcast_state();
        self.pump();
    }

    /// Plays the armed (or, with autoplay, the front) ready clip if idle.
    fn try_start(&mut self) {
        if self.closed || self.playing.is_some() {
            return;
        }
        let idx = match &self.armed {
            Some(Armed::Id(u)) => self.playout.iter().position(|(c, _)| c.clip_id == *u),
            Some(Armed::Front) => (!self.playout.is_empty()).then_some(0),
            None if self.autoplay => (!self.playout.is_empty()).then_some(0),
            None => None,
        };
        let Some(i) = idx else { return };
        let Some((clip, w)) = self.playout.remove(i) else { return };
        self.armed = None;
        let media = self.media.clone();
        let tx = self.tx.clone();
        let id = clip.clip_id;
        let fps = self.fps();
        let h = tokio::spawn(async move {
            play(&media, &w, fps).await;
            let _ = tx.send(Msg::PlayDone { id });
        });
        self.out.broadcast(ServerMsg::broadcast("clip_started", json!({"clip": clip.json()})));
        self.playing = Some((clip, h));
        // A playout slot freed up.
        self.pump();
    }

    async fn play_done(&mut self, id: Uuid) {
        let Some((clip, _)) = self.playing.take_if(|(c, _)| c.clip_id == id) else {
            return;
        };
        self.clips_played += 1;
        self.seconds_sent += clip.seconds;
        self.out.broadcast(ServerMsg::broadcast("clip_finished", json!({"clip": clip.json()})));
        self.try_start();
        if self.playing.is_none() {
            // Nothing armed: hold on black.
            self.media.black().await;
        }
        self.broadcast_queue();
        self.broadcast_state();
    }
}

/// Emits a clip in 3-frame slices with their exact audio, then waits until
/// the pacer released the last frame.
async fn play(media: &MediaPipeline, w: &WireClip, fps: u32) {
    let spf = (WIRE_AUDIO_RATE / fps.max(1)) as usize;
    for (i, chunk) in w.frames.chunks(SLICE_FRAMES).enumerate() {
        let audio = w.audio.as_ref().map(|a| {
            let lo = (i * SLICE_FRAMES * spf).min(a.len());
            let hi = (lo + chunk.len() * spf).min(a.len());
            let mut s = a[lo..hi].to_vec();
            s.resize(chunk.len() * spf, 0.0);
            s
        });
        media.push(chunk.to_vec(), audio).await;
    }
    media.drained().await;
}
