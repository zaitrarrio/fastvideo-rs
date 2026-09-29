//! Duplex mode (design §5.11): client input tracks into a duplex model.
//!
//! The model's input tracks are declared like its output tracks, in all
//! three places, with the direction flipped: `capabilities.tracks`
//! (`sendonly`, client perspective), `track_map` and `x-reactor.tracks`
//! (`in`, model perspective). Names: [`INPUT_VIDEO_TRACK`] and
//! [`INPUT_AUDIO_TRACK`].
//!
//! A client publishes with `PublishTrack{name}`: first come, first served
//! per track name, as RT (`publish_refused` "track already published"
//! otherwise, or for a name the model does not declare). `UnpublishTrack`
//! releases the slot; a closed connection releases its slots. Only the
//! publisher's media reaches the model: every connection's answer receives
//! (camera and microphone m-lines, bitrate capped by `b=AS`), and its
//! [`Ingest`](fastvideo_webrtc::ingest::Ingest) forwards a track only while
//! that connection holds the slot.
//!
//! Commands: `set_paused`, `get_state` (`state_update` with the model's
//! counters, the session context, the publishers and the publisher's ingest
//! counters, also broadcast every second while the session streams).
//! `/start_session` takes `{"context": {"scene", "persona"}, "max_seconds"}`;
//! the length rule is the causal one (design §5.2).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use fastvideo_engine_service::{DuplexCommand, DuplexControl, DuplexReply, DuplexSession, DuplexState, PacedStream};
use fastvideo_media::decode::DecoderPool;
use fastvideo_media::ring::InputBuffers;
use fastvideo_protocol::{ApiError, InputCaps, InputVideoCodec};
use fastvideo_webrtc::host::ReceiveOptions;
use fastvideo_webrtc::ingest::{IngestConfig, IngestStats};
use fastvideo_webrtc::writer::{TrackKind, VideoCodec};
use serde_json::{json, Map, Value};

use crate::driver::{Driver, Outbox, Outcome};
use crate::schema::InputSchema;
use crate::wire::ServerMsg;

/// The camera track a duplex model reads.
pub const INPUT_VIDEO_TRACK: &str = "input_video";
/// The microphone track a duplex model reads.
pub const INPUT_AUDIO_TRACK: &str = "input_audio";

/// `publish_refused` for a slot another connection holds (RT's wording).
pub const ALREADY_PUBLISHED: &str = "track already published";

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `capabilities.tracks` entries of a model's inputs (client perspective:
/// `sendonly`).
pub fn client_input_tracks(caps: &InputCaps) -> Vec<Value> {
    let mut v = Vec::new();
    if caps.video.is_some() {
        v.push(json!({"name": INPUT_VIDEO_TRACK, "kind": "video", "direction": "sendonly"}));
    }
    if caps.audio.is_some() {
        v.push(json!({"name": INPUT_AUDIO_TRACK, "kind": "audio", "direction": "sendonly"}));
    }
    v
}

/// The schema's view of a model's inputs (`in` tracks, the caps).
pub fn input_schema(caps: &InputCaps) -> InputSchema {
    let mut tracks = Vec::new();
    if caps.video.is_some() {
        tracks.push(json!({"name": INPUT_VIDEO_TRACK, "kind": "video", "direction": "in"}));
    }
    if caps.audio.is_some() {
        tracks.push(json!({"name": INPUT_AUDIO_TRACK, "kind": "audio", "direction": "in"}));
    }
    InputSchema { tracks, caps: serde_json::to_value(caps).unwrap_or(Value::Null) }
}

/// A session's input tracks, their publishers and the ingest counters.
#[derive(Debug)]
pub struct Inputs {
    pub caps: InputCaps,
    pub video: Option<String>,
    pub audio: Option<String>,
    pub buffers: Arc<InputBuffers>,
    /// Input stops this long after its first frame (the session limit).
    pub max_seconds: Option<u32>,
    publishers: Mutex<HashMap<String, u32>>,
    stats: Mutex<Option<IngestStats>>,
    /// Decoders started at session start, one per accepted codec, so the
    /// first client frame does not wait for ffmpeg.
    decoders: DecoderPool,
}

impl Inputs {
    pub fn new(caps: &InputCaps, buffers: Arc<InputBuffers>, max_seconds: Option<u32>) -> Self {
        Self {
            caps: caps.clone(),
            video: caps.video.as_ref().map(|_| INPUT_VIDEO_TRACK.to_owned()),
            audio: caps.audio.as_ref().map(|_| INPUT_AUDIO_TRACK.to_owned()),
            buffers,
            max_seconds,
            publishers: Mutex::new(HashMap::new()),
            stats: Mutex::new(None),
            decoders: DecoderPool::per_session(),
        }
    }

    /// Pre-starts the session's decoders (call once, at session start).
    pub fn prewarm(&self) {
        for c in self.base_config().decoder_configs() {
            self.decoders.prewarm(c);
        }
    }

    fn base_config(&self) -> IngestConfig {
        IngestConfig { max_seconds: self.max_seconds, ..IngestConfig::new(self.caps.video.clone(), self.caps.audio.clone()) }
    }

    /// The kind of an input track name.
    pub fn kind_of(&self, name: &str) -> Option<TrackKind> {
        if self.video.as_deref() == Some(name) {
            Some(TrackKind::Video)
        } else if self.audio.as_deref() == Some(name) {
            Some(TrackKind::Audio)
        } else {
            None
        }
    }

    /// The track name of a kind.
    pub fn name_of(&self, kind: TrackKind) -> Option<&str> {
        match kind {
            TrackKind::Video => self.video.as_deref(),
            TrackKind::Audio => self.audio.as_deref(),
        }
    }

    /// `PublishTrack`: first come, first served. Re-publishing a slot the
    /// connection holds succeeds.
    pub fn claim(&self, name: &str, conn: u32) -> Result<TrackKind, String> {
        let Some(kind) = self.kind_of(name) else {
            return Err(format!("the model declares no input track `{name}`"));
        };
        let mut p = lock(&self.publishers);
        match p.get(name) {
            Some(c) if *c != conn => Err(ALREADY_PUBLISHED.into()),
            _ => {
                p.insert(name.to_owned(), conn);
                Ok(kind)
            }
        }
    }

    /// `UnpublishTrack`: releases the slot if `conn` holds it.
    pub fn release(&self, name: &str, conn: u32) -> Option<TrackKind> {
        let mut p = lock(&self.publishers);
        if p.get(name) == Some(&conn) {
            p.remove(name);
            return self.kind_of(name);
        }
        None
    }

    /// Every slot `conn` holds (the connection closed).
    pub fn release_all(&self, conn: u32) -> Vec<String> {
        let mut p = lock(&self.publishers);
        let names: Vec<String> = p.iter().filter(|(_, c)| **c == conn).map(|(n, _)| n.clone()).collect();
        for n in &names {
            p.remove(n);
        }
        names
    }

    pub fn publisher(&self, name: &str) -> Option<u32> {
        lock(&self.publishers).get(name).copied()
    }

    pub fn publishers(&self) -> HashMap<String, u32> {
        lock(&self.publishers).clone()
    }

    /// The publisher's latest ingest counters (the gateway reports them).
    pub fn set_stats(&self, s: IngestStats) {
        *lock(&self.stats) = Some(s);
    }

    pub fn stats(&self) -> Option<IngestStats> {
        lock(&self.stats).clone()
    }

    /// `capabilities.tracks` entries (client perspective: `sendonly`).
    pub fn client_tracks(&self) -> Vec<Value> {
        client_input_tracks(&self.caps)
    }

    /// `track_map` entries (model perspective: `in`).
    pub fn track_map(&self, m: &mut Map<String, Value>) {
        if let (Some(n), Some(c)) = (&self.video, &self.caps.video) {
            m.insert(n.clone(), json!({"kind": "video", "direction": "in", "rate": f64::from(c.max_fps)}));
        }
        if let (Some(n), Some(a)) = (&self.audio, &self.caps.audio) {
            m.insert(n.clone(), json!({"kind": "audio", "direction": "in", "rate": f64::from(a.rate)}));
        }
    }

    /// The schema's view of the inputs.
    pub fn schema(&self) -> InputSchema {
        input_schema(&self.caps)
    }

    /// What every connection's answer receives.
    pub fn receive_options(&self) -> ReceiveOptions {
        let video_codecs = self
            .caps
            .video
            .iter()
            .flat_map(|v| v.codecs.iter())
            .map(|c| match c {
                InputVideoCodec::Vp8 => VideoCodec::Vp8,
                InputVideoCodec::H264 => VideoCodec::H264,
            })
            .collect();
        ReceiveOptions {
            video_codecs,
            audio: self.caps.audio.is_some(),
            max_bitrate_kbps: self.caps.max_bitrate_kbps,
            ..ReceiveOptions::default()
        }
    }

    /// One publisher's ingest settings (the session's warm decoders).
    pub fn ingest_config(&self) -> IngestConfig {
        IngestConfig { pool: Some(self.decoders.clone()), ..self.base_config() }
    }
}

/// The duplex-mode [`Driver`].
pub struct DuplexDriver {
    control: DuplexControl,
    out: Outbox,
    inputs: Arc<Inputs>,
    user_paused: Arc<AtomicBool>,
    peers: Arc<AtomicUsize>,
}

fn state_value(s: &DuplexState, inputs: &Inputs) -> Value {
    let mut v = serde_json::to_value(s).unwrap_or(Value::Null);
    v["publishers"] = serde_json::to_value(inputs.publishers()).unwrap_or(Value::Null);
    v["ingest"] = serde_json::to_value(inputs.stats()).unwrap_or(Value::Null);
    v
}

impl DuplexDriver {
    /// Starts the model worker; the model stays paused until a peer is
    /// connected.
    pub fn start(session: DuplexSession, out: Outbox, inputs: Arc<Inputs>) -> Result<(Self, PacedStream), ApiError> {
        let (control, paced) = session.start()?;
        control.set_paused(true);
        // state_update every second while frames go out.
        let (c, o, i) = (control.clone(), out.clone(), inputs.clone());
        tokio::spawn(async move {
            let mut last = u64::MAX;
            while !c.is_closed() {
                tokio::time::sleep(Duration::from_secs(1)).await;
                let s = c.state();
                if s.frames_out != last && s.frames_out > 0 {
                    last = s.frames_out;
                    o.broadcast(ServerMsg::broadcast("state_update", state_value(&s, &i)));
                }
            }
        });
        let d = Self {
            control,
            out,
            inputs,
            user_paused: Arc::new(AtomicBool::new(false)),
            peers: Arc::new(AtomicUsize::new(0)),
        };
        Ok((d, paced))
    }

    pub fn control(&self) -> &DuplexControl {
        &self.control
    }

    fn apply_pause(&self) {
        let paused = self.user_paused.load(Ordering::Relaxed) || self.peers.load(Ordering::Relaxed) == 0;
        self.control.set_paused(paused);
    }

    fn state(&self) -> Value {
        let mut s = self.control.state();
        s.paused = self.user_paused.load(Ordering::Relaxed);
        state_value(&s, &self.inputs)
    }
}

#[async_trait]
impl Driver for DuplexDriver {
    async fn command(&self, _conn: u32, name: &str, args: Map<String, Value>) -> Outcome {
        let cmd = match name {
            "set_paused" => match args.get("paused").and_then(Value::as_bool) {
                Some(p) => DuplexCommand::SetPaused { paused: p },
                None => return Outcome::Error { code: "invalid_command".into(), message: "`paused` must be a boolean".into() },
            },
            "get_state" => DuplexCommand::GetState,
            _ => return Outcome::Error { code: "invalid_command".into(), message: format!("unknown command `{name}`") },
        };
        if let DuplexCommand::SetPaused { paused } = cmd {
            self.user_paused.store(paused, Ordering::Relaxed);
            self.apply_pause();
            self.out.broadcast(ServerMsg::broadcast("state_update", self.state()));
            return Outcome::Ack;
        }
        match self.control.apply(cmd) {
            DuplexReply::StateUpdate(_) => Outcome::Reply("state_update".into(), self.state()),
            DuplexReply::CommandError { command, reason } => {
                self.out.broadcast(ServerMsg::broadcast("command_error", json!({"command": command, "reason": reason})));
                Outcome::Ack
            }
        }
    }

    fn greet(&self, conn: u32) {
        self.out.to(conn, ServerMsg::broadcast("state_update", self.state()));
    }

    fn peers_changed(&self, connected: usize) {
        self.peers.store(connected, Ordering::Relaxed);
        self.apply_pause();
    }

    async fn close(&self) {
        self.control.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::{AudioInputCaps, VideoInputCaps};

    fn inputs() -> Inputs {
        let caps = InputCaps {
            video: Some(VideoInputCaps {
                width: 64,
                height: 36,
                max_width: 1280,
                max_height: 720,
                max_fps: 30,
                codecs: vec![InputVideoCodec::Vp8, InputVideoCodec::H264],
            }),
            audio: Some(AudioInputCaps { rate: 48_000, channels: 1 }),
            max_bitrate_kbps: 2000,
            buffer_ms: 500,
        };
        let b = InputBuffers::for_caps(&caps);
        Inputs::new(&caps, b, Some(60))
    }

    #[test]
    fn publisher_slots_are_first_come_first_served() {
        let i = inputs();
        assert_eq!(i.claim(INPUT_VIDEO_TRACK, 1001), Ok(TrackKind::Video));
        assert_eq!(i.claim(INPUT_VIDEO_TRACK, 1001), Ok(TrackKind::Video), "re-publish by the holder");
        assert_eq!(i.claim(INPUT_VIDEO_TRACK, 1002), Err(ALREADY_PUBLISHED.into()));
        assert!(i.claim("cam", 1002).unwrap_err().contains("no input track `cam`"));
        assert_eq!(i.claim(INPUT_AUDIO_TRACK, 1002), Ok(TrackKind::Audio));
        assert_eq!(i.release(INPUT_VIDEO_TRACK, 1002), None, "only the holder releases");
        assert_eq!(i.release(INPUT_VIDEO_TRACK, 1001), Some(TrackKind::Video));
        assert_eq!(i.claim(INPUT_VIDEO_TRACK, 1002), Ok(TrackKind::Video));
        let mut gone = i.release_all(1002);
        gone.sort();
        assert_eq!(gone, vec![INPUT_AUDIO_TRACK.to_owned(), INPUT_VIDEO_TRACK.to_owned()]);
        assert!(i.publishers().is_empty());
    }

    #[test]
    fn tracks_in_all_three_places() {
        let i = inputs();
        assert_eq!(
            i.client_tracks(),
            vec![
                json!({"name": "input_video", "kind": "video", "direction": "sendonly"}),
                json!({"name": "input_audio", "kind": "audio", "direction": "sendonly"})
            ]
        );
        let mut m = Map::new();
        i.track_map(&mut m);
        assert_eq!(m["input_video"]["direction"], "in");
        assert_eq!(m["input_audio"]["rate"], 48000.0);
        let s = i.schema();
        assert_eq!(s.tracks[0]["direction"], "in");
        assert_eq!(s.caps["max_bitrate_kbps"], 2000);
        let r = i.receive_options();
        assert_eq!(r.video_codecs, vec![VideoCodec::Vp8, VideoCodec::H264]);
        assert!(r.audio);
        assert_eq!(r.max_bitrate_kbps, 2000);
        assert_eq!(i.ingest_config().max_seconds, Some(60));
    }
}
