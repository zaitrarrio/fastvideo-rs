//! Clip mode (H3, LTX, FastWan): the fast-h3 queue-and-playout contract
//! (reactor §4bis, design §5.5) over WP-12's `ClipPlayer`.
//!
//! The player owns the semantics (queues, reservation, discard on pop,
//! autoplay, `valid_commands`, refusals as broadcast `command_error`,
//! 3-frame lockstep slices). This driver maps validated wire commands to
//! [`ClipCommand`]s, forwards the player's fast-h3 events to every client,
//! gates builds on an audience, and feeds the player's media to the clip
//! pacer with idle policy **Black** (fast-h3's `output.flush()` hold after a
//! clip ends with nothing armed, and on a cut).

use async_trait::async_trait;
use fastvideo_engine_service::{
    spawn_clip_pacer, ClipCommand, ClipPacerConfig, ClipPlayer, ClipPlayerConfig, ClipSession, IdlePolicy,
    PacedStream, TickStart,
};
use fastvideo_protocol::{canvas_for_aspect, ApiError, CanvasCaps};
use serde_json::{Map, Value};

use crate::driver::{Driver, Outbox, Outcome};
use crate::wire::ServerMsg;

/// `(width, height)` for an aspect label such as `16:9`.
pub fn aspect_canvas(caps: &CanvasCaps, aspect: &str, short_edge: u32) -> Option<(u32, u32)> {
    let (w, h) = aspect.split_once(':')?;
    let (w, h): (f64, f64) = (w.parse().ok()?, h.parse().ok()?);
    (w > 0.0 && h > 0.0).then(|| canvas_for_aspect(caps, w / h, short_edge))
}

/// The clip-mode [`Driver`].
pub struct ClipDriver {
    player: ClipPlayer,
    out: Outbox,
}

impl ClipDriver {
    /// Starts the player (no audience yet) and the clip pacer; returns the
    /// driver and the paced stream the media pipeline consumes.
    pub fn start(session: ClipSession, aspect: &str, out: Outbox) -> Result<(Self, PacedStream), ApiError> {
        let spec = session.spec().clone();
        let (player, outputs) = session.into_player(ClipPlayerConfig {
            aspect: Some(aspect.to_owned()),
            audience: false,
            ..ClipPlayerConfig::default()
        })?;
        let paced = spawn_clip_pacer(
            outputs.media,
            ClipPacerConfig {
                idle: IdlePolicy::Black,
                // Ticks from the start: the audio clock runs (silence) and
                // the black start-of-connection frame has a tick to ride.
                start: TickStart::Immediately,
                ..ClipPacerConfig::for_spec(&spec)
            },
        )?;
        let mut events = outputs.events;
        let bout = out.clone();
        tokio::spawn(async move {
            while let Some(ev) = events.recv().await {
                if ev.is_fasth3() {
                    bout.broadcast(ServerMsg::broadcast(ev.type_name(), ev.data()));
                }
            }
        });
        Ok((Self { player, out }, paced))
    }

    pub fn player(&self) -> &ClipPlayer {
        &self.player
    }
}

fn s(a: &Map<String, Value>, k: &str) -> String {
    a.get(k).and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// A validated wire command (see [`crate::commands`]) → [`ClipCommand`].
pub fn to_command(name: &str, a: &Map<String, Value>) -> Option<ClipCommand> {
    let u = |k: &str| a.get(k).and_then(Value::as_u64);
    let f = |k: &str| a.get(k).and_then(Value::as_f64);
    Some(match name {
        "enqueue" => ClipCommand::Enqueue {
            prompt: s(a, "prompt"),
            metadata: s(a, "metadata"),
            seed: u("seed"),
            seconds: f("seconds"),
            position: u("position").map(|p| p.min(u64::from(u32::MAX)) as u32),
        },
        "play" => ClipCommand::Play { clip_id: s(a, "clip_id") },
        "pop" => ClipCommand::Pop { clip_id: s(a, "clip_id") },
        "move" => ClipCommand::Move {
            clip_id: s(a, "clip_id"),
            position: u("position").unwrap_or(0).min(u64::from(u32::MAX)) as u32,
        },
        "stop" => ClipCommand::Stop,
        "reset" => ClipCommand::Reset,
        "set_clip_seconds" => ClipCommand::SetClipSeconds(f("seconds")?),
        "set_seed" => ClipCommand::SetSeed(u("seed")?),
        "set_autoplay" => ClipCommand::SetAutoplay(a.get("enabled").and_then(Value::as_bool)?),
        "set_canvas" => ClipCommand::SetCanvas(s(a, "aspect")),
        "get_queue" => ClipCommand::GetQueue,
        "get_state" => ClipCommand::GetState,
        _ => return None,
    })
}

#[async_trait]
impl Driver for ClipDriver {
    async fn command(&self, _conn: u32, name: &str, args: Map<String, Value>) -> Outcome {
        let Some(cmd) = to_command(name, &args) else {
            return Outcome::Error { code: "invalid_command".into(), message: format!("unknown command `{name}`") };
        };
        match self.player.command(cmd).await {
            Ok(Some(ev)) => Outcome::Reply(ev.type_name().to_owned(), ev.data()),
            Ok(None) => Outcome::Ack,
            Err(e) => Outcome::Error { code: "internal_error".into(), message: e.message },
        }
    }

    fn greet(&self, conn: u32) {
        let player = self.player.clone();
        let out = self.out.clone();
        tokio::spawn(async move {
            for cmd in [ClipCommand::GetState, ClipCommand::GetQueue] {
                if let Ok(Some(ev)) = player.command(cmd).await {
                    out.to(conn, ServerMsg::broadcast(ev.type_name(), ev.data()));
                }
            }
        });
    }

    fn peers_changed(&self, connected: usize) {
        let player = self.player.clone();
        tokio::spawn(async move {
            let _ = player.set_audience(connected > 0).await;
        });
    }

    async fn close(&self) {
        self.player.close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn m(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn wire_commands_map_to_the_player() {
        assert_eq!(
            to_command("enqueue", &m(json!({"prompt": "p", "metadata": "", "seed": null, "seconds": 5.5, "position": 2}))),
            Some(ClipCommand::Enqueue { prompt: "p".into(), metadata: String::new(), seed: None, seconds: Some(5.5), position: Some(2) })
        );
        assert_eq!(to_command("play", &m(json!({"clip_id": ""}))), Some(ClipCommand::Play { clip_id: String::new() }));
        assert_eq!(to_command("set_autoplay", &m(json!({"enabled": true}))), Some(ClipCommand::SetAutoplay(true)));
        assert_eq!(to_command("set_canvas", &m(json!({"aspect": "1:1"}))), Some(ClipCommand::SetCanvas("1:1".into())));
        assert_eq!(to_command("get_state", &Map::new()), Some(ClipCommand::GetState));
        assert_eq!(to_command("nope", &Map::new()), None);
    }

    #[test]
    fn aspects_resolve_on_the_model_grid() {
        let c = CanvasCaps::h3();
        assert_eq!(aspect_canvas(&c, "16:9", 768), Some((1344, 768)));
        let (w, h) = aspect_canvas(&c, "9:16", 768).unwrap();
        assert!(h > w);
        assert_eq!(aspect_canvas(&c, "x", 768), None);
    }
}
