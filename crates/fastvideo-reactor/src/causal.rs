//! Causal mode (SF-Wan): Waypoint-style `InputState` setters `set_prompt`,
//! `set_paused`, `set_seed`, plus `reset` and `get_state` (design §5.7),
//! over WP-15's `CausalControl` and the adaptive causal pacer.
//!
//! Setters answer with a bodyless ack (as RT's auto-generated `set_<field>`
//! setters) and broadcast `state_update{prompt, paused, seed, block_index,
//! unique_fps}`; `get_state` replies with it. Refusals are broadcast
//! `command_error{command, reason}` plus the ack. Generation pauses while no
//! peer is connected (design §5.2 Orphaned), and a `state_update` goes out
//! whenever a new block has been played (at most once a second).
//!
//! A session is length-limited (design §5.2): `/start_session`
//! `max_seconds`, else `[streams] causal_default_max_s` (120 s of video),
//! at most `causal_hard_max_s` (300). `reset` restarts the clock, up to the
//! ceiling in all. At the limit the runtime closes the session with
//! `session_ended{reason}` ([`crate::session::session_limit_reason`]).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use fastvideo_engine_service::{
    spawn_causal_pacer, CausalCommand, CausalControl, CausalPacerConfig, CausalReply, CausalSession, CausalState,
    PacedStream,
};
use fastvideo_protocol::{ApiError, CausalLimits};
use serde_json::{json, Map, Value};

use crate::driver::{Driver, Outbox, Outcome};
use crate::wire::ServerMsg;

/// The causal-mode [`Driver`].
pub struct CausalDriver {
    control: CausalControl,
    out: Outbox,
    user_paused: Arc<AtomicBool>,
    peers: Arc<AtomicUsize>,
}

fn state_msg(s: &CausalState) -> ServerMsg {
    ServerMsg::broadcast("state_update", serde_json::to_value(s).unwrap_or(Value::Null))
}

impl CausalDriver {
    /// Starts the causal pacer over `session` (generation stays parked until
    /// a prompt arrives and a peer is connected).
    /// The pacer ends the stream at the spec's `max_seconds` (restarted by a
    /// `reset`, `limits.hard_max_s` in all).
    pub fn start(
        session: CausalSession,
        seed: u64,
        limits: &CausalLimits,
        out: Outbox,
    ) -> Result<(Self, PacedStream), ApiError> {
        let control = session.control();
        control.set_seed(seed);
        control.set_paused(true);
        let spec = session.spec().clone();
        let paced = spawn_causal_pacer(session, CausalPacerConfig::with_limits(&spec, limits))?;
        // state_update after new blocks.
        let (c, o) = (control.clone(), out.clone());
        tokio::spawn(async move {
            let mut last = u64::MAX;
            while !c.is_closed() {
                tokio::time::sleep(Duration::from_secs(1)).await;
                let s = c.state();
                if s.block_index != last && s.block_index > 0 {
                    last = s.block_index;
                    o.broadcast(state_msg(&s));
                }
            }
        });
        let d = Self {
            control,
            out,
            user_paused: Arc::new(AtomicBool::new(false)),
            peers: Arc::new(AtomicUsize::new(0)),
        };
        Ok((d, paced))
    }

    pub fn control(&self) -> &CausalControl {
        &self.control
    }

    fn apply_pause(&self) {
        let paused = self.user_paused.load(Ordering::Relaxed) || self.peers.load(Ordering::Relaxed) == 0;
        self.control.set_paused(paused);
    }
}

/// A validated wire command (see [`crate::commands`]) → [`CausalCommand`].
pub fn to_command(name: &str, a: &Map<String, Value>) -> Option<CausalCommand> {
    Some(match name {
        "set_prompt" => CausalCommand::SetPrompt { prompt: a.get("prompt")?.as_str()?.to_owned() },
        "set_paused" => CausalCommand::SetPaused { paused: a.get("paused")?.as_bool()? },
        "set_seed" => CausalCommand::SetSeed { seed: a.get("seed")?.as_u64()? },
        "reset" => CausalCommand::Reset,
        "get_state" => CausalCommand::GetState,
        _ => return None,
    })
}

#[async_trait]
impl Driver for CausalDriver {
    async fn command(&self, _conn: u32, name: &str, args: Map<String, Value>) -> Outcome {
        let Some(cmd) = to_command(name, &args) else {
            return Outcome::Error { code: "invalid_command".into(), message: format!("unknown command `{name}`") };
        };
        let get = matches!(cmd, CausalCommand::GetState);
        let pause = match &cmd {
            CausalCommand::SetPaused { paused } => Some(*paused),
            _ => None,
        };
        let reply = self.control.apply(cmd);
        if let (Some(p), CausalReply::StateUpdate(_)) = (pause, &reply) {
            self.user_paused.store(p, Ordering::Relaxed);
            self.apply_pause();
        }
        match reply {
            CausalReply::StateUpdate(mut s) => {
                s.paused = self.user_paused.load(Ordering::Relaxed);
                if get {
                    return Outcome::Reply("state_update".into(), serde_json::to_value(&s).unwrap_or(Value::Null));
                }
                self.out.broadcast(state_msg(&s));
                Outcome::Ack
            }
            CausalReply::CommandError { command, reason } => {
                self.out.broadcast(ServerMsg::broadcast("command_error", json!({"command": command, "reason": reason})));
                Outcome::Ack
            }
        }
    }

    fn greet(&self, conn: u32) {
        let mut s = self.control.state();
        s.paused = self.user_paused.load(Ordering::Relaxed);
        self.out.to(conn, state_msg(&s));
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

    #[test]
    fn wire_commands_map_to_the_control() {
        let m = |v: Value| v.as_object().unwrap().clone();
        assert_eq!(
            to_command("set_prompt", &m(json!({"prompt": "p"}))),
            Some(CausalCommand::SetPrompt { prompt: "p".into() })
        );
        assert_eq!(to_command("set_paused", &m(json!({"paused": true}))), Some(CausalCommand::SetPaused { paused: true }));
        assert_eq!(to_command("set_seed", &m(json!({"seed": 3}))), Some(CausalCommand::SetSeed { seed: 3 }));
        assert_eq!(to_command("reset", &Map::new()), Some(CausalCommand::Reset));
        assert_eq!(to_command("get_state", &Map::new()), Some(CausalCommand::GetState));
        assert_eq!(to_command("enqueue", &Map::new()), None);
    }
}
