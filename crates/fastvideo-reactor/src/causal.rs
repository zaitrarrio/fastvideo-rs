//! The causal (SF-Wan) driver: Waypoint-style `InputState` setters
//! (`set_prompt`, `set_paused`, `set_seed`) plus `reset` (design §5.7).
//!
//! One pump task owns the [`CausalEngine`]: it applies control changes and
//! pulls blocks, pushing their frames into the session pacer (which waits
//! when full: generation ahead of playout, nothing dropped upstream, design
//! §5.4). Setters answer with a bodyless ack (as RT's auto-generated
//! setters) and broadcast `state_update{prompt, paused, seed, block_index,
//! unique_fps}`; a `state_update` also follows every block. Generation is
//! paused while no peer is connected.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, oneshot};

use crate::driver::{Driver, Outbox, Outcome};
use crate::engine::CausalEngine;
use crate::media::MediaPipeline;
use crate::wire::ServerMsg;

#[derive(Clone, Debug, Default)]
struct View {
    prompt: String,
    paused: bool,
    seed: u64,
    block_index: u64,
    unique_fps: f64,
}

impl View {
    fn json(&self) -> Value {
        json!({
            "prompt": self.prompt,
            "paused": self.paused,
            "seed": self.seed,
            "block_index": self.block_index,
            "unique_fps": (self.unique_fps * 100.0).round() / 100.0,
        })
    }
}

enum Ctl {
    Prompt(String),
    Paused(bool),
    Seed(u64),
    Reset,
    Peers(usize),
    Close(oneshot::Sender<()>),
}

/// The causal-mode [`Driver`].
pub struct CausalDriver {
    view: Arc<Mutex<View>>,
    tx: mpsc::UnboundedSender<Ctl>,
    out: Outbox,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl CausalDriver {
    pub fn start(
        engine: Box<dyn CausalEngine>,
        media: Arc<MediaPipeline>,
        out: Outbox,
        seed: u64,
    ) -> Self {
        let view = Arc::new(Mutex::new(View { seed, ..View::default() }));
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(pump(engine, media, out.clone(), view.clone(), rx, seed));
        Self { view, tx, out }
    }

    fn state(&self) -> ServerMsg {
        ServerMsg::broadcast("state_update", lock(&self.view).json())
    }
}

async fn pump(
    mut engine: Box<dyn CausalEngine>,
    media: Arc<MediaPipeline>,
    out: Outbox,
    view: Arc<Mutex<View>>,
    mut rx: mpsc::UnboundedReceiver<Ctl>,
    seed: u64,
) {
    engine.set_seed(seed);
    // No prompt yet and no audience: the engine parks.
    let (mut user_paused, mut peers) = (false, 0usize);
    engine.set_paused(true);
    let mut fps_ema: Option<f64> = None;
    let mut last = Instant::now();
    loop {
        tokio::select! {
            c = rx.recv() => {
                let Some(c) = c else { break };
                match c {
                    Ctl::Prompt(p) => engine.set_prompt(&p),
                    Ctl::Paused(p) => {
                        user_paused = p;
                        engine.set_paused(user_paused || peers == 0);
                    }
                    Ctl::Seed(s) => engine.set_seed(s),
                    Ctl::Reset => {
                        engine.reset();
                        media.flush();
                    }
                    Ctl::Peers(n) => {
                        peers = n;
                        engine.set_paused(user_paused || peers == 0);
                    }
                    Ctl::Close(done) => {
                        engine.close();
                        let _ = done.send(());
                        return;
                    }
                }
            }
            b = engine.next_block() => {
                match b {
                    None => break,
                    Some(Err(e)) => {
                        tracing::warn!(error = %e.message, "causal block failed");
                        out.broadcast(ServerMsg::broadcast(
                            "command_error",
                            json!({"command": "generate", "reason": e.message}),
                        ));
                    }
                    Some(Ok(b)) => {
                        let n = b.frames.len() as f64;
                        let now = Instant::now();
                        let dt = now.duration_since(last).as_secs_f64().max(1e-3);
                        last = now;
                        let inst = n / dt;
                        let ema = fps_ema.map_or(inst, |e| 0.7 * e + 0.3 * inst);
                        fps_ema = Some(ema);
                        {
                            let mut v = lock(&view);
                            v.block_index = b.index;
                            v.unique_fps = ema;
                        }
                        media.push(b.frames, b.audio).await;
                        out.broadcast(ServerMsg::broadcast("state_update", lock(&view).json()));
                    }
                }
            }
        }
    }
    engine.close();
}

#[async_trait]
impl Driver for CausalDriver {
    async fn command(&self, _conn: u32, name: &str, a: Map<String, Value>) -> Outcome {
        let ctl = match name {
            "set_prompt" => {
                let p = a.get("prompt").and_then(Value::as_str).unwrap_or_default().to_owned();
                lock(&self.view).prompt = p.clone();
                Ctl::Prompt(p)
            }
            "set_paused" => {
                let p = a.get("paused").and_then(Value::as_bool).unwrap_or(false);
                lock(&self.view).paused = p;
                Ctl::Paused(p)
            }
            "set_seed" => {
                let s = a.get("seed").and_then(Value::as_u64).unwrap_or_default();
                lock(&self.view).seed = s;
                Ctl::Seed(s)
            }
            "reset" => {
                lock(&self.view).block_index = 0;
                Ctl::Reset
            }
            other => {
                return Outcome::Error { code: "invalid_command".into(), message: format!("unknown command `{other}`") }
            }
        };
        if self.tx.send(ctl).is_err() {
            return Outcome::Error { code: "internal_error".into(), message: "session closed".into() };
        }
        self.out.broadcast(self.state());
        Outcome::Ack
    }

    fn greet(&self, conn: u32) {
        self.out.to(conn, self.state());
    }

    fn peers_changed(&self, connected: usize) {
        let _ = self.tx.send(Ctl::Peers(connected));
    }

    async fn close(&self) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Ctl::Close(tx)).is_ok() {
            let _ = rx.await;
        }
    }
}
