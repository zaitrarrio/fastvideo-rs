//! What a session's command handler looks like to the gateway, and the
//! per-session outbox the handlers broadcast through.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use crate::wire::ServerMsg;

/// Per-connection outbound queue depth (design §5.10: 64 queued messages
/// per peer; a peer that falls further behind is closed).
pub const OUTBOX_DEPTH: usize = 64;

/// The result of one validated command.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Correlated `ModelMessage{type, data}`.
    Reply(String, Value),
    /// Bodyless ack (v1 only; nothing on v0).
    Ack,
    /// A failed command (`Error{code, message}`, v1 only).
    Error { code: String, message: String },
}

/// A session's model side: the fast-h3 clip driver or the causal driver.
#[async_trait]
pub trait Driver: Send + Sync {
    /// Runs one validated command (`args` has every declared field).
    async fn command(&self, conn: u32, name: &str, args: Map<String, Value>) -> Outcome;
    /// A client connected: greet it (fast-h3 sends `state_update` and
    /// `queue_update` to the joining client).
    fn greet(&self, conn: u32);
    /// How many peers are connected now (generation runs only with an
    /// audience).
    fn peers_changed(&self, connected: usize);
    /// Stops everything; the engine session is released.
    async fn close(&self);
}

/// Every connection's outbound queue.
#[derive(Clone, Default)]
pub struct Outbox {
    conns: Arc<Mutex<HashMap<u32, mpsc::Sender<ServerMsg>>>>,
}

impl std::fmt::Debug for Outbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Outbox").field("conns", &self.len()).finish()
    }
}

impl Outbox {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u32, mpsc::Sender<ServerMsg>>> {
        self.conns.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Registers a connection; its gateway drains the receiver.
    pub fn register(&self, conn: u32) -> mpsc::Receiver<ServerMsg> {
        let (tx, rx) = mpsc::channel(OUTBOX_DEPTH);
        self.lock().insert(conn, tx);
        rx
    }

    pub fn unregister(&self, conn: u32) {
        self.lock().remove(&conn);
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sends to one connection. A full queue drops the connection's sender,
    /// which makes its gateway close the peer.
    pub fn to(&self, conn: u32, msg: ServerMsg) {
        let mut m = self.lock();
        if let Some(tx) = m.get(&conn) {
            if tx.try_send(msg).is_err() {
                tracing::warn!(conn, "outbound queue overflow: closing the connection");
                m.remove(&conn);
            }
        }
    }

    /// Sends to every connection.
    pub fn broadcast(&self, msg: ServerMsg) {
        let mut m = self.lock();
        let mut dead = Vec::new();
        for (id, tx) in m.iter() {
            if tx.try_send(msg.clone()).is_err() {
                dead.push(*id);
            }
        }
        for id in dead {
            tracing::warn!(conn = id, "outbound queue overflow: closing the connection");
            m.remove(&id);
        }
    }
}
