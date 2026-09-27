//! `GET /events`: the Server-Sent Events journal of every state transition
//! (reactor §3.4, `RT:http/events.py`). Each event is
//! `id: <seq>\ndata: {"type":"transition","event":…,"from":…,"to":…,"ts":<ms>,"detail":{…}}`,
//! resumable through `?since=` or `Last-Event-ID`. A director-facing
//! surface; SDK clients do not consume it.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde_json::{json, Value};
use tokio::sync::broadcast;

/// Events kept for `?since=` replay.
pub const JOURNAL_KEEP: usize = 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub seq: u64,
    pub data: Value,
}

pub struct Journal {
    ring: Mutex<(u64, VecDeque<Event>)>,
    tx: broadcast::Sender<Event>,
}

impl std::fmt::Debug for Journal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Journal").finish_non_exhaustive()
    }
}

impl Default for Journal {
    fn default() -> Self {
        Self { ring: Mutex::new((0, VecDeque::new())), tx: broadcast::channel(256).0 }
    }
}

fn now_ms() -> i64 {
    (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64
}

impl Journal {
    /// Records `event` (`from` → `to`; journal-only events self-loop).
    pub fn record(&self, event: &str, from: &str, to: &str, detail: Value) {
        let mut g = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        g.0 += 1;
        let e = Event {
            seq: g.0,
            data: json!({"type": "transition", "event": event, "from": from, "to": to, "ts": now_ms(), "detail": detail}),
        };
        g.1.push_back(e.clone());
        while g.1.len() > JOURNAL_KEEP {
            g.1.pop_front();
        }
        let _ = self.tx.send(e);
    }

    /// Events after `since`, plus a live receiver (subscribed before the
    /// replay is read, so nothing falls in between; duplicates are skipped
    /// by sequence number).
    pub fn subscribe(&self, since: u64) -> (Vec<Event>, broadcast::Receiver<Event>) {
        let g = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let rx = self.tx.subscribe();
        (g.1.iter().filter(|e| e.seq > since).cloned().collect(), rx)
    }

    pub fn last_seq(&self) -> u64 {
        self.ring.lock().unwrap_or_else(|e| e.into_inner()).0
    }
}
