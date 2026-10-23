//! Edge sessions held on this worker, and the local sessions they stand for
//! (docs/serve/edge-control-plane.md, "Sessions end on the worker").
//!
//! The edge admits a session (director, Reactor, stream) through a family
//! object, which offers it to a worker: the worker's arbiter reserves the GPU
//! and the edge then forwards the client's signalling. The session's life
//! is the worker's business from there: a director session ends over its
//! data channel (`stop`), when its peer goes, on a missed heartbeat, a
//! failure or its length limit; a Reactor session on `stop_session` or its
//! orphan timer. Whatever ends it, the worker must give the lease back: it
//! frees its arbiter slot and sends `session_end` to the family object, so a
//! queued batch job is pushed at once rather than when the lease's TTL runs
//! out (30 min).
//!
//! This registry ties the two together:
//!
//! - the family links record every admitted session (`admitted`) with how to
//!   end it from the worker side (an [`Ender`]: release the slot, send
//!   `session_end`), and forget it when the dispatcher ends it (`forget`);
//! - a local session binds to its edge session when it starts (`bind`): the
//!   one the edge names in [`EDGE_SESSION_HEADER`] (edges from this change
//!   on), else the oldest unbound one of its kind (edges that do not send
//!   the header: the edge forwards the signalling right after the offer is
//!   acked, and a worker runs one session of a kind at a time);
//! - when the local session ends, `ended` ends its edge session.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub use fastvideo_dispatch_proto::front::EDGE_SESSION_HEADER;

/// Ends an edge session from the worker: `(edge session id, lease, reason)`.
pub type Ender = Arc<dyn Fn(&str, u64, &str) + Send + Sync>;

struct Entry {
    kind: String,
    lease: u64,
    /// Admission order (the fallback binds the oldest).
    seq: u64,
    /// The local session it stands for, once bound.
    local: Option<String>,
    ender: Ender,
}

#[derive(Default)]
struct Inner {
    seq: u64,
    by_edge: BTreeMap<String, Entry>,
}

/// The edge sessions of this process (shared by its family links and the
/// session services).
#[derive(Default)]
pub struct EdgeSessions {
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for EdgeSessions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let g = self.lock();
        f.debug_map()
            .entries(
                g.by_edge
                    .iter()
                    .map(|(k, e)| (k, (&e.kind, e.lease, &e.local))),
            )
            .finish()
    }
}

impl EdgeSessions {
    pub fn new() -> Arc<Self> {
        Arc::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The family object's offer `edge_id` (`kind`, `lease`) was taken here.
    pub fn admitted(&self, edge_id: &str, lease: u64, kind: &str, ender: Ender) {
        let mut g = self.lock();
        g.seq += 1;
        let seq = g.seq;
        g.by_edge.insert(
            edge_id.to_owned(),
            Entry {
                kind: kind.to_owned(),
                lease,
                seq,
                local: None,
                ender,
            },
        );
    }

    /// The dispatcher ended `edge_id` (revoke, release, re-hello): nothing
    /// to send back.
    pub fn forget(&self, edge_id: &str) {
        self.lock().by_edge.remove(edge_id);
    }

    /// A local session `local` of `kind` started: binds it to the edge
    /// session `hint` names (when it is one of ours, of that kind and
    /// unbound), else to the oldest unbound one of its kind. The edge
    /// session id, or `None` (not an edge session: a direct client).
    pub fn bind(&self, kind: &str, local: &str, hint: Option<&str>) -> Option<String> {
        let mut g = self.lock();
        let free = |e: &Entry| e.kind == kind && e.local.is_none();
        let id = match hint.filter(|h| g.by_edge.get(*h).is_some_and(free)) {
            Some(h) => h.to_owned(),
            None => g
                .by_edge
                .iter()
                .filter(|(_, e)| free(e))
                .min_by_key(|(_, e)| e.seq)
                .map(|(k, _)| k.clone())?,
        };
        if let Some(e) = g.by_edge.get_mut(&id) {
            e.local = Some(local.to_owned());
        }
        tracing::info!(session = %local, edge_session = %id, kind, hinted = hint.is_some(), "worker: local session bound to its edge session");
        Some(id)
    }

    /// The local session `local` ended (`reason`): ends its edge session
    /// (frees the slot, tells the family object). The edge session id.
    pub fn ended(&self, local: &str, reason: &str) -> Option<String> {
        let (id, e) = {
            let mut g = self.lock();
            let id = g
                .by_edge
                .iter()
                .find(|(_, e)| e.local.as_deref() == Some(local))
                .map(|(k, _)| k.clone())?;
            let e = g.by_edge.remove(&id)?;
            (id, e)
        };
        (e.ender)(&id, e.lease, reason);
        Some(id)
    }

    /// Edge sessions held (bound or not).
    pub fn len(&self) -> usize {
        self.lock().by_edge.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Log = Arc<Mutex<Vec<(String, u64, String)>>>;

    fn recorder() -> (Log, Ender) {
        let log: Log = Arc::default();
        let l2 = log.clone();
        (
            log,
            Arc::new(move |id: &str, lease: u64, why: &str| {
                l2.lock().unwrap().push((id.into(), lease, why.into()))
            }),
        )
    }

    #[test]
    fn a_local_end_ends_its_edge_session() {
        let s = EdgeSessions::new();
        let (log, ender) = recorder();
        s.admitted("ses-a", 3, "director", ender.clone());
        s.admitted("ses-b", 4, "director", ender.clone());
        s.admitted("ses-r", 5, "reactor", ender);
        // No hint: the oldest unbound one of the kind.
        assert_eq!(s.bind("director", "d1", None).as_deref(), Some("ses-a"));
        // A hint names its own.
        assert_eq!(
            s.bind("director", "d2", Some("ses-b")).as_deref(),
            Some("ses-b")
        );
        // Nothing left of that kind: a direct client.
        assert_eq!(s.bind("director", "d3", None), None);
        // A hint of the wrong kind or bound already falls back (nothing free here).
        assert_eq!(s.bind("director", "d3", Some("ses-r")), None);
        assert_eq!(s.ended("d2", "stopped").as_deref(), Some("ses-b"));
        assert_eq!(s.ended("d2", "stopped"), None, "once");
        assert_eq!(s.ended("d9", "stopped"), None, "not an edge session");
        assert_eq!(
            *log.lock().unwrap(),
            vec![("ses-b".to_owned(), 4, "stopped".to_owned())]
        );
        // The dispatcher ended it first: nothing to send back.
        s.forget("ses-a");
        assert_eq!(s.ended("d1", "stopped"), None);
        assert_eq!(s.len(), 1);
    }
}
