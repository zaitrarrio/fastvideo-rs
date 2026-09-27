//! `Priority` scheduling, queue positions and the exclusive causal lease
//! (design §3.6, §5.2 "Scheduling decision").
//!
//! Pure bookkeeping, no threads: the service holds it under a mutex and the
//! executors pull work from it.
//!
//! Rules:
//!
//! - `Priority::Stream` (clip-session builds) always dispatches before
//!   `Priority::Batch`; FIFO within a priority. Clip builds and batch jobs so
//!   interleave at clip granularity, and streaming wins ties.
//! - An executor holding a **causal lease** (live SF-Wan) runs only that
//!   session's blocks; queued work waits (and keeps reporting its position)
//!   until the lease ends. A lease granted while the executor is busy takes
//!   effect after the running job.
//! - One streaming session (clip or causal) per executor.
//! - `queue_max` bounds queued **batch** jobs; stream builds are bounded by
//!   their session's own reservation logic.
//! - Queue positions are 0-based: the number of queued items dispatched
//!   before this one.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use fastvideo_protocol::{ApiError, JobId, ModelId};
use serde::{Deserialize, Serialize};

use crate::backend::SessionId;

/// Dispatch priority (design §3.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Clip-session builds.
    Stream,
    /// Batch API jobs.
    Batch,
}

impl Priority {
    fn rank(self) -> u8 {
        match self {
            Priority::Stream => 0,
            Priority::Batch => 1,
        }
    }
}

/// One queued job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueItem {
    pub job: JobId,
    pub prio: Priority,
    pub model: ModelId,
    /// Run only on this executor (a clip session's builds).
    pub pin: Option<usize>,
}

/// What an executor should do next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Job(QueueItem),
    /// One turn (one block) of the leased causal session.
    Causal(SessionId),
}

/// Per-executor scheduling state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecState {
    /// Models the backend can serve.
    pub serves: BTreeSet<ModelId>,
    /// Models currently resident.
    pub resident: BTreeSet<ModelId>,
    pub running: Option<JobId>,
    /// Exclusive causal lease.
    pub lease: Option<SessionId>,
    /// The streaming session (clip or causal) occupying this executor.
    pub session: Option<SessionId>,
}

/// The queue and executor table.
#[derive(Debug)]
pub struct Scheduler {
    order: BTreeMap<(u8, u64), JobId>,
    items: HashMap<JobId, ((u8, u64), QueueItem)>,
    seq: u64,
    execs: Vec<ExecState>,
    queue_max: usize,
    swap: bool,
}

impl Scheduler {
    /// `execs[i].serves` / `resident` describe backend `i`.
    pub fn new(execs: Vec<ExecState>, queue_max: usize, swap: bool) -> Self {
        Self {
            order: BTreeMap::new(),
            items: HashMap::new(),
            seq: 0,
            execs,
            queue_max,
            swap,
        }
    }

    pub fn exec(&self, i: usize) -> &ExecState {
        &self.execs[i]
    }

    pub fn exec_mut(&mut self, i: usize) -> &mut ExecState {
        &mut self.execs[i]
    }

    pub fn executors(&self) -> usize {
        self.execs.len()
    }

    /// Whether executor `i` may run `model` (resident, or loadable in swap mode).
    pub fn can_serve(&self, i: usize, model: &ModelId) -> bool {
        let e = &self.execs[i];
        e.serves.contains(model) && (self.swap || e.resident.contains(model))
    }

    /// Whether any executor may run `model`.
    pub fn served(&self, model: &ModelId) -> bool {
        (0..self.execs.len()).any(|i| self.can_serve(i, model))
    }

    pub fn queued_batch(&self) -> usize {
        self.items
            .values()
            .filter(|(_, it)| it.prio == Priority::Batch)
            .count()
    }

    pub fn queued(&self) -> usize {
        self.items.len()
    }

    pub fn is_queued(&self, job: JobId) -> bool {
        self.items.contains_key(&job)
    }

    /// Adds a job. `QueueFull` past `queue_max` batch jobs; `InvalidRequest`
    /// when no executor serves the model.
    pub fn enqueue(&mut self, item: QueueItem) -> Result<u32, ApiError> {
        let servable = match item.pin {
            Some(i) => i < self.execs.len() && self.can_serve(i, &item.model),
            None => self.served(&item.model),
        };
        if !servable {
            return Err(ApiError::invalid_param(
                "model",
                format!("model `{}` is not served by this engine", item.model),
            ));
        }
        if item.prio == Priority::Batch && self.queued_batch() >= self.queue_max {
            return Err(ApiError::queue_full(format!(
                "the queue is full ({} jobs)",
                self.queue_max
            ))
            .with_retry_after(5));
        }
        if self.items.contains_key(&item.job) {
            return Err(ApiError::conflict(format!("job {} is already queued", item.job)));
        }
        let key = (item.prio.rank(), self.seq);
        self.seq += 1;
        self.order.insert(key, item.job);
        self.items.insert(item.job, (key, item));
        Ok(self.position(key))
    }

    fn position(&self, key: (u8, u64)) -> u32 {
        self.order.range(..key).count() as u32
    }

    /// The queue position of `job`, if queued.
    pub fn position_of(&self, job: JobId) -> Option<u32> {
        self.items.get(&job).map(|(k, _)| self.position(*k))
    }

    /// Every queued job with its position, in dispatch order.
    pub fn positions(&self) -> Vec<(JobId, u32)> {
        self.order
            .values()
            .enumerate()
            .map(|(i, j)| (*j, i as u32))
            .collect()
    }

    /// Removes a queued job (cancel). Returns it if it was queued.
    pub fn remove(&mut self, job: JobId) -> Option<QueueItem> {
        let (key, item) = self.items.remove(&job)?;
        self.order.remove(&key);
        Some(item)
    }

    /// Removes every queued job (drain).
    pub fn clear(&mut self) -> Vec<QueueItem> {
        self.order.clear();
        let mut v: Vec<_> = self.items.drain().map(|(_, (k, it))| (k, it)).collect();
        v.sort_by_key(|(k, _)| *k);
        v.into_iter().map(|(_, it)| it).collect()
    }

    /// The next thing executor `i` should do, marking a job running. `None`
    /// when it is busy or nothing runnable is queued for it.
    pub fn next_for(&mut self, i: usize) -> Option<Dispatch> {
        if self.execs[i].running.is_some() {
            return None;
        }
        if let Some(s) = self.execs[i].lease {
            return Some(Dispatch::Causal(s));
        }
        let pick = self.order.iter().find_map(|(key, job)| {
            let (_, it) = &self.items[job];
            let pinned_ok = it.pin.is_none_or(|p| p == i);
            (pinned_ok && self.can_serve(i, &it.model)).then_some(*key)
        })?;
        let job = self.order.remove(&pick)?;
        let (_, item) = self.items.remove(&job)?;
        self.execs[i].running = Some(job);
        Some(Dispatch::Job(item))
    }

    /// Executor `i` finished its running job.
    pub fn finish(&mut self, i: usize) -> Option<JobId> {
        self.execs[i].running.take()
    }

    /// The executor currently running `job`.
    pub fn running_on(&self, job: JobId) -> Option<usize> {
        self.execs.iter().position(|e| e.running == Some(job))
    }

    /// An executor able to host a new session for `model`: serves it
    /// resident and has no session. Prefers an idle one.
    pub fn session_executor(&self, model: &ModelId) -> Option<usize> {
        let free: Vec<usize> = (0..self.execs.len())
            .filter(|&i| {
                let e = &self.execs[i];
                e.session.is_none() && e.serves.contains(model) && e.resident.contains(model)
            })
            .collect();
        free.iter()
            .copied()
            .find(|&i| self.execs[i].running.is_none())
            .or_else(|| free.first().copied())
    }

    /// Occupies executor `i` with a session; `exclusive` also grants the
    /// causal lease.
    pub fn open_session(&mut self, i: usize, s: SessionId, exclusive: bool) -> Result<(), ApiError> {
        let e = &mut self.execs[i];
        if e.session.is_some() {
            return Err(busy());
        }
        e.session = Some(s);
        if exclusive {
            e.lease = Some(s);
        }
        Ok(())
    }

    /// Releases session `s` wherever it is (and its lease).
    pub fn close_session(&mut self, s: SessionId) -> Option<usize> {
        let i = self.execs.iter().position(|e| e.session == Some(s))?;
        let e = &mut self.execs[i];
        e.session = None;
        if e.lease == Some(s) {
            e.lease = None;
        }
        Some(i)
    }

    /// Whether any executor serving `model` resident has a free session slot.
    pub fn session_busy(&self, model: &ModelId) -> bool {
        self.session_executor(model).is_none()
    }
}

/// "One streaming session per executor": the protocol renders its own busy
/// answer (Reactor 409, WMA `/session` 429, `/fv/v1/streams` 429).
pub fn busy() -> ApiError {
    ApiError::conflict("a streaming session is already open on this engine").with_retry_after(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exec(models: &[&str]) -> ExecState {
        let s: BTreeSet<ModelId> = models.iter().map(|m| ModelId::new(*m)).collect();
        ExecState {
            serves: s.clone(),
            resident: s,
            ..ExecState::default()
        }
    }

    fn item(p: Priority, m: &str) -> QueueItem {
        QueueItem {
            job: JobId::new(),
            prio: p,
            model: m.into(),
            pin: None,
        }
    }

    #[test]
    fn stream_before_batch_fifo_within() {
        let mut s = Scheduler::new(vec![exec(&["m"])], 10, false);
        let b1 = item(Priority::Batch, "m");
        let b2 = item(Priority::Batch, "m");
        let s1 = item(Priority::Stream, "m");
        let s2 = item(Priority::Stream, "m");
        assert_eq!(s.enqueue(b1.clone()).unwrap(), 0);
        assert_eq!(s.enqueue(b2.clone()).unwrap(), 1);
        assert_eq!(s.enqueue(s1.clone()).unwrap(), 0);
        assert_eq!(s.enqueue(s2.clone()).unwrap(), 1);
        assert_eq!(s.position_of(b1.job), Some(2));
        assert_eq!(
            s.positions().iter().map(|p| p.0).collect::<Vec<_>>(),
            vec![s1.job, s2.job, b1.job, b2.job]
        );
        let mut got = vec![];
        while let Some(Dispatch::Job(it)) = s.next_for(0) {
            assert!(s.next_for(0).is_none(), "busy executor gets nothing");
            got.push(it.job);
            s.finish(0);
        }
        assert_eq!(got, vec![s1.job, s2.job, b1.job, b2.job]);
    }

    #[test]
    fn queue_max_counts_batch_only() {
        let mut s = Scheduler::new(vec![exec(&["m"])], 1, false);
        s.enqueue(item(Priority::Batch, "m")).unwrap();
        let e = s.enqueue(item(Priority::Batch, "m")).unwrap_err();
        assert_eq!(e.kind, fastvideo_protocol::ErrorKind::QueueFull);
        s.enqueue(item(Priority::Stream, "m")).unwrap();
        assert!(s.enqueue(item(Priority::Batch, "other")).is_err());
    }

    #[test]
    fn remove_updates_positions() {
        let mut s = Scheduler::new(vec![exec(&["m"])], 10, false);
        let a = item(Priority::Batch, "m");
        let b = item(Priority::Batch, "m");
        s.enqueue(a.clone()).unwrap();
        s.enqueue(b.clone()).unwrap();
        assert_eq!(s.remove(a.job).unwrap().job, a.job);
        assert!(s.remove(a.job).is_none());
        assert_eq!(s.position_of(b.job), Some(0));
    }

    #[test]
    fn placement_by_model_and_pin() {
        let mut s = Scheduler::new(vec![exec(&["a"]), exec(&["a", "b"])], 10, false);
        let b = item(Priority::Batch, "b");
        let a = item(Priority::Batch, "a");
        s.enqueue(b.clone()).unwrap();
        s.enqueue(a.clone()).unwrap();
        // Executor 0 cannot run `b`; it skips to `a`.
        assert_eq!(s.next_for(0), Some(Dispatch::Job(a)));
        assert_eq!(s.next_for(1), Some(Dispatch::Job(b)));
        let mut p = item(Priority::Stream, "a");
        p.pin = Some(1);
        s.enqueue(p.clone()).unwrap();
        s.finish(0);
        assert_eq!(s.next_for(0), None);
        s.finish(1);
        assert_eq!(s.next_for(1), Some(Dispatch::Job(p)));
    }

    #[test]
    fn causal_lease_is_exclusive() {
        let mut s = Scheduler::new(vec![exec(&["m"])], 10, false);
        let a = item(Priority::Batch, "m");
        s.enqueue(a.clone()).unwrap();
        let sid = SessionId::new();
        assert_eq!(s.session_executor(&"m".into()), Some(0));
        s.open_session(0, sid, true).unwrap();
        assert!(s.session_busy(&"m".into()));
        assert!(s.open_session(0, SessionId::new(), false).is_err());
        assert_eq!(s.next_for(0), Some(Dispatch::Causal(sid)));
        assert_eq!(s.next_for(0), Some(Dispatch::Causal(sid)));
        assert_eq!(s.position_of(a.job), Some(0));
        assert_eq!(s.close_session(sid), Some(0));
        assert_eq!(s.next_for(0), Some(Dispatch::Job(a)));
    }

    #[test]
    fn lease_waits_for_running_job() {
        let mut s = Scheduler::new(vec![exec(&["m"])], 10, false);
        let a = item(Priority::Batch, "m");
        s.enqueue(a).unwrap();
        assert!(matches!(s.next_for(0), Some(Dispatch::Job(_))));
        let sid = SessionId::new();
        s.open_session(0, sid, true).unwrap();
        assert_eq!(s.next_for(0), None);
        s.finish(0);
        assert_eq!(s.next_for(0), Some(Dispatch::Causal(sid)));
    }

    #[test]
    fn swap_mode_admits_non_resident() {
        let mut e = exec(&["a", "b"]);
        e.resident.remove(&ModelId::new("b"));
        let s = Scheduler::new(vec![e.clone()], 10, false);
        assert!(!s.served(&"b".into()));
        let s = Scheduler::new(vec![e], 10, true);
        assert!(s.served(&"b".into()));
    }
}
