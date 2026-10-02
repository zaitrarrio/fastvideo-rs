//! The per-host capacity arbiter (docs/serve/dispatch-do-family.md §6.3).
//!
//! A worker that serves several model families holds one socket per family
//! Durable Object, and each of them may offer it work at any time. The
//! arbiter is the single place that decides whether this GPU takes an offer:
//!
//! 1. a job is taken only while fewer than `capacity` jobs are held (across
//!    all families) **and** no exclusive session holds the GPU; the check
//!    and the reservation are one step under one lock, so two offers from
//!    two families can never both pass;
//! 2. otherwise the offer is refused, and the link nacks it 429 (the family
//!    object offers it to another worker at once);
//! 3. a session is taken only while fewer than `sessions` sessions are live
//!    and, when sessions are exclusive, no job is held;
//! 4. every change bumps a version that the family links watch: each one
//!    then reports its credits (`slots`) to its object.
//!
//! Draining is the links' business (they report no free slot).

use std::collections::BTreeMap;
use std::sync::Mutex;

use tokio::sync::watch;

/// A slot holder.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Holder {
    family: String,
}

#[derive(Debug, Default)]
struct Inner {
    jobs: BTreeMap<String, Holder>,
    sessions: BTreeMap<String, Holder>,
    /// Most jobs ever held at once (for the overbooking check).
    peak: u32,
}

/// Free slots as the arbiter sees them now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Free {
    pub jobs: u32,
    pub sessions: u32,
}

/// Why an offer was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Every job slot is held (by `families`).
    Full { families: Vec<String> },
    /// An exclusive session holds the GPU.
    Session,
    /// Jobs hold the GPU (exclusive sessions need it idle).
    Busy,
    /// No session slot.
    NoSessionSlot,
    /// Already held (a repeated offer).
    Held,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Full { families } => write!(f, "every slot of this GPU is held (families: {})", families.join(", ")),
            Refusal::Session => f.write_str("a streaming session holds this GPU"),
            Refusal::Busy => f.write_str("jobs hold this GPU (sessions need it idle)"),
            Refusal::NoSessionSlot => f.write_str("no free session slot"),
            Refusal::Held => f.write_str("already held here"),
        }
    }
}

/// One GPU's slot budget, shared by every family link of the process.
#[derive(Debug)]
pub struct Arbiter {
    capacity: u32,
    sessions: u32,
    exclusive: bool,
    inner: Mutex<Inner>,
    version: watch::Sender<u64>,
}

impl Arbiter {
    /// `capacity` jobs at once (≥ 1), `sessions` sessions (0: none);
    /// `exclusive`: a session excludes jobs and the reverse.
    pub fn new(capacity: u32, sessions: u32, exclusive: bool) -> Self {
        let (version, _) = watch::channel(0);
        Self { capacity: capacity.max(1), sessions, exclusive, inner: Mutex::default(), version }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn bump(&self) {
        self.version.send_modify(|v| *v += 1);
    }

    /// Changes of the budget (a new value after every take and release).
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.version.subscribe()
    }

    fn free_of(&self, g: &Inner) -> Free {
        let held = g.jobs.len() as u32;
        let live = g.sessions.len() as u32;
        let jobs = if self.exclusive && live > 0 { 0 } else { self.capacity.saturating_sub(held) };
        let sessions = if self.exclusive && held > 0 { 0 } else { self.sessions.saturating_sub(live) };
        Free { jobs, sessions }
    }

    pub fn free(&self) -> Free {
        let g = self.lock();
        self.free_of(&g)
    }

    /// Reserves a job slot for `job` of `family` (rule 1).
    pub fn try_take_job(&self, family: &str, job: &str) -> Result<(), Refusal> {
        let mut g = self.lock();
        if g.jobs.contains_key(job) {
            return Err(Refusal::Held);
        }
        if self.exclusive && !g.sessions.is_empty() {
            return Err(Refusal::Session);
        }
        if g.jobs.len() as u32 >= self.capacity {
            let mut families: Vec<String> = g.jobs.values().map(|h| h.family.clone()).collect();
            families.dedup();
            return Err(Refusal::Full { families });
        }
        g.jobs.insert(job.to_owned(), Holder { family: family.to_owned() });
        let n = g.jobs.len() as u32;
        g.peak = g.peak.max(n);
        if n > self.capacity {
            // Cannot happen (checked under the lock); counted to prove it.
            metrics::counter!("fv_worker_arbiter_overbooked_total").increment(1);
        }
        drop(g);
        metrics::counter!("fv_worker_arbiter_taken_total", "family" => family.to_owned()).increment(1);
        self.bump();
        Ok(())
    }

    /// Frees `job`'s slot (no-op when it holds none).
    pub fn release_job(&self, job: &str) {
        let removed = self.lock().jobs.remove(job).is_some();
        if removed {
            self.bump();
        }
    }

    /// Reserves the GPU for session `id` of `family` (rule 3).
    pub fn try_take_session(&self, family: &str, id: &str) -> Result<(), Refusal> {
        let mut g = self.lock();
        if g.sessions.contains_key(id) {
            return Ok(());
        }
        if g.sessions.len() as u32 >= self.sessions {
            return Err(Refusal::NoSessionSlot);
        }
        if self.exclusive && !g.jobs.is_empty() {
            return Err(Refusal::Busy);
        }
        g.sessions.insert(id.to_owned(), Holder { family: family.to_owned() });
        drop(g);
        self.bump();
        Ok(())
    }

    pub fn release_session(&self, id: &str) {
        let removed = self.lock().sessions.remove(id).is_some();
        if removed {
            self.bump();
        }
    }

    /// Sessions held for `family` (re-announced after a reconnect).
    pub fn sessions_of(&self, family: &str) -> Vec<String> {
        self.lock().sessions.iter().filter(|(_, h)| h.family == family).map(|(k, _)| k.clone()).collect()
    }

    /// Jobs held now and the most ever held at once.
    pub fn held(&self) -> (u32, u32) {
        let g = self.lock();
        (g.jobs.len() as u32, g.peak)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn concurrent_offers_from_two_families_never_overbook() {
        for _ in 0..50 {
            let a = Arc::new(Arbiter::new(1, 1, true));
            let barrier = Arc::new(std::sync::Barrier::new(8));
            let hs: Vec<_> = (0..8)
                .map(|i| {
                    let (a, b) = (a.clone(), barrier.clone());
                    std::thread::spawn(move || {
                        b.wait();
                        a.try_take_job(if i % 2 == 0 { "wan" } else { "ltx" }, &format!("j{i}")).is_ok()
                    })
                })
                .collect();
            let taken = hs.into_iter().map(|h| h.join().unwrap()).filter(|t| *t).count();
            assert_eq!(taken, 1);
            assert_eq!(a.held(), (1, 1));
        }
    }

    #[test]
    fn exclusive_sessions_and_free_counts() {
        let a = Arbiter::new(2, 1, true);
        let mut rx = a.subscribe();
        assert_eq!(a.free(), Free { jobs: 2, sessions: 1 });
        a.try_take_job("wan", "j1").unwrap();
        assert!(rx.has_changed().unwrap());
        rx.borrow_and_update();
        assert_eq!(a.free(), Free { jobs: 1, sessions: 0 });
        assert_eq!(a.try_take_session("sfwan", "s1"), Err(Refusal::Busy));
        assert_eq!(a.try_take_job("ltx", "j1"), Err(Refusal::Held));
        a.try_take_job("ltx", "j2").unwrap();
        assert!(matches!(a.try_take_job("wan", "j3"), Err(Refusal::Full { .. })));
        a.release_job("j1");
        a.release_job("j2");
        a.try_take_session("sfwan", "s1").unwrap();
        assert_eq!(a.free(), Free { jobs: 0, sessions: 0 });
        assert_eq!(a.try_take_job("wan", "j4"), Err(Refusal::Session));
        assert_eq!(a.sessions_of("sfwan"), vec!["s1".to_string()]);
        a.release_session("s1");
        assert_eq!(a.free(), Free { jobs: 2, sessions: 1 });
        // Non-exclusive: both at once.
        let b = Arbiter::new(1, 1, false);
        b.try_take_session("sfwan", "s").unwrap();
        b.try_take_job("wan", "j").unwrap();
        assert_eq!(b.free(), Free { jobs: 0, sessions: 0 });
    }
}
