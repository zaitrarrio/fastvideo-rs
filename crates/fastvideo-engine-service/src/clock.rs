//! Clocks for the fake backend: real time, or a [`ManualClock`] tests advance
//! by hand so scheduling, progress and cancellation are deterministic.

use std::fmt;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::cancel::lock;

/// Time source used on the executor thread (blocking, never async).
pub trait Clock: Send + Sync + fmt::Debug + 'static {
    /// Time since the clock's origin.
    fn now(&self) -> Duration;
    /// Blocks the calling thread for `d` of this clock's time.
    fn sleep(&self, d: Duration);
}

/// Wall-clock time.
#[derive(Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
    fn sleep(&self, d: Duration) {
        if !d.is_zero() {
            std::thread::sleep(d);
        }
    }
}

#[derive(Debug, Default)]
struct ManualState {
    now: Duration,
    /// Deadlines of the threads in `sleep`, including any an `advance`
    /// released that have not run yet.
    parked: Vec<Duration>,
    /// Sleeps that parked and then completed.
    wakeups: u64,
}

impl ManualState {
    /// Threads in `sleep` that are still waiting for time to move: their
    /// deadline is ahead of `now`. A sleeper released by an `advance` stops
    /// counting here at once, under the same lock, not when its thread next
    /// gets the CPU and leaves `sleep`.
    fn blocked(&self) -> usize {
        self.parked.iter().filter(|&&d| d > self.now).count()
    }
}

/// Virtual time. `sleep(d)` parks the caller until a test [`advance`]s the
/// clock past the deadline; a zero sleep returns at once.
///
/// [`advance`]: ManualClock::advance
#[derive(Debug, Default)]
pub struct ManualClock {
    state: Mutex<ManualState>,
    cv: Condvar,
}

impl ManualClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Moves time forward and wakes every sleeper whose deadline passed.
    pub fn advance(&self, d: Duration) {
        lock(&self.state).now += d;
        self.cv.notify_all();
    }

    /// Threads parked in `sleep` whose deadline is still ahead: blocked
    /// until the clock is advanced again.
    ///
    /// A sleeper an [`advance`](ManualClock::advance) released stops
    /// counting at once, before its thread runs. A driver that advances
    /// whenever this is non-zero therefore advances once per deadline, and
    /// time cannot move on under a sleeper that was released but has not yet
    /// returned (and read `now`). Counting it until it left `sleep`, the
    /// count stayed stale for as long as the woken thread waited for a CPU,
    /// and such a driver advanced past the end of the sleep it had already
    /// released.
    pub fn sleepers(&self) -> usize {
        lock(&self.state).blocked()
    }

    /// Sleeps that parked and then completed since the clock was made
    /// (zero sleeps do not count). With [`wait_for_wakeups`] a test
    /// steps a sleeper without observing anything else it does.
    ///
    /// [`wait_for_wakeups`]: ManualClock::wait_for_wakeups
    pub fn wakeups(&self) -> u64 {
        lock(&self.state).wakeups
    }

    /// Blocks (really) until [`wakeups`](ManualClock::wakeups) reaches `n`,
    /// or `timeout` of wall time passes. Returns whether it did.
    pub fn wait_for_wakeups(&self, n: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = lock(&self.state);
        while st.wakeups < n {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            st = self
                .cv
                .wait_timeout(st, left.min(Duration::from_millis(20)))
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
        true
    }

    /// Blocks (really) until at least `n` threads are blocked in `sleep`
    /// (as [`sleepers`](ManualClock::sleepers) counts them), or `timeout` of
    /// wall time passes. Returns whether the count was reached.
    pub fn wait_for_sleepers(&self, n: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = lock(&self.state);
        while st.blocked() < n {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            st = self
                .cv
                .wait_timeout(st, left.min(Duration::from_millis(20)))
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
        true
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        lock(&self.state).now
    }
    fn sleep(&self, d: Duration) {
        let mut st = lock(&self.state);
        let deadline = st.now + d;
        if st.now >= deadline {
            return;
        }
        st.parked.push(deadline);
        self.cv.notify_all();
        while st.now < deadline {
            st = self.cv.wait(st).unwrap_or_else(|p| p.into_inner());
        }
        if let Some(i) = st.parked.iter().position(|&p| p == deadline) {
            st.parked.swap_remove(i);
        }
        st.wakeups += 1;
        self.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn manual_clock_parks_until_advanced() {
        let c = Arc::new(ManualClock::new());
        c.sleep(Duration::ZERO);
        let c2 = c.clone();
        let t = std::thread::spawn(move || {
            c2.sleep(Duration::from_secs(2));
            c2.now()
        });
        assert!(c.wait_for_sleepers(1, Duration::from_secs(5)));
        c.advance(Duration::from_secs(1));
        assert!(c.wait_for_sleepers(1, Duration::from_secs(5)));
        c.advance(Duration::from_secs(1));
        assert_eq!(t.join().unwrap(), Duration::from_secs(2));
        assert_eq!(c.sleepers(), 0);
        assert_eq!(
            c.wakeups(),
            1,
            "one parked sleep completed; the zero sleep did not count"
        );
        assert!(c.wait_for_wakeups(1, Duration::ZERO));
        assert!(!c.wait_for_wakeups(2, Duration::from_millis(5)));
    }

    /// The `advance` that releases a sleeper takes it out of the count at
    /// once: whether its thread has run yet does not matter.
    #[test]
    fn a_released_sleeper_stops_counting_before_it_runs() {
        let c = Arc::new(ManualClock::new());
        let c2 = c.clone();
        let t = std::thread::spawn(move || {
            c2.sleep(Duration::from_secs(1));
            c2.now()
        });
        assert!(c.wait_for_sleepers(1, Duration::from_secs(5)));
        c.advance(Duration::from_secs(1));
        assert_eq!(c.sleepers(), 0, "released, maybe not yet running");
        assert!(!c.wait_for_sleepers(1, Duration::ZERO));
        assert_eq!(t.join().unwrap(), Duration::from_secs(1));
        assert!(c.wait_for_wakeups(1, Duration::ZERO));
    }

    /// A driver that advances whenever someone sleeps (the engine tests'
    /// `Driver`, polling here as fast as it can) moves time once per
    /// sleep: every sleep returns at exactly its deadline. With a stale
    /// count it advanced again while the released sleeper had not yet
    /// returned, and the sleep measured 2 s or more.
    #[test]
    fn a_driver_advances_once_per_sleep() {
        let c = Arc::new(ManualClock::new());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (c2, s2) = (c.clone(), stop.clone());
        let driver = std::thread::spawn(move || {
            while !s2.load(std::sync::atomic::Ordering::SeqCst) {
                if c2.sleepers() > 0 {
                    c2.advance(Duration::from_secs(1));
                }
                std::thread::yield_now();
            }
        });
        let late: Vec<(usize, Duration)> = (0..500)
            .filter_map(|i| {
                let t0 = c.now();
                c.sleep(Duration::from_secs(1));
                let d = c.now() - t0;
                (d != Duration::from_secs(1)).then_some((i, d))
            })
            .collect();
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        driver.join().unwrap();
        assert!(late.is_empty(), "sleeps that saw extra advances: {late:?}");
    }
}
