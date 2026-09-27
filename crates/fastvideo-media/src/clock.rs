//! Injectable monotonic clock, plus the RTP clocks derived from tick counters.
//!
//! The `Clock`, `MonotonicClock` and `ManualClock` items are ported from
//! strobe-core `src/clock.rs` (strobe, MIT License; see `NOTICE` in this
//! crate for the copyright and permission notice). The RTP clocks below are
//! original to fastvideo-rs.
//!
//! Original strobe-core note: the Python pacer takes `clock: Callable[[], float]`
//! so the adaptive-fps tests can drive time deterministically instead of
//! sleeping. Same idea here: the adaptive estimator is pure arithmetic over
//! arrival timestamps, and it should stay testable without a runtime or a
//! wall-clock dependency.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Monotonic seconds. Only differences are meaningful; the origin is arbitrary.
pub trait Clock: Send + Sync {
    fn now(&self) -> f64;
}

/// Process-monotonic clock. Equivalent to Python's `time.monotonic`.
#[derive(Debug)]
pub struct MonotonicClock {
    origin: Instant,
}

impl MonotonicClock {
    pub fn new() -> Self {
        Self { origin: Instant::now() }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

/// Test clock advanced by hand. Mirrors the `clock={"t": 0.0}` closure the
/// Python pacing tests use.
#[derive(Debug, Default)]
pub struct ManualClock {
    t: AtomicU64,
}

impl ManualClock {
    pub fn new(start: f64) -> Self {
        Self { t: AtomicU64::new(start.to_bits()) }
    }

    pub fn advance(&self, dt: f64) {
        let cur = f64::from_bits(self.t.load(Ordering::Relaxed));
        self.t.store((cur + dt).to_bits(), Ordering::Relaxed);
    }

    pub fn set(&self, t: f64) {
        self.t.store(t.to_bits(), Ordering::Relaxed);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> f64 {
        f64::from_bits(self.t.load(Ordering::Relaxed))
    }
}

impl<C: Clock + ?Sized> Clock for std::sync::Arc<C> {
    fn now(&self) -> f64 {
        (**self).now()
    }
}

impl<C: Clock + ?Sized> Clock for &C {
    fn now(&self) -> f64 {
        (**self).now()
    }
}

/// H.264 RTP clock rate.
pub const VIDEO_RTP_HZ: u32 = 90_000;

/// The 90 kHz video RTP timestamp, derived from the tick count so it never
/// drifts: after `n` ticks at a fixed `fps` it is exactly `n·90000/fps`
/// (floored for rates that do not divide 90000). In adaptive mode each tick
/// advances by `90000/effective_fps` (§5.4); the running sum is kept in f64
/// and rounded, so rounding error never accumulates either.
#[derive(Debug, Clone)]
pub struct VideoRtpClock {
    base: u32,
    ticks: u64,
    /// Fixed-rate numerator and denominator (`fps_num/fps_den` frames per second).
    fps_num: u64,
    fps_den: u64,
    /// Adaptive mode: exact running sum of 90 kHz ticks.
    adaptive_acc: f64,
    adaptive: bool,
}

impl VideoRtpClock {
    /// A fixed-rate clock at `fps` frames per second.
    pub fn fixed(fps: u32, base: u32) -> Self {
        Self { base, ticks: 0, fps_num: u64::from(fps.max(1)), fps_den: 1, adaptive_acc: 0.0, adaptive: false }
    }

    /// An adaptive clock; call [`advance_adaptive`](Self::advance_adaptive).
    pub fn adaptive(base: u32) -> Self {
        Self { base, ticks: 0, fps_num: 1, fps_den: 1, adaptive_acc: 0.0, adaptive: true }
    }

    /// The timestamp of the frame about to be sent.
    pub fn current(&self) -> u32 {
        let off = if self.adaptive {
            self.adaptive_acc.round() as u64
        } else {
            (u128::from(self.ticks) * u128::from(VIDEO_RTP_HZ) * u128::from(self.fps_den) / u128::from(self.fps_num))
                as u64
        };
        self.base.wrapping_add(off as u32)
    }

    /// Fixed mode: return the current timestamp, then advance one frame.
    pub fn advance(&mut self) -> u32 {
        let ts = self.current();
        self.ticks += 1;
        ts
    }

    /// Adaptive mode: return the current timestamp, then advance by
    /// `90000/effective_fps`.
    pub fn advance_adaptive(&mut self, effective_fps: f64) -> u32 {
        let ts = self.current();
        self.ticks += 1;
        self.adaptive_acc += f64::from(VIDEO_RTP_HZ) / effective_fps.max(1e-3);
        ts
    }

    pub fn ticks(&self) -> u64 {
        self.ticks
    }
}

/// The Opus RTP clock (48 kHz): the global sample counter plus a random base.
#[derive(Debug, Clone)]
pub struct AudioRtpClock {
    base: u32,
    samples: u64,
}

impl AudioRtpClock {
    pub fn new(base: u32) -> Self {
        Self { base, samples: 0 }
    }

    /// Return the timestamp of a packet of `frames` samples per channel,
    /// then advance past it.
    pub fn advance(&mut self, frames: u64) -> u32 {
        let ts = self.base.wrapping_add(self.samples as u32);
        self.samples += frames;
        ts
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_advances() {
        let c = ManualClock::new(1.5);
        c.advance(0.25);
        assert_eq!(c.now(), 1.75);
        c.set(10.0);
        assert_eq!(c.now(), 10.0);
    }

    #[test]
    fn monotonic_clock_moves_forward() {
        let c = MonotonicClock::new();
        let a = c.now();
        let b = c.now();
        assert!(b >= a);
    }

    #[test]
    fn video_rtp_fixed_rates_are_exact() {
        for (fps, step) in [(24u32, 3750u64), (16, 5625), (25, 3600), (30, 3000)] {
            let mut c = VideoRtpClock::fixed(fps, 0);
            for _ in 0..1_000_000 {
                c.advance();
            }
            assert_eq!(u64::from(c.current()), (1_000_000 * step) % (1u64 << 32));
        }
    }

    #[test]
    fn video_rtp_adaptive_has_no_rounding_drift() {
        let mut c = VideoRtpClock::adaptive(100);
        for _ in 0..30_000 {
            c.advance_adaptive(7.0);
        }
        let want = 100u64 + (30_000.0f64 * 90_000.0 / 7.0).round() as u64;
        assert_eq!(u64::from(c.current()), want % (1u64 << 32));
    }

    #[test]
    fn audio_rtp_is_the_sample_counter() {
        let mut a = AudioRtpClock::new(u32::MAX - 10);
        assert_eq!(a.advance(960), u32::MAX - 10);
        assert_eq!(a.advance(960), (u32::MAX - 10).wrapping_add(960));
        assert_eq!(a.samples(), 1920);
    }
}
