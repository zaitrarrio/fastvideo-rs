//! Traffic traces: Poisson arrivals with a time-varying rate, from a
//! seeded generator (the same seed gives the same trace everywhere).

use serde::Serialize;

/// SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Exponential with the given mean.
    pub fn exp(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.next_f64()).ln()
    }
}

/// One request.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Arrival {
    /// Seconds (the simulation clock).
    pub t: f64,
    /// A live stream holds one worker for `duration_s`.
    pub stream: bool,
    pub duration_s: f64,
    /// Batch job duration = family mean × this (e.g. 2 for a 10 s clip).
    pub duration_scale: f64,
}

impl Arrival {
    pub fn job(t: f64) -> Self {
        Self { t, stream: false, duration_s: 0.0, duration_scale: 1.0 }
    }
    pub fn stream(t: f64, duration_s: f64) -> Self {
        Self { t, stream: true, duration_s, duration_scale: 1.0 }
    }
}

/// Poisson arrivals on `[start, start + dur)` with rate `rate(t)` per hour
/// (thinning against `max_rate`).
pub fn poisson(rng: &mut Rng, start: f64, dur: f64, max_rate_per_hr: f64, rate_per_hr: impl Fn(f64) -> f64) -> Vec<Arrival> {
    let mut out = Vec::new();
    if max_rate_per_hr <= 0.0 {
        return out;
    }
    let mean_gap = 3600.0 / max_rate_per_hr;
    let mut t = start;
    loop {
        t += rng.exp(mean_gap);
        if t >= start + dur {
            break;
        }
        if rng.next_f64() * max_rate_per_hr < rate_per_hr(t - start) {
            out.push(Arrival::job(t.floor()));
        }
    }
    out
}

/// A constant rate.
pub fn steady(rng: &mut Rng, start: f64, hours: f64, per_hr: f64) -> Vec<Arrival> {
    poisson(rng, start, hours * 3600.0, per_hr, |_| per_hr)
}

/// The diurnal shape `(1 + cos(2π(t - peak)/24h))^k`, normalized to mean 1,
/// with `k` chosen so the busiest hour is `peak_ratio` × the daily average
/// (the business plan: 2.5×).
#[derive(Clone, Debug)]
pub struct Diurnal {
    k: f64,
    norm: f64,
    peak_hour: f64,
}

impl Diurnal {
    pub fn new(peak_ratio: f64, peak_hour: f64) -> Self {
        let shape = |k: f64, h: f64| (1.0 + (2.0 * std::f64::consts::PI * (h - peak_hour) / 24.0).cos()).powf(k);
        // Mean over the day and over the busiest hour, by minutes.
        let stats = |k: f64| {
            let mins: Vec<f64> = (0..1440).map(|m| shape(k, (f64::from(m) + 0.5) / 60.0)).collect();
            let mean = mins.iter().sum::<f64>() / 1440.0;
            let best = mins.chunks(60).map(|c| c.iter().sum::<f64>() / 60.0).fold(0.0, f64::max);
            (mean, best / mean)
        };
        let (mut lo, mut hi) = (0.01, 8.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if stats(mid).1 < peak_ratio {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let k = 0.5 * (lo + hi);
        Self { k, norm: stats(k).0, peak_hour }
    }
    /// Rate multiplier at hour-of-day `h` (mean 1 over a day).
    pub fn factor(&self, h: f64) -> f64 {
        (1.0 + (2.0 * std::f64::consts::PI * (h - self.peak_hour) / 24.0).cos()).powf(self.k) / self.norm
    }
    pub fn max_factor(&self) -> f64 {
        2f64.powf(self.k) / self.norm
    }
    /// Busiest-hour / average, by the hour.
    pub fn busiest_hour_ratio(&self) -> f64 {
        (0..24)
            .map(|h| (0..60).map(|m| self.factor(f64::from(h) + (f64::from(m) + 0.5) / 60.0)).sum::<f64>() / 60.0)
            .fold(0.0, f64::max)
    }
}

/// `hours` of diurnal traffic averaging `avg_per_hr`, peaking at `peak_hour`
/// (hours after `start`, which is taken as midnight).
pub fn diurnal(rng: &mut Rng, start: f64, hours: f64, avg_per_hr: f64, peak_ratio: f64, peak_hour: f64) -> Vec<Arrival> {
    let d = Diurnal::new(peak_ratio, peak_hour);
    let max = avg_per_hr * d.max_factor();
    poisson(rng, start, hours * 3600.0, max, |t| avg_per_hr * d.factor((t / 3600.0) % 24.0))
}

/// A steady base with `mult` × the base rate for `spike_min` minutes
/// starting `at_h` hours in.
pub fn spike(rng: &mut Rng, start: f64, hours: f64, base_per_hr: f64, mult: f64, at_h: f64, spike_min: f64) -> Vec<Arrival> {
    let (a, b) = (at_h * 3600.0, at_h * 3600.0 + spike_min * 60.0);
    poisson(rng, start, hours * 3600.0, base_per_hr * mult, |t| {
        if t >= a && t < b {
            base_per_hr * mult
        } else {
            base_per_hr
        }
    })
}

/// Gives every batch job a duration scale uniform in `1 ± jitter` (so all
/// strategies replay the same job durations).
pub fn jitter(rng: &mut Rng, v: &mut [Arrival], jitter: f64) {
    for a in v.iter_mut().filter(|a| !a.stream) {
        a.duration_scale = 1.0 + jitter * (2.0 * rng.next_f64() - 1.0);
    }
}

/// Nothing for `idle_h` hours, then `n` jobs spread over `burst_s`
/// seconds, then nothing for `tail_h` hours.
pub fn idle_burst(rng: &mut Rng, start: f64, idle_h: f64, n: usize, burst_s: f64, tail_h: f64) -> (Vec<Arrival>, f64) {
    let t0 = start + idle_h * 3600.0;
    let mut v: Vec<Arrival> = (0..n).map(|_| Arrival::job((t0 + rng.next_f64() * burst_s).floor())).collect();
    v.sort_by(|a, b| a.t.total_cmp(&b.t));
    (v, idle_h + burst_s / 3600.0 + tail_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diurnal_busiest_hour_is_2_5x_average() {
        let d = Diurnal::new(2.5, 14.0);
        assert!((d.busiest_hour_ratio() - 2.5).abs() < 0.01, "{}", d.busiest_hour_ratio());
        let mean: f64 = (0..1440).map(|m| d.factor((f64::from(m) + 0.5) / 60.0)).sum::<f64>() / 1440.0;
        assert!((mean - 1.0).abs() < 1e-6);
        // Generated traffic matches: ~avg × 24 jobs, and the peak hour holds ~2.5× the mean hour.
        let mut rng = Rng::new(7);
        let v = diurnal(&mut rng, 0.0, 24.0 * 20.0, 100.0, 2.5, 14.0);
        let per_day = v.len() as f64 / 20.0;
        assert!((per_day - 2400.0).abs() < 100.0, "{per_day}");
        let at14 = v.iter().filter(|a| ((a.t / 3600.0) % 24.0).floor() == 14.0).count() as f64 / 20.0;
        assert!((at14 / 100.0 - 2.5).abs() < 0.25, "{at14}");
    }

    #[test]
    fn seeded_traces_repeat() {
        let a = steady(&mut Rng::new(1), 0.0, 2.0, 60.0);
        let b = steady(&mut Rng::new(1), 0.0, 2.0, 60.0);
        assert_eq!(a, b);
        assert!((a.len() as f64 - 120.0).abs() < 40.0);
        let s = spike(&mut Rng::new(2), 0.0, 4.0, 30.0, 10.0, 2.0, 10.0);
        let inside = s.iter().filter(|x| x.t >= 7200.0 && x.t < 7800.0).count();
        assert!(inside > 25, "{inside}");
        let (b, h) = idle_burst(&mut Rng::new(3), 0.0, 2.0, 20, 120.0, 1.0);
        assert_eq!(b.len(), 20);
        assert!(b.iter().all(|x| x.t >= 7200.0 && x.t < 7320.0));
        assert!((h - 3.0333).abs() < 0.01);
    }
}
