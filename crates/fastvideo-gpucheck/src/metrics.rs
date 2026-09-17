//! Comparison metrics. All accumulate in f64 so large tensors don't lose precision.

use serde_json::{json, Value};

#[derive(Debug, Clone, Copy)]
pub struct Diff {
    pub rel_l2: f64,
    pub max_abs: f64,
    pub cosine: f64,
    pub ref_norm: f64,
    pub non_finite: usize,
}

impl Diff {
    pub fn to_json(self) -> Value {
        json!({
            "rel_l2": self.rel_l2,
            "max_abs": self.max_abs,
            "cosine": self.cosine,
            "ref_norm": self.ref_norm,
            "non_finite": self.non_finite,
        })
    }

    /// Passes when both tensors are finite and `rel_l2 <= max_rel`.
    pub fn within(self, max_rel: f64) -> bool {
        self.non_finite == 0 && self.rel_l2.is_finite() && self.rel_l2 <= max_rel
    }
}

/// `actual` vs `reference`. Length mismatch is reported as a total failure.
pub fn diff(actual: &[f32], reference: &[f32]) -> Diff {
    if actual.len() != reference.len() {
        return Diff {
            rel_l2: f64::INFINITY,
            max_abs: f64::INFINITY,
            cosine: 0.0,
            ref_norm: 0.0,
            non_finite: actual.len().max(reference.len()),
        };
    }
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    let mut act2 = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut non_finite = 0usize;
    for (&a, &r) in actual.iter().zip(reference) {
        if !a.is_finite() || !r.is_finite() {
            non_finite += 1;
            continue;
        }
        let (a, r) = (f64::from(a), f64::from(r));
        let d = a - r;
        err2 += d * d;
        ref2 += r * r;
        act2 += a * a;
        dot += a * r;
        max_abs = max_abs.max(d.abs());
    }
    let ref_norm = ref2.sqrt();
    let rel_l2 = if ref_norm > 0.0 {
        err2.sqrt() / ref_norm
    } else if err2 == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let denom = (ref2 * act2).sqrt();
    let cosine = if denom > 0.0 {
        dot / denom
    } else if ref2 == act2 {
        1.0
    } else {
        0.0
    };
    Diff {
        rel_l2,
        max_abs,
        cosine,
        ref_norm,
        non_finite,
    }
}

/// PSNR in dB for signals with the given peak-to-peak range (2.0 for [-1, 1]).
pub fn psnr(actual: &[f32], reference: &[f32], range: f64) -> f64 {
    if actual.len() != reference.len() || actual.is_empty() {
        return 0.0;
    }
    let mut mse = 0.0f64;
    for (&a, &r) in actual.iter().zip(reference) {
        let (a, r) = (f64::from(a).clamp(-1e6, 1e6), f64::from(r).clamp(-1e6, 1e6));
        mse += (a - r) * (a - r);
    }
    mse /= actual.len() as f64;
    if !mse.is_finite() {
        return 0.0;
    }
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * ((range * range) / mse).log10()
}

pub fn non_finite(values: &[f32]) -> usize {
    values.iter().filter(|v| !v.is_finite()).count()
}

pub fn mean_std(values: &[f32]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mut sum = 0.0f64;
    let mut sq = 0.0f64;
    for &v in values {
        let v = f64::from(v);
        sum += v;
        sq += v * v;
    }
    let mean = sum / n;
    (mean, (sq / n - mean * mean).max(0.0).sqrt())
}

/// JSON-safe float (serde_json turns non-finite into null; keep it explicit).
pub fn jf(v: f64) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        json!(v.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_zero_error() {
        let a = [1.0f32, -2.0, 3.5];
        let d = diff(&a, &a);
        assert_eq!(d.rel_l2, 0.0);
        assert!((d.cosine - 1.0).abs() < 1e-12);
        assert!(d.within(0.0));
    }

    #[test]
    fn zeros_vs_signal_is_total_failure() {
        let d = diff(&[0.0, 0.0], &[1.0, 1.0]);
        assert!((d.rel_l2 - 1.0).abs() < 1e-12);
        assert!(!d.within(0.5));
    }

    #[test]
    fn nan_fails() {
        let d = diff(&[f32::NAN, 1.0], &[1.0, 1.0]);
        assert_eq!(d.non_finite, 1);
        assert!(!d.within(10.0));
    }

    #[test]
    fn psnr_of_small_noise() {
        let r = vec![0.0f32; 1000];
        let a = vec![0.02f32; 1000];
        // mse = 4e-4, range 2 → 10*log10(4/4e-4) = 40 dB
        assert!((psnr(&a, &r, 2.0) - 40.0).abs() < 1e-3);
    }
}
