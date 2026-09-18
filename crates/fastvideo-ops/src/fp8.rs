//! OCP FP8 E4M3 conversion, as the reference for the CUDA kernels.
//!
//! NVRTC in this build has no `cuda_fp8.h` (same constraint that forced bf16
//! through `unsigned short` bit twiddling), so the device kernels open-code the
//! conversion. This module is the oracle they are written against: one place to
//! get the rounding right, exhaustively tested, mirrored line for line in
//! `kernels.rs`.
//!
//! E4M3: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits. Unlike IEEE
//! binary formats it has **no infinity**, its largest finite value is 448, and
//! `0x7F`/`0xFF` are NaN. Out-of-range inputs saturate to ±448 rather than
//! becoming NaN — the saturating behaviour NVIDIA calls SATFINITE, and the only
//! sane choice inside a denoiser, where a NaN poisons every later step.

/// Largest finite E4M3 magnitude. Activation and weight scales are chosen so a
/// tensor's amax maps onto this.
pub const E4M3_MAX: f32 = 448.0;

/// `0x7E` — the bit pattern for +448, i.e. E4M3's largest finite value.
const E4M3_SAT: u32 = 0x7E;

/// Round-to-nearest-even `f32` → E4M3 bits, saturating.
pub fn f32_to_e4m3(x: f32) -> u8 {
    let u = x.to_bits();
    let sign = ((u >> 24) & 0x80) as u32;
    let mag = u & 0x7FFF_FFFF;

    // NaN, infinity and anything at or past 448 all saturate.
    if mag >= 0x7F80_0000 || mag >= 0x43E0_0000 {
        return (sign | E4M3_SAT) as u8;
    }
    let exp = ((mag >> 23) as i32) - 127;
    let man = mag & 0x007F_FFFF;

    let out = if exp >= -6 {
        // Normal: keep 3 mantissa bits, round-to-nearest-even on the rest.
        let mut m = man >> 20;
        let rem = man & 0x000F_FFFF;
        let half = 1 << 19;
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        let mut e = (exp + 7) as u32;
        if m == 8 {
            m = 0;
            e += 1;
        }
        if e > 15 || (e == 15 && m >= 7) {
            return (sign | E4M3_SAT) as u8;
        }
        (e << 3) | m
    } else {
        // Subnormal: the value is m * 2^-9 for m in 0..8, so shift the implicit
        // one back in and round at the new position.
        let shift = 20 + (-6 - exp) as u32;
        if shift > 31 {
            return sign as u8;
        }
        let full = (1u32 << 23) | man;
        let mut m = full >> shift;
        let rem = full & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        // Rounding up out of the subnormal range lands on the smallest normal.
        m
    };
    (sign | out) as u8
}

/// E4M3 bits → `f32`. NaN codes decode to NaN; there is no infinity to handle.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as u32;
    if e == 0 {
        // Subnormal (and zero): m * 2^-9.
        sign * (m as f32) * (1.0 / 512.0)
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        let frac = 1.0 + (m as f32) / 8.0;
        sign * frac * exp2i(e - 7)
    }
}

/// `2^n` without `powi`, so the reference has no libm dependency to disagree on.
fn exp2i(n: i32) -> f32 {
    // E4M3 exponents live in [-6, 8]; f32 represents every one of these exactly.
    f32::from_bits(((n + 127) as u32) << 23)
}

/// The scale that maps a tensor's `amax` onto E4M3's range, and its reciprocal.
///
/// Returns `(scale, inv_scale)` where the quantized value is `x * inv_scale` and
/// dequantization is `q * scale`. An all-zero tensor yields a scale of 1 rather
/// than 0, so a dead channel cannot produce NaN downstream.
pub fn scale_for_amax(amax: f32) -> (f32, f32) {
    if !(amax > 0.0) || !amax.is_finite() {
        return (1.0, 1.0);
    }
    let scale = amax / E4M3_MAX;
    (scale, 1.0 / scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every E4M3 code that is not NaN must survive decode→encode unchanged.
    /// This is the property that catches an off-by-one in either direction, and
    /// with only 256 codes it can be checked exhaustively rather than sampled.
    #[test]
    fn every_code_round_trips() {
        for bits in 0u16..256 {
            let b = bits as u8;
            if (b & 0x7F) == 0x7F {
                continue; // NaN
            }
            let v = e4m3_to_f32(b);
            let back = f32_to_e4m3(v);
            // -0 and +0 both encode as their own sign; compare bit patterns.
            assert_eq!(back, b, "code {b:#04x} decoded to {v} and re-encoded to {back:#04x}");
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(f32_to_e4m3(0.0), 0x00);
        assert_eq!(f32_to_e4m3(1.0), 0x38);
        assert_eq!(f32_to_e4m3(2.0), 0x40);
        assert_eq!(f32_to_e4m3(0.5), 0x30);
        assert_eq!(f32_to_e4m3(-1.0), 0xB8);
        assert_eq!(f32_to_e4m3(448.0), 0x7E);
        assert_eq!(f32_to_e4m3(2f32.powi(-6)), 0x08); // smallest normal
        assert_eq!(f32_to_e4m3(2f32.powi(-9)), 0x01); // smallest subnormal
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
    }

    /// Out-of-range input must clamp, never become NaN: one NaN in a denoiser
    /// trajectory poisons every remaining step.
    #[test]
    fn overflow_saturates_and_never_produces_nan() {
        for x in [449.0, 1.0e4, 3.4e38, f32::INFINITY, f32::MAX] {
            assert_eq!(f32_to_e4m3(x), 0x7E, "{x} should saturate to +448");
            assert_eq!(f32_to_e4m3(-x), 0xFE, "{} should saturate to -448", -x);
        }
        // A NaN input still must not yield a NaN code.
        assert_eq!(f32_to_e4m3(f32::NAN) & 0x7F, 0x7E);
    }

    /// Underflow goes to zero rather than wrapping to a large magnitude.
    #[test]
    fn underflow_goes_to_zero() {
        for x in [1.0e-12f32, 2f32.powi(-20), f32::MIN_POSITIVE] {
            assert_eq!(f32_to_e4m3(x) & 0x7F, 0x00, "{x} should flush to zero");
        }
    }

    /// Rounding is to nearest-even, not truncation — truncation would bias every
    /// quantized weight toward zero and show up as a systematic amplitude loss.
    #[test]
    fn rounds_to_nearest_even_not_toward_zero() {
        // Between 1.0 (0x38) and 1.125 (0x39) the midpoint is 1.0625; ties go to
        // the even mantissa, which is 1.0.
        assert_eq!(f32_to_e4m3(1.0625), 0x38);
        assert_eq!(f32_to_e4m3(1.0626), 0x39);
        // Between 1.125 (0x39) and 1.25 (0x3A) the tie goes up, to even.
        assert_eq!(f32_to_e4m3(1.1875), 0x3A);
        // Truncation would give 0x38 for all three.
    }

    #[test]
    fn quantize_dequantize_keeps_a_tensor_within_e4m3_resolution() {
        // E4M3 carries 3 mantissa bits, so relative error is bounded by 2^-4.
        let xs: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.37).sin() * 12.5).collect();
        let amax = xs.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let (scale, inv) = scale_for_amax(amax);
        let mut worst = 0.0f32;
        for &x in &xs {
            let round = e4m3_to_f32(f32_to_e4m3(x * inv)) * scale;
            worst = worst.max((round - x).abs() / amax);
        }
        assert!(worst < 0.0625, "worst relative error {worst} exceeds 2^-4");
    }

    #[test]
    fn scale_of_an_all_zero_tensor_is_finite() {
        let (s, i) = scale_for_amax(0.0);
        assert_eq!((s, i), (1.0, 1.0));
        assert!(scale_for_amax(f32::NAN).0.is_finite());
    }
}
