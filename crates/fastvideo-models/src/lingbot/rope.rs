//! LingBot 3D RoPE (`LingBotVideoRotaryEmbedding` + `make_joint_position_ids`).
//!
//! The joint sequence is `[video; text]`. Video token `(t, y, x)` sits at
//! position `(text_len + 1 + t, y, x)`, text token `i` at `(i + 1, 0, 0)`.
//! Each axis `d` of `axes_dims` contributes `d / 2` complex frequencies
//! `theta^(-2j/d)` (f64, then the angle rounded to f32 as `.float()` does);
//! per token the three axes concatenate `[t | h | w]` into `head_dim / 2`
//! complex rotations applied to adjacent pairs `(x[2i], x[2i+1])`.

use super::config::LingBotTransformerConfig;

/// Positions `(t, h, w)` of the joint `[video; text]` sequence.
pub fn joint_positions(text_len: usize, gt: usize, gh: usize, gw: usize) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(gt * gh * gw + text_len);
    for t in 0..gt {
        for y in 0..gh {
            for x in 0..gw {
                out.push([(text_len + 1 + t) as u32, y as u32, x as u32]);
            }
        }
    }
    for i in 0..text_len {
        out.push([(i + 1) as u32, 0, 0]);
    }
    out
}

/// Per-axis inverse frequencies (f64), `d / 2` each.
pub fn axis_inv_freqs(cfg: &LingBotTransformerConfig) -> [Vec<f64>; 3] {
    let theta = f64::from(cfg.rope_theta);
    cfg.axes_dims.map(|d| {
        (0..d / 2)
            .map(|j| 1.0 / theta.powf((2 * j) as f64 / d as f64))
            .collect()
    })
}

/// Rotation angles `[S, head_dim / 2]` (f32) for `positions`.
pub fn rope_angles(cfg: &LingBotTransformerConfig, positions: &[[u32; 3]]) -> Vec<f32> {
    let inv = axis_inv_freqs(cfg);
    let half = cfg.rope_half();
    let mut out = vec![0f32; positions.len() * half];
    for (s, pos) in positions.iter().enumerate() {
        let mut o = s * half;
        for axis in 0..3 {
            let p = f64::from(pos[axis]);
            for &f in &inv[axis] {
                out[o] = (p * f) as f32;
                o += 1;
            }
        }
    }
    out
}

/// Interleaved cos/sin tables `[S, head_dim]`: `cos[s, 2i] = cos[s, 2i+1] =
/// cos(angle[s, i])`, the layout `apply_rotary_bshd` consumes.
pub fn rope_tables(cfg: &LingBotTransformerConfig, positions: &[[u32; 3]]) -> (Vec<f32>, Vec<f32>) {
    let angles = rope_angles(cfg, positions);
    let mut cos = vec![0f32; angles.len() * 2];
    let mut sin = vec![0f32; angles.len() * 2];
    for (i, &a) in angles.iter().enumerate() {
        let (s, c) = a.sin_cos();
        cos[2 * i] = c;
        cos[2 * i + 1] = c;
        sin[2 * i] = s;
        sin[2 * i + 1] = s;
    }
    (cos, sin)
}

/// Host reference of `apply_rotary_emb` on `[S, H, D]` (one batch).
pub fn apply_rope_pairs(x: &mut [f32], cos: &[f32], sin: &[f32], seq: usize, heads: usize, d: usize) {
    for s in 0..seq {
        for h in 0..heads {
            let base = (s * heads + h) * d;
            for i in 0..d / 2 {
                let (c, sn) = (cos[s * d + 2 * i], sin[s * d + 2 * i]);
                let (x0, x1) = (x[base + 2 * i], x[base + 2 * i + 1]);
                x[base + 2 * i] = x0 * c - x1 * sn;
                x[base + 2 * i + 1] = x0 * sn + x1 * c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_put_video_first_after_the_text() {
        let p = joint_positions(3, 2, 2, 2);
        assert_eq!(p.len(), 8 + 3);
        assert_eq!(p[0], [4, 0, 0]);
        assert_eq!(p[7], [5, 1, 1]);
        assert_eq!(&p[8..], &[[1, 0, 0], [2, 0, 0], [3, 0, 0]]);
    }

    #[test]
    fn moe_tables_have_head_dim_width() {
        let cfg = LingBotTransformerConfig::moe_30b();
        let pos = joint_positions(5, 2, 3, 4);
        let (cos, sin) = rope_tables(&cfg, &pos);
        assert_eq!(cos.len(), pos.len() * 128);
        assert_eq!(sin.len(), cos.len());
        // Axis split 16 / 24 / 24 complex pairs; the first h-axis frequency
        // of token (t=0, y=1, x=0) is angle 1 rad (theta^0).
        let tok = 4;
        assert_eq!(pos[tok], [6, 1, 0]);
        let a = rope_angles(&cfg, &pos);
        assert!((a[tok * 64 + 16] - 1.0).abs() < 1e-7);
        assert!((a[tok * 64] - 6.0).abs() < 1e-6);
        assert_eq!(a[tok * 64 + 40], 0.0);
    }

    #[test]
    fn inverse_frequencies_follow_theta() {
        let cfg = LingBotTransformerConfig::moe_30b();
        let inv = axis_inv_freqs(&cfg);
        assert_eq!([inv[0].len(), inv[1].len(), inv[2].len()], [16, 24, 24]);
        assert!((inv[0][1] - 256f64.powf(-2.0 / 32.0)).abs() < 1e-15);
        assert!((inv[1][23] - 256f64.powf(-46.0 / 48.0)).abs() < 1e-15);
    }

    #[test]
    fn rotation_preserves_pair_norm() {
        let cfg = LingBotTransformerConfig::tiny();
        let pos = joint_positions(2, 1, 2, 2);
        let (cos, sin) = rope_tables(&cfg, &pos);
        let (s, h, d) = (pos.len(), 2, cfg.head_dim());
        let mut x: Vec<f32> = (0..s * h * d).map(|i| (i as f32 * 0.3).sin()).collect();
        let before = x.clone();
        apply_rope_pairs(&mut x, &cos, &sin, s, h, d);
        for i in (0..x.len()).step_by(2) {
            let n0 = before[i].hypot(before[i + 1]);
            let n1 = x[i].hypot(x[i + 1]);
            assert!((n0 - n1).abs() < 1e-5);
        }
    }
}
