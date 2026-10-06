//! Cosmos3 unified 3D M-RoPE (`get_3d_mrope_ids_*` of
//! `pipeline_cosmos3_omni.py`, `Cosmos3VLTextRotaryEmbedding`).
//!
//! Positions are `(t, h, w)` floats. Text tokens sit at `(i, i, i)` from 0;
//! the vision segment starts its time axis at `und_len + margin` (15000) and,
//! with fps modulation, frame `f` sits at `f / (fps / tcf) * (base_fps / tcf)
//! + offset` (just `f + offset` at the base 24 fps); `h`, `w` restart at 0.
//!
//! The rotary has `head_dim / 2` frequencies `theta^(-2j/d)`. Interleaved
//! M-RoPE takes frequency `j` from the T axis except `j ≡ 1 (mod 3)` for
//! `j < 3·sec_h` (H axis) and `j ≡ 2 (mod 3)` for `j < 3·sec_w` (W axis).
//! Tables are `[N, head_dim]` = `cat(freqs, freqs)`, applied rotate-half
//! style. Angles are the f32 product `inv_freq · position` the reference
//! computes with autocast off.

use super::config::Cosmos3TransformerConfig;

/// Text positions `0..n` on all three axes; returns `(ids, next_offset)`.
pub fn text_positions(n: usize) -> (Vec<[f32; 3]>, usize) {
    ((0..n).map(|i| [i as f32; 3]).collect(), n)
}

/// Vision positions of a `(t, h, w)` patch grid at `temporal_offset`.
pub fn vision_positions(
    cfg: &Cosmos3TransformerConfig,
    grid: [usize; 3],
    temporal_offset: f32,
    fps: Option<f64>,
    temporal_compression: usize,
) -> Vec<[f32; 3]> {
    let [gt, gh, gw] = grid;
    let modulate = cfg.enable_fps_modulation && fps.is_some() && gt > 1;
    let mut out = Vec::with_capacity(gt * gh * gw);
    for f in 0..gt {
        let t = if modulate {
            let tps = fps.unwrap() / temporal_compression as f64;
            let base_tps = cfg.base_fps / temporal_compression as f64;
            ((f as f32) / tps as f32 * base_tps as f32) + temporal_offset
        } else {
            (f as f32) + temporal_offset
        };
        for y in 0..gh {
            for x in 0..gw {
                let (hy, wx) = if cfg.reset_spatial_ids {
                    (y as f32, x as f32)
                } else {
                    (y as f32 + temporal_offset, x as f32 + temporal_offset)
                };
                out.push([t, hy, wx]);
            }
        }
    }
    out
}

/// `inv_freq` in f32 (`1 / theta^(arange(0, d, 2) / d)`).
pub fn inv_freq(cfg: &Cosmos3TransformerConfig) -> Vec<f32> {
    let d = cfg.head_dim;
    (0..d / 2)
        .map(|j| (1.0 / cfg.rope_theta.powf((2 * j) as f64 / d as f64)) as f32)
        .collect()
}

/// Which position axis frequency `j` reads (`apply_interleaved_mrope`).
pub fn axis_of(cfg: &Cosmos3TransformerConfig, j: usize) -> usize {
    let [_, sh, sw] = cfg.mrope_section;
    if j % 3 == 1 && j < 3 * sh {
        1
    } else if j % 3 == 2 && j < 3 * sw {
        2
    } else {
        0
    }
}

/// `(cos, sin)` tables `[N, head_dim]` for `positions`.
pub fn rope_tables(cfg: &Cosmos3TransformerConfig, positions: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>) {
    let inv = inv_freq(cfg);
    let half = inv.len();
    let d = 2 * half;
    let axes: Vec<usize> = (0..half).map(|j| axis_of(cfg, j)).collect();
    let mut cos = vec![0f32; positions.len() * d];
    let mut sin = vec![0f32; positions.len() * d];
    for (n, pos) in positions.iter().enumerate() {
        for j in 0..half {
            let angle = inv[j] * pos[axes[j]];
            let (s, c) = f64::from(angle).sin_cos();
            cos[n * d + j] = c as f32;
            cos[n * d + half + j] = c as f32;
            sin[n * d + j] = s as f32;
            sin[n * d + half + j] = s as f32;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_pattern_matches_reference_slices() {
        let cfg = Cosmos3TransformerConfig::super_64b();
        // H takes slice(1, 60, 3), W slice(2, 60, 3); the rest (and 60..64) T.
        let h: Vec<usize> = (0..64).filter(|&j| axis_of(&cfg, j) == 1).collect();
        let w: Vec<usize> = (0..64).filter(|&j| axis_of(&cfg, j) == 2).collect();
        assert_eq!(h, (1..60).step_by(3).collect::<Vec<_>>());
        assert_eq!(w, (2..60).step_by(3).collect::<Vec<_>>());
        assert_eq!((0..64).filter(|&j| axis_of(&cfg, j) == 0).count(), 24);
    }

    #[test]
    fn official_vision_positions_follow_the_text() {
        let cfg = Cosmos3TransformerConfig::super_64b();
        let (text, next) = text_positions(7);
        assert_eq!(text[3], [3.0, 3.0, 3.0]);
        let offset = (next + cfg.temporal_modality_margin) as f32;
        // 189 frames → 48 latent frames; 720×1280 → 45×80 → 23×40 patches.
        let pos = vision_positions(&cfg, [48, 23, 40], offset, Some(24.0), 4);
        assert_eq!(pos.len(), 48 * 23 * 40);
        assert_eq!(pos[0], [15007.0, 0.0, 0.0]);
        assert_eq!(pos[23 * 40 + 41], [15008.0, 1.0, 1.0]);
        // At 12 fps the time axis stretches by 2.
        let half = vision_positions(&cfg, [3, 1, 1], 0.0, Some(12.0), 4);
        for (p, want) in half.iter().zip([0.0f32, 2.0, 4.0]) {
            assert!((p[0] - want).abs() < 1e-5);
        }
    }

    #[test]
    fn text_tables_reduce_to_plain_rope() {
        let cfg = Cosmos3TransformerConfig::tiny();
        let (pos, _) = text_positions(3);
        let (cos, sin) = rope_tables(&cfg, &pos);
        let inv = inv_freq(&cfg);
        let d = cfg.head_dim;
        for (n, _) in pos.iter().enumerate() {
            for j in 0..d / 2 {
                let a = (inv[j] * n as f32) as f64;
                assert!((cos[n * d + j] as f64 - a.cos()).abs() < 1e-6);
                assert_eq!(cos[n * d + j], cos[n * d + d / 2 + j]);
                assert!((sin[n * d + j] as f64 - a.sin()).abs() < 1e-6);
            }
        }
    }
}
