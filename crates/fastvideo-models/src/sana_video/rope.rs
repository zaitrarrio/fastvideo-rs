//! SANA-Video rotary tables: Diffusers `WanRotaryPosEmbed` inside
//! `transformer_sana_video.py` — three axes (frame, height, width) with
//! `h = w = 2 * (head_dim // 6)` channels and the rest on the frame axis,
//! `get_1d_rotary_pos_embed(use_real=True, repeat_interleave_real=True)` in
//! float64, theta 1e4. The tables are interleaved (each angle repeated for
//! a channel pair), the layout `CudaTensor::apply_rotary_bshd` consumes.

/// `(t_dim, h_dim, w_dim)` for a head of `head_dim` channels.
pub fn axis_dims(head_dim: usize) -> (usize, usize, usize) {
    let hw = 2 * (head_dim / 6);
    (head_dim - 2 * hw, hw, hw)
}

/// `[seq, head_dim]` cos and sin for a `(frames, height, width)` patch grid,
/// tokens in `(f, h, w)` row-major order.
pub fn rope_tables(
    head_dim: usize,
    grid: (usize, usize, usize),
    max_seq_len: usize,
    theta: f64,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (f, h, w) = grid;
    if f > max_seq_len || h > max_seq_len || w > max_seq_len {
        return Err(format!(
            "SANA-Video rope: grid {grid:?} beyond rope_max_seq_len {max_seq_len}"
        ));
    }
    let (td, hd, wd) = axis_dims(head_dim);
    // Per-axis angle rows: angle[pos][k] for k < dim / 2.
    let axis = |dim: usize, n: usize| -> Vec<Vec<f64>> {
        let freqs: Vec<f64> = (0..dim / 2)
            .map(|k| 1.0 / theta.powf((2 * k) as f64 / dim as f64))
            .collect();
        (0..n)
            .map(|p| freqs.iter().map(|fr| p as f64 * fr).collect())
            .collect()
    };
    let (at, ah, aw) = (axis(td, f), axis(hd, h), axis(wd, w));
    let seq = f * h * w;
    let mut cos = Vec::with_capacity(seq * head_dim);
    let mut sin = Vec::with_capacity(seq * head_dim);
    for rt in &at {
        for rh in &ah {
            for rw in &aw {
                for row in [rt, rh, rw] {
                    for &a in row {
                        let (c, s) = (a.cos() as f32, a.sin() as f32);
                        cos.extend_from_slice(&[c, c]);
                        sin.extend_from_slice(&[s, s]);
                    }
                }
            }
        }
    }
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_the_2b_head() {
        assert_eq!(axis_dims(112), (40, 36, 36));
    }

    #[test]
    fn tables_are_interleaved_and_axis_ordered() {
        let (cos, sin) = rope_tables(12, (2, 3, 4), 64, 10_000.0).unwrap();
        // (t, h, w) = (4, 4, 4) channels for head 12.
        assert_eq!(cos.len(), 2 * 3 * 4 * 12);
        // Token (f=1, h=2, w=3): frame angle k=0 is 1 rad, height k=0 is 2 rad,
        // width k=0 is 3 rad, each repeated for its pair.
        let tok = (3 + 2) * 4 + 3;
        let row = &cos[tok * 12..(tok + 1) * 12];
        assert!((row[0] - 1f32.cos()).abs() < 1e-7 && row[0] == row[1]);
        assert!((row[4] - 2f32.cos()).abs() < 1e-7 && row[4] == row[5]);
        assert!((row[8] - 3f32.cos()).abs() < 1e-7 && row[8] == row[9]);
        // Second frequency of the frame axis: 1 / 1e4^(2/4) = 0.01.
        assert!((row[2] - 0.01f32.cos()).abs() < 1e-7);
        let srow = &sin[tok * 12..(tok + 1) * 12];
        assert!((srow[10] - (3.0f64 * 0.01).sin() as f32).abs() < 1e-7);
        assert!(rope_tables(12, (65, 1, 1), 64, 1e4).is_err());
    }
}
