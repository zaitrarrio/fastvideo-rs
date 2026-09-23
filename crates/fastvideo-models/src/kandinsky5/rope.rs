//! Kandinsky 5.0 3-D RoPE / 1-D text RoPE (Diffusers `Kandinsky5RoPE*`).
//!
//! Tables are complex 2×2 blocks `[…, head_dim/2, 2, 2]` applied as
//! `(rope * x_pairs).sum(-1)` — not Wan's interleaved cos/sin half-rotate.

use super::config::Kandinsky5TransformerConfig;

fn freqs(dim_half: usize, max_period: f64) -> Vec<f32> {
    (0..dim_half)
        .map(|i| (-max_period.ln() * (i as f64) / dim_half as f64).exp() as f32)
        .collect()
}

/// Build `[seq, head_dim/2, 2, 2]` rope for 1-D positions (text).
pub fn rope_1d(head_dim: usize, positions: &[usize], max_period: f64) -> Vec<f32> {
    let half = head_dim / 2;
    let freq = freqs(half, max_period);
    let mut out = vec![0f32; positions.len() * half * 4];
    for (pi, &p) in positions.iter().enumerate() {
        for i in 0..half {
            let arg = p as f32 * freq[i];
            let (c, s) = (arg.cos(), arg.sin());
            let base = (pi * half + i) * 4;
            // [[cos, -sin], [sin, cos]]
            out[base] = c;
            out[base + 1] = -s;
            out[base + 2] = s;
            out[base + 3] = c;
        }
    }
    out
}

/// Build `[B, T, H, W, head_dim/2, 2, 2]` visual rope for a latent grid.
pub fn rope_3d(
    cfg: &Kandinsky5TransformerConfig,
    batch: usize,
    frames: usize,
    height: usize,
    width: usize,
    max_period: f64,
) -> Vec<f32> {
    let [t_dim, h_dim, w_dim] = cfg.axes_dims;
    let head_dim = cfg.head_dim();
    let half = head_dim / 2;
    let ft = freqs(t_dim / 2, max_period);
    let fh = freqs(h_dim / 2, max_period);
    let fw = freqs(w_dim / 2, max_period);
    let mut out = vec![0f32; batch * frames * height * width * half * 4];
    for b in 0..batch {
        for t in 0..frames {
            for y in 0..height {
                for x in 0..width {
                    let mut args = Vec::with_capacity(half);
                    for i in 0..t_dim / 2 {
                        args.push(t as f32 * ft[i]);
                    }
                    for i in 0..h_dim / 2 {
                        args.push(y as f32 * fh[i]);
                    }
                    for i in 0..w_dim / 2 {
                        args.push(x as f32 * fw[i]);
                    }
                    let cell = ((b * frames + t) * height + y) * width + x;
                    for (i, arg) in args.iter().enumerate() {
                        let (c, s) = (arg.cos(), arg.sin());
                        let base = (cell * half + i) * 4;
                        out[base] = c;
                        out[base + 1] = -s;
                        out[base + 2] = s;
                        out[base + 3] = c;
                    }
                }
            }
        }
    }
    out
}

/// Apply Kandinsky complex RoPE to `[B, S, Heads, D]` (or flattened last two).
pub fn apply_rope(x: &[f32], rope: &[f32], seq: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let batch = x.len() / (seq * heads * head_dim);
    let mut out = vec![0f32; x.len()];
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                for p in 0..half {
                    let xi = (((b * seq + s) * heads + h) * head_dim) + p * 2;
                    let ri = (s * half + p) * 4; // rope broadcast over batch/heads
                    let (x0, x1) = (x[xi], x[xi + 1]);
                    // rope is [[c,-s],[s,c]] × [x0, x1]
                    out[xi] = rope[ri] * x0 + rope[ri + 1] * x1;
                    out[xi + 1] = rope[ri + 2] * x0 + rope[ri + 3] * x1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_1d_len() {
        let r = rope_1d(8, &[0, 1, 2], 10000.0);
        assert_eq!(r.len(), 3 * 4 * 4);
    }

    #[test]
    fn rope_3d_grid() {
        let cfg = Kandinsky5TransformerConfig::tiny();
        let r = rope_3d(&cfg, 1, 2, 2, 2, 10000.0);
        assert_eq!(r.len(), 1 * 2 * 2 * 2 * (cfg.head_dim() / 2) * 4);
    }
}
