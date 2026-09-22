//! LingBot 3D RoPE helpers (axes_dims layout).

use super::config::LingBotTransformerConfig;

/// Cos/sin tables shaped `[seq, half]` where `half = sum(axes_dims)`.
pub fn rope_freqs(
    cfg: &LingBotTransformerConfig,
    t: usize,
    h: usize,
    w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let [dt, dh, dw] = cfg.axes_dims;
    let [pt, ph, pw] = cfg.patch_size;
    let pe_t = t / pt.max(1);
    let pe_h = h / ph.max(1);
    let pe_w = w / pw.max(1);
    let seq = pe_t * pe_h * pe_w;
    let half = dt + dh + dw;
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    let theta = cfg.rope_theta;
    for ti in 0..pe_t {
        for yi in 0..pe_h {
            for xi in 0..pe_w {
                let cell = (ti * pe_h + yi) * pe_w + xi;
                let mut o = 0usize;
                for (pos, dim) in [(ti, dt), (yi, dh), (xi, dw)] {
                    for i in 0..dim {
                        let freq =
                            (pos as f32) / theta.powf((2 * i) as f32 / (dim.max(1) as f32 * 2.0));
                        cos[cell * half + o] = freq.cos();
                        sin[cell * half + o] = freq.sin();
                        o += 1;
                    }
                }
            }
        }
    }
    (cos, sin)
}

/// Rotate pairs in `[B,H,S,D]` using `[S, D/2]` cos/sin (truncates if axes shorter).
pub fn apply_rope_real(
    q: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    let rope_half = if seq == 0 {
        0
    } else {
        cos.len() / seq
    };
    let n = half.min(rope_half);
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let base = ((b * heads + h) * seq + s) * head_dim;
                for i in 0..n {
                    let c = cos[s * rope_half + i];
                    let sn = sin[s * rope_half + i];
                    let x0 = q[base + 2 * i];
                    let x1 = q[base + 2 * i + 1];
                    q[base + 2 * i] = x0 * c - x1 * sn;
                    q[base + 2 * i + 1] = x0 * sn + x1 * c;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_len() {
        let cfg = LingBotTransformerConfig::tiny();
        let (cos, sin) = rope_freqs(&cfg, 2, 4, 4);
        let seq = (2 / 1) * (4 / 2) * (4 / 2);
        let half: usize = cfg.axes_dims.iter().sum();
        assert_eq!(cos.len(), seq * half);
        assert_eq!(sin.len(), cos.len());
    }
}
