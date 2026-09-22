//! Cosmos 3-D RoPE (Diffusers `CosmosRotaryPosEmbed`).
//!
//! Returns interleaved `cos`/`sin` tables of shape `[seq, head_dim]` matching
//! Diffusers `apply_rotary_emb(..., use_real=True)`.

use super::config::CosmosTransformerConfig;

/// Cos/sin for a latent `[T,H,W]` grid after patchify.
pub fn rope_cos_sin(
    cfg: &CosmosTransformerConfig,
    frames: usize,
    height: usize,
    width: usize,
    fps: Option<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let head = cfg.attention_head_dim;
    let [p_t, p_h, p_w] = cfg.patch_size;
    let pe_t = frames / p_t;
    let pe_h = height / p_h;
    let pe_w = width / p_w;
    let seq = pe_t * pe_h * pe_w;

    let dim_h = head / 6 * 2;
    let dim_w = head / 6 * 2;
    let dim_t = head - dim_h - dim_w;

    let [rope_t, rope_h, rope_w] = cfg.rope_scale;
    let h_ntk = rope_h.powf(dim_h as f32 / (dim_h as f32 - 2.0));
    let w_ntk = rope_w.powf(dim_w as f32 / (dim_w as f32 - 2.0));
    let t_ntk = rope_t.powf(dim_t as f32 / (dim_t as f32 - 2.0));

    let h_theta = 10_000.0 * h_ntk;
    let w_theta = 10_000.0 * w_ntk;
    let t_theta = 10_000.0 * t_ntk;

    let freqs_half = |dim: usize, theta: f32| -> Vec<f32> {
        (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf((i as f32) / dim as f32))
            .collect()
    };
    let fh = freqs_half(dim_h, h_theta);
    let fw = freqs_half(dim_w, w_theta);
    let ft = freqs_half(dim_t, t_theta);

    let max_axis = cfg.max_size[0]
        .max(cfg.max_size[1])
        .max(cfg.max_size[2])
        / p_t.max(p_h).max(p_w);
    let _ = max_axis;

    let base_fps = 24.0f32;
    let mut cos = vec![0f32; seq * head];
    let mut sin = vec![0f32; seq * head];

    for ti in 0..pe_t {
        let t_pos = match fps {
            Some(f) if f > 0.0 => ti as f32 / f * base_fps,
            _ => ti as f32,
        };
        for yi in 0..pe_h {
            for xi in 0..pe_w {
                let mut args = Vec::with_capacity(head / 2);
                for &f in &ft {
                    args.push(t_pos * f);
                }
                for &f in &fh {
                    args.push(yi as f32 * f);
                }
                for &f in &fw {
                    args.push(xi as f32 * f);
                }
                // Diffusers: cat([emb_t, emb_h, emb_w] * 2) → full head_dim args.
                let half = args.clone();
                args.extend_from_slice(&half);
                debug_assert_eq!(args.len(), head);
                let cell = (ti * pe_h + yi) * pe_w + xi;
                for (i, arg) in args.iter().enumerate() {
                    cos[cell * head + i] = arg.cos();
                    sin[cell * head + i] = arg.sin();
                }
            }
        }
    }
    (cos, sin)
}

/// Apply real RoPE to `[B, Heads, S, D]` using `[S, D]` cos/sin (unbind last-2).
pub fn apply_rope_real(
    q: &[f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
) -> Vec<f32> {
    // use_real_unbind_dim=-2: rotate pairs (x0,x1) with (cos,sin) on even/odd.
    let mut out = vec![0f32; q.len()];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let base = ((b * heads + h) * seq + s) * dim;
                let rb = s * dim;
                for i in (0..dim).step_by(2) {
                    let x0 = q[base + i];
                    let x1 = q[base + i + 1];
                    let c = cos[rb + i];
                    let sn = sin[rb + i];
                    out[base + i] = x0 * c - x1 * sn;
                    out[base + i + 1] = x0 * sn + x1 * c;
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
    fn rope_len_matches_seq_head() {
        let cfg = CosmosTransformerConfig::tiny();
        // latent T=2,H=4,W=4 → pe 2×2×2 = 8
        let (c, s) = rope_cos_sin(&cfg, 2, 4, 4, Some(16.0));
        assert_eq!(c.len(), 8 * cfg.attention_head_dim);
        assert_eq!(s.len(), c.len());
    }
}
