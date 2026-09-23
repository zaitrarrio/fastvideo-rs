//! 3-D RoPE for HunyuanVideo 1.5 (`rope_axes_dim` = (t, h, w) on head_dim).

use super::config::Hunyuan15TransformerConfig;

/// Cos/sin tables `[seq, head_dim]` for a latent grid after patchify.
#[derive(Debug, Clone)]
pub struct Hunyuan15RopeTables {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub seq: usize,
    pub head_dim: usize,
}

impl Hunyuan15RopeTables {
    pub fn build(
        cfg: &Hunyuan15TransformerConfig,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Self, String> {
        let d = cfg.attention_head_dim;
        let [t_dim, h_dim, w_dim] = cfg.rope_axes_dim;
        if t_dim + h_dim + w_dim != d {
            return Err(format!(
                "rope axes {:?} must sum to head_dim {d}",
                cfg.rope_axes_dim
            ));
        }
        let seq = frames * height * width;
        let mut cos = vec![0f32; seq * d];
        let mut sin = vec![0f32; seq * d];
        let (t_c, t_s) = rotary_1d(t_dim, frames, cfg.rope_theta);
        let (h_c, h_s) = rotary_1d(h_dim, height, cfg.rope_theta);
        let (w_c, w_s) = rotary_1d(w_dim, width, cfg.rope_theta);
        let mut i = 0usize;
        for ft in 0..frames {
            for yh in 0..height {
                for xw in 0..width {
                    let base = i * d;
                    cos[base..base + t_dim].copy_from_slice(&t_c[ft * t_dim..(ft + 1) * t_dim]);
                    sin[base..base + t_dim].copy_from_slice(&t_s[ft * t_dim..(ft + 1) * t_dim]);
                    cos[base + t_dim..base + t_dim + h_dim]
                        .copy_from_slice(&h_c[yh * h_dim..(yh + 1) * h_dim]);
                    sin[base + t_dim..base + t_dim + h_dim]
                        .copy_from_slice(&h_s[yh * h_dim..(yh + 1) * h_dim]);
                    cos[base + t_dim + h_dim..base + d]
                        .copy_from_slice(&w_c[xw * w_dim..(xw + 1) * w_dim]);
                    sin[base + t_dim + h_dim..base + d]
                        .copy_from_slice(&w_s[xw * w_dim..(xw + 1) * w_dim]);
                    i += 1;
                }
            }
        }
        Ok(Self {
            cos,
            sin,
            seq,
            head_dim: d,
        })
    }
}

fn rotary_1d(dim: usize, seq: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0f32; seq * dim];
    let mut sin = vec![0f32; seq * dim];
    for p in 0..seq {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / dim as f64) as f32;
            let arg = p as f32 * freq;
            for o in [p * dim + 2 * i, p * dim + 2 * i + 1] {
                cos[o] = arg.cos();
                sin[o] = arg.sin();
            }
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hunyuan15::config::Hunyuan15TransformerConfig;

    #[test]
    fn rope_seq_is_grid() {
        let cfg = Hunyuan15TransformerConfig::tiny();
        let t = Hunyuan15RopeTables::build(&cfg, 2, 2, 2).unwrap();
        assert_eq!(t.seq, 8);
        assert_eq!(t.cos.len(), 8 * cfg.attention_head_dim);
    }
}
