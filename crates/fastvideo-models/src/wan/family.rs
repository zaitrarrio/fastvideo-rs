//! Family-specific Wan inference helpers: I2V concat, MoE routing, causal mask.

use super::config::WanVideoArchConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExpert {
    /// `transformer` — high noise, `t >= boundary`.
    HighNoise,
    /// `transformer_2` — low noise, `t < boundary`.
    LowNoise,
}

/// FastVideo denoising-stage routing: `boundary = boundary_ratio * num_train_timesteps`.
pub fn moe_expert(timestep: f64, boundary_ratio: f32, num_train_timesteps: i32) -> MoeExpert {
    let boundary = f64::from(boundary_ratio) * f64::from(num_train_timesteps);
    if timestep < boundary {
        MoeExpert::LowNoise
    } else {
        MoeExpert::HighNoise
    }
}

/// Wan I2V 36-channel concat: `[noisy 16 | mask 4 | cond 16]` along channels.
/// Layout matches Diffusers `WanImageToVideoPipeline` latent packing.
pub fn pack_i2v_channels(
    noisy: &[f32],
    mask: &[f32],
    cond: &[f32],
    frames: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let spatial = frames * height * width;
    if noisy.len() != 16 * spatial {
        return Err(format!("noisy want {} got {}", 16 * spatial, noisy.len()));
    }
    if mask.len() != 4 * spatial {
        return Err(format!("mask want {} got {}", 4 * spatial, mask.len()));
    }
    if cond.len() != 16 * spatial {
        return Err(format!("cond want {} got {}", 16 * spatial, cond.len()));
    }
    let mut out = vec![0.0f32; 36 * spatial];
    for s in 0..spatial {
        for c in 0..16 {
            out[c * spatial + s] = noisy[c * spatial + s];
        }
        for c in 0..4 {
            out[(16 + c) * spatial + s] = mask[c * spatial + s];
        }
        for c in 0..16 {
            out[(20 + c) * spatial + s] = cond[c * spatial + s];
        }
    }
    Ok(out)
}

/// First-frame I2V mask: 1 on frame 0, 0 elsewhere (per spatial location, 4 channels).
pub fn i2v_first_frame_mask(frames: usize, height: usize, width: usize) -> Vec<f32> {
    let spatial = height * width;
    let mut mask = vec![0.0f32; 4 * frames * spatial];
    for c in 0..4 {
        for s in 0..spatial {
            mask[c * frames * spatial + s] = 1.0;
        }
    }
    mask
}

/// Additive attention mask `[seq, seq]`: 0 allowed, `-1e9` blocked.
/// Tokens are time-major patches `(t, h, w)` with `patch_size`.
pub fn causal_temporal_mask(
    cfg: &WanVideoArchConfig,
    frames: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let pt = cfg.patch_size[0].max(1);
    let ph = cfg.patch_size[1].max(1);
    let pw = cfg.patch_size[2].max(1);
    let tf = frames / pt;
    let hf = height / ph;
    let wf = width / pw;
    let seq = tf * hf * wf;
    let mut mask = vec![0.0f32; seq * seq];
    let hw = hf * wf;
    for q in 0..seq {
        let tq = q / hw;
        for k in 0..seq {
            let tk = k / hw;
            let mut ok = tk <= tq;
            if cfg.local_attn_size > 0 {
                let window = cfg.local_attn_size as usize;
                if tq.saturating_sub(tk) > window {
                    ok = false;
                }
            }
            if cfg.sink_size > 0 && tk < cfg.sink_size {
                ok = true;
            }
            if !ok {
                mask[q * seq + k] = -1e9;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_boundary_matches_fastvideo_a14b() {
        // boundary_ratio=0.875, N=1000 → t>=875 high-noise expert
        assert_eq!(moe_expert(900.0, 0.875, 1000), MoeExpert::HighNoise);
        assert_eq!(moe_expert(875.0, 0.875, 1000), MoeExpert::HighNoise);
        assert_eq!(moe_expert(874.0, 0.875, 1000), MoeExpert::LowNoise);
        assert_eq!(moe_expert(12.0, 0.875, 1000), MoeExpert::LowNoise);
    }

    #[test]
    fn i2v_pack_is_36_channels() {
        let frames = 2;
        let h = 4;
        let w = 4;
        let spatial = frames * h * w;
        let noisy: Vec<f32> = (0..16 * spatial).map(|i| i as f32).collect();
        let mask = vec![1.0f32; 4 * spatial];
        let cond = vec![0.5f32; 16 * spatial];
        let packed = pack_i2v_channels(&noisy, &mask, &cond, frames, h, w).unwrap();
        assert_eq!(packed.len(), 36 * spatial);
        let first = i2v_first_frame_mask(frames, h, w);
        assert_eq!(first.len(), 4 * spatial);
        assert_eq!(first[0], 1.0);
        assert_eq!(first[h * w], 0.0);
        assert_eq!(packed[0], 0.0);
        assert_eq!(packed[16 * spatial], 1.0);
        assert_eq!(packed[20 * spatial], 0.5);
        let cfg = WanVideoArchConfig::wan_i2v_14b();
        assert_eq!(cfg.in_channels, packed.len() / spatial);
    }

    #[test]
    fn causal_mask_blocks_future_frames() {
        let mut cfg = WanVideoArchConfig::sf_wan_t2v_1_3b();
        cfg.local_attn_size = -1;
        let frames = 4;
        let h = 4;
        let w = 4;
        let mask = causal_temporal_mask(&cfg, frames, h, w);
        let tf = frames; // patch_size t=1
        let hw = (h / 2) * (w / 2);
        let seq = tf * hw;
        assert_eq!(mask.len(), seq * seq);
        // query in last frame can see first frame
        let q = (tf - 1) * hw;
        assert_eq!(mask[q * seq], 0.0);
        // query in first frame cannot see last frame
        let k = (tf - 1) * hw;
        assert!(mask[k] < -1e8);
    }

    #[test]
    fn local_window_and_sink() {
        let mut cfg = WanVideoArchConfig::sf_wan_t2v_1_3b();
        cfg.local_attn_size = 1;
        cfg.sink_size = 1;
        let mask = causal_temporal_mask(&cfg, 6, 4, 4);
        let hw = 2 * 2;
        let seq = 6 * hw;
        // t=5 attending t=3 is outside window=1 and not sink → blocked
        let q = 5 * hw;
        let k = 3 * hw;
        assert!(mask[q * seq + k] < -1e8);
        // sink frame 0 is always visible
        assert_eq!(mask[q * seq], 0.0);
        // t=5 attending t=4 is inside window
        assert_eq!(mask[q * seq + 4 * hw], 0.0);
    }
}
