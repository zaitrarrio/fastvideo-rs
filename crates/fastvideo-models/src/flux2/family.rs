//! Flux2 family helpers: 2×2 latent pack, text layer-stack, empirical μ.

/// Stack selected hidden-state layers the way Flux2 text postprocess does.
///
/// Upstream: `stack(layers, dim=1)` → `[B, L, S, D]` then
/// `permute(0, 2, 1, 3).reshape(B, S, L*D)`.
pub fn stack_hidden_layers(layers: &[&[f32]], seq: usize, hidden: usize) -> Result<Vec<f32>, String> {
    if layers.is_empty() {
        return Err("need at least one hidden-state layer".into());
    }
    let want = seq * hidden;
    for (i, layer) in layers.iter().enumerate() {
        if layer.len() != want {
            return Err(format!(
                "layer {i} want {want} values (seq={seq} hidden={hidden}), got {}",
                layer.len()
            ));
        }
    }
    let n_layers = layers.len();
    let mut out = vec![0.0f32; seq * n_layers * hidden];
    for s in 0..seq {
        for (li, layer) in layers.iter().enumerate() {
            let src = s * hidden;
            let dst = s * n_layers * hidden + li * hidden;
            out[dst..dst + hidden].copy_from_slice(&layer[src..src + hidden]);
        }
    }
    Ok(out)
}

/// Resolution-dependent μ for the Flux2 flow-match scheduler.
///
/// From Black Forest Labs `sampling.compute_empirical_mu` (FastVideo
/// `flux_2_timestep_preparation.py`).
pub fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let seq = image_seq_len as f64;
    let a1 = 8.73809524e-05;
    let b1 = 1.89833333;
    let a2 = 0.00016927;
    let b2 = 0.45666666;
    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }
    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * (num_steps as f64) + b
}

/// Diffusers exponential time-shift used when `use_dynamic_shifting` is on.
pub fn flux2_time_shift(t: f64, mu: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let exp_mu = mu.exp();
    exp_mu / (exp_mu + (1.0 / t - 1.0))
}

/// 2×2 pack: `[C, H, W]` → `[C*4, H/2, W/2]` (NCHW, batch stripped).
///
/// Pixel order matches Diffusers Flux pack: `(0,0), (0,1), (1,0), (1,1)`
/// stacked along channels.
pub fn pack_latents_2x2(input: &[f32], channels: usize, height: usize, width: usize) -> Result<Vec<f32>, String> {
    if height % 2 != 0 || width % 2 != 0 {
        return Err(format!("pack wants even H/W, got {height}x{width}"));
    }
    let spatial = height * width;
    if input.len() != channels * spatial {
        return Err(format!(
            "pack want {} got {}",
            channels * spatial,
            input.len()
        ));
    }
    let oh = height / 2;
    let ow = width / 2;
    let mut out = vec![0.0f32; channels * 4 * oh * ow];
    for c in 0..channels {
        for i in 0..oh {
            for j in 0..ow {
                let dst_base = |slot: usize| ((c * 4 + slot) * oh + i) * ow + j;
                let src = |y: usize, x: usize| (c * height + y) * width + x;
                out[dst_base(0)] = input[src(i * 2, j * 2)];
                out[dst_base(1)] = input[src(i * 2, j * 2 + 1)];
                out[dst_base(2)] = input[src(i * 2 + 1, j * 2)];
                out[dst_base(3)] = input[src(i * 2 + 1, j * 2 + 1)];
            }
        }
    }
    Ok(out)
}

/// Inverse of [`pack_latents_2x2`]: `[C*4, H/2, W/2]` → `[C, H, W]`.
pub fn unpatchify_2x2(input: &[f32], channels: usize, packed_h: usize, packed_w: usize) -> Result<Vec<f32>, String> {
    let want = channels * 4 * packed_h * packed_w;
    if input.len() != want {
        return Err(format!("unpatchify want {want} got {}", input.len()));
    }
    let height = packed_h * 2;
    let width = packed_w * 2;
    let mut out = vec![0.0f32; channels * height * width];
    for c in 0..channels {
        for i in 0..packed_h {
            for j in 0..packed_w {
                let src = |slot: usize| ((c * 4 + slot) * packed_h + i) * packed_w + j;
                let dst = |y: usize, x: usize| (c * height + y) * width + x;
                out[dst(i * 2, j * 2)] = input[src(0)];
                out[dst(i * 2, j * 2 + 1)] = input[src(1)];
                out[dst(i * 2 + 1, j * 2)] = input[src(2)];
                out[dst(i * 2 + 1, j * 2 + 1)] = input[src(3)];
            }
        }
    }
    Ok(out)
}

/// Packed latent spatial size for a pixel image (`H, W`) and VAE scale.
pub fn packed_hw(height: usize, width: usize, vae_scale: usize) -> (usize, usize) {
    let lh = (height / vae_scale.max(1)) / 2;
    let lw = (width / vae_scale.max(1)) / 2;
    (lh.max(1), lw.max(1))
}

/// Flux2 text token ids: `[1, 1, 1, token]` per sequence position (4-axis RoPE).
pub fn text_ids(seq: usize) -> Vec<f32> {
    let mut ids = vec![0.0f32; seq * 4];
    for i in 0..seq {
        ids[i * 4 + 3] = i as f32;
    }
    ids
}

/// Flux2 image token ids: `[frame, h, w, 0]` over packed spatial.
pub fn image_ids(frames: usize, height: usize, width: usize) -> Vec<f32> {
    let seq = frames * height * width;
    let mut ids = vec![0.0f32; seq * 4];
    let mut n = 0usize;
    for t in 0..frames {
        for y in 0..height {
            for x in 0..width {
                ids[n * 4] = t as f32;
                ids[n * 4 + 1] = y as f32;
                ids[n * 4 + 2] = x as f32;
                n += 1;
            }
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpatchify_roundtrip() {
        let c = 2;
        let h = 4;
        let w = 4;
        let input: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let packed = pack_latents_2x2(&input, c, h, w).unwrap();
        assert_eq!(packed.len(), c * 4 * 2 * 2);
        let back = unpatchify_2x2(&packed, c, 2, 2).unwrap();
        assert_eq!(back, input);
    }

    #[test]
    fn stack_layers_matches_upstream_reshape() {
        let seq = 3;
        let hidden = 2;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = stack_hidden_layers(&[&a, &b], seq, hidden).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 6.0, 50.0, 60.0]);
    }

    #[test]
    fn empirical_mu_long_seq_uses_second_line() {
        let mu = compute_empirical_mu(5000, 50);
        assert!((mu - (0.00016927 * 5000.0 + 0.45666666)).abs() < 1e-9);
    }

    #[test]
    fn packed_hw_1024() {
        assert_eq!(packed_hw(1024, 1024, 8), (64, 64));
    }

    #[test]
    fn time_shift_at_one_is_one() {
        assert!((flux2_time_shift(1.0, 0.5) - 1.0).abs() < 1e-12);
    }
}
