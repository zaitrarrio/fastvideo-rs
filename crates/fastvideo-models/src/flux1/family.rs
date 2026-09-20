//! FLUX.1 helpers: Diffusers 2×2 latent pack, 3-axis RoPE ids, scheduler shift.

/// Diffusers `calculate_shift` (not Flux2 empirical μ).
///
/// `mu = image_seq_len * m + b` with
/// `m = (max_shift - base_shift) / (max_seq_len - base_seq_len)`.
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let seq = image_seq_len as f64;
    let base = base_seq_len as f64;
    let max = max_seq_len as f64;
    let m = (max_shift - base_shift) / (max - base);
    let b = base_shift - m * base;
    seq * m + b
}

/// Published Diffusers Flux.1 defaults (`base_seq=256`, `max_seq=4096`).
pub fn calculate_shift_flux1(image_seq_len: usize) -> f64 {
    calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)
}

/// Packed latent spatial size for a pixel image (`H, W`) and VAE scale.
/// Same 2×2 pack as Flux2 after the 8× VAE downsample.
pub fn packed_hw(height: usize, width: usize, vae_scale: usize) -> (usize, usize) {
    crate::flux2::packed_hw(height, width, vae_scale)
}

/// Diffusers `_pack_latents`: `[C, H, W]` → sequence `[H/2*W/2, C*4]`.
///
/// Pixel order is `(dy, dx)` inside each 2×2, channels outermost in the last
/// dim: `c*4 + dy*2 + dx`. Same 2×2 neighbourhood as Flux2's NCHW pack.
pub fn pack_latents_flux1(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
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
    let seq = oh * ow;
    let mut out = vec![0.0f32; seq * channels * 4];
    for i in 0..oh {
        for j in 0..ow {
            let s = i * ow + j;
            for c in 0..channels {
                let src = |y: usize, x: usize| (c * height + y) * width + x;
                let base = s * channels * 4 + c * 4;
                out[base] = input[src(i * 2, j * 2)];
                out[base + 1] = input[src(i * 2, j * 2 + 1)];
                out[base + 2] = input[src(i * 2 + 1, j * 2)];
                out[base + 3] = input[src(i * 2 + 1, j * 2 + 1)];
            }
        }
    }
    Ok(out)
}

/// Inverse of [`pack_latents_flux1`]: `[seq, C*4]` → `[C, H, W]`.
pub fn unpack_latents_flux1(
    input: &[f32],
    channels: usize,
    packed_h: usize,
    packed_w: usize,
) -> Result<Vec<f32>, String> {
    let seq = packed_h * packed_w;
    let want = seq * channels * 4;
    if input.len() != want {
        return Err(format!("unpack want {want} got {}", input.len()));
    }
    let height = packed_h * 2;
    let width = packed_w * 2;
    let mut out = vec![0.0f32; channels * height * width];
    for i in 0..packed_h {
        for j in 0..packed_w {
            let s = i * packed_w + j;
            for c in 0..channels {
                let src = s * channels * 4 + c * 4;
                let dst = |y: usize, x: usize| (c * height + y) * width + x;
                out[dst(i * 2, j * 2)] = input[src];
                out[dst(i * 2, j * 2 + 1)] = input[src + 1];
                out[dst(i * 2 + 1, j * 2)] = input[src + 2];
                out[dst(i * 2 + 1, j * 2 + 1)] = input[src + 3];
            }
        }
    }
    Ok(out)
}

/// FLUX.1 text ids: zeros `[seq, 3]`.
pub fn text_ids(seq: usize) -> Vec<f32> {
    vec![0.0f32; seq * 3]
}

/// FLUX.1 image ids: `[0, h, w]` over packed spatial (Diffusers `_prepare_latent_image_ids`).
pub fn image_ids(height: usize, width: usize) -> Vec<f32> {
    let seq = height * width;
    let mut ids = vec![0.0f32; seq * 3];
    let mut n = 0usize;
    for y in 0..height {
        for x in 0..width {
            ids[n * 3 + 1] = y as f32;
            ids[n * 3 + 2] = x as f32;
            n += 1;
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_at_published_anchors() {
        assert!((calculate_shift_flux1(256) - 0.5).abs() < 1e-12);
        assert!((calculate_shift_flux1(4096) - 1.15).abs() < 1e-12);
        let mid = calculate_shift_flux1(2176);
        assert!(mid > 0.5 && mid < 1.15);
    }

    #[test]
    fn pack_unpatchify_roundtrip() {
        let c = 2;
        let h = 4;
        let w = 4;
        let input: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let packed = pack_latents_flux1(&input, c, h, w).unwrap();
        assert_eq!(packed.len(), 4 * c * 4);
        let back = unpack_latents_flux1(&packed, c, 2, 2).unwrap();
        assert_eq!(back, input);
    }

    #[test]
    fn pack_matches_diffusers_pixel_order() {
        // C=1, H=2, W=2: sequence length 1, last dim = [00, 01, 10, 11].
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let packed = pack_latents_flux1(&input, 1, 2, 2).unwrap();
        assert_eq!(packed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn packed_hw_1024() {
        assert_eq!(packed_hw(1024, 1024, 8), (64, 64));
    }

    #[test]
    fn image_ids_are_hw() {
        let ids = image_ids(2, 3);
        assert_eq!(ids.len(), 6 * 3);
        assert_eq!(&ids[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&ids[3..6], &[0.0, 0.0, 1.0]);
        assert_eq!(&ids[9..12], &[0.0, 1.0, 0.0]);
    }
}
