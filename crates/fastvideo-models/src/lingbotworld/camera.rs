//! LingBot-World camera injectors: relative c2ws + Plücker rays.
//! Ported from FastVideo `models/dits/lingbotworld/cam_utils.py`.

/// Flat row-major 4×4.
pub type Mat4 = [[f32; 4]; 4];

pub fn identity4() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn se3_inverse(t: &Mat4) -> Mat4 {
    let mut out = identity4();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = t[j][i];
        }
    }
    for i in 0..3 {
        out[i][3] = -(out[i][0] * t[0][3] + out[i][1] * t[1][3] + out[i][2] * t[2][3]);
    }
    out
}

fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

/// Relative poses vs first frame; optionally framewise deltas + translation normalize.
pub fn compute_relative_poses(c2ws: &[Mat4], framewise: bool, normalize_trans: bool) -> Vec<Mat4> {
    if c2ws.is_empty() {
        return Vec::new();
    }
    let ref_w2c = se3_inverse(&c2ws[0]);
    let mut relative: Vec<Mat4> = c2ws.iter().map(|c| mat4_mul(&ref_w2c, c)).collect();
    relative[0] = identity4();
    if framewise && relative.len() > 1 {
        let mut next = relative.clone();
        for i in 1..relative.len() {
            next[i] = mat4_mul(&se3_inverse(&relative[i - 1]), &relative[i]);
        }
        relative = next;
        relative[0] = identity4();
    }
    if normalize_trans {
        let max_norm = relative
            .iter()
            .map(|m| (m[0][3] * m[0][3] + m[1][3] * m[1][3] + m[2][3] * m[2][3]).sqrt())
            .fold(0f32, f32::max);
        if max_norm > 0.0 {
            for m in &mut relative {
                m[0][3] /= max_norm;
                m[1][3] /= max_norm;
                m[2][3] /= max_norm;
            }
        }
    }
    relative
}

/// Linear interpolate SE(3) poses (lerp rot rows + translation) to `tgt_len` frames.
pub fn interpolate_c2ws(src: &[Mat4], tgt_len: usize) -> Vec<Mat4> {
    if src.is_empty() {
        return vec![identity4(); tgt_len];
    }
    if src.len() == 1 || tgt_len <= 1 {
        return vec![src[0]; tgt_len];
    }
    let mut out = Vec::with_capacity(tgt_len);
    for i in 0..tgt_len {
        let alpha = i as f32 / (tgt_len - 1) as f32;
        let src_f = alpha * (src.len() - 1) as f32;
        let i0 = src_f.floor() as usize;
        let i1 = (i0 + 1).min(src.len() - 1);
        let t = src_f - i0 as f32;
        let a = &src[i0];
        let b = &src[i1];
        let mut m = identity4();
        for r in 0..3 {
            for c in 0..4 {
                m[r][c] = a[r][c] + (b[r][c] - a[r][c]) * t;
            }
        }
        out.push(m);
    }
    out
}

/// Intrinsics as `[fx, fy, cx, cy]` per frame.
pub fn get_plucker_embeddings(
    c2ws: &[Mat4],
    ks: &[[f32; 4]],
    height: usize,
    width: usize,
) -> Vec<f32> {
    // Output layout: [F, H, W, 6] row-major flat.
    let f = c2ws.len();
    let mut out = vec![0f32; f * height * width * 6];
    for (fi, c2w) in c2ws.iter().enumerate() {
        let [fx, fy, cx, cy] = ks[fi.min(ks.len() - 1)];
        for y in 0..height {
            for x in 0..width {
                let i = x as f32 + 0.5;
                let j = y as f32 + 0.5;
                let mut dir = [(i - cx) / fx, (j - cy) / fy, 1.0];
                let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-8);
                dir = [dir[0] / n, dir[1] / n, dir[2] / n];
                // rays_d = dir @ R^T
                let rays_d = [
                    dir[0] * c2w[0][0] + dir[1] * c2w[0][1] + dir[2] * c2w[0][2],
                    dir[0] * c2w[1][0] + dir[1] * c2w[1][1] + dir[2] * c2w[1][2],
                    dir[0] * c2w[2][0] + dir[1] * c2w[2][1] + dir[2] * c2w[2][2],
                ];
                // LingBot uses [rays_o, rays_d] (not classic Plücker cross).
                let rays_o = [c2w[0][3], c2w[1][3], c2w[2][3]];
                let o = ((fi * height + y) * width + x) * 6;
                out[o..o + 3].copy_from_slice(&rays_o);
                out[o + 3..o + 6].copy_from_slice(&rays_d);
            }
        }
    }
    out
}

/// Pixel-unshuffle style pack for cam injector: `[1, 6*S*S, F_lat, H/S, W/S]`.
pub fn pack_plucker_for_injector(
    plucker_fhw6: &[f32],
    num_frames: usize,
    height: usize,
    width: usize,
    spatial_scale: usize,
) -> Vec<f32> {
    let lh = height / spatial_scale;
    let lw = width / spatial_scale;
    let c_out = 6 * spatial_scale * spatial_scale;
    let mut out = vec![0f32; c_out * num_frames * lh * lw];
    for f in 0..num_frames {
        for y in 0..lh {
            for x in 0..lw {
                let mut ch = 0usize;
                for dy in 0..spatial_scale {
                    for dx in 0..spatial_scale {
                        let sy = y * spatial_scale + dy;
                        let sx = x * spatial_scale + dx;
                        let src = ((f * height + sy) * width + sx) * 6;
                        for k in 0..6 {
                            let dst = ((ch * num_frames + f) * lh + y) * lw + x;
                            out[dst] = plucker_fhw6[src + k];
                            ch += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Synthetic orbit c2ws for offline tests / default when no poses.npy.
pub fn synthetic_orbit_c2ws(num_frames: usize, radius: f32) -> Vec<Mat4> {
    let mut out = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let theta = std::f32::consts::TAU * (i as f32) / (num_frames.max(1) as f32);
        let mut m = identity4();
        m[0][0] = theta.cos();
        m[0][2] = theta.sin();
        m[2][0] = -theta.sin();
        m[2][2] = theta.cos();
        m[0][3] = radius * theta.sin();
        m[2][3] = radius * (1.0 - theta.cos());
        out.push(m);
    }
    out
}

/// Default pinhole `[fx,fy,cx,cy]` for canvas.
pub fn default_ks(num_frames: usize, height: usize, width: usize) -> Vec<[f32; 4]> {
    let fx = width as f32 * 0.5;
    let fy = height as f32 * 0.5;
    let cx = (width as f32) * 0.5;
    let cy = (height as f32) * 0.5;
    vec![[fx, fy, cx, cy]; num_frames]
}

/// End-to-end cam embedding for generate: relative c2ws → Plücker → injector pack.
pub fn prepare_camera_embedding(
    c2ws: &[Mat4],
    height: usize,
    width: usize,
    spatial_scale: usize,
) -> (Vec<f32>, usize) {
    let num_latent = 1 + (c2ws.len().saturating_sub(1)) / 4;
    let c2ws_lat = interpolate_c2ws(c2ws, num_latent);
    let c2ws_rel = compute_relative_poses(&c2ws_lat, true, true);
    let ks = default_ks(num_latent, height, width);
    let plucker = get_plucker_embeddings(&c2ws_rel, &ks, height, width);
    let packed = pack_plucker_for_injector(&plucker, num_latent, height, width, spatial_scale);
    (packed, c2ws.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_first_is_identity() {
        let c2ws = synthetic_orbit_c2ws(5, 0.1);
        let rel = compute_relative_poses(&c2ws, false, true);
        assert!((rel[0][0][0] - 1.0).abs() < 1e-5);
        assert!((rel[0][0][3]).abs() < 1e-5);
    }

    #[test]
    fn plucker_pack_shape() {
        let c2ws = synthetic_orbit_c2ws(9, 0.05);
        let (emb, n) = prepare_camera_embedding(&c2ws, 16, 16, 8);
        assert_eq!(n, 9);
        // channels = 6*8*8 = 384, latent frames = 3, h=w=2
        assert_eq!(emb.len(), 384 * 3 * 2 * 2);
    }
}
