//! GEN3C 3D cache: unproject + forward warp (host).
//! Ported from FastVideo `pipelines/basic/gen3c/cache_3d.py`.
//!
//! MoGe depth inference is an external weight/API hook; callers supply depth
//! (or use [`synthetic_depth`]) then this module builds warped RGB + masks.

use super::camera::{Mat3, Mat4};

fn inv3(m: &Mat3) -> Mat3 {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det.max(1e-12);
    [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]
}

fn inv4(m: &Mat4) -> Mat4 {
    // Gauss-Jordan for SE(3)-ish 4x4; general inverse.
    let mut a = *m;
    let mut out = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for col in 0..4 {
        let mut piv = col;
        for r in col..4 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return *m; // singular → identity-ish fallback
        }
        a.swap(col, piv);
        out.swap(col, piv);
        let diag = a[col][col];
        for j in 0..4 {
            a[col][j] /= diag;
            out[col][j] /= diag;
        }
        for r in 0..4 {
            if r == col {
                continue;
            }
            let f = a[r][col];
            for j in 0..4 {
                a[r][j] -= f * a[col][j];
                out[r][j] -= f * out[col][j];
            }
        }
    }
    out
}

fn mat3_vec(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat4_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    ]
}

/// Flat depth map `H*W` at constant `depth` (MoGe stand-in for offline tests).
pub fn synthetic_depth(height: usize, width: usize, depth: f32) -> Vec<f32> {
    vec![depth; height * width]
}

/// Unproject depth → world points. Layout: `points[y*W+x] = [X,Y,Z]`.
pub fn unproject_points(
    depth: &[f32],
    height: usize,
    width: usize,
    w2c: &Mat4,
    intrinsic: &Mat3,
) -> Vec<[f32; 3]> {
    let k_inv = inv3(intrinsic);
    let c2w = inv4(w2c);
    let mut out = vec![[0f32; 3]; height * width];
    for y in 0..height {
        for x in 0..width {
            let d = depth[y * width + x].clamp(0.0, 100.0);
            if d <= 0.0 {
                continue;
            }
            let cam = mat3_vec(&k_inv, [x as f32, y as f32, 1.0]);
            let cam = [cam[0] * d, cam[1] * d, cam[2] * d];
            out[y * width + x] = mat4_point(&c2w, cam);
        }
    }
    out
}

/// Project world points → pixel coords + depth. Returns `(u, v, z)` per pixel.
pub fn project_points(
    world: &[[f32; 3]],
    height: usize,
    width: usize,
    w2c: &Mat4,
    intrinsic: &Mat3,
) -> Vec<[f32; 3]> {
    let mut out = vec![[0f32; 3]; height * width];
    for y in 0..height {
        for x in 0..width {
            let p = world[y * width + x];
            let cam = mat4_point(w2c, p);
            let proj = mat3_vec(intrinsic, cam);
            let z = proj[2];
            let inv_z = 1.0 / (z + 1e-7);
            out[y * width + x] = [proj[0] * inv_z, proj[1] * inv_z, z];
        }
    }
    out
}

/// Forward-warp RGB (`CHW` in `[-1,1]`) + mask via bilinear splatting.
///
/// Returns `(warped_rgb CHW, mask HW)`.
pub fn forward_warp_rgb(
    rgb: &[f32],
    mask: Option<&[f32]>,
    depth: &[f32],
    height: usize,
    width: usize,
    src_w2c: &Mat4,
    tgt_w2c: &Mat4,
    src_k: &Mat3,
    tgt_k: &Mat3,
) -> (Vec<f32>, Vec<f32>) {
    let world = unproject_points(depth, height, width, src_w2c, src_k);
    let projected = project_points(&world, height, width, tgt_w2c, tgt_k);

    let mut warped = vec![0f32; 3 * height * width];
    let mut weight = vec![0f32; height * width];
    let spatial = height * width;

    for y in 0..height {
        for x in 0..width {
            let src = y * width + x;
            let m = mask.map(|mm| mm[src]).unwrap_or(1.0);
            if m < 0.5 {
                continue;
            }
            let [u, v, z] = projected[src];
            if z <= 0.0 {
                continue;
            }
            // Soft depth weight (closer → higher).
            let d_w = ((z + 1.0).ln() + 1e-7).recip();
            let u0 = u.floor() as i32;
            let v0 = v.floor() as i32;
            let fu = u - u0 as f32;
            let fv = v - v0 as f32;
            let corners = [
                (u0, v0, (1.0 - fu) * (1.0 - fv)),
                (u0 + 1, v0, fu * (1.0 - fv)),
                (u0, v0 + 1, (1.0 - fu) * fv),
                (u0 + 1, v0 + 1, fu * fv),
            ];
            for (cx, cy, w) in corners {
                if cx < 0 || cy < 0 || cx >= width as i32 || cy >= height as i32 {
                    continue;
                }
                let dst = cy as usize * width + cx as usize;
                let ww = w * m * d_w;
                weight[dst] += ww;
                for ch in 0..3 {
                    warped[ch * spatial + dst] += rgb[ch * spatial + src] * ww;
                }
            }
        }
    }

    let mut out_mask = vec![0f32; spatial];
    for i in 0..spatial {
        if weight[i] > 1e-8 {
            out_mask[i] = 1.0;
            for ch in 0..3 {
                warped[ch * spatial + i] /= weight[i];
            }
        } else {
            for ch in 0..3 {
                warped[ch * spatial + i] = -1.0;
            }
        }
    }
    (warped, out_mask)
}

/// Render a trajectory of warped frames from one source RGB+depth.
///
/// Returns `frames` of length `w2cs.len()`, each `(rgb CHW, mask HW)`.
pub fn render_trajectory(
    rgb: &[f32],
    depth: &[f32],
    height: usize,
    width: usize,
    src_w2c: &Mat4,
    src_k: &Mat3,
    tgt_w2cs: &[Mat4],
    tgt_ks: &[Mat3],
) -> Vec<(Vec<f32>, Vec<f32>)> {
    tgt_w2cs
        .iter()
        .zip(tgt_ks.iter())
        .map(|(tw, tk)| {
            forward_warp_rgb(rgb, None, depth, height, width, src_w2c, tw, src_k, tk)
        })
        .collect()
}

/// Pack warped RGB(+mask) into GEN3C buffer channels without VAE.
///
/// Each buffer is 32 channels: first 16 reserved for VAE latents (here we
/// fold RGB into ch 0..2 and zeros elsewhere), next 16 hold a broadcast mask.
/// With `frame_buffer_max` buffers the flat layout is
/// `[buf0 | buf1 | …]` at latent spatial size `lt*lh*lw`.
///
/// This is the host tensor path when MoGe/VAE warp encoding is unavailable;
/// call sites with a loaded VAE should replace the RGB stub with encoded
/// latents via [`pack_vae_buffers`].
pub fn pack_rgb_buffers(
    warps: &[(Vec<f32>, Vec<f32>)],
    src_h: usize,
    src_w: usize,
    lt: usize,
    lh: usize,
    lw: usize,
    frame_buffer_max: usize,
    channels_per_buffer: usize,
) -> Vec<f32> {
    let buf_ch = frame_buffer_max * channels_per_buffer;
    let spatial = lt * lh * lw;
    let mut out = vec![0f32; buf_ch * spatial];
    let half = channels_per_buffer / 2; // 16 warped + 16 mask typical
    for (bi, (rgb, mask)) in warps.iter().take(frame_buffer_max).enumerate() {
        let base = bi * channels_per_buffer * spatial;
        for ti in 0..lt {
            for y in 0..lh {
                for x in 0..lw {
                    let sy = ((y as f32 + 0.5) * src_h as f32 / lh as f32) as usize;
                    let sx = ((x as f32 + 0.5) * src_w as f32 / lw as f32) as usize;
                    let sy = sy.min(src_h - 1);
                    let sx = sx.min(src_w - 1);
                    let src = sy * src_w + sx;
                    let dst = ti * lh * lw + y * lw + x;
                    for ch in 0..3.min(half) {
                        out[base + ch * spatial + dst] = rgb[ch * src_h * src_w + src];
                    }
                    let m = mask[src];
                    for ch in 0..half.min(channels_per_buffer - half) {
                        out[base + (half + ch) * spatial + dst] = m;
                    }
                }
            }
        }
    }
    out
}

/// Pack VAE-encoded warp latents `(C,T,H,W)` + masks into buffer channels.
///
/// `encoded[b]` is length `latent_channels * lt * lh * lw`; `masks[b]` is
/// length `lt * lh * lw` (latent-space mask).
pub fn pack_vae_buffers(
    encoded: &[Vec<f32>],
    masks: &[Vec<f32>],
    latent_channels: usize,
    lt: usize,
    lh: usize,
    lw: usize,
    frame_buffer_max: usize,
    channels_per_buffer: usize,
) -> Vec<f32> {
    let spatial = lt * lh * lw;
    let buf_ch = frame_buffer_max * channels_per_buffer;
    let mut out = vec![0f32; buf_ch * spatial];
    let half = channels_per_buffer / 2;
    for (bi, enc) in encoded.iter().take(frame_buffer_max).enumerate() {
        let base = bi * channels_per_buffer * spatial;
        let n_c = latent_channels.min(half);
        for ch in 0..n_c {
            for j in 0..spatial {
                out[base + ch * spatial + j] = enc[ch * spatial + j];
            }
        }
        if let Some(mask) = masks.get(bi) {
            for ch in 0..half.min(channels_per_buffer - half) {
                for j in 0..spatial.min(mask.len()) {
                    out[base + (half + ch) * spatial + j] = mask[j];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen3c::camera::{default_intrinsics, identity4};

    #[test]
    fn identity_warp_keeps_center() {
        let h = 8;
        let w = 8;
        let depth = synthetic_depth(h, w, 1.0);
        let mut rgb = vec![0f32; 3 * h * w];
        // Bright center pixel.
        let cx = 4;
        let cy = 4;
        for ch in 0..3 {
            rgb[ch * h * w + cy * w + cx] = 1.0;
        }
        let eye = identity4();
        let k = default_intrinsics(h, w);
        let (warped, mask) = forward_warp_rgb(&rgb, None, &depth, h, w, &eye, &eye, &k, &k);
        assert!(mask[cy * w + cx] > 0.5);
        assert!(warped[cy * w + cx] > 0.5);
    }

    #[test]
    fn pack_rgb_buffer_layout() {
        let h = 16;
        let w = 16;
        let rgb = vec![0.25f32; 3 * h * w];
        let mask = vec![1.0f32; h * w];
        let packed = pack_rgb_buffers(&[(rgb, mask)], h, w, 2, 4, 4, 2, 8);
        assert_eq!(packed.len(), 2 * 8 * 2 * 4 * 4);
        assert!(packed.iter().any(|&v| v != 0.0));
    }
}
