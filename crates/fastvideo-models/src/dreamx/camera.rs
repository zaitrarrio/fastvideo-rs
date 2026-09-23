//! DreamX PRoPE camera conditioning (action → viewmats / K).
//! Ported from FastVideo `pipelines/basic/dreamx_world/camera_conditioning.py`.
//!
//! Builds the SE(3) + intrinsics tensors that PRoPE attention consumes. Full
//! PRoPE QK injection lives in the DiT graph once control-adapter weights load.

const TRANSLATION_BASE: f32 = 1.0;
const ROTATION_BASE: f32 = 10.0;

fn action_to_motion(c: char) -> Option<&'static str> {
    match c {
        'w' => Some("forward"),
        'a' => Some("left"),
        'd' => Some("right"),
        's' => Some("backward"),
        'j' => Some("left_rot"),
        'l' => Some("right_rot"),
        'i' => Some("up_rot"),
        'k' => Some("down_rot"),
        _ => None,
    }
}

fn translation_step(
    motion: &str,
    yaw_deg: f32,
    pitch_deg: f32,
    value: f32,
    duration: usize,
) -> [f32; 3] {
    let duration = duration.max(1) as f32;
    match motion {
        "forward" | "backward" => {
            let yaw = yaw_deg.to_radians();
            let pitch = pitch_deg.to_radians();
            let forward = [
                -yaw.sin() * pitch.cos(),
                pitch.sin(),
                yaw.cos() * pitch.cos(),
            ];
            let dir = if motion == "forward" { 1.0 } else { -1.0 };
            [
                forward[0] * value * dir / duration,
                forward[1] * value * dir / duration,
                forward[2] * value * dir / duration,
            ]
        }
        "left" | "right" => {
            let yaw = yaw_deg.to_radians();
            let right = [yaw.cos(), 0.0, yaw.sin()];
            let dir = if motion == "left" { -1.0 } else { 1.0 };
            [
                right[0] * value * dir / duration,
                0.0,
                right[2] * value * dir / duration,
            ]
        }
        _ => [0.0, 0.0, 0.0],
    }
}

fn rotation_step(motion: &str, value: f32, duration: usize) -> [f32; 3] {
    if !motion.ends_with("_rot") && !motion.ends_with("rot") {
        return [0.0, 0.0, 0.0];
    }
    let duration = duration.max(1) as f32;
    let mut r = [0f32; 3];
    if motion.starts_with("left") {
        r[1] = value / duration;
    } else if motion.starts_with("right") {
        r[1] = -value / duration;
    } else if motion.starts_with("up") {
        r[0] = -value / duration;
    } else if motion.starts_with("down") {
        r[0] = value / duration;
    }
    r
}

fn euler_to_quat(pitch: f32, yaw: f32, roll: f32) -> [f32; 4] {
    let (pitch, yaw, roll) = (pitch.to_radians(), yaw.to_radians(), roll.to_radians());
    let cy = (yaw * 0.5).cos();
    let sy = (yaw * 0.5).sin();
    let cp = (pitch * 0.5).cos();
    let sp = (pitch * 0.5).sin();
    let cr = (roll * 0.5).cos();
    let sr = (roll * 0.5).sin();
    [
        cy * cp * cr + sy * sp * sr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
        cy * cp * sr - sy * sp * cr,
    ]
}

fn quat_to_rot(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [qw, qx, qy, qz] = q;
    [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qw * qz),
            2.0 * (qx * qz + qw * qy),
        ],
        [
            2.0 * (qx * qy + qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qw * qx),
        ],
        [
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]
}

fn invert_se3(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0f32; 4]; 4];
    // R^T
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
    }
    // -R^T t
    for i in 0..3 {
        out[i][3] = -(out[i][0] * m[0][3] + out[i][1] * m[1][3] + out[i][2] * m[2][3]);
    }
    out[3][3] = 1.0;
    out
}

/// PRoPE camera pack: viewmats `[T,4,4]` flat row-major, K `[T,3,3]` flat.
#[derive(Debug, Clone, PartialEq)]
pub struct DreamXCameraCondition {
    pub viewmats: Vec<f32>,
    pub intrinsics: Vec<f32>,
    pub num_latent_frames: usize,
}

impl DreamXCameraCondition {
    pub fn viewmat(&self, t: usize) -> [[f32; 4]; 4] {
        let o = t * 16;
        let mut m = [[0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = self.viewmats[o + i * 4 + j];
            }
        }
        m
    }
}

fn lerp_pose(
    a_t: &[f32; 3],
    a_r: &[f32; 3],
    b_t: &[f32; 3],
    b_r: &[f32; 3],
    alpha: f32,
) -> ([f32; 3], [f32; 3]) {
    let t = [
        a_t[0] + (b_t[0] - a_t[0]) * alpha,
        a_t[1] + (b_t[1] - a_t[1]) * alpha,
        a_t[2] + (b_t[2] - a_t[2]) * alpha,
    ];
    let r = [
        a_r[0] + (b_r[0] - a_r[0]) * alpha,
        a_r[1] + (b_r[1] - a_r[1]) * alpha,
        a_r[2] + (b_r[2] - a_r[2]) * alpha,
    ];
    (t, r)
}

/// Build DreamX camera tensors from action tokens (`w,d,w`) + speeds.
pub fn build_dreamx_camera_condition(
    action_seq: &[String],
    action_speed_list: &[f32],
    num_frames: usize,
) -> Result<DreamXCameraCondition, String> {
    if action_seq.is_empty() {
        return Err("dreamx action_seq empty".into());
    }
    if action_seq.len() != action_speed_list.len() {
        return Err("action_seq and action_speed_list length mismatch".into());
    }
    let duration = (num_frames as f32 / action_seq.len() as f32).ceil() as usize;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut rotations: Vec<[f32; 3]> = Vec::new();
    let mut cur_pos = [0f32; 3];
    let mut cur_rot = [0f32; 3];

    for (action_id, &speed) in action_seq.iter().zip(action_speed_list.iter()) {
        let mut t_step = [0f32; 3];
        let mut r_step = [0f32; 3];
        for ch in action_id.chars() {
            let Some(motion) = action_to_motion(ch.to_ascii_lowercase()) else {
                continue;
            };
            let ts = translation_step(
                motion,
                cur_rot[1],
                cur_rot[0],
                speed * TRANSLATION_BASE,
                duration,
            );
            let rs = rotation_step(motion, speed * ROTATION_BASE, duration);
            for i in 0..3 {
                t_step[i] += ts[i];
                r_step[i] += rs[i];
            }
        }
        for index in 1..=duration {
            let p = [
                cur_pos[0] + t_step[0] * index as f32,
                cur_pos[1] + t_step[1] * index as f32,
                cur_pos[2] + t_step[2] * index as f32,
            ];
            let r = [
                cur_rot[0] + r_step[0] * index as f32,
                cur_rot[1] + r_step[1] * index as f32,
                cur_rot[2] + r_step[2] * index as f32,
            ];
            positions.push(p);
            rotations.push(r);
        }
        if let (Some(lp), Some(lr)) = (positions.last(), rotations.last()) {
            cur_pos = *lp;
            cur_rot = *lr;
        }
    }

    // Truncate / pad to num_frames camera samples then subsample to latent count.
    while positions.len() < num_frames {
        positions.push(cur_pos);
        rotations.push(cur_rot);
    }
    positions.truncate(num_frames);
    rotations.truncate(num_frames);

    let latent_frame_count = 1 + (positions.len().saturating_sub(1)) / 4;
    let mut latent_pos = Vec::with_capacity(latent_frame_count);
    let mut latent_rot = Vec::with_capacity(latent_frame_count);
    for i in 0..latent_frame_count {
        let alpha = if latent_frame_count == 1 {
            0.0
        } else {
            i as f32 / (latent_frame_count - 1) as f32
        };
        let src = alpha * (positions.len() - 1) as f32;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(positions.len() - 1);
        let frac = src - i0 as f32;
        let (t, r) = lerp_pose(
            &positions[i0],
            &rotations[i0],
            &positions[i1],
            &rotations[i1],
            frac,
        );
        latent_pos.push(t);
        latent_rot.push(r);
    }

    // Absolute w2c then relative c2w → viewmats.
    let mut abs_w2c: Vec<[[f32; 4]; 4]> = Vec::with_capacity(latent_frame_count);
    for (pos, rot) in latent_pos.iter().zip(latent_rot.iter()) {
        let r = quat_to_rot(euler_to_quat(rot[0], rot[1], rot[2]));
        let t = [
            -(r[0][0] * pos[0] + r[0][1] * pos[1] + r[0][2] * pos[2]),
            -(r[1][0] * pos[0] + r[1][1] * pos[1] + r[1][2] * pos[2]),
            -(r[2][0] * pos[0] + r[2][1] * pos[1] + r[2][2] * pos[2]),
        ];
        abs_w2c.push([
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]);
    }

    let abs_c2w: Vec<_> = abs_w2c.iter().map(invert_se3).collect();
    let target = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let abs2rel = mat4_mul(&target, &abs_w2c[0]);
    let mut c2ws = vec![target];
    for c2w in abs_c2w.iter().skip(1) {
        c2ws.push(mat4_mul(&abs2rel, c2w));
    }
    let viewmats: Vec<[[f32; 4]; 4]> = c2ws.iter().map(invert_se3).collect();

    let mut view_flat = Vec::with_capacity(latent_frame_count * 16);
    for m in &viewmats {
        for row in m {
            view_flat.extend_from_slice(row);
        }
    }

    // DreamX-World-5B-Cam fixed normalized intrinsics.
    let fx = 969.6969696969696 / (960.0 * 2.0);
    let fy = 969.6969696969696 / (540.0 * 2.0);
    let mut k_flat = Vec::with_capacity(latent_frame_count * 9);
    for _ in 0..latent_frame_count {
        k_flat.extend_from_slice(&[fx, 0.0, 0.0, 0.0, fy, 0.0, 0.0, 0.0, 1.0]);
    }

    Ok(DreamXCameraCondition {
        viewmats: view_flat,
        intrinsics: k_flat,
        num_latent_frames: latent_frame_count,
    })
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_viewmats_for_default_actions() {
        let actions = vec!["w".into(), "d".into(), "w".into()];
        let speeds = vec![4.0, 2.0, 4.0];
        let c = build_dreamx_camera_condition(&actions, &speeds, 161).unwrap();
        assert!(c.num_latent_frames > 1);
        assert_eq!(c.viewmats.len(), c.num_latent_frames * 16);
        assert_eq!(c.intrinsics.len(), c.num_latent_frames * 9);
        // First viewmat near identity.
        let m = c.viewmat(0);
        assert!((m[0][0] - 1.0).abs() < 1e-3);
        assert!((m[3][3] - 1.0).abs() < 1e-5);
    }
}
