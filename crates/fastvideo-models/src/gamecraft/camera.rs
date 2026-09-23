//! GameCraft CameraNet Plücker trajectory builder.
//! Ported from FastVideo `models/camera/trajectory.py`.

/// Action name → motion type.
pub fn resolve_action(action: &str) -> &'static str {
    match action.trim().to_ascii_lowercase().as_str() {
        "w" | "forward" => "forward",
        "s" | "backward" | "back" => "backward",
        "a" | "left" => "left",
        "d" | "right" => "right",
        "left_rot" => "left_rot",
        "right_rot" => "right_rot",
        "up_rot" => "up_rot",
        "down_rot" => "down_rot",
        other if !other.is_empty() => {
            // Fall through to known ids; unknown treated as forward for robustness.
            if matches!(
                other,
                "forward"
                    | "backward"
                    | "left"
                    | "right"
                    | "left_rot"
                    | "right_rot"
                    | "up_rot"
                    | "down_rot"
            ) {
                // unreachable due to match arms above
                "forward"
            } else {
                "forward"
            }
        }
        _ => "forward",
    }
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
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
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

fn generate_motion_segment(
    position: &mut [f32; 3],
    rotation: &mut [f32; 3],
    motion_type: &str,
    value: f32,
    duration: usize,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut positions = Vec::with_capacity(duration);
    let mut rotations = Vec::with_capacity(duration);
    let duration = duration.max(1);

    if matches!(motion_type, "forward" | "backward") {
        let yaw = rotation[1].to_radians();
        let pitch = rotation[0].to_radians();
        let forward = [
            -yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        ];
        let dir = if motion_type == "forward" { 1.0 } else { -1.0 };
        let step = [
            forward[0] * value * dir / duration as f32,
            forward[1] * value * dir / duration as f32,
            forward[2] * value * dir / duration as f32,
        ];
        for i in 1..=duration {
            let p = [
                position[0] + step[0] * i as f32,
                position[1] + step[1] * i as f32,
                position[2] + step[2] * i as f32,
            ];
            positions.push(p);
            rotations.push(*rotation);
        }
        *position = *positions.last().unwrap();
    } else if matches!(motion_type, "left" | "right") {
        let yaw = rotation[1].to_radians();
        let right = [yaw.cos(), 0.0, -yaw.sin()];
        let dir = if motion_type == "right" { -1.0 } else { 1.0 };
        let step = [
            right[0] * value * dir / duration as f32,
            0.0,
            right[2] * value * dir / duration as f32,
        ];
        for i in 1..=duration {
            let p = [
                position[0] + step[0] * i as f32,
                position[1] + step[1] * i as f32,
                position[2] + step[2] * i as f32,
            ];
            positions.push(p);
            rotations.push(*rotation);
        }
        *position = *positions.last().unwrap();
    } else if motion_type.ends_with("rot") {
        let mut total = [0f32; 3];
        if motion_type.starts_with("left") {
            total[0] = value;
        } else if motion_type.starts_with("right") {
            total[0] = -value;
        } else if motion_type.starts_with("up") {
            total[2] = -value;
        } else if motion_type.starts_with("down") {
            total[2] = value;
        }
        let step = [
            total[0] / duration as f32,
            total[1] / duration as f32,
            total[2] / duration as f32,
        ];
        for i in 1..=duration {
            positions.push(*position);
            let r = [
                rotation[0] + step[0] * i as f32,
                rotation[1] + step[1] * i as f32,
                rotation[2] + step[2] * i as f32,
            ];
            rotations.push(r);
        }
        *rotation = *rotations.last().unwrap();
    } else {
        for _ in 0..duration {
            positions.push(*position);
            rotations.push(*rotation);
        }
    }
    (positions, rotations)
}

fn ray_condition(
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    c2ws: &[[[f32; 4]; 4]],
    height: usize,
    width: usize,
) -> Vec<f32> {
    // Layout [F, 6, H, W] flat.
    let f = c2ws.len();
    let spatial = height * width;
    let mut packed = vec![0f32; f * 6 * spatial];
    for (fi, c2w) in c2ws.iter().enumerate() {
        for y in 0..height {
            for x in 0..width {
                let i = x as f32 + 0.5;
                let j = y as f32 + 0.5;
                let mut dir = [(i - cx) / fx, (j - cy) / fy, 1.0];
                let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
                    .sqrt()
                    .max(1e-8);
                dir = [dir[0] / n, dir[1] / n, dir[2] / n];
                let rays_d = [
                    dir[0] * c2w[0][0] + dir[1] * c2w[0][1] + dir[2] * c2w[0][2],
                    dir[0] * c2w[1][0] + dir[1] * c2w[1][1] + dir[2] * c2w[1][2],
                    dir[0] * c2w[2][0] + dir[1] * c2w[2][1] + dir[2] * c2w[2][2],
                ];
                let rays_o = [c2w[0][3], c2w[1][3], c2w[2][3]];
                let cross = [
                    rays_o[1] * rays_d[2] - rays_o[2] * rays_d[1],
                    rays_o[2] * rays_d[0] - rays_o[0] * rays_d[2],
                    rays_o[0] * rays_d[1] - rays_o[1] * rays_d[0],
                ];
                let vals = [
                    cross[0], cross[1], cross[2], rays_d[0], rays_d[1], rays_d[2],
                ];
                for (ch, v) in vals.iter().enumerate() {
                    let dst = ((fi * 6 + ch) * height + y) * width + x;
                    packed[dst] = *v;
                }
            }
        }
    }
    packed
}

/// Create Plücker camera states: flat `[1 * num_frames * 6 * H * W]` (= `[F,6,H,W]`).
pub fn create_camera_trajectory(
    action: &str,
    height: usize,
    width: usize,
    num_frames: usize,
    action_speed: f32,
) -> Vec<f32> {
    let motion = resolve_action(action);
    let mut position = [0f32; 3];
    let mut rotation = [0f32; 3];
    let (positions, rotations) = generate_motion_segment(
        &mut position,
        &mut rotation,
        motion,
        action_speed,
        num_frames,
    );

    // Intrinsics from GameCraft pose format.
    let (fx0, fy0, cx0, cy0) = (0.50505f32, 0.8979f32, 0.5f32, 0.5f32);
    let monst3r_w = cx0 * 2.0;
    let monst3r_h = cy0 * 2.0;
    let ratio_w = width as f32 / monst3r_w;
    let ratio_h = height as f32 / monst3r_h;
    let fx = fx0 * ratio_w;
    let fy = fy0 * ratio_h;
    let cx = cx0 * ratio_w;
    let cy = cy0 * ratio_h;

    // Build c2w relative poses (scale translation ×10 like FastVideo).
    let mut c2ws = Vec::with_capacity(num_frames);
    let target = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    // First frame identity.
    let mut abs_w2c: Vec<[[f32; 4]; 4]> = Vec::with_capacity(num_frames);
    // Identity first
    abs_w2c.push(target);
    for (pos, rot) in positions
        .iter()
        .zip(rotations.iter())
        .take(num_frames.saturating_sub(1))
    {
        let r = quat_to_rot(euler_to_quat(rot[0], rot[1], rot[2]));
        abs_w2c.push([
            [r[0][0], r[0][1], r[0][2], pos[0]],
            [r[1][0], r[1][1], r[1][2], pos[1]],
            [r[2][0], r[2][1], r[2][2], pos[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]);
    }
    while abs_w2c.len() < num_frames {
        abs_w2c.push(*abs_w2c.last().unwrap_or(&target));
    }

    // Invert to c2w then relative.
    fn inv(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut out = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = m[j][i];
            }
        }
        for i in 0..3 {
            out[i][3] = -(out[i][0] * m[0][3] + out[i][1] * m[1][3] + out[i][2] * m[2][3]);
        }
        out
    }
    fn mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut out = [[0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] =
                    a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
            }
        }
        out
    }

    let abs_c2w: Vec<_> = abs_w2c.iter().map(inv).collect();
    let abs2rel = mul(&target, &abs_w2c[0]);
    c2ws.push(target);
    for c2w in abs_c2w.iter().skip(1) {
        let mut rel = mul(&abs2rel, c2w);
        rel[0][3] *= 10.0;
        rel[1][3] *= 10.0;
        rel[2][3] *= 10.0;
        c2ws.push(rel);
    }
    while c2ws.len() < num_frames {
        c2ws.push(*c2ws.last().unwrap());
    }

    ray_condition(fx, fy, cx, cy, &c2ws, height, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_trajectory_shape() {
        let cam = create_camera_trajectory("forward", 32, 32, 9, 0.2);
        assert_eq!(cam.len(), 9 * 6 * 32 * 32);
        assert!(cam.iter().any(|&v| v.abs() > 1e-8));
    }

    #[test]
    fn action_aliases() {
        assert_eq!(resolve_action("w"), "forward");
        assert_eq!(resolve_action("left_rot"), "left_rot");
    }
}
