//! HY-WorldPlay pose string → viewmats / action_in labels.
//! Ported from FastVideo `models/dits/hyworld/pose.py`.
//!
//! SigLIP vision encode is an external weight hook; this module emits the
//! `action_in` integer labels + camera tensors the DiT consumes.

use super::trajectory::{generate_camera_trajectory_local, Mat4, Motion};

pub const DEFAULT_FORWARD_SPEED: f32 = 0.08;
pub const DEFAULT_YAW_SPEED: f32 = 3.0_f32.to_radians();
pub const DEFAULT_PITCH_SPEED: f32 = 3.0_f32.to_radians();

/// Default 1920×1080 intrinsic before normalization.
pub const DEFAULT_INTRINSIC: [[f32; 3]; 3] = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
];

/// Parse `"w-31"` / `"w-3, right-0.5, d-4"` into per-step motions.
pub fn parse_pose_string(
    pose_string: &str,
    forward_speed: f32,
    yaw_speed: f32,
    pitch_speed: f32,
) -> Result<Vec<Motion>, String> {
    let mut motions = Vec::new();
    for cmd in pose_string.split(',') {
        let cmd = cmd.trim();
        if cmd.is_empty() {
            continue;
        }
        let parts: Vec<_> = cmd.split('-').collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid pose command `{cmd}` (expected action-duration)"
            ));
        }
        let action = parts[0].trim().to_ascii_lowercase();
        let duration: f32 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("invalid duration in `{cmd}`"))?;
        let num_frames = duration as usize;
        for _ in 0..num_frames {
            let m = match action.as_str() {
                "w" => Motion {
                    forward: forward_speed,
                    ..Default::default()
                },
                "s" => Motion {
                    forward: -forward_speed,
                    ..Default::default()
                },
                "a" => Motion {
                    right: -forward_speed,
                    ..Default::default()
                },
                "d" => Motion {
                    right: forward_speed,
                    ..Default::default()
                },
                "up" => Motion {
                    pitch: pitch_speed,
                    ..Default::default()
                },
                "down" => Motion {
                    pitch: -pitch_speed,
                    ..Default::default()
                },
                "left" => Motion {
                    yaw: -yaw_speed,
                    ..Default::default()
                },
                "right" => Motion {
                    yaw: yaw_speed,
                    ..Default::default()
                },
                other => {
                    return Err(format!(
                        "unknown action `{other}` (w/s/a/d/up/down/left/right)"
                    ))
                }
            };
            motions.push(m);
        }
    }
    Ok(motions)
}

fn inv4(m: &Mat4) -> Mat4 {
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

fn mul4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

fn one_hot_to_label(row: [i32; 4]) -> i32 {
    match row {
        [0, 0, 0, 0] => 0,
        [1, 0, 0, 0] => 1,
        [0, 1, 0, 0] => 2,
        [0, 0, 1, 0] => 3,
        [0, 0, 0, 1] => 4,
        [1, 0, 1, 0] => 5,
        [1, 0, 0, 1] => 6,
        [0, 1, 1, 0] => 7,
        [0, 1, 0, 1] => 8,
        _ => 0,
    }
}

/// Packed HY-World conditioning tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct HyWorldPoseInput {
    /// World-to-camera `[T*16]` row-major.
    pub viewmats: Vec<f32>,
    /// Normalized intrinsics `[T*9]`.
    pub intrinsics: Vec<f32>,
    /// `action_in` labels `[T]` (`trans*9 + rotate`).
    pub action_labels: Vec<i32>,
    pub latent_num: usize,
}

impl HyWorldPoseInput {
    /// Zero SigLIP placeholder tokens (`[1, N, D]`) for load-hook wiring.
    pub fn zeros_siglip_tokens(num_tokens: usize, dim: usize) -> Vec<f32> {
        vec![0f32; num_tokens * dim]
    }
}

/// Convert pose string → viewmats / K / action_in for `latent_num` latents.
pub fn pose_to_input(pose_string: &str, latent_num: usize) -> Result<HyWorldPoseInput, String> {
    let motions = parse_pose_string(
        pose_string,
        DEFAULT_FORWARD_SPEED,
        DEFAULT_YAW_SPEED,
        DEFAULT_PITCH_SPEED,
    )?;
    let mut c2ws = generate_camera_trajectory_local(&motions);
    // Ensure we have exactly latent_num poses (pad / truncate).
    if c2ws.is_empty() {
        c2ws.push(super::trajectory::identity4());
    }
    while c2ws.len() < latent_num {
        c2ws.push(*c2ws.last().unwrap());
    }
    c2ws.truncate(latent_num);

    let mut w2cs: Vec<Mat4> = c2ws.iter().map(inv4).collect();
    let mut view_flat = Vec::with_capacity(latent_num * 16);
    for m in &w2cs {
        for row in m {
            view_flat.extend_from_slice(row);
        }
    }

    // Normalize intrinsics like FastVideo.
    let mut k = DEFAULT_INTRINSIC;
    k[0][0] /= k[0][2] * 2.0;
    k[1][1] /= k[1][2] * 2.0;
    k[0][2] = 0.5;
    k[1][2] = 0.5;
    let mut k_flat = Vec::with_capacity(latent_num * 9);
    for _ in 0..latent_num {
        for row in &k {
            k_flat.extend_from_slice(row);
        }
    }

    // Relative c2w for action labels.
    let c2ws_arr = c2ws;
    let mut relative = vec![super::trajectory::identity4(); latent_num];
    relative[0] = c2ws_arr[0];
    for i in 1..latent_num {
        relative[i] = mul4(&inv4(&c2ws_arr[i - 1]), &c2ws_arr[i]);
    }

    let mut action_labels = vec![0i32; latent_num];
    let move_norm_valid = 0.0001f32;
    for i in 1..latent_num {
        let move_dirs = [relative[i][0][3], relative[i][1][3], relative[i][2][3]];
        let move_norm = (move_dirs[0] * move_dirs[0]
            + move_dirs[1] * move_dirs[1]
            + move_dirs[2] * move_dirs[2])
            .sqrt();
        let mut trans = [0i32; 4];
        let mut rotate = [0i32; 4];
        if move_norm > move_norm_valid {
            let dirs = [
                move_dirs[0] / move_norm,
                move_dirs[1] / move_norm,
                move_dirs[2] / move_norm,
            ];
            let angles: [f32; 3] = dirs.map(|d| d.clamp(-1.0, 1.0).acos().to_degrees());
            if angles[2] < 60.0 {
                trans[0] = 1;
            } else if angles[2] > 120.0 {
                trans[1] = 1;
            }
            if angles[0] < 60.0 {
                trans[2] = 1;
            } else if angles[0] > 120.0 {
                trans[3] = 1;
            }
        }
        // Rough euler from relative rotation (yz components).
        let r = &relative[i];
        let yaw = r[0][2].atan2(r[0][0]).to_degrees();
        let pitch = (-r[1][2]).asin().to_degrees();
        if yaw > 0.05 {
            rotate[0] = 1;
        } else if yaw < -0.05 {
            rotate[1] = 1;
        }
        if pitch > 0.05 {
            rotate[2] = 1;
        } else if pitch < -0.05 {
            rotate[3] = 1;
        }
        action_labels[i] = one_hot_to_label(trans) * 9 + one_hot_to_label(rotate);
    }

    let _ = &mut w2cs;
    Ok(HyWorldPoseInput {
        viewmats: view_flat,
        intrinsics: k_flat,
        action_labels,
        latent_num,
    })
}

pub fn compute_latent_num(num_frames: usize) -> usize {
    1 + (num_frames.saturating_sub(1)) / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_w31() {
        let m = parse_pose_string(
            "w-31",
            DEFAULT_FORWARD_SPEED,
            DEFAULT_YAW_SPEED,
            DEFAULT_PITCH_SPEED,
        )
        .unwrap();
        assert_eq!(m.len(), 31);
        assert!(m[0].forward > 0.0);
    }

    #[test]
    fn pose_to_input_labels() {
        let lat = compute_latent_num(81);
        let inp = pose_to_input("w-20", lat).unwrap();
        assert_eq!(inp.latent_num, lat);
        assert_eq!(inp.action_labels.len(), lat);
        assert_eq!(inp.viewmats.len(), lat * 16);
        // Later frames should show forward action.
        assert!(inp.action_labels.iter().any(|&a| a != 0));
    }
}
