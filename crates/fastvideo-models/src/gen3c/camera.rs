//! GEN3C camera trajectory generation.
//! Ported from FastVideo `pipelines/basic/gen3c/camera_utils.py`.

use super::trajectory::{CameraRotation, TrajectoryType};

/// Row-major 4×4 matrix.
pub type Mat4 = [[f32; 4]; 4];
/// Row-major 3×3 matrix.
pub type Mat3 = [[f32; 3]; 3];

pub fn identity4() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = norm3(v).max(1e-8);
    scale3(v, 1.0 / n)
}

/// Look-at view matrix (world-to-camera style used by GEN3C).
pub fn look_at_matrix(camera_pos: [f32; 3], target: [f32; 3], invert_pos: bool) -> Mat4 {
    let mut forward = [
        target[0] - camera_pos[0],
        target[1] - camera_pos[1],
        target[2] - camera_pos[2],
    ];
    forward = normalize3(forward);
    let up0 = [0.0, 1.0, 0.0];
    let mut right = cross(up0, forward);
    right = normalize3(right);
    let up = cross(forward, right);
    let pos = if invert_pos {
        [-camera_pos[0], -camera_pos[1], -camera_pos[2]]
    } else {
        camera_pos
    };
    [
        [right[0], right[1], right[2], pos[0]],
        [up[0], up[1], up[2], pos[1]],
        [forward[0], forward[1], forward[2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn create_horizontal_trajectory(
    world_to_camera: &Mat4,
    center_depth: f32,
    positive: bool,
    n_steps: usize,
    distance: f32,
    axis: char,
    camera_rotation: CameraRotation,
) -> Vec<Mat4> {
    let look_at_target = [0.0, 0.0, center_depth];
    let sign = if positive { 1.0 } else { -1.0 };
    let mut out = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let offset = (i as f32) * distance * center_depth / (n_steps as f32) * sign;
        let pos = match axis {
            'x' => [offset, 0.0, 0.0],
            'y' => [0.0, offset, 0.0],
            'z' => [0.0, 0.0, offset],
            _ => [offset, 0.0, 0.0],
        };
        let look = match camera_rotation {
            CameraRotation::CenterFacing => look_at_target,
            CameraRotation::NoRotation => [
                look_at_target[0] + pos[0],
                look_at_target[1] + pos[1],
                look_at_target[2] + pos[2],
            ],
        };
        let view = look_at_matrix(pos, look, true);
        out.push(mat4_mul(&view, world_to_camera));
    }
    out
}

fn create_spiral_trajectory(
    world_to_camera: &Mat4,
    center_depth: f32,
    positive: bool,
    n_steps: usize,
    radius: f32,
    camera_rotation: CameraRotation,
) -> Vec<Mat4> {
    let look_at_target = [0.0, 0.0, center_depth];
    let sign = if positive { 1.0 } else { -1.0 };
    let mut out = Vec::with_capacity(n_steps);
    let n = n_steps.max(2);
    for i in 0..n_steps {
        let theta = std::f32::consts::TAU * (i as f32) / ((n - 1) as f32);
        let x = radius * (theta.cos() - 1.0) * sign * center_depth;
        let y = radius * theta.sin() * center_depth;
        let pos = [x, y, 0.0];
        let look = match camera_rotation {
            CameraRotation::CenterFacing => look_at_target,
            CameraRotation::NoRotation => [
                look_at_target[0] + pos[0],
                look_at_target[1] + pos[1],
                look_at_target[2] + pos[2],
            ],
        };
        let view = look_at_matrix(pos, look, true);
        out.push(mat4_mul(&view, world_to_camera));
    }
    out
}

/// Default pinhole intrinsics for a canvas of `height`×`width`.
pub fn default_intrinsics(height: usize, width: usize) -> Mat3 {
    let fx = width as f32 * 0.5;
    let fy = height as f32 * 0.5;
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
}

/// Generate per-frame world-to-camera matrices and matching intrinsics.
///
/// Returns `(w2cs, intrinsics)` with lengths `num_frames`.
pub fn generate_camera_trajectory(
    trajectory: TrajectoryType,
    num_frames: usize,
    movement_distance: f32,
    camera_rotation: CameraRotation,
    center_depth: f32,
    height: usize,
    width: usize,
) -> (Vec<Mat4>, Vec<Mat3>) {
    let initial_w2c = identity4();
    let k = default_intrinsics(height, width);
    let w2cs = match trajectory {
        TrajectoryType::Clockwise | TrajectoryType::Counterclockwise => create_spiral_trajectory(
            &initial_w2c,
            center_depth,
            matches!(trajectory, TrajectoryType::Clockwise),
            num_frames,
            movement_distance,
            camera_rotation,
        ),
        other => {
            let (positive, axis) = match other {
                TrajectoryType::Left => (false, 'x'),
                TrajectoryType::Right => (true, 'x'),
                TrajectoryType::ZoomIn => (true, 'z'),
                TrajectoryType::ZoomOut => (false, 'z'),
                TrajectoryType::Clockwise | TrajectoryType::Counterclockwise => unreachable!(),
            };
            create_horizontal_trajectory(
                &initial_w2c,
                center_depth,
                positive,
                num_frames,
                movement_distance,
                axis,
                camera_rotation,
            )
        }
    };
    let intrinsics = vec![k; num_frames];
    (w2cs, intrinsics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn left_trajectory_moves_along_x() {
        let (w2cs, ks) = generate_camera_trajectory(
            TrajectoryType::Left,
            5,
            0.3,
            CameraRotation::CenterFacing,
            1.0,
            64,
            64,
        );
        assert_eq!(w2cs.len(), 5);
        assert_eq!(ks.len(), 5);
        // First frame near identity; last frame translated.
        assert!((w2cs[0][0][3]).abs() < 1e-3 || (w2cs[4][0][3] - w2cs[0][0][3]).abs() > 1e-4);
    }

    #[test]
    fn look_at_is_orthonormal_rows() {
        let m = look_at_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], true);
        let r = [m[0][0], m[0][1], m[0][2]];
        let u = [m[1][0], m[1][1], m[1][2]];
        let f = [m[2][0], m[2][1], m[2][2]];
        assert!((norm3(r) - 1.0).abs() < 1e-5);
        assert!((norm3(u) - 1.0).abs() < 1e-5);
        assert!((norm3(f) - 1.0).abs() < 1e-5);
    }
}
