//! HY-WorldPlay local camera trajectory.
//! Ported from FastVideo `models/dits/hyworld/trajectory.py`.

pub type Mat4 = [[f32; 4]; 4];

pub fn identity4() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rot_x(theta: f32) -> [[f32; 3]; 3] {
    let (c, s) = (theta.cos(), theta.sin());
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rot_y(theta: f32) -> [[f32; 3]; 3] {
    let (c, s) = (theta.cos(), theta.sin());
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat3_vec(a: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

/// One motion step along the local camera frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct Motion {
    pub forward: f32,
    pub right: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub third_yaw: f32,
}

/// Generate c2w poses from a motion list (starts with identity).
pub fn generate_camera_trajectory_local(motions: &[Motion]) -> Vec<Mat4> {
    let mut poses = Vec::with_capacity(motions.len() + 1);
    let mut t = identity4();
    poses.push(t);
    for move_ in motions {
        if move_.yaw != 0.0 {
            let r = rot_y(move_.yaw);
            let cur = [
                [t[0][0], t[0][1], t[0][2]],
                [t[1][0], t[1][1], t[1][2]],
                [t[2][0], t[2][1], t[2][2]],
            ];
            let nr = mat3_mul(&cur, &r);
            for i in 0..3 {
                for j in 0..3 {
                    t[i][j] = nr[i][j];
                }
            }
        }
        if move_.pitch != 0.0 {
            let r = rot_x(move_.pitch);
            let cur = [
                [t[0][0], t[0][1], t[0][2]],
                [t[1][0], t[1][1], t[1][2]],
                [t[2][0], t[2][1], t[2][2]],
            ];
            let nr = mat3_mul(&cur, &r);
            for i in 0..3 {
                for j in 0..3 {
                    t[i][j] = nr[i][j];
                }
            }
        }
        if move_.forward != 0.0 {
            let r = [
                [t[0][0], t[0][1], t[0][2]],
                [t[1][0], t[1][1], t[1][2]],
                [t[2][0], t[2][1], t[2][2]],
            ];
            let world = mat3_vec(&r, [0.0, 0.0, move_.forward]);
            t[0][3] += world[0];
            t[1][3] += world[1];
            t[2][3] += world[2];
        }
        if move_.right != 0.0 {
            let r = [
                [t[0][0], t[0][1], t[0][2]],
                [t[1][0], t[1][1], t[1][2]],
                [t[2][0], t[2][1], t[2][2]],
            ];
            let world = mat3_vec(&r, [move_.right, 0.0, 0.0]);
            t[0][3] += world[0];
            t[1][3] += world[1];
            t[2][3] += world[2];
        }
        if move_.third_yaw != 0.0 {
            let theta = -move_.third_yaw;
            let (c, s) = (theta.cos(), theta.sin());
            // Simplified third-person yaw around look-at.
            let r_y = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
            let cur = [
                [t[0][0], t[0][1], t[0][2]],
                [t[1][0], t[1][1], t[1][2]],
                [t[2][0], t[2][1], t[2][2]],
            ];
            let nr = mat3_mul(&cur, &r_y);
            for i in 0..3 {
                for j in 0..3 {
                    t[i][j] = nr[i][j];
                }
            }
        }
        poses.push(t);
    }
    poses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_moves_z() {
        let motions = vec![
            Motion {
                forward: 0.08,
                ..Default::default()
            };
            4
        ];
        let poses = generate_camera_trajectory_local(&motions);
        assert_eq!(poses.len(), 5);
        assert!(poses[4][2][3].abs() > poses[0][2][3].abs());
    }
}
