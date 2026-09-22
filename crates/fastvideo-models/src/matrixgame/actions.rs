//! Matrix-Game keyboard / mouse action packing.
//! Ported from FastVideo `models/dits/matrixgame2/utils.py`.

use crate::matrixgame::MatrixGamePreset;

const CAM_VALUE: f32 = 0.1;

/// One-hot / multi-hot keyboard vector for a single key token.
pub fn keyboard_vec(preset: MatrixGamePreset, key: &str) -> Option<Vec<f32>> {
    let k = key.trim().to_ascii_lowercase();
    let dim = preset.keyboard_dim();
    let mut v = vec![0f32; dim];
    match dim {
        2 => match k.as_str() {
            "w" | "forward" => v[0] = 1.0,
            "s" | "back" | "backward" => v[1] = 1.0,
            "q" | "still" | "" => {}
            _ => return None,
        },
        4 => match k.as_str() {
            "w" | "forward" => v[0] = 1.0,
            "s" | "back" | "backward" => v[1] = 1.0,
            "a" | "left" => v[2] = 1.0,
            "d" | "right" => v[3] = 1.0,
            "q" | "still" | "" => {}
            _ => return None,
        },
        6 => match k.as_str() {
            "w" | "forward" => v[0] = 1.0,
            "s" | "back" | "backward" => v[1] = 1.0,
            "a" | "left" => v[2] = 1.0,
            "d" | "right" => v[3] = 1.0,
            "t1" => v[4] = 1.0,
            "t2" => v[5] = 1.0,
            "q" | "still" | "" => {}
            _ => return None,
        },
        7 => match k.as_str() {
            "q" | "still" => v[0] = 1.0,
            "w" | "forward" => v[1] = 1.0,
            "s" | "back" | "backward" => v[2] = 1.0,
            "j" | "left" => v[3] = 1.0,
            "l" | "right" => v[4] = 1.0,
            "a" => v[5] = 1.0,
            "d" => v[6] = 1.0,
            "" => {}
            _ => return None,
        },
        _ => return None,
    }
    Some(v)
}

/// Mouse `[pitch, yaw]` for camera tokens (`i/k/j/l/u`).
pub fn mouse_vec(key: &str) -> Option<[f32; 2]> {
    match key.trim().to_ascii_lowercase().as_str() {
        "i" | "camera_up" | "up" => Some([CAM_VALUE, 0.0]),
        "k" | "camera_down" | "down" => Some([-CAM_VALUE, 0.0]),
        "j" | "camera_l" | "camera_left" => Some([0.0, -CAM_VALUE]),
        "l" | "camera_r" | "camera_right" => Some([0.0, CAM_VALUE]),
        "u" | "still" | "q" | "" => Some([0.0, 0.0]),
        _ => None,
    }
}

/// Packed action tensors: keyboard `[T, kb_dim]`, mouse `[T, 2]` (row-major flat).
#[derive(Debug, Clone, PartialEq)]
pub struct ActionPack {
    pub keyboard: Vec<f32>,
    pub mouse: Vec<f32>,
    pub num_frames: usize,
    pub keyboard_dim: usize,
}

impl ActionPack {
    pub fn zeros(num_frames: usize, keyboard_dim: usize) -> Self {
        Self {
            keyboard: vec![0f32; num_frames * keyboard_dim],
            mouse: vec![0f32; num_frames * 2],
            num_frames,
            keyboard_dim,
        }
    }

    pub fn keyboard_frame(&self, t: usize) -> &[f32] {
        let o = t * self.keyboard_dim;
        &self.keyboard[o..o + self.keyboard_dim]
    }

    pub fn mouse_frame(&self, t: usize) -> [f32; 2] {
        let o = t * 2;
        [self.mouse[o], self.mouse[o + 1]]
    }
}

/// Hold a single keyboard+mouse action for all frames.
pub fn constant_action(
    preset: MatrixGamePreset,
    num_frames: usize,
    keyboard_key: &str,
    mouse_key: &str,
) -> Option<ActionPack> {
    let kb = keyboard_vec(preset, keyboard_key)?;
    let mouse = mouse_vec(mouse_key).unwrap_or([0.0, 0.0]);
    let mut pack = ActionPack::zeros(num_frames, preset.keyboard_dim());
    for t in 0..num_frames {
        let o = t * pack.keyboard_dim;
        pack.keyboard[o..o + pack.keyboard_dim].copy_from_slice(&kb);
        pack.mouse[t * 2] = mouse[0];
        pack.mouse[t * 2 + 1] = mouse[1];
    }
    Some(pack)
}

/// Deterministic action preset sequence (FastVideo `create_action_presets` simplified).
///
/// Cycles forward → left → right (and camera pans when mouse is enabled).
pub fn create_action_presets(
    preset: MatrixGamePreset,
    num_frames: usize,
    seed: u64,
) -> ActionPack {
    let kb_dim = preset.keyboard_dim();
    let mut pack = ActionPack::zeros(num_frames, kb_dim);
    let use_mouse = kb_dim != 7;
    let keys: &[&str] = match kb_dim {
        2 => &["w", "s", "w"],
        7 => &["w", "a", "d", "q"],
        _ => &["w", "a", "d", "w"],
    };
    let cams: &[&str] = if use_mouse {
        &["u", "j", "l", "u"]
    } else {
        &["u", "u", "u", "u"]
    };
    let mut rng = seed;
    let mut t = 0usize;
    while t < num_frames {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (rng as usize) % keys.len();
        let hold = 4usize.min(num_frames - t).max(1);
        let kb = keyboard_vec(preset, keys[idx]).unwrap_or_else(|| vec![0f32; kb_dim]);
        let mouse = mouse_vec(cams[idx % cams.len()]).unwrap_or([0.0, 0.0]);
        for dt in 0..hold {
            let tt = t + dt;
            let o = tt * kb_dim;
            pack.keyboard[o..o + kb_dim].copy_from_slice(&kb);
            if use_mouse {
                pack.mouse[tt * 2] = mouse[0];
                pack.mouse[tt * 2 + 1] = mouse[1];
            }
        }
        t += hold;
    }
    pack
}

/// Expand per-sample action rows to all frames (`[num_frames, D]` flat).
pub fn expand_action_to_frames(keyboard: &[f32], mouse: &[f32], num_frames: usize) -> ActionPack {
    let kb_dim = if keyboard.is_empty() {
        4
    } else {
        keyboard.len()
    };
    let mut pack = ActionPack::zeros(num_frames, kb_dim);
    for t in 0..num_frames {
        let o = t * kb_dim;
        let copy = keyboard.len().min(kb_dim);
        pack.keyboard[o..o + copy].copy_from_slice(&keyboard[..copy]);
        if mouse.len() >= 2 {
            pack.mouse[t * 2] = mouse[0];
            pack.mouse[t * 2 + 1] = mouse[1];
        }
    }
    pack
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_maps_match_dims() {
        let v = keyboard_vec(MatrixGamePreset::Mg2BaseDistilled, "w").unwrap();
        assert_eq!(v, vec![1.0, 0.0, 0.0, 0.0]);
        let gta = keyboard_vec(MatrixGamePreset::Mg2GtaDistilled, "s").unwrap();
        assert_eq!(gta, vec![0.0, 1.0]);
        let tr = keyboard_vec(MatrixGamePreset::Mg2TempleRunDistilled, "w").unwrap();
        assert_eq!(tr.len(), 7);
        assert_eq!(tr[1], 1.0);
    }

    #[test]
    fn presets_fill_frames() {
        let p = create_action_presets(MatrixGamePreset::Mg2BaseDistilled, 17, 0);
        assert_eq!(p.num_frames, 17);
        assert_eq!(p.keyboard.len(), 17 * 4);
        assert!(p.keyboard.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn constant_forward() {
        let p = constant_action(MatrixGamePreset::Mg3BaseDistilled, 8, "w", "u").unwrap();
        assert_eq!(p.keyboard_frame(3)[0], 1.0);
        assert_eq!(p.mouse_frame(0), [0.0, 0.0]);
    }
}
