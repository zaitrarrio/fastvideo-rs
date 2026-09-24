//! Wan Sol-Attn / PISA routes from NVlabs/Sana `sol-engine` manifests.
//!
//! 14B (`config/wan21_t2v_14b/fullstack.toml`): Sol-Attn tau 1.0, first 10
//! steps dense, layer 0 dense, global Morton3D order.
//! 5B (`config/wan22_ti2v_5b/wan5b_kernel_easycache_pisa.toml`): PISA
//! density 0.10, dense layers 0-3,26-29 and dense steps 0-3,47-49.
//! A14B (`config/wan22_t2v_a14b/singlegpu_opt.toml`): PISA density 0.10,
//! dense layers 0-3,40-43 and dense steps 0-3,37-39.

/// Wan 2.1 14B Sol-Attn tau (`WAN22_SOL_TAU`).
pub const SOL_14B_TAU: f64 = 1.0;
/// First N denoising steps stay dense (`WAN22_SOL_DENSE_STEPS`).
pub const SOL_14B_DENSE_STEPS: usize = 10;
/// Layer 0 stays dense (`WAN22_SOL_DENSE_LAYERS=0`).
pub const SOL_14B_DENSE_LAYER: usize = 0;

/// Published PISA keep fraction (`WAN22_PISA_DENSITY`).
pub const PISA_DENSITY: f64 = 0.10;
/// `pisa_attn` takes sparsity = 1 − density.
pub const PISA_SPARSITY: f64 = 1.0 - PISA_DENSITY;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WanAttnProfile {
    #[default]
    Off,
    /// Wan 2.1 14B Sol-Attn + Morton3D.
    Sol14b,
    /// Wan 2.2 TI2V-5B PISA.
    Pisa5b,
    /// Wan 2.2 A14B PISA.
    PisaA14b,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WanAttnRoute {
    Dense,
    Sol { tau: f64 },
    Pisa { sparsity: f64 },
}

pub fn sol_attn_requested(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1") | Some("sol") | Some("true") | Some("on")
    )
}

pub fn pisa_requested(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1") | Some("pisa") | Some("true") | Some("on")
    )
}

pub fn pisa_5b_dense_layer(layer: usize) -> bool {
    (0..=3).contains(&layer) || (26..=29).contains(&layer)
}

pub fn pisa_5b_dense_step(step: usize) -> bool {
    (0..=3).contains(&step) || (47..=49).contains(&step)
}

pub fn pisa_a14b_dense_layer(layer: usize) -> bool {
    (0..=3).contains(&layer) || (40..=43).contains(&layer)
}

pub fn pisa_a14b_dense_step(step: usize) -> bool {
    (0..=3).contains(&step) || (37..=39).contains(&step)
}

/// Route for one `(step, layer)` on a published Wan profile.
pub fn route(profile: WanAttnProfile, step: usize, layer: usize) -> WanAttnRoute {
    match profile {
        WanAttnProfile::Off => WanAttnRoute::Dense,
        WanAttnProfile::Sol14b => {
            if step < SOL_14B_DENSE_STEPS || layer == SOL_14B_DENSE_LAYER {
                WanAttnRoute::Dense
            } else {
                WanAttnRoute::Sol { tau: SOL_14B_TAU }
            }
        }
        WanAttnProfile::Pisa5b => {
            if pisa_5b_dense_step(step) || pisa_5b_dense_layer(layer) {
                WanAttnRoute::Dense
            } else {
                WanAttnRoute::Pisa {
                    sparsity: PISA_SPARSITY,
                }
            }
        }
        WanAttnProfile::PisaA14b => {
            if pisa_a14b_dense_step(step) || pisa_a14b_dense_layer(layer) {
                WanAttnRoute::Dense
            } else {
                WanAttnRoute::Pisa {
                    sparsity: PISA_SPARSITY,
                }
            }
        }
    }
}

/// Morton3D should wrap the 14B Sol kernel calls only.
pub fn morton3d_on_route(profile: WanAttnProfile, step: usize, layer: usize) -> bool {
    matches!(route(profile, step, layer), WanAttnRoute::Sol { .. })
}

fn part1by2(mut value: u64) -> u64 {
    value &= 0x1f_ffff;
    value = (value | (value << 32)) & 0x1f_0000_0000_ffff;
    value = (value | (value << 16)) & 0x1f_0000_ff00_00ff;
    value = (value | (value << 8)) & 0x100f_00f0_0f00_f00f;
    value = (value | (value << 4)) & 0x10c3_0c30_c30c_30c3;
    value = (value | (value << 2)) & 0x1249_2492_4924_9249;
    value
}

/// Canonical x/y/z-interleaved Morton permutation (`_morton3d_perm`).
///
/// `perm[new] = old` so gathering sequence `perm` yields Morton order.
pub fn morton3d_perm(frames: usize, height: usize, width: usize) -> Vec<usize> {
    let total = frames.saturating_mul(height).saturating_mul(width);
    if total == 0 {
        return Vec::new();
    }
    let frame_area = height * width;
    let mut codes: Vec<(u64, usize)> = (0..total)
        .map(|i| {
            let z = i / frame_area;
            let rem = i - z * frame_area;
            let y = rem / width;
            let x = rem - y * width;
            let code = part1by2(x as u64) | (part1by2(y as u64) << 1) | (part1by2(z as u64) << 2);
            (code, i)
        })
        .collect();
    codes.sort_by_key(|&(code, i)| (code, i));
    codes.into_iter().map(|(_, i)| i).collect()
}

pub fn morton3d_inverse(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; perm.len()];
    for (new, &old) in perm.iter().enumerate() {
        if old < inverse.len() {
            inverse[old] = new;
        }
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sol_14b_keeps_ten_dense_steps_and_layer_zero() {
        assert_eq!(route(WanAttnProfile::Sol14b, 0, 1), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Sol14b, 9, 7), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Sol14b, 10, 0), WanAttnRoute::Dense);
        assert_eq!(
            route(WanAttnProfile::Sol14b, 10, 1),
            WanAttnRoute::Sol { tau: 1.0 }
        );
        assert_eq!(
            route(WanAttnProfile::Sol14b, 39, 39),
            WanAttnRoute::Sol { tau: 1.0 }
        );
        assert!(morton3d_on_route(WanAttnProfile::Sol14b, 10, 1));
        assert!(!morton3d_on_route(WanAttnProfile::Sol14b, 0, 1));
        assert!(!morton3d_on_route(WanAttnProfile::Pisa5b, 10, 5));
    }

    #[test]
    fn pisa_5b_uses_published_dense_sets() {
        assert_eq!(PISA_DENSITY, 0.10);
        assert!((PISA_SPARSITY - 0.90).abs() < 1e-12);
        assert_eq!(route(WanAttnProfile::Pisa5b, 10, 0), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Pisa5b, 10, 3), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Pisa5b, 10, 26), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Pisa5b, 10, 29), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Pisa5b, 0, 10), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Pisa5b, 3, 10), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Pisa5b, 47, 10), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::Pisa5b, 49, 10), WanAttnRoute::Dense);
        assert_eq!(
            route(WanAttnProfile::Pisa5b, 10, 10),
            WanAttnRoute::Pisa { sparsity: 0.90 }
        );
    }

    #[test]
    fn pisa_a14b_uses_published_dense_sets() {
        assert_eq!(route(WanAttnProfile::PisaA14b, 10, 0), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::PisaA14b, 10, 40), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::PisaA14b, 10, 43), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::PisaA14b, 0, 10), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::PisaA14b, 37, 10), WanAttnRoute::Dense);
        assert_eq!(route(WanAttnProfile::PisaA14b, 39, 10), WanAttnRoute::Dense);
        assert_eq!(
            route(WanAttnProfile::PisaA14b, 10, 10),
            WanAttnRoute::Pisa { sparsity: 0.90 }
        );
        assert_eq!(
            route(WanAttnProfile::PisaA14b, 20, 20),
            WanAttnRoute::Pisa { sparsity: 0.90 }
        );
    }

    #[test]
    fn morton3d_is_a_permutation_of_the_grid() {
        let perm = morton3d_perm(2, 2, 2);
        assert_eq!(perm.len(), 8);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
        let inverse = morton3d_inverse(&perm);
        for (new, &old) in perm.iter().enumerate() {
            assert_eq!(inverse[old], new);
        }
        // Identity on a 1×1×1 grid; linear order is stable for a 1-wide strip.
        assert_eq!(morton3d_perm(1, 1, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn env_flags_are_off_until_named() {
        assert!(!sol_attn_requested(None));
        assert!(!sol_attn_requested(Some("off")));
        assert!(sol_attn_requested(Some("1")));
        assert!(sol_attn_requested(Some("sol")));
        assert!(!pisa_requested(None));
        assert!(pisa_requested(Some("pisa")));
    }
}
