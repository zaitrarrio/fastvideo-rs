//! LTX-2.3 stage-2 PISA contract (`models/ltx23/optimized/env.sh` on
//! NVlabs/Sana `sol-engine`).
//!
//! Video self-attention only. Layers 0 and 1 stay dense. Later layers are
//! piecewise sparse attention at sparsity 0.9 and block size 64. Stage-2
//! steps 1 and 2 are the midpoint token-prune steps (keep half, by feature
//! norm). The stage-1 SCSP preset name is recorded here; its skip mask is
//! not applied. LoRA strengths are recorded and not fused.

/// Video blocks that stay dense on every stage-2 forward.
pub const DENSE_LAYERS: [usize; 2] = [0, 1];

pub const LAYERS_PER_FORWARD: usize = 48;

pub const FORWARDS: usize = 3;

pub const SPARSITY: f64 = 0.9;

pub const BLOCK_SIZE: usize = 64;

pub const STAGE1_LORA_STRENGTH: f64 = 0.25;

pub const STAGE2_LORA_STRENGTH: f64 = 0.5;

/// Keep this fraction of video tokens. `1.0 - ratio` is dropped.
pub const PRUNE_RATIO: f64 = 0.5;

/// Refine steps that prune. Step 0 does not.
pub const PRUNE_STEPS: [usize; 2] = [1, 2];

/// Named stage-1 cache preset. The skip schedule itself is not in this repo.
pub const STAGE1_CACHE_PRESET: &str = "8of15_last_29calls";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ltx23PisaRoute {
    Dense,
    /// Piecewise sparse video self-attention. Still dense SDPA until that kernel is linked.
    Pisa {
        sparsity: f64,
        block_size: usize,
    },
}

pub fn route(forward: usize, layer: usize) -> Result<Ltx23PisaRoute, String> {
    if forward >= FORWARDS {
        return Err(format!(
            "ltx2 pisa: stage-2 forward {forward} is past {FORWARDS} forwards"
        ));
    }
    if layer >= LAYERS_PER_FORWARD {
        return Err(format!(
            "ltx2 pisa: layer {layer} is past {LAYERS_PER_FORWARD} video blocks"
        ));
    }
    if DENSE_LAYERS.contains(&layer) {
        Ok(Ltx23PisaRoute::Dense)
    } else {
        Ok(Ltx23PisaRoute::Pisa {
            sparsity: SPARSITY,
            block_size: BLOCK_SIZE,
        })
    }
}

pub fn prunes_step(step: usize) -> bool {
    PRUNE_STEPS.contains(&step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_zero_and_one_stay_dense() {
        assert_eq!(route(0, 0).unwrap(), Ltx23PisaRoute::Dense);
        assert_eq!(route(1, 1).unwrap(), Ltx23PisaRoute::Dense);
        assert_eq!(route(2, 0).unwrap(), Ltx23PisaRoute::Dense);
        assert_eq!(
            route(0, 2).unwrap(),
            Ltx23PisaRoute::Pisa {
                sparsity: 0.9,
                block_size: 64
            }
        );
        assert_eq!(
            route(2, 47).unwrap(),
            Ltx23PisaRoute::Pisa {
                sparsity: SPARSITY,
                block_size: BLOCK_SIZE
            }
        );
        assert!(route(3, 0).is_err());
        assert!(route(0, 48).is_err());
    }

    #[test]
    fn prune_is_the_middle_refine_steps() {
        assert!(!prunes_step(0));
        assert!(prunes_step(1));
        assert!(prunes_step(2));
        assert!(!prunes_step(3));
        assert_eq!(STAGE1_CACHE_PRESET, "8of15_last_29calls");
        assert_eq!(STAGE1_LORA_STRENGTH, 0.25);
        assert_eq!(STAGE2_LORA_STRENGTH, 0.5);
    }
}
