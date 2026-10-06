//! Wan 2.2 A14B: where the two experts live, and which one runs when.
//!
//! The checkpoint (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`) is two Wan-14B DiTs,
//! `transformer` (high noise) and `transformer_2` (low noise), each 57.15 GB
//! of float32 on disk and 26.6 GiB as the bf16 this port runs. Diffusers'
//! `WanPipeline` uses `transformer` while `t >= boundary_ratio * 1000`
//! (`model_index.json`: 0.875), then `transformer_2`: one switch per
//! generation. Both experts plus the f32 UMT5 encoder (21.1 GiB) and the
//! 720p activations do not fit one 96 GB RTX PRO 6000; on a B200 (180 GB) or
//! H200 they do.
//!
//! `FASTVIDEO_WAN_MOE` picks the placement:
//!
//! * `both` — both experts resident (B200 / H200).
//! * `swap` — both parked in pinned host memory; the expert that denoises is
//!   on the device, the other is not. At the boundary the high expert's
//!   device copy is dropped and the low one comes in, its blocks copied on a
//!   second stream ahead of the block computing (the first low-noise
//!   forward overlaps the copy); the next generation swaps back. Two H2D
//!   copies of 26.6 GiB per generation, about 1 s each at PCIe 5 rates.
//! * `auto` (default) — `both` when the free device memory after the text
//!   encoder covers two experts plus [`HEADROOM_ENV`] GiB (default
//!   [`DEFAULT_HEADROOM_GIB`]), else `swap`.
//!
//! FP8 weights (`FASTVIDEO_WAN_QUANT=mxfp8|w8a8`, the reference recipes on
//! every block linear) halve an expert to about 13.7 GiB: both then fit
//! resident on a 96 GB card, at the recipe's (lossy) quality.

use super::config::WanVideoArchConfig;

pub const ENV: &str = "FASTVIDEO_WAN_MOE";
pub const HEADROOM_ENV: &str = "FASTVIDEO_WAN_MOE_HEADROOM_GIB";
/// Activations, workspaces and the VAE decode at 720x1280x81 with CFG
/// batched: generous, so `auto` errs toward the swap on a 96 GB card.
pub const DEFAULT_HEADROOM_GIB: f64 = 24.0;

const GIB: f64 = (1u64 << 30) as f64;

/// The requested placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoeResidency {
    #[default]
    Auto,
    Both,
    Swap,
}

/// The placement a load decided on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoePlan {
    Both,
    Swap,
}

impl MoePlan {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::Swap => "swap",
        }
    }
}

impl MoeResidency {
    pub fn parse(v: &str) -> Result<Self, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "both" | "resident" => Ok(Self::Both),
            "swap" | "parked" => Ok(Self::Swap),
            other => Err(format!("{ENV}={other}: expected auto, both or swap")),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        std::env::var(ENV).map_or(Ok(Self::Auto), |v| Self::parse(&v))
    }

    /// `free`: device bytes free before the experts load (`None`: no
    /// device, nothing to plan for). `expert`: one expert's device bytes.
    pub fn resolve(self, expert: u64, headroom: u64, free: Option<u64>) -> MoePlan {
        match self {
            Self::Both => MoePlan::Both,
            Self::Swap => MoePlan::Swap,
            Self::Auto => match free {
                Some(free) if free < 2 * expert + headroom => MoePlan::Swap,
                _ => MoePlan::Both,
            },
        }
    }
}

/// [`HEADROOM_ENV`] in bytes.
pub fn headroom_bytes() -> u64 {
    let gib = std::env::var(HEADROOM_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .unwrap_or(DEFAULT_HEADROOM_GIB);
    (gib * GIB) as u64
}

/// Parameters of one transformer block (attention projections with
/// biases, RMS norms, `norm2`, the FFN and the modulation table).
pub fn block_params(cfg: &WanVideoArchConfig) -> u64 {
    let d = cfg.hidden_size() as u64;
    let f = cfg.ffn_dim as u64;
    let attn = 4 * (d * d + d) + 2 * d;
    let image_kv = cfg
        .added_kv_proj_dim
        .map_or(0, |k| 2 * (k as u64 * d + d) + d);
    2 * attn + image_kv + 2 * d + (d * f + f) + (f * d + d) + 6 * d
}

/// Parameters outside the blocks: patch embedding, time / text embedders,
/// `time_proj`, the output head and its table.
pub fn global_params(cfg: &WanVideoArchConfig) -> u64 {
    let d = cfg.hidden_size() as u64;
    let patch: u64 = cfg.patch_size.iter().map(|&p| p as u64).product();
    let image = cfg.image_dim.map_or(0, |i| {
        let i = i as u64;
        2 * i + (i * i + i) + (i * d + d) + 2 * d
    });
    (d * cfg.in_channels as u64 * patch + d)
        + (cfg.freq_dim as u64 * d + d)
        + (d * d + d)
        + (d * 6 * d + 6 * d)
        + (cfg.text_dim as u64 * d + d)
        + (d * d + d)
        + image
        + (d * cfg.out_channels as u64 * patch + cfg.out_channels as u64 * patch)
        + 2 * d
}

/// Device bytes of one expert at bf16.
pub fn expert_bf16_bytes(cfg: &WanVideoArchConfig) -> u64 {
    2 * (cfg.num_layers as u64 * block_params(cfg) + global_params(cfg))
}

/// Expert calls per generation: `(high, low)` steps for `timesteps`
/// (descending, `0..1000`) at `boundary_ratio`, the split
/// [`super::moe_expert`] makes.
pub fn expert_steps(timesteps: &[f64], boundary_ratio: f32) -> (usize, usize) {
    let high = timesteps
        .iter()
        .filter(|&&t| super::moe_expert(t, boundary_ratio, 1000) == super::MoeExpert::HighNoise)
        .count();
    (high, timesteps.len() - high)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedulers::{FlowUniPCMultistepScheduler, UniPcSigmas};

    /// `Wan-AI/Wan2.2-T2V-A14B-Diffusers` `transformer/` is 57 153 966 336
    /// bytes of float32 (`diffusion_pytorch_model.safetensors.index.json`
    /// `total_size`): 14.288 B parameters. The 1.3B's two shards hold
    /// 5 676 070 648 bytes minus their headers: 1.419 B.
    #[test]
    fn parameter_counts_match_the_published_checkpoints() {
        let a14b = WanVideoArchConfig::wan_2_2_t2v_a14b();
        let total = a14b.num_layers as u64 * block_params(&a14b) + global_params(&a14b);
        assert_eq!(total * 4, 57_153_966_336 / 4 * 4);
        assert_eq!(total, 14_288_491_584);
        let gib = expert_bf16_bytes(&a14b) as f64 / GIB;
        assert!((26.5..26.7).contains(&gib), "{gib} GiB");
        let small = WanVideoArchConfig::wan_t2v_1_3b();
        let n = small.num_layers as u64 * block_params(&small) + global_params(&small);
        assert_eq!(n, 1_418_996_800);
    }

    #[test]
    fn auto_swaps_on_a_96_gb_card_and_keeps_both_on_a_b200() {
        let cfg = WanVideoArchConfig::wan_2_2_t2v_a14b();
        let expert = expert_bf16_bytes(&cfg);
        let head = (DEFAULT_HEADROOM_GIB * GIB) as u64;
        // RTX PRO 6000: 94.97 GiB, minus the f32 UMT5 (21.1 GiB) and context.
        let pro6000 = ((94.97 - 21.2 - 0.6) * GIB) as u64;
        assert_eq!(
            MoeResidency::Auto.resolve(expert, head, Some(pro6000)),
            MoePlan::Swap
        );
        // B200: 178.4 GiB.
        let b200 = ((178.4 - 21.2 - 0.6) * GIB) as u64;
        assert_eq!(
            MoeResidency::Auto.resolve(expert, head, Some(b200)),
            MoePlan::Both
        );
        assert_eq!(
            MoeResidency::Auto.resolve(expert, head, None),
            MoePlan::Both
        );
        assert_eq!(
            MoeResidency::Both.resolve(expert, head, Some(1)),
            MoePlan::Both
        );
        assert_eq!(
            MoeResidency::Swap.resolve(expert, head, Some(u64::MAX / 4)),
            MoePlan::Swap
        );
        assert_eq!(MoeResidency::parse("SWAP").unwrap(), MoeResidency::Swap);
        assert!(MoeResidency::parse("half").is_err());
    }

    /// sol-engine's 1x GB200 optimized run records 52 `transformer` and 28
    /// `transformer_2` forwards for 40 steps with CFG
    /// (`evals/_golden/wan14b_opt_1g/benchmark.json`
    /// `pisa_step_tracking`): 26 high-noise steps, 14 low-noise, at flow
    /// shift 12 and boundary 0.875. Both sigma lists split the same way.
    #[test]
    fn a14b_boundary_split_matches_the_sol_engine_golden_run() {
        let cfg = WanVideoArchConfig::wan_2_2_t2v_a14b();
        let ratio = cfg.boundary_ratio.expect("moe");
        assert_eq!(ratio, 0.875);
        for kind in [UniPcSigmas::Diffusers, UniPcSigmas::FastVideo] {
            let mut sched = FlowUniPCMultistepScheduler::with_sigmas(kind, 1000, 12.0);
            sched.set_timesteps(40);
            let ts: Vec<f64> = sched
                .inference_timesteps_i64()
                .iter()
                .map(|&t| t as f64)
                .collect();
            assert_eq!(expert_steps(&ts, ratio), (26, 14), "{kind:?}");
            // Two forwards per step under CFG: 52 / 28.
            let (h, l) = expert_steps(&ts, ratio);
            assert_eq!((2 * h, 2 * l), (52, 28));
            // The first low-noise timestep is the first below 875.
            assert!(
                ts[25] >= 875.0 && ts[26] < 875.0,
                "{kind:?}: {:?}",
                &ts[24..28]
            );
        }
    }
}
