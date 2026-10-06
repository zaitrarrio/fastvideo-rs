//! LingBot-Video sol-engine contract and its optimized arm.
//!
//! `models/lingbot_video.toml` (NVlabs/Sana `sol-engine`) publishes the
//! official two-stage T2V run: base 832×480 / 121 f / 40 steps, refiner
//! 1920×1088 / 8 steps, `t_thresh` 0.85, sigma tail 2, guidance 3, shift 3.
//!
//! The published winner (`config/lingbot_video/*cudnn_pisa_easycache_refiner*`,
//! 2.60× over the FA2 baseline on 4×GB200) layers three techniques on the
//! same sampling:
//!
//! * **kernel** — cuDNN flash attention instead of FA2 (here: whichever dense
//!   attention kernel the device runs fastest; nothing to port);
//! * **cache** — step-level "EasyCache" of `pipeline_lingbot_video.py`
//!   (optimized snapshot): reuse the last CFG output pair while the latent's
//!   relative L1 change since the last computed step stays under a threshold,
//!   outside a dense head/tail and for at most `max_reuse` steps in a row
//!   ([`EasyCache`]);
//! * **PISA** — piecewise sparse attention in the refiner only, density 0.10,
//!   block 64, layers 0–3 dense, the first two and last refiner steps dense
//!   ([`PisaPolicy`]).
//!
//! CP4 / FSDP / batched CFG are topology, not algorithms: a one-GPU run does
//! not reproduce them (see docs/ports/lingbot.md, "Comparison").

/// Official base canvas (`[official_config]`).
pub const OFFICIAL_WIDTH: usize = 832;
pub const OFFICIAL_HEIGHT: usize = 480;
pub const OFFICIAL_FRAMES: usize = 121;
pub const OFFICIAL_STEPS: usize = 40;
pub const OFFICIAL_GUIDANCE: f32 = 3.0;
pub const OFFICIAL_SHIFT: f64 = 3.0;
pub const OFFICIAL_FPS: u32 = 24;
pub const OFFICIAL_SEED: u64 = 42;

/// Official 1080p refiner.
pub const REFINER_WIDTH: usize = 1920;
pub const REFINER_HEIGHT: usize = 1088;
pub const REFINER_FRAMES: usize = 121;
pub const REFINER_STEPS: usize = 8;
pub const REFINER_GUIDANCE: f32 = 3.0;
pub const REFINER_SHIFT: f64 = 3.0;
pub const REFINER_T_THRESH: f64 = 0.85;
pub const REFINER_SIGMA_TAIL_STEPS: usize = 2;

/// Published numbers (4× GB200, CP4 + FSDP + batched CFG, 3-prompt median,
/// load excluded, from both models resident to the refined mp4).
pub const PUBLISHED_BASELINE_S: f64 = 375.53;
pub const PUBLISHED_FULLOPT_S: f64 = 144.36;

/// `FASTVIDEO_LINGBOT_OFFICIAL=1` (or `official`) selects the published
/// two-stage sampling contract.
pub fn official_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("official"),
        None => false,
    }
}

/// Which techniques a run turns on (`FASTVIDEO_LINGBOT_SOL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolArm {
    pub easycache: bool,
    pub pisa: bool,
}

impl SolArm {
    pub const BASELINE: Self = Self {
        easycache: false,
        pisa: false,
    };
    pub const FULLOPT: Self = Self {
        easycache: true,
        pisa: true,
    };

    /// `off` / unset → baseline; `1` / `sol` / `fullopt` → both; `cache` /
    /// `easycache` or `pisa` → one; a comma list combines.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        let Some(v) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(Self::BASELINE);
        };
        let mut arm = Self::BASELINE;
        for part in v.split(',').map(|p| p.trim().to_ascii_lowercase()) {
            match part.as_str() {
                "0" | "off" | "false" | "baseline" | "dense" => {}
                "1" | "sol" | "fullopt" | "on" | "true" => arm = Self::FULLOPT,
                "cache" | "easycache" => arm.easycache = true,
                "pisa" => arm.pisa = true,
                other => return Err(format!("FASTVIDEO_LINGBOT_SOL: unknown technique {other:?}")),
            }
        }
        Ok(arm)
    }

    pub fn label(self) -> &'static str {
        match (self.easycache, self.pisa) {
            (false, false) => "baseline",
            (true, true) => "fullopt",
            (true, false) => "easycache",
            (false, true) => "pisa",
        }
    }
}

/// EasyCache knobs of one stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EasyCacheConfig {
    pub threshold: f64,
    pub head_steps: usize,
    pub tail_steps: usize,
    pub max_reuse: usize,
}

impl EasyCacheConfig {
    /// `LINGBOT_EASYCACHE_*` of the published winner (base, ≥ 20 steps).
    pub const BASE: Self = Self {
        threshold: 0.08,
        head_steps: 4,
        tail_steps: 2,
        max_reuse: 2,
    };
    /// `LINGBOT_EASYCACHE_REFINER_*`.
    pub const REFINER: Self = Self {
        threshold: 0.25,
        head_steps: 2,
        tail_steps: 1,
        max_reuse: 2,
    };
    /// `LINGBOT_EASYCACHE_MIN_STEPS`: schedules at least this long use [`Self::BASE`].
    pub const MIN_BASE_STEPS: usize = 20;

    pub fn for_schedule(num_steps: usize) -> Self {
        if num_steps >= Self::MIN_BASE_STEPS {
            Self::BASE
        } else {
            Self::REFINER
        }
    }
}

/// Step-reuse controller. The caller asks [`Self::reuse`] before a step's
/// transformer pass; on `false` it computes and calls [`Self::computed`].
#[derive(Debug, Clone)]
pub struct EasyCache {
    pub cfg: EasyCacheConfig,
    num_steps: usize,
    /// Latent at the last computed step (`_ec["ref"]`).
    reference: Option<Vec<f32>>,
    run: usize,
    pub computed_steps: usize,
    pub reused_steps: usize,
}

impl EasyCache {
    pub fn new(cfg: EasyCacheConfig, num_steps: usize) -> Self {
        Self {
            cfg,
            num_steps,
            reference: None,
            run: 0,
            computed_steps: 0,
            reused_steps: 0,
        }
    }

    /// `mean|x - ref| / (mean|ref| + 1e-8)`.
    pub fn relative_l1(latent: &[f32], reference: &[f32]) -> f64 {
        let n = latent.len().max(1) as f64;
        let diff: f64 = latent
            .iter()
            .zip(reference)
            .map(|(&a, &b)| f64::from((a - b).abs()))
            .sum::<f64>()
            / n;
        let base: f64 = reference.iter().map(|&b| f64::from(b.abs())).sum::<f64>() / n;
        diff / (base + 1e-8)
    }

    /// Whether step `i` may reuse the cached outputs for `latent`.
    pub fn reuse(&mut self, i: usize, latent: &[f32]) -> bool {
        let Some(reference) = self.reference.as_deref() else {
            return false;
        };
        let window = self.cfg.head_steps <= i && i < self.num_steps.saturating_sub(self.cfg.tail_steps);
        if !window || self.run >= self.cfg.max_reuse {
            return false;
        }
        let ok = Self::relative_l1(latent, reference) < self.cfg.threshold;
        if ok {
            self.run += 1;
            self.reused_steps += 1;
        }
        ok
    }

    /// Record a computed step: its input latent becomes the reference.
    pub fn computed(&mut self, latent: &[f32]) {
        self.reference = Some(latent.to_vec());
        self.run = 0;
        self.computed_steps += 1;
    }
}

/// Refiner PISA of the published winner (`LINGBOT_PISA_*`).
#[derive(Debug, Clone, PartialEq)]
pub struct PisaPolicy {
    /// Kept KV-block fraction (`LINGBOT_PISA_DENSITY`); sparsity = 1 − density.
    pub density: f64,
    pub block_size: usize,
    /// Layers that stay dense (`LINGBOT_PISA_DENSE_LAYERS = "0-3"`).
    pub dense_layers: std::ops::RangeInclusive<usize>,
    pub dense_head_steps: usize,
    pub dense_tail_steps: usize,
    /// PISA in the base stage (`LINGBOT_PISA_BASE_ENABLED = 0`).
    pub base: bool,
    /// PISA in the refiner (`LINGBOT_PISA_REFINER_ENABLED = 1`).
    pub refiner: bool,
}

impl PisaPolicy {
    pub fn published() -> Self {
        Self {
            density: 0.10,
            block_size: 64,
            dense_layers: 0..=3,
            dense_head_steps: 2,
            dense_tail_steps: 1,
            base: false,
            refiner: true,
        }
    }

    pub fn sparsity(&self) -> f64 {
        1.0 - self.density
    }

    /// `set_lingbot_pisa_step`: steps `0..head` and the last `tail` are dense.
    pub fn sparse_step(&self, step: usize, num_steps: usize) -> bool {
        let head = self.dense_head_steps.min(num_steps);
        let tail = self.dense_tail_steps.min(num_steps);
        step >= head && step < num_steps - tail
    }

    pub fn sparse_layer(&self, layer: usize) -> bool {
        !self.dense_layers.contains(&layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_parsing() {
        assert_eq!(SolArm::parse(None).unwrap(), SolArm::BASELINE);
        assert_eq!(SolArm::parse(Some("off")).unwrap(), SolArm::BASELINE);
        assert_eq!(SolArm::parse(Some("1")).unwrap(), SolArm::FULLOPT);
        assert_eq!(SolArm::parse(Some("fullopt")).unwrap().label(), "fullopt");
        let c = SolArm::parse(Some("cache")).unwrap();
        assert!(c.easycache && !c.pisa);
        let both = SolArm::parse(Some("cache,pisa")).unwrap();
        assert_eq!(both, SolArm::FULLOPT);
        assert!(SolArm::parse(Some("teacache")).is_err());
    }

    #[test]
    fn official_env_is_off_until_named() {
        assert!(!official_requested(None));
        assert!(!official_requested(Some("off")));
        assert!(official_requested(Some("1")));
        assert!(official_requested(Some("official")));
        assert!(!official_requested(Some("cache")));
    }

    #[test]
    fn official_numbers_match_the_lingbot_toml() {
        assert_eq!((OFFICIAL_WIDTH, OFFICIAL_HEIGHT, OFFICIAL_FRAMES), (832, 480, 121));
        assert_eq!((OFFICIAL_STEPS, OFFICIAL_GUIDANCE, OFFICIAL_SHIFT), (40, 3.0, 3.0));
        assert_eq!((REFINER_WIDTH, REFINER_HEIGHT, REFINER_STEPS), (1920, 1088, 8));
        assert_eq!((REFINER_T_THRESH, REFINER_SIGMA_TAIL_STEPS), (0.85, 2));
        assert!((PUBLISHED_BASELINE_S / PUBLISHED_FULLOPT_S - 2.60).abs() < 0.01);
    }

    #[test]
    fn easycache_follows_the_reference_rule() {
        let mut ec = EasyCache::new(EasyCacheConfig::BASE, 40);
        let x = vec![1.0f32; 16];
        // No cache yet.
        assert!(!ec.reuse(5, &x));
        ec.computed(&x);
        // Inside the head: always compute.
        assert!(!ec.reuse(3, &x));
        // Small change in the window: reuse twice, then forced compute.
        let y: Vec<f32> = x.iter().map(|v| v * 1.05).collect();
        assert!(ec.reuse(4, &y));
        assert!(ec.reuse(5, &y));
        assert!(!ec.reuse(6, &y), "max_reuse 2");
        ec.computed(&y);
        // Large change: compute.
        let z: Vec<f32> = y.iter().map(|v| v * 1.5).collect();
        assert!(!ec.reuse(7, &z));
        // Tail: steps 38, 39 always compute.
        ec.computed(&z);
        assert!(!ec.reuse(38, &z));
        assert_eq!((ec.reused_steps, ec.computed_steps), (2, 3));
        assert_eq!(EasyCacheConfig::for_schedule(9), EasyCacheConfig::REFINER);
        assert_eq!(EasyCacheConfig::for_schedule(40), EasyCacheConfig::BASE);
    }

    #[test]
    fn pisa_dense_guards() {
        let p = PisaPolicy::published();
        assert!((p.sparsity() - 0.9).abs() < 1e-12);
        let n = 10;
        let sparse: Vec<usize> = (0..n).filter(|&s| p.sparse_step(s, n)).collect();
        assert_eq!(sparse, (2..9).collect::<Vec<_>>());
        assert!(!p.sparse_layer(3) && p.sparse_layer(4));
        assert!(p.refiner && !p.base);
    }
}
