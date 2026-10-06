//! Sol-Engine's SANA-Video arms, as far as they are published.
//!
//! The `sol-engine` branch ships only the SANA *5B* baseline wrapper
//! (`models/sana_video`, a private `yitongl/sana_video` bundle with a
//! Qwen text encoder and the LTX-2.3 VAE, 1280x736x193, 50 steps, cfg 8,
//! shift 12). The published *speedup* is for the public 2B 480p model:
//! README / `site_docs/pipelines/sana.md` — EasyCache 0.1 + linear-attention
//! BF16 + QKV merge + `torch.compile`, ~2.77x warm end to end on one GB200 at
//! 832x480, 81 frames, 50 steps. The run script it names
//! (`scripts/sana/sana_video_sglang_run.py`) is not in the branch, so the
//! EasyCache warmup / cooldown are not published; this port uses Sol-Engine's
//! own EasyCache runtime defaults (`WAN22_EASYCACHE_RETAIN_STEPS=7`,
//! `COOLDOWN_STEPS=1`, `models/wan21_t2v_1_3b/optimized/cache_runtime.py`).

/// Published canvas of the speedup claim.
pub const PUBLISHED_HEIGHT: usize = 480;
pub const PUBLISHED_WIDTH: usize = 832;
pub const PUBLISHED_FRAMES: usize = 81;
pub const PUBLISHED_STEPS: usize = 50;
/// Pipeline default `guidance_scale` (the run script's value is unpublished).
pub const PUBLISHED_GUIDANCE: f32 = 6.0;
pub const PUBLISHED_SPEEDUP: f64 = 2.77;
pub const PUBLISHED_EASYCACHE_THRESHOLD: f64 = 0.1;
/// Not published for SANA; Sol-Engine's EasyCache runtime defaults.
pub const EASYCACHE_RETAIN_STEPS: usize = 7;
pub const EASYCACHE_COOLDOWN_STEPS: usize = 1;

/// The optimized-arm switches the port implements. `compile` has no Rust
/// counterpart (the kernels are hand-fused); it is recorded, not applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SanaOptimizations {
    /// EasyCache threshold; `None` = every step computes.
    pub easycache: Option<f64>,
    /// One fused GEMM for self-attention q/k/v.
    pub qkv_merge: bool,
    /// Linear-attention products on bf16-rounded operands.
    pub linattn_bf16: bool,
}

impl SanaOptimizations {
    pub const BASELINE: Self = Self {
        easycache: None,
        qkv_merge: false,
        linattn_bf16: false,
    };

    /// `--easycache 0.1 --linattn-bf16 --qkv-merge --compile`.
    pub const FULL: Self = Self {
        easycache: Some(PUBLISHED_EASYCACHE_THRESHOLD),
        qkv_merge: true,
        linattn_bf16: true,
    };

    /// From `FASTVIDEO_SANA_OPT` (`baseline` | `full`), then the per-switch
    /// overrides `FASTVIDEO_SANA_EASYCACHE` (threshold, `0` = off),
    /// `FASTVIDEO_SANA_QKV_MERGE`, `FASTVIDEO_SANA_LINATTN_BF16` (`0`/`1`).
    pub fn parse(
        arm: Option<&str>,
        easycache: Option<&str>,
        qkv_merge: Option<&str>,
        linattn_bf16: Option<&str>,
    ) -> Result<Self, String> {
        let mut o = match arm.map(str::trim).unwrap_or("") {
            "" | "baseline" | "off" => Self::BASELINE,
            "full" | "fullopt" | "sol" => Self::FULL,
            other => return Err(format!("FASTVIDEO_SANA_OPT={other:?}: baseline | full")),
        };
        if let Some(t) = easycache.map(str::trim).filter(|s| !s.is_empty()) {
            let v: f64 = t
                .parse()
                .map_err(|_| format!("FASTVIDEO_SANA_EASYCACHE={t:?} is not a number"))?;
            o.easycache = (v > 0.0).then_some(v);
        }
        let flag = |name: &str, v: Option<&str>, cur: bool| -> Result<bool, String> {
            match v.map(str::trim) {
                None | Some("") => Ok(cur),
                Some("1" | "true" | "on") => Ok(true),
                Some("0" | "false" | "off") => Ok(false),
                Some(x) => Err(format!("{name}={x:?}: 0 | 1")),
            }
        };
        o.qkv_merge = flag("FASTVIDEO_SANA_QKV_MERGE", qkv_merge, o.qkv_merge)?;
        o.linattn_bf16 = flag("FASTVIDEO_SANA_LINATTN_BF16", linattn_bf16, o.linattn_bf16)?;
        Ok(o)
    }

    pub fn from_env() -> Result<Self, String> {
        let get = |k: &str| std::env::var(k).ok();
        Self::parse(
            get("FASTVIDEO_SANA_OPT").as_deref(),
            get("FASTVIDEO_SANA_EASYCACHE").as_deref(),
            get("FASTVIDEO_SANA_QKV_MERGE").as_deref(),
            get("FASTVIDEO_SANA_LINATTN_BF16").as_deref(),
        )
    }
}

/// Diffusers `ASPECT_RATIO_480_BIN` (`sample_size == 30`).
pub const ASPECT_RATIO_480_BIN: [(f64, usize, usize); 11] = [
    (0.5, 448, 896),
    (0.57, 480, 832),
    (0.68, 528, 768),
    (0.78, 560, 720),
    (1.0, 624, 624),
    (1.13, 672, 592),
    (1.29, 720, 560),
    (1.46, 768, 528),
    (1.67, 816, 496),
    (1.75, 832, 480),
    (2.0, 896, 448),
];

/// Diffusers `ASPECT_RATIO_720_BIN` (`sample_size == 22`).
pub const ASPECT_RATIO_720_BIN: [(f64, usize, usize); 11] = [
    (0.5, 672, 1344),
    (0.57, 704, 1280),
    (0.68, 800, 1152),
    (0.78, 832, 1088),
    (1.0, 960, 960),
    (1.13, 1024, 896),
    (1.29, 1088, 832),
    (1.46, 1152, 800),
    (1.67, 1248, 736),
    (1.75, 1280, 704),
    (2.0, 1344, 672),
];

/// `VideoProcessor.classify_height_width_bin`: the bin whose ratio key is
/// closest to `height / width`. Returns the binned `(height, width)`.
pub fn classify_height_width_bin(height: usize, width: usize, sample_size: usize) -> (usize, usize) {
    let bins: &[(f64, usize, usize)] = if sample_size == 22 {
        &ASPECT_RATIO_720_BIN
    } else {
        &ASPECT_RATIO_480_BIN
    };
    let ar = height as f64 / width as f64;
    let mut best = bins[0];
    for &b in bins {
        if (b.0 - ar).abs() < (best.0 - ar).abs() {
            best = b;
        }
    }
    (best.1, best.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arms_parse() {
        assert_eq!(
            SanaOptimizations::parse(None, None, None, None).unwrap(),
            SanaOptimizations::BASELINE
        );
        assert_eq!(
            SanaOptimizations::parse(Some("full"), None, None, None).unwrap(),
            SanaOptimizations::FULL
        );
        let o = SanaOptimizations::parse(Some("full"), Some("0"), Some("0"), None).unwrap();
        assert_eq!(o.easycache, None);
        assert!(!o.qkv_merge && o.linattn_bf16);
        assert!(SanaOptimizations::parse(Some("fast"), None, None, None).is_err());
        assert!(SanaOptimizations::parse(None, Some("x"), None, None).is_err());
    }

    #[test]
    fn published_canvas_is_its_own_bin() {
        assert_eq!(classify_height_width_bin(480, 832, 30), (480, 832));
        assert_eq!(classify_height_width_bin(832, 480, 30), (832, 480));
        assert_eq!(classify_height_width_bin(720, 1280, 22), (704, 1280));
    }
}
