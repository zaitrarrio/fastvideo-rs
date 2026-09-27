//! Wan Sol-engine cache and attention-route helpers on the cudarc path.
//!
//! Policy lives in [`fastvideo_models::wan::sol_cache`] and
//! [`fastvideo_models::wan::sol`]. This module applies those contracts:
//! profile-selected EasyCache knobs, the A14B block-0 / tail-39 controller,
//! and the host Morton3D gather used only on the 14B Sol route.

use fastvideo_models::wan::sol::{morton3d_inverse, morton3d_perm};
use fastvideo_models::wan::sol_cache::{
    A14bCacheController, EasyCacheProfile, A14B_EASY_THRESHOLD, A14B_MAX_REUSE, A14B_START_STEP,
    A14B_TAIL_STEPS,
};

use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Whole-stack EasyCache is the 5B / 14B path. A14B uses
/// [`A14bCacheController`] (block-0 prefix + blocks 1..=39 residual).
pub fn uses_a14b_cache(is_moe: bool) -> bool {
    is_moe
}

pub fn easy_cache_profile() -> Result<EasyCacheProfile> {
    EasyCacheProfile::from_env().map_err(msg)
}

/// Resolved EasyCache knobs: profile first, then explicit env overrides.
pub fn easy_cache_params() -> Result<(f64, usize, usize)> {
    let (thr, retain, cooldown) = easy_cache_profile()?.params();
    let threshold = env_f64(
        "FASTVIDEO_WAN_EASYCACHE_THRESH",
        "WAN22_EASYCACHE_THRESHOLD",
        thr,
    );
    let retain = env_usize(
        "FASTVIDEO_WAN_EASYCACHE_RETAIN",
        "WAN22_EASYCACHE_RETAIN_STEPS",
        retain,
    );
    let cooldown = env_usize(
        "FASTVIDEO_WAN_EASYCACHE_COOLDOWN",
        "WAN22_EASYCACHE_COOLDOWN_STEPS",
        cooldown,
    );
    Ok((threshold, retain, cooldown))
}

pub fn a14b_controller(num_steps: usize) -> Result<A14bCacheController> {
    A14bCacheController::official(num_steps).map_err(msg)
}

fn env_f64(primary: &str, fallback: &str, default: f64) -> f64 {
    std::env::var(primary)
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var(fallback).ok().and_then(|s| s.parse().ok()))
        .unwrap_or(default)
}

fn env_usize(primary: &str, fallback: &str, default: usize) -> usize {
    std::env::var(primary)
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var(fallback).ok().and_then(|s| s.parse().ok()))
        .unwrap_or(default)
}

/// Gather BHSD along the sequence axis. Host path; pins when `like` is device-fresh.
pub fn gather_bhsd_seq(t: &CudaTensor, perm: &[usize]) -> Result<CudaTensor> {
    if t.rank() != 4 {
        return Err(msg(format!(
            "wan morton reorder expects BHSD, got {:?}",
            t.shape
        )));
    }
    let (b, h, s, d) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    if perm.len() != s {
        return Err(msg(format!(
            "wan morton perm {} does not match sequence {s}",
            perm.len()
        )));
    }
    if perm.iter().any(|&i| i >= s) {
        return Err(msg("wan morton perm is out of range"));
    }
    let host = t.host_cow()?;
    let mut out = vec![0.0f32; b * h * s * d];
    for bi in 0..b {
        for hi in 0..h {
            let plane = (bi * h + hi) * s * d;
            for (dst, &src) in perm.iter().enumerate() {
                let to = plane + dst * d;
                let from = plane + src * d;
                out[to..to + d].copy_from_slice(&host[from..from + d]);
            }
        }
    }
    pin_like(CudaTensor::from_vec(out, t.shape.clone())?, t)
}

/// `(sum |a - b|, sum |b|, n)` for the cache decision metrics. On the device
/// one reduction kernel runs and only the two f64 sums come back (as
/// `h3::transformer::tea_relative_l1`); the host loop is the CPU path.
pub fn abs_sums(a: &CudaTensor, b: &CudaTensor) -> Result<(f64, f64, usize)> {
    if a.shape != b.shape {
        return Err(msg(format!(
            "wan cache metric shapes differ: {:?} != {:?}",
            a.shape, b.shape
        )));
    }
    let n: usize = a.shape.iter().product();
    #[cfg(feature = "cuda")]
    if let (Some(x), Some(y)) = (a.dev()?, b.dev()?) {
        let (diff, prev) = super::ops::abs_diff_sums_device(&x, &y)?;
        return Ok((diff, prev, n));
    }
    let (x, y) = (a.host_cow()?, b.host_cow()?);
    let mut diff = 0.0f64;
    let mut prev = 0.0f64;
    for (p, q) in x.iter().zip(y.iter()) {
        diff += (f64::from(*p) - f64::from(*q)).abs();
        prev += f64::from(*q).abs();
    }
    Ok((diff, prev, n))
}

/// `mean |t|`.
pub fn mean_abs(t: &CudaTensor) -> Result<f64> {
    let (_, sum, n) = abs_sums(t, t)?;
    Ok(sum / n.max(1) as f64)
}

/// `mean |a - b|`.
pub fn mean_abs_delta(a: &CudaTensor, b: &CudaTensor) -> Result<f64> {
    let (diff, _, n) = abs_sums(a, b)?;
    Ok(diff / n.max(1) as f64)
}

/// Morton3D reorder (or its inverse) of a BHSD tensor along the sequence
/// for a `(frames, height, width)` token grid. On the device: one row
/// gather over an index buffer built and uploaded once per grid and
/// `batch * heads`; the host gather is the CPU path.
pub fn morton_gather(
    t: &CudaTensor,
    grid: (usize, usize, usize),
    inverse: bool,
) -> Result<CudaTensor> {
    if t.rank() != 4 {
        return Err(msg(format!(
            "wan morton reorder expects BHSD, got {:?}",
            t.shape
        )));
    }
    let (b, h, s, d) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    if grid.0 * grid.1 * grid.2 != s {
        return Err(msg(format!(
            "wan morton grid {grid:?} does not match sequence {s}"
        )));
    }
    #[cfg(feature = "cuda")]
    if let Some(table) = t.dev()? {
        type Key = ((usize, usize, usize), usize, bool);
        static INDEX: std::sync::OnceLock<
            std::sync::Mutex<
                std::collections::HashMap<Key, std::sync::Arc<cudarc::driver::CudaSlice<u32>>>,
            >,
        > = std::sync::OnceLock::new();
        let key = (grid, b * h, inverse);
        let map = INDEX.get_or_init(Default::default);
        let idx = {
            let mut m = map.lock().expect("morton index");
            match m.get(&key) {
                Some(i) => i.clone(),
                None => {
                    let (perm, inv) = morton3d_pair(grid.0, grid.1, grid.2);
                    let order = if inverse { inv } else { perm };
                    let mut rows = Vec::with_capacity(b * h * s);
                    for plane in 0..b * h {
                        rows.extend(order.iter().map(|&src| (plane * s + src) as u32));
                    }
                    let up = std::sync::Arc::new(super::ops::upload_row_indices(&rows)?);
                    m.insert(key, up.clone());
                    up
                }
            }
        };
        let out = super::ops::index_select_rows_idx_device(&table, d, &idx)?;
        return CudaTensor::from_device_slice(out, t.shape.clone());
    }
    let (perm, inv) = morton3d_pair(grid.0, grid.1, grid.2);
    gather_bhsd_seq(t, if inverse { &inv } else { &perm })
}

/// Morton3D gather + inverse for the 14B Sol route.
pub fn morton3d_pair(frames: usize, height: usize, width: usize) -> (Vec<usize>, Vec<usize>) {
    let perm = morton3d_perm(frames, height, width);
    let inverse = morton3d_inverse(&perm);
    (perm, inverse)
}

fn pin_like(t: CudaTensor, like: &CudaTensor) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    {
        let mut t = t;
        if like.is_device_fresh() {
            t.pin_device()?;
        }
        return Ok(t);
    }
    let _ = like;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::wan::sol_cache::{A14bExpert, EasyCache};

    #[test]
    fn a14b_is_not_the_whole_stack_easycache() {
        assert!(uses_a14b_cache(true));
        assert!(!uses_a14b_cache(false));
        let a14b = a14b_controller(40).unwrap();
        assert!((a14b.threshold - A14B_EASY_THRESHOLD).abs() < 1e-12);
        assert_eq!(
            (a14b.start_step, a14b.tail_steps, a14b.max_reuse),
            (A14B_START_STEP, A14B_TAIL_STEPS, A14B_MAX_REUSE)
        );
        let whole = EasyCache::official(40).unwrap();
        assert_ne!(a14b.threshold, whole.threshold);
        assert_ne!(a14b.start_step, whole.retain_steps);
        assert_eq!(A14bCacheController::placement(), (1, 39, 40));
        let mut cache = a14b;
        for expert in [A14bExpert::HighNoise, A14bExpert::LowNoise] {
            cache.decide(expert, 0, 0.0);
            cache.note_computed(expert, 1.0, 0.0, 1.0);
            cache.decide(expert, 5, 0.0);
            cache.note_computed(expert, 1.0, 0.10, 1.0);
        }
        let high = cache.decide(A14bExpert::HighNoise, 6, 2.0);
        let low = cache.decide(A14bExpert::LowNoise, 6, 0.5);
        assert!(!high.compute);
        assert!(!low.compute);
        assert!((high.estimate.unwrap() - 0.20).abs() < 1e-12);
        assert!((low.estimate.unwrap() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn easycache_profile_defaults_stay_on_code_default() {
        let saved = std::env::var("FASTVIDEO_WAN_EASYCACHE_PROFILE").ok();
        std::env::remove_var("FASTVIDEO_WAN_EASYCACHE_PROFILE");
        std::env::remove_var("WAN22_EASYCACHE_PROFILE");
        let (thr, retain, cooldown) = easy_cache_params().unwrap();
        assert_eq!((thr, retain, cooldown), (0.05, 7, 1));
        std::env::set_var("FASTVIDEO_WAN_EASYCACHE_PROFILE", "fullstack");
        assert_eq!(easy_cache_params().unwrap(), (0.036, 7, 1));
        std::env::set_var("FASTVIDEO_WAN_EASYCACHE_PROFILE", "14b-tuned");
        assert_eq!(easy_cache_params().unwrap(), (0.10, 5, 1));
        std::env::set_var("FASTVIDEO_WAN_EASYCACHE_THRESH", "0.2");
        let (thr, retain, _) = easy_cache_params().unwrap();
        assert!((thr - 0.2).abs() < 1e-12);
        assert_eq!(retain, 5);
        std::env::remove_var("FASTVIDEO_WAN_EASYCACHE_THRESH");
        std::env::remove_var("FASTVIDEO_WAN_EASYCACHE_PROFILE");
        match saved {
            Some(v) => std::env::set_var("FASTVIDEO_WAN_EASYCACHE_PROFILE", v),
            None => std::env::remove_var("FASTVIDEO_WAN_EASYCACHE_PROFILE"),
        }
    }

    #[test]
    fn morton3d_bhsd_gather_roundtrips() {
        let (f, h, w) = (2usize, 2, 2);
        let (perm, inverse) = morton3d_pair(f, h, w);
        let seq = f * h * w;
        let (b, heads, dim) = (1usize, 2usize, 3usize);
        let data: Vec<f32> = (0..b * heads * seq * dim).map(|i| i as f32).collect();
        let t = CudaTensor::from_vec(data.clone(), vec![b, heads, seq, dim]).unwrap();
        let reordered = gather_bhsd_seq(&t, &perm).unwrap();
        let back = gather_bhsd_seq(&reordered, &inverse).unwrap();
        assert_eq!(back.host_cow().unwrap().as_ref(), data.as_slice());
        // First Morton token is the original token at perm[0].
        let got = reordered.host_cow().unwrap();
        let src = perm[0] * dim;
        assert_eq!(&got[0..dim], &data[src..src + dim]);
    }
}
