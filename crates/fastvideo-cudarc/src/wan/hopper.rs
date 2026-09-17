//! Hopper (sm_90) / Ampere+ performance policy for cudarc Wan.
//!
//! Defaults target maximum throughput on H100/H200:
//! - TF32 Tensor Core math (disable with `FASTVIDEO_TF32=0`)
//! - Larger SDPA query chunks on 80GB-class GPUs
//! - NVRTC `--gpu-architecture=compute_90` when SM ≥ 90
//! - Device-side UniPC/Euler (disable with `FASTVIDEO_DEVICE_SCHED=0`)

use super::envflag::CachedBool;

static TF32_CACHE: CachedBool = CachedBool::new();
static DEVICE_SCHED_CACHE: CachedBool = CachedBool::new();

/// True when TF32 Tensor Core math should be used for F32 GEMMs (default on).
/// Read once and cached: this is consulted on every GEMM dispatch, so a raw
/// `std::env::var` per call would add a lock+allocate to the hottest path.
pub fn tf32_enabled() -> bool {
    TF32_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_TF32", true))
}

/// Keep scheduler updates on device (default on). Cached (see [`tf32_enabled`]).
pub fn device_sched_enabled() -> bool {
    DEVICE_SCHED_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_DEVICE_SCHED", true))
}

/// Hopper / Ada / Ampere class (≥ sm_80) for Tensor Core TF32.
pub fn is_tensor_core_gpu(sm_major: i32) -> bool {
    sm_major >= 8
}

pub fn is_hopper(sm_major: i32) -> bool {
    sm_major >= 9
}

/// NVRTC arch string for the live device, if we should pin it.
pub fn nvrtc_arch(sm_major: i32, sm_minor: i32) -> Option<&'static str> {
    match (sm_major, sm_minor) {
        (9, _) => Some("compute_90"),
        (8, 9) => Some("compute_89"),
        (8, _) => Some("compute_80"),
        (7, 5) => Some("compute_75"),
        _ => None,
    }
}

/// SDPA query chunk: larger on Hopper 80GB to cut launch/memcpy overhead.
/// Called once per attention forward (every block, every step), so the env
/// override is cached; there is only ever one live device per process, so
/// caching by `sm_major` isn't necessary in practice.
pub fn sdpa_query_chunk(sm_major: i32) -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        if let Ok(v) = std::env::var("FASTVIDEO_SDPA_CHUNK") {
            if let Ok(n) = v.parse::<usize>() {
                return n.max(64);
            }
        }
        if is_hopper(sm_major) {
            1024
        } else if sm_major >= 8 {
            512
        } else {
            256
        }
    })
}
