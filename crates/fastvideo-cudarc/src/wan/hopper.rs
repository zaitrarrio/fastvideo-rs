//! Hopper (sm_90) / Ampere+ performance policy for cudarc Wan.
//!
//! Defaults target maximum throughput on H100/H200:
//! - TF32 Tensor Core math (disable with `FASTVIDEO_TF32=0`)
//! - Larger SDPA query chunks on 80GB-class GPUs
//! - NVRTC `--gpu-architecture=compute_90` when SM ≥ 90

use super::envflag::CachedBool;

static TF32_CACHE: CachedBool = CachedBool::new();

/// True when TF32 Tensor Core math should be used for F32 GEMMs (default on).
/// Read once and cached: this is consulted on every GEMM dispatch, so a raw
/// `std::env::var` per call would add a lock+allocate to the hottest path.
pub fn tf32_enabled() -> bool {
    TF32_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_TF32", true))
}

/// Hopper / Ada / Ampere class (≥ sm_80) for Tensor Core TF32.
pub fn is_tensor_core_gpu(sm_major: i32) -> bool {
    sm_major >= 8
}

pub fn is_hopper(sm_major: i32) -> bool {
    sm_major >= 9
}

/// NVRTC arch string for the live device, if we should pin it.
/// NVRTC target for a device. `None` would compile with no `-arch` at all,
/// which NVRTC treats as compute_52 — `__CUDA_ARCH__ = 520`, every
/// `#if __CUDA_ARCH__ >= 800` body compiled out, and the PTX then JITs onto
/// whatever the device is with no error. That is exactly what happened on
/// Blackwell before this mapped it: the tensor-core VSA kernel ran as a
/// stub and reported garbage.
///
/// libnvrtc 12.4, which the runtime image ships, tops out at compute_90; PTX
/// is forward-compatible, so sm100 and sm120 JIT from it correctly and keep
/// every sm80+ instruction (mma.sync, ldmatrix, cp.async).
pub fn nvrtc_arch(sm_major: i32, sm_minor: i32) -> Option<&'static str> {
    match (sm_major, sm_minor) {
        (m, _) if m >= 9 => Some("compute_90"),
        (8, 9) => Some("compute_89"),
        (8, _) => Some("compute_80"),
        (7, 5) => Some("compute_75"),
        _ => None,
    }
}

#[cfg(test)]
mod arch_tests {
    use super::nvrtc_arch;

    /// Blackwell must not fall through to "no arch": that silently compiles
    /// every guarded kernel body out. Pinned so the mapping cannot regress.
    #[test]
    fn hopper_and_later_target_compute_90() {
        assert_eq!(nvrtc_arch(9, 0), Some("compute_90"));
        assert_eq!(nvrtc_arch(10, 0), Some("compute_90"));
        assert_eq!(nvrtc_arch(12, 0), Some("compute_90"));
        assert_eq!(nvrtc_arch(12, 1), Some("compute_90"));
        assert_eq!(nvrtc_arch(8, 9), Some("compute_89"));
        assert_eq!(nvrtc_arch(8, 6), Some("compute_80"));
        assert_eq!(nvrtc_arch(7, 5), Some("compute_75"));
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
