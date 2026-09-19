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
/// NVRTC targets for a device, native first, then what an older NVRTC can
/// still take. `nvrtc_arch` is the first entry.
///
/// An empty list would mean compiling with no `-arch` at all, which NVRTC
/// treats as compute_52: `__CUDA_ARCH__ = 520`, every `#if __CUDA_ARCH__ >= 800`
/// body compiled out, and PTX that JITs onto the device with no error. That is
/// exactly what happened on Blackwell before sm12 was mapped; the tensor-core
/// VSA kernel ran as a stub and reported garbage.
///
/// NVRTC 13 knows compute_100/120 natively. A box still on libnvrtc 12.x
/// rejects those, and the compile falls back to compute_90 — PTX is
/// forward-compatible, so sm100/sm120 JIT from it with every sm80+ instruction
/// intact, just without arch-specific codegen. Embedded cubins from build.rs
/// bypass all of this on a device that has one.
pub fn nvrtc_arches(sm_major: i32, sm_minor: i32) -> &'static [&'static str] {
    match (sm_major, sm_minor) {
        (12, _) => &["compute_120", "compute_90"],
        (10, _) => &["compute_100", "compute_90"],
        (9, _) => &["compute_90"],
        (8, 9) => &["compute_89"],
        (8, _) => &["compute_80"],
        (7, 5) => &["compute_75"],
        // Anything newer than we know about takes the forward-compatible
        // Hopper PTX rather than falling through to no arch.
        (m, _) if m > 12 => &["compute_120", "compute_90"],
        _ => &[],
    }
}

pub fn nvrtc_arch(sm_major: i32, sm_minor: i32) -> Option<&'static str> {
    nvrtc_arches(sm_major, sm_minor).first().copied()
}

#[cfg(test)]
mod arch_tests {
    use super::{nvrtc_arch, nvrtc_arches};

    /// No supported device may map to "no arch": that silently compiles
    /// every guarded kernel body out. Pinned so the mapping cannot regress.
    #[test]
    fn every_supported_sm_has_a_native_target_and_a_fallback() {
        assert_eq!(nvrtc_arches(12, 0), &["compute_120", "compute_90"]);
        assert_eq!(nvrtc_arches(12, 1), &["compute_120", "compute_90"]);
        assert_eq!(nvrtc_arches(10, 0), &["compute_100", "compute_90"]);
        assert_eq!(nvrtc_arches(9, 0), &["compute_90"]);
        assert_eq!(nvrtc_arch(8, 9), Some("compute_89"));
        assert_eq!(nvrtc_arch(8, 6), Some("compute_80"));
        assert_eq!(nvrtc_arch(7, 5), Some("compute_75"));
        assert_eq!(nvrtc_arch(13, 0), Some("compute_120"), "unknown future SM must not be None");
        assert_eq!(nvrtc_arch(7, 0), None, "Volta is genuinely unsupported");
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
