//! Device residency policy for the cudarc Wan graph.
//!
//! Residency is **always on** when a global CUDA device is live: every op
//! runs on the device and weights are uploaded once at load. `FASTVIDEO_RESIDENT=0`
//! (or `false`) is refused when a device context is created
//! (`DeviceContext::new`), so a GPU run can never quietly execute the host
//! reference implementations.

use super::envflag::CachedBool;

static RESIDENT_CACHE: CachedBool = CachedBool::new();

/// Whether device residency is enabled (default **true**). Cached after first
/// read: consulted from nearly every op.
pub fn residency_enabled() -> bool {
    RESIDENT_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_RESIDENT", true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residency_default_on() {
        let prev = std::env::var("FASTVIDEO_RESIDENT").ok();
        std::env::remove_var("FASTVIDEO_RESIDENT");
        RESIDENT_CACHE.reset();
        assert!(residency_enabled());
        std::env::set_var("FASTVIDEO_RESIDENT", "0");
        RESIDENT_CACHE.reset();
        assert!(!residency_enabled());
        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_RESIDENT", v),
            None => std::env::remove_var("FASTVIDEO_RESIDENT"),
        }
        RESIDENT_CACHE.reset();
    }
}
