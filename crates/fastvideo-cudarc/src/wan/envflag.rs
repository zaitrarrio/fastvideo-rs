//! Shared parsing for `FASTVIDEO_*` boolean env flags.
//!
//! Every flag getter (`residency_enabled`, `bf16_enabled`, `tf32_enabled`, …)
//! is consulted from per-tensor-op hot paths — potentially tens of thousands
//! of times per `generate()` call. `std::env::var` is not free (it takes a
//! process-wide lock and allocates a `String`), so each getter caches its
//! parsed result behind a local `OnceLock<bool>` and only calls into here
//! once, on first use. This module holds just the shared parse logic so the
//! "0/false/off" and "1/true" spellings stay consistent across all flags.

/// Parse a boolean env flag that defaults to `true` when unset (`FASTVIDEO_RESIDENT`,
/// `FASTVIDEO_BF16`, `FASTVIDEO_TF32`, `FASTVIDEO_TWO_STREAMS`, …): explicit
/// `0` / `false` / `off` (case-insensitive, trimmed) turn it off, anything
/// else (including unset) leaves it on.
pub fn bool_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim();
            if default {
                !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
            } else {
                v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
            }
        }
        Err(_) => default,
    }
}

/// Parse a `usize` env flag, falling back to `default` when unset or invalid.
pub fn f64_flag(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

pub fn usize_flag(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Parse a `f32` env flag, falling back to `default` when unset or invalid.
pub fn f32_flag(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(default)
}

/// Read a string env flag, lower-cased, falling back to `default` when unset.
pub fn string_flag(name: &str, default: &'static str) -> String {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .to_ascii_lowercase()
}

/// A once-computed `bool` cache with a relaxed-atomic fast path.
///
/// Plain `OnceLock<bool>` would work for the hot-path read but can't be
/// reset, and several existing tests mutate `FASTVIDEO_*` env vars at
/// runtime and expect the getter to observe the new value immediately.
/// `CachedBool` supports that via `reset()` (test-only), while the normal
/// `get_or_init` path is a single relaxed atomic load once warm — cheaper
/// than a `OnceLock` bool check and far cheaper than re-parsing the env var.
///
/// Declare one `static` per flag at module scope (not inside the getter fn)
/// so tests in the same module can call `.reset()` on it directly.
pub struct CachedBool(std::sync::atomic::AtomicU8);

impl CachedBool {
    const UNSET: u8 = 0;
    const FALSE: u8 = 1;
    const TRUE: u8 = 2;

    pub const fn new() -> Self {
        Self(std::sync::atomic::AtomicU8::new(Self::UNSET))
    }

    /// Return the cached value, computing (and caching) it on first call.
    pub fn get_or_init(&self, compute: impl FnOnce() -> bool) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        match self.0.load(Relaxed) {
            Self::TRUE => true,
            Self::FALSE => false,
            _ => {
                let v = compute();
                self.0
                    .store(if v { Self::TRUE } else { Self::FALSE }, Relaxed);
                v
            }
        }
    }

    /// Clear the cache so the next `get_or_init` re-reads the environment.
    /// Test-only: production code should never need to invalidate a flag
    /// mid-process.
    #[cfg(test)]
    pub fn reset(&self) {
        self.0
            .store(Self::UNSET, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for CachedBool {
    fn default() -> Self {
        Self::new()
    }
}

/// Same idea as [`CachedBool`] for `String`-valued flags (e.g. `FASTVIDEO_SDPA`).
/// A `Mutex<Option<String>>` is still far cheaper than re-parsing the env on
/// every call — this is consulted per attention call, not per element.
pub struct CachedString(std::sync::Mutex<Option<String>>);

impl CachedString {
    pub const fn new() -> Self {
        Self(std::sync::Mutex::new(None))
    }

    pub fn get_or_init(&self, compute: impl FnOnce() -> String) -> String {
        let mut guard = self.0.lock().expect("cached string lock");
        if let Some(v) = guard.as_ref() {
            return v.clone();
        }
        let v = compute();
        *guard = Some(v.clone());
        v
    }

    #[cfg(test)]
    pub fn reset(&self) {
        *self.0.lock().expect("cached string lock") = None;
    }
}

impl Default for CachedString {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_flag_default_on() {
        let key = "FASTVIDEO_TEST_FLAG_ON";
        std::env::remove_var(key);
        assert!(bool_flag(key, true));
        std::env::set_var(key, "0");
        assert!(!bool_flag(key, true));
        std::env::set_var(key, "false");
        assert!(!bool_flag(key, true));
        std::env::set_var(key, "anything-else");
        assert!(bool_flag(key, true));
        std::env::remove_var(key);
    }

    #[test]
    fn bool_flag_default_off() {
        let key = "FASTVIDEO_TEST_FLAG_OFF";
        std::env::remove_var(key);
        assert!(!bool_flag(key, false));
        std::env::set_var(key, "1");
        assert!(bool_flag(key, false));
        std::env::set_var(key, "true");
        assert!(bool_flag(key, false));
        std::env::remove_var(key);
    }
}
