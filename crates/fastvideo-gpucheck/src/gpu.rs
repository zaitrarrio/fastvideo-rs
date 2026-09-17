//! Device selection, identity, and VRAM sampling.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub spec: String,
    pub name: Option<String>,
    pub compute_capability: Option<(i32, i32)>,
    pub total_mib: Option<u64>,
    pub free_mib: Option<u64>,
}

pub fn on_gpu(spec: &str) -> bool {
    spec.trim().to_ascii_lowercase().starts_with("cuda")
}

/// Initialize the cudarc global device (compiles all NVRTC kernels on CUDA).
pub fn init(spec: &str) -> anyhow::Result<DeviceInfo> {
    fastvideo_cudarc::resolve_device(spec)
        .map_err(|e| anyhow::anyhow!("resolve_device({spec}): {e}"))?;
    Ok(info(spec))
}

#[cfg(feature = "cuda")]
pub fn info(spec: &str) -> DeviceInfo {
    let Some(dev) = fastvideo_cudarc::wan::device::global_device() else {
        return cpu_info(spec);
    };
    let (free, total) = mem_info().unwrap_or((0, 0));
    DeviceInfo {
        spec: spec.to_string(),
        name: dev.ctx.name().ok(),
        compute_capability: Some((dev.sm_major, dev.sm_minor)),
        total_mib: Some(total / (1 << 20)),
        free_mib: Some(free / (1 << 20)),
    }
}

#[cfg(not(feature = "cuda"))]
pub fn info(spec: &str) -> DeviceInfo {
    cpu_info(spec)
}

fn cpu_info(spec: &str) -> DeviceInfo {
    DeviceInfo {
        spec: spec.to_string(),
        name: None,
        compute_capability: None,
        total_mib: None,
        free_mib: None,
    }
}

/// `(free_bytes, total_bytes)` for the global device, if one is live.
#[cfg(feature = "cuda")]
pub fn mem_info() -> Option<(u64, u64)> {
    let dev = fastvideo_cudarc::wan::device::global_device()?;
    dev.ctx.bind_to_thread().ok()?;
    let (free, total) = cudarc::driver::result::mem_get_info().ok()?;
    Some((free as u64, total as u64))
}

#[cfg(not(feature = "cuda"))]
pub fn mem_info() -> Option<(u64, u64)> {
    None
}

/// Background sampler recording peak device memory in use (MiB) between
/// `start` and `stop`. Allocator caching means this is an upper bound on
/// what the workload needs, which is the right side to err on for OOM gates.
pub struct PeakMem {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak: std::sync::Arc<std::sync::atomic::AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakMem {
    pub fn start() -> Self {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let handle = mem_info().map(|_| {
            let stop = stop.clone();
            let peak = peak.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Some((free, total)) = mem_info() {
                        let used = (total - free) / (1 << 20);
                        peak.fetch_max(used, Ordering::Relaxed);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            })
        });
        Self { stop, peak, handle }
    }

    /// Peak MiB in use, or `None` without a live device.
    pub fn stop(mut self) -> Option<u64> {
        use std::sync::atomic::Ordering;
        self.stop.store(true, Ordering::Relaxed);
        let handle = self.handle.take()?;
        let _ = handle.join();
        if let Some((free, total)) = mem_info() {
            self.peak.fetch_max((total - free) / (1 << 20), Ordering::Relaxed);
        }
        Some(self.peak.load(Ordering::Relaxed))
    }
}
