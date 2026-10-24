//! The device facts a loader's plan decides on (`docs/adr/0017-loader-stages.md`).
//!
//! The H3 and LTX-2 loads run in four stages: **plan** (pure: what to load,
//! where, at what precision, with which fusions), **open** (map the
//! checkpoint files, queue their read-ahead), **place** (upload, quantize and
//! offload onto the device) and **fuse** (the passes over the placed model).
//! The plan reads no device: what it needs to know about the card is measured
//! once, before it, into a [`DeviceFacts`] (or given, by a test or a dry run),
//! and every decision the plan makes is a function of those facts, the
//! configuration and the settings snapshot.

/// What a load's plan needs to know about the device: measured once when the
/// load starts ([`DeviceFacts::measure`]), or given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceFacts {
    /// Free device memory in bytes (after `FASTVIDEO_DEVICE_BUDGET_GIB`) when
    /// the load starts ([`super::device::free_memory`]); `None` without a
    /// device.
    pub free: Option<u64>,
    /// The global device's compute capability `(major, minor)`, whether or
    /// not ops run on it; `None` without one.
    pub device_sm: Option<(i32, i32)>,
    /// Ops are expected to run on the device: a live device with residency
    /// on (`stats::device_expected`).
    pub device_expected: bool,
    /// bf16 activations ([`super::tensor::bf16_activations`]), after the
    /// load's pipeline default ([`super::tensor::default_bf16_activations`]).
    pub bf16_activations: bool,
    /// How a `Linear` loads its weights, as a pre-quantized DiT tree records
    /// it: `"device-bf16"`, `"f32"`, or `None` when the route has no stored
    /// form (process-wide `FASTVIDEO_FP8`, or f32 GEMM math on a device).
    pub linear_route: Option<&'static str>,
}

impl DeviceFacts {
    /// The facts of this process's device now. Reads the device and the
    /// once-per-process getters it depends on; allocates nothing.
    pub fn measure() -> Self {
        #[cfg(feature = "cuda")]
        let device_sm = super::device::global_device().map(|d| (d.sm_major, d.sm_minor));
        #[cfg(not(feature = "cuda"))]
        let device_sm = None;
        Self {
            free: super::device::free_memory().map(|(free, _)| free),
            device_sm,
            device_expected: super::stats::device_expected(),
            bf16_activations: super::tensor::bf16_activations(),
            linear_route: linear_route(),
        }
    }

    /// A CPU run: no device, f32 activations, f32 linears.
    pub fn cpu() -> Self {
        Self {
            free: None,
            device_sm: None,
            device_expected: false,
            bf16_activations: false,
            linear_route: Some("f32"),
        }
    }

    /// A live device of compute capability `sm` with `free` bytes free, bf16
    /// activations and bf16 device linears (the served configuration).
    pub fn device(sm: (i32, i32), free: u64) -> Self {
        Self {
            free: Some(free),
            device_sm: Some(sm),
            device_expected: true,
            bf16_activations: true,
            linear_route: Some("device-bf16"),
        }
    }

    /// The compute capability major of the device ops run on (`None` when
    /// they run on the host), as the runtime's precision rules read it.
    pub fn live_sm_major(&self) -> Option<i32> {
        if self.device_expected {
            self.device_sm.map(|(major, _)| major)
        } else {
            None
        }
    }

    /// One line for logs and goldens.
    pub fn describe(&self) -> String {
        let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
        format!(
            "free={} sm={} device_expected={} bf16_act={} linear_route={}",
            self.free
                .map_or("unknown".to_owned(), |f| format!("{:.1}GiB", gib(f))),
            self.device_sm
                .map_or("none".to_owned(), |(a, b)| format!("{a}{b}")),
            self.device_expected,
            self.bf16_activations,
            self.linear_route.unwrap_or("not-stored"),
        )
    }
}

/// [`DeviceFacts::linear_route`] of this process now.
pub fn linear_route() -> Option<&'static str> {
    #[cfg(feature = "cuda")]
    {
        if super::nn::Linear::device_bf16_route() {
            Some("device-bf16")
        } else if super::stats::device_expected() {
            // Process-wide FASTVIDEO_FP8 or f32 GEMM math: not stored.
            None
        } else {
            Some("f32")
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Some("f32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_runs_have_no_live_capability() {
        assert_eq!(DeviceFacts::cpu().live_sm_major(), None);
        let mut f = DeviceFacts::device((12, 0), 1 << 30);
        assert_eq!(f.live_sm_major(), Some(12));
        f.device_expected = false;
        assert_eq!(
            f.live_sm_major(),
            None,
            "a device without residency runs ops on the host"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn a_cpu_build_measures_no_device() {
        let f = DeviceFacts::measure();
        assert_eq!(
            (f.free, f.device_sm, f.device_expected, f.linear_route),
            (None, None, false, Some("f32"))
        );
    }
}
