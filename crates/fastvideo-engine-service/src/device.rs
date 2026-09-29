//! Startup capability check: can this GPU run a model at all?
//!
//! A worker must not report ready on a GPU its model cannot run on. The
//! incident behind this: an A100 (sm80) reported ready, then every H3 job
//! failed with "cuBLASLt has no tensorwise FP8 algorithm … on sm80". The
//! check runs before a model loads (CUDA backend: from the device's compute
//! capability and total memory, read without a context; fake backend: a
//! simulated [`DeviceProfile`]) and a model that fails it is marked failed
//! with the reason, so readiness is `Failed` and `/fv/v1/status` says why.
//!
//! The requirements per model ([`Requirements`]) are deliberately coarse:
//!
//! | Need | Minimum | Why |
//! |---|---|---|
//! | every model | sm80 (Ampere) | bf16 tensor-core GEMMs, the attention kernels |
//! | FP8 linears or an FP8 text encoder (H3 turbo / max / Sol-H3, `resident-fp8`) | sm89 (Ada, Hopper, Blackwell) | FP8 tensor cores (cuBLASLt tensorwise / MXFP8) |
//! | NVFP4 (the LTX draft profile) | sm100 (Blackwell) | FP4 tensor cores |
//! | memory | the DiT's weights on disk ≤ the device's total memory | the resident DiT alone must fit |

use serde::{Deserialize, Serialize};

/// What a GPU offers, as the capability check sees it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// Marketing name (for messages only).
    pub name: String,
    /// Compute capability `(major, minor)`: sm80 = `(8, 0)`.
    pub sm: (u32, u32),
    /// Total memory in bytes, when known.
    pub total_bytes: Option<u64>,
}

const GIB: u64 = 1 << 30;

impl DeviceProfile {
    pub fn new(name: impl Into<String>, sm: (u32, u32), total_bytes: Option<u64>) -> Self {
        Self { name: name.into(), sm, total_bytes }
    }
    /// NVIDIA A100 80 GB (sm80: bf16, no FP8).
    pub fn a100_80gb() -> Self {
        Self::new("NVIDIA A100 80GB", (8, 0), Some(80 * GIB))
    }
    /// NVIDIA L40S (sm89, 48 GB).
    pub fn l40s() -> Self {
        Self::new("NVIDIA L40S", (8, 9), Some(48 * GIB))
    }
    /// NVIDIA H100 80 GB (sm90).
    pub fn h100_80gb() -> Self {
        Self::new("NVIDIA H100 80GB HBM3", (9, 0), Some(80 * GIB))
    }
    /// NVIDIA B200 (sm100).
    pub fn b200() -> Self {
        Self::new("NVIDIA B200", (10, 0), Some(180 * GIB))
    }
    /// RTX PRO 6000 Blackwell Server Edition (sm120, 96 GB).
    pub fn rtx_pro_6000() -> Self {
        Self::new("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0), Some(96 * GIB))
    }

    /// A named profile (`a100`, `l40s`, `h100`, `b200`, `rtx-pro-6000`), or
    /// `sm<NN>[:<GiB>]` (e.g. `sm80:80`): the fake engine's simulated device
    /// (`engine.fake.device`, `FV_FAKE_DEVICE`).
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        Ok(match s.as_str() {
            "a100" | "a100-80gb" => Self::a100_80gb(),
            "l40s" => Self::l40s(),
            "h100" | "h100-80gb" => Self::h100_80gb(),
            "b200" => Self::b200(),
            "rtx-pro-6000" | "rtxpro6000" => Self::rtx_pro_6000(),
            other => {
                let bad = || format!("device profile {other:?}: expected a100, l40s, h100, b200, rtx-pro-6000 or sm<NN>[:<GiB>]");
                let rest = other.strip_prefix("sm").ok_or_else(bad)?;
                let (sm, mem) = match rest.split_once(':') {
                    Some((a, b)) => (a, Some(b.parse::<u64>().map_err(|_| bad())? * GIB)),
                    None => (rest, None),
                };
                let n: u32 = sm.parse().map_err(|_| bad())?;
                if !(50..=199).contains(&n) {
                    return Err(bad());
                }
                Self::new(format!("sm{n} (simulated)"), (n / 10, n % 10), mem)
            }
        })
    }

    fn sm_label(&self) -> String {
        format!("sm{}{}", self.sm.0, self.sm.1)
    }
}

/// What a model needs from the GPU (see the module table).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirements {
    /// FP8 tensor cores (sm89+).
    pub fp8: bool,
    /// NVFP4 tensor cores (sm100+).
    pub nvfp4: bool,
    /// The resident DiT's weight bytes (must fit the device's total memory).
    pub min_total_bytes: Option<u64>,
}

impl Requirements {
    /// The lowest compute capability that runs the model, and why.
    pub fn min_sm(&self) -> ((u32, u32), &'static str) {
        if self.nvfp4 {
            ((10, 0), "NVFP4 tensor cores (Blackwell, sm100 or newer)")
        } else if self.fp8 {
            ((8, 9), "FP8 tensor cores (Ada, Hopper or Blackwell: sm89 or newer)")
        } else {
            ((8, 0), "bf16 tensor cores and the attention kernels (sm80 or newer)")
        }
    }

    /// `Err(reason)` when `dev` cannot run `model` with these requirements.
    pub fn check(&self, model: &str, dev: &DeviceProfile) -> Result<(), String> {
        let (min, why) = self.min_sm();
        if dev.sm < min {
            return Err(format!(
                "model `{model}` cannot run on this GPU ({}, {}): it needs {why}; not loaded, this worker does not report ready",
                dev.name,
                dev.sm_label()
            ));
        }
        if let (Some(need), Some(have)) = (self.min_total_bytes, dev.total_bytes) {
            if need > have {
                let gib = |b: u64| b as f64 / GIB as f64;
                return Err(format!(
                    "model `{model}` cannot run on this GPU ({}, {:.1} GiB): its DiT alone needs {:.1} GiB; not loaded, this worker does not report ready",
                    dev.name,
                    gib(have),
                    gib(need)
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_models_refuse_an_a100() {
        let fp8 = Requirements { fp8: true, ..Requirements::default() };
        let e = fp8.check("h3-turbo", &DeviceProfile::a100_80gb()).unwrap_err();
        assert!(e.contains("A100") && e.contains("sm80") && e.contains("FP8") && e.contains("sm89"), "{e}");
        for d in [DeviceProfile::l40s(), DeviceProfile::h100_80gb(), DeviceProfile::rtx_pro_6000()] {
            assert!(fp8.check("h3-turbo", &d).is_ok(), "{}", d.name);
        }
        assert!(Requirements::default().check("wan", &DeviceProfile::a100_80gb()).is_ok());
        assert!(Requirements::default().check("wan", &DeviceProfile::parse("sm75").unwrap()).is_err());
        let fp4 = Requirements { nvfp4: true, ..Requirements::default() };
        assert!(fp4.check("ltx-draft", &DeviceProfile::h100_80gb()).is_err());
        assert!(fp4.check("ltx-draft", &DeviceProfile::rtx_pro_6000()).is_ok());
    }

    #[test]
    fn memory_and_profile_parsing() {
        let big = Requirements { min_total_bytes: Some(60 * GIB), ..Requirements::default() };
        let e = big.check("m", &DeviceProfile::l40s()).unwrap_err();
        assert!(e.contains("48.0 GiB") && e.contains("60.0 GiB"), "{e}");
        assert!(big.check("m", &DeviceProfile::h100_80gb()).is_ok());
        assert_eq!(DeviceProfile::parse("sm90:80").unwrap(), DeviceProfile::new("sm90 (simulated)", (9, 0), Some(80 * GIB)));
        assert_eq!(DeviceProfile::parse("A100").unwrap(), DeviceProfile::a100_80gb());
        assert!(DeviceProfile::parse("tpu").is_err());
        assert!(DeviceProfile::parse("sm8").is_err());
    }
}
