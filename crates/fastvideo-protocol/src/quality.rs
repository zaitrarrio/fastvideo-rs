//! Quality gate definitions: the reference-free video sanity checks.
//!
//! These mirror the real gates in `fastvideo-gpucheck` (`quality.rs`): they
//! catch NaNs, flat/saturated frames, frozen video, incoherent noise, and
//! brightness flashes. The thresholds below match `QualityGates::default()`;
//! keep them in sync. Served at `GET /fv/v1/quality/gates`.

use serde::Serialize;

/// Numeric gate thresholds (see `QualityGate` for what each means).
#[derive(Debug, Clone, Copy)]
pub struct QualityThresholds {
    /// Minimum per-frame pixel std (0..255 units); below = flat frame.
    pub frame_std_min: f64,
    /// Maximum fraction of samples at the clamp rails; above = saturated.
    pub clipped_fraction_max: f64,
    /// Minimum mean absolute frame difference; below = frozen video.
    pub temporal_mad_min: f64,
    /// Maximum mean absolute frame difference; above = incoherent noise.
    pub temporal_mad_max: f64,
    /// Maximum luma jump between consecutive frames; above = flash.
    pub luma_jump_max: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            frame_std_min: 3.0,
            clipped_fraction_max: 0.25,
            temporal_mad_min: 0.2,
            temporal_mad_max: 45.0,
            luma_jump_max: 40.0,
        }
    }
}

/// One named gate for the definitions API.
#[derive(Debug, Clone, Serialize)]
pub struct QualityGate {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    /// Human-readable threshold, e.g. "0 (any NaN fails)".
    pub threshold: &'static str,
    pub unit: &'static str,
    pub failure_mode: &'static str,
}

pub const QUALITY_GATES: &[QualityGate] = &[
    QualityGate {
        id: "non_finite",
        title: "No NaNs / infinities",
        description: "Counts non-finite pixels in the decoded video tensor.",
        threshold: "0 (any non-finite pixel fails)",
        unit: "pixels",
        failure_mode: "dead weights or wrong latent scaling",
    },
    QualityGate {
        id: "frame_std_min",
        title: "Minimum frame detail",
        description: "Mean over frames of per-frame pixel std must stay above the floor.",
        threshold: "3.0",
        unit: "0..255 pixel std",
        failure_mode: "flat frames (dead weights, wrong latent scaling)",
    },
    QualityGate {
        id: "clipped_fraction_max",
        title: "Maximum saturation",
        description: "Fraction of samples at the clamp rails (|x| >= 0.99) must stay below the ceiling.",
        threshold: "0.25",
        unit: "fraction",
        failure_mode: "saturated frames",
    },
    QualityGate {
        id: "temporal_mad",
        title: "Temporal coherence band",
        description: "Mean absolute difference between consecutive frames must stay inside the band.",
        threshold: "0.2 – 45.0",
        unit: "0..255 MAD",
        failure_mode: "frozen video (too low) or incoherent noise (too high)",
    },
    QualityGate {
        id: "luma_jump_max",
        title: "Maximum brightness jump",
        description: "Largest jump in mean brightness between consecutive frames.",
        threshold: "40.0",
        unit: "0..255 luma",
        failure_mode: "brightness flashes (cache / chunk boundary bugs)",
    },
];
