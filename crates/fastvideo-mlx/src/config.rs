//! FastMetal-QAD / FastH3 MLX model specs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastMetalPreset {
    Qad1_3b,
    Qad5b,
    Qad14b,
}

impl FastMetalPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qad1_3b => "fastmetal_qad_1_3b",
            Self::Qad5b => "fastmetal_qad_5b",
            Self::Qad14b => "fastmetal_qad_14b",
        }
    }

    pub fn hub_id(self) -> &'static str {
        match self {
            Self::Qad1_3b => "FastVideo/FastMetal-1.3B-QAD",
            Self::Qad5b => "FastVideo/FastMetal-5B-QAD",
            Self::Qad14b => "FastVideo/FastMetal-14B-QAD",
        }
    }

    pub fn min_unified_memory_gb(self) -> u32 {
        match self {
            Self::Qad1_3b | Self::Qad5b => 16,
            Self::Qad14b => 36,
        }
    }

    pub fn default_height(self) -> u32 {
        480
    }

    pub fn default_width(self) -> u32 {
        832
    }

    pub fn default_frames(self) -> u32 {
        81
    }

    pub fn default_steps(self) -> u32 {
        3
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastH3MlxPreset {
    Preview,
}

impl FastH3MlxPreset {
    pub fn as_str(self) -> &'static str {
        "fasth3_mlx_preview"
    }

    pub fn hub_id(self) -> &'static str {
        "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"
    }

    pub fn min_unified_memory_gb(self) -> u32 {
        36
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxModelSpec {
    FastMetal(FastMetalPreset),
    FastH3(FastH3MlxPreset),
}

impl MlxModelSpec {
    pub fn hub_id(self) -> &'static str {
        match self {
            Self::FastMetal(p) => p.hub_id(),
            Self::FastH3(p) => p.hub_id(),
        }
    }

    pub fn requires_apple_silicon(self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_ids() {
        assert!(FastMetalPreset::Qad1_3b.hub_id().contains("1.3B"));
        assert!(FastMetalPreset::Qad14b.min_unified_memory_gb() >= 36);
    }
}
