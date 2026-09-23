//! MMAudio host configs. Spec: docs/ports/mmaudio.md.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmAudioPreset {
    Large44kV2,
}

impl MmAudioPreset {
    pub fn as_str(self) -> &'static str {
        "mmaudio_large_44k_v2"
    }
    pub fn sample_rate(self) -> u32 {
        44_100
    }
    pub fn audio_channels(self) -> usize {
        2
    }
    pub fn duration_s(self) -> f32 {
        10.0
    }
    pub fn default_steps(self) -> usize {
        25
    }
    pub fn hop_length(self) -> usize {
        512
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MmAudioDiTConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub text_dim: usize,
    pub visual_dim: usize,
    pub sample_size: usize,
}

impl MmAudioDiTConfig {
    pub fn large_44k_v2() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            num_layers: 28,
            num_attention_heads: 16,
            attention_head_dim: 64,
            text_dim: 1024,
            visual_dim: 1024,
            sample_size: 860,
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 8,
            out_channels: 8,
            num_layers: 2,
            num_attention_heads: 4,
            attention_head_dim: 8,
            text_dim: 32,
            visual_dim: 32,
            sample_size: 32,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MmAudioConfig {
    pub dit: MmAudioDiTConfig,
    pub sample_rate: u32,
    pub audio_channels: usize,
}

impl MmAudioConfig {
    pub fn for_preset(preset: MmAudioPreset) -> Self {
        Self {
            dit: MmAudioDiTConfig::large_44k_v2(),
            sample_rate: preset.sample_rate(),
            audio_channels: preset.audio_channels(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            dit: MmAudioDiTConfig::tiny(),
            sample_rate: 16_000,
            audio_channels: 1,
        }
    }

    pub fn schedule(&self, steps: usize) -> FlowMatchEulerDiscreteScheduler {
        let mut s = FlowMatchEulerDiscreteScheduler::new(1000, 1.0);
        s.set_timesteps(steps);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_dims() {
        let c = MmAudioDiTConfig::large_44k_v2();
        assert_eq!(c.inner_dim(), 1024);
        assert_eq!(c.num_layers, 28);
    }
}
