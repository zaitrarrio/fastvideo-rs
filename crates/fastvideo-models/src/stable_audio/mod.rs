//! Stable Audio Open host configs. Spec: docs/ports/stable-audio.md.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableAudioPreset {
    Open10,
    OpenSmall,
}

impl StableAudioPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open10 => "stable_audio_open_1_0",
            Self::OpenSmall => "stable_audio_open_small",
        }
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
        100
    }
    pub fn hop_length(self) -> usize {
        2048
    }
    pub fn sample_size_audio(self) -> usize {
        2_097_152
    }
    pub fn latent_length(self) -> usize {
        self.sample_size_audio() / self.hop_length()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StableAudioDiTConfig {
    pub sample_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_attention_heads: usize,
    pub cross_attention_dim: usize,
    pub time_proj_dim: usize,
    pub global_states_input_dim: usize,
    pub cross_attention_input_dim: usize,
}

impl StableAudioDiTConfig {
    pub fn open_1_0() -> Self {
        Self {
            sample_size: 1024,
            in_channels: 64,
            out_channels: 64,
            num_layers: 24,
            attention_head_dim: 64,
            num_attention_heads: 24,
            num_key_value_attention_heads: 12,
            cross_attention_dim: 768,
            time_proj_dim: 256,
            global_states_input_dim: 1536,
            cross_attention_input_dim: 768,
        }
    }

    pub fn open_small() -> Self {
        let mut c = Self::open_1_0();
        c.num_layers = 12;
        c
    }

    pub fn for_preset(preset: StableAudioPreset) -> Self {
        match preset {
            StableAudioPreset::Open10 => Self::open_1_0(),
            StableAudioPreset::OpenSmall => Self::open_small(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            sample_size: 32,
            in_channels: 8,
            out_channels: 8,
            num_layers: 2,
            attention_head_dim: 8,
            num_attention_heads: 4,
            num_key_value_attention_heads: 2,
            cross_attention_dim: 16,
            time_proj_dim: 16,
            global_states_input_dim: 32,
            cross_attention_input_dim: 16,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StableAudioConfig {
    pub dit: StableAudioDiTConfig,
    pub sample_rate: u32,
    pub audio_channels: usize,
}

impl StableAudioConfig {
    pub fn for_preset(preset: StableAudioPreset) -> Self {
        Self {
            dit: StableAudioDiTConfig::for_preset(preset),
            sample_rate: preset.sample_rate(),
            audio_channels: preset.audio_channels(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            dit: StableAudioDiTConfig::tiny(),
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
    fn open_dims() {
        let c = StableAudioDiTConfig::open_1_0();
        assert_eq!(c.inner_dim(), 1536);
        assert_eq!(StableAudioPreset::Open10.latent_length(), 1024);
    }
}
