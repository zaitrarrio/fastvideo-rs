//! DreamX-World host configs (Wan TI2V-5B + cam). Spec: docs/ports/dreamx.md.

use crate::wan::WanVideoArchConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamXPreset {
    Cam5b,
    Ar5b,
}

impl DreamXPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cam5b => "dreamx_5b_cam",
            Self::Ar5b => "dreamx_5b_ar",
        }
    }

    pub fn is_ar(self) -> bool {
        matches!(self, Self::Ar5b)
    }

    pub fn default_height(self) -> usize {
        if self.is_ar() {
            704
        } else {
            480
        }
    }

    pub fn default_width(self) -> usize {
        if self.is_ar() {
            1280
        } else {
            832
        }
    }

    pub fn default_num_frames(self) -> usize {
        161
    }

    pub fn default_steps(self) -> usize {
        if self.is_ar() {
            4
        } else {
            30
        }
    }

    pub fn flow_shift(self) -> f64 {
        if self.is_ar() {
            5.0
        } else {
            3.0
        }
    }

    pub fn guidance_scale(self) -> f32 {
        if self.is_ar() {
            1.0
        } else {
            5.0
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DreamXConfig {
    pub wan: WanVideoArchConfig,
    pub cam_method: &'static str,
    pub add_control_adapter: bool,
    pub attn_compress: usize,
    pub local_attn_size: i32,
    pub sink_size: usize,
}

impl DreamXConfig {
    pub fn for_preset(preset: DreamXPreset) -> Self {
        let mut wan = WanVideoArchConfig::wan_2_2_ti2v_5b();
        if preset.is_ar() {
            wan.causal = true;
            wan.local_attn_size = 12;
            wan.sink_size = 3;
        }
        Self {
            wan,
            cam_method: "prope",
            add_control_adapter: true,
            attn_compress: if preset.is_ar() { 4 } else { 1 },
            local_attn_size: if preset.is_ar() { 12 } else { -1 },
            sink_size: if preset.is_ar() { 3 } else { 0 },
        }
    }

    pub fn tiny() -> Self {
        Self {
            wan: WanVideoArchConfig::tiny(),
            cam_method: "prope",
            add_control_adapter: true,
            attn_compress: 1,
            local_attn_size: -1,
            sink_size: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cam_is_ti2v_5b() {
        let c = DreamXConfig::for_preset(DreamXPreset::Cam5b);
        assert_eq!(c.wan.in_channels, 48);
        assert_eq!(c.cam_method, "prope");
        assert!(c.add_control_adapter);
    }

    #[test]
    fn ar_has_causal_window() {
        let c = DreamXConfig::for_preset(DreamXPreset::Ar5b);
        assert!(c.wan.causal);
        assert_eq!(c.local_attn_size, 12);
        assert_eq!(c.sink_size, 3);
    }
}
