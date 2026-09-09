//! Sampling parameters and Wan inference presets.
//!
//! Ported from FastVideo `fastvideo/api/sampling_param.py` and
//! `fastvideo/pipelines/basic/wan/presets.py`.

use crate::registry::{SamplingAlgorithm, WanModelDefinition};

pub const NEGATIVE_PROMPT_EN: &str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards";

pub const NEGATIVE_PROMPT_CN: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    T2V,
    I2V,
    Other,
}

#[derive(Debug, Clone)]
pub struct SamplingParam {
    pub prompt: Option<String>,
    pub negative_prompt: String,
    pub output_path: String,
    pub seed: u64,
    pub num_frames: u32,
    pub height: u32,
    pub width: u32,
    pub fps: u32,
    pub num_inference_steps: u32,
    pub guidance_scale: f32,
    pub guidance_scale_2: Option<f32>,
    pub boundary_ratio: Option<f32>,
    pub save_video: bool,
}

impl Default for SamplingParam {
    fn default() -> Self {
        Self {
            prompt: None,
            negative_prompt: NEGATIVE_PROMPT_EN.to_string(),
            output_path: "outputs/".into(),
            seed: 1024,
            num_frames: 81,
            height: 480,
            width: 832,
            fps: 16,
            num_inference_steps: 50,
            guidance_scale: 3.0,
            guidance_scale_2: None,
            boundary_ratio: None,
            save_video: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferencePreset {
    pub name: &'static str,
    pub description: &'static str,
    pub workload_type: WorkloadType,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub num_frames: Option<u32>,
    pub fps: u32,
    pub guidance_scale: f32,
    pub guidance_scale_2: Option<f32>,
    pub num_inference_steps: u32,
    pub negative_prompt: &'static str,
}

impl InferencePreset {
    pub fn apply(&self) -> SamplingParam {
        let mut p = SamplingParam::default();
        if let Some(h) = self.height {
            p.height = h;
        }
        if let Some(w) = self.width {
            p.width = w;
        }
        if let Some(f) = self.num_frames {
            p.num_frames = f;
        }
        p.fps = self.fps;
        p.guidance_scale = self.guidance_scale;
        p.guidance_scale_2 = self.guidance_scale_2;
        p.num_inference_steps = self.num_inference_steps;
        p.negative_prompt = self.negative_prompt.to_string();
        p
    }
}

pub fn preset_by_name(name: &str) -> Option<&'static InferencePreset> {
    ALL_PRESETS.iter().copied().find(|p| p.name == name)
}

pub fn sampling_from_definition(def: &WanModelDefinition) -> SamplingParam {
    preset_by_name(def.preset)
        .map(InferencePreset::apply)
        .unwrap_or_default()
}

pub static WAN_T2V_1_3B: InferencePreset = InferencePreset {
    name: "wan_t2v_1_3b",
    description: "Wan 2.1 T2V 1.3B at 480p",
    workload_type: WorkloadType::T2V,
    height: Some(480),
    width: Some(832),
    num_frames: Some(81),
    fps: 16,
    guidance_scale: 3.0,
    guidance_scale_2: None,
    num_inference_steps: 50,
    negative_prompt: NEGATIVE_PROMPT_EN,
};

pub static WAN_T2V_14B: InferencePreset = InferencePreset {
    name: "wan_t2v_14b",
    description: "Wan 2.1 T2V 14B at 720p",
    workload_type: WorkloadType::T2V,
    height: Some(720),
    width: Some(1280),
    num_frames: Some(81),
    fps: 16,
    guidance_scale: 5.0,
    guidance_scale_2: None,
    num_inference_steps: 50,
    negative_prompt: NEGATIVE_PROMPT_EN,
};

pub static WAN_I2V_14B_480P: InferencePreset = InferencePreset {
    name: "wan_i2v_14b_480p",
    description: "Wan 2.1 I2V 14B at 480p",
    workload_type: WorkloadType::I2V,
    height: Some(480),
    width: Some(832),
    num_frames: Some(81),
    fps: 16,
    guidance_scale: 5.0,
    guidance_scale_2: None,
    num_inference_steps: 40,
    negative_prompt: NEGATIVE_PROMPT_EN,
};

pub static WAN_I2V_14B_720P: InferencePreset = InferencePreset {
    name: "wan_i2v_14b_720p",
    description: "Wan 2.1 I2V 14B at 720p",
    workload_type: WorkloadType::I2V,
    height: Some(720),
    width: Some(1280),
    num_frames: Some(81),
    fps: 16,
    guidance_scale: 5.0,
    guidance_scale_2: None,
    num_inference_steps: 40,
    negative_prompt: NEGATIVE_PROMPT_EN,
};

pub static WAN_2_2_T2V_A14B: InferencePreset = InferencePreset {
    name: "wan_2_2_t2v_a14b",
    description: "Wan 2.2 T2V A14B with dual guidance scales",
    workload_type: WorkloadType::T2V,
    height: None,
    width: None,
    num_frames: None,
    fps: 16,
    guidance_scale: 4.0,
    guidance_scale_2: Some(3.0),
    num_inference_steps: 40,
    negative_prompt: NEGATIVE_PROMPT_CN,
};

pub static WAN_2_2_I2V_A14B: InferencePreset = InferencePreset {
    name: "wan_2_2_i2v_a14b",
    description: "Wan 2.2 I2V A14B with dual guidance scales",
    workload_type: WorkloadType::I2V,
    height: None,
    width: None,
    num_frames: None,
    fps: 16,
    guidance_scale: 3.5,
    guidance_scale_2: Some(3.5),
    num_inference_steps: 40,
    negative_prompt: NEGATIVE_PROMPT_CN,
};

pub static WAN_FUN_1_3B_INP: InferencePreset = InferencePreset {
    name: "wan_fun_1_3b_inp",
    description: "Wan 2.1 Fun 1.3B InP (image-to-video inpainting)",
    workload_type: WorkloadType::I2V,
    height: Some(480),
    width: Some(832),
    num_frames: Some(81),
    fps: 16,
    guidance_scale: 6.0,
    guidance_scale_2: None,
    num_inference_steps: 50,
    negative_prompt: NEGATIVE_PROMPT_CN,
};

pub static WAN_FUN_1_3B_CONTROL: InferencePreset = InferencePreset {
    name: "wan_fun_1_3b_control",
    description: "Wan 2.1 Fun 1.3B Control (V2V)",
    workload_type: WorkloadType::Other,
    height: Some(832),
    width: Some(480),
    num_frames: Some(49),
    fps: 16,
    guidance_scale: 6.0,
    guidance_scale_2: None,
    num_inference_steps: 50,
    negative_prompt: NEGATIVE_PROMPT_EN,
};

pub static FAST_WAN_T2V_480P: InferencePreset = InferencePreset {
    name: "fast_wan_t2v_480p",
    description: "FastWan 2.1 T2V DMD at 480p (3-step)",
    workload_type: WorkloadType::T2V,
    height: Some(448),
    width: Some(832),
    num_frames: Some(61),
    fps: 16,
    guidance_scale: 3.0,
    guidance_scale_2: None,
    num_inference_steps: 3,
    negative_prompt: NEGATIVE_PROMPT_EN,
};

pub static WAN_2_2_TI2V_5B: InferencePreset = InferencePreset {
    name: "wan_2_2_ti2v_5b",
    description: "Wan 2.2 TI2V 5B",
    workload_type: WorkloadType::T2V,
    height: Some(704),
    width: Some(1280),
    num_frames: Some(121),
    fps: 24,
    guidance_scale: 5.0,
    guidance_scale_2: None,
    num_inference_steps: 50,
    negative_prompt: NEGATIVE_PROMPT_CN,
};

pub static FAST_WAN_2_2_TI2V_5B: InferencePreset = InferencePreset {
    name: "fast_wan_2_2_ti2v_5b",
    description: "FastWan 2.2 TI2V 5B DMD",
    workload_type: WorkloadType::T2V,
    height: Some(704),
    width: Some(1280),
    num_frames: Some(121),
    fps: 24,
    guidance_scale: 5.0,
    guidance_scale_2: None,
    num_inference_steps: 50,
    negative_prompt: NEGATIVE_PROMPT_CN,
};

pub static LUCY_EDIT_DEV: InferencePreset = InferencePreset {
    name: "lucy_edit_dev",
    description: "Lucy Edit Dev 5B video editing",
    workload_type: WorkloadType::T2V,
    height: Some(480),
    width: Some(832),
    num_frames: Some(81),
    fps: 24,
    guidance_scale: 5.0,
    guidance_scale_2: None,
    num_inference_steps: 50,
    negative_prompt: "",
};

pub static SF_WAN_T2V_1_3B: InferencePreset = InferencePreset {
    name: "sf_wan_t2v_1_3b",
    description: "Self-Forcing Wan 2.1 T2V 1.3B (causal)",
    workload_type: WorkloadType::T2V,
    height: Some(480),
    width: Some(832),
    num_frames: Some(81),
    fps: 16,
    guidance_scale: 6.0,
    guidance_scale_2: None,
    num_inference_steps: 50,
    negative_prompt: NEGATIVE_PROMPT_CN,
};

pub static SF_WAN_2_2_T2V_A14B: InferencePreset = InferencePreset {
    name: "sf_wan_2_2_t2v_a14b",
    description: "Self-Forcing Wan 2.2 T2V A14B (causal)",
    workload_type: WorkloadType::T2V,
    height: Some(448),
    width: Some(832),
    num_frames: Some(81),
    fps: 16,
    guidance_scale: 4.0,
    guidance_scale_2: Some(3.0),
    num_inference_steps: 8,
    negative_prompt: NEGATIVE_PROMPT_CN,
};

pub static SF_WAN_2_2_I2V_A14B: InferencePreset = InferencePreset {
    name: "sf_wan_2_2_i2v_a14b",
    description: "Self-Forcing Wan 2.2 I2V A14B (causal)",
    workload_type: WorkloadType::I2V,
    height: Some(448),
    width: Some(832),
    num_frames: Some(81),
    fps: 16,
    guidance_scale: 4.0,
    guidance_scale_2: Some(3.0),
    num_inference_steps: 8,
    negative_prompt: NEGATIVE_PROMPT_CN,
};

pub static ALL_PRESETS: &[&InferencePreset] = &[
    &WAN_T2V_1_3B,
    &WAN_T2V_14B,
    &WAN_I2V_14B_480P,
    &WAN_I2V_14B_720P,
    &WAN_2_2_T2V_A14B,
    &WAN_2_2_I2V_A14B,
    &WAN_FUN_1_3B_INP,
    &WAN_FUN_1_3B_CONTROL,
    &FAST_WAN_T2V_480P,
    &WAN_2_2_TI2V_5B,
    &FAST_WAN_2_2_TI2V_5B,
    &LUCY_EDIT_DEV,
    &SF_WAN_T2V_1_3B,
    &SF_WAN_2_2_T2V_A14B,
    &SF_WAN_2_2_I2V_A14B,
];

/// Default flow-shift / DMD schedule from FastVideo pipeline configs.
pub fn pipeline_defaults(def: &WanModelDefinition) -> PipelineDefaults {
    match def.pipeline_config {
        "WanT2V480PConfig" => PipelineDefaults {
            flow_shift: 3.0,
            dmd_steps: None,
        },
        "WanT2V720PConfig" | "WanI2V720PConfig" | "Wan2_2_TI2V_5B_Config"
        | "SelfForcingWanT2V480PConfig" => PipelineDefaults {
            flow_shift: 5.0,
            dmd_steps: None,
        },
        "FastWan2_1_T2V_480P_Config" => PipelineDefaults {
            flow_shift: 8.0,
            dmd_steps: Some(&[1000, 757, 522]),
        },
        "FastWan2_2_TI2V_5B_Config" => PipelineDefaults {
            flow_shift: 5.0,
            dmd_steps: Some(&[1000, 757, 522]),
        },
        "Wan2_2_T2V_A14B_Config" => PipelineDefaults {
            flow_shift: 12.0,
            dmd_steps: Some(&[1000, 750, 500, 250]),
        },
        "Wan2_2_I2V_A14B_Config" => PipelineDefaults {
            flow_shift: 5.0,
            dmd_steps: None,
        },
        "SelfForcingWan2_2_T2V480PConfig" => PipelineDefaults {
            flow_shift: 12.0,
            dmd_steps: Some(&[1000, 850, 700, 550, 350, 275, 200, 125]),
        },
        _ => match def.sampling {
            SamplingAlgorithm::Dmd => PipelineDefaults {
                flow_shift: 8.0,
                dmd_steps: Some(&[1000, 757, 522]),
            },
            _ => PipelineDefaults {
                flow_shift: 3.0,
                dmd_steps: None,
            },
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineDefaults {
    pub flow_shift: f32,
    pub dmd_steps: Option<&'static [i32]>,
}
