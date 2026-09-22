//! Host-side model math and configs consumed by `fastvideo-cudarc`.

pub mod cosmos;
pub mod dreamx;
pub mod gamecraft;
pub mod gen3c;
pub mod h3;
pub mod hunyuan15;
pub mod hyworld;
pub mod kandinsky5;
pub mod lingbot;
pub mod lingbotworld;
pub mod longcat;
pub mod ltx2;
pub mod matrixgame;
pub mod schedulers;
pub mod wan;

pub use schedulers::{
    DmdSchedule, FlowMatchEulerDiscreteScheduler, FlowUniPCMultistepScheduler, RcmSchedule,
};
pub use wan::{
    causal_temporal_mask, i2v_first_frame_mask, moe_expert, tokenize_prompt, MoeExpert, Umt5Config,
    WanVaeConfig, WanVideoArchConfig,
};
