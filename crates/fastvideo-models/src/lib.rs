//! Wan/FastWan model components. Forwards currently target Candle.

pub mod nn;
pub mod schedulers;
pub mod wan;

pub use schedulers::{DmdSchedule, FlowMatchEulerDiscreteScheduler, FlowUniPCMultistepScheduler};
pub use wan::{
    tokenize_prompt, AutoencoderKlWan, ClipVision, ClipVisionConfig, GenerateConfig, Umt5Config,
    Umt5Encoder, WanPipeline, WanTransformer3D, WanVaeConfig, WanVideoArchConfig,
};
