//! Wan/FastWan model components. Forwards currently target Candle.

pub mod nn;
pub mod schedulers;
pub mod wan;

pub use schedulers::{DmdSchedule, FlowMatchEulerDiscreteScheduler};
pub use wan::{
    AutoencoderKlWan, GenerateConfig, Umt5Config, Umt5Encoder, WanPipeline, WanTransformer3D,
    WanVaeConfig, WanVideoArchConfig,
};
