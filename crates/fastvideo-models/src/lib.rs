//! Host-side model math and configs consumed by `fastvideo-cudarc`.

// Host reference code indexes tensors the way the Python it is checked against
// does; the iterator forms clippy suggests (`.iter().enumerate().take(n)`) hide
// the shape check an out-of-range index gives.
#![allow(clippy::needless_range_loop)]

pub mod cosmos;
pub mod cosmos3;
pub mod dreamx;
pub mod flux;
pub mod flux2;
pub mod gamecraft;
pub mod gen3c;
pub mod glm_image;
pub mod h3;
pub mod hunyuan15;
pub mod hyworld;
pub mod kandinsky5;
pub mod key_fixture;
pub mod lingbot;
pub mod lingbotworld;
pub mod longcat;
pub mod ltx2;
pub mod matrixgame;
pub mod mmaudio;
pub mod ncii;
pub mod nvfp4;
pub mod pisa_attn;
pub mod plug;
pub mod sana_video;
pub mod schedulers;
pub mod sd35;
pub mod sol_attn;
pub mod stable_audio;
pub mod techniques;
pub mod vae;
pub mod wan;
pub mod zimage;

pub use schedulers::{
    DmdSchedule, FlowMatchEulerDiscreteScheduler, FlowUniPCMultistepScheduler, RcmSchedule,
};
/// The `FASTVIDEO_*` settings layer (`techniques::settings`): every read of a
/// `FASTVIDEO_*` name in the workspace goes through it.
pub use techniques::settings;
pub use wan::{
    causal_temporal_mask, i2v_first_frame_mask, moe_expert, tokenize_prompt, MoeExpert, Umt5Config,
    WanVaeConfig, WanVideoArchConfig,
};
