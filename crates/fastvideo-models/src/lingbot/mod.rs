//! LingBot-Video host math for `fastvideo-cudarc::lingbot`.
//! See docs/ports/lingbot.md.

pub mod config;
pub mod rope;
pub mod schedule;

pub use config::{
    tokenize_lingbot_prompt, LingBotPreset, LingBotTransformerConfig, PROMPT_CROP_START,
    PROMPT_TEMPLATE,
};
pub use rope::{apply_rope_real, rope_freqs};
pub use schedule::LingBotSchedule;