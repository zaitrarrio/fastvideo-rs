//! HunyuanVideo 1.5 host configs and math for `fastvideo-cudarc::hunyuan15`.
//! See docs/ports/hunyuan15.md.

pub mod config;
pub mod rope;
pub mod schedule;
pub mod text;

pub use config::{
    Hunyuan15PipelineDefaults, Hunyuan15Preset, Hunyuan15TransformerConfig, Hunyuan15VaeConfig,
};
pub use rope::Hunyuan15RopeTables;
pub use schedule::Hunyuan15Schedule;
pub use text::{
    extract_glyph_texts, format_user_prompt, qwen_hidden_tap, tokenize_byt5, tokenize_qwen,
    QWEN_CROP_START, QWEN_LAYERS_TO_SKIP, QWEN_SYSTEM_MESSAGE,
};
