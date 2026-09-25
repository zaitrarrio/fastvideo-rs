//! MiniMax-H3 / FastH3 model family: configuration structs and the host-side
//! math (packed-sequence layout, position grids, schedules) the device graph in
//! `fastvideo-cudarc::h3` consumes. Reference: diffusers `transformer_minimax_h3.py`
//! and FastVideo `pipelines/basic/minimax_h3`. See docs/ports/h3.md.

pub mod config;
pub mod lora;
pub mod memory;
pub mod mrope;
pub mod packing;
pub mod presentation;
pub mod reference;
pub mod schedule;
pub mod sol;
pub mod spark;
pub mod tokenizer;
pub mod vision_preprocess;
