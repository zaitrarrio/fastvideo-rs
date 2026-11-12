//! The NVRTC option strings of the LingBot MoE module, shared by `build.rs`
//! (which compiles `AOT_MOE` with libnvrtc at build time) and the runtime
//! NVRTC fallback (`rt::ops::moe_compile_options`). These are what cudarc
//! 0.17's `CompileOptions::build` emits for `use_fast_math: Some(true)`,
//! `ftz: Some(true)`: `--ftz=true` and `--fmad=true`, i.e. FTZ and fused
//! multiply-add, with IEEE division, square root and `expf` (no
//! `--use_fast_math`). `--gpu-architecture=<arch>` is appended by each caller.
//! The `moe_nvrtc_options_match_runtime` test keeps the two in step.

/// Option strings, before `--gpu-architecture`.
pub const MOE_NVRTC_OPTIONS: &[&str] = &["--ftz=true", "--fmad=true"];
