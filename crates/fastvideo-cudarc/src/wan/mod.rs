//! Native Wan graph on cudarc: UMT5 → DiT → feat-cache VAE, device-resident
//! on CUDA (see [`tensor`]) with plain-Rust reference ops on CPU runs.

pub mod ar_cache;
pub mod causal;
pub mod clip;
pub mod fuse;
pub mod longlive;
#[cfg(test)]
mod manifest_tests;
pub mod pipeline;
pub mod plug;
pub mod sol_cache;
pub mod stream;
pub mod transformer;
pub mod umt5;
pub mod vae;
pub mod vae21_encode;
pub mod vae22;

// The runtime (`crate::rt`) under its old `wan::` paths. These re-exports keep
// `crate::wan::<module>` and `fastvideo_cudarc::wan::<module>` compiling while
// callers move to `crate::rt::<module>`; new code imports from `crate::rt`.
// To be removed once nothing uses them (plan B1 follow-up).
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(crate) use crate::rt::act16;
pub use crate::rt::affine;
pub use crate::rt::attn;
#[cfg(feature = "cuda")]
pub use crate::rt::attn_dc;
pub use crate::rt::attn_fp4;
pub use crate::rt::attn_fp8;
pub use crate::rt::attn_sage;
pub use crate::rt::bf16_gemm;
#[cfg(feature = "cuda")]
pub use crate::rt::conv;
#[cfg(feature = "cuda")]
pub use crate::rt::cudnn_sdpa;
pub use crate::rt::device;
pub use crate::rt::dump;
pub use crate::rt::envflag;
pub use crate::rt::evalstats;
#[cfg(feature = "cuda")]
pub use crate::rt::fp8;
pub use crate::rt::fused;
pub use crate::rt::gpu_trace;
pub use crate::rt::graph;
pub use crate::rt::hopper;
pub use crate::rt::hostmem;
pub use crate::rt::inject;
#[cfg(feature = "cuda")]
pub use crate::rt::kernels;
pub use crate::rt::ledger;
pub use crate::rt::log;
pub use crate::rt::nn;
pub use crate::rt::nvfp4;
#[cfg(feature = "nvfp4-gemm")]
pub use crate::rt::nvfp4_gemm;
#[cfg(feature = "cuda")]
pub use crate::rt::nvfp4_linear;
#[cfg(feature = "cuda")]
pub use crate::rt::nvrtc_sass;
pub use crate::rt::offload;
pub use crate::rt::ops;
pub use crate::rt::quant;
pub use crate::rt::resident;
pub use crate::rt::sdpa_rule;
pub use crate::rt::sla;
pub use crate::rt::sol_ops;
pub use crate::rt::sp;
#[cfg(feature = "cuda")]
pub use crate::rt::stage_upload;
pub use crate::rt::stats;
pub use crate::rt::taehv;
pub use crate::rt::taehv_ref;
pub use crate::rt::tensor;
pub use crate::rt::vsa;
pub use crate::rt::weights;
pub use crate::rt::writer;

pub use crate::rt::tensor::CudaTensor;
pub use clip::{ClipVision, ClipVisionConfig};
pub use pipeline::{DenoiseStep, GenerateConfig, LoadParts, StepObserver, WanPipeline};
pub use transformer::WanTransformer3D;
pub use vae::AutoencoderKlWan;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn tiny_generate_writes_png() {
        let mut pipe = WanPipeline::tiny();
        let cfg = GenerateConfig {
            tiny: true,
            is_dmd: true,
            flow_shift: 8.0,
            guidance_scale: 1.0,
            output_dir: std::env::temp_dir()
                .join("fastvideo-cudarc-tiny")
                .to_string_lossy()
                .into(),
            ..Default::default()
        };
        let paths = pipe.generate(&cfg).unwrap();
        assert!(!paths.is_empty());
        assert!(Path::new(&paths[0]).exists());
    }

    /// `guidance_scale != 1.0` exercises the batched cond/uncond CFG forward
    /// pass in `dit_cfg` (see pipeline.rs) instead of the `scale == 1.0`
    /// single-pass shortcut — this is the path most changed by the CFG
    /// batching optimization, so it needs its own coverage rather than
    /// relying on `tiny_generate_writes_png` (which sets `guidance_scale =
    /// 1.0` and never exercises it).
    #[test]
    fn tiny_generate_with_cfg_writes_png() {
        let mut pipe = WanPipeline::tiny();
        let cfg = GenerateConfig {
            tiny: true,
            is_dmd: true,
            flow_shift: 8.0,
            guidance_scale: 2.5,
            output_dir: std::env::temp_dir()
                .join("fastvideo-cudarc-tiny-cfg")
                .to_string_lossy()
                .into(),
            ..Default::default()
        };
        let paths = pipe.generate(&cfg).unwrap();
        assert!(!paths.is_empty());
        assert!(Path::new(&paths[0]).exists());
    }

    #[test]
    fn feat_cache_decode_is_4n_plus_1() {
        let cfg = fastvideo_models::wan::WanVaeConfig {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![true, true],
            load_encoder: false,
            decoder_base_dim: 8,
            is_residual: false,
            patch_size: 1,
            latents_mean: vec![0.0; 4],
            latents_std: vec![1.0; 4],
        };
        let vae = AutoencoderKlWan::zeros(cfg);
        let z = CudaTensor::zeros(&[1, 4, 3, 2, 2]);
        let out = vae.decode(&z).unwrap();
        assert_eq!(
            out.shape,
            vec![1, 3, 9, 8, 8],
            "two upsample3d stages: 3 latents → 9 RGB"
        );
    }
}
