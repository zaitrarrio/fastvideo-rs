//! Native Wan graph on cudarc: UMT5 → DiT → feat-cache VAE, device-resident
//! on CUDA (see [`tensor`]) with plain-Rust reference ops on CPU runs.

pub mod affine;
pub mod ar_cache;
pub mod attn;
pub mod bf16_gemm;
pub mod clip;
#[cfg(feature = "cuda")]
pub mod conv;
pub mod device;
pub mod envflag;
#[cfg(feature = "cuda")]
pub mod fp8;
pub mod fused;
pub mod hopper;
#[cfg(feature = "cuda")]
pub mod kernels;
pub mod log;
pub mod nn;
pub mod nvfp4;
pub mod ops;
pub mod pipeline;
pub mod resident;
pub mod sla;
pub mod sol_cache;
pub mod sp;
pub mod stats;
pub mod taehv;
pub mod tensor;
pub mod transformer;
pub mod umt5;
pub mod vae;
pub mod vsa;
pub mod weights;

pub use clip::{ClipVision, ClipVisionConfig};
pub use pipeline::{DenoiseStep, GenerateConfig, LoadParts, StepObserver, WanPipeline};
pub use tensor::CudaTensor;
pub use transformer::WanTransformer3D;
pub use vae::AutoencoderKlWan;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn tiny_generate_writes_png() {
        let mut pipe = WanPipeline::tiny();
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.is_dmd = true;
        cfg.flow_shift = 8.0;
        cfg.guidance_scale = 1.0;
        cfg.output_dir = std::env::temp_dir()
            .join("fastvideo-cudarc-tiny")
            .to_string_lossy()
            .into();
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
        let mut cfg = GenerateConfig::default();
        cfg.tiny = true;
        cfg.is_dmd = true;
        cfg.flow_shift = 8.0;
        cfg.guidance_scale = 2.5;
        cfg.output_dir = std::env::temp_dir()
            .join("fastvideo-cudarc-tiny-cfg")
            .to_string_lossy()
            .into();
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
