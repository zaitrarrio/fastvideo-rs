//! Native Wan tiny graph for Luminal bring-up.
//!
//! Eager `NdTensor` forwards match Candle tiny shapes for full Diffusers loads.
//! Tiny bring-up uses luminal 0.2 `Graph` compile (`GenericCompiler` +
//! `CPUCompiler`) for a fixed-shape DiT step and VAE decode (ADR-0001).

pub mod compiled;
pub mod nn;
pub mod pipeline;
pub mod tensor;
pub mod transformer;
pub mod umt5;
pub mod vae;
pub mod weights;

pub use compiled::{CompiledDitStep, CompiledVaeDecode, TINY_LATENT_SHAPE, TINY_VIDEO_SHAPE};
pub use pipeline::{GenerateConfig, WanPipeline};
pub use tensor::NdTensor;
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
            .join("fastvideo-luminal-tiny")
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
        let z = NdTensor::zeros(&[1, 4, 3, 2, 2]);
        let out = vae.decode(&z).unwrap();
        assert_eq!(
            out.shape,
            vec![1, 3, 9, 8, 8],
            "two upsample3d stages: 3 latents → 9 RGB"
        );
    }

    #[test]
    fn compiled_dit_and_vae_run() {
        let mut dit = CompiledDitStep::tiny();
        let latents = NdTensor::zeros(&TINY_LATENT_SHAPE);
        let t = NdTensor::from_vec(vec![500f32], vec![1]).unwrap();
        let enc = NdTensor::zeros(&[1, 8, 16]);
        let out = dit.run(&latents, &t, &enc).unwrap();
        assert_eq!(out.shape, TINY_LATENT_SHAPE.to_vec());

        let mut vae = CompiledVaeDecode::tiny();
        let video = vae.run(&latents).unwrap();
        assert_eq!(video.shape, TINY_VIDEO_SHAPE.to_vec());
    }
}
