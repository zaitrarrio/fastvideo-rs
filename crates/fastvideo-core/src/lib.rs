//! FastVideo-rs inference core: registry, sampling, and VideoGenerator.

pub mod backend_kind;
pub mod error;
pub mod generator;
pub mod registry;
pub mod sampling;

pub use backend_kind::BackendKind;
pub use error::{FastVideoError, Result};
pub use generator::{GenerateOutput, LoadOptions, VideoGenerator};
pub use registry::{resolve_wan, SamplingAlgorithm, WanModelDefinition, WAN_MODEL_DEFINITIONS};
pub use sampling::{sampling_from_definition, InferencePreset, SamplingParam, ALL_PRESETS};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pretrained_fastwan() {
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Host,
                num_gpus: 1,
                tiny: false,
                weights_path: None,
                output_path: None,
            },
        )
        .unwrap();
        assert_eq!(gen.sampling.num_inference_steps, 3);
        assert_eq!(gen.sampling.height, 448);
        assert_eq!(gen.pipeline.dmd_steps, Some(&[1000, 757, 522][..]));
        assert!(gen.generate_video("a raccoon").is_err());
    }

    #[test]
    fn candle_tiny_generate_writes_png() {
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Candle,
                num_gpus: 1,
                tiny: true,
                weights_path: None,
                output_path: Some(
                    std::env::temp_dir()
                        .join("fastvideo-core-tiny")
                        .to_string_lossy()
                        .into(),
                ),
            },
        )
        .unwrap();
        let out = gen.generate_video("a raccoon").unwrap();
        assert!(!out.frame_paths.is_empty());
        assert!(std::path::Path::new(&out.frame_paths[0]).exists());
    }
}
