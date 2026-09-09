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
                tiny: false,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(gen.sampling.num_inference_steps, 3);
        assert_eq!(gen.sampling.height, 448);
        assert_eq!(gen.pipeline.dmd_steps, Some(&[1000, 757, 522][..]));
        assert!(gen.generate_video("a raccoon").is_err());
    }

    #[test]
    fn host_unipc_generate_tokenizes_cached_umt5() {
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Host,
                output_path: Some(
                    std::env::temp_dir()
                        .join("fastvideo-host-unipc")
                        .to_string_lossy()
                        .into(),
                ),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(gen.definition.sampling, SamplingAlgorithm::UniPc);
        let out = match gen.generate_video("A curious raccoon in a field of sunflowers.") {
            Ok(out) => out,
            Err(err) if err.to_string().contains("tokenizer.json") => {
                eprintln!("skip: {err}");
                return;
            }
            Err(err) => panic!("{err}"),
        };
        let path = out.frame_paths.first().expect("latents json");
        let body = std::fs::read_to_string(path).unwrap();
        assert!(body.contains("token_ids"));
        assert!(!body.contains("\"token_ids\":[0,1,2,3]"));
        assert!(body.contains("\"backend\":\"host\""));
    }

    #[test]
    fn burn_and_luminal_generate_write_latents() {
        for backend in [BackendKind::Burn, BackendKind::Luminal] {
            let gen = VideoGenerator::from_pretrained(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                LoadOptions {
                    backend,
                    output_path: Some(
                        std::env::temp_dir()
                            .join(format!("fastvideo-{backend}-unipc"))
                            .to_string_lossy()
                            .into(),
                    ),
                    ..LoadOptions::default()
                },
            )
            .unwrap();
            match gen.generate_video("A curious raccoon in a field of sunflowers.") {
                Ok(out) => {
                    assert!(std::path::Path::new(&out.frame_paths[0]).exists());
                }
                Err(err) if err.to_string().contains("tokenizer.json") => {}
                Err(err) => panic!("{err}"),
            }
        }
    }

    #[test]
    fn i2v_generate_is_explicit() {
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            LoadOptions {
                backend: BackendKind::Candle,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let err = gen.generate_video("a cat").unwrap_err();
        assert!(err.to_string().contains("I2V"));
    }

    #[test]
    fn sampling_overrides_apply() {
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                height: Some(256),
                width: Some(256),
                num_frames: Some(9),
                num_inference_steps: Some(2),
                guidance_scale: Some(1.0),
                seed: Some(7),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(gen.sampling.height, 256);
        assert_eq!(gen.sampling.width, 256);
        assert_eq!(gen.sampling.num_frames, 9);
        assert_eq!(gen.sampling.num_inference_steps, 2);
        assert_eq!(gen.sampling.guidance_scale, 1.0);
        assert_eq!(gen.sampling.seed, 7);
    }

    #[test]
    fn candle_resolves_cached_1_3b_snapshot() {
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Candle,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let Some(root) = gen.resolved_weights_dir() else {
            eprintln!("skip: Wan-AI/Wan2.1-T2V-1.3B-Diffusers not in HF cache");
            return;
        };
        assert!(root.join("transformer").is_dir());
        assert!(root.join("vae").is_dir());
        assert!(root.join("tokenizer/tokenizer.json").is_file());
        assert!(root.join("text_encoder").is_dir() || root.join("text_encoder_2").is_dir());
    }

    #[test]
    fn candle_real_1_3b_generate_is_gated() {
        if std::env::var("FASTVIDEO_REAL_GENERATE").ok().as_deref() != Some("1") {
            return;
        }
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Candle,
                height: Some(256),
                width: Some(256),
                num_frames: Some(9),
                num_inference_steps: Some(1),
                guidance_scale: Some(1.0),
                dtype: Some("f32".into()),
                device: "cpu".into(),
                output_path: Some(
                    std::env::temp_dir()
                        .join("fastvideo-real-1-3b")
                        .to_string_lossy()
                        .into(),
                ),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let out = gen
            .generate_video("A curious raccoon in a field of sunflowers.")
            .unwrap();
        assert!(!out.frame_paths.is_empty());
        assert!(std::path::Path::new(&out.frame_paths[0]).exists());
    }

    #[test]
    fn moe_and_causal_presets_resolve() {
        let moe = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            LoadOptions::default(),
        )
        .unwrap();
        assert_eq!(moe.pipeline.boundary_ratio, Some(0.875));
        assert_eq!(moe.pipeline.flow_shift, 12.0);
        let causal = VideoGenerator::from_pretrained(
            "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
            LoadOptions::default(),
        )
        .unwrap();
        assert_eq!(causal.definition.sampling, SamplingAlgorithm::CausalDmd);
        let i2v = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            LoadOptions::default(),
        )
        .unwrap();
        assert_eq!(i2v.sampling.height, 720);
        let t14 = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            LoadOptions::default(),
        )
        .unwrap();
        assert_eq!(t14.definition.preset, "wan_t2v_14b");
    }

    #[test]
    fn candle_tiny_generate_writes_png() {
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Candle,
                tiny: true,
                output_path: Some(
                    std::env::temp_dir()
                        .join("fastvideo-core-tiny")
                        .to_string_lossy()
                        .into(),
                ),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let out = gen.generate_video("a raccoon").unwrap();
        assert!(!out.frame_paths.is_empty());
        assert!(std::path::Path::new(&out.frame_paths[0]).exists());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_device_requires_feature() {
        let err = crate::generator::resolve_candle_device("cuda").unwrap_err();
        assert!(err.to_string().contains("features cuda"));
    }

    #[test]
    fn cuda_defaults_to_bf16() {
        assert_eq!(
            crate::generator::resolve_dtype(None, "cuda").unwrap(),
            candle_core::DType::BF16
        );
        assert_eq!(
            crate::generator::resolve_dtype(None, "cpu").unwrap(),
            candle_core::DType::F32
        );
    }
}
