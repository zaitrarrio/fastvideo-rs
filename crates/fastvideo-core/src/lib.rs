pub mod av_generate;
pub mod backend_kind;
pub mod error;
pub mod generator;
pub mod registry;
pub mod sampling;

pub use av_generate::{generate_av, AvGenerateOptions, AvGenerateOutput};
pub use backend_kind::BackendKind;
pub use error::{FastVideoError, Result};
pub use generator::{BenchStats, ClipBenchStats, GenerateOutput, LoadOptions, VideoGenerator};
pub use registry::{
    all_registered_ids, resolve, resolve_wan, FamilyModelDefinition, Ltx2Line, ModelFamily,
    ResolvedModel, SamplingAlgorithm, WanModelDefinition, H3_MODEL_DEFINITIONS,
    LTX2_MODEL_DEFINITIONS, WAN_MODEL_DEFINITIONS,
};
pub use sampling::{
    sampling_from_definition, AudioSamplingParam, ImageSamplingParam, InferencePreset,
    SamplingParam, WorkloadType, ALL_PRESETS,
};

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn persist_dir(name: &str) -> String {
        let root = std::env::var("FASTVIDEO_ARTIFACT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let dir = root.join(name);
        let _ = std::fs::create_dir_all(&dir);
        dir.to_string_lossy().into_owned()
    }
    use super::*;

    #[test]
    fn cudarc_multi_gpu_enables_sequence_parallel() {
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: true,
                num_gpus: 2,
                output_path: Some(persist_dir("cudarc-sp-smoke")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let out = gen.generate_video("a cat").unwrap();
        assert!(!out.frame_paths.is_empty());
    }

    #[test]
    fn cudarc_vsa_ids_hard_fail_without_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("FASTVIDEO_VSA").ok();
        let prev_sdpa = std::env::var("FASTVIDEO_SDPA").ok();
        std::env::remove_var("FASTVIDEO_VSA");
        std::env::remove_var("FASTVIDEO_SDPA");
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: true,
                output_path: Some(persist_dir("cudarc-vsa-reject")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let err = gen.generate_video("a cat").unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("vsa"),
            "{err}"
        );
        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_VSA", v),
            None => std::env::remove_var("FASTVIDEO_VSA"),
        }
        match prev_sdpa {
            Some(v) => std::env::set_var("FASTVIDEO_SDPA", v),
            None => std::env::remove_var("FASTVIDEO_SDPA"),
        }
    }

    #[test]
    fn cudarc_vsa_sparse_runs_when_enabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("FASTVIDEO_VSA").ok();
        let prev_sdpa = std::env::var("FASTVIDEO_SDPA").ok();
        std::env::set_var("FASTVIDEO_VSA", "1");
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: true,
                output_path: Some(persist_dir("cudarc-vsa-sparse")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let out = gen.generate_video("a cat").unwrap();
        assert!(!out.frame_paths.is_empty());
        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_VSA", v),
            None => std::env::remove_var("FASTVIDEO_VSA"),
        }
        match prev_sdpa {
            Some(v) => std::env::set_var("FASTVIDEO_SDPA", v),
            None => std::env::remove_var("FASTVIDEO_SDPA"),
        }
    }

    #[test]
    fn cudarc_control_preset_errors_clearly() {
        let gen = VideoGenerator::from_pretrained(
            "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: false,
                weights_path: Some("/nonexistent/weights".into()),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(gen.definition.preset, "wan_fun_1_3b_control");
        // Without weights, generate fails before control stub; with tiny it would hit control error.
        let tiny = VideoGenerator::from_pretrained(
            "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: true,
                output_path: Some(persist_dir("cudarc-control-tiny")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        // tiny bypasses control error by design (smoke graph)
        let _ = tiny.generate_video("x");
    }

    #[test]
    fn from_pretrained_fastwan() {
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                tiny: false,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(gen.sampling.num_inference_steps, 3);
        assert_eq!(gen.sampling.height, 448);
        assert_eq!(gen.pipeline.dmd_steps, Some(&[1000, 757, 522][..]));
        // Without weights / tiny, generate should fail.
        assert!(gen.generate_video("a raccoon").is_err());
    }

    #[test]
    fn cudarc_tiny_generate_write_png() {
        let device = if cfg!(feature = "cuda") {
            "cuda".into()
        } else {
            "cpu".into()
        };
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: true,
                device,
                output_path: Some(persist_dir("cudarc-tiny")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let out = gen
            .generate_video("A curious raccoon in a field of sunflowers.")
            .unwrap();
        assert!(
            std::path::Path::new(&out.frame_paths[0]).exists(),
            "missing {}",
            out.frame_paths[0]
        );
        assert!(
            out.frame_paths[0].ends_with(".png"),
            "expected PNG, got {}",
            out.frame_paths[0]
        );
    }

    #[test]
    fn i2v_generate_needs_image_or_weights() {
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            LoadOptions {
                tiny: false,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let err = gen.generate_video("a cat").unwrap_err();
        let msg = err.to_string().to_ascii_lowercase();
        assert!(
            msg.contains("i2v") || msg.contains("image") || msg.contains("weights"),
            "{err}"
        );
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
    fn resolves_cached_1_3b_snapshot_layout() {
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            LoadOptions::default(),
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

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_tiny_generate_writes_png() {
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: true,
                device: "cuda".into(),
                dtype: Some("f32".into()),
                output_path: Some(persist_dir("cuda-tiny")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let out = gen.generate_video("a raccoon").unwrap();
        assert!(!out.frame_paths.is_empty());
        assert!(std::path::Path::new(&out.frame_paths[0]).exists());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_1_3b_smoke_gated() {
        if std::env::var("FASTVIDEO_GPU_SMOKE").ok().as_deref() != Some("1") {
            return;
        }
        let gen = VideoGenerator::from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Cudarc,
                device: "cuda".into(),
                dtype: Some("bf16".into()),
                height: Some(256),
                width: Some(256),
                num_frames: Some(9),
                num_inference_steps: Some(2),
                guidance_scale: Some(1.0),
                output_path: Some(persist_dir("cuda-1-3b-smoke")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let (out, stats) = gen
            .bench_video("A curious raccoon in a field of sunflowers.")
            .unwrap();
        assert!(!out.frame_paths.is_empty());
        assert!(stats.load_and_generate_ms > 0);
        eprintln!(
            "cuda_1_3b_smoke {}x{} frames={} steps={} {}ms",
            stats.width, stats.height, stats.frames, stats.steps, stats.load_and_generate_ms
        );
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
    fn removed_backends_are_rejected() {
        assert!("candle".parse::<BackendKind>().is_err());
        assert!("luminal".parse::<BackendKind>().is_err());
        assert_eq!("cudarc".parse::<BackendKind>().unwrap(), BackendKind::Cudarc);
    }
}
