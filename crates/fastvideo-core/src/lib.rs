//! FastVideo-rs inference core: registry, sampling, and VideoGenerator.

pub mod backend_kind;
pub mod error;
pub mod generator;
pub mod registry;
pub mod sampling;

pub use backend_kind::BackendKind;
pub use error::{FastVideoError, Result};
pub use generator::{
    median_u128, BenchStats, ClipBenchStats, GenerateOutput, LoadOptions, VideoGenerator,
};
pub use registry::{
    resolve_model, resolve_wan, Flux2ModelDefinition, ModelDefinition, ModelFamily, SamplingAlgorithm,
    WanModelDefinition, FLUX2_MODEL_DEFINITIONS, WAN_MODEL_DEFINITIONS,
};
pub use sampling::{sampling_from_definition, InferencePreset, SamplingParam, ALL_PRESETS};

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
                output_path: Some(persist_dir("host-unipc")),
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
    fn burn_luminal_cudarc_tiny_generate_write_png() {
        for backend in [BackendKind::Burn, BackendKind::Luminal, BackendKind::Cudarc] {
            let device = if cfg!(feature = "cuda")
                && matches!(backend, BackendKind::Burn | BackendKind::Cudarc)
            {
                "cuda".into()
            } else {
                "cpu".into()
            };
            let gen = VideoGenerator::from_pretrained(
                "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
                LoadOptions {
                    backend,
                    tiny: true,
                    device,
                    output_path: Some(persist_dir(&format!("{backend}-tiny"))),
                    ..LoadOptions::default()
                },
            )
            .unwrap();
            let out = gen
                .generate_video("A curious raccoon in a field of sunflowers.")
                .unwrap();
            assert!(
                std::path::Path::new(&out.frame_paths[0]).exists(),
                "{backend} missing {}",
                out.frame_paths[0]
            );
            assert!(
                out.frame_paths[0].ends_with(".png"),
                "{backend} expected PNG, got {}",
                out.frame_paths[0]
            );
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

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_tiny_generate_writes_png() {
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Candle,
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
                backend: BackendKind::Candle,
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
    fn flux2_registry_and_tiny_generate() {
        let gen = VideoGenerator::from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: true,
                output_path: Some(persist_dir("flux2-dev-tiny")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(gen.definition.family, crate::registry::ModelFamily::Flux2);
        assert_eq!(gen.definition.preset, "flux2_dev");
        assert_eq!(gen.sampling.height, 1024);
        assert_eq!(gen.sampling.num_frames, 1);
        let out = gen.generate_video("a banana on a table").unwrap();
        assert!(out.frame_paths[0].ends_with(".png"));
        assert!(std::path::Path::new(&out.frame_paths[0]).exists());

        let klein = VideoGenerator::from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            LoadOptions {
                backend: BackendKind::Candle,
                tiny: true,
                output_path: Some(persist_dir("flux2-klein-tiny")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(klein.definition.preset, "flux2_klein_4b");
        assert_eq!(klein.sampling.num_inference_steps, 4);
        let out = klein.generate_video("a banana on a table").unwrap();
        assert!(std::path::Path::new(&out.frame_paths[0]).exists());
    }

    #[test]
    fn candle_tiny_generate_writes_png() {
        let gen = VideoGenerator::from_pretrained(
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            LoadOptions {
                backend: BackendKind::Candle,
                tiny: true,
                output_path: Some(persist_dir("candle-tiny")),
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

    #[test]
    fn flux2_tiny_qwen3_candle_matches_cudarc() {
        use candle_core::{Device, DType};
        use candle_nn::VarBuilder;
        use fastvideo_models::flux2::{psnr, Flux2TextEncoder, Qwen3Config, Qwen3Encoder};

        let ids = [1u32, 2, 3, 4];
        let candle_enc =
            Qwen3Encoder::load(Qwen3Config::tiny(), VarBuilder::zeros(DType::F32, &Device::Cpu)).unwrap();
        let candle = Flux2TextEncoder::qwen3(candle_enc, DType::F32)
            .encode_ids(&ids)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let gpu = fastvideo_cudarc::flux2::Flux2TextEncoder::qwen3(
            fastvideo_cudarc::flux2::Qwen3Encoder::zeros(Qwen3Config::tiny()),
        )
        .encode_ids(&ids, None)
        .unwrap()
        .host_cow()
        .unwrap()
        .to_vec();
        assert_eq!(candle.len(), gpu.len());
        let db = psnr(&candle, &gpu, 1.0).unwrap();
        assert!(db.is_infinite() || db > 60.0, "qwen3 psnr={db}");
    }

    #[test]
    fn flux2_small_vae_candle_matches_cudarc() {
        use candle_core::{Device, DType, Tensor};
        use candle_nn::VarBuilder;
        use fastvideo_models::flux2::{psnr, AutoencoderKlFlux2, Flux2VaeConfig};

        let cfg = Flux2VaeConfig::small();
        let candle = AutoencoderKlFlux2::load(cfg.clone(), VarBuilder::zeros(DType::F32, &Device::Cpu)).unwrap();
        let gpu = fastvideo_cudarc::flux2::AutoencoderKlFlux2::zeros(cfg.clone());
        let z_c = Tensor::zeros((1, cfg.latent_channels, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let z_g = fastvideo_cudarc::CudaTensor::zeros(&[1, cfg.latent_channels, 1, 2, 2]);
        let a = candle
            .decode(&z_c)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = gpu.decode(&z_g).unwrap().host_cow().unwrap().to_vec();
        assert_eq!(a.len(), b.len());
        let db = psnr(&a, &b, 1.0).unwrap();
        assert!(db.is_infinite() || db > 60.0, "vae psnr={db}");
    }

    #[test]
    fn median_u128_matches_python_statistics() {
        assert_eq!(crate::median_u128(&[]), 0);
        assert_eq!(crate::median_u128(&[7]), 7);
        assert_eq!(crate::median_u128(&[10, 20]), 15);
        assert_eq!(crate::median_u128(&[3, 1, 2]), 2);
        assert_eq!(crate::median_u128(&[8, 2, 6, 4]), 5);
    }

    #[test]
    fn flux2_tiny_bench_warm_median() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_sdpa = std::env::var("FASTVIDEO_SDPA").ok();
        std::env::set_var("FASTVIDEO_SDPA", "dense");
        struct RestoreSdpa(Option<String>);
        impl Drop for RestoreSdpa {
            fn drop(&mut self) {
                match &self.0 {
                    Some(v) => std::env::set_var("FASTVIDEO_SDPA", v),
                    None => std::env::remove_var("FASTVIDEO_SDPA"),
                }
            }
        }
        let _restore = RestoreSdpa(prev_sdpa);
        let gen = VideoGenerator::from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            LoadOptions {
                backend: BackendKind::Cudarc,
                tiny: true,
                output_path: Some(persist_dir("flux2-klein-tiny-bench")),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let (out, stats) = gen
            .bench_video_with("a banana on a table", 1, 2)
            .unwrap();
        assert!(std::path::Path::new(&out.frame_paths[0]).exists());
        assert_eq!(stats.warmup, 1);
        assert_eq!(stats.runs, 2);
        assert_eq!(stats.runs_ms.len(), 2);
        assert_eq!(stats.generate_ms, stats.runs_ms[1]);
        assert_eq!(stats.median_ms, crate::median_u128(&stats.runs_ms));
        assert_eq!(stats.min_ms, *stats.runs_ms.iter().min().unwrap());
        let label = crate::generator::sdpa_backend_label();
        assert_eq!(stats.sdpa, label);
        assert_eq!(label, "dense");
        let profile = std::path::Path::new(&gen.output_path).join("profile.json");
        assert!(profile.exists(), "profile.json from last timed run");
    }
}
