use std::path::PathBuf;
use std::time::Instant;

use crate::backend_kind::BackendKind;
use crate::error::{FastVideoError, Result};
use crate::registry::{resolve_wan, SamplingAlgorithm, WanModelDefinition};
use crate::sampling::{pipeline_defaults, sampling_from_definition, PipelineDefaults, SamplingParam};

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub backend: BackendKind,
    pub num_gpus: u32,
    pub tiny: bool,
    pub weights_path: Option<String>,
    pub output_path: Option<String>,
    /// `cpu`, `cuda`, or `cuda:0`.
    pub device: String,
    /// `f32`, `f16`, or `bf16`. Empty means bf16 on CUDA, f32 on CPU.
    pub dtype: Option<String>,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub num_frames: Option<u32>,
    pub num_inference_steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub guidance_scale_2: Option<f32>,
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    pub image_path: Option<String>,
    /// Control / reference frame for Fun Control / Lucy.
    pub control_path: Option<String>,
    /// Mux PNG frames to output.mp4 via ffmpeg when true (or FASTVIDEO_SAVE_MP4=1).
    pub save_mp4: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            backend: BackendKind::Cudarc,
            num_gpus: 1,
            tiny: false,
            weights_path: None,
            output_path: None,
            device: "cpu".into(),
            dtype: None,
            height: None,
            width: None,
            num_frames: None,
            num_inference_steps: None,
            guidance_scale: None,
            guidance_scale_2: None,
            seed: None,
            negative_prompt: None,
            image_path: None,
            control_path: None,
            save_mp4: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoGenerator {
    pub model_id: String,
    pub definition: &'static WanModelDefinition,
    pub sampling: SamplingParam,
    pub pipeline: PipelineDefaults,
    pub backend: BackendKind,
    pub num_gpus: u32,
    pub tiny: bool,
    pub weights_path: Option<String>,
    pub output_path: String,
    pub device: String,
    pub dtype: Option<String>,
    pub image_path: Option<String>,
    pub control_path: Option<String>,
    pub save_mp4: bool,
}

impl VideoGenerator {
    pub fn from_pretrained(model_id: impl Into<String>, opts: LoadOptions) -> Result<Self> {
        let model_id = model_id.into();
        let definition = resolve_wan(&model_id)?;
        let mut sampling = sampling_from_definition(definition);
        let pipeline = pipeline_defaults(definition);
        if let Some(path) = &opts.output_path {
            sampling.output_path = path.clone();
        }
        if let Some(h) = opts.height {
            sampling.height = h;
        }
        if let Some(w) = opts.width {
            sampling.width = w;
        }
        if let Some(f) = opts.num_frames {
            sampling.num_frames = f;
        }
        if let Some(s) = opts.num_inference_steps {
            sampling.num_inference_steps = s;
        }
        if let Some(g) = opts.guidance_scale {
            sampling.guidance_scale = g;
        }
        if let Some(g) = opts.guidance_scale_2 {
            sampling.guidance_scale_2 = Some(g);
        }
        if let Some(seed) = opts.seed {
            sampling.seed = seed;
        }
        if let Some(neg) = opts.negative_prompt {
            sampling.negative_prompt = neg;
        }
        if opts.save_mp4 {
            sampling.save_video = true;
        }
        Ok(Self {
            model_id,
            definition,
            sampling,
            pipeline,
            backend: opts.backend,
            num_gpus: opts.num_gpus,
            tiny: opts.tiny,
            weights_path: opts.weights_path,
            output_path: opts.output_path.unwrap_or_else(|| "outputs".to_string()),
            device: opts.device,
            dtype: opts.dtype,
            image_path: opts.image_path,
            control_path: opts.control_path,
            save_mp4: opts.save_mp4,
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "model={} preset={} sampling={} backend={} tiny={} device={} {}x{} frames={} steps={} shift={}",
            self.model_id,
            self.definition.preset,
            self.definition.sampling.as_str(),
            self.backend,
            self.tiny,
            self.device,
            self.sampling.width,
            self.sampling.height,
            self.sampling.num_frames,
            self.sampling.num_inference_steps,
            self.pipeline.flow_shift
        )
    }

    /// Generate video frames via cudarc CUDA.
    pub fn generate_video(&self, prompt: &str) -> Result<GenerateOutput> {
        if self.backend != BackendKind::Cudarc {
            return Err(FastVideoError::Message(format!(
                "backend {} is removed; use --backend cudarc",
                self.backend
            )));
        }
        Ok(self.run_cudarc(prompt)?.0)
    }

    /// Local Diffusers root: `--weights`, `FASTVIDEO_WEIGHTS`, or the HF hub snapshot.
    pub fn resolved_weights_dir(&self) -> Option<PathBuf> {
        if self.tiny {
            return None;
        }
        if let Some(path) = &self.weights_path {
            let p = PathBuf::from(path);
            if p.join("transformer").is_dir() {
                return Some(p);
            }
        }
        if let Ok(path) = std::env::var("FASTVIDEO_WEIGHTS") {
            let p = PathBuf::from(path);
            if p.join("transformer").is_dir() {
                return Some(p);
            }
        }
        fastvideo_models::wan::weights::hf_snapshot(&self.model_id)
            .filter(|p| p.join("transformer").is_dir())
    }

    /// Returns `(output, load_ms, generate_ms)`.
    fn run_cudarc(&self, prompt: &str) -> Result<(GenerateOutput, u128, u128)> {
        let is_dmd = matches!(
            self.definition.sampling,
            SamplingAlgorithm::Dmd | SamplingAlgorithm::CausalDmd
        );
        let is_rcm = matches!(self.definition.sampling, SamplingAlgorithm::Rcm);
        if self.num_gpus > 1 {
            std::env::set_var("FASTVIDEO_SP_WORLD", self.num_gpus.to_string());
            eprintln!(
                "info: enabling sequence parallel world={} (FASTVIDEO_SP_WORLD)",
                self.num_gpus
            );
        } else {
            std::env::remove_var("FASTVIDEO_SP_WORLD");
        }
        if self.model_id.to_ascii_lowercase().contains("vsa") {
            let vsa_on = std::env::var("FASTVIDEO_VSA")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !vsa_on {
                return Err(FastVideoError::Message(
                    "VSA checkpoint detected: set FASTVIDEO_VSA=1 to enable block-sparse \
                     attention (in-tree), or use a dense Wan model id."
                        .into(),
                ));
            }
            std::env::set_var("FASTVIDEO_SDPA", "sparse");
            eprintln!("info: VSA id → block-sparse SDPA (FASTVIDEO_VSA=1)");
        }
        fastvideo_cudarc::resolve_device(&self.device)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        eprintln!(
            "[fastvideo] cudarc run device={} tiny={} {}x{} frames={} steps={} log={}",
            self.device,
            self.tiny,
            self.sampling.width,
            self.sampling.height,
            self.sampling.num_frames,
            self.sampling.num_inference_steps,
            std::env::var("FASTVIDEO_LOG").unwrap_or_else(|_| "info".into()),
        );
        let mut gen_cfg = fastvideo_cudarc::GenerateConfig {
            prompt: prompt.to_string(),
            negative_prompt: self.sampling.negative_prompt.clone(),
            height: self.sampling.height as usize,
            width: self.sampling.width as usize,
            num_frames: self.sampling.num_frames as usize,
            num_inference_steps: self.sampling.num_inference_steps as usize,
            guidance_scale: self.sampling.guidance_scale,
            seed: self.sampling.seed,
            output_dir: self.output_path.clone(),
            tiny: self.tiny,
            is_dmd,
            is_rcm,
            flow_shift: f64::from(self.pipeline.flow_shift),
            dmd_steps: self.pipeline.dmd_steps.map(|s| s.to_vec()),
            rcm_sigma_max: self.pipeline.rcm_sigma_max,
            tokenizer_path: None,
            image_path: self.image_path.clone(),
            control_path: self.control_path.clone(),
            guidance_scale_2: self.sampling.guidance_scale_2,
            boundary_ratio: self.pipeline.boundary_ratio,
            save_video: self.save_mp4
                || std::env::var("FASTVIDEO_SAVE_MP4")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false),
            fps: self.sampling.fps,
        };
        if self.tiny {
            gen_cfg.guidance_scale = 1.0;
            gen_cfg.is_dmd = true;
            gen_cfg.flow_shift = 8.0;
            let t_gen = Instant::now();
            let mut pipe = fastvideo_cudarc::WanPipeline::tiny();
            let frames = pipe
                .generate(&gen_cfg)
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
            let generate_ms = t_gen.elapsed().as_millis();
            return Ok((
                GenerateOutput {
                    output_path: frames.first().cloned(),
                    frame_paths: frames,
                },
                0,
                generate_ms,
            ));
        }
        let root = self.resolved_weights_dir().ok_or_else(|| {
            FastVideoError::Message(
                "cudarc generate needs --weights <diffusers-dir> or a cached HF snapshot".into(),
            )
        })?;
        let tok = root.join("tokenizer").join("tokenizer.json");
        if !tok.is_file() {
            return Err(FastVideoError::Message(format!(
                "missing {}/tokenizer/tokenizer.json",
                root.display()
            )));
        }
        gen_cfg.tokenizer_path = Some(tok.to_string_lossy().into_owned());
        let t_load = Instant::now();
        let mut pipe = fastvideo_cudarc::WanPipeline::load(&root, self.definition.preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let load_ms = t_load.elapsed().as_millis();
        let t_gen = Instant::now();
        let frames = pipe
            .generate(&gen_cfg)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let generate_ms = t_gen.elapsed().as_millis();
        Ok((
            GenerateOutput {
                output_path: frames.first().cloned(),
                frame_paths: frames,
            },
            load_ms,
            generate_ms,
        ))
    }

    /// Split weight materialization from sampling.
    pub fn bench_video(&self, prompt: &str) -> Result<(GenerateOutput, BenchStats)> {
        let (out, load_ms, generate_ms) = self.run_cudarc(prompt)?;
        Ok((
            out.clone(),
            BenchStats {
                model: self.model_id.clone(),
                backend: self.backend.as_str().to_string(),
                device: self.device.clone(),
                dtype: self.dtype.clone().unwrap_or_else(|| "default".into()),
                height: self.sampling.height,
                width: self.sampling.width,
                frames: self.sampling.num_frames,
                steps: self.sampling.num_inference_steps,
                load_ms,
                generate_ms,
                load_and_generate_ms: load_ms.saturating_add(generate_ms),
                frames_written: out.frame_paths.len() as u32,
            },
        ))
    }

    /// CLIP ViT-H encode only (`image_encoder/`).
    pub fn bench_clip(&self, image: &str) -> Result<ClipBenchStats> {
        let root = self.resolved_clip_dir().ok_or_else(|| {
            FastVideoError::Message(
                "CLIP bench needs image_encoder/ (I2V Diffusers snapshot or --weights)".into(),
            )
        })?;
        fastvideo_cudarc::resolve_device(&self.device)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let t0 = Instant::now();
        let map = fastvideo_cudarc::wan::weights::WeightMap::from_dir(&root.join("image_encoder"))
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let clip = fastvideo_cudarc::wan::ClipVision::load(
            fastvideo_cudarc::wan::ClipVisionConfig::vit_h_14(),
            &map,
        )
        .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let load_ms = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let tokens = clip
            .encode_image_file(image)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let encode_ms = t1.elapsed().as_millis();
        Ok(ClipBenchStats {
            hidden: tokens.shape.clone(),
            load_ms,
            encode_ms,
            path: root.display().to_string(),
        })
    }

    pub fn resolved_clip_dir(&self) -> Option<PathBuf> {
        let candidates = [
            self.weights_path.as_ref().map(PathBuf::from),
            std::env::var("FASTVIDEO_WEIGHTS").ok().map(PathBuf::from),
            fastvideo_models::wan::weights::hf_snapshot(&self.model_id),
            fastvideo_models::wan::weights::hf_snapshot("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"),
        ];
        candidates.into_iter().flatten().find(|p| {
            p.join("image_encoder").is_dir()
                && fastvideo_loader::collect_safetensors(&p.join("image_encoder"))
                    .map(|f| !f.is_empty())
                    .unwrap_or(false)
        })
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub output_path: Option<String>,
    pub frame_paths: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BenchStats {
    pub model: String,
    pub backend: String,
    pub device: String,
    pub dtype: String,
    pub height: u32,
    pub width: u32,
    pub frames: u32,
    pub steps: u32,
    /// Diffusers safetensors → in-memory tensors (not Hub network pull).
    pub load_ms: u128,
    /// Sampling + decode after weights are resident.
    pub generate_ms: u128,
    pub load_and_generate_ms: u128,
    pub frames_written: u32,
}

#[derive(Debug, Clone)]
pub struct ClipBenchStats {
    pub hidden: Vec<usize>,
    pub load_ms: u128,
    pub encode_ms: u128,
    pub path: String,
}
