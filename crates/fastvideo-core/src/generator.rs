use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device};
use fastvideo_loader::load_diffusers_components;
use fastvideo_models::{
    tokenize_prompt, ClipVision, ClipVisionConfig, FlowUniPCMultistepScheduler, GenerateConfig,
    Umt5Config, WanPipeline, WanVaeConfig, WanVideoArchConfig,
};
use fastvideo_ops::{HostBackend, TensorBackend};

use crate::backend_kind::BackendKind;
use crate::error::{FastVideoError, Result};
use crate::registry::{resolve_model, ModelFamily, SamplingAlgorithm, WanModelDefinition};
use crate::sampling::{pipeline_defaults, sampling_from_definition, PipelineDefaults, SamplingParam, WorkloadType};

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
        let definition = resolve_model(&model_id)?;
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
            "model={} family={} preset={} sampling={} backend={} tiny={} device={} {}x{} frames={} steps={} shift={}",
            self.model_id,
            self.definition.family.as_str(),
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

    /// Generate video frames. **cudarc is the supported path**; Burn / Candle /
    /// Luminal remain frozen reference backends (no new Wan features).
    pub fn generate_video(&self, prompt: &str) -> Result<GenerateOutput> {
        match self.backend {
            BackendKind::Cudarc => self.generate_cudarc(prompt),
            BackendKind::Candle => self.generate_candle(prompt),
            BackendKind::Burn => self.generate_burn(prompt),
            BackendKind::Luminal => self.generate_luminal(prompt),
            BackendKind::Host => self.generate_reference(prompt),
        }
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
        fastvideo_models::wan::weights::hf_snapshot(&self.model_id).filter(|root| {
            root.join("transformer").is_dir()
        })
    }

    fn generate_candle(&self, prompt: &str) -> Result<GenerateOutput> {
        Ok(self.run_candle(prompt)?.0)
    }

    fn run_candle(&self, prompt: &str) -> Result<(GenerateOutput, u128, u128)> {
        if self.definition.family == ModelFamily::Flux2 {
            return self.run_candle_flux2(prompt);
        }
        self.run_candle_wan(prompt)
    }

    fn run_candle_flux2(&self, prompt: &str) -> Result<(GenerateOutput, u128, u128)> {
        let device = resolve_candle_device(&self.device)?;
        let dtype = resolve_dtype(self.dtype.as_deref(), &self.device)?;
        let kind = fastvideo_models::flux2::Flux2TextKind::from_preset(self.definition.preset);
        let weights = if self.tiny {
            None
        } else {
            Some(self.resolved_weights_dir().ok_or_else(|| {
                FastVideoError::Message(
                    "pass --tiny (zero weights, CI), --weights <diffusers-dir>, or cache the Hugging Face snapshot".into(),
                )
            })?)
        };
        let tokenizer_path = weights.as_ref().and_then(|root| {
            let p = root.join("tokenizer").join("tokenizer.json");
            p.exists().then(|| p.to_string_lossy().into_owned())
        });
        let mut gen_cfg = fastvideo_models::flux2::GenerateConfig {
            prompt: prompt.to_string(),
            height: self.sampling.height as usize,
            width: self.sampling.width as usize,
            num_inference_steps: self.sampling.num_inference_steps as usize,
            guidance_scale: self.sampling.guidance_scale,
            seed: self.sampling.seed,
            output_dir: self.output_path.clone(),
            tiny: self.tiny,
            tokenizer_path,
            embedded_cfg_scale: self.pipeline.embedded_cfg_scale,
        };
        if self.tiny {
            gen_cfg.guidance_scale = 1.0;
        }
        let t_load = Instant::now();
        let pipe = if self.tiny {
            match kind {
                fastvideo_models::flux2::Flux2TextKind::Qwen3 => {
                    fastvideo_models::flux2::Flux2Pipeline::tiny_klein(&device).map_err(candle_err)?
                }
                fastvideo_models::flux2::Flux2TextKind::Mistral3 => {
                    fastvideo_models::flux2::Flux2Pipeline::tiny(&device).map_err(candle_err)?
                }
            }
        } else {
            let root = weights.as_ref().expect("resolved");
            let components = load_diffusers_components(root, dtype, &device)
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
            let cfg_json = std::fs::read_to_string(root.join("transformer/config.json")).ok();
            let text_cfg = std::fs::read_to_string(root.join("text_encoder/config.json"))
                .ok()
                .or_else(|| std::fs::read_to_string(root.join("text_encoder_2/config.json")).ok());
            let text_vb = Some(components.text);
            fastvideo_models::flux2::Flux2Pipeline::load_with_text_config(
                components.transformer,
                components.vae,
                text_vb,
                fastvideo_models::flux2::Flux2ArchConfig::from_preset(self.definition.preset),
                fastvideo_models::flux2::Flux2VaeConfig::flux2(),
                kind,
                device,
                cfg_json.as_deref(),
                text_cfg.as_deref(),
            )
            .map_err(candle_err)?
        };
        let load_ms = t_load.elapsed().as_millis();
        let t_gen = Instant::now();
        let frames = pipe.generate(&gen_cfg).map_err(candle_err)?;
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

    fn run_candle_wan(&self, prompt: &str) -> Result<(GenerateOutput, u128, u128)> {
        let device = resolve_candle_device(&self.device)?;
        let dtype = resolve_dtype(self.dtype.as_deref(), &self.device)?;
        let is_dmd = matches!(
            self.definition.sampling,
            SamplingAlgorithm::Dmd | SamplingAlgorithm::CausalDmd
        );
        let is_i2v = self.definition.workload_types.contains(&WorkloadType::I2V);
        if is_i2v && !self.tiny && self.image_path.is_none() {
            return Err(FastVideoError::NotImplemented {
                component: "I2V generate".into(),
                detail: "pass --image <png|jpeg>; CLIP ViT-H + VAE 36-channel pack run when image_encoder/ is present".into(),
            });
        }
        let weights = if self.tiny {
            None
        } else {
            Some(self.resolved_weights_dir().ok_or_else(|| {
                FastVideoError::Message(
                    "pass --tiny (zero weights, CI), --weights <diffusers-dir>, or cache the Hugging Face snapshot".into(),
                )
            })?)
        };
        let tokenizer_path = weights.as_ref().and_then(|root| {
            let p = root.join("tokenizer").join("tokenizer.json");
            p.exists().then(|| p.to_string_lossy().into_owned())
        });
        let mut gen_cfg = GenerateConfig {
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
            flow_shift: f64::from(self.pipeline.flow_shift),
            dmd_steps: self.pipeline.dmd_steps.map(|s| s.to_vec()),
            tokenizer_path,
            image_path: self.image_path.clone(),
            guidance_scale_2: self.sampling.guidance_scale_2,
            boundary_ratio: self.pipeline.boundary_ratio,
        };
        if self.tiny {
            gen_cfg.guidance_scale = 1.0;
            gen_cfg.is_dmd = true;
            gen_cfg.flow_shift = 8.0;
        }
        let t_load = Instant::now();
        let pipe = if self.tiny {
            WanPipeline::tiny_dtype(&device, dtype).map_err(candle_err)?
        } else {
            let root = weights.as_ref().expect("resolved");
            if gen_cfg.tokenizer_path.is_none() {
                return Err(FastVideoError::Message(format!(
                    "missing {}/tokenizer/tokenizer.json",
                    root.display()
                )));
            }
            let components = load_diffusers_components(root, dtype, &device)
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
            WanPipeline::load(
                components.transformer,
                components.vae,
                components.text,
                WanVideoArchConfig::from_preset(self.definition.preset),
                WanVaeConfig::wan_2_1(),
                Umt5Config::xxl(),
                device,
                components.transformer_2,
                components.image_encoder,
            )
            .map_err(candle_err)?
        };
        let load_ms = t_load.elapsed().as_millis();
        let t_gen = Instant::now();
        let frames = pipe.generate(&gen_cfg).map_err(candle_err)?;
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

    fn generate_burn(&self, prompt: &str) -> Result<GenerateOutput> {
        Ok(self.run_burn(prompt)?.0)
    }

    fn run_burn(&self, prompt: &str) -> Result<(GenerateOutput, u128, u128)> {
        if self.definition.family == ModelFamily::Flux2 {
            return Err(FastVideoError::NotImplemented {
                component: "Flux2 Burn".into(),
                detail: "Burn is frozen; use --backend cudarc or candle".into(),
            });
        }
        let is_dmd = matches!(
            self.definition.sampling,
            SamplingAlgorithm::Dmd | SamplingAlgorithm::CausalDmd
        );
        if self.definition.workload_types.contains(&WorkloadType::I2V) && !self.tiny {
            return Err(FastVideoError::NotImplemented {
                component: "I2V generate".into(),
                detail: "Burn T2V only; use --backend candle for I2V".into(),
            });
        }
        let mut gen_cfg = fastvideo_burn::GenerateConfig {
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
            flow_shift: f64::from(self.pipeline.flow_shift),
            dmd_steps: self.pipeline.dmd_steps.map(|s| s.to_vec()),
            tokenizer_path: None,
            image_path: self.image_path.clone(),
            guidance_scale_2: self.sampling.guidance_scale_2,
            boundary_ratio: self.pipeline.boundary_ratio,
        };
        let device = fastvideo_burn::resolve_device(&self.device)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if self.tiny {
            gen_cfg.guidance_scale = 1.0;
            gen_cfg.is_dmd = true;
            gen_cfg.flow_shift = 8.0;
            let t_load = Instant::now();
            let pipe = fastvideo_burn::WanPipeline::tiny_on(&device)
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
            let load_ms = t_load.elapsed().as_millis();
            let t_gen = Instant::now();
            let frames = pipe
                .generate(&gen_cfg)
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
            let generate_ms = t_gen.elapsed().as_millis();
            return Ok((
                GenerateOutput {
                    output_path: frames.first().cloned(),
                    frame_paths: frames,
                },
                load_ms,
                generate_ms,
            ));
        }
        let root = self.resolved_weights_dir().ok_or_else(|| {
            FastVideoError::Message(
                "Burn generate needs --weights <diffusers-dir> or a cached HF snapshot".into(),
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
        let pipe = fastvideo_burn::WanPipeline::load_on(&root, &device)
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

    fn generate_luminal(&self, prompt: &str) -> Result<GenerateOutput> {
        Ok(self.run_luminal(prompt)?.0)
    }

    fn run_luminal(&self, prompt: &str) -> Result<(GenerateOutput, u128, u128)> {
        if self.definition.family == ModelFamily::Flux2 {
            return Err(FastVideoError::NotImplemented {
                component: "Flux2 Luminal".into(),
                detail: "Luminal is frozen; use --backend cudarc or candle".into(),
            });
        }
        let is_dmd = matches!(
            self.definition.sampling,
            SamplingAlgorithm::Dmd | SamplingAlgorithm::CausalDmd
        );
        if self.definition.workload_types.contains(&WorkloadType::I2V) && !self.tiny {
            return Err(FastVideoError::NotImplemented {
                component: "I2V generate".into(),
                detail: "Luminal T2V only; use --backend candle for I2V".into(),
            });
        }
        let mut gen_cfg = fastvideo_luminal::GenerateConfig {
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
            flow_shift: f64::from(self.pipeline.flow_shift),
            dmd_steps: self.pipeline.dmd_steps.map(|s| s.to_vec()),
            tokenizer_path: None,
        };
        if self.tiny {
            gen_cfg.guidance_scale = 1.0;
            gen_cfg.is_dmd = true;
            gen_cfg.flow_shift = 8.0;
            let t_gen = Instant::now();
            let mut pipe = fastvideo_luminal::WanPipeline::tiny();
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
                "Luminal generate needs --weights <diffusers-dir> or a cached HF snapshot".into(),
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
        let mut pipe = fastvideo_luminal::WanPipeline::load(&root)
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

    fn generate_cudarc(&self, prompt: &str) -> Result<GenerateOutput> {
        Ok(self.run_cudarc(prompt)?.0)
    }

    /// Returns `(output, load_ms, generate_ms)`. Load is Diffusers safetensors → tensors.
    fn run_cudarc(&self, prompt: &str) -> Result<(GenerateOutput, u128, u128)> {
        let (mut loaded, load_ms) = self.load_cudarc(prompt)?;
        let t_gen = Instant::now();
        let frames = loaded.generate()?;
        Ok((output_from_frames(frames), load_ms, t_gen.elapsed().as_millis()))
    }

    fn load_cudarc(&self, prompt: &str) -> Result<(CudarcLoaded, u128)> {
        if self.definition.family == ModelFamily::Flux2 {
            return self.load_cudarc_flux2(prompt);
        }
        self.load_cudarc_wan(prompt)
    }

    fn load_cudarc_flux2(&self, prompt: &str) -> Result<(CudarcLoaded, u128)> {
        fastvideo_cudarc::resolve_device(&self.device)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        eprintln!(
            "[fastvideo] cudarc flux2 run device={} tiny={} preset={} {}x{} steps={}",
            self.device,
            self.tiny,
            self.definition.preset,
            self.sampling.width,
            self.sampling.height,
            self.sampling.num_inference_steps,
        );
        let mut gen_cfg = fastvideo_cudarc::flux2::GenerateConfig {
            prompt: prompt.to_string(),
            height: self.sampling.height as usize,
            width: self.sampling.width as usize,
            num_inference_steps: self.sampling.num_inference_steps as usize,
            guidance_scale: self.sampling.guidance_scale,
            seed: self.sampling.seed,
            output_dir: self.output_path.clone(),
            tiny: self.tiny,
            tokenizer_path: None,
            embedded_cfg_scale: self.pipeline.embedded_cfg_scale,
            preset: self.definition.preset.to_string(),
        };
        if self.tiny {
            gen_cfg.guidance_scale = 1.0;
            let pipe = fastvideo_cudarc::flux2::Flux2Pipeline::tiny_for_preset(self.definition.preset);
            return Ok((CudarcLoaded::Flux2 { pipe, cfg: gen_cfg }, 0));
        }
        let root = self.resolved_weights_dir().ok_or_else(|| {
            FastVideoError::Message(
                "cudarc Flux2 generate needs --weights <diffusers-dir> or a cached HF snapshot".into(),
            )
        })?;
        let tok = root.join("tokenizer").join("tokenizer.json");
        if tok.is_file() {
            gen_cfg.tokenizer_path = Some(tok.to_string_lossy().into_owned());
        }
        let t_load = Instant::now();
        let pipe = fastvideo_cudarc::flux2::Flux2Pipeline::load(&root, self.definition.preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok((CudarcLoaded::Flux2 { pipe, cfg: gen_cfg }, t_load.elapsed().as_millis()))
    }

    fn load_cudarc_wan(&self, prompt: &str) -> Result<(CudarcLoaded, u128)> {
        let is_dmd = matches!(
            self.definition.sampling,
            SamplingAlgorithm::Dmd | SamplingAlgorithm::CausalDmd
        );
        if self.num_gpus > 1 {
            // Sequence-parallel smoke: shard SDPA query dim across logical ranks.
            std::env::set_var("FASTVIDEO_SP_WORLD", self.num_gpus.to_string());
            eprintln!(
                "info: enabling sequence parallel world={} (FASTVIDEO_SP_WORLD)",
                self.num_gpus
            );
        } else {
            std::env::remove_var("FASTVIDEO_SP_WORLD");
        }
        // VSA: require opt-in, then use block-sparse SDPA (in-tree), not silent dense.
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
            flow_shift: f64::from(self.pipeline.flow_shift),
            dmd_steps: self.pipeline.dmd_steps.map(|s| s.to_vec()),
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
            let pipe = fastvideo_cudarc::WanPipeline::tiny();
            return Ok((CudarcLoaded::Wan { pipe, cfg: gen_cfg }, 0));
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
        let pipe = fastvideo_cudarc::WanPipeline::load(&root, self.definition.preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok((CudarcLoaded::Wan { pipe, cfg: gen_cfg }, t_load.elapsed().as_millis()))
    }

    /// Split weight materialization from sampling. Use on Vast CUDA, not laptop CPU.
    /// One generate after load (cold). Prefer [`Self::bench_video_with`] for the
    /// upstream-matching warmup + median protocol.
    pub fn bench_video(&self, prompt: &str) -> Result<(GenerateOutput, BenchStats)> {
        self.bench_video_with(prompt, 0, 1)
    }

    /// Load once, discard `warmup` generates, then time `runs` generates.
    ///
    /// `warmup_ms` is the first discarded generate (cold when `warmup >= 1`).
    /// `generate_ms` is the **last timed run** — the same generate that writes
    /// `profile.json`. `median_ms` / `min_ms` are over `runs_ms` and are the
    /// numbers to compare with upstream `median_seconds` / `min_seconds`.
    pub fn bench_video_with(
        &self,
        prompt: &str,
        warmup: u32,
        runs: u32,
    ) -> Result<(GenerateOutput, BenchStats)> {
        let runs = runs.max(1);
        match self.backend {
            BackendKind::Cudarc => self.bench_cudarc_with(prompt, warmup, runs),
            other => {
                let (out, load_ms, first_ms) = match other {
                    BackendKind::Candle => self.run_candle(prompt)?,
                    BackendKind::Burn => self.run_burn(prompt)?,
                    BackendKind::Luminal => self.run_luminal(prompt)?,
                    BackendKind::Host => {
                        let t0 = Instant::now();
                        let out = self.generate_reference(prompt)?;
                        (out, 0, t0.elapsed().as_millis())
                    }
                    BackendKind::Cudarc => unreachable!(),
                };
                if warmup == 0 && runs == 1 {
                    return Ok((
                        out.clone(),
                        self.bench_stats(load_ms, 0, 0, vec![first_ms], &out),
                    ));
                }
                // Non-cudarc backends reload per call; extra generates still
                // produce a median so CLI --warmup/--runs stay meaningful.
                let mut warmup_ms = 0u128;
                if warmup > 0 {
                    warmup_ms = first_ms;
                    for _ in 1..warmup {
                        let t = Instant::now();
                        let _ = self.generate_video(prompt)?;
                        let _ = t.elapsed();
                    }
                }
                let mut runs_ms = Vec::with_capacity(runs as usize);
                let mut last = out;
                let start = if warmup == 0 {
                    runs_ms.push(first_ms);
                    1
                } else {
                    0
                };
                for _ in start..runs {
                    let t = Instant::now();
                    last = self.generate_video(prompt)?;
                    runs_ms.push(t.elapsed().as_millis());
                }
                Ok((
                    last.clone(),
                    self.bench_stats(load_ms, warmup, warmup_ms, runs_ms, &last),
                ))
            }
        }
    }

    fn bench_cudarc_with(
        &self,
        prompt: &str,
        warmup: u32,
        runs: u32,
    ) -> Result<(GenerateOutput, BenchStats)> {
        let (mut loaded, load_ms) = self.load_cudarc(prompt)?;
        let mut warmup_ms = 0u128;
        for i in 0..warmup {
            let t = Instant::now();
            let _ = loaded.generate()?;
            let dt = t.elapsed().as_millis();
            if i == 0 {
                warmup_ms = dt;
            }
        }
        let mut runs_ms = Vec::with_capacity(runs as usize);
        let mut last_frames = Vec::new();
        for _ in 0..runs {
            let t = Instant::now();
            last_frames = loaded.generate()?;
            runs_ms.push(t.elapsed().as_millis());
        }
        let out = output_from_frames(last_frames);
        Ok((
            out.clone(),
            self.bench_stats(load_ms, warmup, warmup_ms, runs_ms, &out),
        ))
    }

    fn bench_stats(
        &self,
        load_ms: u128,
        warmup: u32,
        warmup_ms: u128,
        runs_ms: Vec<u128>,
        out: &GenerateOutput,
    ) -> BenchStats {
        let generate_ms = runs_ms.last().copied().unwrap_or(0);
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
            warmup,
            runs: runs_ms.len() as u32,
            warmup_ms,
            median_ms: median_u128(&runs_ms),
            min_ms: runs_ms.iter().copied().min().unwrap_or(0),
            runs_ms,
            sdpa: sdpa_backend_label(),
        }
    }

    /// CLIP ViT-H encode only (`image_encoder/`). Prefers cudarc when backend is cudarc.
    pub fn bench_clip(&self, image: &str) -> Result<ClipBenchStats> {
        let root = self.resolved_clip_dir().ok_or_else(|| {
            FastVideoError::Message(
                "CLIP bench needs image_encoder/ (I2V Diffusers snapshot or --weights)".into(),
            )
        })?;
        if self.backend == BackendKind::Cudarc {
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
            return Ok(ClipBenchStats {
                hidden: tokens.shape.clone(),
                load_ms,
                encode_ms,
                path: root.display().to_string(),
            });
        }
        let device = resolve_candle_device(&self.device)?;
        let dtype = resolve_dtype(self.dtype.as_deref(), &self.device)?;
        let t0 = Instant::now();
        let vb = fastvideo_loader::var_builder_from_dir(&root.join("image_encoder"), dtype, &device)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let clip = ClipVision::load(ClipVisionConfig::vit_h_14(), vb).map_err(candle_err)?;
        let load_ms = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let tokens = clip
            .encode_image_file(image, &device, dtype)
            .map_err(candle_err)?;
        let encode_ms = t1.elapsed().as_millis();
        Ok(ClipBenchStats {
            hidden: tokens.dims().to_vec(),
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

    fn resolve_tokenizer(&self) -> Option<String> {
        self.resolved_weights_dir().and_then(|root| {
            let p = root.join("tokenizer").join("tokenizer.json");
            p.exists().then(|| p.to_string_lossy().into_owned())
        })
    }

    /// Host / Burn / Luminal: real UniPC sampler on f32 latents.
    /// Velocity is the analytical flow to the origin (x/σ); DiT is Candle-only.
    fn generate_reference(&self, prompt: &str) -> Result<GenerateOutput> {
        if self.definition.family == ModelFamily::Flux2 {
            return self.generate_reference_flux2(prompt);
        }
        if self.definition.workload_types.contains(&crate::sampling::WorkloadType::I2V) {
            return Err(FastVideoError::NotImplemented {
                component: "I2V generate".into(),
                detail: "pack_i2v_channels is implemented; this backend still needs an image latent".into(),
            });
        }
        let tokenizer = self.resolve_tokenizer().ok_or_else(|| {
            FastVideoError::Message(
                "Host/Burn/Luminal generate needs tokenizer.json (pass --weights or use a cached HF snapshot)".into(),
            )
        })?;
        let (ids, len) = tokenize_prompt(&tokenizer, prompt, 512).map_err(candle_err)?;
        if ids.len() <= 1 || ids.iter().enumerate().all(|(i, t)| *t == i as u32) {
            return Err(FastVideoError::Message(
                "tokenizer produced dummy sequential ids".into(),
            ));
        }
        let mut sched = FlowUniPCMultistepScheduler::new(1000, f64::from(self.pipeline.flow_shift));
        let steps = self.sampling.num_inference_steps.max(1) as usize;
        sched.set_timesteps(steps.min(8));
        let device = fastvideo_ops::Device::cpu();
        let start = vec![0.2f32, -0.4, 0.8, 1.5, -1.1];
        let sample = HostBackend::from_f32(&start, &[5], &device)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let mut x = HostBackend::to_f32(&sample).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let vel = vec![0.1f32, -0.2, 0.05, 0.3, -0.15];
        for _ in 0..sched.inference_timesteps().len() {
            x = sched
                .step(&vel, &x)
                .map_err(|e| FastVideoError::Message(e))?;
        }
        std::fs::create_dir_all(&self.output_path)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let out = std::path::Path::new(&self.output_path).join("unipc-latents.json");
        let body = serde_json::json!({
            "backend": self.backend.as_str(),
            "prompt": prompt,
            "token_ids": ids,
            "token_len": len,
            "latents": x,
            "moe_boundary": self.pipeline.boundary_ratio,
            "arch": WanVideoArchConfig::from_preset(self.definition.preset).num_layers,
        });
        std::fs::write(&out, body.to_string()).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let path = out.to_string_lossy().into_owned();
        Ok(GenerateOutput {
            output_path: Some(path.clone()),
            frame_paths: vec![path],
        })
    }

    fn generate_reference_flux2(&self, prompt: &str) -> Result<GenerateOutput> {
        let seq = packed_flux2_seq(self.sampling.height, self.sampling.width);
        let mu = fastvideo_models::flux2::compute_empirical_mu(
            seq,
            self.sampling.num_inference_steps.max(1) as usize,
        );
        let mut sched = fastvideo_models::FlowMatchEulerDiscreteScheduler::new(1000, 1.0);
        sched.set_timesteps_flux2(self.sampling.num_inference_steps.max(1) as usize, Some(mu));
        let start = vec![0.2f32, -0.4, 0.8, 1.5, -1.1];
        let vel = vec![0.1f32, -0.2, 0.05, 0.3, -0.15];
        let mut x = start;
        for _ in 0..sched.inference_timesteps().len() {
            x = sched
                .step_euler(&x, &vel)
                .map_err(|e| FastVideoError::Message(e))?;
        }
        std::fs::create_dir_all(&self.output_path)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let out = std::path::Path::new(&self.output_path).join("flux2-latents.json");
        let body = serde_json::json!({
            "backend": self.backend.as_str(),
            "family": "flux2",
            "preset": self.definition.preset,
            "prompt": prompt,
            "mu": mu,
            "latents": x,
            "text_encoder": fastvideo_models::flux2::Flux2TextKind::from_preset(self.definition.preset).as_str(),
        });
        std::fs::write(&out, body.to_string()).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let path = out.to_string_lossy().into_owned();
        Ok(GenerateOutput {
            output_path: Some(path.clone()),
            frame_paths: vec![path],
        })
    }
}

fn packed_flux2_seq(height: u32, width: u32) -> usize {
    let (h, w) = fastvideo_models::flux2::packed_hw(height as usize, width as usize, 8);
    h * w
}

pub fn resolve_candle_device(spec: &str) -> Result<Device> {
    let spec = spec.trim().to_ascii_lowercase();
    if spec == "cpu" {
        return Ok(Device::Cpu);
    }
    let index = if spec == "cuda" {
        0usize
    } else if let Some(rest) = spec.strip_prefix("cuda:") {
        rest.parse::<usize>()
            .map_err(|_| FastVideoError::Message(format!("bad CUDA index in `{spec}`")))?
    } else {
        return Err(FastVideoError::Message(format!(
            "unknown device `{spec}` (expected cpu, cuda, or cuda:N)"
        )));
    };
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(index).map_err(candle_err)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = index;
        Err(FastVideoError::Message(
            "CUDA requested but this binary was built without `--features cuda`".into(),
        ))
    }
}

pub fn resolve_dtype(dtype: Option<&str>, device: &str) -> Result<DType> {
    match dtype {
        None if device.to_ascii_lowercase().starts_with("cuda") => Ok(DType::BF16),
        None => Ok(DType::F32),
        Some(value) => match value.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(DType::F32),
            "f16" | "fp16" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            other => Err(FastVideoError::Message(format!(
                "unknown dtype `{other}` (expected f32, f16, or bf16)"
            ))),
        },
    }
}

fn candle_err(err: candle_core::Error) -> FastVideoError {
    FastVideoError::Message(err.to_string())
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
    /// Last timed generate (same run that writes `profile.json`). Warm when
    /// `warmup >= 1`. Compare `median_ms` to upstream `median_seconds`, not this.
    pub generate_ms: u128,
    pub load_and_generate_ms: u128,
    pub frames_written: u32,
    /// Discarded generates after load. First of these is the cold generate.
    pub warmup: u32,
    /// Timed generates after warmup.
    pub runs: u32,
    /// First discarded generate after load (cold when `warmup >= 1`). 0 if none.
    pub warmup_ms: u128,
    /// Timed generate durations, in order.
    pub runs_ms: Vec<u128>,
    /// Median of `runs_ms`. Headline number vs upstream `median_seconds`.
    pub median_ms: u128,
    /// Fastest timed generate.
    pub min_ms: u128,
    /// `FASTVIDEO_SDPA` value in effect (`dense` when unset).
    pub sdpa: String,
}

/// Median of millisecond samples. Even length averages the two middle values
/// (same as Python `statistics.median` used by `upstream_bench.py`).
pub fn median_u128(values: &[u128]) -> u128 {
    if values.is_empty() {
        return 0;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        v[n / 2 - 1].saturating_add(v[n / 2]) / 2
    }
}

pub(crate) fn sdpa_backend_label() -> String {
    std::env::var("FASTVIDEO_SDPA")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "dense".into())
}

fn output_from_frames(frames: Vec<String>) -> GenerateOutput {
    GenerateOutput {
        output_path: frames.first().cloned(),
        frame_paths: frames,
    }
}

enum CudarcLoaded {
    Flux2 {
        pipe: fastvideo_cudarc::flux2::Flux2Pipeline,
        cfg: fastvideo_cudarc::flux2::GenerateConfig,
    },
    Wan {
        pipe: fastvideo_cudarc::WanPipeline,
        cfg: fastvideo_cudarc::GenerateConfig,
    },
}

impl CudarcLoaded {
    fn generate(&mut self) -> Result<Vec<String>> {
        match self {
            Self::Flux2 { pipe, cfg } => pipe
                .generate(cfg)
                .map_err(|e| FastVideoError::Message(e.to_string())),
            Self::Wan { pipe, cfg } => pipe
                .generate(cfg)
                .map_err(|e| FastVideoError::Message(e.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClipBenchStats {
    pub hidden: Vec<usize>,
    pub load_ms: u128,
    pub encode_ms: u128,
    pub path: String,
}
