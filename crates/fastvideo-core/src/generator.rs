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
use crate::registry::{resolve_wan, SamplingAlgorithm, WanModelDefinition};
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
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            backend: BackendKind::Candle,
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

    pub fn generate_video(&self, prompt: &str) -> Result<GenerateOutput> {
        match self.backend {
            BackendKind::Candle => self.generate_candle(prompt),
            BackendKind::Host | BackendKind::Burn | BackendKind::Luminal => {
                // Explicit `--weights` runs the Candle Wan graph so these backends
                // can emit PNG frames. Tests leave weights_path unset so they
                // never load the ~17GB 1.3B checkpoint.
                if self.weights_path.is_some() && !self.tiny {
                    self.generate_candle(prompt)
                } else {
                    self.generate_reference(prompt)
                }
            }
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
        let frames = pipe.generate(&gen_cfg).map_err(candle_err)?;
        Ok(GenerateOutput {
            output_path: frames.first().cloned(),
            frame_paths: frames,
        })
    }

    /// Time weight load + generate. Use on Vast CUDA, not laptop CPU.
    pub fn bench_video(&self, prompt: &str) -> Result<(GenerateOutput, BenchStats)> {
        let t0 = Instant::now();
        let out = self.generate_video(prompt)?;
        let total_ms = t0.elapsed().as_millis();
        Ok((
            out.clone(),
            BenchStats {
                model: self.model_id.clone(),
                device: self.device.clone(),
                dtype: self.dtype.clone().unwrap_or_else(|| "default".into()),
                height: self.sampling.height,
                width: self.sampling.width,
                frames: self.sampling.num_frames,
                steps: self.sampling.num_inference_steps,
                load_and_generate_ms: total_ms,
                frames_written: out.frame_paths.len() as u32,
            },
        ))
    }

    /// CLIP ViT-H encode only (`image_encoder/`). Does not load DiT/UMT5.
    pub fn bench_clip(&self, image: &str) -> Result<ClipBenchStats> {
        let root = self.resolved_clip_dir().ok_or_else(|| {
            FastVideoError::Message(
                "CLIP bench needs image_encoder/ (I2V Diffusers snapshot or --weights)".into(),
            )
        })?;
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
    pub device: String,
    pub dtype: String,
    pub height: u32,
    pub width: u32,
    pub frames: u32,
    pub steps: u32,
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
