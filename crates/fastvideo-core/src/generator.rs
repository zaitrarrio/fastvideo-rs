use candle_core::{DType, Device};
use fastvideo_loader::load_diffusers_components;
use fastvideo_models::{
    GenerateConfig, Umt5Config, WanPipeline, WanVaeConfig, WanVideoArchConfig,
};

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
            other => Err(FastVideoError::NotImplemented {
                component: format!("{other} backend"),
                detail: "Phase 1 generate is Candle-only; Burn/Luminal/Host still stubbed".into(),
            }),
        }
    }

    fn generate_candle(&self, prompt: &str) -> Result<GenerateOutput> {
        let device = resolve_candle_device(&self.device)?;
        let dtype = resolve_dtype(self.dtype.as_deref(), &self.device)?;
        let is_dmd = matches!(
            self.definition.sampling,
            SamplingAlgorithm::Dmd | SamplingAlgorithm::CausalDmd
        );
        let tokenizer_path = self.weights_path.as_ref().and_then(|root| {
            let p = std::path::Path::new(root).join("tokenizer").join("tokenizer.json");
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
        };
        if self.tiny {
            gen_cfg.guidance_scale = 1.0;
            gen_cfg.is_dmd = true;
            gen_cfg.flow_shift = 8.0;
        }
        let pipe = if self.tiny {
            WanPipeline::tiny_dtype(&device, dtype).map_err(candle_err)?
        } else {
            let root = self.weights_path.as_ref().ok_or_else(|| {
                FastVideoError::Message(
                    "pass --tiny (zero weights, CI) or --weights <diffusers-dir>".into(),
                )
            })?;
            if gen_cfg.tokenizer_path.is_none() {
                return Err(FastVideoError::Message(format!(
                    "missing {root}/tokenizer/tokenizer.json"
                )));
            }
            let (t_vb, v_vb, e_vb) =
                load_diffusers_components(std::path::Path::new(root), dtype, &device)
                    .map_err(|e| FastVideoError::Message(e.to_string()))?;
            WanPipeline::load(
                t_vb,
                v_vb,
                e_vb,
                WanVideoArchConfig::wan_t2v_1_3b(),
                WanVaeConfig::wan_2_1(),
                Umt5Config::xxl(),
                device,
            )
            .map_err(candle_err)?
        };
        let frames = pipe.generate(&gen_cfg).map_err(candle_err)?;
        Ok(GenerateOutput {
            output_path: frames.first().cloned(),
            frame_paths: frames,
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
