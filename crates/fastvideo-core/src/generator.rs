use crate::backend_kind::BackendKind;
use crate::error::{FastVideoError, Result};
use crate::registry::{resolve_wan, WanModelDefinition};
use crate::sampling::{pipeline_defaults, sampling_from_definition, PipelineDefaults, SamplingParam};

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub backend: BackendKind,
    pub num_gpus: u32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            backend: BackendKind::Candle,
            num_gpus: 1,
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
}

impl VideoGenerator {
    pub fn from_pretrained(model_id: impl Into<String>, opts: LoadOptions) -> Result<Self> {
        let model_id = model_id.into();
        let definition = resolve_wan(&model_id)?;
        let sampling = sampling_from_definition(definition);
        let pipeline = pipeline_defaults(definition);
        Ok(Self {
            model_id,
            definition,
            sampling,
            pipeline,
            backend: opts.backend,
            num_gpus: opts.num_gpus,
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "model={} preset={} sampling={} backend={} {}x{} frames={} steps={} shift={}",
            self.model_id,
            self.definition.preset,
            self.definition.sampling.as_str(),
            self.backend,
            self.sampling.width,
            self.sampling.height,
            self.sampling.num_frames,
            self.sampling.num_inference_steps,
            self.pipeline.flow_shift
        )
    }

    /// Phase 0: registry + sampling only. DiT / VAE / UMT5 land in Phase 1.
    pub fn generate_video(&self, prompt: &str) -> Result<GenerateOutput> {
        let _ = prompt;
        Err(FastVideoError::NotImplemented {
            component: "WanTransformer3D".into(),
            detail: format!(
                "resolved {}; DiT/VAE/UMT5 forward is Phase 1",
                self.summary()
            ),
        })
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub output_path: Option<String>,
}
