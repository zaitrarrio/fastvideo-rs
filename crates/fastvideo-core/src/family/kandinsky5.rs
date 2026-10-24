//! Kandinsky 5.0: the variant table and the run (a research port behind
//! [`VideoFamily`], plan B2).

use fastvideo_cudarc::hooks::Hooks;

use super::{
    require_cuda, resolve_table, table_variants, AvGenerateOptions, HubVariant, Output, Pipeline,
    Planned, Variant, VideoFamily,
};
use crate::error::{FastVideoError, Result};
use crate::request::Request;
use crate::sampling::WorkloadType;

/// The family name.
pub const NAME: &str = "kandinsky5";

/// The Kandinsky 5.0 family.
pub struct Kandinsky5;

/// Kandinsky 5.0 Hub ids (FastVideo `kandinsky5` presets).
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "k5_lite_t2v_5s",
        &["kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"],
        &[WorkloadType::T2V],
        &["kandinsky-5.0-t2v-lite", "k5-lite-t2v"],
    ),
    HubVariant::new(
        "k5_pro_t2v_5s",
        &["kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers"],
        &[WorkloadType::T2V],
        &["kandinsky-5.0-t2v-pro", "k5-pro-t2v"],
    ),
];

impl VideoFamily for Kandinsky5 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn variants(&self) -> Vec<Variant> {
        table_variants(NAME, VARIANTS)
    }

    fn resolve(&self, model_id: &str) -> Option<Variant> {
        resolve_table(NAME, VARIANTS, model_id)
    }

    fn load(&self, variant: Variant, options: &AvGenerateOptions) -> Result<Box<dyn Pipeline>> {
        Ok(Planned::boxed(variant, options, run))
    }
}

fn run(
    variant: &Variant,
    options: &AvGenerateOptions,
    req: &Request,
    _hooks: Hooks<'_>,
) -> Result<Output> {
    require_cuda(&options.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use super::{output_dir, resolve_device, weights_root, DEFAULT_SEED};
        use fastvideo_cudarc::kandinsky5::pipeline::{Kandinsky5Pipeline, Kandinsky5Request};
        use fastvideo_models::kandinsky5::Kandinsky5Preset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "k5_lite_t2v_5s" => Kandinsky5Preset::LiteT2v5s,
            "k5_pro_t2v_5s" => Kandinsky5Preset::ProT2v5s,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown Kandinsky5 preset {other}"
                )));
            }
        };
        let mut pipe = Kandinsky5Pipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request =
            Kandinsky5Request::lite_5s(req.prompt.clone(), req.seed.unwrap_or(DEFAULT_SEED));
        request.preset = preset;
        if let Some(h) = req.canvas.height {
            request.height = h as usize;
        }
        if let Some(w) = req.canvas.width {
            request.width = w as usize;
        }
        if let Some(f) = req.timing.frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = req.sampler.steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths: Vec::new(),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate Kandinsky 5".into(),
        ))
    }
}
