//! MiniMax-H3 / FastH3 / Sol-H3: the variant table, the H3-only options and
//! the run (plan B2: one [`VideoFamily`] behind [`super::FAMILIES`]).

use std::path::PathBuf;

use fastvideo_cudarc::hooks::Hooks;

use super::{
    require_cuda, resolve_table, table_variants, AvGenerateOptions, HubVariant, Output, Pipeline,
    Planned, Variant, VideoFamily,
};
use crate::error::{FastVideoError, Result};
use crate::request::Request;
use crate::sampling::WorkloadType;

/// The family name.
pub const NAME: &str = "h3";

/// The H3 family.
pub struct H3;

/// MiniMax-H3 / FastH3 Hub ids.
pub static VARIANTS: &[HubVariant] = &[
    // Before `minimax_h3`: `minimax-h3-turbo` also contains `minimax-h3`.
    HubVariant::new(
        "sol_h3_ref2va",
        &["lightx2v/Minimax-h3-Turbo"],
        &[WorkloadType::Ref2VA],
        &["sol-h3-ref2va", "sol_h3_ref2va", "minimax-h3-turbo"],
    ),
    HubVariant::new(
        "sol_h3",
        &["FastVideo/FastH3-4-step-Preview-v1-LoRA"],
        &[
            WorkloadType::T2AV,
            WorkloadType::I2V,
            WorkloadType::FL2VA,
            WorkloadType::Ref2VA,
        ],
        &["sol-h3", "sol_h3"],
    ),
    HubVariant::new(
        "fasth3_8step",
        &[
            "FastVideo/FastVideo-FastH3-8-Step-V2",
            "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2",
        ],
        &[WorkloadType::T2AV, WorkloadType::FL2VA],
        &["fasth3", "fast-h3"],
    ),
    HubVariant::new(
        "minimax_h3",
        &["MiniMaxAI/MiniMax-H3"],
        &[
            WorkloadType::T2AV,
            WorkloadType::FL2VA,
            WorkloadType::Ref2VA,
        ],
        &["minimax-h3", "minimax_h3"],
    ),
];

/// The options only H3 reads (the CLI's H3 flags).
#[derive(Debug, Clone, Default)]
pub struct H3Options {
    /// FastH3 / Sol-H3 recipe (`8step`, `4step-vsa`, `sol-h3`, ...). `None`:
    /// the preset's own (`fasth3_8step` -> `8step`, `sol_h3` -> `sol-h3`,
    /// `sol_h3_ref2va` -> `sol-h3-ref2va`).
    pub recipe: Option<String>,
    /// Sol-H3 / FastH3 Preview LoRA adapter. `None`: searched beside the
    /// MiniMax-H3 root.
    pub adapter: Option<PathBuf>,
    /// Alternate root for `tokenizer/` + `text_encoder/`.
    pub text_weights: Option<PathBuf>,
    /// DiT block residency: `auto`, `resident` or `streamed` (`None`:
    /// `FASTVIDEO_DIT_OFFLOAD`, else `auto`).
    pub dit_offload: Option<String>,
    /// Tiny autoencoder video decoder `taeh3.safetensors` (file or
    /// directory); `None`: the full video VAE.
    pub tae_weights: Option<PathBuf>,
}

impl VideoFamily for H3 {
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

/// Base `minimax_h3` T2AV has no default recipe; only the RTX cell
/// (`--h3-recipe sol-h3-rtx`, 49 forwards on the base checkpoint) wires it.
fn h3_t2v_unwired(preset: &str, recipe: Option<&str>, conditioned: bool) -> bool {
    preset == "minimax_h3"
        && !conditioned
        && !recipe.is_some_and(fastvideo_models::h3::lora::is_sol_h3_rtx_recipe)
}

fn run(
    variant: &Variant,
    options: &AvGenerateOptions,
    req: &Request,
    hooks: Hooks<'_>,
) -> Result<Output> {
    require_cuda(&options.device)?;
    let o = &options.family.h3;
    let has_refs = !req.references().is_empty();
    let has_fl2va = req.first_image().is_some() || req.last_image().is_some();
    if h3_t2v_unwired(variant.preset, o.recipe.as_deref(), has_refs || has_fl2va) {
        return Err(FastVideoError::NotImplemented {
            component: "minimax_h3".into(),
            detail: "base MiniMax-H3 T2AV needs --h3-recipe sol-h3-rtx; or use FastH3 for T2AV, or pass --image/--last-image (FL2VA) / --ref (Ref2VA image/video)".into(),
        });
    }
    if has_refs && has_fl2va {
        return Err(FastVideoError::Message(
            "pass either FL2VA (--image/--last-image) or Ref2VA (--ref), not both".into(),
        ));
    }

    #[cfg(feature = "cuda-cudarc")]
    {
        use super::{dit_offload, msg, output_dir, resolve_device, weights_root, DEFAULT_SEED};
        use fastvideo_cudarc::h3::pipeline::{H3Pipeline, H3PipelineOptions, H3Request};
        use fastvideo_models::h3::reference::{infer_reference_kind, H3ReferenceSpec};

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let seconds = req.timing.seconds.unwrap_or(5) as usize;
        let mut request = H3Request::seconds(
            req.prompt.clone(),
            seconds,
            req.seed.unwrap_or(DEFAULT_SEED),
        )
        .map_err(FastVideoError::Message)?;
        let (height, width, frames) = (req.canvas.height, req.canvas.width, req.timing.frames);
        if let Some(h) = height {
            request.height = h as usize;
        }
        if let Some(w) = width {
            request.width = w as usize;
        }
        if let Some(f) = frames {
            request.num_frames = f as usize;
        }
        if height.is_some() || width.is_some() || frames.is_some() {
            fastvideo_models::h3::config::H3Geometry::checked(
                request.height,
                request.width,
                request.num_frames,
            )
            .map_err(FastVideoError::Message)?;
        }
        request.mp4 = options.save_mp4;
        request.first_image = req.first_image().map(std::path::Path::to_path_buf);
        request.last_image = req.last_image().map(std::path::Path::to_path_buf);
        request.references = req
            .references()
            .iter()
            .map(|path| {
                let kind = infer_reference_kind(path);
                H3ReferenceSpec {
                    path: path.clone(),
                    kind,
                }
            })
            .collect();
        let recipe = o.recipe.clone().or_else(|| match variant.preset {
            "fasth3_8step" => Some("8step".into()),
            "sol_h3" => Some("sol-h3".into()),
            "sol_h3_ref2va" => Some("sol-h3-ref2va".into()),
            _ => None,
        });
        if recipe
            .as_deref()
            .is_some_and(fastvideo_models::h3::lora::is_sol_h3_spark_recipe)
        {
            if height.is_none() {
                request.height = fastvideo_models::h3::sol::SPARK_DRAFT_HEIGHT;
            }
            if width.is_none() {
                request.width = fastvideo_models::h3::sol::SPARK_DRAFT_WIDTH;
            }
            if frames.is_none() {
                request.num_frames = fastvideo_models::h3::sol::SPARK_DRAFT_FRAMES;
            }
        }
        let pipeline_options = H3PipelineOptions {
            recipe,
            text_root: o.text_weights.clone(),
            ref2va: has_refs || variant.preset == "sol_h3_ref2va",
            adapter: o.adapter.clone(),
            dit_offload: dit_offload(o.dit_offload.as_deref())?,
            taeh3: o.tae_weights.clone(),
            ..H3PipelineOptions::default()
        };
        // Load, generate one clip, drop everything (as
        // `fastvideo_cudarc::h3::pipeline::generate`, with the hooks).
        let out_dir = output_dir(variant, options);
        let out = H3Pipeline::load(&weights, pipeline_options)
            .and_then(|pipeline| pipeline.generate_with_hooks(&request, &out_dir, hooks))
            .map_err(msg)?;
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths: out.frame_paths,
            mp4: out.mp4,
            wav: Some(out.wav.to_string_lossy().into_owned()),
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req, hooks, has_refs);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LTX/H3".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::h3_t2v_unwired;

    #[test]
    fn base_h3_t2v_runs_with_the_rtx_recipe() {
        assert!(h3_t2v_unwired("minimax_h3", None, false));
        assert!(h3_t2v_unwired("minimax_h3", Some("sol-h3"), false));
        assert!(!h3_t2v_unwired("minimax_h3", Some("sol-h3-rtx"), false));
        assert!(!h3_t2v_unwired("minimax_h3", Some("sol_h3_rtx"), false));
        assert!(!h3_t2v_unwired("minimax_h3", None, true));
        assert!(!h3_t2v_unwired("sol_h3", None, false));
        assert!(!h3_t2v_unwired("fasth3_8step", None, false));
    }
}
