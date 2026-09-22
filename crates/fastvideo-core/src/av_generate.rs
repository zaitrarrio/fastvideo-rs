//! CLI-facing generate for LTX-2 / FastH3 (Wan stays on [`crate::generator::VideoGenerator`]).

use std::path::PathBuf;

use crate::error::{FastVideoError, Result};
use crate::registry::{FamilyModelDefinition, ModelFamily, ResolvedModel};
#[cfg(feature = "cuda-cudarc")]
use crate::registry::Ltx2Line;

/// Extra knobs for audio-video families (ignored by Wan).
#[derive(Debug, Clone, Default)]
pub struct AvGenerateOptions {
    pub weights: Option<PathBuf>,
    pub text_weights: Option<PathBuf>,
    pub dit: Option<PathBuf>,
    pub output: PathBuf,
    pub prompt: String,
    pub seed: u64,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub num_frames: Option<u32>,
    pub device: String,
    pub save_mp4: bool,
    /// LTX distilled two-stage (2.3 / 2.5).
    pub two_stage: bool,
    /// LTX-2.5 DiffVAE video decode.
    pub diff_vae: bool,
    pub negative_prompt: String,
    pub guidance_scale: Option<f32>,
    pub audio_guidance_scale: Option<f32>,
    pub num_inference_steps: Option<u32>,
    /// LTX two-stage refine steps (2 or 3).
    pub refine_steps: Option<u32>,
    /// First-frame image for LTX I2V / H3 FL2VA (H3 encodes on GPU via cudarc).
    pub image_path: Option<PathBuf>,
    /// H3 FL2VA last-frame image.
    pub last_image_path: Option<PathBuf>,
    /// Ordered H3 Ref2VA image reference paths (video/audio refs later).
    pub reference_images: Vec<PathBuf>,
    /// FastH3 recipe name (`8step`, `4step-vsa`, …).
    pub h3_recipe: Option<String>,
    /// H3 clip length in whole seconds (5..=15); overrides default geometry.
    pub h3_seconds: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct AvGenerateOutput {
    pub family: ModelFamily,
    pub preset: &'static str,
    pub frame_paths: Vec<String>,
    pub mp4: Option<String>,
    pub wav: Option<String>,
}

/// Run LTX or H3 generate. Requires `--features cuda-cudarc` and a CUDA device.
pub fn generate_av(resolved: ResolvedModel, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    match resolved {
        ResolvedModel::Wan(_) => Err(FastVideoError::Message(
            "generate_av is for LTX/H3; use VideoGenerator for Wan".into(),
        )),
        ResolvedModel::Family(def) => match def.family {
            ModelFamily::Ltx2 => generate_ltx2(def, opts),
            ModelFamily::H3 => generate_h3(def, opts),
            ModelFamily::Wan => unreachable!(),
        },
    }
}

fn require_cuda(device: &str) -> Result<()> {
    if !device.to_ascii_lowercase().starts_with("cuda") {
        return Err(FastVideoError::Message(format!(
            "LTX/H3 generate needs --device cuda (got {device}); Mac/CI stay on Wan --tiny or gpucheck"
        )));
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = device;
        return Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LTX/H3".into(),
        ));
    }
    #[cfg(feature = "cuda-cudarc")]
    Ok(())
}

#[cfg(feature = "cuda-cudarc")]
fn weights_root(opts: &AvGenerateOptions) -> Result<PathBuf> {
    opts.weights
        .clone()
        .or_else(|| std::env::var_os("FASTVIDEO_WEIGHTS").map(PathBuf::from))
        .ok_or_else(|| {
            FastVideoError::Message(
                "LTX/H3 generate needs --weights <diffusers root> or FASTVIDEO_WEIGHTS".into(),
            )
        })
}

fn generate_ltx2(def: &'static FamilyModelDefinition, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    let line = def.ltx_line.ok_or_else(|| {
        FastVideoError::Message(format!("ltx preset {} missing ltx_line", def.preset))
    })?;

    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::ltx2::pipeline::{Ltx2Paths, Ltx2Pipeline, Ltx2Request, PipelineOptions};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::ltx2::{
            ltx2_19b, ltx2_19b_distilled, ltx2_23_22b, ltx2_23_22b_distilled, ltx2_5_22b_distilled,
        };

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let cfg = match line {
            Ltx2Line::Distilled25 => ltx2_5_22b_distilled(),
            Ltx2Line::Distilled20 => ltx2_19b_distilled(),
            Ltx2Line::Base20 => ltx2_19b(),
            Ltx2Line::Distilled23 => ltx2_23_22b_distilled(),
            Ltx2Line::Base23 => ltx2_23_22b(),
        };
        let paths = Ltx2Paths {
            weights: weights.clone(),
            dit: opts.dit.unwrap_or_else(|| weights.join("transformer")),
            text: opts.text_weights,
        };
        let d = &cfg.defaults;
        let is_base = matches!(line, Ltx2Line::Base20 | Ltx2Line::Base23);
        let request = Ltx2Request {
            prompt: opts.prompt,
            height: opts.height.map(|h| h as usize).unwrap_or(d.height),
            width: opts.width.map(|w| w as usize).unwrap_or(d.width),
            num_frames: opts.num_frames.map(|f| f as usize).unwrap_or(d.num_frames),
            frame_rate: d.frame_rate,
            seed: opts.seed,
            output_dir: opts.output,
            mp4: opts.save_mp4,
            two_stage: opts.two_stage,
            diff_vae: opts.diff_vae,
            negative_prompt: opts.negative_prompt,
            guidance_scale: opts.guidance_scale.unwrap_or(if is_base { 4.0 } else { 1.0 }),
            audio_guidance_scale: opts.audio_guidance_scale.unwrap_or(if is_base { 7.0 } else { 1.0 }),
            num_inference_steps: opts.num_inference_steps.map(|n| n as usize),
            refine_steps: opts.refine_steps.map(|n| n as usize),
            image_path: opts.image_path,
        };
        request
            .validate()
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let mut pipeline = Ltx2Pipeline::load(&paths, &cfg, &PipelineOptions::default())
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let out = pipeline
            .generate(&request, true, None)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::Ltx2,
            preset: def.preset,
            frame_paths: out.frames,
            mp4: out.mp4,
            wav: Some(out.wav),
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts, line);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LTX/H3".into(),
        ))
    }
}

fn generate_h3(def: &'static FamilyModelDefinition, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    let has_refs = !opts.reference_images.is_empty();
    let has_fl2va = opts.image_path.is_some() || opts.last_image_path.is_some();
    if def.preset == "minimax_h3" && !has_refs && !has_fl2va {
        return Err(FastVideoError::NotImplemented {
            component: "minimax_h3".into(),
            detail: "base MiniMax-H3 T2AV schedule/AdaLN not wired; use FastH3 for T2AV, or pass --image/--last-image (FL2VA) / --ref (Ref2VA image/video)".into(),
        });
    }
    if has_refs && has_fl2va {
        return Err(FastVideoError::Message(
            "pass either FL2VA (--image/--last-image) or Ref2VA (--ref), not both".into(),
        ));
    }

    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::h3::pipeline::{generate, H3PipelineOptions, H3Request};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::h3::reference::{infer_reference_kind, H3ReferenceSpec};

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let seconds = opts.h3_seconds.unwrap_or(5) as usize;
        let mut request = H3Request::seconds(opts.prompt, seconds, opts.seed)
            .map_err(FastVideoError::Message)?;
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        request.mp4 = opts.save_mp4;
        request.first_image = opts.image_path;
        request.last_image = opts.last_image_path;
        request.references = opts
            .reference_images
            .into_iter()
            .map(|path| {
                let kind = infer_reference_kind(&path);
                H3ReferenceSpec { path, kind }
            })
            .collect();
        let recipe = opts.h3_recipe.or_else(|| match def.preset {
            "fasth3_8step" => Some("8step".into()),
            _ => None,
        });
        let options = H3PipelineOptions {
            recipe,
            text_root: opts.text_weights,
            ref2va: has_refs,
            ..H3PipelineOptions::default()
        };
        let out = generate(&weights, options, &request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::H3,
            preset: def.preset,
            frame_paths: out.frame_paths,
            mp4: out.mp4,
            wav: Some(out.wav.to_string_lossy().into_owned()),
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LTX/H3".into(),
        ))
    }
}
