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
            ModelFamily::Hunyuan15 => generate_hunyuan15(def, opts),
            ModelFamily::Kandinsky5 => generate_kandinsky5(def, opts),
            ModelFamily::Cosmos => generate_cosmos(def, opts),
            ModelFamily::LongCat => generate_longcat(def, opts),
            ModelFamily::LingBot => generate_lingbot(def, opts),
            ModelFamily::Gen3C => generate_gen3c(def, opts),
            ModelFamily::MatrixGame => generate_matrixgame(def, opts),
            ModelFamily::DreamX => generate_dreamx(def, opts),
            ModelFamily::LingBotWorld => generate_lingbotworld(def, opts),
            ModelFamily::GameCraft => generate_gamecraft(def, opts),
            ModelFamily::HyWorld => generate_hyworld(def, opts),
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

fn generate_hunyuan15(def: &'static FamilyModelDefinition, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::hunyuan15::pipeline::{Hunyuan15Pipeline, Hunyuan15Request};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::hunyuan15::Hunyuan15Preset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "hy15_480p_t2v" => Hunyuan15Preset::T2v480p,
            "hy15_480p_i2v_distilled" => Hunyuan15Preset::I2v480pDistilled,
            "hy15_720p_t2v" => Hunyuan15Preset::T2v720p,
            "hy15_720p_i2v_distilled" => Hunyuan15Preset::I2v720pDistilled,
            "hy15_1080p_sr" => Hunyuan15Preset::Sr1080p,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown Hunyuan15 preset {other}"
                )));
            }
        };
        let mut pipe = Hunyuan15Pipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        // Optional early weight check so missing transformer/ fails loudly.
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = Hunyuan15Request::t2v_480p(opts.prompt, opts.seed);
        request.preset = preset;
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        request.image_path = opts.image_path;
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::Hunyuan15,
            preset: def.preset,
            frame_paths: Vec::new(),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate HunyuanVideo 1.5".into(),
        ))
    }
}

fn generate_kandinsky5(def: &'static FamilyModelDefinition, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::kandinsky5::pipeline::{Kandinsky5Pipeline, Kandinsky5Request};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::kandinsky5::Kandinsky5Preset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
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
        let mut request = Kandinsky5Request::lite_5s(opts.prompt, opts.seed);
        request.preset = preset;
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::Kandinsky5,
            preset: def.preset,
            frame_paths: Vec::new(),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate Kandinsky 5".into(),
        ))
    }
}

fn generate_cosmos(def: &'static FamilyModelDefinition, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::cosmos::pipeline::{CosmosPipeline, CosmosRequest};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::cosmos::CosmosPreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "cosmos2_v2w_2b" => CosmosPreset::V2w2b,
            "cosmos2_v2w_14b" => CosmosPreset::V2w14b,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown Cosmos preset {other}"
                )));
            }
        };
        let mut pipe = CosmosPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = CosmosRequest::v2w_2b(opts.prompt, opts.seed);
        request.preset = preset;
        request.image_path = opts.image_path.clone();
        request.negative_prompt = opts.negative_prompt.clone();
        if let Some(g) = opts.guidance_scale {
            request.guidance_scale = g;
        }
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let mut frame_paths: Vec<String> = Vec::new();
        if opts.output.is_dir() {
            if let Ok(rd) = std::fs::read_dir(&opts.output) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("png") {
                        frame_paths.push(p.display().to_string());
                    }
                }
                frame_paths.sort();
            }
        }
        Ok(AvGenerateOutput {
            family: ModelFamily::Cosmos,
            preset: def.preset,
            frame_paths,
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate Cosmos".into(),
        ))
    }
}

fn generate_gen3c(def: &'static FamilyModelDefinition, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::gen3c::pipeline::{Gen3CPipeline, Gen3CRequest};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::gen3c::Gen3CPreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "gen3c_cosmos_7b" => Gen3CPreset::Cosmos7b,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown GEN3C preset {other}"
                )));
            }
        };
        let mut pipe = Gen3CPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = Gen3CRequest::cosmos_7b(opts.prompt, opts.seed);
        request.preset = preset;
        request.image_path = opts.image_path.clone();
        request.negative_prompt = opts.negative_prompt.clone();
        if let Some(g) = opts.guidance_scale {
            request.guidance_scale = g;
        }
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let mut frame_paths: Vec<String> = Vec::new();
        if opts.output.is_dir() {
            if let Ok(rd) = std::fs::read_dir(&opts.output) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("png") {
                        frame_paths.push(p.display().to_string());
                    }
                }
                frame_paths.sort();
            }
        }
        Ok(AvGenerateOutput {
            family: ModelFamily::Gen3C,
            preset: def.preset,
            frame_paths,
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate GEN3C".into(),
        ))
    }
}

#[cfg(feature = "cuda-cudarc")]
fn collect_png_frames(output: &std::path::Path) -> Vec<String> {
    let mut frame_paths: Vec<String> = Vec::new();
    if output.is_dir() {
        if let Ok(rd) = std::fs::read_dir(output) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.extension().and_then(|e| e.to_str()) == Some("png") {
                    frame_paths.push(p.display().to_string());
                }
            }
            frame_paths.sort();
        }
    }
    frame_paths
}

fn generate_matrixgame(
    def: &'static FamilyModelDefinition,
    opts: AvGenerateOptions,
) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::matrixgame::pipeline::{MatrixGamePipeline, MatrixGameRequest};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::matrixgame::MatrixGamePreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "mg2_base_distilled" => MatrixGamePreset::Mg2BaseDistilled,
            "mg2_gta_distilled" => MatrixGamePreset::Mg2GtaDistilled,
            "mg2_templerun_distilled" => MatrixGamePreset::Mg2TempleRunDistilled,
            "mg2_base" => MatrixGamePreset::Mg2Base,
            "mg3_base_distilled" => MatrixGamePreset::Mg3BaseDistilled,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown Matrix-Game preset {other}"
                )));
            }
        };
        let mut pipe = MatrixGamePipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = MatrixGameRequest::for_preset(preset, opts.prompt, opts.seed);
        request.image_path = opts.image_path.clone();
        if let Some(g) = opts.guidance_scale {
            request.guidance_scale = g;
        }
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::MatrixGame,
            preset: def.preset,
            frame_paths: collect_png_frames(&opts.output),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate Matrix-Game".into(),
        ))
    }
}

fn generate_dreamx(
    def: &'static FamilyModelDefinition,
    opts: AvGenerateOptions,
) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::dreamx::pipeline::{DreamXPipeline, DreamXRequest};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::dreamx::DreamXPreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "dreamx_5b_cam" => DreamXPreset::Cam5b,
            "dreamx_5b_ar" => DreamXPreset::Ar5b,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown DreamX preset {other}"
                )));
            }
        };
        let mut pipe = DreamXPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = DreamXRequest::for_preset(preset, opts.prompt, opts.seed);
        request.image_path = opts.image_path.clone();
        if let Some(g) = opts.guidance_scale {
            request.guidance_scale = g;
        }
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::DreamX,
            preset: def.preset,
            frame_paths: collect_png_frames(&opts.output),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate DreamX".into(),
        ))
    }
}

fn generate_lingbotworld(
    def: &'static FamilyModelDefinition,
    opts: AvGenerateOptions,
) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::lingbotworld::pipeline::{
            LingBotWorldPipeline, LingBotWorldRequest,
        };
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::lingbotworld::LingBotWorldPreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "lingbotworld_base_cam" => LingBotWorldPreset::BaseCam,
            "lingbotworld2_causal_fast" => LingBotWorldPreset::V2CausalFast,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown LingBot-World preset {other}"
                )));
            }
        };
        let mut pipe = LingBotWorldPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = LingBotWorldRequest::for_preset(preset, opts.prompt, opts.seed);
        request.image_path = opts.image_path.clone();
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::LingBotWorld,
            preset: def.preset,
            frame_paths: collect_png_frames(&opts.output),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LingBot-World".into(),
        ))
    }
}

fn generate_gamecraft(
    def: &'static FamilyModelDefinition,
    opts: AvGenerateOptions,
) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::gamecraft::pipeline::{GameCraftPipeline, GameCraftRequest};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::gamecraft::GameCraftPreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "gamecraft_i2v" => GameCraftPreset::I2v,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown GameCraft preset {other}"
                )));
            }
        };
        let mut pipe = GameCraftPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = GameCraftRequest::i2v(opts.prompt, opts.seed);
        request.image_path = opts.image_path.clone();
        if let Some(g) = opts.guidance_scale {
            request.guidance_scale = g;
        }
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::GameCraft,
            preset: def.preset,
            frame_paths: collect_png_frames(&opts.output),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate GameCraft".into(),
        ))
    }
}

fn generate_hyworld(
    def: &'static FamilyModelDefinition,
    opts: AvGenerateOptions,
) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::hyworld::pipeline::{HyWorldPipeline, HyWorldRequest};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::hyworld::HyWorldPreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "hyworld_bidirectional" => HyWorldPreset::Bidirectional,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown HY-World preset {other}"
                )));
            }
        };
        let mut pipe = HyWorldPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = HyWorldRequest::bidirectional(opts.prompt, opts.seed);
        request.image_path = opts.image_path.clone();
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::HyWorld,
            preset: def.preset,
            frame_paths: collect_png_frames(&opts.output),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate HY-World".into(),
        ))
    }
}

fn generate_longcat(def: &'static FamilyModelDefinition, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::longcat::pipeline::{LongCatPipeline, LongCatRequest};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::longcat::LongCatPreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "longcat_t2v_480p" => LongCatPreset::T2v480p,
            "longcat_t2v_720p" => LongCatPreset::T2v720p,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown LongCat preset {other}"
                )));
            }
        };
        let mut pipe = LongCatPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = LongCatRequest::t2v_480p(opts.prompt, opts.seed);
        request.preset = preset;
        request.enable_bsa = preset.enable_bsa();
        request.height = preset.default_height();
        request.width = preset.default_width();
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::LongCat,
            preset: def.preset,
            frame_paths: Vec::new(),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LongCat".into(),
        ))
    }
}

fn generate_lingbot(def: &'static FamilyModelDefinition, opts: AvGenerateOptions) -> Result<AvGenerateOutput> {
    require_cuda(&opts.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use fastvideo_cudarc::lingbot::pipeline::{LingBotPipeline, LingBotRequest};
        use fastvideo_cudarc::wan::device::resolve_device;
        use fastvideo_models::lingbot::LingBotPreset;

        resolve_device(&opts.device).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let weights = weights_root(&opts)?;
        let preset = match def.preset {
            "lingbot_dense_1_3b" => LingBotPreset::Dense13b,
            "lingbot_moe_30b" => LingBotPreset::Moe30b,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown LingBot preset {other}"
                )));
            }
        };
        let mut pipe = LingBotPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = LingBotRequest::dense_1_3b(opts.prompt, opts.seed);
        request.preset = preset;
        if let Some(h) = opts.height {
            request.height = h as usize;
        }
        if let Some(w) = opts.width {
            request.width = w as usize;
        }
        if let Some(f) = opts.num_frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = opts.num_inference_steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &opts.output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(AvGenerateOutput {
            family: ModelFamily::LingBot,
            preset: def.preset,
            frame_paths: Vec::new(),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (def, opts);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LingBot".into(),
        ))
    }
}
