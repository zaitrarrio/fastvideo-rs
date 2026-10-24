//! LTX-2 / 2.3 / 2.5: the variant table, the checkpoint line of each preset,
//! the LTX-only options and the run (plan B2: one [`VideoFamily`] behind
//! [`super::FAMILIES`]).

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
pub const NAME: &str = "ltx2";

/// The LTX-2 family.
pub struct Ltx2;

/// Distilled / base LTX checkpoint line (maps to
/// [`fastvideo_models::ltx2::Ltx2ModelVersion`] at generate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx2Line {
    /// LTX-2.0 distilled (8-step CFG=1).
    Distilled20,
    /// LTX-2.5 distilled stage-1 (ancestral + Gemma4 + BWE).
    Distilled25,
    /// LTX-2.0 base (40-step CFG).
    Base20,
    /// LTX-2.3 distilled (5+2 / 8+3 Euler + Gemma3 + BWE + spatial upscaler).
    Distilled23,
    /// LTX-2.3 base (30-step CFG; STG deferred).
    Base23,
}

/// LTX-2 / 2.3 / 2.5 Hub ids we recognize.
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "ltx2_distilled_20",
        &[
            "FastVideo/LTX2-Distilled-Diffusers",
            "rootonchair/LTX-2-19b-distilled",
        ],
        &[WorkloadType::T2AV],
        &["ltx2-distilled", "ltx-2-19b-distilled"],
    ),
    HubVariant::new(
        "ltx2_distilled_25",
        &["Lightricks/LTX-2.5-Diffusers"],
        &[WorkloadType::T2AV],
        &["ltx-2.5", "ltx2.5"],
    ),
    HubVariant::new(
        "ltx2_base_20",
        &[
            "Lightricks/LTX-2",
            "FastVideo/LTX2-base",
            "FastVideo/LTX2-Diffusers",
        ],
        &[WorkloadType::T2AV],
        &["ltx2-base", "ltx-2-diffusers"],
    ),
    HubVariant::new(
        "ltx2_distilled_23",
        &[
            "FastVideo/LTX-2.3-Distilled-Diffusers",
            "FastVideo/LTX2.3-Distilled-Diffusers",
            "diffusers/LTX-2.3-Distilled-Diffusers",
        ],
        &[WorkloadType::T2AV, WorkloadType::I2V],
        &["ltx-2.3-distilled", "ltx2.3-distilled"],
    ),
    HubVariant::new(
        "ltx2_base_23",
        &[
            "Lightricks/LTX-2.3",
            "FastVideo/LTX2.3-base",
            "FastVideo/LTX2.3-Diffusers",
            "diffusers/LTX-2.3-Diffusers",
        ],
        &[WorkloadType::T2AV, WorkloadType::I2V],
        &["ltx-2.3", "ltx2.3"],
    ),
];

/// The checkpoint line of a preset (`None` for a name that is not an LTX
/// preset).
pub fn line(preset: &str) -> Option<Ltx2Line> {
    match preset {
        "ltx2_distilled_20" => Some(Ltx2Line::Distilled20),
        "ltx2_distilled_25" => Some(Ltx2Line::Distilled25),
        "ltx2_base_20" => Some(Ltx2Line::Base20),
        "ltx2_distilled_23" => Some(Ltx2Line::Distilled23),
        "ltx2_base_23" => Some(Ltx2Line::Base23),
        _ => None,
    }
}

/// The options only LTX reads (the CLI's LTX flags).
#[derive(Debug, Clone, Default)]
pub struct Ltx2Options {
    /// Alternate root for `tokenizer/` + `text_encoder/`.
    pub text_weights: Option<PathBuf>,
    /// Alternate DiT root (default `<weights>/transformer`).
    pub dit: Option<PathBuf>,
    /// Distilled two-stage (2.3 / 2.5).
    pub two_stage: bool,
    /// LTX-2.5 DiffVAE video decode.
    pub diff_vae: bool,
    /// Two-stage refine steps (2 or 3).
    pub refine_steps: Option<u32>,
    /// LTX-2.5 stage-2 Sol route (3 refine forwards, layer 0 dense).
    pub sol_stage2: bool,
    /// Force a dense stage 2 (overrides the 2.5 distilled Sol default).
    pub dense_stage2: bool,
    /// LTX-2.3 stage-2 PISA route (3 refine forwards, layers 0..=1 dense).
    pub pisa_stage2: bool,
    /// LTX-2.3 official HQ: 15-step stage-1 + 3-sigma refine, guidance 3,
    /// 1920×1088 / 241f. Requires the 2.3 base (dev) line.
    pub ltx23_hq: bool,
    /// DiT block residency: `auto`, `resident` or `streamed` (`None`:
    /// `FASTVIDEO_DIT_OFFLOAD`, else `auto`).
    pub dit_offload: Option<String>,
    /// Tiny autoencoder video decoder `taeltx2_3_wide.safetensors` (file or
    /// directory); `None`: the full video VAE.
    pub tae_weights: Option<PathBuf>,
}

impl VideoFamily for Ltx2 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn variants(&self) -> Vec<Variant> {
        table_variants(NAME, VARIANTS)
    }

    fn resolve(&self, model_id: &str) -> Option<Variant> {
        resolve_table(NAME, VARIANTS, model_id)
    }

    fn second_guidance_is_audio(&self) -> bool {
        true
    }

    fn load(&self, variant: Variant, options: &AvGenerateOptions) -> Result<Box<dyn Pipeline>> {
        Ok(Planned::boxed(variant, options, run))
    }
}

fn run(
    variant: &Variant,
    options: &AvGenerateOptions,
    req: &Request,
    hooks: Hooks<'_>,
) -> Result<Output> {
    require_cuda(&options.device)?;
    let line = line(variant.preset).ok_or_else(|| {
        FastVideoError::Message(format!("ltx preset {} missing ltx_line", variant.preset))
    })?;

    #[cfg(feature = "cuda-cudarc")]
    {
        use super::{dit_offload, msg, output_dir, resolve_device, weights_root, DEFAULT_SEED};
        use fastvideo_cudarc::ltx2::pipeline::{
            Ltx2Paths, Ltx2Pipeline, Ltx2Request, PipelineOptions,
        };
        use fastvideo_models::ltx2::{
            ltx2_19b, ltx2_19b_distilled, ltx2_23_22b, ltx2_23_22b_distilled, ltx2_5_22b_distilled,
        };

        let o = &options.family.ltx2;
        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let cfg = match line {
            Ltx2Line::Distilled25 => ltx2_5_22b_distilled(),
            Ltx2Line::Distilled20 => ltx2_19b_distilled(),
            Ltx2Line::Base20 => ltx2_19b(),
            Ltx2Line::Distilled23 => ltx2_23_22b_distilled(),
            Ltx2Line::Base23 => ltx2_23_22b(),
        };
        let paths = Ltx2Paths {
            weights: weights.clone(),
            dit: o.dit.clone().unwrap_or_else(|| weights.join("transformer")),
            text: o.text_weights.clone(),
        };
        let d = &cfg.defaults;
        let is_base = matches!(line, Ltx2Line::Base20 | Ltx2Line::Base23);
        let hq = o.ltx23_hq
            || fastvideo_models::ltx2::hq::requested(
                fastvideo_models::settings::env_var("FASTVIDEO_LTX2_HQ")
                    .ok()
                    .as_deref(),
            );
        if hq && !matches!(line, Ltx2Line::Base23) {
            return Err(FastVideoError::Message(
                "ltx2 official HQ is the 2.3 base 15-step two-stage contract (Lightricks/LTX-2.3)"
                    .into(),
            ));
        }
        // The stage-2 route: the flags, else the technique profile, else the
        // recipe default (the same default_sol_stage2 inputs as before).
        let stage2 = fastvideo_models::ltx2::techniques::Ltx2Techniques::from_process(
            fastvideo_models::ltx2::techniques::Stage2Flags {
                sol: o.sol_stage2,
                dense: o.dense_stage2,
                pisa: o.pisa_stage2,
            },
            fastvideo_cudarc::ltx2::pipeline::default_sol_stage2(
                &cfg,
                o.two_stage,
                o.refine_steps
                    .map(|n| n as usize)
                    .or(if hq { Some(3) } else { None }),
                false,
                false,
            ),
        )
        .map_err(FastVideoError::Message)?;
        let request = Ltx2Request {
            prompt: req.prompt.clone(),
            height: req.canvas.height.map(|h| h as usize).unwrap_or(if hq {
                fastvideo_models::ltx2::hq::HEIGHT
            } else {
                d.height
            }),
            width: req.canvas.width.map(|w| w as usize).unwrap_or(if hq {
                fastvideo_models::ltx2::hq::WIDTH
            } else {
                d.width
            }),
            num_frames: req.timing.frames.map(|f| f as usize).unwrap_or(if hq {
                fastvideo_models::ltx2::hq::FRAMES
            } else {
                d.num_frames
            }),
            frame_rate: d.frame_rate,
            seed: req.seed.unwrap_or(DEFAULT_SEED),
            output_dir: output_dir(variant, options),
            mp4: options.save_mp4,
            two_stage: o.two_stage || hq,
            diff_vae: o.diff_vae,
            negative_prompt: req.negative.clone().unwrap_or_default(),
            guidance_scale: req.sampler.guidance.unwrap_or(if hq {
                fastvideo_models::ltx2::hq::GUIDANCE_SCALE
            } else if is_base {
                4.0
            } else {
                1.0
            }),
            audio_guidance_scale: req.sampler.audio_guidance.unwrap_or(if is_base {
                7.0
            } else {
                1.0
            }),
            num_inference_steps: req.sampler.steps.map(|n| n as usize).or(if hq {
                Some(fastvideo_models::ltx2::hq::STAGE1_STEPS)
            } else {
                None
            }),
            refine_steps: o
                .refine_steps
                .map(|n| n as usize)
                .or(if hq { Some(3) } else { None }),
            sol_stage2: stage2.sol_stage2(),
            pisa_stage2: stage2.pisa_stage2(),
            image_path: req.first_image().map(std::path::Path::to_path_buf),
            skip_audio_decode: false,
            reference: None,
            audio: None,
            images: Vec::new(),
            edit: None,
        };
        request.validate().map_err(msg)?;
        let pipeline_options = PipelineOptions {
            dit_offload: dit_offload(o.dit_offload.as_deref())?,
            tae: o.tae_weights.clone(),
            ..PipelineOptions::default()
        };
        let mut pipeline = Ltx2Pipeline::load(&paths, &cfg, &pipeline_options).map_err(msg)?;
        let out = pipeline
            .generate_with_hooks(&request, true, None, hooks)
            .map_err(msg)?;
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths: out.frames,
            mp4: out.mp4,
            wav: Some(out.wav),
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req, hooks, line);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LTX/H3".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_has_a_line() {
        for v in VARIANTS {
            assert!(line(v.preset).is_some(), "{}", v.preset);
        }
        assert_eq!(line("ltx2_distilled_25"), Some(Ltx2Line::Distilled25));
        assert_eq!(line("sol_h3"), None);
    }
}
