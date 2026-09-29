//! The CUDA backend's model catalog: every servable configuration, its
//! engine recipe, its tier, and the `ModelCaps` derived from the model
//! configs (WP-11; design §0.3, §0.5, §0.6, §3.2).
//!
//! This module is plain data and is built without the `cuda` feature, so
//! the capability table is unit-tested on CPU. The loaders in
//! [`super::h3`], [`super::ltx2`], [`super::wan`] and [`super::causal`] turn a
//! [`CudaRecipe`] into a resident pipeline.
//!
//! # Tiers (one row per public alias)
//!
//! | Alias | Model id | Recipe | Profile | Why this tier |
//! |---|---|---|---|---|
//! | `h3-max` | `sol-h3` | Sol-H3 4-step (`sol-h3`), Sol engine route tau 1.0/1.25/1.5, official VAE | `h3/sol_h3_4step_engine_ladder` (§0.5) | Owner decision: the tau-ladder route (quality gate PASS, 1.53x denoise vs dense) |
//! | `h3-turbo` | `fasth3-4step-vsa` | FastH3 Preview 4-step (`4step-vsa`), VSA-H3, MXFP8 linears, official VAE, 768p | `h3/fasth3_4step_vsa` | Fastest H3 recipe that passes the gate |
//! | `h3-draft` | `fasth3-4step-vsa-480p-taeh3` | the turbo recipe at 480p with the TAEH3 decoder | `h3/fasth3_4step_vsa` | 8.1 s on RTX PRO 6000; fails the gate (draft) |
//! | — | `fasth3-8step-dense` | FastH3 8-step DMD (`8step`), dense attention, official VAE | none | untiered (explicit id or `recipe = "8step"`) |
//! | `ltx-pro` | `ltx25-distill-dense` | LTX-2.5 22B distilled two-stage (8 + 3), dense stage 2, conv VAE | `ltx2/ltx25_distill_dense` | Only LTX-2.5 generation path; dense (non-lossy) stage 2, full VAE |
//! | `ltx-turbo` | `ltx25-distill-sol` | LTX-2.5 distilled two-stage, Sol stage 2 | `ltx2/ltx25_distill_sol` | The reference single-GPU route; passes the gate |
//! | `ltx-draft` | `ltx25-distill-sol-nvfp4-taehv` | + NVFP4 video FFN + TAEHV (`taeltx2_3_wide`) decode | `ltx2/ltx25_distill_sol_nvfp4` | Faster, fails sharpness at 4K (draft) |
//! | (`ltx-pro` Ref2V) | `ltx25-ref2v` | the `ltx-pro` recipe plus the Ingredients IC-LoRA fused at stage 1 (`ICLoraPipeline`): reference-to-video only, one reference sheet, 1536x896 default | `ltx2/ltx25_distill_dense` | The LTX reference mode (docs/ports/ltx-ref2v.md); `route_task` sends `ltx-pro` Ref2V requests here |
//! | (`ltx-pro` A2V) | `ltx25-a2v-guided` | `A2VidPipelineTwoStage`: the LTX-2.5 dev DiT (`ltx25-dev/transformer_full`), multimodal guider at stage 1 (30 steps, CFG = `guidance_scale`, default 3), the distilled LoRA fused at stage 2, dense; audio-to-video only | `ltx2/ltx25_distill_dense` | Upstream's own audio-to-video; `route_task` sends `ltx-pro` A2V requests here (`ltx25-distill-dense` serves no A2V; `ltx-turbo` keeps the distilled A2V) |
//! | `wan-max` | `wan22-ti2v-5b` | Wan2.2 TI2V-5B, 50 UniPC steps, CFG 5, shift 5, full Wan 2.2 VAE, 704x1280x121 @ 24 fps (short edges 704, 576, 480; up to 161 frames) | none | The checkpoint's recommended recipe |
//! | `wan-turbo` | `fastwan22-ti2v-5b` | FastWan2.2 TI2V-5B (FullAttn) DMD 3-step (1000/757/522), shift 5, full Wan 2.2 VAE, 704x1280x121 @ 24 fps, same canvas and frames as `wan-max` | none | fal's `fal-ai/wan/v2.2-5b/text-to-video/fast-wan` model (docs/serve/fal-parity.md §3) |
//! | `wan-draft` | `fastwan22-ti2v-5b-taehv` | the turbo recipe decoded by TAEHV (`taew2_2`) | none | Tiny decoder: faster, lossy (draft) |
//! | — | `fastwan21-1.3b` | FastWan2.1 1.3B DMD 3-step (1000/757/522), VSA, full Wan VAE, 480x832x81 @ 16 fps | none (`FASTVIDEO_VSA=1`) | untiered (was `wan-turbo`): the published FastWan 1.3B recipe |
//! | — | `fastwan21-1.3b-taehv` | the 1.3B recipe decoded by TAEHV (`taew2_1`) | none (`FASTVIDEO_VSA=1`) | untiered (was `wan-draft`) |
//! | — | `sfwan21-1.3b` | SF-Wan 1.3B causal, 4 Self-Forcing steps, shift 5, TAEHV per block | none | Causal streaming (`StreamCaps::Causal`) |
//!
//! # Process settings
//!
//! Technique profiles and a few `FASTVIDEO_*` flags are **process-wide** and
//! read once (`fastvideo_models::techniques::settings`): linear precision,
//! activation precision, NVFP4, Wan VSA. [`ProcessPlan::for_models`] checks
//! that every model one process serves agrees on them and names the profile
//! to install; per-request techniques (attention route, TeaCache) may still
//! differ per model (H3 `set_arm`, LTX stage-2 route per request). Models
//! that disagree need their own process (and they do not co-reside in one
//! GPU's memory anyway: design risk R18).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use fastvideo_models::h3::config as h3cfg;
use fastvideo_protocol::{
    AudioCaps, CanvasCaps, Family, FpsCaps, FrameGrid, KnobCaps, ModelCaps, ModelId, RefLimits,
    StreamCaps, Task, Tier,
};
use serde::{Deserialize, Serialize};

use crate::caps::{Recipe, SOL_H3_4STEP_PROFILE};

/// FastVideo's English Wan negative prompt (`fv-gpucheck wan gen` default).
pub const WAN_NEGATIVE_EN: &str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards";

/// FastVideo's Chinese Wan 2.2 negative prompt (the TI2V-5B preset's).
pub const WAN_NEGATIVE_CN: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

/// Where the weights live. Defaults follow the Runpod weight volume
/// (`/workspace/weights/<cell>`, scripts/gpu/weights-manifest.tsv).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightLayout {
    pub root: PathBuf,
    /// Tiny autoencoders: `taeh3.safetensors`, `taeltx2_3_wide.safetensors`,
    /// `taew2_1.safetensors` (scripts/gpu/fetch-tae.sh). Default `<root>/auxiliary/tae`.
    pub tae_dir: PathBuf,
}

impl WeightLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            tae_dir: root.join("auxiliary").join("tae"),
            root,
        }
    }

    pub fn with_tae_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.tae_dir = dir.into();
        self
    }

    fn at(&self, cell: &str) -> PathBuf {
        self.root.join(cell)
    }
}

impl Default for WeightLayout {
    fn default() -> Self {
        Self::new("/workspace/weights")
    }
}

/// An H3 configuration (`H3Pipeline::load` options).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct H3Recipe {
    /// Contract name (`8step`, `4step-vsa`, `4step-dense`, `sol-h3`).
    pub recipe: String,
    pub weights: PathBuf,
    /// Dense attention without the compression gate.
    pub dense: bool,
    /// Technique profile (builtin name, e.g. `h3/fasth3_4step_vsa`).
    pub profile: Option<String>,
    /// TAEH3 decoder instead of the official VAE.
    pub taeh3: Option<PathBuf>,
    /// Default short edge (768 or 480).
    pub short_edge: u32,
    /// `auto`, `streamed`, `resident-fp8`, `resident-bf16`.
    pub text_encoder: String,
    /// Root of `tokenizer/` + `text_encoder/` when not `weights`.
    pub text_weights: Option<PathBuf>,
    /// Memoized AdaLN table.
    pub adaln_cache: Option<PathBuf>,
    /// `auto` | `resident` | `streamed` (None: env / auto).
    pub dit_offload: Option<String>,
    pub steps: u32,
    /// Ref2VA: the model loads `transformer_ref/` and serves reference-to-
    /// video only (a Ref2VA pipeline refuses requests without references).
    #[serde(default)]
    pub ref2va: bool,
    /// Root of `transformer_ref/` and the Ref2VA turbo adapter when not
    /// `weights` (the `h3-ref2va` tree beside `h3-base`).
    #[serde(default)]
    pub ref_weights: Option<PathBuf>,
    /// Where the image-to-video multimodal text encoder lives: `auto`
    /// (resident when the text encoder is and the vision tower fits),
    /// `resident`, or `stream` (read from the volume per request).
    #[serde(default = "auto_str")]
    pub i2v_encoder: String,
    /// Run one text-to-video and one image-to-video generation at the default
    /// canvas after the load, before the model is reported ready.
    #[serde(default)]
    pub warmup: bool,
    /// The opt-in native 1080P tier (short edge 1080, generated at
    /// 1920x1088 and cropped; docs/serve/h3-1080p-and-upscaler.md). Offered
    /// only when the GPU passes the tier's memory plan ([`gate_h3_1080p`]).
    /// Catalog: h3-max and h3-turbo; `h3_1080p = false` turns it off.
    #[serde(default)]
    pub hd_1080p: bool,
}

fn auto_str() -> String {
    "auto".to_owned()
}

/// LTX-2 checkpoint line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LtxVersion {
    #[serde(rename = "2.3")]
    V23,
    #[serde(rename = "2.5")]
    V25,
}

/// LTX stage-2 attention route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LtxStage2 {
    Dense,
    Sol,
    Pisa,
}

/// An LTX-2 configuration (`Ltx2Pipeline::load` + per-request flags).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ltx2Recipe {
    pub version: LtxVersion,
    /// Diffusers snapshot (tokenizer, text encoder, VAEs, vocoder, upsampler).
    pub weights: PathBuf,
    /// The distilled DiT + connectors (file or root).
    pub dit: PathBuf,
    pub text_weights: Option<PathBuf>,
    pub two_stage: bool,
    pub stage2: LtxStage2,
    pub profile: Option<String>,
    /// `taeltx2_3_wide.safetensors` decoder instead of the conv VAE.
    pub tae: Option<PathBuf>,
    /// `auto` | `resident` | `streamed`.
    pub text: String,
    /// Default canvas `(width, height)`.
    pub canvas: (u32, u32),
    pub stage1_steps: u32,
    pub refine_steps: u32,
    /// Reference-to-video: the IC-LoRA (the Ingredients reference-sheet LoRA)
    /// fused into stage 1 (`PipelineOptions::ic_lora`). The model then serves
    /// `Task::Ref2V` only, with a dense stage 2 (`ICLoraPipeline`).
    #[serde(default)]
    pub ic_lora: Option<PathBuf>,
    /// Audio-to-video on this model ([`LtxA2v`]).
    #[serde(default)]
    pub a2v: LtxA2v,
}

/// How an LTX-2.5 model serves audio-to-video.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LtxA2v {
    /// The driving audio pinned as clean latents on the distilled stages
    /// (unguided; the fast tier).
    #[default]
    Distilled,
    /// Not served: the tier's guided companion takes it (`ltx-pro`).
    Off,
    /// `A2VidPipelineTwoStage` proper, the model serving audio-to-video only:
    /// `dit` is the LTX-2.5 *dev* transformer (`ltx25-dev/transformer_full`),
    /// stage 1 runs the multimodal guider (CFG / STG / modality / rescale,
    /// `stage1_steps` on the `LTX2Scheduler` schedule), stage 2 the distilled
    /// LoRA fused at 1 on the same DiT.
    Guided,
}

/// The dev transformer of the guided audio-to-video, under the weight root.
pub const LTX25_DEV_DIT: &str = "ltx25-dev/transformer_full";

/// Wan sampler.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WanSampler {
    /// DMD Euler (FastWan): `steps` timesteps, guidance 1.
    Dmd { steps: u32 },
    /// UniPC with CFG.
    Unipc { steps: u32, guidance: f32 },
}

impl WanSampler {
    pub fn steps(&self) -> u32 {
        match self {
            WanSampler::Dmd { steps } | WanSampler::Unipc { steps, .. } => *steps,
        }
    }
}

/// Wan decoder choice (`FASTVIDEO_WAN_VAE` at load).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WanDecoder {
    Auto,
    Full,
    Taehv,
}

impl WanDecoder {
    pub fn env_value(self) -> &'static str {
        match self {
            WanDecoder::Auto => "auto",
            WanDecoder::Full => "full",
            WanDecoder::Taehv => "taehv",
        }
    }
}

/// A Wan configuration (`WanPipeline::load_with` + `GenerateConfig`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WanRecipe {
    /// Registry preset (`fast_wan_t2v_480p`, `wan_2_2_ti2v_5b`, `sf_wan_t2v_1_3b`).
    pub preset: String,
    pub weights: PathBuf,
    pub sampler: WanSampler,
    pub flow_shift: f64,
    /// Video sparse attention (`FASTVIDEO_VSA=1`, process-wide).
    pub vsa: bool,
    pub decoder: WanDecoder,
    /// Default `(width, height, frames, fps)`.
    pub default: (u32, u32, u32, u32),
    /// Largest canvas area.
    pub max_area: u64,
    pub short_edges: Vec<u32>,
    pub multiple: u32,
    pub frames_max: u32,
    pub i2v: bool,
    pub negative: String,
    /// Extra dirs holding `taew2_*.safetensors` (`FASTVIDEO_TAE_DIR`).
    pub tae_dir: Option<PathBuf>,
}

/// SF-Wan causal streaming configuration (`wan::stream::RolloutConfig`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SfWanRecipe {
    pub wan: WanRecipe,
    /// KV window in latent frames (FastVideo `local_attn_size`).
    pub local_attn_frames: u32,
    /// Frames kept at the head of the rolling cache.
    pub sink_frames: u32,
    /// Pixel frames per block after the first (3 latent frames).
    pub block_frames: u32,
}

/// What a model runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum CudaRecipe {
    H3(H3Recipe),
    Ltx2(Ltx2Recipe),
    Wan(WanRecipe),
    SfWan(SfWanRecipe),
}

impl CudaRecipe {
    pub fn family(&self) -> Family {
        match self {
            CudaRecipe::H3(_) => Family::H3,
            CudaRecipe::Ltx2(_) => Family::Ltx2,
            CudaRecipe::Wan(_) | CudaRecipe::SfWan(_) => Family::Wan,
        }
    }

    /// The technique profile installed for this recipe.
    pub fn profile(&self) -> Option<&str> {
        match self {
            CudaRecipe::H3(r) => r.profile.as_deref(),
            CudaRecipe::Ltx2(r) => r.profile.as_deref(),
            CudaRecipe::Wan(_) | CudaRecipe::SfWan(_) => None,
        }
    }

    /// Process-wide `FASTVIDEO_*` variables this recipe needs.
    pub fn process_env(&self) -> Vec<(&'static str, &'static str)> {
        match self {
            CudaRecipe::Wan(w) => vec![("FASTVIDEO_VSA", if w.vsa { "1" } else { "0" })],
            CudaRecipe::SfWan(s) => vec![("FASTVIDEO_VSA", if s.wan.vsa { "1" } else { "0" })],
            _ => Vec::new(),
        }
    }
}

/// One servable model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaModel {
    pub id: ModelId,
    /// Extra names (`fasth3`, FastVideo `/v1/models` ids).
    pub served_names: Vec<String>,
    pub tier: Option<Tier>,
    /// Loaded before readiness (the warm pool); otherwise on demand in swap mode.
    pub resident: bool,
    pub recipe: CudaRecipe,
    /// Short recipe name (`ModelCaps::recipe`, `Recipe::name`).
    pub recipe_name: String,
}

impl CudaModel {
    fn new(id: &str, tier: Option<Tier>, recipe_name: &str, recipe: CudaRecipe) -> Self {
        Self {
            id: ModelId::new(id),
            served_names: vec![id.to_owned()],
            tier,
            resident: true,
            recipe,
            recipe_name: recipe_name.to_owned(),
        }
    }

    pub fn with_served_names(mut self, names: &[&str]) -> Self {
        for n in names {
            if !self.served_names.iter().any(|s| s == n) {
                self.served_names.push((*n).to_owned());
            }
        }
        self
    }

    pub fn family(&self) -> Family {
        self.recipe.family()
    }

    /// The model's weight root (checked before a load).
    pub fn weights(&self) -> &Path {
        match &self.recipe {
            CudaRecipe::H3(r) => &r.weights,
            CudaRecipe::Ltx2(r) => &r.weights,
            CudaRecipe::Wan(r) => &r.weights,
            CudaRecipe::SfWan(r) => &r.wan.weights,
        }
    }

    /// What this model needs from the GPU (the startup capability check,
    /// [`crate::device`]): FP8 tensor cores for the H3 recipes with a
    /// technique profile (MXFP8 linears; tensorwise FP8 off Blackwell) or
    /// an FP8-resident text encoder; NVFP4 for the LTX draft profile; the
    /// DiT's weight bytes (when the weights are on disk) within the device.
    pub fn requirements(&self) -> crate::device::Requirements {
        let (fp8, nvfp4) = match &self.recipe {
            CudaRecipe::H3(r) => (r.profile.is_some() || r.text_encoder == "resident-fp8", false),
            CudaRecipe::Ltx2(r) => {
                let fp4 = r.profile.as_deref() == Some(LTX_DRAFT_PROFILE);
                (fp4, fp4)
            }
            CudaRecipe::Wan(_) | CudaRecipe::SfWan(_) => (false, false),
        };
        crate::device::Requirements { fp8, nvfp4, min_total_bytes: self.dit_bytes() }
    }

    /// On-disk bytes of the DiT, the dominant resident weights: the
    /// `*.safetensors` under `<weights>/transformer` (LTX: the `dit` file
    /// when it is one). `None` when there is nothing to measure. A bf16
    /// checkpoint over-estimates an MXFP8/FP8 residency, which is the safe
    /// side for the co-residency check.
    pub fn dit_bytes(&self) -> Option<u64> {
        let dir = match &self.recipe {
            CudaRecipe::Ltx2(r) if r.dit.is_file() => return r.dit.metadata().ok().map(|m| m.len()),
            CudaRecipe::Ltx2(r) if r.a2v == LtxA2v::Guided => r.dit.clone(),
            CudaRecipe::Ltx2(r) => r.dit.join("transformer"),
            CudaRecipe::H3(r) if r.ref2va => {
                r.ref_weights.as_deref().unwrap_or(&r.weights).join("transformer_ref")
            }
            _ => self.weights().join("transformer"),
        };
        let total: u64 = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        (total > 0).then_some(total)
    }

    /// `ModelCaps`, derived from the model configs.
    pub fn caps(&self) -> ModelCaps {
        let mut c = match &self.recipe {
            CudaRecipe::H3(r) => h3_caps(self.id.as_str(), r),
            CudaRecipe::Ltx2(r) => ltx2_caps(self.id.as_str(), r),
            CudaRecipe::Wan(r) => wan_caps(self.id.as_str(), r),
            CudaRecipe::SfWan(r) => sfwan_caps(self.id.as_str(), r),
        };
        c.served_names = self.served_names.clone();
        c.resident = self.resident;
        c.recipe = Some(self.recipe_name.clone());
        c.tier = self.tier;
        c
    }

    /// The recipe description served in `/fv/v1/capabilities`.
    pub fn describe(&self) -> Recipe {
        let (steps, attention, vae, summary) = match &self.recipe {
            CudaRecipe::H3(r) => (
                r.steps,
                if r.dense || r.ref2va {
                    "dense".to_owned()
                } else if self.recipe_name.starts_with("sol-h3") {
                    "sol-engine-tau-ladder".to_owned()
                } else {
                    "vsa".to_owned()
                },
                if r.taeh3.is_some() { "taeh3" } else { "full" }.to_owned(),
                match self.tier {
                    _ if r.ref2va && r.recipe == "base" => "H3 reference-to-video, highest quality: \
                        the base Ref2VA DiT (transformer_ref), 49 forwards on the 50-point grid, dense \
                        attention, official VAE (the upstream default)"
                        .to_owned(),
                    _ if r.ref2va => format!(
                        "H3 reference-to-video, fast: the Ref2VA DiT with the lightx2v turbo LoRA fused \
                         (Sol-H3 Ref2VA route), {} forwards, dense attention, official VAE",
                        r.steps
                    ),
                    _ if self.recipe_name.starts_with("fasth3-8step") => "FastH3 8-step DMD, dense \
                                        attention (no compression gate), official VAE"
                        .to_owned(),
                    Some(Tier::Draft) => format!(
                        "H3 draft: FastH3 4-step VSA at {}p with the TAEH3 decoder (fails the quality gate)",
                        r.short_edge
                    ),
                    _ if self.recipe_name.starts_with("sol-h3") => {
                        "H3 highest quality: Sol-H3 4-step, Sol engine route tau 1.0/1.25/1.5 on forwards 1-3 \
                         (1.53x denoise vs dense, quality gate PASS)"
                            .to_owned()
                    }
                    _ => "H3 turbo: FastH3 Preview 4-step, VSA-H3, MXFP8 linears, official VAE".to_owned(),
                },
            ),
            CudaRecipe::Ltx2(r) => (
                r.stage1_steps + if r.two_stage { r.refine_steps } else { 0 },
                match r.stage2 {
                    LtxStage2::Dense => "dense",
                    LtxStage2::Sol => "sol-stage2",
                    LtxStage2::Pisa => "pisa-stage2",
                }
                .to_owned(),
                if r.tae.is_some() { "taehv" } else { "full" }.to_owned(),
                if r.a2v == LtxA2v::Guided {
                    format!(
                        "Audio-to-video, guided (A2VidPipelineTwoStage): the LTX-2.5 dev DiT, stage 1 {} steps \
                         with CFG 3 (guidance_scale), STG on block 28, modality guidance 3 and rescale 0.7, stage 2 \
                         the distilled LoRA, dense, conv VAE; the output carries the input audio",
                        r.stage1_steps
                    )
                } else {
                format!(
                    "{}LTX-{} distilled {}, {} stage 2{}{}",
                    if r.ic_lora.is_some() {
                        "Reference-to-video (Ingredients IC-LoRA at stage 1, one reference sheet): "
                    } else {
                        ""
                    },
                    match r.version {
                        LtxVersion::V23 => "2.3",
                        LtxVersion::V25 => "2.5",
                    },
                    if r.two_stage { "two-stage" } else { "single-stage" },
                    match r.stage2 {
                        LtxStage2::Dense => "dense",
                        LtxStage2::Sol => "Sol",
                        LtxStage2::Pisa => "PISA",
                    },
                    if r.profile.as_deref() == Some(LTX_DRAFT_PROFILE) { ", NVFP4 video FFN" } else { "" },
                    if r.tae.is_some() { ", TAEHV decode (fails the quality gate)" } else { ", conv VAE" },
                )
                },
            ),
            CudaRecipe::Wan(r) => (
                r.sampler.steps(),
                if r.vsa { "vsa" } else { "dense" }.to_owned(),
                match r.decoder {
                    WanDecoder::Taehv => "taehv",
                    WanDecoder::Full => "full",
                    WanDecoder::Auto => "auto",
                }
                .to_owned(),
                match &r.sampler {
                    WanSampler::Dmd { steps } => format!(
                        "{} DMD {steps}-step{}, {} decode",
                        r.preset,
                        if r.vsa { ", VSA" } else { "" },
                        r.decoder.env_value()
                    ),
                    WanSampler::Unipc { steps, guidance } => format!(
                        "{} UniPC {steps} steps, CFG {guidance}, shift {}, {} decode",
                        r.preset,
                        r.flow_shift,
                        r.decoder.env_value()
                    ),
                },
            ),
            CudaRecipe::SfWan(r) => (
                r.wan.sampler.steps(),
                "causal-kv".to_owned(),
                "taehv".to_owned(),
                format!(
                    "SF-Wan causal rollout, {} Self-Forcing steps per block, KV window {} latent frames, sink {}",
                    r.wan.sampler.steps(),
                    r.local_attn_frames,
                    r.sink_frames
                ),
            ),
        };
        Recipe {
            name: self.recipe_name.clone(),
            profile: self.recipe.profile().map(str::to_owned),
            steps: Some(steps),
            attention,
            vae,
            summary,
        }
    }
}

/// Frame ceiling of the Wan 5B recipes (fal's `num_frames` 17..=161).
pub const WAN5B_FRAMES_MAX: u32 = 161;
/// Container frame rates a Wan clip may be muxed at (fal's
/// `frames_per_second` 4..=60): the frames do not depend on it.
pub const WAN_FPS_MIN: u32 = 4;
pub const WAN_FPS_MAX: u32 = 60;

/// The LTX draft profile (NVFP4 video FFN).
pub const LTX_DRAFT_PROFILE: &str = "ltx2/ltx25_distill_sol_nvfp4";

fn h3_caps(id: &str, r: &H3Recipe) -> ModelCaps {
    let mut c = ModelCaps::h3(id, r.ref2va);
    if r.ref2va {
        // `transformer_ref/` serves references only: the pipeline refuses a
        // request without them, and FL2VA keyframes need `transformer/`.
        c.tasks = [Task::Ref2V].into_iter().collect();
    }
    let base = CanvasCaps {
        short_edges: vec![
            h3cfg::H3_SHORT_EDGE as u32,
            h3cfg::H3_SHORT_EDGE_480P as u32,
        ],
        ..CanvasCaps::h3()
    };
    c.canvas = if r.short_edge == h3cfg::H3_SHORT_EDGE_480P as u32 {
        // Draft: the 480 tier only, at the 480 pixel budget.
        CanvasCaps {
            max_area: base.area_at(r.short_edge),
            short_edges: vec![r.short_edge],
            ..base
        }
    } else if r.hd_1080p && !r.ref2va {
        // 768 (default), 480, then the opt-in 1080 tier with its own budget.
        base.with_h3_1080p()
    } else {
        base
    };
    c
}

/// Turns the native 1080P tier off on every H3 model when the GPU has less
/// memory than the tier's plan (`plan_1080p_min_device_bytes`, about 72 GiB:
/// 80 GB and 96 GB cards keep it, 48 GB ones do not). `device_total` is the
/// device's memory (under `FASTVIDEO_DEVICE_BUDGET_GIB`, the budget); `None`
/// (not known) turns it off. Returns what it turned off, for the log.
pub fn gate_h3_1080p(models: &mut [CudaModel], device_total: Option<u64>) -> Option<String> {
    let need = fastvideo_models::h3::memory::plan_1080p_min_device_bytes();
    if device_total.is_some_and(|t| t >= need) {
        return None;
    }
    let mut off = Vec::new();
    for m in models.iter_mut() {
        if let CudaRecipe::H3(r) = &mut m.recipe {
            if r.hd_1080p {
                r.hd_1080p = false;
                off.push(m.id.as_str().to_owned());
            }
        }
    }
    let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
    (!off.is_empty()).then(|| {
        format!(
            "H3 1080P tier off for {}: the device has {} and the tier's memory plan needs {:.1} GiB",
            off.join(", "),
            device_total.map_or("unknown memory".to_owned(), |t| format!("{:.1} GiB", gib(t))),
            gib(need)
        )
    })
}

/// LTX caps from the model config (audio from the vocoder).
/// The longest LTX-2 clip served (frames), for every task.
pub const LTX_MAX_FRAMES: u32 = 481;

fn ltx2_caps(id: &str, r: &Ltx2Recipe) -> ModelCaps {
    let cfg = ltx_config(r.version);
    let fps = cfg.defaults.frame_rate.round() as u32;
    let grid = FrameGrid::new(8, 1, 9, LTX_MAX_FRAMES, 121);
    if r.ic_lora.is_some() {
        return ltx2_ref_caps(id, r, fps);
    }
    if r.a2v == LtxA2v::Guided {
        return ltx2_a2v_guided_caps(id, r);
    }
    ModelCaps {
        id: ModelId::new(id),
        family: Family::Ltx2,
        served_names: vec![id.to_owned()],
        // E5 / E9: first-frame image and last-frame keyframe conditioning on
        // the 2.5 distilled pipeline (`ltx2::i2v_encode`, oracle-checked on 2.5).
        // Audio-to-video: the driving audio pinned as clean audio latents on
        // both stages (`ltx2::a2v`, docs/oracle.md "LTX-2.5 audio-to-video").
        // `a2v = off` (`ltx-pro`): its guided companion serves audio-to-video.
        // Retake / extend: one distilled stage at the source size with the
        // kept tokens pinned (`ltx2::v2v`, docs/oracle.md "LTX-2.5 retake and
        // extend"); the guided companion (`ltx2_a2v_guided_caps`) does not.
        tasks: match (r.version, r.a2v) {
            (LtxVersion::V25, LtxA2v::Off) => {
                [Task::T2V, Task::I2V, Task::Keyframes, Task::Retake, Task::Extend].into_iter().collect()
            }
            (LtxVersion::V25, _) => {
                [Task::T2V, Task::I2V, Task::Keyframes, Task::A2V, Task::Retake, Task::Extend].into_iter().collect()
            }
            (LtxVersion::V23, _) => [Task::T2V].into_iter().collect(),
        },
        audio: Some(AudioCaps {
            native_rate: cfg.vocoder.output_sampling_rate as u32,
            channels: cfg.vocoder.out_channels as u8,
            via_sidecar: false,
        }),
        // E4: 24 (default), 25, 48, 50 validated at 1080p.
        fps: crate::caps::ltx_fps_caps(),
        stream: Some(StreamCaps::Clip {
            min_s: grid.min as f32 / fps as f32,
            max_s: grid.max as f32 / fps as f32,
        }),
        frames: grid,
        canvas: CanvasCaps {
            multiple: if r.two_stage { 64 } else { 32 },
            max_area: 3840 * 2176,
            aspect: (0.25, 4.0),
            // The LTX API tiers; the first is the default (1080 is generated
            // at 1088 and cropped back, ltx §3.1).
            short_edges: vec![1080, 720, 1440, 2160],
            pad_and_crop: true,
            hd: None,
        },
        refs: RefLimits::none(),
        // Distilled LTX-2.5 is unguided and the stage steps are fixed by the
        // two-stage contract: only the seed is a per-request knob.
        knobs: KnobCaps {
            seed: true,
            ..KnobCaps::default()
        },
        resident: true,
        tier: None,
        recipe: None,
    }
}

/// Stage-1 bucket of the Ingredients IC-LoRA (model card: trained at
/// 768x448, 121 frames, 24 fps); the two-stage output is twice that.
pub const LTX_REF_CANVAS: (u32, u32) = (1536, 896);
/// Frame ceiling of LTX reference-to-video: the reference doubles stage 1's
/// sequence (and the IC-LoRA keeps an unfused copy of the weights it
/// touches), so the clip stays at 10 s (the LoRA was trained on 121 frames).
pub const LTX_REF_FRAMES_MAX: u32 = 241;

/// LTX-2.5 reference-to-video (the IC-LoRA companion of `ltx-pro`).
fn ltx2_ref_caps(id: &str, r: &Ltx2Recipe, fps: u32) -> ModelCaps {
    let base = ltx2_caps(id, &Ltx2Recipe { ic_lora: None, ..r.clone() });
    let grid = FrameGrid::new(8, 1, 9, LTX_REF_FRAMES_MAX, 121);
    ModelCaps {
        tasks: [Task::Ref2V].into_iter().collect(),
        stream: Some(StreamCaps::Clip {
            min_s: grid.min as f32 / fps as f32,
            max_s: grid.max as f32 / fps as f32,
        }),
        frames: grid,
        canvas: CanvasCaps {
            // The first tier is the default: 16:9 at 896 is 1600x896 (stage 1
            // 800x448); fal's `ingredient` default is exactly 1536x896.
            short_edges: vec![LTX_REF_CANVAS.1, 720, 1080],
            max_area: 1920 * 1088,
            ..base.canvas
        },
        refs: RefLimits::ltx_ingredients(),
        knobs: KnobCaps {
            seed: true,
            reference_strength: true,
            ..KnobCaps::default()
        },
        ..base
    }
}

/// LTX-2.5 guided audio-to-video (the dev-DiT companion of `ltx-pro`):
/// audio-to-video only, the video CFG scale a per-request knob
/// (`guidance_scale`; the reference default 3 when unset).
fn ltx2_a2v_guided_caps(id: &str, r: &Ltx2Recipe) -> ModelCaps {
    let base = ltx2_caps(id, &Ltx2Recipe { a2v: LtxA2v::Distilled, ..r.clone() });
    ModelCaps {
        tasks: [Task::A2V].into_iter().collect(),
        knobs: KnobCaps {
            seed: true,
            guidance: true,
            ..KnobCaps::default()
        },
        ..base
    }
}

fn wan_caps(id: &str, r: &WanRecipe) -> ModelCaps {
    let (_, _, frames, fps) = r.default;
    let grid = FrameGrid::new(4, 1, 9, r.frames_max, frames);
    let mut tasks: BTreeSet<Task> = [Task::T2V].into_iter().collect();
    if r.i2v {
        tasks.insert(Task::I2V);
    }
    let knobs = match r.sampler {
        // DMD runs one conditional pass: no negative prompt, no guidance.
        WanSampler::Dmd { .. } => KnobCaps {
            seed: true,
            steps: true,
            flow_shift: true,
            ..KnobCaps::default()
        },
        WanSampler::Unipc { .. } => KnobCaps {
            guidance_2: false,
            ..KnobCaps::all()
        },
    };
    ModelCaps {
        id: ModelId::new(id),
        family: Family::Wan,
        served_names: vec![id.to_owned()],
        tasks,
        audio: None,
        // The frames do not depend on the fps: the backend muxes the MP4 at
        // the job's fps (FastWan clients send 24 for a 16 fps model), so both
        // container rates are accepted; the model's own rate is the default.
        // Any integer rate 4..=60 (fal's `frames_per_second`), the model's first.
        fps: FpsCaps {
            allowed: std::iter::once(fps).chain((WAN_FPS_MIN..=WAN_FPS_MAX).filter(|&f| f != fps)).collect(),
            default: fps,
            container_only: true,
        },
        stream: Some(StreamCaps::Clip {
            min_s: grid.min as f32 / fps as f32,
            max_s: grid.max as f32 / fps as f32,
        }),
        frames: grid,
        canvas: CanvasCaps {
            multiple: r.multiple,
            max_area: r.max_area,
            aspect: (0.25, 4.0),
            short_edges: r.short_edges.clone(),
            pad_and_crop: false,
            hd: None,
        },
        refs: RefLimits::none(),
        knobs,
        resident: true,
        tier: None,
        recipe: None,
    }
}

fn sfwan_caps(id: &str, r: &SfWanRecipe) -> ModelCaps {
    let mut c = wan_caps(id, &r.wan);
    let fps = r.wan.default.3;
    c.fps = FpsCaps::fixed(fps);
    // A bounded clip is whole causal blocks of latent frames: 4 (L - 1) + 1
    // pixel frames for L a multiple of the block (3: 9, 21, ..., 81).
    let fpb = fastvideo_models::wan::config::WanVideoArchConfig::from_preset(&r.wan.preset).num_frames_per_block.max(1) as u32;
    let first = 4 * (fpb - 1) + 1;
    let max = first + (r.wan.frames_max.saturating_sub(first) / (4 * fpb)) * 4 * fpb;
    c.frames = FrameGrid::new(4 * fpb, first, first, max, r.wan.default.2.clamp(first, max));
    c.stream = Some(StreamCaps::Causal {
        block_frames: r.block_frames,
        target_fps: fps,
    });
    c.knobs = KnobCaps {
        seed: true,
        ..KnobCaps::default()
    };
    c
}

/// The LTX model config for a version.
pub fn ltx_config(v: LtxVersion) -> fastvideo_models::ltx2::config::Ltx2Config {
    match v {
        LtxVersion::V23 => fastvideo_models::ltx2::config::ltx2_23_22b_distilled(),
        LtxVersion::V25 => fastvideo_models::ltx2::config::ltx2_5_22b_distilled(),
    }
}

fn h3(
    layout: &WeightLayout,
    cell: &str,
    recipe: &str,
    profile: Option<&str>,
    steps: u32,
) -> H3Recipe {
    H3Recipe {
        recipe: recipe.to_owned(),
        weights: layout.at(cell),
        dense: false,
        profile: profile.map(str::to_owned),
        taeh3: None,
        short_edge: h3cfg::H3_SHORT_EDGE as u32,
        text_encoder: "auto".to_owned(),
        // The base snapshot carries the text encoder for every H3 cell.
        text_weights: Some(layout.at("h3-base")),
        adaln_cache: None,
        dit_offload: None,
        steps,
        ref2va: false,
        ref_weights: None,
        i2v_encoder: auto_str(),
        warmup: false,
        hd_1080p: false,
    }
}

fn ltx25(layout: &WeightLayout, stage2: LtxStage2, profile: &str) -> Ltx2Recipe {
    Ltx2Recipe {
        version: LtxVersion::V25,
        weights: layout.at("ltx25"),
        dit: layout.at("ltx25"),
        text_weights: None,
        two_stage: true,
        stage2,
        profile: Some(profile.to_owned()),
        tae: None,
        text: "streamed".to_owned(),
        canvas: (1920, 1080),
        stage1_steps: 8,
        refine_steps: 3,
        ic_lora: None,
        a2v: LtxA2v::Distilled,
    }
}

fn fastwan(layout: &WeightLayout, decoder: WanDecoder) -> WanRecipe {
    WanRecipe {
        preset: "fast_wan_t2v_480p".to_owned(),
        weights: layout.at("fastwan21-1.3b"),
        sampler: WanSampler::Dmd { steps: 3 },
        flow_shift: 8.0,
        vsa: true,
        decoder,
        default: (832, 480, 81, 16),
        max_area: 832 * 480,
        short_edges: vec![480],
        multiple: 16,
        frames_max: 129,
        i2v: false,
        negative: WAN_NEGATIVE_EN.to_owned(),
        tae_dir: Some(layout.tae_dir.clone()),
    }
}

/// Every standard configuration, one per tier plus the untiered Sol-H3 and
/// SF-Wan. A deployment picks the ones one process serves (see
/// [`ProcessPlan::for_models`]).
pub fn catalog(layout: &WeightLayout) -> Vec<CudaModel> {
    let tae = |f: &str| layout.tae_dir.join(f);
    let h3_8step = H3Recipe {
        dense: true,
        ..h3(layout, "h3-8step", "8step", None, 8)
    };
    // h3-turbo and h3-max offer the native 1080P tier (9/9 coherent clips
    // at 1920x1088 and 1088x1920, docs/serve/h3-1080p-and-upscaler.md).
    let h3_turbo = H3Recipe {
        hd_1080p: true,
        ..h3(layout, "h3-base", "4step-vsa", Some("h3/fasth3_4step_vsa"), 4)
    };
    let h3_draft = H3Recipe {
        taeh3: Some(tae("taeh3.safetensors")),
        short_edge: h3cfg::H3_SHORT_EDGE_480P as u32,
        hd_1080p: false,
        ..h3_turbo.clone()
    };
    let sol_h3 = H3Recipe {
        hd_1080p: true,
        ..h3(layout, "h3-base", "sol-h3", Some(SOL_H3_4STEP_PROFILE), 4)
    };
    // Ref2VA (docs/ports/h3-ref2v.md): `transformer_ref/` and the turbo LoRA
    // live in `h3-ref2va`; the text encoder and both VAEs come from `h3-base`.
    let h3_ref = |recipe: &str, profile: &str, steps: u32| H3Recipe {
        dense: true,
        ref2va: true,
        ref_weights: Some(layout.at("h3-ref2va")),
        ..h3(layout, "h3-base", recipe, Some(profile), steps)
    };
    let h3_ref_max = h3_ref("base", "h3/h3_ref2va_base", 49);
    let h3_ref_turbo = h3_ref("sol-h3-ref2va", "h3/sol_h3_ref2va", 4);
    // Reference-to-video (docs/ports/ltx-ref2v.md): the dense two-stage with
    // the Ingredients IC-LoRA at stage 1, as `ICLoraPipeline`.
    let ltx_ref = Ltx2Recipe {
        ic_lora: Some(
            layout
                .at(fastvideo_models::ltx2::lora::LTX25_INGREDIENTS_DIR)
                .join(fastvideo_models::ltx2::lora::LTX25_INGREDIENTS_FILE),
        ),
        canvas: LTX_REF_CANVAS,
        ..ltx25(layout, LtxStage2::Dense, "ltx2/ltx25_distill_dense")
    };
    // Audio-to-video on the dev transformer, guided (`A2VidPipelineTwoStage`):
    // the companion of `ltx-pro` for that task.
    let ltx_a2v_guided = Ltx2Recipe {
        dit: layout.root.join(LTX25_DEV_DIT),
        a2v: LtxA2v::Guided,
        stage1_steps: fastvideo_models::ltx2::guidance::LTX25_DEV_STEPS as u32,
        ..ltx25(layout, LtxStage2::Dense, "ltx2/ltx25_distill_dense")
    };
    let ltx_draft = Ltx2Recipe {
        tae: Some(tae("taeltx2_3_wide.safetensors")),
        ..ltx25(layout, LtxStage2::Sol, LTX_DRAFT_PROFILE)
    };
    let wan_max = WanRecipe {
        preset: "wan_2_2_ti2v_5b".to_owned(),
        weights: layout.at("wan22-ti2v-5b"),
        sampler: WanSampler::Unipc {
            steps: 50,
            guidance: 5.0,
        },
        flow_shift: 5.0,
        vsa: false,
        decoder: WanDecoder::Full,
        default: (1280, 704, 121, 24),
        max_area: 1280 * 704,
        // fal's 720p, 580p and 480p (docs/serve/fal-parity.md §3).
        short_edges: vec![704, 576, 480],
        multiple: 32,
        frames_max: WAN5B_FRAMES_MAX,
        i2v: true,
        negative: WAN_NEGATIVE_CN.to_owned(),
        tae_dir: None,
    };
    // FastWan2.2 TI2V-5B (FullAttn): the TI2V-5B network DMD-distilled to 3
    // steps (1000/757/522), shift 5, full attention; trained at 704x1280x121.
    let fastwan22 = |decoder: WanDecoder| WanRecipe {
        preset: "fast_wan_2_2_ti2v_5b".to_owned(),
        weights: layout.at("fastwan22-ti2v-5b"),
        sampler: WanSampler::Dmd { steps: 3 },
        decoder,
        tae_dir: Some(layout.tae_dir.clone()),
        ..wan_max.clone()
    };
    let sfwan = SfWanRecipe {
        wan: WanRecipe {
            preset: "sf_wan_t2v_1_3b".to_owned(),
            weights: layout.at("sfwan21-1.3b"),
            sampler: WanSampler::Dmd { steps: 4 },
            flow_shift: 5.0,
            vsa: false,
            decoder: WanDecoder::Taehv,
            frames_max: 81,
            ..fastwan(layout, WanDecoder::Taehv)
        },
        local_attn_frames: 21,
        // The deep sink (`wan::stream::RolloutConfig`'s default): a one-block
        // sink degrades within a minute (docs/ports/wan.md).
        sink_frames: 15,
        block_frames: 12,
    };
    vec![
        CudaModel::new(
            "sol-h3",
            Some(Tier::Max),
            "sol-h3-4step-engine-ladder",
            CudaRecipe::H3(sol_h3),
        ),
        CudaModel::new(
            "fasth3-4step-vsa",
            Some(Tier::Turbo),
            "fasth3-4step-vsa",
            CudaRecipe::H3(h3_turbo),
        )
        .with_served_names(&["fasth3"]),
        CudaModel::new(
            "fasth3-4step-vsa-480p-taeh3",
            Some(Tier::Draft),
            "fasth3-4step-vsa-480p-taeh3",
            CudaRecipe::H3(h3_draft),
        ),
        CudaModel::new(
            "fasth3-8step-dense",
            None,
            "fasth3-8step-dense",
            CudaRecipe::H3(h3_8step),
        ),
        // Reference-to-video companions of the H3 max / turbo tiers: the
        // routing sends Ref2V requests for `h3-max` / `h3-turbo` here.
        CudaModel::new(
            "h3-ref2v-max",
            Some(Tier::Max),
            "h3-ref2va-base-49step",
            CudaRecipe::H3(h3_ref_max),
        ),
        CudaModel::new(
            "h3-ref2v-turbo",
            Some(Tier::Turbo),
            "sol-h3-ref2va",
            CudaRecipe::H3(h3_ref_turbo),
        ),
        CudaModel::new(
            "ltx25-distill-dense",
            Some(Tier::Max),
            "ltx25-distill-two-stage-dense",
            CudaRecipe::Ltx2(Ltx2Recipe {
                a2v: LtxA2v::Off,
                ..ltx25(layout, LtxStage2::Dense, "ltx2/ltx25_distill_dense")
            }),
        ),
        CudaModel::new(
            "ltx25-distill-sol",
            Some(Tier::Turbo),
            "ltx25-distill-two-stage-sol",
            CudaRecipe::Ltx2(ltx25(layout, LtxStage2::Sol, "ltx2/ltx25_distill_sol")),
        ),
        CudaModel::new(
            "ltx25-distill-sol-nvfp4-taehv",
            Some(Tier::Draft),
            "ltx25-distill-two-stage-sol-nvfp4-taehv",
            CudaRecipe::Ltx2(ltx_draft),
        ),
        // Reference-to-video companion of `ltx-pro`: the routing sends Ref2V
        // requests for `ltx-pro` here.
        CudaModel::new(
            "ltx25-ref2v",
            Some(Tier::Max),
            "ltx25-ic-lora-ingredients-dense",
            CudaRecipe::Ltx2(ltx_ref),
        ),
        // Audio-to-video companion of `ltx-pro`: the routing sends A2V
        // requests for `ltx-pro` here (the fast tier keeps the distilled A2V).
        CudaModel::new(
            "ltx25-a2v-guided",
            Some(Tier::Max),
            "ltx25-dev-a2v-guided",
            CudaRecipe::Ltx2(ltx_a2v_guided),
        ),
        CudaModel::new(
            "wan22-ti2v-5b",
            Some(Tier::Max),
            "wan22-ti2v-5b-unipc50",
            CudaRecipe::Wan(wan_max.clone()),
        ),
        CudaModel::new(
            "fastwan22-ti2v-5b",
            Some(Tier::Turbo),
            "fastwan22-ti2v-5b-dmd3",
            CudaRecipe::Wan(fastwan22(WanDecoder::Full)),
        )
        .with_served_names(&["FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"]),
        CudaModel::new(
            "fastwan22-ti2v-5b-taehv",
            Some(Tier::Draft),
            "fastwan22-ti2v-5b-dmd3-taehv",
            CudaRecipe::Wan(fastwan22(WanDecoder::Taehv)),
        ),
        CudaModel::new(
            "fastwan21-1.3b",
            None,
            "fastwan21-1.3b-dmd3-vsa",
            CudaRecipe::Wan(fastwan(layout, WanDecoder::Full)),
        )
        .with_served_names(&["FastVideo/FastWan2.1-T2V-1.3B-Diffusers"]),
        CudaModel::new(
            "fastwan21-1.3b-taehv",
            None,
            "fastwan21-1.3b-dmd3-vsa-taehv",
            CudaRecipe::Wan(fastwan(layout, WanDecoder::Taehv)),
        ),
        CudaModel::new(
            "sfwan21-1.3b",
            None,
            "sfwan21-1.3b-causal",
            CudaRecipe::SfWan(sfwan),
        ),
    ]
}

/// The catalog model bound to a tier alias (`h3-turbo`) or with this id.
pub fn find(catalog: &[CudaModel], name: &str) -> Option<CudaModel> {
    let tier = crate::caps::parse_tier_alias(name);
    catalog
        .iter()
        .find(|m| match tier {
            Some((f, t)) => m.family() == f && m.tier == Some(t),
            None => m.id.as_str() == name || m.served_names.iter().any(|s| s == name),
        })
        .cloned()
}

/// One `[[models]]` entry of the serve config.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelEntryCfg {
    /// The served model id (replaces the catalog id; tier tags are kept).
    pub id: String,
    /// `h3` | `ltx2` (`ltx`) | `wan`.
    pub family: String,
    /// A tier alias (`h3-turbo`), a catalog id (`fasth3-4step-vsa`) or an H3
    /// contract name (`4step-vsa`, `8step`, `sol-h3`). Empty: the family's
    /// turbo tier.
    pub recipe: String,
    /// The model's weight directory (replaces the catalog's).
    pub weights: Option<PathBuf>,
    pub resident: bool,
    pub served_names: Vec<String>,
    /// Optional per-model settings: `text_encoder`, `i2v_encoder`, `warmup`, `h3_1080p`
    /// (H3), `text` (LTX), `adaln_cache` (H3), `taeh3` / `tae` (tiny
    /// decoders).
    pub extra: BTreeMap<String, String>,
}

/// Resolves a `[[models]]` entry against the catalog.
pub fn model_from_config(layout: &WeightLayout, e: &ModelEntryCfg) -> Result<CudaModel, String> {
    let cat = catalog(layout);
    let family = match e.family.as_str() {
        "h3" => Family::H3,
        "ltx" | "ltx2" => Family::Ltx2,
        "wan" => Family::Wan,
        other => return Err(format!("model `{}`: unknown family `{other}` (h3, ltx2, wan)", e.id)),
    };
    let name = match (family, e.recipe.as_str()) {
        (_, "") => crate::caps::tier_alias(family, Tier::Turbo).unwrap_or_default().to_owned(),
        (Family::H3, "4step-vsa" | "preview-vsa" | "fasth3-4step-vsa") => "h3-turbo".to_owned(),
        (Family::H3, "8step" | "v2" | "fasth3-8step") => "fasth3-8step-dense".to_owned(),
        (Family::H3, "sol-h3" | "sol_h3") => "h3-max".to_owned(),
        (_, r) => r.to_owned(),
    };
    let mut m = find(&cat, &name).ok_or_else(|| {
        format!(
            "model `{}`: recipe `{}` is not in the CUDA catalog ({})",
            e.id,
            e.recipe,
            cat.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", ")
        )
    })?;
    if m.family() != family {
        return Err(format!("model `{}`: recipe `{}` is not a {} recipe", e.id, e.recipe, e.family));
    }
    if !e.id.is_empty() {
        m.id = ModelId::new(&e.id);
        m.served_names.insert(0, e.id.clone());
    }
    for n in &e.served_names {
        if !m.served_names.contains(n) {
            m.served_names.push(n.clone());
        }
    }
    m.served_names.dedup();
    m.resident = e.resident;
    let x = |k: &str| e.extra.get(k).cloned();
    match &mut m.recipe {
        CudaRecipe::H3(r) => {
            if let Some(w) = &e.weights {
                r.weights = w.clone();
            }
            if let Some(t) = x("text_encoder") {
                r.text_encoder = t;
            }
            if let Some(a) = x("adaln_cache") {
                r.adaln_cache = Some(a.into());
            }
            if let Some(t) = x("taeh3").filter(|_| r.taeh3.is_some()) {
                r.taeh3 = Some(t.into());
            }
            if let Some(t) = x("text_weights") {
                r.text_weights = Some(t.into());
            }
            if let Some(t) = x("ref_weights").filter(|_| r.ref2va) {
                r.ref_weights = Some(t.into());
            }
            if let Some(t) = x("i2v_encoder") {
                r.i2v_encoder = t;
            }
            if let Some(w) = x("warmup") {
                r.warmup = match w.as_str() {
                    "true" => true,
                    "false" => false,
                    other => {
                        return Err(format!("model `{}`: warmup = {other:?} (true | false)", e.id))
                    }
                };
            }
            if let Some(w) = x("h3_1080p") {
                r.hd_1080p = match w.as_str() {
                    "true" => {
                        if !r.hd_1080p {
                            return Err(format!(
                                "model `{}`: h3_1080p = true: the native 1080P tier is validated for h3-max and h3-turbo only",
                                e.id
                            ));
                        }
                        true
                    }
                    "false" => false,
                    other => {
                        return Err(format!("model `{}`: h3_1080p = {other:?} (true | false)", e.id))
                    }
                };
            }
            if r.ref2va && r.warmup {
                return Err(format!(
                    "model `{}`: warmup runs text- and image-to-video; a Ref2VA model serves references only",
                    e.id
                ));
            }
        }
        CudaRecipe::Ltx2(r) => {
            if let Some(w) = &e.weights {
                r.weights = w.clone();
                r.dit = w.clone();
            }
            if let Some(t) = x("text") {
                r.text = t;
            }
            if let Some(t) = x("tae").filter(|_| r.tae.is_some()) {
                r.tae = Some(t.into());
            }
            if let Some(t) = x("ic_lora").filter(|_| r.ic_lora.is_some()) {
                r.ic_lora = Some(t.into());
            }
            if r.a2v == LtxA2v::Guided {
                // The dev DiT: `dit = "..."`, else beside the weight root's
                // parent as in the catalog layout.
                r.dit = match x("dit") {
                    Some(d) => d.into(),
                    None => match &e.weights {
                        Some(w) => w.parent().unwrap_or(w).join(LTX25_DEV_DIT),
                        None => r.dit.clone(),
                    },
                };
            }
        }
        CudaRecipe::Wan(r) => {
            if let Some(w) = &e.weights {
                r.weights = w.clone();
            }
        }
        CudaRecipe::SfWan(r) => {
            if let Some(w) = &e.weights {
                r.wan.weights = w.clone();
            }
        }
    }
    Ok(m)
}

/// Loads a technique profile by builtin name (or file path).
pub fn load_profile(name: &str) -> Result<fastvideo_models::techniques::Profile, String> {
    fastvideo_models::techniques::Profile::load(Path::new(name))
        .map_err(|e| format!("profile {name}: {e}"))
}

/// Process-wide settings one process needs for a set of models.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProcessPlan {
    /// The profile to install process-wide (the first model's).
    pub profile: Option<String>,
    /// Its load-time `FASTVIDEO_*` settings.
    pub settings: BTreeMap<String, String>,
    /// Process env the recipes need.
    pub env: BTreeMap<String, String>,
}

fn settings_of(profile: Option<&str>) -> Result<BTreeMap<String, String>, String> {
    let Some(name) = profile else {
        return Ok(BTreeMap::new());
    };
    let p = load_profile(name)?;
    let s = p.settings().map_err(|e| format!("profile {name}: {e}"))?;
    Ok(s.iter()
        .map(|(k, v, _)| (k.to_owned(), v.to_owned()))
        .collect())
}

impl ProcessPlan {
    /// Checks that `models` can share one process and returns what to install.
    pub fn for_models(models: &[CudaModel]) -> Result<Self, String> {
        let mut plan = ProcessPlan::default();
        let mut first: Option<(&str, BTreeMap<String, String>)> = None;
        for m in models {
            let settings = settings_of(m.recipe.profile())?;
            if let CudaRecipe::H3(r) = &m.recipe {
                if let Some(name) = &r.profile {
                    let p = load_profile(name)?;
                    if let Some(want) = &p.recipe {
                        if h3cfg::H3InferenceContract::named(want).ok()
                            != h3cfg::H3InferenceContract::named(&r.recipe).ok()
                        {
                            return Err(format!(
                                "model `{}`: profile {name} runs recipe {want}, the model runs {}",
                                m.id, r.recipe
                            ));
                        }
                    }
                }
            }
            match &first {
                None => {
                    plan.profile = m.recipe.profile().map(str::to_owned);
                    plan.settings = settings.clone();
                    first = Some((m.id.as_str(), settings));
                }
                Some((id, s)) if *s != settings => {
                    return Err(format!(
                        "models `{id}` and `{}` need different process-wide settings ({s:?} vs {settings:?}); \
                         serve them from separate processes",
                        m.id
                    ));
                }
                Some(_) => {
                    if plan.profile.is_none() {
                        plan.profile = m.recipe.profile().map(str::to_owned);
                    }
                }
            }
            for (k, v) in m.recipe.process_env() {
                match plan.env.get(k) {
                    Some(have) if have != v => {
                        return Err(format!(
                            "model `{}` needs {k}={v}, another model in this process needs {k}={have}; \
                             serve them from separate processes",
                            m.id
                        ));
                    }
                    _ => {
                        plan.env.insert(k.to_owned(), v.to_owned());
                    }
                }
            }
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::CapabilityTable;

    fn table(models: &[CudaModel]) -> CapabilityTable {
        CapabilityTable::build(
            vec![models.iter().map(|m| (m.caps(), m.describe())).collect()],
            &BTreeMap::new(),
        )
        .unwrap()
    }

    #[test]
    fn every_tier_alias_is_bound() {
        let cat = catalog(&WeightLayout::default());
        let t = table(&cat);
        let want = [
            ("h3-max", "sol-h3"),
            ("h3-turbo", "fasth3-4step-vsa"),
            ("h3-draft", "fasth3-4step-vsa-480p-taeh3"),
            ("ltx-pro", "ltx25-distill-dense"),
            ("ltx-turbo", "ltx25-distill-sol"),
            ("ltx-draft", "ltx25-distill-sol-nvfp4-taehv"),
            ("wan-max", "wan22-ti2v-5b"),
            ("wan-turbo", "fastwan22-ti2v-5b"),
            ("wan-draft", "fastwan22-ti2v-5b-taehv"),
        ];
        for (alias, id) in want {
            assert_eq!(t.resolve(alias).map(|c| c.id.as_str()), Some(id), "{alias}");
            assert_eq!(find(&cat, alias).unwrap().id.as_str(), id);
            let c = t.resolve(alias).unwrap();
            let (_, tier) = crate::caps::parse_tier_alias(alias).unwrap();
            assert_eq!(c.tier, Some(tier), "{alias} keeps its tier tag");
            assert!(c.resident);
        }
        assert_eq!(t.resolve("fasth3").unwrap().id.as_str(), "fasth3-4step-vsa");
        assert!(t.get(&ModelId::new("fasth3-8step-dense")).unwrap().tier.is_none());
        assert_eq!(t.len(), cat.len());
    }

    #[test]
    fn recipes_and_profiles() {
        let cat = catalog(&WeightLayout::default());
        let t = table(&cat);
        let r = |id: &str| t.recipe(&ModelId::new(id)).unwrap().clone();
        // §0.5: Sol-H3 4-step serves the tau ladder.
        assert_eq!(r("sol-h3").profile.as_deref(), Some(SOL_H3_4STEP_PROFILE));
        assert_eq!(
            r("fasth3-4step-vsa").profile.as_deref(),
            Some("h3/fasth3_4step_vsa")
        );
        assert_eq!(r("fasth3-4step-vsa").steps, Some(4));
        assert_eq!(r("fasth3-4step-vsa").attention, "vsa");
        // The untiered 8-step: no profile (no MXFP8), dense, full VAE, 8 steps.
        let max = r("fasth3-8step-dense");
        assert_eq!(
            (
                max.profile,
                max.steps,
                max.attention.as_str(),
                max.vae.as_str()
            ),
            (None, Some(8), "dense", "full")
        );
        assert_eq!(r("fasth3-4step-vsa-480p-taeh3").vae, "taeh3");
        assert_eq!(
            r("ltx25-distill-sol").profile.as_deref(),
            Some("ltx2/ltx25_distill_sol")
        );
        assert_eq!(r("ltx25-distill-sol").steps, Some(11));
        assert_eq!(r("ltx25-distill-dense").attention, "dense");
        assert_eq!(
            r("ltx25-distill-sol-nvfp4-taehv").profile.as_deref(),
            Some(LTX_DRAFT_PROFILE)
        );
        assert_eq!(r("ltx25-distill-sol-nvfp4-taehv").vae, "taehv");
        assert_eq!(r("fastwan21-1.3b").steps, Some(3));
        assert_eq!(r("fastwan21-1.3b").vae, "full");
        assert_eq!(r("fastwan21-1.3b-taehv").vae, "taehv");
        assert_eq!(r("wan22-ti2v-5b").steps, Some(50));
        // Every named profile is a builtin and runs the model's recipe.
        for m in &cat {
            if let Some(p) = m.recipe.profile() {
                let prof = load_profile(p).unwrap();
                match &m.recipe {
                    CudaRecipe::H3(h) => {
                        assert_eq!(prof.model, "h3", "{p}");
                        let want = prof.recipe.clone().unwrap();
                        assert_eq!(
                            h3cfg::H3InferenceContract::named(&want).unwrap(),
                            h3cfg::H3InferenceContract::named(&h.recipe).unwrap(),
                            "{p}"
                        );
                    }
                    CudaRecipe::Ltx2(_) => assert_eq!(prof.model, "ltx2", "{p}"),
                    _ => panic!("{p} on a Wan model"),
                }
                // Also as a repo file (what the image ships).
                let f =
                    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../profiles/{p}.toml"));
                assert!(f.is_file(), "{}", f.display());
            }
        }
    }

    #[test]
    fn caps_follow_the_configs() {
        let cat = catalog(&WeightLayout::default());
        let get = |id: &str| cat.iter().find(|m| m.id.as_str() == id).unwrap().caps();
        let h3 = get("fasth3-4step-vsa");
        assert_eq!(h3.audio.as_ref().unwrap().native_rate, 32_000);
        assert_eq!(h3.audio.as_ref().unwrap().channels, 2);
        assert_eq!(h3.canvas.short_edges, vec![768, 480, 1080]);
        assert_eq!(h3.canvas.hd, Some(fastvideo_protocol::HdTier::h3_1080p()));
        // The 1080 tier does not move the budgets of the others.
        assert_eq!(h3.canvas.area_at(768), 768 * 1344);
        assert_eq!(h3.canvas.area_at(1080), 1088 * 1920);
        assert_eq!(fastvideo_protocol::canvas_for_aspect(&h3.canvas, 16.0 / 9.0, 480), (832, 480));
        assert_eq!(fastvideo_protocol::canvas_for_aspect(&h3.canvas, 16.0 / 9.0, 768), (1344, 768));
        assert_eq!(get("sol-h3").canvas.short_edges, vec![768, 480, 1080]);
        assert_eq!(get("fasth3-8step-dense").canvas.short_edges, vec![768, 480]);
        assert_eq!(h3.frames.default, 124);
        assert!(h3.frames.contains(107) && h3.frames.contains(362));
        assert!(
            h3.supports(Task::I2V) && h3.supports(Task::Keyframes) && !h3.supports(Task::Ref2V)
        );
        let draft = get("fasth3-4step-vsa-480p-taeh3");
        assert_eq!(draft.canvas.short_edges, vec![480]);
        assert_eq!(draft.canvas.hd, None);
        assert!(draft.canvas.area_at(480) >= 832 * 480);
        assert!(draft.canvas.area_at(480) < 768 * 1344);
        let ltx = get("ltx25-distill-sol");
        let cfg = ltx_config(LtxVersion::V25);
        assert_eq!(
            ltx.audio.as_ref().unwrap().native_rate,
            cfg.vocoder.output_sampling_rate as u32
        );
        assert_eq!(
            ltx.audio.as_ref().unwrap().channels as usize,
            cfg.vocoder.out_channels
        );
        assert_eq!(ltx.fps.default, 24);
        assert_eq!(ltx.canvas.multiple, 64);
        assert!(ltx.canvas.pad_and_crop);
        assert!(ltx.frames.contains(121) && !ltx.frames.contains(120));
        assert_eq!(
            ltx.knobs,
            KnobCaps {
                seed: true,
                ..KnobCaps::default()
            }
        );
        let wan = get("fastwan21-1.3b");
        assert!(wan.audio.is_none());
        assert_eq!((wan.frames.default, wan.fps.default), (81, 16));
        assert!(wan.fps.container_only);
        // FastWan API clients send 24 fps: accepted as the container rate.
        assert!(wan.fps.allows(24) && wan.fps.allows(16));
        let mut req = fastvideo_protocol::GenerationRequest::text(
            fastvideo_protocol::ProtocolId::FastWan,
            "fastwan21-1.3b",
            "a cat",
        );
        req.timing.fps = Some(24);
        let r = fastvideo_protocol::negotiate(&req, &wan, &Default::default()).unwrap();
        assert_eq!((r.fps, r.num_frames), (24, 81));
        assert!(!wan.knobs.negative && !wan.knobs.guidance && wan.knobs.steps);
        let ti2v = get("wan22-ti2v-5b");
        assert!(ti2v.supports(Task::I2V));
        assert!(ti2v.knobs.guidance && ti2v.knobs.negative && !ti2v.knobs.guidance_2);
        assert_eq!(ti2v.fps.default, 24);
        // fal's Wan 5B: 720p/580p/480p, 17..=161 frames, 4..=60 fps (container).
        for id in ["wan22-ti2v-5b", "fastwan22-ti2v-5b", "fastwan22-ti2v-5b-taehv"] {
            let c = get(id);
            assert_eq!(c.canvas.short_edges, vec![704, 576, 480], "{id}");
            assert_eq!((c.frames.min, c.frames.max, c.frames.default), (9, 161, 121), "{id}");
            assert!(c.frames.contains(17) && c.frames.contains(161));
            assert!((4..=60).all(|f| c.fps.allows(f)) && !c.fps.allows(3) && !c.fps.allows(61), "{id}");
            assert!(c.supports(Task::I2V), "{id}");
            let mut req = fastvideo_protocol::GenerationRequest::text(fastvideo_protocol::ProtocolId::Fal, id, "a cat");
            req.canvas = fastvideo_protocol::CanvasSpec::Aspect { ratio: fastvideo_protocol::Ratio::R16_9, short_edge: 576 };
            req.timing = fastvideo_protocol::TimingSpec {
                length: fastvideo_protocol::Length::Frames { value: 161, snap: fastvideo_protocol::Snap::AlignUp },
                fps: Some(60),
            };
            let r = fastvideo_protocol::negotiate(&req, &c, &Default::default()).unwrap();
            assert_eq!((r.width, r.height, r.num_frames, r.fps), (1024, 576, 161, 60), "{id}");
        }
        let fw = get("fastwan22-ti2v-5b");
        assert!(!fw.knobs.negative && !fw.knobs.guidance && fw.knobs.steps && fw.knobs.flow_shift);
        assert_eq!(fw.tier, Some(Tier::Turbo));
        let sf = get("sfwan21-1.3b");
        assert_eq!(
            sf.stream,
            Some(StreamCaps::Causal {
                block_frames: 12,
                target_fps: 16
            })
        );
    }

    #[test]
    fn negotiate_accepts_the_defaults() {
        use fastvideo_protocol::*;
        let cat = catalog(&WeightLayout::default());
        for m in &cat {
            let caps = m.caps();
            let mut req = GenerationRequest::text(ProtocolId::Native, m.id.as_str(), "a cat");
            let mut staged = StagedInputs::default();
            if caps.tasks.iter().copied().collect::<Vec<_>>() == vec![Task::A2V] {
                // The guided A2V companion: its length follows the driving
                // audio (7 s -> 161 frames at 24 fps), its guidance is a knob.
                req.task = Task::A2V;
                req.sampling.guidance = Some(5.0);
                req.audio_in = Some(AudioInput {
                    media: MediaRef::parse("https://e.x/a.wav", "audio_url").unwrap(),
                    role: AudioRole::Drive,
                    max_s: None,
                });
                staged.audio_in = Some(StagedMedia {
                    path: "/stage/a.wav".into(),
                    mime: "audio/wav".into(),
                    bytes: 1,
                    probe: MediaProbe { duration_s: Some(7.0), audio_rate: Some(44_100), ..Default::default() },
                });
                let r = negotiate(&req, &caps, &staged).unwrap_or_else(|e| panic!("{}: {e:?}", m.id));
                assert_eq!((r.num_frames, r.fps, r.sampling.guidance), (161, 24, Some(5.0)), "{}", m.id);
                assert_eq!(r.recipe.as_deref(), Some(m.recipe_name.as_str()));
                continue;
            }
            if !caps.supports(Task::T2V) {
                // Ref2VA models serve reference-to-video only.
                assert_eq!(caps.tasks.iter().copied().collect::<Vec<_>>(), vec![Task::Ref2V]);
                req.task = Task::Ref2V;
                req.references = vec![Reference {
                    kind: MediaKind::Image,
                    media: MediaRef::parse("https://e.x/r.png", "references").unwrap(),
                }];
                staged.references = vec![(
                    MediaKind::Image,
                    StagedMedia {
                        path: "/stage/r.png".into(),
                        mime: "image/png".into(),
                        bytes: 1,
                        probe: MediaProbe { width: Some(1024), height: Some(1024), ..Default::default() },
                    },
                )];
            }
            let r = negotiate(&req, &caps, &staged).unwrap_or_else(|e| panic!("{}: {e:?}", m.id));
            assert_eq!(r.num_frames, caps.frames.default, "{}", m.id);
            assert_eq!(r.recipe.as_deref(), Some(m.recipe_name.as_str()));
        }
    }

    #[test]
    fn ref2va_models_are_task_companions_of_the_h3_tiers() {
        let cat = catalog(&WeightLayout::default());
        let get = |id: &str| cat.iter().find(|m| m.id.as_str() == id).unwrap().clone();
        for (id, tier, recipe, steps) in [
            ("h3-ref2v-max", Tier::Max, "base", 49),
            ("h3-ref2v-turbo", Tier::Turbo, "sol-h3-ref2va", 4),
        ] {
            let m = get(id);
            let CudaRecipe::H3(r) = &m.recipe else { panic!("{id}") };
            assert!(r.ref2va && r.dense, "{id}");
            assert_eq!((r.recipe.as_str(), r.steps), (recipe, steps), "{id}");
            assert!(r.ref_weights.as_ref().unwrap().ends_with("h3-ref2va"), "{id}");
            assert!(r.weights.ends_with("h3-base"), "{id}");
            let c = m.caps();
            assert!(c.supports(Task::Ref2V) && !c.supports(Task::T2V) && !c.supports(Task::I2V));
            assert_eq!(c.refs, fastvideo_protocol::RefLimits::h3());
            assert_eq!(c.tier, Some(tier));
            let d = m.describe();
            assert_eq!((d.attention.as_str(), d.steps), ("dense", Some(steps)), "{id}");
        }
        // The tier aliases still name the base models; Ref2V requests on them
        // route to the companion, other tasks stay.
        let t = table(&cat);
        assert_eq!(t.tier(Family::H3, Tier::Max).unwrap().as_str(), "sol-h3");
        assert_eq!(t.tier(Family::H3, Tier::Turbo).unwrap().as_str(), "fasth3-4step-vsa");
        let models: Vec<&ModelCaps> = t.models().collect();
        for (alias, want) in [("h3-max", "h3-ref2v-max"), ("h3-turbo", "h3-ref2v-turbo")] {
            let base = t.resolve(alias).unwrap();
            assert_eq!(fastvideo_protocol::route_task(base, Task::Ref2V, models.iter().copied()).id.as_str(), want);
            assert_eq!(fastvideo_protocol::route_task(base, Task::I2V, models.iter().copied()).id, base.id);
            assert_eq!(t.get(&ModelId::new(want)).unwrap().tier, base.tier, "{want} keeps its tier tag");
        }
        let tier_max = fastvideo_protocol::resolve_tier(Family::H3, Tier::Max, models.iter().copied()).unwrap();
        assert_eq!(tier_max.id.as_str(), "sol-h3");
    }

    #[test]
    fn ltx_guided_a2v_is_the_task_companion_of_ltx_pro() {
        use fastvideo_protocol::*;
        let cat = catalog(&WeightLayout::default());
        let m = cat.iter().find(|m| m.id.as_str() == "ltx25-a2v-guided").unwrap().clone();
        let CudaRecipe::Ltx2(r) = &m.recipe else { panic!("ltx25-a2v-guided") };
        assert_eq!((r.a2v, r.stage2, r.two_stage), (LtxA2v::Guided, LtxStage2::Dense, true));
        assert_eq!((r.stage1_steps, r.refine_steps), (30, 3));
        assert!(r.dit.ends_with("ltx25-dev/transformer_full"));
        assert!(r.weights.ends_with("ltx25"));
        let c = m.caps();
        assert_eq!(c.tasks.iter().copied().collect::<Vec<_>>(), vec![Task::A2V]);
        assert!(c.knobs.seed && c.knobs.guidance && !c.knobs.steps && !c.knobs.negative);
        assert_eq!(c.tier, Some(Tier::Max));
        assert!(m.describe().summary.starts_with("Audio-to-video, guided"));
        // ltx-pro's plain model no longer takes A2V; the fast tier keeps the
        // distilled one.
        let t = table(&cat);
        let models: Vec<&ModelCaps> = t.models().collect();
        let base = t.resolve("ltx-pro").unwrap();
        assert_eq!(base.id.as_str(), "ltx25-distill-dense");
        assert!(!base.supports(Task::A2V));
        assert_eq!(route_task(base, Task::A2V, models.iter().copied()).id.as_str(), "ltx25-a2v-guided");
        assert_eq!(route_task(base, Task::I2V, models.iter().copied()).id, base.id);
        let turbo = t.resolve("ltx-turbo").unwrap();
        assert!(turbo.supports(Task::A2V));
        assert_eq!(route_task(turbo, Task::A2V, models.iter().copied()).id, turbo.id);
        // A `[[models]]` weight root moves the dev DiT beside it; `dit` names it.
        let e = ModelEntryCfg {
            id: "ltx25-a2v-guided".into(),
            family: "ltx2".into(),
            recipe: "ltx25-a2v-guided".into(),
            weights: Some(PathBuf::from("/w/ltx25")),
            resident: true,
            served_names: Vec::new(),
            extra: BTreeMap::new(),
        };
        let recipe = |e: &ModelEntryCfg| match model_from_config(&WeightLayout::default(), e).unwrap().recipe {
            CudaRecipe::Ltx2(r) => r,
            _ => panic!("not ltx2"),
        };
        let r = recipe(&e);
        assert_eq!(r.dit, PathBuf::from("/w/ltx25-dev/transformer_full"));
        let r2 = recipe(&ModelEntryCfg {
            extra: [("dit".to_owned(), "/x/dev".to_owned())].into_iter().collect(),
            ..e.clone()
        });
        assert_eq!(r2.dit, PathBuf::from("/x/dev"));
        assert_eq!(r.weights, PathBuf::from("/w/ltx25"));
    }

    #[test]
    fn ltx_ref2v_is_the_task_companion_of_ltx_pro() {
        use fastvideo_protocol::*;
        let cat = catalog(&WeightLayout::default());
        let m = cat.iter().find(|m| m.id.as_str() == "ltx25-ref2v").unwrap().clone();
        let CudaRecipe::Ltx2(r) = &m.recipe else { panic!("ltx25-ref2v") };
        assert_eq!(r.stage2, LtxStage2::Dense);
        assert!(r.two_stage);
        assert!(r
            .ic_lora
            .as_ref()
            .unwrap()
            .ends_with("ltx25-ic-lora-ingredients/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors"));
        let c = m.caps();
        assert_eq!(c.tasks.iter().copied().collect::<Vec<_>>(), vec![Task::Ref2V]);
        assert_eq!(c.refs, RefLimits::ltx_ingredients());
        assert!(c.knobs.seed && c.knobs.reference_strength && !c.knobs.steps);
        assert_eq!(c.canvas.short_edges[0], 896);
        assert_eq!((c.frames.max, c.frames.default), (LTX_REF_FRAMES_MAX, 121));
        assert_eq!(c.tier, Some(Tier::Max));
        assert!(m.describe().summary.starts_with("Reference-to-video"));
        // No other LTX model takes the reference knobs.
        for other in cat.iter().filter(|o| o.family() == Family::Ltx2 && o.id != m.id) {
            assert!(!other.caps().knobs.reference_strength, "{}", other.id);
            assert!(!other.caps().supports(Task::Ref2V), "{}", other.id);
        }
        // `ltx-pro` still names the plain model; its Ref2V requests route here.
        let t = table(&cat);
        assert_eq!(t.tier(Family::Ltx2, Tier::Max).unwrap().as_str(), "ltx25-distill-dense");
        let models: Vec<&ModelCaps> = t.models().collect();
        let base = t.resolve("ltx-pro").unwrap();
        assert_eq!(route_task(base, Task::Ref2V, models.iter().copied()).id.as_str(), "ltx25-ref2v");
        assert_eq!(route_task(base, Task::T2V, models.iter().copied()).id, base.id);
        // fal `ingredient`'s default: the sheet at 1536x896, 121 frames, strengths.
        let mut req = GenerationRequest::text(ProtocolId::Fal, "ltx-pro", "Reference sheet: a crab. Generated video: it walks.");
        req.task = Task::Ref2V;
        req.canvas = CanvasSpec::Exact { width: 1536, height: 896 };
        req.references = vec![Reference {
            kind: MediaKind::Image,
            media: MediaRef::parse("https://e.x/sheet.png", "references").unwrap(),
        }];
        req.sampling.reference_strength = Some(0.8);
        req.sampling.reference_lora_strength = Some(1.5);
        let staged = StagedInputs {
            references: vec![(
                MediaKind::Image,
                StagedMedia {
                    path: "/stage/sheet.png".into(),
                    mime: "image/png".into(),
                    bytes: 1,
                    probe: MediaProbe { width: Some(1536), height: Some(896), ..Default::default() },
                },
            )],
            ..Default::default()
        };
        let j = negotiate(&req, &c, &staged).unwrap();
        assert_eq!((j.width, j.height, j.num_frames), (1536, 896, 121));
        assert_eq!(j.sampling.reference_lora_strength, Some(1.5));
        // Two sheets, a strength above 1, or reference knobs on T2V are refused.
        let mut two = req.clone();
        two.references.push(two.references[0].clone());
        assert!(negotiate(&two, &c, &staged).is_err());
        let mut hot = req.clone();
        hot.sampling.reference_strength = Some(1.2);
        assert_eq!(negotiate(&hot, &c, &staged).unwrap_err().param.as_deref(), Some("reference_strength"));
        let plain = get_caps(&cat, "ltx25-distill-dense");
        let mut t2v = GenerationRequest::text(ProtocolId::Fal, "ltx-pro", "a cat");
        t2v.sampling.reference_lora_strength = Some(1.0);
        assert_eq!(
            negotiate(&t2v, &plain, &StagedInputs::default()).unwrap_err().param.as_deref(),
            Some("reference_lora_strength")
        );
    }

    /// Image-to-video (and H3 reference-to-video) with no size or aspect:
    /// every served model derives its canvas from the image at its default
    /// tier, snapped to its multiple and budget, clamped to its aspect range.
    #[test]
    fn image_conditioned_canvas_follows_the_image_on_every_model() {
        use fastvideo_protocol::*;
        let cat = catalog(&WeightLayout::default());
        let staged_image = |w: u32, h: u32| StagedMedia {
            path: "/stage/i.png".into(),
            mime: "image/png".into(),
            bytes: 1,
            probe: MediaProbe { width: Some(w), height: Some(h), ..Default::default() },
        };
        let images = [("landscape", 1920, 1080), ("portrait", 1080, 1920), ("square", 1024, 1024), ("4:5", 1080, 1350), ("extreme", 6000, 500)];
        let mut table = String::new();
        for m in &cat {
            let caps = m.caps();
            let task = if caps.supports(Task::I2V) {
                Task::I2V
            } else if caps.supports(Task::Ref2V) {
                Task::Ref2V
            } else {
                continue;
            };
            let mut row = format!("{:36} {:6}", m.id.as_str(), if task == Task::I2V { "i2v" } else { "ref2v" });
            for (name, w, h) in images {
                let mut req = GenerationRequest::text(ProtocolId::Native, m.id.as_str(), "a cat");
                req.task = task;
                let media = MediaRef::parse("https://e.x/i.png", "image_url").unwrap();
                let mut staged = StagedInputs::default();
                if task == Task::I2V {
                    req.keyframes = vec![Keyframe { at: Anchor::First, image: media }];
                    staged.keyframes = vec![(Anchor::First, staged_image(w, h))];
                } else {
                    req.references = vec![Reference { kind: MediaKind::Image, media }];
                    staged.references = vec![(MediaKind::Image, staged_image(w, h))];
                }
                let (j, notes) = negotiate_noted(&req, &caps, &staged).unwrap_or_else(|e| panic!("{} {name}: {e:?}", m.id));
                let (ow, oh) = j.output_size();
                row.push_str(&format!(" {name}={ow}x{oh}"));
                let c = &caps.canvas;
                assert_eq!((j.width % c.multiple, j.height % c.multiple), (0, 0), "{} {name}", m.id);
                if m.family() == Family::Ltx2 && task == Task::Ref2V {
                    // The reference sheet does not set the canvas: 16:9.
                    assert!(notes.is_empty(), "{} {name}", m.id);
                    assert_eq!((ow, oh), (1592, 896), "{} {name}", m.id);
                    continue;
                }
                let want = (f64::from(w) / f64::from(h)).clamp(0.25, 4.0);
                let got = f64::from(ow) / f64::from(oh);
                // Within one snap step of the image's (clamped) aspect.
                assert!((got / want).ln().abs() < 0.08, "{} {name}: {ow}x{oh} vs {want}", m.id);
                assert_eq!(ow.cmp(&oh), want.total_cmp(&1.0), "{} {name}: {ow}x{oh}", m.id);
                let tier = c.short_edges[0];
                assert!(ow.min(oh) <= tier.max(1088) && u64::from(j.width) * u64::from(j.height) <= c.max_area.max(c.hd.map_or(0, |t| t.max_area)), "{} {name}", m.id);
                assert_eq!(notes.len(), 1, "{} {name}", m.id);
                assert_eq!(notes[0].contains("clamped"), name == "extreme", "{} {name}: {notes:?}", m.id);
            }
            table.push_str(&row);
            table.push('\n');
        }
        eprintln!("{table}");
    }

    fn get_caps(cat: &[CudaModel], id: &str) -> ModelCaps {
        cat.iter().find(|m| m.id.as_str() == id).unwrap().caps()
    }

    #[test]
    fn process_plans() {
        let cat = catalog(&WeightLayout::default());
        let pick = |ids: &[&str]| -> Vec<CudaModel> {
            ids.iter().map(|i| find(&cat, i).unwrap()).collect()
        };
        let p = ProcessPlan::for_models(&pick(&["h3-turbo"])).unwrap();
        assert_eq!(p.profile.as_deref(), Some("h3/fasth3_4step_vsa"));
        assert!(!p.settings.is_empty());
        // Turbo and draft H3 share the load-time settings (per-request
        // techniques may differ).
        let p = ProcessPlan::for_models(&pick(&["h3-turbo", "h3-draft"])).unwrap();
        assert_eq!(p.profile.as_deref(), Some("h3/fasth3_4step_vsa"));
        // Max (the Sol-H3 ladder) shares turbo's load-time settings (MXFP8,
        // bf16 activations); the 8-step dense recipe has no MXFP8.
        ProcessPlan::for_models(&pick(&["h3-max", "h3-turbo"])).unwrap();
        assert!(ProcessPlan::for_models(&pick(&["fasth3-8step-dense", "h3-turbo"])).is_err());
        // NVFP4 is load-time: the LTX draft needs its own process.
        assert!(ProcessPlan::for_models(&pick(&["ltx-turbo", "ltx-draft"])).is_err());
        // FastWan's VSA flag is process-wide; the TI2V-5B recipe runs without it.
        let p = ProcessPlan::for_models(&pick(&["fastwan21-1.3b", "fastwan21-1.3b-taehv"])).unwrap();
        assert_eq!(p.env.get("FASTVIDEO_VSA").map(String::as_str), Some("1"));
        assert!(ProcessPlan::for_models(&pick(&["fastwan21-1.3b", "wan-max"])).is_err());
        // The 5B tiers (FastWan2.2 FullAttn has no VSA) share one process.
        let p = ProcessPlan::for_models(&pick(&["wan-max", "wan-turbo", "wan-draft"])).unwrap();
        assert_eq!(p.env.get("FASTVIDEO_VSA").map(String::as_str), Some("0"));
        let p = ProcessPlan::for_models(&pick(&["wan-max", "sfwan21-1.3b"])).unwrap();
        assert_eq!(p.env.get("FASTVIDEO_VSA").map(String::as_str), Some("0"));
    }

    #[test]
    fn serve_config_entries() {
        let l = WeightLayout::new("/w");
        // configs/serve/runpod.toml's shape.
        let e = ModelEntryCfg {
            id: "fasth3".into(),
            family: "h3".into(),
            recipe: "4step-vsa".into(),
            weights: Some("/w/h3-base".into()),
            resident: true,
            served_names: vec!["fasth3".into()],
            extra: [("text_encoder".to_owned(), "streamed".to_owned())].into_iter().collect(),
        };
        let m = model_from_config(&l, &e).unwrap();
        assert_eq!(m.id.as_str(), "fasth3");
        assert_eq!(m.tier, Some(Tier::Turbo));
        let CudaRecipe::H3(r) = &m.recipe else { panic!() };
        assert_eq!((r.weights.as_path(), r.text_encoder.as_str()), (Path::new("/w/h3-base"), "streamed"));
        assert_eq!((r.i2v_encoder.as_str(), r.warmup), ("auto", false));
        let with = |k: &str, v: &str| ModelEntryCfg {
            extra: [(k.to_owned(), v.to_owned())].into_iter().collect(),
            ..e.clone()
        };
        let h3 = |e: &ModelEntryCfg| match model_from_config(&l, e).map(|m| m.recipe) {
            Ok(CudaRecipe::H3(r)) => Ok(r),
            Ok(_) => panic!("not h3"),
            Err(e) => Err(e),
        };
        assert_eq!(h3(&with("i2v_encoder", "stream")).unwrap().i2v_encoder, "stream");
        assert!(h3(&with("warmup", "true")).unwrap().warmup);
        assert!(!h3(&with("warmup", "false")).unwrap().warmup);
        assert!(h3(&with("warmup", "yes")).is_err());
        assert!(r.hd_1080p);
        assert!(!h3(&with("h3_1080p", "false")).unwrap().hd_1080p);
        assert!(h3(&with("h3_1080p", "true")).unwrap().hd_1080p);
        assert!(h3(&with("h3_1080p", "on")).is_err());
        let e8 = ModelEntryCfg { recipe: "8step".into(), ..with("h3_1080p", "true") };
        assert!(model_from_config(&l, &e8).unwrap_err().contains("h3-max and h3-turbo"));
        // The memory gate: 96 GB and 80 GB cards keep the tier, 48 GB ones not.
        let gib = |g: f64| (g * f64::from(1u32 << 30)) as u64;
        for (total, keep) in [(Some(gib(95.6)), true), (Some(gib(79.6)), true), (Some(gib(44.4)), false), (None, false)] {
            let mut ms = vec![m.clone(), model_from_config(&l, &with("h3_1080p", "false")).unwrap()];
            let off = gate_h3_1080p(&mut ms, total);
            assert_eq!(off.is_none(), keep, "{total:?}");
            assert_eq!(ms[0].caps().canvas.short_edges.contains(&1080), keep);
            if let Some(o) = off {
                assert!(o.contains("fasth3") && o.contains("memory plan"), "{o}");
            }
        }
        let t = CapabilityTable::build(vec![vec![(m.caps(), m.describe())]], &BTreeMap::new()).unwrap();
        assert_eq!(t.resolve("h3-turbo").unwrap().id.as_str(), "fasth3");
        for (fam, rec, want) in [("ltx2", "ltx-turbo", "ltx25-distill-sol"), ("wan", "", "fastwan22-ti2v-5b"), ("wan", "fastwan21-1.3b", "fastwan21-1.3b"), ("h3", "8step", "fasth3-8step-dense"), ("h3", "sol-h3", "sol-h3"), ("h3", "h3-max", "sol-h3")] {
            let e = ModelEntryCfg { family: fam.into(), recipe: rec.into(), resident: true, ..Default::default() };
            assert_eq!(model_from_config(&l, &e).unwrap().id.as_str(), want);
        }
        let bad = ModelEntryCfg { family: "wan".into(), recipe: "h3-turbo".into(), ..Default::default() };
        assert!(model_from_config(&l, &bad).is_err());
    }

    #[test]
    fn layout_paths() {
        let l = WeightLayout::new("/w").with_tae_dir("/t");
        let cat = catalog(&l);
        let m = find(&cat, "h3-draft").unwrap();
        let CudaRecipe::H3(r) = &m.recipe else {
            panic!()
        };
        assert_eq!(r.weights, Path::new("/w/h3-base"));
        assert_eq!(r.taeh3.as_deref(), Some(Path::new("/t/taeh3.safetensors")));
        let m = find(&cat, "h3-max").unwrap();
        let CudaRecipe::H3(r) = &m.recipe else {
            panic!()
        };
        assert_eq!((r.weights.as_path(), r.recipe.as_str()), (Path::new("/w/h3-base"), "sol-h3"));
        assert!(!r.dense);
        let m = find(&cat, "fasth3-8step-dense").unwrap();
        let CudaRecipe::H3(r) = &m.recipe else {
            panic!()
        };
        assert_eq!(r.weights, Path::new("/w/h3-8step"));
        assert!(r.dense);
    }
}
