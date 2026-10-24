//! The CUDA backend's model types: every servable configuration's engine
//! recipe ([`CudaRecipe`]), the served model around it ([`CudaModel`]) and
//! the serve-config resolution onto it ([`model_from_config`]). The rows
//! themselves are not here: the catalog is the projection of the family
//! manifests `families/*.toml` (layer 2, ADR-0008; plan A3),
//! [`super::families::catalog`], and a model's `ModelCaps` is its variant's
//! `caps` + `serving` tables ([`CudaModel::caps`]). The tier each variant
//! holds and why is documented in the family files.
//!
//! This module is plain data and is built without the `cuda` feature, so
//! the projection is unit-tested on CPU. The loaders in [`super::h3`],
//! [`super::ltx2`], [`super::wan`] and [`super::causal`] turn a
//! [`CudaRecipe`] into a resident pipeline.
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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fastvideo_manifest::family::GpuNeeds;
use fastvideo_models::h3::config as h3cfg;
use fastvideo_models::h3::turbo_lora::{parse_turbo_recipe, TurboLora};
use fastvideo_protocol::{Family, ModelCaps, ModelId, StreamCaps, Tier};
use serde::{Deserialize, Serialize};

use crate::caps::Recipe;

pub use super::families::{catalog, catalog_from_manifests, recipe_from_engine, CatalogError};

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
    /// Run one image-to-video and one text-to-video generation at the
    /// default canvas after the load. By default in the background (fast
    /// boot B): the model reports ready once its weights are resident, and
    /// the warm-up runs while the GPU is idle, yielding to every job.
    #[serde(default)]
    pub warmup: bool,
    /// `warmup = "blocking"`: the old behaviour, warm-up inside the load and
    /// ready after it. `FV_WARMUP=blocking|background|off` overrides at run
    /// time ([`WarmupMode::of`]).
    #[serde(default)]
    pub warmup_blocking: bool,
    /// The opt-in native 1080P tier (short edge 1080, generated at
    /// 1920x1088 and cropped; docs/serve/h3-1080p-and-upscaler.md). Offered
    /// only when the GPU passes the tier's memory plan ([`gate_h3_1080p`]).
    /// Catalog: h3-max and h3-turbo; `h3_1080p = false` turns it off.
    #[serde(default)]
    pub hd_1080p: bool,
    /// Turbo LoRA strength (`h3-larry-turbo`, `h3-lightx2v-turbo`; the
    /// larry card's sharpness dial). `None`: 1.0 (or
    /// `FV_H3_TURBO_LORA_STRENGTH`).
    #[serde(default)]
    pub lora_strength: Option<f32>,
    /// VDN-H3 (docs/ports/vdn-h3.md): the VDN checkpoint directory (linear
    /// branch + LoRA adapters) over the h3-base `weights`; `recipe` is a
    /// `vdn-h3-<N>step` contract.
    #[serde(default)]
    pub vdn_weights: Option<PathBuf>,
    /// VDN-H3 8-step: the stage directory the files `vdn_weights` does not
    /// carry come from (stage B's linear branch and default adapter).
    #[serde(default)]
    pub vdn_shared: Option<PathBuf>,
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
    /// SageAttention2 (`fastvideo_cudarc::wan::attn_sage`, INT8 QK / FP8 PV)
    /// on the dense DiT self-attention of this model's two-stage generations,
    /// on sm_120 only (RTX PRO 6000 / RTX 5090); edits stay bf16.
    /// `FASTVIDEO_ATTN_SAGE=0|2` overrides it. Set on `ltx25-distill-dense`
    /// (`ltx-pro`) only: the one recipe that passed the end-to-end gate
    /// (docs/perf/sage-attention.md, Phase 3: 1080p denoise 1.15x).
    #[serde(default)]
    pub sage_attention: bool,
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
    /// LightX2V step-distill Euler (`WanPipeline::set_step_distill`): the
    /// fixed `timesteps` warped with `shift`, one conditional pass per step
    /// (guidance 1), the LongLive-Plug 14B recipe's sampler.
    StepDistill { timesteps: Vec<i32>, shift: f64 },
}

impl WanSampler {
    pub fn steps(&self) -> u32 {
        match self {
            WanSampler::Dmd { steps } | WanSampler::Unipc { steps, .. } => *steps,
            WanSampler::StepDistill { timesteps, .. } => timesteps.len() as u32,
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
    /// A LongLive-Plug recipe (`fastvideo_models::plug`, e.g.
    /// `wan14b-plug-4step`): its adapters merged into `weights/transformer`
    /// on the host at load (`wan::plug::load_merged_transformer`). The
    /// adapters resolve under `$FASTVIDEO_PLUG_ROOT`, else
    /// `<weights>/../longlive-plug` (the volume layout).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plug: Option<String>,
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
    /// LongLive-1.3B on this engine (docs/serve/research-longlive.md):
    /// the LongLive transformer instead of `wan.weights/transformer`, and
    /// its prompt-switch KV re-cache. Opt-in (no catalog model sets it);
    /// pair it with `local_attn_frames: 12`, `sink_frames: 3`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longlive: Option<LongLiveRecipe>,
}

/// The extra served name of a LongLive model: the fal app
/// `fastvideo/longlive` resolves to it (docs/serve/director-causal.md §5).
pub const LONGLIVE_NAME: &str = "longlive";

/// LongLive settings of an [`SfWanRecipe`] (`wan::longlive::LongLiveConfig`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LongLiveRecipe {
    /// The converted checkpoint: `longlive_base.safetensors` and
    /// `lora.safetensors` (same keys as the Hub's `.pt` files).
    pub weights: PathBuf,
    /// Merge `lora.safetensors` (the long-tuned model; off only for the
    /// base-generator ablation).
    #[serde(default = "longlive_on")]
    pub lora: bool,
    /// KV re-cache at a prompt switch (off: keep the cache, the ablation).
    #[serde(default = "longlive_on")]
    pub recache: bool,
    /// The frame sink survives a re-cache (LongLive's `global_sink: true`).
    #[serde(default = "longlive_on")]
    pub global_sink: bool,
    /// Relative (Infinity) RoPE for rollouts past 1024 latent frames; off:
    /// absolute RoPE as the released interactive config.
    #[serde(default)]
    pub relative_rope: bool,
}

fn longlive_on() -> bool {
    true
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

/// One servable model: a variant of a family file under the running
/// process's weight layout, with the deployment's overrides applied
/// ([`model_from_config`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaModel {
    /// The served id (the config's `id`, else the variant id).
    pub id: ModelId,
    /// The family-file variant this model is (`families/*.toml`).
    pub variant: String,
    /// Every name the model answers to: its id first, then the variant's
    /// served names and the deployment's.
    pub served_names: Vec<String>,
    /// The public API names the server maps onto this model
    /// (`VariantManifest::public_names`; the `[aliases]` of old).
    pub public_names: Vec<String>,
    pub tier: Option<Tier>,
    /// Loaded before readiness (the warm pool); otherwise on demand in swap mode.
    pub resident: bool,
    pub recipe: CudaRecipe,
    /// Short recipe name (`ModelCaps::recipe`, `Recipe::name`).
    pub recipe_name: String,
    /// What the variant needs from the GPU (`[variants.gpu]`).
    pub gpu: GpuNeeds,
    /// The variant's `caps` + `serving` as `ModelCaps`, before the
    /// per-deployment fields ([`CudaModel::caps`]).
    pub base_caps: ModelCaps,
}

impl CudaModel {
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
    /// [`crate::device`]): the family file's `gpu.needs` (`fp8` for the H3
    /// recipes with a technique profile or an FP8-resident text encoder,
    /// `nvfp4` for the NVFP4 releases and the LTX draft profile) and the
    /// DiT's weight bytes (when the weights are on disk) within the device.
    pub fn requirements(&self) -> crate::device::Requirements {
        let has = |n: &str| self.gpu.needs.iter().any(|x| x == n);
        crate::device::Requirements {
            fp8: has("fp8") || has("nvfp4"),
            nvfp4: has("nvfp4"),
            min_total_bytes: self.dit_bytes(),
        }
    }

    /// On-disk bytes of the DiT, the dominant resident weights: the
    /// `*.safetensors` under `<weights>/transformer` (LTX: the `dit` file
    /// when it is one). `None` when there is nothing to measure. A bf16
    /// checkpoint over-estimates an MXFP8/FP8 residency, which is the safe
    /// side for the co-residency check.
    pub fn dit_bytes(&self) -> Option<u64> {
        let dir = match &self.recipe {
            CudaRecipe::Ltx2(r) if r.dit.is_file() => {
                return r.dit.metadata().ok().map(|m| m.len())
            }
            CudaRecipe::Ltx2(r) if r.a2v == LtxA2v::Guided => r.dit.clone(),
            CudaRecipe::Ltx2(r) => r.dit.join("transformer"),
            CudaRecipe::H3(r) if r.ref2va => r
                .ref_weights
                .as_deref()
                .unwrap_or(&r.weights)
                .join("transformer_ref"),
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

    /// `ModelCaps`: the variant's `caps` and `serving` tables with this
    /// model's id, names, residency, tier and recipe name, and the two
    /// recipe-dependent parts: the H3 1080P tier is offered only while the
    /// recipe has it on (`h3_1080p = false`, [`gate_h3_1080p`]), and the
    /// streaming shape follows the recipe (an SF-Wan rollout's blocks, KV
    /// window and LongLive re-cache; otherwise a clip bounded by the frame
    /// grid at the default fps).
    pub fn caps(&self) -> ModelCaps {
        let mut c = self.base_caps.clone();
        c.id = self.id.clone();
        c.served_names = self.served_names.clone();
        c.resident = self.resident;
        c.recipe = Some(self.recipe_name.clone());
        c.tier = self.tier;
        if let CudaRecipe::H3(r) = &self.recipe {
            if !r.hd_1080p {
                if let Some(hd) = c.canvas.hd.take() {
                    c.canvas.short_edges.retain(|&s| s != hd.short_edge);
                }
            }
        }
        c.stream = Some(match &self.recipe {
            CudaRecipe::SfWan(r) => StreamCaps::Causal {
                block_frames: r.block_frames,
                target_fps: c.fps.default,
                context: Some(fastvideo_protocol::CausalContext {
                    window_latent_frames: r.local_attn_frames,
                    sink_latent_frames: r.sink_frames,
                    prompt_recache: r.longlive.as_ref().is_some_and(|l| l.recache),
                }),
            },
            _ => StreamCaps::Clip {
                min_s: c.frames.min as f32 / c.fps.default as f32,
                max_s: c.frames.max as f32 / c.fps.default as f32,
            },
        });
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
                    _ if r.taeh3.is_some() && self.tier != Some(Tier::Draft) => format!(
                        "H3 fast preview: {} decoded by TAEH3 instead of the official VAE (explicit choice, never \
                         a default; fails the sharpness gate, 0.82-0.91 against the official decoder; about 10-12% \
                         faster end to end)",
                        preview_base_name(&r.recipe)
                    ),
                    _ if h3cfg::is_trim_nvfp4_fp4attn_recipe(&r.recipe) => "FastH3 Trim 8-step NVFP4, fastest \
                        (opt-in): the release's NVFP4 DiT (42 blocks, every attention and FFN linear W4A4) with the \
                        VSA fine stage on sparse FP4 tensor cores (sm_120 kernel), lightweight VAE; a different model \
                        from FastH3 V2 and not gate-equivalent (9.2 / 24.6 s denoise at 480p / 768p on an RTX PRO 6000)"
                        .to_owned(),
                    _ if h3cfg::is_nvfp4_release_recipe(&r.recipe) => format!(
                        "{}: the release's ModelOpt NVFP4 DiT (FFN W4A4 on NVFP4 tensor cores, sm_100 / sm_120 \
                         only), VSA-H3 with the fine stage on sparse FP4 tensor cores where the kernel exists, its \
                         published NVFP4 VAE decoder (FASTVIDEO_H3_VAE_NVFP4=0: h3-base's, MXFP8)",
                        preview_base_name(&r.recipe)
                    ),
                    _ if h3cfg::is_trim_nvfp4_recipe(&r.recipe) => "FastH3 Trim 8-step DMD with the release's NVFP4 \
                        DiT: 42 blocks, every attention and FFN linear W4A4 on NVFP4 tensor cores (sm_100 / sm_120 \
                        only), rank-16 AdaLN, VSA-H3 (80% sparse), lightweight VAE"
                        .to_owned(),
                    _ if self.recipe_name.starts_with("fasth3-trim") => "FastH3 Trim 8-step DMD: 42 of the \
                        50 H3 blocks, rank-16 AdaLN, VSA-H3 (80% sparse), lightweight VAE; faster and smaller \
                        than FastH3 V2, with some quality cost"
                        .to_owned(),
                    _ if turbo_lora_of(&r.recipe).is_some() => {
                        let t = turbo_lora_of(&r.recipe).unwrap_or(TurboLora::Larry);
                        format!(
                            "H3 + {} Turbo LoRA ({}, folded at load, strength {}), {} forwards, dense \
                             attention, MXFP8 linears, official VAE",
                            t.hub_repo(),
                            t.file(),
                            r.lora_strength.unwrap_or(1.0),
                            r.steps
                        )
                    }
                    _ if self.recipe_name.starts_with("vdn-h3") => format!(
                        "VDN-H3 (Video DeltaNet on MiniMax-H3): hybrid attention, a chunk-aligned window \
                         softmax plus a frame-wise delta-rule linear branch in every block, {}; \
                         {} forwards, bf16 linears, official VAE",
                        if r.vdn_shared.is_some() { "stage-B + DMD turbo LoRA" } else { "stage-B LoRA" },
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
                    WanSampler::StepDistill { timesteps, shift } => format!(
                        "{}{} step-distill Euler {}-step ({}), shift {shift}, no CFG, {} decode",
                        r.preset,
                        r.plug.as_deref().map(|p| format!(" + LongLive-Plug {p}")).unwrap_or_default(),
                        timesteps.len(),
                        timesteps.iter().map(i32::to_string).collect::<Vec<_>>().join("/"),
                        r.decoder.env_value()
                    ),
                },
            ),
            CudaRecipe::SfWan(r) => (
                r.wan.sampler.steps(),
                "causal-kv".to_owned(),
                "taehv".to_owned(),
                format!(
                    "{} causal rollout, {} Self-Forcing steps per block, KV window {} latent frames, sink {}{}",
                    if r.longlive.is_some() { "LongLive" } else { "SF-Wan" },
                    r.wan.sampler.steps(),
                    r.local_attn_frames,
                    r.sink_frames,
                    match &r.longlive {
                        Some(l) if l.recache => ", KV re-cache at prompt switches",
                        _ => "",
                    }
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

/// The public alias of the Wan 2.1 T2V-14B fast tier (a served name of
/// [`WAN14B_PLUG_RECIPE`] in `families/wan.toml`).
pub const WAN14B_TURBO: &str = "wan14b-turbo";
/// The variant id (and LongLive-Plug recipe) behind [`WAN14B_TURBO`].
pub const WAN14B_PLUG_RECIPE: &str = "wan14b-plug-4step";

/// The LTX draft profile (NVFP4 video FFN).
pub const LTX_DRAFT_PROFILE: &str = "ltx2/ltx25_distill_sol_nvfp4";

/// Turns the native 1080P tier off on every H3 model when the GPU has less
/// memory than the tier's plan (`plan_1080p_min_device_bytes`, about 73.5 GiB:
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

/// The Turbo LoRA catalog ids and their technique profiles.
pub const H3_LARRY_TURBO: &str = "h3-larry-turbo";
pub const H3_LIGHTX2V_TURBO: &str = "h3-lightx2v-turbo";
pub const LARRY_TURBO_PROFILE: &str = "h3/larry_turbo_dense";
pub const LIGHTX2V_TURBO_PROFILE: &str = "h3/lightx2v_turbo_dense";

/// The Turbo LoRA a recipe (`larry-turbo`, `lightx2v-turbo-5step`, ...) runs.
pub fn turbo_lora_of(recipe: &str) -> Option<TurboLora> {
    match parse_turbo_recipe(recipe) {
        Some(Ok((t, _))) => Some(t),
        _ => None,
    }
}

/// The FastH3 Trim catalog id, its served alias and its weight tree.
pub const FASTH3_TRIM_ID: &str = "fasth3-trim-8step";
pub const H3_TRIM: &str = "h3-trim";
pub const FASTH3_TRIM_TREE: &str = "fasth3-trim-8step";
/// The FastH3 Trim NVFP4 catalog id, its served alias and its recipe (the
/// published NVFP4 DiT in `fasth3-trim-8step-nvfp4`, read beside
/// [`FASTH3_TRIM_TREE`] by the pipeline).
pub const FASTH3_TRIM_NVFP4_ID: &str = "fasth3-trim-nvfp4";
pub const H3_TRIM_NVFP4: &str = "h3-trim-nvfp4";
pub const FASTH3_TRIM_NVFP4_RECIPE: &str = "trim-8step-nvfp4";

/// The opt-in "fastest" H3 tier (owner decision 2026-10-14): the Trim NVFP4
/// with sparse FP4 VSA turned on by its recipe
/// ([`h3cfg::is_trim_nvfp4_fp4attn_recipe`]); same trees as
/// [`FASTH3_TRIM_NVFP4_ID`].
pub const FASTH3_TRIM_NVFP4_FAST_ID: &str = "fasth3-trim-nvfp4-fast";
pub const H3_TRIM_NVFP4_FAST: &str = "h3-trim-nvfp4-fast";
pub const FASTH3_TRIM_NVFP4_FAST_RECIPE: &str = "trim-8step-nvfp4-fp4attn";
/// The NVFP4 tier (docs/ports/fasth3-nvfp4.md, docs/perf/h3-nvfp4-tier-options.md):
/// the FastH3 V2 / 4-step NVFP4 releases from their own trees (the ModelOpt
/// `transformer/` and the published NVFP4 `vae/` shards 1-2), with
/// `h3-base` for the text encoder, tokenizer, audio VAE and the `vae/`'s
/// byte-identical third shard. Catalog ids, served names, trees, recipes.
pub const FASTH3_V2_NVFP4_ID: &str = "fasth3-v2-nvfp4";
pub const H3_V2_NVFP4: &str = "h3-v2-nvfp4";
pub const FASTH3_V2_NVFP4_TREE: &str = "fasth3-v2-nvfp4";
pub const FASTH3_V2_NVFP4_RECIPE: &str = "v2-nvfp4";
pub const FASTH3_4STEP_NVFP4_ID: &str = "fasth3-4step-vsa-nvfp4";
pub const H3_TURBO_NVFP4: &str = "h3-turbo-nvfp4";
pub const FASTH3_4STEP_NVFP4_TREE: &str = "fasth3-4step-vsa-nvfp4";
pub const FASTH3_4STEP_NVFP4_RECIPE: &str = "4step-vsa-nvfp4";

/// What a recipe is called in a summary.
fn preview_base_name(recipe: &str) -> &'static str {
    match h3cfg::nvfp4_release_contract(recipe) {
        Some(c) if c.transformer_forwards == 8 => "FastH3 V2 NVFP4 (8 steps)",
        Some(_) => "FastH3 4-step VSA DataFree NVFP4",
        None => "FastH3 Preview 4-step VSA (h3-turbo)",
    }
}

/// VDN-H3's catalog id, its contract (`H3InferenceContract::named`), its
/// weight tree and the stage directory inside it (docs/ports/vdn-h3.md).
pub const VDN_H3_ID: &str = "vdn-h3";
pub const VDN_H3_RECIPE: &str = "vdn-h3-50step";
pub const VDN_H3_TREE: &str = "vdn-h3-stage-b-2000";
pub const VDN_H3_STAGE: &str = "stage-b-step-2000";
/// VDN-H3 8-step (the DMD-distilled model): its id, contract, the tree with
/// the turbo LoRA and its stage directory; the branch and default LoRA come
/// from [`VDN_H3_TREE`] (`H3Recipe::vdn_shared`).
pub const VDN_H3_8STEP_ID: &str = "vdn-h3-8step";
pub const VDN_H3_8STEP_RECIPE: &str = "vdn-h3-8step";
pub const VDN_H3_8STEP_TREE: &str = "vdn-h3-dmd-250-turbo";
pub const VDN_H3_8STEP_STAGE: &str = "stage-dmd-step-250";

/// LTX-2.5's canvas tiers, the default first (`families/ltx2.toml`; the fake
/// LTX models reuse the list).
pub const LTX_SHORT_EDGES: [u32; 6] = [1080, 480, 720, 768, 1440, 2160];

/// The LTX model config for a version.
pub fn ltx_config(v: LtxVersion) -> fastvideo_models::ltx2::config::Ltx2Config {
    match v {
        LtxVersion::V23 => fastvideo_models::ltx2::config::ltx2_23_22b_distilled(),
        LtxVersion::V25 => fastvideo_models::ltx2::config::ltx2_5_22b_distilled(),
    }
}

/// How an H3 model warms up (fast boot B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmupMode {
    Off,
    /// Inside the load: ready after the warm-up (the behaviour before fast boot B).
    Blocking,
    /// Ready after the load; warm-up while idle, yielding to jobs.
    Background,
}

/// `FV_WARMUP`: overrides the config's warm-up mode for every model.
pub const WARMUP_ENV: &str = "FV_WARMUP";

impl WarmupMode {
    /// The recipe's mode, with `FV_WARMUP` (`blocking`, `background`, `off`)
    /// applied to a model that has a warm-up; an unknown value keeps the
    /// recipe's (and says so).
    pub fn of(r: &H3Recipe) -> Self {
        Self::resolve(r, std::env::var(WARMUP_ENV).ok().as_deref())
    }

    pub fn resolve(r: &H3Recipe, env: Option<&str>) -> Self {
        // The variable picks how, never whether: a model without a
        // configured warm-up stays without one.
        if !r.warmup {
            return Self::Off;
        }
        let configured = if r.warmup_blocking {
            Self::Blocking
        } else {
            Self::Background
        };
        match env.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") => configured,
            Some("blocking" | "block" | "sync") => Self::Blocking,
            Some("background" | "bg" | "async") => Self::Background,
            Some("off" | "0" | "false" | "none") => Self::Off,
            Some(other) => {
                tracing::warn!("{WARMUP_ENV}={other}: expected blocking|background|off; keeping {configured:?}");
                configured
            }
        }
    }
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
    /// The served model id (replaces the variant id; empty: the variant id).
    pub id: String,
    /// The family-file variant: its id, one of its served names, or a tier
    /// alias (`h3-turbo`).
    pub variant: String,
    /// The model's weight directory (replaces the variant's).
    pub weights: Option<PathBuf>,
    pub resident: bool,
    /// Extra names this deployment serves the model under.
    pub served_names: Vec<String>,
    /// Optional per-model settings: `text_encoder`, `i2v_encoder`, `warmup`, `h3_1080p`
    /// (H3), `text` (LTX), `adaln_cache` (H3), `taeh3` / `tae` (tiny
    /// decoders), `steps` / `lora_strength` (Turbo LoRA), `longlive` /
    /// `longlive_recache` (LongLive).
    pub extra: BTreeMap<String, String>,
}

/// Resolves a `[[models]]` entry against the catalog ([`catalog`]: the
/// family files under `layout`).
pub fn model_from_config(layout: &WeightLayout, e: &ModelEntryCfg) -> Result<CudaModel, String> {
    let cat = catalog(layout).map_err(|err| err.to_string())?;
    if e.variant.trim().is_empty() {
        return Err(format!(
            "model `{}`: `variant` names the family-file variant (a variant id, a served name or a tier alias)",
            e.id
        ));
    }
    let mut m = find(&cat, &e.variant).ok_or_else(|| {
        format!(
            "model `{}`: variant `{}` is not in families/*.toml ({})",
            e.id,
            e.variant,
            cat.iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    // The served names, as the capabilities have always listed them: the
    // deployment's id, the variant id, the deployment's extra names, then
    // the variant's own; neighbouring repeats collapse. (A deployment id
    // that is also one of the variant's served names, `fasth3`, is listed
    // again at the end: kept for byte identity of `/fv/v1/capabilities`.)
    let variant_names: Vec<String> = m.served_names.drain(1..).collect();
    let vid = m.id.as_str().to_owned();
    let id = if e.id.is_empty() {
        vid.clone()
    } else {
        e.id.clone()
    };
    m.id = ModelId::new(&id);
    m.served_names = vec![id, vid];
    for n in &e.served_names {
        if !m.served_names.contains(n) {
            m.served_names.push(n.clone());
        }
    }
    m.served_names.extend(variant_names);
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
            if let Some(t) = x("vdn_weights").filter(|_| r.vdn_weights.is_some()) {
                r.vdn_weights = Some(t.into());
            }
            if let Some(t) = x("vdn_shared").filter(|_| r.vdn_shared.is_some()) {
                r.vdn_shared = Some(t.into());
            }
            if let Some(t) = x("i2v_encoder") {
                r.i2v_encoder = t;
            }
            if let Some(w) = x("warmup") {
                (r.warmup, r.warmup_blocking) = match w.as_str() {
                    "true" | "background" => (true, false),
                    "blocking" => (true, true),
                    "false" => (false, false),
                    other => {
                        return Err(format!(
                            "model `{}`: warmup = {other:?} (true | \"background\" | \"blocking\" | false)",
                            e.id
                        ))
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
                        return Err(format!(
                            "model `{}`: h3_1080p = {other:?} (true | false)",
                            e.id
                        ))
                    }
                };
            }
            if let Some(t) = turbo_lora_of(&r.recipe) {
                // `steps = 4` picks the step count; the adapter file follows
                // the variant (`h3-lightx2v-turbo-v01` carries the v0.1 file).
                let mut forwards = match parse_turbo_recipe(&r.recipe) {
                    Some(Ok((_, n))) => n,
                    _ => t.default_forwards(),
                };
                if let Some(n) = x("steps") {
                    forwards = n.trim().parse().map_err(|_| {
                        format!("model `{}`: steps = {n:?} is not a whole number", e.id)
                    })?;
                }
                let recipe = t.recipe(forwards);
                if let Some(Err(err)) = parse_turbo_recipe(&recipe) {
                    return Err(format!("model `{}`: {err}", e.id));
                }
                r.recipe = recipe;
                r.steps = forwards as u32;
                if let Some(s) = x("lora_strength") {
                    let v: f32 = s.trim().parse().map_err(|_| {
                        format!("model `{}`: lora_strength = {s:?} is not a number", e.id)
                    })?;
                    fastvideo_models::h3::turbo_lora::turbo_strength(Some(v), None)
                        .map_err(|err| format!("model `{}`: {err}", e.id))?;
                    r.lora_strength = Some(v);
                }
            } else if x("steps").is_some() || x("lora_strength").is_some() {
                return Err(format!(
                    "model `{}`: steps / lora_strength apply to the Turbo LoRA models (h3-larry-turbo, h3-lightx2v-turbo) only",
                    e.id
                ));
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
            // LongLive-1.3B: the `longlive-1.3b` variant carries the recipe
            // (window 12, sink 3, KV re-cache); `longlive = "<dir>"` moves its
            // converted checkpoint, `longlive_recache = false` keeps the cache
            // at a prompt switch (the ablation).
            match (&mut r.longlive, x("longlive").filter(|d| !d.is_empty())) {
                (Some(l), dir) => {
                    if let Some(d) = dir {
                        l.weights = d.into();
                    }
                    l.recache = match x("longlive_recache").as_deref() {
                        None | Some("true") => true,
                        Some("false") => false,
                        Some(other) => {
                            return Err(format!(
                                "model `{}`: longlive_recache = {other:?} (true | false)",
                                e.id
                            ))
                        }
                    };
                }
                (None, Some(_)) => {
                    return Err(format!(
                        "model `{}`: `longlive` applies to the LongLive variant (variant = \"longlive-1.3b\")",
                        e.id
                    ))
                }
                (None, None) => {
                    if x("longlive_recache").is_some() {
                        return Err(format!(
                            "model `{}`: longlive_recache applies to the LongLive variant",
                            e.id
                        ));
                    }
                }
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

/// A profile's load-time settings as the settings table holds them (none
/// without a profile): what an H3 model loads and runs under in fv-serve
/// (`settings::scope_model`).
pub fn profile_settings(
    profile: Option<&str>,
) -> Result<fastvideo_models::techniques::settings::Settings, String> {
    match profile {
        None => Ok(Default::default()),
        Some(name) => load_profile(name)?
            .settings()
            .map_err(|e| format!("profile {name}: {e}")),
    }
}

/// H3 settings read only while a model loads or runs, under that model's own
/// settings scope (fv-serve's `CudaBackend`): they may differ between the H3
/// models of one process. `FASTVIDEO_H3_QUANT` picks the DiT's linears
/// (MXFP8 for the MXFP8 profiles; unset, the device's default or the
/// recipe's: VDN-H3's bf16, an NVFP4 release's own path).
const H3_PER_MODEL_SETTINGS: &[&str] = &["FASTVIDEO_H3_QUANT"];

/// What one process must agree on for a model: its load-time settings, for H3
/// without the per-model ones and with `FASTVIDEO_BF16_ACT` unset read as the
/// H3 pipeline's default (`1`: `default_bf16_activations`), since that flag is
/// cached process-wide on its first read.
fn process_key(m: &CudaModel, settings: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut k = settings.clone();
    if matches!(m.recipe, CudaRecipe::H3(_)) {
        for s in H3_PER_MODEL_SETTINGS {
            k.remove(*s);
        }
        k.entry("FASTVIDEO_BF16_ACT".to_owned())
            .or_insert_with(|| "1".to_owned());
    }
    k
}

impl ProcessPlan {
    /// Checks that `models` can share one process and returns what to install.
    ///
    /// H3 models may differ in their per-model settings
    /// (`H3_PER_MODEL_SETTINGS`): fv-serve loads and runs each under its own
    /// (`settings::scope_model`), so e.g. an MXFP8 profile's model, an NVFP4
    /// release and VDN-H3 share one process (one resident at a time with
    /// `[engine] swap`, the family pools). Every other setting and the
    /// process env must agree.
    pub fn for_models(models: &[CudaModel]) -> Result<Self, String> {
        let mut plan = ProcessPlan::default();
        let mut first: Option<(&str, BTreeMap<String, String>)> = None;
        for m in models {
            let raw = settings_of(m.recipe.profile())?;
            let settings = process_key(m, &raw);
            if let CudaRecipe::H3(r) = &m.recipe {
                if let Some(name) = &r.profile {
                    let p = load_profile(name)?;
                    if let Some(want) = &p.recipe {
                        // A Turbo LoRA profile serves every step count and
                        // adapter file of its family (`steps`, the v0.1 recipe):
                        // its techniques do not depend on the ladder.
                        let turbo =
                            turbo_lora_of(want).is_some() && turbo_lora_of(&r.recipe).is_some();
                        if !turbo
                            && h3cfg::H3InferenceContract::named(want).ok()
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
                    plan.settings = raw;
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
                    // The first profile whose settings are the installed ones
                    // (an H3 model of another profile runs under its own).
                    if plan.profile.is_none() && raw == plan.settings {
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
    use crate::caps::{CapabilityTable, SOL_H3_4STEP_PROFILE};
    use fastvideo_protocol::{FrameGrid, KnobCaps, Task};

    fn table(models: &[CudaModel]) -> CapabilityTable {
        CapabilityTable::build(
            vec![models.iter().map(|m| (m.caps(), m.describe())).collect()],
            &BTreeMap::new(),
        )
        .unwrap()
    }

    #[test]
    fn every_tier_alias_is_bound() {
        let cat = catalog(&WeightLayout::default()).unwrap();
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
        assert!(t
            .get(&ModelId::new("fasth3-8step-dense"))
            .unwrap()
            .tier
            .is_none());
        assert_eq!(t.len(), cat.len());
    }

    #[test]
    fn sage_attention_is_ltx_pro_only() {
        let cat = catalog(&WeightLayout::default()).unwrap();
        let on: Vec<&str> = cat
            .iter()
            .filter(|m| matches!(&m.recipe, CudaRecipe::Ltx2(r) if r.sage_attention))
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(on, ["ltx25-distill-dense"]);
        assert_eq!(
            find(&cat, "ltx-pro").unwrap().id.as_str(),
            "ltx25-distill-dense"
        );
    }

    #[test]
    fn recipes_and_profiles() {
        let cat = catalog(&WeightLayout::default()).unwrap();
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
                        // A Turbo LoRA profile serves every step count and
                        // adapter file of its family (`ProcessPlan::for_models`):
                        // `h3-lightx2v-turbo-v01` runs the v1.2 profile.
                        if turbo_lora_of(&want).is_some() && turbo_lora_of(&h.recipe).is_some() {
                            continue;
                        }
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
        let cat = catalog(&WeightLayout::default()).unwrap();
        let get = |id: &str| cat.iter().find(|m| m.id.as_str() == id).unwrap().caps();
        let h3 = get("fasth3-4step-vsa");
        assert_eq!(h3.audio.as_ref().unwrap().native_rate, 32_000);
        assert_eq!(h3.audio.as_ref().unwrap().channels, 2);
        assert_eq!(h3.canvas.short_edges, vec![768, 480, 1080]);
        assert_eq!(h3.canvas.hd, Some(fastvideo_protocol::HdTier::h3_1080p()));
        // The 1080 tier does not move the budgets of the others.
        assert_eq!(h3.canvas.area_at(768), 768 * 1344);
        assert_eq!(h3.canvas.area_at(1080), 1088 * 1920);
        assert_eq!(
            fastvideo_protocol::canvas_for_aspect(&h3.canvas, 16.0 / 9.0, 480),
            (832, 480)
        );
        assert_eq!(
            fastvideo_protocol::canvas_for_aspect(&h3.canvas, 16.0 / 9.0, 768),
            (1344, 768)
        );
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
        for id in [
            "wan22-ti2v-5b",
            "fastwan22-ti2v-5b",
            "fastwan22-ti2v-5b-taehv",
        ] {
            let c = get(id);
            assert_eq!(c.canvas.short_edges, vec![704, 576, 480], "{id}");
            assert_eq!(
                (c.frames.min, c.frames.max, c.frames.default),
                (9, 161, 121),
                "{id}"
            );
            assert!(c.frames.contains(17) && c.frames.contains(161));
            assert!(
                (4..=60).all(|f| c.fps.allows(f)) && !c.fps.allows(3) && !c.fps.allows(61),
                "{id}"
            );
            assert!(c.supports(Task::I2V), "{id}");
            let mut req = fastvideo_protocol::GenerationRequest::text(
                fastvideo_protocol::ProtocolId::Fal,
                id,
                "a cat",
            );
            req.canvas = fastvideo_protocol::CanvasSpec::Aspect {
                ratio: fastvideo_protocol::Ratio::R16_9,
                short_edge: 576,
            };
            req.timing = fastvideo_protocol::TimingSpec {
                length: fastvideo_protocol::Length::Frames {
                    value: 161,
                    snap: fastvideo_protocol::Snap::AlignUp,
                },
                fps: Some(60),
            };
            let r = fastvideo_protocol::negotiate(&req, &c, &Default::default()).unwrap();
            assert_eq!(
                (r.width, r.height, r.num_frames, r.fps),
                (1024, 576, 161, 60),
                "{id}"
            );
        }
        let fw = get("fastwan22-ti2v-5b");
        assert!(!fw.knobs.negative && !fw.knobs.guidance && fw.knobs.steps && fw.knobs.flow_shift);
        assert_eq!(fw.tier, Some(Tier::Turbo));
        let sf = get("sfwan21-1.3b");
        assert!(matches!(
            sf.stream,
            Some(StreamCaps::Causal {
                block_frames: 12,
                target_fps: 16,
                context: Some(fastvideo_protocol::CausalContext {
                    prompt_recache: false,
                    ..
                })
            })
        ));
    }

    #[test]
    fn wan14b_turbo_is_the_plug_recipe() {
        use fastvideo_protocol::*;
        let layout = WeightLayout::default();
        let cat = catalog(&layout).unwrap();
        let m = find(&cat, WAN14B_TURBO).unwrap();
        assert_eq!(m.id.as_str(), WAN14B_PLUG_RECIPE);
        assert_eq!(find(&cat, "wan14b-plug-4step").unwrap().id, m.id);
        // Untiered: the Wan turbo slot stays the 5B (fal's fast-wan app).
        assert_eq!(m.tier, None);
        assert_eq!(
            find(&cat, "wan-turbo").unwrap().id.as_str(),
            "fastwan22-ti2v-5b"
        );
        let CudaRecipe::Wan(r) = &m.recipe else {
            panic!("{:?}", m.recipe)
        };
        assert_eq!(r.preset, "wan_t2v_14b");
        assert_eq!(r.weights, layout.root.join("wan21-t2v-14b"));
        assert_eq!(r.plug.as_deref(), Some("wan14b-plug-4step"));
        let plug = fastvideo_models::plug::PlugRecipe::named(r.plug.as_deref().unwrap()).unwrap();
        assert_eq!(
            (plug.preset, plug.base),
            (r.preset.as_str(), "wan21-t2v-14b")
        );
        assert_eq!(plug.licence, "Apache-2.0");
        match (&r.sampler, &plug.sampler) {
            (
                WanSampler::StepDistill { timesteps, shift },
                fastvideo_models::plug::PlugSampler::WanEuler {
                    timesteps: want,
                    shift: s,
                },
            ) => assert_eq!((timesteps.as_slice(), *shift), (*want, *s)),
            other => panic!("{other:?}"),
        }
        let t = table(&cat);
        assert_eq!(
            t.resolve(WAN14B_TURBO).unwrap().id.as_str(),
            WAN14B_PLUG_RECIPE
        );
        let d = m.describe();
        assert_eq!(
            (d.steps, d.attention.as_str(), d.vae.as_str(), d.profile),
            (Some(4), "dense", "full", None)
        );
        assert!(
            d.summary.contains("wan14b-plug-4step") && d.summary.contains("1000/750/500/250"),
            "{}",
            d.summary
        );
        let c = m.caps();
        assert!(c.served_names.iter().any(|n| n == WAN14B_TURBO));
        assert_eq!(c.tasks, [Task::T2V].into_iter().collect());
        assert_eq!(c.canvas.short_edges, vec![480]);
        assert_eq!(
            (c.frames.min, c.frames.max, c.frames.default, c.fps.default),
            (9, 81, 81, 16)
        );
        assert_eq!(
            c.knobs,
            KnobCaps {
                seed: true,
                ..KnobCaps::default()
            }
        );
        assert!(!m.requirements().fp8 && !m.requirements().nvfp4);
        for (ratio, want) in [(Ratio::R16_9, (832, 480)), (Ratio::R9_16, (480, 832))] {
            let mut req = GenerationRequest::text(ProtocolId::Native, WAN14B_TURBO, "a cat");
            req.canvas = CanvasSpec::Aspect {
                ratio,
                short_edge: 480,
            };
            let j = negotiate(&req, &c, &Default::default()).unwrap();
            assert_eq!(
                (j.width, j.height, j.num_frames, j.fps),
                (want.0, want.1, 81, 16),
                "{ratio:?}"
            );
        }
        // 14B and the 5B tiers share the process settings (no VSA).
        assert!(ProcessPlan::for_models(&[m.clone(), find(&cat, "wan-turbo").unwrap()]).is_ok());
        // `[[models]] recipe = "wan14b-turbo"` binds it.
        let e = ModelEntryCfg {
            id: "wan14b".into(),
            variant: WAN14B_TURBO.into(),
            ..Default::default()
        };
        let got = model_from_config(&layout, &e).unwrap();
        assert!(matches!(&got.recipe, CudaRecipe::Wan(w) if w.plug.is_some()));
        assert!(got.served_names.iter().any(|n| n == WAN14B_TURBO));
    }

    #[test]
    fn negotiate_accepts_the_defaults() {
        use fastvideo_protocol::*;
        let cat = catalog(&WeightLayout::default()).unwrap();
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
                    probe: MediaProbe {
                        duration_s: Some(7.0),
                        audio_rate: Some(44_100),
                        ..Default::default()
                    },
                });
                let r =
                    negotiate(&req, &caps, &staged).unwrap_or_else(|e| panic!("{}: {e:?}", m.id));
                assert_eq!(
                    (r.num_frames, r.fps, r.sampling.guidance),
                    (161, 24, Some(5.0)),
                    "{}",
                    m.id
                );
                assert_eq!(r.recipe.as_deref(), Some(m.recipe_name.as_str()));
                continue;
            }
            if !caps.supports(Task::T2V) {
                // Ref2VA models serve reference-to-video only.
                assert_eq!(
                    caps.tasks.iter().copied().collect::<Vec<_>>(),
                    vec![Task::Ref2V]
                );
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
                        probe: MediaProbe {
                            width: Some(1024),
                            height: Some(1024),
                            ..Default::default()
                        },
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
        let cat = catalog(&WeightLayout::default()).unwrap();
        let get = |id: &str| cat.iter().find(|m| m.id.as_str() == id).unwrap().clone();
        for (id, tier, recipe, steps) in [
            ("h3-ref2v-max", Tier::Max, "base", 49),
            ("h3-ref2v-turbo", Tier::Turbo, "sol-h3-ref2va", 4),
        ] {
            let m = get(id);
            let CudaRecipe::H3(r) = &m.recipe else {
                panic!("{id}")
            };
            assert!(r.ref2va && r.dense, "{id}");
            assert_eq!((r.recipe.as_str(), r.steps), (recipe, steps), "{id}");
            assert!(
                r.ref_weights.as_ref().unwrap().ends_with("h3-ref2va"),
                "{id}"
            );
            assert!(r.weights.ends_with("h3-base"), "{id}");
            let c = m.caps();
            assert!(c.supports(Task::Ref2V) && !c.supports(Task::T2V) && !c.supports(Task::I2V));
            assert_eq!(c.refs, fastvideo_protocol::RefLimits::h3());
            assert_eq!(c.tier, Some(tier));
            let d = m.describe();
            assert_eq!(
                (d.attention.as_str(), d.steps),
                ("dense", Some(steps)),
                "{id}"
            );
        }
        // The tier aliases still name the base models; Ref2V requests on them
        // route to the companion, other tasks stay.
        let t = table(&cat);
        assert_eq!(t.tier(Family::H3, Tier::Max).unwrap().as_str(), "sol-h3");
        assert_eq!(
            t.tier(Family::H3, Tier::Turbo).unwrap().as_str(),
            "fasth3-4step-vsa"
        );
        let models: Vec<&ModelCaps> = t.models().collect();
        for (alias, want) in [("h3-max", "h3-ref2v-max"), ("h3-turbo", "h3-ref2v-turbo")] {
            let base = t.resolve(alias).unwrap();
            assert_eq!(
                fastvideo_protocol::route_task(base, Task::Ref2V, models.iter().copied())
                    .id
                    .as_str(),
                want
            );
            assert_eq!(
                fastvideo_protocol::route_task(base, Task::I2V, models.iter().copied()).id,
                base.id
            );
            assert_eq!(
                t.get(&ModelId::new(want)).unwrap().tier,
                base.tier,
                "{want} keeps its tier tag"
            );
        }
        let tier_max =
            fastvideo_protocol::resolve_tier(Family::H3, Tier::Max, models.iter().copied())
                .unwrap();
        assert_eq!(tier_max.id.as_str(), "sol-h3");
    }

    #[test]
    fn ltx_guided_a2v_is_the_task_companion_of_ltx_pro() {
        use fastvideo_protocol::*;
        let cat = catalog(&WeightLayout::default()).unwrap();
        let m = cat
            .iter()
            .find(|m| m.id.as_str() == "ltx25-a2v-guided")
            .unwrap()
            .clone();
        let CudaRecipe::Ltx2(r) = &m.recipe else {
            panic!("ltx25-a2v-guided")
        };
        assert_eq!(
            (r.a2v, r.stage2, r.two_stage),
            (LtxA2v::Guided, LtxStage2::Dense, true)
        );
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
        assert_eq!(
            route_task(base, Task::A2V, models.iter().copied())
                .id
                .as_str(),
            "ltx25-a2v-guided"
        );
        assert_eq!(
            route_task(base, Task::I2V, models.iter().copied()).id,
            base.id
        );
        let turbo = t.resolve("ltx-turbo").unwrap();
        assert!(turbo.supports(Task::A2V));
        assert_eq!(
            route_task(turbo, Task::A2V, models.iter().copied()).id,
            turbo.id
        );
        // A `[[models]]` weight root moves the dev DiT beside it; `dit` names it.
        let e = ModelEntryCfg {
            id: "ltx25-a2v-guided".into(),
            variant: "ltx25-a2v-guided".into(),
            weights: Some(PathBuf::from("/w/ltx25")),
            resident: true,
            served_names: Vec::new(),
            extra: BTreeMap::new(),
        };
        let recipe = |e: &ModelEntryCfg| match model_from_config(&WeightLayout::default(), e)
            .unwrap()
            .recipe
        {
            CudaRecipe::Ltx2(r) => r,
            _ => panic!("not ltx2"),
        };
        let r = recipe(&e);
        assert_eq!(r.dit, PathBuf::from("/w/ltx25-dev/transformer_full"));
        let r2 = recipe(&ModelEntryCfg {
            extra: [("dit".to_owned(), "/x/dev".to_owned())]
                .into_iter()
                .collect(),
            ..e.clone()
        });
        assert_eq!(r2.dit, PathBuf::from("/x/dev"));
        assert_eq!(r.weights, PathBuf::from("/w/ltx25"));
    }

    #[test]
    fn ltx_ref2v_is_the_task_companion_of_ltx_pro() {
        use fastvideo_protocol::*;
        let cat = catalog(&WeightLayout::default()).unwrap();
        let m = cat
            .iter()
            .find(|m| m.id.as_str() == "ltx25-ref2v")
            .unwrap()
            .clone();
        let CudaRecipe::Ltx2(r) = &m.recipe else {
            panic!("ltx25-ref2v")
        };
        assert_eq!(r.stage2, LtxStage2::Dense);
        assert!(r.two_stage);
        assert!(r.ic_lora.as_ref().unwrap().ends_with(
            "ltx25-ic-lora-ingredients/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors"
        ));
        let c = m.caps();
        assert_eq!(
            c.tasks.iter().copied().collect::<Vec<_>>(),
            vec![Task::Ref2V]
        );
        assert_eq!(c.refs, RefLimits::ltx_ingredients());
        assert!(c.knobs.seed && c.knobs.reference_strength && !c.knobs.steps);
        assert_eq!(c.canvas.short_edges[0], 896);
        assert_eq!((c.frames.max, c.frames.default), (241, 121));
        assert_eq!(c.tier, Some(Tier::Max));
        assert!(m.describe().summary.starts_with("Reference-to-video"));
        // No other LTX model takes the reference knobs.
        for other in cat
            .iter()
            .filter(|o| o.family() == Family::Ltx2 && o.id != m.id)
        {
            assert!(!other.caps().knobs.reference_strength, "{}", other.id);
            assert!(!other.caps().supports(Task::Ref2V), "{}", other.id);
        }
        // `ltx-pro` still names the plain model; its Ref2V requests route here.
        let t = table(&cat);
        assert_eq!(
            t.tier(Family::Ltx2, Tier::Max).unwrap().as_str(),
            "ltx25-distill-dense"
        );
        let models: Vec<&ModelCaps> = t.models().collect();
        let base = t.resolve("ltx-pro").unwrap();
        assert_eq!(
            route_task(base, Task::Ref2V, models.iter().copied())
                .id
                .as_str(),
            "ltx25-ref2v"
        );
        assert_eq!(
            route_task(base, Task::T2V, models.iter().copied()).id,
            base.id
        );
        // fal `ingredient`'s default: the sheet at 1536x896, 121 frames, strengths.
        let mut req = GenerationRequest::text(
            ProtocolId::Fal,
            "ltx-pro",
            "Reference sheet: a crab. Generated video: it walks.",
        );
        req.task = Task::Ref2V;
        req.canvas = CanvasSpec::Exact {
            width: 1536,
            height: 896,
        };
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
                    probe: MediaProbe {
                        width: Some(1536),
                        height: Some(896),
                        ..Default::default()
                    },
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
        assert_eq!(
            negotiate(&hot, &c, &staged).unwrap_err().param.as_deref(),
            Some("reference_strength")
        );
        let plain = get_caps(&cat, "ltx25-distill-dense");
        let mut t2v = GenerationRequest::text(ProtocolId::Fal, "ltx-pro", "a cat");
        t2v.sampling.reference_lora_strength = Some(1.0);
        assert_eq!(
            negotiate(&t2v, &plain, &StagedInputs::default())
                .unwrap_err()
                .param
                .as_deref(),
            Some("reference_lora_strength")
        );
    }

    /// Image-to-video (and H3 reference-to-video) with no size or aspect:
    /// every served model derives its canvas from the image at its default
    /// tier, snapped to its multiple and budget, clamped to its aspect range.
    #[test]
    fn image_conditioned_canvas_follows_the_image_on_every_model() {
        use fastvideo_protocol::*;
        let cat = catalog(&WeightLayout::default()).unwrap();
        let staged_image = |w: u32, h: u32| StagedMedia {
            path: "/stage/i.png".into(),
            mime: "image/png".into(),
            bytes: 1,
            probe: MediaProbe {
                width: Some(w),
                height: Some(h),
                ..Default::default()
            },
        };
        let images = [
            ("landscape", 1920, 1080),
            ("portrait", 1080, 1920),
            ("square", 1024, 1024),
            ("4:5", 1080, 1350),
            ("extreme", 6000, 500),
        ];
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
            let mut row = format!(
                "{:36} {:6}",
                m.id.as_str(),
                if task == Task::I2V { "i2v" } else { "ref2v" }
            );
            for (name, w, h) in images {
                let mut req = GenerationRequest::text(ProtocolId::Native, m.id.as_str(), "a cat");
                req.task = task;
                let media = MediaRef::parse("https://e.x/i.png", "image_url").unwrap();
                let mut staged = StagedInputs::default();
                if task == Task::I2V {
                    req.keyframes = vec![Keyframe {
                        at: Anchor::First,
                        image: media,
                    }];
                    staged.keyframes = vec![(Anchor::First, staged_image(w, h))];
                } else {
                    req.references = vec![Reference {
                        kind: MediaKind::Image,
                        media,
                    }];
                    staged.references = vec![(MediaKind::Image, staged_image(w, h))];
                }
                let (j, notes) = negotiate_noted(&req, &caps, &staged)
                    .unwrap_or_else(|e| panic!("{} {name}: {e:?}", m.id));
                let (ow, oh) = j.output_size();
                row.push_str(&format!(" {name}={ow}x{oh}"));
                let c = &caps.canvas;
                assert_eq!(
                    (j.width % c.multiple, j.height % c.multiple),
                    (0, 0),
                    "{} {name}",
                    m.id
                );
                if m.family() == Family::Ltx2 && task == Task::Ref2V {
                    // The reference sheet does not set the canvas: 16:9.
                    assert!(notes.is_empty(), "{} {name}", m.id);
                    assert_eq!((ow, oh), (1592, 896), "{} {name}", m.id);
                    continue;
                }
                let want = (f64::from(w) / f64::from(h)).clamp(0.25, 4.0);
                let got = f64::from(ow) / f64::from(oh);
                // Within one snap step of the image's (clamped) aspect.
                assert!(
                    (got / want).ln().abs() < 0.08,
                    "{} {name}: {ow}x{oh} vs {want}",
                    m.id
                );
                assert_eq!(
                    ow.cmp(&oh),
                    want.total_cmp(&1.0),
                    "{} {name}: {ow}x{oh}",
                    m.id
                );
                let tier = c.short_edges[0];
                assert!(
                    ow.min(oh) <= tier.max(1088)
                        && u64::from(j.width) * u64::from(j.height)
                            <= c.max_area.max(c.hd.map_or(0, |t| t.max_area)),
                    "{} {name}",
                    m.id
                );
                assert_eq!(notes.len(), 1, "{} {name}", m.id);
                assert_eq!(
                    notes[0].contains("clamped"),
                    name == "extreme",
                    "{} {name}: {notes:?}",
                    m.id
                );
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
        let cat = catalog(&WeightLayout::default()).unwrap();
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
        // bf16 activations).
        ProcessPlan::for_models(&pick(&["h3-max", "h3-turbo"])).unwrap();
        // The 8-step dense recipe has no profile (no MXFP8 setting): it loads
        // under its own settings scope, so it shares the process too; the
        // installed profile stays the first model's.
        let p = ProcessPlan::for_models(&pick(&["fasth3-8step-dense", "h3-turbo"])).unwrap();
        assert_eq!((p.profile, p.settings.is_empty()), (None, true));
        let p = ProcessPlan::for_models(&pick(&["h3-turbo", "fasth3-8step-dense"])).unwrap();
        assert_eq!(p.profile.as_deref(), Some("h3/fasth3_4step_vsa"));
        // NVFP4 is load-time: the LTX draft needs its own process.
        assert!(ProcessPlan::for_models(&pick(&["ltx-turbo", "ltx-draft"])).is_err());
        // FastWan's VSA flag is process-wide; the TI2V-5B recipe runs without it.
        let p =
            ProcessPlan::for_models(&pick(&["fastwan21-1.3b", "fastwan21-1.3b-taehv"])).unwrap();
        assert_eq!(p.env.get("FASTVIDEO_VSA").map(String::as_str), Some("1"));
        assert!(ProcessPlan::for_models(&pick(&["fastwan21-1.3b", "wan-max"])).is_err());
        // The 5B tiers (FastWan2.2 FullAttn has no VSA) share one process.
        let p = ProcessPlan::for_models(&pick(&["wan-max", "wan-turbo", "wan-draft"])).unwrap();
        assert_eq!(p.env.get("FASTVIDEO_VSA").map(String::as_str), Some("0"));
        let p = ProcessPlan::for_models(&pick(&["wan-max", "sfwan21-1.3b"])).unwrap();
        assert_eq!(p.env.get("FASTVIDEO_VSA").map(String::as_str), Some("0"));
    }

    /// Every H3 model of the catalog shares one process (the h3 family pool,
    /// configs/serve/runpod-family-h3.toml): MXFP8 profiles, the NVFP4
    /// releases, Trim and VDN-H3 differ only in `FASTVIDEO_H3_QUANT`, which
    /// each model reads under its own settings scope. A setting cached
    /// process-wide still has to agree.
    #[test]
    fn h3_models_of_different_profiles_share_a_process() {
        let cat = catalog(&WeightLayout::default()).unwrap();
        let h3: Vec<CudaModel> = cat
            .iter()
            .filter(|m| matches!(m.recipe, CudaRecipe::H3(_)))
            .cloned()
            .collect();
        assert!(h3.len() >= 15, "{}", h3.len());
        let p = ProcessPlan::for_models(&h3).unwrap_or_else(|e| panic!("{e}"));
        assert!(p.env.is_empty());
        // Order does not matter, and the first model's profile is installed.
        let mut rev = h3.clone();
        rev.reverse();
        let q = ProcessPlan::for_models(&rev).unwrap();
        assert_eq!(q.profile.as_deref(), rev[0].recipe.profile());
        // Per model: the scoped settings are the model's own profile's.
        let turbo = find(&cat, "h3-turbo").unwrap();
        let own = profile_settings(turbo.recipe.profile()).unwrap();
        assert_eq!(own.get("FASTVIDEO_H3_QUANT"), Some("mxfp8"));
        assert!(profile_settings(None).unwrap().is_empty());
        // An explicit f32-activation profile still needs its own process.
        let key = |m: &CudaModel, kv: &[(&str, &str)]| {
            process_key(
                m,
                &kv.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            )
        };
        assert_eq!(
            key(&turbo, &[]),
            key(
                &turbo,
                &[("FASTVIDEO_BF16_ACT", "1"), ("FASTVIDEO_H3_QUANT", "mxfp8")]
            )
        );
        assert_ne!(
            key(&turbo, &[]),
            key(&turbo, &[("FASTVIDEO_BF16_ACT", "0")])
        );
        assert_ne!(
            key(&turbo, &[]),
            key(&turbo, &[("FASTVIDEO_DIT_OFFLOAD", "streamed")])
        );
        // Not for other families: Wan / LTX settings stay strict.
        let wan = find(&cat, "wan-max").unwrap();
        assert_eq!(
            key(&wan, &[("FASTVIDEO_H3_QUANT", "x")])
                .get("FASTVIDEO_H3_QUANT")
                .map(String::as_str),
            Some("x")
        );
        assert!(key(&wan, &[]).is_empty());
    }

    /// The Turbo LoRA models: untiered, base H3 + their adapter, dense, T2V
    /// and FL2VA (first / last frame), `steps` and `lora_strength` per model.
    #[test]
    fn turbo_lora_models() {
        let l = WeightLayout::new("/w");
        let cat = catalog(&l).unwrap();
        for (id, recipe, steps, profile) in [
            (H3_LARRY_TURBO, "larry-turbo", 6, LARRY_TURBO_PROFILE),
            (
                H3_LIGHTX2V_TURBO,
                "lightx2v-turbo",
                4,
                LIGHTX2V_TURBO_PROFILE,
            ),
        ] {
            let m = find(&cat, id).unwrap();
            assert_eq!(m.tier, None, "{id}");
            let CudaRecipe::H3(r) = &m.recipe else {
                panic!("{id}")
            };
            assert_eq!(
                (r.recipe.as_str(), r.steps, r.dense, r.ref2va),
                (recipe, steps, true, false)
            );
            assert_eq!(r.profile.as_deref(), Some(profile));
            assert!(r.weights.ends_with("h3-base"));
            assert!(
                fastvideo_models::techniques::builtin::get(profile).is_some(),
                "{profile}"
            );
            let caps = m.caps();
            assert_eq!(
                caps.tasks,
                [Task::T2V, Task::I2V, Task::Keyframes]
                    .into_iter()
                    .collect()
            );
            assert!(m.describe().summary.contains("Turbo LoRA"), "{id}");
            // Same load-time settings as h3-turbo / h3-max (MXFP8, bf16 activations).
            ProcessPlan::for_models(&[m.clone(), find(&cat, "h3-turbo").unwrap()]).unwrap();
        }
        // The tier aliases are unchanged.
        assert_eq!(
            find(&cat, "h3-turbo").unwrap().id.as_str(),
            "fasth3-4step-vsa"
        );
        let entry = |variant: &str, extra: &[(&str, &str)]| ModelEntryCfg {
            id: String::new(),
            variant: variant.into(),
            extra: extra
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        };
        let h3r = |m: CudaModel| match m.recipe {
            CudaRecipe::H3(r) => r,
            _ => panic!(),
        };
        let r = h3r(model_from_config(&l, &entry("h3-larry-turbo", &[("steps", "4")])).unwrap());
        assert_eq!((r.recipe.as_str(), r.steps), ("larry-turbo-4step", 4));
        let r = h3r(model_from_config(
            &l,
            &entry(
                "h3-larry-turbo",
                &[("steps", "5"), ("lora_strength", "1.1")],
            ),
        )
        .unwrap());
        assert_eq!(
            (r.recipe.as_str(), r.steps, r.lora_strength),
            ("larry-turbo-5step", 5, Some(1.1))
        );
        // The v0.1 adapter is its own variant (its own tree), 4 forwards.
        let r = h3r(model_from_config(&l, &entry("h3-lightx2v-turbo-v01", &[])).unwrap());
        assert_eq!((r.recipe.as_str(), r.steps), ("lightx2v-turbo-v0.1", 4));
        assert_eq!(
            find(&cat, "h3-lightx2v-turbo-v01")
                .unwrap()
                .recipe
                .profile(),
            Some(LIGHTX2V_TURBO_PROFILE)
        );
        // Another step count or the v0.1 file keeps the family's profile and process plan.
        for (rec, extra) in [
            ("h3-larry-turbo", ("steps", "4")),
            ("h3-lightx2v-turbo-v01", ("steps", "4")),
        ] {
            let m = model_from_config(&l, &entry(rec, &[extra])).unwrap();
            ProcessPlan::for_models(&[m]).unwrap_or_else(|e| panic!("{rec}: {e}"));
        }
        assert!(model_from_config(&l, &entry("h3-larry-turbo", &[("steps", "12")])).is_err());
        assert!(
            model_from_config(&l, &entry("h3-larry-turbo", &[("lora_strength", "nan")])).is_err()
        );
        assert!(model_from_config(&l, &entry("h3-turbo", &[("steps", "4")])).is_err());
    }

    #[test]
    fn serve_config_entries() {
        let l = WeightLayout::new("/w");
        // configs/serve/runpod.toml's shape.
        let e = ModelEntryCfg {
            id: "fasth3".into(),
            variant: "h3-turbo".into(),
            weights: Some("/w/h3-base".into()),
            resident: true,
            served_names: vec!["fasth3".into()],
            extra: [("text_encoder".to_owned(), "streamed".to_owned())]
                .into_iter()
                .collect(),
        };
        let m = model_from_config(&l, &e).unwrap();
        assert_eq!(m.id.as_str(), "fasth3");
        assert_eq!(m.tier, Some(Tier::Turbo));
        let CudaRecipe::H3(r) = &m.recipe else {
            panic!()
        };
        assert_eq!(
            (r.weights.as_path(), r.text_encoder.as_str()),
            (Path::new("/w/h3-base"), "streamed")
        );
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
        assert_eq!(
            h3(&with("i2v_encoder", "stream")).unwrap().i2v_encoder,
            "stream"
        );
        assert!(h3(&with("warmup", "true")).unwrap().warmup);
        assert!(!h3(&with("warmup", "false")).unwrap().warmup);
        assert!(h3(&with("warmup", "yes")).is_err());
        // Fast boot B: `true` warms up in the background; "blocking" is the
        // old ready-after-warm-up behaviour; FV_WARMUP overrides either.
        let bg = h3(&with("warmup", "true")).unwrap();
        let blocking = h3(&with("warmup", "blocking")).unwrap();
        assert_eq!(WarmupMode::resolve(&bg, None), WarmupMode::Background);
        assert_eq!(
            WarmupMode::resolve(&h3(&with("warmup", "background")).unwrap(), None),
            WarmupMode::Background
        );
        assert_eq!(WarmupMode::resolve(&blocking, None), WarmupMode::Blocking);
        assert_eq!(
            WarmupMode::resolve(&bg, Some("blocking")),
            WarmupMode::Blocking
        );
        assert_eq!(
            WarmupMode::resolve(&blocking, Some("background")),
            WarmupMode::Background
        );
        assert_eq!(WarmupMode::resolve(&bg, Some("off")), WarmupMode::Off);
        assert_eq!(
            WarmupMode::resolve(&bg, Some("bogus")),
            WarmupMode::Background
        );
        assert_eq!(
            WarmupMode::resolve(r, Some("background")),
            WarmupMode::Off,
            "no warm-up configured stays off"
        );
        assert!(r.hd_1080p);
        assert!(!h3(&with("h3_1080p", "false")).unwrap().hd_1080p);
        assert!(h3(&with("h3_1080p", "true")).unwrap().hd_1080p);
        assert!(h3(&with("h3_1080p", "on")).is_err());
        let e8 = ModelEntryCfg {
            variant: "fasth3-8step-dense".into(),
            ..with("h3_1080p", "true")
        };
        assert!(model_from_config(&l, &e8)
            .unwrap_err()
            .contains("h3-max and h3-turbo"));
        // The memory gate: 96 GB and 80 GB cards keep the tier, 48 GB ones not.
        let gib = |g: f64| (g * f64::from(1u32 << 30)) as u64;
        for (total, keep) in [
            (Some(gib(95.6)), true),
            (Some(gib(79.6)), true),
            (Some(gib(44.4)), false),
            (None, false),
        ] {
            let mut ms = vec![
                m.clone(),
                model_from_config(&l, &with("h3_1080p", "false")).unwrap(),
            ];
            let off = gate_h3_1080p(&mut ms, total);
            assert_eq!(off.is_none(), keep, "{total:?}");
            assert_eq!(ms[0].caps().canvas.short_edges.contains(&1080), keep);
            if let Some(o) = off {
                assert!(o.contains("fasth3") && o.contains("memory plan"), "{o}");
            }
        }
        let t =
            CapabilityTable::build(vec![vec![(m.caps(), m.describe())]], &BTreeMap::new()).unwrap();
        assert_eq!(t.resolve("h3-turbo").unwrap().id.as_str(), "fasth3");
        for (variant, want) in [
            ("ltx-turbo", "ltx25-distill-sol"),
            ("wan-turbo", "fastwan22-ti2v-5b"),
            ("fastwan21-1.3b", "fastwan21-1.3b"),
            ("fasth3-8step-dense", "fasth3-8step-dense"),
            ("sol-h3", "sol-h3"),
            ("h3-max", "sol-h3"),
            ("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", "fastwan21-1.3b"),
        ] {
            let e = ModelEntryCfg {
                variant: variant.into(),
                resident: true,
                ..Default::default()
            };
            assert_eq!(model_from_config(&l, &e).unwrap().id.as_str(), want);
        }
        // Not a variant, a served name or a tier alias: refused with the list.
        for bad in ["", "4step-vsa", "8step", "nope"] {
            let e = ModelEntryCfg {
                variant: bad.into(),
                ..Default::default()
            };
            assert!(model_from_config(&l, &e).is_err(), "{bad}");
        }
    }

    #[test]
    fn layout_paths() {
        let l = WeightLayout::new("/w").with_tae_dir("/t");
        let cat = catalog(&l).unwrap();
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
        assert_eq!(
            (r.weights.as_path(), r.recipe.as_str()),
            (Path::new("/w/h3-base"), "sol-h3")
        );
        assert!(!r.dense);
        let m = find(&cat, "fasth3-8step-dense").unwrap();
        let CudaRecipe::H3(r) = &m.recipe else {
            panic!()
        };
        assert_eq!(r.weights, Path::new("/w/h3-8step"));
        assert!(r.dense);
    }

    /// FastH3 Trim: its own tree for the DiT and VAE, h3-base for the text
    /// encoder and audio VAE, the trained VSA recipe (not dense), untiered,
    /// text-to-video only, addressed by id, `h3-trim` or `recipe = "trim-8step"`.
    #[test]
    fn trim_catalog_entry() {
        let l = WeightLayout::new("/w");
        let cat = catalog(&l).unwrap();
        for name in [FASTH3_TRIM_ID, H3_TRIM] {
            let m = find(&cat, name).unwrap();
            assert_eq!(m.id.as_str(), FASTH3_TRIM_ID, "{name}");
            assert!(m.tier.is_none());
            let CudaRecipe::H3(r) = &m.recipe else {
                panic!()
            };
            assert_eq!(r.weights, Path::new("/w/fasth3-trim-8step"));
            assert_eq!(r.text_weights.as_deref(), Some(Path::new("/w/h3-base")));
            assert_eq!(
                (r.recipe.as_str(), r.steps, r.dense, r.profile.as_deref()),
                ("trim-8step", 8, false, None)
            );
            assert!(!r.ref2va && !r.hd_1080p && r.taeh3.is_none());
            // The recipe name is one the pipeline's contract resolver knows.
            assert_eq!(
                h3cfg::H3InferenceContract::named(&r.recipe).unwrap(),
                h3cfg::H3InferenceContract::fasth3_8step()
            );
            let c = m.caps();
            assert_eq!(c.tasks.iter().copied().collect::<Vec<_>>(), [Task::T2V]);
            assert!(c.audio.is_some());
            assert_eq!(c.canvas.short_edges, vec![768, 480]);
            assert_eq!(c.frames, FrameGrid::h3());
            let d = m.describe();
            assert_eq!(
                (d.steps, d.attention.as_str(), d.vae.as_str()),
                (Some(8), "vsa", "full")
            );
            assert!(d.summary.contains("Trim"), "{}", d.summary);
        }
        assert!(m_served(&cat, H3_TRIM));
        for rec in ["fasth3-trim", H3_TRIM, FASTH3_TRIM_ID] {
            let e = ModelEntryCfg {
                variant: rec.into(),
                resident: true,
                ..Default::default()
            };
            assert_eq!(
                model_from_config(&l, &e).unwrap().id.as_str(),
                FASTH3_TRIM_ID,
                "{rec}"
            );
        }
    }

    /// FastH3 Trim NVFP4: the Trim's trees and recipe contract, the NVFP4
    /// recipe (the pipeline reads `fasth3-trim-8step-nvfp4` beside the tree),
    /// untiered and opt-in, text-to-video only; the bf16 Trim is unchanged.
    #[test]
    fn trim_nvfp4_catalog_entry() {
        let l = WeightLayout::new("/w");
        let cat = catalog(&l).unwrap();
        for name in [FASTH3_TRIM_NVFP4_ID, H3_TRIM_NVFP4] {
            let m = find(&cat, name).unwrap();
            assert_eq!(m.id.as_str(), FASTH3_TRIM_NVFP4_ID, "{name}");
            assert!(m.tier.is_none());
            let CudaRecipe::H3(r) = &m.recipe else {
                panic!()
            };
            assert_eq!(r.weights, Path::new("/w/fasth3-trim-8step"));
            assert_eq!(r.text_weights.as_deref(), Some(Path::new("/w/h3-base")));
            assert_eq!(
                (r.recipe.as_str(), r.steps, r.dense, r.profile.as_deref()),
                ("trim-8step-nvfp4", 8, false, None)
            );
            assert_eq!(
                h3cfg::H3InferenceContract::named(&r.recipe).unwrap(),
                h3cfg::H3InferenceContract::fasth3_8step()
            );
            let c = m.caps();
            assert_eq!(c.tasks.iter().copied().collect::<Vec<_>>(), [Task::T2V]);
            let d = m.describe();
            assert!(d.summary.contains("NVFP4"), "{}", d.summary);
        }
        assert!(m_served(&cat, H3_TRIM_NVFP4));
        for rec in ["fasth3-trim-nvfp4", H3_TRIM_NVFP4] {
            let e = ModelEntryCfg {
                variant: rec.into(),
                resident: true,
                ..Default::default()
            };
            assert_eq!(
                model_from_config(&l, &e).unwrap().id.as_str(),
                FASTH3_TRIM_NVFP4_ID,
                "{rec}"
            );
        }
        // The default Trim stays the bf16 / MXFP8 checkpoint.
        let CudaRecipe::H3(t) = &find(&cat, H3_TRIM).unwrap().recipe else {
            panic!()
        };
        assert_eq!(t.recipe, "trim-8step");
    }

    /// Owner decisions 2026-10-14: the NVFP4 tier (V2 and 4-step NVFP4 from
    /// their own trees, h3-base for the text side), its TAEH3 "fast preview"
    /// twins and h3-turbo's (explicit ids, never a default), and the Trim
    /// NVFP4 + sparse FP4 attention "fastest" tier (FP4 VSA by the recipe).
    #[test]
    fn nvfp4_tier_previews_and_the_fastest_trim() {
        let l = WeightLayout::new("/w");
        let cat = catalog(&l).unwrap();
        let h3 = |name: &str| {
            let m = find(&cat, name).unwrap_or_else(|| panic!("{name}"));
            let CudaRecipe::H3(r) = m.recipe.clone() else {
                panic!("{name}")
            };
            (m, r)
        };
        for (id, served, tree, recipe, steps) in [
            (
                FASTH3_V2_NVFP4_ID,
                H3_V2_NVFP4,
                "fasth3-v2-nvfp4",
                "v2-nvfp4",
                8,
            ),
            (
                FASTH3_4STEP_NVFP4_ID,
                H3_TURBO_NVFP4,
                "fasth3-4step-vsa-nvfp4",
                "4step-vsa-nvfp4",
                4,
            ),
        ] {
            for name in [id, served] {
                let (m, r) = h3(name);
                assert_eq!(m.id.as_str(), id);
                assert!(m.tier.is_none(), "untiered: h3-turbo stays the default");
                assert_eq!(
                    (r.recipe.as_str(), r.steps, r.profile.as_deref()),
                    (recipe, steps, None)
                );
                assert_eq!(r.weights, Path::new("/w").join(tree));
                assert_eq!(
                    r.text_weights.as_deref(),
                    Some(Path::new("/w/h3-base")),
                    "text + vae shard 3"
                );
                assert!(r.taeh3.is_none(), "the official (NVFP4) decoder by default");
                assert_eq!(
                    h3cfg::H3InferenceContract::named(&r.recipe)
                        .unwrap()
                        .transformer_forwards as u32,
                    steps
                );
                assert!(m.requirements().nvfp4, "NVFP4 tensor cores only");
                assert_eq!(
                    m.caps().tasks.iter().copied().collect::<Vec<_>>(),
                    [Task::T2V]
                );
                let d = m.describe();
                assert_eq!(d.vae, "full");
                assert!(d.summary.contains("NVFP4 VAE"), "{}", d.summary);
            }
            let e = ModelEntryCfg {
                variant: served.into(),
                resident: true,
                ..Default::default()
            };
            assert_eq!(
                model_from_config(&l, &e).unwrap().id.as_str(),
                id,
                "{recipe}"
            );
        }
        // Fast preview: the same recipe and canvases, TAEH3 decode, its own id.
        let bases = [
            FASTH3_V2_NVFP4_ID,
            FASTH3_4STEP_NVFP4_ID,
            "fasth3-4step-vsa",
        ];
        let previews = [
            ("fasth3-v2-nvfp4-taeh3", "h3-v2-nvfp4-preview"),
            ("fasth3-4step-vsa-nvfp4-taeh3", "h3-turbo-nvfp4-preview"),
            ("fasth3-4step-vsa-taeh3", "h3-turbo-preview"),
        ];
        for ((id, served), base) in previews.iter().zip(bases) {
            let (m, r) = h3(served);
            let (bm, b) = h3(base);
            assert_eq!(m.id.as_str(), *id);
            assert!(m.tier.is_none(), "never a default");
            assert_eq!(
                r.taeh3.as_deref(),
                Some(Path::new("/w/auxiliary/tae/taeh3.safetensors"))
            );
            assert_eq!(
                (&r.recipe, &r.weights, r.steps, &r.profile),
                (&b.recipe, &b.weights, b.steps, &b.profile)
            );
            assert_eq!(
                r.short_edge, b.short_edge,
                "the base's canvases, 768p default"
            );
            assert!(!r.hd_1080p);
            assert_eq!(m.caps().canvas.short_edges, [768, 480]);
            assert_eq!(m.caps().tasks, bm.caps().tasks);
            let d = m.describe();
            assert_eq!(d.vae, "taeh3");
            assert!(
                d.summary.contains("fast preview") && d.summary.contains("sharpness"),
                "{}",
                d.summary
            );
        }
        // h3-turbo and the draft tier are unchanged.
        let (_, t) = h3("h3-turbo");
        assert!(t.taeh3.is_none() && t.hd_1080p);
        assert_eq!(h3("h3-draft").0.id.as_str(), "fasth3-4step-vsa-480p-taeh3");
        // The fastest tier: the Trim NVFP4's trees, sparse FP4 VSA by its recipe.
        for name in [FASTH3_TRIM_NVFP4_FAST_ID, H3_TRIM_NVFP4_FAST] {
            let (m, r) = h3(name);
            let (_, trim_nv) = h3(H3_TRIM_NVFP4);
            assert_eq!(m.id.as_str(), FASTH3_TRIM_NVFP4_FAST_ID);
            assert!(m.tier.is_none());
            assert_eq!(r.recipe, FASTH3_TRIM_NVFP4_FAST_RECIPE);
            assert!(h3cfg::is_trim_nvfp4_fp4attn_recipe(&r.recipe));
            assert_eq!(
                (&r.weights, &r.text_weights),
                (&trim_nv.weights, &trim_nv.text_weights)
            );
            assert!(
                m.recipe.process_env().is_empty(),
                "no process-wide variable"
            );
            assert!(m.requirements().nvfp4);
            let d = m.describe();
            assert!(
                d.summary.contains("sparse FP4") && d.summary.contains("not gate-equivalent"),
                "{}",
                d.summary
            );
        }
        for rec in [FASTH3_TRIM_NVFP4_FAST_ID, H3_TRIM_NVFP4_FAST] {
            let e = ModelEntryCfg {
                variant: rec.into(),
                resident: true,
                ..Default::default()
            };
            assert_eq!(
                model_from_config(&l, &e).unwrap().id.as_str(),
                FASTH3_TRIM_NVFP4_FAST_ID,
                "{rec}"
            );
        }
        let (_, slow) = h3(H3_TRIM_NVFP4);
        assert!(
            !h3cfg::is_trim_nvfp4_fp4attn_recipe(&slow.recipe),
            "h3-trim-nvfp4 keeps bf16 VSA"
        );
    }

    /// VDN-H3: the stage-B hybrid over h3-base, 50 forwards, text-to-video
    /// only, addressed by id or `recipe = "vdn-h3-50step"`.
    #[test]
    fn vdn_catalog_entry() {
        let l = WeightLayout::new("/w");
        let cat = catalog(&l).unwrap();
        let m = find(&cat, VDN_H3_ID).unwrap();
        assert!(m.tier.is_none());
        let CudaRecipe::H3(r) = &m.recipe else {
            panic!()
        };
        assert_eq!(r.weights, Path::new("/w/h3-base"));
        assert_eq!(r.text_weights.as_deref(), Some(Path::new("/w/h3-base")));
        assert_eq!(
            r.vdn_weights.as_deref(),
            Some(Path::new("/w/vdn-h3-stage-b-2000/stage-b-step-2000"))
        );
        assert_eq!(
            (r.recipe.as_str(), r.steps, r.dense),
            ("vdn-h3-50step", 50, true)
        );
        assert_eq!(r.dit_offload.as_deref(), Some("resident"));
        assert_eq!(
            h3cfg::H3InferenceContract::named(&r.recipe).unwrap(),
            h3cfg::H3InferenceContract::base(51)
        );
        let c = m.caps();
        assert_eq!(c.tasks.iter().copied().collect::<Vec<_>>(), [Task::T2V]);
        assert!(c.audio.is_some());
        let d = m.describe();
        assert_eq!((d.steps, d.attention.as_str()), (Some(50), "dense"));
        assert!(d.summary.contains("VDN-H3"), "{}", d.summary);
        for rec in [VDN_H3_ID, VDN_H3_RECIPE] {
            let e = ModelEntryCfg {
                variant: rec.into(),
                resident: true,
                ..Default::default()
            };
            assert_eq!(
                model_from_config(&l, &e).unwrap().id.as_str(),
                VDN_H3_ID,
                "{rec}"
            );
        }
    }

    /// VDN-H3 8-step: the turbo tree over stage B's branch and default LoRA,
    /// 8 forwards on the base grid, its own id.
    #[test]
    fn vdn_8step_catalog_entry() {
        let l = WeightLayout::new("/w");
        let cat = catalog(&l).unwrap();
        let m = find(&cat, VDN_H3_8STEP_ID).unwrap();
        assert!(m.tier.is_none());
        let CudaRecipe::H3(r) = &m.recipe else {
            panic!()
        };
        assert_eq!(r.weights, Path::new("/w/h3-base"));
        assert_eq!(
            r.vdn_weights.as_deref(),
            Some(Path::new("/w/vdn-h3-dmd-250-turbo/stage-dmd-step-250"))
        );
        assert_eq!(
            r.vdn_shared.as_deref(),
            Some(Path::new("/w/vdn-h3-stage-b-2000/stage-b-step-2000"))
        );
        assert_eq!(
            (r.recipe.as_str(), r.steps, r.dense),
            ("vdn-h3-8step", 8, true)
        );
        assert_eq!(r.dit_offload.as_deref(), Some("resident"));
        assert_eq!(
            h3cfg::H3InferenceContract::named(&r.recipe).unwrap(),
            h3cfg::H3InferenceContract::base(9)
        );
        assert_eq!(
            m.caps().tasks.iter().copied().collect::<Vec<_>>(),
            [Task::T2V]
        );
        let d = m.describe();
        assert_eq!(d.steps, Some(8));
        assert!(d.summary.contains("turbo"), "{}", d.summary);
        let e = ModelEntryCfg {
            variant: "vdn-h3-8step".into(),
            resident: true,
            ..Default::default()
        };
        assert_eq!(
            model_from_config(&l, &e).unwrap().id.as_str(),
            VDN_H3_8STEP_ID
        );
        // The 50-step entry carries no shared directory.
        let CudaRecipe::H3(b) = &find(&cat, VDN_H3_ID).unwrap().recipe else {
            panic!()
        };
        assert!(b.vdn_shared.is_none());
    }

    fn m_served(cat: &[CudaModel], name: &str) -> bool {
        cat.iter().any(|m| m.served_names.iter().any(|s| s == name))
    }

    /// LongLive-1.3B is its own variant (window 12, sink 3, re-cache, the
    /// served name `longlive`, its checkpoint tree); `[[models]]` may move the
    /// checkpoint (`longlive = "<dir>"`) and keep the cache at prompt switches
    /// (`longlive_recache = false`). The SF-Wan variant takes neither.
    #[test]
    fn longlive_variant_and_its_overrides() {
        let layout = WeightLayout::new("/w");
        let mut e = ModelEntryCfg {
            id: "longlive-1.3b".into(),
            variant: "longlive-1.3b".into(),
            weights: None,
            resident: true,
            served_names: Vec::new(),
            extra: BTreeMap::new(),
        };
        let m = model_from_config(&layout, &e).unwrap();
        let CudaRecipe::SfWan(r) = &m.recipe else {
            panic!()
        };
        let l = r.longlive.as_ref().unwrap();
        assert!(l.recache && l.lora && l.global_sink && !l.relative_rope);
        assert_eq!(l.weights, PathBuf::from("/w/longlive-1.3b-safetensors"));
        assert_eq!((r.local_attn_frames, r.sink_frames), (12, 3));
        assert_eq!(r.wan.weights, PathBuf::from("/w/sfwan21-1.3b"));
        let caps = m.caps();
        assert_eq!(caps.served_names, ["longlive-1.3b", LONGLIVE_NAME]);
        assert_eq!(
            caps.stream,
            Some(StreamCaps::Causal {
                block_frames: 12,
                target_fps: 16,
                context: Some(fastvideo_protocol::CausalContext {
                    window_latent_frames: 12,
                    sink_latent_frames: 3,
                    prompt_recache: true
                }),
            })
        );
        assert!(
            m.describe().summary.starts_with("LongLive"),
            "{}",
            m.describe().summary
        );
        e.extra
            .insert("longlive".into(), "/elsewhere/longlive".into());
        e.extra.insert("longlive_recache".into(), "false".into());
        let m = model_from_config(&layout, &e).unwrap();
        let CudaRecipe::SfWan(r) = &m.recipe else {
            panic!()
        };
        let l = r.longlive.as_ref().unwrap();
        assert!(!l.recache);
        assert_eq!(l.weights, PathBuf::from("/elsewhere/longlive"));
        assert!(matches!(
            m.caps().stream,
            Some(StreamCaps::Causal {
                context: Some(fastvideo_protocol::CausalContext {
                    prompt_recache: false,
                    ..
                }),
                ..
            })
        ));
        e.extra.insert("longlive_recache".into(), "maybe".into());
        assert!(model_from_config(&layout, &e).is_err());
        // The plain SF-Wan variant has no LongLive and takes no LongLive keys.
        let plain = ModelEntryCfg {
            id: "sfwan21-1.3b".into(),
            variant: "sfwan21-1.3b".into(),
            ..Default::default()
        };
        let p = model_from_config(&layout, &plain).unwrap();
        let CudaRecipe::SfWan(r) = &p.recipe else {
            panic!()
        };
        assert!(r.longlive.is_none());
        assert!(!p.served_names.iter().any(|n| n == LONGLIVE_NAME));
        let mut with = plain.clone();
        with.extra.insert("longlive".into(), "/w/x".into());
        assert!(model_from_config(&layout, &with).is_err());
        let mut with = plain.clone();
        with.extra.insert("longlive_recache".into(), "false".into());
        assert!(model_from_config(&layout, &with).is_err());
    }

    /// LongLive is opt-in on an SF-Wan recipe: absent from the catalog's
    /// recipe (and its JSON), defaults on when a deployment sets it.
    #[test]
    fn sfwan_longlive_is_opt_in() {
        let cat = catalog(&WeightLayout::default()).unwrap();
        let sf = cat
            .iter()
            .find(|m| m.id.as_str() == "sfwan21-1.3b")
            .unwrap();
        let CudaRecipe::SfWan(r) = &sf.recipe else {
            panic!("sfwan recipe")
        };
        assert!(r.longlive.is_none());
        let mut v = serde_json::to_value(r).unwrap();
        assert!(v.get("longlive").is_none());
        v["longlive"] = serde_json::json!({"weights": "/w/longlive-1.3b-safetensors"});
        v["local_attn_frames"] = 12.into();
        v["sink_frames"] = 3.into();
        let ll: SfWanRecipe = serde_json::from_value(v).unwrap();
        let l = ll.longlive.as_ref().unwrap();
        assert!(l.lora && l.recache && l.global_sink && !l.relative_rope);
        assert_eq!(l.weights, PathBuf::from("/w/longlive-1.3b-safetensors"));
    }
}
