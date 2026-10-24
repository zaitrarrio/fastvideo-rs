//! [`H3Pipeline::load_with_settings`] in four stages
//! (`docs/adr/0017-loader-stages.md`):
//!
//! 1. **plan** ([`H3LoadPlan::new`]): pure. From the options, the settings
//!    snapshot, the snapshot's configs and manifests (`config.json`,
//!    `fastvideo_inference.json`, the MLX and pre-quantized-tree manifests,
//!    safetensors headers) and the [`DeviceFacts`], decide what loads, from
//!    where, at what precision, where it lives and which adapter merges. No
//!    device read, no tensor byte read.
//! 2. **open** ([`H3LoadPlan::open`]): map the checkpoint files and queue
//!    their read-ahead in load order. No device.
//! 3. **place** ([`H3LoadPlan::place`]): upload, quantize and offload onto
//!    the device: the text encoder, the image-to-video tower, the refiner,
//!    the DiT (with the planned adapter merged into each linear as it loads,
//!    before any quantization), the VDN branch, the decoders.
//! 4. **fuse** ([`H3LoadPlan::fuse`]): the passes over the placed model: the
//!    adapter merge's report, the step-output cache (TeaCache); then the
//!    pipeline.
//!
//! The composition is the load it replaces, step for step: the same tensors
//! in the same order and precision, the same placement decisions, the same
//! `load/io` lines and ledger bookings.

use super::*;
use crate::h3::nvfp4::{ModelOptOptions, Nvfp4Dit};
use crate::rt::load_stages::DeviceFacts;
use crate::rt::quant::QuantMode;
use fastvideo_models::h3::memory::H3VaeLinears;
use fastvideo_models::h3::sol::H3TeaCache;
use fastvideo_models::h3::turbo_lora::TurboLora;
use fastvideo_models::techniques::settings::Active;

/// Where the DiT's weights come from.
#[derive(Clone, Debug)]
pub enum H3DitSource {
    /// The snapshot's `transformer/` (T2AV, FL2VA).
    Transformer(PathBuf),
    /// `transformer_ref/` (Ref2VA), under `ref_root` or the snapshot.
    TransformerRef(PathBuf),
    /// A FastVideo MLX export (`mlx_h3_dit.safetensors`, affine INT8/6/4;
    /// [`crate::h3::mlx`]). Opening it sets `FASTVIDEO_H3_AFFINE` for the
    /// linears that load after it.
    Mlx {
        dir: PathBuf,
        manifest: crate::h3::mlx::MlxH3Manifest,
    },
}

/// The published NVFP4 block linears a load reads, if any.
#[derive(Clone, Debug, PartialEq)]
pub enum H3Nvfp4Plan {
    None,
    /// FastH3 Trim's ComfyUI NVFP4 file: its block linears replace the
    /// snapshot's.
    Comfy(PathBuf),
    /// A ModelOpt NVFP4 `transformer/` (FastH3 V2 / 4-step NVFP4 releases).
    ModelOpt(ModelOptOptions),
}

/// The adapter merged into the DiT's linears while they load (the load-time
/// fusion), chosen as the load always chose it.
#[derive(Clone, Debug)]
pub enum H3AdapterPlan {
    /// No adapter.
    None,
    /// A pre-merged, pre-quantized tree ([`crate::h3::dit_tree`]): nothing
    /// merges at load.
    Tree,
    /// FastH3 Preview v1: base MiniMax-H3 + one Preview LoRA.
    Preview(FastH3PreviewVariant),
    /// Sol-H3's adapter (T2V/I2V, Ref2VA or the Spark VSA-DataFree one).
    SolH3 { spec: SolH3AdapterSpec, spark: bool },
    /// VDN-H3: the checkpoint's LoRA adapters at scale 1.
    Vdn,
    /// A LongLive-Plug H3 recipe.
    Plug(PlugRecipe),
    /// A MiniMax-H3 Turbo LoRA folded into the bf16 weights.
    Turbo { turbo: TurboLora, forwards: usize },
}

impl H3AdapterPlan {
    fn describe(&self) -> String {
        match self {
            Self::None => "none".into(),
            Self::Tree => "pre-merged tree".into(),
            Self::Preview(v) => format!("fasth3-preview {}", v.subdir()),
            Self::SolH3 { spec, spark } => format!(
                "sol-h3{} alpha={} scale={}",
                if *spark { "-spark" } else { "" },
                spec.alpha,
                spec.scale
            ),
            Self::Vdn => "vdn".into(),
            Self::Plug(p) => format!("plug {}", p.name),
            Self::Turbo { turbo, forwards } => {
                format!("turbo {} forwards={forwards}", turbo.stem())
            }
        }
    }
}

/// Everything an H3 load decides before it touches the device or reads a
/// tensor ([`Self::new`]).
pub struct H3LoadPlan {
    pub root: PathBuf,
    /// The options as resolved: the recipe (the profile's when the caller
    /// named none), Ref2VA, the reference fit, `dense`, the text encoder
    /// choice. `profile` is taken (`None`).
    pub options: H3PipelineOptions,
    pub contract: H3InferenceContract,
    pub schedule: H3JointSchedule,
    pub cfg: H3TransformerConfig,
    pub vae_source: VideoVaeSource,
    pub techniques: H3Techniques,
    /// `options.dense` before the techniques forced it ([`H3Pipeline::set_arm`]).
    pub base_dense: bool,
    pub vdn: Option<fastvideo_models::h3::vdn::VdnCheckpoint>,
    pub facts: DeviceFacts,
    /// `FASTVIDEO_H3_QUANT` for the DiT and refiner linears on this device
    /// (an invalid value is kept as its error, raised where the load reads it).
    pub quant: std::result::Result<QuantMode, String>,
    /// What an unset `FASTVIDEO_H3_QUANT` means during this load (VDN-H3:
    /// bf16, owner decision 2026-10-14).
    pub quant_default: Option<QuantMode>,
    /// The text encoder choice was `Auto`.
    pub auto_text_encoder: bool,
    /// A resident text encoder loads.
    pub load_text_encoder: bool,
    pub dit: H3DitSource,
    pub nvfp4: H3Nvfp4Plan,
    /// What a pre-quantized DiT tree for this load must have been built
    /// from (`None`: this load has no stored form, or it was not looked for:
    /// MLX, Ref2VA, VDN, NVFP4).
    pub dit_tree_identity: Option<crate::h3::dit_tree::Identity>,
    /// The valid pre-quantized DiT tree this load reads, if any.
    pub dit_tree: Option<PathBuf>,
    /// The DiT loads its VSA gate (`to_gate_compress`).
    pub with_gate: bool,
    /// How the official video decoder's linears load (as the memory plan prices them).
    pub vae_linears: H3VaeLinears,
    /// The TAEH3 weights decoding video instead of the official decoder.
    pub taeh3: Option<PathBuf>,
    pub audio_dir: PathBuf,
    /// Where the refiner and DiT blocks go (`dit_offload`), resolved at place
    /// against the free memory measured there ([`Self::residency`]).
    pub dit_policy: DitOffload,
    /// The default request's resident plan peak ([`dit_placement_need`]).
    pub resident_need: u64,
    /// The default request's plan peak with the DiT streamed.
    pub streamed_need: u64,
    pub adapter: H3AdapterPlan,
    /// The AdaLN projections are not read (a valid cache file, no adapter).
    pub skip_adaln: bool,
    /// The step-output cache the fuse stage installs.
    pub teacache: Option<H3TeaCache>,
    /// The runtime's block fusions under bf16 activations, as the snapshot
    /// sets them: `FASTVIDEO_H3_FUSE` and, with it, `FASTVIDEO_H3_GLUE`
    /// (`crate::h3::fused16`).
    pub kernel_fusions: (bool, bool),
}

/// The planned DiT tensors' mappings and the decoders', opened ([`H3LoadPlan::open`]).
pub struct H3Opened {
    io_base: fastvideo_loader::PrefetchStats,
    io_timer: Instant,
    text_map: Option<WeightMap>,
    map: WeightMap,
    nvfp4_dit: Option<Nvfp4Dit>,
    dit_tree: Option<crate::h3::dit_tree::TreeReader>,
    vae_map: Option<WeightMap>,
    audio_vae_map: Option<WeightMap>,
}

impl H3Opened {
    /// The DiT's map (the MLX export, `transformer/` or `transformer_ref/`).
    pub fn dit_map(&self) -> &WeightMap {
        &self.map
    }
    /// The resident text encoder's map, when one loads.
    pub fn text_map(&self) -> Option<&WeightMap> {
        self.text_map.as_ref()
    }
    /// The video decoder's map (`None`: TAEH3, or no `vae/`).
    pub fn vae_map(&self) -> Option<&WeightMap> {
        self.vae_map.as_ref()
    }
    /// The audio decoder's map (`None`: no `audio_vae/`).
    pub fn audio_vae_map(&self) -> Option<&WeightMap> {
        self.audio_vae_map.as_ref()
    }
    /// The pre-quantized tree opened.
    pub fn has_dit_tree(&self) -> bool {
        self.dit_tree.is_some()
    }
}

/// The placed models ([`H3LoadPlan::place`]), before the fuse stage.
pub struct H3Placed {
    load_timings: H3LoadTimings,
    booking: crate::rt::ledger::Booking,
    text_encoder: Option<Box<dyn HiddenStateEncoder + Send>>,
    multimodal: Option<crate::h3::text::MultimodalEncoder>,
    residency: Residency,
    refiner: H3TextRefiner,
    model: H3Transformer,
    lora: Option<crate::h3::lora::H3LoraFuse>,
    video_vae: Option<VideoDecoder>,
    audio_vae: Option<H3AudioDecoder>,
    vsa_fp4: bool,
}

impl H3LoadPlan {
    /// The plan of loading `root` with `options` under `settings` (the
    /// load's snapshot), the process's `active` profile and the device as
    /// `facts` describe it. Logs what it decides (the profile, the
    /// techniques, the `Auto` text encoder, the recipe, the Sol route) and
    /// reads only configs and manifests.
    pub fn new(
        root: &Path,
        options: H3PipelineOptions,
        settings: &Snapshot,
        active: &Active,
        facts: DeviceFacts,
    ) -> Result<Self> {
        // The callees below read their settings through the scope (a2); a
        // nested scope of the same snapshot when the load already holds it.
        let _scope = settings.scope();
        let mut options = options;
        // The profile this pipeline runs: the caller's own (fv-serve, one per
        // model), else the process's.
        let own = options.profile.take();
        let profile_in_use = match &own {
            Some(p) => p.as_ref(),
            None => active.profile.as_ref(),
        };
        // A technique profile names its recipe; the caller's wins.
        if let Some(profile) = profile_in_use {
            match (&options.recipe, &profile.recipe) {
                (None, Some(r)) => options.recipe = Some(r.clone()),
                (Some(a), Some(b)) if a != b => crate::rt::log::info(format_args!(
                    "h3 techniques: recipe {a} (command line) overrides profile {}'s {b}",
                    profile.name
                )),
                _ => {}
            }
        }
        if options.recipe.as_deref().is_some_and(sol_h3_forces_ref2va) {
            options.ref2va = true;
        }
        if options.recipe.as_deref().is_some_and(is_sol_h3_recipe)
            && options.reference_image_resize == ReferenceImageResize::Auto
        {
            options.reference_image_resize = ReferenceImageResize::Match;
        }
        let contract = resolve_contract(root, options.recipe.as_deref())?;
        let vdn_ckpt = vdn_checkpoint(&options, settings)?;
        // VDN-H3 defaults to bf16 linears (owner decision 2026-10-14; MXFP8
        // cost 9 dB of clip PSNR against the reference, docs/ports/vdn-h3.md
        // §5). `FASTVIDEO_H3_QUANT=mxfp8` still opts in. The load holds the
        // default over its other stages ([`Self::quant_default_guard`]).
        let quant_default = match &vdn_ckpt {
            Some(_) => Some(QuantMode::Off),
            None => QuantMode::current_scoped_default(),
        };
        let quant = QuantMode::resolve_on(
            settings.var(crate::rt::quant::ENV).as_deref(),
            quant_default,
            facts.live_sm_major(),
        );
        let cfg = dit_config(root, &options)?;
        // The video VAE: the tree's own `vae/`, completed from h3-base's for
        // the shards an NVFP4 release does not carry, or h3-base's under
        // `FASTVIDEO_H3_VAE_NVFP4=0` (docs/perf/h3-nvfp4-tier-options.md).
        let vae_source = resolve_video_vae_on(
            root,
            options.text_root.as_deref(),
            crate::h3::vae::nvfp4_setting_from(settings)?,
            facts.live_sm_major(),
        )?;
        let schedule = H3JointSchedule::from_contract(&contract).map_err(msg)?;
        if contract.dense {
            options.dense = true;
        }
        // `H3Techniques::from_settings` with the profile given (`active`),
        // not read from the process: the same resolution for the process's
        // own profile, and a dry run or a test can plan under another.
        let techniques = match &own {
            None => H3Techniques::resolve(
                options.recipe.as_deref(),
                &contract,
                options.ref2va,
                active.profile.as_ref(),
                &|k| settings.var(k),
            ),
            Some(p) => H3Techniques::resolve(
                options.recipe.as_deref(),
                &contract,
                options.ref2va,
                p.as_ref(),
                &|k| settings.var(k),
            ),
        }
        .map_err(msg)?;
        let base_dense = options.dense;
        if techniques.forces_dense() {
            options.dense = true;
        }
        let taeh3 = resolve_taeh3(options.taeh3.as_deref(), settings);
        if techniques.taeh3.is_some() && taeh3.is_none() {
            return Err(msg(
                "techniques.taeh3: no TAEH3 weights (pass --taeh3-weights, set FASTVIDEO_TAEH3_WEIGHTS, or give techniques.taeh3.weights)",
            ));
        }
        if let Some(p) = own.as_ref().and_then(|p| p.as_ref()) {
            crate::rt::log::info(format_args!(
                "h3 technique profile: {} (this model's own)",
                p.name
            ));
            for note in &p.notes {
                crate::rt::log::info(format_args!("h3 technique profile note: {note}"));
            }
        } else if let (None, Some(path)) = (&own, active.path.as_ref()) {
            crate::rt::log::info(format_args!("h3 technique profile: {}", path.display()));
            for note in active.profile.iter().flat_map(|p| p.notes.iter()) {
                crate::rt::log::info(format_args!("h3 technique profile note: {note}"));
            }
            for (k, v, source) in active.settings.iter() {
                crate::rt::log::info(format_args!(
                    "h3 technique setting {k}={v} ({source}{})",
                    if settings.env_var_os(k).is_some() {
                        "; overridden by env"
                    } else {
                        ""
                    }
                ));
            }
        }
        crate::rt::log::info(format_args!("{}", techniques.describe()));

        // How the official video decoder's linears load: its MXFP8 copies
        // (`FASTVIDEO_H3_VAE_MXFP8`, default on sm_100+), or the published
        // NVFP4 linears alone (the NVFP4 tier's default); `MXFP8=1` on a GPU
        // without block-scaled FP8 fails here, before anything loads.
        let vae_linears = vae_source.planned_linears_on(
            options.taeh3.as_deref(),
            settings,
            facts.live_sm_major(),
        )?;
        let mlx_dir = crate::h3::mlx::find(root).filter(|_| !options.ref2va);
        let nvfp4_file = nvfp4_dit_file(root, &options)?;
        let dit_policy =
            DitOffload::from_settings_or(settings, options.dit_offload).map_err(msg)?;

        // `Auto`: the encoder is resident when the memory plan at the DiT's
        // real precision fits the card with it, else streamed; decided on
        // the free memory before any load.
        let free = facts.free;
        let auto_text_encoder = options.text_encoder == TextEncoderChoice::Auto;
        let auto_resident = match (auto_text_encoder, free) {
            (true, Some(f)) => {
                let dit = planned_dit_weights(nvfp4_file.is_some(), mlx_dir.is_some(), &quant)?;
                let (inputs, decision) = auto_text_encoder_plan(
                    f,
                    dit_policy,
                    dit,
                    // An MLX tree may not carry the gate; pricing it is the safe side.
                    dit_loads_vsa_gate(&contract, None),
                    vae_linears,
                    options.auto_text_encoder_request.clone(),
                    facts.bf16_activations,
                    options.i2v_encoder != I2vEncoderChoice::Stream,
                );
                crate::rt::log::info(format_args!(
                    "h3 text encoder: auto -> {}",
                    decision.describe(&inputs)
                ));
                decision.resident
            }
            (true, None) => {
                crate::rt::log::info(format_args!(
                    "h3 text encoder: auto -> streamed (free device memory unknown)"
                ));
                false
            }
            (false, _) => false,
        };
        let (text_encoder_choice, load_text_encoder) =
            options.text_encoder.resolve(auto_resident, free);
        options.text_encoder = text_encoder_choice;

        // The DiT's source.
        let dit = match mlx_dir {
            Some(dir) => H3DitSource::Mlx {
                manifest: crate::h3::mlx::read_manifest(&dir)?,
                dir,
            },
            None if options.ref2va => {
                let p = options
                    .ref_root
                    .as_deref()
                    .unwrap_or(root)
                    .join("transformer_ref");
                if !p.is_dir() {
                    return Err(msg(format!(
                        "Ref2VA needs {} (MiniMax-H3 Base Ref2VA partition)",
                        p.display()
                    )));
                }
                H3DitSource::TransformerRef(p)
            }
            None => H3DitSource::Transformer(root.join("transformer")),
        };
        let mlx = matches!(dit, H3DitSource::Mlx { .. });
        // FastH3 Trim NVFP4 (docs/ports/h3-trim.md §5): the published block
        // linears replace the snapshot's. Refused without NVFP4 tensor cores.
        let nvfp4 = match nvfp4_file {
            Some(path) => {
                if mlx || options.ref2va {
                    return Err(msg(
                        "h3 nvfp4: the published NVFP4 DiT is the Trim T2V transformer; not with an MLX snapshot or Ref2VA",
                    ));
                }
                crate::h3::nvfp4::require_tensor_cores_on(facts.device_sm)?;
                H3Nvfp4Plan::Comfy(path)
            }
            // FastH3 V2 / 4-step NVFP4 releases (docs/ports/fasth3-nvfp4.md):
            // a `transformer/` in ModelOpt's unified layout is read as such.
            None if !mlx
                && !options.ref2va
                && crate::h3::nvfp4::is_modelopt_dir(&root.join("transformer"))? =>
            {
                if vdn_ckpt.is_some() {
                    return Err(msg(
                        "h3 nvfp4: a ModelOpt NVFP4 DiT takes no VDN checkpoint",
                    ));
                }
                crate::h3::nvfp4::require_tensor_cores_on(facts.device_sm)?;
                H3Nvfp4Plan::ModelOpt(ModelOptOptions::from_settings(settings)?)
            }
            None => H3Nvfp4Plan::None,
        };
        // A V2 / 4-step NVFP4 release recipe runs the release's ModelOpt DiT,
        // never another `transformer/` under its contract.
        if let Some(r) = options
            .recipe
            .as_deref()
            .filter(|r| fastvideo_models::h3::config::is_nvfp4_release_recipe(r))
        {
            if !matches!(nvfp4, H3Nvfp4Plan::ModelOpt(_)) {
                return Err(msg(format!(
                    "h3 recipe {r}: {} is not a ModelOpt NVFP4 transformer/ (the fasth3-v2-nvfp4 / fasth3-4step-vsa-nvfp4 tree's); point weights at that tree",
                    root.join("transformer").display()
                )));
            }
        }
        // Fast boot (A): the recipe's resident DiT, pre-merged and
        // pre-quantized beside `transformer/` (`crate::h3::dit_tree`), when a
        // valid one is there; otherwise the checkpoint.
        let (dit_tree_identity, dit_tree) = if !mlx
            && !options.ref2va
            && vdn_ckpt.is_none()
            && matches!(nvfp4, H3Nvfp4Plan::None)
        {
            let identity = dit_tree_identity(
                root,
                &options,
                &schedule,
                &cfg,
                dit_loads_vsa_gate(&contract, None),
                settings,
                &quant,
                facts.linear_route,
            )?;
            let tree =
                crate::h3::dit_tree::select(root, &root.join("transformer"), identity.as_ref())?
                    .map(|(dir, _)| dir);
            (identity, tree)
        } else {
            (None, None)
        };
        // A fused adapter rebuilds the AdaLN table from the projections;
        // otherwise a valid cache file means they are never read.
        let has_adapter = vdn_ckpt.is_some()
            || options.recipe.as_deref().is_some_and(|r| {
                FastH3PreviewVariant::from_recipe(r).is_some()
                    || is_sol_h3_recipe(r)
                    || PlugRecipe::is_plug(r)
                    || parse_turbo_recipe(r).is_some()
            });
        let skip_adaln = !has_adapter && options.adaln_cache.as_deref().is_some_and(Path::is_file);
        let audio_dir = audio_vae_dir(root, options.text_root.as_deref());

        // Gate lives on VSA checkpoints and VSA-DataFree replacements. Sol-H3
        // + MiniMax-H3 is Sol-Attn on a dense backbone (`vsa_sparsity == 0`).
        let with_gate = dit_loads_vsa_gate(
            &contract,
            match &dit {
                H3DitSource::Mlx { manifest, .. } => Some(manifest.vsa_capable),
                _ => None,
            },
        );
        crate::rt::log::info(format_args!(
            "h3 recipe={} steps={} video_shift={} vsa={} dense={}",
            options.recipe.as_deref().unwrap_or("auto"),
            contract.transformer_forwards,
            contract.video_scheduler_shift,
            contract.vsa_sparsity,
            options.dense
        ));
        let preset = techniques
            .sol()
            .and_then(|s| s.preset.as_deref())
            .and_then(fastvideo_models::techniques::methods::SolAttn::preset);
        let named = techniques.sol().is_some_and(|s| preset.as_ref() == Some(s));
        match techniques.sol_policy {
            _ if techniques.sol().is_some() && !named => {
                crate::rt::log::info(format_args!(
                    "h3 sol-attn: {}",
                    techniques
                        .sol()
                        .map(|s| s.route.describe())
                        .unwrap_or_default()
                ));
            }
            H3SolAttnPolicy::Engine => {
                crate::rt::log::info(format_args!(
                    "h3 sol-attn: sol-h3 engine policy (opt-in): forward 0 dense, blocks 0-1 dense, blocks 2-49 sol-attn tau 1.0 (thresh_type=diag), prefix sink"
                ));
            }
            H3SolAttnPolicy::Spark => {
                crate::rt::log::info(format_args!(
                    "h3 sol-attn: spark draft ladder (opt-in): update 0 dense, later updates block 0 dense and blocks 1-49 sol-attn tau 1/1.25/1.5 (thresh_type=diag), [visual | text+audio] suffix sink"
                ));
            }
            H3SolAttnPolicy::Rtx => {
                crate::rt::log::info(format_args!(
                    "h3 sol-attn: rtx forwards 0-9 dense, later forwards blocks 0-1 dense and blocks 2-49 sol-attn tau 1.0 (thresh_type=diag), text sink"
                ));
            }
            H3SolAttnPolicy::Off => {}
        }

        // The adapter the DiT's linears merge as they load.
        let adapter = if dit_tree.is_some() {
            // The tree holds the merged (and quantized) weights already.
            H3AdapterPlan::Tree
        } else if let Some(variant) = options
            .recipe
            .as_deref()
            .and_then(FastH3PreviewVariant::from_recipe)
        {
            H3AdapterPlan::Preview(variant)
        } else if options.recipe.as_deref().is_some_and(is_sol_h3_recipe) {
            let spark = options
                .recipe
                .as_deref()
                .is_some_and(fastvideo_models::h3::lora::is_sol_h3_spark_recipe);
            let spec = if options.ref2va {
                SolH3AdapterSpec::ref2va()
            } else if spark {
                fastvideo_models::h3::spark::spark_adapter_spec()
            } else {
                SolH3AdapterSpec::t2v_i2v()
            };
            H3AdapterPlan::SolH3 { spec, spark }
        } else if vdn_ckpt.is_some() {
            H3AdapterPlan::Vdn
        } else if let Some(plug) = options.recipe.as_deref().and_then(PlugRecipe::named) {
            if options.ref2va {
                return Err(msg(format!(
                    "recipe {}: the LongLive-Plug H3 adapters are T2VA (base transformer), not Ref2VA",
                    plug.name
                )));
            }
            H3AdapterPlan::Plug(plug)
        } else if let Some(turbo) = options.recipe.as_deref().and_then(parse_turbo_recipe) {
            let (turbo, forwards) = turbo.map_err(msg)?;
            if options.ref2va {
                return Err(msg(format!(
                    "recipe {}: the Turbo LoRAs are T2VA / FL2VA (base transformer), not Ref2VA",
                    turbo.stem()
                )));
            }
            H3AdapterPlan::Turbo { turbo, forwards }
        } else {
            H3AdapterPlan::None
        };
        // The step-output seam: TeaCache over the recipe's forwards unless
        // the technique names its own count.
        let teacache = techniques
            .teacache_state(schedule.num_steps())
            .map_err(msg)?;
        // The placement inputs: what the default request (768p, 5 s) needs
        // with everything resident, and with the DiT's blocks streamed from
        // host memory.
        let resident_need = dit_placement_need(with_gate, vae_linears);
        let streamed_need = H3Geometry::default_16x9(5)
            .map(|g| {
                fastvideo_models::h3::memory::plan(
                    &g,
                    fastvideo_models::h3::memory::H3PlanOptions {
                        placement: fastvideo_models::h3::memory::H3Placement::STREAMED,
                        gate: with_gate,
                        vae: vae_linears,
                        ..Default::default()
                    },
                )
                .peak()
            })
            .map_err(msg)?;
        Ok(Self {
            root: root.to_path_buf(),
            options,
            contract,
            schedule,
            cfg,
            vae_source,
            techniques,
            base_dense,
            vdn: vdn_ckpt,
            facts,
            quant,
            quant_default,
            auto_text_encoder,
            load_text_encoder,
            dit,
            nvfp4,
            dit_tree_identity,
            dit_tree,
            with_gate,
            vae_linears,
            taeh3,
            audio_dir,
            dit_policy,
            resident_need,
            streamed_need,
            adapter,
            skip_adaln,
            teacache,
            kernel_fusions: {
                let fuse = settings.bool_flag("FASTVIDEO_H3_FUSE", true);
                (fuse, fuse && settings.bool_flag("FASTVIDEO_H3_GLUE", true))
            },
        })
    }

    /// The thread's `FASTVIDEO_H3_QUANT` default for the rest of the load
    /// (VDN-H3's bf16); `None` when the plan keeps the thread's own.
    pub fn quant_default_guard(&self) -> Option<crate::rt::quant::ScopedQuantDefault> {
        self.vdn
            .as_ref()
            .map(|_| QuantMode::scoped_default(QuantMode::Off))
    }

    /// Where the refiner and DiT blocks live when `free` bytes are free
    /// after the encoders loaded: the policy against the default request's
    /// resident plan; an NVFP4 DiT is resident only (its device operands have
    /// no host-streamed form).
    pub fn residency(&self, free: Option<u64>) -> Result<Residency> {
        let residency = self.dit_policy.resolve(self.resident_need, free);
        match (&self.nvfp4, self.dit_policy) {
            (H3Nvfp4Plan::None, _) => Ok(residency),
            (_, DitOffload::Streamed) => Err(msg(
                "h3 nvfp4: the NVFP4 DiT is resident only (dit_offload = streamed was asked)",
            )),
            (_, _) => Ok(Residency::Resident),
        }
    }

    /// Whether the FL2VA / Ref2VA vision tower loads beside the resident
    /// language model (`lm_resident`: the resident text encoder is a
    /// Qwen3-VL decoder; `text_loaded`: a resident encoder loaded), with
    /// `free` bytes free after the encoder; and why, for the log.
    pub fn i2v_tower(
        &self,
        lm_resident: bool,
        text_loaded: bool,
        free: Option<u64>,
    ) -> Result<(bool, String)> {
        let options = &self.options;
        let need = self.resident_need;
        let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
        Ok(match (options.i2v_encoder, lm_resident) {
            (I2vEncoderChoice::Stream, _) => (false, "configured".to_owned()),
            (I2vEncoderChoice::Resident | I2vEncoderChoice::Auto, false) => {
                if options.i2v_encoder == I2vEncoderChoice::Resident {
                    return Err(msg(format!(
                        "i2v_encoder = resident needs a resident Qwen3-VL text encoder (text_encoder = resident-fp8 | resident-bf16, or auto when the memory plan keeps it resident); it resolved to {:?}{}",
                        options.text_encoder,
                        if text_loaded { "" } else { " (not loaded)" }
                    )));
                }
                (false, format!("the text encoder is {:?}{}, not a resident Qwen3-VL", options.text_encoder, if text_loaded { "" } else { " (streamed)" }))
            }
            (I2vEncoderChoice::Resident, true) => (true, "configured".to_owned()),
            (I2vEncoderChoice::Auto, true) => match free {
                Some(f) if f >= need + I2V_VISION_RESERVE_BYTES => (
                    true,
                    format!(
                        "{:.1} GiB free covers the vision tower ({:.1} GiB) and the resident plan ({:.1} GiB)",
                        gib(f),
                        gib(I2V_VISION_RESERVE_BYTES),
                        gib(need)
                    ),
                ),
                // The DiT streams with or without the tower (e.g. the
                // upsampler sidecar holds 11 GB of a 96 GB card): the
                // tower is 2 GiB beside the streamed plan, while
                // streaming it re-reads the vision tower and the whole
                // language model (~26 GB FP8) from the volume for every
                // image-conditioned build (every anchored director
                // chunk; gpu-check-2026-10-13: 160-214 s chunks).
                Some(f)
                    if i2v_tower_beside_streamed_dit(
                        self.dit_policy,
                        f,
                        need,
                        self.streamed_need,
                    ) =>
                {
                    (
                        true,
                        format!(
                            "the DiT streams either way ({:.1} GiB free < the resident plan {:.1} GiB): {:.1} GiB free covers the vision tower ({:.1} GiB) and the streamed plan ({:.1} GiB)",
                            gib(f),
                            gib(need),
                            gib(f),
                            gib(I2V_VISION_RESERVE_BYTES),
                            gib(self.streamed_need)
                        ),
                    )
                }
                Some(f) => (
                    false,
                    format!(
                        "{:.1} GiB free does not cover the vision tower ({:.1} GiB) and the resident plan ({:.1} GiB)",
                        gib(f),
                        gib(I2V_VISION_RESERVE_BYTES),
                        gib(need)
                    ),
                ),
                None => (false, "free device memory unknown".to_owned()),
            },
        })
    }

    /// The plan's decisions, one `key: value` line each (the golden of
    /// `docs/adr/0017-loader-stages.md`): the facts, what loads from where,
    /// the precisions, the placement (at the facts' free memory), the
    /// adapter merge and the step cache.
    pub fn describe(&self) -> String {
        let gib = |b: u64| format!("{:.2}GiB", b as f64 / f64::from(1u32 << 30));
        // Paths under the snapshot, or beside it (the weights directory).
        let rel = |p: &Path| match (
            p.strip_prefix(&self.root),
            self.root.parent().map(|w| p.strip_prefix(w)),
        ) {
            (Ok(r), _) => format!("<root>/{}", r.display()),
            (_, Some(Ok(r))) => format!("<weights>/{}", r.display()),
            _ => p.display().to_string(),
        };
        let c = &self.contract;
        let mut out = vec![
            format!("facts: {}", self.facts.describe()),
            format!(
                "recipe: {} ref2va={} dense={} base_dense={}",
                self.options.recipe.as_deref().unwrap_or("auto"),
                self.options.ref2va,
                self.options.dense,
                self.base_dense
            ),
            format!(
                "contract: forwards={} steps={} video_shift={} audio_shift={} vsa={} tile={} sigmas={:?} fresh_noise={}",
                c.transformer_forwards,
                c.num_inference_steps,
                c.video_scheduler_shift,
                c.audio_scheduler_shift,
                c.vsa_sparsity,
                c.vsa_tile_size,
                c.sigma_source,
                c.fresh_noise
            ),
            format!(
                "dit: {} layers={} refiner_layers={} gate={}",
                match &self.dit {
                    H3DitSource::Transformer(p) => format!("transformer {}", rel(p)),
                    H3DitSource::TransformerRef(p) => format!("transformer_ref {}", rel(p)),
                    H3DitSource::Mlx { dir, manifest } => format!(
                        "mlx {} int{} g{}",
                        rel(dir),
                        manifest.bits,
                        manifest.group_size
                    ),
                },
                self.cfg.num_layers,
                self.cfg.num_refiner_layers,
                self.with_gate
            ),
            format!(
                "dit precision: quant={} nvfp4={} tree={}",
                match &self.quant {
                    Ok(q) => q.as_str().to_owned(),
                    Err(e) => format!("error({e})"),
                },
                match &self.nvfp4 {
                    H3Nvfp4Plan::None => "none".to_owned(),
                    H3Nvfp4Plan::Comfy(p) => format!("comfy {}", rel(p)),
                    H3Nvfp4Plan::ModelOpt(o) => format!(
                        "modelopt attention={} static_act={}",
                        o.attention, o.static_act
                    ),
                },
                self.dit_tree.as_deref().map_or("none".to_owned(), rel),
            ),
            format!(
                "text encoder: {:?} auto={} load={}",
                self.options.text_encoder, self.auto_text_encoder, self.load_text_encoder
            ),
            format!("i2v encoder: {}", self.options.i2v_encoder.as_str()),
            // Place decides on the free memory it measures after the text
            // encoder and the i2v tower; the load-start figure shows the rule.
            format!(
                "dit offload: {} resident_need={} streamed_need={}; at the load-start free memory -> {} (place re-measures after the encoders)",
                self.dit_policy.as_str(),
                gib(self.resident_need),
                gib(self.streamed_need),
                self.residency(self.facts.free)
                    .map_or_else(|e| format!("error({e})"), |r| r.as_str().to_owned())
            ),
            format!(
                "video vae: {} {:?} linears={:?} taeh3={}",
                rel(&self.vae_source.dir),
                self.vae_source.choice,
                self.vae_linears,
                self.taeh3.as_deref().map_or("none".to_owned(), rel)
            ),
            format!("audio vae: {}", rel(&self.audio_dir)),
            format!("fuse adapter: {}", self.adapter.describe()),
            format!(
                "fuse step cache: {}",
                self.teacache.as_ref().map_or("none".to_owned(), |t| format!(
                    "teacache threshold={} retain={} cooldown={}",
                    t.threshold, t.retain_steps, t.cooldown_steps
                ))
            ),
            format!(
                "fuse kernels: fused16={} glue={}",
                self.kernel_fusions.0, self.kernel_fusions.1
            ),
            format!("adaln projections: {}", if self.skip_adaln { "cached" } else { "read" }),
            format!("techniques: {}", self.techniques.describe()),
        ];
        out.push(String::new());
        out.join("\n")
    }

    /// Stage 2: map every component's shards (headers only) and queue their
    /// tensors for read-ahead in load order — encoder, refiner, AdaLN
    /// projections, blocks, decoders — so the volume stays busy across
    /// component boundaries (E12). The place stage reads these mappings.
    pub fn open(&self) -> Result<H3Opened> {
        let root = self.root.as_path();
        let options = &self.options;
        let io_base = fastvideo_loader::PrefetchStats::now();
        let io_timer = Instant::now();
        let text_map = match options
            .text_encoder
            .precision()
            .filter(|_| self.load_text_encoder)
        {
            Some(precision) => Some(crate::h3::text::open_resident_encoder(
                options.text_root.as_deref().unwrap_or(root),
                precision,
            )?),
            None => None,
        };
        let map = match &self.dit {
            H3DitSource::Mlx { dir, .. } => {
                let (map, spec) = crate::h3::mlx::open_map(dir)?;
                crate::rt::log::info(format_args!(
                    "h3 dit=mlx affine int{} g{} ({})",
                    spec.bits,
                    spec.group_size,
                    spec.weights.display()
                ));
                map
            }
            H3DitSource::TransformerRef(p) => {
                crate::rt::log::info(format_args!("h3 dit=transformer_ref"));
                WeightMap::open(p)?
            }
            H3DitSource::Transformer(p) => WeightMap::open(p)?,
        };
        let cfg = &self.cfg;
        let nvfp4_dit = match &self.nvfp4 {
            H3Nvfp4Plan::None => None,
            H3Nvfp4Plan::Comfy(path) => {
                let t = Instant::now();
                let nv = Nvfp4Dit::open(path, cfg, &map)?;
                crate::rt::log::info(format_args!(
                    "h3 nvfp4: {} ({} blocks x 5 NVFP4 linears, {:.2} GiB; checked against the bf16 tree in {:.1}s)",
                    nv.path().display(),
                    cfg.num_layers,
                    nv.linear_bytes(cfg, dit_loads_vsa_gate(&self.contract, None)) as f64 / f64::from(1u32 << 30),
                    t.elapsed().as_secs_f64()
                ));
                Some(nv)
            }
            H3Nvfp4Plan::ModelOpt(opts) => {
                let t = Instant::now();
                let nv = Nvfp4Dit::open_modelopt(&root.join("transformer"), cfg, *opts)?;
                crate::rt::log::info(format_args!(
                    "h3 nvfp4: ModelOpt {} ({} blocks x 2 NVFP4 FFN linears, {:.2} GiB; attention {}; FFN activation scale {}; checked in {:.1}s)",
                    nv.path().display(),
                    cfg.num_layers,
                    nv.linear_bytes(cfg, false) as f64 / f64::from(1u32 << 30),
                    if opts.attention { "NVFP4 (quantized at load)" } else { "as FASTVIDEO_H3_QUANT" },
                    if opts.static_act { "static (calibrated)" } else { "dynamic" },
                    t.elapsed().as_secs_f64()
                ));
                Some(nv)
            }
        };
        let dit_tree = match &self.dit_tree {
            Some(dir) => Some(crate::h3::dit_tree::TreeReader::open(dir)?),
            None => None,
        };
        if dit_tree.is_none() {
            let skip_adaln = self.skip_adaln;
            let adaln = |k: &str| k.contains(".adaln_proj.");
            let refiner = |k: &str| {
                k.starts_with("token_refiner")
                    || k.starts_with("context_embedder")
                    || k.starts_with("refiner.")
            };
            let table = |k: &str| {
                k.starts_with("time_embedder")
                    || k.starts_with("norm_out.linear")
                    || (!skip_adaln && adaln(k))
            };
            // Under the Trim's NVFP4 file the snapshot's block linears are never
            // read; a ModelOpt DiT is the snapshot's own `transformer/`.
            let nv = nvfp4_dit
                .as_ref()
                .is_some_and(|n| n.layout() == crate::h3::nvfp4::Nvfp4Layout::Comfy);
            let nv_linear = |k: &str| {
                nv && k.starts_with("transformer_blocks.")
                    && (k.contains(".attn.to_") || k.contains(".ff.net."))
            };
            let blocks = |k: &str| {
                (k.starts_with("transformer_blocks.") || k.starts_with("blocks."))
                    && !adaln(k)
                    && !nv_linear(k)
            };
            let rest = |k: &str| !adaln(k) && !nv_linear(k);
            map.prefetch_groups(&[&refiner, &table, &blocks, &rest]);
            if let Some(nv) = nvfp4_dit.as_ref().filter(|_| nv) {
                nv.map().prefetch_all();
            }
        }
        let not_encoder = |k: &str| !k.starts_with("encoder.");
        let vae_map = match &self.taeh3 {
            Some(_) => None,
            None => self.vae_source.open().ok(),
        };
        if let Some(m) = &vae_map {
            m.prefetch_groups(&[&not_encoder]);
        }
        let audio_vae_map = WeightMap::open(&self.audio_dir).ok();
        if let Some(m) = &audio_vae_map {
            m.prefetch_groups(&[&not_encoder]);
        }
        Ok(H3Opened {
            io_base,
            io_timer,
            text_map,
            map,
            nvfp4_dit,
            dit_tree,
            vae_map,
            audio_vae_map,
        })
    }

    /// Stage 3: upload, quantize and offload onto the device, in the load's
    /// order: the resident text encoder (while the card is otherwise empty:
    /// its FP8 quantization runs with f32 transients), the image-to-video
    /// tower, the refiner and the DiT at the residency the free memory then
    /// allows (the planned adapter merged into each linear as it loads,
    /// before quantization), the VDN branch, the decoders. Every component
    /// is booked in the device ledger.
    pub fn place(&self, opened: H3Opened) -> Result<H3Placed> {
        let root = self.root.as_path();
        let options = &self.options;
        let cfg = &self.cfg;
        let H3Opened {
            io_base,
            io_timer,
            text_map,
            map,
            nvfp4_dit,
            dit_tree,
            mut vae_map,
            mut audio_vae_map,
        } = opened;
        let mut load_timings = H3LoadTimings::default();
        let timed = |slot: &mut f64, timer: Instant| *slot = timer.elapsed().as_secs_f64();
        let mut booking = crate::rt::ledger::Booking::default();

        let timer = Instant::now();
        let text_encoder = booking.track(
            crate::rt::ledger::TEXT_ENCODER,
            || -> Result<Option<Box<dyn HiddenStateEncoder + Send>>> {
                Ok(match options.text_encoder {
                    TextEncoderChoice::Recovered8b => {
                        let text_root = options.text_root.as_deref().unwrap_or(root);
                        Some(Box::new(crate::h3::recovered_8b::Recovered8bEncoder::load(
                            text_root,
                        )?))
                    }
                    other => match (other.precision(), text_map.as_ref()) {
                        (Some(precision), Some(map)) => Some(Box::new(
                            crate::h3::text::load_resident_encoder_from(map, precision)?,
                        )),
                        _ => None,
                    },
                })
            },
            |r| match r {
                Ok(Some(e)) => Some(e.resident_bytes()),
                _ => None,
            },
        )?;
        drop(text_map);
        if text_encoder.is_some() {
            timed(&mut load_timings.text_encoder_s, timer);
            crate::rt::weights::log_load_io(
                "h3 text_encoder",
                &io_base,
                io_timer.elapsed().as_secs_f64(),
            );
        }
        // What the DiT placement decides on: the card as the encoder left it.
        let free = if text_encoder.is_some() {
            crate::rt::device::free_memory().map(|(free, _)| free)
        } else {
            self.facts.free
        };

        // The adapter's factors, read and checked against the DiT's map; the
        // DiT's linear loads merge them.
        let mut lora = match &self.adapter {
            H3AdapterPlan::None | H3AdapterPlan::Tree => None,
            H3AdapterPlan::Preview(variant) => {
                // FastH3 Preview v1 = base MiniMax-H3 + one Preview LoRA
                // (`run_fasth3_lora_preview_*_datafree.sh`).
                let has_gate = map.has_tensor("transformer_blocks.0.attn.to_gate_compress.weight");
                let has_json = [
                    "fastvideo_inference.json",
                    "transformer/fastvideo_inference.json",
                ]
                .iter()
                .any(|rel| root.join(rel).is_file());
                check_preview_base(has_gate, has_json).map_err(msg)?;
                let strength =
                    preview_lora_strength(std::env::var(FASTH3_LORA_STRENGTH_ENV).ok().as_deref())
                        .map_err(msg)?;
                let spec = variant.adapter_spec(strength);
                let recipe = options.recipe.as_deref().unwrap_or_default();
                let path = spec.resolve(root, options.adapter.as_deref()).map_err(|e| {
                    msg(format!(
                        "recipe {recipe} needs the FastH3 Preview v1 {} adapter (FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA): {e}",
                        variant.subdir()
                    ))
                })?;
                let fuse = crate::h3::lora::H3LoraFuse::open(&map, &path, spec.alpha, spec.scale)?;
                variant
                    .check_adapter(&fuse.replacement_params())
                    .map_err(|e| msg(format!("{}: {e}", path.display())))?;
                crate::rt::log::info(format_args!(
                    "h3 fasth3-preview {}: fuse {} pairs rank={} strength={} diffs={} set_weight={} ({})",
                    variant.subdir(),
                    fuse.pairs_total,
                    fuse.rank,
                    fuse.effective_scale(),
                    fuse.diffs_total,
                    fuse.replacements_total,
                    path.display()
                ));
                Some(fuse)
            }
            H3AdapterPlan::SolH3 { spec, spark } => {
                let adapter_root = match (options.ref2va, options.ref_root.as_deref()) {
                    (true, Some(r)) => r,
                    _ => root,
                };
                let path = spec
                    .resolve(adapter_root, options.adapter.as_deref())
                    .map_err(msg)?;
                let fuse = crate::h3::lora::H3LoraFuse::open(&map, &path, spec.alpha, spec.scale)?;
                crate::rt::log::info(format_args!(
                    "h3 sol-h3: fuse {} pairs rank={} alpha={} scale={} diffs={} ({})",
                    fuse.pairs_total,
                    fuse.rank,
                    fuse.alpha,
                    fuse.effective_scale(),
                    fuse.diffs_total,
                    path.display()
                ));
                if *spark {
                    let contract = &self.contract;
                    crate::rt::log::info(format_args!(
                        "h3 sol-h3-spark: stage-1 VSA {} tile {} BF16 FastH3_VSA_DataFree strength {}; DiT {} (upstream: W8A8 after the bf16 LoRA merge; FASTVIDEO_H3_QUANT=w8a8). Draft {}x{} {}f. H3×2 upscaler and H3-to-LTX adapter run when their checkpoints are set. Joint 3-step LTX refine uses the fixed prompt and cached Gemma",
                        contract.vsa_sparsity,
                        contract.vsa_tile_size,
                        spec.scale,
                        self.quant
                            .as_ref()
                            .map(|m| m.as_str())
                            .unwrap_or("invalid FASTVIDEO_H3_QUANT"),
                        fastvideo_models::h3::sol::SPARK_DRAFT_WIDTH,
                        fastvideo_models::h3::sol::SPARK_DRAFT_HEIGHT,
                        fastvideo_models::h3::sol::SPARK_DRAFT_FRAMES,
                    ));
                }
                Some(fuse)
            }
            H3AdapterPlan::Vdn => match &self.vdn {
                // VDN-H3: the checkpoint's LoRA adapters merged into the base
                // projections (M:merge_lora_state, scale 1); the branch attaches
                // after the blocks load.
                Some(ck) => {
                    let contract = &self.contract;
                    let fuse = crate::h3::lora::H3LoraFuse::open_vdn(&map, &ck.adapters)?;
                    crate::rt::log::info(format_args!(
                        "h3 vdn: {} ({}): fuse {} pairs max rank {} scale 1, chunk {} radius {}{}",
                        ck.dir.display(),
                        ck.adapter_names().join(" + "),
                        fuse.pairs_total,
                        fuse.rank,
                        ck.transform.chunk,
                        ck.transform.radius,
                        if ck.is_distilled() || contract.transformer_forwards >= 40 {
                            String::new()
                        } else {
                            format!(
                                "; NOTE: {} forwards on the 50-step stage-B model (the 8-step model adds adapters/turbo)",
                                contract.transformer_forwards
                            )
                        }
                    ));
                    Some(fuse)
                }
                None => None,
            },
            H3AdapterPlan::Plug(plug) => {
                let fuse = crate::h3::lora::H3LoraFuse::open_plug(
                    &map,
                    root,
                    plug,
                    options.adapter.as_deref(),
                )?;
                crate::rt::log::info(format_args!(
                    "h3 {}: fuse {} pairs rank={} alpha={} scale={} ({}; licence: {})",
                    plug.name,
                    fuse.pairs_total,
                    fuse.rank,
                    fuse.alpha,
                    fuse.effective_scale(),
                    fuse.path,
                    plug.licence
                ));
                Some(fuse)
            }
            H3AdapterPlan::Turbo { turbo, forwards } => {
                // MiniMax-H3 Turbo LoRA (the MiniMax-H3-Turbo-Lora Space): base
                // MiniMax-H3 + one adapter folded into the bf16 weights here, before
                // any quantization of the linears (`load_linear` merges, the block
                // loader quantizes what it returns).
                let turbo = *turbo;
                let strength = turbo_strength(
                    options.adapter_strength,
                    std::env::var(TURBO_LORA_STRENGTH_ENV).ok().as_deref(),
                )
                .map_err(msg)?;
                let path = turbo
                    .resolve(root, options.adapter.as_deref())
                    .map_err(msg)?;
                let fuse = crate::h3::lora::H3LoraFuse::open_turbo(&map, &path, turbo, strength)?;
                crate::rt::log::info(format_args!(
                    "h3 {}: fuse {} pairs rank={} alpha={} multiplier={} strength={strength} forwards={forwards} ({}; {} @ {}; licence {})",
                    turbo.stem(),
                    fuse.pairs_total,
                    fuse.rank,
                    turbo.alpha().map_or("rank".to_owned(), |a| a.to_string()),
                    fuse.effective_scale(),
                    fuse.path,
                    turbo.hub_repo(),
                    &turbo.hub_revision()[..7],
                    turbo.licence()
                ));
                Some(fuse)
            }
        };
        // The multimodal (FL2VA / Ref2VA) text encoder: the vision tower
        // beside the resident language model, decided before the DiT
        // placement so the placement sees what it leaves.
        let (multimodal, free) = {
            let lm = text_encoder
                .as_deref()
                .and_then(|e| e.decoder())
                .map(|d| d.precision());
            let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
            let (load, why) = self.i2v_tower(lm.is_some(), text_encoder.is_some(), free)?;
            if load {
                let timer = Instant::now();
                let text_root = options.text_root.as_deref().unwrap_or(root);
                let before = crate::rt::device::free_memory().map(|(f, _)| f);
                let mm = crate::h3::text::MultimodalEncoder::load(text_root)?;
                let after = crate::rt::device::free_memory().map(|(f, _)| f);
                let held = before.zip(after).map(|(b, a)| b.saturating_sub(a));
                load_timings.vision_s = timer.elapsed().as_secs_f64();
                crate::rt::log::info(format_args!(
                    "h3 i2v encoder: resident ({why}): vision tower {} GiB loaded in {:.1} s; the language model is the resident text encoder ({:?}), {:.1} GiB free now",
                    held.map_or("?".into(), |b| format!("{:.2}", gib(b))),
                    load_timings.vision_s,
                    options.text_encoder,
                    after.map_or(0.0, gib),
                ));
                (Some(mm), after.or(free))
            } else {
                crate::rt::log::info(format_args!(
                    "h3 i2v encoder: stream ({why}): FL2VA/Ref2VA requests read the vision tower and the {} language model from the volume per request",
                    match options.text_encoder.precision() {
                        Some(crate::llm::WeightPrecision::Fp8Rows) => "fp8",
                        _ => "bf16",
                    }
                ));
                (None, free)
            }
        };
        let residency = {
            let residency = self.residency(free)?;
            crate::rt::log::info(format_args!(
                "h3 dit offload: {} -> {} (resident plan {:.1} GiB, {} free before load)",
                self.dit_policy.as_str(),
                residency.as_str(),
                self.resident_need as f64 / f64::from(1u32 << 30),
                free.map_or("unknown".into(), |f| format!(
                    "{:.1} GiB",
                    f as f64 / f64::from(1u32 << 30)
                )),
            ));
            residency
        };
        let with_gate = self.with_gate;
        let schedule = &self.schedule;
        let timer = Instant::now();
        let refiner = booking.track(
            crate::rt::ledger::TEXT_REFINER,
            || match &dit_tree {
                Some(tree) => H3TextRefiner::from_tree(cfg, tree, residency),
                None => H3TextRefiner::load_with_residency(cfg, &map, &mut lora, residency),
            },
            |_| None,
        )?;
        timed(&mut load_timings.refiner_s, timer);
        let timer = Instant::now();
        let mut model = booking.track(
            crate::rt::ledger::DIT_NONLINEAR,
            || match &dit_tree {
                Some(tree) => {
                    H3Transformer::from_tree(cfg.clone(), tree, schedule, with_gate, residency)
                }
                None => H3Transformer::load_with_nvfp4(
                    cfg.clone(),
                    &map,
                    schedule,
                    with_gate,
                    options.adaln_cache.as_deref(),
                    &mut lora,
                    residency,
                    nvfp4_dit.as_ref(),
                ),
            },
            |_| None,
        )?;
        if let Some(ck) = &self.vdn {
            let branch = WeightMap::open_files(std::slice::from_ref(&ck.branch))?;
            model.attach_vdn(&branch)?;
        }
        // Every target of the merged adapter was applied by the linear loads.
        if let Some(fuse) = lora.as_ref() {
            fuse.finish()?;
        }
        timed(&mut load_timings.dit_s, timer);
        load_timings.dit_from_tree = dit_tree.is_some();
        drop(dit_tree);
        drop(map);
        crate::rt::weights::log_load_io("h3 dit", &io_base, io_timer.elapsed().as_secs_f64());
        // Streamed: the decoders are loaded by each decode (the reference makes
        // the full VAE resident only for decoding).
        let (video_vae, audio_vae) = if residency.is_streamed() {
            (None, None)
        } else {
            let timer = Instant::now();
            let video_vae = booking.track(
                crate::rt::ledger::VAE,
                || {
                    load_video_decoder_from(
                        &self.vae_source,
                        options.taeh3.as_deref(),
                        vae_map.take(),
                    )
                },
                |_| None,
            )?;
            timed(&mut load_timings.video_vae_s, timer);
            let timer = Instant::now();
            let audio_vae = booking.track(
                crate::rt::ledger::AUDIO_VAE,
                || load_audio_decoder_from(&self.audio_dir, audio_vae_map.take()),
                |_| None,
            )?;
            timed(&mut load_timings.audio_vae_s, timer);
            (Some(video_vae), Some(audio_vae))
        };
        drop((vae_map, audio_vae_map));
        crate::rt::weights::log_load_io("h3 total", &io_base, io_timer.elapsed().as_secs_f64());
        let modelopt_nvfp4 = matches!(self.nvfp4, H3Nvfp4Plan::ModelOpt(_));
        let comfy_nvfp4 = matches!(self.nvfp4, H3Nvfp4Plan::Comfy(_));
        let vsa_fp4 = vsa_fp4_choice(
            modelopt_nvfp4,
            comfy_nvfp4,
            options
                .recipe
                .as_deref()
                .is_some_and(fastvideo_models::h3::config::is_trim_nvfp4_fp4attn_recipe),
        )?;
        Ok(H3Placed {
            load_timings,
            booking,
            text_encoder,
            multimodal,
            residency,
            refiner,
            model,
            lora,
            video_vae,
            audio_vae,
            vsa_fp4,
        })
    }

    /// Stage 4: the passes over the placed model, then the pipeline: the
    /// adapter merge's report (where it ran, and the device merge's
    /// verification counts), the step-output cache (TeaCache).
    pub fn fuse(self, placed: H3Placed) -> Result<H3Pipeline> {
        let H3Placed {
            load_timings,
            booking,
            text_encoder,
            multimodal,
            residency,
            refiner,
            model,
            lora,
            video_vae,
            audio_vae,
            vsa_fp4,
        } = placed;
        // The step-output seam: TeaCache over the recipe's forwards unless
        // the technique names its own count.
        if let Some(state) = self.teacache {
            if self.techniques.teacache_is_official() {
                crate::rt::log::info(format_args!(
                    "{}",
                    fastvideo_models::h3::sol::TEACACHE_APPLIED
                ));
            } else {
                crate::rt::log::info(format_args!(
                    "h3 teacache: threshold {} retain {} cooldown {} coefficients {:?}",
                    state.threshold, state.retain_steps, state.cooldown_steps, state.coefficients
                ));
            }
            model.enable_teacache(state)?;
        }
        #[cfg(feature = "cuda")]
        if lora.is_some() {
            use crate::h3::transformer::lora_device;
            crate::rt::log::info(format_args!(
                "h3 lora merge: {}",
                if lora_device::enabled() {
                    "device"
                } else {
                    "host"
                }
            ));
            if lora_device::verify_enabled() {
                let (ok, bad) = lora_device::verify_counts();
                crate::rt::log::info(format_args!(
                    "load/verify h3 lora {{\"device_equal\":{ok},\"device_different\":{bad}}}"
                ));
            }
        }
        drop(lora);
        Ok(H3Pipeline {
            root: self.root,
            vae_source: self.vae_source,
            options: self.options,
            cfg: self.cfg,
            contract: self.contract,
            techniques: self.techniques,
            base_dense: self.base_dense,
            schedule: self.schedule,
            refiner,
            model,
            video_vae,
            audio_vae,
            residency,
            text_encoder: std::sync::Mutex::new(text_encoder),
            auto_text_encoder: self.auto_text_encoder,
            multimodal,
            i2v_stream: false,
            load_timings,
            booking: std::sync::Mutex::new(booking),
            vdn: self.vdn.map(|c| c.transform),
            vsa_fp4,
            #[cfg(feature = "cuda")]
            graphs: std::sync::Mutex::new(None),
        })
    }
}
