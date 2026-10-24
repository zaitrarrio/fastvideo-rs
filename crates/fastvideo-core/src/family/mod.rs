//! The family list (plan B2, review §7.3 item 2, ADR-0013).
//!
//! Every model family implements [`VideoFamily`]: its variant table (the Hub
//! ids it answers to), how an id resolves to one of its variants, and
//! [`VideoFamily::load`], which plans a [`Pipeline`] for a variant. The static
//! [`FAMILIES`] list is the only place a family is registered;
//! [`resolve`] and [`registered`] walk it. A new family is one module here
//! plus its line in `FAMILIES` (docs/ports/CHECKLIST.md).
//!
//! What a run needs is split three ways: [`AvGenerateOptions`] (the common
//! load and output options: weights, device, output, mp4), [`FamilyOptions`]
//! (the options only one family reads) and the common
//! [`Request`](crate::request::Request) (what to generate).

use std::path::PathBuf;

use fastvideo_cudarc::hooks::Hooks;

use crate::error::{FastVideoError, Result};
use crate::request::Request;
use crate::sampling::WorkloadType;

pub mod cosmos;
pub mod dreamx;
pub mod flux;
pub mod flux2;
pub mod gamecraft;
pub mod gen3c;
pub mod glm_image;
pub mod h3;
pub mod hunyuan15;
pub mod hyworld;
pub mod kandinsky5;
pub mod lingbot;
pub mod lingbotworld;
pub mod longcat;
pub mod ltx2;
pub mod matrixgame;
pub mod mmaudio;
pub mod sd35;
pub mod stable_audio;
pub mod wan;
pub mod zimage;

/// Every family, in resolution order: an id is offered to each in turn
/// (after the exact-id pass of the families that detect last, see
/// [`VideoFamily::detects_last`]); `list-models` prints them in this order.
pub static FAMILIES: &[&dyn VideoFamily] = &[
    &wan::Wan,
    &ltx2::Ltx2,
    &h3::H3,
    &hunyuan15::Hunyuan15,
    &kandinsky5::Kandinsky5,
    &cosmos::Cosmos,
    &longcat::LongCat,
    &lingbot::LingBot,
    &gen3c::Gen3C,
    &matrixgame::MatrixGame,
    &dreamx::DreamX,
    &lingbotworld::LingBotWorld,
    &gamecraft::GameCraft,
    &hyworld::HyWorld,
    &zimage::ZImage,
    &sd35::Sd35,
    &flux::Flux,
    &flux2::Flux2,
    &glm_image::GlmImage,
    &stable_audio::StableAudio,
    &mmaudio::MmAudio,
];

/// One variant of a family: a preset and the Hub ids that select it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variant {
    /// The family's name ([`VideoFamily::name`]).
    pub family: &'static str,
    /// The preset (FastVideo's preset name for the ids we port).
    pub preset: &'static str,
    pub hub_ids: &'static [&'static str],
    pub workloads: &'static [WorkloadType],
}

/// A model family.
pub trait VideoFamily: Sync {
    /// The family's name (`wan`, `ltx2`, `h3`, ...), as `list-models` prints it.
    fn name(&self) -> &'static str;

    /// The variant table, in resolution order.
    fn variants(&self) -> Vec<Variant>;

    /// The variant a model id selects: an exact Hub id (case-insensitive)
    /// first, then the family's detectors (tokens of local paths and
    /// preset-like names).
    fn resolve(&self, model_id: &str) -> Option<Variant>;

    /// The variant whose Hub id equals `model_id` (case-insensitive).
    fn resolve_exact(&self, model_id: &str) -> Option<Variant> {
        self.variants()
            .into_iter()
            .find(|v| v.hub_ids.iter().any(|id| id.eq_ignore_ascii_case(model_id)))
    }

    /// Whether this family's detectors run after every other family's (its
    /// exact ids still win first). Wan: its broad detectors (`wanpipeline`,
    /// Hub-path substrings) must not take LTX / H3 ids.
    fn detects_last(&self) -> bool {
        false
    }

    /// Whether the zero-weight `--tiny` graph exists (Wan only).
    fn supports_tiny(&self) -> bool {
        false
    }

    /// Whether the family runs only on CUDA (the CLI's default device for it
    /// is then `cuda` whatever the build).
    fn requires_cuda(&self) -> bool {
        true
    }

    /// Plan a pipeline for `variant`. No device work happens here; the
    /// pipeline loads its weights on its first [`Pipeline::generate`].
    fn load(&self, variant: Variant, options: &AvGenerateOptions) -> Result<Box<dyn Pipeline>>;
}

/// A planned pipeline.
pub trait Pipeline {
    /// The line the CLI prints before generating.
    fn describe(&self, request: &Request) -> Result<String>;

    /// Generate one output. `hooks` carries cancellation and progress to the
    /// pipelines that take them (Wan, LTX-2, H3); the research ports run
    /// without.
    fn generate(&mut self, request: &Request, hooks: Hooks<'_>) -> Result<Output>;
}

/// What a generate call wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    pub family: &'static str,
    pub preset: &'static str,
    pub frame_paths: Vec<String>,
    pub mp4: Option<String>,
    pub wav: Option<String>,
}

/// The common options of a run: where the weights are, where the output goes
/// and on which device. What to generate is the
/// [`Request`](crate::request::Request); what only one family reads is in
/// [`FamilyOptions`].
#[derive(Debug, Clone, Default)]
pub struct AvGenerateOptions {
    /// The model id as given (Wan reads it for the Hub cache and the VSA
    /// detector).
    pub model_id: String,
    /// Diffusers root. `None`: `FASTVIDEO_WEIGHTS` (Wan also the Hub cache).
    pub weights: Option<PathBuf>,
    /// Output directory. `None`: the family default (`outputs/<preset>`;
    /// Wan `outputs`).
    pub output: Option<PathBuf>,
    /// `cpu`, `cuda` or `cuda:N`.
    pub device: String,
    /// Mux the frames to an mp4.
    pub save_mp4: bool,
    pub family: FamilyOptions,
}

/// The options only one family reads. The CLI fills all of them from its
/// flags; each family takes its own and ignores the rest (as it always did).
#[derive(Debug, Clone, Default)]
pub struct FamilyOptions {
    pub wan: wan::WanOptions,
    pub ltx2: ltx2::Ltx2Options,
    pub h3: h3::H3Options,
}

/// A resolved model id: the family and its variant.
#[derive(Clone, Copy)]
pub struct Resolved {
    pub family: &'static dyn VideoFamily,
    pub variant: Variant,
}

impl std::fmt::Debug for Resolved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resolved")
            .field("family", &self.family.name())
            .field("variant", &self.variant)
            .finish()
    }
}

impl Resolved {
    pub fn preset(&self) -> &'static str {
        self.variant.preset
    }

    pub fn workloads(&self) -> &'static [WorkloadType] {
        self.variant.workloads
    }
}

/// Resolve a Hub id, a local path or a preset-like name: the exact ids of
/// the families that detect last, then every other family in [`FAMILIES`]
/// order (exact id, then detectors), then the late families' detectors.
pub fn resolve(model_id: &str) -> Result<Resolved> {
    let found = |f: &'static dyn VideoFamily, v: Variant| Resolved {
        family: f,
        variant: v,
    };
    for &f in FAMILIES.iter().filter(|f| f.detects_last()) {
        if let Some(v) = f.resolve_exact(model_id) {
            return Ok(found(f, v));
        }
    }
    for &f in FAMILIES.iter().filter(|f| !f.detects_last()) {
        if let Some(v) = f.resolve(model_id) {
            return Ok(found(f, v));
        }
    }
    for &f in FAMILIES.iter().filter(|f| f.detects_last()) {
        if let Some(v) = f.resolve(model_id) {
            return Ok(found(f, v));
        }
    }
    Err(FastVideoError::UnknownModel(model_id.to_string()))
}

/// Every variant of every family, in [`FAMILIES`] order.
pub fn registered() -> Vec<(&'static dyn VideoFamily, Variant)> {
    FAMILIES
        .iter()
        .flat_map(|&f| f.variants().into_iter().map(move |v| (f, v)))
        .collect()
}

/// A row of a family's variant table (every family but Wan, whose rows carry
/// its sampler: [`wan::WanModelDefinition`]).
#[derive(Debug, Clone, Copy)]
pub struct HubVariant {
    pub preset: &'static str,
    pub hub_ids: &'static [&'static str],
    pub workloads: &'static [WorkloadType],
    /// Lowercase tokens a local path or name may contain.
    pub match_any: &'static [&'static str],
}

impl HubVariant {
    pub const fn new(
        preset: &'static str,
        hub_ids: &'static [&'static str],
        workloads: &'static [WorkloadType],
        match_any: &'static [&'static str],
    ) -> Self {
        Self {
            preset,
            hub_ids,
            workloads,
            match_any,
        }
    }

    fn variant(&self, family: &'static str) -> Variant {
        Variant {
            family,
            preset: self.preset,
            hub_ids: self.hub_ids,
            workloads: self.workloads,
        }
    }

    fn matches_exact(&self, model_id: &str) -> bool {
        self.hub_ids
            .iter()
            .any(|p| p.eq_ignore_ascii_case(model_id))
    }

    fn matches_fuzzy(&self, model_id: &str) -> bool {
        let value = model_id.to_ascii_lowercase();
        if self.matches_exact(model_id) {
            return true;
        }
        if !self.match_any.is_empty() {
            return self.match_any.iter().any(|t| value.contains(t));
        }
        self.hub_ids
            .iter()
            .any(|p| value.contains(&p.to_ascii_lowercase()))
            || value.contains(&self.preset.replace('_', "-"))
    }
}

/// The variants of a [`HubVariant`] table.
pub fn table_variants(family: &'static str, table: &'static [HubVariant]) -> Vec<Variant> {
    table.iter().map(|d| d.variant(family)).collect()
}

/// Resolve within one table: an exact Hub id, then the first detector match.
pub fn resolve_table(
    family: &'static str,
    table: &'static [HubVariant],
    model_id: &str,
) -> Option<Variant> {
    table
        .iter()
        .find(|d| d.matches_exact(model_id))
        .or_else(|| table.iter().find(|d| d.matches_fuzzy(model_id)))
        .map(|d| d.variant(family))
}

/// The family-level line the CLI prints before a non-Wan generate.
pub(crate) fn describe_av(variant: &Variant, options: &AvGenerateOptions) -> String {
    format!(
        "family={} preset={} device={}",
        variant.family, variant.preset, options.device
    )
}

/// The seed when the request leaves it unset (every family but Wan, whose
/// presets carry their own).
pub const DEFAULT_SEED: u64 = 1024;

/// The family's run: load the variant's weights and generate one output.
pub(crate) type RunFn = fn(&Variant, &AvGenerateOptions, &Request, Hooks<'_>) -> Result<Output>;

/// A pipeline planned by a family whose loader and generate are one call
/// today (every family but Wan; B6 splits the big loaders into plan, open,
/// place, fuse).
pub(crate) struct Planned {
    variant: Variant,
    options: AvGenerateOptions,
    run: RunFn,
}

impl Planned {
    pub(crate) fn boxed(
        variant: Variant,
        options: &AvGenerateOptions,
        run: RunFn,
    ) -> Box<dyn Pipeline> {
        Box::new(Self {
            variant,
            options: options.clone(),
            run,
        })
    }
}

impl Pipeline for Planned {
    fn describe(&self, _request: &Request) -> Result<String> {
        Ok(describe_av(&self.variant, &self.options))
    }

    fn generate(&mut self, request: &Request, hooks: Hooks<'_>) -> Result<Output> {
        (self.run)(&self.variant, &self.options, request, hooks)
    }
}

/// The output directory: the option, else `outputs/<preset>`.
#[cfg(feature = "cuda-cudarc")]
pub(crate) fn output_dir(variant: &Variant, options: &AvGenerateOptions) -> PathBuf {
    options
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("outputs/{}", variant.preset)))
}

pub(crate) fn require_cuda(device: &str) -> Result<()> {
    if !device.to_ascii_lowercase().starts_with("cuda") {
        return Err(FastVideoError::Message(format!(
            "LTX/H3 generate needs --device cuda (got {device}); Mac/CI stay on Wan --tiny or gpucheck"
        )));
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = device;
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LTX/H3".into(),
        ))
    }
    #[cfg(feature = "cuda-cudarc")]
    Ok(())
}

#[cfg(feature = "cuda-cudarc")]
pub(crate) fn weights_root(options: &AvGenerateOptions) -> Result<PathBuf> {
    options
        .weights
        .clone()
        .or_else(|| fastvideo_models::settings::env_var_os("FASTVIDEO_WEIGHTS").map(PathBuf::from))
        .ok_or_else(|| {
            FastVideoError::Message(
                "LTX/H3 generate needs --weights <diffusers root> or FASTVIDEO_WEIGHTS".into(),
            )
        })
}

/// `--dit-offload`, parsed (`None` leaves the choice to the environment).
#[cfg(feature = "cuda-cudarc")]
pub(crate) fn dit_offload(
    name: Option<&str>,
) -> Result<Option<fastvideo_cudarc::rt::offload::DitOffload>> {
    name.map(fastvideo_cudarc::rt::offload::DitOffload::parse)
        .transpose()
        .map_err(FastVideoError::Message)
}

/// The `*.png` files of `output`, sorted.
#[cfg(feature = "cuda-cudarc")]
pub(crate) fn collect_png_frames(output: &std::path::Path) -> Vec<String> {
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

/// `device` resolved on the runtime (errors as the pipelines word them).
#[cfg(feature = "cuda-cudarc")]
pub(crate) fn resolve_device(device: &str) -> Result<()> {
    fastvideo_cudarc::rt::device::resolve_device(device)
        .map(|_| ())
        .map_err(|e| FastVideoError::Message(e.to_string()))
}

/// An error from a pipeline, as the CLI prints it.
#[cfg(feature = "cuda-cudarc")]
pub(crate) fn msg(e: impl std::fmt::Display) -> FastVideoError {
    FastVideoError::Message(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_names_are_unique_and_tables_are_not_empty() {
        let mut names: Vec<&str> = FAMILIES.iter().map(|f| f.name()).collect();
        assert_eq!(names.len(), 21);
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 21, "duplicate family name");
        for f in FAMILIES {
            assert!(!f.variants().is_empty(), "{} has no variant", f.name());
            for v in f.variants() {
                assert_eq!(v.family, f.name());
            }
        }
    }

    #[test]
    fn eighty_three_hub_ids_resolve_to_their_family() {
        let mut n = 0;
        for (f, v) in registered() {
            for id in v.hub_ids {
                n += 1;
                let r = resolve(id).unwrap_or_else(|e| panic!("{id}: {e}"));
                assert_eq!(r.family.name(), f.name(), "{id}");
            }
        }
        assert_eq!(n, 83);
    }

    #[test]
    fn presets_are_unique_within_a_family() {
        for f in FAMILIES {
            let mut p: Vec<&str> = f.variants().iter().map(|v| v.preset).collect();
            let n = p.len();
            p.sort_unstable();
            p.dedup();
            assert_eq!(p.len(), n, "{}", f.name());
        }
    }

    #[test]
    fn non_wan_load_plans_without_device_work() {
        let options = AvGenerateOptions {
            device: "cpu".into(),
            ..AvGenerateOptions::default()
        };
        for (f, v) in registered() {
            if f.name() == "wan" {
                continue;
            }
            let mut p = f
                .load(v, &options)
                .unwrap_or_else(|e| panic!("{}: {e}", v.preset));
            let line = p
                .describe(&Request::text("x"))
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                line,
                format!("family={} preset={} device=cpu", f.name(), v.preset)
            );
            let err = p
                .generate(&Request::text("x"), Hooks::NONE)
                .expect_err("cpu is refused");
            assert!(err.to_string().contains("--device cuda"), "{err}");
        }
    }
}
