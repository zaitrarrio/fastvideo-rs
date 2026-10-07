//! `CapabilityTable`: every servable model's `ModelCaps`, its engine recipe,
//! and the quality tiers per family (design §3.6, owner decisions §0.3, §0.5).
//!
//! Tiers ([`Tier`], from `fastvideo-protocol`, carried on `ModelCaps::tier`):
//!
//! - **Max** (`h3-max`, `ltx-pro`): the family's highest-quality configuration
//!   (full step count, non-lossy attention route, full VAE). fal
//!   `minimax/h3-max/*`, MiniMax `MiniMax-H3-Max` and LTX `ltx-2-5-pro` /
//!   `ltx-2-3-pro` map here.
//! - **Turbo** (`h3-turbo`, `ltx-turbo`, `wan-turbo`): the fastest
//!   configuration that still passes the quality gate (e.g. FastH3 4-step
//!   VSA; LTX-2.5 distilled two-stage Sol).
//! - **Draft** (`h3-draft`, `ltx-draft`, `wan-draft`, §0.6): faster
//!   configurations that do not pass the gate (previews).
//!
//! The CUDA catalog (`crate::cuda::caps`) documents each tier's recipe.
//!
//! A backend tags each model's caps with its tier and recipe name
//! (`ModelCaps::with_tier`) and describes the recipe in detail through
//! `EngineBackend::recipe`. The table keeps one model per (family, tier):
//! the first tagged one, unless `EngineConfig::tier_overrides` rebinds the
//! tier alias, in which case the tags move with it so
//! `fastvideo_protocol::resolve_tier` agrees with the table.
//!
//! Technique profiles (§0.5): when a model runs the Sol-H3 4-step recipe, the
//! table selects [`SOL_H3_4STEP_PROFILE`] (the tau-ladder engine route, 1.53x
//! denoise, gate PASS) unless the recipe names a profile explicitly (e.g.
//! [`SOL_H3_4STEP_DENSE_PROFILE`]). The CUDA backend (WP-11) installs
//! [`Recipe::profile`] at load.

use std::collections::BTreeMap;

use fastvideo_models::h3::lora::{is_sol_h3_recipe, is_sol_h3_spark_recipe, sol_h3_forces_ref2va};
use fastvideo_protocol::{ApiError, Family, FpsCaps, ModelCaps, ModelId, Task, Tier};
use serde::{Deserialize, Serialize};

/// The profile Sol-H3 4-step serves with (design §0.5): Sol engine route,
/// tau 1.0 / 1.25 / 1.5 on forwards 1-3 (`profiles/h3/…`).
pub const SOL_H3_4STEP_PROFILE: &str = "h3/sol_h3_4step_engine_ladder";
/// The dense Sol-H3 4-step route, selectable as an explicit profile.
pub const SOL_H3_4STEP_DENSE_PROFILE: &str = "h3/sol_h3_4step";
/// LongLive-Plug `h3-plug-4step` with h3-max's techniques (Sol engine
/// ladder, MXFP8; docs/serve/research-longlive.md §12.8). In no catalog tier:
/// a recipe `h3-plug-4step` with this profile is the h3-max alternative the
/// owner can switch to.
pub const PLUG_H3_4STEP_PROFILE: &str = "h3/plug_h3_4step_engine_ladder";

/// The LTX-2 frame rates the engine serves (serve E4): 24, and 25 / 48 / 50,
/// validated at 1080p on the LTX-2.5 distilled two-stage (exact frame count,
/// mp4 rate and audio track; `artifacts/serve/e4-ltx-fps/benchmark.json`).
/// The model is conditioned on the rate (RoPE time and audio length), so
/// these are generated rates, not container-only ones.
pub const LTX_FPS: [u32; 4] = [24, 25, 48, 50];

/// [`LTX_FPS`] as caps (default 24) for an LTX-2 backend's `ModelCaps::fps`.
pub fn ltx_fps_caps() -> FpsCaps {
    FpsCaps {
        allowed: LTX_FPS.to_vec(),
        default: 24,
        container_only: false,
    }
}

/// The canonical tier alias for `family`: `h3-max`, `h3-turbo`, `h3-draft`,
/// `ltx-pro`, `ltx-turbo`, `ltx-draft`, `wan-max`, `wan-turbo`, `wan-draft`.
/// `None` for MMAudio.
pub fn tier_alias(family: Family, tier: Tier) -> Option<&'static str> {
    Some(match (family, tier) {
        (Family::H3, Tier::Max) => "h3-max",
        (Family::H3, Tier::Turbo) => "h3-turbo",
        (Family::H3, Tier::Draft) => "h3-draft",
        (Family::Ltx2, Tier::Max) => "ltx-pro",
        (Family::Ltx2, Tier::Turbo) => "ltx-turbo",
        (Family::Ltx2, Tier::Draft) => "ltx-draft",
        (Family::Wan, Tier::Max) => "wan-max",
        (Family::Wan, Tier::Turbo) => "wan-turbo",
        (Family::Wan, Tier::Draft) => "wan-draft",
        (Family::MmAudio | Family::Loopback, _) => return None,
    })
}

/// Parses a canonical tier alias back to `(family, tier)`.
pub fn parse_tier_alias(alias: &str) -> Option<(Family, Tier)> {
    [Family::H3, Family::Ltx2, Family::Wan]
        .into_iter()
        .flat_map(|f| [Tier::Max, Tier::Turbo, Tier::Draft].into_iter().map(move |t| (f, t)))
        .find(|(f, t)| tier_alias(*f, *t) == Some(alias))
}

/// The technique profile the engine selects for a recipe when none is named
/// (design §0.5). Only Sol-H3 4-step (not ref2va, not Spark) has one.
pub fn default_profile(family: Family, recipe: &str) -> Option<&'static str> {
    (family == Family::H3
        && is_sol_h3_recipe(recipe)
        && !sol_h3_forces_ref2va(recipe)
        && !is_sol_h3_spark_recipe(recipe))
        .then_some(SOL_H3_4STEP_PROFILE)
}

/// The internal configuration a model id stands for. Serialized into
/// `/fv/v1/capabilities` and available for response metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recipe {
    /// The pipeline recipe name (`ModelCaps::recipe`), e.g. `4step-vsa`, `sol-h3`.
    pub name: String,
    /// Technique profile installed at load (`h3/sol_h3_4step_engine_ladder`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// Denoise steps, when fixed by the recipe.
    pub steps: Option<u32>,
    /// Attention route, e.g. `dense`, `vsa`, `sol`.
    pub attention: String,
    /// VAE route, e.g. `full`, `tiny`.
    pub vae: String,
    /// One line on what the recipe is and why it has its tier.
    pub summary: String,
}

impl Recipe {
    /// Sol-H3 4-step as served: the tau-ladder engine route (§0.5).
    pub fn sol_h3_4step() -> Self {
        Self {
            name: "sol-h3".into(),
            profile: Some(SOL_H3_4STEP_PROFILE.into()),
            steps: Some(4),
            attention: "sol-engine-tau-ladder".into(),
            vae: "full".into(),
            summary: "Sol-H3 4-step, Sol engine route tau 1.0/1.25/1.5 on forwards 1-3 \
                      (1.53x denoise vs dense, quality gate PASS)"
                .into(),
        }
    }

    /// Sol-H3 4-step on the dense route (explicit profile; parity reference).
    pub fn sol_h3_4step_dense() -> Self {
        Self {
            profile: Some(SOL_H3_4STEP_DENSE_PROFILE.into()),
            attention: "dense".into(),
            summary: "Sol-H3 4-step, dense attention (explicit profile)".into(),
            ..Self::sol_h3_4step()
        }
    }

    /// LongLive-Plug H3 4-step on h3-max's techniques (§12.8 of
    /// docs/serve/research-longlive.md). Not in the catalog; what an
    /// `h3-max` switch to the Plug adapter would serve.
    pub fn plug_h3_4step_engine_ladder() -> Self {
        Self {
            name: "h3-plug-4step".into(),
            profile: Some(PLUG_H3_4STEP_PROFILE.into()),
            summary: "LongLive-Plug H3 4-step (fresh-noise sampler), Sol engine route \
                      tau 1.0/1.25/1.5 on forwards 1-3, MXFP8"
                .into(),
            ..Self::sol_h3_4step()
        }
    }
}

/// One tier alias bound to a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierBinding {
    pub alias: String,
    pub family: Family,
    pub tier: Tier,
    pub model: ModelId,
}

/// One model as served: caps, recipe, and the executors that can run it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub caps: ModelCaps,
    pub recipe: Recipe,
    /// Indices of the executors (backends) that serve this model.
    pub executors: Vec<usize>,
}

/// Every model the engine serves (design §3.6 `CapabilityTable`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilityTable {
    models: BTreeMap<ModelId, ModelEntry>,
    tiers: BTreeMap<String, TierBinding>,
}

impl CapabilityTable {
    /// Builds the table from each backend's `(caps, recipe)` list, in
    /// executor order, then binds tiers and applies `overrides`
    /// (tier alias -> model id).
    ///
    /// Normalization: `caps.recipe` and `recipe.name` fill each other when
    /// one is missing (they must agree when both are set), and a recipe
    /// without a profile gets [`default_profile`].
    pub fn build(
        per_executor: Vec<Vec<(ModelCaps, Recipe)>>,
        overrides: &BTreeMap<String, ModelId>,
    ) -> Result<Self, ApiError> {
        let mut models: BTreeMap<ModelId, ModelEntry> = BTreeMap::new();
        let mut order: Vec<ModelId> = Vec::new();
        for (idx, list) in per_executor.into_iter().enumerate() {
            for (mut caps, mut recipe) in list {
                match (&caps.recipe, recipe.name.is_empty()) {
                    (Some(n), true) => recipe.name = n.clone(),
                    (None, false) => caps.recipe = Some(recipe.name.clone()),
                    (Some(n), false) if *n != recipe.name => {
                        return Err(ApiError::internal(format!(
                            "model `{}`: caps recipe `{n}` differs from recipe `{}`",
                            caps.id, recipe.name
                        )))
                    }
                    _ => {}
                }
                if recipe.profile.is_none() {
                    recipe.profile = default_profile(caps.family, &recipe.name).map(str::to_owned);
                }
                match models.get_mut(&caps.id) {
                    Some(e) => {
                        if e.caps != caps {
                            return Err(ApiError::internal(format!(
                                "model `{}` is declared with different caps by two backends",
                                caps.id
                            )));
                        }
                        e.executors.push(idx);
                    }
                    None => {
                        order.push(caps.id.clone());
                        models.insert(
                            caps.id.clone(),
                            ModelEntry {
                                caps,
                                recipe,
                                executors: vec![idx],
                            },
                        );
                    }
                }
            }
        }
        let mut tiers: BTreeMap<String, TierBinding> = BTreeMap::new();
        for id in &order {
            let c = &models[id].caps;
            let Some(tier) = c.tier else { continue };
            let Some(alias) = tier_alias(c.family, tier) else {
                continue;
            };
            // The first tagged model binds the tier, except that a model
            // serving text-to-video displaces a task companion bound first
            // (`fastvideo_protocol::resolve_tier` makes the same choice).
            let displaces = |b: &TierBinding| {
                c.supports(Task::T2V) && !models[&b.model].caps.supports(Task::T2V)
            };
            match tiers.get(alias) {
                Some(b) if !displaces(b) => {}
                _ => {
                    tiers.insert(
                        alias.to_owned(),
                        TierBinding {
                            alias: alias.to_owned(),
                            family: c.family,
                            tier,
                            model: c.id.clone(),
                        },
                    );
                }
            }
        }
        for (alias, model) in overrides {
            let (family, tier) = parse_tier_alias(alias).ok_or_else(|| {
                ApiError::internal(format!("unknown tier alias `{alias}` in tier overrides"))
            })?;
            let fam = models
                .get(model)
                .ok_or_else(|| {
                    ApiError::internal(format!("tier `{alias}` bound to unknown model `{model}`"))
                })?
                .caps
                .family;
            if fam != family {
                return Err(ApiError::internal(format!(
                    "tier `{alias}` bound to model `{model}` of another family"
                )));
            }
            tiers.insert(
                alias.clone(),
                TierBinding {
                    alias: alias.clone(),
                    family,
                    tier,
                    model: model.clone(),
                },
            );
        }
        // Tags follow the bindings: the bound model of a family carries
        // each tier, and so does a task companion of it (a model serving a
        // task the bound one does not, e.g. the H3 Ref2VA DiT), which
        // `fastvideo_protocol::route_task` finds by (family, tier).
        let bound_tasks: BTreeMap<ModelId, std::collections::BTreeSet<Task>> = tiers
            .values()
            .map(|b| (b.model.clone(), models[&b.model].caps.tasks.clone()))
            .collect();
        for e in models.values_mut() {
            let c = &mut e.caps;
            let bound = |t: Tier| {
                tier_alias(c.family, t)
                    .and_then(|a| tiers.get(a))
                    .is_some_and(|b| {
                        b.model == c.id
                            || (c.tier == Some(t)
                                && !c.supports(Task::T2V)
                                && c.tasks.iter().any(|k| !bound_tasks[&b.model].contains(k)))
                    })
            };
            let held: Vec<Tier> = [Tier::Max, Tier::Turbo, Tier::Draft]
                .into_iter()
                .filter(|t| bound(*t))
                .collect();
            c.tier = match c.tier {
                Some(t) if held.contains(&t) => Some(t),
                _ => held.first().copied(),
            };
        }
        Ok(Self { models, tiers })
    }

    pub fn get(&self, id: &ModelId) -> Option<&ModelCaps> {
        self.models.get(id).map(|e| &e.caps)
    }

    pub fn entry(&self, id: &ModelId) -> Option<&ModelEntry> {
        self.models.get(id)
    }

    pub fn recipe(&self, id: &ModelId) -> Option<&Recipe> {
        self.models.get(id).map(|e| &e.recipe)
    }

    /// Every model's caps, by id.
    pub fn models(&self) -> impl Iterator<Item = &ModelCaps> {
        self.models.values().map(|e| &e.caps)
    }

    pub fn entries(&self) -> impl Iterator<Item = &ModelEntry> {
        self.models.values()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// The model bound to `family` at `tier`.
    pub fn tier(&self, family: Family, tier: Tier) -> Option<&ModelId> {
        let alias = tier_alias(family, tier)?;
        self.tiers.get(alias).map(|b| &b.model)
    }

    /// Every bound tier.
    pub fn tier_bindings(&self) -> impl Iterator<Item = &TierBinding> {
        self.tiers.values()
    }

    /// `(alias, model id)` pairs to merge into a serve alias map.
    /// `served: <model ids>; aliases: <name> -> <model>, …` (tier aliases
    /// and served names other than the id): what a not-served error tells
    /// the client this server does serve.
    pub fn served_summary(&self) -> String {
        let ids: Vec<&str> = self.models.keys().map(ModelId::as_str).collect();
        let mut aliases: Vec<String> = self.tiers.values().map(|b| format!("{} -> {}", b.alias, b.model.as_str())).collect();
        for e in self.models.values() {
            for n in &e.caps.served_names {
                if n != e.caps.id.as_str() {
                    aliases.push(format!("{n} -> {}", e.caps.id.as_str()));
                }
            }
        }
        let list = |v: &[String]| if v.is_empty() { "none".to_owned() } else { v.join(", ") };
        let ids: Vec<String> = ids.into_iter().map(str::to_owned).collect();
        format!("served: {}; aliases: {}", list(&ids), list(&aliases))
    }

    pub fn tier_aliases(&self) -> Vec<(String, ModelId)> {
        self.tiers
            .values()
            .map(|b| (b.alias.clone(), b.model.clone()))
            .collect()
    }

    /// Resolves a name: a tier alias, a model id, or a served name.
    pub fn resolve(&self, name: &str) -> Option<&ModelCaps> {
        if let Some(b) = self.tiers.get(name) {
            return self.get(&b.model);
        }
        self.models
            .get(&ModelId::new(name))
            .map(|e| &e.caps)
            .or_else(|| self.models().find(|c| c.answers_to(name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(name: &str) -> Recipe {
        Recipe {
            name: name.into(),
            ..Recipe::default()
        }
    }

    fn h3(id: &str, tier: Option<Tier>) -> ModelCaps {
        let c = ModelCaps::h3(id, false);
        match tier {
            Some(t) => c.with_tier(t, format!("r-{id}")),
            None => c,
        }
    }

    #[test]
    fn aliases_round_trip() {
        for f in [Family::H3, Family::Ltx2, Family::Wan] {
            for t in [Tier::Max, Tier::Turbo, Tier::Draft] {
                assert_eq!(parse_tier_alias(tier_alias(f, t).unwrap()), Some((f, t)));
            }
        }
        assert_eq!(parse_tier_alias("h3-draft"), Some((Family::H3, Tier::Draft)));
        assert_eq!(parse_tier_alias("ltx-draft"), Some((Family::Ltx2, Tier::Draft)));
        assert_eq!(tier_alias(Family::MmAudio, Tier::Max), None);
        assert_eq!(parse_tier_alias("nope"), None);
    }

    /// The not-served error's list: model ids and the tier aliases bound to them.
    #[test]
    fn served_summary_lists_ids_and_aliases() {
        let t = CapabilityTable::build(vec![vec![(h3("sol-h3", Some(Tier::Max)), Recipe::default())]], &BTreeMap::new()).unwrap();
        assert_eq!(t.served_summary(), "served: sol-h3; aliases: h3-max -> sol-h3");
        let empty = CapabilityTable::build(vec![], &BTreeMap::new()).unwrap();
        assert_eq!(empty.served_summary(), "served: none; aliases: none");
    }

    #[test]
    fn binds_first_tagged_and_overrides() {
        let a = h3("a", Some(Tier::Max));
        let b = h3("b", Some(Tier::Turbo));
        let c = h3("c", Some(Tier::Max));
        let t = CapabilityTable::build(
            vec![
                vec![(a.clone(), Recipe::default()), (b.clone(), Recipe::default())],
                vec![(c.clone(), Recipe::default()), (a.clone(), Recipe::default())],
            ],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(t.tier(Family::H3, Tier::Max).unwrap().as_str(), "a");
        assert_eq!(t.tier(Family::H3, Tier::Turbo).unwrap().as_str(), "b");
        assert_eq!(t.entry(&"a".into()).unwrap().executors, vec![0, 1]);
        assert_eq!(t.resolve("h3-turbo").unwrap().id.as_str(), "b");
        assert_eq!(t.resolve("c").unwrap().id.as_str(), "c");
        assert!(t.resolve("ltx-pro").is_none());
        // The losing `Max` tag is dropped so resolve_tier agrees.
        assert_eq!(t.get(&"c".into()).unwrap().tier, None);
        let r = fastvideo_protocol::resolve_tier(Family::H3, Tier::Max, t.models()).unwrap();
        assert_eq!(r.id.as_str(), "a");
        assert_eq!(t.recipe(&"a".into()).unwrap().name, "r-a");

        let mut ov = BTreeMap::new();
        ov.insert("h3-max".to_owned(), ModelId::new("c"));
        let t = CapabilityTable::build(
            vec![vec![(a.clone(), Recipe::default()), (h3("c", None), rec("x"))]],
            &ov,
        )
        .unwrap();
        assert_eq!(t.tier(Family::H3, Tier::Max).unwrap().as_str(), "c");
        assert_eq!(t.get(&"c".into()).unwrap().tier, Some(Tier::Max));
        assert_eq!(t.get(&"a".into()).unwrap().tier, None);
        assert_eq!(t.get(&"c".into()).unwrap().recipe.as_deref(), Some("x"));
        let r = fastvideo_protocol::resolve_tier(Family::H3, Tier::Max, t.models()).unwrap();
        assert_eq!(r.id.as_str(), "c");
        ov.insert("ltx-turbo".to_owned(), ModelId::new("a"));
        assert!(CapabilityTable::build(vec![vec![(a, Recipe::default())]], &ov).is_err());
    }

    #[test]
    fn conflicting_declarations_rejected() {
        let a = ModelCaps::h3("a", false);
        let a2 = ModelCaps::h3("a", true);
        assert!(CapabilityTable::build(
            vec![vec![(a.clone(), rec("r"))], vec![(a2, rec("r"))]],
            &BTreeMap::new()
        )
        .is_err());
        let tagged = a.with_tier(Tier::Max, "one");
        assert!(CapabilityTable::build(vec![vec![(tagged, rec("two"))]], &BTreeMap::new()).is_err());
    }

    #[test]
    fn sol_h3_4step_selects_the_ladder_profile() {
        // §0.5: a Sol-H3 4-step entry without a profile gets the ladder.
        let sol = ModelCaps::h3("sol", false).with_tier(Tier::Turbo, "sol-h3");
        let dense = ModelCaps::h3("sol-dense", false);
        let refva = ModelCaps::h3("sol-ref", true);
        let fast = ModelCaps::h3("fast", false);
        let t = CapabilityTable::build(
            vec![vec![
                (sol, Recipe::default()),
                (dense, Recipe::sol_h3_4step_dense()),
                (refva, rec("sol-h3-ref2va")),
                (fast, rec("4step-vsa")),
            ]],
            &BTreeMap::new(),
        )
        .unwrap();
        let p = |id: &str| t.recipe(&ModelId::new(id)).unwrap().profile.clone();
        assert_eq!(p("sol").as_deref(), Some(SOL_H3_4STEP_PROFILE));
        assert_eq!(p("sol-dense").as_deref(), Some(SOL_H3_4STEP_DENSE_PROFILE));
        assert_eq!(p("sol-ref"), None);
        assert_eq!(p("fast"), None);
        assert_eq!(Recipe::sol_h3_4step().profile.as_deref(), Some(SOL_H3_4STEP_PROFILE));
        assert_eq!(default_profile(Family::H3, "sol_h3"), Some(SOL_H3_4STEP_PROFILE));
        assert_eq!(default_profile(Family::H3, "sol-h3-spark"), None);
        assert_eq!(default_profile(Family::Ltx2, "sol-h3"), None);
    }

    #[test]
    fn named_profiles_exist_in_the_repo() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../profiles");
        for p in [SOL_H3_4STEP_PROFILE, SOL_H3_4STEP_DENSE_PROFILE] {
            let f = root.join(format!("{p}.toml"));
            let text = std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
            assert!(text.contains("recipe = \"sol-h3\""), "{p} must run the sol-h3 recipe");
        }
        let p = Recipe::plug_h3_4step_engine_ladder();
        let f = root.join(format!("{}.toml", p.profile.as_deref().unwrap()));
        let text = std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        assert!(text.contains(&format!("recipe = \"{}\"", p.name)), "{PLUG_H3_4STEP_PROFILE}");
    }
}
