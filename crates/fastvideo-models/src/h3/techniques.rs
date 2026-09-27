//! H3 on the technique layer: the recipe's defaults, the active profile and
//! the legacy `FASTVIDEO_H3_*` env flags resolved into one [`H3Techniques`].
//!
//! Precedence per seam, highest first (the env flags keep working and
//! override the profile, which overrides the recipe):
//!
//! | seam | env flag | profile | recipe default |
//! |---|---|---|---|
//! | attention_backend | `FASTVIDEO_H3_SOL_ATTN` | `dense_attention` / `sol_attn` / `vsa` | `sol-h3-rtx`: Sol RTX route; VSA recipes: VSA; else dense |
//! | VSA sparsity / group | `FASTVIDEO_VSA_SPARSITY` / `_GROUP` | `vsa.sparsity` / `.group` | contract / 8 |
//! | step_output | `FASTVIDEO_H3_SOL_CACHE` | `teacache` | off |
//! | ffn / activation precision, decoder, residency, kernels, fusion | their `FASTVIDEO_*` names | techniques' settings | built-in |
//!
//! The last row needs nothing here: those techniques install settings that
//! the deep readers take through [`crate::techniques::settings::var`].
//!
//! With no profile the resolution is, decision for decision, the code it
//! replaced (`h3::sol::recipe_sol_attn_policy`, `teacache_requested`, the
//! pipeline's `FASTVIDEO_VSA_SPARSITY` parse): the off-identity tests below
//! check that for every recipe and flag value.

use super::config::H3InferenceContract;
use super::lora::is_sol_h3_spark_recipe;
use super::sol::{
    policy_technique, recipe_sol_attn_policy, sink_policy, teacache_requested, H3SolAttnPolicy,
    H3TeaCache,
};
use crate::techniques::methods::{
    DenseAttention, SinkMode, SolAttn, TeaCache, TinyDecoder, TinyDecoderKind, Vsa,
};
use crate::techniques::registry::h3_spec;
use crate::techniques::{compose, Plan, Profile, Schedule, Technique, HORIZON};

/// `FASTVIDEO_VSA_GROUP` default for H3 (the query-tile group).
pub const DEFAULT_VSA_GROUP: usize = 8;

/// How the blocks attend, before the request's layout is known.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)] // one per pipeline
pub enum H3Attention {
    /// No sparse route: VSA when the contract has it and nothing forces
    /// dense, dense otherwise (`uses_vsa`).
    Auto,
    /// Dense, whatever the contract says (`dense_attention`, the parity mode).
    Dense,
    /// Sol-Attn with this route and sink.
    Sol(SolAttn),
}

/// The resolved H3 technique set of one pipeline.
#[derive(Debug)]
pub struct H3Techniques {
    /// The recipe the pipeline runs (the caller's, else the profile's).
    pub recipe: Option<String>,
    pub attention: H3Attention,
    /// The named policy of [`H3Attention::Sol`] (by its sink), `Off` otherwise.
    pub sol_policy: H3SolAttnPolicy,
    /// `FASTVIDEO_VSA_SPARSITY` as set (parsed where VSA is built, as before).
    pub vsa_sparsity_env: Option<String>,
    /// The profile's VSA sparsity.
    pub vsa_sparsity: Option<f64>,
    pub vsa_group: usize,
    pub teacache: Option<TeaCache>,
    /// A `taeh3` technique: the video decoder must be TAEH3.
    pub taeh3: Option<TinyDecoder>,
    /// Everything that runs, composed and conflict-checked for H3.
    pub plan: Plan,
    /// Where each seam's choice came from, for the log.
    pub sources: Vec<(&'static str, &'static str)>,
}

fn constant(t: &dyn Technique) -> Result<(), String> {
    match t.enabled().as_const() {
        Some(_) => Ok(()),
        None => Err(format!(
            "h3: technique '{}' is enabled on some steps only; H3 picks it per request (use its own step schedule, e.g. sol_attn.dense_steps)",
            t.name()
        )),
    }
}

impl H3Techniques {
    /// Resolve. `env` is the process environment in production
    /// (`|k| std::env::var(k).ok()`); tests pass a map.
    pub fn resolve(
        recipe: Option<&str>,
        contract: &H3InferenceContract,
        ref2va: bool,
        profile: Option<&Profile>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        if let Some(p) = profile {
            if p.model != "h3" {
                return Err(format!(
                    "technique profile {} is for {}, not h3",
                    p.name, p.model
                ));
            }
        }
        let recipe: Option<String> = recipe
            .map(str::to_owned)
            .or_else(|| profile.and_then(|p| p.recipe.clone()));
        let recipe_ref = recipe.as_deref();
        let spark_recipe = recipe_ref.is_some_and(is_sol_h3_spark_recipe);
        let mut sources = Vec::new();
        let mut items: Vec<Box<dyn Technique>> = Vec::new();

        // Profile techniques this adapter reads itself, and the rest.
        let mut p_attention: Option<Box<dyn Technique>> = None;
        let mut p_teacache: Option<TeaCache> = None;
        let mut taeh3: Option<TinyDecoder> = None;
        if let Some(p) = profile {
            for t in p.plan()?.techniques {
                if t.writes()
                    .contains(&crate::techniques::Seam::AttentionBackend)
                {
                    constant(t.as_ref())?;
                    p_attention = Some(t);
                } else if let Some(tc) = t.downcast_ref::<TeaCache>() {
                    constant(t.as_ref())?;
                    p_teacache = Some(tc.clone());
                } else if let Some(d) = t.downcast_ref::<TinyDecoder>() {
                    if d.kind != TinyDecoderKind::Taeh3 {
                        return Err(format!(
                            "h3: techniques.{} decodes LTX-2 latents; H3 takes taeh3",
                            t.name()
                        ));
                    }
                    constant(t.as_ref())?;
                    taeh3 = Some(d.clone());
                    items.push(t);
                } else {
                    items.push(t);
                }
            }
        }

        // --- attention backend ---
        let env_sol = env("FASTVIDEO_H3_SOL_ATTN");
        let mut vsa_profile: Option<Vsa> = None;
        let profile_attention = match p_attention {
            Some(t) if env_sol.is_none() => Some(t),
            _ => None,
        };
        let attention = if let Some(t) = profile_attention {
            sources.push(("attention_backend", "profile"));
            if let Some(sol) = t.downcast_ref::<SolAttn>() {
                match sol.sink {
                    SinkMode::Suffix if !spark_recipe => {
                        return Err(format!(
                            "h3 sol: sink = \"suffix\" is the Spark draft ladder; recipe {} is not sol-h3-spark",
                            recipe_ref.unwrap_or("auto")
                        ))
                    }
                    SinkMode::Prefix if ref2va => {
                        return Err("h3 sol: the prefix sink (Sol-H3 engine) is T2V/I2V only (Ref2VA needs sol_bsa with a text/audio block mask)".into())
                    }
                    SinkMode::None => {
                        return Err("h3 sol: a Sol route needs a sink (prefix | text | suffix)".into())
                    }
                    _ => {}
                }
                H3Attention::Sol(sol.clone())
            } else if t.downcast_ref::<DenseAttention>().is_some() {
                H3Attention::Dense
            } else if let Some(v) = t.downcast_ref::<Vsa>() {
                if contract.vsa_sparsity <= 0.0 {
                    return Err(format!(
                        "h3: techniques.vsa needs a VSA recipe (the DiT is loaded without to_gate_compress for {})",
                        recipe_ref.unwrap_or("auto")
                    ));
                }
                vsa_profile = Some(v.clone());
                H3Attention::Auto
            } else {
                return Err(format!(
                    "h3: attention technique '{}' is not implemented for H3",
                    t.name()
                ));
            }
        } else {
            // The legacy resolution, unchanged: env wins, else the recipe.
            let policy = recipe_sol_attn_policy(recipe_ref, env_sol.as_deref(), ref2va)?;
            sources.push((
                "attention_backend",
                if env_sol.is_some() {
                    "env FASTVIDEO_H3_SOL_ATTN"
                } else {
                    "recipe"
                },
            ));
            match policy_technique(policy) {
                Some(sol) => H3Attention::Sol(sol),
                None => H3Attention::Auto,
            }
        };
        let sol_policy = match &attention {
            H3Attention::Sol(sol) => sink_policy(sol.sink),
            _ => H3SolAttnPolicy::Off,
        };
        match &attention {
            H3Attention::Sol(sol) => items.push(Box::new(sol.clone())),
            H3Attention::Dense => items.push(Box::new(DenseAttention {
                enabled: Schedule::Const(true),
            })),
            H3Attention::Auto if contract.vsa_sparsity > 0.0 && !contract.dense => {
                items.push(Box::new(vsa_profile.clone().unwrap_or(Vsa {
                    enabled: Schedule::Const(true),
                    sparsity: None,
                    group: None,
                })))
            }
            H3Attention::Auto => {}
        }

        // --- VSA parameters ---
        let vsa_sparsity_env = env("FASTVIDEO_VSA_SPARSITY").filter(|v| !v.is_empty());
        let vsa_group = match env("FASTVIDEO_VSA_GROUP") {
            // `usize_flag`: an unparsable value falls back to the default.
            Some(v) => v.parse::<usize>().unwrap_or(DEFAULT_VSA_GROUP),
            None => vsa_profile
                .as_ref()
                .and_then(|v| v.group)
                .unwrap_or(DEFAULT_VSA_GROUP),
        }
        .max(1);

        // --- step output ---
        let teacache = match env("FASTVIDEO_H3_SOL_CACHE") {
            Some(v) => {
                sources.push(("step_output", "env FASTVIDEO_H3_SOL_CACHE"));
                teacache_requested(Some(&v))
                    .then(|| p_teacache.clone().unwrap_or_else(TeaCache::rtx))
            }
            None => {
                if p_teacache.is_some() {
                    sources.push(("step_output", "profile"));
                }
                p_teacache
            }
        };
        if let Some(tc) = &teacache {
            items.push(Box::new(tc.clone()));
        }

        let plan = compose(items, &h3_spec(), HORIZON).map_err(|e| e.to_string())?;
        Ok(Self {
            recipe,
            attention,
            sol_policy,
            vsa_sparsity_env,
            vsa_sparsity: vsa_profile.and_then(|v| v.sparsity),
            vsa_group,
            teacache,
            taeh3,
            plan,
            sources,
        })
    }

    /// Production resolution: the process env and the active profile.
    pub fn from_process(
        recipe: Option<&str>,
        contract: &H3InferenceContract,
        ref2va: bool,
    ) -> Result<Self, String> {
        let active = crate::techniques::settings::active();
        // The legacy flags through the settings table: the env var, else a
        // profile's raw `[env]` entry of the same name.
        Self::resolve(recipe, contract, ref2va, active.profile.as_ref(), &|k| {
            crate::techniques::settings::var(k)
        })
    }

    /// The Sol technique, if the blocks run Sol-Attn.
    pub fn sol(&self) -> Option<&SolAttn> {
        match &self.attention {
            H3Attention::Sol(s) => Some(s),
            _ => None,
        }
    }

    /// Dense regardless of the contract (a `dense_attention` technique).
    pub fn forces_dense(&self) -> bool {
        self.attention == H3Attention::Dense
    }

    /// VSA sparsity: `FASTVIDEO_VSA_SPARSITY`, else the profile's, else the
    /// contract's. The env value is validated as the pipeline always did.
    pub fn vsa_sparsity(&self, contract: &H3InferenceContract) -> Result<f64, String> {
        match &self.vsa_sparsity_env {
            Some(v) => v
                .parse::<f64>()
                .ok()
                .filter(|s| (0.0..1.0).contains(s))
                .ok_or_else(|| format!("FASTVIDEO_VSA_SPARSITY={v}: need [0, 1)")),
            None => Ok(self.vsa_sparsity.unwrap_or(contract.vsa_sparsity)),
        }
    }

    /// The TeaCache controller for a denoise of `num_forwards` forwards.
    pub fn teacache_state(&self, num_forwards: usize) -> Result<Option<H3TeaCache>, String> {
        self.teacache
            .as_ref()
            .map(|tc| {
                H3TeaCache::new(
                    tc.threshold,
                    tc.retain_steps,
                    tc.cooldown_steps,
                    tc.num_forwards.unwrap_or(num_forwards),
                    tc.coefficients.clone(),
                )
            })
            .transpose()
    }

    /// Whether the TeaCache parameters are the RTX run script's.
    pub fn teacache_is_official(&self) -> bool {
        self.teacache.as_ref().is_some_and(|tc| {
            let o = TeaCache::rtx();
            tc.threshold == o.threshold
                && tc.retain_steps == o.retain_steps
                && tc.cooldown_steps == o.cooldown_steps
                && tc.coefficients == o.coefficients
        })
    }

    pub fn describe(&self) -> String {
        let src: Vec<String> = self
            .sources
            .iter()
            .map(|(s, f)| format!("{s} from {f}"))
            .collect();
        if src.is_empty() {
            self.plan.describe()
        } else {
            format!("{} [{}]", self.plan.describe(), src.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::packing::{H3PackedLayout, KeyframeAnchor};
    use super::super::sol::{policy_route, sink_spec, sink_spec_for, technique_route, H3TeaCache};
    use super::*;
    use std::collections::BTreeMap;

    const RECIPES: [&str; 8] = [
        "8step",
        "4step-vsa",
        "4step-dense",
        "sol-h3",
        "sol-h3-ref2va",
        "sol-h3-spark",
        "sol-h3-rtx",
        "auto",
    ];
    const SOL_ENV: [Option<&str>; 7] = [
        None,
        Some("off"),
        Some("1"),
        Some("engine"),
        Some("spark"),
        Some("rtx"),
        Some("junk"),
    ];

    fn contract(recipe: &str) -> H3InferenceContract {
        H3InferenceContract::named(recipe).unwrap_or_else(|_| H3InferenceContract::fasth3_8step())
    }

    fn env_of(map: BTreeMap<&'static str, String>) -> impl Fn(&str) -> Option<String> {
        move |k: &str| map.get(k).cloned()
    }

    /// Off-identity: with no profile, every recipe x FASTVIDEO_H3_SOL_ATTN x
    /// FASTVIDEO_H3_SOL_CACHE x ref2va resolves to exactly what the legacy
    /// code decided (policy or its error, TeaCache on/off, VSA inputs).
    #[test]
    fn no_profile_is_the_legacy_resolution() {
        for recipe in RECIPES {
            let r = (recipe != "auto").then_some(recipe);
            let c = contract(recipe);
            for sol in SOL_ENV {
                for cache in [None, Some("teacache"), Some("1"), Some("0")] {
                    for ref2va in [false, true] {
                        let mut m = BTreeMap::new();
                        if let Some(v) = sol {
                            m.insert("FASTVIDEO_H3_SOL_ATTN", v.to_string());
                        }
                        if let Some(v) = cache {
                            m.insert("FASTVIDEO_H3_SOL_CACHE", v.to_string());
                        }
                        let legacy = recipe_sol_attn_policy(r, sol, ref2va);
                        let got = H3Techniques::resolve(r, &c, ref2va, None, &env_of(m));
                        match (legacy, got) {
                            (Err(a), Err(b)) => assert_eq!(a, b),
                            (Ok(policy), Ok(t)) => {
                                assert_eq!(t.sol_policy, policy, "{recipe} {sol:?}");
                                assert_eq!(t.teacache.is_some(), teacache_requested(cache));
                                assert!(!t.forces_dense());
                                assert_eq!(t.vsa_group, DEFAULT_VSA_GROUP);
                                assert_eq!(t.vsa_sparsity(&c).unwrap(), c.vsa_sparsity);
                                if let Some(s) = t.sol() {
                                    assert_eq!(Some(s.clone()), policy_technique(policy));
                                }
                            }
                            (a, b) => panic!("{recipe} {sol:?} {ref2va}: legacy {a:?} vs {b:?}"),
                        }
                    }
                }
            }
        }
    }

    /// The DSL routes equal the hand-written clocks at every step and block.
    /// (Past the 50th block both refuse, except that the old Spark clock
    /// skipped the check after update 3; the model has no such block.)
    #[test]
    fn technique_routes_equal_the_policy_routes_everywhere() {
        use super::super::sol::LAYERS_PER_FORWARD;
        for policy in [
            H3SolAttnPolicy::Engine,
            H3SolAttnPolicy::Spark,
            H3SolAttnPolicy::Rtx,
        ] {
            let sol = policy_technique(policy).unwrap();
            for step in 0..HORIZON {
                assert!(technique_route(&sol, step, LAYERS_PER_FORWARD).is_err());
                for layer in 0..LAYERS_PER_FORWARD {
                    assert_eq!(
                        technique_route(&sol, step, layer),
                        policy_route(policy, step, layer),
                        "{policy:?} step {step} layer {layer}"
                    );
                }
            }
        }
    }

    #[test]
    fn technique_sinks_equal_the_policy_sinks() {
        let t2va = H3PackedLayout::new(3, (2, 4, 4), 2, [1, 2, 2]).unwrap();
        let fl2va =
            H3PackedLayout::with_keyframes(3, (2, 4, 4), 2, [1, 2, 2], &[KeyframeAnchor::First])
                .unwrap();
        for layout in [&t2va, &fl2va] {
            for policy in [
                H3SolAttnPolicy::Off,
                H3SolAttnPolicy::Engine,
                H3SolAttnPolicy::Rtx,
            ] {
                let mode = policy_technique(policy).map_or(SinkMode::None, |s| s.sink);
                assert_eq!(sink_spec(policy, layout), sink_spec_for(mode, layout));
            }
        }
    }

    #[test]
    fn teacache_state_is_the_official_controller() {
        let mut m = BTreeMap::new();
        m.insert("FASTVIDEO_H3_SOL_CACHE", "teacache".to_string());
        let c = contract("sol-h3-rtx");
        let t = H3Techniques::resolve(Some("sol-h3-rtx"), &c, false, None, &env_of(m)).unwrap();
        let a = t.teacache_state(49).unwrap().unwrap();
        let b = H3TeaCache::official(49).unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert!(t.teacache_is_official());
    }

    fn profile(text: &str) -> Profile {
        Profile::parse(text).unwrap()
    }

    #[test]
    fn profile_techniques_drive_the_seams_and_env_still_wins() {
        let p = profile(
            "[id]\nname='p'\n[pipeline]\nmodel='h3'\nrecipe='sol-h3-rtx'\n\
             [techniques.sol_attn]\npreset='rtx'\ndense_steps=3\n[techniques.teacache]\nthreshold=0.2\n",
        );
        let c = contract("sol-h3-rtx");
        let none = |_: &str| None;
        let t = H3Techniques::resolve(None, &c, false, Some(&p), &none).unwrap();
        assert_eq!(
            t.recipe.as_deref(),
            Some("sol-h3-rtx"),
            "the profile names the recipe"
        );
        let sol = t.sol().unwrap();
        assert_eq!(
            technique_route(sol, 3, 2).unwrap(),
            super::super::sol::H3SolRoute::Sol { tau: 1.0 }
        );
        assert_eq!(t.teacache.as_ref().unwrap().threshold, 0.2);
        assert!(!t.teacache_is_official());
        assert_eq!(t.plan.names(), vec!["sol_attn", "teacache"]);
        // FASTVIDEO_H3_SOL_ATTN=off and FASTVIDEO_H3_SOL_CACHE=0 override both.
        let mut m = BTreeMap::new();
        m.insert("FASTVIDEO_H3_SOL_ATTN", "off".to_string());
        m.insert("FASTVIDEO_H3_SOL_CACHE", "0".to_string());
        let t = H3Techniques::resolve(None, &c, false, Some(&p), &env_of(m)).unwrap();
        assert_eq!(t.sol_policy, H3SolAttnPolicy::Off);
        assert!(t.teacache.is_none());
        assert!(t.plan.techniques.is_empty());
    }

    #[test]
    fn profile_attention_checks_match_the_env_checks() {
        let none = |_: &str| None;
        let spark = profile(
            "[id]\nname='p'\n[pipeline]\nmodel='h3'\n[techniques.sol_attn]\npreset='spark'\n",
        );
        assert!(H3Techniques::resolve(
            Some("sol-h3"),
            &contract("sol-h3"),
            false,
            Some(&spark),
            &none
        )
        .is_err());
        assert!(H3Techniques::resolve(
            Some("sol-h3-spark"),
            &contract("sol-h3-spark"),
            true,
            Some(&spark),
            &none
        )
        .is_ok());
        let engine = profile(
            "[id]\nname='p'\n[pipeline]\nmodel='h3'\n[techniques.sol_attn]\npreset='engine'\n",
        );
        assert!(H3Techniques::resolve(
            Some("sol-h3"),
            &contract("sol-h3"),
            true,
            Some(&engine),
            &none
        )
        .is_err());
        let vsa = profile(
            "[id]\nname='p'\n[pipeline]\nmodel='h3'\n[techniques.vsa]\nsparsity=0.5\ngroup=4\n",
        );
        assert!(H3Techniques::resolve(
            Some("sol-h3-rtx"),
            &contract("sol-h3-rtx"),
            false,
            Some(&vsa),
            &none
        )
        .is_err());
        let t = H3Techniques::resolve(Some("8step"), &contract("8step"), false, Some(&vsa), &none)
            .unwrap();
        assert_eq!(
            (t.vsa_sparsity(&contract("8step")).unwrap(), t.vsa_group),
            (0.5, 4)
        );
        let dense =
            profile("[id]\nname='p'\n[pipeline]\nmodel='h3'\n[techniques.dense_attention]\n");
        let t = H3Techniques::resolve(
            Some("sol-h3-rtx"),
            &contract("sol-h3-rtx"),
            false,
            Some(&dense),
            &none,
        )
        .unwrap();
        assert!(t.forces_dense() && t.sol().is_none());
        let ltx = profile("[id]\nname='p'\n[pipeline]\nmodel='ltx2'\n");
        assert!(H3Techniques::resolve(None, &contract("8step"), false, Some(&ltx), &none).is_err());
    }

    /// The FastH3 8-step arms: Sol dense nowhere; TeaCache free to reuse
    /// only the middle steps 3 and 4 of 8.
    #[test]
    fn fasth3_8step_arms_resolve_as_described() {
        let none = |_: &str| None;
        let c = contract("8step");
        let load = |n: &str| {
            Profile::parse(crate::techniques::builtin::get(n).unwrap()).unwrap()
        };
        let sol = H3Techniques::resolve(None, &c, false, Some(&load("h3/fasth3_8step_sol")), &none).unwrap();
        let s = sol.sol().unwrap();
        for step in 0..8 {
            for layer in 0..super::super::sol::LAYERS_PER_FORWARD {
                assert_eq!(
                    technique_route(s, step, layer).unwrap(),
                    super::super::sol::H3SolRoute::Sol { tau: 1.0 }
                );
            }
        }
        assert!(sol.teacache.is_none());
        for name in ["h3/fasth3_8step_teacache", "h3/fasth3_8step_sol_teacache"] {
            let t = H3Techniques::resolve(None, &c, false, Some(&load(name)), &none).unwrap();
            let mut tc = t.teacache_state(8).unwrap().unwrap();
            // Every step's signal is tiny: exactly the middle steps reuse.
            let reused: Vec<usize> = (0..8)
                .filter(|&s| {
                    let d = tc.decide(s, 0.01);
                    if d.compute {
                        tc.note_computed();
                    }
                    !d.compute
                })
                .collect();
            assert_eq!(reused, vec![3, 4], "{name}");
        }
    }

    /// OFF: listing a technique with `enabled = false` resolves exactly like
    /// not listing it.
    #[test]
    fn disabled_profile_techniques_are_the_baseline() {
        let none = |_: &str| None;
        let off = profile(
            "[id]\nname='p'\n[pipeline]\nmodel='h3'\n\
             [techniques.sol_attn]\nenabled=false\n[techniques.teacache]\nenabled=false\n\
             [techniques.mxfp8]\nenabled=false\n",
        );
        for recipe in RECIPES {
            let r = (recipe != "auto").then_some(recipe);
            let c = contract(recipe);
            let base = H3Techniques::resolve(r, &c, false, None, &none);
            let with = H3Techniques::resolve(r, &c, false, Some(&off), &none);
            match (base, with) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.attention, b.attention);
                    assert_eq!(a.teacache, b.teacache);
                    assert_eq!(a.plan.names(), b.plan.names());
                }
                (Err(a), Err(b)) => assert_eq!(a, b),
                (a, b) => panic!("{recipe}: {a:?} vs {b:?}"),
            }
        }
        assert!(off.settings().unwrap().is_empty());
    }
}
