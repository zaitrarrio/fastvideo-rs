//! Technique profiles: a TOML file naming a pipeline and the techniques,
//! kernels and settings it runs with (`profiles/<model>/*.toml`, documented
//! in `docs/techniques.md`).
//!
//! ```toml
//! [id]
//! name = "h3_rtx5090_fullopt"
//! family = "minimax_h3"
//! upstream = "sol-engine config/minimax_h3/rtx5090_fullopt.toml"
//!
//! [pipeline]
//! model = "h3"
//! recipe = "sol-h3-rtx"
//!
//! [techniques.sol_attn]
//! preset = "rtx"
//! dense_steps = 10       # SOL_ATTN_FIRST_DENSE_STEPS
//! dense_layers = 2       # SOL_ATTN_FIRST_DENSE_LAYERS
//! tau = 1.0              # SOL_ATTN_TAU
//!
//! [techniques.teacache]
//! threshold = 0.10       # H3_TEACACHE_THRESHOLD
//!
//! [kernels]
//! sol_attention = "nvcc:x4f"
//!
//! [env]                  # raw FASTVIDEO_* settings, lowest level
//! FASTVIDEO_VSA_TMA = "1"
//! ```
//!
//! A sol-engine config (`config/minimax_h3/rtx5090_*.toml`, top-level
//! `id = "..."` and an `[env]` of `SOL_ATTN_*` / `H3_TEACACHE_*` keys) loads
//! too: [`Profile::from_sol_engine`] maps the keys whose concepts match onto
//! the same techniques and records the rest in [`Profile::notes`].

use std::collections::BTreeMap;
use std::path::Path;

use super::compose::{compose, Plan, HORIZON};
use super::kernels::{resolve, KernelChoice, KernelOp};
use super::methods::{self, DenseAttention, LinearPrecision, LinearRecipe, SolAttn, TeaCache};
use super::registry;
use super::schedule::{parse_index_set, parse_tau, Schedule};
use super::settings::Settings;
use super::technique::Technique;

#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub name: String,
    pub family: String,
    pub description: String,
    /// The upstream config this profile reproduces, if any.
    pub upstream: Option<String>,
    /// `h3` | `ltx2`.
    pub model: String,
    /// The pipeline recipe (H3 `--h3-recipe`); `None` leaves it to the caller.
    pub recipe: Option<String>,
    /// Every listed technique, including disabled ones (compose drops those).
    pub techniques: Vec<Box<dyn Technique>>,
    pub kernels: Vec<KernelChoice>,
    /// Raw `FASTVIDEO_*` settings.
    pub env: BTreeMap<String, String>,
    /// Upstream keys with no effect here, and why.
    pub notes: Vec<String>,
}

const TOP_KEYS: [&str; 6] = ["id", "pipeline", "techniques", "kernels", "env", "notes"];

fn str_of<'a>(t: &'a toml::Table, key: &str, ctx: &str) -> Result<Option<&'a str>, String> {
    match t.get(key) {
        None => Ok(None),
        Some(toml::Value::String(s)) => Ok(Some(s)),
        Some(other) => Err(format!("{ctx}.{key} = {other}: expected a string")),
    }
}

fn table_of<'a>(t: &'a toml::Table, key: &str) -> Result<Option<&'a toml::Table>, String> {
    match t.get(key) {
        None => Ok(None),
        Some(toml::Value::Table(t)) => Ok(Some(t)),
        Some(other) => Err(format!("[{key}] must be a table, got {other}")),
    }
}

fn only_keys(t: &toml::Table, allowed: &[&str], ctx: &str) -> Result<(), String> {
    let bad: Vec<&String> = t
        .keys()
        .filter(|k| !allowed.contains(&k.as_str()))
        .collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{ctx}: unknown key(s) {bad:?} (known: {allowed:?})"
        ))
    }
}

fn is_sol_engine(t: &toml::Table) -> bool {
    matches!(t.get("id"), Some(toml::Value::String(_))) || t.contains_key("model_profile")
}

impl Profile {
    /// A file, else a builtin (`h3/rtx5090_sol`, [`super::builtin`]).
    pub fn load(path: &Path) -> Result<Self, String> {
        if path.is_file() {
            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            return Self::parse(&text);
        }
        match super::builtin::get(&path.to_string_lossy()) {
            Some(text) => Self::parse(text),
            None => Err(format!(
                "no such file, and not a builtin profile ({})",
                super::builtin::names().join(", ")
            )),
        }
    }

    /// A typed profile, or a sol-engine config.
    pub fn parse(text: &str) -> Result<Self, String> {
        let t: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
        if is_sol_engine(&t) {
            return Self::from_sol_engine(&t);
        }
        only_keys(&t, &TOP_KEYS, "profile")?;
        let mut p = Profile::default();
        if let Some(id) = table_of(&t, "id")? {
            only_keys(id, &["name", "family", "description", "upstream"], "[id]")?;
            p.name = str_of(id, "name", "id")?.unwrap_or_default().to_string();
            p.family = str_of(id, "family", "id")?.unwrap_or_default().to_string();
            p.description = str_of(id, "description", "id")?
                .unwrap_or_default()
                .to_string();
            p.upstream = str_of(id, "upstream", "id")?.map(str::to_owned);
        }
        if p.name.is_empty() {
            return Err("[id].name is required".into());
        }
        let pipe = table_of(&t, "pipeline")?
            .ok_or("[pipeline] is required (model = \"h3\" | \"ltx2\")")?;
        only_keys(pipe, &["model", "recipe"], "[pipeline]")?;
        p.model = str_of(pipe, "model", "pipeline")?
            .ok_or("[pipeline].model is required")?
            .to_string();
        let spec = registry::model_spec(&p.model)
            .ok_or_else(|| format!("[pipeline].model = {:?}: expected h3 | ltx2", p.model))?;
        p.recipe = str_of(pipe, "recipe", "pipeline")?.map(str::to_owned);
        if p.recipe.is_some() && p.model != "h3" {
            return Err(format!(
                "[pipeline].recipe is H3's --h3-recipe; {} workloads are set on the command line",
                p.model
            ));
        }
        if let Some(techs) = table_of(&t, "techniques")? {
            for (name, v) in techs {
                let tab = v
                    .as_table()
                    .ok_or_else(|| format!("[techniques.{name}] must be a table"))?;
                p.techniques.push(registry::build(name, tab)?);
            }
        }
        if let Some(k) = table_of(&t, "kernels")? {
            for (key, v) in k {
                let op = KernelOp::from_key(key).ok_or_else(|| {
                    format!(
                        "[kernels].{key}: unknown op (known: {:?})",
                        KernelOp::ALL.map(KernelOp::key)
                    )
                })?;
                let choice = v
                    .as_str()
                    .ok_or_else(|| format!("[kernels].{key} must be a string"))?;
                if let Some(c) = resolve(op, choice, None)? {
                    p.kernels.push(c);
                }
            }
        }
        if let Some(env) = table_of(&t, "env")? {
            for (k, v) in env {
                if !k.starts_with("FASTVIDEO_") {
                    return Err(format!("[env].{k}: only FASTVIDEO_* settings belong here"));
                }
                let v = match v {
                    toml::Value::String(s) => s.clone(),
                    toml::Value::Integer(i) => i.to_string(),
                    toml::Value::Float(f) => f.to_string(),
                    toml::Value::Boolean(b) => if *b { "1" } else { "0" }.to_string(),
                    other => return Err(format!("[env].{k} = {other}: expected a scalar")),
                };
                p.env.insert(k.clone(), v);
            }
        }
        if let Some(n) = t.get("notes") {
            p.notes = n
                .as_array()
                .and_then(|a| a.iter().map(|x| x.as_str().map(str::to_owned)).collect())
                .ok_or("notes must be a list of strings")?;
        }
        // Validate composition now: a conflict is a config error at load.
        compose(p.techniques.clone(), &spec, HORIZON)
            .map_err(|e| format!("profile {}: {e}", p.name))?;
        Ok(p)
    }

    /// Compose this profile's techniques for its model.
    pub fn plan(&self) -> Result<Plan, String> {
        let spec = registry::model_spec(&self.model)
            .ok_or_else(|| format!("profile {}: unknown model {:?}", self.name, self.model))?;
        compose(self.techniques.clone(), &spec, HORIZON).map_err(|e| e.to_string())
    }

    /// Techniques' settings, then kernels, then raw `[env]`. A raw setting
    /// that disagrees with a technique's is a conflict.
    pub fn settings(&self) -> Result<Settings, String> {
        let spec = registry::model_spec(&self.model)
            .ok_or_else(|| format!("profile {}: unknown model {:?}", self.name, self.model))?;
        let plan = self.plan()?;
        let mut s = Settings::default();
        for t in &plan.techniques {
            for (k, v) in methods::settings_for(t.as_ref(), &spec)? {
                s.set(k, &v, t.name())?;
            }
        }
        for c in &self.kernels {
            let (k, v) = c.setting();
            s.set(k, v, "kernels")?;
        }
        for (k, v) in &self.env {
            s.set(k, v, "env")?;
        }
        Ok(s)
    }

    /// A sol-engine config (`config/minimax_h3/rtx5090_{dense,sol,fullopt}.toml`
    /// and friends). Mapped keys:
    ///
    /// * `H3_RTX5090_PROFILE`: `dense` is [`DenseAttention`]; `sol` and
    ///   `fullopt` are [`SolAttn::rtx`]; `fullopt` adds [`TeaCache`] unless
    ///   `H3_TEACACHE_ENABLED=0` (`run_minimax_h3_gpu.sh:75-93`);
    /// * `SOL_ATTN_TAU` / `_THRESH_TYPE` / `_FIRST_DENSE_STEPS` /
    ///   `_FIRST_DENSE_LAYERS` / `_CORRECTNESS_GATE` / `_FORCE_DENSE`: the
    ///   Sol route (`adapter.py:452-461, :476-477`);
    /// * `H3_TEACACHE_*`: TeaCache (`teacache.py:23-50, :136`);
    /// * `[official_config] transformer_dtype = "bf16"`: [`LinearRecipe::Bf16`];
    ///   `steps = 50` with the RTX profile: recipe `sol-h3-rtx`.
    ///
    /// Everything else (paths, warmup counts, the full-VAE-after-denoise
    /// switch, whose decoders here stay resident unless the DiT streams) is
    /// recorded in [`Profile::notes`].
    pub fn from_sol_engine(t: &toml::Table) -> Result<Self, String> {
        let mut p = Profile {
            name: match t.get("id") {
                Some(toml::Value::String(s)) => s.clone(),
                _ => "sol-engine".into(),
            },
            description: str_of(t, "description", "")?
                .unwrap_or_default()
                .to_string(),
            family: str_of(t, "model_profile", "")?
                .unwrap_or_default()
                .to_string(),
            upstream: Some("sol-engine config".into()),
            ..Default::default()
        };
        match p.family.as_str() {
            "minimax_h3" => p.model = "h3".into(),
            "ltx25" | "ltx23" | "ltx2" => p.model = "ltx2".into(),
            other => {
                return Err(format!(
                    "sol-engine config: model_profile {other:?} is not ported"
                ))
            }
        }
        let env: BTreeMap<String, String> = match t.get("env") {
            Some(toml::Value::Table(e)) => e
                .iter()
                .map(|(k, v)| {
                    let v = match v {
                        toml::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    (k.clone(), v)
                })
                .collect(),
            _ => BTreeMap::new(),
        };
        let official = table_of(t, "official_config")?;
        let get = |k: &str| env.get(k).map(String::as_str);
        let mut used: Vec<&str> = Vec::new();
        let num = |k: &str| -> Result<Option<toml::Value>, String> {
            env.get(k)
                .map(|v| {
                    let v = v.trim();
                    v.parse::<i64>()
                        .map(toml::Value::Integer)
                        .or_else(|_| v.parse::<f64>().map(toml::Value::Float))
                        .map_err(|_| format!("sol-engine [env].{k} = {v:?}: not a number"))
                })
                .transpose()
        };
        let rtx_profile = get("H3_RTX5090_PROFILE").map(|s| s.trim().to_ascii_lowercase());
        used.push("H3_RTX5090_PROFILE");
        match rtx_profile.as_deref() {
            None => {}
            Some("dense") => p.techniques.push(Box::new(DenseAttention {
                enabled: Schedule::Const(true),
            })),
            Some("sol") | Some("fullopt") => {
                let mut sol = SolAttn::rtx();
                if let Some(v) = num("SOL_ATTN_TAU")? {
                    sol.route.tau = parse_tau(&v)?;
                }
                if let Some(v) = num("SOL_ATTN_FIRST_DENSE_STEPS")? {
                    sol.route.dense_steps = parse_index_set(&v, "SOL_ATTN_FIRST_DENSE_STEPS")?;
                }
                if let Some(v) = num("SOL_ATTN_FIRST_DENSE_LAYERS")? {
                    sol.route.dense_layers = parse_index_set(&v, "SOL_ATTN_FIRST_DENSE_LAYERS")?;
                }
                if let Some(v) = get("SOL_ATTN_THRESH_TYPE") {
                    if v != "diag" {
                        return Err(format!(
                            "SOL_ATTN_THRESH_TYPE={v}: only diag is implemented"
                        ));
                    }
                }
                sol.correctness_gate =
                    get("SOL_ATTN_CORRECTNESS_GATE").is_some_and(|v| v.trim() == "1");
                if get("SOL_ATTN_FORCE_DENSE").is_some_and(|v| v.trim() == "1") {
                    sol.route.dense_steps = super::schedule::StepSet::from(0);
                }
                p.techniques.push(Box::new(sol));
                let tea_on = match get("H3_TEACACHE_ENABLED") {
                    Some(v) => v.trim() == "1",
                    None => rtx_profile.as_deref() == Some("fullopt"),
                };
                if tea_on && rtx_profile.as_deref() == Some("fullopt") {
                    let mut tc = TeaCache::rtx();
                    if let Some(v) = num("H3_TEACACHE_THRESHOLD")? {
                        tc.threshold = v
                            .as_float()
                            .or(v.as_integer().map(|i| i as f64))
                            .unwrap_or(tc.threshold);
                    }
                    if let Some(v) = num("H3_TEACACHE_RETAIN_STEPS")? {
                        tc.retain_steps = v.as_integer().unwrap_or(5) as usize;
                    }
                    if let Some(v) = num("H3_TEACACHE_COOLDOWN_STEPS")? {
                        tc.cooldown_steps = v.as_integer().unwrap_or(1) as usize;
                    }
                    if let Some(v) = num("H3_TEACACHE_NUM_FORWARDS")? {
                        tc.num_forwards = v.as_integer().map(|i| i as usize);
                    }
                    if let Some(v) = get("H3_TEACACHE_COEFFICIENTS") {
                        tc.coefficients = v
                            .split(',')
                            .map(|x| x.trim().parse::<f64>())
                            .collect::<Result<_, _>>()
                            .map_err(|_| format!("H3_TEACACHE_COEFFICIENTS={v}: not numbers"))?;
                    }
                    p.techniques.push(Box::new(tc));
                }
            }
            Some(other) => {
                return Err(format!(
                    "H3_RTX5090_PROFILE={other}: expected dense|sol|fullopt"
                ))
            }
        }
        used.extend([
            "SOL_ATTN_TAU",
            "SOL_ATTN_THRESH_TYPE",
            "SOL_ATTN_FIRST_DENSE_STEPS",
            "SOL_ATTN_FIRST_DENSE_LAYERS",
            "SOL_ATTN_CORRECTNESS_GATE",
            "SOL_ATTN_FORCE_DENSE",
            "H3_TEACACHE_ENABLED",
            "H3_TEACACHE_THRESHOLD",
            "H3_TEACACHE_RETAIN_STEPS",
            "H3_TEACACHE_COOLDOWN_STEPS",
            "H3_TEACACHE_NUM_FORWARDS",
            "H3_TEACACHE_COEFFICIENTS",
        ]);
        if let Some(oc) = official {
            if oc.get("transformer_dtype").and_then(|v| v.as_str()) == Some("bf16") {
                p.techniques.push(Box::new(LinearPrecision {
                    enabled: Schedule::Const(true),
                    recipe: LinearRecipe::Bf16,
                    nvfp4_rule: None,
                }));
            }
            let steps = oc.get("steps").and_then(|v| v.as_integer());
            if p.model == "h3" && rtx_profile.is_some() && steps == Some(50) {
                p.recipe = Some("sol-h3-rtx".into());
            }
            for (k, v) in oc {
                if !matches!(k.as_str(), "transformer_dtype" | "steps") {
                    p.notes.push(format!(
                        "official_config.{k} = {v} (workload; set on the command line)"
                    ));
                }
            }
        }
        for (k, v) in &env {
            if !used.contains(&k.as_str()) {
                let why = if k.starts_with("H3_FULL_VAE") {
                    "decoders stay resident unless the DiT streams"
                } else {
                    "harness / path key, no technique"
                };
                p.notes.push(format!("{k}={v}: {why}"));
            }
        }
        if p.techniques.iter().any(|t| {
            t.downcast_ref::<SolAttn>()
                .is_some_and(|s| s.correctness_gate)
        }) {
            p.notes.push(
                "SOL_ATTN_CORRECTNESS_GATE=1: sol-engine's sampled dense-vs-Sol gate is not implemented here; Sol runs ungated".into(),
            );
        }
        let spec = registry::model_spec(&p.model).expect("mapped above");
        compose(p.techniques.clone(), &spec, HORIZON).map_err(|e| e.to_string())?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_profile_parses_composes_and_settles() {
        for (name, text) in super::super::builtin::PROFILES {
            let p = Profile::parse(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(!p.name.is_empty(), "{name}");
            p.settings().unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(Profile::load(Path::new(name)).is_ok(), "{name}");
        }
    }

    /// Each shipped rtx5090 profile is the sol-engine config it names: the
    /// same techniques with the same parameters.
    #[test]
    fn rtx5090_profiles_equal_the_sol_engine_configs() {
        let cases = [
            (
                "h3/rtx5090_dense",
                include_str!("fixtures/sol_engine_rtx5090_dense.toml"),
            ),
            (
                "h3/rtx5090_sol",
                include_str!("fixtures/sol_engine_rtx5090_sol.toml"),
            ),
            (
                "h3/rtx5090_fullopt",
                include_str!("fixtures/sol_engine_rtx5090_fullopt.toml"),
            ),
        ];
        for (ours, upstream) in cases {
            let a = Profile::parse(super::super::builtin::get(ours).unwrap()).unwrap();
            let b = Profile::parse(upstream).unwrap();
            assert_eq!(a.model, b.model, "{ours}");
            assert_eq!(a.recipe, b.recipe, "{ours}");
            let (pa, pb) = (a.plan().unwrap(), b.plan().unwrap());
            assert_eq!(pa.names(), pb.names(), "{ours}");
            assert_eq!(
                pa.get::<SolAttn>()
                    .map(|s| (&s.route, s.sink, &s.thresh_type)),
                pb.get::<SolAttn>()
                    .map(|s| (&s.route, s.sink, &s.thresh_type)),
                "{ours}"
            );
            assert_eq!(pa.get::<TeaCache>(), pb.get::<TeaCache>(), "{ours}");
            assert_eq!(a.settings().unwrap(), b.settings().unwrap(), "{ours}");
        }
        let full =
            Profile::parse(include_str!("fixtures/sol_engine_rtx5090_fullopt.toml")).unwrap();
        assert!(
            full.notes.iter().any(|n| n.contains("CORRECTNESS_GATE")),
            "{:?}",
            full.notes
        );
        assert!(
            full.notes
                .iter()
                .any(|n| n.starts_with("H3_FULL_VAE_AFTER_DENOISE")),
            "{:?}",
            full.notes
        );
    }

    #[test]
    fn builtin_names_resolve_in_every_spelling() {
        for n in [
            "h3/rtx5090_sol",
            "h3/rtx5090_sol.toml",
            "profiles/h3/rtx5090_sol.toml",
        ] {
            assert!(super::super::builtin::get(n).is_some(), "{n}");
        }
        assert!(Profile::load(Path::new("h3/nope")).is_err());
    }
}
