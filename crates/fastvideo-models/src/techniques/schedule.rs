//! Step-schedule DSL: a technique parameter as a function of `(step, stage)`
//! and, for attention routes, of `(step, layer)`.
//!
//! Port of sol-engine `techniques/schedule.py`: [`StepSet::parse`] is
//! `parse_steps` (:18-47), [`Schedule`] is `Schedule[T]` with its
//! constructors `const` (:69-70), `at_steps` (:73-79), `before` (:82-87) and
//! `by_stage` (:96-103), and [`Schedule::truthy_steps`] is the overlap probe
//! `compose()` uses (:62-63). Two deliberate differences:
//!
//! * a malformed step token is a config error here; `parse_steps` skips it
//!   silently (:36-37, :44-45), which turns a typo into "never";
//! * a step set may end open (`"4-"`, every step from 4 on), which the
//!   Spark ladder needs ("steps past the fourth update stay dense").
//!
//! [`SparseRoute`] is the two-axis rule every Sol-Attn route in the repo is
//! written in: dense on a set of steps, dense on a set of layers, otherwise
//! sparse at a per-step tau. The RTX cell (`adapter.py` `_dense_policy`:
//! `step_index < 10 or layer_index < 2`), the Sol-H3 engine
//! (`dense_steps=1, dense_layers=2`) and the Spark ladder (update 0 and layer
//! 0 dense, taus 1 / 1.25 / 1.5) are three values of it.

use std::collections::BTreeMap;
use std::fmt;

/// A set of step (or layer) indices: explicit points, closed ranges and at
/// most one open tail `n-`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepSet {
    ranges: Vec<(usize, usize)>,
    open_from: Option<usize>,
}

impl StepSet {
    pub fn empty() -> Self {
        Self::default()
    }

    /// `0..n`: the first `n` indices (sol-engine's `SOL_ATTN_FIRST_DENSE_STEPS`
    /// / `_LAYERS` integers). `first(0)` is empty.
    pub fn first(n: usize) -> Self {
        if n == 0 {
            return Self::empty();
        }
        Self {
            ranges: vec![(0, n - 1)],
            open_from: None,
        }
    }

    /// Every index from `n` on.
    pub fn from(n: usize) -> Self {
        Self {
            ranges: Vec::new(),
            open_from: Some(n),
        }
    }

    /// `"1-2,5,7-9"`, `"4-"` (open tail), `""` (empty). Reversed ranges are
    /// normalised as `parse_steps` does (:38-39).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut out = Self::empty();
        for tok in spec.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            let num = |s: &str| -> Result<usize, String> {
                s.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("step set {spec:?}: {tok:?} is not an index or range"))
            };
            match tok.split_once('-') {
                Some((lo, hi)) if hi.trim().is_empty() => {
                    let lo = num(lo)?;
                    out.open_from = Some(out.open_from.map_or(lo, |o| o.min(lo)));
                }
                Some((lo, hi)) => {
                    let (a, b) = (num(lo)?, num(hi)?);
                    out.ranges.push((a.min(b), a.max(b)));
                }
                None => {
                    let i = num(tok)?;
                    out.ranges.push((i, i));
                }
            }
        }
        Ok(out)
    }

    pub fn contains(&self, i: usize) -> bool {
        self.open_from.is_some_and(|o| i >= o)
            || self.ranges.iter().any(|&(a, b)| (a..=b).contains(&i))
    }

    /// Members below `horizon`.
    pub fn members(&self, horizon: usize) -> Vec<usize> {
        (0..horizon).filter(|&i| self.contains(i)).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty() && self.open_from.is_none()
    }
}

impl fmt::Display for StepSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = self
            .ranges
            .iter()
            .map(|&(a, b)| {
                if a == b {
                    a.to_string()
                } else {
                    format!("{a}-{b}")
                }
            })
            .collect();
        if let Some(o) = self.open_from {
            parts.push(format!("{o}-"));
        }
        if parts.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&parts.join(","))
        }
    }
}

/// A value resolved per `(step, stage)` (`schedule.py` `Schedule[T]`).
#[derive(Clone, Debug, PartialEq)]
pub enum Schedule<T> {
    /// `const(value)`.
    Const(T),
    /// `at_steps(steps, value, default)`.
    AtSteps {
        steps: StepSet,
        value: T,
        default: T,
    },
    /// `before(n, value, then)`: `value` for steps `< n`.
    Before { n: usize, value: T, then: T },
    /// `by_stage(mapping, default)`: per-stage schedule (LTX-2 stage 1 / 2).
    ByStage {
        stages: BTreeMap<String, Schedule<T>>,
        default: T,
    },
    /// One value per listed step, `default` elsewhere (a per-step tau ladder).
    PerStep {
        values: BTreeMap<usize, T>,
        default: T,
    },
}

impl<T: Clone> Schedule<T> {
    pub fn at(&self, step: usize, stage: &str) -> T {
        match self {
            Self::Const(v) => v.clone(),
            Self::AtSteps {
                steps,
                value,
                default,
            } => {
                if steps.contains(step) {
                    value.clone()
                } else {
                    default.clone()
                }
            }
            Self::Before { n, value, then } => {
                if step < *n {
                    value.clone()
                } else {
                    then.clone()
                }
            }
            Self::ByStage { stages, default } => match stages.get(stage) {
                Some(s) => s.at(step, stage),
                None => default.clone(),
            },
            Self::PerStep { values, default } => values.get(&step).unwrap_or(default).clone(),
        }
    }

    /// Whether the value is the same at every step and stage.
    pub fn as_const(&self) -> Option<&T> {
        match self {
            Self::Const(v) => Some(v),
            _ => None,
        }
    }
}

impl Schedule<bool> {
    /// Steps in `[0, horizon)` where the schedule is true in any stage
    /// (`truthy_steps`, :62-63; compose's default horizon is 64, :60).
    pub fn truthy_steps(&self, horizon: usize) -> Vec<usize> {
        let stages = self.stage_names();
        (0..horizon)
            .filter(|&s| self.at(s, "") || stages.iter().any(|st| self.at(s, st)))
            .collect()
    }

    fn stage_names(&self) -> Vec<String> {
        match self {
            Self::ByStage { stages, .. } => stages.keys().cloned().collect(),
            _ => Vec::new(),
        }
    }

    /// Parse an `enabled` value: `true` / `false`, a step set string
    /// (`"0-3"`: on at those steps), or `{ before = n }` / `{ from = n }` /
    /// `{ stage2 = true, ... }` tables.
    pub fn parse_enabled(v: &toml::Value) -> Result<Self, String> {
        match v {
            toml::Value::Boolean(b) => Ok(Self::Const(*b)),
            toml::Value::String(s) => Ok(Self::AtSteps {
                steps: StepSet::parse(s)?,
                value: true,
                default: false,
            }),
            toml::Value::Table(t) => {
                if let Some(n) = t.get("before") {
                    let n = as_usize(n).ok_or("enabled.before must be a step count")?;
                    return Ok(Self::Before {
                        n,
                        value: true,
                        then: false,
                    });
                }
                if let Some(n) = t.get("from") {
                    let n = as_usize(n).ok_or("enabled.from must be a step index")?;
                    return Ok(Self::Before {
                        n,
                        value: false,
                        then: true,
                    });
                }
                let mut stages = BTreeMap::new();
                for (k, v) in t {
                    stages.insert(k.clone(), Self::parse_enabled(v)?);
                }
                Ok(Self::ByStage {
                    stages,
                    default: false,
                })
            }
            other => Err(format!(
                "enabled = {other}: expected a bool, a step set or a table"
            )),
        }
    }
}

pub(crate) fn as_usize(v: &toml::Value) -> Option<usize> {
    v.as_integer().and_then(|i| usize::try_from(i).ok())
}

/// A step set given as an integer (the first `n`) or a set string.
pub fn parse_index_set(v: &toml::Value, what: &str) -> Result<StepSet, String> {
    match v {
        toml::Value::Integer(_) => Ok(StepSet::first(
            as_usize(v).ok_or_else(|| format!("{what} must be >= 0"))?,
        )),
        toml::Value::String(s) => StepSet::parse(s),
        other => Err(format!(
            "{what} = {other}: expected a count (the first n) or a set like \"0,4-\""
        )),
    }
}

/// What one attention call does under a [`SparseRoute`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Route {
    Dense,
    Sparse { tau: f64 },
}

/// Dense on `dense_steps` or `dense_layers`, otherwise sparse at `tau(step)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseRoute {
    pub dense_steps: StepSet,
    pub dense_layers: StepSet,
    pub tau: Schedule<f64>,
}

impl SparseRoute {
    pub fn at(&self, step: usize, layer: usize) -> Route {
        if self.dense_steps.contains(step) || self.dense_layers.contains(layer) {
            Route::Dense
        } else {
            Route::Sparse {
                tau: self.tau.at(step, ""),
            }
        }
    }

    /// Steps below `horizon` with at least one sparse layer among `layers`.
    pub fn sparse_steps(&self, horizon: usize, layers: usize) -> Vec<usize> {
        (0..horizon)
            .filter(|&s| (0..layers).any(|l| matches!(self.at(s, l), Route::Sparse { .. })))
            .collect()
    }

    pub fn describe(&self) -> String {
        let tau = match &self.tau {
            Schedule::Const(t) => format!("tau {t}"),
            Schedule::PerStep { values, .. } => format!(
                "tau {}",
                values
                    .iter()
                    .map(|(s, t)| format!("{s}:{t}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            other => format!("tau {other:?}"),
        };
        format!(
            "dense steps {} , dense layers {}, {tau}",
            self.dense_steps, self.dense_layers
        )
        .replace(" ,", ",")
    }
}

/// `tau = 1.0` or `tau = { "1" = 1.0, "2" = 1.25 }` (steps not listed take
/// `default`, 1.0 unless given as `default = x`).
pub fn parse_tau(v: &toml::Value) -> Result<Schedule<f64>, String> {
    let num = |v: &toml::Value| -> Option<f64> {
        v.as_float().or_else(|| v.as_integer().map(|i| i as f64))
    };
    match v {
        toml::Value::Table(t) => {
            let mut values = BTreeMap::new();
            let mut default = 1.0;
            for (k, v) in t {
                let tau = num(v).ok_or_else(|| format!("tau.{k} must be a number"))?;
                if k == "default" {
                    default = tau;
                    continue;
                }
                let step = k
                    .parse::<usize>()
                    .map_err(|_| format!("tau: key {k:?} is not a step index"))?;
                values.insert(step, tau);
            }
            Ok(Schedule::PerStep { values, default })
        }
        other => num(other)
            .map(Schedule::Const)
            .ok_or_else(|| format!("tau = {other}: expected a number or a step table")),
    }
}

/// A VSA sparsity per `(step, layer)` (the `vsa` technique's schedule):
/// `dense_steps` / `dense_layers` run at `dense_sparsity` (0 keeps every
/// tile; the compression branch stays on), other steps at their `per_step`
/// value, else the default (the technique's `sparsity`, else the recipe's).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VsaSchedule {
    pub per_step: BTreeMap<usize, f64>,
    pub dense_steps: StepSet,
    pub dense_layers: StepSet,
    pub dense_sparsity: f64,
}

impl VsaSchedule {
    /// No per-step or per-layer entry: every call runs at the default.
    pub fn is_uniform(&self) -> bool {
        self.per_step.is_empty() && self.dense_steps.is_empty() && self.dense_layers.is_empty()
    }

    pub fn at(&self, default: f64, step: usize, layer: usize) -> f64 {
        if self.dense_steps.contains(step) || self.dense_layers.contains(layer) {
            self.dense_sparsity
        } else {
            self.per_step.get(&step).copied().unwrap_or(default)
        }
    }

    pub fn describe(&self, default: f64) -> String {
        let mut parts = vec![format!("sparsity {default}")];
        for (s, v) in &self.per_step {
            parts.push(format!("step {s}: {v}"));
        }
        if !self.dense_steps.is_empty() {
            parts.push(format!("steps {} at {}", self.dense_steps, self.dense_sparsity));
        }
        if !self.dense_layers.is_empty() {
            parts.push(format!("layers {} at {}", self.dense_layers, self.dense_sparsity));
        }
        parts.join(", ")
    }
}

/// `sparsity = 0.9` or `sparsity = { default = 0.9, 0 = 0.8 }`: the default
/// (if given) and the per-step values. Every value must be in `[0, 1)`.
#[allow(clippy::type_complexity)]
pub fn parse_sparsity(v: &toml::Value) -> Result<(Option<f64>, BTreeMap<usize, f64>), String> {
    let num = |v: &toml::Value, what: &str| -> Result<f64, String> {
        let x = v
            .as_float()
            .or_else(|| v.as_integer().map(|i| i as f64))
            .ok_or_else(|| format!("{what} must be a number"))?;
        if (0.0..1.0).contains(&x) {
            Ok(x)
        } else {
            Err(format!("{what} = {x}: need [0, 1)"))
        }
    };
    match v {
        toml::Value::Table(t) => {
            let mut values = BTreeMap::new();
            let mut default = None;
            for (k, v) in t {
                if k == "default" {
                    default = Some(num(v, "sparsity.default")?);
                    continue;
                }
                let step = k
                    .parse::<usize>()
                    .map_err(|_| format!("sparsity: key {k:?} is not a step index"))?;
                values.insert(step, num(v, &format!("sparsity.{k}"))?);
            }
            Ok((default, values))
        }
        other => Ok((Some(num(other, "sparsity")?), BTreeMap::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vsa_schedules_resolve_per_step_and_layer() {
        let s = VsaSchedule {
            per_step: [(0, 0.8)].into(),
            dense_steps: StepSet::empty(),
            dense_layers: StepSet::parse("0,49").unwrap(),
            dense_sparsity: 0.5,
        };
        assert_eq!(s.at(0.9, 0, 5), 0.8);
        assert_eq!(s.at(0.9, 1, 5), 0.9);
        assert_eq!(s.at(0.9, 3, 0), 0.5);
        assert_eq!(s.at(0.9, 3, 49), 0.5);
        assert!(VsaSchedule::default().is_uniform());
        let v: toml::Value = toml::from_str::<toml::Table>("s = { default = 0.9, 0 = 0.8 }")
            .unwrap()["s"]
            .clone();
        let (d, m) = parse_sparsity(&v).unwrap();
        assert_eq!((d, m.get(&0).copied()), (Some(0.9), Some(0.8)));
        let bad: toml::Value = toml::Value::Float(1.0);
        assert!(parse_sparsity(&bad).is_err());
    }

    #[test]
    fn step_sets_parse_like_parse_steps() {
        let s = StepSet::parse("1-2,5,7-9").unwrap();
        assert_eq!(s.members(12), vec![1, 2, 5, 7, 8, 9]);
        assert_eq!(StepSet::parse("9-7").unwrap().members(12), vec![7, 8, 9]);
        assert!(StepSet::parse("").unwrap().is_empty());
        assert_eq!(StepSet::parse("0,4-").unwrap().members(7), vec![0, 4, 5, 6]);
        assert!(
            StepSet::parse("1,x").is_err(),
            "a typo is an error, not 'never'"
        );
        assert_eq!(StepSet::first(3).members(10), vec![0, 1, 2]);
        assert!(StepSet::first(0).is_empty());
        assert_eq!(StepSet::parse("0,4-").unwrap().to_string(), "0,4-");
    }

    #[test]
    fn schedules_follow_the_python_constructors() {
        let s = Schedule::AtSteps {
            steps: StepSet::parse("16-28").unwrap(),
            value: true,
            default: false,
        };
        assert_eq!(s.truthy_steps(64), (16..=28).collect::<Vec<_>>());
        let b = Schedule::Before {
            n: 2,
            value: "hp",
            then: "lp",
        };
        assert_eq!((b.at(1, ""), b.at(2, "")), ("hp", "lp"));
        let st = Schedule::ByStage {
            stages: [("stage2".to_string(), Schedule::Const(true))].into(),
            default: false,
        };
        assert!(!st.at(0, "stage1") && st.at(0, "stage2"));
        assert_eq!(st.truthy_steps(3), vec![0, 1, 2]);
    }

    #[test]
    fn enabled_values_parse() {
        let v: toml::Value = toml::from_str::<toml::Table>("a = \"0-3\"").unwrap()["a"].clone();
        assert_eq!(
            Schedule::parse_enabled(&v).unwrap().truthy_steps(8),
            vec![0, 1, 2, 3]
        );
        let v = toml::Value::Boolean(false);
        assert!(Schedule::parse_enabled(&v)
            .unwrap()
            .truthy_steps(64)
            .is_empty());
        let t: toml::Table = toml::from_str("before = 2").unwrap();
        assert_eq!(
            Schedule::parse_enabled(&toml::Value::Table(t))
                .unwrap()
                .truthy_steps(5),
            vec![0, 1]
        );
    }

    #[test]
    fn sparse_route_is_dense_on_either_axis() {
        let r = SparseRoute {
            dense_steps: StepSet::first(10),
            dense_layers: StepSet::first(2),
            tau: Schedule::Const(1.0),
        };
        assert_eq!(r.at(9, 30), Route::Dense);
        assert_eq!(r.at(10, 1), Route::Dense);
        assert_eq!(r.at(10, 2), Route::Sparse { tau: 1.0 });
        assert_eq!(r.sparse_steps(12, 50), vec![10, 11]);
        let t: toml::Table = toml::from_str("tau = { 1 = 1.0, 2 = 1.25, 3 = 1.5 }").unwrap();
        let tau = parse_tau(&t["tau"]).unwrap();
        assert_eq!(
            (tau.at(2, ""), tau.at(3, ""), tau.at(7, "")),
            (1.25, 1.5, 1.0)
        );
    }
}
