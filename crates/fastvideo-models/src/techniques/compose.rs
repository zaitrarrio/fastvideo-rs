//! `compose()`: capability check, seam-conflict check, ordered [`Plan`].
//!
//! Port of sol-engine `techniques/compose.py`:
//!
//! 1. capability (type) check: every item's required capabilities must be
//!    in the model's spec (:163-171);
//! 2. conflict (effect) check (:60-94):
//!    * an exclusive seam has at most one active writer across the plan;
//!      two runtime techniques only clash when their enabled schedules
//!      overlap, anything involving a transform always clashes (:65-80);
//!    * inside one runtime phase, a write-read on a shared seam between two
//!      co-active techniques is order-ambiguous (:82-93);
//! 3. deterministic order: transforms by phase, then runtime techniques by
//!    phase (:176-178).
//!
//! A technique whose schedule is never true (`enabled = false`) is inactive
//! and takes part in none of this (`_always_on`, :48-53): it is dropped from
//! the plan, which is how OFF stays byte-identical to not listing it.

use std::fmt;

use super::settings::Settings;
use super::technique::{Kind, ModelSpec, Seam, Technique};

/// compose's overlap horizon (`compose.py:60`, `horizon: int = 64`).
pub const HORIZON: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionError(pub String);

impl fmt::Display for CompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CompositionError {}

fn overlap(a: &dyn Technique, b: &dyn Technique, horizon: usize) -> bool {
    let sa = a.enabled().truthy_steps(horizon);
    let sb = b.enabled().truthy_steps(horizon);
    sa.iter().any(|s| sb.contains(s))
}

fn is_runtime(t: &dyn Technique) -> bool {
    matches!(t.kind(), Kind::Runtime(_))
}

/// Structural-conflict messages; empty means provably clean (:60-94).
pub fn check_conflicts(items: &[Box<dyn Technique>], horizon: usize) -> Vec<String> {
    let active: Vec<&dyn Technique> = items
        .iter()
        .map(|b| b.as_ref())
        .filter(|t| t.active_somewhere(horizon))
        .collect();
    let mut problems = Vec::new();
    for seam in Seam::ALL.iter().copied().filter(|s| s.is_exclusive()) {
        let writers: Vec<&dyn Technique> = active
            .iter()
            .copied()
            .filter(|t| t.writes().contains(&seam))
            .collect();
        let mut names: Vec<&str> = Vec::new();
        for (i, a) in writers.iter().enumerate() {
            for b in &writers[i + 1..] {
                if is_runtime(*a) && is_runtime(*b) && !overlap(*a, *b, horizon) {
                    continue;
                }
                names.push(a.name());
                names.push(b.name());
            }
        }
        if !names.is_empty() {
            names.sort_unstable();
            names.dedup();
            problems.push(format!(
                "exclusive seam '{}' has multiple active writers: {names:?}",
                seam.as_str()
            ));
        }
    }
    for (i, a) in active.iter().enumerate() {
        for b in &active[i + 1..] {
            let (Kind::Runtime(pa), Kind::Runtime(pb)) = (a.kind(), b.kind()) else {
                continue;
            };
            if pa != pb || !overlap(*a, *b, horizon) {
                continue;
            }
            let mut wr: Vec<&str> = a
                .writes()
                .iter()
                .filter(|s| b.reads().contains(s))
                .chain(b.writes().iter().filter(|s| a.reads().contains(s)))
                .filter(|s| !s.is_exclusive())
                .map(|s| s.as_str())
                .collect();
            wr.sort_unstable();
            wr.dedup();
            if !wr.is_empty() {
                problems.push(format!(
                    "write-read conflict on {wr:?} between '{}' and '{}' in the same phase {pa:?}",
                    a.name(),
                    b.name()
                ));
            }
        }
    }
    problems
}

/// An ordered, conflict-checked set of techniques for one model.
#[derive(Debug)]
pub struct Plan {
    pub model: &'static str,
    pub techniques: Vec<Box<dyn Technique>>,
}

impl Plan {
    /// The active technique of type `T`, if the plan has one.
    pub fn get<T: Technique>(&self) -> Option<&T> {
        self.techniques.iter().find_map(|t| t.downcast_ref::<T>())
    }

    /// The technique writing `seam`, if any (exclusive seams have at most one).
    pub fn writer(&self, seam: Seam) -> Option<&dyn Technique> {
        self.techniques
            .iter()
            .map(|t| t.as_ref())
            .find(|t| t.writes().contains(&seam))
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.techniques.iter().map(|t| t.name()).collect()
    }

    /// Every technique's process-wide settings (`Plan.apply_transforms`,
    /// :110-115). Two techniques setting one name differently is an error.
    pub fn settings(&self) -> Result<Settings, CompositionError> {
        let mut s = Settings::default();
        for t in &self.techniques {
            for (k, v) in t.settings() {
                s.set(k, &v, t.name()).map_err(CompositionError)?;
            }
        }
        Ok(s)
    }

    pub fn describe(&self) -> String {
        if self.techniques.is_empty() {
            return format!("{} techniques: none", self.model);
        }
        format!(
            "{} techniques: {}",
            self.model,
            self.techniques
                .iter()
                .map(|t| t.describe())
                .collect::<Vec<_>>()
                .join("; ")
        )
    }
}

/// Type-check, conflict-check and order `items` for `spec` (:154-179).
pub fn compose(
    items: Vec<Box<dyn Technique>>,
    spec: &ModelSpec,
    horizon: usize,
) -> Result<Plan, CompositionError> {
    let mut items: Vec<Box<dyn Technique>> = items
        .into_iter()
        .filter(|t| t.active_somewhere(horizon))
        .collect();
    for it in &items {
        let missing = spec.missing(it.required_capabilities());
        if !missing.is_empty() {
            return Err(CompositionError(format!(
                "'{}' requires capabilities {missing:?} not provided by model '{}' (provided: {:?})",
                it.name(),
                spec.name,
                spec.capabilities
            )));
        }
    }
    let problems = check_conflicts(&items, horizon);
    if !problems.is_empty() {
        return Err(CompositionError(format!(
            "conflicts:\n  - {}",
            problems.join("\n  - ")
        )));
    }
    items.sort_by_key(|t| t.kind().order());
    Ok(Plan {
        model: spec.name,
        techniques: items,
    })
}
