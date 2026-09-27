//! Warm model pool: per-(executor, model) residency and the engine's
//! readiness (design §3.6, §6.3 "Warm start").
//!
//! Every model a backend declares `resident` is loaded by its executor before
//! the engine reports [`Readiness::Ready`]. Non-resident models load on demand
//! only in swap mode (risk R18), evicting what the executor held.

use std::collections::BTreeMap;

use fastvideo_protocol::{ApiError, ModelId};
use serde::{Deserialize, Serialize};

/// Engine readiness (design §3.6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Readiness {
    Loading { done: u32, total: u32 },
    Ready,
    Failed(String),
}

impl Readiness {
    pub fn is_ready(&self) -> bool {
        matches!(self, Readiness::Ready)
    }
}

/// One model's state on one executor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Residency {
    /// Declared resident, load not started.
    Pending,
    Loading {
        stage: Option<String>,
        done: u64,
        total: u64,
    },
    Resident,
    /// Not loaded (swap-mode model, or evicted).
    Unloaded,
    Failed(ApiError),
}

/// One row of the pool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoolEntry {
    pub executor: usize,
    pub model: ModelId,
    /// Counted by readiness.
    pub warm: bool,
    pub state: Residency,
}

/// Residency of every (executor, model).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelPool {
    entries: BTreeMap<(usize, ModelId), PoolEntry>,
}

impl ModelPool {
    /// Declares a model on an executor; `warm` ones must load before ready.
    pub fn declare(&mut self, executor: usize, model: ModelId, warm: bool) {
        let state = if warm {
            Residency::Pending
        } else {
            Residency::Unloaded
        };
        self.entries.insert(
            (executor, model.clone()),
            PoolEntry {
                executor,
                model,
                warm,
                state,
            },
        );
    }

    pub fn set(&mut self, executor: usize, model: &ModelId, state: Residency) {
        if let Some(e) = self.entries.get_mut(&(executor, model.clone())) {
            e.state = state;
        }
    }

    pub fn state(&self, executor: usize, model: &ModelId) -> Option<&Residency> {
        self.entries
            .get(&(executor, model.clone()))
            .map(|e| &e.state)
    }

    pub fn entries(&self) -> impl Iterator<Item = &PoolEntry> {
        self.entries.values()
    }

    /// Whether `model` is resident on any executor.
    pub fn is_resident(&self, model: &ModelId) -> bool {
        self.entries
            .values()
            .any(|e| &e.model == model && e.state == Residency::Resident)
    }

    /// The model's best state across executors: resident beats loading beats
    /// pending beats unloaded beats failed.
    pub fn model_state(&self, model: &ModelId) -> Option<Residency> {
        let rank = |r: &Residency| match r {
            Residency::Resident => 0,
            Residency::Loading { .. } => 1,
            Residency::Pending => 2,
            Residency::Unloaded => 3,
            Residency::Failed(_) => 4,
        };
        self.entries
            .values()
            .filter(|e| &e.model == model)
            .map(|e| &e.state)
            .min_by_key(|r| rank(r))
            .cloned()
    }

    /// Resident models, deduplicated, in id order.
    pub fn resident_models(&self) -> Vec<ModelId> {
        let mut v: Vec<ModelId> = self
            .entries
            .values()
            .filter(|e| e.state == Residency::Resident)
            .map(|e| e.model.clone())
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// `Failed` if any warm load failed, else `Loading` until every warm
    /// model is resident, else `Ready`.
    pub fn readiness(&self) -> Readiness {
        let warm: Vec<&PoolEntry> = self.entries.values().filter(|e| e.warm).collect();
        if let Some(e) = warm.iter().find_map(|e| match &e.state {
            Residency::Failed(err) => Some((e, err)),
            _ => None,
        }) {
            return Readiness::Failed(format!("loading `{}` failed: {}", e.0.model, e.1.message));
        }
        let total = warm.len() as u32;
        let done = warm
            .iter()
            .filter(|e| !matches!(e.state, Residency::Pending | Residency::Loading { .. }))
            .count() as u32;
        if done == total {
            Readiness::Ready
        } else {
            Readiness::Loading { done, total }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_tracks_warm_loads() {
        let mut p = ModelPool::default();
        let a = ModelId::new("a");
        let b = ModelId::new("b");
        p.declare(0, a.clone(), true);
        p.declare(0, b.clone(), false);
        p.declare(1, a.clone(), true);
        assert_eq!(p.readiness(), Readiness::Loading { done: 0, total: 2 });
        p.set(0, &a, Residency::Resident);
        assert_eq!(p.readiness(), Readiness::Loading { done: 1, total: 2 });
        assert!(p.is_resident(&a));
        assert!(!p.is_resident(&b));
        assert_eq!(p.model_state(&b), Some(Residency::Unloaded));
        p.set(1, &a, Residency::Resident);
        assert_eq!(p.readiness(), Readiness::Ready);
        assert_eq!(p.resident_models(), vec![a.clone()]);
        p.set(1, &a, Residency::Failed(ApiError::engine_failed("oom")));
        assert!(matches!(p.readiness(), Readiness::Failed(m) if m.contains("oom")));
        assert_eq!(p.model_state(&a), Some(Residency::Resident));
    }
}
