//! Where workers come from.
//!
//! - [`runpod`] (feature `runpod`): a Runpod serverless endpoint steered
//!   through `workersMin`/`workersMax`/scaler/idle timeout, and Runpod pods
//!   created from a template with GPU-type and region fallback.
//! - [`crate::sim`]: the simulated provider used by tests and the
//!   simulation harness.

#[cfg(feature = "runpod")]
pub mod runpod;

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::config::{PoolConfig, PoolKind};
use crate::types::{PoolDecision, PoolObservation};

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("provider {0}")]
    Api(String),
    #[error("provider: no capacity: {0}")]
    NoCapacity(String),
    #[error("provider: {0}")]
    Other(String),
}

/// What an `apply` did.
#[derive(Clone, Debug, Default, serde::Serialize, PartialEq)]
pub struct ApplyReport {
    pub created: Vec<String>,
    pub deleted: Vec<String>,
    pub drained: Vec<String>,
    pub undrained: Vec<String>,
    pub endpoint_patched: bool,
    /// Non-fatal problems (a GPU type out of stock, one delete failed, ...).
    pub notes: Vec<String>,
}

/// One kind of worker source.
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &'static str;
    /// The pool's workers (and endpoint settings for serverless).
    async fn observe(&self, pool: &PoolConfig, now_s: f64) -> Result<PoolObservation, ProviderError>;
    /// Carry out a decision. Never deletes a worker with `busy > 0`.
    async fn apply(&self, pool: &PoolConfig, decision: &PoolDecision, now_s: f64) -> Result<ApplyReport, ProviderError>;
}

/// The account balance in dollars.
#[async_trait::async_trait]
pub trait BalanceSource: Send + Sync {
    async fn balance_usd(&self) -> Result<f64, ProviderError>;
}

/// A balance that never changes (tests, dry runs without an API key).
pub struct FixedBalance(pub f64);

#[async_trait::async_trait]
impl BalanceSource for FixedBalance {
    async fn balance_usd(&self) -> Result<f64, ProviderError> {
        Ok(self.0)
    }
}

/// Providers by pool kind, plus the balance.
#[derive(Clone)]
pub struct Providers {
    pub by_kind: BTreeMap<&'static str, Arc<dyn Provider>>,
    pub balance: Arc<dyn BalanceSource>,
}

impl Providers {
    pub fn new(balance: Arc<dyn BalanceSource>) -> Self {
        Self { by_kind: BTreeMap::new(), balance }
    }
    pub fn with(mut self, kind: PoolKind, p: Arc<dyn Provider>) -> Self {
        self.by_kind.insert(kind_key(kind), p);
        self
    }
    pub fn get(&self, kind: PoolKind) -> Option<&Arc<dyn Provider>> {
        self.by_kind.get(kind_key(kind))
    }
}

fn kind_key(k: PoolKind) -> &'static str {
    match k {
        PoolKind::Serverless => "serverless",
        PoolKind::Pod => "pod",
    }
}
