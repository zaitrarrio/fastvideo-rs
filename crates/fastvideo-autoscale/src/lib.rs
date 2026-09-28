//! Autoscaler for the fv-serve gateway (docs/serve/gateway.md,
//! "Autoscaling"): one GPU worker pool per model family, steered from the
//! gateway's queue signals.
//!
//! - [`config`]: the `[autoscale]` table (pools, budgets, prices, lease).
//! - [`policy`]: the pure, deterministic scaling policy.
//! - [`provider`]: where workers come from (Runpod serverless endpoints and
//!   Runpod pods behind feature `runpod`; the simulator in [`sim`]).
//! - [`gateway`]: the interface the gateway implements (signals, worker
//!   registry).
//! - [`controller`]: the leader-elected loop; [`lease`] the D1 lease row;
//!   [`admin`] `/fv/v1/admin/autoscale`.
//! - [`sim`]: the simulated world, traces and the harness behind
//!   `fv-autoscale-sim`.

pub mod admin;
pub mod config;
pub mod controller;
pub mod gateway;
pub mod lease;
pub mod policy;
pub mod provider;
pub mod sim;
pub mod types;

pub use config::{AutoscaleConfig, PoolConfig, PoolKind};
pub use controller::Controller;
pub use policy::Policy;
pub use types::{Action, PoolDecision, PoolObservation, PoolSignals, Worker, WorkerState};
