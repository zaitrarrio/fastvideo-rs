//! Deploy targets for fv-serve (docs/serve/design.md §6, WP-16).
//!
//! - [`runpod`]: the Runpod serverless queue worker (job-take, ping, stop,
//!   progress, stream, done with retries) over a [`runpod::Transport`], and a
//!   local simulator of the queue ([`runpod::sim`]).
//! - [`dispatch`]: runs the **native** job envelope (`kind: http | stream |
//!   info`) against the fv-serve router in-process; also the Vast
//!   forwarder route `POST /fv/v1/forward`.
//! - [`env`]: `RUNPOD_*` / `VAST_*` discovery: platform, public IP, mapped
//!   ports, ICE candidates, public base URL.
//! - [`vast`]: the Vast `env` dict, onstart script and the PyWorker forwarder
//!   (`deploy/vast/worker.py`).
//! - [`boot`]: `fv-boot`, the one boot sequence of a serve container (driver
//!   check, config, weights plan, sidecar, tunnel and phase reports,
//!   watchdog, exec fv-serve; docs/adr/0014-fv-boot.md). The binary needs the
//!   `boot` feature (its HTTP client).
//!
//! Deploy scripts: `scripts/serve/{runpod-pod,runpod-endpoint,vast,vast-serverless}.sh`.

pub mod boot;
pub mod dispatch;
pub mod env;
pub mod runpod;
pub mod vast;
