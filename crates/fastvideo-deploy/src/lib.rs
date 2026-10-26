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
//! - [`provider`], [`launch`], [`guard`], [`lifecycle`]: `fv-deploy`
//!   (docs/adr/0018-fv-deploy.md): the `Provider` trait with Runpod pods and
//!   serverless endpoints, the `Launch` schema built from a deployment
//!   preset, the shared money / region guards and the detached backstop, and
//!   the `up` / `smoke` / `down` lifecycle. The binary needs the `deploy`
//!   feature (its command line and HTTP client); the library is tested on
//!   the seams without it.
//!
//! - [`release`]: `fv-deploy templates | release | reconcile` (plan C3d,
//!   docs/adr/0018-fv-deploy.md §10): the Runpod serve templates, release
//!   channels (GHCR tags, the D1 history) and the D1 `deployments` ledger
//!   fv-control and the scripts share.
//!
//! Deploy scripts: `scripts/serve/{runpod-pod,runpod-endpoint,vast,vast-serverless}.sh`.

pub mod boot;
pub mod dispatch;
pub mod env;
pub mod guard;
pub mod launch;
pub mod lifecycle;
pub mod provider;
pub mod release;
pub mod runpod;
pub mod vast;
