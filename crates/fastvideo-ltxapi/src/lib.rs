//! LTX API adapter: `/v2/*` async jobs, `/v1/*` sync, `/v1/upload`, and the
//! `403 permission_error` stubs (design §4.5, research-ltx-api.md).
//!
//! - [`routes`]: [`router`] with every LTX route and [`LtxConfig`].
//! - [`request`]: request validation and normalization (`SubmitEndpoint`).
//! - [`models`]: `model` ids → engine tiers, and the support matrix.
//! - [`v2`]: `202 {id, created_at}` and the `oneOf` status shapes.
//! - [`v1`]: sync MP4 replies and the per-key concurrency limit.
//! - [`upload`]: `POST /v1/upload` bodies (`ltx://uploads/<token>`).
//! - [`stubs`]: endpoints without an engine path.
//! - [`error`]: the error envelope and its eleven types.
//!
//! Owned by WP-08 (docs/serve/design.md §8).

pub mod error;
pub mod models;
pub mod request;
pub mod routes;
pub mod stubs;
pub mod upload;
pub mod v1;
pub mod v2;

pub use error::{Api, LtxErrorType, LtxProtocol};
pub use models::{LtxModel, LtxModels, ModelClass, Target};
pub use request::{Endpoint, Submit};
pub use routes::{router, LtxConfig};
pub use v1::V1View;
pub use v2::V2View;
