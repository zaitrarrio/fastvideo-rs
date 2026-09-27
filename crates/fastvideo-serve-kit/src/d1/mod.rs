//! Cloudflare D1 job store (design §0 decision 7): jobs live in D1 (SQLite
//! over the D1 HTTP API), media in R2 through the S3 artifact store.
//!
//! - [`client`]: the `/query` HTTP API ([`D1Client`]) with retry/backoff.
//! - [`schema`]: the `jobs` table, indexes and migrations.
//! - [`D1JobStore`]: the [`fastvideo_protocol::JobStore`] with an in-memory
//!   authoritative cache for this worker's jobs and throttled progress writes.
//! - `mock` (feature `d1-mock`): a SQLite-backed stand-in for the HTTP API.
//!
//! The HTTPS transport needs the `fetch` feature.

pub mod client;
#[cfg(any(test, feature = "d1-mock"))]
pub mod mock;
pub mod schema;
mod store;

pub use client::{D1Client, D1Config, D1Error, D1Transport, RetryPolicy, Stmt, StmtResult};
#[cfg(feature = "fetch")]
pub use client::HttpD1Transport;
pub use store::{D1JobStore, D1Options, D1Stats};

#[cfg(test)]
mod tests;
