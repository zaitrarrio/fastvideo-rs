//! The `Provider` trait of fv-deploy (docs/adr/0018-fv-deploy.md, plan C3):
//! one interface every place GPUs come from implements, promoted from
//! `fastvideo-autoscale`'s provider module (review §7.4).
//!
//! - [`types`]: [`Launch`], [`Instance`], [`InstanceState`], [`Offer`],
//!   [`Account`] (serde + schemars; [`schema`] exports them).
//! - [`runpod`]: [`runpod::Pod`] and [`runpod::Serverless`] over the Runpod
//!   REST v1 / GraphQL APIs, on the blocking [`Http`] seam fv-boot uses.
//! - [`Reach`]: how the readiness poll and the smoke reach a running instance
//!   (a pod's HTTPS proxy; a serverless endpoint's queue, one native job per
//!   call).
//!
//! The trait is blocking, like the rest of this crate's CLI half (fv-boot):
//! every caller is a sequential command-line flow, and the seams (`Http`,
//! `Clock`) run the whole lifecycle in tests without a network or a clock.
//! An async caller (the autoscaler) runs it on `spawn_blocking`.

pub mod runpod;
pub mod schema;
pub mod types;

#[cfg(test)]
pub(crate) mod tests;

use serde_json::Value;

use crate::boot::sys::Method;

pub use types::{
    Account, ApiCall, CloudType, Cursor, Disk, GpuSpec, Guards, Instance, InstanceId,
    InstanceState, Launch, LaunchSource, LogPage, Offer, OfferQuery, Placement, ProviderKind,
    ServerlessSpec, VolumeMount, WeightsSource, LAUNCH_SCHEMA,
};

/// Why a provider call failed. Messages never carry the API key (it is only
/// ever a header; response bodies are scrubbed of it).
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("{what}: HTTP {status}: {body}")]
    Api {
        what: String,
        status: u16,
        body: String,
    },
    #[error("no capacity: {0}")]
    NoCapacity(String),
    #[error("{what}: transport: {why}")]
    Transport { what: String, why: String },
    #[error("{what}: unexpected answer: {why}")]
    Decode { what: String, why: String },
    #[error("{0}")]
    Unsupported(String),
    #[error("{0} is not set")]
    MissingKey(&'static str),
}

/// One readiness probe of a running instance (`/ping`: 502 / 404 while the
/// container starts, 204 while fv-serve loads, 200 ready).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Probe {
    /// No answer yet (or an error page); why.
    Down(String),
    /// fv-serve answers but is loading (204).
    Loading,
    /// Ready (200).
    Ready,
}

/// A running instance's HTTP API, as the readiness poll and the smoke use it.
pub trait Reach {
    /// One readiness probe.
    fn ping(&mut self) -> Probe;
    /// One native API call: the HTTP status and the JSON body (`Value::Null`
    /// when empty, a string when not JSON).
    fn call(
        &mut self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<(u16, Value), ProviderError>;
}

/// One place GPUs come from (docs/adr/0018-fv-deploy.md §2).
pub trait Provider {
    fn kind(&self) -> ProviderKind;
    /// Quotes for the queried GPU types (one [`Offer`] per type, `usd_per_hr`
    /// `None` when the provider gave no price).
    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>, ProviderError>;
    /// The exact requests [`Provider::create`] sends, without sending them
    /// (`fv-deploy plan`).
    fn payload(&self, launch: &Launch) -> Result<Vec<ApiCall>, ProviderError>;
    /// Creates the instance (one request sequence, no retry, no guard: the
    /// caller, `crate::lifecycle`, applies the guards around it).
    fn create(&self, launch: &Launch) -> Result<Instance, ProviderError>;
    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError>;
    fn logs(&self, id: &InstanceId, since: Option<&Cursor>) -> Result<LogPage, ProviderError>;
    /// Deletes it; an instance already gone is not an error.
    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError>;
    /// The balance, for the shared balance floor.
    fn account(&self) -> Result<Account, ProviderError>;
    /// The instance's API (`key`: the run's API key for fv-serve, when the
    /// launch set `FV_API_KEYS`).
    fn reach<'a>(
        &'a self,
        instance: &Instance,
        key: Option<&str>,
    ) -> Result<Box<dyn Reach + 'a>, ProviderError>;
}

/// `text` with every occurrence of `secret` replaced (no-op for an empty
/// secret), cut to `max` characters.
pub fn scrub(text: &str, secret: &str, max: usize) -> String {
    let t = if secret.is_empty() {
        text.to_owned()
    } else {
        text.replace(secret, "<redacted>")
    };
    t.chars().take(max).collect()
}

/// A body as JSON: `Null` when empty, a string when it is not JSON.
pub fn body_json(body: &[u8]) -> Value {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Value::Null;
    }
    serde_json::from_slice(body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(body).into_owned()))
}
