//! Shared axum glue for every fv-serve adapter (design §2.1, WP-05).
//!
//! - [`ctx`]: [`ServeCtx`], the state every route shares, and the
//!   [`EngineGate`] seam to the engine service.
//! - [`auth`]: `none | keys | trust-gateway` and per-API schemes.
//! - [`store`]: [`MemJobStore`], an in-memory `JobStore` with optional
//!   durable JSON manifests, restart recovery and expiry sweep.
//! - [`artifacts`]: [`ArtifactStore`]s (local disk with HMAC-signed
//!   `/files/...` URLs, S3-compatible with SigV4 presigned URLs).
//! - [`uploads`]: [`UploadStore`] and `PUT /uploads/{token}`.
//! - [`ingest`]: media staging (HTTPS fetch, data URIs, upload ids) with
//!   per-API limits, an SSRF guard and probing hooks.
//! - [`callback`]: MiniMax callbacks (challenge echo) and fal webhooks
//!   (Ed25519) with retry schedules.
//! - [`d1`]: [`D1JobStore`], the Cloudflare D1 job store (design §0.7).
//! - [`keys`]: minted API keys ([`KeyStore`], memory / file / D1), the
//!   admin token and the `/fv/v1/admin/keys` routes.
//! - [`sse`]: server-sent events from an `SseSpec`.
//! - [`handlers`]: `HttpReply` -> axum response, and the generic
//!   `submit`/`status`/`result` handlers.
//! - [`events`]: applies engine progress to jobs and fires callbacks.
//!
//! Outbound HTTP (media fetch, callbacks, webhooks, S3 upload) is behind the
//! `fetch` feature.
//!
//! Owned by WP-05 (docs/serve/design.md §8).

pub mod artifacts;
pub mod auth;
pub mod callback;
pub mod ctx;
pub mod d1;
pub mod events;
pub mod handlers;
pub mod ingest;
pub mod keys;
pub mod net;
pub mod sse;
pub mod store;
pub mod uploads;

pub use artifacts::{
    files_router, ArtifactBody, ArtifactMeta, ArtifactStore, LocalArtifactStore, S3ArtifactStore, S3Config,
    UrlKey,
};
pub use auth::{Auth, AuthMode, AuthPolicy, KeyRing, Scheme};
pub use callback::{CallbackRender, CallbackSender, Delivery, RetrySchedule, WebhookSigner};
pub use d1::{D1Client, D1Config, D1JobStore, D1Options};
pub use ctx::{EngineGate, SafetyFilter, ServeConfig, ServeCtx};
pub use events::{apply_event, FinishedOutput, JobEvent};
pub use handlers::{into_response, SubmitOpts};
pub use keys::{admin_routes, AdminToken, KeyStore};
pub use ingest::{IngestPolicy, Ingestor, KindLimits, Prober};
pub use store::MemJobStore;
pub use uploads::{UploadStore, UploadTicket};

/// Lowercase hex helpers (no extra dependency).
pub(crate) mod hex {
    pub fn encode(b: &[u8]) -> String {
        const D: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(b.len() * 2);
        for x in b {
            s.push(D[(x >> 4) as usize] as char);
            s.push(D[(x & 15) as usize] as char);
        }
        s
    }
    pub fn decode(s: &str) -> Option<Vec<u8>> {
        if s.len() % 2 != 0 {
            return None;
        }
        let nib = |c: u8| match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        };
        s.as_bytes()
            .chunks(2)
            .map(|p| Some(nib(p[0])? << 4 | nib(p[1])?))
            .collect()
    }
}

/// A fresh unguessable token: 32 lowercase hex characters (122 random bits).
pub fn random_token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}
