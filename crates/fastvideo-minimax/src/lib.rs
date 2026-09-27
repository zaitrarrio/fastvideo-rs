//! MiniMax V2 `/v2/video_generation` family (design §4.3, WP-07).
//!
//! Wire source: `docs/serve/research-minimax-fastvideo.md` §1.2-1.7. V1
//! (`/v1/video_generation`, `/v1/files/*`) is a non-goal (design §1.2).
//!
//! | Route | Module |
//! |---|---|
//! | `POST /v2/video_generation` → `{"task_id"}` | [`create`] |
//! | `GET /v2/query/video_generation/{task_id}` → `{"task": VideoTask}` | [`query`] |
//! | `GET /v2/query/video_generation` → `{"items", "total"}` | [`query`] |
//! | `DELETE /v2/video_generation/{task_id}` → `{"task_id","action","status"}` | [`delete`] |
//! | `POST /v2/h3_context_ir`, `POST /v2/video_regeneration` → 400 | [`create`] |
//!
//! Callbacks (`callback_url`): serve-kit's sender does the challenge echo;
//! this crate renders the `{"task": VideoTask}` bodies ([`callback`]).
//! Errors use the `OaiError` envelope with real HTTP codes ([`error`]).
//!
//! # Models and tiers (design §0.3, §0.6)
//!
//! | `model` | Engine model | Durations | Resolutions |
//! |---|---|---|---|
//! | `MiniMax-H3` | `minimax.models` / alias, else the H3 `max` tier, else the first H3 model | 4–15 s | `768P`, `2K` (2K → 400 gap) |
//! | `MiniMax-H3-Max` | alias, else the H3 `max` tier | 5–15 s (4 → 400) | `480P`, `768P` |
//! | `MiniMax-H3-Turbo` | alias, else the H3 `turbo` tier | 4–15 s | `480P`, `768P` |
//! | `MiniMax-H3-Draft` | alias, else the H3 `draft` tier | 4–15 s | `480P`, `768P` |
//!
//! Turbo and Draft are our own ids; they take the H3-Max request shape
//! (including `extra`) with the H3 duration range, since the engine grid
//! admits 4 s.
//!
//! Owned by WP-07 (docs/serve/design.md §8).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::Router;
use fastvideo_protocol::{
    resolve_model, resolve_tier, ApiError, BatchProtocol, ErrorCtx, Family, HttpReply, JobId,
    ModelCaps, ProtocolId, Tier,
};
use fastvideo_serve_kit::{IngestPolicy, KindLimits, ServeCtx};

pub mod callback;
pub mod create;
pub mod delete;
pub mod error;
mod limits;
pub mod query;

pub use callback::callback_renderer;
pub use create::{ContentItem, CreateBody, CreateEndpoint, MediaUrl};
pub use error::{classify, render_oai_error, OaiClass};
pub use query::{nearest_ratio, task_json, TaskView};

/// The model ids this API accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MiniMaxModel {
    /// `MiniMax-H3`.
    H3,
    /// `MiniMax-H3-Max`.
    H3Max,
    /// `MiniMax-H3-Turbo` (design §0.3).
    H3Turbo,
    /// `MiniMax-H3-Draft` (design §0.6).
    H3Draft,
}

impl MiniMaxModel {
    pub const ALL: [MiniMaxModel; 4] = [
        MiniMaxModel::H3,
        MiniMaxModel::H3Max,
        MiniMaxModel::H3Turbo,
        MiniMaxModel::H3Draft,
    ];

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str() == s)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            MiniMaxModel::H3 => "MiniMax-H3",
            MiniMaxModel::H3Max => "MiniMax-H3-Max",
            MiniMaxModel::H3Turbo => "MiniMax-H3-Turbo",
            MiniMaxModel::H3Draft => "MiniMax-H3-Draft",
        }
    }

    /// The tier the name selects when no alias is configured. `MiniMax-H3`
    /// has none (it prefers `Max`, then any H3 model).
    pub fn tier(&self) -> Option<Tier> {
        match self {
            MiniMaxModel::H3 => None,
            MiniMaxModel::H3Max => Some(Tier::Max),
            MiniMaxModel::H3Turbo => Some(Tier::Turbo),
            MiniMaxModel::H3Draft => Some(Tier::Draft),
        }
    }

    /// Shortest `duration` in seconds: 4 for `MiniMax-H3` (and our Turbo /
    /// Draft), 5 for `MiniMax-H3-Max` (research §1.3: "4 is rejected").
    pub fn min_duration(&self) -> u32 {
        match self {
            MiniMaxModel::H3Max => 5,
            _ => 4,
        }
    }

    /// Longest `duration` in seconds.
    pub fn max_duration(&self) -> u32 {
        15
    }

    /// Resolutions the model's schema lists.
    pub fn resolutions(&self) -> &'static [Resolution] {
        match self {
            MiniMaxModel::H3 => &[Resolution::P768, Resolution::K2],
            _ => &[Resolution::P480, Resolution::P768],
        }
    }

    /// Whether `resolution` may be omitted (H3-Max "defaults to 768P").
    pub fn resolution_optional(&self) -> bool {
        !matches!(self, MiniMaxModel::H3)
    }

    /// Whether `extra` (`prompt_expansion_mode`) is accepted.
    pub fn accepts_extra(&self) -> bool {
        !matches!(self, MiniMaxModel::H3)
    }
}

impl std::fmt::Display for MiniMaxModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `resolution` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Resolution {
    P480,
    P768,
    /// Short edge 1440 (strobe's reading); H3 answers `Unsupported(H3Resolution2K)`.
    K2,
}

impl Resolution {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "480P" => Some(Resolution::P480),
            "768P" => Some(Resolution::P768),
            "2K" => Some(Resolution::K2),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Resolution::P480 => "480P",
            Resolution::P768 => "768P",
            Resolution::K2 => "2K",
        }
    }
    pub fn short_edge(&self) -> u32 {
        match self {
            Resolution::P480 => 480,
            Resolution::P768 => 768,
            Resolution::K2 => 1440,
        }
    }
}

/// MiniMax adapter settings (the `[minimax]` config section).
#[derive(Clone, Debug)]
pub struct MiniMaxConfig {
    /// `content.url` lifetime; re-signed on every query (`minimax.url_ttl`, 24 h).
    pub url_ttl: Duration,
    /// `minimax.models`: MiniMax model name -> engine model id or alias. Wins
    /// over the engine alias table and the tier fallback.
    pub models: BTreeMap<String, String>,
    /// Creates per minute per API key (research §1.7: 300). 0 disables.
    pub rpm: u32,
    /// Queued + running tasks per API key (research §1.7: 30). 0 disables.
    pub max_in_flight: u32,
    /// Request body cap (research §1.3: 64 MB), answered with 400 `(2013)`.
    pub body_max: usize,
    /// Media ingestion limits (research §1.3).
    pub ingest: IngestPolicy,
}

const MB: u64 = 1024 * 1024;

impl Default for MiniMaxConfig {
    fn default() -> Self {
        Self {
            url_ttl: Duration::from_secs(24 * 3600),
            models: BTreeMap::new(),
            rpm: 300,
            max_in_flight: 30,
            body_max: 64 * 1024 * 1024,
            ingest: ingest_policy(),
        }
    }
}

/// Research §1.3 media limits: images 30 MB, reference video 50 MB,
/// reference audio 15 MB; data URIs bounded by the 64 MB body.
pub fn ingest_policy() -> IngestPolicy {
    let d = IngestPolicy::default();
    IngestPolicy {
        image: KindLimits { max_bytes: 30 * MB, data_uri_max_encoded: 41 * MB, ..d.image },
        video: KindLimits { max_bytes: 50 * MB, data_uri_max_encoded: 64 * MB, ..d.video },
        audio: KindLimits { max_bytes: 15 * MB, data_uri_max_encoded: 21 * MB, ..d.audio },
        ..d
    }
}

/// The MiniMax V2 protocol: ids, errors, model resolution, admission.
#[derive(Debug)]
pub struct MiniMax {
    cfg: MiniMaxConfig,
    limiter: limits::RateLimiter,
}

impl Default for MiniMax {
    fn default() -> Self {
        Self::new(MiniMaxConfig::default())
    }
}

impl MiniMax {
    pub fn new(cfg: MiniMaxConfig) -> Self {
        Self { limiter: limits::RateLimiter::new(cfg.rpm), cfg }
    }

    pub fn config(&self) -> &MiniMaxConfig {
        &self.cfg
    }

    /// The query / list / callback view.
    pub fn view(&self) -> TaskView {
        TaskView { url_ttl: self.cfg.url_ttl }
    }

    /// Picks the engine model for a MiniMax model name, in order:
    /// `minimax.models`, the server alias of the name itself, the server
    /// alias of the canonical tier alias (`h3-max` / `h3-turbo` /
    /// `h3-draft`; `MiniMax-H3` uses `h3-max`), the model tagged with that
    /// tier, and for `MiniMax-H3` finally the first H3 model. `alias` is the
    /// server alias table (`EngineGate::alias`).
    ///
    /// Also returns the tier the name resolved through (`None` for explicit
    /// config or name aliases), so a result can be marked with its tier even
    /// when the engine's caps carry no tag.
    pub fn resolve_caps<'a>(
        &self,
        name: &str,
        alias: &dyn Fn(&str) -> Option<String>,
        models: &'a [ModelCaps],
    ) -> Result<(&'a ModelCaps, Option<Tier>), ApiError> {
        if let Some(target) = self.cfg.models.get(name) {
            return resolve_model(target, alias, models).map(|c| (c, None));
        }
        if alias(name).is_some() {
            return resolve_model(name, alias, models).map(|c| (c, None));
        }
        let not_served = || {
            ApiError::invalid_param("model", format!("model `{name}` is not served here"))
        };
        let model = MiniMaxModel::parse(name).ok_or_else(not_served)?;
        let tier = model.tier().unwrap_or(Tier::Max);
        let canonical = match tier {
            Tier::Max => "h3-max",
            Tier::Turbo => "h3-turbo",
            Tier::Draft => "h3-draft",
        };
        if alias(canonical).is_some() {
            if let Ok(c) = resolve_model(canonical, alias, models) {
                return Ok((c, Some(tier)));
            }
        }
        if let Ok(c) = resolve_tier(Family::H3, tier, models) {
            return Ok((c, Some(tier)));
        }
        match model {
            MiniMaxModel::H3 => models
                .iter()
                .find(|m| m.family == Family::H3)
                .map(|c| (c, None))
                .ok_or_else(not_served),
            _ => Err(not_served()),
        }
    }

    /// The router for every MiniMax route. The `ServeCtx` must register
    /// [`callback_renderer`] for `ProtocolId::MiniMaxV2` so `callback_url`
    /// receives status posts.
    pub fn router(self: &Arc<Self>) -> Router<ServeCtx> {
        type St = State<ServeCtx>;
        type Id = Path<String>;
        let (c, q, l, d, u) = (self.clone(), self.clone(), self.clone(), self.clone(), self.clone());
        let u2 = self.clone();
        Router::new()
            .route(
                "/v2/video_generation",
                post(move |st: St, h: HeaderMap, b: Body| create::handle(c.clone(), st, h, b)),
            )
            .route(
                "/v2/video_generation/{task_id}",
                axum::routing::delete(move |st: St, p: Id, h: HeaderMap| delete::handle(d.clone(), st, p, h)),
            )
            .route(
                "/v2/query/video_generation",
                get(move |st: St, qs: Query<Vec<(String, String)>>, h: HeaderMap| {
                    query::list_handle(l.clone(), st, qs, h)
                }),
            )
            .route(
                "/v2/query/video_generation/{task_id}",
                get(move |st: St, p: Id, h: HeaderMap| query::query_handle(q.clone(), st, p, h)),
            )
            .route(
                "/v2/h3_context_ir",
                post(move |st: St, h: HeaderMap| create::unsupported(u.clone(), st, h, "/v2/h3_context_ir")),
            )
            .route(
                "/v2/video_regeneration",
                post(move |st: St, h: HeaderMap| {
                    create::unsupported(u2.clone(), st, h, "/v2/video_regeneration")
                }),
            )
    }
}

impl BatchProtocol for MiniMax {
    fn id(&self) -> ProtocolId {
        ProtocolId::MiniMaxV2
    }

    /// 18 decimal digits, like MiniMax's numeric ids (strobe: the first 18
    /// digits of `uuid4().int`).
    fn new_external_id(&self, job: JobId) -> String {
        task_id_for(job)
    }

    fn render_error(&self, err: &ApiError, cx: &ErrorCtx) -> HttpReply {
        render_oai_error(err, cx.request_id.as_deref())
    }
}

/// The 18-digit task id of a job: the first 18 decimal digits of its uuid
/// (a v4 uuid is at least 2^76, so always 23+ digits; never a leading zero).
pub fn task_id_for(job: JobId) -> String {
    let s = job.0.as_u128().to_string();
    format!("{:0>18}", &s[..s.len().min(18)])
}

/// Whether `s` looks like one of our task ids (18 digits).
pub fn is_task_id(s: &str) -> bool {
    s.len() == 18 && s.bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_ids_are_18_digits() {
        for _ in 0..1000 {
            let id = task_id_for(JobId::new());
            assert!(is_task_id(&id), "{id}");
            assert_ne!(id.as_bytes()[0], b'0');
        }
        assert!(!is_task_id("12345"));
    }

    #[test]
    fn models_and_resolutions() {
        for m in MiniMaxModel::ALL {
            assert_eq!(MiniMaxModel::parse(m.as_str()), Some(m));
        }
        assert_eq!(MiniMaxModel::parse("MiniMax-Hailuo-02"), None);
        assert_eq!(MiniMaxModel::H3.min_duration(), 4);
        assert_eq!(MiniMaxModel::H3Max.min_duration(), 5);
        assert_eq!(Resolution::parse("2K").map(|r| r.short_edge()), Some(1440));
        assert_eq!(Resolution::parse("720P"), None);
    }
}
