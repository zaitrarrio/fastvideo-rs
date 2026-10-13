//! Prompt upsampling for fv-serve (docs/serve/prompt-upsampling.md).
//!
//! - [`chat`]: the `/chat/completions` seam ([`ChatBackend`]) and the hedge
//!   ([`chat::hedged`]: two concurrent attempts, first success wins, all
//!   aborted at the timeout);
//! - [`http`] (feature `http`): [`http::OpenAiChat`], any OpenAI-compatible
//!   server;
//! - [`director`]: the owner's showrunner upsampler, ported 1:1 from
//!   docs/serve/upsampler-spec.ts (world bible, entity registry, ledger);
//! - [`rewrite`]: one-shot rewrites of batch prompts with a system prompt
//!   per model family.
//!
//! [`Service`] is what fv-serve holds when `[upsampler]` is configured: the
//! backend plus the operator's settings, with metrics and log lines.

pub mod chat;
pub mod director;
#[cfg(feature = "http")]
pub mod http;
pub mod rewrite;

use std::sync::Arc;
use std::time::{Duration, Instant};

pub use chat::{ChatBackend, ChatRequest, UpsampleError};
pub use director::{Direction, Entity, EntityKind, ExpandedSegment, Upsampler, UpsamplerConfig, WorldBible};
pub use rewrite::{Effort, PromptFamily, RewriteSpec};

/// Per-family prompt budgets in characters, from each text encoder's token
/// limit at roughly 3.5-4 characters per token of English, with headroom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxChars {
    /// umT5-XXL, 512 tokens (Wan, FastWan, SF-Wan, LongLive): 1800.
    pub wan: usize,
    /// Gemma 3, `max_sequence_length` 1024 (LTX-2.x): 3600.
    pub ltx: usize,
    /// The H3 text tower (Qwen3-VL), 1024 tokens, refused beyond (no
    /// truncation): 3200.
    pub h3: usize,
    /// Any other family: 1800.
    pub other: usize,
}

impl Default for MaxChars {
    fn default() -> Self {
        Self { wan: 1800, ltx: 3600, h3: 3200, other: 1800 }
    }
}

impl MaxChars {
    pub fn for_family(&self, f: PromptFamily) -> usize {
        match f {
            PromptFamily::Wan => self.wan,
            PromptFamily::Ltx => self.ltx,
            PromptFamily::H3 => self.h3,
            PromptFamily::Generic => self.other,
        }
    }
}

/// The operator's `[upsampler]` settings (the transport lives in the
/// backend).
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Shown in logs and `session_info` (never the key).
    pub model: String,
    /// One call, every hedged attempt included.
    pub timeout: Duration,
    /// Two concurrent attempts per call, first success wins.
    pub hedge: bool,
    /// Director: segments kept as context when `configure.memory` is absent.
    pub memory: usize,
    pub max_chars: MaxChars,
    /// Clients that omit the flag get their protocol's documented default
    /// (fal `balanced`, director on); otherwise upsampling is opt-in only.
    pub default_on: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: String::new(),
            timeout: Duration::from_secs(10),
            hedge: true,
            memory: 12,
            max_chars: MaxChars::default(),
            default_on: false,
        }
    }
}

/// What a metric line is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A batch rewrite.
    Rewrite,
    /// A director segment.
    Director,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Rewrite => "rewrite",
            Kind::Director => "director",
        }
    }
}

/// Counts one upsampling call: `fv_upsampler_calls_total{kind,outcome}` and
/// `fv_upsampler_latency_seconds{kind}` (outcome `ok` or the error label;
/// a fallback to the raw prompt is every non-`ok` outcome).
pub fn record(kind: Kind, outcome: Result<(), &UpsampleError>, latency: Duration) {
    let label = match outcome {
        Ok(()) => "ok",
        Err(e) => e.label(),
    };
    metrics::counter!("fv_upsampler_calls_total", "kind" => kind.as_str(), "outcome" => label).increment(1);
    metrics::histogram!("fv_upsampler_latency_seconds", "kind" => kind.as_str()).record(latency.as_secs_f64());
}

/// The configured upsampler: backend + settings.
#[derive(Clone)]
pub struct Service {
    backend: Arc<dyn ChatBackend>,
    settings: Settings,
}

impl std::fmt::Debug for Service {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Service").field("backend", &self.backend.describe()).field("settings", &self.settings).finish()
    }
}

impl Service {
    pub fn new(backend: Arc<dyn ChatBackend>, settings: Settings) -> Self {
        Self { backend, settings }
    }

    /// Over HTTP: `base_url` + `/chat/completions`, `model` from the
    /// settings, the key as a bearer token when set.
    #[cfg(feature = "http")]
    pub fn http(base_url: &str, api_key: Option<String>, settings: Settings) -> Result<Self, UpsampleError> {
        let chat = http::OpenAiChat::new(base_url, api_key, &settings.model)?;
        Ok(Self::new(Arc::new(chat), settings))
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn backend(&self) -> &Arc<dyn ChatBackend> {
        &self.backend
    }

    pub fn describe(&self) -> String {
        self.backend.describe()
    }

    /// One batch rewrite (logged and counted). The caller falls back to
    /// the original prompt on `Err`.
    pub async fn rewrite(&self, family: PromptFamily, effort: Effort, image: bool, prompt: &str) -> Result<String, UpsampleError> {
        let spec = RewriteSpec { family, effort, image, max_chars: self.settings.max_chars.for_family(family) };
        let t0 = Instant::now();
        let r = rewrite::rewrite(&self.backend, self.settings.hedge, self.settings.timeout, &spec, prompt).await;
        let dt = t0.elapsed();
        record(Kind::Rewrite, r.as_ref().map(|_| ()), dt);
        match &r {
            Ok(p) => tracing::info!(
                family = family.as_str(),
                effort = effort.as_str(),
                latency_ms = dt.as_millis() as u64,
                chars_in = prompt.chars().count(),
                chars_out = p.chars().count(),
                "upsampler: prompt rewritten"
            ),
            Err(e) => tracing::warn!(
                family = family.as_str(),
                effort = effort.as_str(),
                latency_ms = dt.as_millis() as u64,
                error = %e,
                "upsampler: rewrite failed, the original prompt is used"
            ),
        }
        r
    }

    /// A director session's upsampler for `world` (`memory`: the
    /// session's `configure.memory`, else the configured default).
    pub fn director(&self, world: WorldBible, family: PromptFamily, memory: Option<usize>) -> Upsampler {
        let cfg = UpsamplerConfig {
            memory: memory.unwrap_or(self.settings.memory).min(50),
            max_prompt_chars: self.settings.max_chars.for_family(family),
            hedge: self.settings.hedge,
            timeout: self.settings.timeout,
        };
        Upsampler::new(cfg, world, self.backend.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    #[async_trait::async_trait]
    impl ChatBackend for Echo {
        async fn complete(&self, req: &ChatRequest) -> Result<String, UpsampleError> {
            Ok(format!("Rewritten: {}", req.user))
        }
    }

    #[tokio::test]
    async fn service_rewrites_within_the_family_budget() {
        let s = Service::new(Arc::new(Echo), Settings { max_chars: MaxChars { wan: 12, ..MaxChars::default() }, ..Settings::default() });
        assert_eq!(s.rewrite(PromptFamily::H3, Effort::Balanced, false, "a cat").await.unwrap(), "Rewritten: a cat");
        assert_eq!(s.rewrite(PromptFamily::Wan, Effort::Quality, false, "a cat on a mat").await.unwrap(), "Rewritten:");
        let u = s.director(WorldBible::default(), PromptFamily::Ltx, Some(80));
        assert_eq!((u.config().memory, u.config().max_prompt_chars), (50, 3600));
        assert_eq!(s.director(WorldBible::default(), PromptFamily::Wan, None).config().memory, 12);
    }
}
