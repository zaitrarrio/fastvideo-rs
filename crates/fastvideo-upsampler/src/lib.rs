//! Prompt upsampling for fv-serve (docs/serve/prompt-upsampling.md).
//!
//! - [`chat`]: the `/chat/completions` seam ([`ChatBackend`]) and the hedge
//!   ([`chat::hedged`]: two concurrent attempts, first success wins, all
//!   aborted at the timeout);
//! - [`http`] (feature `http`): [`http::OpenAiChat`], any OpenAI-compatible
//!   server;
//! - [`director`]: the owner's showrunner upsampler, ported from
//!   docs/serve/upsampler-spec.ts (world bible, entity registry, ledger),
//!   modelled on fal's H3 Max director (long premises, token budget,
//!   premise digest, planning ahead);
//! - [`assemble`]: the per-family segment prompt (H3's Context-IR layout,
//!   the owner's prose) within the encoder's limit;
//! - [`tokens`]: token counting (the DiT encoders' tokenizers, the LLM
//!   context estimate);
//! - [`rewrite`]: one-shot rewrites of batch prompts with a system prompt
//!   per model family.
//!
//! [`Service`] is what fv-serve holds when `[upsampler]` is configured: the
//! backend plus the operator's settings, with metrics and log lines.

pub mod assemble;
pub mod chat;
pub mod director;
#[cfg(feature = "http")]
pub mod http;
pub mod rewrite;
pub mod tokens;

use std::sync::Arc;
use std::time::{Duration, Instant};

pub use chat::{ChatBackend, ChatRequest, UpsampleError};
pub use assemble::Layout;
pub use director::{Direction, Entity, EntityKind, ExpandedSegment, Frames, PremiseReport, Upsampler, UpsamplerConfig, WorldBible};
pub use rewrite::{Effort, PromptFamily, RewriteSpec};
pub use tokens::{PromptLimit, TokenCounter, Unit};

/// Per-family text-encoder limits in tokens (used when the family's
/// tokenizer is available; else [`MaxChars`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxTokens {
    /// umT5-XXL `text_len` 512 (Wan, FastWan, SF-Wan, LongLive).
    pub wan: usize,
    /// Gemma 3 `max_sequence_length` 1024 (LTX-2.x).
    pub ltx: usize,
    /// `H3_MAX_PROMPT_TOKENS` 1024, refused beyond (no truncation).
    pub h3: usize,
    pub other: usize,
}

impl Default for MaxTokens {
    fn default() -> Self {
        Self { wan: 512, ltx: 1024, h3: 1024, other: 512 }
    }
}

impl MaxTokens {
    pub fn for_family(&self, f: PromptFamily) -> usize {
        match f {
            PromptFamily::Wan => self.wan,
            PromptFamily::Ltx => self.ltx,
            PromptFamily::H3 => self.h3,
            PromptFamily::Generic => self.other,
        }
    }
}

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
    pub max_tokens: MaxTokens,
    /// Clients that omit the flag get their protocol's documented default
    /// (fal `balanced`, director on); otherwise upsampling is opt-in only.
    pub default_on: bool,
    /// Director: the LLM's context in tokens (the input budget is this less
    /// the system prompt and the answer's reserve).
    pub context_tokens: usize,
    /// Director: segments planned ahead (fal's `prompt_deck_size`).
    pub deck_size: usize,
    /// The LLM sees images (`image_url` content parts): a vision model
    /// such as Qwen3-VL behind llama-server `--mmproj`.
    pub vision: bool,
    /// Ask the LLM server's `/tokenize` for exact counts (once; a server
    /// without it is remembered).
    pub tokenize: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: String::new(),
            timeout: Duration::from_secs(10),
            hedge: true,
            memory: 12,
            max_chars: MaxChars::default(),
            max_tokens: MaxTokens::default(),
            default_on: false,
            context_tokens: director::DEFAULT_CONTEXT_TOKENS,
            deck_size: 6,
            vision: false,
            tokenize: true,
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

/// The text encoders' tokenizers fv-serve found, per family.
#[derive(Clone, Default)]
pub struct Counters {
    pub wan: Option<Arc<dyn TokenCounter>>,
    pub ltx: Option<Arc<dyn TokenCounter>>,
    pub h3: Option<Arc<dyn TokenCounter>>,
}

impl Counters {
    pub fn for_family(&self, f: PromptFamily) -> Option<Arc<dyn TokenCounter>> {
        match f {
            PromptFamily::Wan => self.wan.clone(),
            PromptFamily::Ltx => self.ltx.clone(),
            PromptFamily::H3 => self.h3.clone(),
            PromptFamily::Generic => None,
        }
    }
}

/// The configured upsampler: backend + settings.
#[derive(Clone)]
pub struct Service {
    backend: Arc<dyn ChatBackend>,
    settings: Settings,
    counters: Counters,
}

impl std::fmt::Debug for Service {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = |c: &Option<Arc<dyn TokenCounter>>| c.as_ref().map(|c| c.describe());
        f.debug_struct("Service")
            .field("backend", &self.backend.describe())
            .field("settings", &self.settings)
            .field("tokenizers", &(c(&self.counters.wan), c(&self.counters.ltx), c(&self.counters.h3)))
            .finish()
    }
}

impl Service {
    pub fn new(backend: Arc<dyn ChatBackend>, settings: Settings) -> Self {
        Self { backend, settings, counters: Counters::default() }
    }

    /// With the text encoders' tokenizers (limits in tokens).
    pub fn with_counters(mut self, counters: Counters) -> Self {
        self.counters = counters;
        self
    }

    /// A family's DiT prompt limit: tokens by its tokenizer when fv-serve
    /// has it, else characters.
    pub fn limit(&self, family: PromptFamily) -> PromptLimit {
        PromptLimit {
            counter: self.counters.for_family(family),
            max_tokens: self.settings.max_tokens.for_family(family),
            max_chars: self.settings.max_chars.for_family(family),
        }
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
        let limit = self.limit(family);
        let r = rewrite::rewrite(&self.backend, self.settings.hedge, self.settings.timeout, &spec, &limit, prompt).await;
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
            limit: self.limit(family),
            layout: Layout::for_family(family),
            hedge: self.settings.hedge,
            timeout: self.settings.timeout,
            context_tokens: self.settings.context_tokens,
            tokenize: self.settings.tokenize,
            digest_timeout: self.settings.timeout.max(Duration::from_secs(120)),
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
        assert_eq!(s.rewrite(PromptFamily::H3, Effort::Balanced, false, "a cat").await.unwrap(), "integrated_multimodal_description: Rewritten: a cat");
        assert_eq!(s.rewrite(PromptFamily::Wan, Effort::Quality, false, "a cat on a mat").await.unwrap(), "Rewritten:");
        let u = s.director(WorldBible::default(), PromptFamily::Ltx, Some(80));
        assert_eq!((u.config().memory, u.config().limit.max_chars, u.config().limit.max_tokens), (50, 3600, 1024));
        assert_eq!(u.config().layout, Layout::Prose);
        assert_eq!(s.director(WorldBible::default(), PromptFamily::H3, None).config().layout, Layout::H3);
        assert_eq!(s.director(WorldBible::default(), PromptFamily::Wan, None).config().memory, 12);
    }
}
