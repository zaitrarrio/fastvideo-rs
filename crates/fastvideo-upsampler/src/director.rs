//! The director's showrunner upsampler: the owner's TypeScript `Upsampler`
//! (docs/serve/upsampler-spec.ts, the spec) ported to Rust, then modelled
//! on fal's H3 Max director (docs/serve/prompt-upsampling.md §Modelled on
//! fal's H3 Max director).
//!
//! Each segment is the NEXT segment of one unbroken take: the LLM gets the
//! world bible's premise, the entity registry, the most recent segments and
//! an optional new direction, and answers the strict `segment` JSON schema
//! ([`output_schema`]); [`crate::assemble`] turns that into the DiT prompt
//! for the model family (H3's Context-IR layout, or the owner's prose). New
//! entities are accepted only under a new id, and `entity_ids` are filtered
//! to the registry.
//!
//! Kept from the spec: the types, `SYSTEM_PROMPT` (verbatim, one rule added
//! for `music`), the schema (plus `music`), the user message's four
//! sections and their order, the registry and the ledger, the hedge.
//!
//! Added for long premises and fal's planning (each listed in the docs):
//!
//! - **A token budget for the LLM input** ([`UpsamplerConfig::context_tokens`]):
//!   the premise first and verbatim (a stable prefix the server's prompt
//!   cache reuses), then the registry, then as many of the last `memory`
//!   segments as fit (the oldest dropped first), then the direction.
//!   Tokens are estimated at [`crate::tokens::CHARS_PER_TOKEN`] characters
//!   each, calibrated once per session by the LLM server's `/tokenize`
//!   when it has one ([`Upsampler::prepare`]).
//! - **A premise digest:** a premise that alone overflows the budget is
//!   condensed once, at configure time, into a "series bible"
//!   ([`Upsampler::prepare`]); the digest replaces it in every message.
//! - **Frames:** a given first frame or a final frame the segment must end
//!   on is one line under the direction ([`Frames`]); a vision model also
//!   gets the images (the caller attaches them to the request).
//! - **A speculative chain:** [`Upsampler::user_message_after`] writes the
//!   message as if planned-but-unexecuted segments had been committed, so
//!   the director can plan several segments ahead; only executed segments
//!   are [`Upsampler::commit`]ted.
//! - **Per-family assembly** within the encoder's limit in tokens
//!   ([`crate::assemble`]).
//!
//! Where Rust needs a choice the TS does not make: lengths are Unicode
//! scalar values; parsing is strict (serde requires every schema field but
//! `music`, which defaults to empty); an unknown entity `kind` parses as
//! [`EntityKind::Other`]; `next` is split into `user_message` →
//! [`expand_with_hedge`] → `commit`; `memory = 0` keeps the whole ledger
//! (`slice(-0)`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::assemble::{assemble, Layout};
use crate::chat::{hedged, ChatBackend, ChatRequest, UpsampleError};
use crate::tokens::{cut_to_fit, estimate_tokens_at, PromptLimit, CHARS_PER_TOKEN};

/// `EntityKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityKind {
    Character,
    Location,
    Prop,
    /// Any other string the model returns (the TS keeps it as is; only
    /// `location` changes how a cast member is placed).
    #[serde(other)]
    Other,
}

impl EntityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EntityKind::Character => "character",
            EntityKind::Location => "location",
            EntityKind::Prop => "prop",
            EntityKind::Other => "other",
        }
    }
}

/// `Entity`: `descriptor` is the canonical appearance text, spliced
/// verbatim into every segment the entity appears in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub kind: EntityKind,
    pub name: String,
    pub descriptor: String,
}

/// `WorldBible`: the premise (the director's `configure.prompt`, up to
/// 50 000 characters), the style appended to every segment and the opening
/// cast.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldBible {
    pub premise: String,
    pub style: String,
    pub entities: Vec<Entity>,
}

/// `Direction`: a new client direction (the director's `prompt` message;
/// `id` is its `prompt_version`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Direction {
    pub id: String,
    pub text: String,
}

/// Frames the segment is pinned to (ours; fal: "The opening prompt
/// expansion also sees this image", "Director jointly plans that
/// arrival").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Frames {
    /// The segment starts from a given image (`configure.image_url`).
    pub first: bool,
    /// The segment must end on a given image (`end_image_url`, an
    /// end-image script beat).
    pub end: bool,
    /// The images are attached to the request (a vision model).
    pub attached: bool,
}

/// `SegmentRecord`: one ledger entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRecord {
    pub index: usize,
    pub direction_id: Option<String>,
    pub summary: String,
    pub prompt: String,
}

/// `ExpandedSegment`: the record plus the entities this segment introduced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpandedSegment {
    #[serde(flatten)]
    pub record: SegmentRecord,
    pub new_entities: Vec<Entity>,
}

/// `DialogueLine`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialogueLine {
    pub speaker_id: String,
    pub line: String,
}

/// `ExpansionOutput`: the `segment` schema's object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpansionOutput {
    pub beat_summary: String,
    pub entity_ids: Vec<String>,
    pub new_entities: Vec<Entity>,
    pub action: String,
    pub camera: String,
    pub lighting: String,
    pub audio: String,
    /// The non-diegetic score ("" for none). Ours: in the schema's
    /// `required` (strict), but an answer without it parses as "".
    #[serde(default)]
    pub music: String,
    pub dialogue: Vec<DialogueLine>,
}

/// `UpsamplerConfig` without the transport (`baseUrl`, `apiKey`, `model`
/// live in the [`ChatBackend`]).
#[derive(Clone, Debug, PartialEq)]
pub struct UpsamplerConfig {
    /// Prior segments kept as context (Director default 12, max 50).
    pub memory: usize,
    /// The DiT prompt limit (the spec's `maxPromptChars`, in tokens when
    /// the family's tokenizer is available).
    pub limit: PromptLimit,
    /// The prompt layout of the model family.
    pub layout: Layout,
    pub hedge: bool,
    pub timeout: Duration,
    /// The LLM's context, in tokens: the system prompt, the user message
    /// and the answer must fit (default 28 000, a local Qwen3-4B at 32k).
    pub context_tokens: usize,
    /// Ask the LLM server's `/tokenize` for the premise's exact count.
    pub tokenize: bool,
    /// The premise digest's calls, all of them.
    pub digest_timeout: Duration,
}

impl Default for UpsamplerConfig {
    fn default() -> Self {
        Self {
            memory: 12,
            limit: PromptLimit::chars(1800),
            layout: Layout::Prose,
            hedge: true,
            timeout: Duration::from_secs(10),
            context_tokens: DEFAULT_CONTEXT_TOKENS,
            tokenize: true,
            digest_timeout: Duration::from_secs(120),
        }
    }
}

/// `[upsampler] context_tokens` default.
pub const DEFAULT_CONTEXT_TOKENS: usize = 28_000;
/// Tokens kept free for the answer (a segment's JSON is a few hundred).
pub const OUTPUT_RESERVE_TOKENS: usize = 1_536;
/// Tokens kept free next to the premise for the registry, the direction
/// and at least one recent segment, at the least (or 1/8 of the input
/// budget, whichever is larger).
pub const REST_RESERVE_TOKENS: usize = 2_048;

/// `SYSTEM_PROMPT`, verbatim.
pub const SYSTEM_PROMPT: &str = "You are the showrunner for a continuous, never-ending live video stream.
Each turn you write the NEXT segment of one unbroken take. You receive the world bible,
the registry of established entities, the recent segment history, and an optional new direction.

Rules:
- Continue directly from the last segment. No recaps, no time skips unless the direction demands one.
- If a direction is given, make it happen in this segment, staying consistent with established people and places.
- If no direction is given, advance the story by one natural beat.
- Refer to existing people, places, and props ONLY by their entity ids. Never describe their appearance.
- Introduce a new entity only if the story needs it; give it a short id and a concrete visual descriptor.
- \"action\" is present-tense, physical and visible: what moves, where, how. One continuous shot.
- \"audio\" describes ambience, foley and score. Spoken words go only in \"dialogue\", verbatim.
- Write concrete visual language. No meta commentary, no references to cameras \"filming a show\".";

/// The rule added for the `music` field (ours).
pub const MUSIC_RULE: &str = "- \"music\" is the non-diegetic score only, or an empty string for none; then \"audio\" keeps ambience and foley.";

/// The system prompt sent: [`SYSTEM_PROMPT`] plus [`MUSIC_RULE`].
pub fn system_prompt() -> String {
    format!("{SYSTEM_PROMPT}\n{MUSIC_RULE}")
}

/// The spec's sampling temperature.
pub const TEMPERATURE: f32 = 0.8;

/// `OUTPUT_SCHEMA` (the `json_schema` object of `response_format`), plus
/// `music` (required, as strict mode wants every property listed; an empty
/// string is a valid answer).
pub fn output_schema() -> Value {
    json!({
        "name": "segment",
        "strict": true,
        "schema": {
            "type": "object",
            "additionalProperties": false,
            "required": [
                "beat_summary",
                "entity_ids",
                "new_entities",
                "action",
                "camera",
                "lighting",
                "audio",
                "music",
                "dialogue",
            ],
            "properties": {
                "beat_summary": {"type": "string"},
                "entity_ids": {"type": "array", "items": {"type": "string"}},
                "new_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["id", "kind", "name", "descriptor"],
                        "properties": {
                            "id": {"type": "string"},
                            "kind": {"type": "string", "enum": ["character", "location", "prop"]},
                            "name": {"type": "string"},
                            "descriptor": {"type": "string"},
                        },
                    },
                },
                "action": {"type": "string"},
                "camera": {"type": "string"},
                "lighting": {"type": "string"},
                "audio": {"type": "string"},
                "music": {"type": "string"},
                "dialogue": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["speaker_id", "line"],
                        "properties": {
                            "speaker_id": {"type": "string"},
                            "line": {"type": "string"},
                        },
                    },
                },
            },
        },
    })
}

/// The entity registry: a JS `Map` (insertion order kept; `set` on an
/// existing id replaces the value in place).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Registry {
    entries: Vec<Entity>,
}

impl Registry {
    pub fn get(&self, id: &str) -> Option<&Entity> {
        self.entries.iter().find(|e| e.id == id)
    }
    pub fn has(&self, id: &str) -> bool {
        self.get(id).is_some()
    }
    pub fn set(&mut self, e: Entity) {
        match self.entries.iter_mut().find(|x| x.id == e.id) {
            Some(slot) => *slot = e,
            None => self.entries.push(e),
        }
    }
    pub fn values(&self) -> &[Entity] {
        &self.entries
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn record_text(s: &SegmentRecord) -> String {
    format!("[{}] {}\n{}", s.index, s.summary, s.prompt)
}

fn frame_lines(frames: Frames) -> String {
    let seen = if frames.attached {
        "attached"
    } else {
        "not shown to you"
    };
    let mut v = Vec::new();
    if frames.first {
        v.push(format!(
            "FIRST FRAME: this segment starts from a given image ({seen}); continue from it."
        ));
    }
    if frames.end {
        v.push(format!(
            "FINAL FRAME: this segment must end exactly on a given image ({seen}); move everything so that the shot arrives there."
        ));
    }
    v.join("\n")
}

/// `buildUserMessage` (the premise as given: the digest when there is one).
pub fn build_user_message(
    world: &WorldBible,
    registry: &Registry,
    history: &[SegmentRecord],
    direction: Option<&Direction>,
) -> String {
    build_message(
        &world.premise,
        registry,
        history,
        direction,
        Frames::default(),
    )
}

/// The four sections of the spec, with the frame lines (ours) after the
/// direction.
pub fn build_message(
    premise: &str,
    registry: &Registry,
    history: &[SegmentRecord],
    direction: Option<&Direction>,
    frames: Frames,
) -> String {
    let entities = registry
        .values()
        .iter()
        .map(|e| format!("- {} ({}, \"{}\")", e.id, e.kind.as_str(), e.name))
        .collect::<Vec<_>>()
        .join("\n");
    let recent = history
        .iter()
        .map(record_text)
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut dir = match direction.filter(|d| !d.text.trim().is_empty()) {
        Some(d) => format!("NEW DIRECTION:\n{}", d.text),
        None => "NEW DIRECTION:\n(none, advance the story one beat)".to_owned(),
    };
    let fl = frame_lines(frames);
    if !fl.is_empty() {
        dir = format!("{dir}\n{fl}");
    }
    [
        format!("WORLD BIBLE:\n{premise}"),
        format!(
            "ESTABLISHED ENTITIES:\n{}",
            if entities.is_empty() {
                "(none)"
            } else {
                &entities
            }
        ),
        format!(
            "RECENT SEGMENTS (oldest first):\n{}",
            if recent.is_empty() {
                "(stream is starting)"
            } else {
                &recent
            }
        ),
        dir,
    ]
    .join("\n\n")
}

/// The chat request of `callModel` for `user` (`images`: for a vision
/// model, else empty).
pub fn request_for(user: String, images: Vec<String>) -> ChatRequest {
    ChatRequest {
        system: system_prompt(),
        user,
        temperature: TEMPERATURE,
        response_format: Some(json!({"type": "json_schema", "json_schema": output_schema()})),
        images,
    }
}

/// `callModel`: one completion, parsed as [`ExpansionOutput`].
pub async fn call_model(
    backend: &dyn ChatBackend,
    req: &ChatRequest,
) -> Result<ExpansionOutput, UpsampleError> {
    let content = backend.complete(req).await?;
    // Tolerant of small local models: a leading `<think>` block or a code
    // fence around the JSON is dropped first (an addition to the spec).
    serde_json::from_str(crate::chat::json_body(&content))
        .map_err(|e| UpsampleError::BadOutput(e.to_string()))
}

/// `expandWithHedge`: two concurrent calls with the hedge (one without),
/// the first success wins, everything is aborted at the timeout.
pub async fn expand_with_hedge(
    backend: &Arc<dyn ChatBackend>,
    hedge: bool,
    timeout: Duration,
    user: String,
) -> Result<ExpansionOutput, UpsampleError> {
    expand_request(backend, hedge, timeout, request_for(user, Vec::new())).await
}

/// [`expand_with_hedge`] for a prepared request (with images).
pub async fn expand_request(
    backend: &Arc<dyn ChatBackend>,
    hedge: bool,
    timeout: Duration,
    req: ChatRequest,
) -> Result<ExpansionOutput, UpsampleError> {
    let attempts = if hedge { 2 } else { 1 };
    hedged(attempts, timeout, || call_model(backend.as_ref(), &req)).await
}

/// `assemblePrompt` in the owner's prose layout, within `max_chars`
/// characters (trimmed by priority, see [`crate::assemble`]).
pub fn assemble_prompt(
    out: &ExpansionOutput,
    registry: &Registry,
    world: &WorldBible,
    max_chars: usize,
) -> String {
    assemble(
        Layout::Prose,
        out,
        registry,
        world,
        &PromptLimit::chars(max_chars),
    )
}

/// The premise digest's system prompt (ours; fal's expander and its system
/// prompt are undocumented).
pub const DIGEST_SYSTEM_PROMPT: &str = "You condense a long series premise into a series bible for a showrunner who writes one short video segment at a time.
Keep: the world and its rules; every named character, place and recurring prop with its visual description copied verbatim; relationships; the story so far and the planned arcs in order; tone and visual style.
Drop: repetition, flourishes and meta commentary.
Plain text in short labelled paragraphs, no markdown. Reply with the series bible only.";

/// What [`Upsampler::prepare`] found and did about the premise (reported
/// in `configured` and the logs).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PremiseReport {
    pub chars: usize,
    /// Tokens by the LLM server's `/tokenize` (`tokens_exact`), else
    /// estimated.
    pub tokens: usize,
    pub tokens_exact: bool,
    /// Characters per token used for every estimate of this session.
    pub chars_per_token: f64,
    pub context_tokens: usize,
    /// The largest premise sent verbatim.
    pub premise_budget_tokens: usize,
    /// `None`: the premise is sent verbatim.
    pub digest: Option<DigestReport>,
}

/// The premise digest.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct DigestReport {
    pub chars: usize,
    pub tokens: usize,
    /// LLM calls (more than one when the premise does not fit one call).
    pub calls: usize,
    pub latency_ms: u64,
    /// The calls failed: the premise's head, cut to the budget, is used.
    pub fallback: bool,
    pub error: Option<String>,
}

/// A read-only copy of the upsampler's state (`snapshot()`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub registry: Vec<Entity>,
    pub ledger: Vec<SegmentRecord>,
}

/// A planned (expanded, not yet executed) segment of the speculative chain.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedOutput {
    pub direction: Option<Direction>,
    pub out: ExpansionOutput,
}

/// `class Upsampler`.
pub struct Upsampler {
    cfg: UpsamplerConfig,
    world: WorldBible,
    backend: Arc<dyn ChatBackend>,
    registry: Registry,
    ledger: Vec<SegmentRecord>,
    /// The series-bible digest that replaces a too-long premise.
    digest: Option<String>,
    chars_per_token: f64,
    report: PremiseReport,
}

impl std::fmt::Debug for Upsampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upsampler")
            .field("cfg", &self.cfg)
            .field("entities", &self.registry.len())
            .field("segments", &self.ledger.len())
            .field("digest", &self.digest.is_some())
            .finish_non_exhaustive()
    }
}

/// Splits `s` into pieces of at most `max` characters at paragraph, else
/// sentence, else word boundaries.
fn split_text(s: &str, max: usize) -> Vec<&str> {
    let max = max.max(64);
    let mut out = Vec::new();
    let mut rest = s.trim();
    while rest.chars().count() > max {
        let end = rest.char_indices().nth(max).map_or(rest.len(), |(i, _)| i);
        let head = &rest[..end];
        let half = head.len() / 2;
        let cut = [
            head.rfind("\n\n"),
            head.rfind(". ").map(|i| i + 1),
            head.rfind(char::is_whitespace),
        ]
        .into_iter()
        .flatten()
        .find(|&i| i >= half)
        .unwrap_or(end);
        out.push(rest[..cut].trim());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

impl Upsampler {
    pub fn new(cfg: UpsamplerConfig, world: WorldBible, backend: Arc<dyn ChatBackend>) -> Self {
        let mut registry = Registry::default();
        for e in &world.entities {
            registry.set(e.clone());
        }
        let mut u = Self {
            cfg,
            world,
            backend,
            registry,
            ledger: Vec::new(),
            digest: None,
            chars_per_token: CHARS_PER_TOKEN,
            report: PremiseReport::default(),
        };
        u.report = u.measure(None);
        u
    }

    pub fn config(&self) -> &UpsamplerConfig {
        &self.cfg
    }

    pub fn world(&self) -> &WorldBible {
        &self.world
    }

    pub fn backend(&self) -> &Arc<dyn ChatBackend> {
        &self.backend
    }

    /// Segments recorded so far.
    pub fn ledger_len(&self) -> usize {
        self.ledger.len()
    }

    /// The last executed segment's prompt.
    pub fn last_prompt(&self) -> Option<&str> {
        self.ledger.last().map(|r| r.prompt.as_str())
    }

    /// The premise as the LLM sees it: the digest when there is one.
    pub fn premise_text(&self) -> &str {
        self.digest.as_deref().unwrap_or(&self.world.premise)
    }

    pub fn premise_report(&self) -> &PremiseReport {
        &self.report
    }

    /// Estimated LLM tokens of `s` at this session's calibration.
    pub fn estimate(&self, s: &str) -> usize {
        estimate_tokens_at(s, self.chars_per_token)
    }

    /// Tokens the user message may take: the context less the system
    /// prompt and the answer's reserve.
    pub fn input_budget(&self) -> usize {
        self.cfg
            .context_tokens
            .saturating_sub(self.estimate(&system_prompt()) + OUTPUT_RESERVE_TOKENS)
            .max(256)
    }

    /// The largest premise sent verbatim: the input budget less room for
    /// the registry, the direction and at least one segment.
    pub fn premise_budget(&self) -> usize {
        let b = self.input_budget();
        b.saturating_sub(REST_RESERVE_TOKENS.max(b / 8)).max(128)
    }

    fn measure(&self, exact: Option<usize>) -> PremiseReport {
        PremiseReport {
            chars: self.world.premise.chars().count(),
            tokens: exact.unwrap_or_else(|| self.estimate(&self.world.premise)),
            tokens_exact: exact.is_some(),
            chars_per_token: (self.chars_per_token * 1000.0).round() / 1000.0,
            context_tokens: self.cfg.context_tokens,
            premise_budget_tokens: self.premise_budget(),
            digest: None,
        }
    }

    /// Configure time: calibrates the token estimate with the LLM server's
    /// `/tokenize` (when enabled and available), and condenses a premise
    /// that alone overflows [`Self::premise_budget`] into a series bible,
    /// once, cached for the session. Never fails: a failed digest falls
    /// back to the premise's head within the budget.
    pub async fn prepare(&mut self) -> &PremiseReport {
        let premise = self.world.premise.clone();
        let exact = if self.cfg.tokenize && !premise.trim().is_empty() {
            self.backend.tokenize(&premise).await
        } else {
            None
        };
        if let Some(n) = exact.filter(|&n| n > 0) {
            let ratio = premise.chars().count() as f64 / n as f64;
            self.chars_per_token = ratio.clamp(1.5, 6.0);
        }
        let mut report = self.measure(exact);
        if report.tokens > report.premise_budget_tokens {
            let t0 = Instant::now();
            let (digest, calls, error) =
                match tokio::time::timeout(self.cfg.digest_timeout, self.digest_premise()).await {
                    Ok(Ok((d, calls))) => (Some(d), calls, None),
                    Ok(Err((e, calls))) => (None, calls, Some(e.to_string())),
                    Err(_) => (
                        None,
                        0,
                        Some(
                            UpsampleError::Timeout(self.cfg.digest_timeout.as_millis() as u64)
                                .to_string(),
                        ),
                    ),
                };
            let budget = report.premise_budget_tokens;
            let fits = |s: &str| self.estimate(s) <= budget;
            let fallback = digest.is_none();
            let text = match digest {
                Some(d) => cut_to_fit(&d, &fits),
                None => cut_to_fit(&premise, &fits),
            };
            report.digest = Some(DigestReport {
                chars: text.chars().count(),
                tokens: self.estimate(&text),
                calls,
                latency_ms: t0.elapsed().as_millis() as u64,
                fallback,
                error,
            });
            self.digest = Some(text);
        }
        self.report = report;
        &self.report
    }

    /// The digest calls: one when the premise fits one call, else one per
    /// piece and (when the joined pieces are still over the budget) one
    /// more over them.
    async fn digest_premise(&self) -> Result<(String, usize), (UpsampleError, usize)> {
        let budget = self.premise_budget();
        // Each call: the piece plus the instruction, and the answer.
        let piece_tokens = self
            .cfg
            .context_tokens
            .saturating_sub(
                self.estimate(DIGEST_SYSTEM_PROMPT) + budget.min(self.cfg.context_tokens / 3) + 256,
            )
            .max(512);
        let piece_chars = (piece_tokens as f64 * self.chars_per_token) as usize;
        let pieces = split_text(&self.world.premise, piece_chars);
        let n = pieces.len().max(1);
        let mut calls = 0;
        let mut parts = Vec::new();
        for p in &pieces {
            let words = (budget * 3 / 4 / n).max(100);
            calls += 1;
            parts.push(self.digest_call(p, words).await.map_err(|e| (e, calls))?);
        }
        let mut joined = parts.join("\n\n");
        if n > 1 && self.estimate(&joined) > budget {
            calls += 1;
            joined = self
                .digest_call(&joined, budget * 3 / 4)
                .await
                .map_err(|e| (e, calls))?;
        }
        Ok((joined, calls))
    }

    async fn digest_call(&self, text: &str, words: usize) -> Result<String, UpsampleError> {
        let req = ChatRequest {
            system: DIGEST_SYSTEM_PROMPT.to_owned(),
            user: format!("Condense this series premise to at most {words} words:\n\n{text}"),
            temperature: 0.3,
            response_format: None,
            images: Vec::new(),
        };
        let attempts = if self.cfg.hedge { 2 } else { 1 };
        hedged(attempts, self.cfg.digest_timeout, || async {
            let a = self.backend.complete(&req).await?;
            let t = crate::chat::strip_think(&a).trim().to_owned();
            if t.is_empty() {
                Err(UpsampleError::NoContent)
            } else {
                Ok(t)
            }
        })
        .await
    }

    /// The message within the input budget: premise, registry, then the
    /// newest of `history` (at most `memory`) that fit, then the direction.
    fn budgeted(
        &self,
        registry: &Registry,
        history: &[SegmentRecord],
        direction: Option<&Direction>,
        frames: Frames,
    ) -> String {
        let m = self.cfg.memory;
        let window = if m == 0 {
            history
        } else {
            &history[history.len().saturating_sub(m)..]
        };
        let fixed = self.estimate(&build_message(
            self.premise_text(),
            registry,
            &[],
            direction,
            frames,
        ));
        let budget = self.input_budget();
        let mut used = fixed;
        let mut start = window.len();
        while start > 0 {
            let t = self.estimate(&record_text(&window[start - 1])) + 1;
            if used + t > budget {
                break;
            }
            used += t;
            start -= 1;
        }
        build_message(
            self.premise_text(),
            registry,
            &window[start..],
            direction,
            frames,
        )
    }

    /// The prompt of `next(direction)` on the current state.
    pub fn user_message(&self, direction: Option<&Direction>) -> String {
        self.budgeted(&self.registry, &self.ledger, direction, Frames::default())
    }

    /// The message for the segment after the `planned` ones (expanded but
    /// not executed), as if they had been committed.
    pub fn user_message_after(
        &self,
        planned: &[PlannedOutput],
        direction: Option<&Direction>,
        frames: Frames,
    ) -> String {
        if planned.is_empty() {
            return self.budgeted(&self.registry, &self.ledger, direction, frames);
        }
        let mut registry = self.registry.clone();
        let keep = self.cfg.memory.max(50);
        let mut history: Vec<SegmentRecord> =
            self.ledger[self.ledger.len().saturating_sub(keep)..].to_vec();
        for (k, p) in planned.iter().enumerate() {
            let (rec, _) = commit_into(
                &mut registry,
                self.ledger.len() + k,
                p.direction.as_ref(),
                p.out.clone(),
                &self.world,
                &self.cfg,
            );
            history.push(rec);
        }
        self.budgeted(&registry, &history, direction, frames)
    }

    /// The second half of `next`: accept the new entities (new ids only),
    /// keep only registered `entity_ids`, record the segment.
    pub fn commit(
        &mut self,
        direction: Option<&Direction>,
        out: ExpansionOutput,
    ) -> ExpandedSegment {
        let (record, accepted) = commit_into(
            &mut self.registry,
            self.ledger.len(),
            direction,
            out,
            &self.world,
            &self.cfg,
        );
        self.ledger.push(record.clone());
        ExpandedSegment {
            record,
            new_entities: accepted,
        }
    }

    /// The prompt of a segment whose expansion failed: the direction's
    /// text, else the last executed segment's prompt, else the premise (its
    /// digest), within the limit. Never the whole premise when it is over.
    pub fn fallback_prompt(&self, direction: Option<&str>) -> String {
        let text = direction
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .or_else(|| self.last_prompt())
            .unwrap_or_else(|| self.premise_text());
        let limit = &self.cfg.limit;
        cut_to_fit(text, &|s: &str| limit.fits(s))
    }

    /// `next(direction)`.
    pub async fn next(
        &mut self,
        direction: Option<Direction>,
    ) -> Result<ExpandedSegment, UpsampleError> {
        let user = self.user_message(direction.as_ref());
        let out = expand_with_hedge(&self.backend, self.cfg.hedge, self.cfg.timeout, user).await?;
        Ok(self.commit(direction.as_ref(), out))
    }

    /// `snapshot()`.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            registry: self.registry.values().to_vec(),
            ledger: self.ledger.clone(),
        }
    }
}

fn commit_into(
    registry: &mut Registry,
    index: usize,
    direction: Option<&Direction>,
    mut out: ExpansionOutput,
    world: &WorldBible,
    cfg: &UpsamplerConfig,
) -> (SegmentRecord, Vec<Entity>) {
    let accepted: Vec<Entity> = out
        .new_entities
        .iter()
        .filter(|e| !registry.has(&e.id))
        .cloned()
        .collect();
    for e in &accepted {
        registry.set(e.clone());
    }
    out.entity_ids.retain(|id| registry.has(id));
    let record = SegmentRecord {
        index,
        direction_id: direction
            .filter(|d| !d.text.trim().is_empty())
            .map(|d| d.id.clone()),
        summary: out.beat_summary.clone(),
        prompt: assemble(cfg.layout, &out, registry, world, &cfg.limit),
    };
    (record, accepted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn ent(id: &str, kind: EntityKind, name: &str, d: &str) -> Entity {
        Entity {
            id: id.into(),
            kind,
            name: name.into(),
            descriptor: d.into(),
        }
    }

    fn world() -> WorldBible {
        WorldBible {
            premise: "A lighthouse keeper on a stormy island.".into(),
            style: "35mm film, muted teal palette.".into(),
            entities: vec![
                ent(
                    "mara",
                    EntityKind::Character,
                    "Mara",
                    "a wiry woman in her sixties, yellow oilskin coat, grey braid",
                ),
                ent(
                    "tower",
                    EntityKind::Location,
                    "The lighthouse",
                    "a white stone lighthouse on black rocks, waves breaking below",
                ),
            ],
        }
    }

    fn out() -> ExpansionOutput {
        ExpansionOutput {
            beat_summary: "Mara climbs the stairs.".into(),
            entity_ids: vec!["mara".into(), "tower".into(), "ghost".into()],
            new_entities: vec![],
            action: "Mara climbs the spiral stairs, lamp swinging.".into(),
            camera: "slow tracking shot upward".into(),
            lighting: "warm lamp glow against blue dusk".into(),
            audio: "wind howling, iron steps clanging".into(),
            music: String::new(),
            dialogue: vec![],
        }
    }

    fn cfg() -> UpsamplerConfig {
        UpsamplerConfig {
            hedge: false,
            limit: PromptLimit::chars(10_000),
            timeout: Duration::from_secs(5),
            ..UpsamplerConfig::default()
        }
    }

    #[test]
    fn user_message_matches_the_spec_layout() {
        let w = world();
        let mut r = Registry::default();
        assert_eq!(
            build_user_message(&w, &r, &[], None),
            "WORLD BIBLE:\nA lighthouse keeper on a stormy island.\n\n\
             ESTABLISHED ENTITIES:\n(none)\n\n\
             RECENT SEGMENTS (oldest first):\n(stream is starting)\n\n\
             NEW DIRECTION:\n(none, advance the story one beat)"
        );
        for e in &w.entities {
            r.set(e.clone());
        }
        let h = vec![
            SegmentRecord {
                index: 0,
                direction_id: None,
                summary: "s0".into(),
                prompt: "p0".into(),
            },
            SegmentRecord {
                index: 1,
                direction_id: Some("2".into()),
                summary: "s1".into(),
                prompt: "p1".into(),
            },
        ];
        let d = Direction {
            id: "3".into(),
            text: "A ship appears.".into(),
        };
        assert_eq!(
            build_user_message(&w, &r, &h, Some(&d)),
            "WORLD BIBLE:\nA lighthouse keeper on a stormy island.\n\n\
             ESTABLISHED ENTITIES:\n- mara (character, \"Mara\")\n- tower (location, \"The lighthouse\")\n\n\
             RECENT SEGMENTS (oldest first):\n[0] s0\np0\n\n[1] s1\np1\n\n\
             NEW DIRECTION:\nA ship appears."
        );
        // Frames (ours): one line each under the direction, no description.
        let m = build_message(
            "P",
            &r,
            &[],
            None,
            Frames {
                first: true,
                end: true,
                attached: false,
            },
        );
        assert!(m.ends_with(
            "NEW DIRECTION:\n(none, advance the story one beat)\n\
             FIRST FRAME: this segment starts from a given image (not shown to you); continue from it.\n\
             FINAL FRAME: this segment must end exactly on a given image (not shown to you); move everything so that the shot arrives there."
        ), "{m}");
        assert!(build_message(
            "P",
            &r,
            &[],
            None,
            Frames {
                first: true,
                end: false,
                attached: true
            }
        )
        .contains("(attached)"));
    }

    #[test]
    fn assemble_orders_location_cast_action_and_style() {
        let w = world();
        let mut r = Registry::default();
        for e in &w.entities {
            r.set(e.clone());
        }
        // `ghost` is unknown: dropped from the cast.
        let p = assemble_prompt(&out(), &r, &w, 10_000);
        assert_eq!(
            p,
            "a white stone lighthouse on black rocks, waves breaking below \
             Mara: a wiry woman in her sixties, yellow oilskin coat, grey braid \
             Mara climbs the spiral stairs, lamp swinging. \
             Camera: slow tracking shot upward \
             Lighting: warm lamp glow against blue dusk \
             Sound: wind howling, iron steps clanging No spoken words. \
             35mm film, muted teal palette."
        );
        let mut o = out();
        o.entity_ids = vec!["mara".into()];
        o.dialogue = vec![
            DialogueLine {
                speaker_id: "mara".into(),
                line: "Light's out again.".into(),
            },
            DialogueLine {
                speaker_id: "nobody".into(),
                line: "Hello?".into(),
            },
        ];
        let w2 = WorldBible {
            style: String::new(),
            ..w.clone()
        };
        assert_eq!(
            assemble_prompt(&o, &r, &w2, 10_000),
            "Mara: a wiry woman in her sixties, yellow oilskin coat, grey braid \
             Mara climbs the spiral stairs, lamp swinging. \
             Mara says, \"Light's out again.\" a voice says, \"Hello?\" \
             Camera: slow tracking shot upward \
             Lighting: warm lamp glow against blue dusk \
             Sound: wind howling, iron steps clanging"
        );
    }

    /// Ours, not the TS: the style goes first, then (prose) lighting and
    /// camera; the sound line stays; nothing is sliced mid-word.
    #[test]
    fn assemble_trims_by_priority_and_keeps_the_sound() {
        let w = world();
        let mut r = Registry::default();
        for e in &w.entities {
            r.set(e.clone());
        }
        let full = assemble_prompt(&out(), &r, &w, 10_000);
        let n = full.chars().count();
        assert_eq!(assemble_prompt(&out(), &r, &w, n), full);
        let no_style = full
            .trim_end_matches(" 35mm film, muted teal palette.")
            .to_owned();
        assert_eq!(assemble_prompt(&out(), &r, &w, n - 1), no_style);
        let w2 = WorldBible {
            style: String::new(),
            ..w.clone()
        };
        let p = assemble_prompt(&out(), &r, &w2, n - 40);
        assert!(
            p.contains("Sound: wind howling") && !p.contains("Lighting:"),
            "{p}"
        );
        // Far under: the description is cut at a word, the sound kept.
        let p = assemble_prompt(&out(), &r, &w, 100);
        assert!(
            p.ends_with("Sound: wind howling, iron steps clanging No spoken words.")
                && p.chars().count() <= 100,
            "{p}"
        );
        assert!(p.starts_with("a white stone"), "{p}");
    }

    /// A backend answering a queue of canned contents, recording requests.
    struct Canned {
        answers: Mutex<Vec<Result<String, UpsampleError>>>,
        seen: Mutex<Vec<ChatRequest>>,
        tokens: Option<usize>,
    }

    #[async_trait::async_trait]
    impl ChatBackend for Canned {
        async fn complete(&self, req: &ChatRequest) -> Result<String, UpsampleError> {
            self.seen.lock().unwrap().push(req.clone());
            let mut a = self.answers.lock().unwrap();
            if a.is_empty() {
                return Err(UpsampleError::NoContent);
            }
            a.remove(0)
        }
        async fn tokenize(&self, _text: &str) -> Option<usize> {
            self.tokens
        }
    }

    fn canned(answers: Vec<Result<String, UpsampleError>>) -> Arc<Canned> {
        Arc::new(Canned {
            answers: Mutex::new(answers),
            seen: Mutex::new(Vec::new()),
            tokens: None,
        })
    }

    #[tokio::test]
    async fn next_registers_new_entities_once_and_keeps_the_ledger() {
        let mut o1 = out();
        o1.new_entities = vec![
            ent(
                "ship",
                EntityKind::Prop,
                "The ship",
                "a rusted trawler, red hull",
            ),
            ent(
                "mara",
                EntityKind::Character,
                "Mara",
                "someone else entirely",
            ),
        ];
        o1.entity_ids = vec!["tower".into(), "ship".into(), "mara".into(), "nope".into()];
        let mut o2 = out();
        o2.new_entities = vec![ent("ship", EntityKind::Prop, "Ship 2", "different")];
        let be = canned(vec![
            Ok(serde_json::to_string(&o1).unwrap()),
            Ok(serde_json::to_string(&o2).unwrap()),
        ]);
        let mut u = Upsampler::new(UpsamplerConfig { memory: 1, ..cfg() }, world(), be.clone());
        let s1 = u
            .next(Some(Direction {
                id: "2".into(),
                text: "A ship appears.".into(),
            }))
            .await
            .unwrap();
        assert_eq!(s1.record.index, 0);
        assert_eq!(s1.record.direction_id.as_deref(), Some("2"));
        assert_eq!(
            s1.new_entities,
            vec![ent(
                "ship",
                EntityKind::Prop,
                "The ship",
                "a rusted trawler, red hull"
            )]
        );
        assert!(
            s1.record
                .prompt
                .contains("The ship: a rusted trawler, red hull"),
            "{}",
            s1.record.prompt
        );
        assert!(
            s1.record.prompt.contains("Mara: a wiry woman"),
            "{}",
            s1.record.prompt
        );
        let s2 = u.next(None).await.unwrap();
        assert_eq!((s2.record.index, s2.record.direction_id.clone()), (1, None));
        assert!(s2.new_entities.is_empty());
        let snap = u.snapshot();
        assert_eq!(
            snap.registry
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>(),
            ["mara", "tower", "ship"]
        );
        assert_eq!(
            snap.registry[0].descriptor,
            "a wiry woman in her sixties, yellow oilskin coat, grey braid"
        );
        assert_eq!(snap.registry[2].name, "The ship");
        assert_eq!(snap.ledger.len(), 2);
        let seen = be.seen.lock().unwrap();
        assert!(seen[0].system.starts_with(SYSTEM_PROMPT) && seen[0].system.ends_with(MUSIC_RULE));
        assert_eq!(seen[0].temperature, 0.8);
        assert_eq!(
            seen[0].response_format.as_ref().unwrap()["json_schema"]["name"],
            "segment"
        );
        assert!(seen[0].user.contains("NEW DIRECTION:\nA ship appears."));
        assert!(
            seen[1].user.contains("[0] Mara climbs the stairs."),
            "{}",
            seen[1].user
        );
        assert!(seen[1].user.contains("- ship (prop, \"The ship\")"));
    }

    #[tokio::test]
    async fn a_failed_call_records_nothing() {
        let be = canned(vec![Ok("{\"beat_summary\": 1}".into())]);
        let mut u = Upsampler::new(cfg(), world(), be);
        let e = u.next(None).await.unwrap_err();
        assert!(matches!(e, UpsampleError::BadOutput(_)), "{e:?}");
        assert_eq!(u.ledger_len(), 0);
        let e: Entity =
            serde_json::from_str(r#"{"id":"x","kind":"animal","name":"X","descriptor":"d"}"#)
                .unwrap();
        assert_eq!(e.kind, EntityKind::Other);
        let schema = output_schema();
        assert_eq!(
            schema["schema"]["properties"]["new_entities"]["items"]["properties"]["kind"]["enum"],
            json!(["character", "location", "prop"])
        );
        // `music` is listed in `required` (strict) and is a plain string.
        assert!(schema["schema"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("music")));
        assert_eq!(
            schema["schema"]["properties"]["music"],
            json!({"type": "string"})
        );
    }

    #[tokio::test]
    async fn think_blocks_and_fences_are_tolerated_missing_fields_are_not() {
        let body = serde_json::to_string(&out()).unwrap();
        let be = canned(vec![
            Ok(format!(
                "<think>\nThe user wants {{...}}\n</think>\n```json\n{body}\n```"
            )),
            Ok("<think>no answer".into()),
            Ok("{\"beat_summary\": \"x\", \"entity_ids\": []}".into()),
        ]);
        let mut u = Upsampler::new(cfg(), world(), be);
        assert_eq!(
            u.next(None).await.unwrap().record.summary,
            "Mara climbs the stairs."
        );
        assert!(matches!(
            u.next(None).await.unwrap_err(),
            UpsampleError::BadOutput(_)
        ));
        assert!(matches!(
            u.next(None).await.unwrap_err(),
            UpsampleError::BadOutput(_)
        ));
        assert_eq!(u.ledger_len(), 1);
    }

    #[test]
    fn memory_windows_the_history() {
        let be = canned(vec![]);
        let mut u = Upsampler::new(UpsamplerConfig { memory: 2, ..cfg() }, world(), be.clone());
        for _ in 0..3 {
            u.commit(None, out());
        }
        let m = u.user_message(None);
        assert!(
            !m.contains("[0] ") && m.contains("[1] ") && m.contains("[2] "),
            "{m}"
        );
        let mut all = Upsampler::new(UpsamplerConfig { memory: 0, ..cfg() }, world(), be);
        for _ in 0..3 {
            all.commit(None, out());
        }
        assert!(all.user_message(None).contains("[0] "));
    }

    fn long_premise(chars: usize) -> String {
        let s = "The keeper Mara logs every ship that passes the island, and the island keeps its own secrets. ";
        let mut p = s.repeat(chars / s.len() + 1);
        p.truncate(chars);
        p
    }

    /// A 50 000-character premise (fal's limit) goes first and verbatim
    /// under the default 28 000-token budget, which it fits without a
    /// digest; the message stays within the budget.
    #[tokio::test]
    async fn a_50k_premise_is_sent_verbatim_first() {
        let premise = long_premise(50_000);
        let be = canned(vec![]);
        let w = WorldBible {
            premise: premise.clone(),
            ..world()
        };
        let mut u = Upsampler::new(cfg(), w, be.clone());
        let r = u.prepare().await.clone();
        assert_eq!(
            (r.chars, r.tokens, r.tokens_exact, r.digest.is_none()),
            (50_000, 14_286, false, true)
        );
        assert!(be.seen.lock().unwrap().is_empty(), "no digest call");
        for _ in 0..12 {
            u.commit(None, out());
        }
        let m = u.user_message(None);
        assert!(m.starts_with(&format!("WORLD BIBLE:\n{premise}\n\nESTABLISHED ENTITIES:")));
        assert!(
            m.contains("[11] ") && m.contains("[0] "),
            "12 short segments fit"
        );
        assert!(u.estimate(&m) <= u.input_budget());
    }

    /// The history is cut to the budget from the oldest end; the premise,
    /// the registry and the direction stay.
    #[test]
    fn budget_drops_the_oldest_segments_first() {
        let be = canned(vec![]);
        let mut u = Upsampler::new(
            UpsamplerConfig {
                context_tokens: 4_000,
                memory: 50,
                ..cfg()
            },
            world(),
            be,
        );
        let mut o = out();
        o.action = long_premise(2_000);
        for _ in 0..20 {
            u.commit(None, o.clone());
        }
        let d = Direction {
            id: "9".into(),
            text: "The storm breaks.".into(),
        };
        let m = u.user_message(Some(&d));
        assert!(
            u.estimate(&m) <= u.input_budget(),
            "{} > {}",
            u.estimate(&m),
            u.input_budget()
        );
        assert!(
            m.starts_with("WORLD BIBLE:\nA lighthouse keeper")
                && m.contains("- mara (character")
                && m.ends_with("NEW DIRECTION:\nThe storm breaks.")
        );
        let kept: Vec<usize> = (0..20)
            .filter(|i| m.contains(&format!("\n[{i}] ")))
            .collect();
        assert!(!kept.is_empty() && kept.len() < 20, "{kept:?}");
        assert_eq!(*kept.last().unwrap(), 19, "the newest stays");
        assert_eq!(
            kept,
            (kept[0]..20).collect::<Vec<_>>(),
            "a contiguous newest run"
        );
    }

    /// A premise over the budget is condensed at configure time: one call
    /// per piece plus one over the pieces, cached and used in every
    /// message; `/tokenize` makes the count exact.
    #[tokio::test]
    async fn an_oversized_premise_is_digested_once() {
        let premise = long_premise(50_000);
        let answers = (0..10)
            .map(|k| {
                Ok(format!(
                    "<think>x</think>BIBLE PART {k}. Mara keeps the light."
                ))
            })
            .collect();
        let be = Arc::new(Canned {
            answers: Mutex::new(answers),
            seen: Mutex::new(Vec::new()),
            tokens: Some(12_500),
        });
        let w = WorldBible {
            premise: premise.clone(),
            ..world()
        };
        let mut u = Upsampler::new(
            UpsamplerConfig {
                context_tokens: 8_000,
                ..cfg()
            },
            w,
            be.clone(),
        );
        let r = u.prepare().await.clone();
        assert!(
            r.tokens_exact && r.tokens == 12_500 && r.chars_per_token == 4.0,
            "{r:?}"
        );
        let d = r.digest.clone().expect("digested");
        let seen = be.seen.lock().unwrap().clone();
        assert!(d.calls >= 2 && seen.len() == d.calls, "{d:?}");
        assert!(seen
            .iter()
            .all(|q| q.system == DIGEST_SYSTEM_PROMPT && q.response_format.is_none()));
        assert!(seen[0]
            .user
            .starts_with("Condense this series premise to at most "));
        assert!(!d.fallback && d.error.is_none());
        let m = u.user_message(None);
        assert!(
            m.starts_with("WORLD BIBLE:\nBIBLE PART 0. Mara keeps the light."),
            "{m}"
        );
        assert!(!m.contains("logs every ship"));
        assert!(u.estimate(&m) <= u.input_budget());
        assert_eq!(
            u.premise_report().digest.as_ref().map(|d| d.calls),
            Some(seen.len())
        );
    }

    #[tokio::test]
    async fn a_failed_digest_falls_back_to_the_premise_head() {
        let premise = long_premise(50_000);
        let be = canned(vec![Err(UpsampleError::Status {
            status: 500,
            body: "x".into(),
        })]);
        let w = WorldBible { premise, ..world() };
        let mut u = Upsampler::new(
            UpsamplerConfig {
                context_tokens: 8_000,
                ..cfg()
            },
            w,
            be,
        );
        let d = u.prepare().await.digest.clone().unwrap();
        assert!(
            d.fallback && d.error.as_deref() == Some("upsampler LLM 500: x"),
            "{d:?}"
        );
        assert!(
            u.premise_text()
                .starts_with("The keeper Mara logs every ship")
                && u.estimate(u.premise_text()) <= u.premise_budget()
        );
    }

    /// Planned segments chain: the message after them sees them as the
    /// newest history and their new entities in the registry; nothing is
    /// committed.
    #[test]
    fn the_speculative_chain_is_not_committed() {
        let u = Upsampler::new(cfg(), world(), canned(vec![]));
        let mut o = out();
        o.beat_summary = "planned one".into();
        o.new_entities = vec![ent("ship", EntityKind::Prop, "Ship", "a red trawler")];
        let planned = vec![PlannedOutput {
            direction: Some(Direction {
                id: "2".into(),
                text: "A ship.".into(),
            }),
            out: o,
        }];
        let m = u.user_message_after(&planned, None, Frames::default());
        assert!(
            m.contains("[0] planned one\n") && m.contains("- ship (prop, \"Ship\")"),
            "{m}"
        );
        assert_eq!(u.ledger_len(), 0);
        assert!(!u.user_message(None).contains("planned one"));
    }

    #[test]
    fn fallbacks_stay_within_the_limit() {
        let be = canned(vec![]);
        let w = WorldBible {
            premise: long_premise(50_000),
            ..world()
        };
        let mut u = Upsampler::new(
            UpsamplerConfig {
                limit: PromptLimit::chars(200),
                ..cfg()
            },
            w,
            be,
        );
        let p = u.fallback_prompt(None);
        assert!(
            p.chars().count() <= 200 && p.starts_with("The keeper Mara"),
            "{p}"
        );
        assert_eq!(u.fallback_prompt(Some("Night falls.")), "Night falls.");
        u.commit(None, out());
        assert_eq!(u.fallback_prompt(None), u.last_prompt().unwrap());
    }

    #[test]
    fn split_text_cuts_at_boundaries() {
        let s = long_premise(1_000);
        let parts = split_text(&s, 300);
        assert!(parts.len() >= 4 && parts.iter().all(|p| p.chars().count() <= 300));
        assert!(
            parts[..parts.len() - 1].iter().all(|p| p.ends_with('.')),
            "{parts:?}"
        );
        assert_eq!(
            parts.join(" ").split_whitespace().count(),
            s.split_whitespace().count()
        );
    }
}
