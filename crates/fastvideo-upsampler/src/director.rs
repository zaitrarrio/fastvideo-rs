//! The director's showrunner upsampler: a 1:1 Rust port of the owner's
//! TypeScript `Upsampler` (docs/serve/upsampler-spec.ts, the spec).
//!
//! Each [`Upsampler::next`] writes the NEXT segment of one unbroken take:
//! the LLM gets the world bible's premise, the entity registry, the last
//! `memory` segments and an optional new direction, and answers the strict
//! `segment` JSON schema ([`output_schema`]); [`assemble_prompt`] turns that
//! into the generation prompt (location descriptor, `Name: descriptor` for
//! the rest of the cast, action, dialogue, camera / lighting / sound, the
//! world style last). New entities are accepted only under a new id, and
//! `entity_ids` are filtered to the registry.
//!
//! Where Rust needs a choice the TS does not make (each also listed in
//! docs/serve/prompt-upsampling.md §Director port):
//!
//! - **Lengths are Unicode scalar values** (`chars()`), not UTF-16 code
//!   units (JS `length` / `slice`); the cut never splits a character.
//! - **Strict parsing:** `JSON.parse(...) as ExpansionOutput` checks
//!   nothing; serde requires every schema field with its type (a missing
//!   field fails the call, where the TS would fail later on `undefined`).
//!   An unknown entity `kind` parses as [`EntityKind::Other`] (the TS keeps
//!   any string; only `"location"` is special in either).
//! - **`next` is split** into [`Upsampler::user_message`] (read-only),
//!   [`expand_with_hedge`] (the call) and [`Upsampler::commit`] (registry
//!   and ledger), so the director can expand the next segment while the
//!   current one generates and drop a speculative expansion when a new
//!   direction arrives; `next` itself is exactly the three in a row.
//! - `memory = 0` keeps the whole ledger, as `slice(-0)` does in JS (the
//!   director never passes 0: `configure.memory` is 1..50).
//! - Over `max_prompt_chars` the last part is dropped before the cut, as in
//!   the TS: that is the world style when there is one, and the `Sound:`
//!   line when the style is empty.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::chat::{hedged, ChatBackend, ChatRequest, UpsampleError};

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

/// `WorldBible`: the premise (the director's `configure.prompt`), the style
/// appended to every segment and the opening cast.
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
    pub dialogue: Vec<DialogueLine>,
}

/// `UpsamplerConfig` without the transport (`baseUrl`, `apiKey`, `model`
/// live in the [`ChatBackend`]).
#[derive(Clone, Debug, PartialEq)]
pub struct UpsamplerConfig {
    /// Prior segments kept as context (Director default 12, max 50).
    pub memory: usize,
    /// Keep under the text encoder limit (~1800 chars for umT5 512 tokens).
    pub max_prompt_chars: usize,
    pub hedge: bool,
    pub timeout: Duration,
}

impl Default for UpsamplerConfig {
    fn default() -> Self {
        Self { memory: 12, max_prompt_chars: 1800, hedge: true, timeout: Duration::from_secs(10) }
    }
}

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

/// The spec's sampling temperature.
pub const TEMPERATURE: f32 = 0.8;

/// `OUTPUT_SCHEMA` (the `json_schema` object of `response_format`).
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

/// `buildUserMessage`.
pub fn build_user_message(world: &WorldBible, registry: &Registry, history: &[SegmentRecord], direction: Option<&Direction>) -> String {
    let entities = registry
        .values()
        .iter()
        .map(|e| format!("- {} ({}, \"{}\")", e.id, e.kind.as_str(), e.name))
        .collect::<Vec<_>>()
        .join("\n");
    let recent = history
        .iter()
        .map(|s| format!("[{}] {}\n{}", s.index, s.summary, s.prompt))
        .collect::<Vec<_>>()
        .join("\n\n");
    [
        format!("WORLD BIBLE:\n{}", world.premise),
        format!("ESTABLISHED ENTITIES:\n{}", if entities.is_empty() { "(none)" } else { &entities }),
        format!("RECENT SEGMENTS (oldest first):\n{}", if recent.is_empty() { "(stream is starting)" } else { &recent }),
        match direction {
            Some(d) => format!("NEW DIRECTION:\n{}", d.text),
            None => "NEW DIRECTION:\n(none, advance the story one beat)".to_owned(),
        },
    ]
    .join("\n\n")
}

/// The chat request of `callModel` for `user`.
pub fn request_for(user: String) -> ChatRequest {
    ChatRequest {
        system: SYSTEM_PROMPT.to_owned(),
        user,
        temperature: TEMPERATURE,
        response_format: Some(json!({"type": "json_schema", "json_schema": output_schema()})),
    }
}

/// `callModel`: one completion, parsed as [`ExpansionOutput`].
pub async fn call_model(backend: &dyn ChatBackend, req: &ChatRequest) -> Result<ExpansionOutput, UpsampleError> {
    let content = backend.complete(req).await?;
    // Tolerant of small local models: a leading `<think>` block or a code
    // fence around the JSON is dropped first (an addition to the spec).
    serde_json::from_str(crate::chat::json_body(&content)).map_err(|e| UpsampleError::BadOutput(e.to_string()))
}

/// `expandWithHedge`: two concurrent calls with the hedge (one without),
/// the first success wins, everything is aborted at the timeout.
pub async fn expand_with_hedge(backend: &Arc<dyn ChatBackend>, hedge: bool, timeout: Duration, user: String) -> Result<ExpansionOutput, UpsampleError> {
    let req = request_for(user);
    let attempts = if hedge { 2 } else { 1 };
    hedged(attempts, timeout, || call_model(backend.as_ref(), &req)).await
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn char_slice(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `assemblePrompt`.
pub fn assemble_prompt(out: &ExpansionOutput, registry: &Registry, world: &WorldBible, max_chars: usize) -> String {
    let cast: Vec<&Entity> = out.entity_ids.iter().filter_map(|id| registry.get(id)).collect();
    let location = cast.iter().find(|e| e.kind == EntityKind::Location);
    let others = cast.iter().filter(|e| e.kind != EntityKind::Location);
    let name_of = |id: &str| registry.get(id).map_or("a voice", |e| e.name.as_str()).to_owned();

    let mut parts: Vec<String> = Vec::new();
    parts.push(location.map_or_else(String::new, |l| l.descriptor.clone()));
    parts.extend(others.map(|e| format!("{}: {}", e.name, e.descriptor)));
    parts.push(out.action.clone());
    parts.extend(out.dialogue.iter().map(|d| format!("{} says, \"{}\"", name_of(&d.speaker_id), d.line)));
    parts.push(format!("Camera: {}", out.camera));
    parts.push(format!("Lighting: {}", out.lighting));
    parts.push(format!("Sound: {}{}", out.audio, if out.dialogue.is_empty() { " No spoken words." } else { "" }));
    parts.push(world.style.clone());
    let parts: Vec<String> = parts.into_iter().filter(|p| !p.trim().is_empty()).collect();

    let prompt = parts.join(" ");
    if char_len(&prompt) > max_chars {
        // Style is the most expendable; identity text and action are not.
        let kept = parts[..parts.len().saturating_sub(1)].join(" ");
        return char_slice(&kept, max_chars).to_owned();
    }
    prompt
}

/// A read-only copy of the upsampler's state (`snapshot()`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub registry: Vec<Entity>,
    pub ledger: Vec<SegmentRecord>,
}

/// `class Upsampler`.
pub struct Upsampler {
    cfg: UpsamplerConfig,
    world: WorldBible,
    backend: Arc<dyn ChatBackend>,
    registry: Registry,
    ledger: Vec<SegmentRecord>,
}

impl std::fmt::Debug for Upsampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upsampler")
            .field("cfg", &self.cfg)
            .field("entities", &self.registry.len())
            .field("segments", &self.ledger.len())
            .finish_non_exhaustive()
    }
}

impl Upsampler {
    pub fn new(cfg: UpsamplerConfig, world: WorldBible, backend: Arc<dyn ChatBackend>) -> Self {
        let mut registry = Registry::default();
        for e in &world.entities {
            registry.set(e.clone());
        }
        Self { cfg, world, backend, registry, ledger: Vec::new() }
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

    /// The prompt of `next(direction)` on the current state: the last
    /// `memory` segments (all of them for `memory = 0`, as `slice(-0)`).
    pub fn user_message(&self, direction: Option<&Direction>) -> String {
        let m = self.cfg.memory;
        let start = if m == 0 { 0 } else { self.ledger.len().saturating_sub(m) };
        build_user_message(&self.world, &self.registry, &self.ledger[start..], direction)
    }

    /// The second half of `next`: accept the new entities (new ids only),
    /// keep only registered `entity_ids`, record the segment.
    pub fn commit(&mut self, direction: Option<&Direction>, mut out: ExpansionOutput) -> ExpandedSegment {
        let accepted: Vec<Entity> = out.new_entities.iter().filter(|e| !self.registry.has(&e.id)).cloned().collect();
        for e in &accepted {
            self.registry.set(e.clone());
        }
        out.entity_ids.retain(|id| self.registry.has(id));
        let record = SegmentRecord {
            index: self.ledger.len(),
            direction_id: direction.map(|d| d.id.clone()),
            summary: out.beat_summary.clone(),
            prompt: assemble_prompt(&out, &self.registry, &self.world, self.cfg.max_prompt_chars),
        };
        self.ledger.push(record.clone());
        ExpandedSegment { record, new_entities: accepted }
    }

    /// `next(direction)`.
    pub async fn next(&mut self, direction: Option<Direction>) -> Result<ExpandedSegment, UpsampleError> {
        let user = self.user_message(direction.as_ref());
        let out = expand_with_hedge(&self.backend, self.cfg.hedge, self.cfg.timeout, user).await?;
        Ok(self.commit(direction.as_ref(), out))
    }

    /// `snapshot()`.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot { registry: self.registry.values().to_vec(), ledger: self.ledger.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn ent(id: &str, kind: EntityKind, name: &str, d: &str) -> Entity {
        Entity { id: id.into(), kind, name: name.into(), descriptor: d.into() }
    }

    fn world() -> WorldBible {
        WorldBible {
            premise: "A lighthouse keeper on a stormy island.".into(),
            style: "35mm film, muted teal palette.".into(),
            entities: vec![
                ent("mara", EntityKind::Character, "Mara", "a wiry woman in her sixties, yellow oilskin coat, grey braid"),
                ent("tower", EntityKind::Location, "The lighthouse", "a white stone lighthouse on black rocks, waves breaking below"),
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
            dialogue: vec![],
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
            SegmentRecord { index: 0, direction_id: None, summary: "s0".into(), prompt: "p0".into() },
            SegmentRecord { index: 1, direction_id: Some("2".into()), summary: "s1".into(), prompt: "p1".into() },
        ];
        let d = Direction { id: "3".into(), text: "A ship appears.".into() };
        assert_eq!(
            build_user_message(&w, &r, &h, Some(&d)),
            "WORLD BIBLE:\nA lighthouse keeper on a stormy island.\n\n\
             ESTABLISHED ENTITIES:\n- mara (character, \"Mara\")\n- tower (location, \"The lighthouse\")\n\n\
             RECENT SEGMENTS (oldest first):\n[0] s0\np0\n\n[1] s1\np1\n\n\
             NEW DIRECTION:\nA ship appears."
        );
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
        // Dialogue: `Name says, "…"`, unknown speakers are "a voice", and
        // the "No spoken words." suffix goes.
        let mut o = out();
        o.entity_ids = vec!["mara".into()];
        o.dialogue = vec![
            DialogueLine { speaker_id: "mara".into(), line: "Light's out again.".into() },
            DialogueLine { speaker_id: "nobody".into(), line: "Hello?".into() },
        ];
        let w2 = WorldBible { style: String::new(), ..w.clone() };
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

    #[test]
    fn assemble_drops_the_style_then_cuts() {
        let w = world();
        let mut r = Registry::default();
        for e in &w.entities {
            r.set(e.clone());
        }
        let full = assemble_prompt(&out(), &r, &w, 10_000);
        let n = full.chars().count();
        // Exactly at the limit: unchanged.
        assert_eq!(assemble_prompt(&out(), &r, &w, n), full);
        // One under: the style goes (and the rest fits).
        let no_style = full.trim_end_matches(" 35mm film, muted teal palette.").to_owned();
        assert_eq!(assemble_prompt(&out(), &r, &w, n - 1), no_style);
        // Far under: without the style, cut to the limit.
        assert_eq!(assemble_prompt(&out(), &r, &w, 20), "a white stone lighth");
        assert_eq!(assemble_prompt(&out(), &r, &w, 20).chars().count(), 20);
        // The cut counts characters, never splitting one.
        let mut o = out();
        o.entity_ids.clear();
        o.action = "é".repeat(30);
        let p = assemble_prompt(&o, &r, &w, 10);
        assert_eq!(p, "é".repeat(10));
        // The TS drops the last part whatever it is: with no style, `Sound:`.
        let w2 = WorldBible { style: String::new(), ..w.clone() };
        let p = assemble_prompt(&out(), &r, &w2, n - 40);
        assert!(!p.contains("Sound:"), "{p}");
    }

    /// A backend answering a queue of canned contents, recording requests.
    struct Canned {
        answers: Mutex<Vec<Result<String, UpsampleError>>>,
        seen: Mutex<Vec<ChatRequest>>,
    }

    #[async_trait::async_trait]
    impl ChatBackend for Canned {
        async fn complete(&self, req: &ChatRequest) -> Result<String, UpsampleError> {
            self.seen.lock().unwrap().push(req.clone());
            self.answers.lock().unwrap().remove(0)
        }
    }

    fn canned(answers: Vec<Result<String, UpsampleError>>) -> Arc<Canned> {
        Arc::new(Canned { answers: Mutex::new(answers), seen: Mutex::new(Vec::new()) })
    }

    #[tokio::test]
    async fn next_registers_new_entities_once_and_keeps_the_ledger() {
        let mut o1 = out();
        o1.new_entities = vec![
            ent("ship", EntityKind::Prop, "The ship", "a rusted trawler, red hull"),
            // An existing id is not accepted (the registered one stays).
            ent("mara", EntityKind::Character, "Mara", "someone else entirely"),
        ];
        o1.entity_ids = vec!["tower".into(), "ship".into(), "mara".into(), "nope".into()];
        let mut o2 = out();
        o2.new_entities = vec![ent("ship", EntityKind::Prop, "Ship 2", "different")];
        let be = canned(vec![Ok(serde_json::to_string(&o1).unwrap()), Ok(serde_json::to_string(&o2).unwrap())]);
        let cfg = UpsamplerConfig { memory: 1, max_prompt_chars: 10_000, hedge: false, timeout: Duration::from_secs(5) };
        let mut u = Upsampler::new(cfg, world(), be.clone());
        let s1 = u.next(Some(Direction { id: "2".into(), text: "A ship appears.".into() })).await.unwrap();
        assert_eq!(s1.record.index, 0);
        assert_eq!(s1.record.direction_id.as_deref(), Some("2"));
        assert_eq!(s1.new_entities, vec![ent("ship", EntityKind::Prop, "The ship", "a rusted trawler, red hull")]);
        assert!(s1.record.prompt.contains("The ship: a rusted trawler, red hull"), "{}", s1.record.prompt);
        assert!(s1.record.prompt.contains("Mara: a wiry woman"), "{}", s1.record.prompt);
        let s2 = u.next(None).await.unwrap();
        assert_eq!((s2.record.index, s2.record.direction_id.clone()), (1, None));
        assert!(s2.new_entities.is_empty());
        let snap = u.snapshot();
        assert_eq!(snap.registry.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["mara", "tower", "ship"]);
        assert_eq!(snap.registry[0].descriptor, "a wiry woman in her sixties, yellow oilskin coat, grey braid");
        assert_eq!(snap.registry[2].name, "The ship");
        assert_eq!(snap.ledger.len(), 2);
        // The requests: the spec's system prompt, schema and temperature;
        // memory 1 shows only the last segment.
        let seen = be.seen.lock().unwrap();
        assert_eq!(seen[0].system, SYSTEM_PROMPT);
        assert_eq!(seen[0].temperature, 0.8);
        assert_eq!(seen[0].response_format.as_ref().unwrap()["json_schema"]["name"], "segment");
        assert!(seen[0].user.contains("NEW DIRECTION:\nA ship appears."));
        assert!(seen[1].user.contains("[0] Mara climbs the stairs."), "{}", seen[1].user);
        assert!(seen[1].user.contains("- ship (prop, \"The ship\")"));
    }

    #[tokio::test]
    async fn a_failed_call_records_nothing() {
        let be = canned(vec![Ok("{\"beat_summary\": 1}".into())]);
        let cfg = UpsamplerConfig { hedge: false, ..UpsamplerConfig::default() };
        let mut u = Upsampler::new(cfg, world(), be);
        let e = u.next(None).await.unwrap_err();
        assert!(matches!(e, UpsampleError::BadOutput(_)), "{e:?}");
        assert_eq!(u.ledger_len(), 0);
        // Unknown kinds parse (as Other); the schema keeps the 3 kinds.
        let e: Entity = serde_json::from_str(r#"{"id":"x","kind":"animal","name":"X","descriptor":"d"}"#).unwrap();
        assert_eq!(e.kind, EntityKind::Other);
        assert_eq!(output_schema()["schema"]["properties"]["new_entities"]["items"]["properties"]["kind"]["enum"], json!(["character", "location", "prop"]));
    }

    #[tokio::test]
    async fn think_blocks_and_fences_are_tolerated_missing_fields_are_not() {
        let body = serde_json::to_string(&out()).unwrap();
        let be = canned(vec![
            Ok(format!("<think>\nThe user wants {{...}}\n</think>\n```json\n{body}\n```")),
            Ok("<think>no answer".into()),
            Ok("{\"beat_summary\": \"x\", \"entity_ids\": []}".into()),
        ]);
        let cfg = UpsamplerConfig { hedge: false, ..UpsamplerConfig::default() };
        let mut u = Upsampler::new(cfg, world(), be);
        assert_eq!(u.next(None).await.unwrap().record.summary, "Mara climbs the stairs.");
        assert!(matches!(u.next(None).await.unwrap_err(), UpsampleError::BadOutput(_)));
        assert!(matches!(u.next(None).await.unwrap_err(), UpsampleError::BadOutput(_)));
        assert_eq!(u.ledger_len(), 1);
    }

    #[test]
    fn memory_windows_the_history() {
        let be = canned(vec![]);
        let mut u = Upsampler::new(UpsamplerConfig { memory: 2, ..UpsamplerConfig::default() }, world(), be.clone());
        for _ in 0..3 {
            u.commit(None, out());
        }
        let m = u.user_message(None);
        assert!(!m.contains("[0] ") && m.contains("[1] ") && m.contains("[2] "), "{m}");
        let mut all = Upsampler::new(UpsamplerConfig { memory: 0, ..UpsamplerConfig::default() }, world(), be);
        for _ in 0..3 {
            all.commit(None, out());
        }
        assert!(all.user_message(None).contains("[0] "));
    }
}
