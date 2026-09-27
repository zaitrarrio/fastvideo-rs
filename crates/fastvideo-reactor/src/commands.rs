//! Command and model-message schemas for the two Reactor modes (design §5.7).
//!
//! - **Clip** (H3, LTX, FastWan): the fast-h3 command set verbatim
//!   (reactor §4bis): `enqueue`, `play`, `pop`, `move`, `stop`, `get_queue`,
//!   `get_state`, `set_clip_seconds`, `set_seed`, `set_autoplay`,
//!   `set_canvas`, `reset`.
//! - **Causal** (SF-Wan): Waypoint-style `InputState` setters `set_prompt`,
//!   `set_paused`, `set_seed`, plus `reset`.
//!
//! A [`CommandTable`] validates `Command.data` like RT's model contract
//! (types, bounds, `max_length`, choices, required fields, defaults) and
//! feeds the OpenAPI document of [`crate::schema`]. A command that fails
//! validation is answered `Error{code:"invalid_command"}` (v1).

use serde_json::{json, Map, Value};

use crate::engine::Mode;

/// A parameter's type and constraints (RT `InputField`).
#[derive(Clone, Debug, PartialEq)]
pub enum ParamType {
    String { max_len: Option<usize> },
    Integer { min: Option<i64>, max: Option<i64> },
    Number { min: Option<f64>, max: Option<f64> },
    Boolean,
    Choice(Vec<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: &'static str,
    pub ty: ParamType,
    pub required: bool,
    /// `null` is accepted (`T | None`).
    pub nullable: bool,
    /// Filled in when the field is absent.
    pub default: Option<Value>,
    pub description: &'static str,
    /// `x-reactor-moderate` (we run no moderation, design R8).
    pub moderate: bool,
}

impl Param {
    fn new(name: &'static str, ty: ParamType, description: &'static str) -> Self {
        Self { name, ty, required: true, nullable: false, default: None, description, moderate: false }
    }
    fn optional(mut self, default: Value) -> Self {
        self.required = false;
        self.nullable = default.is_null();
        self.default = Some(default);
        self
    }
    fn moderate(mut self) -> Self {
        self.moderate = true;
        self
    }

    /// JSON schema of this parameter.
    pub fn schema(&self) -> Value {
        let mut s = match &self.ty {
            ParamType::String { max_len } => {
                let mut s = json!({"type": "string"});
                if let Some(m) = max_len {
                    s["maxLength"] = json!(m);
                }
                s
            }
            ParamType::Integer { min, max } => {
                let mut s = json!({"type": "integer"});
                if let Some(m) = min {
                    s["minimum"] = json!(m);
                }
                if let Some(m) = max {
                    s["maximum"] = json!(m);
                }
                s
            }
            ParamType::Number { min, max } => {
                let mut s = json!({"type": "number"});
                if let Some(m) = min {
                    s["minimum"] = json!(m);
                }
                if let Some(m) = max {
                    s["maximum"] = json!(m);
                }
                s
            }
            ParamType::Boolean => json!({"type": "boolean"}),
            ParamType::Choice(c) => json!({"type": "string", "enum": c}),
        };
        if self.nullable {
            s = json!({"anyOf": [s, {"type": "null"}]});
        }
        s["description"] = json!(self.description);
        if let Some(d) = &self.default {
            s["default"] = d.clone();
        }
        s["x-reactor-moderate"] = json!(self.moderate);
        s
    }

    fn check(&self, v: &Value) -> Result<Value, String> {
        if v.is_null() {
            return if self.nullable {
                Ok(Value::Null)
            } else {
                Err(format!("`{}` may not be null", self.name))
            };
        }
        let bad = |what: &str| Err(format!("`{}` must be {what}", self.name));
        match &self.ty {
            ParamType::String { max_len } => {
                let Some(s) = v.as_str() else { return bad("a string") };
                if let Some(m) = max_len {
                    if s.chars().count() > *m {
                        return Err(format!("`{}` is longer than {m} characters", self.name));
                    }
                }
                Ok(v.clone())
            }
            ParamType::Integer { min, max } => {
                let i = match v {
                    Value::Number(n) if n.is_i64() => n.as_i64(),
                    Value::Number(n) if n.is_u64() => n.as_u64().and_then(|u| i64::try_from(u).ok()),
                    Value::Number(n) => n.as_f64().filter(|f| f.fract() == 0.0 && f.abs() < 9.0e15).map(|f| f as i64),
                    _ => None,
                };
                let Some(i) = i else { return bad("an integer") };
                if min.is_some_and(|m| i < m) || max.is_some_and(|m| i > m) {
                    return Err(format!("`{}` = {i} is out of range", self.name));
                }
                Ok(json!(i))
            }
            ParamType::Number { min, max } => {
                let Some(f) = v.as_f64() else { return bad("a number") };
                if !f.is_finite() || min.is_some_and(|m| f < m) || max.is_some_and(|m| f > m) {
                    return Err(format!("`{}` = {f} is out of range", self.name));
                }
                Ok(v.clone())
            }
            ParamType::Boolean => match v {
                Value::Bool(_) => Ok(v.clone()),
                _ => bad("a boolean"),
            },
            ParamType::Choice(c) => match v.as_str() {
                Some(s) if c.iter().any(|x| x == s) => Ok(v.clone()),
                _ => Err(format!("`{}` must be one of {}", self.name, c.join(", "))),
            },
        }
    }
}

/// One command (`paths["/events/<name>"]`).
#[derive(Clone, Debug, PartialEq)]
pub struct CommandSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub params: Vec<Param>,
    /// The message type of the correlated reply; `None`: bodyless ack.
    pub reply: Option<&'static str>,
}

/// One model message (`webhooks`).
#[derive(Clone, Debug, PartialEq)]
pub struct MessageSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON schema of `ModelMessage.data`.
    pub schema: Value,
}

/// A mode's command set.
#[derive(Clone, Debug, PartialEq)]
pub struct CommandTable {
    pub mode: Mode,
    pub commands: Vec<CommandSpec>,
    pub messages: Vec<MessageSpec>,
}

impl CommandTable {
    pub fn get(&self, name: &str) -> Option<&CommandSpec> {
        self.commands.iter().find(|c| c.name == name)
    }

    /// Validates `data` against the command's parameters and fills in the
    /// defaults. Unknown fields are ignored.
    pub fn validate(&self, name: &str, data: &Map<String, Value>) -> Result<Map<String, Value>, String> {
        let c = self.get(name).ok_or_else(|| format!("unknown command `{name}`"))?;
        let mut out = Map::new();
        for p in &c.params {
            match data.get(p.name) {
                Some(v) => {
                    out.insert(p.name.to_owned(), p.check(v)?);
                }
                None if p.required => return Err(format!("`{}` is required", p.name)),
                None => {
                    out.insert(p.name.to_owned(), p.default.clone().unwrap_or(Value::Null));
                }
            }
        }
        Ok(out)
    }

    pub fn for_mode(mode: Mode, bounds: ClipBounds) -> Self {
        match mode {
            Mode::Clip => clip_table(bounds),
            Mode::Causal => causal_table(),
        }
    }
}

/// Clip-length bounds and default from the model caps (design §5.7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipBounds {
    pub min_s: f64,
    pub max_s: f64,
    pub default_s: f64,
}

/// `set_canvas` aspects (fast-h3).
pub const ASPECTS: [&str; 4] = ["16:9", "1:1", "9:16", "4:3"];
/// fast-h3 limits.
pub const PROMPT_MAX: usize = 800;
pub const METADATA_MAX: usize = 2000;

fn s(max: usize) -> ParamType {
    ParamType::String { max_len: Some(max) }
}
fn uint() -> ParamType {
    ParamType::Integer { min: Some(0), max: None }
}

fn obj(props: Value, required: &[&str]) -> Value {
    json!({"type": "object", "properties": props, "required": required})
}

fn clip_info_schema() -> Value {
    obj(
        json!({
            "clip_id": {"type": "string", "format": "uuid"},
            "prompt": {"type": "string"}, "metadata": {"type": "string"},
            "frames": {"type": "integer"}, "seconds": {"type": "number"},
            "seed": {"type": "integer"}, "ready": {"type": "boolean"}
        }),
        &["clip_id", "prompt", "metadata", "frames", "seconds", "seed", "ready"],
    )
}

fn clip_table(b: ClipBounds) -> CommandTable {
    let clip_id = |req: bool| {
        let p = Param::new("clip_id", ParamType::String { max_len: Some(64) }, "Clip id (uuid).");
        if req { p } else { p.optional(json!("")) }
    };
    let seconds = || ParamType::Number { min: Some(b.min_s), max: Some(b.max_s) };
    let commands = vec![
        CommandSpec {
            name: "enqueue",
            description: "Queue a clip for generation.",
            params: vec![
                Param::new("prompt", s(PROMPT_MAX), "What the clip shows.").moderate(),
                Param::new("metadata", s(METADATA_MAX), "Opaque; echoed back in every clip message.")
                    .optional(json!(""))
                    .moderate(),
                Param::new("seed", uint(), "Seed; null uses the session seed.").optional(Value::Null),
                Param::new("seconds", seconds(), "Clip length; snapped to the model frame grid. null uses the session clip length.")
                    .optional(Value::Null),
                Param::new("position", uint(), "Generation-queue position (0 = front); null appends.")
                    .optional(Value::Null),
            ],
            reply: Some("clip_queued"),
        },
        CommandSpec { name: "play", description: "Play a ready clip (blank: the playout front).", params: vec![clip_id(false)], reply: None },
        CommandSpec { name: "pop", description: "Remove a queued clip.", params: vec![clip_id(true)], reply: Some("clip_popped") },
        CommandSpec {
            name: "move",
            description: "Move a queued clip.",
            params: vec![clip_id(true), Param::new("position", uint(), "New position (0 = front).")],
            reply: Some("clip_moved"),
        },
        CommandSpec { name: "stop", description: "Stop the playing clip and hold on black.", params: vec![], reply: None },
        CommandSpec { name: "get_queue", description: "The queues.", params: vec![], reply: Some("queue_update") },
        CommandSpec { name: "get_state", description: "The session state.", params: vec![], reply: Some("state_update") },
        CommandSpec {
            name: "set_clip_seconds",
            description: "Default clip length for new clips.",
            params: vec![Param::new("seconds", seconds(), "Seconds; snapped to the model frame grid.")],
            reply: Some("clip_length_accepted"),
        },
        CommandSpec { name: "set_seed", description: "Session seed.", params: vec![Param::new("seed", uint(), "Seed.")], reply: Some("seed_accepted") },
        CommandSpec {
            name: "set_autoplay",
            description: "Play ready clips back to back.",
            params: vec![Param::new("enabled", ParamType::Boolean, "Autoplay on or off.")],
            reply: Some("autoplay_accepted"),
        },
        CommandSpec {
            name: "set_canvas",
            description: "Output aspect; only while nothing is queued or playing.",
            params: vec![Param::new("aspect", ParamType::Choice(ASPECTS.iter().map(|a| a.to_string()).collect()), "Aspect ratio.")],
            reply: Some("canvas_accepted"),
        },
        CommandSpec { name: "reset", description: "Clear both queues and stop playback.", params: vec![], reply: Some("session_reset") },
    ];
    let clip = json!({"clip": clip_info_schema()});
    let m = |name, description, schema| MessageSpec { name, description, schema };
    let messages = vec![
        m("clip_queued", "A clip entered the generation queue.", obj(clip.clone(), &["clip"])),
        m("clip_generated", "A clip finished building and is ready to play.", obj(clip.clone(), &["clip"])),
        m("clip_moved", "A clip moved.", obj(json!({"clip": clip_info_schema(), "queue": {"type": "string"}, "position": {"type": "integer"}}), &["clip", "queue", "position"])),
        m("clip_started", "A clip started playing.", obj(clip.clone(), &["clip"])),
        m("clip_finished", "A clip played to its end.", obj(clip.clone(), &["clip"])),
        m("clip_stopped", "The playing clip was stopped.", obj(clip.clone(), &["clip"])),
        m("clip_popped", "A clip was removed.", obj(clip.clone(), &["clip"])),
        m("clip_failed", "A clip failed to build.", obj(json!({"clip": clip_info_schema(), "reason": {"type": "string"}}), &["clip", "reason"])),
        m("clip_length_accepted", "New default clip length.", obj(json!({"clip_seconds": {"type": "number"}, "frames": {"type": "integer"}}), &["clip_seconds", "frames"])),
        m("seed_accepted", "New session seed.", obj(json!({"seed": {"type": "integer"}}), &["seed"])),
        m("autoplay_accepted", "Autoplay changed.", obj(json!({"enabled": {"type": "boolean"}}), &["enabled"])),
        m("canvas_accepted", "New canvas.", obj(json!({"aspect": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}}), &["aspect", "width", "height"])),
        m("session_reset", "Queues cleared.", obj(json!({"cleared_clips": {"type": "integer"}, "was_playing": {"type": "boolean"}}), &["cleared_clips", "was_playing"])),
        m("state_update", "Session state.", obj(json!({
            "clip_seconds": {"type": "number"}, "clip_seconds_min": {"type": "number"}, "clip_seconds_max": {"type": "number"},
            "seed": {"type": "integer"}, "autoplay": {"type": "boolean"}, "aspect": {"type": "string"},
            "width": {"type": "integer"}, "height": {"type": "integer"}, "playing": {"type": "boolean"},
            "playing_clip_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "generation_queued": {"type": "integer"}, "generation_capacity": {"type": "integer"},
            "playout_queued": {"type": "integer"}, "playout_capacity": {"type": "integer"},
            "clips_played": {"type": "integer"}, "seconds_sent": {"type": "number"},
            "valid_commands": {"type": "array", "items": {"type": "string"}}
        }), &[])),
        m("queue_update", "The queues.", obj(json!({
            "generation": {"type": "array", "items": clip_info_schema()},
            "playout": {"type": "array", "items": clip_info_schema()},
            "playing": {"anyOf": [clip_info_schema(), {"type": "null"}]}
        }), &["generation", "playout"])),
        m("command_error", "A refused command.", command_error_schema()),
    ];
    CommandTable { mode: Mode::Clip, commands, messages }
}

fn command_error_schema() -> Value {
    obj(json!({"command": {"type": "string"}, "reason": {"type": "string"}}), &["command", "reason"])
}

fn causal_table() -> CommandTable {
    let commands = vec![
        CommandSpec {
            name: "set_prompt",
            description: "Prompt; applied at the next block boundary.",
            params: vec![Param::new("prompt", s(PROMPT_MAX), "Prompt.").moderate()],
            reply: None,
        },
        CommandSpec {
            name: "set_paused",
            description: "Pause or resume generation.",
            params: vec![Param::new("paused", ParamType::Boolean, "Paused.")],
            reply: None,
        },
        CommandSpec { name: "set_seed", description: "Seed.", params: vec![Param::new("seed", uint(), "Seed.")], reply: None },
        CommandSpec { name: "reset", description: "Clear the KV cache and restart at block 0.", params: vec![], reply: None },
    ];
    let messages = vec![
        MessageSpec {
            name: "state_update",
            description: "Session state.",
            schema: obj(
                json!({
                    "prompt": {"type": "string"}, "paused": {"type": "boolean"}, "seed": {"type": "integer"},
                    "block_index": {"type": "integer"}, "unique_fps": {"type": "number"}
                }),
                &["prompt", "paused", "seed", "block_index", "unique_fps"],
            ),
        },
        MessageSpec { name: "command_error", description: "A refused command.", schema: command_error_schema() },
    ];
    CommandTable { mode: Mode::Causal, commands, messages }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b() -> ClipBounds {
        ClipBounds { min_s: 5.167, max_s: 14.375, default_s: 5.167 }
    }

    fn m(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn clip_set_is_fast_h3_verbatim() {
        let t = CommandTable::for_mode(Mode::Clip, b());
        let names: Vec<&str> = t.commands.iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            ["enqueue", "play", "pop", "move", "stop", "get_queue", "get_state", "set_clip_seconds", "set_seed", "set_autoplay", "set_canvas", "reset"]
        );
        let msgs: Vec<&str> = t.messages.iter().map(|c| c.name).collect();
        for want in ["clip_queued", "clip_generated", "clip_moved", "clip_started", "clip_finished", "clip_stopped", "clip_popped", "clip_failed", "clip_length_accepted", "seed_accepted", "autoplay_accepted", "canvas_accepted", "session_reset", "state_update", "queue_update", "command_error"] {
            assert!(msgs.contains(&want), "{want}");
        }
    }

    #[test]
    fn validation_follows_the_contract() {
        let t = CommandTable::for_mode(Mode::Clip, b());
        let ok = t.validate("enqueue", &m(json!({"prompt": "a cat"}))).unwrap();
        assert_eq!(Value::Object(ok), json!({"prompt": "a cat", "metadata": "", "seed": null, "seconds": null, "position": null}));
        assert!(t.validate("enqueue", &m(json!({}))).unwrap_err().contains("required"));
        assert!(t.validate("enqueue", &m(json!({"prompt": "x".repeat(801)}))).is_err());
        assert!(t.validate("enqueue", &m(json!({"prompt": "a", "seconds": 3.0}))).is_err());
        assert!(t.validate("enqueue", &m(json!({"prompt": "a", "seed": -1}))).is_err());
        // Struct numbers are doubles: a whole 7.0 is an integer.
        assert_eq!(t.validate("set_seed", &m(json!({"seed": 7.0}))).unwrap()["seed"], json!(7));
        assert!(t.validate("set_seed", &m(json!({"seed": 7.5}))).is_err());
        assert!(t.validate("set_canvas", &m(json!({"aspect": "21:9"}))).is_err());
        assert!(t.validate("set_canvas", &m(json!({"aspect": "9:16"}))).is_ok());
        assert!(t.validate("set_autoplay", &m(json!({"enabled": "yes"}))).is_err());
        assert!(t.validate("nope", &Map::new()).unwrap_err().contains("unknown"));
        // Extra fields are ignored.
        assert!(t.validate("get_state", &m(json!({"x": 1}))).is_ok());

        let c = CommandTable::for_mode(Mode::Causal, b());
        let names: Vec<&str> = c.commands.iter().map(|c| c.name).collect();
        assert_eq!(names, ["set_prompt", "set_paused", "set_seed", "reset"]);
        assert!(c.validate("set_paused", &m(json!({"paused": true}))).is_ok());
    }
}
