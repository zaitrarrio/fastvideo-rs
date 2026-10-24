//! `ParamSpec`: one typed description per (surface, model) of every request
//! parameter (plan E2, ADR-0012).
//!
//! A [`ParamSpec`] lists, for one submit endpoint of one API surface and one
//! served model, every field the endpoint reads: its wire name, JSON type,
//! whether it is required or nullable, its default, the values this server
//! accepts ([`Domain`]), whether the model honours it ([`Support`]), where
//! the check runs ([`Param::wire`] at parse time, [`Param::domain`] for the
//! model) and which [`crate::GenerationRequest`] field it fills.
//!
//! Two sources, and nothing else, decide a spec:
//!
//! - the **surface rules** ([`rules`]): every limit or allowed set an
//!   upstream contract fixes (the vendored specs under `docs/serve/specs/`),
//!   one constant each, cited. Adapters validate with these constants and
//!   build their specs from them; no adapter keeps a numeric limit of its
//!   own (the `param_scan` test enforces it);
//! - the model's [`ModelCaps`] (the family manifests' layer 1 plus
//!   `[variants.serving]`, projected by the engine): frame grid, rates,
//!   canvas tiers, knobs, reference limits.
//!
//! The adapters build their surface's specs (`param_specs` in each adapter
//! crate); `fv-serve` serves them all at `GET /fv/v1/params`. The plan's E3
//! certification matrix walks them: every [`Param`] yields a valid and an
//! invalid value ([`Domain::valid_example`], [`Domain::invalid_examples`]).

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::caps::{CanvasCaps, FrameGrid, ModelCaps};
use crate::error::GapId;

mod json_schema;
pub mod rules;

pub use json_schema::params_from_json_schema;

// ---- bounds --------------------------------------------------------------------

/// An inclusive integer range `min..=max`. Displays as `min..=max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntRange {
    pub min: i64,
    pub max: i64,
}

impl IntRange {
    pub const fn new(min: i64, max: i64) -> Self {
        Self { min, max }
    }
    pub const fn contains(&self, v: i64) -> bool {
        v >= self.min && v <= self.max
    }
    /// [`IntRange::contains`] for an unsigned value.
    pub fn contains_u64(&self, v: u64) -> bool {
        i64::try_from(v).is_ok_and(|v| self.contains(v))
    }
    /// `min` as `u32` (0 when negative).
    pub fn min_u32(&self) -> u32 {
        u32::try_from(self.min.max(0)).unwrap_or(u32::MAX)
    }
    /// `max` as `u32` (saturating).
    pub fn max_u32(&self) -> u32 {
        u32::try_from(self.max.max(0)).unwrap_or(u32::MAX)
    }
}

impl fmt::Display for IntRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..={}", self.min, self.max)
    }
}

/// An inclusive number range `min..=max` (NaN is never inside). Displays
/// with Rust's shortest float form (`0..=20`, `1..=3.5`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumRange {
    pub min: f64,
    pub max: f64,
}

impl NumRange {
    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
    pub fn contains(&self, v: f64) -> bool {
        v >= self.min && v <= self.max
    }
}

impl fmt::Display for NumRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..={}", self.min, self.max)
    }
}

/// An integer field: its range and the value an omitted field takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntField {
    pub range: IntRange,
    pub default: i64,
}

impl IntField {
    pub const fn new(min: i64, max: i64, default: i64) -> Self {
        Self {
            range: IntRange::new(min, max),
            default,
        }
    }
    /// `default` as `u32` (0 when negative, saturating).
    pub fn default_u32(&self) -> u32 {
        u32::try_from(self.default.max(0)).unwrap_or(u32::MAX)
    }
    /// `default` as `usize` (0 when negative, saturating).
    pub fn default_usize(&self) -> usize {
        usize::try_from(self.default.max(0)).unwrap_or(usize::MAX)
    }
}

/// A number field: its range and the value an omitted field takes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumField {
    pub range: NumRange,
    pub default: f64,
}

impl NumField {
    pub const fn new(min: f64, max: f64, default: f64) -> Self {
        Self {
            range: NumRange::new(min, max),
            default,
        }
    }
}

// ---- the spec ------------------------------------------------------------------

/// JSON type of a field on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    Integer,
    Number,
    String,
    Boolean,
    Object,
    Array,
    /// An integer or a string (`seconds: "8"`, fal LTX `duration: "auto"`).
    IntegerOrString,
    /// A JSON value the schema does not type (accepted as-is or refused).
    Any,
}

/// The media an URL field carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaField {
    Image,
    Video,
    Audio,
}

/// The values a field may take.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Domain {
    /// Any value of the field's type.
    Any,
    /// An integer within bounds (`None`: unbounded on that side).
    Int {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<i64>,
    },
    /// A number within bounds; `exclusive_min`: `min` itself is outside.
    Number {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<f64>,
        #[serde(default, skip_serializing_if = "is_false")]
        exclusive_min: bool,
    },
    /// One of these values.
    OneOf { values: Vec<Value> },
    /// A frame count `offset + step * k` within `min..=max` (the model's
    /// [`FrameGrid`]).
    Frames {
        step: u32,
        offset: u32,
        min: u32,
        max: u32,
    },
    /// A string of `min_chars..=max_chars` characters; `non_blank`: not
    /// only whitespace.
    Text {
        min_chars: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_chars: Option<usize>,
        #[serde(default, skip_serializing_if = "is_false")]
        non_blank: bool,
    },
    /// A list of at most `max` items.
    Items { max: u32 },
    /// An output canvas (`WxH`, or the width / height of one) the model's
    /// [`CanvasCaps`] admit: sides on `multiple` (or padded to it when
    /// `pad_and_crop`), `width / height` within `aspect`, at most
    /// `max_area` pixels at the largest tier.
    Canvas {
        multiple: u32,
        max_area: u64,
        aspect: (f32, f32),
        short_edges: Vec<u32>,
        pad_and_crop: bool,
    },
    /// A `W:H` aspect ratio with `min <= W/H <= max`.
    Aspect { min: f32, max: f32 },
    /// A media URL (https, data URI or the surface's upload scheme).
    Media { media: MediaField },
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Whether the model honours a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Support {
    /// Validated and applied.
    Honoured,
    /// Validated and accepted; no effect on this server (noted in the job
    /// log where the surface does so).
    NoOp,
    /// Refused with `InvalidRequest` naming the field (design §3.2 rule 7:
    /// refuse, never drop).
    Refused,
    /// Refused as an engine gap: `Unsupported(gap)`.
    Gap { gap: GapId },
}

/// Whether unknown fields are refused or ignored by a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownFields {
    /// `deny_unknown_fields` / pydantic `extra="forbid"`: 400.
    Reject,
    /// Ignored, as the upstream schema does not forbid them.
    Ignore,
}

/// One request field of one endpoint, for one model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Param {
    /// The wire name; nested fields are dotted (`video_params.fps`,
    /// `extra.prompt_expansion_mode`).
    pub name: String,
    #[serde(rename = "type")]
    pub ty: ParamType,
    pub required: bool,
    /// JSON `null` reads as absent.
    pub nullable: bool,
    /// The value an omitted field takes (`None`: no default, or the
    /// model's own, see `note`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// What this server accepts for this model.
    pub domain: Domain,
    /// The wider domain the surface's parser enforces, when the model
    /// narrows it: a value inside `wire` but outside `domain` passes parsing
    /// and is refused by negotiation against the model's caps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wire: Option<Domain>,
    pub support: Support,
    /// The internal field it fills (`sampling.steps`, `timing.length`,
    /// `canvas`, ...).
    pub maps_to: String,
    /// The upstream contract fixing the field, under `docs/serve/specs/`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Param {
    /// An optional, honoured field of type `ty` with any value.
    pub fn new(name: impl Into<String>, ty: ParamType, maps_to: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty,
            required: false,
            nullable: false,
            default: None,
            domain: Domain::Any,
            wire: None,
            support: Support::Honoured,
            maps_to: maps_to.into(),
            upstream: None,
            note: None,
        }
    }
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }
    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }
    pub fn with_default(mut self, v: impl Into<Value>) -> Self {
        self.default = Some(v.into());
        self
    }
    pub fn domain(mut self, d: Domain) -> Self {
        self.domain = d;
        self
    }
    /// The parser's wider domain; dropped when equal to the domain.
    pub fn wire(mut self, d: Domain) -> Self {
        self.wire = (d != self.domain).then_some(d);
        self
    }
    pub fn support(mut self, s: Support) -> Self {
        self.support = s;
        self
    }
    pub fn upstream(mut self, s: impl Into<String>) -> Self {
        self.upstream = Some(s.into());
        self
    }
    pub fn note(mut self, s: impl Into<String>) -> Self {
        self.note = Some(s.into());
        self
    }
    /// The domain the parser enforces ([`Param::wire`], else the domain).
    pub fn parse_domain(&self) -> &Domain {
        self.wire.as_ref().unwrap_or(&self.domain)
    }
}

/// Every parameter of one submit endpoint of one surface, for one model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamSpec {
    /// The API (`openai_videos`, `fastwan`, `minimax`, `ltx`, `fal`,
    /// `native`).
    pub surface: String,
    /// `METHOD path` (`POST /v1/videos`; fal: the endpoint id).
    pub endpoint: String,
    /// The model name a client sends on this surface (the engine id where
    /// the surface has no model names of its own).
    pub model: String,
    /// The engine model (`ModelCaps::id`) the name resolves to here.
    pub engine_model: String,
    pub unknown_fields: UnknownFields,
    pub params: Vec<Param>,
}

impl ParamSpec {
    pub fn new(
        surface: impl Into<String>,
        endpoint: impl Into<String>,
        model: impl Into<String>,
        caps: &ModelCaps,
        unknown_fields: UnknownFields,
    ) -> Self {
        Self {
            surface: surface.into(),
            endpoint: endpoint.into(),
            model: model.into(),
            engine_model: caps.id.0.clone(),
            unknown_fields,
            params: Vec::new(),
        }
    }
    pub fn push(&mut self, p: Param) -> &mut Self {
        self.params.push(p);
        self
    }
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }
}

// ---- domains from bounds and caps ------------------------------------------------

impl Domain {
    pub fn int(r: IntRange) -> Self {
        Domain::Int {
            min: Some(r.min),
            max: Some(r.max),
        }
    }
    /// An integer of at least `min`.
    pub fn int_min(min: i64) -> Self {
        Domain::Int {
            min: Some(min),
            max: None,
        }
    }
    pub fn num(r: NumRange) -> Self {
        Domain::Number {
            min: Some(r.min),
            max: Some(r.max),
            exclusive_min: false,
        }
    }
    /// A number above `min` (exclusive).
    pub fn num_above(min: f64) -> Self {
        Domain::Number {
            min: Some(min),
            max: None,
            exclusive_min: true,
        }
    }
    /// A number of at least `min`.
    pub fn num_min(min: f64) -> Self {
        Domain::Number {
            min: Some(min),
            max: None,
            exclusive_min: false,
        }
    }
    pub fn one_of<T: Into<Value>>(values: impl IntoIterator<Item = T>) -> Self {
        Domain::OneOf {
            values: values.into_iter().map(Into::into).collect(),
        }
    }
    pub fn strings(values: &[&str]) -> Self {
        Self::one_of(values.iter().copied())
    }
    /// The model's frame grid.
    pub fn frames(g: &FrameGrid) -> Self {
        Domain::Frames {
            step: g.step,
            offset: g.offset,
            min: g.min,
            max: g.max,
        }
    }
    /// The model's canvas.
    pub fn canvas(c: &CanvasCaps) -> Self {
        Domain::Canvas {
            multiple: c.multiple,
            max_area: c.max_area,
            aspect: c.aspect,
            short_edges: c.short_edges.clone(),
            pad_and_crop: c.pad_and_crop,
        }
    }
    /// The model's aspect range.
    pub fn aspect(c: &CanvasCaps) -> Self {
        Domain::Aspect {
            min: c.aspect.0,
            max: c.aspect.1,
        }
    }
    /// A string of 1 to `max` characters.
    pub fn chars(max: usize) -> Self {
        Domain::Text {
            min_chars: 1,
            max_chars: Some(max),
            non_blank: false,
        }
    }
    /// A string with some non-whitespace character, at most `max` long.
    pub fn non_blank(max: Option<usize>) -> Self {
        Domain::Text {
            min_chars: 1,
            max_chars: max,
            non_blank: true,
        }
    }
    pub fn items(max: u32) -> Self {
        Domain::Items { max }
    }
    pub fn media(m: MediaField) -> Self {
        Domain::Media { media: m }
    }

    /// Whether `v` lies in the domain. `Canvas` checks a `WxH` string or a
    /// side; `Aspect` a `W:H` string; `Media` a non-blank string (the
    /// scheme rules are the surface's own).
    pub fn contains(&self, v: &Value) -> bool {
        match self {
            Domain::Any => true,
            Domain::Int { min, max } => {
                as_int(v).is_some_and(|i| min.is_none_or(|m| i >= m) && max.is_none_or(|m| i <= m))
            }
            Domain::Number {
                min,
                max,
                exclusive_min,
            } => v.as_f64().is_some_and(|f| {
                let lo = min.is_none_or(|m| if *exclusive_min { f > m } else { f >= m });
                lo && max.is_none_or(|m| f <= m)
            }),
            Domain::OneOf { values } => values.iter().any(|x| json_eq(x, v)),
            Domain::Frames {
                step,
                offset,
                min,
                max,
            } => as_int(v)
                .and_then(|i| u32::try_from(i).ok())
                .is_some_and(|n| {
                    let g = FrameGrid::new(*step, *offset, *min, *max, *min);
                    g.contains(n)
                }),
            Domain::Text {
                min_chars,
                max_chars,
                non_blank,
            } => v.as_str().is_some_and(|s| {
                let n = s.chars().count();
                n >= *min_chars
                    && max_chars.is_none_or(|m| n <= m)
                    && (!non_blank || !s.trim().is_empty())
            }),
            Domain::Items { max } => v
                .as_array()
                .is_some_and(|a| u32::try_from(a.len()).is_ok_and(|n| n <= *max)),
            Domain::Canvas {
                multiple,
                max_area,
                aspect,
                ..
            } => match v {
                Value::String(s) => parse_wxh(s).is_some_and(|(w, h)| {
                    let caps = canvas_of(*multiple, *max_area, *aspect);
                    w > 0
                        && h > 0
                        && caps.aspect_ok(w, h)
                        && u64::from(w) * u64::from(h) <= *max_area
                }),
                other => as_int(other).is_some_and(|i| i > 0),
            },
            Domain::Aspect { min, max } => v
                .as_str()
                .and_then(parse_ratio)
                .is_some_and(|r| r >= f64::from(*min) - 1e-6 && r <= f64::from(*max) + 1e-6),
            Domain::Media { .. } => v.as_str().is_some_and(|s| !s.trim().is_empty()),
        }
    }

    /// A value inside the domain (the lower bound, the first allowed value,
    /// the default canvas), for certification. `None` when the domain is
    /// empty or has no natural representative ([`Domain::Any`]).
    pub fn valid_example(&self) -> Option<Value> {
        match self {
            Domain::Any => None,
            Domain::Int { min, max } => Some(Value::from(min.or(*max).unwrap_or(1))),
            Domain::Number {
                min,
                max,
                exclusive_min,
            } => {
                let v = match (min, max, exclusive_min) {
                    (Some(lo), Some(hi), true) => (lo + hi) / 2.0,
                    (Some(lo), None, true) => lo + 1.0,
                    (Some(lo), _, false) => *lo,
                    (None, Some(hi), _) => *hi,
                    (None, None, _) => 1.0,
                };
                serde_json::Number::from_f64(v).map(Value::Number)
            }
            Domain::OneOf { values } => values.first().cloned(),
            Domain::Frames { min, .. } => Some(Value::from(*min)),
            Domain::Text { min_chars, .. } => Some(Value::from("a".repeat((*min_chars).max(1)))),
            Domain::Items { .. } => Some(Value::Array(Vec::new())),
            Domain::Canvas {
                multiple,
                max_area,
                aspect,
                short_edges,
                ..
            } => {
                let c = CanvasCaps {
                    short_edges: short_edges.clone(),
                    ..canvas_of(*multiple, *max_area, *aspect)
                };
                let short = short_edges.first().copied()?;
                let (w, h) = crate::negotiate::canvas_for_aspect(&c, 16.0 / 9.0, short);
                Some(Value::from(format!("{w}x{h}")))
            }
            Domain::Aspect { min, max } => Some(Value::from(
                if f64::from(*min) <= 16.0 / 9.0 && 16.0 / 9.0 <= f64::from(*max) {
                    "16:9"
                } else {
                    "1:1"
                },
            )),
            Domain::Media { media } => Some(Value::from(match media {
                MediaField::Image => "https://example.com/image.png",
                MediaField::Video => "https://example.com/video.mp4",
                MediaField::Audio => "https://example.com/audio.mp3",
            })),
        }
    }

    /// Values outside the domain (just past each bound, a value not in the
    /// set, a value off the grid, too long a string or list), for
    /// certification. Type errors are left to the caller.
    pub fn invalid_examples(&self) -> Vec<Value> {
        let mut out = Vec::new();
        match self {
            Domain::Any | Domain::Media { .. } => {}
            Domain::Int { min, max } => {
                if let Some(m) = min {
                    out.push(Value::from(m - 1));
                }
                if let Some(m) = max {
                    out.push(Value::from(m.saturating_add(1)));
                }
            }
            Domain::Number {
                min,
                max,
                exclusive_min,
            } => {
                let num = |f: f64| serde_json::Number::from_f64(f).map(Value::Number);
                if let Some(m) = min {
                    out.extend(num(if *exclusive_min { *m } else { m - 1.0 }));
                }
                if let Some(m) = max {
                    out.extend(num(m + 1.0));
                }
            }
            Domain::OneOf { values } => {
                let probe = if values.iter().all(Value::is_number) {
                    let top = values.iter().filter_map(Value::as_i64).max().unwrap_or(0);
                    Value::from(top + 1)
                } else {
                    Value::from("__not_a_listed_value__")
                };
                if !values.contains(&probe) {
                    out.push(probe);
                }
            }
            Domain::Frames {
                step,
                offset,
                min,
                max,
            } => {
                if *step > 1 {
                    out.push(Value::from(min + 1));
                }
                let g = FrameGrid::new(*step, *offset, *min, *max, *min);
                if let Some(n) = g.next_on_grid(max + 1) {
                    out.push(Value::from(n));
                }
            }
            Domain::Text {
                min_chars,
                max_chars,
                non_blank,
            } => {
                if *non_blank {
                    out.push(Value::from(" "));
                } else if *min_chars > 0 {
                    out.push(Value::from(""));
                }
                if let Some(m) = max_chars {
                    out.push(Value::from("a".repeat(m + 1)));
                }
            }
            Domain::Items { max } => {
                let n = usize::try_from(*max).unwrap_or(usize::MAX - 1) + 1;
                out.push(Value::Array(vec![Value::from("https://example.com/x"); n]));
            }
            Domain::Canvas { max_area, .. } => {
                let side = ((*max_area as f64).sqrt() as u64 + 1) * 2;
                out.push(Value::from(format!("{side}x{side}")));
            }
            Domain::Aspect { max, .. } => {
                let w = (f64::from(*max) * 2.0).ceil() as u64 + 1;
                out.push(Value::from(format!("{w}:1")));
            }
        }
        out
    }
}

fn canvas_of(multiple: u32, max_area: u64, aspect: (f32, f32)) -> CanvasCaps {
    CanvasCaps {
        multiple,
        max_area,
        aspect,
        short_edges: Vec::new(),
        pad_and_crop: false,
        hd: None,
    }
}

/// An integer (an integral float such as `8.0` counts).
fn as_int(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| {
        v.as_f64()
            .filter(|f| f.is_finite() && f.fract() == 0.0 && f.abs() < 9.0e15)
            .map(|f| f as i64)
    })
}

/// Numbers compare by value (`24` equals `24.0`); everything else by JSON.
fn json_eq(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

fn parse_wxh(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.split_once('x')?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

fn parse_ratio(s: &str) -> Option<f64> {
    let (w, h) = s.split_once(':')?;
    let (w, h): (u32, u32) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    (w > 0 && h > 0).then(|| f64::from(w) / f64::from(h))
}

// ---- caps-derived pieces every surface uses --------------------------------------

/// The whole-second lengths that resolve on `grid` at `fps` when aligned up
/// to the grid (`Length::Seconds { snap: AlignUp }`: `/v1/videos`
/// `seconds`, native `seconds`, MiniMax `duration`). `None` when no whole
/// second does.
pub fn seconds_aligned_up(grid: &FrameGrid, fps: u32) -> Option<IntRange> {
    if fps == 0 {
        return None;
    }
    let top = grid.max / fps + 1;
    let ok: Vec<u32> = (1..=top)
        .filter(|s| s.checked_mul(fps).and_then(|n| grid.align_up(n)).is_some())
        .collect();
    Some(IntRange::new(
        i64::from(*ok.first()?),
        i64::from(*ok.last()?),
    ))
}

/// The whole-second range a surface rule `rule` leaves on the model:
/// `rule` intersected with [`seconds_aligned_up`] at the default rate.
pub fn seconds_on_model(rule: Option<IntRange>, caps: &ModelCaps) -> Option<IntRange> {
    let m = seconds_aligned_up(&caps.frames, caps.fps.default)?;
    let r = match rule {
        Some(r) => IntRange::new(r.min.max(m.min), r.max.min(m.max)),
        None => m,
    };
    (r.min <= r.max).then_some(r)
}

/// `Domain::Int` over [`seconds_on_model`] (`Domain::Int` with no bounds
/// when nothing fits: the model cannot serve whole seconds).
pub fn seconds_domain(rule: Option<IntRange>, caps: &ModelCaps) -> Domain {
    match seconds_on_model(rule, caps) {
        Some(r) => Domain::int(r),
        None => Domain::OneOf { values: Vec::new() },
    }
}

/// The model's rates as a set.
pub fn fps_domain(caps: &ModelCaps) -> Domain {
    Domain::one_of(caps.fps.allowed.iter().copied())
}

/// The canvas tiers as a set.
pub fn short_edge_domain(caps: &ModelCaps) -> Domain {
    Domain::one_of(caps.canvas.short_edges.iter().copied())
}

/// [`Support::Honoured`] when `honoured`, else `refused`.
pub fn knob(honoured: bool, refused: Support) -> Support {
    if honoured {
        Support::Honoured
    } else {
        refused
    }
}

/// Spec-parser agreement, for an adapter's tests (and plan E3): for every
/// top-level parameter the model serves (honoured or a no-op) whose
/// parse-time domain ([`Param::parse_domain`]) is a bound, a set or a text
/// length, `parse` must accept `base` with the field set to the domain's
/// valid example and refuse each invalid example naming the field.
/// Parameters in `skip` are left out. Returns the disagreements.
pub fn parse_agreement(
    spec: &ParamSpec,
    base: &serde_json::Map<String, Value>,
    parse: &dyn Fn(Value) -> Result<(), crate::ApiError>,
    skip: &[&str],
) -> Vec<String> {
    let mut out = Vec::new();
    for p in &spec.params {
        let nested = p.name.contains(['.', '(', '[']);
        let served = matches!(p.support, Support::Honoured | Support::NoOp);
        let bounded = matches!(
            p.parse_domain(),
            Domain::Int { .. } | Domain::Number { .. } | Domain::OneOf { .. } | Domain::Text { .. }
        );
        if nested || !served || !bounded || skip.contains(&p.name.as_str()) {
            continue;
        }
        let with = |v: Value| {
            let mut b = base.clone();
            b.insert(p.name.clone(), v);
            Value::Object(b)
        };
        if let Some(v) = p.parse_domain().valid_example() {
            if let Err(e) = parse(with(v.clone())) {
                out.push(format!("{}: valid {v} refused: {}", p.name, e.message));
            }
        }
        for v in p.parse_domain().invalid_examples() {
            match parse(with(v.clone())) {
                Ok(()) => out.push(format!("{}: invalid {v} accepted", p.name)),
                Err(e) if e.param.as_deref() != Some(p.name.as_str()) => out.push(format!(
                    "{}: invalid {v} refused on {:?}: {}",
                    p.name, e.param, e.message
                )),
                Err(_) => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ranges_display_like_the_messages_they_feed() {
        assert_eq!(IntRange::new(1, 200).to_string(), "1..=200");
        assert_eq!(NumRange::new(0.0, 20.0).to_string(), "0..=20");
        assert_eq!(NumRange::new(1.0, 3.5).to_string(), "1..=3.5");
        assert!(!NumRange::new(0.0, 1.0).contains(f64::NAN));
        assert!(IntRange::new(5, 15).contains_u64(15));
        assert!(!IntRange::new(5, 15).contains_u64(u64::MAX));
    }

    #[test]
    fn examples_fall_on_the_right_side() {
        let doms = [
            Domain::int(IntRange::new(1, 200)),
            Domain::int_min(0),
            Domain::num(NumRange::new(0.0, 20.0)),
            Domain::num_above(0.0),
            Domain::one_of([24, 25]),
            Domain::strings(&["a", "b"]),
            Domain::frames(&FrameGrid::h3()),
            Domain::chars(10),
            Domain::non_blank(None),
            Domain::items(3),
            Domain::canvas(&CanvasCaps::h3()),
            Domain::aspect(&CanvasCaps::h3()),
        ];
        for d in doms {
            let v = d.valid_example().unwrap();
            assert!(d.contains(&v), "{d:?} {v}");
            let bad = d.invalid_examples();
            assert!(!bad.is_empty(), "{d:?}");
            for b in bad {
                assert!(!d.contains(&b), "{d:?} {b}");
            }
        }
        assert!(Domain::one_of([24]).contains(&json!(24.0)));
    }

    #[test]
    fn seconds_follow_the_grid() {
        // H3 17n+5 at 24 fps, 107..=362: 4..=15 s align up onto the grid.
        let g = FrameGrid::h3();
        assert_eq!(seconds_aligned_up(&g, 24), Some(IntRange::new(4, 15)));
        let caps = ModelCaps::h3("h3", false);
        assert_eq!(
            seconds_on_model(Some(IntRange::new(5, 15)), &caps),
            Some(IntRange::new(5, 15))
        );
        // LTX 8k+1 9..=481 at 24 fps: 1..=20 s.
        let l = FrameGrid::new(8, 1, 9, 481, 121);
        assert_eq!(seconds_aligned_up(&l, 24), Some(IntRange::new(1, 20)));
        assert_eq!(seconds_aligned_up(&l, 0), None);
    }

    #[test]
    fn wire_is_dropped_when_equal() {
        let p = Param::new("fps", ParamType::Integer, "timing.fps")
            .domain(Domain::one_of([24]))
            .wire(Domain::one_of([24]));
        assert!(p.wire.is_none());
        let p = p.wire(Domain::int_min(1));
        assert_eq!(p.parse_domain(), &Domain::int_min(1));
    }
}
