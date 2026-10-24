//! The certification matrix (plan E3, ADR-0016): every parameter of every
//! served `ParamSpec`, on every shipped config, driven through the real
//! surface with a valid and an invalid value, the outcomes written to
//! `docs/serve/certification.tsv`.
//!
//! - **Specs**: `GET /fv/v1/params` of every shipped config
//!   (`configs/serve/*.toml`, `tests/compat/serve.toml`) on the fake engine
//!   serving the config's models with their real caps and the CUDA
//!   pipelines' job checks, as `param_spec_golden` renders them; each
//!   distinct spec (the golden's key) is driven once, on the first config
//!   that serves it. The Reactor command tables of the configs that mount
//!   Reactor and the director's `configure` / `prompt` messages join them.
//! - **Cases** ([`plan`]): per parameter, the domain's valid example and
//!   every invalid example (`Domain::valid_example` / `invalid_examples`,
//!   of the model's domain and of the parser's wider one), media as data
//!   URIs (the server stages inputs at submit), a wrong JSON type where the
//!   domain has no invalid value, and the overrides of [`special`] for the
//!   nested and composite fields. A refused or gap parameter has no valid
//!   case: any value is its invalid case. A parameter with no invalid case
//!   carries an exemption in [`EXEMPT`] with its reason.
//! - **Drivers**: HTTP (`POST` in-process through the assembled router,
//!   the same `App::build` the e2e suites use; an accepted job is checked
//!   against the engine's job checks and cancelled), or, for the WebRTC
//!   data channels (director, Reactor), the channel's parser.
//! - **Judgement**: a valid value must be accepted; an invalid one refused
//!   with a 4xx naming the parameter (the wire's `param` / `loc` where the
//!   surface has one, else the message). Disagreements are recorded in the
//!   `finding` column, not fixed here; a 5xx or a panic fails the test.
//!
//! Tests:
//!
//! - `every_param_has_a_valid_and_an_invalid_case` (fast, no server): the
//!   plan of every parameter of `goldens/param_specs.json`, the director
//!   and every Reactor mode has a valid case (or is refused) and an
//!   invalid case, else an exemption; an exemption no parameter needs
//!   fails too.
//! - `certification_matrix_is_golden`: runs the matrix and compares the
//!   TSV with `docs/serve/certification.tsv`; `FV_CERT_WRITE=1` rewrites
//!   it (`FV_CERT_WRITE=<file>` writes there instead; list the change in
//!   the PR).

#![cfg(all(
    feature = "openai-videos",
    feature = "minimax",
    feature = "fal",
    feature = "ltxapi",
    feature = "reactor"
))]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use axum::body::Body;
use axum::http::{HeaderMap, Request};
use axum::Router;
use base64::Engine as _;
use fastvideo_engine_service::JobValidator;
use fastvideo_protocol::params::{
    Domain, MediaField, Param, ParamSpec, ParamType, Support, UnknownFields,
};
use fastvideo_protocol::{ModelCaps, ProtocolId};
use fastvideo_serve::config::EngineBackendKind;
use fastvideo_serve::{App, Overrides};
use futures::StreamExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// Keys rotate so per-key limits (MiniMax: 300 creates a minute, 30 in
/// flight) never refuse a case.
const KEYS: usize = 64;
/// Requests in flight per config.
const CONCURRENCY: usize = 16;

fn key(i: usize) -> String {
    format!("sk-cert-{i}")
}

fn tsv_path() -> PathBuf {
    common::root().join("docs/serve/certification.tsv")
}

fn golden_specs_path() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/param_specs.json")
}

// ------------------------------------------------------------------ media

fn data_uri(mime: &str, bytes: &[u8]) -> String {
    format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn png(w: u32, h: u32) -> String {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([90, 120, 150]));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    data_uri("image/png", &out.into_inner())
}

/// A silent 16-bit mono WAV of `secs` seconds at 16 kHz.
fn wav(secs: f64) -> String {
    let rate: u32 = 16_000;
    let frames = (secs.max(0.0) * f64::from(rate)) as u32;
    let data = frames * 2;
    let mut v = b"RIFF".to_vec();
    v.extend((36 + data).to_le_bytes());
    v.extend(b"WAVEfmt ");
    v.extend(16u32.to_le_bytes());
    v.extend(1u16.to_le_bytes());
    v.extend(1u16.to_le_bytes());
    v.extend(rate.to_le_bytes());
    v.extend((rate * 2).to_le_bytes());
    v.extend(2u16.to_le_bytes());
    v.extend(16u16.to_le_bytes());
    v.extend(b"data");
    v.extend(data.to_le_bytes());
    v.extend(vec![0u8; data as usize]);
    data_uri("audio/wav", &v)
}

/// A header-only MP4 at 24 fps with audio (ingestion probes it exactly).
fn mp4(w: u16, h: u16, frames: u32) -> String {
    let b =
        fastvideo_media::mp4::inspect::header_only_mp4(w, h, 24_000, 1000, frames, Some(44_100));
    data_uri("video/mp4", &b)
}

struct Media {
    image: String,
    image_portrait: String,
    audio: String,
    /// A source video for edits: 505 frames (21 s) at 24 fps, 768x512.
    video: String,
    /// A reference clip: 3 s.
    ref_video: String,
    ref_audio: String,
}

fn media() -> &'static Media {
    static M: OnceLock<Media> = OnceLock::new();
    M.get_or_init(|| Media {
        image: png(320, 180),
        image_portrait: png(180, 320),
        audio: wav(8.0),
        video: mp4(768, 512, 505),
        ref_video: mp4(768, 512, 72),
        ref_audio: wav(3.0),
    })
}

/// A short label for a value in the TSV (data URIs by kind, long strings
/// and lists by length).
fn label(v: &Value) -> String {
    let m = media();
    match v {
        Value::String(s) if s.starts_with("data:") => {
            let k = if *s == m.image {
                "image 320x180"
            } else if *s == m.image_portrait {
                "image 180x320"
            } else if *s == m.audio {
                "audio 8 s"
            } else if *s == m.video {
                "video 768x512 21 s"
            } else if *s == m.ref_video {
                "video 3 s"
            } else if *s == m.ref_audio {
                "audio 3 s"
            } else if s.starts_with("data:audio/wav") {
                return format!("<data:audio/wav {} s>", wav_secs(s));
            } else {
                "media"
            };
            format!("<data: {k}>")
        }
        Value::String(s) if s.chars().count() > 24 => {
            let c = s.chars().next().unwrap_or(' ');
            if s.chars().all(|x| x == c) {
                format!("\"{c}\"x{}", s.chars().count())
            } else {
                format!(
                    "\"{}…\"({} chars)",
                    s.chars().take(20).collect::<String>(),
                    s.chars().count()
                )
            }
        }
        Value::Array(a) if a.len() > 2 && a.iter().all(|x| x == &a[0]) => {
            format!("[{} x{}]", label(&a[0]), a.len())
        }
        Value::Array(a) => format!("[{}]", a.iter().map(label).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{k}: {}", label(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// The seconds of one of [`wav`]'s data URIs.
fn wav_secs(s: &str) -> String {
    let b64 = s.split_once(',').map(|x| x.1).unwrap_or("");
    let n = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map(|b| b.len().saturating_sub(44))
        .unwrap_or(0);
    let secs = n as f64 / 32_000.0;
    format!("{secs}")
}

// ------------------------------------------------------------------ plans

/// Field assignments on a body (a dotted name nests; `None` removes).
type Assigns = Vec<(String, Option<Value>)>;

/// One request variation: field assignments on top of the base body (a
/// dotted name sets a nested field; `None` removes it).
#[derive(Clone, Debug)]
struct Probe {
    label: String,
    assigns: Vec<(String, Option<Value>)>,
}

impl Probe {
    fn set(name: &str, v: Value) -> Self {
        Probe {
            label: label(&v),
            assigns: vec![(name.to_owned(), Some(v))],
        }
    }
    fn labelled(label: impl Into<String>, assigns: Vec<(String, Option<Value>)>) -> Self {
        Probe {
            label: label.into(),
            assigns,
        }
    }
}

/// The valid side of a parameter.
#[derive(Clone, Debug)]
enum Valid {
    Probe(Probe),
    /// No valid value on this model (`n/a (...)` in the TSV).
    None(String),
}

/// The cases of one parameter.
#[derive(Clone, Debug)]
struct Plan {
    valid: Valid,
    /// Other valid values, tried in order when the surface refuses
    /// `valid` (a set's other members, a range's upper bound): the row
    /// records the first one accepted.
    alternates: Vec<Probe>,
    invalid: Vec<Probe>,
}

/// Parameters with no invalid case, by (surface, name): every value of the
/// field's type is accepted by design. A new parameter without an invalid
/// case fails `every_param_has_a_valid_and_an_invalid_case`; so does an
/// entry here no parameter needs.
const EXEMPT: &[(&str, &str, &str)] = &[
    (
        "minimax",
        "prompt_optimizer",
        "MiniMax V1's switch: a non-boolean is ignored, as the V1 API does (no value is refused)",
    ),
    (
        "ltx",
        "enhance_prompt",
        "ours: a non-boolean is ignored (no value is refused)",
    ),
    (
        "fal_director",
        "output",
        "ours: an output-copy object checked by the session when `[director] allow_output`, not by the parser",
    ),
    (
        "fal_director",
        "x_fv_output",
        "ours: an output-copy object checked by the session, not by the parser",
    ),
];

fn exemption(surface: &str, name: &str) -> Option<&'static str> {
    EXEMPT
        .iter()
        .find(|(s, n, _)| *s == surface && *n == name)
        .map(|(_, _, r)| *r)
}

fn type_example(ty: ParamType) -> Value {
    match ty {
        ParamType::String => json!("a fox"),
        ParamType::Integer | ParamType::IntegerOrString => json!(1),
        ParamType::Number => json!(1.5),
        ParamType::Boolean => json!(true),
        ParamType::Object => json!({}),
        ParamType::Array => json!([]),
        ParamType::Any => json!("a"),
    }
}

fn wrong_type(ty: ParamType) -> Option<Value> {
    match ty {
        ParamType::String => Some(json!(7)),
        ParamType::Integer | ParamType::Number => Some(json!("x")),
        ParamType::Boolean | ParamType::Object | ParamType::Array => Some(json!("x")),
        ParamType::IntegerOrString => Some(json!(true)),
        ParamType::Any => None,
    }
}

fn media_of(m: MediaField) -> Value {
    let md = media();
    json!(match m {
        MediaField::Image => &md.image,
        MediaField::Video => &md.video,
        MediaField::Audio => &md.audio,
    })
}

/// The media an item of a list parameter carries, from its name.
fn item_kind(name: &str) -> MediaField {
    if name.contains("video") {
        MediaField::Video
    } else if name.contains("audio") {
        MediaField::Audio
    } else {
        MediaField::Image
    }
}

/// One item of a list parameter on `surface`.
fn item(surface: &str, name: &str) -> Value {
    let md = media();
    let kind = item_kind(name);
    let uri = match kind {
        MediaField::Image => md.image.clone(),
        MediaField::Video => md.ref_video.clone(),
        MediaField::Audio => md.ref_audio.clone(),
    };
    if surface == "openai_videos" {
        let k = match kind {
            MediaField::Image => "image_url",
            MediaField::Video => "video_url",
            MediaField::Audio => "audio_url",
        };
        json!({ k: uri })
    } else {
        json!(uri)
    }
}

/// `(width, height)` of a `WxH` value.
fn wxh(v: &Value) -> Option<(u64, u64)> {
    let (w, h) = v.as_str()?.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// The partner of a canvas side (`width` <-> `height`, nested alike).
fn partner(name: &str) -> Option<String> {
    if let Some(p) = name.strip_suffix("width") {
        Some(format!("{p}height"))
    } else {
        name.strip_suffix("height").map(|p| format!("{p}width"))
    }
}

/// A domain value as probes of `p` (media as data URIs, canvas sides as a
/// (width, height) pair, lists as items).
fn probes_of(surface: &str, spec: &ParamSpec, p: &Param, v: Value, valid: bool) -> Option<Probe> {
    let d = &p.domain;
    match (d, p.ty) {
        (Domain::Media { media }, _) if valid => Some(Probe::set(&p.name, media_of(*media))),
        (Domain::Canvas { .. }, ParamType::Integer) => {
            let (w, h) = wxh(&v)?;
            let other = partner(&p.name)?;
            let (mine, theirs) = if p.name.ends_with("width") {
                (w, h)
            } else {
                (h, w)
            };
            let mut assigns = vec![(p.name.clone(), Some(json!(mine)))];
            if spec.param(&other).is_some() {
                assigns.push((other.clone(), Some(json!(theirs))));
            }
            Some(Probe::labelled(
                format!("{mine} (with {other} {theirs})"),
                assigns,
            ))
        }
        (Domain::Items { max }, _) => {
            let n = if valid {
                1
            } else {
                usize::try_from(*max).ok()? + 1
            };
            let list = vec![item(surface, &p.name); n];
            let l = format!("{n} x {}", label(&list[0]));
            Some(Probe::labelled(
                l,
                vec![(p.name.clone(), Some(Value::Array(list)))],
            ))
        }
        _ => Some(Probe::set(&p.name, v)),
    }
}

/// Composite fields the generic rules cannot express.
fn special(surface: &str, spec: &ParamSpec, p: &Param) -> Option<Plan> {
    let md = media();
    let text = json!({"type": "text", "text": "a fox runs"});
    let content = |items: Vec<Value>| {
        let mut c = vec![text.clone()];
        c.extend(items);
        Probe::labelled(
            label(&Value::Array(c.clone())),
            vec![("content".to_owned(), Some(Value::Array(c)))],
        )
    };
    let img_item =
        |role: &str| json!({"type": "image_url", "image_url": {"url": md.image}, "role": role});
    let plan = |valid: Probe, invalid: Vec<Probe>| Plan {
        valid: Valid::Probe(valid),
        alternates: Vec::new(),
        invalid,
    };
    match (surface, p.name.as_str()) {
        ("minimax", "content") => Some(plan(
            content(vec![]),
            vec![Probe::set("content", json!("x"))],
        )),
        ("minimax", "content[].type") => Some(plan(
            content(vec![]),
            vec![Probe::labelled(
                "[{type: \"__not_a_listed_value__\"}]",
                vec![(
                    "content".into(),
                    Some(json!([{"type": "__not_a_listed_value__", "text": "a fox runs"}])),
                )],
            )],
        )),
        ("minimax", "content[].role") => Some(plan(
            content(vec![img_item("first_frame")]),
            vec![content(vec![img_item("__not_a_listed_value__")])],
        )),
        ("minimax", "content[].text") => {
            let max = match &p.domain {
                Domain::Text { max_chars, .. } => *max_chars,
                _ => None,
            };
            let with = |t: Value| {
                Probe::labelled(
                    format!("text {}", label(&t)),
                    vec![("content".into(), Some(json!([{"type": "text", "text": t}])))],
                )
            };
            let mut bad = vec![with(json!(" "))];
            if let Some(m) = max {
                bad.push(with(json!("a".repeat(m + 1))));
            }
            Some(plan(with(json!("a fox runs")), bad))
        }
        ("minimax", "content[].image_url.url") => Some(plan(
            content(vec![img_item("first_frame")]),
            vec![content(vec![
                json!({"type": "image_url", "image_url": {"url": "not-a-url"}, "role": "first_frame"}),
            ])],
        )),
        ("minimax", n @ ("content[].video_url.url" | "content[].audio_url.url")) => {
            let (ty, role, uri) = if n.contains("video") {
                ("video_url", "reference_video", &md.ref_video)
            } else {
                ("audio_url", "reference_audio", &md.ref_audio)
            };
            let it = |u: &str| json!({"type": ty, ty: {"url": u}, "role": role});
            Some(plan(
                content(vec![it(uri)]),
                vec![content(vec![it("not-a-url")])],
            ))
        }
        ("minimax", n) if n.starts_with("content(") => {
            let Domain::Items { max } = p.domain else {
                return None;
            };
            let role = n.trim_start_matches("content(").trim_end_matches(')');
            let one = |role: &str| match role {
                "reference_video" => {
                    json!({"type": "video_url", "video_url": {"url": md.ref_video}, "role": "reference_video"})
                }
                "reference_audio" => {
                    json!({"type": "audio_url", "audio_url": {"url": md.ref_audio}, "role": "reference_audio"})
                }
                _ => img_item("reference_image"),
            };
            let n = usize::try_from(max).ok()? + 1;
            let many: Vec<Value> = if role == "references" {
                // Over the total: images first (each kind within its own cap
                // as far as the total allows).
                (0..n)
                    .map(|i| match i % 3 {
                        0 => one("reference_image"),
                        1 => one("reference_video"),
                        _ => one("reference_audio"),
                    })
                    .collect()
            } else {
                vec![one(role); n]
            };
            let valid = if role == "references" {
                one("reference_image")
            } else {
                one(role)
            };
            Some(plan(
                Probe::labelled(
                    format!("1 {role} item"),
                    vec![("content".into(), Some(json!([text, valid])))],
                ),
                vec![Probe::labelled(
                    format!("{n} {role} items"),
                    vec![(
                        "content".into(),
                        Some({
                            let mut c = vec![text.clone()];
                            c.extend(many);
                            Value::Array(c)
                        }),
                    )],
                )],
            ))
        }
        ("ltx", "audio_uri(duration)") => {
            let Domain::Number {
                min: Some(lo),
                max: Some(hi),
                ..
            } = p.domain
            else {
                return None;
            };
            let a = |s: f64| {
                Probe::labelled(
                    format!("audio {s} s"),
                    vec![("audio_uri".into(), Some(json!(wav(s))))],
                )
            };
            Some(Plan {
                alternates: vec![a(hi)],
                ..plan(
                    a(lo.max(a2v_audio_min_s())),
                    vec![a((lo - 1.0).max(0.5)), a(hi + 1.0)],
                )
            })
        }
        // The end of a retake window (with `start_s` 0 from the
        // companions): greater than `start_s`, a 2 to 20 s window.
        ("native", "end_s") => Some(plan(
            Probe::set("end_s", json!(4.0)),
            vec![
                Probe::set("end_s", json!(0.0)),
                Probe::set("end_s", json!(30.0)),
                Probe::labelled(
                    "\"x\" (wrong type)",
                    vec![("end_s".into(), Some(json!("x")))],
                ),
            ],
        )),
        ("minimax", "callback_url") => Some(plan(
            Probe::set(
                "callback_url",
                json!("https://example.com/fv-cert-callback"),
            ),
            vec![
                Probe::labelled(
                    "7 (wrong type)",
                    vec![("callback_url".into(), Some(json!(7)))],
                ),
                Probe::set("callback_url", json!("not-a-url")),
            ],
        )),
        ("ltx" | "native" | "openai_videos" | "minimax", "model") => {
            let mut bad = vec![Probe::set("model", json!("fv-cert-no-such-model"))];
            for v in p.domain.invalid_examples() {
                if !bad.iter().any(|b| b.label == label(&v)) {
                    bad.push(Probe::set("model", v));
                }
            }
            Some(plan(Probe::set("model", json!(spec.model)), bad))
        }
        ("fal_director", "script") => {
            let beat = json!({"offset": 0, "prompt": "a fox runs"});
            let n = rules_script_max() + 1;
            Some(plan(
                Probe::set("script", json!([beat])),
                vec![
                    Probe::set("script", json!([])),
                    Probe::labelled(
                        format!("{n} beats"),
                        vec![("script".into(), Some(Value::Array(vec![beat; n])))],
                    ),
                ],
            ))
        }
        ("fal_director", "entities") => {
            let e = |i: usize| json!({"id": format!("e{i}"), "kind": "character", "name": "Fox", "descriptor": "a red fox"});
            let n = rules_entities_max() + 1;
            Some(plan(
                Probe::set("entities", json!([e(0)])),
                vec![
                    Probe::labelled(
                        format!("{n} entities"),
                        vec![(
                            "entities".into(),
                            Some(Value::Array((0..n).map(e).collect())),
                        )],
                    ),
                    Probe::labelled(
                        "2 entities, one id",
                        vec![("entities".into(), Some(json!([e(0), e(0)])))],
                    ),
                ],
            ))
        }
        _ => None,
    }
}

fn a2v_audio_min_s() -> f64 {
    fastvideo_protocol::A2V_AUDIO_MIN_S
}

fn rules_script_max() -> usize {
    fastvideo_protocol::params::rules::fal_director::SCRIPT_MAX_BEATS
}

fn rules_entities_max() -> usize {
    fastvideo_protocol::params::rules::fal_director::ENTITIES_MAX
}

/// The cases of `p` on `spec` (pure: the coverage test plans the golden's
/// specs without a server).
fn plan(spec: &ParamSpec, p: &Param) -> Plan {
    let surface = spec.surface.as_str();
    let refused = matches!(p.support, Support::Refused | Support::Gap { .. });
    let why = match p.support {
        Support::Gap { gap } => format!("n/a (engine gap `{}`)", gap.code()),
        _ => "n/a (refused parameter)".to_owned(),
    };
    if let Some(pl) = special(surface, spec, p) {
        return match (refused, pl.valid) {
            // Refused: the valid value is the invalid case.
            (true, Valid::Probe(pr)) => Plan {
                valid: Valid::None(why),
                alternates: Vec::new(),
                invalid: vec![pr],
            },
            (_, valid) => Plan {
                valid,
                alternates: pl.alternates,
                invalid: pl.invalid,
            },
        };
    }
    let example = p
        .domain
        .valid_example()
        .or_else(|| p.parse_domain().valid_example())
        .unwrap_or_else(|| type_example(p.ty));
    if refused {
        let v = match (&p.domain, p.ty) {
            (Domain::Media { media }, _) => media_of(*media),
            // `{}` reads as absent on some surfaces: a value with a key.
            (_, ParamType::Object) => json!({"fv_cert": 1}),
            _ => example.clone(),
        };
        let probe =
            probes_of(surface, spec, p, v.clone(), true).unwrap_or_else(|| Probe::set(&p.name, v));
        return Plan {
            valid: Valid::None(why),
            alternates: Vec::new(),
            invalid: vec![probe],
        };
    }
    let empty_domain = matches!(&p.domain, Domain::OneOf { values } if values.is_empty());
    let valid = if empty_domain {
        Valid::None("n/a (empty domain: no value fits this model)".into())
    } else {
        match probes_of(surface, spec, p, example.clone(), true) {
            Some(pr) => Valid::Probe(pr),
            None => Valid::Probe(Probe::set(&p.name, example)),
        }
    };
    let mut invalid: Vec<Probe> = Vec::new();
    fn push(invalid: &mut Vec<Probe>, pr: Probe) {
        if !invalid.iter().any(|x| x.label == pr.label) {
            invalid.push(pr);
        }
    }
    let mut values = p.domain.invalid_examples();
    if p.wire.is_some() {
        values.extend(p.parse_domain().invalid_examples());
    }
    for v in values {
        if let Some(pr) = probes_of(surface, spec, p, v, false) {
            push(&mut invalid, pr);
        }
    }
    if let Domain::Media { .. } = p.domain {
        let bad = if surface == "fal_director" {
            json!("")
        } else {
            json!("not-a-url")
        };
        push(&mut invalid, Probe::set(&p.name, bad));
    }
    if invalid.is_empty() && exemption(surface, &p.name).is_none() {
        if let Some(v) = wrong_type(p.ty) {
            push(
                &mut invalid,
                Probe::labelled(
                    format!("{} (wrong type)", label(&v)),
                    vec![(p.name.clone(), Some(v))],
                ),
            );
        }
    }
    // Other valid values: the rest of a set, a range's upper bound.
    let others: Vec<Value> = match &p.domain {
        Domain::OneOf { values } => values.iter().skip(1).take(8).cloned().collect(),
        Domain::Int { max: Some(m), .. } => vec![json!(m)],
        Domain::Number { max: Some(m), .. } => vec![json!(m)],
        Domain::Frames { max, .. } => vec![json!(max)],
        _ => Vec::new(),
    };
    let alternates = others
        .into_iter()
        .filter_map(|v| probes_of(surface, spec, p, v, true))
        .collect();
    Plan {
        valid,
        alternates,
        invalid,
    }
}

/// Fields a parameter needs beside it to be read at all (an edit field
/// needs its source video, an aspect its tier), applied before the
/// parameter's own value when the parameter is served.
fn companions(spec: &ParamSpec, name: &str) -> Vec<(String, Option<Value>)> {
    let md = media();
    let has = |n: &str| spec.param(n).is_some();
    let first = |n: &str| {
        spec.param(n)
            .and_then(|p| p.domain.valid_example())
            .unwrap_or(Value::Null)
    };
    let retake = || {
        vec![
            ("video_url".to_owned(), Some(json!(md.video))),
            ("start_s".to_owned(), Some(json!(0))),
            ("end_s".to_owned(), Some(json!(4))),
        ]
    };
    match (spec.surface.as_str(), name) {
        ("native" | "openai_videos", "aspect_ratio") if has("short_edge") => {
            vec![("short_edge".into(), Some(first("short_edge")))]
        }
        ("native" | "openai_videos", "short_edge") if has("aspect_ratio") => {
            vec![("aspect_ratio".into(), Some(json!("16:9")))]
        }
        ("native", "start_s" | "end_s" | "retake_mode" | "video_url") => retake(),
        ("native", "extend_s" | "extend_at" | "context_s") => vec![
            ("video_url".to_owned(), Some(json!(md.video))),
            ("extend_s".to_owned(), Some(json!(4))),
        ],
        ("openai_videos", "input_reference" | "reference_url") => {
            vec![("image_reference".into(), None), ("task".into(), None)]
        }
        ("openai_videos", "audio_reference") => {
            vec![(
                "image_reference".into(),
                Some(json!([{"image_url": md.image}])),
            )]
        }
        ("ltx", "last_frame_uri") if has("audio_uri") => {
            vec![("image_uri".into(), Some(json!(md.image)))]
        }
        _ => Vec::new(),
    }
}

// ------------------------------------------------------------------ drivers

/// What one request (or one control message) observed.
#[derive(Clone, Debug, Default)]
struct Obs {
    /// HTTP status (`0`: a parse driver).
    status: u16,
    accepted: bool,
    /// The surface's error code / type (`-` when accepted).
    code: String,
    /// The parameter the wire names (`None`: the surface's error has no
    /// such field).
    param: Option<String>,
    message: String,
    /// Accepted, but the engine's job checks refuse the resolved job.
    engine: Option<String>,
    /// 2xx without a job.
    odd: Option<String>,
    /// The engine model an accepted job resolved to.
    model: Option<String>,
}

#[derive(Clone)]
enum Driver {
    Http {
        uri: String,
        auth: &'static str,
        protocol: ProtocolId,
        id_field: &'static str,
    },
    Director,
    Reactor(fastvideo_reactor::commands::CommandTable, String),
}

impl Driver {
    fn name(&self) -> &'static str {
        match self {
            Driver::Http { .. } => "http",
            Driver::Director => "parse (director data channel)",
            Driver::Reactor(..) => "parse (Reactor data channel)",
        }
    }
}

fn driver_of(spec: &ParamSpec) -> Option<Driver> {
    let ep = spec.endpoint.as_str();
    Some(match spec.surface.as_str() {
        "native" => Driver::Http {
            uri: "/fv/v1/jobs".into(),
            auth: "Bearer",
            protocol: ProtocolId::Native,
            id_field: "id",
        },
        "openai_videos" => Driver::Http {
            uri: "/v1/videos".into(),
            auth: "Bearer",
            protocol: ProtocolId::OpenAiVideos,
            id_field: "id",
        },
        "fastwan" => Driver::Http {
            uri: "/generate".into(),
            auth: "Bearer",
            protocol: ProtocolId::FastWan,
            id_field: "prompt_id",
        },
        "minimax" => Driver::Http {
            uri: "/v2/video_generation".into(),
            auth: "Bearer",
            protocol: ProtocolId::MiniMaxV2,
            id_field: "task_id",
        },
        "ltx" => Driver::Http {
            uri: ep.strip_prefix("POST ")?.to_owned(),
            auth: "Bearer",
            protocol: ProtocolId::LtxV2,
            id_field: "id",
        },
        "fal" => Driver::Http {
            uri: format!("/{ep}"),
            auth: "Key",
            protocol: ProtocolId::Fal,
            id_field: "request_id",
        },
        _ => return None,
    })
}

struct Server {
    app: App,
    validator: Option<JobValidator>,
    next_key: std::sync::atomic::AtomicUsize,
    /// The server's state dir (staged inputs): removed after the app.
    _dir: tempfile::TempDir,
}

impl Server {
    fn key(&self) -> String {
        key(self
            .next_key
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % KEYS)
    }
}

async fn post(router: &Router, uri: &str, auth: &str, body: &Value) -> (u16, HeaderMap, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .header("authorization", auth)
        .body(Body::from(body.to_string()))
        .unwrap();
    let r = router.clone().oneshot(req).await.unwrap();
    let (status, headers) = (r.status().as_u16(), r.headers().clone());
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 26).await.unwrap();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, headers, v)
}

/// The surface's error code, wire parameter and message.
fn read_error(surface: &str, headers: &HeaderMap, v: &Value) -> (String, Option<String>, String) {
    let s = |x: &Value| x.as_str().map(str::to_owned);
    match surface {
        "native" => {
            let kind = &v["error"]["kind"];
            let code = match kind {
                Value::String(k) => k.clone(),
                Value::Object(o) => o
                    .iter()
                    .map(|(k, g)| format!("{k}:{}", g.as_str().unwrap_or("?")))
                    .collect::<Vec<_>>()
                    .join(","),
                other => other.to_string(),
            };
            let param = Some(s(&v["error"]["param"]).unwrap_or_else(|| "none".into()));
            (code, param, s(&v["error"]["message"]).unwrap_or_default())
        }
        "openai_videos" => (
            s(&v["error"]["type"]).unwrap_or_else(|| "?".into()),
            Some(s(&v["error"]["param"]).unwrap_or_else(|| "none".into())),
            s(&v["error"]["message"]).unwrap_or_default(),
        ),
        "fastwan" => (
            "n/a (FastAPI {detail})".into(),
            None,
            s(&v["detail"]).unwrap_or_else(|| v.to_string()),
        ),
        "minimax" | "ltx" => (
            s(&v["error"]["type"]).unwrap_or_else(|| "?".into()),
            None,
            s(&v["error"]["message"]).unwrap_or_else(|| v.to_string()),
        ),
        "fal" => {
            let code = headers
                .get("x-fal-error-type")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("?")
                .to_owned();
            match &v["detail"] {
                Value::Array(a) => {
                    let first = a.first().cloned().unwrap_or(Value::Null);
                    let loc = first["loc"]
                        .as_array()
                        .and_then(|l| l.last())
                        .map(|x| {
                            x.as_str()
                                .map(str::to_owned)
                                .unwrap_or_else(|| x.to_string())
                        })
                        .unwrap_or_else(|| "none".into());
                    (code, Some(loc), s(&first["msg"]).unwrap_or_default())
                }
                d => (
                    code,
                    Some("none".into()),
                    s(d).unwrap_or_else(|| d.to_string()),
                ),
            }
        }
        _ => ("?".into(), None, v.to_string()),
    }
}

/// The field a director parse error names: its own checks name it first
/// (`` `memory` must be ... ``); serde's (`` invalid `configure` message:
/// invalid type ... ``) name none, except an unknown field.
fn director_param(msg: &str) -> String {
    if let Some(rest) = msg.strip_prefix("invalid `") {
        let rest = rest.split_once("` message: ").map_or(rest, |x| x.1);
        return match rest.split_once("unknown field `") {
            Some((_, f)) => f.split('`').next().unwrap_or("none").to_owned(),
            None => "none".into(),
        };
    }
    backticked(msg).unwrap_or_else(|| "none".into())
}

/// The first backticked name in a parse error (`` `memory` must be ... ``).
fn backticked(msg: &str) -> Option<String> {
    let (_, rest) = msg.split_once('`')?;
    let (name, _) = rest.split_once('`')?;
    Some(name.to_owned())
}

/// Requests (and control messages) the matrix sent.
static SENT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

async fn run(server: Option<&Server>, driver: &Driver, surface: &str, body: &Value) -> Obs {
    SENT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match driver {
        Driver::Director => match fastvideo_fal::director::messages::parse(&body.to_string()) {
            Ok(_) => Obs {
                accepted: true,
                code: "-".into(),
                ..Obs::default()
            },
            Err(e) => Obs {
                code: "invalid_message".into(),
                param: Some(director_param(&e.error)),
                message: e.error,
                ..Obs::default()
            },
        },
        Driver::Reactor(table, cmd) => {
            let data = body.as_object().cloned().unwrap_or_default();
            match table.validate(cmd, &data) {
                Ok(_) => Obs {
                    accepted: true,
                    code: "-".into(),
                    ..Obs::default()
                },
                Err(e) => Obs {
                    code: "invalid_command".into(),
                    param: Some(backticked(&e).unwrap_or_else(|| "none".into())),
                    message: e,
                    ..Obs::default()
                },
            }
        }
        Driver::Http {
            uri,
            auth,
            protocol,
            id_field,
        } => {
            let Some(server) = server else {
                return Obs::default();
            };
            let app = &server.app;
            let (status, headers, v) =
                post(&app.router, uri, &format!("{auth} {}", server.key()), body).await;
            if (200..300).contains(&status) {
                let job = match v[*id_field].as_str() {
                    Some(id) => app.ctx.jobs().by_external(*protocol, id).await,
                    None => None,
                };
                let Some(job) = job else {
                    return Obs {
                        status,
                        code: "-".into(),
                        odd: Some(format!("{status} without a job: {}", short(&v))),
                        ..Obs::default()
                    };
                };
                let engine = server
                    .validator
                    .as_ref()
                    .and_then(|val| val.check(&job.resolved).err())
                    .map(|e| e.message);
                let _ = fastvideo_serve_kit::events::cancel_job(&app.ctx, job.id).await;
                return Obs {
                    status,
                    accepted: true,
                    code: "-".into(),
                    engine,
                    model: Some(job.resolved.model.0.clone()),
                    ..Obs::default()
                };
            }
            let (code, param, message) = read_error(surface, &headers, &v);
            Obs {
                status,
                code,
                param,
                message,
                ..Obs::default()
            }
        }
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 160 {
        format!("{}…", s.chars().take(160).collect::<String>())
    } else {
        s
    }
}

/// `body` with `assigns` applied (dotted names nest).
fn apply(base: &Value, assigns: &[(String, Option<Value>)]) -> Value {
    let mut b = base.clone();
    for (k, v) in assigns {
        let parts: Vec<&str> = k.split('.').collect();
        let mut cur = &mut b;
        for part in &parts[..parts.len() - 1] {
            if !cur[*part].is_object() {
                cur[*part] = json!({});
            }
            cur = &mut cur[*part];
        }
        let last = parts[parts.len() - 1];
        match v {
            Some(v) => cur[last] = v.clone(),
            None => {
                if let Some(o) = cur.as_object_mut() {
                    o.remove(last);
                }
            }
        }
    }
    b
}

// ------------------------------------------------------------------ bases

/// The candidate base bodies of a spec, in order; the first accepted one
/// is the base every case is applied to.
fn bases(spec: &ParamSpec) -> Vec<(String, Value)> {
    let md = media();
    let m = spec.model.as_str();
    let required: Vec<(String, Option<Value>)> = spec
        .params
        .iter()
        .filter(|p| p.required && !p.name.contains(['[', '(']))
        .flat_map(|p| match plan(spec, p).valid {
            Valid::Probe(pr) => pr.assigns,
            Valid::None(_) => Vec::new(),
        })
        .collect();
    let req = apply(&json!({}), &required);
    let with = |b: &Value, extra: Value| {
        let mut b = b.clone();
        for (k, v) in extra.as_object().cloned().unwrap_or_default() {
            b[k] = v;
        }
        b
    };
    let named = |l: &str, v: Value| (l.to_owned(), v);
    match spec.surface.as_str() {
        "native" => {
            let t = json!({"model": m, "prompt": "a fox runs"});
            vec![
                named("text", t.clone()),
                named("image", with(&t, json!({"image_url": md.image}))),
                named(
                    "references",
                    with(&t, json!({"reference_urls": [md.image]})),
                ),
                named(
                    "audio+image",
                    with(&t, json!({"audio_url": md.audio, "image_url": md.image})),
                ),
                named("audio", with(&t, json!({"audio_url": md.audio}))),
                named(
                    "edit",
                    with(&t, json!({"video_url": md.video, "start_s": 0, "end_s": 4})),
                ),
            ]
        }
        "openai_videos" => {
            let t = json!({"model": m, "prompt": "a fox runs"});
            vec![
                named("text", t.clone()),
                named(
                    "ref2va",
                    with(
                        &t,
                        json!({"task": "ref2va", "image_reference": [{"image_url": md.image}]}),
                    ),
                ),
                named("image", with(&t, json!({"input_reference": md.image}))),
            ]
        }
        "fastwan" => vec![named("text", json!({"prompt": "a fox runs"}))],
        "minimax" => {
            let text = json!({"type": "text", "text": "a fox runs"});
            let t = with(
                &req,
                json!({"model": m, "ratio": "16:9", "content": [text]}),
            );
            let img = |role: &str| json!({"type": "image_url", "image_url": {"url": md.image}, "role": role});
            vec![
                named("text", t.clone()),
                named(
                    "first_frame",
                    with(&t, json!({"content": [text, img("first_frame")]})),
                ),
                named(
                    "reference_image",
                    with(&t, json!({"content": [text, img("reference_image")]})),
                ),
            ]
        }
        "ltx" => {
            // Audio-to-video takes a prompt unless it has an image.
            let t = with(&req, json!({"model": m, "prompt": "a fox runs"}));
            vec![named("required+prompt", t)]
        }
        "fal" => vec![
            named("required", req.clone()),
            named("required+image", with(&req, json!({"image_url": md.image}))),
            named(
                "required+references",
                with(&req, json!({"reference_image_urls": [md.image]})),
            ),
        ],
        "fal_director" => {
            let ty = spec.endpoint.trim_start_matches("message ");
            vec![named(ty, with(&req, json!({"type": ty})))]
        }
        "reactor" => vec![named("required", req)],
        _ => Vec::new(),
    }
}

fn keys_of(v: &Value) -> String {
    match v.as_object() {
        Some(o) if !o.is_empty() => o.keys().cloned().collect::<Vec<_>>().join(","),
        _ => "{}".into(),
    }
}

// ------------------------------------------------------------------ judging

fn leaf(name: &str) -> String {
    let base = name.split(['(', '[']).next().unwrap_or(name);
    let after_bracket = name.rsplit("].").next().unwrap_or(name);
    let n = if name.contains("[].") {
        after_bracket
    } else {
        base
    };
    n.rsplit('.').next().unwrap_or(n).to_owned()
}

/// Whether the reported `got` names `p`.
fn names(p: &str, got: &str) -> bool {
    let l = leaf(p);
    let base = p.split(['(', '[']).next().unwrap_or(p);
    got == p
        || got == l
        || got == base
        || got.starts_with(&format!("{l}["))
        || got.starts_with(&format!("{base}["))
        || got.strip_prefix(base).is_some_and(|r| r.starts_with('.'))
}

fn outcome(o: &Obs) -> String {
    if let Some(odd) = &o.odd {
        return format!("odd: {odd}");
    }
    if o.accepted {
        return match &o.engine {
            Some(e) => format!("accepted; engine refuses: {}", truncate(e, 120)),
            None => "accepted".into(),
        };
    }
    let status = if o.status == 0 {
        "refused".to_owned()
    } else {
        format!("refused {}", o.status)
    };
    let param = match &o.param {
        Some(p) => p.clone(),
        None => "n/a".into(),
    };
    format!("{status} {} param={param}", o.code)
}

/// The message of a refusal, shortened (the engine's when accepted).
fn short_msg(o: &Obs) -> String {
    let m = o.engine.as_deref().unwrap_or(&o.message);
    truncate(m, 100)
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() > n {
        format!("{}…", s.chars().take(n).collect::<String>())
    } else {
        s.to_owned()
    }
}

/// The words of a dotted / indexed name a message may use for it
/// (`content[].video_url.url`: `video_url`; `content(reference_image)`:
/// `reference_image`, `content`).
fn name_tokens(p: &str) -> Vec<String> {
    let all: Vec<String> = p
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect();
    let some: Vec<String> = all.iter().filter(|t| *t != "url").cloned().collect();
    if some.is_empty() {
        all
    } else {
        some
    }
}

/// The token of `p` the message names, if any.
fn message_names(p: &str, msg: &str) -> Option<String> {
    name_tokens(p)
        .into_iter()
        .rev()
        .find(|t| msg.contains(t.as_str()))
}

/// `n/a` for the wire `param` of a surface without one: what the message
/// says instead.
fn param_cell(p: &str, o: &Obs) -> String {
    if o.accepted || o.odd.is_some() {
        return "n/a (accepted)".into();
    }
    match &o.param {
        Some(x) => x.clone(),
        None => match message_names(p, &o.message) {
            Some(t) => format!("n/a (no param field; message names `{t}`)"),
            None => format!(
                "n/a (no param field; message: {})",
                truncate(&o.message, 60)
            ),
        },
    }
}

/// The finding for an invalid probe, if it disagrees with the spec.
fn invalid_finding(p: &Param, pr: &Probe, o: &Obs, engine_model: &str) -> Option<String> {
    if o.status >= 500 {
        return Some(format!("{}: {} (5xx)", pr.label, o.status));
    }
    if let Some(odd) = &o.odd {
        return Some(format!("{}: {odd}", pr.label));
    }
    if o.accepted {
        return Some(match &o.model {
            Some(m) if m != engine_model => {
                format!("invalid {} accepted (routed to `{m}`)", pr.label)
            }
            _ => format!("invalid {} accepted", pr.label),
        });
    }
    match &o.param {
        Some(got) if !names(&p.name, got) => Some(format!("{} reported as `{got}`", pr.label)),
        None if message_names(&p.name, &o.message).is_none() => Some(format!(
            "{}: message does not name `{}` ({})",
            pr.label,
            leaf(&p.name),
            truncate(&o.message, 80)
        )),
        _ => None,
    }
}

// ------------------------------------------------------------------ rows

const HEADER: [&str; 20] = [
    "section",
    "surface",
    "endpoint",
    "model",
    "engine_model",
    "configs",
    "driver",
    "base",
    "param",
    "type",
    "required",
    "support",
    "domain",
    "valid_value",
    "valid_outcome",
    "invalid_value",
    "invalid_status",
    "invalid_code",
    "invalid_param",
    "upstream",
];

#[derive(Clone, Debug)]
struct Row {
    cells: Vec<String>,
    finding: String,
}

fn ty_name(t: ParamType) -> &'static str {
    match t {
        ParamType::Integer => "integer",
        ParamType::Number => "number",
        ParamType::String => "string",
        ParamType::Boolean => "boolean",
        ParamType::Object => "object",
        ParamType::Array => "array",
        ParamType::IntegerOrString => "integer|string",
        ParamType::Any => "any",
    }
}

fn support_name(s: &Support) -> String {
    match s {
        Support::Honoured => "honoured".into(),
        Support::NoOp => "no_op".into(),
        Support::Refused => "refused".into(),
        Support::Gap { gap } => format!("gap:{}", gap.code()),
    }
}

fn num(f: f64) -> String {
    format!("{f}")
}

fn domain_text(d: &Domain, ty: ParamType) -> String {
    match d {
        Domain::Any => format!("any {}", ty_name(ty)),
        Domain::Int { min, max } => match (min, max) {
            (Some(a), Some(b)) => format!("int {a}..={b}"),
            (Some(a), None) => format!("int >= {a}"),
            (None, Some(b)) => format!("int <= {b}"),
            (None, None) => "int".into(),
        },
        Domain::Number {
            min,
            max,
            exclusive_min,
        } => {
            let lo =
                min.map(|m| format!("{}{}", if *exclusive_min { "> " } else { ">= " }, num(m)));
            match (lo, max) {
                (Some(l), Some(b)) => format!("number {l}, <= {}", num(*b)),
                (Some(l), None) => format!("number {l}"),
                (None, Some(b)) => format!("number <= {}", num(*b)),
                (None, None) => "number".into(),
            }
        }
        Domain::OneOf { values } if values.is_empty() => "one of [] (empty)".into(),
        Domain::OneOf { values } => {
            let vs: Vec<String> = values.iter().take(10).map(|v| v.to_string()).collect();
            let more = if values.len() > 10 {
                format!(", … ({})", values.len())
            } else {
                String::new()
            };
            format!("one of [{}{more}]", vs.join(", "))
        }
        Domain::Frames {
            step,
            offset,
            min,
            max,
        } => format!("frames {step}k+{offset} in {min}..={max}"),
        Domain::Text {
            min_chars,
            max_chars,
            non_blank,
        } => {
            let r = match max_chars {
                Some(m) => format!("{min_chars}..={m}"),
                None => format!(">= {min_chars}"),
            };
            format!(
                "text {r} chars{}",
                if *non_blank { ", non-blank" } else { "" }
            )
        }
        Domain::Items { max } => format!("items <= {max}"),
        Domain::Canvas {
            multiple,
            max_area,
            aspect,
            short_edges,
            pad_and_crop,
        } => format!(
            "canvas x{multiple}{}, <= {max_area} px, aspect {}..={}, tiers {:?}",
            if *pad_and_crop { " (pad+crop)" } else { "" },
            aspect.0,
            aspect.1,
            short_edges
        ),
        Domain::Aspect { min, max } => format!("aspect W:H {min}..={max}"),
        Domain::Media { media } => format!(
            "media {}",
            match media {
                MediaField::Image => "image",
                MediaField::Video => "video",
                MediaField::Audio => "audio",
            }
        ),
    }
}

fn cell(s: &str) -> String {
    let c = s.replace(['\t', '\n', '\r'], " ");
    if c.trim().is_empty() {
        "-".into()
    } else {
        c
    }
}

/// One spec to drive: where, and how.
#[derive(Clone)]
struct Job {
    spec: ParamSpec,
    configs: String,
    driver: Driver,
}

/// Drives one spec's parameters; returns its rows.
async fn drive(server: Option<&Server>, job: &Job) -> Vec<Row> {
    let spec = &job.spec;
    let surface = spec.surface.as_str();
    // The base: the first candidate the surface accepts.
    let mut base = None;
    // A base routed to another model (a tier's companion for the task)
    // is kept only when no candidate stays on the spec's model.
    let mut routed = None;
    let mut first = None;
    for (l, b) in bases(spec) {
        let o = run(server, &job.driver, surface, &b).await;
        if o.accepted && o.engine.is_none() {
            match &o.model {
                Some(m) if *m != spec.engine_model => {
                    if routed.is_none() {
                        routed = Some((l, b, o));
                    }
                    continue;
                }
                _ => {
                    base = Some((l, b, o));
                    break;
                }
            }
        }
        if first.is_none() {
            first = Some((l, b, o));
        }
    }
    let (base_label, base_body, base_obs, base_finding) = match (base, routed, first) {
        (Some((l, b, o)), _, _) => (format!("{l} {{{}}}", keys_of(&b)), b, o, None),
        (None, Some((l, b, o)), _) => {
            let m = o.model.clone().unwrap_or_default();
            (
                format!("{l} {{{}}} (routed to {m})", keys_of(&b)),
                b,
                o,
                Some(format!("every base routes the job to `{m}`")),
            )
        }
        (None, None, Some((l, b, o))) => {
            let f = format!(
                "no base accepted: `{l}` {} ({})",
                outcome(&o),
                short_msg(&o)
            );
            ("none accepted".to_owned(), b, o, Some(f))
        }
        (None, None, None) => (
            "none".to_owned(),
            json!({}),
            Obs::default(),
            Some("no base".into()),
        ),
    };
    let head = |section: &str, param: &str| {
        vec![
            section.to_owned(),
            spec.surface.clone(),
            spec.endpoint.clone(),
            spec.model.clone(),
            spec.engine_model.clone(),
            job.configs.clone(),
            job.driver.name().to_owned(),
            base_label.clone(),
            param.to_owned(),
        ]
    };
    let mut rows = Vec::new();
    // The base itself: what every case is applied to.
    let mut cells = head("base", "(base)");
    let mut base_out = outcome(&base_obs);
    if let Some(m) = base_obs.model.as_ref().filter(|m| **m != spec.engine_model) {
        base_out = format!("{base_out} (routed to {m})");
    }
    cells.extend([
        "object".to_owned(),
        "true".into(),
        "base".into(),
        "n/a (the request every case starts from)".into(),
        label(&base_body),
        base_out,
        "n/a (base)".into(),
        "n/a (base)".into(),
        "n/a (base)".into(),
        "n/a (base)".into(),
        "the spec's required fields plus the surface's minimal task inputs".into(),
    ]);
    rows.push(Row {
        cells: cells.iter().map(|c| cell(c)).collect(),
        finding: base_finding.clone().unwrap_or_else(|| "none".into()),
    });
    for p in &spec.params {
        let pl = plan(spec, p);
        let mut finding: Vec<String> = Vec::new();
        let served = matches!(p.support, Support::Honoured | Support::NoOp);
        let comp = if served {
            companions(spec, &p.name)
        } else {
            Vec::new()
        };
        let body_of = |pr: &Probe| {
            let mut a = comp.clone();
            a.extend(pr.assigns.iter().cloned());
            apply(&base_body, &a)
        };
        let (valid_value, valid_outcome) = match &pl.valid {
            Valid::None(why) => (why.clone(), why.clone()),
            Valid::Probe(first) => {
                let mut tried = Vec::new();
                let mut chosen = None;
                for pr in std::iter::once(first).chain(&pl.alternates) {
                    let o = run(server, &job.driver, surface, &body_of(pr)).await;
                    if o.status >= 500 {
                        finding.push(format!("valid {}: {} (5xx)", pr.label, o.status));
                    }
                    let ok = o.accepted && o.engine.is_none();
                    tried.push((pr, o));
                    if ok {
                        chosen = tried.last().map(|(p, o)| (*p, o.clone()));
                        break;
                    }
                }
                match chosen {
                    Some((pr, o)) => {
                        // A set member refused before it is valid with other
                        // fields (the cell says so); a range's lower bound
                        // refused while the upper is accepted is a finding.
                        if !matches!(p.domain, Domain::OneOf { .. }) {
                            for (q, r) in &tried[..tried.len() - 1] {
                                finding.push(format!(
                                    "valid {} refused ({})",
                                    q.label,
                                    short_msg(r)
                                ));
                            }
                        }
                        let label = if tried.len() > 1 {
                            format!("{} (first accepted of {} tried)", pr.label, tried.len())
                        } else {
                            pr.label.clone()
                        };
                        let mut out = outcome(&o);
                        if let Some(m) = o.model.as_ref().filter(|m| **m != spec.engine_model) {
                            out = format!("{out} (routed to {m})");
                        }
                        (label, out)
                    }
                    None => {
                        let (pr, o) = &tried[0];
                        if o.accepted {
                            finding.push(format!(
                                "valid {} accepted, engine refuses: {}",
                                pr.label,
                                truncate(o.engine.as_deref().unwrap_or(""), 80)
                            ));
                        } else {
                            finding.push(format!("valid {} refused ({})", pr.label, short_msg(o)));
                        }
                        (pr.label.clone(), outcome(o))
                    }
                }
            }
        };
        let mut iv = Vec::new();
        let mut is = Vec::new();
        let mut ic = Vec::new();
        let mut ip = Vec::new();
        for pr in &pl.invalid {
            let o = run(server, &job.driver, surface, &body_of(pr)).await;
            iv.push(pr.label.clone());
            is.push(if o.accepted {
                "accepted".to_owned()
            } else if o.status == 0 {
                "n/a (data channel: no HTTP status)".to_owned()
            } else {
                o.status.to_string()
            });
            ic.push(o.code.clone());
            ip.push(param_cell(&p.name, &o));
            if let Some(f) = invalid_finding(p, pr, &o, &spec.engine_model) {
                finding.push(f);
            }
        }
        let none = |what: &str| -> String {
            match exemption(surface, &p.name) {
                Some(r) => format!("n/a (exempt: {r})"),
                None => format!("n/a (no {what})"),
            }
        };
        let join = |v: &[String], what: &str| {
            if v.is_empty() {
                none(what)
            } else {
                v.join(" ; ")
            }
        };
        let mut cells = head("param", &p.name);
        cells.extend([
            ty_name(p.ty).to_owned(),
            p.required.to_string(),
            support_name(&p.support),
            match &p.wire {
                Some(w) => format!(
                    "{} (parse: {})",
                    domain_text(&p.domain, p.ty),
                    domain_text(w, p.ty)
                ),
                None => domain_text(&p.domain, p.ty),
            },
            valid_value,
            valid_outcome,
            join(&iv, "invalid case"),
            join(&is, "invalid case"),
            join(&ic, "invalid case"),
            join(&ip, "invalid case"),
            p.upstream
                .clone()
                .unwrap_or_else(|| "ours (no upstream contract)".into()),
        ]);
        rows.push(Row {
            cells: cells.iter().map(|c| cell(c)).collect(),
            finding: if finding.is_empty() {
                "none".into()
            } else {
                finding.join("; ")
            },
        });
    }
    // Unknown fields: the surface's policy.
    let unknown = "x_fv_cert_unknown";
    let o = run(
        server,
        &job.driver,
        surface,
        &apply(&base_body, &[(unknown.into(), Some(json!(1)))]),
    )
    .await;
    let (policy, want_accept) = match spec.unknown_fields {
        UnknownFields::Reject => ("reject", false),
        UnknownFields::Ignore => ("ignore", true),
    };
    let finding = if o.status >= 500 {
        format!("unknown field: {} (5xx)", o.status)
    } else if o.accepted != want_accept {
        format!("unknown-field policy `{policy}`, observed {}", outcome(&o))
    } else if !o.accepted {
        invalid_finding(
            &Param::new(unknown, ParamType::Integer, "-"),
            &Probe::set(unknown, json!(1)),
            &o,
            &spec.engine_model,
        )
        .unwrap_or_else(|| "none".into())
    } else {
        "none".into()
    };
    let mut cells = head("unknown_field", unknown);
    cells.extend([
        "integer".to_owned(),
        "false".into(),
        format!("policy:{policy}"),
        "n/a (not in the spec)".into(),
        "n/a (unknown field)".into(),
        "n/a (unknown field)".into(),
        "1".into(),
        if o.accepted {
            "accepted".into()
        } else if o.status == 0 {
            "n/a (data channel: no HTTP status)".to_owned()
        } else {
            o.status.to_string()
        },
        o.code.clone(),
        param_cell(unknown, &o),
        "the surface's unknown-field policy (ParamSpec::unknown_fields)".into(),
    ]);
    rows.push(Row {
        cells: cells.iter().map(|c| cell(c)).collect(),
        finding,
    });
    // Pairs: fields valid alone, refused together.
    for pair in PAIRS.iter().filter(|x| x.surface == surface) {
        let honoured = |n: &str| {
            spec.param(n)
                .is_some_and(|p| matches!(p.support, Support::Honoured | Support::NoOp))
        };
        if !pair.needs.iter().all(|n| honoured(n)) {
            continue;
        }
        let assigns = (pair.assigns)(spec);
        if assigns.is_empty() {
            continue;
        }
        let body = apply(&base_body, &assigns);
        let o = run(server, &job.driver, surface, &body).await;
        let named = match &o.param {
            Some(g) => pair.names.iter().any(|n| names(n, g)),
            None => pair
                .names
                .iter()
                .any(|n| message_names(n, &o.message).is_some()),
        };
        // FastWan's wire carries no field: the parser's own (plan E2
        // census) is read from the adapter.
        let internal = (surface == "fastwan" && !o.accepted).then(|| fastwan_param(spec, &body));
        let mut finding = Vec::new();
        if o.status >= 500 {
            finding.push(format!("{} (5xx)", o.status));
        } else if o.accepted || !named {
            finding.push(format!(
                "{}: expected refused naming {}, observed {}",
                pair.name,
                pair.names.join(" or "),
                outcome(&o)
            ));
        }
        let mut pcell = pair
            .names
            .iter()
            .map(|n| param_cell(n, &o))
            .find(|c| !c.contains("message: "))
            .unwrap_or_else(|| param_cell(pair.names[0], &o));
        if let Some(ip) = &internal {
            pcell = format!("{pcell}; the parser's param `{ip}`");
            if !pair.names.contains(&ip.as_str()) {
                finding.push(format!(
                    "{}: the parser names `{ip}` (not on FastWan's wire, which has no param field)",
                    pair.name
                ));
            }
        }
        let shown: Vec<String> = assigns
            .iter()
            .map(|(k, v)| match v {
                Some(v) => format!("{k}: {}", label(v)),
                None => format!("no {k}"),
            })
            .collect();
        let mut cells = head("pair", pair.name);
        cells.extend([
            "-".to_owned(),
            "false".into(),
            "pair".into(),
            pair.rule.to_owned(),
            "n/a (pair)".into(),
            "n/a (pair)".into(),
            shown.join(", "),
            if o.accepted {
                "accepted".into()
            } else if o.status == 0 {
                "n/a (data channel: no HTTP status)".to_owned()
            } else {
                o.status.to_string()
            },
            o.code.clone(),
            pcell,
            pair.upstream.to_owned(),
        ]);
        rows.push(Row {
            cells: cells.iter().map(|c| cell(c)).collect(),
            finding: if finding.is_empty() {
                "none".into()
            } else {
                finding.join("; ")
            },
        });
    }
    rows
}

/// The parameter FastWan's parser names for `body` (its wire error is
/// `{detail}` only).
fn fastwan_param(spec: &ParamSpec, body: &Value) -> String {
    use fastvideo_protocol::SubmitEndpoint as _;
    let ep = fastvideo_openai_videos::fastwan::FastWanSubmit {
        model: Some(spec.model.clone()),
    };
    let parsed = match serde_json::from_value(body.clone()) {
        Ok(b) => b,
        Err(_) => return "none (body error)".into(),
    };
    match ep.normalize(
        parsed,
        &fastvideo_protocol::NormalizeCtx::new(time::OffsetDateTime::UNIX_EPOCH),
    ) {
        Ok(_) => "none (the parser accepts; negotiation refuses)".into(),
        Err(e) => e.param.unwrap_or_else(|| "none".into()),
    }
}

// ------------------------------------------------------------------ pairs

/// Two fields each valid alone, refused together (or one without the
/// other it needs), naming one of `names`.
struct Pair {
    surface: &'static str,
    name: &'static str,
    /// Fields the spec must serve for the pair to apply.
    needs: &'static [&'static str],
    rule: &'static str,
    /// The assignments; none: the pair does not apply to this spec.
    assigns: fn(&ParamSpec) -> Assigns,
    names: &'static [&'static str],
    upstream: &'static str,
}

fn a(k: &str, v: Value) -> (String, Option<Value>) {
    (k.to_owned(), Some(v))
}

fn img() -> Value {
    json!(media().image)
}

/// A side of the spec's canvas example (`0`: width, `1`: height).
fn side(s: &ParamSpec, name: &str, i: usize) -> u64 {
    s.param(name)
        .and_then(|p| p.domain.valid_example())
        .and_then(|v| wxh(&v))
        .map_or(480, |x| if i == 0 { x.0 } else { x.1 })
}

const PAIRS: &[Pair] = &[
    Pair {
        surface: "native",
        name: "image_url+reference_urls",
        needs: &["image_url", "reference_urls"],
        rule: "a first frame or references, not both",
        assigns: |_| vec![a("image_url", img()), a("reference_urls", json!([img()]))],
        names: &["image_url", "reference_urls"],
        upstream: "ours (docs/serve/design.md §3.2)",
    },
    Pair {
        surface: "native",
        name: "width without height",
        needs: &["width", "height"],
        rule: "width and height go together",
        assigns: |s| {
            vec![
                a("width", json!(side(s, "width", 0))),
                ("height".into(), None),
            ]
        },
        names: &["width", "height"],
        upstream: "ours",
    },
    Pair {
        surface: "native",
        name: "start_s+extend_s",
        needs: &["start_s", "extend_s", "video_url"],
        rule: "a retake (start_s, end_s) or an extend (extend_s), not both",
        assigns: |_| {
            vec![
                a("video_url", json!(media().video)),
                a("start_s", json!(0)),
                a("end_s", json!(4)),
                a("extend_s", json!(4)),
            ]
        },
        names: &["start_s", "end_s", "extend_s"],
        upstream: "ours",
    },
    Pair {
        surface: "openai_videos",
        name: "input_reference+reference_url",
        needs: &["input_reference", "reference_url"],
        rule: "at most one legacy first-frame field",
        assigns: |_| vec![a("input_reference", img()), a("reference_url", img())],
        names: &["input_reference", "reference_url"],
        upstream: "docs/serve/specs/fastvideo/videos.schema.json",
    },
    Pair {
        surface: "openai_videos",
        name: "input_reference+image_reference",
        needs: &["input_reference", "image_reference"],
        rule: "a legacy first-frame field or image_reference, not both",
        assigns: |_| {
            vec![
                a("input_reference", img()),
                a("image_reference", json!([{"image_url": img()}])),
            ]
        },
        names: &["input_reference", "image_reference"],
        upstream: "docs/serve/specs/fastvideo/videos.schema.json",
    },
    Pair {
        surface: "openai_videos",
        name: "width without height",
        needs: &["width", "height"],
        rule: "width and height go together",
        assigns: |s| {
            vec![
                a("width", json!(side(s, "width", 0))),
                ("height".into(), None),
            ]
        },
        names: &["width", "height"],
        upstream: "docs/serve/specs/fastvideo/videos.schema.json",
    },
    Pair {
        surface: "fastwan",
        name: "height without width",
        needs: &["width", "height"],
        rule: "width and height go together",
        assigns: |s| {
            vec![
                a("height", json!(side(s, "height", 1))),
                ("width".into(), None),
            ]
        },
        names: &["height"],
        upstream: "docs/serve/research-minimax-fastvideo.md §2.2 (fastwan_link.py)",
    },
    Pair {
        surface: "fastwan",
        name: "width 0 with height",
        needs: &["width", "height"],
        rule: "both sides positive",
        assigns: |s| {
            vec![
                a("width", json!(0)),
                a("height", json!(side(s, "height", 1))),
            ]
        },
        names: &["width"],
        upstream: "docs/serve/research-minimax-fastvideo.md §2.2 (fastwan_link.py)",
    },
    Pair {
        surface: "minimax",
        name: "first_frame+reference_image",
        needs: &["content(reference_image)", "content[].image_url.url"],
        rule: "first_frame / last_frame images or reference items, not both",
        assigns: |_| {
            let text = json!({"type": "text", "text": "a fox runs"});
            let it = |role: &str| json!({"type": "image_url", "image_url": {"url": img()}, "role": role});
            vec![a(
                "content",
                json!([text, it("first_frame"), it("reference_image")]),
            )]
        },
        names: &["content", "reference_image", "first_frame"],
        upstream: "docs/serve/specs/minimax/v2-video-generation.json",
    },
    Pair {
        surface: "minimax",
        name: "two text items",
        needs: &["content[].text"],
        rule: "exactly one text item",
        assigns: |_| {
            let text = json!({"type": "text", "text": "a fox runs"});
            vec![a("content", json!([text, text]))]
        },
        names: &["content", "text"],
        upstream: "docs/serve/specs/minimax/v2-video-generation.json",
    },
    Pair {
        surface: "ltx",
        name: "last_frame_uri without image_uri",
        needs: &["last_frame_uri", "audio_uri"],
        rule: "audio-to-video: last_frame_uri requires image_uri",
        assigns: |_| vec![("image_uri".into(), None), a("last_frame_uri", img())],
        names: &["last_frame_uri", "image_uri"],
        upstream: "docs/serve/specs/ltx/openapi.json#/components/schemas/AudioToVideoRequest",
    },
    Pair {
        surface: "ltx",
        name: "no prompt and no image_uri",
        needs: &["audio_uri", "image_uri"],
        rule: "audio-to-video: prompt required without image_uri",
        assigns: |_| vec![("image_uri".into(), None), ("prompt".into(), None)],
        names: &["prompt", "image_uri"],
        upstream: "docs/serve/specs/ltx/openapi.json#/components/schemas/AudioToVideoRequest",
    },
    Pair {
        surface: "fal",
        name: "image_url+reference_image_urls",
        needs: &["image_url", "reference_image_urls"],
        rule: "a first frame or references, not both",
        assigns: |_| {
            vec![
                a("image_url", img()),
                a("reference_image_urls", json!([img()])),
            ]
        },
        names: &["image_url", "reference_image_urls"],
        upstream: "docs/serve/specs/fal (the endpoint's openapi.json)",
    },
];

// ------------------------------------------------------------------ specs

/// The golden layout's key of a spec (`surface endpoint model @ engine`).
fn spec_key(s: &ParamSpec) -> String {
    format!(
        "{} {} {} @ {}",
        s.surface, s.endpoint, s.model, s.engine_model
    )
}

/// A server for one shipped config on the fake engine (as the golden
/// renders it), whose jobs never start a step (each is cancelled once
/// read), with the CUDA pipelines' job checks for `[[models]]`.
async fn server(name: &str, path: &std::path::Path) -> (Server, Vec<ParamSpec>, Vec<Job>) {
    let dir = tempfile::Builder::new()
        .prefix(&format!("fv-cert-{name}-"))
        .tempdir()
        .unwrap();
    let keys: Vec<String> = (0..KEYS).map(key).collect();
    let mut c = common::load(name, path, dir.path(), &keys);
    if c.engine.backend == EngineBackendKind::Cuda {
        c.engine.backend = EngineBackendKind::Fake;
        c.engine.fake = Default::default();
    }
    c.engine.fake.mp4 = false;
    c.engine.fake.placeholder_output = true;
    c.engine.fake.load_ms = 0;
    c.engine.fake.step_ms = 600_000;
    c.limits.queue_max = 1_000_000;
    let models = fastvideo_serve::app::resolve_models(&c).unwrap_or_else(|e| panic!("{name}: {e}"));
    let validator = (!models.is_empty()).then(|| JobValidator::cuda(&models));
    let fal_director = c.protocols.fal_director;
    let app = App::build(c, Overrides::default())
        .await
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    app.gate.engine().wait_ready().await;
    let req = Request::builder()
        .method("GET")
        .uri("/fv/v1/params")
        .header("authorization", format!("Bearer {}", key(0)))
        .body(Body::empty())
        .unwrap();
    let r = app.router.clone().oneshot(req).await.unwrap();
    assert_eq!(r.status().as_u16(), 200, "{name}: /fv/v1/params");
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 26).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let specs: Vec<ParamSpec> = serde_json::from_value(body["specs"].clone()).unwrap();
    let mut extra = Vec::new();
    if let Some(r) = &app.reactor {
        if let Some(caps) = r.model() {
            if let Some(t) = r.table(&caps) {
                for s in t.param_specs(&caps) {
                    let cmd = s
                        .endpoint
                        .trim_start_matches("command ")
                        .split(' ')
                        .next()
                        .unwrap_or("")
                        .to_owned();
                    extra.push(Job {
                        spec: s,
                        configs: String::new(),
                        driver: Driver::Reactor(t.clone(), cmd),
                    });
                }
            }
        }
    }
    if fal_director {
        for s in fastvideo_fal::params::director_specs() {
            extra.push(Job {
                spec: s,
                configs: String::new(),
                driver: Driver::Director,
            });
        }
    }
    (
        Server {
            app,
            validator,
            next_key: std::sync::atomic::AtomicUsize::new(0),
            _dir: dir,
        },
        specs,
        extra,
    )
}

fn render(rows: &[Row]) -> String {
    let mut out = HEADER.join("\t");
    out += "\tfinding\n";
    for r in rows {
        out += &r.cells.join("\t");
        out += "\t";
        out += &cell(&r.finding);
        out += "\n";
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn certification_matrix_is_golden() {
    let t0 = Instant::now();
    // Which config first serves each distinct spec, and every config
    // serving it.
    let mut seen: BTreeMap<String, (ParamSpec, Vec<String>)> = BTreeMap::new();
    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
    let mut requests_by_config = Vec::new();
    for (name, path) in common::configs() {
        let (srv, specs, extra) = server(&name, &path).await;
        let mut jobs: Vec<Job> = Vec::new();
        let all: Vec<(ParamSpec, Driver)> = specs
            .into_iter()
            .filter_map(|s| driver_of(&s).map(|d| (s, d)))
            .chain(extra.into_iter().map(|j| (j.spec, j.driver)))
            .collect();
        for (s, d) in all {
            // `#n` when another config serves a different spec under the key.
            let base = spec_key(&s);
            let mut k = base.clone();
            let mut n = 1;
            while seen.get(&k).is_some_and(|(x, _)| x != &s) {
                n += 1;
                k = format!("{base} #{n}");
            }
            match seen.get_mut(&k) {
                Some((_, cfgs)) => {
                    if !cfgs.contains(&name) {
                        cfgs.push(name.clone());
                    }
                }
                None => {
                    seen.insert(k.clone(), (s.clone(), vec![name.clone()]));
                    jobs.push(Job {
                        spec: s,
                        configs: k,
                        driver: d,
                    });
                }
            }
        }
        let tc = Instant::now();
        let n_jobs = jobs.len();
        let srv_ref = &srv;
        let rows_ref = &rows;
        futures::stream::iter(jobs)
            .for_each_concurrent(CONCURRENCY, |j| async move {
                let r = drive(Some(srv_ref), &j).await;
                // `configs` holds the spec key until every config is seen.
                rows_ref.lock().unwrap().extend(r);
            })
            .await;
        requests_by_config.push(format!(
            "{name}: {n_jobs} new specs in {:.1?}",
            tc.elapsed()
        ));
        drop(srv);
    }
    // The spec key -> "first config (+N)".
    let mut rows = rows.into_inner().unwrap();
    for r in &mut rows {
        let (_, cfgs) = &seen[&r.cells[5]];
        r.cells[5] = if cfgs.len() > 1 {
            format!("{} (+{})", cfgs[0], cfgs.len() - 1)
        } else {
            cfgs[0].clone()
        };
    }
    let order = |s: &str| match s {
        "base" => 0,
        "param" => 1,
        "unknown_field" => 2,
        _ => 3,
    };
    rows.sort_by(|x, y| {
        (order(&x.cells[0]), &x.cells[1..9]).cmp(&(order(&y.cells[0]), &y.cells[1..9]))
    });
    let text = render(&rows);
    // No empty cell, every row the header's width.
    for (i, line) in text.lines().enumerate() {
        let cells: Vec<&str> = line.split('\t').collect();
        assert_eq!(cells.len(), HEADER.len() + 1, "line {}: {line}", i + 1);
        assert!(
            cells.iter().all(|c| !c.trim().is_empty()),
            "line {}: empty cell: {line}",
            i + 1
        );
    }
    let fivexx: Vec<&Row> = rows
        .iter()
        .filter(|r| r.finding.contains("(5xx)"))
        .collect();
    let surfaces: BTreeSet<&str> = rows.iter().map(|r| r.cells[1].as_str()).collect();
    let models: BTreeSet<&str> = rows.iter().map(|r| r.cells[4].as_str()).collect();
    let findings = rows.iter().filter(|r| r.finding != "none").count();
    eprintln!(
        "certification matrix: {} rows, {} specs, {} surfaces, {} engine models, {} rows with findings, {} requests, {:.1?}\n{}",
        rows.len(),
        seen.len(),
        surfaces.len(),
        models.len(),
        findings,
        SENT.load(std::sync::atomic::Ordering::Relaxed),
        t0.elapsed(),
        requests_by_config.join("\n")
    );
    assert!(
        fivexx.is_empty(),
        "5xx in the matrix: {:#?}",
        fivexx
            .iter()
            .map(|r| (&r.cells[1..9], &r.finding))
            .collect::<Vec<_>>()
    );
    let path = tsv_path();
    if let Some(out) = std::env::var_os("FV_CERT_WRITE") {
        // `1`: the committed file; else the path given.
        let out = if out == "1" {
            path.clone()
        } else {
            PathBuf::from(out)
        };
        std::fs::write(&out, &text).unwrap();
        eprintln!("wrote {}", out.display());
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e} (FV_CERT_WRITE=1 writes it)", path.display()));
    if want == text {
        return;
    }
    let w: BTreeSet<&str> = want.lines().collect();
    let g: BTreeSet<&str> = text.lines().collect();
    let gone: Vec<&&str> = w.difference(&g).take(10).collect();
    let new: Vec<&&str> = g.difference(&w).take(10).collect();
    panic!(
        "{} drifted (FV_CERT_WRITE=1 regenerates; list the change in the PR)\nonly in the file:\n{}\nnow:\n{}",
        path.display(),
        gone.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n"),
        new.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n"),
    );
}

/// Every Reactor mode's table (clip bounds as H3's 5..15 s).
fn reactor_specs() -> Vec<ParamSpec> {
    use fastvideo_reactor::commands::{ClipBounds, CommandTable};
    use fastvideo_reactor::engine::Mode;
    let caps = ModelCaps::h3("reactor-model", false);
    let b = ClipBounds {
        min_s: 5.0,
        max_s: 15.0,
        default_s: 5.0,
    };
    [Mode::Clip, Mode::Causal, Mode::Avatar, Mode::Duplex]
        .into_iter()
        .flat_map(|m| CommandTable::for_mode(m, b).param_specs(&caps))
        .collect()
}

/// The CI assertion (plan E3): every parameter of every spec has a valid
/// case (or is refused / has an empty domain on the model) and an invalid
/// case, else an exemption with its reason; every exemption is needed.
#[test]
fn every_param_has_a_valid_and_an_invalid_case() {
    let text = std::fs::read_to_string(golden_specs_path()).unwrap();
    let golden: Value = serde_json::from_str(&text).unwrap();
    let mut specs: Vec<ParamSpec> = golden["specs"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| serde_json::from_value(v.clone()).unwrap())
        .collect();
    specs.extend(fastvideo_fal::params::director_specs());
    specs.extend(reactor_specs());
    let mut missing = Vec::new();
    let mut used = BTreeSet::new();
    let mut n = 0;
    for s in &specs {
        for p in &s.params {
            n += 1;
            let pl = plan(s, p);
            if let Valid::Probe(pr) = &pl.valid {
                assert!(
                    !pr.assigns.is_empty(),
                    "{} `{}`: an empty valid case",
                    spec_key(s),
                    p.name
                );
            }
            let refused = matches!(p.support, Support::Refused | Support::Gap { .. });
            if refused {
                assert!(
                    matches!(pl.valid, Valid::None(_)) && !pl.invalid.is_empty(),
                    "{} `{}`: a refused parameter's case is any value, refused",
                    spec_key(s),
                    p.name
                );
            }
            if pl.invalid.is_empty() {
                match exemption(&s.surface, &p.name) {
                    Some(_) => {
                        used.insert((s.surface.clone(), p.name.clone()));
                    }
                    None => missing.push(format!(
                        "{} `{}` ({:?}): no invalid case and no exemption",
                        spec_key(s),
                        p.name,
                        p.domain
                    )),
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} of {n} parameters:\n{}",
        missing.len(),
        missing.join("\n")
    );
    let stale: Vec<_> = EXEMPT
        .iter()
        .filter(|(s, p, _)| !used.contains(&((*s).to_owned(), (*p).to_owned())))
        .collect();
    assert!(stale.is_empty(), "exemptions no parameter needs: {stale:?}");
}
