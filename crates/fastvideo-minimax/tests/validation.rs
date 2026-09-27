//! `content[]` role validation and the duration / resolution / ratio rules
//! (research §1.3), against the pure `normalize`.

use fastvideo_minimax::create::normalize;
use fastvideo_minimax::CreateBody;
use fastvideo_protocol::{
    Anchor, ApiError, CallbackSpec, CanvasSpec, ErrorKind, GapId, GenerationRequest, Length,
    MediaKind, MediaRef, Ratio, Snap, Task,
};
use serde_json::{json, Value};

const IMG: &str = "https://cdn.example.com/a.png";
const VID: &str = "https://cdn.example.com/a.mp4";
const AUD: &str = "https://cdn.example.com/a.mp3";

fn text(t: &str) -> Value {
    json!({"type": "text", "text": t})
}
fn image(role: Option<&str>) -> Value {
    let mut v = json!({"type": "image_url", "image_url": {"url": IMG}});
    if let Some(r) = role {
        v["role"] = json!(r);
    }
    v
}
fn video(role: Option<&str>) -> Value {
    let mut v = json!({"type": "video_url", "video_url": {"url": VID}});
    if let Some(r) = role {
        v["role"] = json!(r);
    }
    v
}
fn audio(role: Option<&str>) -> Value {
    let mut v = json!({"type": "audio_url", "audio_url": {"url": AUD}});
    if let Some(r) = role {
        v["role"] = json!(r);
    }
    v
}

fn body(model: &str, content: Vec<Value>, extra: Value) -> Value {
    let mut b = json!({"model": model, "content": content, "resolution": "768P", "duration": 5, "ratio": "adaptive"});
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        if v.is_null() {
            b.as_object_mut().unwrap().remove(&k);
        } else {
            b[k] = v;
        }
    }
    b
}

fn run(b: Value) -> Result<GenerationRequest, ApiError> {
    let parsed: CreateBody = serde_json::from_value(b).unwrap();
    normalize(parsed)
}

fn err(b: Value) -> ApiError {
    run(b).expect_err("should be refused")
}

fn h3(content: Vec<Value>) -> Value {
    body("MiniMax-H3", content, json!({}))
}

// ---- content roles ---------------------------------------------------------------

#[test]
fn modes_by_role() {
    // t2va needs a real ratio.
    let r = run(body("MiniMax-H3", vec![text("a")], json!({"ratio": "9:16"}))).unwrap();
    assert_eq!(r.task, Task::T2V);
    assert!(r.keyframes.is_empty() && r.references.is_empty());

    // One roleless image = first frame.
    for c in [vec![text("a"), image(None)], vec![image(Some("first_frame")), text("a")]] {
        let r = run(h3(c)).unwrap();
        assert_eq!(r.task, Task::I2V);
        assert_eq!(r.keyframes.len(), 1);
        assert_eq!(r.keyframes[0].at, Anchor::First);
        assert_eq!(r.keyframes[0].image, MediaRef::Http(IMG.parse().unwrap()));
    }
    // Last frame only, and first + last: keyframes (fl2va).
    let r = run(h3(vec![text("a"), image(Some("last_frame"))])).unwrap();
    assert_eq!(r.task, Task::Keyframes);
    assert_eq!(r.keyframes.iter().map(|k| k.at).collect::<Vec<_>>(), [Anchor::Last]);
    let r = run(h3(vec![text("a"), image(Some("last_frame")), image(Some("first_frame"))])).unwrap();
    assert_eq!(r.task, Task::Keyframes);
    assert_eq!(r.keyframes.iter().map(|k| k.at).collect::<Vec<_>>(), [Anchor::First, Anchor::Last]);

    // References keep content order; roleless video/audio are references.
    let r = run(h3(vec![
        audio(Some("reference_audio")),
        text("a"),
        image(Some("reference_image")),
        video(None),
        audio(None),
        video(Some("reference_video")),
    ]))
    .unwrap();
    assert_eq!(r.task, Task::Ref2V);
    let kinds: Vec<MediaKind> = r.references.iter().map(|x| x.kind).collect();
    assert_eq!(kinds, [MediaKind::Audio, MediaKind::Image, MediaKind::Video, MediaKind::Audio, MediaKind::Video]);
    assert!(r.keyframes.is_empty());
    assert_eq!(r.prompt, "a");
}

#[test]
fn content_refusals() {
    let cases: Vec<(&str, Vec<Value>, &str)> = vec![
        ("no text", vec![image(None)], "non-empty text item"),
        ("empty text", vec![text("  ")], "non-empty text item"),
        ("two texts", vec![text("a"), text("b")], "exactly one text item"),
        ("text role", vec![json!({"type": "text", "text": "a", "role": "first_frame"})], "not valid for a text item"),
        ("unknown type", vec![text("a"), json!({"type": "file_url", "file_url": {"url": IMG}})], "unknown content type"),
        ("missing type", vec![text("a"), json!({"image_url": {"url": IMG}})], "type is required"),
        ("missing url", vec![text("a"), json!({"type": "image_url", "image_url": {}})], "is required"),
        ("url object of other type", vec![text("a"), json!({"type": "image_url", "video_url": {"url": VID}})], "is required"),
        ("image as video ref", vec![text("a"), image(Some("reference_video"))], "not valid for a image_url item"),
        ("video as frame", vec![text("a"), video(Some("first_frame"))], "not valid for a video_url item"),
        ("audio as image ref", vec![text("a"), audio(Some("reference_image"))], "not valid for a audio_url item"),
        ("unknown role", vec![text("a"), image(Some("base_video"))], "unknown role"),
        ("two first frames", vec![text("a"), image(None), image(Some("first_frame"))], "one first_frame"),
        ("two last frames", vec![text("a"), image(Some("last_frame")), image(Some("last_frame"))], "one last_frame"),
        ("frame + ref", vec![text("a"), image(Some("first_frame")), image(Some("reference_image"))], "cannot be mixed"),
        ("roleless image + ref", vec![text("a"), image(None), audio(None)], "cannot be mixed"),
        ("last + video ref", vec![text("a"), image(Some("last_frame")), video(None)], "cannot be mixed"),
        ("10 images", std::iter::once(text("a")).chain((0..10).map(|_| image(Some("reference_image")))).collect(), "at most 9 reference_image"),
        ("4 videos", std::iter::once(text("a")).chain((0..4).map(|_| video(None))).collect(), "at most 3 reference_video"),
        ("4 audio", std::iter::once(text("a")).chain((0..4).map(|_| audio(None))).collect(), "at most 3 reference_audio"),
        (
            "13 refs",
            std::iter::once(text("a"))
                .chain((0..9).map(|_| image(Some("reference_image"))))
                .chain((0..3).map(|_| video(None)))
                .chain(std::iter::once(audio(None)))
                .collect(),
            "at most 12 reference items",
        ),
        ("bad url", vec![text("a"), json!({"type": "image_url", "image_url": {"url": "ftp://x/a.png"}})], "must be a public URL"),
        ("ltx upload", vec![text("a"), json!({"type": "image_url", "image_url": {"url": "ltx://uploads/abc"}})], "must be a public URL"),
        ("data uri of wrong type", vec![text("a"), json!({"type": "image_url", "image_url": {"url": "data:video/mp4;base64,AAAA"}})], "data:image/"),
        ("uppercase data format", vec![text("a"), json!({"type": "image_url", "image_url": {"url": "data:image/PNG;base64,AAAA"}})], "lowercase"),
    ];
    for (name, content, needle) in cases {
        let e = err(h3(content));
        assert_eq!(e.kind, ErrorKind::InvalidRequest, "{name}: {e:?}");
        assert!(e.message.contains(needle), "{name}: `{}` lacks `{needle}`", e.message);
    }
    // 12 references in total are fine.
    let ok: Vec<Value> = std::iter::once(text("a"))
        .chain((0..9).map(|_| image(Some("reference_image"))))
        .chain((0..3).map(|_| video(None)))
        .collect();
    assert_eq!(run(h3(ok)).unwrap().references.len(), 12);
    // Prompt length: 7000 characters (not bytes) are fine, 7001 are not.
    let long = "é".repeat(7000);
    assert!(run(body("MiniMax-H3", vec![text(&long)], json!({"ratio": "16:9"}))).is_ok());
    let e = err(body("MiniMax-H3", vec![text(&(long + "x"))], json!({"ratio": "16:9"})));
    assert!(e.message.contains("7000"));
    // mm_file:// is a provider file: the ProviderFiles gap.
    let e = err(h3(vec![text("a"), json!({"type": "image_url", "image_url": {"url": "mm_file://123"}, "role": "first_frame"})]));
    assert_eq!(e.gap(), Some(GapId::ProviderFiles));
    assert_eq!(e.param.as_deref(), Some("content[1].image_url.url"));
    // Data URIs pass through.
    let r = run(h3(vec![text("a"), json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})])).unwrap();
    assert!(matches!(r.keyframes[0].image, MediaRef::DataUri(_)));
}

// ---- duration ----------------------------------------------------------------------

#[test]
fn durations_per_model() {
    let t = |model: &str, d: Value| run(body(model, vec![text("a")], json!({"ratio": "16:9", "duration": d})));
    // MiniMax-H3: 4..15, including 4 s.
    for d in [4, 5, 10, 15] {
        let r = t("MiniMax-H3", json!(d)).unwrap();
        assert_eq!(r.timing.length, Length::Seconds { value: d as f64, snap: Snap::AlignUp });
    }
    // MiniMax-H3-Max: 5..15; 4 is rejected per the MiniMax docs.
    let e = t("MiniMax-H3-Max", json!(4)).unwrap_err();
    assert_eq!(e.kind, ErrorKind::InvalidRequest);
    assert!(e.message.contains("expected 5 to 15"), "{}", e.message);
    assert!(t("MiniMax-H3-Max", json!(5)).is_ok());
    // Turbo / Draft (our ids) admit 4 s like H3.
    assert!(t("MiniMax-H3-Turbo", json!(4)).is_ok());
    assert!(t("MiniMax-H3-Draft", json!(4)).is_ok());
    // Out of range, non-integers, strings, missing.
    for bad in [json!(3), json!(16), json!(0), json!(-5), json!(5.5), json!("5"), Value::Null] {
        let e = t("MiniMax-H3", bad.clone()).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("duration"), "{bad}");
    }
    // An integral float is an integer.
    assert!(t("MiniMax-H3", json!(6.0)).is_ok());
}

// ---- resolution --------------------------------------------------------------------

#[test]
fn resolutions_per_model() {
    let t = |model: &str, r: Value| run(body(model, vec![text("a")], json!({"ratio": "16:9", "resolution": r})));
    let short = |r: GenerationRequest| match r.canvas {
        CanvasSpec::Aspect { short_edge, .. } => short_edge,
        c => panic!("{c:?}"),
    };
    assert_eq!(short(t("MiniMax-H3", json!("768P")).unwrap()), 768);
    // 2K is a valid H3 value; the engine refuses it at negotiation
    // (Unsupported(H3Resolution2K), short edge 1440).
    assert_eq!(short(t("MiniMax-H3", json!("2K")).unwrap()), 1440);
    let e = t("MiniMax-H3", json!("480P")).unwrap_err();
    assert!(e.message.contains("not supported by MiniMax-H3"), "{}", e.message);
    // H3-Max: 480P / 768P, no 2K; optional (768P).
    assert_eq!(short(t("MiniMax-H3-Max", json!("480P")).unwrap()), 480);
    assert_eq!(short(t("MiniMax-H3-Max", json!("768P")).unwrap()), 768);
    assert!(t("MiniMax-H3-Max", json!("2K")).unwrap_err().message.contains("not supported by MiniMax-H3-Max"));
    assert_eq!(short(t("MiniMax-H3-Max", Value::Null).unwrap()), 768);
    // H3 requires it.
    assert_eq!(t("MiniMax-H3", Value::Null).unwrap_err().param.as_deref(), Some("resolution"));
    assert_eq!(t("MiniMax-H3", json!("1080P")).unwrap_err().param.as_deref(), Some("resolution"));
    assert_eq!(t("MiniMax-H3", json!("768p")).unwrap_err().param.as_deref(), Some("resolution"));
}

// ---- ratio -------------------------------------------------------------------------

#[test]
fn ratios_per_mode() {
    // t2va: required, not adaptive; every listed ratio parses.
    for (s, r) in [("21:9", Ratio::new(21, 9)), ("16:9", Ratio::R16_9), ("4:3", Ratio::R4_3), ("1:1", Ratio::R1_1), ("3:4", Ratio::R3_4), ("9:16", Ratio::R9_16)] {
        let got = run(body("MiniMax-H3", vec![text("a")], json!({"ratio": s}))).unwrap();
        assert_eq!(got.canvas, CanvasSpec::Aspect { ratio: r, short_edge: 768 });
    }
    for missing in [json!("adaptive"), Value::Null] {
        let e = err(body("MiniMax-H3", vec![text("a")], json!({"ratio": missing})));
        assert_eq!(e.param.as_deref(), Some("ratio"));
    }
    assert_eq!(err(body("MiniMax-H3", vec![text("a")], json!({"ratio": "2:1"}))).param.as_deref(), Some("ratio"));

    // i2va: always adaptive; another valid value is ignored; an invalid one is not.
    for r in [json!("adaptive"), json!("9:16"), Value::Null] {
        let got = run(body("MiniMax-H3", vec![text("a"), image(None)], json!({"ratio": r}))).unwrap();
        assert_eq!(got.canvas, CanvasSpec::FollowImage { short_edge: 768 });
    }
    assert!(run(body("MiniMax-H3", vec![text("a"), image(None)], json!({"ratio": "5:4"}))).is_err());
    let got = run(body("MiniMax-H3", vec![text("a"), image(Some("last_frame"))], json!({"ratio": "1:1"}))).unwrap();
    assert_eq!(got.canvas, CanvasSpec::FollowImage { short_edge: 768 });

    // r2va: optional, default adaptive (follows the first reference image);
    // an explicit ratio is honoured; no image reference -> 16:9.
    let got = run(body("MiniMax-H3", vec![text("a"), video(None), image(Some("reference_image"))], json!({"ratio": null}))).unwrap();
    assert_eq!(got.canvas, CanvasSpec::FollowImage { short_edge: 768 });
    let got = run(body("MiniMax-H3", vec![text("a"), image(Some("reference_image"))], json!({"ratio": "3:4"}))).unwrap();
    assert_eq!(got.canvas, CanvasSpec::Aspect { ratio: Ratio::R3_4, short_edge: 768 });
    let got = run(body("MiniMax-H3", vec![text("a"), audio(None)], json!({}))).unwrap();
    assert_eq!(got.canvas, CanvasSpec::Aspect { ratio: Ratio::R16_9, short_edge: 768 });
}

// ---- model, extra, callback_url ------------------------------------------------------

#[test]
fn model_extra_callback() {
    for bad in [json!("MiniMax-Hailuo-02"), json!(""), Value::Null] {
        assert_eq!(err(body("x", vec![text("a")], json!({"model": bad, "ratio": "16:9"}))).param.as_deref(), Some("model"));
    }
    // extra: H3-Max (and our tiers) only; enum checked; accepted as a no-op.
    let mk = |model: &str, extra: Value| run(body(model, vec![text("a")], json!({"ratio": "16:9", "extra": extra})));
    for mode in ["disabled", "balanced", "quality"] {
        let r = mk("MiniMax-H3-Max", json!({"prompt_expansion_mode": mode})).unwrap();
        assert_eq!(r.accepted_noop, ["prompt_expansion_mode"]);
    }
    assert!(mk("MiniMax-H3-Turbo", json!({"prompt_expansion_mode": "quality"})).is_ok());
    assert!(mk("MiniMax-H3-Max", json!({})).unwrap().accepted_noop.is_empty());
    assert_eq!(mk("MiniMax-H3", json!({"prompt_expansion_mode": "balanced"})).unwrap_err().param.as_deref(), Some("extra"));
    for bad in [json!({"prompt_expansion_mode": "balance"}), json!({"prompt_expansion_mode": ""}), json!({"prompt_expansion_mode": true})] {
        assert_eq!(mk("MiniMax-H3-Max", bad).unwrap_err().param.as_deref(), Some("extra.prompt_expansion_mode"));
    }
    assert_eq!(mk("MiniMax-H3-Max", json!({"seed": 1})).unwrap_err().param.as_deref(), Some("extra"));
    assert_eq!(mk("MiniMax-H3-Max", json!("quality")).unwrap_err().param.as_deref(), Some("extra"));

    // callback_url
    let r = run(body("MiniMax-H3", vec![text("a")], json!({"ratio": "16:9", "callback_url": "https://hooks.example.com/mm"}))).unwrap();
    assert_eq!(r.callback, Some(CallbackSpec::MiniMax { url: "https://hooks.example.com/mm".parse().unwrap() }));
    for bad in ["ftp://x/y", "not a url"] {
        assert_eq!(err(body("MiniMax-H3", vec![text("a")], json!({"ratio": "16:9", "callback_url": bad}))).param.as_deref(), Some("callback_url"));
    }
    // Unknown top-level fields are ignored (e.g. V1's prompt_optimizer).
    assert!(run(body("MiniMax-H3", vec![text("a")], json!({"ratio": "16:9", "prompt_optimizer": true}))).is_ok());
}
