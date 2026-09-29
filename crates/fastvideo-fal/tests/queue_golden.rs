//! Golden conformance (design §7.1) for the pure parts of the fal queue API:
//! every schema field, enum value and default; normalization of the
//! documented example bodies; the submit, status, result, error, cancel and
//! webhook bodies.
//!
//! Fixtures live in `tests/queue_golden/`. `FV_BLESS=1 cargo test -p fastvideo-fal
//! --test queue_golden` rewrites them; review the diff before committing.

use std::path::PathBuf;
use std::time::Duration;

use fastvideo_fal::error::{error_body, FalProtocol};
use fastvideo_fal::queue::{fal_timestamp, logs_param, status_json, FalView};
use fastvideo_fal::schema::{AspectRatio, Endpoint, FalInput, Resolution};
use fastvideo_fal::webhook::webhook_body;
use fastvideo_fal::{FalApp, FalEndpoint};
use fastvideo_protocol::{
    ApiError, Artifact, ArtifactId, ArtifactLocation, AudioPlan, BatchProtocol, CanvasSpec,
    ErrorCtx, GapId, Job, JobId, JobMetrics, JobView, LogLine, NormalizeCtx, PostProcess,
    ProtocolId, ResolvedJob, SamplingOverrides, SubmitEndpoint, Task, Tier, UrlSigner, ViewCtx,
};
use serde_json::{json, Value};
use time::macros::datetime;
use time::OffsetDateTime;
use url::Url;

const RID: &str = "764cabcf-b745-4b3e-ae38-1200304cf45b";

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/queue_golden").join(name)
}

/// Compares `actual` with `tests/golden/<name>` (or writes it under `FV_BLESS=1`).
fn golden(name: &str, actual: &Value) {
    let p = golden_path(name);
    if std::env::var_os("FV_BLESS").is_some() {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, serde_json::to_string_pretty(actual).unwrap() + "\n").unwrap();
        return;
    }
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e} (run with FV_BLESS=1)", p.display()));
    let want: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        actual,
        &want,
        "golden {name} differs:\nactual: {}",
        serde_json::to_string_pretty(actual).unwrap()
    );
}

fn now() -> OffsetDateTime {
    datetime!(2026-09-27 12:00:00 UTC)
}

fn ncx(query: &[(&str, &str)]) -> NormalizeCtx {
    let mut c = NormalizeCtx::new(now());
    c.query = query.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    c
}

fn ep(endpoint: Endpoint) -> FalEndpoint {
    FalEndpoint { app: FalApp::h3(Tier::Max), endpoint }
}

fn normalize(endpoint: Endpoint, body: Value) -> Result<fastvideo_protocol::GenerationRequest, ApiError> {
    ep(endpoint).normalize(body, &ncx(&[]))
}

/// The rendered 422 (or other) error body for a refused input.
fn refusal(endpoint: Endpoint, body: Value) -> Value {
    let e = normalize(endpoint, body).expect_err("refused");
    let r = FalProtocol.render_error(&e, &ErrorCtx::default());
    json!({"status": r.status, "error_type": r.header("x-fal-error-type"), "body": r.json_body().unwrap()})
}

// ------------------------------------------------------------ requests

const PROMPT: &str = "A white kitten chases a butterfly across a sunlit garden. Gentle camera tracking, natural movement, soft afternoon light filtering through the leaves.";
const IMG: &str = "https://storage.googleapis.com/falserverless/example_inputs/hailuo23/pro_i2v_in.jpg";

#[test]
fn documented_examples_normalize() {
    type Case = (&'static str, Endpoint, Value, &'static [(&'static str, &'static str)]);
    let cases: Vec<Case> = vec![
        // LLMS-T2V required-only request.
        ("t2v_required_only", Endpoint::TextToVideo, json!({"prompt": PROMPT, "prompt_expansion_mode": "disabled"}), &[]),
        // LLMS-T2V full request.
        (
            "t2v_full",
            Endpoint::TextToVideo,
            json!({"prompt": PROMPT, "duration": 5, "resolution": "768P", "enable_safety_checker": true, "prompt_expansion_mode": "disabled", "aspect_ratio": "16:9"}),
            &[],
        ),
        // LLMS-I2V full request.
        (
            "i2v_full",
            Endpoint::ImageToVideo,
            json!({"prompt": "The camera slowly pulls back from the scene", "duration": 5, "resolution": "768P", "enable_safety_checker": true, "prompt_expansion_mode": "disabled", "image_url": IMG}),
            &[],
        ),
        ("i2v_end_only", Endpoint::ImageToVideo, json!({"prompt": "p", "end_image_url": IMG}), &[]),
        ("i2v_first_and_last", Endpoint::ImageToVideo, json!({"prompt": "p", "image_url": IMG, "end_image_url": "data:image/png;base64,iVBORw0KGgo="}), &[]),
        ("i2v_no_images_is_t2v", Endpoint::ImageToVideo, json!({"prompt": "p", "duration": 10, "resolution": "480P"}), &[]),
        (
            "r2v_all_kinds",
            Endpoint::ReferenceToVideo,
            json!({
                "prompt": "Image 1 walks through Video 1 while Audio 1 plays",
                "seed": 1851572118,
                "aspect_ratio": "9:16",
                "reference_audio_urls": ["https://a.test/a1.wav"],
                "reference_video_urls": ["https://a.test/v1.mp4", "https://a.test/v2.mp4"],
                "reference_image_urls": ["https://a.test/i1.png", "https://a.test/i2.png"]
            }),
            &[],
        ),
        ("r2v_adaptive_follows_first_image", Endpoint::ReferenceToVideo, json!({"prompt": "p", "reference_image_urls": ["https://a.test/i1.png"]}), &[]),
        ("r2v_adaptive_without_images", Endpoint::ReferenceToVideo, json!({"prompt": "p", "reference_video_urls": ["https://a.test/v1.mp4"]}), &[]),
        (
            "t2v_webhook_seed_sync_mode",
            Endpoint::TextToVideo,
            json!({"prompt": "p", "seed": 42, "sync_mode": true, "duration": 15, "aspect_ratio": "21:9", "target_audio_url": "https://a.test/t.mp3"}),
            &[("fal_webhook", "https://hooks.example.com/fal?x=1")],
        ),
    ];
    let mut out = serde_json::Map::new();
    for (name, e, body, q) in cases {
        let r = ep(e).normalize(body.clone(), &ncx(q)).unwrap_or_else(|err| panic!("{name}: {err}"));
        out.insert(name.into(), json!({"endpoint": e.sub(), "body": body, "query": q, "normalized": r}));
    }
    golden("requests.json", &Value::Object(out));
}

#[test]
fn defaults_per_endpoint() {
    let mut out = serde_json::Map::new();
    for e in Endpoint::ALL {
        let body = match e {
            Endpoint::ReferenceToVideo => json!({"prompt": "p", "reference_image_urls": ["https://a.test/i.png"]}),
            _ => json!({"prompt": "p"}),
        };
        let input = FalInput::parse(e, &body).unwrap();
        out.insert(e.sub().into(), serde_json::to_value(&input).unwrap());
    }
    golden("defaults.json", &Value::Object(out));
    // Null is the same as absent for every optional field.
    let nulls = json!({"prompt": "p", "duration": null, "resolution": null, "seed": null, "enable_safety_checker": null,
                       "sync_mode": null, "prompt_expansion_mode": null, "aspect_ratio": null, "target_audio_url": null});
    assert_eq!(FalInput::parse(Endpoint::TextToVideo, &nulls).unwrap(), FalInput::parse(Endpoint::TextToVideo, &json!({"prompt": "p"})).unwrap());
    // Unknown fields are ignored (no additionalProperties: false upstream).
    assert!(FalInput::parse(Endpoint::TextToVideo, &json!({"prompt": "p", "extra": 1})).is_ok());
}

#[test]
fn every_enum_value_maps() {
    let mut out = serde_json::Map::new();
    for res in Resolution::ALL {
        let r = normalize(Endpoint::TextToVideo, json!({"prompt": "p", "resolution": res.as_str()})).unwrap();
        out.insert(format!("t2v resolution {}", res.as_str()), serde_json::to_value(&r.canvas).unwrap());
    }
    for a in AspectRatio::T2V {
        let r = normalize(Endpoint::TextToVideo, json!({"prompt": "p", "aspect_ratio": a.as_str()})).unwrap();
        out.insert(format!("t2v aspect_ratio {}", a.as_str()), serde_json::to_value(&r.canvas).unwrap());
    }
    for a in AspectRatio::R2V {
        let r = normalize(Endpoint::ReferenceToVideo, json!({"prompt": "p", "aspect_ratio": a.as_str(), "reference_image_urls": ["https://a.test/i.png"]})).unwrap();
        out.insert(format!("r2v aspect_ratio {}", a.as_str()), serde_json::to_value(&r.canvas).unwrap());
    }
    for d in 5..=15 {
        let r = normalize(Endpoint::TextToVideo, json!({"prompt": "p", "duration": d})).unwrap();
        out.insert(format!("duration {d}"), serde_json::to_value(&r.timing).unwrap());
    }
    golden("enums.json", &Value::Object(out));
    // adaptive is r2v-only.
    assert_eq!(
        normalize(Endpoint::TextToVideo, json!({"prompt": "p", "aspect_ratio": "adaptive"})).unwrap_err().param.as_deref(),
        Some("aspect_ratio")
    );
    // i2v has no aspect_ratio: it is ignored, the canvas follows the image.
    let r = normalize(Endpoint::ImageToVideo, json!({"prompt": "p", "aspect_ratio": "1:1", "image_url": IMG})).unwrap();
    assert_eq!(r.canvas, CanvasSpec::FollowImage { short_edge: 768 });
}

#[test]
fn constraints_are_refused_with_fal_locs() {
    let long = "x".repeat(50_001);
    let r2v = |extra: Value| {
        let mut b = json!({"prompt": "p"});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b
    };
    let urls = |n: usize| Value::Array((0..n).map(|i| format!("https://a.test/{i}")).map(Value::from).collect());
    let cases: Vec<(&str, Endpoint, Value)> = vec![
        ("missing prompt", Endpoint::TextToVideo, json!({})),
        ("empty prompt", Endpoint::TextToVideo, json!({"prompt": ""})),
        ("prompt too long", Endpoint::TextToVideo, json!({"prompt": long})),
        ("prompt not a string", Endpoint::TextToVideo, json!({"prompt": 3})),
        ("duration below 5", Endpoint::TextToVideo, json!({"prompt": "p", "duration": 4})),
        ("duration above 15", Endpoint::TextToVideo, json!({"prompt": "p", "duration": 16})),
        ("duration fractional", Endpoint::TextToVideo, json!({"prompt": "p", "duration": 5.5})),
        ("duration a string", Endpoint::TextToVideo, json!({"prompt": "p", "duration": "5"})),
        ("resolution unknown", Endpoint::TextToVideo, json!({"prompt": "p", "resolution": "720P"})),
        ("aspect unknown", Endpoint::TextToVideo, json!({"prompt": "p", "aspect_ratio": "2:1"})),
        ("seed negative", Endpoint::TextToVideo, json!({"prompt": "p", "seed": -1})),
        ("seed a string", Endpoint::TextToVideo, json!({"prompt": "p", "seed": "1"})),
        ("safety not a bool", Endpoint::TextToVideo, json!({"prompt": "p", "enable_safety_checker": "yes"})),
        ("sync_mode not a bool", Endpoint::TextToVideo, json!({"prompt": "p", "sync_mode": 1})),
        ("expansion not a string", Endpoint::TextToVideo, json!({"prompt": "p", "prompt_expansion_mode": false})),
        ("target audio blank", Endpoint::TextToVideo, json!({"prompt": "p", "target_audio_url": " "})),
        ("image_url not a url", Endpoint::ImageToVideo, json!({"prompt": "p", "image_url": "file:///etc/passwd"})),
        ("too many images", Endpoint::ReferenceToVideo, r2v(json!({"reference_image_urls": urls(10)}))),
        ("too many videos", Endpoint::ReferenceToVideo, r2v(json!({"reference_video_urls": urls(4)}))),
        ("too many audio", Endpoint::ReferenceToVideo, r2v(json!({"reference_audio_urls": urls(4)}))),
        (
            "more than 12 references",
            Endpoint::ReferenceToVideo,
            r2v(json!({"reference_image_urls": urls(9), "reference_video_urls": urls(3), "reference_audio_urls": urls(1)})),
        ),
        ("no references", Endpoint::ReferenceToVideo, json!({"prompt": "p"})),
        ("reference not a list", Endpoint::ReferenceToVideo, r2v(json!({"reference_image_urls": "https://a.test/i.png"}))),
        ("reference item blank", Endpoint::ReferenceToVideo, r2v(json!({"reference_image_urls": ["https://a.test/i.png", ""]}))),
        ("body not an object", Endpoint::TextToVideo, json!(["prompt"])),
        ("bad fal_webhook", Endpoint::TextToVideo, json!({"prompt": "p", "__webhook": true})),
    ];
    let mut out = serde_json::Map::new();
    for (name, e, body) in cases {
        let v = if name == "bad fal_webhook" {
            let err = ep(e).normalize(json!({"prompt": "p"}), &ncx(&[("fal_webhook", "not a url")])).unwrap_err();
            let r = FalProtocol.render_error(&err, &ErrorCtx::default());
            json!({"status": r.status, "error_type": r.header("x-fal-error-type"), "body": r.json_body().unwrap()})
        } else {
            refusal(e, body)
        };
        assert_eq!(v["status"], 422, "{name}: {v}");
        out.insert(name.into(), v);
    }
    golden("refusals.json", &Value::Object(out));
    // Bounds are inclusive; 12 references in total are fine.
    assert!(normalize(Endpoint::TextToVideo, json!({"prompt": "x".repeat(50_000), "duration": 15})).is_ok());
    assert!(normalize(Endpoint::TextToVideo, json!({"prompt": "p", "duration": 5.0})).is_ok());
    assert!(normalize(Endpoint::ReferenceToVideo, r2v(json!({"reference_image_urls": urls(9), "reference_video_urls": urls(3)}))).is_ok());
}

// ------------------------------------------------------------ views

struct Signer;
impl UrlSigner for Signer {
    fn url_for(&self, a: &Artifact, ttl: Duration) -> Url {
        Url::parse(&format!("https://fal.fv.test/files/{}/{}?exp={}&sig=00", a.id, a.file_name, ttl.as_secs())).unwrap()
    }
}

fn resolved(task: Task) -> ResolvedJob {
    ResolvedJob {
        model: "fake-h3-max".into(),
        task,
        prompt: "p".into(),
        negative_prompt: String::new(),
        seed: 1851572118,
        width: 1344,
        height: 768,
        num_frames: 124,
        fps: 24,
        keyframes: vec![],
        references: vec![],
        audio_in: None,
        audio: AudioPlan::Native { rate: 32_000, channels: 2 },
        post: PostProcess::default(),
        sampling: SamplingOverrides::default(),
        tier: Some(Tier::Max),
        recipe: Some("full-dense".into()),
    }
}

fn job(task: Task, endpoint: &str) -> Job {
    let id = JobId("0aa7ecbd-0000-4000-8000-000000000001".parse().unwrap());
    let mut j = Job::new(id, ProtocolId::Fal, RID, resolved(task), now(), Duration::from_secs(86400));
    j.request_echo = json!({"prompt": "p", "model": endpoint});
    j
}

fn succeed(j: &mut Job) {
    j.mark_running(now()).unwrap();
    j.logs.push(LogLine::info("stage: denoise", datetime!(2026-09-27 12:00:01.123 UTC)));
    let a = Artifact {
        id: ArtifactId("a0000000-0000-4000-8000-000000000002".parse().unwrap()),
        mime: "video/mp4".into(),
        file_name: fastvideo_fal::output_file_name(j),
        bytes: 3_554_670,
        location: ArtifactLocation::Local("/tmp/x.mp4".into()),
        width: 1344,
        height: 768,
        frames: 124,
        fps: 24,
        audio: Some((32_000, 2)),
    };
    let stages = [("text", 1.25), ("denoise", 2.5285604159580544), ("video_decode", 0.75), ("encode", 0.25)];
    let metrics = JobMetrics {
        inference_s: Some(2.5285604159580544),
        stage_durations: stages.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
        ..Default::default()
    };
    j.mark_succeeded(datetime!(2026-09-27 12:00:05 UTC), vec![a], metrics).unwrap();
}

fn view_ctx(with_logs: bool) -> (Url, Signer, bool) {
    (Url::parse("https://fal.fv.test").unwrap(), Signer, with_logs)
}

fn cx<'a>(t: &'a (Url, Signer, bool)) -> ViewCtx<'a> {
    ViewCtx { now: now(), urls: &t.1, public_base: &t.0, with_logs: t.2 }
}

fn reply_json(r: &fastvideo_protocol::HttpReply) -> Value {
    let mut headers: Vec<(String, String)> = r.headers.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    headers.sort();
    json!({"status": r.status, "headers": headers, "body": r.json_body().cloned().unwrap_or(Value::Null)})
}

#[test]
fn submit_status_result_bodies() {
    let t = view_ctx(false);
    let tl = view_ctx(true);
    let view = FalView::default();
    let mut out = serde_json::Map::new();

    let mut j = job(Task::T2V, "minimax/h3-max/text-to-video");
    j.queue_position = Some(2);
    out.insert("submit".into(), reply_json(&ep(Endpoint::TextToVideo).submit_reply(&j, &cx(&t))));
    out.insert("status IN_QUEUE".into(), reply_json(&view.status_reply(&j, &cx(&t))));
    out.insert("result IN_QUEUE".into(), reply_json(&view.result_reply(&j, &cx(&t))));

    j.mark_running(now()).unwrap();
    j.logs.push(LogLine::info("Loading model weights...", datetime!(2026-02-17 10:30:01.123 UTC)));
    out.insert("status IN_PROGRESS logs=0".into(), reply_json(&view.status_reply(&j, &cx(&t))));
    out.insert("status IN_PROGRESS logs=1".into(), reply_json(&view.status_reply(&j, &cx(&tl))));

    let mut ok = job(Task::T2V, "minimax/h3-max/text-to-video");
    succeed(&mut ok);
    out.insert("status COMPLETED logs=1".into(), reply_json(&view.status_reply(&ok, &cx(&tl))));
    out.insert("result t2v".into(), reply_json(&view.result_reply(&ok, &cx(&t))));

    let mut r2v = job(Task::Ref2V, "minimax/h3-max/reference-to-video");
    succeed(&mut r2v);
    out.insert("result r2v (seed)".into(), reply_json(&view.result_reply(&r2v, &cx(&t))));

    let mut draft = job(Task::T2V, "minimax/h3-draft/text-to-video");
    draft.resolved.tier = Some(Tier::Draft);
    draft.resolved.recipe = Some("4step-vsa-480p-tae".into());
    succeed(&mut draft);
    out.insert("result draft tier".into(), reply_json(&view.result_reply(&draft, &cx(&t))));
    out.insert("status draft app urls".into(), reply_json(&view.status_reply(&draft, &cx(&t))));

    let mut failed = job(Task::T2V, "minimax/h3-max/text-to-video");
    failed.mark_running(now()).unwrap();
    failed.mark_failed(now(), ApiError::engine_failed("CUDA error: out of memory")).unwrap();
    out.insert("status COMPLETED failed".into(), reply_json(&view.status_reply(&failed, &cx(&t))));
    out.insert("result failed".into(), reply_json(&view.result_reply(&failed, &cx(&t))));

    let mut cancelled = job(Task::T2V, "minimax/h3-max/text-to-video");
    cancelled.mark_cancelled(now()).unwrap();
    out.insert("status COMPLETED cancelled".into(), reply_json(&view.status_reply(&cancelled, &cx(&t))));
    out.insert("result cancelled".into(), reply_json(&view.result_reply(&cancelled, &cx(&t))));

    golden("views.json", &Value::Object(out));

    // The invariants the Python client indexes directly (PY:client.py L711-726).
    for (j, logs) in [(&j, true), (&ok, true), (&failed, true), (&cancelled, true)] {
        let s = status_json(j, &cx(&t));
        assert_eq!(s.get("logs").is_some(), logs, "{s}");
        for k in ["request_id", "response_url", "status_url", "cancel_url", "status"] {
            assert!(s.get(k).is_some(), "{k} missing in {s}");
        }
        assert!(["IN_QUEUE", "IN_PROGRESS", "COMPLETED"].contains(&s["status"].as_str().unwrap()));
    }
    let mut q = job(Task::T2V, "minimax/h3-max/text-to-video");
    q.queue_position = None;
    assert_eq!(status_json(&q, &cx(&t))["queue_position"], 0, "IN_QUEUE always has queue_position");
}

#[test]
fn errors_render_per_kind() {
    let kinds: Vec<(&str, ApiError)> = vec![
        ("invalid", ApiError::invalid_param("duration", "Input should be less than or equal to 15")),
        ("unsupported 1080P", ApiError::unsupported(GapId::H3Refine1080P).with_param("resolution")),
        ("unsupported target audio", ApiError::unsupported(GapId::H3TargetAudio)),
        ("unsupported media", ApiError::unsupported_media("could not decode image").with_param("image_url")),
        ("content filtered", ApiError::content_filtered("flagged").with_param("prompt")),
        ("unauthorized", ApiError::unauthorized("missing credentials")),
        ("forbidden", ApiError::forbidden("no")),
        ("not found", ApiError::not_found("Application \"minimax/h3-draft\" not found")),
        ("already completed", ApiError::already_completed("done")),
        ("conflict", ApiError::conflict("busy")),
        ("payload too large", ApiError::payload_too_large("too big")),
        ("rate limited", ApiError::rate_limited("slow down")),
        ("queue full", ApiError::queue_full("queue is full")),
        ("loading", ApiError::loading("models are loading")),
        ("timeout", ApiError::timeout("took too long")),
        ("cancelled", ApiError::cancelled("Request was cancelled")),
        ("engine failed", ApiError::engine_failed("CUDA error")),
        ("internal", ApiError::internal("bug")),
    ];
    let mut out = serde_json::Map::new();
    for (name, e) in kinds {
        let ecx = ErrorCtx { request_id: Some(RID.into()), route: None, external_id: None };
        let r = fastvideo_serve_kit::handlers::error_reply(&FalProtocol, &e, &ecx);
        out.insert(name.into(), reply_json(&r));
        assert_eq!(r.header("x-fal-error-type"), Some(error_body(&e).1), "every error has X-Fal-Error-Type");
    }
    golden("errors.json", &Value::Object(out));
}

#[test]
fn webhook_bodies() {
    let t = view_ctx(false);
    let mut out = serde_json::Map::new();
    let pending = job(Task::T2V, "minimax/h3-max/text-to-video");
    assert!(webhook_body(&pending, &cx(&t), Duration::from_secs(3600)).is_none(), "only terminal states post");
    let mut ok = job(Task::T2V, "minimax/h3-max/text-to-video");
    succeed(&mut ok);
    out.insert("OK".into(), webhook_body(&ok, &cx(&t), Duration::from_secs(3600)).unwrap());
    let mut failed = job(Task::T2V, "minimax/h3-max/text-to-video");
    failed.mark_failed(now(), ApiError::invalid_param("image_url", "could not fetch the image")).unwrap();
    out.insert("ERROR failed".into(), webhook_body(&failed, &cx(&t), Duration::from_secs(3600)).unwrap());
    let mut cancelled = job(Task::T2V, "minimax/h3-max/text-to-video");
    cancelled.mark_cancelled(now()).unwrap();
    out.insert("ERROR cancelled".into(), webhook_body(&cancelled, &cx(&t), Duration::from_secs(3600)).unwrap());
    golden("webhooks.json", &Value::Object(out));
}

#[test]
fn small_helpers() {
    assert_eq!(fal_timestamp(datetime!(2026-02-17 10:30:01.123456 UTC)), "2026-02-17T10:30:01.123Z");
    let q = |v: &str| vec![("logs".to_string(), v.to_string())];
    for (v, want) in [("1", true), ("true", true), ("True", true), ("0", false), ("false", false), ("False", false)] {
        assert_eq!(logs_param(&q(v)), want, "logs={v}");
    }
    assert!(!logs_param(&[]));
    let j = job(Task::T2V, "minimax/h3-max/text-to-video");
    let name = fastvideo_fal::output_file_name(&j);
    // Named by app (the tier is already a word of `h3-max`).
    assert_eq!(name.len(), 21 + "_minimax-h3-max.mp4".len());
    assert!(name.ends_with("_minimax-h3-max.mp4"), "{name}");
    assert!(name[..21].chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    assert_eq!(name, fastvideo_fal::output_file_name(&j), "stable per job");
    // The tier is appended when the app alias does not name it.
    for (endpoint, tier, want) in [
        ("minimax/h3-turbo/text-to-video", Some(Tier::Turbo), "_minimax-h3-turbo.mp4"),
        ("lightricks/ltx-2.5/text-to-video", Some(Tier::Max), "_ltx-2.5-max.mp4"),
        ("fastvideo/fastwan21-1.3b/text-to-video", None, "_fastwan21-1.3b.mp4"),
    ] {
        let mut t = job(Task::T2V, endpoint);
        t.resolved.tier = tier;
        let name = fastvideo_fal::output_file_name(&t);
        assert!(name.ends_with(want), "{endpoint}: {name}");
    }
    assert_eq!(FalProtocol.id(), ProtocolId::Fal);
    let a = FalProtocol.new_external_id(j.id);
    assert!(uuid::Uuid::parse_str(&a).is_ok() && a != FalProtocol.new_external_id(j.id));
}

/// Behind a gateway `queue` splits into `dispatch` (submit until the worker
/// holds the job and its inputs) and `wait` (then until the engine starts).
#[test]
fn timings_split_queue_into_dispatch_and_wait() {
    let t0 = now();
    let mut j = job(Task::I2V, "minimax/h3-max/image-to-video");
    j.dispatched_at = Some(t0 + time::Duration::milliseconds(1500));
    j.mark_running(t0 + time::Duration::seconds(4)).unwrap();
    let metrics = JobMetrics { inference_s: Some(8.0), ..Default::default() };
    j.mark_succeeded(t0 + time::Duration::seconds(16), vec![], metrics).unwrap();
    let t = fastvideo_fal::queue::timings(&j).unwrap();
    assert_eq!(t["queue"], json!(4.0));
    assert_eq!(t["dispatch"], json!(1.5));
    assert_eq!(t["wait"], json!(2.5));
    assert_eq!(t["total"], json!(12.0));

    // A job that never left its process (or an older worker) reports no split.
    let mut k = job(Task::T2V, "minimax/h3-max/text-to-video");
    k.mark_running(t0 + time::Duration::seconds(1)).unwrap();
    k.mark_succeeded(t0 + time::Duration::seconds(3), vec![], JobMetrics { inference_s: Some(1.0), ..Default::default() })
        .unwrap();
    let t = fastvideo_fal::queue::timings(&k).unwrap();
    assert!(t.get("dispatch").is_none() && t.get("wait").is_none());
    assert_eq!(t["queue"], json!(1.0));

    // The field round-trips and is omitted when unset (stored rows stay as they were).
    let v = serde_json::to_value(&k).unwrap();
    assert!(v.get("dispatched_at").is_none());
    let back: Job = serde_json::from_value(serde_json::to_value(&j).unwrap()).unwrap();
    assert_eq!(back.dispatched_at, j.dispatched_at);
}
