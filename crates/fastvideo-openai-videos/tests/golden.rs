//! Golden conformance (design §7.1): request bodies from the spec are
//! normalized and pinned; `VideoResponse`, list, delete, FastWan and error
//! bodies are rendered from fixed jobs and compared exactly.
//!
//! Fixtures live in `tests/golden/`. `FV_BLESS=1` rewrites the `*.out.json`
//! files (review the diff before committing).

use std::path::PathBuf;
use std::time::Duration;

use fastvideo_openai_videos::error::{body_error, fastapi_error, openai_error, status_of};
use fastvideo_openai_videos::fastwan::{self, FastWanGenerate, FastWanSubmit, FastWanView};
use fastvideo_openai_videos::models::{self, public_names};
use fastvideo_openai_videos::videos::{
    self, apply_model, coerce_form, list_query, merge_extra, parse_request, status_name,
    ListParams, VideosCreate, VideosView,
};
use fastvideo_protocol::{
    Anchor, ApiError, Artifact, ArtifactId, ArtifactLocation, AudioPlan, CanvasSpec, ErrorKind,
    GapId, GenerationRequest, Job, JobId, JobMetrics, JobStatus, JobView, Length, MediaKind,
    ModelCaps, NormalizeCtx, PostProcess, ProtocolId, ReplyBody, ResolvedJob, SamplingOverrides,
    Snap, SubmitEndpoint, Task, Tier, UrlSigner, ViewCtx,
};
use serde_json::{json, Map, Value};
use time::macros::datetime;
use time::OffsetDateTime;

// ---------------------------------------------------------------- helpers

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn fixture(name: &str) -> Value {
    let p = golden_dir().join(name);
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Compares `got` with `tests/golden/<name>`; `FV_BLESS=1` writes it.
fn check(name: &str, got: &Value) {
    let p = golden_dir().join(name);
    if std::env::var_os("FV_BLESS").is_some() {
        std::fs::write(&p, serde_json::to_string_pretty(got).unwrap() + "\n").unwrap();
        return;
    }
    let want = fixture(name);
    assert_eq!(
        got,
        &want,
        "{name}:\n{}",
        serde_json::to_string_pretty(got).unwrap()
    );
}

struct Signer;
impl UrlSigner for Signer {
    fn url_for(&self, a: &Artifact, _ttl: Duration) -> url::Url {
        url::Url::parse(&format!(
            "https://r2.example.com/fv-media/{}?X-Amz-Signature=x",
            a.file_name
        ))
        .unwrap()
    }
}

const NOW: OffsetDateTime = datetime!(2026-09-27 12:00:00 UTC);

fn vcx<'a>(base: &'a url::Url) -> ViewCtx<'a> {
    ViewCtx {
        now: NOW,
        urls: &Signer,
        public_base: base,
        with_logs: false,
    }
}

fn ncx() -> NormalizeCtx {
    NormalizeCtx::new(NOW)
}

fn normalize(body: Value) -> Result<GenerationRequest, ApiError> {
    let Value::Object(mut m) = body else {
        panic!("object")
    };
    merge_extra(&mut m)?;
    VideosCreate.normalize(parse_request(m)?, &ncx())
}

fn resolved(model: &str, w: u32, h: u32, frames: u32, fps: u32) -> ResolvedJob {
    ResolvedJob {
        model: model.into(),
        task: Task::T2V,
        prompt: "A red fox runs through fresh snow at dawn".into(),
        negative_prompt: String::new(),
        seed: 1000,
        width: w,
        height: h,
        num_frames: frames,
        fps,
        keyframes: vec![],
        references: vec![],
        audio_in: None,
        audio: AudioPlan::Native {
            rate: 32_000,
            channels: 2,
        },
        post: PostProcess::default(),
        sampling: SamplingOverrides::default(),
        tier: Some(Tier::Turbo),
        recipe: Some("4step-vsa".into()),
        edit: None,
    }
}

fn job(proto: ProtocolId, ext: &str, echo: Value) -> Job {
    let id = JobId(uuid::Uuid::from_u128(
        0x0123_4567_89ab_cdef_0123_4567_89ab_cdef,
    ));
    let mut j = Job::new(
        id,
        proto,
        ext,
        resolved("fake-h3-turbo", 1344, 768, 124, 24),
        NOW,
        Duration::from_secs(86_400),
    );
    j.request_echo = echo;
    j
}

fn fv_job() -> Job {
    job(
        ProtocolId::OpenAiVideos,
        "video_gen_0123456789abcdef0123456789abcdef",
        json!({"model": "h3-turbo", "prompt": "A red fox runs through fresh snow at dawn", "seconds": "5", "size": "1344x768"}),
    )
}

fn completed(mut j: Job) -> Job {
    j.mark_running(NOW + Duration::from_secs(1)).unwrap();
    j.set_step(2, 4);
    let mut metrics = JobMetrics {
        inference_s: Some(21.75),
        peak_memory_mb: Some(61_440.0),
        ..JobMetrics::default()
    };
    metrics.stage_durations.insert("text_encode".into(), 1.5);
    metrics.stage_durations.insert("denoise".into(), 21.75);
    metrics.stage_durations.insert("decode".into(), 3.25);
    let art = Artifact {
        id: ArtifactId(uuid::Uuid::from_u128(7)),
        mime: "video/mp4".into(),
        file_name: "video.mp4".into(),
        bytes: 1234,
        location: ArtifactLocation::Local("/state/artifacts/7/video.mp4".into()),
        width: 1344,
        height: 768,
        frames: 124,
        fps: 24,
        audio: Some((32_000, 2)),
    };
    j.mark_succeeded(NOW + Duration::from_secs(30), vec![art], metrics)
        .unwrap();
    j
}

fn failed(mut j: Job) -> Job {
    j.mark_running(NOW + Duration::from_secs(1)).unwrap();
    j.mark_failed(
        NOW + Duration::from_secs(5),
        ApiError::engine_failed("CUDA out of memory"),
    )
    .unwrap();
    j
}

fn json_of(r: &fastvideo_protocol::HttpReply) -> Value {
    r.json_body().cloned().unwrap_or(Value::Null)
}

// ---------------------------------------------------------------- requests

#[test]
fn fastvideo_request_goldens() {
    // Each fixture is a request body from the spec (minimax-fastvideo §2.1,
    // §2.5); the normalized `GenerationRequest` is pinned next to it.
    for name in [
        "create_t2v_cookbook",
        "create_fl2va_adapter",
        "create_ref2va_mixed",
        "create_aspect_frames",
    ] {
        let body = fixture(&format!("fastvideo/{name}.json"));
        let req = normalize(body).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        check(
            &format!("fastvideo/{name}.out.json"),
            &serde_json::to_value(&req).unwrap(),
        );
    }
}

#[test]
fn fastvideo_request_semantics() {
    // fl2va with one image is first-frame I2V; with two, first + last.
    let r = normalize(json!({"prompt": "p", "task": "fl2va", "image_reference": {"image_url": "https://x.test/a.png"}})).unwrap();
    assert_eq!(
        (r.task, r.keyframes.len(), r.keyframes[0].at),
        (Task::I2V, 1, Anchor::First)
    );
    let r = normalize(json!({"prompt": "p", "task": "fl2va", "image_reference": [{"image_url": "https://x.test/a.png"}, {"image_url": "https://x.test/b.png"}]})).unwrap();
    assert_eq!(r.task, Task::Keyframes);
    assert_eq!(
        r.keyframes.iter().map(|k| k.at).collect::<Vec<_>>(),
        [Anchor::First, Anchor::Last]
    );
    // ref2va reorders to images, videos, audio.
    let r = normalize(json!({"prompt": "p", "task": "ref2va",
        "audio_reference": [{"audio_url": "https://x.test/a.wav"}],
        "video_reference": {"video_url": "https://x.test/v.mp4"},
        "image_reference": [{"image_url": "https://x.test/i.png"}]}))
    .unwrap();
    assert_eq!(
        r.references.iter().map(|x| x.kind).collect::<Vec<_>>(),
        [MediaKind::Image, MediaKind::Video, MediaKind::Audio]
    );
    // Legacy single image.
    let r = normalize(json!({"prompt": "p", "input_reference": "https://x.test/a.png"})).unwrap();
    assert_eq!(r.task, Task::I2V);
    // Precedence: size > width/height > video_params > aspect_ratio.
    let r = normalize(
        json!({"prompt": "p", "size": "768x1344", "width": 1, "height": 1, "aspect_ratio": "16:9"}),
    )
    .unwrap();
    assert_eq!(
        r.canvas,
        CanvasSpec::Exact {
            width: 768,
            height: 1344
        }
    );
    let r = normalize(json!({"prompt": "p", "video_params": {"width": 832, "height": 480, "num_frames": 81, "fps": 16}})).unwrap();
    assert_eq!(
        r.canvas,
        CanvasSpec::Exact {
            width: 832,
            height: 480
        }
    );
    assert_eq!(
        (r.timing.length.clone(), r.timing.fps),
        (
            Length::Frames {
                value: 81,
                snap: Snap::Exact
            },
            Some(16)
        )
    );
    // seconds as an int or a digit string; num_frames wins over seconds.
    for s in [json!(6), json!("6")] {
        let r = normalize(json!({"prompt": "p", "seconds": s})).unwrap();
        assert_eq!(
            r.timing.length,
            Length::Seconds {
                value: 6.0,
                snap: Snap::AlignUp
            }
        );
    }
    let r = normalize(json!({"prompt": "p", "seconds": "6", "num_frames": 141})).unwrap();
    assert_eq!(
        r.timing.length,
        Length::Frames {
            value: 141,
            snap: Snap::Exact
        }
    );
    // extra_body / extra_json merge into the top level.
    let r = normalize(json!({"prompt": "p", "extra_body": {"seed": 7}, "extra_json": {"negative_prompt": "blur"}})).unwrap();
    assert_eq!(
        (r.seed, r.negative_prompt.as_deref()),
        (Some(7), Some("blur"))
    );
    // generate_sound / quality / user are accepted no-ops.
    let r = normalize(json!({"prompt": "p", "generate_sound": true, "quality": "hd", "user": "u"}))
        .unwrap();
    assert_eq!(r.accepted_noop, ["generate_sound", "quality", "user"]);
    // file_id refs become provider files (refused later as a gap).
    let r = normalize(json!({"prompt": "p", "image_reference": {"file_id": "file-abc"}})).unwrap();
    assert!(matches!(
        r.keyframes[0].image,
        fastvideo_protocol::MediaRef::ProviderFile(_)
    ));
}

#[test]
fn fastvideo_request_refusals() {
    let cases: Vec<(Value, &str, Option<&str>)> = vec![
        (
            json!({"prompt": "p", "bogus": 1}),
            "invalid_request",
            Some("bogus"),
        ),
        (json!({"model": "m"}), "invalid_request", Some("prompt")),
        (json!({"prompt": "  "}), "invalid_request", Some("prompt")),
        (json!({"prompt": "p", "n": 2}), "invalid_request", Some("n")),
        (
            json!({"prompt": "p", "num_outputs_per_prompt": 3}),
            "invalid_request",
            Some("num_outputs_per_prompt"),
        ),
        (
            json!({"prompt": "p", "n": 11}),
            "invalid_request",
            Some("n"),
        ),
        (
            json!({"prompt": "p", "seconds": "05"}),
            "invalid_request",
            Some("seconds"),
        ),
        (
            json!({"prompt": "p", "seconds": 0}),
            "invalid_request",
            Some("seconds"),
        ),
        (
            json!({"prompt": "p", "size": "1344*768"}),
            "invalid_request",
            Some("size"),
        ),
        (
            json!({"prompt": "p", "width": 1344}),
            "invalid_request",
            Some("width"),
        ),
        (
            json!({"prompt": "p", "short_edge": 768}),
            "invalid_request",
            Some("short_edge"),
        ),
        (
            json!({"prompt": "p", "aspect_ratio": "wide"}),
            "invalid_request",
            Some("aspect_ratio"),
        ),
        (
            json!({"prompt": "p", "task": "i2v"}),
            "invalid_request",
            Some("task"),
        ),
        (
            json!({"prompt": "p", "task": "t2va", "image_reference": {"image_url": "https://x.test/a.png"}}),
            "invalid_request",
            Some("task"),
        ),
        (
            json!({"prompt": "p", "task": "fl2va"}),
            "invalid_request",
            Some("task"),
        ),
        (
            json!({"prompt": "p", "task": "ref2va", "audio_reference": {"audio_url": "https://x.test/a.wav"}}),
            "invalid_request",
            Some("audio_reference"),
        ),
        (
            json!({"prompt": "p", "input_reference": "https://x.test/a.png", "reference_url": "https://x.test/b.png"}),
            "invalid_request",
            Some("reference_url"),
        ),
        (
            json!({"prompt": "p", "image_reference": {"image_url": "/srv/local.png"}}),
            "invalid_request",
            Some("image_reference"),
        ),
        (
            json!({"prompt": "p", "image_reference": {}}),
            "invalid_request",
            Some("image_reference"),
        ),
        (
            json!({"prompt": "p", "video_url": "https://x.test/v.mp4"}),
            "invalid_request",
            Some("video_url"),
        ),
        (
            json!({"prompt": "p", "true_cfg_scale": 4.0}),
            "invalid_request",
            Some("true_cfg_scale"),
        ),
        (
            json!({"prompt": "p", "enable_teacache": true}),
            "invalid_request",
            Some("enable_teacache"),
        ),
        (
            json!({"prompt": "p", "enable_frame_interpolation": true}),
            "invalid_request",
            Some("enable_frame_interpolation"),
        ),
        (
            json!({"prompt": "p", "max_sequence_length": 512}),
            "invalid_request",
            Some("max_sequence_length"),
        ),
        (
            json!({"prompt": "p", "sound_duration": 5.0}),
            "invalid_request",
            Some("sound_duration"),
        ),
        (
            json!({"prompt": "p", "start_time_seconds": 1.0}),
            "invalid_request",
            Some("start_time_seconds"),
        ),
        (
            json!({"prompt": "p", "lora": {"name": "x", "path": "/l"}}),
            "unsupported",
            Some("lora"),
        ),
        (
            json!({"prompt": "p", "extra_params": {"vsa_mode": "x"}}),
            "invalid_request",
            Some("extra_params"),
        ),
        (
            json!({"prompt": "p", "extra_params": {"nope": 1}}),
            "invalid_request",
            Some("extra_params"),
        ),
        (
            json!({"prompt": "p", "quality": "ultra"}),
            "invalid_request",
            Some("quality"),
        ),
        (
            json!({"prompt": "p", "num_inference_steps": 0}),
            "invalid_request",
            Some("num_inference_steps"),
        ),
        (
            json!({"prompt": "p", "guidance_scale": 21.0}),
            "invalid_request",
            Some("guidance_scale"),
        ),
        (
            json!({"prompt": "p", "seed": -1}),
            "invalid_request",
            Some("seed"),
        ),
        (
            json!({"prompt": "p", "extra_body": 3}),
            "invalid_request",
            Some("extra_body"),
        ),
    ];
    for (body, kind, param) in cases {
        let e = normalize(body.clone()).expect_err(&body.to_string());
        assert_eq!(
            (e.kind.code(), e.param.as_deref()),
            (kind, param),
            "{body}: {e:?}"
        );
        assert_eq!(
            status_of(e.kind),
            400,
            "validation errors are 400, never 422: {body}"
        );
    }
    // enable_teacache: false asks for nothing.
    assert!(normalize(json!({"prompt": "p", "enable_teacache": false})).is_ok());
}

#[test]
fn form_bodies_are_coerced() {
    let mut m: Map<String, Value> = [
        ("prompt", "p"),
        ("seconds", "5"),
        ("width", "1344"),
        ("height", "768"),
        ("seed", "42"),
        ("guidance_scale", "1.0"),
        ("generate_sound", "true"),
        (
            "image_reference",
            r#"[{"image_url":"https://x.test/a.png"}]"#,
        ),
        ("extra_body", r#"{"fps":24}"#),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), Value::String(v.to_owned())))
    .collect();
    coerce_form(&mut m).unwrap();
    merge_extra(&mut m).unwrap();
    let r = VideosCreate
        .normalize(parse_request(m).unwrap(), &ncx())
        .unwrap();
    assert_eq!(
        r.canvas,
        CanvasSpec::Exact {
            width: 1344,
            height: 768
        }
    );
    assert_eq!(
        (r.seed, r.timing.fps, r.task),
        (Some(42), Some(24), Task::I2V)
    );
    assert_eq!(r.sampling.guidance, Some(1.0));
    let mut bad: Map<String, Value> = [(
        "image_reference".to_owned(),
        Value::String("{not json".into()),
    )]
    .into_iter()
    .collect();
    assert_eq!(
        coerce_form(&mut bad).unwrap_err().param.as_deref(),
        Some("image_reference")
    );
}

#[test]
fn model_rules_after_resolution() {
    let h3 = ModelCaps::h3("fake-h3-turbo", false);
    // Aspect without short_edge takes the model's default tier.
    let mut r = normalize(json!({"prompt": "p", "aspect_ratio": "9:16"})).unwrap();
    apply_model(&mut r, &h3, false).unwrap();
    assert_eq!(
        r.canvas,
        CanvasSpec::Aspect {
            ratio: fastvideo_protocol::Ratio::R9_16,
            short_edge: 768
        }
    );
    assert_eq!(r.model, "fake-h3-turbo");
    // FastH3's fixed guidance 1 and a negative prompt are no-ops on H3.
    let mut r =
        normalize(json!({"prompt": "p", "guidance_scale": 1.0, "negative_prompt": "blurry"}))
            .unwrap();
    apply_model(&mut r, &h3, false).unwrap();
    assert_eq!(
        (r.sampling.guidance, r.negative_prompt.clone()),
        (None, None)
    );
    assert_eq!(r.accepted_noop, ["guidance_scale", "negative_prompt"]);
    // Other guidance values stay (and negotiate refuses them).
    let mut r = normalize(json!({"prompt": "p", "guidance_scale": 3.0})).unwrap();
    apply_model(&mut r, &h3, false).unwrap();
    assert_eq!(r.sampling.guidance, Some(3.0));
    // FastVideo's H3 range starts at 5 s (124 frames at 24 fps).
    for (body, param) in [
        (json!({"prompt": "p", "seconds": 4}), "seconds"),
        (json!({"prompt": "p", "num_frames": 107}), "num_frames"),
    ] {
        let mut r = normalize(body).unwrap();
        let e = apply_model(&mut r, &h3, false).unwrap_err();
        assert_eq!(
            (e.kind, e.param.as_deref()),
            (ErrorKind::InvalidRequest, Some(param))
        );
    }
    let mut r = normalize(json!({"prompt": "p", "seconds": 5})).unwrap();
    apply_model(&mut r, &h3, true).unwrap();
    // `task` is H3-only.
    let mut wan = ModelCaps::h3("wan", false);
    wan.family = fastvideo_protocol::Family::Wan;
    let mut r = normalize(json!({"prompt": "p", "task": "t2va"})).unwrap();
    assert_eq!(
        apply_model(&mut r, &wan, true)
            .unwrap_err()
            .param
            .as_deref(),
        Some("task")
    );
}

// ---------------------------------------------------------------- responses

#[test]
fn video_response_goldens() {
    let base = url::Url::parse("http://fv.test").unwrap();
    let cx = vcx(&base);
    check(
        "fastvideo/video_queued.out.json",
        &json_of(&VideosCreate.submit_reply(&fv_job(), &cx)),
    );
    let mut running = fv_job();
    running.mark_running(NOW).unwrap();
    running.set_step(1, 4);
    check(
        "fastvideo/video_in_progress.out.json",
        &json_of(&VideosView.status_reply(&running, &cx)),
    );
    let done = completed(fv_job());
    check(
        "fastvideo/video_completed.out.json",
        &json_of(&VideosView.status_reply(&done, &cx)),
    );
    let mut bad = failed(fv_job());
    bad.external_id = "video_gen_fedcba9876543210fedcba9876543210".into();
    let r = VideosView.status_reply(&bad, &cx);
    assert_eq!(r.status, 200, "a failed job is 200 with status failed");
    check("fastvideo/video_failed.out.json", &json_of(&r));
    check(
        "fastvideo/list.out.json",
        &videos::list_body(&[done.clone(), bad], false, &cx),
    );
    check(
        "fastvideo/list_empty.out.json",
        &videos::list_body(&[], false, &cx),
    );

    // The VideoResponse keys are exactly the spec's (protocol.py:213-234),
    // plus `metadata` for the tier and recipe (design §0.3/§0.6).
    let spec = fixture("fastvideo/video_response_keys.json");
    let got = json_of(&VideosView.status_reply(&done, &cx));
    let mut keys: Vec<&str> = got
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .filter(|k| *k != "metadata")
        .collect();
    keys.sort();
    let mut want: Vec<&str> = spec
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    want.sort();
    assert_eq!(keys, want);
}

#[test]
fn draft_results_are_marked() {
    let mut j = fv_job();
    j.resolved.tier = Some(Tier::Draft);
    j.resolved.recipe = Some("fasth3-4step-vsa-480p-taeh3".into());
    let base = url::Url::parse("http://fv.test").unwrap();
    let v = videos::video_response(&j, &vcx(&base));
    assert_eq!(
        v["metadata"],
        json!({"tier": "draft", "quality_gate": false, "recipe": "fasth3-4step-vsa-480p-taeh3"})
    );
    assert_eq!(fastwan::status_body(&j)["metadata"]["quality_gate"], false);
    j.resolved.tier = None;
    j.resolved.recipe = None;
    assert!(videos::video_response(&j, &vcx(&base))
        .get("metadata")
        .is_none());
}

#[test]
fn content_replies() {
    let base = url::Url::parse("http://fv.test").unwrap();
    let cx = vcx(&base);
    let r = VideosView.result_reply(&fv_job(), &cx);
    assert_eq!(r.status, 404);
    check("fastvideo/content_in_progress.out.json", &json_of(&r));
    let r = VideosView.result_reply(&failed(fv_job()), &cx);
    assert_eq!(r.status, 422);
    check("fastvideo/content_failed.out.json", &json_of(&r));
    let r = VideosView.result_reply(&completed(fv_job()), &cx);
    assert_eq!(
        r.body,
        ReplyBody::File {
            path: "/state/artifacts/7/video.mp4".into(),
            mime: "video/mp4".into()
        }
    );
    // R2/S3 artifacts redirect to the presigned URL.
    let mut j = completed(fv_job());
    j.artifacts[0].location = ArtifactLocation::Object {
        bucket: "fv-media".into(),
        key: "a/video.mp4".into(),
    };
    let r = VideosView.result_reply(&j, &cx);
    assert_eq!(
        (r.status, r.header("location")),
        (
            302,
            Some("https://r2.example.com/fv-media/video.mp4?X-Amz-Signature=x")
        )
    );
}

#[test]
fn status_mapping() {
    use JobStatus::*;
    let table = [
        (Queued, "queued", "queued"),
        (Running, "in_progress", "processing"),
        (Succeeded, "completed", "completed"),
        (Failed, "failed", "failed"),
        (Cancelled, "failed", "failed"),
    ];
    for (s, fv, fw) in table {
        assert_eq!(status_name(s), fv);
        assert_eq!(fastwan::status_name(s), fw);
    }
    // Progress is an integer percentage (FastVideo sends only 0 or 100).
    let base = url::Url::parse("http://fv.test").unwrap();
    let mut j = fv_job();
    j.mark_running(NOW).unwrap();
    j.set_step(1, 3);
    assert_eq!(videos::video_response(&j, &vcx(&base))["progress"], 33);
    assert_eq!(
        videos::video_response(&completed(fv_job()), &vcx(&base))["progress"],
        100
    );
}

#[test]
fn error_mapping() {
    // design §4.6: FastVideo and FastWan share one status table.
    let table: Vec<(ApiError, u16)> = vec![
        (ApiError::invalid_param("size", "bad"), 400),
        (
            ApiError::unsupported(GapId::PerRequestSteps).with_param("num_inference_steps"),
            400,
        ),
        (ApiError::content_filtered("no"), 400),
        (ApiError::unauthorized("who"), 401),
        (ApiError::forbidden("no"), 403),
        (ApiError::not_found("gone"), 404),
        (ApiError::already_completed("done"), 409),
        (ApiError::payload_too_large("big"), 413),
        (ApiError::unsupported_media("png?"), 415),
        (ApiError::queue_full("busy"), 429),
        (ApiError::rate_limited("slow"), 429),
        (ApiError::internal("x"), 500),
        (ApiError::engine_failed("x"), 500),
        (ApiError::loading("warming"), 503),
        (ApiError::timeout("late"), 504),
    ];
    for (e, status) in &table {
        let o = openai_error(e);
        let f = fastapi_error(e);
        assert_eq!((o.status, f.status), (*status, *status), "{e:?}");
        let body = json_of(&o);
        assert_eq!(body["error"]["code"], *status);
        let ty = if *status >= 500 {
            "server_error"
        } else {
            "invalid_request_error"
        };
        assert_eq!(body["error"]["type"], ty);
        assert_eq!(json_of(&f), json!({"detail": e.message}));
    }
    check(
        "fastvideo/error_400.out.json",
        &json_of(&openai_error(&ApiError::invalid_param(
            "duration",
            "length must be within 124..=362 frames",
        ))),
    );
    check(
        "fastvideo/error_gap.out.json",
        &json_of(&openai_error(
            &ApiError::unsupported(GapId::PerRequestSteps).with_param("num_inference_steps"),
        )),
    );
    let l = openai_error(&ApiError::loading("models are loading"));
    assert_eq!(l.header("retry-after"), Some("1"));
    check("fastvideo/error_503.out.json", &json_of(&l));
    check(
        "fastwan/error_400.out.json",
        &json_of(&fastapi_error(&ApiError::invalid_param(
            "num_frames",
            "num_frames 50 must be one of 49..=121 frames",
        ))),
    );
    assert_eq!(
        fastapi_error(&ApiError::loading("warming")).header("retry-after"),
        Some("1")
    );
    // serde errors name the field.
    let e = serde_json::from_value::<fastvideo_openai_videos::VideoGenerationRequest>(
        json!({"prompt": "p", "nope": 1}),
    )
    .unwrap_err();
    assert_eq!(body_error(&e).param.as_deref(), Some("nope"));
}

// ---------------------------------------------------------------- FastWan

#[test]
fn fastwan_goldens() {
    // The body fastwan_link.py:_generate sends, at the client defaults.
    let body: FastWanGenerate = serde_json::from_value(fixture("fastwan/generate.json")).unwrap();
    let ep = FastWanSubmit {
        model: Some("fastwan-5b".into()),
    };
    let req = ep.normalize(body, &ncx()).unwrap();
    check(
        "fastwan/generate.out.json",
        &serde_json::to_value(&req).unwrap(),
    );
    assert!(FastWanSubmit { model: None }
        .normalize(
            FastWanGenerate {
                prompt: "p".into(),
                ..Default::default()
            },
            &ncx()
        )
        .is_err());
    // Unknown fields are ignored (FastAPI default).
    let _: FastWanGenerate = serde_json::from_value(json!({"prompt": "p", "extra": 1})).unwrap();

    let base = url::Url::parse("http://fv.test").unwrap();
    let cx = vcx(&base);
    let mk = || {
        job(
            ProtocolId::FastWan,
            "0e7a5f7c-4d7b-4f43-9d3a-6f0d0c1c2b3a",
            json!({"prompt": "p"}),
        )
    };
    check(
        "fastwan/generate_reply.out.json",
        &json_of(&ep.submit_reply(&mk(), &cx)),
    );
    let mut running = mk();
    running.mark_running(NOW).unwrap();
    check(
        "fastwan/status_processing.out.json",
        &json_of(&FastWanView.status_reply(&running, &cx)),
    );
    let done = completed(mk());
    check(
        "fastwan/status_completed.out.json",
        &json_of(&FastWanView.status_reply(&done, &cx)),
    );
    let bad = failed(mk());
    check(
        "fastwan/status_failed.out.json",
        &json_of(&FastWanView.status_reply(&bad, &cx)),
    );
    assert!(
        json_of(&FastWanView.status_reply(&bad, &cx))["error"].is_string(),
        "error is a string"
    );
    assert_eq!(
        FastWanView.result_reply(&done, &cx).body,
        ReplyBody::File {
            path: "/state/artifacts/7/video.mp4".into(),
            mime: "video/mp4".into()
        }
    );
    assert_eq!(FastWanView.result_reply(&bad, &cx).status, 422);
    assert_eq!(FastWanView.result_reply(&mk(), &cx).status, 404);
}

// ---------------------------------------------------------------- models

#[test]
fn model_cards() {
    let mut a = ModelCaps::h3("fake-h3-turbo", false).with_tier(Tier::Turbo, "4step-vsa");
    a.served_names = vec!["fasth3".into()];
    let b = ModelCaps::h3("fake-h3-max", true).with_tier(Tier::Max, "full-dense");
    let models = vec![a, b];
    assert_eq!(
        public_names(&models),
        [
            ("fasth3", "fake-h3-turbo"),
            ("fake-h3-max", "fake-h3-max"),
            ("h3-turbo", "fake-h3-turbo"),
            ("h3-max", "fake-h3-max")
        ]
        .map(|(a, b)| (a.to_owned(), b.to_owned()))
    );
    check(
        "fastvideo/models.out.json",
        &models::list_body(&models, 1_700_000_000),
    );
    let r = models::get_reply(&models, "h3-turbo", 1_700_000_000);
    assert_eq!(r.status, 200);
    assert_eq!(json_of(&r)["root"], "fake-h3-turbo");
    let r = models::get_reply(&models, "sora-2", 1_700_000_000);
    assert_eq!(
        (r.status, json_of(&r)["error"]["code"].clone()),
        (404, json!(404))
    );
    assert_eq!(
        models::model_info_body(&models, None),
        json!({"model_path": "fake-h3-turbo", "served_model_name": "fasth3", "lora": null})
    );
    assert_eq!(
        models::default_model(&models, Some("h3-max")).as_deref(),
        Some("h3-max")
    );
}

#[test]
fn list_params() {
    let q = list_query(&ListParams::default()).unwrap();
    assert_eq!(
        (q.limit, q.order, q.protocol),
        (
            20,
            fastvideo_protocol::SortOrder::Desc,
            Some(ProtocolId::OpenAiVideos)
        )
    );
    let q = list_query(&ListParams {
        after: Some("video_gen_x".into()),
        limit: Some("100".into()),
        order: Some("asc".into()),
    })
    .unwrap();
    assert_eq!(
        (q.limit, q.order, q.after.as_deref()),
        (100, fastvideo_protocol::SortOrder::Asc, Some("video_gen_x"))
    );
    for (l, o) in [
        (Some("0"), None),
        (Some("101"), None),
        (Some("x"), None),
        (None, Some("up")),
    ] {
        let p = ListParams {
            after: None,
            limit: l.map(Into::into),
            order: o.map(Into::into),
        };
        assert_eq!(list_query(&p).unwrap_err().kind, ErrorKind::InvalidRequest);
    }
}

#[test]
fn delete_reply_shape() {
    // `{id, deleted: true, object: "video.deleted"}` (OpenAI DeleteVideo).
    let want = fixture("fastvideo/deleted.json");
    assert_eq!(
        want,
        json!({"id": want["id"], "deleted": true, "object": "video.deleted"})
    );
}

#[test]
fn ids() {
    use fastvideo_protocol::BatchProtocol;
    let id = JobId(uuid::Uuid::from_u128(0xabc));
    let fv = fastvideo_openai_videos::OpenAiVideos.new_external_id(id);
    assert!(
        fv.starts_with("video_gen_")
            && fv.len() == 42
            && fv[10..].bytes().all(|b| b.is_ascii_hexdigit())
    );
    let fw = fastvideo_openai_videos::FastWanApi.new_external_id(id);
    assert!(uuid::Uuid::parse_str(&fw).is_ok());
}
