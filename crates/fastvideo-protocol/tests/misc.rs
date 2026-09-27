//! Error mapping, HTTP reply helpers and the protocol traits, streaming types
//! and A/V buffers.

mod common;

use std::time::Duration;

use common::*;
use fastvideo_protocol::*;
use serde_json::json;
use time::macros::datetime;

// ---- errors ------------------------------------------------------------------------

#[test]
fn canonical_statuses() {
    let table = [
        (ErrorKind::InvalidRequest, 400),
        (ErrorKind::Unsupported(GapId::Lora), 400),
        (ErrorKind::Unauthorized, 401),
        (ErrorKind::Forbidden, 403),
        (ErrorKind::NotFound, 404),
        (ErrorKind::AlreadyCompleted, 409),
        (ErrorKind::Conflict, 409),
        (ErrorKind::Cancelled, 409),
        (ErrorKind::PayloadTooLarge, 413),
        (ErrorKind::UnsupportedMedia, 415),
        (ErrorKind::ContentFiltered, 422),
        (ErrorKind::RateLimited, 429),
        (ErrorKind::QueueFull, 429),
        (ErrorKind::Loading, 503),
        (ErrorKind::Timeout, 504),
        (ErrorKind::EngineFailed, 500),
        (ErrorKind::Internal, 500),
    ];
    for (k, s) in table {
        assert_eq!(k.http_status(), s, "{k:?}");
        assert_eq!(k.is_client_error(), s < 500, "{k:?}");
        assert_eq!(
            k.is_retryable(),
            matches!(
                k,
                ErrorKind::RateLimited
                    | ErrorKind::QueueFull
                    | ErrorKind::Loading
                    | ErrorKind::Timeout
            )
        );
    }
}

#[test]
fn constructors_and_display() {
    let cases = [
        (ApiError::invalid("m"), ErrorKind::InvalidRequest),
        (ApiError::unauthorized("m"), ErrorKind::Unauthorized),
        (ApiError::forbidden("m"), ErrorKind::Forbidden),
        (ApiError::not_found("m"), ErrorKind::NotFound),
        (
            ApiError::already_completed("m"),
            ErrorKind::AlreadyCompleted,
        ),
        (ApiError::conflict("m"), ErrorKind::Conflict),
        (ApiError::payload_too_large("m"), ErrorKind::PayloadTooLarge),
        (
            ApiError::unsupported_media("m"),
            ErrorKind::UnsupportedMedia,
        ),
        (ApiError::content_filtered("m"), ErrorKind::ContentFiltered),
        (ApiError::rate_limited("m"), ErrorKind::RateLimited),
        (ApiError::queue_full("m"), ErrorKind::QueueFull),
        (ApiError::timeout("m"), ErrorKind::Timeout),
        (ApiError::cancelled("m"), ErrorKind::Cancelled),
        (ApiError::engine_failed("m"), ErrorKind::EngineFailed),
        (ApiError::internal("m"), ErrorKind::Internal),
        (
            ApiError::unsupported_msg(GapId::Lora, "m"),
            ErrorKind::Unsupported(GapId::Lora),
        ),
    ];
    for (e, k) in cases {
        assert_eq!(
            (e.kind, e.message.as_str(), e.to_string()),
            (k, "m", "m".to_string())
        );
        assert_eq!(e.param, None);
    }
    let e = ApiError::invalid_param("size", "bad").with_retry_after(3);
    assert_eq!(
        (e.param.as_deref(), e.retry_after_s, e.http_status()),
        (Some("size"), Some(3), 400)
    );
    assert_eq!(ApiError::loading("x").retry_after_s, Some(1));
    assert_eq!(ApiError::invalid("x").gap(), None);
    let boxed: Box<dyn std::error::Error> = Box::new(ApiError::internal("e"));
    assert_eq!(boxed.to_string(), "e");
}

#[test]
fn gap_packages_match_design_4_7() {
    let want = [
        (GapId::H3FourSeconds, Some("E3")),
        (GapId::H3Resolution2K, None),
        (GapId::H3Refine1080P, None),
        (GapId::H3TargetAudio, Some("E10")),
        (GapId::H3Ref2vaNotLoaded, Some("E11")),
        (GapId::LtxKeyframes, Some("E9")),
        (GapId::Ltx25I2V, Some("E5")),
        (GapId::LtxAutoDuration, None),
        (GapId::LtxCameraMotion, None),
        (GapId::LtxFps, Some("E4")),
        (GapId::LtxEndpoint, None),
        (GapId::ProviderFiles, None),
        (GapId::PerRequestSteps, None),
        (GapId::Lora, None),
    ];
    assert_eq!(want.len(), GapId::ALL.len());
    for (g, wp) in want {
        assert_eq!(g.work_package(), wp, "{g:?}");
        assert_eq!(g.is_permanent(), wp.is_none());
        assert!(!g.default_message().is_empty());
        assert_eq!(g.to_string(), g.code());
        let e = ApiError::unsupported(g);
        assert_eq!(
            (e.gap(), e.message.as_str()),
            (Some(g), g.default_message())
        );
    }
    assert_eq!(
        GapId::LtxEndpoint.default_message(),
        "endpoint not available for the account"
    );
    assert_eq!(GapId::LtxFps.param(), Some("fps"));
    assert_eq!(GapId::LtxEndpoint.param(), None);
}

// ---- requests ----------------------------------------------------------------------

#[test]
fn ratio_parse() {
    assert_eq!("16:9".parse::<Ratio>().unwrap(), Ratio::R16_9);
    assert_eq!(" 9 : 16 ".parse::<Ratio>().unwrap(), Ratio::R9_16);
    assert_eq!(Ratio::R4_3.to_string(), "4:3");
    assert!((Ratio::R16_9.value() - 16.0 / 9.0).abs() < 1e-12);
    for bad in ["16x9", "0:9", "a:b", "", "16:", "adaptive"] {
        let e = bad.parse::<Ratio>().unwrap_err();
        assert_eq!(e.param.as_deref(), Some("aspect_ratio"), "{bad}");
    }
}

#[test]
fn media_ref_parse() {
    assert_eq!(
        MediaRef::parse("https://e.x/a.png", "image_url").unwrap(),
        url("https://e.x/a.png")
    );
    assert_eq!(
        MediaRef::parse("http://e.x/a", "p").unwrap(),
        url("http://e.x/a")
    );
    assert_eq!(
        MediaRef::parse("data:image/png;base64,AA==", "p").unwrap(),
        MediaRef::DataUri("data:image/png;base64,AA==".into())
    );
    assert_eq!(
        MediaRef::parse("ltx://uploads/abc", "p").unwrap(),
        MediaRef::Upload(UploadId("abc".into()))
    );
    assert_eq!(
        MediaRef::parse("mm_file://123", "p").unwrap(),
        MediaRef::ProviderFile("mm_file://123".into())
    );
    for bad in ["ftp://e.x/a", "not a url", "ltx://uploads/", "/local/path"] {
        assert_eq!(
            MediaRef::parse(bad, "image_url")
                .unwrap_err()
                .param
                .as_deref(),
            Some("image_url"),
            "{bad}"
        );
    }
}

#[test]
fn request_helpers() {
    let mut r = t2v("m", "p");
    assert!(r.sampling.is_empty());
    r.sampling.steps = Some(1);
    assert!(!r.sampling.is_empty());
    r.keyframes.push(Keyframe {
        at: Anchor::First,
        image: url("https://e.x/1"),
    });
    r.references.push(Reference {
        kind: MediaKind::Image,
        media: url("https://e.x/2"),
    });
    r.audio_in = Some(AudioInput {
        media: url("https://e.x/3"),
        role: AudioRole::Drive,
    });
    let all: Vec<String> = r.media_refs().map(|m| format!("{m:?}")).collect();
    assert_eq!(all.len(), 3);
    assert!(all[0].contains("/1") && all[1].contains("/2") && all[2].contains("/3"));
    assert!(Task::Retake.is_edit_endpoint() && !Task::Ref2V.is_edit_endpoint());
    assert_eq!(ModelId::from("x").to_string(), "x");
    let c = CallbackSpec::FalWebhook {
        url: url::Url::parse("https://h.x/").unwrap(),
    };
    assert_eq!(c.url().as_str(), "https://h.x/");
}

#[test]
fn caps_helpers() {
    let c = h3();
    assert!(c.answers_to("fasth3") && !c.answers_to("MiniMax-H3"));
    assert!(
        c.has_native_audio() && !wan_sidecar().has_native_audio() && !fastwan().has_native_audio()
    );
    assert!(
        c.supports(Task::Keyframes)
            && !c.supports(Task::Ref2V)
            && h3_ref2va().supports(Task::Ref2V)
    );
    assert_eq!(
        c.stream,
        Some(StreamCaps::Clip {
            min_s: 107.0 / 24.0,
            max_s: 362.0 / 24.0
        })
    );
    let cv = CanvasCaps::h3();
    assert_eq!(cv.area_at(768), 768 * 1344);
    assert_eq!(
        cv.area_at(1080),
        768 * 1344,
        "above the top tier the cap holds"
    );
    assert!(cv.aspect_ok(4, 1) && cv.aspect_ok(1, 4) && !cv.aspect_ok(5, 1) && !cv.aspect_ok(0, 1));
    assert!(FpsCaps::fixed(24).allows(24) && !FpsCaps::fixed(24).allows(25));
    assert_eq!(
        RefLimits::h3(),
        RefLimits {
            images: 9,
            videos: 3,
            audio: 3,
            total: 12
        }
    );
    assert!(KnobCaps::all().flow_shift);
}

// ---- HTTP replies and traits --------------------------------------------------------

#[test]
fn http_reply_builders() {
    let r = HttpReply::json(202, json!({"id": "x"})).with_header("x-request-id", "abc");
    assert_eq!((r.status, r.header("X-Request-Id")), (202, Some("abc")));
    assert_eq!(r.json_body(), Some(&json!({"id": "x"})));
    let mut r = HttpReply::empty(204);
    r.push_header("retry-after", "1");
    assert_eq!(
        (r.body.clone(), r.header("retry-after")),
        (ReplyBody::Empty, Some("1"))
    );
    assert_eq!(r.json_body(), None);
    let r = HttpReply::bytes(200, "video/mp4", vec![1u8, 2]);
    assert_eq!(
        r.body,
        ReplyBody::Bytes {
            mime: "video/mp4".into(),
            data: bytes::Bytes::from_static(&[1, 2])
        }
    );
    let r = HttpReply::file(200, "/a.mp4", "video/mp4");
    assert_eq!(
        r.body,
        ReplyBody::File {
            path: "/a.mp4".into(),
            mime: "video/mp4".into()
        }
    );
    let r = HttpReply::json_of(200, &ApiError::invalid("x"));
    assert_eq!(r.json_body().unwrap()["kind"], "invalid_request");
    let spec = SseSpec {
        initial: vec![SseEvent::data("{}")],
        follow: Some(SseFollow::JobStatus {
            job: JobId::new(),
            close_on_terminal: true,
            with_logs: false,
        }),
        keepalive: Some(Duration::from_secs(15)),
    };
    assert_eq!(HttpReply::sse(spec.clone()).body, ReplyBody::Sse(spec));
}

#[test]
fn sse_event_wire_form() {
    assert_eq!(
        SseEvent::data(r#"{"a":1}"#).to_wire(),
        "data: {\"a\":1}\n\n"
    );
    let e = SseEvent {
        event: Some("status".into()),
        id: Some("7".into()),
        data: "l1\nl2".into(),
    };
    assert_eq!(e.to_wire(), "event: status\nid: 7\ndata: l1\ndata: l2\n\n");
}

#[test]
fn normalize_ctx_lookups() {
    let mut cx = NormalizeCtx::new(datetime!(2026-01-01 0:00 UTC));
    cx.query = vec![
        ("fal_webhook".into(), "https://h/".into()),
        ("fal_webhook".into(), "second".into()),
    ];
    cx.headers = vec![("x-fal-target-url".into(), "https://queue.fal.run/x".into())];
    assert_eq!(cx.query_param("fal_webhook"), Some("https://h/"));
    assert_eq!(cx.query_param("nope"), None);
    assert_eq!(
        cx.header("X-Fal-Target-Url"),
        Some("https://queue.fal.run/x")
    );
}

struct Signer;
impl UrlSigner for Signer {
    fn url_for(&self, a: &Artifact, ttl: Duration) -> url::Url {
        url::Url::parse(&format!(
            "https://h.x/files/{}/{}?exp={}",
            a.id,
            a.file_name,
            ttl.as_secs()
        ))
        .unwrap()
    }
}

/// A toy adapter proving the traits are implementable and object-safe the
/// way serve-kit uses them.
struct Toy;
#[derive(serde::Deserialize)]
struct ToyBody {
    prompt: String,
}
impl BatchProtocol for Toy {
    fn id(&self) -> ProtocolId {
        ProtocolId::Native
    }
    fn new_external_id(&self, job: JobId) -> String {
        format!("toy_{}", job.0.simple())
    }
    fn render_error(&self, err: &ApiError, cx: &ErrorCtx) -> HttpReply {
        let r = HttpReply::json(
            err.http_status(),
            json!({"error": err.message, "param": err.param}),
        );
        match &cx.request_id {
            Some(id) => r.with_header("x-request-id", id.clone()),
            None => r,
        }
    }
}
impl SubmitEndpoint for Toy {
    type Body = ToyBody;
    fn normalize(&self, body: ToyBody, cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError> {
        let mut r = GenerationRequest::text(ProtocolId::Native, "fasth3", body.prompt);
        if let Some(u) = cx.query_param("webhook") {
            let url =
                url::Url::parse(u).map_err(|_| ApiError::invalid_param("webhook", "bad url"))?;
            r.callback = Some(CallbackSpec::FalWebhook { url });
        }
        Ok(r)
    }
    fn submit_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        HttpReply::json(
            200,
            json!({"id": job.external_id, "status_url": cx.public_url(&format!("/jobs/{}", job.external_id)).as_str()}),
        )
    }
}
impl JobView for Toy {
    fn status_reply(&self, job: &Job, _cx: &ViewCtx) -> HttpReply {
        HttpReply::json(
            200,
            json!({"status": job.status().as_str(), "progress": (job.progress * 100.0).round()}),
        )
    }
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        match job.artifacts.first() {
            Some(a) if job.status() == JobStatus::Succeeded => HttpReply::json(
                200,
                json!({"url": cx.urls.url_for(a, Duration::from_secs(60)).as_str()}),
            ),
            _ => self.render_error(&ApiError::not_found("not ready"), &ErrorCtx::default()),
        }
    }
}

#[test]
fn traits_drive_a_toy_adapter() {
    let toy = Toy;
    let proto: &dyn BatchProtocol = &toy;
    let view: &dyn JobView = &toy;
    let base = url::Url::parse("https://api.example/prefix").unwrap();
    let cx = ViewCtx {
        now: datetime!(2026-01-01 0:00 UTC),
        urls: &Signer,
        public_base: &base,
        with_logs: false,
    };
    assert!(format!("{cx:?}").contains("api.example"));

    let body: ToyBody = serde_json::from_value(json!({"prompt": "a cat"})).unwrap();
    let mut ncx = NormalizeCtx::new(cx.now);
    ncx.query.push(("webhook".into(), "https://hook.x/".into()));
    let req = toy.normalize(body, &ncx).unwrap();
    assert!(matches!(
        req.callback,
        Some(CallbackSpec::FalWebhook { .. })
    ));
    let resolved = negotiate(&req, &h3(), &StagedInputs::default()).unwrap();
    let id = JobId::new();
    let mut job = Job::new(
        id,
        proto.id(),
        proto.new_external_id(id),
        resolved,
        cx.now,
        Duration::from_secs(60),
    );
    assert_eq!(job.external_id.len(), 4 + 32);

    let r = toy.submit_reply(&job, &cx);
    assert_eq!(
        r.json_body().unwrap()["status_url"],
        format!("https://api.example/prefix/jobs/{}", job.external_id)
    );
    assert_eq!(
        view.status_reply(&job, &cx).json_body().unwrap()["status"],
        "queued"
    );
    assert_eq!(view.result_reply(&job, &cx).status, 404);
    job.mark_running(cx.now).unwrap();
    job.mark_succeeded(
        cx.now,
        vec![Artifact {
            id: ArtifactId::new(),
            mime: "video/mp4".into(),
            file_name: "o.mp4".into(),
            bytes: 1,
            location: ArtifactLocation::Local("/o.mp4".into()),
            width: 1344,
            height: 768,
            frames: 124,
            fps: 24,
            audio: None,
        }],
        JobMetrics::default(),
    )
    .unwrap();
    assert!((job.artifacts[0].duration_s() - 124.0 / 24.0).abs() < 1e-9);
    let r = view.result_reply(&job, &cx);
    assert!(r.json_body().unwrap()["url"]
        .as_str()
        .unwrap()
        .ends_with("/o.mp4?exp=60"));
    assert_eq!(
        view.status_reply(&job, &cx).json_body().unwrap()["progress"],
        100.0
    );

    let err = proto.render_error(
        &ApiError::invalid_param("size", "bad"),
        &ErrorCtx {
            request_id: Some("rid".into()),
            ..ErrorCtx::default()
        },
    );
    assert_eq!((err.status, err.header("x-request-id")), (400, Some("rid")));
}

#[test]
fn public_url_joins_under_base_path() {
    let base = url::Url::parse("https://h.x/api/").unwrap();
    let cx = ViewCtx {
        now: datetime!(2026-01-01 0:00 UTC),
        urls: &Signer,
        public_base: &base,
        with_logs: true,
    };
    assert_eq!(
        cx.public_url("/minimax/h3-max/requests/1").as_str(),
        "https://h.x/api/minimax/h3-max/requests/1"
    );
    let root = url::Url::parse("https://h.x").unwrap();
    let cx = ViewCtx {
        public_base: &root,
        ..cx
    };
    assert_eq!(cx.public_url("files/a").as_str(), "https://h.x/files/a");
}

// ---- streaming types --------------------------------------------------------------------

#[test]
fn track_sets() {
    let t = TrackSet::for_model(
        &h3(),
        (1344, 768),
        24,
        ("main_video", "main_audio"),
        1,
        false,
    );
    assert_eq!(t.names(), vec!["main_video", "main_audio"]);
    assert_eq!(
        t.audio.as_ref().map(|a| (a.rate, a.channels)),
        Some((WIRE_AUDIO_RATE, 1))
    );
    assert_eq!(t.samples_per_frame().unwrap(), 2000);
    let v = TrackSet::for_model(
        &fastwan(),
        (832, 480),
        16,
        ("main_video", "main_audio"),
        2,
        true,
    );
    assert!(
        !v.has_audio(),
        "no audio m-line for a video-only model, even asking for sidecar"
    );
    assert_eq!(v.names(), vec!["main_video"]);
    assert_eq!(v.samples_per_frame().unwrap(), 3000);
    let s = TrackSet::for_model(&wan_sidecar(), (832, 480), 16, ("v", "a"), 2, true);
    assert!(s.has_audio());
    assert!(!TrackSet::for_model(&wan_sidecar(), (832, 480), 16, ("v", "a"), 2, false).has_audio());
    let mut bad = t.clone();
    bad.video.fps = 7;
    assert_eq!(
        bad.samples_per_frame().unwrap_err().param.as_deref(),
        Some("fps")
    );
    bad.video.fps = 0;
    assert!(bad.samples_per_frame().is_err());
}

#[test]
fn session_lifecycle() {
    use SessionState::*;
    let closed = Closed(EndReason::Stopped);
    let ok = [
        (Starting, Ready),
        (Ready, Streaming),
        (Streaming, Orphaned),
        (Orphaned, Streaming),
        (Ready, Orphaned),
        (Starting, Closing),
        (Streaming, Closing),
        (Orphaned, Closing),
        (Closing, closed.clone()),
        (Starting, closed.clone()),
    ];
    for (a, b) in &ok {
        assert!(a.can_transition_to(b), "{a:?} -> {b:?}");
    }
    let bad = [
        (Starting, Streaming),
        (Streaming, Ready),
        (Closing, Streaming),
        (closed.clone(), Starting),
        (closed.clone(), Closed(EndReason::TimedOut)),
    ];
    for (a, b) in &bad {
        assert!(!a.can_transition_to(b), "{a:?} -> {b:?}");
    }
    assert!(Starting.is_busy() && Closing.is_busy() && !closed.is_busy());
    assert_eq!(Continuity::default(), Continuity::Crossfade { ms: 20 });
}

/// `StreamProtocol` is object-safe and yields the protocol's track names.
#[test]
fn stream_protocol_trait() {
    struct Reactorish;
    impl StreamProtocol for Reactorish {
        fn id(&self) -> ProtocolId {
            ProtocolId::Reactor
        }
        fn tracks(&self, caps: &ModelCaps, canvas: (u32, u32), fps: u32) -> TrackSet {
            TrackSet::for_model(caps, canvas, fps, ("main_video", "main_audio"), 1, false)
        }
    }
    let p: &dyn StreamProtocol = &Reactorish;
    assert_eq!(p.id(), ProtocolId::Reactor);
    assert_eq!(
        p.tracks(&fastwan(), (832, 480), 16).names(),
        vec!["main_video"]
    );
}

// ---- A/V buffers ---------------------------------------------------------------------------

#[test]
fn rgb_frames() {
    assert!(RgbFrame::new(2, 2, bytes::Bytes::from(vec![0u8; 12]), 0).is_ok());
    assert_eq!(
        RgbFrame::new(2, 2, bytes::Bytes::from(vec![0u8; 11]), 0)
            .unwrap_err()
            .kind,
        ErrorKind::Internal
    );
    let f = RgbFrame::solid(3, 2, [1, 2, 3], 9);
    assert_eq!(
        (f.data.len(), f.index, f.pixel(2, 1), f.pixel(3, 0)),
        (18, 9, Some([1, 2, 3]), None)
    );
    assert_eq!(RgbFrame::black(4, 4, 0).pixel(0, 0), Some([0, 0, 0]));
    assert_eq!(RgbFrame::byte_len(1344, 768), 1344 * 768 * 3);
}

#[test]
fn pcm_buffers() {
    let p = Pcm::new(48_000, 2, vec![1.0, 0.0, 0.5, 0.5]);
    assert_eq!((p.frames(), p.is_empty()), (2, false));
    assert!((p.duration_s() - 2.0 / 48_000.0).abs() < 1e-12);
    let m = p.to_mono();
    assert_eq!((m.channels, &m.samples[..]), (1, &[0.5f32, 0.5][..]));
    assert_eq!(m.to_mono(), m);
    let u = m.upmix(2);
    assert_eq!(
        (u.channels, &u.samples[..]),
        (2, &[0.5f32, 0.5, 0.5, 0.5][..])
    );
    assert_eq!(p.upmix(4), p, "only mono upmixes");
    let s = Pcm::silence(32_000, 2, 6000);
    assert_eq!((s.frames(), s.samples.len()), (6000, 12_000));
    assert!(s.samples.iter().all(|&x| x == 0.0));
    let z = Pcm::new(0, 0, Vec::<f32>::new());
    assert_eq!((z.frames(), z.duration_s(), z.is_empty()), (0, 0.0, true));
    // Cheap clones share the buffer.
    let c = p.clone();
    assert!(std::sync::Arc::ptr_eq(&c.samples, &p.samples));
}
