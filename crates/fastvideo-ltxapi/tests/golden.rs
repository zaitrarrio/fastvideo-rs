//! Golden conformance (design §7.1-§7.2): wire shapes from the OpenAPI-derived
//! research (ltx §1.5, §2.1, §2.10) compared exactly, normalized-request
//! snapshots, the model matrix, the pad-and-crop geometry and the gap table.

use std::path::PathBuf;
use std::time::Duration;

use fastvideo_engine_service::FakeModel;
use fastvideo_ltxapi::error::{error_body, render, status_of};
use fastvideo_ltxapi::models::{resolution_strings, LtxModels, API_FPS};
use fastvideo_ltxapi::request::normalize;
use fastvideo_ltxapi::upload::upload_body;
use fastvideo_ltxapi::v2::{created_reply, fmt_ts};
use fastvideo_ltxapi::{Api, Endpoint, LtxErrorType, LtxProtocol, V1View, V2View};
use fastvideo_protocol::{
    negotiate, precheck, Artifact, ArtifactId, ArtifactLocation, AudioPlan, BatchProtocol,
    ErrorCtx, ErrorKind, Family, GapId, Job, JobId, JobMetrics, JobView, ModelCaps, PostProcess,
    ProtocolId, ReplyBody, ResolvedJob, SamplingOverrides, StagedInputs, Task, Tier, UrlSigner,
    ViewCtx,
};
use fastvideo_serve_kit::UploadTicket;
use serde_json::{json, Value};
use time::macros::datetime;
use time::OffsetDateTime;
use url::Url;

const ID: &str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";

fn golden(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name);
    serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

struct FixedUrl;
impl UrlSigner for FixedUrl {
    fn url_for(&self, _a: &Artifact, _ttl: Duration) -> Url {
        Url::parse("https://storage.googleapis.com/example/video.mp4").unwrap()
    }
}

fn resolved(task: Task) -> ResolvedJob {
    ResolvedJob {
        model: "fake-ltx-turbo".into(),
        task,
        prompt: "p".into(),
        negative_prompt: String::new(),
        seed: 1,
        width: 1920,
        height: 1088,
        num_frames: 193,
        fps: 24,
        keyframes: vec![],
        references: vec![],
        audio_in: None,
        audio: AudioPlan::Native { rate: 48_000, channels: 2 },
        post: PostProcess { crop: Some((1920, 1080)), drop_audio: false },
        sampling: SamplingOverrides::default(),
        tier: Some(Tier::Turbo),
        recipe: Some("two-stage-sol".into()),
    }
}

fn job(created: OffsetDateTime, task: Task) -> Job {
    let jid = JobId(uuid::Uuid::parse_str(ID).unwrap());
    let ext = LtxProtocol::V2.new_external_id(jid);
    Job::new(jid, ProtocolId::LtxV2, ext, resolved(task), created, Duration::from_secs(86_400))
}

fn artifact() -> Artifact {
    Artifact {
        id: ArtifactId::new(),
        mime: "video/mp4".into(),
        file_name: "output.mp4".into(),
        bytes: 10,
        location: ArtifactLocation::Local("/tmp/none.mp4".into()),
        width: 1920,
        height: 1080,
        frames: 193,
        fps: 24,
        audio: Some((48_000, 2)),
    }
}

fn view_status(job: &Job, ep: Endpoint) -> (u16, Value) {
    let base = Url::parse("https://api.example.test").unwrap();
    let cx = ViewCtx { now: datetime!(2026-01-15 10:05 UTC), urls: &FixedUrl, public_base: &base, with_logs: false };
    let r = V2View::new(ep, Duration::from_secs(3600)).status_reply(job, &cx);
    (r.status, r.json_body().cloned().unwrap_or(Value::Null))
}

#[test]
fn job_created() {
    let j = job(datetime!(2026-09-06 12:00 UTC), Task::T2V);
    assert_eq!(j.external_id, ID);
    let r = created_reply(&j);
    assert_eq!(r.status, 202);
    assert_eq!(r.json_body().unwrap(), &golden("job_created.json"));
    assert_eq!(r.header("x-fv-tier"), Some("turbo"));
    assert_eq!(r.header("x-fv-recipe"), Some("two-stage-sol"));
    assert_eq!(fmt_ts(datetime!(2026-09-06 14:00:00.123456 +2)), "2026-09-06T12:00:00.123Z");
}

#[test]
fn status_one_of_shapes() {
    let t0 = datetime!(2026-01-15 10:00 UTC);
    let t1 = datetime!(2026-01-15 10:02:30 UTC);
    let mut j = job(t0, Task::T2V);
    assert_eq!(view_status(&j, Endpoint::TextToVideo), (200, golden("status_pending.json")));
    j.mark_running(datetime!(2026-01-15 10:00:01 UTC)).unwrap();
    assert_eq!(view_status(&j, Endpoint::TextToVideo), (200, golden("status_processing.json")));
    let mut done = j.clone();
    done.mark_succeeded(t1, vec![artifact()], JobMetrics::default()).unwrap();
    assert_eq!(view_status(&done, Endpoint::TextToVideo), (200, golden("status_completed.json")));
    let mut failed = j.clone();
    failed.mark_failed(t1, fastvideo_protocol::ApiError::engine_failed("Unexpected server error")).unwrap();
    assert_eq!(view_status(&failed, Endpoint::TextToVideo), (200, golden("status_failed.json")));
    // Cancelled jobs report failed / api_error.
    let mut c = j.clone();
    c.mark_cancelled(t1).unwrap();
    let (_, b) = view_status(&c, Endpoint::TextToVideo);
    assert_eq!((b["status"].as_str(), b["error"]["type"].as_str()), (Some("failed"), Some("api_error")));
    // The endpoint segment must match the submit endpoint.
    let (s, b) = view_status(&j, Endpoint::ImageToVideo);
    assert_eq!(s, 404);
    assert_eq!(b["error"]["type"], "not_found_error");
    for t in [Task::I2V, Task::Keyframes] {
        assert_eq!(view_status(&job(t0, t), Endpoint::ImageToVideo).0, 200);
    }
}

#[test]
fn draft_results_are_marked() {
    let mut j = job(datetime!(2026-01-15 10:00 UTC), Task::T2V);
    j.resolved.tier = Some(Tier::Draft);
    let r = created_reply(&j);
    assert_eq!(r.header("x-fv-quality"), Some("draft"));
    assert_eq!(r.header("x-fv-tier"), Some("draft"));
}

#[test]
fn every_error_type() {
    let fixtures = golden("errors.json");
    let fixtures = fixtures.as_array().unwrap();
    assert_eq!(fixtures.len(), 11);
    for f in fixtures {
        let body = &f["body"];
        let name = body["error"]["type"].as_str().unwrap();
        let t = LtxErrorType::ALL.into_iter().find(|t| t.as_str() == name).unwrap();
        assert_eq!(t.status() as u64, f["http"].as_u64().unwrap(), "{name}");
        assert_eq!(&error_body(t, body["error"]["message"].as_str().unwrap()), body, "{name}");
    }
    // Every ErrorKind renders into one of the documented types and statuses.
    let kinds = [
        (ErrorKind::InvalidRequest, Api::V2, 400, "invalid_request_error"),
        (ErrorKind::Unsupported(GapId::LtxFps), Api::V2, 400, "invalid_request_error"),
        (ErrorKind::Unsupported(GapId::LtxEndpoint), Api::V2, 403, "permission_error"),
        (ErrorKind::PayloadTooLarge, Api::V2, 400, "invalid_request_error"),
        (ErrorKind::UnsupportedMedia, Api::V1, 400, "invalid_request_error"),
        (ErrorKind::Unauthorized, Api::V1, 401, "authentication_error"),
        (ErrorKind::Forbidden, Api::V2, 403, "permission_error"),
        (ErrorKind::NotFound, Api::V2, 404, "not_found_error"),
        (ErrorKind::ContentFiltered, Api::V2, 422, "content_filtered_error"),
        (ErrorKind::QueueFull, Api::V1, 429, "concurrency_limit_error"),
        (ErrorKind::QueueFull, Api::V2, 429, "rate_limit_error"),
        (ErrorKind::RateLimited, Api::V2, 429, "rate_limit_error"),
        (ErrorKind::Internal, Api::V2, 500, "api_error"),
        (ErrorKind::EngineFailed, Api::V1, 500, "api_error"),
        (ErrorKind::Cancelled, Api::V1, 500, "api_error"),
        (ErrorKind::Loading, Api::V2, 503, "service_unavailable_error"),
        (ErrorKind::Timeout, Api::V1, 504, "api_error"),
    ];
    for (kind, api, status, ty) in kinds {
        let e = fastvideo_protocol::ApiError::new(kind, "m");
        let cx = ErrorCtx { request_id: Some("1234567890abcdef1234567890abcdef".into()), ..Default::default() };
        let r = render(&e, api, &cx);
        assert_eq!((r.status, status_of(&e, api)), (status, status), "{kind:?}");
        assert_eq!(r.json_body().unwrap(), &json!({"type": "error", "error": {"type": ty, "message": "m"}}));
        assert_eq!(r.header("x-request-id"), Some("1234567890abcdef1234567890abcdef"));
        assert_eq!(r.header("retry-after").is_some(), status == 429, "{kind:?}");
    }
}

#[test]
fn v1_result_shapes() {
    let base = Url::parse("https://api.example.test").unwrap();
    let cx = ViewCtx { now: datetime!(2026-01-15 10:05 UTC), urls: &FixedUrl, public_base: &base, with_logs: false };
    let mut j = job(datetime!(2026-01-15 10:00 UTC), Task::T2V);
    j.mark_running(datetime!(2026-01-15 10:00 UTC)).unwrap();
    assert_eq!(V1View.result_reply(&j, &cx).status, 504);
    let mut ok = j.clone();
    ok.mark_succeeded(datetime!(2026-01-15 10:01 UTC), vec![artifact()], JobMetrics::default()).unwrap();
    let r = V1View.result_reply(&ok, &cx);
    assert_eq!(r.status, 200);
    assert!(matches!(&r.body, ReplyBody::File { mime, .. } if mime == "video/mp4"));
    let mut obj = ok.clone();
    obj.artifacts[0].location = ArtifactLocation::Object { bucket: "b".into(), key: "k".into() };
    assert_eq!(V1View.result_reply(&obj, &cx).status, 500);
    let mut f = j.clone();
    f.mark_failed(datetime!(2026-01-15 10:01 UTC), fastvideo_protocol::ApiError::engine_failed("boom")).unwrap();
    let r = V1View.result_reply(&f, &cx);
    assert_eq!(r.status, 500);
    assert_eq!(r.json_body().unwrap(), &json!({"type": "error", "error": {"type": "api_error", "message": "boom"}}));
}

#[test]
fn upload_shape() {
    let t = UploadTicket {
        token: "TOKEN".into(),
        upload_url: Url::parse("https://api.example.test/uploads/TOKEN").unwrap(),
        expires_at: datetime!(2026-09-06 13:00 UTC),
        file_name: "upload".into(),
    };
    let s = upload_body(&t).to_string().replace("TOKEN", "<token>");
    assert_eq!(serde_json::from_str::<Value>(&s).unwrap(), golden("upload.json"));
}

#[test]
fn request_snapshots() {
    for name in ["t2v.json", "i2v.json", "keyframes.json"] {
        let g = golden(&format!("requests/{name}"));
        let ep = Endpoint::from_segment(g["endpoint"].as_str().unwrap()).unwrap();
        let api = if g["api"] == "v1" { Api::V1 } else { Api::V2 };
        let req = normalize(ep, api, &LtxModels::default(), &g["body"]).unwrap();
        assert_eq!(serde_json::to_value(&req).unwrap(), g["request"], "{name}");
    }
}

/// The ltx §3 matrix, written independently of `models::allowed_durations`.
fn spec_allows(model: &str, res: &str, fps: u32, d: u32) -> bool {
    let fast = !model.ends_with("-pro");
    let hd = matches!(res, "1280x720" | "720x1280" | "1920x1080" | "1080x1920");
    let known_res = resolution_strings().iter().any(|r| r == res);
    let fps_ok = [24, 25, 48, 50].contains(&fps);
    let long = fast && hd && (fps == 24 || fps == 25);
    let d_ok = d % 2 == 0 && d >= 6 && d <= if long { 20 } else { 10 };
    known_res && fps_ok && d_ok
}

#[test]
fn model_matrix_table() {
    let models = LtxModels::default();
    let mut checked = 0;
    for model in models.ids().collect::<Vec<_>>() {
        for res in resolution_strings().into_iter().chain(["1920x1088".to_owned(), "768x512".to_owned()]) {
            for fps in API_FPS.into_iter().chain([30]) {
                for d in 4..=22 {
                    let body = json!({"prompt": "p", "model": model, "resolution": res, "fps": fps, "duration": d});
                    let got = normalize(Endpoint::TextToVideo, Api::V2, &models, &body).is_ok();
                    assert_eq!(got, spec_allows(model, &res, fps, d), "{model} {res} {fps} {d}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 5000);
}

fn ltx_caps() -> ModelCaps {
    FakeModel::ltx_turbo().caps
}

#[test]
fn pad_and_crop_geometry() {
    let caps = ltx_caps();
    // (resolution, duration) -> (generated w, h, frames, delivered crop)
    let table = [
        ("1280x720", 6, (1280, 768, 145), Some((1280, 720))),
        ("720x1280", 8, (768, 1280, 193), Some((720, 1280))),
        ("1920x1080", 20, (1920, 1088, 481), Some((1920, 1080))),
        ("1080x1920", 10, (1088, 1920, 241), Some((1080, 1920))),
        ("2560x1440", 6, (2560, 1472, 145), Some((2560, 1440))),
        ("1440x2560", 8, (1472, 2560, 193), Some((1440, 2560))),
        ("3840x2160", 10, (3840, 2176, 241), Some((3840, 2160))),
        ("2160x3840", 6, (2176, 3840, 145), Some((2160, 3840))),
    ];
    for (res, d, (w, h, n), crop) in table {
        let body = json!({"prompt": "p", "model": "ltx-2-5-fast", "resolution": res, "duration": d});
        let req = normalize(Endpoint::TextToVideo, Api::V2, &LtxModels::default(), &body).unwrap();
        let r = negotiate(&req, &caps, &StagedInputs::default()).unwrap();
        assert_eq!((r.width, r.height, r.num_frames, r.post.crop), (w, h, n, crop), "{res}");
        assert_eq!(r.output_size(), crop.unwrap());
        assert_eq!((r.width % 64, r.height % 64), (0, 0));
        assert!(r.width >= crop.unwrap().0 && r.height >= crop.unwrap().1);
        assert_eq!(r.fps, 24);
        assert_eq!(r.tier, Some(Tier::Turbo));
    }
    // generate_audio:false drops the native track in post.
    let body = json!({"prompt": "p", "model": "ltx-turbo", "resolution": "1280x720", "duration": 6, "generate_audio": false});
    let req = normalize(Endpoint::TextToVideo, Api::V2, &LtxModels::default(), &body).unwrap();
    let r = negotiate(&req, &caps, &StagedInputs::default()).unwrap();
    assert_eq!((r.audio, r.post.drop_audio), (AudioPlan::Drop, true));
}

#[test]
fn engine_gaps_render_400() {
    let caps = ltx_caps();
    let mut t2v_only = caps.clone();
    t2v_only.tasks.remove(&Task::I2V);
    // An engine that has not validated the extra rates (serve E4) serves 24 only.
    let mut fps24 = caps.clone();
    fps24.fps = fastvideo_protocol::FpsCaps::fixed(24);
    let base = json!({"prompt": "p", "model": "ltx-2-5-fast", "resolution": "1920x1080", "duration": 8});
    let with = |k: &str, v: Value| {
        let mut b = base.clone();
        b[k] = v;
        b
    };
    // The engine's validated rates pass the precheck.
    for fps in fastvideo_engine_service::LTX_FPS {
        let req = normalize(Endpoint::TextToVideo, Api::V2, &LtxModels::default(), &with("fps", json!(fps))).unwrap();
        assert!(precheck(&req, &caps).is_ok(), "fps {fps}");
    }
    let cases: Vec<(Endpoint, Value, &ModelCaps, GapId)> = vec![
        (Endpoint::TextToVideo, with("fps", json!(25)), &fps24, GapId::LtxFps),
        (Endpoint::TextToVideo, with("fps", json!(50)), &fps24, GapId::LtxFps),
        (Endpoint::TextToVideo, with("duration", Value::Null), &caps, GapId::LtxAutoDuration),
        (Endpoint::ImageToVideo, {
            let mut b = with("image_uri", json!("ltx://uploads/a"));
            b["last_frame_uri"] = json!("ltx://uploads/b");
            b
        }, &caps, GapId::LtxKeyframes),
        (Endpoint::ImageToVideo, with("image_uri", json!("ltx://uploads/a")), &t2v_only, GapId::Ltx25I2V),
    ];
    for (ep, body, caps, gap) in cases {
        let req = normalize(ep, Api::V2, &LtxModels::default(), &body).unwrap();
        let e = precheck(&req, caps).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Unsupported(gap), "{body}");
        let r = render(&e, Api::V2, &ErrorCtx::default());
        assert_eq!(r.status, 400);
        assert_eq!(r.json_body().unwrap()["error"]["type"], "invalid_request_error");
    }
    // camera_motion is refused at normalization.
    let e = normalize(Endpoint::TextToVideo, Api::V2, &LtxModels::default(), &with("camera_motion", json!("static"))).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Unsupported(GapId::LtxCameraMotion));
}

#[test]
fn tier_aliases_match_engine() {
    for t in [Tier::Max, Tier::Turbo, Tier::Draft] {
        assert_eq!(
            Some(fastvideo_ltxapi::models::tier_alias(t)),
            fastvideo_engine_service::tier_alias(Family::Ltx2, t)
        );
    }
}
