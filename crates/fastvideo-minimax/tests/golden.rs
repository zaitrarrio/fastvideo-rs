//! Golden JSON: create (normalize + reply), query, list, delete, errors and
//! callbacks. Fixtures live in `tests/golden/`; `FV_UPDATE_GOLDEN=1`
//! rewrites the generated ones. The `*.minimax.json`, `error_400.json`,
//! `error_401.json` and `create_*.body.json` fixtures are MiniMax's own
//! examples (research §1.3-§1.5) and are never rewritten.

mod common;

use std::time::Duration;

use common::*;
use fastvideo_minimax::delete::delete_body;
use fastvideo_minimax::{render_oai_error, task_json, CreateBody, CreateEndpoint, MiniMax, TaskView};
use fastvideo_protocol::{
    Anchor, ApiError, Artifact, ArtifactId, ArtifactLocation, AudioPlan, BatchProtocol, ErrorCtx,
    GapId, Job, JobId, JobView, MediaKind, NormalizeCtx, PostProcess, ProtocolId, ResolvedJob,
    SamplingOverrides, SubmitEndpoint, Task, Tier, UrlSigner, ViewCtx,
};
use fastvideo_serve_kit::CallbackRender;
use serde_json::{json, Value};
use time::OffsetDateTime;
use url::Url;

const RID: &str = "021785229015510a2c883cf675b9804d";

struct Signer;
impl UrlSigner for Signer {
    fn url_for(&self, a: &Artifact, _ttl: Duration) -> Url {
        Url::parse(&format!("https://cdn.example.com/{}", a.file_name)).unwrap()
    }
}

fn at(s: i64) -> OffsetDateTime {
    OffsetDateTime::from_unix_timestamp(s).unwrap()
}

fn resolved(task: Task) -> ResolvedJob {
    ResolvedJob {
        model: "fake-h3-max".into(),
        task,
        prompt: "a red fox in snow".into(),
        negative_prompt: String::new(),
        seed: 7,
        width: 1344,
        height: 768,
        num_frames: 124,
        fps: 24,
        keyframes: if task == Task::I2V { vec![(Anchor::First, "/in/0.png".into())] } else { vec![] },
        references: vec![],
        audio_in: None,
        audio: AudioPlan::Native { rate: 32000, channels: 2 },
        post: PostProcess::default(),
        sampling: SamplingOverrides::default(),
        tier: None,
        recipe: None,
        edit: None,
    }
}

fn job(ext: &str, echo: Value, task: Task) -> Job {
    let id = JobId("6f1c2a52-8d7e-4b8e-9a51-2f3a4b5c6d7e".parse().unwrap());
    let mut j = Job::new(id, ProtocolId::MiniMaxV2, ext, resolved(task), at(1785125529), Duration::from_secs(7 * 86400));
    j.request_echo = echo;
    j
}

fn artifact() -> Artifact {
    Artifact {
        id: ArtifactId("0b7e0f5e-1111-4222-8333-944455556666".parse().unwrap()),
        mime: "video/mp4".into(),
        file_name: "output.mp4".into(),
        bytes: 1234,
        location: ArtifactLocation::Local("/state/artifacts/x/output.mp4".into()),
        width: 1344,
        height: 768,
        frames: 124,
        fps: 24,
        audio: Some((32000, 2)),
    }
}

fn view_ctx<'a>(base: &'a Url) -> ViewCtx<'a> {
    ViewCtx { now: at(1785126000), urls: &Signer, public_base: base, with_logs: false }
}

fn tv() -> TaskView {
    MiniMax::default().view()
}

// ---- create ----------------------------------------------------------------

#[test]
fn create_examples_normalize() {
    for name in ["create_t2v", "create_i2v", "create_r2v"] {
        let body: CreateBody = serde_json::from_value(load_golden(&format!("{name}.body.json"))).unwrap();
        let req = CreateEndpoint.normalize(body, &NormalizeCtx::new(at(0))).unwrap();
        assert_golden(&format!("{name}.normalized.json"), &serde_json::to_value(&req).unwrap());
    }
}

#[test]
fn create_reply_is_task_id_only() {
    let j = job("424010985738629", json!({}), Task::T2V);
    let base = Url::parse("http://fv.test").unwrap();
    let r = CreateEndpoint.submit_reply(&j, &view_ctx(&base));
    assert_eq!(r.status, 200);
    assert_golden("create_reply.json", r.json_body().unwrap());
    // Our ids are 18 digits.
    let id = MiniMax::default().new_external_id(JobId::new());
    assert!(id.len() == 18 && id.bytes().all(|b| b.is_ascii_digit()), "{id}");
}

// ---- query ------------------------------------------------------------------

#[test]
fn query_succeeded_matches_minimax_example() {
    let echo = json!({"model": "MiniMax-H3", "resolution": "2K", "duration": 5, "ratio": "16:9",
        "_fv": {"input_image_count": 1, "input_video_seconds": 0.0, "input_audio_seconds": 6.0}});
    let mut j = job("424010985738629", echo, Task::I2V);
    j.mark_running(at(1785125600)).unwrap();
    j.mark_succeeded(at(1785125946), vec![artifact()], Default::default()).unwrap();
    let base = Url::parse("http://fv.test").unwrap();
    let got = tv().status_reply(&j, &view_ctx(&base));
    assert_eq!(got.status, 200);
    let got = got.json_body().unwrap().clone();
    // MiniMax's example minus the billing token fields (design Q7: omitted).
    let mut want = load_golden("query_succeeded.minimax.json");
    let usage = want["task"]["usage"].as_object_mut().unwrap();
    for k in ["total_tokens", "prompt_tokens", "completion_tokens"] {
        usage.remove(k);
    }
    assert_eq!(got, want);
    // The callback body is the same object.
    assert_eq!(tv().callback_body(&j, &view_ctx(&base)).unwrap(), got);
}

#[test]
fn query_states_golden() {
    let base = Url::parse("http://fv.test").unwrap();
    let cx = view_ctx(&base);
    let echo = json!({"model": "MiniMax-H3-Max", "resolution": "768P", "duration": 6, "ratio": "adaptive",
        "_fv": {"input_image_count": 1, "input_video_seconds": 0.0, "input_audio_seconds": null}});
    let mut j = job("123456789012345678", echo, Task::I2V);
    let mut all = serde_json::Map::new();
    all.insert("queued".into(), task_json(&j, &cx, Duration::from_secs(60)));
    j.mark_running(at(1785125600)).unwrap();
    all.insert("running".into(), task_json(&j, &cx, Duration::from_secs(60)));
    let mut failed = j.clone();
    failed.mark_failed(at(1785125700), ApiError::engine_failed("injected failure at step 1/8")).unwrap();
    all.insert("failed".into(), task_json(&failed, &cx, Duration::from_secs(60)));
    let mut filtered = j.clone();
    filtered.mark_failed(at(1785125700), ApiError::content_filtered("video description contains sensitive content")).unwrap();
    all.insert("failed_filtered".into(), task_json(&filtered, &cx, Duration::from_secs(60)));
    let mut cancelled = j.clone();
    cancelled.mark_cancelled(at(1785125650)).unwrap();
    all.insert("cancelled".into(), task_json(&cancelled, &cx, Duration::from_secs(60)));
    j.resolved.tier = Some(Tier::Draft);
    j.resolved.recipe = Some("4step-vsa-480p-tiny-vae".into());
    j.mark_succeeded(at(1785125946), vec![artifact()], Default::default()).unwrap();
    all.insert("succeeded_draft".into(), task_json(&j, &cx, Duration::from_secs(60)));
    let all = Value::Object(all);
    assert_golden("query_states.json", &all);
    // Invariants: error only when failed, usage only on success.
    for (k, t) in all.as_object().unwrap() {
        assert_eq!(t.get("error").is_some(), k.starts_with("failed"), "{k}");
        assert_eq!(t.get("usage").is_some(), k.starts_with("succeeded"), "{k}");
    }
    assert_eq!(all["failed_filtered"]["error"]["code"], "1026");
    assert_eq!(all["succeeded_draft"]["metadata"]["quality"], "draft");
    assert_eq!(all["succeeded_draft"]["ratio"], "16:9", "adaptive reports the generated ratio");
}

#[test]
fn list_golden() {
    let base = Url::parse("http://fv.test").unwrap();
    let cx = view_ctx(&base);
    let a = job("100000000000000001", json!({"model": "MiniMax-H3", "resolution": "768P", "duration": 5, "ratio": "16:9"}), Task::T2V);
    let mut b = job("100000000000000002", json!({"model": "MiniMax-H3", "resolution": "768P", "duration": 4, "ratio": "9:16"}), Task::T2V);
    b.resolved.width = 768;
    b.resolved.height = 1344;
    b.mark_running(at(1785125600)).unwrap();
    let items: Vec<Value> = [&b, &a].iter().map(|j| task_json(j, &cx, Duration::from_secs(60))).collect();
    assert_golden("list.json", &json!({"items": items, "total": 2}));
}

// ---- delete -----------------------------------------------------------------

#[test]
fn delete_golden() {
    assert_golden(
        "delete.json",
        &json!({"cancelled": delete_body("123456789012345678", "cancelled"),
                "deleted": delete_body("123456789012345678", "deleted")}),
    );
}

// ---- errors -----------------------------------------------------------------

#[test]
fn error_envelopes() {
    let mm = MiniMax::default();
    let cx = ErrorCtx { request_id: Some(RID.into()), ..Default::default() };
    let r = mm.render_error(&ApiError::invalid("content must include a non-empty text item (prompt is required)"), &cx);
    assert_eq!(r.status, 400);
    assert_eq!(r.json_body().unwrap(), &load_golden("error_400.json"));
    let r = mm.render_error(&ApiError::unauthorized("missing credentials"), &cx);
    assert_eq!(r.status, 401);
    assert_eq!(r.json_body().unwrap(), &load_golden("error_401.json"));

    let mut all = serde_json::Map::new();
    for (name, e) in [
        ("unknown_task", ApiError::not_found("`1` was not found")),
        ("gap_2k", ApiError::unsupported(GapId::H3Resolution2K)),
        ("gap_provider_file", ApiError::unsupported(GapId::ProviderFiles)),
        ("delete_running", ApiError::conflict("task 123456789012345678 is running and cannot be cancelled or deleted")),
        ("body_too_large", ApiError::payload_too_large("request body exceeds 64 MB")),
        ("bad_key", ApiError::unauthorized("invalid credentials")),
        ("filtered", ApiError::content_filtered("video description contains sensitive content")),
        ("rate_limited", ApiError::rate_limited("please retry later")),
        ("queue_full", ApiError::queue_full("job queue is full")),
        ("loading", ApiError::loading("models are loading")),
        ("internal", ApiError::internal("boom")),
    ] {
        let r = render_oai_error(&e, Some(RID));
        all.insert(name.into(), json!({"http": r.status, "body": r.json_body().unwrap()}));
    }
    assert_golden("errors.json", &Value::Object(all));
}

// ---- callbacks (end to end, normalized) ------------------------------------------

fn normalize_callback(v: &Value) -> Value {
    let mut v = v.clone();
    if v.get("challenge").is_some() {
        assert_eq!(v["challenge"].as_str().unwrap().len(), 32);
        v["challenge"] = json!("<challenge>");
        return v;
    }
    let t = &mut v["task"];
    t["id"] = json!("<task_id>");
    t["created_at"] = json!(0);
    t["updated_at"] = json!(0);
    if t["content"].get("url").is_some() {
        t["content"]["url"] = json!("<url>");
    }
    v
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_sequence_golden() {
    let f = fixture().await;
    let mut body = t2v("MiniMax-H3", 5);
    body["callback_url"] = json!(f.rx.url("/cb"));
    let id = create_ok(&f, body).await;
    let seen = f.rx.wait_terminal("/cb").await;
    assert!(seen[1..].iter().all(|v| v["task"]["id"] == id.as_str()));
    let norm: Vec<Value> = seen.iter().map(normalize_callback).collect();
    assert_golden("callback_sequence.json", &Value::Array(norm));

    // The final callback equals a fresh query (same structure, same fields).
    let q = query(&f, &id).await;
    let last = seen.last().unwrap();
    assert_eq!(normalize_callback(&json!({"task": q})), normalize_callback(last));
}

#[test]
fn reference_kinds_keep_content_order() {
    let body: CreateBody = serde_json::from_value(load_golden("create_r2v.body.json")).unwrap();
    let req = CreateEndpoint.normalize(body, &NormalizeCtx::new(at(0))).unwrap();
    let kinds: Vec<MediaKind> = req.references.iter().map(|r| r.kind).collect();
    assert_eq!(kinds, [MediaKind::Video, MediaKind::Audio]);
}
