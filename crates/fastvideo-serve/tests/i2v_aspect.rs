//! Image-to-video output aspect follows the input image (after EXIF
//! orientation) on every API, unless the request sets a size or aspect:
//! native `/fv/v1/jobs`, fal (`minimax/h3-*`, `lightricks/ltx-2.5`,
//! `fal-ai/wan`), MiniMax V2, `/v1/videos` and the LTX API. The engine is
//! the fake backend serving the real CUDA catalog's caps (canvas tiers,
//! multiples, budgets, aspect ranges), so the resolved canvases are the ones
//! a GPU server would produce. Jobs are read back from the store right after
//! submission; nothing is rendered.

#![cfg(all(feature = "openai-videos", feature = "minimax", feature = "fal", feature = "ltxapi"))]

use std::collections::BTreeMap;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use base64::Engine as _;
use fastvideo_engine_service::cuda::caps::{catalog, WeightLayout};
use fastvideo_engine_service::{EngineConfig, EngineService, FakeBackend, FakeConfig, FakeModel, FakeTiming, Mp4Mode};
use fastvideo_protocol::{Job, ProtocolId};
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-aspect";

async fn app(tag: &str) -> App {
    let dir = tempfile::Builder::new().prefix(&format!("fv-serve-aspect-{tag}-")).tempdir().unwrap().keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.limits.queue_max = 1000;
    for a in ["lightricks/ltx-2.5", "fal-ai/wan"] {
        c.protocols.fal_apps.push(a.to_owned());
    }
    c.validate().unwrap();
    // The real catalog's caps on the fake backend; the jobs never finish
    // within the test (slow steps, no MP4).
    let models = catalog(&WeightLayout::default())
        .into_iter()
        .map(|m| {
            let mut caps = m.caps();
            caps.resident = true;
            FakeModel { caps, recipe: m.describe() }
        })
        .collect();
    let fc = FakeConfig {
        models,
        timing: FakeTiming { load: std::time::Duration::ZERO, step: std::time::Duration::from_secs(60), ..FakeTiming::default() },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    };
    let cfg = EngineConfig { queue_max: 1000, output_dir: dir.join("engine-out"), ..EngineConfig::default() };
    let engine = EngineService::start(cfg, vec![Box::new(FakeBackend::new(fc))]).unwrap();
    let a = App::build(c, Overrides { engine: Some(engine), ..Default::default() }).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

async fn call(app: &Router, uri: &str, body: Value, auth: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("authorization", auth)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let r = app.clone().oneshot(req).await.unwrap();
    let status = r.status();
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 24).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn png(w: u32, h: u32) -> String {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([90, 120, 150]));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(out.into_inner()))
}

/// A JPEG stored `w`x`h` with EXIF orientation 6 (displayed rotated 90°
/// clockwise, so `h`x`w`): what a phone writes for a portrait photo.
fn jpeg_rotated(w: u32, h: u32) -> String {
    use image::ImageEncoder as _;
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([90, 120, 150]));
    let exif = vec![0x49, 0x49, 0x2a, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0];
    let mut out = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90);
    enc.set_exif_metadata(exif).unwrap();
    enc.write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8).unwrap();
    format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(out))
}

/// The test images: name, data URI, upright aspect (`w / h`).
fn images() -> Vec<(&'static str, String, f64)> {
    vec![
        ("landscape", png(320, 180), 16.0 / 9.0),
        ("portrait", png(180, 320), 9.0 / 16.0),
        ("square", png(200, 200), 1.0),
        ("extreme", png(600, 50), 12.0),
        ("exif-rotated", jpeg_rotated(320, 180), 9.0 / 16.0),
    ]
}

/// The delivered size of a stored job.
fn out_size(j: &Job) -> (u32, u32) {
    j.resolved.output_size()
}

/// The delivered canvas has the image's aspect (clamped to 1:4..4:1), to
/// within the canvas snap; and says so in the job's log.
fn assert_follows(j: &Job, image: f64, what: &str) {
    let (w, h) = out_size(j);
    let want = image.clamp(0.25, 4.0);
    let got = f64::from(w) / f64::from(h);
    assert!((got / want).ln().abs() < 0.08, "{what}: {w}x{h} for aspect {want:.3}");
    let note = j.logs.iter().find(|l| l.message.starts_with("canvas:")).unwrap_or_else(|| panic!("{what}: no canvas note"));
    assert_eq!(note.message.contains("clamped"), !(0.25..=4.0).contains(&image), "{what}: {}", note.message);
}

async fn job(a: &App, p: ProtocolId, id: &str) -> Job {
    a.ctx.jobs().by_external(p, id).await.unwrap_or_else(|| panic!("job {id}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_jobs_follow_the_image() {
    let a = app("native").await;
    let auth = format!("Bearer {KEY}");
    for model in ["h3-max", "h3-turbo", "h3-draft", "ltx-pro", "ltx-turbo", "wan-max", "wan-turbo"] {
        for (name, uri, aspect) in images() {
            let (s, v) = call(&a.router, "/fv/v1/jobs", json!({"model": model, "prompt": "a fox", "image_url": uri}), &auth).await;
            assert_eq!(s, 202, "{model} {name}: {v}");
            let j = job(&a, ProtocolId::Native, v["id"].as_str().unwrap()).await;
            assert_follows(&j, aspect, &format!("native {model} {name}"));
            assert!(v["notes"][0].as_str().unwrap().starts_with("canvas:"), "{v}");
        }
    }
    // `short_edge` alone picks the tier (H3 480, LTX 720).
    let (s, v) = call(&a.router, "/fv/v1/jobs", json!({"model": "h3-max", "prompt": "a fox", "image_url": png(180, 320), "short_edge": 480}), &auth).await;
    assert_eq!(s, 202, "{v}");
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(480), Some(832)));
    let (s, v) = call(&a.router, "/fv/v1/jobs", json!({"model": "ltx-pro", "prompt": "a fox", "image_url": png(180, 320), "short_edge": 720}), &auth).await;
    assert_eq!(s, 202, "{v}");
    let j = job(&a, ProtocolId::Native, v["id"].as_str().unwrap()).await;
    assert_eq!(out_size(&j), (720, 1280));
    // An explicit aspect or size still wins over the image.
    let (_, v) = call(&a.router, "/fv/v1/jobs", json!({"model": "h3-max", "prompt": "a fox", "image_url": png(180, 320), "aspect_ratio": "16:9", "short_edge": 768}), &auth).await;
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(1344), Some(768)), "{v}");
    let (_, v) = call(&a.router, "/fv/v1/jobs", json!({"model": "h3-max", "prompt": "a fox", "image_url": png(180, 320), "size": "1344x768"}), &auth).await;
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(1344), Some(768)), "{v}");
    // Text-to-video keeps 16:9; `short_edge` alone needs an image.
    let (_, v) = call(&a.router, "/fv/v1/jobs", json!({"model": "h3-max", "prompt": "a fox"}), &auth).await;
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(1344), Some(768)), "{v}");
    let (s, v) = call(&a.router, "/fv/v1/jobs", json!({"model": "h3-max", "prompt": "a fox", "short_edge": 480}), &auth).await;
    assert_eq!((s, v["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("short_edge")), "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fal_endpoints_follow_the_image() {
    let a = app("fal").await;
    let auth = format!("Key {KEY}");
    let endpoints: [(&str, Value); 7] = [
        ("/minimax/h3-max/image-to-video", json!({})),
        ("/minimax/h3-max/image-to-video", json!({"resolution": "1080P"})),
        ("/minimax/h3-turbo/image-to-video", json!({})),
        ("/minimax/h3-draft/image-to-video", json!({})),
        ("/lightricks/ltx-2.5/image-to-video/fast", json!({})),
        ("/lightricks/ltx-2.5/image-to-video/pro", json!({"aspect_ratio": "auto"})),
        ("/fal-ai/wan/v2.2-5b/image-to-video", json!({})),
    ];
    for (path, extra) in endpoints {
        for (name, uri, aspect) in images() {
            let mut body = json!({"prompt": "a fox", "image_url": uri});
            body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let (s, v) = call(&a.router, path, body, &auth).await;
            assert_eq!(s, 200, "{path} {extra} {name}: {v}");
            let j = job(&a, ProtocolId::Fal, v["request_id"].as_str().unwrap()).await;
            assert_follows(&j, aspect, &format!("fal {path} {extra} {name}"));
        }
    }
    // H3 1080P follows the image at the 1080 tier.
    let (_, v) = call(&a.router, "/minimax/h3-max/image-to-video", json!({"prompt": "a fox", "image_url": png(180, 320), "resolution": "1080P"}), &auth).await;
    let j = job(&a, ProtocolId::Fal, v["request_id"].as_str().unwrap()).await;
    assert_eq!(out_size(&j), (1080, 1920));
    // End frame only: the canvas follows the end frame.
    let (_, v) = call(&a.router, "/minimax/h3-max/image-to-video", json!({"prompt": "a fox", "end_image_url": png(180, 320)}), &auth).await;
    let j = job(&a, ProtocolId::Fal, v["request_id"].as_str().unwrap()).await;
    assert!(j.resolved.height > j.resolved.width);
    // An explicit aspect wins: LTX 16:9 and Wan 1:1 with a portrait image.
    let (_, v) = call(&a.router, "/lightricks/ltx-2.5/image-to-video/fast", json!({"prompt": "a fox", "image_url": png(180, 320), "aspect_ratio": "16:9"}), &auth).await;
    assert_eq!(out_size(&job(&a, ProtocolId::Fal, v["request_id"].as_str().unwrap()).await), (1920, 1080));
    let (_, v) = call(&a.router, "/fal-ai/wan/v2.2-5b/image-to-video", json!({"prompt": "a fox", "image_url": png(180, 320), "aspect_ratio": "1:1"}), &auth).await;
    let j = job(&a, ProtocolId::Fal, v["request_id"].as_str().unwrap()).await;
    assert_eq!(out_size(&j).0, out_size(&j).1);
    // H3 reference-to-video `adaptive` (the default) follows the first image.
    let (s, v) = call(&a.router, "/minimax/h3-max/reference-to-video", json!({"prompt": "a fox", "reference_image_urls": [png(180, 320)]}), &auth).await;
    assert_eq!(s, 200, "{v}");
    let j = job(&a, ProtocolId::Fal, v["request_id"].as_str().unwrap()).await;
    assert!(j.resolved.height > j.resolved.width);
    // The job log (fal's status `logs` once the job runs) carries the note.
    let (_, v) = call(&a.router, "/minimax/h3-max/image-to-video", json!({"prompt": "a fox", "image_url": png(600, 50)}), &auth).await;
    let j = job(&a, ProtocolId::Fal, v["request_id"].as_str().unwrap()).await;
    assert!(j.logs.iter().any(|l| l.message.contains("600x50") && l.message.contains("clamped")), "{:?}", j.logs);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn minimax_follows_the_image() {
    let a = app("minimax").await;
    let auth = format!("Bearer {KEY}");
    for (model, res) in [("MiniMax-H3", "768P"), ("MiniMax-H3-Turbo", "768P"), ("MiniMax-H3-Turbo", "480P")] {
        for (name, uri, aspect) in images() {
            // MiniMax i2v is always adaptive: a ratio sent along is ignored.
            let body = json!({"model": model, "resolution": res, "duration": 5, "ratio": "16:9",
                "content": [{"type": "text", "text": "a fox"}, {"type": "image_url", "image_url": {"url": uri}, "role": "first_frame"}]});
            let (s, v) = call(&a.router, "/v2/video_generation", body, &auth).await;
            assert_eq!(s, 200, "{model} {res} {name}: {v}");
            let j = job(&a, ProtocolId::MiniMaxV2, v["task_id"].as_str().unwrap()).await;
            assert_follows(&j, aspect, &format!("minimax {model} {res} {name}"));
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn openai_videos_follow_the_image() {
    let a = app("openai").await;
    let auth = format!("Bearer {KEY}");
    for model in ["h3-max", "h3-turbo", "ltx-pro", "wan-max"] {
        for (name, uri, aspect) in images() {
            let (s, v) = call(&a.router, "/v1/videos", json!({"model": model, "prompt": "a fox", "seconds": "5", "input_reference": uri}), &auth).await;
            assert_eq!(s, 200, "{model} {name}: {v}");
            let j = job(&a, ProtocolId::OpenAiVideos, v["id"].as_str().unwrap()).await;
            assert_follows(&j, aspect, &format!("/v1/videos {model} {name}"));
        }
    }
    // `short_edge` alone with an image picks the tier; `size` wins.
    let (s, v) = call(&a.router, "/v1/videos", json!({"model": "h3-max", "prompt": "a fox", "seconds": "5", "input_reference": png(180, 320), "short_edge": 480}), &auth).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["size"], "480x832");
    let (_, v) = call(&a.router, "/v1/videos", json!({"model": "h3-max", "prompt": "a fox", "seconds": "5", "input_reference": png(180, 320), "size": "1344x768"}), &auth).await;
    assert_eq!(v["size"], "1344x768");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ltx_api_keeps_its_required_resolution() {
    // The LTX API requires `resolution` (16:9 or 9:16 tiers); the image is
    // fitted to it, as upstream ("the image is resized to the configured
    // resolution").
    let a = app("ltxapi").await;
    let auth = format!("Bearer {KEY}");
    for (res, want) in [("1920x1080", (1920, 1080)), ("1080x1920", (1080, 1920))] {
        let (s, v) = call(&a.router, "/v2/image-to-video", json!({"prompt": "a fox", "model": "ltx-2-5-fast", "duration": 6, "resolution": res, "image_uri": png(180, 320)}), &auth).await;
        assert_eq!(s.as_u16() / 100, 2, "{v}");
        let id = v["id"].as_str().or(v["job_id"].as_str()).unwrap_or_else(|| panic!("{v}")).to_owned();
        let j = job(&a, ProtocolId::LtxV2, &id).await;
        assert_eq!(out_size(&j), want);
    }
}

#[tokio::test]
async fn console_director_defaults_to_the_image_aspect() {
    // The page renders the director schema's aspect_ratio (#14): the clip
    // director's enum offers `auto` first and defaults to it.
    let schema = include_str!("../../fastvideo-fal/src/catalog.rs");
    assert!(
        schema.contains(r#""enum": ["auto", "16:9", "9:16", "1:1"], "default": "auto""#),
        "the director form offers `auto` first, by default"
    );
    let js = include_str!("../console/director.js");
    assert!(js.contains("if (aspect.value && aspect.value !== 'auto') cfg.aspect_ratio = aspect.value;"), "`auto` sends no aspect_ratio");
}
