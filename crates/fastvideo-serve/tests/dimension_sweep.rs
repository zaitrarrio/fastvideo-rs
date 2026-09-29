//! Dimension sweep: every size, aspect, duration, frame count and frame rate
//! an API or the console offers, on every model, tier and app, with and
//! without input images of many shapes.
//!
//! The engine is the fake backend serving the real CUDA catalog (its caps:
//! canvas tiers, multiples, budgets, aspect ranges, frame grids, fps) with
//! the CUDA pipelines' own job checks ([`JobValidator::cuda`], the
//! functions `fastvideo_engine_service::cuda::validate` the GPU engine runs
//! before generating). The invariant, per combination:
//!
//! - accepted at submit ⇒ the resolved job passes the engine's checks
//!   (never a submit-accept followed by an engine failure);
//! - refused ⇒ a 4xx whose message names what is valid;
//! - offered by the console (the served fal schemas, the director form) or
//!   promised by the model's caps (native, `/v1/videos`) ⇒ accepted.
//!
//! Every violation is collected and printed as a table (model x API), then
//! the test fails listing them. `FV_SWEEP_REPORT=<path>` also writes the
//! table as TSV.

#![cfg(all(feature = "openai-videos", feature = "minimax", feature = "fal", feature = "ltxapi"))]

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use base64::Engine as _;
use fastvideo_engine_service::cuda::caps::{catalog, gate_h3_1080p, CudaModel, WeightLayout};
use fastvideo_engine_service::{EngineConfig, EngineService, FakeBackend, FakeConfig, FakeTiming, JobValidator, Mp4Mode};
use fastvideo_protocol::{
    negotiate, Anchor, CanvasSpec, GenerationRequest, Job, Keyframe, Length, MediaKind, MediaProbe, MediaRef,
    ModelCaps, ProtocolId, Ratio, Reference, Snap, StagedInputs, StagedMedia, Task, TimingSpec,
};
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use futures::StreamExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// Keys rotate so per-key rate limits (MiniMax: 300 creates a minute) never
/// refuse a combination.
const KEYS: usize = 96;

fn key(i: usize) -> String {
    format!("sk-sweep-{i}")
}

// ---------------------------------------------------------------- images

fn png(w: u32, h: u32) -> String {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([90, 120, 150]));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(out.into_inner()))
}

fn jpeg(w: u32, h: u32, exif_orientation: Option<u8>) -> String {
    use image::ImageEncoder as _;
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([90, 120, 150]));
    let mut out = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 60);
    if let Some(o) = exif_orientation {
        let exif = vec![0x49, 0x49, 0x2a, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, o, 0, 0, 0, 0, 0, 0, 0];
        enc.set_exif_metadata(exif).unwrap();
    }
    enc.write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8).unwrap();
    format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(out))
}

/// One input image: its name and data URI.
#[derive(Clone)]
struct Img {
    name: &'static str,
    uri: String,
}

/// The input shapes: landscape, portrait, square, 4:5, a 3:4 phone photo
/// (3024x4032), a 9:16 phone frame, a tall phone screenshot (1080x2340, the
/// 6:13 aspect of the D1 failure), extremes both ways, and a portrait photo
/// stored landscape with EXIF orientation 6.
fn images() -> Vec<Img> {
    vec![
        Img { name: "landscape-16:9", uri: png(320, 180) },
        Img { name: "portrait-9:16", uri: png(180, 320) },
        Img { name: "square", uri: png(200, 200) },
        Img { name: "4:5", uri: png(160, 200) },
        Img { name: "phone-3024x4032", uri: jpeg(3024, 4032, None) },
        Img { name: "phone-1080x1920", uri: jpeg(1080, 1920, None) },
        Img { name: "tall-1080x2340", uri: png(108, 234) },
        Img { name: "extreme-12:1", uri: png(600, 50) },
        Img { name: "extreme-1:12", uri: png(50, 600) },
        Img { name: "exif-rotated", uri: jpeg(320, 180, Some(6)) },
    ]
}

// ---------------------------------------------------------------- harness

/// How a combination is judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Offer {
    /// The console offers it (served fal schema, director form): must run.
    Console,
    /// The model's caps promise it (native, `/v1/videos`): must run.
    Caps,
    /// The API's schema lists it: runs, or a clear 4xx naming valid values.
    Api,
}

#[derive(Clone)]
struct Case {
    api: &'static str,
    /// Row label: the model or app the combination targets.
    model: String,
    combo: String,
    uri: String,
    body: Value,
    auth: String,
    protocol: ProtocolId,
    id_field: &'static str,
    offer: Offer,
}

#[derive(Clone, Debug)]
struct Violation {
    api: &'static str,
    model: String,
    combo: String,
    kind: &'static str,
    detail: String,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    cases: usize,
    accepted: usize,
    refused: usize,
    violations: usize,
}

#[derive(Default)]
struct Report {
    tally: BTreeMap<(String, &'static str), Tally>,
    violations: Vec<Violation>,
}

struct Sweep {
    app: App,
    /// Appended to every row label (`[no 1080P]`: a GPU below the H3
    /// 1080P tier's memory plan, where the tier is off).
    variant: &'static str,
    catalog: Vec<CudaModel>,
    validator: JobValidator,
    report: Mutex<Report>,
    next_key: std::sync::atomic::AtomicUsize,
}

/// The catalog a server on an 80 GB-class GPU serves (`hd`: the H3 1080P
/// tier on), or on a smaller one (the tier off, `gate_h3_1080p`).
fn served_catalog(hd: bool) -> Vec<CudaModel> {
    let mut cat = catalog(&WeightLayout::default());
    if !hd {
        gate_h3_1080p(&mut cat, None);
    }
    cat
}

async fn harness(hd: bool) -> Sweep {
    let dir = std::env::temp_dir().join(format!(
        "fv-serve-sweep-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut env = BTreeMap::new();
    let hashes: Vec<String> = (0..KEYS).map(|i| KeyRing::hash_hex(&key(i))).collect();
    env.insert("FV_API_KEYS".to_owned(), hashes.join(","));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.limits.queue_max = 1_000_000;
    for a in ["lightricks/ltx-2.5", "fal-ai/wan", "fal-ai/ltx-2.3-quality"] {
        c.protocols.fal_apps.push(a.to_owned());
    }
    c.validate().unwrap();
    let cat = served_catalog(hd);
    let fc = FakeConfig {
        // Jobs never finish within the sweep (each is cancelled once read).
        timing: FakeTiming { load: Duration::ZERO, step: Duration::from_secs(600), ..FakeTiming::default() },
        mp4: Mp4Mode::Off,
        ..FakeConfig::cuda_catalog(&cat)
    };
    let validator = fc.validator.clone().unwrap();
    let cfg = EngineConfig { queue_max: 1_000_000, output_dir: dir.join("engine-out"), ..EngineConfig::default() };
    let engine = EngineService::start(cfg, vec![Box::new(FakeBackend::new(fc))]).unwrap();
    let app = App::build(c, Overrides { engine: Some(engine), ..Default::default() }).await.unwrap();
    app.gate.engine().wait_ready().await;
    Sweep {
        app,
        variant: if hd { "" } else { " [no 1080P]" },
        catalog: cat,
        validator,
        report: Mutex::new(Report::default()),
        next_key: std::sync::atomic::AtomicUsize::new(0),
    }
}

async fn call(router: &Router, method: &str, uri: &str, body: Option<&Value>, auth: &str) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri).header("content-type", "application/json");
    if !auth.is_empty() {
        req = req.header("authorization", auth);
    }
    let req = req.body(body.map_or(Body::empty(), |b| Body::from(b.to_string()))).unwrap();
    let r = router.clone().oneshot(req).await.unwrap();
    let status = r.status();
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 26).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// A refusal "names what is valid": its message lists allowed values, a
/// range, or the nearest valid value.
fn clear(body: &Value) -> bool {
    let s = body.to_string().to_ascii_lowercase();
    [
        "supported", "allowed", "expected", "must be", "one of", "input should be", "nearest", "within", "maximum",
        "at most", "at least", "use ", "longest", "required", "not served", "not available", "not enabled",
        "is only", "only defined", "takes ", "needs ",
    ]
    .iter()
    .any(|k| s.contains(k))
        || range_named(&s)
}

/// `"5 to 15"`, `"9..=481"`: a message naming a numeric range.
fn range_named(s: &str) -> bool {
    let b = s.as_bytes();
    (0..b.len()).any(|i| {
        b[i].is_ascii_digit() && [" to ", "..=", "..", " - "].iter().any(|sep| s[i + 1..].starts_with(sep) && s[i + 1 + sep.len()..].starts_with(|c: char| c.is_ascii_digit()))
    })
}

impl Sweep {
    fn key(&self) -> String {
        let i = self.next_key.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        key(i % KEYS)
    }

    fn record(&self, c: &Case, accepted: bool, violation: Option<(&'static str, String)>) {
        let mut r = self.report.lock().unwrap();
        let label = format!("{}{}", c.model, self.variant);
        let t = r.tally.entry((label.clone(), c.api)).or_default();
        t.cases += 1;
        if accepted {
            t.accepted += 1;
        } else {
            t.refused += 1;
        }
        if let Some((kind, detail)) = violation {
            t.violations += 1;
            r.violations.push(Violation { api: c.api, model: label, combo: c.combo.clone(), kind, detail });
        }
    }

    /// Submits one combination and judges it.
    async fn run(&self, c: Case) {
        let router = &self.app.router;
        let auth = c.auth.replace("{KEY}", &self.key());
        let sync = c.protocol == ProtocolId::LtxV1;
        let before = if sync { self.newest(ProtocolId::LtxV1).await } else { None };
        let reply = if sync {
            // `/v1/*` blocks until the clip is done: a reply within the wait
            // is a refusal, a timeout an accepted (running) job.
            tokio::time::timeout(Duration::from_millis(1500), call(router, "POST", &c.uri, Some(&c.body), &auth)).await.ok()
        } else {
            Some(call(router, "POST", &c.uri, Some(&c.body), &auth).await)
        };
        let job = match &reply {
            None => {
                let mut j = None;
                for _ in 0..50 {
                    j = self.newest(ProtocolId::LtxV1).await.filter(|j| Some(j.id) != before.as_ref().map(|b| b.id));
                    if j.is_some() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                j
            }
            Some((s, v)) if s.is_success() => match v[c.id_field].as_str() {
                Some(id) => self.app.ctx.jobs().by_external(c.protocol, id).await,
                None => None,
            },
            Some(_) => None,
        };
        let Some(job) = job else {
            let (s, v) = reply.unwrap_or((StatusCode::OK, Value::Null));
            let violation = if s.is_server_error() || s.is_success() {
                Some(("5xx/odd reply", format!("{s}: {}", short(&v))))
            } else if c.offer != Offer::Api {
                Some((if c.offer == Offer::Console { "console offer refused" } else { "caps promise refused" }, format!("{s}: {}", short(&v))))
            } else if !clear(&v) {
                Some(("unclear 4xx", format!("{s}: {}", short(&v))))
            } else {
                None
            };
            self.record(&c, false, violation);
            return;
        };
        let r = &job.resolved;
        let violation = self.validator.check(r).err().map(|e| {
            (
                "accepted, engine refuses",
                format!("{} {}x{} {} frames @{} fps: {}", r.model, r.width, r.height, r.num_frames, r.fps, e.message),
            )
        });
        self.record(&c, true, violation);
        let _ = fastvideo_serve_kit::events::cancel_job(&self.app.ctx, job.id).await;
    }

    async fn newest(&self, p: ProtocolId) -> Option<Job> {
        let q = fastvideo_protocol::ListQuery { protocol: Some(p), limit: 1, ..Default::default() };
        self.app.ctx.jobs().list(q).await.items.into_iter().next()
    }

    async fn run_all(&self, cases: Vec<Case>) {
        // `/v1/*` (LTX sync) runs one at a time: an accepted job is found as
        // the newest LTX v1 job.
        let (sync, other): (Vec<Case>, Vec<Case>) = cases.into_iter().partition(|c| c.protocol == ProtocolId::LtxV1);
        futures::stream::iter(other).for_each_concurrent(24, |c| self.run(c)).await;
        for c in sync {
            self.run(c).await;
        }
    }


    /// Prints the table; returns the violations.
    fn finish(&self, what: &str) -> Vec<Violation> {
        let r = self.report.lock().unwrap();
        let mut table = format!("\n== {what}: model x API ==\nmodel\tapi\tcases\taccepted\trefused\tviolations\n");
        for ((m, api), t) in &r.tally {
            table += &format!("{m}\t{api}\t{}\t{}\t{}\t{}\n", t.cases, t.accepted, t.refused, t.violations);
        }
        let mut kinds: BTreeMap<(String, &str, &str), (usize, String, String)> = BTreeMap::new();
        for v in &r.violations {
            let e = kinds.entry((v.model.clone(), v.api, v.kind)).or_insert((0, v.combo.clone(), v.detail.clone()));
            e.0 += 1;
        }
        table += "\n-- violations (grouped: model, api, kind, count, first combination, detail) --\n";
        for ((m, api, kind), (n, combo, detail)) in &kinds {
            table += &format!("{m}\t{api}\t{kind}\t{n}\t{combo}\t{detail}\n");
        }
        println!("{table}");
        if let Ok(path) = std::env::var("FV_SWEEP_REPORT") {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
            f.write_all(table.as_bytes()).unwrap();
        }
        r.violations.clone()
    }
}

/// Runs `cases` on both catalogs (1080P tier on and off) and fails on any
/// violation.
fn check(what: &str, violations: Vec<Violation>) {
    assert!(violations.is_empty(), "{} violations in {what}; first: {:#?}", violations.len(), &violations[..violations.len().min(20)]);
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 300 {
        format!("{}…", &s[..300])
    } else {
        s
    }
}

fn with(base: &Value, extra: &[(&str, Value)]) -> Value {
    let mut b = base.clone();
    for (k, v) in extra {
        b[*k] = v.clone();
    }
    b
}

/// Standard aspects plus the ends of every range.
const ASPECTS: [&str; 10] = ["21:9", "16:9", "4:3", "1:1", "4:5", "3:4", "9:16", "9:21", "4:1", "1:4"];

fn ratio(a: &str) -> f64 {
    let (w, h) = a.split_once(':').unwrap();
    w.parse::<f64>().unwrap() / h.parse::<f64>().unwrap()
}

// ---------------------------------------------------------------- native

fn native_cases(s: &Sweep, imgs: &[Img]) -> Vec<Case> {
    let mut out = Vec::new();
    let case = |model: &str, combo: String, body: Value, offer: Offer| Case {
        api: "native",
        model: model.to_owned(),
        combo,
        uri: "/fv/v1/jobs".into(),
        body,
        auth: "Bearer {KEY}".into(),
        protocol: ProtocolId::Native,
        id_field: "id",
        offer,
    };
    for m in &s.catalog {
        let caps = m.caps();
        let id = m.id.as_str();
        let base = json!({"model": id, "prompt": "a fox runs"});
        // The input a task needs.
        let task_inputs: Vec<(&str, Vec<(&str, Value)>)> = {
            let mut v = Vec::new();
            if caps.supports(Task::T2V) {
                v.push(("t2v", vec![]));
            }
            if caps.supports(Task::I2V) {
                v.push(("i2v", vec![("image_url", json!(imgs[0].uri))]));
            }
            if caps.supports(Task::Ref2V) {
                v.push(("ref2v", vec![("reference_urls", json!([imgs[0].uri]))]));
            }
            v
        };
        let (plain, plain_in) = task_inputs[0].clone();
        // Canvas: every tier at every aspect (promised within the aspect range).
        for &se in &caps.canvas.short_edges {
            for a in ASPECTS {
                let promised = caps.canvas.aspect_ok((ratio(a) * 1e4) as u32, 10_000);
                let mut extra = plain_in.clone();
                extra.extend([("aspect_ratio", json!(a)), ("short_edge", json!(se))]);
                out.push(case(id, format!("{plain} aspect {a} short_edge {se}"), with(&base, &extra), if promised { Offer::Caps } else { Offer::Api }));
            }
        }
        out.push(case(id, format!("{plain} default canvas"), with(&base, &plain_in), Offer::Caps));
        // Image-conditioned: every image, default tier and every tier.
        for (task, _) in &task_inputs {
            let field = match *task {
                "i2v" => "image_url",
                "ref2v" => "reference_urls",
                _ => continue,
            };
            for img in imgs {
                let v = if field == "reference_urls" { json!([img.uri]) } else { json!(img.uri) };
                out.push(case(id, format!("{task} image {}", img.name), with(&base, &[(field, v.clone())]), Offer::Caps));
                for &se in &caps.canvas.short_edges {
                    out.push(case(id, format!("{task} image {} short_edge {se}", img.name), with(&base, &[(field, v.clone()), ("short_edge", json!(se))]), Offer::Caps));
                }
            }
            if caps.supports(Task::Keyframes) {
                for img in imgs.iter().take(3) {
                    out.push(case(id, format!("keyframes last {}", img.name), with(&base, &[("last_image_url", json!(img.uri))]), Offer::Caps));
                    out.push(case(id, format!("keyframes first+last {}", img.name), with(&base, &[("image_url", json!(imgs[1].uri)), ("last_image_url", json!(img.uri))]), Offer::Caps));
                }
            }
        }
        // Timing: every frame count on the grid, every fps, whole seconds.
        let g = &caps.frames;
        let fpss: Vec<u32> = if caps.fps.allowed.len() > 6 {
            vec![caps.fps.default, 4, 16, 24, 30, 60]
        } else {
            caps.fps.allowed.clone()
        };
        let mut n = g.min;
        while n <= g.max {
            let mut extra = plain_in.clone();
            extra.push(("num_frames", json!(n)));
            out.push(case(id, format!("{plain} num_frames {n}"), with(&base, &extra), Offer::Caps));
            n += g.step.max(1);
        }
        for &fps in &fpss {
            for secs in 1..=20u32 {
                let frames = (secs * fps).max(1);
                let promised = g.align_up(frames).is_some();
                let mut extra = plain_in.clone();
                extra.extend([("seconds", json!(secs)), ("fps", json!(fps))]);
                out.push(case(id, format!("{plain} {secs} s @ {fps} fps"), with(&base, &extra), if promised { Offer::Caps } else { Offer::Api }));
            }
            let mut extra = plain_in.clone();
            extra.extend([("num_frames", json!(g.max)), ("fps", json!(fps))]);
            out.push(case(id, format!("{plain} max frames @ {fps} fps"), with(&base, &extra), Offer::Caps));
        }
        // The largest canvas with the longest clip.
        if let Some(&top) = caps.canvas.short_edges.iter().max() {
            let mut extra = plain_in.clone();
            extra.extend([("aspect_ratio", json!("16:9")), ("short_edge", json!(top)), ("num_frames", json!(g.max))]);
            out.push(case(id, format!("{plain} 16:9 at {top}, {} frames", g.max), with(&base, &extra), Offer::Caps));
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn native_api() {
    let imgs = images();
    let mut bad = Vec::new();
    for hd in [true, false] {
        let s = harness(hd).await;
        let cases = native_cases(&s, &imgs);
        s.run_all(cases).await;
        bad.extend(s.finish("native /fv/v1/jobs"));
    }
    check("native /fv/v1/jobs", bad);
}

// ---------------------------------------------------------------- fal

/// The dimension fields of a schema property: its enum, or every integer in
/// its range.
fn values_of(p: &Value) -> Vec<Value> {
    let e = p.get("enum").or_else(|| p.pointer("/anyOf/0/enum"));
    if let Some(list) = e.and_then(Value::as_array) {
        return list.clone();
    }
    match (p.get("minimum").and_then(Value::as_i64), p.get("maximum").and_then(Value::as_i64)) {
        (Some(lo), Some(hi)) if p["type"] == "integer" && hi - lo <= 600 => (lo..=hi).map(Value::from).collect(),
        _ => Vec::new(),
    }
}

const CANVAS_FIELDS: [&str; 2] = ["resolution", "aspect_ratio"];
const TIMING_FIELDS: [&str; 4] = ["duration", "fps", "num_frames", "frames_per_second"];

fn product(fields: &[(String, Vec<Value>)]) -> Vec<Vec<(String, Value)>> {
    let mut out = vec![Vec::new()];
    for (k, vs) in fields {
        let mut next = Vec::new();
        for prefix in &out {
            for v in vs {
                let mut p = prefix.clone();
                p.push((k.clone(), v.clone()));
                next.push(p);
            }
        }
        out = next;
    }
    out
}

async fn fal_cases(s: &Sweep, imgs: &[Img]) -> Vec<Case> {
    let router = &s.app.router;
    let (st, cat) = call(router, "GET", "/fal/schema", None, "").await;
    assert_eq!(st, 200, "{cat}");
    let mut out = Vec::new();
    for app in cat["apps"].as_array().unwrap() {
        let app_id = app["id"].as_str().unwrap();
        let kind: fastvideo_fal::AppKind = serde_json::from_value(app["kind"].clone()).unwrap();
        for ep in app["endpoints"].as_array().unwrap() {
            let sub = ep["sub"].as_str().unwrap();
            let endpoint = fastvideo_fal::Endpoint::from_sub(sub).unwrap();
            // What the API lists (fal's own schema) and what the console form
            // offers (the schema this server serves for the form).
            let api = fastvideo_fal::catalog::input_schema_for(kind, endpoint);
            let (st, served) = call(router, "GET", &format!("/fal/schema/{app_id}/{sub}"), None, "").await;
            assert_eq!(st, 200, "{served}");
            let offered = |k: &str, v: &Value| -> bool {
                let Some(p) = served["properties"].get(k) else { return false };
                let vs = values_of(p);
                vs.is_empty() || vs.contains(v)
            };
            let props = api["properties"].as_object().unwrap();
            let label = format!("{app_id}/{sub}");
            let base = json!({"prompt": "A reference sheet: a fox. Generated video: the fox runs"});
            // Image inputs this endpoint takes.
            let required: Vec<&str> = api["required"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
            let image_fields: Vec<&str> = props
                .iter()
                .filter(|(_, p)| p["x-fv-media"] == "image")
                .map(|(k, _)| k.as_str())
                .collect();
            let mut inputs: Vec<(String, Vec<(String, Value)>)> = Vec::new();
            let needs_image = image_fields.iter().any(|f| required.contains(f)) || api.get("x-fv-min-references").is_some();
            if !needs_image {
                inputs.push(("no image".into(), vec![]));
            }
            // A required image field always carries an image (the landscape
            // one when another field is the one varied).
            let base_inputs: Vec<(String, Value)> = image_fields
                .iter()
                .filter(|f| required.contains(f))
                .map(|f| (f.to_string(), json!(imgs[0].uri)))
                .collect();
            for img in imgs {
                for f in &image_fields {
                    let v = if props[*f]["type"] == "array" { json!([img.uri]) } else { json!(img.uri) };
                    let mut set: Vec<(String, Value)> = base_inputs.iter().filter(|(k, _)| k != f).cloned().collect();
                    set.push((f.to_string(), v));
                    inputs.push((format!("{f}={}", img.name), set));
                }
            }
            if image_fields.contains(&"end_image_url") && image_fields.contains(&"image_url") {
                inputs.push(("first+last".into(), vec![("image_url".into(), json!(imgs[0].uri)), ("end_image_url".into(), json!(imgs[1].uri))]));
            }
            let first_input = inputs.iter().find(|(n, _)| n != "no image").cloned().unwrap_or_else(|| inputs[0].clone());
            let dims = |fields: &[&str]| -> Vec<(String, Vec<Value>)> {
                fields.iter().filter_map(|f| props.get(*f).map(|p| (f.to_string(), values_of(p)))).filter(|(_, v)| !v.is_empty()).collect()
            };
            let mut push = |combo: Vec<(String, Value)>, input: &(String, Vec<(String, Value)>)| {
                let mut body = base.clone();
                for (k, v) in combo.iter().chain(&input.1) {
                    body[k.as_str()] = v.clone();
                }
                let console = combo.iter().all(|(k, v)| offered(k, v));
                let desc = combo.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ");
                out.push(Case {
                    api: "fal",
                    model: label.clone(),
                    combo: format!("{desc} [{}]", input.0),
                    uri: format!("/{app_id}/{sub}"),
                    body,
                    auth: "Key {KEY}".into(),
                    protocol: ProtocolId::Fal,
                    id_field: "request_id",
                    offer: if console { Offer::Console } else { Offer::Api },
                });
            };
            // Canvas fields x every input, at the default timing.
            let canvas = dims(&CANVAS_FIELDS);
            for combo in product(&canvas) {
                for input in &inputs {
                    push(combo.clone(), input);
                }
            }
            // Timing fields x resolution, with the first input. Frame counts
            // and rates with a wide range are swept one at a time.
            let res = dims(&["resolution"]);
            let timing = dims(&TIMING_FIELDS);
            let wide: Vec<_> = timing.iter().filter(|(_, v)| v.len() > 12).cloned().collect();
            let narrow: Vec<_> = timing.iter().filter(|(_, v)| v.len() <= 12).cloned().collect();
            let mut grid = res.clone();
            grid.extend(narrow);
            for combo in product(&grid) {
                push(combo, &first_input);
            }
            for w in &wide {
                let mut grid = res.clone();
                grid.push(w.clone());
                for combo in product(&grid) {
                    push(combo, &first_input);
                }
            }
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn fal_apps() {
    let imgs = images();
    let mut bad = Vec::new();
    for hd in [true, false] {
        let s = harness(hd).await;
        let cases = fal_cases(&s, &imgs).await;
        s.run_all(cases).await;
        bad.extend(s.finish("fal apps"));
    }
    check("fal apps", bad);
}

// ---------------------------------------------------------------- director

/// The director's configure: every resolution and aspect its served form
/// offers (plus `auto` following each image), at every chunk length, built
/// into the chunk job the session submits and checked by the engine.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fal_director_configure() {
    let mut bad = Vec::new();
    for hd in [true, false] {
        let s = harness(hd).await;
        director_sweep(&s).await;
        bad.extend(s.finish("fal director configure"));
    }
    check("fal director configure", bad);
}

async fn director_sweep(s: &Sweep) {
    use fastvideo_fal::director::messages::{Aspect, Resolution};
    use fastvideo_fal::director::service::{canvas_for, limits_for};
    let router = &s.app.router;
    let (_, cat) = call(router, "GET", "/fal/schema", None, "").await;
    let cfg = fastvideo_fal::director::DirectorConfig::default();
    for app in cat["apps"].as_array().unwrap().iter().filter(|a| a["director"] == true) {
        let app_id = app["id"].as_str().unwrap();
        let (st, form) = call(router, "GET", &format!("/fal/schema/{app_id}/director"), None, "").await;
        let label = format!("{app_id}/director");
        let c = Case {
            api: "fal director",
            model: label.clone(),
            combo: String::new(),
            uri: String::new(),
            body: Value::Null,
            auth: String::new(),
            protocol: ProtocolId::FalDirector,
            id_field: "",
            offer: Offer::Console,
        };
        if st != 200 {
            s.record(&Case { combo: "form schema".into(), ..c.clone() }, false, Some(("console offer refused", format!("GET /fal/schema/{app_id}/director: {st} {form}"))));
            continue;
        }
        let model = app["model"].as_str().unwrap();
        let models = s.app.ctx.engine().models();
        let caps = fastvideo_protocol::resolve_model(model, |n| s.app.ctx.engine().alias(n), &models).ok().cloned().or_else(|| {
            let tier = app["tier"].as_str().and_then(|t| serde_json::from_value(json!(t)).ok())?;
            fastvideo_protocol::resolve_tier(fastvideo_protocol::Family::H3, tier, &models).ok().cloned()
        });
        let caps = caps.unwrap_or_else(|| panic!("{app_id}: no model"));
        let limits = limits_for(&cfg, &caps);
        let res: Vec<Resolution> = values_of(&form["properties"]["resolution"]).iter().map(|v| serde_json::from_value(v.clone()).unwrap()).collect();
        let aspects: Vec<String> = values_of(&form["properties"]["aspect_ratio"]).iter().map(|v| v.as_str().unwrap().to_owned()).collect();
        assert!(!res.is_empty() && !aspects.is_empty(), "{form}");
        let chunk_s: Vec<f64> = (limits.min_chunk_seconds.ceil() as u32..=limits.max_chunk_seconds.floor() as u32).map(f64::from).collect();
        for r in &res {
            let served = limits.resolutions.contains(r);
            for a in &aspects {
                // `auto` follows the image at the nearest served aspect.
                let follow: Vec<Aspect> = if a == "auto" {
                    vec![Aspect::Landscape, Aspect::Portrait, Aspect::Square]
                } else {
                    vec![serde_json::from_value(json!(a)).unwrap()]
                };
                for asp in follow {
                    for &secs in &chunk_s {
                        let combo = format!("resolution={} aspect_ratio={a} ({}) chunk {secs} s", r.as_str(), asp.as_str());
                        let case = Case { combo, ..c.clone() };
                        if !served {
                            s.record(&case, false, Some(("console offer refused", format!("{} is not served by {}", r.as_str(), caps.id))));
                            continue;
                        }
                        let (w, h) = canvas_for(&caps, *r, asp);
                        let frames = fastvideo_fal::director::engine::frames_for(&caps, limits.fps, secs).unwrap_or(caps.frames.default);
                        let job = chunk_job(&caps, w, h, frames, limits.fps);
                        let v = s.validator.check(&job).err().map(|e| ("accepted, engine refuses", format!("{w}x{h} {frames} frames: {}", e.message)));
                        s.record(&case, true, v);
                    }
                }
            }
        }
    }
}

fn chunk_job(caps: &ModelCaps, w: u32, h: u32, frames: u32, fps: u32) -> fastvideo_protocol::ResolvedJob {
    let mut req = GenerationRequest::text(ProtocolId::FalDirector, caps.id.0.clone(), "a fox");
    req.canvas = CanvasSpec::Exact { width: w, height: h };
    req.timing = TimingSpec { length: Length::ModelDefault, fps: Some(fps) };
    let mut job = negotiate(&GenerationRequest { canvas: CanvasSpec::ModelDefault, ..req }, caps, &StagedInputs::default()).unwrap();
    (job.width, job.height, job.num_frames, job.fps) = (w, h, frames, fps);
    job
}

// ---------------------------------------------------------------- MiniMax

const MINIMAX_MODELS: [&str; 4] = ["MiniMax-H3", "MiniMax-H3-Max", "MiniMax-H3-Turbo", "MiniMax-H3-Draft"];
const MINIMAX_RES: [Option<&str>; 4] = [None, Some("480P"), Some("768P"), Some("2K")];
const MINIMAX_RATIOS: [Option<&str>; 8] = [None, Some("adaptive"), Some("21:9"), Some("16:9"), Some("4:3"), Some("1:1"), Some("3:4"), Some("9:16")];

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn minimax_v2() {
    let mut bad = Vec::new();
    for hd in [true, false] {
        let s = harness(hd).await;
        minimax_sweep(&s).await;
        bad.extend(s.finish("MiniMax V2"));
    }
    check("MiniMax V2", bad);
}

async fn minimax_sweep(s: &Sweep) {
    let imgs = images();
    let text = json!({"type": "text", "text": "a fox runs"});
    let item = |uri: &str, role: &str| json!({"type": "image_url", "image_url": {"url": uri}, "role": role});
    let mut contents: Vec<(String, Value)> = vec![("text only".into(), json!([text]))];
    for img in &imgs {
        contents.push((format!("first_frame {}", img.name), json!([text, item(&img.uri, "first_frame")])));
        contents.push((format!("reference_image {}", img.name), json!([text, item(&img.uri, "reference_image")])));
    }
    for img in imgs.iter().take(3) {
        contents.push((format!("last_frame {}", img.name), json!([text, item(&img.uri, "last_frame")])));
        contents.push((format!("first+last {}", img.name), json!([text, item(&imgs[1].uri, "first_frame"), item(&img.uri, "last_frame")])));
    }
    let mut cases = Vec::new();
    let case = |model: &str, combo: String, body: Value| Case {
        api: "minimax v2",
        model: model.to_owned(),
        combo,
        uri: "/v2/video_generation".into(),
        body,
        auth: "Bearer {KEY}".into(),
        protocol: ProtocolId::MiniMaxV2,
        id_field: "task_id",
        offer: Offer::Api,
    };
    for model in MINIMAX_MODELS {
        for res in MINIMAX_RES {
            for ratio in MINIMAX_RATIOS {
                for (cname, content) in &contents {
                    let mut b = json!({"model": model, "duration": 5, "content": content});
                    if let Some(r) = res {
                        b["resolution"] = json!(r);
                    }
                    if let Some(r) = ratio {
                        b["ratio"] = json!(r);
                    }
                    cases.push(case(model, format!("resolution={res:?} ratio={ratio:?} {cname}"), b));
                }
            }
            for d in 3..=16 {
                let mut b = json!({"model": model, "duration": d, "ratio": "16:9", "content": [text]});
                if let Some(r) = res {
                    b["resolution"] = json!(r);
                }
                cases.push(case(model, format!("resolution={res:?} duration={d} t2v"), b.clone()));
                let mut b = json!({"model": model, "duration": d, "content": [text, item(&imgs[1].uri, "first_frame")]});
                if let Some(r) = res {
                    b["resolution"] = json!(r);
                }
                cases.push(case(model, format!("resolution={res:?} duration={d} i2v"), b));
            }
        }
    }
    s.run_all(cases).await;
}

// ---------------------------------------------------------------- LTX API

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn ltx_api() {
    let s = harness(true).await;
    let imgs = images();
    let models = ["ltx-2-5-pro", "ltx-2-3-pro", "ltx-2-5-fast", "ltx-2-3-fast", "ltx-turbo", "ltx-draft"];
    let resolutions = fastvideo_ltxapi::models::resolution_strings();
    let mut cases = Vec::new();
    for (api, prefix, protocol, id_field) in [("ltx v2", "/v2", ProtocolId::LtxV2, "id"), ("ltx v1", "/v1", ProtocolId::LtxV1, "")] {
        for model in models {
            let case = |ep: &str, combo: String, body: Value| Case {
                api,
                model: model.to_owned(),
                combo,
                uri: format!("{prefix}/{ep}"),
                body,
                auth: "Bearer {KEY}".into(),
                protocol,
                id_field,
                offer: Offer::Api,
            };
            for res in &resolutions {
                for fps in [24, 25, 48, 50] {
                    for d in (6..=20).step_by(2) {
                        if api == "ltx v1" && !(res == "1920x1080" && fps == 25 && (d == 6 || d == 20)) {
                            continue; // v1 blocks per accepted job: a slice of the matrix
                        }
                        let b = json!({"prompt": "a fox", "model": model, "resolution": res, "fps": fps, "duration": d});
                        cases.push(case("text-to-video", format!("t2v {res} {fps} fps {d} s"), b));
                    }
                }
                for img in &imgs {
                    if api == "ltx v1" && (img.name != "portrait-9:16" || res != "1920x1080") {
                        continue;
                    }
                    let b = json!({"prompt": "a fox", "model": model, "resolution": res, "duration": 6, "image_uri": img.uri});
                    cases.push(case("image-to-video", format!("i2v {res} image {}", img.name), b));
                }
                if api == "ltx v2" {
                    let b = json!({"prompt": "a fox", "model": model, "resolution": res, "duration": 8, "image_uri": imgs[0].uri, "last_frame_uri": imgs[1].uri});
                    cases.push(case("image-to-video", format!("keyframes {res}"), b));
                }
            }
        }
    }
    s.run_all(cases).await;
    check("LTX API", s.finish("LTX API"));
}

// ---------------------------------------------------------------- /v1/videos

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn openai_videos() {
    let mut bad = Vec::new();
    for hd in [true, false] {
        let s = harness(hd).await;
        openai_sweep(&s).await;
        bad.extend(s.finish("/v1/videos"));
    }
    check("/v1/videos", bad);
}

async fn openai_sweep(s: &Sweep) {
    let imgs = images();
    let (st, list) = call(&s.app.router, "GET", "/v1/models", None, &format!("Bearer {}", key(0))).await;
    assert_eq!(st, 200, "{list}");
    let names: Vec<String> = list["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap().to_owned()).collect();
    let engine = s.app.ctx.engine().clone();
    let mut cases = Vec::new();
    for name in &names {
        let Ok(caps) = fastvideo_openai_videos::models::resolve_public(engine.as_ref(), name) else { continue };
        if matches!(caps.stream, Some(fastvideo_protocol::StreamCaps::Causal { .. })) && !caps.supports(Task::T2V) {
            continue;
        }
        let h3 = caps.family == fastvideo_protocol::Family::H3;
        let case = |combo: String, body: Value, offer: Offer| Case {
            api: "/v1/videos",
            model: name.clone(),
            combo,
            uri: "/v1/videos".into(),
            body,
            auth: "Bearer {KEY}".into(),
            protocol: ProtocolId::OpenAiVideos,
            id_field: "id",
            offer,
        };
        let (t2v, plain): (&str, Vec<(&str, Value)>) = if caps.supports(Task::T2V) {
            ("t2v", vec![])
        } else if h3 {
            ("ref2v", vec![("task", json!("ref2va")), ("image_reference", json!([{"image_url": imgs[0].uri}]))])
        } else {
            continue; // LTX reference-to-video is not on this API (`task` is H3-only)
        };
        let base = json!({"model": name, "prompt": "a fox runs", "seconds": "5"});
        for &se in &caps.canvas.short_edges {
            for a in ASPECTS {
                let promised = caps.canvas.aspect_ok((ratio(a) * 1e4) as u32, 10_000);
                let mut extra = plain.clone();
                extra.extend([("aspect_ratio", json!(a)), ("short_edge", json!(se))]);
                cases.push(case(format!("{t2v} aspect {a} short_edge {se}"), with(&base, &extra), if promised { Offer::Caps } else { Offer::Api }));
            }
        }
        cases.push(case(format!("{t2v} default"), with(&base, &plain), Offer::Caps));
        if caps.supports(Task::I2V) {
            for img in &imgs {
                cases.push(case(format!("i2v {}", img.name), with(&base, &[("input_reference", json!(img.uri))]), Offer::Caps));
                for &se in &caps.canvas.short_edges {
                    cases.push(case(format!("i2v {} short_edge {se}", img.name), with(&base, &[("input_reference", json!(img.uri)), ("short_edge", json!(se))]), Offer::Caps));
                }
            }
        }
        let g = &caps.frames;
        for secs in 1..=20u32 {
            // FastVideo's H3 floor is 5 s on this API.
            let f = secs * caps.fps.default;
            let promised = g.align_up(f).is_some() && !(h3 && secs < 5);
            let mut extra = plain.clone();
            extra.push(("seconds", json!(secs.to_string())));
            cases.push(case(format!("{t2v} seconds {secs}"), with(&base, &extra), if promised { Offer::Caps } else { Offer::Api }));
        }
        let mut n = g.min;
        while n <= g.max {
            let promised = !(h3 && n < g.next_on_grid(5 * caps.fps.default).unwrap_or(0));
            let mut extra = plain.clone();
            extra.push(("num_frames", json!(n)));
            let mut b = with(&base, &extra);
            b.as_object_mut().unwrap().remove("seconds");
            cases.push(case(format!("{t2v} num_frames {n}"), b, if promised { Offer::Caps } else { Offer::Api }));
            n += g.step.max(1);
        }
    }
    s.run_all(cases).await;
}

// ---------------------------------------------------------------- negotiate

/// `negotiate` over the real catalog, exhaustively: every model, every
/// canvas tier, every image aspect `n:d` for `n, d` in 1..=48 (plus the
/// D1 case), every standard aspect, every frame count and whole-second
/// length at every rate. Whatever negotiate accepts, the engine must.
#[test]
fn negotiate_matches_the_engine() {
    for hd in [true, false] {
        negotiate_sweep(&served_catalog(hd));
    }
}

fn negotiate_sweep(cat: &[CudaModel]) {
    let validator = JobValidator::cuda(cat);
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let image = |w: u32, h: u32| StagedMedia {
        path: "/tmp/in.png".into(),
        mime: "image/png".into(),
        bytes: 1,
        probe: MediaProbe { width: Some(w), height: Some(h), ..Default::default() },
    };
    for m in cat {
        let caps = m.caps();
        let task = [Task::I2V, Task::Ref2V].into_iter().find(|t| caps.supports(*t));
        let mut check = |req: &GenerationRequest, staged: &StagedInputs, what: String| {
            if let Ok(job) = negotiate(req, &caps, staged) {
                checked += 1;
                if let Err(e) = validator.check(&job) {
                    bad.push(format!("{} {what}: {}x{} {} frames @{}: {}", caps.id, job.width, job.height, job.num_frames, job.fps, e.message));
                }
            }
        };
        let t2v = if caps.supports(Task::T2V) { Some(GenerationRequest::text(ProtocolId::Native, caps.id.0.clone(), "a fox")) } else { None };
        // Image-derived canvases.
        if let Some(task) = task {
            let mut req = GenerationRequest::text(ProtocolId::Native, caps.id.0.clone(), "a fox");
            req.task = task;
            let media = MediaRef::DataUri("data:image/png;base64,".into());
            let mut staged = StagedInputs::default();
            if task == Task::I2V {
                req.keyframes = vec![Keyframe { at: Anchor::First, image: media }];
            } else {
                req.references = vec![Reference { kind: MediaKind::Image, media }];
            }
            let mut dims: Vec<(u32, u32)> = (1..=48u32).flat_map(|n| (1..=48u32).map(move |d| (n * 97, d * 97))).collect();
            dims.extend([(1080, 2340), (3024, 4032), (4032, 3024), (600, 50), (50, 600), (1, 1000), (1000, 1)]);
            for &se in &caps.canvas.short_edges {
                for &(w, h) in &dims {
                    if task == Task::I2V {
                        staged.keyframes = vec![(Anchor::First, image(w, h))];
                    } else {
                        staged.references = vec![(MediaKind::Image, image(w, h))];
                    }
                    req.canvas = CanvasSpec::FollowImage { short_edge: se };
                    check(&req, &staged, format!("image {w}x{h} at {se}"));
                }
            }
        }
        if let Some(mut req) = t2v {
            for &se in &caps.canvas.short_edges {
                for n in 1..=24u32 {
                    for d in 1..=24u32 {
                        req.canvas = CanvasSpec::Aspect { ratio: Ratio::new(n, d), short_edge: se };
                        check(&req, &StagedInputs::default(), format!("aspect {n}:{d} at {se}"));
                    }
                }
            }
            req.canvas = CanvasSpec::ModelDefault;
            let g = caps.frames.clone();
            for &fps in &caps.fps.allowed {
                for n in 1..=g.max + 20 {
                    req.timing = TimingSpec { length: Length::Frames { value: n, snap: Snap::AlignUp }, fps: Some(fps) };
                    check(&req, &StagedInputs::default(), format!("{n} frames align-up @{fps}"));
                    req.timing = TimingSpec { length: Length::Frames { value: n, snap: Snap::Exact }, fps: Some(fps) };
                    check(&req, &StagedInputs::default(), format!("{n} frames exact @{fps}"));
                }
                for secs in 1..=30u32 {
                    req.timing = TimingSpec { length: Length::Seconds { value: f64::from(secs), snap: Snap::AlignUp }, fps: Some(fps) };
                    check(&req, &StagedInputs::default(), format!("{secs} s @{fps}"));
                }
            }
        }
    }
    assert!(checked > 10_000, "{checked}");
    assert!(bad.is_empty(), "{} of {checked} negotiated jobs fail the engine checks; first:\n{}", bad.len(), bad[..bad.len().min(40)].join("\n"));
}

/// The fake backend runs the engine checks: a job the CUDA engine would
/// refuse fails on the fake too (the wiring the sweep relies on).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_fake_engine_runs_the_cuda_checks() {
    // A GPU without the 1080P tier: the trained 768x1344 budget applies.
    let cat = served_catalog(false);
    let fc = FakeConfig { timing: FakeTiming { load: Duration::ZERO, step: Duration::ZERO, ..FakeTiming::default() }, mp4: Mp4Mode::Off, ..FakeConfig::cuda_catalog(&cat) };
    let dir = std::env::temp_dir().join(format!("fv-sweep-wiring-{}", std::process::id()));
    let cfg = EngineConfig { output_dir: dir, ..EngineConfig::default() };
    let engine = EngineService::start(cfg, vec![Box::new(FakeBackend::new(fc))]).unwrap();
    engine.wait_ready().await;
    let caps = cat.iter().find(|m| m.id.as_str() == "sol-h3").unwrap().caps();
    let mut job = negotiate(&GenerationRequest::text(ProtocolId::Native, "sol-h3", "a fox"), &caps, &StagedInputs::default()).unwrap();
    // The D1 failure: 704x1504 is above the trained 768x1344 budget.
    (job.width, job.height) = (704, 1504);
    let h = engine.submit(fastvideo_protocol::JobId::new(), job, fastvideo_engine_service::Priority::Batch).await.unwrap();
    let e = h.wait().await.expect_err("the engine check refuses the canvas");
    assert!(e.message.contains("1058816 pixels"), "{}", e.message);
}
