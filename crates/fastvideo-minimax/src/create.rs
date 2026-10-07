//! `POST /v2/video_generation` (design §4.3, research §1.3).
//!
//! Body: `model`, `content[]`, `resolution`, `duration`, `ratio`, `extra`,
//! `callback_url`. Reply: 200 `{"task_id": "<18 digits>"}` (no `base_resp`).
//!
//! Mode by `content[]` roles (research §1.3.1):
//!
//! | Content | Task |
//! |---|---|
//! | one `text` | `T2V` (t2va): `ratio` required, not `adaptive` |
//! | `text` + `image_url` `first_frame` (or one roleless image) | `I2V`: always `adaptive` (other valid ratios ignored) |
//! | `text` + `last_frame` (± `first_frame`) | `Keyframes` (fl2va): always `adaptive` |
//! | `text` + `reference_image` ≤ 9 / `reference_video` ≤ 3 / `reference_audio` ≤ 3 (≤ 12) | `Ref2V`, **content order kept**: `ratio` optional, default `adaptive` |
//!
//! Frame roles and reference roles cannot be mixed. A roleless `video_url`
//! / `audio_url` is a reference (the only role its type allows). URLs are
//! `http(s)`, `data:<type>/<fmt>;base64,…` or `mm_file://…` (→ 400
//! `Unsupported(ProviderFiles)`: we have no MiniMax file store).
//!
//! `POST /v2/h3_context_ir` and `POST /v2/video_regeneration` answer 400
//! "not supported by this server (2013)" (design §1.2).

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use fastvideo_models::h3::config::H3Geometry;
use fastvideo_protocol::{
    negotiate_noted, precheck, Anchor, ApiError, BatchProtocol, CallbackSpec, CanvasSpec, ErrorCtx,
    Family, GapId, GenerationRequest, HttpReply, Job, JobId, KeyId, Keyframe, Length, ListQuery,
    MediaKind, MediaRef, ModelCaps, NormalizeCtx, ProtocolId, Ratio, Reference, Snap,
    StagedInputs, SubmitEndpoint, Task, TimingSpec, ViewCtx,
};
use fastvideo_serve_kit::handlers::{error_reply, into_response};
use fastvideo_serve_kit::{random_token, ServeCtx};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{MiniMax, MiniMaxModel, Resolution};

/// Longest `text` item (research §1.3.1).
pub const MAX_PROMPT_CHARS: usize = 7000;
/// Reference limits (research §1.3.1): images, videos, audio, total.
pub const MAX_REFS: (usize, usize, usize, usize) = (9, 3, 3, 12);
/// `ratio` values (research §1.3).
pub const RATIOS: [&str; 7] = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
/// `extra.prompt_expansion_mode` values.
pub const PROMPT_EXPANSION_MODES: [&str; 3] = ["disabled", "balanced", "quality"];

/// `VideoGenerationV2Req`. Unknown top-level fields are ignored (the spec
/// does not say; strobe's adapter ignores them).
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct CreateBody {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<ContentItem>>,
    #[serde(default)]
    pub resolution: Option<String>,
    /// Kept raw so a non-integer gets a MiniMax-style message.
    #[serde(default)]
    pub duration: Option<Value>,
    #[serde(default)]
    pub ratio: Option<String>,
    #[serde(default)]
    pub extra: Option<Value>,
    #[serde(default)]
    pub callback_url: Option<String>,
}

/// `ContentItem`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct ContentItem {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<MediaUrl>,
    #[serde(default)]
    pub video_url: Option<MediaUrl>,
    #[serde(default)]
    pub audio_url: Option<MediaUrl>,
    #[serde(default)]
    pub role: Option<String>,
}

/// `{"url": "…"}`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct MediaUrl {
    #[serde(default)]
    pub url: Option<String>,
}

/// The create endpoint (pure `normalize` / `submit_reply`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CreateEndpoint;

impl SubmitEndpoint for CreateEndpoint {
    type Body = CreateBody;

    fn normalize(&self, b: CreateBody, _cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError> {
        normalize(b)
    }

    fn submit_reply(&self, job: &Job, _cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, json!({ "task_id": job.external_id }))
    }
}

/// What `content[]` asks for.
#[derive(Clone, Debug, Default, PartialEq)]
struct Content {
    prompt: String,
    first: Vec<MediaRef>,
    last: Vec<MediaRef>,
    refs: Vec<Reference>,
}

fn bad(param: impl Into<String>, msg: impl Into<String>) -> ApiError {
    ApiError::invalid_param(param, msg)
}

/// The pure part of create: body -> [`GenerationRequest`] with the
/// MiniMax name in `model` (resolved later by [`MiniMax::resolve_caps`]).
pub fn normalize(b: CreateBody) -> Result<GenerationRequest, ApiError> {
    let name = b
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad("model", "model is required"))?;
    let model = MiniMaxModel::parse(name).ok_or_else(|| {
        let all: Vec<&str> = MiniMaxModel::ALL.iter().map(|m| m.as_str()).collect();
        bad("model", format!("model `{name}` is not supported; expected one of {}", all.join(", ")))
    })?;

    let content = parse_content(b.content.as_deref().unwrap_or_default())?;
    let task = if !content.refs.is_empty() {
        Task::Ref2V
    } else if !content.last.is_empty() {
        Task::Keyframes
    } else if !content.first.is_empty() {
        Task::I2V
    } else {
        Task::T2V
    };

    let resolution = match b.resolution.as_deref() {
        None if model.resolution_optional() => Resolution::P768,
        None => return Err(bad("resolution", "resolution is required")),
        Some(s) => Resolution::parse(s).ok_or_else(|| {
            bad("resolution", format!("resolution `{s}` is invalid; expected 480P, 768P or 2K"))
        })?,
    };
    if !model.resolutions().contains(&resolution) {
        let ok: Vec<&str> = model.resolutions().iter().map(|r| r.as_str()).collect();
        return Err(bad(
            "resolution",
            format!("resolution {} is not supported by {model}; supported: {}", resolution.as_str(), ok.join(", ")),
        ));
    }

    let duration = parse_duration(b.duration.as_ref(), model)?;
    let ratio = parse_ratio(b.ratio.as_deref())?;
    let short_edge = resolution.short_edge();
    let canvas = match task {
        Task::T2V => match ratio {
            Some(r) => CanvasSpec::Aspect { ratio: r, short_edge },
            None => {
                return Err(bad(
                    "ratio",
                    "ratio is required for text-to-video and must not be adaptive",
                ))
            }
        },
        // i2va: always adaptive; another valid value is ignored (§1.3).
        Task::I2V | Task::Keyframes => CanvasSpec::FollowImage { short_edge },
        _ => match ratio {
            Some(r) => CanvasSpec::Aspect { ratio: r, short_edge },
            // r2va adaptive follows the first reference image; with no image
            // reference there is nothing to follow, so 16:9 (INFERRED).
            None if content.refs.iter().any(|r| r.kind == MediaKind::Image) => {
                CanvasSpec::FollowImage { short_edge }
            }
            None => CanvasSpec::Aspect { ratio: Ratio::R16_9, short_edge },
        },
    };

    let mut req = GenerationRequest::text(ProtocolId::MiniMaxV2, model.as_str(), content.prompt);
    req.task = task;
    req.canvas = canvas;
    req.timing = TimingSpec {
        length: Length::Seconds { value: duration as f64, snap: Snap::AlignUp },
        fps: None,
    };
    req.keyframes = content
        .first
        .into_iter()
        .map(|image| Keyframe { at: Anchor::First, image })
        .chain(content.last.into_iter().map(|image| Keyframe { at: Anchor::Last, image }))
        .collect();
    req.references = content.refs;

    match &b.extra {
        None | Some(Value::Null) => {}
        Some(_) if !model.accepts_extra() => {
            return Err(bad("extra", format!("extra is not supported by {model}")));
        }
        Some(Value::Object(m)) => {
            for (k, v) in m {
                if k != "prompt_expansion_mode" {
                    return Err(bad("extra", format!("unknown extra field `{k}`")));
                }
                match v.as_str() {
                    Some(s) if PROMPT_EXPANSION_MODES.contains(&s) => {}
                    _ => {
                        return Err(bad(
                            "extra.prompt_expansion_mode",
                            "prompt_expansion_mode must be one of disabled, balanced, quality",
                        ))
                    }
                }
                // We have no prompt expander (design §4.3): accepted, no effect.
                req.note_noop("prompt_expansion_mode");
            }
        }
        Some(_) => return Err(bad("extra", "extra must be an object")),
    }

    if let Some(u) = b.callback_url.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let url = url::Url::parse(u)
            .ok()
            .filter(|u| matches!(u.scheme(), "http" | "https"))
            .ok_or_else(|| bad("callback_url", "callback_url must be an http(s) URL"))?;
        req.callback = Some(CallbackSpec::MiniMax { url });
    }
    Ok(req)
}

fn parse_content(items: &[ContentItem]) -> Result<Content, ApiError> {
    let mut texts: Vec<String> = Vec::new();
    let mut c = Content::default();
    for (i, it) in items.iter().enumerate() {
        let p = |f: &str| format!("content[{i}].{f}");
        let ty = it
            .kind
            .as_deref()
            .ok_or_else(|| bad(p("type"), format!("content[{i}].type is required")))?;
        let (kind, obj) = match ty {
            "text" => {
                if let Some(r) = &it.role {
                    return Err(bad(p("role"), format!("role `{r}` is not valid for a text item")));
                }
                let t = it.text.as_deref().unwrap_or_default();
                if t.trim().is_empty() {
                    return Err(bad(
                        p("text"),
                        "content must include a non-empty text item (prompt is required)",
                    ));
                }
                if t.chars().count() > MAX_PROMPT_CHARS {
                    return Err(bad(p("text"), format!("text exceeds {MAX_PROMPT_CHARS} characters")));
                }
                texts.push(t.to_owned());
                continue;
            }
            "image_url" => (MediaKind::Image, &it.image_url),
            "video_url" => (MediaKind::Video, &it.video_url),
            "audio_url" => (MediaKind::Audio, &it.audio_url),
            other => {
                return Err(bad(
                    p("type"),
                    format!("unknown content type `{other}`; expected text, image_url, video_url or audio_url"),
                ))
            }
        };
        let param = format!("content[{i}].{ty}.url");
        let url = obj
            .as_ref()
            .and_then(|o| o.url.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| bad(param.clone(), format!("{param} is required")))?;
        let media = parse_media(url, kind, &param)?;
        match (kind, it.role.as_deref()) {
            (MediaKind::Image, None | Some("first_frame")) => c.first.push(media),
            (MediaKind::Image, Some("last_frame")) => c.last.push(media),
            (MediaKind::Image, Some("reference_image")) => c.refs.push(Reference { kind, media }),
            (MediaKind::Video, None | Some("reference_video")) => c.refs.push(Reference { kind, media }),
            (MediaKind::Audio, None | Some("reference_audio")) => c.refs.push(Reference { kind, media }),
            (_, Some(r)) => {
                let known = ["first_frame", "last_frame", "reference_image", "reference_video", "reference_audio"];
                let msg = if known.contains(&r) {
                    format!("role `{r}` is not valid for a {ty} item")
                } else {
                    format!("unknown role `{r}`; expected one of {}", known.join(", "))
                };
                return Err(bad(p("role"), msg));
            }
        }
    }
    match texts.len() {
        0 => {
            return Err(bad(
                "content",
                "content must include a non-empty text item (prompt is required)",
            ))
        }
        1 => c.prompt = texts.remove(0),
        n => return Err(bad("content", format!("content must include exactly one text item, got {n}"))),
    }
    if c.first.len() > 1 {
        return Err(bad("content", "at most one first_frame image is allowed"));
    }
    if c.last.len() > 1 {
        return Err(bad("content", "at most one last_frame image is allowed"));
    }
    let has_frames = !c.first.is_empty() || !c.last.is_empty();
    if !c.refs.is_empty() && has_frames {
        return Err(bad(
            "content",
            "first_frame/last_frame images cannot be mixed with reference_image, reference_video or reference_audio items",
        ));
    }
    let n = |k: MediaKind| c.refs.iter().filter(|r| r.kind == k).count();
    let (mi, mv, ma, mt) = MAX_REFS;
    for (got, max, what) in [
        (n(MediaKind::Image), mi, "reference_image items"),
        (n(MediaKind::Video), mv, "reference_video items"),
        (n(MediaKind::Audio), ma, "reference_audio items"),
        (c.refs.len(), mt, "reference items in total"),
    ] {
        if got > max {
            return Err(bad("content", format!("at most {max} {what} are allowed, got {got}")));
        }
    }
    Ok(c)
}

/// `http(s)://`, `data:<type>/<fmt>;base64,…` (type matching the item,
/// lowercase format) or `mm_file://…` (a provider file: 400 gap).
fn parse_media(url: &str, kind: MediaKind, param: &str) -> Result<MediaRef, ApiError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let want = match kind {
            MediaKind::Image => "image/",
            MediaKind::Video => "video/",
            MediaKind::Audio => "audio/",
        };
        let mime = rest.split([';', ',']).next().unwrap_or_default();
        if !mime.starts_with(want) || mime.chars().any(|c| c.is_ascii_uppercase()) {
            return Err(bad(
                param,
                format!("data URI must be data:{want}<format>;base64,… with a lowercase format"),
            ));
        }
        return Ok(MediaRef::DataUri(url.to_owned()));
    }
    if url.starts_with("mm_file://") {
        return Err(ApiError::unsupported(GapId::ProviderFiles).with_param(param));
    }
    match url::Url::parse(url) {
        Ok(u) if matches!(u.scheme(), "http" | "https") => Ok(MediaRef::Http(u)),
        _ => Err(bad(param, format!("{param} must be a public URL, a data URI or mm_file://"))),
    }
}

fn parse_duration(v: Option<&Value>, model: MiniMaxModel) -> Result<u32, ApiError> {
    let v = v.filter(|v| !v.is_null()).ok_or_else(|| bad("duration", "duration is required"))?;
    let n = v
        .as_u64()
        .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0 && *f >= 0.0).map(|f| f as u64))
        .ok_or_else(|| bad("duration", "duration must be an integer number of seconds"))?;
    let (lo, hi) = (model.min_duration() as u64, model.max_duration() as u64);
    if !(lo..=hi).contains(&n) {
        return Err(bad("duration", format!("duration {n} is not supported by {model}; expected {lo} to {hi}")));
    }
    Ok(n as u32)
}

/// `None` = `adaptive` (also when absent).
fn parse_ratio(s: Option<&str>) -> Result<Option<Ratio>, ApiError> {
    match s {
        None | Some("adaptive") => Ok(None),
        Some(r) if RATIOS.contains(&r) => Ok(Some(r.parse()?)),
        Some(r) => Err(bad("ratio", format!("ratio `{r}` is invalid; expected one of {}", RATIOS.join(", ")))),
    }
}

// ---- HTTP -----------------------------------------------------------------------

pub(crate) async fn handle(mm: Arc<MiniMax>, State(ctx): State<ServeCtx>, headers: HeaderMap, body: Body) -> Response {
    let rid = random_token();
    let reply = match create(&mm, &ctx, &headers, body, &rid).await {
        Ok(r) => r,
        Err(e) => error_reply(&*mm, &e, &ErrorCtx { request_id: Some(rid), route: Some("/v2/video_generation".into()), external_id: None }),
    };
    into_response(reply, &ctx, None).await
}

async fn create(mm: &MiniMax, ctx: &ServeCtx, headers: &HeaderMap, body: Body, rid: &str) -> Result<HttpReply, ApiError> {
    let owner = ctx.auth().authenticate(ProtocolId::MiniMaxV2, headers)?;
    let max = mm.config().body_max;
    let bytes = axum::body::to_bytes(body, max).await.map_err(|_| {
        const MIB: usize = 1024 * 1024;
        let cap = if max % MIB == 0 { format!("{} MB", max / MIB) } else { format!("{max} bytes") };
        ApiError::payload_too_large(format!("request body exceeds {cap}"))
    })?;
    if bytes.is_empty() {
        return Err(ApiError::invalid("request body is required"));
    }
    let raw: Value = serde_json::from_slice(&bytes).map_err(|e| ApiError::invalid(format!("invalid JSON body: {e}")))?;
    if !raw.is_object() {
        return Err(ApiError::invalid("request body must be a JSON object"));
    }
    let parsed: CreateBody = serde_json::from_value(raw).map_err(|e| ApiError::invalid(e.to_string()))?;
    let mut ncx = NormalizeCtx::new(ctx.now());
    ncx.owner = owner.clone();
    ncx.request_id = rid.to_owned();
    let mut echo = request_echo(&parsed);
    let defaulted = parsed.resolution.is_none();
    let mut req = CreateEndpoint.normalize(parsed, &ncx)?;
    if defaulted {
        default_resolution(ctx, mm, &mut req, &mut echo);
    }
    if let Length::Seconds { value, .. } = req.timing.length {
        echo["duration"] = json!(value as u64);
    }
    // The ratio actually used: i2va is always adaptive (a sent ratio is
    // ignored), and the query reports the generated ratio for adaptive.
    echo["ratio"] = match &req.canvas {
        CanvasSpec::Aspect { ratio, .. } => json!(ratio.to_string()),
        _ => json!("adaptive"),
    };
    if let Some(cb) = &req.callback {
        fastvideo_serve_kit::net::check_url(cb.url(), &ctx.callbacks().target)
            .map_err(|e| bad("callback_url", format!("callback_url is not allowed: {e}")))?;
    }
    mm.limiter.hit(owner.as_ref(), ctx.now())?;
    let limit = mm.config().max_in_flight;
    if limit > 0 {
        let q = ListQuery {
            owner: owner.clone(),
            protocol: Some(ProtocolId::MiniMaxV2),
            statuses: vec![fastvideo_protocol::JobStatus::Queued, fastvideo_protocol::JobStatus::Running],
            limit: 0,
            ..ListQuery::default()
        };
        if ctx.jobs().list(q).await.total >= limit as usize {
            return Err(ApiError::rate_limited(format!("at most {limit} tasks may be queued or running"))
                .with_retry_after(10));
        }
    }
    let job = submit_generation(ctx, mm, req, owner, echo).await?;
    Ok(CreateEndpoint.submit_reply(&job, &ctx.view_ctx(false)))
}

/// An omitted `resolution` is 768P (the H3-Max contract); on a model
/// without the 768 tier (`MiniMax-H3-Draft` on the 480P draft recipe) it is
/// the model's own default tier instead, so a body without `resolution`
/// runs on every model.
fn default_resolution(ctx: &ServeCtx, mm: &MiniMax, req: &mut GenerationRequest, echo: &mut Value) {
    let models = ctx.engine().models();
    let engine = ctx.engine().clone();
    let alias = move |n: &str| engine.alias(n);
    let Ok((caps, _)) = mm.resolve_caps(&req.model, &alias, &models) else { return };
    let tiers = &caps.canvas.short_edges;
    let p768 = Resolution::P768.short_edge();
    let Some(&first) = tiers.first().filter(|_| !tiers.contains(&p768)) else { return };
    let Some(r) = [Resolution::P480, Resolution::P768].into_iter().find(|r| r.short_edge() == first) else { return };
    match &mut req.canvas {
        CanvasSpec::Aspect { short_edge, .. } | CanvasSpec::FollowImage { short_edge } if *short_edge == p768 => {
            *short_edge = first;
            echo["resolution"] = json!(r.as_str());
        }
        _ => {}
    }
}

/// What the job keeps of the body for the views: the scalar fields as sent
/// (never the `content` media, which may be large data URIs).
fn request_echo(b: &CreateBody) -> Value {
    let prompt = b
        .content
        .iter()
        .flatten()
        .find(|c| c.kind.as_deref() == Some("text"))
        .and_then(|c| c.text.clone());
    json!({
        "model": b.model,
        "prompt": prompt,
        "resolution": b.resolution.clone().unwrap_or_else(|| Resolution::P768.as_str().to_owned()),
        "duration": b.duration,
        "ratio": b.ratio.clone().unwrap_or_else(|| "adaptive".to_owned()),
    })
}

/// The MiniMax `duration: 4` path (design §0: 4 s is allowed for
/// MiniMax-H3). `fastvideo-protocol` still answers `Unsupported(H3FourSeconds)`
/// for 107 frames on every API, while FastVideo parity (and only that) needs
/// the 5 s floor. Until the protocol makes that refusal per-API, negotiate
/// with the model's default length and then pin the frame count to the
/// model grid's value for 4 s, checked against `H3Geometry`.
fn four_second_frames(req: &GenerationRequest, caps: &ModelCaps) -> Result<Option<u32>, ApiError> {
    if caps.family != Family::H3 {
        return Ok(None);
    }
    match req.timing.length {
        Length::Seconds { value: 4.0, .. } => {}
        _ => return Ok(None),
    }
    let fps = req.timing.fps.unwrap_or(caps.fps.default);
    caps.frames
        .align_up(4 * fps)
        .map(Some)
        .ok_or_else(|| ApiError::unsupported(GapId::H3FourSeconds).with_param("duration"))
}

/// Usage inputs recorded at create time (views are pure and cannot probe).
fn input_usage(staged: &StagedInputs) -> Value {
    let secs = |k: MediaKind| -> f64 {
        staged
            .references
            .iter()
            .filter(|(kind, _)| *kind == k)
            .map(|(_, m)| m.probe.duration_s.unwrap_or(0.0))
            .sum()
    };
    let images = staged.keyframes.len()
        + staged.references.iter().filter(|(k, _)| *k == MediaKind::Image).count();
    let has_audio = staged.references.iter().any(|(k, _)| *k == MediaKind::Audio);
    json!({
        "input_image_count": images,
        "input_video_seconds": secs(MediaKind::Video),
        "input_audio_seconds": if has_audio { json!(secs(MediaKind::Audio)) } else { Value::Null },
    })
}

/// The create pipeline (serve-kit's `submit_request` with MiniMax model
/// resolution and the 4 s path): admission, safety, model, precheck,
/// ingestion, `negotiate`, store insert, engine submit, first callback.
pub async fn submit_generation(
    ctx: &ServeCtx,
    mm: &MiniMax,
    req: GenerationRequest,
    owner: Option<KeyId>,
    mut echo: Value,
) -> Result<Job, ApiError> {
    ctx.engine().admit()?;
    ctx.safety().check_request(&req)?;
    let models = ctx.engine().models();
    let engine = ctx.engine().clone();
    let alias = move |n: &str| engine.alias(n);
    let (caps, via_tier) = mm.resolve_caps(&req.model, &alias, &models)?;
    let caps = fastvideo_protocol::route_task(caps, req.task, &models);
    let four = four_second_frames(&req, caps)?;
    let mut plan = req.clone();
    if four.is_some() {
        plan.timing.length = Length::ModelDefault;
    }
    precheck(&plan, caps)?;
    let id = JobId::new();
    let dir = ctx.inputs_dir(id);
    let staged = async {
        let staged = ctx.ingestor().stage(&plan, &mm.config().ingest, &dir, ctx.now()).await?;
        let (mut resolved, notes) = negotiate_noted(&plan, caps, &staged)?;
        if let Some(n) = four {
            H3Geometry::new(resolved.height as usize, resolved.width as usize, n as usize)
                .map_err(|e| bad("duration", format!("H3 geometry: {e}")))?;
            resolved.num_frames = n;
        }
        // Caps without a tier tag (e.g. the engine table drops the draft
        // tag) still report the tier the name resolved through.
        if resolved.tier.is_none() {
            resolved.tier = via_tier;
        }
        Ok::<_, ApiError>((staged, resolved, notes))
    }
    .await;
    let (staged, resolved, notes) = match staged {
        Ok(r) => r,
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return Err(e);
        }
    };
    echo["_fv"] = input_usage(&staged);
    let now = ctx.now();
    let mut job = Job::new(id, ProtocolId::MiniMaxV2, mm.new_external_id(id), resolved, now, ctx.config().retention(ProtocolId::MiniMaxV2));
    job.owner = owner;
    job.request_echo = echo;
    job.callback = req.callback.clone();
    // docs/serve/tracing.md: the request's trace (set by the HTTP layer).
    let trace = fastvideo_trace::current();
    job.trace = trace.map(|t| t.traceparent(t.parent));
    for n in notes {
        tracing::info!(job = %id, model = %caps.id, "{n}");
        job.logs.push(fastvideo_protocol::LogLine::info(n, now));
    }
    if !req.accepted_noop.is_empty() {
        tracing::debug!(job = %id, fields = ?req.accepted_noop, "accepted no-op fields");
    }
    let t_insert = trace.map(|_| fastvideo_trace::now_ns());
    if let Err(e) = ctx.jobs().insert(job.clone()).await {
        let _ = tokio::fs::remove_dir_all(&dir).await;
        return Err(e.into());
    }
    if let (Some(t), Some(s)) = (trace, t_insert) {
        t.span_since(fastvideo_trace::Comp::Store, "insert", s, 0);
    }
    // The `queued` callback goes out before the engine can move the job on,
    // so the receiver always sees queued -> running -> terminal in order.
    ctx.notify(&job);
    let t_submit = trace.map(|_| fastvideo_trace::now_ns());
    let submitted = ctx.engine().submit(&job).await;
    if let (Some(t), Some(s)) = (trace, t_submit) {
        t.span_since(fastvideo_trace::Comp::Queue, "submit", s, 0);
    }
    if let Err(e) = submitted {
        // The receiver already saw `queued`: close the sequence with `failed`.
        let (now, err) = (ctx.now(), e.clone());
        if let Ok(j) = ctx.jobs().update(id, Box::new(move |j| {
            let _ = j.mark_failed(now, err);
        })).await {
            ctx.notify(&j);
        }
        ctx.jobs().remove(id).await;
        return Err(e);
    }
    Ok(ctx.jobs().get(id).await.unwrap_or(job))
}

/// `POST /v2/h3_context_ir`, `POST /v2/video_regeneration`: authenticated,
/// then 400 (design §1.2: they need MiniMax platform components).
pub(crate) async fn unsupported(mm: Arc<MiniMax>, State(ctx): State<ServeCtx>, headers: HeaderMap, route: &'static str) -> Response {
    let rid = random_token();
    let err = match ctx.auth().authenticate(ProtocolId::MiniMaxV2, &headers) {
        Err(e) => e,
        Ok(_) => ApiError::invalid(format!("{route} is not supported by this server")),
    };
    let reply = error_reply(&*mm, &err, &ErrorCtx { request_id: Some(rid), route: Some(route.into()), external_id: None });
    into_response(reply, &ctx, None).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(v: Value) -> CreateBody {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn t2v_basics() {
        let r = normalize(body(json!({"model":"MiniMax-H3","content":[{"type":"text","text":"a cat"}],
            "resolution":"768P","duration":5,"ratio":"16:9"})))
        .unwrap();
        assert_eq!(r.task, Task::T2V);
        assert_eq!(r.prompt, "a cat");
        assert_eq!(r.canvas, CanvasSpec::Aspect { ratio: Ratio::R16_9, short_edge: 768 });
        assert_eq!(r.timing.length, Length::Seconds { value: 5.0, snap: Snap::AlignUp });
    }

    #[test]
    fn four_second_plan() {
        let caps = ModelCaps::h3("h", false);
        let mut r = GenerationRequest::text(ProtocolId::MiniMaxV2, "MiniMax-H3", "x");
        r.timing.length = Length::Seconds { value: 4.0, snap: Snap::AlignUp };
        assert_eq!(four_second_frames(&r, &caps).unwrap(), Some(107));
        r.timing.length = Length::Seconds { value: 5.0, snap: Snap::AlignUp };
        assert_eq!(four_second_frames(&r, &caps).unwrap(), None);
        // A model grid that starts at 5 s keeps the gap.
        let mut c5 = caps.clone();
        c5.frames.min = 124;
        r.timing.length = Length::Seconds { value: 4.0, snap: Snap::AlignUp };
        assert_eq!(four_second_frames(&r, &c5).unwrap_err().gap(), Some(GapId::H3FourSeconds));
    }
}
