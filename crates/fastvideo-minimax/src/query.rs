//! Query and list routes (design §4.3, research §1.4).
//!
//! `GET /v2/query/video_generation/{task_id}` → `{"task": VideoTask}`:
//!
//! | Field | Value |
//! |---|---|
//! | `id` | the 18-digit task id |
//! | `model` | the model name as sent |
//! | `status` | `queued` / `running` / `succeeded` / `failed` / `cancelled` |
//! | `error` | only when failed: `{code: "1000", message}` (`1026` filtered, `1001` timeout) |
//! | `created_at`, `updated_at` | unix seconds (`updated_at`: last state change) |
//! | `content` | `{url}` once succeeded, re-signed on every query (TTL `minimax.url_ttl`); `{}` before |
//! | `resolution`, `duration` | as requested |
//! | `usage` | only on success: `total_seconds`, `input_seconds`, `output_seconds`, `input_image_count`, `input_audio_seconds` (only with audio references). Token fields are omitted (design Q7) |
//! | `ratio` | as requested, or for `adaptive` the ratio nearest the generated canvas |
//! | `task_type`, `modality` | `generation`, `video` |
//! | `metadata` | **native addition** (design §0.3, §0.6): `{tier, recipe, quality}` when the model is tiered; `quality` is `"draft"` for draft-tier results |
//!
//! `GET /v2/query/video_generation?page_num&page_size&filter.status&filter.task_ids&filter.model&filter.task_type`
//! → `{"items": [VideoTask], "total"}`, newest first, the caller's tasks only.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use fastvideo_protocol::{
    ApiError, ErrorCtx, HttpReply, Job, JobState, JobStatus, JobView, ListQuery, ProtocolId,
    SortOrder, Tier, ViewCtx,
};
use fastvideo_serve_kit::handlers::{error_reply, find_job, into_response};
use fastvideo_serve_kit::{random_token, CallbackRender, ServeCtx};
use serde_json::{json, Map, Value};

use crate::error::task_error_code;
use crate::MiniMax;

/// Renders jobs as MiniMax `VideoTask`s (query, list and callback bodies).
#[derive(Clone, Copy, Debug)]
pub struct TaskView {
    /// `content.url` lifetime.
    pub url_ttl: std::time::Duration,
}

impl JobView for TaskView {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, json!({ "task": task_json(job, cx, self.url_ttl) }))
    }
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        self.status_reply(job, cx)
    }
}

impl CallbackRender for TaskView {
    /// The callback body has the query's structure, `{"task": VideoTask}`
    /// (research §1.6, INFERRED).
    fn callback_body(&self, job: &Job, cx: &ViewCtx) -> Option<Value> {
        Some(json!({ "task": task_json(job, cx, self.url_ttl) }))
    }
}

/// The MiniMax status name.
pub fn status_str(s: JobStatus) -> &'static str {
    s.as_str()
}

/// The allowed `ratio` value nearest to `width:height` (log distance).
pub fn nearest_ratio(width: u32, height: u32) -> &'static str {
    if width == 0 || height == 0 {
        return "";
    }
    let r = (width as f64 / height as f64).ln();
    [("21:9", 21.0f64 / 9.0), ("16:9", 16.0 / 9.0), ("4:3", 4.0 / 3.0), ("1:1", 1.0), ("3:4", 0.75), ("9:16", 9.0 / 16.0)]
        .into_iter()
        .min_by(|a, b| (a.1.ln() - r).abs().total_cmp(&(b.1.ln() - r).abs()))
        .map(|x| x.0)
        .unwrap_or("")
}

fn unix(t: time::OffsetDateTime) -> i64 {
    t.unix_timestamp()
}

fn secs(v: &Value) -> u64 {
    v.as_f64().map(|f| f.max(0.0).ceil() as u64).unwrap_or(0)
}

/// One `VideoTask`.
pub fn task_json(job: &Job, cx: &ViewCtx, url_ttl: std::time::Duration) -> Value {
    let echo = &job.request_echo;
    let mut t = Map::new();
    t.insert("id".into(), json!(job.external_id));
    t.insert("model".into(), json!(job.requested_model()));
    t.insert("status".into(), json!(status_str(job.status())));
    if let JobState::Failed(e) = &job.state {
        t.insert("error".into(), json!({ "code": task_error_code(e.kind), "message": e.message }));
    }
    let updated = job.completed_at.or(job.started_at).unwrap_or(job.created_at);
    t.insert("created_at".into(), json!(unix(job.created_at)));
    t.insert("updated_at".into(), json!(unix(updated)));
    let mut content = Map::new();
    if job.status() == JobStatus::Succeeded {
        if let Some(a) = job.artifacts.first() {
            content.insert("url".into(), json!(cx.urls.url_for(a, url_ttl).as_str()));
        }
    }
    t.insert("content".into(), Value::Object(content));
    t.insert("resolution".into(), echo.get("resolution").cloned().unwrap_or(json!("768P")));
    let duration = echo.get("duration").map(secs).unwrap_or_else(|| job.resolved.duration_s().round() as u64);
    t.insert("duration".into(), json!(duration));
    if job.status() == JobStatus::Succeeded {
        let fv = echo.get("_fv").cloned().unwrap_or_default();
        let input = fv.get("input_video_seconds").map(secs).unwrap_or(0);
        let mut u = Map::new();
        u.insert("total_seconds".into(), json!(input + duration));
        u.insert("input_seconds".into(), json!(input));
        u.insert("output_seconds".into(), json!(duration));
        u.insert("input_image_count".into(), json!(fv.get("input_image_count").and_then(Value::as_u64).unwrap_or(0)));
        if let Some(a) = fv.get("input_audio_seconds").filter(|v| !v.is_null()) {
            u.insert("input_audio_seconds".into(), json!(secs(a)));
        }
        t.insert("usage".into(), Value::Object(u));
    }
    let ratio = match echo.get("ratio").and_then(Value::as_str) {
        Some(r) if r != "adaptive" => r.to_owned(),
        _ => {
            let (w, h) = job.resolved.output_size();
            nearest_ratio(w, h).to_owned()
        }
    };
    t.insert("ratio".into(), json!(ratio));
    t.insert("task_type".into(), json!("generation"));
    t.insert("modality".into(), json!("video"));
    if job.resolved.tier.is_some() || job.resolved.recipe.is_some() {
        let quality = match job.resolved.tier {
            Some(Tier::Draft) => "draft",
            _ => "standard",
        };
        t.insert(
            "metadata".into(),
            json!({ "tier": job.resolved.tier, "recipe": job.resolved.recipe, "quality": quality }),
        );
    }
    Value::Object(t)
}

pub(crate) async fn query_handle(mm: Arc<MiniMax>, State(ctx): State<ServeCtx>, Path(task_id): Path<String>, headers: HeaderMap) -> Response {
    let ecx = ErrorCtx { request_id: Some(random_token()), route: Some("/v2/query/video_generation/{task_id}".into()), external_id: Some(task_id.clone()) };
    let reply = async {
        let owner = ctx.auth().authenticate(ProtocolId::MiniMaxV2, &headers)?;
        let job = find_job(&ctx, &*mm, &task_id, owner.as_ref()).await?;
        Ok::<_, ApiError>(mm.view().status_reply(&job, &ctx.view_ctx(false)))
    }
    .await
    .unwrap_or_else(|e| error_reply(&*mm, &e, &ecx));
    into_response(reply, &ctx, None).await
}

/// Default and maximum `page_size`.
pub const PAGE_SIZE: (usize, usize) = (20, 100);

/// Parses the list query string into a [`ListQuery`] for `owner`, or
/// `Ok(None)` when the filter can match nothing (`filter.task_type` other
/// than `generation`).
pub fn parse_list_query(params: &[(String, String)]) -> Result<Option<ListQuery>, ApiError> {
    let mut q = ListQuery {
        protocol: Some(ProtocolId::MiniMaxV2),
        order: SortOrder::Desc,
        limit: PAGE_SIZE.0,
        ..ListQuery::default()
    };
    let mut page = 1usize;
    let num = |k: &str, v: &str| {
        v.trim()
            .parse::<usize>()
            .ok()
            .filter(|n| *n >= 1)
            .ok_or_else(|| ApiError::invalid_param(k, format!("{k} must be a positive integer")))
    };
    let mut empty = false;
    for (k, v) in params {
        match k.as_str() {
            "page_num" => page = num(k, v)?,
            "page_size" => q.limit = num(k, v)?.min(PAGE_SIZE.1),
            "filter.status" | "filter[status]" => {
                for s in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    q.statuses.push(match s {
                        "queued" => JobStatus::Queued,
                        "running" => JobStatus::Running,
                        "succeeded" => JobStatus::Succeeded,
                        "failed" => JobStatus::Failed,
                        "cancelled" => JobStatus::Cancelled,
                        other => {
                            return Err(ApiError::invalid_param(
                                "filter.status",
                                format!("unknown status `{other}`"),
                            ))
                        }
                    });
                }
            }
            "filter.task_ids" | "filter.task_ids[]" | "filter[task_ids]" => q
                .external_ids
                .extend(v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned)),
            "filter.model" | "filter[model]" => {
                if !v.trim().is_empty() {
                    q.model = Some(v.trim().to_owned());
                }
            }
            "filter.task_type" | "filter[task_type]" => match v.trim() {
                "" | "generation" => {}
                "h3_context_ir" | "regeneration" => empty = true,
                other => {
                    return Err(ApiError::invalid_param(
                        "filter.task_type",
                        format!("unknown task_type `{other}`"),
                    ))
                }
            },
            _ => {}
        }
    }
    q.offset = (page - 1).saturating_mul(q.limit);
    Ok((!empty).then_some(q))
}

pub(crate) async fn list_handle(mm: Arc<MiniMax>, State(ctx): State<ServeCtx>, Query(params): Query<Vec<(String, String)>>, headers: HeaderMap) -> Response {
    let ecx = ErrorCtx { request_id: Some(random_token()), route: Some("/v2/query/video_generation".into()), external_id: None };
    let reply = async {
        let owner = ctx.auth().authenticate(ProtocolId::MiniMaxV2, &headers)?;
        let Some(mut q) = parse_list_query(&params)? else {
            return Ok(HttpReply::json(200, json!({ "items": [], "total": 0 })));
        };
        q.owner = owner;
        let page = ctx.jobs().list(q).await;
        let cx = ctx.view_ctx(false);
        let ttl = mm.config().url_ttl;
        let items: Vec<Value> = page.items.iter().map(|j| task_json(j, &cx, ttl)).collect();
        Ok::<_, ApiError>(HttpReply::json(200, json!({ "items": items, "total": page.total })))
    }
    .await
    .unwrap_or_else(|e| error_reply(&*mm, &e, &ecx));
    into_response(reply, &ctx, None).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios() {
        assert_eq!(nearest_ratio(1344, 768), "16:9");
        assert_eq!(nearest_ratio(832, 480), "16:9");
        assert_eq!(nearest_ratio(768, 1344), "9:16");
        assert_eq!(nearest_ratio(768, 768), "1:1");
        assert_eq!(nearest_ratio(1024, 768), "4:3");
        assert_eq!(nearest_ratio(1504, 640), "21:9");
    }

    #[test]
    fn list_query() {
        let p = |v: &[(&str, &str)]| -> Vec<(String, String)> {
            v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
        };
        let q = parse_list_query(&p(&[("page_num", "3"), ("page_size", "5"), ("filter.status", "succeeded"),
            ("filter.task_ids", "1"), ("filter.task_ids", "2,3"), ("filter.model", "MiniMax-H3")]))
        .unwrap()
        .unwrap();
        assert_eq!((q.offset, q.limit), (10, 5));
        assert_eq!(q.statuses, [JobStatus::Succeeded]);
        assert_eq!(q.external_ids, ["1", "2", "3"]);
        assert_eq!(q.model.as_deref(), Some("MiniMax-H3"));
        assert_eq!(parse_list_query(&p(&[("page_size", "1000")])).unwrap().unwrap().limit, 100);
        assert!(parse_list_query(&p(&[("filter.task_type", "regeneration")])).unwrap().is_none());
        assert!(parse_list_query(&p(&[("page_num", "0")])).is_err());
        assert!(parse_list_query(&p(&[("filter.status", "done")])).is_err());
    }
}
