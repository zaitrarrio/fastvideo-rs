//! `/v2/{text,image}-to-video` submit and status (design §4.5, ltx §2.1).
//!
//! - Submit: `202 {id, created_at}` (`V2JobCreatedResponse`).
//! - `GET /v2/{endpoint}/{id}`: `V2JobStatusResponse`, a `oneOf` on
//!   `status`: `pending` / `processing` `{status,id,created_at}`;
//!   `completed` adds `completed_at` and `result:{video_url}`; `failed` adds
//!   `completed_at` and `error:{type,message}`. A job looked up under another
//!   endpoint segment is `404 not_found_error`.
//!
//! State mapping: `Queued` → `pending`, `Running` → `processing`,
//! `Succeeded` → `completed`, `Failed` / `Cancelled` → `failed` (cancelled
//! jobs report `api_error`).
//!
//! The resolved tier and recipe ride in `x-fv-tier` / `x-fv-recipe` response
//! headers (the JSON schema has no room for them); draft-tier results also
//! carry `x-fv-quality: draft` (design §0.3, §0.6).

use std::time::Duration;

use fastvideo_protocol::{
    ApiError, ErrorCtx, HttpReply, Job, JobState, JobView, Tier, ViewCtx,
};
use serde_json::{json, Value};
use time::format_description::FormatItem;
use time::macros::format_description;
use time::{OffsetDateTime, UtcOffset};

use crate::error::{self, Api, LtxErrorType};
use crate::request::Endpoint;

const TS: &[FormatItem<'static>] =
    format_description!("[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z");

/// ISO 8601 UTC with milliseconds, e.g. `2026-09-06T12:00:00.000Z` (ltx §2.1).
pub fn fmt_ts(t: OffsetDateTime) -> String {
    t.to_offset(UtcOffset::UTC)
        .format(TS)
        .unwrap_or_else(|_| t.unix_timestamp().to_string())
}

/// `202 {id, created_at}`.
pub fn created_reply(job: &Job) -> HttpReply {
    let r = HttpReply::json(
        202,
        json!({ "id": job.external_id, "created_at": fmt_ts(job.created_at) }),
    );
    with_meta(r, job)
}

/// The wire `status` of a job.
pub fn wire_status(state: &JobState) -> &'static str {
    match state {
        JobState::Queued => "pending",
        JobState::Running => "processing",
        JobState::Succeeded => "completed",
        JobState::Failed(_) | JobState::Cancelled => "failed",
    }
}

/// The `{type, message}` of a failed (or cancelled) job.
pub fn job_error(job: &Job) -> Option<Value> {
    match &job.state {
        JobState::Failed(e) => Some(error::error_object(
            LtxErrorType::of(e.kind, Api::V2),
            &e.message,
        )),
        JobState::Cancelled => Some(error::error_object(
            LtxErrorType::Api,
            "the job was cancelled",
        )),
        _ => None,
    }
}

/// Adds the tier/recipe metadata headers.
pub fn with_meta(mut r: HttpReply, job: &Job) -> HttpReply {
    if let Some(t) = job.resolved.tier {
        r.push_header("x-fv-tier", t.as_str());
        if t == Tier::Draft {
            r.push_header("x-fv-quality", "draft");
        }
    }
    if let Some(rec) = &job.resolved.recipe {
        r.push_header("x-fv-recipe", rec.clone());
    }
    r
}

/// The `V2JobStatusResponse` body.
pub fn status_body(job: &Job, cx: &ViewCtx, url_ttl: Duration) -> Value {
    let mut v = json!({
        "status": wire_status(&job.state),
        "id": job.external_id,
        "created_at": fmt_ts(job.created_at),
    });
    let completed_at = job.completed_at.unwrap_or(cx.now);
    match &job.state {
        JobState::Succeeded => match job.artifacts.first() {
            Some(a) => {
                v["completed_at"] = json!(fmt_ts(completed_at));
                v["result"] = json!({ "video_url": cx.urls.url_for(a, url_ttl).as_str() });
            }
            None => {
                // Never expected: a success without an output file.
                v["status"] = json!("failed");
                v["completed_at"] = json!(fmt_ts(completed_at));
                v["error"] = error::error_object(LtxErrorType::Api, "the job produced no output");
            }
        },
        JobState::Failed(_) | JobState::Cancelled => {
            v["completed_at"] = json!(fmt_ts(completed_at));
            v["error"] = job_error(job).unwrap_or(Value::Null);
        }
        JobState::Queued | JobState::Running => {}
    }
    v
}

/// Status view for one v2 endpoint.
#[derive(Clone, Copy, Debug)]
pub struct V2View {
    pub endpoint: Endpoint,
    /// Lifetime of `result.video_url`.
    pub url_ttl: Duration,
}

impl V2View {
    pub fn new(endpoint: Endpoint, url_ttl: Duration) -> Self {
        Self { endpoint, url_ttl }
    }
}

/// `404 not_found_error` for a job id.
pub fn not_found(id: &str) -> ApiError {
    ApiError::not_found(format!("job `{id}` was not found"))
}

impl JobView for V2View {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        if Endpoint::of_task(job.task()) != Some(self.endpoint) {
            return error::render(&not_found(&job.external_id), Api::V2, &ErrorCtx::default());
        }
        with_meta(HttpReply::json(200, status_body(job, cx, self.url_ttl)), job)
    }
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        self.status_reply(job, cx)
    }
}
