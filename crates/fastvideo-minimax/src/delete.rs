//! `DELETE /v2/video_generation/{task_id}` (design §4.3, research §1.4).
//!
//! | Task status | Effect | Reply |
//! |---|---|---|
//! | `queued` | cancelled (callback fires) | 200 `{"task_id","action":"cancelled","status":"cancelled"}` |
//! | `succeeded`, `failed` | record and output deleted | 200 `{"task_id","action":"deleted","status":"deleted"}` |
//! | `running`, `cancelled` | none | 400 `bad_request_error (2013)` |
//! | unknown / another key's | none | 404 `invalid task_id (2013)` |

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::Response;
use fastvideo_protocol::{ApiError, ErrorCtx, HttpReply, Job, JobStatus, ProtocolId};
use fastvideo_serve_kit::events::cancel_job;
use fastvideo_serve_kit::handlers::{error_reply, find_job, into_response};
use fastvideo_serve_kit::{random_token, ServeCtx};
use serde_json::json;

use crate::MiniMax;

/// The delete reply body.
pub fn delete_body(task_id: &str, action: &str) -> serde_json::Value {
    json!({ "task_id": task_id, "action": action, "status": action })
}

/// Applies the table in the module docs to `job`.
pub async fn delete_task(ctx: &ServeCtx, job: Job) -> Result<HttpReply, ApiError> {
    let ext = job.external_id.clone();
    match job.status() {
        JobStatus::Queued => {
            // A job that started in between is still cancelled: its token is
            // tripped and the engine reports `Cancelled` at the next step.
            cancel_job(ctx, job.id).await?;
            Ok(HttpReply::json(200, delete_body(&ext, "cancelled")))
        }
        JobStatus::Succeeded | JobStatus::Failed => {
            ctx.jobs().remove(job.id).await;
            Ok(HttpReply::json(200, delete_body(&ext, "deleted")))
        }
        s @ (JobStatus::Running | JobStatus::Cancelled) => Err(ApiError::conflict(format!(
            "task {ext} is {} and cannot be cancelled or deleted",
            s.as_str()
        ))
        .with_param("task_id")),
    }
}

pub(crate) async fn handle(mm: Arc<MiniMax>, State(ctx): State<ServeCtx>, Path(task_id): Path<String>, headers: HeaderMap) -> Response {
    let ecx = ErrorCtx { request_id: Some(random_token()), route: Some("/v2/video_generation/{task_id}".into()), external_id: Some(task_id.clone()) };
    let reply = async {
        let owner = ctx.auth().authenticate(ProtocolId::MiniMaxV2, &headers)?;
        let job = find_job(&ctx, &*mm, &task_id, owner.as_ref()).await?;
        delete_task(&ctx, job).await
    }
    .await
    .unwrap_or_else(|e| error_reply(&*mm, &e, &ecx));
    into_response(reply, &ctx, None).await
}
