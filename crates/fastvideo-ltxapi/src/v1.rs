//! `/v1/*` sync routes (design §4.5, ltx §2.2).
//!
//! `POST /v1/{text,image}-to-video` take the v2 request body, hold the
//! connection until the job ends and answer `200` with the MP4 bytes
//! (`Content-Type: video/mp4`, ltx §1.6). A generation longer than the sync
//! timeout is cancelled and answers `504`. Failures come back on the same
//! request in the LTX error envelope. A per-owner concurrency limit (upstream
//! default 2, ltx §1.4) answers `429 concurrency_limit_error` with
//! `Retry-After`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fastvideo_protocol::{
    ApiError, ArtifactLocation, ErrorCtx, HttpReply, Job, JobState, JobView, ViewCtx,
};

use crate::error::{self, Api, DEFAULT_RETRY_AFTER_S};
use crate::v2::with_meta;

/// The MIME type of v1 success bodies.
pub const VIDEO_MP4: &str = "video/mp4";

/// Renders a finished v1 job: the video file, or the job's error.
#[derive(Clone, Copy, Debug, Default)]
pub struct V1View;

impl V1View {
    fn error(err: &ApiError) -> HttpReply {
        error::render(err, Api::V1, &ErrorCtx::default())
    }
}

impl JobView for V1View {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        self.result_reply(job, cx)
    }

    fn result_reply(&self, job: &Job, _cx: &ViewCtx) -> HttpReply {
        match &job.state {
            JobState::Succeeded => match job.artifacts.first() {
                Some(a) => match &a.location {
                    ArtifactLocation::Local(path) => {
                        with_meta(HttpReply::file(200, path.clone(), VIDEO_MP4), job)
                    }
                    ArtifactLocation::Object { .. } => Self::error(&ApiError::internal(
                        "sync generation needs a local artifact store; use the /v2 endpoints",
                    )),
                },
                None => Self::error(&ApiError::internal("the job produced no output")),
            },
            JobState::Failed(e) => Self::error(e),
            // `Cancelled` renders as `500 api_error`.
            JobState::Cancelled => Self::error(&ApiError::cancelled("the job was cancelled")),
            JobState::Queued | JobState::Running => {
                Self::error(&ApiError::timeout("generation did not finish in time"))
            }
        }
    }
}

/// Per-owner limit on concurrent `/v1/*` generations.
#[derive(Debug)]
pub struct ConcurrencyLimit {
    max: usize,
    active: Mutex<HashMap<String, usize>>,
}

/// Releases one slot on drop.
#[derive(Debug)]
pub struct Slot {
    limit: Arc<ConcurrencyLimit>,
    key: String,
}

impl ConcurrencyLimit {
    pub fn new(max: usize) -> Arc<Self> {
        Arc::new(Self {
            max: max.max(1),
            active: Mutex::new(HashMap::new()),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, usize>> {
        self.active.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Takes a slot for `owner` (`""` for anonymous callers), or
    /// `RateLimited` (rendered as `429 concurrency_limit_error` on v1).
    pub fn acquire(self: &Arc<Self>, owner: &str) -> Result<Slot, ApiError> {
        let mut g = self.lock();
        let n = g.entry(owner.to_owned()).or_insert(0);
        if *n >= self.max {
            return Err(ApiError::rate_limited(
                "Too many concurrent requests. Please try again later.",
            )
            .with_retry_after(DEFAULT_RETRY_AFTER_S));
        }
        *n += 1;
        Ok(Slot {
            limit: self.clone(),
            key: owner.to_owned(),
        })
    }

    /// Slots in use by `owner`.
    pub fn active(&self, owner: &str) -> usize {
        self.lock().get(owner).copied().unwrap_or(0)
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let mut g = self.limit.lock();
        if let Some(n) = g.get_mut(&self.key) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                g.remove(&self.key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_per_owner() {
        let l = ConcurrencyLimit::new(2);
        let a = l.acquire("k1").unwrap();
        let _b = l.acquire("k1").unwrap();
        let e = l.acquire("k1").unwrap_err();
        assert_eq!(e.retry_after_s, Some(DEFAULT_RETRY_AFTER_S));
        let r = error::render(&e, Api::V1, &ErrorCtx::default());
        assert_eq!(r.status, 429);
        assert_eq!(r.json_body().unwrap()["error"]["type"], "concurrency_limit_error");
        assert!(l.acquire("k2").is_ok());
        drop(a);
        assert_eq!(l.active("k1"), 1);
        assert!(l.acquire("k1").is_ok());
    }
}
