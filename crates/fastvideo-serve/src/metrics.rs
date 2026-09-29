//! Prometheus metrics (design §6, WP-10).
//!
//! One process-wide recorder (`metrics-exporter-prometheus`, rendered by
//! `GET /metrics`). Series:
//!
//! - `fv_http_requests_total{method,route,status}` and
//!   `fv_http_request_duration_seconds{method,route}` (route = the matched
//!   template, never the raw path, so ids do not explode cardinality);
//! - `fv_jobs_submitted_total{api}`, `fv_jobs_finished_total{api,status}`,
//!   `fv_job_duration_seconds{api}` (the engine gate);
//! - `fv_engine_queued{priority}`, `fv_engine_running`, `fv_engine_sessions`,
//!   `fv_ready` (refreshed on scrape).
//! - `fv_encoder_restart_duration_seconds{codec,spare}` and
//!   `fv_encoder_restarts_total{codec,spare}`: keyframe restarts of the
//!   ffmpeg pipe encoders (director, Reactor), from the restart to the new
//!   process's first frame; `spare` is `warm`, `warming` or `cold`
//!   (`fastvideo_media::pipe`).

use std::sync::OnceLock;
use std::time::Instant;

use axum::extract::{MatchedPath, Request};
use axum::middleware::Next;
use axum::response::Response;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

static HANDLE: OnceLock<Option<PrometheusHandle>> = OnceLock::new();

/// Installs the global recorder once; later calls return the same handle.
/// `None` when another recorder was installed first.
pub fn install() -> Option<PrometheusHandle> {
    HANDLE
        .get_or_init(|| {
            let b = PrometheusBuilder::new()
                .set_buckets_for_metric(
                    metrics_exporter_prometheus::Matcher::Suffix("duration_seconds".into()),
                    &[0.005, 0.025, 0.1, 0.5, 1.0, 5.0, 15.0, 60.0, 180.0, 600.0],
                )
                .ok()?;
            match b.install_recorder() {
                Ok(h) => Some(h),
                Err(e) => {
                    tracing::warn!(error = %e, "metrics recorder not installed");
                    None
                }
            }
        })
        .clone()
}

/// Axum middleware counting requests by matched route.
pub async fn track(req: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = req.method().as_str().to_owned();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".into());
    let resp = next.run(req).await;
    let status = resp.status().as_u16().to_string();
    metrics::counter!("fv_http_requests_total", "method" => method.clone(), "route" => route.clone(), "status" => status)
        .increment(1);
    metrics::histogram!("fv_http_request_duration_seconds", "method" => method, "route" => route)
        .record(start.elapsed().as_secs_f64());
    resp
}
