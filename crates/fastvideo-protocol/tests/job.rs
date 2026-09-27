//! Job state machine, progress, recovery, expiry and listing.

mod common;

use std::time::Duration;

use common::*;
use fastvideo_protocol::*;
use time::macros::datetime;
use time::OffsetDateTime;

const T0: OffsetDateTime = datetime!(2026-09-27 12:00:00 UTC);

fn job_at(protocol: ProtocolId, ext: &str, created: OffsetDateTime) -> Job {
    let mut r = t2v("fasth3", "x");
    r.seed = Some(1);
    let resolved = nego(&r, &h3()).unwrap();
    Job::new(
        JobId::new(),
        protocol,
        ext,
        resolved,
        created,
        protocol.default_retention(),
    )
}

fn secs(n: i64) -> time::Duration {
    time::Duration::seconds(n)
}

#[test]
fn transition_table_is_exhaustive() {
    use JobStatus::*;
    let all = [Queued, Running, Succeeded, Failed, Cancelled];
    let allowed = [
        (Queued, Running),
        (Queued, Failed),
        (Queued, Cancelled),
        (Running, Succeeded),
        (Running, Failed),
        (Running, Cancelled),
    ];
    for a in all {
        for b in all {
            assert_eq!(
                a.can_transition_to(b),
                allowed.contains(&(a, b)),
                "{a:?} -> {b:?}"
            );
        }
        assert_eq!(a.is_terminal(), matches!(a, Succeeded | Failed | Cancelled));
    }
    // JobState delegates to JobStatus.
    let failed = JobState::Failed(ApiError::internal("x"));
    assert!(JobState::Running.can_transition_to(&failed));
    assert!(!failed.can_transition_to(&JobState::Running));
    assert_eq!(failed.error().map(|e| e.kind), Some(ErrorKind::Internal));
    assert_eq!(JobState::Queued.error(), None);
}

#[test]
fn happy_path() {
    let mut j = job_at(ProtocolId::Fal, "a", T0);
    assert_eq!(
        (j.status(), j.progress, j.started_at),
        (JobStatus::Queued, 0.0, None)
    );
    assert_eq!(j.expires_at, T0 + secs(24 * 3600));
    j.queue_position = Some(3);
    j.mark_running(T0 + secs(5)).unwrap();
    assert_eq!(
        (j.status(), j.started_at, j.queue_position),
        (JobStatus::Running, Some(T0 + secs(5)), None)
    );
    j.set_step(5, 10);
    assert_eq!(j.progress, 0.5);
    j.mark_succeeded(
        T0 + secs(30),
        vec![],
        JobMetrics {
            inference_s: Some(20.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        (j.status(), j.progress, j.completed_at),
        (JobStatus::Succeeded, 1.0, Some(T0 + secs(30)))
    );
    assert_eq!(j.metrics.inference_s, Some(20.0));
    assert!(j.is_terminal());
}

#[test]
fn illegal_transitions_leave_job_untouched() {
    let mut j = job_at(ProtocolId::Fal, "a", T0);
    let e = j
        .mark_succeeded(T0, vec![], JobMetrics::default())
        .unwrap_err();
    assert_eq!((e.from, e.to), (JobStatus::Queued, JobStatus::Succeeded));
    assert_eq!(ApiError::from(e).kind, ErrorKind::Conflict);
    assert_eq!(j.status(), JobStatus::Queued);
    assert_eq!(j.completed_at, None);

    j.mark_cancelled(T0 + secs(1)).unwrap();
    assert!(j.cancel_requested);
    let before = j.clone();
    for r in [
        j.mark_running(T0 + secs(2)),
        j.mark_failed(T0 + secs(2), ApiError::internal("x")),
        j.mark_cancelled(T0 + secs(2)),
        j.mark_succeeded(T0 + secs(2), vec![], JobMetrics::default()),
    ] {
        let e = r.unwrap_err();
        assert_eq!(e.from, JobStatus::Cancelled);
        assert_eq!(ApiError::from(e).kind, ErrorKind::AlreadyCompleted);
    }
    assert_eq!(j, before, "terminal state never changes");
}

#[test]
fn fail_and_cancel_from_queued_and_running() {
    for running in [false, true] {
        let mut j = job_at(ProtocolId::LtxV2, "a", T0);
        if running {
            j.mark_running(T0).unwrap();
        }
        let mut f = j.clone();
        f.queue_position = Some(1);
        f.mark_failed(T0 + secs(1), ApiError::engine_failed("boom"))
            .unwrap();
        assert_eq!(f.state.error().unwrap().message, "boom");
        assert_eq!(
            (f.completed_at, f.queue_position),
            (Some(T0 + secs(1)), None)
        );
        j.mark_cancelled(T0 + secs(2)).unwrap();
        assert_eq!(j.status(), JobStatus::Cancelled);
    }
}

#[test]
fn progress_is_clamped_monotonic_and_frozen_when_terminal() {
    let mut j = job_at(ProtocolId::Fal, "a", T0);
    j.set_progress(0.3);
    j.set_progress(0.1);
    assert_eq!(j.progress, 0.3);
    j.set_progress(7.0);
    assert_eq!(j.progress, 1.0);
    let mut j = job_at(ProtocolId::Fal, "b", T0);
    j.set_progress(f32::NAN);
    j.set_step(1, 0);
    assert_eq!(j.progress, 0.0);
    j.mark_failed(T0, ApiError::internal("x")).unwrap();
    j.set_progress(0.9);
    assert_eq!(j.progress, 0.0);
}

#[test]
fn restart_recovery() {
    let mut q = job_at(ProtocolId::MiniMaxV2, "1", T0);
    assert!(q.recover_after_restart(T0 + secs(9)));
    let e = q.state.error().unwrap();
    assert_eq!(
        (e.kind, e.message.as_str()),
        (ErrorKind::Internal, "interrupted by restart")
    );
    let mut r = job_at(ProtocolId::MiniMaxV2, "2", T0);
    r.mark_running(T0).unwrap();
    assert!(r.recover_after_restart(T0));
    let mut done = job_at(ProtocolId::MiniMaxV2, "3", T0);
    done.mark_cancelled(T0).unwrap();
    assert!(!done.recover_after_restart(T0));
    assert_eq!(done.status(), JobStatus::Cancelled);
}

#[test]
fn retention_and_expiry() {
    assert_eq!(
        ProtocolId::MiniMaxV2.default_retention(),
        Duration::from_secs(7 * 86400)
    );
    for p in [
        ProtocolId::Fal,
        ProtocolId::LtxV1,
        ProtocolId::LtxV2,
        ProtocolId::OpenAiVideos,
        ProtocolId::FastWan,
    ] {
        assert_eq!(p.default_retention(), Duration::from_secs(86400), "{p:?}");
    }
    let j = job_at(ProtocolId::MiniMaxV2, "1", T0);
    assert!(!j.is_expired(T0 + secs(7 * 86400 - 1)));
    assert!(j.is_expired(T0 + secs(7 * 86400)));
}

#[test]
fn snapshot_reflects_job() {
    let mut j = job_at(ProtocolId::Fal, "a", T0);
    j.queue_position = Some(2);
    j.logs.push(LogLine::info("hi", T0));
    let s = j.snapshot(9);
    assert_eq!(
        (s.id, s.seq, s.state.clone(), s.queue_position, s.log_count),
        (j.id, 9, JobState::Queued, Some(2), 1)
    );
}

#[test]
fn requested_model_prefers_echo() {
    let mut j = job_at(ProtocolId::MiniMaxV2, "1", T0);
    assert_eq!(j.requested_model(), "fasth3");
    j.request_echo = serde_json::json!({"model": "MiniMax-H3"});
    assert_eq!(j.requested_model(), "MiniMax-H3");
    assert_eq!(j.task(), Task::T2V);
}

fn corpus() -> Vec<Job> {
    let mut v = Vec::new();
    for i in 0..6 {
        let p = if i % 2 == 0 {
            ProtocolId::MiniMaxV2
        } else {
            ProtocolId::Fal
        };
        let mut j = job_at(p, &format!("e{i}"), T0 + secs(i));
        j.owner = Some(KeyId(if i < 3 { "a" } else { "b" }.into()));
        j.request_echo =
            serde_json::json!({"model": if i == 4 { "MiniMax-H3-Max" } else { "MiniMax-H3" }});
        if i == 1 {
            j.mark_running(T0).unwrap();
        }
        if i == 2 {
            j.mark_failed(T0, ApiError::internal("x")).unwrap();
        }
        v.push(j);
    }
    v
}

fn ext(p: &Page<Job>) -> Vec<&str> {
    p.items.iter().map(|j| j.external_id.as_str()).collect()
}

#[test]
fn list_query_ordering_and_paging() {
    let jobs = corpus();
    let q = ListQuery {
        limit: 100,
        ..ListQuery::default()
    };
    let p = q.apply(&jobs);
    assert_eq!(
        (ext(&p), p.total, p.has_more),
        (vec!["e5", "e4", "e3", "e2", "e1", "e0"], 6, false)
    );
    let asc = ListQuery {
        order: SortOrder::Asc,
        limit: 2,
        ..ListQuery::default()
    };
    let p = asc.apply(&jobs);
    assert_eq!((ext(&p), p.total, p.has_more), (vec!["e0", "e1"], 6, true));
    assert_eq!(p.first().unwrap().external_id, "e0");
    assert_eq!(p.last().unwrap().external_id, "e1");
    // Cursor: strictly after e1.
    let p = ListQuery {
        after: Some("e1".into()),
        ..asc.clone()
    }
    .apply(&jobs);
    assert_eq!((ext(&p), p.has_more), (vec!["e2", "e3"], true));
    let p = ListQuery {
        after: Some("e4".into()),
        ..asc.clone()
    }
    .apply(&jobs);
    assert_eq!((ext(&p), p.has_more), (vec!["e5"], false));
    // Unknown cursor: empty page.
    let p = ListQuery {
        after: Some("zz".into()),
        ..asc.clone()
    }
    .apply(&jobs);
    assert!(p.items.is_empty() && !p.has_more);
    // Offset paging (MiniMax page_num/page_size).
    let p = ListQuery {
        offset: 4,
        limit: 10,
        ..asc.clone()
    }
    .apply(&jobs);
    assert_eq!(ext(&p), vec!["e4", "e5"]);
    let p = ListQuery { offset: 40, ..asc }.apply(&jobs);
    assert!(p.items.is_empty());
    assert_eq!(p.total, 6);
}

#[test]
fn list_query_filters() {
    let jobs = corpus();
    let run = |q: ListQuery| {
        ext(&ListQuery {
            order: SortOrder::Asc,
            limit: 100,
            ..q
        }
        .apply(&jobs))
        .join(",")
    };
    assert_eq!(
        run(ListQuery {
            owner: Some(KeyId("b".into())),
            ..Default::default()
        }),
        "e3,e4,e5"
    );
    assert_eq!(
        run(ListQuery {
            protocol: Some(ProtocolId::Fal),
            ..Default::default()
        }),
        "e1,e3,e5"
    );
    assert_eq!(
        run(ListQuery {
            statuses: vec![JobStatus::Running, JobStatus::Failed],
            ..Default::default()
        }),
        "e1,e2"
    );
    assert_eq!(
        run(ListQuery {
            model: Some("MiniMax-H3-Max".into()),
            ..Default::default()
        }),
        "e4"
    );
    assert_eq!(
        run(ListQuery {
            model: Some("fasth3".into()),
            ..Default::default()
        }),
        "e0,e1,e2,e3,e4,e5"
    );
    assert_eq!(
        run(ListQuery {
            task: Some(Task::I2V),
            ..Default::default()
        }),
        ""
    );
    assert_eq!(
        run(ListQuery {
            external_ids: vec!["e2".into(), "e5".into()],
            ..Default::default()
        }),
        "e2,e5"
    );
    assert_eq!(
        run(ListQuery {
            owner: Some(KeyId("a".into())),
            protocol: Some(ProtocolId::MiniMaxV2),
            statuses: vec![JobStatus::Queued],
            ..Default::default()
        }),
        "e0"
    );
}

#[test]
fn store_errors_map_to_api_errors() {
    let id = JobId::new();
    let cases = [
        (StoreError::NotFound(id), ErrorKind::NotFound),
        (StoreError::Full, ErrorKind::QueueFull),
        (StoreError::AlreadyExists(id), ErrorKind::Conflict),
        (
            StoreError::DuplicateExternal(ProtocolId::Fal, "x".into()),
            ErrorKind::Conflict,
        ),
        (StoreError::Io("disk".into()), ErrorKind::Internal),
    ];
    for (e, k) in cases {
        let msg = e.to_string();
        let a = ApiError::from(e);
        assert_eq!((a.kind, a.message), (k, msg));
    }
}

/// `JobStore` must stay object-safe: serve-kit stores `Arc<dyn JobStore>`.
#[test]
fn job_store_is_object_safe() {
    fn takes(_: Option<&dyn JobStore>) {}
    takes(None);
    let _: Option<JobUpdate> = Some(Box::new(|j: &mut Job| j.progress = 0.5));
}
