//! A traced job (docs/serve/tracing.md) over the fake engine: the queue
//! wait, the run, every stage and denoise step on the host clock and on the
//! device clock (the fake's marks stand in for CUDA events: recorded during
//! the run, resolved after it on the trace drain thread).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::*;
use fastvideo_engine_service::{EngineConfig, EngineEvent, ManualClock, Priority};
use fastvideo_protocol::JobId;
use fastvideo_trace::Trace;

#[tokio::test]
async fn a_traced_job_records_queue_run_stages_and_device_steps() {
    let clock = Arc::new(ManualClock::new());
    let e = ready(EngineConfig::default(), manual_fake(&clock)).await;
    let _drive = Driver::new(&clock);
    let t = Trace::new_root();
    let mut h = e.submit_traced(JobId::new(), wan("traced"), Priority::Batch, Some(t)).await.unwrap();
    let evs = until_terminal(&mut h).await;
    assert!(matches!(evs.last(), Some(EngineEvent::Finished(_))), "{evs:?}");

    let d = fastvideo_trace::snapshot(&t.id.hex(), Duration::from_secs(10)).expect("trace recorded");
    let has = |comp: &str, name: &str| d.events.iter().any(|e| e.comp == comp && e.name == name);
    assert!(has("queue", "wait"), "{:#?}", d.events);
    assert!(has("engine", "run"));
    assert!(has("engine", "text_encode"));
    // Device spans: one per denoise step, each the scripted 1 s step.
    let steps: Vec<_> = d.events.iter().filter(|e| e.comp == "gpu" && e.name == "denoise.step").collect();
    assert_eq!(steps.len(), 3, "{:#?}", d.events);
    for (i, s) in steps.iter().enumerate() {
        assert_eq!(s.clock, "gpu");
        assert_eq!(s.arg, Some(i as i64 + 1));
        assert_eq!(s.dur_ns, 1_000_000_000, "step {}", i + 1);
    }
    // Host spans of the same steps exist too (host clock).
    assert_eq!(d.events.iter().filter(|e| e.comp == "engine" && e.name == "denoise.step").count(), 3);
    // Resolved off the engine's thread, by the drain.
    assert!(fastvideo_engine_service::fake::marks_resolved_on().iter().any(|n| n == "fv-trace-drain"));
    assert_eq!(d.stats.dropped, 0);
}

#[tokio::test]
async fn an_untraced_job_records_nothing() {
    let clock = Arc::new(ManualClock::new());
    let e = ready(EngineConfig::default(), manual_fake(&clock)).await;
    let _drive = Driver::new(&clock);
    let before = fastvideo_trace::recent().len();
    let mut h = e.submit(JobId::new(), wan("plain"), Priority::Batch).await.unwrap();
    assert!(matches!(until_terminal(&mut h).await.last(), Some(EngineEvent::Finished(_))));
    if fastvideo_trace::started() {
        fastvideo_trace::global().flush(Duration::from_secs(5));
    }
    // Only the other test's trace (if it ran first) can be there.
    assert!(fastvideo_trace::recent().len() <= before.max(1));
}
