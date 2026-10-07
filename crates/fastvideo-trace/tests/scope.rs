//! The task-local trace follows a request's task (and only it).

use fastvideo_trace::{current, current_traceparent, scope, Trace};

#[tokio::test]
async fn scope_sets_current_for_the_task_only() {
    let t = Trace::from_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01").unwrap();
    let inner = scope(t, async {
        let seen = current();
        let tp = current_traceparent();
        // A spawned task does not inherit it (call sites pass the trace on).
        let spawned = tokio::spawn(async { current() }).await.unwrap();
        (seen, tp, spawned)
    })
    .await;
    assert_eq!(inner.0, Some(t));
    assert_eq!(inner.1.as_deref(), Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"));
    assert_eq!(inner.2, None);
    assert_eq!(current(), None);
}
