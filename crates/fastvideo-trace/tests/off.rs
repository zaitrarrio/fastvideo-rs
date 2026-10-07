//! Tracing off costs nothing: an untraced request decides `None` once and
//! never touches the recorder, whose drain thread is never started. Its own
//! test binary, so no other test has started the global recorder.

use std::time::{Duration, Instant};

use fastvideo_trace::{current, decide, started, Mode, Trace};

/// What every call site does: `if let Some(t) = trace { t.point(..) }`.
#[inline(never)]
fn site(trace: Option<&Trace>, i: i64) -> i64 {
    if let Some(t) = trace {
        t.point(fastvideo_trace::Comp::Engine, "step", i);
    }
    i
}

#[test]
fn untraced_requests_never_start_the_recorder() {
    // A request without the opt-in header, in each mode but `all`.
    for mode in [Mode::Off, Mode::OptIn] {
        assert!(decide(mode, Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"), None).is_none());
    }
    assert!(current().is_none(), "no trace outside a traced request");
    let t0 = Instant::now();
    let mut acc = 0i64;
    for i in 0..10_000_000 {
        acc = acc.wrapping_add(site(std::hint::black_box(None), i));
    }
    let took = t0.elapsed();
    std::hint::black_box(acc);
    assert!(!started(), "the recorder (and its thread) exist only once something is traced");
    // 10 M untraced call sites: a branch each (well under 1 s even unoptimised).
    assert!(took < Duration::from_secs(5), "{took:?}");
}
