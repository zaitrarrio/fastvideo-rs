//! Which requests are traced.
//!
//! `FV_TRACE` (read once): `off` traces nothing and ignores the headers;
//! `opt-in` (the default) traces a request that asks with
//! `x-fv-trace: 1` (or the query flag `fv_trace=1`, which the HTTP layer
//! maps to the same answer); `all` traces every request. A plain
//! `traceparent` from an instrumented client does not turn tracing on by
//! itself (many SDKs send one on every call); it only supplies the trace id
//! when the request opted in.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::id::Trace;

/// The opt-in request header (`1`, `true` or `on`).
pub const OPT_IN_HEADER: &str = "x-fv-trace";
/// W3C trace-context header.
pub const TRACEPARENT: &str = "traceparent";
/// Response header of a traced hop: `<recv wall ns>;<send wall ns>` on the
/// answering host's clock (the NTP-style samples of docs/serve/tracing.md
/// "Clock alignment").
pub const TIME_HEADER: &str = "x-fv-trace-t";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    OptIn,
    All,
}

impl Mode {
    /// `off|0|false|no`, `all|always`; anything else (empty included) is opt-in.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "no" | "none" => Mode::Off,
            "all" | "always" => Mode::All,
            _ => Mode::OptIn,
        }
    }

    fn code(self) -> u8 {
        match self {
            Mode::Off => 1,
            Mode::OptIn => 2,
            Mode::All => 3,
        }
    }
}

static MODE: AtomicU8 = AtomicU8::new(0);

/// The process's mode (`FV_TRACE`, read on first use; [`set_mode`] wins).
pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        1 => Mode::Off,
        2 => Mode::OptIn,
        3 => Mode::All,
        _ => {
            let m = Mode::parse(&std::env::var("FV_TRACE").unwrap_or_default());
            MODE.store(m.code(), Ordering::Relaxed);
            m
        }
    }
}

/// Overrides `FV_TRACE` (config, tests).
pub fn set_mode(m: Mode) {
    MODE.store(m.code(), Ordering::Relaxed);
}

fn truthy(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

/// Whether a request is traced, and under which id: `traceparent` (when
/// valid) supplies the id, else a new root. `opt_in` is the
/// [`OPT_IN_HEADER`] value (or `Some("1")` for the query flag).
pub fn decide(mode: Mode, traceparent: Option<&str>, opt_in: Option<&str>) -> Option<Trace> {
    let wanted = match mode {
        Mode::Off => false,
        Mode::OptIn => opt_in.is_some_and(truthy),
        Mode::All => true,
    };
    if !wanted {
        return None;
    }
    Some(
        traceparent
            .and_then(Trace::from_traceparent)
            .unwrap_or_else(Trace::new_root),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TP: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn opt_in_needs_the_flag() {
        assert!(
            decide(Mode::OptIn, Some(TP), None).is_none(),
            "a bare traceparent does not opt in"
        );
        assert!(decide(Mode::OptIn, None, Some("0")).is_none());
        let t = decide(Mode::OptIn, Some(TP), Some("1")).unwrap();
        assert_eq!(t.id.hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert!(
            decide(Mode::OptIn, Some("junk"), Some("true")).is_some(),
            "a new root when traceparent is bad"
        );
    }

    #[test]
    fn off_and_all() {
        assert!(decide(Mode::Off, Some(TP), Some("1")).is_none());
        assert!(decide(Mode::All, None, None).is_some());
        assert_eq!(Mode::parse(""), Mode::OptIn);
        assert_eq!(Mode::parse("OFF"), Mode::Off);
        assert_eq!(Mode::parse("all"), Mode::All);
    }
}
