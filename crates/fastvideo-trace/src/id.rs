//! Trace ids and the W3C `traceparent` header
//! (`00-<32 hex trace id>-<16 hex parent id>-<2 hex flags>`).

use std::fmt;

use crate::recorder::{global, now_ns, Clock, Comp, Rec};

/// A 128-bit trace id (W3C trace-context).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct TraceId(pub [u8; 16]);

impl TraceId {
    /// A random id (v4 UUID bytes: 122 random bits).
    pub fn random() -> Self {
        Self(*uuid::Uuid::new_v4().as_bytes())
    }

    /// 32 hex digits, not all zero (W3C: the all-zero id is invalid).
    pub fn parse_hex(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.len() != 32 {
            return None;
        }
        let mut b = [0u8; 16];
        for (i, out) in b.iter_mut().enumerate() {
            *out = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
        }
        (b != [0u8; 16]).then_some(Self(b))
    }

    pub fn hex(&self) -> String {
        let mut s = String::with_capacity(32);
        for b in self.0 {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// A random non-zero 64-bit span id.
pub fn new_span_id() -> u64 {
    let b = uuid::Uuid::new_v4();
    let (hi, _) = b.as_u64_pair();
    hi.max(1)
}

/// A traced request's context: its trace id and the caller's span id.
/// `Copy` and 24 bytes, so it rides along in jobs and closures for free.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Trace {
    pub id: TraceId,
    /// The caller's span (`traceparent` parent id); 0 for a new root.
    pub parent: u64,
}

impl Trace {
    /// A new trace (random id, no parent).
    pub fn new_root() -> Self {
        Self { id: TraceId::random(), parent: 0 }
    }

    /// Parses `traceparent` (version 00 layout; later versions are read the
    /// same way, as the spec asks). The sampled flag is ignored: whether a
    /// request is traced is decided by [`crate::policy`].
    pub fn from_traceparent(v: &str) -> Option<Self> {
        let mut parts = v.trim().split('-');
        let version = parts.next()?;
        let id = parts.next()?;
        let parent = parts.next()?;
        let flags = parts.next()?;
        if version.len() != 2 || version == "ff" || parent.len() != 16 || flags.len() != 2 {
            return None;
        }
        u8::from_str_radix(version, 16).ok()?;
        u8::from_str_radix(flags, 16).ok()?;
        let id = TraceId::parse_hex(id)?;
        let parent = u64::from_str_radix(parent, 16).ok()?;
        (parent != 0).then_some(Self { id, parent })
    }

    /// `traceparent` naming `span` as the parent of the next hop (sampled).
    pub fn traceparent(&self, span: u64) -> String {
        format!("00-{}-{:016x}-01", self.id, span.max(1))
    }

    /// The same trace with `span` as the parent of what follows.
    pub fn child(&self, span: u64) -> Self {
        Self { id: self.id, parent: span }
    }

    /// A point event now. Hot path: a clock read and a channel push.
    #[inline]
    pub fn point(&self, comp: Comp, name: &'static str, arg: i64) {
        global().emit(Rec::point(self.id, comp, name, now_ns(), arg));
    }

    /// A span from `start_ns` ([`now_ns`]) to now.
    #[inline]
    pub fn span_since(&self, comp: Comp, name: &'static str, start_ns: u64, arg: i64) {
        let now = now_ns();
        global().emit(Rec {
            trace: self.id,
            comp,
            name,
            stage: "",
            clock: Clock::Host,
            t_ns: start_ns,
            dur_ns: now.saturating_sub(start_ns),
            arg,
        });
    }

    /// A guard that records a span from now until it is dropped.
    #[inline]
    pub fn span(&self, comp: Comp, name: &'static str) -> Span {
        Span { trace: *self, comp, name, start: now_ns(), arg: 0 }
    }
}

/// See [`Trace::span`].
#[must_use = "the span ends when the guard is dropped"]
pub struct Span {
    trace: Trace,
    comp: Comp,
    name: &'static str,
    start: u64,
    arg: i64,
}

impl Span {
    /// Sets the span's argument (a status code, a byte count, a step).
    pub fn arg(&mut self, arg: i64) {
        self.arg = arg;
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        self.trace.span_since(self.comp, self.name, self.start, self.arg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traceparent_round_trip() {
        let v = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let t = Trace::from_traceparent(v).unwrap();
        assert_eq!(t.id.hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(t.parent, 0x00f067aa0ba902b7);
        assert_eq!(t.traceparent(t.parent), v);
        // Unsampled is still parsed; policy decides.
        assert!(Trace::from_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00").is_some());
    }

    #[test]
    fn traceparent_rejects_bad_values() {
        for v in [
            "",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e47zz-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        ] {
            assert!(Trace::from_traceparent(v).is_none(), "{v}");
        }
    }

    #[test]
    fn random_ids_differ() {
        assert_ne!(TraceId::random(), TraceId::random());
        assert_eq!(TraceId::parse_hex(&TraceId::random().hex()).map(|t| t.hex().len()), Some(32));
    }
}
