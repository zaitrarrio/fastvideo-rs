//! The chat-completions seam and the hedge (the TS spec's `callModel` /
//! `expandWithHedge`, docs/serve/upsampler-spec.ts).
//!
//! [`ChatBackend`] is one `/chat/completions` call that returns the first
//! choice's message content; [`crate::http::OpenAiChat`] is the HTTP one
//! (feature `http`), tests use in-process fakes. [`hedged`] runs one or two
//! attempts at once: the first success wins, a failed attempt waits for the
//! other, and when the timeout fires every attempt still running is dropped
//! (which aborts its HTTP request), like the spec's `AbortController`s.

use std::future::Future;
use std::time::Duration;

use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::Value;

/// One chat request: a system and a user message.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatRequest {
    pub system: String,
    pub user: String,
    pub temperature: f32,
    /// `response_format` as sent (the director's strict `json_schema`);
    /// `None`: plain text.
    pub response_format: Option<Value>,
}

/// Why an upsampling call produced nothing usable. Messages never carry
/// the API key.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum UpsampleError {
    /// A non-2xx answer (the spec's `upsampler LLM ${status}: ${text}`; the
    /// body is cut to 500 characters).
    #[error("upsampler LLM {status}: {body}")]
    Status { status: u16, body: String },
    /// No `choices[0].message.content` (the spec's "returned no content").
    #[error("upsampler LLM returned no content")]
    NoContent,
    /// The content is not the expected JSON (the spec's `JSON.parse`).
    #[error("upsampler LLM output is not valid: {0}")]
    BadOutput(String),
    /// Connection, TLS or body-read failure.
    #[error("upsampler LLM request failed: {0}")]
    Transport(String),
    /// The whole call (every hedged attempt) ran past the timeout.
    #[error("upsampler LLM timed out after {0} ms")]
    Timeout(u64),
    /// Every hedged attempt failed (the spec's `AggregateError`).
    #[error("every upsampler attempt failed: {0}")]
    AllFailed(String),
    /// No backend (built without `http`, or a configuration error).
    #[error("upsampler unavailable: {0}")]
    Unavailable(String),
}

impl UpsampleError {
    /// A short label for metrics (`outcome`).
    pub fn label(&self) -> &'static str {
        match self {
            UpsampleError::Status { .. } => "status",
            UpsampleError::NoContent => "no_content",
            UpsampleError::BadOutput(_) => "bad_output",
            UpsampleError::Transport(_) => "transport",
            UpsampleError::Timeout(_) => "timeout",
            UpsampleError::AllFailed(_) => "all_failed",
            UpsampleError::Unavailable(_) => "unavailable",
        }
    }
}

/// One `/chat/completions` call: the first choice's message content.
#[async_trait::async_trait]
pub trait ChatBackend: Send + Sync + 'static {
    async fn complete(&self, req: &ChatRequest) -> Result<String, UpsampleError>;
    /// What to log about the backend (never the key).
    fn describe(&self) -> String {
        "chat backend".into()
    }
}

/// Runs `attempts` (1, or 2 with the hedge) calls of `f` at once: the first
/// `Ok` wins and the others are dropped; all failing gives
/// [`UpsampleError::AllFailed`] (one attempt: its own error); the timeout
/// drops every attempt still running and gives [`UpsampleError::Timeout`].
pub async fn hedged<T, F, Fut>(attempts: usize, timeout: Duration, f: F) -> Result<T, UpsampleError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, UpsampleError>>,
{
    let mut set: FuturesUnordered<Fut> = (0..attempts.max(1)).map(|_| f()).collect();
    let race = async move {
        let mut errs = Vec::new();
        while let Some(r) = set.next().await {
            match r {
                Ok(v) => return Ok(v),
                Err(e) => errs.push(e),
            }
        }
        Err(if errs.len() == 1 {
            errs.remove(0)
        } else {
            UpsampleError::AllFailed(errs.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))
        })
    };
    match tokio::time::timeout(timeout, race).await {
        Ok(r) => r,
        Err(_) => Err(UpsampleError::Timeout(timeout.as_millis() as u64)),
    }
}

/// The answer without a leading reasoning block: reasoning models may emit
/// `<think>…</think>` before the answer (or only the closing tag, when the
/// chat template opened the block). An unclosed block leaves nothing.
pub fn strip_think(s: &str) -> &str {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix("<think>") {
        return rest.split_once("</think>").map_or("", |(_, a)| a).trim_start();
    }
    match t.split_once("</think>") {
        Some((_, a)) if !t.contains("<think>") => a.trim_start(),
        _ => t,
    }
}

/// A JSON answer as small local models write it: without a leading think
/// block and without a Markdown code fence around it.
pub fn json_body(s: &str) -> &str {
    let t = strip_think(s).trim();
    let Some(rest) = t.strip_prefix("```") else { return t };
    // An optional language tag on the fence line.
    let rest = rest.split_once('\n').map_or(rest, |(tag, body)| if tag.trim().contains(' ') { rest } else { body });
    rest.trim_end().strip_suffix("```").unwrap_or(rest).trim()
}

/// Cuts `s` to at most `n` characters (error bodies in logs).
pub fn clip(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn first_success_wins_and_a_failure_waits_for_the_other() {
        let n = Arc::new(AtomicUsize::new(0));
        // Attempt 0 fails at once, attempt 1 succeeds after 20 ms.
        let r = hedged(2, Duration::from_secs(5), || {
            let k = n.fetch_add(1, Ordering::SeqCst);
            async move {
                if k == 0 {
                    Err(UpsampleError::NoContent)
                } else {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    Ok(k)
                }
            }
        })
        .await;
        assert_eq!(r, Ok(1));
        assert_eq!(n.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn without_the_hedge_one_attempt_and_its_own_error() {
        let n = Arc::new(AtomicUsize::new(0));
        let r: Result<(), _> = hedged(1, Duration::from_secs(5), || {
            n.fetch_add(1, Ordering::SeqCst);
            async { Err(UpsampleError::Status { status: 500, body: "boom".into() }) }
        })
        .await;
        assert_eq!(r, Err(UpsampleError::Status { status: 500, body: "boom".into() }));
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn both_failing_aggregates() {
        let r: Result<(), _> = hedged(2, Duration::from_secs(5), || async { Err(UpsampleError::NoContent) }).await;
        let Err(UpsampleError::AllFailed(m)) = r else { panic!("{r:?}") };
        assert_eq!(m, "upsampler LLM returned no content; upsampler LLM returned no content");
    }

    #[tokio::test]
    async fn the_timeout_drops_every_attempt() {
        let dropped = Arc::new(AtomicUsize::new(0));
        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let t0 = std::time::Instant::now();
        let r: Result<(), _> = hedged(2, Duration::from_millis(50), || {
            let g = Guard(dropped.clone());
            async move {
                let _g = g;
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok(())
            }
        })
        .await;
        assert_eq!(r, Err(UpsampleError::Timeout(50)));
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert_eq!(dropped.load(Ordering::SeqCst), 2, "both attempts aborted");
    }

    #[test]
    fn think_blocks_and_fences_are_stripped() {
        assert_eq!(strip_think("<think>\nhmm {\"a\":1}\n</think>\n\n{\"b\":2}"), "{\"b\":2}");
        assert_eq!(strip_think("  reasoning…</think>{\"b\":2}"), "{\"b\":2}");
        assert_eq!(strip_think("<think>never closed"), "");
        assert_eq!(strip_think("A cat walks."), "A cat walks.");
        assert_eq!(json_body("<think>x</think>```json\n{\"b\":2}\n```"), "{\"b\":2}");
        assert_eq!(json_body("```\n{\"b\":2}```"), "{\"b\":2}");
        assert_eq!(json_body(" {\"b\":2} "), "{\"b\":2}");
    }

    #[test]
    fn clip_counts_characters() {
        assert_eq!(clip("héllo", 2), "hé…");
        assert_eq!(clip("hi", 2), "hi");
    }
}
