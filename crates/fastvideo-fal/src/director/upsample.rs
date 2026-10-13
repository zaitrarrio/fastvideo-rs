//! The director's prompt upsampling (docs/serve/prompt-upsampling.md
//! §Director): which segment needs which expansion, expanded ahead of time.
//!
//! A session that turned it on (`configure.upsample`) holds one
//! [`SegmentUpsampler`] around the showrunner [`Upsampler`] (the port of
//! the owner's spec). Every segment the director needs (each clip chunk;
//! each director chunk or prompt switch of a causal rollout) takes its
//! prompt from `Upsampler.next(direction)`: a new client `prompt` message
//! is the [`Direction`] (id = its `prompt_version`), no pending direction is
//! `null` (advance one beat).
//!
//! The call runs off the critical path: the next beat is expanded while the
//! current segment generates (and a new direction starts its expansion the
//! moment it arrives, dropping a speculative one), so a segment normally
//! finds its prompt ready. One that does not waits for it (clip models: the
//! chunk is held; causal models: the engine keeps its current prompt), at
//! most the configured timeout. A failed or timed-out call falls back to
//! the raw direction / premise text, which the client sees in
//! `segment_expanded` (`fallback: true`, `error`), so the stream never
//! stalls on the LLM.

use std::time::{Duration, Instant};

use fastvideo_upsampler::director::{expand_with_hedge, ExpansionOutput};
use fastvideo_upsampler::{Direction, ExpandedSegment, Kind, UpsampleError, Upsampler};
use tokio::sync::mpsc;

/// The new direction a segment carries: `(prompt_version, text)`, or
/// `None` (advance one beat).
pub type Key = Option<(u64, String)>;

/// A finished expansion, from its task to the session loop.
#[derive(Debug)]
pub struct Done {
    seq: u64,
    result: Result<ExpansionOutput, UpsampleError>,
    latency: Duration,
}

struct Inflight {
    seq: u64,
    key: Key,
    ledger_len: usize,
    handle: tokio::task::JoinHandle<()>,
}

struct Ready {
    key: Key,
    ledger_len: usize,
    result: Result<ExpansionOutput, UpsampleError>,
    latency: Duration,
}

/// A segment's prompt: the upsampler's, or the raw text it fell back to.
#[derive(Clone, Debug, PartialEq)]
pub struct Taken {
    pub prompt: String,
    /// `None` on a fallback.
    pub segment: Option<ExpandedSegment>,
    pub error: Option<String>,
    pub latency: Duration,
}

/// One session's upsampling (module docs).
pub struct SegmentUpsampler {
    up: Upsampler,
    tx: mpsc::UnboundedSender<Done>,
    seq: u64,
    inflight: Option<Inflight>,
    ready: Option<Ready>,
    /// The last direction a segment consumed.
    last: Key,
    /// Calls started (for logs and tests).
    started: u64,
}

impl Drop for SegmentUpsampler {
    fn drop(&mut self) {
        if let Some(f) = self.inflight.take() {
            f.handle.abort();
        }
    }
}

fn direction(key: &Key) -> Option<Direction> {
    key.as_ref().map(|(v, t)| Direction { id: v.to_string(), text: t.clone() })
}

impl SegmentUpsampler {
    /// `tx` feeds the session loop, which hands each [`Done`] back to
    /// [`Self::on_done`].
    pub fn new(up: Upsampler, tx: mpsc::UnboundedSender<Done>) -> Self {
        Self { up, tx, seq: 0, inflight: None, ready: None, last: None, started: 0 }
    }

    pub fn upsampler(&self) -> &Upsampler {
        &self.up
    }

    /// LLM calls started so far.
    pub fn started(&self) -> u64 {
        self.started
    }

    /// Whether a call is running.
    pub fn busy(&self) -> bool {
        self.inflight.is_some()
    }

    /// The key of a planned segment: its direction when that is new.
    pub fn key_for(&self, version: u64, direction: Option<&str>) -> Key {
        let d = direction.map(str::trim).filter(|d| !d.is_empty())?;
        let k = Some((version, d.to_owned()));
        if k == self.last {
            None
        } else {
            k
        }
    }

    /// Makes sure the expansion for `key` on the current ledger is running
    /// or ready; anything else (a speculative beat when a direction came)
    /// is dropped.
    pub fn request(&mut self, key: &Key) {
        let len = self.up.ledger_len();
        if self.ready.as_ref().is_some_and(|r| &r.key == key && r.ledger_len == len) {
            return;
        }
        if self.inflight.as_ref().is_some_and(|f| &f.key == key && f.ledger_len == len) {
            return;
        }
        if let Some(f) = self.inflight.take() {
            f.handle.abort();
        }
        self.ready = None;
        self.seq += 1;
        self.started += 1;
        let seq = self.seq;
        let user = self.up.user_message(direction(key).as_ref());
        let backend = self.up.backend().clone();
        let (hedge, timeout) = (self.up.config().hedge, self.up.config().timeout);
        let tx = self.tx.clone();
        let handle = tokio::spawn(async move {
            let t0 = Instant::now();
            let result = expand_with_hedge(&backend, hedge, timeout, user).await;
            let _ = tx.send(Done { seq, result, latency: t0.elapsed() });
        });
        self.inflight = Some(Inflight { seq, key: key.clone(), ledger_len: len, handle });
    }

    /// A finished call: `true` when it is the one awaited (stale ones are
    /// dropped).
    pub fn on_done(&mut self, d: Done) -> bool {
        match &self.inflight {
            Some(f) if f.seq == d.seq => {
                let f = self.inflight.take().expect("inflight");
                self.ready = Some(Ready { key: f.key, ledger_len: f.ledger_len, result: d.result, latency: d.latency });
                true
            }
            _ => false,
        }
    }

    /// The segment for `key` when its expansion is ready: committed to the
    /// ledger, or (the call failed) `fallback`, the raw text. `None`: not
    /// ready yet ([`Self::request`] it first).
    pub fn take(&mut self, key: &Key, fallback: &str) -> Option<Taken> {
        let len = self.up.ledger_len();
        if !self.ready.as_ref().is_some_and(|r| &r.key == key && r.ledger_len == len) {
            return None;
        }
        let r = self.ready.take().expect("ready");
        if key.is_some() {
            self.last = key.clone();
        }
        Some(match r.result {
            Ok(out) => {
                fastvideo_upsampler::record(Kind::Director, Ok(()), r.latency);
                let seg = self.up.commit(direction(key).as_ref(), out);
                Taken { prompt: seg.record.prompt.clone(), segment: Some(seg), error: None, latency: r.latency }
            }
            Err(e) => {
                fastvideo_upsampler::record(Kind::Director, Err(&e), r.latency);
                Taken { prompt: fallback.to_owned(), segment: None, error: Some(e.to_string()), latency: r.latency }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use fastvideo_upsampler::{ChatBackend, ChatRequest, UpsamplerConfig, WorldBible};

    /// Answers each call with a segment whose summary names the call, after
    /// `delay`; records the user messages.
    struct Llm {
        n: AtomicUsize,
        delay: Duration,
        fail: bool,
        seen: Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl ChatBackend for Llm {
        async fn complete(&self, req: &ChatRequest) -> Result<String, UpsampleError> {
            let n = self.n.fetch_add(1, Ordering::SeqCst);
            self.seen.lock().unwrap().push(req.user.clone());
            tokio::time::sleep(self.delay).await;
            if self.fail {
                return Err(UpsampleError::Status { status: 503, body: "down".into() });
            }
            Ok(serde_json::json!({
                "beat_summary": format!("beat {n}"), "entity_ids": [], "new_entities": [],
                "action": format!("action {n}."), "camera": "c", "lighting": "l", "audio": "a", "dialogue": [],
            })
            .to_string())
        }
    }

    fn setup(delay_ms: u64, fail: bool) -> (SegmentUpsampler, mpsc::UnboundedReceiver<Done>, Arc<Llm>) {
        let llm = Arc::new(Llm { n: AtomicUsize::new(0), delay: Duration::from_millis(delay_ms), fail, seen: Mutex::new(Vec::new()) });
        let cfg = UpsamplerConfig { memory: 12, max_prompt_chars: 1800, hedge: false, timeout: Duration::from_secs(5) };
        let up = Upsampler::new(cfg, WorldBible { premise: "premise".into(), ..WorldBible::default() }, llm.clone());
        let (tx, rx) = mpsc::unbounded_channel();
        (SegmentUpsampler::new(up, tx), rx, llm)
    }

    async fn settle(s: &mut SegmentUpsampler, rx: &mut mpsc::UnboundedReceiver<Done>) {
        let d = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap();
        s.on_done(d);
    }

    #[tokio::test]
    async fn segments_advance_beats_and_take_new_directions_once() {
        let (mut s, mut rx, llm) = setup(0, false);
        // Chunk 0: no direction (premise only).
        let k0 = s.key_for(1, None);
        assert_eq!(k0, None);
        s.request(&k0);
        assert!(s.take(&k0, "premise").is_none(), "not ready yet");
        settle(&mut s, &mut rx).await;
        let t = s.take(&k0, "premise").unwrap();
        assert_eq!(t.prompt, "action 0. Camera: c Lighting: l Sound: a No spoken words.");
        assert_eq!(t.segment.as_ref().unwrap().record.index, 0);
        // A direction (v2) is new once; the next chunk carrying it again
        // advances a beat.
        let k1 = s.key_for(2, Some("A storm."));
        assert_eq!(k1, Some((2, "A storm.".into())));
        s.request(&k1);
        settle(&mut s, &mut rx).await;
        let t = s.take(&k1, "premise A storm.").unwrap();
        assert_eq!(t.segment.as_ref().unwrap().record.direction_id.as_deref(), Some("2"));
        assert_eq!(s.key_for(2, Some("A storm.")), None);
        assert!(llm.seen.lock().unwrap()[1].ends_with("NEW DIRECTION:\nA storm."));
        assert!(llm.seen.lock().unwrap()[1].contains("[0] beat 0"));
        assert_eq!(s.upsampler().ledger_len(), 2);
    }

    #[tokio::test]
    async fn a_new_direction_replaces_a_speculative_beat() {
        let (mut s, mut rx, llm) = setup(200, false);
        // Prefetch the next beat, then a direction arrives.
        s.request(&None);
        assert!(s.busy());
        let k = s.key_for(3, Some("Cut to the harbour."));
        s.request(&k);
        // Requesting the same key again changes nothing.
        s.request(&k);
        assert_eq!(s.started(), 2);
        settle(&mut s, &mut rx).await;
        assert!(s.take(&None, "x").is_none(), "the speculative beat is gone");
        let t = s.take(&k, "x").unwrap();
        assert_eq!(t.segment.unwrap().record.direction_id.as_deref(), Some("3"));
        // The aborted call never answered.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(rx.try_recv().is_err());
        let seen = llm.seen.lock().unwrap();
        assert!(seen.last().unwrap().ends_with("NEW DIRECTION:\nCut to the harbour."));
    }

    #[tokio::test]
    async fn a_failed_call_falls_back_to_the_raw_text() {
        let (mut s, mut rx, _) = setup(0, true);
        let k = s.key_for(2, Some("Night falls."));
        s.request(&k);
        settle(&mut s, &mut rx).await;
        let t = s.take(&k, "premise Night falls.").unwrap();
        assert_eq!(t.prompt, "premise Night falls.");
        assert!(t.segment.is_none());
        assert_eq!(t.error.as_deref(), Some("upsampler LLM 503: down"));
        // Consumed: the next segment advances a beat; nothing was recorded.
        assert_eq!(s.key_for(2, Some("Night falls.")), None);
        assert_eq!(s.upsampler().ledger_len(), 0);
    }
}
