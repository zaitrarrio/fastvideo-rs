//! The director's prompt upsampling (docs/serve/prompt-upsampling.md
//! §Director), modelled on fal's H3 Max director: a **prompt deck** of
//! planned segment prompts.
//!
//! A session that turned it on (`configure.upsample`) holds one
//! [`SegmentUpsampler`] around the showrunner [`Upsampler`]. The session
//! tells it, after every change, which segments come next ([`Cue`]s, from
//! the control's planned chunks: a new direction, a given first frame, an
//! end frame to arrive at, or nothing = advance one beat), and it keeps up
//! to `deck_size` of them expanded ahead ([`SegmentUpsampler::sync`]):
//!
//! - one LLM call at a time, front first, so the next segments are always
//!   expanded before speculative ones further out;
//! - each planned segment is written as if the ones before it had played
//!   ([`Upsampler::user_message_after`]), so the plan is one continuous
//!   story;
//! - the planned prefix whose cues still match is kept; the rest is
//!   discarded and an in-flight call for it aborted. So `replan: true`
//!   (a new direction at the next chunk) busts the whole deck, and
//!   `replan: false` (appended after the planned chunks) keeps it;
//! - only the segment that executes ([`SegmentUpsampler::take`]) is
//!   committed to the ledger and memory.
//!
//! A segment whose expansion is not ready waits for it (clip models: the
//! chunk is held; causal models: the engine keeps its current prompt), at
//! most the configured timeout. A failed or timed-out call falls back to
//! [`Upsampler::fallback_prompt`] (the direction text, else the previous
//! segment's prompt, within the encoder's limit; never the long premise),
//! reported in `segment_expanded` (`fallback: true`, `error`).
//!
//! Vision (`[upsampler] vision`): the first frame and an end frame go to
//! the LLM as `image_url` parts, only in the segment they pin (the opening
//! expansion, the arrival), so every other request keeps the same text
//! prefix for the server's prompt cache.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_upsampler::director::{expand_request, request_for, ExpansionOutput, PlannedOutput};
use fastvideo_upsampler::{Direction, ExpandedSegment, Frames, Kind, UpsampleError, Upsampler};
use tokio::sync::mpsc;

/// What a planned segment is cued on (empty: advance one beat).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cue {
    /// A new direction: `(prompt_version, text)`.
    pub direction: Option<(u64, String)>,
    /// Starts from this image (`configure.image_url`, chunk 0).
    pub first_frame: Option<PathBuf>,
    /// Must end on this image (`end_image_url`, an end-image beat).
    pub end_frame: Option<PathBuf>,
}

impl Cue {
    fn direction(&self) -> Option<Direction> {
        self.direction.as_ref().map(|(v, t)| Direction { id: v.to_string(), text: t.clone() })
    }
}

/// One upcoming segment as the control plans it.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanView {
    pub version: u64,
    pub direction: Option<String>,
    pub first_image: Option<PathBuf>,
    pub end_image: Option<PathBuf>,
}

/// Turns an image file into what the LLM gets (a `data:` URL), or `None`.
pub type ImageLoader = Arc<dyn Fn(&Path) -> Option<String> + Send + Sync>;

/// A finished expansion, from its task to the session loop.
#[derive(Debug)]
pub struct Done {
    seq: u64,
    result: Result<ExpansionOutput, UpsampleError>,
    latency: Duration,
}

struct Inflight {
    seq: u64,
    key: Cue,
    handle: tokio::task::JoinHandle<()>,
}

struct Slot {
    key: Cue,
    result: Result<ExpansionOutput, UpsampleError>,
    latency: Duration,
}

/// A segment's prompt: the upsampler's, or the text it fell back to.
#[derive(Clone, Debug, PartialEq)]
pub struct Taken {
    pub prompt: String,
    /// `None` on a fallback.
    pub segment: Option<ExpandedSegment>,
    pub error: Option<String>,
    pub latency: Duration,
    /// Segments still planned (expanded) after this one.
    pub planned: usize,
}

/// One session's upsampling (module docs).
pub struct SegmentUpsampler {
    up: Upsampler,
    tx: mpsc::UnboundedSender<Done>,
    seq: u64,
    planned: VecDeque<Slot>,
    inflight: Option<Inflight>,
    /// The upcoming cues of the last [`Self::sync`].
    wanted: Vec<Cue>,
    /// The last direction a segment consumed.
    last: Option<(u64, String)>,
    deck: usize,
    images: Option<ImageLoader>,
    /// Calls started (for logs and tests).
    started: u64,
    /// Planned segments discarded unexecuted (busts).
    discarded: u64,
}

impl Drop for SegmentUpsampler {
    fn drop(&mut self) {
        if let Some(f) = self.inflight.take() {
            f.handle.abort();
        }
    }
}

impl SegmentUpsampler {
    /// `tx` feeds the session loop, which hands each [`Done`] back to
    /// [`Self::on_done`]. `deck`: segments planned ahead (at least 1);
    /// `images`: the vision loader (`None`: a text-only LLM).
    pub fn new(up: Upsampler, tx: mpsc::UnboundedSender<Done>, deck: usize, images: Option<ImageLoader>) -> Self {
        Self { up, tx, seq: 0, planned: VecDeque::new(), inflight: None, wanted: Vec::new(), last: None, deck: deck.max(1), images, started: 0, discarded: 0 }
    }

    pub fn upsampler(&self) -> &Upsampler {
        &self.up
    }

    pub fn upsampler_mut(&mut self) -> &mut Upsampler {
        &mut self.up
    }

    pub fn deck_size(&self) -> usize {
        self.deck
    }

    /// LLM calls started so far.
    pub fn started(&self) -> u64 {
        self.started
    }

    /// Planned segments discarded unexecuted.
    pub fn discarded(&self) -> u64 {
        self.discarded
    }

    /// Whether a call is running.
    pub fn busy(&self) -> bool {
        self.inflight.is_some()
    }

    /// Segments expanded and waiting to execute.
    pub fn planned(&self) -> usize {
        self.planned.len()
    }

    /// Planned segments plus the one being expanded: what an appended
    /// direction goes behind.
    pub fn ahead(&self) -> usize {
        self.planned.len() + usize::from(self.inflight.is_some())
    }

    /// The last direction consumed by an executed segment.
    pub fn last_direction(&self) -> Option<&(u64, String)> {
        self.last.as_ref()
    }

    /// The cues of `plans`, in order: a direction is new once (compared
    /// with the one before it, starting from `last`).
    pub fn cues_from(last: Option<&(u64, String)>, plans: &[PlanView]) -> Vec<Cue> {
        let mut last = last.cloned();
        plans
            .iter()
            .map(|p| {
                let d = p.direction.as_deref().map(str::trim).filter(|d| !d.is_empty()).map(|d| (p.version, d.to_owned()));
                let new = d.is_some() && d != last;
                if new {
                    last = d.clone();
                }
                Cue { direction: if new { d } else { None }, first_frame: p.first_image.clone(), end_frame: p.end_image.clone() }
            })
            .collect()
    }

    /// [`Self::cues_from`] the last consumed direction.
    pub fn cues(&self, plans: &[PlanView]) -> Vec<Cue> {
        Self::cues_from(self.last.as_ref(), plans)
    }

    /// The upcoming segments are `wanted` (front: the next to execute).
    /// Keeps the planned prefix that matches, discards the rest (aborting
    /// a call for a discarded position) and starts the next call.
    pub fn sync(&mut self, wanted: Vec<Cue>) {
        let keep = self.planned.iter().zip(&wanted).take_while(|(s, w)| &s.key == *w).count();
        if keep < self.planned.len() {
            self.discarded += (self.planned.len() - keep) as u64;
            self.planned.truncate(keep);
        }
        if let Some(f) = &self.inflight {
            if wanted.get(self.planned.len()) != Some(&f.key) {
                let f = self.inflight.take().expect("inflight");
                f.handle.abort();
                self.discarded += 1;
            }
        }
        self.wanted = wanted;
        self.pump();
    }

    /// Starts the next planned position when idle and below the deck.
    fn pump(&mut self) {
        let pos = self.planned.len();
        if self.inflight.is_some() || pos >= self.deck {
            return;
        }
        let Some(key) = self.wanted.get(pos).cloned() else { return };
        let chain: Vec<PlannedOutput> = self
            .planned
            .iter()
            .filter_map(|s| s.result.as_ref().ok().map(|out| PlannedOutput { direction: s.key.direction(), out: out.clone() }))
            .collect();
        // Images only in the segment they pin.
        let paths: Vec<PathBuf> = key.first_frame.iter().chain(&key.end_frame).cloned().collect();
        let attach = self.images.is_some() && !paths.is_empty();
        let frames = Frames { first: key.first_frame.is_some(), end: key.end_frame.is_some(), attached: attach };
        let user = self.up.user_message_after(&chain, key.direction().as_ref(), frames);
        self.seq += 1;
        self.started += 1;
        let seq = self.seq;
        let backend = self.up.backend().clone();
        let (hedge, timeout) = (self.up.config().hedge, self.up.config().timeout);
        let tx = self.tx.clone();
        let loader = if attach { self.images.clone() } else { None };
        let handle = tokio::spawn(async move {
            let t0 = Instant::now();
            let images = match loader {
                Some(l) => tokio::task::spawn_blocking(move || paths.iter().filter_map(|p| l(p)).collect::<Vec<_>>()).await.unwrap_or_default(),
                None => Vec::new(),
            };
            let result = expand_request(&backend, hedge, timeout, request_for(user, images)).await;
            let _ = tx.send(Done { seq, result, latency: t0.elapsed() });
        });
        self.inflight = Some(Inflight { seq, key, handle });
    }

    /// A finished call: `true` when it is the one awaited (stale ones are
    /// dropped). Starts the next planned position.
    pub fn on_done(&mut self, d: Done) -> bool {
        match &self.inflight {
            Some(f) if f.seq == d.seq => {
                let f = self.inflight.take().expect("inflight");
                self.planned.push_back(Slot { key: f.key, result: d.result, latency: d.latency });
                self.pump();
                true
            }
            _ => false,
        }
    }

    /// Whether the next segment's expansion (for `key`) is ready.
    pub fn ready(&self, key: &Cue) -> bool {
        self.planned.front().is_some_and(|s| &s.key == key)
    }

    /// The next segment, cued `key`, when its expansion is ready: committed
    /// to the ledger, or (the call failed) its fallback. `None`: not ready
    /// yet ([`Self::sync`] it first).
    pub fn take(&mut self, key: &Cue) -> Option<Taken> {
        if !self.ready(key) {
            return None;
        }
        let s = self.planned.pop_front().expect("ready");
        if s.key.direction.is_some() {
            self.last = s.key.direction.clone();
        }
        let dir = s.key.direction();
        let planned = self.planned.len();
        // The wanted list moves with the deck.
        if !self.wanted.is_empty() {
            self.wanted.remove(0);
        }
        let t = match s.result {
            Ok(out) => {
                fastvideo_upsampler::record(Kind::Director, Ok(()), s.latency);
                let seg = self.up.commit(dir.as_ref(), out);
                Taken { prompt: seg.record.prompt.clone(), segment: Some(seg), error: None, latency: s.latency, planned }
            }
            Err(e) => {
                fastvideo_upsampler::record(Kind::Director, Err(&e), s.latency);
                let prompt = self.up.fallback_prompt(dir.as_ref().map(|d| d.text.as_str()));
                Taken { prompt, segment: None, error: Some(e.to_string()), latency: s.latency, planned }
            }
        };
        self.pump();
        Some(t)
    }
}

/// The first and end frames as `data:` URLs for a vision LLM: decoded,
/// scaled to at most `max_side` pixels on the long side, JPEG-encoded,
/// refused (`None`) above `max_bytes`.
pub fn image_data_url(path: &Path, max_side: u32, max_bytes: usize) -> Option<String> {
    use base64::Engine as _;
    let bytes = std::fs::read(path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let img = if img.width().max(img.height()) > max_side { img.resize(max_side, max_side, image::imageops::FilterType::Triangle) } else { img };
    let mut out = std::io::Cursor::new(Vec::new());
    img.to_rgb8().write_to(&mut out, image::ImageFormat::Jpeg).ok()?;
    let out = out.into_inner();
    let url = format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(&out));
    (url.len() <= max_bytes).then_some(url)
}

/// [`image_data_url`] at the director's bounds: 1024 px, 4 MB.
pub fn default_image_loader() -> ImageLoader {
    Arc::new(|p: &Path| {
        let r = image_data_url(p, 1024, 4 * 1024 * 1024);
        if r.is_none() {
            tracing::warn!(path = %p.display(), "director: the image could not be prepared for the upsampler (unreadable or over 4 MB); the LLM is told it exists");
        }
        r
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use fastvideo_upsampler::{ChatBackend, ChatRequest, PromptLimit, UpsamplerConfig, WorldBible};

    /// Answers each call with a segment whose summary names the call, after
    /// `delay`; records the requests.
    struct Llm {
        n: AtomicUsize,
        delay: Duration,
        fail: bool,
        seen: Mutex<Vec<ChatRequest>>,
    }

    #[async_trait::async_trait]
    impl ChatBackend for Llm {
        async fn complete(&self, req: &ChatRequest) -> Result<String, UpsampleError> {
            let n = self.n.fetch_add(1, Ordering::SeqCst);
            self.seen.lock().unwrap().push(req.clone());
            tokio::time::sleep(self.delay).await;
            if self.fail {
                return Err(UpsampleError::Status { status: 503, body: "down".into() });
            }
            Ok(serde_json::json!({
                "beat_summary": format!("beat {n}"), "entity_ids": [], "new_entities": [],
                "action": format!("action {n}."), "camera": "c", "lighting": "l", "audio": "a", "music": "", "dialogue": [],
            })
            .to_string())
        }
    }

    fn setup(delay_ms: u64, fail: bool, deck: usize, images: Option<ImageLoader>) -> (SegmentUpsampler, mpsc::UnboundedReceiver<Done>, Arc<Llm>) {
        let llm = Arc::new(Llm { n: AtomicUsize::new(0), delay: Duration::from_millis(delay_ms), fail, seen: Mutex::new(Vec::new()) });
        let cfg = UpsamplerConfig { memory: 12, limit: PromptLimit::chars(1800), hedge: false, timeout: Duration::from_secs(5), ..UpsamplerConfig::default() };
        let up = Upsampler::new(cfg, WorldBible { premise: "premise".into(), ..WorldBible::default() }, llm.clone());
        let (tx, rx) = mpsc::unbounded_channel();
        (SegmentUpsampler::new(up, tx, deck, images), rx, llm)
    }

    async fn settle(s: &mut SegmentUpsampler, rx: &mut mpsc::UnboundedReceiver<Done>) {
        let d = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap();
        s.on_done(d);
    }

    fn plan(version: u64, d: Option<&str>) -> PlanView {
        PlanView { version, direction: d.map(Into::into), first_image: None, end_image: None }
    }

    fn dir(v: u64, t: &str) -> Cue {
        Cue { direction: Some((v, t.into())), ..Cue::default() }
    }

    #[test]
    fn cues_mark_a_direction_new_once() {
        let plans = [plan(1, None), plan(2, Some("A storm.")), plan(2, Some("A storm.")), plan(3, Some("Calm.")), plan(3, Some("Calm."))];
        let c = SegmentUpsampler::cues_from(None, &plans);
        assert_eq!(c, vec![Cue::default(), dir(2, "A storm."), Cue::default(), dir(3, "Calm."), Cue::default()]);
        assert_eq!(SegmentUpsampler::cues_from(Some(&(2, "A storm.".into())), &plans[1..2]), vec![Cue::default()]);
    }

    /// The deck plans ahead one call at a time up to `deck`, each planned
    /// segment written after the ones before it; executing one commits it
    /// and plans the next.
    #[tokio::test]
    async fn the_deck_plans_ahead_and_only_executed_segments_are_committed() {
        let (mut s, mut rx, llm) = setup(0, false, 3, None);
        s.sync(vec![Cue::default(); 6]);
        assert_eq!(s.started(), 1, "one call at a time");
        for _ in 0..3 {
            settle(&mut s, &mut rx).await;
        }
        assert_eq!((s.planned(), s.started(), s.busy()), (3, 3, false), "stops at the deck size");
        {
            let seen = llm.seen.lock().unwrap();
            assert!(seen[1].user.contains("[0] beat 0\n") && seen[2].user.contains("[1] beat 1\n"), "{}", seen[2].user);
        }
        assert_eq!(s.upsampler().ledger_len(), 0, "planned, not executed");
        let t = s.take(&Cue::default()).unwrap();
        assert_eq!((t.prompt.as_str(), t.planned), ("action 0. Camera: c Lighting: l Sound: a No spoken words.", 2));
        assert_eq!(s.upsampler().ledger_len(), 1);
        assert!(s.busy(), "the next position is planned at once");
    }

    /// `replan: true`: a direction at the next chunk busts the deck (the
    /// in-flight call is aborted, planned segments discarded unexecuted);
    /// `replan: false`: appended after the planned ones, the deck stays.
    #[tokio::test]
    async fn bust_and_append() {
        let (mut s, mut rx, llm) = setup(0, false, 3, None);
        s.sync(vec![Cue::default(); 3]);
        for _ in 0..3 {
            settle(&mut s, &mut rx).await;
        }
        // Append: the direction comes after the three planned beats.
        s.sync(vec![Cue::default(), Cue::default(), Cue::default(), dir(2, "Night.")]);
        assert_eq!((s.planned(), s.discarded(), s.busy()), (3, 0, false), "full deck: the appended one waits");
        let t = s.take(&Cue::default()).unwrap();
        assert_eq!(t.segment.unwrap().record.summary, "beat 0");
        // ...and is planned once there is room, after beats 1 and 2.
        assert!(s.busy());
        settle(&mut s, &mut rx).await;
        assert!(llm.seen.lock().unwrap().last().unwrap().user.contains("[2] beat 2\n"));
        assert!(llm.seen.lock().unwrap().last().unwrap().user.ends_with("NEW DIRECTION:\nNight."));
        // Bust: a direction now; everything planned goes, nothing committed.
        s.sync(vec![dir(3, "Dawn."), Cue::default()]);
        assert_eq!((s.planned(), s.discarded()), (0, 3));
        assert_eq!(s.upsampler().ledger_len(), 1);
        settle(&mut s, &mut rx).await;
        let t = s.take(&dir(3, "Dawn.")).unwrap();
        assert_eq!(t.segment.unwrap().record.direction_id.as_deref(), Some("3"));
        assert_eq!(s.last_direction(), Some(&(3, "Dawn.".into())));
        // The new plan was written on the executed ledger only.
        let seen = llm.seen.lock().unwrap();
        let dawn = seen.iter().find(|r| r.user.ends_with("NEW DIRECTION:\nDawn.")).unwrap();
        assert!(dawn.user.contains("[0] beat 0\n") && !dawn.user.contains("beat 1"), "{}", dawn.user);
    }

    #[tokio::test]
    async fn an_in_flight_speculative_call_is_aborted_by_a_bust() {
        let (mut s, mut rx, _) = setup(200, false, 3, None);
        s.sync(vec![Cue::default()]);
        assert!(s.busy());
        s.sync(vec![dir(3, "Cut to the harbour.")]);
        assert_eq!((s.started(), s.discarded()), (2, 1));
        settle(&mut s, &mut rx).await;
        assert!(s.take(&Cue::default()).is_none());
        assert!(s.take(&dir(3, "Cut to the harbour.")).is_some());
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(rx.try_recv().is_err(), "the aborted call never answered");
    }

    #[tokio::test]
    async fn a_failed_call_falls_back_to_the_direction() {
        let (mut s, mut rx, _) = setup(0, true, 2, None);
        s.sync(vec![dir(2, "Night falls.")]);
        settle(&mut s, &mut rx).await;
        let t = s.take(&dir(2, "Night falls.")).unwrap();
        assert_eq!(t.prompt, "Night falls.");
        assert!(t.segment.is_none());
        assert_eq!(t.error.as_deref(), Some("upsampler LLM 503: down"));
        assert_eq!(s.cues(&[plan(2, Some("Night falls."))]), vec![Cue::default()], "consumed");
        assert_eq!(s.upsampler().ledger_len(), 0);
    }

    /// Vision: the first frame goes as an image only with the segment it
    /// pins; the others carry no image (stable prompt-cache prefix).
    #[tokio::test]
    async fn images_only_with_the_segment_they_pin() {
        let loader: ImageLoader = Arc::new(|p: &Path| Some(format!("data:image/jpeg;base64,{}", p.display())));
        let (mut s, mut rx, llm) = setup(0, false, 3, Some(loader));
        let first = Cue { first_frame: Some("first.png".into()), ..Cue::default() };
        let end = Cue { end_frame: Some("end.png".into()), ..Cue::default() };
        s.sync(vec![first, Cue::default(), end]);
        for _ in 0..3 {
            settle(&mut s, &mut rx).await;
        }
        let seen = llm.seen.lock().unwrap();
        assert_eq!(seen[0].images, vec!["data:image/jpeg;base64,first.png".to_owned()]);
        assert!(seen[0].user.contains("FIRST FRAME: this segment starts from a given image (attached)"));
        assert!(seen[1].images.is_empty() && !seen[1].user.contains("FRAME:"));
        assert_eq!(seen[2].images, vec!["data:image/jpeg;base64,end.png".to_owned()]);
        assert!(seen[2].user.contains("FINAL FRAME: this segment must end exactly on a given image (attached)"));
    }

    #[test]
    fn data_urls_are_scaled_jpegs() {
        let dir = std::env::temp_dir().join(format!("fv-ups-img-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("big.png");
        image::RgbImage::from_pixel(3000, 1500, image::Rgb([200, 10, 10])).save(&p).unwrap();
        let url = image_data_url(&p, 1024, 4 << 20).unwrap();
        let b64 = url.strip_prefix("data:image/jpeg;base64,").unwrap();
        use base64::Engine as _;
        let img = image::load_from_memory(&base64::engine::general_purpose::STANDARD.decode(b64).unwrap()).unwrap();
        assert_eq!((img.width(), img.height()), (1024, 512));
        assert!(image_data_url(&p, 1024, 100).is_none(), "over the cap: refused");
        assert!(image_data_url(&dir.join("missing.png"), 1024, 4 << 20).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
