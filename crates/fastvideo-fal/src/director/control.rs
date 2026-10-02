//! The director's control state machine (design §5.6; fal §8.3-8.5).
//!
//! Pure and synchronous: it turns parsed client messages into server
//! messages and a **deck** of planned chunks, and hands out the next chunk
//! to build when the session asks. The session (`session.rs`) owns the
//! transport, the engine and playout, stages image URLs to files before it
//! calls in here, and adds the last-frame anchor to each continuation.
//!
//! Rules implemented here:
//!
//! - `configure` exactly once, with `prompt_version: 1`; a second one is
//!   `error{code:"immutable_settings"}`. `resolution` must be served
//!   (`1080p` without the H3 1080P tier → `invalid_input`), `audio_url` → `invalid_initial_audio`
//!   (target audio arrives with E10), script beats with audio →
//!   `invalid_initial_script`; these are session failures.
//! - `prompt` before `configured` → `error{code:"not_configured"}`.
//! - Versions: a `prompt` whose version is ≤ the last one seen gets
//!   `prompt_rejected{reason:"stale_prompt_version"}`; gaps are allowed.
//! - `replan:true` (default) replaces every planned-but-undispatched chunk
//!   (policy `replace-pending`: the replaced versions get no final event);
//!   `replan:false` appends, and a full deck (`prompt_deck_size`) answers
//!   `queue_full`. `prompt_applied` is emitted when the chunk carrying that
//!   version is dispatched.
//! - Scripts (upfront or `prompt.script`) become planned chunks cut at the
//!   beat offsets: an end-image beat ends a chunk exactly there (5-15 s
//!   chunks, end images ≥ 3 s apart, else `infeasible_timing` /
//!   `invalid_initial_script`); a text beat directs from the first chunk
//!   that starts at or after its offset.
//!
//! We have no prompt expander: a chunk's prompt is the series premise
//! followed by the current direction (the latest text update).

use std::collections::VecDeque;
use std::path::PathBuf;

use serde_json::{json, Value};

use super::messages::{self as m, Aspect, Configure, ErrorCode, Prompt, RejectReason, Resolution, ScriptBeat, ScriptMode};

/// How refusals name a causal model.
const CAUSAL_MODEL: &str = "this causal streaming model (LongLive / SF-Wan)";

/// Server-side limits and defaults (our `session_info` constants).
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    /// Model frame rate (24 for H3).
    pub fps: u32,
    /// Default chunk duration (`default_chunk_duration`, 10 s).
    pub chunk_seconds: f64,
    pub min_chunk_seconds: f64,
    pub max_chunk_seconds: f64,
    /// Planned-chunk deck (`prompt_deck_size`).
    pub deck_size: usize,
    /// Scripts queued behind the running one (`script_max_queued`).
    pub script_max_queued: usize,
    pub script_max_end_images: usize,
    pub end_image_spacing_seconds: f64,
    /// Resolutions this model serves (`resolutions`).
    pub resolutions: Vec<Resolution>,
    pub default_memory: u32,
    /// The longest chunk at 1080p (the H3 1080P tier's clip cap: 5 s, or
    /// 10 s with the `h3_1080p_long` experimental flag). `None`: no cap.
    pub hd_max_chunk_seconds: Option<f64>,
    /// A causal model (one continuous rollout; docs/serve/director-causal.md).
    pub causal: Option<CausalLimits>,
}

/// The block structure of a causal director session.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalLimits {
    /// Pixel frames per block (12 for SF-Wan / LongLive).
    pub block_frames: u32,
    /// `block_frames / fps` (0.75 s).
    pub block_seconds: f64,
    /// Blocks per director chunk: one `chunk` message, and how long an
    /// appended (`replan:false`) direction holds.
    pub chunk_blocks: u32,
    /// The model's KV window and prompt-switch policy, when known.
    pub context: Option<fastvideo_protocol::CausalContext>,
}

impl Limits {
    /// The limits a session configured at `res` runs with: at 1080p the
    /// chunk lengths are capped by [`Limits::hd_max_chunk_seconds`].
    pub fn at(&self, res: Resolution) -> Limits {
        let mut l = self.clone();
        if let (Resolution::R1080, Some(cap)) = (res, self.hd_max_chunk_seconds) {
            l.max_chunk_seconds = l.max_chunk_seconds.min(cap);
            l.min_chunk_seconds = l.min_chunk_seconds.min(l.max_chunk_seconds);
            l.chunk_seconds = l.chunk_seconds.clamp(l.min_chunk_seconds, l.max_chunk_seconds);
        }
        l
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            fps: 24,
            chunk_seconds: 10.0,
            min_chunk_seconds: 5.0,
            max_chunk_seconds: 15.0,
            deck_size: 6,
            script_max_queued: 4,
            script_max_end_images: 16,
            end_image_spacing_seconds: 3.0,
            resolutions: vec![Resolution::R768],
            default_memory: 12,
            hd_max_chunk_seconds: None,
            causal: None,
        }
    }
}

/// What `configure` settled (immutable for the session).
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub resolution: Resolution,
    pub aspect: Aspect,
    pub memory: u32,
    pub audio_bitrate: Option<u32>,
    pub seed: Option<u64>,
    pub has_initial_image: bool,
}

/// A `prompt_applied` owed when the chunk carrying it is dispatched.
#[derive(Clone, Debug, PartialEq)]
struct Announce {
    version: u64,
    /// `(script_origin_chunk_index, script_queued, script_beats, script_mode)`.
    script: Option<(u32, u32, u32, ScriptMode)>,
}

/// One planned, undispatched chunk.
#[derive(Clone, Debug, PartialEq)]
struct Planned {
    /// Direction for this chunk (`None`: the current direction).
    direction: Option<String>,
    end_image: Option<PathBuf>,
    seconds: f64,
    /// The version whose direction this chunk renders.
    version: u64,
    announce: Vec<Announce>,
    /// Direction that becomes current once this chunk is dispatched.
    persist: Option<String>,
    /// Which script (group id) planned this chunk.
    script: Option<u64>,
    /// Causal sessions: blocks this entry directs (1 on clip models).
    blocks: u32,
}

/// The chunk the session should build next.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkPlan {
    pub index: u32,
    pub prompt: String,
    pub seconds: f64,
    /// `configure.image_url` for chunk 0 (exact first frame).
    pub first_image: Option<PathBuf>,
    /// Exact last frame (`end_image_url`, end-image script beats).
    pub end_image: Option<PathBuf>,
    pub prompt_version: u64,
}

/// A script chunk cut: its length, the text beat active at its start and
/// the end-image beat it lands on.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub seconds: f64,
    pub text: Option<usize>,
    pub end_image: Option<usize>,
    /// Causal blocks this segment covers (1 on clip models).
    pub blocks: u32,
}

/// Cuts a script into chunks (see the module docs). `Err` explains why the
/// timing is infeasible.
pub fn plan_script(beats: &[ScriptBeat], l: &Limits) -> Result<(Vec<Segment>, Option<usize>), String> {
    if let Some(c) = &l.causal {
        return Ok(plan_script_blocks(beats, c));
    }
    let mut order: Vec<usize> = (0..beats.len()).collect();
    order.sort_by_key(|&i| beats[i].offset);
    let ends: Vec<usize> = order.iter().copied().filter(|&i| beats[i].end_image_url.is_some()).collect();
    if ends.len() > l.script_max_end_images {
        return Err(format!("at most {} end images per script", l.script_max_end_images));
    }
    for w in ends.windows(2) {
        let (a, b) = (beats[w[0]].offset as f64, beats[w[1]].offset as f64);
        if b - a < l.end_image_spacing_seconds {
            return Err(format!(
                "end images at {a}s and {b}s are closer than {}s",
                l.end_image_spacing_seconds
            ));
        }
    }
    let text_at = |t: f64| -> Option<usize> {
        order.iter().copied().rfind(|&i| beats[i].prompt.is_some() && beats[i].offset as f64 <= t + 1e-9)
    };
    let mut segs = Vec::new();
    let mut pos = 0.0f64;
    let mut boundaries: Vec<(f64, Option<usize>)> = Vec::new();
    for &i in &order {
        let t = beats[i].offset as f64;
        let e = beats[i].end_image_url.is_some().then_some(i);
        match boundaries.last_mut() {
            Some((bt, be)) if (*bt - t).abs() < 1e-9 => {
                if e.is_some() {
                    *be = e;
                }
            }
            _ => boundaries.push((t, e)),
        }
    }
    for (b, end) in boundaries {
        if b <= pos + 1e-9 {
            if end.is_some() {
                return Err(format!("an end image at {b}s cannot end a chunk (a chunk is at least {}s)", l.min_chunk_seconds));
            }
            continue;
        }
        let mut gap = b - pos;
        if gap < l.min_chunk_seconds {
            if end.is_some() {
                return Err(format!(
                    "the end image at {b}s leaves a {gap}s chunk; chunks are {}..{}s",
                    l.min_chunk_seconds, l.max_chunk_seconds
                ));
            }
            // A text beat inside the next chunk directs from the one after.
            continue;
        }
        while gap > l.max_chunk_seconds {
            let cut = l.chunk_seconds.min(gap - l.min_chunk_seconds).max(l.min_chunk_seconds);
            segs.push(Segment { seconds: cut, text: text_at(pos), end_image: None, blocks: 1 });
            pos += cut;
            gap -= cut;
        }
        segs.push(Segment { seconds: gap, text: text_at(pos), end_image: end, blocks: 1 });
        pos = b;
    }
    if segs.is_empty() {
        segs.push(Segment { seconds: l.chunk_seconds, text: text_at(0.0), end_image: None, blocks: 1 });
    }
    // The direction that stays current after the script: its last text beat.
    let last_text = order.iter().copied().rfind(|&i| beats[i].prompt.is_some());
    Ok((segs, last_text))
}

/// A text-only script on a causal model's block clock: the beat at offset
/// `t` s switches at block `round(t / block_seconds)` after the script's
/// start (about one block of tolerance); the last direction holds one block
/// and then stays current. End-image beats are refused before this.
pub fn plan_script_blocks(beats: &[ScriptBeat], c: &CausalLimits) -> (Vec<Segment>, Option<usize>) {
    let bs = c.block_seconds.max(1e-3);
    let mut order: Vec<usize> = (0..beats.len()).collect();
    order.sort_by_key(|&i| beats[i].offset);
    let text_at = |t: u64| -> Option<usize> { order.iter().copied().rfind(|&i| beats[i].prompt.is_some() && beats[i].offset <= t) };
    let mut points: Vec<u64> = std::iter::once(0).chain(order.iter().map(|&i| beats[i].offset)).collect();
    points.dedup();
    let block = |t: u64| (t as f64 / bs).round() as u64;
    let mut segs = Vec::new();
    for w in points.windows(2) {
        let n = block(w[1]).saturating_sub(block(w[0]));
        if n == 0 {
            continue;
        }
        let n = n.min(u64::from(u32::MAX)) as u32;
        segs.push(Segment { seconds: f64::from(n) * bs, text: text_at(w[0]), end_image: None, blocks: n });
    }
    let last = *points.last().unwrap_or(&0);
    segs.push(Segment { seconds: bs, text: text_at(last), end_image: None, blocks: 1 });
    (segs, order.iter().copied().rfind(|&i| beats[i].prompt.is_some()))
}

/// The control state of one director session.
#[derive(Debug)]
pub struct Control {
    limits: Limits,
    settings: Option<Settings>,
    premise: String,
    direction: Option<String>,
    /// Version of the current direction (1 after configure).
    current_version: u64,
    last_version: u64,
    deck: VecDeque<Planned>,
    next_index: u32,
    initial_image: Option<PathBuf>,
    stopped: bool,
    next_script: u64,
}

impl Control {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            settings: None,
            premise: String::new(),
            direction: None,
            current_version: 0,
            last_version: 0,
            deck: VecDeque::new(),
            next_index: 0,
            initial_image: None,
            stopped: false,
            next_script: 1,
        }
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// `configure.resolution` default: `768p`, else the highest served.
    pub fn default_resolution(&self) -> Resolution {
        if self.limits.resolutions.contains(&Resolution::R768) {
            Resolution::R768
        } else {
            self.limits.resolutions.last().copied().unwrap_or(Resolution::R768)
        }
    }

    pub fn settings(&self) -> Option<&Settings> {
        self.settings.as_ref()
    }

    pub fn is_configured(&self) -> bool {
        self.settings.is_some()
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Chunks handed out so far.
    pub fn dispatched(&self) -> u32 {
        self.next_index
    }

    /// Planned, undispatched chunks.
    pub fn deck_len(&self) -> usize {
        self.deck.len()
    }

    pub fn last_version(&self) -> u64 {
        self.last_version
    }

    /// Checks a `configure` before its images are staged. `Err` carries the
    /// reply; `.1` says whether it is a session failure (end the session).
    pub fn precheck_configure(&self, c: &Configure) -> Result<(), (Value, bool)> {
        if self.settings.is_some() {
            let e = m::error(ErrorCode::ImmutableSettings, "the session is already configured; reconnect to change settings", Some(c.prompt_version));
            return Err((e, false));
        }
        let fail = |code: ErrorCode, msg: String| Err((m::error(code, msg, Some(c.prompt_version)), true));
        if c.prompt_version != 1 {
            return fail(ErrorCode::InvalidInput, "`configure` starts the version sequence at prompt_version 1".into());
        }
        let res = c.resolution.unwrap_or(self.default_resolution());
        if !self.limits.resolutions.contains(&res) {
            let served: Vec<&str> = self.limits.resolutions.iter().map(|r| r.as_str()).collect();
            return fail(ErrorCode::InvalidInput, format!("resolution {} is not served here (served: {})", res.as_str(), served.join(", ")));
        }
        if c.audio_url.is_some() {
            return fail(ErrorCode::InvalidInitialAudio, "target-audio conditioning (`audio_url`) is not supported by this server yet".into());
        }
        if self.limits.causal.is_some() {
            if c.image_url.is_some() {
                return fail(ErrorCode::InvalidInitialImage, format!("{CAUSAL_MODEL} is text-to-video only: `image_url` (image conditioning) is not supported"));
            }
            if c.end_image_url.is_some() {
                return fail(ErrorCode::InvalidInitialImage, format!("{CAUSAL_MODEL} is text-to-video only: `end_image_url` is not supported"));
            }
            if let Some(a) = c.aspect_ratio.filter(|a| *a != Aspect::Landscape) {
                return fail(ErrorCode::InvalidInput, format!("aspect ratio {} is not served: {CAUSAL_MODEL} generates its 16:9 canvas only", a.as_str()));
            }
            if c.script.iter().flatten().any(|b| b.end_image_url.is_some()) {
                return fail(ErrorCode::InvalidInitialScript, format!("script end images are not supported: {CAUSAL_MODEL} is text-to-video only"));
            }
        }
        if let Some(script) = &c.script {
            if c.end_image_url.is_some() || c.audio_url.is_some() {
                return fail(ErrorCode::InvalidInitialScript, "`script` cannot be combined with `end_image_url` or `audio_url`; place them in the script".into());
            }
            if script.iter().any(|b| b.audio_url.is_some()) {
                return fail(ErrorCode::InvalidInitialScript, "script audio beats are not supported by this server yet".into());
            }
            if let Err(e) = plan_script(script, &self.limits.at(res)) {
                return fail(ErrorCode::InvalidInitialScript, e);
            }
        }
        Ok(())
    }

    /// Applies a prechecked `configure` with its staged images
    /// (`script_images[i]` for beat `i`). Returns `configured` and plans chunk 0.
    pub fn configure(&mut self, c: &Configure, image: Option<PathBuf>, end_image: Option<PathBuf>, script_images: Vec<Option<PathBuf>>) -> Value {
        let settings = Settings {
            resolution: c.resolution.unwrap_or(self.default_resolution()),
            aspect: c.aspect_ratio.unwrap_or(Aspect::Landscape),
            memory: c.memory.unwrap_or(self.limits.default_memory),
            audio_bitrate: c.audio_bitrate,
            seed: c.seed.map(|s| s as u64),
            has_initial_image: image.is_some(),
        };
        // 1080p: chunks within the tier's clip cap.
        self.limits = self.limits.at(settings.resolution);
        self.premise = c.prompt.clone();
        self.current_version = c.prompt_version;
        self.last_version = c.prompt_version;
        self.initial_image = image;
        let reply = json!({
            "type": "configured",
            "prompt_version": c.prompt_version,
            "enable_safety_checker": false,
            "aspect_ratio": settings.aspect.as_str(),
            "memory": settings.memory,
            "chunk_duration": self.limits.chunk_seconds.round() as u64,
            "audio_bitrate": settings.audio_bitrate,
            "resolution": settings.resolution.as_str(),
            "has_initial_image": settings.has_initial_image,
            "has_initial_audio": false,
            "acceleration": null,
        });
        self.settings = Some(settings);
        if let Some(script) = &c.script {
            let entries = self.script_entries(script, script_images, c.prompt_version, ScriptMode::Replace, 0);
            self.deck.extend(entries);
        } else if end_image.is_some() {
            self.deck.push_back(Planned {
                direction: None,
                end_image,
                seconds: self.limits.chunk_seconds,
                version: c.prompt_version,
                announce: Vec::new(),
                persist: None,
                script: None,
                blocks: 1,
            });
        }
        reply
    }

    fn script_entries(&mut self, script: &[ScriptBeat], images: Vec<Option<PathBuf>>, version: u64, mode: ScriptMode, queued: u32) -> Vec<Planned> {
        let (segs, last_text) = plan_script(script, &self.limits).unwrap_or_default();
        let group = self.next_script;
        self.next_script += 1;
        let origin = self.next_index + self.deck.len() as u32;
        let beats = script.len() as u32;
        let n = segs.len();
        segs.into_iter()
            .enumerate()
            .map(|(k, s)| Planned {
                direction: s.text.and_then(|i| script[i].prompt.clone()),
                end_image: s.end_image.and_then(|i| images.get(i).cloned().flatten()),
                seconds: s.seconds,
                version,
                announce: if k == 0 {
                    vec![Announce { version, script: Some((origin, queued, beats, mode)) }]
                } else {
                    Vec::new()
                },
                persist: if k + 1 == n { last_text.and_then(|i| script[i].prompt.clone()) } else { None },
                script: Some(group),
                blocks: s.blocks.max(1),
            })
            .collect()
    }

    /// Checks a `prompt` before its images are staged: not configured,
    /// stale versions and refusals that need no staging. `Err` is the reply.
    pub fn precheck_prompt(&mut self, p: &Prompt) -> Result<(), Vec<Value>> {
        if self.settings.is_none() {
            return Err(vec![m::error(ErrorCode::NotConfigured, "send `configure` and wait for `configured` first", Some(p.prompt_version))]);
        }
        let v = p.prompt_version;
        if v <= self.last_version {
            return Err(vec![m::prompt_rejected(
                v,
                RejectReason::StalePromptVersion,
                format!("prompt_version {v} is not newer than {}", self.last_version),
            )]);
        }
        // A version is spent once seen, whatever its outcome.
        self.last_version = v;
        if p.audio_url.is_some() || p.audio_behavior.is_some() {
            return Err(vec![m::prompt_rejected(v, RejectReason::InvalidAudio, "target-audio conditioning is not supported by this server yet")]);
        }
        if self.limits.causal.is_some()
            && (p.end_image_url.is_some() || p.script.iter().flatten().any(|b| b.end_image_url.is_some()))
        {
            return Err(vec![m::prompt_rejected(v, RejectReason::InvalidImage, format!("end images are not supported: {CAUSAL_MODEL} is text-to-video only"))]);
        }
        if let Some(script) = &p.script {
            if p.prompt.is_some() || p.end_image_url.is_some() {
                return Err(vec![m::prompt_rejected(v, RejectReason::InvalidScript, "`script` is exclusive with `prompt`, `end_image_url` and `audio_url`")]);
            }
            if script.iter().any(|b| b.audio_url.is_some()) {
                return Err(vec![m::prompt_rejected(v, RejectReason::InvalidAudio, "script audio beats are not supported by this server yet")]);
            }
            if let Err(e) = plan_script(script, &self.limits) {
                return Err(vec![m::prompt_rejected(v, RejectReason::InfeasibleTiming, e)]);
            }
        } else if p.prompt.is_none() && p.end_image_url.is_none() {
            return Err(vec![m::prompt_rejected(v, RejectReason::PreparationFailed, "the update carries no prompt, end image or script")]);
        }
        let append = match &p.script {
            Some(_) => p.script_mode == Some(ScriptMode::Append),
            None => p.replan == Some(false),
        };
        if append {
            if self.deck.len() >= self.limits.deck_size {
                return Err(vec![m::prompt_rejected(v, RejectReason::QueueFull, format!("the planned deck is full ({} chunks)", self.limits.deck_size))]);
            }
            if p.script.is_some() && self.queued_scripts() >= self.limits.script_max_queued {
                return Err(vec![m::prompt_rejected(v, RejectReason::QueueFull, format!("at most {} scripts may be queued", self.limits.script_max_queued))]);
            }
        }
        Ok(())
    }

    fn queued_scripts(&self) -> usize {
        let mut g: Vec<u64> = self.deck.iter().filter_map(|p| p.script).collect();
        g.dedup();
        g.len()
    }

    /// Applies a prechecked `prompt` with its staged images: `prompt_pending`,
    /// then the deck change.
    pub fn prompt(&mut self, p: &Prompt, end_image: Option<PathBuf>, script_images: Vec<Option<PathBuf>>) -> Vec<Value> {
        let v = p.prompt_version;
        let out = vec![m::prompt_pending(v)];
        if let Some(script) = &p.script {
            let mode = p.script_mode.unwrap_or(ScriptMode::Replace);
            if mode == ScriptMode::Replace {
                self.deck.clear();
            }
            let queued = self.queued_scripts() as u32;
            let entries = self.script_entries(script, script_images, v, mode, queued);
            self.deck.extend(entries);
            return out;
        }
        let entry = Planned {
            direction: p.prompt.clone(),
            end_image,
            seconds: self.limits.chunk_seconds,
            version: v,
            announce: vec![Announce { version: v, script: None }],
            persist: p.prompt.clone(),
            script: None,
            // Causal: an appended direction holds one director chunk.
            blocks: self.limits.causal.as_ref().map_or(1, |c| c.chunk_blocks.max(1)),
        };
        if p.replan != Some(false) {
            // replace-pending: undispatched plans (and their versions) go.
            self.deck.clear();
        }
        self.deck.push_back(entry);
        out
    }

    /// `stop`: nothing new is dispatched. Returns `false` if already stopped.
    pub fn stop(&mut self) -> bool {
        !std::mem::replace(&mut self.stopped, true)
    }

    fn compose(&self, direction: Option<&str>) -> String {
        match direction {
            Some(d) if !d.is_empty() && d != self.premise => format!("{} {}", self.premise, d),
            _ => self.premise.clone(),
        }
    }

    /// The next chunk to build, with the `prompt_applied` messages owed at
    /// its dispatch. `None` before `configure` and after `stop`.
    pub fn next_chunk(&mut self) -> Option<(ChunkPlan, Vec<Value>)> {
        if self.settings.is_none() || self.stopped {
            return None;
        }
        let index = self.next_index;
        self.next_index += 1;
        let first_image = if index == 0 { self.initial_image.take() } else { None };
        // A causal entry directs several blocks: hand out one block's share,
        // announcing it once and persisting its direction with the last.
        let next = match self.deck.front_mut() {
            Some(f) if f.blocks > 1 => {
                let mut share = f.clone();
                share.blocks = 1;
                share.persist = None;
                f.blocks -= 1;
                f.announce.clear();
                Some(share)
            }
            _ => self.deck.pop_front(),
        };
        let (plan, announce) = match next {
            Some(p) => {
                if p.direction.is_some() || p.version > self.current_version {
                    self.current_version = self.current_version.max(p.version);
                }
                let direction = p.direction.clone().or_else(|| self.direction.clone());
                if let Some(d) = &p.persist {
                    self.direction = Some(d.clone());
                } else if p.script.is_none() {
                    if let Some(d) = &p.direction {
                        self.direction = Some(d.clone());
                    }
                }
                let plan = ChunkPlan {
                    index,
                    prompt: self.compose(direction.as_deref()),
                    seconds: p.seconds,
                    first_image,
                    end_image: p.end_image,
                    prompt_version: p.version,
                };
                (plan, p.announce)
            }
            None => (
                ChunkPlan {
                    index,
                    prompt: self.compose(self.direction.clone().as_deref()),
                    seconds: self.limits.chunk_seconds,
                    first_image,
                    end_image: None,
                    prompt_version: self.current_version,
                },
                Vec::new(),
            ),
        };
        let msgs = announce.into_iter().map(|a| m::prompt_applied(a.version, a.script)).collect();
        Some((plan, msgs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::director::messages::{parse, ClientMessage};

    fn configure(c: &mut Control, s: &str) -> Result<Value, (Value, bool)> {
        let ClientMessage::Configure(cfg) = parse(s).unwrap() else { panic!() };
        c.precheck_configure(&cfg)?;
        let n = cfg.script.as_ref().map_or(0, Vec::len);
        let end = cfg.end_image_url.as_ref().map(PathBuf::from);
        let img = cfg.image_url.as_ref().map(PathBuf::from);
        let script_imgs = cfg.script.iter().flatten().map(|b| b.end_image_url.as_ref().map(PathBuf::from)).collect();
        let _ = n;
        Ok(c.configure(&cfg, img, end, script_imgs))
    }

    fn prompt(c: &mut Control, s: &str) -> Vec<Value> {
        let ClientMessage::Prompt(p) = parse(s).unwrap() else { panic!() };
        if let Err(out) = c.precheck_prompt(&p) {
            return out;
        }
        let end = p.end_image_url.as_ref().map(PathBuf::from);
        let imgs = p.script.iter().flatten().map(|b| b.end_image_url.as_ref().map(PathBuf::from)).collect();
        c.prompt(&p, end, imgs)
    }

    fn types(v: &[Value]) -> Vec<String> {
        v.iter().map(|m| m["type"].as_str().unwrap().to_owned()).collect()
    }

    #[test]
    fn configure_once_and_validate() {
        let mut c = Control::new(Limits::default());
        assert!(c.next_chunk().is_none(), "nothing before configure");
        // Refusals are session failures with their codes.
        let e = configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"x","resolution":"1080p"}"#).unwrap_err();
        assert_eq!((e.0["code"].as_str(), e.1), (Some("invalid_input"), true));
        let e = configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"x","resolution":"480p"}"#).unwrap_err();
        assert_eq!(e.0["code"], "invalid_input");
        let e = configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"x","resolution":"720p"}"#).unwrap_err();
        assert_eq!(e.0["code"], "invalid_input");
        assert!(e.0["error"].as_str().unwrap().contains("served: 768p"), "{}", e.0);
        let e = configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"x","audio_url":"https://a/b.wav"}"#).unwrap_err();
        assert_eq!(e.0["code"], "invalid_initial_audio");
        let e = configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"x","script":[{"offset":0,"audio_url":"https://a"}]}"#).unwrap_err();
        assert_eq!(e.0["code"], "invalid_initial_script");
        let e = configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"x","end_image_url":"e","script":[{"offset":0,"prompt":"p"}]}"#).unwrap_err();
        assert_eq!(e.0["code"], "invalid_initial_script");
        let e = configure(&mut c, r#"{"type":"configure","prompt_version":2,"prompt":"x"}"#).unwrap_err();
        assert_eq!(e.0["code"], "invalid_input");
        // Prompts before configure are diagnostics.
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":2,"prompt":"y"}"#);
        assert_eq!(out[0]["code"], "not_configured");

        let r = configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"A sitcom","memory":3,"image_url":"first.png"}"#).unwrap();
        assert_eq!(
            r,
            json!({"type":"configured","prompt_version":1,"enable_safety_checker":false,"aspect_ratio":"16:9","memory":3,
                   "chunk_duration":10,"audio_bitrate":null,"resolution":"768p","has_initial_image":true,
                   "has_initial_audio":false,"acceleration":null})
        );
        let e = configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"again"}"#).unwrap_err();
        assert_eq!((e.0["code"].as_str(), e.1), (Some("immutable_settings"), false));
        let (p0, applied) = c.next_chunk().unwrap();
        assert!(applied.is_empty());
        assert_eq!((p0.index, p0.prompt.as_str(), p0.seconds, p0.prompt_version), (0, "A sitcom", 10.0, 1));
        assert_eq!(p0.first_image, Some(PathBuf::from("first.png")));
        let (p1, _) = c.next_chunk().unwrap();
        assert_eq!((p1.index, p1.first_image.clone()), (1, None));
    }

    #[test]
    fn versions_replan_and_deck() {
        let mut c = Control::new(Limits::default());
        configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"P"}"#).unwrap();
        let (_, _) = c.next_chunk().unwrap(); // chunk 0 dispatched
        // Stale and equal versions.
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":1,"prompt":"old"}"#);
        assert_eq!((out[0]["type"].as_str(), out[0]["reason"].as_str()), (Some("prompt_rejected"), Some("stale_prompt_version")));
        // Pending, replaced by a newer replan (no final event for v2), applied at dispatch.
        assert_eq!(types(&prompt(&mut c, r#"{"type":"prompt","prompt_version":2,"prompt":"A"}"#)), ["prompt_pending"]);
        assert_eq!(types(&prompt(&mut c, r#"{"type":"prompt","prompt_version":5,"prompt":"B"}"#)), ["prompt_pending"]);
        assert_eq!(c.deck_len(), 1);
        // Gaps are fine but reuse is not.
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":4,"prompt":"C"}"#);
        assert_eq!(out[0]["reason"], "stale_prompt_version");
        let (p, applied) = c.next_chunk().unwrap();
        assert_eq!((p.prompt.as_str(), p.prompt_version), ("P B", 5));
        assert_eq!(applied, vec![json!({"type":"prompt_applied","prompt_version":5})]);
        // The direction persists for continuations.
        let (p, applied) = c.next_chunk().unwrap();
        assert_eq!((p.prompt.as_str(), p.prompt_version, applied.len()), ("P B", 5, 0));
        // replan:false appends; the deck is bounded.
        for v in 6..12u64 {
            let out = prompt(&mut c, &format!(r#"{{"type":"prompt","prompt_version":{v},"prompt":"s{v}","replan":false}}"#));
            assert_eq!(types(&out), ["prompt_pending"], "v{v}");
        }
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":12,"prompt":"full","replan":false}"#);
        assert_eq!(out[0]["reason"], "queue_full");
        let order: Vec<u64> = (0..6).map(|_| c.next_chunk().unwrap().0.prompt_version).collect();
        assert_eq!(order, [6, 7, 8, 9, 10, 11]);
        // Audio and empty updates.
        assert_eq!(prompt(&mut c, r#"{"type":"prompt","prompt_version":20,"audio_url":"a.wav"}"#)[0]["reason"], "invalid_audio");
        assert_eq!(prompt(&mut c, r#"{"type":"prompt","prompt_version":21}"#)[0]["reason"], "preparation_failed");
        // End image only: keeps the direction, lands on the next chunk.
        prompt(&mut c, r#"{"type":"prompt","prompt_version":22,"end_image_url":"end.png"}"#);
        let (p, applied) = c.next_chunk().unwrap();
        assert_eq!((p.end_image, p.prompt.as_str()), (Some(PathBuf::from("end.png")), "P s11"));
        assert_eq!(applied[0]["prompt_version"], 22);
        // Stop ends dispatching.
        assert!(c.stop());
        assert!(!c.stop());
        assert!(c.next_chunk().is_none());
    }

    fn causal_limits() -> Limits {
        Limits {
            fps: 16,
            chunk_seconds: 3.0,
            min_chunk_seconds: 3.0,
            max_chunk_seconds: 3.0,
            resolutions: vec![Resolution::R480],
            script_max_end_images: 0,
            causal: Some(CausalLimits {
                block_frames: 12,
                block_seconds: 0.75,
                chunk_blocks: 4,
                context: Some(fastvideo_protocol::CausalContext { window_latent_frames: 12, sink_latent_frames: 3, prompt_recache: true }),
            }),
            ..Limits::default()
        }
    }

    /// A causal model refuses image, end-image and audio conditioning,
    /// other aspects and resolutions, with clear codes.
    #[test]
    fn causal_refusals() {
        let mut c = Control::new(causal_limits());
        assert_eq!(c.default_resolution(), Resolution::R480);
        let refused = [
            (r#"{"type":"configure","prompt_version":1,"prompt":"x","image_url":"https://a/b.png"}"#, "invalid_initial_image", "image_url"),
            (r#"{"type":"configure","prompt_version":1,"prompt":"x","end_image_url":"https://a/b.png"}"#, "invalid_initial_image", "end_image_url"),
            (r#"{"type":"configure","prompt_version":1,"prompt":"x","audio_url":"https://a/b.wav"}"#, "invalid_initial_audio", "audio"),
            (r#"{"type":"configure","prompt_version":1,"prompt":"x","aspect_ratio":"9:16"}"#, "invalid_input", "16:9"),
            (r#"{"type":"configure","prompt_version":1,"prompt":"x","aspect_ratio":"1:1"}"#, "invalid_input", "16:9"),
            (r#"{"type":"configure","prompt_version":1,"prompt":"x","resolution":"768p"}"#, "invalid_input", "served: 480p"),
            (r#"{"type":"configure","prompt_version":1,"prompt":"x","script":[{"offset":4,"end_image_url":"e"}]}"#, "invalid_initial_script", "end images"),
        ];
        for (msg, code, needle) in refused {
            let e = configure(&mut c, msg).unwrap_err();
            assert_eq!((e.0["code"].as_str(), e.1), (Some(code), true), "{msg}");
            assert!(e.0["error"].as_str().unwrap().contains(needle), "{}", e.0);
        }
        configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"P","aspect_ratio":"16:9","resolution":"480p"}"#).unwrap();
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":2,"end_image_url":"e.png"}"#);
        assert_eq!((out[0]["type"].as_str(), out[0]["reason"].as_str()), (Some("prompt_rejected"), Some("invalid_image")));
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":3,"script":[{"offset":3,"end_image_url":"e"}]}"#);
        assert_eq!(out[0]["reason"], "invalid_image");
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":4,"audio_url":"a.wav"}"#);
        assert_eq!(out[0]["reason"], "invalid_audio");
    }

    /// Causal: one plan per block; a replanning prompt takes the next block,
    /// an appended one holds a director chunk (4 blocks), the direction
    /// persists; scripts switch on the block clock.
    #[test]
    fn causal_blocks_follow_prompts_and_scripts() {
        let mut c = Control::new(causal_limits());
        configure(&mut c, r#"{"type":"configure","prompt_version":1,"prompt":"P"}"#).unwrap();
        let (b0, a) = c.next_chunk().unwrap();
        assert_eq!((b0.prompt.as_str(), b0.prompt_version, a.len()), ("P", 1, 0));
        assert_eq!(types(&prompt(&mut c, r#"{"type":"prompt","prompt_version":2,"prompt":"A"}"#)), ["prompt_pending"]);
        let (b1, a) = c.next_chunk().unwrap();
        assert_eq!((b1.prompt.as_str(), b1.prompt_version), ("P A", 2));
        assert_eq!(a, vec![json!({"type":"prompt_applied","prompt_version":2})]);
        // Every direction holds a director chunk (4 blocks) before an
        // appended one takes over: A's 3 remaining blocks, B for 4, then C
        // (announced once each), and C stays.
        prompt(&mut c, r#"{"type":"prompt","prompt_version":3,"prompt":"B","replan":false}"#);
        prompt(&mut c, r#"{"type":"prompt","prompt_version":4,"prompt":"C","replan":false}"#);
        let mut seen = Vec::new();
        for _ in 0..12 {
            let (p, a) = c.next_chunk().unwrap();
            seen.push((p.prompt.clone(), a.len()));
        }
        let want: Vec<(String, usize)> = [("P A", 0), ("P A", 0), ("P A", 0), ("P B", 1), ("P B", 0), ("P B", 0), ("P B", 0), ("P C", 1), ("P C", 0), ("P C", 0), ("P C", 0), ("P C", 0)]
            .iter()
            .map(|(p, n)| ((*p).to_owned(), *n))
            .collect();
        assert_eq!(seen, want);
        // A script: beats at 0, 3 and 6 s → 4 blocks, 4 blocks, then "z" stays.
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":5,"script":[{"offset":0,"prompt":"x"},{"offset":3,"prompt":"y"},{"offset":6,"prompt":"z"}]}"#);
        assert_eq!(types(&out), ["prompt_pending"]);
        let got: Vec<String> = (0..10).map(|_| c.next_chunk().unwrap().0.prompt).collect();
        assert_eq!(got, ["P x", "P x", "P x", "P x", "P y", "P y", "P y", "P y", "P z", "P z"]);
        let (segs, last) = plan_script_blocks(
            &serde_json::from_value::<Vec<ScriptBeat>>(json!([{"offset": 2, "prompt": "late"}, {"offset": 1_000_000_000, "prompt": "far"}])).unwrap(),
            c.limits().causal.as_ref().unwrap(),
        );
        assert_eq!(segs.iter().map(|s| s.blocks).collect::<Vec<_>>(), [3, 1_333_333_330, 1]);
        assert_eq!((segs[0].text, last), (None, Some(1)));
    }

    #[test]
    fn scripts_plan_chunks_at_beats() {
        let l = Limits::default();
        let beats: Vec<ScriptBeat> = serde_json::from_value(json!([
            {"offset": 0, "prompt": "open"},
            {"offset": 8, "end_image_url": "e8.png"},
            {"offset": 12, "prompt": "turn"},
            {"offset": 40, "end_image_url": "e40.png", "prompt": "arrive"},
        ]))
        .unwrap();
        let (segs, last) = plan_script(&beats, &l).unwrap();
        let secs: Vec<f64> = segs.iter().map(|s| s.seconds).collect();
        assert_eq!(secs, [8.0, 10.0, 10.0, 12.0]);
        assert_eq!(segs[0].end_image, Some(1));
        assert_eq!(segs[0].text, Some(0));
        // The beat at 12 s directs the chunk starting at 18 s (8 + 10 > 12).
        assert_eq!(segs[1].text, Some(0));
        assert_eq!(segs[2].text, Some(2));
        assert_eq!(segs[3].end_image, Some(3));
        assert_eq!(last, Some(3));
        // Infeasible timings.
        let near: Vec<ScriptBeat> = serde_json::from_value(json!([{"offset": 3, "end_image_url": "a"}])).unwrap();
        assert!(plan_script(&near, &l).is_err());
        let close: Vec<ScriptBeat> =
            serde_json::from_value(json!([{"offset": 10, "end_image_url": "a"}, {"offset": 12, "end_image_url": "b"}])).unwrap();
        assert!(plan_script(&close, &l).unwrap_err().contains("closer"));

        let mut c = Control::new(l);
        configure(
            &mut c,
            r#"{"type":"configure","prompt_version":1,"prompt":"P","script":[{"offset":0,"prompt":"open"},{"offset":8,"end_image_url":"e8.png"}]}"#,
        )
        .unwrap();
        let (p0, _) = c.next_chunk().unwrap();
        assert_eq!((p0.seconds, p0.prompt.as_str(), p0.end_image.clone()), (8.0, "P open", Some(PathBuf::from("e8.png"))));
        let (p1, _) = c.next_chunk().unwrap();
        assert_eq!((p1.seconds, p1.prompt.as_str()), (10.0, "P open"));

        // A live script replaces the deck and announces itself once.
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":3,"script":[{"offset":0,"prompt":"night"},{"offset":6,"end_image_url":"n.png"}]}"#);
        assert_eq!(types(&out), ["prompt_pending"]);
        let (p, applied) = c.next_chunk().unwrap();
        assert_eq!((p.prompt.as_str(), p.seconds), ("P night", 6.0));
        assert_eq!(applied[0]["script_origin_chunk_index"], 2);
        assert_eq!(applied[0]["script_mode"], "replace");
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":4,"prompt":"x","script":[{"offset":0,"prompt":"y"}]}"#);
        assert_eq!(out[0]["reason"], "invalid_script");
        let out = prompt(&mut c, r#"{"type":"prompt","prompt_version":5,"script":[{"offset":2,"end_image_url":"z"}]}"#);
        assert_eq!(out[0]["reason"], "infeasible_timing");
    }
}
