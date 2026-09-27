//! Generation and playout queues with submit-time reservation (design §5.5,
//! WP-12).
//!
//! A port of fast-h3's `fasth3_queue.py` (infinite-livestream): a clip passes
//! three stages, enqueued (the **generation queue**), built (the **playout
//! queue**) and consumed (playing removed it). Both queues are one bounded,
//! ordered, position-addressable container, [`ClipQueue`]; the player
//! (`stream::clip`) owns when entries cross between them. Pure bookkeeping,
//! no tokio and no engine, so it is tested on its own.
//!
//! [`ClipInfo`] is the wire form every message about a clip carries
//! (`{clip_id, prompt, metadata, frames, seconds, seed, ready}`, fast-h3
//! `ClipInfo`); [`ClipEntry::info`] is its single producer.

use std::path::PathBuf;
use std::sync::Arc;

use fastvideo_protocol::{JobMetrics, Pcm, RgbFrame};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Default generation-queue capacity (fast-h3 `generation_queue_size`).
pub const GENERATION_CAPACITY: usize = 20;
/// Default playout-queue capacity (fast-h3 `queue_size`); every entry holds a
/// built clip in host RAM, so this is also the memory budget (~1 GB per 14 s
/// at 768p).
pub const PLAYOUT_CAPACITY: usize = 10;

/// A clip as every clip-referencing message carries it (fast-h3 `ClipInfo`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipInfo {
    pub clip_id: Uuid,
    pub prompt: String,
    /// Opaque, echoed back on every message about the clip.
    pub metadata: String,
    pub frames: u32,
    /// `frames / fps`, rounded to 3 decimals.
    pub seconds: f64,
    pub seed: u64,
    /// Built (in the playout queue, or playing).
    pub ready: bool,
}

/// Which queue a clip sits in (`clip_moved.queue`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueName {
    Generation,
    Playout,
}

impl QueueName {
    pub fn as_str(self) -> &'static str {
        match self {
            QueueName::Generation => "generation",
            QueueName::Playout => "playout",
        }
    }
}

/// A built clip, ready for playout: frames plus wire-rate audio.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltClip {
    pub frames: Arc<Vec<RgbFrame>>,
    /// 48 kHz at the session's channel count, exactly
    /// `round(frames/fps·48000)` sample frames long (design §5.5 lockstep);
    /// `None` for a video-only session.
    pub audio: Option<Pcm>,
    /// Submit → built, seconds (wall).
    pub build_s: f64,
    pub metrics: JobMetrics,
}

/// One clip, from request to built payload (fast-h3 `ClipEntry`).
///
/// The client-facing fields are frozen at enqueue time.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipEntry {
    pub clip_id: Uuid,
    pub prompt: String,
    pub metadata: String,
    pub frames: u32,
    pub fps: u32,
    pub seed: u64,
    /// Director `prompt_version` of a `Chunk` (native extension).
    pub prompt_version: Option<u64>,
    /// `Keyframe{First}` requested for this clip (director `image_url`).
    /// With `AnchorLastFrame` an entry without one is anchored on the previous
    /// build's last frame.
    pub first_frame: Option<PathBuf>,
    /// `Keyframe{Last}` (director `end_image_url`).
    pub last_frame: Option<PathBuf>,
    /// A build for this entry is in flight (never submitted twice).
    pub building: bool,
    pub built: Option<BuiltClip>,
}

impl ClipEntry {
    /// A fresh entry with a new UUID.
    pub fn new(prompt: impl Into<String>, metadata: impl Into<String>, frames: u32, fps: u32, seed: u64) -> Self {
        Self {
            clip_id: Uuid::new_v4(),
            prompt: prompt.into(),
            metadata: metadata.into(),
            frames,
            fps,
            seed,
            prompt_version: None,
            first_frame: None,
            last_frame: None,
            building: false,
            built: None,
        }
    }

    pub fn ready(&self) -> bool {
        self.built.is_some()
    }

    /// Exact playout length.
    pub fn seconds(&self) -> f64 {
        self.frames as f64 / self.fps.max(1) as f64
    }

    /// The wire form.
    pub fn info(&self) -> ClipInfo {
        ClipInfo {
            clip_id: self.clip_id,
            prompt: self.prompt.clone(),
            metadata: self.metadata.clone(),
            frames: self.frames,
            seconds: (self.seconds() * 1000.0).round() / 1000.0,
            seed: self.seed,
            ready: self.ready(),
        }
    }
}

/// The queue is at capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueFull(pub usize);

impl std::fmt::Display for QueueFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the queue is full ({} clips)", self.0)
    }
}

/// A bounded, ordered, position-addressable queue of [`ClipEntry`].
#[derive(Clone, Debug)]
pub struct ClipQueue {
    capacity: usize,
    entries: Vec<ClipEntry>,
}

impl ClipQueue {
    /// `capacity` is clamped to at least 1.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Vec::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Another add would exceed the capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Inserts at `position` (clamped into `0..=len`; `None` appends) and
    /// returns the landing index.
    pub fn add(&mut self, entry: ClipEntry, position: Option<usize>) -> Result<usize, QueueFull> {
        if self.is_full() {
            return Err(QueueFull(self.capacity));
        }
        let i = position.map_or(self.entries.len(), |p| p.min(self.entries.len()));
        self.entries.insert(i, entry);
        Ok(i)
    }

    /// Repositions `id` (clamped); `None` when it is not here.
    pub fn move_to(&mut self, id: Uuid, position: usize) -> Option<usize> {
        let from = self.index_of(id)?;
        let e = self.entries.remove(from);
        let i = position.min(self.entries.len());
        self.entries.insert(i, e);
        Some(i)
    }

    pub fn index_of(&self, id: Uuid) -> Option<usize> {
        self.entries.iter().position(|e| e.clip_id == id)
    }

    pub fn get(&self, id: Uuid) -> Option<&ClipEntry> {
        self.entries.iter().find(|e| e.clip_id == id)
    }

    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut ClipEntry> {
        self.entries.iter_mut().find(|e| e.clip_id == id)
    }

    pub fn contains(&self, id: Uuid) -> bool {
        self.index_of(id).is_some()
    }

    pub fn head(&self) -> Option<&ClipEntry> {
        self.entries.first()
    }

    /// The front-most entry no build is running for.
    pub fn next_to_build(&mut self) -> Option<&mut ClipEntry> {
        self.entries.iter_mut().find(|e| !e.building)
    }

    /// Takes `id` out.
    pub fn remove(&mut self, id: Uuid) -> Option<ClipEntry> {
        let i = self.index_of(id)?;
        Some(self.entries.remove(i))
    }

    /// Takes the front entry out.
    pub fn pop_front(&mut self) -> Option<ClipEntry> {
        (!self.entries.is_empty()).then(|| self.entries.remove(0))
    }

    /// Drops every entry (built payloads included); returns how many.
    pub fn clear(&mut self) -> usize {
        let n = self.entries.len();
        self.entries.clear();
        n
    }

    pub fn iter(&self) -> impl Iterator<Item = &ClipEntry> {
        self.entries.iter()
    }

    /// Every entry's wire form, front first.
    pub fn snapshot(&self) -> Vec<ClipInfo> {
        self.entries.iter().map(ClipEntry::info).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(p: &str) -> ClipEntry {
        ClipEntry::new(p, "", 124, 24, 1)
    }

    fn prompts(q: &ClipQueue) -> Vec<String> {
        q.iter().map(|e| e.prompt.clone()).collect()
    }

    #[test]
    fn keeps_add_order_and_inserts_clamped() {
        let mut q = ClipQueue::new(5);
        q.add(e("a"), None).unwrap();
        q.add(e("b"), None).unwrap();
        assert_eq!(q.add(e("c"), Some(0)).unwrap(), 0);
        assert_eq!(q.add(e("d"), Some(99)).unwrap(), 3);
        assert_eq!(prompts(&q), ["c", "a", "b", "d"]);
    }

    #[test]
    fn move_repositions_within_the_queue() {
        let mut q = ClipQueue::new(5);
        let ids: Vec<Uuid> = ["a", "b", "c"]
            .iter()
            .map(|p| {
                let x = e(p);
                let id = x.clip_id;
                q.add(x, None).unwrap();
                id
            })
            .collect();
        assert_eq!(q.move_to(ids[2], 0), Some(0));
        assert_eq!(prompts(&q), ["c", "a", "b"]);
        assert_eq!(q.move_to(ids[2], 50), Some(2));
        assert_eq!(prompts(&q), ["a", "b", "c"]);
        assert_eq!(q.move_to(Uuid::new_v4(), 0), None);
    }

    #[test]
    fn bounded_and_distinct_ids() {
        let mut q = ClipQueue::new(2);
        let a = e("a");
        let b = e("b");
        assert_ne!(a.clip_id, b.clip_id);
        q.add(a, None).unwrap();
        q.add(b, None).unwrap();
        assert!(q.is_full());
        assert_eq!(q.add(e("c"), None), Err(QueueFull(2)));
    }

    #[test]
    fn building_entries_are_not_resubmitted() {
        let mut q = ClipQueue::new(3);
        q.add(e("a"), None).unwrap();
        q.add(e("b"), None).unwrap();
        q.next_to_build().unwrap().building = true;
        assert_eq!(q.next_to_build().unwrap().prompt, "b");
    }

    #[test]
    fn info_is_the_published_struct() {
        let x = ClipEntry::new("p", "m", 124, 24, 7);
        let v = serde_json::to_value(x.info()).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        let mut keys = keys;
        keys.sort_unstable();
        assert_eq!(keys, ["clip_id", "frames", "metadata", "prompt", "ready", "seconds", "seed"]);
        assert_eq!(v["seconds"], 5.167);
        assert_eq!(v["ready"], false);
    }
}
