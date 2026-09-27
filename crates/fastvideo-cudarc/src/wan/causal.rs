//! Causal Wan (Self-Forcing) inference as FastVideo runs it: the clip is
//! generated `num_frames_per_block` latent frames at a time, each block's
//! self-attention reading a per-layer KV cache of everything generated
//! before it (`fastvideo/models/wan/causal_transformer.py`
//! `CausalWanSelfAttention.forward` with a `kv_cache`,
//! `pipelines/basic/wan/stages/causal_denoising.py CausalDMDDenosingStage`).
//!
//! Per block of queries (`current_start .. current_end` tokens):
//!
//! * the block's roped K and V are written to the cache: appended when the
//!   block is new (`current_end > global_end`), written over the block's own
//!   slots when the same block is run again (the next denoising step, then
//!   the clean-context pass); with `local_attn_size != -1` a full cache first
//!   drops its oldest tokens after the `sink_size` sink frames;
//! * the queries attend, with no mask, to the last `max_attention` cached
//!   tokens up to the end of the block (their own block included, both ways).
//!
//! `max_attention` is `local_attn_size` frames, or `sliding_window_num_frames`
//! (21) frames when `local_attn_size == -1`, where a clip longer than that is
//! an error as in FastVideo. RoPE is absolute (`start_frame` offsets the
//! table) by default. FastVideo's `relativistic` cache policy
//! (`rope_cache_policy`, `models/dits/_relative_rope.py`) is
//! [`KvSpec::relativistic`]: the cache holds un-roped keys and every forward
//! ropes the window at positions `[0, window)`, the queries at its tail, so
//! positions stay in the trained range however long the rollout runs (the
//! open-ended stream, `wan::stream`).

use std::sync::Mutex;

use fastvideo_models::wan::WanVideoArchConfig;

use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Token geometry of one causal KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSpec {
    /// Tokens per latent frame.
    pub frame_tokens: usize,
    /// Physical cache length (FastVideo `kv_cache_size`).
    pub capacity: usize,
    /// Tokens kept at the head when the cache rolls (`sink_size` frames).
    pub sink: usize,
    /// Keys each query block reads, back from the end of the block.
    pub max_attention: usize,
    /// `local_attn_size != -1`: the cache rolls instead of refusing.
    pub rolling: bool,
    /// FastVideo `rope_cache_policy = "relativistic"`: raw (normed, un-roped)
    /// keys in the cache; the window is roped from position 0 each forward.
    pub relativistic: bool,
    /// [`KvRope::RebasedSink`]: absolute keys, the sink re-roped each block
    /// to sit just before the rest of the window.
    pub rebase_sink: bool,
}

/// How cached keys carry RoPE in a rolling cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvRope {
    /// Keys roped at their absolute frame, the sink too (FastVideo
    /// `rope_cache_policy = "absolute"`, the bounded path's).
    Absolute,
    /// FastVideo `relativistic`: un-roped keys cached, the whole window
    /// roped from position 0 at every forward, the queries at its tail.
    Relativistic,
    /// The relativistic geometry at the absolute policy's cost. Attention
    /// only sees relative positions, and the rolled part of the window is
    /// contiguous up to the queries either way; only the sink differs. So
    /// keep absolute keys and, once the cache rolls, re-rope the sink keys
    /// (from an un-roped copy taken once) to the frames just before the
    /// rolled part. Every query–key offset is then the relativistic one;
    /// the per-forward cost is one sink-sized rope per layer per block.
    RebasedSink,
}

impl KvSpec {
    /// FastVideo `_initialize_kv_cache` and `CausalWanSelfAttention` sizes.
    pub fn for_config(cfg: &WanVideoArchConfig, frame_tokens: usize) -> Self {
        let sink = cfg.sink_size * frame_tokens;
        match usize::try_from(cfg.local_attn_size) {
            Ok(l) => Self {
                frame_tokens,
                capacity: l * frame_tokens,
                sink,
                max_attention: l * frame_tokens,
                rolling: true,
                relativistic: false,
                rebase_sink: false,
            },
            Err(_) => {
                let n = cfg.sliding_window_num_frames * frame_tokens;
                Self {
                    frame_tokens,
                    capacity: n,
                    sink,
                    max_attention: n,
                    rolling: false,
                    relativistic: false,
                    rebase_sink: false,
                }
            }
        }
    }

    /// A rolling cache of `local_attn_frames` frames that keeps the first
    /// `sink_frames` frames when it rolls (FastVideo `local_attn_size`,
    /// `sink_size`), with the given RoPE policy.
    pub fn rolling(
        frame_tokens: usize,
        local_attn_frames: usize,
        sink_frames: usize,
        rope: KvRope,
    ) -> Result<Self> {
        if local_attn_frames == 0 || sink_frames >= local_attn_frames {
            return Err(msg(format!(
                "causal kv: sink {sink_frames} frames must be below the window \
                 ({local_attn_frames} frames)"
            )));
        }
        Ok(Self {
            frame_tokens,
            capacity: local_attn_frames * frame_tokens,
            sink: sink_frames * frame_tokens,
            max_attention: local_attn_frames * frame_tokens,
            rolling: true,
            relativistic: rope == KvRope::Relativistic,
            rebase_sink: rope == KvRope::RebasedSink && sink_frames > 0,
        })
    }

    /// First frame the sink keys should sit at when the write that ends at
    /// frame `end_frame` is done: just before the rolled part of a full
    /// window, and where they are (0) until the cache rolls.
    pub fn sink_target(&self, end_frame: usize) -> usize {
        end_frame.saturating_sub(self.window_frames())
    }

    /// Frames the queries can read (the RoPE table a relativistic window
    /// needs).
    pub fn window_frames(&self) -> usize {
        self.max_attention / self.frame_tokens.max(1)
    }
}

/// The valid prefix `[.., local_end)` of one layer's cache, `[B, H, len, D]`.
#[derive(Default)]
struct KvLayer {
    k: Option<CudaTensor>,
    v: Option<CudaTensor>,
    global_end: usize,
    local_end: usize,
    /// [`KvRope::RebasedSink`]: the sink keys without RoPE (f32), taken
    /// once, and the first frame the cached sink keys are roped at now.
    raw_sink: Option<CudaTensor>,
    sink_at: usize,
}

/// Wan's interleaved RoPE (`is_neox_style=False`, tables repeat-interleaved)
/// on BHSD `x` with `[S, D]` tables: BHSD is BSHD with batch `B·H` and one
/// head. The result is f32.
pub(crate) fn rope_bhsd_f32(x: &CudaTensor, cos: &CudaTensor, sin: &CudaTensor) -> Result<CudaTensor> {
    let [b, h, s, d] = x.shape[..] else {
        return Err(msg(format!("rope_bhsd expects BHSD, got {:?}", x.shape)));
    };
    x.reshape(vec![b * h, s, 1, d])?
        .apply_rotary_bshd(cos, sin)?
        .reshape(vec![b, h, s, d])
}

/// [`rope_bhsd_f32`] in `x`'s dtype (FastVideo's `.type_as(v)`).
pub(crate) fn rope_bhsd(x: &CudaTensor, cos: &CudaTensor, sin: &CudaTensor) -> Result<CudaTensor> {
    let y = rope_bhsd_f32(x, cos, sin)?;
    if x.is_bf16() && !y.is_bf16() {
        y.quantize_bf16()
    } else {
        Ok(y)
    }
}

/// Where the cache pointers moved for one write (the testable arithmetic of
/// FastVideo's `CausalWanSelfAttention.forward` kv branch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvMove {
    /// Tokens dropped after the sink before writing (`num_evicted_tokens`).
    pub evicted: usize,
    /// Where the block's K/V are written, `[local_start, local_end)`.
    pub local_start: usize,
    pub local_end: usize,
    /// The window the queries read, `[window_start, local_end)`.
    pub window_start: usize,
}

impl KvSpec {
    /// The pointer arithmetic for `n` new tokens at `current_start`, from the
    /// cache state `(global_end, local_end)`.
    pub fn plan(
        &self,
        global_end: usize,
        local_end_prev: usize,
        current_start: usize,
        n: usize,
    ) -> Result<KvMove> {
        let current_end = current_start + n;
        if !self.rolling && current_end > self.max_attention {
            return Err(msg(format!(
                "causal Wan local_attn_size=-1 keeps a {}-frame KV window; got current_end={current_end} \
                 tokens with frame_seqlen={} (set local_attn_size for longer rollouts)",
                self.max_attention / self.frame_tokens.max(1),
                self.frame_tokens
            )));
        }
        if current_end < global_end && current_end + local_end_prev < global_end {
            return Err(msg("causal kv: block before the cache start"));
        }
        let advance = current_end.saturating_sub(global_end);
        let evicted =
            if self.rolling && current_end > global_end && n + local_end_prev > self.capacity {
                n + local_end_prev - self.capacity
            } else {
                0
            };
        let local_end = (local_end_prev + advance)
            .checked_sub(if current_end >= global_end {
                0
            } else {
                global_end - current_end
            })
            .and_then(|e| e.checked_sub(evicted))
            .ok_or_else(|| msg("causal kv: pointer underflow"))?;
        let local_start = local_end
            .checked_sub(n)
            .ok_or_else(|| msg("causal kv: block longer than the cache"))?;
        if local_end > self.capacity.max(n) {
            return Err(msg(format!(
                "causal kv: write end {local_end} past the cache ({} tokens)",
                self.capacity
            )));
        }
        Ok(KvMove {
            evicted,
            local_start,
            local_end,
            window_start: local_end.saturating_sub(self.max_attention),
        })
    }
}

/// Per-layer K/V of one causal generation (batch as given by the caller).
pub struct CausalKvCache {
    pub spec: KvSpec,
    layers: Vec<Mutex<KvLayer>>,
}

impl CausalKvCache {
    pub fn new(spec: KvSpec, num_layers: usize) -> Self {
        Self {
            spec,
            layers: (0..num_layers)
                .map(|_| Mutex::new(KvLayer::default()))
                .collect(),
        }
    }

    /// Write one block's `k`/`v` (`[B, H, n, D]`, RoPE applied) for layer
    /// `layer` at token `current_start`; returns the K/V window its queries
    /// attend to.
    pub fn update(
        &self,
        layer: usize,
        k: &CudaTensor,
        v: &CudaTensor,
        current_start: usize,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let n = k.shape[2];
        let mut slot = self
            .layers
            .get(layer)
            .ok_or_else(|| msg(format!("causal kv: no layer {layer}")))?
            .lock()
            .expect("causal kv lock");
        let mv = self
            .spec
            .plan(slot.global_end, slot.local_end, current_start, n)?;
        let keep = |t: Option<&CudaTensor>| -> Result<Option<CudaTensor>> {
            let Some(t) = t else { return Ok(None) };
            // Roll: [0, sink) stays, [sink + evicted, local_end_prev) moves
            // down to sink.
            let rolled = if mv.evicted > 0 {
                let len = t.shape[2];
                let sink = self.spec.sink.min(len);
                let tail_start = (sink + mv.evicted).min(len);
                let mut parts = Vec::new();
                if sink > 0 {
                    parts.push(t.narrow(2, 0, sink)?);
                }
                if len > tail_start {
                    parts.push(t.narrow(2, tail_start, len - tail_start)?);
                }
                match parts.len() {
                    0 => None,
                    1 => parts.pop(),
                    _ => Some(CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 2)?),
                }
            } else {
                Some(t.clone())
            };
            // Everything before the write position survives.
            match rolled {
                Some(r) if mv.local_start > 0 => {
                    if mv.local_start > r.shape[2] {
                        return Err(msg(format!(
                            "causal kv: write at {} leaves a gap after {} cached tokens",
                            mv.local_start, r.shape[2]
                        )));
                    }
                    Ok(Some(r.narrow(2, 0, mv.local_start)?))
                }
                _ => Ok(None),
            }
        };
        let (k_keep, v_keep) = (keep(slot.k.as_ref())?, keep(slot.v.as_ref())?);
        if mv.local_start > 0 && k_keep.is_none() {
            return Err(msg("causal kv: write past an empty cache"));
        }
        let join = |prefix: Option<CudaTensor>, new: &CudaTensor| -> Result<CudaTensor> {
            match prefix {
                Some(p) => CudaTensor::cat(&[&p, new], 2),
                None => Ok(new.clone()),
            }
        };
        let k_all = join(k_keep, k)?;
        let v_all = join(v_keep, v)?;
        slot.global_end = current_start + n;
        slot.local_end = mv.local_end;
        let window = |t: &CudaTensor| t.narrow(2, mv.window_start, mv.local_end - mv.window_start);
        let out = (window(&k_all)?, window(&v_all)?);
        slot.k = Some(k_all);
        slot.v = Some(v_all);
        Ok(out)
    }

    /// Cached tokens of layer 0 (`local_end`), for logs and tests.
    pub fn len(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.lock().expect("causal kv lock").local_end)
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tokens of the window the next write of `n` tokens at `current_start`
    /// will read (layer 0's pointers; every layer moves in step).
    pub fn window_after(&self, current_start: usize, n: usize) -> Result<usize> {
        let (global_end, local_end) = self
            .layers
            .first()
            .map(|l| {
                let l = l.lock().expect("causal kv lock");
                (l.global_end, l.local_end)
            })
            .unwrap_or((0, 0));
        let mv = self.spec.plan(global_end, local_end, current_start, n)?;
        Ok(mv.local_end - mv.window_start)
    }

    /// [`KvRope::RebasedSink`]: re-rope every layer's sink keys to start at
    /// frame `target` (see [`KvSpec::sink_target`]). `orig` are the RoPE
    /// tables of the sink's own frames `0 .. sink` (to take the un-roped
    /// copy the first time), `new` those of `target .. target + sink`
    /// (`[sink_tokens, D]` each). A no-op when the sink is already there.
    pub fn rebase_sink(
        &self,
        target: usize,
        orig: (&CudaTensor, &CudaTensor),
        new: (&CudaTensor, &CudaTensor),
    ) -> Result<()> {
        let sink = self.spec.sink;
        if sink == 0 {
            return Ok(());
        }
        let mut neg_orig: Option<CudaTensor> = None;
        for l in &self.layers {
            let mut slot = l.lock().expect("causal kv lock");
            if slot.sink_at == target {
                continue;
            }
            let Some(k) = slot.k.clone() else { continue };
            if k.shape[2] < sink {
                continue;
            }
            if slot.raw_sink.is_none() {
                // Rotating back by the sink's own positions (sin negated)
                // recovers the un-roped keys up to one rounding; kept f32
                // so re-roping never compounds error block after block.
                if neg_orig.is_none() {
                    neg_orig = Some(orig.1.mul_scalar(-1.0));
                }
                let neg = neg_orig.as_ref().expect("negated sin");
                slot.raw_sink = Some(rope_bhsd_f32(&k.narrow(2, 0, sink)?, orig.0, neg)?);
            }
            let raw = slot.raw_sink.as_ref().expect("raw sink");
            let mut head = rope_bhsd_f32(raw, new.0, new.1)?;
            if k.is_bf16() {
                head = head.quantize_bf16()?;
            }
            let len = k.shape[2];
            slot.k = Some(if len > sink {
                CudaTensor::cat(&[&head, &k.narrow(2, sink, len - sink)?], 2)?
            } else {
                head
            });
            slot.sink_at = target;
        }
        Ok(())
    }

    /// Drop every cached K/V (a new rollout from block 0).
    pub fn reset(&self) {
        for l in &self.layers {
            *l.lock().expect("causal kv lock") = KvLayer::default();
        }
    }

    /// Bytes of K and V held across all layers (for memory-growth checks).
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                let l = l.lock().expect("causal kv lock");
                [l.k.as_ref(), l.v.as_ref()]
                    .into_iter()
                    .flatten()
                    .map(|t| t.numel() * if t.is_bf16() { 2 } else { 4 })
                    .sum::<usize>()
            })
            .sum()
    }
}

/// One layer's view of the cache for a block forward.
#[derive(Clone, Copy)]
pub struct KvAt<'a> {
    pub cache: &'a CausalKvCache,
    pub layer: usize,
    pub current_start: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn global(frames: usize) -> KvSpec {
        KvSpec {
            frame_tokens: 10,
            capacity: frames * 10,
            sink: 0,
            max_attention: frames * 10,
            rolling: false,
            relativistic: false,
            rebase_sink: false,
        }
    }

    #[test]
    fn blocks_append_and_steps_overwrite() {
        let s = global(21);
        // Block 0 (30 tokens), four steps and the context pass: same slots.
        let m = s.plan(0, 0, 0, 30).unwrap();
        assert_eq!((m.local_start, m.local_end, m.window_start), (0, 30, 0));
        let m = s.plan(30, 30, 0, 30).unwrap();
        assert_eq!((m.local_start, m.local_end), (0, 30));
        // Block 1 appends and reads blocks 0 and 1.
        let m = s.plan(30, 30, 30, 30).unwrap();
        assert_eq!((m.local_start, m.local_end, m.window_start), (30, 60, 0));
        // 22 frames with a global window: FastVideo refuses.
        assert!(s.plan(210, 210, 210, 10).is_err());
    }

    #[test]
    fn rolling_cache_keeps_the_sink() {
        // local_attn_size 6 frames, sink 1 frame, 3-frame blocks of 10 tokens.
        let s = KvSpec {
            frame_tokens: 10,
            capacity: 60,
            sink: 10,
            max_attention: 60,
            rolling: true,
            relativistic: false,
            rebase_sink: false,
        };
        assert_eq!(s.plan(0, 0, 0, 30).unwrap().local_end, 30);
        assert_eq!(s.plan(30, 30, 30, 30).unwrap().local_end, 60);
        // Block 2: 30 tokens evicted after the sink; written at the end.
        let m = s.plan(60, 60, 60, 30).unwrap();
        assert_eq!(
            (m.evicted, m.local_start, m.local_end, m.window_start),
            (30, 30, 60, 0)
        );
        // The next step of block 2 overwrites in place.
        let m = s.plan(90, 60, 60, 30).unwrap();
        assert_eq!((m.evicted, m.local_start, m.local_end), (0, 30, 60));
    }

    /// Block by block through the cache at one timestep is the whole clip
    /// under FastVideo's blockwise mask: the same K/V reach every query, and
    /// each block's RoPE starts at its first frame.
    #[test]
    fn cached_blocks_equal_the_masked_whole_clip() {
        use crate::wan::transformer::WanTransformer3D;
        use crate::wan::weights::WeightMap;
        let mut cfg = WanVideoArchConfig::tiny();
        cfg.causal = true;
        cfg.num_layers = 2;
        cfg.num_frames_per_block = 2;
        let map = WeightMap::generated(|key, shape| {
            let n: usize = shape.iter().product();
            let seed = key
                .bytes()
                .fold(7u64, |a, b| a.wrapping_mul(131).wrapping_add(u64::from(b)));
            (0..n)
                .map(|i| {
                    let x = seed
                        .wrapping_add(i as u64)
                        .wrapping_mul(6364136223846793005)
                        >> 35;
                    ((x % 2001) as f32 / 1000.0 - 1.0) * 0.3
                })
                .collect()
        });
        let dit = WanTransformer3D::load(cfg.clone(), &map).unwrap();
        let (c, t, h, w) = (cfg.in_channels, 6usize, 4usize, 6usize);
        let lat: Vec<f32> = (0..c * t * h * w)
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let latents = CudaTensor::from_vec(lat, vec![1, c, t, h, w]).unwrap();
        let text: Vec<f32> = (0..cfg.text_len * cfg.text_dim)
            .map(|i| ((i as f32) * 0.11).cos())
            .collect();
        let text = CudaTensor::from_vec(text, vec![1, cfg.text_len, cfg.text_dim]).unwrap();
        let ts = CudaTensor::from_vec(vec![937.5], vec![1]).unwrap();
        let whole = dit.forward_ctx(&latents, &ts, &text, None).unwrap();
        let frame_tokens = (h / 2) * (w / 2);
        let cache = CausalKvCache::new(KvSpec::for_config(&cfg, frame_tokens), cfg.num_layers);
        let fpb = cfg.num_frames_per_block;
        let mut parts = Vec::new();
        for blk in 0..t / fpb {
            let x = latents.narrow(2, blk * fpb, fpb).unwrap();
            parts.push(dit.forward_kv(&x, &ts, &text, &cache, blk * fpb).unwrap());
        }
        let blocks = CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 2).unwrap();
        let (a, b) = (whole.host_cow().unwrap(), blocks.host_cow().unwrap());
        let err = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let scale = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(scale > 1e-3, "degenerate output");
        assert!(
            err <= 1e-4 * scale.max(1.0),
            "max abs {err} (scale {scale})"
        );
        assert_eq!(cache.len(), t * frame_tokens);
    }

    fn tiny_causal_dit() -> (crate::wan::transformer::WanTransformer3D, WanVideoArchConfig) {
        use crate::wan::transformer::WanTransformer3D;
        use crate::wan::weights::WeightMap;
        let mut cfg = WanVideoArchConfig::tiny();
        cfg.causal = true;
        cfg.num_layers = 2;
        cfg.num_frames_per_block = 2;
        let map = WeightMap::generated(|key, shape| {
            let n: usize = shape.iter().product();
            let seed = key
                .bytes()
                .fold(7u64, |a, b| a.wrapping_mul(131).wrapping_add(u64::from(b)));
            (0..n)
                .map(|i| {
                    let x = seed
                        .wrapping_add(i as u64)
                        .wrapping_mul(6364136223846793005)
                        >> 35;
                    ((x % 2001) as f32 / 1000.0 - 1.0) * 0.3
                })
                .collect()
        });
        (WanTransformer3D::load(cfg.clone(), &map).unwrap(), cfg)
    }

    /// `blocks` blocks of the tiny causal DiT through a rolling cache.
    fn roll(rope: KvRope, window: usize, sink: usize, blocks: usize) -> Vec<f32> {
        let (dit, cfg) = tiny_causal_dit();
        let (c, h, w, fpb) = (cfg.in_channels, 4usize, 6usize, cfg.num_frames_per_block);
        let frame_tokens = (h / 2) * (w / 2);
        let text: Vec<f32> = (0..cfg.text_len * cfg.text_dim)
            .map(|i| ((i as f32) * 0.11).cos())
            .collect();
        let text = CudaTensor::from_vec(text, vec![1, cfg.text_len, cfg.text_dim]).unwrap();
        let ts = CudaTensor::from_vec(vec![500.0], vec![1]).unwrap();
        let spec = KvSpec::rolling(frame_tokens, window, sink, rope).unwrap();
        let cache = CausalKvCache::new(spec, cfg.num_layers);
        let mut out = Vec::new();
        for blk in 0..blocks {
            let lat: Vec<f32> = (0..c * fpb * h * w)
                .map(|i| ((i as f32) * 0.37 + blk as f32).sin())
                .collect();
            let x = CudaTensor::from_vec(lat, vec![1, c, fpb, h, w]).unwrap();
            let y = dit.forward_kv(&x, &ts, &text, &cache, blk * fpb).unwrap();
            out.extend_from_slice(&y.host_cow().unwrap());
        }
        assert_eq!(cache.len(), window.min(blocks * fpb) * frame_tokens);
        out
    }

    /// (max abs difference, max abs of `a`).
    fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
        let err = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let scale = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(scale > 1e-3, "degenerate output");
        (err, scale)
    }

    /// Relativistic RoPE only moves every position by the same amount while
    /// the window is contiguous (no sink), and attention sees relative
    /// positions only: rolling past the window, it matches absolute RoPE.
    #[test]
    fn relativistic_rope_matches_absolute_without_a_sink() {
        let a = roll(KvRope::Absolute, 4, 0, 5);
        let r = roll(KvRope::Relativistic, 4, 0, 5);
        let (err, scale) = max_diff(&a, &r);
        assert!(err <= 1e-4 * scale.max(1.0), "max abs {err} (scale {scale})");
    }

    /// With a sink the policies part once the cache rolls; the rebased sink
    /// gives FastVideo's relativistic offsets.
    #[test]
    fn rebased_sink_matches_relativistic() {
        let r = roll(KvRope::Relativistic, 6, 2, 8);
        let b = roll(KvRope::RebasedSink, 6, 2, 8);
        let a = roll(KvRope::Absolute, 6, 2, 8);
        let (err, scale) = max_diff(&r, &b);
        assert!(err <= 1e-4 * scale.max(1.0), "rebased: max abs {err} (scale {scale})");
        let (apart, _) = max_diff(&r, &a);
        assert!(apart > 1e2 * err.max(1e-6), "absolute should differ with a sink ({apart})");
    }

    /// Every policy runs past the checkpoint's RoPE table
    /// (`rope_max_seq_len`) in constant cache memory.
    #[test]
    fn rollouts_outlive_the_rope_table() {
        let (dit, cfg) = tiny_causal_dit();
        let (c, h, w, fpb) = (cfg.in_channels, 4usize, 6usize, cfg.num_frames_per_block);
        let frame_tokens = (h / 2) * (w / 2);
        let text = CudaTensor::zeros(&[1, cfg.text_len, cfg.text_dim]);
        let ts = CudaTensor::from_vec(vec![0.0], vec![1]).unwrap();
        let x = CudaTensor::from_vec(
            (0..c * fpb * h * w).map(|i| (i as f32 * 0.1).cos()).collect(),
            vec![1, c, fpb, h, w],
        )
        .unwrap();
        for rope in [KvRope::Absolute, KvRope::Relativistic, KvRope::RebasedSink] {
            let spec = KvSpec::rolling(frame_tokens, 6, 2, rope).unwrap();
            let cache = CausalKvCache::new(spec, cfg.num_layers);
            let blocks = cfg.rope_max_seq_len / fpb + 3;
            let mut bytes = 0;
            for blk in 0..blocks {
                let y = dit.forward_kv(&x, &ts, &text, &cache, blk * fpb).unwrap();
                assert!(y.host_cow().unwrap().iter().all(|v| v.is_finite()));
                if blk == 3 {
                    bytes = cache.bytes();
                }
                dit.forget_rotary_before((blk * fpb).saturating_sub(6));
            }
            assert_eq!(cache.len(), 6 * frame_tokens);
            assert_eq!(cache.bytes(), bytes, "{rope:?}: the cache grew after it filled");
            cache.reset();
            assert!(cache.is_empty());
        }
    }

    #[test]
    fn update_keeps_prefix_and_window() {
        let spec = KvSpec {
            frame_tokens: 2,
            capacity: 8,
            sink: 2,
            max_attention: 8,
            rolling: true,
            relativistic: false,
            rebase_sink: false,
        };
        let cache = CausalKvCache::new(spec, 1);
        let blk = |base: f32| {
            let data: Vec<f32> = (0..4).map(|i| base + i as f32).collect();
            CudaTensor::from_vec(data, vec![1, 1, 4, 1]).unwrap()
        };
        let (k, _) = cache.update(0, &blk(0.0), &blk(0.0), 0).unwrap();
        assert_eq!(k.host_cow().unwrap().to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
        // Same block again (next step): overwritten.
        let (k, _) = cache.update(0, &blk(10.0), &blk(10.0), 0).unwrap();
        assert_eq!(k.host_cow().unwrap().to_vec(), vec![10.0, 11.0, 12.0, 13.0]);
        let (k, _) = cache.update(0, &blk(20.0), &blk(20.0), 4).unwrap();
        assert_eq!(k.shape[2], 8);
        // Third block: 4 evicted after the 2-token sink.
        let (k, _) = cache.update(0, &blk(30.0), &blk(30.0), 8).unwrap();
        assert_eq!(
            k.host_cow().unwrap().to_vec(),
            vec![10.0, 11.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0]
        );
        assert_eq!(cache.len(), 8);
    }
}
