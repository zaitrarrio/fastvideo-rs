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
    /// Static mode ([`CausalKvCache::new_static`]): the persistent buffers,
    /// allocated at the first write and kept across [`CausalKvCache::reset`].
    st: Option<StaticKv>,
    /// Static mode: `st.raw_sink` holds the un-roped sink.
    raw_valid: bool,
}

/// One layer's persistent storage in a static cache: `[B, H, capacity, D]`
/// K and V in the keys' dtype, valid over `[0, local_end)`, and the f32
/// un-roped sink `[B, H, sink, D]` of [`KvRope::RebasedSink`]. Every write
/// lands in place, so the addresses never change: what a CUDA graph of a
/// block forward needs (`wan::graph`).
struct StaticKv {
    k: CudaTensor,
    v: CudaTensor,
    raw_sink: Option<CudaTensor>,
}

/// A fresh tensor of `shape` in `like`'s dtype (f32 with `f32_only`) and
/// residency: contents unspecified on the device, zero on the host.
fn alloc_like(like: &CudaTensor, shape: Vec<usize>, f32_only: bool) -> Result<CudaTensor> {
    let n: usize = shape.iter().product();
    #[cfg(feature = "cuda")]
    if like.is_device_fresh() {
        return super::act16::OutBuf::new(n, like.is_bf16() && !f32_only)?.into_tensor(shape);
    }
    let dtype = if f32_only {
        super::tensor::TensorDType::F32
    } else {
        like.dtype()
    };
    Ok(CudaTensor::host_only_dtype(vec![0.0; n], shape, dtype))
}

/// `dst[:, :, dst_off .. dst_off + len] = src[:, :, src_off .. src_off + len]`
/// on BHSD tensors, in place in `dst`'s storage.
fn copy_tokens(
    dst: &mut CudaTensor,
    dst_off: usize,
    src: &CudaTensor,
    src_off: usize,
    len: usize,
) -> Result<()> {
    let (&[b, h, s_dst, d], &[sb, sh, s_src, sd]) = (&dst.shape[..], &src.shape[..]) else {
        return Err(msg("causal kv: copy_tokens expects BHSD"));
    };
    if (b, h, d) != (sb, sh, sd) || dst_off + len > s_dst || src_off + len > s_src {
        return Err(msg(format!(
            "causal kv: copy {len} tokens {:?}@{src_off} -> {:?}@{dst_off}",
            src.shape, dst.shape
        )));
    }
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    if dst.is_device_fresh() {
        return super::act16::copy_rows_into(
            src,
            dst,
            b * h,
            len * d,
            s_src * d,
            s_dst * d,
            src_off * d,
            dst_off * d,
        );
    }
    let sv = src.host_cow()?.into_owned();
    let dv = dst.host_mut()?;
    for r in 0..b * h {
        let (o, i) = ((r * s_dst + dst_off) * d, (r * s_src + src_off) * d);
        dv[o..o + len * d].copy_from_slice(&sv[i..i + len * d]);
    }
    Ok(())
}

/// Move tokens `[from, from + len)` down to `[to, to + len)` (`to < from`)
/// inside `t`, in chunks no longer than the shift so that no chunk's source
/// overlaps its destination (the device copy runs its elements in parallel).
fn shift_tokens_down(t: &mut CudaTensor, from: usize, to: usize, len: usize) -> Result<()> {
    let shift = from - to;
    let mut done = 0;
    while done < len {
        let n = shift.min(len - done);
        let src = t.clone();
        copy_tokens(t, to + done, &src, from + done, n)?;
        done += n;
    }
    Ok(())
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
    /// Persistent in-place buffers instead of a new tensor per write.
    static_mode: bool,
    /// LongLive's recompute guard ([`Self::set_sink_guard`]): a write that
    /// re-runs cached frames (`current_end <= global_end`, `current_start >
    /// 0`) leaves the sink slots as they are.
    sink_guard: std::sync::atomic::AtomicBool,
}

/// One layer's cache pointers (see [`CausalKvCache::pointers`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvPointers {
    pub global_end: usize,
    pub local_end: usize,
    pub sink_at: usize,
    pub raw_valid: bool,
}

/// What decides the device work of one block forward through a static
/// cache: two forwards with the same key run the same kernels on the same
/// buffers with the same shapes (one CUDA graph per key, `wan::graph`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvBlockKey {
    /// Valid cached tokens before the forward.
    pub local_end: usize,
    /// The write appends (a new block) rather than overwriting in place.
    pub appends: bool,
    /// Tokens dropped after the sink first.
    pub evicted: usize,
    /// The sink is re-roped ([`KvRope::RebasedSink`]) first ...
    pub rebase: bool,
    /// ... from an un-roped copy taken before that.
    pub take_raw: bool,
}

impl CausalKvCache {
    pub fn new(spec: KvSpec, num_layers: usize) -> Self {
        Self {
            spec,
            layers: (0..num_layers)
                .map(|_| Mutex::new(KvLayer::default()))
                .collect(),
            static_mode: false,
            sink_guard: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// A cache whose K/V live in one persistent `[B, H, capacity, D]`
    /// buffer pair per layer, written in place (the roll moves tokens down
    /// inside it), so a block forward touches the same addresses every time:
    /// what CUDA-graph replay needs. Same values as [`Self::new`]: the
    /// window the queries read is the same tokens in the same order.
    pub fn new_static(spec: KvSpec, num_layers: usize) -> Self {
        Self {
            static_mode: true,
            ..Self::new(spec, num_layers)
        }
    }

    pub fn is_static(&self) -> bool {
        self.static_mode
    }

    /// LongLive's KV re-cache (`CausalWanSelfAttention.forward`,
    /// `is_recompute`): while on, a write that re-runs frames already in the
    /// cache (`current_start > 0` and `current_end <= global_end`) does not
    /// write the sink slots: the queries still read the sink as it was (the
    /// frame sink keeps the first prompt's keys, `global_sink = true`). Off
    /// (the default) every write lands whole, as FastVideo does. Only the
    /// one-off re-cache forward turns it on.
    pub fn set_sink_guard(&self, on: bool) {
        self.sink_guard.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn sink_guard(&self) -> bool {
        self.sink_guard.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Leading tokens of an `n`-token write at `current_start` (planned as
    /// `mv` from `global_end`) that the sink guard keeps out of the cache.
    fn guarded(&self, global_end: usize, current_start: usize, n: usize, mv: &KvMove) -> usize {
        let recompute = current_start > 0 && current_start + n <= global_end;
        if self.sink_guard() && recompute {
            self.spec.sink.saturating_sub(mv.local_start).min(n)
        } else {
            0
        }
    }

    /// Static mode: every layer's buffers exist (after the first write).
    pub fn is_allocated(&self) -> bool {
        self.static_mode
            && self
                .layers
                .iter()
                .all(|l| l.lock().expect("causal kv lock").st.is_some())
    }

    /// Every layer's pointers.
    pub fn pointers(&self) -> Vec<KvPointers> {
        self.layers
            .iter()
            .map(|l| {
                let l = l.lock().expect("causal kv lock");
                KvPointers {
                    global_end: l.global_end,
                    local_end: l.local_end,
                    sink_at: l.sink_at,
                    raw_valid: l.raw_valid,
                }
            })
            .collect()
    }

    /// Put pointers back: a CUDA-graph replay moves no host state, and a
    /// failed capture must leave none behind. Static mode only (the buffers
    /// hold whatever the device work wrote).
    pub fn set_pointers(&self, p: &[KvPointers]) -> Result<()> {
        if !self.static_mode || p.len() != self.layers.len() {
            return Err(msg(
                "causal kv: set_pointers needs a static cache and one entry per layer",
            ));
        }
        for (l, p) in self.layers.iter().zip(p) {
            let mut l = l.lock().expect("causal kv lock");
            l.global_end = p.global_end;
            l.local_end = p.local_end;
            l.sink_at = p.sink_at;
            l.raw_valid = p.raw_valid;
        }
        Ok(())
    }

    /// The [`KvBlockKey`] of a forward of `n` tokens at `current_start`
    /// whose RebasedSink target frame is `sink_target` (0: none), from layer
    /// 0's pointers (every layer moves in step).
    pub fn block_key(
        &self,
        current_start: usize,
        n: usize,
        sink_target: usize,
    ) -> Result<KvBlockKey> {
        let p = self.pointers().first().copied().unwrap_or_default();
        let mv = self.spec.plan(p.global_end, p.local_end, current_start, n)?;
        let rebase = self.spec.rebase_sink
            && self.spec.sink > 0
            && sink_target > 0
            && p.sink_at != sink_target
            && p.local_end >= self.spec.sink;
        Ok(KvBlockKey {
            local_end: p.local_end,
            appends: current_start + n > p.global_end,
            evicted: mv.evicted,
            rebase,
            take_raw: rebase && !p.raw_valid,
        })
    }

    /// The host bookkeeping of one block forward of `n` tokens at
    /// `current_start` whose RebasedSink target frame is `sink_target` (0:
    /// none), without its device work: what a CUDA-graph replay of that
    /// forward leaves behind. [`Self::rebase_sink`]'s pointer moves, then
    /// [`Self::update`]'s, on every layer. Static mode only.
    pub fn advance(&self, current_start: usize, n: usize, sink_target: usize) -> Result<()> {
        if !self.static_mode {
            return Err(msg("causal kv: advance needs a static cache"));
        }
        let sink = self.spec.sink;
        for l in &self.layers {
            let mut slot = l.lock().expect("causal kv lock");
            if self.spec.rebase_sink
                && sink > 0
                && sink_target > 0
                && slot.sink_at != sink_target
                && slot.st.is_some()
                && slot.local_end >= sink
            {
                slot.raw_valid = true;
                slot.sink_at = sink_target;
            }
            let mv = self
                .spec
                .plan(slot.global_end, slot.local_end, current_start, n)?;
            if slot.st.is_none() {
                return Err(msg("causal kv: advance over an unwritten cache"));
            }
            slot.global_end = current_start + n;
            slot.local_end = mv.local_end;
        }
        Ok(())
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
        let skip = self.guarded(slot.global_end, current_start, n, &mv);
        if self.static_mode {
            return self.update_static(&mut slot, mv, skip, k, v, current_start);
        }
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
            // Everything before the write position survives (with the sink
            // guard, the guarded sink slots too).
            let at = mv.local_start + skip;
            match rolled {
                Some(r) if at > 0 => {
                    if at > r.shape[2] {
                        return Err(msg(format!(
                            "causal kv: write at {at} leaves a gap after {} cached tokens",
                            r.shape[2]
                        )));
                    }
                    Ok(Some(r.narrow(2, 0, at)?))
                }
                _ => Ok(None),
            }
        };
        let (k_keep, v_keep) = (keep(slot.k.as_ref())?, keep(slot.v.as_ref())?);
        if mv.local_start + skip > 0 && k_keep.is_none() {
            return Err(msg("causal kv: write past an empty cache"));
        }
        let join = |prefix: Option<CudaTensor>, new: &CudaTensor| -> Result<CudaTensor> {
            let new = if skip > 0 { new.narrow(2, skip, n - skip)? } else { new.clone() };
            match prefix {
                Some(p) if new.shape[2] > 0 => CudaTensor::cat(&[&p, &new], 2),
                Some(p) => Ok(p),
                None => Ok(new),
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

    /// [`Self::update`] on the persistent buffers: roll in place, write the
    /// block at `[local_start, local_end)`, read the window back.
    fn update_static(
        &self,
        slot: &mut KvLayer,
        mv: KvMove,
        skip: usize,
        k: &CudaTensor,
        v: &CudaTensor,
        current_start: usize,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let n = k.shape[2];
        let [b, h, _, d] = k.shape[..] else {
            return Err(msg(format!("causal kv: keys {:?}", k.shape)));
        };
        if v.shape.len() != 4 || v.shape[..3] != k.shape[..3] {
            return Err(msg(format!("causal kv: keys {:?} values {:?}", k.shape, v.shape)));
        }
        if slot.st.is_none() {
            let cap = self.spec.capacity.max(n);
            let sink = self.spec.sink;
            slot.st = Some(StaticKv {
                k: alloc_like(k, vec![b, h, cap, d], false)?,
                v: alloc_like(v, vec![b, h, cap, v.shape[3]], false)?,
                raw_sink: if self.spec.rebase_sink && sink > 0 {
                    Some(alloc_like(k, vec![b, h, sink, d], true)?)
                } else {
                    None
                },
            });
        }
        let prev = slot.local_end;
        let st = slot.st.as_mut().expect("static kv");
        if st.k.shape[..2] != k.shape[..2]
            || st.k.shape[3] != d
            || st.k.shape[2] < n
            || st.k.is_bf16() != k.is_bf16()
            || st.v.is_bf16() != v.is_bf16()
        {
            return Err(msg(format!(
                "causal kv: static buffers {:?} ({:?}) for keys {:?} ({:?})",
                st.k.shape,
                st.k.dtype(),
                k.shape,
                k.dtype()
            )));
        }
        // Roll: [0, sink) stays, [sink + evicted, prev) moves down to sink.
        let mut valid = prev;
        if mv.evicted > 0 {
            let sink = self.spec.sink.min(prev);
            let tail_start = (sink + mv.evicted).min(prev);
            let tail = prev - tail_start;
            if tail > 0 {
                shift_tokens_down(&mut st.k, tail_start, sink, tail)?;
                shift_tokens_down(&mut st.v, tail_start, sink, tail)?;
            }
            valid = sink + tail;
        }
        if mv.local_start + skip > valid {
            return Err(msg(format!(
                "causal kv: write at {} leaves a gap after {valid} cached tokens",
                mv.local_start + skip
            )));
        }
        copy_tokens(&mut st.k, mv.local_start + skip, k, skip, n - skip)?;
        copy_tokens(&mut st.v, mv.local_start + skip, v, skip, n - skip)?;
        slot.global_end = current_start + n;
        slot.local_end = mv.local_end;
        let len = mv.local_end - mv.window_start;
        Ok((
            st.k.narrow(2, mv.window_start, len)?,
            st.v.narrow(2, mv.window_start, len)?,
        ))
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
            if self.static_mode {
                let (local_end, raw_valid) = (slot.local_end, slot.raw_valid);
                let Some(st) = slot.st.as_mut() else { continue };
                if local_end < sink {
                    continue;
                }
                let raw_buf = st
                    .raw_sink
                    .as_mut()
                    .ok_or_else(|| msg("causal kv: static rebased sink without its buffer"))?;
                if !raw_valid {
                    if neg_orig.is_none() {
                        neg_orig = Some(orig.1.mul_scalar(-1.0));
                    }
                    let neg = neg_orig.as_ref().expect("negated sin");
                    let raw = rope_bhsd_f32(&st.k.narrow(2, 0, sink)?, orig.0, neg)?;
                    copy_tokens(raw_buf, 0, &raw, 0, sink)?;
                }
                let mut head = rope_bhsd_f32(raw_buf, new.0, new.1)?;
                if st.k.is_bf16() {
                    head = head.quantize_bf16()?;
                }
                copy_tokens(&mut st.k, 0, &head, 0, sink)?;
                slot.raw_valid = true;
                slot.sink_at = target;
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
            let mut l = l.lock().expect("causal kv lock");
            // A static cache keeps its buffers (and so its addresses).
            let st = l.st.take();
            *l = KvLayer {
                st,
                ..KvLayer::default()
            };
        }
    }

    /// Bytes of K and V held across all layers (for memory-growth checks).
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                let l = l.lock().expect("causal kv lock");
                let st = l.st.as_ref();
                [
                    l.k.as_ref(),
                    l.v.as_ref(),
                    st.map(|s| &s.k),
                    st.map(|s| &s.v),
                    st.and_then(|s| s.raw_sink.as_ref()),
                ]
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
        for cache in [CausalKvCache::new(spec, 1), CausalKvCache::new_static(spec, 1)] {
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
            // Fourth block, 1-token shifts: the roll moves 2 tokens by 4.
            let (k, _) = cache.update(0, &blk(40.0), &blk(40.0), 12).unwrap();
            assert_eq!(
                k.host_cow().unwrap().to_vec(),
                vec![10.0, 11.0, 32.0, 33.0, 40.0, 41.0, 42.0, 43.0]
            );
        }
    }

    /// The static cache (persistent buffers written in place) is the
    /// allocating cache, bit for bit, through the stream's pattern: every
    /// block run several times (denoising steps, then the context pass),
    /// the window filling, rolling past the sink, and each RoPE policy.
    #[test]
    fn static_cache_is_the_allocating_cache() {
        let (dit, cfg) = tiny_causal_dit();
        let (c, h, w, fpb) = (cfg.in_channels, 4usize, 6usize, cfg.num_frames_per_block);
        let frame_tokens = (h / 2) * (w / 2);
        let text: Vec<f32> = (0..cfg.text_len * cfg.text_dim)
            .map(|i| ((i as f32) * 0.11).cos())
            .collect();
        let text = CudaTensor::from_vec(text, vec![1, cfg.text_len, cfg.text_dim]).unwrap();
        for rope in [KvRope::Absolute, KvRope::Relativistic, KvRope::RebasedSink] {
            let spec = KvSpec::rolling(frame_tokens, 6, 2, rope).unwrap();
            let run = |cache: &CausalKvCache| -> Vec<Vec<u32>> {
                let mut out = Vec::new();
                for blk in 0..7 {
                    let lat: Vec<f32> = (0..c * fpb * h * w)
                        .map(|i| ((i as f32) * 0.37 + blk as f32).sin())
                        .collect();
                    let x = CudaTensor::from_vec(lat, vec![1, c, fpb, h, w]).unwrap();
                    for ts in [750.0f32, 250.0, 0.0] {
                        let t = CudaTensor::from_vec(vec![ts], vec![1]).unwrap();
                        let y = dit.forward_kv(&x, &t, &text, cache, blk * fpb).unwrap();
                        out.push(y.host_cow().unwrap().iter().map(|v| v.to_bits()).collect());
                    }
                }
                out
            };
            let a = run(&CausalKvCache::new(spec, cfg.num_layers));
            let st = CausalKvCache::new_static(spec, cfg.num_layers);
            let b = run(&st);
            assert!(a == b, "{rope:?}: static cache differs");
            assert!(st.is_allocated());
            // After a reset the buffers stay and the rollout repeats exactly.
            st.reset();
            assert!(st.is_empty() && st.is_allocated());
            assert!(run(&st) == b, "{rope:?}: static cache after reset differs");
        }
    }

    /// Keys follow the fill: appends while the window fills, the first roll
    /// takes the un-roped sink, then every block has the same key.
    #[test]
    fn block_keys_settle_once_the_window_is_full() {
        let spec = KvSpec::rolling(10, 6, 2, KvRope::RebasedSink).unwrap();
        let cache = CausalKvCache::new_static(spec, 1);
        let blk = |start: usize| {
            let data: Vec<f32> = (0..40).map(|i| (start + i) as f32).collect();
            CudaTensor::from_vec(data, vec![1, 1, 20, 2]).unwrap()
        };
        let tab = CudaTensor::from_vec(vec![1.0; 40], vec![20, 2]).unwrap();
        let zero = CudaTensor::from_vec(vec![0.0; 40], vec![20, 2]).unwrap();
        let mut keys = Vec::new();
        for b in 0..6 {
            let start = b * 20;
            let target = spec.sink_target(b * 2 + 2);
            let key = cache.block_key(start, 20, target).unwrap();
            keys.push(key);
            if key.rebase {
                cache.rebase_sink(target, (&tab, &zero), (&tab, &zero)).unwrap();
            }
            cache.update(0, &blk(start), &blk(start), start).unwrap();
            // The in-place re-run of the same block (next step).
            let again = cache.block_key(start, 20, target).unwrap();
            assert!(!again.appends && !again.rebase && again.evicted == 0);
        }
        assert_eq!(keys[0].local_end, 0);
        assert!(keys[1].appends && keys[1].evicted == 0 && !keys[1].rebase);
        assert!(keys[3].rebase && keys[3].take_raw && keys[3].evicted == 20);
        assert!(keys[4].rebase && !keys[4].take_raw);
        assert_eq!(keys[4], keys[5]);
        let p = cache.pointers();
        cache.set_pointers(&p).unwrap();
        assert_eq!(cache.pointers(), p);
    }

    /// `advance` (a graph replay's host bookkeeping) moves the pointers as
    /// the real rebase + write do, through the fill, the first roll (which
    /// takes the un-roped sink) and the steady state, several forwards per
    /// block.
    #[test]
    fn advance_moves_the_pointers_as_a_forward_does() {
        let spec = KvSpec::rolling(10, 6, 2, KvRope::RebasedSink).unwrap();
        let real = CausalKvCache::new_static(spec, 2);
        let replay = CausalKvCache::new_static(spec, 2);
        let blk = |start: usize| {
            let data: Vec<f32> = (0..40).map(|i| (start + i) as f32).collect();
            CudaTensor::from_vec(data, vec![1, 1, 20, 2]).unwrap()
        };
        let tab = CudaTensor::from_vec(vec![1.0; 40], vec![20, 2]).unwrap();
        let zero = CudaTensor::from_vec(vec![0.0; 40], vec![20, 2]).unwrap();
        for b in 0..7 {
            let start = b * 20;
            let target = spec.sink_target(b * 2 + 2);
            for _ in 0..3 {
                if target > 0 {
                    real.rebase_sink(target, (&tab, &zero), (&tab, &zero)).unwrap();
                }
                for layer in 0..2 {
                    real.update(layer, &blk(start), &blk(start), start).unwrap();
                }
                if b == 0 {
                    // The buffers exist after a real first write.
                    for layer in 0..2 {
                        replay.update(layer, &blk(start), &blk(start), start).unwrap();
                    }
                } else {
                    replay.advance(start, 20, target).unwrap();
                }
                assert_eq!(real.pointers(), replay.pointers(), "block {b}");
            }
        }
        assert!(CausalKvCache::new(spec, 1).advance(0, 20, 0).is_err());
    }

    /// Keys of `frames` frames from `first` (10 tokens each, one value per
    /// token: `1000 · tag + frame`), `[1, 1, frames·10, 1]`.
    fn tagged(first: usize, frames: usize, tag: f32) -> CudaTensor {
        let data: Vec<f32> = (0..frames * 10).map(|i| 1000.0 * tag + (first + i / 10) as f32).collect();
        CudaTensor::from_vec(data, vec![1, 1, frames * 10, 1]).unwrap()
    }

    fn frames_of(t: &CudaTensor) -> Vec<f32> {
        t.host_cow().unwrap().iter().step_by(10).copied().collect()
    }

    /// LongLive's re-cache on the cache alone (`window 12`, `sink 3`,
    /// 3-frame blocks, as `local_attn_size: 12`, `sink_size: 3`): after 15
    /// frames the cache holds the sink (frames 0-2) and frames 6-14. The
    /// re-cache runs frames 3-14 (12) at frame 3 under the new prompt with
    /// the guard: the sink slots stay, slots 3-11 take the new frames 6-14,
    /// the pointers do not move, and the next block rolls as before.
    #[test]
    fn sink_guard_keeps_the_sink_through_a_recache() {
        for static_mode in [false, true] {
            let spec = KvSpec::rolling(10, 12, 3, KvRope::Absolute).unwrap();
            let cache = if static_mode { CausalKvCache::new_static(spec, 1) } else { CausalKvCache::new(spec, 1) };
            for b in 0..5 {
                let (start, k) = (b * 3, tagged(b * 3, 3, 1.0));
                cache.update(0, &k, &k, start * 10).unwrap();
            }
            let before = cache.pointers();
            assert_eq!((before[0].global_end, before[0].local_end), (150, 120));
            let plan = crate::wan::longlive::recache_plan(15, 12, true).unwrap();
            assert_eq!((plan.start_frame, plan.frames, plan.guard_sink), (3, 12, true));
            let k = tagged(plan.start_frame, plan.frames, 2.0);
            cache.set_sink_guard(plan.guard_sink);
            let (kw, vw) = cache.update(0, &k, &k, plan.start_frame * 10).unwrap();
            cache.set_sink_guard(false);
            assert_eq!(cache.pointers(), before, "static={static_mode}");
            let want: Vec<f32> = [1000.0, 1001.0, 1002.0]
                .into_iter()
                .chain((6..15).map(|f| 2000.0 + f as f32))
                .collect();
            assert_eq!(frames_of(&kw), want, "static={static_mode}");
            assert_eq!(frames_of(&vw), want);
            // The next block: one block evicted after the sink, new-prompt
            // frames 9-14 and the block.
            let k = tagged(15, 3, 3.0);
            let (kw, _) = cache.update(0, &k, &k, 150).unwrap();
            let want: Vec<f32> = [1000.0, 1001.0, 1002.0]
                .into_iter()
                .chain((9..15).map(|f| 2000.0 + f as f32))
                .chain((15..18).map(|f| 3000.0 + f as f32))
                .collect();
            assert_eq!(frames_of(&kw), want, "static={static_mode}");
        }
    }

    /// Before the cache rolls a re-cache starts at frame 0 and rewrites
    /// everything, the sink too (no `is_recompute` at `current_start == 0`);
    /// and the guard is inert on ordinary writes.
    #[test]
    fn early_recache_rewrites_the_sink() {
        for static_mode in [false, true] {
            let spec = KvSpec::rolling(10, 12, 3, KvRope::Absolute).unwrap();
            let cache = if static_mode { CausalKvCache::new_static(spec, 1) } else { CausalKvCache::new(spec, 1) };
            for b in 0..2 {
                let k = tagged(b * 3, 3, 1.0);
                cache.update(0, &k, &k, b * 30).unwrap();
            }
            let plan = crate::wan::longlive::recache_plan(6, 12, true).unwrap();
            assert_eq!((plan.start_frame, plan.frames, plan.guard_sink), (0, 6, false));
            let k = tagged(0, 6, 2.0);
            cache.set_sink_guard(true); // inert at current_start 0
            let (kw, _) = cache.update(0, &k, &k, 0).unwrap();
            assert_eq!(frames_of(&kw), (0..6).map(|f| 2000.0 + f as f32).collect::<Vec<_>>());
            // A new block with the guard on writes whole (not a recompute).
            let k = tagged(6, 3, 3.0);
            let (kw, _) = cache.update(0, &k, &k, 60).unwrap();
            cache.set_sink_guard(false);
            assert_eq!(frames_of(&kw)[6..], [3006.0, 3007.0, 3008.0]);
        }
    }

    /// The re-cache through the tiny causal DiT: a rollout under prompt A,
    /// a switch to B with the re-cache, more blocks. The allocating and the
    /// static caches agree bit for bit (graph mode runs the static one), the
    /// guarded sink keys are untouched in every layer, and the re-cache
    /// changes what the next block makes (against keeping the cache).
    #[test]
    fn recache_through_the_dit() {
        let (dit, cfg) = tiny_causal_dit();
        let (c, h, w, fpb) = (cfg.in_channels, 4usize, 6usize, cfg.num_frames_per_block);
        let frame_tokens = (h / 2) * (w / 2);
        let text = |phase: f32| {
            let v: Vec<f32> = (0..cfg.text_len * cfg.text_dim).map(|i| ((i as f32) * 0.11 + phase).cos()).collect();
            CudaTensor::from_vec(v, vec![1, cfg.text_len, cfg.text_dim]).unwrap()
        };
        let (a, b) = (text(0.0), text(1.3));
        let (window, sink) = (6usize, 2usize);
        let lat = |blk: usize| {
            let v: Vec<f32> = (0..c * fpb * h * w).map(|i| ((i as f32) * 0.37 + blk as f32).sin()).collect();
            CudaTensor::from_vec(v, vec![1, c, fpb, h, w]).unwrap()
        };
        let t0 = CudaTensor::from_vec(vec![0.0], vec![1]).unwrap();
        let t5 = CudaTensor::from_vec(vec![500.0], vec![1]).unwrap();
        for rope in [KvRope::Absolute, KvRope::RebasedSink, KvRope::Relativistic] {
            let spec = KvSpec::rolling(frame_tokens, window, sink, rope).unwrap();
            let run = |cache: &CausalKvCache, recache: bool| -> (Vec<u32>, Vec<f32>, Vec<f32>) {
                let mut hist = Vec::new();
                for blk in 0..5 {
                    dit.forward_kv(&lat(blk), &t5, &a, cache, blk * fpb).unwrap();
                    dit.forward_kv(&lat(blk), &t0, &a, cache, blk * fpb).unwrap();
                    hist.push(lat(blk));
                }
                let sink_keys = || -> Vec<f32> {
                    let l = cache.layers[1].lock().unwrap();
                    let k = l.k.clone().unwrap_or_else(|| l.st.as_ref().expect("written cache").k.clone());
                    k.narrow(2, 0, sink * frame_tokens).unwrap().host_cow().unwrap().into_owned()
                };
                let sink_before = sink_keys();
                let mut sink_after = sink_before.clone();
                if recache {
                    let current = 5 * fpb;
                    let plan = crate::wan::longlive::recache_plan(current, window, true).unwrap();
                    assert!(plan.guard_sink);
                    let all = CudaTensor::cat(&hist.iter().collect::<Vec<_>>(), 2).unwrap();
                    let x = all.narrow(2, current - plan.frames, plan.frames).unwrap();
                    let before = cache.pointers();
                    cache.set_sink_guard(true);
                    dit.forward_kv(&x, &t0, &b, cache, plan.start_frame).unwrap();
                    cache.set_sink_guard(false);
                    assert_eq!(cache.pointers(), before, "{rope:?}");
                    sink_after = sink_keys();
                }
                let mut out = Vec::new();
                for blk in 5..7 {
                    let y = dit.forward_kv(&lat(blk), &t5, &b, cache, blk * fpb).unwrap();
                    out.extend(y.host_cow().unwrap().iter().map(|v| v.to_bits()));
                    dit.forward_kv(&lat(blk), &t0, &b, cache, blk * fpb).unwrap();
                }
                (out, sink_before, sink_after)
            };
            let (dyn_out, s0, s1) = run(&CausalKvCache::new(spec, cfg.num_layers), true);
            assert_eq!(s0, s1, "{rope:?}: the guarded sink moved");
            let (st_out, ..) = run(&CausalKvCache::new_static(spec, cfg.num_layers), true);
            assert!(dyn_out == st_out, "{rope:?}: static cache differs through the re-cache");
            let (keep_out, ..) = run(&CausalKvCache::new(spec, cfg.num_layers), false);
            assert!(keep_out != dyn_out, "{rope:?}: the re-cache changed nothing");
        }
    }
}
