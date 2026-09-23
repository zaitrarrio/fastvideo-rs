//! LongLive causal KV cache.
//!
//! Rolling window plus a sink prefix, from `causal_model.py`
//! `_update_cache_and_get_kv` / `_apply_cache_updates`. The multi-shot pinned
//! region is not part of this cache. When a scale rule is set, the resident
//! slots are packed NVFP4 rows; a push dequants the attended span before it
//! returns.

use crate::nvfp4::{self, ScaleRule, BLOCK};

/// Geometry of one layer's cache. Sequence axis is tokens, not frames.
#[derive(Debug, Clone, Copy)]
pub struct ArKvSpec {
    pub heads: usize,
    pub dim: usize,
    pub capacity: usize,
    pub sink_tokens: usize,
    /// Sliding window in tokens. `0` attends everything stored.
    pub max_attention: usize,
}

#[derive(Debug, Clone)]
struct PackedRow {
    packed: Vec<u8>,
    scales: Vec<u8>,
    amax: f32,
}

#[derive(Debug, Clone)]
enum Slot {
    /// `[heads, dim]` row-major.
    Dense(Vec<f32>),
    /// One packed row per head.
    Nvfp4(Vec<PackedRow>),
}

/// Per-layer autoregressive K/V. Tokens append; a full buffer rolls everything
/// after the sink toward the front.
#[derive(Debug, Clone)]
pub struct ArKvCache {
    spec: ArKvSpec,
    local_end: usize,
    global_end: usize,
    k: Vec<Slot>,
    v: Vec<Slot>,
    rule: Option<ScaleRule>,
}

impl ArKvCache {
    pub fn open(spec: ArKvSpec, rule: Option<ScaleRule>) -> Result<Self, String> {
        if spec.heads == 0 || spec.dim == 0 || spec.capacity == 0 {
            return Err("ar kv: heads, dim, and capacity must be non-zero".into());
        }
        if spec.sink_tokens > spec.capacity {
            return Err(format!(
                "ar kv: sink {} exceeds capacity {}",
                spec.sink_tokens, spec.capacity
            ));
        }
        if rule.is_some() && !spec.dim.is_multiple_of(BLOCK) {
            return Err(format!(
                "ar kv: head dim {} is not a multiple of {BLOCK}",
                spec.dim
            ));
        }
        let empty = || Slot::Dense(vec![0.0; spec.heads * spec.dim]);
        Ok(Self {
            k: vec![empty(); spec.capacity],
            v: vec![empty(); spec.capacity],
            spec,
            local_end: 0,
            global_end: 0,
            rule,
        })
    }

    pub fn local_end(&self) -> usize {
        self.local_end
    }

    pub fn global_end(&self) -> usize {
        self.global_end
    }

    pub fn quantized(&self) -> bool {
        self.rule.is_some()
    }

    /// Append `[n_new, heads, dim]` K and V. Returns the attended span in the
    /// same layout, already dequantized when the cache is packed.
    pub fn push(&mut self, k_new: &[f32], v_new: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        let width = self.spec.heads * self.spec.dim;
        let n_new = k_new.len() / width;
        if n_new == 0 || k_new.len() != n_new * width || v_new.len() != n_new * width {
            return Err(format!(
                "ar kv: new K {} V {} is not [n, {}, {}]",
                k_new.len(),
                v_new.len(),
                self.spec.heads,
                self.spec.dim
            ));
        }
        if n_new > self.spec.capacity {
            return Err(format!(
                "ar kv: chunk {n_new} exceeds capacity {}",
                self.spec.capacity
            ));
        }
        let sink = self.spec.sink_tokens;
        let (local_end, local_start, roll) =
            plan_append(self.spec.capacity, sink, self.local_end, n_new)?;
        if let Some((src, len)) = roll {
            roll_slots(&mut self.k, sink, src, len);
            roll_slots(&mut self.v, sink, src, len);
        }
        let k_slots = pack_chunk(
            k_new,
            n_new,
            self.spec.heads,
            self.spec.dim,
            self.rule,
            true,
        )?;
        let v_slots = pack_chunk(
            v_new,
            n_new,
            self.spec.heads,
            self.spec.dim,
            self.rule,
            false,
        )?;
        for i in 0..n_new {
            self.k[local_start + i] = k_slots[i].clone();
            self.v[local_start + i] = v_slots[i].clone();
        }
        self.local_end = local_end;
        self.global_end += n_new;
        let parts = attend_parts(sink, self.spec.max_attention, local_end);
        Ok((
            gather(&self.k, &parts, self.spec.dim, self.rule)?,
            gather(&self.v, &parts, self.spec.dim, self.rule)?,
        ))
    }
}

fn plan_append(
    capacity: usize,
    sink: usize,
    local_end: usize,
    n_new: usize,
) -> Result<(usize, usize, Option<(usize, usize)>), String> {
    let need_roll = n_new + local_end > capacity;
    let (local_end_new, roll) = if need_roll {
        let evicted = n_new + local_end - capacity;
        let rolled = local_end.saturating_sub(evicted + sink);
        let local_end_new = local_end + n_new - evicted;
        let roll = (rolled > 0).then_some((sink + evicted, rolled));
        (local_end_new, roll)
    } else {
        (local_end + n_new, None)
    };
    let local_start = local_end_new - n_new;
    if roll.is_some() && local_start < sink {
        return Err(format!(
            "ar kv: write at {local_start} overlaps sink {sink}"
        ));
    }
    if local_end_new > capacity {
        return Err(format!(
            "ar kv: fill {local_end_new} exceeds capacity {capacity}"
        ));
    }
    Ok((local_end_new, local_start, roll))
}

/// Ranges in cache order. A sink that has slid out of the window is prepended.
fn attend_parts(sink: usize, max_attention: usize, local_end: usize) -> Vec<(usize, usize)> {
    if local_end == 0 {
        return Vec::new();
    }
    if max_attention == 0 {
        return vec![(0, local_end)];
    }
    let window_start = local_end.saturating_sub(max_attention);
    if sink > 0 && window_start > 0 {
        let local = max_attention.saturating_sub(sink);
        let start = sink.max(local_end.saturating_sub(local));
        let mut parts = vec![(0, sink.min(local_end))];
        if start < local_end {
            parts.push((start, local_end));
        }
        parts
    } else {
        vec![(window_start, local_end)]
    }
}

fn roll_slots(slots: &mut [Slot], dst: usize, src: usize, len: usize) {
    let moved: Vec<Slot> = slots[src..src + len].to_vec();
    for (i, slot) in moved.into_iter().enumerate() {
        slots[dst + i] = slot;
    }
}

fn pack_chunk(
    values: &[f32],
    n_new: usize,
    heads: usize,
    dim: usize,
    rule: Option<ScaleRule>,
    smooth: bool,
) -> Result<Vec<Slot>, String> {
    let mut rows = values.to_vec();
    if let Some(rule) = rule {
        if smooth {
            nvfp4::k_smooth(&mut rows, n_new * heads, dim);
        }
        let qt = nvfp4::quantize(&rows, n_new * heads, dim, rule)?;
        let scales_per = dim / BLOCK;
        let packed_per = dim / 2;
        let mut slots = Vec::with_capacity(n_new);
        for t in 0..n_new {
            let mut heads_rows = Vec::with_capacity(heads);
            for h in 0..heads {
                let r = t * heads + h;
                heads_rows.push(PackedRow {
                    packed: qt.packed[r * packed_per..(r + 1) * packed_per].to_vec(),
                    scales: qt.scales[r * scales_per..(r + 1) * scales_per].to_vec(),
                    amax: qt.amax,
                });
            }
            slots.push(Slot::Nvfp4(heads_rows));
        }
        Ok(slots)
    } else {
        let width = heads * dim;
        Ok((0..n_new)
            .map(|t| Slot::Dense(rows[t * width..(t + 1) * width].to_vec()))
            .collect())
    }
}

fn gather(
    slots: &[Slot],
    parts: &[(usize, usize)],
    dim: usize,
    rule: Option<ScaleRule>,
) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for &(start, end) in parts {
        for slot in &slots[start..end] {
            match (slot, rule) {
                (Slot::Dense(row), None) => out.extend_from_slice(row),
                (Slot::Nvfp4(rows), Some(rule)) => {
                    for row in rows {
                        out.extend(dequant_row(row, dim, rule)?);
                    }
                }
                _ => return Err("ar kv: slot does not match the cache rule".into()),
            }
        }
    }
    Ok(out)
}

fn dequant_row(row: &PackedRow, dim: usize, rule: ScaleRule) -> Result<Vec<f32>, String> {
    let scales_per = dim / BLOCK;
    if row.packed.len() != dim / 2 || row.scales.len() != scales_per {
        return Err(format!(
            "ar kv: packed {} scales {} for dim {dim}",
            row.packed.len(),
            row.scales.len()
        ));
    }
    let mut out = vec![0.0f32; dim];
    for b in 0..scales_per {
        let scale = row.scales[b];
        for i in 0..(BLOCK / 2) {
            let byte = row.packed[b * (BLOCK / 2) + i];
            out[b * BLOCK + 2 * i] = nvfp4::dequant_elem(byte & 0x0F, scale, row.amax, rule);
            out[b * BLOCK + 2 * i + 1] = nvfp4::dequant_elem(byte >> 4, scale, row.amax, rule);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(capacity: usize, sink: usize, window: usize) -> ArKvSpec {
        ArKvSpec {
            heads: 1,
            dim: 16,
            capacity,
            sink_tokens: sink,
            max_attention: window,
        }
    }

    fn seq(n: usize, scale: f32) -> Vec<f32> {
        (0..n * 16)
            .map(|i| ((i as f32) * 0.17 + scale).sin())
            .collect()
    }

    #[test]
    fn append_without_roll_returns_the_whole_cache() {
        let mut cache = ArKvCache::open(spec(8, 0, 0), None).unwrap();
        let (k, v) = cache.push(&seq(2, 0.2), &seq(2, 1.0)).unwrap();
        assert_eq!(k, seq(2, 0.2));
        assert_eq!(v, seq(2, 1.0));
        assert_eq!(cache.local_end(), 2);
        assert_eq!(cache.global_end(), 2);
        let (k2, _) = cache.push(&seq(2, 0.4), &seq(2, 1.2)).unwrap();
        assert_eq!(k2.len(), 4 * 16);
        assert_eq!(&k2[..32], &seq(2, 0.2));
        assert_eq!(cache.global_end(), 4);
    }

    #[test]
    fn roll_keeps_the_sink_and_drops_the_oldest_tail() {
        let mut cache = ArKvCache::open(spec(4, 1, 0), None).unwrap();
        cache.push(&seq(2, 0.1), &seq(2, 0.1)).unwrap();
        cache.push(&seq(2, 0.2), &seq(2, 0.2)).unwrap();
        let (k, _) = cache.push(&seq(2, 0.3), &seq(2, 0.3)).unwrap();
        assert_eq!(cache.local_end(), 4);
        assert_eq!(cache.global_end(), 6);
        assert_eq!(&k[..16], &seq(2, 0.1)[..16]);
        assert_eq!(&k[16..32], &seq(2, 0.2)[16..]);
        assert_eq!(&k[32..], &seq(2, 0.3));
    }

    #[test]
    fn window_prepends_a_sink_that_fell_outside() {
        let mut cache = ArKvCache::open(spec(8, 1, 2), None).unwrap();
        cache.push(&seq(4, 0.1), &seq(4, 0.1)).unwrap();
        let (k, _) = cache.push(&seq(2, 0.5), &seq(2, 0.5)).unwrap();
        assert_eq!(k.len(), 2 * 16);
        assert_eq!(&k[..16], &seq(4, 0.1)[..16]);
        assert_eq!(&k[16..], &seq(2, 0.5)[16..]);
    }

    #[test]
    fn packed_push_dequants_to_the_smoothed_key() {
        let mut cache = ArKvCache::open(spec(8, 0, 0), Some(ScaleRule::Mse)).unwrap();
        assert!(cache.quantized());
        let raw_k = seq(2, 0.3);
        let raw_v = seq(2, 0.8);
        let (k, v) = cache.push(&raw_k, &raw_v).unwrap();
        let mut smoothed = raw_k.clone();
        nvfp4::k_smooth(&mut smoothed, 2, 16);
        let want_k = nvfp4::reconstruct(&smoothed, 2, 16, ScaleRule::Mse).unwrap();
        let want_v = nvfp4::reconstruct(&raw_v, 2, 16, ScaleRule::Mse).unwrap();
        for (a, b) in want_k.iter().zip(k.iter()) {
            assert!((a - b).abs() <= 1e-5, "{a} vs {b}");
        }
        for (a, b) in want_v.iter().zip(v.iter()) {
            assert!((a - b).abs() <= 1e-5, "{a} vs {b}");
        }
        assert!(matches!(cache.k[0], Slot::Nvfp4(_)));
    }

    #[test]
    fn a_later_chunk_does_not_rescale_the_first() {
        let mut cache = ArKvCache::open(spec(8, 0, 0), Some(ScaleRule::Mse)).unwrap();
        let first = seq(1, 0.2);
        let (k1, _) = cache.push(&first, &first).unwrap();
        let mut big = seq(1, 4.0);
        for v in &mut big {
            *v *= 50.0;
        }
        let (k2, _) = cache.push(&big, &big).unwrap();
        assert_eq!(k2.len(), 2 * 16);
        for (a, b) in k1.iter().zip(k2.iter().take(16)) {
            assert!((a - b).abs() <= 1e-5);
        }
    }
}
