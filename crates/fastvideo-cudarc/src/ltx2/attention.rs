//! `LTX2Attention` and the feed-forward it is paired with — the two layers the
//! connectors and every DiT block are made of.
//!
//! One attention class serves five roles (self, text cross, audio→video,
//! video→audio, connector). They differ only in widths and in which rotary
//! table, if any, rotates q and k, so that is all [`Attention::forward`] takes.
//!
//! Two conventions are easy to get wrong:
//!
//! * `qk_norm = "rms_norm_across_heads"`: q and k are RMS-normalised over the
//!   *whole* inner width (one statistic per token across all heads) with a
//!   learned `[inner]` weight, before the head split. It is not a per-head norm.
//! * the rotary is "split": rotate_half inside each head with a table that
//!   *differs per head*. [`DeviceRope`] folds the head axis into the row axis
//!   of `rope_half`'s `[rows, D]` table — `[B, H, S, D]` viewed as
//!   `[B, 1, H·S, D]` is the same memory — so no kernel is needed for it.

use fastvideo_models::ltx2::SplitRope;

use crate::wan::nn::{scaled_dot_product_attention, Linear};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::keys::Keys;
use super::{msg, pinned};

/// A [`SplitRope`] on the device, in `rope_half`'s layout: `[H·S, D]` cos and
/// sin, head-major, each pair's value duplicated across the two halves.
#[derive(Debug, Clone)]
pub struct DeviceRope {
    cos: CudaTensor,
    sin: CudaTensor,
    heads: usize,
    tokens: usize,
    head_dim: usize,
}

impl DeviceRope {
    pub fn upload(table: &SplitRope) -> Result<Self> {
        let (cos, sin) = table.rotate_half_tables();
        let (rows, d) = (table.heads * table.tokens, table.half * 2);
        Ok(Self {
            cos: pinned(cos, vec![rows, d])?,
            sin: pinned(sin, vec![rows, d])?,
            heads: table.heads,
            tokens: table.tokens,
            head_dim: d,
        })
    }

    /// Rotate `[1, H, S, D]`. Batch 1 only: with the head axis folded into the
    /// rows, a second batch element would need the table repeated.
    pub fn apply(&self, x: &CudaTensor) -> Result<CudaTensor> {
        if x.shape != [1, self.heads, self.tokens, self.head_dim] {
            return Err(msg(format!(
                "split rope built for [1, {}, {}, {}] applied to {:?}",
                self.heads, self.tokens, self.head_dim, x.shape
            )));
        }
        x.reshape(vec![1, 1, self.heads * self.tokens, self.head_dim])?
            .rope_half(&self.cos, &self.sin)?
            .reshape(x.shape.clone())
    }
}

/// Widths of one attention layer. `query_dim` is also the output width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionDims {
    pub query_dim: usize,
    /// Width of the key/value source; equals `query_dim` for self-attention.
    pub context_dim: usize,
    pub heads: usize,
    pub head_dim: usize,
}

impl AttentionDims {
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }
}

pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    to_gate_logits: Option<Linear>,
    norm_q: CudaTensor,
    norm_k: CudaTensor,
    dims: AttentionDims,
    eps: f32,
}

impl Attention {
    /// `prefix` is the diffusers module path (`transformer_blocks.3.attn1`);
    /// `keys` spells it for the checkpoint at hand.
    pub fn load(
        map: &WeightMap,
        keys: &Keys,
        prefix: &str,
        dims: AttentionDims,
        eps: f32,
        gated: bool,
    ) -> Result<Self> {
        let inner = dims.inner();
        let lin = |name: &str, i: usize, o: usize| {
            Linear::load(map, &keys.key(&format!("{prefix}.{name}")), i, o, true)
        };
        let norm = |name: &str| -> Result<CudaTensor> {
            let mut w =
                cuda_tensor_shaped(map, &keys.key(&format!("{prefix}.{name}.weight")), &[inner])?;
            w.pin_device()?;
            Ok(w)
        };
        let to_gate_logits = if gated {
            Some(lin("to_gate_logits", dims.query_dim, dims.heads)?)
        } else {
            None
        };
        Ok(Self {
            to_q: lin("to_q", dims.query_dim, inner)?,
            to_k: lin("to_k", dims.context_dim, inner)?,
            to_v: lin("to_v", dims.context_dim, inner)?,
            to_out: lin("to_out.0", inner, dims.query_dim)?,
            to_gate_logits,
            norm_q: norm("norm_q")?,
            norm_k: norm("norm_k")?,
            dims,
            eps,
        })
    }

    /// Per-head `2·σ(logits)` on SDPA output, diffusers `LTX2AudioVideoAttnProcessor`.
    fn apply_head_gates(&self, out: &CudaTensor, gate_logits: &CudaTensor) -> Result<CudaTensor> {
        let heads = self.dims.heads;
        let gates = gate_logits.try_sigmoid()?.try_mul_scalar(2.0)?;
        let gates = gates
            .permute(&[0, 2, 1])?
            .reshape(vec![1, heads, gates.shape[1], 1])?;
        out.mul(&gates)
    }

    /// `x`: `[1, Sq, query_dim]`. `context`: `[1, Sk, context_dim]`, or `None`
    /// for self-attention. `q_rope` rotates q; k is rotated by `k_rope`, or by
    /// `q_rope` when there is none (self-attention) — and not at all when
    /// `q_rope` is `None` (text cross-attention). No mask: every mask LTX-2.0
    /// builds for these layers is all ones.
    pub fn forward(
        &self,
        x: &CudaTensor,
        context: Option<&CudaTensor>,
        q_rope: Option<&DeviceRope>,
        k_rope: Option<&DeviceRope>,
    ) -> Result<CudaTensor> {
        let (heads, d) = (self.dims.heads, self.dims.head_dim);
        let gate_logits = match &self.to_gate_logits {
            Some(l) => Some(l.forward(x)?),
            None => None,
        };
        let ctx = context.unwrap_or(x);
        let q = self
            .to_q
            .forward(x)?
            .rms_norm(&self.norm_q, self.eps)?
            .split_heads_bhsd(0, heads, d)?;
        let k = self
            .to_k
            .forward(ctx)?
            .rms_norm(&self.norm_k, self.eps)?
            .split_heads_bhsd(0, heads, d)?;
        let v = self.to_v.forward(ctx)?.split_heads_bhsd(0, heads, d)?;
        let (q, k) = match q_rope {
            Some(rope) => (rope.apply(&q)?, k_rope.unwrap_or(rope).apply(&k)?),
            None => (q, k),
        };
        let out = scaled_dot_product_attention(&q, &k, &v, Some((d as f32).powf(-0.5)))?;
        let merged = match gate_logits {
            Some(logits) => self.apply_head_gates(&out, &logits)?.merge_heads()?,
            None => out.merge_heads()?,
        };
        self.to_out.forward(&merged)
    }
}

/// diffusers `FeedForward(dim, activation_fn="gelu-approximate")`:
/// `Linear(d, 4d)` → tanh-GELU → `Linear(4d, d)`, keys `net.0.proj` / `net.2`.
pub struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    pub fn load(
        map: &WeightMap,
        keys: &Keys,
        prefix: &str,
        dim: usize,
        inner: usize,
        has_bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            up: Linear::load(
                map,
                &keys.key(&format!("{prefix}.net.0.proj")),
                dim,
                inner,
                has_bias,
            )?,
            down: Linear::load(
                map,
                &keys.key(&format!("{prefix}.net.2")),
                inner,
                dim,
                has_bias,
            )?,
        })
    }

    pub fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.down.forward(&self.up.forward_gelu(x)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ltx2::keys::Layout;

    /// Deterministic weights by key: norm weights near 1, the rest small and
    /// signed so activations stay tame through several layers.
    pub(crate) fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key
                .bytes()
                .fold(11u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let n: usize = shape.iter().product();
            (0..n)
                .map(|i| {
                    let v = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32
                        / (1u32 << 24) as f32;
                    if key.contains("norm") {
                        0.5 + v
                    } else {
                        (v - 0.5) * 0.6
                    }
                })
                .collect()
        })
    }

    pub(crate) fn get(map: &WeightMap, key: &str, shape: &[usize]) -> Vec<f32> {
        cuda_tensor_shaped(map, key, shape)
            .unwrap()
            .host_cow()
            .unwrap()
            .into_owned()
    }

    /// `y = W x + b`, `W` row-major `[o, i]`.
    pub(crate) fn linear(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        b.iter()
            .enumerate()
            .map(|(r, b)| {
                b + x
                    .iter()
                    .enumerate()
                    .map(|(c, a)| a * w[r * x.len() + c])
                    .sum::<f32>()
            })
            .collect()
    }

    pub(crate) fn rms(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let ms = x.iter().map(|a| a * a).sum::<f32>() / x.len() as f32;
        x.iter()
            .enumerate()
            .map(|(i, a)| a / (ms + eps).sqrt() * w.map_or(1.0, |w| w[i]))
            .collect()
    }

    /// The reference attention written as loops over tokens, heads and pairs:
    /// full-width QK norm, per-head half-width rotary tables, plain softmax.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_reference(
        map: &WeightMap,
        prefix: &str,
        dims: AttentionDims,
        x: &[Vec<f32>],
        ctx: &[Vec<f32>],
        q_rope: Option<&SplitRope>,
        k_rope: Option<&SplitRope>,
        gated: bool,
    ) -> Vec<Vec<f32>> {
        let (inner, h, d) = (dims.inner(), dims.heads, dims.head_dim);
        let w = |n: &str, o: usize, i: usize| {
            (
                get(map, &format!("{prefix}.{n}.weight"), &[o, i]),
                get(map, &format!("{prefix}.{n}.bias"), &[o]),
            )
        };
        let (wq, wk, wv, wo) = (
            w("to_q", inner, dims.query_dim),
            w("to_k", inner, dims.context_dim),
            w("to_v", inner, dims.context_dim),
            w("to_out.0", dims.query_dim, inner),
        );
        let gate_w = gated.then(|| w("to_gate_logits", dims.heads, dims.query_dim));
        let (nq, nk) = (
            get(map, &format!("{prefix}.norm_q.weight"), &[inner]),
            get(map, &format!("{prefix}.norm_k.weight"), &[inner]),
        );
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        let rotate = |v: &[f32], rope: Option<&SplitRope>, tok: usize| -> Vec<f32> {
            let Some(r) = rope else { return v.to_vec() };
            let mut out = v.to_vec();
            for head in 0..h {
                for j in 0..r.half {
                    let at = (head * r.tokens + tok) * r.half + j;
                    let (c, s) = (r.cos[at], r.sin[at]);
                    let (a, b) = (v[head * d + j], v[head * d + r.half + j]);
                    out[head * d + j] = a * c - b * s;
                    out[head * d + r.half + j] = b * c + a * s;
                }
            }
            out
        };
        let q: Vec<Vec<f32>> = x
            .iter()
            .enumerate()
            .map(|(t, v)| rotate(&rms(&linear(v, &wq.0, &wq.1), Some(&nq), 1e-6), q_rope, t))
            .collect();
        let k: Vec<Vec<f32>> = ctx
            .iter()
            .enumerate()
            .map(|(t, v)| {
                rotate(
                    &rms(&linear(v, &wk.0, &wk.1), Some(&nk), 1e-6),
                    if q_rope.is_some() {
                        k_rope.or(q_rope)
                    } else {
                        None
                    },
                    t,
                )
            })
            .collect();
        let v: Vec<Vec<f32>> = ctx.iter().map(|c| linear(c, &wv.0, &wv.1)).collect();
        q.iter()
            .enumerate()
            .map(|(tok, qi)| {
                let mut merged = vec![0f32; inner];
                let gates: Vec<f32> = gate_w
                    .as_ref()
                    .map(|(gw, gb)| {
                        let logits = linear(&x[tok], gw, gb);
                        logits.iter().map(|l| 2.0 * sigmoid(*l)).collect()
                    })
                    .unwrap_or_else(|| vec![1.0; h]);
                for head in 0..h {
                    let span = head * d..(head + 1) * d;
                    let scores: Vec<f32> = k
                        .iter()
                        .map(|kj| {
                            qi[span.clone()]
                                .iter()
                                .zip(&kj[span.clone()])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                / (d as f32).sqrt()
                        })
                        .collect();
                    let mx = scores.iter().copied().fold(f32::MIN, f32::max);
                    let z: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                    let g = gates[head];
                    for (j, s) in scores.iter().enumerate() {
                        let p = (s - mx).exp() / z;
                        for c in 0..d {
                            merged[head * d + c] += p * v[j][head * d + c] * g;
                        }
                    }
                }
                linear(&merged, &wo.0, &wo.1)
            })
            .collect()
    }

    pub(crate) fn rows(t: &CudaTensor, width: usize) -> Vec<Vec<f32>> {
        t.host_cow()
            .unwrap()
            .chunks_exact(width)
            .map(<[f32]>::to_vec)
            .collect()
    }

    pub(crate) fn tokens(n: usize, width: usize, k: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|t| {
                (0..width)
                    .map(|c| ((t * width + c) as f32 * k).sin())
                    .collect()
            })
            .collect()
    }

    pub(crate) fn tensor(rows: &[Vec<f32>]) -> CudaTensor {
        CudaTensor::from_vec(rows.concat(), vec![1, rows.len(), rows[0].len()]).unwrap()
    }

    pub(crate) fn assert_close(got: &[Vec<f32>], want: &[Vec<f32>], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: token count");
        for (t, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.len(), w.len(), "{what}: width");
            for (c, (a, b)) in g.iter().zip(w).enumerate() {
                assert!((a - b).abs() <= tol, "{what}: token {t} ch {c}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn self_attention_with_a_per_head_table_matches_a_loop_reference() {
        let dims = AttentionDims {
            query_dim: 12,
            context_dim: 12,
            heads: 3,
            head_dim: 4,
        };
        let map = weights();
        let attn = Attention::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.attn1",
            dims,
            1e-6,
            false,
        )
        .unwrap();
        let x = tokens(5, 12, 0.37);
        // Three axes over a 12-wide table: 2 freqs per axis, no pad; the three
        // heads get different slices of it.
        let fr: Vec<f32> = (0..15).map(|i| (i as f32 * 0.13).fract()).collect();
        let table = SplitRope::from_fractions(&fr, 3, 12, 3, 10_000.0);
        let rope = DeviceRope::upload(&table).unwrap();
        let got = attn.forward(&tensor(&x), None, Some(&rope), None).unwrap();
        let want = attention_reference(&map, "blk.attn1", dims, &x, &x, Some(&table), None, false);
        assert_close(&rows(&got, 12), &want, 2e-5, "self attention");
        // The table really is per head: rotating every head with head 0's
        // slice must give a different answer.
        let mut same = table.clone();
        for h in 1..3 {
            let (head0, rest) = same.cos.split_at_mut(h * 5 * 2);
            rest[..10].copy_from_slice(&head0[..10]);
            let (head0, rest) = same.sin.split_at_mut(h * 5 * 2);
            rest[..10].copy_from_slice(&head0[..10]);
        }
        let other = attn
            .forward(
                &tensor(&x),
                None,
                Some(&DeviceRope::upload(&same).unwrap()),
                None,
            )
            .unwrap();
        assert!(rows(&other, 12)
            .concat()
            .iter()
            .zip(want.concat())
            .any(|(a, b)| (a - b).abs() > 1e-3));
    }

    #[test]
    fn cross_attention_rotates_each_side_with_its_own_table_and_lengths_may_differ() {
        // Query stream 16 wide, context 8 wide, attention in the context's
        // head layout — the audio→video shape in miniature.
        let dims = AttentionDims {
            query_dim: 16,
            context_dim: 8,
            heads: 2,
            head_dim: 4,
        };
        let map = weights();
        let attn = Attention::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.a2v",
            dims,
            1e-6,
            false,
        )
        .unwrap();
        let (x, ctx) = (tokens(6, 16, 0.21), tokens(3, 8, 0.53));
        let qt = SplitRope::from_fractions(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 1, 8, 2, 10_000.0);
        let kt = SplitRope::from_fractions(&[0.05, 0.25, 0.45], 1, 8, 2, 10_000.0);
        let got = attn
            .forward(
                &tensor(&x),
                Some(&tensor(&ctx)),
                Some(&DeviceRope::upload(&qt).unwrap()),
                Some(&DeviceRope::upload(&kt).unwrap()),
            )
            .unwrap();
        assert_eq!(got.shape, vec![1, 6, 16]);
        let want =
            attention_reference(&map, "blk.a2v", dims, &x, &ctx, Some(&qt), Some(&kt), false);
        assert_close(&rows(&got, 16), &want, 2e-5, "a2v attention");
    }

    #[test]
    fn text_cross_attention_is_not_rotated() {
        let dims = AttentionDims {
            query_dim: 8,
            context_dim: 8,
            heads: 2,
            head_dim: 4,
        };
        let map = weights();
        let attn = Attention::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.attn2",
            dims,
            1e-6,
            false,
        )
        .unwrap();
        let (x, ctx) = (tokens(4, 8, 0.4), tokens(7, 8, 0.9));
        let got = attn
            .forward(&tensor(&x), Some(&tensor(&ctx)), None, None)
            .unwrap();
        let want = attention_reference(&map, "blk.attn2", dims, &x, &ctx, None, None, false);
        assert_close(&rows(&got, 8), &want, 2e-5, "text cross attention");
    }

    #[test]
    fn a_rope_built_for_another_length_is_refused() {
        let table = SplitRope::from_fractions(&[0.1, 0.2], 1, 8, 2, 10_000.0);
        let rope = DeviceRope::upload(&table).unwrap();
        let err = rope.apply(&CudaTensor::zeros(&[1, 2, 3, 4])).unwrap_err();
        assert!(err.to_string().contains("split rope built for"), "{err}");
    }

    #[test]
    fn gated_attention_matches_two_sigmoid_per_head_reference() {
        let dims = AttentionDims {
            query_dim: 12,
            context_dim: 12,
            heads: 3,
            head_dim: 4,
        };
        let map = weights();
        let attn = Attention::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.gattn",
            dims,
            1e-6,
            true,
        )
        .unwrap();
        let x = tokens(4, 12, 0.31);
        let got = attn.forward(&tensor(&x), None, None, None).unwrap();
        let want = attention_reference(&map, "blk.gattn", dims, &x, &x, None, None, true);
        assert_close(&rows(&got, 12), &want, 2e-5, "gated self attention");
    }

    #[test]
    fn feed_forward_loads_without_bias_when_requested() {
        let map = weights();
        let err = FeedForward::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.nobias_ff",
            6,
            24,
            false,
        );
        assert!(err.is_ok());
    }

    #[test]
    fn feed_forward_is_linear_tanh_gelu_linear() {
        let map = weights();
        let ff = FeedForward::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.ff",
            6,
            24,
            true,
        )
        .unwrap();
        let x = tokens(3, 6, 0.77);
        let got = ff.forward(&tensor(&x)).unwrap();
        let (w0, b0) = (
            get(&map, "blk.ff.net.0.proj.weight", &[24, 6]),
            get(&map, "blk.ff.net.0.proj.bias", &[24]),
        );
        let (w2, b2) = (
            get(&map, "blk.ff.net.2.weight", &[6, 24]),
            get(&map, "blk.ff.net.2.bias", &[6]),
        );
        let gelu = |v: f32| {
            0.5 * v
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v * v * v)).tanh())
        };
        let want: Vec<Vec<f32>> = x
            .iter()
            .map(|v| {
                linear(
                    &linear(v, &w0, &b0)
                        .iter()
                        .map(|a| gelu(*a))
                        .collect::<Vec<_>>(),
                    &w2,
                    &b2,
                )
            })
            .collect();
        assert_close(&rows(&got, 6), &want, 1e-5, "feed forward");
    }
}
