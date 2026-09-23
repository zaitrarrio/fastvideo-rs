//! `LTX2VideoDiffusionDecoderModel` — LTX-2.5 DiffVAE.
//!
//! Stages 1–4 deterministically upsample de-normalized latents with
//! neighborhood attention; stage 5 does a single x0 pixel prediction.
//! Neighborhood attention uses gather + SDPA with NATTEN’s inward-shifted
//! window (no Hub kernels). See docs/ports/ltx25.md §DiffVAE.

use fastvideo_models::ltx2::config::Ltx2DiffusionDecoderConfig;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::wan::nn::{self, sinusoidal_timesteps, Linear};
use crate::wan::ops::host as host_ops;
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::{msg, pinned};

const SWIGLU_TILE: usize = 16_384;
const NA_QUERY_TILE: usize = 512;
const ADALN_CHUNKS: usize = 7;

fn lin(map: &WeightMap, prefix: &str, cin: usize, cout: usize, bias: bool) -> Result<Linear> {
    Linear::load(map, prefix, cin, cout, bias)
}

fn rms_w(map: &WeightMap, prefix: &str, dim: usize) -> Result<CudaTensor> {
    let mut w = cuda_tensor_shaped(map, &format!("{prefix}.weight"), &[dim])?;
    w.pin_device()?;
    Ok(w)
}

/// Permute at any rank. Device `gather_nd` only covers rank ≤ 6; higher ranks
/// (pixel-shuffle / patchify) rearrange on host without tripping `host_fallback`.
fn permute_any(x: &CudaTensor, dims: &[usize]) -> Result<CudaTensor> {
    let rank = x.rank();
    if dims.len() != rank || dims.iter().any(|&d| d >= rank) {
        return Err(msg(format!("invalid permute {dims:?} for {:?}", x.shape)));
    }
    let mut seen = vec![false; rank];
    if dims.iter().any(|&d| std::mem::replace(&mut seen[d], true)) {
        return Err(msg(format!("permute dims not a permutation: {dims:?}")));
    }
    if dims.iter().enumerate().all(|(i, &d)| i == d) {
        return Ok(x.clone());
    }
    let out_shape: Vec<usize> = dims.iter().map(|&d| x.shape[d]).collect();
    let non_one: Vec<usize> = dims.iter().copied().filter(|&d| x.shape[d] != 1).collect();
    if non_one.windows(2).all(|w| w[0] < w[1]) {
        return x.reshape(out_shape);
    }
    if rank <= 6 {
        return x.permute(dims);
    }
    let out = host_ops::permute(&x.host_cow()?, &x.shape, dims);
    CudaTensor::from_vec(out, out_shape)
}

/// Space-to-depth on H/W: `[B,C,F,H,W] → [B, C·p², F, H/p, W/p]` with channel
/// order `(c, w_off, h_off)` matching Diffusers `_patchify`.
fn patchify(x: &CudaTensor, p: usize) -> Result<CudaTensor> {
    let [b, c, f, h, w] = match x.shape[..] {
        [b, c, f, h, w] => [b, c, f, h, w],
        _ => return Err(msg(format!("patchify expects rank 5, got {:?}", x.shape))),
    };
    if !h.is_multiple_of(p) || !w.is_multiple_of(p) {
        return Err(msg(format!("patchify: {h}x{w} not divisible by {p}")));
    }
    // [B, C, F, H/p, p, W/p, p] → permute (0,1,6,4,2,3,5) → [B, C, p, p, F, H/p, W/p]
    let y = permute_any(
        &x.reshape(vec![b, c, f, h / p, p, w / p, p])?,
        &[0, 1, 6, 4, 2, 3, 5],
    )?;
    y.reshape(vec![b, c * p * p, f, h / p, w / p])
}

fn unpatchify(x: &CudaTensor, p: usize) -> Result<CudaTensor> {
    let [b, c4, f, h, w] = match x.shape[..] {
        [b, c4, f, h, w] => [b, c4, f, h, w],
        _ => return Err(msg(format!("unpatchify expects rank 5, got {:?}", x.shape))),
    };
    if !c4.is_multiple_of(p * p) {
        return Err(msg(format!("unpatchify: channels {c4} not {p}²·C")));
    }
    let c = c4 / (p * p);
    // Inverse of patchify permute.
    let y = permute_any(
        &x.reshape(vec![b, c, p, p, f, h, w])?,
        &[0, 1, 4, 5, 3, 6, 2],
    )?;
    y.reshape(vec![b, c, f, h * p, w * p])
}

/// Inward-shifted window start (NATTEN / Flex neighborhood mask).
fn window_start(q: usize, size: usize, kernel: usize) -> usize {
    let k = kernel.min(size);
    q.saturating_sub(k / 2).min(size.saturating_sub(k))
}

/// 3D RoPE on `[B,T,H,W,heads,D]` (last dim split T / H / W).
fn rope_dim_split(head_dim: usize) -> (usize, usize, usize) {
    let mut dim_t = (head_dim / 4) / 2 * 2;
    let mut dim_hw = (head_dim - dim_t) / 2;
    if dim_hw % 2 != 0 {
        dim_t -= 2;
        dim_hw = (head_dim - dim_t) / 2;
    }
    (dim_t, dim_hw, dim_hw)
}

fn rotate_pairs(
    x: &[f32],
    positions: &[f32],
    inv_freqs: &[f32],
    axis_len: usize,
    stride_axis: usize,
    outer: usize,
    inner: usize,
) -> Vec<f32> {
    // x layout: for each of `outer` blocks of `axis_len * inner` floats, rotate pairs along axis.
    let mut out = x.to_vec();
    let pairs = inv_freqs.len();
    for o in 0..outer {
        for a in 0..axis_len {
            let base = o * axis_len * inner + a * inner;
            let pos = positions[a];
            for p in 0..pairs {
                let i0 = base + p * 2;
                let i1 = i0 + 1;
                if i1 >= base + inner {
                    break;
                }
                let ang = pos * inv_freqs[p];
                let (c, s) = (ang.cos(), ang.sin());
                let (e, odd) = (x[i0], x[i1]);
                out[i0] = e * c - odd * s;
                out[i1] = e * s + odd * c;
            }
            let _ = stride_axis;
        }
    }
    out
}

fn inv_freqs(dim: usize, base: f32) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| (1.0 / base.powf(2.0 * i as f32 / dim as f32)))
        .collect()
}

fn apply_rope3d(
    x: &[f32],
    b: usize,
    t: usize,
    h: usize,
    w: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let (dt, dh, dw) = rope_dim_split(head_dim);
    let inv_t = inv_freqs(dt, 10_000.0);
    let inv_h = inv_freqs(dh, 10_000.0);
    let inv_w = inv_freqs(dw, 10_000.0);
    let pos_t: Vec<f32> = (0..t).map(|i| i as f32).collect();
    let pos_h: Vec<f32> = (0..h).map(|i| i as f32).collect();
    let pos_w: Vec<f32> = (0..w).map(|i| i as f32).collect();
    let n = b * t * h * w * heads;
    let mut out = x.to_vec();
    for i in 0..n {
        let base = i * head_dim;
        let rem = i % (t * h * w * heads);
        let tw = rem / (h * w * heads);
        let rem2 = rem % (h * w * heads);
        let hh = rem2 / (w * heads);
        let rem3 = rem2 % (w * heads);
        let ww = rem3 / heads;
        // Rotate T chunk
        for p in 0..dt / 2 {
            let ang = pos_t[tw] * inv_t[p];
            let (c, s) = (ang.cos(), ang.sin());
            let i0 = base + p * 2;
            let (e, o) = (x[i0], x[i0 + 1]);
            out[i0] = e * c - o * s;
            out[i0 + 1] = e * s + o * c;
        }
        for p in 0..dh / 2 {
            let ang = pos_h[hh] * inv_h[p];
            let (c, s) = (ang.cos(), ang.sin());
            let i0 = base + dt + p * 2;
            let (e, o) = (x[i0], x[i0 + 1]);
            out[i0] = e * c - o * s;
            out[i0 + 1] = e * s + o * c;
        }
        for p in 0..dw / 2 {
            let ang = pos_w[ww] * inv_w[p];
            let (c, s) = (ang.cos(), ang.sin());
            let i0 = base + dt + dh + p * 2;
            let (e, o) = (x[i0], x[i0 + 1]);
            out[i0] = e * c - o * s;
            out[i0 + 1] = e * s + o * c;
        }
    }
    let _ = rotate_pairs; // silence if unused in some builds
    out
}

/// Softmax attention for one query against `kw` keys (already scaled Q).
fn attn_one(q: &[f32], k: &[f32], v: &[f32], d: usize, kw: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; kw];
    let mut m = f32::NEG_INFINITY;
    for i in 0..kw {
        let mut s = 0.0f32;
        for j in 0..d {
            s += q[j] * k[i * d + j];
        }
        scores[i] = s;
        m = m.max(s);
    }
    let mut sum = 0.0f32;
    for s in &mut scores {
        *s = (*s - m).exp();
        sum += *s;
    }
    let inv = 1.0 / sum.max(1e-20);
    let mut out = vec![0.0f32; d];
    for i in 0..kw {
        let a = scores[i] * inv;
        for j in 0..d {
            out[j] += a * v[i * d + j];
        }
    }
    out
}

/// Neighborhood attention on `[B,T,H,W,heads,D]` → same shape (host gather + SDPA).
fn neighborhood_attn_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    t: usize,
    h: usize,
    w: usize,
    heads: usize,
    d: usize,
    kernel: [usize; 3],
) -> Vec<f32> {
    let (kt, kh, kw) = (kernel[0].min(t), kernel[1].min(h), kernel[2].min(w));
    let kwin = kt * kh * kw;
    let spatial = t * h * w;
    let mut out = vec![0.0f32; q.len()];
    let jobs: Vec<_> = (0..b * spatial * heads).collect();
    let results: Vec<_> = jobs
        .par_iter()
        .map(|&idx| {
            let bi = idx / (spatial * heads);
            let rem = idx % (spatial * heads);
            let qi = rem / heads;
            let head = rem % heads;
            let qt = qi / (h * w);
            let qh = (qi / w) % h;
            let qw = qi % w;
            let st = window_start(qt, t, kt);
            let sh = window_start(qh, h, kh);
            let sw = window_start(qw, w, kw);
            let q_off = ((((bi * t + qt) * h + qh) * w + qw) * heads + head) * d;
            let mut kk = vec![0.0f32; kwin * d];
            let mut vv = vec![0.0f32; kwin * d];
            let mut wi = 0;
            for tt in st..st + kt {
                for hh in sh..sh + kh {
                    for ww in sw..sw + kw {
                        let k_off = ((((bi * t + tt) * h + hh) * w + ww) * heads + head) * d;
                        kk[wi * d..(wi + 1) * d].copy_from_slice(&k[k_off..k_off + d]);
                        vv[wi * d..(wi + 1) * d].copy_from_slice(&v[k_off..k_off + d]);
                        wi += 1;
                    }
                }
            }
            (q_off, attn_one(&q[q_off..q_off + d], &kk, &vv, d, kwin))
        })
        .collect();
    for (off, row) in results {
        out[off..off + d].copy_from_slice(&row);
    }
    let _ = NA_QUERY_TILE;
    out
}

struct NeighborhoodAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: CudaTensor,
    norm_k: CudaTensor,
    heads: usize,
    head_dim: usize,
    kernel: [usize; 3],
    scale: f32,
}

impl NeighborhoodAttention {
    fn load(
        map: &WeightMap,
        prefix: &str,
        dim: usize,
        head_dim: usize,
        kernel: [usize; 3],
    ) -> Result<Self> {
        if !dim.is_multiple_of(head_dim) {
            return Err(msg(format!(
                "NA: dim {dim} not divisible by head_dim {head_dim}"
            )));
        }
        Ok(Self {
            to_q: lin(map, &format!("{prefix}.to_q"), dim, dim, true)?,
            to_k: lin(map, &format!("{prefix}.to_k"), dim, dim, true)?,
            to_v: lin(map, &format!("{prefix}.to_v"), dim, dim, true)?,
            to_out: lin(map, &format!("{prefix}.to_out.0"), dim, dim, true)?,
            norm_q: rms_w(map, &format!("{prefix}.norm_q"), head_dim)?,
            norm_k: rms_w(map, &format!("{prefix}.norm_k"), head_dim)?,
            heads: dim / head_dim,
            head_dim,
            kernel,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// Channels-last `[B,T,H,W,C]`.
    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, t, h, w, c] = match x.shape[..] {
            [b, t, h, w, c] => [b, t, h, w, c],
            _ => return Err(msg(format!("NA expects [B,T,H,W,C], got {:?}", x.shape))),
        };
        let (kt, kh, kw) = (self.kernel[0], self.kernel[1], self.kernel[2]);
        if t < kt || h < kh || w < kw {
            return Err(msg(format!(
                "NA grid ({t},{h},{w}) smaller than kernel {:?}",
                self.kernel
            )));
        }
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;
        // Reshape to [B,T,H,W,heads,D], RMSNorm last dim, scale Q, RoPE.
        let shape = vec![b, t, h, w, self.heads, self.head_dim];
        let mut qh = q.reshape(shape.clone())?.host_cow()?.into_owned();
        let mut kh = k.reshape(shape.clone())?.host_cow()?.into_owned();
        let vh = v.reshape(shape.clone())?.host_cow()?.into_owned();
        let nq = b * t * h * w * self.heads;
        let wq = self.norm_q.host_cow()?;
        let wk = self.norm_k.host_cow()?;
        for i in 0..nq {
            let base = i * self.head_dim;
            let mut ms = 0.0f32;
            for j in 0..self.head_dim {
                ms += qh[base + j] * qh[base + j];
            }
            let inv = (ms / self.head_dim as f32 + 1e-6).sqrt().recip();
            for j in 0..self.head_dim {
                qh[base + j] = qh[base + j] * inv * wq[j] * self.scale;
            }
            let mut ms = 0.0f32;
            for j in 0..self.head_dim {
                ms += kh[base + j] * kh[base + j];
            }
            let inv = (ms / self.head_dim as f32 + 1e-6).sqrt().recip();
            for j in 0..self.head_dim {
                kh[base + j] = kh[base + j] * inv * wk[j];
            }
        }
        qh = apply_rope3d(&qh, b, t, h, w, self.heads, self.head_dim);
        kh = apply_rope3d(&kh, b, t, h, w, self.heads, self.head_dim);
        let out = neighborhood_attn_host(
            &qh,
            &kh,
            &vh,
            b,
            t,
            h,
            w,
            self.heads,
            self.head_dim,
            self.kernel,
        );
        let flat = CudaTensor::from_vec(out, vec![b, t, h, w, c])?;
        self.to_out.forward(&flat)
    }
}

struct SwiGLU {
    w_up: Linear,
    w_gate: Linear,
    w_down: Linear,
}

impl SwiGLU {
    fn load(map: &WeightMap, prefix: &str, dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            w_up: lin(map, &format!("{prefix}.w_up"), dim, hidden, false)?,
            w_gate: lin(map, &format!("{prefix}.w_gate"), dim, hidden, false)?,
            w_down: lin(map, &format!("{prefix}.w_down"), hidden, dim, false)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, t, h, w, c] = match x.shape[..] {
            [b, t, h, w, c] => [b, t, h, w, c],
            _ => {
                return Err(msg(format!(
                    "SwiGLU expects [B,T,H,W,C], got {:?}",
                    x.shape
                )))
            }
        };
        let tokens = t * h * w;
        if tokens <= SWIGLU_TILE {
            let up = self.w_up.forward(x)?;
            let gate = self.w_gate.forward(x)?.silu();
            return self.w_down.forward(&up.mul(&gate)?);
        }
        let flat = x.reshape(vec![b, tokens, c])?;
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < tokens {
            let end = (start + SWIGLU_TILE).min(tokens);
            let tile = flat.narrow(1, start, end - start)?;
            let up = self.w_up.forward(&tile)?;
            let gate = self.w_gate.forward(&tile)?.silu();
            pieces.push(self.w_down.forward(&up.mul(&gate)?)?);
            start = end;
        }
        let refs: Vec<&CudaTensor> = pieces.iter().collect();
        CudaTensor::cat(&refs, 1)?.reshape(vec![b, t, h, w, c])
    }
}

struct NaBlock {
    norm1: CudaTensor,
    attn: NeighborhoodAttention,
    norm2: CudaTensor,
    mlp: SwiGLU,
}

impl NaBlock {
    fn load(
        map: &WeightMap,
        prefix: &str,
        dim: usize,
        head_dim: usize,
        kernel: [usize; 3],
        hidden: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm1: rms_w(map, &format!("{prefix}.norm1"), dim)?,
            attn: NeighborhoodAttention::load(
                map,
                &format!("{prefix}.attn"),
                dim,
                head_dim,
                kernel,
            )?,
            norm2: rms_w(map, &format!("{prefix}.norm2"), dim)?,
            mlp: SwiGLU::load(map, &format!("{prefix}.mlp"), dim, hidden)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let h = self.attn.forward(&nn::rms_norm(x, &self.norm1, 1e-6)?)?;
        let x = x.add(&h)?;
        let h = self.mlp.forward(&nn::rms_norm(&x, &self.norm2, 1e-6)?)?;
        x.add(&h)
    }
}

struct AdaLnZero {
    proj: Linear,
    dim: usize,
}

impl AdaLnZero {
    fn load(map: &WeightMap, prefix: &str, dim: usize, t_emb: usize) -> Result<Self> {
        Ok(Self {
            proj: lin(
                map,
                &format!("{prefix}.proj"),
                t_emb,
                ADALN_CHUNKS * dim,
                true,
            )?,
            dim,
        })
    }

    /// Seven `[B,1,1,1,C]` chunks.
    fn forward(&self, t_emb: &CudaTensor) -> Result<Vec<CudaTensor>> {
        let y = self.proj.forward(&t_emb.silu())?;
        let [b, _] = match y.shape[..] {
            [b, c] if c == ADALN_CHUNKS * self.dim => [b, c],
            _ => return Err(msg(format!("AdaLN got {:?}", y.shape))),
        };
        let host = y.host_cow()?;
        let mut out = Vec::with_capacity(ADALN_CHUNKS);
        for i in 0..ADALN_CHUNKS {
            let mut chunk = Vec::with_capacity(b * self.dim);
            for bi in 0..b {
                let off = bi * ADALN_CHUNKS * self.dim + i * self.dim;
                chunk.extend_from_slice(&host[off..off + self.dim]);
            }
            out.push(CudaTensor::from_vec(chunk, vec![b, 1, 1, 1, self.dim])?);
        }
        Ok(out)
    }
}

struct DiffusionNaBlock {
    context_proj: Linear,
    scale_shift_table: CudaTensor,
    norm1: CudaTensor,
    attn: NeighborhoodAttention,
    norm2: CudaTensor,
    mlp: SwiGLU,
    dim: usize,
}

impl DiffusionNaBlock {
    fn load(
        map: &WeightMap,
        prefix: &str,
        dim: usize,
        ctx: usize,
        head_dim: usize,
        kernel: [usize; 3],
        hidden: usize,
    ) -> Result<Self> {
        Ok(Self {
            context_proj: lin(map, &format!("{prefix}.context_proj"), ctx, dim, true)?,
            scale_shift_table: pinned(
                cuda_tensor_shaped(
                    map,
                    &format!("{prefix}.scale_shift_table"),
                    &[ADALN_CHUNKS, dim],
                )?
                .host_cow()?
                .into_owned(),
                vec![ADALN_CHUNKS, dim],
            )?,
            norm1: rms_w(map, &format!("{prefix}.norm1"), dim)?,
            attn: NeighborhoodAttention::load(
                map,
                &format!("{prefix}.attn"),
                dim,
                head_dim,
                kernel,
            )?,
            norm2: rms_w(map, &format!("{prefix}.norm2"), dim)?,
            mlp: SwiGLU::load(map, &format!("{prefix}.mlp"), dim, hidden)?,
            dim,
        })
    }

    fn forward(
        &self,
        x: &CudaTensor,
        context: &CudaTensor,
        mod_chunks: &[CudaTensor],
    ) -> Result<CudaTensor> {
        let table = self.scale_shift_table.host_cow()?;
        let mut scale_msa = mod_chunks[0].host_cow()?.into_owned();
        let mut shift_msa = mod_chunks[1].host_cow()?.into_owned();
        let mut scale_mlp = mod_chunks[3].host_cow()?.into_owned();
        let mut shift_mlp = mod_chunks[4].host_cow()?.into_owned();
        let b = x.shape[0];
        for bi in 0..b {
            for c in 0..self.dim {
                scale_msa[bi * self.dim + c] += table[c];
                shift_msa[bi * self.dim + c] += table[self.dim + c];
                scale_mlp[bi * self.dim + c] += table[3 * self.dim + c];
                shift_mlp[bi * self.dim + c] += table[4 * self.dim + c];
            }
        }
        let scale_msa = CudaTensor::from_vec(scale_msa, vec![b, 1, 1, 1, self.dim])?;
        let shift_msa = CudaTensor::from_vec(shift_msa, vec![b, 1, 1, 1, self.dim])?;
        let scale_mlp = CudaTensor::from_vec(scale_mlp, vec![b, 1, 1, 1, self.dim])?;
        let shift_mlp = CudaTensor::from_vec(shift_mlp, vec![b, 1, 1, 1, self.dim])?;

        let x = x.add(&self.context_proj.forward(context)?)?;
        let n1 = nn::rms_norm(&x, &self.norm1, 1e-6)?;
        let ones = CudaTensor::ones(&[b, 1, 1, 1, self.dim]);
        let h = self
            .attn
            .forward(&n1.mul(&ones.add(&scale_msa)?)?.add(&shift_msa)?)?;
        let x = x.add(&h)?;
        let n2 = nn::rms_norm(&x, &self.norm2, 1e-6)?;
        let h = self
            .mlp
            .forward(&n2.mul(&ones.add(&scale_mlp)?)?.add(&shift_mlp)?)?;
        x.add(&h)
    }
}

struct PixelShuffleUpsampler {
    proj: Linear,
    stride: [usize; 3],
    out_channels: usize,
}

impl PixelShuffleUpsampler {
    fn load(
        map: &WeightMap,
        prefix: &str,
        in_ch: usize,
        stride: [usize; 3],
        reduction: usize,
    ) -> Result<Self> {
        let proj_out = stride.iter().product::<usize>() * in_ch / reduction;
        let out_channels = proj_out / stride.iter().product::<usize>();
        Ok(Self {
            proj: lin(map, &format!("{prefix}.proj"), in_ch, proj_out, true)?,
            stride,
            out_channels,
        })
    }

    fn forward(&self, x: &CudaTensor, drop_leading: bool) -> Result<CudaTensor> {
        let [b, f, h, w, _] = match x.shape[..] {
            [b, f, h, w, c] => [b, f, h, w, c],
            _ => {
                return Err(msg(format!(
                    "upsample expects [B,T,H,W,C], got {:?}",
                    x.shape
                )))
            }
        };
        let [st, sh, sw] = self.stride;
        let y = self.proj.forward(x)?;
        // [B,F,H,W, Cout, st, sh, sw] → [B, F*st, H*sh, W*sw, Cout]
        let y = permute_any(
            &y.reshape(vec![b, f, h, w, self.out_channels, st, sh, sw])?,
            &[0, 1, 5, 2, 6, 3, 7, 4],
        )?
        .reshape(vec![b, f * st, h * sh, w * sw, self.out_channels])?;
        if st == 2 && drop_leading {
            y.narrow(1, 1, y.shape[1] - 1)
        } else {
            Ok(y)
        }
    }
}

struct TimestepEmbed {
    linear_1: Linear,
    linear_2: Linear,
    sinusoid: usize,
}

impl TimestepEmbed {
    fn load(map: &WeightMap, prefix: &str, emb_dim: usize) -> Result<Self> {
        // PixArtAlphaCombinedTimestepSizeEmbeddings → timestep_embedder (TimestepEmbedding).
        Ok(Self {
            linear_1: lin(
                map,
                &format!("{prefix}.timestep_embedder.linear_1"),
                256,
                emb_dim,
                true,
            )?,
            linear_2: lin(
                map,
                &format!("{prefix}.timestep_embedder.linear_2"),
                emb_dim,
                emb_dim,
                true,
            )?,
            sinusoid: 256,
        })
    }

    fn forward(&self, timestep: f32) -> Result<CudaTensor> {
        let t = CudaTensor::from_vec(vec![timestep], vec![1])?;
        let s = sinusoidal_timesteps(&t, self.sinusoid)?;
        let h = self.linear_1.forward(&s)?.silu();
        self.linear_2.forward(&h)
    }
}

/// DiffVAE: de-normalized latents → RGB.
pub struct DiffusionDecoder {
    cfg: Ltx2DiffusionDecoderConfig,
    conv_in: Linear,
    det_stages: Vec<Vec<NaBlock>>,
    upsamples: Vec<PixelShuffleUpsampler>,
    t_embedder: TimestepEmbed,
    conv_in_x_t: Linear,
    shared_adaln: AdaLnZero,
    diff_blocks: Vec<DiffusionNaBlock>,
    norm_out: CudaTensor,
    conv_out: Linear,
    #[allow(dead_code)]
    latents_mean: CudaTensor,
    #[allow(dead_code)]
    latents_std: CudaTensor,
}

impl DiffusionDecoder {
    /// `map` is the `diffusion_decoder/` folder (keys under `decoder.`).
    pub fn load(map: &WeightMap, cfg: &Ltx2DiffusionDecoderConfig) -> Result<Self> {
        if cfg.model_output_type != "x0" {
            return Err(msg("DiffVAE: only model_output_type=x0 is supported"));
        }
        let p = "decoder";
        let n_det = cfg.upsample_strides.len();
        let mut det_stages = Vec::with_capacity(n_det);
        let mut upsamples = Vec::with_capacity(n_det);
        for stage in 0..n_det {
            let dim = cfg.stage_channels[stage];
            let hidden = cfg.swiglu_hidden_dim(dim);
            let kernel = cfg.stage_kernels[stage];
            let depth = cfg.stage_depths[stage];
            let blocks = (0..depth)
                .map(|i| {
                    NaBlock::load(
                        map,
                        &format!("{p}.det_stages.{stage}.{i}"),
                        dim,
                        cfg.head_dim,
                        kernel,
                        hidden,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            det_stages.push(blocks);
            upsamples.push(PixelShuffleUpsampler::load(
                map,
                &format!("{p}.upsamples.{stage}"),
                dim,
                cfg.upsample_strides[stage],
                cfg.upsample_channel_reductions[stage],
            )?);
        }
        let s5 = cfg.context_channels();
        let hidden5 = cfg.swiglu_hidden_dim(s5);
        let pix_ch = cfg.out_channels * cfg.patch_size * cfg.patch_size;
        let diff_blocks = (0..cfg.stage_depths[n_det])
            .map(|i| {
                DiffusionNaBlock::load(
                    map,
                    &format!("{p}.diff_blocks.{i}"),
                    s5,
                    s5,
                    cfg.head_dim,
                    cfg.stage5_kernel,
                    hidden5,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Outer model buffers (may be zeros/ones if not written; VAE stats used by pipeline denorm).
        let mean = cuda_tensor_shaped(map, "latents_mean", &[cfg.latent_channels])
            .unwrap_or_else(|_| CudaTensor::zeros(&[cfg.latent_channels]));
        let std = cuda_tensor_shaped(map, "latents_std", &[cfg.latent_channels])
            .unwrap_or_else(|_| CudaTensor::ones(&[cfg.latent_channels]));
        Ok(Self {
            conv_in: lin(
                map,
                &format!("{p}.conv_in"),
                cfg.latent_channels,
                cfg.stage_channels[0],
                true,
            )?,
            t_embedder: TimestepEmbed::load(map, &format!("{p}.t_embedder"), cfg.t_emb_dim)?,
            conv_in_x_t: lin(map, &format!("{p}.conv_in_x_t"), pix_ch, s5, true)?,
            shared_adaln: AdaLnZero::load(map, &format!("{p}.shared_adaln"), s5, cfg.t_emb_dim)?,
            norm_out: rms_w(map, &format!("{p}.norm_out"), s5)?,
            conv_out: lin(map, &format!("{p}.conv_out"), s5, pix_ch, true)?,
            latents_mean: pinned(mean.host_cow()?.into_owned(), vec![cfg.latent_channels])?,
            latents_std: pinned(std.host_cow()?.into_owned(), vec![cfg.latent_channels])?,
            det_stages,
            upsamples,
            diff_blocks,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Ltx2DiffusionDecoderConfig {
        &self.cfg
    }

    fn stages_1_to_3(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let pad = self.cfg.trailing_pad_latent_frames();
        let mut x = if pad > 0 {
            let last = latents.narrow(2, latents.shape[2] - 1, 1)?;
            let mut parts = vec![latents.clone()];
            for _ in 0..pad {
                parts.push(last.clone());
            }
            let refs: Vec<&CudaTensor> = parts.iter().collect();
            CudaTensor::cat(&refs, 2)?
        } else {
            latents.clone()
        };
        // [B,C,F,H,W] → [B,F,H,W,C]
        x = x.permute(&[0, 2, 3, 4, 1])?;
        x = self.conv_in.forward(&x)?;
        let n = self.det_stages.len();
        for stage in 0..n - 1 {
            for block in &self.det_stages[stage] {
                x = block.forward(&x)?;
            }
            x = self.upsamples[stage].forward(&x, true)?;
        }
        Ok(x)
    }

    fn stage_4(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let stage = self.det_stages.len() - 1;
        let mut x = x.clone();
        for block in &self.det_stages[stage] {
            x = block.forward(&x)?;
        }
        x = self.upsamples[stage].forward(&x, true)?;
        let pad = self.cfg.trailing_pad_latent_frames();
        if pad > 0 {
            let crop = pad * self.cfg.temporal_compression_ratio;
            x = x.narrow(1, 0, x.shape[1] - crop)?;
        }
        Ok(x)
    }

    fn diffusion_step(
        &self,
        context: &CudaTensor,
        x_t: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let t_emb = self
            .t_embedder
            .forward(timestep * self.cfg.timestep_scale_multiplier)?;
        // Expand batch if needed.
        let b = context.shape[0];
        let t_emb = if b > 1 {
            let row = t_emb.host_cow()?;
            let mut all = Vec::with_capacity(b * row.len());
            for _ in 0..b {
                all.extend_from_slice(&row);
            }
            CudaTensor::from_vec(all, vec![b, row.len()])?
        } else {
            t_emb
        };
        let mods = self.shared_adaln.forward(&t_emb)?;
        let p = self.cfg.patch_size;
        let mut h = patchify(x_t, p)?.permute(&[0, 2, 3, 4, 1])?;
        h = self.conv_in_x_t.forward(&h)?;
        for block in &self.diff_blocks {
            h = block.forward(&h, context, &mods)?;
        }
        h = nn::rms_norm(&h, &self.norm_out, 1e-6)?;
        h = self.conv_out.forward(&h)?;
        let h = h.permute(&[0, 4, 1, 2, 3])?;
        unpatchify(&h, p)
    }

    /// De-normalized latents `[B,C,F,H,W]` → RGB `[B,3,frames,H_px,W_px]`.
    pub fn decode(&self, latents: &CudaTensor, seed: u64) -> Result<CudaTensor> {
        let context = self.stage_4(&self.stages_1_to_3(latents)?)?;
        let [b, frames, hp, wp, _] = match context.shape[..] {
            [b, f, h, w, c] => [b, f, h, w, c],
            _ => return Err(msg(format!("DiffVAE context {:?}", context.shape))),
        };
        let p = self.cfg.patch_size;
        let (hf, wf) = (hp * p, wp * p);
        let n = b * self.cfg.out_channels * frames * hf * wf;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..n)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let x_t = CudaTensor::from_vec(noise, vec![b, self.cfg.out_channels, frames, hf, wf])?;
        // Shipped: 1-step x0 at t=1 → prediction is the image.
        self.diffusion_step(&context, &x_t, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::weights;
    use super::*;

    fn tiny_cfg() -> Ltx2DiffusionDecoderConfig {
        Ltx2DiffusionDecoderConfig {
            out_channels: 3,
            latent_channels: 4,
            patch_size: 2,
            scaling_factor: 1.0,
            head_dim: 8,
            stage_channels: vec![32, 16, 16, 16, 16],
            stage_depths: vec![1, 1, 1, 1, 1],
            stage_kernels: vec![[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            upsample_strides: vec![[1, 2, 2], [2, 1, 1], [2, 2, 2], [2, 2, 2]],
            upsample_channel_reductions: vec![2, 1, 1, 1],
            stage5_kernel: [3, 3, 3],
            t_emb_dim: 32,
            timestep_scale_multiplier: 1000.0,
            model_output_type: "x0",
            num_inference_steps: 1,
            spatial_compression_ratio: 32,
            temporal_compression_ratio: 8,
            mlp_ratio: 2.0,
        }
    }

    #[test]
    fn window_start_is_inward_shifted() {
        assert_eq!(window_start(0, 10, 3), 0);
        assert_eq!(window_start(1, 10, 3), 0);
        assert_eq!(window_start(5, 10, 3), 4);
        assert_eq!(window_start(9, 10, 3), 7);
    }

    #[test]
    fn patchify_unpatchify_roundtrip() {
        let x = CudaTensor::from_vec(
            (0..2 * 4 * 3 * 4 * 6).map(|i| i as f32).collect(),
            vec![2, 4, 3, 4, 6],
        )
        .unwrap();
        let y = unpatchify(&patchify(&x, 2).unwrap(), 2).unwrap();
        assert_eq!(y.shape, x.shape);
        assert_eq!(&*y.host_cow().unwrap(), &*x.host_cow().unwrap());
    }

    #[test]
    fn tiny_decode_is_finite_rgb() {
        let cfg = tiny_cfg();
        // Generated weights invent keys; load may fail on missing nested paths — use full load via generated map.
        let map = weights();
        let dec = match DiffusionDecoder::load(&map, &cfg) {
            Ok(d) => d,
            Err(e) => {
                // Generated map invents any key; if load still fails, surface it.
                panic!("load: {e}");
            }
        };
        // Latent grid must survive kernels after pad + upsamples. Start large enough.
        // pad=2 on T=5 → 7; after strides temporal *1*2*2*2 with drops ≈ enough for stage5 kernel 3.
        let x = CudaTensor::zeros(&[1, 4, 5, 4, 4]);
        let y = dec.decode(&x, 7).expect("decode");
        assert_eq!(y.shape[0], 1);
        assert_eq!(y.shape[1], 3);
        assert!(y.host_cow().unwrap().iter().all(|v| v.is_finite()));
    }
}
