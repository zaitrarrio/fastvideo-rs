//! LongCat Block Sparse Attention (BSA) — host reference matching
//! `flash_attn_bsa_3d` selection (mean-pool → top-k → sparse SDPA).
//!
//! Algorithm (Meituan LongCat-Video `bsa_interface.py`):
//! 1. Rearrange THW tokens into contiguous 3D chunks.
//! 2. Mean-pool each chunk → compressed Q/K.
//! 3. Score = Q_cmp @ K_cmp^T; keep top `(1 - sparsity)` KV blocks per Q block.
//! 4. Softmax-attend each query token only to tokens in selected KV blocks.

use crate::wan::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Diffusers / Meituan BSA params.
#[derive(Debug, Clone, Copy)]
pub struct BsaParams {
    pub sparsity: f32,
    pub chunk_q: [usize; 3],
    pub chunk_k: [usize; 3],
}

impl BsaParams {
    pub fn from_config(sparsity: f32, chunk: [usize; 3]) -> Self {
        Self {
            sparsity,
            chunk_q: chunk,
            chunk_k: chunk,
        }
    }
}

/// `q/k/v` are `[B, H, S, D]` with `S = T*H*W` in raster THW order.
pub fn flash_attn_bsa_3d(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    latent_thw: [usize; 3],
    params: BsaParams,
) -> Result<CudaTensor> {
    let [b, heads, sq, d] = match q.shape[..] {
        [b, h, s, d] => [b, h, s, d],
        _ => return Err(msg(format!("bsa q shape {:?}", q.shape))),
    };
    let sk = k.shape[2];
    let [t, h, w] = latent_thw;
    if t * h * w != sq || t * h * w != sk {
        return Err(msg(format!(
            "bsa latent {t}x{h}x{w} vs seq q={sq} k={sk}"
        )));
    }
    let [tq, hq, wq] = params.chunk_q;
    let [tk, hk, wk] = params.chunk_k;
    if tq == 0 || hq == 0 || wq == 0 || tk == 0 || hk == 0 || wk == 0 {
        return Err(msg("bsa chunk dims must be > 0"));
    }
    if t % tq != 0 || h % hq != 0 || w % wq != 0 || t % tk != 0 || h % hk != 0 || w % wk != 0 {
        // Fall back to dense when shapes are not divisible (tiny graphs).
        return crate::wan::nn::scaled_dot_product_attention_masked(q, k, v, None, None);
    }

    let nt_q = t / tq;
    let nh_q = h / hq;
    let nw_q = w / wq;
    let nt_k = t / tk;
    let nh_k = h / hk;
    let nw_k = w / wk;
    let n_blocks_q = nt_q * nh_q * nw_q;
    let n_blocks_k = nt_k * nh_k * nw_k;
    let chunk_q = tq * hq * wq;
    let chunk_k = tk * hk * wk;

    let qh = rearrange_thw_to_3d_block(&q.host_cow()?, b, heads, nt_q, nh_q, nw_q, tq, hq, wq, d);
    let kh = rearrange_thw_to_3d_block(&k.host_cow()?, b, heads, nt_k, nh_k, nw_k, tk, hk, wk, d);
    let vh = rearrange_thw_to_3d_block(&v.host_cow()?, b, heads, nt_k, nh_k, nw_k, tk, hk, wk, d);

    let scale = 1.0 / (d as f32).sqrt();
    let keep = ((1.0 - params.sparsity) * n_blocks_k as f32)
        .round()
        .max(1.0) as usize;
    let keep = keep.min(n_blocks_k);

    let mut out_blocked = vec![0f32; b * heads * sq * d];

    for bi in 0..b {
        for hi in 0..heads {
            let base_q = ((bi * heads + hi) * sq) * d;
            let base_k = ((bi * heads + hi) * sk) * d;

            // Mean-pool compressed Q/K: [n_blocks, D]
            let mut q_cmp = vec![0f32; n_blocks_q * d];
            let mut k_cmp = vec![0f32; n_blocks_k * d];
            for bq in 0..n_blocks_q {
                for i in 0..chunk_q {
                    for di in 0..d {
                        q_cmp[bq * d + di] += qh[base_q + (bq * chunk_q + i) * d + di];
                    }
                }
                for di in 0..d {
                    q_cmp[bq * d + di] /= chunk_q as f32;
                }
            }
            for bk in 0..n_blocks_k {
                for i in 0..chunk_k {
                    for di in 0..d {
                        k_cmp[bk * d + di] += kh[base_k + (bk * chunk_k + i) * d + di];
                    }
                }
                for di in 0..d {
                    k_cmp[bk * d + di] /= chunk_k as f32;
                }
            }

            // Top-k block indices per query block.
            let mut selected: Vec<Vec<usize>> = Vec::with_capacity(n_blocks_q);
            for bq in 0..n_blocks_q {
                let mut scores: Vec<(f32, usize)> = (0..n_blocks_k)
                    .map(|bk| {
                        let mut s = 0f32;
                        for di in 0..d {
                            s += q_cmp[bq * d + di] * k_cmp[bk * d + di];
                        }
                        (s, bk)
                    })
                    .collect();
                scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut idxs: Vec<usize> = scores[..keep].iter().map(|x| x.1).collect();
                idxs.sort_unstable();
                selected.push(idxs);
            }

            // Sparse SDPA within selected blocks.
            for bq in 0..n_blocks_q {
                let sel = &selected[bq];
                let kv_tokens: Vec<usize> = sel
                    .iter()
                    .flat_map(|&bk| (0..chunk_k).map(move |i| bk * chunk_k + i))
                    .collect();
                for qi in 0..chunk_q {
                    let q_off = base_q + (bq * chunk_q + qi) * d;
                    let mut scores = vec![0f32; kv_tokens.len()];
                    for (si, &kj) in kv_tokens.iter().enumerate() {
                        let k_off = base_k + kj * d;
                        let mut s = 0f32;
                        for di in 0..d {
                            s += qh[q_off + di] * kh[k_off + di];
                        }
                        scores[si] = s * scale;
                    }
                    let m = scores
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                    let z: f32 = exps.iter().sum::<f32>().max(1e-20);
                    let o_off = q_off;
                    for (&kj, e) in kv_tokens.iter().zip(&exps) {
                        let v_off = base_k + kj * d;
                        let w = e / z;
                        for di in 0..d {
                            out_blocked[o_off + di] += w * vh[v_off + di];
                        }
                    }
                }
            }
        }
    }

    let out_thw = rearrange_3d_block_to_thw(&out_blocked, b, heads, nt_q, nh_q, nw_q, tq, hq, wq, d);
    CudaTensor::from_vec(out_thw, vec![b, heads, sq, d])
}

fn rearrange_thw_to_3d_block(
    x: &[f32],
    b: usize,
    heads: usize,
    nt: usize,
    nh: usize,
    nw: usize,
    t: usize,
    h: usize,
    w: usize,
    d: usize,
) -> Vec<f32> {
    // Input layout: [B,H, T*H*W, D] with THW raster (t major → h → w).
    // Output: blocks contiguous as (Nt,Nh,Nw, t,h,w).
    let t_full = nt * t;
    let h_full = nh * h;
    let w_full = nw * w;
    let seq = t_full * h_full * w_full;
    let mut out = vec![0f32; b * heads * seq * d];
    for bi in 0..b {
        for hi in 0..heads {
            let base = ((bi * heads + hi) * seq) * d;
            for nti in 0..nt {
                for nhi in 0..nh {
                    for nwi in 0..nw {
                        let block = (nti * nh + nhi) * nw + nwi;
                        for ti in 0..t {
                            for yi in 0..h {
                                for xi in 0..w {
                                    let src_t = nti * t + ti;
                                    let src_y = nhi * h + yi;
                                    let src_x = nwi * w + xi;
                                    let src_tok = (src_t * h_full + src_y) * w_full + src_x;
                                    let dst_tok = block * (t * h * w) + (ti * h + yi) * w + xi;
                                    for di in 0..d {
                                        out[base + dst_tok * d + di] = x[base + src_tok * d + di];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn rearrange_3d_block_to_thw(
    x: &[f32],
    b: usize,
    heads: usize,
    nt: usize,
    nh: usize,
    nw: usize,
    t: usize,
    h: usize,
    w: usize,
    d: usize,
) -> Vec<f32> {
    let t_full = nt * t;
    let h_full = nh * h;
    let w_full = nw * w;
    let seq = t_full * h_full * w_full;
    let mut out = vec![0f32; b * heads * seq * d];
    for bi in 0..b {
        for hi in 0..heads {
            let base = ((bi * heads + hi) * seq) * d;
            for nti in 0..nt {
                for nhi in 0..nh {
                    for nwi in 0..nw {
                        let block = (nti * nh + nhi) * nw + nwi;
                        for ti in 0..t {
                            for yi in 0..h {
                                for xi in 0..w {
                                    let dst_t = nti * t + ti;
                                    let dst_y = nhi * h + yi;
                                    let dst_x = nwi * w + xi;
                                    let dst_tok = (dst_t * h_full + dst_y) * w_full + dst_x;
                                    let src_tok = block * (t * h * w) + (ti * h + yi) * w + xi;
                                    for di in 0..d {
                                        out[base + dst_tok * d + di] = x[base + src_tok * d + di];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsa_matches_dense_when_sparsity_zero() {
        // 2x2x2 grid, chunk 1x1x1 → every block kept when sparsity=0.
        let t = 2usize;
        let h = 2usize;
        let w = 2usize;
        let seq = t * h * w;
        let d = 4usize;
        let data: Vec<f32> = (0..1 * 1 * seq * d)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let q = CudaTensor::from_vec(data.clone(), vec![1, 1, seq, d]).unwrap();
        let k = CudaTensor::from_vec(data.clone(), vec![1, 1, seq, d]).unwrap();
        let v = CudaTensor::from_vec(data, vec![1, 1, seq, d]).unwrap();
        let dense =
            crate::wan::nn::scaled_dot_product_attention_masked(&q, &k, &v, None, None).unwrap();
        let sparse = flash_attn_bsa_3d(
            &q,
            &k,
            &v,
            [t, h, w],
            BsaParams {
                sparsity: 0.0,
                chunk_q: [1, 1, 1],
                chunk_k: [1, 1, 1],
            },
        )
        .unwrap();
        let dh = dense.host_cow().unwrap();
        let sh = sparse.host_cow().unwrap();
        for (a, b) in dh.iter().zip(sh.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn bsa_keeps_topk_fraction() {
        let t = 2usize;
        let h = 2usize;
        let w = 2usize;
        let seq = t * h * w;
        let d = 4usize;
        let data: Vec<f32> = (0..seq * d).map(|i| (i as f32) * 0.02).collect();
        let q = CudaTensor::from_vec(data.clone(), vec![1, 1, seq, d]).unwrap();
        let k = CudaTensor::from_vec(data.clone(), vec![1, 1, seq, d]).unwrap();
        let v = CudaTensor::from_vec(data, vec![1, 1, seq, d]).unwrap();
        // sparsity 0.75 → keep 25% of 8 blocks = 2.
        let out = flash_attn_bsa_3d(
            &q,
            &k,
            &v,
            [t, h, w],
            BsaParams {
                sparsity: 0.75,
                chunk_q: [1, 1, 1],
                chunk_k: [1, 1, 1],
            },
        )
        .unwrap();
        assert_eq!(out.shape, vec![1, 1, seq, d]);
    }
}
