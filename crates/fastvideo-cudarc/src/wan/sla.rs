//! Sparse-Linear Attention (SLA) for TurboWan / TurboDiffusion.
//!
//! Port of FastVideo `attention/backends/sla.py` (Zhang et al., arXiv:2509.24006):
//! block top-k sparse attention + linear attention with a softmax feature map,
//! combined as `o_s + proj_l(o_l)`. Block sizes default to TurboDiffusion's
//! `BLKQ=128`, `BLKK=64`, `topk_ratio=0.1`.
//!
//! Algorithmic host path (gather / GEMM). SageSLA INT8/FP8 kernels are not
//! ported; set `FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN` to use this.

use super::envflag::{f32_flag, string_flag, usize_flag};
use super::nn::Linear;
use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

#[derive(Debug, Clone, Copy)]
pub struct SlaConfig {
    pub topk_ratio: f32,
    pub blk_q: usize,
    pub blk_k: usize,
}

impl SlaConfig {
    pub fn from_env() -> Self {
        Self {
            topk_ratio: f32_flag("FASTVIDEO_SLA_TOPK", 0.1).clamp(0.01, 1.0),
            blk_q: usize_flag("FASTVIDEO_SLA_BLKQ", 128).max(1),
            blk_k: usize_flag("FASTVIDEO_SLA_BLKK", 64).max(1),
        }
    }
}

/// `FASTVIDEO_ATTENTION_BACKEND` is `SLA_ATTN` / `sla` / `sagesla`.
pub fn sla_enabled() -> bool {
    let b = string_flag("FASTVIDEO_ATTENTION_BACKEND", "");
    b.eq_ignore_ascii_case("SLA_ATTN")
        || b.eq_ignore_ascii_case("sla")
        || b.eq_ignore_ascii_case("sagesla")
        || b.eq_ignore_ascii_case("SAGE_SLA_ATTN")
}

/// SLA over `[B, H, S, D]` Q/K/V. `proj_l` is `Linear(D→D)` on the last dim.
pub fn sla_attention(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    proj_l: Option<&Linear>,
    cfg: &SlaConfig,
) -> Result<CudaTensor> {
    let [b, h, s, d] = match q.shape[..] {
        [b, h, s, d] => [b, h, s, d],
        _ => return Err(msg(format!("sla: q shape {:?} want [B,H,S,D]", q.shape))),
    };
    if k.shape != q.shape || v.shape != q.shape {
        return Err(msg(format!(
            "sla: q/k/v shapes {:?}/{:?}/{:?}",
            q.shape, k.shape, v.shape
        )));
    }
    let qh = q.host_cow()?.into_owned();
    let kh = k.host_cow()?.into_owned();
    let vh = v.host_cow()?.into_owned();

    if let Some(p) = proj_l {
        let w = p.weight.host_cow()?.into_owned();
        if w.len() != d * d {
            let (o_s, o_l) = sla_host_branches(&qh, &kh, &vh, b, h, s, d, cfg)?;
            let o_l_t = CudaTensor::from_vec(o_l, vec![b * h * s, d])?.to_device()?;
            let projected = p.forward(&o_l_t)?.reshape(vec![b, h, s, d])?;
            let o_s_t = CudaTensor::from_vec(o_s, vec![b, h, s, d])?.to_device()?;
            return o_s_t.add(&projected);
        }
        let bias = match &p.bias {
            Some(b) => Some(b.host_cow()?.into_owned()),
            None => None,
        };
        let out = sla_host_inner(
            &qh,
            &kh,
            &vh,
            b,
            h,
            s,
            d,
            cfg,
            Some((w.as_slice(), bias.as_deref())),
        )?;
        return CudaTensor::from_vec(out, vec![b, h, s, d])?.to_device();
    }

    let out = sla_host_inner(&qh, &kh, &vh, b, h, s, d, cfg, None)?;
    CudaTensor::from_vec(out, vec![b, h, s, d])?.to_device()
}

/// Host SLA without `proj_l` (raw linear branch added to sparse).
pub fn sla_attention_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    cfg: &SlaConfig,
) -> Result<Vec<f32>> {
    sla_host_inner(q, k, v, b, h, s, d, cfg, None)
}

fn sla_host_branches(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    heads: usize,
    s: usize,
    d: usize,
    cfg: &SlaConfig,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = b * heads * s * d;
    let mut o_s_all = vec![0f32; n];
    let mut o_l_all = vec![0f32; n];
    for bh in 0..(b * heads) {
        let base = bh * s * d;
        let (o_s, o_l) = one_head_branches(
            &q[base..base + s * d],
            &k[base..base + s * d],
            &v[base..base + s * d],
            s,
            d,
            cfg,
        )?;
        o_s_all[base..base + s * d].copy_from_slice(&o_s);
        o_l_all[base..base + s * d].copy_from_slice(&o_l);
    }
    Ok((o_s_all, o_l_all))
}

fn sla_host_inner(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    heads: usize,
    s: usize,
    d: usize,
    cfg: &SlaConfig,
    proj: Option<(&[f32], Option<&[f32]>)>,
) -> Result<Vec<f32>> {
    let n = b * heads * s * d;
    if q.len() != n || k.len() != n || v.len() != n {
        return Err(msg("sla host: buffer length mismatch"));
    }
    let mut out = vec![0f32; n];
    for bh in 0..(b * heads) {
        let base = bh * s * d;
        let (o_s, mut o_l) = one_head_branches(
            &q[base..base + s * d],
            &k[base..base + s * d],
            &v[base..base + s * d],
            s,
            d,
            cfg,
        )?;
        if let Some((w, bias)) = proj {
            if w.len() != d * d {
                return Err(msg("sla proj_l weight shape mismatch"));
            }
            let mut projected = vec![0f32; s * d];
            for t in 0..s {
                for o in 0..d {
                    let mut acc = 0f32;
                    for i in 0..d {
                        acc += w[o * d + i] * o_l[t * d + i];
                    }
                    if let Some(b) = bias {
                        acc += b[o];
                    }
                    projected[t * d + o] = acc;
                }
            }
            o_l = projected;
        }
        for i in 0..s * d {
            out[base + i] = o_s[i] + o_l[i];
        }
    }
    Ok(out)
}

fn one_head_branches(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    s: usize,
    d: usize,
    cfg: &SlaConfig,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let scale = 1.0 / (d as f32).sqrt();
    let nq = (s + cfg.blk_q - 1) / cfg.blk_q;
    let nk = (s + cfg.blk_k - 1) / cfg.blk_k;
    let topk = ((cfg.topk_ratio * nk as f32).round() as usize).clamp(1, nk);

    let mut k_smooth = kh.to_vec();
    for di in 0..d {
        let mut m = 0f32;
        for t in 0..s {
            m += k_smooth[t * d + di];
        }
        m /= s as f32;
        for t in 0..s {
            k_smooth[t * d + di] -= m;
        }
    }

    let pooled_q = mean_pool(qh, s, d, cfg.blk_q);
    let pooled_k = mean_pool(&k_smooth, s, d, cfg.blk_k);
    let mut scores = vec![0f32; nq * nk];
    for qi in 0..nq {
        for kj in 0..nk {
            let mut acc = 0f32;
            for di in 0..d {
                acc += pooled_q[qi * d + di] * pooled_k[kj * d + di];
            }
            scores[qi * nk + kj] = acc;
        }
    }
    let lut = topk_indices(&scores, nq, nk, topk);

    let mut o_s = vec![0f32; s * d];
    for qi in 0..nq {
        let q0 = qi * cfg.blk_q;
        let q1 = (q0 + cfg.blk_q).min(s);
        let qlen = q1 - q0;
        let mut k_sel = Vec::with_capacity(topk * cfg.blk_k * d);
        let mut v_sel = Vec::with_capacity(topk * cfg.blk_k * d);
        let mut klen = 0usize;
        for &kj in &lut[qi * topk..(qi + 1) * topk] {
            let k0 = kj * cfg.blk_k;
            let k1 = (k0 + cfg.blk_k).min(s);
            for t in k0..k1 {
                k_sel.extend_from_slice(&kh[t * d..(t + 1) * d]);
                v_sel.extend_from_slice(&vh[t * d..(t + 1) * d]);
                klen += 1;
            }
        }
        for local_q in 0..qlen {
            let qrow = &qh[(q0 + local_q) * d..(q0 + local_q + 1) * d];
            let mut logits = vec![0f32; klen];
            let mut mx = f32::NEG_INFINITY;
            for ki in 0..klen {
                let mut acc = 0f32;
                for di in 0..d {
                    acc += qrow[di] * k_sel[ki * d + di];
                }
                let v = acc * scale;
                logits[ki] = v;
                if v > mx {
                    mx = v;
                }
            }
            let mut sum = 0f32;
            for l in &mut logits {
                *l = (*l - mx).exp();
                sum += *l;
            }
            let inv = 1.0 / sum.max(1e-20);
            for di in 0..d {
                let mut acc = 0f32;
                for ki in 0..klen {
                    acc += logits[ki] * inv * v_sel[ki * d + di];
                }
                o_s[(q0 + local_q) * d + di] = acc;
            }
        }
    }

    let mut q_lin = vec![0f32; s * d];
    let mut k_lin = vec![0f32; s * d];
    for t in 0..s {
        softmax_row(&qh[t * d..(t + 1) * d], &mut q_lin[t * d..(t + 1) * d]);
        softmax_row(&kh[t * d..(t + 1) * d], &mut k_lin[t * d..(t + 1) * d]);
    }
    let mut kvsum = vec![0f32; d * d];
    let mut ksum = vec![0f32; d];
    for t in 0..s {
        for i in 0..d {
            ksum[i] += k_lin[t * d + i];
            for j in 0..d {
                kvsum[i * d + j] += k_lin[t * d + i] * vh[t * d + j];
            }
        }
    }
    let mut o_l = vec![0f32; s * d];
    for t in 0..s {
        let mut denom = 0f32;
        for i in 0..d {
            denom += q_lin[t * d + i] * ksum[i];
        }
        let inv = 1.0 / (denom + 1e-5);
        for j in 0..d {
            let mut acc = 0f32;
            for i in 0..d {
                acc += q_lin[t * d + i] * kvsum[i * d + j];
            }
            o_l[t * d + j] = acc * inv;
        }
    }
    Ok((o_s, o_l))
}

fn mean_pool(x: &[f32], s: usize, d: usize, blk: usize) -> Vec<f32> {
    let nblocks = (s + blk - 1) / blk;
    let mut out = vec![0f32; nblocks * d];
    for bi in 0..nblocks {
        let t0 = bi * blk;
        let t1 = (t0 + blk).min(s);
        let n = (t1 - t0) as f32;
        for t in t0..t1 {
            for di in 0..d {
                out[bi * d + di] += x[t * d + di];
            }
        }
        for di in 0..d {
            out[bi * d + di] /= n;
        }
    }
    out
}

fn topk_indices(scores: &[f32], nq: usize, nk: usize, topk: usize) -> Vec<usize> {
    let mut lut = vec![0usize; nq * topk];
    for qi in 0..nq {
        let row = &scores[qi * nk..(qi + 1) * nk];
        let mut idx: Vec<usize> = (0..nk).collect();
        idx.select_nth_unstable_by(topk - 1, |&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for t in 0..topk {
            lut[qi * topk + t] = idx[t];
        }
    }
    lut
}

fn softmax_row(x: &[f32], out: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for &v in x {
        if v > mx {
            mx = v;
        }
    }
    let mut sum = 0f32;
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = (v - mx).exp();
        sum += *o;
    }
    let inv = 1.0 / sum.max(1e-20);
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// Optional `proj_l` under an attention prefix; `None` when absent.
pub fn load_proj_l(map: &super::weights::WeightMap, prefix: &str, head_dim: usize) -> Result<Option<Linear>> {
    let key = |name: &str| super::weights::join_key(prefix, name);
    let candidates = [
        key("proj_l"),
        key("attn_op.local_attn.proj_l"),
        key("local_attn.proj_l"),
    ];
    for p in &candidates {
        if map.has_tensor(&format!("{p}.weight")) {
            return Ok(Some(Linear::load(map, p, head_dim, head_dim, true)?));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sla_host_runs_small() {
        let (b, h, s, d) = (1usize, 2, 32, 8);
        let n = b * h * s * d;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();
        let cfg = SlaConfig {
            topk_ratio: 0.5,
            blk_q: 8,
            blk_k: 8,
        };
        let out = sla_attention_host(&q, &k, &v, b, h, s, d, &cfg).unwrap();
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn topk_picks_largest() {
        let scores = vec![0.1f32, 0.9, 0.5, 0.2];
        let lut = topk_indices(&scores, 1, 4, 2);
        assert!(lut.contains(&1));
        assert!(lut.contains(&2));
    }
}
