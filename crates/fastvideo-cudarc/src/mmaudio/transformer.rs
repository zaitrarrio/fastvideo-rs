//! The MMAudio flow-matching DiT (`mmaudio/model/networks.py`, `MMAudio`),
//! `large_44k_v2`: 7 joint blocks (audio latent, CLIP and text streams
//! attending jointly) then 14 fused blocks on the latent alone, hidden 896,
//! 14 heads of 64.
//!
//! Load-time rewrites (the math is unchanged):
//!
//! * `SelfAttention.qkv` splits its output as `b n (h d j)` (q, k, v
//!   interleaved per channel). The rows are regrouped to `[q | k | v]`.
//! * RoPE rotates interleaved pairs `(x[2i], x[2i+1])`. q and k (and the
//!   q/k RMSNorm weights) get their head channels reordered to
//!   `(0, 2, …, 62, 1, 3, …, 63)`; the rotation is then the half-split one
//!   (`rope_half`) and `q·k` is unchanged. v keeps its order.
//!
//! Conditions (`preprocess_conditions`) are computed once per generation
//! ([`Conditions`]); [`MmAudioTransformer::predict_flow`] is the per-step
//! network. Batch size 1 per call: classifier-free guidance runs the
//! conditional and the empty conditions as two calls, as upstream does.

use fastvideo_models::mmaudio::{nearest_exact_indices, rope_angles, MmAudioDiTConfig};

use super::layers::{
    chunk_last, gather_rows, host_values, interleave_to_half, linear, linear_from, mean_seq,
    modulate, msg, pinned, rope_tables, Conv1d, ConvMlp, Mlp,
};
use crate::wan::dump;
use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::WeightMap;

/// `SelfAttention` + the rest of `MMDitSingleBlock`.
struct SingleBlock {
    /// `adaLN_modulation.1`: `2 D` (pre-only) or `6 D`.
    ada: Linear,
    qkv: Linear,
    q_norm: CudaTensor,
    k_norm: CudaTensor,
    pre_only: bool,
    /// `linear1` / `ffn`: convolutions (k = 3) or linears (text stream).
    post: Option<Post>,
}

enum Post {
    Conv { linear1: Conv1d, ffn: ConvMlp },
    Linear { linear1: Linear, ffn: Mlp },
}

struct Qkv {
    q: CudaTensor,
    k: CudaTensor,
    v: CudaTensor,
}

/// `gate_msa, shift_mlp, scale_mlp, gate_mlp`.
struct PostMod {
    gate_msa: CudaTensor,
    shift_mlp: CudaTensor,
    scale_mlp: CudaTensor,
    gate_mlp: CudaTensor,
}

impl SingleBlock {
    fn load(
        map: &WeightMap,
        prefix: &str,
        cfg: &MmAudioDiTConfig,
        pre_only: bool,
        conv: bool,
    ) -> Result<Self> {
        let d = cfg.hidden_dim;
        let (h, hd) = (cfg.num_heads, cfg.head_dim());
        // Regroup `(h d j)` rows to `[j][h][d']`.
        let w = host_values(map, &format!("{prefix}.attn.qkv.weight"), &[3 * d, d])?;
        let b = host_values(map, &format!("{prefix}.attn.qkv.bias"), &[3 * d])?;
        let perm = interleave_to_half(hd);
        let mut w2 = Vec::with_capacity(w.len());
        let mut b2 = Vec::with_capacity(b.len());
        for j in 0..3 {
            for head in 0..h {
                for p in 0..hd {
                    let dd = if j < 2 { perm[p] } else { p };
                    let row = (head * hd + dd) * 3 + j;
                    w2.extend_from_slice(&w[row * d..(row + 1) * d]);
                    b2.push(b[row]);
                }
            }
        }
        let qn = host_values(map, &format!("{prefix}.attn.q_norm.weight"), &[hd])?;
        let kn = host_values(map, &format!("{prefix}.attn.k_norm.weight"), &[hd])?;
        let q_norm = pinned(perm.iter().map(|&i| qn[i]).collect(), vec![hd])?;
        let k_norm = pinned(perm.iter().map(|&i| kn[i]).collect(), vec![hd])?;
        let ada_out = if pre_only { 2 * d } else { 6 * d };
        let post = if pre_only {
            None
        } else if conv {
            Some(Post::Conv {
                linear1: Conv1d::load(map, &format!("{prefix}.linear1"), d, d, 3, true)?,
                ffn: ConvMlp::load(map, &format!("{prefix}.ffn"), d, cfg.ffn_dim(), 3)?,
            })
        } else {
            Some(Post::Linear {
                linear1: linear(map, &format!("{prefix}.linear1"), d, d, true)?,
                ffn: Mlp::load(map, &format!("{prefix}.ffn"), d, cfg.ffn_dim())?,
            })
        };
        Ok(Self {
            ada: linear(map, &format!("{prefix}.adaLN_modulation.1"), d, ada_out, true)?,
            qkv: linear_from(w2, Some(b2), 3 * d, d)?,
            q_norm,
            k_norm,
            pre_only,
            post,
        })
    }

    /// `pre_attention`: modulated norm, qkv, q/k RMSNorm, RoPE.
    /// `c`: `[1, 1, D]` or `[1, N, D]`.
    fn pre_attention(
        &self,
        x: &CudaTensor,
        c: &CudaTensor,
        rope: Option<&(CudaTensor, CudaTensor)>,
        cfg: &MmAudioDiTConfig,
    ) -> Result<(Qkv, Option<PostMod>)> {
        let m = self.ada.forward(&c.silu())?;
        let (shift_msa, scale_msa, post) = if self.pre_only {
            let ch = chunk_last(&m, 2)?;
            (ch[0].clone(), ch[1].clone(), None)
        } else {
            let ch = chunk_last(&m, 6)?;
            (
                ch[0].clone(),
                ch[1].clone(),
                Some(PostMod {
                    gate_msa: ch[2].clone(),
                    shift_mlp: ch[3].clone(),
                    scale_mlp: ch[4].clone(),
                    gate_mlp: ch[5].clone(),
                }),
            )
        };
        let xn = modulate(&x.layer_norm(1e-5, None, None)?, &shift_msa, &scale_msa)?;
        let qkv = self.qkv.forward(&xn)?;
        let [b, n, _] = qkv.shape[..] else {
            return Err(msg(format!("qkv {:?}", qkv.shape)));
        };
        let (h, hd, d) = (cfg.num_heads, cfg.head_dim(), cfg.hidden_dim);
        let heads = |i: usize| -> Result<CudaTensor> {
            qkv.narrow(2, i * d, d)?
                .reshape(vec![b, n, h, hd])?
                .permute(&[0, 2, 1, 3])
        };
        let mut q = heads(0)?.rms_norm(&self.q_norm, cfg.qk_norm_eps)?;
        let mut k = heads(1)?.rms_norm(&self.k_norm, cfg.qk_norm_eps)?;
        if let Some((cos, sin)) = rope {
            q = q.rope_half(cos, sin)?;
            k = k.rope_half(cos, sin)?;
        }
        Ok((Qkv { q, k, v: heads(2)? }, post))
    }

    /// `post_attention` (a no-op for pre-only blocks). `attn`: `[1, N, D]`.
    fn post_attention(&self, x: &CudaTensor, attn: &CudaTensor, m: Option<PostMod>) -> Result<CudaTensor> {
        let (Some(post), Some(m)) = (self.post.as_ref(), m) else {
            return Ok(x.clone());
        };
        let (a, ffn): (CudaTensor, &dyn Fn(&CudaTensor) -> Result<CudaTensor>) = match post {
            Post::Conv { linear1, ffn } => (linear1.forward_cl(attn)?, &|r| ffn.forward_cl(r)),
            Post::Linear { linear1, ffn } => (linear1.forward(attn)?, &|r| ffn.forward(r)),
        };
        let x = x.add(&a.mul(&m.gate_msa)?)?;
        let r = modulate(&x.layer_norm(1e-5, None, None)?, &m.shift_mlp, &m.scale_mlp)?;
        x.add(&ffn(&r)?.mul(&m.gate_mlp)?)
    }
}

/// `[B, H, N, hd]` attention output to `[B, N, H * hd]`.
fn merge_heads(o: &CudaTensor) -> Result<CudaTensor> {
    let [b, h, n, hd] = o.shape[..] else {
        return Err(msg(format!("attention out {:?}", o.shape)));
    };
    o.permute(&[0, 2, 1, 3])?.reshape(vec![b, n, h * hd])
}

struct JointBlock {
    latent: SingleBlock,
    clip: SingleBlock,
    text: SingleBlock,
}

/// `TimestepEmbedder`.
struct TEmbed {
    freqs: Vec<f32>,
    l0: Linear,
    l2: Linear,
}

impl TEmbed {
    /// `t` is already rounded to the network dtype by the caller.
    fn forward(&self, t: f32) -> Result<CudaTensor> {
        let mut e = Vec::with_capacity(2 * self.freqs.len());
        let args: Vec<f32> = self.freqs.iter().map(|f| t * f).collect();
        e.extend(args.iter().map(|a| a.cos()));
        e.extend(args.iter().map(|a| a.sin()));
        let n = e.len();
        let x = CudaTensor::from_vec(e, vec![1, 1, n])?;
        self.l2.forward(&self.l0.forward(&x)?.silu())
    }
}

/// `PreprocessedConditions` of one sample.
#[derive(Clone)]
pub struct Conditions {
    pub clip_f: CudaTensor,
    /// Upsampled to the latent length: `[1, N, D]`.
    pub sync_f: CudaTensor,
    pub text_f: CudaTensor,
    /// `global_cond_mlp(clip_f_c + text_f_c)`: `[1, 1, D]` (the t-independent
    /// part of `global_c`).
    pub global: CudaTensor,
}

pub struct MmAudioTransformer {
    pub cfg: MmAudioDiTConfig,
    audio_in0: Conv1d,
    audio_in2: ConvMlp,
    clip_in0: Linear,
    clip_in2: ConvMlp,
    sync_in0: Conv1d,
    sync_in2: ConvMlp,
    text_in0: Linear,
    text_in2: Mlp,
    clip_cond_proj: Linear,
    text_cond_proj: Linear,
    global_cond_mlp: Mlp,
    /// `[8 * sync_dim]` host copy (`sync_pos_emb`).
    sync_pos_emb: Vec<f32>,
    t_embed: TEmbed,
    joint: Vec<JointBlock>,
    fused: Vec<SingleBlock>,
    final_ada: Linear,
    final_conv: Conv1d,
    pub latent_mean: Vec<f32>,
    pub latent_std: Vec<f32>,
    pub empty_clip_feat: Vec<f32>,
    pub empty_sync_feat: Vec<f32>,
    pub empty_string_feat: Vec<f32>,
}

impl MmAudioTransformer {
    pub fn load(cfg: MmAudioDiTConfig, map: &WeightMap) -> Result<Self> {
        if !cfg.v2 {
            return Err(msg("MMAudio: only the v2 networks (SiLU projections) are ported"));
        }
        let (d, ff) = (cfg.hidden_dim, cfg.ffn_dim());
        let joint = (0..cfg.joint_depth())
            .map(|i| {
                let p = format!("joint_blocks.{i}");
                let pre_only = i == cfg.joint_depth() - 1;
                Ok(JointBlock {
                    latent: SingleBlock::load(map, &format!("{p}.latent_block"), &cfg, false, true)?,
                    clip: SingleBlock::load(map, &format!("{p}.clip_block"), &cfg, pre_only, true)?,
                    text: SingleBlock::load(map, &format!("{p}.text_block"), &cfg, pre_only, false)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let fused = (0..cfg.fused_depth)
            .map(|i| SingleBlock::load(map, &format!("fused_blocks.{i}"), &cfg, false, true))
            .collect::<Result<Vec<_>>>()?;
        let tf = cfg.t_freq_dim();
        let rb = |v: Vec<f32>| -> Vec<f32> {
            if cfg.bf16_buffers {
                v.into_iter().map(fastvideo_models::mmaudio::bf16_round).collect()
            } else {
                v
            }
        };
        Ok(Self {
            audio_in0: Conv1d::load(map, "audio_input_proj.0", cfg.latent_dim, d, 7, true)?,
            audio_in2: ConvMlp::load(map, "audio_input_proj.2", d, ff, 7)?,
            clip_in0: linear(map, "clip_input_proj.0", cfg.clip_dim, d, true)?,
            clip_in2: ConvMlp::load(map, "clip_input_proj.2", d, ff, 3)?,
            sync_in0: Conv1d::load(map, "sync_input_proj.0", cfg.sync_dim, d, 7, true)?,
            sync_in2: ConvMlp::load(map, "sync_input_proj.2", d, ff, 3)?,
            text_in0: linear(map, "text_input_proj.0", cfg.text_dim, d, true)?,
            text_in2: Mlp::load(map, "text_input_proj.2", d, ff)?,
            clip_cond_proj: linear(map, "clip_cond_proj", d, d, true)?,
            text_cond_proj: linear(map, "text_cond_proj", d, d, true)?,
            global_cond_mlp: Mlp::load(map, "global_cond_mlp", d, ff)?,
            sync_pos_emb: rb(host_values(map, "sync_pos_emb", &[1, 1, 8, cfg.sync_dim])?),
            t_embed: TEmbed {
                freqs: cfg.t_freqs(),
                l0: linear(map, "t_embed.mlp.0", tf, d, true)?,
                l2: linear(map, "t_embed.mlp.2", d, d, true)?,
            },
            joint,
            fused,
            final_ada: linear(map, "final_layer.adaLN_modulation.1", d, 2 * d, true)?,
            final_conv: Conv1d::load(map, "final_layer.conv", d, cfg.latent_dim, 7, true)?,
            latent_mean: rb(host_values(map, "latent_mean", &[1, 1, cfg.latent_dim])?),
            latent_std: rb(host_values(map, "latent_std", &[1, 1, cfg.latent_dim])?),
            empty_clip_feat: rb(host_values(map, "empty_clip_feat", &[1, cfg.clip_dim])?),
            empty_sync_feat: rb(host_values(map, "empty_sync_feat", &[1, cfg.sync_dim])?),
            empty_string_feat: host_values(
                map,
                "empty_string_feat",
                &[cfg.text_seq_len, cfg.text_dim],
            )?,
            cfg,
        })
    }

    /// `get_empty_clip_sequence`: `[1, clip_len, clip_dim]`.
    pub fn empty_clip(&self, clip_len: usize) -> Result<CudaTensor> {
        CudaTensor::from_vec(self.empty_clip_feat.repeat(clip_len), vec![1, clip_len, self.cfg.clip_dim])
    }

    pub fn empty_sync(&self, sync_len: usize) -> Result<CudaTensor> {
        CudaTensor::from_vec(self.empty_sync_feat.repeat(sync_len), vec![1, sync_len, self.cfg.sync_dim])
    }

    /// `preprocess_conditions` for one sample: `clip_f [1, Lc, 1024]`,
    /// `sync_f [1, Ls, 768]` (`Ls` a multiple of 8), `text_f [1, 77, 1024]`.
    pub fn preprocess(
        &self,
        clip_f: &CudaTensor,
        sync_f: &CudaTensor,
        text_f: &CudaTensor,
        latent_len: usize,
    ) -> Result<Conditions> {
        let ls = sync_f.shape[1];
        let sd = self.cfg.sync_dim;
        if ls % 8 != 0 || sync_f.shape[2] != sd {
            return Err(msg(format!("sync features {:?}", sync_f.shape)));
        }
        // `+ sync_pos_emb` per 8-frame segment.
        let pos = CudaTensor::from_vec(self.sync_pos_emb.repeat(ls / 8), vec![1, ls, sd])?;
        let sync = sync_f.add(&pos)?;
        let clip = self.clip_in2.forward_cl(&self.clip_in0.forward(clip_f)?.silu())?;
        let sync = self.sync_in2.forward_cl(&self.sync_in0.forward_cl(&sync)?.silu())?;
        let text = self.text_in2.forward(&self.text_in0.forward(text_f)?.silu())?;
        // nearest-exact upsample of the sync stream to the latent length.
        let sync = gather_rows(&sync, &nearest_exact_indices(ls, latent_len))?
            .reshape(vec![1, latent_len, self.cfg.hidden_dim])?;
        let clip_c = self.clip_cond_proj.forward(&mean_seq(&clip)?)?;
        let text_c = self.text_cond_proj.forward(&mean_seq(&text)?)?;
        let global = self.global_cond_mlp.forward(&clip_c.add(&text_c)?)?;
        Ok(Conditions {
            clip_f: clip,
            sync_f: sync,
            text_f: text,
            global,
        })
    }

    /// RoPE tables for the latent and the CLIP streams.
    pub fn rotations(&self, latent_len: usize, clip_len: usize) -> Result<[(CudaTensor, CudaTensor); 2]> {
        let hd = self.cfg.head_dim();
        let lat = rope_angles(latent_len, hd, 1.0);
        let clip = rope_angles(clip_len, hd, latent_len as f32 / clip_len as f32);
        Ok([rope_tables(&lat, latent_len, hd)?, rope_tables(&clip, clip_len, hd)?])
    }

    /// `predict_flow` for one sample. `latent`: `[1, N, latent_dim]`
    /// (normalized latent space), `t` already in the network dtype.
    pub fn predict_flow(
        &self,
        latent: &CudaTensor,
        t: f32,
        cond: &Conditions,
        rot: &[(CudaTensor, CudaTensor); 2],
    ) -> Result<CudaTensor> {
        let cfg = &self.cfg;
        let n = latent.shape[1];
        let x = self.audio_in0.forward_cl(latent)?.silu();
        let mut x = self.audio_in2.forward_cl(&x)?;
        let global_c = self.t_embed.forward(t)?.add(&cond.global)?;
        let extended_c = cond.sync_f.add(&global_c)?;
        let mut clip = cond.clip_f.clone();
        let mut text = cond.text_f.clone();
        let (nc, nt) = (clip.shape[1], text.shape[1]);
        let blocks = dump::blocks();
        for (i, blk) in self.joint.iter().enumerate() {
            let (xq, xm) = blk.latent.pre_attention(&x, &extended_c, Some(&rot[0]), cfg)?;
            let (cq, cm) = blk.clip.pre_attention(&clip, &global_c, Some(&rot[1]), cfg)?;
            let (tq, tm) = blk.text.pre_attention(&text, &global_c, None, cfg)?;
            let q = CudaTensor::cat(&[&xq.q, &cq.q, &tq.q], 2)?;
            let k = CudaTensor::cat(&[&xq.k, &cq.k, &tq.k], 2)?;
            let v = CudaTensor::cat(&[&xq.v, &cq.v, &tq.v], 2)?;
            let o = merge_heads(&nn::scaled_dot_product_attention(&q, &k, &v, None)?)?;
            x = blk.latent.post_attention(&x, &o.narrow(1, 0, n)?, xm)?;
            if !blk.clip.pre_only {
                clip = blk.clip.post_attention(&clip, &o.narrow(1, n, nc)?, cm)?;
                text = blk.text.post_attention(&text, &o.narrow(1, n + nc, nt)?, tm)?;
            }
            if blocks {
                dump::tensor(&dump::named(&format!("mm_step00_block_{i}")), &x)?;
            }
        }
        for (i, blk) in self.fused.iter().enumerate() {
            let (q, m) = blk.pre_attention(&x, &extended_c, Some(&rot[0]), cfg)?;
            let o = merge_heads(&nn::scaled_dot_product_attention(&q.q, &q.k, &q.v, None)?)?;
            x = blk.post_attention(&x, &o, m)?;
            if blocks {
                let idx = self.joint.len() + i;
                dump::tensor(&dump::named(&format!("mm_step00_block_{idx}")), &x)?;
            }
        }
        let m = self.final_ada.forward(&global_c.silu())?;
        let ch = chunk_last(&m, 2)?;
        let x = modulate(&x.layer_norm(1e-5, None, None)?, &ch[0], &ch[1])?;
        self.final_conv.forward_cl(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qkv_regroup_is_a_permutation() {
        let perm = interleave_to_half(8);
        assert_eq!(perm, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    }
}
