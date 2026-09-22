//! LingBot Dense / MoE DiT — RoPE + self/cross + dense or routed FFN.

use fastvideo_models::lingbot::{apply_rope_real, rope_freqs, LingBotTransformerConfig};

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

struct Attn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attn {
    fn zeros(dim: usize, heads: usize, bias: bool) -> Result<Self> {
        Ok(Self {
            to_q: Linear::zeros(dim, dim, bias),
            to_k: Linear::zeros(dim, dim, bias),
            to_v: Linear::zeros(dim, dim, bias),
            to_out: Linear::zeros(dim, dim, true),
            heads,
            head_dim: dim / heads,
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize, qkv_bias: bool) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            to_q: Linear::load(map, &key("to_q"), dim, dim, qkv_bias)?,
            to_k: Linear::load(map, &key("to_k"), dim, dim, qkv_bias)?,
            to_v: Linear::load(map, &key("to_v"), dim, dim, qkv_bias)?,
            to_out: Linear::load(map, &key("to_out.0"), dim, dim, true)?,
            heads,
            head_dim: dim / heads,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: Option<&CudaTensor>,
        rope: Option<(&Vec<f32>, &Vec<f32>)>,
    ) -> Result<CudaTensor> {
        let [b, s, _] = match hidden.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("lingbot attn: {:?}", hidden.shape))),
        };
        let q = self.to_q.forward(hidden)?;
        let (k, v, ks) = match encoder {
            Some(enc) => {
                let ks = enc.shape[1];
                (self.to_k.forward(enc)?, self.to_v.forward(enc)?, ks)
            }
            None => (self.to_k.forward(hidden)?, self.to_v.forward(hidden)?, s),
        };
        let to_bhsd = |t: CudaTensor, seq: usize| -> Result<CudaTensor> {
            t.reshape(vec![b, seq, self.heads, self.head_dim])?
                .transpose(1, 2)
        };
        let mut q = to_bhsd(q, s)?;
        let mut k = to_bhsd(k, ks)?;
        let v = to_bhsd(v, ks)?;
        if let Some((cos, sin)) = rope {
            let mut qh = q.host_cow()?.to_vec();
            let mut kh = k.host_cow()?.to_vec();
            apply_rope_real(&mut qh, cos, sin, b, self.heads, s, self.head_dim);
            apply_rope_real(&mut kh, cos, sin, b, self.heads, ks, self.head_dim);
            q = CudaTensor::from_vec(qh, q.shape.clone())?;
            k = CudaTensor::from_vec(kh, k.shape.clone())?;
        }
        let out = nn::scaled_dot_product_attention_masked(&q, &k, &v, None, None)?;
        self.to_out.forward(&out.merge_heads()?)
    }
}

/// Dense GELU FFN (Dense 1.3B path).
struct DenseFfn {
    up: Linear,
    down: Linear,
}

impl DenseFfn {
    fn zeros(dim: usize, mid: usize) -> Result<Self> {
        Ok(Self {
            up: Linear::zeros(dim, mid, true),
            down: Linear::zeros(mid, dim, true),
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, mid: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            up: Linear::load(map, &key("net.0.proj"), dim, mid, true)?,
            down: Linear::load(map, &key("net.2"), mid, dim, true)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.down.forward(&self.up.forward(x)?.gelu_erf())
    }
}

/// One MoE expert: SwiGLU `down(silu(gate(x)) * up(x))`.
struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Expert {
    fn zeros(dim: usize, mid: usize) -> Result<Self> {
        Ok(Self {
            gate: Linear::zeros(dim, mid, false),
            up: Linear::zeros(dim, mid, false),
            down: Linear::zeros(mid, dim, false),
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, mid: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        // Prefer SwiGLU keys; fall back to gate_proj/up_proj/down_proj.
        let (g, u, d) = if map.contains(&key("gate_proj.weight")) {
            ("gate_proj", "up_proj", "down_proj")
        } else {
            ("w1", "w3", "w2")
        };
        Ok(Self {
            gate: Linear::load(map, &key(g), dim, mid, false)?,
            up: Linear::load(map, &key(u), dim, mid, false)?,
            down: Linear::load(map, &key(d), mid, dim, false)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let gate = self.gate.forward(x)?.silu();
        let up = self.up.forward(x)?;
        self.down.forward(&gate.mul(&up)?)
    }
}

/// Routed MoE FFN (FastVideo: sigmoid scores, top-k, optional norm, scale).
struct MoeFfn {
    router: Linear,
    experts: Vec<Expert>,
    top_k: usize,
    sigmoid: bool,
    norm_topk: bool,
    routed_scale: f32,
}

impl MoeFfn {
    fn zeros(cfg: &LingBotTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let mid = cfg.moe_intermediate_size;
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for _ in 0..cfg.num_experts {
            experts.push(Expert::zeros(dim, mid)?);
        }
        Ok(Self {
            router: Linear::zeros(dim, cfg.num_experts, false),
            experts,
            top_k: cfg.num_experts_per_tok.min(cfg.num_experts),
            sigmoid: cfg.score_func_sigmoid,
            norm_topk: cfg.norm_topk_prob,
            routed_scale: cfg.routed_scaling_factor,
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &LingBotTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let mid = cfg.moe_intermediate_size;
        let key = |n: &str| weights::join_key(prefix, n);
        let router = if map.contains(&key("gate.weight")) {
            Linear::load(map, &key("gate"), dim, cfg.num_experts, false)?
        } else {
            Linear::load(map, &key("router"), dim, cfg.num_experts, false)?
        };
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for e in 0..cfg.num_experts {
            let ep = key(&format!("experts.{e}"));
            experts.push(Expert::load(map, &ep, dim, mid)?);
        }
        Ok(Self {
            router,
            experts,
            top_k: cfg.num_experts_per_tok.min(cfg.num_experts),
            sigmoid: cfg.score_func_sigmoid,
            norm_topk: cfg.norm_topk_prob,
            routed_scale: cfg.routed_scaling_factor,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, s, d] = match x.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("moe ffn: {:?}", x.shape))),
        };
        let n_tok = b * s;
        let flat = x.reshape(vec![n_tok, d])?;
        let logits = self.router.forward(&flat)?; // [N, E]
        let mut scores = logits.host_cow()?.to_vec();
        let e = self.experts.len();
        if self.sigmoid {
            for v in &mut scores {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        } else {
            // Softmax per token.
            for ti in 0..n_tok {
                let row = &mut scores[ti * e..][..e];
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0f32;
                for v in row.iter_mut() {
                    *v = (*v - m).exp();
                    z += *v;
                }
                for v in row.iter_mut() {
                    *v /= z.max(1e-20);
                }
            }
        }

        let mut out = vec![0f32; n_tok * d];
        let xh = flat.host_cow()?;
        for ti in 0..n_tok {
            let row = &scores[ti * e..][..e];
            let mut idxs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idxs.truncate(self.top_k);
            let mut weights: Vec<f32> = idxs.iter().map(|x| x.1).collect();
            if self.norm_topk {
                let z: f32 = weights.iter().sum::<f32>().max(1e-20);
                for w in &mut weights {
                    *w /= z;
                }
            }
            let tok = CudaTensor::from_vec(xh[ti * d..][..d].to_vec(), vec![1, d])?;
            for ((ei, _), w) in idxs.iter().zip(&weights) {
                let y = self.experts[*ei].forward(&tok)?;
                let yh = y.host_cow()?;
                let scale = w * self.routed_scale;
                for di in 0..d {
                    out[ti * d + di] += yh[di] * scale;
                }
            }
        }
        Ok(CudaTensor::from_vec(out, vec![b, s, d])?)
    }
}

enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

impl Ffn {
    fn zeros(cfg: &LingBotTransformerConfig) -> Result<Self> {
        if cfg.is_moe() {
            Ok(Self::Moe(MoeFfn::zeros(cfg)?))
        } else {
            Ok(Self::Dense(DenseFfn::zeros(
                cfg.hidden_size,
                cfg.intermediate_size,
            )?))
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &LingBotTransformerConfig) -> Result<Self> {
        if cfg.is_moe() {
            Ok(Self::Moe(MoeFfn::load(map, prefix, cfg)?))
        } else {
            Ok(Self::Dense(DenseFfn::load(
                map,
                prefix,
                cfg.hidden_size,
                cfg.intermediate_size,
            )?))
        }
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        match self {
            Self::Dense(f) => f.forward(x),
            Self::Moe(f) => f.forward(x),
        }
    }
}

struct Block {
    attn1: Attn,
    attn2: Attn,
    ffn: Ffn,
}

impl Block {
    fn zeros(cfg: &LingBotTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        Ok(Self {
            attn1: Attn::zeros(dim, heads, false)?,
            attn2: Attn::zeros(dim, heads, false)?,
            ffn: Ffn::zeros(cfg)?,
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &LingBotTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            attn1: Attn::load(map, &key("attn1"), dim, heads, false)?,
            attn2: Attn::load(map, &key("attn2"), dim, heads, false)?,
            ffn: Ffn::load(map, &key("ffn"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &CudaTensor,
        encoder: &CudaTensor,
        rope: (&Vec<f32>, &Vec<f32>),
    ) -> Result<CudaTensor> {
        let n = nn::layer_norm(x, 1e-6, None, None)?;
        let a = self.attn1.forward(&n, None, Some(rope))?;
        let mut h = x.add(&a)?;
        let n = nn::layer_norm(&h, 1e-6, None, None)?;
        let a = self.attn2.forward(&n, Some(encoder), None)?;
        h = h.add(&a)?;
        let n = nn::layer_norm(&h, 1e-6, None, None)?;
        h.add(&self.ffn.forward(&n)?)
    }
}

pub struct LingBotTransformer {
    pub cfg: LingBotTransformerConfig,
    patch: Linear,
    time_1: Linear,
    time_2: Linear,
    text_proj: Linear,
    blocks: Vec<Block>,
    norm_out: bool,
    proj_out: Linear,
}

impl LingBotTransformer {
    pub fn zeros(cfg: LingBotTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let [pt, ph, pw] = cfg.patch_size;
        let patch_in = cfg.in_channels * pt * ph * pw;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for _ in 0..cfg.depth {
            blocks.push(Block::zeros(&cfg)?);
        }
        Ok(Self {
            patch: Linear::zeros(patch_in, dim, true),
            time_1: Linear::zeros(cfg.freq_dim, dim, true),
            time_2: Linear::zeros(dim, dim, true),
            text_proj: Linear::zeros(cfg.text_dim, dim, true),
            blocks,
            norm_out: true,
            proj_out: Linear::zeros(dim, pt * ph * pw * cfg.out_channels, true),
            cfg,
        })
    }

    pub fn load(cfg: LingBotTransformerConfig, map: &WeightMap) -> Result<Self> {
        let dim = cfg.hidden_size;
        let [pt, ph, pw] = cfg.patch_size;
        let patch_in = cfg.in_channels * pt * ph * pw;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Block::load(map, &format!("blocks.{i}"), &cfg)?);
        }
        Ok(Self {
            patch: Linear::load(map, "patch_embedding", patch_in, dim, true)?,
            time_1: Linear::load(map, "condition_embedder.time_proj", cfg.freq_dim, dim, true)?,
            time_2: Linear::load(map, "condition_embedder.time_embedder", dim, dim, true)?,
            text_proj: Linear::load(map, "condition_embedder.text_embedder", cfg.text_dim, dim, true)?,
            blocks,
            norm_out: true,
            proj_out: Linear::load(map, "proj_out", dim, pt * ph * pw * cfg.out_channels, true)?,
            cfg,
        })
    }

    pub fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match hidden.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg(format!("lingbot dit: {:?}", hidden.shape))),
        };
        let [pt, ph, pw] = self.cfg.patch_size;
        let pe_t = t / pt;
        let pe_h = h / ph;
        let pe_w = w / pw;
        let seq = pe_t * pe_h * pe_w;
        let (cos, sin) = rope_freqs(&self.cfg, t, h, w);

        let host = hidden.host_cow()?;
        let mut tokens = vec![0f32; b * seq * c * pt * ph * pw];
        for bi in 0..b {
            for ti in 0..pe_t {
                for yi in 0..pe_h {
                    for xi in 0..pe_w {
                        let tok = (ti * pe_h + yi) * pe_w + xi;
                        let mut o = 0usize;
                        for pti in 0..pt {
                            for phi in 0..ph {
                                for pwi in 0..pw {
                                    for ci in 0..c {
                                        let src = ((((bi * c + ci) * t + ti * pt + pti) * h
                                            + yi * ph
                                            + phi)
                                            * w
                                            + xi * pw
                                            + pwi);
                                        tokens[((bi * seq + tok) * c * pt * ph * pw) + o] =
                                            host[src];
                                        o += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let flat = CudaTensor::from_vec(tokens, vec![b * seq, c * pt * ph * pw])?;
        let mut hs = self
            .patch
            .forward(&flat)?
            .reshape(vec![b, seq, self.cfg.hidden_size])?;

        let temb = {
            let freqs = nn::sinusoidal_timesteps(
                &CudaTensor::from_vec(vec![timestep; b], vec![b])?,
                self.cfg.freq_dim,
            )?;
            self.time_2.forward(&self.time_1.forward(&freqs)?.silu())?
        };
        let temb_s = {
            let th = temb.host_cow()?;
            let mut out = vec![0f32; b * seq * self.cfg.hidden_size];
            for bi in 0..b {
                for s in 0..seq {
                    out[(bi * seq + s) * self.cfg.hidden_size..][..self.cfg.hidden_size]
                        .copy_from_slice(&th[bi * self.cfg.hidden_size..][..self.cfg.hidden_size]);
                }
            }
            CudaTensor::from_vec(out, vec![b, seq, self.cfg.hidden_size])?
        };
        hs = hs.add(&temb_s)?;

        let enc = self.text_proj.forward(encoder)?;
        for block in &self.blocks {
            hs = block.forward(&hs, &enc, (&cos, &sin))?;
        }
        if self.norm_out {
            hs = nn::layer_norm(&hs, self.cfg.norm_eps, None, None)?;
        }
        let out = self.proj_out.forward(&hs)?;
        let oc = self.cfg.out_channels;
        let oh = out.host_cow()?;
        let mut pixels = vec![0f32; b * oc * t * h * w];
        for bi in 0..b {
            for ti in 0..pe_t {
                for yi in 0..pe_h {
                    for xi in 0..pe_w {
                        let tok = (ti * pe_h + yi) * pe_w + xi;
                        let mut o = 0usize;
                        for pti in 0..pt {
                            for phi in 0..ph {
                                for pwi in 0..pw {
                                    for ci in 0..oc {
                                        let dst = ((((bi * oc + ci) * t + ti * pt + pti) * h
                                            + yi * ph
                                            + phi)
                                            * w
                                            + xi * pw
                                            + pwi);
                                        pixels[dst] = oh[((bi * seq + tok)
                                            * (pt * ph * pw * oc))
                                            + o];
                                        o += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(CudaTensor::from_vec(pixels, vec![b, oc, t, h, w])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shapes() {
        let cfg = LingBotTransformerConfig::tiny();
        let dit = LingBotTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_dim]);
        let out = dit.forward(&x, &enc, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
    }

    #[test]
    fn tiny_moe_forward_shapes() {
        let cfg = LingBotTransformerConfig::tiny_moe();
        assert!(cfg.is_moe());
        let dit = LingBotTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_dim]);
        let out = dit.forward(&x, &enc, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
    }
}
