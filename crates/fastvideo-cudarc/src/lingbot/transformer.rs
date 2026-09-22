//! LingBot Dense DiT — tiny-forward scaffold (RoPE + self/cross + FFN).

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

struct Ffn {
    up: Linear,
    down: Linear,
}

impl Ffn {
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

struct Block {
    norm1: bool,
    attn1: Attn,
    norm2: bool,
    attn2: Attn,
    norm3: bool,
    ffn: Ffn,
}

impl Block {
    fn zeros(cfg: &LingBotTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        Ok(Self {
            norm1: true,
            attn1: Attn::zeros(dim, heads, false)?,
            norm2: true,
            attn2: Attn::zeros(dim, heads, false)?,
            norm3: true,
            ffn: Ffn::zeros(dim, cfg.intermediate_size)?,
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &LingBotTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            norm1: true,
            attn1: Attn::load(map, &key("attn1"), dim, heads, false)?,
            norm2: true,
            attn2: Attn::load(map, &key("attn2"), dim, heads, false)?,
            norm3: true,
            ffn: Ffn::load(map, &key("ffn"), dim, cfg.intermediate_size)?,
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
        if self.cfg.is_moe() {
            // Dense path only for this stage; MoE routed FFN pending.
        }
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
        // Add time to tokens (broadcast).
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
}
