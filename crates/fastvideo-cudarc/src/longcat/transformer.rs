//! LongCat DiT (`LongCatVideoTransformer3DModel`) with optional BSA.

use fastvideo_models::longcat::LongCatTransformerConfig;

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

use super::bsa::{flash_attn_bsa_3d, BsaParams};

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
    fn zeros(dim: usize, heads: usize) -> Result<Self> {
        let head_dim = dim / heads;
        Ok(Self {
            to_q: Linear::zeros(dim, dim, false),
            to_k: Linear::zeros(dim, dim, false),
            to_v: Linear::zeros(dim, dim, false),
            to_out: Linear::zeros(dim, dim, false),
            heads,
            head_dim,
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize) -> Result<Self> {
        let head_dim = dim / heads;
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            to_q: Linear::load(map, &key("to_q"), dim, dim, false)?,
            to_k: Linear::load(map, &key("to_k"), dim, dim, false)?,
            to_v: Linear::load(map, &key("to_v"), dim, dim, false)?,
            to_out: Linear::load(map, &key("to_out"), dim, dim, false)?,
            heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: Option<&CudaTensor>,
        bsa: Option<( [usize; 3], BsaParams )>,
    ) -> Result<CudaTensor> {
        let [b, s, _] = match hidden.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("longcat attn: {:?}", hidden.shape))),
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
        let q = to_bhsd(q, s)?;
        let k = to_bhsd(k, ks)?;
        let v = to_bhsd(v, ks)?;
        // BSA is self-attn only (T>1); cross-attn stays dense.
        let out = match (encoder, bsa) {
            (None, Some((thw, params))) if thw[0] > 1 => {
                flash_attn_bsa_3d(&q, &k, &v, thw, params)?
            }
            _ => nn::scaled_dot_product_attention_masked(&q, &k, &v, None, None)?,
        };
        self.to_out.forward(&out.merge_heads()?)
    }
}

struct SwiGlu {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGlu {
    fn zeros(dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            w1: Linear::zeros(dim, hidden, false),
            w2: Linear::zeros(hidden, dim, false),
            w3: Linear::zeros(dim, hidden, false),
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, hidden: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            w1: Linear::load(map, &key("w1"), dim, hidden, false)?,
            w2: Linear::load(map, &key("w2"), hidden, dim, false)?,
            w3: Linear::load(map, &key("w3"), dim, hidden, false)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let gate = self.w1.forward(x)?.silu();
        let up = self.w3.forward(x)?;
        self.w2.forward(&gate.mul(&up)?)
    }
}

struct Block {
    adaln: Linear,
    norm_attn: bool,
    attn: Attn,
    norm_cross: CudaTensor,
    cross: Attn,
    ffn: SwiGlu,
    dim: usize,
}

impl Block {
    fn zeros(cfg: &LongCatTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        Ok(Self {
            adaln: Linear::zeros(cfg.adaln_tembed_dim, 6 * dim, true),
            norm_attn: true,
            attn: Attn::zeros(dim, cfg.num_heads)?,
            norm_cross: CudaTensor::ones(&[dim]),
            cross: Attn::zeros(dim, cfg.num_heads)?,
            ffn: SwiGlu::zeros(dim, cfg.mlp_hidden())?,
            dim,
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &LongCatTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            adaln: Linear::load(map, &key("adaln_linear_1"), cfg.adaln_tembed_dim, 6 * dim, true)?,
            norm_attn: true,
            attn: Attn::load(map, &key("self_attn"), dim, cfg.num_heads)?,
            norm_cross: weights::cuda_tensor_shaped(map, &key("norm_cross.weight"), &[dim])?,
            cross: Attn::load(map, &key("cross_attn"), dim, cfg.num_heads)?,
            ffn: SwiGlu::load(map, &key("ffn"), dim, cfg.mlp_hidden())?,
            dim,
        })
    }

    fn forward(
        &self,
        x: &CudaTensor,
        y: &CudaTensor,
        temb: &CudaTensor,
        bsa: Option<( [usize; 3], BsaParams )>,
    ) -> Result<CudaTensor> {
        let mods = self.adaln.forward(&temb.silu())?;
        let shift_msa = mods.narrow(1, 0, self.dim)?;
        let scale_msa = mods.narrow(1, self.dim, self.dim)?;
        let gate_msa = mods.narrow(1, 2 * self.dim, self.dim)?;
        let shift_mlp = mods.narrow(1, 3 * self.dim, self.dim)?;
        let scale_mlp = mods.narrow(1, 4 * self.dim, self.dim)?;
        let gate_mlp = mods.narrow(1, 5 * self.dim, self.dim)?;

        let n = if self.norm_attn {
            nn::layer_norm(x, 1e-6, None, None)?
        } else {
            x.clone()
        };
        let n = n
            .mul(&expand_token(&scale_msa, x.shape[1])?.try_add_scalar(1.0)?)?
            .add(&expand_token(&shift_msa, x.shape[1])?)?;
        let a = self.attn.forward(&n, None, bsa)?;
        let mut h = x.add(&a.mul(&expand_token(&gate_msa, x.shape[1])?)?)?;

        let nc = h.rms_norm(&self.norm_cross, 1e-6)?;
        let c = self.cross.forward(&nc, Some(y), None)?;
        h = h.add(&c)?;

        let n = nn::layer_norm(&h, 1e-6, None, None)?;
        let n = n
            .mul(&expand_token(&scale_mlp, h.shape[1])?.try_add_scalar(1.0)?)?
            .add(&expand_token(&shift_mlp, h.shape[1])?)?;
        let f = self.ffn.forward(&n)?;
        h.add(&f.mul(&expand_token(&gate_mlp, h.shape[1])?)?)
    }
}

fn expand_token(x: &CudaTensor, seq: usize) -> Result<CudaTensor> {
    let [b, c] = match x.shape[..] {
        [b, c] => [b, c],
        _ => return Err(msg(format!("expand_token: {:?}", x.shape))),
    };
    let host = x.host_cow()?;
    let mut out = vec![0f32; b * seq * c];
    for bi in 0..b {
        for s in 0..seq {
            out[(bi * seq + s) * c..][..c].copy_from_slice(&host[bi * c..][..c]);
        }
    }
    Ok(CudaTensor::from_vec(out, vec![b, seq, c])?)
}

pub struct LongCatTransformer {
    pub cfg: LongCatTransformerConfig,
    patch: Linear,
    time_1: Linear,
    time_2: Linear,
    caption_1: Linear,
    caption_2: Linear,
    blocks: Vec<Block>,
    final_adaln: Linear,
    final_norm: bool,
    final_proj: Linear,
}

impl LongCatTransformer {
    pub fn zeros(cfg: LongCatTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let [pt, ph, pw] = cfg.patch_size;
        let patch_in = cfg.in_channels * pt * ph * pw;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for _ in 0..cfg.depth {
            blocks.push(Block::zeros(&cfg)?);
        }
        Ok(Self {
            patch: Linear::zeros(patch_in, dim, false),
            time_1: Linear::zeros(cfg.frequency_embedding_size, cfg.adaln_tembed_dim, true),
            time_2: Linear::zeros(cfg.adaln_tembed_dim, cfg.adaln_tembed_dim, true),
            caption_1: Linear::zeros(cfg.caption_channels, dim, true),
            caption_2: Linear::zeros(dim, dim, true),
            blocks,
            final_adaln: Linear::zeros(cfg.adaln_tembed_dim, 2 * dim, true),
            final_norm: true,
            final_proj: Linear::zeros(dim, pt * ph * pw * cfg.out_channels, false),
            cfg,
        })
    }

    pub fn load(cfg: LongCatTransformerConfig, map: &WeightMap) -> Result<Self> {
        let dim = cfg.hidden_size;
        let [pt, ph, pw] = cfg.patch_size;
        let patch_in = cfg.in_channels * pt * ph * pw;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Block::load(map, &format!("blocks.{i}"), &cfg)?);
        }
        Ok(Self {
            patch: Linear::load(map, "patch_embed.proj", patch_in, dim, false)?,
            time_1: Linear::load(
                map,
                "time_embedder.linear_1",
                cfg.frequency_embedding_size,
                cfg.adaln_tembed_dim,
                true,
            )?,
            time_2: Linear::load(
                map,
                "time_embedder.linear_2",
                cfg.adaln_tembed_dim,
                cfg.adaln_tembed_dim,
                true,
            )?,
            caption_1: Linear::load(map, "caption_embedder.linear_1", cfg.caption_channels, dim, true)?,
            caption_2: Linear::load(map, "caption_embedder.linear_2", dim, dim, true)?,
            blocks,
            final_adaln: Linear::load(map, "final_layer.adaln_linear", cfg.adaln_tembed_dim, 2 * dim, true)?,
            final_norm: true,
            final_proj: Linear::load(
                map,
                "final_layer.proj",
                dim,
                pt * ph * pw * cfg.out_channels,
                false,
            )?,
            cfg,
        })
    }

    pub fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        self.forward_with_bsa(hidden, encoder, timestep, self.cfg.enable_bsa)
    }

    /// Like [`Self::forward`] but overrides `enable_bsa` (CLI / preset runtime).
    pub fn forward_with_bsa(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        timestep: f32,
        enable_bsa: bool,
    ) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match hidden.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg(format!("longcat dit: {:?}", hidden.shape))),
        };
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "longcat channels {c} vs {}",
                self.cfg.in_channels
            )));
        }

        let [pt, ph, pw] = self.cfg.patch_size;
        let pe_t = t / pt;
        let pe_h = h / ph;
        let pe_w = w / pw;
        let seq = pe_t * pe_h * pe_w;
        let bsa = if enable_bsa {
            Some((
                [pe_t, pe_h, pe_w],
                BsaParams::from_config(self.cfg.bsa_sparsity, self.cfg.bsa_chunk),
            ))
        } else {
            None
        };
        let host = hidden.host_cow()?;
        let cin = c;
        let mut tokens = vec![0f32; b * seq * cin * pt * ph * pw];
        for bi in 0..b {
            for ti in 0..pe_t {
                for yi in 0..pe_h {
                    for xi in 0..pe_w {
                        let tok = (ti * pe_h + yi) * pe_w + xi;
                        let mut o = 0usize;
                        for pti in 0..pt {
                            for phi in 0..ph {
                                for pwi in 0..pw {
                                    for ci in 0..cin {
                                        let src = ((((bi * cin + ci) * t + ti * pt + pti) * h
                                            + yi * ph
                                            + phi)
                                            * w
                                            + xi * pw
                                            + pwi);
                                        tokens[((bi * seq + tok) * cin * pt * ph * pw) + o] =
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
        let flat = CudaTensor::from_vec(tokens, vec![b * seq, cin * pt * ph * pw])?;
        let mut hs = self
            .patch
            .forward(&flat)?
            .reshape(vec![b, seq, self.cfg.hidden_size])?;

        let temb = {
            let freqs = nn::sinusoidal_timesteps(
                &CudaTensor::from_vec(vec![timestep; b], vec![b])?,
                self.cfg.frequency_embedding_size,
            )?;
            self.time_2.forward(&self.time_1.forward(&freqs)?.silu())?
        };
        let y = self
            .caption_2
            .forward(&self.caption_1.forward(encoder)?.silu())?;

        for block in &self.blocks {
            hs = block.forward(&hs, &y, &temb, bsa)?;
        }

        let mods = self.final_adaln.forward(&temb.silu())?;
        let shift = mods.narrow(1, 0, self.cfg.hidden_size)?;
        let scale = mods.narrow(1, self.cfg.hidden_size, self.cfg.hidden_size)?;
        let n = if self.final_norm {
            nn::layer_norm(&hs, 1e-6, None, None)?
        } else {
            hs.clone()
        };
        let n = n
            .mul(&expand_token(&scale, seq)?.try_add_scalar(1.0)?)?
            .add(&expand_token(&shift, seq)?)?;
        let out = self.final_proj.forward(&n)?;

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
        let cfg = LongCatTransformerConfig::tiny();
        let dit = LongCatTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let enc = CudaTensor::zeros(&[1, 4, cfg.caption_channels]);
        let out = dit.forward(&x, &enc, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
    }

    #[test]
    fn tiny_forward_with_bsa() {
        let mut cfg = LongCatTransformerConfig::tiny();
        cfg.enable_bsa = true;
        cfg.bsa_sparsity = 0.5;
        cfg.bsa_chunk = [1, 1, 1];
        // pe_t=2, pe_h=2, pe_w=2 with patch [1,2,2] on [2,4,4].
        let dit = LongCatTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let enc = CudaTensor::zeros(&[1, 4, cfg.caption_channels]);
        let out = dit.forward(&x, &enc, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
    }
}
