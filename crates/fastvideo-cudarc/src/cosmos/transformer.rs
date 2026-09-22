//! Cosmos Predict2 DiT (`CosmosTransformer3DModel`).
//!
//! Latents `[B,C,T,H,W]` with optional condition + padding channels. Tiny
//! configs exercise patch → blocks → unpatch without Hub weights.

use fastvideo_models::cosmos::{
    apply_rope_real, rope_cos_sin, CosmosTransformerConfig, ExtraPosEmbed,
};

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn pinned(mut t: CudaTensor) -> Result<CudaTensor> {
    t.pin_device()?;
    Ok(t)
}

fn rms_norm_last(x: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
    x.rms_norm(weight, eps)
}

struct Attn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm: CudaTensor,
    k_norm: CudaTensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

impl Attn {
    fn zeros(dim: usize, heads: usize, head_dim: usize, out_bias: bool) -> Result<Self> {
        Ok(Self {
            to_q: Linear::from_tensors(CudaTensor::zeros(&[dim, dim]), None)?,
            to_k: Linear::from_tensors(CudaTensor::zeros(&[dim, dim]), None)?,
            to_v: Linear::from_tensors(CudaTensor::zeros(&[dim, dim]), None)?,
            to_out: Linear::from_tensors(
                CudaTensor::zeros(&[dim, dim]),
                if out_bias {
                    Some(CudaTensor::zeros(&[dim]))
                } else {
                    None
                },
            )?,
            q_norm: pinned(CudaTensor::ones(&[head_dim]))?,
            k_norm: pinned(CudaTensor::ones(&[head_dim]))?,
            heads,
            head_dim,
            eps: 1e-6,
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize, head_dim: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            to_q: Linear::load(map, &key("to_q"), dim, dim, false)?,
            to_k: Linear::load(map, &key("to_k"), dim, dim, false)?,
            to_v: Linear::load(map, &key("to_v"), dim, dim, false)?,
            to_out: Linear::load(map, &key("to_out.0"), dim, dim, false)?,
            q_norm: pinned(weights::cuda_tensor_shaped(map, &key("norm_q.weight"), &[head_dim])?)?,
            k_norm: pinned(weights::cuda_tensor_shaped(map, &key("norm_k.weight"), &[head_dim])?)?,
            heads,
            head_dim,
            eps: 1e-6,
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
            _ => return Err(msg(format!("cosmos attn: {:?}", hidden.shape))),
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
        q = rms_heads(&q, &self.q_norm, self.eps)?;
        k = rms_heads(&k, &self.k_norm, self.eps)?;
        if let Some((cos, sin)) = rope {
            let qh = q.host_cow()?;
            let kh = k.host_cow()?;
            let qr = apply_rope_real(&qh, cos, sin, b, self.heads, s, self.head_dim);
            let kr = apply_rope_real(&kh, cos, sin, b, self.heads, ks, self.head_dim);
            q = CudaTensor::from_vec(qr, q.shape.clone())?.to_device()?;
            k = CudaTensor::from_vec(kr, k.shape.clone())?.to_device()?;
        }
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let out = attn
            .transpose(1, 2)?
            .reshape(vec![b, s, self.heads * self.head_dim])?;
        self.to_out.forward(&out)
    }
}

fn rms_heads(x: &CudaTensor, gamma: &CudaTensor, eps: f32) -> Result<CudaTensor> {
    // [B,H,S,D] → reshape [B*H*S, D] rms → back
    let [b, h, s, d] = match x.shape[..] {
        [b, h, s, d] => [b, h, s, d],
        _ => return Err(msg("rms_heads: want BHSD")),
    };
    let flat = x.reshape(vec![b * h * s, d])?;
    let n = rms_norm_last(&flat, gamma, eps)?;
    n.reshape(vec![b, h, s, d])
}

/// AdaLN-Zero: `(normed, gate)`.
struct AdaLnZero {
    linear_1: Option<Linear>,
    linear_2: Linear,
    hidden: usize,
}

impl AdaLnZero {
    fn zeros(dim: usize, lora: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Some(Linear::from_tensors(CudaTensor::zeros(&[lora, dim]), None)?),
            linear_2: Linear::from_tensors(CudaTensor::zeros(&[3 * dim, lora]), None)?,
            hidden: dim,
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, lora: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            linear_1: Some(Linear::load(map, &key("linear_1"), dim, lora, false)?),
            linear_2: Linear::load(map, &key("linear_2"), lora, 3 * dim, false)?,
            hidden: dim,
        })
    }

    fn forward(
        &self,
        x: &CudaTensor,
        embedded: &CudaTensor,
        temb: Option<&CudaTensor>,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let mut e = embedded.silu();
        if let Some(l1) = &self.linear_1 {
            e = l1.forward(&e)?;
        }
        let mut mods = self.linear_2.forward(&e)?;
        if let Some(t) = temb {
            mods = mods.add(t)?;
        }
        // mods [B, 3D] or [B,S,3D]
        let chunk = self.hidden;
        let shift = mods.narrow(mods.rank() - 1, 0, chunk)?;
        let scale = mods.narrow(mods.rank() - 1, chunk, chunk)?;
        let gate = mods.narrow(mods.rank() - 1, 2 * chunk, chunk)?;
        let n = nn::layer_norm(x, 1e-6, None, None)?;
        let mut h = n.mul(&scale.try_add_scalar(1.0)?)?.add(&shift)?;
        if shift.rank() == 2 && x.rank() == 3 {
            // broadcast already via try ops if shapes match with unsqueeze — ensure
            let _ = &mut h;
        }
        Ok((h, gate))
    }
}

struct FeedForward {
    net_0: Linear,
    net_2: Linear,
}

impl FeedForward {
    fn zeros(dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            net_0: Linear::from_tensors(CudaTensor::zeros(&[hidden, dim]), None)?,
            net_2: Linear::from_tensors(CudaTensor::zeros(&[dim, hidden]), None)?,
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, hidden: usize) -> Result<Self> {
        // Diffusers FeedForward: net.0.proj, net.2
        let p0 = weights::join_key(prefix, "net.0.proj");
        let p2 = weights::join_key(prefix, "net.2");
        Ok(Self {
            net_0: Linear::load(map, &p0, dim, hidden, false)?,
            net_2: Linear::load(map, &p2, hidden, dim, false)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let h = self.net_0.forward(x)?.gelu_erf();
        self.net_2.forward(&h)
    }
}

struct Block {
    norm1: AdaLnZero,
    attn1: Attn,
    norm2: AdaLnZero,
    attn2: Attn,
    norm3: AdaLnZero,
    ff: FeedForward,
}

impl Block {
    fn zeros(cfg: &CosmosTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let heads = cfg.num_attention_heads;
        let hd = cfg.attention_head_dim;
        Ok(Self {
            norm1: AdaLnZero::zeros(dim, cfg.adaln_lora_dim)?,
            attn1: Attn::zeros(dim, heads, hd, false)?,
            norm2: AdaLnZero::zeros(dim, cfg.adaln_lora_dim)?,
            attn2: Attn::zeros(dim, heads, hd, false)?,
            norm3: AdaLnZero::zeros(dim, cfg.adaln_lora_dim)?,
            ff: FeedForward::zeros(dim, cfg.mlp_hidden())?,
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &CosmosTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let heads = cfg.num_attention_heads;
        let hd = cfg.attention_head_dim;
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            norm1: AdaLnZero::load(map, &key("norm1"), dim, cfg.adaln_lora_dim)?,
            attn1: Attn::load(map, &key("attn1"), dim, heads, hd)?,
            norm2: AdaLnZero::load(map, &key("norm2"), dim, cfg.adaln_lora_dim)?,
            attn2: Attn::load(map, &key("attn2"), dim, heads, hd)?,
            norm3: AdaLnZero::load(map, &key("norm3"), dim, cfg.adaln_lora_dim)?,
            ff: FeedForward::load(map, &key("ff"), dim, cfg.mlp_hidden())?,
        })
    }

    fn forward(
        &self,
        x: &CudaTensor,
        encoder: &CudaTensor,
        embedded: &CudaTensor,
        temb: &CudaTensor,
        rope: (&Vec<f32>, &Vec<f32>),
        extra_pos: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let mut h = if let Some(pe) = extra_pos {
            x.add(pe)?
        } else {
            x.clone()
        };
        let (n, gate) = self.norm1.forward(&h, embedded, Some(temb))?;
        let a = self.attn1.forward(&n, None, Some(rope))?;
        h = h.add(&a.mul(&gate)?)?;

        let (n, gate) = self.norm2.forward(&h, embedded, Some(temb))?;
        let a = self.attn2.forward(&n, Some(encoder), None)?;
        h = h.add(&a.mul(&gate)?)?;

        let (n, gate) = self.norm3.forward(&h, embedded, Some(temb))?;
        let f = self.ff.forward(&n)?;
        h.add(&f.mul(&gate)?)
    }
}

struct TimeEmbed {
    time_proj_dim: usize,
    t_linear_1: Linear,
    t_linear_2: Linear,
    norm_w: CudaTensor,
}

impl TimeEmbed {
    fn zeros(hidden: usize) -> Result<Self> {
        Ok(Self {
            time_proj_dim: hidden,
            t_linear_1: Linear::from_tensors(CudaTensor::zeros(&[hidden, hidden]), None)?,
            t_linear_2: Linear::from_tensors(CudaTensor::zeros(&[3 * hidden, hidden]), None)?,
            norm_w: pinned(CudaTensor::ones(&[hidden]))?,
        })
    }

    fn load(map: &WeightMap, hidden: usize) -> Result<Self> {
        Ok(Self {
            time_proj_dim: hidden,
            t_linear_1: Linear::load(map, "time_embed.t_embedder.linear_1", hidden, hidden, false)?,
            t_linear_2: Linear::load(
                map,
                "time_embed.t_embedder.linear_2",
                hidden,
                3 * hidden,
                false,
            )?,
            norm_w: pinned(weights::cuda_tensor_shaped(
                map,
                "time_embed.norm.weight",
                &[hidden],
            )?)?,
        })
    }

    fn forward(&self, timestep_scalar: f32, batch: usize) -> Result<(CudaTensor, CudaTensor)> {
        // Timesteps embedding (flip_sin_to_cos) → [B, hidden]
        let half = self.time_proj_dim / 2;
        let mut emb = vec![0f32; batch * self.time_proj_dim];
        for b in 0..batch {
            for i in 0..half {
                let freq = (-(10_000f32.ln()) * (i as f32) / half as f32).exp();
                let arg = timestep_scalar * freq;
                emb[b * self.time_proj_dim + i] = arg.cos();
                emb[b * self.time_proj_dim + half + i] = arg.sin();
            }
        }
        let proj = CudaTensor::from_vec(emb, vec![batch, self.time_proj_dim])?;
        let temb = self
            .t_linear_2
            .forward(&self.t_linear_1.forward(&proj)?.silu())?;
        let embedded = rms_norm_last(&proj, &self.norm_w, 1e-6)?;
        Ok((temb, embedded))
    }
}

pub struct CosmosTransformer {
    pub cfg: CosmosTransformerConfig,
    patch: Linear,
    blocks: Vec<Block>,
    time: TimeEmbed,
    norm_out_l1: Linear,
    norm_out_l2: Linear,
    proj_out: Linear,
    pos_t: Option<CudaTensor>,
    pos_h: Option<CudaTensor>,
    pos_w: Option<CudaTensor>,
}

impl CosmosTransformer {
    pub fn zeros(cfg: CosmosTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let [p_t, p_h, p_w] = cfg.patch_size;
        let patch_in = cfg.patch_in_channels() * p_t * p_h * p_w;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            blocks.push(Block::zeros(&cfg)?);
        }
        let (pos_t, pos_h, pos_w) = if cfg.extra_pos_embed_type == ExtraPosEmbed::Learnable {
            let mt = cfg.max_size[0] / p_t;
            let mh = cfg.max_size[1] / p_h;
            let mw = cfg.max_size[2] / p_w;
            (
                Some(CudaTensor::zeros(&[mt, dim])),
                Some(CudaTensor::zeros(&[mh, dim])),
                Some(CudaTensor::zeros(&[mw, dim])),
            )
        } else {
            (None, None, None)
        };
        Ok(Self {
            patch: Linear::from_tensors(CudaTensor::zeros(&[dim, patch_in]), None)?,
            blocks,
            time: TimeEmbed::zeros(dim)?,
            norm_out_l1: Linear::from_tensors(
                CudaTensor::zeros(&[cfg.adaln_lora_dim, dim]),
                None,
            )?,
            norm_out_l2: Linear::from_tensors(
                CudaTensor::zeros(&[2 * dim, cfg.adaln_lora_dim]),
                None,
            )?,
            proj_out: Linear::from_tensors(
                CudaTensor::zeros(&[p_t * p_h * p_w * cfg.out_channels, dim]),
                None,
            )?,
            pos_t,
            pos_h,
            pos_w,
            cfg,
        })
    }

    pub fn load(cfg: CosmosTransformerConfig, map: &WeightMap) -> Result<Self> {
        let dim = cfg.hidden_size();
        let [p_t, p_h, p_w] = cfg.patch_size;
        let patch_in = cfg.patch_in_channels() * p_t * p_h * p_w;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::load(map, &format!("transformer_blocks.{i}"), &cfg)?);
        }
        let (pos_t, pos_h, pos_w) = if cfg.extra_pos_embed_type == ExtraPosEmbed::Learnable {
            let mt = cfg.max_size[0] / p_t;
            let mh = cfg.max_size[1] / p_h;
            let mw = cfg.max_size[2] / p_w;
            (
                Some(weights::cuda_tensor_shaped(
                    map,
                    "learnable_pos_embed.pos_emb_t",
                    &[mt, dim],
                )?),
                Some(weights::cuda_tensor_shaped(
                    map,
                    "learnable_pos_embed.pos_emb_h",
                    &[mh, dim],
                )?),
                Some(weights::cuda_tensor_shaped(
                    map,
                    "learnable_pos_embed.pos_emb_w",
                    &[mw, dim],
                )?),
            )
        } else {
            (None, None, None)
        };
        Ok(Self {
            patch: Linear::load(map, "patch_embed.proj", patch_in, dim, false)?,
            blocks,
            time: TimeEmbed::load(map, dim)?,
            norm_out_l1: Linear::load(
                map,
                "norm_out.linear_1",
                dim,
                cfg.adaln_lora_dim,
                false,
            )?,
            norm_out_l2: Linear::load(
                map,
                "norm_out.linear_2",
                cfg.adaln_lora_dim,
                2 * dim,
                false,
            )?,
            proj_out: Linear::load(
                map,
                "proj_out",
                dim,
                p_t * p_h * p_w * cfg.out_channels,
                false,
            )?,
            pos_t,
            pos_h,
            pos_w,
            cfg,
        })
    }

    /// `hidden` `[B,C,T,H,W]` already including condition (+ optional pad later).
    /// `encoder` `[B,S,text_dim]`. `timestep` scalar in `[0,1]` Cosos `current_t`.
    pub fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        timestep: f32,
        fps: Option<f32>,
    ) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match hidden.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg(format!("cosmos dit: {:?}", hidden.shape))),
        };
        let want_c = if self.cfg.concat_padding_mask {
            // Caller may already have concatenated padding; accept patch_in channel count
            // via pre-patchify helper — here expect full patch_embed input channels.
            self.cfg.patch_in_channels()
        } else {
            self.cfg.in_channels
        };
        if c != want_c && c != self.cfg.in_channels {
            return Err(msg(format!(
                "cosmos dit channels {c} vs in {} / patch {}",
                self.cfg.in_channels,
                self.cfg.patch_in_channels()
            )));
        }
        let mut x = hidden.clone();
        if c == self.cfg.in_channels && self.cfg.concat_padding_mask {
            // Zero padding mask channel.
            let zeros = CudaTensor::zeros(&[b, 1, t, h, w]);
            x = CudaTensor::cat(&[&x, &zeros], 1)?;
        }

        let [p_t, p_h, p_w] = self.cfg.patch_size;
        let pe_t = t / p_t;
        let pe_h = h / p_h;
        let pe_w = w / p_w;
        let seq = pe_t * pe_h * pe_w;
        let (cos, sin) = rope_cos_sin(&self.cfg, t, h, w, fps);

        // Patchify on host for clarity (tiny graphs).
        let host = x.host_cow()?;
        let cin = x.shape[1];
        let mut tokens = vec![0f32; b * seq * cin * p_t * p_h * p_w];
        for bi in 0..b {
            for ti in 0..pe_t {
                for yi in 0..pe_h {
                    for xi in 0..pe_w {
                        let tok = (ti * pe_h + yi) * pe_w + xi;
                        let mut o = 0usize;
                        for pt in 0..p_t {
                            for ph in 0..p_h {
                                for pw in 0..p_w {
                                    for ci in 0..cin {
                                        let src = ((((bi * cin + ci) * t + ti * p_t + pt) * h
                                            + yi * p_h
                                            + ph)
                                            * w
                                            + xi * p_w
                                            + pw);
                                        tokens[((bi * seq + tok) * cin * p_t * p_h * p_w) + o] =
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
        let flat = CudaTensor::from_vec(tokens, vec![b * seq, cin * p_t * p_h * p_w])?;
        let mut hs = self.patch.forward(&flat)?.reshape(vec![b, seq, self.cfg.hidden_size()])?;

        let extra = self.learnable_pos(b, pe_t, pe_h, pe_w)?;
        let (temb, embedded) = self.time.forward(timestep, b)?;
        // Expand [B,C] → [B,S,C] for per-token AdaLN.
        let temb_s = expand_tokens(&temb, seq)?;
        let emb_s = expand_tokens(&embedded, seq)?;

        for block in &self.blocks {
            hs = block.forward(
                &hs,
                encoder,
                &emb_s,
                &temb_s,
                (&cos, &sin),
                extra.as_ref(),
            )?;
        }

        // Final AdaLN (shift/scale only)
        let mut e = emb_s.silu();
        e = self.norm_out_l1.forward(&e)?;
        let mods = self.norm_out_l2.forward(&e)?;
        let shift = mods.narrow(2, 0, self.cfg.hidden_size())?;
        let scale = mods.narrow(2, self.cfg.hidden_size(), self.cfg.hidden_size())?;
        let n = nn::layer_norm(&hs, 1e-6, None, None)?;
        hs = n.mul(&scale.try_add_scalar(1.0)?)?.add(&shift)?;
        let out = self.proj_out.forward(&hs)?; // [B,S, p*out]

        // Unpatchify to [B, out_c, T, H, W]
        let oc = self.cfg.out_channels;
        let oh = out.host_cow()?;
        let mut pixels = vec![0f32; b * oc * t * h * w];
        for bi in 0..b {
            for ti in 0..pe_t {
                for yi in 0..pe_h {
                    for xi in 0..pe_w {
                        let tok = (ti * pe_h + yi) * pe_w + xi;
                        let mut o = 0usize;
                        for pt in 0..p_t {
                            for ph in 0..p_h {
                                for pw in 0..p_w {
                                    for ci in 0..oc {
                                        let dst = ((((bi * oc + ci) * t + ti * p_t + pt) * h
                                            + yi * p_h
                                            + ph)
                                            * w
                                            + xi * p_w
                                            + pw);
                                        pixels[dst] = oh[((bi * seq + tok)
                                            * (p_t * p_h * p_w * oc))
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

    fn learnable_pos(
        &self,
        batch: usize,
        pe_t: usize,
        pe_h: usize,
        pe_w: usize,
    ) -> Result<Option<CudaTensor>> {
        let (Some(pt), Some(ph), Some(pw)) = (&self.pos_t, &self.pos_h, &self.pos_w) else {
            return Ok(None);
        };
        let dim = self.cfg.hidden_size();
        let seq = pe_t * pe_h * pe_w;
        let th = pt.host_cow()?;
        let hh = ph.host_cow()?;
        let wh = pw.host_cow()?;
        let mut out = vec![0f32; batch * seq * dim];
        for b in 0..batch {
            for t in 0..pe_t {
                for y in 0..pe_h {
                    for x in 0..pe_w {
                        let cell = (t * pe_h + y) * pe_w + x;
                        for d in 0..dim {
                            out[(b * seq + cell) * dim + d] =
                                th[t * dim + d] + hh[y * dim + d] + wh[x * dim + d];
                        }
                    }
                }
            }
        }
        // Cosos learnable pos: L2 normalize with eps scaling.
        let eps = 1e-6f32;
        let numel = out.len() as f32;
        for b in 0..batch {
            for s in 0..seq {
                let base = (b * seq + s) * dim;
                let mut sq = 0f32;
                for d in 0..dim {
                    sq += out[base + d] * out[base + d];
                }
                let norm = (sq.max(0.0)).sqrt() + eps * (numel / (batch * seq) as f32).sqrt();
                for d in 0..dim {
                    out[base + d] /= norm;
                }
            }
        }
        Ok(Some(CudaTensor::from_vec(out, vec![batch, seq, dim])?))
    }
}

fn expand_tokens(x: &CudaTensor, seq: usize) -> Result<CudaTensor> {
    // [B, C] → [B, S, C]
    let [b, c] = match x.shape[..] {
        [b, c] => [b, c],
        _ => return Err(msg(format!("expand_tokens: {:?}", x.shape))),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shapes() {
        let cfg = CosmosTransformerConfig::tiny();
        let dit = CosmosTransformer::zeros(cfg.clone()).unwrap();
        // T=2,H=4,W=4, C=in (5) — padding added inside
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_embed_dim]);
        let out = dit.forward(&x, &enc, 0.5, Some(16.0)).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
    }
}
