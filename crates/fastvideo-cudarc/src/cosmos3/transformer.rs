//! Cosmos3-Super T2V DiT scaffold.
//!
//! Tiny configs exercise patch → AdaLN blocks → unpatch. Super 64B widths
//! are `TODO(upstream)` — do not load Hub weights here.
//!
//! TeaCache is [`SolCosmosTea`] from Predict2 (`cosmos::sol`). This file
//! does not invent a second cache.

use std::sync::{Arc, Mutex};

use fastvideo_models::cosmos::sol::SolCosmosTea;
use fastvideo_models::cosmos3::Cosmos3TransformerConfig;

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};

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
    fn zeros(dim: usize, heads: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            to_q: Linear::zeros(dim, dim, false),
            to_k: Linear::zeros(dim, dim, false),
            to_v: Linear::zeros(dim, dim, false),
            to_out: Linear::zeros(dim, dim, false),
            heads,
            head_dim,
        })
    }

    fn forward(&self, hidden: &CudaTensor, encoder: Option<&CudaTensor>) -> Result<CudaTensor> {
        let [b, s, _] = match hidden.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("cosmos3 attn: {:?}", hidden.shape))),
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
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let out = attn
            .transpose(1, 2)?
            .reshape(vec![b, s, self.heads * self.head_dim])?;
        self.to_out.forward(&out)
    }
}

struct Block {
    ada: Linear,
    self_attn: Attn,
    cross_attn: Attn,
    ff1: Linear,
    ff2: Linear,
}

impl Block {
    fn zeros(cfg: &Cosmos3TransformerConfig) -> Result<Self> {
        let h = cfg.hidden_size();
        let mlp = cfg.mlp_hidden();
        Ok(Self {
            ada: Linear::zeros(h, 6 * h, false),
            self_attn: Attn::zeros(h, cfg.num_attention_heads, cfg.attention_head_dim)?,
            cross_attn: Attn::zeros(h, cfg.num_attention_heads, cfg.attention_head_dim)?,
            ff1: Linear::zeros(h, mlp, false),
            ff2: Linear::zeros(mlp, h, false),
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        temb: &CudaTensor,
    ) -> Result<CudaTensor> {
        let mods = self.ada.forward(&temb.silu())?;
        let h = hidden.shape[2];
        let shift1 = mods.narrow(mods.rank() - 1, 0, h)?;
        let scale1 = mods.narrow(mods.rank() - 1, h, h)?;
        let gate1 = mods.narrow(mods.rank() - 1, 2 * h, h)?;
        let shift2 = mods.narrow(mods.rank() - 1, 3 * h, h)?;
        let scale2 = mods.narrow(mods.rank() - 1, 4 * h, h)?;
        let gate2 = mods.narrow(mods.rank() - 1, 5 * h, h)?;

        let n1 = nn::layer_norm(hidden, 1e-6, None, None)?;
        let n1 = n1.mul(&scale1.try_add_scalar(1.0)?)?.add(&shift1)?;
        let sa = self.self_attn.forward(&n1, None)?;
        let mut hs = hidden.add(&sa.mul(&gate1)?)?;

        let ca = self.cross_attn.forward(&hs, Some(encoder))?;
        hs = hs.add(&ca)?;

        let n2 = nn::layer_norm(&hs, 1e-6, None, None)?;
        let n2 = n2.mul(&scale2.try_add_scalar(1.0)?)?.add(&shift2)?;
        let ff = self.ff2.forward(&self.ff1.forward(&n2)?.gelu_erf())?;
        hs.add(&ff.mul(&gate2)?)
    }
}

struct CosmosTeaRuntime {
    state: SolCosmosTea,
    pending_step: Option<usize>,
    residual: Option<CudaTensor>,
}

pub struct Cosmos3Transformer {
    cfg: Cosmos3TransformerConfig,
    patch: Linear,
    time1: Linear,
    time2: Linear,
    txt_proj: Linear,
    blocks: Vec<Block>,
    norm_out: Linear,
    proj_out: Linear,
    sol_tea: Arc<Mutex<Option<CosmosTeaRuntime>>>,
}

impl Cosmos3Transformer {
    pub fn zeros(cfg: Cosmos3TransformerConfig) -> Result<Self> {
        let h = cfg.hidden_size();
        let patch_in = cfg.in_channels * cfg.patch_volume();
        let blocks = (0..cfg.num_layers)
            .map(|_| Block::zeros(&cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch: Linear::zeros(patch_in, h, false),
            time1: Linear::zeros(256, h, false),
            time2: Linear::zeros(h, h, false),
            txt_proj: Linear::zeros(cfg.text_embed_dim, h, false),
            blocks,
            norm_out: Linear::zeros(h, 2 * h, false),
            proj_out: Linear::zeros(h, cfg.out_channels * cfg.patch_volume(), false),
            sol_tea: Default::default(),
            cfg,
        })
    }

    pub fn config(&self) -> &Cosmos3TransformerConfig {
        &self.cfg
    }

    /// Install Predict2 TeaCache (1.15 / start 10 / max 3). Default path is dense.
    pub fn enable_sol_teacache(&self) {
        let mut slot = self.sol_tea.lock().expect("cosmos3 sol tea");
        *slot = Some(CosmosTeaRuntime {
            state: SolCosmosTea::official(),
            pending_step: None,
            residual: None,
        });
    }

    pub fn sol_teacache_enabled(&self) -> bool {
        self.sol_tea.lock().expect("cosmos3 sol tea").is_some()
    }

    pub fn arm_sol_teacache(&self, step: usize) {
        let mut slot = self.sol_tea.lock().expect("cosmos3 sol tea");
        if let Some(runtime) = slot.as_mut() {
            runtime.pending_step = Some(step);
        }
    }

    fn begin_sol_tea(&self, signal: &CudaTensor) -> Result<Option<bool>> {
        let mut slot = self.sol_tea.lock().expect("cosmos3 sol tea");
        let Some(runtime) = slot.as_mut() else {
            return Ok(None);
        };
        let Some(step) = runtime.pending_step.take() else {
            return Ok(None);
        };
        let host = signal.host_cow()?;
        let compute = runtime.state.decide(step, &host);
        if compute {
            Ok(Some(false))
        } else {
            Ok(Some(true))
        }
    }

    fn add_sol_tea_residual(&self, hidden: CudaTensor) -> Result<CudaTensor> {
        let residual = {
            let slot = self.sol_tea.lock().expect("cosmos3 sol tea");
            slot.as_ref()
                .and_then(|r| r.residual.clone())
                .expect("cosmos3 sol tea residual")
        };
        hidden.add(&residual)
    }

    fn finish_sol_tea(&self, before: &CudaTensor, after: &CudaTensor) -> Result<()> {
        let mut slot = self.sol_tea.lock().expect("cosmos3 sol tea");
        let runtime = slot.as_mut().expect("cosmos3 sol tea");
        runtime.residual = Some(after.sub(before)?);
        Ok(())
    }

    /// `latents` `[B,C,T,H,W]` → `[B, out_c, T, H, W]`.
    pub fn forward(
        &self,
        latents: &CudaTensor,
        encoder: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match latents.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => {
                return Err(msg(format!(
                    "cosmos3 dit: {:?} want [B,C,T,H,W]",
                    latents.shape
                )))
            }
        };
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "cosmos3 dit channels {c} vs {}",
                self.cfg.in_channels
            )));
        }
        let [p_t, p_h, p_w] = self.cfg.patch_size;
        if t % p_t != 0 || h % p_h != 0 || w % p_w != 0 {
            return Err(msg(format!(
                "cosmos3 dit: [{t},{h},{w}] not divisible by patch {p_t}x{p_h}x{p_w}"
            )));
        }
        let pe_t = t / p_t;
        let pe_h = h / p_h;
        let pe_w = w / p_w;
        let seq = pe_t * pe_h * pe_w;
        let patch_in = c * p_t * p_h * p_w;

        let host = latents.host_cow()?;
        let mut tokens = vec![0f32; b * seq * patch_in];
        for bi in 0..b {
            for ti in 0..pe_t {
                for yi in 0..pe_h {
                    for xi in 0..pe_w {
                        let tok = (ti * pe_h + yi) * pe_w + xi;
                        let mut o = 0usize;
                        for pt in 0..p_t {
                            for ph in 0..p_h {
                                for pw in 0..p_w {
                                    for ci in 0..c {
                                        let src = (((bi * c + ci) * t + ti * p_t + pt) * h
                                            + yi * p_h
                                            + ph)
                                            * w
                                            + xi * p_w
                                            + pw;
                                        tokens[(bi * seq + tok) * patch_in + o] = host[src];
                                        o += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let flat = CudaTensor::from_vec(tokens, vec![b * seq, patch_in])?;
        let mut hs = self
            .patch
            .forward(&flat)?
            .reshape(vec![b, seq, self.cfg.hidden_size()])?;

        let temb_in = sinusoid_timestep(timestep, 256);
        let temb = CudaTensor::from_vec(temb_in, vec![1, 256])?;
        let temb = self.time2.forward(&self.time1.forward(&temb)?.silu())?;
        let temb = temb.reshape(vec![1, 1, self.cfg.hidden_size()])?;
        let encoder = self.txt_proj.forward(encoder)?;

        let tea = self.begin_sol_tea(&temb)?;
        if tea == Some(true) {
            hs = self.add_sol_tea_residual(hs)?;
        } else {
            let before = if tea == Some(false) {
                Some(hs.clone())
            } else {
                None
            };
            for block in &self.blocks {
                hs = block.forward(&hs, &encoder, &temb)?;
            }
            if let Some(before) = before.as_ref() {
                self.finish_sol_tea(before, &hs)?;
            }
        }

        let mods = self.norm_out.forward(&temb.silu())?;
        let hidden = self.cfg.hidden_size();
        let shift = mods.narrow(mods.rank() - 1, 0, hidden)?;
        let scale = mods.narrow(mods.rank() - 1, hidden, hidden)?;
        let n = nn::layer_norm(&hs, 1e-6, None, None)?;
        hs = n.mul(&scale.try_add_scalar(1.0)?)?.add(&shift)?;
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
                        for pt in 0..p_t {
                            for ph in 0..p_h {
                                for pw in 0..p_w {
                                    for ci in 0..oc {
                                        let dst = (((bi * oc + ci) * t + ti * p_t + pt) * h
                                            + yi * p_h
                                            + ph)
                                            * w
                                            + xi * p_w
                                            + pw;
                                        pixels[dst] =
                                            oh[(bi * seq + tok) * (p_t * p_h * p_w * oc) + o];
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

fn sinusoid_timestep(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let freq = (10000f64).powf(-(i as f64) / half as f64) as f32;
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shapes() {
        let cfg = Cosmos3TransformerConfig::tiny();
        let dit = Cosmos3Transformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_embed_dim]);
        let out = dit.forward(&x, &enc, 0.5).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
    }

    #[test]
    fn sol_teacache_reuses_predict2_window() {
        let cfg = Cosmos3TransformerConfig::tiny();
        let dit = Cosmos3Transformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 2, 4, 4]);
        let enc = CudaTensor::zeros(&[1, 4, cfg.text_embed_dim]);
        dit.enable_sol_teacache();
        assert!(dit.sol_teacache_enabled());
        for step in 0..14 {
            dit.arm_sol_teacache(step);
            let out = dit.forward(&x, &enc, 0.4).unwrap();
            assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
        }
    }
}
