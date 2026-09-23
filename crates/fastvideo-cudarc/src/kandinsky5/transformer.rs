//! Kandinsky 5 DiT: text encoder blocks + visual decoder blocks.
//!
//! Diffusers `Kandinsky5Transformer3DModel`. Latents arrive as `[B,T,H,W,C]`
//! (channel-last). Tiny configs exercise the regular-attention path.

use fastvideo_models::kandinsky5::{apply_rope, rope_1d, rope_3d, Kandinsky5TransformerConfig};

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

fn scale_shift(x: &CudaTensor, scale: &CudaTensor, shift: &CudaTensor) -> Result<CudaTensor> {
    x.mul(&scale.try_add_scalar(1.0)?)?.add(shift)
}

fn gated(x: &CudaTensor, y: &CudaTensor, gate: &CudaTensor) -> Result<CudaTensor> {
    x.add(&y.mul(gate)?)
}

/// Diffusers `Kandinsky5Attention` (self or cross).
struct Attn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    out: Linear,
    q_norm: CudaTensor,
    k_norm: CudaTensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

impl Attn {
    fn zeros(dim: usize, head_dim: usize) -> Result<Self> {
        let heads = dim / head_dim;
        Ok(Self {
            to_q: Linear::zeros(dim, dim, true),
            to_k: Linear::zeros(dim, dim, true),
            to_v: Linear::zeros(dim, dim, true),
            out: Linear::zeros(dim, dim, true),
            q_norm: pinned(CudaTensor::ones(&[head_dim]))?,
            k_norm: pinned(CudaTensor::ones(&[head_dim]))?,
            heads,
            head_dim,
            eps: 1e-6,
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, head_dim: usize) -> Result<Self> {
        let heads = dim / head_dim;
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            to_q: Linear::load(map, &key("to_query"), dim, dim, true)?,
            to_k: Linear::load(map, &key("to_key"), dim, dim, true)?,
            to_v: Linear::load(map, &key("to_value"), dim, dim, true)?,
            out: Linear::load(map, &key("out_layer"), dim, dim, true)?,
            q_norm: pinned(weights::cuda_tensor_shaped(
                map,
                &key("query_norm.weight"),
                &[head_dim],
            )?)?,
            k_norm: pinned(weights::cuda_tensor_shaped(
                map,
                &key("key_norm.weight"),
                &[head_dim],
            )?)?,
            heads,
            head_dim,
            eps: 1e-6,
        })
    }

    /// `hidden` `[B,S,D]`; optional `encoder` for cross-attn; rope `[S, D/2, 2, 2]` flat.
    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: Option<&CudaTensor>,
        rope: Option<&[f32]>,
    ) -> Result<CudaTensor> {
        let [b, s, _] = match hidden.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("k5 attn: {:?}", hidden.shape))),
        };
        let q = self.to_q.forward(hidden)?;
        let (k, v, ks) = match encoder {
            Some(enc) => {
                let ks = enc.shape[1];
                (self.to_k.forward(enc)?, self.to_v.forward(enc)?, ks)
            }
            None => (self.to_k.forward(hidden)?, self.to_v.forward(hidden)?, s),
        };
        let q = reshape_heads(&q, b, s, self.heads, self.head_dim)?;
        let k = reshape_heads(&k, b, ks, self.heads, self.head_dim)?;
        let v = reshape_heads(&v, b, ks, self.heads, self.head_dim)?;
        let q = rms_heads(&q, &self.q_norm, self.eps)?;
        let k = rms_heads(&k, &self.k_norm, self.eps)?;
        let (q, k) = if let Some(r) = rope {
            (
                apply_rope_tensor(&q, r, s, self.heads, self.head_dim)?,
                apply_rope_tensor(&k, r, ks, self.heads, self.head_dim)?,
            )
        } else {
            (q, k)
        };
        // BHSD for sdpa
        let q = to_bhsd(&q, b, s, self.heads, self.head_dim)?;
        let k = to_bhsd(&k, b, ks, self.heads, self.head_dim)?;
        let v = to_bhsd(&v, b, ks, self.heads, self.head_dim)?;
        let attn = nn::scaled_dot_product_attention_masked(&q, &k, &v, None, None)?;
        let flat = attn.merge_heads()?;
        self.out.forward(&flat)
    }
}

fn reshape_heads(x: &CudaTensor, b: usize, s: usize, heads: usize, d: usize) -> Result<CudaTensor> {
    // [B,S,H*D] → keep as [B,S,H,D] via reshape for host ops
    x.reshape(vec![b, s, heads, d])
}

fn to_bhsd(x: &CudaTensor, b: usize, s: usize, heads: usize, d: usize) -> Result<CudaTensor> {
    // [B,S,H,D] → [B,H,S,D]
    x.permute(&[0, 2, 1, 3])?.reshape(vec![b, heads, s, d])
}

fn rms_heads(x: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
    // RMS over last dim of [B,S,H,D]
    let [b, s, h, d] = match x.shape[..] {
        [b, s, h, d] => [b, s, h, d],
        _ => return Err(msg("rms_heads shape")),
    };
    let flat = x.reshape(vec![b * s * h, d])?;
    let n = flat.rms_norm(weight, eps)?;
    n.reshape(vec![b, s, h, d])
}

fn apply_rope_tensor(
    x: &CudaTensor,
    rope: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<CudaTensor> {
    let host = x.host_cow()?;
    let out = apply_rope(&host, rope, seq, heads, head_dim);
    Ok(CudaTensor::from_vec(out, x.shape.clone())?.to_device()?)
}

struct FeedForward {
    inn: Linear,
    out: Linear,
}

impl FeedForward {
    fn zeros(dim: usize, ff: usize) -> Self {
        Self {
            inn: Linear::zeros(dim, ff, false),
            out: Linear::zeros(ff, dim, false),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, ff: usize) -> Result<Self> {
        Ok(Self {
            inn: Linear::load(map, &weights::join_key(prefix, "in_layer"), dim, ff, false)?,
            out: Linear::load(map, &weights::join_key(prefix, "out_layer"), ff, dim, false)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.out.forward(&self.inn.forward(x)?.gelu_erf())
    }
}

struct Modulation {
    out: Linear,
    #[allow(dead_code)]
    n: usize,
}

impl Modulation {
    fn zeros(time_dim: usize, model_dim: usize, n: usize) -> Self {
        Self {
            out: Linear::zeros(time_dim, n * model_dim, true),
            n,
        }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        time_dim: usize,
        model_dim: usize,
        n: usize,
    ) -> Result<Self> {
        Ok(Self {
            out: Linear::load(
                map,
                &weights::join_key(prefix, "out_layer"),
                time_dim,
                n * model_dim,
                true,
            )?,
            n,
        })
    }

    fn forward(&self, t: &CudaTensor) -> Result<CudaTensor> {
        self.out.forward(&t.silu())
    }
}

/// Text self-attn block (`Kandinsky5TransformerEncoderBlock`).
struct TextBlock {
    mod_: Modulation,
    attn: Attn,
    ff: FeedForward,
}

impl TextBlock {
    fn zeros(cfg: &Kandinsky5TransformerConfig) -> Result<Self> {
        Ok(Self {
            mod_: Modulation::zeros(cfg.time_dim, cfg.model_dim, 6),
            attn: Attn::zeros(cfg.model_dim, cfg.head_dim())?,
            ff: FeedForward::zeros(cfg.model_dim, cfg.ff_dim),
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Kandinsky5TransformerConfig) -> Result<Self> {
        Ok(Self {
            mod_: Modulation::load(
                map,
                &format!("{prefix}.text_modulation"),
                cfg.time_dim,
                cfg.model_dim,
                6,
            )?,
            attn: Attn::load(
                map,
                &format!("{prefix}.self_attention"),
                cfg.model_dim,
                cfg.head_dim(),
            )?,
            ff: FeedForward::load(
                map,
                &format!("{prefix}.feed_forward"),
                cfg.model_dim,
                cfg.ff_dim,
            )?,
        })
    }

    fn forward(&self, x: &CudaTensor, time: &CudaTensor, rope: &[f32]) -> Result<CudaTensor> {
        let mods = self.mod_.forward(time)?.unsqueeze(1)?;
        let parts = mods.chunk(2, mods.rank() - 1)?;
        let attn_p = parts[0].chunk(3, parts[0].rank() - 1)?;
        let ff_p = parts[1].chunk(3, parts[1].rank() - 1)?;
        let n = x.layer_norm(1e-6, None, None)?;
        let out = self
            .attn
            .forward(&scale_shift(&n, &attn_p[1], &attn_p[0])?, None, Some(rope))?;
        let x = gated(x, &out, &attn_p[2])?;
        let n = x.layer_norm(1e-6, None, None)?;
        let out = self.ff.forward(&scale_shift(&n, &ff_p[1], &ff_p[0])?)?;
        gated(&x, &out, &ff_p[2])
    }
}

/// Visual self + cross + FF (`Kandinsky5TransformerDecoderBlock`).
struct VisualBlock {
    mod_: Modulation,
    self_attn: Attn,
    cross_attn: Attn,
    ff: FeedForward,
}

impl VisualBlock {
    fn zeros(cfg: &Kandinsky5TransformerConfig) -> Result<Self> {
        Ok(Self {
            mod_: Modulation::zeros(cfg.time_dim, cfg.model_dim, 9),
            self_attn: Attn::zeros(cfg.model_dim, cfg.head_dim())?,
            cross_attn: Attn::zeros(cfg.model_dim, cfg.head_dim())?,
            ff: FeedForward::zeros(cfg.model_dim, cfg.ff_dim),
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Kandinsky5TransformerConfig) -> Result<Self> {
        Ok(Self {
            mod_: Modulation::load(
                map,
                &format!("{prefix}.visual_modulation"),
                cfg.time_dim,
                cfg.model_dim,
                9,
            )?,
            self_attn: Attn::load(
                map,
                &format!("{prefix}.self_attention"),
                cfg.model_dim,
                cfg.head_dim(),
            )?,
            cross_attn: Attn::load(
                map,
                &format!("{prefix}.cross_attention"),
                cfg.model_dim,
                cfg.head_dim(),
            )?,
            ff: FeedForward::load(
                map,
                &format!("{prefix}.feed_forward"),
                cfg.model_dim,
                cfg.ff_dim,
            )?,
        })
    }

    fn forward(
        &self,
        visual: &CudaTensor,
        text: &CudaTensor,
        time: &CudaTensor,
        rope: &[f32],
    ) -> Result<CudaTensor> {
        let mods = self.mod_.forward(time)?.unsqueeze(1)?;
        let parts = mods.chunk(3, mods.rank() - 1)?;
        let self_p = parts[0].chunk(3, parts[0].rank() - 1)?;
        let cross_p = parts[1].chunk(3, parts[1].rank() - 1)?;
        let ff_p = parts[2].chunk(3, parts[2].rank() - 1)?;

        let n = visual.layer_norm(1e-6, None, None)?;
        let out =
            self.self_attn
                .forward(&scale_shift(&n, &self_p[1], &self_p[0])?, None, Some(rope))?;
        let visual = gated(visual, &out, &self_p[2])?;

        let n = visual.layer_norm(1e-6, None, None)?;
        let out = self.cross_attn.forward(
            &scale_shift(&n, &cross_p[1], &cross_p[0])?,
            Some(text),
            None,
        )?;
        let visual = gated(&visual, &out, &cross_p[2])?;

        let n = visual.layer_norm(1e-6, None, None)?;
        let out = self.ff.forward(&scale_shift(&n, &ff_p[1], &ff_p[0])?)?;
        gated(&visual, &out, &ff_p[2])
    }
}

/// Full Kandinsky 5 transformer (or tiny).
pub struct Kandinsky5Transformer {
    cfg: Kandinsky5TransformerConfig,
    time_in: Linear,
    time_out: Linear,
    text_in: Linear,
    pooled_in: Linear,
    visual_in: Linear,
    text_blocks: Vec<TextBlock>,
    visual_blocks: Vec<VisualBlock>,
    out_mod: Modulation,
    out_linear: Linear,
}

impl Kandinsky5Transformer {
    pub fn zeros(cfg: Kandinsky5TransformerConfig) -> Result<Self> {
        let vin = cfg.visual_embed_in_dim();
        let patch = cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2];
        let text_blocks = (0..cfg.num_text_blocks)
            .map(|_| TextBlock::zeros(&cfg))
            .collect::<Result<Vec<_>>>()?;
        let visual_blocks = (0..cfg.num_visual_blocks)
            .map(|_| VisualBlock::zeros(&cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            time_in: Linear::zeros(cfg.model_dim, cfg.time_dim, true),
            time_out: Linear::zeros(cfg.time_dim, cfg.time_dim, true),
            text_in: Linear::zeros(cfg.in_text_dim, cfg.model_dim, true),
            pooled_in: Linear::zeros(cfg.in_text_dim2, cfg.time_dim, true),
            visual_in: Linear::zeros(patch * vin, cfg.model_dim, true),
            text_blocks,
            visual_blocks,
            out_mod: Modulation::zeros(cfg.time_dim, cfg.model_dim, 2),
            out_linear: Linear::zeros(cfg.model_dim, patch * cfg.out_visual_dim, true),
            cfg,
        })
    }

    pub fn load(cfg: Kandinsky5TransformerConfig, map: &WeightMap) -> Result<Self> {
        let vin = cfg.visual_embed_in_dim();
        let patch = cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2];
        let text_blocks = (0..cfg.num_text_blocks)
            .map(|i| TextBlock::load(map, &format!("text_transformer_blocks.{i}"), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let visual_blocks = (0..cfg.num_visual_blocks)
            .map(|i| VisualBlock::load(map, &format!("visual_transformer_blocks.{i}"), &cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            time_in: Linear::load(
                map,
                "time_embeddings.in_layer",
                cfg.model_dim,
                cfg.time_dim,
                true,
            )?,
            time_out: Linear::load(
                map,
                "time_embeddings.out_layer",
                cfg.time_dim,
                cfg.time_dim,
                true,
            )?,
            text_in: Linear::load(
                map,
                "text_embeddings.in_layer",
                cfg.in_text_dim,
                cfg.model_dim,
                true,
            )?,
            pooled_in: Linear::load(
                map,
                "pooled_text_embeddings.in_layer",
                cfg.in_text_dim2,
                cfg.time_dim,
                true,
            )?,
            visual_in: Linear::load(
                map,
                "visual_embeddings.in_layer",
                patch * vin,
                cfg.model_dim,
                true,
            )?,
            text_blocks,
            visual_blocks,
            out_mod: Modulation::load(map, "out_layer.modulation", cfg.time_dim, cfg.model_dim, 2)?,
            out_linear: Linear::load(
                map,
                "out_layer.out_layer",
                cfg.model_dim,
                patch * cfg.out_visual_dim,
                true,
            )?,
            cfg,
        })
    }

    pub fn config(&self) -> &Kandinsky5TransformerConfig {
        &self.cfg
    }

    /// `latents` `[B,T,H,W,C]` → same layout, `out_visual_dim` channels.
    pub fn forward(
        &self,
        latents: &CudaTensor,
        text: &CudaTensor,
        pooled: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let [b, t, h, w, c] = match latents.shape[..] {
            [b, t, h, w, c] => [b, t, h, w, c],
            _ => {
                return Err(msg(format!(
                    "k5: latents {:?} want [B,T,H,W,C]",
                    latents.shape
                )))
            }
        };
        let want_c = self.cfg.visual_embed_in_dim();
        if c != want_c {
            return Err(msg(format!("k5: channels {c} vs visual_in {want_c}")));
        }
        let [pt, ph, pw] = self.cfg.patch_size;
        if t % pt != 0 || h % ph != 0 || w % pw != 0 {
            return Err(msg(format!(
                "k5: grid {t}x{h}x{w} not divisible by patch {pt}x{ph}x{pw}"
            )));
        }
        let (tp, hp, wp) = (t / pt, h / ph, w / pw);

        // Patchify → [B, Tp, Hp, Wp, patch*C]
        let x = latents
            .reshape(vec![b, tp, pt, hp, ph, wp, pw, c])?
            .permute(&[0, 1, 3, 5, 2, 4, 6, 7])?
            .reshape(vec![b, tp, hp, wp, pt * ph * pw * c])?;
        let mut visual =
            self.visual_in
                .forward(&x.reshape(vec![b, tp * hp * wp, pt * ph * pw * c])?)?;
        visual = visual.reshape(vec![b, tp, hp, wp, self.cfg.model_dim])?;

        let mut text = self.text_in.forward(text)?;
        text = text.layer_norm(1e-6, None, None)?; // Diffusers TextEmbeddings has LayerNorm

        let temb = sinusoid_timestep(timestep, self.cfg.model_dim);
        let temb = CudaTensor::from_vec(temb, vec![1, self.cfg.model_dim])?.to_device()?;
        let mut time = self
            .time_out
            .forward(&self.time_in.forward(&temb)?.silu())?;
        let pooled = self
            .pooled_in
            .forward(pooled)?
            .layer_norm(1e-6, None, None)?;
        time = time.add(&pooled)?;

        let text_len = text.shape[1];
        let text_rope = rope_1d(
            self.cfg.head_dim(),
            &(0..text_len).collect::<Vec<_>>(),
            10000.0,
        );
        for block in &self.text_blocks {
            text = block.forward(&text, &time, &text_rope)?;
        }

        let vis_rope = rope_3d(&self.cfg, b, tp, hp, wp, 10000.0);
        // Flatten spatial for attention: [B, Tp*Hp*Wp, D]
        let seq = tp * hp * wp;
        let mut visual = visual.reshape(vec![b, seq, self.cfg.model_dim])?;
        // Flatten rope [B,T,H,W,half,2,2] → per-token for apply (batch 0)
        let rope_flat = if b == 1 {
            vis_rope
        } else {
            // Broadcast batch-0 table; full multi-batch rope apply later.
            vis_rope[..seq * (self.cfg.head_dim() / 2) * 4].to_vec()
        };
        for block in &self.visual_blocks {
            visual = block.forward(&visual, &text, &time, &rope_flat)?;
        }
        visual = visual.reshape(vec![b, tp, hp, wp, self.cfg.model_dim])?;

        let mods = self.out_mod.forward(&time)?.unsqueeze(1)?;
        let parts = mods.chunk(2, mods.rank() - 1)?;
        let n = visual
            .reshape(vec![b, seq, self.cfg.model_dim])?
            .layer_norm(1e-6, None, None)?;
        // scale/shift broadcast: Diffusers uses [:, None, None] on time — we have [B,1,D]
        let n = scale_shift(&n, &parts[1], &parts[0])?;
        let tokens = self.out_linear.forward(&n)?;
        // Unpatchify to [B,T,H,W,out_c]
        let oc = self.cfg.out_visual_dim;
        let tokens = tokens.reshape(vec![b, tp, hp, wp, oc, pt, ph, pw])?;
        let out = tokens
            .permute(&[0, 1, 5, 2, 6, 3, 7, 4])?
            .reshape(vec![b, t, h, w, oc])?;
        Ok(out)
    }
}

fn sinusoid_timestep(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let freq = (-(10000f64).ln() * (i as f64) / half as f64).exp() as f32;
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::kandinsky5::Kandinsky5TransformerConfig;

    #[test]
    fn tiny_forward_shapes() {
        let cfg = Kandinsky5TransformerConfig::tiny();
        let dit = Kandinsky5Transformer::zeros(cfg.clone()).unwrap();
        // [B,T,H,W,C] channel-last
        let lat = CudaTensor::zeros(&[1, 1, 2, 2, cfg.in_visual_dim]);
        let text = CudaTensor::zeros(&[1, 3, cfg.in_text_dim]);
        let pooled = CudaTensor::zeros(&[1, cfg.in_text_dim2]);
        let out = dit.forward(&lat, &text, &pooled, 1.0).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2, cfg.out_visual_dim]);
    }
}
