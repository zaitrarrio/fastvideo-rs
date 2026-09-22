//! HunyuanVideo 1.5 MMDiT: patch embed, dual-stream double blocks, final layer.
//!
//! Custom weight prefix `Hunyuan15.*` (FastVideo remap from HF
//! `transformer_blocks`). Tiny configs exercise the joint-attention path.

use fastvideo_models::hunyuan15::{Hunyuan15RopeTables, Hunyuan15TransformerConfig};

use crate::wan::fused::Rope;
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

/// Wan `qk_norm_rope_bhsd` wants RMS weight `[heads * head_dim]`. Diffusers
/// Hunyuan stores per-head `[head_dim]`; tile across heads.
fn qk_norm_weight(map: &WeightMap, key: &str, heads: usize, head_dim: usize) -> Result<CudaTensor> {
    let full = heads * head_dim;
    if let Ok(w) = weights::cuda_tensor_shaped(map, key, &[full]) {
        return pinned(w);
    }
    let w = weights::cuda_tensor_shaped(map, key, &[head_dim])?;
    let row = w.host_cow()?;
    let mut tiled = Vec::with_capacity(full);
    for _ in 0..heads {
        tiled.extend_from_slice(&row);
    }
    pinned(CudaTensor::from_vec(tiled, vec![full])?)
}

fn qk_norm_ones(heads: usize, head_dim: usize) -> Result<CudaTensor> {
    pinned(CudaTensor::ones(&[heads * head_dim]))
}

/// `x · (1 + scale) + shift` with broadcast modulation rows (`[1, dim]` or `[B,1,dim]`).
fn scale_shift(x: &CudaTensor, scale: &CudaTensor, shift: &CudaTensor) -> Result<CudaTensor> {
    x.mul(&scale.try_add_scalar(1.0)?)?.add(shift)
}

fn gated(x: &CudaTensor, y: &CudaTensor, gate: &CudaTensor) -> Result<CudaTensor> {
    x.add(&y.mul(gate)?)
}

/// One MM double-stream block (img + txt, joint attention).
struct DoubleBlock {
    img_mod: Linear,
    txt_mod: Linear,
    img_attn_qkv: Linear,
    txt_attn_qkv: Linear,
    img_attn_proj: Linear,
    txt_attn_proj: Linear,
    img_mlp_fc1: Linear,
    img_mlp_fc2: Linear,
    txt_mlp_fc1: Linear,
    txt_mlp_fc2: Linear,
    img_q_norm: CudaTensor,
    img_k_norm: CudaTensor,
    txt_q_norm: CudaTensor,
    txt_k_norm: CudaTensor,
    heads: usize,
    dim_head: usize,
    eps: f32,
}

impl DoubleBlock {
    fn load(map: &WeightMap, prefix: &str, cfg: &Hunyuan15TransformerConfig) -> Result<Self> {
        let h = cfg.hidden_size();
        let mlp = cfg.mlp_hidden();
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            img_mod: Linear::load(map, &key("img_mod.linear"), h, 6 * h, true)?,
            txt_mod: Linear::load(map, &key("txt_mod.linear"), h, 6 * h, true)?,
            img_attn_qkv: Linear::load(map, &key("img_attn_qkv"), h, 3 * h, true)?,
            txt_attn_qkv: Linear::load(map, &key("txt_attn_qkv"), h, 3 * h, true)?,
            img_attn_proj: Linear::load(map, &key("img_attn_proj"), h, h, true)?,
            txt_attn_proj: Linear::load(map, &key("txt_attn_proj"), h, h, true)?,
            img_mlp_fc1: Linear::load(map, &key("img_mlp.fc_in"), h, mlp, true)?,
            img_mlp_fc2: Linear::load(map, &key("img_mlp.fc_out"), mlp, h, true)?,
            txt_mlp_fc1: Linear::load(map, &key("txt_mlp.fc_in"), h, mlp, true)?,
            txt_mlp_fc2: Linear::load(map, &key("txt_mlp.fc_out"), mlp, h, true)?,
            img_q_norm: qk_norm_weight(map, &key("img_attn_q_norm.weight"), cfg.num_attention_heads, cfg.attention_head_dim)?,
            img_k_norm: qk_norm_weight(map, &key("img_attn_k_norm.weight"), cfg.num_attention_heads, cfg.attention_head_dim)?,
            txt_q_norm: qk_norm_weight(map, &key("txt_attn_q_norm.weight"), cfg.num_attention_heads, cfg.attention_head_dim)?,
            txt_k_norm: qk_norm_weight(map, &key("txt_attn_k_norm.weight"), cfg.num_attention_heads, cfg.attention_head_dim)?,
            heads: cfg.num_attention_heads,
            dim_head: cfg.attention_head_dim,
            eps: cfg.rms_eps,
        })
    }

    fn zeros(cfg: &Hunyuan15TransformerConfig) -> Result<Self> {
        let h = cfg.hidden_size();
        let mlp = cfg.mlp_hidden();
        let heads = cfg.num_attention_heads;
        let hd = cfg.attention_head_dim;
        Ok(Self {
            img_mod: Linear::zeros(h, 6 * h, true),
            txt_mod: Linear::zeros(h, 6 * h, true),
            img_attn_qkv: Linear::zeros(h, 3 * h, true),
            txt_attn_qkv: Linear::zeros(h, 3 * h, true),
            img_attn_proj: Linear::zeros(h, h, true),
            txt_attn_proj: Linear::zeros(h, h, true),
            img_mlp_fc1: Linear::zeros(h, mlp, true),
            img_mlp_fc2: Linear::zeros(mlp, h, true),
            txt_mlp_fc1: Linear::zeros(h, mlp, true),
            txt_mlp_fc2: Linear::zeros(mlp, h, true),
            img_q_norm: qk_norm_ones(heads, hd)?,
            img_k_norm: qk_norm_ones(heads, hd)?,
            txt_q_norm: qk_norm_ones(heads, hd)?,
            txt_k_norm: qk_norm_ones(heads, hd)?,
            heads,
            dim_head: hd,
            eps: cfg.rms_eps,
        })
    }

    fn forward(
        &self,
        img: &CudaTensor,
        txt: &CudaTensor,
        vec: &CudaTensor,
        rope: Option<&(CudaTensor, CudaTensor)>,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let mods_i = self.img_mod.forward(&vec.silu())?;
        let mods_t = self.txt_mod.forward(&vec.silu())?;
        let parts_i = mods_i.chunk(6, mods_i.rank() - 1)?;
        let parts_t = mods_t.chunk(6, mods_t.rank() - 1)?;
        let (i_sh, i_sc, i_gate, i_msh, i_msc, i_mgate) = (
            &parts_i[0], &parts_i[1], &parts_i[2], &parts_i[3], &parts_i[4], &parts_i[5],
        );
        let (t_sh, t_sc, t_gate, t_msh, t_msc, t_mgate) = (
            &parts_t[0], &parts_t[1], &parts_t[2], &parts_t[3], &parts_t[4], &parts_t[5],
        );

        let img_n = scale_shift(&img.layer_norm(1e-6, None, None)?, i_sc, i_sh)?;
        let txt_n = scale_shift(&txt.layer_norm(1e-6, None, None)?, t_sc, t_sh)?;

        let iqkv = self.img_attn_qkv.forward(&img_n)?;
        let tqkv = self.txt_attn_qkv.forward(&txt_n)?;
        let rope_r = rope.map(|(c, s)| Rope { cos: c, sin: s });
        let iq = iqkv.qk_norm_rope_bhsd(0, self.heads, &self.img_q_norm, rope_r, self.eps)?;
        let ik = iqkv.qk_norm_rope_bhsd(
            self.heads * self.dim_head,
            self.heads,
            &self.img_k_norm,
            rope.map(|(c, s)| Rope { cos: c, sin: s }),
            self.eps,
        )?;
        let iv = iqkv.split_heads_bhsd(2 * self.heads * self.dim_head, self.heads, self.dim_head)?;
        let tq = tqkv.qk_norm_rope_bhsd(0, self.heads, &self.txt_q_norm, None, self.eps)?;
        let tk = tqkv.qk_norm_rope_bhsd(
            self.heads * self.dim_head,
            self.heads,
            &self.txt_k_norm,
            None,
            self.eps,
        )?;
        let tv = tqkv.split_heads_bhsd(2 * self.heads * self.dim_head, self.heads, self.dim_head)?;

        let q = CudaTensor::cat(&[&iq, &tq], 2)?;
        let k = CudaTensor::cat(&[&ik, &tk], 2)?;
        let v = CudaTensor::cat(&[&iv, &tv], 2)?;
        let attn = nn::scaled_dot_product_attention_masked(&q, &k, &v, None, None)?;
        let img_len = img.shape[1];
        let txt_len = attn.shape[2] - img_len;
        let img_attn = attn.narrow(2, 0, img_len)?.merge_heads()?;
        let txt_attn = attn.narrow(2, img_len, txt_len)?.merge_heads()?;

        let img = gated(img, &self.img_attn_proj.forward(&img_attn)?, i_gate)?;
        let txt = gated(txt, &self.txt_attn_proj.forward(&txt_attn)?, t_gate)?;

        let img_m = scale_shift(&img.layer_norm(1e-6, None, None)?, i_msc, i_msh)?;
        let txt_m = scale_shift(&txt.layer_norm(1e-6, None, None)?, t_msc, t_msh)?;
        let img = gated(
            &img,
            &self.img_mlp_fc2.forward(&self.img_mlp_fc1.forward(&img_m)?.gelu_tanh())?,
            i_mgate,
        )?;
        let txt = gated(
            &txt,
            &self.txt_mlp_fc2.forward(&self.txt_mlp_fc1.forward(&txt_m)?.gelu_tanh())?,
            t_mgate,
        )?;
        Ok((img, txt))
    }
}

/// Full transformer (or tiny) for HunyuanVideo 1.5.
pub struct Hunyuan15Transformer {
    cfg: Hunyuan15TransformerConfig,
    img_in: Linear,
    time_in: Linear,
    time_in_2: Linear,
    txt_proj: Linear,
    txt2_proj: Linear,
    blocks: Vec<DoubleBlock>,
    final_mod: Linear,
    final_linear: Linear,
}

impl Hunyuan15Transformer {
    pub fn zeros(cfg: Hunyuan15TransformerConfig) -> Result<Self> {
        let h = cfg.hidden_size();
        let blocks = (0..cfg.num_layers)
            .map(|_| DoubleBlock::zeros(&cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            img_in: Linear::zeros(cfg.in_channels, h, true),
            time_in: Linear::zeros(256, h, true),
            time_in_2: Linear::zeros(h, h, true),
            txt_proj: Linear::zeros(cfg.text_embed_dim, h, true),
            txt2_proj: Linear::zeros(cfg.text_embed_2_dim, h, true),
            blocks,
            final_mod: Linear::zeros(h, 2 * h, true),
            final_linear: Linear::zeros(h, cfg.out_channels, true),
            cfg,
        })
    }

    pub fn load(cfg: Hunyuan15TransformerConfig, map: &WeightMap) -> Result<Self> {
        let prefix = "Hunyuan15";
        let h = cfg.hidden_size();
        let key = |n: &str| format!("{prefix}.{n}");
        let blocks = (0..cfg.num_layers)
            .map(|i| DoubleBlock::load(map, &format!("{prefix}.double_blocks.{i}"), &cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            img_in: Linear::load(map, &key("img_in.proj"), cfg.in_channels, h, true)?,
            time_in: Linear::load(map, &key("time_in.timestep_embedder.mlp.fc_in"), 256, h, true)?,
            time_in_2: Linear::load(map, &key("time_in.timestep_embedder.mlp.fc_out"), h, h, true)?,
            txt_proj: Linear::load(map, &key("txt_in.input_embedder"), cfg.text_embed_dim, h, true)?,
            txt2_proj: Linear::load(map, &key("txt_in_2.linear_3"), cfg.text_embed_2_dim, h, true).or_else(|_| {
                Linear::load(map, &key("txt_in_2.linear_1"), cfg.text_embed_2_dim, h, true)
            })?,
            blocks,
            final_mod: Linear::load(map, &key("final_layer.adaLN_modulation.linear"), h, 2 * h, true)?,
            final_linear: Linear::load(map, &key("final_layer.linear"), h, cfg.out_channels, true)?,
            cfg,
        })
    }

    pub fn config(&self) -> &Hunyuan15TransformerConfig {
        &self.cfg
    }

    /// `latents` `[B,C,T,H,W]` → `[B, T*H*W, out_channels]`.
    pub fn forward(
        &self,
        latents: &CudaTensor,
        text: &CudaTensor,
        text2: Option<&CudaTensor>,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match latents.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg(format!("hy15: latents {:?} want [B,C,T,H,W]", latents.shape))),
        };
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "hy15: in_channels {c} vs config {}",
                self.cfg.in_channels
            )));
        }
        let seq = t * h * w;
        let x = latents.permute(&[0, 2, 3, 4, 1])?.reshape(vec![b, seq, c])?;
        let mut img = self.img_in.forward(&x)?;

        let temb = sinusoid_timestep(timestep, 256);
        let temb = CudaTensor::from_vec(temb, vec![1, 256])?.to_device()?;
        let vec = self.time_in_2.forward(&self.time_in.forward(&temb)?.silu())?;
        let vec = vec.reshape(vec![1, self.cfg.hidden_size()])?;

        let mut txt = self.txt_proj.forward(text)?;
        if let Some(t2) = text2 {
            let p2 = self.txt2_proj.forward(t2)?;
            txt = CudaTensor::cat(&[&txt, &p2], 1)?;
        }

        let rope_tables = Hunyuan15RopeTables::build(&self.cfg, t, h, w).map_err(msg)?;
        let cos =
            CudaTensor::from_vec(rope_tables.cos, vec![seq, self.cfg.attention_head_dim])?.to_device()?;
        let sin =
            CudaTensor::from_vec(rope_tables.sin, vec![seq, self.cfg.attention_head_dim])?.to_device()?;
        let rope = (cos, sin);

        for block in &self.blocks {
            let (ni, nt) = block.forward(&img, &txt, &vec, Some(&rope))?;
            img = ni;
            txt = nt;
        }

        let mods = self.final_mod.forward(&vec.silu())?;
        let parts = mods.chunk(2, mods.rank() - 1)?;
        let img = scale_shift(&img.layer_norm(1e-6, None, None)?, &parts[1], &parts[0])?;
        self.final_linear.forward(&img)
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
    use fastvideo_models::hunyuan15::Hunyuan15TransformerConfig;

    #[test]
    fn tiny_forward_shapes() {
        let cfg = Hunyuan15TransformerConfig::tiny();
        let dit = Hunyuan15Transformer::zeros(cfg.clone()).unwrap();
        let lat = CudaTensor::zeros(&[1, cfg.in_channels, 2, 2, 2]);
        let text = CudaTensor::zeros(&[1, 4, cfg.text_embed_dim]);
        let out = dit.forward(&lat, &text, None, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, 8, cfg.out_channels]);
    }
}
