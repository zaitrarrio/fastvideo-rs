//! WanTransformer3D matching Hugging Face Diffusers weight names (zeros init).

use burn::prelude::*;

use fastvideo_models::WanVideoArchConfig;

use super::nn::{
    conv2d_nhwc, gelu_tanh, layer_norm, rms_norm, sdpa, silu, sinusoidal_timesteps, Linear, B,
    Device,
};
use super::weights::{self, WeightMap};
use crate::error::Result;

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor<B, 1>,
    eps: f32,
}

impl RmsNorm {
    fn zeros(dim: usize, eps: f32, device: &Device) -> Self {
        Self {
            weight: Tensor::zeros([dim], device),
            eps,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, eps: f32, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: weights::tensor1_shaped(map, &weights::join_key(prefix, "weight"), dim, device)?,
            eps,
        })
    }

    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        rms_norm(xs, self.weight.clone(), self.eps)
    }
}

#[derive(Debug, Clone)]
struct WanAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    add_k: Option<Linear>,
    add_v: Option<Linear>,
    heads: usize,
    dim_head: usize,
}

impl WanAttention {
    fn zeros(dim: usize, heads: usize, eps: f32, added_kv: Option<usize>, device: &Device) -> Self {
        let dim_head = dim / heads;
        let (add_k, add_v) = if let Some(extra) = added_kv {
            (
                Some(Linear::zeros(extra, dim, device)),
                Some(Linear::zeros(extra, dim, device)),
            )
        } else {
            (None, None)
        };
        Self {
            to_q: Linear::zeros(dim, dim, device),
            to_k: Linear::zeros(dim, dim, device),
            to_v: Linear::zeros(dim, dim, device),
            to_out: Linear::zeros(dim, dim, device),
            norm_q: RmsNorm::zeros(dim, eps, device),
            norm_k: RmsNorm::zeros(dim, eps, device),
            add_k,
            add_v,
            heads,
            dim_head,
        }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        dim: usize,
        heads: usize,
        eps: f32,
        added_kv: Option<usize>,
        device: &Device,
    ) -> Result<Self> {
        let dim_head = dim / heads;
        let (add_k, add_v) = if let Some(extra) = added_kv {
            (
                Some(Linear::load(
                    map,
                    &weights::join_key(prefix, "add_k_proj"),
                    extra,
                    dim,
                    true,
                    device,
                )?),
                Some(Linear::load(
                    map,
                    &weights::join_key(prefix, "add_v_proj"),
                    extra,
                    dim,
                    true,
                    device,
                )?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            to_q: Linear::load(map, &weights::join_key(prefix, "to_q"), dim, dim, true, device)?,
            to_k: Linear::load(map, &weights::join_key(prefix, "to_k"), dim, dim, true, device)?,
            to_v: Linear::load(map, &weights::join_key(prefix, "to_v"), dim, dim, true, device)?,
            to_out: Linear::load(
                map,
                &weights::join_key(prefix, "to_out.0"),
                dim,
                dim,
                true,
                device,
            )?,
            norm_q: RmsNorm::load(map, &weights::join_key(prefix, "norm_q"), dim, eps, device)?,
            norm_k: RmsNorm::load(map, &weights::join_key(prefix, "norm_k"), dim, eps, device)?,
            add_k,
            add_v,
            heads,
            dim_head,
        })
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        encoder: Option<Tensor<B, 3>>,
        rotary: Option<&(Tensor<B, 4>, Tensor<B, 4>)>,
        image: Option<Tensor<B, 3>>,
        attn_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let ctx = encoder.unwrap_or_else(|| hidden.clone());
        let q = self.norm_q.forward(self.to_q.forward_3(hidden));
        let mut k = self.norm_k.forward(self.to_k.forward_3(ctx.clone()));
        let mut v = self.to_v.forward_3(ctx);
        if let (Some(add_k), Some(add_v), Some(img)) = (&self.add_k, &self.add_v, image) {
            let ik = add_k.forward_3(img.clone());
            let iv = add_v.forward_3(img);
            k = Tensor::cat(vec![ik, k], 1);
            v = Tensor::cat(vec![iv, v], 1);
        }
        let [b, sq, _] = q.dims();
        let sk = k.dims()[1];
        let q = q.reshape([b, sq, self.heads, self.dim_head]);
        let k = k.reshape([b, sk, self.heads, self.dim_head]);
        let v = v.reshape([b, sk, self.heads, self.dim_head]);
        let (q, k) = if let Some((cos, sin)) = rotary {
            (apply_rotary(q, cos, sin), apply_rotary(k, cos, sin))
        } else {
            (q, k)
        };
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        let attn = sdpa(q, k, v, attn_mask);
        let attn = attn
            .swap_dims(1, 2)
            .reshape([b, sq, self.heads * self.dim_head]);
        self.to_out.forward_3(attn)
    }
}

fn pair_last_dim(xs: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let d = xs.dims()[3];
    let half = d / 2;
    let [b, h, s, _] = xs.dims();
    let xs = xs.reshape([b, h, s, half, 2]);
    let even = xs.clone().narrow(4, 0, 1).squeeze::<4>(4);
    let odd = xs.narrow(4, 1, 1).squeeze::<4>(4);
    (even, odd)
}

fn apply_rotary(xs: Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
    let (x1, x2) = pair_last_dim(xs);
    let (cos_e, _) = pair_last_dim(cos.clone());
    let (_, sin_o) = pair_last_dim(sin.clone());
    let out1 = x1.clone() * cos_e.clone() - x2.clone() * sin_o.clone();
    let out2 = x1 * sin_o + x2 * cos_e;
    let stacked = Tensor::cat(
        vec![out1.unsqueeze_dim::<5>(4), out2.unsqueeze_dim::<5>(4)],
        4,
    );
    let [b, h, s, half, pair] = stacked.dims();
    stacked.reshape([b, h, s, half * pair])
}

fn rotary_1d(dim: usize, seq: usize, theta: f64, device: &Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half = dim / 2;
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta.powf(2.0 * i as f64 / dim as f64) as f32))
        .collect();
    let pos = Tensor::<B, 1>::from_floats(pos.as_slice(), device).reshape([seq, 1]);
    let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), device).reshape([1, half]);
    let args = pos * freqs;
    let cos = args.clone().cos();
    let sin = args.sin();
    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();
    for i in 0..half {
        let c = cos.clone().narrow(1, i, 1);
        let s = sin.clone().narrow(1, i, 1);
        cos_parts.push(c.clone());
        cos_parts.push(c);
        sin_parts.push(s.clone());
        sin_parts.push(s);
    }
    (Tensor::cat(cos_parts, 1), Tensor::cat(sin_parts, 1))
}

fn wan_rope(
    cfg: &WanVideoArchConfig,
    frames: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let d = cfg.attention_head_dim;
    let h_dim = 2 * (d / 6);
    let w_dim = h_dim;
    let t_dim = d - h_dim - w_dim;
    let (cos_t, sin_t) = rotary_1d(t_dim, cfg.rope_max_seq_len, 10000.0, device);
    let (cos_h, sin_h) = rotary_1d(h_dim, cfg.rope_max_seq_len, 10000.0, device);
    let (cos_w, sin_w) = rotary_1d(w_dim, cfg.rope_max_seq_len, 10000.0, device);
    let ppf = frames / cfg.patch_size[0];
    let pph = height / cfg.patch_size[1];
    let ppw = width / cfg.patch_size[2];
    let cos_f = cos_t
        .narrow(0, 0, ppf)
        .reshape([ppf, 1, 1, t_dim])
        .expand([ppf, pph, ppw, t_dim]);
    let cos_hh = cos_h
        .narrow(0, 0, pph)
        .reshape([1, pph, 1, h_dim])
        .expand([ppf, pph, ppw, h_dim]);
    let cos_ww = cos_w
        .narrow(0, 0, ppw)
        .reshape([1, 1, ppw, w_dim])
        .expand([ppf, pph, ppw, w_dim]);
    let sin_f = sin_t
        .narrow(0, 0, ppf)
        .reshape([ppf, 1, 1, t_dim])
        .expand([ppf, pph, ppw, t_dim]);
    let sin_hh = sin_h
        .narrow(0, 0, pph)
        .reshape([1, pph, 1, h_dim])
        .expand([ppf, pph, ppw, h_dim]);
    let sin_ww = sin_w
        .narrow(0, 0, ppw)
        .reshape([1, 1, ppw, w_dim])
        .expand([ppf, pph, ppw, w_dim]);
    let seq = ppf * pph * ppw;
    let cos = Tensor::cat(vec![cos_f, cos_hh, cos_ww], 3).reshape([1, seq, 1, d]);
    let sin = Tensor::cat(vec![sin_f, sin_hh, sin_ww], 3).reshape([1, seq, 1, d]);
    (cos, sin)
}

#[derive(Debug, Clone)]
struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn zeros(dim: usize, ffn_dim: usize, device: &Device) -> Self {
        Self {
            proj: Linear::zeros(dim, ffn_dim, device),
            out: Linear::zeros(ffn_dim, dim, device),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(
                map,
                &weights::join_key(prefix, "net.0.proj"),
                dim,
                ffn_dim,
                true,
                device,
            )?,
            out: Linear::load(
                map,
                &weights::join_key(prefix, "net.2"),
                ffn_dim,
                dim,
                true,
                device,
            )?,
        })
    }

    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        self.out.forward_3(gelu_tanh(self.proj.forward_3(xs)))
    }
}

#[derive(Debug, Clone)]
struct TextProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl TextProjection {
    fn zeros(in_dim: usize, dim: usize, device: &Device) -> Self {
        Self {
            linear_1: Linear::zeros(in_dim, dim, device),
            linear_2: Linear::zeros(dim, dim, device),
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(
                map,
                &weights::join_key(prefix, "linear_1"),
                in_dim,
                dim,
                true,
                device,
            )?,
            linear_2: Linear::load(
                map,
                &weights::join_key(prefix, "linear_2"),
                dim,
                dim,
                true,
                device,
            )?,
        })
    }

    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear_2
            .forward_3(gelu_tanh(self.linear_1.forward_3(xs)))
    }
}

#[derive(Debug, Clone)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn zeros(in_dim: usize, dim: usize, device: &Device) -> Self {
        Self {
            linear_1: Linear::zeros(in_dim, dim, device),
            linear_2: Linear::zeros(dim, dim, device),
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(
                map,
                &weights::join_key(prefix, "linear_1"),
                in_dim,
                dim,
                true,
                device,
            )?,
            linear_2: Linear::load(
                map,
                &weights::join_key(prefix, "linear_2"),
                dim,
                dim,
                true,
                device,
            )?,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear_2.forward_2(silu(self.linear_1.forward_2(xs)))
    }
}

#[derive(Debug, Clone)]
struct WanBlock {
    norm1_eps: f32,
    attn1: WanAttention,
    attn2: WanAttention,
    norm2_weight: Tensor<B, 1>,
    norm2_bias: Tensor<B, 1>,
    ffn: FeedForward,
    scale_shift_table: Tensor<B, 3>,
}

impl WanBlock {
    fn zeros(cfg: &WanVideoArchConfig, device: &Device) -> Self {
        let dim = cfg.hidden_size();
        Self {
            norm1_eps: cfg.eps,
            attn1: WanAttention::zeros(dim, cfg.num_attention_heads, cfg.eps, None, device),
            attn2: WanAttention::zeros(
                dim,
                cfg.num_attention_heads,
                cfg.eps,
                cfg.added_kv_proj_dim,
                device,
            ),
            norm2_weight: Tensor::zeros([dim], device),
            norm2_bias: Tensor::zeros([dim], device),
            ffn: FeedForward::zeros(dim, cfg.ffn_dim, device),
            scale_shift_table: Tensor::zeros([1, 6, dim], device),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &WanVideoArchConfig, device: &Device) -> Result<Self> {
        let dim = cfg.hidden_size();
        Ok(Self {
            norm1_eps: cfg.eps,
            attn1: WanAttention::load(
                map,
                &weights::join_key(prefix, "attn1"),
                dim,
                cfg.num_attention_heads,
                cfg.eps,
                None,
                device,
            )?,
            attn2: WanAttention::load(
                map,
                &weights::join_key(prefix, "attn2"),
                dim,
                cfg.num_attention_heads,
                cfg.eps,
                cfg.added_kv_proj_dim,
                device,
            )?,
            norm2_weight: weights::tensor1_shaped(
                map,
                &weights::join_key(prefix, "norm2.weight"),
                dim,
                device,
            )?,
            norm2_bias: weights::tensor1_shaped(
                map,
                &weights::join_key(prefix, "norm2.bias"),
                dim,
                device,
            )?,
            ffn: FeedForward::load(map, &weights::join_key(prefix, "ffn"), dim, cfg.ffn_dim, device)?,
            scale_shift_table: weights::tensor3_shaped(
                map,
                &weights::join_key(prefix, "scale_shift_table"),
                [1, 6, dim],
                device,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        encoder: &Tensor<B, 3>,
        temb: &Tensor<B, 3>,
        rotary: &(Tensor<B, 4>, Tensor<B, 4>),
        image: Option<Tensor<B, 3>>,
        attn_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let e = self.scale_shift_table.clone() + temb.clone();
        let chunks: Vec<_> = (0..6)
            .map(|i| e.clone().narrow(1, i, 1))
            .collect();
        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];
        let c_shift = &chunks[3];
        let c_scale = &chunks[4];
        let c_gate = &chunks[5];

        let normed = layer_norm(hidden.clone(), self.norm1_eps, None, None);
        let normed = normed * (scale_msa.clone().add_scalar(1.0)) + shift_msa.clone();
        let attn = self
            .attn1
            .forward(normed, None, Some(rotary), None, attn_mask);
        let hidden = hidden + attn * gate_msa.clone();

        let normed = layer_norm(
            hidden.clone(),
            self.norm1_eps,
            Some(&self.norm2_weight),
            Some(&self.norm2_bias),
        );
        let attn = self
            .attn2
            .forward(normed, Some(encoder.clone()), None, image, None);
        let hidden = hidden + attn;

        let normed = layer_norm(hidden.clone(), self.norm1_eps, None, None);
        let normed = normed * (c_scale.clone().add_scalar(1.0)) + c_shift.clone();
        let ff = self.ffn.forward(normed);
        hidden + ff * c_gate.clone()
    }
}

#[derive(Debug, Clone)]
pub struct WanTransformer3D {
    pub cfg: WanVideoArchConfig,
    patch_weight: Tensor<B, 5>,
    patch_bias: Tensor<B, 1>,
    time_embedder: TimestepEmbedding,
    time_proj: Linear,
    text_embedder: TextProjection,
    blocks: Vec<WanBlock>,
    proj_out: Linear,
    scale_shift_table: Tensor<B, 3>,
    freq_dim: usize,
    device: Device,
}

impl WanTransformer3D {
    pub fn zeros(cfg: WanVideoArchConfig, device: &Device) -> Self {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            blocks.push(WanBlock::zeros(&cfg, device));
        }
        Self {
            patch_weight: Tensor::zeros([dim, cfg.in_channels, p[0], p[1], p[2]], device),
            patch_bias: Tensor::zeros([dim], device),
            time_embedder: TimestepEmbedding::zeros(cfg.freq_dim, dim, device),
            time_proj: Linear::zeros(dim, dim * 6, device),
            text_embedder: TextProjection::zeros(cfg.text_dim, dim, device),
            blocks,
            proj_out: Linear::zeros(dim, cfg.out_channels * p.iter().product::<usize>(), device),
            scale_shift_table: Tensor::zeros([1, 2, dim], device),
            freq_dim: cfg.freq_dim,
            device: device.clone(),
            cfg,
        }
    }

    pub fn load(cfg: WanVideoArchConfig, map: &WeightMap, device: &Device) -> Result<Self> {
        Self::from_map(cfg, map, device)
    }

    pub fn from_map(cfg: WanVideoArchConfig, map: &WeightMap, device: &Device) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(WanBlock::load(map, &format!("blocks.{i}"), &cfg, device)?);
        }
        Ok(Self {
            patch_weight: weights::tensor5_shaped(
                map,
                "patch_embedding.weight",
                [dim, cfg.in_channels, p[0], p[1], p[2]],
                device,
            )?,
            patch_bias: weights::tensor1_shaped(map, "patch_embedding.bias", dim, device)?,
            time_embedder: TimestepEmbedding::load(
                map,
                "condition_embedder.time_embedder",
                cfg.freq_dim,
                dim,
                device,
            )?,
            time_proj: Linear::load(
                map,
                "condition_embedder.time_proj",
                dim,
                dim * 6,
                true,
                device,
            )?,
            text_embedder: TextProjection::load(
                map,
                "condition_embedder.text_embedder",
                cfg.text_dim,
                dim,
                device,
            )?,
            blocks,
            proj_out: Linear::load(
                map,
                "proj_out",
                dim,
                cfg.out_channels * p.iter().product::<usize>(),
                true,
                device,
            )?,
            scale_shift_table: weights::tensor3_shaped(
                map,
                "scale_shift_table",
                [1, 2, dim],
                device,
            )?,
            freq_dim: cfg.freq_dim,
            device: device.clone(),
            cfg,
        })
    }

    fn patch_embed(&self, xs: Tensor<B, 5>) -> Tensor<B, 3> {
        let [b, c, t, h, w] = xs.dims();
        let p = self.cfg.patch_size;
        let x = xs.swap_dims(1, 2).reshape([b * t, c, h, w]);
        let k = self
            .patch_weight
            .clone()
            .reshape([self.cfg.hidden_size(), c * p[0], p[1], p[2]]);
        let y = conv2d_nhwc(x, k, None, 0, p[1]);
        let y = y
            + self
                .patch_bias
                .clone()
                .reshape([1, self.cfg.hidden_size(), 1, 1]);
        let [_, dim, hh, ww] = y.dims();
        y.reshape([b, t, dim, hh, ww])
            .permute([0, 2, 1, 3, 4])
            .reshape([b, dim, t * hh * ww])
            .swap_dims(1, 2)
    }

    pub fn forward(
        &self,
        latents: Tensor<B, 5>,
        timestep: Tensor<B, 1>,
        encoder: Tensor<B, 3>,
    ) -> Tensor<B, 5> {
        self.forward_ctx(latents, timestep, encoder, None)
    }

    pub fn forward_ctx(
        &self,
        latents: Tensor<B, 5>,
        timestep: Tensor<B, 1>,
        encoder: Tensor<B, 3>,
        image: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 5> {
        let [b, _c, t, h, w] = latents.dims();
        let device = &self.device;
        let rotary = wan_rope(&self.cfg, t, h, w, device);
        let mut hidden = self.patch_embed(latents);
        let temb_in = sinusoidal_timesteps(timestep, self.freq_dim, device);
        let temb = self.time_embedder.forward(temb_in);
        let timestep_proj = self
            .time_proj
            .forward_2(silu(temb.clone()))
            .reshape([b, 6, self.cfg.hidden_size()]);
        let encoder = self.text_embedder.forward(encoder);
        for block in &self.blocks {
            hidden = block.forward(
                hidden,
                &encoder,
                &timestep_proj,
                &rotary,
                image.clone(),
                None,
            );
        }
        let ss = self.scale_shift_table.clone() + temb.unsqueeze_dim::<3>(1);
        let shift = ss.clone().narrow(1, 0, 1);
        let scale = ss.narrow(1, 1, 1);
        hidden = layer_norm(hidden, self.cfg.eps, None, None);
        hidden = hidden * (scale.add_scalar(1.0)) + shift;
        hidden = self.proj_out.forward_3(hidden);
        let p = self.cfg.patch_size;
        let ppf = t / p[0];
        let pph = h / p[1];
        let ppw = w / p[2];
        unpatchify(hidden, b, self.cfg.out_channels, ppf, pph, ppw, p)
    }
}

/// Pixel-shuffle unpatch without rank-8 tensors (NdArray max rank is 6).
fn unpatchify(
    hidden: Tensor<B, 3>,
    b: usize,
    oc: usize,
    ppf: usize,
    pph: usize,
    ppw: usize,
    p: [usize; 3],
) -> Tensor<B, 5> {
    // [B, ppf*pph*ppw, p0*p1*p2*C]
    let x = hidden.reshape([b, ppf, pph, ppw, p[0] * p[1] * p[2] * oc]);
    // [B, ppf, pph, ppw, p0, p1*p2*C]
    let x = x.reshape([b, ppf, pph, ppw, p[0], p[1] * p[2] * oc]);
    // [B, ppf, p0, pph, ppw, p1*p2*C]
    let x = x.permute([0, 1, 4, 2, 3, 5]);
    // [B, ppf*p0, pph, ppw, p1, p2*C]
    let x = x.reshape([b, ppf * p[0], pph, ppw, p[1], p[2] * oc]);
    // [B, ppf*p0, pph, p1, ppw, p2*C]
    let x = x.permute([0, 1, 2, 4, 3, 5]);
    // [B, ppf*p0, pph*p1, ppw, p2, C]
    let x = x.reshape([b, ppf * p[0], pph * p[1], ppw, p[2], oc]);
    // [B, T, H, W, C]
    let x = x.reshape([b, ppf * p[0], pph * p[1], ppw * p[2], oc]);
    // [B, C, T, H, W]
    x.permute([0, 4, 1, 2, 3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_dit_forward_shape() {
        let device = Default::default();
        let cfg = WanVideoArchConfig::tiny();
        let model = WanTransformer3D::zeros(cfg, &device);
        let latents = Tensor::<B, 5>::zeros([1, 4, 2, 8, 8], &device);
        let t = Tensor::<B, 1>::from_floats([500f32], &device);
        let enc = Tensor::<B, 3>::zeros([1, 8, 16], &device);
        let out = model.forward(latents, t, enc);
        assert_eq!(out.dims(), [1, 4, 2, 8, 8]);
    }
}
