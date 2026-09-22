//! Decoder-only language models used as *encoders*: one causal forward over a
//! prompt, hidden states out, no sampling and no KV cache.
//!
//! The audio-video models condition on an LLM's internals rather than on a
//! text encoder built for the job — MiniMax-H3 on a middle layer of
//! Qwen3-VL-32B, LTX-2 on Gemma-3-12B — and those are 24–68 GB of weights that
//! are needed for a few hundred milliseconds. So the forward here *streams*:
//! each layer's weights are read from the mapped shards, uploaded, used once
//! and dropped before the next layer's are touched. Device memory holds a few
//! layers (about 1 GB each), host memory holds one, and layers past the last
//! requested hidden state are never read at all. On a device the next layer is
//! staged through pinned memory on a copy stream while the current one
//! computes (see `llm/prefetch.rs`); the numbers are the same either way.
//!
//! One implementation covers both families; [`DecoderConfig`] carries the
//! differences (norm offset, sandwich norms, activation, per-layer rope and
//! window, embedding scale, key prefix).

use crate::wan::nn::{scaled_dot_product_attention_masked, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// `W += scale * lora_up @ lora_down` with Comfy shapes `lora_down: [r, in]`,
/// `lora_up: [out, r]`. Host-only; used by SearchingMan ARA and unit tests.
pub fn merge_comfy_lora_into(
    weight: &mut [f32],
    ara: &WeightMap,
    base: &str,
    out_dim: usize,
    in_dim: usize,
    scale: f32,
) -> Result<()> {
    if weight.len() != out_dim * in_dim {
        return Err(msg(format!("lora merge: {} elements for [{out_dim}, {in_dim}]", weight.len())));
    }
    let down_key = format!("{base}.lora_down.weight");
    let up_key = format!("{base}.lora_up.weight");
    let lazy = ara.lazy().ok_or_else(|| msg("ara map is not lazy"))?;
    let d = lazy.view(&down_key).map_err(|e| msg(e.to_string()))?;
    let u = lazy.view(&up_key).map_err(|e| msg(e.to_string()))?;
    if d.shape.len() != 2 || u.shape.len() != 2 {
        return Err(msg(format!("{base}: lora tensors must be rank-2")));
    }
    let (r, in_a) = (d.shape[0], d.shape[1]);
    let (out_b, r_b) = (u.shape[0], u.shape[1]);
    if in_a != in_dim || out_b != out_dim || r != r_b {
        return Err(msg(format!(
            "{base}: lora shapes down {:?} up {:?} vs weight [{out_dim}, {in_dim}]",
            d.shape, u.shape
        )));
    }
    let down = cuda_tensor_shaped(ara, &down_key, &[r, in_dim])?.host_cow()?.into_owned();
    let up = cuda_tensor_shaped(ara, &up_key, &[out_dim, r])?.host_cow()?.into_owned();
    merge_comfy_lora_host(weight, &down, &up, out_dim, in_dim, r, scale);
    Ok(())
}

/// Host fold: `W[out, in] += scale * up[out, r] @ down[r, in]`.
pub fn merge_comfy_lora_host(
    weight: &mut [f32],
    down: &[f32],
    up: &[f32],
    out_dim: usize,
    in_dim: usize,
    rank: usize,
    scale: f32,
) {
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(down.len(), rank * in_dim);
    debug_assert_eq!(up.len(), out_dim * rank);
    for o in 0..out_dim {
        for i in 0..in_dim {
            let mut acc = 0f32;
            for k in 0..rank {
                acc += up[o * rank + k] * down[k * in_dim + i];
            }
            weight[o * in_dim + i] += scale * acc;
        }
    }
}

/// Conditioning adapter host math: `proj(RMSNorm(x)) + up(silu(down(RMSNorm(x))))`.
pub fn conditioning_adapter_host(
    x: &[f32],
    seq: usize,
    hidden: usize,
    out: usize,
    bottleneck: usize,
    norm: &[f32],
    proj: &[f32],
    down: &[f32],
    up: &[f32],
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), seq * hidden);
    debug_assert_eq!(norm.len(), hidden);
    debug_assert_eq!(proj.len(), out * hidden);
    debug_assert_eq!(down.len(), bottleneck * hidden);
    debug_assert_eq!(up.len(), out * bottleneck);
    let mut y = vec![0f32; seq * out];
    for s in 0..seq {
        let row = &x[s * hidden..(s + 1) * hidden];
        let mut mean_sq = 0f32;
        for &v in row {
            mean_sq += v * v;
        }
        mean_sq /= hidden as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let mut xn = vec![0f32; hidden];
        for i in 0..hidden {
            xn[i] = row[i] * inv * norm[i];
        }
        let mut bottleneck_act = vec![0f32; bottleneck];
        for b in 0..bottleneck {
            let mut acc = 0f32;
            for i in 0..hidden {
                acc += down[b * hidden + i] * xn[i];
            }
            bottleneck_act[b] = acc / (1.0 + (-acc).exp()); // silu
        }
        for o in 0..out {
            let mut direct = 0f32;
            for i in 0..hidden {
                direct += proj[o * hidden + i] * xn[i];
            }
            let mut residual = 0f32;
            for b in 0..bottleneck {
                residual += up[o * bottleneck + b] * bottleneck_act[b];
            }
            y[s * out + o] = direct + residual;
        }
    }
    y
}

/// How a decoder's linear weights rest on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightPrecision {
    /// Whatever `Linear::load` gives: bf16 under bf16 GEMM math, f32 otherwise.
    #[default]
    Native,
    /// Weight-only FP8 (E4M3 codes, one scale per output row): one byte per
    /// parameter, dequantized to bf16 per GEMM. For an encoder that has to stay
    /// resident beside a DiT and does not fit at bf16 (Qwen3-VL-32B: 50 GB).
    Fp8Rows,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    /// SwiGLU: `down(silu(gate(x)) * up(x))` — Qwen, Llama.
    Silu,
    /// GeGLU with the tanh GELU: `down(gelu_tanh(gate(x)) * up(x))` — Gemma.
    GeluTanh,
}

/// What one layer's attention sees: its rotary base and, for local layers, a
/// sliding window. Gemma-3 alternates; Qwen uses one setting throughout.
/// Gemma-4 full-attention layers may override head layout and partial rotary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerAttn {
    pub rope_theta: f64,
    /// Linear position scaling: angles use `position / rope_factor`.
    pub rope_factor: f64,
    /// Attend only to keys with `query - key < window`.
    pub window: Option<usize>,
    /// When set, overrides [`DecoderConfig::heads`] / [`DecoderConfig::head_dim`].
    pub q_heads: Option<usize>,
    pub q_head_dim: Option<usize>,
    pub kv_heads: Option<usize>,
    pub kv_head_dim: Option<usize>,
    /// Fraction of each head width that gets RoPE (`rope_half` table width); the
    /// tail passes through. Gemma-4 full layers use 0.25 on `global_head_dim`.
    pub partial_rotary: Option<f64>,
}

impl LayerAttn {
    pub const fn sliding(rope_theta: f64, window: usize) -> Self {
        Self { rope_theta, rope_factor: 1.0, window: Some(window), q_heads: None, q_head_dim: None, kv_heads: None, kv_head_dim: None, partial_rotary: None }
    }

    pub const fn global(rope_theta: f64, rope_factor: f64) -> Self {
        Self { rope_theta, rope_factor, window: None, q_heads: None, q_head_dim: None, kv_heads: None, kv_head_dim: None, partial_rotary: None }
    }

    /// RoPE table width for a head of `head_dim` channels.
    pub fn rotary_width(&self, head_dim: usize) -> usize {
        match self.partial_rotary {
            Some(f) => {
                let r = (head_dim as f64 * f).round() as usize;
                r.max(2) & !1
            }
            None => head_dim,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Rows of the embedding table. Checked against the checkpoint; it is what
    /// sizes the table when weights are generated rather than read.
    pub vocab: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rms_eps: f32,
    /// Added to every RMSNorm weight at load: 1.0 for Gemma's `x * (1 + w)`.
    pub norm_offset: f32,
    pub act: Act,
    /// Per-head RMSNorm on q and k (weight `[head_dim]`) before the rotary.
    pub qk_norm: bool,
    /// Gemma-3's four norms per layer: the attention and MLP outputs are
    /// normed again before their residual add.
    pub sandwich_norms: bool,
    /// Multiplies the token embeddings: `sqrt(hidden)` for Gemma, 1 otherwise.
    pub embed_scale: f32,
    /// Softmax scale; `head_dim^-0.5` unless the model says otherwise.
    pub attn_scale: f32,
    /// One entry per layer; its length is the layer count.
    pub layers: Vec<LayerAttn>,
    /// Key of the layer list, without the trailing `.N.` (e.g.
    /// `model.language_model.layers`).
    pub layer_prefix: String,
    pub embed_key: String,
    pub final_norm_key: String,
    /// Gemma-4: K and V share one projection; only `k_proj` is loaded.
    pub attention_k_eq_v: bool,
}

impl DecoderConfig {
    pub fn layer_heads(&self, layer: usize) -> usize {
        self.layers[layer].q_heads.unwrap_or(self.heads)
    }

    pub fn layer_head_dim(&self, layer: usize) -> usize {
        self.layers[layer].q_head_dim.unwrap_or(self.head_dim)
    }

    pub fn layer_kv_heads(&self, layer: usize) -> usize {
        self.layers[layer].kv_heads.unwrap_or(self.kv_heads)
    }

    pub fn layer_kv_head_dim(&self, layer: usize) -> usize {
        self.layers[layer].kv_head_dim.unwrap_or(self.head_dim)
    }

    /// The text half of Qwen3-VL-32B (`text_config` of the checkpoint MiniMax-H3
    /// ships). For text-only input the three M-RoPE axes carry the same
    /// position, which makes the interleaved M-RoPE an ordinary rotary.
    pub fn qwen3_vl_32b_text() -> Self {
        let head_dim = 128;
        Self {
            vocab: 151_936,
            hidden: 5120,
            heads: 64,
            kv_heads: 8,
            head_dim,
            intermediate: 25600,
            rms_eps: 1e-6,
            norm_offset: 0.0,
            act: Act::Silu,
            qk_norm: true,
            sandwich_norms: false,
            embed_scale: 1.0,
            attn_scale: (head_dim as f32).powf(-0.5),
            layers: vec![LayerAttn::global(5_000_000.0, 1.0); 64],
            layer_prefix: "model.language_model.layers".into(),
            embed_key: "model.language_model.embed_tokens.weight".into(),
            final_norm_key: "model.language_model.norm.weight".into(),
            attention_k_eq_v: false,
        }
    }

    /// Text half of Qwen3-VL-8B as packaged by SearchingMan's recovered_8b
    /// release: 24 language layers, hidden 4096, keys under `model.layers`
    /// (no `language_model.` prefix). Tap 24 is the un-normed residual after
    /// layer 23; a 4096→5120 conditioning adapter then matches the DiT.
    pub fn qwen3_vl_8b_text() -> Self {
        let head_dim = 128;
        Self {
            vocab: 151_936,
            hidden: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim,
            intermediate: 12288,
            rms_eps: 1e-6,
            norm_offset: 0.0,
            act: Act::Silu,
            qk_norm: true,
            sandwich_norms: false,
            embed_scale: 1.0,
            attn_scale: (head_dim as f32).powf(-0.5),
            layers: vec![LayerAttn::global(5_000_000.0, 1.0); 24],
            layer_prefix: "model.layers".into(),
            embed_key: "model.embed_tokens.weight".into(),
            final_norm_key: "model.norm.weight".into(),
            attention_k_eq_v: false,
        }
    }

    /// The text half of Gemma-3-12B: five sliding-window layers (1024 tokens,
    /// rotary base 1e4) then one global layer (base 1e6, positions / 8).
    pub fn gemma3_12b_text() -> Self {
        let layers = (0..48)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    LayerAttn::global(1_000_000.0, 8.0)
                } else {
                    LayerAttn::sliding(10_000.0, 1024)
                }
            })
            .collect();
        Self {
            vocab: 262_208,
            hidden: 3840,
            heads: 16,
            kv_heads: 8,
            head_dim: 256,
            intermediate: 15360,
            rms_eps: 1e-6,
            norm_offset: 1.0,
            act: Act::GeluTanh,
            qk_norm: true,
            sandwich_norms: true,
            embed_scale: (3840f32).sqrt(),
            // query_pre_attn_scalar = 256.
            attn_scale: (256f32).powf(-0.5),
            layers,
            layer_prefix: "language_model.model.layers".into(),
            embed_key: "language_model.model.embed_tokens.weight".into(),
            final_norm_key: "language_model.model.norm.weight".into(),
            attention_k_eq_v: false,
        }
    }

    /// Gemma-4-12B unified text tower in `Lightricks/LTX-2.5-Diffusers`
    /// (`text_encoder/config.json` → `text_config`). Sliding layers match
    /// Gemma-3; every sixth layer is full attention with `global_head_dim=512`,
    /// one KV head, and partial rotary (0.25). Keys are `model.language_model.*`
    /// in the Diffusers pack (Gemma-3 LTX-2 uses `language_model.model.*`).
    ///
    /// TODO: full-attention layers use `rope_type=proportional` in HF; this path
    /// uses the same inverse-frequency layout as Gemma-3 global layers (θ=1e6,
    /// positions ÷ 8) with partial rotary on the first 128 of 512 channels.
    pub fn gemma4_12b_text() -> Self {
        let global = LayerAttn {
            rope_theta: 1_000_000.0,
            rope_factor: 8.0,
            window: None,
            q_heads: None,
            q_head_dim: Some(512),
            kv_heads: Some(1),
            kv_head_dim: Some(512),
            partial_rotary: Some(0.25),
        };
        let layers = (0..48)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    global
                } else {
                    LayerAttn::sliding(10_000.0, 1024)
                }
            })
            .collect();
        Self {
            vocab: 262_144,
            hidden: 3840,
            heads: 16,
            kv_heads: 8,
            head_dim: 256,
            intermediate: 15360,
            rms_eps: 1e-6,
            norm_offset: 1.0,
            act: Act::GeluTanh,
            qk_norm: true,
            sandwich_norms: true,
            embed_scale: (3840f32).sqrt(),
            attn_scale: (256f32).powf(-0.5),
            layers,
            layer_prefix: "model.language_model.layers".into(),
            embed_key: "model.language_model.embed_tokens.weight".into(),
            final_norm_key: "model.language_model.norm.weight".into(),
            attention_k_eq_v: true,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Match a reference that ran the model in bfloat16. transformers casts
    /// the embedding scale to the weight dtype before multiplying, so a bf16
    /// Gemma-3-12B scales by exactly 62.0 rather than sqrt(3840) = 61.9677 — a
    /// systematic 5e-4 that a per-tap gate is too coarse to see but that is
    /// simply the wrong constant against such a reference. A float32 reference
    /// wants the config as built.
    pub fn for_bf16_reference(mut self) -> Self {
        self.embed_scale = half::bf16::from_f32(self.embed_scale).to_f32();
        self
    }
}

/// `weight + offset`, on the device. Gemma stores `w` and computes `1 + w`.
fn norm_weight(map: &WeightMap, key: &str, width: usize, offset: f32) -> Result<CudaTensor> {
    norm_from(cuda_tensor_shaped(map, key, &[width])?, offset)
}

fn norm_from(w: CudaTensor, offset: f32) -> Result<CudaTensor> {
    let mut w = if offset == 0.0 { w } else { w.try_add_scalar(offset)? };
    w.pin_device()?;
    Ok(w)
}

/// The seven projections of a layer: key suffix, input width, output width.
fn linear_specs(cfg: &DecoderConfig, layer: usize) -> [(&'static str, usize, usize); 7] {
    let h = cfg.hidden;
    let hq = cfg.layer_heads(layer);
    let hkv = cfg.layer_kv_heads(layer);
    let dq = cfg.layer_head_dim(layer);
    let dkv = cfg.layer_kv_head_dim(layer);
    [
        ("self_attn.q_proj", h, hq * dq),
        ("self_attn.k_proj", h, hkv * dkv),
        ("self_attn.v_proj", h, hkv * dkv),
        ("self_attn.o_proj", hq * dq, h),
        ("mlp.gate_proj", h, cfg.intermediate),
        ("mlp.up_proj", h, cfg.intermediate),
        ("mlp.down_proj", cfg.intermediate, h),
    ]
}

/// Largest linear weight block in any layer (for prefetch pinned sizing).
pub(crate) fn max_layer_linear_elems(cfg: &DecoderConfig) -> usize {
    (0..cfg.num_layers()).map(|i| linear_specs(cfg, i).iter().map(|(_, i, o)| i * o).sum()).max().unwrap_or(0)
}

/// The norm weights of a layer: key suffix (without `.weight`) and width.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn norm_specs(cfg: &DecoderConfig, layer: usize) -> Vec<(&'static str, usize)> {
    let h = cfg.hidden;
    let dq = cfg.layer_head_dim(layer);
    let dkv = cfg.layer_kv_head_dim(layer);
    let mut v = vec![("input_layernorm", h), ("post_attention_layernorm", h)];
    if cfg.sandwich_norms {
        v.push(("pre_feedforward_layernorm", h));
        v.push(("post_feedforward_layernorm", h));
    }
    if cfg.qk_norm {
        v.push(("self_attn.q_norm", dq));
        v.push(("self_attn.k_norm", dkv));
    }
    v
}

pub(crate) struct Layer {
    q: Linear,
    k: Linear,
    v: Option<Linear>,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    q_norm: Option<CudaTensor>,
    k_norm: Option<CudaTensor>,
    /// Before attention.
    norm_attn_in: CudaTensor,
    /// After attention, before its residual add (sandwich norms only).
    norm_attn_out: Option<CudaTensor>,
    /// Before the MLP.
    norm_mlp_in: CudaTensor,
    /// After the MLP, before its residual add (sandwich norms only).
    norm_mlp_out: Option<CudaTensor>,
}

impl Layer {
    fn load(map: &WeightMap, cfg: &DecoderConfig, index: usize, precision: WeightPrecision) -> Result<Self> {
        let p = format!("{}.{index}", cfg.layer_prefix);
        Self::assemble(
            cfg,
            index,
            &mut |name, i, o| match precision {
                WeightPrecision::Native => Linear::load(map, &format!("{p}.{name}"), i, o, false),
                WeightPrecision::Fp8Rows => Linear::load_fp8_rows(map, &format!("{p}.{name}"), i, o, false),
            },
            &mut |name, width| norm_weight(map, &format!("{p}.{name}.weight"), width, cfg.norm_offset),
        )
    }

    /// One layer from wherever its parts come from: the checkpoint directly, or
    /// a layer the prefetcher already staged. Both go through here so the two
    /// cannot disagree on which key feeds which slot.
    pub(crate) fn assemble(
        cfg: &DecoderConfig,
        layer: usize,
        lin: &mut dyn FnMut(&str, usize, usize) -> Result<Linear>,
        norm: &mut dyn FnMut(&str, usize) -> Result<CudaTensor>,
    ) -> Result<Self> {
        let h = cfg.hidden;
        let [q, k, v, o, gate, up, down] = linear_specs(cfg, layer);
        // The two families name the pre-MLP norm differently: in a sandwich
        // layer `post_attention_layernorm` really is after attention and the
        // pre-MLP norm is `pre_feedforward_layernorm`.
        let (attn_out, mlp_in, mlp_out) = if cfg.sandwich_norms {
            (
                Some(norm("post_attention_layernorm", h)?),
                norm("pre_feedforward_layernorm", h)?,
                Some(norm("post_feedforward_layernorm", h)?),
            )
        } else {
            (None, norm("post_attention_layernorm", h)?, None)
        };
        let dq = cfg.layer_head_dim(layer);
        let dkv = cfg.layer_kv_head_dim(layer);
        Ok(Self {
            q: lin(q.0, q.1, q.2)?,
            k: lin(k.0, k.1, k.2)?,
            v: if cfg.attention_k_eq_v { None } else { Some(lin(v.0, v.1, v.2)?) },
            o: lin(o.0, o.1, o.2)?,
            gate: lin(gate.0, gate.1, gate.2)?,
            up: lin(up.0, up.1, up.2)?,
            down: lin(down.0, down.1, down.2)?,
            q_norm: if cfg.qk_norm { Some(norm("self_attn.q_norm", dq)?) } else { None },
            k_norm: if cfg.qk_norm { Some(norm("self_attn.k_norm", dkv)?) } else { None },
            norm_attn_in: norm("input_layernorm", h)?,
            norm_attn_out: attn_out,
            norm_mlp_in: mlp_in,
            norm_mlp_out: mlp_out,
        })
    }

    /// `x`: `[1, S, hidden]`. `cos`/`sin`: `[S, R]` with `R` the rotary width.
    /// `mask`: `[1, 1, S, S]`.
    fn forward(
        &self,
        cfg: &DecoderConfig,
        layer: usize,
        x: &CudaTensor,
        cos: &CudaTensor,
        sin: &CudaTensor,
        mask: &CudaTensor,
    ) -> Result<CudaTensor> {
        let s = x.shape[1];
        let hq = cfg.layer_heads(layer);
        let hkv = cfg.layer_kv_heads(layer);
        let dq = cfg.layer_head_dim(layer);
        let dkv = cfg.layer_kv_head_dim(layer);
        if hq % hkv != 0 {
            return Err(msg(format!("llm layer {layer}: {hq} query heads over {hkv} kv heads")));
        }

        let h = x.rms_norm(&self.norm_attn_in, cfg.rms_eps)?;
        // [1, S, H*D] -> [1, S, H, D]: the per-head norm is an RMSNorm over D.
        let split = |t: CudaTensor, heads: usize, d: usize, norm: &Option<CudaTensor>| -> Result<CudaTensor> {
            let t = t.reshape(vec![1, s, heads, d])?;
            let t = match norm {
                Some(w) => t.rms_norm(w, cfg.rms_eps)?,
                None => t,
            };
            t.transpose(1, 2)
        };
        let q = split(self.q.forward(&h)?, hq, dq, &self.q_norm)?.rope_half(cos, sin)?;
        let k_h = self.k.forward(&h)?;
        let v_h = if let Some(v) = &self.v { v.forward(&h)? } else { k_h.clone() };
        let k = split(k_h, hkv, dkv, &self.k_norm)?.rope_half(cos, sin)?;
        let v = split(v_h, hkv, dkv, &None)?;
        let (k, v) = (k.repeat_kv(hq / hkv)?, v.repeat_kv(hq / hkv)?);
        let a = scaled_dot_product_attention_masked(&q, &k, &v, Some(cfg.attn_scale), Some(mask))?;
        let a = self.o.forward(&a.transpose(1, 2)?.reshape(vec![1, s, hq * dq])?)?;
        let a = match &self.norm_attn_out {
            Some(w) => a.rms_norm(w, cfg.rms_eps)?,
            None => a,
        };
        let x = x.add(&a)?;

        let h = x.rms_norm(&self.norm_mlp_in, cfg.rms_eps)?;
        let g = self.gate.forward(&h)?;
        let g = match cfg.act {
            Act::Silu => g.silu(),
            Act::GeluTanh => g.gelu_tanh(),
        };
        let m = self.down.forward(&g.mul(&self.up.forward(&h)?)?)?;
        let m = match &self.norm_mlp_out {
            Some(w) => m.rms_norm(w, cfg.rms_eps)?,
            None => m,
        };
        x.add(&m)
    }
}

/// `[S, R]` cos/sin in the rotate_half layout (`cat(freqs, freqs)`),
/// computed in f64 like the references do before casting. `head_dim` is the
/// per-head channel width; `R` is the rotary table width (full head or partial).
fn rope_tables(positions: &[u32], head_dim: usize, la: &LayerAttn) -> Result<(CudaTensor, CudaTensor)> {
    let r = la.rotary_width(head_dim);
    let half = r / 2;
    let s = positions.len();
    let (mut cos, mut sin) = (vec![0f32; s * r], vec![0f32; s * r]);
    for (p, &pos) in positions.iter().enumerate() {
        for k in 0..half {
            let inv = la.rope_theta.powf(-((2 * k) as f64) / head_dim as f64);
            let ang = f64::from(pos) / la.rope_factor * inv;
            for j in [k, k + half] {
                cos[p * r + j] = ang.cos() as f32;
                sin[p * r + j] = ang.sin() as f32;
            }
        }
    }
    let mut c = CudaTensor::from_vec(cos, vec![s, r])?;
    let mut sn = CudaTensor::from_vec(sin, vec![s, r])?;
    c.pin_device()?;
    sn.pin_device()?;
    Ok((c, sn))
}

/// Qwen3-VL interleaved mRoPE: three position axes + `mrope_section` frequency
/// interleave (`apply_interleaved_mrope` in transformers). `positions[s] =
/// [temporal, height, width]`.
fn rope_tables_mrope(
    positions: &[[f64; 3]],
    mrope_section: [usize; 3],
    head_dim: usize,
    la: &LayerAttn,
) -> Result<(CudaTensor, CudaTensor)> {
    let r = la.rotary_width(head_dim);
    let half = r / 2;
    if mrope_section.iter().sum::<usize>() != half {
        return Err(msg(format!(
            "llm mRoPE: section {:?} sums to {}, need head_dim/2={half}",
            mrope_section, mrope_section.iter().sum::<usize>()
        )));
    }
    let s = positions.len();
    let (mut cos, mut sin) = (vec![0f32; s * r], vec![0f32; s * r]);
    for (p, pos) in positions.iter().enumerate() {
        let mut axis_freq = vec![[0f64; 3]; half];
        for k in 0..half {
            let inv = la.rope_theta.powf(-((2 * k) as f64) / head_dim as f64) / la.rope_factor;
            for a in 0..3 {
                axis_freq[k][a] = pos[a] * inv;
            }
        }
        // Interleave: start from temporal, overwrite H/W slots.
        let mut interleaved = vec![0f64; half];
        for k in 0..half {
            interleaved[k] = axis_freq[k][0];
        }
        for (dim, offset) in [(1usize, 1usize), (2, 2)] {
            let length = mrope_section[dim] * 3;
            let mut idx = offset;
            while idx < length && idx < half {
                interleaved[idx] = axis_freq[idx][dim];
                idx += 3;
            }
        }
        for k in 0..half {
            let ang = interleaved[k];
            for j in [k, k + half] {
                cos[p * r + j] = ang.cos() as f32;
                sin[p * r + j] = ang.sin() as f32;
            }
        }
    }
    let mut c = CudaTensor::from_vec(cos, vec![s, r])?;
    let mut sn = CudaTensor::from_vec(sin, vec![s, r])?;
    c.pin_device()?;
    sn.pin_device()?;
    Ok((c, sn))
}

/// Optional vision inject for Qwen3-VL DeepStack + mRoPE.
pub struct MultimodalCtx {
    /// Per-token `[t, h, w]` rotary positions (length = sequence).
    pub mrope_positions: Vec<[f64; 3]>,
    pub mrope_section: [usize; 3],
    /// True at image/video pad tokens (vision features were scattered here).
    pub visual_mask: Vec<bool>,
    /// DeepStack features for the first `deepstack.len()` LM layers: each
    /// `[n_visual, hidden]` matching `visual_mask` true count in order.
    pub deepstack: Vec<CudaTensor>,
}

fn inject_deepstack(x: &CudaTensor, visual_mask: &[bool], deep: &CudaTensor) -> Result<CudaTensor> {
    let [b, s, h] = match x.shape[..] {
        [b, s, h] => [b, s, h],
        _ => return Err(msg(format!("deepstack expects [B,S,H], got {:?}", x.shape))),
    };
    if b != 1 || visual_mask.len() != s {
        return Err(msg(format!(
            "deepstack: batch={b} mask={} seq={s}",
            visual_mask.len()
        )));
    }
    let n_vis = visual_mask.iter().filter(|&&m| m).count();
    if deep.shape != [n_vis, h] {
        return Err(msg(format!(
            "deepstack features {:?} vs {n_vis} visual × hidden {h}",
            deep.shape
        )));
    }
    if n_vis == 0 {
        return Ok(x.clone());
    }
    let mut host = x.host_cow()?.into_owned();
    let deep_h = deep.host_cow()?;
    let mut vi = 0usize;
    for (si, &is_vis) in visual_mask.iter().enumerate() {
        if !is_vis {
            continue;
        }
        let base = si * h;
        let db = vi * h;
        for d in 0..h {
            host[base + d] += deep_h[db + d];
        }
        vi += 1;
    }
    CudaTensor::from_vec(host, vec![1, s, h])?.to_device()
}

/// Additive `[1, 1, S, S]` mask: causal, keys limited to `attend`, and to the
/// last `window` positions when the layer is local.
fn attn_mask(attend: &[bool], window: Option<usize>) -> Result<CudaTensor> {
    let s = attend.len();
    let mut m = vec![f32::MIN; s * s];
    for i in 0..s {
        for j in 0..=i {
            if attend[j] && window.is_none_or(|w| i - j < w) {
                m[i * s + j] = 0.0;
            }
        }
        // A padded query row has no valid key under a left-padded prompt;
        // let it see itself so its softmax is finite. Its output is never read.
        if m[i * s..i * s + s].iter().all(|&v| v != 0.0) {
            m[i * s + i] = 0.0;
        }
    }
    let mut t = CudaTensor::from_vec(m, vec![1, 1, s, s])?;
    t.pin_device()?;
    Ok(t)
}

fn embed(map: &WeightMap, cfg: &DecoderConfig, ids: &[u32]) -> Result<CudaTensor> {
    let rows: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
    let x = match map.lazy() {
        Some(lazy) => {
            let (width, values) = lazy.rows_f32(&cfg.embed_key, &rows).map_err(|e| msg(e.to_string()))?;
            if width != cfg.hidden {
                return Err(msg(format!("{}: width {width} != hidden {}", cfg.embed_key, cfg.hidden)));
            }
            CudaTensor::from_vec(values, vec![rows.len(), width])?.to_device()?
        }
        None => {
            cuda_tensor_shaped(map, &cfg.embed_key, &[cfg.vocab, cfg.hidden])?.embedding_rows(&rows)?
        }
    };
    let x = x.reshape(vec![1, ids.len(), cfg.hidden])?;
    if cfg.embed_scale == 1.0 {
        Ok(x)
    } else {
        x.try_mul_scalar(cfg.embed_scale)
    }
}

/// Where the layer loop gets its weights: read from the checkpoint and dropped
/// per layer (streamed), or already on the device (resident).
trait LayerSource {
    fn with_layer<R>(&mut self, index: usize, f: impl FnOnce(&Layer) -> Result<R>) -> Result<R>;
    fn final_norm(&mut self) -> Result<CudaTensor>;
}

struct Streamed<'a> {
    map: &'a WeightMap,
    cfg: &'a DecoderConfig,
}

impl LayerSource for Streamed<'_> {
    fn with_layer<R>(&mut self, index: usize, f: impl FnOnce(&Layer) -> Result<R>) -> Result<R> {
        // Loaded here and dropped on return: one layer on the device at a time.
        let layer = Layer::load(self.map, self.cfg, index, WeightPrecision::Native)?;
        f(&layer)
    }

    fn final_norm(&mut self) -> Result<CudaTensor> {
        norm_weight(self.map, &self.cfg.final_norm_key, self.cfg.hidden, self.cfg.norm_offset)
    }
}

/// The one layer loop both modes run, so a resident encoder cannot drift from
/// the streamed one that the parity gates judged.
fn encode<S: LayerSource>(
    cfg: &DecoderConfig,
    source: &mut S,
    embedded: CudaTensor,
    positions: &[u32],
    attend: &[bool],
    taps: &[usize],
    progress: bool,
    multimodal: Option<&MultimodalCtx>,
) -> Result<Vec<CudaTensor>> {
    let s = embedded.shape[1];
    if s == 0 || attend.len() != s {
        return Err(msg(format!("llm: {s} ids, {} attend flags", attend.len())));
    }
    if multimodal.is_none() && positions.len() != s {
        return Err(msg(format!("llm: {s} ids, {} positions", positions.len())));
    }
    if let Some(mm) = multimodal {
        if mm.mrope_positions.len() != s || mm.visual_mask.len() != s {
            return Err(msg(format!(
                "llm multimodal: seq={s} mrope={} mask={}",
                mm.mrope_positions.len(),
                mm.visual_mask.len()
            )));
        }
    }
    let n = cfg.num_layers();
    let last = *taps.iter().max().ok_or_else(|| msg("llm: no taps requested"))?;
    if last > n {
        return Err(msg(format!("llm: tap {last} of a {n}-layer model")));
    }

    let mut out: Vec<Option<CudaTensor>> = vec![None; taps.len()];
    let mut keep = |k: usize, x: &CudaTensor| {
        for (slot, &t) in out.iter_mut().zip(taps) {
            if t == k {
                *slot = Some(x.clone());
            }
        }
    };

    let mut x = embedded;
    // Rotary tables and masks are shared by every layer with the same settings
    // (one kind for Qwen, two for Gemma-3), so build each once.
    let mut ropes: Vec<(LayerAttn, usize, (CudaTensor, CudaTensor))> = Vec::new();
    let mut masks: Vec<(Option<usize>, CudaTensor)> = Vec::new();
    for (i, la) in cfg.layers.iter().enumerate().take(last) {
        keep(i, &x);
        let hd = cfg.layer_head_dim(i);
        if !ropes.iter().any(|(k, h, _)| k == la && *h == hd) {
            let tables = if let Some(mm) = multimodal {
                rope_tables_mrope(&mm.mrope_positions, mm.mrope_section, hd, la)?
            } else {
                rope_tables(positions, hd, la)?
            };
            ropes.push((*la, hd, tables));
        }
        if !masks.iter().any(|(w, _)| *w == la.window) {
            masks.push((la.window, attn_mask(attend, la.window)?));
        }
        let (cos, sin) = &ropes.iter().find(|(k, h, _)| k == la && *h == hd).expect("just inserted").2;
        let mask = &masks.iter().find(|(w, _)| *w == la.window).expect("just inserted").1;
        x = source.with_layer(i, |layer| layer.forward(cfg, i, &x, cos, sin, mask))?;
        if let Some(mm) = multimodal {
            if let Some(deep) = mm.deepstack.get(i) {
                x = inject_deepstack(&x, &mm.visual_mask, deep)?;
            }
        }
        if progress {
            crate::wan::log::info(format_args!("llm layer {}/{last}", i + 1));
        }
    }
    if last == n {
        // Tap `num_layers` is normally post-final-norm (HF). SearchingMan's
        // recovered-8B tap-24 is the *un-normed* residual after the last
        // block — `load_with_comfy_lora` leaves final_norm unloaded so we
        // keep x as-is when the norm is not resident.
        match source.final_norm() {
            Ok(norm) => x = x.rms_norm(&norm, cfg.rms_eps)?,
            Err(_) => {}
        }
    }
    keep(last, &x);
    out.into_iter()
        .map(|t| t.ok_or_else(|| msg("llm: a tap was not reached")))
        .collect()
}

/// Hidden states of one prompt, in Hugging Face's `output_hidden_states`
/// numbering: tap `0` is the (scaled) embeddings, tap `k` the output of layer
/// `k`, and tap `num_layers` is that last output *after the final norm* — the
/// one entry of the tuple HF norms. Layers past the largest tap are not read.
///
/// `positions` are the rotary positions, given explicitly because references
/// disagree: transformers numbers a left-padded prompt `0..S` across the
/// padding (real tokens end up at `S-n..S-1`), other stacks restart at the
/// first real token. Pass what the reference used. `attend[j]` says whether
/// position `j` may be used as a key (false for padding).
///
/// Streams: one layer's weights on the device at a time. For a process that
/// encodes many prompts and has the memory, see [`ResidentDecoder`].
///
/// Returns one `[1, S, hidden]` tensor per tap, in the order asked.
pub fn hidden_states(
    map: &WeightMap,
    cfg: &DecoderConfig,
    ids: &[u32],
    positions: &[u32],
    attend: &[bool],
    taps: &[usize],
) -> Result<Vec<CudaTensor>> {
    hidden_states_opt(map, cfg, ids, positions, attend, taps, true, true, None)
}

/// Like [`hidden_states`], but starts from pre-built embeddings (vision pads
/// already scattered) and applies Qwen3-VL mRoPE + DeepStack.
pub fn hidden_states_multimodal(
    map: &WeightMap,
    cfg: &DecoderConfig,
    embedded: CudaTensor,
    attend: &[bool],
    taps: &[usize],
    multimodal: &MultimodalCtx,
) -> Result<Vec<CudaTensor>> {
    let s = embedded.shape.get(1).copied().unwrap_or(0);
    if s == 0 {
        return Err(msg("llm: empty multimodal prompt"));
    }
    let positions: Vec<u32> = (0..s as u32).collect();
    #[cfg(feature = "cuda")]
    if let Some(stage) = map.lazy().and_then(|lazy| prefetch::Stage::new(lazy, cfg)) {
        let last = taps.iter().copied().max().unwrap_or(0).min(cfg.num_layers());
        return stage.run(map, cfg, last, |source| {
            encode(cfg, source, embedded, &positions, attend, taps, true, Some(multimodal))
        });
    }
    encode(
        cfg,
        &mut Streamed { map, cfg },
        embedded,
        &positions,
        attend,
        taps,
        true,
        Some(multimodal),
    )
}

/// Scatter token embedding rows, then replace pad positions with vision features.
pub fn embed_with_vision(
    map: &WeightMap,
    cfg: &DecoderConfig,
    ids: &[u32],
    image_token_id: u32,
    image_features: Option<&CudaTensor>,
    video_token_id: u32,
    video_features: Option<&CudaTensor>,
) -> Result<(CudaTensor, Vec<bool>, Vec<bool>)> {
    let mut embedded = embed(map, cfg, ids)?;
    let mut image_mask = vec![false; ids.len()];
    let mut video_mask = vec![false; ids.len()];
    for (i, &id) in ids.iter().enumerate() {
        if id == image_token_id {
            image_mask[i] = true;
        } else if id == video_token_id {
            video_mask[i] = true;
        }
    }
    if let Some(feat) = image_features {
        embedded = scatter_visual_embeds(&embedded, &image_mask, feat, "image")?;
    } else if image_mask.iter().any(|&m| m) {
        return Err(msg("llm: image pad tokens present but no image features"));
    }
    if let Some(feat) = video_features {
        embedded = scatter_visual_embeds(&embedded, &video_mask, feat, "video")?;
    } else if video_mask.iter().any(|&m| m) {
        return Err(msg("llm: video pad tokens present but no video features"));
    }
    Ok((embedded, image_mask, video_mask))
}

fn scatter_visual_embeds(
    embedded: &CudaTensor,
    mask: &[bool],
    features: &CudaTensor,
    label: &str,
) -> Result<CudaTensor> {
    let [b, s, h] = match embedded.shape[..] {
        [b, s, h] => [b, s, h],
        _ => return Err(msg(format!("embed scatter expects [B,S,H], got {:?}", embedded.shape))),
    };
    if b != 1 || mask.len() != s {
        return Err(msg(format!("{label} scatter: bad mask/seq")));
    }
    let n = mask.iter().filter(|&&m| m).count();
    if features.shape != [n, h] {
        return Err(msg(format!(
            "Qwen3-VL {label} features and placeholder tokens do not match: tokens={n}, features={:?}",
            features.shape
        )));
    }
    if n == 0 {
        return Ok(embedded.clone());
    }
    let mut host = embedded.host_cow()?.into_owned();
    let feat = features.host_cow()?;
    let mut vi = 0usize;
    for (si, &is_pad) in mask.iter().enumerate() {
        if !is_pad {
            continue;
        }
        let base = si * h;
        let fb = vi * h;
        host[base..base + h].copy_from_slice(&feat[fb..fb + h]);
        vi += 1;
    }
    CudaTensor::from_vec(host, vec![1, s, h])?.to_device()
}

#[allow(clippy::too_many_arguments)]
fn hidden_states_opt(
    map: &WeightMap,
    cfg: &DecoderConfig,
    ids: &[u32],
    positions: &[u32],
    attend: &[bool],
    taps: &[usize],
    allow_prefetch: bool,
    progress: bool,
    multimodal: Option<&MultimodalCtx>,
) -> Result<Vec<CudaTensor>> {
    if ids.is_empty() {
        return Err(msg("llm: empty prompt"));
    }
    let embedded = embed(map, cfg, ids)?;
    // With a mapped checkpoint and bf16 linears, the next layer is read and
    // uploaded (pinned memory, its own stream) while this one computes.
    #[cfg(feature = "cuda")]
    if let Some(stage) = map.lazy().filter(|_| allow_prefetch).and_then(|lazy| prefetch::Stage::new(lazy, cfg)) {
        let last = taps.iter().copied().max().unwrap_or(0).min(cfg.num_layers());
        return stage.run(map, cfg, last, |source| {
            encode(cfg, source, embedded, positions, attend, taps, progress, multimodal)
        });
    }
    let _ = allow_prefetch;
    encode(
        cfg,
        &mut Streamed { map, cfg },
        embedded,
        positions,
        attend,
        taps,
        progress,
        multimodal,
    )
}

/// Result of [`prefetch_self_check`].
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct PrefetchCheck {
    /// Dtype the checkpoint was stored in.
    pub stored: &'static str,
    /// Largest |prefetched - plain| over every tap. Must be exactly zero.
    pub max_abs_diff: f32,
    pub elements: usize,
}

/// The device check for the prefetcher: a small sandwich-norm, qk-norm decoder
/// (every norm slot in use) is written to `dir` as a real safetensors file,
/// opened lazily, and encoded twice — layers staged ahead through pinned memory
/// on the copy stream, and the plain one-layer-at-a-time path. The two must
/// agree bit for bit. Run once per stored dtype (`BF16` is copied, `F32` is
/// rounded on the way into the pinned buffer). Errors if prefetching is
/// unavailable (no device, or bf16 linears off) rather than comparing the plain
/// path with itself.
#[cfg(feature = "cuda")]
pub fn prefetch_self_check(dir: &std::path::Path, stored_bf16: bool) -> Result<PrefetchCheck> {
    use fastvideo_loader::{LazyDType, SafetensorsWriter, TensorSpec};
    let mut cfg = DecoderConfig::gemma3_12b_text();
    (cfg.vocab, cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.intermediate) = (48, 96, 4, 2, 24, 256);
    cfg.layers.truncate(9);
    let bf16_file = stored_bf16;
    let (stored, label) = if bf16_file { (LazyDType::BF16, "BF16") } else { (LazyDType::F32, "F32") };

    let mut tensors: Vec<(String, Vec<usize>)> = vec![(cfg.embed_key.clone(), vec![cfg.vocab, cfg.hidden])];
    for l in 0..cfg.num_layers() {
        let p = format!("{}.{l}", cfg.layer_prefix);
        tensors.extend(linear_specs(&cfg, l).iter().map(|(n, i, o)| (format!("{p}.{n}.weight"), vec![*o, *i])));
        tensors.extend(norm_specs(&cfg, l).iter().map(|(n, w)| (format!("{p}.{n}.weight"), vec![*w])));
    }
    tensors.push((cfg.final_norm_key.clone(), vec![cfg.hidden]));

    std::fs::create_dir_all(dir).map_err(|e| msg(e.to_string()))?;
    let path = dir.join(format!("prefetch-check-{label}.safetensors"));
    let specs: Vec<TensorSpec> = tensors.iter().map(|(k, s)| TensorSpec::new(k.clone(), stored.clone(), s.clone())).collect();
    let mut w = SafetensorsWriter::create(&path, &specs, &[]).map_err(|e| msg(e.to_string()))?;
    for (key, shape) in &tensors {
        let seed = key.bytes().fold(7u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
        let is_norm = key.contains("norm");
        let mut bytes = Vec::new();
        for i in 0..shape.iter().product::<usize>() {
            let v = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
            let v = if is_norm { 0.5 + v } else { (v - 0.5) * 0.2 };
            if bf16_file {
                bytes.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
            } else {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        w.write(key, &bytes).map_err(|e| msg(e.to_string()))?;
    }
    w.finish().map_err(|e| msg(e.to_string()))?;

    let map = WeightMap::open_files(std::slice::from_ref(&path))?;
    let lazy = map.lazy().ok_or_else(|| msg("prefetch self-check: the map is not lazy"))?;
    if prefetch::Stage::new(lazy, &cfg).is_none() {
        return Err(msg("prefetch self-check: prefetching is unavailable here (bf16 linears off, or no pinned memory)"));
    }
    let ids: Vec<u32> = (0..37u32).map(|i| (i * 7 + 3) % cfg.vocab as u32).collect();
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    let attend = vec![true; ids.len()];
    let taps: Vec<usize> = (0..=cfg.num_layers()).collect();
    let ahead = hidden_states_opt(&map, &cfg, &ids, &positions, &attend, &taps, true, false, None)?;
    let plain = hidden_states_opt(&map, &cfg, &ids, &positions, &attend, &taps, false, false, None)?;
    let (mut worst, mut elements) = (0f32, 0usize);
    for (a, b) in ahead.iter().zip(&plain) {
        let (a, b) = (a.host_cow()?, b.host_cow()?);
        elements += a.len();
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            // A NaN on either side must fail, not vanish in a `max`.
            worst = if d.is_nan() { f32::INFINITY } else { worst.max(d) };
        }
    }
    let _ = std::fs::remove_file(&path);
    Ok(PrefetchCheck { stored: label, max_abs_diff: worst, elements })
}

#[cfg(feature = "cuda")]
mod prefetch;

/// The embedding table, held on the host in whatever dtype the checkpoint
/// stores: a prompt needs a few hundred of its 150k-260k rows, and 2-4 GB of
/// device memory is better spent on the DiT sitting next to this encoder.
enum EmbedTable {
    F32(Vec<f32>),
    Bf16(Vec<u8>),
    F16(Vec<u8>),
}

impl EmbedTable {
    fn load(map: &WeightMap, cfg: &DecoderConfig) -> Result<Self> {
        use fastvideo_loader::LazyDType;
        if let Some(lazy) = map.lazy() {
            let v = lazy.view(&cfg.embed_key).map_err(|e| msg(e.to_string()))?;
            if v.shape != [cfg.vocab, cfg.hidden] {
                return Err(msg(format!("{}: shape {:?}, expected [{}, {}]", cfg.embed_key, v.shape, cfg.vocab, cfg.hidden)));
            }
            return match v.dtype {
                LazyDType::BF16 => Ok(Self::Bf16(v.bytes.to_vec())),
                LazyDType::F16 => Ok(Self::F16(v.bytes.to_vec())),
                LazyDType::F32 => Ok(Self::F32(lazy.to_f32(&cfg.embed_key).map_err(|e| msg(e.to_string()))?.1)),
                other => Err(msg(format!("{}: {other:?} embedding table", cfg.embed_key))),
            };
        }
        let t = cuda_tensor_shaped(map, &cfg.embed_key, &[cfg.vocab, cfg.hidden])?;
        Ok(Self::F32(t.host_cow()?.into_owned()))
    }

    fn rows(&self, ids: &[u32], hidden: usize) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(ids.len() * hidden);
        for &id in ids {
            let r = id as usize;
            match self {
                Self::F32(t) => {
                    let row = t.get(r * hidden..(r + 1) * hidden).ok_or_else(|| msg(format!("llm: token id {id} past the embedding table")))?;
                    out.extend_from_slice(row);
                }
                Self::Bf16(b) | Self::F16(b) => {
                    let row = b.get(r * hidden * 2..(r + 1) * hidden * 2).ok_or_else(|| msg(format!("llm: token id {id} past the embedding table")))?;
                    let half_is_bf16 = matches!(self, Self::Bf16(_));
                    out.extend(row.chunks_exact(2).map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        if half_is_bf16 { half::bf16::from_bits(bits).to_f32() } else { half::f16::from_bits(bits).to_f32() }
                    }));
                }
            }
        }
        Ok(out)
    }

    fn host_bytes(&self) -> u64 {
        match self {
            Self::F32(t) => (t.len() * 4) as u64,
            Self::Bf16(b) | Self::F16(b) => b.len() as u64,
        }
    }
}

/// A decoder whose layers stay on the device, for a process that encodes
/// prompt after prompt next to a resident DiT.
///
/// Streaming is the right default — the encoder is needed for milliseconds of
/// compute and costs tens of GB — but it makes every new prompt pay for the
/// whole checkpoint again: 10.6 s for Qwen3-VL-32B (50 GB read), 15.7 s for
/// Gemma-3-12B (47 GB of float32 read and narrowed). Where the card has room
/// (Gemma at bf16 is 23.5 GB beside a 38 GB LTX-2 DiT on 96 GB) the same forward
/// takes a fraction of a second once the weights are simply left in place.
///
/// Same layer loop as [`hidden_states`], so the numbers are the same numbers.
pub struct ResidentDecoder {
    cfg: DecoderConfig,
    layers: Vec<Layer>,
    final_norm: Option<CudaTensor>,
    embed: EmbedTable,
    precision: WeightPrecision,
}

struct Resident<'a>(&'a ResidentDecoder);

impl LayerSource for Resident<'_> {
    fn with_layer<R>(&mut self, index: usize, f: impl FnOnce(&Layer) -> Result<R>) -> Result<R> {
        let layer = self.0.layers.get(index).ok_or_else(|| {
            msg(format!("llm: layer {index} is not resident (loaded {})", self.0.layers.len()))
        })?;
        f(layer)
    }

    fn final_norm(&mut self) -> Result<CudaTensor> {
        self.0.final_norm.clone().ok_or_else(|| msg("llm: the final norm is not resident (load every layer to tap the last state)"))
    }
}

impl ResidentDecoder {
    /// Load layers `0..layers` and keep them on the device. Taps up to `layers`
    /// are then available; pass `cfg.num_layers()` to include the final norm
    /// (the last tap is post-norm). Reads either checkpoint layout — the
    /// original shards or a slim rewrite — since both go through `Linear::load`.
    pub fn load(map: &WeightMap, cfg: &DecoderConfig, layers: usize) -> Result<Self> {
        Self::load_with(map, cfg, layers, WeightPrecision::Native)
    }

    /// [`Self::load_with`] merging Comfy-style LoRA adapters (`lora_down` /
    /// `lora_up`, scale = `alpha / rank`) into the listed layers before the
    /// linear is pinned. SearchingMan ARA uses this shape on layers 16..=23.
    pub fn load_with_comfy_lora(
        map: &WeightMap,
        ara: &WeightMap,
        cfg: &DecoderConfig,
        layers: usize,
        lora_layers: &[usize],
        suffixes: &[&str],
        alpha: f32,
        rank: usize,
    ) -> Result<Self> {
        let n = cfg.num_layers();
        if layers == 0 || layers > n {
            return Err(msg(format!("llm: {layers} resident layers of a {n}-layer model")));
        }
        let scale = alpha / rank as f32;
        let mut loaded = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("{}.{i}", cfg.layer_prefix);
            let apply = lora_layers.contains(&i);
            let layer = Layer::assemble(
                cfg,
                i,
                &mut |name, in_dim, out_dim| {
                    let key = format!("{p}.{name}");
                    let wt = cuda_tensor_shaped(map, &format!("{key}.weight"), &[out_dim, in_dim])?;
                    let mut w = wt.host_cow()?.into_owned();
                    if apply && suffixes.iter().any(|s| *s == name) {
                        let base = format!("layers.{i}.{name}");
                        merge_comfy_lora_into(&mut w, ara, &base, out_dim, in_dim, scale)?;
                    }
                    Linear::from_tensors(CudaTensor::from_vec(w, vec![out_dim, in_dim])?, None)
                },
                &mut |name, width| norm_weight(map, &format!("{p}.{name}.weight"), width, cfg.norm_offset),
            )?;
            loaded.push(layer);
            crate::wan::log::info(format_args!("llm resident layer {}/{layers}", i + 1));
        }
        Ok(Self {
            cfg: cfg.clone(),
            layers: loaded,
            final_norm: None,
            embed: EmbedTable::load(map, cfg)?,
            precision: WeightPrecision::Native,
        })
    }

    /// [`Self::load`] with the linear weights held at `precision`. Norms, the
    /// embedding table and all activations are unaffected.
    pub fn load_with(map: &WeightMap, cfg: &DecoderConfig, layers: usize, precision: WeightPrecision) -> Result<Self> {
        let n = cfg.num_layers();
        if layers == 0 || layers > n {
            return Err(msg(format!("llm: {layers} resident layers of a {n}-layer model")));
        }
        let mut loaded = Vec::with_capacity(layers);
        for i in 0..layers {
            loaded.push(Layer::load(map, cfg, i, precision)?);
            crate::wan::log::info(format_args!("llm resident layer {}/{layers}", i + 1));
        }
        let final_norm = if layers == n {
            Some(norm_weight(map, &cfg.final_norm_key, cfg.hidden, cfg.norm_offset)?)
        } else {
            None
        };
        Ok(Self { cfg: cfg.clone(), layers: loaded, final_norm, embed: EmbedTable::load(map, cfg)?, precision })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    pub fn precision(&self) -> WeightPrecision {
        self.precision
    }

    /// Same contract and same numbers as the free [`hidden_states`].
    pub fn hidden_states(&self, ids: &[u32], positions: &[u32], attend: &[bool], taps: &[usize]) -> Result<Vec<CudaTensor>> {
        if ids.is_empty() {
            return Err(msg("llm: empty prompt"));
        }
        let h = self.cfg.hidden;
        let x = CudaTensor::from_vec(self.embed.rows(ids, h)?, vec![1, ids.len(), h])?.to_device()?;
        let x = if self.cfg.embed_scale == 1.0 { x } else { x.try_mul_scalar(self.cfg.embed_scale)? };
        encode(&self.cfg, &mut Resident(self), x, positions, attend, taps, false, None)
    }

    /// Multimodal forward: pre-scattered embeds + mRoPE + DeepStack.
    pub fn hidden_states_multimodal(
        &self,
        embedded: CudaTensor,
        attend: &[bool],
        taps: &[usize],
        multimodal: &MultimodalCtx,
    ) -> Result<Vec<CudaTensor>> {
        let s = embedded.shape.get(1).copied().unwrap_or(0);
        if s == 0 {
            return Err(msg("llm: empty multimodal prompt"));
        }
        let positions: Vec<u32> = (0..s as u32).collect();
        encode(
            &self.cfg,
            &mut Resident(self),
            embedded,
            &positions,
            attend,
            taps,
            false,
            Some(multimodal),
        )
    }

    /// Device bytes the layers occupy, from the shapes: linears at the width
    /// they were loaded (bf16 under bf16 GEMM math, f32 otherwise), norms in f32.
    pub fn device_bytes(&self) -> u64 {
        let c = &self.cfg;
        let linear = c.hidden * (c.heads + 2 * c.kv_heads) * c.head_dim + c.heads * c.head_dim * c.hidden + 3 * c.hidden * c.intermediate;
        let norms = c.hidden * if c.sandwich_norms { 4 } else { 2 } + if c.qk_norm { 2 * c.head_dim } else { 0 };
        // Per-row FP8: a byte per parameter plus one f32 scale per output row.
        let scale_rows = (c.heads + 2 * c.kv_heads) * c.head_dim + c.hidden + 2 * c.intermediate + c.hidden;
        let per_layer = match self.precision {
            WeightPrecision::Fp8Rows => linear + scale_rows * 4,
            WeightPrecision::Native => linear * if crate::wan::nn::bf16_linears_active() { 2 } else { 4 },
        };
        (self.layers.len() * (per_layer + norms * 4) + self.final_norm.as_ref().map_or(0, |_| c.hidden * 4)) as u64
    }

    /// Host bytes held by the embedding table.
    pub fn host_bytes(&self) -> u64 {
        self.embed.host_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(sandwich: bool) -> DecoderConfig {
        DecoderConfig {
            vocab: 16,
            hidden: 8,
            heads: 4,
            kv_heads: 2,
            head_dim: 4,
            intermediate: 12,
            rms_eps: 1e-6,
            norm_offset: if sandwich { 1.0 } else { 0.0 },
            act: if sandwich { Act::GeluTanh } else { Act::Silu },
            qk_norm: true,
            sandwich_norms: sandwich,
            embed_scale: if sandwich { 8f32.sqrt() } else { 1.0 },
            attn_scale: 0.5,
            layers: vec![
                LayerAttn {
                    rope_theta: 10_000.0,
                    rope_factor: 1.0,
                    window: if sandwich { Some(2) } else { None },
                    q_heads: None,
                    q_head_dim: None,
                    kv_heads: None,
                    kv_head_dim: None,
                    partial_rotary: None,
                },
                LayerAttn::global(1_000_000.0, if sandwich { 8.0 } else { 1.0 }),
            ],
            layer_prefix: "m.layers".into(),
            embed_key: "m.embed.weight".into(),
            final_norm_key: "m.norm.weight".into(),
            attention_k_eq_v: false,
        }
    }

    /// Deterministic weights by key, small enough to keep activations tame.
    fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key.bytes().fold(7u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let n: usize = shape.iter().product();
            (0..n)
                .map(|i| {
                    let v = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
                    if key.contains("norm") { 0.5 + v } else { v - 0.5 }
                })
                .collect()
        })
    }

    fn run(cfg: &DecoderConfig, ids: &[u32], taps: &[usize]) -> Vec<Vec<f32>> {
        let pos: Vec<u32> = (0..ids.len() as u32).collect();
        hidden_states(&weights(), cfg, ids, &pos, &vec![true; ids.len()], taps)
            .unwrap()
            .iter()
            .map(|t| t.host_cow().unwrap().into_owned())
            .collect()
    }

    /// A causal model's state at position p cannot depend on tokens after p —
    /// in either family, through rotary, GQA, windows and sandwich norms.
    #[test]
    fn later_tokens_do_not_change_earlier_states() {
        for sandwich in [false, true] {
            let cfg = tiny(sandwich);
            let a = run(&cfg, &[1, 5, 2, 7, 3], &[2]);
            let b = run(&cfg, &[1, 5, 2, 9, 0], &[2]);
            let h = cfg.hidden;
            for p in 0..3 {
                for j in 0..h {
                    assert!((a[0][p * h + j] - b[0][p * h + j]).abs() < 1e-6, "sandwich={sandwich} pos {p}");
                }
            }
            assert!((0..h).any(|j| (a[0][3 * h + j] - b[0][3 * h + j]).abs() > 1e-4), "position 3 must differ");
        }
    }

    /// Tap 0 is the scaled embedding; taps come back in the order asked; a tap
    /// short of the last layer never reads the layers after it.
    #[test]
    fn taps_follow_the_hf_numbering() {
        let cfg = tiny(true);
        let ids = [2u32, 0, 3];
        let got = run(&cfg, &ids, &[1, 0]);
        let (_, table) = weights_embed(&cfg);
        for (p, &id) in ids.iter().enumerate() {
            for j in 0..cfg.hidden {
                let want = table[id as usize * cfg.hidden + j] * cfg.embed_scale;
                assert!((got[1][p * cfg.hidden + j] - want).abs() < 1e-6);
            }
        }
        assert_ne!(got[0], got[1]);
        // Same tap 1 whether or not layer 2 and the final norm are ever run.
        assert_eq!(run(&cfg, &ids, &[1])[0], run(&cfg, &ids, &[1, 2])[0]);
        let pos = [0u32, 1, 2];
        assert!(hidden_states(&weights(), &cfg, &ids, &pos, &[true; 3], &[3]).is_err(), "tap past the model");
    }

    fn weights_embed(cfg: &DecoderConfig) -> (Vec<usize>, Vec<f32>) {
        let t = cuda_tensor_shaped(&weights(), &cfg.embed_key, &[cfg.vocab, cfg.hidden]).unwrap();
        (t.shape.clone(), t.host_cow().unwrap().into_owned())
    }

    /// One Qwen-style layer against attention and SwiGLU written out with
    /// plain loops: per-head QK norm, rotate_half rotary, GQA and the causal
    /// softmax are all in play, none of them through the code under test.
    #[test]
    fn one_layer_matches_a_loop_reference() {
        let mut cfg = tiny(false);
        cfg.layers.truncate(1);
        let ids = [1u32, 3, 0, 2];
        let got = &run(&cfg, &ids, &[1])[0];

        let map = weights();
        let get = |key: &str, shape: &[usize]| cuda_tensor_shaped(&map, key, shape).unwrap().host_cow().unwrap().into_owned();
        let (h, hq, hkv, d, ff, s) = (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.intermediate, ids.len());
        let table = get(&cfg.embed_key, &[cfg.vocab, h]);
        let x: Vec<Vec<f32>> = ids.iter().map(|&i| table[i as usize * h..(i as usize + 1) * h].to_vec()).collect();
        let rms = |v: &[f32], w: &[f32]| -> Vec<f32> {
            let ms = v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32;
            v.iter().zip(w).map(|(a, g)| a / (ms + cfg.rms_eps).sqrt() * g).collect()
        };
        let lin = |v: &[f32], w: &[f32], o: usize| -> Vec<f32> {
            (0..o).map(|r| v.iter().enumerate().map(|(c, a)| a * w[r * v.len() + c]).sum()).collect()
        };
        let p = "m.layers.0";
        let (wq, wk, wv, wo) = (
            get(&format!("{p}.self_attn.q_proj.weight"), &[hq * d, h]),
            get(&format!("{p}.self_attn.k_proj.weight"), &[hkv * d, h]),
            get(&format!("{p}.self_attn.v_proj.weight"), &[hkv * d, h]),
            get(&format!("{p}.self_attn.o_proj.weight"), &[h, hq * d]),
        );
        let (qn, kn) = (get(&format!("{p}.self_attn.q_norm.weight"), &[d]), get(&format!("{p}.self_attn.k_norm.weight"), &[d]));
        let (n1, n2) = (get(&format!("{p}.input_layernorm.weight"), &[h]), get(&format!("{p}.post_attention_layernorm.weight"), &[h]));
        let final_norm = get(&cfg.final_norm_key, &[h]);
        let (wg, wu, wd) = (
            get(&format!("{p}.mlp.gate_proj.weight"), &[ff, h]),
            get(&format!("{p}.mlp.up_proj.weight"), &[ff, h]),
            get(&format!("{p}.mlp.down_proj.weight"), &[h, ff]),
        );
        let rope = |v: &[f32], pos: usize| -> Vec<f32> {
            let half = d / 2;
            (0..d)
                .map(|j| {
                    let k = j % half;
                    let ang = pos as f64 * 10_000f64.powf(-((2 * k) as f64) / d as f64);
                    let rot = if j < half { -v[j + half] } else { v[j - half] };
                    v[j] * ang.cos() as f32 + rot * ang.sin() as f32
                })
                .collect()
        };
        let normed: Vec<Vec<f32>> = x.iter().map(|v| rms(v, &n1)).collect();
        let heads = |w: &[f32], n: usize, norm: Option<&[f32]>, rot: bool| -> Vec<Vec<Vec<f32>>> {
            normed
                .iter()
                .enumerate()
                .map(|(pos, v)| {
                    let full = lin(v, w, n * d);
                    (0..n)
                        .map(|hd| {
                            let t = full[hd * d..(hd + 1) * d].to_vec();
                            let t = norm.map_or(t.clone(), |g| rms(&t, g));
                            if rot { rope(&t, pos) } else { t }
                        })
                        .collect()
                })
                .collect()
        };
        let (q, k, v) = (heads(&wq, hq, Some(&qn), true), heads(&wk, hkv, Some(&kn), true), heads(&wv, hkv, None, false));
        for i in 0..s {
            let mut attn = vec![0f32; hq * d];
            for hd in 0..hq {
                let kv = hd / (hq / hkv);
                let scores: Vec<f32> = (0..=i)
                    .map(|j| q[i][hd].iter().zip(&k[j][kv]).map(|(a, b)| a * b).sum::<f32>() * cfg.attn_scale)
                    .collect();
                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                let z: f32 = scores.iter().map(|sc| (sc - mx).exp()).sum();
                for (j, sc) in scores.iter().enumerate() {
                    let w = (sc - mx).exp() / z;
                    for c in 0..d {
                        attn[hd * d + c] += w * v[j][kv][c];
                    }
                }
            }
            let after: Vec<f32> = x[i].iter().zip(lin(&attn, &wo, h)).map(|(a, b)| a + b).collect();
            let m = rms(&after, &n2);
            let (g, u) = (lin(&m, &wg, ff), lin(&m, &wu, ff));
            let act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
            let want: Vec<f32> = after.iter().zip(lin(&act, &wd, h)).map(|(a, b)| a + b).collect();
            // Tap 1 of a one-layer model is the last tap, which HF returns normed.
            let want = rms(&want, &final_norm);
            for j in 0..h {
                assert!((got[i * h + j] - want[j]).abs() < 2e-5, "pos {i} ch {j}: {} vs {}", got[i * h + j], want[j]);
            }
        }
    }

    /// Resident and streamed share the layer loop, so they must agree to the
    /// bit — including a partial load that stops at the tap, and a second
    /// prompt through the same resident weights.
    #[test]
    fn a_resident_decoder_gives_the_streamed_numbers() {
        for sandwich in [false, true] {
            let cfg = tiny(sandwich);
            let full = ResidentDecoder::load(&weights(), &cfg, 2).unwrap();
            for ids in [&[1u32, 3, 0, 2][..], &[2, 2, 1][..]] {
                let pos: Vec<u32> = (0..ids.len() as u32).collect();
                let att = vec![true; ids.len()];
                let want = run(&cfg, ids, &[0, 1, 2]);
                let got = full.hidden_states(ids, &pos, &att, &[0, 1, 2]).unwrap();
                for (w, g) in want.iter().zip(&got) {
                    assert_eq!(w, &*g.host_cow().unwrap(), "sandwich={sandwich}");
                }
            }
            let one = ResidentDecoder::load(&weights(), &cfg, 1).unwrap();
            let got = one.hidden_states(&[1, 3, 0], &[0, 1, 2], &[true; 3], &[1]).unwrap();
            assert_eq!(run(&cfg, &[1, 3, 0], &[1])[0], &*got[0].host_cow().unwrap());
            assert!(one.hidden_states(&[1, 3, 0], &[0, 1, 2], &[true; 3], &[2]).is_err(), "tap past the resident layers");
            assert!(full.device_bytes() > one.device_bytes());
            assert!(full.hidden_states(&[16], &[0], &[true], &[1]).is_err(), "token id past the table");
        }
        assert!(ResidentDecoder::load(&weights(), &tiny(false), 3).is_err());
    }

    /// Weight-only FP8 is a storage choice: the encoder it produces tracks the
    /// native one to within E4M3's 3-bit mantissa, holds a quarter of the f32
    /// bytes, and is deterministic.
    #[test]
    fn an_fp8_resident_decoder_tracks_the_native_one() {
        let cfg = tiny(false);
        let ids = [1u32, 3, 0, 2, 5];
        let pos: Vec<u32> = (0..5).collect();
        let native = ResidentDecoder::load(&weights(), &cfg, 2).unwrap();
        let fp8 = ResidentDecoder::load_with(&weights(), &cfg, 2, WeightPrecision::Fp8Rows).unwrap();
        assert_eq!((native.precision(), fp8.precision()), (WeightPrecision::Native, WeightPrecision::Fp8Rows));
        let a = native.hidden_states(&ids, &pos, &[true; 5], &[2]).unwrap().remove(0).host_cow().unwrap().into_owned();
        let b = fp8.hidden_states(&ids, &pos, &[true; 5], &[2]).unwrap().remove(0).host_cow().unwrap().into_owned();
        let num: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        let den: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
        let rel = (num / den).sqrt();
        assert!(rel > 0.0, "fp8 must actually quantize");
        assert!(rel < 0.08, "fp8 drifted {rel} from the native encoder");
        let again = fp8.hidden_states(&ids, &pos, &[true; 5], &[2]).unwrap().remove(0);
        assert_eq!(b, &*again.host_cow().unwrap());
        // A byte per parameter plus a scale per row: under half of f32 even at
        // this toy width, where the scales are a large share of a tiny layer
        // (at Qwen3-VL-32B's width they are 0.03% and 50 layers come to 24.4 GB).
        assert!(fp8.device_bytes() * 2 < native.device_bytes(), "{} vs {}", fp8.device_bytes(), native.device_bytes());
    }

    #[test]
    fn padding_is_not_attended_and_windows_limit_reach() {
        let m = attn_mask(&[false, true, true, true], Some(2)).unwrap();
        let v = m.host_cow().unwrap();
        let open = |i: usize, j: usize| v[i * 4 + j] == 0.0;
        assert!(open(0, 0), "a padded row still sees itself");
        assert!(!open(1, 0) && open(1, 1), "padding is never a key");
        assert!(open(3, 3) && open(3, 2) && !open(3, 1), "window of 2");
        assert!(!open(1, 2), "causal");
    }

    /// The prefetcher stages a layer from `linear_specs` / `norm_specs` and
    /// `Layer::assemble` then asks for parts by name: every name asked for must
    /// have been staged, with the same width, and nothing staged may go unused.
    #[test]
    fn the_specs_are_exactly_what_a_layer_asks_for() {
        for cfg in [tiny(false), tiny(true), DecoderConfig::qwen3_vl_32b_text(), DecoderConfig::gemma3_12b_text(), DecoderConfig::gemma4_12b_text()] {
            let (mut lins, mut norms) = (Vec::new(), Vec::new());
            Layer::assemble(
                &cfg,
                0,
                &mut |name, i, o| {
                    lins.push((name.to_string(), i, o));
                    Ok(Linear::zeros(i, o, false))
                },
                &mut |name, width| {
                    norms.push((name.to_string(), width));
                    Ok(CudaTensor::zeros(&[width]))
                },
            )
            .unwrap();
            let mut want: Vec<_> = linear_specs(&cfg, 0).iter().map(|(n, i, o)| (n.to_string(), *i, *o)).collect();
            if cfg.attention_k_eq_v {
                want.retain(|(n, _, _)| n != "self_attn.v_proj");
            }
            assert_eq!(lins, want, "linears are served in spec order");
            let mut staged: Vec<_> = norm_specs(&cfg, 0).iter().map(|(n, w)| (n.to_string(), *w)).collect();
            staged.sort();
            norms.sort();
            assert_eq!(norms, staged);
        }
    }

    #[test]
    fn presets_have_consistent_shapes() {
        let q = DecoderConfig::qwen3_vl_32b_text();
        assert_eq!((q.num_layers(), q.heads * q.head_dim, q.kv_heads * q.head_dim), (64, 8192, 1024));
        assert_eq!((q.vocab, DecoderConfig::gemma3_12b_text().vocab), (151_936, 262_208));
        let g = DecoderConfig::gemma3_12b_text();
        assert_eq!(g.num_layers(), 48);
        assert_eq!(g.layers.iter().filter(|l| l.window.is_none()).count(), 8);
        assert_eq!(g.layers[5].rope_theta, 1_000_000.0);
        assert_eq!(g.layers[4].window, Some(1024));
        assert!((g.embed_scale - 61.967_734).abs() < 1e-4);
        assert_eq!(g.for_bf16_reference().embed_scale, 62.0);
        assert_eq!(DecoderConfig::qwen3_vl_32b_text().for_bf16_reference().embed_scale, 1.0);
    }

    #[test]
    fn gemma4_12b_text_layout() {
        let g = DecoderConfig::gemma4_12b_text();
        assert_eq!(g.num_layers(), 48);
        assert_eq!(g.vocab, 262_144);
        assert!(g.attention_k_eq_v);
        assert_eq!(g.layer_prefix, "model.language_model.layers");
        let globals: Vec<usize> = (0..g.num_layers()).filter(|&i| g.layers[i].window.is_none()).collect();
        assert_eq!(globals, vec![5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(g.layers[4].window, Some(1024));
        assert_eq!(g.layers[5].rope_theta, 1_000_000.0);
        assert_eq!(g.layer_head_dim(5), 512);
        assert_eq!(g.layer_kv_heads(5), 1);
        assert_eq!(g.layers[5].partial_rotary, Some(0.25));
        assert_eq!(g.layers[5].rotary_width(512), 128);
        assert_eq!(g.layer_head_dim(0), 256);
        assert_eq!(g.for_bf16_reference().embed_scale, 62.0);
    }
}
