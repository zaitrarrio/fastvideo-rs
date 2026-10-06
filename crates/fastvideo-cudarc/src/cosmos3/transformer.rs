//! Cosmos3-Super Mixture-of-Transformers (`Cosmos3OmniTransformer`, diffusers
//! `transformer_cosmos3.py`), text-to-video path.
//!
//! Each of the 64 layers holds two towers over one packed sequence
//! `[text (und) | vision (gen)]`:
//!
//! * und (text): `input_layernorm` → `to_q/k/v` → per-head RMS `norm_q/k` →
//!   rotate-half M-RoPE → **causal** GQA attention over the text only →
//!   `to_out`; `post_attention_layernorm` → SwiGLU `mlp`;
//! * gen (vision): `input_layernorm_moe_gen` → `add_q/k/v_proj` →
//!   `norm_added_q/k` → M-RoPE → full GQA attention over **text keys and
//!   vision keys** → `to_add_out`; `post_attention_layernorm_moe_gen` →
//!   `mlp_moe_gen`; finally `norm_moe_gen` → `proj_out`.
//!
//! The text tower never reads the vision tokens, so its per-layer K/V are
//! fixed for a prompt: [`Cosmos3Transformer::und_cache`] runs it once per
//! prompt (layer weights streamed from the checkpoint unless resident) and
//! every denoising step runs only the gen tower against the cached keys. This
//! is exact, not an approximation, and is what makes the 62 GB gen tower the
//! only resident weight set on a 96 GB card.

use fastvideo_models::cosmos3::rope::rope_tables;
use fastvideo_models::cosmos3::Cosmos3TransformerConfig;

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn pinned(v: &[f32], dims: &[usize]) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(v.to_vec(), dims.to_vec())?;
    t.pin_device()?;
    Ok(t)
}

fn norm_weight(map: &WeightMap, key: &str, dim: usize) -> Result<CudaTensor> {
    let mut t = weights::cuda_tensor_shaped(map, key, &[dim])?;
    t.pin_device()?;
    Ok(t)
}

/// Tokens per MLP chunk (`FASTVIDEO_COSMOS3_MLP_CHUNK`, default 8192): bounds
/// the `[tokens, 25600]` SwiGLU intermediates.
fn mlp_chunk() -> usize {
    crate::wan::envflag::usize_flag("FASTVIDEO_COSMOS3_MLP_CHUNK", 8192).max(1)
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(map: &WeightMap, prefix: &str, h: usize, m: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            gate: Linear::load(map, &key("gate_proj"), h, m, false)?,
            up: Linear::load(map, &key("up_proj"), h, m, false)?,
            down: Linear::load(map, &key("down_proj"), m, h, false)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, s, d] = match x.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("cosmos3 mlp: {:?}", x.shape))),
        };
        let flat = x.reshape(vec![b * s, d])?;
        let chunk = mlp_chunk();
        let mut parts = Vec::new();
        let mut start = 0;
        while start < b * s {
            let n = chunk.min(b * s - start);
            let xc = flat.narrow(0, start, n)?;
            let g = self.gate.forward(&xc)?.silu();
            parts.push(self.down.forward(&g.mul(&self.up.forward(&xc)?)?)?);
            start += n;
        }
        let refs: Vec<&CudaTensor> = parts.iter().collect();
        CudaTensor::cat(&refs, 0)?.reshape(vec![b, s, d])
    }
}

/// Which weight set of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tower {
    Und,
    Gen,
}

/// One tower of one layer.
pub struct TowerLayer {
    ln_in: CudaTensor,
    ln_post: CudaTensor,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm_q: Option<CudaTensor>,
    norm_k: Option<CudaTensor>,
    mlp: Mlp,
}

impl TowerLayer {
    pub fn load(map: &WeightMap, cfg: &Cosmos3TransformerConfig, layer: usize, tower: Tower) -> Result<Self> {
        let (h, q, kv, m, hd) = (cfg.hidden_size, cfg.q_dim(), cfg.kv_dim(), cfg.intermediate_size, cfg.head_dim);
        let p = |n: &str| format!("layers.{layer}.{n}");
        let (ln_in, ln_post, names, norms, mlp) = match tower {
            Tower::Und => (
                "input_layernorm",
                "post_attention_layernorm",
                ["self_attn.to_q", "self_attn.to_k", "self_attn.to_v", "self_attn.to_out"],
                cfg.qk_norm_for_text.then_some(["self_attn.norm_q", "self_attn.norm_k"]),
                "mlp",
            ),
            Tower::Gen => (
                "input_layernorm_moe_gen",
                "post_attention_layernorm_moe_gen",
                [
                    "self_attn.add_q_proj",
                    "self_attn.add_k_proj",
                    "self_attn.add_v_proj",
                    "self_attn.to_add_out",
                ],
                Some(["self_attn.norm_added_q", "self_attn.norm_added_k"]),
                "mlp_moe_gen",
            ),
        };
        let (norm_q, norm_k) = match norms {
            Some([nq, nk]) => (
                Some(norm_weight(map, &p(&format!("{nq}.weight")), hd)?),
                Some(norm_weight(map, &p(&format!("{nk}.weight")), hd)?),
            ),
            None => (None, None),
        };
        Ok(Self {
            ln_in: norm_weight(map, &p(&format!("{ln_in}.weight")), h)?,
            ln_post: norm_weight(map, &p(&format!("{ln_post}.weight")), h)?,
            q: Linear::load(map, &p(names[0]), h, q, false)?,
            k: Linear::load(map, &p(names[1]), h, kv, false)?,
            v: Linear::load(map, &p(names[2]), h, kv, false)?,
            out: Linear::load(map, &p(names[3]), q, h, false)?,
            norm_q,
            norm_k,
            mlp: Mlp::load(map, &p(mlp), h, m)?,
        })
    }

    /// `q` `[1, Hq, S, D]` and `k`, `v` `[1, Hkv, S, D]` (q, k normed and rotated).
    fn qkv(
        &self,
        xn: &CudaTensor,
        cfg: &Cosmos3TransformerConfig,
        cos: &CudaTensor,
        sin: &CudaTensor,
    ) -> Result<(CudaTensor, CudaTensor, CudaTensor)> {
        let s = xn.shape[1];
        let (hq, hkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let proj = |lin: &Linear, heads: usize, norm: Option<&CudaTensor>, rope: bool| -> Result<CudaTensor> {
            let mut t = lin.forward(xn)?.reshape(vec![1, s, heads, hd])?;
            if let Some(w) = norm {
                t = t.rms_norm(w, cfg.rms_norm_eps)?;
            }
            let t = t.transpose(1, 2)?;
            if rope {
                t.rope_half(cos, sin)
            } else {
                Ok(t)
            }
        };
        Ok((
            proj(&self.q, hq, self.norm_q.as_ref(), true)?,
            proj(&self.k, hkv, self.norm_k.as_ref(), true)?,
            proj(&self.v, hkv, None, false)?,
        ))
    }
}

/// Per-layer text K/V of one prompt (after norm and rotary), `[1, Hkv, L, D]`.
pub struct UndCache {
    pub len: usize,
    pub layers: Vec<(CudaTensor, CudaTensor)>,
}

/// Rows of an embedding table without materializing it (bf16 or f32 lazily
/// mapped; any map through its generator / f32 view otherwise).
fn embed_rows(map: &WeightMap, key: &str, ids: &[u32], vocab: usize, h: usize) -> Result<CudaTensor> {
    use fastvideo_loader::LazyDType;
    if let Some(lazy) = map.lazy().filter(|_| map.has_tensor(key)) {
        let view = lazy.view(key).map_err(|e| msg(e.to_string()))?;
        if view.shape != [vocab, h] {
            return Err(msg(format!("{key}: {:?} != [{vocab}, {h}]", view.shape)));
        }
        let mut out = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            let r = id as usize;
            if r >= vocab {
                return Err(msg(format!("token id {r} >= vocab {vocab}")));
            }
            match view.dtype {
                LazyDType::BF16 => {
                    let b = &view.bytes[r * h * 2..(r + 1) * h * 2];
                    out.extend(b.chunks_exact(2).map(|c| {
                        half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()
                    }));
                }
                LazyDType::F32 => {
                    let b = &view.bytes[r * h * 4..(r + 1) * h * 4];
                    out.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                }
                other => return Err(msg(format!("{key}: dtype {other:?}"))),
            }
        }
        return CudaTensor::from_vec(out, vec![1, ids.len(), h]);
    }
    let table = weights::cuda_tensor_shaped(map, key, &[vocab, h])?;
    let rows: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
    table.index_select_rows(&rows)?.reshape(vec![1, ids.len(), h])
}

fn causal_mask(s: usize) -> Result<CudaTensor> {
    let mut m = vec![f32::MIN; s * s];
    for i in 0..s {
        for j in 0..=i {
            m[i * s + j] = 0.0;
        }
    }
    pinned(&m, &[1, 1, s, s])
}

/// `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)`.
pub fn timestep_sinusoid(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

/// TeaCache instruction for one gen-tower pass.
pub enum GenCache<'a> {
    /// Plain forward.
    Off,
    /// Run the layers and store `out − in` of the layer stack in the slot.
    Compute(&'a mut Option<CudaTensor>),
    /// Skip the layers: `in + slot`.
    Reuse(&'a Option<CudaTensor>),
}

pub struct Cosmos3Transformer {
    pub cfg: Cosmos3TransformerConfig,
    gen: Vec<TowerLayer>,
    /// Resident text tower (`FASTVIDEO_COSMOS3_UND=resident`); streamed when `None`.
    und: Option<Vec<TowerLayer>>,
    norm_gen: CudaTensor,
    proj_in: Linear,
    proj_out: Linear,
    time_1: (Vec<f32>, Vec<f32>),
    time_2: (Vec<f32>, Vec<f32>),
}

fn host_linear(map: &WeightMap, prefix: &str, i: usize, o: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let w = weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "weight"), &[o, i])?;
    let b = weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "bias"), &[o])?;
    Ok((w.host_cow()?.into_owned(), b.host_cow()?.into_owned()))
}

fn apply_host(l: &(Vec<f32>, Vec<f32>), x: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    let (w, b) = l;
    let i = x.len();
    (0..b.len())
        .into_par_iter()
        .map(|o| w[o * i..(o + 1) * i].iter().zip(x).map(|(a, c)| a * c).sum::<f32>() + b[o])
        .collect()
}

impl Cosmos3Transformer {
    /// Load the gen tower and heads; the text tower too when `und_resident`.
    pub fn load(cfg: Cosmos3TransformerConfig, map: &WeightMap, und_resident: bool) -> Result<Self> {
        cfg.validate().map_err(msg)?;
        let h = cfg.hidden_size;
        let mut gen = Vec::with_capacity(cfg.num_layers);
        let mut und = und_resident.then(|| Vec::with_capacity(cfg.num_layers));
        for i in 0..cfg.num_layers {
            gen.push(TowerLayer::load(map, &cfg, i, Tower::Gen)?);
            if let Some(u) = und.as_mut() {
                u.push(TowerLayer::load(map, &cfg, i, Tower::Und)?);
            }
        }
        Ok(Self {
            gen,
            und,
            norm_gen: norm_weight(map, "norm_moe_gen.weight", h)?,
            proj_in: Linear::load(map, "proj_in", cfg.patch_latent_dim, h, true)?,
            proj_out: Linear::load(map, "proj_out", h, cfg.patch_latent_dim, true)?,
            time_1: host_linear(map, "time_embedder.linear_1", 256, h)?,
            time_2: host_linear(map, "time_embedder.linear_2", h, h)?,
            cfg,
        })
    }

    /// `time_embedder(time_proj(t))` for the scaled timestep (`t · 0.001`), f32.
    pub fn time_embed(&self, t: f32) -> Vec<f32> {
        let f = timestep_sinusoid(t, 256);
        let hdn: Vec<f32> = apply_host(&self.time_1, &f)
            .into_iter()
            .map(|v| v / (1.0 + (-v).exp()))
            .collect();
        apply_host(&self.time_2, &hdn)
    }

    /// Run the text tower over `ids` (positions `0..L` on all axes) and keep
    /// each layer's rotated keys and values. `map` supplies streamed layers.
    pub fn und_cache(&self, map: &WeightMap, ids: &[u32]) -> Result<UndCache> {
        let cfg = &self.cfg;
        let l = ids.len();
        if l == 0 {
            return Err(msg("cosmos3: empty prompt"));
        }
        let mut x = embed_rows(map, "embed_tokens.weight", ids, cfg.vocab_size, cfg.hidden_size)?;
        let (pos, _) = fastvideo_models::cosmos3::rope::text_positions(l);
        let (c, s) = rope_tables(cfg, &pos);
        let cos = pinned(&c, &[l, cfg.head_dim])?;
        let sin = pinned(&s, &[l, cfg.head_dim])?;
        let mask = causal_mask(l)?;
        let scale = (cfg.head_dim as f32).powf(-0.5);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let streamed;
            let layer = match &self.und {
                Some(u) => &u[i],
                None => {
                    streamed = TowerLayer::load(map, cfg, i, Tower::Und)?;
                    &streamed
                }
            };
            let xn = x.rms_norm(&layer.ln_in, cfg.rms_norm_eps)?;
            let (q, k, v) = layer.qkv(&xn, cfg, &cos, &sin)?;
            let last = i + 1 == cfg.num_layers;
            if !last {
                let a = crate::llm::attn::scaled_dot_product_attention_gqa(&q, &k, &v, Some(scale), Some(&mask))?;
                x = x.add(&layer.out.forward(&a.merge_heads()?)?)?;
                let xn = x.rms_norm(&layer.ln_post, cfg.rms_norm_eps)?;
                x = x.add(&layer.mlp.forward(&xn)?)?;
            }
            layers.push((k, v));
        }
        Ok(UndCache { len: l, layers })
    }

    /// Gen-tower velocity for `latents` `[1, C, T, H, W]` (DiT latent space)
    /// at scaled timestep `t` (`int(sigma·1000) · 0.001`). `cos`/`sin` are the
    /// vision M-RoPE tables `[T·⌈H/p⌉·⌈W/p⌉, head_dim]`.
    pub fn forward_gen(
        &self,
        latents: &CudaTensor,
        temb: &[f32],
        und: &UndCache,
        cos: &CudaTensor,
        sin: &CudaTensor,
        cache: GenCache<'_>,
    ) -> Result<CudaTensor> {
        let cfg = &self.cfg;
        let [b, c, t, h, w] = match latents.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg(format!("cosmos3 dit: {:?}", latents.shape))),
        };
        if b != 1 || c != cfg.latent_channel {
            return Err(msg(format!("cosmos3 dit: latents {:?}", latents.shape)));
        }
        let p = cfg.latent_patch_size;
        let (hp, wp) = (h.div_ceil(p), w.div_ceil(p));
        let n = t * hp * wp;
        let tokens = patchify(&latents.host_cow()?, c, t, h, w, p);
        let x = self
            .proj_in
            .forward(&CudaTensor::from_vec(tokens, vec![n, cfg.patch_latent_dim])?)?
            .reshape(vec![1, n, cfg.hidden_size])?;
        let x = x.add(&pinned(temb, &[1, 1, cfg.hidden_size])?)?;
        let g = cfg.num_attention_heads / cfg.num_key_value_heads;
        let scale = (cfg.head_dim as f32).powf(-0.5);
        let run = |mut x: CudaTensor| -> Result<CudaTensor> {
            for (i, layer) in self.gen.iter().enumerate() {
                let xn = x.rms_norm(&layer.ln_in, cfg.rms_norm_eps)?;
                let (q, k, v) = layer.qkv(&xn, cfg, cos, sin)?;
                let (ku, vu) = &und.layers[i];
                let k_all = CudaTensor::cat(&[ku, &k], 2)?.repeat_kv(g)?;
                let v_all = CudaTensor::cat(&[vu, &v], 2)?.repeat_kv(g)?;
                let a = nn::scaled_dot_product_attention(&q, &k_all, &v_all, Some(scale))?;
                x = x.add(&layer.out.forward(&a.merge_heads()?)?)?;
                let xn = x.rms_norm(&layer.ln_post, cfg.rms_norm_eps)?;
                x = x.add(&layer.mlp.forward(&xn)?)?;
            }
            Ok(x)
        };
        let x = match cache {
            GenCache::Off => run(x)?,
            GenCache::Compute(slot) => {
                let out = run(x.clone())?;
                *slot = Some(out.sub(&x)?);
                out
            }
            GenCache::Reuse(slot) => {
                let r = slot.as_ref().ok_or_else(|| msg("cosmos3 teacache: reuse before compute"))?;
                x.add(r)?
            }
        };
        let y = self
            .proj_out
            .forward(&x.rms_norm(&self.norm_gen, cfg.rms_norm_eps)?)?;
        let out = unpatchify(&y.host_cow()?, c, t, h, w, p);
        CudaTensor::from_vec(out, vec![1, c, t, h, w])
    }
}

/// `_patchify_and_pack_latents`: pad H, W up to a multiple of `p`, then
/// `cthpwq -> thwpqc` → `[T·Hp·Wp, p·p·C]`.
pub fn patchify(x: &[f32], c: usize, t: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (hp, wp) = (h.div_ceil(p), w.div_ceil(p));
    let dim = p * p * c;
    let mut out = vec![0f32; t * hp * wp * dim];
    for ti in 0..t {
        for yi in 0..hp {
            for xi in 0..wp {
                let base = ((ti * hp + yi) * wp + xi) * dim;
                for a in 0..p {
                    for bb in 0..p {
                        let (y, xx) = (yi * p + a, xi * p + bb);
                        if y >= h || xx >= w {
                            continue;
                        }
                        for ci in 0..c {
                            out[base + (a * p + bb) * c + ci] = x[((ci * t + ti) * h + y) * w + xx];
                        }
                    }
                }
            }
        }
    }
    out
}

/// `_unpatchify_and_unpack_latents` for an all-noisy item (crop the padding).
pub fn unpatchify(y: &[f32], c: usize, t: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (hp, wp) = (h.div_ceil(p), w.div_ceil(p));
    let dim = p * p * c;
    let mut out = vec![0f32; c * t * h * w];
    for ti in 0..t {
        for yi in 0..hp {
            for xi in 0..wp {
                let base = ((ti * hp + yi) * wp + xi) * dim;
                for a in 0..p {
                    for bb in 0..p {
                        let (yy, xx) = (yi * p + a, xi * p + bb);
                        if yy >= h || xx >= w {
                            continue;
                        }
                        for ci in 0..c {
                            out[((ci * t + ti) * h + yy) * w + xx] = y[base + (a * p + bb) * c + ci];
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::cosmos3::rope::vision_positions;

    fn seeded(key: &str, shape: &[usize]) -> Vec<f32> {
        let h = key
            .bytes()
            .fold(1469598103934665603u64, |a, b| (a ^ b as u64).wrapping_mul(1099511628211));
        let n: usize = shape.iter().product();
        let norm = key.contains("norm") || key.contains("layernorm");
        (0..n)
            .map(|i| {
                let v = ((h.wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15)) >> 11) as f64
                    / (1u64 << 53) as f64) as f32;
                let u = (v - 0.5) * 0.4;
                if norm {
                    1.0 + u
                } else {
                    u
                }
            })
            .collect()
    }

    #[test]
    fn patchify_pads_and_roundtrips() {
        let (c, t, h, w, p) = (3, 2, 3, 4, 2);
        let x: Vec<f32> = (0..c * t * h * w).map(|i| i as f32 + 1.0).collect();
        let y = patchify(&x, c, t, h, w, p);
        assert_eq!(y.len(), t * 2 * 2 * p * p * c);
        // Token (t0, row 1, col 0) covers source row 2 and padding row 3.
        let tok = 2;
        let base = tok * p * p * c;
        assert_eq!(y[base], x[(2) * w]); // (a=0, b=0, c=0) ← row 2, col 0
        assert_eq!(y[base + 2 * c], 0.0); // (a=1, b=0) is padding
        assert_eq!(unpatchify(&y, c, t, h, w, p), x);
    }

    #[test]
    fn und_cache_equals_joint_causal_text_prefix() {
        // The cached text K/V must be what a joint pass computes for the
        // text rows: the text tower never sees vision tokens.
        let cfg = Cosmos3TransformerConfig::tiny();
        let map = WeightMap::generated(seeded);
        let dit = Cosmos3Transformer::load(cfg.clone(), &map, true).unwrap();
        let ids = [3u32, 7, 11, 2];
        let a = dit.und_cache(&map, &ids).unwrap();
        let streamed = Cosmos3Transformer::load(cfg.clone(), &map, false).unwrap();
        let b = streamed.und_cache(&map, &ids).unwrap();
        assert_eq!(a.layers.len(), cfg.num_layers);
        for ((ka, va), (kb, vb)) in a.layers.iter().zip(&b.layers) {
            assert_eq!(ka.shape, vec![1, cfg.num_key_value_heads, 4, cfg.head_dim]);
            assert_eq!(ka.host_cow().unwrap(), kb.host_cow().unwrap());
            assert_eq!(va.host_cow().unwrap(), vb.host_cow().unwrap());
        }
        // Causality: the first token's K/V do not depend on later tokens.
        let c = dit.und_cache(&map, &ids[..2]).unwrap();
        for ((kc, _), (ka, _)) in c.layers.iter().zip(&a.layers) {
            let d = cfg.head_dim;
            let kc = kc.host_cow().unwrap();
            let ka = ka.host_cow().unwrap();
            for hh in 0..cfg.num_key_value_heads {
                for j in 0..d {
                    let x = kc[(hh * 2) * d + j];
                    let y = ka[(hh * 4) * d + j];
                    assert!((x - y).abs() < 1e-5);
                }
            }
        }
    }

    /// Golden: `scripts/ref/cosmos3_reference.py`, a NumPy transcription of
    /// diffusers' joint `[und | gen]` forward (both towers, causal text
    /// attention). Checks the text-K/V cache decomposition, the interleaved
    /// M-RoPE, the fps-modulated vision positions and the 2×2 patching.
    #[test]
    fn golden_tiny_matches_numpy_reference() {
        let dir = std::env::temp_dir().join(format!("cosmos3-golden-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), include_bytes!("fixtures/tiny_golden.st")).unwrap();
        let map = WeightMap::load_dir(&dir).unwrap();
        let cfg = Cosmos3TransformerConfig::tiny();
        let dit = Cosmos3Transformer::load(cfg.clone(), &map, false).unwrap();
        let get = |k: &str| {
            let (shape, v) = map.get_f32(k).unwrap();
            CudaTensor::from_vec(v, shape).unwrap()
        };
        let ids: Vec<u32> = get("test.ids").host_cow().unwrap().iter().map(|&v| v as u32).collect();
        let lat = get("test.latents");
        let t = get("test.timestep").host_cow().unwrap()[0] as i64;
        let und = dit.und_cache(&map, &ids).unwrap();
        let (_, _, tt, hh, ww) = (lat.shape[0], lat.shape[1], lat.shape[2], lat.shape[3], lat.shape[4]);
        let grid = [tt, hh.div_ceil(2), ww.div_ceil(2)];
        let offset = (ids.len() + cfg.temporal_modality_margin) as f32;
        let pos = vision_positions(&cfg, grid, offset, Some(24.0), 4);
        let (c, s) = rope_tables(&cfg, &pos);
        let cos = pinned(&c, &[pos.len(), cfg.head_dim]).unwrap();
        let sin = pinned(&s, &[pos.len(), cfg.head_dim]).unwrap();
        let temb = dit.time_embed(fastvideo_models::cosmos3::schedule::transformer_timestep(t, cfg.timestep_scale));
        let got = dit.forward_gen(&lat, &temb, &und, &cos, &sin, GenCache::Off).unwrap();
        let want = get("test.expected");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got.shape, want.shape);
        let mut max = 0f32;
        for (a, b) in got.host_cow().unwrap().iter().zip(want.host_cow().unwrap().iter()) {
            max = max.max((a - b).abs());
        }
        assert!(max < 2e-4, "max |rust - numpy| = {max}");
    }

    #[test]
    fn gen_forward_shapes_and_teacache_reuse() {
        let cfg = Cosmos3TransformerConfig::tiny();
        let map = WeightMap::generated(seeded);
        let dit = Cosmos3Transformer::load(cfg.clone(), &map, false).unwrap();
        let und = dit.und_cache(&map, &[1, 2, 3]).unwrap();
        let (t, h, w) = (2, 3, 4);
        let pos = vision_positions(&cfg, [t, 2, 2], (3 + cfg.temporal_modality_margin) as f32, Some(24.0), 4);
        let (c, s) = rope_tables(&cfg, &pos);
        let cos = pinned(&c, &[pos.len(), cfg.head_dim]).unwrap();
        let sin = pinned(&s, &[pos.len(), cfg.head_dim]).unwrap();
        let x: Vec<f32> = (0..cfg.latent_channel * t * h * w).map(|i| (i as f32 * 0.3).sin()).collect();
        let lat = CudaTensor::from_vec(x, vec![1, cfg.latent_channel, t, h, w]).unwrap();
        let temb = dit.time_embed(0.5);
        let mut slot = None;
        let a = dit
            .forward_gen(&lat, &temb, &und, &cos, &sin, GenCache::Compute(&mut slot))
            .unwrap();
        assert_eq!(a.shape, vec![1, cfg.latent_channel, t, h, w]);
        assert!(a.host_cow().unwrap().iter().all(|v| v.is_finite()));
        // Reusing the residual on the same input reproduces the output.
        let r = dit.forward_gen(&lat, &temb, &und, &cos, &sin, GenCache::Reuse(&slot)).unwrap();
        for (x, y) in a.host_cow().unwrap().iter().zip(r.host_cow().unwrap().iter()) {
            assert!((x - y).abs() < 1e-4);
        }
        let off = dit.forward_gen(&lat, &temb, &und, &cos, &sin, GenCache::Off).unwrap();
        assert_eq!(off.host_cow().unwrap(), a.host_cow().unwrap());
    }
}
