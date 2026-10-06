//! SANA-Video DiT (`SanaVideoTransformer3DModel`): patch embed, AdaLN-single
//! timestep modulation, 20 blocks of {ReLU linear self-attention with 3-axis
//! RoPE, softmax cross-attention to the caption, GLUMBTempConv FFN}, final
//! modulated norm and unpatchify. Spec: docs/ports/sana-video.md.
//!
//! Layouts: tokens are `(frame, h, w)` row-major, channel-last
//! `[1, N, dim]`. The 1x1 convolutions of the FFN are linears on the channel
//! axis; the depthwise 3x3 runs NCHW per frame (cuDNN on the device); the
//! temporal `(3, 1)` convolution is three shifted linears over frames.
//!
//! Batch is 1: the pipeline runs the CFG branches as two calls.

use fastvideo_models::sana_video::config::{
    SANA_CAPTION_NORM_EPS, SANA_LINEAR_ATTN_EPS, SANA_QK_NORM_EPS,
};
use fastvideo_models::sana_video::{rope, SanaOptimizations, SanaVideoTransformerConfig};

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

fn tensor(map: &WeightMap, key: &str, shape: &[usize]) -> Result<CudaTensor> {
    pinned(weights::cuda_tensor_shaped(map, key, shape)?)
}

/// A `1x1` convolution `[out, in, 1, 1]` as a linear.
fn conv1x1(map: &WeightMap, prefix: &str, cin: usize, cout: usize, bias: bool) -> Result<Linear> {
    let w = weights::cuda_tensor_shaped(map, &format!("{prefix}.weight"), &[cout, cin, 1, 1])?
        .reshape(vec![cout, cin])?;
    let b = if bias {
        Some(weights::cuda_tensor_shaped(
            map,
            &format!("{prefix}.bias"),
            &[cout],
        )?)
    } else {
        None
    };
    Linear::from_tensors(w, b)
}

/// `x · (1 + scale) + shift`.
fn modulate(x: &CudaTensor, shift: &CudaTensor, scale: &CudaTensor) -> Result<CudaTensor> {
    x.mul(&scale.try_add_scalar(1.0)?)?.add(shift)
}

/// Self-attention projections: `[q, k, v]`, or with `--qkv-merge` one
/// linear whose rows are `[q; k; v]`.
struct Qkv(Vec<Linear>);

struct Block {
    /// `[6, dim]`.
    table: CudaTensor,
    qkv1: Qkv,
    norm_q1: CudaTensor,
    norm_k1: CudaTensor,
    out1: Linear,
    q2: Linear,
    kv2: Linear,
    norm_q2: CudaTensor,
    norm_k2: CudaTensor,
    out2: Linear,
    ff_inverted: Linear,
    /// `[2 * hidden, 1, 3, 3]` depthwise.
    ff_depth_w: CudaTensor,
    ff_depth_b: CudaTensor,
    ff_point: Linear,
    /// `conv_temp` taps `k = 0, 1, 2` (frame offsets -1, 0, +1).
    ff_temp: [Linear; 3],
}

impl Block {
    fn load(
        map: &WeightMap,
        i: usize,
        cfg: &SanaVideoTransformerConfig,
        opt: &SanaOptimizations,
    ) -> Result<Self> {
        let d = cfg.inner_dim();
        let hid = cfg.ff_hidden();
        let p = |n: &str| format!("transformer_blocks.{i}.{n}");
        let bias1 = cfg.attention_bias;
        let qkv1 = if opt.qkv_merge {
            Qkv(vec![Linear::load_fused(
                map,
                &[&p("attn1.to_q"), &p("attn1.to_k"), &p("attn1.to_v")],
                d,
                d,
                bias1,
            )?])
        } else {
            Qkv(vec![
                Linear::load(map, &p("attn1.to_q"), d, d, bias1)?,
                Linear::load(map, &p("attn1.to_k"), d, d, bias1)?,
                Linear::load(map, &p("attn1.to_v"), d, d, bias1)?,
            ])
        };
        let temp = weights::cuda_tensor_shaped(map, &p("ff.conv_temp.weight"), &[d, d, 3, 1])?;
        let temp = temp.host_cow()?;
        let tap = |k: usize| -> Result<Linear> {
            let mut w = Vec::with_capacity(d * d);
            for o in 0..d {
                for c in 0..d {
                    w.push(temp[(o * d + c) * 3 + k]);
                }
            }
            Linear::from_tensors(CudaTensor::from_vec(w, vec![d, d])?, None)
        };
        Ok(Self {
            table: tensor(map, &p("scale_shift_table"), &[6, d])?,
            qkv1,
            norm_q1: tensor(map, &p("attn1.norm_q.weight"), &[d])?,
            norm_k1: tensor(map, &p("attn1.norm_k.weight"), &[d])?,
            out1: Linear::load(map, &p("attn1.to_out.0"), d, d, true)?,
            q2: Linear::load(map, &p("attn2.to_q"), d, d, true)?,
            kv2: Linear::load_fused(map, &[&p("attn2.to_k"), &p("attn2.to_v")], d, d, true)?,
            norm_q2: tensor(map, &p("attn2.norm_q.weight"), &[d])?,
            norm_k2: tensor(map, &p("attn2.norm_k.weight"), &[d])?,
            out2: Linear::load(map, &p("attn2.to_out.0"), d, d, true)?,
            ff_inverted: conv1x1(map, &p("ff.conv_inverted"), d, 2 * hid, true)?,
            ff_depth_w: tensor(map, &p("ff.conv_depth.weight"), &[2 * hid, 1, 3, 3])?,
            ff_depth_b: tensor(map, &p("ff.conv_depth.bias"), &[2 * hid])?,
            ff_point: conv1x1(map, &p("ff.conv_point"), hid, d, false)?,
            ff_temp: [tap(0)?, tap(1)?, tap(2)?],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &CudaTensor,
        tmod: &CudaTensor,
        text: &CudaTensor,
        cos: &CudaTensor,
        sin: &CudaTensor,
        grid: (usize, usize, usize),
        cfg: &SanaVideoTransformerConfig,
        opt: &SanaOptimizations,
    ) -> Result<CudaTensor> {
        let d = cfg.inner_dim();
        let n = x.shape[1];
        // scale_shift_table[None, None] + timestep.reshape(B, 1, 6, dim)
        let mods = self
            .table
            .add(&tmod.reshape(vec![6, d])?)?
            .reshape(vec![6, 1, 1, d])?;
        let m: Vec<CudaTensor> = (0..6)
            .map(|i| mods.narrow(0, i, 1)?.reshape(vec![1, 1, d]))
            .collect::<Result<_>>()?;
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            (&m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);

        // 1. Linear self-attention.
        let h = modulate(&x.layer_norm(cfg.norm_eps, None, None)?, shift_msa, scale_msa)?;
        let (q, k, v) = match &self.qkv1.0[..] {
            [q, k, v] => (q.forward(&h)?, k.forward(&h)?, v.forward(&h)?),
            _ => {
                let qkv = self.qkv1.0[0].forward(&h)?;
                (
                    qkv.narrow(2, 0, d)?,
                    qkv.narrow(2, d, d)?,
                    qkv.narrow(2, 2 * d, d)?,
                )
            }
        };
        let attn = linear_attention(
            &q.rms_norm(&self.norm_q1, SANA_QK_NORM_EPS)?,
            &k.rms_norm(&self.norm_k1, SANA_QK_NORM_EPS)?,
            &v,
            cfg.num_attention_heads,
            cos,
            sin,
            opt.linattn_bf16,
        )?;
        let x = x.add(&self.out1.forward(&attn)?.mul(gate_msa)?)?;

        // 2. Cross-attention (no norm, no gate).
        let heads = cfg.num_cross_attention_heads;
        let qd = self.q2.forward(&x)?;
        let q = qd.qk_norm_rope_bhsd(0, heads, &self.norm_q2, None, SANA_QK_NORM_EPS)?;
        let kv = self.kv2.forward(text)?;
        let k = kv.qk_norm_rope_bhsd(0, heads, &self.norm_k2, None, SANA_QK_NORM_EPS)?;
        let v = kv.split_heads_bhsd(d, heads, d / heads)?;
        let a = nn::scaled_dot_product_attention(&q, &k, &v, None)?.merge_heads()?;
        let x = x.add(&self.out2.forward(&a)?)?;

        // 3. GLUMBTempConv.
        let h = modulate(&x.layer_norm(cfg.norm_eps, None, None)?, shift_mlp, scale_mlp)?;
        let ff = self.glumb(&h, grid, cfg)?;
        debug_assert_eq!(ff.shape, vec![1, n, d]);
        x.add(&ff.mul(gate_mlp)?)
    }

    /// `GLUMBTempConv(residual_connection=False, norm_type=None)` on
    /// channel-last `[1, N, dim]`.
    fn glumb(
        &self,
        h: &CudaTensor,
        (f, hh, ww): (usize, usize, usize),
        cfg: &SanaVideoTransformerConfig,
    ) -> Result<CudaTensor> {
        let d = cfg.inner_dim();
        let hid = cfg.ff_hidden();
        let n = f * hh * ww;
        let a = self.ff_inverted.forward(h)?.silu();
        // Depthwise 3x3 per frame, NCHW.
        let a = a
            .reshape(vec![f, hh, ww, 2 * hid])?
            .permute(&[0, 3, 1, 2])?
            .conv2d_groups(&self.ff_depth_w, Some(&self.ff_depth_b), 1, 1, 2 * hid)?;
        let g = a.narrow(1, 0, hid)?.mul(&a.narrow(1, hid, hid)?.silu())?;
        let g = g.permute(&[0, 2, 3, 1])?.reshape(vec![1, n, hid])?;
        let p = self.ff_point.forward(&g)?;
        // x + conv_temp(x): taps over frames with zero padding.
        let pf = p.reshape(vec![1, f, hh * ww, d])?;
        let padded = pf.pad(1, 1, 1, crate::wan::ops::PadMode::Zeros)?;
        let mut out = pf.clone();
        for (k, tap) in self.ff_temp.iter().enumerate() {
            out = out.add(&tap.forward(&padded.narrow(1, k, f)?)?)?;
        }
        out.reshape(vec![1, n, d])
    }
}

/// `SanaLinearAttnProcessor3_0` on `[1, N, H*D]` q/k/v (q, k already
/// RMS-normed across heads). ReLU feature maps, RoPE on the features, and
/// the normaliser from the *unrotated* features:
/// `out = (q_rot · (k_rotᵀ v)) / (q · Σ_n k + 1e-15)`, in f32.
pub fn linear_attention(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    heads: usize,
    cos: &CudaTensor,
    sin: &CudaTensor,
    bf16_operands: bool,
) -> Result<CudaTensor> {
    let [b, n, width] = q.shape[..] else {
        return Err(msg(format!("sana linear attn: q {:?}", q.shape)));
    };
    if width % heads != 0 || k.shape != q.shape || v.shape != q.shape {
        return Err(msg(format!(
            "sana linear attn: q {:?} k {:?} v {:?} heads {heads}",
            q.shape, k.shape, v.shape
        )));
    }
    let dh = width / heads;
    let relu = |t: &CudaTensor| -> Result<CudaTensor> {
        Ok(t.reshape(vec![b, n, heads, dh])?.clamp(0.0, f32::INFINITY))
    };
    let (q, k) = (relu(q)?, relu(k)?);
    let round = |t: CudaTensor| -> Result<CudaTensor> {
        if bf16_operands {
            t.quantize_bf16()?.to_f32_act()
        } else {
            t.to_f32_act()
        }
    };
    let bhsd = |t: &CudaTensor| t.transpose(1, 2);
    let q_rot = round(bhsd(&q.apply_rotary_bshd(cos, sin)?)?)?;
    let k_rot = round(bhsd(&k.apply_rotary_bshd(cos, sin)?)?)?;
    let v = round(bhsd(&v.reshape(vec![b, n, heads, dh])?)?)?;
    let (q, k) = (bhsd(&q)?.to_f32_act()?, bhsd(&k)?.to_f32_act()?);
    // Σ_n k: [B, H, 1, D] = ones[B, H, 1, N] @ k.
    let ones = CudaTensor::ones(&[b, heads, 1, n]).to_device()?;
    let k_sum = ones.matmul(&k)?;
    // q · Σ k: [B, H, N, 1].
    let denom = q.matmul(&k_sum.transpose(2, 3)?)?.try_add_scalar(SANA_LINEAR_ATTN_EPS)?;
    // k_rotᵀ v: [B, H, D, D]; out = q_rot @ that.
    let kv = k_rot.transpose(2, 3)?.matmul(&v)?;
    let out = q_rot.matmul(&kv)?.div(&denom)?;
    out.merge_heads()
}

/// Full SANA-Video DiT.
pub struct SanaVideoTransformer {
    cfg: SanaVideoTransformerConfig,
    opt: SanaOptimizations,
    patch: Linear,
    t_lin1: Linear,
    t_lin2: Linear,
    t_mod: Linear,
    cap1: Linear,
    cap2: Linear,
    cap_norm: CudaTensor,
    /// `[2, dim]`: shift, scale of the output norm.
    out_table: CudaTensor,
    blocks: Vec<Block>,
    proj_out: Linear,
}

impl SanaVideoTransformer {
    pub fn load(
        cfg: SanaVideoTransformerConfig,
        opt: SanaOptimizations,
        map: &WeightMap,
    ) -> Result<Self> {
        cfg.validate().map_err(msg)?;
        let d = cfg.inner_dim();
        let [pt, ph, pw] = cfg.patch_size;
        let patch_in = cfg.in_channels * pt * ph * pw;
        let pw_t = weights::cuda_tensor_shaped(
            map,
            "patch_embedding.weight",
            &[d, cfg.in_channels, pt, ph, pw],
        )?
        .reshape(vec![d, patch_in])?;
        let patch = Linear::from_tensors(
            pw_t,
            Some(weights::cuda_tensor_shaped(map, "patch_embedding.bias", &[d])?),
        )?;
        let blocks = (0..cfg.num_layers)
            .map(|i| Block::load(map, i, &cfg, &opt))
            .collect::<Result<Vec<_>>>()?;
        let te = "time_embed.emb.timestep_embedder";
        Ok(Self {
            patch,
            t_lin1: Linear::load(map, &format!("{te}.linear_1"), 256, d, true)?,
            t_lin2: Linear::load(map, &format!("{te}.linear_2"), d, d, true)?,
            t_mod: Linear::load(map, "time_embed.linear", d, 6 * d, true)?,
            cap1: Linear::load(map, "caption_projection.linear_1", cfg.caption_channels, d, true)?,
            cap2: Linear::load(map, "caption_projection.linear_2", d, d, true)?,
            cap_norm: tensor(map, "caption_norm.weight", &[d])?,
            out_table: tensor(map, "scale_shift_table", &[2, d])?,
            blocks,
            proj_out: Linear::load(
                map,
                "proj_out",
                d,
                cfg.patch_volume() * cfg.out_channels,
                true,
            )?,
            cfg,
            opt,
        })
    }

    pub fn config(&self) -> &SanaVideoTransformerConfig {
        &self.cfg
    }

    pub fn optimizations(&self) -> &SanaOptimizations {
        &self.opt
    }

    /// Caption tokens `[1, L, caption_channels]` → `[1, L, dim]`
    /// (`caption_projection` + `caption_norm`). Per prompt, reusable across steps.
    pub fn project_caption(&self, text: &CudaTensor) -> Result<CudaTensor> {
        let h = self.cap2.forward(&self.cap1.forward(text)?.gelu_tanh())?;
        h.rms_norm(&self.cap_norm, SANA_CAPTION_NORM_EPS)
    }

    /// Rotary tables for a latent `(T, H, W)`, on the device when one is live.
    pub fn rope(&self, t: usize, h: usize, w: usize) -> Result<(CudaTensor, CudaTensor)> {
        let [pt, ph, pw] = self.cfg.patch_size;
        let grid = (t / pt, h / ph, w / pw);
        let (c, s) = rope::rope_tables(
            self.cfg.attention_head_dim,
            grid,
            self.cfg.rope_max_seq_len,
            10_000.0,
        )
        .map_err(msg)?;
        let n = grid.0 * grid.1 * grid.2;
        let hd = self.cfg.attention_head_dim;
        Ok((
            pinned(CudaTensor::from_vec(c, vec![n, hd])?)?,
            pinned(CudaTensor::from_vec(s, vec![n, hd])?)?,
        ))
    }

    /// `latents` `[1, C, T, H, W]`, `caption` from [`Self::project_caption`]
    /// `[1, L, dim]`, integer `timestep` → velocity `[1, C, T, H, W]`.
    pub fn forward(
        &self,
        latents: &CudaTensor,
        caption: &CudaTensor,
        timestep: f32,
        rope: &(CudaTensor, CudaTensor),
    ) -> Result<CudaTensor> {
        let cfg = &self.cfg;
        let [1, c, t, h, w] = latents.shape[..] else {
            return Err(msg(format!(
                "sana dit: latents {:?} want [1, C, T, H, W]",
                latents.shape
            )));
        };
        let [_, ph, pw] = cfg.patch_size;
        if c != cfg.in_channels || h % ph != 0 || w % pw != 0 {
            return Err(msg(format!(
                "sana dit: latents {:?} vs in_channels {} / patch {:?}",
                latents.shape, cfg.in_channels, cfg.patch_size
            )));
        }
        let (hh, ww) = (h / ph, w / pw);
        let n = t * hh * ww;
        let d = cfg.inner_dim();
        // Patchify (p_t = 1): [1, C, T*H', ph, W', pw] → [1, N, C*ph*pw].
        let x = latents
            .reshape(vec![1, c, t * hh, ph, ww, pw])?
            .permute(&[0, 2, 4, 1, 3, 5])?
            .reshape(vec![1, n, c * ph * pw])?;
        let mut x = self.patch.forward(&x)?;

        // AdaLayerNormSingle.
        let tt = CudaTensor::from_vec(vec![timestep], vec![1])?;
        let temb = nn::sinusoidal_timesteps(&tt, 256)?.to_device()?;
        let emb = self.t_lin2.forward(&self.t_lin1.forward(&temb)?.silu())?; // [1, d]
        let tmod = self.t_mod.forward(&emb.silu())?; // [1, 6d]

        for block in &self.blocks {
            x = block.forward(
                &x,
                &tmod,
                caption,
                &rope.0,
                &rope.1,
                (t, hh, ww),
                cfg,
                &self.opt,
            )?;
        }

        // SanaModulatedNorm: shift, scale = table + emb.
        let ss = self.out_table.add(&emb)?.reshape(vec![2, 1, 1, d])?;
        let shift = ss.narrow(0, 0, 1)?.reshape(vec![1, 1, d])?;
        let scale = ss.narrow(0, 1, 1)?.reshape(vec![1, 1, d])?;
        let x = modulate(&x.layer_norm(1e-6, None, None)?, &shift, &scale)?;
        let x = self.proj_out.forward(&x)?; // [1, N, ph*pw*C]
        // Unpatchify: (ph, pw, C) per token → [1, C, T, H, W].
        x.reshape(vec![1, t * hh, ww, ph, pw, c])?
            .permute(&[0, 5, 1, 3, 2, 4])?
            .reshape(vec![1, c, t, h, w])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn small_random(key: &str, shape: &[usize]) -> Vec<f32> {
        // Deterministic pseudo-random values in [-0.2, 0.2], seeded by the key.
        let mut h: u64 = 1469598103934665603;
        for b in key.bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(1099511628211);
        }
        let n: usize = shape.iter().product();
        (0..n)
            .map(|i| {
                let mut z = h.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15);
                z ^= z >> 31;
                ((z % 10_000) as f32 / 10_000.0 - 0.5) * 0.4
            })
            .collect()
    }

    type Seen = Arc<Mutex<Vec<(String, Vec<usize>)>>>;

    fn recording_map(seen: Seen) -> WeightMap {
        WeightMap::generated(move |key, shape| {
            seen.lock().unwrap().push((key.to_string(), shape.to_vec()));
            if key.ends_with("norm_q.weight")
                || key.ends_with("norm_k.weight")
                || key == "caption_norm.weight"
            {
                return vec![1.0; shape.iter().product()];
            }
            small_random(key, shape)
        })
    }

    /// The loader reads exactly the keys and shapes the host key spec lists
    /// (which the models crate checks against the Hub header).
    #[test]
    fn loader_reads_exactly_the_key_spec() {
        for opt in [SanaOptimizations::BASELINE, SanaOptimizations::FULL] {
            let cfg = SanaVideoTransformerConfig::tiny();
            let seen = Arc::new(Mutex::new(Vec::new()));
            SanaVideoTransformer::load(cfg.clone(), opt, &recording_map(seen.clone())).unwrap();
            let got: std::collections::BTreeMap<String, Vec<usize>> =
                seen.lock().unwrap().iter().cloned().collect();
            let want = fastvideo_models::sana_video::keys::transformer_keys(&cfg);
            assert_eq!(got, want);
        }
    }

    /// Tiny random-weight forward: shapes, finiteness, and the QKV merge /
    /// separate projections agreeing.
    #[test]
    fn tiny_forward_runs_and_qkv_merge_is_exact() {
        let cfg = SanaVideoTransformerConfig::tiny();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let map = recording_map(seen);
        let base = SanaVideoTransformer::load(cfg.clone(), SanaOptimizations::BASELINE, &map).unwrap();
        let merged = SanaVideoTransformer::load(
            cfg.clone(),
            SanaOptimizations {
                qkv_merge: true,
                ..SanaOptimizations::BASELINE
            },
            &map,
        )
        .unwrap();
        let (c, t, h, w) = (cfg.in_channels, 3, 4, 6);
        let lat = CudaTensor::from_vec(small_random("lat", &[c * t * h * w]), vec![1, c, t, h, w])
            .unwrap();
        let text = CudaTensor::from_vec(
            small_random("txt", &[5 * cfg.caption_channels]),
            vec![1, 5, cfg.caption_channels],
        )
        .unwrap();
        let cap = base.project_caption(&text).unwrap();
        let rope = base.rope(t, h, w).unwrap();
        let a = base.forward(&lat, &cap, 999.0, &rope).unwrap();
        let b = merged.forward(&lat, &cap, 999.0, &rope).unwrap();
        assert_eq!(a.shape, vec![1, c, t, h, w]);
        let (a, b) = (a.host_cow().unwrap(), b.host_cow().unwrap());
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(a.iter().any(|v| v.abs() > 1e-6));
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    /// Linear attention against a direct per-token evaluation of the
    /// reference formula.
    #[test]
    fn linear_attention_matches_the_reference_formula() {
        let (n, heads, dh) = (6usize, 2usize, 4usize);
        let width = heads * dh;
        let mk = |k: &str| {
            CudaTensor::from_vec(small_random(k, &[n * width]), vec![1, n, width]).unwrap()
        };
        let (q, k, v) = (mk("q"), mk("k"), mk("v"));
        // Arbitrary interleaved angles (each pair shares one).
        let ang: Vec<f32> = (0..n * dh)
            .map(|i| ((i / dh + 1) * ((i % dh) / 2 + 1)) as f32 * 0.3)
            .collect();
        let cos = CudaTensor::from_vec(ang.iter().map(|a| a.cos()).collect(), vec![n, dh]).unwrap();
        let sin = CudaTensor::from_vec(ang.iter().map(|a| a.sin()).collect(), vec![n, dh]).unwrap();
        let got = linear_attention(&q, &k, &v, heads, &cos, &sin, false).unwrap();
        let got = got.host_cow().unwrap();
        let (qh, kh, vh) = (q.host_cow().unwrap(), k.host_cow().unwrap(), v.host_cow().unwrap());
        let (ch, sh) = (cos.host_cow().unwrap(), sin.host_cow().unwrap());
        let relu = |x: f32| x.max(0.0);
        let rot = |x: &[f32], tok: usize| -> Vec<f32> {
            let mut o = vec![0f32; dh];
            for p in 0..dh / 2 {
                let (x1, x2) = (relu(x[2 * p]), relu(x[2 * p + 1]));
                let (c, s) = (ch[tok * dh + 2 * p], sh[tok * dh + 2 * p + 1]);
                o[2 * p] = x1 * c - x2 * s;
                o[2 * p + 1] = x1 * s + x2 * c;
            }
            o
        };
        for hd in 0..heads {
            let row = |t: &[f32], tok: usize| t[tok * width + hd * dh..][..dh].to_vec();
            let mut ksum = vec![0f32; dh];
            for tok in 0..n {
                for (s, x) in ksum.iter_mut().zip(row(&kh, tok)) {
                    *s += relu(x);
                }
            }
            for i in 0..n {
                let qi = row(&qh, i);
                let z: f32 = qi.iter().zip(&ksum).map(|(a, b)| relu(*a) * b).sum::<f32>() + 1e-15;
                let qr = rot(&qi, i);
                for c1 in 0..dh {
                    let mut acc = 0f32;
                    for j in 0..n {
                        let kr = rot(&row(&kh, j), j);
                        let vj = row(&vh, j);
                        let dot: f32 = qr.iter().zip(&kr).map(|(a, b)| a * b).sum();
                        acc += dot * vj[c1];
                    }
                    let want = acc / z;
                    let g = got[i * width + hd * dh + c1];
                    assert!((g - want).abs() < 1e-4 * (1.0 + want.abs()), "{g} vs {want}");
                }
            }
        }
    }
}
