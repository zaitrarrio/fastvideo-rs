//! WanTransformer3D on CudaTensor (device-resident when CUDA + residency are on).

use fastvideo_models::wan::WanVideoArchConfig;

use super::nn::{self, Linear};
use super::tensor::{CudaTensor, Result, TensorError};
use super::weights::{self, WeightMap};

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: CudaTensor,
    eps: f32,
}

impl RmsNorm {
    fn zeros(dim: usize, eps: f32) -> Self {
        let mut s = Self {
            weight: CudaTensor::ones(&[dim]),
            eps,
        };
        let _ = s.weight.pin_device();
        s
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, eps: f32) -> Result<Self> {
        let mut s = Self {
            weight: weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "weight"), &[dim])?,
            eps,
        };
        let _ = s.weight.pin_device();
        Ok(s)
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        nn::rms_norm(xs, &self.weight, self.eps)
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
    fn zeros(dim: usize, heads: usize, eps: f32, added_kv: Option<usize>) -> Self {
        let (add_k, add_v) = if let Some(extra) = added_kv {
            (
                Some(Linear::zeros(extra, dim, true)),
                Some(Linear::zeros(extra, dim, true)),
            )
        } else {
            (None, None)
        };
        Self {
            to_q: Linear::zeros(dim, dim, true),
            to_k: Linear::zeros(dim, dim, true),
            to_v: Linear::zeros(dim, dim, true),
            to_out: Linear::zeros(dim, dim, true),
            norm_q: RmsNorm::zeros(dim, eps),
            norm_k: RmsNorm::zeros(dim, eps),
            add_k,
            add_v,
            heads,
            dim_head: dim / heads,
        }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        dim: usize,
        heads: usize,
        eps: f32,
        added_kv: Option<usize>,
    ) -> Result<Self> {
        let (add_k, add_v) = if let Some(extra) = added_kv {
            (
                Some(Linear::load(
                    map,
                    &weights::join_key(prefix, "add_k_proj"),
                    extra,
                    dim,
                    true,
                )?),
                Some(Linear::load(
                    map,
                    &weights::join_key(prefix, "add_v_proj"),
                    extra,
                    dim,
                    true,
                )?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            to_q: Linear::load(map, &weights::join_key(prefix, "to_q"), dim, dim, true)?,
            to_k: Linear::load(map, &weights::join_key(prefix, "to_k"), dim, dim, true)?,
            to_v: Linear::load(map, &weights::join_key(prefix, "to_v"), dim, dim, true)?,
            to_out: Linear::load(map, &weights::join_key(prefix, "to_out.0"), dim, dim, true)?,
            norm_q: RmsNorm::load(map, &weights::join_key(prefix, "norm_q"), dim, eps)?,
            norm_k: RmsNorm::load(map, &weights::join_key(prefix, "norm_k"), dim, eps)?,
            add_k,
            add_v,
            heads,
            dim_head: dim / heads,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: Option<&CudaTensor>,
        rotary: Option<&(CudaTensor, CudaTensor)>,
        image: Option<&CudaTensor>,
        attn_mask: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let ctx = encoder.unwrap_or(hidden);
        let q = self.norm_q.forward(&self.to_q.forward(hidden)?)?;
        let mut k = self.norm_k.forward(&self.to_k.forward(ctx)?)?;
        let mut v = self.to_v.forward(ctx)?;
        if let (Some(add_k), Some(add_v), Some(img)) = (&self.add_k, &self.add_v, image) {
            let ik = add_k.forward(img)?;
            let iv = add_v.forward(img)?;
            k = CudaTensor::cat(&[&ik, &k], 1)?;
            v = CudaTensor::cat(&[&iv, &v], 1)?;
        }
        let (b, sq, _) = (q.shape[0], q.shape[1], q.shape[2]);
        let sk = k.shape[1];
        let mut q = q.reshape(vec![b, sq, self.heads, self.dim_head])?;
        k = k.reshape(vec![b, sk, self.heads, self.dim_head])?;
        v = v.reshape(vec![b, sk, self.heads, self.dim_head])?;
        if let Some((cos, sin)) = rotary {
            q = apply_rotary(&q, cos, sin)?;
            k = apply_rotary(&k, cos, sin)?;
        }
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let attn = nn::scaled_dot_product_attention_masked(&q, &k, &v, None, attn_mask)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape(vec![b, sq, self.heads * self.dim_head])?;
        self.to_out.forward(&attn)
    }
}

fn pair_last_dim(xs: &CudaTensor) -> Result<(CudaTensor, CudaTensor)> {
    let mut dims = xs.shape.clone();
    let d = dims.pop().ok_or_else(|| TensorError::Message("empty rotary".into()))?;
    if d % 2 != 0 {
        return Err(TensorError::Message("rotary last dim must be even".into()));
    }
    dims.push(d / 2);
    dims.push(2);
    let xs = xs.reshape(dims)?;
    let rank = xs.rank();
    let even = xs.narrow(rank - 1, 0, 1)?.squeeze(rank - 1)?;
    let odd = xs.narrow(rank - 1, 1, 1)?.squeeze(rank - 1)?;
    Ok((even, odd))
}

fn apply_rotary(xs: &CudaTensor, cos: &CudaTensor, sin: &CudaTensor) -> Result<CudaTensor> {
    // Fast device path: NVRTC rope_interleaved kernel when the last dim is even
    // and the buffers are device-fresh. Skips narrow/cat/mul host bounces.
    #[cfg(feature = "cuda")]
    {
        if let Some(out) = apply_rotary_device(xs, cos, sin) {
            return Ok(out);
        }
    }
    let (x1, x2) = pair_last_dim(xs)?;
    let (cos_e, _) = pair_last_dim(cos)?;
    let (_, sin_o) = pair_last_dim(sin)?;
    let out1 = x1.mul(&cos_e)?.sub(&x2.mul(&sin_o)?)?;
    let out2 = x1.mul(&sin_o)?.add(&x2.mul(&cos_e)?)?;
    let out1 = out1.unsqueeze(out1.rank())?;
    let out2 = out2.unsqueeze(out2.rank())?;
    let stacked = CudaTensor::cat(&[&out1, &out2], out1.rank() - 1)?;
    let mut out_dims = stacked.shape.clone();
    let pair = out_dims.pop().unwrap_or(2);
    let half = out_dims.pop().unwrap_or(0);
    out_dims.push(half * pair);
    stacked.reshape(out_dims)
}

#[cfg(feature = "cuda")]
fn apply_rotary_device(
    xs: &CudaTensor,
    cos: &CudaTensor,
    sin: &CudaTensor,
) -> Option<CudaTensor> {
    use super::device;
    if !super::resident::residency_enabled() {
        return None;
    }
    let d = xs.shape.last().copied()?;
    if d < 2 || d % 2 != 0 {
        return None;
    }
    let mut xs2 = xs.clone();
    let mut cos2 = cos.clone();
    let mut sin2 = sin.clone();
    xs2.ensure_device().ok();
    cos2.ensure_device().ok();
    sin2.ensure_device().ok();
    if !xs2.is_device_fresh() || !cos2.is_device_fresh() || !sin2.is_device_fresh() {
        return None;
    }
    let x_dev = xs2.device_slice()?;
    let c_dev = cos2.device_slice()?;
    let s_dev = sin2.device_slice()?;
    let _ = device::global_device()?;
    super::ops::rope_interleaved_device(x_dev, c_dev, s_dev, d).map(|dev| {
        CudaTensor::from_device_slice(dev, xs.shape.clone()).expect("shape ok")
    })
}

fn rotary_1d(dim: usize, seq: usize, theta: f64) -> Result<(CudaTensor, CudaTensor)> {
    let half = dim / 2;
    let mut cos = vec![0.0f32; seq * dim];
    let mut sin = vec![0.0f32; seq * dim];
    for p in 0..seq {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / dim as f64) as f32;
            let arg = p as f32 * freq;
            // repeat_interleave 2
            cos[p * dim + 2 * i] = arg.cos();
            cos[p * dim + 2 * i + 1] = arg.cos();
            sin[p * dim + 2 * i] = arg.sin();
            sin[p * dim + 2 * i + 1] = arg.sin();
        }
    }
    Ok((
        CudaTensor::from_vec(cos, vec![seq, dim])?,
        CudaTensor::from_vec(sin, vec![seq, dim])?,
    ))
}

fn wan_rope(
    cfg: &WanVideoArchConfig,
    frames: usize,
    height: usize,
    width: usize,
) -> Result<(CudaTensor, CudaTensor)> {
    let d = cfg.attention_head_dim;
    let h_dim = 2 * (d / 6);
    let w_dim = h_dim;
    let t_dim = d - h_dim - w_dim;
    let (cos_t, sin_t) = rotary_1d(t_dim, cfg.rope_max_seq_len, 10000.0)?;
    let (cos_h, sin_h) = rotary_1d(h_dim, cfg.rope_max_seq_len, 10000.0)?;
    let (cos_w, sin_w) = rotary_1d(w_dim, cfg.rope_max_seq_len, 10000.0)?;
    let ppf = frames / cfg.patch_size[0];
    let pph = height / cfg.patch_size[1];
    let ppw = width / cfg.patch_size[2];
    let seq = ppf * pph * ppw;
    let mut cos = vec![0.0f32; seq * d];
    let mut sin = vec![0.0f32; seq * d];
    let mut idx = 0usize;
    for ft in 0..ppf {
        for fh in 0..pph {
            for fw in 0..ppw {
                let mut o = 0usize;
                for i in 0..t_dim {
                    cos[idx * d + o] = cos_t.data[ft * t_dim + i];
                    sin[idx * d + o] = sin_t.data[ft * t_dim + i];
                    o += 1;
                }
                for i in 0..h_dim {
                    cos[idx * d + o] = cos_h.data[fh * h_dim + i];
                    sin[idx * d + o] = sin_h.data[fh * h_dim + i];
                    o += 1;
                }
                for i in 0..w_dim {
                    cos[idx * d + o] = cos_w.data[fw * w_dim + i];
                    sin[idx * d + o] = sin_w.data[fw * w_dim + i];
                    o += 1;
                }
                idx += 1;
            }
        }
    }
    Ok((
        CudaTensor::from_vec(cos, vec![1, seq, 1, d])?,
        CudaTensor::from_vec(sin, vec![1, seq, 1, d])?,
    ))
}

#[derive(Debug, Clone)]
struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn zeros(dim: usize, ffn_dim: usize) -> Self {
        Self {
            proj: Linear::zeros(dim, ffn_dim, true),
            out: Linear::zeros(ffn_dim, dim, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, ffn_dim: usize) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(map, &weights::join_key(prefix, "net.0.proj"), dim, ffn_dim, true)?,
            out: Linear::load(map, &weights::join_key(prefix, "net.2"), ffn_dim, dim, true)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        // Fast path: BF16 chain — proj(F32→BF16) → gelu_bf16 → out(BF16→F32).
        // Eliminates the F32 intermediate buffer and the cast round-trips that
        // the naive F32 path incurs.
        #[cfg(feature = "cuda")]
        if super::bf16_gemm::bf16_enabled() {
            if let Ok(result) = self.forward_bf16_chained(xs) {
                return Ok(result);
            }
        }
        let h = nn::gelu_tanh(&self.proj.forward(xs)?);
        self.out.forward(&h)
    }

    /// Chained BF16 FFN: proj(F32→BF16) → in-place gelu_bf16 → out(BF16→F32).
    /// Returns Err if BF16 is unavailable; caller falls back to F32 path.
    #[cfg(feature = "cuda")]
    fn forward_bf16_chained(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        use cudarc::driver::DevicePtrMut;
        let dev = super::device::global_device().ok_or_else(|| {
            TensorError::Message("no CUDA device".into())
        })?;
        let (m, out_shape_proj) = match xs.rank() {
            2 => (xs.shape[0], vec![xs.shape[0], self.proj.weight.shape[0]]),
            3 => (
                xs.shape[0] * xs.shape[1],
                vec![xs.shape[0], xs.shape[1], self.proj.weight.shape[0]],
            ),
            _ => return Err(TensorError::Message("ffn bf16: unsupported rank".into())),
        };
        let ffn_dim = self.proj.weight.shape[0];
        let dim = self.out.weight.shape[0];
        // Allocate BF16 intermediate buffer: m * ffn_dim elements.
        let mut h_bf16 = dev
            .stream
            .alloc_zeros::<half::bf16>(m * ffn_dim)
            .map_err(|e| TensorError::Message(e.to_string()))?;
        // proj: F32 → BF16 (with bias fused).
        self.proj.forward_into_bf16(xs, &mut h_bf16)?;
        // In-place gelu_bf16 on the BF16 buffer (reinterpret as u16 for the kernel).
        let h_bits: &mut cudarc::driver::CudaSlice<u16> = unsafe {
            &mut *((&mut h_bf16) as *mut cudarc::driver::CudaSlice<half::bf16>
                as *mut cudarc::driver::CudaSlice<u16>)
        };
        nn::gelu_tanh_bf16_inplace(h_bits)?;
        // out: BF16 → F32 (with bias fused).
        let out_shape = match xs.rank() {
            2 => vec![xs.shape[0], dim],
            3 => vec![xs.shape[0], xs.shape[1], dim],
            _ => unreachable!(),
        };
        let _ = out_shape_proj;
        self.out.forward_from_bf16(&h_bf16, m, out_shape)
    }
}

#[derive(Debug, Clone)]
struct TextProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl TextProjection {
    fn zeros(in_dim: usize, dim: usize) -> Self {
        Self {
            linear_1: Linear::zeros(in_dim, dim, true),
            linear_2: Linear::zeros(dim, dim, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(map, &weights::join_key(prefix, "linear_1"), in_dim, dim, true)?,
            linear_2: Linear::load(map, &weights::join_key(prefix, "linear_2"), dim, dim, true)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let h = nn::gelu_tanh(&self.linear_1.forward(xs)?);
        self.linear_2.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn zeros(in_dim: usize, dim: usize) -> Self {
        Self {
            linear_1: Linear::zeros(in_dim, dim, true),
            linear_2: Linear::zeros(dim, dim, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(map, &weights::join_key(prefix, "linear_1"), in_dim, dim, true)?,
            linear_2: Linear::load(map, &weights::join_key(prefix, "linear_2"), dim, dim, true)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let h = nn::silu(&self.linear_1.forward(xs)?);
        self.linear_2.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct ImageEmbedder {
    norm1_w: CudaTensor,
    norm1_b: CudaTensor,
    proj: Linear,
    out: Linear,
    norm2_w: CudaTensor,
    norm2_b: CudaTensor,
}

impl ImageEmbedder {
    fn load(map: &WeightMap, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(Self {
            norm1_w: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm1.weight"),
                &[in_dim],
            )?,
            norm1_b: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm1.bias"),
                &[in_dim],
            )?,
            proj: Linear::load(
                map,
                &weights::join_key(prefix, "ff.net.0.proj"),
                in_dim,
                in_dim,
                true,
            )?,
            out: Linear::load(
                map,
                &weights::join_key(prefix, "ff.net.2"),
                in_dim,
                out_dim,
                true,
            )?,
            norm2_w: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm2.weight"),
                &[out_dim],
            )?,
            norm2_b: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm2.bias"),
                &[out_dim],
            )?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let x = nn::layer_norm(xs, 1e-5, Some(&self.norm1_w), Some(&self.norm1_b))?;
        let x = nn::gelu_tanh(&self.proj.forward(&x)?);
        let x = self.out.forward(&x)?;
        nn::layer_norm(&x, 1e-5, Some(&self.norm2_w), Some(&self.norm2_b))
    }
}

#[derive(Debug, Clone)]
struct WanBlock {
    norm1_eps: f32,
    attn1: WanAttention,
    attn2: WanAttention,
    norm2_weight: CudaTensor,
    norm2_bias: CudaTensor,
    ffn: FeedForward,
    scale_shift_table: CudaTensor, // [1, 6, dim]
}

impl WanBlock {
    fn zeros(cfg: &WanVideoArchConfig) -> Self {
        let dim = cfg.hidden_size();
        Self {
            norm1_eps: cfg.eps,
            attn1: WanAttention::zeros(dim, cfg.num_attention_heads, cfg.eps, None),
            attn2: WanAttention::zeros(
                dim,
                cfg.num_attention_heads,
                cfg.eps,
                cfg.added_kv_proj_dim,
            ),
            norm2_weight: CudaTensor::ones(&[dim]),
            norm2_bias: CudaTensor::zeros(&[dim]),
            ffn: FeedForward::zeros(dim, cfg.ffn_dim),
            scale_shift_table: CudaTensor::zeros(&[1, 6, dim]),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &WanVideoArchConfig) -> Result<Self> {
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
            )?,
            attn2: WanAttention::load(
                map,
                &weights::join_key(prefix, "attn2"),
                dim,
                cfg.num_attention_heads,
                cfg.eps,
                cfg.added_kv_proj_dim,
            )?,
            norm2_weight: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm2.weight"),
                &[dim],
            )?,
            norm2_bias: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm2.bias"),
                &[dim],
            )?,
            ffn: FeedForward::load(map, &weights::join_key(prefix, "ffn"), dim, cfg.ffn_dim)?,
            scale_shift_table: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "scale_shift_table"),
                &[1, 6, dim],
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        temb: &CudaTensor,
        rotary: &(CudaTensor, CudaTensor),
        image: Option<&CudaTensor>,
        attn_mask: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let e = self.scale_shift_table.add(temb)?;
        let chunks = e.chunk(6, 1)?;
        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];
        let c_shift = &chunks[3];
        let c_scale = &chunks[4];
        let c_gate = &chunks[5];

        // Helper: squeeze [batch,1,dim] → [batch,dim] for fused LN+AdaLN kernel.
        let squeeze2d = |t: &CudaTensor| -> Result<CudaTensor> {
            if t.rank() == 3 && t.shape[1] == 1 {
                t.reshape(vec![t.shape[0], t.shape[2]])
            } else {
                Ok(t.clone())
            }
        };

        // Pre-SA norm: try fused LayerNorm + AdaLN modulate.
        let normed = {
            let sc2 = squeeze2d(scale_msa)?;
            let sh2 = squeeze2d(shift_msa)?;
            match nn::layer_norm_adaln(hidden, &sc2, &sh2, self.norm1_eps)? {
                Some(fused) => fused,
                None => {
                    let n = nn::layer_norm(hidden, self.norm1_eps, None, None)?;
                    n.modulate(scale_msa, shift_msa)?
                }
            }
        };
        let attn = self
            .attn1
            .forward(&normed, None, Some(rotary), None, attn_mask)?;
        let hidden = hidden.add(&attn.gate_mul(gate_msa)?)?;

        let normed = nn::layer_norm(
            &hidden,
            self.norm1_eps,
            Some(&self.norm2_weight),
            Some(&self.norm2_bias),
        )?;
        let attn = self
            .attn2
            .forward(&normed, Some(encoder), None, image, None)?;
        let hidden = hidden.add(&attn)?;

        // Pre-FFN norm: try fused LayerNorm + AdaLN modulate.
        let normed = {
            let sc2 = squeeze2d(c_scale)?;
            let sh2 = squeeze2d(c_shift)?;
            match nn::layer_norm_adaln(&hidden, &sc2, &sh2, self.norm1_eps)? {
                Some(fused) => fused,
                None => {
                    let n = nn::layer_norm(&hidden, self.norm1_eps, None, None)?;
                    n.modulate(c_scale, c_shift)?
                }
            }
        };

        // FFN (chained BF16 path attempted inside FeedForward::forward).
        let ff = self.ffn.forward(&normed)?;
        hidden.add(&ff.gate_mul(c_gate)?)
    }
}

/// Lazily create one extra CUDA stream and an event recorded on the primary
/// stream. The returned side stream has called `wait` on the event so any
/// work submitted to it will see the primary stream's prior launches. Returns
/// `None` when CUDA is unavailable or `FASTVIDEO_TWO_STREAMS=0`.
#[cfg(feature = "cuda")]
fn maybe_record_two_stream_event() -> Option<(cudarc::driver::CudaEvent, std::sync::Arc<cudarc::driver::CudaStream>)> {
    if !super::streams::two_streams_enabled() {
        return None;
    }
    let dev = super::device::global_device()?;
    let side = cudarc::driver::CudaContext::new_stream(&dev.ctx).ok()?;
    let event = dev.ctx.new_event(None).ok()?;
    if event.record(&dev.stream).is_err() {
        return None;
    }
    if side.wait(&event).is_err() {
        return None;
    }
    static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    super::log::info_once(
        &ONCE,
        format_args!("dit: 2-stream event recorded (FASTVIDEO_TWO_STREAMS=1)"),
    );
    Some((event, side))
}

#[derive(Debug, Clone)]
pub struct WanTransformer3D {
    pub cfg: WanVideoArchConfig,
    patch_weight: CudaTensor, // [dim, in_c, pt, ph, pw]
    patch_bias: CudaTensor,
    time_embedder: TimestepEmbedding,
    time_proj: Linear,
    text_embedder: TextProjection,
    image_embedder: Option<ImageEmbedder>,
    blocks: Vec<WanBlock>,
    proj_out: Linear,
    scale_shift_table: CudaTensor, // [1, 2, dim]
    freq_dim: usize,
    /// Memoized RoPE cos/sin tables keyed by `(seq_len, dim)`. Built once on the
    /// first step that matches the shape; subsequent denoise steps reuse the
    /// pinned device buffers without rebuilding or re-uploading.
    rotary_cache: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<(usize, usize), (CudaTensor, CudaTensor)>>>,
    /// cuGraph cache placeholder (see `captured_graph` docs). The actual graph
    /// objects are not stored here because `cudarc::driver::CudaGraph` is
    /// `!Send + !Sync`; this map is reserved for a future Arc-wrapped variant.
    /// Kept behind `#[allow(dead_code)]` so it compiles until the cache lands.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    graph_cache: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<(usize, usize), ()>>>,
}

impl WanTransformer3D {
    pub fn zeros(cfg: WanVideoArchConfig) -> Self {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            blocks.push(WanBlock::zeros(&cfg));
        }
        Self {
            patch_weight: CudaTensor::zeros(&[dim, cfg.in_channels, p[0], p[1], p[2]]),
            patch_bias: CudaTensor::zeros(&[dim]),
            time_embedder: TimestepEmbedding::zeros(cfg.freq_dim, dim),
            time_proj: Linear::zeros(dim, dim * 6, true),
            text_embedder: TextProjection::zeros(cfg.text_dim, dim),
            image_embedder: None,
            freq_dim: cfg.freq_dim,
            blocks,
            proj_out: Linear::zeros(dim, cfg.out_channels * p.iter().product::<usize>(), true),
            scale_shift_table: CudaTensor::zeros(&[1, 2, dim]),
            rotary_cache: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            #[cfg(feature = "cuda")]
            graph_cache: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            cfg,
        }
    }

    pub fn load(cfg: WanVideoArchConfig, map: &WeightMap) -> Result<Self> {
        Self::from_map(cfg, map)
    }

    pub fn from_map(cfg: WanVideoArchConfig, map: &WeightMap) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(WanBlock::load(map, &format!("blocks.{i}"), &cfg)?);
        }
        let image_embedder = match (cfg.image_dim, cfg.added_kv_proj_dim) {
            (Some(in_dim), Some(out_dim)) => {
                Some(ImageEmbedder::load(map, "condition_embedder.image_embedder", in_dim, out_dim)?)
            }
            _ => None,
        };
        Ok(Self {
            patch_weight: weights::cuda_tensor_shaped(
                map,
                "patch_embedding.weight",
                &[dim, cfg.in_channels, p[0], p[1], p[2]],
            )?,
            patch_bias: weights::cuda_tensor_shaped(map, "patch_embedding.bias", &[dim])?,
            time_embedder: TimestepEmbedding::load(
                map,
                "condition_embedder.time_embedder",
                cfg.freq_dim,
                dim,
            )?,
            time_proj: Linear::load(map, "condition_embedder.time_proj", dim, dim * 6, true)?,
            text_embedder: TextProjection::load(
                map,
                "condition_embedder.text_embedder",
                cfg.text_dim,
                dim,
            )?,
            image_embedder,
            proj_out: Linear::load(
                map,
                "proj_out",
                dim,
                cfg.out_channels * p.iter().product::<usize>(),
                true,
            )?,
            scale_shift_table: weights::cuda_tensor_shaped(map, "scale_shift_table", &[1, 2, dim])?,
            freq_dim: cfg.freq_dim,
            blocks,
            rotary_cache: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            #[cfg(feature = "cuda")]
            graph_cache: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            cfg,
        })
    }

    fn patch_embed(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        // xs: [B, C, T, H, W]
        let (b, c, t, h, w) = (
            xs.shape[0],
            xs.shape[1],
            xs.shape[2],
            xs.shape[3],
            xs.shape[4],
        );
        let p = self.cfg.patch_size;
        let x = xs
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(vec![b * t, c, h, w])?;
        let k = self
            .patch_weight
            .reshape(vec![self.cfg.hidden_size(), c * p[0], p[1], p[2]])?;
        let y = nn::conv2d(&x, &k, 0, p[1])?;
        let bias = self
            .patch_bias
            .reshape(vec![1, self.cfg.hidden_size(), 1, 1])?;
        let y = y.add(&bias)?;
        let (_, dim, hh, ww) = (y.shape[0], y.shape[1], y.shape[2], y.shape[3]);
        y.reshape(vec![b, t, dim, hh, ww])?
            .permute(&[0, 2, 1, 3, 4])?
            .flatten_from(2)?
            .transpose(1, 2)
    }

    pub fn forward(
        &self,
        latents: &CudaTensor,
        timestep: &CudaTensor,
        encoder: &CudaTensor,
    ) -> Result<CudaTensor> {
        self.forward_ctx(latents, timestep, encoder, None)
    }

    /// cuGraph placeholder hook. Returns `None` when cuGraph is disabled or no
    /// graph has been captured yet for the given shape. The capture itself is
    /// gated by `FASTVIDEO_CUGRAPH=1` and only succeeds when the forward path
    /// is fully device-resident (no D2H copies mid-block). Today the DiT
    /// block path bounces activations through host-side `.add/.mul/.cat` so
    /// capture would fail; this stub lets the env flag be honored without
    /// silently dropping it.
    ///
    /// **Limitation:** `cudarc::driver::CudaGraph` is `!Send + !Sync` and the
    /// capture/replay contract requires fixed input/output buffers. Until we
    /// switch the linear/attention helpers to write into a shared buffer
    /// pool, a real capture-then-replay path is unsafe. The hook exposes the
    /// cudarc 0.17 API surface (`begin_capture` / `end_capture` /
    /// `CudaGraph::launch`) so future work doesn't need another refactor.
    #[cfg(feature = "cuda")]
    pub fn captured_graph(
        &self,
        _seq_len: usize,
        _hidden_dim: usize,
    ) -> Option<()> {
        if !super::streams::cugraph_enabled() {
            return None;
        }
        // See the comment above; the map is intentionally unused for now.
        None
    }

    /// Try to begin capture of the next forward pass. The caller is expected
    /// to call `end_cugraph_capture` after the forward returns. No-op when
    /// cuGraph is disabled. Returns `true` if capture actually began.
    #[cfg(feature = "cuda")]
    pub fn begin_cugraph_capture(&self) -> bool {
        if !super::streams::cugraph_enabled() {
            return false;
        }
        let Some(dev) = super::device::global_device() else {
            return false;
        };
        let mode =
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
        if dev.stream.begin_capture(mode).is_err() {
            return false;
        }
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        super::log::info_once(
            &ONCE,
            format_args!("cugraph: capture started (FASTVIDEO_CUGRAPH=1)"),
        );
        true
    }

    /// End capture and return the resulting graph for replay. Returns `None`
    /// if capture failed or the env flag is off. The graph is **not** cached
    /// in `graph_cache` because `CudaGraph` is `!Send + !Sync`; see the
    /// `captured_graph` doc comment for the full rationale.
    #[cfg(feature = "cuda")]
    pub fn end_cugraph_capture(
        &self,
        _seq_len: usize,
        _hidden_dim: usize,
    ) -> Option<cudarc::driver::CudaGraph> {
        if !super::streams::cugraph_enabled() {
            return None;
        }
        let Some(dev) = super::device::global_device() else {
            return None;
        };
        let flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        match dev.stream.end_capture(flags) {
            Ok(Some(graph)) => {
                let mut map = self.graph_cache.lock().expect("graph cache lock");
                // Drop any prior graph for this shape; we replace it.
                map.remove(&(_seq_len, _hidden_dim));
                // We intentionally do not move `graph` into the cache because
                // `CudaGraph` is `!Send`. The next replay lives on this
                // thread; future refactor can wrap the graph in an Arc and
                // store it.
                drop(map);
                Some(graph)
            }
            _ => None,
        }
    }

    /// Get-or-build the RoPE cos/sin tables for the given `(t, h, w)`. The
    /// tables only depend on `(seq_len, dim)`, which stays constant across
    /// denoise steps, so we cache them once on first use and pin on device.
    pub fn rotary_for(
        &self,
        t: usize,
        h: usize,
        w: usize,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let ppf = t / self.cfg.patch_size[0];
        let pph = h / self.cfg.patch_size[1];
        let ppw = w / self.cfg.patch_size[2];
        let seq = ppf * pph * ppw;
        let dim = self.cfg.attention_head_dim;
        let key = (seq, dim);
        let mut map = self.rotary_cache.lock().expect("rotary cache lock");
        if let Some(pair) = map.get(&key) {
            return Ok(pair.clone());
        }
        let (mut cos, mut sin) = wan_rope(&self.cfg, t, h, w)?;
        let _ = cos.pin_device();
        let _ = sin.pin_device();
        map.insert(key, (cos.clone(), sin.clone()));
        Ok((cos, sin))
    }

    pub fn forward_ctx(
        &self,
        latents: &CudaTensor,
        timestep: &CudaTensor,
        encoder: &CudaTensor,
        image: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let (b, _c, t, h, w) = (
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
        );
        let rotary = self.rotary_for(t, h, w)?;
        let attn_mask = if self.cfg.causal {
            let mask = fastvideo_models::wan::causal_temporal_mask(&self.cfg, t, h, w);
            let seq = (mask.len() as f64).sqrt() as usize;
            Some(CudaTensor::from_vec(mask, vec![1, 1, seq, seq])?)
        } else {
            None
        };
        let mut hidden = self.patch_embed(latents)?;
        let temb_in = nn::sinusoidal_timesteps(timestep, self.freq_dim)?;
        let temb = self.time_embedder.forward(&temb_in)?;
        let timestep_proj = self
            .time_proj
            .forward(&nn::silu(&temb))?
            .reshape(vec![b, 6, self.cfg.hidden_size()])?;
        let encoder = self.text_embedder.forward(encoder)?;
        let image = match (image, &self.image_embedder) {
            (Some(img), Some(emb)) => Some(emb.forward(img)?),
            (Some(img), None) => Some(img.clone()),
            _ => None,
        };
        let image = image.as_ref();
        for block in &self.blocks {
            hidden = block.forward(
                &hidden,
                &encoder,
                &timestep_proj,
                &rotary,
                image,
                attn_mask.as_ref(),
            )?;
        }
        let temb_f = temb.unsqueeze(1)?;
        let ss = self.scale_shift_table.add(&temb_f)?;
        let chunks = ss.chunk(2, 1)?;
        let shift = &chunks[0];
        let scale = &chunks[1];
        hidden = nn::layer_norm(&hidden, self.cfg.eps, None, None)?;
        hidden = hidden.modulate(scale, shift)?;
        hidden = self.proj_out.forward(&hidden)?;
        let p = self.cfg.patch_size;
        let ppf = t / p[0];
        let pph = h / p[1];
        let ppw = w / p[2];
        let hidden = hidden.reshape(vec![
            b,
            ppf,
            pph,
            ppw,
            p[0],
            p[1],
            p[2],
            self.cfg.out_channels,
        ])?;
        hidden
            .permute(&[0, 7, 1, 4, 2, 5, 3, 6])?
            .reshape(vec![
                b,
                self.cfg.out_channels,
                ppf * p[0],
                pph * p[1],
                ppw * p[2],
            ])
    }
}
