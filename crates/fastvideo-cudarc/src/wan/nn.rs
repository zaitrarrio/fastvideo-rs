//! Layer primitives for the Wan graph.

use super::ops::host;
use super::stats;
use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct Linear {
    /// `[out, in]` F32 weight. Empty (`[0, in]`) when the weight lives only
    /// as bfloat16 on the device (see [`Self::weight_bf16`]).
    pub weight: CudaTensor,
    pub bias: Option<CudaTensor>,
    in_dim: usize,
    out_dim: usize,
    /// Fast mode on a Tensor Core GPU: the weight as bfloat16 on the device.
    /// Activations are cast in, multiplied with bf16 buffers throughout (the
    /// only form cuBLAS runs as true bf16 kernels on every GPU generation),
    /// and cast back with the bias and activation fused.
    #[cfg(feature = "cuda")]
    weight_bf16: Option<std::sync::Arc<cudarc::driver::CudaSlice<half::bf16>>>,
}

/// bfloat16 linears apply when the context runs bf16 GEMM math.
#[cfg(feature = "cuda")]
fn bf16_linears() -> bool {
    stats::device_expected()
        && super::device::global_device().is_some_and(|d| d.gemm_math == super::device::GemmMath::Bf16)
}

impl Linear {
    /// Wrap weight/bias and keep them on the device (a no-op on CPU runs).
    pub fn from_tensors(mut weight: CudaTensor, mut bias: Option<CudaTensor>) -> Result<Self> {
        if weight.rank() != 2 || bias.as_ref().is_some_and(|b| b.numel() != weight.shape[0]) {
            return Err(msg(format!("linear weight {:?} bias {:?}", weight.shape, bias.as_ref().map(|b| &b.shape))));
        }
        let (out_dim, in_dim) = (weight.shape[0], weight.shape[1]);
        if let Some(b) = &mut bias {
            b.pin_device()?;
        }
        #[cfg(feature = "cuda")]
        if bf16_linears() {
            let dev = super::device::global_device().ok_or_else(|| msg("no device"))?;
            let host: Vec<half::bf16> = weight.host_cow()?.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let slice = dev.stream.memcpy_stod(&host).map_err(|e| msg(e.to_string()))?;
            stats::record_h2d(host.len() / 2);
            return Ok(Self {
                weight: CudaTensor::from_vec(Vec::new(), vec![0, in_dim])?,
                bias,
                in_dim,
                out_dim,
                weight_bf16: Some(std::sync::Arc::new(slice)),
            });
        }
        weight.pin_device()?;
        Ok(Self {
            weight,
            bias,
            in_dim,
            out_dim,
            #[cfg(feature = "cuda")]
            weight_bf16: None,
        })
    }

    pub fn zeros(in_dim: usize, out_dim: usize, bias: bool) -> Self {
        Self::from_tensors(
            CudaTensor::zeros(&[out_dim, in_dim]),
            bias.then(|| CudaTensor::zeros(&[out_dim])),
        )
        .expect("zero linear")
    }

    pub fn load(
        map: &super::weights::WeightMap,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
    ) -> Result<Self> {
        Self::load_fused(map, &[prefix], in_dim, out_dim, has_bias)
    }

    /// Several same-input projections as one linear with rows stacked in
    /// `prefixes` order (fused QKV / KV): one GEMM instead of one per prefix.
    pub fn load_fused(
        map: &super::weights::WeightMap,
        prefixes: &[&str],
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let mut w = Vec::with_capacity(prefixes.len() * out_dim * in_dim);
        let mut b = Vec::with_capacity(prefixes.len() * out_dim);
        for prefix in prefixes {
            let wt = super::weights::cuda_tensor_shaped(map, &super::weights::join_key(prefix, "weight"), &[out_dim, in_dim])?;
            w.extend_from_slice(&wt.host_cow()?);
            if has_bias {
                let bt = super::weights::cuda_tensor_shaped(map, &super::weights::join_key(prefix, "bias"), &[out_dim])?;
                b.extend_from_slice(&bt.host_cow()?);
            }
        }
        let rows = prefixes.len() * out_dim;
        Self::from_tensors(
            CudaTensor::from_vec(w, vec![rows, in_dim])?,
            if has_bias { Some(CudaTensor::from_vec(b, vec![rows])?) } else { None },
        )
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    fn out_shape(&self, xs: &CudaTensor) -> Result<(usize, usize, Vec<usize>)> {
        let k = *xs.shape.last().ok_or_else(|| msg("linear on scalar"))?;
        if xs.rank() < 2 || k != self.in_dim {
            return Err(msg(format!("linear input {:?} for weight [{}, {}]", xs.shape, self.out_dim, self.in_dim)));
        }
        let mut shape = xs.shape.clone();
        *shape.last_mut().unwrap() = self.out_dim();
        Ok((xs.numel() / k, k, shape))
    }

    pub fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.forward_act(xs, false)
    }

    /// `gelu_tanh(W x + b)` with the bias add and activation in one launch.
    pub fn forward_gelu(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.forward_act(xs, true)
    }

    fn forward_act(&self, xs: &CudaTensor, gelu: bool) -> Result<CudaTensor> {
        let (m, k, out_shape) = self.out_shape(xs)?;
        let n = self.out_dim();
        #[cfg(feature = "cuda")]
        if let Some(w16) = &self.weight_bf16 {
            let x = xs.dev()?.ok_or_else(|| msg("bf16 linear without a device"))?;
            let x16 = super::ops::cast_f32_bf16_device(&x)?;
            let mut c16 = unsafe { super::device::global_device().ok_or_else(|| msg("no device"))?.stream.alloc::<half::bf16>((m * n).max(1)) }
                .map_err(|e| msg(e.to_string()))?;
            super::device::matmul_linear_wt_bf16(&x16, w16, &mut c16, m, k, n).map_err(|e| msg(e.to_string()))?;
            drop(x16);
            let bias = match &self.bias {
                Some(b) => Some(b.dev()?.ok_or_else(|| msg("bias"))?),
                None => None,
            };
            let c = super::ops::cast_bf16_f32_bias_act_device(&c16, bias.as_deref(), gelu)?;
            return CudaTensor::from_dev_result(c, out_shape);
        }
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(w)) = (xs.dev()?, self.weight.dev()?) {
            let mut c = super::ops::alloc((m * n).max(1))?;
            super::device::matmul_linear_wt_device(&x, &w, &mut c, m, k, n).map_err(|e| msg(e.to_string()))?;
            match (&self.bias, gelu) {
                (Some(b), true) => {
                    let b = b.dev()?.ok_or_else(|| msg("bias"))?;
                    super::ops::bias_gelu_inplace_device(&mut c, &b)?
                }
                (Some(b), false) => {
                    let b = b.dev()?.ok_or_else(|| msg("bias"))?;
                    super::ops::add_bias_inplace_device(&mut c, &b, 1)?
                }
                (None, true) => c = super::ops::unary_device(&c, super::ops::ElemUnary::GeluTanh)?,
                (None, false) => {}
            }
            return CudaTensor::from_dev_result(c, out_shape);
        }
        let x = xs.host_cow()?;
        let w = self.weight.host_cow()?;
        let b = self.bias.as_ref().map(|b| b.host_cow()).transpose()?;
        use rayon::prelude::*;
        let mut out = vec![0.0f32; m * n];
        out.par_chunks_mut(n.max(1)).enumerate().for_each(|(i, row)| {
            let xi = &x[i * k..(i + 1) * k];
            for (j, o) in row.iter_mut().enumerate() {
                let wj = &w[j * k..(j + 1) * k];
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += xi[t] * wj[t];
                }
                if let Some(b) = &b {
                    acc += b[j];
                }
                *o = if gelu { host::gelu_tanh(acc) } else { acc };
            }
        });
        CudaTensor::from_vec(out, out_shape)
    }
}

pub fn silu(xs: &CudaTensor) -> CudaTensor {
    xs.silu()
}

pub fn gelu_tanh(xs: &CudaTensor) -> CudaTensor {
    xs.gelu_tanh()
}

/// Approximate GELU via tanh variant (sufficient for CLIP MLP).
pub fn gelu(xs: &CudaTensor) -> CudaTensor {
    xs.gelu_tanh()
}

pub fn rms_norm(xs: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
    xs.rms_norm(weight, eps)
}

pub fn layer_norm(xs: &CudaTensor, eps: f32, weight: Option<&CudaTensor>, bias: Option<&CudaTensor>) -> Result<CudaTensor> {
    xs.layer_norm(eps, weight, bias)
}

pub fn softmax(xs: &CudaTensor, dim: isize) -> Result<CudaTensor> {
    xs.softmax(dim)
}

/// `[cos(t·f_i), sin(t·f_i)]` embedding. Built on host from scalar timesteps
/// (an input boundary, uploaded once per step by the first linear).
pub fn sinusoidal_timesteps(timesteps: &CudaTensor, dim: usize) -> Result<CudaTensor> {
    let half = dim / 2;
    let host = timesteps.host_cow()?;
    let n = host.len();
    let mut out = vec![0.0f32; n * dim];
    for (ti, &t) in host.iter().enumerate() {
        for i in 0..half {
            let freq = (-(10000f32.ln()) * (i as f32) / half as f32).exp();
            let arg = t * freq;
            out[ti * dim + i] = arg.cos();
            out[ti * dim + half + i] = arg.sin();
        }
    }
    CudaTensor::from_vec(out, vec![n, dim])
}

static SP_WORLD_CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static SDPA_BACKEND_CACHE: super::envflag::CachedString = super::envflag::CachedString::new();
static VSA_CACHE: super::envflag::CachedBool = super::envflag::CachedBool::new();

fn sp_world() -> usize {
    *SP_WORLD_CACHE.get_or_init(|| super::envflag::usize_flag("FASTVIDEO_SP_WORLD", 1).max(1))
}

/// `FASTVIDEO_SDPA`: `dense` (default: cuBLAS QKᵀ + softmax + PV, query-chunked
/// to bound memory), `flash` (tiled NVRTC kernel), `host` (CPU runs only).
pub fn sdpa_backend() -> String {
    SDPA_BACKEND_CACHE.get_or_init(|| super::envflag::string_flag("FASTVIDEO_SDPA", "dense"))
}

pub fn vsa_enabled() -> bool {
    VSA_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_VSA", false))
}

/// Scaled dot-product attention. q/k/v: [B, H, S, D].
pub fn scaled_dot_product_attention(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, scale: Option<f32>) -> Result<CudaTensor> {
    scaled_dot_product_attention_masked(q, k, v, scale, None)
}

/// Run `compute` on each of `world` query-sequence shards, one GPU per rank,
/// and gather the results (host-mediated; see [`super::sp`]).
fn dispatch_sharded(q: &CudaTensor, world: usize, compute: impl Fn(&CudaTensor) -> Result<CudaTensor> + Sync) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    if super::device::global_device().is_some() {
        return dispatch_sharded_multi_gpu(q, world, compute);
    }
    let mut shards = Vec::with_capacity(world);
    for rank in 0..world {
        shards.push(compute(&super::sp::shard_tensor(q, 2, rank, world)?)?);
    }
    super::sp::all_gather_seq(&shards, 2)
}

#[cfg(feature = "cuda")]
fn dispatch_sharded_multi_gpu(q: &CudaTensor, world: usize, compute: impl Fn(&CudaTensor) -> Result<CudaTensor> + Sync) -> Result<CudaTensor> {
    let results: Vec<std::sync::Mutex<Option<Result<CudaTensor>>>> = (0..world).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for rank in 0..world {
            let results = &results;
            let compute = &compute;
            scope.spawn(move || {
                let outcome = (|| -> Result<CudaTensor> {
                    let idx = super::sp::device_for_rank(rank, world);
                    let dev = super::device::device_for_index(idx).map_err(|e| msg(e.to_string()))?;
                    super::device::set_thread_device(Some(dev));
                    let qc = super::sp::shard_tensor(&q.clone(), 2, rank, world)?;
                    let mut out = compute(&qc)?;
                    out.ensure_host()?;
                    Ok(CudaTensor::from_vec(out.host_cow()?.into_owned(), out.shape.clone())?)
                })();
                super::device::set_thread_device(None);
                *results[rank].lock().expect("sdpa shard result lock") = Some(outcome);
            });
        }
    });
    let mut shards = Vec::with_capacity(world);
    for r in results {
        shards.push(r.into_inner().expect("sdpa shard result lock").ok_or_else(|| msg("sdpa shard produced no result"))??);
    }
    super::sp::all_gather_seq(&shards, 2)
}

pub fn scaled_dot_product_attention_masked(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: Option<&CudaTensor>,
) -> Result<CudaTensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(msg("sdpa expects BHSD"));
    }
    let run = |q: &CudaTensor| -> Result<CudaTensor> {
        if mask.is_none() {
            // FASTVIDEO_VSA now selects the real video sparse attention in
            // self-attention (see wan::vsa); this window-sparse host prototype
            // keeps its own opt-in so enabling VSA does not route cross
            // attention into a path with no device kernel.
            if sdpa_backend() == "sparse" {
                let window = super::envflag::usize_flag("FASTVIDEO_VSA_WINDOW", 128);
                return super::attn::block_sparse_sdpa(q, k, v, scale, window);
            }
            if sdpa_backend() == "host" {
                return super::attn::flash_style_sdpa_host(q, k, v, scale);
            }
            if sdpa_backend() == "flash" {
                if let Some(out) = super::attn::device_flash_sdpa(q, k, v, scale)? {
                    return Ok(out);
                }
            }
            if let Some(out) = super::attn::device_dense_sdpa(q, k, v, scale)? {
                return Ok(out);
            }
        }
        sdpa_composed(q, k, v, scale, mask)
    };
    let world = sp_world();
    if world > 1 {
        return dispatch_sharded(q, world, run);
    }
    run(q)
}

/// SDPA from tensor ops (masked attention, CPU runs). Every op here has a
/// device kernel, so on GPU runs this stays on the device.
fn sdpa_composed(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, scale: Option<f32>, mask: Option<&CudaTensor>) -> Result<CudaTensor> {
    let d = q.shape[3] as f32;
    let scale = scale.unwrap_or(1.0 / d.sqrt());
    let mut scores = q.matmul(&k.transpose(2, 3)?)?.try_mul_scalar(scale)?;
    if let Some(m) = mask {
        scores = scores.add(m)?;
    }
    scores.softmax(-1)?.matmul(v)
}

pub fn conv2d(xs: &CudaTensor, kernel: &CudaTensor, padding: usize, stride: usize) -> Result<CudaTensor> {
    xs.conv2d(kernel, None, padding, stride)
}

/// CPU-only helper guard: call before host-only algorithms with no kernel.
pub(crate) fn host_only_op(op: &'static str, detail: impl std::fmt::Display) -> Result<()> {
    stats::host_fallback(op, detail)
}
