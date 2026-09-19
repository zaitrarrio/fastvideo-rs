//! Layer primitives for the Wan graph.

use super::ops::host;
use super::stats;
use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// A weight held as E4M3 codes with one dequantization scale per output row.
#[derive(Debug)]
pub struct Fp8Rows {
    rows: usize,
    cols: usize,
    /// CPU runs: codes and scales on the host.
    host: Option<(Vec<u8>, Vec<f32>)>,
    #[cfg(feature = "cuda")]
    dev: Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<f32>)>,
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
    /// `FASTVIDEO_FP8=1`: the weight quantized once to E4M3 with a per-tensor
    /// scale. Activations are quantized per call and the product runs on FP8
    /// tensor cores. Only sound on a checkpoint trained to tolerate it — see
    /// [`fp8_linears`].
    #[cfg(feature = "cuda")]
    weight_fp8: Option<std::sync::Arc<super::fp8::Fp8Weight>>,
    /// Weight-only FP8 ([`Linear::load_fp8_rows`]): a per-instance choice for
    /// models that must be resident and do not fit at bf16. Dequantized to bf16
    /// before each GEMM, so activations and the GEMM itself are untouched.
    weight_fp8_rows: Option<std::sync::Arc<Fp8Rows>>,
}

/// FP8 linears are opt-in and never a default.
///
/// Per-tensor E4M3 is coarse — one scalar across a 1536x8960 weight — and a
/// stock checkpoint has no reason to survive it. A QAD checkpoint does, because
/// quantization-aware distillation trained it against exactly this error. So
/// this is a flag the caller sets knowing which weights are loaded, not
/// something inferred from the device.
#[cfg(feature = "cuda")]
fn fp8_linears() -> bool {
    static FLAG: super::envflag::CachedBool = super::envflag::CachedBool::new();
    FLAG.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_FP8", false)) && stats::device_expected()
}

/// bfloat16 linears apply when the context runs bf16 GEMM math.
#[cfg(feature = "cuda")]
fn bf16_linears() -> bool {
    stats::device_expected()
        && super::device::global_device().is_some_and(|d| d.gemm_math == super::device::GemmMath::Bf16)
}

/// Whether `Linear` weights load as device bfloat16 (2 bytes per parameter)
/// rather than f32: what a memory estimate needs to know.
pub fn bf16_linears_active() -> bool {
    #[cfg(feature = "cuda")]
    {
        bf16_linears()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
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
        if fp8_linears() {
            let dev = super::device::global_device().ok_or_else(|| msg("no device"))?;
            // A shape cuBLASLt cannot serve falls back to bf16/F32 rather than
            // failing the run, but says so once: a silent fallback would let an
            // FP8 benchmark quietly measure something else.
            match super::fp8::fp8_gemm_supported(&dev, out_dim, 16, in_dim) {
                Ok(()) => {
                    let host = weight.host_cow()?;
                    let q = super::fp8::Fp8Weight::quantize(&dev, &host, out_dim, in_dim)?;
                    stats::record_h2d(host.len() / 4);
                    return Ok(Self {
                        weight: CudaTensor::from_vec(Vec::new(), vec![0, in_dim])?,
                        bias,
                        in_dim,
                        out_dim,
                        weight_bf16: None,
                        weight_fp8: Some(std::sync::Arc::new(q)),
                        weight_fp8_rows: None,
                    });
                }
                Err(why) => {
                    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    super::log::info_once(&WARNED, format_args!("fp8 linear disabled: {why}"));
                }
            }
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
                weight_fp8: None,
                weight_fp8_rows: None,
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
            #[cfg(feature = "cuda")]
            weight_fp8: None,
            weight_fp8_rows: None,
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
        #[cfg(feature = "cuda")]
        if bf16_linears() && !fp8_linears() {
            if let Some(l) = Self::load_fused_bf16(map, prefixes, in_dim, out_dim, has_bias)? {
                return Ok(l);
            }
        }
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

    /// A lazily mapped checkpoint straight to a device bf16 weight: no f32
    /// copy of the weight ever exists, on the host or the device, and the host
    /// holds one tensor at a time. `None` when the map is not lazy, so eager
    /// and generated maps keep their path.
    #[cfg(feature = "cuda")]
    fn load_fused_bf16(
        map: &super::weights::WeightMap,
        prefixes: &[&str],
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
    ) -> Result<Option<Self>> {
        if map.lazy().is_none() {
            return Ok(None);
        }
        let dev = super::device::global_device().ok_or_else(|| msg("no device"))?;
        let mut host: Vec<half::bf16> = Vec::new();
        for prefix in prefixes {
            let key = super::weights::join_key(prefix, "weight");
            let (shape, values) = map.lazy_bf16(&key)?.ok_or_else(|| msg("lazy map lost its store"))?;
            if shape != [out_dim, in_dim] {
                return Err(msg(format!("key {key}: shape {shape:?} != expected {:?}", [out_dim, in_dim])));
            }
            if prefixes.len() == 1 {
                host = values;
            } else {
                host.extend_from_slice(&values);
            }
        }
        let slice = dev.stream.memcpy_stod(&host).map_err(|e| msg(e.to_string()))?;
        stats::record_h2d(host.len() / 2);
        drop(host);
        let rows = prefixes.len() * out_dim;
        let bias = if has_bias {
            let mut b = Vec::with_capacity(rows);
            for prefix in prefixes {
                let bt = super::weights::cuda_tensor_shaped(map, &super::weights::join_key(prefix, "bias"), &[out_dim])?;
                b.extend_from_slice(&bt.host_cow()?);
            }
            let mut b = CudaTensor::from_vec(b, vec![rows])?;
            b.pin_device()?;
            Some(b)
        } else {
            None
        };
        Ok(Some(Self {
            weight: CudaTensor::from_vec(Vec::new(), vec![0, in_dim])?,
            bias,
            in_dim,
            out_dim: rows,
            weight_bf16: Some(std::sync::Arc::new(slice)),
            weight_fp8: None,
            weight_fp8_rows: None,
        }))
    }

    /// Load `prefix.weight` as weight-only FP8: E4M3 codes, one scale per output
    /// row, one byte per parameter on the device. Quantized on the device, so a
    /// 130M-element weight costs a transient upload rather than seconds of host
    /// arithmetic. A per-instance choice, unlike the process-wide `FASTVIDEO_FP8`
    /// (which also quantizes activations and would reach the DiT loaded beside it).
    pub fn load_fp8_rows(
        map: &super::weights::WeightMap,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let key = super::weights::join_key(prefix, "weight");
        let mut bias = if has_bias {
            Some(super::weights::cuda_tensor_shaped(map, &super::weights::join_key(prefix, "bias"), &[out_dim])?)
        } else {
            None
        };
        if let Some(b) = &mut bias {
            b.pin_device()?;
        }
        let mut rows = Fp8Rows {
            rows: out_dim,
            cols: in_dim,
            host: None,
            #[cfg(feature = "cuda")]
            dev: None,
        };
        #[cfg(feature = "cuda")]
        if stats::device_expected() {
            let dev = super::device::global_device().ok_or_else(|| msg("no device"))?;
            let wf = match map.lazy_bf16(&key)? {
                // bf16 on disk: upload 2 bytes/param and widen on the device.
                Some((shape, values)) => {
                    if shape != [out_dim, in_dim] {
                        return Err(msg(format!("key {key}: shape {shape:?} != expected {:?}", [out_dim, in_dim])));
                    }
                    let w16 = dev.stream.memcpy_stod(&values).map_err(|e| msg(e.to_string()))?;
                    stats::record_h2d(values.len() / 2);
                    super::ops::cast_bf16_f32_bias_act_device(&w16, None, false)?
                }
                None => {
                    let w = super::weights::cuda_tensor_shaped(map, &key, &[out_dim, in_dim])?;
                    let host = w.host_cow()?;
                    stats::record_h2d(host.len());
                    dev.stream.memcpy_stod(&host[..]).map_err(|e| msg(e.to_string()))?
                }
            };
            rows.dev = Some(super::ops::fp8_rows_quantize_device(&wf, out_dim, in_dim)?);
        }
        #[cfg(feature = "cuda")]
        let on_device = rows.dev.is_some();
        #[cfg(not(feature = "cuda"))]
        let on_device = false;
        if !on_device {
            let w = super::weights::cuda_tensor_shaped(map, &key, &[out_dim, in_dim])?;
            rows.host = Some(host::fp8_rows_quantize(&w.host_cow()?, out_dim, in_dim));
        }
        Ok(Self {
            weight: CudaTensor::from_vec(Vec::new(), vec![0, in_dim])?,
            bias,
            in_dim,
            out_dim,
            #[cfg(feature = "cuda")]
            weight_bf16: None,
            #[cfg(feature = "cuda")]
            weight_fp8: None,
            weight_fp8_rows: Some(std::sync::Arc::new(rows)),
        })
    }

    /// Whether this linear holds its weight as per-row FP8.
    pub fn is_fp8_rows(&self) -> bool {
        self.weight_fp8_rows.is_some()
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
        if let Some(wq) = &self.weight_fp8 {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let dev = super::device::global_device().ok_or_else(|| msg("no device"))?;
            let ltc = super::fp8::lt_context(&dev)?;
            let x = xs.dev()?.ok_or_else(|| msg("fp8 linear without a device"))?;
            let (xq, x_scale) = super::ops::quantize_e4m3_device(&x)?;
            let mut c = super::ops::alloc((m * n).max(1))?;
            {
                let (wp, _gw) = wq.data.device_ptr(&dev.stream);
                let (wsp, _gws) = wq.scale.device_ptr(&dev.stream);
                let (xp, _gx) = xq.device_ptr(&dev.stream);
                let (xsp, _gxs) = x_scale.device_ptr(&dev.stream);
                let (cp, _gc) = c.device_ptr_mut(&dev.stream);
                // cuBLASLt is column-major: a row-major [m, n] result with
                // leading dimension n is a column-major [n, m] with the same
                // leading dimension, so the weight goes in as A and the
                // activations as B, and (m, n, k) becomes (out, tokens, in).
                unsafe { super::fp8::gemm_e4m3(&dev, &ltc, n, m, k, wp, wsp, xp, xsp, cp)? };
            }
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
        #[cfg(feature = "cuda")]
        let dequantized = match self.weight_fp8_rows.as_deref() {
            Some(Fp8Rows { dev: Some((q, scales)), cols, .. }) => {
                Some(std::sync::Arc::new(super::ops::fp8_rows_dequant_bf16_device(q, scales, *cols)?))
            }
            _ => None,
        };
        #[cfg(feature = "cuda")]
        if let Some(w16) = dequantized.as_ref().or(self.weight_bf16.as_ref()) {
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
        let w: std::borrow::Cow<'_, [f32]> = match self.weight_fp8_rows.as_deref() {
            Some(Fp8Rows { host: Some((q, scales)), cols, .. }) => std::borrow::Cow::Owned(host::fp8_rows_dequant(q, scales, *cols)),
            Some(_) => return Err(msg("fp8 linear: device weight but no device tensor to multiply")),
            None => self.weight.host_cow()?,
        };
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

#[cfg(test)]
mod fp8_rows_tests {
    use super::*;
    use crate::wan::weights::WeightMap;

    fn map() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let n: usize = shape.iter().product();
            let seed = key.len() as f32;
            // Rows of very different magnitude: what per-row scales are for.
            (0..n).map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * (1.0 + (i / 7) as f32 * seed)).collect()
        })
    }

    /// The forward is a plain matmul with the dequantized weight, exactly; the
    /// dequantized weight is within E4M3's step of the original, per row; and a
    /// dead row neither divides by zero nor produces a NaN.
    #[test]
    fn fp8_rows_is_a_matmul_with_the_dequantized_weight() {
        let (i, o) = (7usize, 5usize);
        let lin = Linear::load_fp8_rows(&map(), "p", i, o, true).unwrap();
        assert!(lin.is_fp8_rows());
        let w = crate::wan::weights::cuda_tensor_shaped(&map(), "p.weight", &[o, i]).unwrap().host_cow().unwrap().into_owned();
        let b = crate::wan::weights::cuda_tensor_shaped(&map(), "p.bias", &[o]).unwrap().host_cow().unwrap().into_owned();
        let (q, scales) = host::fp8_rows_quantize(&w, o, i);
        let wd = host::fp8_rows_dequant(&q, &scales, i);
        for r in 0..o {
            let amax = w[r * i..(r + 1) * i].iter().fold(0f32, |a, v| a.max(v.abs()));
            for c in 0..i {
                let (orig, deq) = (w[r * i + c], wd[r * i + c]);
                assert!((orig - deq).abs() <= amax * 0.07 + 1e-6, "row {r} col {c}: {orig} -> {deq}");
            }
        }
        let x: Vec<f32> = (0..3 * i).map(|k| (k as f32 * 0.37).sin()).collect();
        let y = lin.forward(&CudaTensor::from_vec(x.clone(), vec![3, i]).unwrap()).unwrap();
        let got = y.host_cow().unwrap();
        for t in 0..3 {
            for r in 0..o {
                let want: f32 = (0..i).map(|c| x[t * i + c] * wd[r * i + c]).sum::<f32>() + b[r];
                assert!((got[t * o + r] - want).abs() <= 1e-4 * want.abs().max(1.0), "token {t} row {r}");
            }
        }
        let (qz, sz) = host::fp8_rows_quantize(&[0.0; 6], 2, 3);
        assert_eq!((qz, sz), (vec![0u8; 6], vec![1.0, 1.0]));
    }
}
