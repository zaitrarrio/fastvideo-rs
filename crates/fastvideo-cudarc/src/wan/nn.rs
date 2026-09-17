//! Layer primitives matching Candle `fastvideo_models::nn` for the host Wan graph.

#[cfg(feature = "cuda")]
use std::sync::atomic::AtomicBool;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

use super::resident::residency_enabled;
use super::tensor::{CudaTensor, Result, TensorError};

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: CudaTensor, // [out, in]
    pub bias: Option<CudaTensor>,
    /// Cached BF16 device weights (filled on first successful upload).
    #[cfg(feature = "cuda")]
    weight_bf16: Arc<OnceLock<cudarc::driver::CudaSlice<half::bf16>>>,
    /// Resident BF16 activation scratch + identity of the last f32 input.
    /// Avoids re-casting X→BF16 between consecutive DiT steps whose input
    /// activation buffer has not changed. Gated by `FASTVIDEO_BF16_ACT=1`.
    #[cfg(feature = "cuda")]
    bf16_act: Arc<Mutex<Option<Bf16Activation>>>,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Bf16Activation {
    /// u16-bits buffer (length = input.len()); same bytes as `bf16`.
    bits: cudarc::driver::CudaSlice<u16>,
    /// Identity hash of the last f32 input (ptr ^ len mix).
    identity: u64,
    /// Shape dims (for shape-change invalidation).
    shape_len: usize,
}

impl Linear {
    fn new_empty_cache(weight: CudaTensor, bias: Option<CudaTensor>) -> Self {
        Self {
            weight,
            bias,
            #[cfg(feature = "cuda")]
            weight_bf16: Arc::new(OnceLock::new()),
            #[cfg(feature = "cuda")]
            bf16_act: Arc::new(Mutex::new(None)),
        }
    }

    pub fn zeros(in_dim: usize, out_dim: usize, bias: bool) -> Self {
        let mut s = Self::new_empty_cache(
            CudaTensor::zeros(&[out_dim, in_dim]),
            if bias {
                Some(CudaTensor::zeros(&[out_dim]))
            } else {
                None
            },
        );
        let _ = s.pin();
        s
    }

    pub fn load(
        map: &super::weights::WeightMap,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let w_key = super::weights::join_key(prefix, "weight");
        let weight = super::weights::cuda_tensor_shaped(map, &w_key, &[out_dim, in_dim])?;
        let bias = if has_bias {
            let b_key = super::weights::join_key(prefix, "bias");
            Some(super::weights::cuda_tensor_shaped(map, &b_key, &[out_dim])?)
        } else {
            None
        };
        let mut s = Self::new_empty_cache(weight, bias);
        let _ = s.pin();
        Ok(s)
    }

    /// Pin weight/bias on device when residency is enabled; cache BF16 once.
    pub fn pin(&mut self) -> Result<()> {
        if !residency_enabled() {
            return Ok(());
        }
        self.weight.pin_device()?;
        if let Some(b) = &mut self.bias {
            b.pin_device()?;
        }
        #[cfg(feature = "cuda")]
        {
            let _ = self.ensure_bf16_cache();
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_bf16_cache(&self) -> Option<&cudarc::driver::CudaSlice<half::bf16>> {
        if !super::bf16_gemm::bf16_enabled() {
            return None;
        }
        if let Some(w) = self.weight_bf16.get() {
            return Some(w);
        }
        let host = self.weight.host_cow().ok()?;
        let uploaded = super::bf16_gemm::upload_bf16(&host).ok()?;
        let _ = self.weight_bf16.set(uploaded);
        static ONCE: AtomicBool = AtomicBool::new(false);
        super::log::debug_once(
            &ONCE,
            format_args!("linear: cached BF16 weights"),
        );
        self.weight_bf16.get()
    }

    /// Run the BF16 GEMM, reusing the per-Linear cached X→BF16 cast buffer
    /// when the input activation identity matches the previous call (avoids
    /// the f32→bf16 cast kernel launch on the hot DiT step path).
    #[cfg(feature = "cuda")]
    fn bf16_matmul(
        &self,
        x_dev: &cudarc::driver::CudaSlice<f32>,
        w_bf16: &cudarc::driver::CudaSlice<half::bf16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        if !super::bf16_gemm::bf16_act_cache_enabled() {
            return super::bf16_gemm::matmul_linear_wt_bf16_to_f32(
                x_dev, w_bf16, m, k, n,
            )
            .map_err(|e| TensorError::Message(e.to_string()));
        }
        let dev = super::device::global_device().ok_or_else(|| {
            TensorError::Message("no global CUDA device context".into())
        })?;
        let identity = super::bf16_gemm::activation_identity(x_dev);
        let mut guard = self.bf16_act.lock().expect("bf16 act lock");
        let reuse = match guard.as_ref() {
            Some(existing) => existing.identity == identity && existing.bits.len() == x_dev.len(),
            None => false,
        };
        if reuse {
            super::bf16_gemm::matmul_linear_wt_bf16_to_f32_with_bits(
                x_dev,
                w_bf16,
                &mut guard.as_mut().unwrap().bits,
                m,
                k,
                n,
            )
            .map_err(|e| TensorError::Message(e.to_string()))
        } else {
            let mut bits = dev
                .stream
                .alloc_zeros::<u16>(x_dev.len())
                .map_err(|e| TensorError::Message(e.to_string()))?;
            let c = super::bf16_gemm::matmul_linear_wt_bf16_to_f32_with_bits(
                x_dev,
                w_bf16,
                &mut bits,
                m,
                k,
                n,
            )
            .map_err(|e| TensorError::Message(e.to_string()))?;
            *guard = Some(Bf16Activation {
                bits,
                identity,
                shape_len: x_dev.len(),
            });
            Ok(c)
        }
    }

    /// Forward to a caller-owned bf16 output buffer (skips the final f32 cast).
    /// Used to chain stacked linears (e.g., W1 → W2 in an FFN) so W2 can read
    /// bf16 directly without re-casting W1's f32 output. Returns `Ok(())` on
    /// success; on bf16-disabled path returns `Err` and the caller should
    /// fall back to [`Self::forward`].
    #[cfg(feature = "cuda")]
    pub fn forward_into_bf16(
        &self,
        xs: &CudaTensor,
        c_bf16: &mut cudarc::driver::CudaSlice<half::bf16>,
    ) -> Result<()> {
        if !super::bf16_gemm::bf16_enabled() {
            return Err(TensorError::Message(
                "forward_into_bf16: BF16 disabled (FASTVIDEO_BF16=0)".into(),
            ));
        }
        let Some(w_bf16) = self.ensure_bf16_cache() else {
            return Err(TensorError::Message(
                "forward_into_bf16: weight bf16 cache unavailable".into(),
            ));
        };
        let dev = super::device::global_device().ok_or_else(|| {
            TensorError::Message("no global CUDA device context".into())
        })?;
        let (m, k, n) = match xs.rank() {
            2 => (xs.shape[0], xs.shape[1], self.weight.shape[0]),
            3 => (
                xs.shape[0] * xs.shape[1],
                xs.shape[2],
                self.weight.shape[0],
            ),
            4 => (
                xs.shape[0] * xs.shape[1] * xs.shape[2],
                xs.shape[3],
                self.weight.shape[0],
            ),
            r => {
                return Err(TensorError::Message(format!(
                    "forward_into_bf16 unsupported rank {r}"
                )))
            }
        };
        if k != self.weight.shape[1] {
            return Err(TensorError::Message(format!(
                "forward_into_bf16 inner dim {k} vs {}",
                self.weight.shape[1]
            )));
        }
        let mut x = xs.clone();
        x.ensure_device()?;
        let Some(x_dev) = x.device_slice() else {
            return Err(TensorError::Message(
                "forward_into_bf16: activation not on device".into(),
            ));
        };
        if x_dev.len() != m * k || c_bf16.len() != m * n {
            return Err(TensorError::Message(
                "forward_into_bf16: size mismatch".into(),
            ));
        }
        let mut bits = dev
            .stream
            .alloc_zeros::<u16>(x_dev.len())
            .map_err(|e| TensorError::Message(e.to_string()))?;
        super::bf16_gemm::matmul_linear_wt_bf16_to_bf16(
            x_dev, w_bf16, &mut bits, c_bf16, m, k, n,
        )
        .map_err(|e| TensorError::Message(e.to_string()))?;
        Ok(())
    }

    pub fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = self.forward_resident(xs)? {
                return Ok(out);
            }
        }
        let w_t = self.weight.transpose(0, 1)?;
        let mut out = match xs.rank() {
            2 => xs.matmul(&w_t)?,
            3 => {
                let (b, s, i) = (xs.shape[0], xs.shape[1], xs.shape[2]);
                let flat = xs.reshape(vec![b * s, i])?;
                flat.matmul(&w_t)?.reshape(vec![b, s, self.weight.shape[0]])?
            }
            4 => {
                let (b1, b2, s, i) = (xs.shape[0], xs.shape[1], xs.shape[2], xs.shape[3]);
                let flat = xs.reshape(vec![b1 * b2 * s, i])?;
                flat.matmul(&w_t)?
                    .reshape(vec![b1, b2, s, self.weight.shape[0]])?
            }
            _ => {
                return Err(TensorError::Message(format!(
                    "linear unsupported rank {}",
                    xs.rank()
                )));
            }
        };
        if let Some(bias) = &self.bias {
            let mut bshape = vec![1; out.rank()];
            bshape[out.rank() - 1] = bias.shape[0];
            let b = bias.reshape(bshape)?;
            out = out.add(&b)?;
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn forward_resident(&self, xs: &CudaTensor) -> Result<Option<CudaTensor>> {
        if !residency_enabled() {
            return Ok(None);
        }
        let Some(dev) = super::device::global_device() else {
            return Ok(None);
        };
        let mut w = self.weight.clone();
        w.ensure_device()?;
        let Some(w_dev) = w.device_slice() else {
            return Ok(None);
        };
        let (m, k, n, out_shape) = match xs.rank() {
            2 => {
                let m = xs.shape[0];
                let k = xs.shape[1];
                (m, k, self.weight.shape[0], vec![m, self.weight.shape[0]])
            }
            3 => {
                let (b, s, i) = (xs.shape[0], xs.shape[1], xs.shape[2]);
                (
                    b * s,
                    i,
                    self.weight.shape[0],
                    vec![b, s, self.weight.shape[0]],
                )
            }
            4 => {
                let (b1, b2, s, i) = (xs.shape[0], xs.shape[1], xs.shape[2], xs.shape[3]);
                (
                    b1 * b2 * s,
                    i,
                    self.weight.shape[0],
                    vec![b1, b2, s, self.weight.shape[0]],
                )
            }
            _ => return Ok(None),
        };
        if k != self.weight.shape[1] {
            return Ok(None);
        }
        let mut x = xs.clone();
        x.ensure_device()?;
        let Some(x_dev) = x.device_slice() else {
            return Ok(None);
        };
        if x_dev.len() != m * k {
            return Ok(None);
        }
        let mut c_dev = if let Some(w_bf16) = self.ensure_bf16_cache() {
            match self.bf16_matmul(x_dev, w_bf16, m, k, n) {
                Ok(c) => {
                    static BF16_ONCE: AtomicBool = AtomicBool::new(false);
                    super::log::debug_once(
                        &BF16_ONCE,
                        format_args!("linear: BF16 GemmEx m={m} k={k} n={n}"),
                    );
                    c
                }
                Err(e) => {
                    static FB_ONCE: AtomicBool = AtomicBool::new(false);
                    super::log::info_once(
                        &FB_ONCE,
                        format_args!("linear: BF16 GemmEx failed ({e}); using F32 cuBLAS"),
                    );
                    let mut c = dev
                        .stream
                        .alloc_zeros::<f32>(m * n)
                        .map_err(|e| TensorError::Message(e.to_string()))?;
                    super::device::matmul_linear_wt_device(x_dev, w_dev, &mut c, m, k, n)
                        .map_err(|e| TensorError::Message(e.to_string()))?;
                    c
                }
            }
        } else {
            static F32_ONCE: AtomicBool = AtomicBool::new(false);
            super::log::debug_once(
                &F32_ONCE,
                format_args!("linear: F32 cuBLAS m={m} k={k} n={n}"),
            );
            let mut c = dev
                .stream
                .alloc_zeros::<f32>(m * n)
                .map_err(|e| TensorError::Message(e.to_string()))?;
            super::device::matmul_linear_wt_device(x_dev, w_dev, &mut c, m, k, n)
                .map_err(|e| TensorError::Message(e.to_string()))?;
            c
        };
        if let Some(bias) = &self.bias {
            let mut b = bias.clone();
            b.ensure_device()?;
            if let Some(b_dev) = b.device_slice() {
                super::ops::add_bias_last_inplace(&mut c_dev, b_dev)?;
            }
        }
        Ok(Some(CudaTensor::from_device_slice(c_dev, out_shape)?))
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

pub fn layer_norm(
    xs: &CudaTensor,
    eps: f32,
    weight: Option<&CudaTensor>,
    bias: Option<&CudaTensor>,
) -> Result<CudaTensor> {
    xs.layer_norm(eps, weight, bias)
}

pub fn softmax(xs: &CudaTensor, dim: isize) -> Result<CudaTensor> {
    xs.softmax(dim)
}

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

/// Scaled dot-product attention. q/k/v: [B, H, S, D]
///
/// Backend selection (`FASTVIDEO_SDPA`):
/// - `flash` (default): GPU dense SDPA when CUDA is live (chunked QK); host online-softmax fallback
/// - `dense`: same GPU path / materialize QK^T
/// - `sparse` / VSA: block-sparse local+global window (`FASTVIDEO_VSA=1` forces this)
/// - `host`: force CPU flash-style path
///
/// Sequence parallel: `FASTVIDEO_SP_WORLD=N` shards the query sequence.
pub fn scaled_dot_product_attention(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    scaled_dot_product_attention_masked(q, k, v, scale, None)
}

const SDPA_QUERY_CHUNK: usize = 512;

static SP_WORLD_CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static SDPA_BACKEND_CACHE: super::envflag::CachedString = super::envflag::CachedString::new();
static VSA_CACHE: super::envflag::CachedBool = super::envflag::CachedBool::new();

fn sp_world() -> usize {
    *SP_WORLD_CACHE.get_or_init(|| super::envflag::usize_flag("FASTVIDEO_SP_WORLD", 1).max(1))
}

/// Cached (see [`super::resident::residency_enabled`] doc for why): consulted
/// multiple times per attention call.
fn sdpa_backend() -> String {
    SDPA_BACKEND_CACHE.get_or_init(|| super::envflag::string_flag("FASTVIDEO_SDPA", "flash"))
}

fn vsa_enabled() -> bool {
    VSA_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_VSA", false))
}

/// Test-only: clears the `FASTVIDEO_SDPA` cache so a test that flips the env
/// var mid-process (see `attn::tests::flash_style_matches_dense_small`) sees
/// the new value.
#[cfg(test)]
pub(crate) fn reset_sdpa_backend_cache_for_test() {
    SDPA_BACKEND_CACHE.reset();
}

/// Run `compute` on each of `world` query-sequence shards of `q` and gather
/// the results back into a full-sequence tensor.
///
/// When a CUDA device is live, this dispatches each rank's shard to its own
/// physical GPU (`super::sp::device_for_rank`) on its own OS thread, running
/// genuinely in parallel — this is what `--num-gpus N` is supposed to do.
/// Previously `world > 1` ran every rank sequentially on the single global
/// device (see the old sp.rs doc comment / decision log): smaller batched
/// GEMMs than the unsharded call, zero benefit from extra GPUs, and real
/// overhead from the shard/gather step. `device_for_rank` existed but was
/// never called.
///
/// Each rank downloads its shard to host (`ensure_host`) *on its own thread*
/// before returning: a `CudaSlice` belongs to the `CudaContext` that
/// allocated it, so touching it from the joining thread (a different
/// context) would be operating on a foreign-context pointer. The gather then
/// runs on host data and re-uploads to the caller's own device — exactly the
/// "host-mediated gather" `sp.rs` always documented as the plan, now
/// actually wired to real per-rank devices instead of one shared one. NCCL
/// P2P (skipping the host round trip) remains a follow-up.
fn dispatch_sharded(
    q: &CudaTensor,
    world: usize,
    compute: impl Fn(&CudaTensor) -> Result<CudaTensor> + Sync,
) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    {
        if super::device::global_device().is_some() {
            return dispatch_sharded_multi_gpu(q, world, compute);
        }
    }
    // No live CUDA device: nothing to parallelize across (host-only build or
    // CPU path), so a plain sequential shard loop is already correct.
    let mut shards = Vec::with_capacity(world);
    for rank in 0..world {
        let qc = super::sp::shard_tensor(q, 2, rank, world)?;
        shards.push(compute(&qc)?);
    }
    super::sp::all_gather_seq(&shards, 2)
}

#[cfg(feature = "cuda")]
fn dispatch_sharded_multi_gpu(
    q: &CudaTensor,
    world: usize,
    compute: impl Fn(&CudaTensor) -> Result<CudaTensor> + Sync,
) -> Result<CudaTensor> {
    let results: Vec<std::sync::Mutex<Option<Result<CudaTensor>>>> =
        (0..world).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for rank in 0..world {
            let results = &results;
            let compute = &compute;
            scope.spawn(move || {
                let outcome = (|| -> Result<CudaTensor> {
                    let idx = super::sp::device_for_rank(rank, world);
                    let dev = super::device::device_for_index(idx)
                        .map_err(|e| TensorError::Message(e.to_string()))?;
                    super::device::set_thread_device(Some(dev));
                    let qc = super::sp::shard_tensor(q, 2, rank, world)?;
                    let mut out = compute(&qc)?;
                    out.ensure_host()?;
                    Ok(out)
                })();
                super::device::set_thread_device(None);
                *results[rank].lock().expect("sdpa shard result lock") = Some(outcome);
            });
        }
    });
    let mut shards = Vec::with_capacity(world);
    for r in results {
        let outcome = r
            .into_inner()
            .expect("sdpa shard result lock")
            .ok_or_else(|| {
                TensorError::Message("sdpa shard thread did not produce a result".into())
            })?;
        shards.push(outcome?);
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
        return Err(TensorError::Message("sdpa expects BHSD".into()));
    }
    if mask.is_none() {
        let world = sp_world();
        let run = |q: &CudaTensor| -> Result<CudaTensor> {
            if vsa_enabled() || sdpa_backend() == "sparse" {
                static WINDOW_CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                let window =
                    *WINDOW_CACHE.get_or_init(|| super::envflag::usize_flag("FASTVIDEO_VSA_WINDOW", 128));
                return super::attn::block_sparse_sdpa(q, k, v, scale, window);
            }
            if sdpa_backend() == "host" {
                return super::attn::flash_style_sdpa_host(q, k, v, scale);
            }
            // flash / dense / default → GPU-first
            if let Some(out) = super::attn::device_dense_sdpa(q, k, v, scale)? {
                return Ok(out);
            }
            if sdpa_backend() != "dense" {
                return super::attn::flash_style_sdpa(q, k, v, scale);
            }
            sdpa_qk(q, k, v, scale, None)
        };
        if world > 1 {
            return dispatch_sharded(q, world, run);
        }
        return run(q);
    }
    let world = sp_world();
    if world > 1 {
        return dispatch_sharded(q, world, |qc| sdpa_qk(qc, k, v, scale, None));
    }
    let sq = q.shape[2];
    if sq <= SDPA_QUERY_CHUNK {
        return sdpa_qk(q, k, v, scale, mask);
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < sq {
        let len = SDPA_QUERY_CHUNK.min(sq - start);
        let qc = q.narrow(2, start, len)?;
        let mask_c = match mask {
            Some(m) => Some(m.narrow(m.rank() - 2, start, len)?),
            None => None,
        };
        chunks.push(sdpa_qk(&qc, k, v, scale, mask_c.as_ref())?);
        start += len;
    }
    let refs: Vec<&CudaTensor> = chunks.iter().collect();
    CudaTensor::cat(&refs, 2)
}

fn sdpa_qk(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: Option<&CudaTensor>,
) -> Result<CudaTensor> {
    if mask.is_none() {
        if let Some(out) = super::attn::device_dense_sdpa(q, k, v, scale)? {
            return Ok(out);
        }
    }
    let d = q.shape[3] as f32;
    let scale = scale.unwrap_or(1.0 / d.sqrt());
    let k_t = k.transpose(2, 3)?;
    let mut scores = q.matmul(&k_t)?;
    scores = scores.mul_scalar(scale);
    if let Some(m) = mask {
        scores = scores.add(m)?;
    }
    let attn = scores.softmax(-1)?;
    attn.matmul(v)
}

pub fn conv2d(
    xs: &CudaTensor,
    kernel: &CudaTensor,
    padding: usize,
    stride: usize,
) -> Result<CudaTensor> {
    xs.conv2d(kernel, None, padding, stride)
}
