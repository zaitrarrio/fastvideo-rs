//! The one dense-attention entry (plan B5, `docs/adr/0015-one-sdpa-entry.md`):
//! [`sdpa`]`(q, k, v, &`[`Plan`]`)`.
//!
//! A [`Plan`] says what a call site computes: plain softmax attention (with
//! an optional additive mask), Wan's block-causal mask, VDN-H3's window
//! mask, the KV-cache window of causal Wan, or the H3 dense block that tries
//! the FP8 kernel first. Which kernel runs is chosen by [`super::sdpa_rule`]
//! alone, from the settings and the device ([`Live`]); this module only
//! executes that choice: bf16 rounding of the operands under bf16
//! activations, the sequence-parallel query shards, and the fallback from a
//! kernel that declines to the next one of the route.
//!
//! Every family's model code calls [`sdpa`]. The kernels below it
//! (`attn::device_*`, `attn_fp8`, `attn_sage`, `cudnn_sdpa`, `attn_dc`) stay
//! public for the kernel checks in `fv-gpucheck`, and for the fused H3 ViT
//! decoder, which launches the fused kernel the rule picks
//! (`attn::device_mma_sdpa`) inside its own fused block chain.

use super::attn::{BlockCausal, VdnWindow};
use super::sdpa_rule::{self, AutoMode, Backend, Probe, Shape, Step};
use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// What one attention call computes. Built at the call site; the kernel is
/// [`super::sdpa_rule`]'s choice.
#[derive(Clone, Copy)]
pub struct Plan<'a> {
    scale: Option<f32>,
    attn: Attn<'a>,
}

#[derive(Clone, Copy)]
enum Attn<'a> {
    /// `softmax(scale q kᵀ + mask) v` through the dense route.
    Dense {
        mask: Option<&'a CudaTensor>,
        fp8_first: bool,
    },
    /// The composed tensor ops alone (no rounding, no shards, no kernel).
    Composed {
        mask: Option<&'a CudaTensor>,
    },
    BlockCausal(BlockCausal),
    VdnWindow(VdnWindow),
    /// `flash: None` is `FASTVIDEO_WAN_CAUSAL_FLASH`.
    KvWindow {
        flash: Option<bool>,
    },
}

impl<'a> Plan<'a> {
    /// Softmax attention; `scale` defaults to `1/sqrt(d)`.
    pub fn dense(scale: Option<f32>) -> Self {
        Self::masked(scale, None)
    }

    /// Softmax attention with an additive `[.., sq, sk]` mask (composed
    /// tensor ops: no fused kernel takes an arbitrary mask).
    pub fn masked(scale: Option<f32>, mask: Option<&'a CudaTensor>) -> Self {
        Self {
            scale,
            attn: Attn::Dense {
                mask,
                fp8_first: false,
            },
        }
    }

    /// An unmasked dense call that first tries the FP8 `Q K^T` kernel when
    /// the request runs `fp8_attention` on dense calls (the H3 dense blocks).
    pub fn fp8_first(self) -> Self {
        match self.attn {
            Attn::Dense { mask: None, .. } => Self {
                attn: Attn::Dense {
                    mask: None,
                    fp8_first: true,
                },
                ..self
            },
            _ => self,
        }
    }

    /// The composed tensor-op SDPA with an optional additive mask, on the
    /// operands as given: for callers that must not be rounded or sharded
    /// (FLUX.2's text encoder).
    pub fn composed(scale: Option<f32>, mask: Option<&'a CudaTensor>) -> Self {
        Self {
            scale,
            attn: Attn::Composed { mask },
        }
    }

    /// Wan's block-causal temporal mask ([`BlockCausal`]): one flash launch
    /// that never builds the `[S, S]` mask on a tensor-core device; the
    /// dense additive mask elsewhere (CPU runs, sequence-parallel shards,
    /// unsupported head dims, `FASTVIDEO_WAN_CAUSAL_FLASH=0`).
    pub fn block_causal(scale: Option<f32>, mask: BlockCausal) -> Self {
        Self {
            scale,
            attn: Attn::BlockCausal(mask),
        }
    }

    /// VDN-H3's window mask ([`VdnWindow`]): one flash launch on a
    /// tensor-core device (`flash_mma_fwd2_window_d128`), else the dense
    /// additive mask (CPU runs, head dims other than 128; small shapes only).
    pub fn vdn_window(scale: Option<f32>, mask: VdnWindow) -> Self {
        Self {
            scale,
            attn: Attn::VdnWindow(mask),
        }
    }

    /// Causal Wan's autoregressive self-attention: one block's queries
    /// against the KV-cache window, no mask (FastVideo
    /// `CausalWanSelfAttention` with a `kv_cache`). The dense route, or the
    /// composed f32 reference under `FASTVIDEO_WAN_CAUSAL_FLASH=0`.
    pub fn kv_window() -> Self {
        Self {
            scale: None,
            attn: Attn::KvWindow { flash: None },
        }
    }

    /// [`Plan::kv_window`] with the route fixed by the caller (`false`: the
    /// composed reference, f32 scores and softmax).
    pub fn kv_window_with(flash: bool) -> Self {
        Self {
            scale: None,
            attn: Attn::KvWindow { flash: Some(flash) },
        }
    }
}

/// Scaled dot-product attention over BHSD `q [b, h, sq, d]`, `k` / `v`
/// `[b, h, sk, d]`, as `plan` says ([`Plan`]).
pub fn sdpa(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, plan: &Plan<'_>) -> Result<CudaTensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(msg("sdpa expects BHSD"));
    }
    let scale = plan.scale;
    match plan.attn {
        Attn::Dense { mask, fp8_first } => {
            #[cfg(feature = "cuda")]
            if fp8_first && sdpa_rule::fp8_first(&Live) {
                let out_bf16 = super::tensor::bf16_activations();
                if let Some(out) = super::attn_fp8::dense_sdpa(q, k, v, scale, out_bf16)? {
                    return Ok(out);
                }
            }
            #[cfg(not(feature = "cuda"))]
            let _ = fp8_first;
            dense(q, k, v, scale, mask)
        }
        Attn::Composed { mask } => sdpa_composed(q, k, v, scale, mask),
        Attn::BlockCausal(mask) => block_causal(q, k, v, scale, mask),
        Attn::VdnWindow(mask) => vdn_window(q, k, v, scale, mask),
        Attn::KvWindow { flash } => kv_window(q, k, v, sdpa_rule::kv_window_flash(flash, &Live)),
    }
}

/// bf16-rounded copies of the operands under bf16 activations.
fn act16_operands(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
) -> Result<Option<(CudaTensor, CudaTensor, CudaTensor)>> {
    if super::tensor::bf16_activations() {
        Ok(Some((
            q.quantize_bf16()?,
            k.quantize_bf16()?,
            v.quantize_bf16()?,
        )))
    } else {
        Ok(None)
    }
}

/// The dense route ([`sdpa_rule::dense_route`]) on each query shard.
fn dense(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: Option<&CudaTensor>,
) -> Result<CudaTensor> {
    let act = act16_operands(q, k, v)?;
    let (q, k, v) = match &act {
        Some((q, k, v)) => (q, k, v),
        None => (q, k, v),
    };
    let run = |q: &CudaTensor| -> Result<CudaTensor> {
        let shape = Shape::of(&q.shape, &k.shape, &v.shape);
        let out_bf16 = super::tensor::bf16_activations();
        let route = sdpa_rule::dense_route(&shape, mask.is_some(), out_bf16, &Live);
        sdpa_rule::run_route(&route, |step| run_step(step, q, k, v, scale, mask))?
            .ok_or_else(|| msg("sdpa: every kernel of the route declined"))
    };
    let world = sp_world();
    let out = if world > 1 {
        dispatch_sharded(q, world, run)?
    } else {
        run(q)?
    };
    if super::tensor::bf16_activations() {
        out.quantize_bf16()
    } else {
        Ok(out)
    }
}

/// One step of a dense route; `None` when its kernel declines the call.
fn run_step(
    step: Step,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: Option<&CudaTensor>,
) -> Result<Option<CudaTensor>> {
    match step {
        Step::Retired(backend) => Err(msg(retired(backend))),
        Step::FlashF32 => super::attn::device_flash_sdpa(q, k, v, scale),
        Step::Sage { out_bf16 } => super::attn_sage::dense_sdpa(q, k, v, scale, out_bf16),
        Step::Mma { out_bf16 } => super::attn::device_mma_sdpa(q, k, v, scale, out_bf16),
        Step::Cublas => super::attn::device_dense_sdpa(q, k, v, scale),
        Step::Composed => sdpa_composed(q, k, v, scale, mask).map(Some),
    }
}

/// Why a retired `FASTVIDEO_SDPA` value is refused.
fn retired(backend: Backend) -> String {
    match backend {
        Backend::Sparse => "FASTVIDEO_SDPA=sparse (the host block-sparse window prototype, \
             FASTVIDEO_VSA_WINDOW) is retired: FASTVIDEO_VSA=1 selects video sparse attention \
             on the device; unset FASTVIDEO_SDPA for dense attention"
            .into(),
        _ => "FASTVIDEO_SDPA=host (the host online-softmax attention) is retired: unset \
             FASTVIDEO_SDPA (the dense default; CPU runs compute the composed reference)"
            .into(),
    }
}

fn block_causal(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: BlockCausal,
) -> Result<CudaTensor> {
    let (sq, sk) = (q.shape[2], k.shape[2]);
    let act16 = super::tensor::bf16_activations();
    if sdpa_rule::block_causal_first(&Live) {
        let act = act16_operands(q, k, v)?;
        let (q, k, v) = match &act {
            Some((q, k, v)) => (q, k, v),
            None => (q, k, v),
        };
        if let Some(out) = super::attn::device_mma_sdpa_causal(q, k, v, scale, act16, mask)? {
            return Ok(out);
        }
    }
    let dense_mask = CudaTensor::from_vec(mask.dense_mask(sq, sk), vec![1, 1, sq, sk])?;
    dense(q, k, v, scale, Some(&dense_mask))
}

fn vdn_window(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: VdnWindow,
) -> Result<CudaTensor> {
    let (sq, sk) = (q.shape[2], k.shape[2]);
    let act16 = super::tensor::bf16_activations();
    if sdpa_rule::vdn_window_first(&Live) {
        let act = act16_operands(q, k, v)?;
        let (q, k, v) = match &act {
            Some((q, k, v)) => (q, k, v),
            None => (q, k, v),
        };
        if let Some(out) = super::attn::device_mma_sdpa_window(q, k, v, scale, act16, mask)? {
            return Ok(out);
        }
    }
    // The masked fallback materializes [sq, sk]: small (test) shapes only.
    if sq.saturating_mul(sk) > 1 << 26 {
        return Err(msg(format!(
            "sdpa_vdn_window: the window kernel cannot run this {sq}x{sk} call and the dense mask is too large"
        )));
    }
    let dense_mask = CudaTensor::from_vec(mask.dense_mask(sq, sk), vec![1, 1, sq, sk])?;
    dense(q, k, v, scale, Some(&dense_mask))
}

fn kv_window(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, flash: bool) -> Result<CudaTensor> {
    if flash {
        return dense(q, k, v, None, None);
    }
    let act16 = super::tensor::bf16_activations();
    let act = act16_operands(q, k, v)?;
    let (q, k, v) = match &act {
        Some((q, k, v)) => (q, k, v),
        None => (q, k, v),
    };
    let out = super::tensor::with_bf16_act(false, || {
        sdpa_composed(
            &q.to_f32_act()?,
            &k.to_f32_act()?,
            &v.to_f32_act()?,
            None,
            None,
        )
    })?;
    if act16 {
        out.quantize_bf16()
    } else {
        Ok(out)
    }
}

/// SDPA from tensor ops (masked attention, CPU runs). Every op here has a
/// device kernel, so on GPU runs this stays on the device.
fn sdpa_composed(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: Option<&CudaTensor>,
) -> Result<CudaTensor> {
    let d = q.shape[3] as f32;
    let scale = scale.unwrap_or(1.0 / d.sqrt());
    let mut scores = q.matmul(&k.transpose(2, 3)?)?.try_mul_scalar(scale)?;
    if let Some(m) = mask {
        scores = scores.add(m)?;
    }
    scores.softmax(-1)?.matmul(v)
}

static SP_WORLD_CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// `FASTVIDEO_SP_WORLD`: query-sequence shards of a dense call (one GPU per
/// rank, host-mediated gather; ADR-0003).
fn sp_world() -> usize {
    *SP_WORLD_CACHE.get_or_init(|| super::envflag::usize_flag("FASTVIDEO_SP_WORLD", 1).max(1))
}

/// Run `compute` on each of `world` query-sequence shards, one GPU per rank,
/// and gather the results (host-mediated; see [`super::sp`]).
fn dispatch_sharded(
    q: &CudaTensor,
    world: usize,
    compute: impl Fn(&CudaTensor) -> Result<CudaTensor> + Sync,
) -> Result<CudaTensor> {
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
                    let dev =
                        super::device::device_for_index(idx).map_err(|e| msg(e.to_string()))?;
                    super::device::set_thread_device(Some(dev));
                    let qc = super::sp::shard_tensor(&q.clone(), 2, rank, world)?;
                    let mut out = compute(&qc)?;
                    out.ensure_host()?;
                    Ok(CudaTensor::from_vec(
                        out.host_cow()?.into_owned(),
                        out.shape.clone(),
                    )?)
                })();
                super::device::set_thread_device(None);
                *results[rank].lock().expect("sdpa shard result lock") = Some(outcome);
            });
        }
    });
    let mut shards = Vec::with_capacity(world);
    for r in results {
        shards.push(
            r.into_inner()
                .expect("sdpa shard result lock")
                .ok_or_else(|| msg("sdpa shard produced no result"))??,
        );
    }
    super::sp::all_gather_seq(&shards, 2)
}

/// The runtime's [`Probe`]: the settings and the calling thread's device.
pub struct Live;

impl Probe for Live {
    fn backend(&self) -> Backend {
        sdpa_rule::backend()
    }

    fn sp_world(&self) -> usize {
        sp_world()
    }

    fn mma_default(&self) -> bool {
        super::attn::mma_sdpa_default()
    }

    fn sage_min_seq(&self) -> Option<usize> {
        super::attn_sage::dense_enabled().then(super::attn_sage::min_seq)
    }

    fn fp8_dense(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            super::attn_fp8::dense_enabled()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn causal_flash(&self) -> bool {
        sdpa_rule::causal_flash_enabled()
    }

    fn kernel_choice(&self) -> String {
        fastvideo_models::techniques::kernels::choice(
            fastvideo_models::techniques::kernels::KernelOp::DenseAttention,
        )
    }

    fn sm_major(&self) -> Option<i32> {
        #[cfg(feature = "cuda")]
        {
            super::device::global_device().map(|d| d.sm_major)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn sms(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
            super::device::global_device().map_or(1, |dev| {
                dev.ctx
                    .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                    .map_or(1, |n| n.max(1) as usize)
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            1
        }
    }

    fn dc_loaded(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            super::attn_dc::dense().is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn auto_mode(&self) -> AutoMode {
        sdpa_rule::auto_mode()
    }

    fn cudnn_has_plan(&self, bh: usize, sq: usize, sk: usize, d: usize) -> bool {
        #[cfg(feature = "cuda")]
        {
            super::cudnn_sdpa::has_plan(bh, sq, sk, d)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (bh, sq, sk, d);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::tensor::with_bf16_act;

    fn t(n: usize, k: f32, shape: &[usize]) -> CudaTensor {
        CudaTensor::from_vec(
            (0..n).map(|i| (i as f32 * k).sin() * 0.9).collect(),
            shape.to_vec(),
        )
        .unwrap()
    }

    /// On a CPU run every dense plan is the composed reference.
    #[test]
    fn cpu_dense_plans_are_the_composed_reference() {
        let (b, h, sq, sk, d) = (1usize, 2usize, 5usize, 7usize, 8usize);
        let q = t(b * h * sq * d, 0.37, &[b, h, sq, d]);
        let k = t(b * h * sk * d, 0.71, &[b, h, sk, d]);
        let v = t(b * h * sk * d, 1.13, &[b, h, sk, d]);
        let want = with_bf16_act(false, || sdpa_composed(&q, &k, &v, None, None)).unwrap();
        for plan in [
            Plan::dense(None),
            Plan::dense(None).fp8_first(),
            Plan::masked(None, None),
            Plan::composed(None, None),
            Plan::kv_window_with(true),
            Plan::kv_window_with(false),
        ] {
            let got = with_bf16_act(false, || sdpa(&q, &k, &v, &plan)).unwrap();
            assert_eq!(got.host_cow().unwrap(), want.host_cow().unwrap());
        }
        let scaled = with_bf16_act(false, || sdpa(&q, &k, &v, &Plan::dense(Some(0.5)))).unwrap();
        assert_ne!(scaled.host_cow().unwrap(), want.host_cow().unwrap());
    }

    #[test]
    fn non_bhsd_operands_are_refused() {
        let q = t(8, 0.1, &[2, 4]);
        let err = sdpa(&q, &q, &q, &Plan::dense(None)).unwrap_err();
        assert!(err.to_string().contains("BHSD"), "{err}");
    }

    #[test]
    fn the_retired_backends_say_what_replaces_them() {
        assert!(retired(Backend::Sparse).contains("FASTVIDEO_VSA=1"));
        assert!(retired(Backend::Host).contains("retired"));
    }

    #[test]
    fn fp8_first_only_marks_unmasked_dense_calls() {
        let m = t(4, 0.2, &[1, 1, 2, 2]);
        assert!(matches!(
            Plan::masked(None, Some(&m)).fp8_first().attn,
            Attn::Dense {
                fp8_first: false,
                ..
            }
        ));
        assert!(matches!(
            Plan::dense(None).fp8_first().attn,
            Attn::Dense {
                fp8_first: true,
                mask: None
            }
        ));
    }
}
