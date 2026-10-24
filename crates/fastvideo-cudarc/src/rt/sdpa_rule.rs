//! The one chooser of dense attention (plan B5, `docs/adr/0015-one-sdpa-entry.md`).
//!
//! Every dense SDPA in the runtime goes through [`super::sdpa::sdpa`]; what
//! runs there is decided here and nowhere else:
//!
//! - [`dense_route`]: the kernels an unmasked or additive-masked call tries,
//!   in fallback order (a kernel that cannot run the call declines and the
//!   next one runs; [`Step::Composed`] never declines);
//! - [`mma_kernel`]: which fused tensor-core kernel the [`Step::Mma`] step
//!   launches (V1 / V2 / V3 / V3s / cuDNN / attn_dc, or the per-shape
//!   cuDNN-or-fwd2 rule below);
//! - [`fp8_first`], [`block_causal_first`], [`vdn_window_first`] and
//!   [`kv_window_flash`]: the routes of the H3 FP8 dense blocks, Wan's
//!   block-causal and KV-window attention and VDN-H3's window mask;
//! - the settings these read: `FASTVIDEO_SDPA` ([`backend`]),
//!   `FASTVIDEO_SDPA_AUTO` ([`auto_mode`]), `FASTVIDEO_WAN_CAUSAL_FLASH`
//!   ([`causal_flash_enabled`]) and `FASTVIDEO_H3_VAE_SDPA`
//!   ([`h3_vae_kernel`]); the `[kernels] dense_attention` seam and the
//!   device facts come in through [`Probe`].
//!
//! The functions are pure over a [`Probe`]: the runtime's probe reads the
//! settings and the device ([`super::sdpa::Live`]), the identity test
//! (`sdpa_rule_identity.rs`) a table of every reachable input, against the
//! merge-base choosers it transcribes.
//!
//! # The cuDNN-or-fwd2 rule
//!
//! Until 2026-09-29 `auto` timed both kernels on the first call of every
//! `(bh, sq, sk, d)` in a process and kept the faster. The two kernels do not
//! round alike (1-2 bf16 ulp apart), and at shapes where they run equally
//! fast (text / audio cross-attention: 0.63 vs 0.63 ms) the winner changed
//! from process to process, so the same request gave different bytes on
//! different boots (LTX 1080p: 33.6-34.3 dB between boots). The pick is now a
//! pure function of `(sm_major, sq, sk, d)`: [`RULES`] below, first match
//! wins, else `flash_mma_fwd2`.
//!
//! The table comes from timings, offline: `fv-gpucheck kernels --groups
//! sdpa_rule` times both kernels over the real shapes and the class
//! boundaries and prints, per shape, the faster kernel, the margin and what
//! [`pick`] says (docs/perf/determinism.md). `FASTVIDEO_SDPA_AUTO=timed`
//! restores the old per-process timing (not reproducible across processes;
//! for diagnosis only), and `FASTVIDEO_FLASH_KERNEL=v2|cudnn` still fixes the
//! kernel outright.
//!
//! SageAttention2 (`rt::attn_sage`) is decided before this rule, per recipe
//! and shape ([`dense_route`]): when it is on (default for `ltx-pro` on
//! sm_120), sequences of at least `FASTVIDEO_ATTN_SAGE_MIN_SEQ` on both sides
//! run it, and only the shorter ones (cross-attention) reach this rule.

/// The kernel the rule picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DensePick {
    /// cuDNN's unified SDPA node (heuristic mode A, first deterministic config).
    Cudnn,
    /// `flash_mma_fwd2` (the 128-query double-buffered kernel).
    V2,
}

impl DensePick {
    pub fn name(self) -> &'static str {
        match self {
            DensePick::Cudnn => "cudnn",
            DensePick::V2 => "flash_mma_fwd2",
        }
    }
}

/// One row of the rule: a shape class on one architecture.
#[derive(Clone, Copy, Debug)]
pub struct DenseRule {
    /// Compute capability major the row applies to.
    pub sm_major: i32,
    /// Head dim.
    pub d: usize,
    /// Inclusive bounds on the query length.
    pub min_sq: usize,
    pub max_sq: usize,
    /// Inclusive bounds on the key length.
    pub min_sk: usize,
    pub max_sk: usize,
    pub pick: DensePick,
}

impl DenseRule {
    pub fn matches(&self, sm_major: i32, sq: usize, sk: usize, d: usize) -> bool {
        self.sm_major == sm_major
            && self.d == d
            && (self.min_sq..=self.max_sq).contains(&sq)
            && (self.min_sk..=self.max_sk).contains(&sk)
    }
}

/// The rule for sm_12x, RTX PRO 6000 Blackwell (cuDNN 9.26, heuristic mode A
/// config 0, engine 11), bf16 in and out, d = 128. Median of three
/// synchronized calls (`attn3_bench`, `sdpa_rule`; the timed picks logged by
/// earlier runs):
///
/// | shape (bh x sq x sk) | cuDNN | fwd2 | faster |
/// |---|---:|---:|---|
/// | 24 x 23 616 x 23 616 | 19.08 | 18.85 | fwd2 1 % |
/// | 48 x 23 616 x 23 616 | 37.16 | 37.42 | cuDNN 0.7 % |
/// | 24 x 27 280 x 27 280 | 22.9-23.8 | 24.7-25.2 | cuDNN 4-9 % |
/// | 24 / 48 x 36 080 x 36 080 | 38.6 / 80.8 | 42.6 / 87.4 | cuDNN 8-10 % |
/// | 56 x 37 710 x 37 710 (H3 768p) | 104.8-105.0 | 108.4-109.1 | cuDNN 3 % |
/// | 32 x 124 440 x 124 440 (LTX 1080p 20 s) | 658.8-659.8 | 682.2-682.7 | cuDNN 3.5 % |
/// | 32 x 130 560 x 130 560 (LTX 4K 5 s) | 766.6-769.0 | 757.2-759.9 | fwd2 1.2 % |
/// | 32 x 6 144 x 6 144 (LTX 512p) | 1.97-1.99 | 1.98-2.00 | tie |
/// | any sq x 512 (text cross-attention) | 0.58-1.59 | 0.56-1.60 | tie, either way by run |
///
/// So cuDNN for self-attention-sized keys from 24 576 to 128 000 tokens,
/// fwd2 everywhere else (short keys are a tie; fwd2 needs no plan build).
pub const RULES: &[DenseRule] = &[DenseRule {
    sm_major: 12,
    d: 128,
    min_sq: 24_576,
    max_sq: usize::MAX,
    min_sk: 24_576,
    max_sk: 128_000,
    pick: DensePick::Cudnn,
}];

/// The rule's kernel for a bf16-output dense SDPA of `sq` queries over `sk`
/// keys at head dim `d` on an `sm_major` device: the first matching row of
/// [`RULES`], else fwd2. Does not look at `bh`: the rows were measured at
/// 24-56 heads and the margins do not change sign across that range.
pub fn pick(sm_major: i32, sq: usize, sk: usize, d: usize) -> DensePick {
    RULES
        .iter()
        .find(|r| r.matches(sm_major, sq, sk, d))
        .map_or(DensePick::V2, |r| r.pick)
}

/// How `auto` chooses on sm_12x: `rule` (default, [`pick`]) or `timed` (the
/// old first-call timing, `FASTVIDEO_SDPA_AUTO=timed`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoMode {
    Rule,
    Timed,
}

pub fn auto_mode() -> AutoMode {
    static MODE: std::sync::OnceLock<AutoMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        parse_auto_mode(&super::envflag::string_flag("FASTVIDEO_SDPA_AUTO", "rule"))
    })
}

pub fn parse_auto_mode(v: &str) -> AutoMode {
    match v.trim() {
        "timed" => AutoMode::Timed,
        _ => AutoMode::Rule,
    }
}

/// The table as the offline tool prints it: one line per row.
pub fn describe() -> String {
    RULES
        .iter()
        .map(|r| {
            format!(
                "sm_{}x d={} sq {}..={} sk {}..={} -> {}",
                r.sm_major,
                r.d,
                r.min_sq,
                if r.max_sq == usize::MAX {
                    "inf".to_string()
                } else {
                    r.max_sq.to_string()
                },
                r.min_sk,
                r.max_sk,
                r.pick.name()
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// `FASTVIDEO_SDPA`: `dense` (default: the fused tensor-core kernel on
/// bf16-GEMM sm80+ contexts for d=64/128, else cuBLAS QKᵀ + softmax + PV,
/// query-chunked to bound memory), `cublas` (always the cuBLAS path),
/// `flash` (tiled f32 kernel). Any other value is `dense`, except the two
/// retired host backends `host` and `sparse` ([`Step::Retired`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Dense,
    Cublas,
    Flash,
    /// Retired: the host online-softmax twin of the flash kernel.
    Host,
    /// Retired: the host block-sparse window prototype.
    Sparse,
}

pub fn parse_backend(v: &str) -> Backend {
    match v {
        "cublas" => Backend::Cublas,
        "flash" => Backend::Flash,
        "host" => Backend::Host,
        "sparse" => Backend::Sparse,
        _ => Backend::Dense,
    }
}

static BACKEND_CACHE: super::envflag::CachedString = super::envflag::CachedString::new();

/// The raw `FASTVIDEO_SDPA` value (lower-cased; the device banner prints it).
pub fn backend_setting() -> String {
    BACKEND_CACHE.get_or_init(|| super::envflag::string_flag("FASTVIDEO_SDPA", "dense"))
}

/// `FASTVIDEO_SDPA`, parsed ([`parse_backend`]).
pub fn backend() -> Backend {
    parse_backend(&backend_setting())
}

/// `FASTVIDEO_WAN_CAUSAL_FLASH` (default on): causal Wan self-attention runs
/// the flash kernels (the masked one over a whole clip, the dense one over
/// the KV-cache window of autoregressive generation). `=0` runs the composed
/// reference instead (f32 scores; the whole-clip path materializes the
/// `[S, S]` mask).
pub fn causal_flash_enabled() -> bool {
    static FLAG: super::envflag::CachedBool = super::envflag::CachedBool::new();
    FLAG.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_WAN_CAUSAL_FLASH", true))
}

/// `FASTVIDEO_H3_VAE_SDPA=auto|v1|v2|v3|v3s|cudnn` (default `auto`, the
/// dense-SDPA seam's pick, [`mma_kernel`]): the fused kernel for the H3 ViT
/// decoder's attention only. `None` is `auto`.
pub fn h3_vae_kernel() -> Option<FlashKernel> {
    static FLAG: super::envflag::CachedString = super::envflag::CachedString::new();
    parse_kernel_override(
        FLAG.get_or_init(|| super::envflag::string_flag("FASTVIDEO_H3_VAE_SDPA", "auto"))
            .trim(),
    )
}

/// [`h3_vae_kernel`]'s parse: a fixed kernel, or `None` (`auto` and any
/// other value, `dc` included).
pub fn parse_kernel_override(v: &str) -> Option<FlashKernel> {
    match v {
        "v1" => Some(FlashKernel::V1),
        "v2" => Some(FlashKernel::V2),
        "v3" => Some(FlashKernel::V3),
        "v3s" => Some(FlashKernel::V3s),
        "cudnn" => Some(FlashKernel::Cudnn),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The fused kernels
// ---------------------------------------------------------------------------

/// Which fused dense kernel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlashKernel {
    /// `flash_mma_fwd_d*`: 64-query CTAs, 4 warps, one K and one V buffer.
    V1,
    /// `flash_mma_fwd2_d*`: 128-query CTAs (8 warps x 16 rows) and
    /// double-buffered K/V; bit-identical to V1 (same per-row arithmetic).
    V2,
    /// `flash_mma_fwd3_d128`: V2's CTA and arithmetic with S_{j+1} issued
    /// before softmax(S_j) and a three-stage K/V ring; bit-identical to V2.
    /// d=128 only (V2 otherwise).
    V3,
    /// `flash_mma_fwd3s_d128`: V3 that skips the O rescale when no row max
    /// rose (the factor V2 applies there is within a few ulp of 1).
    V3s,
    /// cuDNN's fused attention engine (`cudnn_sdpa`), bf16 output only;
    /// V2 whenever cuDNN offers no engine or the caller wants f32 out.
    Cudnn,
    /// `attn_dc.cu` (d = 128 on 9.0 / 10.0): tcgen05 + TMEM on B200, wgmma
    /// on H100 / H200, TMA-fed and warp-specialised; V2 anywhere else.
    Dc,
}

/// Query/key tile of the fused kernel (one CTA per 64 queries).
pub const MMA_TILE: usize = 64;

/// Whether V2's grid (one 256-thread CTA per SM) is at least 16 waves.
pub fn flash_v2_fills(sq: usize, bh: usize, sms: usize) -> bool {
    sq.div_ceil(2 * MMA_TILE).saturating_mul(bh) >= 16 * sms.max(1)
}

/// Whether `auto` picks the 128-query double-buffered kernel: yes. It is
/// bit-identical to V1 and 4% faster on RTX PRO 6000 at the H3 / LTX shapes
/// (`attn_bench`: 373 vs 358 TFLOPS). `FASTVIDEO_FLASH_KERNEL=v1` restores V1.
pub fn flash_v2_default() -> bool {
    true
}

/// Whether `auto` lets a bf16-output dense SDPA that V2 would run go to
/// cuDNN's unified SDPA node instead, per shape ([`pick`]): on sm_12x only.
/// RTX PRO 6000, cuDNN 9.26, `attn3_bench` (two pods): 104.8-105.0 vs
/// 108.4-109.1 ms at H3 768p, 658.8-659.8 vs 682.2-682.7 ms at LTX 1080p
/// 20 s, but 766.6-769.0 vs 757.2-759.9 ms at 4K 5 s (engine 11, heuristic
/// mode A config 0), hence the per-shape rule. `FASTVIDEO_FLASH_KERNEL=v2`
/// (or `=cudnn`) fixes the kernel.
pub fn cudnn_default(sm_major: i32) -> bool {
    CUDNN_DEFAULT_ON && sm_major == 12
}

/// On since the attn3 generation A/B (FastH3 4-step dense 768p denoise 33.23 -> 32.28 s,
/// LTX-2.5 1080p 20 s dense stage 2 44.0 -> 42.7 s/step).
const CUDNN_DEFAULT_ON: bool = true;

// ---------------------------------------------------------------------------
// The chooser
// ---------------------------------------------------------------------------

/// What the chooser reads. Each method is called only where the rule needs
/// it, in the order the merge-base choosers read it; [`Probe::dc_loaded`]
/// (loads the attn_dc module) and [`Probe::cudnn_has_plan`] (builds a cuDNN
/// plan) have effects, so [`mma_kernel`] asks for them only when the
/// [`Step::Mma`] step runs.
pub trait Probe {
    /// `FASTVIDEO_SDPA` ([`backend`]).
    fn backend(&self) -> Backend;
    /// `FASTVIDEO_SP_WORLD`, at least 1.
    fn sp_world(&self) -> usize;
    /// The fused kernels' context: a device is expected, it runs bf16 GEMM
    /// math and is sm80+ (`attn::mma_sdpa_default`).
    fn mma_default(&self) -> bool;
    /// `Some(FASTVIDEO_ATTN_SAGE_MIN_SEQ)` when SageAttention2 is on for this
    /// thread and device (`attn_sage::dense_enabled`), else `None`.
    fn sage_min_seq(&self) -> Option<usize>;
    /// Whether the request runs `fp8_attention` on dense calls
    /// (`attn_fp8::dense_enabled`; always false without CUDA).
    fn fp8_dense(&self) -> bool;
    /// `FASTVIDEO_WAN_CAUSAL_FLASH` ([`causal_flash_enabled`]).
    fn causal_flash(&self) -> bool;
    /// The kernel seam (`[kernels] dense_attention`, else
    /// `FASTVIDEO_FLASH_KERNEL`), lower-cased, not trimmed.
    fn kernel_choice(&self) -> String;
    /// The global device's compute capability major, `None` without one.
    fn sm_major(&self) -> Option<i32>;
    /// The global device's SM count (at least 1).
    fn sms(&self) -> usize;
    /// Whether the attn_dc module is loaded for the device (sm_90 / sm_100).
    fn dc_loaded(&self) -> bool;
    /// `FASTVIDEO_SDPA_AUTO` ([`auto_mode`]).
    fn auto_mode(&self) -> AutoMode;
    /// Whether cuDNN has an engine for a bf16 `[bh, sq, d] x [bh, sk, d]` SDPA.
    fn cudnn_has_plan(&self, bh: usize, sq: usize, sk: usize, d: usize) -> bool;
}

/// A BHSD call whose K and V agree with Q: `q [b, h, sq, d]`, `k` and `v`
/// `[b, h, sk, d]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dims {
    pub b: usize,
    pub h: usize,
    pub sq: usize,
    pub sk: usize,
    pub d: usize,
}

/// The shape a dense call is chosen on: the query and key lengths of the
/// rank-4 operands, and the [`Dims`] when the three agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub sq: usize,
    pub sk: usize,
    pub dims: Option<Dims>,
}

impl Shape {
    /// From the operands' shapes (each rank 4).
    pub fn of(q: &[usize], k: &[usize], v: &[usize]) -> Self {
        let dims = match (q, k) {
            ([b, h, sq, d], [_, _, sk, _]) if k == [*b, *h, *sk, *d] && v == k => Some(Dims {
                b: *b,
                h: *h,
                sq: *sq,
                sk: *sk,
                d: *d,
            }),
            _ => None,
        };
        Self {
            sq: q.get(2).copied().unwrap_or(0),
            sk: k.get(2).copied().unwrap_or(0),
            dims,
        }
    }
}

/// One kernel of a [`dense_route`]. All but [`Step::Composed`] and
/// [`Step::Retired`] may decline a call they cannot run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    /// `FASTVIDEO_SDPA=host|sparse`: the two host backends, retired (B5):
    /// the call is refused. Both were host-only, so a GPU run already
    /// refused them (`host_only_op`).
    Retired(Backend),
    /// `attn::device_flash_sdpa` (`FASTVIDEO_SDPA=flash`, f32 tiles).
    FlashF32,
    /// `attn_sage::dense_sdpa` (INT8 QK, FP8 PV).
    Sage { out_bf16: bool },
    /// `attn::device_mma_sdpa`: the fused kernel [`mma_kernel`] picks.
    Mma { out_bf16: bool },
    /// `attn::device_dense_sdpa` (cuBLAS, query-chunked).
    Cublas,
    /// The composed tensor-op SDPA (masked calls, CPU runs).
    Composed,
}

/// The kernels an unmasked (`masked = false`) or additive-masked dense call
/// tries, in order; the first that does not decline runs. `out_bf16` is the
/// fused kernels' output type (bf16 activations).
pub fn dense_route(shape: &Shape, masked: bool, out_bf16: bool, p: &dyn Probe) -> Vec<Step> {
    let mut route = Vec::with_capacity(4);
    if !masked {
        let backend = p.backend();
        if matches!(backend, Backend::Host | Backend::Sparse) {
            return vec![Step::Retired(backend)];
        }
        if backend == Backend::Flash {
            route.push(Step::FlashF32);
        }
        if backend != Backend::Cublas && p.mma_default() {
            // Opt-in SageAttention2 (INT8 QK, FP8 PV): long self-attention only.
            if let Some(min) = p.sage_min_seq() {
                if shape.sq >= min && shape.sk >= min {
                    route.push(Step::Sage { out_bf16 });
                }
            }
            route.push(Step::Mma { out_bf16 });
        }
        route.push(Step::Cublas);
    }
    route.push(Step::Composed);
    route
}

/// Runs a route: each step through `attempt` until one returns a result.
/// `None` only when every step declined (a route ending in
/// [`Step::Composed`] never does).
pub fn run_route<T, E>(
    route: &[Step],
    mut attempt: impl FnMut(Step) -> std::result::Result<Option<T>, E>,
) -> std::result::Result<Option<T>, E> {
    for &step in route {
        if let Some(out) = attempt(step)? {
            return Ok(Some(out));
        }
    }
    Ok(None)
}

/// The fused kernel a [`Step::Mma`] launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MmaKernel {
    /// This kernel, with the caller's output type.
    Fixed(FlashKernel),
    /// `auto` on sm_12x, bf16 out: the per-shape rule's kernel ([`pick`],
    /// cuDNN only where it has a plan), logged once per shape.
    Rule { kernel: FlashKernel, sm_major: i32 },
    /// `FASTVIDEO_SDPA_AUTO=timed`: the old first-call timing.
    Timed,
}

/// The kernel seam's choice for a `bh` x `sq` grid on `sms` SMs: a fixed
/// `v1|v2|v3|v3s|cudnn|dc`, else (`auto` or any other value) attn_dc where it
/// is loaded (9.0 / 10.0; d = 128, V2 for other head dims), else V2 when its
/// 128-query grid fills the GPU for many waves ([`flash_v2_fills`]), V1
/// otherwise (at LTX 768x512, 6 144 tokens x 32 heads, V2's one-CTA-per-SM
/// tail costs 2%).
pub fn flash_kernel_for(sq: usize, bh: usize, sms: usize, p: &dyn Probe) -> FlashKernel {
    match p.kernel_choice().as_str() {
        "v1" => FlashKernel::V1,
        "v2" => FlashKernel::V2,
        "v3" => FlashKernel::V3,
        "v3s" => FlashKernel::V3s,
        "cudnn" => FlashKernel::Cudnn,
        "dc" => FlashKernel::Dc,
        // auto on a 9.0 / 10.0 device: the datacenter kernel (sm_120 and
        // older keep the choice below; `v2` / `v1` are the escape hatch).
        _ if p.dc_loaded() => FlashKernel::Dc,
        _ if flash_v2_default() && flash_v2_fills(sq, bh, sms) => FlashKernel::V2,
        _ => FlashKernel::V1,
    }
}

/// The fused kernel for a call of `dims` (`None`: the operands disagree, the
/// kernel then declines) with output type `out_bf16`. `auto` on sm_12x with
/// bf16 out applies the per-shape rule (an explicit `v2` stays V2).
pub fn mma_kernel(dims: Option<Dims>, out_bf16: bool, p: &dyn Probe) -> MmaKernel {
    let (Some(d), Some(sm_major)) = (dims, dims.and_then(|_| p.sm_major())) else {
        return MmaKernel::Fixed(flash_kernel_for(usize::MAX, 1, 1, p));
    };
    let sms = p.sms();
    match flash_kernel_for(d.sq, d.b * d.h, sms, p) {
        FlashKernel::V2 if out_bf16 && cudnn_default(sm_major) && p.kernel_choice() == "auto" => {
            if p.auto_mode() == AutoMode::Timed {
                return MmaKernel::Timed;
            }
            let kernel = match pick(sm_major, d.sq, d.sk, d.d) {
                DensePick::Cudnn if p.cudnn_has_plan(d.b * d.h, d.sq, d.sk, d.d) => {
                    FlashKernel::Cudnn
                }
                _ => FlashKernel::V2,
            };
            MmaKernel::Rule { kernel, sm_major }
        }
        k => MmaKernel::Fixed(k),
    }
}

/// Whether an unmasked dense call first tries the FP8 `Q K^T` kernel (the
/// H3 dense blocks under `fp8_attention`; it declines where it cannot run).
pub fn fp8_first(p: &dyn Probe) -> bool {
    p.fp8_dense()
}

/// Whether a block-causal call first tries `flash_mma_fwd2_causal_d*`
/// (else, and when it declines, the dense additive mask).
pub fn block_causal_first(p: &dyn Probe) -> bool {
    p.causal_flash() && p.sp_world() == 1 && p.mma_default()
}

/// Whether a VDN-window call first tries `flash_mma_fwd2_window_d128`.
pub fn vdn_window_first(p: &dyn Probe) -> bool {
    p.sp_world() == 1 && p.mma_default()
}

/// Whether a KV-window call runs the dense route (`true`) or the f32
/// composed reference: the caller's choice, else [`causal_flash_enabled`].
pub fn kv_window_flash(flash: Option<bool>, p: &dyn Probe) -> bool {
    flash.unwrap_or_else(|| p.causal_flash())
}

#[cfg(test)]
#[path = "sdpa_rule_identity.rs"]
mod identity;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_measured_shapes_get_the_faster_kernel_or_a_tie() {
        // Self-attention where cuDNN measured faster.
        for n in [27_280, 36_080, 37_710, 124_440] {
            assert_eq!(pick(12, n, n, 128), DensePick::Cudnn, "{n}");
        }
        // fwd2 faster (4K 5 s, 23 616 at 24 heads) or a tie (512p, cross-attention).
        for (sq, sk) in [
            (130_560, 130_560),
            (23_616, 23_616),
            (6_144, 6_144),
            (27_280, 512),
            (38_760, 1_024),
            (124_440, 151),
        ] {
            assert_eq!(pick(12, sq, sk, 128), DensePick::V2, "{sq} x {sk}");
        }
    }

    #[test]
    fn only_sm12x_d128_has_rows() {
        assert_eq!(pick(9, 37_710, 37_710, 128), DensePick::V2);
        assert_eq!(pick(10, 37_710, 37_710, 128), DensePick::V2);
        assert_eq!(pick(12, 37_710, 37_710, 64), DensePick::V2);
    }

    #[test]
    fn the_pick_is_a_pure_function_of_the_shape() {
        let a: Vec<DensePick> = (0..64)
            .map(|i| pick(12, 20_000 + i * 2_000, 20_000 + i * 2_000, 128))
            .collect();
        let b: Vec<DensePick> = (0..64)
            .map(|i| pick(12, 20_000 + i * 2_000, 20_000 + i * 2_000, 128))
            .collect();
        assert_eq!(a, b);
        assert!(describe().contains("cudnn"));
    }

    #[test]
    fn auto_mode_parses() {
        assert_eq!(parse_auto_mode("timed"), AutoMode::Timed);
        assert_eq!(parse_auto_mode("rule"), AutoMode::Rule);
        assert_eq!(parse_auto_mode(""), AutoMode::Rule);
    }

    #[test]
    fn backend_parses_like_the_string_compares_it_replaces() {
        assert_eq!(parse_backend("dense"), Backend::Dense);
        assert_eq!(parse_backend("cublas"), Backend::Cublas);
        assert_eq!(parse_backend("flash"), Backend::Flash);
        assert_eq!(parse_backend("host"), Backend::Host);
        assert_eq!(parse_backend("sparse"), Backend::Sparse);
        // Unknown values (and untrimmed ones) were "not sparse / host /
        // flash / cublas": the dense default.
        for v in ["", "fused", " cublas", "dense "] {
            assert_eq!(parse_backend(v), Backend::Dense, "{v:?}");
        }
    }

    #[test]
    fn the_h3_vae_override_parses() {
        assert_eq!(parse_kernel_override("v3s"), Some(FlashKernel::V3s));
        assert_eq!(parse_kernel_override("cudnn"), Some(FlashKernel::Cudnn));
        for v in ["auto", "dc", "", "V2"] {
            assert_eq!(parse_kernel_override(v), None, "{v:?}");
        }
    }

    #[test]
    fn flash_v2_only_where_its_grid_fills_the_gpu() {
        // RTX PRO 6000: 188 SMs.
        assert!(flash_v2_fills(37_710, 56, 188)); // H3 768p
        assert!(flash_v2_fills(124_440, 32, 188)); // LTX 1080p 20 s stage 2
        assert!(flash_v2_fills(130_560, 32, 188)); // LTX 4K 5 s stage 2
        assert!(!flash_v2_fills(6_144, 32, 188)); // LTX 768x512: V1 wins by 2%
        assert!(!flash_v2_fills(1, 1, 188));
    }

    #[test]
    fn shapes_agree_only_when_k_and_v_match_q() {
        let s = Shape::of(&[1, 2, 5, 64], &[1, 2, 7, 64], &[1, 2, 7, 64]);
        assert_eq!(
            s.dims,
            Some(Dims {
                b: 1,
                h: 2,
                sq: 5,
                sk: 7,
                d: 64
            })
        );
        assert_eq!((s.sq, s.sk), (5, 7));
        assert_eq!(
            Shape::of(&[1, 2, 5, 64], &[1, 2, 7, 64], &[1, 2, 6, 64]).dims,
            None
        );
        assert_eq!(
            Shape::of(&[1, 2, 5, 64], &[1, 3, 7, 64], &[1, 3, 7, 64]).dims,
            None
        );
        assert_eq!(
            Shape::of(&[1, 2, 5, 64], &[1, 2, 7, 32], &[1, 2, 7, 32]).dims,
            None
        );
    }
}
