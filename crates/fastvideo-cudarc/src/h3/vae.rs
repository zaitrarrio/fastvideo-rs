//! MiniMax-H3 video VAE, decoder half (`AutoencoderKLMiniMaxH3.decode`).
//!
//! The decoder has no convolutional upsampling path: it is a 36-layer
//! non-causal ViT over latent voxels whose last linear emits a whole
//! `3 x 4 x 16 x 16` pixel patch per token. What surrounds the ViT is
//! arithmetic the released frames depend on, so it is reproduced exactly:
//!
//! * **Spatial tiling is part of the output**, not a memory workaround: the
//!   model was released with 256 px tiles at >= 64 px overlap, each tile
//!   decoded on its own with rotary positions normalized to *that tile's*
//!   extent, then cross-faded. Decoding the full canvas in one pass gives
//!   different (and untrained-for) positions.
//! * **Temporal chunks mirror the encoder's 17-frame clips.** One call sees 7
//!   latent frames and yields 28 pixel frames: 3 pre-padding frames to drop, a
//!   17-frame body, 3 more pre-padding frames, and a 5-frame tail that is
//!   cross-faded into the head of the next chunk's body.
//! * Blends read the **raw** neighbouring tile (`tiles[i-1][j]`, `row[j-1]` in
//!   `_stitch_tiles`), not the neighbour's already-blended result.
//!
//! Every tile of a chunk has the same shape and therefore the same rotary
//! tables, so tiles are batched through the ViT (evenly: 28 tiles as 4 x 7).
//! Frames are handed to a sink chunk by chunk (17 frames at a time), which
//! keeps the working set at about one chunk regardless of clip length.
//!
//! With bf16 GEMMs (the default on tensor-core GPUs) the ViT keeps its
//! activations in bf16 between GEMMs, which is where the unfused path rounds
//! them anyway, and one kernel carries each GEMM's output to the next GEMM's
//! input ([`H3VideoDecoder::vit_fused`]); the residual stream stays f32.
//! FastVideo runs this decoder under fp16 autocast (fp16 GEMM operands and
//! outputs, fp32 residual), so neither side is the fp32 checkpoint's math.
//!
//! See docs/ports/h3.md, section c.

use std::sync::atomic::{AtomicBool, Ordering};

use fastvideo_models::h3::config::{H3VideoVaeConfig, H3_PIXEL_MEAN, H3_PIXEL_STD};

use crate::wan::nn::{scaled_dot_product_attention, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// `FASTVIDEO_H3_VAE_FUSED=0` keeps the ViT on the f32-activation path.
#[cfg(feature = "cuda")]
fn fused_enabled() -> bool {
    static FLAG: crate::wan::envflag::CachedBool = crate::wan::envflag::CachedBool::new();
    FLAG.get_or_init(|| crate::wan::envflag::bool_flag("FASTVIDEO_H3_VAE_FUSED", true))
}

/// `FASTVIDEO_H3_VAE_CHECK=1`: see [`H3VideoDecoder::check_paths`].
fn check_requested() -> bool {
    static FLAG: crate::wan::envflag::CachedBool = crate::wan::envflag::CachedBool::new();
    FLAG.get_or_init(|| crate::wan::envflag::bool_flag("FASTVIDEO_H3_VAE_CHECK", false))
}

/// The check has run (it runs once per process).
static CHECKED: AtomicBool = AtomicBool::new(false);
/// The check found the fused ViT disagreeing with the unfused one.
static FUSED_BROKEN: AtomicBool = AtomicBool::new(false);
/// Both paths round the same values to bf16 at the same points; what differs
/// is reduction order, so decoded pixels agree to bf16 noise, far below this.
const FUSED_CHECK_REL_L2: f64 = 1e-2;

/// `FASTVIDEO_H3_VAE_MXFP8` (a numerics change): the four block linears of
/// the fused ViT run as MXFP8 GEMMs (E4M3, one E8M0 scale per 32 values, the
/// DiT's `[techniques.mxfp8]` recipe and cuBLASLt path), weights quantized
/// from the bf16 checkpoint at load (kept beside the bf16 ones, +2.3 GiB for
/// the 36-block decoder), activations quantized by the kernel that produces
/// them. See docs/perf/h3-vae-decode-speed.md.
///
/// **Default on** (owner decision 2026-10-14) where it was measured: a live
/// device with block-scaled FP8 (sm_100+, as the DiT's MXFP8 default) and a
/// decoder of [`MXFP8_DEFAULT_LAYERS`] blocks (the 36-block H3 decoder; the
/// 26-block lightweight decoder of FastH3 Trim, gate passed 2026-10-14).
/// Unset elsewhere (pre-Blackwell GPUs, CPU runs, a decoder of another
/// depth) it means bf16, said once in the log. `=0`
/// is the bf16 decode exactly (byte-identical to the earlier default); `=1`
/// asks for MXFP8 on any decoder and is an error on a GPU without
/// block-scaled FP8 (cuBLASLt has no MXFP8 kernels below sm_100). Any other
/// value is an error.
pub const MXFP8_ENV: &str = "FASTVIDEO_H3_VAE_MXFP8";

/// Decoder depths the MXFP8 default was measured on and passed the quality
/// gate with (docs/perf/h3-vae-decode-speed.md §3 and §6): the 36-block H3
/// decoder and FastH3 Trim's 26-block one.
pub const MXFP8_DEFAULT_LAYERS: &[usize] = &[36, 26];

/// [`MXFP8_ENV`] resolved for a decoder of `layers` blocks: `explicit` is
/// the variable's value (`None` when unset), `sm_major` the live device's
/// compute capability (`None` without a device). `Ok((on, why))`: `why`
/// says why an unset variable means bf16 (for the log).
pub fn resolve_mxfp8(
    explicit: Option<&str>,
    sm_major: Option<i32>,
    layers: usize,
) -> Result<(bool, Option<String>)> {
    let explicit = explicit.map(str::trim).filter(|v| !v.is_empty());
    match explicit.map(str::to_ascii_lowercase).as_deref() {
        Some("0" | "false" | "off") => Ok((false, None)),
        Some("1" | "true" | "on") => match sm_major {
            Some(sm) if sm < 10 => Err(msg(format!(
                "{MXFP8_ENV}=1 needs block-scaled FP8 (sm_100+; this GPU is sm_{sm}x): unset it or set {MXFP8_ENV}=0 for the bf16 decode"
            ))),
            _ => Ok((true, None)),
        },
        Some(v) => Err(msg(format!("{MXFP8_ENV}={v}: expected 0 or 1"))),
        None => match sm_major {
            None => Ok((false, None)),
            Some(sm) if sm < 10 => Ok((
                false,
                Some(format!(
                    "bf16 decoder linears (MXFP8 needs sm_100+, this GPU is sm_{sm}x)"
                )),
            )),
            Some(_) if !MXFP8_DEFAULT_LAYERS.contains(&layers) => Ok((
                false,
                Some(format!(
                    "bf16 decoder linears (the MXFP8 default is measured on {MXFP8_DEFAULT_LAYERS:?}-block decoders, this one has {layers}; {MXFP8_ENV}=1 opts in)"
                )),
            )),
            Some(_) => Ok((true, None)),
        },
    }
}

/// [`resolve_mxfp8`] for this process: the variable (or the technique
/// profile's value) and the live device. Whether a decoder of `layers`
/// blocks whose linears are bf16 (`nvfp4 = false`; the published NVFP4
/// linears hold no bf16 weight to quantize) loads MXFP8 copies of its block
/// linears (the memory plan prices them with it,
/// [`fastvideo_models::h3::memory::H3PlanOptions::vae`]).
pub fn mxfp8_wanted(layers: usize, nvfp4: bool) -> Result<bool> {
    mxfp8_resolved(layers, nvfp4).map(|(on, _)| on)
}

fn mxfp8_resolved(layers: usize, nvfp4: bool) -> Result<(bool, Option<String>)> {
    let mx = fastvideo_models::techniques::settings::var(MXFP8_ENV);
    // The published NVFP4 linears replace the bf16 ones: nothing to quantize.
    if nvfp4 {
        return match mx.as_deref().map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("1" | "true" | "on") => Err(msg(format!(
                "{MXFP8_ENV}=1 with the NVFP4 decoder linears: the decoder linears are one or the other \
                 ({NVFP4_ENV}=0 decodes with h3-base's vae/, whose linears can be MXFP8)"
            ))),
            _ => Ok((false, Some("NVFP4 decoder linears (the tree's published vae/)".to_owned()))),
        };
    }
    resolve_mxfp8(mx.as_deref(), live_sm(), layers)
}

/// The live device's compute capability major, `None` without one.
fn live_sm() -> Option<i32> {
    #[cfg(feature = "cuda")]
    {
        crate::wan::stats::device_expected()
            .then(crate::wan::device::global_device)
            .flatten()
            .map(|d| d.sm_major)
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// `FASTVIDEO_H3_VAE_NVFP4` (a numerics change): the decoder's block linears
/// (`to_q/k/v`, `to_out.0`, `ff.net.0.proj`, `ff.net.2`) are read **as
/// published** in ModelOpt's NVFP4 layout (E2M1 codes, E4M3 scales per 16,
/// `weight_scale_2`, a calibrated static `input_scale`) and run on cuBLASLt
/// `VEC16_UE4M3` (W4A4), the DiT's NVFP4 GEMM path
/// ([`super::nvfp4::PublishedLinear::read_modelopt`]). Only the `vae/` of the
/// FastH3 V2 / 4-step NVFP4 releases carries them (shards 1-2 differ from
/// h3-base's, shard 3 is byte-identical; docs/ports/fasth3-nvfp4.md §1). No
/// bf16 copy is held (the tree has none), so the MXFP8 copies are off.
///
/// **Default on for a `vae/` that carries them** (owner decision
/// 2026-10-14: the NVFP4 tier decodes with its own published VAE; gate pass
/// against bf16 and MXFP8, -5 GiB of peak, docs/perf/h3-nvfp4-tier-options.md):
/// unset or `=1`, such a tree decodes with them, on sm_100 / sm_120 only (an
/// error elsewhere: there is no bf16 copy to fall back to). `=0` decodes with
/// h3-base's `vae/` instead (the text weights' tree), bf16 / MXFP8 as
/// [`MXFP8_ENV`] resolves: the decoder the tier ran before; an error when no
/// such `vae/` is found. `=1` on a tree without them is an error; unset or
/// `=0` there changes nothing. See [`resolve_nvfp4`].
pub const NVFP4_ENV: &str = "FASTVIDEO_H3_VAE_NVFP4";

/// `FASTVIDEO_H3_VAE_NVFP4_ACT=static|dynamic` (default `static`): quantize
/// the NVFP4 linears' activations with the checkpoint's calibrated
/// `input_scale`, or with the amax measured on every call.
pub const NVFP4_ACT_ENV: &str = "FASTVIDEO_H3_VAE_NVFP4_ACT";

/// `FASTVIDEO_H3_VAE_NVFP4_REF=<bf16 vae dir>`: with
/// `FASTVIDEO_H3_VAE_CHECK=1` and the NVFP4 linears, the check also decodes
/// its tile batch through the bf16 decoder of that tree (h3-base's `vae/`)
/// on the unfused, fused bf16 and fused MXFP8 paths, and logs the NVFP4
/// output's rel-L2 against each. Loaded for the check only.
pub const NVFP4_REF_ENV: &str = "FASTVIDEO_H3_VAE_NVFP4_REF";

/// The tensor whose presence says a `vae/` carries ModelOpt NVFP4 decoder linears.
pub const NVFP4_PROBE: &str = "decoder.transformer_blocks.0.attn.to_q.weight_scale";

/// The NVFP4 decode against the bf16 decoder on the check's tile batch: a
/// sanity bound (a wrong code order or scale layout gives O(1)), not a
/// quality bar; quality is the lossy gate's call on whole clips.
const NVFP4_CHECK_REL_L2: f64 = 0.25;

/// Whether the opened `vae/` files hold [`NVFP4_PROBE`] (a generated or
/// eager map never does).
pub fn carries_nvfp4(map: &WeightMap) -> bool {
    map.lazy().is_some_and(|l| l.contains(NVFP4_PROBE))
}

/// [`NVFP4_ENV`] as set (or a technique profile's value): `None` unset (the
/// tree decides), `Some(on)` for `1` / `0`.
pub fn nvfp4_setting() -> Result<Option<bool>> {
    match fastvideo_models::techniques::settings::var(NVFP4_ENV)
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("") => Ok(None),
        Some("0" | "false" | "off") => Ok(Some(false)),
        Some("1" | "true" | "on") => Ok(Some(true)),
        Some(v) => Err(msg(format!("{NVFP4_ENV}={v}: expected 0 or 1"))),
    }
}

/// Which decoder a load runs ([`resolve_nvfp4`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeLinearsChoice {
    /// The tree's own `vae/` with its published NVFP4 block linears.
    Nvfp4,
    /// The tree's own `vae/`, bf16 linears (+ MXFP8 copies as
    /// [`MXFP8_ENV`] resolves).
    Official,
    /// [`NVFP4_ENV`]`=0` on a tree whose `vae/` holds the NVFP4 linears only:
    /// h3-base's `vae/` (bf16 + MXFP8), the decoder the tier ran before.
    BaseOfficial,
}

/// [`NVFP4_ENV`] (`setting`, [`nvfp4_setting`]) resolved for a `vae/` that
/// does (`carries`) or does not carry ModelOpt NVFP4 decoder linears, with or
/// without h3-base's bf16 `vae/` at hand (`base`), on a device of `sm_major`.
/// Never a silent fallback: what cannot run is an error that names the fix.
pub fn resolve_nvfp4(
    setting: Option<bool>,
    carries: bool,
    base: bool,
    sm_major: Option<i32>,
) -> Result<VaeLinearsChoice> {
    match (setting, carries) {
        (None | Some(false), false) => Ok(VaeLinearsChoice::Official),
        (Some(true), false) => Err(msg(format!(
            "{NVFP4_ENV}=1 needs a vae/ whose decoder linears are ModelOpt NVFP4 (the FastH3 V2 / 4-step NVFP4 releases' vae/); this one has none ({NVFP4_PROBE} missing)"
        ))),
        (Some(false), true) if base => Ok(VaeLinearsChoice::BaseOfficial),
        (Some(false), true) => Err(msg(format!(
            "h3 vae: {NVFP4_ENV}=0 decodes with h3-base's bf16 vae/ (this tree's vae/ holds ModelOpt NVFP4 decoder linears only), but none was found: give the h3-base tree as the text weights (--text-weights, or the catalog's text_weights), or unset {NVFP4_ENV} to decode with the NVFP4 linears"
        ))),
        (None | Some(true), true) => match sm_major {
            Some(sm) if sm >= 10 => Ok(VaeLinearsChoice::Nvfp4),
            Some(sm) => Err(msg(format!(
                "h3 vae: this vae/ holds ModelOpt NVFP4 decoder linears (no bf16 copy), which need NVFP4 tensor cores (sm_100 / sm_120; this GPU is sm_{sm}x): set {NVFP4_ENV}=0 to decode with h3-base's vae/ instead"
            ))),
            None => Err(msg(format!(
                "h3 vae: this vae/ holds ModelOpt NVFP4 decoder linears (no bf16 copy), which need a CUDA device with NVFP4 tensor cores: set {NVFP4_ENV}=0 to decode with h3-base's vae/ instead"
            ))),
        },
    }
}

/// [`resolve_nvfp4`] for `map` (an opened `vae/`) on the live device, with
/// no h3-base `vae/` at hand (the pipeline resolves its own source with one,
/// `super::pipeline::resolve_video_vae`).
pub fn nvfp4_for(map: &WeightMap) -> Result<bool> {
    Ok(resolve_nvfp4(nvfp4_setting()?, carries_nvfp4(map), false, live_sm())? == VaeLinearsChoice::Nvfp4)
}

/// The live device's compute capability major (`None` without one).
pub fn device_sm() -> Option<i32> {
    live_sm()
}

fn nvfp4_static_act() -> Result<bool> {
    match fastvideo_models::techniques::settings::var(NVFP4_ACT_ENV)
        .as_deref()
        .map(str::trim)
    {
        None | Some("" | "static") => Ok(true),
        Some("dynamic") => Ok(false),
        Some(v) => Err(msg(format!("{NVFP4_ACT_ENV}={v}: expected static|dynamic"))),
    }
}

/// The check found the NVFP4 forward off by more than [`NVFP4_CHECK_REL_L2`]
/// (logged; the NVFP4 output is still used: there is no bf16 copy).
static NVFP4_OFF: AtomicBool = AtomicBool::new(false);

/// Which weights a decoder's block linears hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearMode {
    /// bf16 (+ MXFP8 copies when [`mxfp8_resolved`] says so).
    Official,
    /// The published ModelOpt NVFP4 linears ([`NVFP4_ENV`]); `static_act`:
    /// the calibrated activation scales.
    Nvfp4 { static_act: bool },
}

/// The MXFP8 forward against the unfused f32-activation ViT, in
/// [`H3VideoDecoder::check_paths`]: a sanity bound (wrong scales or layout
/// give O(1)), not a quality bar; quality is the gate's call.
const MXFP8_CHECK_REL_L2: f64 = 1e-1;

/// The check found the MXFP8 forward off by more than [`MXFP8_CHECK_REL_L2`].
static MXFP8_BROKEN: AtomicBool = AtomicBool::new(false);

/// [`with_bf16_linears`] is running: the fused ViT takes its bf16 GEMMs.
static MXFP8_SUSPENDED: AtomicBool = AtomicBool::new(false);

/// `FASTVIDEO_H3_VAE_AB=1` (measurement only, off by default): after each
/// clip's video decode, when that decode ran the MXFP8 linears, decode the
/// same latents again with the bf16 linears (the `=0` arithmetic, from the
/// bf16 weights kept beside the MXFP8 copies) into `<clip>/vae-bf16/`, so one
/// process gives both decodes of identical latents (the DiT runs once).
pub const AB_ENV: &str = "FASTVIDEO_H3_VAE_AB";

/// Whether [`AB_ENV`] asks for the bf16 re-decode.
pub fn ab_requested() -> bool {
    static FLAG: crate::wan::envflag::CachedBool = crate::wan::envflag::CachedBool::new();
    FLAG.get_or_init(|| crate::wan::envflag::bool_flag(AB_ENV, false))
}

/// Run `f` with every decoder's block GEMMs in bf16 even where MXFP8 copies
/// are loaded (for [`AB_ENV`]): the same kernels and weights as a decoder
/// loaded with `FASTVIDEO_H3_VAE_MXFP8=0`. Not for concurrent decodes.
pub fn with_bf16_linears<R>(f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            MXFP8_SUSPENDED.store(false, Ordering::Relaxed);
        }
    }
    MXFP8_SUSPENDED.store(true, Ordering::Relaxed);
    let _restore = Restore;
    f()
}

/// `FASTVIDEO_H3_VAE_VEC` (default on; `=0` restores the scalar kernels):
/// the vector-access twins of the merge-heads and SwiGLU kernels, whose
/// outputs are bit-identical (768p video VAE 6.30 -> 6.20 s, frames
/// byte-identical; docs/perf/h3-vae-decode-speed.md).
#[cfg(feature = "cuda")]
fn vec_enabled() -> bool {
    static FLAG: crate::wan::envflag::CachedBool = crate::wan::envflag::CachedBool::new();
    FLAG.get_or_init(|| crate::wan::envflag::bool_flag("FASTVIDEO_H3_VAE_VEC", VEC_DEFAULT))
}

/// Default of [`vec_enabled`].
#[cfg(feature = "cuda")]
const VEC_DEFAULT: bool = true;

/// `FASTVIDEO_H3_VAE_GEMM`: `cublas` (default; cuBLAS `GemmEx`'s own pick),
/// `lt` or `lt:N` / `ltN` (cuBLASLt's first or N-th heuristic algorithm) for the
/// fused ViT's bf16 GEMMs. Same product either way up to accumulation order.
#[cfg(feature = "cuda")]
fn gemm_algo() -> Option<usize> {
    static FLAG: crate::wan::envflag::CachedString = crate::wan::envflag::CachedString::new();
    let v =
        FLAG.get_or_init(|| crate::wan::envflag::string_flag("FASTVIDEO_H3_VAE_GEMM", "cublas"));
    parse_algo(&v, "lt")
}

/// `FASTVIDEO_H3_VAE_MXFP8_ALGO=N`: cuBLASLt's N-th heuristic algorithm for
/// the MXFP8 GEMMs (default 0, its first pick).
#[cfg(feature = "cuda")]
fn mx_algo() -> usize {
    crate::wan::envflag::usize_flag("FASTVIDEO_H3_VAE_MXFP8_ALGO", 0)
}

/// `lt` -> 0, `lt:N` or `ltN` -> N, anything else -> `None`.
#[cfg(feature = "cuda")]
fn parse_algo(v: &str, prefix: &str) -> Option<usize> {
    let v = v.trim();
    if v == prefix {
        return Some(0);
    }
    v.strip_prefix(prefix)
        .map(|r| r.strip_prefix(':').unwrap_or(r))
        .and_then(|n| n.parse().ok())
}

/// `FASTVIDEO_H3_VAE_SDPA=auto|v1|v2|cudnn` (default `auto`, the dense-SDPA
/// seam's pick): the fused kernel for the ViT's attention only.
#[cfg(feature = "cuda")]
fn sdpa_kernel() -> Option<crate::wan::attn::FlashKernel> {
    use crate::wan::attn::FlashKernel;
    static FLAG: crate::wan::envflag::CachedString = crate::wan::envflag::CachedString::new();
    match FLAG
        .get_or_init(|| crate::wan::envflag::string_flag("FASTVIDEO_H3_VAE_SDPA", "auto"))
        .trim()
    {
        "v1" => Some(FlashKernel::V1),
        "v2" => Some(FlashKernel::V2),
        "v3" => Some(FlashKernel::V3),
        "v3s" => Some(FlashKernel::V3s),
        "cudnn" => Some(FlashKernel::Cudnn),
        _ => None,
    }
}

/// A fused-ViT GEMM operand: bf16 rows, or MXFP8 codes + scales.
#[cfg(feature = "cuda")]
enum Act {
    Bf16(cudarc::driver::CudaSlice<half::bf16>),
    Mx(crate::wan::quant::MxAct),
}

/// Per-op wall time of one ViT forward, each op synchronized: only for
/// [`H3VideoDecoder::check_paths`]; off, it is a plain call.
#[derive(Default)]
struct Prof {
    on: bool,
    acc: Vec<(&'static str, f64)>,
}

impl Prof {
    fn on() -> Self {
        Self {
            on: true,
            acc: Vec::new(),
        }
    }

    fn time<T>(&mut self, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        if !self.on {
            return f();
        }
        let sync = || crate::wan::device::synchronize().map_err(|e| msg(e.to_string()));
        sync()?;
        let timer = std::time::Instant::now();
        let out = f()?;
        sync()?;
        let secs = timer.elapsed().as_secs_f64();
        match self.acc.iter_mut().find(|(n, _)| *n == name) {
            Some((_, s)) => *s += secs,
            None => self.acc.push((name, secs)),
        }
        Ok(out)
    }

    fn summary(&self) -> String {
        let mut acc = self.acc.clone();
        acc.sort_by(|a, b| b.1.total_cmp(&a.1));
        acc.iter()
            .map(|(n, s)| format!("{n} {s:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn pinned(data: Vec<f32>, shape: Vec<usize>) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(data, shape)?;
    t.pin_device()?;
    Ok(t)
}

fn pinned_weight(map: &WeightMap, key: &str, shape: &[usize]) -> Result<CudaTensor> {
    let mut t = cuda_tensor_shaped(map, key, shape)?;
    t.pin_device()?;
    Ok(t)
}

/// `prefixes` (each `[rows, k]`, ModelOpt NVFP4 with an f32 bias) as one
/// bias-carrying [`Linear::from_nvfp4`] stack of `rows · prefixes.len()`
/// outputs. With `static_act`, the activation is quantized with the largest
/// of the parts' calibrated `input_scale`s (the parts read one input); the
/// scale used is returned.
#[cfg(feature = "cuda")]
fn nvfp4_linear(
    map: &WeightMap,
    prefixes: &[&str],
    rows: usize,
    k: usize,
    static_act: bool,
) -> Result<(Linear, Option<f32>)> {
    use crate::wan::nvfp4_linear::{Nvfp4Linear, Nvfp4Operand, Nvfp4Stack};
    let rule = fastvideo_models::nvfp4::ScaleRule::Static6;
    let store = map
        .lazy()
        .ok_or_else(|| msg("h3 vae nvfp4: the vae/ weights are not mapped"))?;
    let mut parts = Vec::with_capacity(prefixes.len());
    let mut bias = Vec::with_capacity(rows * prefixes.len());
    let mut act: Option<f32> = None;
    for p in prefixes {
        let (l, input_scale) = super::nvfp4::PublishedLinear::read_modelopt(store, p, rows, k)?;
        if static_act {
            let s = input_scale
                .ok_or_else(|| msg(format!("h3 vae nvfp4: {p}.input_scale missing ({NVFP4_ACT_ENV}=dynamic measures it)")))?;
            act = Some(act.map_or(s, |a| a.max(s)));
        }
        let op = Nvfp4Operand::from_host_layout(&l.codes, &l.scales, l.rows, l.k, l.decode_scale, rule)?;
        parts.push(Nvfp4Linear::from_operand(op, None, rule)?);
        bias.extend(cuda_tensor_shaped(map, &format!("{p}.bias"), &[rows])?.host_cow()?.iter());
    }
    let mut stack = Nvfp4Stack::new(parts)?;
    if let Some(s) = act {
        stack = stack.with_static_act_scale(s)?;
    }
    let mut lin = Linear::from_nvfp4(stack)?;
    lin.bias = Some(pinned(bias, vec![rows * prefixes.len()])?);
    Ok((lin, act))
}

struct Block {
    norm1: CudaTensor,
    norm2: CudaTensor,
    /// LayerScale on the attention and FFN branches, `[dim]`.
    scale1: CudaTensor,
    scale2: CudaTensor,
    /// `to_q`, `to_k`, `to_v` stacked in that order: one GEMM, `[3 dim, dim]`.
    qkv: Linear,
    to_out: Linear,
    ff_in: Linear,
    ff_out: Linear,
    /// MXFP8 copies of `[qkv, to_out, ff_in, ff_out]` (`FASTVIDEO_H3_VAE_MXFP8`).
    #[cfg(feature = "cuda")]
    mx: Option<Box<[crate::wan::quant::QuantWeight; 4]>>,
}

impl Block {
    fn load(map: &WeightMap, prefix: &str, cfg: &H3VideoVaeConfig) -> Result<Self> {
        let dim = cfg.decoder_dim();
        let hidden = dim * cfg.decoder_ffn_mult;
        let lin = |name: &str, i: usize, o: usize| {
            Linear::load(map, &format!("{prefix}.{name}"), i, o, true)
        };
        let vec = |name: &str| pinned_weight(map, &format!("{prefix}.{name}"), &[dim]);
        let qkv: Vec<String> = ["to_q", "to_k", "to_v"]
            .iter()
            .map(|p| format!("{prefix}.attn.{p}"))
            .collect();
        let qkv: Vec<&str> = qkv.iter().map(String::as_str).collect();
        Ok(Self {
            norm1: vec("norm1.weight")?,
            norm2: vec("norm2.weight")?,
            scale1: vec("scale1")?,
            scale2: vec("scale2")?,
            qkv: Linear::load_fused(map, &qkv, dim, dim, true)?,
            to_out: lin("attn.to_out.0", dim, dim)?,
            // SwiGLU: one projection to (value, gate), value half first.
            ff_in: lin("ff.net.0.proj", dim, 2 * hidden)?,
            ff_out: lin("ff.net.2", hidden, dim)?,
            #[cfg(feature = "cuda")]
            mx: None,
        })
    }

    /// The block with its four linears read as published ModelOpt NVFP4
    /// ([`NVFP4_ENV`]): `to_q/k/v` as one stack (one activation quantization,
    /// each part its own global scale; the largest of their three calibrated
    /// activation scales), biases f32 as published.
    #[cfg(feature = "cuda")]
    fn load_nvfp4(
        map: &WeightMap,
        prefix: &str,
        cfg: &H3VideoVaeConfig,
        static_act: bool,
    ) -> Result<(Self, [Option<f32>; 4])> {
        let dim = cfg.decoder_dim();
        let hidden = dim * cfg.decoder_ffn_mult;
        let vec = |name: &str| pinned_weight(map, &format!("{prefix}.{name}"), &[dim]);
        let qkv: Vec<String> = ["to_q", "to_k", "to_v"]
            .iter()
            .map(|p| format!("{prefix}.attn.{p}"))
            .collect();
        let qkv: Vec<&str> = qkv.iter().map(String::as_str).collect();
        let (qkv, s0) = nvfp4_linear(map, &qkv, dim, dim, static_act)?;
        let (to_out, s1) = nvfp4_linear(map, &[&format!("{prefix}.attn.to_out.0")], dim, dim, static_act)?;
        let (ff_in, s2) = nvfp4_linear(map, &[&format!("{prefix}.ff.net.0.proj")], 2 * hidden, dim, static_act)?;
        let (ff_out, s3) = nvfp4_linear(map, &[&format!("{prefix}.ff.net.2")], dim, hidden, static_act)?;
        Ok((
            Self {
                norm1: vec("norm1.weight")?,
                norm2: vec("norm2.weight")?,
                scale1: vec("scale1")?,
                scale2: vec("scale2")?,
                qkv,
                to_out,
                ff_in,
                ff_out,
                mx: None,
            },
            [s0, s1, s2, s3],
        ))
    }

    /// Quantize the four block linears' device bf16 weights to MXFP8 (the
    /// same kernel the DiT recipe quantizes its weights with).
    #[cfg(feature = "cuda")]
    fn quantize_mx(&mut self) -> Result<()> {
        use crate::wan::quant::{QuantKind, QuantLayout, QuantWeight, Section};
        let one = |l: &Linear| -> Result<QuantWeight> {
            let w = l
                .weight_bf16_shared()
                .ok_or_else(|| msg("h3 vae mxfp8: linear has no device bf16 weight"))?;
            let layout = QuantLayout::new(
                QuantKind::Mxfp8,
                l.in_dim(),
                vec![Section {
                    rows: l.out_dim(),
                    quantized: true,
                }],
            )?;
            QuantWeight::from_device_bf16(layout, &w)
        };
        self.mx = Some(Box::new([
            one(&self.qkv)?,
            one(&self.to_out)?,
            one(&self.ff_in)?,
            one(&self.ff_out)?,
        ]));
        Ok(())
    }
}

/// `[tokens, rotary]` cos/sin for a `(t, h, w)` grid plus `extra` suffix tokens
/// at angle zero. Float32 in the reference's order of operations: positions
/// `2 (i + 0.5) / size - 1`, angles `(2 pi * pos) * theta^(-k / n)`, laid out
/// `cat(A, A)` with `A = [t x n | h x n | w x n]`.
fn rope_tables(grid: [usize; 3], extra: usize, rotary: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let per_axis = rotary / 6;
    let inv_freq: Vec<f32> = (0..per_axis)
        .map(|k| {
            // torch.arange(0, 1, 6 / rotary) in float32, then 1 / theta ** it.
            let exponent = k as f32 * (6.0f32 / rotary as f32);
            1.0f32 / (theta as f32).powf(exponent)
        })
        .collect();
    let two_pi = (2.0 * std::f64::consts::PI) as f32;
    let axis = |size: usize| -> Vec<f32> {
        (0..size)
            .map(|i| 2.0f32 * ((i as f32 + 0.5) / size as f32) - 1.0)
            .collect()
    };
    let (pt, ph, pw) = (axis(grid[0]), axis(grid[1]), axis(grid[2]));
    let tokens = grid[0] * grid[1] * grid[2] + extra;
    let (mut cos, mut sin) = (vec![1.0f32; tokens * rotary], vec![0.0f32; tokens * rotary]);
    let half = rotary / 2;
    for (ti, &t) in pt.iter().enumerate() {
        for (hi, &h) in ph.iter().enumerate() {
            for (wi, &w) in pw.iter().enumerate() {
                let row = ((ti * grid[1] + hi) * grid[2] + wi) * rotary;
                for (a, pos) in [t, h, w].into_iter().enumerate() {
                    for (k, f) in inv_freq.iter().enumerate() {
                        let angle = (two_pi * pos) * f;
                        for j in [a * per_axis + k, half + a * per_axis + k] {
                            cos[row + j] = angle.cos();
                            sin[row + j] = angle.sin();
                        }
                    }
                }
            }
        }
    }
    (cos, sin)
}

/// `a * (1 - j/E) + b * (j/E)` over the last `E` entries of `a` and the first
/// `E` of `b` along `dim`, followed by the rest of `b` (`_blend`). The first
/// blended entry is entirely `a`.
fn blend(a: &CudaTensor, b: &CudaTensor, extent: usize, dim: usize) -> Result<CudaTensor> {
    let extent = extent.min(a.shape[dim]).min(b.shape[dim]);
    if extent == 0 {
        return Ok(b.clone());
    }
    let ramp: Vec<f32> = (0..extent).map(|j| j as f32 / extent as f32).collect();
    let mut shape = vec![1usize; a.rank()];
    shape[dim] = extent;
    let weight_a = CudaTensor::from_vec(ramp.iter().map(|w| 1.0 - w).collect(), shape.clone())?;
    let weight_b = CudaTensor::from_vec(ramp, shape)?;
    let tail = a.narrow(dim, a.shape[dim] - extent, extent)?;
    let head = b.narrow(dim, 0, extent)?;
    let blended = tail.mul(&weight_a)?.add(&head.mul(&weight_b)?)?;
    if extent == b.shape[dim] {
        return Ok(blended);
    }
    CudaTensor::cat(
        &[&blended, &b.narrow(dim, extent, b.shape[dim] - extent)?],
        dim,
    )
}

/// Tiles per ViT forward: as few forwards as `max` allows, split evenly (28
/// tiles at a cap of 8 run as 4 x 7, not 8 + 8 + 8 + 4).
fn balanced_batch(total: usize, max: usize) -> usize {
    let max = max.max(1);
    total.div_ceil(total.div_ceil(max).max(1)).max(1)
}

/// Tile starts, tile length and overlaps along one axis, in **latent** units.
struct AxisTiles {
    starts: Vec<usize>,
    len: usize,
    overlaps: Vec<usize>,
}

pub struct H3VideoDecoder {
    cfg: H3VideoVaeConfig,
    /// `latents_std` / `latents_mean`, `[C]`, applied on channel-last latents.
    std: CudaTensor,
    mean: CudaTensor,
    /// The 1x1x1 `post_quant_conv`, which is a per-voxel linear.
    post_quant: Linear,
    proj_in: Linear,
    /// The learned register tokens followed by the all-zero token, `[1, R + 1, dim]`.
    suffix: CudaTensor,
    blocks: Vec<Block>,
    norm_out_weight: CudaTensor,
    norm_out_bias: CudaTensor,
    proj_out: Linear,
    /// Unit weight for the parameter-free per-head QK norm.
    qk_unit: CudaTensor,
    /// Tiles per ViT forward; they share rotary tables, so this is only a
    /// throughput / memory trade.
    tile_batch: usize,
    /// The block linears are the published NVFP4 ones ([`NVFP4_ENV`]).
    nvfp4: bool,
    /// The bf16 decoder [`NVFP4_REF_ENV`] names, for the check only.
    reference: Option<Box<H3VideoDecoder>>,
}

impl H3VideoDecoder {
    /// Reads `decoder.*` and `post_quant_conv.*`; the convolutional encoder is
    /// never touched. A `vae/` carrying ModelOpt NVFP4 decoder linears loads
    /// them (the default, [`NVFP4_ENV`]); `=0` is refused here, since this
    /// map has no bf16 linears (the pipeline opens h3-base's `vae/` for it).
    pub fn load(cfg: H3VideoVaeConfig, map: &WeightMap) -> Result<Self> {
        let nv = nvfp4_for(map)?;
        Self::load_choice(cfg, map, nv)
    }

    /// [`Self::load`] with the choice made: `nvfp4` reads the published
    /// NVFP4 block linears (the map must carry them), otherwise the bf16 ones.
    pub fn load_choice(cfg: H3VideoVaeConfig, map: &WeightMap, nvfp4: bool) -> Result<Self> {
        if nvfp4 && !carries_nvfp4(map) {
            return Err(msg(format!(
                "h3 vae: NVFP4 decoder linears asked for, but this vae/ has none ({NVFP4_PROBE} missing)"
            )));
        }
        if !nvfp4 {
            return Self::load_with(cfg, map, LinearMode::Official);
        }
        let mut dec = Self::load_with(
            cfg,
            map,
            LinearMode::Nvfp4 {
                static_act: nvfp4_static_act()?,
            },
        )?;
        if check_requested() {
            if let Some(dir) = fastvideo_models::techniques::settings::var(NVFP4_REF_ENV)
                .filter(|v| !v.trim().is_empty())
            {
                let dir = std::path::PathBuf::from(dir.trim());
                let cfg_ref = dec.cfg.clone();
                let files = super::pipeline::vae_files(&dir).map_err(|e| msg(e.to_string()))?;
                let ref_map = WeightMap::open_files(&files)?;
                // bf16 + the MXFP8 copies (sm_100+), as the default decoder.
                let mx = resolve_mxfp8(None, live_sm(), cfg_ref.decoder_num_layers)?.0;
                let mut r = Self::load_with(cfg_ref, &ref_map, LinearMode::Official)?;
                #[cfg(feature = "cuda")]
                if mx && fused_enabled() && crate::wan::stats::device_expected() {
                    for b in &mut r.blocks {
                        b.quantize_mx()?;
                    }
                }
                crate::wan::log::info(format_args!(
                    "h3 vae check: bf16 reference decoder from {} (MXFP8 copies {})",
                    dir.display(),
                    if r.mx_ready() { "on" } else { "off" }
                ));
                dec.reference = Some(Box::new(r));
            }
        }
        Ok(dec)
    }

    fn load_with(cfg: H3VideoVaeConfig, map: &WeightMap, mode: LinearMode) -> Result<Self> {
        let (dim, lc, head_dim) = (
            cfg.decoder_dim(),
            cfg.latent_channels,
            cfg.decoder_attention_head_dim,
        );
        let rotary = cfg.decoder_rotary_dim();
        if rotary == 0 || rotary % 6 != 0 || rotary > head_dim {
            return Err(msg(format!(
                "h3 vae: {rotary} rotary channels of a {head_dim}-wide head is not 3 axes of pairs"
            )));
        }
        if cfg.token_drop == 0 || cfg.token_drop >= cfg.tokens_chunk_size() {
            return Err(msg(format!(
                "h3 vae: token_drop {} outside (0, {})",
                cfg.token_drop,
                cfg.tokens_chunk_size()
            )));
        }
        let pq = cuda_tensor_shaped(map, "post_quant_conv.weight", &[lc, lc, 1, 1, 1])?
            .reshape(vec![lc, lc])?;
        let pq_bias = cuda_tensor_shaped(map, "post_quant_conv.bias", &[lc])?;
        let registers = cfg.decoder_num_register_tokens;
        let mut suffix = cuda_tensor_shaped(map, "decoder.register_tokens", &[1, registers, dim])?
            .host_cow()?
            .into_owned();
        suffix.extend(std::iter::repeat_n(0.0f32, dim));
        #[allow(unused_mut)]
        let nvfp4 = matches!(mode, LinearMode::Nvfp4 { .. });
        let mut blocks = match mode {
            LinearMode::Official => (0..cfg.decoder_num_layers)
                .map(|i| Block::load(map, &format!("decoder.transformer_blocks.{i}"), &cfg))
                .collect::<Result<Vec<_>>>()?,
            LinearMode::Nvfp4 { static_act } => {
                #[cfg(feature = "cuda")]
                {
                    let t = std::time::Instant::now();
                    let mut scales: Vec<[Option<f32>; 4]> = Vec::new();
                    let blocks = (0..cfg.decoder_num_layers)
                        .map(|i| {
                            Block::load_nvfp4(map, &format!("decoder.transformer_blocks.{i}"), &cfg, static_act)
                                .map(|(b, s)| {
                                    scales.push(s);
                                    b
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let held: u64 = blocks
                        .iter()
                        .flat_map(|b| [&b.qkv, &b.to_out, &b.ff_in, &b.ff_out])
                        .map(Linear::held_bytes)
                        .sum();
                    let range = |j: usize| {
                        let v: Vec<f32> = scales.iter().filter_map(|s| s[j]).collect();
                        if v.is_empty() {
                            "dynamic".to_string()
                        } else {
                            let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                            format!("{lo:.3e}..{hi:.3e}")
                        }
                    };
                    crate::wan::log::info(format_args!(
                        "h3 vae: NVFP4 decoder ({NVFP4_ENV} {}): {} blocks x 4 linears as published ModelOpt NVFP4 (W4A4, E4M3 per 16; {:.2} GiB with biases, loaded in {:.1}s); activation scales {}: qkv {}, to_out {}, ff_in {}, ff_out {}",
                        if fastvideo_models::techniques::settings::var(NVFP4_ENV).is_some() { "=1" } else { "default: the tree carries them" },
                        blocks.len(),
                        held as f64 / f64::from(1u32 << 30),
                        t.elapsed().as_secs_f64(),
                        if static_act { "static (calibrated input_scale)" } else { "dynamic" },
                        range(0),
                        range(1),
                        range(2),
                        range(3)
                    ));
                    blocks
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = static_act;
                    return Err(msg(format!("{NVFP4_ENV}=1 needs a CUDA build")));
                }
            }
        };
        let (mx_on, mx_why) = mxfp8_resolved(blocks.len(), nvfp4)?;
        if let Some(why) = mx_why {
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                crate::wan::log::info(format_args!("h3 vae: {MXFP8_ENV} unset: {why}"));
            }
        }
        // The MXFP8 GEMMs run only inside the fused ViT, so without it there
        // is nothing to quantize for.
        #[cfg(feature = "cuda")]
        if mx_on
            && crate::wan::stats::device_expected()
            && fused_enabled()
            && blocks.iter().all(|b| {
                [&b.qkv, &b.to_out, &b.ff_in, &b.ff_out]
                    .iter()
                    .all(|l| l.has_bf16_gemm())
            })
        {
            for b in &mut blocks {
                b.quantize_mx()?;
            }
            crate::wan::log::info(format_args!(
                "h3 vae: {MXFP8_ENV} {}: {} blocks x 4 linears as MXFP8 (E4M3, E8M0 per 32); {MXFP8_ENV}=0 for bf16",
                if fastvideo_models::techniques::settings::var(MXFP8_ENV).is_some() {
                    "=1"
                } else {
                    "default"
                },
                blocks.len()
            ));
        }
        #[cfg(not(feature = "cuda"))]
        let _ = mx_on;
        let to_vec = |v: &[f64]| pinned(v.iter().take(lc).map(|&x| x as f32).collect(), vec![lc]);
        Ok(Self {
            std: to_vec(&cfg.latents_std)?,
            mean: to_vec(&cfg.latents_mean)?,
            post_quant: Linear::from_tensors(pq, Some(pq_bias))?,
            proj_in: Linear::load(map, "decoder.proj_in", lc, dim, true)?,
            suffix: pinned(suffix, vec![1, registers + 1, dim])?,
            blocks,
            norm_out_weight: pinned_weight(map, "decoder.norm_out.weight", &[dim])?,
            norm_out_bias: pinned_weight(map, "decoder.norm_out.bias", &[dim])?,
            proj_out: Linear::load(map, "decoder.proj_out", dim, cfg.decoder_patch_dim(), true)?,
            qk_unit: pinned(vec![1.0; head_dim], vec![head_dim])?,
            tile_batch: crate::wan::envflag::usize_flag("FASTVIDEO_H3_VAE_TILE_BATCH", 8).max(1),
            nvfp4,
            reference: None,
            cfg,
        })
    }

    pub fn config(&self) -> &H3VideoVaeConfig {
        &self.cfg
    }

    /// Whether a decode now runs the MXFP8 block linears (loaded, not
    /// refused by the check, not inside [`with_bf16_linears`]). The fused
    /// ViT must also take the shape; every H3 decode geometry does.
    pub fn mxfp8_active(&self) -> bool {
        self.mx_ready()
    }

    pub fn with_tile_batch(mut self, tiles: usize) -> Self {
        self.tile_batch = tiles.max(1);
        self
    }

    /// Pixel frames a `latent_frames`-long clip decodes to.
    pub fn decoded_frames(&self, latent_frames: usize) -> usize {
        self.cfg.temporal_decode_plan(latent_frames).2
    }

    /// The ViT on `tiles`: `[B, t*h*w, C]` channel-last **denormalized**
    /// latents of one `(t, h, w)` geometry, to `B` clips of `[3, 4t, 16h, 16w]`.
    fn decode_tiles(&self, tiles: &CudaTensor, grid: [usize; 3]) -> Result<Vec<CudaTensor>> {
        if check_requested() && !CHECKED.swap(true, Ordering::Relaxed) {
            return self.check_paths(tiles, grid);
        }
        self.decode_tiles_observed(tiles, grid, &mut |_, _| Ok(()))
    }

    /// [`Self::decode_tiles`], showing `observe(block, x)` each ViT block's
    /// `[B, tokens + 5, dim]` output (the diffusers reference tests hook the same point).
    pub(crate) fn decode_tiles_observed(
        &self,
        tiles: &CudaTensor,
        grid: [usize; 3],
        observe: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<Vec<CudaTensor>> {
        let fused = self.fused_ready(tiles.shape.first().copied().unwrap_or(0), grid);
        let mx = fused && self.mx_ready();
        self.decode_tiles_with(tiles, grid, observe, fused, mx, &mut Prof::default())
    }

    /// Whether the fused ViT runs its block GEMMs as MXFP8: requested, the
    /// weights quantized at load, and the check (if run) did not refuse it.
    fn mx_ready(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            !MXFP8_BROKEN.load(Ordering::Relaxed)
                && !MXFP8_SUSPENDED.load(Ordering::Relaxed)
                && self.blocks.iter().all(|b| b.mx.is_some())
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Whether the ViT can run with bf16 activations between its GEMMs
    /// ([`Self::vit_fused`]): a device with the fast bf16 path, every block
    /// linear a plain device bf16 weight, the fused SDPA kernel for this
    /// shape, and `FASTVIDEO_H3_VAE_FUSED` not set to 0.
    fn fused_ready(&self, batch: usize, grid: [usize; 3]) -> bool {
        #[cfg(feature = "cuda")]
        {
            let cfg = &self.cfg;
            let (heads, head_dim) = (
                cfg.decoder_num_attention_heads,
                cfg.decoder_attention_head_dim,
            );
            let tokens = grid.iter().product::<usize>() + self.suffix.shape[1];
            let Some(dev) = crate::wan::device::global_device() else {
                return false;
            };
            !FUSED_BROKEN.load(Ordering::Relaxed)
                && fused_enabled()
                && crate::wan::stats::device_expected()
                && !crate::wan::tensor::bf16_activations()
                && crate::wan::attn::mma_sdpa_default()
                && crate::wan::attn::mma_sdpa_supported(
                    batch,
                    heads,
                    tokens,
                    tokens,
                    head_dim,
                    dev.sm_major,
                )
                && head_dim % 32 == 0
                && head_dim <= 128
                && cfg.decoder_rotary_dim().is_multiple_of(2)
                && self.blocks.iter().all(|b| {
                    [&b.qkv, &b.to_out, &b.ff_in, &b.ff_out]
                        .iter()
                        .all(|l| (l.has_bf16_gemm() || l.is_nvfp4_published()) && l.bias.is_some())
                })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, grid);
            false
        }
    }

    fn decode_tiles_with(
        &self,
        tiles: &CudaTensor,
        grid: [usize; 3],
        observe: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
        fused: bool,
        mx: bool,
        prof: &mut Prof,
    ) -> Result<Vec<CudaTensor>> {
        let cfg = &self.cfg;
        let eps = cfg.decoder_norm_eps as f32;
        let [batch, patches, _] = tiles.shape[..] else {
            return Err(msg(format!(
                "h3 vae tiles must be [B, tokens, C], got {:?}",
                tiles.shape
            )));
        };
        if patches != grid.iter().product::<usize>() {
            return Err(msg(format!("h3 vae: {patches} tokens for grid {grid:?}")));
        }
        let extra = self.suffix.shape[1];
        let tokens = patches + extra;
        let rotary = cfg.decoder_rotary_dim();
        let (cos, sin) = rope_tables(grid, extra, rotary, cfg.decoder_rope_theta);
        let (cos, sin) = (
            pinned(cos, vec![tokens, rotary])?,
            pinned(sin, vec![tokens, rotary])?,
        );

        let x = prof.time("vae_proj_in", || {
            let x = self.proj_in.forward(&self.post_quant.forward(tiles)?)?;
            let suffixes: Vec<&CudaTensor> = (0..batch).map(|_| &self.suffix).collect();
            let suffix = CudaTensor::cat(&suffixes, 0)?;
            CudaTensor::cat(&[&x, &suffix], 1)
        })?;
        let x = if fused {
            #[cfg(feature = "cuda")]
            {
                self.vit_fused(x, batch, tokens, (&cos, &sin), observe, mx, prof)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = mx;
                return Err(msg("h3 vae: the fused ViT needs the cuda feature"));
            }
        } else {
            self.vit_unfused(x, batch, tokens, (&cos, &sin), observe, prof)?
        };
        let x = prof.time("vae_proj_out", || {
            let x = x.layer_norm(eps, Some(&self.norm_out_weight), Some(&self.norm_out_bias))?;
            self.proj_out.forward(&x.narrow(1, 0, patches)?)
        })?;

        // [t, h, w, C, pt, ph, pw] -> [C, t*pt, h*ph, w*pw]. The device permute
        // stops at rank 6, so move (C, pt) first with the pixel patch folded,
        // then interleave the spatial axes.
        let (pt, ps, oc) = (
            cfg.temporal_compression_ratio(),
            cfg.spatial_compression_ratio(),
            cfg.out_channels,
        );
        let [t, h, w] = grid;
        prof.time("vae_unpatchify", || {
            (0..batch)
                .map(|b| {
                    x.narrow(0, b, 1)?
                        .reshape(vec![t, h, w, oc, pt, ps * ps])?
                        .permute(&[3, 0, 4, 1, 2, 5])?
                        .reshape(vec![oc * t * pt, h, w, ps, ps])?
                        .permute(&[0, 1, 3, 2, 4])?
                        .reshape(vec![oc, t * pt, h * ps, w * ps])
                })
                .collect()
        })
    }

    /// The 36 blocks with f32 activations around every op: the reference
    /// formulation, and the path for CPU runs and exact (non-bf16) modes.
    fn vit_unfused(
        &self,
        mut x: CudaTensor,
        batch: usize,
        tokens: usize,
        (cos, sin): (&CudaTensor, &CudaTensor),
        observe: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
        prof: &mut Prof,
    ) -> Result<CudaTensor> {
        let cfg = &self.cfg;
        let (dim, heads, head_dim) = (
            cfg.decoder_dim(),
            cfg.decoder_num_attention_heads,
            cfg.decoder_attention_head_dim,
        );
        let eps = cfg.decoder_norm_eps as f32;
        let split = |t: CudaTensor, norm: bool| -> Result<CudaTensor> {
            let t = t.reshape(vec![batch, tokens, heads, head_dim])?;
            // Per-head RMSNorm without a learned weight, then [B, H, S, D].
            let t = if norm {
                t.rms_norm(&self.qk_unit, eps)?
            } else {
                t
            };
            t.transpose(1, 2)
        };
        for (index, block) in self.blocks.iter().enumerate() {
            let n = prof.time("vae_norm", || x.rms_norm(&block.norm1, eps))?;
            let qkv = prof.time("vae_qkv_linear", || block.qkv.forward(&n))?;
            let (q, k, v) = prof.time("vae_qkv_heads", || {
                Ok((
                    split(qkv.narrow(2, 0, dim)?, true)?.rope_half(cos, sin)?,
                    split(qkv.narrow(2, dim, dim)?, true)?.rope_half(cos, sin)?,
                    split(qkv.narrow(2, 2 * dim, dim)?, false)?,
                ))
            })?;
            drop(qkv);
            let a = prof.time("vae_sdpa", || {
                scaled_dot_product_attention(&q, &k, &v, None)
            })?;
            let a = prof.time("vae_out_linear", || {
                block
                    .to_out
                    .forward(&a.transpose(1, 2)?.reshape(vec![batch, tokens, dim])?)
            })?;
            x = prof.time("vae_residual", || x.add(&a.mul(&block.scale1)?))?;

            let n = prof.time("vae_norm", || x.rms_norm(&block.norm2, eps))?;
            let h = prof.time("vae_ff_in_linear", || block.ff_in.forward(&n))?;
            let act = prof.time("vae_swiglu", || {
                let hidden = h.shape[2] / 2;
                let value = h.narrow(2, 0, hidden)?;
                let gate = h.narrow(2, hidden, hidden)?;
                value.mul(&gate.silu())
            })?;
            drop(h);
            let f = prof.time("vae_ff_out_linear", || block.ff_out.forward(&act))?;
            x = prof.time("vae_residual", || x.add(&f.mul(&block.scale2)?))?;
            observe(index, &x)?;
        }
        Ok(x)
    }

    /// The blocks with bf16 activations between the GEMMs, as the unfused
    /// path already feeds them (every bf16 linear rounds its input to bf16):
    /// each GEMM writes bf16, and one kernel takes it from there to the next
    /// GEMM's bf16 input (bias, residual with LayerScale, RMSNorm; per-head
    /// QK-norm, RoPE and the BHSD layout; SwiGLU). The residual stream stays
    /// f32. Same arithmetic as [`Self::vit_unfused`] up to reduction order.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn vit_fused(
        &self,
        x: CudaTensor,
        batch: usize,
        tokens: usize,
        (cos, sin): (&CudaTensor, &CudaTensor),
        observe: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
        mx: bool,
        prof: &mut Prof,
    ) -> Result<CudaTensor> {
        use crate::wan::ops;
        use crate::wan::quant::MxAct;
        let cfg = &self.cfg;
        let (dim, heads, head_dim) = (
            cfg.decoder_dim(),
            cfg.decoder_num_attention_heads,
            cfg.decoder_attention_head_dim,
        );
        let hidden = dim * cfg.decoder_ffn_mult;
        let rotary = cfg.decoder_rotary_dim();
        let eps = cfg.decoder_norm_eps as f32;
        let rows = batch * tokens;
        let vec = vec_enabled();
        let algo = gemm_algo();
        let mx_nth = mx_algo();
        let sdpa = sdpa_kernel();
        // Once per process and GEMM kind (the check runs both kinds).
        static SAID: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];
        if !SAID[usize::from(mx)].swap(true, Ordering::Relaxed) {
            crate::wan::log::info(format_args!(
                "h3 vae fused ViT: gemm {}, sdpa {}, vector kernels {}",
                if mx {
                    format!("mxfp8 (cuBLASLt algo #{mx_nth})")
                } else if self.nvfp4 {
                    "nvfp4 (published ModelOpt, cuBLASLt VEC16_UE4M3)".to_string()
                } else {
                    match algo {
                        None => "bf16 cublas".to_string(),
                        Some(n) => format!("bf16 cuBLASLt algo #{n}"),
                    }
                },
                sdpa.map_or_else(|| "auto".to_string(), |k| format!("{k:?}")),
                if vec { "on" } else { "off" }
            ));
        }
        fn dev(t: &CudaTensor) -> Result<crate::wan::tensor::DevRef<'_>> {
            t.dev()?
                .ok_or_else(|| msg("h3 vae: tensor not on the device"))
        }
        fn bias(l: &Linear) -> Result<crate::wan::tensor::DevRef<'_>> {
            dev(l
                .bias
                .as_ref()
                .ok_or_else(|| msg("h3 vae: linear without a bias"))?)
        }
        // Block linear `which` (0 qkv, 1 to_out, 2 ff_in, 3 ff_out) on `act`:
        // bf16 [rows, out] (or [rows_pad, out] from an MXFP8 GEMM, pad rows zero).
        let gemm = |block: &Block,
                    which: usize,
                    act: &Act|
         -> Result<cudarc::driver::CudaSlice<half::bf16>> {
            let l = [&block.qkv, &block.to_out, &block.ff_in, &block.ff_out][which];
            match act {
                Act::Mx(a) => block
                    .mx
                    .as_ref()
                    .ok_or_else(|| msg("h3 vae: no MXFP8 weights"))?[which]
                    .forward_mx_padded(a, mx_nth),
                Act::Bf16(x16) if l.is_nvfp4_published() => l
                    .gemm_nvfp4_bf16(x16, rows)?
                    .ok_or_else(|| msg("h3 vae: linear has no NVFP4 GEMM")),
                Act::Bf16(x16) => match algo {
                    None => l
                        .gemm_bf16(x16, rows)?
                        .ok_or_else(|| msg("h3 vae: linear has no bf16 GEMM")),
                    Some(nth) => {
                        let w = l
                            .weight_bf16_shared()
                            .ok_or_else(|| msg("h3 vae: linear has no bf16 GEMM"))?;
                        let ctx = crate::wan::device::global_device()
                            .ok_or_else(|| msg("h3 vae: no device"))?;
                        let mut out = unsafe { ctx.stream.alloc::<half::bf16>(rows * l.out_dim()) }
                            .map_err(|e| msg(e.to_string()))?;
                        crate::wan::quant::lt_gemm_bf16(
                            x16,
                            &w,
                            &mut out,
                            rows,
                            l.in_dim(),
                            l.out_dim(),
                            nth,
                        )?;
                        Ok(out)
                    }
                },
            }
        };
        // Residual step + RMSNorm into the next GEMM's operand.
        let res_norm = |x: &cudarc::driver::CudaSlice<f32>,
                        update: Option<(
            &cudarc::driver::CudaSlice<half::bf16>,
            &cudarc::driver::CudaSlice<f32>,
            &cudarc::driver::CudaSlice<f32>,
        )>,
                        w: &cudarc::driver::CudaSlice<f32>|
         -> Result<(Option<cudarc::driver::CudaSlice<f32>>, Act)> {
            if mx {
                let mut a = MxAct::alloc(rows, dim)?;
                let x1 = ops::h3v_residual_norm_mx_device(x, dim, update, w, eps, &mut a)?;
                Ok((x1, Act::Mx(a)))
            } else {
                let (x1, n) = ops::h3v_residual_norm_bf16_device(x, dim, update, Some(w), eps)?;
                Ok((
                    x1,
                    Act::Bf16(n.ok_or_else(|| msg("h3 vae: no normed rows"))?),
                ))
            }
        };
        let (cos, sin) = (dev(cos)?, dev(sin)?);

        let mut x = x;
        let mut n1 = prof
            .time("vae_residual_norm", || {
                res_norm(&*dev(&x)?, None, &*dev(&self.blocks[0].norm1)?)
            })?
            .1;
        for (index, block) in self.blocks.iter().enumerate() {
            let qkv = prof.time("vae_qkv_gemm", || gemm(block, 0, &n1))?;
            let [q, k, v] = prof.time("vae_qkv_heads", || {
                // The vector twin of this kernel measured slower (h3vae_bench:
                // 0.50 vs 0.39 ms at 7 tiles), so it stays scalar.
                ops::h3v_qkv_heads_bf16_device_with(
                    false,
                    &qkv,
                    &*bias(&block.qkv)?,
                    &cos,
                    &sin,
                    batch,
                    tokens,
                    heads,
                    head_dim,
                    rotary,
                    eps,
                )
            })?;
            drop(qkv);
            let bhsd = vec![batch, heads, tokens, head_dim];
            let [q, k, v] = [q, k, v].map(|t| CudaTensor::from_device_slice_bf16(t, bhsd.clone()));
            let a = prof.time("vae_sdpa", || {
                let (q, k, v) = (q?, k?, v?);
                match sdpa {
                    None => crate::wan::attn::device_mma_sdpa(&q, &k, &v, None, true)?,
                    Some(kernel) => {
                        crate::wan::attn::device_mma_sdpa_with(&q, &k, &v, None, true, kernel)?
                    }
                }
                .ok_or_else(|| msg("h3 vae: fused SDPA refused the shape"))
            })?;
            let a = prof.time("vae_merge_heads", || {
                let a16 = a
                    .device_slice_bf16()
                    .ok_or_else(|| msg("h3 vae: SDPA output is not bf16"))?;
                if mx {
                    let mut m = MxAct::alloc(rows, dim)?;
                    ops::h3v_merge_heads_mx_device(a16, batch, heads, tokens, head_dim, &mut m)?;
                    Ok(Act::Mx(m))
                } else {
                    ops::h3v_merge_heads_bf16_device_with(vec, a16, batch, heads, tokens, head_dim)
                        .map(Act::Bf16)
                }
            })?;
            let o = prof.time("vae_out_gemm", || gemm(block, 1, &a))?;
            drop(a);
            let (x1, n2) = prof.time("vae_residual_norm", || {
                res_norm(
                    &*dev(&x)?,
                    Some((&o, &*bias(&block.to_out)?, &*dev(&block.scale1)?)),
                    &*dev(&block.norm2)?,
                )
            })?;
            drop(o);
            let x1 = x1.ok_or_else(|| msg("h3 vae: no residual rows"))?;
            let h = prof.time("vae_ff_in_gemm", || gemm(block, 2, &n2))?;
            drop(n2);
            let act = prof.time("vae_swiglu", || {
                if mx {
                    let mut m = MxAct::alloc(rows, hidden)?;
                    ops::h3v_swiglu_mx_device(&h, &*bias(&block.ff_in)?, hidden, &mut m)?;
                    Ok(Act::Mx(m))
                } else {
                    ops::h3v_swiglu_bf16_device_with(vec, &h, &*bias(&block.ff_in)?, hidden, rows)
                        .map(Act::Bf16)
                }
            })?;
            drop(h);
            let f = prof.time("vae_ff_out_gemm", || gemm(block, 3, &act))?;
            drop(act);
            let next = self.blocks.get(index + 1);
            let (x2, next_n) = prof.time("vae_residual_norm", || match next {
                Some(nb) => res_norm(
                    &x1,
                    Some((&f, &*bias(&block.ff_out)?, &*dev(&block.scale2)?)),
                    &*dev(&nb.norm1)?,
                )
                .map(|(x2, n)| (x2, Some(n))),
                None => ops::h3v_residual_norm_bf16_device(
                    &x1,
                    dim,
                    Some((&f, &*bias(&block.ff_out)?, &*dev(&block.scale2)?)),
                    None,
                    eps,
                )
                .map(|(x2, _)| (x2, None)),
            })?;
            x = CudaTensor::from_device_slice(
                x2.ok_or_else(|| msg("h3 vae: no residual rows"))?,
                vec![batch, tokens, dim],
            )?;
            observe(index, &x)?;
            if let Some(n) = next_n {
                n1 = n;
            }
        }
        Ok(x)
    }

    /// `FASTVIDEO_H3_VAE_CHECK=1`, once per process on the first tile batch:
    /// decode it through both ViT paths, each op synchronized and timed, log
    /// the two breakdowns and the relative L2 between them, and keep the
    /// fused path only if they agree.
    fn check_paths(&self, tiles: &CudaTensor, grid: [usize; 3]) -> Result<Vec<CudaTensor>> {
        let batch = tiles.shape.first().copied().unwrap_or(0);
        let run = |fused: bool, mx: bool| -> Result<(Vec<CudaTensor>, f64, Prof)> {
            let mut prof = Prof::on();
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            let timer = std::time::Instant::now();
            let out =
                self.decode_tiles_with(tiles, grid, &mut |_, _| Ok(()), fused, mx, &mut prof)?;
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            Ok((out, timer.elapsed().as_secs_f64(), prof))
        };
        if self.nvfp4 {
            return self.check_nvfp4(tiles, grid);
        }
        if !self.fused_ready(batch, grid) {
            let (out, secs, prof) = run(false, false)?;
            crate::wan::log::info(format_args!(
                "h3 vae check: fused ViT unavailable; unfused batch of {batch} in {secs:.3}s [{}]",
                prof.summary()
            ));
            return Ok(out);
        }
        // Warm the fused path's allocations so neither timing pays first use.
        drop(run(true, false)?);
        let (reference, ref_secs, ref_prof) = run(false, false)?;
        let (fused, fused_secs, fused_prof) = run(true, false)?;
        let rel = |got: &[CudaTensor]| -> Result<f64> {
            let (mut num, mut den) = (0f64, 0f64);
            for (a, b) in got.iter().zip(&reference) {
                for (x, y) in a.host_cow()?.iter().zip(b.host_cow()?.iter()) {
                    num += f64::from(x - y).powi(2);
                    den += f64::from(*y).powi(2);
                }
            }
            Ok((num / den.max(f64::MIN_POSITIVE)).sqrt())
        };
        let rel_l2 = rel(&fused)?;
        let agree = rel_l2.is_finite() && rel_l2 < FUSED_CHECK_REL_L2;
        if agree && self.mx_ready() {
            drop(run(true, true)?);
            let (mx_out, mx_secs, mx_prof) = run(true, true)?;
            let mx_rel = rel(&mx_out)?;
            let fused_rel = {
                let (mut num, mut den) = (0f64, 0f64);
                for (a, b) in mx_out.iter().zip(&fused) {
                    for (x, y) in a.host_cow()?.iter().zip(b.host_cow()?.iter()) {
                        num += f64::from(x - y).powi(2);
                        den += f64::from(*y).powi(2);
                    }
                }
                (num / den.max(f64::MIN_POSITIVE)).sqrt()
            };
            let keep = mx_rel.is_finite() && mx_rel < MXFP8_CHECK_REL_L2;
            crate::wan::log::info(format_args!(
                "h3 vae check: batch {batch} grid {grid:?}: unfused {ref_secs:.3}s; fused bf16 {fused_secs:.3}s, rel_l2 {rel_l2:.3e}; fused mxfp8 {mx_secs:.3}s [{}], rel_l2 {mx_rel:.3e} vs unfused, {fused_rel:.3e} vs fused bf16 ({})",
                mx_prof.summary(),
                if keep { "mxfp8 kept" } else { "MXFP8 OFF THE MARK, disabled" }
            ));
            if keep {
                return Ok(mx_out);
            }
            MXFP8_BROKEN.store(true, Ordering::Relaxed);
        }
        crate::wan::log::info(format_args!(
            "h3 vae check: batch {batch} grid {grid:?}: unfused {ref_secs:.3}s [{}]; fused {fused_secs:.3}s [{}]; rel_l2 {rel_l2:.3e} ({})",
            ref_prof.summary(),
            fused_prof.summary(),
            if agree { "fused kept" } else { "FUSED DISAGREES, disabled" }
        ));
        if agree {
            Ok(fused)
        } else {
            FUSED_BROKEN.store(true, Ordering::Relaxed);
            Ok(reference)
        }
    }
    /// [`Self::check_paths`] for the NVFP4 linears: the fused NVFP4 forward
    /// (timed, per op) against the unfused NVFP4 one (the integration: same
    /// weights, f32 activations between ops) and, with [`NVFP4_REF_ENV`],
    /// against the bf16 decoder's unfused, fused bf16 and fused MXFP8 outputs
    /// on the same tiles. Logs every rel-L2; the NVFP4 output is returned
    /// either way (there is no bf16 copy to fall back to).
    fn check_nvfp4(&self, tiles: &CudaTensor, grid: [usize; 3]) -> Result<Vec<CudaTensor>> {
        let batch = tiles.shape.first().copied().unwrap_or(0);
        let run = |dec: &Self, fused: bool, mx: bool| -> Result<(Vec<CudaTensor>, f64, Prof)> {
            let mut prof = Prof::on();
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            let timer = std::time::Instant::now();
            let out = dec.decode_tiles_with(tiles, grid, &mut |_, _| Ok(()), fused, mx, &mut prof)?;
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            Ok((out, timer.elapsed().as_secs_f64(), prof))
        };
        let rel = |got: &[CudaTensor], want: &[CudaTensor]| -> Result<f64> {
            let (mut num, mut den) = (0f64, 0f64);
            for (a, b) in got.iter().zip(want) {
                for (x, y) in a.host_cow()?.iter().zip(b.host_cow()?.iter()) {
                    num += f64::from(x - y).powi(2);
                    den += f64::from(*y).powi(2);
                }
            }
            Ok((num / den.max(f64::MIN_POSITIVE)).sqrt())
        };
        let fused = self.fused_ready(batch, grid);
        // Warm the allocations so the timing pays no first use.
        drop(run(self, fused, false)?);
        let (nv, nv_secs, nv_prof) = run(self, fused, false)?;
        let (nv_unfused, unfused_secs, _) = run(self, false, false)?;
        let integ = rel(&nv, &nv_unfused)?;
        let mut line = format!(
            "h3 vae check (nvfp4): batch {batch} grid {grid:?}: {} nvfp4 {nv_secs:.3}s [{}]; unfused nvfp4 {unfused_secs:.3}s, rel_l2 {integ:.3e}",
            if fused { "fused" } else { "UNFUSED" },
            nv_prof.summary()
        );
        let mut worst = integ;
        if let Some(r) = self.reference.as_deref() {
            let (bf_unfused, s0, _) = run(r, false, false)?;
            let vs_unfused = rel(&nv, &bf_unfused)?;
            line += &format!("; vs bf16 unfused ({s0:.3}s) rel_l2 {vs_unfused:.3e}");
            worst = worst.max(vs_unfused);
            if r.fused_ready(batch, grid) {
                drop(run(r, true, false)?);
                let (bf_fused, s1, _) = run(r, true, false)?;
                line += &format!(
                    "; vs bf16 fused ({s1:.3}s) {:.3e} (bf16 fused vs unfused {:.3e})",
                    rel(&nv, &bf_fused)?,
                    rel(&bf_fused, &bf_unfused)?
                );
                if r.mx_ready() {
                    drop(run(r, true, true)?);
                    let (mx_out, s2, _) = run(r, true, true)?;
                    line += &format!(
                        "; vs mxfp8 fused ({s2:.3}s) {:.3e} (mxfp8 vs bf16 unfused {:.3e})",
                        rel(&nv, &mx_out)?,
                        rel(&mx_out, &bf_unfused)?
                    );
                }
            }
        }
        let ok = worst.is_finite() && worst < NVFP4_CHECK_REL_L2;
        if !ok {
            NVFP4_OFF.store(true, Ordering::Relaxed);
        }
        crate::wan::log::info(format_args!(
            "{line} ({})",
            if ok {
                format!("nvfp4 within the {NVFP4_CHECK_REL_L2} sanity bound")
            } else {
                format!("NVFP4 OFF THE MARK (>= {NVFP4_CHECK_REL_L2})")
            }
        ));
        Ok(nv)
    }

    /// Whether [`Self::check_nvfp4`] found the NVFP4 forward off the mark.
    pub fn nvfp4_off_the_mark() -> bool {
        NVFP4_OFF.load(Ordering::Relaxed)
    }

    /// Whether the block linears are the published NVFP4 ones.
    pub fn is_nvfp4(&self) -> bool {
        self.nvfp4
    }

    fn axis_tiles(&self, latent_len: usize) -> Result<AxisTiles> {
        let ratio = self.cfg.spatial_compression_ratio();
        let (starts, overlaps) = self.cfg.split_tiles(latent_len * ratio);
        if starts.iter().chain(&overlaps).any(|v| v % ratio != 0)
            || self.cfg.tile_sample_min_size % ratio != 0
        {
            return Err(msg(format!("h3 vae: tile plan {starts:?}/{overlaps:?} is not aligned to the {ratio}px latent grid")));
        }
        let len = if starts.len() == 1 {
            latent_len
        } else {
            self.cfg.tile_sample_min_size / ratio
        };
        Ok(AxisTiles {
            starts: starts.iter().map(|s| s / ratio).collect(),
            len,
            overlaps,
        })
    }

    /// One temporal clip: `z` is `[t, H, W, C]` denormalized, channel-last.
    /// Returns `[3, 4t, 16H, 16W]`, tiles decoded independently and cross-faded.
    fn decode_clip(&self, z: &CudaTensor) -> Result<CudaTensor> {
        let [t, h, w, c] = z.shape[..] else {
            return Err(msg(format!(
                "h3 vae clip must be [t, H, W, C], got {:?}",
                z.shape
            )));
        };
        let (ys, xs) = (self.axis_tiles(h)?, self.axis_tiles(w)?);
        let grid = [t, ys.len, xs.len];
        let mut pending: Vec<CudaTensor> = Vec::new();
        let mut decoded: Vec<CudaTensor> = Vec::with_capacity(ys.starts.len() * xs.starts.len());
        let total = ys.starts.len() * xs.starts.len();
        let batch = balanced_batch(total, self.tile_batch);
        for &y0 in &ys.starts {
            for &x0 in &xs.starts {
                pending.push(
                    z.narrow(1, y0, ys.len)?
                        .narrow(2, x0, xs.len)?
                        .reshape(vec![1, t * ys.len * xs.len, c])?,
                );
                if pending.len() == batch || decoded.len() + pending.len() == total {
                    let refs: Vec<&CudaTensor> = pending.iter().collect();
                    decoded.extend(self.decode_tiles(&CudaTensor::cat(&refs, 0)?, grid)?);
                    pending.clear();
                }
            }
        }
        // `_stitch_tiles`: vertical blend against the RAW tile above, then
        // horizontal against the RAW tile to the left, then crop own trailing overlaps.
        let cols = xs.starts.len();
        let mut rows = Vec::with_capacity(ys.starts.len());
        for i in 0..ys.starts.len() {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                let mut tile = decoded[i * cols + j].clone();
                if i > 0 {
                    tile = blend(&decoded[(i - 1) * cols + j], &tile, ys.overlaps[i - 1], 2)?;
                }
                if j > 0 {
                    tile = blend(&decoded[i * cols + j - 1], &tile, xs.overlaps[j - 1], 3)?;
                }
                if i + 1 < ys.starts.len() {
                    tile = tile.narrow(2, 0, tile.shape[2] - ys.overlaps[i])?;
                }
                if j + 1 < cols {
                    tile = tile.narrow(3, 0, tile.shape[3] - xs.overlaps[j])?;
                }
                row.push(tile);
            }
            let refs: Vec<&CudaTensor> = row.iter().collect();
            rows.push(CudaTensor::cat(&refs, 3)?);
        }
        let refs: Vec<&CudaTensor> = rows.iter().collect();
        CudaTensor::cat(&refs, 2)
    }

    /// DiT-space latents `[1, C, T, H, W]` to ImageNet-normalized RGB, streamed:
    /// `sink(frame_offset, frames)` receives `[3, f, 16H, 16W]` chunks in order
    /// as each temporal chunk finishes. Returns the number of frames emitted.
    /// This is the tensor the reference's `decode` returns, before the pixel
    /// de-normalization and clamp (see [`Self::to_display`]).
    ///
    /// The decoder always runs with f32 activations: its fused ViT
    /// ([`Self::fused_ready`]) keeps bf16 between GEMMs itself, with the
    /// reference's rounding points, and is the path measured against
    /// FastVideo. The DiT's bf16-activation default must not route the VAE
    /// onto the slower unfused path, so bf16 latents are widened here.
    pub fn decode_raw_streaming(
        &self,
        latents: &CudaTensor,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<usize> {
        crate::wan::tensor::with_bf16_act(false, || {
            let latents = latents.to_f32_act()?;
            self.decode_raw_streaming_f32(&latents, sink)
        })
    }

    fn decode_raw_streaming_f32(
        &self,
        latents: &CudaTensor,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<usize> {
        let [n, c, frames, _, _] = latents.shape[..] else {
            return Err(msg(format!(
                "h3 vae expects [1, C, T, H, W] latents, got {:?}",
                latents.shape
            )));
        };
        if n != 1 || c != self.cfg.latent_channels || frames == 0 {
            return Err(msg(format!(
                "h3 vae: latents {:?} for {} latent channels",
                latents.shape, self.cfg.latent_channels
            )));
        }
        let cfg = &self.cfg;
        // Channel-last once, so tiles and chunks are plain narrows and the
        // per-channel denormalization broadcasts over the last axis.
        let mut z = latents
            .reshape(latents.shape[1..].to_vec())?
            .permute(&[1, 2, 3, 0])?
            .mul(&self.std)?
            .add(&self.mean)?;
        let (pad_tokens, num_chunks, total_frames) = cfg.temporal_decode_plan(frames);
        if pad_tokens > 0 {
            // A length that is not `5n + 2` is completed by repeating the last latent frame.
            let last = z.narrow(0, frames - 1, 1)?;
            let mut parts: Vec<&CudaTensor> = vec![&z];
            parts.extend(std::iter::repeat_n(&last, pad_tokens));
            z = CudaTensor::cat(&parts, 0)?;
        }
        let (chunk, ratio, pre) = (
            cfg.tokens_chunk_size(),
            cfg.temporal_compression_ratio(),
            cfg.frame_pre_padding(),
        );
        let chunk_frames = chunk * ratio;
        let window = chunk + cfg.token_overlap();

        let mut emitted = 0usize;
        let mut emit = |piece: CudaTensor, emitted: &mut usize| -> Result<()> {
            // Frames decoded from the repeated padding latents are never shown.
            let keep = piece.shape[1].min(total_frames - *emitted);
            if keep > 0 {
                sink(*emitted, &piece.narrow(1, 0, keep)?)?;
                *emitted += keep;
            }
            Ok(())
        };
        let mut overlap: Option<CudaTensor> = None;
        for i in 0..num_chunks {
            let clip = self.decode_clip(&z.narrow(0, i * chunk, window)?)?;
            let clip_frames = clip.shape[1];
            let mut body = clip.narrow(1, pre, chunk_frames - pre)?;
            if let Some(prev) = &overlap {
                body = blend(prev, &body, cfg.frame_overlap(), 1)?;
            }
            emit(body, &mut emitted)?;
            // The second window of the clip, minus its own pre-padding frames.
            let tail_start = chunk_frames + pre;
            overlap = Some(clip.narrow(
                1,
                tail_start,
                clip_frames.min(2 * chunk_frames) - tail_start,
            )?);
            crate::wan::log::info(format_args!("h3 vae chunk {}/{num_chunks}", i + 1));
        }
        if let Some(tail) = overlap {
            emit(tail, &mut emitted)?;
        }
        if emitted != total_frames {
            return Err(msg(format!(
                "h3 vae emitted {emitted} frames, planned {total_frames}"
            )));
        }
        Ok(emitted)
    }

    /// Raw decoder output `[3, f, H, W]` to the pipeline's frame convention,
    /// `[f, 3, H, W]` in `[-1, 1]`: `rgb01 = clamp(raw * std + mean, 0, 1)`,
    /// then `2 rgb01 - 1`, folded into one affine and one clamp.
    pub fn to_display(raw: &CudaTensor) -> Result<CudaTensor> {
        if raw.rank() != 4 || raw.shape[0] != 3 {
            return Err(msg(format!(
                "h3 vae frames must be [3, f, H, W], got {:?}",
                raw.shape
            )));
        }
        let scale = CudaTensor::from_vec(
            H3_PIXEL_STD.iter().map(|&s| (2.0 * s) as f32).collect(),
            vec![3, 1, 1, 1],
        )?;
        let shift = CudaTensor::from_vec(
            H3_PIXEL_MEAN
                .iter()
                .map(|&m| (2.0 * m - 1.0) as f32)
                .collect(),
            vec![3, 1, 1, 1],
        )?;
        raw.mul(&scale)?
            .add(&shift)?
            .clamp(-1.0, 1.0)
            .permute(&[1, 0, 2, 3])
    }

    /// [`Self::decode_raw_streaming`] with each chunk converted by
    /// [`Self::to_display`], the form `VideoWriter` / `frames_to_rgb8` consume.
    pub fn decode_streaming(
        &self,
        latents: &CudaTensor,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<usize> {
        self.decode_raw_streaming(latents, &mut |offset, raw| {
            sink(offset, &Self::to_display(raw)?)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> H3VideoVaeConfig {
        let mut cfg = H3VideoVaeConfig::fasth3_8step();
        cfg.latent_channels = 5;
        cfg.spatial_downsample_factors = [2, 2, 1, 1, 1, 1]; // 4 px per latent
        cfg.decoder_num_layers = 2;
        cfg.decoder_num_attention_heads = 2;
        cfg.decoder_attention_head_dim = 8; // 6 rotary channels
        cfg.decoder_ffn_mult = 2;
        cfg.tile_sample_min_size = 8; // 2 latents
        cfg.tile_sample_min_overlap = 4; // 1 latent
        cfg
    }

    fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key
                .bytes()
                .fold(5u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let n: usize = shape.iter().product();
            (0..n)
                .map(|i| {
                    let u = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32
                        / (1u32 << 24) as f32;
                    if key.contains("norm") || key.contains("scale") {
                        0.5 + u
                    } else {
                        u - 0.5
                    }
                })
                .collect()
        })
    }

    fn get(map: &WeightMap, key: &str, shape: &[usize]) -> Vec<f32> {
        cuda_tensor_shaped(map, key, shape)
            .unwrap()
            .host_cow()
            .unwrap()
            .into_owned()
    }

    fn lin(map: &WeightMap, prefix: &str, x: &[f32], i: usize, o: usize) -> Vec<f32> {
        let (w, b) = (
            get(map, &format!("{prefix}.weight"), &[o, i]),
            get(map, &format!("{prefix}.bias"), &[o]),
        );
        (0..o)
            .map(|r| b[r] + (0..i).map(|c| x[c] * w[r * i + c]).sum::<f32>())
            .collect()
    }

    fn rms(v: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let ms = v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32;
        v.iter()
            .enumerate()
            .map(|(i, a)| a / (ms + eps).sqrt() * w.map_or(1.0, |w| w[i]))
            .collect()
    }

    /// The ViT on one tile, token by token, returning `[3][4t][ps h][ps w]` flattened.
    fn reference_tile(
        cfg: &H3VideoVaeConfig,
        map: &WeightMap,
        z: &[f32],
        grid: [usize; 3],
    ) -> Vec<f32> {
        let (dim, heads, hd, lc) = (
            cfg.decoder_dim(),
            cfg.decoder_num_attention_heads,
            cfg.decoder_attention_head_dim,
            cfg.latent_channels,
        );
        let eps = 1e-5f32;
        let patches = grid[0] * grid[1] * grid[2];
        let pq_w = get(map, "post_quant_conv.weight", &[lc, lc, 1, 1, 1]);
        let pq_b = get(map, "post_quant_conv.bias", &[lc]);
        let mut x: Vec<Vec<f32>> = (0..patches)
            .map(|p| {
                let v = &z[p * lc..(p + 1) * lc];
                let q: Vec<f32> = (0..lc)
                    .map(|o| pq_b[o] + (0..lc).map(|c| v[c] * pq_w[o * lc + c]).sum::<f32>())
                    .collect();
                lin(map, "decoder.proj_in", &q, lc, dim)
            })
            .collect();
        let reg = get(map, "decoder.register_tokens", &[1, 4, dim]);
        for r in 0..4 {
            x.push(reg[r * dim..(r + 1) * dim].to_vec());
        }
        x.push(vec![0.0; dim]);
        let tokens = x.len();
        // Angles: 3 axes x 1 frequency (rotary 6 -> theta^0 = 1), so angle = 2 pi pos.
        let rotary = cfg.decoder_rotary_dim();
        assert_eq!(rotary, 6);
        let pos = |i: usize, size: usize| 2.0 * ((i as f32 + 0.5) / size as f32) - 1.0;
        let angles: Vec<[f32; 3]> = (0..tokens)
            .map(|p| {
                if p >= patches {
                    return [0.0; 3];
                }
                let (ti, rem) = (p / (grid[1] * grid[2]), p % (grid[1] * grid[2]));
                let tau = 2.0 * std::f32::consts::PI;
                [
                    tau * pos(ti, grid[0]),
                    tau * pos(rem / grid[2], grid[1]),
                    tau * pos(rem % grid[2], grid[2]),
                ]
            })
            .collect();
        let rope = |v: &[f32], p: usize| -> Vec<f32> {
            let mut out = v.to_vec();
            for a in 0..3 {
                let (c, s) = (angles[p][a].cos(), angles[p][a].sin());
                out[a] = v[a] * c - v[a + 3] * s;
                out[a + 3] = v[a + 3] * c + v[a] * s;
            }
            out
        };
        for b in 0..cfg.decoder_num_layers {
            let p = format!("decoder.transformer_blocks.{b}");
            let (n1, n2) = (
                get(map, &format!("{p}.norm1.weight"), &[dim]),
                get(map, &format!("{p}.norm2.weight"), &[dim]),
            );
            let (s1, s2) = (
                get(map, &format!("{p}.scale1"), &[dim]),
                get(map, &format!("{p}.scale2"), &[dim]),
            );
            let normed: Vec<Vec<f32>> = x.iter().map(|v| rms(v, Some(&n1), eps)).collect();
            let qkv = |name: &str, norm: bool| -> Vec<Vec<Vec<f32>>> {
                normed
                    .iter()
                    .enumerate()
                    .map(|(pi, v)| {
                        let full = lin(map, &format!("{p}.attn.{name}"), v, dim, dim);
                        (0..heads)
                            .map(|h| {
                                let head = &full[h * hd..(h + 1) * hd];
                                if norm {
                                    rope(&rms(head, None, eps), pi)
                                } else {
                                    head.to_vec()
                                }
                            })
                            .collect()
                    })
                    .collect()
            };
            let (q, k, v) = (qkv("to_q", true), qkv("to_k", true), qkv("to_v", false));
            let mut next = Vec::with_capacity(tokens);
            for i in 0..tokens {
                let mut attn = vec![0f32; dim];
                for h in 0..heads {
                    let scores: Vec<f32> = (0..tokens)
                        .map(|j| {
                            q[i][h]
                                .iter()
                                .zip(&k[j][h])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                / (hd as f32).sqrt()
                        })
                        .collect();
                    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let zsum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                    for (j, s) in scores.iter().enumerate() {
                        let w = (s - mx).exp() / zsum;
                        for c in 0..hd {
                            attn[h * hd + c] += w * v[j][h][c];
                        }
                    }
                }
                let o = lin(map, &format!("{p}.attn.to_out.0"), &attn, dim, dim);
                let after: Vec<f32> = (0..dim).map(|c| x[i][c] + o[c] * s1[c]).collect();
                let m = rms(&after, Some(&n2), eps);
                let hidden = dim * cfg.decoder_ffn_mult;
                let f = lin(map, &format!("{p}.ff.net.0.proj"), &m, dim, 2 * hidden);
                let act: Vec<f32> = (0..hidden)
                    .map(|c| f[c] * (f[hidden + c] / (1.0 + (-f[hidden + c]).exp())))
                    .collect();
                let o = lin(map, &format!("{p}.ff.net.2"), &act, hidden, dim);
                next.push(
                    (0..dim)
                        .map(|c| after[c] + o[c] * s2[c])
                        .collect::<Vec<f32>>(),
                );
            }
            x = next;
        }
        let (nw, nb) = (
            get(map, "decoder.norm_out.weight", &[dim]),
            get(map, "decoder.norm_out.bias", &[dim]),
        );
        let (pt, ps) = (4usize, cfg.spatial_compression_ratio());
        let (fh, fw) = (grid[1] * ps, grid[2] * ps);
        let mut out = vec![0f32; 3 * grid[0] * pt * fh * fw];
        for p in 0..patches {
            let mean = x[p].iter().sum::<f32>() / dim as f32;
            let var = x[p].iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / dim as f32;
            let ln: Vec<f32> = (0..dim)
                .map(|c| (x[p][c] - mean) / (var + eps).sqrt() * nw[c] + nb[c])
                .collect();
            let patch = lin(map, "decoder.proj_out", &ln, dim, cfg.decoder_patch_dim());
            let (ti, rem) = (p / (grid[1] * grid[2]), p % (grid[1] * grid[2]));
            let (hi, wi) = (rem / grid[2], rem % grid[2]);
            for c in 0..3 {
                for dt in 0..pt {
                    for dy in 0..ps {
                        for dx in 0..ps {
                            let src = ((c * pt + dt) * ps + dy) * ps + dx;
                            let dst = ((c * grid[0] * pt + ti * pt + dt) * fh + hi * ps + dy) * fw
                                + wi * ps
                                + dx;
                            out[dst] = patch[src];
                        }
                    }
                }
            }
        }
        out
    }

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k).sin() * 0.8).collect()
    }

    #[test]
    fn one_tile_matches_a_token_loop_reference() {
        let (cfg, map) = (tiny_cfg(), weights());
        let dec = H3VideoDecoder::load(cfg.clone(), &map).unwrap();
        let grid = [3usize, 2, 2];
        let z = seeded(12 * cfg.latent_channels, 0.37);
        let tiles = CudaTensor::from_vec(z.clone(), vec![1, 12, cfg.latent_channels]).unwrap();
        let got = dec.decode_tiles(&tiles, grid).unwrap().remove(0);
        assert_eq!(got.shape, vec![3, 12, 8, 8]);
        let want = reference_tile(&cfg, &map, &z, grid);
        let got = got.host_cow().unwrap();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.1);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 2e-4 * scale.max(1.0),
                "value {i}: {g} vs {w}"
            );
        }
    }

    /// The 5 s benchmark geometries tile exactly as FastVideo's
    /// `AutoencoderKLMiniMaxH3._split_tiles` / `_temporal_decode_plan` do:
    /// 7 chunks of 4 x 7 tiles at 1344x768, of 3 x 4 tiles at 832x480.
    #[test]
    fn benchmark_tile_plans_match_the_reference() {
        let cfg = H3VideoVaeConfig::fasth3_8step();
        assert_eq!(
            cfg.split_tiles(1344),
            (
                vec![0, 176, 352, 528, 704, 896, 1088],
                vec![80, 80, 80, 80, 64, 64]
            )
        );
        assert_eq!(
            cfg.split_tiles(768),
            (vec![0, 160, 336, 512], vec![96, 80, 80])
        );
        assert_eq!(
            cfg.split_tiles(832),
            (vec![0, 192, 384, 576], vec![64, 64, 64])
        );
        assert_eq!(cfg.split_tiles(480), (vec![0, 112, 224], vec![144, 144]));
        assert_eq!(cfg.temporal_decode_plan(37), (0, 7, 124));
        assert_eq!((balanced_batch(28, 8), balanced_batch(12, 8)), (7, 6));
        assert_eq!((balanced_batch(3, 8), balanced_batch(9, 1)), (3, 1));
    }

    /// MXFP8 decoder linears are the default on sm_100+ for the measured
    /// decoders, 36 and 26 blocks (owner decision 2026-10-14); `=0` is bf16
    /// everywhere, `=1` asks for MXFP8 and fails on a GPU without
    /// block-scaled FP8.
    #[test]
    fn vae_mxfp8_defaults_on_for_the_measured_decoders_on_sm100_plus() {
        let r = |e: Option<&str>, sm: Option<i32>, layers: usize| resolve_mxfp8(e, sm, layers);
        // Unset.
        assert_eq!(
            r(None, Some(12), 36).unwrap(),
            (true, None),
            "RTX PRO 6000 / RTX 5090"
        );
        assert_eq!(r(None, Some(10), 36).unwrap(), (true, None), "B200");
        assert_eq!(
            r(Some(""), Some(10), 36).unwrap(),
            (true, None),
            "empty = unset"
        );
        let (on, why) = r(None, Some(9), 36).unwrap();
        assert!(!on && why.unwrap().contains("sm_100+"), "H100: bf16, said");
        assert_eq!(
            r(None, Some(12), 26).unwrap(),
            (true, None),
            "FastH3 Trim's 26-block decoder"
        );
        assert_eq!(r(None, Some(10), 26).unwrap(), (true, None), "Trim on B200");
        let (on, why) = r(None, Some(9), 26).unwrap();
        assert!(
            !on && why.unwrap().contains("sm_100+"),
            "Trim on H100: bf16, said"
        );
        let (on, why) = r(None, Some(12), 30).unwrap();
        assert!(
            !on && why.unwrap().contains("30"),
            "an unmeasured depth: bf16, said"
        );
        assert_eq!(r(None, None, 36).unwrap(), (false, None), "CPU run");
        // =0: bf16 everywhere, nothing to say.
        for v in ["0", "false", "OFF", " 0 "] {
            assert_eq!(r(Some(v), Some(12), 36).unwrap(), (false, None), "{v}");
            assert_eq!(r(Some(v), Some(9), 36).unwrap(), (false, None), "{v}");
        }
        // =1: MXFP8 on any decoder; an error below sm_100.
        for v in ["1", "true", "on"] {
            assert_eq!(r(Some(v), Some(12), 36).unwrap(), (true, None));
            assert_eq!(r(Some(v), Some(12), 26).unwrap(), (true, None));
            let e = r(Some(v), Some(9), 36).unwrap_err().to_string();
            assert!(e.contains("sm_100+") && e.contains("=0"), "{e}");
        }
        assert!(r(Some("yes please"), Some(12), 36).is_err());
    }

    /// `FASTVIDEO_H3_VAE_NVFP4` (owner decision 2026-10-14): a `vae/` that
    /// carries the published linears decodes with them by default (unset or
    /// `=1`), on NVFP4 tensor cores only; `=0` takes h3-base's `vae/` (the
    /// previous decoder) and is an error without one; `=1` on a bf16 tree is
    /// an error; a bf16 tree is unchanged otherwise. No silent fallback.
    #[test]
    fn vae_nvfp4_is_the_default_of_a_tree_that_carries_it() {
        use VaeLinearsChoice::*;
        let r = resolve_nvfp4;
        // A bf16 vae/ (h3-base, the Trim): unchanged whatever is at hand.
        for base in [false, true] {
            assert_eq!(r(None, false, base, Some(12)).unwrap(), Official, "default: bf16 tree");
            assert_eq!(r(Some(false), false, base, Some(12)).unwrap(), Official, "=0 on a bf16 tree");
            let e = r(Some(true), false, base, Some(12)).unwrap_err().to_string();
            assert!(e.contains(NVFP4_PROBE), "{e}");
        }
        // The NVFP4 tier's vae/: NVFP4 unset or =1, on sm_100 / sm_120.
        for setting in [None, Some(true)] {
            for base in [false, true] {
                assert_eq!(r(setting, true, base, Some(12)).unwrap(), Nvfp4, "RTX PRO 6000");
                assert_eq!(r(setting, true, base, Some(10)).unwrap(), Nvfp4, "B200");
                let e = r(setting, true, base, Some(9)).unwrap_err().to_string();
                assert!(e.contains("sm_9") && e.contains(&format!("{NVFP4_ENV}=0")), "{e}");
                let e = r(setting, true, base, None).unwrap_err().to_string();
                assert!(e.contains(&format!("{NVFP4_ENV}=0")), "no device: {e}");
            }
        }
        // =0: h3-base's vae/ (bf16 + MXFP8), whatever the device; without it, an error naming the fix.
        assert_eq!(r(Some(false), true, true, Some(12)).unwrap(), BaseOfficial);
        assert_eq!(r(Some(false), true, true, Some(9)).unwrap(), BaseOfficial);
        let e = r(Some(false), true, false, Some(12)).unwrap_err().to_string();
        assert!(e.contains("h3-base") && e.contains("--text-weights"), "{e}");
    }

    /// `FASTVIDEO_H3_VAE_AB`'s bf16 re-decode suspends the MXFP8 GEMMs only
    /// inside the closure, and lifts the suspension even if it panics.
    #[test]
    fn bf16_linears_are_scoped_to_the_closure() {
        assert!(!MXFP8_SUSPENDED.load(Ordering::Relaxed));
        assert!(with_bf16_linears(|| MXFP8_SUSPENDED.load(Ordering::Relaxed)));
        assert!(!MXFP8_SUSPENDED.load(Ordering::Relaxed));
        let r = std::panic::catch_unwind(|| with_bf16_linears(|| panic!("decode failed")));
        assert!(r.is_err() && !MXFP8_SUSPENDED.load(Ordering::Relaxed));
    }

    #[test]
    fn rotary_tables_put_the_register_tokens_at_angle_zero() {
        let (cos, sin) = rope_tables([2, 1, 3], 5, 48, 100.0);
        let tokens = 2 * 3 + 5;
        assert_eq!(cos.len(), tokens * 48);
        assert!(cos[6 * 48..].iter().all(|&c| c == 1.0) && sin[6 * 48..].iter().all(|&s| s == 0.0));
        // Token (t=1, h=0, w=2): positions (0.5, 0, 2/3); channel 8 is h's first frequency, 17 is w's second.
        let row = &cos[5 * 48..6 * 48];
        let tau = 2.0 * std::f32::consts::PI;
        assert!((row[0] - (tau * 0.5).cos()).abs() < 1e-6);
        assert_eq!(row[8], 1.0);
        let w2 = 2.0f32 * (2.5 / 3.0) - 1.0;
        assert!((row[17] - (tau * w2 * 100f32.powf(-0.125)).cos()).abs() < 1e-5);
        assert_eq!(row[17], row[17 + 24], "cat(A, A)");
    }

    /// Tiling and temporal chunking as plain loops over per-tile decodes (the
    /// tile itself is judged by the test above).
    #[test]
    fn tiles_and_chunks_are_cross_faded_as_the_reference_does() {
        let (cfg, map) = (tiny_cfg(), weights());
        let dec = H3VideoDecoder::load(cfg.clone(), &map)
            .unwrap()
            .with_tile_batch(3);
        let (c, t, h, w) = (cfg.latent_channels, 12usize, 3usize, 3usize);
        let lat = seeded(c * t * h * w, 0.11);
        let mut chunks: Vec<(usize, Vec<usize>, Vec<f32>)> = Vec::new();
        let frames = dec
            .decode_raw_streaming(
                &CudaTensor::from_vec(lat.clone(), vec![1, c, t, h, w]).unwrap(),
                &mut |off, fr| {
                    chunks.push((off, fr.shape.clone(), fr.host_cow()?.into_owned()));
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(frames, 39, "12 latents = 2 chunks = 2 * 17 + 5 frames");
        assert_eq!(
            chunks
                .iter()
                .map(|(o, s, _)| (*o, s[1]))
                .collect::<Vec<_>>(),
            vec![(0, 17), (17, 17), (34, 5)]
        );
        let (fh, fw) = (12usize, 12usize);
        let mut got = vec![0f32; 3 * 39 * fh * fw];
        for (off, shape, data) in &chunks {
            for ch in 0..3 {
                for f in 0..shape[1] {
                    let src = (ch * shape[1] + f) * fh * fw;
                    let dst = (ch * 39 + off + f) * fh * fw;
                    got[dst..dst + fh * fw].copy_from_slice(&data[src..src + fh * fw]);
                }
            }
        }

        // Reference. Tiles: starts (0, 1) latents, 2 latents wide, overlap 1 latent = 4 px.
        let z = |ch: usize, ti: usize, y: usize, x: usize| {
            lat[((ch * t + ti) * h + y) * w + x] * cfg.latents_std[ch] as f32
                + cfg.latents_mean[ch] as f32
        };
        let tile = |t0: usize, y0: usize, x0: usize| -> Vec<f32> {
            let mut rows = Vec::new();
            for ti in 0..7 {
                for y in 0..2 {
                    for x in 0..2 {
                        rows.extend((0..c).map(|ch| z(ch, t0 + ti, y0 + y, x0 + x)));
                    }
                }
            }
            let tiles = CudaTensor::from_vec(rows, vec![1, 28, c]).unwrap();
            dec.decode_tiles(&tiles, [7, 2, 2])
                .unwrap()
                .remove(0)
                .host_cow()
                .unwrap()
                .into_owned() // [3, 28, 8, 8]
        };
        let at =
            |v: &[f32], ch: usize, f: usize, y: usize, x: usize| v[((ch * 28 + f) * 8 + y) * 8 + x];
        let clip = |t0: usize| -> Vec<f32> {
            let raw = [
                [tile(t0, 0, 0), tile(t0, 0, 1)],
                [tile(t0, 1, 0), tile(t0, 1, 1)],
            ];
            let mut out = vec![0f32; 3 * 28 * fh * fw];
            for ch in 0..3 {
                for f in 0..28 {
                    for y in 0..fh {
                        for x in 0..fw {
                            // Which tile owns this pixel after cropping, and its local coordinates.
                            let (i, ly) = if y < 4 { (0, y) } else { (1, y - 4) };
                            let (j, lx) = if x < 4 { (0, x) } else { (1, x - 4) };
                            // Vertical blend of tile (i, j) with the raw tile above.
                            let vert = |jj: usize, lx: usize| -> f32 {
                                let cur = at(&raw[i][jj], ch, f, ly, lx);
                                if i > 0 && ly < 4 {
                                    let wgt = ly as f32 / 4.0;
                                    at(&raw[i - 1][jj], ch, f, 4 + ly, lx) * (1.0 - wgt) + cur * wgt
                                } else {
                                    cur
                                }
                            };
                            let mut v = vert(j, lx);
                            if j > 0 && lx < 4 {
                                let wgt = lx as f32 / 4.0;
                                // The left neighbour enters RAW: no vertical blend of its own.
                                v = at(&raw[i][j - 1], ch, f, ly, 4 + lx) * (1.0 - wgt) + v * wgt;
                            }
                            out[((ch * 28 + f) * fh + y) * fw + x] = v;
                        }
                    }
                }
            }
            out
        };
        let (clip0, clip1) = (clip(0), clip(5));
        let px = |v: &[f32], ch: usize, f: usize, p: usize| v[(ch * 28 + f) * fh * fw + p];
        for ch in 0..3 {
            for f in 0..39 {
                for p in 0..fh * fw {
                    let want = match f {
                        0..=16 => px(&clip0, ch, 3 + f, p),
                        17..=21 => {
                            let wgt = (f - 17) as f32 / 5.0;
                            px(&clip0, ch, 23 + (f - 17), p) * (1.0 - wgt)
                                + px(&clip1, ch, 3 + (f - 17), p) * wgt
                        }
                        22..=33 => px(&clip1, ch, 3 + (f - 17), p),
                        _ => px(&clip1, ch, 23 + (f - 34), p),
                    };
                    let g = got[(ch * 39 + f) * fh * fw + p];
                    assert!(
                        (g - want).abs() < 1e-4 * want.abs().max(1.0),
                        "ch {ch} frame {f} px {p}: {g} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn display_frames_are_frame_major_and_in_the_pipeline_range() {
        let raw = CudaTensor::from_vec(vec![0.0, 100.0, -100.0, 1.0, 0.5, -0.5], vec![3, 2, 1, 1])
            .unwrap();
        let out = H3VideoDecoder::to_display(&raw).unwrap();
        assert_eq!(out.shape, vec![2, 3, 1, 1]);
        let v = out.host_cow().unwrap();
        // frame 0: channels (0.0, -100, 0.5); frame 1: (100, 1.0, -0.5)
        assert!((v[0] - (2.0 * 0.485 - 1.0)).abs() < 1e-6);
        assert_eq!((v[1], v[3]), (-1.0, 1.0));
        assert!((v[4] - (2.0 * (0.224 + 0.456) - 1.0)).abs() < 1e-6);
    }
}
