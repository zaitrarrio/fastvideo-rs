//! Reference FP8 linear recipes and their host twins.
//!
//! Two recipes, each exactly as its reference implements it:
//!
//! * **W8A8** — FastVideo `layers/quantization/fp8_config.py`
//!   (`FP8QuantizeMethod("tensor")`), the route Sol-H3-Spark stage 1 installs
//!   after merging its LoRA in bf16 (`Sol-H3-Spark/runtime/stage1.py:125-147`,
//!   `fp8_route = native_FastVideo_tensorwise_W8A8_after_original_BF16_LoRA_merge`).
//!   One f32 scale per weight tensor and one per activation tensor, both
//!   `max(amax / 448, 1 / (448 * 512))`; codes are
//!   `e4m3(clamp(bf16(x / bf16(scale)), ±448))` (the division runs in bf16 and
//!   the scale is rounded to bf16 for it, while `_scaled_mm` dequantizes with
//!   the f32 scale). Output bf16, f32 accumulation. The linears are the ones
//!   `stage1_ops/lookup.py::quantized_linear_name` tags: `to_q`, `to_k`,
//!   `to_v`, `to_out`, `ff.fc_in`, `ff.fc_out` of all 50 blocks and both text
//!   refiner blocks — 312 — each with its own tensor scale (so the fused QKV
//!   stack here keeps three scales, one GEMM per section).
//! * **MXFP8** — Sol-H3 `h3_runtime/mxfp8.py` + `compute_quant.py`: E4M3 values
//!   with one E8M0 scale per 32 values along K (the smallest power of two that
//!   keeps the block max within 448), scales in cuBLASLt's `VEC32_UE8M0`
//!   (torch `SWIZZLE_32_4_4`) layout, for weights and activations alike. Only
//!   transformer blocks 2..=46 (`FIRST/LAST_QUANTIZED_BLOCK`), and in them the
//!   fused QKV, `to_out`, `ff` up and down projections; AdaLN, refiners and the
//!   boundary blocks stay bf16. Activations are produced already quantized by
//!   the RMSNorm+modulate, residual+gate+RMSNorm+modulate and SwiGLU kernels
//!   (`fusion_install.py`); `to_out` quantizes its bf16 input with one kernel.
//!
//! The previous per-tensor E4M3 paths (`FASTVIDEO_FP8` with f32 activations
//! and an unclamped scale, and the H3-only `FASTVIDEO_H3_FFN_FP8`) matched no
//! reference and measured 17-20 dB against bf16; `FASTVIDEO_FP8` now runs this
//! W8A8 recipe and `FASTVIDEO_H3_FFN_FP8` is retired in favour of
//! `FASTVIDEO_H3_QUANT=w8a8|mxfp8|off`.

use super::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

pub const E4M3_MAX: f32 = 448.0;
/// `fp8_config.FP8_MIN_SCALE` (`1 / (448 * 512)`, a Python double cast to f32).
pub const W8A8_MIN_SCALE: f32 = (1.0f64 / 229_376.0) as f32;
/// Values per MX scale.
pub const MX_BLOCK: usize = 32;
/// Sol-H3 `compute_quant.py`: MXFP8 covers blocks `[2, 46]` inclusive.
pub const MXFP8_FIRST_BLOCK: usize = 2;
pub const MXFP8_LAST_BLOCK: usize = 46;
pub const ENV: &str = "FASTVIDEO_H3_QUANT";

// ---------------------------------------------------------------------------
// Scalar references (bit-exact twins of the kernels in kernels.cu)
// ---------------------------------------------------------------------------

/// f32 → bf16 → f32, round to nearest even (torch `.to(torch.bfloat16)`).
pub fn bf16_round(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

/// f32 → E4M3 as `cvt.rn.satfinite` / torch clamp-then-cast: RNE, finite
/// overflow saturates to ±448, NaN → 0x7F. Twin of `fv_e4m3_sat`.
pub fn e4m3_satfinite(x: f32) -> u8 {
    let u = x.to_bits();
    let sign = ((u >> 24) & 0x80) as u8;
    let a = u & 0x7FFF_FFFF;
    if a > 0x7F80_0000 {
        return sign | 0x7F;
    }
    if a >= 0x43E0_0000 {
        return sign | 0x7E;
    }
    if a < 0x3C80_0000 {
        let m = (f32::from_bits(a) * 512.0).round_ties_even() as u32;
        return sign | m as u8;
    }
    let mut e = (a >> 23) as i32 - 127;
    let mut m = (a >> 20) & 7;
    let rem = a & 0xF_FFFF;
    if rem > 0x8_0000 || (rem == 0x8_0000 && (m & 1) == 1) {
        m += 1;
        if m == 8 {
            m = 0;
            e += 1;
        }
    }
    sign | (((e + 7) as u32) << 3) as u8 | m as u8
}

/// E4M3 (fn variant: no infinities, 0x7F/0xFF NaN) → f32.
pub fn e4m3_decode(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    if e == 15 && (b & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 {
        return sign * m / 512.0;
    }
    sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
}

/// `mxfp8.py::_mx_e8m0_from_amax`: (biased E8M0 byte, 2^-exponent).
pub fn mx_e8m0(amax: f32) -> (u8, f32) {
    let bits = amax.to_bits() as i32;
    let e0 = ((bits >> 23) & 0xFF) - 135;
    let threshold = f32::from_bits((((e0 + 135) << 23) | 0x60_0000) as u32);
    let mut e = e0 + i32::from(amax > threshold);
    e = e.max(-127);
    let inv = f32::from_bits(((127 - e) << 23) as u32);
    ((e + 127) as u8, inv)
}

/// Scale position of `(row, group)` in the `VEC32_UE8M0` swizzle
/// (`mxfp8.py::_mx_scale_offsets`).
pub fn mx_scale_offset(row: usize, group: usize, column_blocks: usize) -> usize {
    let tile = (row / 128) * column_blocks + group / 4;
    tile * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4
}

/// Scale bytes for a `[rows, k]` operand: rows padded to 128, groups to 4.
pub fn mx_scale_len(rows: usize, k: usize) -> usize {
    rows.div_ceil(128) * 128 * (k / MX_BLOCK).div_ceil(4) * 4
}

/// Host `_mxfp8_quant_kernel`: codes `[rows, k]` and swizzled scales.
pub fn mxfp8_quantize(x: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(
        k.is_multiple_of(MX_BLOCK) && x.len() == rows * k,
        "mxfp8 needs K % 32 == 0"
    );
    let mut q = vec![0u8; rows * k];
    let mut s = vec![0u8; mx_scale_len(rows, k)];
    let cb = (k / MX_BLOCK).div_ceil(4);
    for r in 0..rows {
        for g in 0..k / MX_BLOCK {
            let blk = &x[r * k + g * MX_BLOCK..][..MX_BLOCK];
            let amax = blk.iter().fold(0.0f32, |a, &v| {
                let v = v.abs();
                if v > a || v.is_nan() {
                    v
                } else {
                    a
                }
            });
            let (byte, inv) = mx_e8m0(amax);
            s[mx_scale_offset(r, g, cb)] = byte;
            for (j, &v) in blk.iter().enumerate() {
                q[r * k + g * MX_BLOCK + j] = e4m3_satfinite(v * inv);
            }
        }
    }
    (q, s)
}

/// `mxfp8_dequantize_swizzled` (f32).
pub fn mxfp8_dequantize(q: &[u8], s: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let cb = (k / MX_BLOCK).div_ceil(4);
    (0..rows * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let e = i32::from(s[mx_scale_offset(r, c / MX_BLOCK, cb)]) - 127;
            e4m3_decode(q[i]) * 2f32.powi(e)
        })
        .collect()
}

/// max |x|; a NaN wins (the kernel propagates it too).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn amax_abs(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |a, &v| {
        let v = v.abs();
        if !(v <= a) {
            v
        } else {
            a
        }
    })
}

/// `_quantize_tensorwise`'s f32 scale.
pub fn w8a8_scale(amax: f32) -> f32 {
    (amax / E4M3_MAX).max(W8A8_MIN_SCALE)
}

/// `_quantize_tensorwise` codes for values `x` (read as stored — bf16 values
/// in the reference) under the f32 `scale`.
pub fn w8a8_quantize(x: &[f32], scale: f32) -> Vec<u8> {
    let s16 = bf16_round(scale);
    x.iter()
        .map(|&v| e4m3_satfinite(bf16_round(v / s16).clamp(-E4M3_MAX, E4M3_MAX)))
        .collect()
}

pub fn w8a8_dequantize(q: &[u8], scale: f32) -> Vec<f32> {
    q.iter().map(|&b| e4m3_decode(b) * scale).collect()
}

// ---------------------------------------------------------------------------
// bf16 fused-op references (the rounding points of Sol-H3 fusions.py)
// ---------------------------------------------------------------------------

fn rms_factor(x: &[f32], eps: f32) -> f32 {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    1.0 / (ss / x.len() as f32 + eps).sqrt()
}

/// Threads per row of the fused H3 norm kernels (`cfg_rows`).
pub const NORM_THREADS: usize = 256;

/// The fused H3 norm kernels' normalizer, bit for bit: lane `t` accumulates
/// `x[t], x[t + 256], ...` with FMA, the 256 partials reduce by the kernel's
/// halving tree, then `rsqrt(sum / n + eps)` correctly rounded
/// (`__frsqrt_rn`). Sol-H3's Triton kernel uses its own order and an
/// approximate rsqrt, so no implementation reproduces it bitwise; fixing both
/// here keeps the device and this twin identical. The rounding points (f32
/// chain, one bf16 rounding) are the reference's.
pub fn row_rsqrt(x: &[f32], eps: f32) -> f32 {
    let mut part = [0.0f32; NORM_THREADS];
    for (t, p) in part.iter_mut().enumerate() {
        let mut j = t;
        while j < x.len() {
            *p = x[j].mul_add(x[j], *p);
            j += NORM_THREADS;
        }
    }
    let mut s = NORM_THREADS / 2;
    while s > 0 {
        for t in 0..s {
            part[t] += part[t + s];
        }
        s >>= 1;
    }
    let v = part[0] / x.len() as f32 + eps;
    // f64 then one rounding: the correctly rounded rsqrt except where the
    // f64 value sits within 2^-53 of an f32 midpoint.
    (1.0 / f64::from(v).sqrt()) as f32
}

/// One row of `_rmsnorm_modulate_kernel`: `bf16(rms(x) * w * scale1p + shift)`
/// with `scale1p = 1 + scale` already formed in f32.
pub fn norm_mod_row(x: &[f32], w: &[f32], scale1p: &[f32], shift: &[f32], eps: f32) -> Vec<f32> {
    let r = row_rsqrt(x, eps);
    x.iter()
        .enumerate()
        .map(|(j, &v)| bf16_round((v * r * w[j]).mul_add(scale1p[j], shift[j])))
        .collect()
}

/// One row of `_residual_gate_rmsnorm_modulate_kernel`: returns
/// `(bf16(res + gate * branch), bf16(norm_mod(f32 hidden)))` — the norm sees
/// the unrounded f32 hidden.
#[allow(clippy::too_many_arguments)]
pub fn res_gate_norm_mod_row(
    res: &[f32],
    branch: &[f32],
    gate: &[f32],
    w: &[f32],
    scale1p: &[f32],
    shift: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let h: Vec<f32> = (0..res.len())
        .map(|j| gate[j].mul_add(branch[j], res[j]))
        .collect();
    let normed = norm_mod_row(&h, w, scale1p, shift, eps);
    (h.into_iter().map(bf16_round).collect(), normed)
}

/// Eager `residual + gate * branch` in bf16: the product rounds first.
pub fn gate_residual_eager(res: f32, gate: f32, branch: f32) -> f32 {
    bf16_round(res + bf16_round(gate * branch))
}

/// `_swiglu_kernel`: `bf16(value * (gate * sigmoid(gate)))`, value half first.
pub fn swiglu_row(x: &[f32]) -> Vec<f32> {
    let half = x.len() / 2;
    (0..half)
        .map(|j| {
            let (v, g) = (x[j], x[half + j]);
            let sg = 1.0 / (1.0 + (-g).exp());
            bf16_round(v * (g * sg))
        })
        .collect()
}

/// `_qknorm_partial_rope_kernel` for one head vector: RMS over `d`, weight,
/// then rotate_half over the leading `r` channels with f32 `cos`/`sin`.
pub fn qk_norm_rope_row(
    x: &[f32],
    w: &[f32],
    rope: Option<(&[f32], &[f32])>,
    eps: f32,
) -> Vec<f32> {
    let r_f = rms_factor(x, eps);
    let normed: Vec<f32> = x.iter().zip(w).map(|(&v, &wv)| v * r_f * wv).collect();
    match rope {
        None => normed.into_iter().map(bf16_round).collect(),
        Some((cos, sin)) => {
            let r = cos.len();
            let half = r / 2;
            (0..x.len())
                .map(|p| {
                    if p >= r {
                        return bf16_round(normed[p]);
                    }
                    let partner = if p < half { p + half } else { p - half };
                    let rot = if p < half {
                        -normed[partner]
                    } else {
                        normed[partner]
                    };
                    bf16_round(normed[p] * cos[p] + rot * sin[p])
                })
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Recipe tables
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QuantMode {
    #[default]
    Off,
    W8A8,
    Mxfp8,
}

impl QuantMode {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "off" | "0" | "none" | "bf16" => Ok(Self::Off),
            "w8a8" | "fp8" | "w8a8_fp8" => Ok(Self::W8A8),
            "mxfp8" | "mx" => Ok(Self::Mxfp8),
            other => Err(format!("{ENV}={other}: expected w8a8|mxfp8|off")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::W8A8 => "w8a8",
            Self::Mxfp8 => "mxfp8",
        }
    }

    /// `FASTVIDEO_H3_QUANT` (default off). An unknown value is an error, not
    /// a silent bf16 run.
    pub fn from_env() -> std::result::Result<Self, String> {
        match std::env::var(ENV) {
            Ok(v) => Self::parse(&v),
            Err(_) => Ok(Self::Off),
        }
    }

    pub fn kind(self) -> Option<QuantKind> {
        match self {
            Self::Off => None,
            Self::W8A8 => Some(QuantKind::W8A8),
            Self::Mxfp8 => Some(QuantKind::Mxfp8),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    W8A8,
    Mxfp8,
}

/// Which H3 linears a mode quantizes (the reference recipe tables).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H3QuantPlan {
    pub mode: QuantMode,
    pub num_layers: usize,
    pub num_refiner_layers: usize,
}

impl H3QuantPlan {
    pub fn new(mode: QuantMode, num_layers: usize, num_refiner_layers: usize) -> Self {
        Self {
            mode,
            num_layers,
            num_refiner_layers,
        }
    }

    /// DiT block `i`'s attention and FFN linears.
    pub fn dit_block(&self, i: usize) -> Option<QuantKind> {
        match self.mode {
            QuantMode::Off => None,
            QuantMode::W8A8 => (i < self.num_layers).then_some(QuantKind::W8A8),
            QuantMode::Mxfp8 => (MXFP8_FIRST_BLOCK
                ..=MXFP8_LAST_BLOCK.min(self.num_layers.saturating_sub(1)))
                .contains(&i)
                .then_some(QuantKind::Mxfp8),
        }
    }

    /// The text refiner's blocks: W8A8 tags them too (`quantized_linear_name`
    /// matches `token_refiner.refiner_blocks.*`); MXFP8 leaves refiners bf16.
    pub fn refiner(&self) -> Option<QuantKind> {
        match self.mode {
            QuantMode::W8A8 => Some(QuantKind::W8A8),
            _ => None,
        }
    }

    /// Linears as the reference counts them: W8A8 six per block (q, k, v,
    /// out, fc_in, fc_out); MXFP8 four (fused qkv, out, up, down).
    pub fn reference_linear_count(&self) -> usize {
        let dit = (0..self.num_layers)
            .filter(|&i| self.dit_block(i).is_some())
            .count();
        match self.mode {
            QuantMode::Off => 0,
            QuantMode::W8A8 => 6 * (dit + self.num_refiner_layers),
            QuantMode::Mxfp8 => 4 * dit,
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized weights
// ---------------------------------------------------------------------------

/// A row range of a (possibly fused) weight: quantized, or kept bf16 (the
/// VSA `to_gate_compress` rows the recipes never tag).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    pub rows: usize,
    pub quantized: bool,
}

#[derive(Clone, Copy, Debug)]
struct SectionLayout {
    row0: usize,
    data_off: usize,
    scale_off: usize,
}

/// One byte blob holds every section: E4M3 codes (1 B/elem) or bf16 rows
/// (2 B/elem) in order, then each MX section's swizzled scales. As a single
/// buffer it streams through [`super::offload`] like a bf16 weight.
#[derive(Clone, Debug)]
pub struct QuantLayout {
    pub kind: QuantKind,
    pub in_dim: usize,
    pub out_dim: usize,
    pub sections: Vec<Section>,
    lay: Vec<SectionLayout>,
    pub blob_bytes: usize,
}

impl QuantLayout {
    pub fn new(kind: QuantKind, in_dim: usize, sections: Vec<Section>) -> Result<Self> {
        if !in_dim.is_multiple_of(16) {
            return Err(msg(format!("fp8 linear needs K % 16 == 0, got {in_dim}")));
        }
        if kind == QuantKind::Mxfp8 && !in_dim.is_multiple_of(MX_BLOCK) {
            return Err(msg(format!("MXFP8 needs K % 32 == 0, got {in_dim}")));
        }
        let mut lay = Vec::with_capacity(sections.len());
        let (mut off, mut row0) = (0usize, 0usize);
        for s in &sections {
            if s.quantized && !s.rows.is_multiple_of(16) {
                return Err(msg(format!(
                    "fp8 section needs N % 16 == 0, got {}",
                    s.rows
                )));
            }
            lay.push(SectionLayout {
                row0,
                data_off: off,
                scale_off: 0,
            });
            off += s.rows * in_dim * if s.quantized { 1 } else { 2 };
            row0 += s.rows;
        }
        if kind == QuantKind::Mxfp8 {
            for (s, l) in sections.iter().zip(lay.iter_mut()) {
                if s.quantized {
                    l.scale_off = off;
                    off += mx_scale_len(s.rows, in_dim);
                }
            }
        }
        Ok(Self {
            kind,
            in_dim,
            out_dim: row0,
            sections,
            lay,
            blob_bytes: off.next_multiple_of(16),
        })
    }

    /// Quantize a row-major `[out_dim, in_dim]` weight on the host; values are
    /// rounded to bf16 first (the reference quantizes bf16 weights — after the
    /// bf16 LoRA merge). Returns the blob and one W8A8 scale per section.
    pub fn quantize_host(&self, w: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let k = self.in_dim;
        if w.len() != self.out_dim * k {
            return Err(msg(format!(
                "quant weight {} elements for [{}, {k}]",
                w.len(),
                self.out_dim
            )));
        }
        let mut blob = vec![0u8; self.blob_bytes];
        let mut scales = vec![0.0f32; self.sections.len()];
        for (i, (s, l)) in self.sections.iter().zip(&self.lay).enumerate() {
            let rows: Vec<f32> = w[l.row0 * k..(l.row0 + s.rows) * k]
                .iter()
                .map(|&v| bf16_round(v))
                .collect();
            if !s.quantized {
                for (j, v) in rows.iter().enumerate() {
                    let b = half::bf16::from_f32(*v).to_bits().to_le_bytes();
                    blob[l.data_off + 2 * j..][..2].copy_from_slice(&b);
                }
                continue;
            }
            match self.kind {
                QuantKind::W8A8 => {
                    let amax = amax_abs(&rows);
                    let amax = if amax.is_nan() { 0.0 } else { amax };
                    let scale = w8a8_scale(amax);
                    scales[i] = scale;
                    blob[l.data_off..][..rows.len()].copy_from_slice(&w8a8_quantize(&rows, scale));
                }
                QuantKind::Mxfp8 => {
                    let (q, sc) = mxfp8_quantize(&rows, s.rows, k);
                    blob[l.data_off..][..q.len()].copy_from_slice(&q);
                    blob[l.scale_off..][..sc.len()].copy_from_slice(&sc);
                }
            }
        }
        Ok((blob, scales))
    }

    /// The dequantized weight a quantized GEMM effectively multiplies by.
    pub fn dequantize_host(&self, blob: &[u8], scales: &[f32]) -> Vec<f32> {
        let k = self.in_dim;
        let mut w = vec![0.0f32; self.out_dim * k];
        for (i, (s, l)) in self.sections.iter().zip(&self.lay).enumerate() {
            let dst = &mut w[l.row0 * k..(l.row0 + s.rows) * k];
            if !s.quantized {
                for (j, d) in dst.iter_mut().enumerate() {
                    let b = [blob[l.data_off + 2 * j], blob[l.data_off + 2 * j + 1]];
                    *d = half::bf16::from_bits(u16::from_le_bytes(b)).to_f32();
                }
                continue;
            }
            let q = &blob[l.data_off..][..s.rows * k];
            match self.kind {
                QuantKind::W8A8 => dst.copy_from_slice(&w8a8_dequantize(q, scales[i])),
                QuantKind::Mxfp8 => {
                    let sc = &blob[l.scale_off..][..mx_scale_len(s.rows, k)];
                    dst.copy_from_slice(&mxfp8_dequantize(q, sc, s.rows, k));
                }
            }
        }
        w
    }

    /// Host emulation of the quantized linear on `x` `[m, in_dim]` (bf16
    /// values): the activation goes through the same recipe, the product
    /// accumulates in f64, the output rounds to bf16. bf16 sections multiply
    /// the unquantized activation.
    pub fn forward_host(&self, blob: &[u8], scales: &[f32], x: &[f32], m: usize) -> Vec<f32> {
        let k = self.in_dim;
        let w = self.dequantize_host(blob, scales);
        let xq: Vec<f32> = match self.kind {
            QuantKind::W8A8 => {
                let s = w8a8_scale(amax_abs(x));
                w8a8_dequantize(&w8a8_quantize(x, s), s)
            }
            QuantKind::Mxfp8 => {
                let (q, sc) = mxfp8_quantize(x, m, k);
                mxfp8_dequantize(&q, &sc, m, k)
            }
        };
        let n = self.out_dim;
        let mut out = vec![0.0f32; m * n];
        for (s, l) in self.sections.iter().zip(&self.lay) {
            let xs = if s.quantized { &xq } else { x };
            for i in 0..m {
                for o in l.row0..l.row0 + s.rows {
                    let acc: f64 = (0..k)
                        .map(|t| f64::from(xs[i * k + t]) * f64::from(w[o * k + t]))
                        .sum();
                    out[i * n + o] = bf16_round(acc as f32);
                }
            }
        }
        out
    }
}

/// A linear's quantized weight: layout, W8A8 scales, and the blob on the
/// device (streamable) or the host (CPU runs, emulated).
#[derive(Clone, Debug)]
pub struct QuantWeight {
    pub layout: std::sync::Arc<QuantLayout>,
    /// One f32 per section (W8A8; zero elsewhere).
    pub scales: Vec<f32>,
    #[cfg(feature = "cuda")]
    pub(crate) scales_dev: Option<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// The blob as bf16 words (`blob_bytes / 2`), so it streams like a weight.
    #[cfg(feature = "cuda")]
    pub(crate) blob_dev: Option<std::sync::Arc<cudarc::driver::CudaSlice<half::bf16>>>,
    pub(crate) blob_host: Option<std::sync::Arc<Vec<u8>>>,
}

impl QuantWeight {
    /// Host quantization (CPU runs and small weights).
    pub fn from_host(layout: QuantLayout, w: &[f32]) -> Result<Self> {
        let (blob, scales) = layout.quantize_host(w)?;
        #[cfg(feature = "cuda")]
        if super::stats::device_expected() {
            if let Some(dev) = super::device::global_device() {
                let words: Vec<half::bf16> = blob
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let blob_dev = dev
                    .stream
                    .memcpy_stod(&words)
                    .map_err(|e| msg(e.to_string()))?;
                let scales_dev = dev
                    .stream
                    .memcpy_stod(&scales)
                    .map_err(|e| msg(e.to_string()))?;
                super::stats::record_h2d(words.len() / 2);
                return Ok(Self {
                    layout: std::sync::Arc::new(layout),
                    scales,
                    scales_dev: Some(std::sync::Arc::new(scales_dev)),
                    blob_dev: Some(std::sync::Arc::new(blob_dev)),
                    blob_host: None,
                });
            }
        }
        Ok(Self {
            layout: std::sync::Arc::new(layout),
            scales,
            #[cfg(feature = "cuda")]
            scales_dev: None,
            #[cfg(feature = "cuda")]
            blob_dev: None,
            blob_host: Some(std::sync::Arc::new(blob)),
        })
    }

    pub fn kind(&self) -> QuantKind {
        self.layout.kind
    }

    pub fn held_bytes(&self) -> u64 {
        self.layout.blob_bytes as u64
    }

    /// Every output row is quantized (no bf16 section needing the raw input).
    pub fn layout_all_quantized(&self) -> bool {
        self.layout.sections.iter().all(|s| s.quantized)
    }

    pub fn is_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.scales_dev.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Host emulation (CPU runs).
    pub fn forward_host(&self, x: &[f32], m: usize) -> Result<Vec<f32>> {
        let blob = self
            .blob_host
            .as_ref()
            .ok_or_else(|| msg("quantized linear: device weight on a host run"))?;
        Ok(self.layout.forward_host(blob, &self.scales, x, m))
    }
}

// ---------------------------------------------------------------------------
// Device: quantization kernels and cuBLASLt GEMMs
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub use device_impl::*;
// `launch!` names `super::stats` / `super::device` from its call site.
#[cfg(feature = "cuda")]
use super::{device, stats};

#[cfg(feature = "cuda")]
mod device_impl {
    use std::sync::Arc;

    use cudarc::cublaslt::sys as lt;
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

    use super::super::device::{global_device, DeviceContext};
    use super::super::fp8::LtContext;
    use super::super::kernels::{cfg_n, cfg_rows, launch};
    use super::*;

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(e.to_string())
    }

    fn ctx() -> Result<Arc<DeviceContext>> {
        global_device().ok_or_else(|| msg("no global CUDA device context"))
    }

    pub fn ptr<T>(s: &CudaSlice<T>) -> u64 {
        let dev = global_device().expect("device");
        let (p, _g) = s.device_ptr(&dev.stream);
        p
    }

    pub fn ptr_mut<T>(s: &mut CudaSlice<T>) -> u64 {
        let dev = global_device().expect("device");
        let (p, _g) = s.device_ptr_mut(&dev.stream);
        p
    }

    /// An MXFP8 activation: `[rows_pad, k]` codes (pad rows zero, `rows_pad`
    /// = rows rounded up to 16) and swizzled scales for `rows`.
    pub struct MxAct {
        pub q: CudaSlice<u8>,
        pub s: CudaSlice<u8>,
        pub rows: usize,
        pub k: usize,
    }

    /// A W8A8 (tensorwise) activation: `[m_pad, k]` E4M3 codes (pad rows
    /// zero) and its f32 scale.
    pub struct W8Act {
        pub q: CudaSlice<u8>,
        pub scale: CudaSlice<f32>,
        pub m: usize,
        pub k: usize,
    }

    /// The last W8A8 activation quantized, keyed by its source buffer.
    struct W8Cached {
        key: super::super::tensor::ActKey,
        act: Arc<W8Act>,
    }
    static W8_CACHE: std::sync::Mutex<Option<W8Cached>> = std::sync::Mutex::new(None);

    /// The W8A8 quantization of `x` (`m x k`), shared by consecutive linears
    /// that read the same buffer (Q/K/V, cross-attention K/V): the
    /// tensorwise activation quantization depends on the activation alone.
    /// `None` for a host tensor.
    pub fn w8_cached_act(
        x: &super::super::tensor::CudaTensor,
        m: usize,
        k: usize,
    ) -> Result<Option<Arc<W8Act>>> {
        let Some(key) = x.act_key() else { return Ok(None) };
        let mut slot = W8_CACHE.lock().expect("w8a8 activation cache");
        if let Some(c) = slot.as_ref() {
            if c.key.matches(&key) && c.act.m == m && c.act.k == k {
                return Ok(Some(c.act.clone()));
            }
        }
        // Drop the previous entry before quantizing: its codes are dead weight.
        *slot = None;
        let x16 = x
            .dev_bf16()?
            .ok_or_else(|| msg("quantized linear without a device tensor"))?;
        let act = Arc::new(W8Act::quantize(ptr(x16.as_ref()), m, k)?);
        *slot = Some(W8Cached {
            key,
            act: act.clone(),
        });
        Ok(Some(act))
    }

    /// Forget the cached activation (its memory goes back to the pool);
    /// [`super::device::trim_pool`] calls this between phases.
    pub fn clear_w8_act_cache() {
        if let Ok(mut slot) = W8_CACHE.lock() {
            *slot = None;
        }
    }

    impl W8Act {
        /// Quantize bf16 `x` (`m x k`, device pointer) with the recipe's
        /// activation quantizer.
        pub fn quantize(x_bf16: u64, m: usize, k: usize) -> Result<Self> {
            let dev = ctx()?;
            let m_pad = m.next_multiple_of(16).max(16);
            let mut q = unsafe { dev.stream.alloc::<u8>(m_pad * k) }.map_err(err)?;
            let mut scale = dev.stream.alloc_zeros::<f32>(1).map_err(err)?;
            let (qp, sp) = (ptr_mut(&mut q), ptr_mut(&mut scale));
            w8a8_quantize_raw(x_bf16, true, m * k, qp, m_pad * k, sp)?;
            Ok(Self { q, scale, m, k })
        }
    }

    impl MxAct {
        /// Buffers for `rows x k`; the pad rows' codes and every scale start zero.
        pub fn alloc(rows: usize, k: usize) -> Result<Self> {
            let dev = ctx()?;
            let rows_pad = rows.next_multiple_of(16).max(16);
            let mut q = unsafe { dev.stream.alloc::<u8>(rows_pad * k) }.map_err(err)?;
            if rows_pad > rows {
                let mut tail = q.slice_mut(rows * k..);
                dev.stream.memset_zeros(&mut tail).map_err(err)?;
            }
            let s = dev
                .stream
                .alloc_zeros::<u8>(mx_scale_len(rows_pad, k))
                .map_err(err)?;
            Ok(Self { q, s, rows, k })
        }
    }

    /// `x` (f32 or bf16 bits at `x_ptr`) → MXFP8 into `out`.
    pub fn mxfp8_quantize_raw(x_ptr: u64, x16: bool, out: &mut MxAct) -> Result<()> {
        let dev = ctx()?;
        let (rows_i, k_i, x16_i) = (out.rows as i32, out.k as i32, i32::from(x16));
        let (qp, sp) = (ptr_mut(&mut out.q), ptr_mut(&mut out.s));
        launch!(dev.stream, &dev.kernels.mxfp8_quantize, cfg_rows(out.rows);
            &x_ptr, &x16_i, &qp, &sp, &rows_i, &k_i)
        .map_err(err)
    }

    /// The W8A8 activation / weight quantizer: amax, then codes and the f32
    /// scale. `n_total >= n_valid` pads with zero codes.
    pub fn w8a8_quantize_raw(
        x_ptr: u64,
        x16: bool,
        n_valid: usize,
        q_ptr: u64,
        n_total: usize,
        scale_ptr: u64,
    ) -> Result<()> {
        let dev = ctx()?;
        let mut amax = dev.stream.alloc_zeros::<f32>(1).map_err(err)?;
        let ap = ptr_mut(&mut amax);
        let x16_i = i32::from(x16);
        let n = n_valid as i64;
        let blocks = n_valid.div_ceil(256).clamp(1, 4096) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        launch!(dev.stream, &dev.kernels.amax_abs_mixed, cfg; &x_ptr, &x16_i, &ap, &n)
            .map_err(err)?;
        let (nv, nt) = (n_valid as i64, n_total as i64);
        launch!(dev.stream, &dev.kernels.w8a8_quantize, cfg_n(n_total);
            &x_ptr, &x16_i, &ap, &q_ptr, &scale_ptr, &nv, &nt)
        .map_err(err)
    }

    impl QuantWeight {
        /// Quantize a device bf16 `[out_dim, in_dim]` weight with the same
        /// kernels the activations use (bit-exact with [`QuantLayout::quantize_host`]).
        pub fn from_device_bf16(layout: QuantLayout, w: &CudaSlice<half::bf16>) -> Result<Self> {
            let dev = ctx()?;
            let k = layout.in_dim;
            if w.len() != layout.out_dim * k {
                return Err(msg(format!(
                    "quant weight {} elements for [{}, {k}]",
                    w.len(),
                    layout.out_dim
                )));
            }
            let mut blob = dev
                .stream
                .alloc_zeros::<half::bf16>(layout.blob_bytes / 2)
                .map_err(err)?;
            let mut scales_dev = dev
                .stream
                .alloc_zeros::<f32>(layout.sections.len().max(1))
                .map_err(err)?;
            let (bp, sp, wp) = (ptr_mut(&mut blob), ptr_mut(&mut scales_dev), ptr(w));
            for (i, (s, l)) in layout.sections.iter().zip(&layout.lay).enumerate() {
                let src = wp + (l.row0 * k * 2) as u64;
                let n = s.rows * k;
                if !s.quantized {
                    // Plain bf16 rows: a device-to-device byte copy.
                    let src_view = w.slice(l.row0 * k..l.row0 * k + n);
                    let mut dst_bytes = blob.slice_mut(l.data_off / 2..l.data_off / 2 + n);
                    dev.stream
                        .memcpy_dtod(&src_view, &mut dst_bytes)
                        .map_err(err)?;
                    continue;
                }
                match layout.kind {
                    QuantKind::W8A8 => {
                        w8a8_quantize_raw(
                            src,
                            true,
                            n,
                            bp + l.data_off as u64,
                            n,
                            sp + 4 * i as u64,
                        )?;
                    }
                    QuantKind::Mxfp8 => {
                        let (rows_i, k_i, one) = (s.rows as i32, k as i32, 1i32);
                        let (qp, qsp) = (bp + l.data_off as u64, bp + l.scale_off as u64);
                        launch!(dev.stream, &dev.kernels.mxfp8_quantize, cfg_rows(s.rows);
                            &src, &one, &qp, &qsp, &rows_i, &k_i)
                        .map_err(err)?;
                    }
                }
            }
            let scales = dev.stream.memcpy_dtov(&scales_dev).map_err(err)?;
            let mut scales = scales;
            scales.truncate(layout.sections.len());
            Ok(Self {
                layout: Arc::new(layout),
                scales,
                scales_dev: Some(Arc::new(scales_dev)),
                blob_dev: Some(Arc::new(blob)),
                blob_host: None,
            })
        }

        /// Take the blob out for streaming.
        pub(crate) fn take_blob(&mut self) -> Option<Arc<CudaSlice<half::bf16>>> {
            self.blob_dev.take()
        }

        pub(crate) fn put_blob(&mut self, b: Arc<CudaSlice<half::bf16>>) {
            self.blob_dev = Some(b);
        }

        pub(crate) fn has_blob(&self) -> bool {
            self.blob_dev.is_some()
        }

        /// `y [m, out_dim]` bf16 = x @ Wᵀ. `x` is bf16 bits (`x_bf16`, needed
        /// by bf16 sections and by the quantizer) and/or an already-quantized
        /// MX activation from a fused producer.
        pub fn forward_device(
            &self,
            x_bf16: Option<u64>,
            pre: Option<&MxAct>,
            m: usize,
        ) -> Result<CudaSlice<half::bf16>> {
            self.forward_device_with(x_bf16, pre, None, m)
        }

        /// [`Self::forward_device`] with an already-quantized W8A8 activation
        /// (`pre_w8`, from [`W8Act::quantize`] of the same `x_bf16`): the
        /// tensorwise activation quantization depends only on the activation,
        /// so linears sharing an input (Q/K/V) quantize it once.
        pub fn forward_device_with(
            &self,
            x_bf16: Option<u64>,
            pre: Option<&MxAct>,
            pre_w8: Option<&W8Act>,
            m: usize,
        ) -> Result<CudaSlice<half::bf16>> {
            let dev = ctx()?;
            let lay = &*self.layout;
            let (k, n_out) = (lay.in_dim, lay.out_dim);
            let blob = self
                .blob_dev
                .as_ref()
                .ok_or_else(|| msg("quantized linear: weight is streamed out"))?;
            let bp = ptr(blob.as_ref());
            let any_quant = lay.sections.iter().any(|s| s.quantized);
            let m_pad = m.next_multiple_of(16).max(16);
            // Activation operand for the FP8 sections.
            let mut own_mx = None;
            // Owns this call's quantized activation until the GEMMs are enqueued.
            let mut _own_w8 = None;
            let mut xq_w8: Option<u64> = None;
            let mut x_scale: Option<u64> = None;
            if any_quant {
                match lay.kind {
                    QuantKind::Mxfp8 => {
                        if pre.is_none() {
                            let xp = x_bf16.ok_or_else(|| msg("mxfp8 linear without input"))?;
                            let mut a = MxAct::alloc(m, k)?;
                            mxfp8_quantize_raw(xp, true, &mut a)?;
                            own_mx = Some(a);
                        }
                    }
                    QuantKind::W8A8 => match pre_w8 {
                        Some(a) if a.m == m && a.k == k => {
                            xq_w8 = Some(ptr(&a.q));
                            x_scale = Some(ptr(&a.scale));
                        }
                        Some(_) => return Err(msg("w8a8 linear: pre-quantized activation shape")),
                        None => {
                            let xp = x_bf16.ok_or_else(|| msg("w8a8 linear without input"))?;
                            let a = W8Act::quantize(xp, m, k)?;
                            xq_w8 = Some(ptr(&a.q));
                            x_scale = Some(ptr(&a.scale));
                            _own_w8 = Some(a);
                        }
                    },
                }
            }
            let mx = pre.or(own_mx.as_ref());
            let run = |n_tok: usize, d_ptr: u64| -> Result<()> {
                let ltc = super::super::fp8::lt_context(&dev)?;
                for (i, (s, l)) in lay.sections.iter().zip(&lay.lay).enumerate() {
                    let d = d_ptr + (l.row0 * 2) as u64;
                    let a = bp + l.data_off as u64;
                    let g = if !s.quantized {
                        LtGemm {
                            m: s.rows,
                            n: n_tok,
                            k,
                            a,
                            b: x_bf16.ok_or_else(|| msg("bf16 section without input"))?,
                            ab_type: lt::cudaDataType_t::CUDA_R_16BF,
                            d,
                            ldd: n_out,
                            scale: LtScale::None,
                            bias: None,
                        }
                    } else {
                        match lay.kind {
                            QuantKind::W8A8 => LtGemm {
                                m: s.rows,
                                n: n_tok,
                                k,
                                a,
                                b: xq_w8.ok_or_else(|| msg("w8a8 activation missing"))?,
                                ab_type: lt::cudaDataType_t::CUDA_R_8F_E4M3,
                                d,
                                ldd: n_out,
                                scale: LtScale::Tensor {
                                    a: ptr(self.scales_dev.as_ref().unwrap().as_ref())
                                        + 4 * i as u64,
                                    b: x_scale.ok_or_else(|| msg("w8a8 scale missing"))?,
                                },
                                bias: None,
                            },
                            QuantKind::Mxfp8 => {
                                let act = mx.ok_or_else(|| msg("mxfp8 activation missing"))?;
                                LtGemm {
                                    m: s.rows,
                                    n: n_tok,
                                    k,
                                    a,
                                    b: ptr(&act.q),
                                    ab_type: lt::cudaDataType_t::CUDA_R_8F_E4M3,
                                    d,
                                    ldd: n_out,
                                    scale: LtScale::Mx {
                                        a: bp + l.scale_off as u64,
                                        b: ptr(&act.s),
                                    },
                                    bias: None,
                                }
                            }
                        }
                    };
                    unsafe { lt_matmul(&dev, &ltc, &g)? };
                }
                Ok(())
            };
            let mut out =
                unsafe { dev.stream.alloc::<half::bf16>((m * n_out).max(1)) }.map_err(err)?;
            let op = ptr_mut(&mut out);
            match run(m, op) {
                Ok(()) => Ok(out),
                Err(first) if m != m_pad => {
                    // No algorithm for an unaligned token count: run the
                    // zero-padded rows and keep the first `m`. Every such
                    // call pays a failed heuristic query and a copy: said once.
                    static SAID: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!(
                            "[fastvideo] quantized linear: {first}; running {m_pad} zero-padded rows instead (and copying back) on every such call"
                        );
                    }
                    let mut padded =
                        unsafe { dev.stream.alloc::<half::bf16>(m_pad * n_out) }.map_err(err)?;
                    let pp = ptr_mut(&mut padded);
                    if lay.sections.iter().any(|s| !s.quantized) {
                        return Err(msg(
                            "quantized linear: unaligned token count with a bf16 section",
                        ));
                    }
                    run(m_pad, pp)?;
                    let src = padded.slice(0..m * n_out);
                    dev.stream.memcpy_dtod(&src, &mut out).map_err(err)?;
                    Ok(out)
                }
                Err(e) => Err(e),
            }
        }
    }

    // ---- cuBLASLt ------------------------------------------------------

    pub(crate) enum LtScale {
        None,
        /// Per-tensor f32 device scalars.
        Tensor {
            a: u64,
            b: u64,
        },
        /// `VEC32_UE8M0` swizzled scale tensors.
        Mx {
            a: u64,
            b: u64,
        },
    }

    /// Row-major `Y[n, m] = X[n, k] · A[m, k]ᵀ` as cuBLAS's column-major TN:
    /// `A` is the weight (`k x m`, ld k, transposed), `B` the activations
    /// (`k x n`, ld k), `D` bf16 `m x n` with leading dimension `ldd` (a
    /// row slice of a wider output).
    pub(crate) struct LtGemm {
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub a: u64,
        pub b: u64,
        pub ab_type: lt::cudaDataType_t,
        pub d: u64,
        pub ldd: usize,
        pub scale: LtScale,
        /// bf16 `[m]` bias added in the epilogue (one rounding, as torch's
        /// addmm epilogue).
        pub bias: Option<u64>,
    }

    // cublasLt.h attribute codes. Set through the untyped entry point so the
    // block-scale attributes (CUDA >= 12.8) do not depend on which toolkit
    // cudarc's enum was generated for.
    const TRANSA: u32 = 3;
    const TRANSB: u32 = 4;
    const EPILOGUE: u32 = 7;
    const BIAS_POINTER: u32 = 8;
    const A_SCALE_POINTER: u32 = 17;
    const B_SCALE_POINTER: u32 = 18;
    const BIAS_DATA_TYPE: u32 = 26;
    const A_SCALE_MODE: u32 = 31;
    const B_SCALE_MODE: u32 = 32;
    const EPILOGUE_BIAS: u32 = 4;
    const SCALE_VEC32_UE8M0: i32 = 2;

    fn check(status: lt::cublasStatus_t, what: &str) -> Result<()> {
        if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(msg(format!("{what} failed: {status:?}")))
        }
    }

    unsafe fn set_raw<T>(desc: lt::cublasLtMatmulDesc_t, attr: u32, v: &T) -> Result<()> {
        type SetAttr = unsafe extern "C" fn(
            lt::cublasLtMatmulDesc_t,
            u32,
            *const std::ffi::c_void,
            usize,
        ) -> lt::cublasStatus_t;
        // Same C ABI: the typed binding's enum argument is a 32-bit C enum.
        let f: SetAttr = std::mem::transmute(lt::culib().cublasLtMatmulDescSetAttribute);
        check(
            f(desc, attr, (v as *const T).cast(), std::mem::size_of::<T>()),
            &format!("cublasLtMatmulDescSetAttribute({attr})"),
        )
    }

    struct Desc(lt::cublasLtMatmulDesc_t);
    impl Drop for Desc {
        fn drop(&mut self) {
            unsafe { lt::cublasLtMatmulDescDestroy(self.0) };
        }
    }
    struct Layout(lt::cublasLtMatrixLayout_t);
    impl Drop for Layout {
        fn drop(&mut self) {
            unsafe { lt::cublasLtMatrixLayoutDestroy(self.0) };
        }
    }
    struct Pref(lt::cublasLtMatmulPreference_t);
    impl Drop for Pref {
        fn drop(&mut self) {
            unsafe { lt::cublasLtMatmulPreferenceDestroy(self.0) };
        }
    }

    /// # Safety
    /// Every pointer is a live device allocation of the implied size.
    pub(crate) unsafe fn lt_matmul(dev: &DeviceContext, ltc: &LtContext, g: &LtGemm) -> Result<()> {
        let f32_ty = lt::cudaDataType_t::CUDA_R_32F;
        let bf = lt::cudaDataType_t::CUDA_R_16BF;
        let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        check(
            lt::cublasLtMatmulDescCreate(
                &mut desc,
                lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                f32_ty,
            ),
            "cublasLtMatmulDescCreate",
        )?;
        let desc = Desc(desc);
        use cudarc::cublas::sys::cublasOperation_t;
        set_raw(desc.0, TRANSA, &(cublasOperation_t::CUBLAS_OP_T as i32))?;
        set_raw(desc.0, TRANSB, &(cublasOperation_t::CUBLAS_OP_N as i32))?;
        match g.scale {
            LtScale::None => {}
            LtScale::Tensor { a, b } => {
                set_raw(desc.0, A_SCALE_POINTER, &a)?;
                set_raw(desc.0, B_SCALE_POINTER, &b)?;
            }
            LtScale::Mx { a, b } => {
                set_raw(desc.0, A_SCALE_MODE, &SCALE_VEC32_UE8M0)?;
                set_raw(desc.0, B_SCALE_MODE, &SCALE_VEC32_UE8M0)?;
                set_raw(desc.0, A_SCALE_POINTER, &a)?;
                set_raw(desc.0, B_SCALE_POINTER, &b)?;
            }
        }
        if let Some(bias) = g.bias {
            set_raw(desc.0, EPILOGUE, &EPILOGUE_BIAS)?;
            set_raw(desc.0, BIAS_POINTER, &bias)?;
            set_raw(desc.0, BIAS_DATA_TYPE, &(bf as i32))?;
        }
        let mut la: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check(
            lt::cublasLtMatrixLayoutCreate(&mut la, g.ab_type, g.k as u64, g.m as u64, g.k as i64),
            "layout A",
        )?;
        let la = Layout(la);
        check(
            lt::cublasLtMatrixLayoutCreate(&mut lb, g.ab_type, g.k as u64, g.n as u64, g.k as i64),
            "layout B",
        )?;
        let lb = Layout(lb);
        check(
            lt::cublasLtMatrixLayoutCreate(&mut ld, bf, g.m as u64, g.n as u64, g.ldd as i64),
            "layout D",
        )?;
        let ld = Layout(ld);
        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        check(lt::cublasLtMatmulPreferenceCreate(&mut pref), "preference")?;
        let pref = Pref(pref);
        let ws = ltc.workspace_bytes;
        check(
            lt::cublasLtMatmulPreferenceSetAttribute(
                pref.0,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws as *const usize).cast(),
                std::mem::size_of::<usize>(),
            ),
            "preference workspace",
        )?;
        let mut heuristic = std::mem::MaybeUninit::<lt::cublasLtMatmulHeuristicResult_t>::uninit();
        let mut found: i32 = 0;
        let st = lt::cublasLtMatmulAlgoGetHeuristic(
            ltc.handle,
            desc.0,
            la.0,
            lb.0,
            ld.0,
            ld.0,
            pref.0,
            1,
            heuristic.as_mut_ptr(),
            &mut found,
        );
        if st != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || found == 0 {
            return Err(msg(format!(
                "cuBLASLt has no {} algorithm for m={} n={} k={} ldd={} on sm{}{} ({st:?})",
                match g.scale {
                    LtScale::None => "bf16",
                    LtScale::Tensor { .. } => "tensorwise FP8",
                    LtScale::Mx { .. } => "MXFP8 (VEC32_UE8M0, CUDA >= 12.8)",
                },
                g.m,
                g.n,
                g.k,
                g.ldd,
                dev.sm_major,
                dev.sm_minor
            )));
        }
        let heuristic = heuristic.assume_init();
        let (alpha, beta) = (1.0f32, 0.0f32);
        let (ws_ptr, _ws_guard) = ltc.workspace.device_ptr(&dev.stream);
        check(
            lt::cublasLtMatmul(
                ltc.handle,
                desc.0,
                (&alpha as *const f32).cast(),
                g.a as *const _,
                la.0,
                g.b as *const _,
                lb.0,
                (&beta as *const f32).cast(),
                g.d as *const _,
                ld.0,
                g.d as *mut _,
                ld.0,
                &heuristic.algo,
                ws_ptr as *mut _,
                ltc.workspace_bytes,
                dev.stream.cu_stream() as *mut _,
            ),
            "cublasLtMatmul",
        )
    }

    /// bf16 `[m, n]` = `x [m, k]` · `w [n, k]ᵀ` + bias (bf16, epilogue), one
    /// rounding: torch's `F.linear` on bf16. `None` when cuBLASLt declines.
    pub fn linear_bf16_bias(
        x: &CudaSlice<half::bf16>,
        w: &CudaSlice<half::bf16>,
        bias16: &CudaSlice<half::bf16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Option<CudaSlice<half::bf16>>> {
        let dev = ctx()?;
        let ltc = super::super::fp8::lt_context(&dev)?;
        let mut out = unsafe { dev.stream.alloc::<half::bf16>((m * n).max(1)) }.map_err(err)?;
        let g = LtGemm {
            m: n,
            n: m,
            k,
            a: ptr(w),
            b: ptr(x),
            ab_type: lt::cudaDataType_t::CUDA_R_16BF,
            d: ptr_mut(&mut out),
            ldd: n,
            scale: LtScale::None,
            bias: Some(ptr(bias16)),
        };
        match unsafe { lt_matmul(&dev, &ltc, &g) } {
            Ok(()) => Ok(Some(out)),
            Err(e) => {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                super::super::log::info_once(
                    &WARNED,
                    format_args!("bf16 bias epilogue unavailable: {e}"),
                );
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_satfinite_matches_the_ops_reference_on_finite_values() {
        // fastvideo_ops::fp8 is the exhaustively tested RNE/saturating
        // converter; the two differ only on NaN (hardware keeps NaN).
        let mut x = -500.0f32;
        while x < 500.0 {
            assert_eq!(e4m3_satfinite(x), fastvideo_ops::fp8::f32_to_e4m3(x), "{x}");
            x += 0.013;
        }
        for &v in &[
            0.0f32,
            -0.0,
            1e-9,
            2f32.powi(-9),
            2f32.powi(-10),
            3.0 * 2f32.powi(-10),
            448.0,
            464.0,
            1e9,
            f32::INFINITY,
        ] {
            assert_eq!(e4m3_satfinite(v), fastvideo_ops::fp8::f32_to_e4m3(v), "{v}");
        }
        assert_eq!(e4m3_satfinite(f32::NAN) & 0x7F, 0x7F);
        for b in 0..=255u8 {
            let v = e4m3_decode(b);
            if v.is_nan() {
                continue;
            }
            assert_eq!(e4m3_satfinite(v), b, "code {b:#x} round-trips");
        }
    }

    #[test]
    fn mx_scale_is_the_smallest_power_of_two_keeping_the_block_under_448() {
        for &amax in &[
            1.0f32, 448.0, 449.0, 1.75, 1.7500001, 3.5, 0.001, 1e-30, 65504.0,
        ] {
            let (byte, inv) = mx_e8m0(amax);
            let e = i32::from(byte) - 127;
            assert_eq!(inv, 2f32.powi(-e));
            assert!(amax * inv <= 448.0, "{amax}: {}", amax * inv);
            // One power of two smaller would overflow 448 (or hit the floor).
            assert!(e == -127 || amax * inv * 2.0 > 448.0, "{amax} not tight");
        }
        assert_eq!(mx_e8m0(0.0).0, 0);
    }

    #[test]
    fn mx_swizzle_is_a_bijection_on_padded_tiles() {
        let (rows, k) = (256, 32 * 12);
        let cb = (k / 32usize).div_ceil(4);
        let mut seen = vec![false; mx_scale_len(rows, k)];
        for r in 0..rows {
            for g in 0..k / 32 {
                let o = mx_scale_offset(r, g, cb);
                assert!(!seen[o]);
                seen[o] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
        // First tile, torch SWIZZLE_32_4_4: row 33 group 5 lands at 1*512+1*16+1*4+1.
        assert_eq!(mx_scale_offset(33, 5, cb), 512 + 16 + 4 + 1);
    }

    #[test]
    fn mxfp8_round_trip_error_is_bounded_by_the_block_scale() {
        let (rows, k) = (3, 64);
        let x: Vec<f32> = (0..rows * k)
            .map(|i| bf16_round(((i as f32 * 0.37).sin()) * 10f32.powi((i % 7) as i32 - 3)))
            .collect();
        let (q, s) = mxfp8_quantize(&x, rows, k);
        let y = mxfp8_dequantize(&q, &s, rows, k);
        for r in 0..rows {
            for g in 0..k / 32 {
                let blk = &x[r * k + g * 32..][..32];
                let amax = amax_abs(blk);
                for j in 0..32 {
                    let i = r * k + g * 32 + j;
                    // Half an E4M3 ulp at the block's top binade.
                    assert!(
                        (y[i] - x[i]).abs() <= amax / 16.0 + 1e-30,
                        "{i}: {} vs {}",
                        y[i],
                        x[i]
                    );
                }
            }
        }
    }

    #[test]
    fn w8a8_rounds_the_quotient_to_bf16_before_fp8_and_clamps_the_scale() {
        assert_eq!(w8a8_scale(0.0), W8A8_MIN_SCALE);
        assert_eq!(w8a8_scale(448.0), 1.0);
        // A scale whose bf16 rounding changes the quotient: the codes follow
        // bf16(x / bf16(s)), not x / s.
        let s = w8a8_scale(3.0);
        let s16 = bf16_round(s);
        assert_ne!(s, s16);
        let x = [3.0f32, -1.25, 0.001];
        let q = w8a8_quantize(&x, s);
        for (&v, &b) in x.iter().zip(&q) {
            assert_eq!(b, e4m3_satfinite(bf16_round(v / s16)));
        }
        // amax lands on (or within one bf16 step of) 448.
        assert!(e4m3_decode(q[0]).abs() >= 440.0);
    }

    #[test]
    fn recipe_tables_match_the_reference_counts() {
        let w8 = H3QuantPlan::new(QuantMode::W8A8, 50, 2);
        assert_eq!(w8.reference_linear_count(), 312, "stage1.py expects 312");
        assert_eq!(w8.refiner(), Some(QuantKind::W8A8));
        let mx = H3QuantPlan::new(QuantMode::Mxfp8, 50, 2);
        assert_eq!(mx.dit_block(1), None);
        assert_eq!(mx.dit_block(2), Some(QuantKind::Mxfp8));
        assert_eq!(mx.dit_block(46), Some(QuantKind::Mxfp8));
        assert_eq!(mx.dit_block(47), None);
        assert_eq!(mx.refiner(), None);
        assert_eq!(mx.reference_linear_count(), 45 * 4);
        assert_eq!(QuantMode::parse("W8A8").unwrap(), QuantMode::W8A8);
        assert_eq!(QuantMode::parse("off").unwrap(), QuantMode::Off);
        assert!(QuantMode::parse("int4").is_err());
    }

    #[test]
    fn a_section_layout_quantizes_each_tensor_with_its_own_scale() {
        let k = 32;
        let lay = QuantLayout::new(
            QuantKind::W8A8,
            k,
            vec![
                Section {
                    rows: 16,
                    quantized: true,
                },
                Section {
                    rows: 16,
                    quantized: true,
                },
                Section {
                    rows: 16,
                    quantized: false,
                },
            ],
        )
        .unwrap();
        let mut w = vec![0.0f32; 48 * k];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f32 * 0.1).sin() * if i < 16 * k { 1.0 } else { 0.01 };
        }
        let (blob, scales) = lay.quantize_host(&w).unwrap();
        assert!(
            (scales[0] / scales[1]) > 50.0,
            "per-section scales {scales:?}"
        );
        let dq = lay.dequantize_host(&blob, &scales);
        // bf16 section is exact to bf16.
        for i in 32 * k..48 * k {
            assert_eq!(dq[i], bf16_round(w[i]));
        }
        let x: Vec<f32> = (0..2 * k)
            .map(|i| bf16_round((i as f32 * 0.3).cos()))
            .collect();
        let y = lay.forward_host(&blob, &scales, &x, 2);
        assert_eq!(y.len(), 2 * 48);
        // bf16 section of the output uses the unquantized activation.
        let want: f32 = (0..k).map(|t| x[t] * bf16_round(w[40 * k + t])).sum();
        assert!((y[40] - bf16_round(want)).abs() <= want.abs() * 1e-2 + 1e-6);
    }

    #[test]
    fn the_norm_twin_follows_the_kernel_reduction_tree() {
        // 600 values: lanes 0..87 hold three, the rest two; the tree then halves.
        let x: Vec<f32> = (0..600).map(|i| ((i as f32) * 0.731).sin() * 3.0).collect();
        let mut part = vec![0.0f32; NORM_THREADS];
        for (j, &v) in x.iter().enumerate() {
            let p = &mut part[j % NORM_THREADS];
            *p = v.mul_add(v, *p);
        }
        while part.len() > 1 {
            let h = part.len() / 2;
            let (a, b) = part.split_at(h);
            part = a.iter().zip(b).map(|(p, q)| p + q).collect();
        }
        let want = (1.0 / f64::from(part[0] / 600.0 + 1e-6).sqrt()) as f32;
        assert_eq!(row_rsqrt(&x, 1e-6).to_bits(), want.to_bits());
        let naive = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / 600.0 + 1e-6).sqrt();
        assert!((row_rsqrt(&x, 1e-6) - naive).abs() <= naive * 4e-7);
    }

    #[test]
    fn fused_references_pin_their_rounding_points() {
        let d = 64;
        let x: Vec<f32> = (0..d)
            .map(|i| bf16_round((i as f32 * 0.7).sin() * 3.0))
            .collect();
        let w: Vec<f32> = (0..d).map(|i| bf16_round(1.0 + i as f32 * 1e-3)).collect();
        let sc: Vec<f32> = (0..d).map(|i| 1.0 + bf16_round(i as f32 * 1e-2)).collect();
        let sh: Vec<f32> = (0..d).map(|i| bf16_round(-(i as f32) * 1e-2)).collect();
        let out = norm_mod_row(&x, &w, &sc, &sh, 1e-6);
        // Every output is bf16, and it is the single rounding of the f32 chain.
        for (j, &o) in out.iter().enumerate() {
            assert_eq!(o, bf16_round(o));
            let r = row_rsqrt(&x, 1e-6);
            assert_eq!(o, bf16_round((x[j] * r * w[j]).mul_add(sc[j], sh[j])));
        }
        // Residual kernel normalizes the f32 hidden, not the stored bf16 one.
        let br: Vec<f32> = (0..d)
            .map(|i| bf16_round((i as f32 * 0.11).cos()))
            .collect();
        let g: Vec<f32> = vec![bf16_round(0.3337); d];
        let (h, n2) = res_gate_norm_mod_row(&x, &br, &g, &w, &sc, &sh, 1e-6);
        let h32: Vec<f32> = (0..d).map(|j| g[j].mul_add(br[j], x[j])).collect();
        assert_eq!(n2, norm_mod_row(&h32, &w, &sc, &sh, 1e-6));
        assert_eq!(h, h32.iter().map(|&v| bf16_round(v)).collect::<Vec<_>>());
        // Eager gate residual rounds the product first.
        let (r, gt, b) = (1.0f32, 0.3337f32, 0.001234f32);
        assert_eq!(
            gate_residual_eager(r, gt, b),
            bf16_round(r + bf16_round(gt * b))
        );
        // SwiGLU is value-first and rounds once.
        let packed = [2.0f32, -1.0, 0.5, 3.0];
        let s = swiglu_row(&packed);
        assert_eq!(s[0], bf16_round(2.0 * (0.5 / (1.0 + (-0.5f32).exp()))));
    }
}
