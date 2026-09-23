//! LongLive 2.0 NVFP4 (NVlabs/LongLive `utils/quant.py`,
//! `fouroversix/quantize/pytorch/reference.py`).
//!
//! Shared, default-off W4A4 primitive for DiT linears. Unset / `off` / `none` /
//! `false` / `0` keeps today's dense path. `FASTVIDEO_NVFP4=1` (or `mse` /
//! `nvfp4`) applies the published FourOverSix MSE scale rule; `static_6` is
//! plain NVFP4 (`amax / 6` with E4M3 block scales).
//!
//! Format (inference yaml + FourOverSix reference):
//! - E2M1 codes packed two-per-byte, low nibble first
//! - 16-wide blocks along the last dim
//! - per-block E4M3FN scales and one FP32 tensor `amax`
//! - dequant `e2m1 * e4m3_scale * amax / (e2m1_max * e4m3_max)`
//! - matmul is dequant-then-FP32 (`fouroversix/matmul/pytorch.py`)
//!
//! cuda-oxide (custom rustc PTX backend, `cargo oxide`, Linux nightly) and
//! cutile-rs (tile DSL + that backend) are not linked: they are not crates.io
//! dependencies this workspace can typecheck without a new undocumented
//! toolchain. CUTLASS SM100/SM120 GEMM, Blackwell `to_blocked` scale layout,
//! RHT, 2D block scales, stochastic rounding, and the fused CUDA KV-dequant
//! kernel are gap-logged — they are kernel names or layouts without a host
//! recipe we can run here.

use fastvideo_ops::fp8::{e4m3_to_f32, f32_to_e4m3, E4M3_MAX};

/// Env var LongLive's `model_quant` maps onto in this repo.
pub const ENV: &str = "FASTVIDEO_NVFP4";

/// NVFP4 block width (`DataType.nvfp4.block_size()`).
pub const BLOCK: usize = 16;

/// Largest finite E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;

/// FourOverSix MSE/MAE/abs_max uses 256, not 448, as the E4M3 peak.
pub const E4M3_MAX_FOUROVERSIX: f32 = 256.0;

/// Expansion that turns a `static_6` scale into the FourOverSix "4" candidate
/// (`6 / 4 = 1.5` in `quantize_to_nvfp4(..., scale_expansion_factor=1.5)`).
const FOUR_OVER_SIX_EXPANSION: f32 = 1.5;

/// Why the Blackwell GEMM / fused KV kernel / oxide compilers stay out.
pub const GAP: &str = "\
longlive nvfp4: dequant-then-matmul only (fouroversix pytorch backend). \
CUTLASS SM100/SM120 GEMM, TransformerEngine NVFP4BlockScaling, Blackwell \
to_blocked scale layout, RHT, 2d block scales, stochastic rounding, and \
utils/kernel/kv_dequant_cuda.cu are unpublished as host math. \
fused Triton RoPE/adaLN (utils/rope_triton.py, utils/adaln_triton.py) is a \
launch fusion of recipes this repo already has (rope_half, ln_adaln_e). \
cuda-oxide and cutile-rs need cargo-oxide + nightly rustc-codegen-cuda, \
not a crate this workspace can depend on. \
TorchAO PerRow FP8 PTQ (utils/fp8.py) is not this flag; FASTVIDEO_FP8 is \
the existing per-tensor E4M3 path.";

/// LongLive `DEFAULT_GENERATOR_FILTERED_MODULES` — stay BF16.
const EXACT_DENSE: &[&str] = &[
    "text_embedding.0",
    "text_embedding.2",
    "patch_embedding",
    "time_projection.1",
    "time_embedding.0",
    "time_embedding.2",
    "head.head",
    "head.modulation",
];

/// `re:.*norm_k$` / `norm_q` / `norm1` / `norm2` / `norm3`.
const DENSE_SUFFIX: &[&str] = &["norm_k", "norm_q", "norm1", "norm2", "norm3"];

/// This repo's Wan / DiT names for the same conditioning and output linears.
const DENSE_ALIASES: &[&str] = &[
    "time_proj",
    "time_embedder",
    "time_embedding",
    "time_projection",
    "text_embedder",
    "text_embedding",
    "image_embedder",
    "proj_out",
    "patch_embedding",
];

/// Block-scale selection from `fouroversix.utils.ScaleRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleRule {
    /// `amax / 6`, E4M3 peak 448. TransformerEngine `NVFP4BlockScaling`.
    Static6,
    /// `amax / 4`, E4M3 peak 448.
    Static4,
    /// LongLive inference default (`model_quant_scale_rule: mse`): try 6 and
    /// 4-equivalent scales, keep the lower per-block MSE.
    Mse,
}

impl ScaleRule {
    pub fn e2m1_max(self) -> f32 {
        match self {
            Self::Static4 => 4.0,
            Self::Static6 | Self::Mse => E2M1_MAX,
        }
    }

    pub fn e4m3_max(self) -> f32 {
        match self {
            Self::Static4 | Self::Static6 => E4M3_MAX,
            Self::Mse => E4M3_MAX_FOUROVERSIX,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Static6 => "static_6",
            Self::Static4 => "static_4",
            Self::Mse => "mse",
        }
    }
}

/// `FASTVIDEO_NVFP4`. Unset / off / none / false / 0 → [`None`].
pub fn requested(value: Option<&str>) -> Option<ScaleRule> {
    let Some(v) = value.map(str::trim) else {
        return None;
    };
    if v.is_empty() {
        return None;
    }
    match v.to_ascii_lowercase().as_str() {
        "0" | "off" | "false" | "none" | "no" => None,
        "static_6" | "static6" | "te" => Some(ScaleRule::Static6),
        "static_4" | "static4" => Some(ScaleRule::Static4),
        "1" | "true" | "on" | "nvfp4" | "mse" | "4o6" | "fouroversix" | "w4a4" => {
            Some(ScaleRule::Mse)
        }
        _ => None,
    }
}

/// Process env. Safe to call from load paths; not cached (tests mutate env).
pub fn from_env() -> Option<ScaleRule> {
    requested(std::env::var(ENV).ok().as_deref())
}

/// LongLive generator filter plus this repo's equivalent module names.
pub fn linear_stays_dense(name: &str) -> bool {
    let name = name.trim().trim_matches('.');
    if name.is_empty() {
        return true;
    }
    if EXACT_DENSE
        .iter()
        .any(|p| name == *p || name.ends_with(&format!(".{p}")))
    {
        return true;
    }
    if DENSE_SUFFIX
        .iter()
        .any(|s| name == *s || name.ends_with(&format!(".{s}")))
    {
        return true;
    }
    DENSE_ALIASES
        .iter()
        .any(|s| name == *s || name.ends_with(&format!(".{s}")) || name.contains(&format!(".{s}.")))
}

/// Core DiT linear: flag on, not a filtered module, K divisible by 16.
pub fn linear_eligible(name: &str, in_dim: usize) -> bool {
    from_env().is_some() && !linear_stays_dense(name) && in_dim.is_multiple_of(BLOCK) && in_dim > 0
}

/// Packed NVFP4 tensor. Scales are E4M3 codes, `[rows, cols/BLOCK]`.
#[derive(Debug, Clone)]
pub struct Nvfp4Tensor {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub amax: f32,
    pub rows: usize,
    pub cols: usize,
    pub rule: ScaleRule,
}

impl Nvfp4Tensor {
    pub fn dequantize(&self) -> Vec<f32> {
        let n_blocks = self.cols / BLOCK;
        let decode = if self.amax > 0.0 && self.amax.is_finite() {
            self.amax / (self.rule.e2m1_max() * self.rule.e4m3_max())
        } else {
            0.0
        };
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for b in 0..n_blocks {
                let scale = e4m3_to_f32(self.scales[r * n_blocks + b]) * decode;
                let packed_off = r * (self.cols / 2) + b * (BLOCK / 2);
                let out_off = r * self.cols + b * BLOCK;
                for i in 0..BLOCK / 2 {
                    let byte = self.packed[packed_off + i];
                    out[out_off + 2 * i] = e2m1_to_f32(byte & 0x0F) * scale;
                    out[out_off + 2 * i + 1] = e2m1_to_f32(byte >> 4) * scale;
                }
            }
        }
        out
    }
}

/// Quantize `[rows, cols]` (row-major) with `cols` a multiple of [`BLOCK`].
pub fn quantize(
    x: &[f32],
    rows: usize,
    cols: usize,
    rule: ScaleRule,
) -> Result<Nvfp4Tensor, String> {
    if x.len() != rows.saturating_mul(cols) {
        return Err(format!("nvfp4: {} elements for [{rows}, {cols}]", x.len()));
    }
    if cols == 0 || !cols.is_multiple_of(BLOCK) {
        return Err(format!("nvfp4: cols {cols} is not a multiple of {BLOCK}"));
    }
    let n_blocks = rows * (cols / BLOCK);
    let mut blocks = vec![0.0f32; n_blocks * BLOCK];
    for r in 0..rows {
        for b in 0..(cols / BLOCK) {
            let src = r * cols + b * BLOCK;
            let dst = (r * (cols / BLOCK) + b) * BLOCK;
            blocks[dst..dst + BLOCK].copy_from_slice(&x[src..src + BLOCK]);
        }
    }
    let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let (fake, scales) = match rule {
        ScaleRule::Static6 | ScaleRule::Static4 => {
            let (scaled, scales) = scale_blocks(&blocks, n_blocks, amax, rule, 1.0);
            (fake_quantize_e2m1(&scaled), scales)
        }
        ScaleRule::Mse => {
            let (scaled6, scales6) = scale_blocks(&blocks, n_blocks, amax, rule, 1.0);
            let (scaled4, scales4) =
                scale_blocks(&blocks, n_blocks, amax, rule, FOUR_OVER_SIX_EXPANSION);
            let fake6 = fake_quantize_e2m1(&scaled6);
            let fake4 = fake_quantize_e2m1(&scaled4);
            select_mse(&blocks, &fake6, &scales6, &fake4, &scales4, amax, n_blocks)
        }
    };
    let packed = pack_e2m1(&fake, rows, cols);
    Ok(Nvfp4Tensor {
        packed,
        scales,
        amax,
        rows,
        cols,
        rule,
    })
}

/// Quantize then dequantize. The Linear W4A4 op.
pub fn reconstruct(
    x: &[f32],
    rows: usize,
    cols: usize,
    rule: ScaleRule,
) -> Result<Vec<f32>, String> {
    Ok(quantize(x, rows, cols, rule)?.dequantize())
}

/// LongLive `k_smooth`: subtract the last-dim mean before KV quant.
pub fn k_smooth(k: &mut [f32], tokens: usize, dim: usize) {
    assert_eq!(k.len(), tokens.saturating_mul(dim));
    if dim == 0 {
        return;
    }
    let inv = 1.0 / dim as f32;
    for t in 0..tokens {
        let row = &mut k[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f32>() * inv;
        for v in row {
            *v -= mean;
        }
    }
}

fn scale_blocks(
    blocks: &[f32],
    n_blocks: usize,
    amax: f32,
    rule: ScaleRule,
    expansion: f32,
) -> (Vec<f32>, Vec<u8>) {
    let mut scaled = vec![0.0f32; n_blocks * BLOCK];
    let mut scales = vec![0u8; n_blocks];
    if amax == 0.0 || !amax.is_finite() {
        return (scaled, scales);
    }
    let e2 = rule.e2m1_max();
    let e4 = rule.e4m3_max();
    let encode = (e2 * e4) / amax;
    let decode = amax / (e2 * e4);
    for b in 0..n_blocks {
        let src = &blocks[b * BLOCK..(b + 1) * BLOCK];
        let block_amax = src.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let code = f32_to_e4m3((block_amax / e2) * encode * expansion);
        scales[b] = code;
        let s = e4m3_to_f32(code);
        if s != 0.0 {
            let inv = 1.0 / (decode * s);
            for i in 0..BLOCK {
                scaled[b * BLOCK + i] = src[i] * inv;
            }
        }
    }
    (scaled, scales)
}

fn select_mse(
    orig: &[f32],
    fake6: &[f32],
    scales6: &[u8],
    fake4: &[f32],
    scales4: &[u8],
    amax: f32,
    n_blocks: usize,
) -> (Vec<f32>, Vec<u8>) {
    let decode = if amax > 0.0 {
        amax / (E2M1_MAX * E4M3_MAX_FOUROVERSIX)
    } else {
        0.0
    };
    let mut fake = vec![0.0f32; n_blocks * BLOCK];
    let mut scales = vec![0u8; n_blocks];
    for b in 0..n_blocks {
        let o = &orig[b * BLOCK..(b + 1) * BLOCK];
        let s6 = e4m3_to_f32(scales6[b]) * decode;
        let s4 = e4m3_to_f32(scales4[b]) * decode;
        let mut mse6 = 0.0f32;
        let mut mse4 = 0.0f32;
        for i in 0..BLOCK {
            let d6 = fake6[b * BLOCK + i] * s6 - o[i];
            let d4 = fake4[b * BLOCK + i] * s4 - o[i];
            mse6 += d6 * d6;
            mse4 += d4 * d4;
        }
        let (src, code) = if mse4 < mse6 {
            (fake4, scales4[b])
        } else {
            (fake6, scales6[b])
        };
        fake[b * BLOCK..(b + 1) * BLOCK].copy_from_slice(&src[b * BLOCK..(b + 1) * BLOCK]);
        scales[b] = code;
    }
    (fake, scales)
}

/// FourOverSix `fake_quantize_to_e2m1` nearest (ties-to-even, like PyTorch).
pub fn fake_quantize_e2m1(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            if !v.is_finite() {
                return 0.0;
            }
            let a = v.abs();
            let mag = if a < 2.0 {
                round_ties_even(2.0 * a) * 0.5
            } else if a < 4.0 {
                round_ties_even(a)
            } else {
                2.0 * round_ties_even(a * 0.5)
            };
            if v.is_sign_negative() {
                -mag
            } else {
                mag
            }
        })
        .collect()
}

fn round_ties_even(x: f32) -> f32 {
    let f = x.floor();
    let frac = x - f;
    if frac < 0.5 {
        f
    } else if frac > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

fn pack_e2m1(fake: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let mut packed = vec![0u8; rows * (cols / 2)];
    for r in 0..rows {
        for c in 0..(cols / 2) {
            let lo = e2m1_from_f32(fake[r * cols + 2 * c]);
            let hi = e2m1_from_f32(fake[r * cols + 2 * c + 1]);
            packed[r * (cols / 2) + c] = lo | (hi << 4);
        }
    }
    packed
}

fn e2m1_from_f32(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x8 } else { 0 };
    let a = x.abs();
    let mag = if a < 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a < 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a < 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a < 5.0 {
        6
    } else {
        7
    };
    sign | mag
}

fn e2m1_to_f32(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(code & 7) as usize];
    if code & 8 != 0 {
        -v
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_is_off_until_named() {
        assert_eq!(requested(None), None);
        assert_eq!(requested(Some("")), None);
        assert_eq!(requested(Some("off")), None);
        assert_eq!(requested(Some("none")), None);
        assert_eq!(requested(Some("false")), None);
        assert_eq!(requested(Some("0")), None);
        assert_eq!(requested(Some("1")), Some(ScaleRule::Mse));
        assert_eq!(requested(Some("NVFP4")), Some(ScaleRule::Mse));
        assert_eq!(requested(Some("mse")), Some(ScaleRule::Mse));
        assert_eq!(requested(Some("static_6")), Some(ScaleRule::Static6));
        assert_eq!(requested(Some("static_4")), Some(ScaleRule::Static4));
        assert_eq!(requested(Some("w4a4")), Some(ScaleRule::Mse));
    }

    #[test]
    fn dense_modules_match_longlive_and_wan_aliases() {
        assert!(linear_stays_dense("text_embedding.0"));
        assert!(linear_stays_dense("generator.text_embedding.2"));
        assert!(linear_stays_dense("patch_embedding"));
        assert!(linear_stays_dense("time_projection.1"));
        assert!(linear_stays_dense("condition_embedder.time_proj"));
        assert!(linear_stays_dense(
            "condition_embedder.time_embedder.linear_1"
        ));
        assert!(linear_stays_dense("head.head"));
        assert!(linear_stays_dense("proj_out"));
        assert!(linear_stays_dense("blocks.0.norm1"));
        assert!(linear_stays_dense("blocks.3.attn1.norm_k"));
        assert!(!linear_stays_dense("blocks.0.attn1.to_q"));
        assert!(!linear_stays_dense("blocks.0.ffn.net.0.proj"));
        assert!(!linear_stays_dense("blocks.0.attn1.to_out.0"));
    }

    #[test]
    fn e2m1_codes_match_nibble_table() {
        for (v, bits) in [
            (0.0, 0x0),
            (0.5, 0x1),
            (1.0, 0x2),
            (1.5, 0x3),
            (2.0, 0x4),
            (3.0, 0x5),
            (4.0, 0x6),
            (6.0, 0x7),
            (-1.0, 0xA),
            (-6.0, 0xF),
        ] {
            assert_eq!(e2m1_from_f32(v), bits, "{v}");
            assert_eq!(e2m1_to_f32(bits), v, "{bits:#x}");
        }
    }

    #[test]
    fn fake_quant_uses_e2m1_grid() {
        let q = fake_quantize_e2m1(&[0.2, 0.6, 1.1, 1.6, 2.4, 3.4, 4.4, 5.6, -0.6]);
        assert_eq!(q, vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5]);
    }

    #[test]
    fn static6_full_scale_block_is_exact() {
        let x = vec![6.0f32; BLOCK];
        let rec = reconstruct(&x, 1, BLOCK, ScaleRule::Static6).unwrap();
        for (i, v) in rec.iter().enumerate() {
            assert!((v - 6.0).abs() < 1e-5, "elem {i}: {v}");
        }
    }

    #[test]
    fn static6_round_trips_a_block() {
        let mut x = vec![0.0f32; BLOCK];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 * 0.37).sin() * 3.0;
        }
        let rec = reconstruct(&x, 1, BLOCK, ScaleRule::Static6).unwrap();
        let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        for (a, b) in x.iter().zip(rec.iter()) {
            assert!(
                (a - b).abs() <= amax * 0.2 + 1e-5,
                "{a} -> {b} (amax {amax})"
            );
        }
        let qt = quantize(&x, 1, BLOCK, ScaleRule::Static6).unwrap();
        assert_eq!(qt.packed.len(), BLOCK / 2);
        assert_eq!(qt.scales.len(), 1);
        assert!(qt.amax > 0.0);
    }

    #[test]
    fn mse_picks_a_finite_reconstruction() {
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.41).cos() * 5.0).collect();
        let rec = reconstruct(&x, 2, BLOCK, ScaleRule::Mse).unwrap();
        assert_eq!(rec.len(), 32);
        assert!(rec.iter().all(|v| v.is_finite()));
        let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let err: f32 = x
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err <= amax * 0.25 + 1e-5, "max abs err {err} amax {amax}");
    }

    #[test]
    fn zero_tensor_stays_zero() {
        let z = vec![0.0f32; BLOCK];
        let rec = reconstruct(&z, 1, BLOCK, ScaleRule::Static6).unwrap();
        assert!(rec.iter().all(|&v| v == 0.0));
        let qt = quantize(&z, 1, BLOCK, ScaleRule::Mse).unwrap();
        assert_eq!(qt.amax, 0.0);
        assert!(qt.dequantize().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn k_smooth_subtracts_last_dim_mean() {
        let mut k = vec![1.0f32, 3.0, 5.0, 2.0, 2.0, 2.0];
        k_smooth(&mut k, 2, 3);
        assert!((k[0] + 2.0).abs() < 1e-6 && (k[1]).abs() < 1e-6 && (k[2] - 2.0).abs() < 1e-6);
        assert!(k[3..].iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn scale_limits_match_fouroversix() {
        assert_eq!(
            (ScaleRule::Static6.e2m1_max(), ScaleRule::Static6.e4m3_max()),
            (6.0, 448.0)
        );
        assert_eq!(
            (ScaleRule::Static4.e2m1_max(), ScaleRule::Static4.e4m3_max()),
            (4.0, 448.0)
        );
        assert_eq!(
            (ScaleRule::Mse.e2m1_max(), ScaleRule::Mse.e4m3_max()),
            (6.0, 256.0)
        );
    }

    #[test]
    fn gap_names_the_unpublished_kernels() {
        assert!(GAP.contains("CUTLASS"));
        assert!(GAP.contains("kv_dequant"));
        assert!(GAP.contains("cuda-oxide"));
        assert!(GAP.contains("cutile-rs"));
    }
}
