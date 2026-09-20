//! FastVideo MLX affine quantization on CUDA: weight-only INT8 / INT6 / INT4,
//! group 64, activations stay BF16/F32.
//!
//! Matches `mx.quantize(..., mode="affine", group_size=64)` — the recipe in
//! `fastvideo.mlx_runtime.quant_backends` / `MLXQuantizationSpec.from_name`.
//! Groups are consecutive elements along the last dim of `[out, in]`. Dequant
//! is `w = scale * q + bias` with MLX's representable-zero scale flip.
//!
//! Packed layout is MLX's byte packing (same bits as their uint32 little-endian
//! packs for power-of-two widths): INT8 is one byte per element, INT4 two
//! nibbles per byte (low first), INT6 four values in three bytes.

#[cfg(feature = "cuda")]
use super::stats;
use super::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Official MLX H3 group size (`affine_int8_g64` / INT6 / INT4).
pub const GROUP: usize = 64;

/// A `[out, in]` weight stored as affine-quantized codes plus per-group
/// scales and biases. Groups run along `in`.
#[derive(Debug)]
pub struct AffineWeight {
    pub rows: usize,
    pub cols: usize,
    pub bits: u8,
    pub group: usize,
    host: Option<(Vec<u8>, Vec<f32>, Vec<f32>)>,
    #[cfg(feature = "cuda")]
    dev: Option<(
        cudarc::driver::CudaSlice<u8>,
        cudarc::driver::CudaSlice<f32>,
        cudarc::driver::CudaSlice<f32>,
    )>,
}

impl AffineWeight {
    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn is_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.dev.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

/// `FASTVIDEO_H3_AFFINE=int8|int6|int4` (also `8`/`6`/`4`). Empty / off → `None`.
pub fn bits_from_env() -> Option<u8> {
    static FLAG: super::envflag::CachedString = super::envflag::CachedString::new();
    parse_bits(&FLAG.get_or_init(|| super::envflag::string_flag("FASTVIDEO_H3_AFFINE", "")))
}

/// Apply a manifest / CLI choice before the first Linear load. No-op when `bits` is `None`.
pub fn apply_env(bits: Option<u8>) {
    if let Some(b) = bits {
        std::env::set_var("FASTVIDEO_H3_AFFINE", b.to_string());
    }
}

pub fn parse_bits(raw: &str) -> Option<u8> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" | "none" | "no" => None,
        "int8" | "8" | "affine_int8_g64" => Some(8),
        "int6" | "6" => Some(6),
        "int4" | "4" => Some(4),
        _ => None,
    }
}

pub fn packed_bytes(cols: usize, bits: u8) -> Result<usize> {
    match bits {
        8 => Ok(cols),
        4 if cols % 2 == 0 => Ok(cols / 2),
        6 if cols % 4 == 0 => Ok(cols * 3 / 4),
        _ => Err(msg(format!("affine: cols {cols} is not packable at {bits}-bit"))),
    }
}

pub fn n_groups(cols: usize, group: usize) -> Result<usize> {
    if group == 0 || cols % group != 0 {
        return Err(msg(format!("affine: cols {cols} is not divisible by group {group}")));
    }
    Ok(cols / group)
}

/// MLX affine scale/bias for one group, then unsigned codes in `[0, 2^bits-1]`.
pub fn quantize_group(w: &[f32], bits: u8) -> (f32, f32, Vec<u8>) {
    let n_bins = ((1u32 << bits) - 1) as f32;
    let (mut w_min, mut w_max) = (w[0], w[0]);
    for &v in &w[1..] {
        w_min = w_min.min(v);
        w_max = w_max.max(v);
    }
    let mut scale = (w_max - w_min) / n_bins;
    if scale < 1e-7 {
        scale = 1e-7;
    }
    let side = w_min.abs() > w_max.abs();
    if !side {
        scale = -scale;
    }
    let edge = if side { w_min } else { w_max };
    let q0 = (edge / scale).round();
    let at_zero = q0 == 0.0;
    if !at_zero {
        scale = edge / q0;
    }
    let bias = if at_zero { 0.0 } else { edge };
    let codes = w
        .iter()
        .map(|&v| {
            let q = ((v - bias) / scale).round();
            q.clamp(0.0, n_bins) as u8
        })
        .collect();
    (scale, bias, codes)
}

fn pack_row(codes: &[u8], bits: u8) -> Result<Vec<u8>> {
    match bits {
        8 => Ok(codes.to_vec()),
        4 => {
            if codes.len() % 2 != 0 {
                return Err(msg("affine int4: odd number of codes"));
            }
            Ok(codes.chunks_exact(2).map(|c| c[0] | (c[1] << 4)).collect())
        }
        6 => {
            if codes.len() % 4 != 0 {
                return Err(msg("affine int6: codes not a multiple of 4"));
            }
            let mut out = Vec::with_capacity(codes.len() * 3 / 4);
            for c in codes.chunks_exact(4) {
                out.push(c[0] | ((c[1] & 0x03) << 6));
                out.push(((c[1] >> 2) & 0x0f) | ((c[2] & 0x0f) << 4));
                out.push(((c[2] >> 4) & 0x03) | (c[3] << 2));
            }
            Ok(out)
        }
        _ => Err(msg(format!("affine bits {bits}"))),
    }
}

fn unpack_code(packed: &[u8], col: usize, bits: u8) -> u8 {
    match bits {
        8 => packed[col],
        4 => {
            let b = packed[col / 2];
            if col % 2 == 0 { b & 0x0f } else { b >> 4 }
        }
        6 => {
            let w = &packed[(col / 4) * 3..];
            match col % 4 {
                0 => w[0] & 0x3f,
                1 => ((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2),
                2 => ((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4),
                _ => (w[2] >> 2) & 0x3f,
            }
        }
        _ => 0,
    }
}

/// `[rows, cols]` f32 → packed codes and `[rows, cols/group]` scales/biases.
pub fn quantize(w: &[f32], rows: usize, cols: usize, bits: u8) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    quantize_grouped(w, rows, cols, bits, GROUP)
}

pub fn quantize_grouped(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
    group: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    if w.len() != rows * cols {
        return Err(msg(format!("affine quantize: {} values for [{rows}, {cols}]", w.len())));
    }
    let ng = n_groups(cols, group)?;
    let pb = packed_bytes(cols, bits)?;
    let mut codes = vec![0u8; rows * pb];
    let mut scales = vec![0f32; rows * ng];
    let mut biases = vec![0f32; rows * ng];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut unpacked = Vec::with_capacity(cols);
        for g in 0..ng {
            let (s, b, q) = quantize_group(&row[g * group..(g + 1) * group], bits);
            scales[r * ng + g] = s;
            biases[r * ng + g] = b;
            unpacked.extend_from_slice(&q);
        }
        let packed = pack_row(&unpacked, bits)?;
        codes[r * pb..(r + 1) * pb].copy_from_slice(&packed);
    }
    Ok((codes, scales, biases))
}

/// Packed codes → f32 weight (`scale * q + bias` per group).
pub fn dequant(codes: &[u8], scales: &[f32], biases: &[f32], cols: usize, bits: u8) -> Result<Vec<f32>> {
    dequant_grouped(codes, scales, biases, cols, bits, GROUP)
}

pub fn dequant_grouped(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    cols: usize,
    bits: u8,
    group: usize,
) -> Result<Vec<f32>> {
    let pb = packed_bytes(cols, bits)?;
    if pb == 0 || codes.len() % pb != 0 {
        return Err(msg(format!("affine dequant: {} packed bytes, {pb} per row", codes.len())));
    }
    let rows = codes.len() / pb;
    let ng = n_groups(cols, group)?;
    if scales.len() != rows * ng || biases.len() != rows * ng {
        return Err(msg(format!("affine dequant: {} scales for {rows}x{ng}", scales.len())));
    }
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let packed = &codes[r * pb..(r + 1) * pb];
        for c in 0..cols {
            let g = c / group;
            let q = unpack_code(packed, c, bits) as f32;
            out[r * cols + c] = scales[r * ng + g] * q + biases[r * ng + g];
        }
    }
    Ok(out)
}

/// `C = X @ Wᵀ` with `W_ij = scale[i, j/g] * q[i, j] + bias[i, j/g]`.
/// `x` is `[m, k]`, weight is `[n, k]`.
pub fn gemm(
    x: &[f32],
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    m: usize,
    n: usize,
    k: usize,
    bits: u8,
) -> Result<Vec<f32>> {
    gemm_grouped(x, codes, scales, biases, m, n, k, bits, GROUP)
}

pub fn gemm_grouped(
    x: &[f32],
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    m: usize,
    n: usize,
    k: usize,
    bits: u8,
    group: usize,
) -> Result<Vec<f32>> {
    if x.len() != m * k {
        return Err(msg(format!("affine gemm: {} activations for [{m}, {k}]", x.len())));
    }
    let w = dequant_grouped(codes, scales, biases, k, bits, group)?;
    if w.len() != n * k {
        return Err(msg(format!("affine gemm: dequant [{}, {k}] for n={n}", w.len() / k.max(1))));
    }
    use rayon::prelude::*;
    let mut out = vec![0f32; m * n];
    out.par_chunks_mut(n.max(1)).enumerate().for_each(|(i, row)| {
        let xi = &x[i * k..(i + 1) * k];
        for (j, o) in row.iter_mut().enumerate() {
            let wj = &w[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += xi[t] * wj[t];
            }
            *o = acc;
        }
    });
    Ok(out)
}

impl AffineWeight {
    pub fn from_host(w: &[f32], rows: usize, cols: usize, bits: u8) -> Result<Self> {
        let (codes, scales, biases) = quantize(w, rows, cols, bits)?;
        Ok(Self {
            rows,
            cols,
            bits,
            group: GROUP,
            host: Some((codes, scales, biases)),
            #[cfg(feature = "cuda")]
            dev: None,
        })
    }

    pub fn from_packed(
        codes: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        rows: usize,
        cols: usize,
        bits: u8,
    ) -> Result<Self> {
        let pb = packed_bytes(cols, bits)?;
        let ng = n_groups(cols, GROUP)?;
        if codes.len() != rows * pb || scales.len() != rows * ng || biases.len() != rows * ng {
            return Err(msg(format!(
                "affine packed: {} codes / {} scales for [{rows}, {cols}] {bits}-bit g{}",
                codes.len(),
                scales.len(),
                GROUP
            )));
        }
        Ok(Self { rows, cols, bits, group: GROUP, host: Some((codes, scales, biases)), #[cfg(feature = "cuda")] dev: None })
    }

    #[cfg(feature = "cuda")]
    fn upload(&mut self) -> Result<()> {
        if self.dev.is_some() || !stats::device_expected() {
            return Ok(());
        }
        let Some((codes, scales, biases)) = &self.host else {
            return Ok(());
        };
        let dev = super::device::global_device().ok_or_else(|| msg("no device"))?;
        let q = dev.stream.memcpy_stod(codes).map_err(|e| msg(e.to_string()))?;
        let s = dev.stream.memcpy_stod(scales).map_err(|e| msg(e.to_string()))?;
        let b = dev.stream.memcpy_stod(biases).map_err(|e| msg(e.to_string()))?;
        stats::record_h2d(codes.len() + (scales.len() + biases.len()) * 4);
        self.dev = Some((q, s, b));
        self.host = None;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn gemm_device(&self, x: &cudarc::driver::CudaSlice<f32>, m: usize) -> Result<cudarc::driver::CudaSlice<f32>> {
        let (q, s, b) = self.dev.as_ref().ok_or_else(|| msg("affine weight is not on the device"))?;
        super::ops::affine_gemm_device(x, q, s, b, m, self.rows, self.cols, self.bits, self.group)
    }

    pub fn gemm_host(&self, x: &[f32], m: usize) -> Result<Vec<f32>> {
        let (q, s, b) = self.host.as_ref().ok_or_else(|| msg("affine linear: device weight but no device tensor to multiply"))?;
        gemm(x, q, s, b, m, self.rows, self.cols, self.bits)
    }
}

/// Load one `[out, in]` linear as affine. Prefers a pre-quantized
/// `{prefix}.weight` + `.scales` + `.biases` (MLX artifact); otherwise
/// quantizes the float weight on load.
pub fn load(
    map: &super::weights::WeightMap,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    bits: u8,
) -> Result<AffineWeight> {
    load_fused(map, &[prefix], in_dim, out_dim, bits)
}

/// Stack several same-`in` projections (fused QKV / QKVG) as one affine weight.
pub fn load_fused(
    map: &super::weights::WeightMap,
    prefixes: &[&str],
    in_dim: usize,
    out_dim: usize,
    bits: u8,
) -> Result<AffineWeight> {
    let rows = prefixes.len() * out_dim;
    let ng = n_groups(in_dim, GROUP)?;
    let pb = packed_bytes(in_dim, bits)?;
    let mut codes = Vec::with_capacity(rows * pb);
    let mut scales = Vec::with_capacity(rows * ng);
    let mut biases = Vec::with_capacity(rows * ng);
    for prefix in prefixes {
        let (q, s, b) = load_one(map, prefix, in_dim, out_dim, bits)?;
        codes.extend_from_slice(&q);
        scales.extend_from_slice(&s);
        biases.extend_from_slice(&b);
    }
    let aff = {
        #[allow(unused_mut)]
        let mut aff = AffineWeight::from_packed(codes, scales, biases, rows, in_dim, bits)?;
        #[cfg(feature = "cuda")]
        aff.upload()?;
        aff
    };
    Ok(aff)
}

fn load_one(
    map: &super::weights::WeightMap,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    bits: u8,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let key = super::weights::join_key(prefix, "weight");
    let scales_key = format!("{key}.scales");
    if map.has_tensor(&scales_key) {
        return load_packed(map, &key, in_dim, out_dim, bits);
    }
    let wt = super::weights::cuda_tensor_shaped(map, &key, &[out_dim, in_dim])?;
    quantize(&wt.host_cow()?, out_dim, in_dim, bits)
}

fn load_packed(
    map: &super::weights::WeightMap,
    key: &str,
    in_dim: usize,
    out_dim: usize,
    bits: u8,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let pb = packed_bytes(in_dim, bits)?;
    let ng = n_groups(in_dim, GROUP)?;
    let (shape, bytes) = map.get_raw(key)?;
    let expect_elems = out_dim * pb;
    if bytes.len() != expect_elems {
        return Err(msg(format!(
            "key {key}: {} packed bytes (shape {shape:?}) != {out_dim}x{pb} for {bits}-bit",
            bytes.len()
        )));
    }
    let (_, scales) = map.get_f32(&format!("{key}.scales"))?;
    if scales.len() != out_dim * ng {
        return Err(msg(format!("key {key}.scales: {} values != {out_dim}x{ng}", scales.len())));
    }
    let biases = if map.has_tensor(&format!("{key}.biases")) {
        let (_, b) = map.get_f32(&format!("{key}.biases"))?;
        if b.len() != out_dim * ng {
            return Err(msg(format!("key {key}.biases: {} values != {out_dim}x{ng}", b.len())));
        }
        b
    } else {
        vec![0f32; out_dim * ng]
    };
    Ok((bytes, scales, biases))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.37 + seed).sin()) * (1.0 + (i % 11) as f32 * 0.15)).collect()
    }

    #[test]
    fn int8_roundtrip_stays_inside_group_step() {
        let (rows, cols) = (3usize, 128usize);
        let w = seeded(rows * cols, 0.4);
        let (q, s, b) = quantize(&w, rows, cols, 8).unwrap();
        let d = dequant(&q, &s, &b, cols, 8).unwrap();
        for r in 0..rows {
            for g in 0..cols / GROUP {
                let sl = r * cols + g * GROUP;
                let span = w[sl..sl + GROUP].iter().fold(f32::MAX, |a, v| a.min(*v)).abs()
                    + w[sl..sl + GROUP].iter().fold(f32::MIN, |a, v| a.max(*v)).abs();
                let step = (span / 255.0).max(1e-6);
                for c in 0..GROUP {
                    let (orig, got) = (w[sl + c], d[sl + c]);
                    assert!((orig - got).abs() <= step + 1e-5, "r{r} g{g} c{c}: {orig} -> {got} step={step}");
                }
            }
        }
    }

    #[test]
    fn pack_unpack_int4_and_int6() {
        for bits in [4u8, 6, 8] {
            let cols = 64usize;
            let w = seeded(cols, bits as f32);
            let (q, s, b) = quantize(&w, 1, cols, bits).unwrap();
            let d = dequant(&q, &s, &b, cols, bits).unwrap();
            let n_bins = ((1u32 << bits) - 1) as f32;
            let span = w.iter().fold(f32::MAX, |a, v| a.min(*v)).abs() + w.iter().fold(f32::MIN, |a, v| a.max(*v)).abs();
            let step = (span / n_bins).max(1e-6);
            for (i, (orig, got)) in w.iter().zip(&d).enumerate() {
                assert!((orig - got).abs() <= step + 1e-5, "bits={bits} i={i}: {orig} -> {got}");
            }
        }
    }

    #[test]
    fn zero_group_dequants_to_zero() {
        let (q, s, b) = quantize(&[0.0; 64], 1, 64, 8).unwrap();
        let d = dequant(&q, &s, &b, 64, 8).unwrap();
        for v in d {
            assert!(v.abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn mlx_representable_zero_prefers_the_far_edge() {
        // |min| < |max| → scale is flipped so a representable code hits the
        // far edge exactly (MLX CUDA affine_quantize).
        let mut w = vec![0.0f32; 64];
        w[0] = -1.0;
        w[1] = 3.0;
        let (scale, bias, codes) = quantize_group(&w, 8);
        assert!(scale < 0.0, "scale should flip when |min| <= |max|: {scale}");
        assert!((bias - 3.0).abs() < 1e-6);
        assert_eq!(codes[1], 0); // far edge sits on code 0 after the flip
        let restored_min = scale * codes[0] as f32 + bias;
        let restored_max = scale * codes[1] as f32 + bias;
        assert!((restored_max - 3.0).abs() < 1e-5);
        assert!((restored_min + 1.0).abs() < 0.05);
    }

    #[test]
    fn gemm_matches_dequant_then_matmul() {
        let (m, n, k) = (5usize, 4usize, 64usize);
        let w = seeded(n * k, 1.1);
        let x = seeded(m * k, 0.2);
        let (q, s, b) = quantize(&w, n, k, 8).unwrap();
        let wd = dequant(&q, &s, &b, k, 8).unwrap();
        let got = gemm(&x, &q, &s, &b, m, n, k, 8).unwrap();
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|t| x[i * k + t] * wd[j * k + t]).sum();
                let g = got[i * n + j];
                assert!((g - want).abs() <= 1e-4 * want.abs().max(1.0), "m{i} n{j}: {g} vs {want}");
            }
        }
    }

    #[test]
    fn parse_bits_names() {
        assert_eq!(parse_bits("int8"), Some(8));
        assert_eq!(parse_bits("INT6"), Some(6));
        assert_eq!(parse_bits("4"), Some(4));
        assert_eq!(parse_bits(""), None);
        assert_eq!(parse_bits("off"), None);
        assert_eq!(parse_bits("nvfp4"), None);
    }
}
