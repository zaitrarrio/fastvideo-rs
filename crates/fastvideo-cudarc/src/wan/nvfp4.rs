//! LongLive NVFP4 seam. `FASTVIDEO_NVFP4` dequants beforehand: weights once
//! at load inside `Linear`, activations and K/V on each forward here. The
//! existing GEMM and attention then consume those dense tensors. Host math
//! lives in [`fastvideo_models::nvfp4`].

use fastvideo_models::nvfp4::{self, Nvfp4Tensor, ScaleRule, BLOCK};

#[cfg(feature = "cuda")]
use super::stats;
use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// A `[rows, cols]` weight stored as packed E2M1 + E4M3 block scales + per-row
/// decode `amax / (e2m1_max * e4m3_max)`. The FP32 matrix is never the resident
/// form.
#[derive(Debug)]
pub struct Nvfp4Weight {
    pub rows: usize,
    pub cols: usize,
    pub rule: ScaleRule,
    host: Option<HostPack>,
    #[cfg(feature = "cuda")]
    dev: Option<DevPack>,
}

#[derive(Debug, Clone)]
struct HostPack {
    packed: Vec<u8>,
    scales: Vec<u8>,
    decode: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct DevPack {
    packed: cudarc::driver::CudaSlice<u8>,
    scales: cudarc::driver::CudaSlice<u8>,
    decode: cudarc::driver::CudaSlice<f32>,
}

impl Nvfp4Weight {
    pub fn from_host(w: &[f32], rows: usize, cols: usize, rule: ScaleRule) -> Result<Self> {
        let qt = nvfp4::quantize(w, rows, cols, rule).map_err(msg)?;
        Ok(Self::from_tensor(&qt))
    }

    pub fn from_prefixes(chunks: &[(Vec<f32>, usize, usize)], rule: ScaleRule) -> Result<Self> {
        if chunks.is_empty() {
            return Err(msg("nvfp4: no prefixes"));
        }
        let cols = chunks[0].2;
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut decode = Vec::new();
        let mut rows = 0;
        for (host, r, c) in chunks {
            if *c != cols {
                return Err(msg(format!("nvfp4 fused: cols {c} != {cols}")));
            }
            let qt = nvfp4::quantize(host, *r, *c, rule).map_err(msg)?;
            let factor = nvfp4::dequant_factor(qt.amax, rule);
            packed.extend_from_slice(&qt.packed);
            scales.extend_from_slice(&qt.scales);
            decode.extend(std::iter::repeat_n(factor, *r));
            rows += *r;
        }
        Ok(Self {
            rows,
            cols,
            rule,
            host: Some(HostPack {
                packed,
                scales,
                decode,
            }),
            #[cfg(feature = "cuda")]
            dev: None,
        })
    }

    fn from_tensor(qt: &Nvfp4Tensor) -> Self {
        let factor = nvfp4::dequant_factor(qt.amax, qt.rule);
        Self {
            rows: qt.rows,
            cols: qt.cols,
            rule: qt.rule,
            host: Some(HostPack {
                packed: qt.packed.clone(),
                scales: qt.scales.clone(),
                decode: vec![factor; qt.rows],
            }),
            #[cfg(feature = "cuda")]
            dev: None,
        }
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

    #[cfg(feature = "cuda")]
    pub fn upload(&mut self) -> Result<()> {
        if self.dev.is_some() || !stats::device_expected() {
            return Ok(());
        }
        let Some(h) = &self.host else {
            return Ok(());
        };
        let Some(dev) = super::device::global_device() else {
            return Ok(());
        };
        let packed = dev
            .stream
            .memcpy_stod(&h.packed)
            .map_err(|e| msg(e.to_string()))?;
        let scales = dev
            .stream
            .memcpy_stod(&h.scales)
            .map_err(|e| msg(e.to_string()))?;
        let decode = dev
            .stream
            .memcpy_stod(&h.decode)
            .map_err(|e| msg(e.to_string()))?;
        stats::record_h2d(h.packed.len() + h.scales.len() + h.decode.len() * 4);
        self.dev = Some(DevPack {
            packed,
            scales,
            decode,
        });
        self.host = None;
        Ok(())
    }

    pub fn gemm_host(&self, x: &[f32], m: usize) -> Result<Vec<f32>> {
        let h = self
            .host
            .as_ref()
            .ok_or_else(|| msg("nvfp4 linear: device weight but no host pack"))?;
        if x.len() != m * self.cols {
            return Err(msg(format!(
                "nvfp4 gemm: {} activations for [{m}, {}]",
                x.len(),
                self.cols
            )));
        }
        let a = nvfp4::quantize(x, m, self.cols, self.rule).map_err(msg)?;
        // Per-row decode can differ across fused prefixes.
        Ok(gemm_rows(
            &a, &h.packed, &h.scales, &h.decode, self.rows, self.cols,
        ))
    }

    #[cfg(feature = "cuda")]
    pub fn gemm_device(
        &self,
        x: &cudarc::driver::CudaSlice<f32>,
        m: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        let d = self
            .dev
            .as_ref()
            .ok_or_else(|| msg("nvfp4 weight is not on the device"))?;
        if x.len() != m * self.cols {
            return Err(msg(format!(
                "nvfp4 gemm: {} activations for [{m}, {}]",
                x.len(),
                self.cols
            )));
        }
        // Activations are packed on the host (same quantize as the CPU path)
        // then uploaded; the GEMM kernel dequants both sides in-tile.
        let host = {
            let mut buf = vec![0.0f32; x.len()];
            super::device::global_device()
                .ok_or_else(|| msg("no device"))?
                .stream
                .memcpy_dtov(x)
                .map_err(|e| msg(e.to_string()))
                .and_then(|v| {
                    if v.len() != buf.len() {
                        return Err(msg("nvfp4 activation download size"));
                    }
                    buf = v;
                    Ok(buf)
                })?
        };
        let a = nvfp4::quantize(&host, m, self.cols, self.rule).map_err(msg)?;
        let a_decode = nvfp4::dequant_factor(a.amax, a.rule);
        super::ops::nvfp4_gemm_device(
            &a.packed, &a.scales, a_decode, &d.packed, &d.scales, &d.decode, m, self.rows,
            self.cols,
        )
    }
}

/// On-the-fly W4A4: each output row uses its own baked decode factor.
fn gemm_rows(
    a: &Nvfp4Tensor,
    w_packed: &[u8],
    w_scales: &[u8],
    w_decode: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let m = a.rows;
    let n_blocks = k / BLOCK;
    let packed_cols = k / 2;
    let a_dec = nvfp4::dequant_factor(a.amax, a.rule);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            let w_dec = w_decode[j];
            for b in 0..n_blocks {
                let sa = fastvideo_ops::fp8::e4m3_to_f32(a.scales[i * n_blocks + b]) * a_dec;
                let sw = fastvideo_ops::fp8::e4m3_to_f32(w_scales[j * n_blocks + b]) * w_dec;
                let ap = i * packed_cols + b * (BLOCK / 2);
                let wp = j * packed_cols + b * (BLOCK / 2);
                for t in 0..BLOCK / 2 {
                    let ab = a.packed[ap + t];
                    let wb = w_packed[wp + t];
                    acc += nvfp4::e2m1_to_f32(ab & 0x0F) * sa * nvfp4::e2m1_to_f32(wb & 0x0F) * sw;
                    acc += nvfp4::e2m1_to_f32(ab >> 4) * sa * nvfp4::e2m1_to_f32(wb >> 4) * sw;
                }
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Expand a block-aligned activation (or K/V row-major tensor) to FP32
/// before the existing GEMM or attention. Last dim must be a multiple of 16.
pub fn dequant_beforehand(xs: &CudaTensor, rule: ScaleRule) -> Result<CudaTensor> {
    let k = *xs
        .shape
        .last()
        .ok_or_else(|| msg("nvfp4 dequant on a scalar"))?;
    if k == 0 || !k.is_multiple_of(BLOCK) {
        return Err(msg(format!(
            "nvfp4 dequant: last dim {k} is not a multiple of {BLOCK}"
        )));
    }
    let rows = xs.numel() / k;
    let rec = nvfp4::reconstruct(&xs.host_cow()?, rows, k, rule).map_err(msg)?;
    let mut out = CudaTensor::from_vec(rec, xs.shape.clone())?;
    out.pin_device()?;
    Ok(out)
}

/// Quantize K (after `k_smooth`) and V, dequant into the tensors attention
/// already consumes. `None` when the flag is off or the last dim is not
/// block-aligned — caller keeps the dense path.
pub fn kv_for_attention(
    k: &CudaTensor,
    v: &CudaTensor,
) -> Result<Option<(CudaTensor, CudaTensor)>> {
    let Some(rule) = nvfp4::from_env() else {
        return Ok(None);
    };
    let k_out = dequant_bhsd(k, rule, true)?;
    let v_out = dequant_bhsd(v, rule, false)?;
    match (k_out, v_out) {
        (Some(k), Some(v)) => Ok(Some((k, v))),
        _ => Ok(None),
    }
}

fn dequant_bhsd(t: &CudaTensor, rule: ScaleRule, smooth: bool) -> Result<Option<CudaTensor>> {
    if t.rank() != 4 {
        return Ok(None);
    }
    let d = t.shape[3];
    if d == 0 || !d.is_multiple_of(BLOCK) {
        return Ok(None);
    }
    let rows = t.numel() / d;
    let mut host = t.host_cow()?.into_owned();
    if smooth {
        nvfp4::k_smooth(&mut host, rows, d);
    }
    let smoothed = CudaTensor::from_vec(host, t.shape.clone())?;
    dequant_beforehand(&smoothed, rule).map(Some)
}

/// Apply [`kv_for_attention`] when the flag is on; otherwise the inputs.
pub fn maybe_kv(k: CudaTensor, v: CudaTensor) -> Result<(CudaTensor, CudaTensor)> {
    Ok(match kv_for_attention(&k, &v)? {
        Some(pair) => pair,
        None => (k, v),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_gemm_matches_reconstructed_matmul() {
        let (m, n, k) = (2usize, 3usize, 16usize);
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.29).sin() * 2.5)
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.13).cos() * 3.1)
            .collect();
        let wt = Nvfp4Weight::from_host(&w, n, k, ScaleRule::Mse).unwrap();
        let got = wt.gemm_host(&x, m).unwrap();
        let rec_w = nvfp4::reconstruct(&w, n, k, ScaleRule::Mse).unwrap();
        let rec_x = nvfp4::reconstruct(&x, m, k, ScaleRule::Mse).unwrap();
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|t| rec_x[i * k + t] * rec_w[j * k + t]).sum();
                let g = got[i * n + j];
                assert!(
                    (g - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "C[{i},{j}] {g} vs {want}"
                );
            }
        }
        assert_eq!(wt.host.as_ref().unwrap().packed.len(), n * (k / 2));
    }

    #[test]
    fn kv_off_is_none() {
        nvfp4::with_env(None, || {
            let k = CudaTensor::from_vec(vec![1.0; 32], vec![1, 1, 2, 16]).unwrap();
            let v = CudaTensor::from_vec(vec![2.0; 32], vec![1, 1, 2, 16]).unwrap();
            assert!(kv_for_attention(&k, &v).unwrap().is_none());
        });
    }

    #[test]
    fn kv_on_k_smooths_then_fake_quants() {
        let (gk, gv, raw_k, raw_v) = nvfp4::with_env(Some("1"), || {
            let raw_k: Vec<f32> = (0..32).map(|i| i as f32 * 0.25).collect();
            let raw_v: Vec<f32> = (0..32).map(|i| (i as f32 * 0.11).sin()).collect();
            let k = CudaTensor::from_vec(raw_k.clone(), vec![1, 1, 2, 16]).unwrap();
            let v = CudaTensor::from_vec(raw_v.clone(), vec![1, 1, 2, 16]).unwrap();
            let (gk, gv) = kv_for_attention(&k, &v).unwrap().expect("flag on");
            (gk, gv, raw_k, raw_v)
        });
        let mut sk = raw_k;
        nvfp4::k_smooth(&mut sk, 2, 16);
        let want_k = nvfp4::reconstruct(&sk, 2, 16, ScaleRule::Mse).unwrap();
        let want_v = nvfp4::reconstruct(&raw_v, 2, 16, ScaleRule::Mse).unwrap();
        let got_k = gk.host_cow().unwrap();
        let got_v = gv.host_cow().unwrap();
        for (a, b) in want_k.iter().zip(got_k.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        for (a, b) in want_v.iter().zip(got_v.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
