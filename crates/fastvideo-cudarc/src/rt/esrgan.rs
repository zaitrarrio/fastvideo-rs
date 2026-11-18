//! ×2 / ×3 / ×4 pixel upscaler for finished clips: Real-ESRGAN's compact
//! network (SRVGGNetCompact, checkpoint `realesr-animevideov3`, BSD-3,
//! 2.5 MB), per frame. Ported from fastwan-rs's `upscale.rs` and h3-rs's
//! `esrgan.rs` / `finish.rs` (`docs/ports/upscaler.md`).
//!
//! conv3×3 3→F, PReLU → `convs` × (conv3×3 F→F, PReLU) → conv3×3 F→3·s²
//! → pixel-shuffle s, plus the input upsampled ×s (nearest), clamp, round.
//!
//! On the device every conv goes through [`super::conv::cudnn_conv_fma`]
//! (FMA math whatever `FASTVIDEO_TF32` says: under the TF32 default 7 % of a
//! real ×4 frame's bytes came out one level off the host reference) with
//! the fused `bias_prelu_inplace` epilogue (the last conv bias-only), then
//! `esrgan_pack_rgb8` does the pixel shuffle, the residual add from the exact
//! 8-bit input, the clamp and the rounding and writes interleaved RGB8 on
//! the device, so a frame comes down as 3 bytes per output pixel. ×2 is the
//! same ×4 network, its clamped ×4 picture box-filtered 2×2 before the
//! rounding, in that kernel (`down = 2`: fastwan-rs
//! found no fast ×2 model better than that, and the network is the cost
//! either way); ×3 is the ×4 frames Lanczos-resampled on the host, like
//! Real-ESRGAN's `--outscale` (h3-rs's rule). Frames run in batches sized by
//! a device-memory budget ([`Esrgan::batch_frames`]).
//!
//! The plain-Rust path ([`Esrgan::upscale_rgb8_cpu`]) is the reference the
//! device path and `fv-gpucheck kernels --groups esrgan` are checked
//! against, and what runs when no device is live. Its convs accumulate in
//! f64 and store f32 per layer, so an independent numpy SRVGGNetCompact
//! (`scripts/ref/esrgan_parity.py`) reproduces it byte for byte.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;

use super::ops::host;
use crate::wan::pipeline::{PipelineError, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// The pinned checkpoint (`scripts/gpu/fetch-tae.sh`, `weights-manifest.tsv`
/// `auxiliary/upscalers/`).
pub const ESRGAN_FILE: &str = "realesr-animevideov3.pth";
pub const ESRGAN_URL: &str =
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth";
pub const ESRGAN_SHA256: &str = "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d";

/// Device-memory budget for one batch's activations (`FASTVIDEO_ESRGAN_BUDGET_GIB`
/// overrides; default 2 GiB, far under what any served card has beside a
/// resident pipeline).
pub const DEFAULT_BUDGET_BYTES: u64 = 2 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EsrganConfig {
    /// Feature channels.
    pub feat: usize,
    /// Body convs after the first one.
    pub convs: usize,
    /// The network's own factor (the pixel-shuffle).
    pub scale: usize,
}

impl EsrganConfig {
    /// `realesr-animevideov3`: 64 features, 16 body convs, ×4.
    pub fn anime_video_v3() -> Self {
        Self {
            feat: 64,
            convs: 16,
            scale: 4,
        }
    }

    /// A tiny shape for tests and parity cases.
    pub fn tiny() -> Self {
        Self {
            feat: 8,
            convs: 2,
            scale: 2,
        }
    }

    /// `(cin, cout, prelu)` per conv, in order.
    fn layers(&self) -> Vec<(usize, usize, bool)> {
        let mut v = vec![(3, self.feat, true)];
        v.extend((0..self.convs).map(|_| (self.feat, self.feat, true)));
        v.push((self.feat, 3 * self.scale * self.scale, false));
        v
    }
}

/// Checkpoint keys and shapes (the `nn.Sequential` positions of `body`:
/// conv at `2i`, its PReLU at `2i + 1`).
pub fn esrgan_manifest(cfg: &EsrganConfig) -> Vec<(String, Vec<usize>)> {
    let mut v = Vec::new();
    for (i, (cin, cout, prelu)) in cfg.layers().into_iter().enumerate() {
        v.push((format!("body.{}.weight", 2 * i), vec![cout, cin, 3, 3]));
        v.push((format!("body.{}.bias", 2 * i), vec![cout]));
        if prelu {
            v.push((format!("body.{}.weight", 2 * i + 1), vec![cout]));
        }
    }
    v
}

struct Layer {
    cin: usize,
    cout: usize,
    /// `[cout, cin, 3, 3]`.
    w: Vec<f32>,
    b: Vec<f32>,
    /// PReLU slope per channel; `None` for the last conv.
    slope: Option<Vec<f32>>,
}

#[cfg(feature = "cuda")]
struct DeviceLayer {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    slope: Option<CudaSlice<f32>>,
}

pub struct Esrgan {
    pub cfg: EsrganConfig,
    layers: Vec<Layer>,
    /// Uploaded on first device use.
    #[cfg(feature = "cuda")]
    device: OnceLock<Vec<DeviceLayer>>,
}

impl Esrgan {
    /// Weights from `get(key, shape)`, every manifest key required.
    pub fn from_source(
        cfg: EsrganConfig,
        mut get: impl FnMut(&str, &[usize]) -> Result<Vec<f32>>,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for (i, (cin, cout, prelu)) in cfg.layers().into_iter().enumerate() {
            let w = get(&format!("body.{}.weight", 2 * i), &[cout, cin, 3, 3])?;
            let b = get(&format!("body.{}.bias", 2 * i), &[cout])?;
            let slope = if prelu {
                Some(get(&format!("body.{}.weight", 2 * i + 1), &[cout])?)
            } else {
                None
            };
            layers.push(Layer {
                cin,
                cout,
                w,
                b,
                slope,
            });
        }
        Ok(Self {
            cfg,
            layers,
            #[cfg(feature = "cuda")]
            device: OnceLock::new(),
        })
    }

    /// Deterministic pseudo-random weights (tests, parity cases): small
    /// enough that 19 layers neither blow up nor vanish.
    pub fn synthetic(cfg: EsrganConfig, seed: u64) -> Self {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            // xorshift64*, mapped to (-1, 1).
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            (v >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        Self::from_source(cfg, |key, shape| {
            let n: usize = shape.iter().product();
            let fan_in = if shape.len() == 4 { shape[1] * 9 } else { 1 };
            let amp = if key.ends_with(".bias") {
                0.05
            } else if shape.len() == 1 {
                0.25
            } else {
                1.0 / (fan_in as f64).sqrt()
            };
            Ok((0..n).map(|_| (next() * amp) as f32).collect())
        })
        .expect("synthetic weights")
    }

    /// `realesr-animevideov3` from a `.pth` (torch zip, the state dict under
    /// `params_ema` / `params`, read by `fastvideo_loader::pth`, no Python)
    /// or a safetensors conversion with the same keys. A missing key is an
    /// error, never a default.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with(path, EsrganConfig::anime_video_v3())
    }

    pub fn load_with(path: &Path, cfg: EsrganConfig) -> Result<Self> {
        let ctx = |e: String| msg(format!("upscaler weights {}: {e}", path.display()));
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let check = |key: &str, shape: &[usize], got: Vec<usize>, data: Vec<f32>| {
            if got != shape {
                return Err(ctx(format!("{key}: shape {got:?}, expected {shape:?}")));
            }
            Ok(data)
        };
        if ext == "pth" || ext == "pt" {
            let wanted: std::collections::BTreeSet<String> =
                esrgan_manifest(&cfg).into_iter().map(|(k, _)| k).collect();
            let mut tensors =
                fastvideo_loader::pth::read_pth_nested(path, &["params_ema", "params"], |k| {
                    wanted.contains(k)
                })
                .map_err(|e| ctx(e.to_string()))?;
            Self::from_source(cfg, |key, shape| {
                let t = tensors
                    .remove(key)
                    .ok_or_else(|| ctx(format!("missing key {key}")))?;
                check(key, shape, t.shape, t.data)
            })
        } else {
            let map = super::weights::WeightMap::open_files(&[path.to_path_buf()])
                .map_err(|e| ctx(e.to_string()))?;
            Self::from_source(cfg, |key, shape| {
                if !map.contains(key) {
                    return Err(ctx(format!("missing key {key}")));
                }
                let (got, data) = map.get_f32(key).map_err(|e| ctx(e.to_string()))?;
                check(key, shape, got, data)
            })
        }
    }

    /// Bytes one frame of `h`×`w` needs on the device at the network's
    /// widest point: the f32 NCHW input and two `feat`-wide activations
    /// (cuDNN writes a fresh output while the previous one is still alive),
    /// the last conv's `3·s²` planes, plus the 8-bit input and output.
    pub fn frame_bytes(&self, h: usize, w: usize, factor: usize) -> u64 {
        let (s, plane) = (self.cfg.scale, (h * w) as u64);
        let f32s = (3 + 2 * self.cfg.feat + 3 * s * s) as u64;
        let out = if s.is_multiple_of(factor) {
            plane * (factor * factor) as u64
        } else {
            plane * (s * s) as u64
        };
        plane * f32s * 4 + plane * 3 + out * 3
    }

    /// Frames per batch so one batch's activations stay within `budget`
    /// bytes; at least 1, at most `frames`.
    pub fn batch_frames(
        &self,
        frames: usize,
        h: usize,
        w: usize,
        factor: usize,
        budget: u64,
    ) -> usize {
        let per = self.frame_bytes(h, w, factor).max(1);
        ((budget / per).max(1) as usize).min(frames.max(1))
    }

    /// `FASTVIDEO_ESRGAN_BUDGET_GIB`, else [`DEFAULT_BUDGET_BYTES`].
    pub fn budget_bytes() -> u64 {
        fastvideo_models::settings::env_var("FASTVIDEO_ESRGAN_BUDGET_GIB")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|g| *g > 0.0)
            .map_or(DEFAULT_BUDGET_BYTES, |g| (g * (1u64 << 30) as f64) as u64)
    }

    /// The whole network on the host: `x` is `[n, 3, h, w]` in `[0, 1]`;
    /// the result the last conv's `[n, 3·s², h, w]` (bias added, before the
    /// pixel shuffle). Each conv accumulates in f64 and stores f32; the PReLU
    /// is f32.
    pub fn network_cpu(&self, x: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
        assert_eq!(x.len(), n * 3 * h * w);
        let mut cur = x.to_vec();
        for l in &self.layers {
            let mut y = conv3x3_cpu(&cur, n, l.cin, h, w, &l.w, &l.b, l.cout);
            if let Some(slope) = &l.slope {
                prelu_cpu(&mut y, slope, h * w);
            }
            cur = y;
        }
        cur
    }

    /// [`Esrgan::upscale_rgb8`] on the host, whatever device is live.
    pub fn upscale_rgb8_cpu(
        &self,
        rgb: &[u8],
        frames: usize,
        h: usize,
        w: usize,
        factor: usize,
        on_batch: &mut dyn FnMut(usize, usize, Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        let plan = self.plan(rgb, frames, h, w, factor)?;
        // Host batches: bounded by memory too, in the same units.
        let batch = self.batch_frames(frames, h, w, factor, Self::budget_bytes());
        let mut f0 = 0;
        while f0 < frames {
            let n = batch.min(frames - f0);
            let src = &rgb[f0 * h * w * 3..(f0 + n) * h * w * 3];
            let x = planar_unit(src, n, h, w);
            let y = self.network_cpu(&x, n, h, w);
            let packed = host::esrgan_pack_rgb8(&y, src, n, h, w, self.cfg.scale, plan.down);
            on_batch(f0, n, plan.finish(packed, n))?;
            f0 += n;
        }
        Ok(())
    }

    /// Upscale `frames` interleaved 8-bit RGB frames (`[frames, h, w, 3]`) by
    /// `factor` (2 ..= the network's scale) on the live device, or on the
    /// host when none is; `on_batch(first, n, rgb)` receives each batch's
    /// `n` frames at `[n, h·factor, w·factor, 3]`, in order, so a caller can
    /// write them without holding the whole clip.
    pub fn upscale_rgb8(
        &self,
        rgb: &[u8],
        frames: usize,
        h: usize,
        w: usize,
        factor: usize,
        on_batch: &mut dyn FnMut(usize, usize, Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if super::device::global_device().is_some() {
            return self.upscale_rgb8_device(rgb, frames, h, w, factor, on_batch);
        }
        self.upscale_rgb8_cpu(rgb, frames, h, w, factor, on_batch)
    }

    fn plan(&self, rgb: &[u8], frames: usize, h: usize, w: usize, factor: usize) -> Result<Plan> {
        let s = self.cfg.scale;
        if !(2..=s).contains(&factor) {
            return Err(msg(format!(
                "upscale ×{factor}: the factor is 2 to {s} (the network's)"
            )));
        }
        if rgb.len() != frames * h * w * 3 || h == 0 || w == 0 {
            return Err(msg(format!(
                "upscale: {} bytes is not [{frames}, {h}, {w}, 3]",
                rgb.len()
            )));
        }
        Ok(Plan {
            s,
            h,
            w,
            factor,
            down: if s.is_multiple_of(factor) {
                s / factor
            } else {
                1
            },
        })
    }

    #[cfg(feature = "cuda")]
    fn device_layers(&self) -> Result<&[DeviceLayer]> {
        if let Some(d) = self.device.get() {
            return Ok(d);
        }
        let dev = super::device::global_device().ok_or_else(|| msg("no global CUDA device"))?;
        let up = |v: &[f32]| dev.stream.memcpy_stod(v).map_err(|e| msg(e.to_string()));
        let mut out = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            out.push(DeviceLayer {
                w: up(&l.w)?,
                b: up(&l.b)?,
                slope: l.slope.as_deref().map(up).transpose()?,
            });
        }
        Ok(self.device.get_or_init(|| out))
    }

    /// The network on the device: `x` `[n, 3, h, w]` → the last conv's
    /// `[n, 3·s², h, w]` (bias added).
    #[cfg(feature = "cuda")]
    pub fn network_device(
        &self,
        x: CudaSlice<f32>,
        n: usize,
        h: usize,
        w: usize,
    ) -> Result<CudaSlice<f32>> {
        let layers = self.device_layers()?;
        let plane = h * w;
        let mut cur = x;
        for (l, d) in self.layers.iter().zip(layers) {
            let (mut y, _) = super::conv::cudnn_conv_fma(
                &cur,
                &[n, l.cin, h, w],
                &d.w,
                &[l.cout, l.cin, 3, 3],
                &[1, 1],
                &[1, 1],
            )
            .map_err(|e| msg(e.to_string()))?;
            drop(cur);
            match &d.slope {
                Some(slope) => super::ops::bias_prelu_inplace_device(&mut y, &d.b, slope, plane)?,
                None => super::ops::add_bias_inplace_device(&mut y, &d.b, plane)?,
            }
            cur = y;
        }
        Ok(cur)
    }

    #[cfg(feature = "cuda")]
    fn upscale_rgb8_device(
        &self,
        rgb: &[u8],
        frames: usize,
        h: usize,
        w: usize,
        factor: usize,
        on_batch: &mut dyn FnMut(usize, usize, Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        let plan = self.plan(rgb, frames, h, w, factor)?;
        let dev = super::device::global_device().ok_or_else(|| msg("no global CUDA device"))?;
        let batch = self.batch_frames(frames, h, w, factor, Self::budget_bytes());
        let mut f0 = 0;
        while f0 < frames {
            let n = batch.min(frames - f0);
            let src = &rgb[f0 * h * w * 3..(f0 + n) * h * w * 3];
            let x = planar_unit(src, n, h, w);
            let dx = dev.stream.memcpy_stod(&x).map_err(|e| msg(e.to_string()))?;
            let base = dev
                .stream
                .memcpy_stod(src)
                .map_err(|e| msg(e.to_string()))?;
            drop(x);
            let y = self.network_device(dx, n, h, w)?;
            let packed =
                super::ops::esrgan_pack_rgb8_device(&y, &base, n, h, w, plan.s, plan.down)?;
            drop((y, base));
            on_batch(f0, n, plan.finish(packed, n))?;
            f0 += n;
        }
        Ok(())
    }
}

/// How one request's factor is reached from the network's scale.
struct Plan {
    s: usize,
    h: usize,
    w: usize,
    factor: usize,
    /// Box filter in the pack kernel (`s / factor` when that divides, else 1
    /// and the ×s frames are resampled).
    down: usize,
}

impl Plan {
    /// The packed batch at the requested factor: as packed when the box
    /// filter reached it, else Lanczos-resampled from ×s.
    fn finish(&self, packed: Vec<u8>, n: usize) -> Vec<u8> {
        if self.s / self.down == self.factor {
            return packed;
        }
        let (sw, sh) = (self.w * self.s, self.h * self.s);
        let (tw, th) = (self.w * self.factor, self.h * self.factor);
        let mut out = vec![0u8; n * tw * th * 3];
        out.par_chunks_mut(tw * th * 3)
            .zip(packed.par_chunks(sw * sh * 3))
            .for_each(|(dst, src)| dst.copy_from_slice(&resize_lanczos3(src, sw, sh, tw, th)));
        out
    }
}

/// `[n, h, w, 3]` bytes → `[n, 3, h, w]` f32 in `[0, 1]` (`b / 255`, one
/// correctly rounded f32 division, as `astype(float32) / 255.0`).
pub fn planar_unit(rgb: &[u8], n: usize, h: usize, w: usize) -> Vec<f32> {
    let plane = h * w;
    assert_eq!(rgb.len(), n * plane * 3);
    let mut x = vec![0f32; n * 3 * plane];
    x.par_chunks_mut(plane).enumerate().for_each(|(k, dst)| {
        let (f, c) = (k / 3, k % 3);
        let src = &rgb[f * plane * 3..(f + 1) * plane * 3];
        for (p, v) in dst.iter_mut().enumerate() {
            *v = f32::from(src[p * 3 + c]) / 255.0;
        }
    });
    x
}

/// 3×3 cross-correlation, padding 1, `x` `[n, cin, h, w]`, `wgt`
/// `[cout, cin, 3, 3]`, bias added; f64 accumulation, f32 result.
#[allow(clippy::too_many_arguments)]
fn conv3x3_cpu(
    x: &[f32],
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    wgt: &[f32],
    bias: &[f32],
    cout: usize,
) -> Vec<f32> {
    let plane = h * w;
    assert_eq!(x.len(), n * cin * plane);
    assert_eq!(wgt.len(), cout * cin * 9);
    assert_eq!(bias.len(), cout);
    let mut out = vec![0f32; n * cout * plane];
    out.par_chunks_mut(w).enumerate().for_each(|(row, o)| {
        let (f, rem) = (row / (cout * h), row % (cout * h));
        let (oc, y) = (rem / h, rem % h);
        let xf = &x[f * cin * plane..(f + 1) * cin * plane];
        let wk = &wgt[oc * cin * 9..(oc + 1) * cin * 9];
        for (xx, o) in o.iter_mut().enumerate() {
            let mut acc = f64::from(bias[oc]);
            for ic in 0..cin {
                let xp = &xf[ic * plane..(ic + 1) * plane];
                let wp = &wk[ic * 9..(ic + 1) * 9];
                for ky in 0..3 {
                    let sy = (y + ky).wrapping_sub(1);
                    if sy >= h {
                        continue;
                    }
                    for kx in 0..3 {
                        let sx = (xx + kx).wrapping_sub(1);
                        if sx >= w {
                            continue;
                        }
                        acc += f64::from(xp[sy * w + sx]) * f64::from(wp[ky * 3 + kx]);
                    }
                }
            }
            *o = acc as f32;
        }
    });
    out
}

/// PReLU over `[n, c, inner]` with a slope per channel (f32, as torch).
fn prelu_cpu(x: &mut [f32], slope: &[f32], inner: usize) {
    x.par_chunks_mut(inner).enumerate().for_each(|(k, chunk)| {
        let a = slope[k % slope.len()];
        for v in chunk {
            if *v < 0.0 {
                *v *= a;
            }
        }
    });
}

// ---- Lanczos resampling (×3 from the ×4 frames) ---------------------------

fn lanczos3(x: f32) -> f32 {
    const A: f32 = 3.0;
    if x == 0.0 {
        return 1.0;
    }
    if x.abs() >= A {
        return 0.0;
    }
    let px = std::f32::consts::PI * x;
    A * px.sin() * (px / A).sin() / (px * px)
}

/// The taps of one output position along an axis of `src_len` → `dst_len`:
/// `(first source index, weights)`, normalized; the support widens by the
/// scale when shrinking (an antialiased resample).
fn lanczos_taps(src_len: usize, dst_len: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src_len as f32 / dst_len as f32;
    let widen = scale.max(1.0);
    let support = 3.0 * widen;
    (0..dst_len)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale - 0.5;
            let lo = ((center - support).floor().max(0.0)) as usize;
            let hi = ((center + support).ceil() as usize).min(src_len - 1);
            let mut ws: Vec<f32> = (lo..=hi)
                .map(|j| lanczos3((j as f32 - center) / widen))
                .collect();
            let sum: f32 = ws.iter().sum();
            if sum != 0.0 {
                ws.iter_mut().for_each(|v| *v /= sum);
            }
            (lo, ws)
        })
        .collect()
}

/// Lanczos-3 resample of one interleaved RGB8 frame `w`×`h` to `ow`×`oh`
/// (separable, f32 between the passes, rounded once at the end; row
/// parallel).
pub fn resize_lanczos3(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    assert_eq!(src.len(), w * h * 3);
    assert!(w > 0 && h > 0 && ow > 0 && oh > 0);
    if (ow, oh) == (w, h) {
        return src.to_vec();
    }
    let xt = lanczos_taps(w, ow);
    let yt = lanczos_taps(h, oh);
    // Horizontal pass: [h, ow, 3] f32.
    let mut mid = vec![0f32; h * ow * 3];
    mid.par_chunks_mut(ow * 3).enumerate().for_each(|(y, row)| {
        let s = &src[y * w * 3..(y + 1) * w * 3];
        for (x, (lo, ws)) in xt.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, wt) in ws.iter().enumerate() {
                let p = (lo + k) * 3;
                for c in 0..3 {
                    acc[c] += wt * f32::from(s[p + c]);
                }
            }
            row[x * 3..x * 3 + 3].copy_from_slice(&acc);
        }
    });
    // Vertical pass: [oh, ow, 3] u8.
    let mut out = vec![0u8; oh * ow * 3];
    out.par_chunks_mut(ow * 3).enumerate().for_each(|(y, row)| {
        let (lo, ws) = &yt[y];
        for i in 0..ow * 3 {
            let mut acc = 0f32;
            for (k, wt) in ws.iter().enumerate() {
                acc += wt * mid[(lo + k) * ow * 3 + i];
            }
            row[i] = (acc + 0.5).clamp(0.0, 255.0) as u8;
        }
    });
    out
}

// ---- rules shared by the CLIs ---------------------------------------------

/// h3-rs's `cli_factor`: the network's ×4 up to 832×480, ×2 above.
pub fn default_factor(w: usize, h: usize) -> usize {
    if w * h <= 832 * 480 {
        4
    } else {
        2
    }
}

/// Largest output a browser-bound clip is given by default (h3-rs's
/// `SERVE_MAX_OUT_PIXELS`).
pub const SERVE_MAX_OUT_PIXELS: usize = 2560 * 1440;

/// h3-rs's `serve_factor`: the largest factor ≤ `scale` whose output fits
/// [`SERVE_MAX_OUT_PIXELS`], at least 2 (480×272 → ×4, 832×480 → ×3,
/// 1344×768 → ×2).
pub fn serve_factor(w: usize, h: usize, scale: usize) -> usize {
    (2..=scale)
        .rev()
        .find(|k| w * k * h * k <= SERVE_MAX_OUT_PIXELS)
        .unwrap_or(2)
}

/// Where the checkpoint is, in order: `explicit` (`--upscale-weights`),
/// `FASTVIDEO_ESRGAN_WEIGHTS`, `$FASTVIDEO_TAE_DIR/realesr-animevideov3.pth`,
/// then `auxiliary/upscalers/realesr-animevideov3.pth` under each weights
/// root (`root`, its parent, `FASTVIDEO_WEIGHTS`, `FV_WEIGHTS`). A missing
/// file names every path tried and the pinned URL.
pub fn resolve_weights(explicit: Option<&Path>, root: Option<&Path>) -> Result<PathBuf> {
    let env = |k: &str| {
        fastvideo_models::settings::env_var(k)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from)
    };
    let mut tried: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(p.to_path_buf());
        }
        return Err(msg(format!(
            "upscaler weights {} do not exist (the pinned file is {ESRGAN_URL}, sha256 {ESRGAN_SHA256})",
            p.display()
        )));
    }
    if let Some(p) = env("FASTVIDEO_ESRGAN_WEIGHTS") {
        tried.push(p);
    }
    if let Some(d) = env("FASTVIDEO_TAE_DIR") {
        tried.push(d.join(ESRGAN_FILE));
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(r) = root {
        roots.push(r.to_path_buf());
        if let Some(parent) = r.parent() {
            roots.push(parent.to_path_buf());
        }
    }
    roots.extend(env("FASTVIDEO_WEIGHTS"));
    roots.extend(env("FV_WEIGHTS"));
    for r in roots {
        tried.push(r.join("auxiliary").join("upscalers").join(ESRGAN_FILE));
    }
    if let Some(p) = tried.iter().find(|p| p.is_file()) {
        return Ok(p.clone());
    }
    Err(msg(format!(
        "upscaler weights not found: pass --upscale-weights, set FASTVIDEO_ESRGAN_WEIGHTS, or put {ESRGAN_FILE} in $FASTVIDEO_TAE_DIR or <weights>/auxiliary/upscalers/ (scripts/gpu/fetch-tae.sh fetches it; pinned {ESRGAN_URL}, sha256 {ESRGAN_SHA256}); tried {}",
        tried
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

// ---- a clip directory -----------------------------------------------------

/// What [`upscale_dir`] did.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UpscaleReport {
    pub factor: usize,
    pub frames: usize,
    pub in_width: usize,
    pub in_height: usize,
    pub out_width: usize,
    pub out_height: usize,
    /// Reading the PNGs.
    pub read_s: f64,
    /// Loading (and uploading) the weights.
    pub load_s: f64,
    /// The network, the pack and any host resample, every batch.
    pub upscale_s: f64,
    /// Writing `frames-up/` and `output-up.mp4`.
    pub write_s: f64,
    /// Frames per device batch.
    pub batch: usize,
    pub weights: String,
    pub frame_paths: Vec<String>,
    pub mp4: Option<String>,
    /// `cuda` or `cpu`.
    pub device: &'static str,
}

/// The `frame-NNN.png` files of `dir`, sorted.
fn frame_pngs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| msg(format!("{}: {e}", dir.display())))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("png")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("frame-"))
        })
        .collect();
    v.sort();
    Ok(v)
}

/// The frame rate of an mp4 (`ffprobe`'s `r_frame_rate`, rounded).
pub fn probe_fps(mp4: &Path) -> Option<u32> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
        ])
        .arg(mp4)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let rate = text.trim();
    let (num, den) = match rate.split_once('/') {
        Some((a, b)) => (a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?),
        None => (rate.parse::<f64>().ok()?, 1.0),
    };
    if den <= 0.0 || num <= 0.0 {
        return None;
    }
    Some((num / den).round().max(1.0) as u32)
}

/// Upscale the `frame-NNN.png` frames of a clip directory by `factor` into
/// `<dir>/frames-up/frame-NNN.png`, and, when `<dir>/output.mp4` exists (or
/// `fps` is given), `<dir>/output-up.mp4` at `fps` (default: the original
/// mp4's rate) with `audio` (a WAV, e.g. H3's `audio.wav`) muxed in, through
/// the pipelines' own [`super::writer::VideoWriter`]. The shared helper
/// behind `fv-cli generate --upscale`, `fv-gpucheck wan gen --upscale` and
/// `fv-gpucheck h3 gen --upscale`.
pub fn upscale_dir(
    dir: &Path,
    factor: usize,
    weights: &Path,
    fps: Option<u32>,
    audio: Option<&Path>,
) -> Result<UpscaleReport> {
    let pngs = frame_pngs(dir)?;
    if pngs.is_empty() {
        return Err(msg(format!(
            "upscale: no frame-NNN.png in {}",
            dir.display()
        )));
    }
    let timer = Instant::now();
    let decoded: Vec<Result<(usize, usize, Vec<u8>)>> = pngs
        .par_iter()
        .map(|p| {
            let img = image::open(p)
                .map_err(|e| msg(format!("{}: {e}", p.display())))?
                .to_rgb8();
            Ok((img.width() as usize, img.height() as usize, img.into_raw()))
        })
        .collect();
    let mut rgb = Vec::new();
    let (mut w, mut h) = (0usize, 0usize);
    for (i, d) in decoded.into_iter().enumerate() {
        let (fw, fh, data) = d?;
        if i == 0 {
            (w, h) = (fw, fh);
            rgb.reserve(pngs.len() * fw * fh * 3);
        } else if (fw, fh) != (w, h) {
            return Err(msg(format!(
                "upscale: {} is {fw}x{fh}, frame 0 is {w}x{h}",
                pngs[i].display()
            )));
        }
        rgb.extend_from_slice(&data);
    }
    let frames = pngs.len();
    let read_s = timer.elapsed().as_secs_f64();

    let timer = Instant::now();
    let up = Esrgan::load(weights)?;
    #[cfg(feature = "cuda")]
    let on_device = super::device::global_device().is_some();
    #[cfg(not(feature = "cuda"))]
    let on_device = false;
    #[cfg(feature = "cuda")]
    if on_device {
        up.device_layers()?;
    }
    let load_s = timer.elapsed().as_secs_f64();
    let batch = up.batch_frames(frames, h, w, factor, Esrgan::budget_bytes());
    let (ow, oh) = (w * factor, h * factor);

    let mp4_in = dir.join("output.mp4");
    let fps = fps.or_else(|| mp4_in.is_file().then(|| probe_fps(&mp4_in)).flatten());
    let want_mp4 = fps.is_some() && (mp4_in.is_file() || audio.is_some());
    let up_dir = dir.join("frames-up");
    let mut writer =
        super::writer::VideoWriter::spawn_with_audio(&up_dir, fps.unwrap_or(0), want_mp4, audio)?;
    super::log::info(format_args!(
        "upscale ×{factor}: {frames} frames {w}x{h} → {ow}x{oh}, {batch} per batch, {} ({})",
        if on_device { "cuda" } else { "cpu" },
        weights.display()
    ));
    let timer = Instant::now();
    let mut upscale_s = 0.0;
    let mut write_s = 0.0;
    up.upscale_rgb8(&rgb, frames, h, w, factor, &mut |f0, _n, out| {
        upscale_s = timer.elapsed().as_secs_f64() - write_s;
        let t = Instant::now();
        writer.push(f0, oh, ow, out)?;
        write_s += t.elapsed().as_secs_f64();
        Ok(())
    })?;
    #[cfg(feature = "cuda")]
    if on_device {
        super::device::synchronize().map_err(|e| msg(e.to_string()))?;
    }
    upscale_s = timer.elapsed().as_secs_f64() - write_s;
    let t = Instant::now();
    let (frame_paths, mp4) = writer.finish()?;
    let mp4 = match mp4 {
        Some(p) => {
            let dst = dir.join("output-up.mp4");
            std::fs::rename(&p, &dst).map_err(|e| msg(format!("{p} → {}: {e}", dst.display())))?;
            Some(dst.to_string_lossy().into_owned())
        }
        None => None,
    };
    write_s += t.elapsed().as_secs_f64();
    super::log::info(format_args!(
        "upscale ×{factor}: {ow}x{oh} in {upscale_s:.2}s (read {read_s:.2}s, load {load_s:.2}s, write {write_s:.2}s)"
    ));
    Ok(UpscaleReport {
        factor,
        frames,
        in_width: w,
        in_height: h,
        out_width: ow,
        out_height: oh,
        read_s,
        load_s,
        upscale_s,
        write_s,
        batch,
        weights: weights.display().to_string(),
        frame_paths,
        mp4,
        device: if on_device { "cuda" } else { "cpu" },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic test frame (`scripts/ref/esrgan_parity.py` makes the
    /// same one): LCG bytes, `[h, w, 3]`.
    pub(crate) fn lcg_frame(h: usize, w: usize, seed: u32) -> Vec<u8> {
        let mut x = seed;
        (0..h * w * 3)
            .map(|_| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
                ((x >> 16) & 0xff) as u8
            })
            .collect()
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::Digest;
        sha2::Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    #[test]
    fn manifest_is_the_checkpoints_keys_and_shapes() {
        let m = esrgan_manifest(&EsrganConfig::anime_video_v3());
        // 18 convs with PReLU (3 keys each) + the last conv (2 keys).
        assert_eq!(m.len(), 17 * 3 + 2);
        assert_eq!(m[0], ("body.0.weight".into(), vec![64, 3, 3, 3]));
        assert_eq!(m[1], ("body.0.bias".into(), vec![64]));
        assert_eq!(m[2], ("body.1.weight".into(), vec![64]));
        assert_eq!(m[3], ("body.2.weight".into(), vec![64, 64, 3, 3]));
        assert_eq!(
            m[m.len() - 2],
            ("body.34.weight".into(), vec![48, 64, 3, 3])
        );
        assert_eq!(m[m.len() - 1], ("body.34.bias".into(), vec![48]));
        let t = esrgan_manifest(&EsrganConfig::tiny());
        assert_eq!(t.len(), 3 * 3 + 2);
        assert_eq!(t[t.len() - 2].1, vec![12, 8, 3, 3]);
        let total: usize = m.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
        // 2.5 MB of f32: the 2504012-byte .pth holds these plus the zip.
        assert_eq!(
            total,
            3 * 64 * 9 + 64 + 64 + 16 * (64 * 64 * 9 + 64 + 64) + 48 * 64 * 9 + 48
        );
    }

    #[test]
    fn missing_key_is_an_error() {
        let err = Esrgan::from_source(EsrganConfig::tiny(), |k, shape| {
            if k == "body.3.weight" {
                Err(msg(format!("missing key {k}")))
            } else {
                Ok(vec![0.0; shape.iter().product()])
            }
        })
        .err()
        .expect("missing key");
        assert!(err.to_string().contains("body.3.weight"), "{err}");
    }

    #[test]
    fn batch_rule() {
        let up = Esrgan::synthetic(EsrganConfig::anime_video_v3(), 1);
        // 480x272: (3 + 128 + 48) f32 + 3 B in + 48 B out per pixel.
        let per = up.frame_bytes(272, 480, 4);
        assert_eq!(per, 272 * 480 * (179 * 4 + 3 + 48));
        assert_eq!(
            up.batch_frames(81, 272, 480, 4, 2 << 30),
            (2u64 << 30) as usize / per as usize
        );
        assert_eq!(up.batch_frames(81, 272, 480, 4, 2 << 30), 21);
        // ×2 writes a quarter of the pixels; ×3 is packed at ×4 first.
        assert!(up.frame_bytes(272, 480, 2) < per);
        assert_eq!(up.frame_bytes(272, 480, 3), per);
        // At least one frame, never more than the clip.
        assert_eq!(up.batch_frames(81, 272, 480, 4, 1), 1);
        assert_eq!(up.batch_frames(3, 272, 480, 4, u64::MAX), 3);
        assert_eq!(up.batch_frames(0, 272, 480, 4, u64::MAX), 1);
    }

    #[test]
    fn factors_and_sizes() {
        assert_eq!(default_factor(480, 272), 4);
        assert_eq!(default_factor(832, 480), 4);
        assert_eq!(default_factor(1344, 768), 2);
        assert_eq!(serve_factor(480, 272, 4), 4);
        assert_eq!(serve_factor(832, 480, 4), 3);
        assert_eq!(serve_factor(1344, 768, 4), 2);
        assert_eq!(serve_factor(480, 272, 2), 2);
    }

    #[test]
    fn tiny_end_to_end_every_factor() {
        let mut cfg = EsrganConfig::tiny();
        cfg.scale = 4;
        let up = Esrgan::synthetic(cfg, 3);
        let (w, h, frames) = (6, 4, 3);
        let rgb: Vec<u8> = (0..frames * w * h * 3)
            .map(|i| (i * 7 % 251) as u8)
            .collect();
        for factor in [4, 3, 2] {
            let mut got = Vec::new();
            let mut batches = 0;
            up.upscale_rgb8_cpu(&rgb, frames, h, w, factor, &mut |f0, n, x| {
                assert_eq!(f0, got.len() / (w * factor * h * factor * 3));
                assert_eq!(x.len(), n * w * factor * h * factor * 3);
                got.extend(x);
                batches += 1;
                Ok(())
            })
            .unwrap();
            assert_eq!(got.len(), frames * w * factor * h * factor * 3);
            assert_eq!(batches, 1);
            // The residual keeps the picture near the nearest-upsampled input.
            let mut close = 0;
            for f in 0..frames {
                for y in 0..h * factor {
                    for x in 0..w * factor {
                        for c in 0..3 {
                            let base = rgb[((f * h + y / factor) * w + x / factor) * 3 + c];
                            let v = got[((f * h * factor + y) * w * factor + x) * 3 + c];
                            if (i32::from(base) - i32::from(v)).abs() <= 40 {
                                close += 1;
                            }
                        }
                    }
                }
            }
            assert!(
                close * 10 > got.len() * 9,
                "factor {factor}: {close} of {}",
                got.len()
            );
        }
        assert!(up
            .upscale_rgb8_cpu(&rgb, frames, h, w, 5, &mut |_, _, _| Ok(()))
            .is_err());
        assert!(up
            .upscale_rgb8_cpu(&rgb, frames, h, w, 1, &mut |_, _, _| Ok(()))
            .is_err());
        // The ×2 network takes ×2 only.
        let up2 = Esrgan::synthetic(EsrganConfig::tiny(), 3);
        assert!(up2
            .upscale_rgb8_cpu(&rgb, frames, h, w, 2, &mut |_, _, _| Ok(()))
            .is_ok());
        assert!(up2
            .upscale_rgb8_cpu(&rgb, frames, h, w, 3, &mut |_, _, _| Ok(()))
            .is_err());
    }

    /// ×2 is the ×4 output box-filtered: each ×2 pixel is the rounded mean of
    /// its 2×2 block of the ×4 picture before rounding.
    #[test]
    fn x2_is_the_box_filter_of_x4() {
        let mut cfg = EsrganConfig::tiny();
        cfg.scale = 4;
        let up = Esrgan::synthetic(cfg, 5);
        let (w, h) = (5, 3);
        let rgb = lcg_frame(h, w, 9);
        let x = planar_unit(&rgb, 1, h, w);
        let y = up.network_cpu(&x, 1, h, w);
        let x4 = host::esrgan_pack_rgb8(&y, &rgb, 1, h, w, 4, 1);
        let x2 = host::esrgan_pack_rgb8(&y, &rgb, 1, h, w, 4, 2);
        assert_eq!(x2.len(), (2 * h) * (2 * w) * 3);
        let mut off_by_one = 0;
        for oy in 0..2 * h {
            for ox in 0..2 * w {
                for c in 0..3 {
                    let mean = (0..2)
                        .flat_map(|dy| (0..2).map(move |dx| (dy, dx)))
                        .map(|(dy, dx)| {
                            f64::from(x4[((oy * 2 + dy) * 4 * w + ox * 2 + dx) * 3 + c])
                        })
                        .sum::<f64>()
                        / 4.0;
                    let got = f64::from(x2[(oy * 2 * w + ox) * 3 + c]);
                    // The ×4 bytes were rounded first, so the two can differ by
                    // the rounding of the mean, never more.
                    assert!(
                        (got - mean).abs() <= 1.0,
                        "({oy},{ox},{c}): {got} vs {mean}"
                    );
                    if got != mean.round() {
                        off_by_one += 1;
                    }
                }
            }
        }
        assert!(off_by_one < x2.len() / 4, "{off_by_one} of {}", x2.len());
    }

    #[test]
    fn lanczos_resample_keeps_flat_and_sizes() {
        let (w, h) = (8, 6);
        let flat = vec![[10u8, 200, 77]; w * h].concat();
        let out = resize_lanczos3(&flat, w, h, 6, 4);
        assert_eq!(out.len(), 6 * 4 * 3);
        assert!(out.chunks(3).all(|p| p == [10, 200, 77]), "{out:?}");
        let same = resize_lanczos3(&flat, w, h, w, h);
        assert_eq!(same, flat);
        // A gradient stays monotone through the resample.
        let grad: Vec<u8> = (0..h)
            .flat_map(|_| (0..w).flat_map(|x| [(x * 30) as u8; 3]))
            .collect();
        let g = resize_lanczos3(&grad, w, h, 12, 9);
        let row: Vec<u8> = (0..12).map(|x| g[(4 * 12 + x) * 3]).collect();
        assert!(row.windows(2).all(|p| p[0] <= p[1]), "{row:?}");
        // The ×3 taps shrink 4 → 3: every output has at least 6 taps.
        let taps = lanczos_taps(16, 12);
        assert!(taps.iter().all(|(_, ws)| ws.len() >= 6));
        assert!(taps
            .iter()
            .all(|(_, ws)| (ws.iter().sum::<f32>() - 1.0).abs() < 1e-5));
    }

    #[test]
    fn pack_twin_rounds_like_the_reference() {
        // y = 0 everywhere: the output is the nearest-upsampled input.
        let (w, h, s) = (3, 2, 2);
        let rgb = lcg_frame(h, w, 1);
        let y = vec![0f32; 3 * s * s * w * h];
        let out = host::esrgan_pack_rgb8(&y, &rgb, 1, h, w, s, 1);
        for yy in 0..h * s {
            for xx in 0..w * s {
                for c in 0..3 {
                    assert_eq!(
                        out[((yy * w * s) + xx) * 3 + c],
                        rgb[((yy / s) * w + xx / s) * 3 + c]
                    );
                }
            }
        }
        // Clamp and round: 1/255 of a level below .5 rounds down, at .5 up.
        // The channels c·s² + py·s + px of source pixel (0, 0) land at
        // output (py, px); channel k is plane k of `y`.
        let plane = w * h;
        let mut y = vec![0f32; 3 * s * s * plane];
        y[0] = 0.5 / 255.0; // (0,0): base + 0.5 → base + 1
        y[plane] = 0.49 / 255.0; // (0,1): rounds down to base
        y[2 * plane] = 100.0; // (1,0): clamped to 255
        y[3 * plane] = -100.0; // (1,1): clamped to 0
        let out = host::esrgan_pack_rgb8(&y, &rgb, 1, h, w, s, 1);
        let base = rgb[0];
        assert_eq!(out[0], base.saturating_add(1));
        assert_eq!(out[3], base);
        assert_eq!(out[(w * s) * 3], 255);
        assert_eq!(out[(w * s + 1) * 3], 0);
    }

    #[test]
    fn bias_prelu_twin() {
        let mut x = vec![-2.0f32, 1.0, -0.5, 3.0, -1.0, -1.0];
        host::bias_prelu(&mut x, &[0.5, -2.0], &[0.1, 0.25], 3);
        assert_eq!(x, vec![-0.15, 1.5, 0.0, 1.0, -0.75, -0.75]);
    }

    /// A second, independently written loop over the same definition
    /// (im2col-free, per output pixel, PyTorch's conv/PReLU/pixel-shuffle
    /// semantics spelled out), f64 throughout then stored f32 per layer as
    /// the reference does.
    fn srvgg_plain(up: &Esrgan, rgb: &[u8], h: usize, w: usize) -> Vec<u8> {
        let s = up.cfg.scale;
        let mut cur: Vec<Vec<f32>> = (0..3)
            .map(|c| {
                (0..h * w)
                    .map(|p| f32::from(rgb[p * 3 + c]) / 255.0)
                    .collect()
            })
            .collect();
        for l in &up.layers {
            let mut next = vec![vec![0f32; h * w]; l.cout];
            for oc in 0..l.cout {
                for y in 0..h {
                    for x in 0..w {
                        let mut acc = 0.0f64;
                        for ic in 0..l.cin {
                            for ky in 0..3usize {
                                for kx in 0..3usize {
                                    let (sy, sx) =
                                        (y as i64 + ky as i64 - 1, x as i64 + kx as i64 - 1);
                                    if sy < 0 || sx < 0 || sy >= h as i64 || sx >= w as i64 {
                                        continue;
                                    }
                                    let v = cur[ic][sy as usize * w + sx as usize];
                                    let wv = l.w[((oc * l.cin + ic) * 3 + ky) * 3 + kx];
                                    acc += f64::from(v) * f64::from(wv);
                                }
                            }
                        }
                        let mut v = (acc + f64::from(l.b[oc])) as f32;
                        if let Some(a) = &l.slope {
                            if v < 0.0 {
                                v *= a[oc];
                            }
                        }
                        next[oc][y * w + x] = v;
                    }
                }
            }
            cur = next;
        }
        let (oh, ow) = (h * s, w * s);
        let mut out = vec![0u8; oh * ow * 3];
        for c in 0..3 {
            for oy in 0..oh {
                for ox in 0..ow {
                    let ch = c * s * s + (oy % s) * s + ox % s;
                    let v = f64::from(cur[ch][(oy / s) * w + ox / s]) * 255.0
                        + f64::from(rgb[((oy / s) * w + ox / s) * 3 + c]);
                    out[(oy * ow + ox) * 3 + c] = (v + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
        out
    }

    #[test]
    fn tiny_reference_matches_a_plain_loop() {
        let up = Esrgan::synthetic(EsrganConfig::tiny(), 11);
        let (w, h) = (7, 5);
        let rgb = lcg_frame(h, w, 3);
        let mut got = Vec::new();
        up.upscale_rgb8_cpu(&rgb, 1, h, w, 2, &mut |_, _, x| {
            got = x;
            Ok(())
        })
        .unwrap();
        assert_eq!(got, srvgg_plain(&up, &rgb, h, w));
    }

    /// The real checkpoint (`FASTVIDEO_ESRGAN_WEIGHTS`, else skipped) on a
    /// random 24×16 frame: byte-identical to the independent plain loop and
    /// to the numpy SRVGGNetCompact of `scripts/ref/esrgan_parity.py` (its
    /// sha256 of the ×4 bytes pinned here), and ×2 / ×3 sizes.
    #[test]
    fn real_weights_match_numpy_and_a_plain_loop() {
        let Some(path) = fastvideo_models::settings::env_var_os("FASTVIDEO_ESRGAN_WEIGHTS") else {
            eprintln!("FASTVIDEO_ESRGAN_WEIGHTS unset: skipping the real-weights parity test");
            return;
        };
        let path = PathBuf::from(path);
        let up = Esrgan::load(&path).unwrap();
        assert_eq!(up.cfg, EsrganConfig::anime_video_v3());
        let (w, h) = (24, 16);
        let rgb = lcg_frame(h, w, 2026);
        let mut x4 = Vec::new();
        up.upscale_rgb8_cpu(&rgb, 1, h, w, 4, &mut |_, _, x| {
            x4 = x;
            Ok(())
        })
        .unwrap();
        assert_eq!(x4.len(), 96 * 64 * 3);
        assert_eq!(x4, srvgg_plain(&up, &rgb, h, w), "plain loop");
        assert_eq!(
            sha256_hex(&x4),
            "23224e6c7df1449084cb7d9f323b731b326bb9145e1465ac4afe3f750c9041f9",
            "numpy SRVGGNetCompact (scripts/ref/esrgan_parity.py)"
        );
        for factor in [2, 3] {
            let mut got = Vec::new();
            up.upscale_rgb8_cpu(&rgb, 1, h, w, factor, &mut |_, _, x| {
                got = x;
                Ok(())
            })
            .unwrap();
            assert_eq!(got.len(), w * factor * h * factor * 3);
        }
    }
}
