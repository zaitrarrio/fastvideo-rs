//! The LTX video VAE decoder as the reference runs it: bf16, channels-last,
//! one tile at a time.
//!
//! sol-engine's LTX-2.5 profile decodes with `ltx_core`'s memory-efficient
//! path (`memory_efficient_decode.py`, installed by `VideoDecoder` in
//! `ltx_pipelines/utils/blocks.py`): bf16 weights and activations,
//! `channels_last_3d` workspaces, cuDNN convolutions, every tile decoded whole
//! (temporal conv chunks of at most 16 frames). The f32 streaming decoder in
//! [`super`] computes the same function, but one latent frame at a time
//! through NCDHW f32 buffers: each conv pads and concatenates its input,
//! casts it (and its weight) to bf16 for cuDNN and back, then adds the bias,
//! and the norms and residuals are more f32 passes.
//!
//! Here activations are bf16 `[T, H*W, C]` frames (batch 1). A buffer a conv
//! reads carries one extra frame on each side holding the replicate-padded
//! edge frames, so a temporal chunk of conv input is one contiguous slice and
//! cuDNN (bf16 NDHWC, f32 accumulate, zero spatial padding) runs on it in
//! place. Between convs one fused kernel does bias, residual, PixelNorm and
//! SiLU (`ltxv_norm_silu`) with the reference's bf16 rounding points; the
//! upsamplers' depth-to-space and the output unpatchify fold the bias into
//! their rearrangement. See `kernels.cu`, region "ltx video vae".
//!
//! `FASTVIDEO_LTX_VAE_FAST=0` keeps the f32 streaming decoder.
//! `FASTVIDEO_LTX_VAE_CHECK=1` decodes the first tile both ways and logs the
//! difference and both timings.

/// `ltxv_norm_silu` flags (must match `kernels.cu`).
pub mod flags {
    pub const Y: i32 = 1;
    pub const BIAS: i32 = 2;
    pub const READ_X: i32 = 4;
    pub const WRITE_X: i32 = 8;
    pub const WRITE_P: i32 = 16;
    pub const X_EDGES: i32 = 32;
}

/// Plain-Rust twins of the `ltxv_*` kernels (what `fv-gpucheck kernels`
/// compares them with). bf16 values travel as [`half::bf16`].
pub mod host {
    use half::bf16;

    fn r(v: f32) -> f32 {
        bf16::from_f32(v).to_f32()
    }

    /// See `ltxv_norm_silu`. `y` is the chunk `[nt, p, c]`, `x` the whole
    /// `[t_total, p, c]` interior (updated in place under `WRITE_X`), `pad`
    /// the `[t_total + 2, p, c]` padded output (under `WRITE_P`). `X_EDGES`
    /// has no host twin (it needs the padded buffer around `x`).
    #[allow(clippy::too_many_arguments)]
    pub fn norm_silu(
        y: Option<&[bf16]>,
        bias: Option<&[f32]>,
        x: &mut [bf16],
        pad: &mut [bf16],
        p: usize,
        c: usize,
        t0: usize,
        nt: usize,
        t_total: usize,
        eps: f32,
        flags: i32,
    ) {
        use super::flags as f;
        let mut v = vec![0f32; c];
        for tl in 0..nt {
            let t = t0 + tl;
            for q in 0..p {
                let xo = (t * p + q) * c;
                for (ch, vv) in v.iter_mut().enumerate() {
                    let mut a = 0.0f32;
                    if flags & f::Y != 0 {
                        a = y.expect("y")[(tl * p + q) * c + ch].to_f32();
                        if flags & f::BIAS != 0 {
                            a = r(a + bias.expect("bias")[ch]);
                        }
                    }
                    if flags & f::READ_X != 0 {
                        let b = x[xo + ch].to_f32();
                        a = if flags & f::Y != 0 { r(b + a) } else { b };
                    }
                    *vv = a;
                }
                if flags & f::WRITE_X != 0 {
                    for (ch, vv) in v.iter().enumerate() {
                        x[xo + ch] = bf16::from_f32(*vv);
                    }
                }
                if flags & f::WRITE_P == 0 {
                    continue;
                }
                let ss: f32 = v.iter().map(|a| r(a * a)).sum();
                let mean = r(ss * (1.0 / c as f32));
                let rms = r(r(mean + eps).sqrt());
                let out: Vec<bf16> = v
                    .iter()
                    .map(|a| {
                        let n = r(a / rms);
                        bf16::from_f32(n / (1.0 + (-n).exp()))
                    })
                    .collect();
                let mut put = |frame: usize| {
                    pad[(frame * p + q) * c..(frame * p + q + 1) * c].copy_from_slice(&out)
                };
                put(t + 1);
                if t == 0 {
                    put(0);
                }
                if t + 1 == t_total {
                    put(t_total + 1);
                }
            }
        }
    }

    /// See `ltxv_latent_in`: `z` `[c, f, p]` → padded `[f + 2, p, c]`.
    pub fn latent_in(
        z: &[f32],
        std: &[f32],
        mean: &[f32],
        p: usize,
        c: usize,
        f: usize,
    ) -> Vec<bf16> {
        let mut out = vec![bf16::ZERO; (f + 2) * p * c];
        for tp in 0..f + 2 {
            let t = tp.saturating_sub(1).min(f - 1);
            for q in 0..p {
                for ch in 0..c {
                    let v = r(z[(ch * f + t) * p + q]);
                    let v = r(v * std[ch]);
                    out[(tp * p + q) * c + ch] = bf16::from_f32(v + mean[ch]);
                }
            }
        }
        out
    }

    /// See `ltxv_d2s_bias`: chunk `y` `[nt, h, w, c*st*sh*sw]` (input frames
    /// `t0..`) into `out` `[t_out, sh*h, sw*w, c]`.
    #[allow(clippy::too_many_arguments)]
    pub fn d2s_bias(
        y: &[bf16],
        bias: &[f32],
        out: &mut [bf16],
        h: usize,
        w: usize,
        c: usize,
        (st, sh, sw): (usize, usize, usize),
        t0: usize,
        nt: usize,
        drop: usize,
    ) {
        let prod = st * sh * sw;
        let (ho_n, wo_n) = (sh * h, sw * w);
        for tl in 0..nt {
            for s in 0..st {
                let Some(of) = (st * (t0 + tl) + s).checked_sub(drop) else {
                    continue;
                };
                for ho in 0..ho_n {
                    for wo in 0..wo_n {
                        for ch in 0..c {
                            let (hh, j, ww, k) = (ho / sh, ho % sh, wo / sw, wo % sw);
                            let cy = ch * prod + s * sh * sw + j * sw + k;
                            let v = y[((tl * h + hh) * w + ww) * c * prod + cy].to_f32() + bias[cy];
                            out[((of * ho_n + ho) * wo_n + wo) * c + ch] = bf16::from_f32(v);
                        }
                    }
                }
            }
        }
    }

    /// See `ltxv_out_unpatch`: chunk `y` `[nt, h, w, co*pz*pz]` into frames
    /// `t0..` of `out` `[t, co, pz*h, pz*w]`.
    #[allow(clippy::too_many_arguments)]
    pub fn out_unpatch(
        y: &[bf16],
        bias: &[f32],
        out: &mut [bf16],
        h: usize,
        w: usize,
        co: usize,
        pz: usize,
        t0: usize,
        nt: usize,
    ) {
        let (ho_n, wo_n) = (pz * h, pz * w);
        for tl in 0..nt {
            for ch in 0..co {
                for ho in 0..ho_n {
                    for wo in 0..wo_n {
                        let (hh, b, ww, a) = (ho / pz, ho % pz, wo / pz, wo % pz);
                        let cy = ch * pz * pz + a * pz + b;
                        let v = y[((tl * h + hh) * w + ww) * co * pz * pz + cy].to_f32() + bias[cy];
                        out[(((t0 + tl) * co + ch) * ho_n + ho) * wo_n + wo] = bf16::from_f32(v);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use device::*;

#[cfg(feature = "cuda")]
mod device {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::cudnn::{sys, ConvDescriptor, ConvForward, FilterDescriptor, TensorDescriptor};
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use half::bf16;

    use super::super::{Block, TemporalConv, VideoDecoder};
    use super::flags;
    use crate::wan::device::{global_device, DeviceContext};
    use crate::wan::ops::PadMode;
    use crate::wan::tensor::{CudaTensor, Result, TensorError};

    fn msg(s: impl Into<String>) -> TensorError {
        TensorError::Message(s.into())
    }

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(e.to_string())
    }

    fn ctx() -> Result<Arc<DeviceContext>> {
        global_device().ok_or_else(|| msg("ltx vae: no CUDA device"))
    }

    fn ptr<T>(dev: &DeviceContext, s: &CudaSlice<T>) -> u64 {
        let (p, _g) = s.device_ptr(&dev.stream);
        p
    }

    fn ptr_mut<T>(dev: &DeviceContext, s: &mut CudaSlice<T>) -> u64 {
        let (p, _g) = s.device_ptr_mut(&dev.stream);
        p
    }

    fn cfg_n(n: usize) -> LaunchConfig {
        LaunchConfig::for_num_elems(n.max(1) as u32)
    }

    /// Whether the channels-last decoder handles tiled decodes
    /// (`FASTVIDEO_LTX_VAE_FAST`, default on).
    pub fn enabled() -> bool {
        match OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
        static ON: crate::wan::envflag::CachedBool = crate::wan::envflag::CachedBool::new();
        ON.get_or_init(|| crate::wan::envflag::bool_flag("FASTVIDEO_LTX_VAE_FAST", false))
    }

    static OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

    /// Force the decoder choice for this process (`Some(true)`: channels-last,
    /// `Some(false)`: f32 streaming, `None`: the environment). For benches
    /// that time both.
    pub fn set_override(fast: Option<bool>) {
        let v = match fast {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        };
        OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// `FASTVIDEO_LTX_VAE_CHECK=1`: decode the first tile both ways.
    pub fn check_enabled() -> bool {
        static ON: crate::wan::envflag::CachedBool = crate::wan::envflag::CachedBool::new();
        ON.get_or_init(|| crate::wan::envflag::bool_flag("FASTVIDEO_LTX_VAE_CHECK", false))
    }

    // ---- kernel launchers (raw device addresses; bf16 buffers) -------------

    /// `ltxv_norm_silu`. `x` is frame 0 of the `[t_total, p, c]` interior,
    /// `pad` the `[t_total + 2, p, c]` output; a 0 address is unread.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_norm_silu(
        y: u64,
        bias: u64,
        x: u64,
        pad: u64,
        p: usize,
        c: usize,
        t0: usize,
        nt: usize,
        t_total: usize,
        eps: f32,
        fl: i32,
    ) -> Result<()> {
        if c == 0 || c % 128 != 0 || c > 1024 {
            return Err(msg(format!("ltxv_norm_silu: {c} channels")));
        }
        let dev = ctx()?;
        let pixels = nt * p;
        let cfg = LaunchConfig {
            grid_dim: (pixels.div_ceil(8).max(1) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (p_i, c_i, t0_i, nt_i, t_i) =
            (p as i64, c as i32, t0 as i32, nt as i32, t_total as i32);
        crate::wan::stats::record_launch();
        let mut b = dev.stream.launch_builder(&dev.kernels.ltxv_norm_silu);
        b.arg(&y)
            .arg(&bias)
            .arg(&x)
            .arg(&pad)
            .arg(&p_i)
            .arg(&c_i)
            .arg(&t0_i)
            .arg(&nt_i)
            .arg(&t_i)
            .arg(&eps)
            .arg(&fl);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(err)
    }

    /// `ltxv_latent_in`: f32 `[c, f, p]` → padded bf16 `[f + 2, p, c]`.
    pub fn launch_latent_in(
        z: u64,
        std: u64,
        mean: u64,
        out: u64,
        p: usize,
        c: usize,
        f: usize,
    ) -> Result<()> {
        let dev = ctx()?;
        let n = (f + 2) * p * c;
        let (p_i, c_i, f_i) = (p as i64, c as i32, f as i32);
        crate::wan::stats::record_launch();
        let mut b = dev.stream.launch_builder(&dev.kernels.ltxv_latent_in);
        b.arg(&z)
            .arg(&std)
            .arg(&mean)
            .arg(&out)
            .arg(&p_i)
            .arg(&c_i)
            .arg(&f_i);
        unsafe { b.launch(cfg_n(n)) }.map(|_| ()).map_err(err)
    }

    /// `ltxv_d2s_bias`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_d2s_bias(
        y: u64,
        bias: u64,
        out: u64,
        h: usize,
        w: usize,
        c: usize,
        (st, sh, sw): (usize, usize, usize),
        t0: usize,
        nt: usize,
        drop: usize,
    ) -> Result<()> {
        let dev = ctx()?;
        let n = nt * st * sh * h * sw * w * c;
        let a = [h, w, c, st, sh, sw, t0, nt, drop].map(|v| v as i32);
        crate::wan::stats::record_launch();
        let mut b = dev.stream.launch_builder(&dev.kernels.ltxv_d2s_bias);
        b.arg(&y).arg(&bias).arg(&out);
        for v in &a {
            b.arg(v);
        }
        unsafe { b.launch(cfg_n(n)) }.map(|_| ()).map_err(err)
    }

    /// `ltxv_out_unpatch`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_out_unpatch(
        y: u64,
        bias: u64,
        out: u64,
        h: usize,
        w: usize,
        co: usize,
        pz: usize,
        t0: usize,
        nt: usize,
    ) -> Result<()> {
        let dev = ctx()?;
        let n = nt * co * pz * h * pz * w;
        let a = [h, w, co, pz, t0, nt].map(|v| v as i32);
        crate::wan::stats::record_launch();
        let mut b = dev.stream.launch_builder(&dev.kernels.ltxv_out_unpatch);
        b.arg(&y).arg(&bias).arg(&out);
        for v in &a {
            b.arg(v);
        }
        unsafe { b.launch(cfg_n(n)) }.map(|_| ()).map_err(err)
    }

    // ---- weights ------------------------------------------------------------

    /// A 3x3x3 conv's bf16 weight `[cout, 3, 3, 3, cin]` (cuDNN's NHWC filter
    /// layout) and its bias as the bf16 values the reference adds, in f32.
    struct Conv {
        w: CudaTensor,
        b: CudaTensor,
        cin: usize,
        cout: usize,
    }

    impl Conv {
        fn from(tc: &TemporalConv) -> Result<Self> {
            let (cout, cin) = (tc.weight.shape[0], tc.weight.shape[1]);
            if tc.weight.shape[2..] != [3, 3, 3] {
                return Err(msg(format!("ltx vae: conv weight {:?}", tc.weight.shape)));
            }
            let w = tc
                .weight
                .to_f32_act()?
                .permute(&[0, 2, 3, 4, 1])?
                .quantize_bf16()?;
            let b = tc.bias.quantize_bf16()?.to_f32_act()?;
            Ok(Self { w, b, cin, cout })
        }

        fn w(&self) -> Result<&CudaSlice<bf16>> {
            self.w
                .device_slice_bf16()
                .ok_or_else(|| msg("ltx vae: conv weight not on the device"))
        }

        fn bias(&self, dev: &DeviceContext) -> Result<u64> {
            self.b
                .device_slice()
                .map(|s| ptr(dev, s))
                .ok_or_else(|| msg("ltx vae: conv bias not on the device"))
        }
    }

    struct Up {
        conv: Conv,
        stride: (usize, usize, usize),
        drop_first: bool,
    }

    struct FastBlock {
        up: Option<Up>,
        resnets: Vec<(Conv, Conv)>,
    }

    /// The decoder's weights in the channels-last bf16 form.
    pub struct FastDecoder {
        conv_in: Conv,
        blocks: Vec<FastBlock>,
        conv_out: Conv,
        /// bf16 values of `latents_std / scaling_factor` and `latents_mean`.
        std: CudaTensor,
        mean: CudaTensor,
        eps: f32,
        patch: usize,
        out_channels: usize,
    }

    /// Why the channels-last decoder cannot run this configuration, if it can't.
    pub(in crate::ltx2) fn unsupported(dec: &VideoDecoder) -> Option<String> {
        if dec.conv_in.spatial_pad != PadMode::Zeros {
            return Some("reflect spatial padding".into());
        }
        for (i, b) in dec.blocks.iter().enumerate() {
            if let Some(up) = &b.upsampler {
                if up.residual {
                    return Some("residual upsampler".into());
                }
                if up.stride.0 == 1 && up.drop_first_frame {
                    return Some("frame drop without temporal upsampling".into());
                }
            }
            if b.resnets.is_empty() {
                return Some(format!("block {i} without resnets"));
            }
            let c = b.resnets[0].conv1.weight.shape[0];
            if c % 128 != 0 || c > 1024 {
                return Some(format!("block {i}: {c} channels"));
            }
        }
        None
    }

    impl FastDecoder {
        pub(in crate::ltx2) fn build(dec: &VideoDecoder) -> Result<Self> {
            if let Some(why) = unsupported(dec) {
                return Err(msg(format!("ltx vae channels-last decoder: {why}")));
            }
            let blocks = dec
                .blocks
                .iter()
                .map(|b: &Block| -> Result<FastBlock> {
                    Ok(FastBlock {
                        up: b
                            .upsampler
                            .as_ref()
                            .map(|u| -> Result<Up> {
                                Ok(Up {
                                    conv: Conv::from(&u.conv)?,
                                    stride: u.stride,
                                    drop_first: u.drop_first_frame,
                                })
                            })
                            .transpose()?,
                        resnets: b
                            .resnets
                            .iter()
                            .map(|r| Ok((Conv::from(&r.conv1)?, Conv::from(&r.conv2)?)))
                            .collect::<Result<Vec<_>>>()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let scale = dec.cfg.scaling_factor as f32;
            let round = |v: f32| bf16::from_f32(v).to_f32();
            let std: Vec<f32> = dec
                .latents_std
                .host_cow()?
                .iter()
                .map(|&s| round(s / scale))
                .collect();
            let mean: Vec<f32> = dec
                .latents_mean
                .host_cow()?
                .iter()
                .map(|&m| round(m))
                .collect();
            let n = std.len();
            let up = |v: Vec<f32>| -> Result<CudaTensor> {
                CudaTensor::from_vec(v, vec![n])?.to_device()
            };
            Ok(Self {
                conv_in: Conv::from(&dec.conv_in)?,
                blocks,
                conv_out: Conv::from(&dec.conv_out)?,
                std: up(std)?,
                mean: up(mean)?,
                eps: dec.cfg.pixel_norm_eps as f32,
                patch: dec.cfg.patch_size,
                out_channels: dec.cfg.out_channels,
            })
        }

        /// One tile: latents `[1, C, F, h, w]` (DiT-normalised, f32 or bf16) →
        /// `[frames, 3, 32h, 32w]` bf16, the reference decoder's output.
        pub fn decode_tile(&self, latents: &CudaTensor, convs: &mut Convs) -> Result<CudaTensor> {
            let dev = ctx()?;
            let [1, c, f, h, w] = latents.shape[..] else {
                return Err(msg(format!("ltx vae tile: latents {:?}", latents.shape)));
            };
            if c != self.conv_in.cin || f == 0 {
                return Err(msg(format!("ltx vae tile: latents {:?}", latents.shape)));
            }
            let z = latents
                .dev()?
                .ok_or_else(|| msg("ltx vae tile: latents not on the device"))?;
            let mut x = Act::new(&dev, f, h, w, c)?;
            let (sp, mp) = (
                self.std.device_slice().ok_or_else(|| msg("ltx vae: std"))?,
                self.mean
                    .device_slice()
                    .ok_or_else(|| msg("ltx vae: mean"))?,
            );
            let xp = ptr_mut(&dev, &mut x.buf);
            launch_latent_in(
                ptr(&dev, &*z),
                ptr(&dev, sp),
                ptr(&dev, mp),
                xp,
                h * w,
                c,
                f,
            )?;
            drop(z);

            // conv_in, then its bias rounded into the first block's hidden
            // state; the first block starts with a resnet, so its norm too.
            let c0 = self.conv_in.cout;
            let mut hs = Act::new(&dev, f, h, w, c0)?;
            let mut pa = Act::new(&dev, f, h, w, c0)?;
            let first_norm = self.blocks.first().is_some_and(|b| b.up.is_none());
            self.conv_pass(&dev, convs, &self.conv_in, &x, |dev, y, t0, nt| {
                let fl = flags::Y
                    | flags::BIAS
                    | flags::WRITE_X
                    | flags::X_EDGES
                    | if first_norm { flags::WRITE_P } else { 0 };
                launch_norm_silu(
                    y,
                    self.conv_in.bias(dev)?,
                    hs.interior(dev),
                    pa.base(dev),
                    hs.p(),
                    c0,
                    t0,
                    nt,
                    f,
                    self.eps,
                    fl,
                )
            })?;
            drop(x);

            let nblocks = self.blocks.len();
            for (bi, block) in self.blocks.iter().enumerate() {
                if let Some(up) = &block.up {
                    let (st, sh, sw) = up.stride;
                    let drop = usize::from(up.drop_first);
                    let cn = up.conv.cout / (st * sh * sw);
                    let tn = st * hs.t - drop;
                    let mut out = Act::new(&dev, tn, sh * hs.h, sw * hs.w, cn)?;
                    let (hh, ww) = (hs.h, hs.w);
                    let optr = out.interior(&dev);
                    self.conv_pass(&dev, convs, &up.conv, &hs, |dev, y, t0, nt| {
                        launch_d2s_bias(
                            y,
                            up.conv.bias(dev)?,
                            optr,
                            hh,
                            ww,
                            cn,
                            (st, sh, sw),
                            t0,
                            nt,
                            drop,
                        )
                    })?;
                    hs = out;
                    pa = Act::new(&dev, hs.t, hs.h, hs.w, hs.c)?;
                    // The block's first resnet norm.
                    launch_norm_silu(
                        0,
                        0,
                        hs.interior(&dev),
                        pa.base(&dev),
                        hs.p(),
                        hs.c,
                        0,
                        hs.t,
                        hs.t,
                        self.eps,
                        flags::READ_X | flags::WRITE_P,
                    )?;
                }
                let mut pb = Act::new(&dev, hs.t, hs.h, hs.w, hs.c)?;
                let n = block.resnets.len();
                for (ri, (c1, c2)) in block.resnets.iter().enumerate() {
                    let (t, cc) = (hs.t, hs.c);
                    let pbp = pb.base(&dev);
                    self.conv_pass(&dev, convs, c1, &pa, |dev, y, t0, nt| {
                        launch_norm_silu(
                            y,
                            c1.bias(dev)?,
                            0,
                            pbp,
                            hs.p(),
                            cc,
                            t0,
                            nt,
                            t,
                            self.eps,
                            flags::Y | flags::BIAS | flags::WRITE_P,
                        )
                    })?;
                    // After the block's last resnet: the next block's
                    // upsampler reads the hidden state itself (edges filled),
                    // or, after the last block, conv_norm_out + SiLU.
                    let last = ri + 1 == n;
                    let norm_next = !last || bi + 1 == nblocks;
                    let fl = flags::Y
                        | flags::BIAS
                        | flags::READ_X
                        | flags::WRITE_X
                        | if norm_next {
                            flags::WRITE_P
                        } else {
                            flags::X_EDGES
                        };
                    let (xp, pap) = (hs.interior(&dev), pa.base(&dev));
                    let p = hs.p();
                    self.conv_pass(&dev, convs, c2, &pb, |dev, y, t0, nt| {
                        launch_norm_silu(y, c2.bias(dev)?, xp, pap, p, cc, t0, nt, t, self.eps, fl)
                    })?;
                }
                drop(pb);
            }
            drop(hs);

            // conv_out on the final norm, bias, unpatchify.
            let (t, h, w) = (pa.t, pa.h, pa.w);
            let (co, pz) = (self.out_channels, self.patch);
            if self.conv_out.cout != co * pz * pz {
                return Err(msg("ltx vae: conv_out width"));
            }
            let mut out =
                unsafe { dev.stream.alloc::<bf16>(t * co * pz * h * pz * w) }.map_err(err)?;
            let optr = ptr_mut(&dev, &mut out);
            self.conv_pass(&dev, convs, &self.conv_out, &pa, |dev, y, t0, nt| {
                launch_out_unpatch(y, self.conv_out.bias(dev)?, optr, h, w, co, pz, t0, nt)
            })?;
            CudaTensor::from_device_slice_bf16(out, vec![t, co, pz * h, pz * w])
        }

        /// Run `conv` over every output frame of the padded input `x`, in
        /// temporal chunks, handing each chunk's `[nt, H*W, cout]` output (no
        /// bias) to `then(dev, y_addr, first_frame, frames)`.
        fn conv_pass(
            &self,
            dev: &Arc<DeviceContext>,
            convs: &mut Convs,
            conv: &Conv,
            x: &Act,
            mut then: impl FnMut(&DeviceContext, u64, usize, usize) -> Result<()>,
        ) -> Result<()> {
            if x.c != conv.cin {
                return Err(msg(format!(
                    "ltx vae conv: {} channels in, weight wants {}",
                    x.c, conv.cin
                )));
            }
            let p = x.p();
            let max_frames = chunk_frames(p, conv.cin, conv.cout);
            let plan = balanced(x.t, max_frames);
            let biggest = plan.iter().map(|c| c.1).max().unwrap_or(0);
            let mut y = unsafe { dev.stream.alloc::<bf16>((biggest * p * conv.cout).max(1)) }
                .map_err(err)?;
            let w = conv.w()?;
            for (t0, nt) in plan {
                let xv = x.buf.slice(t0 * p * conv.cin..(t0 + nt + 2) * p * conv.cin);
                let mut yv = y.slice_mut(0..nt * p * conv.cout);
                convs.forward(dev, &xv, w, &mut yv, [nt, x.h, x.w], conv.cin, conv.cout)?;
                drop(yv);
                let yp = ptr(dev, &y);
                then(dev, yp, t0, nt)?;
            }
            Ok(())
        }
    }

    /// A bf16 activation `[t + 2, h*w, c]`: frames 1..=t are the tensor, 0 and
    /// t + 1 its replicate padding (written by whoever needs them).
    struct Act {
        buf: CudaSlice<bf16>,
        t: usize,
        h: usize,
        w: usize,
        c: usize,
    }

    impl Act {
        fn new(dev: &DeviceContext, t: usize, h: usize, w: usize, c: usize) -> Result<Self> {
            let buf = unsafe { dev.stream.alloc::<bf16>((t + 2) * h * w * c) }.map_err(err)?;
            Ok(Self { buf, t, h, w, c })
        }
        fn p(&self) -> usize {
            self.h * self.w
        }
        fn base(&mut self, dev: &DeviceContext) -> u64 {
            ptr_mut(dev, &mut self.buf)
        }
        fn interior(&mut self, dev: &DeviceContext) -> u64 {
            self.base(dev) + (self.p() * self.c * 2) as u64
        }
    }

    /// Output frames per conv launch: at most `FASTVIDEO_LTX_VAE_CHUNK_FRAMES`
    /// (default 16, the reference's largest temporal split), and a conv
    /// output of at most `FASTVIDEO_LTX_VAE_CHUNK_MB` (default 1024); cuDNN's
    /// 32-bit tensor index caps the input.
    fn chunk_frames(p: usize, cin: usize, cout: usize) -> usize {
        let max_f = crate::wan::envflag::usize_flag("FASTVIDEO_LTX_VAE_CHUNK_FRAMES", 16).max(1);
        let mb = crate::wan::envflag::usize_flag("FASTVIDEO_LTX_VAE_CHUNK_MB", 1024).max(1);
        let by_bytes = (mb << 20) / (2 * p * cout).max(1);
        let by_index = (i32::MAX as usize / (p * cin.max(cout)).max(1)).saturating_sub(2);
        max_f.min(by_bytes).min(by_index).max(1)
    }

    /// `t` frames in the fewest chunks of at most `max` frames, as even as
    /// possible: `(first, len)` each.
    pub fn balanced(t: usize, max: usize) -> Vec<(usize, usize)> {
        if t == 0 {
            return Vec::new();
        }
        let n = t.div_ceil(max.max(1));
        let (base, extra) = (t / n, t % n);
        let mut out = Vec::with_capacity(n);
        let mut at = 0;
        for i in 0..n {
            let len = base + usize::from(i < extra);
            out.push((at, len));
            at += len;
        }
        out
    }

    // ---- cuDNN --------------------------------------------------------------

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct Key {
        nt: usize,
        h: usize,
        w: usize,
        cin: usize,
        cout: usize,
    }

    struct Plan {
        conv: ConvDescriptor<f32>,
        x: TensorDescriptor<bf16>,
        w: FilterDescriptor<bf16>,
        y: TensorDescriptor<bf16>,
        algo: sys::cudnnConvolutionFwdAlgo_t,
        ws: usize,
    }

    /// cuDNN plans (bf16 NDHWC, f32 accumulate) for one decode, and the
    /// shared workspace.
    #[derive(Default)]
    pub struct Convs {
        plans: HashMap<Key, Plan>,
        ws: Option<CudaSlice<u8>>,
    }

    /// Algorithms picked by `FASTVIDEO_LTX_VAE_CONV_ALGO=tune`, per shape.
    static TUNED: OnceLock<Mutex<HashMap<Key, sys::cudnnConvolutionFwdAlgo_t>>> = OnceLock::new();

    fn ndhwc(t: usize, h: usize, w: usize, c: usize) -> ([i32; 5], [i32; 5]) {
        let d = [1, c, t, h, w].map(|v| v as i32);
        let s = [t * h * w * c, 1, h * w * c, w * c, c].map(|v| v as i32);
        (d, s)
    }

    impl Convs {
        fn plan(&mut self, dev: &DeviceContext, key: Key) -> Result<&Plan> {
            if !self.plans.contains_key(&key) {
                let cudnn = &dev.cudnn;
                let mut conv = cudnn
                    .create_convnd::<f32>(
                        &[0, 1, 1],
                        &[1, 1, 1],
                        &[1, 1, 1],
                        sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                    )
                    .map_err(err)?;
                conv.set_math_type(sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)
                    .map_err(err)?;
                let (xd, xs) = ndhwc(key.nt + 2, key.h, key.w, key.cin);
                let (yd, ys) = ndhwc(key.nt, key.h, key.w, key.cout);
                let x = cudnn.create_nd_tensor::<bf16>(&xd, &xs).map_err(err)?;
                let y = cudnn.create_nd_tensor::<bf16>(&yd, &ys).map_err(err)?;
                let w = cudnn
                    .create_nd_filter::<bf16>(
                        sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
                        &[key.cout as i32, key.cin as i32, 3, 3, 3],
                    )
                    .map_err(err)?;
                let op = ConvForward {
                    conv: &conv,
                    x: &x,
                    w: &w,
                    y: &y,
                };
                let tuned = TUNED
                    .get_or_init(Default::default)
                    .lock()
                    .expect("tuned lock")
                    .get(&key)
                    .copied();
                let algo = match tuned {
                    Some(a) => a,
                    None => op.pick_algorithm().map_err(err)?,
                };
                let ws = op.get_workspace_size(algo).map_err(err)?;
                crate::wan::log::debug(format_args!(
                    "ltx vae conv plan {key:?}: {algo:?}, workspace {ws} B"
                ));
                self.plans.insert(
                    key,
                    Plan {
                        conv,
                        x,
                        w,
                        y,
                        algo,
                        ws,
                    },
                );
            }
            Ok(&self.plans[&key])
        }

        fn ensure_ws(&mut self, dev: &DeviceContext, need: usize) -> Result<()> {
            if need > 0 && self.ws.as_ref().is_none_or(|w| w.len() < need) {
                self.ws = None;
                self.ws = Some(unsafe { dev.stream.alloc::<u8>(need) }.map_err(err)?);
            }
            Ok(())
        }

        /// `y = conv(x)` for one temporal chunk: `x` holds `nt + 2` padded
        /// input frames, `y` gets `nt` output frames.
        #[allow(clippy::too_many_arguments)]
        fn forward<X, Y>(
            &mut self,
            dev: &DeviceContext,
            x: &X,
            w: &CudaSlice<bf16>,
            y: &mut Y,
            [nt, h, ww]: [usize; 3],
            cin: usize,
            cout: usize,
        ) -> Result<()>
        where
            X: DevicePtr<bf16>,
            Y: DevicePtrMut<bf16>,
        {
            let key = Key {
                nt,
                h,
                w: ww,
                cin,
                cout,
            };
            if tune_enabled() && !self.plans.contains_key(&key) {
                self.tune(dev, key, x, w, y)?;
            }
            let need = self.plan(dev, key)?.ws;
            self.ensure_ws(dev, need)?;
            let Convs { plans, ws } = self;
            let plan = &plans[&key];
            let op = ConvForward {
                conv: &plan.conv,
                x: &plan.x,
                w: &plan.w,
                y: &plan.y,
            };
            unsafe {
                op.launch(
                    plan.algo,
                    if plan.ws > 0 { ws.as_mut() } else { None },
                    (bf16::from_f32(1.0), bf16::from_f32(0.0)),
                    x,
                    w,
                    y,
                )
            }
            .map_err(err)
        }

        /// Time every forward algorithm cuDNN accepts for `key` (second of two
        /// runs, synchronized) and remember the fastest.
        fn tune<X, Y>(
            &mut self,
            dev: &DeviceContext,
            key: Key,
            x: &X,
            w: &CudaSlice<bf16>,
            y: &mut Y,
        ) -> Result<()>
        where
            X: DevicePtr<bf16>,
            Y: DevicePtrMut<bf16>,
        {
            use sys::cudnnConvolutionFwdAlgo_t as A;
            if TUNED
                .get_or_init(Default::default)
                .lock()
                .expect("tuned lock")
                .contains_key(&key)
            {
                return Ok(());
            }
            let heuristic = self.plan(dev, key)?.algo;
            let mut best: Option<(f64, A)> = None;
            let mut report = Vec::new();
            for algo in [
                A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
            ] {
                let plan = &self.plans[&key];
                let op = ConvForward {
                    conv: &plan.conv,
                    x: &plan.x,
                    w: &plan.w,
                    y: &plan.y,
                };
                let Ok(need) = op.get_workspace_size(algo) else {
                    continue;
                };
                if need > (4usize << 30) {
                    continue;
                }
                self.ensure_ws(dev, need)?;
                let mut run = || -> Result<f64> {
                    let plan = &self.plans[&key];
                    let op = ConvForward {
                        conv: &plan.conv,
                        x: &plan.x,
                        w: &plan.w,
                        y: &plan.y,
                    };
                    dev.synchronize().map_err(err)?;
                    let t = std::time::Instant::now();
                    unsafe {
                        op.launch(
                            algo,
                            if need > 0 { self.ws.as_mut() } else { None },
                            (bf16::from_f32(1.0), bf16::from_f32(0.0)),
                            x,
                            w,
                            y,
                        )
                    }
                    .map_err(err)?;
                    dev.synchronize().map_err(err)?;
                    Ok(t.elapsed().as_secs_f64())
                };
                if run().is_err() {
                    continue;
                }
                let Ok(secs) = run() else { continue };
                report.push(format!("{algo:?} {:.2}ms", secs * 1e3));
                if best.is_none_or(|(b, _)| secs < b) {
                    best = Some((secs, algo));
                }
            }
            let pick = best.map_or(heuristic, |b| b.1);
            crate::wan::log::info(format_args!(
                "ltx vae conv {key:?}: heuristic {heuristic:?}; {} → {pick:?}",
                report.join(", ")
            ));
            TUNED
                .get_or_init(Default::default)
                .lock()
                .expect("tuned lock")
                .insert(key, pick);
            self.plans.remove(&key);
            Ok(())
        }
    }

    fn tune_enabled() -> bool {
        crate::wan::envflag::string_flag("FASTVIDEO_LTX_VAE_CONV_ALGO", "heuristic") == "tune"
    }
}
