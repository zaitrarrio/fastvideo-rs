//! Plain-Rust reference of madebyollin's `taehv.py`, written block by block
//! from the Python and nothing else: no `CudaTensor`, no shared helpers with
//! [`super::taehv`], `f64` accumulation, index arithmetic spelled out.
//!
//! It exists so the port can be judged against an independent reading of the
//! module: the unit tests compare [`super::taehv`]'s host path with it, and
//! `fv-gpucheck taehv-device` compares the device path with it on a live GPU
//! (where the port's own host path is refused). It runs the reference's
//! **parallel** schedule (`apply_model_with_memblocks_parallel`, whole clip,
//! every block over every frame at once), while the port streams chunks with
//! a per-block carry — the two agree only if the carry is right.
//!
//! Sources (sol-engine vendored copies, identical block code):
//! `models/ltx2.5-refiner/GB200/vendor/taehv/taehv.py` (commit 32ac014,
//! `taeltx2_3_wide`) and `models/minimax_h3/super_acceleration/vendor/taeh3/taehv.py`
//! (commit e589fdd, `taeh3` wrap). Line numbers below are the LTX copy's.

use super::taehv::TaeArch;

/// Frames-first activations: `[frames, channels, height, width]`, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Act {
    pub data: Vec<f32>,
    pub shape: [usize; 4],
}

impl Act {
    pub fn new(data: Vec<f32>, shape: [usize; 4]) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "act {shape:?}");
        Self { data, shape }
    }

    pub fn zeros(shape: [usize; 4]) -> Self {
        Self::new(vec![0.0; shape.iter().product()], shape)
    }

    fn frame_len(&self) -> usize {
        self.shape[1] * self.shape[2] * self.shape[3]
    }

    fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self::new(self.data.iter().map(|&v| f(v)).collect(), self.shape)
    }
}

/// Weight source: `(key, shape) -> values`.
pub type Weights<'a> = &'a dyn Fn(&str, &[usize]) -> Vec<f32>;

/// `nn.Conv2d(..., padding=k//2, stride, groups)` (`taehv.py:14-15` for the
/// 3x3 `conv`; 1x1 layers have no padding). `w` is `[out, in/groups, k, k]`.
pub fn conv2d(
    x: &Act,
    w: &[f32],
    out_c: usize,
    k: usize,
    bias: Option<&[f32]>,
    stride: usize,
    groups: usize,
) -> Act {
    let [n, c, h, wd] = x.shape;
    assert!(
        c.is_multiple_of(groups) && out_c.is_multiple_of(groups),
        "conv groups"
    );
    let (cg, og) = (c / groups, out_c / groups);
    assert_eq!(w.len(), out_c * cg * k * k, "conv weight");
    let pad = k / 2;
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (wd + 2 * pad - k) / stride + 1;
    let mut y = vec![0.0f32; n * out_c * oh * ow];
    for f in 0..n {
        for o in 0..out_c {
            let g = o / og;
            for yy in 0..oh {
                for xx in 0..ow {
                    let mut acc = bias.map_or(0.0, |b| f64::from(b[o]));
                    for ci in 0..cg {
                        let cin = g * cg + ci;
                        for ky in 0..k {
                            let iy = (yy * stride + ky) as isize - pad as isize;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            for kx in 0..k {
                                let ix = (xx * stride + kx) as isize - pad as isize;
                                if ix < 0 || ix >= wd as isize {
                                    continue;
                                }
                                let xv =
                                    x.data[((f * c + cin) * h + iy as usize) * wd + ix as usize];
                                let wv = w[((o * cg + ci) * k + ky) * k + kx];
                                acc += f64::from(xv) * f64::from(wv);
                            }
                        }
                    }
                    y[((f * out_c + o) * oh + yy) * ow + xx] = acc as f32;
                }
            }
        }
    }
    Act::new(y, [n, out_c, oh, ow])
}

pub fn relu(x: &Act) -> Act {
    x.map(|v| v.max(0.0))
}

/// `Clamp`: `tanh(x / 3) * 3` (`taehv.py:17-19`).
pub fn clamp_tanh(x: &Act) -> Act {
    x.map(|v| (v / 3.0).tanh() * 3.0)
}

pub fn add(a: &Act, b: &Act) -> Act {
    assert_eq!(a.shape, b.shape);
    Act::new(
        a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(),
        a.shape,
    )
}

/// `torch.cat([x, past], 1)`.
pub fn cat_channels(a: &Act, b: &Act) -> Act {
    let [n, ca, h, w] = a.shape;
    assert_eq!([b.shape[0], b.shape[2], b.shape[3]], [n, h, w]);
    let cb = b.shape[1];
    let mut out = Vec::with_capacity(n * (ca + cb) * h * w);
    for f in 0..n {
        out.extend_from_slice(&a.data[f * ca * h * w..(f + 1) * ca * h * w]);
        out.extend_from_slice(&b.data[f * cb * h * w..(f + 1) * cb * h * w]);
    }
    Act::new(out, [n, ca + cb, h, w])
}

/// Parallel-path memory (`taehv.py:83-88`): the block's own input shifted one
/// frame later, a zero frame in front.
pub fn past_of(x: &Act) -> Act {
    let fl = x.frame_len();
    let mut data = vec![0.0; x.data.len()];
    data[fl..].copy_from_slice(&x.data[..x.data.len() - fl]);
    Act::new(data, x.shape)
}

/// `nn.Upsample(scale_factor=2)` (nearest).
pub fn upsample2(x: &Act) -> Act {
    let [n, c, h, w] = x.shape;
    let mut y = vec![0.0; n * c * 4 * h * w];
    for p in 0..n * c {
        for yy in 0..2 * h {
            for xx in 0..2 * w {
                y[(p * 2 * h + yy) * 2 * w + xx] = x.data[(p * h + yy / 2) * w + xx / 2];
            }
        }
    }
    Act::new(y, [n, c, 2 * h, 2 * w])
}

/// `F.pixel_shuffle(x, r)`: out `[c, h*r+i, w*r+j] = in[c*r*r + i*r + j, h, w]`.
pub fn pixel_shuffle(x: &Act, r: usize) -> Act {
    if r == 1 {
        return x.clone();
    }
    let [n, cr, h, w] = x.shape;
    let c = cr / (r * r);
    let mut y = vec![0.0; x.data.len()];
    for f in 0..n {
        for ch in 0..c {
            for i in 0..r {
                for j in 0..r {
                    for yy in 0..h {
                        for xx in 0..w {
                            let src = ((f * cr + ch * r * r + i * r + j) * h + yy) * w + xx;
                            let dst = ((f * c + ch) * h * r + yy * r + i) * w * r + xx * r + j;
                            y[dst] = x.data[src];
                        }
                    }
                }
            }
        }
    }
    Act::new(y, [n, c, h * r, w * r])
}

/// `F.pixel_unshuffle(x, r)`, the inverse of [`pixel_shuffle`].
pub fn pixel_unshuffle(x: &Act, r: usize) -> Act {
    if r == 1 {
        return x.clone();
    }
    let [n, c, hr, wr] = x.shape;
    let (h, w) = (hr / r, wr / r);
    let mut y = vec![0.0; x.data.len()];
    for f in 0..n {
        for ch in 0..c {
            for i in 0..r {
                for j in 0..r {
                    for yy in 0..h {
                        for xx in 0..w {
                            let dst = ((f * c * r * r + ch * r * r + i * r + j) * h + yy) * w + xx;
                            let src = ((f * c + ch) * hr + yy * r + i) * wr + xx * r + j;
                            y[dst] = x.data[src];
                        }
                    }
                }
            }
        }
    }
    Act::new(y, [n, c * r * r, h, w])
}

fn conv_w(ws: Weights, key: &str, o: usize, i: usize, k: usize) -> Vec<f32> {
    ws(&format!("{key}.weight"), &[o, i, k, k])
}

fn conv_b(ws: Weights, key: &str, o: usize) -> Vec<f32> {
    ws(&format!("{key}.bias"), &[o])
}

/// `MemBlock(n, n)` (`taehv.py:21-28`): `relu(conv(cat) + x)`, three 3x3
/// convs with ReLU between. Every checkpoint here has `n_in == n_out`, so
/// `skip` is `nn.Identity`.
pub fn memblock(ws: Weights, key: &str, n: usize, x: &Act, past: &Act) -> Act {
    let h = cat_channels(x, past);
    let c0 = format!("{key}.conv.0");
    let c2 = format!("{key}.conv.2");
    let c4 = format!("{key}.conv.4");
    let h = relu(&conv2d(
        &h,
        &conv_w(ws, &c0, n, 2 * n, 3),
        n,
        3,
        Some(&conv_b(ws, &c0, n)),
        1,
        1,
    ));
    let h = relu(&conv2d(
        &h,
        &conv_w(ws, &c2, n, n, 3),
        n,
        3,
        Some(&conv_b(ws, &c2, n)),
        1,
        1,
    ));
    let h = conv2d(
        &h,
        &conv_w(ws, &c4, n, n, 3),
        n,
        3,
        Some(&conv_b(ws, &c4, n)),
        1,
        1,
    );
    relu(&add(&h, x))
}

/// `WideMemBlock(n, n)` (`taehv.py:30-43`): 1x1 (2n→n), ReLU, grouped 3x3
/// (`groups = max(1, n // 64)`), ReLU, 1x1, ReLU, grouped 3x3; then
/// `relu(. + x)` (identity skip).
pub fn wide_memblock(ws: Weights, key: &str, n: usize, x: &Act, past: &Act) -> Act {
    let g = (n / 64).max(1);
    let h = cat_channels(x, past);
    let k = |i: usize| format!("{key}.conv.{i}");
    let h = relu(&conv2d(
        &h,
        &conv_w(ws, &k(0), n, 2 * n, 1),
        n,
        1,
        Some(&conv_b(ws, &k(0), n)),
        1,
        1,
    ));
    let h = relu(&conv2d(
        &h,
        &conv_w(ws, &k(2), n, n / g, 3),
        n,
        3,
        Some(&conv_b(ws, &k(2), n)),
        1,
        g,
    ));
    let h = relu(&conv2d(
        &h,
        &conv_w(ws, &k(4), n, n, 1),
        n,
        1,
        Some(&conv_b(ws, &k(4), n)),
        1,
        1,
    ));
    let h = conv2d(
        &h,
        &conv_w(ws, &k(6), n, n / g, 3),
        n,
        3,
        Some(&conv_b(ws, &k(6), n)),
        1,
        g,
    );
    relu(&add(&h, x))
}

/// `TPool(n, stride)` (`taehv.py:45-52`): consecutive `stride` frames stacked
/// on channels, then a 1x1 conv (no bias) back to `n`.
pub fn tpool(ws: Weights, key: &str, n: usize, stride: usize, x: &Act) -> Act {
    let [f, c, h, w] = x.shape;
    assert!(f % stride == 0, "tpool {f} frames, stride {stride}");
    let stacked = Act::new(x.data.clone(), [f / stride, c * stride, h, w]);
    conv2d(
        &stacked,
        &conv_w(ws, &format!("{key}.conv"), n, n * stride, 1),
        n,
        1,
        None,
        1,
        1,
    )
}

/// `TGrow(n, stride)` (`taehv.py:54-62`): 1x1 conv (no bias) to `n * stride`,
/// then each frame's channels split into `stride` consecutive frames.
pub fn tgrow(ws: Weights, key: &str, n: usize, stride: usize, x: &Act) -> Act {
    let y = conv2d(
        x,
        &conv_w(ws, &format!("{key}.conv"), n * stride, n, 1),
        n * stride,
        1,
        None,
        1,
        1,
    );
    let [f, _, h, w] = y.shape;
    Act::new(y.data, [f * stride, n, h, w])
}

/// The decoder `nn.Sequential` in parallel mode, raw (before pixel shuffle,
/// trim and clamp). `z` is `[T, C, H, W]`.
pub fn decoder_raw(arch: TaeArch, ws: Weights, z: &Act) -> Act {
    let nf = arch.decoder_widths();
    let tu = arch.decoder_time_upscale();
    let wide = arch.wide_decoder();
    let mut x = clamp_tanh(z);
    x = relu(&conv2d(
        &x,
        &conv_w(ws, "decoder.1", nf[0], arch.latent_channels(), 3),
        nf[0],
        3,
        Some(&conv_b(ws, "decoder.1", nf[0])),
        1,
        1,
    ));
    // Stages start at indices 3, 9, 15: three memblocks, upsample, TGrow, conv.
    for s in 0..3 {
        let base = 3 + 6 * s;
        for i in base..base + 3 {
            let past = past_of(&x);
            let key = format!("decoder.{i}");
            x = if wide {
                wide_memblock(ws, &key, nf[s], &x, &past)
            } else {
                memblock(ws, &key, nf[s], &x, &past)
            };
        }
        x = upsample2(&x);
        x = tgrow(ws, &format!("decoder.{}", base + 4), nf[s], tu[s], &x);
        let key = format!("decoder.{}", base + 5);
        x = conv2d(
            &x,
            &conv_w(ws, &key, nf[s + 1], nf[s], 3),
            nf[s + 1],
            3,
            None,
            1,
            1,
        );
    }
    x = relu(&x);
    let out = 3 * arch.patch_size() * arch.patch_size();
    conv2d(
        &x,
        &conv_w(ws, "decoder.22", out, nf[3], 3),
        out,
        3,
        Some(&conv_b(ws, "decoder.22", out)),
        1,
        1,
    )
}

/// `decode_video` (`taehv.py:282-300`; H3: `_decode_h3_video`, taeh3 copy
/// `taehv.py:279-288`). Returns `[F, 3, H, W]` in the reference's `[0, 1]`.
pub fn decode(arch: TaeArch, ws: Weights, z: &Act) -> Act {
    let raw = decoder_raw(arch, ws, z);
    let trim = arch.t_upscale() - 1;
    let x = match arch {
        TaeArch::H3 => {
            // Pad to a multiple of 5*t_upscale with zeros, drop the first
            // `trim` frames of each group, then the last 3*t_upscale.
            let group = 5 * arch.t_upscale();
            let fl = raw.frame_len();
            let [f, c, h, w] = raw.shape;
            let pad = (group - f % group) % group;
            let mut data = raw.data.clone();
            data.extend(std::iter::repeat_n(0.0, pad * fl));
            let groups = (f + pad) / group;
            let mut kept = Vec::new();
            for g in 0..groups {
                for i in trim..group {
                    let at = (g * group + i) * fl;
                    kept.extend_from_slice(&data[at..at + fl]);
                }
            }
            let n = kept.len() / fl - 3 * arch.t_upscale();
            kept.truncate(n * fl);
            pixel_shuffle(&Act::new(kept, [n, c, h, w]), arch.patch_size())
        }
        _ => {
            let x = pixel_shuffle(&raw, arch.patch_size());
            let fl = x.frame_len();
            let [f, c, h, w] = x.shape;
            Act::new(x.data[trim * fl..].to_vec(), [f - trim, c, h, w])
        }
    };
    x.map(|v| v.clamp(0.0, 1.0))
}

/// `encode_video` (`taehv.py:259-275`) on `[F, 3, H, W]` RGB in `[0, 1]`:
/// pad the end to a multiple of `t_downscale` by repeating the last frame,
/// pixel-unshuffle, then the encoder `nn.Sequential` (`taehv.py:206-212`).
/// Returns the latent `[T, C, H / (8 * patch), W / (8 * patch)]`.
pub fn encode(arch: TaeArch, ws: Weights, pixels: &Act) -> Act {
    let td = arch.encoder_time_downscale();
    let t_down: usize = td.iter().product();
    let fl = pixels.frame_len();
    let [f, c, h, w] = pixels.shape;
    let pad = (t_down - f % t_down) % t_down;
    let mut data = pixels.data.clone();
    let last = data[(f - 1) * fl..].to_vec();
    for _ in 0..pad {
        data.extend_from_slice(&last);
    }
    let mut x = pixel_unshuffle(&Act::new(data, [f + pad, c, h, w]), arch.patch_size());
    let cin = 3 * arch.patch_size() * arch.patch_size();
    x = relu(&conv2d(
        &x,
        &conv_w(ws, "encoder.0", 64, cin, 3),
        64,
        3,
        Some(&conv_b(ws, "encoder.0", 64)),
        1,
        1,
    ));
    // Stages at 2, 7, 12: TPool, stride-2 conv, three memblocks.
    for (s, &stride) in td.iter().enumerate() {
        let base = 2 + 5 * s;
        x = tpool(ws, &format!("encoder.{base}"), 64, stride, &x);
        x = conv2d(
            &x,
            &conv_w(ws, &format!("encoder.{}", base + 1), 64, 64, 3),
            64,
            3,
            None,
            2,
            1,
        );
        for i in base + 2..base + 5 {
            let past = past_of(&x);
            x = memblock(ws, &format!("encoder.{i}"), 64, &x, &past);
        }
    }
    let lc = arch.latent_channels();
    conv2d(
        &x,
        &conv_w(ws, "encoder.17", lc, 64, 3),
        lc,
        3,
        Some(&conv_b(ws, "encoder.17", lc)),
        1,
        1,
    )
}
