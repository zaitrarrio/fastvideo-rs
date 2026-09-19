//! Kernel-level parity on a live GPU: every NVRTC kernel, cuBLAS and cuDNN
//! path the Wan graph uses, called through its public device wrapper and
//! compared with an independent plain-Rust reference. Also times the two
//! conv3d strategies at VAE sizes and checks that GPU runs refuse to compute
//! on the host.
//!
//! Shapes deliberately include non-power-of-two widths, partial reduction
//! blocks, chunked attention, rank-6 permutes and B=2.

use cudarc::driver::CudaSlice;
use fastvideo_cudarc::wan::device::GemmMath;
use fastvideo_cudarc::wan::fused::Rope;
use fastvideo_cudarc::wan::ops::{self, host, BcastOp};
use fastvideo_cudarc::wan::nn::Linear;
use fastvideo_cudarc::wan::{attn, conv, device, kernels as k, vsa};
use fastvideo_cudarc::CudaTensor;
use serde_json::json;

use crate::metrics::diff;
use crate::mode::Limits;
use crate::rand_weights::randn;
use crate::report::{Report, StageError, StageResult};

fn dev() -> anyhow::Result<std::sync::Arc<device::DeviceContext>> {
    device::global_device().ok_or_else(|| anyhow::anyhow!("no live CUDA device"))
}

fn up(v: &[f32]) -> anyhow::Result<CudaSlice<f32>> {
    Ok(dev()?.stream.memcpy_stod(v)?)
}

fn down(s: &CudaSlice<f32>) -> anyhow::Result<Vec<f32>> {
    Ok(dev()?.stream.memcpy_dtov(s)?)
}

fn t(data: Vec<f32>, shape: &[usize]) -> anyhow::Result<CudaTensor> {
    Ok(CudaTensor::from_vec(data, shape.to_vec())?.to_device()?)
}

fn host_of(x: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    Ok(x.host_cow()?.into_owned())
}

struct Ctx<'r> {
    report: &'r mut Report,
    seed: u64,
}

impl Ctx<'_> {
    fn rand(&mut self, n: usize, std: f32) -> Vec<f32> {
        self.seed += 1;
        randn(self.seed, n, std)
    }

    fn cmp(&mut self, name: &str, got: &[f32], want: &[f32], limit: f64) -> StageResult<()> {
        let d = diff(got, want);
        self.report.check(name, d.within(limit), d.to_json(), json!({"rel_l2": limit}))
    }
}

// ---- independent references (f64 accumulation, index math) ---------------

fn ref_softmax(x: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (row, o) in x.chunks(width).zip(out.chunks_mut(width)) {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s: f64 = row.iter().map(|&v| f64::from(v - m).exp()).sum();
        for (oi, &v) in o.iter_mut().zip(row) {
            *oi = (f64::from(v - m).exp() / s) as f32;
        }
    }
    out
}

fn ref_rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let width = w.len();
    let mut out = vec![0.0; x.len()];
    for (row, o) in x.chunks(width).zip(out.chunks_mut(width)) {
        let ms = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / width as f64;
        let inv = 1.0 / (ms + f64::from(eps)).sqrt();
        for i in 0..width {
            o[i] = (f64::from(row[i]) * inv) as f32 * w[i];
        }
    }
    out
}

fn ref_layer_norm(x: &[f32], width: usize, affine: Option<(&[f32], &[f32])>, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (row, o) in x.chunks(width).zip(out.chunks_mut(width)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / width as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for i in 0..width {
            let y = ((f64::from(row[i]) - mean) * inv) as f32;
            o[i] = match affine {
                Some((w, b)) => y * w[i] + b[i],
                None => y,
            };
        }
    }
    out
}

fn ref_gelu_tanh(v: f32) -> f32 {
    let t = 0.797_884_6 * (v + 0.044715 * v * v * v);
    0.5 * v * (1.0 + t.tanh())
}

/// Row-major `a[m,k] @ b[k,n]`.
fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let acc: f64 = (0..k).map(|t| f64::from(a[i * k + t]) * f64::from(b[t * n + j])).sum();
            out[i * n + j] = acc as f32;
        }
    }
    out
}

fn transpose2(w: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = w[i * c + j];
        }
    }
    out
}

/// `[bh, sq, d] × [bh, sk, d]` scaled dot-product attention.
#[allow(clippy::too_many_arguments)]
fn ref_sdpa(q: &[f32], kk: &[f32], v: &[f32], bh: usize, sq: usize, sk: usize, d: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0; bh * sq * d];
    let mut scores = vec![0.0f32; sk];
    for h in 0..bh {
        for i in 0..sq {
            let qi = &q[(h * sq + i) * d..(h * sq + i + 1) * d];
            for (j, s) in scores.iter_mut().enumerate() {
                let kj = &kk[(h * sk + j) * d..(h * sk + j + 1) * d];
                *s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let p = ref_softmax(&scores, sk);
            let o = &mut out[(h * sq + i) * d..(h * sq + i + 1) * d];
            for (j, pj) in p.iter().enumerate() {
                let vj = &v[(h * sk + j) * d..(h * sk + j + 1) * d];
                for t in 0..d {
                    o[t] += pj * vj[t];
                }
            }
        }
    }
    out
}

/// Naive 3-D cross-correlation (`kt=1` gives conv2d) with symmetric padding.
#[allow(clippy::too_many_arguments)]
fn ref_conv3d(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    (n, ic, it, ih, iw): (usize, usize, usize, usize, usize),
    (oc, kt, kh, kw): (usize, usize, usize, usize),
    (pt, ph, pw): (usize, usize, usize),
    (st, sh, sw): (usize, usize, usize),
) -> (Vec<f32>, [usize; 3]) {
    let (ot, oh, ow) = ((it + 2 * pt - kt) / st + 1, (ih + 2 * ph - kh) / sh + 1, (iw + 2 * pw - kw) / sw + 1);
    let mut out = vec![0.0; n * oc * ot * oh * ow];
    for b in 0..n {
        for o in 0..oc {
            for tt in 0..ot {
                for y in 0..oh {
                    for xx in 0..ow {
                        let mut acc = f64::from(bias.map_or(0.0, |bs| bs[o]));
                        for c in 0..ic {
                            for dt in 0..kt {
                                for dy in 0..kh {
                                    for dx in 0..kw {
                                        let (zt, zy, zx) = (tt * st + dt, y * sh + dy, xx * sw + dx);
                                        if zt < pt || zy < ph || zx < pw || zt - pt >= it || zy - ph >= ih || zx - pw >= iw {
                                            continue;
                                        }
                                        let xi = (((b * ic + c) * it + zt - pt) * ih + zy - ph) * iw + zx - pw;
                                        let wi = (((o * ic + c) * kt + dt) * kh + dy) * kw + dx;
                                        acc += f64::from(x[xi]) * f64::from(w[wi]);
                                    }
                                }
                            }
                        }
                        out[(((b * oc + o) * ot + tt) * oh + y) * ow + xx] = acc as f32;
                    }
                }
            }
        }
    }
    (out, [ot, oh, ow])
}

fn ref_permute(x: &[f32], shape: &[usize], perm: &[usize]) -> Vec<f32> {
    let rank = shape.len();
    let strides: Vec<usize> = (0..rank).map(|i| shape[i + 1..].iter().product()).collect();
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let mut out = vec![0.0; x.len()];
    for (i, o) in out.iter_mut().enumerate() {
        let (mut rem, mut src) = (i, 0usize);
        for k in (0..rank).rev() {
            src += (rem % out_shape[k]) * strides[perm[k]];
            rem /= out_shape[k];
        }
        *o = x[src];
    }
    out
}

// ---- suite ----------------------------------------------------------------

/// Run one kernel group. An error inside a group becomes a failed check, so
/// with `--keep-going` the remaining groups still run.
fn group(c: &mut Ctx<'_>, name: &str, f: impl FnOnce(&mut Ctx<'_>) -> StageResult<()>) -> StageResult<()> {
    match f(c) {
        Err(StageError::Error(e)) => c.report.check(format!("{name}/completed"), false, json!({"error": format!("{e:#}")}), json!({})),
        other => other,
    }
}

pub fn run(report: &mut Report, lim: Limits, seed: u64) -> StageResult<()> {
    let info = crate::gpu::init("cuda")?;
    report.set("device", &info);
    report.set("limits", lim);
    let math = dev()?.gemm_math;
    report.set("gemm_math", format!("{math:?}"));
    let mut c = Ctx { report, seed };
    let op = lim.op;
    // cuBLAS math: TF32 perturbs F32 results; bf16 compute more so.
    let gemm = op.max(match math {
        GemmMath::F32 => 0.0,
        GemmMath::Tf32 => 2e-3,
        GemmMath::Bf16 => 5e-3,
    });

    group(&mut c, "elementwise", |c| {
        for n in [1usize, 1000, 65_537] {
            let (a, b, z) = (c.rand(n, 1.0), c.rand(n, 1.0), c.rand(n, 1.0));
            let (da, db, dz) = (up(&a)?, up(&b)?, up(&z)?);
            for (kind, name, f) in [
                (ops::ElemBinary::Add, "add", (|x: f32, y: f32| x + y) as fn(f32, f32) -> f32),
                (ops::ElemBinary::Sub, "sub", |x, y| x - y),
                (ops::ElemBinary::Mul, "mul", |x, y| x * y),
            ] {
                let got = down(&ops::elem_binary_device(&da, &db, kind)?)?;
                let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect();
                c.cmp(&format!("elem_{name}_n{n}"), &got, &want, op)?;
            }
            c.cmp(&format!("mul_scalar_n{n}"), &down(&ops::mul_scalar_device(&da, -1.75)?)?, &a.iter().map(|v| v * -1.75).collect::<Vec<_>>(), op)?;
            c.cmp(&format!("add_scalar_n{n}"), &down(&ops::add_scalar_device(&da, 0.5)?)?, &a.iter().map(|v| v + 0.5).collect::<Vec<_>>(), op)?;
            let want: Vec<f32> = a.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
            c.cmp(&format!("silu_n{n}"), &down(&ops::unary_device(&da, ops::ElemUnary::Silu)?)?, &want, op)?;
            let want: Vec<f32> = a.iter().map(|&v| ref_gelu_tanh(v)).collect();
            c.cmp(&format!("gelu_tanh_n{n}"), &down(&ops::unary_device(&da, ops::ElemUnary::GeluTanh)?)?, &want, op)?;
            c.cmp(&format!("clamp_n{n}"), &down(&ops::clamp_device(&da, -0.3, 0.7)?)?, &a.iter().map(|v| v.clamp(-0.3, 0.7)).collect::<Vec<_>>(), op)?;
            c.cmp(&format!("fill_n{n}"), &down(&ops::fill_device(n, 0.25)?)?, &vec![0.25; n], 0.0)?;
            // Five terms: one lincomb3 launch plus one chained launch.
            let coefs = [0.7f32, -1.3, 0.25, 2.0, -0.5];
            let bufs = [&da, &db, &dz, &da, &db];
            let terms: Vec<(f32, &CudaSlice<f32>)> = coefs.iter().copied().zip(bufs).collect();
            let want: Vec<f32> = (0..n).map(|i| {
                let v = [a[i], b[i], z[i], a[i], b[i]];
                coefs.iter().zip(v).map(|(k, x)| f64::from(*k) * f64::from(x)).sum::<f64>() as f32
            }).collect();
            c.cmp(&format!("lincomb5_n{n}"), &down(&ops::lincomb_device(&terms)?)?, &want, op)?;
        }
        // Broadcast binary: trailing repeat, channel broadcast, leading batch.
        for (big_shape, inner, period) in [(vec![2usize, 6, 7], 1usize, 42usize), (vec![2, 5, 3, 4], 12, 5), (vec![3, 4, 5], 1, 5)] {
            let nb: usize = big_shape.iter().product();
            let (big, small) = (c.rand(nb, 1.0), c.rand(period, 1.0).iter().map(|v| v + 2.0).collect::<Vec<_>>());
            let (dbig, dsmall) = (up(&big)?, up(&small)?);
            for bop in [BcastOp::Add, BcastOp::Sub, BcastOp::Mul, BcastOp::Div, BcastOp::RSub, BcastOp::RDiv] {
                let got = down(&ops::bcast_binary_device(&dbig, &dsmall, inner, period, bop)?)?;
                let want: Vec<f32> = (0..nb).map(|i| bop.apply(big[i], small[(i / inner) % period])).collect();
                c.cmp(&format!("bcast_{bop:?}_{big_shape:?}"), &got, &want, op)?;
            }
        }
        Ok(())
    })?;

    group(&mut c, "row_reductions", |c| {
        // Widths straddle the 256-thread block and non-powers of 2.
        for (rows, width) in [(1usize, 7usize), (5, 64), (33, 255), (3, 256), (4, 300), (2, 1536), (2, 4096)] {
            let x = c.rand(rows * width, 2.0);
            let dx = up(&x)?;
            c.cmp(&format!("softmax_{rows}x{width}"), &down(&ops::softmax_last_device(&dx, width)?)?, &ref_softmax(&x, width), op)?;
            // bf16 probabilities for dense attention: same softmax, half the
            // bytes, so it is held to bf16 round-off rather than `op`.
            let got: Vec<f32> = dev()?
                .stream
                .memcpy_dtov(&ops::softmax_last_bf16_device(&dx, width)?)?
                .iter()
                .map(|v: &half::bf16| v.to_f32())
                .collect();
            c.cmp(&format!("softmax_bf16_{rows}x{width}"), &got, &ref_softmax(&x, width), 8e-3)?;
            let w: Vec<f32> = c.rand(width, 0.1).iter().map(|v| 1.0 + v).collect();
            let b = c.rand(width, 0.1);
            let (dw, db) = (up(&w)?, up(&b)?);
            c.cmp(&format!("rms_norm_{rows}x{width}"), &down(&ops::rms_norm_last_device(&dx, &dw, 1e-6)?)?, &ref_rms(&x, &w, 1e-6), op)?;
            let got = down(&ops::layer_norm_last_device(&dx, Some((&dw, &db)), width, 1e-6)?)?;
            c.cmp(&format!("layer_norm_affine_{rows}x{width}"), &got, &ref_layer_norm(&x, width, Some((&w, &b)), 1e-6), op)?;
            let got = down(&ops::layer_norm_last_device(&dx, None, width, 1e-6)?)?;
            c.cmp(&format!("layer_norm_plain_{rows}x{width}"), &got, &ref_layer_norm(&x, width, None, 1e-6), op)?;
        }
        Ok(())
    })?;

    group(&mut c, "dit_fused", |c| {
        for (b, s, d) in [(1usize, 17usize, 64usize), (2, 300, 1536)] {
            let x = c.rand(b * s * d, 1.0);
            let e = c.rand(b * 6 * d, 0.2);
            let upd = c.rand(b * s * d, 1.0);
            let (xt, et, ut) = (t(x.clone(), &[b, s, d])?, t(e.clone(), &[b, 6, d])?, t(upd.clone(), &[b, s, d])?);
            let normed = ref_layer_norm(&x, d, None, 1e-6);
            let mut want = vec![0.0; x.len()];
            let mut want_gate = vec![0.0; x.len()];
            for i in 0..x.len() {
                let (bi, j) = (i / (s * d), i % d);
                want[i] = normed[i] * (1.0 + e[(bi * 6 + 4) * d + j]) + e[(bi * 6 + 3) * d + j];
                want_gate[i] = x[i] + upd[i] * e[(bi * 6 + 5) * d + j];
            }
            c.cmp(&format!("ln_adaln_e_{b}x{s}x{d}"), &host_of(&xt.ln_adaln_e(&et, 4, 3, 1e-6)?)?, &want, op)?;
            c.cmp(&format!("residual_gate_add_e_{b}x{s}x{d}"), &host_of(&xt.residual_gate_add_e(&ut, &et, 5)?)?, &want_gate, op)?;

            let bias = c.rand(d, 0.5);
            let mut out = up(&x)?;
            ops::add_bias_inplace_device(&mut out, &up(&bias)?, 1)?;
            c.cmp(&format!("add_bias_last_{b}x{s}x{d}"), &down(&out)?, &x.iter().enumerate().map(|(i, v)| v + bias[i % d]).collect::<Vec<_>>(), op)?;
            let mut out = up(&x)?;
            ops::bias_gelu_inplace_device(&mut out, &up(&bias)?)?;
            let want: Vec<f32> = x.iter().enumerate().map(|(i, v)| ref_gelu_tanh(v + bias[i % d])).collect();
            c.cmp(&format!("bias_gelu_{b}x{s}x{d}"), &down(&out)?, &want, op)?;
        }
        // q/k prep from a fused QKV row: [b, s, 3*heads*hd].
        for (b, s, heads, hd) in [(1usize, 9usize, 2usize, 8usize), (2, 128, 12, 128)] {
            let width = heads * hd;
            let x = c.rand(b * s * 3 * width, 1.0);
            let w: Vec<f32> = c.rand(width, 0.1).iter().map(|v| 1.0 + v).collect();
            let ang = c.rand(s * hd, 3.0);
            let (cos, sin): (Vec<f32>, Vec<f32>) = ang.iter().map(|a| (a.cos(), a.sin())).unzip();
            let xt = t(x.clone(), &[b, s, 3 * width])?;
            let (wt, ct, st) = (t(w.clone(), &[width])?, t(cos.clone(), &[s, hd])?, t(sin.clone(), &[s, hd])?);
            for (col, rope) in [(0usize, true), (1, true), (1, false)] {
                let got = xt.qk_norm_rope_bhsd(col * width, heads, &wt, rope.then_some(Rope { cos: &ct, sin: &st }), 1e-6)?;
                let slice: Vec<f32> = (0..b * s).flat_map(|r| x[r * 3 * width + col * width..r * 3 * width + (col + 1) * width].to_vec()).collect();
                let n = ref_rms(&slice, &w, 1e-6);
                let mut want = vec![0.0; b * heads * s * hd];
                for bi in 0..b {
                    for si in 0..s {
                        for h in 0..heads {
                            for p in 0..hd {
                                let src = (bi * s + si) * width + h * hd + p;
                                let o = ((bi * heads + h) * s + si) * hd + p;
                                want[o] = if rope {
                                    let even = p & !1;
                                    let (x1, x2) = (n[src - (p & 1)], n[src - (p & 1) + 1]);
                                    let (cs, sn) = (cos[si * hd + even], sin[si * hd + even + 1]);
                                    if p & 1 == 1 { x1 * sn + x2 * cs } else { x1 * cs - x2 * sn }
                                } else {
                                    n[src]
                                };
                            }
                        }
                    }
                }
                c.cmp(&format!("qk_norm_rope_bhsd_col{col}_rope{rope}_{b}x{s}x{heads}x{hd}"), &host_of(&got)?, &want, op)?;
            }
            let v = xt.split_heads_bhsd(2 * width, heads, hd)?;
            let merged = host_of(&v.merge_heads()?)?;
            let want: Vec<f32> = (0..b * s).flat_map(|r| x[r * 3 * width + 2 * width..r * 3 * width + 3 * width].to_vec()).collect();
            c.cmp(&format!("split_merge_heads_{b}x{s}x{heads}x{hd}"), &merged, &want, 0.0)?;
        }
        Ok(())
    })?;

    group(&mut c, "data_movement", |c| {
        for (shape, perm) in [
            (vec![2usize, 3, 5, 7], vec![0usize, 2, 1, 3]),
            (vec![2, 3, 5, 7], vec![3, 2, 1, 0]),
            (vec![1, 4, 3, 6, 5], vec![0, 2, 1, 3, 4]),
            (vec![2, 3, 2, 4, 2, 3], vec![0, 5, 1, 3, 2, 4]),
            (vec![6, 2, 3, 4, 2, 5], vec![0, 1, 5, 2, 4, 3]),
        ] {
            let x = c.rand(shape.iter().product(), 1.0);
            let got = down(&ops::gather_nd_device(&up(&x)?, &shape, &perm)?)?;
            c.cmp(&format!("gather_nd_{shape:?}_{perm:?}"), &got, &ref_permute(&x, &shape, &perm), 0.0)?;
        }
        // Tensor-level narrow / cat / pad on every axis (block_copy).
        let shape = [2usize, 3, 4, 5, 6];
        let x = c.rand(shape.iter().product(), 1.0);
        let xt = t(x.clone(), &shape)?;
        for dim in 0..shape.len() {
            let d = shape[dim];
            let (outer, inner): (usize, usize) = (shape[..dim].iter().product(), shape[dim + 1..].iter().product());
            let a = xt.narrow(dim, 1, d - 1)?;
            let mut want = Vec::new();
            for o in 0..outer {
                want.extend_from_slice(&x[(o * d + 1) * inner..(o + 1) * d * inner]);
            }
            c.cmp(&format!("narrow_dim{dim}"), &host_of(&a)?, &want, 0.0)?;
            let first = xt.narrow(dim, 0, 1)?;
            c.cmp(&format!("cat_dim{dim}"), &host_of(&CudaTensor::cat(&[&first, &a], dim)?)?, &x, 0.0)?;
            let padded = xt.pad_zeros(dim, 2, 1)?;
            let mut want = vec![0.0f32; outer * (d + 3) * inner];
            for o in 0..outer {
                let dst = (o * (d + 3) + 2) * inner;
                want[dst..dst + d * inner].copy_from_slice(&x[o * d * inner..(o + 1) * d * inner]);
            }
            c.cmp(&format!("pad_dim{dim}"), &host_of(&padded)?, &want, 0.0)?;
        }
        {
            let (nc, h, w) = (6usize, 7usize, 9usize);
            let x = c.rand(nc * h * w, 1.0);
            let got = down(&ops::upsample_nearest_device(&up(&x)?, nc, h, w, 2, 2)?)?;
            c.cmp("upsample_nearest_2x", &got, &host::upsample_nearest(&x, nc, h, w, 2, 2), 0.0)?;
            let mut want = vec![0.0; nc * 4 * h * w];
            for ci in 0..nc {
                for y in 0..2 * h {
                    for xx in 0..2 * w {
                        want[(ci * 2 * h + y) * 2 * w + xx] = x[(ci * h + y / 2) * w + xx / 2];
                    }
                }
            }
            c.cmp("upsample_nearest_2x_ref", &got, &want, 0.0)?;
        }
        {
            let (n, ch, spatial) = (2usize, 12usize, 5 * 9 * 11);
            let x = c.rand(n * ch * spatial, 1.0);
            let g: Vec<f32> = c.rand(ch, 0.1).iter().map(|v| 1.0 + v).collect();
            let got = down(&ops::rms_norm_channels_device(&up(&x)?, &up(&g)?, n, ch, spatial, 1e-12, false)?)?;
            let mut want = vec![0.0; x.len()];
            for ni in 0..n {
                for s in 0..spatial {
                    let acc: f64 = (0..ch).map(|ci| f64::from(x[(ni * ch + ci) * spatial + s]).powi(2)).sum();
                    let inv = 1.0 / (acc / ch as f64 + 1e-12).sqrt();
                    for ci in 0..ch {
                        let i = (ni * ch + ci) * spatial + s;
                        want[i] = (f64::from(x[i]) * inv) as f32 * g[ci];
                    }
                }
            }
            c.cmp("rms_norm_channels", &got, &want, op)?;
            // Folding SiLU into the norm must not move the result: it is the
            // same expression, written in one pass instead of two.
            let fused = down(&ops::rms_norm_channels_device(&up(&x)?, &up(&g)?, n, ch, spatial, 1e-12, true)?)?;
            let want_silu: Vec<f32> = want.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
            c.cmp("rms_norm_channels_silu", &fused, &want_silu, op)?;
        }
        {
            let (v, d) = (50usize, 33usize);
            let table = c.rand(v * d, 1.0);
            let idx: Vec<usize> = (0..17).map(|i| (i * 13 + 7) % v).collect();
            let got = host_of(&t(table.clone(), &[v, d])?.index_select_rows(&idx)?)?;
            let want: Vec<f32> = idx.iter().flat_map(|&i| table[i * d..(i + 1) * d].to_vec()).collect();
            c.cmp("index_select_rows", &got, &want, 0.0)?;
        }
        Ok(())
    })?;

    group(&mut c, "gemm", |c| {
        for (m, kk, n) in [(1usize, 1usize, 1usize), (37, 64, 19), (512, 1536, 1536), (300, 1536, 8960)] {
            let (a, b) = (c.rand(m * kk, 1.0), c.rand(kk * n, 0.05));
            let mut out = up(&vec![0.0; m * n])?;
            device::matmul_2d_f32_device(&up(&a)?, &up(&b)?, &mut out, m, kk, n)?;
            c.cmp(&format!("gemm_2d_{m}x{kk}x{n}"), &down(&out)?, &ref_matmul(&a, &b, m, kk, n), gemm)?;
            let w = c.rand(n * kk, 0.05);
            let mut out = up(&vec![0.0; m * n])?;
            device::matmul_linear_wt_device(&up(&a)?, &up(&w)?, &mut out, m, kk, n)?;
            c.cmp(&format!("gemm_linear_{m}x{kk}x{n}"), &down(&out)?, &ref_matmul(&a, &transpose2(&w, n, kk), m, kk, n), gemm)?;
        }
        {
            let (batch, m, kk, n) = (24usize, 50usize, 32usize, 70usize);
            let (a, b) = (c.rand(batch * m * kk, 1.0), c.rand(batch * kk * n, 0.2));
            let mut out = up(&vec![0.0; batch * m * n])?;
            device::matmul_2d_strided_batched(&up(&a)?, &up(&b)?, &mut out, batch, m, kk, n)?;
            let want: Vec<f32> = (0..batch).flat_map(|bi| ref_matmul(&a[bi * m * kk..(bi + 1) * m * kk], &b[bi * kk * n..(bi + 1) * kk * n], m, kk, n)).collect();
            c.cmp("gemm_strided_batched", &down(&out)?, &want, gemm)?;
        }
        {
            // bf16 casts: round-trip error is bf16 precision; bias+GELU fused on the way out.
            let x = c.rand(100_002, 3.0); // multiple of the bias width
            let bias = c.rand(7, 0.5);
            let x16 = ops::cast_f32_bf16_device(&up(&x)?)?;
            let bits: Vec<f32> = dev()?.stream.memcpy_dtov(&x16)?.iter().map(|v| v.to_f32()).collect();
            let want: Vec<f32> = x.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
            c.cmp("cast_f32_bf16_matches_half", &bits, &want, 0.0)?;
            let got = down(&ops::cast_bf16_f32_bias_act_device(&x16, Some(&up(&bias)?), true)?)?;
            let want: Vec<f32> = want.iter().enumerate().map(|(i, v)| ref_gelu_tanh(v + bias[i % 7])).collect();
            c.cmp("cast_bf16_f32_bias_gelu", &got, &want, op)?;
            let got = down(&ops::cast_bf16_f32_bias_act_device(&x16, None, false)?)?;
            c.cmp("cast_bf16_f32_plain", &got, &bits, 0.0)?;
        }
        {
            // Ops the decoder-only text encoders and the audio decoders add.
            // Each is held to its host twin, which the unit tests hold to an
            // independent formula, so the chain reaches the kernel.
            use fastvideo_cudarc::wan::ops::host;
            let x = c.rand(4099, 2.0);
            let got = down(&ops::unary_device(&up(&x)?, ops::ElemUnary::GeluErf)?)?;
            c.cmp("gelu_erf", &got, &host::map1(&x, host::gelu_erf), op)?;
            let got = down(&ops::leaky_relu_device(&up(&x)?, 0.1)?)?;
            c.cmp("leaky_relu", &got, &host::map1(&x, |v| host::leaky_relu(v, 0.1)), 0.0)?;

            let (n, ch, l) = (2usize, 7usize, 129usize);
            let x = c.rand(n * ch * l, 1.5);
            let alpha: Vec<f32> = c.rand(ch, 0.3).iter().map(|v| (1.0 + v).abs() + 0.1).collect();
            let inv_beta: Vec<f32> = alpha.iter().map(|a| 1.0 / (a + 1e-9)).collect();
            let got = down(&ops::snake_beta_device(&up(&x)?, &up(&alpha)?, &up(&inv_beta)?, ch, l)?)?;
            c.cmp("snake_beta", &got, &host::snake_beta(&x, &alpha, &inv_beta, ch, l), op)?;

            // Qwen3 / H3 head shape: D = 128 with 96 rotated channels, and the
            // full-rotary case.
            for (b, h, s, d, r) in [(1usize, 8usize, 77usize, 128usize, 96usize), (2, 4, 33, 64, 64)] {
                let x = c.rand(b * h * s * d, 1.0);
                let ang = c.rand(s * r, 3.0);
                let (cs, sn): (Vec<f32>, Vec<f32>) = ang.iter().map(|a| (a.cos(), a.sin())).unzip();
                let got = down(&ops::rope_half_device(&up(&x)?, &up(&cs)?, &up(&sn)?, s, d, r)?)?;
                c.cmp(&format!("rope_half_d{d}_r{r}"), &got, &host::rope_half(&x, &cs, &sn, s, d, r), op)?;
            }
            let (b, hkv, s, d, rep) = (2usize, 8usize, 19usize, 128usize, 8usize);
            let x = c.rand(b * hkv * s * d, 1.0);
            let got = down(&ops::repeat_kv_device(&up(&x)?, hkv, rep, s * d)?)?;
            c.cmp("repeat_kv_8x8", &got, &host::repeat_kv(&x, hkv, rep, s * d), 0.0)?;
        }
        {
            // Weight-only FP8 with per-row scales (a resident Qwen3-VL beside the
            // H3 DiT). Codes and scales must equal the host quantizer's EXACTLY —
            // the host twin is what the unit tests hold to the format — and the
            // dequantized bf16 weight must equal the host's, bit for bit.
            use fastvideo_cudarc::wan::ops::host;
            for (rows, cols) in [(64usize, 5120usize), (257, 1000), (3, 25600)] {
                // Rows of very different magnitude, plus one dead row.
                let mut w = c.rand(rows * cols, 1.0);
                for (r, row) in w.chunks_mut(cols).enumerate() {
                    let g = if r == 1 { 0.0 } else { 10f32.powi((r % 7) as i32 - 3) };
                    row.iter_mut().for_each(|v| *v *= g);
                }
                let (q, scales) = ops::fp8_rows_quantize_device(&up(&w)?, rows, cols)?;
                let (hq, hs) = host::fp8_rows_quantize(&w, rows, cols);
                let dq = dev()?.stream.memcpy_dtov(&q)?;
                let mism = dq.iter().zip(&hq).filter(|(a, b)| a != b).count();
                c.report.check(
                    &format!("fp8_rows_codes_{rows}x{cols}"),
                    mism == 0,
                    serde_json::json!({"mismatched_codes": mism, "n": dq.len()}),
                    serde_json::json!({"mismatched_codes": 0}),
                )?;
                c.cmp(&format!("fp8_rows_scales_{rows}x{cols}"), &down(&scales)?, &hs, 0.0)?;
                let w16: Vec<f32> = dev()?
                    .stream
                    .memcpy_dtov(&ops::fp8_rows_dequant_bf16_device(&q, &scales, cols)?)?
                    .iter()
                    .map(|v: &half::bf16| v.to_f32())
                    .collect();
                c.cmp(&format!("fp8_rows_dequant_bf16_{rows}x{cols}"), &w16, &host::fp8_rows_dequant(&hq, &hs, cols), 0.0)?;
            }
        }
        {
            // The streamed text encoder stages layer i+1 (pinned memory, copy
            // stream) while layer i computes. Same bits as the plain path, for
            // a stored-bf16 and a stored-f32 checkpoint.
            let dir = std::env::temp_dir().join(format!("fv-gpucheck-prefetch-{}", std::process::id()));
            // Prefetching needs bf16 linears, which exact mode turns off: the
            // fast-mode kernels stage is where this runs.
            let stored_kinds: &[bool] = if fastvideo_cudarc::wan::nn::bf16_linears_active() { &[true, false] } else { &[] };
            for &stored in stored_kinds {
                let r = fastvideo_cudarc::llm::prefetch_self_check(&dir, stored)?;
                c.report.check(
                    &format!("llm_prefetch_equals_streamed_{}", r.stored),
                    r.max_abs_diff == 0.0,
                    serde_json::json!({"max_abs_diff": r.max_abs_diff, "elements": r.elements}),
                    serde_json::json!({"max_abs_diff": 0.0}),
                )?;
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
        {
            // Padding modes and GroupNorm for the VAE decoders. GroupNorm at a
            // group size in the millions is the case that needs the f64
            // reduction; a small group exercises the narrow-block path.
            use fastvideo_cudarc::wan::ops::{host, PadMode};
            let (outer, len, inner) = (6usize, 37usize, 29usize);
            let x = c.rand(outer * len * inner, 1.0);
            for (mode, name) in [(PadMode::Zeros, "zeros"), (PadMode::Reflect, "reflect"), (PadMode::Replicate, "replicate")] {
                let got = down(&ops::pad_axis_device(&up(&x)?, len, inner, 5, 3, mode)?)?;
                c.cmp(&format!("pad_axis_{name}"), &got, &host::pad_axis(&x, len, inner, 5, 3, mode), 0.0)?;
            }
            for (n, ch, spatial, groups) in [(1usize, 128usize, 4 * 96 * 168usize, 32usize), (3, 32, 7, 32), (2, 512, 1024, 32)] {
                // An offset mean makes the variance a small difference of large sums.
                let x: Vec<f32> = c.rand(n * ch * spatial, 0.5).iter().map(|v| v + 3.0).collect();
                let w: Vec<f32> = c.rand(ch, 0.1).iter().map(|v| 1.0 + v).collect();
                let b = c.rand(ch, 0.1);
                for silu in [false, true] {
                    let got = down(&ops::group_norm_device(&up(&x)?, &up(&w)?, &up(&b)?, n, ch, spatial, groups, 1e-6, silu)?)?;
                    let want = host::group_norm(&x, &w, &b, ch, spatial, groups, 1e-6, silu);
                    c.cmp(&format!("group_norm_{n}x{ch}x{spatial}{}", if silu { "_silu" } else { "" }), &got, &want, op)?;
                }
            }
        }
        {
            // 1-D convolutions for the audio decoders: cuDNN over a unit height,
            // forward and backward-data, against the host loops (which the unit
            // tests hold to the adjoint identity and to hand-worked cases).
            // Shapes are a DAC/BigVGAN decoder's: dilated residual convs, a
            // strided transposed upsampler, a depthwise anti-alias filter.
            for (name, ch, oc, k, pad, stride, dil, groups, l) in [
                ("resblock_d3", 64usize, 64usize, 7usize, 9usize, 1usize, 3usize, 1usize, 400usize),
                ("downsample_depthwise", 32, 32, 12, 5, 2, 1, 32, 801),
                ("grouped", 48, 96, 3, 1, 1, 1, 4, 257),
            ] {
                let x = c.rand(2 * ch * l, 1.0);
                let w = c.rand(oc * (ch / groups) * k, 0.2);
                let b = c.rand(oc, 0.1);
                let got = t(x.clone(), &[2, ch, l])?.conv1d(&t(w.clone(), &[oc, ch / groups, k])?, Some(&t(b.clone(), &[oc])?), pad, stride, dil, groups)?;
                let (mut want, lo) = ops::host::conv1d(&x, (2, ch, l), &w, (oc, k), pad, stride, dil, groups);
                for (i, v) in want.iter_mut().enumerate() {
                    *v += b[(i / lo) % oc];
                }
                c.cmp(&format!("conv1d_{name}"), &host_of(&got)?, &want, op)?;
            }
            for (name, ch, og, k, pad, stride, groups, out_pad, l) in [
                ("upsample_x5", 128usize, 64usize, 10usize, 3usize, 5usize, 1usize, 1usize, 200usize),
                ("upsample_x2", 32, 16, 4, 1, 2, 1, 0, 1000),
                ("upsample_depthwise", 24, 1, 12, 5, 2, 24, 0, 513),
            ] {
                let x = c.rand(2 * ch * l, 1.0);
                let w = c.rand(ch * og * k, 0.2);
                let got = t(x.clone(), &[2, ch, l])?.conv_transpose1d(&t(w.clone(), &[ch, og, k])?, None, pad, stride, 1, groups, out_pad)?;
                let (want, _) = ops::host::conv_transpose1d(&x, (2, ch, l), &w, (og, k), pad, stride, 1, groups, out_pad);
                c.cmp(&format!("conv_transpose1d_{name}"), &host_of(&got)?, &want, op)?;
            }
        }
        {
            // Frame output: the device packer must produce byte-for-byte what
            // the host writer produced, including truncation and the clamp of
            // out-of-range decoder values. A one-code difference here would be
            // invisible in every metric and visible in every video.
            let (frames, h, w) = (3usize, 17usize, 23usize);
            let x = c.rand(frames * 3 * h * w, 0.8);
            let planar = fastvideo_cudarc::CudaTensor::from_vec(x.clone(), vec![frames, 3, h, w])?;
            let got = ops::pack_rgb_u8_device(&up(&x)?, frames, h, w, 127.5, 127.5)?;
            let plane = h * w;
            let mut want: Vec<u8> = Vec::with_capacity(frames * plane * 3);
            for i in 0..frames * plane {
                let (f, p) = (i / plane, i % plane);
                for ch in 0..3 {
                    want.push(((x[(f * 3 + ch) * plane + p] + 1.0) * 127.5).clamp(0.0, 255.0) as u8);
                }
            }
            let mism = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            c.report.check(
                "pack_rgb_u8_matches_host",
                got.len() == want.len() && mism == 0,
                serde_json::json!({"mismatched_bytes": mism, "n": got.len()}),
                serde_json::json!({"mismatched_bytes": 0}),
            )?;
            // And the pipeline entry point takes the device path for a device tensor.
            let via_tensor = fastvideo_cudarc::wan::pipeline::frames_to_rgb8(&planar)?;
            c.report.check(
                "frames_to_rgb8_uses_device",
                via_tensor == want,
                serde_json::json!({"equal": via_tensor == want}),
                serde_json::json!({"equal": true}),
            )?;
        }
        {
            // FP8 E4M3: the device quantizer must agree with the host reference
            // *exactly*, not approximately. The reference is checked over all
            // 256 codes by a unit test, so an exact match here transfers that
            // guarantee to the kernel; any tolerance would hide a rounding-mode
            // divergence, which is precisely the bug worth catching.
            use fastvideo_ops::fp8;
            let x = c.rand(65_536, 4.0);
            let amax = x.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            let (scale, inv) = fp8::scale_for_amax(amax);
            let (q, scale_dev) = ops::quantize_e4m3_device(&up(&x)?)?;
            let got_bits = dev()?.stream.memcpy_dtov(&q)?;
            let want_bits: Vec<u8> = x.iter().map(|&v| fp8::f32_to_e4m3(v * inv)).collect();
            let mism = got_bits.iter().zip(&want_bits).filter(|(a, b)| a != b).count();
            c.report.check(
                "fp8_quantize_matches_reference",
                mism == 0,
                serde_json::json!({"mismatched_codes": mism, "n": got_bits.len()}),
                serde_json::json!({"mismatched_codes": 0}),
            )?;
            // The device-computed scale must match the host amax reduction too:
            // a wrong scale is invisible in the codes but wrecks the GEMM.
            let got_scale = dev()?.stream.memcpy_dtov(&scale_dev)?[0];
            c.cmp("fp8_scale_from_amax", &[got_scale], &[scale], 1e-6)?;
            // Round-trip lands within E4M3's 3-bit mantissa.
            let deq = down(&ops::dequantize_e4m3_device(&q, &scale_dev)?)?;
            let want: Vec<f32> = want_bits.iter().map(|&b| fp8::e4m3_to_f32(b) * scale).collect();
            c.cmp("fp8_dequantize", &deq, &want, 1e-6)?;
        }
        {
            // The cuBLASLt FP8 GEMM itself, at a real DiT linear shape. This is
            // the part that can only be answered on hardware: whether cuBLASLt
            // has an FP8 algorithm for our shapes at all, and what per-tensor
            // E4M3 actually costs against an f64-accumulated reference.
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use fastvideo_ops::fp8;
            let d = dev()?;
            let (tokens, kk, out_dim) = (256usize, 1536usize, 1536usize);
            match fastvideo_cudarc::wan::fp8::fp8_gemm_supported(&d, out_dim, tokens, kk) {
                Err(why) => c.report.note("fp8_gemm_skipped", serde_json::json!({"reason": why})),
                Ok(()) => {
                    let w = c.rand(out_dim * kk, 0.05); // weight-like magnitudes
                    let x = c.rand(tokens * kk, 1.0);
                    let ltc = fastvideo_cudarc::wan::fp8::lt_context(&d)?;
                    let wq = fastvideo_cudarc::wan::fp8::Fp8Weight::quantize(&d, &w, out_dim, kk)?;
                    let (xq, xs_scale) = fastvideo_cudarc::wan::ops::quantize_e4m3_device(&up(&x)?)?;
                    let mut got = up(&vec![0.0; tokens * out_dim])?;
                    {
                        let (wp, _a) = wq.data.device_ptr(&d.stream);
                        let (wsp, _b) = wq.scale.device_ptr(&d.stream);
                        let (xp, _e) = xq.device_ptr(&d.stream);
                        let (xsp, _f) = xs_scale.device_ptr(&d.stream);
                        let (cp, _g) = got.device_ptr_mut(&d.stream);
                        unsafe {
                            fastvideo_cudarc::wan::fp8::gemm_e4m3(
                                &d, &ltc, out_dim, tokens, kk, wp, wsp, xp, xsp, cp,
                            )?
                        };
                    }
                    // Reference: quantize both operands with the *host* routine
                    // and accumulate in f64, so the only error measured is E4M3
                    // resolution, not the reference's own rounding.
                    let wamax = w.iter().fold(0.0f32, |a, b| a.max(b.abs()));
                    let xamax = x.iter().fold(0.0f32, |a, b| a.max(b.abs()));
                    let (ws, winv) = fp8::scale_for_amax(wamax);
                    let (xsc, xinv) = fp8::scale_for_amax(xamax);
                    let wdq: Vec<f32> = w.iter().map(|&v| fp8::e4m3_to_f32(fp8::f32_to_e4m3(v * winv)) * ws).collect();
                    let xdq: Vec<f32> = x.iter().map(|&v| fp8::e4m3_to_f32(fp8::f32_to_e4m3(v * xinv)) * xsc).collect();
                    let mut want = vec![0.0f32; tokens * out_dim];
                    for i in 0..tokens {
                        for j in 0..out_dim {
                            let acc: f64 = (0..kk)
                                .map(|t| f64::from(xdq[i * kk + t]) * f64::from(wdq[j * kk + t]))
                                .sum();
                            want[i * out_dim + j] = acc as f32;
                        }
                    }
                    // Tolerance covers cuBLASLt's accumulation order only: both
                    // sides see the same quantized operands, so a larger error
                    // means the GEMM is wrong, not that FP8 is imprecise.
                    c.cmp("fp8_gemm_matches_quantized_reference", &down(&got)?, &want, 2e-2)?;
                }
            }
        }
        for (rows, kk, n, gelu) in [(97usize, 1536usize, 8960usize, true), (300, 1536, 1536, false)] {
            // Production Linear: bf16 buffers in fast mode, F32 GEMM in exact mode.
            let (x, w, b) = (c.rand(rows * kk, 1.0), c.rand(n * kk, 0.05), c.rand(n, 0.1));
            let lin = Linear::from_tensors(CudaTensor::from_vec(w.clone(), vec![n, kk])?, Some(CudaTensor::from_vec(b.clone(), vec![n])?))?;
            let xt = t(x.clone(), &[1, rows, kk])?;
            let got = if gelu { lin.forward_gelu(&xt)? } else { lin.forward(&xt)? };
            let mut want = ref_matmul(&x, &transpose2(&w, n, kk), rows, kk, n);
            for (i, v) in want.iter_mut().enumerate() {
                *v += b[i % n];
                if gelu {
                    *v = ref_gelu_tanh(*v);
                }
            }
            c.cmp(&format!("linear_{math:?}_{rows}x{kk}x{n}_gelu{gelu}"), &host_of(&got)?, &want, gemm)?;
        }
        {
            // 1×1 conv as a shared-weight GEMM over channel-first activations.
            let (batch, oc, ic, s) = (3usize, 7usize, 5usize, 41usize);
            let (w, x) = (c.rand(oc * ic, 0.3), c.rand(batch * ic * s, 1.0));
            let mut out = up(&vec![0.0; batch * oc * s])?;
            device::matmul_shared_left(&up(&w)?, &up(&x)?, &mut out, batch, oc, ic, s)?;
            let want: Vec<f32> = (0..batch).flat_map(|bi| ref_matmul(&w, &x[bi * ic * s..(bi + 1) * ic * s], oc, ic, s)).collect();
            c.cmp("gemm_shared_left", &down(&out)?, &want, gemm)?;
        }
        Ok(())
    })?;

    group(&mut c, "vsa", |c| {
        // Grids cover full tiles, a partial tile on every axis, and the real
        // 8s-clip aspect at reduced depth. Device output is checked against the
        // host reference, which is itself checked against dense attention.
        for (grid, heads, dim, sparsity) in [
            ((4usize, 4usize, 4usize), 2usize, 64usize, 0.0f64),
            ((2, 3, 5), 2, 64, 0.0),
            ((5, 6, 9), 3, 128, 0.8),
            ((9, 8, 13), 2, 128, 0.8),
        ] {
            let plan = vsa::TilePlan::new(grid)?;
            let (b, seq, nb) = (1usize, plan.seq, plan.num_tiles());
            let topk = vsa::topk_for(sparsity, nb);
            let bh = b * heads;
            let n = bh * seq * dim;
            let (q, k, v, g) = (c.rand(n, 1.0), c.rand(n, 1.0), c.rand(n, 1.0), c.rand(n, 0.5));
            let scale = 1.0 / (dim as f32).sqrt();
            let want = vsa::vsa_attention_host(&q, &k, &v, Some(&g), &plan, topk, b, heads, dim, scale)?;

            let dev = dev()?;
            let (qd, kd, vd, gd) = (
                dev.stream.memcpy_stod(&q)?,
                dev.stream.memcpy_stod(&k)?,
                dev.stream.memcpy_stod(&v)?,
                dev.stream.memcpy_stod(&g)?,
            );
            let plan_dev = ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, vsa::TILE_ELEMS)?;
            // A group smaller than the tile count exercises the chunked loop.
            for group in [nb, 2.min(nb)] {
                // Pinned by name: on the first tensor-core run `auto` picked
                // the new kernel here too, and a "gather" failure was really a
                // second copy of the mma failure.
                std::env::set_var("FASTVIDEO_VSA_KERNEL", "gather");
                let got = vsa::vsa_attention_device(
                    &qd, &kd, &vd, Some(&gd), &plan_dev, topk, bh, seq, dim, scale, group,
                );
                std::env::remove_var("FASTVIDEO_VSA_KERNEL");
                let got = got?;
                let host = dev.stream.memcpy_dtov(&got)?;
                let tag = format!("vsa_{}x{}x{}_h{heads}_d{dim}_k{topk}_g{group}", grid.0, grid.1, grid.2);
                // bf16 gathers on the fine stage, so this is bf16 round-off.
                c.cmp(&tag, &host, &want, 2e-2)?;
            }
            // The fused fine stage must land on the same answer as gather +
            // batched GEMM: same algorithm, different execution.
            std::env::set_var("FASTVIDEO_VSA_FUSED", "1");
            let fused = vsa::vsa_attention_device(
                &qd, &kd, &vd, Some(&gd), &plan_dev, topk, bh, seq, dim, scale, nb,
            )?;
            std::env::remove_var("FASTVIDEO_VSA_FUSED");
            let host_fused = dev.stream.memcpy_dtov(&fused)?;
            let tag = format!("vsa_fused_{}x{}x{}_h{heads}_d{dim}_k{topk}", grid.0, grid.1, grid.2);
            c.cmp(&tag, &host_fused, &want, 2e-2)?;

            // The tensor-core fine stage: same answer as the host reference,
            // through mma.sync + ldmatrix + cp.async instead of a gather.
            // P is bf16 on this path exactly as on the gather path, so the
            // limit is the same bf16 round-off.
            if dim == 128 && dev.sm_major >= 8 {
                std::env::set_var("FASTVIDEO_VSA_KERNEL", "mma");
                let got = vsa::vsa_attention_device(
                    &qd, &kd, &vd, Some(&gd), &plan_dev, topk, bh, seq, dim, scale, nb,
                )?;
                std::env::remove_var("FASTVIDEO_VSA_KERNEL");
                let tag = format!("vsa_mma_{}x{}x{}_h{heads}_d{dim}_k{topk}", grid.0, grid.1, grid.2);
                c.cmp(&tag, &dev.stream.memcpy_dtov(&got)?, &want, 2e-2)?;
            } else {
                c.report.note(
                    format!("vsa_mma_{}x{}x{}_skipped", grid.0, grid.1, grid.2),
                    serde_json::json!({"dim": dim, "sm_major": dev.sm_major, "needs": "dim 128, sm80+"}),
                );
            }

            // Tile means and top-k are exact, so they get tight limits of their own.
            let qc = ops::vsa_tile_mean_device(&qd, &plan_dev, bh, seq, dim)?;
            let mut want_mean = vec![0.0f32; bh * nb * dim];
            for h in 0..bh {
                for t in 0..nb {
                    let cnt = plan.block_sizes[t] as usize;
                    for sl in t * vsa::TILE_ELEMS..t * vsa::TILE_ELEMS + cnt {
                        let tok = plan.slot_src[sl] as usize;
                        for d in 0..dim {
                            want_mean[(h * nb + t) * dim + d] += q[(h * seq + tok) * dim + d] / cnt as f32;
                        }
                    }
                }
            }
            c.cmp(&format!("vsa_tile_mean_{}x{}x{}", grid.0, grid.1, grid.2), &down(&qc)?, &want_mean, op)?;
        }

        // Speed, at the token counts the scalar fused kernel was rejected at
        // (1,456 / 4,368 / 13,104 — FVID-2026-09-18-fused-block-sparse-rejected
        // recorded 0.107 / 0.870 s for the gather path). Median of three
        // synchronized runs after a warm-up; the gather path is the baseline
        // and the number that matters is the ratio.
        let dev = dev()?;
        if dev.sm_major >= 8 {
            let time = |f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<f64> {
                f()?;
                let mut t = Vec::new();
                for _ in 0..3 {
                    device::synchronize()?;
                    let s = std::time::Instant::now();
                    f()?;
                    device::synchronize()?;
                    t.push(s.elapsed().as_secs_f64());
                }
                t.sort_by(|a, b| a.total_cmp(b));
                Ok(t[1])
            };
            for grid in [(1usize, 28usize, 52usize), (3, 28, 52), (9, 28, 52)] {
                let plan = vsa::TilePlan::new(grid)?;
                let (heads, dim) = (12usize, 128usize);
                let (bh, seq, nb) = (heads, plan.seq, plan.num_tiles());
                let topk = vsa::topk_for(0.8, nb);
                let n = bh * seq * dim;
                let (q, k, v, g) = (c.rand(n, 1.0), c.rand(n, 1.0), c.rand(n, 1.0), c.rand(n, 0.5));
                let (qd, kd, vd, gd) = (
                    dev.stream.memcpy_stod(&q)?,
                    dev.stream.memcpy_stod(&k)?,
                    dev.stream.memcpy_stod(&v)?,
                    dev.stream.memcpy_stod(&g)?,
                );
                let plan_dev = ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, vsa::TILE_ELEMS)?;
                let scale = 1.0 / (dim as f32).sqrt();
                let mut run = |kernel: &str| -> anyhow::Result<f64> {
                    std::env::set_var("FASTVIDEO_VSA_KERNEL", kernel);
                    let r = time(&mut || {
                        vsa::vsa_attention_device(&qd, &kd, &vd, Some(&gd), &plan_dev, topk, bh, seq, dim, scale, 32)?;
                        Ok(())
                    });
                    std::env::remove_var("FASTVIDEO_VSA_KERNEL");
                    r
                };
                let gather_s = run("gather")?;
                let mma_s = run("mma")?;
                c.report.note(
                    format!("vsa_fine_time_{}tok", seq),
                    serde_json::json!({
                        "tokens": seq, "tiles": nb, "topk": topk,
                        "gather_s": gather_s, "mma_s": mma_s,
                        "mma_speedup": gather_s / mma_s,
                    }),
                );
            }
        }
        Ok(())
    })?;

    group(&mut c, "attention", |c| {
        // Includes the Wan VAE mid-block shape (one 384-wide head) and a small
        // score budget that forces the chunked in-place path.
        for (b, h, sq, sk, d, budget) in [
            (1usize, 2usize, 70usize, 70usize, 64usize, usize::MAX),
            (2, 12, 257, 257, 128, usize::MAX),
            (1, 12, 300, 512, 128, usize::MAX),
            (2, 12, 300, 512, 128, 12 * 512 * 64),
            (2, 1, 256, 256, 384, usize::MAX),
        ] {
            let q = c.rand(b * h * sq * d, 1.0);
            let kk = c.rand(b * h * sk * d, 1.0);
            let v = c.rand(b * h * sk * d, 1.0);
            let scale = 1.0 / (d as f32).sqrt();
            let want = ref_sdpa(&q, &kk, &v, b * h, sq, sk, d, scale);
            let (qt, kt, vt) = (t(q, &[b, h, sq, d])?, t(kk, &[b, h, sk, d])?, t(v, &[b, h, sk, d])?);
            let tag = format!("{b}x{h}x{sq}x{sk}x{d}");
            let dense = attn::device_dense_sdpa_with_budget(&qt, &kt, &vt, Some(scale), budget)?
                .ok_or_else(|| anyhow::anyhow!("dense sdpa declined on a live device"))?;
            let chunked = if budget == usize::MAX { "" } else { "_chunked" };
            c.cmp(&format!("dense_sdpa{chunked}_{tag}"), &host_of(&dense)?, &want, gemm.max(op))?;
            if d % 32 == 0 && d <= attn::FLASH_MAX_HEAD_DIM {
                let got = attn::device_flash_sdpa(&qt, &kt, &vt, Some(scale))?.ok_or_else(|| anyhow::anyhow!("flash declined"))?;
                c.cmp(&format!("flash_sdpa_{tag}"), &host_of(&got)?, &want, op.max(1e-4))?;
            }
        }
        Ok(())
    })?;

    group(&mut c, "conv", |c| {
        // cuDNN conv2d (patch embed, VAE resample): pad/stride variants + bias.
        let (n, ci, h, w, co) = (2usize, 3usize, 17usize, 20usize, 6usize);
        let x = c.rand(n * ci * h * w, 1.0);
        let wt = c.rand(co * ci * 9, 0.2);
        let bias = c.rand(co, 0.1);
        let (xt, wtt, bt) = (t(x.clone(), &[n, ci, h, w])?, t(wt.clone(), &[co, ci, 3, 3])?, t(bias.clone(), &[co])?);
        for (pad, stride) in [(0usize, 1usize), (1, 1), (1, 2)] {
            let got = xt.conv2d(&wtt, Some(&bt), pad, stride)?;
            let (want, _) = ref_conv3d(&x, &wt, Some(&bias), (n, ci, 1, h, w), (co, 1, 3, 3), (0, pad, pad), (1, stride, stride));
            c.cmp(&format!("cudnn_conv2d_p{pad}_s{stride}"), &host_of(&got)?, &want, gemm.max(op))?;
        }
        // 1×1 conv2d → GEMM path.
        let w1 = c.rand(co * ci, 0.3);
        let got = xt.conv2d(&t(w1.clone(), &[co, ci, 1, 1])?, Some(&bt), 0, 1)?;
        let (want, _) = ref_conv3d(&x, &w1, Some(&bias), (n, ci, 1, h, w), (co, 1, 1, 1), (0, 0, 0), (1, 1, 1));
        c.cmp("conv2d_1x1_gemm", &host_of(&got)?, &want, gemm.max(op))?;

        // 3-D: cuDNN N-D and temporal unfold, stride 1 and spatial stride 2.
        for (dims, stride) in [((1usize, 4usize, 5usize, 9usize, 11usize), [1usize, 1, 1]), ((2, 3, 4, 10, 12), [1, 2, 2])] {
            let (n, ic, it, ih, iw) = dims;
            let oc = 5usize;
            let x = c.rand(n * ic * it * ih * iw, 1.0);
            let w = c.rand(oc * ic * 27, 0.2);
            let bias = c.rand(oc, 0.1);
            let (want, _) = ref_conv3d(&x, &w, Some(&bias), dims, (oc, 3, 3, 3), (0, 1, 1), (stride[0], stride[1], stride[2]));
            let xt = t(x, &[n, ic, it, ih, iw])?;
            let wt = t(w, &[oc, ic, 3, 3, 3])?;
            let tag = format!("{n}x{ic}x{it}x{ih}x{iw}_s{}", stride[1]);
            let got = xt.conv3d(&wt, Some(&t(bias.clone(), &[oc])?), [0, 1, 1], stride)?;
            c.cmp(&format!("cudnn_conv3d_{tag}"), &host_of(&got)?, &want, gemm.max(op))?;
            let (x_dev, w_dev) = (xt.device_slice().unwrap(), wt.device_slice().unwrap());
            let (y, y_shape) = conv::conv3d_unfold(x_dev, &xt.shape, w_dev, &wt.shape, [1, 1], stride)?;
            let got = CudaTensor::from_device_slice(y, y_shape)?.add_bias(&t(bias, &[oc])?, 1)?;
            c.cmp(&format!("unfold_conv3d_{tag}"), &host_of(&got)?, &want, gemm.max(op))?;
        }
        // 1×1×1 conv3d → GEMM path.
        let (n, ic, tt, hh, ww, oc) = (1usize, 6usize, 3usize, 5usize, 7usize, 4usize);
        let x = c.rand(n * ic * tt * hh * ww, 1.0);
        let w = c.rand(oc * ic, 0.3);
        let got = t(x.clone(), &[n, ic, tt, hh, ww])?.conv3d(&t(w.clone(), &[oc, ic, 1, 1, 1])?, None, [0, 0, 0], [1, 1, 1])?;
        let (want, _) = ref_conv3d(&x, &w, None, (n, ic, tt, hh, ww), (oc, 1, 1, 1), (0, 0, 0), (1, 1, 1));
        c.cmp("conv3d_1x1x1_gemm", &host_of(&got)?, &want, gemm.max(op))?;
        Ok(())
    })?;

    group(&mut c, "conv3d_bench", |c| {
        // VAE decoder shapes at 480p-class resolution: time both strategies.
        for (ch, t_in, hh, ww) in [(384usize, 3usize, 30usize, 52usize), (192, 6, 120, 208), (96, 6, 240, 416)] {
            let x = t(c.rand(ch * t_in * hh * ww, 1.0), &[1, ch, t_in, hh, ww])?;
            let w = t(c.rand(ch * ch * 27, 0.02), &[ch, ch, 3, 3, 3])?;
            let (xd, wd) = (x.device_slice().unwrap(), w.device_slice().unwrap());
            let mut values = serde_json::Map::new();
            for backend in ["cudnn", "unfold"] {
                // One warm-up (plan build, allocator) then the timed call.
                let mut secs = 0.0;
                for iter in 0..2 {
                    device::synchronize()?;
                    let timer = std::time::Instant::now();
                    let y = if backend == "cudnn" {
                        conv::cudnn_conv(xd, &x.shape, wd, &w.shape, &[0, 1, 1], &[1, 1, 1])?.0
                    } else {
                        conv::conv3d_unfold(xd, &x.shape, wd, &w.shape, [1, 1], [1, 1, 1])?.0
                    };
                    device::synchronize()?;
                    if iter == 1 {
                        secs = timer.elapsed().as_secs_f64();
                    }
                    drop(y);
                }
                values.insert(format!("{backend}_seconds"), json!(secs));
            }
            c.report.note(format!("conv3d_bench_{ch}x{t_in}x{hh}x{ww}"), serde_json::Value::Object(values));
        }
        Ok(())
    })?;

    group(&mut c, "no_host_fallback", |c| {
        // With a device live, an op without a device path must error, not
        // silently compute on the CPU.
        let x = t(c.rand(16, 1.0), &[16])?;
        let refused = x.sqrt().is_err();
        c.report.check("sqrt_refuses_host_compute", refused, json!({"refused": refused}), json!({}))?;
        let a = t(c.rand(6, 1.0), &[2, 1, 3])?;
        let b = t(c.rand(4, 1.0), &[1, 4, 1])?;
        let refused = a.mul(&b).is_err();
        c.report.check("generic_broadcast_refuses_host_compute", refused, json!({"refused": refused}), json!({}))?;
        Ok(())
    })?;
    Ok(())
}

/// NVRTC compile of the kernel module for each architecture. Needs only
/// `libnvrtc` (no GPU, no driver), so it runs in CI or any Linux container.
pub fn nvrtc(report: &mut Report, archs: &[(i32, i32)]) -> StageResult<()> {
    for &(maj, min) in archs {
        let timer = std::time::Instant::now();
        let result = k::compile_ptx(maj, min);
        let secs = timer.elapsed().as_secs_f64();
        let (ok, err) = match &result {
            Ok(_) => (true, serde_json::Value::Null),
            Err(e) => (false, json!(e.to_string())),
        };
        report.check(format!("compile_sm{maj}{min}"), ok, json!({"seconds": secs, "error": err, "kernels": k::KERNEL_NAMES.len()}), json!({}))?;
    }
    Ok(())
}
