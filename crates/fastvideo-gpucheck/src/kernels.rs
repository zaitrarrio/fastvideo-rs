//! Kernel-level parity on a live GPU: every NVRTC kernel and cuBLAS path the
//! Wan graph uses, called through its public device wrapper and compared with
//! a plain-Rust reference. A wrapper returning `None` is a failure, not a skip:
//! it means production would silently fall back to host compute.
//!
//! Shapes deliberately include non-power-of-two widths, partial reduction
//! blocks, key lengths that are not multiples of the flash tile, and B=2.

use cudarc::driver::CudaSlice;
use fastvideo_cudarc::wan::{attn, bf16_gemm, device, kernels as k, nn, ops};
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

fn some<T>(name: &str, v: Option<T>) -> anyhow::Result<T> {
    v.ok_or_else(|| anyhow::anyhow!("{name}: device path returned None (would silently run on host)"))
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

// ---- references -----------------------------------------------------------

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

fn ref_layer_norm(x: &[f32], width: usize, w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (row, o) in x.chunks(width).zip(out.chunks_mut(width)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / width as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for i in 0..width {
            let mut y = ((f64::from(row[i]) - mean) * inv) as f32;
            if let Some(w) = w {
                y *= w[i];
            }
            if let Some(b) = b {
                y += b[i];
            }
            o[i] = y;
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
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += f64::from(a[i * k + t]) * f64::from(b[t * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

/// `[bh, sq, d] × [bh, sk, d]` scaled dot-product attention.
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

fn ref_conv3d(
    x: &[f32], w: &[f32], bias: &[f32],
    (n, ic, it, ih, iw): (usize, usize, usize, usize, usize),
    (oc, kt, kh, kw): (usize, usize, usize, usize),
    (st, sh, sw): (usize, usize, usize),
) -> (Vec<f32>, [usize; 3]) {
    let (ot, oh, ow) = ((it - kt) / st + 1, (ih - kh) / sh + 1, (iw - kw) / sw + 1);
    let mut out = vec![0.0; n * oc * ot * oh * ow];
    for b in 0..n {
        for o in 0..oc {
            for t in 0..ot {
                for y in 0..oh {
                    for xx in 0..ow {
                        let mut acc = f64::from(bias[o]);
                        for c in 0..ic {
                            for dt in 0..kt {
                                for dy in 0..kh {
                                    for dx in 0..kw {
                                        let xi = (((b * ic + c) * it + t * st + dt) * ih + y * sh + dy) * iw + xx * sw + dx;
                                        let wi = (((o * ic + c) * kt + dt) * kh + dy) * kw + dx;
                                        acc += f64::from(x[xi]) * f64::from(w[wi]);
                                    }
                                }
                            }
                        }
                        out[(((b * oc + o) * ot + t) * oh + y) * ow + xx] = acc as f32;
                    }
                }
            }
        }
    }
    (out, [ot, oh, ow])
}

// ---- suite ----------------------------------------------------------------

/// Run one kernel group. An error inside a group (device wrapper returned
/// `None`, CUDA error) becomes a failed check, so with `--keep-going` the
/// remaining groups still run and one rental lists every broken kernel.
fn group(c: &mut Ctx<'_>, name: &str, f: impl FnOnce(&mut Ctx<'_>) -> StageResult<()>) -> StageResult<()> {
    match f(c) {
        Err(StageError::Error(e)) => c.report.check(
            format!("{name}/completed"),
            false,
            json!({"error": format!("{e:#}")}),
            json!({}),
        ),
        other => other,
    }
}

pub fn run(report: &mut Report, lim: Limits, seed: u64) -> StageResult<()> {
    let info = crate::gpu::init("cuda")?;
    report.set("device", &info);
    report.set("limits", lim);
    let mut c = Ctx { report, seed };
    let op = lim.op;
    // Shared GEMM tolerance: TF32 (when on) perturbs cuBLAS F32 results.
    let gemm = op.max(if device::global_device().is_some_and(|d| d.tf32) { 2e-3 } else { 0.0 });
    group(&mut c, "elementwise", |c: &mut Ctx<'_>| -> StageResult<()> {
        // Elementwise.
        for n in [1usize, 1000, 65_537] {
            let (a, b) = (c.rand(n, 1.0), c.rand(n, 1.0));
            let (da, db) = (up(&a)?, up(&b)?);
            for (kind, name, f) in [
                (ops::ElemBinary::Add, "add", (|x: f32, y: f32| x + y) as fn(f32, f32) -> f32),
                (ops::ElemBinary::Sub, "sub", |x, y| x - y),
                (ops::ElemBinary::Mul, "mul", |x, y| x * y),
            ] {
                let got = down(&some(name, ops::elem_binary_device(&da, &db, kind))?)?;
                let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect();
                c.cmp(&format!("elem_{name}_n{n}"), &got, &want, op)?;
            }
            let got = down(&some("mul_scalar", ops::mul_scalar_device(&da, -1.75))?)?;
            c.cmp(&format!("mul_scalar_n{n}"), &got, &a.iter().map(|v| v * -1.75).collect::<Vec<_>>(), op)?;
            let got = down(&some("add_scalar", ops::add_scalar_device(&da, 0.5))?)?;
            c.cmp(&format!("add_scalar_n{n}"), &got, &a.iter().map(|v| v + 0.5).collect::<Vec<_>>(), op)?;
            let got = down(&some("silu", ops::unary_device(&da, ops::ElemUnary::Silu))?)?;
            let want: Vec<f32> = a.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
            c.cmp(&format!("silu_n{n}"), &got, &want, op)?;
            let got = down(&some("gelu_tanh", ops::unary_device(&da, ops::ElemUnary::GeluTanh))?)?;
            c.cmp(&format!("gelu_tanh_n{n}"), &got, &a.iter().map(|&v| ref_gelu_tanh(v)).collect::<Vec<_>>(), op)?;
            let got = down(&some("clamp", ops::clamp_device(&da, -0.3, 0.7))?)?;
            c.cmp(&format!("clamp_n{n}"), &got, &a.iter().map(|v| v.clamp(-0.3, 0.7)).collect::<Vec<_>>(), op)?;
        }

        Ok(())
    })?;
    group(&mut c, "row_reductions", |c: &mut Ctx<'_>| -> StageResult<()> {
        // Row reductions: widths straddle the 256-thread block and non-powers of 2.
        for (rows, width) in [(1usize, 7usize), (5, 64), (33, 255), (3, 256), (4, 300), (2, 1536), (2, 4096)] {
            let x = c.rand(rows * width, 2.0);
            let dx = up(&x)?;
            let got = down(&some("softmax_last", ops::softmax_last_device(&dx, width))?)?;
            c.cmp(&format!("softmax_{rows}x{width}"), &got, &ref_softmax(&x, width), op)?;
            let w: Vec<f32> = c.rand(width, 0.1).iter().map(|v| 1.0 + v).collect();
            let b = c.rand(width, 0.1);
            let (dw, db) = (up(&w)?, up(&b)?);
            let got = down(&some("rms_norm_last", ops::rms_norm_last_device(&dx, &dw, 1e-6))?)?;
            c.cmp(&format!("rms_norm_{rows}x{width}"), &got, &ref_rms(&x, &w, 1e-6), op)?;
            let got = down(&some("layer_norm_last", ops::layer_norm_last_device(&dx, Some(&dw), Some(&db), width, 1e-6))?)?;
            c.cmp(&format!("layer_norm_affine_{rows}x{width}"), &got, &ref_layer_norm(&x, width, Some(&w), Some(&b), 1e-6), op)?;
            let got = down(&some("layer_norm_last", ops::layer_norm_last_device(&dx, None, None, width, 1e-6))?)?;
            c.cmp(&format!("layer_norm_plain_{rows}x{width}"), &got, &ref_layer_norm(&x, width, None, None, 1e-6), op)?;
        }

        Ok(())
    })?;
    group(&mut c, "adaln", |c: &mut Ctx<'_>| -> StageResult<()> {
        // AdaLN broadcast ops and the fused LN+AdaLN kernel.
        for (b, s, d) in [(1usize, 17usize, 64usize), (2, 300, 1536)] {
            let x = c.rand(b * s * d, 1.0);
            let scale = c.rand(b * d, 0.2);
            let shift = c.rand(b * d, 0.2);
            let (dx, dsc, dsh) = (up(&x)?, up(&scale)?, up(&shift)?);
            let mut want_mod = vec![0.0; x.len()];
            let mut want_gate = vec![0.0; x.len()];
            for bi in 0..b {
                for si in 0..s {
                    for di in 0..d {
                        let i = (bi * s + si) * d + di;
                        want_mod[i] = x[i] * (1.0 + scale[bi * d + di]) + shift[bi * d + di];
                        want_gate[i] = x[i] * scale[bi * d + di];
                    }
                }
            }
            let got = down(&some("modulate", ops::modulate_scale_shift_device(&dx, &dsc, &dsh, b, s, d))?)?;
            c.cmp(&format!("modulate_{b}x{s}x{d}"), &got, &want_mod, op)?;
            let got = down(&some("gate_mul", ops::broadcast_mul_last_device(&dx, &dsc, b, s, d))?)?;
            c.cmp(&format!("gate_mul_{b}x{s}x{d}"), &got, &want_gate, op)?;

            let normed = ref_layer_norm(&x, d, None, None, 1e-6);
            let mut want = vec![0.0; x.len()];
            for bi in 0..b {
                for si in 0..s {
                    for di in 0..d {
                        let i = (bi * s + si) * d + di;
                        want[i] = normed[i] * (1.0 + scale[bi * d + di]) + shift[bi * d + di];
                    }
                }
            }
            let xt = CudaTensor::from_vec(x.clone(), vec![b, s, d])?;
            let st = CudaTensor::from_vec(scale.clone(), vec![b, d])?;
            let sh = CudaTensor::from_vec(shift.clone(), vec![b, d])?;
            let got = some("layer_norm_adaln", nn::layer_norm_adaln(&xt, &st, &sh, 1e-6)?)?;
            c.cmp(&format!("layer_norm_adaln_{b}x{s}x{d}"), &got.host_cow()?, &want, op)?;

            let mut out = up(&x)?;
            let bias = c.rand(d, 0.5);
            ops::add_bias_last_inplace(&mut out, &up(&bias)?)?;
            let want: Vec<f32> = x.iter().enumerate().map(|(i, v)| v + bias[i % d]).collect();
            c.cmp(&format!("add_bias_{b}x{s}x{d}"), &down(&out)?, &want, op)?;
        }

        Ok(())
    })?;
    group(&mut c, "data_movement", |c: &mut Ctx<'_>| -> StageResult<()> {
        // Data movement kernels.
        let shape = [2usize, 3, 5, 7];
        let x = c.rand(shape.iter().product(), 1.0);
        let dx = up(&x)?;
        for dims in [[0usize, 2, 1, 3], [0, 1, 3, 2], [3, 2, 1, 0]] {
            let got = down(&device::permute_4d_device(&dx, shape, dims)?)?;
            let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
            let mut want = vec![0.0; x.len()];
            let in_str = [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1];
            let mut idx = 0;
            for a in 0..out_shape[0] {
                for b in 0..out_shape[1] {
                    for cc in 0..out_shape[2] {
                        for d in 0..out_shape[3] {
                            let o = [a, b, cc, d];
                            let mut src = 0;
                            for (oi, &dd) in dims.iter().enumerate() {
                                src += o[oi] * in_str[dd];
                            }
                            want[idx] = x[src];
                            idx += 1;
                        }
                    }
                }
            }
            c.cmp(&format!("permute_4d_{dims:?}"), &got, &want, 0.0)?;
        }
        {
            let (outer, in_stride, out_stride, len, in_off, out_off) = (6usize, 40usize, 25usize, 17usize, 9usize, 4usize);
            let x = c.rand(outer * in_stride, 1.0);
            let mut out = up(&vec![0.0; outer * out_stride])?;
            some("block_copy", ops::block_copy_device(&up(&x)?, &mut out, outer, len, in_stride, out_stride, in_off, out_off))?;
            let mut want = vec![0.0; outer * out_stride];
            for o in 0..outer {
                for i in 0..len {
                    want[o * out_stride + out_off + i] = x[o * in_stride + in_off + i];
                }
            }
            c.cmp("block_copy", &down(&out)?, &want, 0.0)?;
        }
        {
            let (rows, dim) = (70usize, 128usize);
            let x = c.rand(rows * dim, 1.0);
            let angles = c.rand(rows * dim, 3.0);
            let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
            let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
            let got = down(&some("rope_interleaved", ops::rope_interleaved_device(&up(&x)?, &up(&cos)?, &up(&sin)?, dim))?)?;
            // Matches transformer::apply_rotary: even slots use cos/sin at the even index.
            let mut want = vec![0.0; x.len()];
            for r in 0..rows {
                for p in 0..dim / 2 {
                    let (i0, i1) = (r * dim + 2 * p, r * dim + 2 * p + 1);
                    let (x1, x2, cs, sn) = (x[i0], x[i1], cos[i0], sin[i1]);
                    want[i0] = x1 * cs - x2 * sn;
                    want[i1] = x1 * sn + x2 * cs;
                }
            }
            c.cmp("rope_interleaved", &got, &want, op)?;
        }
        {
            let (n, ch, spatial) = (2usize, 12usize, 5 * 9 * 11);
            let x = c.rand(n * ch * spatial, 1.0);
            let g: Vec<f32> = c.rand(ch, 0.1).iter().map(|v| 1.0 + v).collect();
            let got = down(&some("rms_norm_channels", ops::rms_norm_channels_device(&up(&x)?, &up(&g)?, n, ch, spatial, 1e-12))?)?;
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
        }

        Ok(())
    })?;
    group(&mut c, "gemm_f32_bf16", |c: &mut Ctx<'_>| -> StageResult<()> {
        // cuBLAS F32 (TF32 when enabled) GEMMs.
        for (m, kk, n) in [(1usize, 1usize, 1usize), (37, 64, 19), (512, 1536, 1536), (300, 1536, 8960)] {
            let (a, b) = (c.rand(m * kk, 1.0), c.rand(kk * n, 0.05));
            let mut out = up(&vec![0.0; m * n])?;
            device::matmul_2d_f32_device(&up(&a)?, &up(&b)?, &mut out, m, kk, n)?;
            c.cmp(&format!("gemm_2d_{m}x{kk}x{n}"), &down(&out)?, &ref_matmul(&a, &b, m, kk, n), gemm)?;
            // Linear: x[m,k] @ w[n,k]^T.
            let w = c.rand(n * kk, 0.05);
            let mut wt = vec![0.0; kk * n];
            for r in 0..n {
                for col in 0..kk {
                    wt[col * n + r] = w[r * kk + col];
                }
            }
            let mut out = up(&vec![0.0; m * n])?;
            device::matmul_linear_wt_device(&up(&a)?, &up(&w)?, &mut out, m, kk, n)?;
            let want = ref_matmul(&a, &wt, m, kk, n);
            c.cmp(&format!("gemm_linear_{m}x{kk}x{n}"), &down(&out)?, &want, gemm)?;

            // BF16 GEMM (the default DiT linear path) must match F32 to BF16 precision.
            let w_bf16 = bf16_gemm::upload_bf16(&w)?;
            let got = bf16_gemm::matmul_linear_wt_bf16_to_f32(&up(&a)?, &w_bf16, m, kk, n)?;
            c.cmp(&format!("bf16_gemm_linear_{m}x{kk}x{n}"), &down(&got)?, &want, 5e-3)?;
        }
        {
            let (batch, m, kk, n) = (24usize, 50usize, 32usize, 70usize);
            let (a, b) = (c.rand(batch * m * kk, 1.0), c.rand(batch * kk * n, 0.2));
            let mut out = up(&vec![0.0; batch * m * n])?;
            device::matmul_2d_strided_batched(&up(&a)?, &up(&b)?, &mut out, batch, m, kk, n)?;
            let mut want = Vec::with_capacity(batch * m * n);
            for bi in 0..batch {
                want.extend(ref_matmul(&a[bi * m * kk..(bi + 1) * m * kk], &b[bi * kk * n..(bi + 1) * kk * n], m, kk, n));
            }
            c.cmp("gemm_strided_batched", &down(&out)?, &want, gemm)?;
        }

        Ok(())
    })?;
    group(&mut c, "bf16_elementwise", |c: &mut Ctx<'_>| -> StageResult<()> {
        // BF16 elementwise kernels used by the chained FFN.
        {
            let x = c.rand(10_000, 2.0);
            let bits: Vec<u16> = x.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
            let mut dbits = dev()?.stream.memcpy_stod(&bits)?;
            nn::gelu_tanh_bf16_inplace(&mut dbits)?;
            let got: Vec<f32> = dev()?.stream.memcpy_dtov(&dbits)?.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
            let want: Vec<f32> = x.iter().map(|&v| ref_gelu_tanh(v)).collect();
            c.cmp("gelu_tanh_bf16", &got, &want, 5e-3)?;

            let mut out16 = dev()?.stream.alloc_zeros::<u16>(x.len())?;
            unsafe {
                k::launch_f32_to_bf16(&dev()?.stream, &dev()?.kernels.f32_to_bf16, &up(&x)?, &mut out16, x.len() as i32)?;
            }
            let got: Vec<f32> = dev()?.stream.memcpy_dtov(&out16)?.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
            c.cmp("f32_to_bf16", &got, &x, 5e-3)?;
        }

        Ok(())
    })?;
    group(&mut c, "attention", |c: &mut Ctx<'_>| -> StageResult<()> {
        // Attention: flash kernel and GPU dense SDPA vs reference, including
        // cross-attention lengths and Sk not a multiple of the 32-wide tile.
        // Includes the Wan VAE mid-block shape: one 384-wide head over 16x16 tokens.
        for (b, h, sq, sk, d) in [(1usize, 2usize, 70usize, 70usize, 64usize), (2, 12, 257, 257, 128), (1, 12, 300, 512, 128), (1, 2, 1100, 1100, 32), (2, 1, 256, 256, 384)] {
            let q = c.rand(b * h * sq * d, 1.0);
            let kk = c.rand(b * h * sk * d, 1.0);
            let v = c.rand(b * h * sk * d, 1.0);
            let scale = 1.0 / (d as f32).sqrt();
            let want = ref_sdpa(&q, &kk, &v, b * h, sq, sk, d, scale);
            let qt = CudaTensor::from_vec(q, vec![b, h, sq, d])?;
            let kt = CudaTensor::from_vec(kk, vec![b, h, sk, d])?;
            let vt = CudaTensor::from_vec(v, vec![b, h, sk, d])?;
            let tag = format!("{b}x{h}x{sq}x{sk}x{d}");
            let got = some("dense_sdpa", attn::device_dense_sdpa(&qt, &kt, &vt, Some(scale))?)?;
            c.cmp(&format!("dense_sdpa_{tag}"), &got.host_cow()?, &want, gemm.max(op))?;
            if d % 32 == 0 && d <= attn::FLASH_MAX_HEAD_DIM {
                let got = some("flash_sdpa", attn::device_flash_sdpa(&qt, &kt, &vt, Some(scale))?)?;
                c.cmp(&format!("flash_sdpa_{tag}"), &got.host_cow()?, &want, op.max(1e-4))?;
            } else {
                // Beyond the flash kernel's limits the wrapper must decline
                // (None → dense fallback), never launch and fail.
                let declined = attn::device_flash_sdpa(&qt, &kt, &vt, Some(scale))?.is_none();
                c.report.check(format!("flash_sdpa_declines_{tag}"), declined, json!({"declined": declined}), json!({}))?;
            }
        }

        Ok(())
    })?;
    group(&mut c, "causal_conv3d", |c: &mut Ctx<'_>| -> StageResult<()> {
        // Causal conv3d (VAE hot path), stride 1 and spatial stride 2.
        for (dims, stride) in [((1usize, 4usize, 3usize, 9usize, 11usize), (1usize, 1usize, 1usize)), ((2, 3, 4, 10, 12), (1, 2, 2))] {
            let (n, ic, it, ih, iw) = dims;
            let (oc, kt, kh, kw) = (5usize, 3usize, 3usize, 3usize);
            let x = c.rand(n * ic * it * ih * iw, 1.0);
            let w = c.rand(oc * ic * kt * kh * kw, 0.2);
            let bias = c.rand(oc, 0.1);
            let (want, [ot, oh, ow]) = ref_conv3d(&x, &w, &bias, dims, (oc, kt, kh, kw), stride);
            let mut out = up(&vec![0.0; want.len()])?;
            let (dx, dw, db) = (up(&x)?, up(&w)?, up(&bias)?);
            unsafe {
                k::launch_causal_conv3d(
                    &dev()?.stream, &dev()?.kernels.causal_conv3d_f32, &dx, &dw, Some(&db), &mut out,
                    n as i32, ic as i32, it as i32, ih as i32, iw as i32,
                    oc as i32, kt as i32, kh as i32, kw as i32,
                    ot as i32, oh as i32, ow as i32,
                    stride.0 as i32, stride.1 as i32, stride.2 as i32,
                )?;
            }
            c.cmp(&format!("causal_conv3d_{n}x{ic}x{it}x{ih}x{iw}_s{}", stride.1), &down(&out)?, &want, op)?;
        }

        Ok(())
    })?;
    group(&mut c, "cudnn_conv2d", |c: &mut Ctx<'_>| -> StageResult<()> {
        // cuDNN conv2d (patch embed / VAE resample host-upload path).
        {
            let (n, ci, h, w, co, kh, kw) = (2usize, 3usize, 17usize, 20usize, 6usize, 3usize, 3usize);
            let x = c.rand(n * ci * h * w, 1.0);
            let wt = c.rand(co * ci * kh * kw, 0.2);
            for (pad, stride) in [(0usize, 1usize), (1, 1), (1, 2)] {
                let got = device::conv2d_f32(&x, &wt, n, ci, h, w, co, kh, kw, pad, stride)?;
                // Reference via conv3d with kt=1 on explicitly zero-padded input.
                let (ph, pw) = (h + 2 * pad, w + 2 * pad);
                let mut xp = vec![0.0; n * ci * ph * pw];
                for b in 0..n {
                    for cc in 0..ci {
                        for y in 0..h {
                            for xx in 0..w {
                                xp[((b * ci + cc) * ph + y + pad) * pw + xx + pad] = x[((b * ci + cc) * h + y) * w + xx];
                            }
                        }
                    }
                }
                let (want, _) = ref_conv3d(&xp, &wt, &vec![0.0; co], (n, ci, 1, ph, pw), (co, 1, kh, kw), (1, stride, stride));
                c.cmp(&format!("cudnn_conv2d_p{pad}_s{stride}"), &got, &want, gemm.max(op))?;
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// NVRTC compile of the kernel module for each architecture. Needs only
/// `libnvrtc` (no GPU, no driver), so it runs in CI or any Linux container.
pub fn nvrtc(report: &mut Report, archs: &[(i32, i32)]) -> StageResult<()> {
    for &(maj, min) in archs {
        let t = std::time::Instant::now();
        let result = k::compile_ptx(maj, min);
        let secs = t.elapsed().as_secs_f64();
        let (ok, err) = match &result {
            Ok(_) => (true, serde_json::Value::Null),
            Err(e) => (false, json!(e.to_string())),
        };
        report.check(
            format!("compile_sm{maj}{min}"),
            ok,
            json!({"seconds": secs, "error": err, "kernels": k::KERNEL_NAMES.len()}),
            json!({}),
        )?;
    }
    Ok(())
}
