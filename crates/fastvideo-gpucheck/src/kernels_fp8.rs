//! Kernel checks for the reference FP8 recipes ([`fastvideo_cudarc::wan::quant`])
//! and the bf16-native activation kernels (`FASTVIDEO_BF16_ACT`).
//!
//! * **fp8_recipes** — the W8A8 and MXFP8 quantizers bit for bit against the
//!   host twins of the Python quantizers (FastVideo `fp8_config.py`
//!   `_quantize_tensorwise`, Sol-H3 `mxfp8.py`); the cuBLASLt GEMMs against a
//!   host emulation of the same quantized operands and against the bf16
//!   product at FastVideo's `test_mxfp8.py` tolerance (cosine > 0.995,
//!   rel-L2 < 0.10); the fused MXFP8 producers; timing at H3 FFN shapes.
//! * **bf16_act** — every bf16-native kernel against its host rounding-point
//!   reference: exact where the op is a single IEEE operation, else within one
//!   bf16 ulp (transcendentals and rsqrt differ from the host by f32 ulps).
//! * **bf16_attention_routes** — every attention route the H3 / LTX-2 DiTs can
//!   take accepts bf16-stored q/k/v on the device (no host round trip) and
//!   matches the same route on f32-stored copies of the same values.

use cudarc::driver::CudaSlice;
use fastvideo_cudarc::h3::fused16::{self, AdaRows, NormOut};
use fastvideo_cudarc::wan::device;
use fastvideo_cudarc::wan::nn::Linear;
use fastvideo_cudarc::wan::quant::{self, QuantKind, QuantLayout, Section};
use fastvideo_cudarc::wan::tensor::with_bf16_act;
use fastvideo_cudarc::CudaTensor;
use serde_json::json;

use crate::metrics::diff;
use crate::rand_weights::randn;
use crate::report::{Report, StageResult};

fn dev() -> anyhow::Result<std::sync::Arc<device::DeviceContext>> {
    device::global_device().ok_or_else(|| anyhow::anyhow!("no live CUDA device"))
}

fn rand(seed: &mut u64, n: usize, std: f32) -> Vec<f32> {
    *seed += 1;
    randn(*seed, n, std)
}

fn bf16v(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(quant::bf16_round).collect()
}

fn to_bf16(v: &[f32]) -> Vec<half::bf16> {
    v.iter().map(|&x| half::bf16::from_f32(x)).collect()
}

/// A device bf16 tensor holding `v` (already bf16 values).
fn t16(v: &[f32], shape: &[usize]) -> anyhow::Result<CudaTensor> {
    let s = dev()?.stream.memcpy_stod(&to_bf16(v))?;
    Ok(CudaTensor::from_device_slice_bf16(s, shape.to_vec())?)
}

fn t32(v: &[f32], shape: &[usize]) -> anyhow::Result<CudaTensor> {
    Ok(CudaTensor::from_vec(v.to_vec(), shape.to_vec())?.to_device()?)
}

fn host(t: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.host_cow()?.into_owned())
}

/// One bf16 ulp at `w`'s magnitude.
fn ulp16(w: f32) -> f32 {
    let a = w.abs().max(f32::MIN_POSITIVE);
    2f32.powi(a.log2().floor() as i32 - 7)
}

/// `(max |got - want| in bf16 ulps, count above one ulp)`.
fn ulps(got: &[f32], want: &[f32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut over = 0usize;
    for (&g, &w) in got.iter().zip(want) {
        let u = if g == w {
            0.0
        } else {
            (g - w).abs() / ulp16(w)
        };
        let u = if u.is_nan() { f32::INFINITY } else { u };
        worst = worst.max(u);
        over += usize::from(u > 1.0);
    }
    (worst, over)
}

fn check_ulps(
    report: &mut Report,
    name: &str,
    got: &[f32],
    want: &[f32],
    exact: bool,
) -> StageResult<()> {
    let (worst, over) = ulps(got, want);
    let ok = got.len() == want.len() && if exact { worst == 0.0 } else { over == 0 };
    report.check(
        name,
        ok,
        json!({"max_ulps": worst, "over_1ulp": over, "n": got.len()}),
        json!({"max_ulps": if exact { 0 } else { 1 }}),
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-300)
}

/// `x [m, k] @ w [n, k]ᵀ` in f64 (bf16 reference with exact accumulation).
fn matmul_ref(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for (i, row) in out.chunks_exact_mut(n).enumerate() {
        for (j, o) in row.iter_mut().enumerate() {
            *o = (0..k)
                .map(|t| f64::from(x[i * k + t]) * f64::from(w[j * k + t]))
                .sum::<f64>() as f32;
        }
    }
    out
}

fn sync() -> anyhow::Result<()> {
    dev()?.stream.synchronize()?;
    Ok(())
}

/// Median wall time of `f` over `iters` synchronized runs, in ms.
fn time_ms(iters: usize, mut f: impl FnMut() -> anyhow::Result<()>) -> anyhow::Result<f64> {
    f()?;
    sync()?;
    let mut t = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = std::time::Instant::now();
        f()?;
        sync()?;
        t.push(s.elapsed().as_secs_f64() * 1e3);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[t.len() / 2])
}

pub fn fp8_recipes(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    let d = dev()?;
    // 1. W8A8 (`_quantize_tensorwise`): codes, the f32 scale, zero padding.
    for (label, n, std) in [
        ("normal", 65_536usize, 3.0f32),
        ("tiny", 4096, 1e-7),
        ("zero", 512, 0.0),
    ] {
        let x = bf16v(rand(seed, n, std));
        let xd = d.stream.memcpy_stod(&to_bf16(&x))?;
        let pad = 16;
        let mut q = d.stream.alloc_zeros::<u8>(n + pad)?;
        let mut s = d.stream.alloc_zeros::<f32>(1)?;
        quant::w8a8_quantize_raw(
            quant::ptr(&xd),
            true,
            n,
            quant::ptr_mut(&mut q),
            n + pad,
            quant::ptr_mut(&mut s),
        )?;
        let got = d.stream.memcpy_dtov(&q)?;
        let got_scale = d.stream.memcpy_dtov(&s)?[0];
        let want_scale = quant::w8a8_scale(quant::amax_abs(&x));
        let want = quant::w8a8_quantize(&x, want_scale);
        let mism = got[..n].iter().zip(&want).filter(|(a, b)| a != b).count();
        let pad_ok = got[n..].iter().all(|&b| b == 0);
        report.check(
            format!("w8a8_quantize_bit_exact_{label}"),
            mism == 0 && pad_ok && got_scale.to_bits() == want_scale.to_bits(),
            json!({"mismatched_codes": mism, "n": n, "scale": got_scale, "want_scale": want_scale, "pad_zero": pad_ok}),
            json!({"mismatched_codes": 0}),
        )?;
    }
    // 2. MXFP8 (`_mxfp8_quant_kernel`): codes and swizzled E8M0 bytes, with a
    //    zero block, a tiny block and a large block, rows not a multiple of 128.
    {
        let (rows, k) = (300usize, 5376usize);
        let mut x = rand(seed, rows * k, 1.0);
        x[..32].fill(0.0);
        x[32..64].iter_mut().for_each(|v| *v *= 1e-30);
        x[64..96].iter_mut().for_each(|v| *v *= 3e4);
        let x = bf16v(x);
        let xd = d.stream.memcpy_stod(&to_bf16(&x))?;
        let mut act = quant::MxAct::alloc(rows, k)?;
        quant::mxfp8_quantize_raw(quant::ptr(&xd), true, &mut act)?;
        let (wq, ws) = quant::mxfp8_quantize(&x, rows, k);
        let gq = d.stream.memcpy_dtov(&act.q)?;
        let gs = d.stream.memcpy_dtov(&act.s)?;
        let mq = gq[..rows * k]
            .iter()
            .zip(&wq)
            .filter(|(a, b)| a != b)
            .count();
        let ms = gs[..ws.len()]
            .iter()
            .zip(&ws)
            .filter(|(a, b)| a != b)
            .count();
        report.check(
            "mxfp8_quantize_bit_exact",
            mq == 0 && ms == 0,
            json!({"mismatched_codes": mq, "mismatched_scales": ms, "rows": rows, "k": k}),
            json!({"mismatched_codes": 0, "mismatched_scales": 0}),
        )?;
    }
    // 3. GEMMs: the quantized linear vs a host emulation of the same operands
    //    (only accumulation order and the bf16 output differ), and vs the bf16
    //    product at the reference test's tolerance. W8A8 runs a fused QKV-like
    //    stack (three tensor scales and a bf16 gate section: strided outputs).
    for kind in [QuantKind::W8A8, QuantKind::Mxfp8] {
        let (m, k, sec) = (200usize, 1024usize, 256usize);
        let sections = match kind {
            QuantKind::W8A8 => vec![
                Section {
                    rows: sec,
                    quantized: true,
                },
                Section {
                    rows: sec,
                    quantized: true,
                },
                Section {
                    rows: sec,
                    quantized: true,
                },
                Section {
                    rows: sec,
                    quantized: false,
                },
            ],
            QuantKind::Mxfp8 => vec![Section {
                rows: 3 * sec,
                quantized: true,
            }],
        };
        let n: usize = sections.iter().map(|s| s.rows).sum();
        // Per-section magnitudes differ, so one scale per tensor matters.
        let mut w = rand(seed, n * k, 0.02);
        for (i, v) in w.iter_mut().enumerate() {
            *v *= [1.0f32, 4.0, 0.25, 1.0][(i / k) / sec % 4];
        }
        let w = bf16v(w);
        let x = bf16v(rand(seed, m * k, 1.0));
        let mut lin = Linear::from_tensors(CudaTensor::from_vec(w.clone(), vec![n, k])?, None)?;
        lin.quantize(kind, sections.clone())?;
        let got = with_bf16_act(true, || {
            lin.forward(&t16(&x, &[1, m, k])?)
                .map_err(anyhow::Error::from)
        })?;
        let is16 = got.is_bf16();
        let got = host(&got)?;
        let layout = QuantLayout::new(kind, k, sections)?;
        let (blob, scales) = layout.quantize_host(&w)?;
        let emu = layout.forward_host(&blob, &scales, &x, m);
        let dq = diff(&got, &emu);
        report.check(
            format!("{kind:?}_gemm_matches_quantized_emulation").to_lowercase(),
            dq.within(1e-2) && is16,
            json!({"diff": dq.to_json(), "bf16_output": is16, "m": m, "k": k, "n": n}),
            json!({"rel_l2": 1e-2}),
        )?;
        let dense = matmul_ref(&x, &w, m, k, n);
        let (cos, dd) = (cosine(&got, &dense), diff(&got, &dense));
        report.check(
            format!("{kind:?}_gemm_vs_bf16_reference").to_lowercase(),
            cos > 0.995 && dd.rel_l2 < 0.10,
            json!({"cosine": cos, "rel_l2": dd.rel_l2}),
            json!({"cosine_min": 0.995, "rel_l2_max": 0.10, "source": "FastVideo tests/ops/quantization/test_mxfp8.py"}),
        )?;
    }
    // 3b. The generic (LTX) quantized linear: bias and GELU epilogue on f32
    //     and bf16 activations against the host emulation, and a second
    //     forward on the same input (the shared, cached activation
    //     quantization Q/K/V use) bit-identical to the first.
    {
        let (m, k, n) = (200usize, 1024usize, 512usize);
        let w = bf16v(rand(seed, n * k, 0.02));
        let b = bf16v(rand(seed, n, 0.5));
        let x = bf16v(rand(seed, m * k, 1.0));
        let sections = vec![Section {
            rows: n,
            quantized: true,
        }];
        let mut lin = Linear::from_tensors(
            CudaTensor::from_vec(w.clone(), vec![n, k])?,
            Some(CudaTensor::from_vec(b.clone(), vec![n])?),
        )?;
        lin.quantize(QuantKind::W8A8, sections.clone())?;
        let layout = QuantLayout::new(QuantKind::W8A8, k, sections)?;
        let (blob, scales) = layout.quantize_host(&w)?;
        let emu = layout.forward_host(&blob, &scales, &x, m);
        for (act16, gelu) in [(false, false), (false, true), (true, false), (true, true)] {
            let want = fastvideo_cudarc::wan::ops::host::quant_linear_epilogue(&emu, Some(&b), gelu);
            let xt = if act16 {
                t16(&x, &[1, m, k])?
            } else {
                t32(&x, &[1, m, k])?
            };
            let run = || {
                with_bf16_act(act16, || {
                    if gelu {
                        lin.forward_gelu(&xt)
                    } else {
                        lin.forward(&xt)
                    }
                    .map_err(anyhow::Error::from)
                })
            };
            let first = run()?;
            let second = run()?;
            let dtype_ok = first.is_bf16() == act16;
            let (a, a2) = (host(&first)?, host(&second)?);
            let dq = diff(&a, &want);
            report.check(
                format!(
                    "w8a8_linear_bias{}_{}_matches_emulation",
                    if gelu { "_gelu" } else { "" },
                    if act16 { "bf16" } else { "f32" }
                ),
                dq.within(1e-2) && a == a2 && dtype_ok,
                json!({"diff": dq.to_json(), "cached_rerun_identical": a == a2, "dtype_ok": dtype_ok}),
                json!({"rel_l2": 1e-2}),
            )?;
        }
    }
    // 4. Fused MXFP8 producers vs host bf16 value + host quantizer: compare
    //    the dequantized activations (the bf16 value may differ by one ulp
    //    where rsqrt / exp differ, which can move a code).
    {
        let (rows, dim) = (300usize, 5376usize);
        let x = bf16v(rand(seed, rows * dim, 1.0));
        let w = bf16v(rand(seed, dim, 0.1).into_iter().map(|v| 1.0 + v).collect());
        let (tab, idx) = small_table(seed, dim, rows);
        let ada = ada_rows(&tab, &idx, dim)?;
        let xt = t16(&x, &[1, rows, dim])?;
        let wt = t32(&w, &[dim])?;
        let NormOut::Mx(act) = fused16::norm_mod(&xt, &wt, &ada, 0, 1, 0, 1e-6, true)? else {
            return Err(anyhow::anyhow!("norm_mod declined MXFP8").into());
        };
        let host_rows = host_norm_mod(&x, &w, &tab, &idx, dim);
        let (hq, hs) = quant::mxfp8_quantize(&host_rows, rows, dim);
        let want = quant::mxfp8_dequantize(&hq, &hs, rows, dim);
        let got = mx_host(&act)?;
        let dm = diff(&got, &want);
        report.check(
            "h3_norm_mod_mxfp8_matches_host",
            dm.within(2e-3),
            json!({"diff": dm.to_json()}),
            json!({"rel_l2": 2e-3}),
        )?;
        let half = 1024usize;
        let h = bf16v(rand(seed, rows * 2 * half, 2.0));
        let act = fused16::swiglu_mx(&t16(&h, &[1, rows, 2 * half])?)?
            .ok_or_else(|| anyhow::anyhow!("swiglu_mx declined"))?;
        let sw: Vec<f32> = h
            .chunks_exact(2 * half)
            .flat_map(quant::swiglu_row)
            .collect();
        let (hq, hs) = quant::mxfp8_quantize(&sw, rows, half);
        let want = quant::mxfp8_dequantize(&hq, &hs, rows, half);
        let dm = diff(&mx_host(&act)?, &want);
        report.check(
            "h3_swiglu_mxfp8_matches_host",
            dm.within(2e-3),
            json!({"diff": dm.to_json()}),
            json!({"rel_l2": 2e-3}),
        )?;
    }
    // 5. Timing at the H3 FFN shapes (5 s, 1344x768: 37756 packed rows).
    ffn_timing(report, seed)?;
    Ok(())
}

/// A `[2 * 1 * 3, 6, dim]` table (one block, ladder + keyframe) and a row index.
fn small_table(seed: &mut u64, dim: usize, rows: usize) -> (Vec<f32>, Vec<u32>) {
    let t = 6usize;
    let mut tab = bf16v(rand(seed, t * 6 * dim, 0.3));
    for r in 0..t {
        for slot in [1usize, 4] {
            for v in &mut tab[(r * 6 + slot) * dim..(r * 6 + slot + 1) * dim] {
                *v += 1.0; // SCALE slots hold 1 + scale
            }
        }
    }
    let idx = (0..rows).map(|r| (r % t) as u32).collect();
    (tab, idx)
}

fn ada_rows(tab: &[f32], idx: &[u32], dim: usize) -> anyhow::Result<AdaRows> {
    Ok(AdaRows {
        tab: t32(tab, &[tab.len() / (6 * dim), 6, dim])?,
        hidden: dim,
        idx: std::sync::Arc::new(idx.to_vec()),
        idx_dev: Some(std::sync::Arc::new(dev()?.stream.memcpy_stod(idx)?)),
    })
}

fn host_norm_mod(x: &[f32], w: &[f32], tab: &[f32], idx: &[u32], dim: usize) -> Vec<f32> {
    let row = |r: usize, slot: usize| &tab[(idx[r] as usize * 6 + slot) * dim..][..dim];
    x.chunks_exact(dim)
        .enumerate()
        .flat_map(|(r, xr)| quant::norm_mod_row(xr, w, row(r, 1), row(r, 0), 1e-6))
        .collect()
}

fn mx_host(act: &quant::MxAct) -> anyhow::Result<Vec<f32>> {
    let d = dev()?;
    let q = d.stream.memcpy_dtov(&act.q)?;
    let s = d.stream.memcpy_dtov(&act.s)?;
    Ok(quant::mxfp8_dequantize(
        &q[..act.rows * act.k],
        &s,
        act.rows,
        act.k,
    ))
}

/// bf16 vs W8A8 vs MXFP8 GEMM time for the H3 FFN up / down projections,
/// plus the activation quantizers. INFO only.
fn ffn_timing(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    let d = dev()?;
    let cfg = fastvideo_models::h3::config::H3TransformerConfig::fasth3_8step();
    let tokens = 37_756usize;
    let base = bf16v(rand(seed, 1 << 20, 1.0));
    let fill = |n: usize, scale: f32| -> anyhow::Result<CudaSlice<half::bf16>> {
        let v: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(base[i % base.len()] * scale))
            .collect();
        Ok(d.stream.memcpy_stod(&v)?)
    };
    for (name, k, n) in [
        ("ff_in", cfg.hidden_size, 2 * cfg.ffn_dim),
        ("ff_out", cfg.ffn_dim, cfg.hidden_size),
    ] {
        let x16 = fill(tokens * k, 1.0)?;
        let w16 = fill(n * k, 0.02)?;
        let flops = 2.0 * tokens as f64 * k as f64 * n as f64;
        let mut row = serde_json::Map::new();
        row.insert("tokens".into(), json!(tokens));
        row.insert("k".into(), json!(k));
        row.insert("n".into(), json!(n));
        {
            let mut out = unsafe { d.stream.alloc::<half::bf16>(tokens * n) }?;
            let ms = time_ms(5, || {
                device::matmul_linear_wt_bf16(&x16, &w16, &mut out, tokens, k, n)?;
                Ok(())
            })?;
            row.insert("bf16_ms".into(), json!(ms));
            row.insert("bf16_tflops".into(), json!(flops / ms / 1e9));
        }
        let x = CudaTensor::from_device_slice_bf16(x16.clone(), vec![1, tokens, k])?;
        for kind in [QuantKind::W8A8, QuantKind::Mxfp8] {
            let tag = format!("{kind:?}").to_lowercase();
            let wt = CudaTensor::from_device_slice_bf16(w16.clone(), vec![n, k])?;
            let mut lin = Linear::from_tensors(wt, None)?;
            match lin.quantize(
                kind,
                vec![Section {
                    rows: n,
                    quantized: true,
                }],
            ) {
                Ok(()) => {}
                Err(e) => {
                    row.insert(format!("{tag}_error"), json!(e.to_string()));
                    continue;
                }
            }
            match time_ms(5, || {
                with_bf16_act(true, || lin.forward(&x))?;
                Ok(())
            }) {
                Ok(ms) => {
                    row.insert(format!("{tag}_ms"), json!(ms));
                    row.insert(format!("{tag}_tflops"), json!(flops / ms / 1e9));
                }
                Err(e) => {
                    row.insert(format!("{tag}_error"), json!(e.to_string()));
                }
            }
        }
        // The activation quantizers alone (included in the linear times above).
        {
            let mut act = quant::MxAct::alloc(tokens, k)?;
            let ms = time_ms(5, || {
                quant::mxfp8_quantize_raw(quant::ptr(&x16), true, &mut act)?;
                Ok(())
            })?;
            row.insert("mxfp8_quantize_ms".into(), json!(ms));
            let mut q = d
                .stream
                .alloc_zeros::<u8>(tokens.next_multiple_of(16) * k)?;
            let mut s = d.stream.alloc_zeros::<f32>(1)?;
            let (qp, sp) = (quant::ptr_mut(&mut q), quant::ptr_mut(&mut s));
            let ms = time_ms(5, || {
                quant::w8a8_quantize_raw(
                    quant::ptr(&x16),
                    true,
                    tokens * k,
                    qp,
                    tokens.next_multiple_of(16) * k,
                    sp,
                )?;
                Ok(())
            })?;
            row.insert("w8a8_quantize_ms".into(), json!(ms));
        }
        report.note(
            format!("h3_ffn_{name}_timing"),
            serde_json::Value::Object(row),
        );
    }
    Ok(())
}

pub fn bf16_act(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    with_bf16_act(true, || bf16_act_inner(report, seed))?;
    block_timing(report, seed)
}

fn bf16_act_inner(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    let (s, dm) = (97usize, 640usize);
    let a = bf16v(rand(seed, s * dm, 1.5));
    let b = bf16v(rand(seed, s * dm, 1.5));
    let (at, bt) = (t16(&a, &[1, s, dm])?, t16(&b, &[1, s, dm])?);
    let r16 = quant::bf16_round;
    // Elementwise: one IEEE op then one rounding — exact.
    let got = at.add(&bt)?;
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| r16(x + y)).collect();
    report.check(
        "bf16_add_is_bf16",
        got.is_bf16(),
        json!({"bf16": got.is_bf16()}),
        json!({"bf16": true}),
    )?;
    check_ulps(report, "bf16_add", &host(&got)?, &want, true)?;
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| r16(x * y)).collect();
    check_ulps(report, "bf16_mul", &host(&at.mul(&bt)?)?, &want, true)?;
    let want: Vec<f32> = a.iter().map(|&x| r16(x / (1.0 + (-x).exp()))).collect();
    check_ulps(report, "bf16_silu", &host(&at.silu())?, &want, false)?;
    // Norms: f32 statistics, one rounding.
    let w = bf16v(rand(seed, dm, 0.2).into_iter().map(|v| 1.0 + v).collect());
    let wt = t32(&w, &[dm])?;
    let want: Vec<f32> = a
        .chunks_exact(dm)
        .flat_map(|row| {
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let r = 1.0 / (ss / dm as f32 + 1e-6).sqrt();
            row.iter().zip(&w).map(move |(&v, &wv)| r16(v * r * wv))
        })
        .collect();
    check_ulps(
        report,
        "bf16_rms_norm",
        &host(&at.rms_norm(&wt, 1e-6)?)?,
        &want,
        false,
    )?;
    // Gated residual: eager rounding points (product, then sum).
    let e = bf16v(rand(seed, 3 * dm, 0.5));
    let et = t32(&e, &[1, 3, dm])?;
    let got = at.residual_gate_add_e(&bt, &et, 2)?;
    let want: Vec<f32> = (0..s * dm)
        .map(|i| quant::gate_residual_eager(a[i], e[2 * dm + i % dm], b[i]))
        .collect();
    check_ulps(
        report,
        "bf16_residual_gate_add_e",
        &host(&got)?,
        &want,
        true,
    )?;
    // SwiGLU (value first, Sol-H3 form).
    let packed = t16(&a, &[1, s, dm])?;
    let want: Vec<f32> = a.chunks_exact(dm).flat_map(quant::swiglu_row).collect();
    check_ulps(
        report,
        "bf16_swiglu",
        &host(&packed.swiglu_value_first()?)?,
        &want,
        false,
    )?;
    // Data movement stays bf16 and exact.
    let n = at.narrow(2, 64, 128)?;
    let want: Vec<f32> = a
        .chunks_exact(dm)
        .flat_map(|r| r[64..192].to_vec())
        .collect();
    check_ulps(report, "bf16_narrow", &host(&n)?, &want, true)?;
    let c = CudaTensor::cat(&[&at, &bt], 1)?;
    let want: Vec<f32> = a.iter().chain(&b).copied().collect();
    check_ulps(report, "bf16_cat", &host(&c)?, &want, true)?;
    let heads = 5usize;
    let d = 128usize;
    let qkv = bf16v(rand(seed, s * 3 * heads * d, 1.0));
    let qt = t16(&qkv, &[1, s, 3 * heads * d])?;
    let v = qt.split_heads_bhsd(2 * heads * d, heads, d)?;
    let merged = v.merge_heads()?;
    let want: Vec<f32> = qkv
        .chunks_exact(3 * heads * d)
        .flat_map(|r| r[2 * heads * d..].to_vec())
        .collect();
    report.check(
        "bf16_split_merge_heads_dtype",
        v.is_bf16() && merged.is_bf16(),
        json!({"split_bf16": v.is_bf16(), "merge_bf16": merged.is_bf16()}),
        json!({"bf16": true}),
    )?;
    check_ulps(
        report,
        "bf16_split_merge_heads",
        &host(&merged)?,
        &want,
        true,
    )?;
    // H3 fused kernels vs the fused16 host twins (same rounding points).
    let norm_q = t32(
        &bf16v(rand(seed, d, 0.1).into_iter().map(|v| 1.0 + v).collect()),
        &[d],
    )?;
    let r = 96usize;
    let ang: Vec<f32> = (0..s * r).map(|i| (i as f32 * 0.013).sin()).collect();
    let cos: Vec<f32> = ang.iter().map(|a| a.cos()).collect();
    let sin: Vec<f32> = ang.iter().map(|a| a.sin()).collect();
    let (ct, st) = (t32(&cos, &[s, r])?, t32(&sin, &[s, r])?);
    let got = fused16::qk_norm_rope(&qt, &norm_q, Some((&ct, &st)), heads, d, heads * d, 1e-6)?;
    let host_packed = CudaTensor::from_vec(qkv.clone(), vec![1, s, 3 * heads * d])?;
    let host_want = {
        let hc = CudaTensor::from_vec(cos.clone(), vec![s, r])?;
        let hs = CudaTensor::from_vec(sin.clone(), vec![s, r])?;
        let hw = CudaTensor::from_vec(host(&norm_q)?, vec![d])?;
        host_twin(|| {
            Ok(fused16::qk_norm_rope(
                &host_packed,
                &hw,
                Some((&hc, &hs)),
                heads,
                d,
                heads * d,
                1e-6,
            )?)
        })?
    };
    check_ulps(report, "h3_qk_norm_rope", &host(&got)?, &host_want, false)?;
    let (tab, idx) = small_table(seed, dm, s);
    let ada = ada_rows(&tab, &idx, dm)?;
    let host_ada = AdaRows {
        tab: CudaTensor::from_vec(tab.clone(), vec![6, 6, dm])?,
        hidden: dm,
        idx: std::sync::Arc::new(idx.clone()),
        idx_dev: None,
    };
    let host_a = CudaTensor::from_vec(a.clone(), vec![1, s, dm])?;
    let host_b = CudaTensor::from_vec(b.clone(), vec![1, s, dm])?;
    let host_w = CudaTensor::from_vec(w.clone(), vec![dm])?;
    let NormOut::T(got) = fused16::norm_mod(&at, &wt, &ada, 0, 1, 0, 1e-6, false)? else {
        return Err(anyhow::anyhow!("norm_mod returned MXFP8 when bf16 was asked").into());
    };
    let want = host_twin(|| {
        match fused16::norm_mod(&host_a, &host_w, &host_ada, 0, 1, 0, 1e-6, false)? {
            NormOut::T(t) => Ok(t),
            NormOut::Mx(_) => Err(anyhow::anyhow!("host MX").into()),
        }
    })?;
    // Bit-reproducible normalizer (quant::row_rsqrt): exact, not 1 ulp.
    check_ulps(report, "h3_norm_mod", &host(&got)?, &want, true)?;
    let (hid, n2) = fused16::res_gate_norm_mod(&at, &bt, &wt, &ada, 0, (2, 4, 3), 1e-6, false)?;
    let NormOut::T(n2) = n2 else {
        return Err(anyhow::anyhow!("res_gate_norm_mod returned MXFP8").into());
    };
    let (want_h, want_n) = {
        let (h, n) = fused16::res_gate_norm_mod(
            &host_a,
            &host_b,
            &host_w,
            &host_ada,
            0,
            (2, 4, 3),
            1e-6,
            false,
        )?;
        let NormOut::T(n) = n else {
            return Err(anyhow::anyhow!("host MX").into());
        };
        (host(&h)?, host(&n)?)
    };
    check_ulps(report, "h3_res_gate_hidden", &host(&hid)?, &want_h, true)?;
    check_ulps(report, "h3_res_gate_norm_mod", &host(&n2)?, &want_n, true)?;
    let got = fused16::gate_residual(&at, &bt, &ada, 0, 5)?;
    let want = host(&fused16::gate_residual(&host_a, &host_b, &host_ada, 0, 5)?)?;
    check_ulps(report, "h3_gate_residual", &host(&got)?, &want, true)?;
    // Linear: bf16 in, bf16 out, no cast sandwich (dtype), within GEMM noise.
    let (k, nout) = (dm, 384usize);
    let lw = bf16v(rand(seed, nout * k, 0.03));
    let lb = bf16v(rand(seed, nout, 0.1));
    let lin = Linear::from_tensors(
        CudaTensor::from_vec(lw.clone(), vec![nout, k])?,
        Some(CudaTensor::from_vec(lb.clone(), vec![nout])?),
    )?;
    let y = lin.forward(&at)?;
    let mut want = matmul_ref(&a, &lw, s, k, nout);
    for (i, v) in want.iter_mut().enumerate() {
        *v = r16(*v + lb[i % nout]);
    }
    let dl = diff(&host(&y)?, &want);
    report.check(
        "bf16_linear_native",
        y.is_bf16() && dl.within(1e-2),
        json!({"bf16": y.is_bf16(), "diff": dl.to_json()}),
        json!({"rel_l2": 1e-2}),
    )?;
    Ok(())
}

/// One DiT-block-shaped chain (norm + AdaLN modulate, QKV-sized linear,
/// fused-MMA attention, output linear, gated residual, GELU FFN, gated
/// residual) timed with f32 and with bf16 activations. The bf16 path must not
/// be slower: a regression here is a cast sandwich somewhere in the chain.
fn block_timing(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    let (s, dm, heads) = (8192usize, 2048usize, 16usize);
    let d = dm / heads;
    let x = bf16v(rand(seed, s * dm, 1.0));
    let lin = |seed: &mut u64, o: usize, i: usize| -> anyhow::Result<Linear> {
        Ok(Linear::from_tensors(
            CudaTensor::from_vec(bf16v(rand(seed, o * i, 0.02)), vec![o, i])?,
            Some(CudaTensor::from_vec(bf16v(rand(seed, o, 0.02)), vec![o])?),
        )?)
    };
    let (qkv, out, up, down) = (
        lin(seed, 3 * dm, dm)?,
        lin(seed, dm, dm)?,
        lin(seed, 4 * dm, dm)?,
        lin(seed, dm, 4 * dm)?,
    );
    let w = t32(
        &bf16v(rand(seed, dm, 0.1).into_iter().map(|v| 1.0 + v).collect()),
        &[dm],
    )?;
    let e = t32(&bf16v(rand(seed, 6 * dm, 0.1)), &[1, 6, dm])?;
    let run = |bf: bool| -> anyhow::Result<f64> {
        with_bf16_act(bf, || {
            let x0 = if bf {
                t16(&x, &[1, s, dm])?
            } else {
                t32(&x, &[1, s, dm])?
            };
            time_ms(3, || {
                let n = x0.ln_adaln_e(&e, 1, 0, 1e-6)?;
                let p = qkv.forward(&n)?;
                let q = p.split_heads_bhsd(0, heads, d)?;
                let k = p.split_heads_bhsd(dm, heads, d)?;
                let v = p.split_heads_bhsd(2 * dm, heads, d)?;
                let a = fastvideo_cudarc::wan::nn::scaled_dot_product_attention(&q, &k, &v, None)?;
                let a = out.forward(&a.merge_heads()?)?;
                let x1 = x0.residual_gate_add_e(&a, &e, 2)?;
                let h = up.forward_gelu(&x1.rms_norm(&w, 1e-6)?)?;
                let f = down.forward(&h)?;
                x1.residual_gate_add_e(&f, &e, 5)?;
                Ok(())
            })
        })
    };
    let (f32_ms, bf16_ms) = (run(false)?, run(true)?);
    let ratio = bf16_ms / f32_ms;
    report.check(
        "bf16_act_block_not_slower",
        ratio <= 1.05,
        json!({"f32_act_ms": f32_ms, "bf16_act_ms": bf16_ms, "ratio": ratio, "tokens": s, "dim": dm}),
        json!({"ratio_max": 1.05}),
    )
}

/// Run a fused16 function on host tensors (its host twin): the device path
/// only engages for device tensors, so host inputs take the reference.
fn host_twin(f: impl FnOnce() -> StageResult<CudaTensor>) -> StageResult<Vec<f32>> {
    Ok(host(&f()?)?)
}

pub fn bf16_attention_routes(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    with_bf16_act(true, || routes_inner(report, seed))
}

#[allow(clippy::type_complexity)]
fn routes_inner(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    use fastvideo_cudarc::wan::{attn, nn};
    let (b, h, s, d) = (1usize, 4usize, 1000usize, 128usize);
    let mk = |seed: &mut u64, n: usize| bf16v(rand(seed, n, 1.0));
    let (q, k, v) = (
        mk(seed, b * h * s * d),
        mk(seed, b * h * s * d),
        mk(seed, b * h * s * d),
    );
    let shape = [b, h, s, d];
    let (q16, k16, v16) = (t16(&q, &shape)?, t16(&k, &shape)?, t16(&v, &shape)?);
    let (q32, k32, v32) = (t32(&q, &shape)?, t32(&k, &shape)?, t32(&v, &shape)?);
    let mut route = |name: &str,
                     f: &dyn Fn(
        &CudaTensor,
        &CudaTensor,
        &CudaTensor,
    ) -> anyhow::Result<Option<CudaTensor>>,
                     (qa, ka, va): (&CudaTensor, &CudaTensor, &CudaTensor),
                     (qb, kb, vb): (&CudaTensor, &CudaTensor, &CudaTensor)|
     -> StageResult<()> {
        let got = f(qa, ka, va);
        let want = f(qb, kb, vb);
        match (got, want) {
            (Ok(Some(g)), Ok(Some(w))) => {
                let dd = diff(&host(&g)?, &host(&w)?);
                report.check(
                    format!("bf16_route_{name}"),
                    dd.within(1e-2) && g.is_device_fresh(),
                    json!({"diff": dd.to_json(), "out_bf16": g.is_bf16(), "on_device": g.is_device_fresh()}),
                    json!({"rel_l2": 1e-2}),
                )
            }
            (Ok(None), _) | (_, Ok(None)) => {
                report.note(format!("bf16_route_{name}_unavailable"), json!({}));
                Ok(())
            }
            (Err(e), _) => report.check(
                format!("bf16_route_{name}"),
                false,
                json!({"error": e.to_string()}),
                json!({}),
            ),
            (_, Err(e)) => report.check(
                format!("bf16_route_{name}_f32_twin"),
                false,
                json!({"error": e.to_string()}),
                json!({}),
            ),
        }
    };
    let bf = (&q16, &k16, &v16);
    let fl = (&q32, &k32, &v32);
    route(
        "sdpa_default",
        &|q, k, v| Ok(Some(nn::scaled_dot_product_attention(q, k, v, None)?)),
        bf,
        fl,
    )?;
    route(
        "fused_mma_sdpa",
        &|q, k, v| Ok(attn::device_mma_sdpa(q, k, v, None, true)?),
        bf,
        fl,
    )?;
    route(
        "dense_sdpa",
        &|q, k, v| Ok(attn::device_dense_sdpa(q, k, v, None)?),
        bf,
        fl,
    )?;
    route(
        "sol_attn",
        &|q, k, v| {
            Ok(Some(fastvideo_cudarc::sol_attn::sol_attn(
                q, k, v, 1.0, None, None, 0,
            )?))
        },
        bf,
        fl,
    )?;
    route(
        "sol_attn_spliced_prefix",
        &|q, k, v| {
            let o = fastvideo_cudarc::sol_attn::sol_attn(q, k, v, 1.0, None, None, 0)?;
            Ok(Some(fastvideo_cudarc::sol_attn::splice_dense_prefix(
                &o, q, k, v, 128, None,
            )?))
        },
        bf,
        fl,
    )?;
    route(
        "pisa",
        &|q, k, v| {
            Ok(Some(fastvideo_cudarc::pisa_attn::pisa_attn(
                q, k, v, 0.5, None,
            )?))
        },
        bf,
        fl,
    )?;
    // Cross attention: a shorter key/value sequence.
    let sk = 333usize;
    let (kc, vc) = (mk(seed, b * h * sk * d), mk(seed, b * h * sk * d));
    let kshape = [b, h, sk, d];
    let (kc16, vc16, kc32, vc32) = (
        t16(&kc, &kshape)?,
        t16(&vc, &kshape)?,
        t32(&kc, &kshape)?,
        t32(&vc, &kshape)?,
    );
    route(
        "cross_attention",
        &|q, k, v| Ok(Some(nn::scaled_dot_product_attention(q, k, v, None)?)),
        (&q16, &kc16, &vc16),
        (&q32, &kc32, &vc32),
    )?;
    // VSA-H3 (with the compression gate) over a small packed layout.
    {
        use fastvideo_cudarc::h3::vsa::{H3Vsa, H3VsaConfig};
        let layout =
            fastvideo_models::h3::packing::H3PackedLayout::new(70, (5, 8, 12), 33, [1, 2, 2])
                .map_err(anyhow::Error::msg)?;
        let seq = layout.sequence_length();
        let vs = [1, 2, seq, d];
        let n = 2 * seq * d;
        let (vq, vk, vv, vg) = (mk(seed, n), mk(seed, n), mk(seed, n), mk(seed, n));
        let vsa = H3Vsa::new(
            &layout,
            2,
            d,
            H3VsaConfig {
                sparsity: 0.5,
                group: 2,
                tile_size: 64,
            },
        )?;
        let run = |bf: bool| -> anyhow::Result<CudaTensor> {
            let mkt = |x: &[f32]| if bf { t16(x, &vs) } else { t32(x, &vs) };
            Ok(vsa.attend(mkt(&vq)?, mkt(&vk)?, mkt(&vv)?, Some(mkt(&vg)?))?)
        };
        match (run(true), run(false)) {
            (Ok(g), Ok(w)) => {
                let dd = diff(&host(&g)?, &host(&w)?);
                report.check(
                    "bf16_route_vsa_h3",
                    dd.within(1e-2) && g.is_bf16() && g.is_device_fresh(),
                    json!({"diff": dd.to_json(), "out_bf16": g.is_bf16()}),
                    json!({"rel_l2": 1e-2}),
                )?;
            }
            (Err(e), _) | (_, Err(e)) => report.check(
                "bf16_route_vsa_h3",
                false,
                json!({"error": e.to_string()}),
                json!({}),
            )?,
        }
    }
    Ok(())
}
