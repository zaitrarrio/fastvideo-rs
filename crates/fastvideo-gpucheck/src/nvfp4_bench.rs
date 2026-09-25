//! NVFP4 W4A4 GEMM on Blackwell, kernel level (no model path uses it yet):
//!
//! 1. the device quantizer is bit-identical to the host TransformerEngine
//!    `NVFP4BlockScaling` recipe (`fastvideo_models::nvfp4::quantize`,
//!    `static_6`; `static_4` and FourOverSix `mse` too);
//! 2. the cuBLASLt scale swizzle matches its host twin;
//! 3. cuBLASLt block-scaled FP4 (`VEC16_UE4M3`) and every embedded oxide
//!    Tile-IR cubin reproduce the host NVFP4 GEMM (sm >= 100 only);
//! 4. INFO timings at LTX-2.5 video FFN and H3 FFN shapes: bf16 cuBLAS vs
//!    cuBLASLt NVFP4 vs oxide NVFP4 (`FV_NVFP4_BENCH=0` skips).

use std::time::Instant;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use fastvideo_cudarc::wan::nvfp4_gemm::{self as g, Nvfp4Dev};
use fastvideo_cudarc::wan::{device, fp8, kernels::OxideGemm, ops};
use fastvideo_models::nvfp4::{self, ScaleRule};
use half::bf16;
use serde_json::json;

use crate::metrics::diff;
use crate::rand_weights::randn;
use crate::report::{Report, StageResult};

/// bf16 output rounding (2^-9 relative) plus accumulation order.
const BF16_OUT_LIMIT: f64 = 4e-3;
/// f32 output: accumulation order only.
const F32_OUT_LIMIT: f64 = 1e-5;

fn dev() -> anyhow::Result<std::sync::Arc<device::DeviceContext>> {
    device::global_device().ok_or_else(|| anyhow::anyhow!("no live CUDA device"))
}

fn bf16_to_f32(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

fn mismatches(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// Mean milliseconds per call: one warm-up, then enough calls for ~0.2 s.
fn time_ms(mut f: impl FnMut() -> anyhow::Result<()>) -> anyhow::Result<f64> {
    f()?;
    device::synchronize()?;
    let t = Instant::now();
    f()?;
    device::synchronize()?;
    let once = t.elapsed().as_secs_f64();
    let iters = ((0.2 / once.max(1e-6)) as usize).clamp(3, 50);
    let t = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    device::synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn upload(v: &[u8]) -> anyhow::Result<CudaSlice<u8>> {
    Ok(dev()?.stream.memcpy_stod(v)?)
}

/// Run one oxide variant into a fresh output; returns it as f32.
#[allow(clippy::too_many_arguments)]
fn run_oxide(
    v: &OxideGemm,
    a: &Nvfp4Dev,
    w_packed: &CudaSlice<u8>,
    w_scales: &CudaSlice<u8>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) -> anyhow::Result<Vec<f32>> {
    let d = dev()?;
    let (xp, _a) = a.packed.device_ptr(&d.stream);
    let (xs, _b) = a.scales.device_ptr(&d.stream);
    let (yp, _c) = w_packed.device_ptr(&d.stream);
    let (ys, _d) = w_scales.device_ptr(&d.stream);
    if v.out == "f32" {
        let mut z = d.stream.alloc_zeros::<f32>(m * n)?;
        {
            let (zp, _z) = z.device_ptr_mut(&d.stream);
            unsafe { g::oxide_gemm(v, zp, xp, yp, xs, ys, m, n, k, alpha)? };
        }
        Ok(d.stream.memcpy_dtov(&z)?)
    } else {
        let mut z = d.stream.alloc_zeros::<bf16>(m * n)?;
        {
            let (zp, _z) = z.device_ptr_mut(&d.stream);
            unsafe { g::oxide_gemm(v, zp, xp, yp, xs, ys, m, n, k, alpha)? };
        }
        Ok(bf16_to_f32(&d.stream.memcpy_dtov(&z)?))
    }
}

pub fn run(report: &mut Report, seed: u64) -> StageResult<()> {
    let d = dev()?;
    report.set(
        "oxide_cubins",
        fastvideo_cudarc::wan::kernels::oxide_cubins(),
    );
    report.set(
        "oxide_loaded",
        d.kernels
            .oxide_gemms
            .iter()
            .map(|g| g.label())
            .collect::<Vec<_>>(),
    );

    // ---- 1. quantize: device == host TE reference, bit for bit ----------------
    let (rows, cols) = (300usize, 5376usize);
    let mut x = randn(seed ^ 0x4e56_4650, rows * cols, 1.0);
    // Outliers so some blocks saturate the E4M3 scale and others underflow.
    x[17] = 40.0;
    x[5000] = -1e-6;
    for rule in [ScaleRule::Static6, ScaleRule::Static4, ScaleRule::Mse] {
        let host = nvfp4::quantize(&x, rows, cols, rule).map_err(anyhow::Error::msg)?;
        let q = match g::quantize_device(&d.stream.memcpy_stod(&x)?, rows, cols, rule) {
            Ok(q) => q,
            Err(e) => {
                report.check(
                    format!("nvfp4_quantize_{}_matches_te_reference", rule.as_str()),
                    false,
                    json!({"error": format!("{e:#}")}),
                    json!({}),
                )?;
                continue;
            }
        };
        let packed = d.stream.memcpy_dtov(&q.packed)?;
        let scales = d.stream.memcpy_dtov(&q.scales)?;
        let amax = d.stream.memcpy_dtov(&q.amax)?[0];
        let (mp, ms) = (
            mismatches(&packed, &host.packed),
            mismatches(&scales, &host.scales),
        );
        report.check(
            format!("nvfp4_quantize_{}_matches_te_reference", rule.as_str()),
            mp == 0 && ms == 0 && amax.to_bits() == host.amax.to_bits(),
            json!({"packed_mismatch": mp, "scale_mismatch": ms, "amax": amax,
                   "host_amax": host.amax, "rows": rows, "cols": cols}),
            json!({"packed_mismatch": 0, "scale_mismatch": 0}),
        )?;
    }

    // ---- 2. cuBLASLt scale swizzle -----------------------------------------------
    {
        let host =
            nvfp4::quantize(&x, rows, cols, ScaleRule::Static6).map_err(anyhow::Error::msg)?;
        let sw = g::swizzle_scales_device(&upload(&host.scales)?, rows, cols)?;
        let got = d.stream.memcpy_dtov(&sw)?;
        let want = g::swizzle_scales_host(&host.scales, rows, cols / 16);
        let mm = mismatches(&got, &want);
        report.check(
            "nvfp4_scale_swizzle_matches_host",
            mm == 0,
            json!({"mismatch": mm, "bytes": want.len()}),
            json!({"mismatch": 0}),
        )?;
    }

    if d.sm_major < 10 {
        report.note(
            "nvfp4_gemm_skipped",
            json!({"reason": format!("NVFP4 tensor cores need sm_100+, device is sm_{}{}", d.sm_major, d.sm_minor)}),
        );
        return Ok(());
    }

    // ---- 3. GEMM correctness vs the host NVFP4 GEMM ---------------------------------
    report.check(
        "nvfp4_oxide_cubins_loaded",
        !d.kernels.oxide_gemms.is_empty() && d.kernels.oxide_error.is_none(),
        json!({"loaded": d.kernels.oxide_gemms.len(), "error": d.kernels.oxide_error,
               "embedded": fastvideo_cudarc::wan::kernels::oxide_cubins()}),
        json!({"loaded": ">0"}),
    )?;
    let (m, n, k) = (512usize, 512usize, 1024usize);
    let xa = randn(seed ^ 0xa11ce, m * k, 1.0);
    let wf = randn(seed ^ 0xbeef, n * k, 0.05);
    let a = g::quantize_device(&d.stream.memcpy_stod(&xa)?, m, k, ScaleRule::Static6)?;
    let a_host = nvfp4::quantize(&xa, m, k, ScaleRule::Static6).map_err(anyhow::Error::msg)?;
    let w_host = nvfp4::quantize(&wf, n, k, ScaleRule::Static6).map_err(anyhow::Error::msg)?;
    // TE reference GEMM (f32, what the CPU path runs) and an f64 one over the
    // same dequantized operands, so the only thing measured is the kernel.
    let want_te = nvfp4::gemm(&a_host, &w_host).map_err(anyhow::Error::msg)?;
    let (ad, wd) = (a_host.dequantize(), w_host.dequantize());
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let acc: f64 = (0..k)
                .map(|t| f64::from(ad[i * k + t]) * f64::from(wd[j * k + t]))
                .sum();
            want[i * n + j] = acc as f32;
        }
    }
    let alpha = nvfp4::dequant_factor(a_host.amax, ScaleRule::Static6)
        * nvfp4::dequant_factor(w_host.amax, ScaleRule::Static6);
    let (wp, ws) = (upload(&w_host.packed)?, upload(&w_host.scales)?);
    for v in &d.kernels.oxide_gemms {
        let limit = if v.out == "f32" {
            F32_OUT_LIMIT
        } else {
            BF16_OUT_LIMIT
        };
        let got = match run_oxide(v, &a, &wp, &ws, m, n, k, alpha) {
            Ok(got) => got,
            Err(e) => {
                report.check(
                    format!("nvfp4_oxide_{}_matches_te_gemm", v.label()),
                    false,
                    json!({"error": format!("{e:#}")}),
                    json!({}),
                )?;
                continue;
            }
        };
        let dv = diff(&got, &want);
        report.check(
            format!("nvfp4_oxide_{}_matches_te_gemm", v.label()),
            dv.within(limit),
            json!({"vs_f64": dv.to_json(), "vs_te_f32": diff(&got, &want_te).to_json(),
                   "m": m, "n": n, "k": k}),
            json!({"rel_l2": limit}),
        )?;
    }
    {
        let ltc = fp8::lt_context(&d)?;
        let (wsw, xsw) = (
            g::swizzle_scales_device(&ws, n, k)?,
            g::swizzle_scales_device(&a.scales, m, k)?,
        );
        let mut c = d.stream.alloc_zeros::<bf16>(m * n)?;
        let result = (|| -> anyhow::Result<Vec<f32>> {
            let (wsp, _a) = wsw.device_ptr(&d.stream);
            let (xsp, _b) = xsw.device_ptr(&d.stream);
            let plan = unsafe { g::LtNvfp4Plan::new(&ltc, m, n, k, wsp, xsp, alpha, 1)? };
            {
                let (w_ptr, _c) = wp.device_ptr(&d.stream);
                let (x_ptr, _d) = a.packed.device_ptr(&d.stream);
                let (c_ptr, _e) = c.device_ptr_mut(&d.stream);
                unsafe { plan.run(&ltc, 0, w_ptr, x_ptr, c_ptr)? };
            }
            Ok(bf16_to_f32(&d.stream.memcpy_dtov(&c)?))
        })();
        match result {
            Ok(got) => {
                let dv = diff(&got, &want);
                report.check(
                    "nvfp4_cublaslt_matches_te_gemm",
                    dv.within(BF16_OUT_LIMIT),
                    json!({"vs_f64": dv.to_json(), "vs_te_f32": diff(&got, &want_te).to_json(),
                           "m": m, "n": n, "k": k}),
                    json!({"rel_l2": BF16_OUT_LIMIT}),
                )?;
            }
            Err(e) => report.check(
                "nvfp4_cublaslt_matches_te_gemm",
                false,
                json!({"error": format!("{e:#}")}),
                json!({}),
            )?,
        }
    }

    // ---- 4. timings ----------------------------------------------------------------------
    if std::env::var("FV_NVFP4_BENCH").is_ok_and(|v| v == "0") {
        report.note("nvfp4_bench_skipped", json!({"reason": "FV_NVFP4_BENCH=0"}));
        return Ok(());
    }
    // LTX-2.5 video FFN (ltx2/config.rs: inner 32x128 = 4096, ff_mult 4 ->
    // 16384; GELU, not gated) at 130560 and 32640 video tokens, and the H3
    // SwiGLU FFN (h3/config.rs: hidden 5376, ffn_dim 14336, fc_in fused to
    // 2 x 14336) at 32640 tokens.
    let shapes: [(&str, usize, usize, usize); 6] = [
        ("ltx25_ffn_up_m130560", 130_560, 4096, 16_384),
        ("ltx25_ffn_down_m130560", 130_560, 16_384, 4096),
        ("ltx25_ffn_up_m32640", 32_640, 4096, 16_384),
        ("ltx25_ffn_down_m32640", 32_640, 16_384, 4096),
        ("h3_ffn_in_m32640", 32_640, 5376, 28_672),
        ("h3_ffn_out_m32640", 32_640, 14_336, 5376),
    ];
    let mut table = Vec::new();
    for (i, &(name, m, k, n)) in shapes.iter().enumerate() {
        match bench_shape(seed.wrapping_add(i as u64 * 7919), m, k, n) {
            Ok(row) => {
                let agree = row["oxide_vs_cublaslt_rel_l2"].as_f64();
                table.push((name, row.clone()));
                report.note(format!("nvfp4_bench_{name}"), row);
                if let Some(r) = agree {
                    // Same packed operands, two bf16 outputs: only rounding may differ.
                    report.check(
                        format!("nvfp4_bench_{name}_oxide_agrees_cublaslt"),
                        r <= 2.0 * BF16_OUT_LIMIT,
                        json!({"rel_l2": r}),
                        json!({"rel_l2": 2.0 * BF16_OUT_LIMIT}),
                    )?;
                }
            }
            Err(e) => report.check(
                format!("nvfp4_bench_{name}/completed"),
                false,
                json!({"error": format!("{e:#}")}),
                json!({}),
            )?,
        }
    }
    eprintln!(
        "[INFO] nvfp4 GEMM (ms, TFLOPS) on sm_{}{}:",
        d.sm_major, d.sm_minor
    );
    eprintln!(
        "[INFO] {:<24} {:>18} {:>18} {:>18} {:>9} {:>30}",
        "shape (M,K,N)",
        "bf16 cuBLAS",
        "cuBLASLt NVFP4",
        "oxide NVFP4",
        "quant",
        "oxide tile / lt algo"
    );
    for (name, r) in &table {
        let cell = |ms: &str, tf: &str| match (r[ms].as_f64(), r[tf].as_f64()) {
            (Some(a), Some(b)) => format!("{a:8.3} ({b:6.0})"),
            _ => "n/a".to_string(),
        };
        eprintln!(
            "[INFO] {:<24} {:>18} {:>18} {:>18} {:>9} {:>30}",
            name,
            cell("bf16_ms", "bf16_tflops"),
            cell("cublaslt_ms", "cublaslt_tflops"),
            cell("oxide_ms", "oxide_tflops"),
            r["quantize_ms"]
                .as_f64()
                .map(|v| format!("{v:.3}"))
                .unwrap_or_default(),
            format!(
                "{} / {}",
                r["oxide_variant"].as_str().unwrap_or("-"),
                r["cublaslt_algo"]
            ),
        );
    }
    Ok(())
}

/// First and last 16 rows of a bf16 `[m, n]` output: the last rows are where
/// an i32 offset overflow would show at M=130560.
fn edge_rows(c: &CudaSlice<bf16>, m: usize, n: usize) -> anyhow::Result<Vec<f32>> {
    let s = dev()?.stream.clone();
    let rows = 16.min(m);
    let mut out = bf16_to_f32(&s.memcpy_dtov(&c.slice(0..rows * n))?);
    out.extend(bf16_to_f32(
        &s.memcpy_dtov(&c.slice((m - rows) * n..m * n))?,
    ));
    Ok(out)
}

/// Time one `[m, k] @ [n, k]ᵀ` three ways. Operands are generated on the
/// device (up to 8.6 GB of f32 activations at M=130560, K=16384).
fn bench_shape(seed: u64, m: usize, k: usize, n: usize) -> anyhow::Result<serde_json::Value> {
    let d = dev()?;
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = |ms: f64| flop / (ms * 1e-3) / 1e12;
    let mut row = serde_json::Map::new();
    row.insert("m".into(), json!(m));
    row.insert("k".into(), json!(k));
    row.insert("n".into(), json!(n));

    // bf16 cuBLAS (the production dense linear).
    let w = g::fill_uniform_device(n * k, seed ^ 1, 0.05)?;
    let wb = ops::cast_f32_bf16_device(&w)?;
    let (a, bf16_sample) = {
        let x = g::fill_uniform_device(m * k, seed, 1.0)?;
        let xb = ops::cast_f32_bf16_device(&x)?;
        let mut out = unsafe { d.stream.alloc::<bf16>(m * n)? };
        let ms = time_ms(|| {
            device::matmul_linear_wt_bf16(&xb, &wb, &mut out, m, k, n)?;
            Ok(())
        })?;
        row.insert("bf16_ms".into(), json!(ms));
        row.insert("bf16_tflops".into(), json!(tflops(ms)));
        let sample = edge_rows(&out, m, n)?;
        drop((xb, out));
        // Activation quantize (amax + pack) — what a W4A4 linear adds per call.
        let ms = time_ms(|| {
            g::quantize_device(&x, m, k, ScaleRule::Static6)?;
            Ok(())
        })?;
        row.insert("quantize_ms".into(), json!(ms));
        (g::quantize_device(&x, m, k, ScaleRule::Static6)?, sample)
    };
    drop(wb);
    let wq = g::quantize_device(&w, n, k, ScaleRule::Static6)?;
    drop(w);
    let alpha = a.decode()? * wq.decode()?;

    // cuBLASLt block-scaled FP4: best of the heuristic's top candidates.
    let mut lt_sample = None;
    let lt_result = (|| -> anyhow::Result<()> {
        let ltc = fp8::lt_context(&d)?;
        let wsw = g::swizzle_scales_device(&wq.scales, n, k)?;
        let xsw = g::swizzle_scales_device(&a.scales, m, k)?;
        let mut c = unsafe { d.stream.alloc::<bf16>(m * n)? };
        let (wsp, _a) = wsw.device_ptr(&d.stream);
        let (xsp, _b) = xsw.device_ptr(&d.stream);
        let plan = unsafe { g::LtNvfp4Plan::new(&ltc, m, n, k, wsp, xsp, alpha, 8)? };
        let mut best: Option<(usize, f64)> = None;
        let mut per_algo = Vec::new();
        for algo in 0..plan.algo_count() {
            let r = time_ms(|| {
                let (w_ptr, _c) = wq.packed.device_ptr(&d.stream);
                let (x_ptr, _d) = a.packed.device_ptr(&d.stream);
                let (c_ptr, _e) = c.device_ptr_mut(&d.stream);
                unsafe { plan.run(&ltc, algo, w_ptr, x_ptr, c_ptr)? };
                Ok(())
            });
            match r {
                Ok(ms) => {
                    per_algo.push(json!(ms));
                    if best.is_none_or(|(_, b)| ms < b) {
                        best = Some((algo, ms));
                    }
                }
                Err(e) => per_algo.push(json!(format!("{e:#}"))),
            }
        }
        let (algo, ms) = best.ok_or_else(|| anyhow::anyhow!("no cuBLASLt NVFP4 algorithm ran"))?;
        {
            let (w_ptr, _c) = wq.packed.device_ptr(&d.stream);
            let (x_ptr, _d) = a.packed.device_ptr(&d.stream);
            let (c_ptr, _e) = c.device_ptr_mut(&d.stream);
            unsafe { plan.run(&ltc, algo, w_ptr, x_ptr, c_ptr)? };
        }
        lt_sample = Some(edge_rows(&c, m, n)?);
        row.insert("cublaslt_ms".into(), json!(ms));
        row.insert("cublaslt_tflops".into(), json!(tflops(ms)));
        row.insert("cublaslt_algo".into(), json!(algo));
        row.insert("cublaslt_algo_ms".into(), json!(per_algo));
        Ok(())
    })();
    if let Err(e) = lt_result {
        row.insert("cublaslt_error".into(), json!(format!("{e:#}")));
    }

    // oxide Tile-IR: best embedded bf16 variant that fits.
    let mut ox_sample = None;
    let variants = g::oxide_variants("bf16")?;
    let mut per_variant = serde_json::Map::new();
    let mut best: Option<(OxideGemm, f64)> = None;
    let mut z = unsafe { d.stream.alloc::<bf16>(m * n)? };
    for v in &variants {
        if let Err(why) = g::oxide_fits(v, m, n, k) {
            per_variant.insert(v.label(), json!(why));
            continue;
        }
        let r = time_ms(|| {
            let (xp, _a) = a.packed.device_ptr(&d.stream);
            let (xs, _b) = a.scales.device_ptr(&d.stream);
            let (yp, _c) = wq.packed.device_ptr(&d.stream);
            let (ys, _d) = wq.scales.device_ptr(&d.stream);
            let (zp, _z) = z.device_ptr_mut(&d.stream);
            unsafe { g::oxide_gemm(v, zp, xp, yp, xs, ys, m, n, k, alpha)? };
            Ok(())
        });
        match r {
            Ok(ms) => {
                per_variant.insert(v.label(), json!(ms));
                if best.as_ref().is_none_or(|(_, b)| ms < *b) {
                    best = Some((v.clone(), ms));
                }
            }
            Err(e) => {
                per_variant.insert(v.label(), json!(format!("{e:#}")));
            }
        }
    }
    row.insert(
        "oxide_variant_ms".into(),
        serde_json::Value::Object(per_variant),
    );
    if let Some((v, ms)) = best {
        {
            let (xp, _a) = a.packed.device_ptr(&d.stream);
            let (xs, _b) = a.scales.device_ptr(&d.stream);
            let (yp, _c) = wq.packed.device_ptr(&d.stream);
            let (ys, _d) = wq.scales.device_ptr(&d.stream);
            let (zp, _z) = z.device_ptr_mut(&d.stream);
            unsafe { g::oxide_gemm(&v, zp, xp, yp, xs, ys, m, n, k, alpha)? };
        }
        ox_sample = Some(edge_rows(&z, m, n)?);
        row.insert("oxide_ms".into(), json!(ms));
        row.insert("oxide_tflops".into(), json!(tflops(ms)));
        row.insert("oxide_variant".into(), json!(v.label()));
    }
    let bf16_ms = row.get("bf16_ms").and_then(|v| v.as_f64());
    for key in ["cublaslt", "oxide"] {
        if let (Some(b), Some(x)) = (
            bf16_ms,
            row.get(&format!("{key}_ms")).and_then(|v| v.as_f64()),
        ) {
            row.insert(format!("{key}_speedup_vs_bf16"), json!(b / x));
        }
    }
    // Edge rows of each output: oxide vs cuBLASLt (same NVFP4 operands) and
    // NVFP4 vs bf16 (the quantization error itself, informational).
    if let (Some(o), Some(l)) = (&ox_sample, &lt_sample) {
        row.insert("oxide_vs_cublaslt_rel_l2".into(), json!(diff(o, l).rel_l2));
    }
    if let Some(l) = lt_sample.as_ref().or(ox_sample.as_ref()) {
        row.insert(
            "nvfp4_vs_bf16_rel_l2".into(),
            json!(diff(l, &bf16_sample).rel_l2),
        );
    }
    Ok(serde_json::Value::Object(row))
}
