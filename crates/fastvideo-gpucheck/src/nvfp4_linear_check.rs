//! The NVFP4 linear of the model path (`wan::nvfp4_linear`, the LTX-2 video
//! FFN under `FASTVIDEO_NVFP4`):
//!
//! 1. the bf16 operand quantizer (plain and GELU-fused) writes exactly the
//!    host TE `static_6` codes and, in cuBLASLt's tiled layout, the host
//!    scales (`swizzle_scales_host`), rows padded to 16 / 128 with zeros;
//! 2. `Nvfp4Linear` (device alpha, bias epilogue, an unaligned token count)
//!    reproduces the exact dequantized host math (f64) within bf16 output
//!    rounding, for the up (plain) and down (GELU input) projection;
//! 3. PSNR of the NVFP4 output against the bf16 linear at an FFN shape
//!    (informational: the quantization error itself);
//! 4. timings at the LTX-2.5 video FFN shapes the pipeline runs (4K and
//!    1080p, stage 1 whole and the stage-2 16 384-row chunks): cuBLAS bf16,
//!    the W8A8 FP8 linear, the NVFP4 GEMM alone and the whole NVFP4 linear
//!    (activation quantize + GEMM + bias). `FV_NVFP4_BENCH=0` skips.

use std::time::Instant;

use cudarc::driver::CudaSlice;
use fastvideo_cudarc::wan::nvfp4_gemm::swizzle_scales_host;
use fastvideo_cudarc::wan::nvfp4_linear::{self as nl, Nvfp4Linear};
use fastvideo_cudarc::wan::quant::{self, ptr, QuantKind, QuantLayout, QuantWeight, Section};
use fastvideo_cudarc::wan::{device, ops};
use fastvideo_models::nvfp4::{self, ScaleRule};
use half::bf16;
use serde_json::json;

use crate::metrics::{diff, psnr};
use crate::rand_weights::randn;
use crate::report::{Report, StageResult};

/// bf16 output rounding (2^-9 relative) plus accumulation order.
const BF16_OUT_LIMIT: f64 = 4e-3;

fn dev() -> anyhow::Result<std::sync::Arc<device::DeviceContext>> {
    device::global_device().ok_or_else(|| anyhow::anyhow!("no live CUDA device"))
}

fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn to_f32(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// The kernel's `bf16(gelu_tanh(x))`.
fn gelu16(x: f32) -> f32 {
    let k = 0.797_884_6_f32;
    let g = 0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh());
    bf16::from_f32(g).to_f32()
}

fn mismatches(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn time_ms(mut f: impl FnMut() -> anyhow::Result<()>) -> anyhow::Result<f64> {
    f()?;
    device::synchronize()?;
    let t = Instant::now();
    f()?;
    device::synchronize()?;
    let once = t.elapsed().as_secs_f64();
    let iters = ((0.3 / once.max(1e-6)) as usize).clamp(3, 50);
    let t = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    device::synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

pub fn run(report: &mut Report, seed: u64) -> StageResult<()> {
    let d = dev()?;
    if d.sm_major < 10 {
        report.note(
            "nvfp4_linear_skipped",
            json!({"reason": format!("NVFP4 tensor cores need sm_100+, device is sm_{}{}", d.sm_major, d.sm_minor)}),
        );
        return Ok(());
    }

    // ---- 1. operand quantizer vs host TE static_6 ------------------------------------
    let (rows, k) = (300usize, 1024usize);
    let mut xf = randn(seed ^ 0x6e76_6670, rows * k, 1.0);
    xf[17] = 40.0;
    xf[5000] = -1e-6;
    let x16 = to_bf16(&xf);
    let xr: Vec<f32> = to_f32(&x16);
    let xd = d.stream.memcpy_stod(&x16)?;
    for gelu in [false, true] {
        let src: Vec<f32> = if gelu {
            xr.iter().map(|&v| gelu16(v)).collect()
        } else {
            xr.clone()
        };
        let host = nvfp4::quantize(&src, rows, k, ScaleRule::Static6).map_err(anyhow::Error::msg)?;
        let name = format!("nvfp4_operand_{}_matches_te_reference", if gelu { "gelu" } else { "plain" });
        let q = match nl::quantize_bf16_operand(ptr(&xd), rows, k, gelu, ScaleRule::Static6) {
            Ok(q) => q,
            Err(e) => {
                report.check(name, false, json!({"error": format!("{e:#}")}), json!({}))?;
                continue;
            }
        };
        let packed = d.stream.memcpy_dtov(&q.packed)?;
        let scales = d.stream.memcpy_dtov(&q.scales)?;
        let amax = d.stream.memcpy_dtov(&q.amax)?[0];
        let want_s = swizzle_scales_host(&host.scales, rows, k / 16);
        let mp = mismatches(&packed[..rows * k / 2], &host.packed);
        let pad_nonzero = packed[rows * k / 2..].iter().filter(|&&b| b != 0).count();
        let ms = mismatches(&scales, &want_s);
        // gelu: host tanh vs CUDA tanhf can flip a bf16 rounding (rare).
        let allowed = if gelu { rows * k / 2 / 1000 } else { 0 };
        report.check(
            name,
            mp <= allowed
                && ms <= allowed / 8
                && pad_nonzero == 0
                && (gelu || amax.to_bits() == host.amax.to_bits()),
            json!({"packed_mismatch": mp, "scale_mismatch": ms, "pad_nonzero": pad_nonzero,
                   "amax": amax, "host_amax": host.amax, "rows": rows, "rows_pack": q.rows_pack,
                   "scale_bytes": scales.len()}),
            json!({"packed_mismatch": allowed, "scale_mismatch": allowed / 8, "pad_nonzero": 0}),
        )?;
    }

    // ---- 2. Nvfp4Linear vs exact dequantized host math -------------------------------
    for (gelu, n) in [(false, 512usize), (true, 256usize)] {
        let name = format!("nvfp4_linear_{}_matches_host", if gelu { "gelu_in" } else { "plain" });
        let r = (|| -> anyhow::Result<serde_json::Value> {
            let wf = randn(seed ^ 0xbeef ^ n as u64, n * k, 0.05);
            let w16 = to_bf16(&wf);
            let wr = to_f32(&w16);
            let bf = randn(seed ^ 0xb1a5, n, 0.3);
            let b16 = to_bf16(&bf);
            let br = to_f32(&b16);
            let wd = d.stream.memcpy_stod(&w16)?;
            let bd = std::sync::Arc::new(d.stream.memcpy_stod(&b16)?);
            let lin = Nvfp4Linear::from_bf16(ptr(&wd), n, k, Some(bd), ScaleRule::Static6)?;
            let (y, bias_in) = lin.forward_bf16(ptr(&xd), rows, gelu)?;
            let mut got = to_f32(&d.stream.memcpy_dtov(&y)?);
            if !bias_in {
                for (i, v) in got.iter_mut().enumerate() {
                    *v += br[i % n];
                }
            }
            let src: Vec<f32> = if gelu {
                xr.iter().map(|&v| gelu16(v)).collect()
            } else {
                xr.clone()
            };
            let aq = nvfp4::quantize(&src, rows, k, ScaleRule::Static6).map_err(anyhow::Error::msg)?;
            let wq = nvfp4::quantize(&wr, n, k, ScaleRule::Static6).map_err(anyhow::Error::msg)?;
            let (ad, wdq) = (aq.dequantize(), wq.dequantize());
            let mut want = vec![0.0f32; rows * n];
            let mut dense = vec![0.0f32; rows * n];
            for i in 0..rows {
                for j in 0..n {
                    let (mut acc, mut acc_d) = (0.0f64, 0.0f64);
                    for t in 0..k {
                        acc += f64::from(ad[i * k + t]) * f64::from(wdq[j * k + t]);
                        acc_d += f64::from(src[i * k + t]) * f64::from(wr[j * k + t]);
                    }
                    want[i * n + j] = (acc + f64::from(br[j])) as f32;
                    dense[i * n + j] = (acc_d + f64::from(br[j])) as f32;
                }
            }
            let dv = diff(&got, &want);
            let peak = dense.iter().fold(0.0f32, |a, v| a.max(v.abs())) as f64;
            Ok(json!({"ok": dv.within(BF16_OUT_LIMIT), "vs_exact": dv.to_json(),
                      "bias_in_epilogue": bias_in, "m": rows, "n": n, "k": k,
                      "vs_dense_rel_l2": diff(&got, &dense).rel_l2,
                      "vs_dense_psnr_db": psnr(&got, &dense, 2.0 * peak)}))
        })();
        match r {
            Ok(v) => {
                let ok = v["ok"].as_bool().unwrap_or(false);
                report.check(name, ok, v, json!({"rel_l2": BF16_OUT_LIMIT}))?;
            }
            Err(e) => report.check(name, false, json!({"error": format!("{e:#}")}), json!({}))?,
        }
    }

    // ---- 3. NVFP4 vs the bf16 linear at an FFN shape ---------------------------------
    {
        let r = (|| -> anyhow::Result<serde_json::Value> {
            let (m, k, n) = (2048usize, 4096usize, 16384usize);
            let x = ops::cast_f32_bf16_device(&fastvideo_cudarc::wan::nvfp4_gemm::fill_uniform_device(
                m * k,
                seed ^ 3,
                1.0,
            )?)?;
            let w = ops::cast_f32_bf16_device(&fastvideo_cudarc::wan::nvfp4_gemm::fill_uniform_device(
                n * k,
                seed ^ 4,
                0.05,
            )?)?;
            let mut yb = unsafe { d.stream.alloc::<bf16>(m * n)? };
            device::matmul_linear_wt_bf16(&x, &w, &mut yb, m, k, n)?;
            let lin = Nvfp4Linear::from_bf16(ptr(&w), n, k, None, ScaleRule::Static6)?;
            let (yq, _) = lin.forward_bf16(ptr(&x), m, false)?;
            let (a, b) = (to_f32(&d.stream.memcpy_dtov(&yq)?), to_f32(&d.stream.memcpy_dtov(&yb)?));
            let peak = b.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) as f64;
            Ok(json!({"m": m, "k": k, "n": n, "rel_l2": diff(&a, &b).rel_l2,
                      "cosine": diff(&a, &b).cosine, "psnr_db": psnr(&a, &b, 2.0 * peak)}))
        })();
        match r {
            Ok(v) => report.note("nvfp4_linear_vs_bf16_ffn_up", v),
            Err(e) => report.check(
                "nvfp4_linear_vs_bf16_ffn_up/completed",
                false,
                json!({"error": format!("{e:#}")}),
                json!({}),
            )?,
        }
    }

    // ---- 4. timings --------------------------------------------------------------------
    if std::env::var("FV_NVFP4_BENCH").is_ok_and(|v| v == "0") {
        report.note("nvfp4_linear_bench_skipped", json!({"reason": "FV_NVFP4_BENCH=0"}));
        return Ok(());
    }
    // LTX-2.5 video FFN (4096 -> 16384 -> 4096). Stage 1 runs whole (below
    // FeedForwardChunking's 65 536-token threshold); stage 2 in 16 384-row
    // chunks. 4k5s: stage 1 32 640 tokens, stage 2 130 560; 1080p20s: stage
    // 1 32 130 (not a multiple of 16), stage 2 128 520.
    let shapes: [(&str, usize, usize, usize, bool); 7] = [
        ("s1_4k_up_m32640", 32_640, 4096, 16_384, false),
        ("s1_4k_down_m32640", 32_640, 16_384, 4096, true),
        ("s1_1080p_up_m32130", 32_130, 4096, 16_384, false),
        ("s1_1080p_down_m32130", 32_130, 16_384, 4096, true),
        ("s2_chunk_up_m16384", 16_384, 4096, 16_384, false),
        ("s2_chunk_down_m16384", 16_384, 16_384, 4096, true),
        ("s2_4k_whole_up_m130560", 130_560, 4096, 16_384, false),
    ];
    let mut table = Vec::new();
    for (i, &(name, m, k, n, gelu)) in shapes.iter().enumerate() {
        match bench(seed.wrapping_add(i as u64 * 7919), m, k, n, gelu) {
            Ok(row) => {
                report.note(format!("nvfp4_linear_bench_{name}"), row.clone());
                table.push((name, row));
            }
            Err(e) => report.check(
                format!("nvfp4_linear_bench_{name}/completed"),
                false,
                json!({"error": format!("{e:#}")}),
                json!({}),
            )?,
        }
    }
    eprintln!("[INFO] NVFP4 linear (ms) on sm_{}{}:", d.sm_major, d.sm_minor);
    eprintln!(
        "[INFO] {:<24} {:>10} {:>12} {:>12} {:>12} {:>10} {:>9}",
        "shape", "bf16 GEMM", "bf16 linear", "FP8 linear", "NVFP4 GEMM", "NVFP4 lin", "speedup"
    );
    for (name, r) in &table {
        let f = |k: &str| r[k].as_f64().map_or("n/a".into(), |v| format!("{v:.3}"));
        eprintln!(
            "[INFO] {:<24} {:>10} {:>12} {:>12} {:>12} {:>10} {:>9}",
            name,
            f("bf16_gemm_ms"),
            f("bf16_linear_ms"),
            f("fp8_w8a8_linear_ms"),
            f("nvfp4_gemm_ms"),
            f("nvfp4_linear_ms"),
            f("nvfp4_linear_speedup_vs_bf16_linear"),
        );
    }
    Ok(())
}

/// One `[m, k] @ [n, k]ᵀ` linear four ways. `gelu`: the input is the up
/// projection's pre-activation (the down projection), so the bf16 linear
/// pays its GELU pass and the NVFP4 linear quantizes `gelu(x)`.
fn bench(seed: u64, m: usize, k: usize, n: usize, gelu: bool) -> anyhow::Result<serde_json::Value> {
    let d = dev()?;
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = |ms: f64| flop / (ms * 1e-3) / 1e12;
    let mut row = serde_json::Map::new();
    row.insert("m".into(), json!(m));
    row.insert("k".into(), json!(k));
    row.insert("n".into(), json!(n));
    row.insert("gelu_in".into(), json!(gelu));
    let x: CudaSlice<bf16> = ops::cast_f32_bf16_device(&fastvideo_cudarc::wan::nvfp4_gemm::fill_uniform_device(
        m * k, seed, 1.0,
    )?)?;
    let w: CudaSlice<bf16> = ops::cast_f32_bf16_device(&fastvideo_cudarc::wan::nvfp4_gemm::fill_uniform_device(
        n * k,
        seed ^ 1,
        0.05,
    )?)?;
    let b16 = std::sync::Arc::new(ops::cast_f32_bf16_device(
        &fastvideo_cudarc::wan::nvfp4_gemm::fill_uniform_device(n, seed ^ 2, 0.1)?,
    )?);
    // bf16: the GEMM alone, then the linear the bf16 FFN runs (GELU pass on
    // the input for the down projection, bias in the epilogue).
    {
        let mut out = unsafe { d.stream.alloc::<bf16>(m * n)? };
        let ms = time_ms(|| {
            device::matmul_linear_wt_bf16(&x, &w, &mut out, m, k, n)?;
            Ok(())
        })?;
        row.insert("bf16_gemm_ms".into(), json!(ms));
        row.insert("bf16_gemm_tflops".into(), json!(tflops(ms)));
        drop(out);
        let xt = fastvideo_cudarc::wan::tensor::CudaTensor::from_device_slice_bf16(
            d.stream.clone_dtod(&x)?,
            vec![m, k],
        )?;
        let ms = time_ms(|| {
            let src = if gelu { xt.gelu_tanh() } else { xt.clone() };
            let s = src.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16"))?;
            quant::linear_bf16_bias(s, &w, &b16, m, k, n)?
                .ok_or_else(|| anyhow::anyhow!("bf16 bias epilogue unavailable"))?;
            Ok(())
        })?;
        row.insert("bf16_linear_ms".into(), json!(ms));
    }
    // FP8 W8A8 (FASTVIDEO_FP8): tensorwise activation quantize + FP8 GEMM.
    {
        let r = (|| -> anyhow::Result<f64> {
            let layout = QuantLayout::new(
                QuantKind::W8A8,
                k,
                vec![Section {
                    rows: n,
                    quantized: true,
                }],
            )?;
            let q = QuantWeight::from_device_bf16(layout, &w)?;
            time_ms(|| {
                q.forward_device(Some(ptr(&x)), None, m)?;
                Ok(())
            })
        })();
        match r {
            Ok(ms) => {
                row.insert("fp8_w8a8_linear_ms".into(), json!(ms));
            }
            Err(e) => {
                row.insert("fp8_error".into(), json!(format!("{e:#}")));
            }
        }
    }
    // NVFP4.
    let lin = Nvfp4Linear::from_bf16(ptr(&w), n, k, Some(b16.clone()), ScaleRule::Static6)?;
    let ms = time_ms(|| {
        nl::quantize_bf16_operand(ptr(&x), m, k, gelu, ScaleRule::Static6)?;
        Ok(())
    })?;
    row.insert("nvfp4_quantize_ms".into(), json!(ms));
    let a = nl::quantize_bf16_operand(ptr(&x), m, k, gelu, ScaleRule::Static6)?;
    let ms = time_ms(|| {
        lin.gemm(&a)?;
        Ok(())
    })?;
    row.insert("nvfp4_gemm_ms".into(), json!(ms));
    row.insert("nvfp4_gemm_tflops".into(), json!(tflops(ms)));
    drop(a);
    let ms = time_ms(|| {
        lin.forward_bf16(ptr(&x), m, gelu)?;
        Ok(())
    })?;
    row.insert("nvfp4_linear_ms".into(), json!(ms));
    let get = |k: &str, row: &serde_json::Map<String, serde_json::Value>| row.get(k).and_then(|v| v.as_f64());
    if let (Some(b), Some(q)) = (get("bf16_linear_ms", &row), get("nvfp4_linear_ms", &row)) {
        row.insert("nvfp4_linear_speedup_vs_bf16_linear".into(), json!(b / q));
    }
    if let (Some(b), Some(q)) = (get("bf16_gemm_ms", &row), get("nvfp4_gemm_ms", &row)) {
        row.insert("nvfp4_gemm_speedup_vs_bf16_gemm".into(), json!(b / q));
    }
    if let (Some(f), Some(q)) = (get("fp8_w8a8_linear_ms", &row), get("nvfp4_linear_ms", &row)) {
        row.insert("nvfp4_linear_speedup_vs_fp8_linear".into(), json!(f / q));
    }
    Ok(serde_json::Value::Object(row))
}
