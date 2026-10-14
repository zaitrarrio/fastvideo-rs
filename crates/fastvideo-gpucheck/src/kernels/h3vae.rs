//! `h3vae_bench`: the H3 video VAE's fused-ViT ops at its real shapes
//! (docs/perf/h3-vae-decode-speed.md), one group so a decode-speed question
//! costs a minute of GPU instead of a full generation.
//!
//! Rows are `tiles x 1797` (1792 voxels of a `7 x 16 x 16` tile plus the 5
//! suffix tokens); `FV_H3VAE_TILES` (default `7,28`: the default tile batch
//! at 768p and the per-chunk cap). Width 2048, 32 heads of 64, SwiGLU 8192.
//!
//! * GEMMs: cuBLAS `GemmEx` (what the decoder runs), cuBLASLt's heuristic
//!   algorithms 0..7 (bf16, bit mismatches against GemmEx), and MXFP8
//!   (quantize + cuBLASLt algorithms 0..3, rel-L2 against GemmEx).
//! * SDPA: V1 / V2 / cuDNN / `auto` at `B*32` heads x 1797 x 64.
//! * Element-wise: each `h3v_*` kernel, its vector twin (must be bit-equal)
//!   and its MXFP8 producer (must equal `mxfp8_quantize` of the bf16 kernel's
//!   output, codes and scales), with achieved bandwidth; a device copy as the
//!   bandwidth roofline.

use super::*;
use fastvideo_cudarc::wan::attn::FlashKernel;
use fastvideo_cudarc::wan::quant::{self, MxAct, QuantKind, QuantLayout, QuantWeight, Section};
use half::bf16;

const DIM: usize = 2048;
const HEADS: usize = 32;
const HEAD_D: usize = 64;
const HIDDEN: usize = 8192;
const TOKENS: usize = 1797;
const ROTARY: usize = 60;

fn median3(f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<f64> {
    f()?;
    device::synchronize()?;
    let mut t = Vec::new();
    for _ in 0..3 {
        let s = std::time::Instant::now();
        f()?;
        device::synchronize()?;
        t.push(s.elapsed().as_secs_f64());
    }
    t.sort_by(|a, b| a.total_cmp(b));
    Ok(t[1])
}

fn bf16_dev(v: &[f32]) -> anyhow::Result<CudaSlice<bf16>> {
    let h: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
    Ok(dev()?.stream.memcpy_stod(&h)?)
}

fn bits(s: &CudaSlice<bf16>, n: usize) -> anyhow::Result<Vec<u16>> {
    let v = dev()?.stream.memcpy_dtov(&s.slice(0..n))?;
    Ok(v.iter().map(|b| b.to_bits()).collect())
}

fn bytes(s: &CudaSlice<u8>) -> anyhow::Result<Vec<u8>> {
    Ok(dev()?.stream.memcpy_dtov(s)?)
}

fn mism<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn rel_l2(a: &[u16], b: &[u16]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (
            f64::from(bf16::from_bits(x).to_f32()),
            f64::from(bf16::from_bits(y).to_f32()),
        );
        n += (x - y).powi(2);
        d += y * y;
    }
    (n / d.max(f64::MIN_POSITIVE)).sqrt()
}

/// MXFP8 of a bf16 `[rows, k]` buffer through the reference quantizer.
fn mx_of(x: &CudaSlice<bf16>, rows: usize, k: usize) -> anyhow::Result<MxAct> {
    let mut a = MxAct::alloc(rows, k)?;
    quant::mxfp8_quantize_raw(quant::ptr(x), true, &mut a)?;
    Ok(a)
}

fn mx_equal(a: &MxAct, b: &MxAct) -> anyhow::Result<(usize, usize)> {
    Ok((mism(&bytes(&a.q)?, &bytes(&b.q)?), mism(&bytes(&a.s)?, &bytes(&b.s)?)))
}

pub(super) fn bench(c: &mut Ctx<'_>) -> StageResult<()> {
    if c.only.is_none() {
        c.report
            .note("h3vae_bench_skipped", json!({"why": "opt-in: --groups h3vae_bench"}));
        return Ok(());
    }
    let d = dev()?;
    if d.sm_major < 8 {
        c.report
            .note("h3vae_bench_skipped", json!({"sm_major": d.sm_major, "needs": "sm80+"}));
        return Ok(());
    }
    let tiles: Vec<usize> = std::env::var("FV_H3VAE_TILES")
        .unwrap_or_else(|_| "7,28".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    // Bandwidth roofline: a 1 GiB device-to-device copy.
    {
        let n = 1usize << 29;
        let src = d.stream.alloc_zeros::<u16>(n)?;
        let mut dst = d.stream.alloc_zeros::<u16>(n)?;
        let s = median3(&mut || {
            d.stream.memcpy_dtod(&src, &mut dst)?;
            Ok(())
        })?;
        c.report.note(
            "h3vae_bench_copy_roofline",
            json!({"bytes_moved": 4 * n, "ms": s * 1e3, "gb_s": 4.0 * n as f64 / s / 1e9}),
        );
    }
    for &t in &tiles {
        let rows = t * TOKENS;
        gemms(c, t, rows)?;
        sdpa(c, t)?;
        elementwise(c, t, rows)?;
    }
    Ok(())
}

fn gemms(c: &mut Ctx<'_>, t: usize, rows: usize) -> StageResult<()> {
    let d = dev()?;
    for (name, k, n) in [
        ("qkv", DIM, 3 * DIM),
        ("to_out", DIM, DIM),
        ("ff_in", DIM, 2 * HIDDEN),
        ("ff_out", HIDDEN, DIM),
    ] {
        let x = bf16_dev(&c.rand(rows * k, 1.0))?;
        let w = bf16_dev(&c.rand(n * k, 0.02))?;
        let flops = 2.0 * rows as f64 * k as f64 * n as f64;
        let mut out = unsafe { d.stream.alloc::<bf16>(rows * n) }?;
        let mut row = serde_json::Map::new();
        let s = median3(&mut || {
            device::matmul_linear_wt_bf16(&x, &w, &mut out, rows, k, n)?;
            Ok(())
        })?;
        row.insert("cublas_ms".into(), json!(s * 1e3));
        row.insert("cublas_tflops".into(), json!(flops / s / 1e12));
        let cmp_n = (rows * n).min(1 << 24);
        let want = bits(&out, cmp_n)?;
        let mut best = (s, "cublas".to_string());
        for nth in 0..8 {
            let mut o2 = unsafe { d.stream.alloc::<bf16>(rows * n) }?;
            let r = median3(&mut || {
                quant::lt_gemm_bf16(&x, &w, &mut o2, rows, k, n, nth)?;
                Ok(())
            });
            match r {
                Ok(s2) => {
                    let got = bits(&o2, cmp_n)?;
                    row.insert(
                        format!("lt{nth}"),
                        json!({"ms": s2 * 1e3, "tflops": flops / s2 / 1e12,
                               "bit_mismatches": mism(&got, &want), "rel_l2": rel_l2(&got, &want)}),
                    );
                    if s2 < best.0 {
                        best = (s2, format!("lt{nth}"));
                    }
                }
                Err(e) => {
                    row.insert(format!("lt{nth}"), json!({"error": format!("{e:#}")}));
                    break;
                }
            }
        }
        // MXFP8: weight quantized once; activation quantize timed apart.
        let layout = QuantLayout::new(QuantKind::Mxfp8, k, vec![Section { rows: n, quantized: true }])?;
        let qw = QuantWeight::from_device_bf16(layout, &w)?;
        let mut act = MxAct::alloc(rows, k)?;
        let sq = median3(&mut || {
            quant::mxfp8_quantize_raw(quant::ptr(&x), true, &mut act)?;
            Ok(())
        })?;
        row.insert("mx_quantize_ms".into(), json!(sq * 1e3));
        for nth in 0..4 {
            let r = median3(&mut || {
                qw.forward_mx_padded(&act, nth)?;
                Ok(())
            });
            match r {
                Ok(s3) => {
                    let o3 = qw.forward_mx_padded(&act, nth)?;
                    let got = bits(&o3, cmp_n)?;
                    row.insert(
                        format!("mx{nth}"),
                        json!({"ms": s3 * 1e3, "tflops": flops / s3 / 1e12,
                               "rel_l2_vs_bf16": rel_l2(&got, &want)}),
                    );
                }
                Err(e) => {
                    row.insert(format!("mx{nth}"), json!({"error": format!("{e:#}")}));
                    break;
                }
            }
        }
        row.insert("fastest_bf16".into(), json!(best.1));
        c.report.check(
            format!("h3vae_bench_gemm_{name}_t{t}"),
            true,
            serde_json::Value::Object(row),
            json!({"rows": rows, "k": k, "n": n}),
        )?;
    }
    Ok(())
}

fn sdpa(c: &mut Ctx<'_>, t: usize) -> StageResult<()> {
    let shape = vec![t, HEADS, TOKENS, HEAD_D];
    let n = t * HEADS * TOKENS * HEAD_D;
    let mk = |c: &mut Ctx<'_>, std: f32| -> anyhow::Result<CudaTensor> {
        Ok(CudaTensor::from_device_slice_bf16(
            bf16_dev(&c.rand(n, std))?,
            shape.clone(),
        )?)
    };
    let (q, k, v) = (mk(c, 1.0)?, mk(c, 1.0)?, mk(c, 1.0)?);
    let flops = 4.0 * (t * HEADS) as f64 * (TOKENS * TOKENS) as f64 * HEAD_D as f64;
    let mut row = serde_json::Map::new();
    let mut outs = std::collections::HashMap::new();
    for (kname, kernel) in [
        ("v1", FlashKernel::V1),
        ("v2", FlashKernel::V2),
        ("cudnn", FlashKernel::Cudnn),
    ] {
        let s = median3(&mut || {
            attn::device_mma_sdpa_with(&q, &k, &v, None, true, kernel)?
                .ok_or_else(|| anyhow::anyhow!("flash declined"))?;
            Ok(())
        })?;
        let o = attn::device_mma_sdpa_with(&q, &k, &v, None, true, kernel)?
            .ok_or_else(|| anyhow::anyhow!("flash declined"))?;
        let b = bits(o.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("not bf16"))?, n)?;
        row.insert(kname.into(), json!({"ms": s * 1e3, "tflops": flops / s / 1e12}));
        outs.insert(kname, b);
    }
    let s = median3(&mut || {
        attn::device_mma_sdpa(&q, &k, &v, None, true)?
            .ok_or_else(|| anyhow::anyhow!("flash declined"))?;
        Ok(())
    })?;
    let o = attn::device_mma_sdpa(&q, &k, &v, None, true)?
        .ok_or_else(|| anyhow::anyhow!("flash declined"))?;
    let b = bits(o.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("not bf16"))?, n)?;
    let same_as: Vec<&str> = outs
        .iter()
        .filter(|(_, v)| **v == b)
        .map(|(k, _)| *k)
        .collect();
    row.insert(
        "auto".into(),
        json!({"ms": s * 1e3, "tflops": flops / s / 1e12, "bit_equal_to": same_as}),
    );
    row.insert("v1_vs_v2_mismatches".into(), json!(mism(&outs["v1"], &outs["v2"])));
    row.insert(
        "cudnn_vs_v2_rel_l2".into(),
        json!(rel_l2(&outs["cudnn"], &outs["v2"])),
    );
    c.report.check(
        format!("h3vae_bench_sdpa_t{t}"),
        true,
        serde_json::Value::Object(row),
        json!({"bh": t * HEADS, "tokens": TOKENS, "d": HEAD_D}),
    )
}

fn elementwise(c: &mut Ctx<'_>, t: usize, rows: usize) -> StageResult<()> {
    let d = dev()?;
    let gbs = |bytes: f64, s: f64| bytes / s / 1e9;
    let mut row = serde_json::Map::new();
    let mut ok = true;

    // Residual step + RMSNorm (bf16 out) and its MXFP8 producer.
    let x = up(&c.rand(rows * DIM, 1.0))?;
    let upd = bf16_dev(&c.rand(rows * DIM, 1.0))?;
    let bias = up(&c.rand(DIM, 0.1))?;
    let scale = up(&c.rand(DIM, 0.1))?;
    let w = up(&c.rand(DIM, 1.0))?;
    let s = median3(&mut || {
        ops::h3v_residual_norm_bf16_device(&x, DIM, Some((&upd, &bias, &scale)), Some(&w), 1e-6)?;
        Ok(())
    })?;
    let rn_bytes = (rows * DIM) as f64 * (4.0 + 2.0 + 4.0 + 2.0);
    row.insert(
        "residual_norm".into(),
        json!({"ms": s * 1e3, "gb_s": gbs(rn_bytes, s)}),
    );
    let (_, n16) =
        ops::h3v_residual_norm_bf16_device(&x, DIM, Some((&upd, &bias, &scale)), Some(&w), 1e-6)?;
    let want = mx_of(n16.as_ref().ok_or_else(|| anyhow::anyhow!("no norm"))?, rows, DIM)?;
    let mut got = MxAct::alloc(rows, DIM)?;
    let s = median3(&mut || {
        ops::h3v_residual_norm_mx_device(&x, DIM, Some((&upd, &bias, &scale)), &w, 1e-6, &mut got)?;
        Ok(())
    })?;
    let (mq, ms) = mx_equal(&got, &want)?;
    ok &= mq == 0 && ms == 0;
    row.insert(
        "residual_norm_mx".into(),
        json!({"ms": s * 1e3, "gb_s": gbs((rows * DIM) as f64 * (4.0 + 2.0 + 4.0 + 1.0), s),
               "code_mismatches": mq, "scale_mismatches": ms}),
    );

    // QKV heads.
    let qkv = bf16_dev(&c.rand(rows * 3 * DIM, 1.0))?;
    let qb = up(&c.rand(3 * DIM, 0.1))?;
    let cs = up(&c.rand(TOKENS * ROTARY, 1.0))?;
    let sn = up(&c.rand(TOKENS * ROTARY, 1.0))?;
    let qkv_bytes = (rows * 3 * DIM) as f64 * 4.0;
    let mut outs = Vec::new();
    for vec in [false, true] {
        let s = median3(&mut || {
            ops::h3v_qkv_heads_bf16_device_with(
                vec, &qkv, &qb, &cs, &sn, t, TOKENS, HEADS, HEAD_D, ROTARY, 1e-6,
            )?;
            Ok(())
        })?;
        let o = ops::h3v_qkv_heads_bf16_device_with(
            vec, &qkv, &qb, &cs, &sn, t, TOKENS, HEADS, HEAD_D, ROTARY, 1e-6,
        )?;
        let b: Vec<Vec<u16>> = o.iter().map(|s| bits(s, rows * DIM)).collect::<anyhow::Result<_>>()?;
        row.insert(
            format!("qkv_heads{}", if vec { "_vec" } else { "" }),
            json!({"ms": s * 1e3, "gb_s": gbs(qkv_bytes, s)}),
        );
        outs.push(b);
    }
    let qm: usize = (0..3).map(|i| mism(&outs[0][i], &outs[1][i])).sum();
    ok &= qm == 0;
    row.insert("qkv_heads_vec_bit_mismatches".into(), json!(qm));

    // Merge heads.
    let bhsd = bf16_dev(&c.rand(rows * DIM, 1.0))?;
    let mut mouts = Vec::new();
    for vec in [false, true] {
        let s = median3(&mut || {
            ops::h3v_merge_heads_bf16_device_with(vec, &bhsd, t, HEADS, TOKENS, HEAD_D)?;
            Ok(())
        })?;
        let o = ops::h3v_merge_heads_bf16_device_with(vec, &bhsd, t, HEADS, TOKENS, HEAD_D)?;
        row.insert(
            format!("merge_heads{}", if vec { "_vec" } else { "" }),
            json!({"ms": s * 1e3, "gb_s": gbs((rows * DIM) as f64 * 4.0, s)}),
        );
        mouts.push(o);
    }
    let mm = mism(&bits(&mouts[0], rows * DIM)?, &bits(&mouts[1], rows * DIM)?);
    ok &= mm == 0;
    row.insert("merge_heads_vec_bit_mismatches".into(), json!(mm));
    let want = mx_of(&mouts[0], rows, DIM)?;
    let mut got = MxAct::alloc(rows, DIM)?;
    let s = median3(&mut || {
        ops::h3v_merge_heads_mx_device(&bhsd, t, HEADS, TOKENS, HEAD_D, &mut got)?;
        Ok(())
    })?;
    let (mq, ms) = mx_equal(&got, &want)?;
    ok &= mq == 0 && ms == 0;
    row.insert(
        "merge_heads_mx".into(),
        json!({"ms": s * 1e3, "code_mismatches": mq, "scale_mismatches": ms}),
    );

    // SwiGLU.
    let h = bf16_dev(&c.rand(rows * 2 * HIDDEN, 1.0))?;
    let hb = up(&c.rand(2 * HIDDEN, 0.1))?;
    let mut souts = Vec::new();
    for vec in [false, true] {
        let s = median3(&mut || {
            ops::h3v_swiglu_bf16_device_with(vec, &h, &hb, HIDDEN, rows)?;
            Ok(())
        })?;
        let o = ops::h3v_swiglu_bf16_device_with(vec, &h, &hb, HIDDEN, rows)?;
        row.insert(
            format!("swiglu{}", if vec { "_vec" } else { "" }),
            json!({"ms": s * 1e3, "gb_s": gbs((rows * HIDDEN) as f64 * 6.0, s)}),
        );
        souts.push(o);
    }
    let sm = mism(&bits(&souts[0], rows * HIDDEN)?, &bits(&souts[1], rows * HIDDEN)?);
    ok &= sm == 0;
    row.insert("swiglu_vec_bit_mismatches".into(), json!(sm));
    let want = mx_of(&souts[0], rows, HIDDEN)?;
    let mut got = MxAct::alloc(rows, HIDDEN)?;
    let s = median3(&mut || {
        ops::h3v_swiglu_mx_device(&h, &hb, HIDDEN, &mut got)?;
        Ok(())
    })?;
    let (mq, ms) = mx_equal(&got, &want)?;
    ok &= mq == 0 && ms == 0;
    row.insert(
        "swiglu_mx".into(),
        json!({"ms": s * 1e3, "gb_s": gbs((rows * HIDDEN) as f64 * 5.0, s),
               "code_mismatches": mq, "scale_mismatches": ms}),
    );
    let _ = d;
    c.report.check(
        format!("h3vae_bench_elementwise_t{t}"),
        ok,
        serde_json::Value::Object(row),
        json!({"vector twins": "bit-equal", "mx producers": "equal mxfp8_quantize of the bf16 output"}),
    )
}
