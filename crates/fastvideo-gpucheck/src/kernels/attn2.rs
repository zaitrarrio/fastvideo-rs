//! Phase 3b attention kernels: parity of each new kernel against the kernel
//! it replaces, and timings at the real model shapes.
//!
//! `attn2_parity` holds every new kernel to the old one. The new kernels do
//! the same per-row arithmetic in the same order (only the schedule
//! changes), so the checks are bit-exact where that holds (dense v2, Sol ws
//! at one split, VSA ring) and tight elsewhere (Sol KV splits: the split
//! partials are merged in f32, which is a different summation order).
//!
//! `attn_bench` times old vs new at the shapes the generations run:
//! H3 768p (56 heads, 37 710 tokens), LTX-2.5 stage 2 at 4K 5 s (130 560),
//! 1080p 20 s (124 440) and 768x512 (6 144), 32 heads, d=128, bf16 in and
//! out, as the bf16-activation pipelines call them. Each number is the
//! median of three synchronized calls after a warm-up.

use super::*;
use fastvideo_cudarc::wan::attn::FlashKernel;
use fastvideo_cudarc::wan::ops::{SolFwdPick, SolKernel};
use fastvideo_models::sol_attn::{num_blocks, SolParams};
use half::bf16;

const D: usize = 128;

fn bf16_down(x: &CudaSlice<bf16>, n: usize) -> anyhow::Result<Vec<u16>> {
    let dev = dev()?;
    let v = dev.stream.memcpy_dtov(&x.slice(0..n))?;
    Ok(v.iter().map(|b| b.to_bits()).collect())
}

fn f32_bits_down(x: &CudaSlice<f32>) -> anyhow::Result<Vec<u32>> {
    Ok(down(x)?.iter().map(|f| f.to_bits()).collect())
}

/// Count of differing elements between two bit patterns (length mismatch = all).
fn bit_mismatches<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// One head of "structured" Sol data (per-block bases + noise, as the `sol`
/// group builds it) so routing is non-trivial.
fn structured_head(c: &mut Ctx<'_>, tokens: usize) -> Vec<f32> {
    let n = num_blocks(tokens);
    let base = c.rand(n * D, 1.5);
    let common = c.rand(n * D, 1.5);
    let noise = c.rand(tokens * D, 0.3);
    (0..tokens * D)
        .map(|i| {
            let (t, d) = (i / D, i % D);
            let b = t / 64;
            let src = if b % 4 < 2 { &common } else { &base };
            src[b * D + d] + noise[i]
        })
        .collect()
}

/// `head` (one head, `[T, 128]` f32) repeated over `bh` heads, as bf16 (RNE).
fn tile_heads_bf16(head: &[f32], bh: usize) -> anyhow::Result<CudaSlice<bf16>> {
    let one: Vec<bf16> = head.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut all = Vec::with_capacity(one.len() * bh);
    for _ in 0..bh {
        all.extend_from_slice(&one);
    }
    Ok(dev()?.stream.memcpy_stod(&all)?)
}

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

pub(super) fn parity(c: &mut Ctx<'_>) -> StageResult<()> {
    let dev = dev()?;
    if dev.sm_major < 8 {
        c.report
            .note("attn2_skipped", json!({"sm_major": dev.sm_major, "needs": "sm80+"}));
        return Ok(());
    }

    // ---- dense: flash_mma_fwd2 vs flash_mma_fwd, bit for bit -------------
    for (b, h, sq, sk, d) in [
        (1usize, 1usize, 1usize, 1usize, 128usize),
        (1, 2, 64, 64, 128),
        (1, 2, 70, 70, 64),
        (2, 3, 257, 257, 128),
        (1, 4, 300, 512, 128),
        (2, 2, 1000, 333, 64),
        (1, 3, 130, 1, 128),
        (1, 2, 63, 4097, 128),
        (1, 2, 129, 200, 128),
        (1, 2, 2048, 2048, 128),
    ] {
        let bh = b * h;
        let q = c.rand(bh * sq * d, 1.5);
        let kk = c.rand(bh * sk * d, 1.5);
        let v = c.rand(bh * sk * d, 1.0);
        let scale = 1.0 / (d as f32).sqrt();
        let want = ref_sdpa(&q, &kk, &v, bh, sq, sk, d, scale);
        let (qt, kt, vt) = (
            t(q, &[b, h, sq, d])?,
            t(kk, &[b, h, sk, d])?,
            t(v, &[b, h, sk, d])?,
        );
        let tag = format!("{b}x{h}x{sq}x{sk}x{d}");
        for out16 in [false, true] {
            let v1 = attn::device_mma_sdpa_with(&qt, &kt, &vt, Some(scale), out16, FlashKernel::V1)?
                .ok_or_else(|| anyhow::anyhow!("flash v1 declined {tag}"))?;
            let v2 = attn::device_mma_sdpa_with(&qt, &kt, &vt, Some(scale), out16, FlashKernel::V2)?
                .ok_or_else(|| anyhow::anyhow!("flash v2 declined {tag}"))?;
            let (a, bb) = (host_of(&v1)?, host_of(&v2)?);
            let bad = bit_mismatches(
                &a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &bb.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            );
            let o = if out16 { "_bf16out" } else { "" };
            c.report.check(
                format!("flash2_{tag}{o}_bitexact_vs_v1"),
                bad == 0,
                json!({"mismatched": bad, "of": a.len()}),
                json!({"mismatched": 0}),
            )?;
            if !out16 {
                c.cmp(&format!("flash2_{tag}_vs_f32"), &bb, &want, 1e-2)?;
            }
        }
        // cuDNN fused attention (bf16 out), called directly so a missing
        // engine is reported rather than silently replaced by flash v2.
        let (q16, k16, v16) = (qt.quantize_bf16()?, kt.quantize_bf16()?, vt.quantize_bf16()?);
        let (qs, ks, vs) = (
            q16.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16 q"))?,
            k16.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16 k"))?,
            v16.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16 v"))?,
        );
        let mut o16 = unsafe { dev.stream.alloc::<bf16>(bh * sq * d) }?;
        let ran = fastvideo_cudarc::wan::cudnn_sdpa::sdpa_bf16(qs, ks, vs, &mut o16, bh, sq, sk, d, scale)?;
        if ran {
            let got: Vec<f32> = dev.stream.memcpy_dtov(&o16)?.iter().map(|x| x.to_f32()).collect();
            c.cmp(&format!("cudnn_sdpa_{tag}_vs_f32"), &got, &want, 1e-2)?;
        } else {
            c.report.note(format!("cudnn_sdpa_{tag}_no_engine"), json!({"bh": bh, "sq": sq, "sk": sk, "d": d}));
        }
    }

    // ---- Sol: sol_mma_fwd2 (ws) vs sol_mma_fwd ---------------------------
    use fastvideo_models::sol_attn::sink_blocks;
    let sc = 1.0 / (D as f32).sqrt();
    if dev.sm_major >= 9 {
        let cases: Vec<(&str, usize, usize, SolParams)> = vec![
            ("t4096_tau1", 2, 4096, SolParams::diag(1.0, sc)),
            ("t4033_tail1", 2, 4033, SolParams::diag(1.0, sc)),
            ("t8256_g3", 2, 8256, SolParams::diag(1.25, sc)),
            ("t16384_g4", 2, 16384, SolParams::diag(1.5, sc)),
            (
                "sink_1000_300",
                2,
                4096,
                SolParams {
                    sink_start: Some(1000),
                    sink_tokens: 300,
                    ..SolParams::diag(1.0, sc)
                },
            ),
            ("tau_neg_1e4", 1, 2048, SolParams::diag(-1.0e4, sc)),
            ("tau_1e4_local", 2, 4096, SolParams::diag(1.0e4, sc)),
            ("tiny_1", 1, 1, SolParams::diag(1.0, sc)),
            ("tiny_65", 1, 65, SolParams::diag(1.0, sc)),
            ("batch6_t1000", 6, 1000, SolParams::diag(1.25, sc)),
        ];
        for (tag, bh, tokens, p) in cases {
            let mut q = Vec::new();
            let mut k = Vec::new();
            for _ in 0..bh {
                q.extend(structured_head(c, tokens));
                k.extend(structured_head(c, tokens));
            }
            let v = c.rand(bh * tokens * D, 1.0);
            let (qd, kd, vd) = (up(&q)?, up(&k)?, up(&v)?);
            let prep =
                ops::sol_prep_device(&qd, &kd, &vd, bh, tokens, D, p.tau, p.scale, p.thresh)?;
            let sinks = sink_blocks(tokens, p.sink_start, p.sink_tokens);
            let pick = |kernel, splits| SolFwdPick {
                kernel: Some(kernel),
                splits: Some(splits),
            };
            let v1 = ops::sol_fwd_device_pick(&prep, p.scale, sinks, true, true, pick(SolKernel::V1, 1))?;
            let (o1, l1) = (down(&v1.out)?, down(v1.lse.as_ref().expect("lse"))?);
            let r1 = dev.stream.memcpy_dtov(v1.route.as_ref().expect("route"))?;
            // x4: bit-exact; x4f: exp2 within 2 ulp (bf16 P may round apart).
            for (kname, kernel) in [("x4", SolKernel::X4), ("x4f", SolKernel::X4f)] {
                let w = ops::sol_fwd_device_pick(&prep, p.scale, sinks, true, true, pick(kernel, 1))?;
                let (o2, l2) = (down(&w.out)?, down(w.lse.as_ref().expect("lse"))?);
                let r2 = dev.stream.memcpy_dtov(w.route.as_ref().expect("route"))?;
                let route_bad = bit_mismatches(&r1, &r2);
                let ob = bit_mismatches(
                    &o1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    &o2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                );
                let d = diff(&o2, &o1);
                let lerr = l1
                    .iter()
                    .zip(&l2)
                    .map(|(a, b)| if a == b { 0.0 } else { (a - b).abs() })
                    .fold(0.0f32, f32::max);
                let pass = if kernel == SolKernel::X4 {
                    ob == 0 && lerr == 0.0 && route_bad == 0
                } else {
                    d.within(2e-3) && d.max_abs <= 4e-3 && lerr <= 1e-5 && route_bad == 0
                };
                c.report.check(
                    format!("sol_{kname}_{tag}_vs_v1"),
                    pass,
                    json!({"out_mismatched": ob, "diff": d.to_json(), "lse_max_abs": lerr,
                           "route_mismatched": route_bad}),
                    json!({"x4": "bit-exact", "x4f": {"rel_l2": 2e-3, "max_abs": 4e-3,
                           "lse_max_abs": 1e-5, "route_mismatched": 0}}),
                )?;
            }
            let groups = num_blocks(tokens).div_ceil(fastvideo_models::sol_attn::ROUTE_GROUP);
            for splits in [1usize, 2, 4] {
                if splits > groups {
                    continue;
                }
                let w = ops::sol_fwd_device_pick(&prep, p.scale, sinks, true, true, pick(SolKernel::Ws, splits))?;
                let (o2, l2) = (down(&w.out)?, down(w.lse.as_ref().expect("lse"))?);
                let r2 = dev.stream.memcpy_dtov(w.route.as_ref().expect("route"))?;
                let route_bad = bit_mismatches(&r1, &r2);
                if splits == 1 {
                    let ob = bit_mismatches(
                        &o1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        &o2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    );
                    let lb = bit_mismatches(
                        &l1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        &l2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    );
                    c.report.check(
                        format!("sol_ws_{tag}_bitexact_vs_v1"),
                        ob == 0 && lb == 0 && route_bad == 0,
                        json!({"out_mismatched": ob, "lse_mismatched": lb, "route_mismatched": route_bad}),
                        json!({"all": 0}),
                    )?;
                } else {
                    let d = diff(&o2, &o1);
                    let lerr = l1
                        .iter()
                        .zip(&l2)
                        .map(|(a, b)| {
                            if a == b {
                                0.0
                            } else {
                                (a - b).abs()
                            }
                        })
                        .fold(0.0f32, f32::max);
                    // Each split rounds P = 2^(s - m_split) to bf16 against its
                    // own running max, so P's bf16 rounding (2^-9) differs from
                    // the unsplit pass: a bf16-level difference, as in the
                    // reference's sm90 splits (which also store bf16 partials).
                    c.report.check(
                        format!("sol_ws_{tag}_splits{splits}_vs_v1"),
                        d.within(2e-3) && d.max_abs <= 4e-3 && lerr <= 1e-5 && route_bad == 0,
                        json!({"diff": d.to_json(), "lse_max_abs": lerr, "route_mismatched": route_bad}),
                        json!({"rel_l2": 2e-3, "max_abs": 4e-3, "lse_max_abs": 1e-5, "route_mismatched": 0}),
                    )?;
                }
            }
            // bf16 activations: no-copy prep + in-place operands. The bf16
            // output must be RNE(f32 output of the f32 path), for both kernels.
            let to16 = |x: &[f32]| -> Vec<bf16> { x.iter().map(|&f| bf16::from_f32(f)).collect() };
            let (q16, k16, v16) = (
                dev.stream.memcpy_stod(&to16(&q))?,
                dev.stream.memcpy_stod(&to16(&k))?,
                dev.stream.memcpy_stod(&to16(&v))?,
            );
            let want16: Vec<u16> = o1.iter().map(|&f| bf16::from_f32(f).to_bits()).collect();
            for (kname, kernel) in [("v1", SolKernel::V1), ("ws", SolKernel::Ws)] {
                let got = ops::sol_fused_device_bf16_pick(
                    &q16,
                    &k16,
                    &v16,
                    bh,
                    tokens,
                    D,
                    &p,
                    pick(kernel, 1),
                )?;
                let bad = bit_mismatches(&bf16_down(&got, bh * tokens * D)?, &want16);
                c.report.check(
                    format!("sol_bf16_nocopy_{kname}_{tag}_vs_f32_path"),
                    bad == 0,
                    json!({"mismatched": bad, "of": want16.len()}),
                    json!({"mismatched": 0}),
                )?;
            }
        }
    } else {
        c.report
            .note("sol_ws_skipped", json!({"sm_major": dev.sm_major, "needs": "sm90+"}));
    }

    // ---- VSA: three-slot TMA ring vs the two-stage TMA kernel ------------
    if dev.sm_major >= 9 {
        for (grid, heads, sparsity) in [
            ((4usize, 4usize, 4usize), 2usize, 0.0f64),
            ((5, 6, 9), 3, 0.8),
            ((9, 8, 13), 2, 0.8),
            ((9, 8, 13), 2, 0.97),
        ] {
            let plan = vsa::TilePlan::new(grid)?;
            let (seq, nb) = (plan.seq, plan.num_tiles());
            let topk = vsa::topk_for(sparsity, nb);
            let bh = heads;
            let n = bh * seq * D;
            let (q, k, v, g) = (c.rand(n, 1.0), c.rand(n, 1.0), c.rand(n, 1.0), c.rand(n, 0.5));
            let (qd, kd, vd, gd) = (up(&q)?, up(&k)?, up(&v)?, up(&g)?);
            let plan_dev = ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, vsa::TILE_ELEMS)?;
            let scale = 1.0 / (D as f32).sqrt();
            let run = |kernel: &str| -> anyhow::Result<Vec<u32>> {
                std::env::set_var("FASTVIDEO_VSA_KERNEL", kernel);
                let r = vsa::vsa_attention_device(
                    &qd,
                    &kd,
                    &vd,
                    Some(&gd),
                    &plan_dev,
                    topk,
                    bh,
                    seq,
                    D,
                    scale,
                    nb,
                );
                std::env::remove_var("FASTVIDEO_VSA_KERNEL");
                f32_bits_down(&r?)
            };
            let a = run("tma")?;
            let b = run("tma2")?;
            let bad = bit_mismatches(&a, &b);
            c.report.check(
                format!("vsa_tma2_{}x{}x{}_h{heads}_k{topk}_bitexact_vs_tma", grid.0, grid.1, grid.2),
                bad == 0,
                json!({"mismatched": bad, "of": a.len()}),
                json!({"mismatched": 0}),
            )?;
        }
    } else {
        c.report
            .note("vsa_tma2_skipped", json!({"sm_major": dev.sm_major, "needs": "sm90+"}));
    }
    Ok(())
}

/// Real-shape workloads: (name, heads, tokens, Sol taus).
const SHAPES: &[(&str, usize, usize, &[f32])] = &[
    ("h3_768p", 56, 37_710, &[1.0]),
    ("ltx_512p", 32, 6_144, &[1.0, 1.25, 1.5]),
    ("ltx_1080p20s", 32, 124_440, &[1.0, 1.25, 1.5]),
    ("ltx_4k5s", 32, 130_560, &[1.0, 1.25, 1.5]),
];

pub(super) fn bench(c: &mut Ctx<'_>) -> StageResult<()> {
    let dev = dev()?;
    if dev.sm_major < 9 {
        c.report
            .note("attn_bench_skipped", json!({"sm_major": dev.sm_major, "needs": "sm90+"}));
        return Ok(());
    }
    let only = std::env::var("FV_ATTN_BENCH_SHAPES").unwrap_or_default();
    let sc = 1.0 / (D as f32).sqrt();
    for &(name, bh, tokens, taus) in SHAPES {
        if !only.is_empty() && !only.split(',').any(|s| s.trim() == name) {
            continue;
        }
        let q = tile_heads_bf16(&structured_head(c, tokens), bh)?;
        let k = tile_heads_bf16(&structured_head(c, tokens), bh)?;
        let v = tile_heads_bf16(&c.rand(tokens * D, 1.0), bh)?;
        let n = bh * tokens * D;
        // Two heads (or all, when small) for the real-shape bit checks.
        let cmp_n = n.min(2 * tokens * D);

        // ---- dense ----
        let shape = vec![1, bh, tokens, D];
        let (qt, kt, vt) = (
            CudaTensor::from_device_slice_bf16(q.clone(), shape.clone())?,
            CudaTensor::from_device_slice_bf16(k.clone(), shape.clone())?,
            CudaTensor::from_device_slice_bf16(v.clone(), shape.clone())?,
        );
        let dense = |kernel: FlashKernel| -> anyhow::Result<f64> {
            median3(&mut || {
                attn::device_mma_sdpa_with(&qt, &kt, &vt, None, true, kernel)?
                    .ok_or_else(|| anyhow::anyhow!("flash declined"))?;
                Ok(())
            })
        };
        let d1 = dense(FlashKernel::V1)?;
        let d2 = dense(FlashKernel::V2)?;
        let dc = dense(FlashKernel::Cudnn)?;
        // attn_dc.cu (tcgen05 / wgmma) where this device has it, else V2 again.
        let has_dc = fastvideo_cudarc::wan::attn_dc::dense().is_some();
        let ddc = dense(FlashKernel::Dc)?;
        let bits = |kernel| -> anyhow::Result<Vec<u16>> {
            let o = attn::device_mma_sdpa_with(&qt, &kt, &vt, None, true, kernel)?
                .ok_or_else(|| anyhow::anyhow!("flash declined"))?;
            let s = o
                .device_slice_bf16()
                .ok_or_else(|| anyhow::anyhow!("flash: expected bf16 output"))?;
            bf16_down(s, cmp_n)
        };
        let dense_bad = bit_mismatches(&bits(FlashKernel::V1)?, &bits(FlashKernel::V2)?);
        let flops = 4.0 * (bh * tokens) as f64 * tokens as f64 * D as f64;
        c.report.check(
            format!("attn_bench_dense_{name}"),
            dense_bad == 0,
            json!({
                "bh": bh, "tokens": tokens,
                "v1_ms": d1 * 1e3, "v2_ms": d2 * 1e3, "speedup": d1 / d2,
                "v1_tflops": flops / d1 / 1e12, "v2_tflops": flops / d2 / 1e12,
                "cudnn_or_v2_fallback_ms": dc * 1e3, "cudnn_tflops": flops / dc / 1e12,
                "dc_kernel": has_dc, "dc_ms": ddc * 1e3, "dc_tflops": flops / ddc / 1e12,
                "bit_mismatches_first_heads": dense_bad,
            }),
            json!({"bit_mismatches_first_heads": 0}),
        )?;
        drop((qt, kt, vt));

        // ---- Sol ----
        for &tau in taus {
            let p = SolParams::diag(tau, sc);
            let sol = |pick: SolFwdPick| -> anyhow::Result<f64> {
                median3(&mut || {
                    ops::sol_fused_device_bf16_pick(&q, &k, &v, bh, tokens, D, &p, pick)?;
                    Ok(())
                })
            };
            let pk = |kernel, splits| SolFwdPick {
                kernel: Some(kernel),
                splits: Some(splits),
            };
            let s1 = sol(pk(SolKernel::V1, 1))?;
            let sx4 = sol(pk(SolKernel::X4, 1))?;
            let sx4f = sol(pk(SolKernel::X4f, 1))?;
            let s2 = sol(pk(SolKernel::Ws, 1))?;
            let s2x2 = if num_blocks(tokens) > 64 {
                Some(sol(pk(SolKernel::Ws, 2))?)
            } else {
                None
            };
            let prep_s = median3(&mut || {
                ops::sol_prep_device_bf16(&q, &k, &v, bh, tokens, D, p.tau, p.scale, p.thresh)?;
                Ok(())
            })?;
            let bits = |kernel| -> anyhow::Result<Vec<u16>> {
                let o = ops::sol_fused_device_bf16_pick(&q, &k, &v, bh, tokens, D, &p, pk(kernel, 1))?;
                bf16_down(&o, cmp_n)
            };
            let sol_bad = bit_mismatches(&bits(SolKernel::V1)?, &bits(SolKernel::Ws)?);
            // Exact fraction of one head (every head is the same data).
            let exact_fraction = {
                let one = |x: &CudaSlice<bf16>| -> anyhow::Result<CudaSlice<f32>> {
                    let h: Vec<f32> = dev
                        .stream
                        .memcpy_dtov(&x.slice(0..tokens * D))?
                        .iter()
                        .map(|b| b.to_f32())
                        .collect();
                    up(&h)
                };
                let (q1, k1, v1) = (one(&q)?, one(&k)?, one(&v)?);
                let prep = ops::sol_prep_device(&q1, &k1, &v1, 1, tokens, D, p.tau, p.scale, p.thresh)?;
                let fwd = ops::sol_fwd_device(&prep, p.scale, (0, 0), false, true)?;
                let route = dev.stream.memcpy_dtov(fwd.route.as_ref().expect("route"))?;
                let nt = num_blocks(tokens);
                route.iter().map(|w| w.count_ones() as f64).sum::<f64>() / (nt * nt) as f64
            };
            c.report.check(
                format!("attn_bench_sol_{name}_tau{tau}"),
                sol_bad == 0,
                json!({
                    "bh": bh, "tokens": tokens, "tau": tau,
                    "exact_fraction": exact_fraction,
                    "v1_ms": s1 * 1e3, "x4_ms": sx4 * 1e3, "x4f_ms": sx4f * 1e3, "ws_ms": s2 * 1e3, "ws_splits2_ms": s2x2.map(|s| s * 1e3),
                    "prep_ms": prep_s * 1e3, "speedup": s1 / s2,
                    "dense_v1_ms": d1 * 1e3,
                    "bit_mismatches_first_heads": sol_bad,
                }),
                json!({"bit_mismatches_first_heads": 0}),
            )?;
        }
        drop((q, k, v));

        // ---- VSA fine stage (H3 only) ----
        if name == "h3_768p" {
            vsa_bench(c, bh)?;
        }
    }
    Ok(())
}

/// VSA fine stage at the H3 768p video grid (37 x 28 x 36 tokens, 4x4x4
/// tiles), sparsity 0.9: two-stage TMA vs the three-slot ring, through
/// `vsa_mma_attn_range_device` as H3 calls it (tile + fine attention).
fn vsa_bench(c: &mut Ctx<'_>, bh: usize) -> StageResult<()> {
    let dev = dev()?;
    let plan = vsa::TilePlan::new((37, 28, 36))?;
    let (seq, nb) = (plan.seq, plan.num_tiles());
    let topk = vsa::topk_for(0.9, nb);
    let head = |c: &mut Ctx<'_>| -> anyhow::Result<CudaSlice<f32>> {
        let one = c.rand(seq * D, 1.0);
        let mut all = Vec::with_capacity(one.len() * bh);
        for _ in 0..bh {
            all.extend_from_slice(&one);
        }
        up(&all)
    };
    let (qd, kd, vd) = (head(c)?, head(c)?, head(c)?);
    let plan_dev = ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, vsa::TILE_ELEMS)?;
    // topk distinct tiles per query tile: its own, then a fixed stride.
    let mut sel = Vec::with_capacity(bh * nb * topk);
    for _ in 0..bh {
        for qt in 0..nb {
            for i in 0..topk {
                sel.push(((qt + i * 7) % nb) as u32);
            }
        }
    }
    let selected = dev.stream.memcpy_stod(&sel)?;
    let scale = 1.0 / (D as f32).sqrt();
    let run = |kernel: &str| -> anyhow::Result<f64> {
        std::env::set_var("FASTVIDEO_VSA_KERNEL", kernel);
        let r = median3(&mut || {
            ops::vsa_mma_attn_range_device(
                &qd, &kd, &vd, &selected, &plan_dev, bh, seq, D, topk, scale, 0, nb,
            )?;
            Ok(())
        });
        std::env::remove_var("FASTVIDEO_VSA_KERNEL");
        r
    };
    let a = run("tma")?;
    let b = run("tma2")?;
    let flops = 4.0 * (bh * nb * 64) as f64 * (topk * 64) as f64 * D as f64;
    c.report.note(
        "attn_bench_vsa_h3_768p",
        json!({
            "bh": bh, "tokens": seq, "tiles": nb, "topk": topk,
            "tma_ms": a * 1e3, "tma2_ms": b * 1e3, "speedup": a / b,
            "tma_tflops": flops / a / 1e12, "tma2_tflops": flops / b / 1e12,
        }),
    );
    Ok(())
}

/// `attn_dc`: the datacenter kernels (attn_dc.cu: tcgen05 + TMEM on 10.0,
/// wgmma on 9.0) against `flash_mma_fwd2` at the same bf16 inputs and
/// against the f32 reference. They advance the running max per 128 keys
/// (not 64) and skip the unit rescale, so P's bf16 rounding differs from
/// V2's: the bound is rel L2 4e-3 vs V2 (the attn2 Sol x4f / split bound is
/// 2e-3 on structured data; random data at this scale sits near 1e-3) and
/// the usual 1e-2 vs f32. The bf16 output must be RNE of the f32 output.
pub(super) fn dc_parity(c: &mut Ctx<'_>) -> StageResult<()> {
    let dev = dev()?;
    let Some(k) = fastvideo_cudarc::wan::attn_dc::dense() else {
        c.report.note(
            "attn_dc_skipped",
            json!({"sm": dev.sm_major * 10 + dev.sm_minor, "needs": "sm 9.0 or 10.0 with the attn_dc module"}),
        );
        return Ok(());
    };
    c.report.note("attn_dc_kernel", json!({"sm": k.sm, "origin": k.origin, "rows_per_cta": k.rows}));
    for (b, h, sq, sk, amp) in [
        (1usize, 1usize, 1usize, 1usize, 1.0f32),
        (1, 2, 64, 64, 1.5),
        (1, 2, 70, 70, 1.5),
        (2, 3, 257, 257, 1.5),
        (1, 4, 300, 512, 1.5),
        (2, 2, 1000, 333, 1.5),
        (1, 3, 130, 1, 1.0),
        (1, 2, 63, 4097, 1.5),
        (1, 2, 129, 200, 1.0),
        (1, 2, 2048, 2048, 1.5),
        (1, 1, 513, 1000, 4.0),
    ] {
        let (bh, d) = (b * h, 128usize);
        let q = c.rand(bh * sq * d, amp);
        let kk = c.rand(bh * sk * d, amp);
        let v = c.rand(bh * sk * d, 1.0);
        let scale = 1.0 / (d as f32).sqrt();
        let want = ref_sdpa(&q, &kk, &v, bh, sq, sk, d, scale);
        let (qt, kt, vt) = (t(q, &[b, h, sq, d])?, t(kk, &[b, h, sk, d])?, t(v, &[b, h, sk, d])?);
        let tag = format!("{b}x{h}x{sq}x{sk}");
        let run = |kernel, out16| -> anyhow::Result<Vec<f32>> {
            let o = attn::device_mma_sdpa_with(&qt, &kt, &vt, Some(scale), out16, kernel)?
                .ok_or_else(|| anyhow::anyhow!("sdpa declined {tag}"))?;
            host_of(&o)
        };
        let v2 = run(FlashKernel::V2, false)?;
        let dc = run(FlashKernel::Dc, false)?;
        let dc16 = run(FlashKernel::Dc, true)?;
        let dv = diff(&dc, &v2);
        c.report.check(
            format!("attn_dc_{tag}_vs_v2"),
            dv.within(4e-3),
            dv.to_json(),
            json!({"rel_l2": 4e-3}),
        )?;
        c.cmp(&format!("attn_dc_{tag}_vs_f32"), &dc, &want, 1e-2)?;
        let bad = dc
            .iter()
            .zip(&dc16)
            .filter(|(a, b)| bf16::from_f32(**a).to_bits() != bf16::from_f32(**b).to_bits())
            .count();
        c.report.check(
            format!("attn_dc_{tag}_bf16out_is_rne_of_f32out"),
            bad == 0,
            json!({"mismatched": bad, "of": dc.len()}),
            json!({"mismatched": 0}),
        )?;
    }
    Ok(())
}
