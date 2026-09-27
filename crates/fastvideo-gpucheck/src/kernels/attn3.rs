//! sm_120 dense attention tuning and the VSA stage breakdown.
//!
//! `attn3_parity`: `flash_mma_fwd3` bit for bit against `flash_mma_fwd2`
//! (same per-row arithmetic, rescheduled), `flash_mma_fwd3s` within a
//! documented tolerance of it (it skips an O rescale by a factor within a
//! few ulp of 1), and every cuDNN SDPA graph (`cudnn_sdpa::SdpaGraph`) that
//! yields a plan against the f32 reference and against fwd2's bf16 output.
//!
//! `attn3_bench`: fwd2 / fwd3 / fwd3s and each cuDNN graph (heuristic engine
//! configs 0..4) at the real shapes of `attn_bench` (H3 768p, LTX 512p,
//! 1080p 20 s, 4K 5 s), bf16 in and out, median of three synchronized calls.
//!
//! `vsa_stages`: our VSA stage by stage at the grids
//! scripts/gpu/upstream/bench_vsa.py times FastVideo's `video_sparse_attn`
//! at (FastH3 768p / 480p at 56 heads, sparsity 0.8 and 0.9; FastWan 1.3B at
//! 12 heads, 0.8), tile (4, 4, 4), d=128.

use super::*;
use fastvideo_cudarc::wan::attn::FlashKernel;
use fastvideo_cudarc::wan::cudnn_sdpa::{self, SdpaGraph};
use half::bf16;

const D: usize = 128;

fn bits16(x: &CudaTensor, n: usize) -> anyhow::Result<Vec<u16>> {
    let s = x
        .device_slice_bf16()
        .ok_or_else(|| anyhow::anyhow!("expected bf16 output"))?;
    let v = dev()?.stream.memcpy_dtov(&s.slice(0..n))?;
    Ok(v.iter().map(|b| b.to_bits()).collect())
}

fn mismatches<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter().zip(b).filter(|(x, y)| x != y).count()
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

/// One head of `attn_bench`'s structured data (per-64-block bases + noise).
fn structured_head(c: &mut Ctx<'_>, tokens: usize) -> Vec<f32> {
    let n = tokens.div_ceil(64);
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

fn repeat_heads_bf16(head: &[f32], bh: usize) -> anyhow::Result<CudaSlice<bf16>> {
    let one: Vec<bf16> = head.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut all = Vec::with_capacity(one.len() * bh);
    for _ in 0..bh {
        all.extend_from_slice(&one);
    }
    Ok(dev()?.stream.memcpy_stod(&all)?)
}

fn repeat_heads_f32(head: &[f32], bh: usize) -> anyhow::Result<CudaSlice<f32>> {
    let mut all = Vec::with_capacity(head.len() * bh);
    for _ in 0..bh {
        all.extend_from_slice(head);
    }
    up(&all)
}

fn to_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
}

pub(super) fn parity(c: &mut Ctx<'_>) -> StageResult<()> {
    let dev = dev()?;
    if dev.sm_major < 8 {
        c.report
            .note("attn3_skipped", json!({"sm_major": dev.sm_major, "needs": "sm80+"}));
        return Ok(());
    }
    for (b, h, sq, sk) in [
        (1usize, 1usize, 1usize, 1usize),
        (1, 2, 64, 64),
        (1, 2, 64, 128),
        (1, 2, 64, 192),
        (2, 3, 257, 257),
        (1, 4, 300, 512),
        (1, 3, 130, 1),
        (1, 2, 63, 4097),
        (1, 2, 129, 200),
        (1, 2, 2048, 2048),
        (1, 1, 1000, 3000),
    ] {
        let bh = b * h;
        let d = D;
        let q = c.rand(bh * sq * d, 1.5);
        let kk = c.rand(bh * sk * d, 1.5);
        let v = c.rand(bh * sk * d, 1.0);
        let scale = 1.0 / (d as f32).sqrt();
        let want = ref_sdpa(&q, &kk, &v, bh, sq, sk, d, scale);
        let (qt, kt, vt) = (t(q, &[b, h, sq, d])?, t(kk, &[b, h, sk, d])?, t(v, &[b, h, sk, d])?);
        let tag = format!("{b}x{h}x{sq}x{sk}x{d}");
        let run = |kernel, out16| -> anyhow::Result<Vec<f32>> {
            let o = attn::device_mma_sdpa_with(&qt, &kt, &vt, Some(scale), out16, kernel)?
                .ok_or_else(|| anyhow::anyhow!("flash {kernel:?} declined {tag}"))?;
            host_of(&o)
        };
        for out16 in [false, true] {
            let o = if out16 { "_bf16out" } else { "" };
            let v2 = run(FlashKernel::V2, out16)?;
            let v3 = run(FlashKernel::V3, out16)?;
            let v3s = run(FlashKernel::V3s, out16)?;
            let bad = mismatches(
                &v2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &v3.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            );
            c.report.check(
                format!("flash3_{tag}{o}_bitexact_vs_v2"),
                bad == 0,
                json!({"mismatched": bad, "of": v2.len()}),
                json!({"mismatched": 0}),
            )?;
            // fwd3s: O skips a rescale by 2^(m*sl2 - rn(m*sl2)) (|.| <= a few
            // ulp of 1) per tile with no new max; the f32 output moves by
            // ~1e-6 relative, the bf16 output by at most one bf16 ulp here and there.
            let dd = diff(&v3s, &v2);
            let (lim, max_abs) = if out16 { (2e-3, 1.6e-2) } else { (1e-5, 1e-4) };
            c.report.check(
                format!("flash3s_{tag}{o}_vs_v2"),
                dd.within(lim) && dd.max_abs <= max_abs,
                dd.to_json(),
                json!({"rel_l2": lim, "max_abs": max_abs}),
            )?;
            if !out16 {
                c.cmp(&format!("flash3_{tag}_vs_f32"), &v3, &want, 1e-2)?;
                c.cmp(&format!("flash3s_{tag}_vs_f32"), &v3s, &want, 1e-2)?;
            }
        }
        // cuDNN, every graph: vs the f32 reference and vs fwd2's bf16 output.
        let (q16, k16, v16) = (qt.quantize_bf16()?, kt.quantize_bf16()?, vt.quantize_bf16()?);
        let (qs, ks, vs) = (
            q16.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16 q"))?,
            k16.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16 k"))?,
            v16.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16 v"))?,
        );
        let v2_16 = run(FlashKernel::V2, true)?;
        for graph in SdpaGraph::ALL {
            match cudnn_sdpa::plan_for_graph(graph, 0, bh, sq, sk, d) {
                Ok(plan) => {
                    let mut o16 = unsafe { dev.stream.alloc::<bf16>(bh * sq * d) }?;
                    cudnn_sdpa::execute(&plan, qs, ks, vs, &mut o16, scale)?;
                    let got: Vec<f32> = dev.stream.memcpy_dtov(&o16)?.iter().map(|x| x.to_f32()).collect();
                    c.cmp(&format!("cudnn_{}_{tag}_vs_f32", graph.name()), &got, &want, 1e-2)?;
                    let dd = diff(&got, &v2_16);
                    c.report.check(
                        format!("cudnn_{}_{tag}_vs_flash2_bf16", graph.name()),
                        dd.within(1e-2),
                        json!({"diff": dd.to_json(), "plan": plan.info, "configs": plan.configs}),
                        json!({"rel_l2": 1e-2}),
                    )?;
                }
                Err(e) => c.report.note(
                    format!("cudnn_{}_{tag}_no_plan", graph.name()),
                    json!({"error": e}),
                ),
            }
        }
    }
    Ok(())
}

/// Real-shape workloads: (name, heads, tokens).
const SHAPES: &[(&str, usize, usize)] = &[
    ("h3_768p", 56, 37_710),
    ("ltx_512p", 32, 6_144),
    ("ltx_1080p20s", 32, 124_440),
    ("ltx_4k5s", 32, 130_560),
];

pub(super) fn bench(c: &mut Ctx<'_>) -> StageResult<()> {
    let dev = dev()?;
    if dev.sm_major < 8 {
        c.report
            .note("attn3_bench_skipped", json!({"sm_major": dev.sm_major, "needs": "sm80+"}));
        return Ok(());
    }
    let only = std::env::var("FV_ATTN_BENCH_SHAPES").unwrap_or_default();
    let cfgs = std::env::var("FV_CUDNN_BENCH_CFGS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);
    for &(name, bh, tokens) in SHAPES {
        if !only.is_empty() && !only.split(',').any(|s| s.trim() == name) {
            continue;
        }
        let q = repeat_heads_bf16(&structured_head(c, tokens), bh)?;
        let k = repeat_heads_bf16(&structured_head(c, tokens), bh)?;
        let v = repeat_heads_bf16(&c.rand(tokens * D, 1.0), bh)?;
        let n = bh * tokens * D;
        let cmp_n = n.min(2 * tokens * D);
        let shape = vec![1, bh, tokens, D];
        let (qt, kt, vt) = (
            CudaTensor::from_device_slice_bf16(q.clone(), shape.clone())?,
            CudaTensor::from_device_slice_bf16(k.clone(), shape.clone())?,
            CudaTensor::from_device_slice_bf16(v.clone(), shape.clone())?,
        );
        let flops = 4.0 * (bh * tokens) as f64 * tokens as f64 * D as f64;
        let mut row = serde_json::Map::new();
        row.insert("bh".into(), json!(bh));
        row.insert("tokens".into(), json!(tokens));
        let mut bits = std::collections::HashMap::new();
        for (kname, kernel) in [("v2", FlashKernel::V2), ("v3", FlashKernel::V3), ("v3s", FlashKernel::V3s)] {
            let s = median3(&mut || {
                attn::device_mma_sdpa_with(&qt, &kt, &vt, None, true, kernel)?
                    .ok_or_else(|| anyhow::anyhow!("flash declined"))?;
                Ok(())
            })?;
            row.insert(format!("{kname}_ms"), json!(s * 1e3));
            row.insert(format!("{kname}_tflops"), json!(flops / s / 1e12));
            let o = attn::device_mma_sdpa_with(&qt, &kt, &vt, None, true, kernel)?
                .ok_or_else(|| anyhow::anyhow!("flash declined"))?;
            bits.insert(kname, bits16(&o, cmp_n)?);
        }
        let v3_bad = mismatches(&bits["v2"], &bits["v3"]);
        let v3s_diff = diff(&to_f32(&bits["v3s"]), &to_f32(&bits["v2"]));
        row.insert("v3_bit_mismatches_first_heads".into(), json!(v3_bad));
        row.insert("v3s_vs_v2_first_heads".into(), v3s_diff.to_json());
        let scale = 1.0 / (D as f32).sqrt();
        let mut out16 = unsafe { dev.stream.alloc::<bf16>(n) }?;
        for graph in SdpaGraph::ALL {
            let mut g = serde_json::Map::new();
            for cfg in 0..cfgs {
                match cudnn_sdpa::plan_for_graph(graph, cfg, bh, tokens, tokens, D) {
                    Ok(plan) => {
                        let s = median3(&mut || {
                            cudnn_sdpa::execute(&plan, &q, &k, &v, &mut out16, scale)?;
                            Ok(())
                        })?;
                        let got = dev.stream.memcpy_dtov(&out16.slice(0..cmp_n))?;
                        let got: Vec<f32> = got.iter().map(|x| x.to_f32()).collect();
                        let dd = diff(&got, &to_f32(&bits["v2"]));
                        g.insert(
                            format!("cfg{cfg}"),
                            json!({"ms": s * 1e3, "tflops": flops / s / 1e12, "plan": plan.info,
                                   "configs": plan.configs, "vs_v2_first_heads": dd.to_json()}),
                        );
                    }
                    Err(e) => {
                        g.insert(format!("cfg{cfg}"), json!({"error": e}));
                        break;
                    }
                }
            }
            row.insert(format!("cudnn_{}", graph.name()), serde_json::Value::Object(g));
        }
        c.report.check(
            format!("attn3_bench_dense_{name}"),
            v3_bad == 0,
            serde_json::Value::Object(row),
            json!({"v3_bit_mismatches_first_heads": 0}),
        )?;
    }
    Ok(())
}

/// VSA workloads: (name, heads, token grid, sparsities).
const VSA_WORKLOADS: &[(&str, usize, (usize, usize, usize), &[f64])] = &[
    ("fasth3_768p", 56, (37, 24, 42), &[0.8, 0.9]),
    ("fasth3_480p", 56, (37, 15, 26), &[0.8, 0.9]),
    ("fastwan13", 12, (21, 30, 52), &[0.8]),
];

pub(super) fn vsa_stages(c: &mut Ctx<'_>) -> StageResult<()> {
    let dev = dev()?;
    if dev.sm_major < 9 {
        c.report
            .note("vsa_stages_skipped", json!({"sm_major": dev.sm_major, "needs": "sm90+"}));
        return Ok(());
    }
    let only = std::env::var("FV_VSA_WORKLOADS").unwrap_or_default();
    for &(name, bh, grid, sparsities) in VSA_WORKLOADS {
        if !only.is_empty() && !only.split(',').any(|s| s.trim() == name) {
            continue;
        }
        let plan = vsa::TilePlan::new(grid)?;
        let (seq, nb) = (plan.seq, plan.num_tiles());
        let plan_dev = ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, vsa::TILE_ELEMS)?;
        let q = repeat_heads_f32(&c.rand(seq * D, 1.0), bh)?;
        let k = repeat_heads_f32(&c.rand(seq * D, 1.0), bh)?;
        let v = repeat_heads_f32(&c.rand(seq * D, 1.0), bh)?;
        let gate = repeat_heads_f32(&c.rand(seq * D, 1.0), bh)?;
        let scale = 1.0 / (D as f32).sqrt();
        let mut row = serde_json::Map::new();
        row.insert("bh".into(), json!(bh));
        row.insert("grid".into(), json!([grid.0, grid.1, grid.2]));
        row.insert("tokens".into(), json!(seq));
        row.insert("tiles".into(), json!(nb));

        // H3 widens its bf16 q/k/v/gate to f32 before VSA and narrows the
        // output again: the cost of that round trip at this shape.
        let shape = vec![1, bh, seq, D];
        let q16 = CudaTensor::from_device_slice(q.clone(), shape.clone())?.quantize_bf16()?;
        let widen = median3(&mut || {
            for _ in 0..4 {
                q16.to_f32_act()?;
            }
            Ok(())
        })?;
        row.insert("h3_widen_qkvg_ms".into(), json!(widen * 1e3));
        drop(q16);

        // bf16-input kernels (H3's bf16 activations): bit for bit against the
        // f32 kernels on the widened tensors, and their timings.
        {
            let to16 = |x: &CudaSlice<f32>| -> anyhow::Result<CudaSlice<bf16>> {
                let t = CudaTensor::from_device_slice(x.clone(), shape.clone())?.quantize_bf16()?;
                Ok(t.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16"))?.clone())
            };
            let widen = |x: &CudaSlice<bf16>| -> anyhow::Result<CudaSlice<f32>> {
                let t = CudaTensor::from_device_slice_bf16(x.clone(), shape.clone())?.to_f32_act()?;
                Ok(t.device_slice().ok_or_else(|| anyhow::anyhow!("f32"))?.clone())
            };
            let (q16, g16) = (to16(&q)?, to16(&gate)?);
            let (qw, gw) = (widen(&q16)?, widen(&g16)?);
            let m32 = down(&ops::vsa_tile_mean_round_device(&qw, &plan_dev, bh, seq, D, true)?)?;
            let m16 = down(&ops::vsa_tile_mean_bf16_device(&q16, &plan_dev, bh, seq, D)?)?;
            let bad_mean = mismatches(
                &m32.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &m16.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            );
            let t32 = dev.stream.memcpy_dtov(&ops::vsa_tile_qkv_device(&qw, &plan_dev, bh, seq, D)?)?;
            let t16 = dev.stream.memcpy_dtov(&ops::vsa_tile_qkv_bf16_device(&q16, &plan_dev, bh, seq, D)?)?;
            let bad_tile = mismatches(
                &t32.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &t16.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            );
            let sparse = up(&c.rand(bh * nb * 64 * D, 1.0))?;
            let coarse0 = up(&c.rand(bh * nb * D, 1.0))?;
            let mut o32 = ops::fill_device(bh * seq * D, 0.0)?;
            let mut o16 = ops::fill_device(bh * seq * D, 0.0)?;
            ops::vsa_combine_round_device(&sparse, &coarse0, Some(&gw), &plan_dev, &mut o32, bh, nb, 0, seq, D, true)?;
            ops::vsa_combine_gate16_device(&sparse, &coarse0, Some(&g16), &plan_dev, &mut o16, bh, nb, 0, seq, D)?;
            let bad_comb = mismatches(
                &down(&o32)?.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &down(&o16)?.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            );
            let tm16 = median3(&mut || {
                for _ in 0..3 {
                    ops::vsa_tile_mean_bf16_device(&q16, &plan_dev, bh, seq, D)?;
                }
                Ok(())
            })?;
            let tq16 = median3(&mut || {
                for _ in 0..3 {
                    ops::vsa_tile_qkv_bf16_device(&q16, &plan_dev, bh, seq, D)?;
                }
                Ok(())
            })?;
            let cb16 = median3(&mut || {
                ops::vsa_combine_gate16_device(&sparse, &coarse0, Some(&g16), &plan_dev, &mut o16, bh, nb, 0, seq, D)?;
                Ok(())
            })?;
            c.report.check(
                format!("vsa_b16_{name}_bitexact_vs_f32"),
                bad_mean == 0 && bad_tile == 0 && bad_comb == 0,
                json!({"tile_mean_mismatched": bad_mean, "tile_qkv_mismatched": bad_tile,
                       "combine_mismatched": bad_comb,
                       "tile_mean_x3_ms": tm16 * 1e3, "tile_qkv_x3_ms": tq16 * 1e3, "combine_ms": cb16 * 1e3}),
                json!({"all": 0}),
            )?;
        }

        let coarse_fn = || -> anyhow::Result<(CudaSlice<f32>, CudaSlice<f32>)> {
            let qc = ops::vsa_tile_mean_device(&q, &plan_dev, bh, seq, D)?;
            let kc = ops::vsa_tile_mean_device(&k, &plan_dev, bh, seq, D)?;
            let vc = ops::vsa_tile_mean_device(&v, &plan_dev, bh, seq, D)?;
            let mut scores = ops::alloc(bh * nb * nb)?;
            device::matmul_linear_wt_strided_batched_f32(&qc, &kc, &mut scores, bh, nb, D, nb, scale)?;
            let probs = ops::softmax_last_device(&scores, nb)?;
            let mut coarse = ops::alloc(bh * nb * D)?;
            device::matmul_2d_strided_batched_f32(&probs, &vc, &mut coarse, bh, nb, nb, D)?;
            Ok((scores, coarse))
        };
        let coarse_s = median3(&mut || {
            coarse_fn()?;
            Ok(())
        })?;
        row.insert("coarse_ms".into(), json!(coarse_s * 1e3));
        let tile_s = median3(&mut || {
            for x in [&q, &k, &v] {
                ops::vsa_tile_qkv_device(x, &plan_dev, bh, seq, D)?;
            }
            Ok(())
        })?;
        row.insert("tile_ms".into(), json!(tile_s * 1e3));
        let (scores, coarse) = coarse_fn()?;
        for &sp in sparsities {
            let topk = vsa::topk_for(sp, nb);
            let topk_s = median3(&mut || {
                ops::vsa_topk_device(&scores, bh * nb, nb, topk)?;
                Ok(())
            })?;
            let selected = ops::vsa_topk_device(&scores, bh * nb, nb, topk)?;
            let mut fine = std::collections::BTreeMap::new();
            for kernel in ["tma2", "tma"] {
                std::env::set_var("FASTVIDEO_VSA_KERNEL", kernel);
                let s = median3(&mut || {
                    ops::vsa_mma_attn_range_device(&q, &k, &v, &selected, &plan_dev, bh, seq, D, topk, scale, 0, nb)?;
                    Ok(())
                });
                std::env::remove_var("FASTVIDEO_VSA_KERNEL");
                fine.insert(kernel, s?);
            }
            let sparse = ops::vsa_mma_attn_range_device(&q, &k, &v, &selected, &plan_dev, bh, seq, D, topk, scale, 0, nb)?;
            let mut out = ops::alloc(bh * seq * D)?;
            let combine_s = median3(&mut || {
                ops::vsa_combine_device(&sparse, &coarse, Some(&gate), &plan_dev, &mut out, bh, nb, 0, seq, D)?;
                Ok(())
            })?;
            drop((sparse, out));
            let total_s = median3(&mut || {
                vsa::vsa_attention_device(&q, &k, &v, Some(&gate), &plan_dev, topk, bh, seq, D, scale, nb)?;
                Ok(())
            })?;
            let fine_s = fine["tma2"] - tile_s;
            let flops = 4.0 * (bh * nb * 64) as f64 * (topk * 64) as f64 * D as f64;
            row.insert(
                format!("sparsity{sp}"),
                json!({
                    "topk": topk,
                    "topk_ms": topk_s * 1e3,
                    "tile_plus_fine_tma2_ms": fine["tma2"] * 1e3,
                    "tile_plus_fine_tma_ms": fine["tma"] * 1e3,
                    "fine_ms": fine_s * 1e3,
                    "fine_tflops": flops / fine_s / 1e12,
                    "combine_ms": combine_s * 1e3,
                    "total_ms": total_s * 1e3,
                }),
            );
        }
        c.report.note(format!("vsa_stages_{name}"), serde_json::Value::Object(row));
    }
    Ok(())
}
