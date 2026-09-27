//! FP8 attention checks (`fv-gpucheck kernels --groups attn_fp8`).
//!
//! * **attn_fp8** — the SageAttention-style kernels of
//!   [`fastvideo_cudarc::wan::attn_fp8`] (FP8 `Q K^T`, bf16 `P V`) against
//!   the bf16 flash kernel and an f64 host SDPA, dense and VSA fine stage,
//!   on synthetic inputs in two regimes: unit-variance q/k ("flat" scores)
//!   and q x 3 with a per-channel K offset ("peaked", the case K smoothing
//!   is for). The tolerance is stated relative to the bf16 kernel:
//!   rel-L2 <= 5e-2 and cosine >= 0.998, and the f64 error of both kernels
//!   is reported beside it. Then both are timed at the H3 768p shapes
//!   (dense: 56 heads x 43 008 rows; VSA: 11 prefix + 660 video tiles at
//!   top-k 11 + 132, the 8-step recipe's 0.8).

use fastvideo_cudarc::wan::attn::{self, FlashKernel};
use fastvideo_cudarc::wan::attn_fp8;
use fastvideo_cudarc::wan::ops;
use serde_json::json;

use crate::kernels_fp8::{bf16v, dev, host, rand, t16, t32, time_ms};
use crate::report::{Report, StageResult};

/// Pass limits against the bf16 kernel.
const MAX_REL_L2: f64 = 5e-2;
const MIN_COSINE: f64 = 0.998;

struct Err3 {
    rel_l2: f64,
    cosine: f64,
    max_abs: f64,
}

fn compare(got: &[f32], want: &[f32]) -> Err3 {
    let (mut d2, mut w2, mut g2, mut gw, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        d2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        gw += g * w;
        mx = mx.max((g - w).abs());
    }
    let finite = got.iter().all(|x| x.is_finite());
    Err3 {
        rel_l2: if finite {
            (d2 / w2.max(1e-300)).sqrt()
        } else {
            f64::INFINITY
        },
        cosine: if finite {
            gw / (g2 * w2).sqrt().max(1e-300)
        } else {
            0.0
        },
        max_abs: mx,
    }
}

fn js(e: &Err3) -> serde_json::Value {
    json!({"rel_l2": e.rel_l2, "cosine": e.cosine, "max_abs": e.max_abs})
}

/// f64 SDPA of `[bh, s, d]` inputs.
fn sdpa_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0f32; bh * sq * d];
    let mut p = vec![0f64; sk];
    for b in 0..bh {
        for i in 0..sq {
            let qi = &q[(b * sq + i) * d..][..d];
            let mut mx = f64::NEG_INFINITY;
            for (j, pj) in p.iter_mut().enumerate() {
                let kj = &k[(b * sk + j) * d..][..d];
                *pj = qi
                    .iter()
                    .zip(kj)
                    .map(|(&a, &c)| f64::from(a) * f64::from(c))
                    .sum::<f64>()
                    * scale;
                mx = mx.max(*pj);
            }
            let mut z = 0.0;
            for pj in p.iter_mut() {
                *pj = (*pj - mx).exp();
                z += *pj;
            }
            for c in 0..d {
                let acc: f64 = p
                    .iter()
                    .enumerate()
                    .map(|(j, pj)| pj * f64::from(v[(b * sk + j) * d + c]))
                    .sum();
                out[(b * sq + i) * d + c] = (acc / z) as f32;
            }
        }
    }
    out
}

/// q, k, v (bf16 values) in a regime: `peaked` scales q by 3 and adds a
/// per-channel offset to K.
fn inputs(
    seed: &mut u64,
    bh: usize,
    s: usize,
    d: usize,
    peaked: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = bh * s * d;
    let mut q = rand(seed, n, 1.0);
    let mut k = rand(seed, n, 1.0);
    let v = rand(seed, n, 1.0);
    if peaked {
        let off = rand(seed, bh * d, 2.0);
        for x in q.iter_mut() {
            *x *= 3.0;
        }
        for (i, x) in k.iter_mut().enumerate() {
            let (b, c) = (i / (s * d), i % d);
            *x += off[b * d + c];
        }
    }
    (bf16v(q), bf16v(k), bf16v(v))
}

pub fn attn_fp8_group(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    if !attn_fp8::supported() {
        report.note(
            "attn_fp8",
            json!({"skipped": "needs sm_89+ (FP8 mma.sync)"}),
        );
        return Ok(());
    }
    let d = 128usize;
    // ---- dense parity
    for &(bh, s, peaked) in &[
        (2usize, 300usize, false),
        (2, 300, true),
        (4, 1000, false),
        (4, 1000, true),
        (8, 4097, true),
    ] {
        let (q, k, v) = inputs(seed, bh, s, d, peaked);
        let shape = [1, bh, s, d];
        let (q16, k16, v16) = (t16(&q, &shape)?, t16(&k, &shape)?, t16(&v, &shape)?);
        let bf = attn::device_mma_sdpa_with(&q16, &k16, &v16, None, false, FlashKernel::V2)?
            .ok_or_else(|| anyhow::anyhow!("bf16 flash kernel did not run"))?;
        let f8 = attn_fp8::dense_sdpa(&q16, &k16, &v16, None, false)?
            .ok_or_else(|| anyhow::anyhow!("attn_fp8 dense did not run"))?;
        let f8b = attn_fp8::dense_sdpa(&q16, &k16, &v16, None, true)?
            .ok_or_else(|| anyhow::anyhow!("attn_fp8 dense (bf16 out) did not run"))?;
        let (bf, f8, f8b) = (host(&bf)?, host(&f8)?, host(&f8b)?);
        let e = compare(&f8, &bf);
        let eb = compare(&f8b, &f8);
        let mut values = json!({"vs_bf16": js(&e), "bf16_out_vs_f32_out": js(&eb)});
        if s <= 1000 && bh <= 2 {
            let exact = sdpa_host(&q, &k, &v, bh, s, s, d);
            values["fp8_vs_f64"] = js(&compare(&f8, &exact));
            values["bf16_vs_f64"] = js(&compare(&bf, &exact));
        }
        let tag = format!(
            "dense_bh{bh}_s{s}_{}",
            if peaked { "peaked" } else { "flat" }
        );
        report.check(
            format!("attn_fp8_{tag}"),
            e.rel_l2 <= MAX_REL_L2 && e.cosine >= MIN_COSINE && eb.rel_l2 <= 1e-2,
            values,
            json!({"rel_l2": MAX_REL_L2, "cosine": MIN_COSINE, "bf16_out_rel_l2": 1e-2}),
        )?;
    }

    // ---- VSA fine-stage parity (same selection for both kernels)
    let vsa_case = |seed: &mut u64,
                    grid: (usize, usize, usize),
                    heads: usize,
                    topk: usize,
                    peaked: bool|
     -> anyhow::Result<(Err3, f64, f64)> {
        let plan = fastvideo_cudarc::wan::vsa::TilePlan::new(grid)?;
        let (nb, seq) = (plan.num_tiles(), plan.seq);
        let dp = ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, 64)?;
        let (q, k, v) = inputs(seed, heads, seq, d, peaked);
        let shape = [1, heads, seq, d];
        let (q32, k32, v32) = (t32(&q, &shape)?, t32(&k, &shape)?, t32(&v, &shape)?);
        let (qd, kd, vd) = (
            q32.device_slice().ok_or_else(|| anyhow::anyhow!("q"))?,
            k32.device_slice().ok_or_else(|| anyhow::anyhow!("k"))?,
            v32.device_slice().ok_or_else(|| anyhow::anyhow!("v"))?,
        );
        // Deterministic distinct key tiles per (head, query tile).
        let topk = topk.min(nb);
        let mut sel = Vec::with_capacity(heads * nb * topk);
        for h in 0..heads {
            for i in 0..nb {
                for j in 0..topk {
                    sel.push(((i + h * 7 + j) % nb) as u32);
                }
            }
        }
        let selected = dev()?.stream.memcpy_stod(&sel)?;
        let scale = 1.0 / (d as f32).sqrt();
        let run_bf = || {
            ops::vsa_mma_attn_range_device(
                qd, kd, vd, &selected, &dp, heads, seq, d, topk, scale, 0, nb,
            )
        };
        let run_f8 = || {
            attn_fp8::vsa_fine(
                qd, kd, vd, &selected, &dp, heads, seq, d, topk, scale, 0, nb,
            )
        };
        let bf = run_bf()?;
        let f8 = run_f8()?;
        let bf = dev()?.stream.memcpy_dtov(&bf)?;
        let f8 = dev()?.stream.memcpy_dtov(&f8)?;
        // Compare live slots only (padding rows are unused by the combine).
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for h in 0..heads {
            for (slot, &src) in plan.slot_src.iter().enumerate() {
                if src >= 0 {
                    let o = (h * nb * 64 + slot) * d;
                    a.extend_from_slice(&f8[o..o + d]);
                    b.extend_from_slice(&bf[o..o + d]);
                }
            }
        }
        let e = compare(&a, &b);
        let t_bf = time_ms(3, || {
            run_bf()?;
            Ok(())
        })?;
        let t_f8 = time_ms(3, || {
            run_f8()?;
            Ok(())
        })?;
        Ok((e, t_bf, t_f8))
    };
    for &(grid, heads, topk, peaked) in &[
        ((4usize, 12usize, 20usize), 4usize, 6usize, false),
        ((4, 12, 20), 4, 6, true),
        ((5, 9, 14), 2, 7, true),
    ] {
        let (e, _, _) = vsa_case(seed, grid, heads, topk, peaked)?;
        report.check(
            format!(
                "attn_fp8_vsa_{}x{}x{}_{}",
                grid.0,
                grid.1,
                grid.2,
                if peaked { "peaked" } else { "flat" }
            ),
            e.rel_l2 <= MAX_REL_L2 && e.cosine >= MIN_COSINE,
            json!({"vs_bf16": js(&e)}),
            json!({"rel_l2": MAX_REL_L2, "cosine": MIN_COSINE}),
        )?;
    }

    // ---- timing at the H3 768p shapes
    {
        // Dense: 56 heads x 43 008 rows (the 768p 5 s packed sequence, rounded).
        let (h, s) = (56usize, 43008usize);
        let n = h * s * d;
        let mk = |seed: &mut u64| t16(&bf16v(rand(seed, n, 1.0)), &[1, h, s, d]);
        let (q16, k16, v16) = (mk(seed)?, mk(seed)?, mk(seed)?);
        let bf = time_ms(3, || {
            attn::device_mma_sdpa_with(&q16, &k16, &v16, None, true, FlashKernel::V2)?;
            Ok(())
        })?;
        let f8 = time_ms(3, || {
            attn_fp8::dense_sdpa(&q16, &k16, &v16, None, true)?;
            Ok(())
        })?;
        let flops = 4.0 * (s as f64) * (s as f64) * (d as f64) * (h as f64);
        report.note(
            "attn_fp8_dense_bench_h56_s43008",
            json!({"bf16_ms": bf, "fp8_ms": f8, "speedup": bf / f8, "bf16_tflops": flops / bf * 1e-9, "fp8_tflops": flops / f8 * 1e-9}),
        );
        drop((q16, k16, v16));
        // VSA: 8-step 768p grid (660 video tiles), top-k 132 (0.8) and 66 (0.9);
        // the prefix tiles are left out (they only shift the tile count).
        for topk in [132usize, 66] {
            let (e, bf, f8) = vsa_case(seed, (20, 48, 44), 56, topk, false)?;
            report.note(
                format!("attn_fp8_vsa_bench_660tiles_topk{topk}"),
                json!({"bf16_ms": bf, "fp8_ms": f8, "speedup": bf / f8, "vs_bf16": js(&e)}),
            );
        }
    }
    Ok(())
}
