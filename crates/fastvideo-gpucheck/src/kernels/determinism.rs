//! `determinism`: the kernels whose selection or reduction order could make a
//! result differ between runs, each run several times on the same inputs in
//! one process; every repeat must equal the first bit for bit
//! (docs/perf/determinism.md). Across processes the same bits are checked by
//! the generations of `runpod-matrix.sh determinism`.

use super::*;
use half::bf16;

const REPEATS: usize = 3;

fn bits_of(x: &CudaTensor) -> anyhow::Result<Vec<u32>> {
    Ok(host_of(x)?.iter().map(|v| v.to_bits()).collect())
}

fn same<T: PartialEq>(runs: &[Vec<T>]) -> usize {
    runs.iter()
        .skip(1)
        .map(|r| r.iter().zip(&runs[0]).filter(|(a, b)| a != b).count() + r.len().abs_diff(runs[0].len()))
        .sum()
}

pub(super) fn run(c: &mut Ctx<'_>) -> StageResult<()> {
    let dev = dev()?;
    // Dense SDPA under `auto` (the fixed rule; on sm_12x cuDNN or fwd2 per
    // shape): a self-attention shape the rule sends to cuDNN on sm_12x, and a
    // cross-attention shape (a tie, fwd2).
    for (name, bh, sq, sk) in [("self_24k6", 8usize, 24_832usize, 24_832usize), ("cross", 8, 9_690, 1_024)] {
        let d = 128;
        let q = t(c.rand(bh * sq * d, 1.0), &[1, bh, sq, d])?;
        let k = t(c.rand(bh * sk * d, 1.0), &[1, bh, sk, d])?;
        let v = t(c.rand(bh * sk * d, 1.0), &[1, bh, sk, d])?;
        let mut runs = Vec::new();
        for _ in 0..REPEATS {
            let o = attn::device_mma_sdpa(&q, &k, &v, None, true)?
                .ok_or_else(|| anyhow::anyhow!("fused sdpa declined"))?;
            runs.push(bits_of(&o)?);
        }
        let bad = same(&runs);
        let pick = fastvideo_cudarc::wan::sdpa_rule::pick(dev.sm_major, sq, sk, d);
        c.report.check(
            format!("determinism_sdpa_{name}"),
            bad == 0,
            json!({"mismatches": bad, "repeats": REPEATS, "rule": pick.name(), "sq": sq, "sk": sk}),
            json!({"mismatches": 0}),
        )?;
    }

    // The per-block f64 reductions behind the TeaCache / FBCache decisions.
    let n = 5_000_003;
    let (a, b) = (up(&c.rand(n, 1.0))?, up(&c.rand(n, 1.0))?);
    let mut sums = Vec::new();
    let mut ltx = Vec::new();
    for _ in 0..REPEATS {
        let (x, y) = ops::abs_diff_sums_device(&a, &b)?;
        sums.push(vec![x.to_bits(), y.to_bits()]);
        let (x, y) = ops::ltx_abs_diff_sums_device(&a, &b)?;
        ltx.push(vec![x.to_bits(), y.to_bits()]);
    }
    let (bad_a, bad_l) = (same(&sums), same(&ltx));
    c.report.check(
        "determinism_abs_diff_sums",
        bad_a == 0 && bad_l == 0,
        json!({"teacache_mismatches": bad_a, "fbcache_mismatches": bad_l, "n": n}),
        json!({"mismatches": 0}),
    )?;

    // Transposed conv (vocoder / audio VAE upsampling): cuDNN backward-data,
    // never its atomic ALGO_0.
    let (ci, co, l, kw) = (64usize, 32usize, 4_096usize, 16usize);
    let x = t(c.rand(ci * l, 1.0), &[1, ci, l])?;
    let w = t(c.rand(ci * co * kw, 0.1), &[ci, co, kw])?;
    let mut runs = Vec::new();
    for _ in 0..REPEATS {
        runs.push(bits_of(&x.conv_transpose1d(&w, None, 4, 8, 1, 1, 0)?)?);
    }
    let bad = same(&runs);
    c.report.check(
        "determinism_conv_transpose1d",
        bad == 0,
        json!({"mismatches": bad, "repeats": REPEATS}),
        json!({"mismatches": 0}),
    )?;

    // FP8 attention (opt-in profiles): its K column sums are two-pass now.
    if fastvideo_cudarc::wan::attn_fp8::supported() {
        let (bh, s, d) = (4usize, 4_160usize, 128usize);
        let mk = |c: &mut Ctx<'_>| -> anyhow::Result<CudaTensor> {
            let v: Vec<bf16> = c.rand(bh * s * d, 1.0).iter().map(|&x| bf16::from_f32(x)).collect();
            Ok(CudaTensor::from_device_slice_bf16(dev.stream.memcpy_stod(&v)?, vec![1, bh, s, d])?)
        };
        let (q, k, v) = (mk(c)?, mk(c)?, mk(c)?);
        let mut runs = Vec::new();
        for _ in 0..REPEATS {
            match fastvideo_cudarc::wan::attn_fp8::dense_sdpa(&q, &k, &v, None, false)? {
                Some(o) => runs.push(bits_of(&o)?),
                None => break,
            }
        }
        if runs.len() == REPEATS {
            let bad = same(&runs);
            c.report.check(
                "determinism_fp8_attention",
                bad == 0,
                json!({"mismatches": bad, "repeats": REPEATS}),
                json!({"mismatches": 0}),
            )?;
        }
    }
    Ok(())
}
