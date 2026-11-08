//! `moe_aot`: the LingBot MoE module built ahead of time (`AOT_MOE`, nvcc
//! fast math + FTZ) against the same source NVRTC-compiled at run time (the
//! pre-ADR-0024 path). Every entry point runs on the same seeded inputs under
//! both builds; outputs must be equal bit for bit
//! (FVID-2026-11-07-moe-aot-flags, ADR-0024 step 1a).

use super::*;
use fastvideo_cudarc::rt::ops::{GroupTopk, MoeSource};

/// Every MoE entry point on fixed inputs, as raw bits.
fn outputs(c: &mut Ctx<'_>) -> anyhow::Result<Vec<(&'static str, Vec<u32>)>> {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let mut out = Vec::new();

    // rope_real: [B, H, S, D] with [S, D] tables.
    let (b, h, s, d) = (2usize, 3usize, 17usize, 64usize);
    let x = c.rand(b * h * s * d, 1.0);
    let cos: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.37).cos()).collect();
    let sin: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.37).sin()).collect();
    let y = ops::rope_real_device(&up(&x)?, &up(&cos)?, &up(&sin)?, s, d)?;
    out.push(("rope_real", bits(&down(&y)?)));

    // sigmoid_f: wide range, incl. saturating values (fast-math expf).
    let mut a = c.rand(4099, 6.0);
    a.extend_from_slice(&[0.0, -0.0, 88.0, -88.0, 1e-8, -1e-8]);
    out.push(("sigmoid_f", bits(&down(&ops::sigmoid_device(&up(&a)?)?)?)));

    // topk_last, with and without normalisation, and a row of exact ties.
    let (rows, width, k) = (37usize, 128usize, 8usize);
    let mut scores = c.rand(rows * width, 1.0);
    scores[..width].fill(0.25);
    for norm in [false, true] {
        let (idx, val) = ops::topk_last_device(&up(&scores)?, width, k, norm)?;
        out.push((
            if norm {
                "topk_last_norm/idx"
            } else {
                "topk_last/idx"
            },
            idx,
        ));
        out.push((
            if norm {
                "topk_last_norm/val"
            } else {
                "topk_last/val"
            },
            bits(&val),
        ));
    }

    // scatter_add_rows (+ copy_f): repeated destination rows accumulate.
    let (base_rows, src_rows, dd) = (19usize, 53usize, 96usize);
    let base = c.rand(base_rows * dd, 1.0);
    let src = c.rand(src_rows * dd, 1.0);
    let idx: Vec<u32> = (0..src_rows)
        .map(|i| ((i * 7) % base_rows) as u32)
        .collect();
    let w = c.rand(src_rows, 0.5);
    let y = ops::scatter_add_rows_device(&up(&base)?, &up(&src)?, &idx, &w, dd)?;
    out.push(("scatter_add_rows", bits(&down(&y)?)));

    // moe_group_topk: plain and group-limited, softmax and sigmoid routing.
    let (tokens, experts) = (41usize, 128usize);
    let logits = c.rand(tokens * experts, 2.0);
    let bias = c.rand(experts, 0.05);
    for (name, n_group, topk_group, softmax, round_bf16) in [
        ("moe_group_topk/plain", 1usize, 1usize, true, false),
        ("moe_group_topk/grouped", 8, 4, false, true),
    ] {
        let p = GroupTopk {
            experts,
            top_k: 8,
            n_group,
            topk_group,
            softmax,
            norm: true,
            scale: 2.5,
            round_bf16,
        };
        let (i, v) = ops::moe_group_topk_device(&up(&logits)?, &up(&bias)?, p)?;
        out.push((name, i));
        out.push((name, bits(&v)));
    }

    // moe_combine: n tokens, k slots, rows of ys gathered by pos.
    let (n, kk, d2, ys_rows) = (29usize, 4usize, 80usize, 61usize);
    let ys = c.rand(ys_rows * d2, 1.0);
    let pos: Vec<u32> = (0..n * kk).map(|i| ((i * 13) % ys_rows) as u32).collect();
    let wc = c.rand(n * kk, 0.5);
    let y = ops::moe_combine_device(&up(&ys)?, &pos, &wc, n, kk, d2)?;
    out.push(("moe_combine", bits(&down(&y)?)));
    Ok(out)
}

pub(super) fn run(c: &mut Ctx<'_>) -> StageResult<()> {
    let seed = c.seed;
    let mut runs = Vec::new();
    for source in [MoeSource::Embedded, MoeSource::Nvrtc] {
        ops::set_moe_source(source);
        c.seed = seed; // same inputs for both builds
        let res = ops::moe_kernels_origin()
            .map_err(anyhow::Error::from)
            .and_then(|origin| Ok((origin, outputs(c)?)));
        ops::set_moe_source(MoeSource::Auto);
        let (origin, outs) = res?;
        runs.push((source, origin, outs));
    }
    let (_, aot_origin, aot) = &runs[0];
    let (_, nvrtc_origin, jit) = &runs[1];
    c.report.check(
        "moe_aot/origins",
        aot_origin.starts_with("cubin ") && nvrtc_origin.starts_with("nvrtc "),
        json!({"embedded": aot_origin, "nvrtc": nvrtc_origin}),
        json!({"embedded": "cubin sm<N>", "nvrtc": "nvrtc ..."}),
    )?;
    for ((name, a), (_, b)) in aot.iter().zip(jit) {
        let differ = a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
        c.report.check(
            format!("moe_aot/{name}"),
            differ == 0,
            json!({"elements": a.len(), "differing": differ}),
            json!({"differing": 0}),
        )?;
    }
    Ok(())
}
