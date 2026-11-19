//! `attn_aot`: the FP8 and SageAttention dense kernels as served (embedded
//! nvcc cubins, `--use_fast_math`) against the NVRTC build of the same
//! source (`FASTVIDEO_KERNELS=nvrtc`: the fallback for an unbuilt GPU and the
//! "keep the source path exercised" switch). cudarc 0.17's `use_fast_math:
//! Some(true)` only emits `--fmad=true`, so until the fallback also passed
//! `--use_fast_math` the two builds differed (RTX PRO 6000, `1a92cf4`: 3-15%
//! of elements, max |diff| 0.006 fp8 / 0.011 sage). Now they must be equal
//! bit for bit. Found during ADR-0024 WO-1a, where the same flag gap made
//! the MoE module fail bit equality.

use super::*;
use fastvideo_cudarc::rt::{attn_fp8, attn_sage, device};
use half::bf16;

const BH: usize = 4;
const S: usize = 1_000; // not a multiple of the 128 / 64 tiles
const D: usize = 128;

struct Run {
    fp8_origin: String,
    sage_origin: String,
    /// (name, f32 output) per (module, out_bf16).
    outs: Vec<(String, Vec<f32>)>,
}

/// One full pass on a fresh thread, so the thread-local kernel caches load
/// from scratch under the current `FASTVIDEO_KERNELS`.
fn run_on_fresh_thread(q: &[bf16], k: &[bf16], v: &[bf16]) -> anyhow::Result<Run> {
    let (q, k, v) = (q.to_vec(), k.to_vec(), v.to_vec());
    std::thread::spawn(move || -> anyhow::Result<Run> {
        let dev = device::global_device().ok_or_else(|| anyhow::anyhow!("no CUDA device"))?;
        let up = |x: &[bf16]| -> anyhow::Result<CudaTensor> {
            Ok(CudaTensor::from_device_slice_bf16(
                dev.stream.memcpy_stod(x)?,
                vec![1, BH, S, D],
            )?)
        };
        let (q, k, v) = (up(&q)?, up(&k)?, up(&v)?);
        let mut outs = Vec::new();
        for out_bf16 in [false, true] {
            let tag = if out_bf16 { "bf16" } else { "f32" };
            let o = attn_fp8::dense_sdpa(&q, &k, &v, None, out_bf16)?
                .ok_or_else(|| anyhow::anyhow!("attn_fp8 declined the shape"))?;
            outs.push((format!("fp8/{tag}"), host_of(&o)?));
            let o = attn_sage::dense_sdpa(&q, &k, &v, None, out_bf16)?
                .ok_or_else(|| anyhow::anyhow!("attn_sage declined the shape"))?;
            outs.push((format!("sage/{tag}"), host_of(&o)?));
        }
        Ok(Run {
            fp8_origin: attn_fp8::kernels_origin()?,
            sage_origin: attn_sage::kernels_origin()?,
            outs,
        })
    })
    .join()
    .map_err(|_| anyhow::anyhow!("attn_aot worker panicked"))?
}

pub(super) fn run(c: &mut Ctx<'_>) -> StageResult<()> {
    if !attn_fp8::supported() || !attn_sage::supported() {
        return Ok(());
    }
    let mk = |c: &mut Ctx<'_>| -> Vec<bf16> {
        c.rand(BH * S * D, 1.0)
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect()
    };
    let (q, k, v) = (mk(c), mk(c), mk(c));

    let saved = fastvideo_models::settings::env_var("FASTVIDEO_KERNELS").ok();
    std::env::remove_var("FASTVIDEO_KERNELS");
    let served = run_on_fresh_thread(&q, &k, &v);
    std::env::set_var("FASTVIDEO_KERNELS", "nvrtc");
    let nvrtc = run_on_fresh_thread(&q, &k, &v);
    match &saved {
        Some(s) => std::env::set_var("FASTVIDEO_KERNELS", s),
        None => std::env::remove_var("FASTVIDEO_KERNELS"),
    }
    let (served, nvrtc) = (served?, nvrtc?);

    let embedded = |o: &str| o.starts_with("cubin ") || o.starts_with("ptx ");
    c.report.check(
        "attn_aot/origins",
        embedded(&served.fp8_origin)
            && embedded(&served.sage_origin)
            && nvrtc.fp8_origin.starts_with("nvrtc")
            && nvrtc.sage_origin.starts_with("nvrtc"),
        json!({
            "served": [served.fp8_origin, served.sage_origin],
            "nvrtc": [nvrtc.fp8_origin, nvrtc.sage_origin],
        }),
        json!({"served": "cubin|ptx", "nvrtc": "nvrtc ..."}),
    )?;
    for ((name, a), (_, b)) in served.outs.iter().zip(&nvrtc.outs) {
        let differing = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let max_abs = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        c.report.check(
            format!("attn_aot/{name}"),
            differing == 0 && a.len() == b.len(),
            json!({"elements": a.len(), "differing": differing, "max_abs_diff": max_abs}),
            json!({"differing": 0}),
        )?;
    }
    Ok(())
}
