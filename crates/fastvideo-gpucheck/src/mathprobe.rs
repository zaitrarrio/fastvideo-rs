//! Which cuBLAS math actually runs on this GPU + cuBLAS build.
//!
//! Times the DiT's hot linears at the 8s-clip token count under each math
//! option and compares every result with plain FP32. An option whose output
//! is bit-identical to FP32 and no faster was not honored by cuBLAS.

use cudarc::driver::CudaSlice;
use fastvideo_cudarc::wan::device::{self, GemmMath};
use serde_json::json;

use crate::metrics::diff;
use crate::rand_weights::randn;
use crate::report::{Report, StageResult};

const REPEATS: usize = 3;

fn dev() -> anyhow::Result<std::sync::Arc<device::DeviceContext>> {
    device::global_device().ok_or_else(|| anyhow::anyhow!("no live CUDA device"))
}

/// Median seconds of `REPEATS` synchronized calls after one warm-up.
fn time(mut f: impl FnMut() -> anyhow::Result<()>) -> anyhow::Result<f64> {
    f()?;
    let mut secs = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        device::synchronize()?;
        let t = std::time::Instant::now();
        f()?;
        device::synchronize()?;
        secs.push(t.elapsed().as_secs_f64());
    }
    secs.sort_by(|a, b| a.total_cmp(b));
    Ok(secs[REPEATS / 2])
}

pub fn run(report: &mut Report) -> StageResult<()> {
    let info = crate::gpu::init("cuda")?;
    report.set("device", &info);
    report.set(
        "cublas_lib",
        std::fs::read_link("/workspace/fv-libs/libcublas.so").ok(),
    );
    let d = dev()?;
    // (name, tokens, in, out): 480p 8s clip = 32760 tokens, batch 1 (DMD).
    for (name, m, k, n) in [
        ("ffn_up", 32_760usize, 1536usize, 8960usize),
        ("qkv", 32_760, 1536, 4608),
        ("ffn_down", 32_760, 8960, 1536),
    ] {
        let x = randn(1, m * k, 1.0);
        let w = randn(2, n * k, 0.02);
        let (xd, wd) = (d.stream.memcpy_stod(&x)?, d.stream.memcpy_stod(&w)?);
        let mut out = unsafe { d.stream.alloc::<f32>(m * n) }?;
        let mut results = serde_json::Map::new();
        let mut fp32: Option<Vec<f32>> = None;
        for math in [GemmMath::F32, GemmMath::Tf32, GemmMath::Bf16] {
            let secs = time(|| {
                Ok(device::matmul_linear_wt_math(
                    &xd, &wd, &mut out, m, k, n, math,
                )?)
            })?;
            let host = d.stream.memcpy_dtov(&out)?;
            let (rel, identical) = match &fp32 {
                Some(r) => (diff(&host, r).rel_l2, host == *r),
                None => (0.0, true),
            };
            results.insert(
                format!("{math:?}"),
                json!({"seconds": secs, "rel_l2_vs_f32": rel, "bit_identical_to_f32": identical}),
            );
            if fp32.is_none() {
                fp32 = Some(host);
            }
        }
        // True bf16 buffers: weights and activations stored as bfloat16.
        let to_bf =
            |v: &[f32]| -> Vec<half::bf16> { v.iter().map(|&x| half::bf16::from_f32(x)).collect() };
        let (xb, wb): (CudaSlice<half::bf16>, CudaSlice<half::bf16>) = (
            d.stream.memcpy_stod(&to_bf(&x))?,
            d.stream.memcpy_stod(&to_bf(&w))?,
        );
        let mut ob = unsafe { d.stream.alloc::<half::bf16>(m * n) }?;
        let secs = time(|| Ok(device::matmul_linear_wt_bf16(&xb, &wb, &mut ob, m, k, n)?))?;
        let host: Vec<f32> = d
            .stream
            .memcpy_dtov(&ob)?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let rel = diff(&host, fp32.as_ref().unwrap()).rel_l2;
        results.insert(
            "bf16_buffers".into(),
            json!({"seconds": secs, "rel_l2_vs_f32": rel}),
        );
        report.note(
            format!("{name}_{m}x{k}x{n}"),
            serde_json::Value::Object(results),
        );
    }
    Ok(())
}
