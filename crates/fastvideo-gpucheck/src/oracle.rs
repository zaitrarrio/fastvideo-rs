//! Our stack against an *external* reference.
//!
//! Every other stage compares fastvideo-rs to fastvideo-rs — `parity` is GPU
//! vs our own CPU path, `compare` diffs two of our own clip dirs — so a wrong
//! port passes all of them. `scripts/gpu/upstream_oracle.py` runs transformers'
//! UMT5 and diffusers' `WanTransformer3DModel` on the same weights and saves
//! its inputs and outputs; this stage replays them through us.
//!
//! Three comparisons, on byte-identical inputs, so a failure says *where*:
//!
//! | check | ours                            | tells us          |
//! |-------|---------------------------------|-------------------|
//! | text  | our embedding vs the oracle's   | the UMT5 port     |
//! | dit   | our DiT on the oracle embedding | the DiT port      |
//! | e2e   | our DiT on our own embedding    | what a clip gets  |
//!
//! `dit` isolates the transformer because it removes the text encoder from the
//! comparison: both sides get the same conditioning.

use std::path::Path;

use fastvideo_cudarc::{CudaTensor, LoadParts, WanPipeline};
use serde_json::json;

use crate::metrics::diff;
use crate::model::measure;
use crate::report::{Report, StageResult};
use crate::st::{self, F32Tensor};

pub struct OracleGates {
    pub max_text_rel: f64,
    pub max_dit_rel: f64,
    pub max_e2e_rel: f64,
}

fn host(t: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.host_cow()?.into_owned())
}

/// The conditional row of an embeds file, as `[1, text_len, dim]`.
fn cond_row(t: &F32Tensor, path: &Path) -> anyhow::Result<CudaTensor> {
    let [b, len, dim] = t.shape[..] else {
        anyhow::bail!("{}: expected [rows, text_len, dim], got {:?}", path.display(), t.shape);
    };
    // Our embed stage writes [negative, prompt]; the oracle writes the prompt alone.
    let row = if b > 1 { 1 } else { 0 };
    Ok(CudaTensor::from_vec(t.data[row * len * dim..(row + 1) * len * dim].to_vec(), vec![1, len, dim])?)
}

pub fn run(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    embeds: &Path,
    device: &str,
    gates: OracleGates,
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    report.set(
        "gates",
        json!({
            "text_rel": gates.max_text_rel,
            "dit_rel": gates.max_dit_rel,
            "e2e_rel": gates.max_e2e_rel,
        }),
    );

    let mut orc = st::load(oracle)?;
    let o_text = st::take(&mut orc, "text", oracle)?;
    let o_noise = st::take(&mut orc, "noise", oracle)?;
    let o_step = st::take(&mut orc, "timestep", oracle)?;
    let o_dit = st::take(&mut orc, "dit", oracle)?;

    let mut ours = st::load(embeds)?;
    let e_text = st::take(&mut ours, "embeds", embeds)?;

    report.set("shapes", json!({"text": o_text.shape, "noise": o_noise.shape, "dit": o_dit.shape}));
    let cond_oracle = cond_row(&o_text, oracle)?;
    let cond_ours = cond_row(&e_text, embeds)?;

    // 1. The text encoder, before anything consumes it.
    let d_text = diff(&host(&cond_ours)?, &host(&cond_oracle)?);
    let noise = CudaTensor::from_vec(o_noise.data.clone(), o_noise.shape.clone())?;
    let t = CudaTensor::from_vec(o_step.data.clone(), vec![1])?;

    let (pipe, load_s) = measure(report, "load", || {
        Ok(WanPipeline::load_with(weights, "wan_t2v_1_3b", LoadParts { text_encoder: false })?)
    })?;
    report.note("load", json!({"seconds": load_s}));

    let forward = |cond: &CudaTensor| -> anyhow::Result<Vec<f32>> {
        host(&pipe.transformer().forward_ctx(&noise, &t, cond, None)?)
    };
    if crate::gpu::on_gpu(device) {
        forward(&cond_oracle)?; // warm-up: plans and allocator growth stay out of the numbers
    }
    let (y_dit, dit_s) = measure(report, "dit_on_oracle_text", || forward(&cond_oracle))?;
    let (y_e2e, e2e_s) = measure(report, "dit_on_our_text", || forward(&cond_ours))?;

    let d_dit = diff(&y_dit, &o_dit.data);
    let d_e2e = diff(&y_e2e, &o_dit.data);

    // Every metric lands before the first gate can stop the stage.
    report.set(
        "metrics",
        json!({
            "text": d_text.to_json(),
            "dit": d_dit.to_json(),
            "e2e": d_e2e.to_json(),
            "seconds": {"dit": dit_s, "e2e": e2e_s},
        }),
    );
    report.check("text", d_text.within(gates.max_text_rel), d_text.to_json(), json!({"rel_l2": gates.max_text_rel}))?;
    report.check("dit", d_dit.within(gates.max_dit_rel), d_dit.to_json(), json!({"rel_l2": gates.max_dit_rel}))?;
    report.check("e2e", d_e2e.within(gates.max_e2e_rel), d_e2e.to_json(), json!({"rel_l2": gates.max_e2e_rel}))?;
    Ok(())
}
