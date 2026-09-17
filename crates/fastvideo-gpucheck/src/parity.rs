//! Real 1.3B weights: GPU path vs cudarc's CPU path on a small case
//! (128×128, 5 frames → 2 latent frames × 8×8 = 128 DiT tokens), so the CPU
//! reference stays affordable on the test box.

use std::path::Path;
use std::time::Instant;

use fastvideo_cudarc::{CudaTensor, GenerateConfig, LoadParts, WanPipeline};
use serde_json::json;

use crate::mode::{limits, Mode};
use crate::rand_weights::randn;
use crate::reference::RefIo;
use crate::report::{Report, StageResult};

fn host(t: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.host_cow()?.into_owned())
}

pub fn run(report: &mut Report, refs: &mut RefIo, weights: &Path, device: &str, mode: Mode) -> StageResult<()> {
    let lim = limits(mode);
    report.set("limits", lim);
    report.set("device", crate::gpu::init(device)?);

    let timer = Instant::now();
    let pipe = WanPipeline::load_with(weights, "wan_t2v_1_3b", LoadParts { text_encoder: false })?;
    report.note("load", json!({"seconds": timer.elapsed().as_secs_f64()}));

    let lat_shape = vec![1usize, 16, 2, 16, 16];
    let n_lat: usize = lat_shape.iter().product();
    // Random stand-in for UMT5 output: numerical parity doesn't need real text.
    let embeds = CudaTensor::from_vec(
        [randn(101, 512 * 4096, 0.2), randn(102, 512 * 4096, 0.2)].concat(),
        vec![2, 512, 4096],
    )?;

    let timer = Instant::now();
    let y = pipe.transformer().forward_ctx(
        &CudaTensor::from_vec(randn(103, n_lat, 1.0), lat_shape.clone())?,
        &CudaTensor::from_vec(vec![999.0], vec![1])?,
        &embeds.narrow(0, 1, 1)?,
        None,
    )?;
    let y = host(&y)?;
    report.note("dit_forward_seconds", json!({"seconds": timer.elapsed().as_secs_f64()}));
    refs.output(report, "dit_forward_t999", &lat_shape, y, lim.forward, false)?;

    let timer = Instant::now();
    let video = pipe.decode_latents(&CudaTensor::from_vec(randn(104, n_lat, 1.0), lat_shape.clone())?)?;
    let shape = video.shape.clone();
    let video = host(&video)?;
    report.note("vae_decode_seconds", json!({"seconds": timer.elapsed().as_secs_f64()}));
    refs.output(report, "vae_decode", &shape, video, lim.forward, true)?;

    let cfg = GenerateConfig {
        height: 128,
        width: 128,
        num_frames: 5,
        num_inference_steps: 2,
        guidance_scale: 5.0,
        flow_shift: 3.0,
        seed: 105,
        ..GenerateConfig::default()
    };
    let timer = Instant::now();
    let out = pipe.denoise(&cfg, pipe.initial_latents(&cfg)?, &embeds, None)?;
    report.note("unipc_seconds", json!({"seconds": timer.elapsed().as_secs_f64()}));
    refs.output(report, "unipc_2step_cfg5", &out.shape.clone(), host(&out)?, lim.denoise, false)?;
    Ok(())
}
