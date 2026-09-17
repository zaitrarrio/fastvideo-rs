//! Random-weight model checks (no downloads): UMT5, DiT, feat-cache VAE and
//! both samplers on small configs that keep the real architecture (head_dim
//! 128 so the flash kernel is eligible, multi-layer stacks, a temporal
//! upsample and mid attention in the VAE).
//!
//! Outputs are dumped from cudarc's CPU path and compared on the GPU (see
//! [`crate::reference`]). The GPU run adds self-consistency checks that need no
//! reference: batched rows must equal single-row forwards, and a repeated
//! forward must be reproducible.

use fastvideo_cudarc::wan::umt5::Umt5Encoder;
use fastvideo_cudarc::wan::{AutoencoderKlWan, WanTransformer3D};
use fastvideo_cudarc::{CudaTensor, GenerateConfig, WanPipeline};
use fastvideo_models::wan::{Umt5Config, WanVaeConfig, WanVideoArchConfig};
use serde_json::json;

use crate::metrics::diff;
use crate::mode::{limits, Mode};
use crate::rand_weights::{randn, random_map};
use crate::reference::{Out, RefIo};
use crate::report::{Report, StageResult};

pub fn dit_config() -> WanVideoArchConfig {
    WanVideoArchConfig {
        patch_size: [1, 2, 2],
        text_len: 16,
        num_attention_heads: 2,
        attention_head_dim: 128,
        in_channels: 16,
        out_channels: 16,
        text_dim: 64,
        freq_dim: 64,
        ffn_dim: 512,
        num_layers: 2,
        ..WanVideoArchConfig::wan_t2v_1_3b()
    }
}

pub fn umt5_config() -> Umt5Config {
    Umt5Config {
        vocab_size: 512,
        d_model: 64,
        d_kv: 32,
        d_ff: 128,
        num_heads: 2,
        num_layers: 2,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        dropout: 0.0,
        eps: 1e-6,
    }
}

pub fn vae_config() -> WanVaeConfig {
    WanVaeConfig {
        base_dim: 32,
        z_dim: 16,
        dim_mult: vec![1, 2, 2],
        num_res_blocks: 1,
        temporal_upsample: vec![true, false],
        load_encoder: false,
    }
}

fn host(t: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.host_cow()?.into_owned())
}

/// Run `f` inside a synchronized [`crate::gpu::Span`]; records the transfers
/// it caused as `transfers/<name>` and returns `(value, seconds)`.
pub fn measure<T>(report: &mut Report, name: &str, f: impl FnOnce() -> anyhow::Result<T>) -> StageResult<(T, f64)> {
    let span = crate::gpu::span()?;
    let value = f()?;
    let (secs, transfers) = span.finish()?;
    report.note(format!("transfers/{name}"), transfers);
    Ok((value, secs))
}

pub fn run(report: &mut Report, refs: &mut RefIo, device: &str, mode: Mode, seed: u64) -> StageResult<()> {
    let lim = limits(mode);
    report.set("limits", lim);
    report.set("device", crate::gpu::init(device)?);
    let gpu = crate::gpu::on_gpu(device);

    let text = Umt5Encoder::load(umt5_config(), &random_map(seed ^ 0x7e57))?;
    let dit = WanTransformer3D::load(dit_config(), &random_map(seed ^ 0xd17))?;
    let vae = AutoencoderKlWan::load(vae_config(), &random_map(seed ^ 0xfae))?;

    // UMT5 on 11 ids (odd length exercises the relative-position buckets).
    let ids: Vec<u32> = (0..11u32).map(|i| (i * 37 + 5) % 512).collect();
    if gpu {
        // Warm-up: NVRTC/cuBLAS/cuDNN setup and allocator growth stay out of timings.
        text.forward(&ids, 1, ids.len())?;
    }
    let (out, s) = measure(report, "umt5_forward", || Ok(text.forward(&ids, 1, ids.len())?))?;
    let data = host(&out)?;
    refs.output(report, "umt5_forward", &out.shape, data, lim.forward, s, Out::Tensor)?;

    // DiT forward, batch 1: [1,16,3,6,8] → 36 tokens.
    let lat_shape = vec![1usize, 16, 3, 6, 8];
    let lat = randn(seed + 1, lat_shape.iter().product(), 1.0);
    let enc = randn(seed + 2, 16 * 64, 0.5);
    let enc2 = randn(seed + 3, 16 * 64, 0.5);
    let t = 731.0f32;
    let fwd = |l: &[f32], e: &[f32], b: usize| -> anyhow::Result<CudaTensor> {
        let mut shape = lat_shape.clone();
        shape[0] = b;
        Ok(dit.forward_ctx(
            &CudaTensor::from_vec(l.to_vec(), shape)?,
            &CudaTensor::from_vec(vec![t; b], vec![b])?,
            &CudaTensor::from_vec(e.to_vec(), vec![b, 16, 64])?,
            None,
        )?)
    };
    if gpu {
        fwd(&lat, &enc, 1)?;
    }
    let (y1, s) = measure(report, "dit_forward_b1", || fwd(&lat, &enc, 1))?;
    let y1 = host(&y1)?;
    refs.output(report, "dit_forward_b1", &lat_shape, y1.clone(), lim.forward, s, Out::Tensor)?;

    // Batch 2 with different text per row (the batched-CFG shape).
    let lat_b2 = [lat.as_slice(), lat.as_slice()].concat();
    let enc_b2 = [enc2.as_slice(), enc.as_slice()].concat();
    let (y2, y2_s) = measure(report, "dit_forward_b2", || fwd(&lat_b2, &enc_b2, 2))?;
    let y2 = host(&y2)?;
    let mut b2_shape = lat_shape.clone();
    b2_shape[0] = 2;
    refs.output(report, "dit_forward_b2", &b2_shape, y2.clone(), lim.forward, y2_s, Out::Tensor)?;
    if gpu {
        let half = y2.len() / 2;
        let d = diff(&y2[half..], &y1);
        report.check("gpu/b2_row_equals_b1", d.within(lim.forward), d.to_json(), json!({"rel_l2": lim.forward}))?;
        let again = host(&fwd(&lat, &enc, 1)?)?;
        let d = diff(&again, &y1);
        report.check("gpu/forward_reproducible", d.within(1e-6), d.to_json(), json!({"rel_l2": 1e-6}))?;
    }

    // VAE decode: 3 latent frames → 5 RGB frames at 4x spatial.
    let z_shape = vec![1usize, 16, 3, 6, 8];
    let z = CudaTensor::from_vec(randn(seed + 4, z_shape.iter().product(), 1.0), z_shape)?;
    if gpu {
        vae.decode(&z)?;
    }
    let (video, s) = measure(report, "vae_decode", || Ok(vae.decode(&z)?))?;
    let data = host(&video)?;
    refs.output(report, "vae_decode", &video.shape.clone(), data, lim.forward, s, Out::Video)?;

    // Samplers through the production WanPipeline::denoise path.
    let pipe = WanPipeline::from_parts(None, dit, vae);
    let embeds = CudaTensor::from_vec(
        [randn(seed + 5, 16 * 64, 0.5), randn(seed + 6, 16 * 64, 0.5)].concat(),
        vec![2, 16, 64],
    )?;
    let cfg = GenerateConfig {
        height: 48,
        width: 64,
        num_frames: 9,
        num_inference_steps: 4,
        guidance_scale: 3.0,
        flow_shift: 3.0,
        seed: seed + 7,
        ..GenerateConfig::default()
    };
    let noise = pipe.initial_latents(&cfg)?;
    let (out, denoise_s) = measure(report, "unipc_4step_cfg3", || Ok(pipe.denoise(&cfg, noise.clone(), &embeds, None)?))?;
    let data = host(&out)?;
    refs.output(report, "unipc_4step_cfg3", &out.shape.clone(), data, lim.denoise, denoise_s, Out::Tensor)?;
    decode_video(report, refs, &pipe, &out, "unipc_4step_cfg3_video", lim.denoise, denoise_s)?;

    let dmd_cfg = GenerateConfig {
        is_dmd: true,
        dmd_steps: Some(vec![1000, 757, 522]),
        flow_shift: 8.0,
        guidance_scale: 1.0,
        ..cfg
    };
    let (out, denoise_s) = measure(report, "dmd_3step", || Ok(pipe.denoise(&dmd_cfg, noise, &embeds, None)?))?;
    let data = host(&out)?;
    // DMD is x0 + re-noise with host-seeded noise: no sampler-order gap, forward limit.
    refs.output(report, "dmd_3step", &out.shape.clone(), data, lim.forward, denoise_s, Out::Tensor)?;
    decode_video(report, refs, &pipe, &out, "dmd_3step_video", lim.forward, denoise_s)?;
    Ok(())
}

/// Decode sampler latents through the pipeline VAE into a compared, saved
/// video. The recorded time is end-to-end generation: denoise + decode.
pub fn decode_video(
    report: &mut Report,
    refs: &mut RefIo,
    pipe: &WanPipeline,
    latents: &CudaTensor,
    name: &str,
    rel_limit: f64,
    denoise_s: f64,
) -> StageResult<()> {
    let (video, decode_s) = measure(report, &format!("{name}_decode"), || Ok(pipe.decode_latents(latents)?))?;
    let data = host(&video)?;
    report.note(
        format!("timing/{name}"),
        json!({"denoise_seconds": denoise_s, "decode_seconds": decode_s, "total_seconds": denoise_s + decode_s}),
    );
    refs.output(report, name, &video.shape.clone(), data, rel_limit, denoise_s + decode_s, Out::Video)
}
