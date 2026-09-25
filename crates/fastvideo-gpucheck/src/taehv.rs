//! Our TAEHV port against madebyollin's own implementation.
//!
//! `scripts/gpu/taehv_oracle.py` runs the reference on a fixed latent and saves
//! both sides of the call; this replays the identical input through us.
//!
//! The port was written by reading `taehv.py`, so checking it against that same
//! file is the only comparison that means anything — every other stage we have
//! would compare our reading of TAEHV against our reading of TAEHV.

use std::path::Path;

use fastvideo_cudarc::wan::taehv::TaeHv;
use fastvideo_cudarc::wan::weights::WeightMap;
use fastvideo_cudarc::CudaTensor;
use serde_json::json;

use crate::metrics::diff;
use crate::model::measure;
use crate::report::{Report, StageResult};
use crate::st;

pub fn run(
    report: &mut Report,
    weights: &Path,
    oracle: &Path,
    device: &str,
    max_rel: f64,
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    report.set("gates", json!({ "rel_l2": max_rel }));

    let mut orc = st::load(oracle)?;
    let latent = st::take(&mut orc, "latent", oracle)?;
    let want = st::take(&mut orc, "video", oracle)?;
    let [t, c, h, w] = latent.shape[..] else {
        return Err(crate::report::StageError::Check(format!(
            "oracle latent should be [T, C, H, W], got {:?}",
            latent.shape
        )));
    };
    report.set(
        "shapes",
        json!({ "latent": latent.shape, "video": want.shape }),
    );

    // The oracle stores frames first; our decode takes [N, C, T, H, W].
    let z = CudaTensor::from_vec(latent.data.clone(), vec![t, c, h, w])?
        .permute(&[1, 0, 2, 3])?
        .reshape(vec![1, c, t, h, w])?
        .to_device()?;

    // The oracle drops taew2_1.safetensors in this directory alongside taehv.py.
    let map = WeightMap::from_dir(weights)?;
    let (tae, load_s) = measure(report, "load", || Ok(TaeHv::load(&map)?))?;
    report.note("load", json!({ "seconds": load_s }));

    let (got, decode_s) = measure(report, "decode", || Ok(tae.decode(&z)?))?;
    report.note("decode", json!({ "seconds": decode_s }));

    // Ours is [1, 3, F, H, W] in [-1, 1]; the reference is [F, 3, H, W] in
    // [0, 1]. Convert ours rather than the reference, so the thing being judged
    // is the thing that moves.
    let f = got.shape[2];
    let got = got
        .reshape(vec![3, f, got.shape[3], got.shape[4]])?
        .permute(&[1, 0, 2, 3])?
        .add_scalar(1.0)
        .mul_scalar(0.5);
    let got_host = got.host_cow()?.into_owned();

    if got.shape != want.shape {
        report.check(
            "video_shape",
            false,
            json!({ "ours": got.shape, "reference": want.shape }),
            json!({ "equal": true }),
        )?;
        return Ok(());
    }
    report.check(
        "video_shape",
        true,
        json!({ "shape": got.shape }),
        json!({}),
    )?;

    let d = diff(&got_host, &want.data);
    report.set(
        "metrics",
        json!({ "video": d.to_json(), "seconds": decode_s }),
    );
    report.check(
        "video",
        d.within(max_rel),
        d.to_json(),
        json!({ "rel_l2": max_rel }),
    )?;
    Ok(())
}

// ---- device path vs the plain-Rust transcription ----------------------------

use fastvideo_cudarc::wan::taehv::TaeArch;
use fastvideo_cudarc::wan::taehv_ref as reference;

fn parse_arch(arch: &str) -> anyhow::Result<TaeArch> {
    match arch {
        "wan" | "taew2_1" => Ok(TaeArch::Wan),
        "h3" | "taeh3" => Ok(TaeArch::H3),
        "ltx" | "ltx-wide" | "taeltx2_3_wide" => Ok(TaeArch::LtxWide),
        other => Err(anyhow::anyhow!("--arch {other}: expected wan, h3 or ltx")),
    }
}

/// Uniform weights scaled by `sqrt(3 / fan_in)`, deterministic per key, so
/// real-width stacks keep unit-scale activations and a wrong index is a large
/// error rather than a saturated one.
fn scaled_weights(key: &str, shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
    let amp = if key.ends_with(".bias") {
        0.1
    } else {
        (3.0 / fan_in as f32).sqrt()
    };
    let seed = key.bytes().fold(0x51f1u64, |a, b| {
        a.wrapping_mul(131).wrapping_add(u64::from(b))
    });
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_mul(6364136223846793005)
                >> 40;
            ((x % 20001) as f32 / 10000.0 - 1.0) * amp
        })
        .collect()
}

/// Real weights when given (a file, or a directory holding the arch's file),
/// else [`scaled_weights`] at the real shapes.
fn weight_map(arch: TaeArch, weights: Option<&Path>) -> anyhow::Result<WeightMap> {
    Ok(match weights {
        Some(p) if p.is_file() => WeightMap::open_files(&[p.to_path_buf()])?,
        Some(p) if p.join(arch.file_name()).is_file() => {
            WeightMap::open_files(&[p.join(arch.file_name())])?
        }
        Some(p) => WeightMap::from_dir(p)?,
        None => WeightMap::generated(scaled_weights),
    })
}

/// Our `[1, 3, F, H, W]` in [-1, 1] → the reference's `[F, 3, H, W]` in [0, 1].
fn frames01(video: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    let [_, c, f, h, w] = video.shape[..] else {
        anyhow::bail!("video shape {:?}", video.shape);
    };
    let x = video
        .reshape(vec![c, f, h, w])?
        .permute(&[1, 0, 2, 3])?
        .add_scalar(1.0)
        .mul_scalar(0.5);
    Ok(x.host_cow()?.into_owned())
}

/// `[C, T, H, W]` → `[T, C, H, W]` on the host.
fn frames_first(data: &[f32], c: usize, t: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for ci in 0..c {
        for ti in 0..t {
            let src = (ci * t + ti) * hw;
            let dst = (ti * c + ci) * hw;
            out[dst..dst + hw].copy_from_slice(&data[src..src + hw]);
        }
    }
    out
}

/// `fv-gpucheck taehv-device`: the device decode (and, for the LTX wide
/// checkpoint, encode) against [`reference`], a plain transcription of
/// `taehv.py`'s parallel path. The port's own host path cannot run on a GPU
/// box (host compute is refused there); the unit tests tie it to the same
/// transcription, so this closes device == host == reference.
pub fn run_device(
    report: &mut Report,
    arch: &str,
    weights: Option<&Path>,
    device: &str,
    max_rel: f64,
) -> StageResult<()> {
    let arch = parse_arch(arch)?;
    report.set("device", crate::gpu::init(device)?);
    report.set("arch", format!("{arch:?}"));
    report.set("gates", json!({ "rel_l2": max_rel }));
    report.set(
        "weights",
        weights.map_or("generated".to_string(), |p| p.display().to_string()),
    );
    let map = weight_map(arch, weights)?;
    // The reference reads the same tensors as host f32.
    let ws = |key: &str, shape: &[usize]| -> Vec<f32> {
        fastvideo_cudarc::wan::weights::cuda_tensor_shaped(&map, key, shape)
            .and_then(|t| Ok(t.host_cow()?.into_owned()))
            .unwrap_or_else(|e| panic!("reference weight {key}: {e}"))
    };
    let (tae, load_s) = measure(report, "load", || {
        Ok(fastvideo_cudarc::wan::taehv::TaeHv::load_arch(&map, arch)?)
    })?;
    report.note("load", json!({ "seconds": load_s }));

    // Small grids at the real channel widths: every block runs, the host
    // reference stays within seconds.
    let c = arch.latent_channels();
    let [t, h, w] = match arch {
        TaeArch::Wan => [5usize, 4, 4],
        TaeArch::H3 => [7, 2, 3],
        TaeArch::LtxWide => [3, 2, 2],
    };
    let zdata = crate::rand_weights::randn(0x7ae, c * t * h * w, 1.0);
    let z = CudaTensor::from_vec(zdata.clone(), vec![1, c, t, h, w])?.to_device()?;
    let zr = frames_first(&zdata, c, t, h * w);
    let want = reference::decode(arch, &ws, &reference::Act::new(zr, [t, c, h, w]));
    report.set(
        "shapes",
        json!({ "latent": [1, c, t, h, w], "video": want.shape }),
    );

    let mut outputs = Vec::new();
    for chunk in [1usize, t] {
        let (got, secs) = measure(report, &format!("decode_chunk{chunk}"), || {
            Ok(tae.decode_streaming_chunked(&z, chunk, &mut |_, _| Ok(()))?)
        })?;
        let [_, _, f, oh, ow] = got.shape[..] else {
            return Err(anyhow::anyhow!("decode shape {:?}", got.shape).into());
        };
        report.check(
            format!("decode_shape_chunk{chunk}"),
            [f, 3, oh, ow] == want.shape,
            json!({ "ours": got.shape }),
            json!({ "reference": want.shape }),
        )?;
        let got = frames01(&got)?;
        let d = diff(&got, &want.data);
        report.check(
            format!("decode_chunk{chunk}"),
            d.within(max_rel),
            json!({ "diff": d.to_json(), "seconds": secs }),
            json!({ "rel_l2": max_rel }),
        )?;
        outputs.push(got);
    }
    // Chunking only regroups the same kernels: the seam carry must make the
    // two (all but) identical.
    let seam = diff(&outputs[0], &outputs[1]);
    report.check(
        "decode_chunk_seam",
        seam.within(1e-5),
        seam.to_json(),
        json!({ "rel_l2": 1e-5 }),
    )?;

    if arch.has_encoder() {
        let s = arch.spatial_scale();
        let (f, ph, pw) = (17usize, 2 * s, 3 * s);
        let pix = crate::rand_weights::randn(0xe1c, 3 * f * ph * pw, 0.6);
        let video = CudaTensor::from_vec(pix.clone(), vec![1, 3, f, ph, pw])?.to_device()?;
        let pr: Vec<f32> = frames_first(&pix, 3, f, ph * pw)
            .into_iter()
            .map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0))
            .collect();
        let want = reference::encode(arch, &ws, &reference::Act::new(pr, [f, 3, ph, pw]));
        for chunk in [arch.t_downscale(), f + arch.t_downscale()] {
            let (got, secs) = measure(report, &format!("encode_chunk{chunk}"), || {
                Ok(tae.encode_chunked(&video, chunk)?)
            })?;
            let [_, lc, lt, lh, lw] = got.shape[..] else {
                return Err(anyhow::anyhow!("encode shape {:?}", got.shape).into());
            };
            report.check(
                format!("encode_shape_chunk{chunk}"),
                [lt, lc, lh, lw] == want.shape,
                json!({ "ours": got.shape }),
                json!({ "reference": want.shape }),
            )?;
            let got = frames_first(&got.host_cow()?, lc, lt, lh * lw);
            let d = diff(&got, &want.data);
            report.check(
                format!("encode_chunk{chunk}"),
                d.within(max_rel),
                json!({ "diff": d.to_json(), "seconds": secs }),
                json!({ "rel_l2": max_rel }),
            )?;
        }
    }
    Ok(())
}
