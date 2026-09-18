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

pub fn run(report: &mut Report, weights: &Path, oracle: &Path, device: &str, max_rel: f64) -> StageResult<()> {
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
    report.set("shapes", json!({ "latent": latent.shape, "video": want.shape }));

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
    report.check("video_shape", true, json!({ "shape": got.shape }), json!({}))?;

    let d = diff(&got_host, &want.data);
    report.set("metrics", json!({ "video": d.to_json(), "seconds": decode_s }));
    report.check("video", d.within(max_rel), d.to_json(), json!({ "rel_l2": max_rel }))?;
    Ok(())
}
