//! `fv-gpucheck wan oracle`: Wan 2.2 TI2V-5B module parity against the
//! Diffusers reference dump written by `scripts/gpu/upstream/oracle_wan22.py`
//! (docs/oracle.md, "Wan 2.2 TI2V-5B").
//!
//! The reference's inputs are injected as they are: the VAE encodes its video
//! (`vae_enc_in` → `vae_enc_out`, the posterior mean), decodes its latents
//! (`vae_dec_in`, VAE space → `vae_dec_out`), and the DiT runs one forward per
//! case (`t2v_`: one timestep; `i2v_`: frame 0 at timestep 0, the TI2V
//! `expand_timesteps` input) on `<case>_dit_latents`, `<case>_dit_encoder`
//! and `<case>_dit_timestep[_frames]`, dumping the patch embedding, the time
//! projection and every block's output (every 64th row) under the same names.
//! Everything is written to `--out` for `compare-dumps`; the decode's PSNR and
//! max-abs against the reference are checked here.

use std::path::Path;

use fastvideo_cudarc::wan::dump;
use fastvideo_cudarc::wan::weights::WeightMap;
use fastvideo_cudarc::wan::{AutoencoderKlWan, CudaTensor, WanTransformer3D};
use fastvideo_models::wan::{WanVaeConfig, WanVideoArchConfig};
use serde_json::json;

use crate::report::{Report, StageResult};

pub struct Args<'a> {
    pub weights: &'a Path,
    pub reference: &'a Path,
    pub out: &'a Path,
    pub preset: &'a str,
    pub taehv: Option<&'a Path>,
    pub min_psnr: f64,
    pub skip_dit: bool,
    pub device: &'a str,
}

fn read(dir: &Path, name: &str) -> anyhow::Result<Option<CudaTensor>> {
    if !dir.join(format!("{name}.f32")).is_file() {
        return Ok(None);
    }
    let (shape, data) = dump::read_raw(&dir.join(name))?;
    Ok(Some(CudaTensor::from_vec(data, shape)?))
}

fn write(dir: &Path, name: &str, t: &CudaTensor) -> anyhow::Result<Vec<f32>> {
    let host = t.host_cow()?.into_owned();
    dump::write_raw(&dir.join(name), &t.shape, &host)?;
    Ok(host)
}

fn sync() -> anyhow::Result<()> {
    fastvideo_cudarc::wan::device::synchronize()?;
    Ok(())
}

pub fn run(report: &mut Report, a: &Args<'_>) -> StageResult<()> {
    std::fs::create_dir_all(a.out).map_err(anyhow::Error::from)?;
    // The DiT's block-output hooks (`dump::blocks`) write into --out.
    std::env::set_var("FASTVIDEO_DUMP_DIR", a.out);
    report.set("device", crate::gpu::init(a.device)?);
    let r = a.reference;
    let mut checks: Vec<(String, bool, serde_json::Value, serde_json::Value)> = Vec::new();

    let vae_dir = a.weights.join("vae");
    let vae_cfg = WanVaeConfig::from_dir(&vae_dir).map_err(anyhow::Error::msg)?;
    report.set(
        "vae_config",
        json!({"z_dim": vae_cfg.z_dim, "patch": vae_cfg.patch_size, "residual": vae_cfg.is_residual,
               "base": vae_cfg.base_dim, "decoder_base": vae_cfg.decoder_base_dim}),
    );
    let vae = AutoencoderKlWan::load(vae_cfg, &WeightMap::from_dir(&vae_dir)?)
        .map_err(anyhow::Error::from)?;

    if let Some(video) = read(r, "vae_enc_in")? {
        let t = std::time::Instant::now();
        let mu = vae.encode_video(&video.to_device()?)?;
        sync()?;
        let secs = t.elapsed().as_secs_f64();
        let ours = write(a.out, "vae_enc_out", &mu)?;
        if let Some(want) = read(r, "vae_enc_out")? {
            let want = want.host_cow()?.into_owned();
            let d = crate::metrics::diff(&ours, &want);
            let mut v = d.to_json();
            v["seconds"] = json!(secs);
            v["shape"] = json!(mu.shape);
            report.note("vae_encode", v.clone());
            checks.push((
                "vae_encode_rel_l2".into(),
                d.within(1e-2),
                v,
                json!({"max_rel_l2": 1e-2}),
            ));
        }
    }

    if let Some(z) = read(r, "vae_dec_in")? {
        let z = z.to_device()?;
        let t = std::time::Instant::now();
        let video = vae.decode(&z)?;
        sync()?;
        let secs = t.elapsed().as_secs_f64();
        let ours = write(a.out, "vae_dec_out", &video)?;
        if let Some(want) = read(r, "vae_dec_out")? {
            let want = want.host_cow()?.into_owned();
            let d = crate::metrics::diff(&ours, &want);
            let psnr = crate::metrics::psnr(&ours, &want, 2.0);
            let mut v = d.to_json();
            v["psnr_db"] = json!(psnr);
            v["seconds"] = json!(secs);
            v["shape"] = json!(video.shape);
            report.note("vae_decode", v.clone());
            checks.push((
                "vae_decode_psnr".into(),
                d.non_finite == 0 && psnr >= a.min_psnr,
                v,
                json!({"min_psnr_db": a.min_psnr}),
            ));
            // TAEHV (taew2_2) on the same latents: it reads DiT space.
            if let Some(path) = a.taehv {
                let arch = fastvideo_cudarc::wan::pipeline::taehv_arch(z.shape[1]);
                let tae = fastvideo_cudarc::wan::taehv::TaeHv::load_from_path(path, arch)?;
                let t = std::time::Instant::now();
                let tv = tae.decode(&vae.normalize_latents(&z)?)?;
                sync()?;
                let secs = t.elapsed().as_secs_f64();
                let tv = write(a.out, "taehv_dec_out", &tv)?;
                let d = crate::metrics::diff(&tv, &want);
                let mut v = d.to_json();
                v["psnr_db"] = json!(crate::metrics::psnr(&tv, &want, 2.0));
                v["seconds"] = json!(secs);
                report.note("taehv_vs_vae_decode", v);
            }
        }
    }

    if !a.skip_dit {
        fastvideo_cudarc::wan::tensor::default_bf16_activations();
        let cfg = WanVideoArchConfig::from_preset(a.preset);
        let dit =
            WanTransformer3D::load(cfg, &WeightMap::from_dir(&a.weights.join("transformer"))?)
                .map_err(anyhow::Error::from)?;
        for case in ["t2v", "i2v"] {
            let Some(latents) = read(r, &format!("{case}_dit_latents"))? else {
                continue;
            };
            let timestep = match read(r, &format!("{case}_dit_timestep_frames"))? {
                Some(t) => t,
                None => read(r, &format!("{case}_dit_timestep"))?
                    .ok_or_else(|| anyhow::anyhow!("{case}: no timestep in the reference"))?,
            };
            let encoder = read(r, &format!("{case}_dit_encoder"))?
                .ok_or_else(|| anyhow::anyhow!("{case}: no encoder states in the reference"))?;
            dump::set_prefix(&format!("{case}_"));
            dump::set_blocks(true);
            let t = std::time::Instant::now();
            let out = dit.forward(&latents.to_device()?, &timestep, &encoder.to_device()?);
            dump::set_blocks(false);
            dump::set_prefix("");
            let out = out?;
            sync()?;
            let secs = t.elapsed().as_secs_f64();
            let ours = write(a.out, &format!("{case}_dit_out"), &out)?;
            if let Some(want) = read(r, &format!("{case}_dit_out"))? {
                let want = want.host_cow()?.into_owned();
                let d = crate::metrics::diff(&ours, &want);
                let mut v = d.to_json();
                v["seconds"] = json!(secs);
                report.note(format!("{case}_dit_out"), v.clone());
                checks.push((
                    format!("{case}_dit_out_rel_l2"),
                    d.within(5e-2),
                    v,
                    json!({"max_rel_l2": 5e-2}),
                ));
            }
        }
    }

    for (name, ok, v, lim) in checks {
        report.check(name, ok, v, lim)?;
    }
    Ok(())
}
