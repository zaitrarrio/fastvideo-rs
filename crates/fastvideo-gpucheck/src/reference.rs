//! GPU-vs-CPU-path references produced on the test box itself.
//!
//! The same `fv-gpucheck` binary runs a stage twice in separate processes
//! (cudarc caches its device and precision flags per process):
//!
//! 1. `--device cpu --dump DIR`: cudarc's host path, exact mode → outputs saved.
//! 2. `--device cuda --reference DIR`: the GPU path recomputes the same outputs
//!    from the same seeded weights/inputs and every output is compared.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use serde_json::json;

use crate::metrics::{diff, jf, psnr};
use crate::report::{Report, StageResult};
use crate::st::{self, F32Tensor};

pub enum RefIo {
    Dump {
        path: PathBuf,
        outputs: Vec<(String, F32Tensor)>,
    },
    Compare {
        path: PathBuf,
        reference: HashMap<String, F32Tensor>,
    },
}

impl RefIo {
    pub fn new(dump: Option<&Path>, reference: Option<&Path>, stage: &str) -> anyhow::Result<Self> {
        match (dump, reference) {
            (Some(dir), None) => Ok(RefIo::Dump {
                path: dir.join(format!("{stage}.safetensors")),
                outputs: Vec::new(),
            }),
            (None, Some(dir)) => {
                let path = dir.join(format!("{stage}.safetensors"));
                let meta: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(path.with_extension("json"))
                        .with_context(|| format!("no reference manifest for {} (run with --dump first)", path.display()))?,
                )?;
                if meta["device"] != "cpu" || meta["mode"] != "exact" {
                    bail!("{}: reference must come from --device cpu --mode exact, got {meta}", path.display());
                }
                if let (Some(want), Ok(have)) = (meta["git_sha"].as_str(), std::env::var("FV_GIT_SHA")) {
                    if want != have {
                        bail!("reference was dumped at {want}, this binary is {have}; re-dump");
                    }
                }
                Ok(RefIo::Compare {
                    reference: st::load(&path)?,
                    path,
                })
            }
            _ => bail!("pass exactly one of --dump <dir> (CPU path) or --reference <dir> (GPU run)"),
        }
    }

    pub fn is_dump(&self) -> bool {
        matches!(self, RefIo::Dump { .. })
    }

    /// Record (dump) or compare one output. `rel_limit` bounds relative L2;
    /// `video` adds PSNR to the report for [-1,1] video tensors.
    pub fn output(
        &mut self,
        report: &mut Report,
        name: &str,
        shape: &[usize],
        data: Vec<f32>,
        rel_limit: f64,
        video: bool,
    ) -> StageResult<()> {
        let bad = crate::metrics::non_finite(&data);
        match self {
            RefIo::Dump { outputs, .. } => {
                report.check(
                    format!("{name}/finite"),
                    bad == 0,
                    json!({"non_finite": bad, "numel": data.len()}),
                    json!({"non_finite": 0}),
                )?;
                outputs.push((name.to_string(), F32Tensor::new(shape.to_vec(), data)?));
                Ok(())
            }
            RefIo::Compare { reference, path } => {
                let want = reference
                    .get(name)
                    .with_context(|| format!("{} has no output `{name}`", path.display()))?;
                if want.shape != shape {
                    return report.check(
                        format!("{name}/shape"),
                        false,
                        json!({"gpu": shape, "cpu_path": want.shape}),
                        json!({}),
                    );
                }
                let d = diff(&data, &want.data);
                let mut values = d.to_json();
                if video {
                    values["psnr_db"] = jf(psnr(&data, &want.data, 2.0));
                }
                report.check(name, d.within(rel_limit), values, json!({"rel_l2": rel_limit}))
            }
        }
    }

    pub fn finish(self, report: &mut Report, mode: crate::mode::Mode, device: &str) -> anyhow::Result<()> {
        if let RefIo::Dump { path, outputs } = self {
            let refs: Vec<(&str, &F32Tensor)> = outputs.iter().map(|(n, t)| (n.as_str(), t)).collect();
            st::save(&path, &refs)?;
            let meta = json!({
                "device": device,
                "mode": mode,
                "git_sha": std::env::var("FV_GIT_SHA").ok(),
                "outputs": outputs.iter().map(|(n, t)| json!({"name": n, "shape": t.shape})).collect::<Vec<_>>(),
            });
            std::fs::write(path.with_extension("json"), serde_json::to_string_pretty(&meta)?)?;
            report.set("reference_written", path);
        }
        Ok(())
    }
}
