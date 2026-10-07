//! Cold-start tooling (E12 / E13): raw read throughput of a weight volume,
//! page-cache eviction for cold measurements, and the offline pre-quantizer
//! of the resident FP8 text encoders.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::report::{Report, StageError, StageResult};

fn files_under(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for d in dirs {
        if d.is_file() {
            out.push(d.clone());
        } else if let Ok(mut v) = fastvideo_loader::collect_safetensors(d) {
            out.append(&mut v);
        }
    }
    out
}

/// `io-bench`: parallel `pread` throughput per thread count, and the mapped
/// page-fault path the loaders used before E12, over the same files.
pub fn io_bench(
    report: &mut Report,
    dirs: &[PathBuf],
    threads: &[usize],
    chunk_mb: usize,
    limit_gb: f64,
    evict: bool,
    mmap: bool,
) -> StageResult<()> {
    let files = files_under(dirs);
    if files.is_empty() {
        return Err(StageError::Error(anyhow::anyhow!(
            "no .safetensors under {dirs:?}"
        )));
    }
    let limit = (limit_gb * 1e9) as u64;
    report.set("files", files.len());
    report.set(
        "mem_available_gb",
        fastvideo_loader::prefetch::mem_available().map(|b| b as f64 / 1e9),
    );
    for &t in threads {
        if evict {
            for f in &files {
                let _ = fastvideo_loader::evict_page_cache(f);
            }
        }
        let (bytes, secs) = fastvideo_loader::read_bench(&files, t, chunk_mb << 20, limit);
        report.note(
            format!("pread_t{t}"),
            json!({"threads": t, "chunk_mb": chunk_mb, "gb": bytes as f64 / 1e9, "seconds": secs, "gbps": bytes as f64 / 1e9 / secs.max(1e-9), "evicted": evict}),
        );
    }
    if mmap {
        // The pre-E12 consumer: touch the mapping page by page, in parallel
        // across rayon (as `fill_bf16` does per tensor).
        if evict {
            for f in &files {
                let _ = fastvideo_loader::evict_page_cache(f);
            }
        }
        use rayon::prelude::*;
        let t0 = Instant::now();
        let mut total = 0u64;
        for f in &files {
            if limit > 0 && total >= limit {
                break;
            }
            let file = std::fs::File::open(f).map_err(anyhow::Error::from)?;
            // SAFETY: read-only mapping of a file nobody writes during the bench.
            let map = unsafe { memmap2::Mmap::map(&file) }.map_err(anyhow::Error::from)?;
            let n = if limit > 0 {
                map.len().min((limit - total) as usize)
            } else {
                map.len()
            };
            let sum: u64 = map[..n]
                .par_chunks(1 << 18)
                .map(|c| c.iter().step_by(4096).map(|b| u64::from(*b)).sum::<u64>())
                .sum();
            std::hint::black_box(sum);
            total += n as u64;
        }
        let secs = t0.elapsed().as_secs_f64();
        report.note(
            "mmap_fault",
            json!({"gb": total as f64 / 1e9, "seconds": secs, "gbps": total as f64 / 1e9 / secs.max(1e-9), "evicted": evict}),
        );
    }
    Ok(())
}

/// `evict-cache`: drop the page cache of every file under the paths.
pub fn evict(report: &mut Report, paths: &[PathBuf]) -> StageResult<()> {
    let mut total = 0u64;
    for p in paths {
        total += fastvideo_loader::evict_page_cache(p).map_err(anyhow::Error::from)?;
    }
    report.note("evicted", json!({"gb": total as f64 / 1e9, "paths": paths}));
    Ok(())
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Digest over the captured codes and scales, in load order.
fn capture_digest(c: &fastvideo_cudarc::wan::nn::Fp8Capture) -> String {
    let mut h = Sha256::new();
    for (prefix, codes, scales) in c {
        h.update(prefix.as_bytes());
        h.update((codes.len() as u64).to_le_bytes());
        h.update(codes);
        for s in scales {
            h.update(s.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

/// Which decoder, and how many layers the resident form keeps.
fn family(name: &str) -> anyhow::Result<(fastvideo_cudarc::llm::DecoderConfig, usize)> {
    use fastvideo_cudarc::llm::DecoderConfig;
    Ok(match name {
        "h3" => (
            DecoderConfig::qwen3_vl_32b_text().for_bf16_reference(),
            fastvideo_models::h3::config::H3TextEncoderConfig::fasth3_8step()
                .output_hidden_state_index,
        ),
        "ltx2-gemma4" | "ltx25" => {
            let c = DecoderConfig::gemma4_12b_text().for_bf16_reference();
            let n = c.num_layers();
            (c, n)
        }
        "ltx2-gemma3" | "ltx23" => {
            let c = DecoderConfig::gemma3_12b_text().for_bf16_reference();
            let n = c.num_layers();
            (c, n)
        }
        other => anyhow::bail!("unknown family '{other}' (h3|ltx2-gemma4|ltx2-gemma3)"),
    })
}

/// `quantize-text-encoder`: build `<root>/text_encoder_fp8/` from
/// `<root>/text_encoder/` exactly as the resident FP8 load quantizes it, then
/// (with `verify`) load the tree back and require every code and scale to be
/// byte-identical to the load-time quantization, and every copied tensor to
/// equal its source.
#[cfg(feature = "cuda")]
pub fn quantize_text_encoder(
    report: &mut Report,
    family_name: &str,
    root: &Path,
    out: Option<&Path>,
    hash_sources: bool,
    verify: bool,
    verify_only: bool,
) -> StageResult<()> {
    use fastvideo_cudarc::llm::{prequant, ResidentDecoder, WeightPrecision};
    use fastvideo_cudarc::wan::nn::capture_fp8_rows;
    use fastvideo_cudarc::wan::weights::WeightMap;

    let (cfg, layers) = family(family_name)?;
    let source = root.join("text_encoder");
    let out_dir = out.map_or_else(|| root.join(prequant::DIR_NAME), Path::to_path_buf);
    report.set("family", family_name);
    report.set("source", source.display().to_string());
    report.set("tree", out_dir.display().to_string());
    report.set("layers", layers);

    let load = |map: &WeightMap| {
        capture_fp8_rows(|| {
            let d = ResidentDecoder::load_with(map, &cfg, layers, WeightPrecision::Fp8Rows)?;
            drop(d);
            Ok(())
        })
    };

    let mut reference_digest = None;
    if !verify_only {
        let t0 = Instant::now();
        let hashes: HashMap<String, String> = if hash_sources {
            use rayon::prelude::*;
            prequant::source_files(&source)?
                .par_iter()
                .map(|p| {
                    let name = p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or_default()
                        .to_string();
                    sha256_file(p).map(|h| (name, h))
                })
                .collect::<std::io::Result<_>>()
                .map_err(anyhow::Error::from)?
        } else {
            HashMap::new()
        };
        let hash_s = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let map = WeightMap::open(&source)?;
        if let Some(lazy) = map.lazy() {
            lazy.prefetch(&prequant::kept_keys(lazy, &cfg, layers));
        }
        let ((), captured) = load(&map)?;
        let quantize_s = t0.elapsed().as_secs_f64();
        drop(map);
        let digest = capture_digest(&captured);
        let t0 = Instant::now();
        let build = std::env::var("FV_BUILD_ID")
            .ok()
            .or_else(|| {
                std::fs::read_to_string("/opt/fastvideo-rs/target/release/fv-gpucheck.build-id")
                    .ok()
            })
            .unwrap_or_default();
        let manifest = prequant::write_tree(
            &source,
            &out_dir,
            &cfg,
            layers,
            &captured,
            &hashes,
            build.trim(),
        )?;
        let write_s = t0.elapsed().as_secs_f64();
        report.note(
            "written",
            json!({"linears": captured.len(), "load_time_digest": digest, "hash_s": hash_s, "quantize_load_s": quantize_s, "write_s": write_s, "files": manifest.files, "sources": manifest.source.len()}),
        );
        reference_digest = Some(digest);
        drop(captured);
    }
    if verify || verify_only {
        // Load-time quantization from the source (again, when verify-only).
        let digest = match reference_digest {
            Some(d) => d,
            None => {
                let map = WeightMap::open(&source)?;
                if let Some(lazy) = map.lazy() {
                    lazy.prefetch(&prequant::kept_keys(lazy, &cfg, layers));
                }
                let ((), c) = load(&map)?;
                capture_digest(&c)
            }
        };
        let manifest = prequant::check(&source, &out_dir, layers)
            .map_err(|e| anyhow::anyhow!("tree rejected: {e}"))?;
        let t0 = Instant::now();
        let tree = WeightMap::open(&out_dir)?;
        if let Some(lazy) = tree.lazy() {
            lazy.prefetch(&prequant::kept_keys(lazy, &cfg, layers));
        }
        let ((), c) = load(&tree)?;
        let tree_load_s = t0.elapsed().as_secs_f64();
        let tree_digest = capture_digest(&c);
        drop(c);
        // Every copied tensor equals its source.
        let src = WeightMap::open(&source)?;
        let (sl, tl) = (src.lazy().expect("lazy"), tree.lazy().expect("lazy"));
        let mut copied = 0usize;
        let mut mismatched = Vec::new();
        for k in prequant::kept_keys(sl, &cfg, layers) {
            let scale = format!(
                "{}.{}",
                k.strip_suffix(".weight").unwrap_or(&k),
                fastvideo_cudarc::wan::nn::FP8_ROWS_SCALE_SUFFIX
            );
            if tl.contains(&scale) {
                continue;
            }
            copied += 1;
            let (a, b) = (
                sl.view(&k).map_err(|e| anyhow::anyhow!("{e}"))?,
                tl.view(&k).map_err(|e| anyhow::anyhow!("{e}"))?,
            );
            if a.dtype != b.dtype || a.shape != b.shape || a.bytes != b.bytes {
                mismatched.push(k);
            }
        }
        report.check(
            "fp8_tree_identity",
            tree_digest == digest && mismatched.is_empty(),
            json!({"load_time_digest": digest, "tree_digest": tree_digest, "linears": manifest.quantized_linears, "copied_tensors": copied, "copied_mismatched": mismatched, "tree_load_s": tree_load_s}),
            json!({"digests_equal": true, "copied_mismatched": 0}),
        )?;
    }
    Ok(())
}

/// `quantize-dit`: build the pre-quantized resident DiT tree of a served H3
/// model (`h3-turbo`, `h3-max`, a catalog id) under `<weights>/<tree name>`
/// (fast boot A, `fastvideo_cudarc::h3::dit_tree`).
///
/// 1. Install the model's process plan exactly as fv-serve does and load its
///    pipeline from the checkpoint (`FASTVIDEO_H3_DIT_TREE=off`): adapter
///    merge, AdaLN table, quantization at load as served.
/// 2. Read every resident tensor of the refiner and DiT back (the load-time
///    result, byte for byte), write them to `tmp` (a new directory; never an
///    existing tree), hash the written file and the sources, write the
///    manifest.
/// 3. `verify`: load again from `tmp` (`FASTVIDEO_H3_DIT_TREE=require`) and
///    require the read-back state to equal step 2's tensor for tensor, and
///    the file's sha256 to equal the manifest's.
/// 4. `finalize`: rename `tmp` to the tree name (only when it does not exist).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn quantize_dit(
    report: &mut Report,
    model: &str,
    weights_root: &Path,
    tmp: Option<&Path>,
    hash_sources: bool,
    verify: bool,
    verify_only: bool,
    finalize: bool,
) -> StageResult<()> {
    use fastvideo_cudarc::h3::dit_tree::{self, TreeWriter};
    use fastvideo_cudarc::h3::pipeline::{dit_tree_identity_for, H3Pipeline, I2vEncoderChoice, TextEncoderChoice};
    use fastvideo_engine_service::cuda::{self, h3::H3Model, CudaRecipe, ProcessPlan, WeightLayout};

    let layout = WeightLayout::new(weights_root);
    let cat = cuda::catalog(&layout);
    let m = cuda::find(&cat, model).ok_or_else(|| anyhow::anyhow!("unknown model {model}"))?;
    let CudaRecipe::H3(r) = &m.recipe else {
        return Err(StageError::Error(anyhow::anyhow!("{model} is not an H3 model")));
    };
    let plan = ProcessPlan::for_models(std::slice::from_ref(&m)).map_err(|e| anyhow::anyhow!(e))?;
    cuda::install_process_plan(&plan).map_err(|e| anyhow::anyhow!("{e}"))?;
    fastvideo_cudarc::resolve_device("cuda:0").map_err(|e| anyhow::anyhow!("cuda:0: {e}"))?;
    let mut options = H3Model::pipeline_options(r, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    // The text and vision encoders are not part of the tree: stream them.
    options.text_encoder = TextEncoderChoice::Streamed;
    options.i2v_encoder = I2vEncoderChoice::Stream;
    let identity = dit_tree_identity_for(&r.weights, &options)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("{model}: this recipe has no pre-quantized form"))?;
    if identity.linear_route != "device-bf16" {
        // fv-serve loads with bf16 device linears; a tree built otherwise
        // (e.g. `--mode exact`, FASTVIDEO_BF16=0) would never be selected.
        return Err(StageError::Error(anyhow::anyhow!(
            "linear route {} is not fv-serve's (device-bf16): run with --mode fast",
            identity.linear_route
        )));
    }
    let quant = fastvideo_cudarc::wan::quant::QuantMode::parse(&identity.quant).map_err(|e| anyhow::anyhow!(e))?;
    let name = dit_tree::dir_name(&identity.recipe, quant);
    let final_dir = r.weights.join(&name);
    let tmp = tmp.map_or_else(
        || r.weights.join(format!("{name}.tmp-{}", std::process::id())),
        Path::to_path_buf,
    );
    report.set("model", model);
    report.set("weights", r.weights.display().to_string());
    report.set("tree", final_dir.display().to_string());
    report.set("tmp", tmp.display().to_string());
    report.set("identity", &identity);
    if !verify_only && final_dir.exists() {
        return Err(StageError::Error(anyhow::anyhow!(
            "{} exists; trees are add-only (verify it with --verify-only)",
            final_dir.display()
        )));
    }
    let transformer = r.weights.join("transformer");

    let load = |mode: &str, dir: Option<&Path>| -> anyhow::Result<(TreeWriter, bool, f64)> {
        std::env::set_var(dit_tree::ENV, mode);
        match dir {
            Some(d) => std::env::set_var(dit_tree::DIR_ENV, d),
            None => std::env::remove_var(dit_tree::DIR_ENV),
        }
        let t0 = Instant::now();
        let pipe = H3Pipeline::load(&r.weights, options.clone()).map_err(|e| anyhow::anyhow!("{e}"))?;
        let load_s = t0.elapsed().as_secs_f64();
        let from_tree = pipe.load_timings.dit_from_tree;
        let mut w = TreeWriter::default();
        pipe.export_dit(&mut w).map_err(|e| anyhow::anyhow!("{e}"))?;
        drop(pipe);
        let _ = fastvideo_cudarc::wan::device::trim_pool();
        Ok((w, from_tree, load_s))
    };

    let (digest, per_tensor) = if verify_only {
        let (w, _, load_s) = load("off", None)?;
        report.note("load_time", json!({"load_s": load_s, "tensors": w.tensors.len(), "digest": w.digest()}));
        (w.digest(), w.tensor_digests())
    } else {
        let t0 = Instant::now();
        let (w, from_tree, load_s) = load("off", None)?;
        if from_tree {
            return Err(StageError::Error(anyhow::anyhow!("the reference load came from a tree")));
        }
        let export_s = t0.elapsed().as_secs_f64() - load_s;
        let digest = w.digest();
        let per = w.tensor_digests();
        let t0 = Instant::now();
        let build = std::env::var("FV_BUILD_ID")
            .ok()
            .or_else(|| std::fs::read_to_string("/opt/fastvideo-rs/target/release/fv-gpucheck.build-id").ok())
            .unwrap_or_default();
        let layers = identity.num_layers.to_string();
        let file = w
            .write(&tmp, &[("format", dit_tree::FORMAT), ("recipe", &identity.recipe), ("quant", &identity.quant), ("layers", &layers)])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let write_s = t0.elapsed().as_secs_f64();
        let (tensors, quantized, stored) = (w.tensors.len(), w.quantized_linears, w.bytes());
        drop(w);
        let t0 = Instant::now();
        let sha = sha256_file(&tmp.join(dit_tree::MODEL_FILE)).map_err(anyhow::Error::from)?;
        let mut source = dit_tree::source_entries(&transformer)?;
        if hash_sources {
            use rayon::prelude::*;
            let hashes: Vec<String> = source
                .par_iter()
                .map(|f| sha256_file(&transformer.join(&f.name)))
                .collect::<std::io::Result<_>>()
                .map_err(anyhow::Error::from)?;
            for (f, h) in source.iter_mut().zip(hashes) {
                f.sha256 = h;
            }
        }
        let mut identity = identity.clone();
        if let Some(a) = identity.adapter.as_mut() {
            let base = r.weights.parent().unwrap_or(&r.weights);
            let p = if Path::new(&a.name).is_absolute() { PathBuf::from(&a.name) } else { base.join(&a.name) };
            a.sha256 = sha256_file(&p).map_err(anyhow::Error::from)?;
        }
        let hash_s = t0.elapsed().as_secs_f64();
        let manifest = dit_tree::Manifest {
            format: dit_tree::FORMAT.into(),
            version: dit_tree::VERSION,
            identity,
            source,
            files: vec![dit_tree::FileEntry { sha256: sha.clone(), ..file }],
            tensors,
            quantized_linears: quantized,
            digest: digest.clone(),
            created_by: build.trim().to_string(),
        };
        dit_tree::write_manifest(&tmp, &manifest)?;
        report.note(
            "written",
            json!({"tensors": tensors, "quantized_linears": quantized, "bytes": stored, "file_bytes": manifest.files[0].bytes,
                   "file_sha256": sha, "digest": digest, "load_s": load_s, "export_s": export_s, "write_s": write_s, "hash_s": hash_s}),
        );
        (digest, per)
    };

    if verify || verify_only {
        let dir = if verify_only && !tmp.exists() { final_dir.clone() } else { tmp.clone() };
        let manifest = dit_tree::check(&transformer, &dir, &identity).map_err(|e| anyhow::anyhow!("tree rejected: {e}"))?;
        let (w, from_tree, load_s) = load("require", Some(&dir))?;
        let tree_digest = w.digest();
        let differs: Vec<String> = w
            .tensor_digests()
            .into_iter()
            .filter(|(k, v)| per_tensor.get(k) != Some(v))
            .map(|(k, _)| k)
            .chain(per_tensor.keys().filter(|k| !w.tensors.iter().any(|t| &t.name == *k)).cloned())
            .take(20)
            .collect();
        drop(w);
        let t0 = Instant::now();
        let sha = sha256_file(&dir.join(dit_tree::MODEL_FILE)).map_err(anyhow::Error::from)?;
        let sha_s = t0.elapsed().as_secs_f64();
        report.check(
            "dit_tree_identity",
            from_tree && tree_digest == digest && manifest.digest == digest && differs.is_empty() && sha == manifest.files[0].sha256,
            json!({"load_time_digest": digest, "tree_digest": tree_digest, "manifest_digest": manifest.digest, "from_tree": from_tree,
                   "differing_tensors": differs, "file_sha256": sha, "manifest_sha256": manifest.files[0].sha256, "tree_load_s": load_s, "sha_s": sha_s}),
            json!({"digests_equal": true, "sha256_equal": true}),
        )?;
    }
    if finalize && !verify_only {
        if final_dir.exists() {
            return Err(StageError::Error(anyhow::anyhow!("{} appeared meanwhile; leaving {}", final_dir.display(), tmp.display())));
        }
        std::fs::rename(&tmp, &final_dir).map_err(anyhow::Error::from)?;
        report.note("finalized", json!({"from": tmp.display().to_string(), "to": final_dir.display().to_string()}));
    }
    Ok(())
}
