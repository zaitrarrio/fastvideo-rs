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
