//! Pre-quantized resident text encoders (E13).
//!
//! A resident FP8 encoder ([`super::WeightPrecision::Fp8Rows`]) is built at
//! load by reading the bf16 checkpoint and quantizing every linear on the
//! device: E4M3 codes plus one f32 scale per output row. The result is a
//! constant of the checkpoint, so an offline tool (`fv-gpucheck
//! quantize-text-encoder`) writes it once as safetensors next to the source
//! (`<root>/text_encoder_fp8/`) and the loader reads half the bytes and skips
//! the quantization.
//!
//! The tree is self-contained for the layers it was built for: each
//! quantized linear's `weight` is `F8_E4M3 [out, in]` with an F32
//! `weight_scale_rows [out]` beside it; every other tensor those layers, the
//! embedding table and (when all layers are kept) the final norm read is
//! copied byte for byte. `manifest.json` records the rule, the version, the
//! layer count, the source shards (name, size, sha256) and the written files.
//! A tree whose manifest is missing, of another version or rule, or whose
//! source shard sizes no longer match is ignored and the encoder is quantized
//! at load as before.
//!
//! `FASTVIDEO_TEXT_FP8_TREE=0` ignores the tree.

use std::path::{Path, PathBuf};

use fastvideo_loader::{LazyDType, LazyStore, TensorSpec};
use serde::{Deserialize, Serialize};

use super::DecoderConfig;
use crate::wan::nn::{Fp8Capture, FP8_ROWS_SCALE_SUFFIX};
use crate::wan::tensor::{Result, TensorError};
use crate::wan::weights::WeightMap;

/// Directory name beside the source `text_encoder/`.
pub const DIR_NAME: &str = "text_encoder_fp8";
pub const MANIFEST: &str = "manifest.json";
pub const FORMAT: &str = "fastvideo-rs/fp8-rows";
pub const VERSION: u32 = 1;
/// The quantization the codes come from (`fp8_rows_quantize_device`).
pub const RULE: &str =
    "weight-only E4M3 (e4m3fn), one f32 scale per output row = max|w_row|/448 (0 → 1), codes = e4m3(w/scale) round-to-nearest-even; device kernel fp8_row_scales + fp8_rows_quantize on the bf16 checkpoint widened to f32";

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileEntry {
    pub name: String,
    pub bytes: u64,
    /// Hex sha256; empty when not computed.
    #[serde(default)]
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub rule: String,
    /// Decoder layers held (`0..layers`).
    pub layers: usize,
    pub layer_prefix: String,
    pub quantized_linears: usize,
    /// The bf16 shards the codes were computed from.
    pub source: Vec<FileEntry>,
    /// The files of this tree.
    pub files: Vec<FileEntry>,
    /// Build id of the writer.
    #[serde(default)]
    pub created_by: String,
}

/// `.safetensors` (and their index) of a text encoder directory, by name.
pub fn source_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| msg(format!("{}: {e}", dir.display())))? {
        let p = e.map_err(|e| msg(e.to_string()))?.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_file() && name.ends_with(".safetensors") {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn entry(p: &Path, sha256: String) -> Result<FileEntry> {
    let bytes = std::fs::metadata(p)
        .map_err(|e| msg(format!("{}: {e}", p.display())))?
        .len();
    Ok(FileEntry {
        name: p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string(),
        bytes,
        sha256,
    })
}

/// Why `fp8_dir` cannot stand in for `source_dir` with `layers` layers, or
/// `Ok(manifest)`. Checks sizes, not hashes: hashing the 60 GB source at load
/// would cost what the tree saves (the tool's `--verify` does hash).
pub fn check(
    source_dir: &Path,
    fp8_dir: &Path,
    layers: usize,
) -> std::result::Result<Manifest, String> {
    let raw = std::fs::read_to_string(fp8_dir.join(MANIFEST))
        .map_err(|e| format!("no manifest ({e})"))?;
    let m: Manifest = serde_json::from_str(&raw).map_err(|e| format!("manifest: {e}"))?;
    if m.format != FORMAT || m.version != VERSION || m.rule != RULE {
        return Err(format!(
            "manifest {} v{} (want {FORMAT} v{VERSION}) or another rule",
            m.format, m.version
        ));
    }
    if m.layers < layers {
        return Err(format!("tree holds {} layers, {layers} needed", m.layers));
    }
    let now = source_files(source_dir).map_err(|e| e.to_string())?;
    let now: Vec<FileEntry> = now
        .iter()
        .map(|p| entry(p, String::new()))
        .collect::<Result<_>>()
        .map_err(|e| e.to_string())?;
    let strip = |v: &[FileEntry]| -> Vec<(String, u64)> {
        v.iter().map(|f| (f.name.clone(), f.bytes)).collect()
    };
    if strip(&now) != strip(&m.source) {
        return Err("source shards differ from the ones the tree was built from".into());
    }
    for f in &m.files {
        let got = std::fs::metadata(fp8_dir.join(&f.name))
            .map(|m| m.len())
            .ok();
        if got != Some(f.bytes) {
            return Err(format!(
                "{}: {got:?} bytes, manifest says {}",
                f.name, f.bytes
            ));
        }
    }
    Ok(m)
}

/// The pre-quantized map for `source_dir` (`…/text_encoder`) when a valid
/// sibling tree exists, else `None` (logged).
pub fn open_tree(source_dir: &Path, layers: usize) -> Result<Option<WeightMap>> {
    if std::env::var("FASTVIDEO_TEXT_FP8_TREE").is_ok_and(|v| v.trim() == "0") {
        return Ok(None);
    }
    let Some(parent) = source_dir.parent() else {
        return Ok(None);
    };
    let dir = parent.join(DIR_NAME);
    if !dir.is_dir() {
        return Ok(None);
    }
    match check(source_dir, &dir, layers) {
        Ok(m) => {
            crate::wan::log::info(format_args!(
                "text encoder: pre-quantized fp8 tree {} ({} linears, {} layers)",
                dir.display(),
                m.quantized_linears,
                m.layers
            ));
            Ok(Some(WeightMap::open(&dir)?))
        }
        Err(why) => {
            crate::wan::log::info(format_args!(
                "text encoder: ignoring {} ({why}); quantizing at load",
                dir.display()
            ));
            Ok(None)
        }
    }
}

/// Keys of `src` a resident decoder with `layers` layers reads.
pub fn kept_keys(src: &LazyStore, cfg: &DecoderConfig, layers: usize) -> Vec<String> {
    let mut keys: Vec<String> = src
        .keys()
        .filter(|k| {
            if *k == cfg.embed_key || (layers == cfg.num_layers() && *k == cfg.final_norm_key) {
                return true;
            }
            let Some(rest) = k
                .strip_prefix(&cfg.layer_prefix)
                .and_then(|r| r.strip_prefix('.'))
            else {
                return false;
            };
            rest.split('.')
                .next()
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i < layers)
        })
        .map(str::to_string)
        .collect();
    keys.sort_by(|a, b| fastvideo_loader::natural_cmp(a, b));
    keys
}

/// Write the tree for `source_dir` into `out_dir` (created; must not hold a
/// manifest already). `captured` is [`crate::wan::nn::capture_fp8_rows`] of a
/// resident FP8 load of the same `cfg` / `layers` from `source_dir`.
/// `source_hashes` maps source file names to their sha256.
pub fn write_tree(
    source_dir: &Path,
    out_dir: &Path,
    cfg: &DecoderConfig,
    layers: usize,
    captured: &Fp8Capture,
    source_hashes: &std::collections::HashMap<String, String>,
    created_by: &str,
) -> Result<Manifest> {
    if out_dir.join(MANIFEST).exists() {
        return Err(msg(format!(
            "{} already has a manifest; refusing to overwrite",
            out_dir.display()
        )));
    }
    std::fs::create_dir_all(out_dir).map_err(|e| msg(format!("{}: {e}", out_dir.display())))?;
    let src = LazyStore::open(source_dir).map_err(|e| msg(e.to_string()))?;
    let by_weight: std::collections::HashMap<String, usize> = captured
        .iter()
        .enumerate()
        .map(|(i, (p, _, _))| (format!("{p}.weight"), i))
        .collect();
    let keys = kept_keys(&src, cfg, layers);
    enum Part<'a> {
        Codes(usize),
        Scales(usize),
        Copy(&'a str),
    }
    let mut specs = Vec::new();
    let mut parts = Vec::new();
    for k in &keys {
        let shape = src
            .shape(k)
            .ok_or_else(|| msg(format!("{k}: vanished")))?
            .to_vec();
        if let Some(&i) = by_weight.get(k) {
            let (_, codes, scales) = &captured[i];
            if codes.len() != shape.iter().product::<usize>() || scales.len() != shape[0] {
                return Err(msg(format!("{k}: captured codes do not match {shape:?}")));
            }
            specs.push(TensorSpec::new(k.clone(), LazyDType::F8E4M3, shape.clone()));
            parts.push(Part::Codes(i));
            let base = k.strip_suffix(".weight").expect("weight key");
            specs.push(TensorSpec::new(
                format!("{base}.{FP8_ROWS_SCALE_SUFFIX}"),
                LazyDType::F32,
                vec![shape[0]],
            ));
            parts.push(Part::Scales(i));
        } else {
            let v = src.view(k).map_err(|e| msg(e.to_string()))?;
            specs.push(TensorSpec::new(k.clone(), v.dtype.clone(), shape));
            parts.push(Part::Copy(k));
        }
    }
    if by_weight.keys().any(|k| !keys.contains(k)) {
        return Err(msg("a captured linear is outside the kept keys"));
    }
    let path = out_dir.join("model.safetensors");
    let layers_s = layers.to_string();
    let data = |i: usize| -> std::result::Result<std::borrow::Cow<'_, [u8]>, fastvideo_loader::LoaderError> {
        Ok(match parts[i] {
            Part::Codes(c) => std::borrow::Cow::Borrowed(&captured[c].1[..]),
            Part::Scales(c) => std::borrow::Cow::Owned(
                captured[c].2.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            Part::Copy(k) => std::borrow::Cow::Borrowed(src.view(k)?.bytes),
        })
    };
    fastvideo_loader::write_parallel(
        &path,
        &specs,
        &[("format", FORMAT), ("rule", RULE), ("layers", &layers_s)],
        &data,
        16 << 20,
    )
    .map_err(|e| msg(e.to_string()))?;
    let source = source_files(source_dir)?
        .iter()
        .map(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            entry(p, source_hashes.get(name).cloned().unwrap_or_default())
        })
        .collect::<Result<Vec<_>>>()?;
    let manifest = Manifest {
        format: FORMAT.into(),
        version: VERSION,
        rule: RULE.into(),
        layers,
        layer_prefix: cfg.layer_prefix.clone(),
        quantized_linears: captured.len(),
        source,
        files: vec![entry(&path, String::new())?],
        created_by: created_by.into(),
    };
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| msg(e.to_string()))?;
    let tmp = out_dir.join(format!("{MANIFEST}.part"));
    std::fs::write(&tmp, text).map_err(|e| msg(e.to_string()))?;
    std::fs::rename(&tmp, out_dir.join(MANIFEST)).map_err(|e| msg(e.to_string()))?;
    Ok(manifest)
}
