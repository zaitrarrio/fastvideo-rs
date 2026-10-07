//! Pre-quantized resident H3 DiT trees (fast boot, part A).
//!
//! A served H3 recipe builds its DiT the same way on every boot: read the
//! base `transformer/` (98.8 GB viewed on h3-turbo: 26 GB of AdaLN
//! projections, 41 GB of blocks, the rest), merge the recipe's adapter, fold
//! the AdaLN projections into the precomputed modulation table, and quantize
//! blocks 2..=46 to MXFP8. Everything that comes out is a constant of
//! (checkpoint, adapter, strength, recipe, quantization mode, ladder). This
//! module stores exactly that resident state once, beside the source:
//!
//! `<root>/transformer_prequant_<recipe>_<quant>/model.safetensors` + `manifest.json`
//!
//! and the loader rebuilds the same linears, norms and AdaLN table from it,
//! reading ~26 GB instead of ~99 GB and doing no merge, no AdaLN evaluation
//! and no quantization.
//!
//! **What is stored, per linear** ([`crate::wan::nn::Linear::snapshot`]):
//! `<p>.weight` (BF16 as resident on the device, or F32 off the bf16 route)
//! or, for a reference-recipe weight, `<p>.qblob` (U8: the one byte blob the
//! GEMM reads: E4M3 codes, bf16 rows of unquantized sections such as the VSA
//! gate, then the swizzled E8M0 scales), `<p>.qscales` (F32, per section; zero
//! for MXFP8) and `<p>.qlayout` (U8: [`QuantLayout::encode`]); `<p>.bias`
//! (F32). Fused projections are stored fused (`<block>.attn.qkvg`). Norm
//! weights are stored as the F32 values the pinned tensors hold, and the
//! AdaLN table (`adaln_table.*`) as its F32 arrays. Every byte is read back
//! from the loaded model, so the tree is the load-time result by
//! construction; `fv-gpucheck quantize-dit` then loads it back and requires
//! a byte-identical digest of every tensor.
//!
//! **Selection.** The loader takes the tree when the directory exists and its
//! manifest matches: format and version, the [`Identity`] of this load
//! (recipe, quantization mode, gate, layer counts, the ladder's timesteps,
//! the adapter's file name, size and strength, the linear route), the source
//! shards' names and sizes, and each tree file's size. Hashes are checked by
//! the tool and `verify-weights.sh dit-prequant` (reading 26 GB at boot to
//! hash it would cost what the tree saves). Anything else falls back to
//! quantize-at-load, and says why.
//!
//! **Knobs.** `FASTVIDEO_H3_DIT_TREE` = `auto` (default: use a valid tree),
//! `0` / `off` (always quantize at load), `1` / `require` (fail the load when
//! no valid tree is there). `FASTVIDEO_H3_DIT_TREE_DIR=<dir>` names the tree
//! explicitly (the tool's verify pass, tests).

use std::path::{Path, PathBuf};

use fastvideo_loader::{LazyDType, TensorSpec};
use serde::{Deserialize, Serialize};

use crate::wan::nn::{Linear, LinearSnapshot, SnapshotRef, SnapshotWeight};
use crate::wan::quant::{QuantLayout, QuantMode};
use crate::wan::tensor::{CudaTensor, Result, TensorDType, TensorError};
use crate::wan::weights::WeightMap;

pub const FORMAT: &str = "fastvideo-rs/h3-dit-resident";
pub const VERSION: u32 = 1;
pub const MANIFEST: &str = "manifest.json";
pub const MODEL_FILE: &str = "model.safetensors";
pub const ENV: &str = "FASTVIDEO_H3_DIT_TREE";
pub const DIR_ENV: &str = "FASTVIDEO_H3_DIT_TREE_DIR";

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// The tree directory of `recipe` under `quant`, beside `transformer/`.
pub fn dir_name(recipe: &str, quant: QuantMode) -> String {
    format!("transformer_prequant_{recipe}_{}", quant.as_str())
}

/// `FASTVIDEO_H3_DIT_TREE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeMode {
    Auto,
    Off,
    Require,
}

impl TreeMode {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "0" | "off" | "false" | "no" => Ok(Self::Off),
            "1" | "on" | "require" | "true" | "yes" => Ok(Self::Require),
            other => Err(format!("{ENV}={other}: expected auto|off|require")),
        }
    }

    pub fn from_env() -> std::result::Result<Self, String> {
        match std::env::var(ENV) {
            Ok(v) => Self::parse(&v),
            Err(_) => Ok(Self::Auto),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileEntry {
    pub name: String,
    pub bytes: u64,
    /// Hex sha256; empty when not computed.
    #[serde(default)]
    pub sha256: String,
}

/// Everything the resident DiT is a function of, besides the source shards.
/// A tree whose identity differs from the load's is never used.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Identity {
    /// The H3 recipe name the load resolved (`4step-vsa`, `sol-h3`, ...).
    pub recipe: String,
    /// `FASTVIDEO_H3_QUANT` as resolved for the device (`mxfp8`, `w8a8`, `off`).
    pub quant: String,
    /// Whether the blocks hold `to_gate_compress` (VSA recipes).
    pub with_gate: bool,
    pub num_layers: usize,
    pub num_refiner_layers: usize,
    pub hidden_size: usize,
    /// The ladder: f32 bits of (video, audio) timesteps per step.
    pub timesteps: Vec<[u32; 2]>,
    /// f32 bits of the keyframe noise-augmentation timestep.
    pub keyframe_t: u32,
    /// The merged adapter: path relative to the weight root's parent, size
    /// (and sha256 when the tool hashed it; not compared at load).
    pub adapter: Option<FileEntry>,
    /// f32 bits of the adapter's effective scale.
    pub adapter_scale: u32,
    /// `device-bf16` (weights bf16 on the device) or `f32`.
    pub linear_route: String,
}

impl Identity {
    /// Equality as the loader checks it (the adapter by name and size).
    pub fn matches(&self, want: &Identity) -> std::result::Result<(), String> {
        let strip = |i: &Identity| {
            let mut i = i.clone();
            if let Some(a) = i.adapter.as_mut() {
                a.sha256.clear();
            }
            i
        };
        let (have, want) = (strip(self), strip(want));
        if have == want {
            return Ok(());
        }
        let mut diff = Vec::new();
        macro_rules! cmp {
            ($f:ident) => {
                if have.$f != want.$f {
                    diff.push(format!(
                        "{} {:?} (want {:?})",
                        stringify!($f),
                        have.$f,
                        want.$f
                    ));
                }
            };
        }
        cmp!(recipe);
        cmp!(quant);
        cmp!(with_gate);
        cmp!(num_layers);
        cmp!(num_refiner_layers);
        cmp!(hidden_size);
        cmp!(timesteps);
        cmp!(keyframe_t);
        cmp!(adapter);
        cmp!(adapter_scale);
        cmp!(linear_route);
        Err(format!("built for another load: {}", diff.join(", ")))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub identity: Identity,
    /// `transformer/*.safetensors` the tree was built from.
    pub source: Vec<FileEntry>,
    /// The files of this tree.
    pub files: Vec<FileEntry>,
    pub tensors: usize,
    /// Linears stored as a reference-recipe blob.
    pub quantized_linears: usize,
    /// sha256 over every stored tensor (name, dtype, shape, bytes) in name
    /// order: [`TreeWriter::digest`] of the load-time model.
    pub digest: String,
    /// Build id of the writer.
    #[serde(default)]
    pub created_by: String,
}

/// `.safetensors` of a directory, by name.
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

pub fn entry(p: &Path, name: String, sha256: String) -> Result<FileEntry> {
    let bytes = std::fs::metadata(p)
        .map_err(|e| msg(format!("{}: {e}", p.display())))?
        .len();
    Ok(FileEntry {
        name,
        bytes,
        sha256,
    })
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

/// The source shard list of `transformer` (names and sizes).
pub fn source_entries(transformer: &Path) -> Result<Vec<FileEntry>> {
    source_files(transformer)?
        .iter()
        .map(|p| entry(p, file_name(p), String::new()))
        .collect()
}

/// The adapter's entry: path relative to `root`'s parent when it lies under
/// it (the volume layout), size, no hash.
pub fn adapter_entry(root: &Path, adapter: &Path) -> Result<FileEntry> {
    let base = root.parent().unwrap_or(root);
    let name = adapter
        .strip_prefix(base)
        .unwrap_or(adapter)
        .to_string_lossy()
        .into_owned();
    entry(adapter, name, String::new())
}

/// Why the tree in `dir` cannot stand in for `transformer` under `want`, or
/// `Ok(manifest)`. Sizes and identity only (see the module docs).
pub fn check(transformer: &Path, dir: &Path, want: &Identity) -> std::result::Result<Manifest, String> {
    let raw = std::fs::read_to_string(dir.join(MANIFEST)).map_err(|e| format!("no manifest ({e})"))?;
    let m: Manifest = serde_json::from_str(&raw).map_err(|e| format!("manifest: {e}"))?;
    if m.format != FORMAT || m.version != VERSION {
        return Err(format!(
            "manifest {} v{} (want {FORMAT} v{VERSION})",
            m.format, m.version
        ));
    }
    m.identity.matches(want)?;
    let now = source_entries(transformer).map_err(|e| e.to_string())?;
    let strip = |v: &[FileEntry]| -> Vec<(String, u64)> {
        v.iter().map(|f| (f.name.clone(), f.bytes)).collect()
    };
    if strip(&now) != strip(&m.source) {
        return Err("source shards differ from the ones the tree was built from".into());
    }
    for f in &m.files {
        let got = std::fs::metadata(dir.join(&f.name)).map(|m| m.len()).ok();
        if got != Some(f.bytes) {
            return Err(format!("{}: {got:?} bytes, manifest says {}", f.name, f.bytes));
        }
    }
    Ok(m)
}

/// The tree to load `transformer` from under `want`, by `FASTVIDEO_H3_DIT_TREE`
/// (and `_DIR`): `Ok(Some)` for a valid tree, `Ok(None)` to quantize at load
/// (logged with the reason), `Err` when the mode requires a tree.
pub fn select(
    root: &Path,
    transformer: &Path,
    want: Option<&Identity>,
) -> Result<Option<(PathBuf, Manifest)>> {
    let mode = TreeMode::from_env().map_err(msg)?;
    let quant = want.map_or("?", |w| w.quant.as_str());
    let recipe = want.map_or("?", |w| w.recipe.as_str());
    let dir = match std::env::var(DIR_ENV) {
        Ok(d) if !d.trim().is_empty() => PathBuf::from(d.trim()),
        _ => root.join(format!("transformer_prequant_{recipe}_{quant}")),
    };
    let refuse = |why: String| -> Result<Option<(PathBuf, Manifest)>> {
        if mode == TreeMode::Require {
            return Err(msg(format!(
                "h3 dit: {ENV}=require but no usable pre-quantized tree: {why}"
            )));
        }
        crate::wan::log::info(format_args!("h3 dit: quantizing at load ({why})"));
        Ok(None)
    };
    if mode == TreeMode::Off {
        crate::wan::log::info(format_args!("h3 dit: quantizing at load ({ENV}=off)"));
        return Ok(None);
    }
    let Some(want) = want else {
        return refuse("this load has no pre-quantized form (adapter or layout not covered)".into());
    };
    if !dir.is_dir() {
        return refuse(format!("no tree at {}", dir.display()));
    }
    match check(transformer, &dir, want) {
        Ok(m) => {
            crate::wan::log::info(format_args!(
                "h3 dit: pre-quantized tree {} ({} tensors, {} quantized linears, {:.2} GB, digest {})",
                dir.display(),
                m.tensors,
                m.quantized_linears,
                m.files.iter().map(|f| f.bytes).sum::<u64>() as f64 / 1e9,
                &m.digest[..m.digest.len().min(12)]
            ));
            Ok(Some((dir, m)))
        }
        Err(why) => refuse(format!("ignoring {}: {why}", dir.display())),
    }
}

// ---------------------------------------------------------------------------
// Writing: every resident tensor as bytes
// ---------------------------------------------------------------------------

/// One stored tensor.
pub struct Stored {
    pub name: String,
    pub dtype: LazyDType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Collects a loaded model's resident state ([`super::transformer`]'s
/// `export` methods call it).
#[derive(Default)]
pub struct TreeWriter {
    pub tensors: Vec<Stored>,
    pub quantized_linears: usize,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl TreeWriter {
    fn push(&mut self, name: String, dtype: LazyDType, shape: Vec<usize>, bytes: Vec<u8>) {
        self.tensors.push(Stored {
            name,
            dtype,
            shape,
            bytes,
        });
    }

    /// A linear under `prefix` ([`Linear::snapshot`]).
    pub fn linear(&mut self, prefix: &str, l: &Linear) -> Result<()> {
        let LinearSnapshot {
            in_dim,
            out_dim,
            weight,
            bias,
        } = l.snapshot().map_err(|e| msg(format!("{prefix}: {e}")))?;
        match weight {
            SnapshotWeight::Bf16(b) => {
                self.push(format!("{prefix}.weight"), LazyDType::BF16, vec![out_dim, in_dim], b)
            }
            SnapshotWeight::F32(v) => self.push(
                format!("{prefix}.weight"),
                LazyDType::F32,
                vec![out_dim, in_dim],
                f32_bytes(&v),
            ),
            SnapshotWeight::Quant {
                layout,
                scales,
                blob,
            } => {
                let enc = layout.encode();
                self.push(format!("{prefix}.qlayout"), LazyDType::U8, vec![enc.len()], enc);
                self.push(
                    format!("{prefix}.qscales"),
                    LazyDType::F32,
                    vec![scales.len()],
                    f32_bytes(&scales),
                );
                self.push(format!("{prefix}.qblob"), LazyDType::U8, vec![blob.len()], blob);
                self.quantized_linears += 1;
            }
        }
        if let Some(b) = bias {
            self.push(format!("{prefix}.bias"), LazyDType::F32, vec![b.len()], f32_bytes(&b));
        }
        Ok(())
    }

    /// An F32 tensor (a pinned norm weight) as it holds its values.
    pub fn tensor(&mut self, key: &str, t: &CudaTensor) -> Result<()> {
        if t.dtype != TensorDType::F32 {
            return Err(msg(format!("{key}: only F32 tensors are stored, got {:?}", t.dtype)));
        }
        let v = t.host_cow()?;
        self.push(key.to_string(), LazyDType::F32, t.shape.clone(), f32_bytes(&v));
        Ok(())
    }

    /// Raw F32 values (the AdaLN table).
    pub fn values(&mut self, key: &str, shape: Vec<usize>, v: &[f32]) -> Result<()> {
        if shape.iter().product::<usize>() != v.len() {
            return Err(msg(format!("{key}: {} values for {shape:?}", v.len())));
        }
        self.push(key.to_string(), LazyDType::F32, shape, f32_bytes(v));
        Ok(())
    }

    /// sha256 over every tensor (name, dtype, shape, bytes) in name order:
    /// equal digests mean equal resident state.
    pub fn digest(&self) -> String {
        digest_of(self.tensors.iter().map(|t| (t.name.as_str(), t.dtype.as_str(), &t.shape[..], &t.bytes[..])))
    }

    /// sha256 of each tensor's bytes, by name (to name what differs when
    /// two digests do).
    pub fn tensor_digests(&self) -> std::collections::BTreeMap<String, String> {
        use sha2::{Digest, Sha256};
        self.tensors
            .iter()
            .map(|t| {
                let d = Sha256::digest(&t.bytes);
                (t.name.clone(), d.iter().map(|b| format!("{b:02x}")).collect())
            })
            .collect()
    }

    /// Bytes stored.
    pub fn bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes.len() as u64).sum()
    }

    /// Write `model.safetensors` into `out_dir` (created; must hold no
    /// manifest). Returns its entry (no hash).
    pub fn write(&self, out_dir: &Path, metadata: &[(&str, &str)]) -> Result<FileEntry> {
        if out_dir.join(MANIFEST).exists() {
            return Err(msg(format!(
                "{} already has a manifest; refusing to overwrite",
                out_dir.display()
            )));
        }
        std::fs::create_dir_all(out_dir).map_err(|e| msg(format!("{}: {e}", out_dir.display())))?;
        let mut order: Vec<usize> = (0..self.tensors.len()).collect();
        order.sort_by(|&a, &b| fastvideo_loader::natural_cmp(&self.tensors[a].name, &self.tensors[b].name));
        let specs: Vec<TensorSpec> = order
            .iter()
            .map(|&i| {
                let t = &self.tensors[i];
                TensorSpec::new(t.name.clone(), t.dtype.clone(), t.shape.clone())
            })
            .collect();
        let path = out_dir.join(MODEL_FILE);
        let data = |i: usize| -> std::result::Result<std::borrow::Cow<'_, [u8]>, fastvideo_loader::LoaderError> {
            Ok(std::borrow::Cow::Borrowed(&self.tensors[order[i]].bytes[..]))
        };
        fastvideo_loader::write_parallel(&path, &specs, metadata, &data, 16 << 20)
            .map_err(|e| msg(e.to_string()))?;
        entry(&path, MODEL_FILE.into(), String::new())
    }
}

/// The digest [`TreeWriter::digest`] computes, over any (name, dtype, shape,
/// bytes) set.
pub fn digest_of<'a>(items: impl Iterator<Item = (&'a str, &'a str, &'a [usize], &'a [u8])>) -> String {
    use sha2::{Digest, Sha256};
    let mut v: Vec<_> = items.collect();
    v.sort_by(|a, b| a.0.cmp(b.0));
    let mut h = Sha256::new();
    for (name, dtype, shape, bytes) in v {
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update(dtype.as_bytes());
        h.update((shape.len() as u64).to_le_bytes());
        for d in shape {
            h.update((*d as u64).to_le_bytes());
        }
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// sha256 of a file (hex).
pub fn sha256_file(p: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Write `manifest.json` (through a `.part` file, renamed into place).
pub fn write_manifest(out_dir: &Path, m: &Manifest) -> Result<()> {
    let text = serde_json::to_string_pretty(m).map_err(|e| msg(e.to_string()))?;
    let tmp = out_dir.join(format!("{MANIFEST}.part"));
    std::fs::write(&tmp, text).map_err(|e| msg(e.to_string()))?;
    std::fs::rename(&tmp, out_dir.join(MANIFEST)).map_err(|e| msg(e.to_string()))
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// The opened tree ([`super::transformer`]'s `from_tree` constructors read it).
pub struct TreeReader {
    pub map: WeightMap,
    pub dir: PathBuf,
}

impl TreeReader {
    /// Open `dir` and queue its tensors for read-ahead: refiner, AdaLN table,
    /// blocks, the rest (the order the loader consumes them).
    pub fn open(dir: &Path) -> Result<Self> {
        let map = WeightMap::open(dir)?;
        let refiner = |k: &str| k.starts_with("token_refiner") || k.starts_with("context_embedder");
        let table = |k: &str| k.starts_with("adaln_table");
        let blocks = |k: &str| k.starts_with("transformer_blocks.");
        let rest = |_: &str| true;
        map.prefetch_groups(&[&refiner, &table, &blocks, &rest]);
        Ok(Self {
            map,
            dir: dir.to_path_buf(),
        })
    }

    fn lazy(&self) -> Result<&fastvideo_loader::LazyStore> {
        self.map.lazy().ok_or_else(|| msg("dit tree: not a mapped store"))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.map.lazy().is_some_and(|l| l.contains(key))
    }

    fn view(&self, key: &str) -> Result<fastvideo_loader::LazyView<'_>> {
        self.lazy()?
            .view(key)
            .map_err(|e| msg(format!("dit tree {}: {e}", self.dir.display())))
    }

    fn f32s(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let v = self.view(key)?;
        if *v.dtype != LazyDType::F32 {
            return Err(msg(format!("dit tree {key}: {:?}, want F32", v.dtype)));
        }
        let vals = v
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((v.shape.to_vec(), vals))
    }

    /// The linear stored under `prefix`, `[out_dim, in_dim]`.
    pub fn linear(&self, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
        let bias = if self.contains(&format!("{prefix}.bias")) {
            Some(self.f32s(&format!("{prefix}.bias"))?.1)
        } else {
            None
        };
        let blob_key = format!("{prefix}.qblob");
        if self.contains(&blob_key) {
            let layout = QuantLayout::decode(self.view(&format!("{prefix}.qlayout"))?.bytes)?;
            let scales = self.f32s(&format!("{prefix}.qscales"))?.1;
            let blob = self.view(&blob_key)?;
            return Linear::from_snapshot(
                in_dim,
                out_dim,
                SnapshotRef::Quant {
                    layout,
                    scales,
                    blob: blob.bytes,
                },
                bias,
            )
            .map_err(|e| msg(format!("{prefix}: {e}")));
        }
        let key = format!("{prefix}.weight");
        let v = self.view(&key)?;
        if v.shape != [out_dim, in_dim] {
            return Err(msg(format!(
                "dit tree {key}: shape {:?} != [{out_dim}, {in_dim}]",
                v.shape
            )));
        }
        let w = match *v.dtype {
            LazyDType::BF16 => SnapshotRef::Bf16(v.bytes),
            LazyDType::F32 => SnapshotRef::F32(v.bytes),
            ref d => return Err(msg(format!("dit tree {key}: dtype {d:?}"))),
        };
        Linear::from_snapshot(in_dim, out_dim, w, bias).map_err(|e| msg(format!("{prefix}: {e}")))
    }

    /// A pinned F32 tensor of `shape`.
    pub fn pinned(&self, key: &str, shape: &[usize]) -> Result<CudaTensor> {
        let (got, v) = self.f32s(key)?;
        if got != shape {
            return Err(msg(format!("dit tree {key}: shape {got:?} != {shape:?}")));
        }
        let mut t = CudaTensor::from_vec(v, got)?;
        t.pin_device()?;
        Ok(t)
    }

    /// Raw F32 values with their shape.
    pub fn values(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        self.f32s(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wan::quant::{QuantKind, Section};

    fn ramp(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed) as f32 / u32::MAX as f32;
                (x - 0.5) * 0.2
            })
            .collect()
    }

    #[test]
    fn layout_round_trips() {
        for kind in [QuantKind::W8A8, QuantKind::Mxfp8] {
            let l = QuantLayout::new(
                kind,
                64,
                vec![
                    Section { rows: 48, quantized: true },
                    Section { rows: 16, quantized: false },
                ],
            )
            .unwrap();
            let d = QuantLayout::decode(&l.encode()).unwrap();
            assert_eq!((d.kind, d.in_dim, d.out_dim, d.blob_bytes), (l.kind, l.in_dim, l.out_dim, l.blob_bytes));
            assert_eq!(d.sections, l.sections);
        }
        assert!(QuantLayout::decode(&[0u8; 12]).is_err());
    }

    #[test]
    fn tree_mode_parses() {
        assert_eq!(TreeMode::parse("").unwrap(), TreeMode::Auto);
        assert_eq!(TreeMode::parse("off").unwrap(), TreeMode::Off);
        assert_eq!(TreeMode::parse("0").unwrap(), TreeMode::Off);
        assert_eq!(TreeMode::parse("require").unwrap(), TreeMode::Require);
        assert!(TreeMode::parse("maybe").is_err());
    }

    /// Sample tensors through the whole path, on whatever device the run
    /// has: quantize a linear as the load does (MXFP8 with a bf16 gate
    /// section, W8A8 per section, and a plain one), store it, read it back,
    /// and require the same bytes and the same GEMM output.
    #[test]
    fn stored_linears_match_load_time_quantization() {
        let dir = std::env::temp_dir().join(format!("fv-dit-tree-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (k, n) = (64usize, 96usize);
        let cases: Vec<(&str, Option<(QuantKind, Vec<Section>)>, bool)> = vec![
            (
                "mx",
                Some((
                    QuantKind::Mxfp8,
                    vec![Section { rows: 64, quantized: true }, Section { rows: 32, quantized: false }],
                )),
                false,
            ),
            (
                "w8",
                Some((
                    QuantKind::W8A8,
                    vec![
                        Section { rows: 32, quantized: true },
                        Section { rows: 32, quantized: true },
                        Section { rows: 32, quantized: true },
                    ],
                )),
                true,
            ),
            ("plain", None, true),
        ];
        let mut w = TreeWriter::default();
        let mut originals = Vec::new();
        for (i, (name, q, bias)) in cases.iter().enumerate() {
            let weight = CudaTensor::from_vec(ramp(n * k, i as u32), vec![n, k]).unwrap();
            let b = bias.then(|| CudaTensor::from_vec(ramp(n, 99 + i as u32), vec![n]).unwrap());
            let mut l = Linear::from_tensors(weight, b).unwrap();
            if let Some((kind, sections)) = q {
                l.quantize(*kind, sections.clone()).unwrap();
            }
            w.linear(name, &l).unwrap();
            originals.push(l);
        }
        let norm = CudaTensor::from_vec(ramp(k, 7), vec![k]).unwrap();
        w.tensor("norm.weight", &norm).unwrap();
        let digest = w.digest();
        assert_eq!(w.quantized_linears, 2);
        let file = w.write(&dir, &[("format", FORMAT)]).unwrap();
        assert_eq!(file.bytes, std::fs::metadata(dir.join(MODEL_FILE)).unwrap().len());
        write_manifest(
            &dir,
            &Manifest {
                format: FORMAT.into(),
                version: VERSION,
                identity: sample_identity(),
                source: Vec::new(),
                files: vec![file],
                tensors: w.tensors.len(),
                quantized_linears: w.quantized_linears,
                digest: digest.clone(),
                created_by: String::new(),
            },
        )
        .unwrap();
        assert!(w.write(&dir, &[]).is_err(), "a written tree is never overwritten");

        let r = TreeReader::open(&dir).unwrap();
        let mut again = TreeWriter::default();
        let x = ramp(5 * k, 3);
        for ((name, _, _), orig) in cases.iter().zip(&originals) {
            let l = r.linear(name, k, n).unwrap();
            assert_eq!(l.quant_kind(), orig.quant_kind(), "{name}");
            again.linear(name, &l).unwrap();
            let xs = CudaTensor::from_vec(x.clone(), vec![1, 5, k]).unwrap();
            let (a, b) = (orig.forward(&xs).unwrap(), l.forward(&xs).unwrap());
            let (a, b) = (a.host_cow().unwrap().into_owned(), b.host_cow().unwrap().into_owned());
            assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "{name}: outputs differ");
        }
        again.tensor("norm.weight", &r.pinned("norm.weight", &[k]).unwrap()).unwrap();
        assert_eq!(again.digest(), digest, "read-back state differs from the stored one");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn sample_identity() -> Identity {
        Identity {
            recipe: "4step-vsa".into(),
            quant: "mxfp8".into(),
            with_gate: true,
            num_layers: 50,
            num_refiner_layers: 2,
            hidden_size: 5376,
            timesteps: vec![[1, 2], [3, 4]],
            keyframe_t: 5,
            adapter: Some(FileEntry {
                name: "FastH3-4-step-Preview-v1-LoRA/vsa-datafree/adapter_model.safetensors".into(),
                bytes: 10,
                sha256: "ab".into(),
            }),
            adapter_scale: 1.0f32.to_bits(),
            linear_route: "device-bf16".into(),
        }
    }

    #[test]
    fn identity_mismatch_names_the_field() {
        let a = sample_identity();
        let mut b = a.clone();
        b.adapter.as_mut().unwrap().sha256.clear();
        assert!(a.matches(&b).is_ok(), "the adapter hash is not compared at load");
        b.quant = "w8a8".into();
        let e = a.matches(&b).unwrap_err();
        assert!(e.contains("quant") && e.contains("w8a8"), "{e}");
        let mut c = a.clone();
        c.adapter.as_mut().unwrap().bytes = 11;
        assert!(a.matches(&c).unwrap_err().contains("adapter"));
    }

    #[test]
    fn check_rejects_changed_sources_and_files() {
        let root = std::env::temp_dir().join(format!("fv-dit-check-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let tf = root.join("transformer");
        let tree = root.join("tree");
        std::fs::create_dir_all(&tf).unwrap();
        std::fs::create_dir_all(&tree).unwrap();
        std::fs::write(tf.join("a.safetensors"), b"12345").unwrap();
        std::fs::write(tree.join(MODEL_FILE), b"xyz").unwrap();
        let m = Manifest {
            format: FORMAT.into(),
            version: VERSION,
            identity: sample_identity(),
            source: source_entries(&tf).unwrap(),
            files: vec![entry(&tree.join(MODEL_FILE), MODEL_FILE.into(), String::new()).unwrap()],
            tensors: 0,
            quantized_linears: 0,
            digest: "00".into(),
            created_by: String::new(),
        };
        write_manifest(&tree, &m).unwrap();
        assert!(check(&tf, &tree, &sample_identity()).is_ok());
        let mut other = sample_identity();
        other.recipe = "sol-h3".into();
        assert!(check(&tf, &tree, &other).unwrap_err().contains("recipe"));
        std::fs::write(tf.join("a.safetensors"), b"123456").unwrap();
        assert!(check(&tf, &tree, &sample_identity()).unwrap_err().contains("source shards"));
        std::fs::write(tf.join("a.safetensors"), b"12345").unwrap();
        std::fs::write(tree.join(MODEL_FILE), b"xy").unwrap();
        assert!(check(&tf, &tree, &sample_identity()).unwrap_err().contains("bytes"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
