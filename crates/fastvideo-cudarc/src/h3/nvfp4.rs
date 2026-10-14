//! FastH3 Trim's published NVFP4 DiT (R2 tree `fasth3-trim-8step-nvfp4`,
//! `diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors`;
//! docs/ports/h3-trim.md §5).
//!
//! The release is a ComfyUI single file. Each of a block's five big linears
//! is a comfy-kitchen NVFP4 tensor (`comfy_quant` = `{"format": "nvfp4"}`):
//!
//! | key | dtype, shape | meaning |
//! |---|---|---|
//! | `<l>.weight` | U8 `[out, in/2]` | E2M1 codes, **the even element in the high nibble** |
//! | `<l>.weight_scale` | F8_E4M3 `[out, in/16]` | per-16 block scales, **already in the 128x4 tiled (`to_blocked`) layout** cuBLASLt reads (`out` a multiple of 128, `in/16` of 4) |
//! | `<l>.weight_scale_2` | F32 scalar | the tensor's global decode factor (about `amax / (6 · 448)`) |
//!
//! so `w[r, c] = e2m1(code) · e4m3(scale[r, c/16]) · weight_scale_2`. Measured
//! against the bf16 Trim checkpoint (block 0 `to_out.0`, every 128-row tile
//! of blocks 0 / 20 / 41 sampled): rel-L2 0.093-0.101 with this reading, the
//! same as our own TE `static_6` quantization of the bf16 weight (0.095);
//! low-nibble-first or a row-major scale reading gives 0.72-1.45.
//!
//! Block `i` maps onto the diffusers names the port loads:
//!
//! | file (`blocks.{i}.`) | ours (`transformer_blocks.{i}.`) |
//! |---|---|
//! | `attn.qkv_proj` `[3·inner, hidden]` | `attn.to_q` / `to_k` / `to_v`, in that order (fused `qkvg` rows) |
//! | `attn.to_gate_compress` | `attn.to_gate_compress` (`qkvg`'s gate rows, own global scale) |
//! | `attn.out_proj` | `attn.to_out.0` |
//! | `mlp.fc1` `[2·ffn, hidden]` | `ff.net.0.proj` with its two halves **swapped** (file rows `ffn..2·ffn` are ours `0..ffn`) |
//! | `mlp.fc2` | `ff.net.2` |
//! | `norm1` / `norm2`, `attn.q_norm` / `attn.k_norm` (BF16) | `norm1` / `norm2`, `attn.norm_q` / `attn.norm_k` |
//!
//! Everything else (patch projections, text refiner, AdaLN, output head) is
//! bf16 / F32 in the file under ComfyUI names and comes from the bf16 Trim
//! tree, which the load needs anyway for `transformer/config.json`, the
//! schedule and the VAE; [`Nvfp4Dit::open`] checks the two are the same
//! model (every block's norms bit-identical).
//!
//! The codes are converted to cuBLASLt's low-nibble-first order on the host
//! (one rotate per byte); the scales and the global factor are used as
//! published (`Nvfp4Operand::from_host_layout`). The activation is quantized
//! per call with the TE `static_6` rule (a dynamic per-tensor scale, as
//! comfy-kitchen does without a stored `input_scale`). The GEMM is cuBLASLt
//! `VEC16_UE4M3`, sm_100 / sm_120 only: [`require_tensor_cores`] fails the load
//! on any other GPU (no bf16 fallback).

use std::path::{Path, PathBuf};

use fastvideo_loader::{LazyDType, LazyStore};
use fastvideo_models::h3::config::H3TransformerConfig;

use crate::wan::tensor::{Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// The R2 tree (`weights-manifest.tsv`) holding the file.
pub const TREE: &str = "fasth3-trim-8step-nvfp4";
/// Its directory inside the tree.
pub const SUBDIR: &str = "diffusion_models";
/// The published file.
pub const FILE: &str = "fastvideo_fasth3_trim_8step_nvfp4.safetensors";
/// `comfy_quant` of every quantized linear.
pub const COMFY_FORMAT: &str = "nvfp4";

/// The NVFP4 recipe names (the V2 contract, served from the NVFP4 file).
pub fn is_nvfp4_recipe(recipe: &str) -> bool {
    fastvideo_models::h3::config::is_trim_nvfp4_recipe(recipe)
}

/// The file at `p`: the file itself, or `<p>/diffusion_models/<FILE>` / `<p>/<FILE>` for a directory.
pub fn resolve_file(p: &Path) -> Result<PathBuf> {
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    for c in [p.join(SUBDIR).join(FILE), p.join(FILE)] {
        if c.is_file() {
            return Ok(c);
        }
    }
    Err(msg(format!(
        "h3 nvfp4: no {FILE} at {} (the R2 tree {TREE}: {SUBDIR}/{FILE})",
        p.display()
    )))
}

/// Where an NVFP4 recipe looks when no file is given: the tree beside the
/// Trim's (`$FV_WEIGHTS/fasth3-trim-8step-nvfp4` next to `$FV_WEIGHTS/fasth3-trim-8step`).
pub fn sibling_tree(root: &Path) -> PathBuf {
    let root = Path::new(root.to_str().map_or(root.as_os_str(), |s| s.trim_end_matches('/').as_ref()));
    root.parent().unwrap_or(Path::new(".")).join(TREE)
}

/// Fail unless the device has NVFP4 tensor cores (sm_100 / sm_120 family).
/// There is no fallback: the published weight has no bf16 form to run.
pub fn require_tensor_cores() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let Some(d) = crate::wan::device::global_device() else {
            return Err(msg("h3 nvfp4: needs a CUDA device (NVFP4 tensor cores, sm_100 / sm_120)"));
        };
        if d.sm_major < 10 {
            return Err(msg(format!(
                "h3 nvfp4: this GPU is sm_{}{}; the published NVFP4 DiT runs only on NVFP4 tensor cores (sm_100 / sm_120: B200, RTX PRO 6000, RTX 50xx). No bf16 fallback: serve the bf16 / MXFP8 Trim instead",
                d.sm_major, d.sm_minor
            )));
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(msg("h3 nvfp4: needs a CUDA build and NVFP4 tensor cores (sm_100 / sm_120)"))
    }
}

/// Byte offset of block scale `(r, c)` in the 128x4 tiled layout of a
/// `[rows, sc]` scale matrix (`sc % 4 == 0`): 128-row x 4-column tiles in
/// row-major tile order, 512 bytes each, `(r % 32) · 16 + (r % 128) / 32 · 4 + c % 4` inside.
pub fn blocked_offset(r: usize, c: usize, sc: usize) -> usize {
    let tile = (r / 128) * (sc / 4) + c / 4;
    let rr = r % 128;
    tile * 512 + (rr % 32) * 16 + (rr / 32) * 4 + c % 4
}

/// The comfy-kitchen code order (even element high) to cuBLASLt's (even
/// element low): swap the nibbles of every byte.
pub fn swap_nibbles(codes: &mut [u8]) {
    for b in codes {
        *b = b.rotate_left(4);
    }
}

/// One published linear on the host, in cuBLASLt's operand layout.
#[derive(Clone, Debug, PartialEq)]
pub struct PublishedLinear {
    /// `[rows, k/2]`, the even element in the **low** nibble.
    pub codes: Vec<u8>,
    /// 128x4 tiled E4M3 block scales, `rows` padded to 128 by `k/16`.
    pub scales: Vec<u8>,
    pub rows: usize,
    pub k: usize,
    /// `weight_scale_2`.
    pub decode_scale: f32,
}

impl PublishedLinear {
    /// `prefix.{weight, weight_scale, weight_scale_2, comfy_quant}` from the
    /// store, checked (dtypes, shapes, format) and converted.
    pub fn read(store: &LazyStore, prefix: &str, rows: usize, k: usize) -> Result<Self> {
        let view = |name: &str| {
            store
                .view(&format!("{prefix}.{name}"))
                .map_err(|e| msg(format!("h3 nvfp4: {e}")))
        };
        let shape_err = |name: &str, dtype: &LazyDType, shape: &[usize], want: &str| {
            msg(format!(
                "h3 nvfp4: {prefix}.{name} is {dtype:?} {shape:?}, expected {want}"
            ))
        };
        if !k.is_multiple_of(64) || !rows.is_multiple_of(16) {
            return Err(msg(format!("h3 nvfp4: {prefix}: [{rows}, {k}] (rows % 16, k % 64)")));
        }
        let q = view("comfy_quant")?;
        let fmt: serde_json::Value = serde_json::from_slice(q.bytes)
            .map_err(|e| msg(format!("h3 nvfp4: {prefix}.comfy_quant: {e}")))?;
        if fmt.get("format").and_then(|v| v.as_str()) != Some(COMFY_FORMAT) {
            return Err(msg(format!(
                "h3 nvfp4: {prefix}.comfy_quant = {fmt}, expected {{\"format\": \"{COMFY_FORMAT}\"}}"
            )));
        }
        let w = view("weight")?;
        if *w.dtype != LazyDType::U8 || w.shape != [rows, k / 2] {
            return Err(shape_err("weight", w.dtype, w.shape, &format!("U8 [{rows}, {}]", k / 2)));
        }
        let rows_scale = rows.next_multiple_of(128);
        let s = view("weight_scale")?;
        if *s.dtype != LazyDType::F8E4M3 || s.shape != [rows_scale, k / 16] {
            return Err(shape_err(
                "weight_scale",
                s.dtype,
                s.shape,
                &format!("F8_E4M3 [{rows_scale}, {}]", k / 16),
            ));
        }
        let g = view("weight_scale_2")?;
        if *g.dtype != LazyDType::F32 || g.numel() != 1 || g.bytes.len() != 4 {
            return Err(shape_err("weight_scale_2", g.dtype, g.shape, "F32 scalar"));
        }
        let decode_scale = f32::from_le_bytes(g.bytes.try_into().expect("4 bytes"));
        if !(decode_scale.is_finite() && decode_scale > 0.0) {
            return Err(msg(format!(
                "h3 nvfp4: {prefix}.weight_scale_2 = {decode_scale}"
            )));
        }
        let mut codes = w.bytes.to_vec();
        swap_nibbles(&mut codes);
        Ok(Self {
            codes,
            scales: s.bytes.to_vec(),
            rows,
            k,
            decode_scale,
        })
    }

    /// Swap the first and second halves of the rows (`mlp.fc1` to
    /// `ff.net.0.proj`). Exact: the halves share the global factor, and a
    /// half that is a whole number of 128-row tiles moves its scale tiles
    /// as contiguous bytes.
    pub fn swap_row_halves(&mut self) -> Result<()> {
        let half = self.rows / 2;
        if self.rows % 2 != 0 || !half.is_multiple_of(128) {
            return Err(msg(format!(
                "h3 nvfp4: cannot swap the halves of {} rows (each a multiple of 128)",
                self.rows
            )));
        }
        let cb = half * self.k / 2;
        self.codes.rotate_left(cb);
        let sb = half * self.k / 16;
        self.scales.rotate_left(sb);
        Ok(())
    }

    /// Dequantized `[rows, k]` (host; tests and checks).
    pub fn dequantize(&self) -> Vec<f32> {
        let sc = self.k / 16;
        let mut out = vec![0f32; self.rows * self.k];
        for r in 0..self.rows {
            for c in 0..self.k {
                let byte = self.codes[r * self.k / 2 + c / 2];
                let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                let s = crate::wan::quant::e4m3_decode(self.scales[blocked_offset(r, c / 16, sc)]);
                out[r * self.k + c] = fastvideo_models::nvfp4::e2m1_to_f32(code) * s * self.decode_scale;
            }
        }
        out
    }
}

/// The published file of one load: its store, checked against the config
/// and the bf16 Trim tree the rest of the DiT comes from.
pub struct Nvfp4Dit {
    /// The file, with the diffusers names of the block norms aliased onto
    /// its own (`transformer_blocks.{i}.attn.norm_q` → `blocks.{i}.attn.q_norm`).
    map: WeightMap,
    path: PathBuf,
}

/// `blocks.{i}.*` (ComfyUI) for a diffusers block key, for the tensors the
/// load reads from the file.
fn comfy_block_key(key: &str) -> Option<String> {
    let rest = key.strip_prefix("transformer_blocks.")?;
    let (i, name) = rest.split_once('.')?;
    let name = match name {
        "norm1.weight" | "norm2.weight" => name.to_owned(),
        "attn.norm_q.weight" => "attn.q_norm.weight".to_owned(),
        "attn.norm_k.weight" => "attn.k_norm.weight".to_owned(),
        _ => return None,
    };
    Some(format!("blocks.{i}.{name}"))
}

/// The five published linears of block `i`: `(file prefix, rows, k)`.
pub fn block_linears(cfg: &H3TransformerConfig, i: usize) -> [(String, usize, usize); 5] {
    let (h, inner, ffn) = (cfg.hidden_size, cfg.inner_dim(), cfg.ffn_dim);
    [
        (format!("blocks.{i}.attn.qkv_proj"), 3 * inner, h),
        (format!("blocks.{i}.attn.to_gate_compress"), inner, h),
        (format!("blocks.{i}.attn.out_proj"), h, inner),
        (format!("blocks.{i}.mlp.fc1"), 2 * ffn, h),
        (format!("blocks.{i}.mlp.fc2"), h, ffn),
    ]
}

impl Nvfp4Dit {
    /// Open and check the file: `cfg.num_layers` blocks and no more, every
    /// published linear's dtypes / shapes / format, and every block's norms
    /// bit-identical to `trim`'s (the bf16 Trim tree of this load).
    pub fn open(path: &Path, cfg: &H3TransformerConfig, trim: &WeightMap) -> Result<Self> {
        let path = resolve_file(path)?;
        let map = WeightMap::open_files(std::slice::from_ref(&path))?.with_alias(comfy_block_key);
        let store = map
            .lazy()
            .ok_or_else(|| msg("h3 nvfp4: the file is not mapped"))?;
        if store.contains(&format!("blocks.{}.mlp.fc1.weight", cfg.num_layers)) {
            return Err(msg(format!(
                "h3 nvfp4: {} has more than {} blocks",
                path.display(),
                cfg.num_layers
            )));
        }
        for i in 0..cfg.num_layers {
            for (prefix, rows, k) in block_linears(cfg, i) {
                // Headers only: the views are not read here.
                for (name, dtype, shape) in [
                    ("weight", LazyDType::U8, vec![rows, k / 2]),
                    ("weight_scale", LazyDType::F8E4M3, vec![rows.next_multiple_of(128), k / 16]),
                    ("comfy_quant", LazyDType::U8, Vec::new()),
                ] {
                    let key = format!("{prefix}.{name}");
                    let v = store
                        .view(&key)
                        .map_err(|e| msg(format!("h3 nvfp4: {} is not the Trim NVFP4 file: {e}", path.display())))?;
                    if *v.dtype != dtype || (!shape.is_empty() && v.shape != shape.as_slice()) {
                        return Err(msg(format!(
                            "h3 nvfp4: {key} is {:?} {:?}, expected {dtype:?} {shape:?}",
                            v.dtype, v.shape
                        )));
                    }
                }
            }
            for name in ["norm1.weight", "norm2.weight", "attn.norm_q.weight", "attn.norm_k.weight"] {
                let key = format!("transformer_blocks.{i}.{name}");
                let (a, b) = (map.get_bf16_bytes(&key)?, trim.get_bf16_bytes(&key)?);
                if a != b {
                    return Err(msg(format!(
                        "h3 nvfp4: {key} differs between {} and the bf16 Trim tree: not the same model",
                        path.display()
                    )));
                }
            }
        }
        Ok(Self { map, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file as a map: the block norms under their diffusers names.
    pub fn map(&self) -> &WeightMap {
        &self.map
    }

    /// Block `i`'s published linear `prefix` on the host, in our row order.
    pub fn read_linear(&self, prefix: &str, rows: usize, k: usize) -> Result<PublishedLinear> {
        let store = self
            .map
            .lazy()
            .ok_or_else(|| msg("h3 nvfp4: the file is not mapped"))?;
        let mut l = PublishedLinear::read(store, prefix, rows, k)?;
        if prefix.ends_with(".mlp.fc1") {
            l.swap_row_halves()?;
        }
        Ok(l)
    }

    /// Bytes of the published linears (what the device holds for them).
    pub fn linear_bytes(&self, cfg: &H3TransformerConfig, gate: bool) -> u64 {
        (0..cfg.num_layers)
            .flat_map(|i| block_linears(cfg, i))
            .filter(|(p, _, _)| gate || !p.ends_with(".to_gate_compress"))
            .map(|(_, rows, k)| (rows * k / 2 + rows.next_multiple_of(128) * k / 16) as u64)
            .sum()
    }

    /// Block `i`'s four linears on the device, as the block runs them:
    /// `qkvg` (`[q; k; v]`, then the gate rows when `gate`), `to_out`,
    /// `ff_in`, `ff_out`.
    #[cfg(feature = "cuda")]
    pub fn device_linears(
        &self,
        cfg: &H3TransformerConfig,
        i: usize,
        gate: bool,
    ) -> Result<[crate::wan::nn::Linear; 4]> {
        use crate::wan::nn::Linear;
        use crate::wan::nvfp4_linear::{Nvfp4Linear, Nvfp4Operand, Nvfp4Stack};
        let rule = fastvideo_models::nvfp4::ScaleRule::Static6;
        let dev = |l: PublishedLinear| -> Result<Nvfp4Linear> {
            let op = Nvfp4Operand::from_host_layout(&l.codes, &l.scales, l.rows, l.k, l.decode_scale, rule)?;
            Nvfp4Linear::from_operand(op, None, rule)
        };
        let [qkv, g, out, fc1, fc2] = block_linears(cfg, i);
        let read = |(p, rows, k): &(String, usize, usize)| self.read_linear(p, *rows, *k);
        let mut parts = vec![dev(read(&qkv)?)?];
        if gate {
            parts.push(dev(read(&g)?)?);
        }
        Ok([
            Linear::from_nvfp4(Nvfp4Stack::new(parts)?)?,
            Linear::from_nvfp4(Nvfp4Stack::new(vec![dev(read(&out)?)?])?)?,
            Linear::from_nvfp4(Nvfp4Stack::new(vec![dev(read(&fc1)?)?])?)?,
            Linear::from_nvfp4(Nvfp4Stack::new(vec![dev(read(&fc2)?)?])?)?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::nvfp4::{quantize, ScaleRule};

    /// Our TE quantization of `w`, written as the release writes it
    /// (high nibble first, blocked scales, `weight_scale_2`), and the
    /// file bytes for `prefix`.
    fn publish(w: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>, f32, Vec<f32>) {
        let t = quantize(w, rows, k, ScaleRule::Static6).unwrap();
        let reference = t.dequantize();
        let sc = k / 16;
        let mut blocked = vec![0u8; rows.next_multiple_of(128) * sc];
        for r in 0..rows {
            for c in 0..sc {
                blocked[blocked_offset(r, c, sc)] = t.scales[r * sc + c];
            }
        }
        let mut codes = t.packed.clone();
        swap_nibbles(&mut codes);
        let ds = fastvideo_models::nvfp4::dequant_factor(t.amax, ScaleRule::Static6);
        (codes, blocked, ds, reference)
    }

    fn header_file(dir: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
            );
            data.extend_from_slice(bytes);
        }
        let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        f.extend(data);
        let p = dir.join("t.safetensors");
        std::fs::write(&p, f).unwrap();
        p
    }

    fn weights(rows: usize, k: usize, seed: f32) -> Vec<f32> {
        (0..rows * k)
            .map(|i| ((i as f32 * 0.37 + seed).sin() * 0.05) * if i % 97 == 0 { 8.0 } else { 1.0 })
            .collect()
    }

    /// A temporary directory removed on drop.
    struct TmpDir(PathBuf);

    impl TmpDir {
        fn new() -> Self {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!("fv-h3-nvfp4-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn blocked_offset_matches_a_tile_reshape() {
        // to_blocked as a reshape: [rows/128, 4, 32, sc/4, 4] -> [rows/128, sc/4, 32, 4, 4].
        let (rows, sc) = (384, 12);
        let mut seen = vec![false; rows * sc];
        for r in 0..rows {
            for c in 0..sc {
                let (rt, r4, r32, ct, c4) = (r / 128, (r % 128) / 32, r % 32, c / 4, c % 4);
                let want = (((rt * (sc / 4) + ct) * 32 + r32) * 4 + r4) * 4 + c4;
                assert_eq!(blocked_offset(r, c, sc), want);
                seen[want] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn blocked_offset_is_cublas_to_blocked() {
        // The cuBLASLt operand quantizer's layout (wan/nvfp4_gemm.rs swizzle_scales_host).
        let (rows, sc) = (256, 8);
        let linear: Vec<u8> = (0..rows * sc).map(|i| (i % 251) as u8).collect();
        let sw = crate::wan::nvfp4_gemm::swizzle_scales_host(&linear, rows, sc);
        for r in 0..rows {
            for c in 0..sc {
                assert_eq!(sw[blocked_offset(r, c, sc)], linear[r * sc + c]);
            }
        }
    }

    #[test]
    fn published_layout_dequantizes_to_the_quantized_weight() {
        let (rows, k) = (256, 128);
        let w = weights(rows, k, 0.0);
        let (codes, scales, ds, reference) = publish(&w, rows, k);
        let tmp = TmpDir::new();
        let p = header_file(
            tmp.path(),
            &[
                ("l.weight", "U8", vec![rows, k / 2], codes),
                ("l.weight_scale", "F8_E4M3", vec![rows, k / 16], scales),
                ("l.weight_scale_2", "F32", vec![], ds.to_le_bytes().to_vec()),
                ("l.comfy_quant", "U8", vec![19], br#"{"format": "nvfp4"}"#.to_vec()),
            ],
        );
        let store = LazyStore::open_files(&[p]).unwrap();
        let l = PublishedLinear::read(&store, "l", rows, k).unwrap();
        assert_eq!(l.decode_scale, ds);
        let got = l.dequantize();
        for (i, (a, b)) in got.iter().zip(&reference).enumerate() {
            assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-6), "{i}: {a} vs {b}");
        }
        // The swapped halves are the other half's rows, exactly.
        let mut s = l.clone();
        s.swap_row_halves().unwrap();
        let sw = s.dequantize();
        assert_eq!(&sw[..rows / 2 * k], &got[rows / 2 * k..]);
        assert_eq!(&sw[rows / 2 * k..], &got[..rows / 2 * k]);
    }

    #[test]
    fn read_refuses_other_formats_and_shapes() {
        let (rows, k) = (128, 64);
        let (codes, scales, ds, _) = publish(&weights(rows, k, 1.0), rows, k);
        let tmp = TmpDir::new();
        let p = header_file(
            tmp.path(),
            &[
                ("l.weight", "U8", vec![rows, k / 2], codes.clone()),
                ("l.weight_scale", "F8_E4M3", vec![rows, k / 16], scales.clone()),
                ("l.weight_scale_2", "F32", vec![], ds.to_le_bytes().to_vec()),
                ("l.comfy_quant", "U8", vec![19], br#"{"format": "mxfp8"}"#.to_vec()),
                ("m.weight", "U8", vec![rows, k / 2], codes),
                ("m.weight_scale", "U8", vec![rows, k / 16], scales),
                ("m.weight_scale_2", "F32", vec![], ds.to_le_bytes().to_vec()),
                ("m.comfy_quant", "U8", vec![19], br#"{"format": "nvfp4"}"#.to_vec()),
            ],
        );
        let store = LazyStore::open_files(&[p]).unwrap();
        let e = PublishedLinear::read(&store, "l", rows, k).unwrap_err().to_string();
        assert!(e.contains("comfy_quant"), "{e}");
        let e = PublishedLinear::read(&store, "m", rows, k).unwrap_err().to_string();
        assert!(e.contains("weight_scale"), "{e}");
        let e = PublishedLinear::read(&store, "l", rows, 2 * k).unwrap_err().to_string();
        assert!(e.contains("weight_scale") || e.contains("comfy_quant") || e.contains("weight"), "{e}");
    }

    #[test]
    fn nibble_order_and_names() {
        let mut b = [0x12u8, 0xf0];
        swap_nibbles(&mut b);
        assert_eq!(b, [0x21, 0x0f]);
        assert_eq!(
            comfy_block_key("transformer_blocks.7.attn.norm_q.weight").as_deref(),
            Some("blocks.7.attn.q_norm.weight")
        );
        assert_eq!(comfy_block_key("transformer_blocks.7.norm2.weight").as_deref(), Some("blocks.7.norm2.weight"));
        assert_eq!(comfy_block_key("transformer_blocks.7.attn.to_q.weight"), None);
        assert!(is_nvfp4_recipe("trim-8step-nvfp4") && is_nvfp4_recipe("fasth3-trim-nvfp4"));
        assert!(!is_nvfp4_recipe("trim-8step"));
        assert_eq!(sibling_tree(Path::new("/w/fasth3-trim-8step/")), Path::new("/w").join(TREE));
    }

    #[test]
    fn trim_linear_shapes() {
        let cfg = H3TransformerConfig::fasth3_trim_8step();
        let l = block_linears(&cfg, 41);
        let shapes: Vec<(usize, usize)> = l.iter().map(|(_, r, k)| (*r, *k)).collect();
        // The published file's headers: qkv [21504, 2688], gate [7168, 2688],
        // out [5376, 3584], fc1 [28672, 2688], fc2 [5376, 7168] (U8 = k / 2).
        assert_eq!(shapes, [(21504, 5376), (7168, 5376), (5376, 7168), (28672, 5376), (5376, 14336)]);
        assert_eq!(l[3].0, "blocks.41.mlp.fc1");
    }
}
