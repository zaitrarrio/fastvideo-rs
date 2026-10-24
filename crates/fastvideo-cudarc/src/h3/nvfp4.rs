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

use crate::rt::tensor::{Result, TensorError};
use crate::rt::weights::WeightMap;

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
    let root = Path::new(
        root.to_str()
            .map_or(root.as_os_str(), |s| s.trim_end_matches('/').as_ref()),
    );
    root.parent().unwrap_or(Path::new(".")).join(TREE)
}

/// Fail unless the device has NVFP4 tensor cores (sm_100 / sm_120 family).
/// There is no fallback: the published weight has no bf16 form to run.
pub fn require_tensor_cores() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let Some(d) = crate::rt::device::global_device() else {
            return Err(msg(
                "h3 nvfp4: needs a CUDA device (NVFP4 tensor cores, sm_100 / sm_120)",
            ));
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
        Err(msg(
            "h3 nvfp4: needs a CUDA build and NVFP4 tensor cores (sm_100 / sm_120)",
        ))
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
            return Err(msg(format!(
                "h3 nvfp4: {prefix}: [{rows}, {k}] (rows % 16, k % 64)"
            )));
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
            return Err(shape_err(
                "weight",
                w.dtype,
                w.shape,
                &format!("U8 [{rows}, {}]", k / 2),
            ));
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
        if !self.rows.is_multiple_of(2) || !half.is_multiple_of(128) {
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
                let s = crate::rt::quant::e4m3_decode(self.scales[blocked_offset(r, c / 16, sc)]);
                out[r * self.k + c] =
                    fastvideo_models::nvfp4::e2m1_to_f32(code) * s * self.decode_scale;
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
    layout: Nvfp4Layout,
}

/// Which release a [`Nvfp4Dit`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Layout {
    /// FastH3 Trim's ComfyUI single file: all five block linears NVFP4.
    Comfy,
    /// ModelOpt's unified Hugging Face layout (FastH3 V2 / 4-step NVFP4
    /// releases): the FFN NVFP4 with calibrated activation scales, attention
    /// bf16 in the same `transformer/` ([`ModelOptOptions`]).
    ModelOpt(ModelOptOptions),
}

/// How a ModelOpt NVFP4 DiT runs (docs/ports/fasth3-nvfp4.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelOptOptions {
    /// Quantize the bf16 attention projections (`to_q/k/v/out`, and the VSA
    /// `to_gate_compress` when loaded) to NVFP4 at load, per linear with the
    /// TE `static_6` rule (FastVideo's `--quantize-attention --quantize-gate`,
    /// `layer_profile = h3_dit_vsa`). `false`: they follow
    /// `FASTVIDEO_H3_QUANT` like any bf16 block (`h3_dit_ffn`).
    pub attention: bool,
    /// Quantize the FFN activations with the checkpoint's calibrated static
    /// `input_scale` (as FastVideo's export keeps it); `false` measures the
    /// amax on every call.
    pub static_act: bool,
}

/// `FASTVIDEO_H3_NVFP4_ATTN=1`: [`ModelOptOptions::attention`].
pub const ATTN_ENV: &str = "FASTVIDEO_H3_NVFP4_ATTN";
/// `FASTVIDEO_H3_NVFP4_ACT=static|dynamic` (default static):
/// [`ModelOptOptions::static_act`].
pub const ACT_ENV: &str = "FASTVIDEO_H3_NVFP4_ACT";

impl ModelOptOptions {
    pub fn from_env() -> Result<Self> {
        Self::from_settings(&fastvideo_models::settings::Live)
    }

    /// [`Self::from_env`] with [`ATTN_ENV`] and [`ACT_ENV`] from `settings`
    /// (a load's snapshot).
    pub fn from_settings(settings: &dyn fastvideo_models::settings::Source) -> Result<Self> {
        let attention = match settings.var(ATTN_ENV).as_deref().map(str::trim) {
            None | Some("" | "0" | "off" | "false" | "bf16") => false,
            Some("1" | "on" | "true" | "nvfp4") => true,
            Some(other) => return Err(msg(format!("{ATTN_ENV}={other}: expected 0|1"))),
        };
        let static_act = match settings.var(ACT_ENV).as_deref().map(str::trim) {
            None | Some("" | "static") => true,
            Some("dynamic") => false,
            Some(other) => return Err(msg(format!("{ACT_ENV}={other}: expected static|dynamic"))),
        };
        Ok(Self {
            attention,
            static_act,
        })
    }
}

/// Whether `transformer_dir/config.json` declares ModelOpt NVFP4
/// (`quantization_config.quant_method = "modelopt"`, `quant_algo = "NVFP4"`).
/// No `quantization_config`, or one of another kind, is `false` (the loads
/// that existed before keep their path); ModelOpt with another algorithm is
/// refused.
pub fn is_modelopt_dir(transformer_dir: &Path) -> Result<bool> {
    let path = transformer_dir.join("config.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(false);
    };
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| msg(format!("{}: {e}", path.display())))?;
    let Some(q) = v.get("quantization_config") else {
        return Ok(false);
    };
    let method = q.get("quant_method").and_then(|x| x.as_str());
    let algo = q.get("quant_algo").and_then(|x| x.as_str());
    match (method, algo) {
        (Some("modelopt"), Some("NVFP4")) => Ok(true),
        (Some("modelopt"), _) => Err(msg(format!(
            "h3 nvfp4: {} declares quantization_config {method:?} / {algo:?}; only ModelOpt NVFP4 is read",
            path.display()
        ))),
        _ => Ok(false),
    }
}

/// A block's two ModelOpt NVFP4 FFN linears: `(prefix, rows, k)`.
pub fn modelopt_ffn_linears(cfg: &H3TransformerConfig, i: usize) -> [(String, usize, usize); 2] {
    let (h, ffn) = (cfg.hidden_size, cfg.ffn_dim);
    [
        (format!("transformer_blocks.{i}.ff.net.0.proj"), 2 * ffn, h),
        (format!("transformer_blocks.{i}.ff.net.2"), h, ffn),
    ]
}

impl PublishedLinear {
    /// `prefix.{weight, weight_scale, weight_scale_2, input_scale}` in
    /// ModelOpt's unified layout: U8 `[rows, k/2]` codes, **the even element
    /// in the low nibble** (cuBLASLt's order: kept as is); F8_E4M3
    /// `[rows, k/16]` block scales **row-major** (moved into the 128x4 tiled
    /// layout here); F32 scalar `weight_scale_2`; the optional F32 scalar
    /// static activation scale `input_scale`, returned beside.
    pub fn read_modelopt(
        store: &LazyStore,
        prefix: &str,
        rows: usize,
        k: usize,
    ) -> Result<(Self, Option<f32>)> {
        let view = |name: &str| {
            store
                .view(&format!("{prefix}.{name}"))
                .map_err(|e| msg(format!("h3 nvfp4: {e}")))
        };
        let bad = |name: &str, dtype: &LazyDType, shape: &[usize], want: &str| {
            msg(format!(
                "h3 nvfp4: {prefix}.{name} is {dtype:?} {shape:?}, expected {want}"
            ))
        };
        if !k.is_multiple_of(64) || !rows.is_multiple_of(16) {
            return Err(msg(format!(
                "h3 nvfp4: {prefix}: [{rows}, {k}] (rows % 16, k % 64)"
            )));
        }
        let w = view("weight")?;
        if *w.dtype != LazyDType::U8 || w.shape != [rows, k / 2] {
            return Err(bad(
                "weight",
                w.dtype,
                w.shape,
                &format!("U8 [{rows}, {}]", k / 2),
            ));
        }
        let sc = k / 16;
        let s = view("weight_scale")?;
        if *s.dtype != LazyDType::F8E4M3 || s.shape != [rows, sc] {
            return Err(bad(
                "weight_scale",
                s.dtype,
                s.shape,
                &format!("F8_E4M3 [{rows}, {sc}] (row-major)"),
            ));
        }
        let scalar = |name: &str| -> Result<Option<f32>> {
            if !store.contains(&format!("{prefix}.{name}")) {
                return Ok(None);
            }
            let g = view(name)?;
            if *g.dtype != LazyDType::F32 || g.numel() != 1 || g.bytes.len() != 4 {
                return Err(bad(name, g.dtype, g.shape, "F32 scalar"));
            }
            let v = f32::from_le_bytes(g.bytes.try_into().expect("4 bytes"));
            if !(v.is_finite() && v > 0.0) {
                return Err(msg(format!("h3 nvfp4: {prefix}.{name} = {v}")));
            }
            Ok(Some(v))
        };
        let decode_scale = scalar("weight_scale_2")?
            .ok_or_else(|| msg(format!("h3 nvfp4: {prefix}.weight_scale_2 missing")))?;
        let input_scale = scalar("input_scale")?;
        let mut scales = vec![0u8; rows.next_multiple_of(128) * sc];
        for r in 0..rows {
            for c in 0..sc {
                scales[blocked_offset(r, c, sc)] = s.bytes[r * sc + c];
            }
        }
        Ok((
            Self {
                codes: w.bytes.to_vec(),
                scales,
                rows,
                k,
                decode_scale,
            },
            input_scale,
        ))
    }
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
                    (
                        "weight_scale",
                        LazyDType::F8E4M3,
                        vec![rows.next_multiple_of(128), k / 16],
                    ),
                    ("comfy_quant", LazyDType::U8, Vec::new()),
                ] {
                    let key = format!("{prefix}.{name}");
                    let v = store.view(&key).map_err(|e| {
                        msg(format!(
                            "h3 nvfp4: {} is not the Trim NVFP4 file: {e}",
                            path.display()
                        ))
                    })?;
                    if *v.dtype != dtype || (!shape.is_empty() && v.shape != shape.as_slice()) {
                        return Err(msg(format!(
                            "h3 nvfp4: {key} is {:?} {:?}, expected {dtype:?} {shape:?}",
                            v.dtype, v.shape
                        )));
                    }
                }
            }
            for name in [
                "norm1.weight",
                "norm2.weight",
                "attn.norm_q.weight",
                "attn.norm_k.weight",
            ] {
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
        Ok(Self {
            map,
            path,
            layout: Nvfp4Layout::Comfy,
        })
    }

    /// Open a ModelOpt NVFP4 `transformer/` ([`is_modelopt_dir`]): every
    /// block's two FFN linears (dtypes, shapes, scales) and its bf16 attention
    /// projections are checked from the headers; nothing is read.
    pub fn open_modelopt(
        dir: &Path,
        cfg: &H3TransformerConfig,
        options: ModelOptOptions,
    ) -> Result<Self> {
        let map = WeightMap::open(dir)?;
        let store = map
            .lazy()
            .ok_or_else(|| msg(format!("h3 nvfp4: {} is not mapped", dir.display())))?;
        if store.contains(&format!(
            "transformer_blocks.{}.ff.net.2.weight",
            cfg.num_layers
        )) {
            return Err(msg(format!(
                "h3 nvfp4: {} has more than {} blocks",
                dir.display(),
                cfg.num_layers
            )));
        }
        let (h, inner) = (cfg.hidden_size, cfg.inner_dim());
        for i in 0..cfg.num_layers {
            for (prefix, rows, k) in modelopt_ffn_linears(cfg, i) {
                for (name, dtype, shape) in [
                    ("weight", LazyDType::U8, vec![rows, k / 2]),
                    ("weight_scale", LazyDType::F8E4M3, vec![rows, k / 16]),
                    ("weight_scale_2", LazyDType::F32, Vec::new()),
                ] {
                    let key = format!("{prefix}.{name}");
                    let v = store.view(&key).map_err(|e| {
                        msg(format!(
                            "h3 nvfp4: {} is not a ModelOpt NVFP4 H3 transformer: {e}",
                            dir.display()
                        ))
                    })?;
                    if *v.dtype != dtype || (!shape.is_empty() && v.shape != shape.as_slice()) {
                        return Err(msg(format!(
                            "h3 nvfp4: {key} is {:?} {:?}, expected {dtype:?} {shape:?}",
                            v.dtype, v.shape
                        )));
                    }
                }
            }
            for (name, want) in [
                ("to_q", [inner, h]),
                ("to_k", [inner, h]),
                ("to_v", [inner, h]),
                ("to_out.0", [h, inner]),
            ] {
                let key = format!("transformer_blocks.{i}.attn.{name}.weight");
                let v = store
                    .view(&key)
                    .map_err(|e| msg(format!("h3 nvfp4: {e}")))?;
                if v.shape != want
                    || !matches!(v.dtype, LazyDType::BF16 | LazyDType::F16 | LazyDType::F32)
                {
                    return Err(msg(format!(
                        "h3 nvfp4: {key} is {:?} {:?}, expected a float {want:?} (ModelOpt keeps attention dense)",
                        v.dtype, v.shape
                    )));
                }
            }
        }
        Ok(Self {
            map,
            path: dir.to_path_buf(),
            layout: Nvfp4Layout::ModelOpt(options),
        })
    }

    pub fn layout(&self) -> Nvfp4Layout {
        self.layout
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
        let fp4 = |(_, rows, k): (String, usize, usize)| {
            (rows * k / 2 + rows.next_multiple_of(128) * k / 16) as u64
        };
        match self.layout {
            Nvfp4Layout::Comfy => (0..cfg.num_layers)
                .flat_map(|i| block_linears(cfg, i))
                .filter(|(p, _, _)| gate || !p.ends_with(".to_gate_compress"))
                .map(fp4)
                .sum(),
            Nvfp4Layout::ModelOpt(_) => (0..cfg.num_layers)
                .flat_map(|i| modelopt_ffn_linears(cfg, i))
                .map(fp4)
                .sum(),
        }
    }

    /// Block `i`'s two FFN linears (`ff_in`, `ff_out`) from a ModelOpt
    /// checkpoint, each with its calibrated static activation scale when
    /// [`ModelOptOptions::static_act`] (and the checkpoint has one).
    #[cfg(feature = "cuda")]
    pub fn modelopt_ffn(
        &self,
        cfg: &H3TransformerConfig,
        i: usize,
    ) -> Result<[crate::rt::nn::Linear; 2]> {
        use crate::rt::nn::Linear;
        use crate::rt::nvfp4_linear::{Nvfp4Linear, Nvfp4Operand, Nvfp4Stack};
        let Nvfp4Layout::ModelOpt(opts) = self.layout else {
            return Err(msg("h3 nvfp4: modelopt_ffn on a ComfyUI file"));
        };
        let rule = fastvideo_models::nvfp4::ScaleRule::Static6;
        let store = self
            .map
            .lazy()
            .ok_or_else(|| msg("h3 nvfp4: the transformer is not mapped"))?;
        let one = |(p, rows, k): &(String, usize, usize)| -> Result<Linear> {
            let (l, input_scale) = PublishedLinear::read_modelopt(store, p, *rows, *k)?;
            let op = Nvfp4Operand::from_host_layout(
                &l.codes,
                &l.scales,
                l.rows,
                l.k,
                l.decode_scale,
                rule,
            )?;
            let mut stack = Nvfp4Stack::new(vec![Nvfp4Linear::from_operand(op, None, rule)?])?;
            if let (true, Some(s)) = (opts.static_act, input_scale) {
                stack = stack.with_static_act_scale(s)?;
            }
            Linear::from_nvfp4(stack)
        };
        let [fc_in, fc_out] = modelopt_ffn_linears(cfg, i);
        Ok([one(&fc_in)?, one(&fc_out)?])
    }

    /// Block `i`'s attention projections quantized to NVFP4 at load from the
    /// checkpoint's bf16 weights, one global scale per projection (TE
    /// `static_6`: FastVideo's `quantize_dense_linear`): `qkvg` (`[q; k; v]`,
    /// then the gate rows when `gate`) and `to_out`. The activation scale is
    /// measured per call.
    #[cfg(feature = "cuda")]
    pub fn modelopt_attention(
        &self,
        cfg: &H3TransformerConfig,
        i: usize,
        gate: bool,
    ) -> Result<[crate::rt::nn::Linear; 2]> {
        use crate::rt::nn::Linear;
        use crate::rt::nvfp4_linear::{Nvfp4Linear, Nvfp4Stack};
        use cudarc::driver::DevicePtr;
        let rule = fastvideo_models::nvfp4::ScaleRule::Static6;
        let d =
            crate::rt::device::global_device().ok_or_else(|| msg("h3 nvfp4: no CUDA device"))?;
        let quantize = |name: &str, rows: usize, k: usize| -> Result<Nvfp4Linear> {
            let key = format!("transformer_blocks.{i}.attn.{name}.weight");
            let (shape, values) = self
                .map
                .lazy_bf16(&key)?
                .ok_or_else(|| msg(format!("h3 nvfp4: {key} is not mapped")))?;
            if shape != [rows, k] {
                return Err(msg(format!(
                    "h3 nvfp4: {key} is {shape:?}, expected [{rows}, {k}]"
                )));
            }
            let w = d
                .stream
                .memcpy_stod(&values)
                .map_err(|e| msg(e.to_string()))?;
            let l = {
                let (p, _guard) = w.device_ptr(&d.stream);
                Nvfp4Linear::from_bf16(p, rows, k, None, rule)?
            };
            // Stream-ordered free: the quantizer has read it by then.
            drop(w);
            Ok(l)
        };
        let (h, inner) = (cfg.hidden_size, cfg.inner_dim());
        let mut parts = vec![
            quantize("to_q", inner, h)?,
            quantize("to_k", inner, h)?,
            quantize("to_v", inner, h)?,
        ];
        if gate {
            parts.push(quantize("to_gate_compress", inner, h)?);
        }
        Ok([
            Linear::from_nvfp4(Nvfp4Stack::new(parts)?)?,
            Linear::from_nvfp4(Nvfp4Stack::new(vec![quantize("to_out.0", h, inner)?])?)?,
        ])
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
    ) -> Result<[crate::rt::nn::Linear; 4]> {
        use crate::rt::nn::Linear;
        use crate::rt::nvfp4_linear::{Nvfp4Linear, Nvfp4Operand, Nvfp4Stack};
        let rule = fastvideo_models::nvfp4::ScaleRule::Static6;
        let dev = |l: PublishedLinear| -> Result<Nvfp4Linear> {
            let op = Nvfp4Operand::from_host_layout(
                &l.codes,
                &l.scales,
                l.rows,
                l.k,
                l.decode_scale,
                rule,
            )?;
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

    #[cfg(feature = "nvfp4-gemm")]
    #[test]
    fn blocked_offset_is_cublas_to_blocked() {
        // The cuBLASLt operand quantizer's layout (wan/nvfp4_gemm.rs swizzle_scales_host).
        let (rows, sc) = (256, 8);
        let linear: Vec<u8> = (0..rows * sc).map(|i| (i % 251) as u8).collect();
        let sw = crate::rt::nvfp4_gemm::swizzle_scales_host(&linear, rows, sc);
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
                (
                    "l.comfy_quant",
                    "U8",
                    vec![19],
                    br#"{"format": "nvfp4"}"#.to_vec(),
                ),
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
                (
                    "l.weight_scale",
                    "F8_E4M3",
                    vec![rows, k / 16],
                    scales.clone(),
                ),
                ("l.weight_scale_2", "F32", vec![], ds.to_le_bytes().to_vec()),
                (
                    "l.comfy_quant",
                    "U8",
                    vec![19],
                    br#"{"format": "mxfp8"}"#.to_vec(),
                ),
                ("m.weight", "U8", vec![rows, k / 2], codes),
                ("m.weight_scale", "U8", vec![rows, k / 16], scales),
                ("m.weight_scale_2", "F32", vec![], ds.to_le_bytes().to_vec()),
                (
                    "m.comfy_quant",
                    "U8",
                    vec![19],
                    br#"{"format": "nvfp4"}"#.to_vec(),
                ),
            ],
        );
        let store = LazyStore::open_files(&[p]).unwrap();
        let e = PublishedLinear::read(&store, "l", rows, k)
            .unwrap_err()
            .to_string();
        assert!(e.contains("comfy_quant"), "{e}");
        let e = PublishedLinear::read(&store, "m", rows, k)
            .unwrap_err()
            .to_string();
        assert!(e.contains("weight_scale"), "{e}");
        let e = PublishedLinear::read(&store, "l", rows, 2 * k)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("weight_scale") || e.contains("comfy_quant") || e.contains("weight"),
            "{e}"
        );
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
        assert_eq!(
            comfy_block_key("transformer_blocks.7.norm2.weight").as_deref(),
            Some("blocks.7.norm2.weight")
        );
        assert_eq!(
            comfy_block_key("transformer_blocks.7.attn.to_q.weight"),
            None
        );
        assert!(is_nvfp4_recipe("trim-8step-nvfp4") && is_nvfp4_recipe("fasth3-trim-nvfp4"));
        assert!(!is_nvfp4_recipe("trim-8step"));
        assert_eq!(
            sibling_tree(Path::new("/w/fasth3-trim-8step/")),
            Path::new("/w").join(TREE)
        );
    }

    /// ModelOpt's unified layout is our TE quantization as stored: low
    /// nibble first, row-major scales. 144 rows: the scale rows pad to 256.
    #[test]
    fn modelopt_layout_reads_as_the_quantized_weight() {
        let (rows, k) = (144, 128);
        let w = weights(rows, k, 2.0);
        let t = quantize(&w, rows, k, ScaleRule::Static6).unwrap();
        let reference = t.dequantize();
        let ds = fastvideo_models::nvfp4::dequant_factor(t.amax, ScaleRule::Static6);
        let f32b = |v: f32| v.to_le_bytes().to_vec();
        let tmp = TmpDir::new();
        let p = header_file(
            tmp.path(),
            &[
                ("l.weight", "U8", vec![rows, k / 2], t.packed.clone()),
                (
                    "l.weight_scale",
                    "F8_E4M3",
                    vec![rows, k / 16],
                    t.scales.clone(),
                ),
                ("l.weight_scale_2", "F32", vec![], f32b(ds)),
                ("l.input_scale", "F32", vec![], f32b(0.375)),
                ("m.weight", "U8", vec![rows, k / 2], t.packed.clone()),
                (
                    "m.weight_scale",
                    "F8_E4M3",
                    vec![rows, k / 16],
                    t.scales.clone(),
                ),
                ("m.weight_scale_2", "F32", vec![], f32b(ds)),
                ("b.weight", "U8", vec![rows, k / 2], t.packed.clone()),
                (
                    "b.weight_scale",
                    "F8_E4M3",
                    vec![rows.next_multiple_of(128), k / 16],
                    vec![0; 256 * k / 16],
                ),
                ("b.weight_scale_2", "F32", vec![], f32b(ds)),
            ],
        );
        let store = LazyStore::open_files(&[p]).unwrap();
        let (l, input_scale) = PublishedLinear::read_modelopt(&store, "l", rows, k).unwrap();
        assert_eq!(input_scale, Some(0.375));
        assert_eq!(l.decode_scale, ds);
        assert_eq!(l.codes, t.packed, "codes are kept low nibble first");
        let got = l.dequantize();
        for (i, (a, b)) in got.iter().zip(&reference).enumerate() {
            assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-6), "{i}: {a} vs {b}");
        }
        // The same weight through the Trim reader's layout is the same operand.
        let (_, blocked, _, _) = publish(&w, rows, k);
        assert_eq!(l.scales, blocked);
        // No input_scale: dynamic; tiled (Comfy-shaped) scales refused.
        assert_eq!(
            PublishedLinear::read_modelopt(&store, "m", rows, k)
                .unwrap()
                .1,
            None
        );
        let e = PublishedLinear::read_modelopt(&store, "b", rows, k)
            .unwrap_err()
            .to_string();
        assert!(e.contains("row-major"), "{e}");
    }

    #[test]
    fn modelopt_config_detection() {
        let tmp = TmpDir::new();
        let write = |json: &str| std::fs::write(tmp.path().join("config.json"), json).unwrap();
        assert!(!is_modelopt_dir(tmp.path()).unwrap(), "no config.json");
        write(r#"{"_class_name": "MiniMaxH3Transformer3DModel"}"#);
        assert!(!is_modelopt_dir(tmp.path()).unwrap());
        write(r#"{"quantization_config": {"quant_method": "fp8"}}"#);
        assert!(
            !is_modelopt_dir(tmp.path()).unwrap(),
            "another kind keeps its path"
        );
        write(r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}"#);
        assert!(is_modelopt_dir(tmp.path()).unwrap());
        write(r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}"#);
        assert!(is_modelopt_dir(tmp.path()).is_err());
        let cfg = H3TransformerConfig::fasth3_trim_8step();
        let [a, b] = modelopt_ffn_linears(&cfg, 3);
        assert_eq!(
            (a.0.as_str(), a.1, a.2),
            (
                "transformer_blocks.3.ff.net.0.proj",
                2 * cfg.ffn_dim,
                cfg.hidden_size
            )
        );
        assert_eq!(
            (b.0.as_str(), b.1, b.2),
            (
                "transformer_blocks.3.ff.net.2",
                cfg.hidden_size,
                cfg.ffn_dim
            )
        );
    }

    #[test]
    fn trim_linear_shapes() {
        let cfg = H3TransformerConfig::fasth3_trim_8step();
        let l = block_linears(&cfg, 41);
        let shapes: Vec<(usize, usize)> = l.iter().map(|(_, r, k)| (*r, *k)).collect();
        // The published file's headers: qkv [21504, 2688], gate [7168, 2688],
        // out [5376, 3584], fc1 [28672, 2688], fc2 [5376, 7168] (U8 = k / 2).
        assert_eq!(
            shapes,
            [
                (21504, 5376),
                (7168, 5376),
                (5376, 7168),
                (28672, 5376),
                (5376, 14336)
            ]
        );
        assert_eq!(l[3].0, "blocks.41.mlp.fc1");
    }
}
