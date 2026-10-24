//! MiniMax-H3 Turbo LoRAs: what the `MiniMaxAI/MiniMax-H3-Turbo-Lora` Space
//! (revision `9a76d679`) folds into the bf16 transformer at load
//! (`h3_lora.py`), as named recipes. docs/ports/h3-turbo-loras.md.
//!
//! Two adapters, one fold (`W += strength * scale * B @ A`, float32, one
//! bf16 rounding):
//!
//! * **larry** (`larryvrh/MiniMax-H3-Turbo-Lora`, Apache-2.0) targets the
//!   ComfyUI (reference) module tree: `blocks.N.attn.qkv_proj`,
//!   `blocks.N.mlp.fc1`, `token_refiner.blocks.N`,
//!   `final_layer.adaln_proj.linear`, with `alpha == rank` (scale 1, ranks 64
//!   and 16 in one file). Each `lora_B` gets the row transform the base
//!   weights got in the diffusers conversion ([`comfy_targets`], a port of
//!   the Space's `_larry_targets`): fused-QKV row thirds onto
//!   `attn.to_q/k/v` (sharing `lora_A`), the SwiGLU `[gate; value]` →
//!   `[value; gate]` swap onto `ff.net.0.proj`, `fc2` → `ff.net.2`,
//!   `out_proj` → `to_out.0`, `blocks.` → `transformer_blocks.`,
//!   `token_refiner.blocks.` → `token_refiner.refiner_blocks.`,
//!   `final_layer.adaln_proj.linear` → `norm_out.linear`; the block AdaLN
//!   rows keep their layout.
//! * **lightx2v** (`lightx2v/Minimax-h3-Turbo`, Apache-2.0) is PEFT against
//!   the diffusers tree, rank 128, alpha 8 (scale 0.0625); keys map name for
//!   name ([`super::lora::plan_from_keys`]).
//!
//! Recipes (`H3InferenceContract::named`): `larry-turbo[-<N>step]` (default
//! 6 forwards) and `lightx2v-turbo[-<N>step]` / `lightx2v-turbo-v0.1[-<N>step]`
//! (default 4). All are dense attention, guidance 1 (guidance-distilled),
//! the uniform grid `linspace(1, 0, N + 1)` with the adapter's video / audio
//! shifts. **Steps count transformer forwards**, as everywhere in this repo;
//! the Space's `steps` is diffusers' `num_inference_steps` (grid points), so
//! its default `6` is 5 forwards here (`larry-turbo-5step`).

use std::path::{Path, PathBuf};

/// One Turbo adapter: where it comes from, how it is scaled and sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboLora {
    /// `larryvrh/MiniMax-H3-Turbo-Lora` `minimax_h3_turbo_4step_ema_ckpt850`
    /// (the Space's default file), ComfyUI layout, alpha = rank.
    Larry,
    /// `lightx2v/Minimax-h3-Turbo` FL2VA/T2VA 4-step v1.2 768p (newest
    /// 4-step 768p file; trained at video shift 6 / audio shift 3).
    Lightx2v,
    /// `lightx2v/Minimax-h3-Turbo` FL2VA/T2VA 4-step v0.1 (the file the
    /// Space folds; trained at 544p, shifts 12 / 3). R2 tree
    /// `h3-turbo-loras/lightx2v-v0.1` (owner approval 2026-10-14).
    Lightx2vV01,
}

/// The tree under the weight root that holds the published Turbo adapters
/// (`scripts/gpu/weights-manifest.tsv`).
pub const TURBO_LORA_TREE: &str = "h3-turbo-loras";

/// Per-model strength override for the process (the Space's
/// `H3_LORA_STRENGTH`): used when the pipeline options give none.
pub const TURBO_LORA_STRENGTH_ENV: &str = "FV_H3_TURBO_LORA_STRENGTH";

impl TurboLora {
    pub const ALL: [TurboLora; 3] = [Self::Larry, Self::Lightx2v, Self::Lightx2vV01];

    /// Recipe stem (`<stem>` or `<stem>-<N>step`).
    pub fn stem(self) -> &'static str {
        match self {
            Self::Larry => "larry-turbo",
            Self::Lightx2v => "lightx2v-turbo",
            Self::Lightx2vV01 => "lightx2v-turbo-v0.1",
        }
    }

    pub fn hub_repo(self) -> &'static str {
        match self {
            Self::Larry => "larryvrh/MiniMax-H3-Turbo-Lora",
            Self::Lightx2v | Self::Lightx2vV01 => "lightx2v/Minimax-h3-Turbo",
        }
    }

    /// Pinned Hub revision (`weights-revisions.tsv`).
    pub fn hub_revision(self) -> &'static str {
        match self {
            Self::Larry => "43a74557ac3f6539db8e0f2a959d03feb7a81480",
            Self::Lightx2v | Self::Lightx2vV01 => "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
        }
    }

    pub fn file(self) -> &'static str {
        match self {
            Self::Larry => "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
            Self::Lightx2v => "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors",
            Self::Lightx2vV01 => "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
        }
    }

    /// The Hub LFS SHA-256 of [`Self::file`].
    pub fn sha256(self) -> &'static str {
        match self {
            Self::Larry => "5a6eeba171cf183020a4ad48774bb2968f29f8168afd6ec17a04987f3528b4ea",
            Self::Lightx2v => "c3d4a2cf618efea71b9e21a4baaa12d412f1eb6c2b6f86efacaf0ebb6814b689",
            Self::Lightx2vV01 => "5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29",
        }
    }

    /// Sub-tree of [`TURBO_LORA_TREE`].
    pub fn subdir(self) -> &'static str {
        match self {
            Self::Larry => "larry",
            Self::Lightx2v => "lightx2v",
            Self::Lightx2vV01 => "lightx2v-v0.1",
        }
    }

    /// PEFT alpha; `None` means alpha = rank (scale 1, the larry card).
    pub fn alpha(self) -> Option<u32> {
        match self {
            Self::Larry => None,
            Self::Lightx2v | Self::Lightx2vV01 => Some(8),
        }
    }

    /// Default forwards: larry's card recommends 6-8 (4 is the minimum);
    /// lightx2v's 4-step files are distilled for 4 NFE.
    pub fn default_forwards(self) -> usize {
        match self {
            Self::Larry => 6,
            Self::Lightx2v | Self::Lightx2vV01 => 4,
        }
    }

    /// `(video, audio)` scheduler shifts. larry samples on the base 12 / 3
    /// (its `generate.py`); lightx2v's 768p v1.x files were trained at 6 / 3,
    /// v0.1 at 12 / 3 (ModelTC/Minimax-H3-Turbo "Model specs").
    pub fn shifts(self) -> (f64, f64) {
        match self {
            Self::Larry | Self::Lightx2vV01 => (12.0, 3.0),
            Self::Lightx2v => (6.0, 3.0),
        }
    }

    pub fn licence(self) -> &'static str {
        "Apache-2.0"
    }

    /// The recipe for `forwards` transformer evaluations.
    pub fn recipe(self, forwards: usize) -> String {
        if forwards == self.default_forwards() {
            self.stem().to_owned()
        } else {
            format!("{}-{forwards}step", self.stem())
        }
    }

    /// Where the adapter is searched: `explicit`, then
    /// `<base>/h3-turbo-loras/<subdir>/<file>`, `<base>/<subdir>/<file>` and
    /// `<base>/<file>` for `base` in (root, root's parent).
    pub fn resolve(self, root: &Path, explicit: Option<&Path>) -> Result<PathBuf, String> {
        if let Some(path) = explicit {
            return if path.is_file() {
                Ok(path.to_path_buf())
            } else {
                Err(format!("Turbo LoRA does not exist: {}", path.display()))
            };
        }
        let mut tried = Vec::new();
        for base in [Some(root), root.parent()].into_iter().flatten() {
            for rel in [
                Path::new(TURBO_LORA_TREE)
                    .join(self.subdir())
                    .join(self.file()),
                Path::new(self.subdir()).join(self.file()),
                PathBuf::from(self.file()),
            ] {
                let path = base.join(rel);
                if path.is_file() {
                    return Ok(path);
                }
                tried.push(path.display().to_string());
            }
        }
        Err(format!(
            "{} adapter {} ({} @ {}) not found. Looked for: {}",
            self.stem(),
            self.file(),
            self.hub_repo(),
            &self.hub_revision()[..7],
            tried.join(", ")
        ))
    }
}

/// Smallest and largest forwards a Turbo recipe accepts. larry's card: 4-8
/// useful ("past 8 it stops helping"); 3 is the Space's `steps = 4`.
pub const TURBO_MIN_FORWARDS: usize = 3;
pub const TURBO_MAX_FORWARDS: usize = 8;

/// `(adapter, forwards)` of a Turbo recipe name, `None` for any other name.
/// An out-of-range step count is an error.
pub fn parse_turbo_recipe(name: &str) -> Option<Result<(TurboLora, usize), String>> {
    // Longest stem first: `lightx2v-turbo-v0.1` starts with `lightx2v-turbo`.
    let mut by_len = TurboLora::ALL;
    by_len.sort_by_key(|t| std::cmp::Reverse(t.stem().len()));
    for t in by_len {
        let Some(rest) = name.strip_prefix(t.stem()) else {
            continue;
        };
        if rest.is_empty() {
            return Some(Ok((t, t.default_forwards())));
        }
        let Some(n) = rest.strip_prefix('-').and_then(|r| r.strip_suffix("step")) else {
            continue;
        };
        let Ok(n) = n.parse::<usize>() else {
            continue;
        };
        if !(TURBO_MIN_FORWARDS..=TURBO_MAX_FORWARDS).contains(&n) {
            return Some(Err(format!(
                "recipe {name}: {} takes {TURBO_MIN_FORWARDS}..={TURBO_MAX_FORWARDS} steps (transformer forwards)",
                t.stem()
            )));
        }
        return Some(Ok((t, n)));
    }
    None
}

/// The strength a Turbo recipe fuses at: the option, else
/// [`TURBO_LORA_STRENGTH_ENV`], else 1.0. Finite and in `0..=4`.
pub fn turbo_strength(option: Option<f32>, env: Option<&str>) -> Result<f32, String> {
    let s = match (option, env.map(str::trim).filter(|s| !s.is_empty())) {
        (Some(s), _) => s,
        (None, Some(raw)) => raw
            .parse()
            .map_err(|_| format!("{TURBO_LORA_STRENGTH_ENV}={raw:?} is not a number"))?,
        (None, None) => 1.0,
    };
    if !s.is_finite() || !(0.0..=4.0).contains(&s) {
        return Err(format!(
            "Turbo LoRA strength {s} must be finite and in 0..=4"
        ));
    }
    Ok(s)
}

// ---- ComfyUI-layout (larry) key conversion ---------------------------------

/// Which rows of a ComfyUI `lora_B` one diffusers target takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowMap {
    /// The whole matrix.
    All,
    /// Rows `start..start + len` (one third of a fused QKV).
    Range { start: usize, len: usize },
    /// `[gate; value]` → `[value; gate]` (SwiGLU `fc1` → `ff.net.0.proj`).
    SwapHalves,
}

/// One diffusers target of a ComfyUI LoRA module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComfyTarget {
    /// Diffusers module name (the parameter is `<module>.weight`).
    pub module: String,
    pub rows: RowMap,
}

/// Whether adapter keys are in the ComfyUI (reference) layout: a module under
/// `blocks.`, `token_refiner.blocks.` or `final_layer.`, or a fused
/// `qkv_proj` / `mlp.fc1`. Diffusers-layout files use
/// `transformer_blocks.` / `token_refiner.refiner_blocks.`.
pub fn is_comfy_layout(keys: &[String]) -> bool {
    keys.iter().any(|k| {
        let k = k.strip_prefix("diffusion_model.").unwrap_or(k);
        k.starts_with("blocks.")
            || k.starts_with("token_refiner.blocks.")
            || k.starts_with("final_layer.")
            || k.contains(".attn.qkv_proj.")
            || k.contains(".mlp.fc1.")
    })
}

/// The Space's `_larry_targets`: one ComfyUI module name (no `.lora_*`) and
/// the row count of its `lora_B` onto diffusers module names and row maps.
/// `inner_dim` is `heads * head_dim` (7168), the row count of each of q, k, v.
pub fn comfy_targets(
    name: &str,
    b_rows: usize,
    inner_dim: usize,
) -> Result<Vec<ComfyTarget>, String> {
    let name = name.strip_prefix("diffusion_model.").unwrap_or(name);
    let mut target = if let Some(rest) = name.strip_prefix("token_refiner.blocks.") {
        format!("token_refiner.refiner_blocks.{rest}")
    } else if let Some(rest) = name.strip_prefix("blocks.") {
        format!("transformer_blocks.{rest}")
    } else {
        name.to_owned()
    };
    target = target.replace("final_layer.adaln_proj.linear", "norm_out.linear");

    if let Some(prefix) = target
        .strip_suffix("qkv_proj")
        .filter(|_| target.ends_with(".attn.qkv_proj"))
    {
        if inner_dim == 0 || b_rows != 3 * inner_dim {
            return Err(format!(
                "{name}: fused QKV lora_B has {b_rows} rows, expected 3 x {inner_dim}"
            ));
        }
        return Ok(["q", "k", "v"]
            .iter()
            .enumerate()
            .map(|(i, kind)| ComfyTarget {
                module: format!("{prefix}to_{kind}"),
                rows: RowMap::Range {
                    start: i * inner_dim,
                    len: inner_dim,
                },
            })
            .collect());
    }
    if let Some(stem) = target.strip_suffix(".mlp.fc1") {
        if b_rows == 0 || !b_rows.is_multiple_of(2) {
            return Err(format!(
                "{name}: SwiGLU fc1 lora_B has an odd row count {b_rows}"
            ));
        }
        return Ok(vec![ComfyTarget {
            module: format!("{stem}.ff.net.0.proj"),
            rows: RowMap::SwapHalves,
        }]);
    }
    if let Some(stem) = target.strip_suffix(".mlp.fc2") {
        return Ok(vec![ComfyTarget {
            module: format!("{stem}.ff.net.2"),
            rows: RowMap::All,
        }]);
    }
    if let Some(stem) = target.strip_suffix(".attn.out_proj") {
        return Ok(vec![ComfyTarget {
            module: format!("{stem}.attn.to_out.0"),
            rows: RowMap::All,
        }]);
    }
    // Block `adaln_proj.linear` and the final `norm_out.linear`: identical row
    // layout on both sides.
    Ok(vec![ComfyTarget {
        module: target,
        rows: RowMap::All,
    }])
}

/// Apply a [`RowMap`] to a row-major `lora_B` of `rank` columns.
pub fn map_rows(b: &[f32], rank: usize, rows: RowMap) -> Result<Vec<f32>, String> {
    if rank == 0 || !b.len().is_multiple_of(rank) {
        return Err(format!(
            "lora_B len {} is not a multiple of rank {rank}",
            b.len()
        ));
    }
    let n = b.len() / rank;
    match rows {
        RowMap::All => Ok(b.to_vec()),
        RowMap::Range { start, len } => {
            if start + len > n {
                return Err(format!("rows {start}..{} of {n}", start + len));
            }
            Ok(b[start * rank..(start + len) * rank].to_vec())
        }
        RowMap::SwapHalves => {
            if !n.is_multiple_of(2) {
                return Err(format!("cannot swap halves of {n} rows"));
            }
            let half = n / 2 * rank;
            let (gate, value) = b.split_at(half);
            Ok(value.iter().chain(gate).copied().collect())
        }
    }
}

/// `(module, A key, B key)` of a ComfyUI-layout adapter, sorted by module.
/// Keys are `<module>.lora_A.weight` / `.lora_B.weight` (also
/// `.lora_down` / `.lora_up`); anything else is refused, as is a half pair.
pub fn comfy_pairs(keys: &[String], path: &str) -> Result<Vec<(String, String, String)>, String> {
    use std::collections::BTreeMap;
    let mut a: BTreeMap<String, String> = BTreeMap::new();
    let mut b: BTreeMap<String, String> = BTreeMap::new();
    let mut bad = Vec::new();
    for key in keys {
        let side = [
            (".lora_A.weight", true),
            (".lora_down.weight", true),
            (".lora_B.weight", false),
            (".lora_up.weight", false),
        ]
        .iter()
        .find_map(|(suffix, is_a)| key.strip_suffix(suffix).map(|stem| (stem, *is_a)));
        match side {
            Some((stem, true)) => {
                a.insert(stem.to_owned(), key.clone());
            }
            Some((stem, false)) => {
                b.insert(stem.to_owned(), key.clone());
            }
            None => bad.push(key.clone()),
        }
    }
    if !bad.is_empty() {
        return Err(format!(
            "{path}: unsupported ComfyUI adapter keys (alpha / diff tensors are not folded): {:?}",
            &bad[..bad.len().min(3)]
        ));
    }
    let lone: Vec<_> = a
        .keys()
        .filter(|k| !b.contains_key(*k))
        .chain(b.keys().filter(|k| !a.contains_key(*k)))
        .cloned()
        .collect();
    if !lone.is_empty() {
        return Err(format!(
            "{path}: unpaired LoRA tensors: {:?}",
            &lone[..lone.len().min(3)]
        ));
    }
    if a.is_empty() {
        return Err(format!("{path}: no LoRA tensors"));
    }
    Ok(a.into_iter()
        .map(|(m, ak)| {
            let bk = b.remove(&m).expect("paired");
            (m, ak, bk)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `B @ A` for row-major `B [out, r]`, `A [r, inn]`.
    fn matmul(b: &[f32], a: &[f32], out: usize, r: usize, inn: usize) -> Vec<f32> {
        let mut w = vec![0.0; out * inn];
        for o in 0..out {
            for k in 0..r {
                for i in 0..inn {
                    w[o * inn + i] += b[o * r + k] * a[k * inn + i];
                }
            }
        }
        w
    }

    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
    }

    #[test]
    fn names_follow_the_space() {
        let inner = 4;
        let t = comfy_targets("blocks.3.attn.qkv_proj", 12, inner).unwrap();
        assert_eq!(
            t.iter().map(|t| t.module.as_str()).collect::<Vec<_>>(),
            [
                "transformer_blocks.3.attn.to_q",
                "transformer_blocks.3.attn.to_k",
                "transformer_blocks.3.attn.to_v"
            ]
        );
        assert_eq!(t[1].rows, RowMap::Range { start: 4, len: 4 });
        let t = comfy_targets("token_refiner.blocks.1.mlp.fc1", 8, inner).unwrap();
        assert_eq!(t[0].module, "token_refiner.refiner_blocks.1.ff.net.0.proj");
        assert_eq!(t[0].rows, RowMap::SwapHalves);
        assert_eq!(
            comfy_targets("blocks.0.mlp.fc2", 5, inner).unwrap()[0].module,
            "transformer_blocks.0.ff.net.2"
        );
        assert_eq!(
            comfy_targets("blocks.49.attn.out_proj", 5, inner).unwrap()[0].module,
            "transformer_blocks.49.attn.to_out.0"
        );
        assert_eq!(
            comfy_targets("blocks.7.adaln_proj.linear", 18, inner).unwrap(),
            vec![ComfyTarget {
                module: "transformer_blocks.7.adaln_proj.linear".into(),
                rows: RowMap::All
            }]
        );
        assert_eq!(
            comfy_targets("final_layer.adaln_proj.linear", 2, inner).unwrap()[0].module,
            "norm_out.linear"
        );
        assert_eq!(
            comfy_targets("diffusion_model.blocks.2.attn.out_proj", 5, inner).unwrap()[0].module,
            "transformer_blocks.2.attn.to_out.0"
        );
        // `blocks.` only at the start; the refiner keeps its own prefix.
        assert!(
            comfy_targets("token_refiner.blocks.0.attn.qkv_proj", 12, inner).unwrap()[0]
                .module
                .starts_with("token_refiner.refiner_blocks.0.attn.")
        );
        assert!(comfy_targets("blocks.0.attn.qkv_proj", 11, inner).is_err());
        assert!(comfy_targets("blocks.0.mlp.fc1", 7, inner).is_err());
    }

    /// The fold of every target equals the diffusers conversion of the fused
    /// delta: split rows of `B_qkv @ A` into thirds, swap fc1's halves.
    #[test]
    fn row_maps_reproduce_the_converted_delta() {
        let (inner, r, inn) = (3usize, 2usize, 5usize);
        let a = ramp(r * inn, 1.0);
        let b_qkv = ramp(3 * inner * r, 2.0);
        let fused = matmul(&b_qkv, &a, 3 * inner, r, inn);
        for (i, t) in comfy_targets("blocks.0.attn.qkv_proj", 3 * inner, inner)
            .unwrap()
            .iter()
            .enumerate()
        {
            let b = map_rows(&b_qkv, r, t.rows).unwrap();
            let delta = matmul(&b, &a, inner, r, inn);
            assert_eq!(
                delta,
                fused[i * inner * inn..(i + 1) * inner * inn],
                "{}",
                t.module
            );
        }
        // fc1: the diffusers weight is cat([value, gate]) of the reference's [gate; value].
        let ffn = 2usize;
        let b_fc1 = ramp(2 * ffn * r, 3.0);
        let fused = matmul(&b_fc1, &a, 2 * ffn, r, inn);
        let (gate, value) = fused.split_at(ffn * inn);
        let want: Vec<f32> = value.iter().chain(gate).copied().collect();
        let t = &comfy_targets("blocks.0.mlp.fc1", 2 * ffn, inner).unwrap()[0];
        let b = map_rows(&b_fc1, r, t.rows).unwrap();
        assert_eq!(matmul(&b, &a, 2 * ffn, r, inn), want);
        assert_eq!(map_rows(&b_fc1, r, RowMap::All).unwrap(), b_fc1);
        assert!(map_rows(&b_fc1, r, RowMap::Range { start: 3, len: 2 }).is_err());
        assert!(map_rows(&b_fc1, 3, RowMap::All).is_err());
    }

    #[test]
    fn layout_detection_and_pairs() {
        let larry: Vec<String> = [
            "blocks.0.attn.qkv_proj.lora_A.weight",
            "blocks.0.attn.qkv_proj.lora_B.weight",
            "final_layer.adaln_proj.linear.lora_A.weight",
            "final_layer.adaln_proj.linear.lora_B.weight",
        ]
        .map(String::from)
        .to_vec();
        assert!(is_comfy_layout(&larry));
        let pairs = comfy_pairs(&larry, "larry").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "blocks.0.attn.qkv_proj");
        let peft: Vec<String> = [
            "transformer_blocks.0.attn.to_q.lora_A.default.weight",
            "token_refiner.refiner_blocks.0.ff.net.0.proj.lora_B.default.weight",
        ]
        .map(String::from)
        .to_vec();
        assert!(!is_comfy_layout(&peft));
        let mut half = larry.clone();
        half.pop();
        assert!(comfy_pairs(&half, "x").unwrap_err().contains("unpaired"));
        let mut alpha = larry;
        alpha.push("blocks.0.attn.qkv_proj.alpha".into());
        assert!(comfy_pairs(&alpha, "x").is_err());
    }

    #[test]
    fn recipes_parse() {
        assert_eq!(
            parse_turbo_recipe("larry-turbo"),
            Some(Ok((TurboLora::Larry, 6)))
        );
        assert_eq!(
            parse_turbo_recipe("larry-turbo-4step"),
            Some(Ok((TurboLora::Larry, 4)))
        );
        assert_eq!(
            parse_turbo_recipe("lightx2v-turbo"),
            Some(Ok((TurboLora::Lightx2v, 4)))
        );
        assert_eq!(
            parse_turbo_recipe("lightx2v-turbo-v0.1-5step"),
            Some(Ok((TurboLora::Lightx2vV01, 5)))
        );
        assert_eq!(
            parse_turbo_recipe("lightx2v-turbo-v0.1"),
            Some(Ok((TurboLora::Lightx2vV01, 4)))
        );
        assert!(matches!(
            parse_turbo_recipe("larry-turbo-9step"),
            Some(Err(_))
        ));
        assert!(matches!(
            parse_turbo_recipe("larry-turbo-2step"),
            Some(Err(_))
        ));
        assert_eq!(parse_turbo_recipe("larry-turbox"), None);
        assert_eq!(parse_turbo_recipe("sol-h3"), None);
        for t in TurboLora::ALL {
            for n in TURBO_MIN_FORWARDS..=TURBO_MAX_FORWARDS {
                assert_eq!(parse_turbo_recipe(&t.recipe(n)), Some(Ok((t, n))));
            }
            assert_eq!(t.sha256().len(), 64);
            assert_eq!(t.hub_revision().len(), 40);
        }
    }

    /// ModelTC/Minimax-H3-Turbo "Note on shift": NFE 4, shifts 12 / 3 gives
    /// video sigmas [1, 0.9730, 0.9231, 0.8000] -> 0 and audio
    /// [1, 0.9, 0.75, 0.5] -> 0; larry's `timesteps(n)` is the same grid.
    #[test]
    fn contracts_are_the_uniform_grid() {
        use crate::h3::config::{H3InferenceContract, H3SigmaSource};
        use crate::h3::schedule::H3JointSchedule;
        let c = H3InferenceContract::named("lightx2v-turbo-v0.1").unwrap();
        assert_eq!((c.transformer_forwards, c.num_inference_steps), (4, 5));
        assert!(c.dense && c.vsa_sparsity == 0.0 && c.guidance_scale == 1.0);
        assert_eq!(c.sigma_source, H3SigmaSource::Uniform);
        let s = H3JointSchedule::from_contract(&c).unwrap();
        for (got, want) in s
            .video
            .sigmas
            .iter()
            .zip([1.0f32, 0.9730, 0.9231, 0.8, 0.0])
        {
            assert!((got - want).abs() < 1e-4, "video {got} vs {want}");
        }
        for (got, want) in s.audio.sigmas.iter().zip([1.0f32, 0.9, 0.75, 0.5, 0.0]) {
            assert!((got - want).abs() < 1e-4, "audio {got} vs {want}");
        }
        let c = H3InferenceContract::named("lightx2v-turbo").unwrap();
        assert_eq!(
            (c.video_scheduler_shift, c.audio_scheduler_shift),
            (6.0, 3.0)
        );
        let c = H3InferenceContract::named("larry-turbo").unwrap();
        assert_eq!((c.transformer_forwards, c.video_scheduler_shift), (6, 12.0));
        // The Space's default `steps = 6` (diffusers grid points) is 5 forwards.
        let space = H3InferenceContract::named("larry-turbo-5step").unwrap();
        assert_eq!(space.num_inference_steps, 6);
        assert!(H3InferenceContract::named("larry-turbo-12step").is_err());
    }

    #[test]
    fn strength() {
        assert_eq!(turbo_strength(None, None).unwrap(), 1.0);
        assert_eq!(turbo_strength(None, Some(" 1.1 ")).unwrap(), 1.1);
        assert_eq!(turbo_strength(Some(0.9), Some("2")).unwrap(), 0.9);
        assert!(turbo_strength(None, Some("x")).is_err());
        assert!(turbo_strength(Some(f32::NAN), None).is_err());
        assert!(turbo_strength(Some(-0.1), None).is_err());
    }

    #[test]
    fn adapter_resolves_beside_base() {
        let tmp = std::env::temp_dir().join(format!("fv-turbo-lora-{}", std::process::id()));
        let base = tmp.join("h3-base");
        let dir = tmp.join(TURBO_LORA_TREE).join("larry");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join(TurboLora::Larry.file());
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(TurboLora::Larry.resolve(&base, None).unwrap(), file);
        let err = TurboLora::Lightx2v.resolve(&base, None).unwrap_err();
        assert!(err.contains("v1.2_768p"), "{err}");
        assert!(TurboLora::Larry
            .resolve(&base, Some(&tmp.join("nope")))
            .is_err());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn weights_are_recorded() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/gpu");
        let manifest = std::fs::read_to_string(root.join("weights-manifest.tsv")).unwrap();
        let revisions = std::fs::read_to_string(root.join("weights-revisions.tsv")).unwrap();
        let sha = std::fs::read_to_string(root.join("weights-sha256.tsv")).unwrap();
        for t in TurboLora::ALL {
            let dest = format!("{TURBO_LORA_TREE}/{}", t.subdir());
            assert!(
                manifest.contains(&format!("{dest}\t{}\t", t.hub_repo())),
                "{dest}"
            );
            assert!(manifest.contains(t.file()), "{dest}");
            assert!(
                revisions.contains(&format!("{dest}\t{}\t{}", t.hub_repo(), t.hub_revision())),
                "{dest}"
            );
            assert!(sha.contains(t.sha256()), "{dest}");
        }
    }
}
