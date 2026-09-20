//! FastVideo `mlx_h3_dit.safetensors` + `mlx_h3_dit.json` (affine INT8/6/4).

use std::path::{Path, PathBuf};

use crate::wan::affine;
use crate::wan::tensor::{Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

pub const WEIGHTS: &str = "mlx_h3_dit.safetensors";
pub const MANIFEST: &str = "mlx_h3_dit.json";

#[derive(Debug, Clone)]
pub struct MlxH3Manifest {
    pub bits: u8,
    pub group_size: usize,
    pub vsa_capable: bool,
    pub weights: PathBuf,
}

/// Look for an official FastVideo MLX H3 export under `transformer/` or `root`.
pub fn find(root: &Path) -> Option<PathBuf> {
    for dir in [root.join("transformer"), root.to_path_buf()] {
        if dir.join(MANIFEST).is_file() && dir.join(WEIGHTS).is_file() {
            return Some(dir);
        }
    }
    None
}

pub fn read_manifest(dir: &Path) -> Result<MlxH3Manifest> {
    let text = std::fs::read_to_string(dir.join(MANIFEST)).map_err(|e| msg(format!("{}: {e}", dir.join(MANIFEST).display())))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| msg(format!("mlx_h3_dit.json: {e}")))?;
    let q = v.get("quantization").ok_or_else(|| msg("mlx_h3_dit.json: missing quantization"))?;
    let mode = q.get("mode").and_then(|x| x.as_str()).unwrap_or("affine");
    if mode != "affine" {
        return Err(msg(format!("mlx_h3_dit.json: unsupported mode {mode}")));
    }
    let bits = q.get("bits").and_then(|x| x.as_u64()).unwrap_or(8) as u8;
    if !matches!(bits, 4 | 6 | 8) {
        return Err(msg(format!("mlx_h3_dit.json: bits={bits}")));
    }
    let group_size = q.get("group_size").and_then(|x| x.as_u64()).unwrap_or(64) as usize;
    if group_size != affine::GROUP {
        return Err(msg(format!("mlx_h3_dit.json: group_size={group_size}, this kernel is {}", affine::GROUP)));
    }
    let vsa_capable = v.pointer("/vsa/capable").and_then(|x| x.as_bool()).unwrap_or(false);
    Ok(MlxH3Manifest { bits, group_size, vsa_capable, weights: dir.join(WEIGHTS) })
}

/// Open the MLX artifact (flattened `blocks.` keys) and set `FASTVIDEO_H3_AFFINE`.
pub fn open_map(dir: &Path) -> Result<(WeightMap, MlxH3Manifest)> {
    let spec = read_manifest(dir)?;
    affine::apply_env(Some(spec.bits));
    let map = WeightMap::open_files(&[spec.weights.clone()])?.with_mlx_h3_aliases();
    Ok((map, spec))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_official_int8_manifest() {
        let dir = std::env::temp_dir().join(format!("fv-mlx-h3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(MANIFEST),
            r#"{"quantization":{"mode":"affine","bits":8,"group_size":64},"vsa":{"capable":false}}"#,
        )
        .unwrap();
        std::fs::write(dir.join(WEIGHTS), []).unwrap();
        let spec = read_manifest(&dir).unwrap();
        assert_eq!((spec.bits, spec.group_size, spec.vsa_capable), (8, 64, false));
        assert_eq!(find(&dir).as_deref(), Some(dir.as_path()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
