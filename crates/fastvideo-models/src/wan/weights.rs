//! Local Diffusers checkpoint discovery. Never hits the Hub from tests.

use std::path::{Path, PathBuf};

/// `FASTVIDEO_WEIGHTS` or the Hugging Face hub snapshot for
/// `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
pub fn local_wan_t2v_1_3b() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FASTVIDEO_WEIGHTS") {
        let p = PathBuf::from(path);
        if p.join("transformer").is_dir() {
            return Some(p);
        }
    }
    hf_snapshot("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
}

pub fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let hub = Path::new(&std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    let mut snaps: Vec<_> = std::fs::read_dir(&hub)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    snaps.sort();
    snaps.pop()
}

pub fn transformer_config_json(root: &Path) -> Option<String> {
    std::fs::read_to_string(root.join("transformer/config.json")).ok()
}
