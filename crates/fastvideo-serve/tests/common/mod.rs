//! The shipped serve configs as the tests load them (`param_spec_golden`,
//! `cert_matrix`): `configs/serve/*.toml` sorted by name, then the compat
//! suite's `tests/compat/serve.toml`, with the environment a test needs.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fastvideo_serve::config::Config;
use fastvideo_serve_kit::KeyRing;

/// The workspace root.
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every shipped config: `configs/serve/*.toml` sorted by name, then the
/// compat suite's.
pub fn configs() -> Vec<(String, PathBuf)> {
    let dir = root().join("configs/serve");
    let mut out: Vec<(String, PathBuf)> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), p))
        .collect();
    out.sort();
    out.push((
        "compat-serve".to_owned(),
        root().join("tests/compat/serve.toml"),
    ));
    out
}

/// The config as the server loads it, with API keys `keys` (plain text;
/// hashed here) and local stores (the specs and the matrix run on the fake
/// engine).
pub fn load(name: &str, path: &Path, state_dir: &Path, keys: &[String]) -> Config {
    let text = std::fs::read_to_string(path).unwrap();
    let mut c = Config::from_toml(&text, name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut env = BTreeMap::new();
    let hashes: Vec<String> = keys.iter().map(|k| KeyRing::hash_hex(k)).collect();
    env.insert("FV_API_KEYS".to_owned(), hashes.join(","));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "golden-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), state_dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("FV_WEIGHTS".to_owned(), "/workspace/weights".to_owned());
    c.apply_env(&env).unwrap();
    c.jobs.backend = fastvideo_serve::config::JobBackend::Memory;
    c.artifacts.backend = fastvideo_serve::config::ArtifactBackend::Local;
    c.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
    c
}
