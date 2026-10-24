//! Fixture helpers shared by the integration tests. The library does no
//! I/O; the tests read the fixtures and hand the bytes over.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn fixture_path(rel: &str) -> PathBuf {
    crate_root().join("fixtures").join(rel)
}

pub fn read_fixture(rel: &str) -> String {
    let p = fixture_path(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

pub fn read_fixture_bytes(rel: &str) -> Vec<u8> {
    let p = fixture_path(rel);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Files directly under `fixtures/<rel>` with the given extension, sorted.
pub fn fixture_files(rel: &str, ext: &str) -> Vec<PathBuf> {
    let dir = fixture_path(rel);
    let mut out: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("list {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == ext))
        .collect();
    out.sort();
    assert!(
        !out.is_empty(),
        "no .{ext} fixtures under {}",
        dir.display()
    );
    out
}

pub fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Regeneration mode: when `FV_MANIFEST_WRITE=<dir>` is set, generated
/// outputs are written under that directory (mirroring the crate layout)
/// instead of being compared. `bash scripts/dev/build-pod.sh fetch` copies
/// them back.
pub fn write_dir() -> Option<PathBuf> {
    std::env::var_os("FV_MANIFEST_WRITE").map(PathBuf::from)
}

pub fn write_generated(rel: &str, text: &str) {
    let Some(dir) = write_dir() else { return };
    let p = dir.join(rel);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(&p, text).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
}

pub fn keyring() -> fastvideo_manifest::KeyRing {
    ["test-key-1", "test-key-2"]
        .into_iter()
        .map(|kid| {
            let pem = read_fixture(&format!("keys/{kid}.pub.pem"));
            (
                kid,
                fastvideo_manifest::verifying_key_from_pem(&pem).expect("public key PEM"),
            )
        })
        .collect()
}

pub fn signing_key(kid: &str) -> ed25519_dalek::SigningKey {
    let pem = read_fixture(&format!("keys/{kid}.pem"));
    fastvideo_manifest::signing_key_from_pem(&pem).expect("private key PEM")
}
