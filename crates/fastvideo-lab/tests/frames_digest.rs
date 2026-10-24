//! The per-arm frames record (docs/adr/0019-fv-lab.md §8): `fv-lab run`'s
//! `frames.sha256.json` is byte-identical to what `runpod-matrix.sh`'s
//! `frames_digest` writes for the same frames (`fixtures/frames-sample.sha256.json`,
//! written by the script's own function through `scripts/gpu/tests/fv-lab-goldens.sh`).

use std::path::PathBuf;

use fastvideo_lab::run::frames_digest;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

#[test]
fn equals_the_scripts_frames_digest() {
    let ours = frames_digest(&fixtures().join("frames-sample"), "sample").expect("frames/ exists");
    let theirs = std::fs::read_to_string(fixtures().join("frames-sample.sha256.json")).unwrap();
    assert_eq!(ours, theirs);
    // PNGs only, byte order (upper case first), prompt and untimed directories included.
    assert!(ours.contains("\"count\":5}"));
    assert!(!ours.contains("clip.webm"));
    assert!(ours.find("B.png") < ours.find("cold/frame-000.png"));
}

#[test]
fn no_frames_directory_no_record() {
    assert_eq!(frames_digest(&fixtures(), "x"), None);
}
