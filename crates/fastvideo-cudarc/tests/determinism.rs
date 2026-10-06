//! Same seed, same bytes: the host-side (CPU-checkable) parts of an LTX-2.5
//! and an H3 generation, digested, must come out identical twice in one
//! process and once more in a second process (this test binary re-run as a
//! child). The GPU side of the same promise is checked on a GPU pod
//! (docs/perf/determinism.md, `runpod-matrix.sh determinism`).
//!
//! Covered: the seeded noise streams (LTX token-major bf16 draws and the
//! stage-2 renoise draw, H3 row noise), the RoPE tables the device gathers
//! from (the factored LUTs at the 1080p stage-1 and stage-2 grids, and with a
//! keyframe block), the distilled sigma schedules, the fixed dense-SDPA
//! kernel rule over a shape sweep, the ordered second pass of the per-block
//! f64 reductions, and a conditioning-cache round trip.

use std::process::Command;

use fastvideo_models::ltx2::rope::{Ltx2RopeLuts, ScalarDivision};
use sha2::{Digest, Sha256};

const SEED: u64 = 1024;
const CHILD_ENV: &str = "FV_DETERMINISM_CHILD";
const DIGEST_TAG: &str = "DETERMINISM_DIGEST=";

fn feed_f32(h: &mut Sha256, label: &str, v: &[f32]) {
    h.update(label.as_bytes());
    h.update((v.len() as u64).to_le_bytes());
    for x in v {
        h.update(x.to_bits().to_le_bytes());
    }
}

fn feed_f64(h: &mut Sha256, label: &str, v: &[f64]) {
    h.update(label.as_bytes());
    h.update((v.len() as u64).to_le_bytes());
    for x in v {
        h.update(x.to_bits().to_le_bytes());
    }
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every CPU-checkable input of an LTX-2.5 1080p two-stage run and an H3
/// run, in one digest.
fn digest() -> String {
    let mut h = Sha256::new();

    // LTX-2.5 distilled, 1920x1088, 145 frames at 24 fps: stage 1 at half
    // size (17 x 30), stage 2 at full size (34 x 60), 19 latent frames.
    let cfg = fastvideo_models::ltx2::config::ltx2_5_22b_distilled();
    let audio_tokens = cfg.transformer.audio_tokens(145, 24.0);
    let mut noise = fastvideo_cudarc::ltx2::pipeline::NoiseStream::new(SEED, true);
    let (video, audio) =
        fastvideo_cudarc::ltx2::pipeline::initial_noise(&cfg, [19, 17, 30], audio_tokens, &mut noise)
            .expect("ltx noise");
    feed_f32(&mut h, "ltx_noise_video", &video.host_cow().expect("host"));
    feed_f32(&mut h, "ltx_noise_audio", &audio.host_cow().expect("host"));
    // The stage-2 renoise continues the same stream.
    let renoise = noise
        .draw(&[&[1, 19 * 34 * 60, cfg.transformer.in_channels]])
        .expect("renoise");
    feed_f32(&mut h, "ltx_renoise", &renoise[0].host_cow().expect("host"));

    for (name, grid, extra) in [
        ("rope_s1", [19usize, 17, 30], &[][..]),
        ("rope_s2", [19, 34, 60], &[][..]),
        ("rope_s2_keyframe", [19, 34, 60], &[144usize][..]),
    ] {
        let luts = Ltx2RopeLuts::with_conditioning(
            &cfg.transformer,
            grid,
            extra,
            None,
            audio_tokens,
            24.0,
            ScalarDivision::Reciprocal,
        );
        // The factored tables the device gathers from (the gather itself is
        // checked bit for bit by `fv-gpucheck kernels --groups ltx_rope`).
        for (part, t) in [
            ("video", &luts.video),
            ("audio", &luts.audio),
            ("cross_video", &luts.cross_video),
            ("cross_audio", &luts.cross_audio),
        ] {
            h.update(format!("{name}_{part}_index").as_bytes());
            for i in &t.index {
                h.update(i.to_le_bytes());
            }
            feed_f32(&mut h, &format!("{name}_{part}_cos"), &t.cos);
            feed_f32(&mut h, &format!("{name}_{part}_sin"), &t.sin);
        }
    }

    let s1 = fastvideo_models::ltx2::schedule::Ltx2Schedule::distilled();
    let s2 = fastvideo_models::ltx2::schedule::Ltx2Schedule::distilled_stage_2();
    feed_f64(&mut h, "ltx_sigmas_s1", &s1.sigmas);
    feed_f64(&mut h, "ltx_sigmas_s2", &s2.sigmas);

    // H3 at 1920x1088, 5 s.
    let h3 = fastvideo_models::h3::config::H3TransformerConfig::fasth3_8step();
    let geometry = fastvideo_models::h3::config::H3Geometry::new(1088, 1920, 124).expect("h3 geometry");
    let (hv, ha) = fastvideo_cudarc::h3::pipeline::seeded_noise(&h3, &geometry, SEED).expect("h3 noise");
    feed_f32(&mut h, "h3_noise_video", &hv);
    feed_f32(&mut h, "h3_noise_audio", &ha);

    // The dense-SDPA kernel rule over a shape sweep (every class boundary).
    use fastvideo_cudarc::wan::sdpa_rule;
    let mut picks = Vec::new();
    for sm in [8, 9, 10, 12] {
        for n in (512..140_000).step_by(1_531) {
            for sk in [n, 151, 512, 1024] {
                picks.push(sdpa_rule::pick(sm, n, sk, 128) as u8);
            }
        }
    }
    h.update(b"sdpa_rule");
    h.update(&picks);

    // The ordered second pass of the per-block f64 reductions.
    let pairs: Vec<f64> = (0..8192).map(|i| ((i * 7919) % 1000) as f64 * 1e-3 + 1e-9 * i as f64).collect();
    let (a, b) = fastvideo_cudarc::wan::ops::sum_pairs_in_order(&pairs);
    feed_f64(&mut h, "pair_sums", &[a, b]);

    hex(&h.finalize())
}

#[test]
fn same_seed_same_bytes_within_one_process() {
    let a = digest();
    let b = digest();
    assert_eq!(a, b, "two digests in one process differ");
}

/// Prints the digest for the parent test when run as its child.
#[test]
fn determinism_child_digest() {
    let d = digest();
    if std::env::var_os(CHILD_ENV).is_some() {
        // libtest prints "test <name> ... " on the same line first.
        println!("\n{DIGEST_TAG}{d}");
    }
}

#[test]
fn same_seed_same_bytes_across_processes() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let exe = std::env::current_exe().expect("test binary");
    let out = Command::new(exe)
        .args(["--exact", "determinism_child_digest", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn the child process");
    assert!(out.status.success(), "child failed: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let child = stdout
        .lines()
        .find_map(|l| l.split(DIGEST_TAG).nth(1).and_then(|r| r.split_whitespace().next()))
        .unwrap_or_else(|| panic!("no digest in the child's output:\n{stdout}"))
        .to_string();
    assert_eq!(digest(), child, "the child process computed a different digest");
}

#[test]
fn a_cached_conditioning_reads_back_bit_for_bit() {
    use fastvideo_cudarc::ltx2::text_cache::TextCache;
    use fastvideo_cudarc::wan::tensor::CudaTensor;
    let dir = std::env::temp_dir().join(format!("fv-determinism-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cache = TextCache::new(&dir);
    let prompt = fastvideo_cudarc::ltx2::text::PaddedPrompt::from_ids(&[2, 7, 9], 8).expect("prompt");
    // Values a bf16 activation path produces, and awkward f32 ones.
    let video: Vec<f32> = (0..8 * 6).map(|i| half::bf16::from_f32((i as f32 * 0.37).sin()).to_f32()).collect();
    let audio: Vec<f32> = (0..8 * 4).map(|i| (i as f32 * 1.1).cos() / 3.0).collect();
    let (v, a) = (
        CudaTensor::from_vec(video.clone(), vec![1, 8, 6]).expect("video"),
        CudaTensor::from_vec(audio.clone(), vec![1, 8, 4]).expect("audio"),
    );
    cache.store("k", &prompt, &v, &a).expect("store");
    let first = std::fs::read(cache.path("k")).expect("entry");
    let hit = cache.load("k", &prompt).expect("hit");
    assert_eq!(&*hit.video.host_cow().expect("host"), video.as_slice());
    assert_eq!(&*hit.audio.host_cow().expect("host"), audio.as_slice());
    cache.store("k", &prompt, &hit.video, &hit.audio).expect("store again");
    assert_eq!(std::fs::read(cache.path("k")).expect("entry"), first);
    let _ = std::fs::remove_dir_all(&dir);
}
