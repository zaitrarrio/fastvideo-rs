//! OpenH264 backend: stream shape (Constrained Baseline, level, GOP, forced
//! IDR, headers on every IDR), a decode round trip with the OpenH264 decoder,
//! and the 768p per-frame timing gate (§8 WP-03: ≤15 ms/frame, recorded).
#![cfg(feature = "openh264")]

use std::time::Instant;

use fastvideo_media::h264::{self, H264Level};
use fastvideo_media::video::openh264_backend::decode_rgb;
use fastvideo_media::video::{create_encoder, EncoderBackend, H264Config};
use fastvideo_media::RgbFrame;

fn pattern(w: u32, h: u32, i: u64) -> RgbFrame {
    let mut d = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            d.push(((x + i as u32 * 3) % 256) as u8);
            d.push(((y * 255) / h) as u8);
            d.push(((x / 4 + y / 4 + i as u32) % 256) as u8);
        }
    }
    fastvideo_media::av::frame(w, h, d, i).unwrap()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

#[test]
fn constrained_baseline_1344x768_gop_and_forced_idr() {
    let cfg = H264Config::new(1344, 768, 24);
    let mut enc = create_encoder(EncoderBackend::OpenH264, cfg).unwrap();
    let mut aus = Vec::new();
    for i in 0..100u64 {
        if i == 30 {
            enc.force_idr(); // PLI
        }
        aus.extend(enc.encode(&pattern(1344, 768, i)).unwrap());
    }
    aus.extend(enc.finish().unwrap());
    assert_eq!(aus.len(), 100, "one AU per frame, no skips");
    let keys: Vec<u64> = aus.iter().filter(|a| a.keyframe).map(|a| a.index).collect();
    assert!(keys.contains(&0) && keys.contains(&30), "IDR at start and on force: {keys:?}");
    // Periodic IDR every 2 s (48 frames) from the last IDR, never scene-cut.
    for w in keys.windows(2) {
        assert!(w[1] - w[0] <= 48, "{keys:?}");
    }
    assert!(keys.len() <= 4, "no extra IDRs: {keys:?}");
    for a in aus.iter().filter(|a| a.keyframe) {
        let sps = h264::find_sps(&a.data).expect("SPS/PPS with every IDR");
        assert_eq!(sps.profile_idc, 66, "{sps:?}");
        assert!(sps.is_constrained_baseline(), "constraint_set1 must be set: {sps:?}");
        assert!(sps.level_idc >= H264Level::L3_2.idc(), "1344x768 needs level 3.2+: {sps:?}");
        assert_eq!((sps.width, sps.height), (1344, 768));
    }
    // Round trip through the OpenH264 decoder.
    let refs: Vec<&[u8]> = aus.iter().map(|a| &a.data[..]).collect();
    let dec = decode_rgb(&refs).unwrap();
    assert_eq!(dec.len(), 100);
    for i in [0usize, 29, 30, 99] {
        assert_eq!((dec[i].width, dec[i].height), (1344, 768));
        let p = psnr(&dec[i].data, &pattern(1344, 768, i as u64).data);
        assert!(p > 28.0, "frame {i} PSNR {p:.1} dB");
    }
}

#[test]
fn configured_level_4_and_small_canvas() {
    let mut cfg = H264Config::new(1344, 768, 24);
    cfg.level = Some(H264Level::L4_0);
    let mut enc = create_encoder(EncoderBackend::OpenH264, cfg).unwrap();
    let au = enc.encode(&pattern(1344, 768, 0)).unwrap().remove(0);
    assert_eq!(h264::find_sps(&au.data).unwrap().level_idc, 40);
    // Too-small levels are refused up front.
    let mut bad = H264Config::new(1344, 768, 24);
    bad.level = Some(H264Level::L3_1);
    assert!(create_encoder(EncoderBackend::OpenH264, bad).is_err());
    // SF-Wan 832x480 at 16 fps: GOP 32.
    let mut e = create_encoder(EncoderBackend::OpenH264, H264Config::new(832, 480, 16)).unwrap();
    let mut keys = Vec::new();
    for i in 0..70 {
        for a in e.encode(&pattern(832, 480, i)).unwrap() {
            if a.keyframe {
                keys.push(a.index);
            }
        }
    }
    assert_eq!(keys, vec![0, 32, 64]);
    // Wrong frame size is an error, not a silent re-init.
    assert!(e.encode(&pattern(64, 64, 0)).is_err());
}

/// The WP-03 gate: 768p OpenH264 encode time per frame. Release builds only
/// (the C encoder is unoptimised in debug); run with
/// `cargo test --release -p fastvideo-media --features openh264 --test openh264_encode -- --ignored --nocapture`.
#[test]
#[ignore = "timing; run in release"]
fn bench_768p_ms_per_frame() {
    let mut enc = create_encoder(EncoderBackend::OpenH264, H264Config::new(1344, 768, 24)).unwrap();
    let frames: Vec<RgbFrame> = (0..24).map(|i| pattern(1344, 768, i)).collect();
    for f in &frames {
        enc.encode(f).unwrap(); // warm up
    }
    let n = 240;
    let t = Instant::now();
    for i in 0..n {
        enc.encode(&frames[i % frames.len()]).unwrap();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("openh264 1344x768@24 6 Mb/s: {ms:.2} ms/frame over {n} frames (incl. RGB->I420)");
    if !cfg!(debug_assertions) {
        assert!(ms <= 15.0, "{ms:.2} ms/frame exceeds the 15 ms gate");
    }
}
