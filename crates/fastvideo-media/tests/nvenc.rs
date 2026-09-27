//! NVENC (production encoder) on a real GPU. Skips unless ffmpeg can open
//! `h264_nvenc` (GPU present, driver `video` capability, ffmpeg with nvenc).
//! Run on a GPU host with `FV_FFMPEG` pointing at an nvenc-enabled ffmpeg:
//! `cargo test -p fastvideo-media --test nvenc -- --nocapture`.

use std::time::Instant;

use fastvideo_media::h264::{self, H264Level};
use fastvideo_media::mp4::{self, Mp4Spec};
use fastvideo_media::video::{create_encoder, nvenc_available, EncoderBackend, H264Config, PublishTarget};
use fastvideo_media::{Pcm, RgbFrame};

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

fn gpu() -> bool {
    let ok = nvenc_available();
    if !ok {
        eprintln!("skipping: h264_nvenc not available");
    }
    ok
}

fn run(target: PublishTarget, want: (u32, u32, H264Level)) {
    let cfg = H264Config::for_publish(target, 1344, 768, 24);
    let mut enc = create_encoder(EncoderBackend::Nvenc, cfg).unwrap();
    let frames: Vec<RgbFrame> = (0..24).map(|i| pattern(1344, 768, i)).collect();
    let mut aus = Vec::new();
    let t = Instant::now();
    for i in 0..120u64 {
        if i == 70 {
            enc.force_idr();
        }
        aus.extend(enc.encode(&frames[(i % 24) as usize]).unwrap());
    }
    aus.extend(enc.finish().unwrap());
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 120.0;
    let keys: Vec<u64> = aus.iter().filter(|a| a.keyframe).map(|a| a.index).collect();
    println!("nvenc {target:?}: {} AUs, IDRs {keys:?}, {ms:.2} ms/frame wall (pipe, incl. one restart)", aus.len());
    assert_eq!(aus.len(), 120);
    assert_eq!(keys, vec![0, 48, 70, 118]);
    for a in aus.iter().filter(|a| a.keyframe) {
        let sps = h264::find_sps(&a.data).expect("SPS with every IDR");
        println!("  idr {} sps {sps:?} plid {}", a.index, sps.profile_level_id());
        assert!(sps.is_constrained_baseline(), "{sps:?}");
        assert_eq!((sps.width, sps.height, sps.level_idc), (want.0, want.1, want.2.idc()));
    }
}

#[test]
fn nvenc_cloudflare_720p_level_3_1() {
    if gpu() {
        run(PublishTarget::Cloudflare, (1280, 720, H264Level::L3_1));
    }
}

#[test]
fn nvenc_mediamtx_native_level_4_0() {
    if gpu() {
        run(PublishTarget::Mediamtx, (1344, 768, H264Level::L4_0));
    }
}

#[test]
fn nvenc_fal_mp4() {
    if !gpu() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("h3.mp4");
    let frames: Vec<RgbFrame> = (0..48).map(|i| pattern(1344, 768, i)).collect();
    let audio = Pcm::silence(32_000, 2, 64_000);
    mp4::write_mp4(&out, Mp4Spec::fal_h3(1344, 768), &frames, Some(&audio)).unwrap();
    let i = mp4::inspect(&out).unwrap();
    println!("nvenc fal mp4: {i:?}");
    assert!(i.fal_h3_problems().is_empty(), "{:?}", i.fal_h3_problems());
}
