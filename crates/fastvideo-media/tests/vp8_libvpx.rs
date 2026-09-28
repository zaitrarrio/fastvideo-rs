//! Inter-frame VP8 through ffmpeg `libvpx` (`fastvideo_media::vp8`): one
//! frame out per frame in, keyframe then inter frames, forced keyframes,
//! CBR near the target, and the IVF stream decodes with ffmpeg.
//! Skips (passes with a note) when ffmpeg has no `libvpx`.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use fastvideo_media::tools;
use fastvideo_media::vp8::{self, Vp8Config, Vp8Encoder};
use fastvideo_media::RgbFrame;

fn have_libvpx() -> bool {
    let ok = vp8::libvpx_available();
    if !ok {
        eprintln!("skipping: ffmpeg with libvpx not available (set FV_FFMPEG)");
    }
    ok
}

/// A moving gradient.
fn pattern(w: u32, h: u32, i: u64) -> RgbFrame {
    let mut d = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            d.push(((x + i as u32 * 4) % 256) as u8);
            d.push(((y * 255) / h.max(1)) as u8);
            d.push((((x + y) / 2 + i as u32) % 256) as u8);
        }
    }
    fastvideo_media::av::frame(w, h, d, i).unwrap()
}

/// Decodes a VP8 frame sequence (wrapped back into IVF) with ffmpeg; returns
/// the decoded frame count.
fn decode_count(w: u32, h: u32, frames: &[bytes::Bytes]) -> usize {
    let mut ivf = b"DKIF".to_vec();
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP80");
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    ivf.extend_from_slice(&24u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (i, f) in frames.iter().enumerate() {
        ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&(i as u64).to_le_bytes());
        ivf.extend_from_slice(f);
    }
    let mut c = Command::new(tools::ffmpeg_bin())
        .args(["-hide_banner", "-loglevel", "error", "-f", "ivf", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = c.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&ivf));
    let out = c.wait_with_output().unwrap();
    writer.join().unwrap().unwrap();
    assert!(out.status.success(), "decode failed: {}", String::from_utf8_lossy(&out.stderr));
    out.stdout.len() / (w * h * 3) as usize
}

#[test]
fn inter_frames_forced_keyframes_and_cbr() {
    if !have_libvpx() {
        return;
    }
    let (w, h, fps) = (320u32, 192u32, 24u32);
    let mut cfg = Vp8Config::new(w, h, fps);
    cfg.bitrate_bps = 500_000;
    let mut e = Vp8Encoder::new(cfg).unwrap();
    let mut out = Vec::new();
    let t0 = Instant::now();
    for i in 0..48u64 {
        if i == 30 {
            e.force_keyframe();
        }
        let got = e.encode(&pattern(w, h, i)).unwrap();
        out.extend(got);
    }
    out.extend(e.finish().unwrap());
    let elapsed = t0.elapsed();
    assert_eq!(out.len(), 48, "one frame out per frame in");
    assert!(out.iter().enumerate().all(|(i, f)| f.index == i as u64));
    let keys: Vec<usize> = out.iter().enumerate().filter(|(_, f)| f.keyframe).map(|(i, _)| i).collect();
    assert_eq!(keys, vec![0, 30], "keyframes at the start and where forced (gop is 48)");
    assert_eq!(e.restarts(), 1);
    let kbps = out.iter().map(|f| f.data.len()).sum::<usize>() as f64 * 8.0 / 2.0 / 1000.0;
    eprintln!("vp8 libvpx 320x192@24 target 500 kb/s: {kbps:.0} kb/s over 2 s, encode {elapsed:?}");
    assert!(kbps < 1500.0, "{kbps} kb/s is far above the 500 kb/s target");
    let data: Vec<bytes::Bytes> = out.iter().map(|f| f.data.clone()).collect();
    assert_eq!(decode_count(w, h, &data), 48);
}

#[test]
fn rejects_a_frame_of_another_size() {
    if !have_libvpx() {
        return;
    }
    let mut e = Vp8Encoder::new(Vp8Config::new(64, 48, 24)).unwrap();
    assert!(e.encode(&RgbFrame::black(32, 32, 0)).is_err());
    let mut n = e.encode(&RgbFrame::black(64, 48, 0)).unwrap().len();
    n += e.finish().unwrap().len();
    assert_eq!(n, 1);
}
