//! VP8 bitrate and quality of the two Reactor VP8 encoders on the same
//! frames: the intra-only libwebp fallback (quality 70, the Reactor
//! default) and inter-frame ffmpeg `libvpx` (CBR at `fastvideo-media`'s
//! default bitrate for the canvas, or `--bitrate`).
//!
//! ```text
//! cargo run -p fastvideo-reactor --example vp8_bitrate -- \
//!     [--size 1344x768] [--fps 24] [--seconds 4] [--source testsrc2|fake] [--bitrate 6000000]
//! ```
//!
//! `testsrc2` is ffmpeg's moving test pattern; `fake` is the fake engine's
//! frames (a static gradient with a frame counter, so near-static content).
//! Each stream is decoded back with ffmpeg and compared with its source
//! (luma PSNR). Needs ffmpeg with libvpx.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use bytes::Bytes;
use fastvideo_media::vp8::{Vp8Config, Vp8Encoder as Libvpx};
use fastvideo_protocol::RgbFrame;
use fastvideo_reactor::media::{vp8::Vp8Encoder as Webp, FrameEncoder};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn testsrc2(w: u32, h: u32, fps: u32, n: usize) -> Vec<RgbFrame> {
    let out = Command::new(fastvideo_media::tools::ffmpeg_bin())
        .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=s={w}x{h}:r={fps}"))
        .args(["-frames:v", &n.to_string(), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output()
        .expect("ffmpeg");
    let len = (w * h * 3) as usize;
    out.stdout
        .chunks_exact(len)
        .enumerate()
        .map(|(i, c)| RgbFrame { width: w, height: h, data: Bytes::copy_from_slice(c), index: i as u64 })
        .collect()
}

/// Decodes VP8 frames (wrapped into IVF) with ffmpeg to rgb24.
fn decode(w: u32, h: u32, frames: &[Bytes]) -> Vec<u8> {
    let mut ivf = b"DKIF\0\0\x20\0VP80".to_vec();
    ivf.extend_from_slice(&(w as u16).to_le_bytes());
    ivf.extend_from_slice(&(h as u16).to_le_bytes());
    for v in [24u32, 1, frames.len() as u32, 0] {
        ivf.extend_from_slice(&v.to_le_bytes());
    }
    for (i, f) in frames.iter().enumerate() {
        ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&(i as u64).to_le_bytes());
        ivf.extend_from_slice(f);
    }
    let mut c = Command::new(fastvideo_media::tools::ffmpeg_bin())
        .args(["-hide_banner", "-loglevel", "error", "-f", "ivf", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    let mut stdin = c.stdin.take().expect("stdin");
    let t = std::thread::spawn(move || stdin.write_all(&ivf));
    let out = c.wait_with_output().expect("ffmpeg decode");
    t.join().expect("writer").expect("write");
    out.stdout
}

/// Luma PSNR (BT.601 Y from RGB): both encoders take 4:2:0, so an RGB PSNR
/// would mostly measure chroma subsampling.
fn psnr(src: &[RgbFrame], dec: &[u8]) -> f64 {
    let y = |p: &[u8]| 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]);
    let mut se = 0f64;
    let mut n = 0usize;
    for (f, d) in src.iter().zip(dec.chunks_exact(src[0].data.len())) {
        for (a, b) in f.data.chunks_exact(3).zip(d.chunks_exact(3)) {
            let e = y(a) - y(b);
            se += e * e;
            n += 1;
        }
    }
    if se == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / (se / n as f64)).log10()
}

fn report(name: &str, src: &[RgbFrame], frames: &[(Bytes, bool)], secs: f64, took: f64) {
    let bytes: usize = frames.iter().map(|f| f.0.len()).sum();
    let keys = frames.iter().filter(|f| f.1).count();
    let data: Vec<Bytes> = frames.iter().map(|f| f.0.clone()).collect();
    let dec = decode(src[0].width, src[0].height, &data);
    println!(
        "{name:<28} {:>7.0} kb/s  {:>5.1} dB Y-PSNR  {keys:>3} key / {:>3} frames  encode {:>5.1} fps",
        bytes as f64 * 8.0 / secs / 1000.0,
        psnr(src, &dec),
        frames.len(),
        frames.len() as f64 / took
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size = arg(&args, "--size").unwrap_or_else(|| "1344x768".into());
    let (w, h) = size.split_once('x').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).expect("--size WxH");
    let fps: u32 = arg(&args, "--fps").map_or(24, |v| v.parse().unwrap());
    let secs: f64 = arg(&args, "--seconds").map_or(4.0, |v| v.parse().unwrap());
    let source = arg(&args, "--source").unwrap_or_else(|| "testsrc2".into());
    let n = (secs * f64::from(fps)) as usize;
    let src: Vec<RgbFrame> = match source.as_str() {
        "fake" => (0..n as u64).map(|i| fastvideo_engine_service::fake::render_frame(7, "a lighthouse", w, h, i)).collect(),
        _ => testsrc2(w, h, fps, n),
    };
    println!("source {source} {w}x{h} @ {fps} fps, {} frames", src.len());

    let t = Instant::now();
    let mut webp = Webp::new(w, h, 70.0).expect("libwebp (feature vp8)");
    let mut out = Vec::new();
    for (i, f) in src.iter().enumerate() {
        out.extend(webp.encode(f, false, i as u64).expect("webp").into_iter().map(|(_, d)| (d, true)));
    }
    report("libwebp intra q70 (before)", &src, &out, secs, t.elapsed().as_secs_f64());

    let mut cfg = Vp8Config::new(w, h, fps);
    if let Some(b) = arg(&args, "--bitrate") {
        cfg.bitrate_bps = b.parse().unwrap();
    }
    let label = format!("libvpx CBR {} kb/s (after)", cfg.bitrate_bps / 1000);
    let t = Instant::now();
    let mut vpx = Libvpx::new(cfg).expect("ffmpeg libvpx");
    let mut out = Vec::new();
    for f in &src {
        out.extend(vpx.encode(f).expect("libvpx").into_iter().map(|e| (e.data, e.keyframe)));
    }
    out.extend(vpx.finish().expect("libvpx").into_iter().map(|e| (e.data, e.keyframe)));
    report(&label, &src, &out, secs, t.elapsed().as_secs_f64());
}
