//! The client-video decode pipe (`fastvideo_media::decode`): VP8 (libvpx)
//! and H.264 (libx264) encoded here, decoded back to RGB at a different
//! size, colours kept, timestamps in order, a resolution change mid-stream
//! padded into the same output size, H.264 out within one frame time, and
//! the warm spare taking over on a restart. Skips (passes with a note) when
//! ffmpeg or an encoder is missing.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use fastvideo_media::decode::{is_keyframe, keyframe_size, DecodedPicture, DecoderPool, VideoDecoder, VideoDecoderConfig};
use fastvideo_media::tools;
use fastvideo_media::video::{EncodedFrame, FfmpegH264, H264Config, PipeEncoder, VideoEncoder};
use fastvideo_media::vp8::{Vp8Config, Vp8Encoder};
use fastvideo_media::RgbFrame;
use fastvideo_protocol::InputVideoCodec;

fn has_encoder(name: &str) -> bool {
    let out = Command::new(tools::ffmpeg_bin()).args(["-hide_banner", "-encoders"]).stderr(Stdio::null()).output();
    let ok = out.is_ok_and(|o| String::from_utf8_lossy(&o.stdout).split_whitespace().any(|w| w == name));
    if !ok {
        eprintln!("skipping {name}: not in this ffmpeg (set FV_FFMPEG)");
    }
    ok
}

/// A frame of one colour with a white 16x16 square at column `i*4`.
fn frame(w: u32, h: u32, rgb: [u8; 3], i: u64) -> RgbFrame {
    let mut f = RgbFrame::solid(w, h, rgb, i);
    let mut d = f.data.to_vec();
    let x0 = (i as u32 * 4) % (w - 16);
    for y in 8..24 {
        for x in x0..x0 + 16 {
            let o = (y * w + x) as usize * 3;
            d[o..o + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    f.data = d.into();
    f
}

enum Enc {
    H264(PipeEncoder),
    Vp8(Vp8Encoder),
}

impl Enc {
    fn new(codec: InputVideoCodec, w: u32, h: u32) -> Enc {
        match codec {
            InputVideoCodec::H264 => {
                let mut c = H264Config::new(w, h, 30);
                c.gop_seconds = 100.0;
                Enc::H264(PipeEncoder::new(c, FfmpegH264::Libx264CpuTest).unwrap())
            }
            InputVideoCodec::Vp8 => {
                let mut c = Vp8Config::new(w, h, 30);
                c.gop_seconds = 100.0;
                Enc::Vp8(Vp8Encoder::new(c).unwrap())
            }
        }
    }

    /// Every frame of `frames`, encoded, in order (waits for the encoder).
    fn encode_all(&mut self, frames: &[RgbFrame]) -> Vec<EncodedFrame> {
        let mut out = Vec::new();
        for f in frames {
            out.extend(match self {
                Enc::H264(e) => e.encode(f).unwrap(),
                Enc::Vp8(e) => e.encode(f).unwrap(),
            });
        }
        out.extend(match self {
            Enc::H264(e) => e.finish().unwrap(),
            Enc::Vp8(e) => e.finish().unwrap(),
        });
        assert_eq!(out.len(), frames.len());
        out
    }
}

fn codecs() -> Vec<InputVideoCodec> {
    if !tools::ffmpeg_available() {
        eprintln!("skipping: no ffmpeg");
        return Vec::new();
    }
    let mut v = Vec::new();
    if has_encoder("libx264") {
        v.push(InputVideoCodec::H264);
    }
    if has_encoder("libvpx") {
        v.push(InputVideoCodec::Vp8);
    }
    v
}

fn close(a: [u8; 3], b: [u8; 3], tol: i32) -> bool {
    a.iter().zip(b).all(|(x, y)| (i32::from(*x) - i32::from(y)).abs() <= tol)
}

fn decode_all(dec: &mut VideoDecoder, frames: &[EncodedFrame], pts0: u64) -> Vec<DecodedPicture> {
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        dec.push(&f.data, pts0 + i as u64 * 33_333).unwrap();
        out.extend(dec.poll().unwrap());
    }
    out
}

#[test]
fn decodes_scales_and_keeps_order() {
    for codec in codecs() {
        let colours = [[200u8, 30, 30], [30, 200, 30], [30, 30, 200]];
        let src: Vec<RgbFrame> = (0..30).map(|i| frame(320, 240, colours[i as usize / 10], i)).collect();
        let enc = Enc::new(codec, 320, 240).encode_all(&src);
        assert!(enc[0].keyframe && is_keyframe(codec, &enc[0].data), "{codec:?}: first frame is a keyframe");
        assert_eq!(keyframe_size(codec, &enc[0].data), Some((320, 240)), "{codec:?}");
        let mut dec = VideoDecoder::new(VideoDecoderConfig { codec, width: 160, height: 120 }).unwrap();
        let mut got = decode_all(&mut dec, &enc, 1_000);
        got.extend(dec.finish().unwrap());
        assert_eq!(got.len(), 30, "{codec:?}: one picture per frame");
        for (i, p) in got.iter().enumerate() {
            assert_eq!((p.frame.width, p.frame.height), (160, 120));
            assert_eq!(p.pts_us, 1_000 + i as u64 * 33_333, "{codec:?}: pts of picture {i}");
            // Same aspect: no padding; the body colour survives.
            let c = p.frame.pixel(80, 100).unwrap();
            assert!(close(c, colours[i / 10], 40), "{codec:?} picture {i}: {c:?}");
        }
        let s = dec.stats();
        assert_eq!((s.frames_in, s.frames_out), (30, 30));
    }
}

#[test]
fn resolution_change_is_padded_into_the_same_size() {
    for codec in codecs() {
        let a: Vec<RgbFrame> = (0..6).map(|i| frame(320, 240, [220, 220, 40], i)).collect();
        let b: Vec<RgbFrame> = (0..6).map(|i| frame(240, 240, [40, 220, 220], i)).collect();
        let mut enc = Enc::new(codec, 320, 240).encode_all(&a);
        let second = Enc::new(codec, 240, 240).encode_all(&b);
        assert_eq!(keyframe_size(codec, &second[0].data), Some((240, 240)));
        enc.extend(second);
        let mut dec = VideoDecoder::new(VideoDecoderConfig { codec, width: 160, height: 120 }).unwrap();
        let mut got = decode_all(&mut dec, &enc, 0);
        got.extend(dec.finish().unwrap());
        assert_eq!(got.len(), 12, "{codec:?}");
        let last = &got[11].frame;
        assert_eq!((last.width, last.height), (160, 120));
        // A square picture in a 4:3 frame: black bars left and right.
        assert!(close(last.pixel(5, 60).unwrap(), [0, 0, 0], 24), "{codec:?}: {:?}", last.pixel(5, 60));
        assert!(close(last.pixel(80, 100).unwrap(), [40, 220, 220], 40), "{codec:?}: {:?}", last.pixel(80, 100));
    }
}

#[test]
fn h264_pictures_are_at_most_one_frame_behind() {
    one_frame_behind(false);
}

/// The same, and each picture comes out within 250 ms: a latency, which a
/// loaded host cannot keep (scripts/serve/check.sh --realtime runs it
/// serialized; docs/dev/testing.md).
#[test]
#[ignore = "real-time latency: scripts/serve/check.sh --realtime"]
fn realtime_h264_pictures_are_at_most_one_frame_behind() {
    one_frame_behind(true);
}

fn one_frame_behind(realtime: bool) {
    if !codecs().contains(&InputVideoCodec::H264) {
        return;
    }
    let src: Vec<RgbFrame> = (0..20).map(|i| frame(320, 240, [90, 90, 90], i)).collect();
    let enc = Enc::new(InputVideoCodec::H264, 320, 240).encode_all(&src);
    // A pre-started process (as a session has), so ffmpeg's start on a
    // loaded host is not measured.
    let cfg = VideoDecoderConfig { codec: InputVideoCodec::H264, width: 320, height: 240 };
    let pool = DecoderPool::new(true);
    pool.prewarm(cfg);
    let t0 = Instant::now();
    while pool.warm().is_empty() && t0.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut dec = VideoDecoder::with_pool(cfg, pool).unwrap();
    // The first picture waits for the second frame (ffmpeg's stream probe).
    dec.push(&enc[0].data, 0).unwrap();
    let mut got = Vec::new();
    let mut worst = Duration::ZERO;
    for (i, f) in enc.iter().enumerate().skip(1) {
        let t0 = Instant::now();
        dec.push(&f.data, i as u64).unwrap();
        // At most one frame behind (ffmpeg 6.1: none; 5.1 holds one).
        while got.len() < i && t0.elapsed() < Duration::from_secs(30) {
            got.extend(dec.poll_wait(Duration::from_millis(20)).unwrap());
        }
        assert!(got.len() >= i, "picture {} came out", i - 1);
        if i >= 2 {
            // Frame 1 also waits for ffmpeg to start.
            worst = worst.max(t0.elapsed());
        }
    }
    got.extend(dec.finish().unwrap());
    assert_eq!(got.len(), enc.len());
    assert!(got.iter().enumerate().all(|(i, p)| p.pts_us == i as u64));
    eprintln!("h264 decode latency, worst once started: {worst:?}");
    if realtime {
        assert!(worst < Duration::from_millis(250), "{worst:?}");
    }
}

#[test]
fn restart_swaps_in_the_warm_spare() {
    let Some(codec) = codecs().first().copied() else { return };
    let src: Vec<RgbFrame> = (0..4).map(|i| frame(160, 120, [10, 120, 250], i)).collect();
    let enc = Enc::new(codec, 160, 120).encode_all(&src);
    let cfg = VideoDecoderConfig { codec, width: 160, height: 120 };
    let pool = DecoderPool::new(true);
    // The session pre-starts a decoder before any frame; the first decoder
    // adopts it, and the pool starts the decoder's spare.
    pool.prewarm(cfg);
    let t0 = Instant::now();
    while pool.warm().is_empty() && t0.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(pool.warm(), vec![cfg]);
    let mut dec = VideoDecoder::with_pool(cfg, pool.clone()).unwrap();
    let t0 = Instant::now();
    while !dec.has_warm_spare() && t0.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(dec.has_warm_spare(), "a spare follows");
    let before = dec.pid();
    dec.restart().unwrap();
    assert_ne!(dec.pid(), before);
    assert_eq!((dec.stats().restarts, dec.stats().warm_restarts), (1, 1));
    // The swapped-in process decodes from a keyframe.
    let mut got = decode_all(&mut dec, &enc, 0);
    got.extend(dec.finish().unwrap());
    assert_eq!(got.len(), 4);
    assert!(close(got[3].frame.pixel(80, 100).unwrap(), [10, 120, 250], 40));
    // Without a pool, a restart starts cold.
    let mut cold = VideoDecoder::with_spare(cfg, false).unwrap();
    cold.restart().unwrap();
    assert_eq!((cold.stats().restarts, cold.stats().warm_restarts), (1, 0));
}
