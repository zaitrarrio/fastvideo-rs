//! ffmpeg-backed tests: MP4 structure (faststart, tracks, fal format),
//! `finalize`, probing, the ffmpeg pipe encoder and the RTMP/HLS/file sinks.
//! CPU CI has no NVENC, so these run the same ffmpeg plumbing with the
//! CPU-test codec (`FfmpegH264::Libx264CpuTest`); NVENC runs are GPU-only
//! (`tests/nvenc.rs`).
//! Each test skips (passes with a note) when ffmpeg/ffprobe are absent; set
//! `FV_FFMPEG`/`FV_FFPROBE` to point at binaries off `PATH`.

use std::path::Path;
use std::time::Duration;

use fastvideo_media::h264::{self, H264Level};
use fastvideo_media::mp4::{self, AudioTarget, Crop, Mp4Spec, PostProcess};
use fastvideo_media::probe::{self, MediaKind};
use fastvideo_media::sink::{FfmpegSink, SinkAudioIn, SinkConfig, SinkTarget, HLS_PLAYLIST};
use fastvideo_media::tools;
use fastvideo_media::video::{create_encoder, EncoderBackend, FfmpegH264, H264Config, PublishTarget};
use fastvideo_media::{Pcm, RgbFrame};

fn have_ffmpeg() -> bool {
    let ok = tools::ffmpeg_available();
    if !ok {
        eprintln!("skipping: ffmpeg not available (set FV_FFMPEG)");
    }
    ok
}

/// A moving gradient with the frame index burned into the red channel.
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

fn tone(rate: u32, channels: u8, secs: f64) -> Pcm {
    let n = (f64::from(rate) * secs) as usize;
    let mut v = Vec::with_capacity(n * channels as usize);
    for i in 0..n {
        let s = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / f64::from(rate)).sin() as f32 * 0.3;
        for _ in 0..channels {
            v.push(s);
        }
    }
    Pcm::new(rate, channels, v)
}

/// CPU CI stand-in for NVENC in the ffmpeg-based writers.
fn cpu(mut spec: Mp4Spec) -> Mp4Spec {
    spec.encoder = FfmpegH264::Libx264CpuTest;
    spec
}

fn frames(w: u32, h: u32, n: u64) -> Vec<RgbFrame> {
    (0..n).map(|i| pattern(w, h, i)).collect()
}

#[test]
fn fal_h3_mp4_format() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("h3.mp4");
    // H3 native audio: 32 kHz stereo; a 1 s clip (24 frames) at a small canvas.
    let native = tone(32_000, 2, 1.1);
    mp4::write_mp4(&out, cpu(Mp4Spec::fal_h3(128, 72)), &frames(128, 72, 24), Some(&native)).unwrap();
    let info = mp4::inspect(&out).unwrap();
    assert!(info.fal_h3_problems().is_empty(), "{:?}\n{info:#?}", info.fal_h3_problems());
    assert!(info.top_level.iter().position(|t| t == "moov") < info.top_level.iter().position(|t| t == "mdat"));
    let v = info.video().unwrap();
    assert_eq!((v.width, v.height, v.samples), (Some(128), Some(72), 24));
    let a = info.audio().unwrap().audio.clone().unwrap();
    assert_eq!((a.sample_rate, a.channels, a.object_type), (32_000, 2, Some(2)));
    if tools::ffprobe_available() {
        let p = probe::ffprobe(&out).unwrap();
        assert_eq!(p.kind, Some(MediaKind::Video));
        assert_eq!(p.video_codec.as_deref(), Some("h264"));
        assert_eq!(p.audio_codec.as_deref(), Some("aac"));
        assert_eq!(p.audio_profile.as_deref(), Some("LC"));
        assert_eq!((p.audio_rate, p.audio_channels), (Some(32_000), Some(2)));
        assert_eq!(p.fps, Some(24.0));
    }
    // The decoded audio is the lockstep length: 1 s at 32 kHz (+AAC padding trimmed).
    let back = probe::decode_audio(&out, 32_000, 2).unwrap();
    assert!((back.frames() as i64 - 32_000).abs() <= 1024, "{}", back.frames());
}

#[test]
fn mp4_44k_target_and_video_only() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("cd.mp4");
    mp4::write_mp4(&a, cpu(Mp4Spec::new(64, 64, 16, Some(AudioTarget::CD))), &frames(64, 64, 16), Some(&tone(48_000, 1, 1.0)))
        .unwrap();
    let ia = mp4::inspect(&a).unwrap();
    let aa = ia.audio().unwrap().audio.clone().unwrap();
    assert_eq!((aa.sample_rate, aa.channels), (44_100, 2));
    assert_eq!(ia.video().unwrap().fps, Some(16.0)); // SF-Wan rate
    // Video-only (Wan): no audio track at all.
    let b = dir.path().join("wan.mp4");
    mp4::write_mp4(&b, cpu(Mp4Spec::new(64, 64, 16, None)), &frames(64, 64, 17), None).unwrap();
    let ib = mp4::inspect(&b).unwrap();
    assert!(ib.audio().is_none());
    assert!(ib.faststart);
    assert_eq!(ib.video().unwrap().samples, 17);
}

#[test]
fn finalize_faststart_drop_audio_and_crop() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.mp4");
    let mut spec = cpu(Mp4Spec::new(64, 72, 24, Some(AudioTarget::FAL_H3)));
    spec.faststart = false;
    mp4::write_mp4(&src, spec, &frames(64, 72, 12), Some(&tone(32_000, 2, 0.5))).unwrap();
    assert!(!mp4::inspect(&src).unwrap().faststart);

    let fast = dir.path().join("fast.mp4");
    mp4::finalize_with(&src, &fast, &PostProcess::default(), FfmpegH264::Libx264CpuTest).unwrap();
    let i = mp4::inspect(&fast).unwrap();
    assert!(i.faststart);
    assert!(i.audio().is_some());
    assert_eq!(i.video().unwrap().samples, 12);

    let silent = dir.path().join("silent.mp4");
    mp4::finalize_with(&src, &silent, &PostProcess { crop: None, drop_audio: true }, FfmpegH264::Libx264CpuTest).unwrap();
    let i = mp4::inspect(&silent).unwrap();
    assert!(i.faststart && i.audio().is_none());

    // Pad-and-crop: 64x72 -> 64x64 (the 1920x1088 -> 1920x1080 case in miniature).
    let cropped = dir.path().join("crop.mp4");
    let post = PostProcess { crop: Some(Crop { width: 64, height: 64, x: None, y: None }), drop_audio: false };
    mp4::finalize_with(&src, &cropped, &post, FfmpegH264::Libx264CpuTest).unwrap();
    let i = mp4::inspect(&cropped).unwrap();
    assert_eq!((i.video().unwrap().width, i.video().unwrap().height), (Some(64), Some(64)));
    assert!(i.faststart && i.audio().is_some());
    assert_eq!(i.video().unwrap().samples, 12);

    // Probe falls back to the box reader for MP4 facts too.
    let p = probe::probe(&cropped).unwrap();
    assert_eq!((p.width, p.height), (Some(64), Some(64)));
    // Decoding frames back gives the cropped size and count.
    let back = probe::decode_video_rgb(&cropped, None, None).unwrap();
    assert_eq!(back.len(), 12);
    assert_eq!((back[0].width, back[0].height), (64, 64));
}

#[test]
fn pipe_encoder_gop_profile_level() {
    if !have_ffmpeg() {
        return;
    }
    let cfg = H264Config::new(1344, 768, 24);
    let mut enc = create_encoder(EncoderBackend::CpuTestX264, cfg).unwrap();
    let mut aus = Vec::new();
    for i in 0..60 {
        aus.extend(enc.encode(&pattern(1344, 768, i)).unwrap());
    }
    aus.extend(enc.finish().unwrap());
    assert_eq!(aus.len(), 60);
    let keys: Vec<u64> = aus.iter().filter(|a| a.keyframe).map(|a| a.index).collect();
    assert_eq!(keys, vec![0, 48], "IDR every 2 s at 24 fps, no scene-cut IDRs");
    let sps = h264::find_sps(&aus[0].data).expect("sps in the first AU");
    assert!(sps.is_constrained_baseline(), "{sps:?}");
    assert_eq!((sps.width, sps.height), (1344, 768));
    assert_eq!(sps.level_idc, H264Level::L3_2.idc());
    // Headers repeat at every IDR (joiners mid-stream).
    assert!(h264::find_sps(&aus[48].data).is_some());
    if tools::ffprobe_available() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.h264");
        let all: Vec<u8> = aus.iter().flat_map(|a| a.data.iter().copied()).collect();
        std::fs::write(&p, all).unwrap();
        let pr = probe::ffprobe(&p).unwrap();
        assert_eq!(pr.video_profile.as_deref(), Some("Constrained Baseline"));
        assert_eq!((pr.width, pr.height), (Some(1344), Some(768)));
    }
}

fn wait_for(p: &Path, secs: u64) -> bool {
    let t = std::time::Instant::now();
    while t.elapsed() < Duration::from_secs(secs) {
        if p.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

#[test]
fn file_sink_video_only_still_has_audio() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("rec.mp4");
    let mut cfg = SinkConfig::new(SinkTarget::File { path: out.clone() }, 64, 64, 16, None);
    cfg.encoder = FfmpegH264::Libx264CpuTest;
    let mut s = FfmpegSink::start(cfg).unwrap();
    for i in 0..32 {
        s.send(&pattern(64, 64, i), None).unwrap();
        std::thread::sleep(Duration::from_millis(2));
    }
    let st = s.finish(Duration::from_secs(20)).unwrap();
    assert_eq!(st.frames_sent, 32);
    let i = mp4::inspect(&out).unwrap();
    assert!(i.faststart);
    let a = i.audio().expect("a silent AAC track is always present").audio.clone().unwrap();
    assert_eq!((a.sample_rate, a.channels, a.object_type), (48_000, 2, Some(2)));
    assert_eq!(i.video().unwrap().samples, 32);
}

#[test]
fn file_sink_av_over_two_pipes() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("av.mp4");
    let mut cfg = SinkConfig::new(
        SinkTarget::File { path: out.clone() },
        64,
        64,
        24,
        Some(SinkAudioIn { rate: 48_000, channels: 1 }),
    );
    cfg.encoder = FfmpegH264::Libx264CpuTest;
    let mut s = FfmpegSink::start(cfg).unwrap();
    let tick = vec![0.2f32; 2000];
    for i in 0..48 {
        s.send(&pattern(64, 64, i), Some(&tick)).unwrap();
        std::thread::sleep(Duration::from_millis(2));
    }
    let st = s.finish(Duration::from_secs(20)).unwrap();
    assert_eq!((st.frames_sent, st.audio_ticks_sent, st.dropped_video, st.dropped_audio), (48, 48, 0, 0));
    let i = mp4::inspect(&out).unwrap();
    assert_eq!(i.video().unwrap().samples, 48);
    let a = i.audio().unwrap();
    assert!((a.duration_s() - 2.0).abs() < 0.1, "audio {} s", a.duration_s());
}

#[test]
fn hls_sink_writes_a_playlist() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let hls = dir.path().join("hls");
    let mut cfg = SinkConfig::new(SinkTarget::Hls { dir: hls.clone(), segment_s: 1, window: 6 }, 64, 64, 24, None);
    cfg.encoder = FfmpegH264::Libx264CpuTest;
    let mut s = FfmpegSink::start(cfg).unwrap();
    for i in 0..72 {
        s.send(&pattern(64, 64, i), None).unwrap();
        std::thread::sleep(Duration::from_millis(2));
    }
    s.finish(Duration::from_secs(20)).unwrap();
    assert!(wait_for(&hls.join(HLS_PLAYLIST), 5));
    let pl = std::fs::read_to_string(hls.join(HLS_PLAYLIST)).unwrap();
    assert!(pl.contains("#EXTM3U"));
    assert!(pl.contains("segment-"));
}

#[test]
fn pipe_encoder_cloudflare_scale_and_forced_idr_restart() {
    if !have_ffmpeg() {
        return;
    }
    // H3 1344x768 published to Cloudflare: encoded at 1280x720, level 3.1.
    let cfg = H264Config::for_publish(PublishTarget::Cloudflare, 1344, 768, 24);
    let mut enc = create_encoder(EncoderBackend::CpuTestX264, cfg).unwrap();
    let mut aus = Vec::new();
    for i in 0..40 {
        if i == 20 {
            enc.force_idr(); // PLI: the pipe encoder restarts ffmpeg
        }
        aus.extend(enc.encode(&pattern(1344, 768, i)).unwrap());
    }
    aus.extend(enc.finish().unwrap());
    assert_eq!(aus.len(), 40);
    assert!(aus.windows(2).all(|w| w[1].index == w[0].index + 1));
    let keys: Vec<u64> = aus.iter().filter(|a| a.keyframe).map(|a| a.index).collect();
    assert_eq!(keys, vec![0, 20]);
    let sps = h264::find_sps(&aus[20].data).unwrap();
    assert_eq!((sps.width, sps.height, sps.level_idc), (1280, 720, 31));
    assert!(sps.is_constrained_baseline());
}
