//! The session spare pool of the ffmpeg pipe encoders
//! (`fastvideo_media::pipe`): a forced keyframe swaps in the primed spare,
//! the old process flushes its last frames after EOF (exact frame counts,
//! keyframes exactly where forced, nothing from the primers), a new spare
//! follows; a pre-warmed spare becomes the first process; a keyframe asked
//! for before the first one went out needs no restart; one spare per pool,
//! killed with the pool; batch encoders have none. Skips (passes with a
//! note) a codec ffmpeg lacks.
//!
//! `restart_latency_cold_vs_warm` (ignored) measures restart latency with
//! and without the spare: `cargo test -p fastvideo-media --test warm_spare
//! -- --ignored --nocapture`.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use fastvideo_media::pipe::{RestartStats, SparePool, SpareKind, SpareSpec};
use fastvideo_media::tools;
use fastvideo_media::video::{EncodedFrame, FfmpegH264, H264Config, PipeEncoder, VideoEncoder};
use fastvideo_media::vp8::{Vp8Config, Vp8Encoder};
use fastvideo_media::RgbFrame;

const FRAME: Duration = Duration::from_millis(42);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Codec {
    X264,
    Libvpx,
}

fn has_encoder(name: &str) -> bool {
    let out = Command::new(tools::ffmpeg_bin()).args(["-hide_banner", "-encoders"]).stderr(Stdio::null()).output();
    let ok = out.is_ok_and(|o| String::from_utf8_lossy(&o.stdout).split_whitespace().any(|w| w == name));
    if !ok {
        eprintln!("skipping {name}: not in this ffmpeg (set FV_FFMPEG)");
    }
    ok
}

fn codecs() -> Vec<Codec> {
    let mut v = Vec::new();
    if has_encoder("libx264") {
        v.push(Codec::X264);
    }
    if has_encoder("libvpx") {
        v.push(Codec::Libvpx);
    }
    v
}

/// No periodic keyframe within a test (GOP 100 s).
fn h264_cfg(w: u32, h: u32) -> H264Config {
    let mut cfg = H264Config::new(w, h, 24);
    cfg.gop_seconds = 100.0;
    cfg
}

fn vp8_cfg(w: u32, h: u32) -> Vp8Config {
    let mut cfg = Vp8Config::new(w, h, 24);
    cfg.gop_seconds = 100.0;
    cfg.bitrate_bps = 500_000;
    cfg
}

fn spec(c: Codec, w: u32, h: u32) -> SpareSpec {
    match c {
        Codec::X264 => SpareSpec::h264(FfmpegH264::Libx264CpuTest, &h264_cfg(w, h)).unwrap(),
        Codec::Libvpx => SpareSpec::vp8(&vp8_cfg(w, h)).unwrap(),
    }
}

/// Both pipe encoders behind one face.
enum Enc {
    H264(PipeEncoder),
    Vp8(Vp8Encoder),
}

impl Enc {
    fn new(c: Codec, w: u32, h: u32, pool: Option<&SparePool>) -> Enc {
        match c {
            Codec::X264 => {
                Enc::H264(PipeEncoder::with_pool(h264_cfg(w, h), FfmpegH264::Libx264CpuTest, pool.cloned()).unwrap())
            }
            Codec::Libvpx => Enc::Vp8(Vp8Encoder::with_pool(vp8_cfg(w, h), pool.cloned()).unwrap()),
        }
    }
    fn key(&mut self) {
        match self {
            Enc::H264(e) => e.force_idr(),
            Enc::Vp8(e) => e.force_keyframe(),
        }
    }
    fn encode(&mut self, f: &RgbFrame) -> Vec<EncodedFrame> {
        match self {
            Enc::H264(e) => e.encode(f).unwrap(),
            Enc::Vp8(e) => e.encode(f).unwrap(),
        }
    }
    fn poll(&mut self) -> Vec<EncodedFrame> {
        match self {
            Enc::H264(e) => e.poll().unwrap(),
            Enc::Vp8(e) => e.poll().unwrap(),
        }
    }
    fn finish(&mut self) -> Vec<EncodedFrame> {
        match self {
            Enc::H264(e) => e.finish().unwrap(),
            Enc::Vp8(e) => e.finish().unwrap(),
        }
    }
    fn pids(&self) -> Vec<u32> {
        match self {
            Enc::H264(e) => e.pids(),
            Enc::Vp8(e) => e.pids(),
        }
    }
    fn stats(&self) -> RestartStats {
        match self {
            Enc::H264(e) => e.restart_stats().unwrap().clone(),
            Enc::Vp8(e) => e.restart_stats().clone(),
        }
    }
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

/// Decodes the stream with ffmpeg; returns the decoded frame count.
fn decode_count(c: Codec, w: u32, h: u32, frames: &[EncodedFrame]) -> usize {
    let mut input = Vec::new();
    let fmt = match c {
        Codec::X264 => {
            for f in frames {
                input.extend_from_slice(&f.data);
            }
            "h264"
        }
        Codec::Libvpx => {
            input.extend_from_slice(b"DKIF\0\0\x20\0VP80");
            input.extend_from_slice(&(w as u16).to_le_bytes());
            input.extend_from_slice(&(h as u16).to_le_bytes());
            input.extend_from_slice(&24u32.to_le_bytes());
            input.extend_from_slice(&1u32.to_le_bytes());
            input.extend_from_slice(&(frames.len() as u32).to_le_bytes());
            input.extend_from_slice(&0u32.to_le_bytes());
            for (i, f) in frames.iter().enumerate() {
                input.extend_from_slice(&(f.data.len() as u32).to_le_bytes());
                input.extend_from_slice(&(i as u64).to_le_bytes());
                input.extend_from_slice(&f.data);
            }
            "ivf"
        }
    };
    let mut p = Command::new(tools::ffmpeg_bin())
        .args(["-hide_banner", "-loglevel", "error", "-f", fmt, "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = p.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&input));
    let out = p.wait_with_output().unwrap();
    writer.join().unwrap().unwrap();
    assert!(out.status.success(), "decode failed: {}", String::from_utf8_lossy(&out.stderr));
    out.stdout.len() / (w * h * 3) as usize
}

fn alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

/// Encodes paced frames (24 fps) until `until` holds or `limit` passes.
fn run_until(
    e: &mut Enc,
    out: &mut Vec<EncodedFrame>,
    next: &mut u64,
    (w, h): (u32, u32),
    limit: Duration,
    mut until: impl FnMut(&mut Enc) -> bool,
) -> bool {
    let end = Instant::now() + limit;
    while Instant::now() < end {
        out.extend(e.encode(&pattern(w, h, *next)));
        *next += 1;
        std::thread::sleep(FRAME);
        out.extend(e.poll());
        if until(e) {
            return true;
        }
    }
    false
}

fn wait_primed(pool: &SparePool, limit: Duration) -> Option<u32> {
    let end = Instant::now() + limit;
    while Instant::now() < end {
        if let Some((pid, true)) = pool.spare_state() {
            return Some(pid);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

/// Exact: one frame out per frame in, consecutive indices, keyframes at
/// `keys`, and the stream decodes to `n` frames.
fn assert_exact(c: Codec, (w, h): (u32, u32), out: &[EncodedFrame], n: u64, keys: &[u64]) {
    assert_eq!(out.len() as u64, n, "{c:?}: {} frames out for {n} in", out.len());
    assert!(out.iter().enumerate().all(|(i, f)| f.index == i as u64), "{c:?}");
    let got: Vec<u64> = out.iter().filter(|f| f.keyframe).map(|f| f.index).collect();
    assert_eq!(got, keys, "{c:?}: keyframes");
    assert_eq!(decode_count(c, w, h, out), n as usize, "{c:?}");
}

#[test]
fn warm_swap_flushes_the_old_process_and_swaps_in_the_spare() {
    let wh = (320u32, 192u32);
    for c in codecs() {
        let pool = SparePool::new(true);
        let mut e = Enc::new(c, wh.0, wh.1, Some(&pool));
        assert_eq!(e.stats().start, Some(SpareKind::Cold), "{c:?}: nothing to adopt yet");
        let (mut out, mut n) = (Vec::new(), 0u64);
        // The spare starts after the first frame is out, then primes.
        run_until(&mut e, &mut out, &mut n, wh, Duration::from_secs(30), |_| pool.spare_state().is_some());
        let spare = wait_primed(&pool, Duration::from_secs(30)).unwrap_or_else(|| panic!("{c:?}: spare not primed"));
        let old = e.pids()[0];
        // Forced keyframe: the spare becomes the encoder at once, the old
        // process flushes after EOF.
        let forced_at = n;
        e.key();
        out.extend(e.encode(&pattern(wh.0, wh.1, n)));
        n += 1;
        let pids = e.pids();
        assert!(pids[0] == spare && pids[1..].iter().all(|p| *p == old), "{c:?}: the spare is current: {pids:?}");
        // ...and a new spare follows once its first frame is out.
        let refilled = run_until(&mut e, &mut out, &mut n, wh, Duration::from_secs(30), |_| {
            pool.spare_state().is_some_and(|(p, primed)| p != spare && primed)
        });
        assert!(refilled, "{c:?}: no new spare");
        let (next_spare, _) = pool.spare_state().unwrap();
        let reaped = run_until(&mut e, &mut out, &mut n, wh, Duration::from_secs(20), |e| e.pids() == vec![spare]);
        assert!(reaped, "{c:?}: the old process was not reaped: {:?}", e.pids());
        out.extend(e.finish());
        assert_exact(c, wh, &out, n, &[0, forced_at]);
        let s = e.stats();
        assert_eq!((s.restarts, s.warm, s.warming, s.cold, s.skipped), (1, 1, 0, 0, 0), "{c:?}: {s:?}");
        assert_eq!(s.latencies.len(), 1);
        assert_eq!(s.latencies[0].0, SpareKind::Warm);
        eprintln!("{c:?}: warm restart, first frame after {:?}", s.latencies[0].1);

        // Session end: the pool's last handle goes, and the spare with it.
        drop(e);
        assert!(alive(next_spare), "{c:?}: the spare belongs to the session, not the encoder");
        drop(pool);
        for p in [old, spare, next_spare] {
            assert!(!alive(p), "{c:?}: ffmpeg {p} outlived its session");
        }
    }
}

#[test]
fn a_prewarmed_spare_is_the_first_process() {
    let wh = (320u32, 192u32);
    for c in codecs() {
        let pool = SparePool::new(true);
        let s = spec(c, wh.0, wh.1);
        pool.prewarm_with(move || Some(s));
        let spare = wait_primed(&pool, Duration::from_secs(30)).unwrap_or_else(|| panic!("{c:?}: not primed"));
        let mut e = Enc::new(c, wh.0, wh.1, Some(&pool));
        assert_eq!(e.pids(), vec![spare], "{c:?}: adopted");
        assert_eq!(e.stats().start, Some(SpareKind::Warm));
        let (mut out, mut n) = (Vec::new(), 0u64);
        run_until(&mut e, &mut out, &mut n, wh, Duration::from_secs(2), |_| false);
        out.extend(e.finish());
        // The primer's frame is not in the stream; the first real frame is
        // the keyframe.
        assert_exact(c, wh, &out, n, &[0]);
        // A spare of another profile is not adopted.
        let other = spec(c, 64, 48);
        pool.prewarm_with(move || Some(other));
        let e2 = Enc::new(c, wh.0, wh.1, Some(&pool));
        assert_eq!(e2.stats().start, Some(SpareKind::Cold), "{c:?}");
        assert!(pool.spare_state().is_some(), "{c:?}: the other profile's spare stays");
    }
}

#[test]
fn a_keyframe_before_the_first_one_went_out_needs_no_restart() {
    let wh = (320u32, 192u32);
    for c in codecs() {
        let pool = SparePool::new(true);
        let mut e = Enc::new(c, wh.0, wh.1, Some(&pool));
        // A new peer's keyframe request while ffmpeg is still starting.
        e.key();
        let (mut out, mut n) = (Vec::new(), 0u64);
        run_until(&mut e, &mut out, &mut n, wh, Duration::from_secs(1), |_| false);
        out.extend(e.finish());
        assert_exact(c, wh, &out, n, &[0]);
        let s = e.stats();
        assert_eq!((s.restarts, s.skipped), (0, 1), "{c:?}: {s:?}");
    }
}

#[test]
fn batch_encoders_have_no_spare_and_a_pool_holds_one() {
    let wh = (64u32, 48u32);
    let Some(c) = codecs().into_iter().next() else { return };
    // Batch (no pool): a forced keyframe starts ffmpeg cold, as before.
    let mut batch = Enc::new(c, wh.0, wh.1, None);
    let (mut out, mut n) = (Vec::new(), 0u64);
    run_until(&mut batch, &mut out, &mut n, wh, Duration::from_millis(600), |_| false);
    assert_eq!(batch.pids().len(), 1);
    let forced_at = n;
    batch.key();
    run_until(&mut batch, &mut out, &mut n, wh, Duration::from_millis(500), |_| false);
    out.extend(batch.finish());
    assert_exact(c, wh, &out, n, &[0, forced_at]);
    assert_eq!(batch.stats().cold, 1);
    let pool = SparePool::new(false);
    let mut off = Enc::new(c, wh.0, wh.1, Some(&pool));
    let (mut o, mut k) = (Vec::new(), 0);
    run_until(&mut off, &mut o, &mut k, wh, Duration::from_secs(1), |_| false);
    assert!(pool.spare_state().is_none(), "a disabled pool (FV_ENCODER_SPARE=0) keeps none");

    // Two encoders of one session (two profiles): one spare between them.
    let pool = SparePool::new(true);
    let (mut a, mut b) = (Enc::new(c, wh.0, wh.1, Some(&pool)), Enc::new(c, 96, 64, Some(&pool)));
    let (mut oa, mut ob, mut na, mut nb) = (Vec::new(), Vec::new(), 0, 0);
    assert!(run_until(&mut a, &mut oa, &mut na, wh, Duration::from_secs(20), |_| pool.spare_state().is_some()));
    let (first, _) = pool.spare_state().unwrap();
    assert!(run_until(&mut b, &mut ob, &mut nb, (96, 64), Duration::from_secs(20), |_| {
        pool.spare_state().is_some_and(|(p, _)| p != first)
    }));
    assert!(!alive(first), "{c:?}: replaced, not kept");
    // Each refills once per process start: no ping-pong.
    let (second, _) = pool.spare_state().unwrap();
    run_until(&mut a, &mut oa, &mut na, wh, Duration::from_millis(500), |_| false);
    assert_eq!(pool.spare_state().map(|s| s.0), Some(second));
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    v.get(v.len() / 2).copied().unwrap_or_default()
}

/// Restart latency (restart to the new process's first frame) with and
/// without the warm spare, at 832x480 and 24 fps, a forced keyframe every
/// 1.5 s.
#[test]
#[ignore]
fn restart_latency_cold_vs_warm() {
    let wh = (832u32, 480u32);
    for c in codecs() {
        for warm in [false, true] {
            let pool = SparePool::new(true);
            let mut e = Enc::new(c, wh.0, wh.1, warm.then_some(&pool));
            let (mut out, mut n) = (Vec::new(), 0u64);
            let t0 = Instant::now();
            for _ in 0..8 {
                run_until(&mut e, &mut out, &mut n, wh, Duration::from_millis(1500), |_| false);
                e.key();
            }
            run_until(&mut e, &mut out, &mut n, wh, Duration::from_millis(1500), |_| false);
            out.extend(e.finish());
            assert_eq!(out.len() as u64, n, "{c:?}");
            let s = e.stats();
            let mut l: Vec<Duration> = s.latencies.iter().map(|x| x.1).collect();
            let max = l.iter().max().copied().unwrap_or_default();
            let med = median(&mut l);
            eprintln!(
                "restart-latency codec={c:?} spare={} restarts={} warm={} warming={} cold={} median={med:?} max={max:?} wall={:?}",
                if warm { "on" } else { "off" },
                s.restarts,
                s.warm,
                s.warming,
                s.cold,
                t0.elapsed()
            );
        }
    }
}
