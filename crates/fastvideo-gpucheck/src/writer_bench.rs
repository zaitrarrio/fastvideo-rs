//! `writer-bench` (CPU only): the frame writer under a decode-paced producer.
//!
//! Synthetic frames (moving gradients plus a little noise, so PNG sizes are
//! in the range of real clips) are pushed in batches on the schedule of a
//! decode lasting `--produce-s`; the report says how long the producer was
//! blocked by the writer, when the mp4 was complete, and what the PNG frames
//! cost, for each `--png` mode asked for. With a writer that keeps up, the
//! "decode" (start → mp4 complete) is `--produce-s` plus the mp4's tail.

use std::path::Path;
use std::time::{Duration, Instant};

use fastvideo_cudarc::wan::writer::{PngMode, VideoWriter};
use serde_json::json;

use crate::report::{Report, StageResult};

pub struct Args<'a> {
    pub out: &'a Path,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub batch: usize,
    pub fps: u32,
    pub mp4: bool,
    pub modes: &'a str,
    pub produce_s: f64,
}

fn frame(w: usize, h: usize, index: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (index as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
    for y in 0..h {
        for x in 0..w {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state & 7) as i32 - 3;
            let p = (y * w + x) * 3;
            let t = index * 4;
            let v = [
                ((x + t) * 255 / w.max(1)) as i32,
                ((y + t / 2) * 255 / h.max(1)) as i32,
                (((x + y + t) / 8) % 256) as i32,
            ];
            for c in 0..3 {
                out[p + c] = (v[c] + noise).clamp(0, 255) as u8;
            }
        }
    }
    out
}

pub fn run(report: &mut Report, a: &Args<'_>) -> StageResult<()> {
    let pe = |e: fastvideo_cudarc::wan::pipeline::PipelineError| anyhow::anyhow!("{e}");
    report.set(
        "host",
        json!({
            "cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
            "rayon_threads": rayon::current_num_threads(),
            "available_gib": fastvideo_cudarc::wan::writer::available_memory()
                .map(|b| b as f64 / f64::from(1u32 << 30)),
        }),
    );
    // One set of frames reused by every mode (content cost stays out of it).
    let template: Vec<Vec<u8>> = (0..(2 * a.batch).min(a.frames).max(1))
        .map(|i| frame(a.width, a.height, i))
        .collect();
    let mut runs = Vec::new();
    for mode in a.modes.split(',').map(str::trim).filter(|m| !m.is_empty()) {
        let png = match mode {
            "inline" => PngMode::Inline,
            "deferred" => PngMode::Deferred,
            "off" => PngMode::Off,
            other => return Err(anyhow::anyhow!("--png {other}: inline, deferred or off").into()),
        };
        let dir = a.out.join(mode);
        let _ = std::fs::remove_dir_all(&dir);
        let start = Instant::now();
        let mut writer = VideoWriter::spawn_with(&dir, a.fps, a.mp4, None, png).map_err(pe)?;
        let (mut push_s, mut offset) = (0.0, 0usize);
        while offset < a.frames {
            let n = a.batch.min(a.frames - offset);
            let mut rgb = Vec::with_capacity(n * a.width * a.height * 3);
            for i in 0..n {
                rgb.extend_from_slice(&template[(offset + i) % template.len()]);
            }
            // The decode would hand this batch over once its frames exist.
            let due = a.produce_s * (offset + n) as f64 / a.frames as f64;
            let now = start.elapsed().as_secs_f64();
            if due > now {
                std::thread::sleep(Duration::from_secs_f64(due - now));
            }
            let t = Instant::now();
            writer.push(offset, a.height, a.width, rgb).map_err(pe)?;
            push_s += t.elapsed().as_secs_f64();
            offset += n;
        }
        let produced_s = start.elapsed().as_secs_f64();
        writer.finish_video().map_err(pe)?;
        let video_s = start.elapsed().as_secs_f64();
        let t = Instant::now();
        let (paths, mp4) = writer.finish().map_err(pe)?;
        let png_s = t.elapsed().as_secs_f64();
        let s = writer.stats();
        eprintln!(
            "writer-bench {mode}: {}x{}x{} produce {:.2}s → pushed by {produced_s:.2}s (blocked {push_s:.2}s), mp4 complete {video_s:.2}s, png after {png_s:.2}s ({} inline / {} deferred, {:.2} GiB), ffmpeg feed blocked {:.2}s",
            a.width, a.height, a.frames, a.produce_s, s.png_inline, s.png_deferred,
            s.png_bytes as f64 / f64::from(1u32 << 30), s.ffmpeg_feed_s
        );
        runs.push(json!({
            "png": mode, "mp4": mp4.is_some(), "produce_s": a.produce_s,
            "pushed_s": produced_s, "push_blocked_s": push_s,
            "decode_equiv_s": video_s, "mp4_tail_s": s.video_tail_s,
            "png_after_s": png_s, "frames": paths.len(),
            "writer": crate::ltx2_stage::writer_json(s),
        }));
        let _ = std::fs::remove_dir_all(&dir);
    }
    report.set(
        "request",
        json!({"width": a.width, "height": a.height, "frames": a.frames, "batch": a.batch, "fps": a.fps, "mp4": a.mp4}),
    );
    report.set("runs", runs);
    Ok(())
}
