# fastvideo-media

fv-serve media: A/V pacer, resampler, Opus and H.264 encoders, MP4 finalize, probing, sinks. See [docs/serve/design.md](../../docs/serve/design.md) (§0 owner decisions, WP-03, §5.1-§5.10, §7.4).

## Features

| Feature | Adds |
|---|---|
| *(default)* | Pacers, lockstep math, resampler, crossfade, Opus framer, H.264/MP4 parsing, scaler, the NVENC encoder (ffmpeg `h264_nvenc` pipe), MP4 writer/finalize, probe, RTMP/HLS/file sinks |
| `openh264` | In-process OpenH264 encoder/decoder: **CPU test/CI backend only**, never in a deployed image |
| `opus` | In-process libopus encoder and decoder |

ffmpeg and ffprobe are found on `PATH`, or through `FV_FFMPEG` / `FV_FFPROBE`.
Tests that need them skip when they are missing; `tests/nvenc.rs` skips
unless `h264_nvenc` opens.

## Encoder policy (design §0)

- Production H.264 is **NVENC**, via an ffmpeg subprocess (`h264_nvenc`): no
  `unsafe` FFI in this crate, and the same process gives the scaler. Forced
  IDRs restart the process (its first frame is an IDR with SPS/PPS).
- The container needs the driver's `video` capability:
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` (**WP-16: set this in the
  `serve` image**; the runtime stage currently has `compute,utility`).
- Publish profiles (`H264Config::for_publish`): Cloudflare WHIP → capped at
  1280×720 (fit + pad), level 3.1 (`42e01f`); MediaMTX and peer WebRTC →
  native canvas, level 4.0 (`42e028`).
- No x264. libx264 appears only as `FfmpegH264::Libx264CpuTest`, the CPU CI
  stand-in that exercises the same ffmpeg plumbing; it cannot be selected
  from config.

## GPU record (2026-09-27, RTX PRO 6000 Blackwell, driver 595.91, Ubuntu 22.04 ffmpeg 4.4.2)

`cargo test -p fastvideo-media --test nvenc` (debug build), 120 frames of
1344×768 input at 24 fps with one forced IDR at frame 70:

| Profile | Encoded | SPS | IDRs | Wall per frame |
|---|---|---|---|---|
| Cloudflare | 1280×720 | CB (`42c01f`), level 3.1 | 0, 48, 70, 118 | 16.8 ms |
| MediaMTX | 1344×768 | CB (`42c028`), level 4.0 | 0, 48, 70, 118 | 14.8 ms |

Wall time includes the pipe, a debug-build test harness and one ffmpeg
restart; it is not the encoder's per-frame latency. The fal MP4 (NVENC High,
AAC-LC stereo 32 kHz, 24 fps, faststart) passed `fal_h3_problems()`.
BtbN's current ffmpeg master needs NVENC API 13.1 (driver ≥ 610) and fails on
this driver; pin an ffmpeg whose nv-codec-headers match the fleet driver.

## External formats covered by tests

| Client | Expected | Test |
|---|---|---|
| fal (hosted H3) | faststart MP4, H.264, 24 fps, AAC-LC stereo 32 kHz | `tests/ffmpeg_io.rs::fal_h3_mp4_format`, `tests/nvenc.rs::nvenc_fal_mp4` |
| Reactor | Opus, 48 kHz, mono, 10 ms frames | `opus::tests::reactor_format_mono_10ms_round_trip` (`opus` feature) |
| WHIP / browsers | H.264 Constrained Baseline, IDR every 2 s, SPS/PPS on every IDR, level per publish profile | `tests/nvenc.rs`, `tests/openh264_encode.rs`, `tests/ffmpeg_io.rs::pipe_encoder_*` |
| RTMP / HLS | always an AAC track (silence for video-only) | `tests/ffmpeg_io.rs::file_sink_video_only_still_has_audio` |

## Licensing

`src/clock.rs` and the `FramePacer` part of `src/pacer.rs` are ported from
strobe-core (MIT); see [NOTICE](NOTICE).
