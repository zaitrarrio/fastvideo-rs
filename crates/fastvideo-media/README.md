# fastvideo-media

fv-serve media: A/V pacer, resampler, Opus and H.264 encoders, MP4 finalize, probing, sinks. See [docs/serve/design.md](../../docs/serve/design.md) (WP-03, §5.1-§5.10, §7.4).

## Features

| Feature | Adds |
|---|---|
| *(default)* | Pacers, lockstep math, resampler, crossfade, Opus framer, H.264/MP4 parsing, ffmpeg-backed x264 encoder, MP4 writer/finalize, probe, RTMP/HLS/file sinks |
| `openh264` | In-process OpenH264 encoder (built from source) and decoder |
| `opus` | In-process libopus encoder and decoder |

ffmpeg and ffprobe are found on `PATH`, or through `FV_FFMPEG` / `FV_FFPROBE`.
Tests that need them skip when they are missing.

## External formats covered by tests

| Client | Expected | Test |
|---|---|---|
| fal (hosted H3) | faststart MP4, H.264, 24 fps, AAC-LC stereo 32 kHz | `tests/ffmpeg_io.rs::fal_h3_mp4_format` (box reader + ffprobe) |
| Reactor | Opus, 48 kHz, mono, 10 ms frames | `opus::tests::reactor_format_mono_10ms_round_trip` (`opus` feature) |
| WHIP / browsers | H.264 Constrained Baseline, IDR every 2 s, level fits the canvas (3.2 for 1344×768) | `tests/openh264_encode.rs`, `tests/ffmpeg_io.rs::x264_ffmpeg_encoder_gop_profile_level` |
| RTMP / HLS | always an AAC track (silence for video-only) | `tests/ffmpeg_io.rs::file_sink_video_only_still_has_audio` |

## Licensing

`src/clock.rs` and the `FramePacer` part of `src/pacer.rs` are ported from
strobe-core (MIT); see [NOTICE](NOTICE).
