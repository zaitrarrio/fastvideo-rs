//! Media plumbing for fv-serve (design §5.1, §5.3, §5.5, §5.9, §7.4, WP-03).
//!
//! | Module | What |
//! |---|---|
//! | [`av`] | Raw A/V buffers: [`RgbFrame`] (RGB24) and [`Pcm`] (interleaved f32) |
//! | [`clock`] | Injectable monotonic clock (strobe-core port) and the RTP clocks |
//! | [`pacer`] | `FramePacer` (strobe-core port: drop-oldest, freeze, adaptive fps), `AvPacer` (lockstep audio lane, first-frame notify), `Metronome` |
//! | [`lockstep`] | `48000/fps` sample math, clip audio fitting, 3-frame slicing |
//! | [`resample`] | rubato FFT resampler (32k/24k to 48k for Opus, 48k to 32k/44.1k for MP4), channel mixing |
//! | [`crossfade`] | Raised-cosine clip-edge fades that keep sample counts |
//! | [`opus`] | Opus framer (10/20 ms, sample-counter RTP timestamps) and libopus encoder (`opus` feature) |
//! | [`video`] | `VideoEncoder` trait: NVENC (production, via ffmpeg `h264_nvenc`), OpenH264 (CPU test backend, `openh264` feature); publish profiles (Cloudflare 720p/L3.1, MediaMTX/peer native/L4.0) |
//! | [`vp8`] | Inter-frame VP8 via ffmpeg `libvpx` (IVF over a pipe; forced keyframes restart the process) for peers without H.264 |
//! | [`scale`] | Pre-encode canvas scaler (fit+pad or stretch) |
//! | [`h264`] | Annex-B / SPS helpers and the H.264 level table |
//! | [`mp4`] | ffmpeg MP4 writer and `finalize` (faststart, `-an`, crop), plus a pure-Rust box inspector |
//! | [`probe`] | `MediaProbe` via the image crate, ffprobe, or the MP4 inspector; decoding helpers |
//! | [`sink`] | RTMP/HLS/file ffmpeg sinks with two input pipes, always with an audio track |
//! | [`tools`] | Where the ffmpeg/ffprobe binaries are (`FV_FFMPEG`, `FV_FFPROBE`) |
//!
//! Only `clock` and `pacer` carry code ported from strobe-core (MIT, see
//! `NOTICE` in this crate). Everything else is original.
//!
//! Owned by WP-03 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod av;
pub mod clock;
pub mod crossfade;
pub mod error;
pub mod h264;
pub mod lockstep;
pub mod mp4;
pub mod opus;
pub mod pacer;
pub mod probe;
pub mod queue;
pub mod resample;
pub mod scale;
pub mod sink;
pub mod tools;
pub mod video;
pub mod vp8;

pub use av::{Pcm, RgbFrame};
pub use error::{MediaError, Result};
