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
//! | [`video`] | `VideoEncoder` trait: NVENC (production, via ffmpeg `h264_nvenc`), OpenH264 (CPU test backend, `openh264` feature); publish profiles (native canvas; level floor 3.1 for Cloudflare, 4.0 for MediaMTX/peer) |
//! | [`pipe`] | ffmpeg pipe-encoder processes: EOF flush on a keyframe restart, one warm pre-primed spare per streaming session ([`pipe::SparePool`]), restart latency metrics |
//! | [`vp8`] | Inter-frame VP8 (and VP9 profile 0) via ffmpeg `libvpx` / `libvpx-vp9` (IVF over a pipe; forced keyframes restart the process) for peers without H.264 and WHIP publishers that ask for them |
//! | [`scale`] | Pre-encode canvas scaler (fit+pad or stretch) |
//! | [`decode`] | Client video in: warm ffmpeg decode pipe (VP8 in IVF, H.264 Annex-B → RGB24 at the model's input size), keyframe probes |
//! | [`ring`] | Timestamped drop-oldest ring buffers for decoded client input ([`ring::InputBuffers`]) |
//! | [`h264`] | Annex-B / SPS helpers and the H.264 level table |
//! | [`mp4`] | ffmpeg MP4 writer and `finalize` (faststart, `-an`, crop), plus a pure-Rust box inspector |
//! | [`probe`] | `MediaProbe` via the image crate, ffprobe, or the MP4 inspector; decoding helpers |
//! | [`sink`] | RTMP(S)/SRT/HLS/file ffmpeg sinks with two input pipes, always with an audio track; credentials scrubbed from errors |
//! | [`output`] | The output-copy seam of live sessions: a non-blocking tick tap and the `OutputOpener` fv-serve implements |
//! | [`tools`] | Where the ffmpeg/ffprobe binaries are (`FV_FFMPEG`, `FV_FFPROBE`) |
//!
//! Only `clock` and `pacer` carry code ported from strobe-core (MIT, see
//! `NOTICE` in this crate). Everything else is original.
//!
//! Owned by WP-03 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod av;
pub mod clock;
pub mod crossfade;
pub mod decode;
pub mod error;
pub mod h264;
pub mod lockstep;
pub mod mp4;
pub mod opus;
pub mod output;
pub mod pacer;
pub mod pipe;
pub mod probe;
pub mod queue;
pub mod resample;
pub mod ring;
pub mod scale;
pub mod sink;
pub mod tools;
pub mod video;
pub mod vp8;

pub use av::{Pcm, RgbFrame};
pub use error::{MediaError, Result};
