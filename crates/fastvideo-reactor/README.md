# fastvideo-reactor

The Reactor local runtime in Rust (WP-13, [design §5.7](../../docs/serve/design.md)):
unmodified Reactor SDK clients (`reactor_sdk` 1.6.0 and the Rust-core SDKs,
v1 protobuf; older v0 JSON clients) connect to fv-serve as they would to the
Python Reactor Runtime.

- `proto/`: `reactor_wire.v1`, copied verbatim from reactor-team/reactor-runtime
  (Apache-2.0; `proto/LICENSE`, `proto/NOTICE`). Bindings are committed in
  `src/pb/`; `--features proto-codegen` regenerates and checks them.
- Routes: `/start_session`, `/session`, `/stop_session`, `/schema`, `/events`,
  and `/sessions/{sid}/transport/webrtc/{ice_servers,connections,…}`.
- Modes from the model caps: fast-h3 clip commands (H3, LTX, FastWan) over
  the engine's `ClipPlayer`, or SF-Wan causal setters over `CausalControl`.
- Tracks from the model: `[main_video, main_audio]` or `[main_video]`.
- Media: H.264 (NVENC / OpenH264) or intra-only VP8 (libwebp) per peer, and
  10 ms 48 kHz mono Opus.

Tests: `cargo test -p fastvideo-reactor` (codecs, schema, and loopback-peer
end-to-end runs on the fake engine). SDK compatibility:
`bash crates/fastvideo-reactor/tests/compat/run.sh` (installs
`reactor_sdk==1.6.0` into a venv and runs the A/V, video-only and causal
fake models). A standalone fake runtime:
`cargo run -p fastvideo-reactor --example fake_runtime -- --port 8080 --model av`.
