# Test stages and load-tolerant tests

`scripts/serve/check.sh` gates the fv-serve crates. It runs on the shared
build pod ([build-pod.md](build-pod.md)), where several agents compile and
test at once on 32 vCPUs. A test that passes alone but fails when the pod is
busy blocks everyone's merge, so the tests are split by what they measure.

## Stages

| Stage | What | How it runs |
|---|---|---|
| 1. build | `cargo check` of the serve crates, `cargo clippy --workspace --all-targets --no-deps -- -D warnings` (and, with `FV_SERVE_HEAVY=1`, the heavy feature sets) | parallel, as cargo does |
| 2. tests | `cargo test` of every crate and feature set | parallel test threads, next to other agents' jobs: **load-tolerant** |
| 3. realtime | the `realtime_*` tests and the `/console` browser tests | only with `--realtime` (or `FV_SERVE_REALTIME=1`); after stages 1-2; each test binary is built first (`--no-run`), then run with `--test-threads=1`, one binary after the other: nothing of `check.sh` compiles or tests meanwhile |

```bash
bash scripts/serve/check.sh                      # stages 1-2
bash scripts/serve/check.sh --realtime           # stages 1-3
bash scripts/serve/check.sh --realtime-only      # stage 3 alone
FV_SERVE_UI=1 bash scripts/serve/check.sh --realtime   # stage 3 with the browser tests
```

On the build pod: `scripts/dev/build-pod.sh run <agent> -- bash
scripts/serve/check.sh --realtime`. The realtime stage is only as quiet as
the pod: run it when `build-pod.sh status` shows no other heavy job, or
accept that a rate failure there may be the neighbours'.

`FV_SERVE_UI=1` without `--realtime` still runs the browser tests, last,
after every compile.

## Formatting and lint gates

Since plan F2 the workspace is rustfmt-clean and clippy-clean, and both are
gates rather than advice: `task check` runs `cargo fmt --all -- --check`
(`task fmt:check`; `task fmt` fixes drift) and `cargo clippy --workspace
--all-targets --no-deps -- -D warnings` (`task lint:rust`); the
`serve-compat` and `gpucheck-t0` workflows run the `fmt --check` in the
build base image (the build pod's rustfmt, so the pod's `--check` is the
source of truth) on every pull request that touches the crates; and
`scripts/serve/check.sh` stage 1 runs the same clippy line over the whole
workspace, so the tools-release pipeline gates every crate on it. Fix a
lint at its source; where it flags an intentional pattern, the narrowest
`#[allow(clippy::...)]` with a one-line reason is the rule (see
`crates/fastvideo-ops/src/fp8.rs` `scale_for_amax`). The workspace
`rust-version` (1.88) is what clippy's `incompatible_msrv` checks the std
APIs against, so every crate inherits it.

## Coverage

`scripts/serve/coverage.sh` (plan E4) runs `cargo test --locked` over the
adapter crates (`fastvideo-openai-videos`, `-minimax`, `-ltxapi`, `-fal`,
`-reactor`), `fastvideo-serve` and the CPU crates they build on
(`fastvideo-protocol`, `-trace`, `-engine-service`, `-serve-kit`,
`-upsampler`) under [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov),
one instrumented run, and writes `lcov.info`, `html/` and the line / region /
function totals (`summary.json`, `summary.md`) to `target/llvm-cov`
(`FV_COV_OUT` moves it). `task coverage` runs it; `task coverage:tests` (or
`coverage.sh --tests`) runs the same crate set with plain `cargo test
--locked`, no instrumentation.

In CI the `adapter-tests` workflow runs the script in the build base image
on every pull request that touches those crates (and on pushes to main),
uploads the reports as the `adapter-coverage` artifact and prints the
totals in the job summary. It is the only workflow that runs these crates'
Rust tests on pull requests: `serve-compat` drives the client libraries
against the `fv-serve` binary, and `tools-release` runs `check.sh` only on
pushes to main whose tools inputs changed.

Locally: `cargo install cargo-llvm-cov --locked` and `rustup component add
llvm-tools-preview`. On the build pod `coverage.sh` is allowlisted
(`build-pod.sh run <agent> -- bash scripts/serve/coverage.sh --tests` works
today); the instrumented mode waits for `cargo-llvm-cov` and
`llvm-tools-preview` in `docker/build-base.Dockerfile` (a re-pin of the
image, see [build-pod.md](build-pod.md) "Base image"), after which the
workflow's run-time installs can go too.

## What goes where

**Stage 2 (load-tolerant).** Everything that checks behaviour: protocol
messages, state machines, order, content, counts. Rules:

- Wait for a condition, never a fixed sleep: poll the state (or `pump`
  events) until it holds, with a generous limit that only turns a hang into
  a failure (60 s for anything behind an ffmpeg start; the limit is not a
  budget). A negative check ("nothing arrives while paused") may sleep.
- No wall-clock rates or latencies. Check frame counts, RTP grids (video
  times on the 24 fps grid of 3750 ticks, audio on the 20 ms Opus grid of
  960), ordering and payloads instead.
- Timeouts that are the behaviour under test (a negotiation timeout, a
  watchdog) are tested with the smallest configuration that cannot expire
  by accident elsewhere in the test: a long one where the test needs the
  peer alive, a short one on its own host where it checks the expiry, and
  then "not before the timeout" as well as "eventually".
- Unique temporary directories (`tempfile::Builder::new().prefix(..)
  .tempdir()`), never a path built from the time or the pid: tests in one
  binary run in parallel, and pids repeat across runs.
- Ports: bind port 0 and pass the bound listener on; never "find a free
  port, close it, bind it again".

**Stage 3 (realtime).** Tests whose assertion *is* a rate or a latency:
media received at 24 fps / 48 kHz, a clip delivered frame for frame (the
pipelines shed frames on purpose when the host falls behind real time), a
decode latency bound. They are `#[ignore = "real-time ...:
scripts/serve/check.sh --realtime"]` and named `realtime_*`; most are twins
of a stage-2 test running the same scenario with the rate assertions on
(`av_session(true)` vs `av_session(false)`), so stage 2 still covers the
protocol on a loaded host and stage 3 keeps every rate assertion unchanged.

| Target | Realtime tests |
|---|---|
| `fastvideo-fal --features director,openh264 --test director_e2e` | `realtime_av_session_end_to_end`, `realtime_video_only_session`, `realtime_browser_offer_without_a_usable_h264_encoder_gets_vp8`, `realtime_late_chunks_report_deadline_missed`, `realtime_causal_session_streams_and_recaches_once_per_switch` |
| `fastvideo-reactor --test runtime` | `realtime_v1_client_av_session`, `realtime_avatar_script_take_streams_in_windows`, `realtime_causal_session_ends_at_its_length_limit` |
| `fastvideo-media --test decode_pipe` | `realtime_h264_pictures_are_at_most_one_frame_behind` |
| `tests/console/run.sh` (`FV_SERVE_UI=1`) | the Chromium console tests |

Run one by hand: `cargo test -p fastvideo-reactor --test runtime --
--ignored --test-threads=1 realtime_`.

A new rate or latency assertion goes into a `realtime_` test and the table
above (and `realtime_stage` in `check.sh` when it is a new target).
