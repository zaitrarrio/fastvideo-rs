# Open items and owner decisions

A running list of decisions the owner has made and of known bugs, gaps and
improvements parked for later. Re-evaluate the "Parked" list before starting
new work in the same area. Newest decisions first.

## Decisions

### 2026-10-07

- **CloudRift: on hold.** No CloudRift work until the owner lifts the hold.
  Last state: stock read 0 for RTX PRO 6000 / 5090, and the public API has no
  API-key → bearer exchange (both specs checked); a console check or a bearer
  token from the owner was the next step.
- **US weights volume: wait.** Still EU only (`jg48s6o1w0`, EUR-IS-1); the US
  rebuild plan in `docs/ops/runpod-volumes.md` §5 stays unused. Revisit if EU
  serverless or pod capacity keeps failing to place workers.
- **`h3` cluster on `stable`.** Its spec follows the `stable` channel, and a
  start re-resolves the channel's images, so a stopped `h3` starts on the
  current stable (release 54, 899a45d at the time of writing). A roll is only
  needed while it runs.
- **Edge-path trace: approved.** Run `scripts/serve/trace-bench.mjs` through
  the `h3` cluster's edge (docs/serve/tracing.md). Needs: `stable` promoted to
  a build with the tracing (#59, 3246c1f or later) after the usual GPU check,
  and a balance above fv-control's $20 cluster-start minimum.
- **Ref2V hosting: one process, swap only when both models cannot fit.**
  Serve the Ref2VA models from one process. Keep both tiers resident when
  they fit in the GPU's memory together (e.g. B200 / H200); fall back to swap
  mode (one resident, the other swapped in on demand, ~75 s per switch on an
  RTX PRO 6000) only when they cannot. No separate pods or pools per tier.
  Today's `runpod-h3-ref2v.toml` (turbo resident, max swapped) is the swap
  case for 80-96 GB cards.

## Parked: re-evaluate later

### Latency (from the warm-request trace, docs/serve/bench/e2e-warm-trace.md)

- Upload the MP4 to R2 while it encodes instead of after: 0.7-1.2 s per request.
- Push completion to the client instead of 700 ms polling: 0.3-0.6 s.
- Take the two D1 round trips (job insert, terminal write; ~280 ms each) off
  the critical path.
- Background warm-up right after READY held a request 4.7 s in the queue:
  make the first real request preempt it sooner.
- Runpod proxy adds ~150-200 ms per request: measure the edge path to see what
  remains there.

### Models and quality

- LTX-2.3 full-optimisation output shows speckle; cause unknown.
- Benchmarks not run: LingBot full-opt, A14B full-opt rerun (the first ran on
  an image before the scale fix). About $2 together.
- Phase 5: the missing benchmark arms.
- H3 Plug (c) switch (PR #33): plug-fast costs the same as h3-max and is
  closer to base 5/5; the switch rule is met, the switch is the owner's call.
- h3-max W8A8 on H100 untested.

### Serving and API

- `/v1/videos/sync` answers a 302 where callers expect the bytes.
- OpenAI-style retrieve returns `url: null`.
- fal output filenames name the wrong tier.
- The I2V smoke test uses the wrong prompt.
- The director browser suite needs network access the sandbox lacks.
- MoE kernels (LingBot) are still NVRTC-compiled at worker start; move them to
  the build-time cubins like every other kernel.

### Serverless and fv-control

- The serverless console and the load-balancer inline-config boot (#57, #58)
  are tested only against the simulated Runpod; the first real endpoint is the
  first live test.
- Queue-mode serverless endpoints do not carry trace headers; the live pages
  (stream, director, live input, avatar) are off for serverless.
- Mint a `ci` token in fv-control and set the GitHub secret
  `FV_CONTROL_CI_TOKEN` so tools releases build on build pods (18 min vs 23 min
  hosted). Owner action.
- Serverless GPU memory minimums per preset are conservative static values, not
  measured.

### CI

- Two tests failed once each on GitHub-hosted runners and did not recur:
  `the_default_limit_ends_a_live_session_at_120_s` (fastvideo-serve causal
  session; paused time with a real-time fake executor) and
  `h264_pictures_are_at_most_one_frame_behind` (fastvideo-media). Make them
  load-tolerant if they fail again.
- Hosted CI job logs cannot be read from agent sessions (GitHub serves them from
  another host); reproduce on the build pod instead.
