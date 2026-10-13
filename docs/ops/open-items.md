# Open items and owner decisions

A running list of decisions the owner has made and of known bugs, gaps and
improvements parked for later. Re-evaluate the "Parked" list before starting
new work in the same area. Newest decisions first.

## Decisions

### 2026-10-13

- **EU weights volume: reduce to a 500 GB H3 + LTX volume** ("eu volume -
  reduce to 500 gb plan"). Replace `jg48s6o1w0` (2000 GB, $120/month) with a
  500 GB volume (~$35/month) holding only `FV_EU_VOLUME_TREES`
  (`scripts/gpu/volumes.sh`: h3-base and its three nested trees, the FastH3
  LoRA, h3-ref2va, ltx25, ltx25-dev, the IC-LoRA, upscaler, h3-to-ltx,
  auxiliary/tae and lpips: 462.45 GB), filled from R2 with
  `scripts/gpu/r2-fill-volume.sh`; everything else from R2.
  **Blocked (2026-10-13):** Runpod no longer creates network volumes in
  EUR-IS-1 (`POST /networkvolumes`: "Data center EUR-IS-1 not found or does
  not support network volumes"; GraphQL `storageSupport: false`). The only EU
  DC with volumes and RTX PRO 6000 Server stock is EU-RO-1 (stock "Low"),
  which would move the EU GPU region.
  **Owner's choice (2026-10-13): the volume in EU-RO-1, and the `eu` region
  (GPU pods and serverless) moves there.** Done: `bz0ofkfx45`
  (`fv-weights-h3-ltx-500`, 500 GB, EU-RO-1), filled from R2 and compared
  (docs/ops/runpod-volumes.md §0a). Presets whose trees are not on it (wan,
  fastwan21, sfwan, longlive) default to `weights_source r2`. `jg48s6o1w0`
  stays until the coordinator deletes it, after the deploy and one EU-RO-1
  pod reading the new volume.
- **Brev: a stop that does not succeed deletes the VM.** "We should go ahead
  and delete VMs that need to be stopped if a stop command does not succeed.
  If the stop command succeeds then we should NOT delete the VM." On every
  stop path, a refused stop ("does not support stop", the non-stoppable
  types), another stop error after 3 tries, or a VM not STOPPED 10 min after
  an accepted stop → deleted (fv- names, recorded rows), recorded with the
  reason, info alert `brev_stop_deleted`. A successfully stopped VM is still
  parked / held and never auto-deleted (the 2026-10-11 rule); park limits
  still only alert; manual deletes unchanged. A stop while DEPLOYING waits
  (15 min) for it to leave DEPLOYING, then stops it or deletes it. Replaces
  the same morning's `running_unstoppable` handling (PR #80: left running,
  hourly retry, `brev_unstoppable`, the 24 h budget margin).
  docs/serve/deploy-gmi-brev.md §3.4.
- **Brev: non-stoppable types are launched like any other** (earlier the
  same day, replacing rule `brev_stoppable_only`); with the rule above, their
  stop deletes them.
- **Brev warm restart without capacity: a fresh VM, the parked one kept.**
  When Brev's start task (`environment.tasks[]`) says InsufficientCapacity,
  the start is retried 3 times over ~5 min, then the launch creates a fresh
  VM and the parked VM stays parked (its stop succeeded: not deleted).

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
