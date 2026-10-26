# H3 denoise launch overhead: lazy step events and CUDA graphs of the steps

Two levers on the time an H3 denoise spends with the GPU idle. Both sit at the pipeline, step and graph level, and
neither touches a per-op kernel. Code: `crates/fastvideo-cudarc/src/h3/step_graph.rs`. Measured on one RTX PRO 6000
(2026-10-14, image `fastvideo-rs-runtime:sha-edd18a9`, pod `hoirrt9s4b3h3t`, EU-NL-1). Results:
`artifacts/perf/h3-denoise-graphs/run-2026-10-14/` (`summary.txt` from `scripts/gpu/experiments/h3-graphs-summary.py`, now `fv-lab report summary --kind h3-graphs`).

| lever | flag | output | default |
|---|---|---|---|
| (a) lazy CUDA events per step instead of a synchronize per step | on; `FASTVIDEO_H3_STEP_SYNC=1` restores the sync | byte-identical, 72 / 72 clips | **yes** (is the default) |
| (b) CUDA graph per (request shape, step), replayed | opt-in `FASTVIDEO_H3_GRAPHS=1` | byte-identical, eager / captured / replayed, 72 / 72 clips | **no** |

## 1. Where the idle time was

A CUPTI trace of one warm V2-NVFP4 step (sparse FP4 tier, `FASTVIDEO_GPU_TRACE=1`, step 1) shows the following.
- **480p:** 1.393 s busy and 65-68 ms idle per step, with 2790 kernels and 2684 host launches.
- **768p:** 3.705 s busy and 57-65 ms idle, with 2789 kernels.

Almost all of the idle time was one gap at the start of each step. The loop synchronized after every step only to time
it. The next step's first block then waited about 51 ms while the host built and uploaded that step's bf16 AdaLN table.
The rest was:
- about 3,500 gaps under 5 µs (about 5.5 ms per step);
- before this branch, 50 small pageable index uploads per step in VSA's prefix pass, one per block. Each stalled the
  device for about 9 µs. All the arms below already include the fix, so the traces show 1 HtoD copy per step.

## 2. (a) Lazy step events

`StepClock` records a CUDA event after each step and synchronizes once, after the last step. A step's log line prints
once its event has completed; the clock polls for this at the next step boundary and never waits on it. Per-step
seconds are then GPU time between events. The host gets ahead of the device by the step's tail, which hides the next
step's table build. Progress hooks fire when a step is queued, at most a block or two ahead of the device. A GPU trace
still synchronizes per step, because its one-step window needs that.

## 3. (b) CUDA graphs of the steps (`FASTVIDEO_H3_GRAPHS=1`)

**What was captured.** One graph per step: the forward plus the scheduler step. It reads the text and latent rows
from persistent slots and writes the next latents back into them.
- The cache key is everything a step bakes in beside its inputs' values:
  - the packed layout: canvas, frames, and the text length and tags;
  - the VSA settings;
  - the dense FP8 switch.
- A key's first request runs the ordinary denoise, which warms up every lazy setup. The second request captures each
  step on the `wan::graph` stream and launches it. Later requests replay.

**What blocked capture, and what changed for it:**

| blocker | fix |
|---|---|
| bf16 AdaLN table built and uploaded per step (pageable copy) | the cache holds every step's table; the forward reads it through a thread-local pin (`H3Transformer::with_pinned_adaln`) |
| VSA prefix-row gather indices uploaded per block (pageable copy, 50 per step) | resident in the VSA device plan, same gather kernel. This also applies without graphs. |
| allocations inside the step | stream-ordered, so they become graph memory nodes: 3187 alloc / 3187 free, none escape |
| cuBLASLt workspace | the process-wide 32 MiB workspace at a fixed address; the graph stream's cuBLAS handle has an explicit workspace |
| host decisions: VSA `k_vid` per (step, layer), the Sol / VDN route, TeaCache | `k_vid` is fixed per step, so it is baked in. Graphs are off for Sol / VDN, TeaCache, condition rows (FL2VA / Ref2VA), fresh-noise schedules, a streamed DiT and dumps. |
| host round-trips | none inside a step. NVFP4 alpha is read on the device; the VSA selection is a device top-k. |

A capture error, or a step whose allocations escape its graph, gives graphs up for the pipeline. That step runs
eagerly on the same slots, so the output is unchanged; the 768p OOM below took this path.

**Per-step graph:** 9866 nodes, of which 2792 are kernels. With 768p the graph memory pool peaks at 11.1 GiB, and the
step graphs share it (4.7 GiB at 480p).

## 4. Results

Medians of three requests per prompt, seed 1024, 124 frames (5 s). The NVFP4 cells use the sparse FP4 tier
(`FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_H3_VSA_FP4=1`). One process per (checkpoint, arm), with `--warm`.
- **sync:** a synchronize per step, the behaviour before this change.
- **lazy:** (a).
- **replay:** (b) on top of (a), counting only the requests that replayed graphs.

Per block is the median step divided by 50 blocks. It includes the step's non-block work and idle time.

**Denoise seconds:**

| checkpoint | canvas | prompt | sync | lazy (a) | Δ | replay (b) | Δ vs lazy | upstream |
|---|---|---|---:|---:|---:|---:|---:|---:|
| V2-NVFP4 | 480p | h3-demo | 11.665 | **11.143** | -4.5% | 11.039 | -0.9% | 10.25 |
| V2-NVFP4 | 480p | spark-mountain-lake | 11.025 | **10.604** | -3.8% | 10.420 | -1.7% | 9.81 |
| V2-NVFP4 | 768p | h3-demo | 29.961 | **29.498** | -1.5% | 29.409 | -0.3% | 29.28 |
| V2-NVFP4 | 768p | spark-mountain-lake | 29.213 | **28.825** | -1.3% | (fell back) | | 28.41 |
| 4-step NVFP4 | 480p | h3-demo | 5.579 | **5.390** | -3.4% | 5.315 | -1.4% | |
| 4-step NVFP4 | 480p | spark-mountain-lake | 5.272 | **5.117** | -2.9% | 5.009 | -2.1% | |
| 4-step NVFP4 | 768p | h3-demo | 13.869 | **13.681** | -1.4% | 13.588 | -0.7% | |
| 4-step NVFP4 | 768p | spark-mountain-lake | 13.436 | **13.230** | -1.5% | 13.149 | -0.6% | |

**Generate seconds (`total_s`), per block and memory:**

| checkpoint | canvas | prompt | generate s: sync / lazy / replay | per block ms: sync / lazy / replay | peak MiB: lazy / graphs |
|---|---|---|---|---|---|
| V2 | 480p | h3-demo | 14.20 / 13.66 / 13.61 | 29.1 / 27.7 / 27.6 | 25 734 / 27 692 |
| V2 | 480p | spark | 13.57 / 13.12 / 12.92 | 27.6 / 26.4 / 26.0 | 25 638 / 30 956 |
| V2 | 768p | h3-demo | 35.45 / 34.89 / 34.73 | 74.9 / 73.6 / 73.5 | 33 446 / 37 800 |
| V2 | 768p | spark | 34.58 / 34.21 / – | 73.0 / 71.9 / – | 33 318 / 45 390 |
| 4-step | 480p | h3-demo | 8.14 / 8.00 / 7.94 | 27.8 / 26.7 / 26.6 | 25 766 / 27 454 |
| 4-step | 768p | h3-demo | 19.30 / 19.16 / 19.08 | 69.3 / 68.1 / 67.9 | 32 326 / 38 796 |

**Launch count and idle time per step.** From the trace: the V2 step-1 window, `trace/v2-{480p,768p}`. The trace
synchronizes per step, so its eager row is the sync behaviour.

| | kernels | host launches (NVRTC `launch!`; cuBLAS extra) | HtoD copies | idle per step |
|---|---:|---:|---:|---:|
| eager, 480p | 2790 | 2684 | 1 (AdaLN table, 38.7 MB) | 68.0 ms |
| replay, 480p | 2792 | **0** (1 graph launch) | 0 | **2.3 ms** |
| eager, 768p | 2789 | 2684 | 1 | 65.1 ms |
| replay, 768p | 2791 | **0** | 0 | **2.4 ms** |

GPU busy time per step is unchanged: 1.393 s at 480p and 3.71 s at 768p, which is 27.9 / 74.1 ms per block. Neither
lever makes a kernel faster. They remove idle time.

**Byte identity.** Each clip's 124 PNG frames and WAV were hashed: sha256 over names and bytes, in `hashes.tsv`.
- **Groups:** 8 groups (checkpoint × prompt × canvas). Every group has 9 clips: 3 arms × 3 requests, which includes the
  eager, captured and replayed runs and the one that fell back. Each group has **one distinct hash**.
- The `--warm` cold clips match too.

**The graph cost:**
- **First capture:** the first request that captures in a process pays about 4.7 s once (V2 480p h3-demo: 15.8 s
  against 11.1 s). That covers building every step's AdaLN table, creating the graph stream, and capturing and
  instantiating 8 graphs of about 9.9k nodes. Later captures cost 0.1-0.3 s.
- **Memory:** graphs hold 2-12 GB of device memory beside the eager pool.
- **The OOM:** with four cached shapes (2 × 480p + 2 × 768p, 32 graphs), the fourth shape's capture failed at step 2
  with `CUDA_ERROR_OUT_OF_MEMORY`. The card had 45 GB of 96 GB in use. That request fell back to eager, byte-identical.
  The cache now keeps 2 shapes (`MAX_KEYS`, set after this run and not re-measured).

## 5. Verdicts

- **(a) Keep it as the default.** It saves 1.3-4.5% of the denoise, with every output bit unchanged.
  - The largest gain is at 480p and on few-step recipes, where one ~51 ms gap per step is a larger share.
  - V2-NVFP4 at 768p is now 29.50 s against upstream's 29.28 s (+0.75%); at 480p it is 11.14 s against 10.25 s.
- **(b) Keep it opt-in; do not make it the default.**
  - A replay gains another 0.3-2.1%: replay cuts a step's idle time from about 65 ms to about 2.4 ms. After (a), only
    the sub-5 µs launch gaps and the step-boundary handoff were left.
  - A graph replays only for a request with the same key. That means the same text token count as an earlier request,
    which a stream of different prompts rarely has. Every new shape pays an eager run and then a capture.
  - It costs 2-12 GB, and it ran out of graph memory with four shapes.
  - It fits a fixed-prompt loop (benchmarks, re-renders of one prompt at other seeds), not serving.
- **h3-turbo (MXFP8, 768p) was not measured.** No RTX PRO 6000 was in stock in EU-RO-1, where the EU volume holds its
  MXFP8 tree. Its 4-step VSA shape matches the 4-step NVFP4 cells: expect (a) to save about 1.4% at 768p, (b) 0.6-0.7%
  more on a replay, and neither to change a bit. Measure it with the fv-lab tier `h3-graphs` (D1e,
  docs/adr/0019-fv-lab.md §14: this experiment's `FV_CKS=turbo` cells, its commands cut from the pod script; every
  tree is on the EU volume): `fv-lab pod up --sha <7> --family h3-graphs --yes --wait`.

## 6. Reproduce

The h3-turbo cells (the open item of §5) are the fv-lab tier `h3-graphs` (docs/adr/0019-fv-lab.md §14, the GPU check
there): the trace, then sync / lazy / graphs at 768p over the same prompt set, with `--off-identity` compares of
lazy against sync and graphs. The experiment's own driver, below, stays until that check passes; its NVFP4 cells
(R2-only trees) have no tier until the registry's hydrate.

```bash
fv-lab pod up --sha <7> --family h3-graphs --yes --wait    # h3-turbo only (EU volume), from a host with RUNPOD_API_KEY
scripts/gpu/experiments/h3-graphs-validate.sh up ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>   # backstop marker fv-h3graphs-backstop-<pod>
scripts/gpu/experiments/h3-graphs-validate.sh status <pod>
scripts/gpu/experiments/h3-graphs-validate.sh fetch <pod> DIR && scripts/gpu/experiments/h3-graphs-validate.sh down <pod>
fv-lab report summary --kind h3-graphs DIR    # the experiment's run directory (cells/, trace/, hashes.tsv), not an fv-lab one; was h3-graphs-summary.py (D1c)
```

Spend: one pod for 46 min at $2.49/hr, about $1.91.
