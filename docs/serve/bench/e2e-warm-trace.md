# Warm request, end to end, traced (H3 turbo, RTX PRO 6000)

Measured 2026-10-07 with the tracing of [../tracing.md](../tracing.md)
(PR #59, image `h3-turbo-sha-46fac0e`), on one standalone pod
(`fv-control.sh pod launch trace-bench --preset h3-turbo --sha 46fac0e`):
RTX PRO 6000 Blackwell Server Edition, EUR-IS-1, the EU weights volume
read-only, D1 job store, R2 outputs, direct mode (API key), $2.09/h.
Client: `scripts/serve/trace-bench.mjs` in a container in the US, through
the pod's Runpod proxy URL (`https://<pod>-8000.proxy.runpod.net`), the
fal queue API the console uses (`minimax/h3-turbo/text-to-video`), polls
every 700 ms like the console, the video read to its last byte.

**Warm** = model resident, background warm-up finished (`/healthz`
`warming: []`), text encoder resident, two warm-up requests sent first.
Prompts are the same across runs (the prompt cache hits) except in the
"fresh prompt" rows. Raw data: [e2e-warm-trace/](e2e-warm-trace/)
(`*.summary.json`, `*.runs.jsonl`, one full trace per resolution).

**Not measured: the edge path.** A standalone pod is direct only
(docs/control/standalone-pods.md) and the only edge cluster is the owner's;
the edge Worker's tracing (crates/fastvideo-edge/src/trace.rs) is built
and unit-checked (wasm clippy) but was not deployed to the staging edge
for this run, because putting a pod behind `fv-edge-staging` needs an
`edge` cluster from fv-control (reserved env, `control_plane = edge`).
What is missing to run it: start an edge cluster on staging (owner),
`cf-edge.sh deploy` from this branch, then
`trace-bench.mjs --base https://fv-edge-staging.<sub>.workers.dev`; the
waterfall then adds `edge.auth`, `edge.registry`, `edge.forward`,
`front.do_enqueue` (the family DO round trip) and `worker.offer` /
`worker.take`.

## Tracing overhead (A/B, interleaved)

10 traced and 10 untraced warm requests per resolution, alternating
(ABAB…). Median / p90 in ms; diff = median(traced) − median(untraced)
with its 95 % bootstrap interval (the noise estimate). Engine times are
the engine's own stage timers (`timings` in the fal result), reported for
every request, traced or not.

| 480P, 5 s | traced med / p90 | untraced med / p90 | diff [95 % CI] |
|---|---|---|---|
| click → video last byte (e2e) | 13 360 / 13 826 | 13 341 / 14 388 | +19 [−1 142, +624] |
| click → poll saw COMPLETED | 12 426 / 12 609 | 12 474 / 13 267 | −48 [−1 025, +548] |
| submit round trip | 553 / 712 | 469 / 550 | +84 [−12, +196] |
| denoise (GPU, 4 steps) | 7 117 / 7 123 | 7 110 / 7 121 | +6.5 [−3.5, +11.3] |
| video decode | 2 735 / 2 736 | 2 733 / 2 735 | +2.1 [−6.1, +7.1] |
| encode tail | 134 / 151 | 132 / 155 | +2.1 [−3.9, +9.5] |
| engine total (start → end) | 10 402 / 10 450 | 10 380 / 10 428 | +22 [−23, +59] |

| 768P, 5 s | traced med / p90 | untraced med / p90 | diff [95 % CI] |
|---|---|---|---|
| click → video last byte (e2e) | 29 632 / 30 364 | 29 304 / 30 716 | +328 [−649, +794] |
| click → poll saw COMPLETED | 28 589 / 28 943 | 28 414 / 29 214 | +175 [−626, +658] |
| submit round trip | 579 / 883 | 488 / 605 | +91 [−43, +270] |
| denoise (GPU, 4 steps) | 19 173 / 19 227 | 19 159 / 19 182 | +14.5 [−2.7, +41.6] |
| video decode | 6 410 / 6 417 | 6 412 / 6 417 | −1.7 [−5.6, +15.6] |
| encode tail | 131 / 142 | 133 / 141 | −1.4 [−10.1, +9.3] |
| engine total | 26 154 / 26 204 | 26 157 / 26 229 | −2.6 [−36, +42] |

Every interval contains 0: tracing is within the noise. The GPU stages
move by at most ~0.1 % (denoise +6.5 ms of 7.1 s, +14.5 ms of 19.2 s, both
inside their intervals); the client-side numbers are dominated by the
Runpod proxy and the 700 ms polling (the e2e interval is ± ~1 s). The
submit round trip's +84 / +91 ms is the least clear (its interval's lower
end is −12 / −43 ms): the traced submit does the same work on the pod plus
~20 record pushes; with 10 samples per arm over a public proxy this is not
distinguishable from proxy jitter, and a longer run would be needed to
bound it below ~50 ms. **Dropped events: 0** in every traced run
(`stats.dropped`, recorder capacity 65 536); no trace was truncated.

## Waterfall (traced runs, median / p90 ms)

Host and GPU spans of the same engine segment agree to within 0.1 ms: the
H3 pipeline waits for the device at every step boundary, so the CUDA
event times and the host times coincide (the device spans would show any
host run-ahead; there is none to show).

| step | 480P | 768P |
|---|---|---|
| client: submit (fetch → answer headers) | 553 / 712 | 579 / 883 |
| pod: `http.post` (receive → answer) | 278 / 290 | 278 / 298 |
| ↳ `adapter.parse` + `validate` + `ingest_negotiate` | < 0.1 | < 0.1 |
| ↳ **`store.insert` (D1)** | **278 / 290** | **277 / 298** |
| ↳ `queue.submit`, `store.get` | 0.1, < 0.1 | 0.1, < 0.1 |
| `queue.wait` (submit → executor dequeue) | 0.1 | 0.1 |
| `gpu.text` (prompt cache hit) | 298 / 326 | 309 / 352 |
| `gpu.text` (fresh prompt, 4 runs) | 499 / 522 | 506 / 528 |
| `gpu.denoise.step` ×4 (sum) | 7 117 / 7 123 (≈1.75–1.79 s each) | 19 173 / 19 227 (≈4.8 s each) |
| `gpu.denoise.tail` | 29 / 31 | 64 / 75 |
| `gpu.audio_decode` | 71 / 73 | 58 / 59 |
| `gpu.video_decode` (VAE + frames to the NVENC feed) | 2 742 / 2 762 | 6 418 / 6 426 |
| `post.encode_busy` (NVENC feed, overlapped with the decode) | 495 / 656 | 691 / 723 |
| `gpu.encode_tail` (MP4 finish after the last frame) | 135 / 152 | 131 / 142 |
| `post.finalize` | 0.5 | 0.6 |
| **`upload.artifact_put` (3.3 / 7 MB to R2, after the encode)** | **719 / 1 096** | **946 / 1 239** |
| **`store.terminal_write` (D1)** | **278 / 292** | **282 / 339** |
| `store.terminal` (upload + write) | 998 / 1 379 | 1 232 / 1 512 |
| `http.get` per status poll (pod side) | 0.2 | 0.2 |
| client: poll round trip (each) | ~150–270 | ~150–270 |
| client: result request | 301 / 503 | 367 / 573 |
| client: video TTFB (R2 presigned URL) | 406 / 594 | 432 / 532 |
| client: video bytes | 123 / 248 | 291 / 444 |
| **click → video last byte** | **13 360 / 13 826** | **29 632 / 30 364** |
| unaccounted (no known step) | 0.2 / 432 | 127 / 363 |

Critical path (traced runs, median ms; consecutive anchors on the client's
clock):

| phase | 480P | 768P |
|---|---|---|
| click → submit sent | 0.0 | 0.0 |
| submit sent → pod received (client → proxy → pod) | 176 | 196 |
| pod received → engine dequeued (D1 insert) | 278 | 278 |
| engine run (GPU first mark → last mark) | 10 401 | 26 153 |
| engine run end → job terminal (upload + D1 write) | 998 | 1 233 |
| job terminal → client saw COMPLETED (poll latency) | 375 | 288 |
| client saw COMPLETED → result answer | 302 | 368 |
| result answer → video first byte | 406 | 433 |
| video first byte → last byte | 123 | 291 |

Clock alignment (pod vs client): offset ≈ 48 ms, **± 72 ms** (half the
best submit/poll round trip minus the pod's own time, over the Runpod
proxy). Cross-host phases above (client ↔ pod) are only meaningful to
that bound; each span's own duration is exact on its host.

### Biggest non-GPU costs (warm, 480P: 13.4 s end to end, 10.4 s of it GPU)

1. **Output upload to R2 after the encode: 0.7–1.1 s** (`upload.artifact_put`,
   3.3 MB; 0.9–1.2 s for 7 MB at 768P). Direct mode uploads the finished
   file; the edge path's direct upload streams parts while NVENC writes
   (docs/serve/dispatch-do-family.md §7) and should hide most of it.
2. **D1 round trips on the critical path: ~280 ms twice** — the job insert
   before the submit answers (`store.insert`, which also delays the engine
   start) and the terminal write before the job reads as done
   (`store.terminal_write`). Both are a D1 HTTP round trip from EUR-IS-1.
3. **Polling: 0.3–0.6 s median, up to the 700 ms interval** between the job
   turning terminal and the client seeing it, then a result request
   (~0.3–0.5 s through the proxy). A streamed status (SSE) or returning the
   result in the COMPLETED status answer would save ~0.5–1 s.
4. **Video fetch from R2: ~0.4 s TTFB** (presigned URL from the US client to
   the EU bucket) + 0.1–0.3 s of bytes.
5. **Client ↔ pod via the Runpod proxy: ~150–200 ms per round trip**
   (submit, every poll, result).
6. Text encode with a fresh prompt: +200 ms over a cached one (499 vs 298
   ms at 480P).

Also found while setting up: right after the pod reports ready, the
**background warm-up** keeps the executor busy between jobs and a request
waits up to ~4.7 s in `queue.wait` until the warm-up yields at its next
step (first smoke request: `queue.wait` 4 754 ms). Warm requests in this
bench were taken after `/healthz` showed `warming: []`.

## Browser (console click)

Headless Chromium on the pod's console page (`scripts/serve/trace-browser.mjs`,
Trace switch on, Run clicked): the trace starts at the click, the submit
and polls carry `traceparent` / `x-fv-trace`, the beacon and the
waterfall come after the video. Two caveats in this sandbox, so these are
functional checks, not the reference numbers (the bench client above is):
its egress refused Chromium's requests to paths with `minimax` (curl and
Node were fine), so requests to the pod were relayed through Node
(`FV_BROWSER_RELAY=1`), and Playwright's Chromium has no H.264 decoder, so
the player reports `video_error` instead of `canplay` (the console ships
the trace on either). Runs: [browser-480P](e2e-warm-trace/browser-480P.runs.jsonl),
[browser-768P](e2e-warm-trace/browser-768P.runs.jsonl); click → result
answer 12.2 / 12.2 s (480P) and 29.1 / 28.7 s (768P), the bench's ~12.7 s and
~29.0 s; dropped events 0.

## Spend

Pod `trace-bench` (standalone, deleted after the run): 16:37–17:27 UTC,
≈ 50 min at $2.09/h ≈ **$1.75**. Balance 16.49 → 14.56 $ over the
session (account baseline included).
