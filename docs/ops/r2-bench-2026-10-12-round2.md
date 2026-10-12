# R2 cold-start benchmark, round 2 (2026-10-12)

This is the second round with `weights_source: r2`. It ran after PR #76 fixed the seven findings of round 1
(docs/ops/r2-bench-2026-10-12.md). It ran on staging fv-control (main `1f9f61d`, migration 0011) with
`scripts/serve/fv-control.sh`. Every endpoint and pod ran the images that the `serve-image` run built for
`1f9f61d` (`image: {sha: 1f9f61d…}`; run 38218229546, success). Those images carry `python3`, `curl` and
`r2-fetch.py`. The stable channel was not promoted.

| Run | Planned | What happened |
|---|---|---|
| 1a | Serverless `bench-h3-r2` (preset `h3-turbo`) | **H3 now loads from r2 and produces a video.** The cold start took 398 s click → done (fetch 177 GB at 693 MB/s, load 52 s). Two FlashBoot revivals, at +1 and +11 min, each waited 1-2 s. |
| 1b | Serverless `bench-wan-r2` (preset `wan`) | Cold start 303 s click → video (fetch 59 GB at 412 MB/s, of which the tail took 47 s; load 85 s). Two FlashBoot revivals, at +1.5 and +11.5 min, each waited 0.9 s. |
| 2 | Runpod pod outside the EU, `--weights r2` | **Blocked by a new bug (finding 2).** US-NC-1 and US-MO-2 had no RTX PRO 6000 instance. In US-GA-2 an H100 pod booted and asked for its plan once. Its fetcher then printed nothing for 18 min, and fv-serve never started. The pod was deleted after 19 min ($1.26). |
| 3 | Brev `wan` on g7e.2xlarge, warm restart of the parked VM | **Not run.** `brev unpark` worked: the held VM `fv-pod-persist-brev1-1011200856` is now **parked** again, kept and not deleted. This session's permission policy then refused the `pod launch --provider brev`, so no Brev VM was started. |

## Boots (serverless, run 1)

Both endpoints:

- GPU types: H100 80GB HBM3 / H200 / RTX PRO 6000 Blackwell Server Edition, in any DC;
- `allowed_cuda` `["13.0"]`, the new r2 default;
- `workers_min` 0, `workers_max` 1, `idle_timeout_s` 5;
- FlashBoot on, and a 60 min delete backstop.

Both workers were **H100 80GB** (sm_90, 84.4 GB free, $3.99/h), with **driver 580.126.09**. The boot's new driver
check passed on both. fv-control still does not report the worker's DC (`dc: null`), so **the DC is unknown**.

The prompt was the same for both families: "A red fox trots through fresh snow at dawn, low sun, gentle camera push
in".

- h3: fal `minimax/h3-turbo/text-to-video`, 768P, 5 s, seed 7.
- wan: native `/fv/v1/jobs`, `fastwan22-ti2v-5b`, seed 1, which is 1248×704 and 121 frames.

Both requests used the `kind: http` envelope with `wait: true`, invoked with `endpoint invoke --async`. The
round-1 polling fix works.

"Click" is the CLI invoke. "Done" is when the CLI saw `finished_at`, polling every 2 s. "Video" adds the GET of the
mp4 from R2 to its last byte. Times are UTC.

| Boot | Image pull | Cache | Start → plan | GB fetched | Fetch s | MB/s | Slow restarts | Fetch end → fv-serve | Load / ready | Start → ready | Job delay / exec | Click → video |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| h3 `mubmapw4cylsv2` (cold) | 49 s (new image, not on the host) | miss | 7 s | 177.19 | 255.8 | **693** | 44 | 5 s (aux 1.5 s) | 46.6 s resident, **ready_after_s 52.0** (W8A8 quantized at load) | **320 s** | 364.9 s / 30.3 s (first job) | **399.0 s** |
| h3, 2nd invoke, ~1 min after the worker exited | — | FlashBoot revival: no new container, no boot | — | 0 | — | — | — | — | resident | — | **1.1 s / 23.3 s** | 26.9 s |
| h3, 3rd invoke after 10.7 min idle | — | FlashBoot revival | — | 0 | — | — | — | — | resident | — | **1.9 s / 24.0 s** | 28.0 s |
| wan `hzfdre3o4knpjb` (cold) | 47 s (new image, not on the host) | miss | 7 s | 58.78 | 142.7 | **412** (bulk 614) | 49 | 6 s (aux 1.4 s) | 79.3 s resident, ready_after_s 85.5 | **240 s** | 284.9 s / 16.3 s | **303.3 s** |
| wan, 2nd invoke, ~1.5 min after the worker exited | — | FlashBoot revival | — | 0 | — | — | — | — | resident | — | **0.9 s / 16.4 s** | 20.2 s |
| wan, 3rd invoke after 11.5 min idle | — | FlashBoot revival | — | 0 | — | — | — | — | resident | — | **0.9 s / 15.3 s** | 19.5 s |

Notes:

- **The round-1 fixes held.**
  - Each boot step is timed: tools 0.0 s and fetcher 0.0 s, both from the image.
  - The plan arrived 6 s after container start, against 13-145 s in round 1.
  - The aux files came from R2 in 1.4-1.5 s, against up to 151 s in round 1.
  - The driver check passed on both workers.
  - The H3 plan holds the FastH3 LoRA and `h3-base/text_encoder_fp8`. It skipped `transformer_prequant_4step-vsa_mxfp8` on sm_90.
  - 0 retries, 0 bad parts and 0 resumed parts in both fetches.
- **The image pull was 47-49 s on both hosts**, because the `1f9f61d` image was new to every host. Runpod pulled it
  as soon as the endpoint was created, before the invoke. So the pull overlapped the queueing and added only part of
  its time. In round 1 one host already had the image.
- **H3 on H100 quantizes at load.** The log says "no tree at …/transformer_prequant_4step-vsa_w8a8". The W8A8
  prequant tree is not in R2, and the MXFP8 one is useless below sm_100. The load still took 52.0 s, the same as the
  volume's 52 s.
- **The wan exec is not comparable to the baselines.** This is the 5B TI2V model at 1248×704 with 121 frames: 16 s,
  of which video decode took 9.4 s. The gateway baseline served `fastwan21-1.3b`, a 5 s job.
- Runpod listed **three workers per endpoint** with `workers_max` 1. The extra two started at +21 s and +42 s and
  showed `EXITED` at once. Only one worker per endpoint booted the image. Whether the other two were billed is
  **UNVERIFIED**.

### The fetch on a GPU host

The fetcher is round 1's fixed one: 64 streams × 128 MiB GETs with the straggler watchdog.

- It ran at **600-790 MB/s in bulk** on both H100 hosts. The CPU pod in US-NC-1 had done 1.2-1.6 GB/s.
- The h3 fetch reached **693 MB/s overall**: 177 GB in 256 s. The tail was not a problem there.
- The **wan fetch fell to 412 MB/s overall**.
  - 58.3 of 58.8 GB arrived in 95 s, at 614 MB/s.
  - Then one part, `wan22-ti2v-5b/transformer/diffusion_pytorch_model-00003-of-00005.safetensors` part 10, ran at
    1-1.6 MB/s on three new connections in a row. The watchdog cut it each time after about 16 s.
  - That one part took **47 s** at the end of the fetch.
- The watchdog fired after 12-18 s, not 4 s. The 4 s is the slow threshold inside a GET, and the rate check adds
  its own window.
- **Per GET, a GPU host managed about 13-17 MB/s** ("recent parts 13-17 MB/s"), so 64 streams give about 1 GB/s at
  most.
- Proposed follow-ups:
  - more streams on GPU hosts (they have 224 CPUs);
  - a part that has been slow twice could be fetched again as two half-size ranges in parallel;
  - start the replacement GET in parallel with the slow one (hedging) instead of after cancelling it.

## Against the volume and Hub baselines

| | Volume / Hub baseline | R2, round 2 | R2 vs baseline |
|---|---|---|---|
| Serverless wan, cold from zero: start + load | 49 + 70 = **119 s** (gateway.md §8.6) | container start → ready **240 s** (fetch 143 s, load 85 s) | **2.0× slower** |
| Serverless h3, cold from zero: start + load | 80 + 52 = **132 s** | container start → ready **320 s** (fetch 256 s, load 52 s) | **2.4× slower** |
| Serverless h3, first job | 24.4 s, +45 s on a worker's first job | 30.3 s (fal total 23.9 s) | about the same |
| Click → video, wan, from zero (e2e/gateway.md: 158 s, `fastwan21-1.3b`) | 158 s | **303 s** (`fastwan22-ti2v-5b`; the 16 s job instead of 5 s accounts for 11 s) | **1.9× slower** (about 1.6× without the new image's pull) |
| Click → video, h3-turbo, from zero (e2e/gateway.md: 212 s) | 212 s | **399 s** | **1.9× slower** (about 1.7× without the pull) |
| After an idle-out (FlashBoot) | (volume: fv-serve starts fresh, images.md §FlashBoot) | revived with the model resident: delay **0.9-1.9 s**, after 1 and 10-12 min idle | No fetch and no load while FlashBoot keeps the worker |
| Pod h3-turbo, ready 75.7 s after fv-serve start from the EU volume (release-54) | 75.7 s | **not measured**: the pod never reached fv-serve (finding 2) | — |
| Brev FastWan from the Hub: cold 740 s, warm restart 349 s | 740 / 349 s | **not run** (permission refused) | — |

**Measured R2 rate per provider (the fixed fetcher):**

- **Runpod serverless, H100 80GB, DC unknown: 693 and 412 MB/s overall (bulk 600-790 MB/s).** Mean 0.55 GB/s.
- Runpod CPU pod (`cpu3c` 16 vCPU, US-NC-1, r2-weights.md §5.4): 1.2-1.6 GB/s.
- Runpod GPU pod: not measured (finding 2).
- Brev: not measured.

The planning rate `R2_DOWNLOAD_GBPS` moves from 0.5 to **0.55** (`control/src/r2weights.ts`, r2-weights.md §5.4).
An endpoint's outlook already switches to its own measured rate after one boot: after this round it showed
0.693 GB/s for h3 and 0.412 GB/s for wan.

**FlashBoot retention.**

- All four second and third invokes were revivals. There was no new container, no boot row and no fetch, and the
  job waited 0.9-1.9 s. The longest idle was 11.5 min (wan) and 10.7 min (h3).
- Runpod's health showed `idle 1 / ready 1` while its worker list said `EXITED`, as in round 1.
- The balance fell **$0.25** over the 10 idle minutes (07:45 → 07:55) with both workers retained. At the GPU rate,
  two H100s would cost $1.33. The $0.25 includes billing lag from the jobs that ended at 07:44-07:45 and the
  account's 0.188/h baseline. So **a retained worker does not seem to be billed at the GPU rate**. This is two
  balance readings, so it is **UNVERIFIED**.
- How long FlashBoot keeps a worker beyond about 12 min is **UNVERIFIED**.

## Findings

1. **H3 r2 works end to end.** Round 1 found the r2 plan had no LoRA tree. This round, the cold start fetched all
   trees, loaded in 52 s and served a 768P video (5.27 MB) through the fal API. The W8A8 prequant tree is missing on
   H100 / H200 (quantized at load). The load time still matched the volume, so that is a follow-up, not a blocker.
2. **Standalone r2 pods hang silently after the plan (bug; blocks every r2 Runpod pod).**
   - The pod `xy2ofjafrwtv2f` booted normally:
     - H100 SXM, US-GA-2, $3.99/h;
     - pull 15 s, container +22 s;
     - `driver 580.126.09 ok`;
     - `step fetcher: from the image` at 07:59:49.
   - Then the container printed nothing for 18 min. The diagnosis re-reads Runpod's whole log tail every minute, and
     the tail still ended at that line at 08:17.
   - The proxy answered 502, Runpod's uptime grew without a restart, and fv-serve never started.
   - fv-control's `rate_limits` holds exactly **one** `weights-plan:<cluster>` request, in the 07:59 minute.
     `request_plan` retries every failure it does not raise, so the plan request came back. A 4xx or a
     `FetchError` would have been logged.
   - So the fetcher stopped between receiving the plan and its first log line ("N tree(s) to fetch"), which is
     `Fetch.prepare()`. It does the following before logging anything:
     - `os.makedirs` for every file;
     - `shutil.disk_usage`;
     - `os.ftruncate` plus **`os.posix_fallocate`** on every file (177 GB).
   - **Hypothesis (UNVERIFIED):**
     - When the filesystem has no native `fallocate`, glibc's `posix_fallocate` falls back to writing every block,
       for 177 GB. The pod's container disk may be such a filesystem.
     - That would be silent and take tens of minutes. The serverless worker's container disk took 2.5 s from plan
       to first line on the same code.
     - The round-1 CPU-pod bench ran the same `prepare()` without trouble. If the hypothesis holds, the difference
       is the filesystem under `/workspace/weights` on a GPU pod without a volume. That is not checked.
   - Proposed fix (not made in this PR: the fetcher's SHA-256 is pinned in fv-control and the image, so it needs a
     new image):
     - log "preparing N files (X GB)" before `prepare()`'s file loop;
     - drop `posix_fallocate`, since the disk check already ran and `ftruncate` leaves a sparse file, or time-box
       it;
     - log progress during `prepare()`;
     - check by booting one r2 pod on a cheap GPU, with `FV_R2_FETCH_IMAGE` pointing at a patched copy.
   - Also worth adding: the pod path does not pass `--report-url`, so fv-control sees no fetch progress for Runpod
     pods at all (GMI / Brev and serverless do report).
3. **"no stock in [eu]" for any-DC r2 pods (cosmetic).** A pod launched with `--region US-NC-1` or `US-MO-2` and
   r2 failed with Runpod's "There are no instances currently available". The operation and pool status then said
   "no stock in [eu]", because the spec keeps the nominal region `eu` (`regionOf`). The message should name the
   data centre.
4. **Pods in the stock list could not be created.** The validator's stock showed RTX PRO 6000 in US-NC-1 and
   US-MO-2, but pod creates there returned "no instances". The likely cause is the default
   `allowedCudaVersions: ["13.0"]` for r2 pods outside EUR-IS-1 (round-1 caveat: the filter can hide the newest
   hosts), which the stock query does not apply. This is **UNVERIFIED**. A standalone launch cannot set
   `allowed_cuda` (the launch schema has no such field), so `[]` cannot be tried from the CLI.
5. **The wan fetch's tail is still 47 s on a GPU host.** One part stayed slow over three new connections (see "The
   fetch on a GPU host").

## Recommendation

- **`R2_DOWNLOAD_GBPS` = 0.55** (was 0.5): the mean of the two fixed-fetcher boots on GPU hosts. Changed in this
  PR, with r2-weights.md §5.4 and serverless.md §1b.
- **Keep `DEFAULT_SLS_WEIGHTS_SOURCE = "volume"`. It is not flipped.**
  - From zero, r2 is slower end to end than the volume for both families: click → video **1.9×** (wan 303 vs
    158 s, h3 399 vs 212 s).
  - Even without the new image's pull, it is about **1.6-1.7×** slower.
  - Container start → ready is 2.0× (wan) and 2.4× (h3).
  - The load itself is no slower from the local disk (h3 52 s, wan 79-85 s against the volume's 70 s). The
    difference is the fetch, at 0.4-0.7 GB/s on these hosts.
- r2 remains the way to place an endpoint outside EUR-IS-1, or when the EU has no stock. FlashBoot hides the fetch
  after a worker's first boot: the revival waited 1-2 s after 11 min idle.
- **Do not delete the EU volume yet** (the owner decides):
  - r2 is slower on every cold start measured;
  - standalone r2 Runpod pods are blocked by finding 2;
  - Brev and the GPU-pod fetch rates are unmeasured;
  - the W8A8 prequant tree that the volume has for H100 is not in R2.
  - Revisit after:
    - finding 2 is fixed and an r2 pod is measured against the 75.7 s volume pod;
    - the Brev warm restart is measured;
    - the fetch on GPU hosts is faster (more streams or hedged parts), or a per-DC cache volume (r2-weights.md
      §9.4) is weighed for h3.

## Spend

- **Runpod:** balance $29.06 at 07:36 → $26.65 at 08:42, after billing settled. That is **$2.41**:
  - about $0.21 of it is the account's baseline (0.188/h);
  - the h3 standalone pod: $1.26 (19 min at $3.99/h);
  - the two endpoints: about $0.94.

  The budget was ≤ $10 and the floor $12. The balance was never near either.
- **Brev:** $0 (no VM started). The parked VM's storage stays at $0.98/day as before.

Left running: nothing of this round.

- `bench-h3-r2` and `bench-wan-r2` are deleted. `endpoint list` shows them `deleted`, with workers 0.
- The standalone `bench-h3-pod` is deleted (`pod list`: gone). Its earlier attempts in US-NC-1 and US-MO-2 never got
  a pod and were deleted.
- The Brev VM `fv-pod-persist-brev1-1011200856` was moved from held to **parked** by `brev unpark`. It was not
  started, stopped or deleted.
