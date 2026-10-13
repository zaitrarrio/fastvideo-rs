# R2 cold start with streaming load (2026-10-13)

This run measures the cold start with the streaming load of PR #78 (docs/ops/r2-weights.md §11). With it,
fv-serve starts a few seconds into the fetch, and its loader reads each part as soon as the part is verified.

- **Setup.** Staging fv-control at main `680196f`, driven with `scripts/serve/fv-control.sh`. Both endpoints were
  pinned to the images of the `serve-image` run for `680196f` (run 38368357378, `image: {sha: 680196f}`).
- **Image check.** That run's "Prebuilt tools" notice names **tools-v0.1.8** ("built from this commit's tools
  inputs"). The fv-serve binary in the `wan5b` and `h3-turbo` images is a shared layer, `sha256:624e3589…`, pulled
  from ghcr and grepped. It contains `fv-streaming-fetch/1`. The log of every boot showed the line
  `fastvideo-loader: streaming fetch (fv-streaming-fetch/1)` and the steps `stream_ready` and `load_overlap`.
- **Earlier main images lacked the marker.** The first main image with PR #78, `5dcc5a0`, was built with the
  stale prebuilt binary `tools-v0.1.7`, so its fv-serve has no streaming code. tools-v0.1.8 was published 20 min
  after that image was built, and its publish step failed before it could dispatch the image rebuild. PR #82 fixed
  that step, and the images were rebuilt.
- The stable channel was not promoted.

Both endpoints used the same settings as round 2 (docs/ops/r2-bench-2026-10-12-round2.md):

- `weights_source: r2`;
- GPU types: H100 80GB HBM3 / H200 / RTX PRO 6000 Blackwell Server Edition, in any DC;
- `allowed_cuda` `["13.0"]` (the default);
- `workers_min` 0, `workers_max` 1, `idle_timeout_s` 5;
- FlashBoot on, and a 45 min delete backstop.

Each endpoint got one cold invoke from zero, with round 2's prompt and parameters. Both workers that served were
**H100 80GB** (sm_90, driver 580.126.09, $3.99/h). The DC is unknown (`dc: null`).

## Results

Times are UTC. "Start" is the container start, from the boot log's `+s` stamps. "Ready" is `FV-SERVE READY`.
"Video" is the GET of the mp4 to its last byte.

| | wan (`fastwan22-ti2v-5b`) | h3-turbo (`fasth3`, 768P 5 s) |
|---|---|---|
| Worker | `49yidzr9y4w3qk` | `2ili33lzisp2lv` (the second; see "Runpod placement") |
| Cache | miss | miss |
| `stream_ready` (fetcher start → unguarded files and aux trees in place) | 13.6 s (+18.2 s) | 11.6 s (+16.1 s) |
| aux check | 1.5 s | 1.5 s |
| fv-serve start | **+22.8 s** | **+20.7 s** |
| Fetch | 58.78 GB in 101.7 s, **578 MB/s** (41 slow restarts, 0 retries / bad parts) | 177.19 GB in 250.2 s, **708 MB/s** (103 slow restarts, 0 retries / bad parts) |
| Fetch end | +108.3 s | +257.0 s |
| `load_overlap` (fv-serve loading while the fetch ran) | 88.5 s | 240.2 s |
| Model resident (load start → resident) | 166.8 s | 240.4 s (W8A8 quantized at load, `lora_s` 125.3) |
| fv-serve `ready_after_s` | 173.3 | 246.8 |
| **Fetch end → ready** | **87.9 s** | **10.5 s** |
| **Container start → ready** | **196.2 s** | **267.5 s** |
| Job delay / exec | 942.5 s / 17.3 s (run 13.3 s, of which decode 9.5 s) | 869.8 s / 34.1 s (fal total 27.5 s: text 7.9, denoise 14.3, decode 4.8) |
| Container start → video | **216.5 s** | **303.0 s** |
| Click → video, measured | 963.4 s (12.6 min before Runpod started a worker that booted) | 906.0 s (a first worker stopped by Runpod mid-fetch, then a fresh worker) |

### Against round 2 and the EU volume

Round 2's "container start → video" is its click → video minus the time from click to container start. That time
is job delay − (start → ready): about 45 s for both families.

| | EU volume baseline | R2 round 2 (sequential) | R2 streaming (this run) | Streaming vs round 2 |
|---|---|---|---|---|
| wan: fetch | — | 142.7 s at 412 MB/s (47 s straggler tail) | 101.7 s at 578 MB/s | −41 s (a fetch-rate difference, not streaming) |
| wan: fetch end → ready | — | 6 + 85.5 = 91.5 s | 87.9 s | **−3.6 s** |
| wan: container start → ready | 49 + 70 = **119 s** | **240 s** | **196.2 s** | −44 s (−18 %); 1.65× the volume |
| wan: container start → video | — | ≈ 258 s | 216.5 s | −42 s |
| wan: click → video | **158 s** (`fastwan21-1.3b`, 5 s job) | 303.3 s | 963.4 s measured; **≈ 261 s** with round 2's 45 s before the container | ≈ −42 s; ≈ 1.65× the volume |
| h3: fetch | — | 255.8 s at 693 MB/s | 250.2 s at 708 MB/s | about the same |
| h3: fetch end → ready | — | 5 + 52.0 = 57 s | 10.5 s | **−46.5 s** |
| h3: container start → ready | 80 + 52 = **132 s** | **320 s** | **267.5 s** | **−52.5 s (−16 %)**; 2.0× the volume |
| h3: container start → video | — | ≈ 354 s | 303.0 s | −51 s |
| h3-turbo: click → video | **212 s** | 399.0 s | 906.0 s measured; **≈ 348 s** with round 2's 45 s before the container | ≈ −51 s; ≈ 1.64× the volume |

The "≈" click → video rows are estimates. This run's measured click → video was dominated by Runpod placement
(below), which has nothing to do with the boot.

## Findings

1. **On H3, streaming does what it should.**
   - The load ran almost entirely under the fetch: `load_overlap` 240 s of a 240 s load.
   - fv-serve was ready **10.5 s after the last byte**, against 57 s in round 2.
   - The cold start is now the fetch plus about 30 s (16 s to `stream_ready`, about 5 s to fv-serve, 10 s tail).
   - The fetch rate did not drop with the load running: 708 MB/s against 693 MB/s in round 2.
   - The loader's `want` boosts worked: each of the 14 DiT shards was moved to the front as the loader reached it.
   - `fetch_wait_s` 1061.6 is summed over the loader's threads.
2. **On Wan, streaming saved almost nothing: 3.6 s.**
   - fv-serve started at +22.8 s and then sat blocked. The model became resident 87.9 s after the fetch ended,
     which is round 2's load.
   - **Cause** (crates/fastvideo-cudarc/src/wan/pipeline.rs): a one-DiT Wan checkpoint opens its transformer with
     `WeightMap::from_dir`, which is the eager map (the log says `wan load: … (eager dit)`). Eager loaders wait for
     the **whole file** (r2-weights.md §11, "The loader").
   - The fastwan DiT is a single 10 GB `.safetensors` file, and it was the **last file to complete**, at
     08:44:03. It was boosted from 08:42:44, but 30-odd of its parts were cut as slow and fetched again.
   - The DiT is also opened before the text encoder and the VAE, so nothing else loads while it waits.
   - All of wan's 44 s gain against round 2 comes from the fetch: round 2 had a 47 s single-part tail, this run
     had none that long.
   - **Fix (not tried here):** open the Wan DiT lazily while `FASTVIDEO_STREAMING_FETCH=1`. The switch already
     exists as `FASTVIDEO_WAN_DIT_LAZY=1`, and the A14B uses that path. Its `LazyStore` views wait per part.
     Expected: a tail like H3's, and about 196 → 120 s container start → ready (**UNVERIFIED**). A one-endpoint
     test with that env var costs about $0.5.
   - **Tested** in docs/ops/r2-bench-2026-10-13-wan-lazy.md: ready 3.8 s after the last byte, container start →
     ready 106.6 s, output byte-identical to eager.
3. **The wan preset fetches a tree the boot does not load.**
   - The wan endpoint fetched `wan22-ti2v-5b` (34.2 GB, finished at 82 s) as well as `fastwan22-ti2v-5b`
     (24.2 GB, finished last at 102 s). Only `fastwan22-ti2v-5b` is loaded at boot: the log shows one model
     made resident before `FV-SERVE READY`.
   - The two shared the 64 streams. Fetching the boot model's trees first, or the other tree after fv-serve is
     ready, would shorten the critical path. That is not measured.
4. **Runpod placement dominated click → video.** Neither delay came from the boot.
   - **wan:** the first worker, `s09g79aackuu0e`, stayed "initializing" from 08:29:27 to about 08:41, with no
     container log and no system log. It then went "throttled". Runpod placed a third worker, `49yidzr9y4w3qk`,
     whose container started at 08:42:15, 12.5 min after the click.
   - **h3, first worker** (`exsmpzn2qhsl28`):
     - pulled the image in 19 s (08:47:29 → 08:47:48, new to the host);
     - started its container at 08:48:35;
     - reached `stream_ready` at 34.5 s;
     - fetched at only **246 MB/s** (126 of 177 GB in 8.6 min).
   - Runpod's system log then says `stop container` at 08:57:12 (517 s after the start) and `remove container`.
     No boot report said failed. A new worker, `2ili33lzisp2lv`, started at 08:57:27 and fetched from zero.
   - Why Runpod stopped it is **UNVERIFIED**. A Runpod limit on worker initialization time is one guess. A third
     worker, `278p2l5xbmrgbq`, showed `EXITED` at once, as in round 2.
   - A slow host stopped mid-fetch loses its whole fetch, because a new worker has an empty disk.
5. **The tools-release race** made the first main image (`5dcc5a0`) ship without the streaming code (see the
   top). It was fixed by PR #82 and a rebuild.

## Recommendation

- **Keep `DEFAULT_SLS_WEIGHTS_SOURCE = "volume"`.**
  - From zero, r2 with streaming is still about **1.65×** the volume, container start → ready: wan 196 vs 119 s,
    h3 268 vs 132 s.
  - On H3 the fetch is now the whole cold start: about 250 s for 177 GB at 0.7 GB/s.
  - Only a faster fetch moves it further: more streams, hedged parts, the per-DC cache volume, or a smaller tree
    (the W8A8 prequant tree would also drop the 125 s quantize-at-load from the critical path once the fetch is
    faster).
- **Next, for Wan:** make the DiT lazy under streaming (finding 2), and fetch the boot model's trees first
  (finding 3).

## Spend

- **Runpod:** balance **$21.80** before (07:4x UTC) and **$19.76** at 09:04, both endpoints deleted. That is about
  **$2.04**, which includes the account's baseline of 0.188/h. Billing may still settle a little lower. The budget was $3 and the floor $12.
- **Left running:** nothing. `endpoint list` shows `bench-wan-stream` and `bench-h3-stream` as `deleted`, with
  workers 0. No pods were created.
