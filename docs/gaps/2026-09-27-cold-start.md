# Cold start: fast weight loading (E12), pre-quantized FP8 text encoders (E13), serverless measurement (WP-19)

Date: 2026-09-27. Plan: [design §0.4](../serve/design.md). Raw results (one
JSON per job, with the worker's timestamps, `load/io` lines and
`benchmark.json`): `artifacts/runpod/serverless/`.

## Setup

- Runpod serverless **queue** endpoint `ltf9nt77yn545m` (template
  `cto24f7zso`), 1x **H200** (US-CA-2), min 0 / max 1 worker, idle 5 s,
  FlashBoot off, the US weight volume `s2k01690bi` at `/runpod-volume`.
  Both deleted after the runs (`ledger.tsv`).
- Worker: `scripts/gpu/serverless-worker.sh` (bash + curl; job-take,
  job-done, ping from research-deploy §1.1), inlined into the template's
  start command so the same worker runs on any runtime image. One job = one
  fresh `fv-gpucheck` process = one full cold model load. The volume's
  HF-cache trees link absolutely into `/workspace/weights`; the worker links
  that to `/runpod-volume/weights`.
- Driver: `scripts/gpu/serverless-coldstart.sh` (`up`, `job`, `retemplate`,
  `down`).
- Workloads: FastH3 4-step VSA, 1344x768, 5 s, seed 1024, `--text-encoder
  auto` (resident FP8 on this card), no text cache. LTX-2.5 distilled
  two-stage, default workload (4k5s), `--text streamed`, no text cache.
- Every gen job first drops the weight files from the page cache
  (`fv-gpucheck evict-cache`; 12-33 s, measurement only, excluded below).
- Images: before = `sha-c582b63` (main at the branch point); after =
  `sha-47aa7b7` → `0536e9a` → `f6a775c` → `72fdf68` (the E12 iterations).

## Where the time went (before)

The loaders read the network volume through `mmap` page faults: every fault
is a small synchronous request. The consumer (the loader thread) ran at
~0.3-0.6 GB/s. A first read-ahead into the page cache (`47aa7b7`) did not
help: the read-ahead finished 100+ GB early (window full, 2 600 thread-seconds
waiting) while the loader stayed at the same rate, i.e. on this volume the
mapping does not profit from pages read ahead with `pread`. Reading into
anonymous memory the views borrow (`f6a775c`) moves reads to ~1.9-2.5 GB/s
aggregate (16 threads x 16 MiB); after that the loader's own host work is
what is left.

## Load breakdown, FastH3 (H200, seconds)

| image | E12 | E13 tree | load_s | text encoder | DiT + decoders | frames sha256 |
|---|---|---|---:|---:|---:|---|
| c582b63 (before) | – | – | **268.8** | (not instrumented) | | `6a43b801…` |
| 47aa7b7 | page-cache read-ahead | – | 322.0 | 80.6 | 241.4 | `6a43b801…` |
| 47aa7b7 | page-cache read-ahead | yes | 307.1 | 40.7 | 266.4 | `6a43b801…` |
| 0536e9a | + pinned staging, parallel f32→bf16, adapter-free linears skip f32 | yes | 235.4 | 46.3 | 189.1 | `6a43b801…` |
| f6a775c | + read-ahead into memory (verify on) | yes | 245.8 | 44.5 | 201.3 | `6a43b801…` |
| **72fdf68** | + staged FP8 codes / host-fused weights | yes | **212.3** | **18.4** | 193.9 | `6a43b801…` |

Generation itself is unchanged (total_s 26.0-26.7). The text encoder went
80.6 → 18.4 s (E13 halves the bytes and skips the quantization; E12 reads
them at ~2.5 GB/s). The DiT phase (71.6 GB viewed: 26 GB AdaLN projections,
41 GB blocks, LoRA fused on the host for every adapter target) is still
~175 s; its measured parts are LoRA fuse 14.9 s, f32→bf16 11.4 s, staging
fill 3.8 s, read wait 6.7 s. The rest is the host f32 round trip of every
adapter-fused weight (bf16 → f32 alloc/fill → fuse → bf16) and the per-block
AdaLN evaluation. **The H3 < 2 min target is not met** (212 s vs 269 s). The
next cut is fusing the adapter on the device (upload bf16 once, W + m·B·A in
f32 with the host's operation order, round-to-nearest-even) — not done here.

## Load breakdown, LTX-2.5 (H200, seconds)

| image / flags | load_s | text_s (first encode) | e2e | process | frames sha256 |
|---|---:|---:|---:|---:|---|
| c582b63 (before) | – | – | – | 409.3 | `db4e0c86…` |
| 72fdf68 with `FASTVIDEO_PREFETCH=0 FASTVIDEO_STAGED_UPLOAD=0 FASTVIDEO_LTX2_TEXT_WARM=0` | 183.4 | 57.0 | 195.9 | 386.5 | `db4e0c86…` |
| **0536e9a (E12)** | **33.0** | **27.1** | 158.1 | **193.6** | `ad76ebfc…` |
| f6a775c (E12, verify on) | 172.0* | 47.8* | 181.3 | 356.0 | `ad76ebfc…` |
| f6a775c + `FASTVIDEO_LTX2_TEXT_FP8=1` (E13 Gemma tree) | 34.3 | 34.1 (29.5 load) | 185.3 | 222.8 | `1a9cbf6e…` (FP8 numerics) |
| 72fdf68, `TEXT_WARM=0`, verify on | 155.9* | 45.6* | 178.0 | 336.3 | `ad76ebfc…` |

\* verify builds every staged weight a second time on the plain path.

LTX meets the < 1 min load target (183 → 33 s; DiT at 1.2 GB/s through the
pinned stage) and the first streamed Gemma encode drops 57 → 27 s (its
shards are read ahead while the DiT loads). Resident FP8 Gemma from the tree
(opt-in `FASTVIDEO_LTX2_TEXT_FP8=1`) loads in 29.5 s and is not faster than
the warmed streamed encode at this size; it changes the conditioning numerics
and stays opt-in.

**Open: LTX frames differ with E12 on.** E12 off (either image) reproduces
the before frames (`db4e0c86…`, 2 runs); E12 on gives `ad76ebfc…` (3 runs,
reproducible). The Gemma warm path is excluded (`TEXT_WARM=0` still gives
`ad76ebfc…`), and `FASTVIDEO_VERIFY_UPLOAD=1` compared all 1 660 staged
DiT uploads with the plain path on the device: **1 660 equal, 0 different**,
and no mismatch was logged for any later staged load. What remains is the
read-ahead (which never changes a byte a view returns) or an allocation-order
effect of the staged path on a later decision. Not bisected further (budget).
Until it is, `FASTVIDEO_PREFETCH=0 FASTVIDEO_STAGED_UPLOAD=0` restores the
exact before-output for LTX. H3 frames are byte-identical in all six runs.

## E13 trees (identity)

`fv-gpucheck quantize-text-encoder` loads the resident FP8 encoder from the
bf16 shards (capturing every linear's E4M3 codes and row scales), writes
`<root>/text_encoder_fp8/model.safetensors` + `manifest.json` (format, version,
rule, layers, source shard names/sizes/sha256), then loads the tree back and
requires the captured codes+scales digest to equal the load-time one and every
copied tensor to equal its source.

| volume | tree | linears | copied tensors | bytes | load-time digest = tree digest | write |
|---|---|---:|---:|---:|---|---:|
| US `s2k01690bi` | `h3-base/text_encoder_fp8` | 350 | 201 | 25 950 724 552 | `548475f2…` = `548475f2…` PASS | 763 s |
| US `s2k01690bi` | `ltx25/text_encoder_fp8` | 328 | 338 | 12 923 848 536 | `d9a78567…` = `d9a78567…` PASS | 323 s |
| EU `jg48s6o1w0` | not written (see below) | | | | | |

End to end: H3 with the tree produces the same frames as load-time
quantization (`6a43b801…`). Writes to the volume run at ~35-40 MB/s whether
sequential or parallel. The EU trees were not written: creating the EU
endpoint was refused by the session's permission policy (shared resource);
it needs an explicit go-ahead.

## Serverless cold-start timeline (H200, submit → first output)

| | worker start | model loaded | first output (job done) |
|---|---:|---:|---:|
| fresh host, image not cached (ls job, c582b63) | +458 s | | |
| FastH3 before (worker already up) | +6 s | +275 s | +306 s |
| FastH3 after (72fdf68, scale from 0, image cached on host) | +49 s | +261 s** | +314 s** |
| LTX-2.5 before (scale from 0) | +79 s | (≈ +79 +183) | +495 s |
| LTX-2.5 after (0536e9a; queued behind the FastH3 job, times from job start) | (0) | +33 s*** | +194 s*** |

\** includes 24 s of page-cache eviction before the load (measurement only):
submit → first output without it ≈ 290 s, of which load 212 s.
\*** from the process start (after the eviction step); `load_s` 33 s, then 27 s text + 131 s generation. A scale-from-0 LTX job would add the worker start (+5-80 s with the image cached).

Image pull: a fresh H200 host took 458 s from submit to worker start (pull of
the ~6 GB runtime image plus scheduling); with the image on the host, 5-80 s.
FastWan was not measured (budget).

## Knobs

`FASTVIDEO_PREFETCH=0` (read-ahead off), `FASTVIDEO_PREFETCH_MODE=cache`
(page-cache instead of memory), `FASTVIDEO_PREFETCH_THREADS` (16),
`FASTVIDEO_PREFETCH_CHUNK_MB` (16), `FASTVIDEO_PREFETCH_WINDOW_GB` (cache mode),
`FASTVIDEO_STAGED_UPLOAD=0`, `FASTVIDEO_VERIFY_UPLOAD=1`,
`FASTVIDEO_TEXT_FP8_TREE=0` (ignore E13 trees), `FASTVIDEO_LTX2_TEXT_FP8=1`,
`FASTVIDEO_LTX2_TEXT_WARM=0`. `fv-gpucheck io-bench`, `evict-cache`,
`quantize-text-encoder [--verify-only]`.

## Spend

Endpoint billing: ~3 690 s of H200 worker time billed at the last check,
$6.09, plus the final bisect job (~350 s, ~$0.6): about **$6.7**.
