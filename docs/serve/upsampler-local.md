# Local prompt upsampler: llama-server sidecar (Qwen3-VL-4B-Instruct)

Date: 2026-10-13. Owner decisions (2026-10-13): prompt upsampling runs on a **local** small instruct model served by
llama.cpp's `llama-server` as a sidecar next to fv-serve, on the same GPU; the same day the model became
**Qwen3-VL-4B-Instruct**, so the director's opening expansion can see `configure.image_url` and end images. The
upsampler itself (director + batch rewrites, `[upsampler]`, `FV_UPSAMPLER_*`) is fv-serve's
(docs/serve/prompt-upsampling.md, branch `claude/prompt-upsampling`); this page is the model, the image, the boot and
the measurements.

## 1. Model and weights

| | |
|---|---|
| Model | **Qwen3-VL-4B-Instruct** (non-thinking instruct), Qwen's official GGUF repo `Qwen/Qwen3-VL-4B-Instruct-GGUF` @ `1cd86afb9a95c410a6038ab3b40d8b578c892266` |
| Licence | **Apache-2.0** (the repo's and the base model's Hub card; the base repo `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec` says the same). No use restriction. |
| Files | `Qwen3VL-4B-Instruct-Q8_0.gguf` 4 280 406 144 B, sha256 `054721f4…63eb1`; `mmproj-Qwen3VL-4B-Instruct-F16.gguf` (vision projector) 836 180 256 B, sha256 `256f3a43…85331`; `README.md` |
| R2 tree | `qwen3-vl-4b-instruct-gguf`, rev = the Hub revision, **5.12 GB**, `weights/qwen3-vl-4b-instruct-gguf/1cd86afb…/`, manifest sha256 `eb465f96…a7477ab` (scripts/gpu/weights-r2.tsv) |
| Recorded in | weights-manifest.tsv + weights-revisions.tsv (a Hub row), weights-sha256.tsv (LFS SHA-256), verify-weights.sh cell `upsampler-local`, r2-weights.py `verify --deep --record` passed |
| Volume | **none.** The tree is in R2 only (CLAUDE.md: the 500 GB EU volume holds H3 + LTX); every pod fetches it from R2, whatever its weights source |

**Quant: Q8_0 + F16 projector.** At 4 B parameters the whole model is 4.3 GB in Q8_0, a few percent of a 96 GB
RTX PRO 6000 or an 80 GB H100, so there is no memory reason to go lower. Q8_0 is generally
within noise of BF16 in llama.cpp's perplexity / KL-divergence comparisons, while 4-bit quants such as Q4_K_M
(2.5 GB) lose more on small models (general llama.cpp results, not measured on this model here), and the director needs
exact entity ids and strictly valid JSON every segment. The F16 projector (0.84 GB) keeps the image encoder at full
precision (the Q8_0 projector saves 0.38 GB, not worth a loss on the image path). Decode speed is memory bound, and
the measured Q8_0 rates (§5) are far above what one director segment needs.

**Superseded: `qwen3-4b-instruct-gguf`** (rev `gguf-20261013`, 4.28 GB). The text-only Qwen3-4B-Instruct-2507,
which has no official GGUF, converted with llama.cpp b11195 `convert_hf_to_gguf.py --outtype q8_0` from
`Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f` (Apache-2.0; its LICENSE and a provenance.json are in the tree), was
published first the same day. It stays in R2 (add-only) and in the records, unused: every default points at the VL
tree. It can still be served with `FV_UPSAMPLER_TREE=qwen3-4b-instruct-gguf FV_UPSAMPLER_MODEL=…` (text only).

## 2. The image: `llama-server` in the CUDA serve images

- **Build** (`docker/gpucheck.Dockerfile` stage `llama-build`): llama.cpp **b11195** (commit `d834d44e`, 2026-09-26),
  cloned at the tag and checked against the commit; CUDA 13.4 nvcc + cuBLAS + cudart dev packages from
  `scripts/gpu/cuda-13*.pins` (the same versions base-cuda ships); CMake 3.31.6 from PyPI, hash-pinned.
  `CMAKE_CUDA_ARCHITECTURES=80-real;90-real;100-real;120a-real` (A100, H100/H200, B200, RTX PRO 6000).
- **Lean**: static llama/ggml/mtmd libraries; `GGML_CUDA_NO_VMM` (no `libcuda` link: the host's driver is loaded by
  cudart), no OpenMP (`libgomp` is not in the image), no curl, no NCCL, no web UI (no download at build time),
  `GGML_NATIVE=OFF` (portable AVX2 CPU code). OpenSSL is on: llama-server downloads `https://` `image_url` parts
  itself; `libssl3` is already in the base (ca-certificates). DT_NEEDED: glibc, libstdc++, libgcc_s, libssl/libcrypto,
  libcudart, libcublas, libcublasLt: all from base-os / base-cuda. The build fails if `libcuda`, `libgomp`, `libcurl`
  or `libnccl` is linked, and `serve-cuda-bin` fails if `ldd` finds anything unresolved (the image's dependency audit).
- **Built once per pin**: the section between `# >>> llama-server` and `# <<< llama-server` (plus the Ubuntu digest
  and the CUDA pins) is hashed by `scripts/ci/base-images.sh` into `fastvideo-rs-runtime:llama-server-<hash>`, a
  scratch image with the stripped binary, built and pushed by serve-image only when that tag is missing (the same
  mechanism as base-os / base-cuda) and passed back as the `llama-server` named context. Later builds copy it; the
  nvcc compile does not repeat until the pin changes. (The tools-release pattern is for our Rust binaries; a C++
  dependency compiled once per pin fits the base-image pattern better.)
- **In the image** (`serve-cuda-bin`, i.e. every CUDA variant): `/opt/fastvideo-rs/bin/llama-server`,
  `llama-server.version`, and the launcher `/opt/fastvideo-rs/bin/fv-upsampler` (deploy/runpod/fv-upsampler.sh). Not in
  the cpu image, not in the debug / runtime images.
- **Size delta: +82.4 MB compressed** (one layer; the CUDA variant h3-turbo went from 1007.7 MB / 16 layers on
  `h3-turbo-latest` to 1089.8 MB / 19 layers, `fastvideo-rs-serve@sha256:64057749…` from serve-image run 38461096015 on
  this branch; base-os / base-cuda layers unchanged, same digests). The published llama-server image is
  `fastvideo-rs-runtime@sha256:3201140c…` (82.4 MB). Its first build (nvcc, four archs, a 4-vCPU GitHub runner) took
  about 55 min of that run's 59; later runs reuse the tag.

## 3. Boot wiring

The switch is `upsampler_local: true` (default **off**) on a cluster pool, a standalone launch or a serverless
endpoint (fv-control's launch, cluster and serverless forms: "Prompt upsampler (local)"; presets may set it, none
does). GPU only. fv-control then gives the pod `FV_UPSAMPLER_LOCAL=1`, `FV_UPSAMPLER_REV` (the tree's pinned
revision), `FV_UPSAMPLER_MODEL`, the plan URL and fetcher (`FV_WEIGHTS_PLAN_URL`, `FV_R2_FETCH_URL`,
`FV_R2_FETCH_SHA256`) and the token the plan request needs (the cluster's ingest token; a serverless endpoint's weights
token, which an endpoint with the sidecar now gets whatever its weights source).

```
boot ── mkdir /fvstate ── fv-upsampler start ─┐ (background)
     ── weights (volume / r2 / hub) …          │  find <root>/qwen3-vl-4b-instruct-gguf (.complete at FV_UPSAMPLER_REV)
                                               │  or POST /ingest/v1/weights-plan?set=upsampler → r2-fetch.py
                                               │  llama-server … → /health 200 → one 1-token completion
     ── fv-upsampler wait (≤ 240 s) ◀──────────┘  → export FV_UPSAMPLER_BASE_URL / _MODEL / _VISION (+ HEDGE=0, TIMEOUT_MS=30000)
     ── exec fv-serve
```

- **Where the boot runs it**: Runpod cluster / standalone pods (`cluster/payloads.ts bootFor`: only when the env has
  `FV_UPSAMPLER_LOCAL=1`), GMI / Brev (`PROVIDER_BOOT`), serverless workers with a config or r2 (`SLS_BOOT`,
  `SLS_R2_BOOT`), and the image entrypoint `fv-entry` (serverless workers on the image's own config). The snippets are
  no-ops without `FV_UPSAMPLER_LOCAL=1`.
- **The fetch** is the weights' own R2 path: `POST /ingest/v1/weights-plan?set=upsampler` answers the sidecar's tree
  only, for a pool / endpoint whose spec has `upsampler_local` (409 otherwise), with any weights source; `r2-fetch.py`
  checks every part against the manifest. Where it lands: `$FV_WEIGHTS` when the pod's weights are on its own disk
  (r2 / hub; a Brev VM keeps it across stops), else `/fvstate/upsampler-weights` on the container disk: a mounted
  weights volume is never written. A copy already complete at the pinned revision is used without a request.
- **Flags** (fv-upsampler.sh; each can be overridden by env): `llama-server -m <Q8_0> --mmproj <F16 projector>
  --host 127.0.0.1 --port 18080 --alias qwen3-vl-4b-instruct -ngl 99 -c 65536 -np 2 --no-kv-unified -fa on
  -ctk q8_0 -ctv q8_0 --cache-reuse 256 --cache-ram 4096 --fit off --no-webui`.
  - Context: **32 768 tokens per slot, 2 slots** (`-c` is all slots' KV, split evenly). A director session's premise
    is up to 50 000 characters (12-14k tokens), plus up to 50 prior segments (default 12) and the entity registry.
    The model's native context is 262 144 tokens.
  - KV cache: q8_0 K and V with flash attention: 36 layers × 8 KV heads × 128 × 2 × 1.0625 B ≈ 78 KB per token,
    so ≈ **5.1 GB** for 65 536 tokens (half of f16).
  - Prompt cache: each slot keeps its KV between requests and the server reuses the longest common prefix (the
    system prompt + the premise, which every segment of a session sends first); `--cache-reuse 256` also reuses
    chunks after a shifted prefix, and `--cache-ram` keeps evicted prompts in host RAM. §5 shows whether it hits.
- **Exports** for fv-serve: `FV_UPSAMPLER_BASE_URL=http://127.0.0.1:18080/v1`, `FV_UPSAMPLER_MODEL=qwen3-vl-4b-instruct`,
  `FV_UPSAMPLER_VISION=1` (the projector loaded: the director's `[upsampler] vision = true` sends images as OpenAI
  `image_url` parts) and `FV_UPSAMPLER_LOCAL=1`. fv-serve then gives a local server its own defaults for what the
  operator left unset (TOML or env; `UpsamplerCfg` in crates/fastvideo-serve/src/config.rs): **`hedge` off** (a hedge
  doubles the load), **`timeout_ms` 60000** and the director's **`deck_size` 2**. A loopback `base_url` (127.0.0.0/8,
  ::1, localhost) counts as local too, `[upsampler] local = false` / `FV_UPSAMPLER_LOCAL=0` opts out. Why 60 s and 2
  (docs/serve/e2e/gpu-check-2026-10-13.md): while H3 denoises beside it the sidecar decodes at 32-38 tok/s (76-88
  idle), a director segment is 600-1 050 tokens of JSON, so a segment takes 9-28 s and one crossed the old 30 s timeout;
  and a deck of 6 made 17 LLM calls for 3 played segments, each competing with the chunk being built. The defaults
  live in fv-serve rather than in the launcher's exports so that they apply however fv-serve is started (fv-entry,
  fv-control's boots, a hand-written config) and an operator's TOML value is never overridden by an exported env
  default (the launcher used to export `FV_UPSAMPLER_HEDGE` / `FV_UPSAMPLER_TIMEOUT_MS`, which beat the TOML). The
  director also plans nothing beyond the next segment while a chunk waits for its prompt, and only the next segment
  after a `replan: true` direction until it executes (crates/fastvideo-fal/src/director/upsample.rs).
- **Images**: `image_url` takes a `data:` URL, or an `http(s)://` URL llama-server downloads itself (at most 10 MB,
  10 s timeout, no auth headers). Prefer data URLs for anything large, slow or private.
- **Failure**: fv-serve starts anyway. The reason is in the pod log (`[fv-boot] … upsampler: FAILED: …: fv-serve starts
  without it; upsampling falls back`), in the serverless boot steps (`upsampler_failed`; success: `upsampler <s>`),
  and as the GMI / Brev report phase `upsampler`. A llama-server that dies later is restarted (3 times); requests in
  between fall back in fv-serve. An image without the sidecar (older than this change) logs the same and goes on.
- **fv-control checks**: `upsampler_local` on a CPU pool / endpoint is refused; without the R2 weights store
  configured or the tree published, the launch check refuses it. The env keys `FV_UPSAMPLER_LOCAL` and
  `FV_UPSAMPLER_REV` are reserved (fv-control sets them). Container disk: r2 pods / endpoints get the tree's size × 1.15
  more by default.

## 4. Sizing guidance

- **GPU memory: about 11 GB** for the sidecar as configured (measured: 11 064 MiB after load, 11 214 MiB after the
  first image; text only, `FV_UPSAMPLER_VISION=0`: 9 942 MiB). That is the Q8_0 weights (4.3 GB), the q8_0 KV of two
  32k-token slots (≈ 5.1 GB), compute buffers, and the vision encoder (**≈ 1.1 GB** loaded, ≈ 1.3 GB after an image).
  It is allocated at start, before fv-serve loads.
- **Next to a video model** (measured on a 96 GB RTX PRO 6000 Server Edition, docs/serve/e2e/gpu-check-2026-10-13.md):
  the 11 GB fit, but **not every family keeps its resident plan**. Whole-GPU peaks with the sidecar: FastH3 Trim 480p
  51.4 GB (session) / 52.4 GB (768p job), H3 turbo with an image 45.9 GB. With the sidecar loaded first, H3 saw
  **60.6 GiB free** after its FP8 text encoder, against a **64.6 GiB resident plan** (768p 5 s), so `dit_offload auto`
  **streamed the DiT** from host memory: for Trim that cost nothing (57.8 GB/s H2D, 100% hidden behind compute; 1.6
  s/step at 480p either way). For **anchored H3 directors (turbo / max: every continuation chunk is image-to-video)** it
  was not fine with the image of that check: the i2v vision tower was streamed too, so every chunk re-read the vision
  tower and the ~26 GB FP8 language model from the network volume (~0.3 GB/s there): **160-214 s per turbo chunk**.
  Since this branch (`i2v_tower_beside_streamed_dit`, crates/fastvideo-cudarc/src/h3/pipeline.rs) the 2 GiB tower stays
  resident beside a DiT that streams anyway (the streamed plan is < 30 GiB), so the language model is the resident text
  encoder again; the DiT itself still streams beside the sidecar. Not yet re-measured on a GPU (the three pods tried on
  2026-10-14 had 95.8 of 97.9 GB already in use before anything started). **Since the plan-based `auto` text encoder
  (2026-10-14, docs/perf/h3-text-encoder-residency.md)** the encoder no longer takes the DiT's room: beside the sidecar
  (~84 GiB free) `auto` streams the encoder (FP8 numbers, as before) and the DiT stays resident; image-to-video text
  then streams its vision tower and language model per request (from the page cache after the first). If H3 must keep its resident DiT beside the
  sidecar: shrink the sidecar (`FV_UPSAMPLER_CTX=32768 FV_UPSAMPLER_PARALLEL=1`: one 32k slot, KV ≈ 2.6 GB, total ≈ 8.5
  GB; `FV_UPSAMPLER_VISION=0`: −1.1 GB) — 4 GiB is the gap to the resident plan at 768p — or run it on another GPU.
  On an 80 GB H100 / A100, H3 and LTX-2.5 (presets.ts `min_vram_gb: 80`) do not keep their resident plans beside 11
  GB; a 141 GB H200 has room for every family.
- **Context**: 32 768 tokens per slot covers a 50 000-character premise (≈ 12-14k tokens: our 46 000-character prompt
  was 10.7k tokens) + 12 prior segments + the registry with room to spare; 50 prior segments of ≈ 400 characters each
  add ≈ 5k tokens. Raise `FV_UPSAMPLER_CTX` (KV ≈ 78 KB per token) only if a session approaches it; the model itself
  takes 262 144.
- **Slots**: 2 parallel requests (a director segment and a batch rewrite, or two sessions). Each slot keeps its own
  prompt cache, so two interleaved director sessions each keep their premise cached.
- **Disk**: 5.1 GB for the tree on the pod (the image grows by 82 MB).

## 5. Measurements (2026-10-13, RTX PRO 6000)

Pod `ha8wiw8h0lc7pc` (`fv-ups-bench-1013211414`), Runpod SECURE, NVIDIA RTX PRO 6000 Blackwell Server Edition (97 887
MiB, driver 595.91.07, 128 vCPU), $2.49/hr, 204 s from create to delete (GET → 404) = **$0.14**. Image
`fastvideo-rs-serve@sha256:64057749…` (this branch's h3-turbo, fake engine running beside it), driven over
scripts/serve/e2e/pod.sh's sidecar with a 1800 s backstop and a 15 min idle guard. The tree was fetched by
`r2-fetch.py` from a presigned plan; llama-server was started by `fv-upsampler run` (the image's launcher, the default
flags of §3); the requests are `scripts/serve/upsampler-bench.py` (raw lines: `docs/serve/bench/upsampler-local-2026-10-13.jsonl`).
Times are wall clock per request; prefill and decode rates are llama-server's own `timings`.

| | |
|---|---|
| R2 fetch (5.12 GB, 1 tree) | 35.5 s, 144 MB/s, 0 retries (this pod's data centre to the WNAM bucket; GPU pods measured 0.4-0.7 GB/s before, docs/ops/r2-weights.md §5.4) |
| Load (fv-upsampler: start → /health 200 → 1-token completion) | **1.1 s** (files in the page cache right after the fetch; a cold read adds 5.1 GB at the disk's rate) |
| VRAM (llama-server process) | **11 064 MiB** with the vision projector; 11 214 MiB after one image; **9 942 MiB** text only → the vision encoder takes ≈ 1.1-1.3 GB |
| Time to first token (62-token rewrite prompt, streamed) | **18 ms** |
| Decode | **97 tok/s** short prompts; 86-88 tok/s at 0.4-1k tokens of context; 64-73 tok/s at 11k |
| Prefill | 3.7-4.8k tok/s for 300-400-token deltas; **9.6k tok/s** for a 10.7k-token cold prompt (1.11 s) |

**Single-shot rewrite** ("a fox running through snow at dawn" → one paragraph, 87-98 tokens): **0.90-1.07 s**.
> A russet fox sprints through a silent, snow-dusted forest at dawn, its bushy tail flicking, paws crunching softly.
> Soft golden light spills from the horizon, casting long shadows. Camera follows from low angle, tracking its swift,
> agile stride. Mist curls around its feet, backlighting its silhouette. Snowflakes drift gently, illuminated by the
> pale morning glow, peaceful, wild, alive.

**Director, small world bible** (the spec's system prompt and `json_schema`, 4 entities, three consecutive segments,
temperature 0.8): every answer **valid against the schema** (3/3; required fields, no extra properties, enums).

| segment | prompt tokens | evaluated / cached | output tokens | end to end |
|---|---|---|---|---|
| 0 | 381 | 380 / 1 | 306 | 3.60 s |
| 1 | 676 | 319 / 357 | 340 | 4.03 s |
| 2 | 1 034 | 376 / 658 | 576 | 6.61 s |

**Director, H3 Max-style session** (a 46 000-character premise = 10.7k tokens, 12 prior segments, the registry; three
consecutive segments): **3/3 valid**. The premise prefix is **not re-prefilled**: segments 1 and 2 evaluated only the
changed tail (the registry and history follow the premise in the user message).

| segment | prompt tokens | evaluated / cached | prefill | output tokens | end to end |
|---|---|---|---|---|---|
| 0 (cold) | 10 701 | 10 698 / 3 | 1 112 ms (9.6k tok/s) | 186 | 4.26 s |
| 1 | 10 683 | 1 555 / 9 128 | 287 ms | 476 | 6.84 s |
| 2 | 10 949 | 1 832 / 9 117 | 195 ms | 316 | 4.68 s |
| text only, 0 (cold) | 10 701 | 10 701 / 0 | 1 172 ms (9.1k tok/s) | 286 | 5.30 s |
| text only, 1 | 10 797 | 1 669 / 9 128 | 148 ms | 613 | 8.40 s |
| text only, 2 | 11 133 | 2 005 / 9 128 | 166 ms | 258 | 3.69 s |

So prefill is not the cost of a long session once the premise is cached (≈ 0.15-0.3 s per segment); decoding the
JSON (200-600 tokens at ≈ 70 tok/s) is. The first segment of a session pays ≈ 1.1 s for a 50k-character premise.
llama-server disables `--cache-reuse` (chunk reuse after a shifted prefix) when a projector is loaded ("cache_reuse is
not supported by multimodal"); the plain prefix cache per slot, which is what keeps the premise, still works, as the
`cached` counts show. Reuse needs the request to land on the slot that holds the session's prefix: llama-server picks
the slot by prompt similarity, so two interleaved sessions keep one slot each.

Sample (long session, segment 2, the prompt the spec's `assemblePrompt` makes):
> a rusted floating airship dock above a sea of clouds, rope bridges, signal lanterns the Kite: a small patched red
> airship with a canvas balloon and a copper gondola the Stowaway: a small, bundled figure in a patched coat, clutching
> a brass key, eyes wide and blinking rapidly in the lantern light Mira adjusts her lantern's grip while Otto slides a
> brass key into the crate's lock; the stowaway curls into a ball, fingers twitching against the rusted edge of the
> crate. The Kite dips low, its copper gondola scraping the clouds, …

(A new entity, `the Stowaway`, was introduced with a concrete descriptor, as the direction asked.)

**Opening segment with an image** (`configure.image_url` as an OpenAI `image_url` part: a photo of snow peaks at
sunrise above a sea of clouds, 800×533 JPEG, 67 KB): **valid JSON**, the expanded prompt **describes the image**.

| image as | evaluated / cached | prefill (image encode included) | output tokens | end to end |
|---|---|---|---|---|
| `data:image/jpeg;base64,…` | 502 / 234 | 413 ms | 211 | **3.16 s** |
| `https://…` (llama-server downloads it: 78 ms) | 1 / 735 (same image: its tokens cached) | 68 ms | 204 | 2.56 s |

> The sun rises slowly behind the ridgeline, painting the sky in gradients of peach, lavender, and gold; the clouds
> below ripple gently as if breathing. Camera: Static, wide-angle, low-to-mid elevation, framing the peaks and cloud sea
> in a natural composition. Lighting: Soft, directional sunrise light; long shadows cast across the mountain slopes;
> highlights catch the snow caps and cloud tops. …

**Not measured here**: the sidecar beside a generating video model (measured since: §4 and
docs/serve/e2e/gpu-check-2026-10-13.md, decode 32-38 tok/s during an H3 denoise); a cold
disk read of the GGUF; H100 / B200 / A100. The schema check is a structural validator of the subset of JSON Schema the
spec uses (in the bench script), not a full JSON Schema library.

## 6. Upgrade path

**Qwen3-VL-8B-Instruct** (official GGUF repo `Qwen/Qwen3-VL-8B-Instruct-GGUF`, Apache-2.0) is the next step if the 4B
model's expansions are not good enough: Q8_0 ≈ 8.7 GB + its projector; the weights double, and decode (memory bound)
runs roughly half as fast. Check its KV geometry in its config.json before sizing the context. It needs a new R2 tree
(publish-hub, the same records) and `FV_UPSAMPLER_TREE` / `FV_UPSAMPLER_MODEL` (or the defaults in fv-upsampler.sh
and control/src/upsampler.ts); nothing else changes. The text-only Qwen3-8B (thinking off) is the fallback if vision
is dropped again.

## 7. Files

`deploy/runpod/fv-upsampler.sh` (launcher), `deploy/runpod/fv-entry.sh`, `docker/gpucheck.Dockerfile` (llama-build,
llama-server, serve-cuda-bin), `scripts/ci/base-images.sh` (`ensure llama-server`), `.github/workflows/serve-image.yml`,
`control/src/upsampler.ts` (env, boot snippets, checks), `control/src/weights-plan.ts` (`?set=upsampler`),
`control/test/unit/upsampler.test.ts`, `scripts/serve/upsampler-bench.py` (the measurement), the weights records in
`scripts/gpu/`.
