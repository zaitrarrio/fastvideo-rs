# R2 weights store: layout, publish, verify, fetch, migration

Date: 2026-10-11. Owner decision (2026-10-11): "Use R2 for the weights and get rid of the EU volume; can we use
HTTP QUIC / HTTP/3 or any other option to accelerate the transfer of weights." Cloudflare R2 becomes the canonical
weights store. It replaces the Runpod EU network volume `jg48s6o1w0` (`fv-weights-h3-ltx-hy`, EUR-IS-1,
docs/ops/runpod-volumes.md). Pods on any provider (GMI, Brev, Runpod) fetch their trees from R2 at boot onto
local disk (Brev: the persisted `~/workspace`), instead of mounting a volume or pulling from the Hub.

**Status of this page's claims.** The tooling and fv-control's `weights_source r2` exist and are tested (§8).
On 2026-10-11/12 all 46 trees of §6.1 (798 files, 1742.04 GB) were copied off the EU volume into `fv-weights`, and
`runs/` and `weights/wan/` were archived there. Every copied tree was compared with the volume, and read back by
`verify --deep` (§10). Throughput on a GPU node has **not** been measured (§5.4 gives the estimate and how to measure
it). The volume stays until §7's gate passes and the owner confirms.

## 1. Where things are

| What | Where |
|---|---|
| Bucket | `fv-weights`, account `0cd06fd37ee4e07de370821cb3852a8a`, location hint **WNAM** (created 2026-10-11 with `wrangler r2 bucket create fv-weights --location wnam`) |
| Staging bucket (tests) | `fv-weights-staging`, WNAM, empty |
| S3 endpoint | `https://0cd06fd37ee4e07de370821cb3852a8a.r2.cloudflarestorage.com` (path style, region `auto`) |
| Publisher / verifier | `scripts/gpu/r2-weights.py` (Python 3.9+, stdlib only) |
| Pod fetcher | `scripts/gpu/r2-fetch.py` (Python 3.8+, stdlib only). fv-control serves it at `/ingest/v1/r2-fetch.py` |
| Published trees (the record) | `scripts/gpu/weights-r2.tsv` → `control/src/r2-trees.ts` (`node gen-configs.mjs`) |
| fv-control | `control/src/r2weights.ts` (presign, manifests, boot step), `weights-plan.ts` (the plan endpoint), `providers.ts` / `cluster/payloads.ts` (`weights_source r2`) |
| Tests | `scripts/gpu/tests/test_r2_weights.py` (+ `fake_r2.py`), `control/test/unit/r2weights.test.ts`, the integration step "weights_source r2" |

**Why a dedicated bucket, and why WNAM.** The bucket holds nothing but weights. fv-control's read-only key can then
be scoped to it alone (R2 tokens scope by bucket, developers.cloudflare.com/r2/api/tokens), its usage and bill show
on their own, and no outputs or logs share its lifecycle. The location hint follows where the GPUs mostly run.
GMI's IDCs are mostly in the US (Denver ×2, Oregon, Iowa ×4, Ohio: `GET /v1/idcs`, 2026-10-11). Brev's control
plane is us-west-2. Runpod is EUR-IS-1 only while the volume exists. Hints are best-effort and are honoured only
on a bucket's first creation (developers.cloudflare.com/r2/reference/data-location). EU pods reach WNAM over a
longer path. The many parallel streams of §4 hide most of the latency, but this is **UNVERIFIED**. If EU becomes
the main GPU region, a second bucket `fv-weights-eu` (WEUR) can be filled from this one with the same tools.

## 2. Layout and the add-only rule

```
weights/<dest>/<rev>/<path>        the tree's files, byte for byte (plain files, not the HF-cache layout)
parts/<dest>/<rev>/<path>.json     per-file record, written right after the file: size, sha256, per-part sha256
manifests/<dest>/<rev>.json        the tree's manifest, written LAST: the tree is published once it exists
verified/<dest>/<rev>/<stamp>.json a verification record (`verify --deep --record`)
```

- `<dest>` is the `weights-manifest.tsv` name (`h3-base`, `auxiliary/upscalers/seedvr2`, …). Nested trees
  (`h3-base/text_encoder_fp8`) are trees of their own. A pod places them inside their parent.
- `<rev>` is the pinned Hub revision (40 hex, `weights-revisions.tsv`) for Hub trees. For volume-only trees it is
  `eu-<YYYYMMDD>`, the day they were copied off the volume. A disk filled from the Hub and one filled from R2
  carry the same `.fv-revision`, so a Brev VM parked with either kind serves both.
- **Manifest** (`fv-weights-manifest/1`): `dest`, `rev`, `part_size` (64 MiB), `total_bytes`, `source`
  (hub repo + revision + globs, or the directory copied), `created_at`, and `files: [{path, size, sha256, parts:
  [sha256 of each 64 MiB part]}]`. The per-part hashes let a fetcher check every range as it arrives.
- **Add-only, like the volume.** Nothing overwrites or deletes an object. A large file goes up as a multipart
  upload, which stays invisible until it is completed. Its SHA-256 is computed while it streams and checked
  **before** completion against what is known about it: the Hub LFS SHA-256, the git blob SHA-1 of a small Hub file,
  and `weights-sha256.tsv` (sha256 or md5). A mismatch aborts the upload. R2 checks every part body against the
  SHA-256 in its signature (`x-amz-content-sha256`; the fake R2 in the tests does the same). Small files and the
  records and manifest go up with `If-None-Match: *`. On 2026-10-11 R2 answered a second such PUT to the same key
  with 412, and the first object was unchanged. An object that exists is reused only when its size and recorded
  SHA-256 match; otherwise the run stops.
- **The record pins the manifest.** `weights-r2.tsv` holds each published tree's manifest SHA-256. fv-control
  refuses a manifest whose bytes differ (`r2weights.ts r2Manifest`), so pods load what the repo recorded.

## 3. Publish and verify (`scripts/gpu/r2-weights.py`)

Credentials are never printed and never put on a pod's command line. The tool takes
`FV_WEIGHTS_R2_ACCESS_KEY_ID` / `FV_WEIGHTS_R2_SECRET_ACCESS_KEY`. Without them it derives S3 credentials from a
Cloudflare API token with R2 rights (`--cf-token-file`, default `/root/.config/fv/cf_api_token`): the access key
is the token's id and the secret is the SHA-256 of the token (developers.cloudflare.com/r2/api/tokens, "Get S3
API credentials from an API token"). For a pod, create an **Object Read & Write** token scoped to `fv-weights`
in the dashboard (R2 → Manage API tokens). Pass its key pair in the pod's env and delete the token afterwards.

```bash
cd scripts/gpu
python3 r2-weights.py status                         # every known tree: published / not, rev
python3 r2-weights.py publish-hub h3-base            # Hub -> R2, streamed (HF_TOKEN for gated repos)
python3 r2-weights.py publish-dir h3-base/text_encoder_fp8 /workspace/weights/h3-base/text_encoder_fp8 --rev eu-20261012
python3 r2-weights.py verify --deep --record         # re-read every byte: part + file SHA-256; writes verified/…
python3 r2-weights.py compare-dir ltx25 /workspace/weights/ltx25 [--sizes-only]   # the volume copy vs the manifest
python3 r2-weights.py record >> weights-r2.tsv       # then: cd control && node gen-configs.mjs
python3 r2-weights.py plan h3-base --ttl 3600 --out plan.json   # a presigned plan for r2-fetch.py (mode 600)
```

- **publish-hub** lists the repo at the pinned revision (`/api/models/<repo>/tree/<rev>`) and keeps the files
  that match the manifest globs plus `model_index.json`, as `fetch-hub-tree.py` does. It streams each file with
  ranged GETs from the Hub CDN into multipart parts. No local copy is made. Memory is about `--files` ×
  (`--window` + 1) × 64 MiB, 1.3 GB by default.
- **publish-dir** follows symlinks (the HF-cache layout's top-level links) and skips hidden entries
  (`.complete`, `.<x>.partial-*`) and `models--*`. A dest with a Hub row takes that row's globs; any other dest
  takes every file.
- **Idempotent.** A published manifest means nothing is done. A file already uploaded with its parts record is
  skipped (size and SHA-256 checked). Re-run after any failure.
- `--prefix test/<x>/` puts everything under a test prefix (the tests and the staging trial used it).

## 4. Fetch on a pod (`weights_source r2`)

```
pod boot ──POST /ingest/v1/weights-plan {pod}──▶ fv-control          (Bearer: the cluster's ingest token)
         ◀── plan: the pool's trees' manifests + one presigned GET URL per file (TTL 1 h) ──
pod ──GET /ingest/v1/r2-fetch.py──▶ fv-control; checked against FV_R2_FETCH_SHA256 from the pod's env
pod ──N × ranged GET (64 MiB parts) on the presigned URLs──▶ R2   (no R2 key on the pod)
```

- **fv-control** holds a read-only key (secrets `WEIGHTS_R2_ACCESS_KEY_ID` / `WEIGHTS_R2_SECRET_ACCESS_KEY`: an
  **Object Read only** token scoped to `fv-weights`). It reads manifests and presigns URLs (SigV4 query auth,
  WebCrypto, checked against AWS's published example). The plan holds only the trees of the pod's own pool
  (`weightsPlan`), never what the request names. The pod must be a live pod of the token's cluster. Optional vars:
  `WEIGHTS_R2_BUCKET` (default `fv-weights`), `WEIGHTS_R2_ENDPOINT` (default from `CF_ACCOUNT_ID`),
  `WEIGHTS_R2_URL_TTL_S` (3600, clamped to 300-21600), and `WEIGHTS_R2_EXTRA_TREES` (JSON rows like
  `weights-r2.tsv`'s, for tests or as a stopgap until the next deploy).
- **The fetcher** (`r2-fetch.py`, stdlib only; the boot installs `python3` with apt if the image lacks it):
  - Small trees go first. Inside a tree the order is configs and tokenizers, then the text encoder, then
    VAEs and vocoders, then transformer shards.
  - `--concurrency` (default 32, env `FV_R2_CONCURRENCY`) ranged GETs run at once across files. Each is
    streamed with `pwrite` into a file preallocated at its full size under `$FV_WEIGHTS/.r2-partial/<dest>@<rev>/`.
  - Every part is checked against the manifest's SHA-256 as it arrives. A bad or short part is fetched again.
  - Hasher threads re-read each finished file for the whole-file SHA-256 while the rest downloads
    (`--no-full-hash` skips this).
  - Retries use exponential backoff with jitter. A 403 (an expired URL) asks fv-control for a fresh plan.
  - Resume: a partial file from an earlier attempt has its parts hashed on disk, and only the bad ones are fetched.
  - Each tree is renamed into place as soon as it is whole, with `.complete`, `.fv-revision`, `.fv-source`,
    `sha256.txt` and `.fv-manifest.json`.
  - Progress goes to the pod log every 5 s and to fv-control's `/ingest/v1/endpoint` every 15 s (phase
    `weights`, `r2 12.3/150.2 GB, 2450 MB/s, …`; the boot diagnosis shows "fetching from the R2 weights store").
- **GMI / Brev**: `weights_source r2` needs no download approval (it is our own store). It is refused unless every
  tree of the pool is in `weights-r2.tsv` and the key is set. **Unset, it becomes the default** for a pool whose
  trees are all published (`providers.ts weightsSource`). Otherwise it stays `none`, and a pool with real models
  is told why ("models need weights: r2 (… missing), hub …"). `hub` stays available as a fallback.
  - The launch check reads every manifest and shows the sizes ("r2: fetches h3-base 144.0 GB … from the R2
    weights store (manifests checked)").
  - Brev's warm/cold outlook uses the manifests' exact bytes.
  - Keep-on-stop works unchanged: a tree complete at its revision is kept.
- **Runpod**: `weights_source: r2` on a pool fetches onto the container disk instead of mounting the volume.
  - The volume is not mounted. `volume: true` with r2 is refused.
  - `containerDiskInGb` defaults to the trees' bytes × 1.15 + 30 GB, at most 500 GB. Trees larger than that must
    use the volume.
  - The worker boot runs the same fetch step, then the aux files by pinned URL. The pod's id comes from
    `RUNPOD_POD_ID`.
  - The volume path is unchanged and stays the Runpod default until the coordinator removes it. Runpod is still
    limited to the EU region in fv-control (`regions.ts`); lifting that is a follow-up once the volume is gone.
- **Runpod serverless endpoints**: `weights_source: r2` on the endpoint spec (no volume, any data centre, a
  per-endpoint weights token instead of a cluster's ingest token, a cache check that skips the fetch). See §9.
- **Aux files** (TAE decoders, LPIPS) still come from their pinned URLs with SHA-256 (`AUX_FILES`), as with
  `hub`.

## 5. Acceleration: what was evaluated

### 5.1 HTTP/3 / QUIC: not used

- R2's S3 endpoint did not advertise HTTP/3. A ranged GET on a presigned URL returned 206 with no `Alt-Svc`
  header (2026-10-11). That was observed through this environment's HTTPS proxy, so it is **UNVERIFIED** from a
  pod. Check with `curl -sI --http3-only "<presigned url>"` using a curl built with HTTP/3.
- Cloudflare's HTTP/3 setting is per zone and covers client ↔ edge only ("HTTP/3 connection to the origin is not
  yet supported", developers.cloudflare.com/speed/optimization/protocol/http3). So h3 would need a custom domain on
  a zone in front of a **public** bucket.
- Presigned URLs do not work on custom domains (developers.cloudflare.com/r2/api/s3/presigned-urls). Auth there
  would need WAF HMAC token validation (Pro plan or above) or Access.
- The pods' tools have no QUIC: Python's stdlib has none, and Ubuntu's curl (8.5 here) is built with nghttp2 but
  not with an HTTP/3 backend.
- QUIC runs in user space, and one connection is usually CPU-bound well below a datacentre NIC. Its gains are on
  lossy, high-RTT last-mile links, not on bulk transfers between datacentres. Many TCP+TLS streams (§5.2) already
  avoid head-of-line blocking between parts.
- Verdict: no gain worth a public bucket.

### 5.2 Many parallel ranged GETs over TCP + TLS: implemented

- This is the main lever. One stream is bounded by its congestion window and RTT, and N streams add up until the
  NIC, R2 or the client's CPU saturates. The fetcher runs 32 streams by default.
- The client's CPU per GB is about one SHA-256 pass per part, one whole-file pass (optional), TLS decryption
  (AES-GCM, AES-NI) and the copies.
- Measured locally (§8): about 3 user CPU-s per GB on a 4-vCPU VM **without SHA-NI** (SHA-256 at 348 MB/s on one
  core), almost all of it SHA-256. TLS added about 1-2 CPU-s/GB.
- On a GPU node with SHA-NI (Ice Lake and later Xeons, every EPYC since Zen), SHA-256 runs at about 1.5-2 GB/s per
  core (a general figure, not measured here). Python's hashlib, ssl and pwrite release the GIL, so threads use
  several cores.

### 5.3 Other options

| Option | Verdict |
|---|---|
| Cloudflare cache in front of R2 (custom domain) | **No.** The cacheable size limit is 512 MB on Free/Pro/Business and 5 GB on Enterprise, and range requests are checked against the full object size (developers.cloudflare.com/cache/concepts/default-cache-behavior; …/cache/reference/range-requests). Our shards are mostly 2-10 GB. It also needs a public bucket (above) and only helps repeat reads in the same PoP. |
| zstd on bf16 safetensors | **Not now.** Measured on `h3-to-ltx/model.safetensors` (389.5 MB, all BF16): zstd -3 ratio **0.782** (saves 21.8%), -9 0.773, and 0.711 with exponent/mantissa byte-splitting. Decompression ran at about 250 MB/s per thread on this VM, so 25 Gbps needs about 4-12 cores for a 22-29% saving. It would also need per-part frames to keep ranged, verifiable parts. FP8 / NVFP4 trees were not measured (less redundancy expected). Revisit if pods turn out NIC-bound below about 10 Gbps. |
| Fetch only the files a preset needs | **Partly in place.** Trees come per pool (`poolTrees`) and per glob. Bigger win, follow-up: per-preset *file subsets*. For example h3-turbo could fetch `h3-base/transformer_prequant_4step-vsa_mxfp8` (27.3 GB) and `text_encoder_fp8` (26.0 GB) instead of the bf16 `transformer/` and `text_encoder/` of `h3-base` (about 144 GB with the rest). The manifest format supports it; fv-control would need a per-preset file filter, and fv-serve's loader must not need the bf16 files then (**UNVERIFIED**). |
| FP8 trees to halve bytes | Exist for the H3 / LTX-2.5 text encoders and the H3 DiTs (derived, volume-only today, §6). Same follow-up as above. |
| Brev keep-on-stop | **Kept.** A parked VM's `~/workspace/weights` is a local cache. r2 writes the same `.complete` / `.fv-revision`, so warm restarts fetch nothing. |
| Overlap fetch with model load | **Partly.** Small trees and files come first, and each tree is usable as soon as it is renamed in. Starting fv-serve before the transformer shards arrive needs a loader that waits per file: a Rust change, not done here. |

### 5.4 Recommendation and expected numbers

1. Use parallel ranged GETs (32-64 streams, 64 MiB parts), as implemented.
2. Fetch FP8 / prequantized subsets per preset (follow-up).
3. Keep Brev keep-on-stop.
4. Skip HTTP/3, the CDN cache and compression for now.

**Measured (2026-10-12, docs/ops/r2-bench-2026-10-12.md):** three Runpod serverless H100 80GB workers, default
concurrency 32, 0 retries and 0 bad parts each:

| Fetch | GB | s | MB/s |
|---|---:|---:|---:|
| `h3-base` (1 tree) | 144.04 | 288.3 | **500** |
| `fastwan22-ti2v-5b` + `wan22-ti2v-5b`, worker 1 | 58.40 | 76.3 | **766** |
| the same two trees, worker 2 | 58.40 | 177.7 | **329** |

- That is **0.33-0.77 GB/s, median 0.5 GB/s**, not the 1-2.5 GB/s estimated before the run. A 144 GB `h3-base`
  takes about 5 min.
- fv-control now plans with **0.5 GB/s** (`R2_DOWNLOAD_GBPS`). An endpoint's outlook switches to its own last
  measured rate once a boot reports one.
- Not measured yet: a Runpod pod outside the EU (the standalone launch accepts EUR-IS-1 only), Brev (the warm
  restart did not come up), and concurrency 64.
- For comparison, the Hub manages **150-260 MB/s** on an 8-vCPU pod (runpod-volumes.md §5.5), or 10-16 min for the
  same tree.

To measure it on a pod: run `python3 r2-weights.py plan <tree> --out /tmp/plan.json`, then
`python3 r2-fetch.py --root /tmp/w --plan /tmp/plan.json --concurrency 64 --status /tmp/s.json` (the status file has
MB/s). A pod booted with `weights_source r2` reports the same figures to fv-control.

## 6. Migration off the EU volume

### 6.1 What is on the volume, and where each part comes from

**Hub trees: populate Hub → R2 from anywhere.** Use any Linux box with good bandwidth, not this agent container,
whose proxy gave about 25 MB/s. These are the `weights-manifest.tsv` rows whose `weights-revisions.tsv` fetcher is
`hub`, 37 trees (`r2-weights.py status`, kind `hub`):

- H3: `h3-8step`, `h3-base`, `FastH3-4-step-Preview-v1-LoRA`, `upscaler`, `h3-to-ltx`
- LTX: `ltx2`, `ltx25`, `ltx23`, `ltx25-dev`, `ltx25-ic-lora-ingredients`, `ltx23-dev`
- Hunyuan: `hy15-{480,720}-{t2v,i2v}`
- Wan: `fastwan21-1.3b`, `wan22-ti2v-5b`, `fastwan22-ti2v-5b`, `wan21-t2v-14b`, `sfwan21-1.3b`,
  `wan21-t2v-1.3b`, `wan22-t2v-a14b`
- Others: `sana-video-2b-480p`, `lingbot-video-moe-30b-a3b`, `cosmos3-super`
- Upscalers: `auxiliary/upscalers/seedvr2`, `auxiliary/upscalers/flashvsr-v1.1`
- LongLive: `longlive-1.3b`, `longlive2-5b`, `longlive2-5b-nvfp4-s{4,2}`, and the six `longlive-plug/*`

They come to **1554.5 GB**: the glob-matched files of the Hub listing at the pinned revisions, read on
2026-10-11.

- Large downloads need the owner's approval (CLAUDE.md); the 2026-10-11 decision is taken to cover this, but
  confirm.
- Gated repos (Lightricks LTX-2.5, IC-LoRA) need an `HF_TOKEN` whose account accepted the gates.
- Five trees' revisions are *inferred* (`h3-8step`, `FastH3-4-step-Preview-v1-LoRA`, `h3-to-ltx`, `ltx2`,
  `ltx25`). R2 gets the pinned revision. `compare-dir --sizes-only` against the volume (§6.3) shows whether the
  volume holds the same files.

**Volume-only: needs the EU volume mounted once.** These have no plain Hub source, or are derived, or differ
from any Hub file:

| Tree (dest) | Size | Why it is volume-only |
|---|---:|---|
| `h3-base/text_encoder_fp8` | 25.95 GB | derived (`fv-gpucheck quantize-text-encoder`) |
| `ltx25/text_encoder_fp8` | 12.92 GB | derived (same) |
| `h3-base/transformer_prequant_4step-vsa_mxfp8` | 27.29 GB | derived (`quantize-dit`, fast boot A) |
| `h3-base/transformer_prequant_sol-h3_mxfp8` | 23.44 GB | derived (same) |
| `longlive-1.3b-safetensors` | 7.08 GB | converted (`convert-longlive.py`); the EU file hash is per volume |
| `mmaudio-44k-v2` | ~21.46 GB | three repos at `main` (DFN5B revision UNKNOWN) + converted `safetensors/` (no recorded hash) |
| `h3-ref2va` | 69.06 GB | two repos merged. Reproducible from the Hub (`fetch-h3-ref2va.py`), but copying the volume keeps it byte-identical with its `sha256.txt` |
| `auxiliary/tae`, `auxiliary/lpips` | 0.37 GB | pinned URLs (already fetched by pods from those URLs; copy for completeness) |

That is about **188 GB** to copy off the volume.

**On the volume, not copied unless the owner says so** (all are marked "not needed" in runpod-volumes.md §4):

- `FastH3-4-step-Preview-v1-VSA-DataFree`: ~77.97 GB, incomplete
- `weights/wan/`: 12.46 GB, unlisted
- `ltx2`'s second Gemma layout `text_encoder/diffusion_pytorch_model-*`: ~51.61 GB; the loader reads `model-*`
- the older top-level upscaler file: same bytes as the pinned one
- legacy `taeh3/`
- `upstream/`, `runs/`, `.cache/`: ~104 GB, not weights

Each is copied with `publish-dir <name> <dir> --rev eu-<date>` if the owner wants it kept. **The owner must
confirm this list before deletion.**

### 6.2 Commands for the coordinator

**(1) Hub trees → R2.** Run on a Linux VM with Python 3.9+, about 2 GB RAM free and good bandwidth.

```bash
git clone https://github.com/zaitrarrio/fastvideo-rs && cd fastvideo-rs/scripts/gpu
export FV_WEIGHTS_R2_ACCESS_KEY_ID=… FV_WEIGHTS_R2_SECRET_ACCESS_KEY=…   # Object Read & Write, fv-weights only
export HF_TOKEN=…                                                       # gated: Lightricks
for d in $(python3 r2-weights.py status | awk '$2 == "hub" && $4 == "-" {print $1}'); do
  python3 r2-weights.py publish-hub "$d" || echo "FAILED $d" >> failed.txt
done
python3 r2-weights.py verify --deep --record
python3 r2-weights.py record > /tmp/r2-rows.tsv     # append to weights-r2.tsv, node gen-configs.mjs, PR
```

**(2) Volume-only trees → R2, once Runpod is topped up.** Use a CPU pod as in runpod-volumes.md §5.2:
`python:3.12-slim`, `cpu3c` 8 vCPU, the EU volume `jg48s6o1w0` at `/workspace`, a detached DELETE backstop of
about 3 h, and the scripts shipped in the env or cloned. The pod only *reads* the volume.

```bash
W=/workspace/weights; S=eu-$(date -u +%Y%m%d); P="python3 r2-weights.py"
for d in h3-base/text_encoder_fp8 ltx25/text_encoder_fp8 h3-base/transformer_prequant_4step-vsa_mxfp8 \
         h3-base/transformer_prequant_sol-h3_mxfp8 longlive-1.3b-safetensors mmaudio-44k-v2 h3-ref2va \
         auxiliary/tae auxiliary/lpips; do
  $P publish-dir "$d" "$W/$d" --rev "$S" --note "EU volume jg48s6o1w0" || echo "FAILED $d"
done
$P verify --deep --record
# On the same pod, the volume against R2 (reads ~188 GB, then sizes for the Hub trees):
for d in h3-base/text_encoder_fp8 ltx25/text_encoder_fp8 h3-base/transformer_prequant_4step-vsa_mxfp8 \
         h3-base/transformer_prequant_sol-h3_mxfp8 longlive-1.3b-safetensors mmaudio-44k-v2 h3-ref2va \
         auxiliary/tae auxiliary/lpips; do $P compare-dir "$d" "$W/$d"; done
for d in $($P status | awk '$2 == "hub" {print $1}'); do $P compare-dir "$d" "$W/$d" --sizes-only; done
```

- At about 8 vCPU, expect roughly 150-250 MB/s, so about 15-25 min for 188 GB (**estimate**). Pod cost is about
  $0.24/hr.
- The FP8 text-encoder trees hold a `manifest.json` from `fv-gpucheck`, which is uploaded as a normal file.
  `.complete` is not uploaded; pods write their own.

**(3) Verify everything before the volume can be deleted (§7).**

```bash
python3 r2-weights.py status                   # every row "published" except the owner-waived list (§6.1)
python3 r2-weights.py verify --deep --record   # every byte re-read: each part's and file's SHA-256 = manifest
python3 r2-weights.py record | diff - <(grep -v '^#' weights-r2.tsv)   # the repo's record = the bucket
```

- `verify` also checks each manifest SHA-256 against `weights-sha256.tsv` where a sha256 row exists. md5 rows
  (MMAudio) were checked at publish time.
- `compare-dir` (step 2) ties the R2 copies of the volume-only trees to the volume's bytes by SHA-256.

### 6.3 Costs

| Item | Figure |
|---|---|
| R2 storage | $0.015/GB-month, first 10 GB free (developers.cloudflare.com/r2/pricing). Hub trees 1.55 TB + volume-only ~0.19 TB ≈ **1.74 TB → ≈ $26/month** |
| EU volume today | 2000 GB provisioned: **$120/month** (runpod-volumes.md §1). Saving ≈ **$94/month** once deleted |
| Class A (writes) | $4.50/M, 1M/month free. One upload of everything ≈ 1.74 TB / 64 MiB ≈ 26k parts + ~4 ops per file ≈ 35k ops: **inside the free tier** (else ≈ $0.16) |
| Class B (reads) | $0.36/M, 10M/month free. One pod fetching 150 GB ≈ 2.3k ranged GETs: about 4,000 cold boots a month before it costs anything |
| Egress | free (R2 has no egress charge) |
| A full `verify --deep` | ≈ 27k GETs (free tier) plus the bandwidth of the machine running it |

## 7. The deletion gate (coordinator and owner only)

The volume is deleted only when every one of these holds, and nothing here does it:

1. Every Hub tree and every volume-only tree of §6.1 is published: `r2-weights.py status`.
2. `verify --deep --record` passed for every tree. Each has a `verified/<dest>/<rev>/…` record.
3. `compare-dir` passed for every volume-only tree by SHA-256, and for the Hub trees by size. Any size difference
   is explained (an inferred revision) or the tree is copied from the volume with `publish-dir`.
4. `weights-r2.tsv` holds every row, is merged, and fv-control is deployed with it.
5. A GMI or Brev pod booted with `weights_source r2` served a real model, and the measured MB/s is written into §5.4.
6. The owner confirms in writing, including the "not copied" list of §6.1.
7. Then the coordinator deletes the volume and updates CLAUDE.md, runpod-volumes.md and `regions.ts`.

## 8. What was tested (2026-10-11)

- `scripts/gpu/tests/test_r2_weights.py` (19 tests, stdlib `unittest`, against `fake_r2.py`, which checks
  signatures and body hashes like R2):
  - AWS's published SigV4 header and presigned examples
  - publish-dir: manifest, idempotent re-run, resume after one file, an existing object never replaced, a hash
    mismatch aborts before publishing, a recorded md5 checked, R2's body-hash check
  - verify, deep and shallow
  - fetch: whole tree and "present" on re-run, priority order, corrupt parts, cut connections and 500s retried,
    giving up after retries, resume fetching only bad parts, expired URLs refreshed from the plan URL, a stale tree
    replaced, a nested tree placed after its parent, whole-file mismatch, unsafe paths
- `control/test/unit/r2weights.test.ts` (13 tests):
  - presigning (the AWS vector, key encoding), manifests pinned by SHA-256, the plan
  - the r2 default and the launch checks, the GMI/Brev env, the Runpod payload (no volume, disk size, boot)
  - `R2_FETCH_SH` run in bash with the real `r2-fetch.py` against a fake R2 that verifies every presigned URL
    with node:crypto
  - a tampered fetcher refused
- `control/test/unit/serverless-r2.test.ts` (19 tests): the serverless spec (default, placement, disk, refusals),
  the payloads (bash boot, token, no volume, no R2 key), the weights token (wrong, another endpoint's, a volume
  endpoint's, rotation with grace, revocation by delete), the boot reports and the outlook, and `SLS_R2_SH` run in
  bash with the real fetcher and fv-control's real handlers: a cold boot, a cache hit that asks for nothing, a stale
  revision fetched again, a wrong token failing loudly, the add-only cache volume.
- `scripts/gpu/tests/test_r2_weights.py` still passes with `--add-only` added to the fetcher.
- Integration (`node test/integration/run.mjs`, step "weights_source r2"):
  - a GMI standalone pod defaults to r2 for a published tree
  - the launch check reads the manifest
  - the pod fetches the plan and every part through fv-control's presigned URLs (verified by the mock)
  - no R2 credential on the pod
  - plan auth: a bad token gets 401, another pod 404, a deleted pod 404
  - a forged URL gets 403
  - the served fetcher matches its pinned SHA-256
- Real R2, staging bucket `fv-weights-staging`, prefix `test/r2-agent/`:
  - `publish-hub h3-to-ltx` streamed 0.39 GB Hub → R2. Its SHA-256 equalled the Hub LFS one (17 s, ~25 MB/s
    through this environment's proxy).
  - `verify --deep` passed. A re-run did nothing (2 class B ops).
  - `plan` + `r2-fetch.py` pulled it back to local disk with every hash checked (86 MB/s, proxy-bound).
  - `If-None-Match: *` was honoured (412).
  - All of those test objects were then deleted, and the bucket is empty.
- Local throughput (`r2-fetch.py`, 2.15 GB tree on tmpfs, a Node range server on the same 4-vCPU VM, no SHA-NI;
  the server and the fetcher share the cores, so these are lower bounds):

  | Setup | Throughput | Fetcher CPU |
  |---|---|---|
  | plain HTTP, 16 streams, `--no-full-hash` | 0.37-0.56 GB/s | ~6.4 user CPU-s per 2.15 GB, ≈ the SHA-256 cost |
  | TLS (local self-signed), 16-32 streams | 0.16-0.23 GB/s | — |
  | with the whole-file re-hash | 0.22-0.24 GB/s | — |

  System time varied 2-25 s between identical runs (VM noise).

## 9. Serverless endpoints (`weights_source: r2`)

How it works for users, the API and the token: docs/control/serverless.md §1b. This section covers the caching
design and how to benchmark it.

### 9.1 What changes against the volume

- **No network volume**, so there is no `EUR-IS-1`-only restriction: any GPU type in any data centre with stock.
- Every scale-from-zero worker on a fresh host fetches its trees before fv-serve starts. That is about 144 GB for
  H3 (`h3-base`), 125 GB for LTX-2.5 and about 60 GB for Wan 2.2 5B (sizes from brev-park.ts `TREE_GB` until the
  manifests are published).
- At the measured 0.33-0.77 GB/s (§5.4) the H3 fetch adds about 5 min and the Wan fetch 1.3-3 min to every
  scale-from-zero worker on a fresh host. Measured volume cold starts, for comparison: wan 49 + 70 s, h3 80 + 52 s,
  ltx 79 + 33 s (docs/serve/gateway.md §8.6). The comparison is in docs/ops/r2-bench-2026-10-12.md.
- The container disk is sized from the manifests: the bytes × 1.15 + 30 GB (H3 196 GB). Runpod's maximum
  container disk for serverless is **UNVERIFIED**; fv-control allows up to 500 GB.

### 9.2 Auth

The worker has no cluster and no ingest token. Each r2 endpoint gets its own **weights token** (`fvw_…`, in the
template env as `FV_WEIGHTS_TOKEN`). D1 keeps only its SHA-256 (`weights_token_hash`), as clusters keep
`ingest_hash`.

- `POST /ingest/v1/weights-plan` with that token answers the plan of **that endpoint's trees only** (from its
  stored spec). The `fvw_` prefix routes it; a cluster token still works for its pods.
- `POST /ingest/v1/sls-report` takes the same token for the boot's reports.
- A template update rotates the token. The previous one is honoured for 30 min (the rollout).
- A delete revokes both. The R2 key never leaves fv-control.
- Tested: no token, a wrong one, another endpoint named (403), a volume endpoint's token (409), rotation with
  grace, revocation by delete.

### 9.3 Caching across cold starts (what Runpod supports, checked 2026-10-11)

| Mechanism | What Runpod documents | Fit |
|---|---|---|
| Container disk | "Temporary storage that exists only while a worker is running … lost when the worker stops or scales down" (docs.runpod.io/serverless/storage/overview) | A restarted container on the same worker keeps it: the cache check makes that boot a **hit**. |
| FlashBoot | "Reduces cold starts by retaining worker state after spin-down, allowing faster 'revival' than fresh boots" (docs.runpod.io/serverless/endpoints/endpoint-configurations) | **Measured 2026-10-12 (docs/ops/r2-bench-2026-10-12.md): a hit.** About 12 min after its last job, an r2 worker was revived with its process and its model still resident. There was no new container, no boot report and no fetch, and the job waited 0.9 s. A container restart on the same worker also kept the disk: the boot found every tree, skipped the fetch, and reached fv-serve in 3 s. How long FlashBoot keeps a worker is **UNVERIFIED**, and a new worker always fetches. |
| Cached models | Hugging Face only, "one cached model at a time" per endpoint, mounted at `/runpod-volume/huggingface-cache/hub/`, workers placed on hosts that have it, download time not billed (docs.runpod.io/serverless/endpoints/model-caching) | **Not as is.** There is no S3/R2 source, and one repo per endpoint does not cover a family's several trees. It could work with one private HF bundle repo per preset (a large new upload: owner approval). Worth evaluating for H3 after the R2 numbers. |
| Network volume as a cache | $0.07/GB-month; one volume pins the endpoint to its DC; several volumes from different DCs can be attached, one per DC, with no sync between them (docs.runpod.io/serverless/storage/network-volumes) | Implemented, **off by default** (`FV_SLS_R2_CACHE_VOLUMES`, one volume per endpoint). Filled add-only from R2 (`r2-fetch.py --add-only`). It costs storage per DC and brings back the DC pin and the volume's read speed. |

What is implemented:

1. **Skip when present.** The boot checks every tree for `.complete` and `.fv-revision` = the pinned revision
   (`FV_R2_TREE_REVS`, from `weights-r2.tsv`) before it asks fv-control for anything. The fetcher writes those
   markers as brev-park's trees do. A tree at another revision is fetched again.
2. **Resume.** A partial fetch (a crashed container) is resumed part by part (§4).
3. **Disk sized from the manifests**, so a fetch never fails half-way for space.
4. **Optional cache volume** (above), with the add-only rule enforced in the fetcher:
   - per-worker staging names;
   - "another worker finished first" keeps theirs;
   - a different revision is never replaced.

### 9.4 The cache volume (off by default)

The owner creates the volume (CLAUDE.md: never by an agent) in the chosen DC, then sets the Worker var
`FV_SLS_R2_CACHE_VOLUMES='{"<id>":"<DC>"}'`. An r2 endpoint with `network_volume: "<id>"` then runs in that DC.
Its workers fill `/runpod-volume/fv-r2-cache/<dest>` once and load from there afterwards.

Cost per DC: the trees' GB × $0.07/month (H3 + Wan + LTX ≈ 330 GB ≈ $23/month).

Enable it only if §9.5 shows the R2 fetch is slower than a volume load for a family.

### 9.5 How to benchmark (coordinator)

Preconditions:

- `h3-base`, `fastwan22-ti2v-5b` and `wan22-ti2v-5b` are published and recorded in `weights-r2.tsv`
  (`r2-weights.py status`).
- fv-control is deployed with them. `GET /api/serverless/presets` shows `r2.missing: []` for `h3-turbo` and `wan`.
- The balance is above the floor plus margin.

Launch one endpoint per family, `workers_max` 1, FlashBoot on (to test the revival), and a 60 min backstop:

```bash
for p in h3-turbo wan; do
  fv-control.sh endpoint create - <<EOF
{"name": "bench-${p%%-*}-r2", "preset": "$p", "weights_source": "r2",
 "gpu_types": ["NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA RTX PRO 6000 Blackwell Server Edition"],
 "workers_min": 0, "workers_max": 1, "idle_timeout_s": 5, "flashboot": true, "deadline_min": 60, "deadline_action": "delete"}
EOF
done
```

For each endpoint (`bench-h3-r2`, `bench-wan-r2`):

1. **Cold from zero.** `fv-control.sh endpoint invoke bench-h3-r2` (an info job; `--async` if it outlasts /runsync,
   then `endpoint show` until done).
2. Wait until the worker is gone: `fv-control.sh endpoint show bench-h3-r2 | jq '.runpod.workers'` is empty, then
   30 s more.
3. **Second cold start** (revival or a new worker): `fv-control.sh endpoint invoke bench-h3-r2`.
4. Record, from `fv-control.sh endpoint show bench-h3-r2` (`boots`, `jobs`, `outlook`) and the worker's log (Log
   explorer, pod = the worker id; Runpod's system log for the image pull):
   - per boot: worker id, GPU type, DC, `cache` (hit / miss), GB fetched, `fetch_s`, MB/s, and container start
     → fv-serve start (`boot_to_serve_s`);
   - per job: `delay_ms` (the cold start as Runpod counts it), `exec_ms`, and the image pull time from the system
     log;
   - the load: fv-serve's ready time minus the `serve` report;
   - whether the second start was a **hit** (FlashBoot kept the disk) or a **miss** (a new worker or the disk
     lost).
5. `fv-control.sh endpoint delete bench-h3-r2` (the backstop deletes it anyway after 60 min).

Expected cost: two cold starts and two short jobs per endpoint, about 10-15 GPU-minutes each, about $1-2 per
endpoint at H100 flex rates.

Then:

- Write the measured MB/s into §5.4 and `R2_DOWNLOAD_GBPS`.
- Write the breakdown into docs/serve/gateway.md §8.6.
- Flip `DEFAULT_SLS_WEIGHTS_SOURCE` to `"r2"` in `control/src/serverless/spec.ts` if r2 is not slower end to end
  than the volume.

## 10. The copy off the EU volume (2026-10-11/12)

The copy ran on CPU pods in EUR-IS-1 with `jg48s6o1w0` mounted at `/workspace`, which the pods only read
(`scripts/gpu/r2-copy-volume.sh` launches the pod, `r2-copy-volume-pod.sh` runs on it). Nothing on the volume was
written, renamed or deleted, and temp files went to the pod's container disk.

**What went where.**

- **37 Hub trees** (`source hub-via-dir`) are stored at their pinned Hub revision. `publish-dir` takes the file list
  from the Hub listing at that revision, not from the directory. It reads each file from the volume
  (`models--*/snapshots/<rev>` first, then the plain path), and checks each file before its upload completes:
  - LFS files against the Hub's SHA-256;
  - small files against the git blob SHA-1;
  - files that have a row in `weights-sha256.tsv` against that row as well.
- **9 volume-only trees** (`source dir`) are stored at rev `eu-20261011`. They are checked against the repo's
  recorded hashes (`weights-sha256.tsv`, `verify-weights.sh`'s FP8 and prequant lists, and the `auxiliary/` URL
  rows).
- **Archive, with no manifest and no preset wiring:**
  - `/workspace/runs` → `archive/eu-volume/runs/`: 10598 files, 32.51 GB.
  - `/workspace/weights/wan` → `archive/eu-volume/wan/`: 8 files, 12.46 GB.
  - Each archive also has `archive/eu-volume/<name>.index.json`, which lists every file's size and SHA-256.
- **Not copied:**
  - `FastH3-4-step-Preview-v1-VSA-DataFree`, `ltx2`'s second Gemma layout, the top-level upscaler copy, `taeh3/`,
    `upstream/` and `.cache/`, all as §6.1 says.
  - The volume root also holds directories that §6.1 does not list, and these were not copied either:
    `fastboot-logs`, `fv-libs`, `gpucheck-out`, `h3-text-cache`, `hf` (a Hub token) and `scratch`.

**Files with no recorded hash.** For these files the SHA-256 in the manifest was taken from the volume copy itself,
and nothing independent checks it:

- `h3-base/text_encoder_fp8/manifest.json`;
- `h3-ref2va/SOURCE.json` and `h3-ref2va/sha256.txt`;
- `longlive-1.3b-safetensors/keys-longlive_base.txt`, `keys-lora.txt` and `sha256.txt`;
- 17 files in `mmaudio-44k-v2`: the 9 `DFN5B-CLIP-ViT-H-14-384/*` files, `MANIFEST.json`, the 2 `bigvgan` files and
  the 5 `safetensors/*` files. Its 3 md5 rows matched.

**Per tree.** compare-dir compared the volume with R2:

- for the volume-only trees, by the SHA-256 of every 64 MiB part;
- for the Hub trees, by size only, because their bytes had already been checked against the Hub at upload.

| Tree | Rev | Files | GB | Source | Hash checks at upload | compare-dir vs volume |
|---|---|---:|---:|---|---|---|
| `FastH3-4-step-Preview-v1-LoRA` | `f509e629374c` | 2 | 6.82 | hub-via-dir | hub-lfs-sha256 2 | 2/2 files equal (sizes; 6.82 GB, 0s) |
| `auxiliary/lpips` | `eu-20261011` | 2 | 0.24 | dir | repo-sha256 2 | 2/2 files equal (sha256 of every part; 0.24 GB, 0s) |
| `auxiliary/tae` | `eu-20261011` | 4 | 0.13 | dir | repo-sha256 4 | 4/4 files equal (sha256 of every part; 0.13 GB, 0s) |
| `auxiliary/upscalers/flashvsr-v1.1` | `27561b186ded` | 6 | 6.95 | hub-via-dir | hub-git-blob-sha1 2, hub-lfs-sha256 4, repo-sha256 6 | 6/6 files equal (sizes; 6.95 GB, 0s) |
| `auxiliary/upscalers/seedvr2` | `09ced7102363` | 2 | 7.28 | hub-via-dir | hub-lfs-sha256 2, repo-sha256 2 | 2/2 files equal (sizes; 7.28 GB, 0s) |
| `cosmos3-super` | `f543c56225b2` | 40 | 129.47 | hub-via-dir | hub-git-blob-sha1 11, hub-lfs-sha256 29, repo-sha256 40 | 40/40 files equal (sizes; 129.47 GB, 0s) |
| `fastwan21-1.3b` | `25e7ed7f41fd` | 16 | 29.21 | hub-via-dir | hub-git-blob-sha1 7, hub-lfs-sha256 9 | 16/16 files equal (sizes; 29.21 GB, 0s) |
| `fastwan22-ti2v-5b` | `3e187042a324` | 15 | 24.20 | hub-via-dir | hub-git-blob-sha1 8, hub-lfs-sha256 7, repo-sha256 15 | 15/15 files equal (sizes; 24.20 GB, 0s) |
| `h3-8step` | `3da2ddfe1954` | 49 | 147.86 | hub-via-dir | hub-git-blob-sha1 17, hub-lfs-sha256 32 | 49/49 files equal (sizes; 147.86 GB, 0s) |
| `h3-base` | `42ed227ee7df` | 50 | 144.04 | hub-via-dir | hub-git-blob-sha1 18, hub-lfs-sha256 32 | 50/50 files equal (sizes; 144.04 GB, 0s) |
| `h3-base/text_encoder_fp8` | `eu-20261011` | 2 | 25.95 | dir | none 1, repo-sha256 1 | 2/2 files equal (sha256 of every part; 25.95 GB, 58s) |
| `h3-base/transformer_prequant_4step-vsa_mxfp8` | `eu-20261011` | 2 | 27.29 | dir | repo-sha256 2 | 2/2 files equal (sha256 of every part; 27.29 GB, 97s) |
| `h3-base/transformer_prequant_sol-h3_mxfp8` | `eu-20261011` | 2 | 23.44 | dir | repo-sha256 2 | 2/2 files equal (sha256 of every part; 23.44 GB, 31s) |
| `h3-ref2va` | `eu-20261011` | 31 | 69.06 | dir | none 2, repo-sha256 29 | 31/31 files equal (sha256 of every part; 69.06 GB, 83s) |
| `h3-to-ltx` | `1792c42689a0` | 2 | 0.39 | hub-via-dir | hub-git-blob-sha1 1, hub-lfs-sha256 1 | 2/2 files equal (sizes; 0.39 GB, 0s) |
| `hy15-480-i2v` | `854c04a4c8a5` | 25 | 33.78 | hub-via-dir | hub-git-blob-sha1 17, hub-lfs-sha256 8 | 25/25 files equal (sizes; 33.78 GB, 0s) |
| `hy15-480-t2v` | `286be7ce7227` | 32 | 53.38 | hub-via-dir | hub-git-blob-sha1 18, hub-lfs-sha256 14 | 32/32 files equal (sizes; 53.38 GB, 0s) |
| `hy15-720-i2v` | `a1d10cf3af79` | 27 | 53.38 | hub-via-dir | hub-git-blob-sha1 17, hub-lfs-sha256 10 | 27/27 files equal (sizes; 53.38 GB, 0s) |
| `hy15-720-t2v` | `f4dbc4a1efa4` | 32 | 53.38 | hub-via-dir | hub-git-blob-sha1 18, hub-lfs-sha256 14 | 32/32 files equal (sizes; 53.38 GB, 0s) |
| `lingbot-video-moe-30b-a3b` | `f2e538f64afe` | 59 | 129.95 | hub-via-dir | hub-git-blob-sha1 30, hub-lfs-sha256 29, repo-sha256 59 | 59/59 files equal (sizes; 129.95 GB, 0s) |
| `longlive-1.3b` | `cda9138d9387` | 4 | 8.48 | hub-via-dir | hub-git-blob-sha1 2, hub-lfs-sha256 2, repo-sha256 4 | 4/4 files equal (sizes; 8.48 GB, 0s) |
| `longlive-1.3b-safetensors` | `eu-20261011` | 5 | 7.08 | dir | none 3, repo-sha256 2 | 5/5 files equal (sha256 of every part; 7.08 GB, 12s) |
| `longlive-plug/minimax-h3-cfg` | `de1f4e8c7310` | 11 | 2.77 | hub-via-dir | hub-git-blob-sha1 10, hub-lfs-sha256 1, repo-sha256 11 | 11/11 files equal (sizes; 2.77 GB, 0s) |
| `longlive-plug/minimax-h3-few-step` | `b3686f432d3f` | 55 | 2.77 | hub-via-dir | hub-git-blob-sha1 54, hub-lfs-sha256 1, repo-sha256 55 | 55/55 files equal (sizes; 2.77 GB, 0s) |
| `longlive-plug/wan21-t2v-14b-cfg` | `32b8aa3d1db3` | 8 | 2.45 | hub-via-dir | hub-git-blob-sha1 7, hub-lfs-sha256 1, repo-sha256 8 | 8/8 files equal (sizes; 2.45 GB, 0s) |
| `longlive-plug/wan21-t2v-14b-few-step` | `f125af0533b9` | 5 | 1.23 | hub-via-dir | hub-git-blob-sha1 4, hub-lfs-sha256 1, repo-sha256 5 | 5/5 files equal (sizes; 1.23 GB, 0s) |
| `longlive-plug/wan22-ti2v-5b-cfg` | `fa1f9280778a` | 8 | 0.64 | hub-via-dir | hub-git-blob-sha1 7, hub-lfs-sha256 1, repo-sha256 8 | 8/8 files equal (sizes; 0.64 GB, 0s) |
| `longlive-plug/wan22-ti2v-5b-few-step` | `38a6ec4e8f5b` | 7 | 1.29 | hub-via-dir | hub-git-blob-sha1 6, hub-lfs-sha256 1, repo-sha256 7 | 7/7 files equal (sizes; 1.29 GB, 0s) |
| `longlive2-5b` | `8521079b8637` | 2 | 10.00 | hub-via-dir | hub-git-blob-sha1 1, hub-lfs-sha256 1, repo-sha256 2 | 2/2 files equal (sizes; 10.00 GB, 0s) |
| `longlive2-5b-nvfp4-s2` | `9ab6c9c70db9` | 2 | 2.95 | hub-via-dir | hub-git-blob-sha1 1, hub-lfs-sha256 1, repo-sha256 2 | 2/2 files equal (sizes; 2.95 GB, 0s) |
| `longlive2-5b-nvfp4-s4` | `427ffb75f0a3` | 2 | 2.95 | hub-via-dir | hub-git-blob-sha1 1, hub-lfs-sha256 1, repo-sha256 2 | 2/2 files equal (sizes; 2.95 GB, 0s) |
| `ltx2` | `dfcc2108383f` | 31 | 94.74 | hub-via-dir | hub-git-blob-sha1 14, hub-lfs-sha256 17 | 31/31 files equal (sizes; 94.74 GB, 0s) |
| `ltx23` | `22b09fb1860a` | 40 | 71.60 | hub-via-dir | hub-git-blob-sha1 25, hub-lfs-sha256 15 | 40/40 files equal (sizes; 71.60 GB, 0s) |
| `ltx23-dev` | `3c6a4e66e5d0` | 3 | 54.75 | hub-via-dir | hub-lfs-sha256 3, repo-sha256 3 | 3/3 files equal (sizes; 54.75 GB, 0s) |
| `ltx25` | `426936f8b22d` | 40 | 125.12 | hub-via-dir | hub-git-blob-sha1 14, hub-lfs-sha256 26 | 40/40 files equal (sizes; 125.12 GB, 0s) |
| `ltx25-dev` | `426936f8b22d` | 7 | 37.98 | hub-via-dir | hub-git-blob-sha1 3, hub-lfs-sha256 4 | 7/7 files equal (sizes; 37.98 GB, 0s) |
| `ltx25-ic-lora-ingredients` | `12040e4091ac` | 2 | 1.31 | hub-via-dir | hub-git-blob-sha1 1, hub-lfs-sha256 1, repo-sha256 1 | 2/2 files equal (sizes; 1.31 GB, 0s) |
| `ltx25/text_encoder_fp8` | `eu-20261011` | 2 | 12.92 | dir | repo-sha256 2 | 2/2 files equal (sha256 of every part; 12.92 GB, 33s) |
| `mmaudio-44k-v2` | `eu-20261011` | 20 | 21.46 | dir | none 17, repo-md5 3 | 20/20 files equal (sha256 of every part; 21.46 GB, 55s) |
| `sana-video-2b-480p` | `db5f398b13ca` | 17 | 14.00 | hub-via-dir | hub-git-blob-sha1 10, hub-lfs-sha256 7, repo-sha256 17 | 17/17 files equal (sizes; 14.00 GB, 0s) |
| `sfwan21-1.3b` | `4b44356635ae` | 17 | 28.93 | hub-via-dir | hub-git-blob-sha1 8, hub-lfs-sha256 9 | 17/17 files equal (sizes; 28.93 GB, 0s) |
| `upscaler` | `3f941d5d1820` | 1 | 0.69 | hub-via-dir | hub-lfs-sha256 1 | 1/1 files equal (sizes; 0.69 GB, 0s) |
| `wan21-t2v-1.3b` | `0fad780a534b` | 19 | 28.93 | hub-via-dir | hub-git-blob-sha1 9, hub-lfs-sha256 10, repo-sha256 19 | 19/19 files equal (sizes; 28.93 GB, 0s) |
| `wan21-t2v-14b` | `38ec498cb320` | 29 | 80.41 | hub-via-dir | hub-git-blob-sha1 9, hub-lfs-sha256 20 | 29/29 files equal (sizes; 80.41 GB, 0s) |
| `wan22-t2v-a14b` | `5be7df9619b5` | 41 | 126.20 | hub-via-dir | hub-git-blob-sha1 11, hub-lfs-sha256 30, repo-sha256 41 | 41/41 files equal (sizes; 126.20 GB, 0s) |
| `wan22-ti2v-5b` | `b8fff7315c76` | 20 | 34.20 | hub-via-dir | hub-git-blob-sha1 9, hub-lfs-sha256 11 | 20/20 files equal (sizes; 34.20 GB, 0s) |

**Verify.** `verify --deep --record` read every part back from R2 with the read-only key pair and checked the
SHA-256 of every 64 MiB part against the manifest. Each tree has a `verified/<dest>/<rev>/<stamp>.json` record.
The result: **46 of 46 trees OK, 0 failures.**

- 14 trees were verified on the first two verify pods with whole-file hashing, which the record calls "sha256 of
  every part and file".
- The other 32 were verified by parts, at 30-560 MB/s per tree.

**Findings.**

- **Gated repos need a token for the listing, not only for the download.** For Lightricks `ltx25`, `ltx25-dev`
  and `ltx25-ic-lora-ingredients`, an anonymous tree listing returns LFS oids that are not the files' SHA-256. With
  an `HF_TOKEN` whose account accepted the gate, the oids are the real hashes and match the volume.
  `r2-copy-volume-pod.sh` takes the token from the volume's `hf/token`.
- **Dot files.** A Hub listing can include `.gitattributes`, so `publish-dir` keeps the dot files that the listing
  names.
- **`runs/` is 32.51 GB in 10598 files**, not the ~1.8 GB that runpod-volumes.md §4 estimated.
- **R2 read latency has a long tail.** Single ranged GETs took from 1.6 s to 26 s. Hashing each file in order then
  waits on its slowest part (a single-stream compare-dir ran at about 10 MB/s). `compare-dir` and `verify --deep` therefore hash parts in
  parallel across the whole tree, and run at 80-550 MB/s per tree from a 16-vCPU pod.
  `FV_VERIFY_WHOLE_FILE=1` keeps the old whole-file hashing.
- **A pod deleted mid-upload leaves objects without a parts record.** `publish` now adopts such an object only if
  its bytes in R2 equal the source bytes (3 files were adopted in `runs/`).

**Throughput and cost.**

- Publish volume → R2 on a 16-vCPU `cpu3c` pod (3 trees at a time, 64 MiB parts):
  - about 1.6 TB in 61 min (22:27–23:28 UTC), **≈ 0.44 GB/s** on average;
  - peaks of **≈ 0.82 GB/s** while the large trees ran.
- The copy took seven pods, one after another and never two at once: publish, the four trees that failed, the
  `runs/` archive, compare, then verify on three pods. Each pod cost $0.48/hr and had a DELETE backstop of at most
  4 h.
- The whole copy took 22:27 to 04:35 UTC, about 6.1 h of pod time, so ≈ $2.93 at $0.48/hr.
- The Runpod balance went from $35.38 to $31.32. That figure includes the volume's own hourly charge.
- The bucket now holds 1787.0 GB: 1742.04 GB of trees plus 44.97 GB of archive. At $0.015/GB-month, that is
  ≈ $27/month.
