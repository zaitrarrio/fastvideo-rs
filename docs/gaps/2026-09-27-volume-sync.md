# Weight volume sync: EU vs US survey

Date: 2026-09-27
Compares: the two 1000 GB Runpod network volumes that hold the model weights.

- EU: `fv-weights-h3-ltx-hy`, id `jg48s6o1w0`, EUR-IS-1.
- US: `fv-weights-b200-us`, id `s2k01690bi`, US-CA-2.

Method: one CPU pod per volume (`python:3.12-slim`, cpu3c, $0.12/hr, ids
`7icep1blfqhotg` EU and `3n9lj67tw3aa5o` US, about 2 minutes each, both
deleted). Each pod mounted its volume at `/workspace` and wrote only to its
container disk. It ran `find /workspace -printf '%y %s %k %T@ %P %l'` (every
entry, 68 316 on EU and 2 535 on US), `du -sb` / `du -sk` per top-level
directory and per weight tree, and a `.complete` check. It also computed a
SHA-256 of every weight file of 1 MiB or less. The results were served on port
8000 through the Runpod proxy. **Nothing was copied, moved, deleted or
resized.** The GB figures below are 10^9 bytes, the unit Runpod bills in. The
US mount reports 1 000 000 716 800 bytes of capacity.

## Headline

| | EU | US |
|---|---:|---:|
| Used (sum of file bytes) | **1017.32 GB** (full) | **733.61 GB** |
| Weight trees | 15 (one is a 0-byte stub) | 12 |
| Non-weight data | 103.6 GB (`upstream/`, `runs/`, `.cache/`) | 1.8 GB (`runs/`) |
| Stale or partial data inside `weights/` | 18.8 GB partial `.chunked.part` (with the duplicated rest of that tree, 78.0 GB) | 21.46 GB `*.incomplete` |
| Must receive to hold every manifest tree | **149.90 GB** | **223.14 GB** |

The union of every tree in `scripts/gpu/weights-manifest.tsv`, with stale
partials left out, is **933.50 GB**. It fits in 1000 GB with 66.5 GB spare
on each volume, **but only if EU drops 233.7 GB** of non-weight, unlisted
and partial data. Keeping everything that is on EU today as well needs
about 1170 GB on EU and about 1065 GB on US.

## Top-level inventory

`du -sb` bytes (GB). `/workspace/hf`, `fv-libs`, `gpucheck-out` and
`h3-text-cache` are each under 1 MB on both volumes.

| Path | EU | US | Kind |
|---|---:|---:|---|
| `weights/` | 913.20 | 731.84 | weights (trees below) |
| `upstream/` | 66.71 | — | not weights. Upstream venvs and derived weights from before `UP_LOCAL=1`: `upstream/weights/MiniMax-H3` 66.29, `src` 0.35, `bin` 0.05 |
| `runs/` | 32.51 | 1.78 | not weights. Old run outputs. EU: `rtx5090` 17.24, `rtx6000` 10.29, `fastvideo` 4.20, `h3` 0.66, `ltx` 0.10, `wan` 0.02. US: `b200` 1.78 |
| `.cache/` | 4.90 | — | not weights (`.cache/uv`) |

Since `26c46d8` / `161fe04` every writer (`runpod-http.sh`, `pod.sh
UP_LOCAL=1`) keeps its outputs on the container disk. With the default
(`UP_LOCAL=1`), nothing reads `upstream/`, `runs/` or `.cache/` on the volume. `pod.sh` with
`UP_LOCAL=0` would still use `/workspace/upstream`.

## Weight trees

Bytes are regular-file bytes. HF-cache trees are counted once: the
`snapshots/` symlinks point into `blobs/`. "Marker" means a `.complete`
file exists.

| Tree | In manifest | EU GB | EU marker | US GB | US marker | State |
|---|---|---:|---|---:|---|---|
| h3-8step | yes | 147.85 | yes | 147.85 | yes | same: 63 files, sizes equal, symlinks equal |
| h3-base | yes | 144.03 | yes | 144.03 | yes | same |
| ltx25 | yes | 125.12 | yes | 125.12 | yes | same |
| ltx2 | yes | **146.35** | yes | 94.74 | yes | EU has 12 more blobs (51.61 GB): `text_encoder/diffusion_pytorch_model-000{01..12}-of-00012.safetensors`. That is a second copy of the Gemma text encoder, in a layout the manifest glob (`text_encoder/model-*`) does not take. The rest is identical. |
| ltx23 | yes | 40.84 | yes | **47.18** | yes | US has `text_embedding_projection/model.safetensors` (6.34 GB). The manifest lists `text_embedding_projection/*`, so **EU's tree is incomplete despite its marker**. Neither side has the manifest's `ltx-2.3-22b-distilled-lora-384*.safetensors` or `latent_upsampler/`. |
| FastH3-4-step-Preview-v1-LoRA | yes | 6.82 | yes | 6.82 | yes | Same bytes, different layout. EU keeps `vsa-datafree/adapter_model.safetensors` as a plain file (the curl fallback). US keeps it as an HF blob plus symlink. Sizes are equal (5 339 117 712). Content was not hashed (the file is over 1 MiB). |
| upscaler | yes | 0.69 | yes | 0.69 | yes | Same size, different layout (EU plain file `minimax_h3_latent_upscaler_3d_bf16.safetensors` plus fetch logs; US HF cache) |
| h3-to-ltx | yes | 0.39 | yes | 0.39 | yes | same (only the `.complete` / `.fetch.pid` contents differ) |
| hy15-480-t2v | yes | 53.38 | yes | — | | EU only |
| hy15-480-i2v | yes | 33.78 | yes | — | | EU only |
| hy15-720-t2v | yes | 53.38 | yes | — | | EU only |
| hy15-720-i2v | yes | 53.38 | yes | — | | EU only |
| fastwan21-1.3b | yes | 29.21 | yes | — | | EU only |
| wan22-ti2v-5b | yes | **0 (stub)** | **no** | 34.20 | yes | EU has 26 zero-byte files (HF `.lock`s, 3 empty `.incomplete`s, `.fvfetch-owner`), du 7 KiB. That is an aborted fetch. |
| wan21-t2v-14b | yes | — | | 101.87 (80.41 clean) | yes | US only. It carries **21.46 GB of stale `.cache/huggingface/download/*.incomplete`**: 16 files, 7 non-empty, text-encoder and transformer shards from an earlier download attempt. The loaded tree is 80.41 GB. |
| sfwan21-1.3b | yes | — | | 28.93 | yes | US only |
| taeh3 | **no** | — | | 0.02 | yes | US only. Not in the manifest (`fetch-taeh3.sh` puts it on the container disk). |
| FastH3-4-step-Preview-v1-VSA-DataFree | **no** | 77.97 | yes (**wrong**) | — | | EU only. Not in the manifest, and **incomplete despite its marker**: `snapshots/…/transformer/` is empty, and the transformer exists only as 4 `blobs/*.chunked.part` files (18.80 GB) with `.state` files. The rest is 12 text-encoder blobs and the audio VAE (59.17 GB), whose SHA-256 names equal blobs already in `h3-8step` / `h3-base`, so it duplicates their data. The adapter is a symlink into the LoRA tree. No cell or script refers to this tree (`verify-weights.sh` has no entry for it). |

Small-file SHA-256s (≤ 1 MiB) match between the two copies of every shared
tree. The only mismatches are the `.complete` and `.fetch.pid` bookkeeping
files.

## Diff

**Only on EU** (weights): hy15-480-t2v, hy15-480-i2v, hy15-720-t2v,
hy15-720-i2v, fastwan21-1.3b (all in the manifest, 223.14 GB together).
FastH3-4-step-Preview-v1-VSA-DataFree (unlisted, partial, 77.97 GB). The
extra ltx2 text-encoder layout (unlisted, 51.61 GB).

**Only on US** (weights): wan21-t2v-14b (80.41 GB clean), sfwan21-1.3b
(28.93), wan22-ti2v-5b (34.20; EU has a stub), ltx23
`text_embedding_projection` (6.34), taeh3 (0.02, unlisted).

**Differ in layout only**: the FastH3 LoRA's `vsa-datafree` adapter and
upscaler (same sizes).

**Bytes each side must receive** to hold every manifest tree:

| To | Items | GB |
|---|---|---:|
| EU | wan21-t2v-14b (clean) 80.41, wan22-ti2v-5b 34.20, sfwan21-1.3b 28.93, ltx23 `text_embedding_projection` 6.34, taeh3 0.02 | **149.90** |
| US | hy15 x4 193.93, fastwan21-1.3b 29.21 | **223.14** |
| US, optional | ltx2 extra text-encoder layout 51.61; VSA-DataFree (unique bytes are only the 18.80 GB partial transformer) | +51.61 / +18.80 |

## Removal candidates (named; nothing done)

| Volume | Item | GB | Why |
|---|---|---:|---|
| EU | `upstream/` | 66.71 | not weights. Pre-`UP_LOCAL` venvs and `upstream/weights/MiniMax-H3`; `pod.sh` now builds under `/root` |
| EU | `runs/` | 32.51 | not weights. Old run outputs (check that anything still wanted is in `artifacts/` first) |
| EU | `.cache/` | 4.90 | not weights (uv cache) |
| EU | `weights/FastH3-4-step-Preview-v1-VSA-DataFree` | 77.97 | unlisted, transformer never finished, rest duplicates h3 blobs |
| EU | `weights/ltx2/…/blobs/` for the 12 `diffusion_pytorch_model-*-of-00012` text-encoder shards (and their snapshot links) | 51.61 | outside the manifest globs. The loader reads `model-*-of-00011`, which both sides have. |
| EU | `weights/wan22-ti2v-5b` stub | 0 | aborted fetch; would be replaced by the US copy |
| US | `weights/wan21-t2v-14b/.cache/huggingface/download/*.incomplete` | 21.46 | stale partial download |
| US | `runs/b200` | 1.78 | not weights |
| EU total | | **233.70** | |
| US total | | **23.24** | |

## Does the union fit in 1000 GB?

| Content | GB | Fits 1000? |
|---|---:|---|
| Manifest trees only (stale partials excluded) | 933.50 | yes, 66.5 GB spare |
| + taeh3 | 933.52 | yes |
| + ltx2 extra text-encoder layout | 985.13 | yes, 14.9 GB spare (too tight for another model) |
| + VSA-DataFree as it is on EU | 1063.10 | no |
| EU today + what it lacks (nothing removed) | 1167.22 | no |

Resulting fill, per option:

- **A. No resize, trim, then sync the manifest.** EU: 1017.32 − 233.70 +
  149.90 = 933.52 GB. US: 733.61 − 23.24 + 223.14 = 933.51 GB. Both end at
  about 93% full with identical manifest trees. Storage cost stays at
  2 × 1000 GB × $0.07 = **$140/month**. It deletes the named items on EU,
  none of which any current script reads.
- **B. Keep EU as it is, expand both to 1200 GB.** EU would hold 1167 GB
  (1017 + 150). US would hold the union plus EU's extras (~1063 GB) if the
  extras are mirrored, or 957 GB (minus its 21.46 GB of stale files) if they
  are not. Runpod pricing is $0.07/GB-month for the first 1 TB and
  $0.05/GB-month beyond (docs.runpod.io/storage/network-volumes). A volume
  can grow but **cannot shrink**. 1200 GB = $70 + $10 = **$80/month per
  volume**, $160/month for both (+$20/month over today).
- **C. Mixed.** Expand EU only to 1200 GB (+$10/month) and leave EU's
  non-weight data in place. Trim US's 21.46 GB of partials, then sync the
  manifest trees both ways. US ends at 933.5 GB and EU at about 1167 GB.
  The weight sets match, and only EU carries the extras.

Either way, fix the bookkeeping. EU `ltx23` and
`FastH3-4-step-Preview-v1-VSA-DataFree` carry `.complete` markers but are
missing files. After the sync, run `verify-weights.sh` over every tree on
both volumes. It checks index shards and safetensors lengths, but it does
not check `text_embedding_projection` for ltx23.

Transfer: 150 + 223 GB crosses between regions. A CPU pod on each side can
pull from the Hub directly with `runpod.sh`'s CPU fetch path (the manifest
rows are Hub ids), which avoids inter-region copies. At ~$0.12/hr per CPU
pod, the cost is a few pod-hours.

## Raw data

The manifests (`manifest.tsv.gz`), `du.txt`, `df.txt` and the small-file
SHA-256 lists were pulled to the session scratchpad. They are not
committed: 0.9 MB compressed for EU. To regenerate, run a CPU pod with the
inventory start command described above (about 2 minutes per volume).
