# Weight volume sync: EU vs US

Date: 2026-09-27
Compares: the two Runpod network volumes that hold the model weights (1000 GB
at the survey, both resized to 2000 GB by the owner before the sync).

- EU: `fv-weights-h3-ltx-hy`, id `jg48s6o1w0`, EUR-IS-1.
- US: `fv-weights-b200-us`, id `s2k01690bi`, US-CA-2.

> **2026-10-06: EU only.** Runpod deleted the US volume (`s2k01690bi`) on
> about 2026-10-05 while the balance was negative; the owner chose not to
> rebuild it for now. This page is the record of the 2026-09-27 sync. Its
> copy-and-verify method still applies when US is rebuilt (from EU or the
> Hub); until then new weights go on EU only. See
> [docs/ops/runpod-volumes.md](../ops/runpod-volumes.md) §0 and §5.0.

## Sync result (done 2026-09-27)

**Every tree in `weights-manifest.tsv` is now on both volumes with the same
files and sizes, and `auxiliary/` (TAE + LPIPS) is on both.** Before the sync,
the owner deleted the US `wan21-t2v-14b/.cache/**/*.incomplete` partials and
the EU `wan22-ti2v-5b` stub.

Method: CPU pods (`python:3.12-slim`, cpu3c 8 vCPU, $0.24/hr) in each volume's
own data centre re-downloaded the missing trees from the Hub with
`huggingface_hub` 2.0 + `hf_xet`, at the revision the source volume holds
(each equals today's `main`), with the manifest's globs plus
`model_index.json`. A second pass per volume removed this sync's own
`*.incomplete` partials (from two runs that were OOM-killed; 7.35 GB on US,
9.47 GB on EU, only files newer than the sync start and only in the trees it
wrote), SHA-256'd every downloaded LFS file against the Hub's LFS `sha256` at
the pinned revision, and ran `verify-weights.sh` for every tree.

| Copied | To | Repo @ revision | Layout (as the source side) | Bytes (`du -sb`) | Source-side bytes (survey) | LFS SHA-256 vs Hub | `verify-weights.sh` |
|---|---|---|---|---:|---:|---|---|
| fastwan21-1.3b | US | FastVideo/FastWan2.1-T2V-1.3B-Diffusers @ `25e7ed7` | HF cache + top-level links | 29 212 131 136 | 29.21 GB | 9/9 ok | ok |
| hy15-480-t2v | US | …HunyuanVideo-1.5-Diffusers-480p_t2v @ `286be7c` | HF cache + links | 53 384 330 435 | 53.38 GB | 14/14 ok | ok |
| hy15-480-i2v | US | …480p_i2v_step_distilled @ `854c04a` | HF cache + links | 33 780 496 799 | 33.78 GB | 8/8 ok | ok |
| hy15-720-t2v | US | …720p_t2v @ `f4dbc4a` | HF cache + links | 53 384 330 435 | 53.38 GB | 14/14 ok | ok |
| hy15-720-i2v | US | …720p_i2v_distilled @ `a1d10cf` | HF cache + links | 53 384 305 661 | 53.38 GB | 10/10 ok | ok |
| wan21-t2v-14b | EU | Wan-AI/Wan2.1-T2V-14B-Diffusers @ `38ec498` | `local_dir` (plain files) | 80 406 933 703 | 80.41 GB clean | 20/20 ok | ok |
| wan22-ti2v-5b | EU | Wan-AI/Wan2.2-TI2V-5B-Diffusers @ `b8fff73` | `local_dir` | 34 201 427 557 | 34.20 GB | 11/11 ok | ok |
| sfwan21-1.3b | EU | wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers @ `4b44356` | `local_dir` | 28 928 823 445 | 28.93 GB | 9/9 ok | ok |
| ltx23 `text_embedding_projection/` | EU (added into the existing tree) | FastVideo/LTX-2.3-Distilled-Diffusers @ `22b09fb` | HF cache + a `text_embedding_projection` link | +6 344 492 152 (tree now 47 184 305 471) | 47.18 GB | 1/1 ok (`f6ed4ecd…`, the US blob) | see ltx23 below |
| taeh3 (`weights/taeh3/taeh3.safetensors`) | EU | copy of `auxiliary/tae/taeh3.safetensors` | plain file + empty `.complete` (as US) | 22 709 752 | 22 709 752 | `4fd022bf…` = US file = pinned | — |

File lists and sizes: for every copied tree the relative paths and sizes
equal the source volume's survey manifest (no missing, extra or different
file), and every weight file of 1 MiB or less has the source's SHA-256.
`.complete` holds the fetch seconds, as the existing trees do (the ltx23
marker was left as it was). The new US trees and the ltx23 addition were
written by `huggingface_hub` 2.0, whose cache adds a content-addressed
`blobs/xx/<hash>` store (plus `CACHEDIR.TAG`) next to `models--*/`; the
snapshot and top-level links resolve to the same files, so readers see the
same tree. Nothing existing was deleted or rewritten.

### auxiliary/ (both volumes)

`/workspace/weights/auxiliary` (the name `aux` is refused by both volumes'
filesystems: `mkdir` returns EINVAL, a reserved DOS device name). Rows in
`weights-manifest.tsv` (`auxiliary/<file>`, pinned URL, `sha256:… size:…`);
`verify-weights.sh aux` checks size and SHA-256 of each: **ok on EU and US**.
Each directory carries a `.complete`.

| File | Source (pinned) | Bytes | SHA-256 |
|---|---|---:|---|
| `tae/taeh3.safetensors` | madebyollin/taehv @ `e589fdd` (fetch-tae.sh, fetch-taeh3.sh) | 22 709 752 | `4fd022bf…3d4c13` |
| `tae/taeltx2_3_wide.safetensors` | madebyollin/taehv @ `32ac014` (fetch-tae.sh) | 60 359 856 | `0a692914…52e082` |
| `tae/taew2_1.safetensors` | madebyollin/taehv @ `e589fdd` (fetch-tae.sh, fetch_taehv.sh) | 22 642 902 | `04766eac…b1a93f` |
| `tae/taew2_2.safetensors` | madebyollin/taehv @ `e589fdd` (fetch_taehv.sh) | 22 848 048 | `b84609b2…1f5325` |
| `lpips/alexnet-owt-7be5be79.pth` | download.pytorch.org (fetch-lpips.sh) | 244 408 911 | `7be5be79…cdee02` |
| `lpips/lpips_v0.1_alex.pth` | richzhang/PerceptualSimilarity @ `082bb24` (fetch-lpips.sh) | 6 009 | `df73285e…0835c0` |

`fetch-taeh3.sh` used to take `main`; `main`'s taeh3 had the pinned bytes on
2026-09-27 (as did the US `weights/taeh3` copy), so it is now pinned to
`e589fdd` and that SHA-256. `fetch-tae.sh`, `fetch_taehv.sh`,
`fetch-taeh3.sh` and `fetch-lpips.sh` copy from
`$FV_AUX_DIR` (default `<weights>/auxiliary/{tae,lpips}`) first when the hash
matches and download otherwise. `runpod-matrix.sh` reads
`$W/auxiliary/tae` and `$W/auxiliary/lpips` in place when their `.complete`
exists (else the container disk, as before), and the b200 family looks for
`auxiliary/tae/taeh3.safetensors` first.

### verify-weights.sh on both volumes after the sync

New cells: `hy15-480-t2v`, `hy15-480-i2v`, `hy15-720-t2v`, `hy15-720-i2v`,
`ltx23` (includes `text_embedding_projection`) and `aux`. On **both**
volumes: aux, fastwan21-1.3b, the four hy15 trees, wan21-t2v-14b,
wan22-ti2v-5b and sfwan21-1.3b are **ok**. `ltx23` was first **INCOMPLETE on both,
identically**: `text_encoder/gemma/model.safetensors.index.json` names
`text_encoder/gemma/model-0000{1..5}-of-00005.safetensors` (24.37 GB), and
neither volume had them, because the manifest glob `text_encoder/model-*`
does not match the `gemma/` subdirectory. With the owner's approval the
manifest row now also takes `text_encoder/gemma/*`, the `ltx23` cell also
requires `text_encoder/gemma`, and a CPU pod per volume (`ena0opqz49za8o` US,
`c7cyktznj8g6wf` EU, cpu3c 8 vCPU, 4 download workers, no OOM, about 1.5
minutes each, deleted) added `text_encoder/gemma/*` at `22b09fb` into the
existing tree (add-only; nothing else touched). Each volume gained
24 374 828 489 (US) / 24 374 819 368 (EU) bytes; `ltx23` is now
71 559 124 637 (US) / 71 559 124 839 (EU) bytes. All five shards match the
Hub's LFS SHA-256 on both volumes, and **`verify-weights.sh ltx23` is ok on
both**. The manifest's `latent_upsampler/*` and
`ltx-2.3-22b-distilled-lora-384*.safetensors` globs match nothing in the repo
at `22b09fb`.

### Pre-quantized FP8 text encoders (E13, copied US → EU)

`h3-base/text_encoder_fp8` and `ltx25/text_encoder_fp8` were written on US
by `fv-gpucheck quantize-text-encoder` (docs/gaps/2026-09-27-cold-start.md)
and are now on EU too, byte for byte (owner-approved, add-only). They are
derived, not fetched, so they have no `weights-manifest.tsv` row;
`verify-weights.sh text-fp8` checks them (manifest SHA-256, model size and
full length; the model's SHA-256 too with `FV_VERIFY_FP8_SHA=1`).

| Tree | File | Bytes | SHA-256 (US = EU) |
|---|---|---:|---|
| `h3-base/text_encoder_fp8` | `manifest.json` | 2 965 | `e5504561…190a42` |
| | `model.safetensors` | 25 950 724 552 | `c13fab5c…072c2d` |
| `ltx25/text_encoder_fp8` | `manifest.json` | 1 408 | `bb5a3049…ba06db` |
| | `model.safetensors` | 12 923 848 536 | `0615832b…b4ae2f` |

Method: a CPU pod on US (`fuh1p8uetgkvlf`, 2 vCPU, $0.06/hr) served the two
folders read-only over HTTP with Range support (under a random path, through
the Runpod proxy; the pod had no public IP) and SHA-256'd every file on the
volume. A CPU pod on EU (`2elq70bhx34ihm`, cpu3c 8 vCPU) checked that neither
destination existed, pulled into `<tree>/.text_encoder_fp8.partial-<stamp>`
(416 s for 38.9 GB, ~93 MB/s, no retries), fsync'd, re-read every file from
the volume and compared its SHA-256 with the US hash, and checked the files
against `manifest.json` (file list and sizes; the manifest's own `sha256`
fields are empty). Only then it renamed each temp folder to
`text_encoder_fp8`. A separate fresh EU pod (`yj7x7p9n9km0r8`) then ran
`FV_VERIFY_FP8_SHA=1 verify-weights.sh text-fp8`: **ok** (full SHA-256 of
both models = US), and every source shard the manifests name
(`text_encoder/`, 14 for h3-base, 5 for ltx25) has the manifest's size on EU,
which the loader requires before it uses a tree. Nothing else on either
volume was touched. About 20 minutes wall clock, about $0.11 of pods (all
deleted; a watchdog would have deleted them after 2.5 h).

### Spark latent upscaler on EU (2026-09-29)

`verify-weights.sh sol-h3-spark` was INCOMPLETE on EU only: the gate reads
`upscaler/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`
(the Hub layout since `6749d7b`), and EU held only the earlier top-level
`upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors`: same bytes
(690 592 992, SHA-256 `4f57821f…46a5e6`, the LFS oid at both revisions).
US holds the HF cache at `3f941d5` with that path as a link (SHA-256
equal). A 2 vCPU CPU pod on EU downloaded the file from the Hub at `3f941d5`
into `upscaler/.minimax_h3_latent_upscaler_3d_conv_v1.partial-<stamp>/`,
fsync'd, re-read it with the page cache dropped, matched size and SHA-256,
and renamed the folder (add-only; the old file and `.complete` untouched).
The manifest now pins that revision and SHA-256, and the `sol-h3-spark`
cell checks the upscaler's size and SHA-256. Fresh pods then ran
`verify-weights.sh sol-h3-spark`: **ok on EU and US**. Four pods
(`0i0pk5duq3mnkn`, `j0g7h0i4422pdq`, `pa0nbcn5f8dcvo`, `d6re9jgfuly2gw`),
under a minute each at $0.06/hr, all deleted.

### Remaining differences (left alone on purpose)

| Where | What | GB | Why left |
|---|---|---:|---|
| EU only | `weights/FastH3-4-step-Preview-v1-VSA-DataFree` | 77.97 | unlisted, transformer incomplete despite its marker (see below) |
| EU only | `weights/ltx2` 12 `text_encoder/diffusion_pytorch_model-000{01..12}-of-00012` shards | 51.61 | unlisted layout; the loader reads `model-*` |
| EU only | `upstream/`, `runs/`, `.cache/` | 104.1 | not weights |
| US only | `runs/` | 1.78 | not weights |
| both | `weights/mmaudio-44k-v2` (US 21.46 complete, EU 10.73 in progress, no marker at the time) | — | being written by another agent |
| layout | FastH3 LoRA `vsa-datafree` adapter and upscaler: plain file on EU, HF blob on US; new trees: HF-cache 2.0 on US / 1.x on EU | 0 | same bytes |

Weight bytes now (`du -sb` of `weights/`, from the post-sync pass): the two
volumes hold the same manifest trees plus `auxiliary/` (0.37 GB) and `taeh3`.
The US volume reported 957.1 GB used of 2000 GB after the sync.

Pods (all deleted; each had a local delete backstop, killed after the
delete): `ktrxja4eloj1tv` (US, stopped: `aux` EINVAL), `273mwscpn7uuu1` (US,
OOM-killed with `HF_XET_HIGH_PERFORMANCE`), `ul6ticbyj1y3ai` (EU, OOM-killed),
`fkukcxrlu1b1nx` (US), `d0ut8xr2lvo98a` (EU), `1eqmqy2ozjw36e` (US check),
`ane2ky628scnep` (EU check), then `ena0opqz49za8o` / `c7cyktznj8g6wf` (ltx23 Gemma). About 33 pod-minutes at $0.24/hr, about $0.14.

## Survey (before the sync)

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

### Headline

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

### Top-level inventory

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

### Weight trees

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

### Diff

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

### Removal candidates (named; nothing done)

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

### Does the union fit in 1000 GB?

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

### Raw data

The manifests (`manifest.tsv.gz`), `du.txt`, `df.txt` and the small-file
SHA-256 lists were pulled to the session scratchpad. They are not
committed: 0.9 MB compressed for EU. To regenerate, run a CPU pod with the
inventory start command described above (about 2 minutes per volume).
