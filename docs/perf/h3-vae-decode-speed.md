# H3 video VAE decode speed (work order follow-up, 2026-10-14)

**Default flipped (owner decision, 2026-10-14):** the MXFP8 decoder
linears are now the **default** for the 36-block video decoder on sm_100+
GPUs (`FASTVIDEO_H3_VAE_MXFP8` unset); `=0` restores the bf16 decode byte
for byte. See [§5](#5-default-flip-2026-10-14) for what changed, the memory
plan and the sanity run. Since the same day the default also covers h3-trim's
26-block decoder, which passed the same quality gate (−22 % video VAE,
[§6](#6-the-26-block-decoder-h3-trim-2026-10-14)). The text below is the
measurement as written before the flip.

**Verdict.** The ≥ 20 % target is reached only with a numerics change:
`FASTVIDEO_H3_VAE_MXFP8=1` (MXFP8 block linears, then **default off**, the owner
decides) cuts the video VAE by **26 %** at 480p, 768p and 1080p, and its
frames sit at LPIPS 0.001 / PSNR 51 dB from the bf16 decode. The
numerics-preserving candidates add up to **1.6 %** at 768p: the decoder
already runs its GEMMs at 85 % of the card's bf16 peak and leaves the GPU
idle 0.1 % of the decode, so there is little left to win without changing
the arithmetic. One of them (vector merge-heads / SwiGLU kernels,
bit-identical) is now on by default; the rest measured < 1 % and stay off
or unbuilt.

This continues docs/perf/h3-vae-tile-batching.md (tile batching: −1.6 % at
768p). Raw data: `artifacts/perf/h3-vae-decode-speed/` (`pod1-results.tgz`:
every timing cell, the bench and the compares; `gate-results.tgz`: the
quality-gate cells; `trace/`: the decode trace; `driver/`: `ds.sh`,
`summarize.py`, `compare_table.py`, prompt files; `ledger.tsv`).

## Setup

| | |
|---|---|
| Timing pod | `i8mxgk9smv8dt8`, Runpod SECURE, EU-RO-1, RTX PRO 6000 Blackwell Server Edition (97 887 MiB, **0 MiB used at boot**, max SM clock 2430 MHz, 600 W), driver 595.91.07, $2.49/hr. **Every timing below is from this one pod.** Pods that showed 95 785 MiB used at boot were deleted at once (see Cost) |
| Gate pod | `x3mhvv4adol4iu`, same type and region, 0 MiB at boot (quality only; no timing compared across pods) |
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-runtime@sha256:efa9348a68b8…` (`sha-632742a`), the image of the tile-batching run |
| Binary | this branch's `fv-gpucheck`, built on the build pod and uploaded: `70159e7` (bench, `base768`), `6a610ce` (every later timing cell; differs only in what `FASTVIDEO_H3_VAE_VEC` switches, which is off in the baselines — `base768` and `nbase768` agree to the millisecond and byte for byte), `3f93fd0` (gate pod) |
| Weights | EU volume `bz0ofkfx45`, read only |
| Workload | as the tile-batching run: `--techniques h3/fasth3_4step_vsa --mode fast h3 gen --h3-recipe 4step-vsa`, 5 s (124 frames), `--warm`, h3-demo at seeds 0 / 1000 / 2000 (1080p: seed 0 only), `FASTVIDEO_ATTN_SAGE=0 FASTVIDEO_FLASH_KERNEL=cudnn FASTVIDEO_H3_VAE_CHECK=1`. 1080p is 1920x1088 with `--oversize-canvas` |
| Metric | "video VAE" is `DecodeSplit::vae_s` (GPU time between chunk hand-offs, FastVideo's `video_decoding_stage`), median of the timed clips |

**A host quirk.** On the timing pod `ffmpeg` (the mp4 writer) was held to
4 cores and fell behind: in `vec768` and `mx768` the writer push grew to
19-147 s per clip. `DecodeSplit::vae_s` excludes those waits (vec768's VAE
matches its later rerun to 0.1 %), but those two cells' decode and
generate times are void. Every later cell runs with `--no-mp4` (PNG frames
only), and the 768p baseline was retaken that way (`nbase768`).

## 0. 1080p: does tile batching scale differently?

| 1080p 5 s, seed 0 | tiles / forward | video VAE s | frames |
|---|---|---|---|
| default (`TILE_BATCH` 8) | 8 (60 tiles per chunk as 8 x 7.5) | **13.48** | — |
| `FASTVIDEO_H3_VAE_TILE_BATCH=196` | 60 (the per-chunk cap) | **13.86 (+2.8 %)** | not byte-identical: LPIPS 7e-7, PSNR 78.7 dB |

**No: at 1080p batching is slower, not faster.** The synchronized per-op
check shows why: per tile, the 60-tile forward takes the same 32.8 ms as the
8-tile one, but its `ff_in` GEMM (M = 107 820 rows) gets a slower cuBLAS
kernel (10.6 → 11.9 ms per tile), which eats the launch saving. The
different kernel also changes the summation order, hence the rounding-level
pixel difference (at 768p both batch sizes happened to give identical bits).
The work order expected batching to help more at 1080p; the decoder is
GEMM-bound, so it does not. 1080p video VAE is 13.48 / 6.31 = 2.14x the
768p one, for 2.14x the tiles (420 against 196).

## 1. Profile of one 768p decode

`FASTVIDEO_GPU_TRACE_DECODE=1` (CUPTI, unsynchronized; this branch hooks it
to the H3 decode), warm clip, 6.33 s video VAE (0.4 % above the untraced
6.31 s): `trace/h3-decode-768p-gpu_trace.json`.

* **GPU busy 6.35 s of a 6.36 s window: idle 0.1 %** (11 521 gaps, 99.7 %
  of them < 5 µs). 15 009 kernels. Launch overhead is not where the time
  goes.

| category | share | s | what |
|---|---|---|---|
| GEMM | **67.6 %** | 4.28 | cuBLAS `nvjet_sm120` 256x128 (ff_in, qkv, ff_out: 46 %), 96x256 (17 %), 128x160 (4 %) |
| norm / element-wise | 13.4 % | 0.85 | `h3v_swiglu_bf16` 7.1 % (bias + SiLU gate + product), `h3v_residual_norm_bf16` 6.0 % (bias + LayerScale + residual add + RMSNorm + bf16 cast, 2 per block) |
| attention | 9.7 % | 0.62 | cuDNN fused SDPA (softmax inside; pinned by `FASTVIDEO_FLASH_KERNEL=cudnn`) |
| layout | 9.0 % | 0.57 | `h3v_qkv_heads_bf16` 5.7 % (bias + per-head RMSNorm + **RoPE** + BHSD split), `h3v_merge_heads_bf16` 1.3 %, tile gather / block copy 2.0 % |
| cast | 0.2 % | 0.01 | proj_in / proj_out |

There is no separate GELU, cast or RoPE pass: the earlier fusion work
(`vit_fused`) already folds every element-wise op between two GEMMs into
one kernel.

**GEMM efficiency.** The block linears are 1.70 PFLOP per 768p decode
(196 tiles x 36 blocks x 2 x 1797 x 2048 x 32768); 1.70 PFLOP / 4.28 s =
**397 TFLOPS, 85 % of the bf16 dense peak** (188 SMs x 1024 FLOP/clk x
2.43 GHz = 468 TFLOPS; the clock under load is lower, so the true fraction is
higher). Attention: 187 TFLOP / 0.62 s = 304 TFLOPS.

**Kernel bench** (`fv-gpucheck kernels --groups h3vae_bench`, new: the
decoder's ops at its real shapes, 7 and 28 tiles; `bench/kernels.json`):

| op (7 tiles = 12 579 rows) | current | alternatives |
|---|---|---|
| qkv GEMM 2048→6144 | cuBLAS 0.867 ms (365 TFLOPS) | cuBLASLt heuristic #0-7: 0.864-1.223 ms; **MXFP8 0.506-0.532 ms** (+0.087 ms quantize) |
| to_out GEMM 2048→2048 | 0.325 ms (324) | Lt 0.323-0.366; MXFP8 0.206 |
| ff_in GEMM 2048→16384 | 2.172 ms (389) | Lt 2.175-3.072; MXFP8 1.231-1.308 |
| ff_out GEMM 8192→2048 | 1.160 ms (364) | Lt 1.099-1.269 (#7 −5 %, only at 7 tiles); MXFP8 0.660 |
| SDPA 224 x 1797 x 64 | cuDNN 0.668 ms (277 TFLOPS) | V2 0.646 (287, what `auto` picks), V1 0.657 |
| swiglu | 0.470 ms, 1316 GB/s | vector twin 0.444 ms (bit-equal) |
| residual + norm | 0.243 ms, 1275 GB/s | — |
| qkv heads | 0.387 ms, 799 GB/s | vector twin **0.497 ms** (bit-equal, slower) |
| merge heads | 0.092 ms | vector twin 0.043 ms (bit-equal) |
| device copy (roofline) | 1437 GB/s | |

At 28 tiles the picture is the same (cuBLAS 394-415 TFLOPS, Lt never
faster, MXFP8 670-750 TFLOPS). MXFP8 GEMM output is 3.8 % rel-L2 from bf16
per GEMM on random operands; the fused producers write exactly what
`mxfp8_quantize` makes of the bf16 tensor (0 code and 0 scale mismatches).

**Ranking by expected saving** (768p, of 6.31 s):

1. MXFP8 block linears: GEMM time 4.28 → ~2.4 s → **≈ −28 %** (numerics change).
2. Vector merge-heads / SwiGLU: ≈ 0.08 s, ≈ −1.3 %.
3. SDPA kernel: V2 instead of cuDNN where the run pins cuDNN: ≈ 0.02 s,
   −0.4 % (the production `auto` already picks V2 for this shape).
4. cuBLASLt algorithm choice: no shape is faster except ff_out at 7 tiles
   (−5 % of 18 % of GEMM time): < 1 %.
5. Epilogue fusion: cuBLASLt has no SwiGLU / RMSNorm epilogue. Its bias
   epilogue would move the bias before the bf16 rounding (a numerics
   change) and save nothing, because the next kernel reads the bias row
   anyway. Fusing the residual add (`beta * C`, f32 C) moves the same bytes
   as today (the GEMM would write and read f32 rows instead of writing bf16
   and the norm kernel reading it). SwiGLU in an epilogue would save
   ≈ 0.45 s at most, but only with a custom GEMM that matches cuBLAS at
   85 % of peak; a 10 % slower GEMM loses more than that. Not attempted.

## 2. Numerics-preserving candidates (768p, 3 seeds, medians)

| candidate | flag | video VAE s | change | parity | default |
|---|---|---|---|---|---|
| baseline | — | 6.306 | | | |
| a. vector merge-heads + SwiGLU twins | `FASTVIDEO_H3_VAE_VEC` | **6.203** | **−1.6 %** | frames byte-identical (all 4 clips), VAE check pass (rel-L2 1.40e-3) | **on** |
| a'. vector qkv-heads twin | (dropped) | — | bench: +28 % on that kernel | bit-equal | not used |
| b. cuBLASLt algorithm | `FASTVIDEO_H3_VAE_GEMM=lt[N]` | not run | bench: no shape faster than cuBLAS except ff_out @ 7 tiles | Lt #0 bit-equal to cuBLAS | off (`cublas`) |
| c. SDPA kernel | `FASTVIDEO_H3_VAE_SDPA` | not run | bench: V2 3 % faster than cuDNN; `auto` already V2 | V1 = V2 bit-equal; cuDNN 6e-5 rel-L2 | off (`auto`) |

b and c were not run end to end: the bench bounds them below 1 % of the
decode, under the stop rule's 5 %. Their flags stay as tuning seams.

## 3. MXFP8 decoder linears (`FASTVIDEO_H3_VAE_MXFP8`, default on since 2026-10-14)

The four block linears (qkv, out, SwiGLU in, SwiGLU out) of all 36 blocks
run as cuBLASLt MXFP8 GEMMs (E4M3 codes, one E8M0 scale per 32 values, the
DiT's `[techniques.mxfp8]` recipe and kernels). Weights are quantized from
the bf16 checkpoint at load (on the device, +2.4 GB). Activations are
quantized by the kernel that produces them (`h3v_residual_norm_mx`,
`h3v_merge_heads_mx`, `h3v_swiglu_mx`), which write the MXFP8 operand
instead of bf16, so there is no separate quantize pass. Proj in / out, the
residual stream (f32), attention (bf16), tiling, overlap and blend are
unchanged. The VAE check runs the MXFP8 forward too and logs its rel-L2
against the f32-activation ViT (5.3e-3 at 768p, 9.1e-3 at 480p, 9.9e-3 at
1080p; the check disables MXFP8 above 0.1, a sanity bound).

| resolution | baseline video VAE s | MXFP8 video VAE s | change | generate s | vs bf16 decode: LPIPS / PSNR / sharpness / jitter |
|---|---|---|---|---|---|
| 480p, 3 seeds | 2.671 | **1.986** | **−25.6 %** | 10.67 → 9.95 (−6.7 %) | 0.0009 / 50.3 dB / 1.001 / 1.002 |
| 768p, 3 seeds | 6.306 | **4.679** (4.643 in `mx768`) | **−25.8 %** | 26.83 → 25.21 (−6.0 %) | 0.0010 / 51.4 dB / 1.001 / 1.001 |
| 1080p, seed 0 | 13.482 | **9.906** | **−26.5 %** | 63.03 → 59.46 (−5.7 %) | 0.0012 / 51.3 dB / 1.000 / 1.000 |

(`FASTVIDEO_H3_VAE_VEC` has no effect under MXFP8: the MXFP8 producers
replace the kernels it switches.)

### Quality gate

Gate pod, `h3/fasth3_4step_vsa` at 768p, the calibrated set (5 prompts x 3
seeds, `prompts-5x3.json`), one warm process per arm: `g_bf16` (default
decode) and `g_mx` (`FASTVIDEO_H3_VAE_MXFP8=1`), same latents (the DiT is
untouched and deterministic). `fv-gpucheck gate` with
`scripts/gpu/gate-policy.toml` (`gate/gate-mxfp8.json`):

| gate | status | |
|---|---|---|
| quantitative_quality | **pass** (105 / 105 checks) | MXFP8 vs bf16 decode, 15 clips: LPIPS mean median **0.0018** (max of means 0.0026, worst frame 0.0061), PSNR median **49.8 dB** (worst clip mean 48.8), sharpness 0.9988-1.002, jitter 0.9985-1.001 |
| official_config | pass | same workload, seeds, prompts |
| performance | fail (by design) | primary metric is `denoise_s` (19.63 vs 19.64 s: the DiT is untouched); secondary `total_s` 26.45 → 24.78 s, **1.067x** |
| artifact | fail (harness) | the cells ran `--no-mp4` (see the host quirk), so the gate reads both arms' clips as `blocked`; every compare report passed |

Video VAE on the gate pod, medians of 15 clips: 6.281 → 4.715 s (−24.9 %).

**Against the 49-forward base** (recipe `base`, dense, one seed per prompt =
5 clips, rendered on the gate pod, ~7 min each; three seeds would have cost
another 70 min of GPU): the MXFP8 decode does not move the distance to the
base. LPIPS median 0.7225 (bf16 decode) against 0.7226 (MXFP8), PSNR 9.72
against 9.71 dB, sharpness 1.296 against 1.297, jitter 2.037 against 2.039;
per prompt the largest LPIPS change is 0.0004. (The 4-step recipe itself is
far outside the lossy band against the base, as recorded for this profile;
the VAE change is three orders of magnitude below that gap.)

**Verdict: the MXFP8 decode passes the quality gate** (LPIPS 0.002, PSNR
50 dB, sharpness and jitter at 1.00 against the bf16 decode; no measurable
change against the 49-forward base) and is 25-26 % faster at every
resolution. It stayed **off by default** in this run: it is a numerics change and the
default is the owner's call (who turned it on on 2026-10-14, §5). It also needs 2.4 GB more device memory while
the VAE is resident (peak 72.2 against 69.8 GB at 768p).

## 4. Cumulative

| resolution | baseline | numerics-preserving | + MXFP8 (default since 2026-10-14) |
|---|---|---|---|
| 480p | 2.671 s | not measured separately (the VEC kernels are ≈ 1 % of the decode) | 1.986 s (−25.6 %) |
| 768p | 6.306 s | 6.203 s (−1.6 %) | 4.679 s (−25.8 %) |
| 1080p | 13.482 s | not measured separately | 9.906 s (−26.5 %) |

## What did not help

* Tile batching at 1080p: +2.8 % (section 0).
* cuBLASLt algorithm selection: cuBLAS `GemmEx` already picks the fastest
  kernel on 7 of 8 shape / batch cells.
* The vector qkv-heads kernel: bit-equal but 28 % slower than the scalar one
  (the per-lane shuffles cost more than the wider load saves).
* GEMM epilogue fusion with cuBLASLt: nothing to fuse that saves bytes
  without a custom GEMM (section 1).
* CUDA graphs: GPU idle is 0.1 % of the decode.

## Cost

Timing pod 4393 s, gate pod 3892 s, the first pod (rejected after 12 min:
95.8 GB already used at boot, found late) 752 s, ten more pods rejected at
boot (95.8 GB used, or no sidecar) 12-55 s each: 9300 s at $2.49/hr =
**$6.43** of GPU, plus build-pod time on the shared build pod. Every pod was deleted and checked gone (404). The
volume and R2 were only read.

## 5. Default flip (2026-10-14)

**Owner decision, 2026-10-14:** the MXFP8 decoder linears are the default.
`FASTVIDEO_H3_VAE_MXFP8` now resolves as follows (`h3/vae.rs`,
`resolve_mxfp8`):

| setting | sm_100+ (B200, RTX PRO 6000, RTX 5090, GB10) | pre-Blackwell (H100, A100, L40S, ...) | CPU run |
|---|---|---|---|
| unset, 36-block decoder | **MXFP8** | bf16, said once in the log | bf16 |
| unset, 26-block lightweight decoder (h3-trim) | **MXFP8** (since its gate passed, [§6](#6-the-26-block-decoder-h3-trim-2026-10-14)) | bf16, said once | bf16 |
| unset, a decoder of any other depth | bf16, said once (not measured) | bf16, said once | bf16 |
| `=0` | bf16, byte-identical to the old default | bf16 | bf16 |
| `=1` | MXFP8 on either decoder | **error at load** (cuBLASLt has no MXFP8 kernels below sm_100) | bf16 (no device) |
| anything else | error | error | error |

This follows the DiT's MXFP8 default (`FASTVIDEO_H3_QUANT`: MXFP8 unset on
sm_100+, off below). The weights are only quantized when the fused ViT is
on (`FASTVIDEO_H3_VAE_FUSED` not 0), the only path that runs them. The
`FASTVIDEO_H3_VAE_CHECK` harness is unchanged and still drops MXFP8 above
its 0.1 rel-L2 sanity bound. TAEH3 is untouched.

**Memory plan.** The MXFP8 copies sit beside the bf16 weights (the unfused
path and the check read those): 36 blocks x 16 d² x 33/32 bytes = **2.32
GiB**, exactly the ledger's `vae` growth on the pod below (4.52 → 6.84 GiB).
`fastvideo_models::h3::memory` prices them (`video_vae_mxfp8_bytes`,
`H3PlanOptions::vae_mxfp8`, default true) wherever the decoder lives: for
the whole run when the DiT is resident, for the decode stage only when it
streams. The pipeline passes the resolved setting to its residency and
i2v-encoder plans (off with TAEH3, `=0`, or below sm_100). What moves:

| plan | before | after | budget |
|---|---|---|---|
| resident 768p 5 s, dense / with the VSA gate (the `dit_offload = auto` threshold) | 60.0 / 64.6 GiB | 62.3 / 66.9 GiB | 96 GB card with the FP8 encoder resident: ~72 GiB free, still resident; beside the 11 GB upsampler sidecar (60.6 GiB free) it streamed before and still does |
| 1080P tier minimum (`plan_1080p_min_device_bytes`) | 71.2 GiB | 73.5 GiB | 80 GB cards report 79.6 GiB: tier kept; 48 GB: off as before |
| streamed (RTX 5090 32 GiB), 768p 5 s peak | 19.4 GiB (denoise) | 19.4 GiB (denoise; decode 12.6) | < 30 GiB |
| streamed, 480p 124 frames peak | 10.9 GiB (denoise) | 11.9 GiB (video decode) | < 30 GiB |

No supported configuration crosses its budget, so there is no bf16
fallback for memory. The serve engine's live 1080P admission check
(`measured_working_bytes`) compares the working set with the free memory
after the weights are loaded, so a resident decoder's copies are already
counted there.

**Sanity run** (one RTX PRO 6000 pod, `mpoc5rmbsqvfd7`, EU-RO-1, 0 MiB
used at boot, image `fastvideo-rs-runtime@sha256:efa9348a…`, EU volume
`bz0ofkfx45` read only; `h3/fasth3_4step_vsa` = h3-turbo, 768p 5 s, h3-demo
seed 0, `--warm`, the pinned kernels of §Setup; one process per arm; raw:
`artifacts/perf/h3-vae-mxfp8-default/`):

| arm | binary | video VAE s (timed clip) | decode s | peak MiB | frames sha256 (16) |
|---|---|---|---|---|---|
| new default (MXFP8) | this branch | **4.67** | 4.85 | 72 199 | `11487f92cf893526` |
| `FASTVIDEO_H3_VAE_MXFP8=0` | this branch | 6.23 | 6.40 | 69 767 | `454cc62632a08a55` |
| old default | main `3ef71f2` | 6.23 | 6.40 | 69 767 | `454cc62632a08a55` |

* Decode-time win: **−25.0 %** video VAE (6.23 → 4.67 s), in line with §3.
* `=0` against main's default: **byte-identical** (all 124 frames; compare
  `max_abs_diff_uint8` 0).
* MXFP8 against bf16: LPIPS mean 0.0009 (max 0.0019), PSNR 50.8 dB (min
  47.5), sharpness 1.002; the check: rel-L2 5.3e-3 vs the unfused ViT
  ("mxfp8 kept").
* Pod lifetime 912 s at $2.49/hr = **$0.63**; deleted and checked gone (404).

## 6. The 26-block decoder (h3-trim, 2026-10-14)

**Verdict: MXFP8 passes the quality gate on the 26-block decoder, and is
now the default there.** `MXFP8_DEFAULT_LAYERS` in `h3/vae.rs` is
`[36, 26]`. On sm_100+, an unset `FASTVIDEO_H3_VAE_MXFP8` now loads MXFP8
copies for h3-trim too. `=0` restores the bf16 decode byte for byte.
Results against the bf16 decode of the same latents, 15 clips per
resolution:

| | LPIPS mean (median / worst clip) | PSNR | gate quality checks |
|---|---|---|---|
| 480p | 0.0008 / 0.0012 | 49.8 dB | 105 / 105 |
| 768p | 0.0011 / 0.0017 | 50.4 dB | 105 / 105 |

These match the 36-block decoder. The video VAE gets 22 % faster, and the
decoder needs 1.76 GiB more while resident.

**Setup.** Two runs on RTX PRO 6000 Blackwell Workstation Edition pods in
EU-RO-1 at $2.59/hr, each with 2 MiB used at boot. Run 1 was
`53q5tp7o0ojzj7`: timing, memory, the identity check, and 4 compares (its
compare stage then hung, see Cost). Run 2 was `t7m4zmkgitftfs`: the full
gate.
- Image: runtime `fastvideo-rs-runtime:sha-94d6073`, with this branch's
  `fv-gpucheck` from the bundle.
- Weights: the Trim tree from R2; h3-base and LPIPS from the EU volume
  `bz0ofkfx45`, read only.
- Workload: h3-trim as served: `--h3-recipe trim-8step`, the MXFP8 DiT
  resident, the FP8 text encoder resident.
- Clips: 5 s (124 frames), the #106 gate set (`prompts-5x3.json`: 5
  prompts x 3 seeds), `--warm`, `--no-mp4`, the pinned kernels of §Setup.
- Files in `artifacts/perf/h3-trim-vae-mxfp8/`:
  - `driver/`: `pod.sh`, `launch.sh`, `summarize.py`;
  - `results.tgz` (run 2, the full gate) and `results-run1.tgz`;
  - `summary.md` (the tables of run 2) and `ledger.tsv`.

**Same latents, one DiT pass.** Running the trim's DiT twice per arm, as
the #106 gate did, would cost about 33 GPU-minutes. Instead,
`FASTVIDEO_H3_VAE_AB=1` (measurement only, off by default) decodes each
clip's latents a second time in the same process, right after the MXFP8
decode. It uses the bf16 block linears, through `vae::with_bf16_linears`:
the bf16 weights kept beside the MXFP8 copies, with the `=0` kernels. The
bf16 frames go to `<clip>/vae-bf16/`.
- Check against `=0`: in run 1, separate `FASTVIDEO_H3_VAE_MXFP8=0`
  processes rendered h3-demo seed 0 at both resolutions. Their frames are
  **byte-identical** to the in-process bf16 re-decodes. The sha256 of all
  124 PNGs is `f4059fc94511a7a4` at 480p and `8b5916e5425fa05a` at 768p.
  Run 2 reproduced both hashes on another pod.
- The VAE check on the first decode (rel-L2 vs the unfused ViT):

| path | 480p | 768p | 36-block decoder |
|---|---|---|---|
| fused bf16 | 1.9e-3 | 1.6e-3 | — |
| MXFP8 ("mxfp8 kept") | 1.1e-2 | 7.1e-3 | 9.1e-3 (480p) / 5.3e-3 (768p) |

**Video VAE time** (`DecodeSplit::vae_s`, medians of the 15 timed clips
per resolution, both decodes in one warm process):

| resolution | run | bf16 | MXFP8 | change |
|---|---|---|---|---|
| 480p (832x480) | 1 | 2.104 s | **1.627 s** | **−22.7 %** |
| 480p | 2 | 2.135 s | **1.652 s** | **−22.6 %** |
| 768p (1344x768) | 1 | 4.961 s | **3.867 s** | **−22.1 %** |
| 768p | 2 | 5.053 s | **3.944 s** | **−21.9 %** |

This is a smaller gain than the 36-block decoder's −25.6 / −25.8 %:
the block linears are a smaller share of the 26-block decode. On the
trim's end-to-end time it saves 1.1 s at 768p (≈ 49 s) and 0.5 s at 480p
(≈ 19 s).

**Peak memory** (run 1, h3-demo seed 0). The first two columns are
`benchmark.json` stage peaks (allocated); the last is the whole-process
peak from nvidia-smi.

| resolution | arm | denoise (run peak) | video_decode stage | nvidia-smi |
|---|---|---|---|---|
| 480p | bf16 (`=0`) | 53.84 GiB | 51.91 GiB | 56 870 MiB |
| 480p | MXFP8 | 55.60 GiB | 51.74 GiB | 58 504 MiB |
| 768p | bf16 (`=0`) | 60.48 GiB | 52.38 GiB | 63 816 MiB |
| 768p | MXFP8 | 62.24 GiB | 52.33 GiB | 66 568 MiB |

- The MXFP8 copies add 1.76-1.79 GiB to every stage while the decoder is
  resident. Expected: 26 blocks x 16 d² x 33/32 bytes = 1.68 GiB, plus
  allocator rounding. The decode stage itself does not grow.
- **Memory plan: no change needed.** `h3::memory::plan` prices the 36-block
  decoder for every H3 recipe, the trim included. With MXFP8 on (the
  pipeline now resolves it on for the trim) that is 2.32 GiB of copies,
  more than the trim's 1.68 GiB. The new test
  `the_trim_decoders_mxfp8_copies_fit_the_planned_decoder` checks this.
- The trim's working set (≤ 62.3 GiB at 768p on a 96 GB card) stays far
  below every budget.

**Quality gate** (run 2). `compare-clips` (LPIPS alex) compared the bf16
decode (baseline) with the MXFP8 decode (candidate) for every timed clip.
Then `fv-gpucheck gate` ran with `scripts/gpu/gate-policy.toml`, kind
`lossy`. Per-clip rows are in `summary.md`.

| | 480p (15 clips) | 768p (15 clips) |
|---|---|---|
| LPIPS mean: median / worst clip | 0.00082 / 0.00119 | 0.00106 / 0.00174 (spark-mountain-lake-s42) |
| LPIPS worst frame | 0.0024 | 0.0038 |
| PSNR mean: median / worst clip | 49.8 / 48.0 dB | 50.4 / 49.2 dB |
| PSNR worst frame | 45.2 dB | 45.0 dB |
| sharpness ratio | 0.9989-1.001 | 0.9994-1.001 |
| temporal jitter ratio | 0.999-1.003 | 0.9992-1.004 |
| gate `quantitative_quality` | **pass, 105 / 105** (60 hard checks + 45 telemetry rows, all ok) | **pass, 105 / 105** |

The gate's other verdicts fail **by design**, as in §3:
- `performance` fails because both sides are the same cell (`denoise_s`
  1.00x).
- `artifact` fails because the cells ran `--no-mp4`, so the gate reads the
  clips as `blocked`. Every compare report passed.

**Cost.**

| pod | what | GPU time | cost |
|---|---|---|---|
| `53q5tp7o0ojzj7` | run 1 | 2040 s | $1.47 |
| `t7m4zmkgitftfs` | run 2 | 1639 s | $1.18 |
| **total** | | | **$2.65** |

- Run 1 was stopped at its budget's hard limit. Its compare stage had hung
  after four clips: a bare `wait` also waited for the nvidia-smi sampler.
  The fixed `pod.sh` now compares each resolution right after its cell,
  one clip at a time.
- Run 2 skipped the identity cells (`FV_VERIFY=0`).
- Both pods were deleted by hand and checked gone (`GET /pods/<id>`: not
  found), and their backstops were stopped by PID.
- Every staged bundle was removed from `fv-weights-staging/tmp/`. That
  includes the ones left by pod creates that Runpod refused for lack of
  instances. EU-RO-1 had no RTX PRO 6000 stock for about 10 and 25
  minutes before the two runs; no pod existed then.
- Builds and tests ran on the shared build pod.
