# Datacenter kernel plan: H100 / H200 (sm_90) and B200 (sm_100)

Date: 2026-09-29. Scope: a code-level map and a work plan. No GPU was used
for this document. It pairs with `docs/perf/datacenter-profile.md`, which a
separate run is writing from H100 and B200 profiles. Every gain below is an
estimate from existing measurements, marked **(profile)** where the profile
must confirm it before work starts. Section 6 says how each profile number
reorders the plan.

Inputs: `docs/serve/bench/b200.md`,
`docs/gaps/2026-09-27-attention-sm120-cudnn-vsa.md`,
`docs/gaps/2026-09-25-sol-engine-code-level.md`,
`docs/gaps/2026-09-24-codebase-review-gpu-sol-engine-oxide.md`,
`docs/oracle.md`, `scripts/gpu/gate-policy.toml`, `decision-log.md`
(`FVID-2026-09-27-attention-datacenter`, `-attention-datacenter-sm90`,
`-attention-sm120`, `-ltx-nvfp4-ffn-cublaslt`, `FVID-2026-09-24-nvfp4-te-static6-oxide`,
`FVID-2026-09-19-sm120-is-not-umma`), and the source at `d878286`.

## 0. Summary

**Where things stand.** The mma.sync kernels are tuned for the RTX PRO 6000
(sm_120). On datacenter GPUs they reach about PRO 6000 speed and no more
(V2 dense: 369-374 TFLOPS on B200, 303-326 on H100). One op has a real
datacenter kernel: dense attention at d = 128 (`attn_dc.cu`: tcgen05 + TMEM
on B200, wgmma on H100). The ops that do most of the served attention work
still run the mma.sync builds on sm_90 and sm_100: VSA (H3 turbo), Sol (H3
max, LTX stage 2) and block-causal attention. GEMMs go through cuBLAS and
cuBLASLt, so they already use wgmma and tcgen05. H100 and H200 get no FP8,
though: MXFP8 has no sm_90 kernel, and the per-tensor W8A8 fallback failed
H3's quality gate.

**Why it matters.** At Runpod prices a GPU must beat the PRO 6000
($2.09/hr) by the price ratio before it is cheaper per clip:

| GPU | $/hr | Break-even speedup |
|---|---|---|
| H100 | $3.49 | 1.67x |
| H200 | $4.59 | 2.20x |
| B200 | $6.79 | 3.25x |

Today the B200 measures 1.2-2.1x. SF-Wan live on the H100 measures about
1.57x (23.8 fps against 15.2 fps). The work below aims to push H100 past
break-even and roughly double the B200's speedup.

**Top 3 packages to start** (confirm against the profile):

1. **WP-0: datacenter quick wins** (1 session, about $5 of GPU).
   - Let `auto` time cuDNN's unified SDPA against `attn_dc` on 9.0 / 10.0.
     cuDNN measured 1.35-1.49 PFLOPS on B200, against 0.96-1.17 for dc (same shapes, `FVID-2026-09-27-attention-datacenter`).
   - Send d = 64 (the LTX audio stream) and d = 72 (the Qwen-VL vision
     tower) to cuDNN rather than mma.sync V2 or the materialised path.
   - Cache the cuBLASLt matmul descriptors and algorithms per shape.
     `quant.rs` builds and queries them on every call.
2. **WP-F: LTX host-gap removal** (2-3 sessions, about $6).
   - LTX kept the B200 only 29-38% busy, against 77-92% for H3 and Wan.
     Until that gap closes, kernel work on LTX mostly moves idle time
     around.
   - The fix applies to every GPU, the PRO 6000 included.
3. **WP-D: VSA fine stage on tcgen05 (sm_100) and wgmma (sm_90)**
   (3-4 sessions, about $12-15).
   - H3 turbo is the bulk of traffic, and VSA dominates its denoise. On
     B200 the VSA op is estimated at ~45% of denoise **(profile)**.
   - The KV-tile-list TMA producer that WP-D adds to `attn_dc.cu` is the
     piece that Sol's exact blocks (WP-B / WP-C) and block-causal
     attention (WP-E) reuse.

WP-G (VAE conv3d, channels-last bf16) may move up to second place. On B200
the Wan 5B VAE decode is 73-75% of the run, and the H3 decode is 20-24%
**(profile)**.

---

## 1. Dispatch map

Legend:

- **ours-dc**: arch-specific datacenter kernel (`attn_dc.cu`, built for
  `sm_90a` / `sm_100a`).
- **ours-mma**: mma.sync / cp.async / TMA kernel in `kernels.cu`, built as
  a generic `sm_90` / `sm_100` / `sm_120` cubin.
- **cuDNN / cuBLAS / cuBLASLt**: the vendor library picks the SASS
  internally, so it gets wgmma / tcgen05 on datacenter parts.
- **rule**: a fixed choice by SM or shape.
- **timed**: the first call per shape times the candidates and keeps the
  winner.

`kernels.cu` is compiled for `sm_90` and `sm_100`, not `sm_90a` and
`sm_100a` (`build.rs:37-38`, `DC_ARCHS` vs `DEFAULT_SMS`). No kernel in it
can issue wgmma, tcgen05 or setmaxnreg; only `attn_dc.cu` can. The kernel
seam (`[kernels]` profile key, else env) is
`crates/fastvideo-models/src/techniques/kernels.rs:143-253`, and
`available()` at `:277` restricts `dc` to exactly 90 and 100.

### 1.1 Attention

| Op | sm_90 (H100/H200) | sm_100 (B200) | sm_120 (PRO 6000) | Pick | Where |
|---|---|---|---|---|---|
| Dense SDPA, d = 128, bf16 in (H3 dense layers and steps, Wan self/cross, LTX video, text refiners) | **ours-dc** `fa_dc90_fwd_d128` (wgmma, TMA, 3 warpgroups, 128 q/CTA): 565-649 TFLOPS | **ours-dc** `fa_dc100_fwd_d128` (tcgen05 + TMEM, 256 q/CTA): 964-1173 TFLOPS | **timed** cuDNN unified SDPA vs `flash_mma_fwd2` per `(bh, sq, sk, d)` | rule on 9.0 / 10.0 (`dc_default`); timed on 12.x | `wan/attn.rs:173` `flash_kernel_for`, `:198` `dc_default`, `:226` `cudnn_default` (sm 12 only), `:239` `auto_cudnn_or_v2`, `:334` `device_mma_sdpa_with` (cuDNN `:366`, dc `:378`/`:387`); `wan/attn_dc.rs:53` `dense`, `:74` `load`, `:167` `dense_fwd` |
| Dense SDPA, d != 128 (LTX audio d = 64; Qwen-VL vision d = 72) | ours-mma V2 (d = 64); d = 72: cuBLAS materialised ("device dense") | same as sm_90 | same (cuDNN timed only when V2 would run) | rule: `dense_fwd` refuses `d != 128` (`attn_dc.rs:179`) | `attn.rs:106` `mma_sdpa_supported`, `:644` `device_dense_sdpa` |
| Sol-Attn (H3 max, LTX-2.5 stage 2 layers 1-47) | ours-mma `sol_mma_fwd_x4f` (`ws` warp-specialised + KV splits exists, opt-in) | ours-mma `sol_mma_fwd_x4f` | ours-mma `sol_mma_fwd_x4f` | rule: `auto` = x4f everywhere (`sol_ws_default() = false`) | `wan/ops.rs:3402` `SolKernel`, `:3422` `sol_kernel_choice`, `:3443` `sol_ws_default`, `:3453` `sol_splits_for`, `:3542` pick in `sol_fwd_core`; `sol_attn.rs:43`; callers `ltx2/attention.rs:319`, `h3/transformer.rs:560` |
| Sol prep (K/V to bf16, pooled Kc/Vc, thresholds) | ours-mma, 3-4 SIMT launches | same | same | rule | `wan/ops.rs` `sol_prep_*` (`kernels.cu:2784-2888`) |
| VSA coarse (tile means + scores) | ours tile-mean kernels + **cuBLAS f32 (TF32)** GEMM | same | same | rule | `wan/ops.rs:838`, `:851`; `h3/vsa.rs` |
| VSA top-k | ours `vsa_topk2` (radix select) | same | same | rule (`FASTVIDEO_VSA_TOPK`) | `wan/ops.rs:877-893` |
| VSA fine | **ours-dc `fa_dc90_vsa`** (wgmma, KV-tile-list producer; WP-D) | **ours-dc `fa_dc100_vsa`** (tcgen05 + TMEM; log: `vsa fine kernel: dc KV-tile-list (sm100, tcgen05, cubin)`) | ours-mma `vsa_mma_attn_tma2` (TMA 128B swizzle, 3-slot ring, mma.sync; the 9.0 / 10.0 escape hatch `tma2`) | rule: `tma_requested` is `sm_major >= 9`, ring default on | `wan/ops.rs:1096` `vsa_mma_attn_device`, `:1292`, `:1348` `tma_requested`, `:1382` `vsa_ring_default`; `wan/vsa.rs:293` `fine_kernel`; `h3/vsa.rs:600`, `:729` |
| Block-causal full-sequence (causal Wan clip forward) | ours-mma `flash_mma_fwd2_causal_d{64,128}` | same | same | rule | `wan/attn.rs:548` `device_mma_sdpa_causal`; `wan/nn.rs:1754` |
| SF-Wan streaming (queries vs sink + rolling window, no mask) | **ours-dc** (plain dense via `sdpa_kv_window`, d = 128) | **ours-dc** | timed cuDNN / V2 | same as dense | `wan/nn.rs:1793` `sdpa_kv_window`, `:1749` `causal_flash_enabled`; `wan/transformer.rs:257`, `:290` |
| FP8 attention (Sage-style, opt-in, H3 `fp8_attention`) | ours `attn_fp8.cu` (mma.sync e4m3 QK, bf16 PV) | same | same | opt-in; `supported()` = sm >= 8.9 | `wan/attn_fp8.rs:79`, build.rs `fp8_entries` |

### 1.2 GEMMs

| Op | sm_90 | sm_100 | sm_120 | Pick | Where |
|---|---|---|---|---|---|
| BF16 linears (every DiT, text encoders, VAE 1x1) | cuBLAS `gemm_ex`, `CUBLAS_COMPUTE_32F_FAST_16BF` or bf16 operands; cuBLAS heuristic (wgmma kernels) | cuBLAS (tcgen05 kernels) | cuBLAS | cuBLAS default heuristic, no timing | `wan/device.rs:28-45` `GemmMath`, `:722`, `:861`, `:918`; `wan/bf16_gemm.rs` |
| H3 DiT FP8 (blocks 2..=46: fused QKV, out, FFN up/down) | **BF16 by default.** `FASTVIDEO_H3_QUANT=mxfp8` becomes **W8A8** per-tensor cuBLASLt (logged once) | **MXFP8** cuBLASLt `VEC32_UE8M0` (default) | **MXFP8** (default: the rule is `sm_major >= 10`) | rule | `wan/quant.rs:356` `from_env`, `:363` `for_device` (MXFP8 on < 10 becomes W8A8), `:382` `default_for_device` |
| Wan DiT FP8 (all attention + FFN linears) | BF16 (default off) | **MXFP8** (default: `sm_major == 10`) | BF16 (default off; the 1.3B A/B did not win speed and quality) | rule | `wan/quant.rs:478`, `:494` `default_mode` |
| FP8 GEMM call (either recipe) | cuBLASLt; **a new desc, layouts, preference and `AlgoGetHeuristic` on every call**, 1 algo, no timing | same | same | heuristic per call | `wan/quant.rs:1325` `lt_matmul`, `:1392`; linear forward `:1116-1172` |
| NVFP4 LTX video FFN (opt-in profile `ltx25_distill_sol_nvfp4`) | BF16 fallback (`tensor_cores()` is `sm_major >= 10`) | cuBLASLt `VEC16_UE4M3` block-scaled FP4 (plan built at load) | cuBLASLt FP4 (3.5-4x bf16 GEMM, 1.12-1.14x denoise; 4K sharpness gate fail) | rule | `wan/nvfp4_linear.rs:70`, `:134-230` |
| NVFP4 oxide Tile-IR GEMM | not built (`fv-oxide-aot --sm 100,120`) | cubins embedded, **off** (`FASTVIDEO_NVFP4_OXIDE_GEMM=0`) | embedded, off | seam `nvfp4_gemm` | `wan/ops.rs:2864`, `wan/nvfp4_gemm.rs:164-222`, `docker/gpucheck.Dockerfile:79`, `build.rs` oxide rows |
| FP8 text encoder (resident FP8 rows) | dequantized to bf16 per call, then cuBLAS | same | same | rule | `ops::fp8_rows_*` |

### 1.3 Elementwise, norms, RoPE, VAE, upsamplers

All of these are SIMT kernels in `kernels.cu`, with the same source on every
SM. They are bandwidth-bound, so HBM3/HBM3e speeds them up with no code
change.

| Op | All three archs | Where |
|---|---|---|
| H3 norm + modulation, residual + gate + norm, SwiGLU, qk-norm + RoPE (bf16-act; MXFP8 producers write the swizzled scales directly) | ours SIMT | `h3/fused16.rs:72` `norm_mod`, `:110` `res_gate_norm_mod`, `:197` `qk_norm_rope`, `:280` `swiglu_mx` |
| Wan AdaLN / residual-norm fusions, qk-norm + RoPE | ours SIMT (`FASTVIDEO_WAN_FUSE`) | `wan/fuse.rs:236`, `wan/fused.rs` |
| SF-Wan RebasedSink RoPE (re-rotates the **whole K window every block and layer**) | ours SIMT | `wan/causal.rs` `rope_bhsd`, `wan/transformer.rs:251-256` |
| LTX RoPE tables, AdaLN, prune gather/scatter | ours SIMT; prune reads `[S]` scores back to the host for the top-k | `ltx2/transformer.rs:1166` `gather_prune`, `:1243` `scatter_prune` |
| VAE conv3d (Wan 2.1/2.2 VAE, LTX VAE, LTX latent upsampler, H3 Spark latent upscaler) | **cuDNN legacy API, NCDHW**. The bf16 path casts x and **w** f32 to bf16 on every call, then casts back. Algorithm from the cuDNN heuristic; `auto` is deterministic (bf16 under bf16 math); `FASTVIDEO_CONV3D_TIMED=1` times instead | `wan/conv.rs:295` `cudnn_conv_bf16`, `:556` `conv3d`, `:581` rule; `ltx2/latent_upsampler.rs:16-36`; `h3/spark.rs` |
| TAEHV / TAEH3 tiny decoders | cuDNN conv2d + ours | `wan/taehv.rs` |
| SF-Wan CUDA graphs (5 forwards per block, captured once) | same code on every arch; B200 replayed 71 of 81 blocks | `wan/graph.rs:38` `graphs_enabled` |

---

## 2. Gaps on sm_90 / sm_100

Numbered for reference from the work packages.

- **G1. Dense attention is not at the vendor ceiling.**
  - B200: `fa_dc100` runs 0.96-1.17 PFLOPS against cuDNN 1.35-1.49. The
    kernel has one CTA per SM (197 KB smem), no 2-CTA (`cta_group::2`)
    MMA, no persistent or LPT scheduler, no split-KV and no exp2
    emulation on the FMA pipes (FA4's trick).
  - H100: `fa_dc90` runs 565-649 TFLOPS (two runs). cuDNN runs 587-675, so it is
    ahead on some shapes; FA3 is about 740.
  - `auto` never considers cuDNN on 9.0 / 10.0: `cudnn_default` is
    `sm_major == 12` (`attn.rs:226`).
- **G2. Dense attention at other head dims falls off the datacenter
  path.** `dense_fwd` takes d = 128 only (`attn_dc.rs:179`).
  - The LTX audio stream (d = 64) runs mma.sync V2 at about 1/3 of the
    tensor-core rate.
  - The Qwen-VL vision tower (d = 72) runs the materialised cuBLAS path.
- **G3. Sol has no datacenter kernel.**
  - `sol_mma_fwd_x4f` is mma.sync on every arch. B200 LTX 1080p20s at
    tau 1.0: 146 ms, against sol-engine's sm100 kernel at 87.8 ms (1.66x).
  - H100: sol-engine sm90 is 99.4-139 ms with kv_splits.
  - Structural gaps (`sol-engine-code-level.md` §2):
    - sol-engine computes the route GEMM and the approximate P·Vc on
      WGMMA/UMMA; ours uses mma.sync.
    - Prep takes 3-4 separate launches.
    - The `ws` variant loses ~10% on sm_120 (register cap), so no
      sm_90 / sm_100 default was ever measured for it.
- **G4. VSA has no datacenter fine kernel.**
  - `vsa_mma_attn_tma2` loads through TMA but computes with mma.sync, at
    about 330 TFLOPS on any Blackwell.
  - Coarse scoring is f32 tile means plus a TF32 cuBLAS GEMM.
  - In the served FastH3 4-step recipe, the attention inputs reach VSA as
    f32, so the bf16-in-place path (`FASTVIDEO_VSA_BF16`) does not engage
    (`attention-sm120-cudnn-vsa.md` §3).
- **G5. Hopper has no working FP8 path for the DiTs.**
  - MXFP8 needs sm_100: on sm_90 cuBLASLt has no `VEC32_UE8M0` kernels,
    and `for_device` (`quant.rs:363`) swaps to W8A8.
  - W8A8 per-tensor measured 16-20 dB and stays off
    (`FVID-2026-09-24-solh3-spark-vsa-bf16`).
  - So H100 and H200 run the H3 and Wan linears in BF16. That forgoes
    Hopper's 2x FP8 rate, the lever that gave 1.52x denoise on the PRO
    6000.
- **G6. Block-causal attention has no datacenter kernel.**
  `flash_mma_fwd2_causal` is mma.sync. The served SF-Wan stream does not
  use it: it takes the dense dc path over the KV window. The FastVideo
  full-sequence parity forward and any causal clip mode do.
- **G7. Small grids on streaming shapes.**
  - SF-Wan block: bh = 12 and sq = 4680 give 19 x 12 = 228 CTAs of 256
    queries on 148 B200 SMs, which is 1.54 waves. About 23% of the
    attention time is tail **(profile)**.
  - `dc` has no split-KV or smaller-tile variant for that case.
  - The same rule already exists for Sol (`sol_splits_for`).
- **G8. SF-Wan re-applies RoPE to the whole K window every block and
  layer** (`transformer.rs:251-256`, RebasedSink). That is O(window)
  bandwidth per layer, and it could be folded into the query rotation,
  since RoPE is relative **(profile)**.
- **G9. LTX is host-bound on B200** (29-38% util; denoise only 1.13x the
  PRO 6000). Candidates in the code:
  - a device synchronize per step (`ltx2/pipeline.rs:75` `step_sync`);
  - fbcache's `relative_l1`, which reads two f64 sums back per step
    (`ltx2/transformer.rs:386`, `wan/ops.rs:4570`);
  - token-prune `gather_prune`, which reads `[S]` scores to the host, runs
    the top-k there and uploads the indices (`transformer.rs:1166`);
  - per-call cuBLASLt planning (G11);
  - many small launches in the audio branch and Sol prep;
  - no graph capture of the stage-2 step.

  Which of these dominate is the profile's first question.
- **G10. VAE conv3d.**
  - The layout is NCDHW. Datacenter implicit-GEMM conv kernels want
    channels-last (NDHWC).
  - Every bf16 conv casts its weight f32 to bf16 on each call
    (`conv.rs:306-307`) and the activation in and out.
  - The algorithm comes from the legacy heuristic.
  - The B200 decode is 73-75% of Wan 5B runs (3.1 s at 480p, 7.1 s at
    704p) and 20-24% of H3 runs **(profile: conv share vs casts vs
    norms)**.
- **G11. cuBLASLt planning per call.** `lt_matmul` creates the matmul
  descriptor, three layouts and a preference, and asks for a heuristic on
  every FP8 GEMM (`quant.rs:1325-1442`). That is ~180+ calls per H3 step.
  It is small next to a 1 s step but adds to G9-style host gaps and to the
  capture cost.
- **G12. The NVFP4 route on sm_100 is unmeasured.** The LTX FFN profile
  was gated on the PRO 6000 only (1.12-1.14x denoise; the 4K sharpness
  gate failed). The B200 has twice the MXFP8 rate in FP4, but the
  quality question is the same.
- **G13. Build targets.** `sm_100a` cubins run only on 10.0.
  - B300 (10.3) would fall back to mma.sync. `attn_dc.cu` already guards
    `__CUDA_ARCH_FEAT_SM103_ALL`, but `DC_ARCHS` builds no `sm_103a` (or
    family `sm_100f`) cubin.
  - Tile-IR oxide cubins are built for 100 and 120 only. cutile supports
    sm_90 from CUDA 13.3, and the image ships 13.4.

---

## 3. Options per gap

### 3.1 Option catalogue

| Option | What it offers here | License | Build-system fit | Parity approach |
|---|---|---|---|---|
| **cuDNN unified SDPA** (already bound: `wan/cudnn_sdpa.rs`, raw FFI, cuDNN 9.26) | Dense bf16 fwd on sm_90 / sm_100 at vendor speed (B200 1.35-1.49 PFLOPS); any head dim it supports (64, 72, 128); causal / sliding-window masks through the SDPA attributes (not bound yet) | cuDNN EULA, redistributable; already shipped in the runtime image | Zero: same binding and plan cache as sm_120; only the `auto` rule changes | `attn3_parity` (vs f32 reference rel_l2 <= 4.3e-3 on sm_120); per-shape timed pick as on sm_12x |
| **cuDNN FP8 SDPA** (Hopper and Blackwell) | E4M3 Q/K/V with descale / scale tensors, about 1.5-2x the bf16 rate | same | The frontend builds FP8 SDPA as a graph; our backend-only binding would need the FP8 attributes. Whether cuDNN 9.26 still takes that graph (it refused our bf16 *composite* graph) is unknown: **probe first** | Lossy: kernel rel_l2 vs f32, then the quality gate |
| **cuDNN block-sparse** | None that fits VSA or Sol: masks are causal / band / padding / bias; a bias tensor costs dense time | — | — | — |
| **cuBLASLt FP8 on sm_90** | Per-tensor E4M3 (in use: W8A8). Block-scaled FP8 on Hopper (1x128 activation and 1x128 or 128x128 weight scale modes, f32 scales), added in cuBLAS 12.9; **verify the scale-mode enums and TN support against cuBLAS 13.8** | CUDA EULA, in image | The same `LtScale` enum gets a `Vec128` variant; the fused norm kernels emit f32 1x128 scales instead of E8M0 1x32 | Codes / scales bit-exact to a host twin; GEMM rel_l2 vs exact dequantized math; H3 5-prompt quality gate (lossy: a different recipe from Sol-H3's MXFP8) |
| **cuBLASLt MXFP8 / NVFP4 on sm_100** | In use (MXFP8 H3 / Wan; NVFP4 LTX FFN opt-in) | same | done | as `nvfp4_gemm` / `nvfp4_linear` groups |
| **CUTLASS 3.x / 4.x C++** (sm90 wgmma + TMA warp-specialised; sm100 tcgen05 + TMEM, 2-SM) | FP8 blockwise-scaled GEMM on sm_90 (examples 67/68, groupwise scale, K granularity = tile K); epilogue fusion (bias, GELU, **quantize-to-next-FP8** so the norm/quant pass disappears); implicit-GEMM conv fprop (sm90 / sm100, NDHWC, TMA im2col); Blackwell FMHA (example 77), Hopper FMHA | BSD-3 | Header-only, but its kernels take a Params struct holding TMA descriptors built on the host. Fit: a small C++ **host shim** compiled by build.rs (nvcc, `-std=c++17`, `third_party/cutlass` pinned by tag and sha256) exports `extern "C" fv_cutlass_<op>_params(...)` that fills a byte blob, and the kernel itself goes into the embedded cubin table (`DC_AOT`-style, `sm_90a` / `sm_100a`) and launches through the driver API (`cuLaunchKernelEx` with cluster dims). No cudart launches, no NVRTC. Build on the build pod (nvcc 13.4 is there) | kernels tier vs our mma.sync / cuBLAS oracle on the same GPU |
| **FlashAttention-3** (Hopper) | Dense bf16 ~740 TFLOPS and FP8 ~1.2 PFLOPS (d = 128); its mainloop is the model for `fa_dc90`, which already has FA3's intra-warpgroup overlap | BSD-3 | Same C++ shim route as CUTLASS (FA3 is CUTLASS-based); or port pieces (pingpong scheduling between 2 consumer warpgroups, FP8 path, split-KV) into `attn_dc.cu` | as `attn_dc` group |
| **FlashAttention-4 / CUTLASS Blackwell FMHA** (sm_100) | FA4 (CuTe DSL, Python) reaches cuDNN-class speed with 2 softmax warpgroups + a correction warpgroup, polynomial exp2 on FMA units and LPT scheduling | BSD-3 | CuTe DSL is Python; AOT to cubin is possible in CUTLASS 4.x (verify in the pinned version), else capture the JIT cubin on the build pod and embed it like the oxide cubins. Port the ideas into `fa_dc100` rather than the code | as `attn_dc` |
| **sol-engine sm90 / sm100 Sol kernels** (NVlabs/Sana `sol-engine` @ 6c2f582, CuTe DSL) | The reference Sol kernels: route WGMMA/UMMA, ballot masks, single online softmax over route + exact, kv_splits | Sana repo top-level license Apache-2.0; **check the sol-engine subtree headers** | CuTe DSL, same AOT caveat as FA4; or port the schedule into `attn_dc.cu` (preferred: our prep and layout differ) | `attn2_parity` (Sol vs host-faithful oracle), then oracle-tier LTX / Sol-H3 |
| **FastVideo VSA kernels** (`fastvideo-kernel`) | Their H100 fine stage is ThunderKittens (sm_90a); an sm100a tcgen05 body exists (`#if __CUDA_ARCH__ == 1000 && SM100_ALL`, noted in `FVID-2026-09-19-sm120-is-not-umma`) | Apache-2.0 (FastVideo), MIT (ThunderKittens) | TK needs nvcc `-std=c++20` and its headers; it builds through the same arch-specific module route. Their layout is tile-ordered padded q/k/v (ours matches after `tile+fine`) | `vsa_stages` / `vsa_*` groups: selected tile lists identical, output vs host reference rel_l2 (ours today 0.0026-0.0031) |
| **ThunderKittens** (standalone) | H100 and B200 attention / GEMM building blocks | MIT | as above | as above |
| **FlashInfer** block-sparse attention | Variable block-sparse (BSR) attention on its FA3 (sm_90) and CUTLASS (sm_100) backends | Apache-2.0 | JIT-oriented (PyTorch extension); AOT is possible but heavy; better as a speed reference in the upstream harness than a dependency | reference timing only |
| **Our own in `attn_dc.cu`** (hand-written PTX: TMA, mbarrier, wgmma, tcgen05, TMEM) | Already works for dense on both archs; extend with a **KV-tile-list producer** (the TMA warp reads the tile indices) to cover VSA, Sol exact blocks and block-causal ranges | ours | Fits today's build exactly (`DC_ARCHS`, NVRTC `compute_90a` / `compute_100a` fallback) | `attn_dc` group pattern: vs mma.sync oracle + f32 reference |
| **cuda-oxide** (Rust SIMT/tcgen05 via `docker/oxide.Dockerfile`, `artifacts/oxide`) | Has tcgen05, TMA and cluster primitives; its `gemm_sol` reached 58% of cuBLASLt SoL on B200; Tile-IR NVFP4 lost to cuBLASLt (1.4-2.3x vs 3.5-4x bf16) | ours / Apache-2.0 upstream | Nightly rustc + LLVM 21 image; cubins go through the manifest in `build.rs` | same groups |
| **Tile-IR (cutile-rs)** | Portable tile kernels; sm_90 supported since CUDA 13.3 | Apache-2.0 | `fv-oxide-aot --sm 90,100,120` is a one-line change in `docker/gpucheck.Dockerfile:79` plus `FV_REQUIRE_OXIDE` | `nvfp4_gemm` group |

What the catalogue says for the critical path:

- **For attention, extend `attn_dc.cu`.** It already has the TMA ring,
  mbarriers, warp specialisation and both MMA ISAs. Use FA3 / FA4 /
  sol-engine / FastVideo as references to port ideas from, not as
  vendored code. Their kernels assume layouts and preprocessing that
  differ from ours, and the host-shim route costs more than one kernel's
  worth of work.
- **For FP8 GEMM, use cuBLASLt first, and CUTLASS only for fused
  epilogues.**
- **For conv, use the cuDNN frontend graph (NDHWC) first, and CUTLASS
  conv only if cuDNN is slow.**
- **cuda-oxide and Tile-IR stay off the critical path.** They measured
  well below the vendor libraries. Use them only for small SIMT kernels
  where Rust-side authoring helps.

### 3.2 Per-gap recommendation

| Gap | First choice | Fallback | Not recommended |
|---|---|---|---|
| G1 dense ceiling | cuDNN in the timed pick on 9.0 / 10.0 (WP-0); then `fa_dc100` upgrades (2-CTA, exp2 emulation, persistent) in WP-C | FA3 pingpong in `fa_dc90` (WP-B) | vendoring FA4 CuTe DSL |
| G2 other head dims | cuDNN (d = 64, 72) in WP-0 | dc variants templated on d (64: TMEM columns halve) | — |
| G3 Sol | port sol-engine's schedule into `attn_dc.cu` (route UMMA/WGMMA + list producer for exact blocks, single softmax, splits) | enable `ws` + splits on sm_90 if it wins there (cheap test in WP-0) | AOT CuTe DSL sol-engine (layout mismatch, license check) |
| G4 VSA | list-producer dc kernel (WP-D); coarse in bf16 wgmma/cuBLAS | port FastVideo TK sm_90a / sm100a fine stage | FlashInfer as a dependency |
| G5 Hopper FP8 | cuBLASLt block-scaled 1x128 FP8 (WP-A) with a new fused producer | CUTLASS sm90 groupwise FP8 with quantizing epilogue; exact MXFP8 emulation (wgmma k = 32 + per-group promotion) only if the recipe must match Sol-H3 bit for bit | per-tensor W8A8 (failed quality) |
| G6 / G7 causal and small grids | list producer (a causal band is a contiguous range) + split-KV / 128-query tile for < 2 waves (WP-E) | cuDNN band mask if it expresses the frame-block causal pattern exactly (probe) | — |
| G8 RoPE window | rotate q by the rebase offset instead of re-roping K (WP-E) | fuse the re-rope into the attention producer | — |
| G9 LTX host gaps | device-side top-k / l1 decisions, remove per-step syncs, graph-capture stage-2 steps (WP-F) | — | — |
| G10 VAE conv | cuDNN frontend conv graph, NDHWC bf16, resident bf16 weights, fused bias + act (WP-G) | CUTLASS implicit-GEMM conv3d | — |
| G11 Lt planning | plan cache keyed by (m, n, k, types, scale mode, ldd) (WP-0) | — | — |
| G12 NVFP4 sm_100 | measure the existing profile on B200 (WP-H) | — | — |
| G13 targets | add `sm_103a` (or `sm_100f` if its feature set covers tcgen05: verify) to `DC_ARCHS`; add 90 to the oxide SM list only if a Tile-IR kernel is ever enabled | — | — |

---

## 4. Work packages

Execution order, by expected end-to-end gain per dollar, with the profile
allowed to reorder (section 6). Effort is in agent-sessions (one focused
session of build, host tests and a GPU run). GPU cost uses Runpod secure
prices: B200 $6.79/hr, H100 $3.49/hr, H200 $4.59/hr, RTX PRO 6000
$2.09/hr. A kernels-tier run costs 20-30 min of pod time. A 5-prompt gate
costs 45-75 min. Every package keeps the balance floor ($8, CLAUDE.md), and
every pod gets a backstop and is deleted when done.

| # | Package | Main GPUs | Effort | GPU $ | Depends on |
|---|---|---|---|---|---|
| 1 | WP-0 quick wins | B200, H100, PRO 6000 | 1 | ~5 | — |
| 2 | WP-F LTX host gaps | PRO 6000, B200 | 2-3 | ~6 | profile |
| 3 | WP-D VSA datacenter | B200, H100 | 3-4 | 12-15 | WP-0 |
| 4 | WP-G VAE conv3d | B200, H100, PRO 6000 | 2-3 | 6-8 | profile |
| 5 | WP-A Hopper FP8 GEMM | H100 | 2-3 | 8-12 | WP-0 (Lt plan cache) |
| 6 | WP-C Blackwell Sol + dense upgrades | B200 | 4-6 | 15-20 | WP-D (list producer) |
| 7 | WP-B Hopper Sol + dense upgrades | H100 | 3-5 | 10-12 | WP-D, WP-C design |
| 8 | WP-E SF-Wan causal + graphs | H100, B200 | 2-3 | 6-10 | WP-D |
| 9 | WP-H NVFP4 on sm_100 | B200 | 1-2 | ~8 | WP-F |

Total: 20-30 sessions and about $75-95 of GPU, spread so that no single
run needs more than ~$15.

### WP-0. Datacenter quick wins

- **Scope**
  - (a) `cudnn_default` covers 9.0 and 10.0 as well. `auto` on those SMs
    times cuDNN unified SDPA against `dc` per shape, like
    `auto_cudnn_or_v2`. Generalise it to `auto_cudnn_or(kernel)`, keep
    the same one-time timing, and log the pick.
  - (b) On 9.0 / 10.0, d != 128 goes to cuDNN when it has a plan (LTX
    audio d = 64, Qwen-VL d = 72), else V2 / the materialised path as
    today.
  - (c) A cuBLASLt plan cache in `quant.rs` `lt_matmul`, keyed by
    `(m, n, k, ab_type, scale kind, ldd, bias)`. It holds the desc,
    layouts and heuristic result; the scale pointers are set per call.
    `nvfp4_linear.rs` already caches at load; reuse its pattern.
  - (d) One kernels-tier timing of `sol ws` + `FASTVIDEO_SOL_SPLITS=auto`
    against `x4f` on H100 and B200. If ws wins, make it the sm_90 /
    sm_100 default (it was only ever measured on sm_120).
  - (e) Add `sm_103a` to `DC_ARCHS` (build-only; no B300 to test on).
- **Expected gain**
  - Dense attention up to 1.27x on B200 (cuDNN 1.40 vs dc 1.10 PFLOPS).
    It matters for dense layers and steps: H3 dense steps, Sol-H3 step 0,
    LTX layer 0 / text, Wan 5B.
  - The LTX audio attention gets ~2-3x.
  - H100 dense changes by ±5% per shape.
  - The Lt cache takes a few ms per step off every FP8 model.
  - Denoise gain is mostly under 5% except on dense-heavy recipes
    **(profile: dense-attention share per family)**.
- **Effort:** 1 session.
- **GPU cost / validation**
  - kernels tier `attn_dc,attn3_parity,attn3_bench,attn_bench,fp8_recipes`
    on B200 (~$3) and H100 (~$1.5);
  - `attn3_parity` + `fp8_recipes` on PRO 6000 (~$0.7);
  - no generation needed for (c) (exact: plan reuse is bit-identical).
    For (a) / (b), a 1-prompt H3-dense and LTX A/B on B200 (~$2) with
    off-identity by env (`FASTVIDEO_FLASH_KERNEL=dc`).
- **Dependencies:** none.

### WP-F. LTX host-gap removal (every GPU)

- **Scope.** Take the profile's host timeline for one LTX-2.5 1080p job and
  remove the top gaps. Candidates from the code:
  - keep the prune top-k on the device (`gather_prune`) and return only
    the counts;
  - fold fbcache's `relative_l1` decision into a device flag read once
    per step, or overlap its readback with the next step's launches;
  - drop `step_sync` outside timing mode;
  - batch the audio-branch small ops;
  - fuse the Sol prep launches;
  - then capture one stage-2 denoise step as a CUDA graph, reusing
    `wan/graph.rs` `GraphStream`. This needs persistent input / output
    buffers, a capture-safe Lt plan cache (WP-0 c) and no host decisions
    inside the step.
- **Expected gain.** On B200, util 30-38% to 75%+ could take the LTX
  denoise from 23.9 s toward the 12-15 s the kernels allow, up to 1.6-2x
  **(profile)**. On the PRO 6000, a smaller gain (the GPU is slower, so
  the gaps are a smaller share).
- **Effort:** 2-3 sessions.
- **GPU cost:** PRO 6000 traces and A/B (~$2), B200 confirmation (~$4).
- **Validation**
  - Exact kind: frames byte-identical to baseline with the same kernels
    (`off_identity` hard). If a change moves a rounding point (a device
    top-k with the same tie order must not), use the oracle tier on LTX
    512p instead.
  - Gate `performance` on `denoise_s` >= 1.10.
- **Dependencies:** the profile; WP-0 (c) for graphs.

### WP-D. VSA on datacenter GPUs (sm_100 first, then sm_90)

**Status (2026-09-29): landed, `auto` on 9.0 / 10.0** (seam value
`vsa_attention = dc`; `FASTVIDEO_VSA_KERNEL=tma2` restores the mma.sync
kernel; sm_120 never loads the module). Numbers in
`docs/perf/datacenter-profile.md`, "WP-D results".

- **What was built** (`attn_dc.cu`, `attn_dc.rs`)
  - A **KV-tile-list producer**: the consumers read an ordered list of
    32-bit entries (tile index, valid keys, per-64-row-group visibility
    mask) and the TMA warp streams two 64-key tiles (128 keys) per softmax
    step, ring order K_a K_b V_a V_b. Builders produce the list:
    `dcv_build_union` (sm_100) and `dcv_list_sel` (sm_90). Sol's exact
    blocks (per-group ballot masks) and block-causal key ranges (contiguous
    tiles, frame boundary as `valid` / group mask) are meant to be further
    builders over the same entry word and pipelines (WP-B / C / E).
  - `fa_dc100_vsa` (tcgen05 + TMEM): 128 query rows per CTA = two VSA
    query tiles (M = 128, the layout the dense kernel proves), attending the
    **union** of their selections, each row masking the tiles its own tile
    did not select (exactly per-tile VSA). Warps 0-3 softmax, 4 list + TMA,
    5 MMA; S double-buffered in TMEM so QK_{j+1} overlaps softmax_j.
  - `fa_dc90_vsa` (wgmma): three warpgroups as `fa_dc90`; each consumer
    warpgroup owns one query tile (M = 64) and its own selection and K/V
    ring, so no union work is wasted. Every wgmma is unconditional (a
    partial last step repeats tile a, masked), which keeps ptxas from
    serialising them (C7519).
  - `dcv_prep_f32` / `dcv_prep_b16`: one launch tiles q/k/v and pools
    their means, bit for bit `vsa_tile_qkv` + `vsa_tile_mean`, so the f32
    scores and the selection are unchanged (item 3 below became "keep the
    f32 coarse stage, fuse its inputs": a bf16 coarse GEMM would move the
    selection).
  - The fine epilogue can write H3's combine
    `bf16(bf16(sparse) + bf16(bf16(coarse) * gate))` straight into token
    order (bit for bit `vsa_combine(round16)` / `vsa_combine_g16`): no
    `sparse` buffer, no combine launch. H3 (`h3/vsa.rs`
    `attend_device_dc`) takes prep + fine + combine; Wan's VSA and every
    other `vsa_mma_attn_tiled_device` caller take the dc fine stage alone.
  - `sm_103a` cubin built beside `sm_100a` (loaded on 10.3 only with
    `FASTVIDEO_DC_SM103=1`: never run on a B300).
- **Validation**: kernels group `vsa_dc` (99/99 with `attn_dc` on B200
  and H100); E2E H3 turbo on B200 (768p / 1080p, same pod, tma2 vs dc) with
  block dumps and clip comparisons; see the profile doc.
- **Measured vs the estimate**: B200 denoise −22.6% (768p) / −25.3%
  (1080p) against the estimated −24% / −32% of the run. The sm_100 union
  costs 1.25-1.42x the per-tile work on real H3 selections (median 1.33).
- **Next** (not done): sm_100 without the union waste (M = 64 tcgen05, or
  per-row-group lists sharing only common tiles); the sm_90 kernel is
  L2-bandwidth-bound (each warpgroup streams its own K/V: 64 KB per
  64 x 128 step, twice dense's bytes per FLOP), so share tiles both
  warpgroups selected; FA4's partial exp2 emulation on sm_100.

Original scope:

- **Scope**
  - (1) Add to `attn_dc.cu` a **KV-tile-list TMA producer**. The producer
    warp reads `selected[q_tile][0..topk]` and issues TMA loads for those
    64-token K/V tiles, which reuses the dense consumers unchanged.
  - (2) `fa_dc100_vsa` (tcgen05, 128 or 256 queries per CTA over 64-key
    tiles, TMEM S/O) and `fa_dc90_vsa` (wgmma). Include the gate multiply
    and the combine epilogue if the profile shows combine is material.
  - (3) Coarse stage in bf16 on tensor cores: bf16 tile means, and a
    cuBLAS bf16 GEMM or a dc-style kernel. The f32 selection must stay
    identical (the H3 port pins the pooled scores to f32, so keep f32
    accumulation and compare index lists).
  - (4) Make the served H3 path deliver bf16 q/k/v to VSA, so the tiling
    stops widening (G4).
  - `fine_kernel` / `tma_requested` get a `dc` value on 9.0 / 10.0 only.
- **Expected gain**
  - Fine stage from ~330 TFLOPS to 800-1000 on B200 (dense dc reached
    1.1 PFLOPS; sparse tiles lose some to load imbalance).
  - H3 turbo 768p: VSA ≈ 45% of denoise, estimated as follows. At 768p
    the per-call op is ~26.8 ms on the PRO 6000, and mma.sync on the B200
    runs at the same rate, so it is ~5 s of an 11.9 s denoise
    **(profile)**. A 2.5x fine stage saves ~2.5-3 s, which is about 15-18%
    of the run.
  - Larger at 1080p (33 s denoise).
  - H100: similar share, with a wgmma fine stage at ~2x.
- **Effort:** 3-4 sessions (producer + sm_100 first; sm_90 second).
- **GPU cost:** B200 kernels tier twice (~$7), H100 once (~$2), one H3
  turbo 5-prompt gate on B200 (~$7), plus a PRO 6000 regression tier
  (~$0.7).
- **Validation**
  - `vsa_stages` and new `vsa_dc_*` checks: identical `selected` lists.
  - Fine output vs the mma.sync oracle rel_l2 <= 1e-3, and vs the host
    reference <= 3.5e-3 (today's 0.0026-0.0031).
  - Oracle tier `fasth3-4step-vsa` on B200: per-block step-1 rel-L2
    within 10% of the mma.sync kernel's.
  - Gate: H3 turbo, `kind = exact` with PSNR telemetry. H3 moves ~20 dB
    on 1-ulp changes, so sharpness / jitter / LPIPS decide.
- **Dependencies:** WP-0 (so dense layers are settled before re-measuring).

### WP-G. VAE conv3d, channels-last bf16 (every GPU)

- **Scope**
  - (1) Keep the VAE weights resident in bf16 (no per-call weight cast).
  - (2) Store VAE activations as bf16 NDHWC between convs, with the
    GroupNorm / SiLU / residual kernels made layout-aware.
  - (3) Bind the cuDNN frontend conv fprop graph (backend `CONVOLUTION_FORWARD`
    op with NDHWC bf16 descriptors, fused bias + SiLU pointwise where
    cuDNN takes it), with a plan cache per shape.
  - (4) Keep `auto` deterministic: one engine per shape, chosen by
    heuristic, never by timing (the `conv.rs` header explains why).
  - (5) The LTX latent upsampler and the H3 Spark upscaler ride along.
- **Expected gain.** On B200, Wan 5B decode is 3.1 s of a 4.3 s run at
  480p: a 2x decode would make the run ~1.55x faster. The H3 decode is
  1.6-9 s (20-24%). The PRO 6000 benefits too **(profile: conv vs cast vs
  norm share of decode)**.
- **Effort:** 2-3 sessions.
- **GPU cost:** B200 + H100 + PRO 6000 decode A/B (~$6-8).
- **Validation**
  - Per-conv rel_l2 vs cuDNN f32 <= 5e-3.
  - VAE decode PSNR vs today's bf16 path >= 45 dB, and vs the reference
    VAE at today's level: the Wan 2.2 oracle's 65.7 dB / rel-L2 1.8e-3,
    `docs/oracle.md` "Wan 2.2 TI2V-5B modules".
  - The gate with `kind = lossy` quality checks (bf16 vs f32 activations
    moves rounding points).
- **Dependencies:** the profile (to size it).

### WP-A. Hopper FP8 GEMM

- **Scope**
  - (1) Probe cuBLAS 13.8 on H100 for block-scaled FP8. The candidate is
    1x128 f32 scales on activations and 1x128 or 128x128 on weights, TN,
    bf16 out, for the H3 and Wan shapes.
  - (2) Add a `Blk128` recipe to `quant.rs`. It is a third `QuantKind`
    with a host twin; its scales are the per-128 amax / 448 in f32,
    rounded to a power of two if we want it close to MX.
  - (3) Fused producers in `h3/fused16.rs` / `wan/fuse.rs` write codes
    plus f32 scales.
  - (4) `for_device` on 9.x maps `mxfp8` to `blk128` instead of W8A8. The
    default stays BF16 until the gate passes.
  - (5) If cuBLASLt lacks the mode or is slow, use a CUTLASS sm90
    groupwise-scaled FP8 GEMM (examples 67/68) through the host-shim
    route, whose epilogue can also emit the next layer's FP8
    activations.
- **Expected gain.** On the PRO 6000, MXFP8 gave H3 1.52x denoise. Hopper
  FP8 is 2x bf16 peak, so for H3 turbo expect 1.3-1.45x denoise on H100 /
  H200 (VSA and attention are unchanged). For Wan 5B on H100, 1.1-1.2x
  (**profile: GEMM share on sm_90**).
- **Effort:** 2-3 sessions (cuBLASLt route); +2 if CUTLASS is needed.
- **GPU cost**
  - H100 kernels tier (`fp8_recipes` + new `blk128` group, ~$1.5);
  - an H3 turbo 5-prompt gate on H100 (baseline BF16 vs blk128, ~$5-6);
  - a Wan 5B 5-prompt gate (~$3).
- **Validation**
  - Codes and scales bit-exact to the host twin.
  - GEMM rel_l2 vs exact dequantized math <= 2e-3.
  - Per-linear PSNR vs bf16 >= 35 dB.
  - The lossy gate with LPIPS required, sharpness 0.95-1.08, jitter
    0.85-1.20, and PSNR / LPIPS telemetry (>= 20 dB, mean <= 0.20).
  - Comparing against MXFP8 on the PRO 6000 with the same prompts tells
    whether blk128 is as good as the recipe it stands in for.
- **Dependencies:** WP-0 (c), the Lt plan cache.

### WP-C. Blackwell (sm_100) Sol and dense upgrades

- **Scope**
  - (1) `fa_dc100_sol`. Q·Kc^T route scores go into TMEM on tcgen05, then
    a per-row threshold and a ballot mask. The approximate term P·Vc is
    one UMMA per block group. Exact blocks go through the WP-D list
    producer. One online softmax spans route and exact.
  - (2) Fuse the 3-4 `sol_prep_*` launches into one or two.
  - (3) Split-KV by sol-engine's rule (the grid is under two waves, or
    tokens >= 64k).
  - (4) Dense upgrades to `fa_dc100`, each behind its own switch and
    measured alone:
    - 2-CTA `cta_group::2` MMA (256-row tiles over a CTA pair);
    - polynomial exp2 on the FMA pipes for part of each row (FA4);
    - a persistent tile scheduler.
    - Target: cuDNN parity, which would let WP-0's timed pick choose
      ours.
- **Expected gain**
  - Sol 1.66x: 146 to ~88 ms at LTX 1080p20s tau 1.0, the sol-engine
    number on the same GPU. That covers LTX stage 2 (after WP-F) and H3
    max steps 1-3.
  - H3 max 1080p denoise 34.4 s; the Sol share there comes from the
    profile.
- **Effort:** 4-6 sessions.
- **GPU cost:** B200 kernels tier x3 (~$10), an LTX and H3-max gate on
  B200 (~$8).
- **Validation**
  - `attn2_parity`-style Sol checks vs the host-faithful oracle: route
    masks identical; output rel_l2 vs `sol_mma_fwd_x4f` <= 1e-3.
  - Oracle tier `ltx25-512p` on B200 within 10% of the x4f per-block
    numbers.
  - Gate `kind = exact` (PSNR telemetry, sharpness / jitter / LPIPS
    hard).
- **Dependencies:** WP-D (list producer), WP-F (so the gain is visible
  on LTX).

### WP-B. Hopper (sm_90) Sol and dense upgrades

- **Scope**
  - `fa_dc90_sol` with the same structure as WP-C on wgmma (route WGMMA
    in registers, as sol-engine's `sm90/mainloop.py`), with kv_splits 2 /
    4.
  - `fa_dc90` pingpong: 2 consumer warpgroups alternating softmax and
    GEMM (FA3), plus split-KV for small grids.
  - An FP8 dense option (FA3's FP8 path) later, behind the
    `fp8_attention` technique.
- **Expected gain**
  - Sol on H100: from mma.sync x4f (not measured on H100: **profile**) to
    near sol-engine sm90 (99-139 ms at 1080p20s / 4K).
  - Dense +10-15% toward FA3.
- **Effort:** 3-5 sessions.
- **GPU cost:** H100 kernels tier x3 (~$5), gates (~$6).
- **Validation:** as WP-C, on H100 (and one H200 spot check: same SM,
  different clocks and HBM).
- **Dependencies:** WP-D producer; WP-C's design, shared in the header
  comment.

### WP-E. SF-Wan causal and CUDA graphs on sm_90 / sm_100

- **Scope**
  - (1) A small-grid rule for `dense_fwd` (G7): a 128-query CTA variant
    on B200 and split-KV with an LSE merge when
    `ceil(sq / rows) * bh < 2 * SMs`.
  - (2) `fa_dc*_causal`: block-causal as at most two contiguous key
    ranges per CTA through the list producer (G6). It is the parity
    forward, not the served stream, so it is low priority inside the
    package.
  - (3) RebasedSink: rotate the query by the rebase offset instead of
    re-roping the whole K window every block (G8). Split sink and window
    keys into two segments when their offsets differ, with the LSE merge
    from (1).
  - (4) Confirm graph capture on H100: dc, cuDNN and Lt calls inside
    capture, the replay ratio, and memory flat over 60 s.
- **Expected gain.** SF-Wan block time on B200 was 0.414 s (26.9 fps);
  the attention tail and re-rope could be 10-20% **(profile)**. On H100
  (23.8 fps reference), crossing the 1.67x break-even against the PRO
  6000 needs ≥ 25.4 fps (1.67 x 15.2 fps), about 7% more.
- **Effort:** 2-3 sessions.
- **GPU cost:** H100 + B200 `fv-gpucheck wan stream` 60 s runs plus a
  kernels tier (~$6-10).
- **Validation**
  - (1) and (2) as `attn_dc`.
  - (3) is not bit-identical (keys are rounded at a different position),
    so the SF-Wan oracle (`sfwan13` target: `sf_c<c>_s<i>_*`,
    `..._b<l>_attn_x`) must stay at its current rel-L2 on the same GPU,
    and a 2-min stream must pass the R12 visual check.
  - Memory flat; fps >= baseline + 5%.
- **Dependencies:** WP-D producer (for 2).

### WP-H. NVFP4 on sm_100

- **Scope**
  - Run the existing `ltx25_distill_sol_nvfp4` profile on B200.
  - Measure the FFN linears (the FP4 rate is 2x MXFP8) and the gate at
    1080p and 4K.
  - If LTX passes on B200, evaluate H3 FFN NVFP4 (TE `static_6`) the
    same way.
  - Keep the default off unless the gate passes.
- **Expected gain:** 1.1-1.2x LTX denoise on B200 (PRO 6000: 1.12-1.14x)
  **(profile: FFN share on B200)**.
- **Effort:** 1-2 sessions.
- **GPU cost:** ~$8 (two gates on B200).
- **Validation:** `nvfp4_gemm` / `nvfp4_linear` groups, then the lossy
  gate. The PRO 6000 run failed on 4K sharpness (1.098 > 1.08), which
  must now pass.
- **Dependencies:** WP-F (so LTX GPU time is the bottleneck).

---

## 5. Validation protocol

### 5.1 Tiers, in order, per package

1. **Host (build pod, no GPU).**
   - `cargo test -p fastvideo-cudarc --lib`, the `fastvideo-models`
     techniques / kernels registry tests, and every host twin (quant
     recipes, VSA selection, Sol oracle).
   - `cargo check --features fastvideo-cudarc/cuda`.
   - `nvcc` must build the new `sm_90a` / `sm_100a` cubins. The build pod
     has CUDA 13.4, so a ptxas error shows up before any pod is rented.
   - Never `cargo fmt`.
2. **Kernels tier on the target GPU(s).**
   `RUNPOD_NO_VOLUME=1 FV_EXTRA_ENV=FV_KERNEL_GROUPS=<groups> scripts/gpu/runpod-http.sh kernels <sha>`.
   The harness image must carry the new cubins (`Cubin(90)` /
   `Cubin(100)` in the log, not NVRTC).
3. **Oracle tier** (`scripts/gpu/oracle.sh`, `docs/oracle.md`).
   - For attention and conv changes on a served model.
   - Run on the target GPU. The recorded oracle numbers are sm_120, so
     first record the incumbent kernel's numbers on that GPU.
4. **Gate** (`fv-gpucheck gate`, `scripts/gpu/gate-policy.toml`).
   - 5-prompt set, warm, baseline vs candidate on the same pod and image,
     with the technique switched by env for `off_identity`.

### 5.2 Tolerances

| Change | Kernel tier | Oracle tier | Gate |
|---|---|---|---|
| Same arithmetic, different schedule (plan cache, launch fusion, host-gap removal) | bit-identical to incumbent | — | `kind = exact`, off_identity byte-identical, `denoise_s` >= 1.03 (experimental) / 1.10 (promotion) |
| New dense attention kernel (dc variants, cuDNN pick) | vs mma.sync oracle on the same GPU rel_l2 <= 1e-3 (dc100 reached 7.7e-4); vs f32 reference <= 1e-2 or <= V2's error + 5% (`FVID-2026-09-27-attention-datacenter-sm90`); bf16 out = RNE(f32 out) | step-1 per-block rel-L2 at the bf16 floor (~2e-3), smooth with depth, within 10% of the incumbent kernel's per block | `kind = exact`; hard: sharpness 0.95-1.08, temporal jitter 0.85-1.20, LPIPS available; PSNR telemetry (H3 moves ~20 dB on 1-ulp changes, so PSNR is not a gate for H3) |
| Sparse attention (VSA, Sol, causal) | selection (tile lists, route masks) **identical** to the incumbent; output as the dense row, vs host reference <= 3.5e-3 (VSA) | as above, targets `fasth3-4step-vsa`, `ltx25-512p`, `sfwan13` | as above |
| FP8 / FP4 recipe (WP-A, WP-H) | codes and scales bit-exact to the host twin; GEMM vs exact dequantized math rel_l2 <= 2e-3; per-linear PSNR vs bf16 >= 35 dB | not required (lossy by design) | `kind = lossy`: hard quality gates as above, LPIPS mean <= 0.20 and PSNR >= 20 dB recorded; compared side by side with the MXFP8 result of the same prompts |
| VAE conv (WP-G) | per-conv vs cuDNN f32 rel_l2 <= 5e-3 | decode vs reference VAE at today's level (Wan 2.2: 65.7 dB / 1.8e-3) | `kind = lossy` if the activation dtype changes, else exact |

### 5.3 Benchmark matrix

Denoise and run medians of 3 warm jobs, with the same image on both arms.

| Package | B200 | H100 (H200 spot) | PRO 6000 (regression) |
|---|---|---|---|
| WP-0 | H3 dense 768p, H3 max 768p, LTX 1080p, Wan 5B 480p; `attn_bench` shapes | same | `attn3_bench` shapes unchanged; H3 turbo 768p |
| WP-F | LTX 720p / 1080p / 1440p, util from `nvidia-smi` | LTX 1080p | LTX 1080p (must not regress; expected faster) |
| WP-D | H3 turbo 480p / 768p / 1080p / 1080p 10 s | H3 turbo 768p / 1080p | H3 turbo 768p (path unchanged) |
| WP-G | Wan 5B 480p / 704p, H3 turbo 768p / 1080p, LTX 1080p decode | same | same (shared code) |
| WP-A | — | H3 turbo 768p / 1080p, Wan 5B 480p | — (sm_120 keeps MXFP8) |
| WP-C | LTX 1080p / 1440p, H3 max 768p / 1080p | — | LTX 1080p, H3 max 768p (path unchanged) |
| WP-B | — | LTX 1080p, H3 max 768p | — |
| WP-E | SF-Wan 60 s stream, fps / p50 / p90, memory | same | SF-Wan stream (fps must not drop) |
| WP-H | LTX 1080p / 4K with the NVFP4 profile | — | — |

### 5.4 Not regressing sm_120

- **New kernels go in arch-specific modules only.** That means
  `attn_dc.cu` or a sibling in `DC_ARCHS`, loaded on exactly 9.0 / 10.0
  (and 10.3 when built). The PRO 6000 never loads them, so its path is
  unchanged by construction.
- **`auto` rules key on exact SMs.**
  - New rules check `sm == 90` / `sm == 100`, not `sm_major >= 9`.
  - The existing `>= 9` rules (`tma_requested`, Sol `ws`) and `>= 10`
    rules (H3 MXFP8 on sm_120) are intentional. Leave them as they are,
    and add explicit datacenter arms before them.
- **Pin the choices in a golden test.**
  - A host unit test asserts the `auto` picks for sm 120 (dense, Sol,
    VSA, conv, quant defaults) from a table.
  - Any package that changes a sm_120 pick has to edit that table
    visibly, in its own commit with its own evidence.
- **Shared code runs the PRO 6000 checks.** WP-0 (c), WP-F and WP-G touch
  code that sm_120 also runs. Each of them runs:
  - the PRO 6000 kernels tier (`attn3_parity`, `fp8_recipes`,
    `vsa_stages`, ~$0.7);
  - one off-identity generation A/B on the PRO 6000 (~$1-2).
- **Every new default has an escape hatch.** Each gets a `[kernels]`
  value or env flag that restores the previous kernel (as `v2`, `x4f`,
  `tma2` do today), is listed in the seam registry
  (`techniques/kernels.rs`), and is recorded in `decision-log.md`.

---

## 6. What the profile must answer, and how it reorders the plan

| Profile number | If it says | Then |
|---|---|---|
| LTX host idle (gaps between kernels, syncs, D2H) per step on B200 | > 40% of step wall | WP-F stays #2; WP-C's LTX gain is deferred until after WP-F |
| | < 20% | WP-F drops below WP-C; LTX is kernel-bound (then Sol / FFN dominate) |
| VSA op share of H3 turbo denoise (B200, H100) | >= 35% | WP-D stays #3 |
| | < 20% | GEMMs dominate; on H100 move WP-A ahead of WP-D |
| VAE decode breakdown (conv vs cast vs norm) | conv >= 50% of decode | WP-G to #2 or #3 (Wan 5B e2e is decode-bound on B200) |
| Dense attention share (H3 dense, Sol-H3 step 0, Wan) | small | WP-0 (a) is a correctness-of-default change only; skip the dc100 upgrades in WP-C |
| GEMM share of H3 turbo on H100 | >= 35% | WP-A to #3 |
| Sol share of LTX stage 2 / H3 max (B200) | >= 40% after WP-F | WP-C ahead of WP-G |
| SF-Wan attention tail / re-rope share | >= 15% of block time | WP-E ahead of WP-B |
| Per-step cuBLASLt host time | > 2% | WP-0 (c) confirmed; also a graph-capture prerequisite |

## 7. Out of scope here, but on the end-to-end path

- **No NVENC on H100 or B200.** CPU x264 adds 0.9-6.1 s per clip on B200
  (`b200.md`), and live paths lose H.264. It is a serving choice, not a
  kernel, but it can outweigh several of the packages above at 1080p+.
- **Text encoders** (Qwen / Gemma): FP8 rows are dequantized to bf16,
  then run on cuBLAS. An uncached LTX prompt costs 8.7-8.9 s on B200. A
  resident FP8 GEMM there (cuBLASLt per-tensor FP8 works on sm_90 and
  sm_100) is a separate package if the profile shows uncached prompts
  matter.
- **Loader speed on the Wan trees** (93-100 s cold on B200) is a volume /
  loader matter.
