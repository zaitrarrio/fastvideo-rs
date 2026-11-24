# ADR-0024 step 2: what the cuDNN SDPA candidate is worth on sm_12x (2026-11-24)

Owner request 2026-11-24: "start step 2, run the sdpa_rule measurement pass".
This is the measurement that sets **D2** of
[ADR-0024](../adr/0024-driver-only-runtime.md): the slowdown the owner accepts
for dropping cuDNN's fused attention from `sdpa_rule`, the first of the two
cuDNN uses that keep the CUDA layer in the images (the other is the
convolutions, step 3).

## Where cuDNN attention runs today

Only on sm_12x (`sdpa_rule::cudnn_default`), and only for bf16-output dense
attention at `d = 128` with `sq >= 24576` and `24576 <= sk <= 128000` (the
fixed rule, `sdpa_rule::pick`). Every other GPU and every other shape already
runs the in-tree `flash_mma_fwd2`. So dropping the candidate costs nothing on
H100 / B200 / A100-class tiers, and on RTX PRO 6000 it costs the margins
below, on the rule's shapes only.

## The run

`fv-gpucheck kernels --groups sdpa_rule,attn3_bench` on one Runpod RTX PRO
6000 Blackwell Server Edition (97.9 GB, driver 595.112.07, $2.69/h, 2 min 20 s
of pod time), runtime image `fastvideo-rs-runtime:sha-7f864e9` (main at the
#250 merge; build id `88243501c9683b0c`, cuDNN 9.26 of `cuda-13.pins`), no
weights (`RUNPOD_NO_VOLUME=1`), `FV_CUDNN_BENCH_CFGS=2`, through
`scripts/gpu/runpod-http.sh kernels` with its new `FV_KERNEL_GROUPS` knob.
Records: `artifacts/runpod/rtx6000/7f864e9-11242037/` (`kernels.json`,
`kernels.out`, `box.txt`). bf16 in and out, median of three synchronized
calls after a warm-up; `margin` is `(fwd2 - cuDNN) / max`, positive when
cuDNN is faster.

### `sdpa_rule`: cuDNN unified plan against `flash_mma_fwd2`, per shape

| shape | bh | sq | sk | fwd2 ms | cuDNN ms | plan | faster | cuDNN margin | rule picks | rule loss |
|---|---|---|---|---|---|---|---|---|---|---|
| ltx1080_s1_self | 32 | 9690 | 9690 | 4.206 | 3.896 | engine 11 | cudnn | +7.4 % | flash_mma_fwd2 | 8.0 % |
| ltx1080_s1_text | 32 | 9690 | 1024 | 0.553 | 0.558 | engine 8 | flash_mma_fwd2 | -0.9 % | flash_mma_fwd2 | 0.0 % |
| ltx1080_s1_a2v | 32 | 9690 | 151 | 0.202 | 0.211 | engine 8 | flash_mma_fwd2 | -4.4 % | flash_mma_fwd2 | 0.0 % |
| ltx1080_s2_self | 32 | 38760 | 38760 | 65.598 | 64.100 | engine 11 | cudnn | +2.3 % | cudnn | 0.0 % |
| ltx1080_s2_text | 32 | 38760 | 1024 | 1.912 | 1.907 | engine 8 | cudnn | +0.3 % | flash_mma_fwd2 | 0.0 % |
| ltx1080_s2_a2v | 32 | 38760 | 151 | 0.539 | 0.555 | engine 8 | flash_mma_fwd2 | -2.8 % | flash_mma_fwd2 | 0.0 % |
| ltx1080_audio_text | 32 | 151 | 1024 | 0.093 | 0.080 | engine 8 | cudnn | +13.6 % | flash_mma_fwd2 | 0.0 % |
| ltx1080_v2a | 32 | 151 | 38760 | 1.383 | 1.401 | engine 8 | flash_mma_fwd2 | -1.3 % | flash_mma_fwd2 | 0.0 % |
| ltx720_s2_self | 32 | 17480 | 17480 | 13.687 | 12.629 | engine 11 | cudnn | +7.7 % | flash_mma_fwd2 | 8.4 % |
| ltx512p | 32 | 6144 | 6144 | 1.947 | 1.937 | engine 11 | cudnn | +0.5 % | flash_mma_fwd2 | 0.0 % |
| h3_768p | 56 | 37966 | 37966 | 112.292 | 107.161 | engine 11 | cudnn | +4.6 % | cudnn | 0.0 % |
| h3_1080p | 56 | 76150 | 76150 | 450.154 | 431.484 | engine 11 | cudnn | +4.1 % | cudnn | 0.0 % |
| wan_580p | 24 | 23616 | 23616 | 18.293 | 18.338 | engine 11 | flash_mma_fwd2 | -0.2 % | flash_mma_fwd2 | 0.0 % |
| wan_720p | 24 | 27280 | 27280 | 24.665 | 22.776 | engine 11 | cudnn | +7.7 % | cudnn | 0.0 % |
| wan_cross | 24 | 27280 | 512 | 0.609 | 0.609 | engine 8 | flash_mma_fwd2 | -0.1 % | flash_mma_fwd2 | 0.0 % |
| edge_24k_lo | 32 | 24330 | 24330 | 25.949 | 23.967 | engine 11 | cudnn | +7.6 % | flash_mma_fwd2 | 8.3 % |
| edge_24k_hi | 32 | 24822 | 24822 | 27.201 | 25.191 | engine 11 | cudnn | +7.4 % | cudnn | 0.0 % |
| ltx_1080p20s | 32 | 124440 | 124440 | 685.957 | 658.060 | engine 11 | cudnn | +4.1 % | cudnn | 0.0 % |
| ltx_4k5s | 32 | 130560 | 130560 | 754.713 | 760.200 | engine 11 | flash_mma_fwd2 | -0.7 % | flash_mma_fwd2 | 0.0 % |

rule: sm_12x d=128 sq 24576..=inf sk 24576..=128000 -> cudnn | worst rule loss 8.4 %
suggestions: ['ltx1080_s1_self (sq 9690, sk 9690): cudnn is 8.0 % faster', 'ltx720_s2_self (sq 17480, sk 17480): cudnn is 8.4 % faster', 'edge_24k_lo (sq 24330, sk 24330): cudnn is 8.3 % faster']

### `attn3_bench`: the dense kernels side by side (ms)

| shape | bh | tokens | fwd2 | fwd3 | fwd3s | cuDNN unified cfg0 | cfg1 | cuDNN softmax graph | auto (the rule) |
|---|---|---|---|---|---|---|---|---|---|
| h3_768p | 56 | 37710 | 107.4 | 110.5 | 109.4 | 104.3 | 110.7 | 105.0 | 104.1 |
| ltx_512p | 32 | 6144 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 1.9 |
| ltx_1080p20s | 32 | 124440 | 677.4 | 694.1 | 690.5 | 652.8 | 696.8 | 671.0 | 651.9 |
| ltx_4k5s | 32 | 130560 | 752.3 | 770.6 | 767.3 | 760.6 | 754.1 | 734.2 | 755.0 |

(`fwd3` and `fwd3s` are the opt-in variants, still 2-3 % slower than `fwd2`;
the composite graph has no executable plan on this cuDNN; the softmax graph
is excluded from `auto`. `auto` is the rule's pick.)

## Reading

- **cuDNN is 4-8 % faster than `fwd2` on the long self-attention shapes**
  (H3 768p +4.6 %, H3 1080p +4.1 %, LTX 1080p 20 s +4.1 %, Wan 720p +7.7 %,
  the 24k edge +7.4 to +7.6 %, LTX 720p stage 2 +7.7 %, LTX 1080p stage 1
  +7.4 %), within noise or slower on cross attention, short sequences and
  the 4K 5 s shape (-0.7 %). The picture of 2026-09-27
  (`docs/gaps/2026-09-27-attention-sm120-cudnn-vsa.md`) and 2026-10-24
  (execution plan B5 gates) holds, with cuDNN's lead a little wider on this
  driver.
- **The rule as shipped leaves 8 % on the table at three shapes** below its
  24576 boundary (`ltx1080_s1_self`, `ltx720_s2_self`, `edge_24k_lo`): the
  three `sdpa_rule` failures, known since 2026-10-24. Widening the rule is
  the opposite of step 2 and is not proposed.
- **What dropping the candidate costs at the step level.** Attention is one
  part of a denoise step; the generation A/B of 2026-09-27 measured the
  whole-step effect of these same kernel margins at about 3 %: FastH3 4-step
  dense 768p denoise 33.23 -> 32.28 s with cuDNN, LTX-2.5 1080p 20 s stage 2
  44.0 -> 42.7 s/step. So **step 2 costs about 3 % of denoise time on RTX PRO
  6000 at H3 768p / LTX 1080p, up to roughly 5 % where the kernel margin is
  7-8 % (Wan 720p, LTX 720p stage 2), and 0 % everywhere else** (other GPUs,
  cross attention, short shapes, 4K).

## D2: the owner's call

Options, in the order recommended:

1. **Accept the 3-5 % and drop the candidate now** (`CUDNN_DEFAULT_ON =
   false`, the `Cudnn` pick and `cudnn_sdpa.rs` deleted, `FASTVIDEO_FLASH_KERNEL=cudnn`
   refused). Step 2 done; the attention path then has no cuDNN anywhere.
   The 3-5 % is recoverable later by kernel work on `fwd2` for sm_12x, which
   the `fwd3` / `fwd3s` attempts show is not trivial.
2. **Keep cuDNN attention until step 3 is ready and drop both together.**
   Nothing changes now; the image keeps the CUDA layer for the convolutions
   anyway, so step 2 alone frees no bytes. Costs nothing today, but the
   attention removal then rides a much larger change.
3. **Drop it and buy the margin back with SageAttention2 on sm_12x**, which
   is already default for `ltx-pro` and 1.5-1.8x faster than `fwd2`
   (`docs/perf/sage-attention.md`), for the models whose quality gate allows it
   (H3 does not pass it).

Not run: other tier GPUs (the rule never picks cuDNN there, so the cost is
zero by construction); a fresh generation A/B with cuDNN off (the 2026-09-27
A/B is the step-level evidence; a new one is a `fv-lab attn3` tier run, about
$3 and 40 min, if the owner wants the number refreshed before deciding).
