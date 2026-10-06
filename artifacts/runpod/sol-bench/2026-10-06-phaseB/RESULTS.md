| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |
|---|---|---|---:|---|---:|---:|---|---|
| SANA-Video 2B | 832x480x81, 50 st, cfg 6, baseline | 1x GB200 | — | RTX PRO 6000 (sm_120) | 186.04 | — | 32.65 / 2.58 / 179.59 / 2.25 | theirs: ratio only (2.77x) |
| SANA-Video 2B | same, EasyCache 0.1 + QKV merge + bf16 linear attn | 1x GB200 | — | RTX PRO 6000 (sm_120) | 87.93 | — | 32.31 / 2.59 / 81.53 / 2.17 | theirs: ratio only (2.77x) |
| Wan2.1 T2V-1.3B | 832x480x81, 50 st, CFG 6, base (median of 5 prompts) | - | — | RTX PRO 6000 (sm_120) | 92.22 | — | 115.70 / 0.03 / 89.16 / 3.02 | no published number |
| Wan2.1 T2V-1.3B | same, EasyCache 0.036 + Sol-Attn | - | — | RTX PRO 6000 (sm_120) | 29.09 | — | 118.31 / 0.03 / 26.02 / 3.03 | no published number |
| Wan2.2 T2V-A14B | 1280x720x81, 40 st, CFG 4/3, base (expert swap) | 1x GB200 | 449.67 | — | — | — | — / — / — / — | not run: exit 137 |
| Wan2.2 T2V-A14B | same, EasyCache + PISA | 1x GB200 | 207.01 | — | — | — | — / — / — / — | theirs also: kernel fusion; not run: exit 137 |
| LTX-2.3 HQ | 1920x1088x241, res2s 15 + 3, dense stage 2 | 1x GB200 | — | — | — | — | — / — / — / — | theirs: ratio only (2.40x); not run: exit 2 |
| LTX-2.3 HQ | same, SCSP + PISA s2 + midpoint prune + NVFP4 FFN | 1x GB200 | — | — | — | — | — / — / — / — | theirs: ratio only (2.40x); not run: exit 2 |
| LingBot-Video MoE | same, EasyCache + refiner PISA | 4x GB200 | 144.36 | — | — | — | — / — / — / — | theirs: 4 GPUs (CP4); not run: exit 2 |
| Cosmos3-Super 64B | 1280x720x189, 35 st, CFG 6 | 4x GB200 | 130.41 | — | — | — | — / — / — / — | theirs: 4 GPUs (SP); not run: exit 2 |
| Cosmos3-Super 64B | same, TeaCache + W8A8 FP8 | 4x GB200 | — | RTX PRO 6000 (sm_120) | 800.74 | — | 0.00 / 115.33 / 660.70 / 22.01 | theirs: 2.26x incl. NVFP4 |
| Cosmos3-Super 64B | same, no cache, W8A8 FP8 | 4x GB200 | 130.41 | — | — | — | — / — / — / — | theirs: BF16 on 4 GPUs (SP); not run: exit None |

Optimized-arm speedup over our own baseline (same GPU) vs theirs:

| Pair | Ours | Theirs |
|---|---:|---:|
| sana-baseline → sana-full | 2.12x | 2.77x |
| wan13-sol-base → wan13-sol-fullstack | 3.17x | — |
