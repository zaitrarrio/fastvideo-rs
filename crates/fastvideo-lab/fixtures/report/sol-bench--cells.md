| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |
|---|---|---|---:|---|---:|---:|---|---|
| MiniMax-H3 | 768p 124 f 50 st, dense (MXFP8 linears: our sm_100+ default) | RTX 5090 | 1045.40 | RTX PRO 6000 (sm_120) | 99.50 | 0.10x | 1.00 / 2.00 / 3.00 / 4.00 | rtx5090_dense.toml is BF16; first request after load |
| MiniMax-H3 | 768p 124 f 50 st, dense, BF16 | RTX 5090 | 1045.40 | RTX PRO 6000 (sm_120) | 107.25 | 0.10x | 50.50 / 1.25 / 81.00 / 6.50 |  |
| MiniMax-H3 | 768p 124 f 50 st, fullopt (Sol + TeaCache) + MXFP8 linears | RTX 5090 | 231.20 | RTX PRO 6000 (sm_120) | 114.50 | 0.50x | 50.50 / 1.25 / 82.00 / 6.50 | rtx5090_fullopt.toml is BF16 |
| MiniMax-H3 | 768p 124 f 50 st, fullopt (Sol + TeaCache), BF16 | RTX 5090 | 231.20 | RTX PRO 6000 (sm_120) | 121.75 | 0.53x | 50.50 / 1.25 / 83.00 / 6.50 | first request after load |
| LTX-2.5 distilled | 4K 5 s, Sol stage 2, BF16 | RTX 5090 | 273.63 | RTX PRO 6000 (sm_120) | 129.00 | 0.47x | 50.50 / 2.50 / 88.75 / 12.75 | text: cache hit |
| LTX-2.5 distilled | 1080p 20 s, Sol stage 2, BF16 | RTX 5090 | 261.65 | RTX PRO 6000 (sm_120) | 136.25 | 0.52x | 50.50 / 2.50 / 90.75 / 12.75 |  |
| LTX-2.5 distilled | 4K 5 s, Sol stage 2, NVFP4 video FFN | RTX 5090 | 171.82 | RTX PRO 6000 (sm_120) | 143.50 | 0.84x | 50.50 / 2.50 / 92.75 / 12.75 | first request after load |
| LTX-2.5 distilled | 1080p 20 s, Sol stage 2, NVFP4 video FFN | RTX 5090 | 164.40 | RTX PRO 6000 (sm_120) | 150.75 | 0.92x | 50.50 / 2.50 / 94.75 / 12.75 |  |
| Wan2.2 TI2V-5B | 704x1280x121, 50 st, CFG 5, base | 1x GB200 | 70.25 | RTX PRO 6000 (sm_120) | 158.00 | 2.25x | 50.50 / 1.25 / 88.00 / 6.50 | theirs: 5-prompt median; text: cache hit |
| Wan2.2 TI2V-5B | same, EasyCache 0.036 | 1x GB200 | 24.35 | RTX PRO 6000 (sm_120) | 165.25 | 6.79x | 50.50 / 1.25 / 89.00 / 6.50 | theirs: fusion + EasyCache fullopt; first request after load |
| Wan2.2 TI2V-5B | same, EasyCache 0.036 + PISA | 1x GB200 | 28.69 | — | — | — | — / — / — / — | theirs: golden kernel+EasyCache+PISA run; not run: exit 124 |
| Wan2.1 T2V-14B | 1280x720x81, 50 st, CFG 5, base (15 st measured, x50/15) | - | — | RTX PRO 6000 (sm_120) | 1766.67 | — | 50.50 / 1.25 / 1666.67 / 6.50 | absolute number withdrawn upstream; extrapolated: denoise x50/15 |
| Wan2.1 T2V-14B | 1280x720x81, 50 st, EasyCache + Sol-Attn | - | — | RTX PRO 6000 (sm_120) | 187.00 | — | 50.50 / 1.25 / 92.00 / 6.50 | absolute number withdrawn upstream; first request after load; text: cache hit |
| SANA-Video 2B | 832x480x81, 50 st, cfg 6, baseline | 1x GB200 | — | RTX PRO 6000 (sm_120) | 194.25 | — | 50.50 / 1.25 / 93.00 / 6.50 | theirs: ratio only (2.77x) |
| SANA-Video 2B | same, EasyCache 0.1 + QKV merge + bf16 linear attn | 1x GB200 | — | RTX PRO 6000 (sm_120) | 201.50 | — | 50.50 / 1.25 / 94.00 / 6.50 | theirs: ratio only (2.77x) |
| Wan2.1 T2V-1.3B | 832x480x81, 50 st, CFG 6, base (median of 5 prompts) | - | — | RTX PRO 6000 (sm_120) | 92.50 | — | 50.50 / 1.25 / 95.00 / 6.50 | no published number; first request after load |
| Wan2.1 T2V-1.3B | same, EasyCache 0.036 + Sol-Attn | - | — | RTX PRO 6000 (sm_120) | 216.00 | — | 50.50 / 1.25 / 96.00 / 6.50 | no published number; text: cache hit |
| Wan2.2 T2V-A14B | 1280x720x81, 40 st, CFG 4/3, base (expert swap) | 1x GB200 | 449.67 | RTX PRO 6000 (sm_120) | 223.25 | 0.50x | 50.50 / 1.25 / 97.00 / 6.50 |  |
| Wan2.2 T2V-A14B | same, EasyCache + PISA | 1x GB200 | 207.01 | RTX PRO 6000 (sm_120) | 230.50 | 1.11x | 50.50 / 1.25 / 98.00 / 6.50 | theirs also: kernel fusion; first request after load |
| LTX-2.3 HQ | 1920x1088x241, res2s 15 + 3, dense stage 2 | 1x GB200 | — | RTX PRO 6000 (sm_120) | 237.75 | — | 50.50 / 1.25 / 99.00 / 6.50 | theirs: ratio only (2.40x) |
| LTX-2.3 HQ | same, SCSP + PISA s2 + midpoint prune + NVFP4 FFN | 1x GB200 | — | RTX PRO 6000 (sm_120) | 245.00 | — | 50.50 / 1.25 / 100.00 / 6.50 | theirs: ratio only (2.40x); text: cache hit |
| LingBot-Video MoE | 480p 121 f 40 st + 1080p refiner 8 st, 1 prompt | 4x GB200 | 375.53 | RTX PRO 6000 (sm_120) | 171.25 | 0.46x | 50.50 / 3.25 / 131.25 / 10.50 | theirs: 4 GPUs (CP4); first request after load |
| LingBot-Video MoE | 480p 121 f 40 st + 1080p refiner 8 st, 1 prompt (refiner: 3 of 8 st measured, x8/3) | 4x GB200 | 375.53 | RTX PRO 6000 (sm_120) | 223.50 | 0.60x | 50.50 / 3.25 / 182.50 / 10.50 | theirs: 4 GPUs (CP4); extrapolated: refiner denoise x8/3; first request after load |
| LingBot-Video MoE | same, EasyCache + refiner PISA | 4x GB200 | 144.36 | RTX PRO 6000 (sm_120) | 173.25 | 1.20x | 50.50 / 3.25 / 131.25 / 10.50 | theirs: 4 GPUs (CP4); first request after load |
| Cosmos3-Super 64B | 1280x720x189, 35 st, CFG 6 | 4x GB200 | 130.41 | RTX PRO 6000 (sm_120) | 144.50 | 1.11x | 300.00 / 2.50 / 100.25 / 9.75 | theirs: 4 GPUs (SP) |
| Cosmos3-Super 64B | same, TeaCache 1.15/10/3 (BF16) | 4x GB200 | — | — | — | — | — / — / — / — | theirs: 2.26x incl. NVFP4; not run: timeout after 3300 s |
| Cosmos3-Super 64B | same, TeaCache + W8A8 FP8 | 4x GB200 | — | RTX PRO 6000 (sm_120) | 146.50 | — | 300.00 / 2.50 / 100.25 / 9.75 | theirs: 2.26x incl. NVFP4 |
| Cosmos3-Super 64B | same, no cache, W8A8 FP8 | 4x GB200 | 130.41 | RTX PRO 6000 (sm_120) | 147.50 | 1.13x | 300.00 / 2.50 / 100.25 / 9.75 | theirs: BF16 on 4 GPUs (SP) |

LTX-2.5 Sol stage 2 only (RTX 5090 published vs ours on RTX PRO 6000):

| Cell | Theirs stage 2 (s) | Ours stage 2 (s) | Ours / theirs | Ours stage 1 (s) |
|---|---:|---:|---:|---:|
| ltx25-4k5s-sol-bf16 | 130.32 | 54.50 | 0.42x | 34.25 |
| ltx25-1080p20s-sol-bf16 | 122.27 | 55.50 | 0.45x | 35.25 |
| ltx25-4k5s-sol-nvfp4 | 72.82 | 56.50 | 0.78x | 36.25 |
| ltx25-1080p20s-sol-nvfp4 | 66.78 | 57.50 | 0.86x | 37.25 |

Optimized-arm speedup over our own baseline (same GPU) vs theirs:

| Pair | Ours | Theirs |
|---|---:|---:|
| h3-768p-dense-bf16 → h3-768p-fullopt-bf16 | 0.88x | 4.52x |
| wan5b-base → wan5b-easycache | 0.96x | 2.89x |
| wan14-720p-base-s15 → wan14-720p-fullstack | 9.45x | — |
| sana-baseline → sana-full | 0.96x | 2.77x |
| wan13-sol-base → wan13-sol-fullstack | 0.43x | — |
| a14b-sol-base → a14b-sol-fullopt | 0.97x | 2.17x |
| ltx23-hq-base → ltx23-hq-fullopt | 0.97x | 2.40x |
| lingbot-baseline → lingbot-fullopt | 0.99x | 2.60x |
| lingbot-baseline-rs3 → lingbot-fullopt | 1.29x | 2.60x |
| cosmos3-baseline → cosmos3-teacache-fp8 | 0.99x | 2.26x |
| cosmos3-baseline-fp8 → cosmos3-teacache-fp8 | 1.01x | — |
