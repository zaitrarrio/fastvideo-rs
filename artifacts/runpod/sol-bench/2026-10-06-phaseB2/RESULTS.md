| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |
|---|---|---|---:|---|---:|---:|---|---|
| Wan2.2 T2V-A14B | 1280x720x81, 40 st, CFG 4/3, base (expert swap) | 1x GB200 | 449.67 | RTX PRO 6000 (sm_120) | 1441.74 | 3.21x | 314.66 / 1.15 / 1432.79 / 7.77 | first request after load |
| Wan2.2 T2V-A14B | same, EasyCache + PISA | 1x GB200 | 207.01 | — | — | — | — / — / — / — | theirs also: kernel fusion; not run: exit None |
| LTX-2.3 HQ | 1920x1088x241, res2s 15 + 3, dense stage 2 | 1x GB200 | — | RTX PRO 6000 (sm_120) | 213.39 | — | 15.41 / 49.03 / 153.11 / 9.87 | theirs: ratio only (2.40x) |
| LTX-2.3 HQ | same, SCSP + PISA s2 + midpoint prune + NVFP4 FFN | 1x GB200 | — | — | — | — | — / — / — / — | theirs: ratio only (2.40x); not run: exit None |
| LingBot-Video MoE | 480p 121 f 40 st + 1080p refiner 8 st, 1 prompt | 4x GB200 | 375.53 | — | — | — | — / — / — / — | theirs: 4 GPUs (CP4); not run: exit 2 |
| LingBot-Video MoE | same, EasyCache + refiner PISA | 4x GB200 | 144.36 | — | — | — | — / — / — / — | theirs: 4 GPUs (CP4); not run: exit 2 |
| Cosmos3-Super 64B | 1280x720x189, 35 st, CFG 6 | 4x GB200 | 130.41 | RTX PRO 6000 (sm_120) | 1608.56 | 12.33x | 0.00 / 1.55 / 1584.09 / 21.97 | theirs: 4 GPUs (SP) |
| Cosmos3-Super 64B | same, TeaCache 1.15/10/3 (BF16) | 4x GB200 | — | — | — | — | — / — / — / — | theirs: 2.26x incl. NVFP4; not run: budget |
