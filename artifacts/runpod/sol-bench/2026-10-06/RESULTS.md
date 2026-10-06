| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |
|---|---|---|---:|---|---:|---:|---|---|
| MiniMax-H3 | 768p 124 f 50 st, dense (MXFP8 linears: our sm_100+ default) | RTX 5090 | 1045.40 | RTX PRO 6000 (sm_120) | 395.36 | 0.38x | 95.03 / 0.28 / 388.38 / 6.56 | rtx5090_dense.toml is BF16; first request after load; text: cache hit |
| MiniMax-H3 | 768p 124 f 50 st, dense, BF16 | RTX 5090 | 1045.40 | RTX PRO 6000 (sm_120) | 472.11 | 0.45x | 92.25 / 0.48 / 464.94 / 6.52 | first request after load |
| MiniMax-H3 | 768p 124 f 50 st, fullopt (Sol + TeaCache) + MXFP8 linears | RTX 5090 | 231.20 | RTX PRO 6000 (sm_120) | 88.23 | 0.38x | 105.70 / 0.39 / 81.29 / 6.45 | rtx5090_fullopt.toml is BF16; text: cache hit |
| MiniMax-H3 | 768p 124 f 50 st, fullopt (Sol + TeaCache), BF16 | RTX 5090 | 231.20 | RTX PRO 6000 (sm_120) | 110.28 | 0.48x | 193.48 / 0.40 / 103.22 / 6.47 |  |
| LTX-2.5 distilled | 4K 5 s, Sol stage 2, BF16 | RTX 5090 | 273.63 | RTX PRO 6000 (sm_120) | 154.86 | 0.57x | 58.71 / 38.49 / 96.43 / 19.63 |  |
| LTX-2.5 distilled | 1080p 20 s, Sol stage 2, BF16 | RTX 5090 | 261.65 | RTX PRO 6000 (sm_120) | 106.14 | 0.41x | 56.76 / 1.99 / 90.03 / 13.89 | text: cache hit |
| LTX-2.5 distilled | 4K 5 s, Sol stage 2, NVFP4 video FFN | RTX 5090 | 171.82 | RTX PRO 6000 (sm_120) | 134.62 | 0.78x | 56.42 / 2.17 / 83.18 / 49.00 | text: cache hit |
| LTX-2.5 distilled | 1080p 20 s, Sol stage 2, NVFP4 video FFN | RTX 5090 | 164.40 | RTX PRO 6000 (sm_120) | 93.28 | 0.57x | 56.36 / 1.96 / 77.19 / 13.89 | text: cache hit |
| Wan2.2 TI2V-5B | 704x1280x121, 50 st, CFG 5, base | 1x GB200 | 70.25 | RTX PRO 6000 (sm_120) | 164.89 | 2.35x | 84.43 / 0.01 / 151.14 / 13.71 | theirs: 5-prompt median; text: cache hit |
| Wan2.2 TI2V-5B | same, EasyCache 0.036 | 1x GB200 | 24.35 | RTX PRO 6000 (sm_120) | 86.21 | 3.54x | 121.30 / 0.01 / 72.45 / 13.71 | theirs: fusion + EasyCache fullopt; text: cache hit |
| Wan2.2 TI2V-5B | same, EasyCache 0.036 + PISA | 1x GB200 | 28.69 | — | — | — | — / — / — / — | theirs: golden kernel+EasyCache+PISA run; not run: exit 124 |
| Wan2.1 T2V-14B | 1280x720x81, 50 st, CFG 5, base (15 st measured, x50/15) | - | — | RTX PRO 6000 (sm_120) | 1797.48 | — | 167.51 / 0.96 / 1788.73 / 7.76 | absolute number withdrawn upstream; extrapolated: denoise x50/15; first request after load |
| Wan2.1 T2V-14B | 1280x720x81, 50 st, EasyCache + Sol-Attn | - | — | RTX PRO 6000 (sm_120) | 478.58 | — | 217.91 / 2.07 / 468.73 / 7.75 | absolute number withdrawn upstream; first request after load |

LTX-2.5 Sol stage 2 only (RTX 5090 published vs ours on RTX PRO 6000):

| Cell | Theirs stage 2 (s) | Ours stage 2 (s) | Ours / theirs | Ours stage 1 (s) |
|---|---:|---:|---:|---:|
| ltx25-4k5s-sol-bf16 | 130.32 | 54.18 | 0.42x | 39.45 |
| ltx25-1080p20s-sol-bf16 | 122.27 | 50.61 | 0.41x | 36.70 |
| ltx25-4k5s-sol-nvfp4 | 72.82 | 46.01 | 0.63x | 34.47 |
| ltx25-1080p20s-sol-nvfp4 | 66.78 | 42.73 | 0.64x | 31.76 |

Wan2.1 T2V-14B 720p, our fullstack speedup over base: **3.76x** (sol-engine's withdrawn README headline: ~3.48x).

Wan2.2 TI2V-5B wan5b-easycache speedup over our base: **1.91x** (theirs 2.885x fullopt, 2.45x golden).
