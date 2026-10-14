## Video VAE time (same latents, same process; timed clips)

| res | clips | MXFP8 median s | bf16 median s | change | MXFP8 range | bf16 range |
|---|---|---|---|---|---|---|
| 480p | 15 | **1.627** | 2.104 | **-22.7%** | 1.580-1.637 | 2.047-2.114 |
| 768p | 15 | **3.867** | 4.961 | **-22.1%** | 3.841-3.881 | 4.934-4.978 |

## Peak memory

| cell | decoder linears | video_decode stage peak allocated / reserved GiB (median over clips) | whole-run peak allocated GiB | nvidia-smi peak MiB |
|---|---|---|---|---|
| ab480 | MXFP8 (+ bf16 kept) | 51.73 / 52.12 | 55.60 | 58504 |
| v480 | bf16 (=0) | 51.91 / 52.19 | 53.84 | 56870 |
| ab768 | MXFP8 (+ bf16 kept) | 52.33 / 53.03 | 62.24 | 66568 |
| v768 | bf16 (=0) | 52.38 / 52.72 | 60.48 | 63816 |

## Quality: MXFP8 decode against the bf16 decode of the same latents (compare-clips)

### 480p (4 clips)

| clip | status | lpips_mean | lpips_max | psnr_mean | psnr_min | sharpness_ratio_mean | temporal_jitter_ratio_mean |
|---|---|---|---|---|---|---|---|
| h3-demo-s0 | pass | 0.0006077 | 0.0009322 | 50.47 | 48.71 | 0.9998 | 0.9997 |
| h3-demo-s1000 | pass | 0.0005599 | 0.0008703 | 50.19 | 47.95 | 0.9989 | 0.999 |
| h3-demo-s2000 | pass | 0.0005595 | 0.0007863 | 50.42 | 48.63 | 0.9993 | 0.9994 |
| ltx-frogyoga-s1042 | pass | 0.0008379 | 0.001841 | 48.95 | 45.21 | 0.9997 | 1.003 |
| **median** | | **0.0005838** | **0.0009012** | **50.3** | **48.29** | **0.9995** | **0.9995** |
| worst | | 0.0008379 | 0.001841 | 48.95 | 45.21 | 0.9989-0.9998 | 0.999-1.003 |

### 768p (0 clips)

| clip | status | lpips_mean | lpips_max | psnr_mean | psnr_min | sharpness_ratio_mean | temporal_jitter_ratio_mean |
|---|---|---|---|---|---|---|---|

## Gate (quality checks)

