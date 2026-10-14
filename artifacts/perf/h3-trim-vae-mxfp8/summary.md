## Video VAE time (same latents, same process; timed clips)

| res | clips | MXFP8 median s | bf16 median s | change | MXFP8 range | bf16 range |
|---|---|---|---|---|---|---|
| 480p | 15 | **1.652** | 2.135 | **-22.6%** | 1.597-1.668 | 2.066-2.148 |
| 768p | 15 | **3.944** | 5.053 | **-21.9%** | 3.898-3.960 | 5.001-5.074 |

## Peak memory

| cell | decoder linears | video_decode stage peak allocated / reserved GiB (median over clips) | whole-run peak allocated GiB | nvidia-smi peak MiB |
|---|---|---|---|---|
| ab480 | MXFP8 (+ bf16 kept) | 51.73 / 52.12 | 55.60 | 58504 |
| v480 | bf16 (=0) | - | - | None |
| ab768 | MXFP8 (+ bf16 kept) | 52.33 / 53.03 | 62.24 | 66568 |
| v768 | bf16 (=0) | - | - | None |

## Quality: MXFP8 decode against the bf16 decode of the same latents (compare-clips)

### 480p (15 clips)

| clip | status | lpips_mean | lpips_max | psnr_mean | psnr_min | sharpness_ratio_mean | temporal_jitter_ratio_mean |
|---|---|---|---|---|---|---|---|
| h3-demo-s0 | pass | 0.0006077 | 0.0009322 | 50.47 | 48.71 | 0.9998 | 0.9997 |
| h3-demo-s1000 | pass | 0.0005599 | 0.0008703 | 50.19 | 47.95 | 0.9989 | 0.999 |
| h3-demo-s2000 | pass | 0.0005595 | 0.0007863 | 50.42 | 48.63 | 0.9993 | 0.9994 |
| ltx-frogyoga-s1042 | pass | 0.0008379 | 0.001841 | 48.95 | 45.21 | 0.9997 | 1.003 |
| ltx-frogyoga-s2042 | pass | 0.0006082 | 0.0009416 | 49.19 | 47.1 | 0.9994 | 1.003 |
| ltx-frogyoga-s42 | pass | 0.0007808 | 0.001165 | 48.25 | 46.2 | 0.9992 | 1.002 |
| ltx-multishot-s1042 | pass | 0.0008027 | 0.002127 | 50.82 | 47.33 | 0.9999 | 0.9999 |
| ltx-multishot-s2042 | pass | 0.0008225 | 0.002168 | 50.32 | 46.74 | 1 | 0.9998 |
| ltx-multishot-s42 | pass | 0.0008083 | 0.002444 | 49.78 | 46.09 | 0.9998 | 0.9997 |
| ltx-newsbroadcast-s1042 | pass | 0.001103 | 0.002438 | 49.32 | 45.63 | 1.001 | 1 |
| ltx-newsbroadcast-s2042 | pass | 0.000929 | 0.001571 | 49.41 | 45.74 | 1.001 | 1.001 |
| ltx-newsbroadcast-s42 | pass | 0.001025 | 0.002125 | 47.97 | 45.64 | 1.001 | 1 |
| spark-mountain-lake-s1042 | pass | 0.00113 | 0.00137 | 49.94 | 48.14 | 1 | 0.9994 |
| spark-mountain-lake-s2042 | pass | 0.001192 | 0.001542 | 49.9 | 48.75 | 0.9997 | 0.9991 |
| spark-mountain-lake-s42 | pass | 0.001173 | 0.001437 | 49.79 | 48.56 | 1 | 0.9994 |
| **median** | | **0.0008225** | **0.001542** | **49.79** | **47.1** | **0.9998** | **0.9998** |
| worst | | 0.001192 | 0.002444 | 47.97 | 45.21 | 0.9989-1.001 | 0.999-1.003 |

### 768p (15 clips)

| clip | status | lpips_mean | lpips_max | psnr_mean | psnr_min | sharpness_ratio_mean | temporal_jitter_ratio_mean |
|---|---|---|---|---|---|---|---|
| h3-demo-s0 | pass | 0.0005708 | 0.0008545 | 50.67 | 48.25 | 0.9998 | 0.9996 |
| h3-demo-s1000 | pass | 0.0006357 | 0.0008266 | 51.16 | 49.02 | 0.9999 | 0.9996 |
| h3-demo-s2000 | pass | 0.0005928 | 0.000983 | 50.73 | 49.26 | 0.9994 | 0.9995 |
| ltx-frogyoga-s1042 | pass | 0.0007567 | 0.001162 | 50.08 | 47.76 | 1 | 1.002 |
| ltx-frogyoga-s2042 | pass | 0.0007419 | 0.001183 | 49.81 | 48.09 | 0.9998 | 1.004 |
| ltx-frogyoga-s42 | pass | 0.0006243 | 0.0009308 | 50.38 | 48.87 | 0.9998 | 1.004 |
| ltx-multishot-s1042 | pass | 0.001188 | 0.00349 | 50.74 | 47.17 | 1 | 1 |
| ltx-multishot-s2042 | pass | 0.001044 | 0.003753 | 51.16 | 47.88 | 1 | 1 |
| ltx-multishot-s42 | pass | 0.001057 | 0.003755 | 50.93 | 47.76 | 1 | 1 |
| ltx-newsbroadcast-s1042 | pass | 0.00147 | 0.002942 | 49.75 | 45.7 | 1.001 | 1 |
| ltx-newsbroadcast-s2042 | pass | 0.001309 | 0.002284 | 49.21 | 46.29 | 1.001 | 1 |
| ltx-newsbroadcast-s42 | pass | 0.001389 | 0.003683 | 50.67 | 45.04 | 1.001 | 1.001 |
| spark-mountain-lake-s1042 | pass | 0.001703 | 0.002168 | 49.45 | 47.96 | 1 | 0.9998 |
| spark-mountain-lake-s2042 | pass | 0.001512 | 0.00215 | 49.69 | 48.38 | 1 | 0.9993 |
| spark-mountain-lake-s42 | pass | 0.001743 | 0.002457 | 49.54 | 47.68 | 1 | 0.9992 |
| **median** | | **0.001057** | **0.002168** | **50.38** | **47.88** | **1** | **1** |
| worst | | 0.001743 | 0.003755 | 49.21 | 45.04 | 0.9994-1.001 | 0.9992-1.004 |

## Gate (quality checks)

- 480p: quality checks 60 / 60 pass (45 telemetry rows); failed: none
- 768p: quality checks 60 / 60 pass (45 telemetry rows); failed: none
