## Cells (warm; medians over the prompt set)

| cell | denoise s | s / forward | total s | VAE decode s | peak MiB (nvidia-smi) | DiT resident GiB |
|---|---:|---:|---:|---:|---:|---:|
| bf16-480 | 16.86 | 2.11 | 194.40 | 2.05 | 44996 | 33.17 |
| mxfp8-480 | 13.01 | 1.63 | 15.38 | 2.06 | 31972 | 20.3 |
| nvfp4-480 | 10.90 | 1.36 | 13.26 | 2.06 | 20644 | 9.33 |
| bf16-768 | 48.31 | 6.04 | 53.41 | 4.75 | 52452 | 33.17 |
| mxfp8-768 | 38.15 | 4.77 | 43.26 | 4.76 | 39396 | 20.3 |
| nvfp4-768 | 32.42 | 4.05 | 37.54 | 4.76 | 28100 | 9.33 |

## Quality per prompt (baseline -> candidate)

| pair | prompt | PSNR dB | LPIPS mean | sharpness ratio | jitter ratio |
|---|---|---:|---:|---:|---:|
| bf16-480 -> mxfp8-480 | ltx-newsbroadcast | 12.43 | 0.497 | 1.015 | 0.946 |
| bf16-480 -> mxfp8-480 | spark-mountain-lake | 15.50 | 0.424 | 1.064 | 0.989 |
| bf16-480 -> mxfp8-480 | talking-head | 18.18 | 0.265 | 1.019 | 1.012 |
| bf16-480 -> nvfp4-480 | ltx-newsbroadcast | 10.10 | 0.660 | 0.991 | 1.084 |
| bf16-480 -> nvfp4-480 | spark-mountain-lake | 13.83 | 0.608 | 1.430 | 1.134 |
| bf16-480 -> nvfp4-480 | talking-head | 13.10 | 0.452 | 1.012 | 0.973 |
| bf16-768 -> mxfp8-768 | ltx-newsbroadcast | 13.04 | 0.559 | 0.988 | 0.989 |
| bf16-768 -> mxfp8-768 | spark-mountain-lake | 18.98 | 0.325 | 0.966 | 0.966 |
| bf16-768 -> mxfp8-768 | talking-head | 16.39 | 0.313 | 1.003 | 1.070 |
| bf16-768 -> nvfp4-768 | ltx-newsbroadcast | 10.81 | 0.707 | 0.903 | 2.078 |
| bf16-768 -> nvfp4-768 | spark-mountain-lake | 14.73 | 0.574 | 1.205 | 1.104 |
| bf16-768 -> nvfp4-768 | talking-head | 12.97 | 0.549 | 1.047 | 1.286 |
| mxfp8-480 -> nvfp4-480 | ltx-newsbroadcast | 10.46 | 0.644 | 0.976 | 1.231 |
| mxfp8-480 -> nvfp4-480 | spark-mountain-lake | 13.34 | 0.617 | 1.342 | 1.145 |
| mxfp8-480 -> nvfp4-480 | talking-head | 13.40 | 0.440 | 0.993 | 0.976 |
| mxfp8-768 -> nvfp4-768 | ltx-newsbroadcast | 10.92 | 0.702 | 0.914 | 2.078 |
| mxfp8-768 -> nvfp4-768 | spark-mountain-lake | 15.29 | 0.563 | 1.244 | 1.142 |
| mxfp8-768 -> nvfp4-768 | talking-head | 13.09 | 0.540 | 1.044 | 1.238 |

## Gate (scripts/gpu/gate-policy.toml, kind lossy)

| baseline -> candidate | verdict | denoise speed-up | failed hard checks |
|---|---|---:|---|
| bf16-480 -> mxfp8-480 | pass | 1.296 | - |
| bf16-480 -> nvfp4-480 | fail | 1.547 | sharpness_ratio (nvfp4-480-spark-mountain-lake) |
| bf16-768 -> mxfp8-768 | pass | 1.266 | - |
| bf16-768 -> nvfp4-768 | fail | 1.490 | sharpness_ratio (nvfp4-768-ltx-newsbroadcast); temporal_jitter_ratio (nvfp4-768-ltx-newsbroadcast); sharpness_ratio (nvfp4-768-spark-mountain-lake); temporal_jitter_ratio (nvfp4-768-talking-head) |
| mxfp8-480 -> nvfp4-480 | fail | 1.193 | temporal_jitter_ratio (nvfp4-480-ltx-newsbroadcast); sharpness_ratio (nvfp4-480-spark-mountain-lake) |
| mxfp8-768 -> nvfp4-768 | fail | 1.177 | sharpness_ratio (nvfp4-768-ltx-newsbroadcast); temporal_jitter_ratio (nvfp4-768-ltx-newsbroadcast); sharpness_ratio (nvfp4-768-spark-mountain-lake); temporal_jitter_ratio (nvfp4-768-talking-head) |

## Parity (832x480, one prompt, seed 1024, the same cached text; rel-L2 against bf16)

| tensor | video_step01 | video_step02 | video_step04 | video_step08 | video_vel_step01 | audio_step01 | audio_step08 | step00_block_0 | step00_block_20 | step00_block_41 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 -> mxfp8 | 1.12e-03 | 4.80e-03 | 2.69e-02 | 4.76e-01 | 5.87e-02 | 2.08e-03 | 4.16e-01 | 0.00e+00 | 4.69e-02 | 8.29e-02 |
| bf16 -> nvfp4 | 3.37e-03 | 1.07e-02 | 4.60e-02 | 7.01e-01 | 1.76e-01 | 6.80e-03 | 6.68e-01 | 2.65e-02 | 1.49e-01 | 1.96e-01 |

bf16 -> mxfp8 decoded clip: PSNR 18.18 dB, LPIPS 0.263

bf16 -> nvfp4 decoded clip: PSNR 13.10 dB, LPIPS 0.443
