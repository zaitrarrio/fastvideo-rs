## Cells (warm; medians over the prompt set)

| cell | denoise s | s / forward | total s | VAE decode s | peak MiB (nvidia-smi) | DiT resident GiB |
|---|---:|---:|---:|---:|---:|---:|
| bf16-480 | 16.86 | 2.11 | 194.40 | 2.05 | 44996 | ? |
| mxfp8-480 | 13.01 | 1.63 | 15.38 | 2.06 | 31972 | ? |
| nvfp4-480 | 10.90 | 1.36 | 13.26 | 2.06 | 20644 | ? |
| bf16-768 | 48.31 | 6.04 | 53.41 | 4.75 | 52452 | ? |
| mxfp8-768 | 38.15 | 4.77 | 43.26 | 4.76 | 39396 | ? |
| nvfp4-768 | 32.42 | 4.05 | 37.54 | 4.76 | 28100 | ? |

## Quality per prompt (baseline -> candidate)

| pair | prompt | PSNR dB | LPIPS mean | sharpness ratio | jitter ratio |
|---|---|---:|---:|---:|---:|

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
