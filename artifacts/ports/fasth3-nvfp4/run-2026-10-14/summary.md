## Upstream FastVideo (d5253fd, the RTX PRO 6000 page's recipe)

| run | ok | wall s | denoise s | s / forward | decode s | device peak MiB |
|---|---|---:|---:|---:|---:|---:|
| v2-480p-h3-demo-cold | yes | 52.67 | 13.69 | 1.71 | 36.77 | 70063 |
| v2-480p-h3-demo-warm | yes | 14.69 | 10.25 | 1.28 | 3.43 | 70221 |
| v2-480p-spark-mountain-lake | yes | 15.32 | 9.81 | 1.23 | 3.36 | 70221 |
| v2-768p-h3-demo-cold | no: decode failed (upstream VAE compile) | - | 29.30 | 3.66 | - | 78451 |
| v2-768p-h3-demo-warm | no: decode failed (upstream VAE compile) | - | 29.28 | 3.66 | - | 78455 |
| v2-768p-spark-mountain-lake | no: decode failed (upstream VAE compile) | - | 28.41 | 3.55 | - | 78461 |
| v2-480p-h3-demo-8points | no: refused (grid points) | - | - | - | - | 78461 |

load 34.3 s (v2)

| 4s-480p-h3-demo-cold | yes | 19.47 | 6.84 | 1.71 | 9.93 | 70219 |
| 4s-480p-h3-demo-warm | yes | 9.27 | 4.78 | 1.20 | 3.47 | 70221 |
| 4s-480p-spark-mountain-lake | yes | 10.31 | 4.59 | 1.15 | 3.30 | 70221 |
| 4s-768p-h3-demo-cold | no: decode failed (upstream VAE compile) | - | 12.77 | 3.19 | - | 78449 |
| 4s-768p-h3-demo-warm | no: decode failed (upstream VAE compile) | - | 12.74 | 3.18 | - | 78453 |
| 4s-768p-spark-mountain-lake | no: decode failed (upstream VAE compile) | - | 12.30 | 3.08 | - | 78459 |

load 36.8 s (4s)


## Ours (fv-gpucheck h3 gen, one process per arm, warm after the 480p h3-demo generation)

| checkpoint | arm | prompt | denoise s | s / forward (median) | decode s | total s | device peak MiB |
|---|---|---|---:|---:|---:|---:|---:|
| v2 | nvfp4 | h3-demo-480p | 13.84 | 1.731 | 2.88 | 17.08 | 24382 |
| v2 | nvfp4 | h3-demo-768p | 39.90 | 4.990 | 6.59 | 47.00 | 31390 |
| v2 | nvfp4 | spark-mountain-lake-480p | 13.17 | 1.638 | 2.95 | 20.53 | 23840 |
| v2 | nvfp4 | spark-mountain-lake-768p | 38.77 | 4.846 | 6.60 | 45.78 | 31264 |
| v2 | nvfp4ffn-mxfp8 | h3-demo-480p | 15.18 | 1.899 | 2.90 | 18.49 | 30816 |
| v2 | nvfp4ffn-mxfp8 | h3-demo-768p | 43.47 | 5.433 | 6.59 | 50.69 | 38304 |
| v2 | nvfp4ffn-mxfp8 | spark-mountain-lake-480p | 14.21 | 1.778 | 2.89 | 17.42 | 30688 |
| v2 | nvfp4ffn-mxfp8 | spark-mountain-lake-768p | 41.99 | 5.240 | 6.65 | 49.07 | 38144 |
| 4s | nvfp4 | h3-demo-480p | 6.42 | 1.599 | 2.90 | 9.69 | 23966 |
| 4s | nvfp4 | h3-demo-768p | 16.86 | 4.211 | 6.58 | 23.91 | 31358 |
| 4s | nvfp4 | spark-mountain-lake-480p | 6.01 | 1.502 | 2.90 | 9.25 | 23774 |
| 4s | nvfp4 | spark-mountain-lake-768p | 16.27 | 4.067 | 6.59 | 23.25 | 31230 |
| 4s | nvfp4ffn-mxfp8 | h3-demo-480p | 7.01 | 1.754 | 2.91 | 10.31 | 30816 |
| 4s | nvfp4ffn-mxfp8 | h3-demo-768p | 18.42 | 4.605 | 6.63 | 25.48 | 38304 |
| 4s | nvfp4ffn-mxfp8 | spark-mountain-lake-480p | 6.58 | 1.640 | 2.92 | 9.86 | 30688 |
| 4s | nvfp4ffn-mxfp8 | spark-mountain-lake-768p | 17.71 | 4.427 | 6.62 | 24.87 | 38112 |

## Parity: ours (NVFP4 attention + FFN, VSA) on the reference's noise and text, 832x480x124

**v2**: audio_step01 8.93e-03, audio_step02 1.94e-02, audio_step03 3.15e-02, audio_step04 5.06e-02, audio_step05 7.27e-02, audio_step06 1.10e-01, audio_step07 1.70e-01, audio_step08 2.63e-01, step00_block_0 7.46e-02, step00_block_1 5.12e-02, step00_block_12 4.75e-02, step00_block_24 1.34e-01, step00_block_36 4.75e-01, step00_block_49 7.06e-01, text_hidden 1.92e-01, text_refined 8.14e-03, video_step01 1.04e-02, video_step02 1.80e-02, video_step03 2.92e-02, video_step04 4.89e-02, video_step05 8.52e-02, video_step06 1.59e-01, video_step07 3.36e-01, video_step08 6.14e-01, video_vel_step01 4.66e-01

clip PSNR: {"ref": "/srv/up-v2/00_v2-480p-h3-demo-cold.mp4", "cand": "/srv/ours-v2/inj/output.mp4", "video": {"frames_ref": 124, "frames_cand": 124, "frames": 124, "psnr_db": 12.564936851788422, "psnr_db_frame_mean": 12.853163049307163, "psnr_db_frame_min": 10.543787039877362}, "audio": {"samples_ref": 165888, "samples_cand": 165600, "rate": 32000, "psnr_db": 22.807105634301934, "best_lag_samples": 0, "psnr_

clip vs upstream: PSNR 12.812847316592954, LPIPS 0.4649461912009234, sharpness 0.9378079592992843, jitter 1.096866800262719 -> lossy gate hard quality checks: FAIL

**4s**: audio_step01 2.32e-02, audio_step02 6.34e-02, audio_step03 1.67e-01, audio_step04 4.45e-01, step00_block_0 8.14e-02, step00_block_1 5.66e-02, step00_block_12 5.80e-02, step00_block_24 1.34e-01, step00_block_36 1.53e+00, step00_block_49 7.56e-01, text_hidden 1.92e-01, text_refined 8.21e-03, video_step01 2.00e-02, video_step02 4.49e-02, video_step03 1.40e-01, video_step04 6.51e-01, video_vel_step01 4.56e-01

clip PSNR: {"ref": "/srv/up-4s/00_4s-480p-h3-demo-cold.mp4", "cand": "/srv/ours-4s/inj/output.mp4", "video": {"frames_ref": 124, "frames_cand": 124, "frames": 124, "psnr_db": 14.27533773323827, "psnr_db_frame_mean": 14.724422296294192, "psnr_db_frame_min": 8.980252983357646}, "audio": {"samples_ref": 165888, "samples_cand": 165600, "rate": 32000, "psnr_db": 20.029816883319043, "best_lag_samples": 0, "psnr_db

clip vs upstream: PSNR 14.686286999392097, LPIPS 0.5281734828554744, sharpness 1.047054578518934, jitter 1.2259041583483676 -> lossy gate hard quality checks: FAIL


## Per-block profile (ours, V2, 480p, FASTVIDEO_PROFILE=1 phases: synchronizing, so slower than timed)

