## Cells (warm; medians over the five prompts)

| cell | denoise s | s / forward | VAE decode s | video decode s | end to end s (text cached) | peak GiB (in-process) | peak GiB (nvidia-smi) |
|---|---:|---:|---:|---:|---:|---:|---:|
| trim-bf16-480 | 17.17 | 2.145 | 1.92 | 1.93 | 19.19 | 44.1 | 44.1 |
| trim-bf16-768 | 48.55 | 6.068 | 4.51 | 4.53 | 53.24 | 51.3 | 51.3 |
| trim-nv-480 | 11.02 | 1.374 | 1.91 | 1.92 | 13.04 | 20.3 | 20.3 |
| trim-nv-768 | 32.65 | 4.080 | 4.51 | 4.53 | 37.35 | 27.9 | 27.6 |
| trim-nvfp4attn-480 | 9.21 | 1.155 | 1.91 | 1.92 | 11.23 | 19.7 | 19.7 |
| trim-nvfp4attn-768 | 24.64 | 3.078 | 4.50 | 4.52 | 29.31 | 27.1 | 27.1 |
| v2-bf16vae-480 | 11.94 | 1.492 | 2.63 | 2.64 | 14.74 | 29.8 | 29.8 |
| v2-bf16vae-768 | 31.97 | 3.994 | 6.20 | 6.22 | 38.35 | 36.0 | 36.0 |
| v2-mxvae-480 | 11.86 | 1.485 | 1.96 | 1.97 | 13.93 | 32.1 | 32.1 |
| v2-mxvae-768 | 31.93 | 3.991 | 4.64 | 4.66 | 36.76 | 38.5 | 38.5 |
| v2-nvvae-480 | 11.97 | 1.497 | 1.77 | 1.78 | 13.89 | 27.1 | 27.1 |
| v2-nvvae-768 | 31.92 | 3.989 | 4.12 | 4.14 | 36.22 | 33.3 | 33.3 |
| v2-tae-480 | 11.92 | 1.490 | 0.14 | 0.17 | 12.19 | 26.8 | 26.8 |
| v2-tae-768 | 31.88 | 3.986 | 0.44 | 0.49 | 32.95 | 34.2 | 34.2 |

## Quality per pair (baseline -> candidate; median over prompts, [min..max])

| pair | prompts | PSNR dB | LPIPS mean | sharpness ratio | jitter ratio |
|---|---:|---|---|---|---|
| trim-bf16-480 -> trim-nv-480 | 5 | 12.05 [9.94..13.83] | 0.598 [0.452..0.660] | 1.012 [0.957..1.430] | 1.084 [0.852..1.313] |
| trim-bf16-480 -> trim-nvfp4attn-480 | 5 | 11.40 [10.09..13.30] | 0.622 [0.531..0.640] | 0.922 [0.859..1.478] | 0.822 [0.703..1.351] |
| trim-bf16-768 -> trim-nv-768 | 5 | 11.64 [10.03..14.73] | 0.601 [0.549..0.707] | 1.002 [0.795..1.205] | 1.286 [0.754..2.168] |
| trim-bf16-768 -> trim-nvfp4attn-768 | 5 | 11.05 [10.23..14.96] | 0.604 [0.594..0.710] | 0.942 [0.703..1.121] | 0.916 [0.651..1.913] |
| trim-nv-480 -> trim-nvfp4attn-480 | 5 | 13.02 [10.30..15.03] | 0.550 [0.467..0.620] | 0.907 [0.901..1.037] | 0.834 [0.786..0.992] |
| trim-nv-768 -> trim-nvfp4attn-768 | 5 | 13.37 [11.28..18.78] | 0.514 [0.413..0.669] | 0.930 [0.885..0.959] | 0.829 [0.678..0.869] |
| v2-bf16vae-480 -> v2-mxvae-480 | 5 | 45.89 [44.55..49.10] | 0.002 [0.001..0.004] | 1.001 [0.996..1.003] | 1.001 [0.995..1.002] |
| v2-bf16vae-480 -> v2-nvvae-480 | 5 | 35.87 [34.33..39.01] | 0.019 [0.010..0.029] | 1.027 [1.019..1.030] | 1.016 [1.003..1.036] |
| v2-bf16vae-480 -> v2-tae-480 | 5 | 24.77 [23.46..28.79] | 0.171 [0.109..0.204] | 0.896 [0.857..0.907] | 0.990 [0.926..1.069] |
| v2-bf16vae-768 -> v2-mxvae-768 | 5 | 47.84 [47.26..50.33] | 0.002 [0.001..0.004] | 1.003 [1.000..1.003] | 1.001 [0.998..1.004] |
| v2-bf16vae-768 -> v2-nvvae-768 | 5 | 37.63 [36.98..40.12] | 0.018 [0.010..0.028] | 1.028 [1.015..1.035] | 1.020 [1.012..1.033] |
| v2-bf16vae-768 -> v2-tae-768 | 5 | 27.24 [25.49..29.89] | 0.149 [0.095..0.207] | 0.885 [0.817..0.904] | 0.967 [0.859..1.006] |
| v2-mxvae-480 -> trim-nv-480 | 5 | 10.19 [9.19..12.71] | 0.662 [0.637..0.727] | 1.072 [1.001..1.220] | 0.994 [0.850..1.358] |
| v2-mxvae-480 -> trim-nvfp4attn-480 | 5 | 10.19 [9.76..12.52] | 0.678 [0.644..0.713] | 1.089 [0.901..1.105] | 0.798 [0.700..1.003] |
| v2-mxvae-480 -> v2-nvvae-480 | 5 | 35.80 [34.22..38.97] | 0.019 [0.010..0.030] | 1.026 [1.016..1.030] | 1.019 [1.003..1.034] |
| v2-mxvae-480 -> v2-tae-480 | 5 | 24.80 [23.48..28.85] | 0.170 [0.109..0.204] | 0.894 [0.855..0.905] | 0.989 [0.931..1.067] |
| v2-mxvae-768 -> trim-nv-768 | 5 | 10.71 [9.35..13.56] | 0.679 [0.627..0.734] | 1.130 [0.925..1.287] | 1.143 [0.850..1.596] |
| v2-mxvae-768 -> trim-nvfp4attn-768 | 5 | 10.60 [9.59..13.64] | 0.683 [0.628..0.727] | 1.017 [0.811..1.237] | 0.809 [0.725..1.063] |
| v2-mxvae-768 -> v2-nvvae-768 | 5 | 37.58 [36.93..40.09] | 0.018 [0.010..0.029] | 1.025 [1.014..1.032] | 1.020 [1.011..1.030] |
| v2-mxvae-768 -> v2-tae-768 | 5 | 27.27 [25.52..29.97] | 0.149 [0.096..0.206] | 0.882 [0.817..0.901] | 0.965 [0.861..1.002] |

## Gate (scripts/gpu/gate-policy.toml, kind lossy)

| baseline -> candidate | verdict | denoise speed-up | total speed-up | quality | failed hard checks |
|---|---|---:|---:|---|---|
| trim-bf16-480 -> trim-nv-480 | fail | 1.558 | 1.466 | fail | sharpness_ratio  (trim-nv-480-spark-mountain-lake); temporal_jitter_ratio  (trim-nv-480-ltx-frogyoga) |
| trim-bf16-480 -> trim-nvfp4attn-480 | fail | 1.864 | 1.697 | fail | temporal_jitter_ratio  (trim-nvfp4attn-480-talking-head); sharpness_ratio  (trim-nvfp4attn-480-spark-mountain-lake); sharpness_ratio  (trim-nvfp4attn-480-ltx-frogyoga); temporal_jitter_ratio  (trim-nvfp4attn-480-ltx-frogyoga); sharpness_ratio  (trim-nvfp4attn-480-ltx-newsbroadcast); temporal_jitter_ratio  (trim-nvfp4attn-480-ltx-newsbroadcast); sharpness_ratio  (trim-nvfp4attn-480-h3-demo); temporal_jitter_ratio  (trim-nvfp4attn-480-h3-demo) |
| trim-bf16-768 -> trim-nv-768 | fail | 1.487 | 1.421 | fail | temporal_jitter_ratio  (trim-nv-768-talking-head); sharpness_ratio  (trim-nv-768-spark-mountain-lake); temporal_jitter_ratio  (trim-nv-768-ltx-frogyoga); sharpness_ratio  (trim-nv-768-ltx-newsbroadcast); temporal_jitter_ratio  (trim-nv-768-ltx-newsbroadcast); sharpness_ratio  (trim-nv-768-h3-demo); temporal_jitter_ratio  (trim-nv-768-h3-demo) |
| trim-bf16-768 -> trim-nvfp4attn-768 | fail | 1.970 | 1.809 | fail | sharpness_ratio  (trim-nvfp4attn-768-talking-head); sharpness_ratio  (trim-nvfp4attn-768-spark-mountain-lake); temporal_jitter_ratio  (trim-nvfp4attn-768-ltx-frogyoga); sharpness_ratio  (trim-nvfp4attn-768-ltx-newsbroadcast); sharpness_ratio  (trim-nvfp4attn-768-h3-demo); temporal_jitter_ratio  (trim-nvfp4attn-768-h3-demo) |
| trim-nv-480 -> trim-nvfp4attn-480 | fail | 1.196 | 1.158 | fail | temporal_jitter_ratio  (trim-nvfp4attn-480-talking-head); sharpness_ratio  (trim-nvfp4attn-480-ltx-frogyoga); sharpness_ratio  (trim-nvfp4attn-480-ltx-newsbroadcast); temporal_jitter_ratio  (trim-nvfp4attn-480-ltx-newsbroadcast); sharpness_ratio  (trim-nvfp4attn-480-h3-demo); temporal_jitter_ratio  (trim-nvfp4attn-480-h3-demo) |
| trim-nv-768 -> trim-nvfp4attn-768 | fail | 1.325 | 1.273 | fail | sharpness_ratio  (trim-nvfp4attn-768-talking-head); temporal_jitter_ratio  (trim-nvfp4attn-768-talking-head); sharpness_ratio  (trim-nvfp4attn-768-spark-mountain-lake); temporal_jitter_ratio  (trim-nvfp4attn-768-spark-mountain-lake); sharpness_ratio  (trim-nvfp4attn-768-ltx-newsbroadcast); temporal_jitter_ratio  (trim-nvfp4attn-768-ltx-newsbroadcast); sharpness_ratio  (trim-nvfp4attn-768-h3-demo) |
| v2-bf16vae-480 -> v2-mxvae-480 | fail | 1.007 | 0.062 | pass | - |
| v2-bf16vae-480 -> v2-nvvae-480 | fail | 0.998 | 1.058 | pass | - |
| v2-bf16vae-480 -> v2-tae-480 | fail | 1.002 | 1.204 | fail | sharpness_ratio  (v2-tae-480-talking-head); sharpness_ratio  (v2-tae-480-spark-mountain-lake); sharpness_ratio  (v2-tae-480-ltx-frogyoga); sharpness_ratio  (v2-tae-480-ltx-newsbroadcast); sharpness_ratio  (v2-tae-480-h3-demo) |
| v2-bf16vae-768 -> v2-mxvae-768 | fail | 1.001 | 1.043 | pass | - |
| v2-bf16vae-768 -> v2-nvvae-768 | fail | 1.001 | 1.057 | pass | - |
| v2-bf16vae-768 -> v2-tae-768 | fail | 1.003 | 1.159 | fail | sharpness_ratio  (v2-tae-768-talking-head); sharpness_ratio  (v2-tae-768-spark-mountain-lake); sharpness_ratio  (v2-tae-768-ltx-frogyoga); sharpness_ratio  (v2-tae-768-ltx-newsbroadcast); sharpness_ratio  (v2-tae-768-h3-demo) |
| v2-mxvae-480 -> trim-nv-480 | fail | 1.077 | 18.195 | fail | sharpness_ratio  (trim-nv-480-ltx-frogyoga); sharpness_ratio  (trim-nv-480-ltx-newsbroadcast); temporal_jitter_ratio  (trim-nv-480-ltx-newsbroadcast); temporal_jitter_ratio  (trim-nv-480-h3-demo) |
| v2-mxvae-480 -> trim-nvfp4attn-480 | fail | 1.288 | 21.061 | fail | temporal_jitter_ratio  (trim-nvfp4attn-480-talking-head); sharpness_ratio  (trim-nvfp4attn-480-spark-mountain-lake); temporal_jitter_ratio  (trim-nvfp4attn-480-spark-mountain-lake); sharpness_ratio  (trim-nvfp4attn-480-ltx-frogyoga); sharpness_ratio  (trim-nvfp4attn-480-ltx-newsbroadcast); sharpness_ratio  (trim-nvfp4attn-480-h3-demo); temporal_jitter_ratio  (trim-nvfp4attn-480-h3-demo) |
| v2-mxvae-480 -> v2-nvvae-480 | fail | 0.991 | 17.072 | pass | - |
| v2-mxvae-480 -> v2-tae-480 | fail | 0.995 | 19.421 | fail | sharpness_ratio  (v2-tae-480-talking-head); sharpness_ratio  (v2-tae-480-spark-mountain-lake); sharpness_ratio  (v2-tae-480-ltx-frogyoga); sharpness_ratio  (v2-tae-480-ltx-newsbroadcast); sharpness_ratio  (v2-tae-480-h3-demo) |
| v2-mxvae-768 -> trim-nv-768 | fail | 0.978 | 0.983 | fail | sharpness_ratio  (trim-nv-768-talking-head); sharpness_ratio  (trim-nv-768-spark-mountain-lake); temporal_jitter_ratio  (trim-nv-768-spark-mountain-lake); sharpness_ratio  (trim-nv-768-ltx-frogyoga); temporal_jitter_ratio  (trim-nv-768-ltx-newsbroadcast); sharpness_ratio  (trim-nv-768-h3-demo); temporal_jitter_ratio  (trim-nv-768-h3-demo) |
| v2-mxvae-768 -> trim-nvfp4attn-768 | fail | 1.296 | 1.252 | fail | temporal_jitter_ratio  (trim-nvfp4attn-768-talking-head); sharpness_ratio  (trim-nvfp4attn-768-ltx-frogyoga); temporal_jitter_ratio  (trim-nvfp4attn-768-ltx-frogyoga); sharpness_ratio  (trim-nvfp4attn-768-h3-demo); temporal_jitter_ratio  (trim-nvfp4attn-768-h3-demo) |
| v2-mxvae-768 -> v2-nvvae-768 | fail | 1.000 | 1.014 | pass | - |
| v2-mxvae-768 -> v2-tae-768 | fail | 1.001 | 1.111 | fail | sharpness_ratio  (v2-tae-768-talking-head); sharpness_ratio  (v2-tae-768-spark-mountain-lake); sharpness_ratio  (v2-tae-768-ltx-frogyoga); sharpness_ratio  (v2-tae-768-ltx-newsbroadcast); sharpness_ratio  (v2-tae-768-h3-demo) |

## NVFP4 VAE check (FASTVIDEO_H3_VAE_CHECK, first tile batch, h3-base bf16 decoder as reference)

- **static**:
  - `[fastvideo] h3 vae: FASTVIDEO_H3_VAE_NVFP4=1: 36 blocks x 4 linears as published ModelOpt NVFP4 (W4A4, E4M3 per 16; 1.27 GiB with biases, loaded in 0.3s); activation scales static (calibrated input_scale): qkv 3.456e-1..3.744e0, to_out 2.783e-1..2.178e0, ff_in 3.456e-1..3.360e0, ff_out 1.304e-1..8.219e-1`
  - `[fastvideo] h3 vae: FASTVIDEO_H3_VAE_MXFP8 unset: NVFP4 decoder linears (FASTVIDEO_H3_VAE_NVFP4=1)`
  - `[fastvideo] h3 vae check: bf16 reference decoder from /workspace/weights/h3-base/vae (MXFP8 copies on)`
  - `[fastvideo] h3 vae check (nvfp4): batch 6 grid [7, 16, 16]: fused nvfp4 0.146s [vae_ff_in_gemm 0.024, vae_ff_out_gemm 0.022, vae_sdpa 0.019, vae_qkv_gemm 0.018, vae_swiglu 0.018, vae_residual_norm 0.018, vae_qkv_heads 0.012, vae_out_gemm 0.007, vae_unpatchify 0.002, vae_merge_heads 0.002, vae_proj_out 0.001, vae_proj_in 0.000]; unfused nvfp4 0.451s, rel_l2 3.411e-2; vs bf16 unfused (0.510s) rel_l2 4.312e-2; vs bf16 fused (0.196s) 4.312e-2 (bf16 fused vs unfused 1.881e-3); vs mxfp8 fused (0.150s) 4.336e-2 (mxfp8 vs bf16 unfused 1.238e-2) (nvfp4 within the 0.25 sanity bound)`
- **dynamic**:
  - `[fastvideo] h3 vae: FASTVIDEO_H3_VAE_NVFP4=1: 36 blocks x 4 linears as published ModelOpt NVFP4 (W4A4, E4M3 per 16; 1.27 GiB with biases, loaded in 0.4s); activation scales dynamic: qkv dynamic, to_out dynamic, ff_in dynamic, ff_out dynamic`
  - `[fastvideo] h3 vae: FASTVIDEO_H3_VAE_MXFP8 unset: NVFP4 decoder linears (FASTVIDEO_H3_VAE_NVFP4=1)`
  - `[fastvideo] h3 vae check: bf16 reference decoder from /workspace/weights/h3-base/vae (MXFP8 copies on)`
  - `[fastvideo] h3 vae check (nvfp4): batch 6 grid [7, 16, 16]: fused nvfp4 0.145s [vae_ff_out_gemm 0.027, vae_ff_in_gemm 0.024, vae_sdpa 0.019, vae_qkv_gemm 0.019, vae_swiglu 0.016, vae_residual_norm 0.015, vae_qkv_heads 0.012, vae_out_gemm 0.007, vae_unpatchify 0.002, vae_merge_heads 0.002, vae_proj_out 0.001, vae_proj_in 0.000]; unfused nvfp4 0.977s, rel_l2 3.422e-2; vs bf16 unfused (0.550s) rel_l2 4.261e-2; vs bf16 fused (0.201s) 4.261e-2 (bf16 fused vs unfused 1.881e-3); vs mxfp8 fused (0.155s) 4.292e-2 (mxfp8 vs bf16 unfused 1.238e-2) (nvfp4 within the 0.25 sanity bound)`
