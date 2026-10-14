
## nvfp4: per block, 480p (ms; one traced step / 50)
busy base 33.81  new 27.72; kernels 2890 -> 2390; step 1 / 1
| kernel | category | base ms | new ms | calls base/new |
| `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2` | gemm | 7.45 | 7.56 | 300/300 |
| `vsa_mma_attn_tma2` | attention | 7.20 | 7.30 | 50/50 |
| `vsa_tile_qkv` | attention | 2.45 | 0.00 | 150/0 |
| `cast_f32_bf16` | cast | 1.77 | 0.03 | 308/58 |
| `nvfp4_quant_bf16_sw` | cast | 1.74 | 0.61 | 200/100 |
| `h3_qk_norm_rope` | norm_modulate_elementwise | 1.60 | 0.00 | 100/0 |
| `flash_mma_fwd_d128` | attention | 1.52 | 1.48 | 50/50 |
| `nvjet_sm120_ootst_mma_256x128x128_3_64x64x128_tmaAB_alignCD4_bx_vec16_` | gemm | 1.47 | 1.50 | 50/50 |
| `mx_swiglu` | norm_modulate_elementwise | 1.40 | 0.00 | 50/0 |
| `fvf_h3_swiglu_nvfp4` | cast | 0.00 | 1.27 | 0/50 |
| `fvf_h3_qkvg_heads` | other | 0.00 | 1.21 | 0/50 |
| `vsa_tile_qkv_b16` | attention | 0.00 | 1.01 | 0/150 |
| `vsa_tile_mean` | attention | 0.96 | 0.00 | 150/0 |
| `fvf_split_heads_rows` | layout | 0.93 | 0.00 | 100/0 |
| `vsa_combine` | attention | 0.90 | 0.00 | 50/0 |
| `mx_merge_heads` | layout | 0.79 | 0.00 | 50/0 |
| `vsa_combine_g16` | attention | 0.00 | 0.76 | 0/50 |
| `fvf_h3_merge_heads_nvfp4` | cast | 0.00 | 0.67 | 0/50 |
| `fill_f` | layout | 0.59 | 0.61 | 100/100 |
| `vsa_tile_mean_b16` | attention | 0.00 | 0.52 | 0/150 |
| `h3_res_gate_norm_mod` | norm_modulate_elementwise | 0.48 | 0.48 | 50/50 |
| `fvf_h3_gate_res_norm_mod` | norm_modulate_elementwise | 0.46 | 0.46 | 49/49 |
| `fvf_amax_f32_r16` | cast | 0.00 | 0.32 | 0/50 |
| `nvfp4_amax_bf16` | cast | 0.30 | 0.13 | 100/50 |
| `block_copy` | layout | 0.11 | 0.00 | 154/0 |
| `mx_binary` | norm_modulate_elementwise | 0.10 | 0.10 | 104/104 |
| `mx_block_copy` | layout | 0.00 | 0.10 | 0/105 |
| `vsa_topk2` | attention | 0.08 | 0.08 | 50/50 |
| `softmax_last` | attention | 0.05 | 0.05 | 50/50 |
| `index_select_rows` | layout | 0.04 | 0.04 | 50/50 |
| `magma_sgemmEx_kernel<…>` | gemm | 0.04 | 0.04 | 50/50 |
| `cast_bf16_f32_bias_act` | cast | 0.04 | 0.04 | 56/56 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | gemm | 0.03 | 0.03 | 50/50 |
| category | base ms | new ms |
| attention | 13.17 | 11.21 |
| cast | 3.85 | 3.08 |
| gemm | 9.00 | 9.14 |
| layout | 2.49 | 0.78 |
| norm_modulate_elementwise | 4.08 | 1.08 |
| other | 0.00 | 1.21 |

## turbo: per block, 480p (ms; one traced step / 50)
busy base 38.30  new 34.36; kernels 2135 -> 1785; step 1 / 1
| kernel | category | base ms | new ms | calls base/new |
| `nvjet_sm120_qqtst_mma_128x256x64_3_32x128x64_tmaAB_alignCD4_bz_vec32_u` | gemm | 8.31 | 8.55 | 90/90 |
| `cutlass3x_sm120_bstensorop_s16832gemm_block_scaled_ue8m0xe4m3_ue8m0xe4` | gemm | 5.94 | 6.09 | 90/90 |
| `vsa_mma_attn_tma2` | attention | 4.63 | 4.72 | 50/50 |
| `nvjet_sm120_tst_mma_256x128x64_2_64x64x64_tmaAB_alignCD4_bz_TNNN` | gemm | 3.14 | 3.17 | 21/21 |
| `nvjet_sm120_tst_mma_128x128x64_3_64x64x64_tmaAB_alignCD4_bz_TNNN` | gemm | 2.93 | 2.98 | 46/46 |
| `vsa_tile_qkv` | attention | 2.45 | 0.00 | 150/0 |
| `h3_qk_norm_rope` | norm_modulate_elementwise | 1.59 | 0.00 | 100/0 |
| `flash_mma_fwd_d128` | attention | 1.52 | 1.51 | 50/50 |
| `cast_f32_bf16` | cast | 1.26 | 0.09 | 263/63 |
| `fvf_h3_qkvg_heads` | other | 0.00 | 1.18 | 0/50 |
| `vsa_tile_qkv_b16` | attention | 0.00 | 1.01 | 0/150 |
| `vsa_tile_mean` | attention | 0.96 | 0.00 | 150/0 |
| `fvf_split_heads_rows` | layout | 0.93 | 0.00 | 100/0 |
| `vsa_combine` | attention | 0.90 | 0.00 | 50/0 |
| `h3_swiglu_mx` | norm_modulate_elementwise | 0.78 | 0.79 | 45/45 |
| `vsa_combine_g16` | attention | 0.00 | 0.76 | 0/50 |
| `fill_f` | layout | 0.59 | 0.61 | 100/100 |
| `vsa_tile_mean_b16` | attention | 0.00 | 0.52 | 0/150 |
| `h3_res_gate_norm_mod` | norm_modulate_elementwise | 0.50 | 0.51 | 50/50 |
| `fvf_h3_gate_res_norm_mod` | norm_modulate_elementwise | 0.45 | 0.45 | 49/49 |
| `fvf_merge_heads_mx` | layout | 0.38 | 0.38 | 45/45 |
| `mxfp8_quantize` | cast | 0.22 | 0.22 | 45/45 |
| `mx_swiglu` | norm_modulate_elementwise | 0.14 | 0.14 | 5/5 |
| `block_copy` | layout | 0.11 | 0.00 | 154/0 |
| `mx_binary` | norm_modulate_elementwise | 0.10 | 0.10 | 104/104 |
| `mx_block_copy` | layout | 0.00 | 0.10 | 0/105 |
| `vsa_topk2` | attention | 0.08 | 0.08 | 50/50 |
| `mx_merge_heads` | layout | 0.08 | 0.08 | 5/5 |
| `softmax_last` | attention | 0.05 | 0.05 | 50/50 |
| `index_select_rows` | layout | 0.04 | 0.04 | 50/50 |
| `magma_sgemmEx_kernel<…>` | gemm | 0.00 | 0.04 | 0/50 |
| category | base ms | new ms |
| attention | 10.59 | 8.67 |
| cast | 1.52 | 0.34 |
| gemm | 20.39 | 20.86 |
| layout | 2.15 | 1.24 |
| norm_modulate_elementwise | 3.62 | 2.03 |
| other | 0.00 | 1.18 |

## timing (warm, 124 frames)
| model | canvas | base denoise s | new denoise s | base step s | new step s | step win |
| turbo | h3-demo-480p | 7.66 | 7.02 | 1.913 | 1.753 | 8.4% |
| turbo | h3-demo-768p | 20.24 | 18.63 | 5.060 | 4.658 | 8.0% |
| nvfp4 | h3-demo-480p | 13.70 | 11.44 | 1.711 | 1.431 | 16.4% |
| nvfp4 | h3-demo-768p | 39.87 | 34.90 | 4.982 | 4.360 | 12.5% |
