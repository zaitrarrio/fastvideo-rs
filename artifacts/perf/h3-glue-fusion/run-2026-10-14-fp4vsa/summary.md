
## nvfp4: per block, 480p (ms; one traced step / 50)
busy base 28.08  new 22.63; kernels 2790 -> 2290; step 1 / 1
| kernel | category | base ms | new ms | calls base/new |
| `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2` | gemm | 7.45 | 7.45 | 300/300 |
| `attn_fp4_vsa` | attention | 2.71 | 2.70 | 50/50 |
| `cast_f32_bf16` | cast | 1.78 | 0.03 | 308/58 |
| `nvfp4_quant_bf16_sw` | cast | 1.74 | 0.61 | 200/100 |
| `h3_qk_norm_rope` | norm_modulate_elementwise | 1.60 | 0.00 | 100/0 |
| `flash_mma_fwd_d128` | attention | 1.54 | 1.50 | 50/50 |
| `nvjet_sm120_ootst_mma_256x128x128_3_64x64x128_tmaAB_alignCD4_bx_vec16_` | gemm | 1.47 | 1.47 | 50/50 |
| `mx_swiglu` | norm_modulate_elementwise | 1.41 | 0.00 | 50/0 |
| `fvf_h3_swiglu_nvfp4` | cast | 0.00 | 1.27 | 0/50 |
| `fvf_h3_qkvg_heads` | other | 0.00 | 1.21 | 0/50 |
| `attn_fp4_quant_tile` | attention | 1.18 | 0.66 | 50/50 |
| `vsa_tile_mean` | attention | 0.96 | 0.00 | 150/0 |
| `fvf_split_heads_rows` | layout | 0.93 | 0.00 | 100/0 |
| `vsa_combine` | attention | 0.90 | 0.00 | 50/0 |
| `mx_merge_heads` | layout | 0.80 | 0.00 | 50/0 |
| `vsa_combine_g16` | attention | 0.00 | 0.76 | 0/50 |
| `fvf_h3_merge_heads_nvfp4` | cast | 0.00 | 0.67 | 0/50 |
| `fill_f` | layout | 0.59 | 0.59 | 100/100 |
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
| attention | 7.41 | 6.28 |
| cast | 3.86 | 3.08 |
| gemm | 9.00 | 9.00 |
| layout | 2.48 | 0.75 |
| norm_modulate_elementwise | 4.09 | 1.08 |
| other | 0.00 | 1.21 |
turbo: trace missing

## timing (warm, 124 frames)
| model | canvas | base denoise s | new denoise s | base step s | new step s | step win |
| nvfp4 | h3-demo-480p | 11.81 | 9.74 | 1.479 | 1.223 | 17.3% |
| nvfp4 | h3-demo-768p | 30.26 | 25.49 | 3.784 | 3.188 | 15.7% |
