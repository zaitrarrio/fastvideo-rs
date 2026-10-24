// ==== region: DiT block fusions (Phase 3c; ltx2/fuse.rs, h3/transformer.rs) ====
// Each kernel below replaces a chain of the bf16-activation kernels above
// (`mx_rms_norm`, `mx_binary`, `mx_unary`, `mx_split_heads`, `mx_rope_half`,
// `mx_merge_heads`, `mx_residual_gate`) and replays that chain's rounding
// points exactly: the same f32 expressions (products and sums spelled with
// explicit `__f*_rn` / `fmaf` where the unfused kernel's contraction was read
// off its PTX), the same row statistic (per-thread FMA partial sums over a
// `ROW_BLOCK_THREADS` stride, the fixed `fv_block_sum` tree, `rsqrtf`), a bf16
// rounding wherever the unfused chain stored a bf16 tensor. So the fused
// output is bit-identical to the unfused one (`fv-gpucheck kernels`,
// group `dit_fusion`), and FASTVIDEO_LTX_FUSE=0 / FASTVIDEO_H3_FUSE=0 fall
// back to the chain. One block of ROW_BLOCK_THREADS threads per token row.

// `mx_rms_norm`'s normalizer for one row.
__device__ __forceinline__ float fvf_row_rms(const void* x, int x16, long base, int width, float eps, float* sm) {
    float local = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = fv_ld(x, base + j, x16);
        local = fmaf(v, v, local);
    }
    return rsqrtf(fv_block_sum(local, sm) / (float)width + eps);
}

// LTX-2 q/k: RMSNorm across all heads (`rms_norm_across_heads`, [inner]
// weight), head split to BHSD, then the split rotary (`DeviceRope::apply`:
// rotate_half inside each head, head-major [H*S, r] f32 tables). Unfused:
// mx_rms_norm -> bf16, mx_split_heads, mx_rope_half -> bf16.
// `use_rope` = 0 is the text cross-attention (norm + split only). `dup` = 1:
// every table row holds its values twice (`SplitRope::rotate_half_tables`),
// so channel p and p + r/2 read the same (first-half) entry: the same values,
// half the table traffic, which dominates this kernel.
extern "C" __global__ void fvf_ltx_qk_norm_rope(
    const void* x, int x16, const void* w, int w16, const float* cs, const float* sn,
    int use_rope, unsigned short* out, int seq, int heads, int d, int r, float eps, int dup
) {
    extern __shared__ float sdata[];
    int s = blockIdx.x;
    if (s >= seq) return;
    int inner = heads * d;
    long base = (long)s * inner;
    float rr = fvf_row_rms(x, x16, base, inner, eps, sdata);
    int half = r / 2;
    for (int j = threadIdx.x; j < inner; j += blockDim.x) {
        int h = j / d;
        int p = j - h * d;
        float n = fv_r16(__fmul_rn(__fmul_rn(fv_ld(x, base + j, x16), rr), fv_ld(w, j, w16)));
        float o = n;
        if (use_rope && p < r) {
            int q = p < half ? p + half : p - half;
            int jq = h * d + q;
            float np = fv_r16(__fmul_rn(__fmul_rn(fv_ld(x, base + jq, x16), rr), fv_ld(w, jq, w16)));
            float other = p < half ? -np : np;
            long t = ((long)h * seq + s) * r + (dup && p >= half ? p - half : p);
            o = fmaf(n, cs[t], __fmul_rn(other, sn[t]));
        }
        out[((long)h * seq + s) * d + p] = fv_bf16_rne(o);
    }
}

// LTX-2 modulated RMSNorm, optionally after a gated residual, all bf16:
//   res_mode 0: h = x;  1: h = r16(x + r16(u * g[grow]));  2: h = r16(x + u)
//     (residual_gate_add_e / text-cross gate + add / plain add); h -> hidden.
//   mode 0 (`rms_adaln`):  r16(r16(h * rs * r16(1 + scale)) + shift)
//   mode 1 (norm, then `scale_shift`): r16(r16(r16(h * rs * w) * r16(1 + scale)) + shift)
// `tab` is a bf16 [rows, width] modulation table (scale / shift rows);
// `gtab` a bf16 [rows, width] gate table.
extern "C" __global__ void fvf_ltx_res_norm_mod(
    const unsigned short* x, const unsigned short* u, const unsigned short* gtab, int grow, int res_mode,
    const void* w, int w16, const unsigned short* tab, int scale_row, int shift_row, int mode,
    unsigned short* hidden, unsigned short* out, int rows, int width, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    long base = (long)row * width;
    const unsigned short* sc = tab + (long)scale_row * width;
    const unsigned short* sh = tab + (long)shift_row * width;
    float rs;
    if (res_mode == 0) {
        rs = fvf_row_rms(x, 1, base, width, eps, sdata);
    } else {
        // mx_residual_gate / mx_binary, stored; the norm reads the stored value.
        float local = 0.0f;
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float a = fv_bf16_to_f32(x[base + j]);
            float b = fv_bf16_to_f32(u[base + j]);
            if (res_mode == 1) b = fv_r16(__fmul_rn(b, fv_bf16_to_f32(gtab[(long)grow * width + j])));
            unsigned short hb = fv_bf16_rne(__fadd_rn(a, b));
            hidden[base + j] = hb;
            float v = fv_bf16_to_f32(hb);
            local = fmaf(v, v, local);
        }
        rs = rsqrtf(fv_block_sum(local, sdata) / (float)width + eps);
    }
    const unsigned short* src = res_mode == 0 ? x : hidden;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = __fmul_rn(fv_bf16_to_f32(src[base + j]), rs);
        float s1 = fv_r16(__fadd_rn(fv_bf16_to_f32(sc[j]), 1.0f));
        float n;
        if (mode == 0) {
            n = fv_r16(__fmul_rn(v, s1));
        } else {
            n = fv_r16(__fmul_rn(v, fv_ld(w, j, w16)));
            n = fv_r16(__fmul_rn(n, s1));
        }
        out[base + j] = fv_bf16_rne(__fadd_rn(n, fv_bf16_to_f32(sh[j])));
    }
}

// LTX-2.5 per-head output gates and the head merge (`apply_head_gates` +
// `merge_heads`): o [H, S, D] bf16, logits [S, H] bf16;
// out[s, h*D + p] = r16(o * r16(2 * r16(sigmoid(logit[s, h])))).
// One block per token row: the row's `heads` gates once into shared memory,
// then two channels per thread (`d` even).
extern "C" __global__ void fvf_ltx_gate_merge(
    const unsigned short* o, const unsigned short* logits, unsigned short* out,
    int seq, int heads, int d
) {
    extern __shared__ float gsm[];
    int s = blockIdx.x;
    if (s >= seq) return;
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        float l = fv_bf16_to_f32(logits[(long)s * heads + h]);
        float g = fv_r16(1.0f / (1.0f + expf(-l)));
        gsm[h] = fv_r16(__fmul_rn(g, 2.0f));
    }
    __syncthreads();
    int pairs = heads * d / 2;
    const unsigned int* o2 = (const unsigned int*)o;
    unsigned int* out2 = (unsigned int*)out;
    for (int k = threadIdx.x; k < pairs; k += blockDim.x) {
        int j = 2 * k;
        int h = j / d;
        int p = j - h * d;
        unsigned int pair = o2[(((long)h * seq + s) * d + p) >> 1];
        float g = gsm[h];
        float a = __fmul_rn(fv_bf16_to_f32((unsigned short)(pair & 0xFFFFu)), g);
        float b = __fmul_rn(fv_bf16_to_f32((unsigned short)(pair >> 16)), g);
        out2[(long)s * pairs + k] = (unsigned int)fv_bf16_rne(a) | ((unsigned int)fv_bf16_rne(b) << 16);
    }
}

// H3: `merge_heads` of the attention output [H, S, D] straight into the
// MXFP8 activation of an all-MXFP8 `to_out` — the bytes `mxfp8_quantize`
// writes for the merged bf16 tensor (same groups, same lanes). An f32 output
// (`o16` = 0, VSA's) is rounded to the bf16 value its cast would store. One
// block per row; `heads * d % 32 == 0`.
extern "C" __global__ void fvf_merge_heads_mx(
    const void* o, int o16, unsigned char* q, unsigned char* qs, int seq, int heads, int d
) {
    int s = blockIdx.x;
    if (s >= seq) return;
    int k = heads * d;
    long column_blocks = ((long)k / 32 + 3) / 4;
    for (int j0 = 0; j0 < k; j0 += blockDim.x) {
        if ((j0 + (int)(threadIdx.x & ~31u)) >= k) continue;
        int j = j0 + threadIdx.x;
        int h = j / d;
        float v = fv_r16(fv_ld(o, ((long)h * seq + s) * d + (j - h * d), o16));
        fv_mx_store_group(v, s, j, k, q, qs, column_blocks);
    }
}

// H3: a block's last residual as eager torch runs it (`h3_gate_residual`:
// res + bf16(gate[idx + base_g] * branch), stored bf16) and the NEXT block's
// `h3_norm_mod` of that stored hidden (table rows idx + base_n), bf16 or
// MXFP8. The normalizer and output expressions are h3_norm_mod's.
extern "C" __global__ void fvf_h3_gate_res_norm_mod(
    const unsigned short* res, const void* br, int b16, const float* tab, const unsigned int* idx,
    long base_g, int gate_slot, long base_n, int scale_slot, int shift_slot, const float* w,
    unsigned short* hidden, unsigned short* out, unsigned char* q, unsigned char* qs,
    int rows, int dim, float eps, int mx
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    long xb = (long)row * dim;
    const float* g = tab + (((long)idx[row] + base_g) * 6 + gate_slot) * dim;
    long tb = ((long)idx[row] + base_n) * 6;
    const float* sc = tab + (tb + scale_slot) * dim;
    const float* sh = tab + (tb + shift_slot) * dim;
    float local = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float p = fv_r16(__fmul_rn(g[j], fv_ld(br, xb + j, b16)));
        unsigned short hb = fv_bf16_rne(__fadd_rn(fv_bf16_to_f32(res[xb + j]), p));
        hidden[xb + j] = hb;
        float v = fv_bf16_to_f32(hb);
        local = fmaf(v, v, local);
    }
    float r = __frsqrt_rn(fv_block_sum(local, sdata) / (float)dim + eps);
    long column_blocks = ((long)dim / 32 + 3) / 4;
    // Each thread re-reads only the hidden entries it stored above.
    for (int j0 = 0; j0 < dim; j0 += blockDim.x) {
        int j = j0 + threadIdx.x;
        if (mx && (j0 + (int)(threadIdx.x & ~31u)) >= dim) continue;
        float v = 0.0f;
        if (j < dim) v = fv_r16(fmaf(__fmul_rn(__fmul_rn(fv_bf16_to_f32(hidden[xb + j]), r), w[j]), sc[j], sh[j]));
        if (mx) fv_mx_store_group(v, row, j, dim, q, qs, column_blocks);
        else if (j < dim) out[xb + j] = fv_bf16_rne(v);
    }
}

// Head split of columns [col_off, col_off + heads*d) of src [B, S, W] into
// BHSD, one block per (batch, seq) row and two channels per thread (W,
// col_off and d even): `mx_split_heads` without its per-element 64-bit
// index divisions. A pure copy (or bf16 -> f32 widening): o16 = 1 stores
// bf16 bits, 0 stores f32; a bf16 source into bf16 is copied bit for bit.
extern "C" __global__ void fvf_split_heads_rows(
    const void* src, int s16, void* out, int o16,
    int rows, int seq, int heads, int d, int width, int col_off
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int b = row / seq;
    int s = row - b * seq;
    long in_base = (long)row * width + col_off;
    int pairs = heads * d / 2;
    for (int k = threadIdx.x; k < pairs; k += blockDim.x) {
        int j = 2 * k;
        int h = j / d;
        int p = j - h * d;
        long o = (((long)b * heads + h) * seq + s) * d + p;
        if (s16 && o16) {
            ((unsigned int*)out)[o >> 1] = ((const unsigned int*)src)[(in_base + j) >> 1];
            continue;
        }
        float a, c;
        if (s16) {
            unsigned int w2 = ((const unsigned int*)src)[(in_base + j) >> 1];
            a = fv_bf16_to_f32((unsigned short)(w2 & 0xFFFFu));
            c = fv_bf16_to_f32((unsigned short)(w2 >> 16));
        } else {
            float2 f = ((const float2*)src)[(in_base + j) >> 1];
            a = f.x;
            c = f.y;
        }
        if (o16) ((unsigned int*)out)[o >> 1] = (unsigned int)fv_bf16_rne(a) | ((unsigned int)fv_bf16_rne(c) << 16);
        else ((float2*)out)[o >> 1] = make_float2(a, c);
    }
}

// ---- H3 glue (FASTVIDEO_H3_GLUE; h3/fused16.rs) ----
// The fused QKV(+gate) projection [B, S, W] (bf16) to bf16 BHSD q, k, v and
// the VSA gate in one launch: q and k normed per head and rotated over their
// first r channels exactly as h3_qk_norm_rope with d = 128 (dpad 128: four
// warps per row, each a butterfly over its 32 channels, then the four warp
// sums in order), v and gate copied. One warp per (row, part, head), four
// channels per lane (c = 4 * lane + i); the butterfly's xor 16 / 8 / 4 are
// lane xor 4 / 2 / 1 and its xor 2 / 1 the lane's own registers, so every
// sum is the same tree. Needs d == 128, r % 8 == 0, r <= 128, W % 4 == 0.
extern "C" __global__ void fvf_h3_qkvg_heads(
    const unsigned short* __restrict__ src, const float* __restrict__ wq, const float* __restrict__ wk,
    const float* __restrict__ cs, const float* __restrict__ sn, int use_rope,
    unsigned short* __restrict__ q, unsigned short* __restrict__ k, unsigned short* __restrict__ v,
    unsigned short* __restrict__ g, int batch, int seq, int heads, int parts, int r, int width, float eps,
    int dn
) {
    // `dn` (= 128) is the norm's divisor as a runtime value, as in
    // h3_qk_norm_rope: a constant would let the compiler turn `tot / d + eps`
    // into one FMA and round differently.
    const int d = 128;
    int lane = threadIdx.x & 31;
    long item = (long)blockIdx.x * (long)(blockDim.x >> 5) + (threadIdx.x >> 5);
    if (item >= (long)batch * seq * parts * heads) return;
    int h = (int)(item % heads);
    long t = item / heads;
    int part = (int)(t % parts);
    long bs = t / parts;
    int s = (int)(bs % seq);
    int b = (int)(bs / seq);
    long inner = (long)heads * d;
    const unsigned short* row = src + bs * width + part * inner + (long)h * d;
    long o = (((long)b * heads + h) * seq + s) * d + 4 * lane;
    uint2 raw = *(const uint2*)(row + 4 * lane);
    if (part >= 2) {
        *(uint2*)((part == 2 ? v : g) + o) = raw;
        return;
    }
    float x[4] = {fv_bf16_to_f32((unsigned short)(raw.x & 0xFFFFu)), fv_bf16_to_f32((unsigned short)(raw.x >> 16)),
                  fv_bf16_to_f32((unsigned short)(raw.y & 0xFFFFu)), fv_bf16_to_f32((unsigned short)(raw.y >> 16))};
    float sq[4];
    for (int i = 0; i < 4; ++i) {
        sq[i] = x[i] * x[i];
        for (int m = 4; m > 0; m >>= 1) sq[i] += __shfl_xor_sync(0xffffffffu, sq[i], m);
    }
    float l2[4] = {sq[0] + sq[2], sq[1] + sq[3], sq[2] + sq[0], sq[3] + sq[1]};
    float wsum = l2[0] + l2[1];
    float tot = 0.0f;
    for (int w = 0; w < 4; ++w) tot += __shfl_sync(0xffffffffu, wsum, 8 * w);
    float rr = rsqrtf(tot / (float)dn + eps);
    const float* wt = part == 0 ? wq : wk;
    int half = r / 2;
    int c0 = 4 * lane;
    int peer = c0 < half ? lane + half / 4 : lane - half / 4;
    if (peer < 0 || peer > 31) peer = lane;
    unsigned short outv[4];
    for (int i = 0; i < 4; ++i) {
        int p = c0 + i;
        float normed = x[i] * rr * wt[p];
        float pv = __shfl_sync(0xffffffffu, x[i], peer);
        float res = normed;
        if (use_rope && p < r) {
            int partner = p < half ? p + half : p - half;
            float pn = pv * rr * wt[partner];
            float rot = p < half ? -pn : pn;
            res = normed * cs[(long)s * r + p] + rot * sn[(long)s * r + p];
        }
        outv[i] = fv_bf16_rne(res);
    }
    *(uint2*)((part == 0 ? q : k) + o) = make_uint2((unsigned int)outv[0] | ((unsigned int)outv[1] << 16),
                                                   (unsigned int)outv[2] | ((unsigned int)outv[3] << 16));
}

// NVFP4 quantize-on-write: one value per lane, a 16-column NVFP4 block held
// by 16 consecutive lanes (column j at lane j % 16, j % 16 == lane % 16).
// The block's E4M3 scale and E2M1 codes are fv_nvfp4_scale_block's (static_6
// / static_4, e4 448), so the bytes equal nvfp4_quant_bf16_sw on the stored
// bf16 tensor; `v` is that bf16 value. Codes go out as one 32-bit word per 8
// lanes (low nibble first), the scale from the block's first lane, into
// cuBLASLt's VEC16_UE4M3 tiled layout. Every lane of the warp must call it.
__device__ __forceinline__ void fv_nvfp4_store_lane(
    float v, long row, int j, long cols, float amax, float e2,
    unsigned char* packed, unsigned char* scales
) {
    // fv_nvfp4_scale_block's `if (|v| > block_amax)` from 0 skips NaN.
    float a = fabsf(v);
    float m = (a > 0.0f) ? a : 0.0f;
    for (int o = 8; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, m, o);
        if (other > m) m = other;
    }
    unsigned char code = 0;
    float fake = 0.0f;
    if ((amax > 0.0f) && isfinite(amax)) {
        float encode = (e2 * 448.0f) / amax;
        float decode = amax / (e2 * 448.0f);
        code = fv_to_e4m3((m / e2) * encode * 1.0f);
        float s = fv_e4m3_to_f32(code);
        if (s != 0.0f) {
            float inv = 1.0f / (decode * s);
            fake = fv_nvfp4_fake_e2m1(v * inv);
        }
    }
    unsigned int word = (unsigned int)fv_nvfp4_code(fake) << (4 * (j & 7));
    for (int o = 1; o < 8; o <<= 1) word |= __shfl_xor_sync(0xffffffffu, word, o);
    if ((j & 7) == 0) *(unsigned int*)(packed + row * (cols / 2) + j / 2) = word;
    if ((j & 15) == 0) {
        long sc = cols / 16, blk = j / 16;
        long tile = (row / 128) * (sc / 4) + (blk / 4);
        long rr = row % 128;
        scales[tile * 512 + (rr % 32) * 16 + (rr / 32) * 4 + (blk % 4)] = code;
    }
}

// The padding rows of an NVFP4 operand (nvfp4_quant_bf16_sw's): zero scales
// for rows in [rows, rows_scale), zero codes for rows in [rows, rows_pack).
__device__ __forceinline__ void fv_nvfp4_pad_row(
    long row, long rows_pack, long cols, unsigned char* packed, unsigned char* scales
) {
    long sc = cols / 16;
    long tile0 = (row / 128) * (sc / 4);
    long rr = row % 128;
    for (long blk = threadIdx.x; blk < sc; blk += blockDim.x) {
        scales[(tile0 + blk / 4) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + (blk % 4)] = 0;
        if (row < rows_pack) *(uint2*)(packed + row * (cols / 2) + blk * 8) = make_uint2(0u, 0u);
    }
}

// SwiGLU straight into the down projection's NVFP4 operand (static
// activation amax `amax_ptr`): bf16(value * (gate * sigmoid(gate))) as
// mx_swiglu stores it, quantized as nvfp4_quant_bf16_sw would. One block per
// operand row (rows_scale of them; rows past `rows` are padding);
// half % 64 == 0, blockDim.x % 32 == 0.
extern "C" __global__ void fvf_h3_swiglu_nvfp4(
    const unsigned short* x, const float* amax_ptr, unsigned char* packed, unsigned char* scales,
    int rows, int rows_pack, int half, int rule
) {
    long row = blockIdx.x;
    if (row >= rows) {
        fv_nvfp4_pad_row(row, rows_pack, half, packed, scales);
        return;
    }
    float amax = *amax_ptr;
    float e2 = rule == 1 ? 4.0f : 6.0f;
    long xb = row * 2 * (long)half;
    for (int j0 = 0; j0 < half; j0 += blockDim.x) {
        if ((j0 + (int)(threadIdx.x & ~31u)) >= half) continue;
        int j = j0 + threadIdx.x;
        float v = fv_bf16_to_f32(x[xb + j]);
        float gt = fv_bf16_to_f32(x[xb + half + j]);
        float sg = 1.0f / (1.0f + expf(-gt));
        fv_nvfp4_store_lane(fv_r16(v * (gt * sg)), row, j, half, amax, e2, packed, scales);
    }
}

// max |bf16(x)| of an f32 tensor as nvfp4_amax_bf16 measures it on the
// rounded tensor: rounding is monotone, so each block rounds its partial
// max; NaN wins (as there). `out` zeroed by the caller.
extern "C" __global__ void fvf_amax_f32_r16(const float* x, float* out, long n) {
    __shared__ float sm[256];
    int tid = threadIdx.x;
    float acc = 0.0f;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = blockIdx.x * (long)blockDim.x + tid; i < n; i += stride) {
        float a = fabsf(x[i]);
        if (!(a <= acc)) acc = a;
    }
    sm[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { float o = sm[tid + s]; if (!(o <= sm[tid])) sm[tid] = o; }
        __syncthreads();
    }
    if (tid == 0) atomicMax((int*)out, __float_as_int(fv_r16(sm[0])));
}

// merge_heads of the attention output [H, S, D] (f32 rounded to bf16, or
// bf16) straight into `to_out`'s NVFP4 operand, with the amax of the merged
// bf16 tensor (fvf_amax_f32_r16 / nvfp4_amax_bf16). One block per operand
// row; heads * d % 64 == 0.
extern "C" __global__ void fvf_h3_merge_heads_nvfp4(
    const void* o, int o16, const float* amax_ptr, unsigned char* packed, unsigned char* scales,
    int seq, int rows_pack, int heads, int d, int rule
) {
    long s = blockIdx.x;
    int k = heads * d;
    if (s >= seq) {
        fv_nvfp4_pad_row(s, rows_pack, k, packed, scales);
        return;
    }
    float amax = *amax_ptr;
    float e2 = rule == 1 ? 4.0f : 6.0f;
    for (int j0 = 0; j0 < k; j0 += blockDim.x) {
        if ((j0 + (int)(threadIdx.x & ~31u)) >= k) continue;
        int j = j0 + threadIdx.x;
        int h = j / d;
        float v = fv_r16(fv_ld(o, ((long)h * seq + s) * d + (j - h * d), o16));
        fv_nvfp4_store_lane(v, s, j, k, amax, e2, packed, scales);
    }
}
// ==== endregion: DiT block fusions ====

