// ==== region: h3 video vae (ViT decoder glue around bf16 GEMMs) ====
// The ViT decoder's linears run as bf16 GEMMs with bf16 outputs. These fuse
// what used to be separate launches around them (cast_bf16_f32_bias_act,
// rms_norm_last, rope_half, split/merge heads, silu, elem_mul/add,
// cast_f32_bf16) without changing the arithmetic: each value goes through the
// same f32 operations in the same order (explicit _rn intrinsics where the
// unfused kernels were separate launches, so no FMA contraction sneaks in).
// Reductions may sum in a different order; that is the only difference.

// One f32 residual row per block (ROW_BLOCK_THREADS threads, blockDim.x
// floats of shared scratch). has_upd: x += (upd + bias) * scale, where upd is
// a branch's bf16 GEMM output, written to x_out. has_norm: bf16(RMSNorm(x) * w)
// to n_out, the next GEMM's input. Pointers a flag switches off are unread.
extern "C" __global__ void h3v_residual_norm_bf16(
    const float* x, const unsigned short* upd, const float* bias, const float* scale,
    const float* w, float* x_out, unsigned short* n_out,
    int rows, int width, float eps, int has_upd, int has_norm
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    long base = (long)row * (long)width;
    float local = 0.0f;
    for (int j = tid; j < width; j += nthreads) {
        float v = x[base + j];
        if (has_upd) {
            float a = __fadd_rn(fv_bf16_to_f32(upd[base + j]), bias[j]);
            v = __fadd_rn(v, __fmul_rn(a, scale[j]));
            x_out[base + j] = v;
        }
        local += v * v;
    }
    if (!has_norm) return;
    sdata[tid] = local;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] / (float)width + eps);
    const float* fin = has_upd ? x_out : x;
    for (int j = tid; j < width; j += nthreads) {
        n_out[base + j] = fv_bf16_rne(fin[base + j] * inv * w[j]);
    }
}

// Fused-QKV bf16 GEMM output [rows, 3*heads*d] (+ bias) to BHSD bf16 q, k, v
// for rows = batch*seq. q and k: per-head RMSNorm over d without a weight,
// then rotate_half RoPE on channels [0, r) with cos/sin [seq, r]; v: bias only.
// One warp per (row, head, q|k|v); d is a multiple of 32 and at most 128;
// blockDim.x is 256 (8 warps).
extern "C" __global__ void h3v_qkv_heads_bf16(
    const unsigned short* __restrict__ qkv, const float* __restrict__ bias,
    const float* __restrict__ cs, const float* __restrict__ sn,
    unsigned short* __restrict__ q, unsigned short* __restrict__ k,
    unsigned short* __restrict__ v,
    int rows, int seq, int heads, int d, int r, float eps
) {
    __shared__ float buf[8][128];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    long item = (long)blockIdx.x * (long)(blockDim.x >> 5) + warp;
    if (item >= (long)rows * heads * 3) return;
    int which = (int)(item % 3);
    long t = item / 3;
    int h = (int)(t % heads);
    int row = (int)(t / heads);
    int b = row / seq;
    int s = row - b * seq;
    long width = 3L * heads * d;
    long col0 = (long)which * heads * d + (long)h * d;
    const unsigned short* src = qkv + (long)row * width + col0;
    long o = (((long)b * heads + h) * seq + s) * d;
    unsigned short* dst = which == 0 ? q : (which == 1 ? k : v);
    int per = d >> 5;
    float vals[4];
    float local = 0.0f;
    for (int i = 0; i < per; ++i) {
        int c = lane + 32 * i;
        float x = __fadd_rn(fv_bf16_to_f32(src[c]), bias[col0 + c]);
        vals[i] = x;
        local += x * x;
    }
    if (which == 2) {
        for (int i = 0; i < per; ++i) dst[o + lane + 32 * i] = fv_bf16_rne(vals[i]);
        return;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    float inv = rsqrtf(local / (float)d + eps);
    float* rb = buf[warp];
    for (int i = 0; i < per; ++i) {
        vals[i] = vals[i] * inv;
        rb[lane + 32 * i] = vals[i];
    }
    __syncwarp();
    int half = r >> 1;
    for (int i = 0; i < per; ++i) {
        int c = lane + 32 * i;
        float out = vals[i];
        if (c < r) {
            float other = c < half ? -rb[c + half] : rb[c - half];
            long p = (long)s * r + c;
            out = vals[i] * cs[p] + other * sn[p];
        }
        dst[o + c] = fv_bf16_rne(out);
    }
}

// BHSD bf16 -> [batch, seq, heads*d] bf16. grid.x covers heads*d, grid.y
// strides over the batch*seq rows.
extern "C" __global__ void h3v_merge_heads_bf16(
    const unsigned short* __restrict__ src, unsigned short* __restrict__ out,
    int rows, int seq, int heads, int d
) {
    int hd = heads * d;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hd) return;
    int h = j / d;
    int p = j - h * d;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        int b = row / seq;
        int s = row - b * seq;
        out[(long)row * hd + j] = src[(((long)b * heads + h) * seq + s) * d + p];
    }
}

// Value-first SwiGLU on a bf16 GEMM output [rows, 2*half] plus its bias:
// bf16((v + bv) * silu(g + bg)), the next GEMM's input. grid.x covers half,
// grid.y strides over rows.
extern "C" __global__ void h3v_swiglu_bf16(
    const unsigned short* __restrict__ x, const float* __restrict__ bias,
    unsigned short* __restrict__ out, int rows, int half
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= half) return;
    float bv = bias[c];
    float bg = bias[half + c];
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        long in = (long)row * 2L * half;
        float v = __fadd_rn(fv_bf16_to_f32(x[in + c]), bv);
        float g = __fadd_rn(fv_bf16_to_f32(x[in + half + c]), bg);
        float sg = g / (1.0f + expf(-g));
        out[(long)row * half + c] = fv_bf16_rne(__fmul_rn(v, sg));
    }
}
// ---- h3v decode-speed variants (docs/perf/h3-vae-decode-speed.md) ----
// Vector-access twins (FASTVIDEO_H3_VAE_VEC): every element goes through the
// same f32 operations in the same order as the scalar kernel above, so the
// outputs are bit-identical; only the memory access width changes.

// h3v_merge_heads_bf16 as 16-byte copies: d % 8 == 0, so a run of 8 channels
// never crosses a head. One thread per 8 output values.
extern "C" __global__ void h3v_merge_heads_bf16_v(
    const unsigned short* __restrict__ src, unsigned short* __restrict__ out,
    int rows, int seq, int heads, int d
) {
    long hd8 = (long)heads * d / 8;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * hd8) return;
    long row = i / hd8;
    int j = (int)(i - row * hd8) * 8;
    int h = j / d;
    int p = j - h * d;
    long b = row / seq;
    long s = row - b * seq;
    const uint4* from = (const uint4*)(src + ((b * heads + h) * seq + s) * d + p);
    *(uint4*)(out + row * (long)heads * d + j) = *from;
}

// h3v_swiglu_bf16, two adjacent columns per thread (4-byte loads and stores).
// half % 2 == 0.
extern "C" __global__ void h3v_swiglu_bf16_v(
    const unsigned short* __restrict__ x, const float* __restrict__ bias,
    unsigned short* __restrict__ out, int rows, int half
) {
    int c = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= half) return;
    float bv0 = bias[c], bv1 = bias[c + 1];
    float bg0 = bias[half + c], bg1 = bias[half + c + 1];
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        long in = (long)row * 2L * half;
        unsigned int vv = *(const unsigned int*)(x + in + c);
        unsigned int gg = *(const unsigned int*)(x + in + half + c);
        float v0 = __fadd_rn(fv_bf16_to_f32((unsigned short)(vv & 0xFFFFu)), bv0);
        float v1 = __fadd_rn(fv_bf16_to_f32((unsigned short)(vv >> 16)), bv1);
        float g0 = __fadd_rn(fv_bf16_to_f32((unsigned short)(gg & 0xFFFFu)), bg0);
        float g1 = __fadd_rn(fv_bf16_to_f32((unsigned short)(gg >> 16)), bg1);
        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        unsigned int o0 = fv_bf16_rne(__fmul_rn(v0, s0));
        unsigned int o1 = fv_bf16_rne(__fmul_rn(v1, s1));
        *(unsigned int*)(out + (long)row * half + c) = o0 | (o1 << 16);
    }
}

// h3v_qkv_heads_bf16 with each lane loading `per` = d / 32 adjacent values in
// one access; shuffles then hand every lane the values the scalar kernel's
// lane holds (channels lane + 32 i), so the sums and every later operation
// are the scalar kernel's. d is 32, 64 or 128.
extern "C" __global__ void h3v_qkv_heads_bf16_v(
    const unsigned short* __restrict__ qkv, const float* __restrict__ bias,
    const float* __restrict__ cs, const float* __restrict__ sn,
    unsigned short* __restrict__ q, unsigned short* __restrict__ k,
    unsigned short* __restrict__ v,
    int rows, int seq, int heads, int d, int r, float eps
) {
    __shared__ float buf[8][128];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    long item = (long)blockIdx.x * (long)(blockDim.x >> 5) + warp;
    if (item >= (long)rows * heads * 3) return;
    int which = (int)(item % 3);
    long t = item / 3;
    int h = (int)(t % heads);
    int row = (int)(t / heads);
    int b = row / seq;
    int s = row - b * seq;
    long width = 3L * heads * d;
    long col0 = (long)which * heads * d + (long)h * d;
    const unsigned short* src = qkv + (long)row * width + col0;
    long o = (((long)b * heads + h) * seq + s) * d;
    unsigned short* dst = which == 0 ? q : (which == 1 ? k : v);
    int per = d >> 5;
    // raw[e]: channel lane * per + e of this head.
    unsigned int raw[4] = {0u, 0u, 0u, 0u};
    if (per == 4) {
        uint2 w = *(const uint2*)(src + lane * 4);
        raw[0] = w.x & 0xFFFFu; raw[1] = w.x >> 16;
        raw[2] = w.y & 0xFFFFu; raw[3] = w.y >> 16;
    } else if (per == 2) {
        unsigned int w = *(const unsigned int*)(src + lane * 2);
        raw[0] = w & 0xFFFFu; raw[1] = w >> 16;
    } else {
        raw[0] = src[lane];
    }
    float vals[4];
    float local = 0.0f;
    for (int i = 0; i < per; ++i) {
        int c = lane + 32 * i;
        // Channel c sits in lane c / per, slot c % per.
        int from = c / per;
        int slot = c - from * per;
        unsigned int mine = 0u;
        for (int e = 0; e < per; ++e) {
            unsigned int got = __shfl_sync(0xffffffffu, raw[e], from);
            if (e == slot) mine = got;
        }
        float x = __fadd_rn(fv_bf16_to_f32((unsigned short)mine), bias[col0 + c]);
        vals[i] = x;
        local += x * x;
    }
    if (which == 2) {
        for (int i = 0; i < per; ++i) dst[o + lane + 32 * i] = fv_bf16_rne(vals[i]);
        return;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    float inv = rsqrtf(local / (float)d + eps);
    float* rb = buf[warp];
    for (int i = 0; i < per; ++i) {
        vals[i] = vals[i] * inv;
        rb[lane + 32 * i] = vals[i];
    }
    __syncwarp();
    int half = r >> 1;
    for (int i = 0; i < per; ++i) {
        int c = lane + 32 * i;
        float out = vals[i];
        if (c < r) {
            float other = c < half ? -rb[c + half] : rb[c - half];
            long p = (long)s * r + c;
            out = vals[i] * cs[p] + other * sn[p];
        }
        dst[o + c] = fv_bf16_rne(out);
    }
}

// MXFP8 twins of the producers (FASTVIDEO_H3_VAE_MXFP8): each computes the
// bf16 value its bf16 kernel writes and block-quantizes it the way
// mxfp8_quantize would quantize that bf16 tensor (one E8M0 per 32 values
// along the row, VEC32_UE8M0 swizzle), into the next GEMM's MXFP8 operand.
// Widths are multiples of 32 and every warp covers 32 aligned columns.

extern "C" __global__ void h3v_residual_norm_mx(
    const float* x, const unsigned short* upd, const float* bias, const float* scale,
    const float* w, float* x_out, unsigned char* q, unsigned char* qs,
    int rows, int width, float eps, int has_upd
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    long base = (long)row * (long)width;
    float local = 0.0f;
    for (int j = tid; j < width; j += nthreads) {
        float v = x[base + j];
        if (has_upd) {
            float a = __fadd_rn(fv_bf16_to_f32(upd[base + j]), bias[j]);
            v = __fadd_rn(v, __fmul_rn(a, scale[j]));
            x_out[base + j] = v;
        }
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] / (float)width + eps);
    const float* fin = has_upd ? x_out : x;
    long cb = ((long)width / 32 + 3) / 4;
    for (int j = tid; j < width; j += nthreads) {
        fv_mx_store_group(fv_r16(fin[base + j] * inv * w[j]), row, j, width, q, qs, cb);
    }
}

extern "C" __global__ void h3v_merge_heads_mx(
    const unsigned short* __restrict__ src, unsigned char* __restrict__ q,
    unsigned char* __restrict__ qs, int rows, int seq, int heads, int d
) {
    int hd = heads * d;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hd) return;
    int h = j / d;
    int p = j - h * d;
    long cb = ((long)hd / 32 + 3) / 4;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        int b = row / seq;
        int s = row - b * seq;
        float val = fv_bf16_to_f32(src[(((long)b * heads + h) * seq + s) * d + p]);
        fv_mx_store_group(val, row, j, hd, q, qs, cb);
    }
}

extern "C" __global__ void h3v_swiglu_mx(
    const unsigned short* __restrict__ x, const float* __restrict__ bias,
    unsigned char* __restrict__ q, unsigned char* __restrict__ qs, int rows, int half
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= half) return;
    float bv = bias[c];
    float bg = bias[half + c];
    long cb = ((long)half / 32 + 3) / 4;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        long in = (long)row * 2L * half;
        float v = __fadd_rn(fv_bf16_to_f32(x[in + c]), bv);
        float g = __fadd_rn(fv_bf16_to_f32(x[in + half + c]), bg);
        float sg = g / (1.0f + expf(-g));
        fv_mx_store_group(fv_r16(__fmul_rn(v, sg)), row, c, half, q, qs, cb);
    }
}
// ==== endregion: h3 video vae ====
