// ==== region: attn_fp8 ====
// SageAttention-style FP8 attention (opt-in, lossy; crate::wan::attn_fp8).
//
// Q and K are quantized to E4M3 per block of rows (Q per 16 rows, one scale
// per warp's MMA rows; K per 64-row key tile) after K is smoothed by its
// per-head column mean. Smoothing is exact: q.(k - m) = q.k - q.m shifts a
// query's scores by one constant, which softmax cancels; it only removes the
// shared offset that would otherwise dominate K's quantization range
// (SageAttention, Zhang et al. 2024, section 3.2). S = Q K^T runs on FP8
// tensor cores (mma.sync m16n8k32 e4m3, sm_89+, f32 accumulation) and is
// dequantized by q_scale * k_scale; the online softmax and P V are the bf16
// flash kernels' (flash_mma_fwd2_body in kernels.cu), P rounded to bf16.
//
// This file is its own NVRTC module, compiled on first use: nothing here is
// loaded unless FASTVIDEO_ATTN_FP8 / the fp8_attention technique asks for it.
//
//   attn_fp8_colsum_{bf16,f32}   per-(head, dim) column sums (atomics)
//   attn_fp8_quant_bf16          BHSD bf16 rows -> e4m3 rows + scales
//   attn_fp8_quant_tile_f32      VSA: f32 rows gathered into tile slots -> e4m3
//   attn_fp8_fwd_d128            dense, 128-query CTAs, 8 warps
//   attn_fp8_vsa                 VSA fine stage, 64-query tile CTAs, 4 warps

typedef unsigned short fv_u16;
typedef unsigned char fv_u8;

#define F8_D 128            // head dim
#define F8_TILE 64          // key rows per tile
#define F8_KB (F8_TILE * F8_D)          // 8 KB e4m3 K tile
#define F8_VB (F8_TILE * F8_D * 2)      // 16 KB bf16 V tile
#define F8_STAGE (F8_KB + F8_VB)        // 24 KB
#define F8_E4M3_MAX 448.0f

__device__ __forceinline__ float f8_bf16_to_f32(fv_u16 b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
// Four f32 -> four e4m3 bytes (RNE, saturating), x0 in the lowest byte.
__device__ __forceinline__ unsigned int f8_pack4(float x0, float x1, float x2, float x3) {
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(x1), "f"(x0));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(x3), "f"(x2));
    return (unsigned int)lo | ((unsigned int)hi << 16);
}
#endif

// Column sums of a BHSD [bh, rows, 128] tensor, in two passes so the result
// is the same bits in every run (no atomics): grid (ceil(rows / rpb), bh),
// 128 threads (one per dim), each block writes its partial to
// part[(blockIdx.x * bh + head) * 128 + d]; attn_fp8_colsum_reduce then adds
// the partials of each (head, dim) in block order into sum [bh, 128].
extern "C" __global__ void attn_fp8_colsum_bf16(
    const fv_u16* __restrict__ x, float* __restrict__ part, int rows, int rpb
) {
    const long bh = blockIdx.y;
    const int d = threadIdx.x;
    const int r0 = blockIdx.x * rpb, r1 = min(rows, r0 + rpb);
    float acc = 0.f;
    for (int r = r0; r < r1; r++) acc += f8_bf16_to_f32(x[(bh * rows + r) * F8_D + d]);
    part[((long)blockIdx.x * gridDim.y + bh) * F8_D + d] = acc;
}

extern "C" __global__ void attn_fp8_colsum_f32(
    const float* __restrict__ x, float* __restrict__ part, int rows, int rpb
) {
    const long bh = blockIdx.y;
    const int d = threadIdx.x;
    const int r0 = blockIdx.x * rpb, r1 = min(rows, r0 + rpb);
    float acc = 0.f;
    for (int r = r0; r < r1; r++) acc += x[(bh * rows + r) * F8_D + d];
    part[((long)blockIdx.x * gridDim.y + bh) * F8_D + d] = acc;
}

// sum[i] = sum over b in [0, nblk) of part[b * n + i], in that order; one
// thread per (head, dim), n = bh * 128.
extern "C" __global__ void attn_fp8_colsum_reduce(
    const float* __restrict__ part, float* __restrict__ sum, int nblk, int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int b = 0; b < nblk; b++) acc += part[(long)b * n + i];
    sum[i] = acc;
}

// Shared body of the two quantizers: block = 4 warps = 64 rows; warp w owns
// rows [16w, 16w + 16) of the block, lane l dims [4l, 4l + 4). `load(row, d4)`
// returns the four values of a live row (the caller maps rows); rows past the
// end are zeros. With `smooth`, colsum * inv_rows is subtracted first (live
// rows only). grp = 16: one scale per warp; grp = 64: one per block.
// out: [bh, rows_pad, 128] e4m3; scales: [bh, rows_pad / grp].
template <typename Load>
__device__ __forceinline__ void f8_quant_block(
    Load load, int live_rows, const float* __restrict__ colsum, float inv_rows, int smooth,
    fv_u8* __restrict__ out, float* __restrict__ scales, int rows_pad, int grp
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __shared__ float wmax[4];
    const long bh = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * 64 + warp * 16;
    float mean[4] = {0.f, 0.f, 0.f, 0.f};
    if (smooth) {
        #pragma unroll
        for (int i = 0; i < 4; i++) mean[i] = colsum[bh * F8_D + 4 * lane + i] * inv_rows;
    }
    float v[16][4];
    float amax = 0.f;
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        const int row = row0 + r;
        if (row < live_rows) {
            float4 x = load(row, lane);
            v[r][0] = x.x - mean[0]; v[r][1] = x.y - mean[1];
            v[r][2] = x.z - mean[2]; v[r][3] = x.w - mean[3];
        } else {
            v[r][0] = v[r][1] = v[r][2] = v[r][3] = 0.f;
        }
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[r][0]), fabsf(v[r][1])), fmaxf(fabsf(v[r][2]), fabsf(v[r][3]))));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (grp == 64) {
        if (lane == 0) wmax[warp] = amax;
        __syncthreads();
        amax = fmaxf(fmaxf(wmax[0], wmax[1]), fmaxf(wmax[2], wmax[3]));
    }
    const float scale = amax > 0.f ? amax / F8_E4M3_MAX : 1.f;
    const float inv = amax > 0.f ? F8_E4M3_MAX / amax : 0.f;
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        const long row = row0 + r;
        *reinterpret_cast<unsigned int*>(out + (bh * rows_pad + row) * F8_D + 4 * lane) =
            f8_pack4(v[r][0] * inv, v[r][1] * inv, v[r][2] * inv, v[r][3] * inv);
    }
    if (lane == 0) {
        if (grp == 64) {
            if (warp == 0) scales[bh * (rows_pad / 64) + blockIdx.x] = scale;
        } else {
            scales[bh * (rows_pad / 16) + row0 / 16] = scale;
        }
    }
#else
    (void)load; (void)live_rows; (void)colsum; (void)inv_rows; (void)smooth;
    (void)out; (void)scales; (void)rows_pad; (void)grp;
    __trap();
#endif
}

// Dense: x [bh, rows, 128] bf16. grid (rows_pad / 64, bh), 128 threads.
extern "C" __global__ void __launch_bounds__(128) attn_fp8_quant_bf16(
    const fv_u16* __restrict__ x, const float* __restrict__ colsum, float inv_rows, int smooth,
    fv_u8* __restrict__ out, float* __restrict__ scales, int rows, int rows_pad, int grp
) {
    const long bh = blockIdx.y;
    auto load = [&](int row, int lane) {
        const uint2 u = *reinterpret_cast<const uint2*>(x + (bh * rows + row) * F8_D + 4 * lane);
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                           __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
    };
    f8_quant_block(load, rows, colsum, inv_rows, smooth, out, scales, rows_pad, grp);
}

// VSA: x [bh, seq, 128] f32 gathered into tile slots (slot_src[slot] = packed
// row or -1). grid (num_tiles, bh), 128 threads; rows_pad = num_tiles * 64.
extern "C" __global__ void __launch_bounds__(128) attn_fp8_quant_tile_f32(
    const float* __restrict__ x, const int* __restrict__ slot_src, const float* __restrict__ colsum,
    float inv_rows, int smooth, fv_u8* __restrict__ out, float* __restrict__ scales,
    int seq, int rows_pad, int grp
) {
    const long bh = blockIdx.y;
    // Padding slots are zero AFTER smoothing: they are masked keys / unused
    // query rows, and must not widen the tile's range.
    __shared__ int src[64];
    if (threadIdx.x < 64) src[threadIdx.x] = slot_src[blockIdx.x * 64 + threadIdx.x];
    __syncthreads();
    auto load = [&](int row, int lane) {
        const int s = src[row - blockIdx.x * 64];
        if (s < 0) {
            float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;
            if (smooth) {
                m0 = colsum[bh * F8_D + 4 * lane + 0] * inv_rows;
                m1 = colsum[bh * F8_D + 4 * lane + 1] * inv_rows;
                m2 = colsum[bh * F8_D + 4 * lane + 2] * inv_rows;
                m3 = colsum[bh * F8_D + 4 * lane + 3] * inv_rows;
            }
            return make_float4(m0, m1, m2, m3);   // minus the mean -> 0
        }
        return *reinterpret_cast<const float4*>(x + ((long)bh * seq + s) * F8_D + 4 * lane);
    };
    f8_quant_block(load, rows_pad, colsum, inv_rows, smooth, out, scales, rows_pad, grp);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
// Byte offset of (row, byte) in a swizzled tile with RB-byte rows: the
// 16-byte chunk index is XORed with row & 7 (conflict-free ldmatrix phases).
template <int RB> __device__ __forceinline__ unsigned int f8_swz(int row, int byte) {
    return (unsigned int)row * RB + ((((unsigned int)byte >> 4) ^ ((unsigned int)row & 7u)) << 4) + ((unsigned int)byte & 15u);
}
__device__ __forceinline__ unsigned int f8_smem_u32(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void f8_cp16(unsigned int smem, const void* gmem, int valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void f8_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void f8_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
// ROWS x RB-byte tile from global rows of RB bytes (row 0 at g) with NT
// threads; rows >= rows_valid are zero-filled and never read.
template <int RB, int ROWS, int NT>
__device__ __forceinline__ void f8_load(unsigned int smem, const unsigned char* g, int rows_valid, int tid) {
    constexpr int CPR = RB / 16;
    #pragma unroll
    for (int i = 0; i < ROWS * CPR / NT; i++) {
        const int chunk = tid + i * NT;
        const int row = chunk / CPR, c = chunk % CPR;
        const int ok = row < rows_valid;
        f8_cp16(smem + f8_swz<RB>(row, c * 16), ok ? (const void*)(g + (long)row * RB + c * 16) : (const void*)g, ok);
    }
}
__device__ __forceinline__ void f8_ldm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void f8_ldm_x4_trans(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void f8_mma_e4m3(float* c, const unsigned int* a, const unsigned int* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void f8_mma_bf16(float* c, const unsigned int* a, const unsigned int* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned int f8_pack_bf16(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ float f8_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
// This warp's Q fragments (16 rows x 128 e4m3 = 4 k-chunks of 32) from a
// swizzled 128-byte-row tile whose row 0 is the warp's first row.
__device__ __forceinline__ void f8_q_frags(unsigned int sQ, int warp, int lane, unsigned int (&qf)[4][4]) {
    #pragma unroll
    for (int kc = 0; kc < 4; kc++) {
        const int row = warp * 16 + (lane & 15), byte = kc * 32 + (lane >> 4) * 16;
        f8_ldm_x4(sQ + f8_swz<F8_D>(row, byte), qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3]);
    }
}
// S[16 x 64] = Q8[16 x 128] K8[64 x 128]^T (raw e4m3 products, f32).
__device__ __forceinline__ void f8_qk(unsigned int sK, const unsigned int (&qf)[4][4], float (&s)[8][4], int lane) {
    #pragma unroll
    for (int n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
    #pragma unroll
    for (int kc = 0; kc < 4; kc++) {
        #pragma unroll
        for (int n = 0; n < 8; n += 2) {
            unsigned int b[4];
            const int row = n * 8 + (lane & 7) + ((lane >> 4) << 3);
            const int byte = kc * 32 + ((lane >> 3) & 1) * 16;
            f8_ldm_x4(sK + f8_swz<F8_D>(row, byte), b[0], b[1], b[2], b[3]);
            f8_mma_e4m3(s[n], qf[kc], b);
            f8_mma_e4m3(s[n + 1], qf[kc], b + 2);
        }
    }
}
// O[16 x 128] += P[16 x 64] V[64 x 128], V a swizzled bf16 tile (256-byte rows).
__device__ __forceinline__ void f8_pv(unsigned int sV, const unsigned int (&pa)[4][4], float (&o)[16][4], int lane) {
    #pragma unroll
    for (int kc = 0; kc < 4; kc++) {
        #pragma unroll
        for (int n = 0; n < 16; n += 2) {
            unsigned int b[4];
            f8_ldm_x4_trans(sV + f8_swz<2 * F8_D>(kc * 16 + (lane & 15), (n * 8 + ((lane >> 4) << 3)) * 2), b[0], b[1], b[2], b[3]);
            f8_mma_bf16(o[n], pa[kc], b);
            f8_mma_bf16(o[n + 1], pa[kc], b + 2);
        }
    }
}
// One key tile of the online softmax: dequantize S by `dq`, mask columns >=
// len, rescale (m, l, o), exponentiate, pack P for the PV mma.
__device__ __forceinline__ void f8_softmax_tile(
    float (&s)[8][4], float dq, int len, float sl2, int t,
    float& m0, float& m1, float& l0, float& l1, float (&o)[16][4], unsigned int (&pa)[4][4]
) {
    const float NEG = __int_as_float(0xff800000);
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        const int c0 = n * 8 + 2 * t;
        s[n][0] = c0 < len ? s[n][0] * dq : NEG;
        s[n][2] = c0 < len ? s[n][2] * dq : NEG;
        s[n][1] = c0 + 1 < len ? s[n][1] * dq : NEG;
        s[n][3] = c0 + 1 < len ? s[n][3] * dq : NEG;
    }
    float rmax0 = NEG, rmax1 = NEG;
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        rmax0 = fmaxf(rmax0, fmaxf(s[n][0], s[n][1]));
        rmax1 = fmaxf(rmax1, fmaxf(s[n][2], s[n][3]));
    }
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));
    const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
    const float ms0 = (mn0 == NEG) ? 0.f : mn0 * sl2;
    const float ms1 = (mn1 == NEG) ? 0.f : mn1 * sl2;
    const float a0 = f8_exp2(fmaf(m0, sl2, -ms0)), a1 = f8_exp2(fmaf(m1, sl2, -ms1));
    m0 = mn0; m1 = mn1;
    float ls0 = 0.f, ls1 = 0.f;
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        s[n][0] = f8_exp2(fmaf(s[n][0], sl2, -ms0));
        s[n][1] = f8_exp2(fmaf(s[n][1], sl2, -ms0));
        s[n][2] = f8_exp2(fmaf(s[n][2], sl2, -ms1));
        s[n][3] = f8_exp2(fmaf(s[n][3], sl2, -ms1));
        ls0 += s[n][0] + s[n][1];
        ls1 += s[n][2] + s[n][3];
    }
    l0 = fmaf(l0, a0, ls0);
    l1 = fmaf(l1, a1, ls1);
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
    #pragma unroll
    for (int kc = 0; kc < 4; kc++) {
        pa[kc][0] = f8_pack_bf16(s[2 * kc][0], s[2 * kc][1]);
        pa[kc][1] = f8_pack_bf16(s[2 * kc][2], s[2 * kc][3]);
        pa[kc][2] = f8_pack_bf16(s[2 * kc + 1][0], s[2 * kc + 1][1]);
        pa[kc][3] = f8_pack_bf16(s[2 * kc + 1][2], s[2 * kc + 1][3]);
    }
}
#endif

// Dense SDPA, FP8 Q K^T: CTA = 128 queries x (batch*head), 8 warps; the
// flash_mma_fwd2 schedule (double-buffered (K8, V) stages, one barrier per
// key tile). q8 [bh, sq_pad, 128] with qs [bh, sq_pad / 16]; k8 [bh, sk_pad,
// 128] with ks [bh, sk_pad / 64]; v bf16 [bh, sk, 128] (unpadded). Output f32
// or bf16 [bh, sq, 128]. Dynamic shared memory: 2 * 24 KB.
extern "C" __global__ void __launch_bounds__(256, 1) attn_fp8_fwd_d128(
    const fv_u8* __restrict__ q8, const float* __restrict__ qs,
    const fv_u8* __restrict__ k8, const float* __restrict__ ks,
    const fv_u16* __restrict__ v, float* __restrict__ out, fv_u16* __restrict__ out_bf16,
    int out_is_bf16, int sq, int sk, int sq_pad, int sk_pad, float sl2
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    extern __shared__ __align__(128) unsigned char f8_smem[];
    constexpr int BR = 128;
    const unsigned int base = f8_smem_u32(f8_smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int q0 = (int)blockIdx.x * BR;
    const long bh = blockIdx.y;
    if (q0 >= sq || sk <= 0) return;
    const int qlen = min(BR, sq - q0);
    const unsigned char* Qh = q8 + (bh * sq_pad + q0) * F8_D;
    const unsigned char* Kh = k8 + bh * (long)sk_pad * F8_D;
    const unsigned char* Vh = reinterpret_cast<const unsigned char*>(v + bh * (long)sk * F8_D);
    const float* ksh = ks + bh * (long)(sk_pad / F8_TILE);
    const float qscale = qs[bh * (long)(sq_pad / 16) + (q0 >> 4) + warp];
    const int nkt = (sk + F8_TILE - 1) / F8_TILE;

    // Q (128 e4m3 rows, 16 KB) -> stage 1; (K8_0, V_0) -> stage 0.
    f8_load<F8_D, BR, 256>(base + F8_STAGE, Qh, BR, tid);
    f8_commit();
    f8_load<F8_D, F8_TILE, 256>(base, Kh, F8_TILE, tid);
    f8_load<2 * F8_D, F8_TILE, 256>(base + F8_KB, Vh, min(F8_TILE, sk), tid);
    f8_commit();
    f8_wait<1>();
    __syncthreads();
    unsigned int qf[4][4];
    f8_q_frags(base + F8_STAGE, warp, lane, qf);

    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    const float NEG = __int_as_float(0xff800000);
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float s[8][4];
    unsigned int pa[4][4];

    #pragma unroll 1
    for (int j = 0; j < nkt; j++) {
        const int kv0 = j * F8_TILE, len = min(F8_TILE, sk - kv0);
        const unsigned int sK = base + (j & 1) * F8_STAGE, sV = sK + F8_KB;
        f8_wait<0>();
        __syncthreads();   // stage j landed; every warp is past j-1 (and past Q)
        if (j + 1 < nkt) {
            const unsigned int nK = base + ((j + 1) & 1) * F8_STAGE;
            const int nkv = kv0 + F8_TILE;
            f8_load<F8_D, F8_TILE, 256>(nK, Kh + (long)nkv * F8_D, F8_TILE, tid);
            f8_load<2 * F8_D, F8_TILE, 256>(nK + F8_KB, Vh + (long)nkv * 2 * F8_D, min(F8_TILE, sk - nkv), tid);
            f8_commit();
        }
        f8_qk(sK, qf, s, lane);
        f8_softmax_tile(s, qscale * ksh[j], len, sl2, t, m0, m1, l0, l1, o, pa);
        f8_pv(sV, pa, o, lane);
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool ok0 = row0 < qlen, ok1 = row1 < qlen;
    const long r0 = (bh * sq + q0 + row0) * F8_D, r1 = r0 + 8L * F8_D;
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        const int col = n * 8 + 2 * t;
        if (out_is_bf16) {
            if (ok0) *reinterpret_cast<unsigned int*>(out_bf16 + r0 + col) = f8_pack_bf16(o[n][0] * inv0, o[n][1] * inv0);
            if (ok1) *reinterpret_cast<unsigned int*>(out_bf16 + r1 + col) = f8_pack_bf16(o[n][2] * inv1, o[n][3] * inv1);
        } else {
            if (ok0) *reinterpret_cast<float2*>(out + r0 + col) = make_float2(o[n][0] * inv0, o[n][1] * inv0);
            if (ok1) *reinterpret_cast<float2*>(out + r1 + col) = make_float2(o[n][2] * inv1, o[n][3] * inv1);
        }
    }
#else
    (void)q8; (void)qs; (void)k8; (void)ks; (void)v; (void)out; (void)out_bf16;
    (void)out_is_bf16; (void)sq; (void)sk; (void)sq_pad; (void)sk_pad; (void)sl2;
    __trap();
#endif
}

// VSA fine stage, FP8 Q K^T: one CTA per (query tile, batch*head), 4 warps.
// q8 / k8 [bh, padded, 128] e4m3 in tile-slot order (attn_fp8_quant_tile_f32)
// with qs [bh, padded / 16] and ks [bh, num_tiles]; vt [bh, padded, 128]
// bf16 (vsa_tile_qkv). `selected` [bh, num_tiles, topk] key tiles per query
// tile, `block_sizes` the live rows per tile. Output f32 [bh, padded, 128],
// tile-slot order (vsa_mma_attn's contract), from query tile q_base on.
extern "C" __global__ void __launch_bounds__(128, 2) attn_fp8_vsa(
    const fv_u8* __restrict__ q8, const float* __restrict__ qs,
    const fv_u8* __restrict__ k8, const float* __restrict__ ks,
    const fv_u16* __restrict__ vt, const unsigned int* __restrict__ selected,
    const int* __restrict__ block_sizes, float* __restrict__ out,
    int num_tiles, int topk, float sl2, int q_base
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    extern __shared__ __align__(128) unsigned char f8_smem[];
    const unsigned int base = f8_smem_u32(f8_smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int t = lane & 3, g = lane >> 2;
    const int qtile = q_base + (int)blockIdx.x;
    const long bh = blockIdx.y;
    if (qtile >= num_tiles || topk <= 0) return;
    const long padded = (long)num_tiles * F8_TILE;
    const unsigned char* Kb = k8 + bh * padded * F8_D;
    const unsigned char* Vb = reinterpret_cast<const unsigned char*>(vt + bh * padded * F8_D);
    const unsigned int* sel = selected + (bh * num_tiles + qtile) * (long)topk;
    const float* ksh = ks + bh * (long)num_tiles;
    const float qscale = qs[bh * (padded / 16) + qtile * 4 + warp];

    // Q tile (8 KB) -> stage 1; (K8, V) of the first selected tile -> stage 0.
    f8_load<F8_D, F8_TILE, 128>(base + F8_STAGE, q8 + (bh * padded + (long)qtile * F8_TILE) * F8_D, F8_TILE, tid);
    f8_commit();
    unsigned int kt = sel[0];
    f8_load<F8_D, F8_TILE, 128>(base, Kb + (long)kt * F8_TILE * F8_D, F8_TILE, tid);
    f8_load<2 * F8_D, F8_TILE, 128>(base + F8_KB, Vb + (long)kt * F8_TILE * 2 * F8_D, F8_TILE, tid);
    f8_commit();
    f8_wait<1>();
    __syncthreads();
    unsigned int qf[4][4];
    f8_q_frags(base + F8_STAGE, warp, lane, qf);

    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    const float NEG = __int_as_float(0xff800000);
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float s[8][4];
    unsigned int pa[4][4];

    #pragma unroll 1
    for (int i = 0; i < topk; i++) {
        const unsigned int sK = base + (i & 1) * F8_STAGE, sV = sK + F8_KB;
        const unsigned int cur = sel[i];
        f8_wait<0>();
        __syncthreads();
        if (i + 1 < topk) {
            const unsigned int nt = sel[i + 1];
            const unsigned int nK = base + ((i + 1) & 1) * F8_STAGE;
            f8_load<F8_D, F8_TILE, 128>(nK, Kb + (long)nt * F8_TILE * F8_D, F8_TILE, tid);
            f8_load<2 * F8_D, F8_TILE, 128>(nK + F8_KB, Vb + (long)nt * F8_TILE * 2 * F8_D, F8_TILE, tid);
            f8_commit();
        }
        f8_qk(sK, qf, s, lane);
        f8_softmax_tile(s, qscale * ksh[cur], block_sizes[cur], sl2, t, m0, m1, l0, l1, o, pa);
        f8_pv(sV, pa, o, lane);
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    float* ob = out + (bh * padded + (long)qtile * F8_TILE + warp * 16) * F8_D;
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        const int col = n * 8 + t * 2;
        *reinterpret_cast<float2*>(ob + (long)g * F8_D + col) = make_float2(o[n][0] * inv0, o[n][1] * inv0);
        *reinterpret_cast<float2*>(ob + (long)(g + 8) * F8_D + col) = make_float2(o[n][2] * inv1, o[n][3] * inv1);
    }
#else
    (void)q8; (void)qs; (void)k8; (void)ks; (void)vt; (void)selected; (void)block_sizes;
    (void)out; (void)num_tiles; (void)topk; (void)sl2; (void)q_base;
    __trap();
#endif
}
// ==== end region: attn_fp8 ====
