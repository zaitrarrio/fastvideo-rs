// ==== region: attn_sage ====
// SageAttention2-style attention (opt-in, lossy; crate::wan::attn_sage):
// INT8 Q K^T and FP8 P V (Zhang et al., SageAttention2, 2024/2025).
//
// * K is smoothed by its per-head column mean (exact under softmax), then
//   quantized to INT8 per 64-key tile; Q to INT8 per 16 rows (one scale per
//   MMA warp, finer than upstream's per-32-row "per_warp"). Scales amax/127,
//   round-to-nearest, saturating.
// * S = Q8 K8^T on mma.sync m16n8k32 s8 (s32 accumulate, exact), dequantized
//   by q_scale * k_scale; online softmax in f32 (row sums from the f32 P).
// * P (in [0, 1]) is quantized to E4M3 with the fixed scale 448 and V to
//   E4M3 per channel (amax over the head's keys / 448); P V runs on
//   mma.sync m16n8k32 e4m3 with f32 accumulation, and the epilogue divides
//   by 448 * l and multiplies by the channel scale.
// * V is stored transposed, [bh, 128, sk_pad] e4m3, with keys permuted inside
//   every 16-key group (slot 4t+i <- key {2t, 2t+1, 8+2t, 9+2t}[i]) so the
//   softmax's accumulator fragment is directly the A operand of the k32 MMA.
//
// Its own NVRTC module (sm_89+), loaded only when FASTVIDEO_ATTN_SAGE asks.
//
//   attn_sage_colsum_bf16   per-(head, dim) column sums (K smoothing)
//   attn_sage_quant_i8      BHSD bf16 rows -> int8 rows + scales
//   attn_sage_vmax          per-(head, dim) |V| max (atomicMax on the bits)
//   attn_sage_vquant        V bf16 -> e4m3, transposed + permuted
//   attn_sage_fwd_d128      dense, 128-query CTAs, 8 warps

typedef unsigned short sg_u16;
typedef unsigned char sg_u8;

#define SG_D 128                         // head dim
#define SG_TILE 64                       // keys per tile
#define SG_KB (SG_TILE * SG_D)           // 8 KB int8 K tile
#define SG_VB (SG_D * SG_TILE)           // 8 KB e4m3 V^T tile
#define SG_STAGE (SG_KB + SG_VB)         // 16 KB
#define SG_P_SCALE 448.0f

__device__ __forceinline__ float sg_bf16_to_f32(sg_u16 b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

extern "C" __global__ void attn_sage_colsum_bf16(
    const sg_u16* __restrict__ x, float* __restrict__ sum, int rows, int rpb
) {
    const long bh = blockIdx.y;
    const int d = threadIdx.x;
    const int r0 = blockIdx.x * rpb, r1 = min(rows, r0 + rpb);
    float acc = 0.f;
    for (int r = r0; r < r1; r++) acc += sg_bf16_to_f32(x[(bh * rows + r) * SG_D + d]);
    atomicAdd(sum + bh * SG_D + d, acc);
}

__device__ __forceinline__ unsigned int sg_pack_i8(float x0, float x1, float x2, float x3) {
    const int a = max(-127, min(127, __float2int_rn(x0)));
    const int b = max(-127, min(127, __float2int_rn(x1)));
    const int c = max(-127, min(127, __float2int_rn(x2)));
    const int d = max(-127, min(127, __float2int_rn(x3)));
    return (unsigned int)(a & 0xff) | ((unsigned int)(b & 0xff) << 8) |
           ((unsigned int)(c & 0xff) << 16) | ((unsigned int)(d & 0xff) << 24);
}

// x [bh, rows, 128] bf16 -> out [bh, rows_pad, 128] int8, scales [bh,
// rows_pad / grp] (grp 16: one per warp, 64: one per block). grid (rows_pad /
// 64, bh), 128 threads; warp w owns rows [16w, 16w + 16) of the block, lane l
// dims [4l, 4l + 4). With `smooth`, colsum * inv_rows is subtracted first.
extern "C" __global__ void __launch_bounds__(128) attn_sage_quant_i8(
    const sg_u16* __restrict__ x, const float* __restrict__ colsum, float inv_rows, int smooth,
    sg_u8* __restrict__ out, float* __restrict__ scales, int rows, int rows_pad, int grp
) {
    __shared__ float wmax[4];
    const long bh = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * 64 + warp * 16;
    float mean[4] = {0.f, 0.f, 0.f, 0.f};
    if (smooth) {
        #pragma unroll
        for (int i = 0; i < 4; i++) mean[i] = colsum[bh * SG_D + 4 * lane + i] * inv_rows;
    }
    float v[16][4];
    float amax = 0.f;
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        const int row = row0 + r;
        if (row < rows) {
            const uint2 u = *reinterpret_cast<const uint2*>(x + (bh * rows + row) * SG_D + 4 * lane);
            v[r][0] = __uint_as_float(u.x << 16) - mean[0];
            v[r][1] = __uint_as_float(u.x & 0xffff0000u) - mean[1];
            v[r][2] = __uint_as_float(u.y << 16) - mean[2];
            v[r][3] = __uint_as_float(u.y & 0xffff0000u) - mean[3];
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
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        const long row = row0 + r;
        *reinterpret_cast<unsigned int*>(out + (bh * rows_pad + row) * SG_D + 4 * lane) =
            sg_pack_i8(v[r][0] * inv, v[r][1] * inv, v[r][2] * inv, v[r][3] * inv);
    }
    if (lane == 0) {
        if (grp == 64) {
            if (warp == 0) scales[bh * (rows_pad / 64) + blockIdx.x] = scale;
        } else {
            scales[bh * (rows_pad / 16) + row0 / 16] = scale;
        }
    }
}

// vmax [bh, 128] (zeroed; float bits, non-negative so integer order holds):
// grid (ceil(rows / rpb), bh), 128 threads (one per dim).
extern "C" __global__ void attn_sage_vmax(
    const sg_u16* __restrict__ v, unsigned int* __restrict__ vmax, int rows, int rpb
) {
    const long bh = blockIdx.y;
    const int d = threadIdx.x;
    const int r0 = blockIdx.x * rpb, r1 = min(rows, r0 + rpb);
    float m = 0.f;
    for (int r = r0; r < r1; r++) m = fmaxf(m, fabsf(sg_bf16_to_f32(v[(bh * rows + r) * SG_D + d])));
    atomicMax(vmax + bh * SG_D + d, __float_as_uint(m));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
// Four f32 -> four e4m3 bytes (RNE, saturating), x0 in the lowest byte.
__device__ __forceinline__ unsigned int sg_pack_e4m3(float x0, float x1, float x2, float x3) {
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(x1), "f"(x0));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(x3), "f"(x2));
    return (unsigned int)lo | ((unsigned int)hi << 16);
}
#endif

// Key held by slot s (0..15) of a 16-key group: the k32 MMA's A operand gives
// thread t slots 4t..4t+3; the QK accumulator gives it keys 2t, 2t+1 of two
// adjacent 8-key blocks.
__device__ __forceinline__ int sg_perm16(int s) {
    const int t = s >> 2, i = s & 3;
    return (i < 2 ? 2 * t + i : 8 + 2 * t + (i - 2));
}

// V bf16 [bh, rows, 128] -> vt e4m3 [bh, 128, rows_pad] (keys permuted per
// 16-key group, zero past `rows`) and vs [bh, 128] f32 channel scales.
// grid (rows_pad / 64, bh), 128 threads (one per dim).
extern "C" __global__ void __launch_bounds__(128) attn_sage_vquant(
    const sg_u16* __restrict__ v, const unsigned int* __restrict__ vmax,
    sg_u8* __restrict__ vt, float* __restrict__ vs, int rows, int rows_pad
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __shared__ sg_u16 tile[64][SG_D + 2];
    const long bh = blockIdx.y;
    const int d = threadIdx.x;
    const int k0 = blockIdx.x * 64;
    for (int r = 0; r < 64; r++) {
        const int row = k0 + r;
        tile[r][d] = row < rows ? v[(bh * rows + row) * SG_D + d] : (sg_u16)0;
    }
    __syncthreads();
    const float amax = __uint_as_float(vmax[bh * SG_D + d]);
    const float scale = amax > 0.f ? amax / 448.f : 1.f;
    const float inv = amax > 0.f ? 448.f / amax : 0.f;
    if (blockIdx.x == 0) vs[bh * SG_D + d] = scale;
    unsigned int* dst = reinterpret_cast<unsigned int*>(vt + (bh * SG_D + d) * (long)rows_pad + k0);
    #pragma unroll
    for (int w = 0; w < 16; w++) {
        const int g = (w >> 2) * 16, s = (w & 3) * 4;
        dst[w] = sg_pack_e4m3(sg_bf16_to_f32(tile[g + sg_perm16(s + 0)][d]) * inv,
                              sg_bf16_to_f32(tile[g + sg_perm16(s + 1)][d]) * inv,
                              sg_bf16_to_f32(tile[g + sg_perm16(s + 2)][d]) * inv,
                              sg_bf16_to_f32(tile[g + sg_perm16(s + 3)][d]) * inv);
    }
#else
    (void)v; (void)vmax; (void)vt; (void)vs; (void)rows; (void)rows_pad;
    __trap();
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
// Swizzled byte offsets: 128-byte rows XOR the 16-byte chunk with row & 7;
// 64-byte rows (two per 128-byte line) with (row >> 1) & 3.
__device__ __forceinline__ unsigned int sg_swz128(int row, int byte) {
    return (unsigned int)row * 128u + ((((unsigned int)byte >> 4) ^ ((unsigned int)row & 7u)) << 4) + ((unsigned int)byte & 15u);
}
__device__ __forceinline__ unsigned int sg_swz64(int row, int byte) {
    return (unsigned int)row * 64u + ((((unsigned int)byte >> 4) ^ (((unsigned int)row >> 1) & 3u)) << 4) + ((unsigned int)byte & 15u);
}
__device__ __forceinline__ unsigned int sg_smem_u32(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void sg_cp16(unsigned int smem, const void* gmem, int valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void sg_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void sg_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
// ROWS x 128-byte rows (global stride 128) with 256 threads.
template <int ROWS>
__device__ __forceinline__ void sg_load128(unsigned int smem, const unsigned char* g, int tid) {
    #pragma unroll
    for (int i = 0; i < ROWS * 8 / 256; i++) {
        const int chunk = tid + i * 256;
        const int row = chunk >> 3, c = chunk & 7;
        sg_cp16(smem + sg_swz128(row, c * 16), g + (long)row * 128 + c * 16, 1);
    }
}
// V^T tile: 128 dim rows x 64 key bytes, global row stride `gs` bytes.
__device__ __forceinline__ void sg_load_vt(unsigned int smem, const unsigned char* g, long gs, int tid) {
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int chunk = tid + i * 256;
        const int row = chunk >> 2, c = chunk & 3;
        sg_cp16(smem + sg_swz64(row, c * 16), g + (long)row * gs + c * 16, 1);
    }
}
__device__ __forceinline__ void sg_ldm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void sg_mma_s8(int* c, const unsigned int* a, const unsigned int* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void sg_mma_e4m3(float* c, const unsigned int* a, const unsigned int* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned int sg_pack_bf16(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ float sg_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
#endif

// Dense SDPA: CTA = 128 queries x (batch*head), 8 warps x 16 rows, double-
// buffered (K8, V8^T) stages. q8 [bh, sq_pad, 128] int8 + qs [bh, sq_pad /
// 16]; k8 [bh, sk_pad, 128] int8 + ks [bh, sk_pad / 64]; vt [bh, 128, sk_pad]
// e4m3 + vs [bh, 128]. Output f32 or bf16 [bh, sq, 128]. Dynamic shared
// memory: 2 * 16 KB (Q, 16 KB, is staged in stage 1 before the loop).
extern "C" __global__ void __launch_bounds__(256, 1) attn_sage_fwd_d128(
    const sg_u8* __restrict__ q8, const float* __restrict__ qs,
    const sg_u8* __restrict__ k8, const float* __restrict__ ks,
    const sg_u8* __restrict__ vt, const float* __restrict__ vs,
    float* __restrict__ out, sg_u16* __restrict__ out_bf16,
    int out_is_bf16, int sq, int sk, int sq_pad, int sk_pad, float sl2
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    extern __shared__ __align__(128) unsigned char sg_smem[];
    constexpr int BR = 128;
    const unsigned int base = sg_smem_u32(sg_smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int q0 = (int)blockIdx.x * BR;
    const long bh = blockIdx.y;
    if (q0 >= sq || sk <= 0) return;
    const int qlen = min(BR, sq - q0);
    const unsigned char* Qh = q8 + (bh * sq_pad + q0) * SG_D;
    const unsigned char* Kh = k8 + bh * (long)sk_pad * SG_D;
    const unsigned char* Vh = vt + bh * (long)SG_D * sk_pad;
    const float* ksh = ks + bh * (long)(sk_pad / SG_TILE);
    const float qscale = qs[bh * (long)(sq_pad / 16) + (q0 >> 4) + warp];
    const int nkt = (sk + SG_TILE - 1) / SG_TILE;

    // Q (128 int8 rows, 16 KB) -> stage 1; (K8_0, V8^T_0) -> stage 0.
    sg_load128<BR>(base + SG_STAGE, Qh, tid);
    sg_commit();
    sg_load128<SG_TILE>(base, Kh, tid);
    sg_load_vt(base + SG_KB, Vh, sk_pad, tid);
    sg_commit();
    sg_wait<1>();
    __syncthreads();
    unsigned int qf[4][4];
    #pragma unroll
    for (int kc = 0; kc < 4; kc++) {
        const int row = warp * 16 + (lane & 15), byte = kc * 32 + (lane >> 4) * 16;
        sg_ldm_x4(base + SG_STAGE + sg_swz128(row, byte), qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3]);
    }

    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    const float NEG = __int_as_float(0xff800000);
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;

    #pragma unroll 1
    for (int j = 0; j < nkt; j++) {
        const int kv0 = j * SG_TILE, len = min(SG_TILE, sk - kv0);
        const unsigned int sK = base + (j & 1) * SG_STAGE, sV = sK + SG_KB;
        sg_wait<0>();
        __syncthreads();   // stage j landed; every warp is past j-1 (and past Q)
        if (j + 1 < nkt) {
            const unsigned int nK = base + ((j + 1) & 1) * SG_STAGE;
            const int nkv = kv0 + SG_TILE;
            sg_load128<SG_TILE>(nK, Kh + (long)nkv * SG_D, tid);
            sg_load_vt(nK + SG_KB, Vh + nkv, sk_pad, tid);
            sg_commit();
        }
        // S[16 x 64] = Q8 K8^T (exact int32)
        int si[8][4];
        #pragma unroll
        for (int n = 0; n < 8; n++) { si[n][0] = si[n][1] = si[n][2] = si[n][3] = 0; }
        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            #pragma unroll
            for (int n = 0; n < 8; n += 2) {
                unsigned int b[4];
                const int row = n * 8 + (lane & 7) + ((lane >> 4) << 3);
                const int byte = kc * 32 + ((lane >> 3) & 1) * 16;
                sg_ldm_x4(sK + sg_swz128(row, byte), b[0], b[1], b[2], b[3]);
                sg_mma_s8(si[n], qf[kc], b);
                sg_mma_s8(si[n + 1], qf[kc], b + 2);
            }
        }
        // online softmax (f32), dequantized by q_scale * k_scale
        const float dq = qscale * ksh[j];
        float s[8][4];
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            const int c0 = n * 8 + 2 * t;
            s[n][0] = c0 < len ? (float)si[n][0] * dq : NEG;
            s[n][2] = c0 < len ? (float)si[n][2] * dq : NEG;
            s[n][1] = c0 + 1 < len ? (float)si[n][1] * dq : NEG;
            s[n][3] = c0 + 1 < len ? (float)si[n][3] * dq : NEG;
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
        const float a0 = sg_exp2(fmaf(m0, sl2, -ms0)), a1 = sg_exp2(fmaf(m1, sl2, -ms1));
        m0 = mn0; m1 = mn1;
        float ls0 = 0.f, ls1 = 0.f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            s[n][0] = sg_exp2(fmaf(s[n][0], sl2, -ms0));
            s[n][1] = sg_exp2(fmaf(s[n][1], sl2, -ms0));
            s[n][2] = sg_exp2(fmaf(s[n][2], sl2, -ms1));
            s[n][3] = sg_exp2(fmaf(s[n][3], sl2, -ms1));
            ls0 += s[n][0] + s[n][1];
            ls1 += s[n][2] + s[n][3];
        }
        l0 = fmaf(l0, a0, ls0);
        l1 = fmaf(l1, a1, ls1);
        #pragma unroll
        for (int n = 0; n < 16; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
        // P -> e4m3 A fragments (k32 chunk kc = key blocks 4kc..4kc+3)
        unsigned int pa[2][4];
        #pragma unroll
        for (int kc = 0; kc < 2; kc++) {
            const int nb = 4 * kc;
            pa[kc][0] = sg_pack_e4m3(s[nb][0] * SG_P_SCALE, s[nb][1] * SG_P_SCALE, s[nb + 1][0] * SG_P_SCALE, s[nb + 1][1] * SG_P_SCALE);
            pa[kc][1] = sg_pack_e4m3(s[nb][2] * SG_P_SCALE, s[nb][3] * SG_P_SCALE, s[nb + 1][2] * SG_P_SCALE, s[nb + 1][3] * SG_P_SCALE);
            pa[kc][2] = sg_pack_e4m3(s[nb + 2][0] * SG_P_SCALE, s[nb + 2][1] * SG_P_SCALE, s[nb + 3][0] * SG_P_SCALE, s[nb + 3][1] * SG_P_SCALE);
            pa[kc][3] = sg_pack_e4m3(s[nb + 2][2] * SG_P_SCALE, s[nb + 2][3] * SG_P_SCALE, s[nb + 3][2] * SG_P_SCALE, s[nb + 3][3] * SG_P_SCALE);
        }
        // O[16 x 128] += P8[16 x 64] V8[64 x 128] (V^T tile: dim rows, key bytes)
        #pragma unroll
        for (int kc = 0; kc < 2; kc++) {
            #pragma unroll
            for (int n = 0; n < 16; n += 2) {
                unsigned int b[4];
                const int row = n * 8 + (lane & 7) + ((lane >> 4) << 3);
                const int byte = kc * 32 + ((lane >> 3) & 1) * 16;
                sg_ldm_x4(sV + sg_swz64(row, byte), b[0], b[1], b[2], b[3]);
                sg_mma_e4m3(o[n], pa[kc], b);
                sg_mma_e4m3(o[n + 1], pa[kc], b + 2);
            }
        }
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / (l0 * SG_P_SCALE) : 0.f;
    const float inv1 = l1 > 0.f ? 1.f / (l1 * SG_P_SCALE) : 0.f;
    const float* vsh = vs + bh * SG_D;
    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool ok0 = row0 < qlen, ok1 = row1 < qlen;
    const long r0 = (bh * sq + q0 + row0) * SG_D, r1 = r0 + 8L * SG_D;
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        const int col = n * 8 + 2 * t;
        const float c0 = vsh[col], c1 = vsh[col + 1];
        if (out_is_bf16) {
            if (ok0) *reinterpret_cast<unsigned int*>(out_bf16 + r0 + col) = sg_pack_bf16(o[n][0] * inv0 * c0, o[n][1] * inv0 * c1);
            if (ok1) *reinterpret_cast<unsigned int*>(out_bf16 + r1 + col) = sg_pack_bf16(o[n][2] * inv1 * c0, o[n][3] * inv1 * c1);
        } else {
            if (ok0) *reinterpret_cast<float2*>(out + r0 + col) = make_float2(o[n][0] * inv0 * c0, o[n][1] * inv0 * c1);
            if (ok1) *reinterpret_cast<float2*>(out + r1 + col) = make_float2(o[n][2] * inv1 * c0, o[n][3] * inv1 * c1);
        }
    }
#else
    (void)q8; (void)qs; (void)k8; (void)ks; (void)vt; (void)vs; (void)out; (void)out_bf16;
    (void)out_is_bf16; (void)sq; (void)sk; (void)sq_pad; (void)sk_pad; (void)sl2;
    __trap();
#endif
}
