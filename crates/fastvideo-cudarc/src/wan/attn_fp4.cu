// ==== region: attn_fp4 ====
// Sparse FP4 VSA attention for H3 (opt-in, lossy; crate::wan::attn_fp4,
// docs/ports/fasth3-sparse-fp4-attn.md). sm_120a only: block-scaled FP4
// mma.sync (kind::mxf4nvf4, scale_vec::4X, ue4m3 scales per 16 elements).
//
// Quantization follows FastVideo's SageAttention3 FP4 path (attn_qat_infer,
// fp4_quantization_4d.cu): every 16 consecutive elements share one E4M3
// scale amax / 6 (RN, saturating), the elements are x / scale rounded to
// E2M1 (RN, saturating); no smoothing, no global scale; inputs rounded to
// bf16 first (upstream's tensors are bf16).
//   Q, K: 16 consecutive head dims of a row;
//   V:    16 consecutive keys (tile slots) of a head dim (V^T rows);
//   P:    16 consecutive keys of a query row, per key tile, inside the
//         online softmax. Two-level by default (P~ * 448 * 6 before the
//         per-16 scale, so small probabilities keep a nonzero scale; the
//         factor is divided out with the row sum); `p_single` = upstream's
//         single-level default. The row sum is the unquantized P~.
//
// The block mask is VSA-H3's own 64-row tile lists (`selected`, as the bf16
// and FP8 fine kernels): one CTA per 64-query tile, one warp per 16 rows,
// every listed 64-key tile visited. Upstream pairs 64-token tiles into
// 128x128 blocks with quadrant bits; a 64x64 tile pair is exactly one
// quadrant, so the selected (query tile, key tile) set is the same.
//
// K tiles are stored with their rows permuted (f4_kperm, an involution) so
// the S accumulator a thread holds IS the P A-fragment of the P V mma
// (keys 8t..8t+7 and 32+8t..32+8t+7 of rows g and g+8), and ldmatrix reads
// eight consecutive stored rows per matrix. The same idea as upstream's
// permuted K quantization.
//
//   attn_fp4_quant_tile   q / k / v (f32 or bf16, packed rows) -> tile-slot FP4 + scales
//   attn_fp4_vsa          the fine stage, f32 output in tile-slot order
//   attn_fp4_mma_probe    one mma on raw fragments (layout check)

typedef unsigned char fv_u8;
typedef unsigned short fv_u16;

#define F4_D 128
#define F4_TILE 64
#define F4_ROWB 64                      // FP4 bytes per 128-dim row
#define F4_RSB 8                        // scale bytes per row (128 / 16)
#define F4_KB (F4_TILE * F4_ROWB)       // 4096: K codes, stored rows permuted
#define F4_KSB (F4_TILE * F4_RSB)       // 512:  K scales [stored row][8]
#define F4_VB (F4_D * 32)               // 4096: V^T codes [dim][64 keys / 2]
#define F4_VSB (F4_D * 4)               // 512:  V scales [dim][4]
#define F4_STAGE (F4_KB + F4_KSB + F4_VB + F4_VSB)
#define F4_QB (F4_KB + F4_KSB)          // the Q tile and its scales
#define F4_NSTAGE 3
#define F4_SMEM (F4_QB + F4_NSTAGE * F4_STAGE)
#define F4_P2 2688.0f                   // 448 * 6: two-level P pre-scale

__device__ __forceinline__ float f4_bf16r(float x) {
    unsigned int u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

// Stored row of key row k inside a 64-row K tile (and back: an involution).
// Within each 32-row half: 8n + 2t + j <-> 8t + 2n + j.
__device__ __forceinline__ int f4_kperm(int c) {
    const int l = c & 31;
    return (c & 32) | (((l >> 1) & 3) << 3) | ((l >> 3) << 1) | (l & 1);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
// x >= 0 -> E4M3 byte (RN, saturating).
__device__ __forceinline__ unsigned int f4_e4m3(float x) {
    unsigned short h;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(h) : "f"(0.f), "f"(x));
    return h & 0xffu;
}
// E4M3 byte (sign ignored) -> its value.
__device__ __forceinline__ float f4_e4m3_val(unsigned int b) {
    const unsigned int e = (b >> 3) & 15u, m = b & 7u;
    if (e == 0) return (float)m * 0.001953125f;   // m * 2^-9
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}
// Eight floats -> eight E2M1 nibbles (RN, saturating), v[0] lowest.
__device__ __forceinline__ unsigned int f4_e2m1x8(float v0, float v1, float v2, float v3,
                                                  float v4, float v5, float v6, float v7) {
    unsigned int r;
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}\n"
        : "=r"(r)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3), "f"(v4), "f"(v5), "f"(v6), "f"(v7));
    return r;
}
// 16 values -> scale byte and 2 x 8 codes (upstream's scaled_fp4_quant rule).
__device__ __forceinline__ unsigned int f4_quant16(const float* x, unsigned int& c0, unsigned int& c1) {
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[i]));
    const unsigned int sb = f4_e4m3(amax / 6.0f);
    const float sf = f4_e4m3_val(sb);
    const float inv = sf == 0.f ? 0.f : 1.0f / sf;
    c0 = f4_e2m1x8(x[0] * inv, x[1] * inv, x[2] * inv, x[3] * inv, x[4] * inv, x[5] * inv, x[6] * inv, x[7] * inv);
    c1 = f4_e2m1x8(x[8] * inv, x[9] * inv, x[10] * inv, x[11] * inv, x[12] * inv, x[13] * inv, x[14] * inv, x[15] * inv);
    return sb;
}
#endif

// q / k / v [bh, seq, 128] (f32, or bf16 when is_bf16) in packed order ->
// tile-slot FP4. slot_src[slot] = packed row or -1 (padding: zero codes and
// zero scales). grid (num_tiles, bh, 3), z: 0 = q, 1 = k, 2 = v; 128 threads.
//   qk4  [2][bh][padded][64] codes (z = 1: rows permuted within each tile)
//   qksf [2][bh][padded][8]  scales, same rows
//   v4   [bh][num_tiles][128][32] codes of V^T (key pairs per byte)
//   vsf  [bh][num_tiles][128][4]  scales (16 keys each)
extern "C" __global__ void __launch_bounds__(128) attn_fp4_quant_tile(
    const void* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v, int is_bf16,
    const int* __restrict__ slot_src, int seq, int num_tiles,
    fv_u8* __restrict__ qk4, fv_u8* __restrict__ qksf, fv_u8* __restrict__ v4, fv_u8* __restrict__ vsf
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const int tile = blockIdx.x, which = blockIdx.z, tid = threadIdx.x;
    const long bh = blockIdx.y, nbh = gridDim.y;
    const long padded = (long)num_tiles * F4_TILE;
    const void* x = which == 0 ? q : (which == 1 ? k : v);
    __shared__ int src[F4_TILE];
    if (tid < F4_TILE) src[tid] = slot_src[tile * F4_TILE + tid];
    __syncthreads();
    if (which < 2) {
        // Two threads per row, 64 dims each (4 scale groups).
        const int r = tid >> 1, h = tid & 1;
        const int s = src[r];
        float vals[64];
        if (s < 0) {
            #pragma unroll
            for (int i = 0; i < 64; i++) vals[i] = 0.f;
        } else if (is_bf16) {
            const uint4* p = reinterpret_cast<const uint4*>(
                reinterpret_cast<const fv_u16*>(x) + (bh * seq + s) * F4_D + h * 64);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint4 u = p[i];
                const unsigned int w[4] = {u.x, u.y, u.z, u.w};
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    vals[8 * i + 2 * j] = __uint_as_float(w[j] << 16);
                    vals[8 * i + 2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
                }
            }
        } else {
            const float4* p = reinterpret_cast<const float4*>(
                reinterpret_cast<const float*>(x) + (bh * seq + s) * F4_D + h * 64);
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                const float4 f = p[i];
                vals[4 * i] = f4_bf16r(f.x);
                vals[4 * i + 1] = f4_bf16r(f.y);
                vals[4 * i + 2] = f4_bf16r(f.z);
                vals[4 * i + 3] = f4_bf16r(f.w);
            }
        }
        const int pr = which == 1 ? f4_kperm(r) : r;
        const long row = ((long)which * nbh + bh) * padded + (long)tile * F4_TILE + pr;
        unsigned int c[8], sfw = 0;
        #pragma unroll
        for (int g = 0; g < 4; g++) sfw |= f4_quant16(vals + 16 * g, c[2 * g], c[2 * g + 1]) << (8 * g);
        uint4* o = reinterpret_cast<uint4*>(qk4 + row * F4_ROWB + h * 32);
        o[0] = make_uint4(c[0], c[1], c[2], c[3]);
        o[1] = make_uint4(c[4], c[5], c[6], c[7]);
        *reinterpret_cast<unsigned int*>(qksf + row * F4_RSB + h * 4) = sfw;
    } else {
        // One thread per head dim: its 64 slots, 4 groups of 16 keys.
        const int d = tid;
        float vals[64];
        #pragma unroll 8
        for (int r = 0; r < F4_TILE; r++) {
            const int s = src[r];
            float val = 0.f;
            if (s >= 0) {
                val = is_bf16 ? __uint_as_float(((unsigned int)reinterpret_cast<const fv_u16*>(x)[(bh * seq + s) * F4_D + d]) << 16)
                              : f4_bf16r(reinterpret_cast<const float*>(x)[(bh * seq + s) * F4_D + d]);
            }
            vals[r] = val;
        }
        unsigned int c[8], sfw = 0;
        #pragma unroll
        for (int g = 0; g < 4; g++) sfw |= f4_quant16(vals + 16 * g, c[2 * g], c[2 * g + 1]) << (8 * g);
        const long row = (bh * num_tiles + tile) * F4_D + d;
        uint4* o = reinterpret_cast<uint4*>(v4 + row * 32);
        o[0] = make_uint4(c[0], c[1], c[2], c[3]);
        o[1] = make_uint4(c[4], c[5], c[6], c[7]);
        *reinterpret_cast<unsigned int*>(vsf + row * 4) = sfw;
    }
#else
    (void)q; (void)k; (void)v; (void)is_bf16; (void)slot_src; (void)seq; (void)num_tiles;
    (void)qk4; (void)qksf; (void)v4; (void)vsf;
    __trap();
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
__device__ __forceinline__ unsigned int f4_smem_u32(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}
// 64-byte rows (Q, K codes): 16-byte chunk c of row r at chunk c ^ ((r >> 1) & 3);
// eight consecutive rows hit eight distinct 16-byte bank groups.
__device__ __forceinline__ unsigned int f4_swz64(int r, int c) {
    return (unsigned int)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4));
}
// 32-byte rows (V^T codes): chunk c of row r at c ^ ((r >> 2) & 1).
__device__ __forceinline__ unsigned int f4_swz32(int r, int c) {
    return (unsigned int)(r * 32 + ((c ^ ((r >> 2) & 1)) << 4));
}
__device__ __forceinline__ void f4_cp16(unsigned int smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem));
}
__device__ __forceinline__ void f4_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void f4_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void f4_ldm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// D[16x8] += A[16x64] B[64x8], E2M1 operands, a UE4M3 scale per 16 k.
// A: a0 rows g, k 8t..8t+7; a1 rows g + 8; a2 / a3 the same at k + 32.
// B: b0 column g, k 8t..8t+7; b1 at k + 32. sfa: the 4 scales of row
// g + 8 (lane & 1); sfb: the 4 scales of column g (byte i: k 16i..16i+15).
__device__ __forceinline__ void f4_mma(float* c, const unsigned int* a, unsigned int b0, unsigned int b1,
                                       unsigned int sfa, unsigned int sfb) {
    const unsigned short z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}
__device__ __forceinline__ float f4_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
// One (K, V) stage: K codes (swizzled), K scales, V^T codes (swizzled), V scales.
__device__ __forceinline__ void f4_load_stage(
    unsigned int st, const fv_u8* kc, const fv_u8* ks, const fv_u8* vc, const fv_u8* vs, int tid
) {
    #pragma unroll
    for (int c = tid; c < 576; c += 128) {
        if (c < 256) {
            f4_cp16(st + f4_swz64(c >> 2, c & 3), kc + c * 16);
        } else if (c < 288) {
            f4_cp16(st + F4_KB + (c - 256) * 16, ks + (c - 256) * 16);
        } else if (c < 544) {
            const int i = c - 288;
            f4_cp16(st + F4_KB + F4_KSB + f4_swz32(i >> 1, i & 1), vc + i * 16);
        } else {
            f4_cp16(st + F4_KB + F4_KSB + F4_VB + (c - 544) * 16, vs + (c - 544) * 16);
        }
    }
}
#endif

// VSA fine stage, FP4 Q K^T and P V: one CTA per (query tile, batch*head),
// 4 warps of 16 query rows. Inputs from attn_fp4_quant_tile; `selected`
// [bh, num_tiles, topk] key tiles per query tile, `block_sizes` the live rows
// per tile. Output f32 [bh, padded, 128] in tile-slot order from query tile
// q_base on (vsa_mma_attn's contract). sl2 = softmax scale * log2(e).
extern "C" __global__ void __launch_bounds__(128, 3) attn_fp4_vsa(
    const fv_u8* __restrict__ qk4, const fv_u8* __restrict__ qksf,
    const fv_u8* __restrict__ v4, const fv_u8* __restrict__ vsf,
    const unsigned int* __restrict__ selected, const int* __restrict__ block_sizes,
    float* __restrict__ out, int num_tiles, int topk, float sl2, int q_base, int p_single
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    extern __shared__ __align__(128) unsigned char f4_smem[];
    const unsigned int base = f4_smem_u32(f4_smem);
    const unsigned char* sbase = f4_smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int qtile = q_base + (int)blockIdx.x;
    const long bh = blockIdx.y, nbh = gridDim.y;
    if (qtile >= num_tiles || topk <= 0) return;
    const long padded = (long)num_tiles * F4_TILE;
    const fv_u8* Qc = qk4 + (bh * padded + (long)qtile * F4_TILE) * F4_ROWB;
    const fv_u8* Qs = qksf + (bh * padded + (long)qtile * F4_TILE) * F4_RSB;
    const fv_u8* Kc = qk4 + (nbh + bh) * padded * F4_ROWB;
    const fv_u8* Ks = qksf + (nbh + bh) * padded * F4_RSB;
    const fv_u8* Vc = v4 + bh * (long)num_tiles * F4_VB;
    const fv_u8* Vs = vsf + bh * (long)num_tiles * F4_VSB;
    const unsigned int* sel = selected + (bh * num_tiles + qtile) * (long)topk;
    const unsigned int stage0 = base + F4_QB;

    // Q (codes + scales) and the first NSTAGE - 1 key tiles, one group each.
    for (int c = tid; c < 288; c += 128) {
        if (c < 256) f4_cp16(base + f4_swz64(c >> 2, c & 3), Qc + c * 16);
        else f4_cp16(base + F4_KB + (c - 256) * 16, Qs + (c - 256) * 16);
    }
    f4_commit();
    #pragma unroll
    for (int i = 0; i < F4_NSTAGE - 1; i++) {
        if (i < topk) {
            const long kt = sel[i];
            f4_load_stage(stage0 + i * F4_STAGE, Kc + kt * F4_KB, Ks + kt * F4_KSB, Vc + kt * F4_VB, Vs + kt * F4_VSB, tid);
        }
        f4_commit();
    }
    f4_wait<F4_NSTAGE - 1>();
    __syncthreads();
    unsigned int qa[2][4], qsf[2];
    #pragma unroll
    for (int kc = 0; kc < 2; kc++) {
        const int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        f4_ldm_x4(base + f4_swz64(row, 2 * kc + (lane >> 4)), qa[kc][0], qa[kc][1], qa[kc][2], qa[kc][3]);
        const int srow = warp * 16 + g + 8 * (lane & 1);
        qsf[kc] = *reinterpret_cast<const unsigned int*>(sbase + F4_KB + srow * F4_RSB + kc * 4);
    }

    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    const float NEG = __int_as_float(0xff800000);
    const float pmul = p_single ? 1.f : F4_P2;
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;

    #pragma unroll 1
    for (int i = 0; i < topk; i++) {
        f4_wait<F4_NSTAGE - 2>();
        __syncthreads();   // tile i landed; every warp is past tile i - 1
        {
            const int nx = i + F4_NSTAGE - 1;
            if (nx < topk) {
                const long kt = sel[nx];
                f4_load_stage(stage0 + (nx % F4_NSTAGE) * F4_STAGE, Kc + kt * F4_KB, Ks + kt * F4_KSB,
                              Vc + kt * F4_VB, Vs + kt * F4_VSB, tid);
            }
            f4_commit();
        }
        const unsigned int st = stage0 + (i % F4_NSTAGE) * F4_STAGE;
        const unsigned char* sst = sbase + F4_QB + (i % F4_NSTAGE) * F4_STAGE;
        const int len = block_sizes[sel[i]];

        // S[16 x 64] = Q K^T; stored column p is key f4_kperm(p).
        float s[8][4];
        #pragma unroll
        for (int n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
        #pragma unroll
        for (int kc = 0; kc < 2; kc++) {
            #pragma unroll
            for (int n = 0; n < 8; n += 2) {
                unsigned int b[4];
                const int row = 8 * (n + ((lane >> 4) & 1)) + (lane & 7);
                f4_ldm_x4(st + f4_swz64(row, 2 * kc + ((lane >> 3) & 1)), b[0], b[1], b[2], b[3]);
                const unsigned int sf0 = *reinterpret_cast<const unsigned int*>(sst + F4_KB + (8 * n + g) * F4_RSB + kc * 4);
                const unsigned int sf1 = *reinterpret_cast<const unsigned int*>(sst + F4_KB + (8 * n + 8 + g) * F4_RSB + kc * 4);
                f4_mma(s[n], qa[kc], b[0], b[1], qsf[kc], sf0);
                f4_mma(s[n + 1], qa[kc], b[2], b[3], qsf[kc], sf1);
            }
        }
        // Online softmax over the live keys.
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            const int c0 = 8 * n + 2 * t;
            const bool ok0 = f4_kperm(c0) < len, ok1 = f4_kperm(c0 + 1) < len;
            if (!ok0) { s[n][0] = NEG; s[n][2] = NEG; }
            if (!ok1) { s[n][1] = NEG; s[n][3] = NEG; }
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
        const float a0 = f4_exp2(fmaf(m0, sl2, -ms0)), a1 = f4_exp2(fmaf(m1, sl2, -ms1));
        m0 = mn0; m1 = mn1;
        float ls0 = 0.f, ls1 = 0.f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            s[n][0] = f4_exp2(fmaf(s[n][0], sl2, -ms0));
            s[n][1] = f4_exp2(fmaf(s[n][1], sl2, -ms0));
            s[n][2] = f4_exp2(fmaf(s[n][2], sl2, -ms1));
            s[n][3] = f4_exp2(fmaf(s[n][3], sl2, -ms1));
            ls0 += s[n][0] + s[n][1];
            ls1 += s[n][2] + s[n][3];
        }
        l0 = fmaf(l0, a0, ls0);
        l1 = fmaf(l1, a1, ls1);
        #pragma unroll
        for (int n = 0; n < 16; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }

        // P -> E2M1 with a scale per 16 keys of a row. This thread holds keys
        // 8t..8t+7 (s[0..3]) and 32 + 8t.. (s[4..7]) of rows g and g + 8:
        // half of groups t >> 1 and 2 + (t >> 1); lane ^ 1 holds the other half.
        float x0 = 0.f, x1 = 0.f, y0 = 0.f, y1 = 0.f;   // (lo, hi) group max, rows g / g + 8
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
            y0 = fmaxf(y0, fmaxf(s[n][2], s[n][3]));
            x1 = fmaxf(x1, fmaxf(s[n + 4][0], s[n + 4][1]));
            y1 = fmaxf(y1, fmaxf(s[n + 4][2], s[n + 4][3]));
        }
        x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
        x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
        y0 = fmaxf(y0, __shfl_xor_sync(0xffffffffu, y0, 1));
        y1 = fmaxf(y1, __shfl_xor_sync(0xffffffffu, y1, 1));
        const unsigned int bx0 = f4_e4m3(x0 * pmul / 6.0f), bx1 = f4_e4m3(x1 * pmul / 6.0f);
        const unsigned int by0 = f4_e4m3(y0 * pmul / 6.0f), by1 = f4_e4m3(y1 * pmul / 6.0f);
        const float fx0 = f4_e4m3_val(bx0), fx1 = f4_e4m3_val(bx1), fy0 = f4_e4m3_val(by0), fy1 = f4_e4m3_val(by1);
        const float ix0 = fx0 == 0.f ? 0.f : pmul / fx0, ix1 = fx1 == 0.f ? 0.f : pmul / fx1;
        const float iy0 = fy0 == 0.f ? 0.f : pmul / fy0, iy1 = fy1 == 0.f ? 0.f : pmul / fy1;
        unsigned int pa[4];
        pa[0] = f4_e2m1x8(s[0][0] * ix0, s[0][1] * ix0, s[1][0] * ix0, s[1][1] * ix0,
                          s[2][0] * ix0, s[2][1] * ix0, s[3][0] * ix0, s[3][1] * ix0);
        pa[1] = f4_e2m1x8(s[0][2] * iy0, s[0][3] * iy0, s[1][2] * iy0, s[1][3] * iy0,
                          s[2][2] * iy0, s[2][3] * iy0, s[3][2] * iy0, s[3][3] * iy0);
        pa[2] = f4_e2m1x8(s[4][0] * ix1, s[4][1] * ix1, s[5][0] * ix1, s[5][1] * ix1,
                          s[6][0] * ix1, s[6][1] * ix1, s[7][0] * ix1, s[7][1] * ix1);
        pa[3] = f4_e2m1x8(s[4][2] * iy1, s[4][3] * iy1, s[5][2] * iy1, s[5][3] * iy1,
                          s[6][2] * iy1, s[6][3] * iy1, s[7][2] * iy1, s[7][3] * iy1);
        // sfa: this lane supplies row g + 8 (lane & 1) groups 0..3; lane ^ 2
        // holds the other two groups of the same row.
        const unsigned int mine = (t & 1) ? (by0 | (by1 << 8)) : (bx0 | (bx1 << 8));
        const unsigned int peer = __shfl_xor_sync(0xffffffffu, mine, 2);
        const unsigned int psf = (t >> 1)
            ? ((peer & 0xffu) | ((mine & 0xffu) << 8) | ((peer >> 8) << 16) | ((mine >> 8) << 24))
            : ((mine & 0xffu) | ((peer & 0xffu) << 8) | ((mine >> 8) << 16) | ((peer >> 8) << 24));

        // O[16 x 128] += P[16 x 64] V[64 x 128].
        #pragma unroll
        for (int n = 0; n < 16; n += 2) {
            unsigned int b[4];
            const int row = 8 * (n + ((lane >> 4) & 1)) + (lane & 7);
            f4_ldm_x4(st + F4_KB + F4_KSB + f4_swz32(row, (lane >> 3) & 1), b[0], b[1], b[2], b[3]);
            const unsigned int sf0 = *reinterpret_cast<const unsigned int*>(sst + F4_KB + F4_KSB + F4_VB + (8 * n + g) * 4);
            const unsigned int sf1 = *reinterpret_cast<const unsigned int*>(sst + F4_KB + F4_KSB + F4_VB + (8 * n + 8 + g) * 4);
            f4_mma(o[n], pa, b[0], b[1], psf, sf0);
            f4_mma(o[n + 1], pa, b[2], b[3], psf, sf1);
        }
    }
    f4_wait<0>();

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / (l0 * pmul) : 0.f, inv1 = l1 > 0.f ? 1.f / (l1 * pmul) : 0.f;
    float* ob = out + (bh * padded + (long)qtile * F4_TILE + warp * 16) * F4_D;
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        const int col = n * 8 + t * 2;
        *reinterpret_cast<float2*>(ob + (long)g * F4_D + col) = make_float2(o[n][0] * inv0, o[n][1] * inv0);
        *reinterpret_cast<float2*>(ob + (long)(g + 8) * F4_D + col) = make_float2(o[n][2] * inv1, o[n][3] * inv1);
    }
#else
    (void)qk4; (void)qksf; (void)v4; (void)vsf; (void)selected; (void)block_sizes; (void)out;
    (void)num_tiles; (void)topk; (void)sl2; (void)q_base; (void)p_single;
    __trap();
#endif
}

// One warp, one mma on raw fragments: a [32][4], b [32][2], sfa [32], sfb
// [32] per lane; d [32][4] (c = 0). The layout check reads it.
extern "C" __global__ void attn_fp4_mma_probe(
    const unsigned int* __restrict__ a, const unsigned int* __restrict__ b,
    const unsigned int* __restrict__ sfa, const unsigned int* __restrict__ sfb, float* __restrict__ d
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const int l = threadIdx.x & 31;
    unsigned int ar[4] = {a[4 * l], a[4 * l + 1], a[4 * l + 2], a[4 * l + 3]};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    f4_mma(c, ar, b[2 * l], b[2 * l + 1], sfa[l], sfb[l]);
    #pragma unroll
    for (int i = 0; i < 4; i++) d[4 * l + i] = c[i];
#else
    (void)a; (void)b; (void)sfa; (void)sfb; (void)d;
    __trap();
#endif
}
// ==== end region: attn_fp4 ====
