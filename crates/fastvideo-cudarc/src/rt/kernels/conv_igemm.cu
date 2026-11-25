// ---- Convolutions without cuDNN (ADR-0024 step 3, D6) ----------------------
//
// The dense N-D convolution as an implicit GEMM on channels-last bf16 with f32
// accumulation (rt/conv.rs `conv3d_igemm`): the im2col gather is the A-tile
// producer of an mma.sync.m16n8k16 bf16 GEMM, so no unfolded matrix is ever
// written. Two tile configs (WO-3b increment 1, after the WO-3a spike,
// docs/perf/2026-11-25-conv-routes.md):
//
//   conv_igemm_bf16      128 x 128 x 32, 8 warps (2 x 4, 64 x 32 per warp),
//                        3-stage cp.async pipeline (48 KB smem): the wide,
//                        many-pixel stages.
//   conv_igemm_bf16_64    64 x  64 x 32, 4 warps (2 x 2, 32 x 32 per warp),
//                        3 stages (24 KB): the small-M / small-N shapes where
//                        the 128 tile leaves most SMs idle.
//
// Both take a split-K factor (grid.z): split z sums k-tiles [z*nk/splits,
// (z+1)*nk/splits) into its own f32 [M, N] plane, and conv_nhwc_to_nchw_f32
// adds the planes in z order while transposing back, so the result is the
// same bits on every run (no atomics; docs/perf/determinism.md).
//
// The spike's unfold route (conv_unfold_nhwc_bf16 + cuBLASLt) stays as the
// zero-kernel fallback. Constraints: C_in a multiple of 8 (a 16-byte cp.async
// chunk never straddles a tap), groups 1, dilation as a parameter, zero
// padding (out-of-range taps are zero-filled by cp.async, never read).

#define IG_BK 32
// Bytes per shared-memory row: IG_BK bf16 = 4 chunks of 16 B.
#define IG_ROWB 64

// Swizzle for 64-byte rows: ldmatrix reads 8 consecutive rows at one 16 B
// chunk column; rows r and r + 2 would share banks, so chunk ^= (row >> 1) & 3.
__device__ __forceinline__ unsigned int ig_swz(int row, int chunk) {
    return (unsigned int)row * IG_ROWB + ((((unsigned int)chunk) ^ (((unsigned int)row >> 1) & 3u)) << 4);
}

// [n, c, s] f32 -> [n, s, c] bf16 (round to nearest even), 32 x 32 tiles
// through shared memory so both sides coalesce. Grid (ceil(s/32), ceil(c/32),
// n), block (32, 8).
extern "C" __global__ void conv_nchw_to_nhwc_bf16(
    const float* __restrict__ x, unsigned short* __restrict__ out, long c, long s
) {
    __shared__ float tile[32][33];
    const long b = blockIdx.z;
    const long s0 = (long)blockIdx.x * 32, c0 = (long)blockIdx.y * 32;
    const float* xb = x + b * c * s;
    for (int j = threadIdx.y; j < 32; j += 8) {
        long cc = c0 + j, ss = s0 + threadIdx.x;
        tile[j][threadIdx.x] = (cc < c && ss < s) ? xb[cc * s + ss] : 0.0f;
    }
    __syncthreads();
    unsigned short* ob = out + b * s * c;
    for (int j = threadIdx.y; j < 32; j += 8) {
        long ss = s0 + j, cc = c0 + threadIdx.x;
        if (ss < s && cc < c) ob[ss * c + cc] = fv_bf16_rne(tile[threadIdx.x][j]);
    }
}

// [splits][n, s, c] f32 -> [n, c, s] f32: the transpose back, summing the
// split-K planes (`plane` = n * s * c elements each) in z order. Grid
// (ceil(c/32), ceil(s/32), n).
extern "C" __global__ void conv_nhwc_to_nchw_f32(
    const float* __restrict__ y, float* __restrict__ out, long c, long s, long splits, long plane
) {
    __shared__ float tile[32][33];
    const long b = blockIdx.z;
    const long c0 = (long)blockIdx.x * 32, s0 = (long)blockIdx.y * 32;
    const float* yb = y + b * s * c;
    for (int j = threadIdx.y; j < 32; j += 8) {
        long ss = s0 + j, cc = c0 + threadIdx.x;
        float v = 0.0f;
        if (ss < s && cc < c) {
            const long off = ss * c + cc;
            for (long z = 0; z < splits; z++) v += yb[z * plane + off];
        }
        tile[j][threadIdx.x] = v;
    }
    __syncthreads();
    float* ob = out + b * c * s;
    for (int j = threadIdx.y; j < 32; j += 8) {
        long cc = c0 + j, ss = s0 + threadIdx.x;
        if (cc < c && ss < s) ob[cc * s + ss] = tile[threadIdx.x][j];
    }
}

// Weight [oc, ic, taps] f32 -> bf16 with k = tap * ic + ci: `kn` = 0 writes
// [oc, K] (the igemm B operand, K contiguous), `kn` = 1 writes [K, oc] (the
// cuBLASLt B operand of the unfold route).
extern "C" __global__ void conv_weight_bf16(
    const float* __restrict__ w, unsigned short* __restrict__ out, long oc, long ic, long taps, int kn
) {
    long i = IDX();
    long total = oc * ic * taps;
    if (i >= total) return;
    // i enumerates the source: (o, ci, tap).
    long tap = i % taps;
    long r = i / taps;
    long ci = r % ic;
    long o = r / ic;
    long k = tap * ic + ci;
    long K = taps * ic;
    unsigned short v = fv_bf16_rne(w[i]);
    if (kn) out[k * oc + o] = v; else out[o * K + k] = v;
}

// im2col rows for output pixels m0 .. m0 + mc of an NDHWC bf16 activation:
// a[(m - m0) * K + k] with k = tap * C + ci. Zero where the tap falls outside
// the input (zero padding).
extern "C" __global__ void conv_unfold_nhwc_bf16(
    const unsigned short* __restrict__ x, unsigned short* __restrict__ a, long m0, long mc,
    int D, int H, int W, int C, int OD, int OH, int OW,
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW
) {
    long i = IDX();
    long K = (long)KD * KH * KW * C;
    if (i >= mc * K) return;
    long r = i / K;
    int k = (int)(i - r * K);
    long m = m0 + r;
    int ow = (int)(m % OW); long q = m / OW;
    int oh = (int)(q % OH); q /= OH;
    int od = (int)(q % OD);
    long b = q / OD;
    int tap = k / C, ci = k - tap * C;
    int kd = tap / (KH * KW), rem = tap - kd * KH * KW, kh = rem / KW, kw = rem - kh * KW;
    int id = od * SD - PD + kd * DD, ih = oh * SH - PH + kh * DH, iw = ow * SW - PW + kw * DW;
    unsigned short v = 0;
    if ((unsigned)id < (unsigned)D && (unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
        v = x[((((b * D + id) * H + ih) * W + iw) * (long)C) + ci];
    }
    a[i] = v;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// y[z][m, n] (f32, NDHWC rows) = sum over this split's k of A[m, k] W[n, k],
// A gathered from the NDHWC bf16 activation on the fly.
//   BM x BN: the CTA tile; WM x WN: one warp's tile; STAGES: cp.async ring.
//   Threads: (BM / WM) * (BN / WN) * 32. Grid (ceil(N / BN), ceil(M / BM), splits).
template <int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void ig_body(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ y,
    int B, int D, int H, int W, int C, int OD, int OH, int OW, int N,
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW,
    int splits, unsigned char* smem
) {
    constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
    constexpr int NT = WARPS_M * WARPS_N * 32;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int A_CH = BM * 4 / NT, B_CH = BN * 4 / NT;   // 16 B chunks per thread per stage
    constexpr int TILE_A = BM * IG_ROWB, TILE_B = BN * IG_ROWB, STAGE_B = TILE_A + TILE_B;
    static_assert(A_CH * NT == BM * 4 && B_CH * NT == BN * 4, "chunks split evenly over the threads");

    const unsigned int sbase = mma_smem_u32(smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const long M = (long)B * OD * OH * OW;
    const long m0 = (long)blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int taps = KD * KH * KW;
    const long K = (long)taps * C;
    const int nk_all = (int)((K + IG_BK - 1) / IG_BK);
    const int z = blockIdx.z;
    const int kt0 = (int)((long)nk_all * z / splits), kt1 = (int)((long)nk_all * (z + 1) / splits);
    const int nk = kt1 - kt0;
    if (m0 >= M || n0 >= N) return;
    y += (long)z * M * N;

    // Loader state: A chunk = tid + i * NT, row = chunk >> 2, chunk column cc =
    // chunk & 3; the row's output pixel decoded once.
    int a_b[A_CH], a_od[A_CH], a_oh[A_CH], a_ow[A_CH], a_ok[A_CH];
    #pragma unroll
    for (int i = 0; i < A_CH; i++) {
        int row = (tid + i * NT) >> 2;
        long m = m0 + row;
        a_ok[i] = m < M;
        long mm = a_ok[i] ? m : 0;
        a_ow[i] = (int)(mm % OW); long q = mm / OW;
        a_oh[i] = (int)(q % OH); q /= OH;
        a_od[i] = (int)(q % OD);
        a_b[i] = (int)(q / OD);
    }

    float acc[MT][NTL][4];
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
        #pragma unroll
        for (int nt = 0; nt < NTL; nt++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;

    auto load_stage = [&](int kt, int stage) {
        const unsigned int sA = sbase + stage * STAGE_B, sB = sA + TILE_A;
        const long kbase = (long)(kt0 + kt) * IG_BK;
        #pragma unroll
        for (int i = 0; i < A_CH; i++) {
            const int chunk = tid + i * NT;
            const int row = chunk >> 2, cc = chunk & 3;
            const long k = kbase + cc * 8;
            // 8 consecutive k share one tap (C % 8 == 0).
            int ok = a_ok[i] && k < K;
            const unsigned short* src = x;
            if (ok) {
                int kk = (int)k;
                int tap = kk / C, ci = kk - tap * C;
                int kd = tap / (KH * KW), rem = tap - kd * KH * KW, kh = rem / KW, kw = rem - kh * KW;
                int id = a_od[i] * SD - PD + kd * DD, ih = a_oh[i] * SH - PH + kh * DH, iw = a_ow[i] * SW - PW + kw * DW;
                ok = (unsigned)id < (unsigned)D && (unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W;
                if (ok) src = x + ((((long)a_b[i] * D + id) * H + ih) * W + iw) * (long)C + ci;
            }
            mma_cp_async16_zfill(sA + ig_swz(row, cc), src, ok);
        }
        #pragma unroll
        for (int i = 0; i < B_CH; i++) {
            const int chunk = tid + i * NT;
            const int row = chunk >> 2, cc = chunk & 3;
            const long k = kbase + cc * 8;
            const int n = n0 + row;
            const int bok = (n < N) && (k < K);
            mma_cp_async16_zfill(sB + ig_swz(row, cc), bok ? (const void*)(w + (long)n * K + k) : (const void*)w, bok);
        }
    };

    // Prologue: STAGES - 1 stages in flight, one commit group each (an empty
    // group past the end keeps the wait counts uniform).
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nk) load_stage(s, s);
        mma_cp_commit();
    }
    for (int kt = 0; kt < nk; kt++) {
        mma_cp_wait<STAGES - 2>();
        __syncthreads();
        // Every thread is past compute(kt - 1): its stage is free for kt + STAGES - 1.
        {
            const int kn = kt + STAGES - 1;
            if (kn < nk) load_stage(kn, kn % STAGES);
            mma_cp_commit();
        }
        const unsigned int sA = sbase + (kt % STAGES) * STAGE_B, sB = sA + TILE_A;
        #pragma unroll
        for (int kc = 0; kc < IG_BK / 16; kc++) {
            unsigned int af[MT][4];
            #pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const int row = wm * WM + mt * 16 + (lane & 15);
                const int col = kc * 16 + (lane >> 4) * 8;
                mma_ldm_x4(sA + ig_swz(row, col >> 3), af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
            }
            #pragma unroll
            for (int nt = 0; nt < NTL; nt++) {
                unsigned int bfr[2];
                const int row = wn * WN + nt * 8 + (lane & 7);
                const int col = kc * 16 + ((lane >> 3) & 1) * 8;
                mma_ldm_x2(sB + ig_swz(row, col >> 3), bfr[0], bfr[1]);
                #pragma unroll
                for (int mt = 0; mt < MT; mt++) mma_bf16(acc[mt][nt], af[mt], bfr);
            }
        }
    }
    mma_cp_wait<0>();

    // C fragment: c0,c1 = (row g, cols 2t, 2t+1); c2,c3 = (row g + 8, same cols).
    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        #pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
            const int n = n0 + wn * WN + nt * 8 + 2 * t;
            #pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const long m = m0 + wm * WM + mt * 16 + g + hh * 8;
                if (m < M) {
                    float* yr = y + m * N;
                    if (n < N) yr[n] = acc[mt][nt][hh * 2];
                    if (n + 1 < N) yr[n + 1] = acc[mt][nt][hh * 2 + 1];
                }
            }
        }
    }
}
#endif

#define IG_ARGS \
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ y, \
    int B, int D, int H, int W, int C, int OD, int OH, int OW, int N, \
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW, int splits
#define IG_PASS x, w, y, B, D, H, W, C, OD, OH, OW, N, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, splits

// Variants timed side by side by fv-gpucheck `conv_routes` (igemm_128s3,
// igemm_128s2, igemm_64s3, igemm_64s2); rt/conv.rs `IgemmPlan` picks one.
// Every variant asks for two CTAs per SM: the 2026-11-25 run with one CTA
// per SM on the 3-stage 128 tile (`__launch_bounds__(256, 1)`, 48 KB) was
// 1.5-2.1x cuDNN against the 2-stage spike's 1.06-1.35x.

// 128 x 128 x 32, 256 threads, 3 stages: 3 * (128 + 128) * 64 B = 48 KB.
extern "C" __global__ void __launch_bounds__(256, 2) conv_igemm_bf16(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[3 * (128 + 128) * IG_ROWB];
    ig_body<128, 128, 64, 32, 3>(IG_PASS, smem);
#endif
}

// 128 x 128 x 32, 256 threads, 2 stages (the spike's config): 32 KB.
extern "C" __global__ void __launch_bounds__(256, 2) conv_igemm_bf16_s2(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[2 * (128 + 128) * IG_ROWB];
    ig_body<128, 128, 64, 32, 2>(IG_PASS, smem);
#endif
}

// 64 x 64 x 32, 128 threads, 3 stages: 24 KB.
extern "C" __global__ void __launch_bounds__(128, 4) conv_igemm_bf16_64(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[3 * (64 + 64) * IG_ROWB];
    ig_body<64, 64, 32, 32, 3>(IG_PASS, smem);
#endif
}

// 64 x 64 x 32, 128 threads, 2 stages: 16 KB.
extern "C" __global__ void __launch_bounds__(128, 4) conv_igemm_bf16_64_s2(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[2 * (64 + 64) * IG_ROWB];
    ig_body<64, 64, 32, 32, 2>(IG_PASS, smem);
#endif
}
