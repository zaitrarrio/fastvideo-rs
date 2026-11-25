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

// Shared-memory rows hold one CTA tile row's BK bf16 = BK / 8 chunks of 16 B.
// Swizzle: ldmatrix reads 8 consecutive rows at one chunk column; rows that
// share a 128-byte bank window would conflict, so the chunk is XORed with the
// row's position in that window (64 B rows: (row >> 1) & 3; 128 B rows: row & 7).
template <int BK>
__device__ __forceinline__ unsigned int ig_swz(int row, int chunk) {
    constexpr int ROWB = BK * 2, CHUNKS = BK / 8;
    return (unsigned int)row * ROWB + ((((unsigned int)chunk) ^ ((((unsigned int)row * ROWB) / 128u) & (CHUNKS - 1))) << 4);
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

// out[i] = bf16(sum over z of y[z * plane + i]): the split-K planes summed in
// order into a bf16 channels-last result (no transpose; the LTX fast decoder).
extern "C" __global__ void conv_sum_planes_bf16(
    const float* __restrict__ y, unsigned short* __restrict__ out, long plane, long splits
) {
    long i = IDX();
    if (i >= plane) return;
    float v = 0.0f;
    for (long z = 0; z < splits; z++) v += y[z * plane + i];
    out[i] = fv_bf16_rne(v);
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
//   MODE 0: the convolution. MODE 1 (probe): the A tile is read from the output
//   pixel's own input location with no tap math and no bounds checks, as a
//   precomputed im2col would cost (wrong values, representative traffic).
//   MODE 2 (probe): no global loads at all, the mma / ldmatrix ceiling.
template <int BM, int BN, int BK, int WM, int WN, int STAGES, int MODE>
__device__ __forceinline__ void ig_body(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ y,
    unsigned short* __restrict__ y16, int out16,
    int B, int D, int H, int W, int C, int OD, int OH, int OW, int N,
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW,
    int splits, unsigned char* smem
) {
    constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
    constexpr int NT = WARPS_M * WARPS_N * 32;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int ROWB = BK * 2, CHUNKS = BK / 8, KC = BK / 16;
    constexpr int A_CH = BM * CHUNKS / NT, B_CH = BN * CHUNKS / NT;   // 16 B chunks per thread per stage
    constexpr int TILE_A = BM * ROWB, TILE_B = BN * ROWB, STAGE_B = TILE_A + TILE_B;
    static_assert(A_CH * NT == BM * CHUNKS && B_CH * NT == BN * CHUNKS, "chunks split evenly over the threads");
    static_assert(NTL % 2 == 0, "B fragments load in pairs");

    const unsigned int sbase = mma_smem_u32(smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const long M = (long)B * OD * OH * OW;
    const long m0 = (long)blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int taps = KD * KH * KW;
    const long K = (long)taps * C;
    const int nk_all = (int)((K + BK - 1) / BK);
    const int z = blockIdx.z;
    const int kt0 = (int)((long)nk_all * z / splits), kt1 = (int)((long)nk_all * (z + 1) / splits);
    const int nk = kt1 - kt0;
    if (m0 >= M || n0 >= N) return;
    y += (long)z * M * N;   // `out16` (bf16 straight to y16) is only used with one split.

    // Per A chunk: the output pixel's first input coordinate per axis
    // (od * stride - pad, before the tap offset) and the batch's base pointer.
    int a_d0[A_CH], a_h0[A_CH], a_w0[A_CH], a_ok[A_CH];
    const unsigned short* a_base[A_CH];
    #pragma unroll
    for (int i = 0; i < A_CH; i++) {
        int row = (tid + i * NT) / CHUNKS;
        long m = m0 + row;
        a_ok[i] = m < M;
        long mm = a_ok[i] ? m : 0;
        int ow = (int)(mm % OW); long q = mm / OW;
        int oh = (int)(q % OH); q /= OH;
        int od = (int)(q % OD);
        int b = (int)(q / OD);
        a_d0[i] = od * SD - PD;
        a_h0[i] = oh * SH - PH;
        a_w0[i] = ow * SW - PW;
        a_base[i] = x + (long)b * D * H * W * C;
    }
    const long row_h = (long)W * C, row_d = (long)H * row_h;

    float acc[MT][NTL][4];
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
        #pragma unroll
        for (int nt = 0; nt < NTL; nt++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;

    auto load_stage = [&](int kt, int stage) {
        if (MODE == 2) return;
        const unsigned int sA = sbase + stage * STAGE_B, sB = sA + TILE_A;
        const long kbase = (long)(kt0 + kt) * BK;
        if (MODE == 1) {
            #pragma unroll
            for (int i = 0; i < A_CH; i++) {
                const int chunk = tid + i * NT;
                const int row = chunk / CHUNKS, cc = chunk % CHUNKS;
                const long k = kbase + cc * 8;
                const int ok = a_ok[i] && k < K;
                const int ci = (int)(k % C);
                const int id = min(max(a_d0[i], 0), D - 1), ih = min(max(a_h0[i], 0), H - 1), iw = min(max(a_w0[i], 0), W - 1);
                const unsigned short* src = ok ? a_base[i] + id * row_d + ih * row_h + (long)iw * C + ci : x;
                mma_cp_async16_zfill(sA + ig_swz<BK>(row, cc), src, ok);
            }
            #pragma unroll
            for (int i = 0; i < B_CH; i++) {
                const int chunk = tid + i * NT;
                const int row = chunk / CHUNKS, cc = chunk % CHUNKS;
                const long k = kbase + cc * 8;
                const int n = n0 + row;
                const int bok = (n < N) && (k < K);
                mma_cp_async16_zfill(sB + ig_swz<BK>(row, cc), bok ? (const void*)(w + (long)n * K + k) : (const void*)w, bok);
            }
            return;
        }
        // The tile's first k -> (tap, channel), decoded once per k-tile per thread
        // (three divisions); each chunk's 8 channels then advance by 8 within the
        // tap, wrapping to the next tap when C < BK (C % 8 == 0 keeps a chunk
        // inside one tap).
        int tap0 = (int)(kbase / C), c0 = (int)(kbase - (long)tap0 * C);
        int kd0 = tap0 / (KH * KW), rem0 = tap0 - kd0 * KH * KW, kh0 = rem0 / KW, kw0 = rem0 - kh0 * KW;
        #pragma unroll
        for (int i = 0; i < A_CH; i++) {
            const int chunk = tid + i * NT;
            const int row = chunk / CHUNKS, cc = chunk % CHUNKS;
            const long k = kbase + cc * 8;
            int ci = c0 + cc * 8, kd = kd0, kh = kh0, kw = kw0;
            while (ci >= C) {
                ci -= C;
                if (++kw == KW) { kw = 0; if (++kh == KH) { kh = 0; ++kd; } }
            }
            int ok = a_ok[i] && k < K;
            const unsigned short* src = x;
            if (ok) {
                const int id = a_d0[i] + kd * DD, ih = a_h0[i] + kh * DH, iw = a_w0[i] + kw * DW;
                ok = (unsigned)id < (unsigned)D && (unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W;
                if (ok) src = a_base[i] + id * row_d + ih * row_h + (long)iw * C + ci;
            }
            mma_cp_async16_zfill(sA + ig_swz<BK>(row, cc), src, ok);
        }
        #pragma unroll
        for (int i = 0; i < B_CH; i++) {
            const int chunk = tid + i * NT;
            const int row = chunk / CHUNKS, cc = chunk % CHUNKS;
            const long k = kbase + cc * 8;
            const int n = n0 + row;
            const int bok = (n < N) && (k < K);
            mma_cp_async16_zfill(sB + ig_swz<BK>(row, cc), bok ? (const void*)(w + (long)n * K + k) : (const void*)w, bok);
        }
    };

    // Prologue: STAGES - 1 stages in flight, one commit group each (an empty
    // group past the end keeps the wait counts uniform).
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nk) load_stage(s, s);
        mma_cp_commit();
    }
    // Fragments of k16 chunk `kc` of the stage at `sA` / `sB` into ring slot `r`.
    unsigned int af[2][MT][4];
    unsigned int bfr[2][NTL / 2][4];
    auto load_frags = [&](unsigned int sA, unsigned int sB, int kc, int r) {
        #pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const int row = wm * WM + mt * 16 + (lane & 15);
            const int col = kc * 16 + (lane >> 4) * 8;
            mma_ldm_x4(sA + ig_swz<BK>(row, col >> 3), af[r][mt][0], af[r][mt][1], af[r][mt][2], af[r][mt][3]);
        }
        // B: one ldmatrix.x4 covers two n-tiles (lanes 0-15 the first, 16-31
        // the next), as the attention kernels load K.
        #pragma unroll
        for (int np = 0; np < NTL / 2; np++) {
            const int row = wn * WN + np * 16 + (lane & 7) + ((lane >> 4) << 3);
            const int col = kc * 16 + ((lane >> 3) & 1) * 8;
            mma_ldm_x4(sB + ig_swz<BK>(row, col >> 3), bfr[r][np][0], bfr[r][np][1], bfr[r][np][2], bfr[r][np][3]);
        }
    };
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
        // A two-deep fragment ring over the k16 chunks: the next chunk's
        // ldmatrix issue before this chunk's mma.
        load_frags(sA, sB, 0, 0);
        #pragma unroll
        for (int kc = 0; kc < KC; kc++) {
            const int r = kc & 1;
            if (kc + 1 < KC) load_frags(sA, sB, kc + 1, r ^ 1);
            #pragma unroll
            for (int np = 0; np < NTL / 2; np++) {
                #pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    mma_bf16(acc[mt][2 * np], af[r][mt], bfr[r][np]);
                    mma_bf16(acc[mt][2 * np + 1], af[r][mt], bfr[r][np] + 2);
                }
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
                    if (out16) {
                        unsigned short* yr = y16 + m * N;
                        if (n < N) yr[n] = fv_bf16_rne(acc[mt][nt][hh * 2]);
                        if (n + 1 < N) yr[n + 1] = fv_bf16_rne(acc[mt][nt][hh * 2 + 1]);
                    } else {
                        float* yr = y + m * N;
                        if (n < N) yr[n] = acc[mt][nt][hh * 2];
                        if (n + 1 < N) yr[n + 1] = acc[mt][nt][hh * 2 + 1];
                    }
                }
            }
        }
    }
}
#endif


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Named CTA barriers (ids 1..15; 0 is __syncthreads): the producer arrives, the
// consumers sync, or the other way round, `count` threads in all.
__device__ __forceinline__ void ig_bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void ig_bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

// WO-3b-ii: the warp-specialised mainloop. 128 x 128 x 32 tile, four consumer
// warps (2 x 2 of 64 x 64) that never issue a global load, one producer warp
// that fills the STAGES-deep cp.async ring (the im2col gather for A, the
// weight rows for B) and signals each stage through a named barrier; the
// consumers release a stage through another. 160 threads, 48 KB, two CTAs per
// SM. The three capped tuning rounds (docs/perf/2026-11-25-conv-routes.md)
// put the base kernel's loss against cuDNN in the loads' issue and address
// work inside the computing warps, not in latency a deeper ring would hide
// (sm_120's 128 KB per SM caps the ring at three stages anyway).
//   full[s]  (id 1 + s):          producer arrives (32), consumers sync (128).
//   empty[s] (id 1 + STAGES + s): consumers arrive (128), producer syncs (32).
// The producer signals stage kt after issuing kt + 1 and waiting for all but
// the newest group, so two stages are in flight while one is computed, as in
// the base kernel. MODE 0: the convolution. MODE 3 (probe): no epilogue
// stores (the accumulators are reduced to one NaN test per thread).
template <int STAGES, int MODE>
__device__ __forceinline__ void ig_ws_body(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ y,
    unsigned short* __restrict__ y16, int out16,
    int B, int D, int H, int W, int C, int OD, int OH, int OW, int N,
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW,
    int splits, unsigned char* smem
) {
    constexpr int BM = 128, BN = 128, BK = 32, WM = 64, WN = 64;
    constexpr int WARPS_N = BN / WN;
    constexpr int CONSUMERS = (BM / WM) * WARPS_N * 32;   // 128
    constexpr int NT = CONSUMERS + 32;                    // 160
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int ROWB = BK * 2, CHUNKS = BK / 8, KC = BK / 16;
    constexpr int A_CH = BM * CHUNKS / 32, B_CH = BN * CHUNKS / 32;   // 16 B chunks per producer lane per stage
    constexpr int TILE_A = BM * ROWB, TILE_B = BN * ROWB, STAGE_B = TILE_A + TILE_B;
    static_assert(STAGES >= 3 && STAGES <= 7, "named barrier ids 1..15");

    const unsigned int sbase = mma_smem_u32(smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long M = (long)B * OD * OH * OW;
    const long m0 = (long)blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int taps = KD * KH * KW;
    const long K = (long)taps * C;
    const int nk_all = (int)((K + BK - 1) / BK);
    const int z = blockIdx.z;
    const int kt0 = (int)((long)nk_all * z / splits), kt1 = (int)((long)nk_all * (z + 1) / splits);
    const int nk = kt1 - kt0;
    if (m0 >= M || n0 >= N || nk <= 0) return;
    y += (long)z * M * N;

    if (warp == BM / WM * WARPS_N) {
        // ---- the producer warp ----
        // Lane l owns chunk column l % CHUNKS of rows l / CHUNKS + 8 i.
        // Per row (16 per lane): the output pixel's first input coordinate per
        // axis (od * stride - pad, before the tap offset), d and h packed into
        // one int (both below 32768), w alone (INT_MIN marks a row past M: no
        // tap offset brings it in range), and the batch's element offset:
        // four registers a row (the budget is 168: five warps a CTA, ten an SM,
        // three on one sub-partition).
        const int cc = lane % CHUNKS, rowl = lane / CHUNKS;
        int a_dh[A_CH], a_w0[A_CH];
        long a_off[A_CH];
        #pragma unroll
        for (int i = 0; i < A_CH; i++) {
            const int row = rowl + i * (32 / CHUNKS);
            const long m = m0 + row;
            const int ok = m < M;
            const long mm = ok ? m : 0;
            int ow = (int)(mm % OW); long q = mm / OW;
            int oh = (int)(q % OH); q /= OH;
            int od = (int)(q % OD);
            int b = (int)(q / OD);
            a_dh[i] = ((od * SD - PD) << 16) | ((oh * SH - PH) & 0xffff);
            a_w0[i] = ok ? ow * SW - PW : (int)0x80000000;
            a_off[i] = (long)b * D * H * W * C;
        }
        const long row_h = (long)W * C, row_d = (long)H * row_h;
        auto load_stage = [&](int kt, int stage) {
            const unsigned int sA = sbase + stage * STAGE_B, sB = sA + TILE_A;
            const long kbase = (long)(kt0 + kt) * BK;
            // This lane's k is fixed per k-tile: decode (tap, channel) once.
            const long k = kbase + cc * 8;
            const int kok = k < K;
            int tap = (int)(k / C), ci = (int)(k - (long)tap * C);
            int kd = tap / (KH * KW), rem = tap - kd * KH * KW, kh = rem / KW, kw = rem - kh * KW;
            const int kdd = kd * DD, khd = kh * DH, kwd = kw * DW;
            #pragma unroll
            for (int i = 0; i < A_CH; i++) {
                const int row = rowl + i * (32 / CHUNKS);
                const int id = (a_dh[i] >> 16) + kdd, ih = (int)((short)(a_dh[i] & 0xffff)) + khd, iw = a_w0[i] + kwd;
                const int ok = kok && (unsigned)id < (unsigned)D && (unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W;
                const unsigned short* src = ok ? x + a_off[i] + id * row_d + ih * row_h + (long)iw * C + ci : x;
                mma_cp_async16_zfill(sA + ig_swz<BK>(row, cc), src, ok);
            }
            #pragma unroll
            for (int i = 0; i < B_CH; i++) {
                const int row = rowl + i * (32 / CHUNKS);
                const int n = n0 + row;
                const int bok = (n < N) && kok;
                mma_cp_async16_zfill(sB + ig_swz<BK>(row, cc), bok ? (const void*)(w + (long)n * K + k) : (const void*)w, bok);
            }
            mma_cp_commit();
        };
        for (int kt = 0; kt < nk; kt++) {
            const int s = kt % STAGES;
            if (kt >= STAGES) ig_bar_sync(1 + STAGES + s, NT);   // the consumers are done with kt - STAGES
            load_stage(kt, s);
            if (kt >= 1) {
                mma_cp_wait<1>();            // every group but the newest: kt - 1 has landed
                __threadfence_block();
                ig_bar_arrive(1 + (kt - 1) % STAGES, NT);
            }
        }
        mma_cp_wait<0>();
        __threadfence_block();
        ig_bar_arrive(1 + (nk - 1) % STAGES, NT);
        return;
    }

    // ---- the consumer warps ----
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    float acc[MT][NTL][4];
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
        #pragma unroll
        for (int nt = 0; nt < NTL; nt++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;

    // Five warps per CTA, ten per SM: three land on one of the four
    // sub-partitions, so the register budget is 168, not 204. The 64 x 64
    // warp tile's 128 accumulators leave room for one k16 chunk of fragments
    // (no two-deep ring; round 1 measured none from it).
    unsigned int af[MT][4];
    unsigned int bfr[4];
    for (int kt = 0; kt < nk; kt++) {
        const int s = kt % STAGES;
        ig_bar_sync(1 + s, NT);
        const unsigned int sA = sbase + s * STAGE_B, sB = sA + TILE_A;
        #pragma unroll
        for (int kc = 0; kc < KC; kc++) {
            #pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const int row = wm * WM + mt * 16 + (lane & 15);
                const int col = kc * 16 + (lane >> 4) * 8;
                mma_ldm_x4(sA + ig_swz<BK>(row, col >> 3), af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
            }
            #pragma unroll
            for (int np = 0; np < NTL / 2; np++) {
                const int row = wn * WN + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int col = kc * 16 + ((lane >> 3) & 1) * 8;
                mma_ldm_x4(sB + ig_swz<BK>(row, col >> 3), bfr[0], bfr[1], bfr[2], bfr[3]);
                #pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    mma_bf16(acc[mt][2 * np], af[mt], bfr);
                    mma_bf16(acc[mt][2 * np + 1], af[mt], bfr + 2);
                }
            }
        }
        ig_bar_arrive(1 + STAGES + s, NT);
    }

    if (MODE == 3) {
        float t = 0.0f;
        #pragma unroll
        for (int mt = 0; mt < MT; mt++)
            #pragma unroll
            for (int nt = 0; nt < NTL; nt++)
                #pragma unroll
                for (int e = 0; e < 4; e++) t += acc[mt][nt][e];
        if (t != t) y[0] = t;
        return;
    }
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
                    if (out16) {
                        unsigned short* yr = y16 + m * N;
                        if (n < N) yr[n] = fv_bf16_rne(acc[mt][nt][hh * 2]);
                        if (n + 1 < N) yr[n + 1] = fv_bf16_rne(acc[mt][nt][hh * 2 + 1]);
                    } else {
                        float* yr = y + m * N;
                        if (n < N) yr[n] = acc[mt][nt][hh * 2];
                        if (n + 1 < N) yr[n + 1] = acc[mt][nt][hh * 2 + 1];
                    }
                }
            }
        }
    }
}
#endif

#define IG_ARGS \
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ y, \
    unsigned short* __restrict__ y16, int out16, \
    int B, int D, int H, int W, int C, int OD, int OH, int OW, int N, \
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW, int splits
#define IG_PASS x, w, y, y16, out16, B, D, H, W, C, OD, OH, OW, N, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, splits

// Variants timed side by side by fv-gpucheck `conv_routes`; rt/conv.rs
// `IgemmPlan` picks one (docs/perf/2026-11-25-conv-routes.md). Round 1's
// probes: with no global loads at all the 128x128x32 kernel reaches 0.90x
// cuDNN, so the k-tile overhead (one barrier and the fragment loads per 32
// mma) and warp latency are what is left; bk64 halves the barriers per FLOP,
// m256 doubles the mma per barrier at the same warps per SM.

// 128 x 128 x 32, 256 threads (2 x 4 warps of 64 x 32), 3 stages: 48 KB static, two CTAs per SM.
extern "C" __global__ void __launch_bounds__(256, 2) conv_igemm_bf16(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[3 * (128 + 128) * 64];
    ig_body<128, 128, 32, 64, 32, 3, 0>(IG_PASS, smem);
#endif
}

// 128 x 128 x 32, 128 threads (2 x 2 warps of 64 x 64), 3 stages: 48 KB static, two CTAs per SM.
extern "C" __global__ void __launch_bounds__(128, 2) conv_igemm_bf16_w64(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[3 * (128 + 128) * 64];
    ig_body<128, 128, 32, 64, 64, 3, 0>(IG_PASS, smem);
#endif
}

// WO-3b-ii: the warp-specialised mainloop (ig_ws_body), 160 threads, 48 KB
// static, two CTAs per SM; and its no-epilogue probe (MODE 3, wrong values).
extern "C" __global__ void __launch_bounds__(160, 2) conv_igemm_bf16_ws(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[3 * (128 + 128) * 64];
    ig_ws_body<3, 0>(IG_PASS, smem);
#endif
}
extern "C" __global__ void __launch_bounds__(160, 2) conv_igemm_bf16_ws_probe_noepi(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[3 * (128 + 128) * 64];
    ig_ws_body<3, 3>(IG_PASS, smem);
#endif
}

// Probes (fv-gpucheck `conv_routes` timing only; wrong values): the base kernel
// with a gather-free A load (MODE 1) and with no loads (MODE 2).
extern "C" __global__ void __launch_bounds__(256, 2) conv_igemm_bf16_probe_dense(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[3 * (128 + 128) * 64];
    ig_body<128, 128, 32, 64, 32, 3, 1>(IG_PASS, smem);
#endif
}
extern "C" __global__ void __launch_bounds__(256, 2) conv_igemm_bf16_probe_noload(IG_ARGS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[3 * (128 + 128) * 64];
    ig_body<128, 128, 32, 64, 32, 3, 2>(IG_PASS, smem);
#endif
}
