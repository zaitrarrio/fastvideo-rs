// ---- Convolutions without cuDNN (ADR-0024 step 3, WO-3a spike) -------------
//
// Two cuDNN-free routes for the dense N-D convolution, both on channels-last
// bf16 operands with f32 accumulation (rt/conv.rs `conv3d_igemm`,
// `conv3d_unfold_lt`):
//
//   R1 unfold-lt  conv_unfold_nhwc_bf16 writes im2col rows [m, K] (K = taps x
//                 C, tap-major, channel-minor) for a chunk of output pixels;
//                 cuBLASLt multiplies them by the [K, N] weight.
//   R2 igemm      conv_igemm_bf16 is the implicit GEMM: the im2col gather is
//                 the A-tile producer of an mma.sync.m16n8k16 bf16 GEMM, so no
//                 unfolded matrix is ever written. One tile config: 128 x 128
//                 x 32, 8 warps (2 along M x 4 along N, 64 x 32 per warp),
//                 cp.async double buffering, ldmatrix fragments (the helpers
//                 of core.cu's attention kernels).
//
// Both read the activation as NDHWC bf16 (conv_nchw_to_nhwc_bf16: one tiled
// transpose + cast, the counterpart of the per-call cast the cuDNN bf16 path
// already does) and write f32 NDHWC, transposed back by
// conv_nhwc_to_nchw_f32. Weights go through conv_weight_bf16 once per call
// in the spike (3b caches them per ConvKey). Constraints of the spike: C_in
// a multiple of 8 (a 16-byte cp.async chunk never straddles a tap), groups
// 1, dilation as a parameter. Padding is zero padding; out-of-range taps are
// zero-filled by cp.async (never read), so no input is ever padded in
// memory.

#define IG_BM 128
#define IG_BN 128
#define IG_BK 32
// Bytes per shared-memory row: IG_BK bf16 = 4 chunks of 16 B.
#define IG_ROWB 64
#define IG_TILEB (IG_BM * IG_ROWB)

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

// [n, s, c] f32 -> [n, c, s] f32, the transpose back. Same grid as above with
// the roles of c and s swapped: grid (ceil(c/32), ceil(s/32), n).
extern "C" __global__ void conv_nhwc_to_nchw_f32(
    const float* __restrict__ y, float* __restrict__ out, long c, long s
) {
    __shared__ float tile[32][33];
    const long b = blockIdx.z;
    const long c0 = (long)blockIdx.x * 32, s0 = (long)blockIdx.y * 32;
    const float* yb = y + b * s * c;
    for (int j = threadIdx.y; j < 32; j += 8) {
        long ss = s0 + j, cc = c0 + threadIdx.x;
        tile[j][threadIdx.x] = (ss < s && cc < c) ? yb[ss * c + cc] : 0.0f;
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

// y[m, n] (f32, NDHWC rows) = sum_k A[m, k] W[n, k], A gathered from the NDHWC
// bf16 activation on the fly. Grid (ceil(N / 128), ceil(M / 128)), 256 threads.
extern "C" __global__ void __launch_bounds__(256, 2) conv_igemm_bf16(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ y,
    int B, int D, int H, int W, int C, int OD, int OH, int OW, int N,
    int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[4 * IG_TILEB];
    const unsigned int sbase = mma_smem_u32(smem);
    // Stage s: A at 2s, B at 2s + 1.
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const long M = (long)B * OD * OH * OW;
    const long m0 = (long)blockIdx.y * IG_BM;
    const int n0 = blockIdx.x * IG_BN;
    const int taps = KD * KH * KW;
    const long K = (long)taps * C;
    const int nk = (int)((K + IG_BK - 1) / IG_BK);
    if (m0 >= M || n0 >= N) return;

    // Loader state: chunk = tid + i * 256, row = chunk >> 2, chunk column cc =
    // chunk & 3; the row's output pixel decoded once.
    int a_b[2], a_od[2], a_oh[2], a_ow[2], a_ok[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = (tid + i * 256) >> 2;
        long m = m0 + row;
        a_ok[i] = m < M;
        long mm = a_ok[i] ? m : 0;
        a_ow[i] = (int)(mm % OW); long q = mm / OW;
        a_oh[i] = (int)(q % OH); q /= OH;
        a_od[i] = (int)(q % OD);
        a_b[i] = (int)(q / OD);
    }

    float acc[4][4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;

    auto load_stage = [&](int kt, int stage) {
        const unsigned int sA = sbase + (2 * stage) * IG_TILEB, sB = sA + IG_TILEB;
        const long kbase = (long)kt * IG_BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int chunk = tid + i * 256;
            const int row = chunk >> 2, cc = chunk & 3;
            const long k = kbase + cc * 8;
            // A: 8 consecutive k share one tap (C % 8 == 0).
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
            // B: weight row n, k contiguous.
            const int n = n0 + row;
            const int bok = (n < N) && (k < K);
            mma_cp_async16_zfill(sB + ig_swz(row, cc), bok ? (const void*)(w + (long)n * K + k) : (const void*)w, bok);
        }
        mma_cp_commit();
    };

    load_stage(0, 0);
    for (int kt = 0; kt < nk; kt++) {
        const int stage = kt & 1;
        if (kt + 1 < nk) {
            load_stage(kt + 1, stage ^ 1);
            mma_cp_wait<1>();
        } else {
            mma_cp_wait<0>();
        }
        __syncthreads();
        const unsigned int sA = sbase + (2 * stage) * IG_TILEB, sB = sA + IG_TILEB;
        #pragma unroll
        for (int kc = 0; kc < IG_BK / 16; kc++) {
            unsigned int af[4][4];
            #pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                const int row = wm * 64 + mt * 16 + (lane & 15);
                const int col = kc * 16 + (lane >> 4) * 8;
                mma_ldm_x4(sA + ig_swz(row, col >> 3), af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
            }
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                unsigned int bfr[2];
                const int row = wn * 32 + nt * 8 + (lane & 7);
                const int col = kc * 16 + ((lane >> 3) & 1) * 8;
                mma_ldm_x2(sB + ig_swz(row, col >> 3), bfr[0], bfr[1]);
                #pragma unroll
                for (int mt = 0; mt < 4; mt++) mma_bf16(acc[mt][nt], af[mt], bfr);
            }
        }
        __syncthreads();
    }

    // C fragment: c0,c1 = (row g, cols 2t, 2t+1); c2,c3 = (row g + 8, same cols).
    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int n = n0 + wn * 32 + nt * 8 + 2 * t;
            #pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const long m = m0 + wm * 64 + mt * 16 + g + hh * 8;
                if (m < M) {
                    float* yr = y + m * N;
                    if (n < N) yr[n] = acc[mt][nt][hh * 2];
                    if (n + 1 < N) yr[n + 1] = acc[mt][nt][hh * 2 + 1];
                }
            }
        }
    }
#endif
}
