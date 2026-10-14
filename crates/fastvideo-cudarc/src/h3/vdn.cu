// VDN-H3 linear-branch kernels (crates/fastvideo-cudarc/src/h3/vdn.rs,
// docs/ports/vdn-h3.md §1.2 and §2). bf16 values travel as `unsigned short`
// bits; arithmetic is fp32 (double for long sums where cheap), rounded to
// bf16 once at each store where the reference's inference body rounds.
// Compiled with IEEE division / sqrt / exp (no fast math): the solve and the
// decay products are the numerically sensitive part of the branch.
//
// Shapes: C = heads * 128 channels, rows token-major ([rows, C]); per-head
// 128 x 128 matrices row-major, [chunks, heads, 128, 128].

#define VDN_D 128

__device__ __forceinline__ float vbf(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short vrn(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float vround(float f) { return vbf(vrn(f)); }
__device__ __forceinline__ float vsilu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float vsigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ void load4(const unsigned short* p, float* o) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    o[0] = vbf((unsigned short)(u.x & 0xffffu));
    o[1] = vbf((unsigned short)(u.x >> 16));
    o[2] = vbf((unsigned short)(u.y & 0xffffu));
    o[3] = vbf((unsigned short)(u.y >> 16));
}
__device__ __forceinline__ void store4(unsigned short* p, const float* v) {
    uint2 u;
    u.x = (unsigned int)vrn(v[0]) | ((unsigned int)vrn(v[1]) << 16);
    u.y = (unsigned int)vrn(v[2]) | ((unsigned int)vrn(v[3]) << 16);
    *reinterpret_cast<uint2*>(p) = u;
}

// SiLU (+ per-head L2 norm), `x / max(||x||, 1e-6)`, one warp per (row, head),
// four channels per lane. Input rows at `x + row * stride + off`.
__device__ __forceinline__ void vdn_act_store(float* y, int l2, unsigned short* dst) {
    #pragma unroll
    for (int i = 0; i < 4; i++) y[i] = vsilu(y[i]);
    if (l2) {
        float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = 1.0f / fmaxf(sqrtf(ss), 1e-6f);
        #pragma unroll
        for (int i = 0; i < 4; i++) y[i] *= inv;
    }
    store4(dst, y);
}

extern "C" __global__ void __launch_bounds__(256) vdn_act_rows(
    const unsigned short* __restrict__ x, long stride, long off, int rows, int heads, int l2,
    unsigned short* __restrict__ out
) {
    const int lane = threadIdx.x & 31;
    const long w = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (w >= (long)rows * heads) return;
    const long row = w / heads;
    const int h = (int)(w % heads);
    const int c = h * VDN_D + lane * 4;
    float y[4];
    load4(x + row * stride + off + c, y);
    vdn_act_store(y, l2, out + row * (long)heads * VDN_D + c);
}

// Depthwise 5x5 spatial conv per frame, zero padding 2, no bias: rows are
// [frames, gh, gw] tokens at `x + token * stride + off`, weights transposed
// to [25, C] (tap-major). fp32 accumulation, one bf16 rounding (cuDNN's).
extern "C" __global__ void __launch_bounds__(256) vdn_sconv5(
    const unsigned short* __restrict__ x, long stride, long off, int gh, int gw, int C,
    const unsigned short* __restrict__ wt, unsigned short* __restrict__ out
) {
    const int c = (blockIdx.x * 256 + threadIdx.x) * 2;
    if (c >= C) return;
    const long tok = blockIdx.y;
    const int per = gh * gw;
    const long f = tok / per;
    const int rem = (int)(tok % per);
    const int y = rem / gw, xx = rem % gw;
    float a0 = 0.f, a1 = 0.f;
    #pragma unroll
    for (int dy = 0; dy < 5; dy++) {
        const int yy = y + dy - 2;
        if (yy < 0 || yy >= gh) continue;
        #pragma unroll
        for (int dx = 0; dx < 5; dx++) {
            const int xs = xx + dx - 2;
            if (xs < 0 || xs >= gw) continue;
            const long src = (f * per + (long)yy * gw + xs) * stride + off + c;
            const unsigned int u = *reinterpret_cast<const unsigned int*>(x + src);
            const unsigned int wu = *reinterpret_cast<const unsigned int*>(wt + (long)(dy * 5 + dx) * C + c);
            a0 = fmaf(vbf((unsigned short)(wu & 0xffffu)), vbf((unsigned short)(u & 0xffffu)), a0);
            a1 = fmaf(vbf((unsigned short)(wu >> 16)), vbf((unsigned short)(u >> 16)), a1);
        }
    }
    *reinterpret_cast<unsigned int*>(out + tok * C + c) = (unsigned int)vrn(a0) | ((unsigned int)vrn(a1) << 16);
}

// Depthwise 5-tap temporal conv over frames (zero padding), then SiLU (+ L2):
// x [frames * S, C] contiguous, weights [5, C]. One warp per (token, head).
extern "C" __global__ void __launch_bounds__(256) vdn_tconv5_act(
    const unsigned short* __restrict__ x, int frames, int S, int heads,
    const unsigned short* __restrict__ wt, int l2, unsigned short* __restrict__ out
) {
    const int lane = threadIdx.x & 31;
    const long w = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const long rows = (long)frames * S;
    if (w >= rows * heads) return;
    const long row = w / heads;
    const int h = (int)(w % heads);
    const int f = (int)(row / S), s = (int)(row % S);
    const int C = heads * VDN_D;
    const int c = h * VDN_D + lane * 4;
    float y[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int dt = 0; dt < 5; dt++) {
        const int ff = f + dt - 2;
        if (ff < 0 || ff >= frames) continue;
        float xv[4], wv[4];
        load4(x + ((long)ff * S + s) * C + c, xv);
        load4(wt + (long)dt * C + c, wv);
        #pragma unroll
        for (int i = 0; i < 4; i++) y[i] = fmaf(wv[i], xv[i], y[i]);
    }
    vdn_act_store(y, l2, out + row * C + c);
}

// Per-chunk statistics, one CTA per (chunk, head): A = sum_s beta k k^T
// (fp32, raw: the solve symmetrizes) and B = sum_s bf16(v * beta) k^T, the
// latter rounded to bf16 as the reference's bf16 GEMM output. beta is
// bf16(sigmoid(aux[row, beta_off + h])). Thread (ti, tj) owns rows ti*8..,
// columns tj*8.. of both.
extern "C" __global__ void __launch_bounds__(256) vdn_frame_stats(
    const unsigned short* __restrict__ k, const unsigned short* __restrict__ v,
    const unsigned short* __restrict__ aux, long aux_stride, long beta_off,
    int S, int heads, float* __restrict__ A, float* __restrict__ B
) {
    __shared__ float kf[32][VDN_D];
    __shared__ float bk[32][VDN_D];
    __shared__ float vb[32][VDN_D];
    const int chunk = blockIdx.x / heads, h = blockIdx.x % heads;
    const int tid = threadIdx.x, ti = tid >> 4, tj = tid & 15;
    const int C = heads * VDN_D;
    float aa[8][8], ab[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) { aa[i][j] = 0.f; ab[i][j] = 0.f; }
    for (int s0 = 0; s0 < S; s0 += 32) {
        __syncthreads();
        for (int idx = tid; idx < 32 * VDN_D; idx += 256) {
            const int r = idx / VDN_D, cc = idx % VDN_D;
            float kv = 0.f, bkv = 0.f, vbv = 0.f;
            if (s0 + r < S) {
                const long row = (long)chunk * S + s0 + r;
                const float be = vround(vsigmoid(vbf(aux[row * aux_stride + beta_off + h])));
                kv = vbf(k[row * C + h * VDN_D + cc]);
                bkv = kv * be;
                vbv = vround(vbf(v[row * C + h * VDN_D + cc]) * be);
            }
            kf[r][cc] = kv;
            bk[r][cc] = bkv;
            vb[r][cc] = vbv;
        }
        __syncthreads();
        const int nr = min(32, S - s0);
        for (int r = 0; r < nr; r++) {
            float a[8], kc[8], bv[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a[i] = bk[r][ti * 8 + i];
                bv[i] = vb[r][ti * 8 + i];
                kc[i] = kf[r][tj * 8 + i];
            }
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    aa[i][j] = fmaf(a[i], kc[j], aa[i][j]);
                    ab[i][j] = fmaf(bv[i], kc[j], ab[i][j]);
                }
        }
    }
    const long base = (long)blockIdx.x * VDN_D * VDN_D;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const long e = base + (long)(ti * 8 + i) * VDN_D + tj * 8 + j;
            A[e] = aa[i][j];
            B[e] = vround(ab[i][j]);
        }
}

// Mean over each frame's S rows of x [frames * S, hidden] at `x + row0 * hidden`.
extern "C" __global__ void __launch_bounds__(256) vdn_frame_mean(
    const unsigned short* __restrict__ x, long row0, int S, int hidden, float* __restrict__ out
) {
    const int col = blockIdx.x * 256 + threadIdx.x;
    if (col >= hidden) return;
    const long f = blockIdx.y;
    double acc = 0.0;
    for (int s = 0; s < S; s++) acc += (double)vbf(x[(row0 + f * S + s) * hidden + col]);
    out[f * hidden + col] = (float)(acc / (double)S);
}

// z[f, r] = down[r, :] . m[f, :] (fp32 of the bf16 weights), one CTA per (r, f).
extern "C" __global__ void __launch_bounds__(256) vdn_alpha_down(
    const float* __restrict__ m, const unsigned short* __restrict__ down, int hidden, int R,
    float* __restrict__ z
) {
    __shared__ double part[256];
    const int r = blockIdx.x;
    const long f = blockIdx.y;
    double acc = 0.0;
    for (int i = threadIdx.x; i < hidden; i += 256)
        acc += (double)vbf(down[(long)r * hidden + i]) * (double)m[f * hidden + i];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) part[threadIdx.x] += part[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) z[f * R + r] = (float)part[0];
}

// alpha[f, c] = exp(-exp(A_log[c / d]) * softplus(up[c, :] . z[f, :] + dt_bias[c])).
extern "C" __global__ void __launch_bounds__(256) vdn_alpha_up(
    const float* __restrict__ z, const unsigned short* __restrict__ up,
    const unsigned short* __restrict__ dt, const unsigned short* __restrict__ alog,
    int R, int C, int frames, float* __restrict__ alpha
) {
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long)frames * C) return;
    const long f = idx / C;
    const int c = (int)(idx % C);
    float u = 0.f;
    for (int r = 0; r < R; r++) u = fmaf(vbf(up[(long)c * R + r]), z[f * R + r], u);
    u += vbf(dt[c]);
    const float sp = u > 20.f ? u : log1pf(expf(u));
    alpha[idx] = expf(-expf(vbf(alog[c / VDN_D])) * sp);
}

// vdn_solve's factors, one CTA per matrix: M = I + (A + A^T) / 2 inverted in
// shared memory by Gauss-Jordan elimination without pivoting (I + A is SPD),
// then T = Diag(alpha) M^-1 (rows scaled; skipped when !write_t) and
// J = jscale * B M^-1. Dynamic shared memory: (128 * 128 + 256) floats.
extern "C" __global__ void __launch_bounds__(256) vdn_solve(
    const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ alpha,
    int has_alpha, float jscale, int write_t, float* __restrict__ T, float* __restrict__ J
) {
    extern __shared__ float sm[];
    float* M = sm;
    float* rowk = sm + VDN_D * VDN_D;
    float* colk = rowk + VDN_D;
    const long base = (long)blockIdx.x * VDN_D * VDN_D;
    const int tid = threadIdx.x;
    for (int idx = tid; idx < VDN_D * VDN_D; idx += 256) {
        const int i = idx / VDN_D, j = idx % VDN_D;
        M[idx] = 0.5f * (A[base + idx] + A[base + j * VDN_D + i]) + (i == j ? 1.f : 0.f);
    }
    for (int kk = 0; kk < VDN_D; kk++) {
        __syncthreads();
        if (tid < VDN_D) rowk[tid] = M[kk * VDN_D + tid];
        else colk[tid - VDN_D] = M[(tid - VDN_D) * VDN_D + kk];
        __syncthreads();
        const float ip = 1.0f / rowk[kk];
        for (int idx = tid; idx < VDN_D * VDN_D; idx += 256) {
            const int i = idx / VDN_D, j = idx % VDN_D;
            if (i == kk) M[idx] = (j == kk) ? ip : rowk[j] * ip;
            else if (j == kk) M[idx] = -colk[i] * ip;
            else M[idx] = fmaf(-colk[i] * ip, rowk[j], M[idx]);
        }
    }
    __syncthreads();
    if (write_t) {
        for (int idx = tid; idx < VDN_D * VDN_D; idx += 256) {
            const int i = idx / VDN_D;
            const float al = has_alpha ? alpha[(long)blockIdx.x * VDN_D + i] : 1.f;
            T[base + idx] = al * M[idx];
        }
    }
    const int tv = tid >> 4, tj = tid & 15;
    float acc[8][8];
    #pragma unroll
    for (int a = 0; a < 8; a++)
        #pragma unroll
        for (int b = 0; b < 8; b++) acc[a][b] = 0.f;
    for (int i = 0; i < VDN_D; i++) {
        float bv[8], mv[8];
        #pragma unroll
        for (int a = 0; a < 8; a++) {
            bv[a] = B[base + (long)(tv * 8 + a) * VDN_D + i];
            mv[a] = M[i * VDN_D + tj * 8 + a];
        }
        #pragma unroll
        for (int a = 0; a < 8; a++)
            #pragma unroll
            for (int b = 0; b < 8; b++) acc[a][b] = fmaf(bv[a], mv[b], acc[a][b]);
    }
    #pragma unroll
    for (int a = 0; a < 8; a++)
        #pragma unroll
        for (int b = 0; b < 8; b++)
            J[base + (long)(tv * 8 + a) * VDN_D + tj * 8 + b] = jscale * acc[a][b];
}

// The two scans, one CTA per (head, 32-row block, direction):
// state <- state T_f + J_f from `start`, forward into P, reverse into Q.
// Dynamic shared memory: T_f (128 x 128) + the 32 state rows, floats.
extern "C" __global__ void __launch_bounds__(256) vdn_scan(
    const float* __restrict__ T, const float* __restrict__ J, const float* __restrict__ start,
    int frames, int heads, float* __restrict__ P, float* __restrict__ Q
) {
    extern __shared__ float sm[];
    float* Ts = sm;
    float* St = sm + VDN_D * VDN_D;
    const int h = blockIdx.x, rb = blockIdx.y, dir = blockIdx.z;
    const int tid = threadIdx.x, r = tid >> 3, c0 = (tid & 7) * 16;
    const int row = rb * 32 + r;
    float* bank = dir == 0 ? P : Q;
    for (int c = 0; c < 16; c++) St[r * VDN_D + c0 + c] = start[((long)h * VDN_D + row) * VDN_D + c0 + c];
    for (int step = 0; step < frames; step++) {
        const int f = dir == 0 ? step : frames - 1 - step;
        const long mat = ((long)f * heads + h) * VDN_D * VDN_D;
        __syncthreads();
        for (int idx = tid * 4; idx < VDN_D * VDN_D; idx += 1024)
            *reinterpret_cast<float4*>(Ts + idx) = *reinterpret_cast<const float4*>(T + mat + idx);
        __syncthreads();
        float acc[16];
        #pragma unroll
        for (int c = 0; c < 16; c++) acc[c] = J[mat + (long)row * VDN_D + c0 + c];
        for (int i = 0; i < VDN_D; i++) {
            const float sv = St[r * VDN_D + i];
            #pragma unroll
            for (int c = 0; c < 16; c++) acc[c] = fmaf(sv, Ts[i * VDN_D + c0 + c], acc[c]);
        }
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            St[r * VDN_D + c0 + c] = acc[c];
            bank[mat + (long)row * VDN_D + c0 + c] = acc[c];
        }
    }
}

// lp[f + 1, c] = lp[f, c] + log(max(alpha[f, c], 1e-12)), lp[0] = 0 (double).
extern "C" __global__ void __launch_bounds__(256) vdn_logprefix(
    const float* __restrict__ alpha, int frames, int C, double* __restrict__ lp
) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    double acc = 0.0;
    lp[c] = 0.0;
    for (int f = 0; f < frames; f++) {
        acc += log(fmax((double)alpha[(long)f * C + c], 1e-12));
        lp[(long)(f + 1) * C + c] = acc;
    }
}

// The boundary gather with the alpha bridge: per inner frame t (plan row
// [before, after, before_from, after_to], -1 = the start state), the state
// just outside the window, decayed column-wise into t, rounded to bf16.
extern "C" __global__ void __launch_bounds__(256) vdn_gather(
    const float* __restrict__ P, const float* __restrict__ Q, const float* __restrict__ start,
    const double* __restrict__ lp, const int* __restrict__ plan, int frames, int heads,
    unsigned short* __restrict__ out
) {
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    const long per_t = (long)heads * VDN_D * (VDN_D / 2);
    if (idx >= (long)frames * per_t) return;
    const int t = (int)(idx / per_t);
    const long rem = idx % per_t;
    const int h = (int)(rem / (VDN_D * (VDN_D / 2)));
    const int v = (int)((rem / (VDN_D / 2)) % VDN_D);
    const int kk = (int)(rem % (VDN_D / 2)) * 2;
    const int C = heads * VDN_D;
    const int b = plan[t * 4], a = plan[t * 4 + 1], from = plan[t * 4 + 2], to = plan[t * 4 + 3];
    const long e = ((long)h * VDN_D + v) * VDN_D + kk;
    const float* bs = b >= 0 ? P + (long)b * heads * VDN_D * VDN_D + e : start + e;
    const float* as = a >= 0 ? Q + (long)a * heads * VDN_D * VDN_D + e : start + e;
    float o[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int ch = h * VDN_D + kk + i;
        const float db = (float)exp(lp[(long)(t + 1) * C + ch] - lp[(long)from * C + ch]);
        const float da = (float)exp(lp[(long)(to + 1) * C + ch] - lp[(long)t * C + ch]);
        o[i] = fmaf(bs[i], db, as[i] * da);
    }
    *reinterpret_cast<unsigned int*>(out + (long)t * heads * VDN_D * VDN_D + e) =
        (unsigned int)vrn(o[0]) | ((unsigned int)vrn(o[1]) << 16);
}

// dst[r, 0..w) = src[r * stride + off + (0..w)].
extern "C" __global__ void __launch_bounds__(256) vdn_copy_cols(
    const unsigned short* __restrict__ src, long stride, long off, long rows, int w,
    unsigned short* __restrict__ dst
) {
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= rows * w) return;
    const long r = idx / w;
    dst[idx] = src[r * stride + off + idx % w];
}

// Readout, RMSNorm and output gate, one CTA per (64 tokens, head, frame):
// r = bf16(state q) (the reference's bf16 GEMM), then
// bf16(r * rsqrt(mean(r^2) + eps) * w * bf16(sigmoid(og))).
// Dynamic shared memory: state^T (bf16 128 x 128) + q tile (f32 64 x 128).
extern "C" __global__ void __launch_bounds__(256) vdn_readout(
    const unsigned short* __restrict__ state, const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ og, const unsigned short* __restrict__ normw,
    int S, int heads, float eps, unsigned short* __restrict__ out
) {
    extern __shared__ __align__(16) unsigned char vdn_smem[];
    unsigned short* St = reinterpret_cast<unsigned short*>(vdn_smem);   // [k][v]
    float* Qs = reinterpret_cast<float*>(vdn_smem + VDN_D * VDN_D * 2); // [tok][k]
    const int s0 = blockIdx.x * 64, h = blockIdx.y;
    const long f = blockIdx.z;
    const int tid = threadIdx.x;
    const int C = heads * VDN_D;
    const unsigned short* sh = state + (f * heads + h) * (long)VDN_D * VDN_D;
    for (int idx = tid; idx < VDN_D * VDN_D; idx += 256) {
        const int v = idx / VDN_D, k = idx % VDN_D;
        St[k * VDN_D + v] = sh[idx];
    }
    for (int idx = tid; idx < 64 * VDN_D; idx += 256) {
        const int tk = idx / VDN_D, k = idx % VDN_D;
        Qs[idx] = s0 + tk < S ? vbf(q[(f * S + s0 + tk) * C + h * VDN_D + k]) : 0.f;
    }
    __syncthreads();
    const int tg = tid >> 4, vg = tid & 15;
    float acc[4][8];
    #pragma unroll
    for (int a = 0; a < 4; a++)
        #pragma unroll
        for (int b = 0; b < 8; b++) acc[a][b] = 0.f;
    for (int k = 0; k < VDN_D; k++) {
        float qv[4], sv[8];
        #pragma unroll
        for (int a = 0; a < 4; a++) qv[a] = Qs[(tg * 4 + a) * VDN_D + k];
        #pragma unroll
        for (int b = 0; b < 8; b++) sv[b] = vbf(St[k * VDN_D + vg * 8 + b]);
        #pragma unroll
        for (int a = 0; a < 4; a++)
            #pragma unroll
            for (int b = 0; b < 8; b++) acc[a][b] = fmaf(qv[a], sv[b], acc[a][b]);
    }
    #pragma unroll
    for (int a = 0; a < 4; a++) {
        float ss = 0.f;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            acc[a][b] = vround(acc[a][b]);
            ss = fmaf(acc[a][b], acc[a][b], ss);
        }
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rs = 1.0f / sqrtf(ss / (float)VDN_D + eps);
        const int tok = s0 + tg * 4 + a;
        if (tok >= S) continue;
        const long base = (f * S + tok) * C + h * VDN_D + vg * 8;
        #pragma unroll
        for (int b = 0; b < 8; b += 4) {
            float ov[4], gv[4];
            load4(og + base + b, gv);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const float g = vround(vsigmoid(gv[i]));
                ov[i] = acc[a][b + i] * rs * vbf(normw[vg * 8 + b + i]) * g;
            }
            store4(out + base + b, ov);
        }
    }
}

// The softmax branch's per-(row, head) gate on the attention output [H, S, D]
// (bf16, in place): o = bf16(o * bf16(sigmoid(aux[s, off + h]))).
extern "C" __global__ void __launch_bounds__(256) vdn_head_gate(
    unsigned short* __restrict__ o, const unsigned short* __restrict__ aux, long aux_stride,
    long off, int S, int heads
) {
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    const long n = (long)heads * S * (VDN_D / 4);
    if (idx >= n) return;
    const long hs = idx / (VDN_D / 4);
    const int h = (int)(hs / S);
    const long s = hs % S;
    const float g = vround(vsigmoid(vbf(aux[s * aux_stride + off + h])));
    float v[4];
    unsigned short* p = o + hs * VDN_D + (idx % (VDN_D / 4)) * 4;
    load4(p, v);
    #pragma unroll
    for (int i = 0; i < 4; i++) v[i] *= g;
    store4(p, v);
}

// out[off + i] = bf16(out[off + i] + add[i]).
extern "C" __global__ void __launch_bounds__(256) vdn_add_rows(
    unsigned short* __restrict__ out, long off, const unsigned short* __restrict__ add, long n
) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    out[off + i] = vrn(vbf(out[off + i]) + vbf(add[i]));
}
