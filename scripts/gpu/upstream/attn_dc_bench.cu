// Standalone parity + timing harness for the datacenter attention kernels
// (crates/fastvideo-cudarc/src/wan/attn_dc.cu) against the mma.sync kernel
// they replace (flash_mma_fwd2_d128 in kernels.cu). No Rust, no weights:
// pod.sh step `bench:attn_dc` builds it with the pod's nvcc for the GPU's
// arch-specific target (sm_100a / sm_90a) and runs it.
//
//   attn_dc_bench parity           every kernel for this SM vs flash_mma_fwd2
//                                  (and an f64 host reference on small shapes)
//   attn_dc_bench bench [shapes]   timings at the model shapes
//
// Each kernel runs in its own forked process with a wall-clock cap, so a
// trap or a hang in one variant is reported and the others still run.
// Output: one JSON object per line on stdout.
#include <cuda.h>
#include <cuda_runtime.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../../crates/fastvideo-cudarc/src/wan/kernels.cu"
#include "../../../crates/fastvideo-cudarc/src/wan/attn_dc.cu"

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            printf("{\"fatal\": \"%s:%d %s\"}\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            fflush(stdout);                                                                     \
            exit(3);                                                                            \
        }                                                                                       \
    } while (0)

typedef void (*DcKernel)(DcTensorMap, DcTensorMap, DcTensorMap, float*, unsigned short*, int, int, int, float);

struct Variant {
    const char* name;
    int sm;          // compute capability major*10+minor family it runs on (100 or 90)
    DcKernel fn;
    int threads;
    int smem;
    int rows;        // queries per CTA
};

static const Variant VARIANTS[] = {
    {"dc100", 100, fa_dc100_fwd_d128, 384, DC100_SMEM(4), 256},
    {"dc100_nsplit", 100, fa_dc100_fwd_d128_nsplit, 384, DC100_SMEM(4), 256},
    {"dc90", 90, fa_dc90_fwd_d128, 384, DC90_SMEM(4), 128},
    {"dc90_nsplit", 90, fa_dc90_fwd_d128_nsplit, 384, DC90_SMEM(4), 128},
};

static unsigned short f2bf(float x) {
    unsigned int u;
    memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return (unsigned short)((u >> 16) | ((u & 0x7FFFFFu) ? 0x40u : 0u));
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
static float bf2f(unsigned short b) {
    unsigned int u = (unsigned int)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;
static float frand() {  // N(0,1)-ish: sum of uniforms
    float s = 0.f;
    for (int i = 0; i < 4; i++) {
        rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
        s += (float)((rng_state >> 40) & 0xFFFFFF) / 16777216.0f - 0.5f;
    }
    return s * 1.7320508f;
}

// cuTensorMapEncodeTiled through the runtime's driver entry point (no -lcuda).
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encode_fn() {
    static EncodeTiledFn fn = nullptr;
    if (!fn) {
        cudaDriverEntryPointQueryResult q;
        CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&fn, 12000, cudaEnableDefault, &q));
        if (!fn || q != cudaDriverEntryPointSuccess) {
            printf("{\"fatal\": \"no cuTensorMapEncodeTiled\"}\n");
            exit(3);
        }
    }
    return fn;
}

static DcTensorMap encode(const void* ptr, int s, int bh) {
    CUtensorMap m;
    cuuint64_t dims[3] = {128, (cuuint64_t)s, (cuuint64_t)bh};
    cuuint64_t strides[2] = {256, (cuuint64_t)s * 256};
    cuuint32_t box[3] = {64, 128, 1};
    cuuint32_t es[3] = {1, 1, 1};
    CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)ptr, dims, strides, box, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        printf("{\"fatal\": \"cuTensorMapEncodeTiled %d\"}\n", (int)r);
        fflush(stdout);
        exit(3);
    }
    DcTensorMap d;
    memcpy(&d, &m, sizeof(m));
    return d;
}

struct Problem {
    int bh, sq, sk;
    unsigned short *q, *k, *v;
    std::vector<unsigned short> hq, hk, hv;
};

static Problem make_problem(int bh, int sq, int sk, float qs, float ks, bool host) {
    Problem p;
    p.bh = bh;
    p.sq = sq;
    p.sk = sk;
    const size_t nq = (size_t)bh * sq * 128, nk = (size_t)bh * sk * 128;
    std::vector<unsigned short> a(nq), b(nk), c(nk);
    for (auto& x : a) x = f2bf(frand() * qs);
    for (auto& x : b) x = f2bf(frand() * ks);
    for (auto& x : c) x = f2bf(frand());
    CK(cudaMalloc(&p.q, nq * 2));
    CK(cudaMalloc(&p.k, nk * 2));
    CK(cudaMalloc(&p.v, nk * 2));
    CK(cudaMemcpy(p.q, a.data(), nq * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(p.k, b.data(), nk * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(p.v, c.data(), nk * 2, cudaMemcpyHostToDevice));
    if (host) {
        p.hq.swap(a);
        p.hk.swap(b);
        p.hv.swap(c);
    }
    return p;
}
static void free_problem(Problem& p) {
    cudaFree(p.q);
    cudaFree(p.k);
    cudaFree(p.v);
}

// flash_mma_fwd2_d128: the kernel the new ones replace (oracle).
static void run_v2(const Problem& p, float* out, unsigned short* out16, int is16, float sl2) {
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(flash_mma_fwd2_d128, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 64 * 128 * 2));
        init = true;
    }
    dim3 grid((p.sq + 127) / 128, p.bh);
    flash_mma_fwd2_d128<<<grid, 256, 4 * 64 * 128 * 2>>>(p.q, p.k, p.v, out, out16, is16, p.sq, p.sk, sl2);
}

static void run_dc(const Variant& vt, const Problem& p, float* out, unsigned short* out16, int is16, float sl2) {
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(vt.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, vt.smem));
        init = true;
    }
    DcTensorMap tq = encode(p.q, p.sq, p.bh), tk = encode(p.k, p.sk, p.bh), tv = encode(p.v, p.sk, p.bh);
    dim3 grid((p.sq + vt.rows - 1) / vt.rows, p.bh);
    vt.fn<<<grid, vt.threads, vt.smem>>>(tq, tk, tv, out, out16, is16, p.sq, p.sk, sl2);
}

static void host_ref(const Problem& p, float scale, std::vector<double>& out) {
    out.assign((size_t)p.bh * p.sq * 128, 0.0);
    std::vector<double> s(p.sk);
    for (int h = 0; h < p.bh; h++)
        for (int i = 0; i < p.sq; i++) {
            const unsigned short* qi = &p.hq[((size_t)h * p.sq + i) * 128];
            double mx = -1e300;
            for (int j = 0; j < p.sk; j++) {
                const unsigned short* kj = &p.hk[((size_t)h * p.sk + j) * 128];
                double acc = 0;
                for (int d = 0; d < 128; d++) acc += (double)bf2f(qi[d]) * bf2f(kj[d]);
                s[j] = acc * scale;
                mx = std::max(mx, s[j]);
            }
            double l = 0;
            for (int j = 0; j < p.sk; j++) {
                s[j] = std::exp(s[j] - mx);
                l += s[j];
            }
            double* o = &out[((size_t)h * p.sq + i) * 128];
            for (int j = 0; j < p.sk; j++) {
                const unsigned short* vj = &p.hv[((size_t)h * p.sk + j) * 128];
                for (int d = 0; d < 128; d++) o[d] += s[j] * bf2f(vj[d]);
            }
            for (int d = 0; d < 128; d++) o[d] /= l;
        }
}

struct Diff {
    double rel_l2, max_abs;
};
template <class A, class B>
static Diff diff(const A* a, const B* b, size_t n) {
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = (double)a[i], y = (double)b[i];
        const double d = x - y;
        if (!(std::fabs(d) <= 1e30)) return {INFINITY, INFINITY};  // NaN / inf
        num += d * d;
        den += y * y;
        mx = std::max(mx, std::fabs(d));
    }
    return {den > 0 ? std::sqrt(num / den) : std::sqrt(num), mx};
}

static int device_sm() {
    int dev = 0, major = 0, minor = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    return major * 10 + minor;
}

// ---- component probes -------------------------------------------------------
// One 128 x 128 tile through each MMA path of the kernels, against host math:
// S = Q K^T and O = P V (P given as bf16). A failing kernel whose probes pass
// points at the pipeline; a failing probe at a descriptor or a layout.
#define PROBE_SMEM (1024 + 3 * DC_TILEB + 64)

__global__ void __launch_bounds__(128, 1) probe_dc100(const __grid_constant__ DcTensorMap tq,
                                                      const __grid_constant__ DcTensorMap tk,
                                                      const __grid_constant__ DcTensorMap tv,
                                                      const unsigned short* __restrict__ P, float* __restrict__ S_out,
                                                      float* __restrict__ O_out, int split) {
#if defined(DC_SM100)
    extern __shared__ __align__(1024) unsigned char dc_smem[];
    const unsigned int raw = dc_smem_u32(dc_smem), base = (raw + 1023u) & ~1023u;
    const unsigned int sQ = base, sK = base + DC_TILEB, sV = base + 2 * DC_TILEB, bars = base + 3 * DC_TILEB;
    const unsigned int b_ld = bars, b_mma = bars + 8, tslot = bars + 16;
    const int tid = threadIdx.x, warp = tid >> 5;
    if (tid == 0) {
        dc_mbar_init(b_ld, 1);
        dc_mbar_init(b_mma, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == 0) {
        const unsigned int ncols = 256;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(tslot), "r"(ncols) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
    dc_tc_fence_before();
    __syncthreads();
    dc_tc_fence_after();
    const unsigned int tbase = *reinterpret_cast<volatile unsigned int*>(dc_smem + (tslot - raw));
    const unsigned int tl = tbase + ((unsigned int)(warp * 32) << 16);
    if (tid == 0) {
        dc_mbar_expect(b_ld, 3 * DC_TILEB);
        dc_tma_tile(sQ, &tq, b_ld, 0, 0);
        dc_tma_tile(sK, &tk, b_ld, 0, 0);
        dc_tma_tile(sV, &tv, b_ld, 0, 0);
        dc_mbar_wait(b_ld, 0);
        dc100_qk(tbase, sQ, sK);
        dc_umma_commit(b_mma);
    }
    __syncwarp();
    dc_mbar_wait(b_mma, 0);
    dc_tc_fence_after();
    for (int q = 0; q < 4; q++) {
        unsigned int r[32];
        dc_tmem_ld32(tl + 32u * q, r);
        for (int c = 0; c < 32; c++) S_out[tid * 128 + 32 * q + c] = __uint_as_float(r[c]);
    }
    for (int h = 0; h < 2; h++) {
        unsigned int w[32];
        for (int i = 0; i < 32; i++)
            w[i] = (unsigned int)P[tid * 128 + 64 * h + 2 * i] | ((unsigned int)P[tid * 128 + 64 * h + 2 * i + 1] << 16);
        dc_tmem_st32(tl + 64u + 32u * h, w);
    }
    dc_tc_fence_before();
    __syncthreads();
    if (tid == 0) {
        dc_tc_fence_after();
        if (split) dc100_pv<true>(tbase + 128u, tbase + 64u, sV, false);
        else dc100_pv<false>(tbase + 128u, tbase + 64u, sV, false);
        dc_umma_commit(b_mma);
    }
    __syncwarp();
    dc_mbar_wait(b_mma, 1);
    dc_tc_fence_after();
    for (int q = 0; q < 4; q++) {
        unsigned int r[32];
        dc_tmem_ld32(tl + 128u + 32u * q, r);
        for (int c = 0; c < 32; c++) O_out[tid * 128 + 32 * q + c] = __uint_as_float(r[c]);
    }
    dc_tc_fence_before();
    __syncthreads();
    if (warp == 0) {
        dc_tc_fence_after();
        const unsigned int ncols = 256;
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tbase), "r"(ncols) : "memory");
    }
#endif
}

__global__ void __launch_bounds__(128, 1) probe_dc90(const __grid_constant__ DcTensorMap tq,
                                                     const __grid_constant__ DcTensorMap tk,
                                                     const __grid_constant__ DcTensorMap tv,
                                                     const unsigned short* __restrict__ P, float* __restrict__ S_out,
                                                     float* __restrict__ O_out, int split) {
#if defined(DC_SM90)
    extern __shared__ __align__(1024) unsigned char dc_smem[];
    const unsigned int raw = dc_smem_u32(dc_smem), base = (raw + 1023u) & ~1023u;
    const unsigned int sQ = base, sK = base + DC_TILEB, sV = base + 2 * DC_TILEB, b_ld = base + 3 * DC_TILEB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    if (tid == 0) {
        dc_mbar_init(b_ld, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) {
        dc_mbar_expect(b_ld, 3 * DC_TILEB);
        dc_tma_tile(sQ, &tq, b_ld, 0, 0);
        dc_tma_tile(sK, &tk, b_ld, 0, 0);
        dc_tma_tile(sV, &tv, b_ld, 0, 0);
    }
    dc_mbar_wait(b_ld, 0);
    float S[64];
    for (int i = 0; i < 64; i++) S[i] = 0.f;
    dc_wgmma_fence();
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        const unsigned int off = (unsigned int)(kk >> 2) * DC_PANELB + (unsigned int)(kk & 3) * 32u;
        dc_wgmma_ss(S, dc_desc90(sQ + off, 16, 1024), dc_desc90(sK + off, 16, 1024), kk > 0);
    }
    dc_wgmma_commit();
    dc_wgmma_wait<0>();
    dc_fence_regs(S);
    const int row0 = 16 * warp + g, row1 = row0 + 8;
    for (int n = 0; n < 16; n++) {
        const int col = 8 * n + 2 * t;
        S_out[row0 * 128 + col] = S[4 * n];
        S_out[row0 * 128 + col + 1] = S[4 * n + 1];
        S_out[row1 * 128 + col] = S[4 * n + 2];
        S_out[row1 * 128 + col + 1] = S[4 * n + 3];
    }
    unsigned int Pf[8][4];
    auto pp = [&](int r, int c) { return (unsigned int)P[r * 128 + c] | ((unsigned int)P[r * 128 + c + 1] << 16); };
    for (int kk = 0; kk < 8; kk++) {
        Pf[kk][0] = pp(row0, 16 * kk + 2 * t);
        Pf[kk][1] = pp(row1, 16 * kk + 2 * t);
        Pf[kk][2] = pp(row0, 16 * kk + 8 + 2 * t);
        Pf[kk][3] = pp(row1, 16 * kk + 8 + 2 * t);
    }
    float O[64];
    for (int i = 0; i < 64; i++) O[i] = 0.f;
    dc_wgmma_fence();
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        if (split) {
            dc_wgmma_rs64<0>(O, Pf[kk], dc_desc90(sV + 2048u * kk, DC_PANELB, 1024), 1);
            dc_wgmma_rs64<1>(O, Pf[kk], dc_desc90(sV + DC_PANELB + 2048u * kk, DC_PANELB, 1024), 1);
        } else {
            dc_wgmma_rs(O, Pf[kk], dc_desc90(sV + 2048u * kk, DC_PANELB, 1024), 1);
        }
    }
    dc_wgmma_commit();
    dc_wgmma_wait<0>();
    dc_fence_regs(O);
    for (int n = 0; n < 16; n++) {
        const int col = 8 * n + 2 * t;
        O_out[row0 * 128 + col] = O[4 * n];
        O_out[row0 * 128 + col + 1] = O[4 * n + 1];
        O_out[row1 * 128 + col] = O[4 * n + 2];
        O_out[row1 * 128 + col + 1] = O[4 * n + 3];
    }
#endif
}

static int probe_child(const char*, const char*) {
    CK(cudaFree(0));
    const int sm = device_sm();
    const bool b = sm / 10 == 10;
    if (!b && sm != 90) return 0;
    rng_state = 12345;
    Problem p = make_problem(1, 128, 128, 1.f, 1.f, true);
    std::vector<unsigned short> hp(128 * 128);
    for (auto& x : hp) x = f2bf(std::fabs(frand()) * 0.5f);
    unsigned short* dp;
    float *ds, *dout;
    CK(cudaMalloc(&dp, hp.size() * 2));
    CK(cudaMalloc(&ds, 128 * 128 * 4));
    CK(cudaMalloc(&dout, 128 * 128 * 4));
    CK(cudaMemcpy(dp, hp.data(), hp.size() * 2, cudaMemcpyHostToDevice));
    std::vector<double> sref(128 * 128), oref(128 * 128);
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 128; j++) {
            double a = 0, o = 0;
            for (int d = 0; d < 128; d++) a += (double)bf2f(p.hq[i * 128 + d]) * bf2f(p.hk[j * 128 + d]);
            for (int k = 0; k < 128; k++) o += (double)bf2f(hp[i * 128 + k]) * bf2f(p.hv[k * 128 + j]);
            sref[i * 128 + j] = a;
            oref[i * 128 + j] = o;
        }
    DcTensorMap tq = encode(p.q, 128, 1), tk = encode(p.k, 128, 1), tv = encode(p.v, 128, 1);
    auto fn = b ? probe_dc100 : probe_dc90;
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, PROBE_SMEM));
    const int rows = b ? 128 : 64;
    int fails = 0;
    for (int split = 0; split < 2; split++) {
        CK(cudaMemset(ds, 0xFF, 128 * 128 * 4));
        CK(cudaMemset(dout, 0xFF, 128 * 128 * 4));
        fn<<<1, 128, PROBE_SMEM>>>(tq, tk, tv, dp, ds, dout, split);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("{\"probe\": \"%s\", \"split\": %d, \"error\": \"%s\"}\n", b ? "dc100" : "dc90", split, cudaGetErrorString(e));
            return 2;
        }
        std::vector<float> hs(128 * 128), ho(128 * 128);
        CK(cudaMemcpy(hs.data(), ds, hs.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ho.data(), dout, ho.size() * 4, cudaMemcpyDeviceToHost));
        const Diff dS = diff(hs.data(), sref.data(), (size_t)rows * 128);
        const Diff dO = diff(ho.data(), oref.data(), (size_t)rows * 128);
        const bool ok = dS.rel_l2 < 1e-5 && dO.rel_l2 < 1e-5;
        fails += !ok;
        // A few raw values help read a layout mix-up.
        printf("{\"probe\": \"%s\", \"split\": %d, \"pass\": %s, \"qk_rel_l2\": %.3e, \"pv_rel_l2\": %.3e, "
               "\"S00\": [%g, %g, %g], \"S00_ref\": [%g, %g, %g], \"O00\": [%g, %g, %g], \"O00_ref\": [%g, %g, %g]}\n",
               b ? "dc100" : "dc90", split, ok ? "true" : "false", dS.rel_l2, dO.rel_l2, hs[0], hs[1], hs[129],
               sref[0], sref[1], sref[129], ho[0], ho[1], ho[129], oref[0], oref[1], oref[129]);
        fflush(stdout);
    }
    return fails ? 1 : 0;
}

// ---- parity ---------------------------------------------------------------
static int parity_child(const Variant& vt) {
    CK(cudaFree(0));
    const int sm = device_sm();
    if (sm / 10 != vt.sm / 10) {
        printf("{\"variant\": \"%s\", \"skipped\": \"sm %d\"}\n", vt.name, sm);
        return 0;
    }
    struct C {
        int bh, sq, sk;
        float qs, ks;
    };
    const C cases[] = {
        {1, 1, 1, 1.f, 1.f},       {2, 64, 64, 1.f, 1.f},      {2, 70, 70, 1.5f, 1.5f},
        {6, 257, 257, 1.5f, 1.5f}, {4, 300, 512, 1.5f, 1.5f},  {4, 1000, 333, 1.5f, 1.5f},
        {3, 130, 1, 1.f, 1.f},     {2, 63, 4097, 1.5f, 1.5f},  {2, 129, 200, 1.f, 1.f},
        {2, 2048, 2048, 1.5f, 1.5f}, {2, 4096, 4096, 3.f, 3.f}, {1, 513, 1000, 4.f, 4.f},
    };
    int fails = 0;
    for (const C& c : cases) {
        const bool host = (double)c.bh * c.sq * c.sk <= 3.0e6;
        Problem p = make_problem(c.bh, c.sq, c.sk, c.qs, c.ks, host);
        const size_t n = (size_t)c.bh * c.sq * 128;
        const float scale = 1.f / sqrtf(128.f), sl2 = scale * 1.4426950408889634f;
        float *o1, *o2;
        unsigned short *b2, *dummy16;
        float* dummy32;
        CK(cudaMalloc(&o1, n * 4));
        CK(cudaMalloc(&o2, n * 4));
        CK(cudaMalloc(&b2, n * 2));
        CK(cudaMalloc(&dummy16, 16));
        CK(cudaMalloc(&dummy32, 16));
        CK(cudaMemset(o2, 0xFF, n * 4));  // NaN: any row the kernel skips shows up
        CK(cudaMemset(b2, 0xFF, n * 2));
        run_v2(p, o1, dummy16, 0, sl2);
        run_dc(vt, p, o2, dummy16, 0, sl2);
        run_dc(vt, p, dummy32, b2, 1, sl2);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("{\"variant\": \"%s\", \"case\": \"%dx%dx%d\", \"error\": \"%s\"}\n", vt.name, c.bh, c.sq, c.sk,
                   cudaGetErrorString(e));
            fflush(stdout);
            return 2;  // sticky: the context is gone
        }
        std::vector<float> a(n), b(n);
        std::vector<unsigned short> b16(n);
        CK(cudaMemcpy(a.data(), o1, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), o2, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b16.data(), b2, n * 2, cudaMemcpyDeviceToHost));
        const Diff dv = diff(b.data(), a.data(), n);
        size_t mism = 0;
        for (size_t i = 0; i < n; i++) mism += f2bf(b[i]) != b16[i];
        double ref_new = -1, ref_old = -1;
        if (host) {
            std::vector<double> r;
            host_ref(p, scale, r);
            ref_new = diff(b.data(), r.data(), n).rel_l2;
            ref_old = diff(a.data(), r.data(), n).rel_l2;
        }
        const bool pass = dv.rel_l2 <= 4e-3 && mism == 0 && (!host || (ref_new <= 1e-2 && ref_new <= 2.0 * ref_old + 1e-4));
        fails += !pass;
        printf("{\"variant\": \"%s\", \"case\": \"%dx%dx%d\", \"pass\": %s, \"rel_l2_vs_v2\": %.3e, "
               "\"max_abs_vs_v2\": %.3e, \"bf16_out_mismatch\": %zu, \"rel_l2_new_vs_f64\": %.3e, "
               "\"rel_l2_v2_vs_f64\": %.3e}\n",
               vt.name, c.bh, c.sq, c.sk, pass ? "true" : "false", dv.rel_l2, dv.max_abs, mism, ref_new, ref_old);
        fflush(stdout);
        cudaFree(o1);
        cudaFree(o2);
        cudaFree(b2);
        cudaFree(dummy16);
        cudaFree(dummy32);
        free_problem(p);
    }
    return fails ? 1 : 0;
}

// ---- bench ------------------------------------------------------------------
struct Shape {
    const char* name;
    int bh, s;
};
static const Shape SHAPES[] = {
    {"h3_768p", 56, 37710},
    {"ltx_512p", 32, 6144},
    {"ltx_1080p20s", 32, 124440},
    {"ltx_4k5s", 32, 130560},
};

template <class F>
static double median_ms(F f) {
    f();
    CK(cudaDeviceSynchronize());
    std::vector<float> t;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[1];
}

// `which` = "v2" or a variant name.
static int bench_child(const char* which, const char* only) {
    CK(cudaFree(0));
    const int sm = device_sm();
    const Variant* vt = nullptr;
    for (const Variant& v : VARIANTS)
        if (!strcmp(v.name, which)) vt = &v;
    if (vt && sm / 10 != vt->sm / 10) return 0;
    for (const Shape& sh : SHAPES) {
        if (only && *only && !strstr(only, sh.name)) continue;
        rng_state = 0x9E3779B97F4A7C15ull;  // same data in every child
        Problem p = make_problem(sh.bh, sh.s, sh.s, 1.5f, 1.5f, false);
        const size_t n = (size_t)sh.bh * sh.s * 128;
        unsigned short* o16;
        float* dummy;
        CK(cudaMalloc(&o16, n * 2));
        CK(cudaMalloc(&dummy, 16));
        const float sl2 = 1.f / sqrtf(128.f) * 1.4426950408889634f;
        double ms;
        if (vt)
            ms = median_ms([&] { run_dc(*vt, p, dummy, o16, 1, sl2); });
        else
            ms = median_ms([&] { run_v2(p, dummy, o16, 1, sl2); });
        cudaError_t e = cudaDeviceSynchronize();
        // First two heads, as bf16, for the cross-kernel check.
        const size_t cmp = std::min(n, (size_t)2 * sh.s * 128);
        std::vector<unsigned short> h(cmp);
        if (e == cudaSuccess) CK(cudaMemcpy(h.data(), o16, cmp * 2, cudaMemcpyDeviceToHost));
        double sum = 0;
        for (size_t i = 0; i < cmp; i += 97) sum += bf2f(h[i]);
        const double flops = 4.0 * sh.bh * (double)sh.s * sh.s * 128;
        printf("{\"bench\": \"%s\", \"shape\": \"%s\", \"bh\": %d, \"tokens\": %d, \"ms\": %.3f, \"tflops\": %.1f, "
               "\"error\": \"%s\", \"checksum\": %.6e}\n",
               which, sh.name, sh.bh, sh.s, ms, flops / (ms * 1e-3) / 1e12, e == cudaSuccess ? "" : cudaGetErrorString(e), sum);
        fflush(stdout);
        // Dump the compared heads so the parent can diff variants.
        char path[256];
        snprintf(path, sizeof path, "/tmp/attn_dc_%s_%s.bin", which, sh.name);
        if (FILE* f = fopen(path, "wb")) {
            fwrite(h.data(), 2, cmp, f);
            fclose(f);
        }
        cudaFree(o16);
        cudaFree(dummy);
        free_problem(p);
        if (e != cudaSuccess) return 2;
    }
    return 0;
}

static int run_forked(int (*fn)(const char*, const char*), const char* a, const char* b, int cap_s) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) _exit(fn(a, b));
    int status = 0;
    for (int t = 0; t < cap_s * 10; t++) {
        if (waitpid(pid, &status, WNOHANG) == pid) {
            if (WIFSIGNALED(status)) {
                printf("{\"child\": \"%s\", \"signal\": %d}\n", a, WTERMSIG(status));
                return 128 + WTERMSIG(status);
            }
            return WEXITSTATUS(status);
        }
        usleep(100000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    printf("{\"child\": \"%s\", \"timeout_s\": %d}\n", a, cap_s);
    fflush(stdout);
    return 124;
}

static int parity_entry(const char* name, const char*) {
    for (const Variant& v : VARIANTS)
        if (!strcmp(v.name, name)) return parity_child(v);
    return 9;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "parity";
    const char* only = argc > 2 ? argv[2] : "";
    int rc = 0;
    if (!strcmp(mode, "parity")) {
        printf("{\"probe_done\": %d}\n", run_forked(probe_child, "probe", nullptr, 60));
        for (const Variant& v : VARIANTS) {
            const int r = run_forked(parity_entry, v.name, nullptr, 180);
            printf("{\"parity_done\": \"%s\", \"rc\": %d}\n", v.name, r);
            rc |= r != 0;
        }
    } else if (!strcmp(mode, "bench")) {
        run_forked(bench_child, "v2", only, 600);
        for (const Variant& v : VARIANTS) {
            const int r = run_forked(bench_child, v.name, only, 600);
            printf("{\"bench_done\": \"%s\", \"rc\": %d}\n", v.name, r);
            // Cross-check against v2 on the dumped heads.
            for (const Shape& sh : SHAPES) {
                char pa[256], pb[256];
                snprintf(pa, sizeof pa, "/tmp/attn_dc_v2_%s.bin", sh.name);
                snprintf(pb, sizeof pb, "/tmp/attn_dc_%s_%s.bin", v.name, sh.name);
                FILE* fa = fopen(pa, "rb");
                FILE* fb = fopen(pb, "rb");
                if (fa && fb) {
                    std::vector<unsigned short> x(2 * (size_t)sh.s * 128), y(x.size());
                    const size_t na = fread(x.data(), 2, x.size(), fa), nb = fread(y.data(), 2, y.size(), fb);
                    std::vector<float> fx(na), fy(nb);
                    for (size_t i = 0; i < na; i++) fx[i] = bf2f(x[i]);
                    for (size_t i = 0; i < nb; i++) fy[i] = bf2f(y[i]);
                    const Diff d = na == nb ? diff(fy.data(), fx.data(), na) : Diff{INFINITY, INFINITY};
                    printf("{\"bench_check\": \"%s\", \"shape\": \"%s\", \"rel_l2_vs_v2\": %.3e, \"max_abs\": %.3e}\n",
                           v.name, sh.name, d.rel_l2, d.max_abs);
                }
                if (fa) fclose(fa);
                if (fb) fclose(fb);
                remove(pb);
            }
        }
    } else {
        fprintf(stderr, "usage: %s parity|bench [shape,...]\n", argv[0]);
        return 2;
    }
    return rc;
}
