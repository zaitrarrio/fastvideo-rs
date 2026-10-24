// ==== region: attn3 (sm_120 dense tuning) ====
//
// flash_mma_fwd3_d128: flash_mma_fwd2's CTA (128 queries, 8 warps x 16 rows)
// and per-row arithmetic, rescheduled so a warp always has tensor-core work
// independent of its softmax:
//
//   * S_{j+1} = Q K_{j+1}^T is issued before softmax(S_j), in the same basic
//     block, so ptxas interleaves the QK HMMAs with the max / exp2 / sum
//     chain of the previous tile instead of leaving the tensor pipe idle
//     while both warps of an SMSP run their softmax in lockstep;
//   * a three-stage (K, V) ring (96 KB): tile j+2 is issued when tile j
//     starts, one CTA barrier per 64 keys (as fwd2);
//   * tail-key masking by selects, not a branch, to keep that block whole.
//
// SKIP = false is bit-identical to flash_mma_fwd2 (same operations on the
// same values in the same order). SKIP = true (fwd3s) also skips the O
// rescale when no row of the warp raised its max: fwd2 then multiplies O by
// 2^(m*sl2 - rn(m*sl2)), a factor within a few ulp of 1, which fwd3s takes
// as exactly 1 (so it differs from fwd2 by that rounding residue only).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <int D, bool SKIP>
__device__ __forceinline__ void flash_mma_fwd3_body(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2,
    unsigned char* smem
) {
    constexpr int TILEB = FA_TILE * D * 2;
    constexpr int STAGEB = 2 * TILEB;   // K tile then V tile
    constexpr int BR = 128;
    const unsigned int base = mma_smem_u32(smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int q0 = (int)blockIdx.x * BR;
    const long bh = blockIdx.y;
    if (q0 >= sq || sk <= 0) return;
    const int qlen = min(BR, sq - q0);
    const unsigned short* Qh = q + (bh * sq + q0) * D;
    const unsigned short* Kh = k + bh * (long)sk * D;
    const unsigned short* Vh = v + bh * (long)sk * D;
    const int nkt = (sk + FA_TILE - 1) / FA_TILE;
    const float NEG = __int_as_float(0xff800000);

    // Q (128 rows) through stage 2; (K_0, V_0) -> stage 0; (K_1, V_1) -> stage 1.
    fa2_load_rows<D, BR, 256>(base + 2 * STAGEB, Qh, qlen, tid);
    mma_cp_commit();
    fa2_load_rows<D, FA_TILE, 256>(base, Kh, min(FA_TILE, sk), tid);
    fa2_load_rows<D, FA_TILE, 256>(base + TILEB, Vh, min(FA_TILE, sk), tid);
    mma_cp_commit();
    if (nkt > 1) {
        const int nlen = min(FA_TILE, sk - FA_TILE);
        fa2_load_rows<D, FA_TILE, 256>(base + STAGEB, Kh + (long)FA_TILE * D, nlen, tid);
        fa2_load_rows<D, FA_TILE, 256>(base + STAGEB + TILEB, Vh + (long)FA_TILE * D, nlen, tid);
        mma_cp_commit();
        mma_cp_wait<1>();
    } else {
        mma_cp_wait<0>();
    }
    __syncthreads();   // Q and (K_0, V_0) landed
    unsigned int qf[D / 16][4];
    #pragma unroll
    for (int kc = 0; kc < D / 16; kc++) {
        const int row = warp * 16 + (lane & 15), col = kc * 16 + (lane >> 4) * 8;
        mma_ldm_x4(base + 2 * STAGEB + fa_swz<D>(row, col), qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3]);
    }

    float o[D / 8][4];
    #pragma unroll
    for (int n = 0; n < D / 8; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float s[8][4];
    fa_qk<D>(base, qf, s, lane);   // S_0
    unsigned int pa[4][4];
    // Stage offsets of tiles j, j+1, j+2 (rotated each iteration).
    unsigned int st0 = 0, st1 = STAGEB, st2 = 2 * STAGEB;

    #pragma unroll 1
    for (int j = 0; j < nkt; j++) {
        const int kv0 = j * FA_TILE, len = min(FA_TILE, sk - kv0);
        mma_cp_wait<0>();
        __syncthreads();   // (K_{j+1}, V_{j+1}) landed; every warp is past j-1 (and Q)
        if (j + 2 < nkt) {
            const int nlen = min(FA_TILE, sk - kv0 - 2 * FA_TILE);
            fa2_load_rows<D, FA_TILE, 256>(base + st2, Kh + (long)(kv0 + 2 * FA_TILE) * D, nlen, tid);
            fa2_load_rows<D, FA_TILE, 256>(base + st2 + TILEB, Vh + (long)(kv0 + 2 * FA_TILE) * D, nlen, tid);
            mma_cp_commit();
        }
        // S_{j+1}: independent of softmax(S_j) below. On the last tile it
        // reads a stale stage and is discarded.
        float sn[8][4];
        fa_qk<D>(base + st1, qf, sn, lane);

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            const int c0 = n * 8 + 2 * t;
            const bool k0 = c0 < len, k1 = c0 + 1 < len;
            s[n][0] = k0 ? s[n][0] : NEG; s[n][2] = k0 ? s[n][2] : NEG;
            s[n][1] = k1 ? s[n][1] : NEG; s[n][3] = k1 ? s[n][3] : NEG;
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
        float a0 = fa_exp2(fmaf(m0, sl2, -ms0)), a1 = fa_exp2(fmaf(m1, sl2, -ms1));
        bool keep = false;
        if (SKIP) {
            keep = __all_sync(0xffffffffu, mn0 == m0 && mn1 == m1);
            if (mn0 == m0) a0 = 1.f;
            if (mn1 == m1) a1 = 1.f;
        }
        m0 = mn0; m1 = mn1;
        float ls0 = 0.f, ls1 = 0.f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            s[n][0] = fa_exp2(fmaf(s[n][0], sl2, -ms0));
            s[n][1] = fa_exp2(fmaf(s[n][1], sl2, -ms0));
            s[n][2] = fa_exp2(fmaf(s[n][2], sl2, -ms1));
            s[n][3] = fa_exp2(fmaf(s[n][3], sl2, -ms1));
            ls0 += s[n][0] + s[n][1];
            ls1 += s[n][2] + s[n][3];
        }
        l0 = fmaf(l0, a0, ls0);
        l1 = fmaf(l1, a1, ls1);
        if (!keep) {
            #pragma unroll
            for (int n = 0; n < D / 8; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
        }
        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            pa[kc][0] = mma_pack_bf16_rn(s[2 * kc][0], s[2 * kc][1]);
            pa[kc][1] = mma_pack_bf16_rn(s[2 * kc][2], s[2 * kc][3]);
            pa[kc][2] = mma_pack_bf16_rn(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[kc][3] = mma_pack_bf16_rn(s[2 * kc + 1][2], s[2 * kc + 1][3]);
        }
        fa_pv<D>(base + st0 + TILEB, pa, o, lane);
        #pragma unroll
        for (int n = 0; n < 8; n++) { s[n][0] = sn[n][0]; s[n][1] = sn[n][1]; s[n][2] = sn[n][2]; s[n][3] = sn[n][3]; }
        const unsigned int tmp = st0; st0 = st1; st1 = st2; st2 = tmp;
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool ok0 = row0 < qlen, ok1 = row1 < qlen;
    const long r0 = (bh * sq + q0 + row0) * D, r1 = r0 + 8L * D;
    if (out_is_bf16) {
        #pragma unroll
        for (int n = 0; n < D / 8; n++) {
            const int col = n * 8 + 2 * t;
            if (ok0) *reinterpret_cast<unsigned int*>(out_bf16 + r0 + col) = mma_pack_bf16_rn(o[n][0] * inv0, o[n][1] * inv0);
            if (ok1) *reinterpret_cast<unsigned int*>(out_bf16 + r1 + col) = mma_pack_bf16_rn(o[n][2] * inv1, o[n][3] * inv1);
        }
    } else {
        #pragma unroll
        for (int n = 0; n < D / 8; n++) {
            const int col = n * 8 + 2 * t;
            if (ok0) *reinterpret_cast<float2*>(out + r0 + col) = make_float2(o[n][0] * inv0, o[n][1] * inv0);
            if (ok1) *reinterpret_cast<float2*>(out + r1 + col) = make_float2(o[n][2] * inv1, o[n][3] * inv1);
        }
    }
}
#define FA3_ENTRY_BODY(D, SKIP)                                                         \
    extern __shared__ __align__(128) unsigned char fa3_smem[];                          \
    flash_mma_fwd3_body<D, SKIP>(q, k, v, out, out_bf16, out_is_bf16, sq, sk, sl2, fa3_smem);
#else
#define FA3_ENTRY_BODY(D, SKIP)                                                         \
    (void)q; (void)k; (void)v; (void)out; (void)out_bf16; (void)out_is_bf16;            \
    (void)sq; (void)sk; (void)sl2;                                                      \
    __trap();
#endif

// Dynamic shared memory: 3 stages x (K, V) x 64 x 128 bf16 = 96 KB (opt-in).
extern "C" __global__ void __launch_bounds__(256, 1) flash_mma_fwd3_d128(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2
) {
    FA3_ENTRY_BODY(128, false)
}

extern "C" __global__ void __launch_bounds__(256, 1) flash_mma_fwd3s_d128(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2
) {
    FA3_ENTRY_BODY(128, true)
}

// ---- VSA on bf16 activations -----------------------------------------------
// H3 stores q/k/v/gate in bf16 (FASTVIDEO_BF16_ACT). The f32 VSA kernels
// (vsa_tile_mean round16, vsa_tile_qkv, vsa_combine round16) round every
// input to bf16 before use, so reading the bf16 tensors directly gives the
// same bits without the f32 widening of all four tensors.

// vsa_tile_mean(round16 = 1) on bf16 input.
extern "C" __global__ void vsa_tile_mean_b16(
    const unsigned short* x, const int* slot_src, const int* block_sizes, float* out,
    long seq, int dim, int num_tiles, int tile_elems
) {
    int tile = blockIdx.x;
    long bh = blockIdx.y;
    if (tile >= num_tiles) return;
    int n = block_sizes[tile];
    const unsigned short* xb = x + bh * seq * (long)dim;
    float* ob = out + (bh * (long)num_tiles + tile) * (long)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) {
            int tok = slot_src[(long)tile * tile_elems + j];
            acc += __uint_as_float(((unsigned int)xb[(long)tok * dim + d]) << 16);
        }
        ob[d] = n > 0 ? acc / (float)n : 0.0f;
    }
}

// vsa_tile_qkv on bf16 input: a gather, 16 bytes (8 elements) per thread.
// `dim` is a multiple of 8.
extern "C" __global__ void vsa_tile_qkv_b16(
    const unsigned short* __restrict__ x, const int* __restrict__ slot_src, unsigned short* __restrict__ xt,
    long seq, long padded, int dim
) {
    const long chunks = (long)dim / 8;
    long rem = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (rem >= padded * chunks) return;
    long bh = blockIdx.z;
    long slot = rem / chunks;
    long c = rem - slot * chunks;
    int src = slot_src[slot];
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (src >= 0) v = *reinterpret_cast<const uint4*>(x + (bh * seq + src) * dim + c * 8);
    *reinterpret_cast<uint4*>(xt + (bh * padded + slot) * dim + c * 8) = v;
}

// vsa_combine(round16 = 1) with a bf16 gate.
extern "C" __global__ void vsa_combine_g16(
    const float* sparse, const float* coarse, const unsigned short* gate, const int* slot_src,
    float* out, long seq, int dim, int tile_elems, int q_base, int num_tiles, int has_gate
) {
    long slot_in_group = (long)blockIdx.x * blockDim.y + threadIdx.y;
    long g = blockIdx.y;
    long bh = blockIdx.z;
    if (slot_in_group >= tile_elems) return;
    int tile = q_base + (int)g;
    if (tile >= num_tiles) return;
    int src = slot_src[(long)tile * tile_elems + slot_in_group];
    if (src < 0) return;
    const float* sp = sparse + ((bh * gridDim.y + g) * (long)tile_elems + slot_in_group) * (long)dim;
    const float* co = coarse + (bh * (long)num_tiles + tile) * (long)dim;
    float* ob = out + bh * seq * (long)dim + (long)src * dim;
    const unsigned short* ga = has_gate ? gate + bh * seq * (long)dim + (long)src * dim : nullptr;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
#define FV_R16(x) __uint_as_float(((unsigned int)fv_bf16_rne(x)) << 16)
        float c = FV_R16(co[d]);
        float p = ga ? FV_R16(c * __uint_as_float(((unsigned int)ga[d]) << 16)) : c;
        ob[d] = FV_R16(FV_R16(sp[d]) + p);
#undef FV_R16
    }
}

// vsa_topk with the same output, bit for bit: T = the k-th largest sortable
// key (vsa_topk's "largest T with count(key >= T) >= k"), found by an 8-bit
// radix select (4 histogram passes over the row held in shared memory)
// instead of 32 bisection passes; then the indices with key > T in ascending
// order, then those with key == T in ascending order up to k, written by a
// block-wide ordered compaction instead of one thread. 256 threads; the row
// (n <= FV_TOPK2_MAXN keys) lives in shared memory.
#define FV_TOPK2_MAXN 4096
extern "C" __global__ void __launch_bounds__(256) vsa_topk2(
    const float* scores, unsigned int* out, int rows, int n, int k
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned int keys[FV_TOPK2_MAXN];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int sel_prefix, sel_rem, warp_cnt[8], base_s;
    const float* s = scores + (long)row * n;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < n; i += 256) keys[i] = fv_sortable(s[i]);
    if (tid == 0) { sel_prefix = 0u; sel_rem = (unsigned int)k; }
    __syncthreads();
    #pragma unroll 1
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        const unsigned int hi_mask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
        hist[tid] = 0u;
        __syncthreads();
        const unsigned int prefix = sel_prefix;
        for (int i = tid; i < n; i += 256) {
            const unsigned int key = keys[i];
            if ((key & hi_mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (warp == 0) {
            // Lane l owns bins [8l, 8l + 8); suffix counts from the top bin down.
            unsigned int local = 0;
            #pragma unroll
            for (int b = 0; b < 8; b++) local += hist[lane * 8 + b];
            unsigned int suffix = local;   // inclusive sum over lanes >= lane
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const unsigned int o = __shfl_down_sync(0xffffffffu, suffix, off);
                if (lane + off < 32) suffix += o;
            }
            const unsigned int rem = sel_rem;
            // The bin holding the rem-th largest: the lane whose suffix range covers it.
            const unsigned int above = suffix - local;   // keys in lanes > lane
            const bool mine = above < rem && suffix >= rem;
            if (mine) {
                unsigned int acc = above;
                int b = 7;
                for (; b > 0; b--) {
                    const unsigned int c = hist[lane * 8 + b];
                    if (acc + c >= rem) break;
                    acc += c;
                }
                sel_prefix = prefix | ((unsigned int)(lane * 8 + b) << shift);
                sel_rem = rem - acc;
            }
        }
        __syncthreads();
    }
    const unsigned int thr = sel_prefix;
    unsigned int* dst = out + (long)row * k;
    // Ordered compaction: pass 0 takes key > thr, pass 1 key == thr.
    if (tid == 0) base_s = 0u;
    __syncthreads();
    #pragma unroll 1
    for (int pass = 0; pass < 2; pass++) {
        #pragma unroll 1
        for (int c0 = 0; c0 < n; c0 += 256) {
            const int i = c0 + tid;
            const unsigned int key = i < n ? keys[i] : 0u;
            const bool f = i < n && (pass == 0 ? key > thr : key == thr);
            const unsigned int m = __ballot_sync(0xffffffffu, f);
            if (lane == 0) warp_cnt[warp] = __popc(m);
            __syncthreads();
            unsigned int before = base_s;
            for (int w = 0; w < warp; w++) before += warp_cnt[w];
            const unsigned int pos = before + __popc(m & ((1u << lane) - 1u));
            if (f && pos < (unsigned int)k) dst[pos] = (unsigned int)i;
            __syncthreads();
            if (tid == 0) {
                unsigned int tot = 0;
                for (int w = 0; w < 8; w++) tot += warp_cnt[w];
                base_s += tot;
            }
            __syncthreads();
        }
    }
    for (int p = (int)min(base_s, (unsigned int)k) + tid; p < k; p += 256) dst[p] = 0u;
}
// ==== end region: attn3 ====

