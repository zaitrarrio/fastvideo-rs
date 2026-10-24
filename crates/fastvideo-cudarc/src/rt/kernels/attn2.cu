// ==== region: attn2 (Phase 3b attention kernels) ====
//
// Second-generation dense / VSA / Sol forward kernels. Each one does the
// same per-row arithmetic, in the same order, as the kernel it replaces
// (flash_mma_fwd_d*, vsa_mma_attn_tma, sol_mma_fwd), so the old kernel is a
// bit-exact oracle for the new one; only the schedule changes:
//
//   flash_mma_fwd2_d{64,128}  128-query CTAs (8 warps x 16 rows) share each
//                             K/V tile between twice the queries, and K/V are
//                             double-buffered: one CTA barrier per KV tile.
//   vsa_mma_attn_tma2         three-slot TMA ring (48 KB) instead of two
//                             K/V stages (64 KB): two CTAs per SM on SM12x.
//   sol_mma_fwd2              warp-specialised Sol-Attn: one cp.async
//                             producer warp + four MMA warps over mbarriers,
//                             plus KV splits over route groups (sm90 recipe)
//                             merged by sol_split_combine.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// cp.async a ROWS x D bf16 tile (rows of 2D bytes) with NT threads; rows >=
// rows_valid are zero-filled and never read from global.
template <int D, int ROWS, int NT>
__device__ __forceinline__ void fa2_load_rows(unsigned int smem, const unsigned short* g, int rows_valid, int tid) {
    constexpr int CPR = D / 8;
    #pragma unroll
    for (int i = 0; i < ROWS * CPR / NT; i++) {
        const int chunk = tid + i * NT;
        const int row = chunk / CPR, c = chunk % CPR;
        const int ok = row < rows_valid;
        mma_cp_async16_zfill(smem + fa_swz<D>(row, c * 8),
                             ok ? (const void*)(g + (long)row * D + c * 8) : (const void*)g, ok);
    }
}

// Dense SDPA, FA2 schedule: CTA = 128 queries x (batch*head), 8 warps, warp
// w owns rows [16w, 16w+16). Stage s holds K_j at 2s*TILEB and V_j right
// after it. Iteration j: wait for (K_j, V_j), one CTA barrier (every warp is
// also past iteration j-1, so stage j^1 is free), prefetch (K_{j+1},
// V_{j+1}) into it, then S = Q K_j^T, online softmax, O += P V_j. The
// softmax/rounding sequence per row is flash_mma_fwd_body's, line for line.
template <int D>
__device__ __forceinline__ void flash_mma_fwd2_body(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2,
    unsigned char* smem
) {
    constexpr int TILEB = FA_TILE * D * 2;
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

    // Q (128 rows) -> registers through stage 1 (2 * TILEB bytes = 128 rows).
    fa2_load_rows<D, BR, 256>(base + 2 * TILEB, Qh, qlen, tid);
    mma_cp_commit();
    // (K_0, V_0) -> stage 0, in flight while Q is unpacked.
    fa2_load_rows<D, FA_TILE, 256>(base, Kh, min(FA_TILE, sk), tid);
    fa2_load_rows<D, FA_TILE, 256>(base + TILEB, Vh, min(FA_TILE, sk), tid);
    mma_cp_commit();
    mma_cp_wait<1>();
    __syncthreads();
    unsigned int qf[D / 16][4];
    #pragma unroll
    for (int kc = 0; kc < D / 16; kc++) {
        const int row = warp * 16 + (lane & 15), col = kc * 16 + (lane >> 4) * 8;
        mma_ldm_x4(base + 2 * TILEB + fa_swz<D>(row, col), qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3]);
    }

    float o[D / 8][4];
    #pragma unroll
    for (int n = 0; n < D / 8; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float s[8][4];
    unsigned int pa[4][4];

    #pragma unroll 1
    for (int j = 0; j < nkt; j++) {
        const int kv0 = j * FA_TILE, len = min(FA_TILE, sk - kv0);
        const unsigned int sK = base + (j & 1) * 2 * TILEB, sV = sK + TILEB;
        mma_cp_wait<0>();
        __syncthreads();   // (K_j, V_j) landed; every warp is past j-1 (and past Q)
        if (j + 1 < nkt) {
            const unsigned int nK = base + ((j + 1) & 1) * 2 * TILEB;
            const int nlen = min(FA_TILE, sk - kv0 - FA_TILE);
            fa2_load_rows<D, FA_TILE, 256>(nK, Kh + (long)(kv0 + FA_TILE) * D, nlen, tid);
            fa2_load_rows<D, FA_TILE, 256>(nK + TILEB, Vh + (long)(kv0 + FA_TILE) * D, nlen, tid);
            mma_cp_commit();
        }
        fa_qk<D>(sK, qf, s, lane);
        if (len < FA_TILE) {
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                const int c0 = n * 8 + 2 * t;
                if (c0 >= len)     { s[n][0] = NEG; s[n][2] = NEG; }
                if (c0 + 1 >= len) { s[n][1] = NEG; s[n][3] = NEG; }
            }
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
        const float a0 = fa_exp2(fmaf(m0, sl2, -ms0)), a1 = fa_exp2(fmaf(m1, sl2, -ms1));
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
        #pragma unroll
        for (int n = 0; n < D / 8; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            pa[kc][0] = mma_pack_bf16_rn(s[2 * kc][0], s[2 * kc][1]);
            pa[kc][1] = mma_pack_bf16_rn(s[2 * kc][2], s[2 * kc][3]);
            pa[kc][2] = mma_pack_bf16_rn(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[kc][3] = mma_pack_bf16_rn(s[2 * kc + 1][2], s[2 * kc + 1][3]);
        }
        fa_pv<D>(sV, pa, o, lane);
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
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// flash_mma_fwd2_body with Wan's block-causal temporal mask
// (`causal_temporal_mask`, FastVideo `_prepare_blockwise_causal_attn_mask`),
// same per-row arithmetic: query q sees key k iff block(k) <= block(q)
// (block = index / ft, ft = frames per block x frame tokens), within the
// last `win` blocks up to and including q's own when win > 0, or
// block(k) < sink. Key tiles the CTA's 128 queries
// cannot see are never loaded; the tiles it walks are the sink tiles, then
// the contiguous causal band, in key order. Tiles straddling the boundary
// get -inf scores where masked, which the online softmax treats exactly as a
// skipped key (exp2(-inf) = 0, a row max of -inf leaves the state as is).
template <int D, bool CAUSAL>
__device__ __forceinline__ void flash_mma_fwd2_causal_body(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2,
    unsigned char* smem, int ft, int win, int sink
) {
    constexpr int TILEB = FA_TILE * D * 2;
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
    // Walk order: tiles [0, ns) then [mlo, mlo + ne - ns). Dense: 0..nkt.
    int ns = 0, mlo = 0, ne = nkt;
    int fq_lo = 0, fq_hi = 0, tq0 = 0, tq1 = 0;
    if constexpr (CAUSAL) {
        fq_lo = q0 / ft;
        fq_hi = (q0 + qlen - 1) / ft;
        const int hi_tok = min(sk, (fq_hi + 1) * ft);
        const int lo_tok = min(hi_tok, win > 0 ? max(0, fq_lo - win + 1) * ft : 0);
        const int sink_tiles = (min(sk, sink * ft) + FA_TILE - 1) / FA_TILE;
        mlo = lo_tok / FA_TILE;
        const int mhi = (hi_tok + FA_TILE - 1) / FA_TILE;
        ns = min(sink_tiles, mlo);
        ne = ns + (mhi - mlo);
        tq0 = (q0 + warp * 16 + g) / ft;
        tq1 = (q0 + warp * 16 + g + 8) / ft;
    }
    auto tile_of = [&](int e) {
        if constexpr (CAUSAL) return e < ns ? e : mlo + (e - ns);
        else return e;
    };
    const int kt0 = tile_of(0);

    // Q (128 rows) -> registers through stage 1 (2 * TILEB bytes = 128 rows).
    fa2_load_rows<D, BR, 256>(base + 2 * TILEB, Qh, qlen, tid);
    mma_cp_commit();
    // (K_0, V_0) -> stage 0, in flight while Q is unpacked.
    fa2_load_rows<D, FA_TILE, 256>(base, Kh + (long)kt0 * FA_TILE * D, min(FA_TILE, sk - kt0 * FA_TILE), tid);
    fa2_load_rows<D, FA_TILE, 256>(base + TILEB, Vh + (long)kt0 * FA_TILE * D, min(FA_TILE, sk - kt0 * FA_TILE), tid);
    mma_cp_commit();
    mma_cp_wait<1>();
    __syncthreads();
    unsigned int qf[D / 16][4];
    #pragma unroll
    for (int kc = 0; kc < D / 16; kc++) {
        const int row = warp * 16 + (lane & 15), col = kc * 16 + (lane >> 4) * 8;
        mma_ldm_x4(base + 2 * TILEB + fa_swz<D>(row, col), qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3]);
    }

    float o[D / 8][4];
    #pragma unroll
    for (int n = 0; n < D / 8; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float s[8][4];
    unsigned int pa[4][4];

    #pragma unroll 1
    for (int j = 0; j < ne; j++) {
        const int kt = tile_of(j);
        const int kv0 = kt * FA_TILE, len = min(FA_TILE, sk - kv0);
        const unsigned int sK = base + (j & 1) * 2 * TILEB, sV = sK + TILEB;
        mma_cp_wait<0>();
        __syncthreads();   // (K_j, V_j) landed; every warp is past j-1 (and past Q)
        if (j + 1 < ne) {
            const unsigned int nK = base + ((j + 1) & 1) * 2 * TILEB;
            const int nkv0 = tile_of(j + 1) * FA_TILE;
            const int nlen = min(FA_TILE, sk - nkv0);
            fa2_load_rows<D, FA_TILE, 256>(nK, Kh + (long)nkv0 * D, nlen, tid);
            fa2_load_rows<D, FA_TILE, 256>(nK + TILEB, Vh + (long)nkv0 * D, nlen, tid);
            mma_cp_commit();
        }
        fa_qk<D>(sK, qf, s, lane);
        if (len < FA_TILE) {
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                const int c0 = n * 8 + 2 * t;
                if (c0 >= len)     { s[n][0] = NEG; s[n][2] = NEG; }
                if (c0 + 1 >= len) { s[n][1] = NEG; s[n][3] = NEG; }
            }
        }
        if constexpr (CAUSAL) {
            // Every (query, key) pair of the CTA visible: no per-score mask.
            const int tk_lo = kv0 / ft, tk_hi = (kv0 + len - 1) / ft;
            const bool all_ok = tk_hi < sink || (tk_hi <= fq_lo && (win <= 0 || fq_hi - tk_lo < win));
            if (!all_ok) {
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    #pragma unroll
                    for (int c = 0; c < 2; c++) {
                        const int tk = (kv0 + n * 8 + 2 * t + c) / ft;
                        const bool sink_ok = tk < sink;
                        if (!(sink_ok || (tk <= tq0 && (win <= 0 || tq0 - tk < win)))) s[n][c] = NEG;
                        if (!(sink_ok || (tk <= tq1 && (win <= 0 || tq1 - tk < win)))) s[n][2 + c] = NEG;
                    }
                }
            }
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
        const float a0 = fa_exp2(fmaf(m0, sl2, -ms0)), a1 = fa_exp2(fmaf(m1, sl2, -ms1));
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
        #pragma unroll
        for (int n = 0; n < D / 8; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            pa[kc][0] = mma_pack_bf16_rn(s[2 * kc][0], s[2 * kc][1]);
            pa[kc][1] = mma_pack_bf16_rn(s[2 * kc][2], s[2 * kc][3]);
            pa[kc][2] = mma_pack_bf16_rn(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[kc][3] = mma_pack_bf16_rn(s[2 * kc + 1][2], s[2 * kc + 1][3]);
        }
        fa_pv<D>(sV, pa, o, lane);
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
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// VDN-H3's window softmax (docs/ports/vdn-h3.md §1.3) on the packed sequence:
// rows outside the video span [vs, ve) are global (dense both ways); video
// frame f = (row - vs) / tpf; frames 0 and nf-1 are dense anchors (rows and
// columns); otherwise query frame qf sees key frame kf iff kf lies in qf's
// window, chunk-aligned (chunk > 0: whole chunks c-radius..c+radius of
// c = qf / chunk) or centred (chunk == 0).
__device__ __forceinline__ int vdn_win_lo(int f, int chunk, int radius) {
    return chunk > 0 ? (f / chunk - radius) * chunk : f - radius;
}
__device__ __forceinline__ int vdn_win_hi(int f, int chunk, int radius) {
    return chunk > 0 ? (f / chunk + radius + 1) * chunk - 1 : f + radius;
}
__device__ __forceinline__ bool vdn_win_allows(int qr, int kr, int vs, int ve, int tpf, int nf, int chunk, int radius) {
    if (qr < vs || qr >= ve || kr < vs || kr >= ve) return true;
    const int qf = (qr - vs) / tpf, kf = (kr - vs) / tpf;
    if (qf == 0 || qf == nf - 1 || kf == 0 || kf == nf - 1) return true;
    return kf >= vdn_win_lo(qf, chunk, radius) && kf <= vdn_win_hi(qf, chunk, radius);
}

// flash_mma_fwd2_causal_body's arithmetic under the VDN window mask. A CTA
// whose 128 queries are all inner-frame video rows walks at most three key
// tile ranges (globals before the video + anchor frame 0, the union of its
// rows' windows, anchor frame nf-1 + globals after), in key order; any other
// CTA walks every tile. Tiles where every (query, key) pair is kept skip the
// per-score mask; the rest get -inf where masked (a skipped key, exactly).
template <int D>
__device__ __forceinline__ void flash_mma_fwd2_window_body(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2,
    unsigned char* smem, int vs, int ve, int tpf, int nf, int chunk, int radius
) {
    constexpr int TILEB = FA_TILE * D * 2;
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
    const int qlast = q0 + qlen - 1;
    // Rows of this CTA that see only a window: inner-frame video rows.
    const int in_lo = vs + tpf, in_hi = vs + (nf - 1) * tpf;   // [in_lo, in_hi)
    const bool any_window = nf > 2 && max(q0, in_lo) < min(qlast + 1, in_hi);
    const bool all_window = nf > 2 && q0 >= in_lo && qlast < in_hi;
    // Walk: ranges [rs[i], rs[i] + rn[i]) of tiles.
    int rs0 = 0, rn0 = nkt, rs1 = 0, rn1 = 0, rs2 = 0, rn2 = 0;
    // The frames every row of an all-window CTA sees (intersection of windows).
    int iw_lo = 0, iw_hi = -1;
    if (all_window) {
        const int fq_lo = (q0 - vs) / tpf, fq_hi = (qlast - vs) / tpf;
        const int ulo = max(vdn_win_lo(fq_lo, chunk, radius), 0);
        const int uhi = min(vdn_win_hi(fq_hi, chunk, radius), nf - 1);
        iw_lo = vdn_win_lo(fq_hi, chunk, radius);
        iw_hi = vdn_win_hi(fq_lo, chunk, radius);
        const int ta1 = (min(sk, vs + tpf) + FA_TILE - 1) / FA_TILE;
        const int tb0 = (vs + ulo * tpf) / FA_TILE;
        const int tb1 = (min(sk, vs + (uhi + 1) * tpf) + FA_TILE - 1) / FA_TILE;
        const int tc0 = (vs + (nf - 1) * tpf) / FA_TILE;
        const int tc1 = nkt;
        int s[3] = {0, 0, 0}, e[3] = {0, 0, 0}, m = 0;
        const int cs[3] = {0, tb0, tc0}, ce[3] = {ta1, tb1, tc1};
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            if (ce[i] <= cs[i]) continue;
            if (m > 0 && cs[i] <= e[m - 1]) {
                e[m - 1] = max(e[m - 1], ce[i]);
            } else {
                s[m] = cs[i];
                e[m] = ce[i];
                m++;
            }
        }
        rs0 = s[0]; rn0 = m > 0 ? e[0] - s[0] : 0;
        rs1 = s[1]; rn1 = m > 1 ? e[1] - s[1] : 0;
        rs2 = s[2]; rn2 = m > 2 ? e[2] - s[2] : 0;
    }
    const int ne = rn0 + rn1 + rn2;
    auto tile_of = [&](int j) {
        if (j < rn0) return rs0 + j;
        if (j < rn0 + rn1) return rs1 + (j - rn0);
        return rs2 + (j - rn0 - rn1);
    };
    const int kt0 = tile_of(0);
    const int qr0 = q0 + warp * 16 + g, qr1 = qr0 + 8;

    fa2_load_rows<D, BR, 256>(base + 2 * TILEB, Qh, qlen, tid);
    mma_cp_commit();
    fa2_load_rows<D, FA_TILE, 256>(base, Kh + (long)kt0 * FA_TILE * D, min(FA_TILE, sk - kt0 * FA_TILE), tid);
    fa2_load_rows<D, FA_TILE, 256>(base + TILEB, Vh + (long)kt0 * FA_TILE * D, min(FA_TILE, sk - kt0 * FA_TILE), tid);
    mma_cp_commit();
    mma_cp_wait<1>();
    __syncthreads();
    unsigned int qf[D / 16][4];
    #pragma unroll
    for (int kc = 0; kc < D / 16; kc++) {
        const int row = warp * 16 + (lane & 15), col = kc * 16 + (lane >> 4) * 8;
        mma_ldm_x4(base + 2 * TILEB + fa_swz<D>(row, col), qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3]);
    }

    float o[D / 8][4];
    #pragma unroll
    for (int n = 0; n < D / 8; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float s[8][4];
    unsigned int pa[4][4];

    #pragma unroll 1
    for (int j = 0; j < ne; j++) {
        const int kt = tile_of(j);
        const int kv0 = kt * FA_TILE, len = min(FA_TILE, sk - kv0);
        const unsigned int sK = base + (j & 1) * 2 * TILEB, sV = sK + TILEB;
        mma_cp_wait<0>();
        __syncthreads();
        if (j + 1 < ne) {
            const unsigned int nK = base + ((j + 1) & 1) * 2 * TILEB;
            const int nkv0 = tile_of(j + 1) * FA_TILE;
            const int nlen = min(FA_TILE, sk - nkv0);
            fa2_load_rows<D, FA_TILE, 256>(nK, Kh + (long)nkv0 * D, nlen, tid);
            fa2_load_rows<D, FA_TILE, 256>(nK + TILEB, Vh + (long)nkv0 * D, nlen, tid);
            mma_cp_commit();
        }
        fa_qk<D>(sK, qf, s, lane);
        if (len < FA_TILE) {
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                const int c0 = n * 8 + 2 * t;
                if (c0 >= len)     { s[n][0] = NEG; s[n][2] = NEG; }
                if (c0 + 1 >= len) { s[n][1] = NEG; s[n][3] = NEG; }
            }
        }
        if (any_window) {
            // Every pair of the tile kept: a tile of global rows, of one
            // anchor frame, or inside every row's window (all-window CTAs).
            const int k1 = kv0 + len - 1;
            bool all_ok = k1 < vs || kv0 >= ve;
            if (!all_ok && kv0 >= vs && k1 < ve) {
                const int tk_lo = (kv0 - vs) / tpf, tk_hi = (k1 - vs) / tpf;
                all_ok = (tk_lo == tk_hi && (tk_lo == 0 || tk_lo == nf - 1))
                      || (all_window && tk_lo >= iw_lo && tk_hi <= iw_hi);
            }
            if (!all_ok) {
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    #pragma unroll
                    for (int c = 0; c < 2; c++) {
                        const int kr = kv0 + n * 8 + 2 * t + c;
                        if (!vdn_win_allows(qr0, kr, vs, ve, tpf, nf, chunk, radius)) s[n][c] = NEG;
                        if (!vdn_win_allows(qr1, kr, vs, ve, tpf, nf, chunk, radius)) s[n][2 + c] = NEG;
                    }
                }
            }
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
        const float a0 = fa_exp2(fmaf(m0, sl2, -ms0)), a1 = fa_exp2(fmaf(m1, sl2, -ms1));
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
        #pragma unroll
        for (int n = 0; n < D / 8; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            pa[kc][0] = mma_pack_bf16_rn(s[2 * kc][0], s[2 * kc][1]);
            pa[kc][1] = mma_pack_bf16_rn(s[2 * kc][2], s[2 * kc][3]);
            pa[kc][2] = mma_pack_bf16_rn(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[kc][3] = mma_pack_bf16_rn(s[2 * kc + 1][2], s[2 * kc + 1][3]);
        }
        fa_pv<D>(sV, pa, o, lane);
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
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define FA2_ENTRY_BODY(D)                                                               \
    extern __shared__ __align__(128) unsigned char fa2_smem[];                          \
    flash_mma_fwd2_body<D>(q, k, v, out, out_bf16, out_is_bf16, sq, sk, sl2, fa2_smem);
#define FA2C_ENTRY_BODY(D)                                                              \
    extern __shared__ __align__(128) unsigned char fa2_smem[];                          \
    flash_mma_fwd2_causal_body<D, true>(q, k, v, out, out_bf16, out_is_bf16, sq, sk, sl2, fa2_smem, ft, win, sink);
#define FA2W_ENTRY_BODY(D)                                                              \
    extern __shared__ __align__(128) unsigned char fa2_smem[];                          \
    flash_mma_fwd2_window_body<D>(q, k, v, out, out_bf16, out_is_bf16, sq, sk, sl2, fa2_smem, vs, ve, tpf, nf, chunk, radius);
#else
#define FA2W_ENTRY_BODY(D)                                                              \
    (void)q; (void)k; (void)v; (void)out; (void)out_bf16; (void)out_is_bf16;            \
    (void)sq; (void)sk; (void)sl2; (void)vs; (void)ve; (void)tpf; (void)nf;             \
    (void)chunk; (void)radius;                                                          \
    __trap();
#define FA2_ENTRY_BODY(D)                                                               \
    (void)q; (void)k; (void)v; (void)out; (void)out_bf16; (void)out_is_bf16;            \
    (void)sq; (void)sk; (void)sl2;                                                      \
    __trap();
#define FA2C_ENTRY_BODY(D)                                                              \
    (void)q; (void)k; (void)v; (void)out; (void)out_bf16; (void)out_is_bf16;            \
    (void)sq; (void)sk; (void)sl2; (void)ft; (void)win; (void)sink;                     \
    __trap();
#endif

// Dynamic shared memory: 4 * 64 * D * 2 bytes (64 KB at D=128, opt-in).
extern "C" __global__ void __launch_bounds__(256, 1) flash_mma_fwd2_d128(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2
) {
    FA2_ENTRY_BODY(128)
}

extern "C" __global__ void __launch_bounds__(256, 2) flash_mma_fwd2_d64(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2
) {
    FA2_ENTRY_BODY(64)
}

// Block-causal (Wan self-forcing) variants: `ft` tokens per causal block,
// the last `win` blocks visible counting the query's own (0 = unlimited),
// the first `sink` blocks always visible. Same launch geometry and shared memory as the dense kernels.
extern "C" __global__ void __launch_bounds__(256, 1) flash_mma_fwd2_causal_d128(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2,
    int ft, int win, int sink
) {
    FA2C_ENTRY_BODY(128)
}

extern "C" __global__ void __launch_bounds__(256, 2) flash_mma_fwd2_causal_d64(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2,
    int ft, int win, int sink
) {
    FA2C_ENTRY_BODY(64)
}

// VDN-H3 window softmax (flash_mma_fwd2_window_body): video rows [vs, ve),
// `tpf` tokens per frame, `nf` frames, chunk-aligned window of `radius`
// chunks (`chunk` frames each; 0 = a centred frame window). Same launch
// geometry and shared memory as the dense / block-causal kernels.
extern "C" __global__ void __launch_bounds__(256, 1) flash_mma_fwd2_window_d128(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2,
    int vs, int ve, int tpf, int nf, int chunk, int radius
) {
    FA2W_ENTRY_BODY(128)
}

// VSA fine stage, TMA ring. Tiles are consumed in the order K_0 V_0 K_1 V_1
// ...; tile t lives in slot t % 3, each slot with its own mbarrier. While
// K_i is read, V_i and K_{i+1} are in flight; V_{i+1} is issued as soon as
// every warp is done with K_i, K_{i+2} as soon as every warp is done with
// V_i. 3 x 16 KB = 48 KB, so two CTAs fit an SM12x's 100 KB (the two-stage
// kernel's 64 KB fits one). Arithmetic is vsa_mma_attn_tma's exactly
// (ldmatrix.x4 feeds the same mma sequence).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void vsa2_issue(int tile_seq, int topk, const unsigned int* sel, int bh, long padded,
                                           unsigned int s_base, unsigned int mbar_base,
                                           const FvTensorMap* tk0, const FvTensorMap* tk1,
                                           const FvTensorMap* tv0, const FvTensorMap* tv1) {
    if (tile_seq >= 2 * topk) return;
    const int slot = tile_seq % 3;
    const unsigned int mb = mbar_base + 8u * slot;
    const int krow = bh * (int)padded + (int)sel[tile_seq >> 1] * MMA_TILE;
    mma_mbar_expect(mb, MMA_TILEB);
    if (tile_seq & 1) mma_tma_tile(s_base + slot * MMA_TILEB, tv0, tv1, mb, krow);
    else              mma_tma_tile(s_base + slot * MMA_TILEB, tk0, tk1, mb, krow);
}
#endif

extern "C" __global__ void __launch_bounds__(128, 2) vsa_mma_attn_tma2(
    const __grid_constant__ FvTensorMap tq0, const __grid_constant__ FvTensorMap tq1,
    const __grid_constant__ FvTensorMap tk0, const __grid_constant__ FvTensorMap tk1,
    const __grid_constant__ FvTensorMap tv0, const __grid_constant__ FvTensorMap tv1,
    const unsigned int* __restrict__ selected, const int* __restrict__ block_sizes,
    float* __restrict__ out, int num_tiles, int topk, float scale_log2, int q_base
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) unsigned char mma_smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int qtile = q_base + (int)blockIdx.x, bh = blockIdx.y;
    if (qtile >= num_tiles) return;
    const long padded = (long)num_tiles * MMA_TILE;
    const unsigned int* sel = selected + ((long)bh * num_tiles + qtile) * topk;
    const int qrow = bh * (int)padded + qtile * MMA_TILE;

    const unsigned int s_base = mma_smem_u32(mma_smem);
    const unsigned int mbar_base = s_base + 3 * MMA_TILEB;
    unsigned int phase[3] = { 0, 0, 0 };

    if (tid == 0) {
        mma_mbar_init(mbar_base, 1);
        mma_mbar_init(mbar_base + 8, 1);
        mma_mbar_init(mbar_base + 16, 1);
    }
    __syncthreads();

    // Q through slot 0 (one extra fill of slot 0's barrier).
    if (tid == 0) {
        mma_mbar_expect(mbar_base, MMA_TILEB);
        mma_tma_tile(s_base, &tq0, &tq1, mbar_base, qrow);
    }
    mma_mbar_wait(mbar_base, phase[0]);
    phase[0] ^= 1u;
    unsigned int qf[8][4];
    #pragma unroll
    for (int kc = 0; kc < 8; kc++) {
        int row = warp * 16 + (lane & 15), col = kc * 16 + (lane >> 4) * 8;
        mma_ldm_x4(s_base + mma_swz_tma(row, col), qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3]);
    }
    __syncthreads();
    if (tid == 0) {
        vsa2_issue(0, topk, sel, bh, padded, s_base, mbar_base, &tk0, &tk1, &tv0, &tv1);
        vsa2_issue(1, topk, sel, bh, padded, s_base, mbar_base, &tk0, &tk1, &tv0, &tv1);
        vsa2_issue(2, topk, sel, bh, padded, s_base, mbar_base, &tk0, &tk1, &tv0, &tv1);
    }

    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    const float NEG = __int_as_float(0xff800000);
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;

    #pragma unroll 1
    for (int i = 0; i < topk; i++) {
        const int ks = (2 * i) % 3, vs = (2 * i + 1) % 3;
        const unsigned int sK = s_base + ks * MMA_TILEB, sV = s_base + vs * MMA_TILEB;
        mma_mbar_wait(mbar_base + 8u * ks, phase[ks]);
        phase[ks] ^= 1u;

        float s[8][4];
        #pragma unroll
        for (int n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
        #pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            #pragma unroll
            for (int n = 0; n < 8; n += 2) {
                unsigned int b[4];
                mma_ldm_x4(sK + mma_swz_tma(n * 8 + (lane & 7) + ((lane >> 4) << 3), kc * 16 + ((lane >> 3) & 1) * 8),
                           b[0], b[1], b[2], b[3]);
                mma_bf16(s[n], qf[kc], b);
                mma_bf16(s[n + 1], qf[kc], b + 2);
            }
        }
        __syncthreads();   // every warp is done with K_i: its slot takes V_{i+1}
        if (tid == 0) vsa2_issue(2 * i + 3, topk, sel, bh, padded, s_base, mbar_base, &tk0, &tk1, &tv0, &tv1);

        const int valid = block_sizes[sel[i]];
        float rmax0 = NEG, rmax1 = NEG;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int c0 = n * 8 + t * 2;
            if (c0 >= valid)     { s[n][0] = NEG; s[n][2] = NEG; }
            if (c0 + 1 >= valid) { s[n][1] = NEG; s[n][3] = NEG; }
            rmax0 = fmaxf(rmax0, fmaxf(s[n][0], s[n][1]));
            rmax1 = fmaxf(rmax1, fmaxf(s[n][2], s[n][3]));
        }
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float ms0 = (mn0 == NEG) ? 0.f : mn0 * scale_log2;
        const float ms1 = (mn1 == NEG) ? 0.f : mn1 * scale_log2;
        const float a0 = exp2f(m0 * scale_log2 - ms0), a1 = exp2f(m1 * scale_log2 - ms1);
        m0 = mn0; m1 = mn1;
        float rs0 = 0.f, rs1 = 0.f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            s[n][0] = exp2f(s[n][0] * scale_log2 - ms0);
            s[n][1] = exp2f(s[n][1] * scale_log2 - ms0);
            s[n][2] = exp2f(s[n][2] * scale_log2 - ms1);
            s[n][3] = exp2f(s[n][3] * scale_log2 - ms1);
            rs0 += s[n][0] + s[n][1];
            rs1 += s[n][2] + s[n][3];
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * a0 + rs0;
        l1 = l1 * a1 + rs1;
        #pragma unroll
        for (int n = 0; n < 16; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }

        mma_mbar_wait(mbar_base + 8u * vs, phase[vs]);
        phase[vs] ^= 1u;
        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            unsigned int pa[4];
            pa[0] = mma_pack_bf16(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = mma_pack_bf16(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = mma_pack_bf16(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = mma_pack_bf16(s[2 * kc + 1][2], s[2 * kc + 1][3]);
            #pragma unroll
            for (int n = 0; n < 16; n += 2) {
                unsigned int b[4];
                fa_ldm_x4_trans(sV + mma_swz_tma(kc * 16 + (lane & 15), n * 8 + ((lane >> 4) << 3)), b[0], b[1], b[2], b[3]);
                mma_bf16(o[n], pa, b);
                mma_bf16(o[n + 1], pa, b + 2);
            }
        }
        __syncthreads();   // every warp is done with V_i: its slot takes K_{i+2}
        if (tid == 0) vsa2_issue(2 * i + 4, topk, sel, bh, padded, s_base, mbar_base, &tk0, &tk1, &tv0, &tv1);
    }

    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    float* ob = out + (bh * padded + (long)qtile * MMA_TILE + warp * 16) * MMA_DIM;
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        int col = n * 8 + t * 2;
        float2 r0 = make_float2(o[n][0] * inv0, o[n][1] * inv0);
        float2 r1 = make_float2(o[n][2] * inv1, o[n][3] * inv1);
        *reinterpret_cast<float2*>(ob + (long)g * MMA_DIM + col) = r0;
        *reinterpret_cast<float2*>(ob + (long)(g + 8) * MMA_DIM + col) = r1;
    }
#else
    (void)tq0; (void)tq1; (void)tk0; (void)tk1; (void)tv0; (void)tv1;
    (void)selected; (void)block_sizes; (void)out;
    (void)num_tiles; (void)topk; (void)scale_log2; (void)q_base;
    __trap();
#endif
}

// ---- Sol-Attn, warp-specialised, with KV splits -----------------------------
//
// Same algorithm and arithmetic as sol_mma_fwd (itself the CuTe sm120
// mainloop, sol_attn/sm120/mainloop.py:334-647): per 64-block route group,
// S = Q Kc^T, column means vs the threshold, local window and sink,
// ascending ballot compaction, approximate term, then one flash step per
// exact block. What changes is who waits for what:
//
//   warp 4 (producer): every cp.async, in the reference's TMA issue order
//     (Q; per group Kc unless prefetched, Vc; first exact K after routing;
//     V_e after the approximate PV; K_{e+1} or the next group's Kc after
//     QK_e; V_{e+1} after PV_e). Each fill completes a "full" mbarrier via
//     cp.async.mbarrier.arrive.noinc (32 producer arrivals).
//   warps 0-3 (MMA): wait "full", compute, arrive on "empty" (one lane per
//     warp) as soon as they are done reading, never waiting for each other
//     except at the two route barriers per group (colsum reduce: 128
//     threads; route decision published to the producer: 160 threads).
//
// KV splits (sol_attn/sm90/mainloop.py:1734-1747): split z of `splits` owns
// route groups [z*gps, min(G, (z+1)*gps)), gps = ceil(G / splits), and writes
// a normalised f32 partial O and its natural-log LSE; sol_split_combine
// merges them (sol_attn/sm90/split_combine.py:451-471). splits == 1 writes
// the output directly and is bit-identical to sol_mma_fwd.
#define SOL2_SMEM (2 * MMA_TILEB + (4 * 64 + 64) * 4 + 64 * 4 + 16 + 4 * 8)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void sol2_mbar_arrive(unsigned int mbar) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" :: "r"(mbar) : "memory");
}
__device__ __forceinline__ void sol2_cp_arrive(unsigned int mbar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(mbar) : "memory");
}
__device__ __forceinline__ void sol2_named_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory");
}
// Producer warp: one 64 x 128 tile (rows >= rows_valid zero-filled), then
// this lane's arrive-on-completion on `mbar`.
__device__ __forceinline__ void sol2_fill(unsigned int smem, const unsigned short* g, int rows_valid, int lane, unsigned int mbar) {
    #pragma unroll 8
    for (int i = 0; i < 32; i++) {
        const int chunk = lane + i * 32;
        const int row = chunk >> 4, c = chunk & 15;
        const int ok = row < rows_valid;
        mma_cp_async16_zfill(smem + mma_swz(row, c * 8),
                             ok ? (const void*)(g + row * MMA_DIM + c * 8) : (const void*)g, ok);
    }
    sol2_cp_arrive(mbar);
}
#endif

extern "C" __global__ void __maxnreg__(200) sol_mma_fwd2(
    const unsigned short* __restrict__ qb, const unsigned short* __restrict__ kb,
    const unsigned short* __restrict__ vb,
    const unsigned short* __restrict__ kc, const unsigned short* __restrict__ vc,
    const float* __restrict__ thr,
    float* __restrict__ out, unsigned short* __restrict__ out_bf16, int out_is_bf16,
    float* __restrict__ lse, int has_lse,
    unsigned int* __restrict__ route_dbg, int has_dbg,
    int T, int NT, int sink_lo, int sink_hi, float sl2,
    int splits, float* __restrict__ part_o, float* __restrict__ part_lse
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __shared__ __align__(128) unsigned char smem[SOL2_SMEM];
    float* colsum = reinterpret_cast<float*>(smem + 2 * MMA_TILEB);   // [4][64]
    float* colmask = colsum + 4 * 64;                                // [64]
    int* eidx = reinterpret_cast<int*>(colmask + 64);                // [64]
    int* meta = eidx + 64;                                           // [4]
    const unsigned int sK = mma_smem_u32(smem), sV = sK + MMA_TILEB;
    const unsigned int mbar = mma_smem_u32(meta + 4);                // 8-aligned: FK FV EK EV
    const unsigned int FK = mbar, FV = mbar + 8, EK = mbar + 16, EV = mbar + 24;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int qt = blockIdx.x;
    const long bh = blockIdx.y;
    const int split = blockIdx.z;
    if (qt >= NT) return;
    const int q0 = qt * 64, qlen = min(64, T - q0);
    const int G = (NT + 63) >> 6;
    const int gps = (G + splits - 1) / splits;
    const int g_begin = min(G, split * gps), g_end = min(G, g_begin + gps);
    const unsigned short* Kh = kb + bh * (long)T * MMA_DIM;
    const unsigned short* Vh = vb + bh * (long)T * MMA_DIM;
    const unsigned short* KCh = kc + bh * (long)NT * MMA_DIM;
    const unsigned short* VCh = vc + bh * (long)NT * MMA_DIM;
    const float NEG = __int_as_float(0xff800000);

    if (tid == 0) {
        mma_mbar_init(FK, 32);
        mma_mbar_init(FV, 32);
        mma_mbar_init(EK, 4);
        mma_mbar_init(EV, 4);
    }
    __syncthreads();

    if (warp == 4) {
        // ---------------- producer ----------------
        int kfill = 0, vfill = 0;   // fills issued per slot
        auto k_acquire = [&]() { if (kfill > 0) mma_mbar_wait(EK, (unsigned)((kfill - 1) & 1)); kfill++; };
        auto v_acquire = [&]() { if (vfill > 0) mma_mbar_wait(EV, (unsigned)((vfill - 1) & 1)); vfill++; };
        k_acquire();
        sol2_fill(sK, qb + (bh * (long)T + q0) * MMA_DIM, qlen, lane, FK);
        bool kc_pending = false;
        for (int gi = g_begin; gi < g_end; gi++) {
            const int gs = gi * 64, vb_cnt = min(64, NT - gs);
            if (!kc_pending) {
                k_acquire();
                sol2_fill(sK, KCh + (long)gs * MMA_DIM, vb_cnt, lane, FK);
            }
            kc_pending = false;
            v_acquire();
            sol2_fill(sV, VCh + (long)gs * MMA_DIM, vb_cnt, lane, FV);
            sol2_named_sync(1, 160);   // route decision published
            const int ne = meta[0];
            const bool more_groups = gi + 1 < g_end;
            if (ne > 0) {
                const int j0 = eidx[0];
                k_acquire();
                sol2_fill(sK, Kh + (long)j0 * 64 * MMA_DIM, min(64, T - 64 * j0), lane, FK);
                v_acquire();   // approximate PV done
                sol2_fill(sV, Vh + (long)j0 * 64 * MMA_DIM, min(64, T - 64 * j0), lane, FV);
                for (int e = 0; e < ne; e++) {
                    if (e + 1 < ne) {
                        const int jn = eidx[e + 1];
                        k_acquire();   // QK_e done
                        sol2_fill(sK, Kh + (long)jn * 64 * MMA_DIM, min(64, T - 64 * jn), lane, FK);
                        v_acquire();   // PV_e done
                        sol2_fill(sV, Vh + (long)jn * 64 * MMA_DIM, min(64, T - 64 * jn), lane, FV);
                    } else if (more_groups) {
                        k_acquire();   // last QK done: next group's Kc
                        sol2_fill(sK, KCh + (long)(gs + 64) * MMA_DIM, min(64, NT - gs - 64), lane, FK);
                        kc_pending = true;
                    }
                }
            } else if (more_groups) {
                k_acquire();   // QKc done
                sol2_fill(sK, KCh + (long)(gs + 64) * MMA_DIM, min(64, NT - gs - 64), lane, FK);
                kc_pending = true;
            }
        }
        return;
    }

    // ---------------- MMA warps ----------------
    const int g = lane >> 2, t = lane & 3;
    const float th = thr[bh * NT + qt];
    int kuse = 0, vuse = 0;
    auto k_wait = [&]() { mma_mbar_wait(FK, (unsigned)(kuse & 1)); kuse++; };
    auto v_wait = [&]() { mma_mbar_wait(FV, (unsigned)(vuse & 1)); vuse++; };
    auto k_release = [&]() { __syncwarp(); if (lane == 0) sol2_mbar_arrive(EK); };
    auto v_release = [&]() { __syncwarp(); if (lane == 0) sol2_mbar_arrive(EV); };

    k_wait();
    unsigned int qf[8][4];
    #pragma unroll
    for (int k8 = 0; k8 < 8; k8++) {
        int row = warp * 16 + (lane & 15), col = k8 * 16 + (lane >> 4) * 8;
        mma_ldm_x4(sK + mma_swz(row, col), qf[k8][0], qf[k8][1], qf[k8][2], qf[k8][3]);
    }
    k_release();

    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool ok0 = row0 < qlen, ok1 = row1 < qlen;
    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    float s[8][4];
    unsigned int pa[4][4];

    #pragma unroll 1
    for (int gi = g_begin; gi < g_end; gi++) {
        const int gs = gi * 64, vb_cnt = min(64, NT - gs);
        k_wait();
        fa_qk<128>(sK, qf, s, lane);
        k_release();

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            float p0 = (ok0 ? s[n][0] : 0.f) + (ok1 ? s[n][2] : 0.f);
            float p1 = (ok0 ? s[n][1] : 0.f) + (ok1 ? s[n][3] : 0.f);
            #pragma unroll
            for (int off = 4; off <= 16; off <<= 1) {
                p0 += __shfl_xor_sync(0xffffffffu, p0, off);
                p1 += __shfl_xor_sync(0xffffffffu, p1, off);
            }
            if (g == 0) {
                colsum[warp * 64 + n * 8 + 2 * t] = p0;
                colsum[warp * 64 + n * 8 + 2 * t + 1] = p1;
            }
        }
        sol2_named_sync(2, 128);

        if (warp == 0) {
            int base = 0;
            #pragma unroll
            for (int word = 0; word < 2; word++) {
                const int off = word * 32 + lane, kbk = gs + off;
                const bool valid = off < vb_cnt;
                bool ex = false;
                if (valid) {
                    const float cs = ((colsum[off] + colsum[64 + off]) + colsum[128 + off]) + colsum[192 + off];
                    const float cm = __fdiv_rn(__fmul_rn(cs, sl2), (float)qlen);
                    const int dist = qt > kbk ? qt - kbk : kbk - qt;
                    ex = (cm > th) || (dist <= 1) || (kbk >= sink_lo && kbk < sink_hi);
                }
                const unsigned int bal = __ballot_sync(0xffffffffu, ex);
                colmask[off] = (ex || !valid) ? NEG : 0.f;
                if (ex) eidx[base + __popc(bal & ((1u << lane) - 1u))] = kbk;
                base += __popc(bal);
                if (has_dbg && lane == 0) route_dbg[(bh * NT + qt) * 2L * G + 2 * gi + word] = bal;
            }
            if (lane == 0) meta[0] = base;
        }
        sol2_named_sync(1, 160);
        const int ne = meta[0];
        const bool has_approx = ne < vb_cnt;

        v_wait();   // Vc
        if (has_approx) {
            float rmax0 = NEG, rmax1 = NEG;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                const int c0 = n * 8 + 2 * t;
                const float mk0 = colmask[c0], mk1 = colmask[c0 + 1];
                s[n][0] = ok0 ? s[n][0] + mk0 : NEG;
                s[n][1] = ok0 ? s[n][1] + mk1 : NEG;
                s[n][2] = ok1 ? s[n][2] + mk0 : NEG;
                s[n][3] = ok1 ? s[n][3] + mk1 : NEG;
                rmax0 = fmaxf(rmax0, fmaxf(s[n][0], s[n][1]));
                rmax1 = fmaxf(rmax1, fmaxf(s[n][2], s[n][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));
            const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
            const float sf0 = (mn0 == NEG) ? 0.f : mn0, sf1 = (mn1 == NEG) ? 0.f : mn1;
            const float a0 = exp2f((m0 - sf0) * sl2), a1 = exp2f((m1 - sf1) * sl2);
            const float ms0 = sf0 * sl2, ms1 = sf1 * sl2;
            m0 = mn0; m1 = mn1;
            float ls0 = 0.f, ls1 = 0.f;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                const int c0 = n * 8 + 2 * t;
                const float w0 = (float)max(0, min(64, T - 64 * (gs + c0)));
                const float w1 = (float)max(0, min(64, T - 64 * (gs + c0 + 1)));
                s[n][0] = exp2f(s[n][0] * sl2 - ms0);
                s[n][1] = exp2f(s[n][1] * sl2 - ms0);
                s[n][2] = exp2f(s[n][2] * sl2 - ms1);
                s[n][3] = exp2f(s[n][3] * sl2 - ms1);
                ls0 += s[n][0] * w0 + s[n][1] * w1;
                ls1 += s[n][2] * w0 + s[n][3] * w1;
            }
            l0 = l0 * a0 + ls0;
            l1 = l1 * a1 + ls1;
            #pragma unroll
            for (int n = 0; n < 16; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
            #pragma unroll
            for (int k4 = 0; k4 < 4; k4++) {
                pa[k4][0] = mma_pack_bf16_rn(s[2 * k4][0], s[2 * k4][1]);
                pa[k4][1] = mma_pack_bf16_rn(s[2 * k4][2], s[2 * k4][3]);
                pa[k4][2] = mma_pack_bf16_rn(s[2 * k4 + 1][0], s[2 * k4 + 1][1]);
                pa[k4][3] = mma_pack_bf16_rn(s[2 * k4 + 1][2], s[2 * k4 + 1][3]);
            }
            fa_pv<128>(sV, pa, o, lane);
        }
        v_release();

        #pragma unroll 1
        for (int e = 0; e < ne; e++) {
            const int j = eidx[e];
            const int Lj = min(64, T - 64 * j);
            k_wait();
            fa_qk<128>(sK, qf, s, lane);
            k_release();
            float rmax0 = NEG, rmax1 = NEG;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                const int c0 = n * 8 + 2 * t;
                if (!ok0 || c0 >= Lj)     s[n][0] = NEG;
                if (!ok0 || c0 + 1 >= Lj) s[n][1] = NEG;
                if (!ok1 || c0 >= Lj)     s[n][2] = NEG;
                if (!ok1 || c0 + 1 >= Lj) s[n][3] = NEG;
                rmax0 = fmaxf(rmax0, fmaxf(s[n][0], s[n][1]));
                rmax1 = fmaxf(rmax1, fmaxf(s[n][2], s[n][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));
            const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
            const float sf0 = (mn0 == NEG) ? 0.f : mn0, sf1 = (mn1 == NEG) ? 0.f : mn1;
            const float a0 = exp2f((m0 - sf0) * sl2), a1 = exp2f((m1 - sf1) * sl2);
            const float ms0 = sf0 * sl2, ms1 = sf1 * sl2;
            m0 = mn0; m1 = mn1;
            float ls0 = 0.f, ls1 = 0.f;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                s[n][0] = exp2f(s[n][0] * sl2 - ms0);
                s[n][1] = exp2f(s[n][1] * sl2 - ms0);
                s[n][2] = exp2f(s[n][2] * sl2 - ms1);
                s[n][3] = exp2f(s[n][3] * sl2 - ms1);
                ls0 += s[n][0] + s[n][1];
                ls1 += s[n][2] + s[n][3];
            }
            l0 = l0 * a0 + ls0;
            l1 = l1 * a1 + ls1;
            #pragma unroll
            for (int n = 0; n < 16; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
            #pragma unroll
            for (int k4 = 0; k4 < 4; k4++) {
                pa[k4][0] = mma_pack_bf16_rn(s[2 * k4][0], s[2 * k4][1]);
                pa[k4][1] = mma_pack_bf16_rn(s[2 * k4][2], s[2 * k4][3]);
                pa[k4][2] = mma_pack_bf16_rn(s[2 * k4 + 1][0], s[2 * k4 + 1][1]);
                pa[k4][3] = mma_pack_bf16_rn(s[2 * k4 + 1][2], s[2 * k4 + 1][3]);
            }
            v_wait();
            fa_pv<128>(sV, pa, o, lane);
            v_release();
        }
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const bool bad0 = (l0 == 0.f) || ((__float_as_uint(l0) & 0x7fffffffu) > 0x7f800000u);
    const bool bad1 = (l1 == 0.f) || ((__float_as_uint(l1) & 0x7fffffffu) > 0x7f800000u);
    const float inv0 = bad0 ? 1.f : 1.f / l0, inv1 = bad1 ? 1.f : 1.f / l1;
    const long r0 = bh * (long)T + q0 + row0, r1 = r0 + 8;
    const float lse0 = bad0 ? NEG : (m0 * sl2 + log2f(l0)) * SOL_LN2;
    const float lse1 = bad1 ? NEG : (m1 * sl2 + log2f(l1)) * SOL_LN2;
    if (splits > 1) {
        const long rows = (long)gridDim.y * T;
        float* po = part_o + (long)split * rows * MMA_DIM;
        if (t == 0) {
            if (ok0) part_lse[(long)split * rows + r0] = lse0;
            if (ok1) part_lse[(long)split * rows + r1] = lse1;
        }
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            const int col = n * 8 + 2 * t;
            if (ok0) *reinterpret_cast<float2*>(po + r0 * MMA_DIM + col) = make_float2(o[n][0] * inv0, o[n][1] * inv0);
            if (ok1) *reinterpret_cast<float2*>(po + r1 * MMA_DIM + col) = make_float2(o[n][2] * inv1, o[n][3] * inv1);
        }
        return;
    }
    if (has_lse && t == 0) {
        if (ok0) lse[r0] = lse0;
        if (ok1) lse[r1] = lse1;
    }
    if (out_is_bf16) {
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            const int col = n * 8 + 2 * t;
            if (ok0) *reinterpret_cast<unsigned int*>(out_bf16 + r0 * MMA_DIM + col) = mma_pack_bf16_rn(o[n][0] * inv0, o[n][1] * inv0);
            if (ok1) *reinterpret_cast<unsigned int*>(out_bf16 + r1 * MMA_DIM + col) = mma_pack_bf16_rn(o[n][2] * inv1, o[n][3] * inv1);
        }
    } else {
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            const int col = n * 8 + 2 * t;
            if (ok0) *reinterpret_cast<float2*>(out + r0 * MMA_DIM + col) = make_float2(o[n][0] * inv0, o[n][1] * inv0);
            if (ok1) *reinterpret_cast<float2*>(out + r1 * MMA_DIM + col) = make_float2(o[n][2] * inv1, o[n][3] * inv1);
        }
    }
#else
    (void)qb; (void)kb; (void)vb; (void)kc; (void)vc; (void)thr; (void)out; (void)out_bf16;
    (void)out_is_bf16; (void)lse; (void)has_lse; (void)route_dbg; (void)has_dbg;
    (void)T; (void)NT; (void)sink_lo; (void)sink_hi; (void)sl2;
    (void)splits; (void)part_o; (void)part_lse;
    __trap();
#endif
}

// Merge KV-split partials: part_o `[splits, rows, 128]` normalised f32,
// part_lse `[splits, rows]` natural-log LSE (-inf = no mass). One warp per
// row, four channels per lane. out = sum_s w_s o_s / sum_s w_s with
// w_s = exp(lse_s - max lse); lse = max + log(sum w). A row with no mass in
// any split is 0 with lse -inf, as sol_mma_fwd writes it.
extern "C" __global__ void sol_split_combine(
    const float* __restrict__ part_o, const float* __restrict__ part_lse, int splits, long rows,
    float* __restrict__ out, unsigned short* __restrict__ out_bf16, int out_is_bf16,
    float* __restrict__ lse, int has_lse
) {
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float NEG = __int_as_float(0xff800000);
    float mx = NEG;
    for (int s = 0; s < splits; s++) mx = fmaxf(mx, part_lse[(long)s * rows + row]);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float wsum = 0.f;
    if (mx != NEG) {
        for (int s = 0; s < splits; s++) {
            const float ls = part_lse[(long)s * rows + row];
            const float w = (ls == NEG) ? 0.f : expf(ls - mx);
            wsum += w;
            const float4 v = *reinterpret_cast<const float4*>(part_o + ((long)s * rows + row) * 128 + lane * 4);
            acc[0] += w * v.x; acc[1] += w * v.y; acc[2] += w * v.z; acc[3] += w * v.w;
        }
    }
    const float inv = wsum > 0.f ? 1.f / wsum : 0.f;
    const long o = row * 128 + lane * 4;
    if (out_is_bf16) {
        *reinterpret_cast<unsigned int*>(out_bf16 + o) =
            (unsigned int)fv_bf16_rne(acc[0] * inv) | ((unsigned int)fv_bf16_rne(acc[1] * inv) << 16);
        *reinterpret_cast<unsigned int*>(out_bf16 + o + 2) =
            (unsigned int)fv_bf16_rne(acc[2] * inv) | ((unsigned int)fv_bf16_rne(acc[3] * inv) << 16);
    } else {
        *reinterpret_cast<float4*>(out + o) = make_float4(acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv);
    }
    if (has_lse && lane == 0) lse[row] = (wsum > 0.f) ? mx + logf(wsum) : NEG;
}
// ==== end region: attn2 ====

