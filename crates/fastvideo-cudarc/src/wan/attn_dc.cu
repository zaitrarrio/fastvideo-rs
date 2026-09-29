// Datacenter attention kernels: sm_100a (B200 / GB200: tcgen05 MMA into
// tensor memory, TMA, warp-specialised) and sm_90a (H100 / H200: wgmma, TMA,
// warp-specialised producer / consumer warpgroups).
//
// Compiled apart from kernels.cu: build.rs builds this file for sm_90a and
// sm_100a only (tcgen05, wgmma and setmaxnreg exist only on the
// arch-specific targets, whose cubins run on exactly that SM), and the
// runtime loads it only on a 9.0 / 10.x device. Every other GPU (Ampere,
// Ada, consumer Blackwell sm_120) keeps the mma.sync kernels of kernels.cu,
// which also stay the oracle: `fv-gpucheck kernels` group `attn_dc` holds
// each kernel here to flash_mma_fwd2 at the same inputs.
//
// Dense SDPA, head dim 128, bf16 Q/K/V laid out [bh, s, 128]; the TMA maps
// are 3-D (d, s, bh) with a 64 x 128 x 1 box and 128-byte swizzle, so a
// 128-row x 128-column tile is two 16 KB panels (d 0..63, d 64..127) and
// rows past the end of a head are zero-filled by the TMA unit. The softmax
// is flash_mma_fwd2's per-row arithmetic (base-2 domain, sl2 = scale *
// log2 e folded into one FMA, ex2.approx.ftz, bf16 P rounded to nearest
// even, f32 accumulation, 1/l at the end) with two schedule differences: the
// running max advances per 128-key tile instead of per 64 keys, and O is
// rescaled only when a row's max grows (the old kernel multiplies by an
// exp2 of the max's rounding error, 1 +- 2^-24, every tile). Both kernels
// therefore agree with flash_mma_fwd2 to bf16-P rounding, not bit for bit.
//
//   fa_dc100_fwd_d128   CTA = 2 x 128 queries, 12 warps: warps 0-3 / 4-7 own
//                       query tiles 0 / 1 (one TMEM lane = one query row per
//                       thread), warp 8 issues TMA, warp 9 (one thread)
//                       issues tcgen05.mma (10-11 idle). TMEM (512 columns): S0 | S1 |
//                       O0 | O1, 128 each; P (bf16) is written over the upper
//                       half of its S and read by the P.V MMA straight from
//                       TMEM. The MMA thread alternates tiles (FA4 /
//                       CUTLASS sm100 FMHA order), so one tile's softmax
//                       overlaps the other tile's QK and PV.
//   fa_dc90_fwd_d128    CTA = 128 queries, 3 warpgroups: warpgroup 0 issues
//                       TMA (24 registers), warpgroups 1-2 own 64 query rows
//                       each (240 registers): S = Q K^T and O += P V by
//                       wgmma (P from registers), with QK of tile j and PV of
//                       tile j-1 in flight while the softmax of tile j runs
//                       (FA3's intra-warpgroup overlap).
//
// K/V stream through a ring of NS 32 KB slots in the order K0 V0 K1 V1 ...
// (slot = index % NS), each slot with a full (TMA transaction) and an empty
// (consumer release) mbarrier.

struct __align__(128) DcTensorMap {
    unsigned long long v[16];
};

#define DC_D 128
#define DC_BN 128              // keys per K/V tile
#define DC_PANELB 16384        // 128 rows x 64 bf16: one swizzled TMA box
#define DC_TILEB 32768         // 128 x 128 bf16 = two panels
// Dynamic shared memory: 1 KB alignment slack (swizzled tiles need 1 KB
// alignment), the Q tile(s), the K/V ring, mbarriers and the TMEM address.
#define DC100_SMEM(NS) (1024 + 2 * DC_TILEB + (NS) * DC_TILEB + 256)
#define DC90_SMEM(NS) (1024 + DC_TILEB + (NS) * DC_TILEB + 256)

#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
#define DC_SM100 1
#endif
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define DC_SM90 1
#endif

#if defined(DC_SM100) || defined(DC_SM90)
#define DC_DEV __device__ __forceinline__

DC_DEV unsigned int dc_smem_u32(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}
DC_DEV void dc_mbar_init(unsigned int bar, unsigned int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count) : "memory");
}
DC_DEV void dc_mbar_expect(unsigned int bar, unsigned int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
}
DC_DEV void dc_mbar_arrive(unsigned int bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}
DC_DEV unsigned int dc_mbar_try(unsigned int bar, unsigned int parity) {
    unsigned int ok;
    asm volatile(
        "{\n.reg .pred P1;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.u32 %0, 1, 0, P1;\n}\n"
        : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}
// DC_DEBUG_HANG=<cycles>: a wait that outlives it reports and traps instead
// of hanging the GPU (the test harness builds with it).
DC_DEV void dc_mbar_wait(unsigned int bar, unsigned int parity) {
#ifdef DC_DEBUG_HANG
    const long long t0 = clock64();
    while (!dc_mbar_try(bar, parity)) {
        if (clock64() - t0 > (long long)(DC_DEBUG_HANG)) {
            printf("dc hang: block (%d,%d) thread %d bar +%u parity %u\n", (int)blockIdx.x, (int)blockIdx.y,
                   (int)threadIdx.x, bar & 1023u, parity);
            __trap();
        }
    }
#else
    while (!dc_mbar_try(bar, parity)) {
    }
#endif
}
DC_DEV void dc_tma_prefetch(const DcTensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" :: "l"(map) : "memory");
}
DC_DEV void dc_tma_load3(unsigned int dst, const DcTensorMap* map, unsigned int bar, int c0, int c1, int c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];\n"
        :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
// A 128-row x 128-column bf16 tile (rows row..row+127 of head bh): two
// 64-column swizzled panels, 16 KB apart. The caller posts the 32 KB expect.
DC_DEV void dc_tma_tile(unsigned int dst, const DcTensorMap* map, unsigned int bar, int row, int bh) {
    dc_tma_load3(dst, map, bar, 0, row, bh);
    dc_tma_load3(dst + DC_PANELB, map, bar, 64, row, bh);
}
DC_DEV float dc_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
// Two f32 -> bf16x2 (lo in the low half), round to nearest even.
DC_DEV unsigned int dc_pack_bf16(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DC_DEV void dc_store_row32(float* out, unsigned short* out_bf16, int out_is_bf16, long off,
                           const unsigned int (&o)[32], float inv) {
    if (out_is_bf16) {
        uint4* dst = reinterpret_cast<uint4*>(out_bf16 + off);
#pragma unroll
        for (int v = 0; v < 4; v++) {
            const unsigned int* p = o + 8 * v;
            dst[v] = make_uint4(dc_pack_bf16(__uint_as_float(p[0]) * inv, __uint_as_float(p[1]) * inv),
                                dc_pack_bf16(__uint_as_float(p[2]) * inv, __uint_as_float(p[3]) * inv),
                                dc_pack_bf16(__uint_as_float(p[4]) * inv, __uint_as_float(p[5]) * inv),
                                dc_pack_bf16(__uint_as_float(p[6]) * inv, __uint_as_float(p[7]) * inv));
        }
    } else {
        float4* dst = reinterpret_cast<float4*>(out + off);
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const unsigned int* p = o + 4 * v;
            dst[v] = make_float4(__uint_as_float(p[0]) * inv, __uint_as_float(p[1]) * inv,
                                 __uint_as_float(p[2]) * inv, __uint_as_float(p[3]) * inv);
        }
    }
}
#endif

// ---------------------------------------------------------------- sm_100a
#if defined(DC_SM100)
DC_DEV void dc_tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
DC_DEV void dc_tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }

// UMMA shared-memory descriptor (Blackwell): start >> 4, leading / stride
// byte offsets >> 4, version 1 at bit 46, layout SWIZZLE_128B (2) at 61.
// K-major SW128 operand: rows 128 B apart, 8-row groups SBO = 1024 B apart
// (LBO unused). MN-major SW128 operand: 64-element MN chunks LBO apart,
// 8-row K groups SBO = 1024 B apart.
DC_DEV unsigned long long dc_desc100(unsigned int addr, unsigned int lbo, unsigned int sbo) {
    return (unsigned long long)((addr & 0x3FFFFu) >> 4)
         | ((unsigned long long)((lbo >> 4) & 0x3FFFu) << 16)
         | ((unsigned long long)((sbo >> 4) & 0x3FFFu) << 32)
         | (1ull << 46)
         | (2ull << 61);
}
// Instruction descriptor, kind::f16: D f32 (bit 4), A and B bf16 (bits 7,
// 10), K-major A, B major at bit 16, N >> 3 at bit 17, M >> 4 at bit 24.
#define DC_IDESC(M, N, BMN) ((1u << 4) | (1u << 7) | (1u << 10) | ((unsigned int)(BMN) << 16) \
                             | (((unsigned int)(N) >> 3) << 17) | (((unsigned int)(M) >> 4) << 24))

DC_DEV void dc_umma_ss(unsigned int d, unsigned long long a, unsigned long long b, unsigned int idesc, unsigned int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
DC_DEV void dc_umma_ts(unsigned int d, unsigned int a_tmem, unsigned long long b, unsigned int idesc, unsigned int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(d), "r"(a_tmem), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
// Arrive on `bar` once every tcgen05 op this thread issued so far is done.
DC_DEV void dc_umma_commit(unsigned int bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" :: "r"(bar) : "memory");
}

// 32 consecutive TMEM columns of this thread's lane -> r[0..31]; waits for
// the load, so the registers are valid when this returns.
DC_DEV void dc_tmem_ld32(unsigned int taddr, unsigned int (&r)[32]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr) : "memory");
}

// 64 consecutive TMEM columns (two x32 loads, one wait).
DC_DEV void dc_tmem_ld64(unsigned int taddr, unsigned int (&r)[64]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%64];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%65];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
        : "r"(taddr), "r"(taddr + 32u) : "memory");
}

// r[0..31] -> 32 consecutive TMEM columns of this thread's lane; waits for
// the store before returning (the registers may be reused at once).
DC_DEV void dc_tmem_st32(unsigned int taddr, const unsigned int (&r)[32]) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n"
        "tcgen05.wait::st.sync.aligned;\n"
        :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]) : "memory");
}

// S (TMEM, 128 x 128 f32) = Q (128 x 128, K-major) . K^T (128 keys, K-major).
DC_DEV void dc100_qk(unsigned int tS, unsigned int sQ, unsigned int sK) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        const unsigned int off = (unsigned int)(kk >> 2) * DC_PANELB + (unsigned int)(kk & 3) * 32u;
        dc_umma_ss(tS, dc_desc100(sQ + off, 16, 1024), dc_desc100(sK + off, 16, 1024),
                   DC_IDESC(128, 128, 0), kk > 0);
    }
}
// O (TMEM, 128 x 128 f32) (+)= P (TMEM bf16, 128 x 128 keys) . V (128 keys x
// 128, MN-major: the two 64-column panels are LBO = 16 KB apart). SPLIT
// issues each K step as two N = 64 MMAs, one per panel.
template <bool SPLIT>
DC_DEV void dc100_pv(unsigned int tO, unsigned int tP, unsigned int sV, bool acc) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        const unsigned int a = tP + 8u * kk;          // 16 keys = 8 packed columns
        const unsigned int v = sV + 2048u * kk;       // 16 keys = two 8-row groups
        if (SPLIT) {
            dc_umma_ts(tO, a, dc_desc100(v, DC_PANELB, 1024), DC_IDESC(128, 64, 1), acc || kk > 0);
            dc_umma_ts(tO + 64, a, dc_desc100(v + DC_PANELB, DC_PANELB, 1024), DC_IDESC(128, 64, 1), acc || kk > 0);
        } else {
            dc_umma_ts(tO, a, dc_desc100(v, DC_PANELB, 1024), DC_IDESC(128, 128, 1), acc || kk > 0);
        }
    }
}


template <int NS, bool SPLIT>
DC_DEV void dc100_body(const DcTensorMap* tq, const DcTensorMap* tk, const DcTensorMap* tv,
                       float* out, unsigned short* out_bf16, int out_is_bf16, int sq, int sk, float sl2,
                       unsigned char* smem_raw) {
    const unsigned int raw = dc_smem_u32(smem_raw);
    const unsigned int base = (raw + 1023u) & ~1023u;
    const unsigned int sQ = base;
    const unsigned int ring = base + 2 * DC_TILEB;
    const unsigned int bars = ring + NS * DC_TILEB;
    const unsigned int b_q = bars;
    const unsigned int b_full = b_q + 8;
    const unsigned int b_empty = b_full + 8 * NS;
    const unsigned int b_s = b_empty + 8 * NS;
    const unsigned int b_p = b_s + 16;
    const unsigned int b_o = b_p + 16;
    const unsigned int tslot = b_o + 16;
    volatile unsigned int* tslot_ptr = reinterpret_cast<volatile unsigned int*>(smem_raw + (tslot - raw));

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = (int)blockIdx.x * 256;
    const int bh = (int)blockIdx.y;
    if (q0 >= sq || sk <= 0) return;
    const int nkt = (sk + DC_BN - 1) / DC_BN;
    const float NEG = __int_as_float(0xff800000);

    if (tid == 0) {
        dc_mbar_init(b_q, 1);
        for (int i = 0; i < NS; i++) {
            dc_mbar_init(b_full + 8 * i, 1);
            dc_mbar_init(b_empty + 8 * i, 1);
        }
        for (int i = 0; i < 2; i++) {
            dc_mbar_init(b_s + 8 * i, 1);
            dc_mbar_init(b_p + 8 * i, 128);
            dc_mbar_init(b_o + 8 * i, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == 9) {
        const unsigned int ncols = 512;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(tslot), "r"(ncols) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
    dc_tc_fence_before();
    __syncthreads();
    dc_tc_fence_after();
    const unsigned int tbase = *tslot_ptr;
    // Warpgroups 0-1 (softmax) take the registers warpgroup 2 (TMA, MMA,
    // two idle warps) does not need: 2 x 128 x 224 + 128 x 56 = 384 x 168.
    // Each role returns on its own (no code after the split), so ptxas can
    // hold each to its own register budget.
    if (warp < 8) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;\n" ::: "memory");
        // ---- softmax: thread = one query row of tile `tile` ----
        const int tile = warp >> 2, sub = warp & 3;
        const int row = sub * 32 + lane;
        const unsigned int tl = tbase + ((unsigned int)(sub * 32) << 16);
        const unsigned int tS = tl + 128u * tile, tP = tS + 64u, tO = tl + 256u + 128u * tile;
        float m = NEG, l = 0.f;
#pragma unroll 1
        for (int j = 0; j < nkt; j++) {
            dc_mbar_wait(b_s + 8 * tile, j & 1);
            dc_tc_fence_after();
            float s[128];
            {
                unsigned int u[64];
                dc_tmem_ld64(tS, u);
#pragma unroll
                for (int c = 0; c < 64; c++) s[c] = __uint_as_float(u[c]);
                dc_tmem_ld64(tS + 64u, u);
#pragma unroll
                for (int c = 0; c < 64; c++) s[64 + c] = __uint_as_float(u[c]);
            }
            const int lim = sk - j * DC_BN;
            if (lim < DC_BN) {
#pragma unroll
                for (int c = 0; c < 128; c++) s[c] = c < lim ? s[c] : NEG;
            }
            float x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3];
#pragma unroll
            for (int c = 4; c < 128; c += 4) {
                x0 = fmaxf(x0, s[c]);
                x1 = fmaxf(x1, s[c + 1]);
                x2 = fmaxf(x2, s[c + 2]);
                x3 = fmaxf(x3, s[c + 3]);
            }
            const float mn = fmaxf(m, fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)));
            const float ms = (mn == NEG) ? 0.f : mn * sl2;
            const bool grow = mn > m;
            const float a = grow ? dc_exp2(fmaf(m, sl2, -ms)) : 1.f;
            m = mn;
            float y0 = 0.f, y1 = 0.f, y2 = 0.f, y3 = 0.f;
#pragma unroll
            for (int c = 0; c < 128; c += 4) {
                s[c] = dc_exp2(fmaf(s[c], sl2, -ms));
                s[c + 1] = dc_exp2(fmaf(s[c + 1], sl2, -ms));
                s[c + 2] = dc_exp2(fmaf(s[c + 2], sl2, -ms));
                s[c + 3] = dc_exp2(fmaf(s[c + 3], sl2, -ms));
                y0 += s[c];
                y1 += s[c + 1];
                y2 += s[c + 2];
                y3 += s[c + 3];
            }
            l = fmaf(l, a, (y0 + y1) + (y2 + y3));
            // O holds P.V of tiles < j (its MMA finished before S_j's): rescale
            // it when any row of the warp raised its max.
            if (j > 0 && __any_sync(0xffffffffu, grow)) {
#pragma unroll 1
                for (int q = 0; q < 4; q++) {
                    unsigned int o[32];
                    dc_tmem_ld32(tO + 32u * q, o);
#pragma unroll
                    for (int c = 0; c < 32; c++) o[c] = __float_as_uint(__uint_as_float(o[c]) * a);
                    dc_tmem_st32(tO + 32u * q, o);
                }
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                unsigned int pw[32];
#pragma unroll
                for (int w = 0; w < 32; w++) pw[w] = dc_pack_bf16(s[64 * h + 2 * w], s[64 * h + 2 * w + 1]);
                dc_tmem_st32(tP + 32u * h, pw);
            }
            dc_tc_fence_before();
            dc_mbar_arrive(b_p + 8 * tile);
        }
        dc_mbar_wait(b_o + 8 * tile, 0);
        dc_tc_fence_after();
        const float inv = l > 0.f ? 1.f / l : 0.f;
        const int qr = q0 + 128 * tile + row;
        const long off = ((long)bh * sq + qr) * DC_D;
#pragma unroll 1
        for (int q = 0; q < 4; q++) {
            unsigned int o[32];
            dc_tmem_ld32(tO + 32u * q, o);
            if (qr < sq) dc_store_row32(out, out_bf16, out_is_bf16, off + 32 * q, o, inv);
        }
        // Every TMEM read is done: warp 9 may free the columns.
        dc_tc_fence_before();
        asm volatile("bar.sync 1, 288;\n" ::: "memory");
        return;
    }
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;\n" ::: "memory");
    if (warp == 8) {
        // ---- TMA producer ----
        if (lane == 0) {
            dc_tma_prefetch(tq);
            dc_tma_prefetch(tk);
            dc_tma_prefetch(tv);
            dc_mbar_expect(b_q, 2 * DC_TILEB);
            dc_tma_tile(sQ, tq, b_q, q0, bh);
            dc_tma_tile(sQ + DC_TILEB, tq, b_q, q0 + 128, bh);
#pragma unroll 1
            for (int idx = 0; idx < 2 * nkt; idx++) {
                const int slot = idx % NS, use = idx / NS;
                if (use > 0) dc_mbar_wait(b_empty + 8 * slot, (use - 1) & 1);
                dc_mbar_expect(b_full + 8 * slot, DC_TILEB);
                dc_tma_tile(ring + slot * DC_TILEB, (idx & 1) ? tv : tk, b_full + 8 * slot, (idx >> 1) * DC_BN, bh);
            }
        }
        return;
    }
    if (warp == 9) {
        // ---- MMA issuer (one thread) ----
        // Order: S0(0) S1(0) | PV0(0) S0(1) PV1(0) S1(1) | PV0(1) S0(2) ...
        // S_i(j) overwrites the TMEM that PV_i(j-1) reads P from; tcgen05.mma
        // ops from one thread execute in issue order, so that is safe, and
        // S_i(j) done implies PV_i(j-1) done (the softmax relies on it).
        if (lane == 0) {
            dc_mbar_wait(b_q, 0);
            {
                const int slot = 0;
                dc_mbar_wait(b_full + 8 * slot, 0);
                const unsigned int sK = ring;
                dc100_qk(tbase, sQ, sK);
                dc_umma_commit(b_s);
                dc100_qk(tbase + 128u, sQ + DC_TILEB, sK);
                dc_umma_commit(b_s + 8);
                dc_umma_commit(b_empty);
            }
#pragma unroll 1
            for (int j = 1; j < nkt; j++) {
                const int iv = 2 * j - 1, ik = 2 * j;
                dc_mbar_wait(b_full + 8 * (iv % NS), (iv / NS) & 1);
                dc_mbar_wait(b_full + 8 * (ik % NS), (ik / NS) & 1);
                const unsigned int sV = ring + (iv % NS) * DC_TILEB, sK = ring + (ik % NS) * DC_TILEB;
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    dc_mbar_wait(b_p + 8 * i, (j - 1) & 1);
                    dc_tc_fence_after();
                    dc100_pv<SPLIT>(tbase + 256u + 128u * i, tbase + 128u * i + 64u, sV, j > 1);
                    dc100_qk(tbase + 128u * i, sQ + i * DC_TILEB, sK);
                    dc_umma_commit(b_s + 8 * i);
                }
                dc_umma_commit(b_empty + 8 * (iv % NS));
                dc_umma_commit(b_empty + 8 * (ik % NS));
            }
            const int iv = 2 * nkt - 1;
            dc_mbar_wait(b_full + 8 * (iv % NS), (iv / NS) & 1);
            const unsigned int sV = ring + (iv % NS) * DC_TILEB;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                dc_mbar_wait(b_p + 8 * i, (nkt - 1) & 1);
                dc_tc_fence_after();
                dc100_pv<SPLIT>(tbase + 256u + 128u * i, tbase + 128u * i + 64u, sV, nkt > 1);
                dc_umma_commit(b_o + 8 * i);
            }
        }
        __syncwarp();
        asm volatile("bar.sync 1, 288;\n" ::: "memory");
        dc_tc_fence_after();
        const unsigned int ncols = 512;
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tbase), "r"(ncols) : "memory");
    }
}
#endif

#if defined(DC_SM100)
#define DC100_ENTRY(NS, SPLIT)                                                                  \
    extern __shared__ __align__(1024) unsigned char dc_smem[];                                  \
    dc100_body<NS, SPLIT>(&tq, &tk, &tv, out, out_bf16, out_is_bf16, sq, sk, sl2, dc_smem);
#else
#define DC100_ENTRY(NS, SPLIT)                                                                  \
    (void)tq; (void)tk; (void)tv; (void)out; (void)out_bf16; (void)out_is_bf16;                 \
    (void)sq; (void)sk; (void)sl2;                                                              \
    __trap();
#endif

// grid (ceil(sq / 256), bh), 384 threads, DC100_SMEM(4) = 197 888 B dynamic
// shared memory (opt-in). sk >= 1.
extern "C" __global__ void __launch_bounds__(384, 1) fa_dc100_fwd_d128(
    const __grid_constant__ DcTensorMap tq, const __grid_constant__ DcTensorMap tk,
    const __grid_constant__ DcTensorMap tv, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2
) {
    DC100_ENTRY(4, false)
}

// fa_dc100_fwd_d128 with each P.V step as two N = 64 MMAs (one per V panel).
extern "C" __global__ void __launch_bounds__(384, 1) fa_dc100_fwd_d128_nsplit(
    const __grid_constant__ DcTensorMap tq, const __grid_constant__ DcTensorMap tk,
    const __grid_constant__ DcTensorMap tv, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2
) {
    DC100_ENTRY(4, true)
}

// ---------------------------------------------------------------- sm_90a
#if defined(DC_SM90)
// wgmma shared-memory descriptor (Hopper): start >> 4, LBO >> 4 at 16,
// SBO >> 4 at 32, layout SWIZZLE_128B (1) at bit 62.
DC_DEV unsigned long long dc_desc90(unsigned int addr, unsigned int lbo, unsigned int sbo) {
    return (unsigned long long)((addr & 0x3FFFFu) >> 4)
         | ((unsigned long long)((lbo >> 4) & 0x3FFFu) << 16)
         | ((unsigned long long)((sbo >> 4) & 0x3FFFu) << 32)
         | (1ull << 62);
}
DC_DEV void dc_wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DC_DEV void dc_wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> DC_DEV void dc_wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
// Pin accumulator registers at this point (no use of them is moved across).
DC_DEV void dc_fence_regs(float (&d)[64]) {
#pragma unroll
    for (int i = 0; i < 64; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}

// d[64] (+)= A(smem desc) * B(smem desc), m64n128k16 bf16 -> f32, both K-major.
DC_DEV void dc_wgmma_ss(float (&d)[64], unsigned long long da, unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "r"(scale_d));
}

// d[64] += A(registers, bf16x2) * B(smem desc, MN-major), m64n128k16.
DC_DEV void dc_wgmma_rs(float (&d)[64], const unsigned int (&a)[4], unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(scale_d));
}

// d[32H .. 32H + 31] += A(registers) * B(smem desc, MN-major), m64n64k16:
// one 64-column half of an m64n128 accumulator.
template <int H>
DC_DEV void dc_wgmma_rs64(float (&d)[64], const unsigned int (&a)[4], unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 1;\n}\n"
        : "+f"(d[H * 32 + 0]), "+f"(d[H * 32 + 1]), "+f"(d[H * 32 + 2]), "+f"(d[H * 32 + 3]), "+f"(d[H * 32 + 4]), "+f"(d[H * 32 + 5]), "+f"(d[H * 32 + 6]), "+f"(d[H * 32 + 7]), "+f"(d[H * 32 + 8]), "+f"(d[H * 32 + 9]), "+f"(d[H * 32 + 10]), "+f"(d[H * 32 + 11]), "+f"(d[H * 32 + 12]), "+f"(d[H * 32 + 13]), "+f"(d[H * 32 + 14]), "+f"(d[H * 32 + 15]), "+f"(d[H * 32 + 16]), "+f"(d[H * 32 + 17]), "+f"(d[H * 32 + 18]), "+f"(d[H * 32 + 19]), "+f"(d[H * 32 + 20]), "+f"(d[H * 32 + 21]), "+f"(d[H * 32 + 22]), "+f"(d[H * 32 + 23]), "+f"(d[H * 32 + 24]), "+f"(d[H * 32 + 25]), "+f"(d[H * 32 + 26]), "+f"(d[H * 32 + 27]), "+f"(d[H * 32 + 28]), "+f"(d[H * 32 + 29]), "+f"(d[H * 32 + 30]), "+f"(d[H * 32 + 31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(scale_d));
}


template <int NS, bool SPLIT>
DC_DEV void dc90_body(const DcTensorMap* tq, const DcTensorMap* tk, const DcTensorMap* tv,
                      float* out, unsigned short* out_bf16, int out_is_bf16, int sq, int sk, float sl2,
                      unsigned char* smem_raw) {
    const unsigned int raw = dc_smem_u32(smem_raw);
    const unsigned int base = (raw + 1023u) & ~1023u;
    const unsigned int sQ = base;
    const unsigned int ring = base + DC_TILEB;
    const unsigned int bars = ring + NS * DC_TILEB;
    const unsigned int b_q = bars;
    const unsigned int b_full = b_q + 8;
    const unsigned int b_empty = b_full + 8 * NS;

    const int tid = threadIdx.x, wg = tid >> 7, warp = tid >> 5, lane = tid & 31;
    const int q0 = (int)blockIdx.x * 128;
    const int bh = (int)blockIdx.y;
    if (q0 >= sq || sk <= 0) return;
    const int nkt = (sk + DC_BN - 1) / DC_BN;
    const float NEG = __int_as_float(0xff800000);

    if (tid == 0) {
        dc_mbar_init(b_q, 1);
        for (int i = 0; i < NS; i++) {
            dc_mbar_init(b_full + 8 * i, 1);
            dc_mbar_init(b_empty + 8 * i, 8);   // one arrival per consumer warp
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::: "memory");
        if (warp == 0 && lane == 0) {
            dc_tma_prefetch(tq);
            dc_tma_prefetch(tk);
            dc_tma_prefetch(tv);
            dc_mbar_expect(b_q, DC_TILEB);
            dc_tma_tile(sQ, tq, b_q, q0, bh);
#pragma unroll 1
            for (int idx = 0; idx < 2 * nkt; idx++) {
                const int slot = idx % NS, use = idx / NS;
                if (use > 0) dc_mbar_wait(b_empty + 8 * slot, (use - 1) & 1);
                dc_mbar_expect(b_full + 8 * slot, DC_TILEB);
                dc_tma_tile(ring + slot * DC_TILEB, (idx & 1) ? tv : tk, b_full + 8 * slot, (idx >> 1) * DC_BN, bh);
            }
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n" ::: "memory");

    // ---- consumer warpgroup c: query rows [64c, 64c + 64) of the CTA ----
    const int c = wg - 1, w4 = warp & 3, g = lane >> 2, t = lane & 3;
    const unsigned int sQc = sQ + (unsigned int)c * 64u * 128u;
    float S[64], O[64];
    unsigned int P[8][4];
#pragma unroll
    for (int i = 0; i < 64; i++) O[i] = 0.f;
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    auto kv_wait = [&](int idx) -> unsigned int {
        const int slot = idx % NS;
        dc_mbar_wait(b_full + 8 * slot, (idx / NS) & 1);
        return ring + (unsigned int)slot * DC_TILEB;
    };
    auto kv_release = [&](int idx) {
        __syncwarp();
        if (lane == 0) dc_mbar_arrive(b_empty + 8 * (idx % NS));
    };
    auto pv = [&](unsigned int sV) {
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            if (SPLIT) {
                dc_wgmma_rs64<0>(O, P[kk], dc_desc90(sV + 2048u * kk, DC_PANELB, 1024), 1);
                dc_wgmma_rs64<1>(O, P[kk], dc_desc90(sV + DC_PANELB + 2048u * kk, DC_PANELB, 1024), 1);
            } else {
                dc_wgmma_rs(O, P[kk], dc_desc90(sV + 2048u * kk, DC_PANELB, 1024), 1);
            }
        }
    };
    auto qk = [&](unsigned int sK) {
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            const unsigned int off = (unsigned int)(kk >> 2) * DC_PANELB + (unsigned int)(kk & 3) * 32u;
            dc_wgmma_ss(S, dc_desc90(sQc + off, 16, 1024), dc_desc90(sK + off, 16, 1024), kk > 0);
        }
    };
    // Online softmax of tile j in S (in place: S becomes P in f32); returns
    // the O rescale factors of this thread's two rows.
    auto softmax = [&](int j, float& a0, float& a1) {
        // S layout (per warp, 16 rows): S[4n + 0..1] = row g, columns
        // 8n + 2t + {0,1}; S[4n + 2..3] = row g + 8, same columns.
        const int lim = sk - j * DC_BN;
        if (lim < DC_BN) {
#pragma unroll
            for (int n = 0; n < 16; n++) {
                const int c0 = 8 * n + 2 * t;
                if (c0 >= lim) { S[4 * n] = NEG; S[4 * n + 2] = NEG; }
                if (c0 + 1 >= lim) { S[4 * n + 1] = NEG; S[4 * n + 3] = NEG; }
            }
        }
        float r0 = NEG, r1 = NEG;
#pragma unroll
        for (int n = 0; n < 16; n++) {
            r0 = fmaxf(r0, fmaxf(S[4 * n], S[4 * n + 1]));
            r1 = fmaxf(r1, fmaxf(S[4 * n + 2], S[4 * n + 3]));
        }
        r0 = fmaxf(r0, __shfl_xor_sync(0xffffffffu, r0, 1));
        r0 = fmaxf(r0, __shfl_xor_sync(0xffffffffu, r0, 2));
        r1 = fmaxf(r1, __shfl_xor_sync(0xffffffffu, r1, 1));
        r1 = fmaxf(r1, __shfl_xor_sync(0xffffffffu, r1, 2));
        const float mn0 = fmaxf(m0, r0), mn1 = fmaxf(m1, r1);
        const float ms0 = (mn0 == NEG) ? 0.f : mn0 * sl2;
        const float ms1 = (mn1 == NEG) ? 0.f : mn1 * sl2;
        a0 = mn0 > m0 ? dc_exp2(fmaf(m0, sl2, -ms0)) : 1.f;
        a1 = mn1 > m1 ? dc_exp2(fmaf(m1, sl2, -ms1)) : 1.f;
        m0 = mn0;
        m1 = mn1;
        float y0 = 0.f, y1 = 0.f;
#pragma unroll
        for (int n = 0; n < 16; n++) {
            S[4 * n] = dc_exp2(fmaf(S[4 * n], sl2, -ms0));
            S[4 * n + 1] = dc_exp2(fmaf(S[4 * n + 1], sl2, -ms0));
            S[4 * n + 2] = dc_exp2(fmaf(S[4 * n + 2], sl2, -ms1));
            S[4 * n + 3] = dc_exp2(fmaf(S[4 * n + 3], sl2, -ms1));
            y0 += S[4 * n] + S[4 * n + 1];
            y1 += S[4 * n + 2] + S[4 * n + 3];
        }
        l0 = fmaf(l0, a0, y0);
        l1 = fmaf(l1, a1, y1);
    };
    auto rescale_pack = [&](float a0, float a1) {
#pragma unroll
        for (int n = 0; n < 16; n++) {
            O[4 * n] *= a0;
            O[4 * n + 1] *= a0;
            O[4 * n + 2] *= a1;
            O[4 * n + 3] *= a1;
        }
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            P[kk][0] = dc_pack_bf16(S[8 * kk], S[8 * kk + 1]);
            P[kk][1] = dc_pack_bf16(S[8 * kk + 2], S[8 * kk + 3]);
            P[kk][2] = dc_pack_bf16(S[8 * kk + 4], S[8 * kk + 5]);
            P[kk][3] = dc_pack_bf16(S[8 * kk + 6], S[8 * kk + 7]);
        }
    };
    dc_mbar_wait(b_q, 0);
    {
        const unsigned int sK = kv_wait(0);
        dc_wgmma_fence();
        qk(sK);
        dc_wgmma_commit();
        dc_wgmma_wait<0>();
        dc_fence_regs(S);
        kv_release(0);
        float a0, a1;
        softmax(0, a0, a1);
        rescale_pack(a0, a1);
    }
    // Tile j: QK_j and PV_{j-1} in flight together; the softmax of S_j runs
    // while PV_{j-1} finishes, then O is rescaled and P_j packed.
#pragma unroll 1
    for (int j = 1; j < nkt; j++) {
        const unsigned int sK = kv_wait(2 * j);
        dc_wgmma_fence();
        qk(sK);
        dc_wgmma_commit();
        const unsigned int sV = kv_wait(2 * j - 1);
        pv(sV);
        dc_wgmma_commit();
        dc_wgmma_wait<1>();
        dc_fence_regs(S);
        kv_release(2 * j);
        float a0, a1;
        softmax(j, a0, a1);
        dc_wgmma_wait<0>();
        dc_fence_regs(O);
        kv_release(2 * j - 1);
        rescale_pack(a0, a1);
    }
    {
        const unsigned int sV = kv_wait(2 * nkt - 1);
        dc_wgmma_fence();
        pv(sV);
        dc_wgmma_commit();
        dc_wgmma_wait<0>();
        dc_fence_regs(O);
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int row0 = q0 + 64 * c + 16 * w4 + g, row1 = row0 + 8;
    const long r0 = ((long)bh * sq + row0) * DC_D, r1 = r0 + 8L * DC_D;
    if (out_is_bf16) {
#pragma unroll
        for (int n = 0; n < 16; n++) {
            const int col = 8 * n + 2 * t;
            if (row0 < sq) *reinterpret_cast<unsigned int*>(out_bf16 + r0 + col) = dc_pack_bf16(O[4 * n] * inv0, O[4 * n + 1] * inv0);
            if (row1 < sq) *reinterpret_cast<unsigned int*>(out_bf16 + r1 + col) = dc_pack_bf16(O[4 * n + 2] * inv1, O[4 * n + 3] * inv1);
        }
    } else {
#pragma unroll
        for (int n = 0; n < 16; n++) {
            const int col = 8 * n + 2 * t;
            if (row0 < sq) *reinterpret_cast<float2*>(out + r0 + col) = make_float2(O[4 * n] * inv0, O[4 * n + 1] * inv0);
            if (row1 < sq) *reinterpret_cast<float2*>(out + r1 + col) = make_float2(O[4 * n + 2] * inv1, O[4 * n + 3] * inv1);
        }
    }
}
#endif

#if defined(DC_SM90)
#define DC90_ENTRY(NS, SPLIT)                                                                   \
    extern __shared__ __align__(1024) unsigned char dc_smem[];                                  \
    dc90_body<NS, SPLIT>(&tq, &tk, &tv, out, out_bf16, out_is_bf16, sq, sk, sl2, dc_smem);
#else
#define DC90_ENTRY(NS, SPLIT)                                                                   \
    (void)tq; (void)tk; (void)tv; (void)out; (void)out_bf16; (void)out_is_bf16;                 \
    (void)sq; (void)sk; (void)sl2;                                                              \
    __trap();
#endif

// grid (ceil(sq / 128), bh), 384 threads, DC90_SMEM(4) = 165 120 B dynamic
// shared memory (opt-in). sk >= 1.
extern "C" __global__ void __launch_bounds__(384, 1) fa_dc90_fwd_d128(
    const __grid_constant__ DcTensorMap tq, const __grid_constant__ DcTensorMap tk,
    const __grid_constant__ DcTensorMap tv, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2
) {
    DC90_ENTRY(4, false)
}

// fa_dc90_fwd_d128 with each P.V step as two m64n64 wgmmas (one per V panel).
extern "C" __global__ void __launch_bounds__(384, 1) fa_dc90_fwd_d128_nsplit(
    const __grid_constant__ DcTensorMap tq, const __grid_constant__ DcTensorMap tk,
    const __grid_constant__ DcTensorMap tv, float* __restrict__ out,
    unsigned short* __restrict__ out_bf16, int out_is_bf16, int sq, int sk, float sl2
) {
    DC90_ENTRY(4, true)
}

// ============================================================================
// VSA fine stage on the datacenter pipelines (WP-D): a KV-tile-list producer
// feeding the same tcgen05 / wgmma consumers as the dense kernels.
//
// Inputs are VSA's tile-ordered tensors ([bh, num_tiles * 64, 128] bf16,
// `vsa_tile_qkv` / `dcv_prep_*` layout): tile t owns rows [64 t, 64 t + 64)
// of its head, the first `block_sizes[t]` of them real tokens. The TMA maps
// are 3-D (d, rows, bh) with a 64 x 64 x 1 box (128-byte swizzle), so one
// 64-row K or V tile is two 8 KB panels (d 0..63 | d 64..127).
//
// KV-tile-list interface. A consumer row block attends an ordered list of
// 64-key tiles, each one 32-bit entry word:
//   bits  0..19  tile index (row 64 * tile of the K/V maps, per head)
//   bits 20..26  valid keys in the tile (columns >= valid score -inf)
//   bits 27..28  which 64-row query groups of the block may see the tile
// The producer streams the list two tiles (128 keys) per softmax step, in the
// ring order K_a K_b V_a V_b, so step j's tiles are ring indices 4j ..
// 4j + 3 (a partial last step: K_a V_a on sm_100; on sm_90 tile a twice,
// masked, which keeps every wgmma unconditional). The consumers only ever read
// entries; how a list is built is the builder's business:
//   - dcv_build_union (sm_100, here): the ascending union of two query
//     tiles' VSA selections, with per-group visibility bits;
//   - dcv_list_sel (sm_90, here): one query tile's selection, as given;
//   - later, Sol's exact blocks (per-group ballot masks from the route
//     scores) and block-causal attention (contiguous key-tile ranges, with
//     the frame boundary as a `valid` / group mask) are further builders over
//     the same entry word and pipelines.
//
//   fa_dc100_vsa  CTA = 128 query rows = two VSA query tiles (M = 128 is the
//                 tcgen05 shape whose TMEM layout the dense kernel already
//                 proves; M = 64 would halve the tensor-core work but not the
//                 softmax, which is the co-bottleneck). The two tiles attend
//                 the union of their selections, each row masking the tiles
//                 its own group did not select, so the result is exactly
//                 per-tile VSA. 6 warps: 0-3 softmax (one row per thread),
//                 4 list + TMA, 5 MMA. TMEM: S0 | S1 (128 columns each,
//                 double-buffered so QK_{j+1} runs during softmax_j) | O.
//   fa_dc90_vsa   CTA = 3 warpgroups as fa_dc90: warpgroup 0 produces (warp c
//                 feeds consumer c), warpgroups 1-2 each own one VSA query
//                 tile (wgmma M = 64 = one tile) and its own selection, so no
//                 union work is wasted; each has its own K/V ring.
//
// Epilogue: the normalised f32 output either goes to the tile-slot-ordered
// `sparse` buffer (the vsa_mma_attn contract, combined later), or, with
// DCV_FUSE, straight into token order with H3's bf16 combine
// bf16(bf16(sparse) + bf16(bf16(coarse) * gate)) (vsa_combine round16 /
// vsa_combine_g16), bit for bit the unfused result.
// ============================================================================

#define DCV_TILE 64
#define DCV_TILEB 16384            // 64 x 128 bf16
#define DCV_PANELB 8192            // 64 rows x 64 bf16: one swizzled TMA box
#define DCV100_NS 8
#define DCV90_NS 6
#define DCV_FUSE 1                 // combine into token order
#define DCV_GATE 2                 // a gate exists
#define DCV_GATE16 4               // ... and it is bf16
// Fixed shared memory before the list area (bytes); the host adds
// 4 * (2 * ceil(num_tiles / 32) + list_cap) on sm_100.
#define DCV100_SMEM_FIXED(NS) (1024 + 2 * DC_TILEB + (NS) * DCV_TILEB + 256)
#define DCV90_SMEM(NS) (1024 + 2 * DCV_TILEB + 2 * (NS) * DCV_TILEB + 256)

#if defined(DC_SM100) || defined(DC_SM90)
DC_DEV unsigned int dcv_entry(int tile, int valid, unsigned int rows) {
    return (unsigned int)tile | ((unsigned int)valid << 20) | (rows << 27);
}
DC_DEV int dcv_tile(unsigned int e) { return (int)(e & 0xFFFFFu); }
DC_DEV int dcv_valid(unsigned int e) { return (int)((e >> 20) & 127u); }
DC_DEV unsigned int dcv_rows(unsigned int e) { return (e >> 27) & 3u; }

// f32 -> bf16 bits, round to nearest even, NaN kept quiet: kernels.cu's
// fv_bf16_rne, so the fused combine rounds exactly as vsa_combine.
DC_DEV unsigned short dcv_bf16_rne(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        unsigned int h = u >> 16;
        if (u & 0x007FFFFFu) h |= 0x0040u;
        return (unsigned short)h;
    }
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
DC_DEV float dcv_r16(float x) { return __uint_as_float(((unsigned int)dcv_bf16_rne(x)) << 16); }

struct DcvEpi {
    float* out;                    // sparse [bh, padded, 128] or out [bh, seq, 128]
    const float* coarse;           // [bh, num_tiles, 128]
    const float* gate32;           // [bh, seq, 128]
    const unsigned short* gate16;  // [bh, seq, 128]
    const int* slot_src;           // [num_tiles * 64]
    long long seq;
    int num_tiles;
    int mode;
};

// H3's combine of one element: sp = normalised attention, c = coarse,
// g = the gate already as a bf16 value.
DC_DEV float dcv_mix(float sp, float co, float g, bool gated) {
    const float c = dcv_r16(co);
    const float p = gated ? dcv_r16(c * g) : c;
    return dcv_r16(dcv_r16(sp) + p);
}
DC_DEV float dcv_gate_at(const DcvEpi& ep, long idx) {
    return (ep.mode & DCV_GATE16) ? __uint_as_float(((unsigned int)ep.gate16[idx]) << 16) : dcv_r16(ep.gate32[idx]);
}
// Columns [cb, cb + 32) of query `slot` of tile `qt`: f32 `o * inv`.
DC_DEV void dcv_store_row32(const DcvEpi& ep, int bh, int qt, int slot, int cb, const unsigned int (&o)[32], float inv) {
    if (!(ep.mode & DCV_FUSE)) {
        float4* dst = reinterpret_cast<float4*>(
            ep.out + (((long)bh * ep.num_tiles + qt) * DCV_TILE + slot) * DC_D + cb);
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const unsigned int* p = o + 4 * v;
            dst[v] = make_float4(__uint_as_float(p[0]) * inv, __uint_as_float(p[1]) * inv,
                                 __uint_as_float(p[2]) * inv, __uint_as_float(p[3]) * inv);
        }
        return;
    }
    const int src = ep.slot_src[qt * DCV_TILE + slot];
    if (src < 0) return;
    const float* co = ep.coarse + ((long)bh * ep.num_tiles + qt) * DC_D + cb;
    const long row = ((long)bh * ep.seq + src) * DC_D + cb;
    const bool gated = (ep.mode & DCV_GATE) != 0;
    float4* dst = reinterpret_cast<float4*>(ep.out + row);
#pragma unroll
    for (int v = 0; v < 8; v++) {
        float r[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int c = 4 * v + e;
            const float g = gated ? dcv_gate_at(ep, row + c) : 0.f;
            r[e] = dcv_mix(__uint_as_float(o[c]) * inv, co[c], g, gated);
        }
        dst[v] = make_float4(r[0], r[1], r[2], r[3]);
    }
}
// Two adjacent columns (col, col + 1) of query `slot` of tile `qt`, given
// the query's token `src` (fused mode; -1 skips).
DC_DEV void dcv_store_pair(const DcvEpi& ep, int bh, int qt, int slot, int src, int col, float a, float b) {
    if (!(ep.mode & DCV_FUSE)) {
        *reinterpret_cast<float2*>(ep.out + (((long)bh * ep.num_tiles + qt) * DCV_TILE + slot) * DC_D + col) =
            make_float2(a, b);
        return;
    }
    if (src < 0) return;
    const float* co = ep.coarse + ((long)bh * ep.num_tiles + qt) * DC_D + col;
    const long row = ((long)bh * ep.seq + src) * DC_D + col;
    const bool gated = (ep.mode & DCV_GATE) != 0;
    const float g0 = gated ? dcv_gate_at(ep, row) : 0.f, g1 = gated ? dcv_gate_at(ep, row + 1) : 0.f;
    *reinterpret_cast<float2*>(ep.out + row) = make_float2(dcv_mix(a, co[0], g0, gated), dcv_mix(b, co[1], g1, gated));
}
// One 64-row x 128-column tile (rows row..row+63 of head bh): two 8 KB
// panels. The caller posts the 16 KB expect.
DC_DEV void dcv_tma_tile(unsigned int dst, const DcTensorMap* map, unsigned int bar, int row, int bh) {
    dc_tma_load3(dst, map, bar, 0, row, bh);
    dc_tma_load3(dst + DCV_PANELB, map, bar, 64, row, bh);
}
#endif

// ---------------------------------------------------------------- sm_100a
#if defined(DC_SM100)
// S (TMEM, 128 rows x 64 f32 columns at tS) = Q (128 x 128, K-major, 16 KB
// panels) . K^T (one 64-key tile, K-major, 8 KB panels).
DC_DEV void dcv100_qk(unsigned int tS, unsigned int sQ, unsigned int sK) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        const unsigned int oq = (unsigned int)(kk >> 2) * DC_PANELB + (unsigned int)(kk & 3) * 32u;
        const unsigned int ok = (unsigned int)(kk >> 2) * DCV_PANELB + (unsigned int)(kk & 3) * 32u;
        dc_umma_ss(tS, dc_desc100(sQ + oq, 16, 1024), dc_desc100(sK + ok, 16, 1024), DC_IDESC(128, 64, 0), kk > 0);
    }
}
// O (TMEM, 128 x 128 f32) (+)= P (TMEM bf16, the 64 keys packed at tP) .
// V (one 64-key tile, MN-major: the two d panels are LBO = 8 KB apart).
DC_DEV void dcv100_pv(unsigned int tO, unsigned int tP, unsigned int sV, bool acc) {
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        dc_umma_ts(tO, tP + 8u * kk, dc_desc100(sV + 2048u * kk, DCV_PANELB, 1024), DC_IDESC(128, 128, 1),
                   acc || kk > 0);
    }
}

// List builder (one warp): the ascending union of query tile qa's selection
// and, when has_b, query tile qa + 1's, with bit 0 / bit 1 of the group mask
// saying which of the two selected each tile. Returns the entry count.
DC_DEV int dcv_build_union(const unsigned int* sel_a, const unsigned int* sel_b, bool has_b, int topk,
                           const int* block_sizes, int num_tiles, unsigned int* bm_a, unsigned int* bm_b,
                           unsigned int* ent, int lane) {
    const int words = (num_tiles + 31) >> 5;
    for (int w = lane; w < words; w += 32) {
        bm_a[w] = 0u;
        bm_b[w] = 0u;
    }
    __syncwarp();
    for (int i = lane; i < topk; i += 32) {
        const unsigned int ta = sel_a[i];
        if (ta < (unsigned int)num_tiles) atomicOr(bm_a + (ta >> 5), 1u << (ta & 31u));
        if (has_b) {
            const unsigned int tb = sel_b[i];
            if (tb < (unsigned int)num_tiles) atomicOr(bm_b + (tb >> 5), 1u << (tb & 31u));
        }
    }
    __syncwarp();
    int total = 0;
    for (int w0 = 0; w0 < words; w0 += 32) {
        const int w = w0 + lane;
        const unsigned int a = w < words ? bm_a[w] : 0u, b = w < words ? bm_b[w] : 0u;
        unsigned int u = a | b;
        const int cnt = __popc(u);
        int incl = cnt;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, incl, d);
            if (lane >= d) incl += y;
        }
        int pos = total + incl - cnt;
        while (u) {
            const int bit = __ffs(u) - 1;
            const int tile = w * 32 + bit;
            const unsigned int rows = ((a >> bit) & 1u) | (((b >> bit) & 1u) << 1);
            ent[pos++] = dcv_entry(tile, block_sizes[tile], rows);
            u &= u - 1u;
        }
        total += __shfl_sync(0xffffffffu, incl, 31);
    }
    __syncwarp();
    return total;
}

template <int NS>
DC_DEV void dcv100_body(const DcTensorMap* tq, const DcTensorMap* tk, const DcTensorMap* tv,
                        const unsigned int* selected, const int* block_sizes, const DcvEpi& ep,
                        int topk, int q_base, int q_end, float sl2, unsigned char* smem_raw) {
    const int num_tiles = ep.num_tiles;
    const unsigned int raw = dc_smem_u32(smem_raw);
    const unsigned int base = (raw + 1023u) & ~1023u;
    const unsigned int sQ = base;
    const unsigned int ring = base + 2 * DC_TILEB;
    const unsigned int bars = ring + NS * DCV_TILEB;
    const unsigned int b_q = bars;
    const unsigned int b_full = b_q + 8;
    const unsigned int b_empty = b_full + 8 * NS;
    const unsigned int b_s = b_empty + 8 * NS;
    const unsigned int b_p = b_s + 16;
    const unsigned int b_o = b_p + 16;
    const unsigned int tslot = b_o + 8;
    const unsigned int nslot = tslot + 4;
    unsigned char* gbase = smem_raw + (base - raw);
    volatile unsigned int* tslot_ptr = reinterpret_cast<volatile unsigned int*>(gbase + (tslot - base));
    volatile int* nlist_ptr = reinterpret_cast<volatile int*>(gbase + (nslot - base));
    const int words = (num_tiles + 31) >> 5;
    unsigned int* bm_a = reinterpret_cast<unsigned int*>(gbase + (bars - base) + 256);
    unsigned int* bm_b = bm_a + words;
    unsigned int* ent = bm_b + words;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qt0 = q_base + 2 * (int)blockIdx.x;
    const int bh = (int)blockIdx.y;
    if (qt0 >= q_end) return;
    const bool has_b = qt0 + 1 < q_end;
    const float NEG = __int_as_float(0xff800000);

    if (tid == 0) {
        dc_mbar_init(b_q, 1);
        for (int i = 0; i < NS; i++) {
            dc_mbar_init(b_full + 8 * i, 1);
            dc_mbar_init(b_empty + 8 * i, 1);
        }
        for (int i = 0; i < 2; i++) {
            dc_mbar_init(b_s + 8 * i, 1);
            dc_mbar_init(b_p + 8 * i, 128);
        }
        dc_mbar_init(b_o, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == 5) {
        const unsigned int ncols = 512;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(tslot), "r"(ncols) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
    __syncthreads();
    if (warp == 4) {
        // Q first (it does not depend on the list), then the list.
        if (lane == 0) {
            dc_tma_prefetch(tq);
            dc_tma_prefetch(tk);
            dc_tma_prefetch(tv);
            dc_mbar_expect(b_q, 2 * DC_TILEB);
            const int qrow = qt0 * DCV_TILE;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                dc_tma_load3(sQ + h * DCV_PANELB, tq, b_q, 0, qrow + 64 * h, bh);
                dc_tma_load3(sQ + DC_PANELB + h * DCV_PANELB, tq, b_q, 64, qrow + 64 * h, bh);
            }
        }
        const unsigned int* sel_a = selected + ((long)bh * num_tiles + qt0) * topk;
        const int n = dcv_build_union(sel_a, sel_a + topk, has_b, topk, block_sizes, num_tiles, bm_a, bm_b, ent, lane);
        if (lane == 0) *nlist_ptr = n;
    }
    dc_tc_fence_before();
    __syncthreads();
    dc_tc_fence_after();
    const unsigned int tbase = *tslot_ptr;
    const int L = *nlist_ptr;
    const int nsteps = (L + 1) >> 1;

    if (warp < 4) {
        // ---- softmax + epilogue: thread = query row `row` of the block ----
        const int row = warp * 32 + lane, grp = warp >> 1;
        const unsigned int tl = tbase + ((unsigned int)(warp * 32) << 16);
        const unsigned int tO = tl + 256u;
        float m = NEG, l = 0.f;
#pragma unroll 1
        for (int j = 0; j < nsteps; j++) {
            const int buf = j & 1;
            const unsigned int tS = tl + 128u * buf, tP = tS + 64u;
            dc_mbar_wait(b_s + 8 * buf, (j >> 1) & 1);
            dc_tc_fence_after();
            float s[128];
            {
                unsigned int u[64];
                dc_tmem_ld64(tS, u);
#pragma unroll
                for (int c = 0; c < 64; c++) s[c] = __uint_as_float(u[c]);
                dc_tmem_ld64(tS + 64u, u);
#pragma unroll
                for (int c = 0; c < 64; c++) s[64 + c] = __uint_as_float(u[c]);
            }
            const unsigned int ea = ent[2 * j];
            const bool two = 2 * j + 1 < L;
            const unsigned int eb = two ? ent[2 * j + 1] : 0u;
            const int lima = ((dcv_rows(ea) >> grp) & 1u) ? dcv_valid(ea) : 0;
            const int limb = (two && ((dcv_rows(eb) >> grp) & 1u)) ? dcv_valid(eb) : 0;
            if (lima < 64) {
#pragma unroll
                for (int c = 0; c < 64; c++) s[c] = c < lima ? s[c] : NEG;
            }
            if (limb < 64) {
#pragma unroll
                for (int c = 0; c < 64; c++) s[64 + c] = c < limb ? s[64 + c] : NEG;
            }
            float x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3];
#pragma unroll
            for (int c = 4; c < 128; c += 4) {
                x0 = fmaxf(x0, s[c]);
                x1 = fmaxf(x1, s[c + 1]);
                x2 = fmaxf(x2, s[c + 2]);
                x3 = fmaxf(x3, s[c + 3]);
            }
            const float mn = fmaxf(m, fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)));
            const float ms = (mn == NEG) ? 0.f : mn * sl2;
            const bool grow = mn > m;
            const float a = grow ? dc_exp2(fmaf(m, sl2, -ms)) : 1.f;
            m = mn;
            float y0 = 0.f, y1 = 0.f, y2 = 0.f, y3 = 0.f;
#pragma unroll
            for (int c = 0; c < 128; c += 4) {
                s[c] = dc_exp2(fmaf(s[c], sl2, -ms));
                s[c + 1] = dc_exp2(fmaf(s[c + 1], sl2, -ms));
                s[c + 2] = dc_exp2(fmaf(s[c + 2], sl2, -ms));
                s[c + 3] = dc_exp2(fmaf(s[c + 3], sl2, -ms));
                y0 += s[c];
                y1 += s[c + 1];
                y2 += s[c + 2];
                y3 += s[c + 3];
            }
            l = fmaf(l, a, (y0 + y1) + (y2 + y3));
            // O holds P.V of steps < j once PV_{j-1} is done (PV_{j-2} is: it
            // was issued before QK_j). Rescale when a row of the warp grew.
            if (j > 0 && __any_sync(0xffffffffu, grow)) {
                dc_mbar_wait(b_o, (j - 1) & 1);
                dc_tc_fence_after();
#pragma unroll 1
                for (int q = 0; q < 4; q++) {
                    unsigned int o[32];
                    dc_tmem_ld32(tO + 32u * q, o);
#pragma unroll
                    for (int c = 0; c < 32; c++) o[c] = __float_as_uint(__uint_as_float(o[c]) * a);
                    dc_tmem_st32(tO + 32u * q, o);
                }
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                unsigned int pw[32];
#pragma unroll
                for (int w = 0; w < 32; w++) pw[w] = dc_pack_bf16(s[64 * h + 2 * w], s[64 * h + 2 * w + 1]);
                dc_tmem_st32(tP + 32u * h, pw);
            }
            dc_tc_fence_before();
            dc_mbar_arrive(b_p + 8 * buf);
        }
        const int qt = qt0 + grp, slot = row & 63;
        if (nsteps > 0) {
            dc_mbar_wait(b_o, (nsteps - 1) & 1);
            dc_tc_fence_after();
            const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll 1
            for (int q = 0; q < 4; q++) {
                unsigned int o[32];
                dc_tmem_ld32(tO + 32u * q, o);
                if (qt < q_end) dcv_store_row32(ep, bh, qt, slot, 32 * q, o, inv);
            }
        } else if (qt < q_end) {
            unsigned int z[32];
#pragma unroll
            for (int c = 0; c < 32; c++) z[c] = 0u;
            for (int q = 0; q < 4; q++) dcv_store_row32(ep, bh, qt, slot, 32 * q, z, 0.f);
        }
        dc_tc_fence_before();
        asm volatile("bar.sync 1, 160;\n" ::: "memory");
        return;
    }
    if (warp == 4) {
        // ---- TMA producer: the list, two tiles per step, K_a K_b V_a V_b ----
        if (lane == 0) {
#pragma unroll 1
            for (int j = 0; j < nsteps; j++) {
                const int n = min(2, L - 2 * j);
#pragma unroll 1
                for (int t = 0; t < 2 * n; t++) {
                    const int idx = 4 * j + t;
                    const int slot = idx % NS, use = idx / NS;
                    const unsigned int e = ent[2 * j + (t < n ? t : t - n)];
                    if (use > 0) dc_mbar_wait(b_empty + 8 * slot, (use - 1) & 1);
                    dc_mbar_expect(b_full + 8 * slot, DCV_TILEB);
                    dcv_tma_tile(ring + slot * DCV_TILEB, t < n ? tk : tv, b_full + 8 * slot, dcv_tile(e) * DCV_TILE, bh);
                }
            }
        }
        return;
    }
    // ---- warp 5: MMA issuer (one thread) ----
    // Order: QK_0 QK_1 | PV_0 QK_2 | PV_1 QK_3 | ... QK_{j+2} reuses the
    // S / P buffer PV_j reads; tcgen05.mma from one thread runs in issue
    // order, so that is safe, and S_{j+2} done implies PV_j done.
    if (lane == 0 && nsteps > 0) {
        auto kv = [&](int idx) -> unsigned int {
            const int slot = idx % NS;
            dc_mbar_wait(b_full + 8 * slot, (idx / NS) & 1);
            return ring + (unsigned int)slot * DCV_TILEB;
        };
        auto issue_qk = [&](int j) {
            const int n = min(2, L - 2 * j);
            const unsigned int tS = tbase + 128u * (j & 1);
            for (int h = 0; h < n; h++) dcv100_qk(tS + 64u * h, sQ, kv(4 * j + h));
            dc_umma_commit(b_s + 8 * (j & 1));
            for (int h = 0; h < n; h++) dc_umma_commit(b_empty + 8 * ((4 * j + h) % NS));
        };
        dc_mbar_wait(b_q, 0);
        issue_qk(0);
        if (nsteps > 1) issue_qk(1);
#pragma unroll 1
        for (int j = 0; j < nsteps; j++) {
            const int n = min(2, L - 2 * j);
            dc_mbar_wait(b_p + 8 * (j & 1), (j >> 1) & 1);
            dc_tc_fence_after();
            const unsigned int tP = tbase + 128u * (j & 1) + 64u;
            for (int h = 0; h < n; h++) dcv100_pv(tbase + 256u, tP + 32u * h, kv(4 * j + n + h), j > 0 || h > 0);
            dc_umma_commit(b_o);
            for (int h = 0; h < n; h++) dc_umma_commit(b_empty + 8 * ((4 * j + n + h) % NS));
            if (j + 2 < nsteps) issue_qk(j + 2);
        }
    }
    __syncwarp();
    asm volatile("bar.sync 1, 160;\n" ::: "memory");
    dc_tc_fence_after();
    const unsigned int ncols = 512;
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tbase), "r"(ncols) : "memory");
}
#endif

// ---------------------------------------------------------------- sm_90a
#if defined(DC_SM90)
// d[32H .. 32H + 31] (+)= A(smem desc) * B(smem desc), m64n64k16, both
// K-major: one 64-column half of an m64n128 accumulator.
template <int H>
DC_DEV void dc_wgmma_ss64(float (&d)[64], unsigned long long da, unsigned long long db, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[H * 32 + 0]), "+f"(d[H * 32 + 1]), "+f"(d[H * 32 + 2]), "+f"(d[H * 32 + 3]), "+f"(d[H * 32 + 4]), "+f"(d[H * 32 + 5]), "+f"(d[H * 32 + 6]), "+f"(d[H * 32 + 7]), "+f"(d[H * 32 + 8]), "+f"(d[H * 32 + 9]), "+f"(d[H * 32 + 10]), "+f"(d[H * 32 + 11]), "+f"(d[H * 32 + 12]), "+f"(d[H * 32 + 13]), "+f"(d[H * 32 + 14]), "+f"(d[H * 32 + 15]), "+f"(d[H * 32 + 16]), "+f"(d[H * 32 + 17]), "+f"(d[H * 32 + 18]), "+f"(d[H * 32 + 19]), "+f"(d[H * 32 + 20]), "+f"(d[H * 32 + 21]), "+f"(d[H * 32 + 22]), "+f"(d[H * 32 + 23]), "+f"(d[H * 32 + 24]), "+f"(d[H * 32 + 25]), "+f"(d[H * 32 + 26]), "+f"(d[H * 32 + 27]), "+f"(d[H * 32 + 28]), "+f"(d[H * 32 + 29]), "+f"(d[H * 32 + 30]), "+f"(d[H * 32 + 31])
        : "l"(da), "l"(db), "r"(scale_d));
}

// The selection list of one query tile as entry words (all visible to the
// consumer's single 64-row group).
DC_DEV unsigned int dcv_list_sel(const unsigned int* sel, const int* block_sizes, int i) {
    const int tile = (int)sel[i];
    return dcv_entry(tile, block_sizes[tile], 1u);
}

template <int NS>
DC_DEV void dcv90_body(const DcTensorMap* tq, const DcTensorMap* tk, const DcTensorMap* tv,
                       const unsigned int* selected, const int* block_sizes, const DcvEpi& ep,
                       int topk, int q_base, int q_end, float sl2, unsigned char* smem_raw) {
    const int num_tiles = ep.num_tiles;
    const unsigned int raw = dc_smem_u32(smem_raw);
    const unsigned int base = (raw + 1023u) & ~1023u;
    const unsigned int sQ = base;                       // 2 x 16 KB, one per consumer
    const unsigned int rings = base + 2 * DCV_TILEB;    // 2 x NS x 16 KB
    const unsigned int bars = rings + 2 * NS * DCV_TILEB;
    const unsigned int b_q = bars;                      // 2
    const unsigned int b_full = b_q + 16;               // 2 x NS
    const unsigned int b_empty = b_full + 16 * NS;      // 2 x NS

    const int tid = threadIdx.x, wg = tid >> 7, warp = tid >> 5, lane = tid & 31;
    const int qt0 = q_base + 2 * (int)blockIdx.x;
    const int bh = (int)blockIdx.y;
    if (qt0 >= q_end) return;
    const float NEG = __int_as_float(0xff800000);
    const int L = topk, nsteps = (topk + 1) >> 1;

    if (tid == 0) {
        for (int c = 0; c < 2; c++) {
            dc_mbar_init(b_q + 8 * c, 1);
            for (int i = 0; i < NS; i++) {
                dc_mbar_init(b_full + 8 * (c * NS + i), 1);
                dc_mbar_init(b_empty + 8 * (c * NS + i), 4);   // one arrival per consumer warp
            }
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::: "memory");
        // ---- warp c feeds consumer c: its Q tile, then its list ----
        const int c = warp, qt = qt0 + c;
        if (c < 2 && lane == 0 && qt < q_end) {
            const unsigned int* sel = selected + ((long)bh * num_tiles + qt) * topk;
            const unsigned int ring = rings + (unsigned int)c * NS * DCV_TILEB;
            dc_tma_prefetch(tq);
            dc_tma_prefetch(tk);
            dc_tma_prefetch(tv);
            dc_mbar_expect(b_q + 8 * c, DCV_TILEB);
            dcv_tma_tile(sQ + (unsigned int)c * DCV_TILEB, tq, b_q + 8 * c, qt * DCV_TILE, bh);
#pragma unroll 1
            for (int j = 0; j < nsteps; j++) {
                // Always K_a K_b V_a V_b (a partial last step repeats tile a).
#pragma unroll 1
                for (int t = 0; t < 4; t++) {
                    const int idx = 4 * j + t;
                    const int slot = idx % NS, use = idx / NS;
                    const unsigned int e = dcv_list_sel(sel, block_sizes, min(2 * j + (t & 1), L - 1));
                    const unsigned int bf = b_full + 8 * (c * NS + slot);
                    if (use > 0) dc_mbar_wait(b_empty + 8 * (c * NS + slot), (use - 1) & 1);
                    dc_mbar_expect(bf, DCV_TILEB);
                    dcv_tma_tile(ring + slot * DCV_TILEB, t < 2 ? tk : tv, bf, dcv_tile(e) * DCV_TILE, bh);
                }
            }
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n" ::: "memory");

    // ---- consumer warpgroup c: the 64 query rows of tile qt0 + c ----
    const int c = wg - 1, qt = qt0 + c;
    if (qt >= q_end) return;
    const int w4 = warp & 3, g = lane >> 2, t = lane & 3;
    const unsigned int* sel = selected + ((long)bh * num_tiles + qt) * topk;
    const unsigned int sQc = sQ + (unsigned int)c * DCV_TILEB;
    const unsigned int ring = rings + (unsigned int)c * NS * DCV_TILEB;
    const unsigned int bfull = b_full + 8u * c * NS, bempty = b_empty + 8u * c * NS;
    float S[64], O[64];
    unsigned int P[8][4];
#pragma unroll
    for (int i = 0; i < 64; i++) O[i] = 0.f;
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    auto kv_wait = [&](int idx) -> unsigned int {
        const int slot = idx % NS;
        dc_mbar_wait(bfull + 8 * slot, (idx / NS) & 1);
        return ring + (unsigned int)slot * DCV_TILEB;
    };
    auto kv_release = [&](int idx) {
        __syncwarp();
        if (lane == 0) dc_mbar_arrive(bempty + 8 * (idx % NS));
    };
    auto steps = [&](int j) { return min(2, L - 2 * j); };
    // Every step is two tiles, K_a K_b V_a V_b at ring indices 4j .. 4j + 3,
    // and every wait, wgmma and release is unconditional (a wgmma or a
    // barrier in a data-dependent branch makes ptxas serialise the wgmmas,
    // C7519). A partial last step streams tile a twice: its second half is
    // masked to -inf, so that P is exactly 0 and its P.V adds zeros.
    // S (both halves) = Q . [K_a | K_b]^T for step j; the ring slots are
    // waited for before the wgmma fence.
    auto qk = [&](int j) {
        const unsigned int sKa = kv_wait(4 * j), sKb = kv_wait(4 * j + 1);
        dc_wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            const unsigned int off = (unsigned int)(kk >> 2) * DCV_PANELB + (unsigned int)(kk & 3) * 32u;
            dc_wgmma_ss64<0>(S, dc_desc90(sQc + off, 16, 1024), dc_desc90(sKa + off, 16, 1024), kk > 0);
        }
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            const unsigned int off = (unsigned int)(kk >> 2) * DCV_PANELB + (unsigned int)(kk & 3) * 32u;
            dc_wgmma_ss64<1>(S, dc_desc90(sQc + off, 16, 1024), dc_desc90(sKb + off, 16, 1024), kk > 0);
        }
    };
    auto qk_release = [&](int j) {
        kv_release(4 * j);
        kv_release(4 * j + 1);
    };
    // O += P . [V_a ; V_b] for step j (the caller fences).
    auto pv = [&](int j) {
        const unsigned int sVa = kv_wait(4 * j + 2), sVb = kv_wait(4 * j + 3);
#pragma unroll
        for (int kk = 0; kk < 4; kk++) dc_wgmma_rs(O, P[kk], dc_desc90(sVa + 2048u * kk, DCV_PANELB, 1024), 1);
#pragma unroll
        for (int kk = 0; kk < 4; kk++) dc_wgmma_rs(O, P[4 + kk], dc_desc90(sVb + 2048u * kk, DCV_PANELB, 1024), 1);
    };
    auto pv_release = [&](int j) {
        kv_release(4 * j + 2);
        kv_release(4 * j + 3);
    };
    auto softmax = [&](int j, float& a0, float& a1) {
        // S layout (per warp, 16 rows): S[4n + 0..1] = row g, columns
        // 8n + 2t + {0,1}; S[4n + 2..3] = row g + 8. Columns 0..63 are tile a,
        // 64..127 tile b.
        const int n2 = steps(j);
        const int lima = dcv_valid(dcv_list_sel(sel, block_sizes, 2 * j));
        const int limb = n2 > 1 ? dcv_valid(dcv_list_sel(sel, block_sizes, 2 * j + 1)) : 0;
        if (lima < 64 || limb < 64) {
#pragma unroll
            for (int n = 0; n < 16; n++) {
                const int lim = n < 8 ? lima : limb;
                const int c0 = 8 * (n & 7) + 2 * t;
                if (c0 >= lim) { S[4 * n] = NEG; S[4 * n + 2] = NEG; }
                if (c0 + 1 >= lim) { S[4 * n + 1] = NEG; S[4 * n + 3] = NEG; }
            }
        }
        float r0 = NEG, r1 = NEG;
#pragma unroll
        for (int n = 0; n < 16; n++) {
            r0 = fmaxf(r0, fmaxf(S[4 * n], S[4 * n + 1]));
            r1 = fmaxf(r1, fmaxf(S[4 * n + 2], S[4 * n + 3]));
        }
        r0 = fmaxf(r0, __shfl_xor_sync(0xffffffffu, r0, 1));
        r0 = fmaxf(r0, __shfl_xor_sync(0xffffffffu, r0, 2));
        r1 = fmaxf(r1, __shfl_xor_sync(0xffffffffu, r1, 1));
        r1 = fmaxf(r1, __shfl_xor_sync(0xffffffffu, r1, 2));
        const float mn0 = fmaxf(m0, r0), mn1 = fmaxf(m1, r1);
        const float ms0 = (mn0 == NEG) ? 0.f : mn0 * sl2;
        const float ms1 = (mn1 == NEG) ? 0.f : mn1 * sl2;
        a0 = mn0 > m0 ? dc_exp2(fmaf(m0, sl2, -ms0)) : 1.f;
        a1 = mn1 > m1 ? dc_exp2(fmaf(m1, sl2, -ms1)) : 1.f;
        m0 = mn0;
        m1 = mn1;
        float y0 = 0.f, y1 = 0.f;
#pragma unroll
        for (int n = 0; n < 16; n++) {
            S[4 * n] = dc_exp2(fmaf(S[4 * n], sl2, -ms0));
            S[4 * n + 1] = dc_exp2(fmaf(S[4 * n + 1], sl2, -ms0));
            S[4 * n + 2] = dc_exp2(fmaf(S[4 * n + 2], sl2, -ms1));
            S[4 * n + 3] = dc_exp2(fmaf(S[4 * n + 3], sl2, -ms1));
            y0 += S[4 * n] + S[4 * n + 1];
            y1 += S[4 * n + 2] + S[4 * n + 3];
        }
        l0 = fmaf(l0, a0, y0);
        l1 = fmaf(l1, a1, y1);
    };
    auto rescale_pack = [&](float a0, float a1) {
#pragma unroll
        for (int n = 0; n < 16; n++) {
            O[4 * n] *= a0;
            O[4 * n + 1] *= a0;
            O[4 * n + 2] *= a1;
            O[4 * n + 3] *= a1;
        }
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            P[kk][0] = dc_pack_bf16(S[8 * kk], S[8 * kk + 1]);
            P[kk][1] = dc_pack_bf16(S[8 * kk + 2], S[8 * kk + 3]);
            P[kk][2] = dc_pack_bf16(S[8 * kk + 4], S[8 * kk + 5]);
            P[kk][3] = dc_pack_bf16(S[8 * kk + 6], S[8 * kk + 7]);
        }
    };
    dc_mbar_wait(b_q + 8 * c, 0);
    {
        qk(0);
        dc_wgmma_commit();
        dc_wgmma_wait<0>();
        dc_fence_regs(S);
        qk_release(0);
        float a0, a1;
        softmax(0, a0, a1);
        rescale_pack(a0, a1);
    }
    // Step j: QK_j and PV_{j-1} in flight together; the softmax of S_j runs
    // while PV_{j-1} finishes, then O is rescaled and P_j packed.
#pragma unroll 1
    for (int j = 1; j < nsteps; j++) {
        qk(j);
        dc_wgmma_commit();
        pv(j - 1);
        dc_wgmma_commit();
        dc_wgmma_wait<1>();
        dc_fence_regs(S);
        qk_release(j);
        float a0, a1;
        softmax(j, a0, a1);
        dc_wgmma_wait<0>();
        dc_fence_regs(O);
        pv_release(j - 1);
        rescale_pack(a0, a1);
    }
    {
        const int j = nsteps - 1;
        const unsigned int sVa = kv_wait(4 * j + 2), sVb = kv_wait(4 * j + 3);
        dc_wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++) dc_wgmma_rs(O, P[kk], dc_desc90(sVa + 2048u * kk, DCV_PANELB, 1024), 1);
#pragma unroll
        for (int kk = 0; kk < 4; kk++) dc_wgmma_rs(O, P[4 + kk], dc_desc90(sVb + 2048u * kk, DCV_PANELB, 1024), 1);
        dc_wgmma_commit();
        dc_wgmma_wait<0>();
        dc_fence_regs(O);
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int slot0 = 16 * w4 + g, slot1 = slot0 + 8;
    const bool fuse = (ep.mode & DCV_FUSE) != 0;
    const int src0 = fuse ? ep.slot_src[qt * DCV_TILE + slot0] : 0;
    const int src1 = fuse ? ep.slot_src[qt * DCV_TILE + slot1] : 0;
#pragma unroll
    for (int n = 0; n < 16; n++) {
        const int col = 8 * n + 2 * t;
        dcv_store_pair(ep, bh, qt, slot0, src0, col, O[4 * n] * inv0, O[4 * n + 1] * inv0);
        dcv_store_pair(ep, bh, qt, slot1, src1, col, O[4 * n + 2] * inv1, O[4 * n + 3] * inv1);
    }
}
#endif

#define DCV_ARGS                                                                                \
    const __grid_constant__ DcTensorMap tq, const __grid_constant__ DcTensorMap tk,           \
    const __grid_constant__ DcTensorMap tv, const unsigned int* __restrict__ selected,        \
    const int* __restrict__ block_sizes, const int* __restrict__ slot_src,                    \
    float* __restrict__ out, const float* __restrict__ coarse,                                \
    const float* __restrict__ gate32, const unsigned short* __restrict__ gate16,              \
    int num_tiles, int topk, int q_base, int q_end, long long seq, float sl2, int mode
#define DCV_EPI DcvEpi ep{out, coarse, gate32, gate16, slot_src, seq, num_tiles, mode};
#define DCV_UNUSED                                                                              \
    (void)tq; (void)tk; (void)tv; (void)selected; (void)block_sizes; (void)slot_src;          \
    (void)out; (void)coarse; (void)gate32; (void)gate16; (void)num_tiles; (void)topk;         \
    (void)q_base; (void)q_end; (void)seq; (void)sl2; (void)mode;                              \
    __trap();

// grid (ceil((q_end - q_base) / 2), bh), 192 threads, DCV100_SMEM_FIXED(8)
// + 4 * (2 * ceil(num_tiles / 32) + min(2 * topk, num_tiles)) bytes of
// dynamic shared memory. Query tiles [q_base, q_end); topk >= 1.
extern "C" __global__ void __launch_bounds__(192, 1) fa_dc100_vsa(DCV_ARGS) {
#if defined(DC_SM100)
    extern __shared__ __align__(1024) unsigned char dc_smem[];
    DCV_EPI
    dcv100_body<DCV100_NS>(&tq, &tk, &tv, selected, block_sizes, ep, topk, q_base, q_end, sl2, dc_smem);
#else
    DCV_UNUSED
#endif
}

// grid (ceil((q_end - q_base) / 2), bh), 384 threads, DCV90_SMEM(6) bytes of
// dynamic shared memory.
extern "C" __global__ void __launch_bounds__(384, 1) fa_dc90_vsa(DCV_ARGS) {
#if defined(DC_SM90)
    extern __shared__ __align__(1024) unsigned char dc_smem[];
    DCV_EPI
    dcv90_body<DCV90_NS>(&tq, &tk, &tv, selected, block_sizes, ep, topk, q_base, q_end, sl2, dc_smem);
#else
    DCV_UNUSED
#endif
}

// VSA prep in one pass per tensor: the tile-ordered bf16 copy
// (vsa_tile_qkv) and the tile means (vsa_tile_mean) of q, k and v, reading
// each input once. Same arithmetic in the same order as those kernels:
// `round16` pools bf16-rounded values (H3), else the f32 inputs (Wan), and
// the mean is the in-order f32 sum over the tile's real slots divided by
// their count, so the means (and hence the selection) are bit-identical.
// grid (num_tiles, bh, 3: q, k, v), 128 threads (dim 128); `means` bit z
// asks for tensor z's means (v's only feed the gated compression branch).
extern "C" __global__ void __launch_bounds__(128) dcv_prep_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const int* __restrict__ slot_src, const int* __restrict__ block_sizes,
    unsigned short* __restrict__ qt, unsigned short* __restrict__ kt, unsigned short* __restrict__ vt,
    float* __restrict__ qc, float* __restrict__ kc, float* __restrict__ vc,
    long long seq, int num_tiles, int means, int round16
) {
#if defined(DC_SM100) || defined(DC_SM90)
    const int tile = blockIdx.x, z = blockIdx.z, d = threadIdx.x;
    const long bh = blockIdx.y;
    const float* x = z == 0 ? q : (z == 1 ? k : v);
    unsigned short* xt = z == 0 ? qt : (z == 1 ? kt : vt);
    float* xc = z == 0 ? qc : (z == 1 ? kc : vc);
    const int n = block_sizes[tile];
    const float* xb = x + bh * seq * DC_D + d;
    unsigned short* tb = xt + (bh * num_tiles + tile) * (long)(DCV_TILE * DC_D) + d;
    const int* ss = slot_src + (long)tile * DCV_TILE;
    float acc = 0.f;
#pragma unroll 8
    for (int j = 0; j < DCV_TILE; j++) {
        const int src = ss[j];
        const float val = src >= 0 ? xb[(long)src * DC_D] : 0.f;
        const unsigned short b = dcv_bf16_rne(val);
        tb[j * DC_D] = b;
        if (j < n) acc += round16 ? __uint_as_float(((unsigned int)b) << 16) : val;
    }
    if ((means >> z) & 1) xc[(bh * num_tiles + tile) * DC_D + d] = n > 0 ? acc / (float)n : 0.f;
#else
    (void)q; (void)k; (void)v; (void)slot_src; (void)block_sizes; (void)qt; (void)kt; (void)vt;
    (void)qc; (void)kc; (void)vc; (void)seq; (void)num_tiles; (void)means; (void)round16;
    __trap();
#endif
}

// dcv_prep_f32 on bf16 inputs (vsa_tile_qkv_b16 + vsa_tile_mean_b16).
extern "C" __global__ void __launch_bounds__(128) dcv_prep_b16(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const int* __restrict__ slot_src,
    const int* __restrict__ block_sizes,
    unsigned short* __restrict__ qt, unsigned short* __restrict__ kt, unsigned short* __restrict__ vt,
    float* __restrict__ qc, float* __restrict__ kc, float* __restrict__ vc,
    long long seq, int num_tiles, int means
) {
#if defined(DC_SM100) || defined(DC_SM90)
    const int tile = blockIdx.x, z = blockIdx.z, d = threadIdx.x;
    const long bh = blockIdx.y;
    const unsigned short* x = z == 0 ? q : (z == 1 ? k : v);
    unsigned short* xt = z == 0 ? qt : (z == 1 ? kt : vt);
    float* xc = z == 0 ? qc : (z == 1 ? kc : vc);
    const int n = block_sizes[tile];
    const unsigned short* xb = x + bh * seq * DC_D + d;
    unsigned short* tb = xt + (bh * num_tiles + tile) * (long)(DCV_TILE * DC_D) + d;
    const int* ss = slot_src + (long)tile * DCV_TILE;
    float acc = 0.f;
#pragma unroll 8
    for (int j = 0; j < DCV_TILE; j++) {
        const int src = ss[j];
        const unsigned short b = src >= 0 ? xb[(long)src * DC_D] : (unsigned short)0;
        tb[j * DC_D] = b;
        if (j < n) acc += __uint_as_float(((unsigned int)b) << 16);
    }
    if ((means >> z) & 1) xc[(bh * num_tiles + tile) * DC_D + d] = n > 0 ? acc / (float)n : 0.f;
#else
    (void)q; (void)k; (void)v; (void)slot_src; (void)block_sizes; (void)qt; (void)kt; (void)vt;
    (void)qc; (void)kc; (void)vc; (void)seq; (void)num_tiles; (void)means;
    __trap();
#endif
}
