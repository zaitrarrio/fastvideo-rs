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
