// ==== region: sol ====
// Sol-Attn / PISA / SLA device stages. Sequential 64-token blocks reuse
// `vsa_tile_mean` for means; this region adds the sum-pool, threshold
// routing, sentinel-masked exact lists, flash partials, coarse remainder,
// and LSE merge. Unselected blocks contribute a coarse term (pooled-K
// score, summed-V, multiplicity block_len) merged into the same softmax.
#define SOL_SENTINEL 0xFFFFFFFFu
#define SOL_THRESHOLD_EPS 1.0e-6f
#define SOL_LOG2_E 1.4426950408889634f
#define SOL_NEG __int_as_float(0xff800000)

extern "C" __global__ void sol_tile_sum(
    const float* x, const int* slot_src, const int* block_sizes, float* out,
    long seq, int dim, int num_tiles, int tile_elems
) {
    int tile = blockIdx.x;
    long bh = blockIdx.y;
    if (tile >= num_tiles) return;
    int n = block_sizes[tile];
    const float* xb = x + bh * seq * (long)dim;
    float* ob = out + (bh * (long)num_tiles + tile) * (long)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) {
            int tok = slot_src[(long)tile * tile_elems + j];
            if (tok >= 0) acc += xb[(long)tok * dim + d];
        }
        ob[d] = acc;
    }
}

extern "C" __global__ void sol_diag_threshold(
    const float* q_bar, const float* kc, float* thresh,
    int n, int dim, float tau, float scale
) {
    int bh = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sol_th_sm[];
    float* mu = sol_th_sm;
    float* var = sol_th_sm + dim;
    float inv = 1.0f / (float)(n > 0 ? n : 1);
    for (int d = tid; d < dim; d += blockDim.x) {
        float s = 0.0f, s2 = 0.0f;
        const float* row0 = kc + (long)bh * n * dim;
        for (int j = 0; j < n; j++) {
            float x = row0[(long)j * dim + d];
            s += x;
            s2 += x * x;
        }
        mu[d] = s * inv;
        float v = s2 * inv - mu[d] * mu[d];
        var[d] = v > 0.0f ? v : 0.0f;
    }
    __syncthreads();
    float log2_scale = scale * SOL_LOG2_E;
    for (int i = tid; i < n; i += blockDim.x) {
        const float* q = q_bar + ((long)bh * n + i) * dim;
        float mean = 0.0f, vr = 0.0f;
        for (int d = 0; d < dim; d++) {
            mean += q[d] * mu[d];
            vr += q[d] * q[d] * var[d];
        }
        mean *= log2_scale;
        vr *= log2_scale * log2_scale;
        float stdv = sqrtf((vr > 0.0f ? vr : 0.0f) + SOL_THRESHOLD_EPS);
        thresh[(long)bh * n + i] = mean + tau * stdv;
    }
}

extern "C" __global__ void sol_exact_lists(
    const float* scores, const float* thresh, const int* sink_flags,
    const int* block_sizes, unsigned int* lists,
    int tokens, int n
) {
    int i = blockIdx.x;
    int bh = blockIdx.y;
    if (i >= n) return;
    int q_len = block_sizes[i];
    int q_start = i * 64;
    float th = thresh[(long)bh * n + i];
    unsigned int* row = lists + ((long)bh * n + i) * n;
    const float* sc = scores + bh * (long)tokens * n;
    int w = 0;
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int t = 0; t < q_len; t++) sum += sc[(long)(q_start + t) * n + j];
        float cm = q_len > 0 ? sum / (float)q_len : SOL_NEG;
        int local = (i > j ? i - j : j - i) <= 1;
        int sink = sink_flags[j];
        if (cm > th || local || sink) row[w++] = (unsigned int)j;
    }
    for (; w < n; w++) row[w] = SOL_SENTINEL;
}

extern "C" __global__ void sol_fine_partials(
    const float* q, const float* k, const float* v, const unsigned int* lists,
    float* m_out, float* l_out, float* acc_out,
    long seq, int dim, int nq, int max_keep, int blk_q, int blk_k,
    float scale, int log2_space
) {
    int t = blockIdx.x;
    long bh = blockIdx.y;
    if (t >= seq) return;
    int tid = threadIdx.x;
    extern __shared__ float sol_sm[];
    const float* qb = q + (bh * seq + t) * (long)dim;
    float qv = tid < dim ? qb[tid] : 0.0f;
    int qi = t / blk_q;
    const unsigned int* list = lists + (bh * (long)nq + qi) * max_keep;
    float m = SOL_NEG, l = 0.0f, acc = 0.0f;
    for (int s = 0; s < max_keep; s++) {
        unsigned int jb = list[s];
        if (jb == SOL_SENTINEL) break;
        int k0 = (int)jb * blk_k;
        int k1 = k0 + blk_k;
        if (k1 > (int)seq) k1 = (int)seq;
        for (int u = k0; u < k1; u++) {
            const float* kr = k + (bh * seq + u) * (long)dim;
            const float* vr = v + (bh * seq + u) * (long)dim;
            float kv = tid < dim ? kr[tid] : 0.0f;
            sol_sm[tid] = (tid < dim) ? qv * kv : 0.0f;
            __syncthreads();
            for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
                if (tid < off) sol_sm[tid] += sol_sm[tid + off];
                __syncthreads();
            }
            float score = sol_sm[0] * scale;
            float nm = fmaxf(m, score);
            float alpha = (m == SOL_NEG) ? 0.0f : (log2_space ? exp2f(m - nm) : expf(m - nm));
            float p = log2_space ? exp2f(score - nm) : expf(score - nm);
            float vv = tid < dim ? vr[tid] : 0.0f;
            acc = acc * alpha + p * vv;
            l = l * alpha + p;
            m = nm;
            __syncthreads();
        }
    }
    if (tid == 0) {
        m_out[bh * seq + t] = m;
        l_out[bh * seq + t] = l;
    }
    if (tid < dim) acc_out[(bh * seq + t) * (long)dim + tid] = acc;
}

extern "C" __global__ void sol_coarse_partials(
    const float* q, const float* kc, const float* vc, const unsigned int* lists,
    const int* block_sizes, float* m_out, float* l_out, float* acc_out,
    float* tail_out, int has_tail,
    long seq, int dim, int n, int max_keep, float scale, int log2_space
) {
    int t = blockIdx.x;
    long bh = blockIdx.y;
    if (t >= seq) return;
    int tid = threadIdx.x;
    extern __shared__ float sol_csm[];
    const float* qb = q + (bh * seq + t) * (long)dim;
    float qv = tid < dim ? qb[tid] : 0.0f;
    int nq = n;
    int qi = t / 64;
    if (qi >= nq) return;
    const unsigned int* list = lists + (bh * (long)nq + qi) * max_keep;
    float m = SOL_NEG, l = 0.0f, acc = 0.0f, tail = 0.0f;
    for (int j = 0; j < n; j++) {
        int exact = 0;
        for (int s = 0; s < max_keep; s++) {
            unsigned int e = list[s];
            if (e == SOL_SENTINEL) break;
            if (e == (unsigned int)j) { exact = 1; break; }
        }
        if (exact) continue;
        const float* kcj = kc + (bh * (long)n + j) * (long)dim;
        const float* vcj = vc + (bh * (long)n + j) * (long)dim;
        float kv = tid < dim ? kcj[tid] : 0.0f;
        sol_csm[tid] = (tid < dim) ? qv * kv : 0.0f;
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) sol_csm[tid] += sol_csm[tid + off];
            __syncthreads();
        }
        float score = sol_csm[0] * scale;
        float wgt = (float)block_sizes[j];
        float nm = fmaxf(m, score);
        float alpha = (m == SOL_NEG) ? 0.0f : (log2_space ? exp2f(m - nm) : expf(m - nm));
        float p = log2_space ? exp2f(score - nm) : expf(score - nm);
        float vv = tid < dim ? vcj[tid] : 0.0f;
        acc = acc * alpha + p * vv;
        l = l * alpha + p * wgt;
        tail = tail * alpha + p;
        m = nm;
        __syncthreads();
    }
    if (tid == 0) {
        m_out[bh * seq + t] = m;
        l_out[bh * seq + t] = l;
        if (has_tail) tail_out[bh * seq + t] = tail;
    }
    if (tid < dim) acc_out[(bh * seq + t) * (long)dim + tid] = acc;
}

extern "C" __global__ void sol_lse_combine(
    const float* m1, const float* l1, const float* acc1,
    const float* m2, const float* l2, const float* acc2,
    float* m_out, float* l_out, float* acc_out,
    long rows, int dim, int log2_space
) {
    long row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= dim) return;
    float a = m1[row], b = m2[row];
    float m = fmaxf(a, b);
    float s1 = (a == SOL_NEG) ? 0.0f : (log2_space ? exp2f(a - m) : expf(a - m));
    float s2 = (b == SOL_NEG) ? 0.0f : (log2_space ? exp2f(b - m) : expf(b - m));
    if (d == 0) {
        m_out[row] = m;
        l_out[row] = l1[row] * s1 + l2[row] * s2;
    }
    acc_out[row * dim + d] = acc1[row * dim + d] * s1 + acc2[row * dim + d] * s2;
}

extern "C" __global__ void sol_lse_merge(
    const float* m1, const float* l1, const float* acc1,
    const float* m2, const float* l2, const float* acc2,
    float* out, long rows, int dim, int log2_space
) {
    long row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= dim) return;
    float a = m1[row], b = m2[row];
    float m = fmaxf(a, b);
    float s1 = (a == SOL_NEG) ? 0.0f : (log2_space ? exp2f(a - m) : expf(a - m));
    float s2 = (b == SOL_NEG) ? 0.0f : (log2_space ? exp2f(b - m) : expf(b - m));
    float l = l1[row] * s1 + l2[row] * s2;
    float acc = acc1[row * dim + d] * s1 + acc2[row * dim + d] * s2;
    out[row * dim + d] = l > 0.0f ? acc / l : 0.0f;
}

extern "C" __global__ void sol_normalize_partials(
    const float* l, const float* acc, float* out, long rows, int dim
) {
    long row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= dim) return;
    float z = l[row];
    out[row * dim + d] = z > 0.0f ? acc[row * dim + d] / z : 0.0f;
}

extern "C" __global__ void sol_global_h_bar(
    const float* k, const float* v, const float* kc, float* h,
    long seq, int dim, int n, int tile
) {
    int bh = blockIdx.x;
    int ed = blockIdx.y * blockDim.x + threadIdx.x;
    int cells = dim * dim;
    if (ed >= cells) return;
    int e = ed / dim, d = ed - e * dim;
    float acc = 0.0f;
    for (int t = 0; t < (int)seq; t++) {
        int j = t / tile;
        float dk = k[(bh * seq + t) * (long)dim + e] - kc[(bh * (long)n + j) * (long)dim + e];
        acc += dk * v[(bh * seq + t) * (long)dim + d];
    }
    float inv = n > 0 ? 1.0f / (float)n : 0.0f;
    h[(bh * (long)cells) + ed] = acc * inv;
}

extern "C" __global__ void sol_pisa_first_order(
    const float* q, const float* h_bar, const float* coarse_m, const float* coarse_l,
    const float* merged_m, float* acc, long seq, int dim, float fo
) {
    int t = blockIdx.x;
    long bh = blockIdx.y;
    if (t >= seq) return;
    int d = threadIdx.x;
    if (d >= dim) return;
    // coarse_l: the remainder's unweighted probability sum (coarse tail).
    float cm = coarse_m[bh * seq + t];
    float tail = (cm == SOL_NEG) ? 0.0f : coarse_l[bh * seq + t] * expf(cm - merged_m[bh * seq + t]);
    if (tail <= 0.0f) return;
    const float* qi = q + (bh * seq + t) * (long)dim;
    const float* hb = h_bar + bh * (long)dim * dim;
    float qh = 0.0f;
    for (int e = 0; e < dim; e++) qh += qi[e] * hb[e * dim + d];
    // fo: the softmax scale (the logit's Taylor step), 0 = zeroth order only.
    acc[(bh * seq + t) * (long)dim + d] += tail * fo * qh;
}

// Tensor-core fine partials: `vsa_mma_attn` body, sentinel-truncated list,
// emits (m, l, acc) in token order instead of a normalised output.
extern "C" __global__ void __launch_bounds__(128, 2) sol_mma_attn_partials(
    const unsigned short* __restrict__ qt, const unsigned short* __restrict__ kt,
    const unsigned short* __restrict__ vt, const unsigned int* __restrict__ selected,
    const int* __restrict__ block_sizes, const int* __restrict__ slot_src,
    float* __restrict__ m_out, float* __restrict__ l_out, float* __restrict__ acc_out,
    int num_tiles, int topk, float scale_log2, int q_base, long seq
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(128) unsigned char mma_smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int qtile = q_base + (int)blockIdx.x, bh = blockIdx.y;
    if (qtile >= num_tiles) return;
    const long padded = (long)num_tiles * MMA_TILE;
    const unsigned short* qb = qt + (bh * padded + (long)qtile * MMA_TILE) * MMA_DIM;
    const unsigned short* kb = kt + bh * padded * MMA_DIM;
    const unsigned short* vb = vt + bh * padded * MMA_DIM;
    const unsigned int* sel = selected + ((long)bh * num_tiles + qtile) * topk;

    int nsel = 0;
    while (nsel < topk && sel[nsel] != SOL_SENTINEL) nsel++;

    const unsigned int s_base = mma_smem_u32(mma_smem);
    const unsigned int sK[2] = { s_base, s_base + MMA_TILEB };
    const unsigned int sV[2] = { s_base + 2 * MMA_TILEB, s_base + 3 * MMA_TILEB };

    mma_load_tile(sK[0], qb, tid);
    mma_cp_commit();
    mma_cp_wait<0>();
    __syncthreads();
    unsigned int qf[8][4];
    #pragma unroll
    for (int kc = 0; kc < 8; kc++) {
        int row = warp * 16 + (lane & 15), col = kc * 16 + (lane >> 4) * 8;
        mma_ldm_x4(sK[0] + mma_swz(row, col), qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3]);
    }
    __syncthreads();

    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    const float NEG = SOL_NEG;
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;

    if (nsel > 0) {
        unsigned int kt0 = sel[0];
        mma_load_tile(sK[0], kb + (long)kt0 * MMA_TILE * MMA_DIM, tid);
        mma_load_tile(sV[0], vb + (long)kt0 * MMA_TILE * MMA_DIM, tid);
        mma_cp_commit();
    }

    for (int i = 0; i < nsel; i++) {
        const int buf = i & 1;
        if (i + 1 < nsel) {
            unsigned int kn = sel[i + 1];
            mma_load_tile(sK[buf ^ 1], kb + (long)kn * MMA_TILE * MMA_DIM, tid);
            mma_load_tile(sV[buf ^ 1], vb + (long)kn * MMA_TILE * MMA_DIM, tid);
            mma_cp_commit();
            mma_cp_wait<1>();
        } else {
            mma_cp_wait<0>();
        }
        __syncthreads();

        float s[8][4];
        #pragma unroll
        for (int n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
        #pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                unsigned int b[2];
                mma_ldm_x2(sK[buf] + mma_swz(n * 8 + (lane & 7), kc * 16 + ((lane >> 3) & 1) * 8), b[0], b[1]);
                mma_bf16(s[n], qf[kc], b);
            }
        }

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

        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            unsigned int pa[4];
            pa[0] = mma_pack_bf16(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = mma_pack_bf16(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = mma_pack_bf16(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = mma_pack_bf16(s[2 * kc + 1][2], s[2 * kc + 1][3]);
            #pragma unroll
            for (int n = 0; n < 16; n++) {
                unsigned int b[2];
                mma_ldm_x2_trans(sV[buf] + mma_swz(kc * 16 + (lane & 15), n * 8), b[0], b[1]);
                mma_bf16(o[n], pa, b);
            }
        }
        __syncthreads();
    }

    // Token-order (m, l, acc); padding slots are skipped. `m` is stored in
    // the scaled-log2 units `sol_coarse_partials` uses (score * scale_log2),
    // since `sol_lse_merge` combines the two with exp2(m_a - m_b); `l` is
    // already relative to exp2(m * scale_log2). One lane per quad (t == 0)
    // owns rows g and g+8, so every one of the warp's 16 rows is written.
    int r0 = slot_src[(long)qtile * MMA_TILE + warp * 16 + g];
    int r1 = slot_src[(long)qtile * MMA_TILE + warp * 16 + g + 8];
    if (t == 0) {
        const float ms0 = (m0 == NEG) ? NEG : m0 * scale_log2;
        const float ms1 = (m1 == NEG) ? NEG : m1 * scale_log2;
        if (r0 >= 0) { m_out[bh * seq + r0] = ms0; l_out[bh * seq + r0] = l0; }
        if (r1 >= 0) { m_out[bh * seq + r1] = ms1; l_out[bh * seq + r1] = l1; }
    }
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        int col = n * 8 + t * 2;
        if (r0 >= 0) {
            float* ob = acc_out + (bh * seq + r0) * (long)MMA_DIM + col;
            *reinterpret_cast<float2*>(ob) = make_float2(o[n][0], o[n][1]);
        }
        if (r1 >= 0) {
            float* ob = acc_out + (bh * seq + r1) * (long)MMA_DIM + col;
            *reinterpret_cast<float2*>(ob) = make_float2(o[n][2], o[n][3]);
        }
    }
#else
    (void)qt; (void)kt; (void)vt; (void)selected; (void)block_sizes; (void)slot_src;
    (void)m_out; (void)l_out; (void)acc_out;
    (void)num_tiles; (void)topk; (void)scale_log2; (void)q_base; (void)seq;
    __trap();
#endif
}
// ---- Fused Sol-Attn (port of the CuTe sm120 mainloop) ---------------------
//
// Four launches per call: sol_prep_kv (K/V f32 -> bf16 + pooled Kc mean /
// Vc sum), sol_prep_kstats (+ sol_prep_kgram for thresh_type=exact),
// sol_prep_q (Q f32 -> bf16 + per-q-block routing threshold) and
// sol_mma_fwd (route GEMM, ballot compaction, approximate PV and exact flash
// in one CTA per (64-query tile, batch*head)). Operands are bf16 BHSD
// `[bh, T, 128]`; block j of a head is the contiguous slab [64j, 64j+len(j)).
// Rounding points follow the Python reference: Q/K/V, Kc, Vc and P are bf16
// (round-to-nearest-even), every accumulation is f32.
#define SOL_LN2 0.6931471805599453f

// f32 -> bf16 bits, round-to-nearest-even (torch `.to(torch.bfloat16)`).
__device__ __forceinline__ unsigned short sol_bf16_rn(float x) {
    return fv_bf16_rne(x);
}

// P1: one CTA per (block b, head bh), thread d = channel. Reads K/V f32 once,
// writes the bf16 copies, and pools the ROUNDED values (Python pools bf16
// inputs): Kc = bf16(sum / len), Vc = bf16(sum). f32 sequential sums and an
// IEEE divide, so the host oracle reproduces Kc/Vc bit for bit.
// `in16`: k/v hold bf16 bits (bf16 activations); rounding them is the identity,
// so kb/vb (and qb in sol_prep_q) are not written and may be dummies.
__device__ __forceinline__ float sol_ld(const void* p, long i, int in16) {
    return in16 ? fv_bf16_to_f32(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}
extern "C" __global__ void sol_prep_kv(
    const void* __restrict__ k, const void* __restrict__ v,
    unsigned short* __restrict__ kb, unsigned short* __restrict__ vb,
    unsigned short* __restrict__ kc, unsigned short* __restrict__ vc,
    int T, int NT, int in16
) {
    const int b = blockIdx.x, d = threadIdx.x;
    const long bh = blockIdx.y;
    if (b >= NT || d >= 128) return;
    const int L = min(64, T - 64 * b);
    const long base = (bh * T + 64L * b) * 128 + d;
    float sk = 0.f, sv = 0.f;
    for (int r = 0; r < L; r++) {
        const long i = base + (long)r * 128;
        const unsigned short kh = sol_bf16_rn(sol_ld(k, i, in16)), vh = sol_bf16_rn(sol_ld(v, i, in16));
        // bf16 inputs: the copy would be the identity, and the forward
        // reads k/v in place, so nothing is written.
        if (!in16) { kb[i] = kh; vb[i] = vh; }
        sk = __fadd_rn(sk, fv_bf16_to_f32(kh));
        sv = __fadd_rn(sv, fv_bf16_to_f32(vh));
    }
    const long o = (bh * NT + b) * 128 + d;
    kc[o] = sol_bf16_rn(__fdiv_rn(sk, (float)L));
    vc[o] = sol_bf16_rn(sv);
}

// P2: per-head diag statistics of the bf16 Kc rows over all NT blocks:
// kstat[bh] = [mu[128], var[128]], var = max(E[x^2] - mu^2, 0).
extern "C" __global__ void sol_prep_kstats(
    const unsigned short* __restrict__ kc, float* __restrict__ kstat, int NT
) {
    const long bh = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= 128) return;
    float s = 0.f, s2 = 0.f;
    for (int j = 0; j < NT; j++) {
        const float x = fv_bf16_to_f32(kc[(bh * NT + j) * 128 + d]);
        s = __fadd_rn(s, x);
        s2 = __fmaf_rn(x, x, s2);
    }
    const float n = (float)NT;
    const float mu = __fdiv_rn(s, n);
    const float var = fmaxf(__fsub_rn(__fdiv_rn(s2, n), __fmul_rn(mu, mu)), 0.f);
    kstat[bh * 256 + d] = mu;
    kstat[bh * 256 + 128 + d] = var;
}

// P2' (thresh_type=exact only): kM[bh] = bf16(bf16(Kc^T Kc) / NT), the bf16
// matmul then in-place bf16 `div_` of `_compute_exact_threshold`. Grid
// (bh, 8), 256 threads: CTA y owns rows d in [16y, 16y+16); thread = (column
// e, 8 rows). Kc is staged 64 blocks at a time.
extern "C" __global__ void sol_prep_kgram(
    const unsigned short* __restrict__ kc, unsigned short* __restrict__ km, int NT
) {
    __shared__ float tile[64][129];
    const long bh = blockIdx.x;
    const int tid = threadIdx.x, e = tid & 127, half = tid >> 7;
    const int d0 = blockIdx.y * 16 + half * 8;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.f;
    for (int j0 = 0; j0 < NT; j0 += 64) {
        const int cnt = min(64, NT - j0);
        __syncthreads();
        for (int i = tid; i < cnt * 128; i += blockDim.x) {
            const int r = i >> 7, c = i & 127;
            tile[r][c] = fv_bf16_to_f32(kc[(bh * NT + j0 + r) * 128 + c]);
        }
        __syncthreads();
        for (int r = 0; r < cnt; r++) {
            const float xe = tile[r][e];
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = __fmaf_rn(tile[r][d0 + i], xe, acc[i]);
        }
    }
    const float n = (float)NT;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const float m = fv_bf16_to_f32(sol_bf16_rn(acc[i]));
        km[bh * 16384 + (long)(d0 + i) * 128 + e] = sol_bf16_rn(__fdiv_rn(m, n));
    }
}

// Sum of (a, c) over the 128 threads of the CTA; the total is valid in
// thread 0.
__device__ __forceinline__ void sol_block_sum2(float& a, float& c, float* red) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, off);
        c += __shfl_xor_sync(0xffffffffu, c, off);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { red[warp] = a; red[4 + warp] = c; }
    __syncthreads();
    if (threadIdx.x == 0) {
        a = (red[0] + red[1]) + (red[2] + red[3]);
        c = (red[4] + red[5]) + (red[6] + red[7]);
    }
}

// P3: Q f32 -> bf16 plus the routing threshold of query block b, in
// log2-score units. diag: mean = qbar.mu * sl2, var = (qbar^2).var * sl2^2.
// exact: qbar rounded to bf16, var = max(qbar^T M qbar - mean_raw^2, 0) * sl2^2.
// thr = mean + tau * sqrt(max(var, 0) + 1e-6).
extern "C" __global__ void sol_prep_q(
    const void* __restrict__ q, unsigned short* __restrict__ qb,
    const float* __restrict__ kstat, const unsigned short* __restrict__ km,
    float* __restrict__ thr, int T, int NT, float tau, float sl2, int exact_mode, int in16
) {
    __shared__ float red[8];
    __shared__ float sq_bar[128];
    const int b = blockIdx.x, d = threadIdx.x;
    const long bh = blockIdx.y;
    if (b >= NT || d >= 128) return;
    const int L = min(64, T - 64 * b);
    const long base = (bh * T + 64L * b) * 128 + d;
    float sq = 0.f;
    for (int r = 0; r < L; r++) {
        const long i = base + (long)r * 128;
        const unsigned short qh = sol_bf16_rn(sol_ld(q, i, in16));
        if (!in16) qb[i] = qh;
        sq = __fadd_rn(sq, fv_bf16_to_f32(qh));
    }
    float qbar = __fdiv_rn(sq, (float)L);
    const float mu = kstat[bh * 256 + d];
    float a, c;
    if (!exact_mode) {
        a = __fmul_rn(qbar, mu);
        c = __fmul_rn(__fmul_rn(qbar, qbar), kstat[bh * 256 + 128 + d]);
    } else {
        qbar = fv_bf16_to_f32(sol_bf16_rn(qbar));
        sq_bar[d] = qbar;
        __syncthreads();
        a = __fmul_rn(qbar, mu);
        // proj[d] = sum_e qbar[e] * M[e][d]: column d of `q_bar @ M`.
        const unsigned short* M = km + bh * 16384;
        float proj = 0.f;
        for (int e = 0; e < 128; e++) proj = __fmaf_rn(sq_bar[e], fv_bf16_to_f32(M[e * 128 + d]), proj);
        c = __fmul_rn(proj, qbar);
    }
    sol_block_sum2(a, c, red);
    if (d == 0) {
        const float mean = __fmul_rn(a, sl2);
        const float s2 = __fmul_rn(sl2, sl2);
        const float var = exact_mode ? __fmul_rn(fmaxf(__fsub_rn(c, __fmul_rn(a, a)), 0.f), s2)
                                     : __fmul_rn(c, s2);
        const float sd = __fsqrt_rn(__fadd_rn(fmaxf(var, 0.f), 1.0e-6f));
        thr[bh * NT + b] = __fadd_rn(mean, __fmul_rn(tau, sd));
    }
}

#define SOL_FWD_SMEM (2 * MMA_TILEB + (4 * 64 + 64) * 4 + 64 * 4 + 16)

// F: one CTA per (64-query tile, batch*head); 4 warps, warp w owns query
// rows [16w, 16w+16). Per 64-block route group: S = Q Kc^T on tensor cores,
// column means over live rows vs the threshold, local window and sink
// (ballot-compacted exact list, ascending), then the approximate term
// (pooled columns, row sum weighted by len(j), P bf16 @ Vc) and one flash
// step per exact block — the order of `sm120/mainloop.py`. Output f32 (or
// bf16) BHSD, optional natural-log LSE `[bh, T]`, optional route ballots
// `[bh, NT, 2*G]` (CuTe `debug_route_trace`). Sinks are one contiguous KV
// block range [sink_lo, sink_hi) (empty when lo == hi).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Variants of the fused forward, all the same algorithm and schedule:
//   V = 0  sol_mma_fwd      ldmatrix.x2 per mma, exp2f (the original)
//   V = 1  sol_mma_fwd_x4   ldmatrix.x4 feeding two mmas (bit-identical to 0)
//   V = 2  sol_mma_fwd_x4f  + ex2.approx.ftz, the CuTe kernel's
//                           cute.math.exp2(fastmath=True) (sm120/mainloop.py
//                           :1061-1066, :1104-1110)
template <int V> __device__ __forceinline__ void sol_qk(unsigned int sK, const unsigned int (&qf)[8][4], float (&s)[8][4], int lane) {
    if constexpr (V >= 1) fa_qk<128>(sK, qf, s, lane); else mma_qk_64x64(sK, qf, s, lane);
}
template <int V> __device__ __forceinline__ void sol_pv(unsigned int sV, const unsigned int (&pa)[4][4], float (&o)[16][4], int lane) {
    if constexpr (V >= 1) fa_pv<128>(sV, pa, o, lane); else mma_pv_64x128(sV, pa, o, lane);
}
template <int V> __device__ __forceinline__ float sol_ex2(float x) {
    if constexpr (V >= 2) return fa_exp2(x); else return exp2f(x);
}

// PISA mode (`PISA = true`, `pisa_mma_fwd`): the same mainloop with the
// exact set of query tile qt read from a top-k bitmap `pisa_sel`
// `[bh, NT, 2*G]` (bit b of word 2g + w = KV block 64g + 32w + b selected)
// instead of the threshold / local / sink rule; the remainder keeps the
// zeroth-order term (pooled columns, len(j)-weighted row sum, P @ Vc) and
// also sums its unweighted probabilities into `tail`. The epilogue adds the
// paper's Phase-3 first-order term tail * (q @ H_bar) on tensor cores, with
// H_bar `pisa_h` `[bh, 128, 128]` bf16, before the 1/l normalisation:
// `fastvideo_models::pisa_attn::pisa_attn_head` in one CTA per query tile.
__device__ __forceinline__ unsigned int pisa_scale_bf16x2(unsigned int x, float s) {
    return mma_pack_bf16_rn(__uint_as_float(x << 16) * s, __uint_as_float(x & 0xffff0000u) * s);
}

template <int V, bool PISA = false>
__device__ __forceinline__ void sol_mma_fwd_body(
    const unsigned short* __restrict__ qb, const unsigned short* __restrict__ kb,
    const unsigned short* __restrict__ vb,
    const unsigned short* __restrict__ kc, const unsigned short* __restrict__ vc,
    const float* __restrict__ thr,
    float* __restrict__ out, unsigned short* __restrict__ out_bf16, int out_is_bf16,
    float* __restrict__ lse, int has_lse,
    unsigned int* __restrict__ route_dbg, int has_dbg,
    int T, int NT, int sink_lo, int sink_hi, float sl2
,
    unsigned char* smem,
    const unsigned int* __restrict__ pisa_sel = nullptr,
    const unsigned short* __restrict__ pisa_h = nullptr,
    float pisa_fo = 0.f
) {
    float* colsum = reinterpret_cast<float*>(smem + 2 * MMA_TILEB);   // [4][64]
    float* colmask = colsum + 4 * 64;                                // [64]
    int* eidx = reinterpret_cast<int*>(colmask + 64);                // [64]
    int* meta = eidx + 64;                                           // [1]
    const unsigned int sK = mma_smem_u32(smem), sV = sK + MMA_TILEB;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int qt = blockIdx.x;
    const long bh = blockIdx.y;
    if (qt >= NT) return;
    const int q0 = qt * 64, qlen = min(64, T - q0);
    const unsigned short* Qh = qb + bh * (long)T * MMA_DIM;
    const unsigned short* Kh = kb + bh * (long)T * MMA_DIM;
    const unsigned short* Vh = vb + bh * (long)T * MMA_DIM;
    const unsigned short* KCh = kc + bh * (long)NT * MMA_DIM;
    const unsigned short* VCh = vc + bh * (long)NT * MMA_DIM;
    const float th = PISA ? 0.f : thr[bh * NT + qt];
    const int G = (NT + 63) >> 6;
    const unsigned int* sel_row = PISA ? pisa_sel + (bh * NT + qt) * 2L * G : nullptr;
    const float NEG = __int_as_float(0xff800000);

    // Q -> registers through sK (rows >= qlen are zeros).
    mma_load_tile_rows(sK, Qh + (long)q0 * MMA_DIM, qlen, tid);
    mma_cp_commit();
    mma_cp_wait<0>();
    __syncthreads();
    unsigned int qf[8][4];
    #pragma unroll
    for (int k8 = 0; k8 < 8; k8++) {
        int row = warp * 16 + (lane & 15), col = k8 * 16 + (lane >> 4) * 8;
        mma_ldm_x4(sK + mma_swz(row, col), qf[k8][0], qf[k8][1], qf[k8][2], qf[k8][3]);
    }
    __syncthreads();

    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool ok0 = row0 < qlen, ok1 = row1 < qlen;
    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;   // m raw units; l per-thread partials
    float tl0 = 0.f, tl1 = 0.f;                      // PISA: unweighted remainder mass
    float s[8][4];
    unsigned int pa[4][4];

    // cp.async groups: every tile load is its own group, committed in the
    // order it is consumed, so wait<1> = "all but the newest have landed".
    bool kc_pending = false;
    #pragma unroll 1
    for (int gi = 0; gi < G; gi++) {
        const int gs = gi * 64, vb_cnt = min(64, NT - gs);
        // (a) route tiles: Kc (unless prefetched) then Vc.
        if (!kc_pending) {
            mma_load_tile_rows(sK, KCh + (long)gs * MMA_DIM, vb_cnt, tid);
            mma_cp_commit();
        }
        mma_load_tile_rows(sV, VCh + (long)gs * MMA_DIM, vb_cnt, tid);
        mma_cp_commit();
        kc_pending = false;
        mma_cp_wait<1>();
        __syncthreads();

        // (b) raw route scores S = Q Kc^T.
        sol_qk<V>(sK, qf, s, lane);

        // (c) column sums over live rows (reduce_route_columns); PISA routes
        // from its bitmap instead.
        if constexpr (!PISA) {
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
        }
        __syncthreads();

        // (d) route decision + ascending ballot compaction (warp 0).
        if (warp == 0) {
            int base = 0;
            #pragma unroll
            for (int word = 0; word < 2; word++) {
                const int off = word * 32 + lane, kbk = gs + off;
                const bool valid = off < vb_cnt;
                bool ex = false;
                if constexpr (PISA) {
                    ex = valid && ((sel_row[2 * gi + word] >> lane) & 1u);
                } else if (valid) {
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
        __syncthreads();
        const int ne = meta[0];
        const bool has_approx = ne < vb_cnt;

        // (e) sK is free (S lives in registers): prefetch the first exact K
        // tile, or the next group's Kc when this group has no exact block.
        bool issued = false;
        if (ne > 0) {
            const int j0 = eidx[0];
            mma_load_tile_rows(sK, Kh + (long)j0 * 64 * MMA_DIM, min(64, T - 64 * j0), tid);
            mma_cp_commit();
            issued = true;
        } else if (gi + 1 < G) {
            mma_load_tile_rows(sK, KCh + (long)(gs + 64) * MMA_DIM, min(64, NT - gs - 64), tid);
            mma_cp_commit();
            kc_pending = true;
            issued = true;
        }
        if (issued) mma_cp_wait<1>(); else mma_cp_wait<0>();   // Vc landed
        __syncthreads();

        // (f) approximate term: pooled columns, len(j)-weighted row sum.
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
            const float a0 = sol_ex2<V>((m0 - sf0) * sl2), a1 = sol_ex2<V>((m1 - sf1) * sl2);
            const float ms0 = sf0 * sl2, ms1 = sf1 * sl2;
            m0 = mn0; m1 = mn1;
            float ls0 = 0.f, ls1 = 0.f, ts0 = 0.f, ts1 = 0.f;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                const int c0 = n * 8 + 2 * t;
                const float w0 = (float)max(0, min(64, T - 64 * (gs + c0)));
                const float w1 = (float)max(0, min(64, T - 64 * (gs + c0 + 1)));
                s[n][0] = sol_ex2<V>(s[n][0] * sl2 - ms0);
                s[n][1] = sol_ex2<V>(s[n][1] * sl2 - ms0);
                s[n][2] = sol_ex2<V>(s[n][2] * sl2 - ms1);
                s[n][3] = sol_ex2<V>(s[n][3] * sl2 - ms1);
                ls0 += s[n][0] * w0 + s[n][1] * w1;
                ls1 += s[n][2] * w0 + s[n][3] * w1;
                if constexpr (PISA) {
                    ts0 += s[n][0] + s[n][1];
                    ts1 += s[n][2] + s[n][3];
                }
            }
            l0 = l0 * a0 + ls0;
            l1 = l1 * a1 + ls1;
            if constexpr (PISA) {
                tl0 = tl0 * a0 + ts0;
                tl1 = tl1 * a1 + ts1;
            }
            #pragma unroll
            for (int n = 0; n < 16; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
            #pragma unroll
            for (int k4 = 0; k4 < 4; k4++) {
                pa[k4][0] = mma_pack_bf16_rn(s[2 * k4][0], s[2 * k4][1]);
                pa[k4][1] = mma_pack_bf16_rn(s[2 * k4][2], s[2 * k4][3]);
                pa[k4][2] = mma_pack_bf16_rn(s[2 * k4 + 1][0], s[2 * k4 + 1][1]);
                pa[k4][3] = mma_pack_bf16_rn(s[2 * k4 + 1][2], s[2 * k4 + 1][3]);
            }
            sol_pv<V>(sV, pa, o, lane);
        }
        __syncthreads();   // every warp is done with sV (Vc)

        // (g) exact blocks, ascending. Pending groups at the two waits are
        // [K_e, V_e] and then [V_e, next K or next Kc].
        if (ne > 0) {
            const int j0 = eidx[0];
            mma_load_tile_rows(sV, Vh + (long)j0 * 64 * MMA_DIM, min(64, T - 64 * j0), tid);
            mma_cp_commit();
        }
        #pragma unroll 1
        for (int e = 0; e < ne; e++) {
            const int j = eidx[e];
            const int Lj = min(64, T - 64 * j);
            mma_cp_wait<1>();
            __syncthreads();   // K_e landed
            sol_qk<V>(sK, qf, s, lane);
            __syncthreads();   // every warp is done reading sK
            bool next = false;
            if (e + 1 < ne) {
                const int jn = eidx[e + 1];
                mma_load_tile_rows(sK, Kh + (long)jn * 64 * MMA_DIM, min(64, T - 64 * jn), tid);
                mma_cp_commit();
                next = true;
            } else if (gi + 1 < G) {
                mma_load_tile_rows(sK, KCh + (long)(gs + 64) * MMA_DIM, min(64, NT - gs - 64), tid);
                mma_cp_commit();
                kc_pending = true;
                next = true;
            }
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
            const float a0 = sol_ex2<V>((m0 - sf0) * sl2), a1 = sol_ex2<V>((m1 - sf1) * sl2);
            const float ms0 = sf0 * sl2, ms1 = sf1 * sl2;
            m0 = mn0; m1 = mn1;
            float ls0 = 0.f, ls1 = 0.f;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                s[n][0] = sol_ex2<V>(s[n][0] * sl2 - ms0);
                s[n][1] = sol_ex2<V>(s[n][1] * sl2 - ms0);
                s[n][2] = sol_ex2<V>(s[n][2] * sl2 - ms1);
                s[n][3] = sol_ex2<V>(s[n][3] * sl2 - ms1);
                ls0 += s[n][0] + s[n][1];
                ls1 += s[n][2] + s[n][3];
            }
            l0 = l0 * a0 + ls0;
            l1 = l1 * a1 + ls1;
            if constexpr (PISA) { tl0 *= a0; tl1 *= a1; }
            #pragma unroll
            for (int n = 0; n < 16; n++) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
            #pragma unroll
            for (int k4 = 0; k4 < 4; k4++) {
                pa[k4][0] = mma_pack_bf16_rn(s[2 * k4][0], s[2 * k4][1]);
                pa[k4][1] = mma_pack_bf16_rn(s[2 * k4][2], s[2 * k4][3]);
                pa[k4][2] = mma_pack_bf16_rn(s[2 * k4 + 1][0], s[2 * k4 + 1][1]);
                pa[k4][3] = mma_pack_bf16_rn(s[2 * k4 + 1][2], s[2 * k4 + 1][3]);
            }
            if (next) mma_cp_wait<1>(); else mma_cp_wait<0>();
            __syncthreads();   // V_e landed
            sol_pv<V>(sV, pa, o, lane);
            __syncthreads();   // every warp is done reading sV
            if (e + 1 < ne) {
                const int jn = eidx[e + 1];
                mma_load_tile_rows(sV, Vh + (long)jn * 64 * MMA_DIM, min(64, T - 64 * jn), tid);
                mma_cp_commit();
            }
        }
    }

    // (h) finalize: quad-reduce l, normalise, natural-log LSE.
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if constexpr (PISA) if (pisa_fo != 0.f) {
        // First-order term: o += diag(tail * s) Q H_bar, s = `pisa_fo`, the
        // softmax scale (the Taylor step of the logit s q.k; 0 = off).
        // H_bar's 128 rows fill sK (rows 0-63) and sV (rows 64-127); Q's A fragments, scaled per
        // row by tail (a0/a2 row g, a1/a3 row g + 8), are the P operand of
        // two P @ V mmas over the two 64-row halves of the contraction.
        tl0 += __shfl_xor_sync(0xffffffffu, tl0, 1);
        tl0 += __shfl_xor_sync(0xffffffffu, tl0, 2);
        tl1 += __shfl_xor_sync(0xffffffffu, tl1, 1);
        tl1 += __shfl_xor_sync(0xffffffffu, tl1, 2);
        tl0 *= pisa_fo;
        tl1 *= pisa_fo;
        mma_cp_wait<0>();
        __syncthreads();
        const unsigned short* Hh = pisa_h + bh * (long)MMA_DIM * MMA_DIM;
        mma_load_tile_rows(sK, Hh, 64, tid);
        mma_load_tile_rows(sV, Hh + 64L * MMA_DIM, 64, tid);
        mma_cp_commit();
        mma_cp_wait<0>();
        __syncthreads();
        unsigned int qa[2][4][4];
        #pragma unroll
        for (int k8 = 0; k8 < 8; k8++) {
            qa[k8 >> 2][k8 & 3][0] = pisa_scale_bf16x2(qf[k8][0], tl0);
            qa[k8 >> 2][k8 & 3][1] = pisa_scale_bf16x2(qf[k8][1], tl1);
            qa[k8 >> 2][k8 & 3][2] = pisa_scale_bf16x2(qf[k8][2], tl0);
            qa[k8 >> 2][k8 & 3][3] = pisa_scale_bf16x2(qf[k8][3], tl1);
        }
        sol_pv<V>(sK, qa[0], o, lane);
        sol_pv<V>(sV, qa[1], o, lane);
    }
    const bool bad0 = (l0 == 0.f) || ((__float_as_uint(l0) & 0x7fffffffu) > 0x7f800000u);
    const bool bad1 = (l1 == 0.f) || ((__float_as_uint(l1) & 0x7fffffffu) > 0x7f800000u);
    const float inv0 = bad0 ? 1.f : 1.f / l0, inv1 = bad1 ? 1.f : 1.f / l1;
    const long r0 = bh * (long)T + q0 + row0, r1 = r0 + 8;
    if (has_lse && t == 0) {
        if (ok0) lse[r0] = bad0 ? NEG : (m0 * sl2 + log2f(l0)) * SOL_LN2;
        if (ok1) lse[r1] = bad1 ? NEG : (m1 * sl2 + log2f(l1)) * SOL_LN2;
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
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define SOL_FWD_ENTRY(V)                                                                 \
    __shared__ __align__(128) unsigned char smem[SOL_FWD_SMEM];                         \
    sol_mma_fwd_body<V>(qb, kb, vb, kc, vc, thr, out, out_bf16, out_is_bf16, lse, has_lse, route_dbg, has_dbg, T, NT, sink_lo, sink_hi, sl2, smem);
#else
#define SOL_FWD_ENTRY(V)                                                                 \
    (void)qb; (void)kb; (void)vb; (void)kc; (void)vc; (void)thr; (void)out; (void)out_bf16; \
    (void)out_is_bf16; (void)lse; (void)has_lse; (void)route_dbg; (void)has_dbg; \
    (void)T; (void)NT; (void)sink_lo; (void)sink_hi; (void)sl2; \
    __trap();
#endif

extern "C" __global__ void __launch_bounds__(128, 2) sol_mma_fwd(
    const unsigned short* __restrict__ qb, const unsigned short* __restrict__ kb,
    const unsigned short* __restrict__ vb,
    const unsigned short* __restrict__ kc, const unsigned short* __restrict__ vc,
    const float* __restrict__ thr,
    float* __restrict__ out, unsigned short* __restrict__ out_bf16, int out_is_bf16,
    float* __restrict__ lse, int has_lse,
    unsigned int* __restrict__ route_dbg, int has_dbg,
    int T, int NT, int sink_lo, int sink_hi, float sl2

) {
    SOL_FWD_ENTRY(0)
}

extern "C" __global__ void __launch_bounds__(128, 2) sol_mma_fwd_x4(
    const unsigned short* __restrict__ qb, const unsigned short* __restrict__ kb,
    const unsigned short* __restrict__ vb,
    const unsigned short* __restrict__ kc, const unsigned short* __restrict__ vc,
    const float* __restrict__ thr,
    float* __restrict__ out, unsigned short* __restrict__ out_bf16, int out_is_bf16,
    float* __restrict__ lse, int has_lse,
    unsigned int* __restrict__ route_dbg, int has_dbg,
    int T, int NT, int sink_lo, int sink_hi, float sl2

) {
    SOL_FWD_ENTRY(1)
}

extern "C" __global__ void __launch_bounds__(128, 2) sol_mma_fwd_x4f(
    const unsigned short* __restrict__ qb, const unsigned short* __restrict__ kb,
    const unsigned short* __restrict__ vb,
    const unsigned short* __restrict__ kc, const unsigned short* __restrict__ vc,
    const float* __restrict__ thr,
    float* __restrict__ out, unsigned short* __restrict__ out_bf16, int out_is_bf16,
    float* __restrict__ lse, int has_lse,
    unsigned int* __restrict__ route_dbg, int has_dbg,
    int T, int NT, int sink_lo, int sink_hi, float sl2

) {
    SOL_FWD_ENTRY(2)
}

// PISA score route, fused (see the PISA mode of `sol_mma_fwd_body`): one
// CTA per (64-query tile, batch*head), the x4f schedule (ex2.approx), bf16
// operands from sol_prep_kv / sol_prep_q.
extern "C" __global__ void __launch_bounds__(128, 2) pisa_mma_fwd(
    const unsigned short* __restrict__ qb, const unsigned short* __restrict__ kb,
    const unsigned short* __restrict__ vb,
    const unsigned short* __restrict__ kc, const unsigned short* __restrict__ vc,
    const unsigned int* __restrict__ sel, const unsigned short* __restrict__ hbar,
    float* __restrict__ out, unsigned short* __restrict__ out_bf16, int out_is_bf16,
    int T, int NT, float sl2, float fo
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(128) unsigned char smem[SOL_FWD_SMEM];
    sol_mma_fwd_body<2, true>(qb, kb, vb, kc, vc, nullptr, out, out_bf16, out_is_bf16,
        nullptr, 0, nullptr, 0, T, NT, NT, NT, sl2, smem, sel, hbar, fo);
#else
    (void)qb; (void)kb; (void)vb; (void)kc; (void)vc; (void)sel; (void)hbar; (void)out;
    (void)out_bf16; (void)out_is_bf16; (void)T; (void)NT; (void)sl2; (void)fo;
    __trap();
#endif
}

// PISA top-k lists `[rows, keep]` (vsa_topk) -> selection bitmap
// `[rows, words]` (zeroed by the caller): bit j % 32 of word j / 32 = block j.
extern "C" __global__ void pisa_sel_bits(
    const unsigned int* __restrict__ lists, unsigned int* __restrict__ bits,
    long rows, int keep, int words
) {
    const long row = blockIdx.x;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < keep; i += blockDim.x) {
        const unsigned int j = lists[row * keep + i];
        if (j != SOL_SENTINEL && (int)(j >> 5) < words) atomicOr(bits + row * words + (j >> 5), 1u << (j & 31));
    }
}

// PISA global first-order statistic H_bar = (1/n) sum_t (k_t - kc_{t/64}) v_t^T
// (`fastvideo_models::pisa_attn::global_h_bar`), stage 1: CTA (c, bh) sums
// the tokens of blocks [16c, 16c + 16) into part[bh][c] `[128, 128]` f32.
// 256 threads, each owning 8 rows e x 8 columns d; 32 tokens staged per pass.
// (Replaces sol_global_h_bar's one-thread-per-cell loop over every token.)
#define PISA_H_BLOCKS 16
extern "C" __global__ void __launch_bounds__(256) pisa_h_partial(
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ kc,
    float* __restrict__ part, long seq, int n, int chunks
) {
    __shared__ __align__(16) float sdk[32][128];
    __shared__ __align__(16) float sv[32][128];
    const int c = blockIdx.x, tid = threadIdx.x;
    const long bh = blockIdx.y;
    if (c >= chunks) return;
    const int e0 = (tid >> 4) * 8, d0 = (tid & 15) * 8;
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
    }
    const long t_begin = (long)c * PISA_H_BLOCKS * 64;
    const long t_end = min(seq, t_begin + PISA_H_BLOCKS * 64L);
    for (long t0 = t_begin; t0 < t_end; t0 += 32) {
        const int rows = (int)min(32L, t_end - t0);
        __syncthreads();
        for (int i = tid; i < 32 * 32; i += 256) {
            const int r = i >> 5, c4 = (i & 31) * 4;
            float4 dk = make_float4(0.f, 0.f, 0.f, 0.f), vv = dk;
            if (r < rows) {
                const long tok = t0 + r;
                const long idx = (bh * seq + tok) * 128 + c4;
                const float4 kk = *reinterpret_cast<const float4*>(k + idx);
                const float4 mm = *reinterpret_cast<const float4*>(kc + (bh * n + tok / 64) * 128 + c4);
                dk = make_float4(kk.x - mm.x, kk.y - mm.y, kk.z - mm.z, kk.w - mm.w);
                vv = *reinterpret_cast<const float4*>(v + idx);
            }
            *reinterpret_cast<float4*>(&sdk[r][c4]) = dk;
            *reinterpret_cast<float4*>(&sv[r][c4]) = vv;
        }
        __syncthreads();
        for (int r = 0; r < rows; r++) {
            const float4 a0 = *reinterpret_cast<const float4*>(&sdk[r][e0]);
            const float4 a1 = *reinterpret_cast<const float4*>(&sdk[r][e0 + 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&sv[r][d0]);
            const float4 b1 = *reinterpret_cast<const float4*>(&sv[r][d0 + 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
        }
    }
    float* out = part + (bh * chunks + c) * 16384L;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        *reinterpret_cast<float4*>(out + (e0 + i) * 128 + d0) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(out + (e0 + i) * 128 + d0 + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

// Stage 2: H_bar[bh] = bf16((sum over chunks, in order) / n), row e, column d.
extern "C" __global__ void pisa_h_reduce(
    const float* __restrict__ part, unsigned short* __restrict__ hbar, int chunks, int n
) {
    const long bh = blockIdx.y;
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= 16384) return;
    float acc = 0.f;
    for (int c = 0; c < chunks; c++) acc += part[(bh * chunks + c) * 16384L + cell];
    hbar[bh * 16384L + cell] = sol_bf16_rn(n > 0 ? acc / (float)n : 0.f);
}
// ==== end region: sol ====

