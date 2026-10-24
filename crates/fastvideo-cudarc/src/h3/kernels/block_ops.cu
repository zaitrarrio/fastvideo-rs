// ---- H3 fused block ops (Sol-H3 fusions.py / mxfp8.py rounding points) ----
// The AdaLN table is f32 `[T, 6, dim]` whose SCALE entries already hold
// `1 + scale` (f32 addition of the bf16 projection output — the same f32 the
// reference forms in-kernel). Row `r` reads table row `idx[r] + base`.
// One block per sequence row. `mx` = 1 writes MXFP8 (codes + swizzled E8M0
// scales, `dim % 32 == 0`) instead of bf16 `out`.

// normed = RMSNorm(x) * w * (1 + scale) + shift, f32 throughout, one bf16
// rounding (`_rmsnorm_modulate_kernel`); the MX variant quantizes that bf16
// value (`_rmsnorm_modulate_mxfp8_kernel`).
extern "C" __global__ void h3_norm_mod(
    const void* x, int x16, const float* w, const float* tab, const unsigned int* idx, long base,
    int scale_slot, int shift_slot, unsigned short* out, unsigned char* q, unsigned char* qs,
    int rows, int dim, float eps, int mx
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    long xb = (long)row * dim;
    long tb = ((long)idx[row] + base) * 6;
    const float* sc = tab + (tb + scale_slot) * dim;
    const float* sh = tab + (tb + shift_slot) * dim;
    // The normalizer is bit-reproducible on the host (`quant::row_rsqrt`):
    // explicit FMA partial sums per lane, the fixed tree of fv_block_sum, a
    // correctly rounded rsqrt. The output is `normed * scale + shift`, so where
    // the two nearly cancel an f32-ulp change of `r` becomes many bf16 ulps of
    // the result; an unspecified order or rsqrt.approx would make that noise.
    float local = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = fv_ld(x, xb + j, x16);
        local = fmaf(v, v, local);
    }
    float r = __frsqrt_rn(fv_block_sum(local, sdata) / (float)dim + eps);
    long column_blocks = ((long)dim / 32 + 3) / 4;
    for (int j0 = 0; j0 < dim; j0 += blockDim.x) {
        int j = j0 + threadIdx.x;
        if (mx && (j0 + (int)(threadIdx.x & ~31u)) >= dim) continue;
        float v = 0.0f;
        if (j < dim) v = fv_r16(fmaf(fv_ld(x, xb + j, x16) * r * w[j], sc[j], sh[j]));
        if (mx) fv_mx_store_group(v, row, j, dim, q, qs, column_blocks);
        else if (j < dim) out[xb + j] = fv_bf16_rne(v);
    }
}
// hidden = residual + gate * branch (one f32 FMA, stored bf16), then the
// norm+modulate above on the *unrounded* f32 hidden
// (`_residual_gate_rmsnorm_modulate_kernel` and its MXFP8 twin).
extern "C" __global__ void h3_res_gate_norm_mod(
    const void* res, int r16, const void* br, int b16, const float* w, const float* tab,
    const unsigned int* idx, long base, int gate_slot, int scale_slot, int shift_slot,
    unsigned short* hidden, unsigned short* out, unsigned char* q, unsigned char* qs,
    int rows, int dim, float eps, int mx
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    long xb = (long)row * dim;
    long tb = ((long)idx[row] + base) * 6;
    const float* g = tab + (tb + gate_slot) * dim;
    const float* sc = tab + (tb + scale_slot) * dim;
    const float* sh = tab + (tb + shift_slot) * dim;
    float local = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float h = fmaf(g[j], fv_ld(br, xb + j, b16), fv_ld(res, xb + j, r16));
        local = fmaf(h, h, local);
    }
    // Same reproducible normalizer as h3_norm_mod.
    float r = __frsqrt_rn(fv_block_sum(local, sdata) / (float)dim + eps);
    long column_blocks = ((long)dim / 32 + 3) / 4;
    // `hidden` may alias `res` (in-place residual): every read of row `row`
    // happens above or in the same thread before its write below.
    for (int j0 = 0; j0 < dim; j0 += blockDim.x) {
        int j = j0 + threadIdx.x;
        if (mx && (j0 + (int)(threadIdx.x & ~31u)) >= dim) continue;
        float v = 0.0f;
        if (j < dim) {
            float h = fmaf(g[j], fv_ld(br, xb + j, b16), fv_ld(res, xb + j, r16));
            hidden[xb + j] = fv_bf16_rne(h);
            v = fv_r16(fmaf(h * r * w[j], sc[j], sh[j]));
        }
        if (mx) fv_mx_store_group(v, row, j, dim, q, qs, column_blocks);
        else if (j < dim) out[xb + j] = fv_bf16_rne(v);
    }
}
// The block's last residual as eager torch runs it (fusion_install.py):
// residual + bf16(gate[idx] * branch), rounded again.
extern "C" __global__ void h3_gate_residual(
    const void* res, int r16, const void* br, int b16, const float* tab, const unsigned int* idx,
    long base, int gate_slot, void* out, int o16, long n, long dim, int round_product
) {
    long i = IDX();
    if (i >= n) return;
    long row = i / dim;
    long j = i - row * dim;
    float p = tab[(((long)idx[row] + base) * 6 + gate_slot) * dim + j] * fv_ld(br, i, b16);
    if (round_product) p = fv_r16(p);
    fv_st(out, i, o16, fv_ld(res, i, r16) + p);
}
// Per-head RMSNorm over head_dim then partial rotate_half RoPE, gathered from
// the fused QKV projection straight into BHSD (`_qknorm_partial_rope_kernel`):
// f32 normalizer shared by a channel and its rotary partner, f32 cos/sin,
// one rounding. `dpad` threads (a multiple of 32, >= d) per (batch, seq,
// head) row, `blockDim.x / dpad` rows per block (the launcher uses one; the
// per-row sum is the same warp butterfly, then the row's warps in order,
// either way). Shared memory holds one float per warp.
extern "C" __global__ void h3_qk_norm_rope(
    const void* src, int s16, const float* w, const float* cs, const float* sn, int use_rope,
    void* out, int o16, int batch, int seq, int heads, int d, int r, int src_width, int col_off,
    float eps, int dpad
) {
    extern __shared__ float sdata[];
    int slot = threadIdx.x / dpad;
    int wpr = dpad >> 5;
    long row = (long)blockIdx.x * (blockDim.x / dpad) + slot;
    bool live = row < (long)batch * seq * heads;
    if (!live) row = 0;
    int h = (int)(row % heads);
    long bs = row / heads;
    int s = (int)(bs % seq);
    int b = (int)(bs / seq);
    int p = threadIdx.x - slot * dpad;
    const long xb = bs * src_width + col_off + (long)h * d;
    float v = (live && p < d) ? fv_ld(src, xb + p, s16) : 0.0f;
    float sq = v * v;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((p & 31) == 0) sdata[threadIdx.x >> 5] = sq;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < wpr; ++k) tot += sdata[slot * wpr + k];
    float rr = rsqrtf(tot / (float)d + eps);
    if (!live || p >= d) return;
    float normed = v * rr * w[p];
    float o = normed;
    if (use_rope && p < r) {
        int half = r / 2;
        int partner = p < half ? p + half : p - half;
        float pn = fv_ld(src, xb + partner, s16) * rr * w[partner];
        float rot = p < half ? -pn : pn;
        o = normed * cs[(long)s * r + p] + rot * sn[(long)s * r + p];
    }
    long oi = (((long)b * heads + h) * seq + s) * d + p;
    // o16 == 2: the bf16 value stored widened, for an f32 consumer (VSA).
    if (o16 == 2) ((float*)out)[oi] = fv_r16(o);
    else fv_st(out, oi, o16, o);
}
// SwiGLU straight to MXFP8 (`_swiglu_mxfp8_kernel`): value * silu(gate) in f32,
// rounded to bf16, then block-quantized. One block per row; `half % 32 == 0`.
extern "C" __global__ void h3_swiglu_mx(
    const void* x, int x16, unsigned char* q, unsigned char* qs, int rows, int half
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    long xb = (long)row * 2 * half;
    long column_blocks = ((long)half / 32 + 3) / 4;
    for (int j0 = 0; j0 < half; j0 += blockDim.x) {
        if ((j0 + (int)(threadIdx.x & ~31u)) >= half) continue;
        int j = j0 + threadIdx.x;
        float v = fv_ld(x, xb + j, x16);
        float g = fv_ld(x, xb + half + j, x16);
        float sg = 1.0f / (1.0f + expf(-g));
        fv_mx_store_group(fv_r16(v * (g * sg)), row, j, half, q, qs, column_blocks);
    }
}
// Row-major [rows, k] activation (f32 or bf16) to MXFP8 (`_mxfp8_quant_kernel`):
// values are read as stored (bf16 in the reference). Also quantizes weights.
extern "C" __global__ void mxfp8_quantize(
    const void* x, int x16, unsigned char* q, unsigned char* qs, int rows, int k
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    long column_blocks = ((long)k / 32 + 3) / 4;
    for (int j0 = 0; j0 < k; j0 += blockDim.x) {
        if ((j0 + (int)(threadIdx.x & ~31u)) >= k) continue;
        int j = j0 + threadIdx.x;
        fv_mx_store_group(fv_ld(x, (long)row * k + j, x16), row, j, k, q, qs, column_blocks);
    }
}
// Inverse for checks: q * 2^(byte - 127).
extern "C" __global__ void mxfp8_dequantize(
    const unsigned char* q, const unsigned char* qs, float* out, long rows, long k
) {
    long i = IDX();
    if (i >= rows * k) return;
    long row = i / k;
    long col = i - row * k;
    unsigned char byte = qs[fv_mx_scale_offset(row, col / 32, (k / 32 + 3) / 4)];
    out[i] = fv_e4m3_to_f32(q[i]) * ldexpf(1.0f, (int)byte - 127);
}
// max |x| over a (possibly bf16) tensor; `out` zeroed by the caller. NaN wins.
extern "C" __global__ void amax_abs_mixed(const void* a, int a16, float* out, long n) {
    extern __shared__ float sm[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (long i = blockIdx.x * (long)blockDim.x + tid; i < n; i += (long)gridDim.x * blockDim.x) {
        float v = fabsf(fv_ld(a, i, a16));
        if (!(v <= acc)) acc = v;
    }
    sm[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { float o = sm[tid + s]; if (!(o <= sm[tid])) sm[tid] = o; }
        __syncthreads();
    }
    if (tid == 0) atomicMax((int*)out, __float_as_int(sm[0]));
}
// FastVideo `fp8_config._quantize_tensorwise` (the W8A8 recipe), for
// activations and weights alike: scale = max(amax / 448, 1 / (448 * 512)) in
// f32; codes = e4m3(clamp(bf16(x / bf16(scale)), +-448)). The GEMM dequantizes
// with the f32 scale, as `torch._scaled_mm` does. Element `i >= n_valid` (row
// padding) is written as zero. `scale_out` receives the f32 scale.
extern "C" __global__ void w8a8_quantize(
    const void* x, int x16, const float* amax, unsigned char* q, float* scale_out,
    long n_valid, long n_total
) {
    long i = IDX();
    float s = fmaxf(*amax / 448.0f, (float)(1.0 / 229376.0));
    if (i == 0) *scale_out = s;
    if (i >= n_total) return;
    if (i >= n_valid) { q[i] = 0; return; }
    float s16 = fv_r16(s);
    float v = fv_r16(fv_ld(x, i, x16) / s16);
    v = fminf(448.0f, fmaxf(-448.0f, v));
    q[i] = fv_e4m3_sat(v);
}
// E4M3 codes * per-tensor scale -> f32 (checks and host-less inspection).
extern "C" __global__ void w8a8_dequantize(
    const unsigned char* q, const float* scale, float* out, long n
) {
    long i = IDX();
    if (i >= n) return;
    out[i] = fv_e4m3_to_f32(q[i]) * *scale;
}
// ==== endregion: bf16 activations + reference FP8 recipes ====
