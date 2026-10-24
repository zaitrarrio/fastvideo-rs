// ==== region: Wan block (bf16 activations) ====
// The Wan DiT block under FASTVIDEO_BF16_ACT with FastVideo's rounding points
// (fastvideo/models/wan/transformer.py WanTransformerBlock.forward and
// layers/layernorm.py): every op below is spelled with explicit _rn
// intrinsics, so no FMA contraction can make the fused kernels differ from
// the unfused pair that stores the intermediate (FASTVIDEO_WAN_FUSE=0).
//
//   res_mode 0  self-attn residual: hidden = h + a * gate in f32 (bf16 x f32
//               promotes), stored rounded; the norm reads the *unrounded* f32
//               sum (ScaleResidualLayerNormScaleShift, compute_dtype f32).
//   res_mode 1  cross-attn residual: hidden = bf16(h + a); the norm reads the
//               stored bf16 value.
//   ln_mode 0   LayerNorm with f32 affine (norm2), one rounding.
//   ln_mode 1   bf16(bf16(LN(x)) * (1 + scale) + shift): FP32LayerNorm
//               casts back to bf16 before the f32 modulation.

__device__ __forceinline__ float wan_res_val(
    const void* h, int h16, const void* a, int a16, const void* e, int e16,
    long i, long gi, int res_mode
) {
    const float hv = fv_ld(h, i, h16), av = fv_ld(a, i, a16);
    if (res_mode == 0) return __fadd_rn(hv, __fmul_rn(av, fv_ld(e, gi, e16)));
    return fv_r16(__fadd_rn(hv, av));
}

// Two-pass LayerNorm statistics over a row whose values `val(j)` yields.
template <typename F>
__device__ __forceinline__ void wan_ln_stats(F val, int dim, float eps, float* sdata, float* mean, float* inv) {
    float local = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) local = __fadd_rn(local, val(j));
    const float m = __fdiv_rn(fv_block_sum(local, sdata), (float)dim);
    local = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        const float dv = __fsub_rn(val(j), m);
        local = __fadd_rn(local, __fmul_rn(dv, dv));
    }
    *mean = m;
    *inv = rsqrtf(__fadd_rn(__fdiv_rn(fv_block_sum(local, sdata), (float)dim), eps));
}

__device__ __forceinline__ float wan_ln_out(
    float v, float mean, float inv, int ln_mode, int j,
    const void* w, const void* b, int p16, const void* e, int e16, long sc, long sh
) {
    float n = __fmul_rn(__fsub_rn(v, mean), inv);
    if (ln_mode == 0) return __fadd_rn(__fmul_rn(n, fv_ld(w, j, p16)), fv_ld(b, j, p16));
    n = fv_r16(n);
    const float s1 = __fadd_rn(1.0f, fv_ld(e, sc + j, e16));
    return __fadd_rn(__fmul_rn(n, s1), fv_ld(e, sh + j, e16));
}

// Fused: residual (res_mode) + LayerNorm (ln_mode) in one pass per row.
// Rows are [batch * seq, dim]; the AdaLN table e is [batch, e_rows, dim].
extern "C" __global__ void wan_res_ln(
    const void* h, int h16, const void* a, int a16, const void* e, int e16, int gate_slot,
    const void* w, const void* b, int p16, int scale_slot, int shift_slot,
    int res_mode, int ln_mode, unsigned short* hid, unsigned short* normed,
    int rows, int seq, int dim, int e_rows, float eps
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const long base = (long)row * dim;
    const long tb = (long)(row / seq) * e_rows;
    const long gi = (tb + gate_slot) * dim, sc = (tb + scale_slot) * dim, sh = (tb + shift_slot) * dim;
    auto val = [&](int j) { return wan_res_val(h, h16, a, a16, e, e16, base + j, gi + j, res_mode); };
    float mean, inv;
    wan_ln_stats(val, dim, eps, sdata, &mean, &inv);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        const float v = val(j);
        hid[base + j] = fv_bf16_rne(v);
        normed[base + j] = fv_bf16_rne(wan_ln_out(v, mean, inv, ln_mode, j, w, b, p16, e, e16, sc, sh));
    }
}

// Unfused residual: hidden (bf16) and, with has_tmp, the unrounded f32 value
// the norm of res_mode 0 reads. Also the block's last residual
// (bf16(h + ff * gate), res_mode 0 without tmp).
extern "C" __global__ void wan_res_gate(
    const void* h, int h16, const void* a, int a16, const void* e, int e16, int gate_slot,
    int res_mode, unsigned short* hid, float* tmp, int has_tmp,
    long n, int seq, int dim, int e_rows
) {
    const long i = IDX();
    if (i >= n) return;
    const long row = i / dim, j = i % dim;
    const long gi = ((row / seq) * e_rows + gate_slot) * dim + j;
    const float v = wan_res_val(h, h16, a, a16, e, e16, i, gi, res_mode);
    hid[i] = fv_bf16_rne(v);
    if (has_tmp) tmp[i] = v;
}

// Unfused norm over a stored row (f32 or bf16), ln_mode as above.
extern "C" __global__ void wan_ln(
    const void* x, int x16, const void* w, const void* b, int p16,
    const void* e, int e16, int scale_slot, int shift_slot, int ln_mode,
    unsigned short* normed, int rows, int seq, int dim, int e_rows, float eps
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const long base = (long)row * dim;
    const long tb = (long)(row / seq) * e_rows;
    const long sc = (tb + scale_slot) * dim, sh = (tb + shift_slot) * dim;
    auto val = [&](int j) { return fv_ld(x, base + j, x16); };
    float mean, inv;
    wan_ln_stats(val, dim, eps, sdata, &mean, &inv);
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        normed[base + j] = fv_bf16_rne(wan_ln_out(val(j), mean, inv, ln_mode, j, w, b, p16, e, e16, sc, sh));
}

// q/k RMSNorm across all heads + interleaved RoPE, bf16 in (or f32) and bf16
// BHSD out, with FastVideo's rounding points: RMSNorm.forward_native rounds
// x * rsqrt(mean(x^2) + eps) to bf16, multiplies by the bf16 weight (another
// rounding), and _apply_rotary_emb rotates in f32 and rounds once. Tables
// are qk_norm_rope_bhsd's: cos at the even slot, sin at the odd one.
extern "C" __global__ void wan_qk_norm_rope16(
    const void* src, int s16, const void* w, int w16, const float* cos_t, const float* sin_t,
    int use_rope, unsigned short* out,
    int batch, int seq, int heads, int d, int src_width, int col_off, float eps
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    if (row >= batch * seq) return;
    const int b = row / seq, s = row % seq;
    const int width = heads * d;
    const long xb = (long)row * src_width + col_off;
    float local = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        const float v = fv_ld(src, xb + j, s16);
        local = __fadd_rn(local, __fmul_rn(v, v));
    }
    const float inv = rsqrtf(__fadd_rn(__fdiv_rn(fv_block_sum(local, sdata), (float)width), eps));
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        const int hh = j / d, p = j - hh * d;
        const long o = (((long)b * heads + hh) * seq + s) * d + p;
        if (use_rope) {
            const int even = p - (p & 1), j0 = hh * d + even;
            const float x1 = fv_r16(__fmul_rn(fv_r16(__fmul_rn(fv_ld(src, xb + j0, s16), inv)), fv_ld(w, j0, w16)));
            const float x2 = fv_r16(__fmul_rn(fv_r16(__fmul_rn(fv_ld(src, xb + j0 + 1, s16), inv)), fv_ld(w, j0 + 1, w16)));
            const float c = cos_t[(long)s * d + even], sn = sin_t[(long)s * d + even + 1];
            const float r = (p & 1) ? __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, sn))
                                    : __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, sn));
            out[o] = fv_bf16_rne(r);
        } else {
            out[o] = fv_bf16_rne(__fmul_rn(fv_r16(__fmul_rn(fv_ld(src, xb + j, s16), inv)), fv_ld(w, j, w16)));
        }
    }
}
// ==== endregion: Wan block ====

