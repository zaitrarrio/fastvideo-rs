// ==== region: bf16 activations + reference FP8 recipes ====
// Every kernel below reads each operand as f32 or bf16 (`*16` flags: 1 means
// the pointer holds bfloat16 bits), computes in f32, and rounds once when it
// stores a bf16 result — torch's per-op bf16 semantics. Where the reference
// fuses several ops into one Triton kernel (Sol-H3 `fusions.py`, `mxfp8.py`),
// the fused kernel here keeps exactly its rounding points instead.

__device__ __forceinline__ float fv_ld(const void* p, long i, int is16) {
    return is16 ? fv_bf16_to_f32(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}
__device__ __forceinline__ void fv_st(void* p, long i, int is16, float v) {
    if (is16) ((unsigned short*)p)[i] = fv_bf16_rne(v);
    else ((float*)p)[i] = v;
}
__device__ __forceinline__ float fv_r16(float v) { return fv_bf16_to_f32(fv_bf16_rne(v)); }

// f32 -> E4M3 the way `cvt.rn.satfinite.e4m3x2.f32` (Triton's fp32 ->
// float8e4nv) and torch's clamp-then-cast both land: round to nearest even,
// finite values past 448 saturate to 448, NaN stays NaN (0x7F). Mirrors
// `quant::e4m3_satfinite` on the host bit for bit.
__device__ __forceinline__ unsigned char fv_e4m3_sat(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int sign = (u >> 24) & 0x80u;
    unsigned int a = u & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return (unsigned char)(sign | 0x7Fu);
    if (a >= 0x43E00000u) return (unsigned char)(sign | 0x7Eu);
    if (a < 0x3C800000u) {
        // Below 2^-6: subnormal codes, quantum 2^-9. The product is exact.
        unsigned int m = (unsigned int)rintf(__uint_as_float(a) * 512.0f);
        return (unsigned char)(sign | m);
    }
    int e = (int)(a >> 23) - 127;
    unsigned int m = (a >> 20) & 7u;
    unsigned int rem = a & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) {
        m += 1u;
        if (m == 8u) { m = 0u; e += 1; }
    }
    return (unsigned char)(sign | ((unsigned int)(e + 7) << 3) | m);
}

// MX (OCP microscaling) E8M0 exponent for a 32-value block, exactly as
// Sol-H3 `mxfp8.py::_mx_e8m0_from_amax`: the smallest power of two that keeps
// the block maximum within E4M3's 448 after scaling. Returns the biased byte;
// `inv` receives 2^-exponent.
__device__ __forceinline__ unsigned int fv_mx_e8m0(float amax, float* inv) {
    int bits = __float_as_int(amax);
    int e0 = ((bits >> 23) & 0xFF) - 135;
    float threshold = __int_as_float(((e0 + 135) << 23) | 0x600000);
    int e = e0 + (amax > threshold ? 1 : 0);
    if (e < -127) e = -127;
    *inv = __int_as_float((127 - e) << 23);
    return (unsigned int)(e + 127);
}
// cuBLASLt VEC32_UE8M0 (torch SWIZZLE_32_4_4) scale position of (row, group).
__device__ __forceinline__ long fv_mx_scale_offset(long row, long group, long column_blocks) {
    long tile = (row / 128) * column_blocks + (group / 4);
    return tile * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (group % 4);
}
// Quantize one warp's 32 consecutive columns (lane = column % 32) of `row`.
// All 32 lanes must be active.
__device__ __forceinline__ void fv_mx_store_group(
    float v, long row, long col, long k, unsigned char* q, unsigned char* qs, long column_blocks
) {
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float inv;
    unsigned int byte = fv_mx_e8m0(amax, &inv);
    q[row * k + col] = fv_e4m3_sat(v * inv);
    if ((threadIdx.x & 31) == 0) qs[fv_mx_scale_offset(row, col / 32, column_blocks)] = (unsigned char)byte;
}

// Block-wide sum over blockDim.x (a power of two) using dynamic shared memory.
__device__ __forceinline__ float fv_block_sum(float v, float* sm) {
    int tid = threadIdx.x;
    sm[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float r = sm[0];
    __syncthreads();
    return r;
}

// out = op(a[i], b[(i / inner) % period]); op as bcast_binary.
extern "C" __global__ void mx_binary(
    const void* a, int a16, const void* b, int b16, void* out, int o16,
    long n, long inner, long period, int op
) {
    long i = IDX();
    if (i >= n) return;
    float x = fv_ld(a, i, a16);
    float y = fv_ld(b, (i / inner) % period, b16);
    float r;
    if (op == 0) r = x + y;
    else if (op == 1) r = x - y;
    else if (op == 2) r = x * y;
    else if (op == 3) r = x / y;
    else if (op == 4) r = y - x;
    else r = y / x;
    fv_st(out, i, o16, r);
}
// kind: 0 silu, 1 gelu_tanh, 2 gelu_erf, 3 x + p0, 4 x * p0, 5 clamp(p0, p1),
// 6 cast/copy, 7 sigmoid.
extern "C" __global__ void mx_unary(
    const void* a, int a16, void* out, int o16, long n, int kind, float p0, float p1
) {
    long i = IDX();
    if (i >= n) return;
    float x = fv_ld(a, i, a16);
    float r;
    if (kind == 0) r = x / (1.0f + expf(-x));
    else if (kind == 1) {
        const float k = 0.7978845608028654f;
        r = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
    } else if (kind == 2) r = 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    else if (kind == 3) r = x + p0;
    else if (kind == 4) r = x * p0;
    else if (kind == 5) r = fminf(p1, fmaxf(p0, x));
    else if (kind == 6) r = x;
    else r = 1.0f / (1.0f + expf(-x));
    fv_st(out, i, o16, r);
}
// Value-first SwiGLU as Sol-H3 `fusions.py::_swiglu_kernel`:
// value * (gate * sigmoid(gate)) in f32, one rounding.
extern "C" __global__ void mx_swiglu(const void* x, int x16, void* out, int o16, long half, long n) {
    long i = IDX();
    if (i >= n) return;
    long col = i % half;
    long row = i / half;
    float v = fv_ld(x, row * (2 * half) + col, x16);
    float g = fv_ld(x, row * (2 * half) + half + col, x16);
    float sg = 1.0f / (1.0f + expf(-g));
    fv_st(out, i, o16, v * (g * sg));
}
// RMSNorm over the last dim: statistics in f32, x * rsqrt(mean(x^2) + eps) * w
// in f32, one rounding (fused semantics). `has_w` = 0 for an unweighted norm.
extern "C" __global__ void mx_rms_norm(
    const void* x, int x16, const void* w, int w16, int has_w, void* out, int o16,
    int rows, int width, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    long base = (long)row * width;
    float local = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = fv_ld(x, base + j, x16);
        local += v * v;
    }
    float r = rsqrtf(fv_block_sum(local, sdata) / (float)width + eps);
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = fv_ld(x, base + j, x16) * r;
        if (has_w) v *= fv_ld(w, j, w16);
        fv_st(out, base + j, o16, v);
    }
}
// LayerNorm over the last dim (f32 statistics, one rounding).
extern "C" __global__ void mx_layer_norm(
    const void* x, int x16, const void* w, const void* b, int p16, int has_affine,
    void* out, int o16, int rows, int width, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    long base = (long)row * width;
    float local = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) local += fv_ld(x, base + j, x16);
    float mean = fv_block_sum(local, sdata) / (float)width;
    local = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float d = fv_ld(x, base + j, x16) - mean;
        local += d * d;
    }
    float inv = rsqrtf(fv_block_sum(local, sdata) / (float)width + eps);
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = (fv_ld(x, base + j, x16) - mean) * inv;
        if (has_affine) v = v * fv_ld(w, j, p16) + fv_ld(b, j, p16);
        fv_st(out, base + j, o16, v);
    }
}
// Unaffine LayerNorm + AdaLN table (`ln_adaln_e`), one rounding.
extern "C" __global__ void mx_ln_adaln_e(
    const void* x, int x16, const void* e, int e16, void* out, int o16,
    int batch, int seq, int dim, int e_rows, int scale_slot, int shift_slot, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= batch * seq) return;
    int bi = row / seq;
    long base = (long)row * dim;
    long sc = ((long)bi * e_rows + scale_slot) * dim;
    long sh = ((long)bi * e_rows + shift_slot) * dim;
    float local = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) local += fv_ld(x, base + j, x16);
    float mean = fv_block_sum(local, sdata) / (float)dim;
    local = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float d = fv_ld(x, base + j, x16) - mean;
        local += d * d;
    }
    float inv = rsqrtf(fv_block_sum(local, sdata) / (float)dim + eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = (fv_ld(x, base + j, x16) - mean) * inv;
        fv_st(out, base + j, o16, v * (1.0f + fv_ld(e, sc + j, e16)) + fv_ld(e, sh + j, e16));
    }
}
// out = h + a * e[b, slot] with eager bf16 rounding when `round_product`:
// torch materializes gate * branch in bf16 before the residual add.
extern "C" __global__ void mx_residual_gate(
    const void* h, int h16, const void* a, int a16, const void* e, int e16, void* out, int o16,
    long n, long dim, long seq, long e_rows, long slot, int round_product
) {
    long i = IDX();
    if (i >= n) return;
    long d = i % dim;
    long b = (i / dim) / seq;
    float p = fv_ld(a, i, a16) * fv_ld(e, (b * e_rows + slot) * dim + d, e16);
    if (round_product) p = fv_r16(p);
    fv_st(out, i, o16, fv_ld(h, i, h16) + p);
}
extern "C" __global__ void mx_split_heads(
    const void* src, int s16, void* out, int o16,
    long n, long seq, long heads, long d, long src_width, long col_off
) {
    long i = IDX();
    if (i >= n) return;
    long p = i % d;
    long t = i / d;
    long s = t % seq;
    t /= seq;
    long h = t % heads;
    long b = t / heads;
    fv_st(out, i, o16, fv_ld(src, (b * seq + s) * src_width + col_off + h * d + p, s16));
}
extern "C" __global__ void mx_merge_heads(
    const void* src, int s16, void* out, int o16, long n, long seq, long heads, long d
) {
    long i = IDX();
    if (i >= n) return;
    long hd = heads * d;
    long j = i % hd;
    long row = i / hd;
    long s = row % seq;
    long b = row / seq;
    fv_st(out, i, o16, fv_ld(src, ((b * heads + j / d) * seq + s) * d + j % d, s16));
}
extern "C" __global__ void mx_gather_nd(
    const void* in, int i16, void* out, int o16, long n, int rank,
    long s0, long s1, long s2, long s3, long s4, long s5,
    long t0, long t1, long t2, long t3, long t4, long t5
) {
    long i = IDX();
    if (i >= n) return;
    long shape[6] = {s0, s1, s2, s3, s4, s5};
    long str[6] = {t0, t1, t2, t3, t4, t5};
    long rem = i;
    long src = 0;
    for (int k = rank - 1; k >= 0; --k) {
        long c = rem % shape[k];
        rem /= shape[k];
        src += c * str[k];
    }
    fv_st(out, i, o16, fv_ld(in, src, i16));
}
extern "C" __global__ void mx_block_copy(
    const void* in, int i16, void* out, int o16,
    long outer, long len, long in_stride, long out_stride, long in_offset, long out_offset
) {
    long idx = IDX();
    if (idx >= outer * len) return;
    long o = idx / len;
    long j = idx - o * len;
    fv_st(out, o * out_stride + out_offset + j, o16, fv_ld(in, o * in_stride + in_offset + j, i16));
}
// rotate_half RoPE over [B, H, S, D] with f32 [S, R] tables, one rounding.
extern "C" __global__ void mx_rope_half(
    const void* x, int x16, const float* cs, const float* sn, void* out, int o16,
    long s, long d, long r, long n
) {
    long i = IDX();
    if (i >= n) return;
    long j = i % d;
    float v = fv_ld(x, i, x16);
    if (j >= r) { fv_st(out, i, o16, v); return; }
    long p = (i / d) % s;
    long half = r / 2;
    float other = j < half ? -fv_ld(x, i + half, x16) : fv_ld(x, i - half, x16);
    fv_st(out, i, o16, v * cs[p * r + j] + other * sn[p * r + j]);
}

