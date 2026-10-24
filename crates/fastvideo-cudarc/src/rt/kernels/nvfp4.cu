// ==== region: nvfp4 ====
// Device reconstruct (quantize + dequant) replacing host fake-quant in
// dequant_beforehand. Same numeric recipe as fastvideo_models::nvfp4::reconstruct.
// rule: 0 = static_6 (TE NVFP4BlockScaling), 1 = static_4, 2 = FourOverSix MSE.
// One thread per 16-wide block. Tensor amax is a device scalar from amax_abs.
__device__ __forceinline__ float fv_nvfp4_round_even(float x) {
    float f = floorf(x);
    float frac = x - f;
    if (frac < 0.5f) return f;
    if (frac > 0.5f) return f + 1.0f;
    return (((long long)f) % 2 == 0) ? f : (f + 1.0f);
}

__device__ __forceinline__ float fv_nvfp4_fake_e2m1(float v) {
    if (!isfinite(v)) return 0.0f;
    float a = fabsf(v);
    float mag;
    if (a < 2.0f) mag = fv_nvfp4_round_even(2.0f * a) * 0.5f;
    else if (a < 4.0f) mag = fv_nvfp4_round_even(a);
    else mag = 2.0f * fv_nvfp4_round_even(a * 0.5f);
    return signbit(v) ? -mag : mag;
}

__device__ __forceinline__ void fv_nvfp4_scale_block(
    const float* src, float* fake, unsigned char* scale_out,
    float amax, float e2, float e4, float expansion
) {
    if (!(amax > 0.0f) || !isfinite(amax)) {
        for (int i = 0; i < 16; i++) fake[i] = 0.0f;
        *scale_out = 0;
        return;
    }
    float encode = (e2 * e4) / amax;
    float decode = amax / (e2 * e4);
    float block_amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        float v = fabsf(src[i]);
        if (v > block_amax) block_amax = v;
    }
    unsigned char code = fv_to_e4m3((block_amax / e2) * encode * expansion);
    *scale_out = code;
    float s = fv_e4m3_to_f32(code);
    if (s == 0.0f) {
        for (int i = 0; i < 16; i++) fake[i] = 0.0f;
        return;
    }
    float inv = 1.0f / (decode * s);
    for (int i = 0; i < 16; i++) fake[i] = fv_nvfp4_fake_e2m1(src[i] * inv);
}

__device__ __forceinline__ float fv_nvfp4_block_mse(
    const float* orig, const float* fake, unsigned char scale, float decode
) {
    float s = fv_e4m3_to_f32(scale) * decode;
    float mse = 0.0f;
    for (int i = 0; i < 16; i++) {
        float d = fake[i] * s - orig[i];
        mse += d * d;
    }
    return mse;
}

extern "C" __global__ void nvfp4_reconstruct(
    const float* x, float* out, const float* amax_ptr,
    long rows, long cols, int rule
) {
    long n_blocks = rows * (cols / 16);
    long b = IDX();
    if (b >= n_blocks) return;
    long row = b / (cols / 16);
    long blk = b % (cols / 16);
    const float* src = x + row * cols + blk * 16;
    float* dst = out + row * cols + blk * 16;
    float amax = *amax_ptr;
    float e2 = (rule == 1) ? 4.0f : 6.0f;
    float e4 = (rule == 2) ? 256.0f : 448.0f;
    float fake[16];
    unsigned char scale;
    fv_nvfp4_scale_block(src, fake, &scale, amax, e2, e4, 1.0f);
    if (rule == 2) {
        float fake4[16];
        unsigned char s4;
        fv_nvfp4_scale_block(src, fake4, &s4, amax, e2, e4, 1.5f);
        float mse_decode = (amax > 0.0f) ? amax / (6.0f * 256.0f) : 0.0f;
        if (fv_nvfp4_block_mse(src, fake4, s4, mse_decode)
            < fv_nvfp4_block_mse(src, fake, scale, mse_decode)) {
            for (int i = 0; i < 16; i++) fake[i] = fake4[i];
            scale = s4;
        }
    }
    float decode = (amax > 0.0f && isfinite(amax)) ? amax / (e2 * e4) : 0.0f;
    float s = fv_e4m3_to_f32(scale) * decode;
    for (int i = 0; i < 16; i++) dst[i] = fake[i] * s;
}

// E2M1 nibble for an already-rounded value (fastvideo_models::nvfp4::e2m1_from_f32).
__device__ __forceinline__ unsigned char fv_nvfp4_code(float x) {
    unsigned char sign = signbit(x) ? 0x8 : 0x0;
    float a = fabsf(x);
    unsigned char mag = a < 0.25f ? 0 : a < 0.75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3
                      : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.0f ? 6 : 7;
    return sign | mag;
}

// Quantize to the packed form (fastvideo_models::nvfp4::quantize): E2M1 codes
// two per byte, low nibble first, [rows, cols/2]; E4M3 block scales
// [rows, cols/16] row-major. Bit-identical to the host for every rule.
extern "C" __global__ void nvfp4_quantize_pack(
    const float* x, unsigned char* packed, unsigned char* scales,
    const float* amax_ptr, long rows, long cols, int rule
) {
    long n_blocks = rows * (cols / 16);
    long b = IDX();
    if (b >= n_blocks) return;
    long row = b / (cols / 16);
    long blk = b % (cols / 16);
    const float* src = x + row * cols + blk * 16;
    float amax = *amax_ptr;
    float e2 = (rule == 1) ? 4.0f : 6.0f;
    float e4 = (rule == 2) ? 256.0f : 448.0f;
    float fake[16];
    unsigned char scale;
    fv_nvfp4_scale_block(src, fake, &scale, amax, e2, e4, 1.0f);
    if (rule == 2) {
        float fake4[16];
        unsigned char s4;
        fv_nvfp4_scale_block(src, fake4, &s4, amax, e2, e4, 1.5f);
        float mse_decode = (amax > 0.0f) ? amax / (6.0f * 256.0f) : 0.0f;
        if (fv_nvfp4_block_mse(src, fake4, s4, mse_decode)
            < fv_nvfp4_block_mse(src, fake, scale, mse_decode)) {
            for (int i = 0; i < 16; i++) fake[i] = fake4[i];
            scale = s4;
        }
    }
    scales[b] = scale;
    unsigned char* dst = packed + row * (cols / 2) + blk * 8;
    for (int i = 0; i < 8; i++) {
        dst[i] = fv_nvfp4_code(fake[2 * i]) | (fv_nvfp4_code(fake[2 * i + 1]) << 4);
    }
}

// Row-major [rows, sc] E4M3 block scales -> cuBLASLt VEC16_UE4M3 layout:
// 128x4 tiles (row-tile major, scale-column tile minor), 512 bytes each,
// element (r, c) of a tile at (r % 32) * 16 + (r / 32) * 4 + c. Rows pad to
// 128 and columns to 4 with zero scales. One thread per padded element.
extern "C" __global__ void nvfp4_scales_swizzle(
    const unsigned char* s, unsigned char* out, long rows, long sc, long rows_pad, long sc_pad
) {
    long i = IDX();
    if (i >= rows_pad * sc_pad) return;
    long r = i / sc_pad;
    long c = i % sc_pad;
    unsigned char v = (r < rows && c < sc) ? s[r * sc + c] : (unsigned char)0;
    long tile = (r / 128) * (sc_pad / 4) + (c / 4);
    long rr = r % 128;
    out[tile * 512 + (rr % 32) * 16 + (rr / 32) * 4 + (c % 4)] = v;
}

// Deterministic uniform fill in [-scale, scale): benchmark operands too large
// to generate on the host. splitmix64 of (seed, index).
extern "C" __global__ void fill_hash_uniform(float* out, long n, unsigned long long seed, float scale) {
    long i = IDX();
    if (i >= n) return;
    unsigned long long z = seed + 0x9E3779B97F4A7C15ULL * (unsigned long long)(i + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    float u = (float)(z >> 40) * (1.0f / 16777216.0f);
    out[i] = (2.0f * u - 1.0f) * scale;
}

// ---- NVFP4 linear (cuBLASLt VEC16_UE4M3; wan/nvfp4_gemm.rs Nvfp4Linear) ----
// bf16 operands straight into the GEMM's layout. `gelu` = 1 reads
// bf16(gelu_tanh(x)) instead of x (the value the bf16 FFN rounds and feeds its
// down projection), so the GELU never round-trips through memory.
__device__ __forceinline__ float fv_nvfp4_in(unsigned short b, int gelu) {
    float x = fv_bf16_to_f32(b);
    if (!gelu) return x;
    const float k = 0.7978845608028654f;
    float g = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
    return fv_bf16_to_f32(fv_bf16_rne(g));
}

// Tensor amax of a bf16 buffer (of gelu(x) when `gelu`). 8 elements per load
// when n % 8 == 0 (the caller's buffers are 256-byte aligned). `out` zeroed.
extern "C" __global__ void nvfp4_amax_bf16(const unsigned short* x, int gelu, float* out, long n) {
    __shared__ float sm[256];
    int tid = threadIdx.x;
    float acc = 0.0f;
    long stride = (long)gridDim.x * blockDim.x;
    if ((n & 7) == 0) {
        const uint4* v = (const uint4*)x;
        for (long i = blockIdx.x * (long)blockDim.x + tid; i < n / 8; i += stride) {
            uint4 q = v[i];
            unsigned int w[4] = {q.x, q.y, q.z, q.w};
            for (int j = 0; j < 4; j++) {
                float a = fabsf(fv_nvfp4_in((unsigned short)(w[j] & 0xFFFFu), gelu));
                float b = fabsf(fv_nvfp4_in((unsigned short)(w[j] >> 16), gelu));
                if (!(a <= acc)) acc = a;
                if (!(b <= acc)) acc = b;
            }
        }
    } else {
        for (long i = blockIdx.x * (long)blockDim.x + tid; i < n; i += stride) {
            float a = fabsf(fv_nvfp4_in(x[i], gelu));
            if (!(a <= acc)) acc = a;
        }
    }
    sm[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { float o = sm[tid + s]; if (!(o <= sm[tid])) sm[tid] = o; }
        __syncthreads();
    }
    if (tid == 0) atomicMax((int*)out, __float_as_int(sm[0]));
}

// bf16 [rows, cols] -> packed E2M1 [rows_pack, cols/2] (low nibble first) and
// E4M3 block scales directly in cuBLASLt's VEC16_UE4M3 tiled layout (128x4
// tiles, element (r, c) at tile * 512 + (r % 32) * 16 + (r % 128 / 32) * 4 +
// c % 4; the same map as nvfp4_scales_swizzle). The math per block is
// fv_nvfp4_scale_block (static_6: e2 6, static_4: e2 4; e4 448), so the codes
// equal nvfp4_quantize_pack's bit for bit. One thread per (row, 16-block) over
// rows_scale rows: rows in [rows, rows_pack) get zero codes, rows in
// [rows, rows_scale) zero scales. Needs cols % 64 == 0 (no scale-column pad).
extern "C" __global__ void nvfp4_quant_bf16_sw(
    const unsigned short* x, int gelu, const float* amax_ptr,
    unsigned char* packed, unsigned char* scales,
    long rows, long rows_pack, long rows_scale, long cols, int rule
) {
    long sc = cols / 16;
    long b = IDX();
    if (b >= rows_scale * sc) return;
    long row = b / sc;
    long blk = b % sc;
    long tile = (row / 128) * (sc / 4) + (blk / 4);
    long rr = row % 128;
    long s_off = tile * 512 + (rr % 32) * 16 + (rr / 32) * 4 + (blk % 4);
    if (row >= rows) {
        scales[s_off] = 0;
        if (row < rows_pack) {
            *(uint2*)(packed + row * (cols / 2) + blk * 8) = make_uint2(0u, 0u);
        }
        return;
    }
    const uint4* src = (const uint4*)(x + row * cols + blk * 16);
    uint4 q0 = src[0], q1 = src[1];
    unsigned int w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    float v[16];
    for (int j = 0; j < 8; j++) {
        v[2 * j] = fv_nvfp4_in((unsigned short)(w[j] & 0xFFFFu), gelu);
        v[2 * j + 1] = fv_nvfp4_in((unsigned short)(w[j] >> 16), gelu);
    }
    float fake[16];
    unsigned char scale;
    fv_nvfp4_scale_block(v, fake, &scale, *amax_ptr, rule == 1 ? 4.0f : 6.0f, 448.0f, 1.0f);
    scales[s_off] = scale;
    unsigned int lo = 0u, hi = 0u;
    for (int i = 0; i < 4; i++) {
        lo |= (unsigned int)(fv_nvfp4_code(fake[2 * i]) | (fv_nvfp4_code(fake[2 * i + 1]) << 4)) << (8 * i);
        hi |= (unsigned int)(fv_nvfp4_code(fake[8 + 2 * i]) | (fv_nvfp4_code(fake[9 + 2 * i]) << 4)) << (8 * i);
    }
    *(uint2*)(packed + row * (cols / 2) + blk * 8) = make_uint2(lo, hi);
}

// The GEMM's alpha / beta on the device: alpha = decode(a) * decode(w),
// decode = amax / (e2 * 448) (fastvideo_models::nvfp4::dequant_factor), so
// the per-tensor scales never leave the device.
extern "C" __global__ void nvfp4_alpha(const float* a_amax, const float* w_amax, float e2, float* out) {
    float a = *a_amax, w = *w_amax, d = e2 * 448.0f;
    float da = (a > 0.0f && isfinite(a)) ? a / d : 0.0f;
    float dw = (w > 0.0f && isfinite(w)) ? w / d : 0.0f;
    out[0] = da * dw;
    out[1] = 0.0f;
}
// ==== endregion: nvfp4 ====


