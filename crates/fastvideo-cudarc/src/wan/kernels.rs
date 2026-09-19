//! NVRTC-compiled CUDA kernels.
//!
//! Compiled once into [`crate::wan::device::DeviceContext`] when CUDA is initialized.
//! Every kernel has a plain-Rust twin in [`super::ops`] (the CPU path) and a
//! parity check in `fv-gpucheck kernels`. Indices are `long` so no tensor this
//! crate sees can overflow them.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaFunction, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::device::{DeviceError, Result};

const KERNEL_SRC: &str = r#"
#define IDX() ((long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x)

extern "C" __global__ void elem_add(const float* a, const float* b, float* out, long n) {
    long i = IDX();
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void elem_mul(const float* a, const float* b, float* out, long n) {
    long i = IDX();
    if (i < n) out[i] = a[i] * b[i];
}
extern "C" __global__ void elem_sub(const float* a, const float* b, float* out, long n) {
    long i = IDX();
    if (i < n) out[i] = a[i] - b[i];
}
extern "C" __global__ void mul_scalar(const float* a, float s, float* out, long n) {
    long i = IDX();
    if (i < n) out[i] = a[i] * s;
}
extern "C" __global__ void add_scalar(const float* a, float s, float* out, long n) {
    long i = IDX();
    if (i < n) out[i] = a[i] + s;
}
extern "C" __global__ void silu(const float* a, float* out, long n) {
    long i = IDX();
    if (i < n) {
        float x = a[i];
        out[i] = x / (1.0f + expf(-x));
    }
}
// TAEHV's Clamp block: a soft limiter, not a hard clamp, so the decoder never
// sees a latent far outside the range it was trained on.
extern "C" __global__ void tanh_scaled(const float* a, float* out, const float* s, long n) {
    long i = IDX();
    if (i < n) out[i] = tanhf(a[i] / *s) * *s;
}
extern "C" __global__ void gelu_tanh(const float* a, float* out, long n) {
    long i = IDX();
    if (i < n) {
        float x = a[i];
        const float k = 0.7978845608028654f; // sqrt(2/pi)
        float u = k * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(u));
    }
}
extern "C" __global__ void clamp_f(const float* a, float lo, float hi, float* out, long n) {
    long i = IDX();
    if (i < n) out[i] = fminf(hi, fmaxf(lo, a[i]));
}
extern "C" __global__ void fill_f(float* out, float v, long n) {
    long i = IDX();
    if (i < n) out[i] = v;
}
// out = a*x + b*y + c*z. Samplers, CFG and residual mixes in one launch.
extern "C" __global__ void lincomb3(
    const float* x, const float* y, const float* z, float* out,
    float a, float b, float c, long n
) {
    long i = IDX();
    if (i < n) out[i] = a * x[i] + b * y[i] + c * z[i];
}
// Broadcast binary op where `b` repeats: b index = (i / inner) % period.
// Covers trailing repeats (inner=1), channel broadcasts ([1,C,1,1] → inner=H*W)
// and leading singleton batch dims. op: 0 a+b, 1 a-b, 2 a*b, 3 a/b, 4 b-a, 5 b/a.
extern "C" __global__ void bcast_binary(
    const float* a, const float* b, float* out, long n, long inner, long period, int op
) {
    long i = IDX();
    if (i >= n) return;
    float x = a[i];
    float y = b[(i / inner) % period];
    float r;
    if (op == 0) r = x + y;
    else if (op == 1) r = x - y;
    else if (op == 2) r = x * y;
    else if (op == 3) r = x / y;
    else if (op == 4) r = y - x;
    else r = y / x;
    out[i] = r;
}
// In-place bias add: out[i] += bias[(i / inner) % period]. inner=1 is the
// last dim (Linear); inner=H*W with period=C is a per-channel conv bias.
extern "C" __global__ void add_bias_inplace(float* out, const float* bias, long n, long inner, long period) {
    long i = IDX();
    if (i < n) out[i] += bias[(i / inner) % period];
}
// In-place FFN activation: x = gelu_tanh(x + bias[i % width]).
extern "C" __global__ void bias_gelu_inplace(float* x, const float* bias, long n, long width) {
    long i = IDX();
    if (i >= n) return;
    float v = x[i] + bias[i % width];
    const float k = 0.7978845608028654f;
    float u = k * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(u));
}
// f32 -> bfloat16 bits (stored as ushort): sign + 8-bit exponent + top 7
// mantissa bits, rounded to nearest on the dropped bits. Carry into the
// exponent is the correct round-up; inf/NaN and the largest finite value are
// left unrounded.
extern "C" __global__ void cast_f32_bf16(const float* a, unsigned short* out, long n) {
    long i = IDX();
    if (i >= n) return;
    unsigned int u = __float_as_uint(a[i]);
    unsigned int hi = u >> 16;
    if ((u & 0x8000u) != 0u && (hi & 0x7F80u) != 0x7F80u && (hi & 0x7FFFu) != 0x7F7Fu) hi += 1u;
    out[i] = (unsigned short)hi;
}
// bfloat16 bits -> f32, then optional bias[i % width] and GELU-tanh (act=1):
// the output side of a bf16 linear in one launch.
extern "C" __global__ void cast_bf16_f32_bias_act(
    const unsigned short* a, const float* bias, float* out, long n, long width, int has_bias, int act
) {
    long i = IDX();
    if (i >= n) return;
    float v = __uint_as_float(((unsigned int)a[i]) << 16);
    if (has_bias) v += bias[i % width];
    if (act == 1) {
        const float k = 0.7978845608028654f;
        float u = k * (v + 0.044715f * v * v * v);
        v = 0.5f * v * (1.0f + tanhf(u));
    }
    out[i] = v;
}
// Gated residual with the AdaLN table: out = h + a * e[b, slot, d], with
// h/a [batch, seq, dim] and e [batch, e_rows, dim].
extern "C" __global__ void residual_gate_add_e(
    const float* h, const float* a, const float* e, float* out,
    long n, long dim, long seq, long e_rows, long slot
) {
    long i = IDX();
    if (i >= n) return;
    long d = i % dim;
    long b = (i / dim) / seq;
    out[i] = h[i] + a[i] * e[(b * e_rows + slot) * dim + d];
}
// Softmax along a contiguous trailing axis of length `width` (rows = n / width).
// One block per row, `blockDim.x` threads cooperating via shared-memory tree
// reduction; each thread strides across the row (`j += nthreads`).
extern "C" __global__ void softmax_last(const float* a, float* out, int rows, int width) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = a + (long)row * (long)width;
    float* dst = out + (long)row * (long)width;

    float local_max = -1e30f;
    for (int j = tid; j < width; j += nthreads) local_max = fmaxf(local_max, src[j]);
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float m = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < width; j += nthreads) {
        float e = expf(src[j] - m);
        dst[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    __syncthreads();
    for (int j = tid; j < width; j += nthreads) dst[j] *= inv;
}
// Softmax over the last dim writing bfloat16 probabilities. In fast mode
// cuBLAS already rounds its F32 inputs to bf16 for the tensor-core op, so
// storing the probabilities this way is the same math over half the bytes —
// and the probability matrix is the dominant traffic in dense attention.
// Three passes over the f32 scores (max, sum, write) beat keeping f32 exps
// around: the row is far too wide for shared memory.
extern "C" __global__ void softmax_last_bf16(const float* a, unsigned short* out, int rows, int width) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = a + (long)row * (long)width;
    unsigned short* dst = out + (long)row * (long)width;

    float local_max = -1e30f;
    for (int j = tid; j < width; j += nthreads) local_max = fmaxf(local_max, src[j]);
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float m = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < width; j += nthreads) local_sum += expf(src[j] - m);
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    __syncthreads();
    for (int j = tid; j < width; j += nthreads) {
        unsigned int u = __float_as_uint(expf(src[j] - m) * inv);
        unsigned int hi = u >> 16;
        if ((u & 0x8000u) != 0u && (hi & 0x7F80u) != 0x7F80u && (hi & 0x7FFFu) != 0x7F7Fu) hi += 1u;
        dst[j] = (unsigned short)hi;
    }
}
// RMS norm over last dim `width` (rows = n / width). weight length = width.
extern "C" __global__ void rms_norm_last(
    const float* a, const float* w, float* out, int rows, int width, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = a + (long)row * (long)width;
    float* dst = out + (long)row * (long)width;

    float local = 0.0f;
    for (int j = tid; j < width; j += nthreads) { float v = src[j]; local += v * v; }
    sdata[tid] = local;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / (float)width + eps);
    __syncthreads();
    for (int j = tid; j < width; j += nthreads) dst[j] = src[j] * scale * w[j];
}
// LayerNorm over last dim `width`; `has_affine` gates weight/bias uniformly.
extern "C" __global__ void layer_norm_last(
    const float* a, const float* w, const float* b, float* out,
    int rows, int width, float eps, int has_affine
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = a + (long)row * (long)width;
    float* dst = out + (long)row * (long)width;

    float local_sum = 0.0f;
    for (int j = tid; j < width; j += nthreads) local_sum += src[j];
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)width;
    __syncthreads();

    float local_var = 0.0f;
    for (int j = tid; j < width; j += nthreads) {
        float d = src[j] - mean;
        local_var += d * d;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] / (float)width + eps);
    __syncthreads();

    if (has_affine) {
        for (int j = tid; j < width; j += nthreads) dst[j] = (src[j] - mean) * inv * w[j] + b[j];
    } else {
        for (int j = tid; j < width; j += nthreads) dst[j] = (src[j] - mean) * inv;
    }
}
// Unaffine LayerNorm + AdaLN modulate straight from the per-block table:
// out = LN(x) * (1 + e[b, scale_slot]) + e[b, shift_slot]; x [batch, seq, dim],
// e [batch, e_rows, dim]. Replaces add + chunk(6) + LN + modulate.
extern "C" __global__ void ln_adaln_e(
    const float* x, const float* e, float* out,
    int batch, int seq, int dim, int e_rows, int scale_slot, int shift_slot, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= batch * seq) return;
    int b = row / seq;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = x + (long)row * dim;
    float* dst = out + (long)row * dim;
    const float* sc = e + ((long)b * e_rows + scale_slot) * dim;
    const float* sh = e + ((long)b * e_rows + shift_slot) * dim;

    float local_sum = 0.0f;
    for (int j = tid; j < dim; j += nthreads) local_sum += src[j];
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)dim;
    __syncthreads();

    float local_var = 0.0f;
    for (int j = tid; j < dim; j += nthreads) {
        float d = src[j] - mean;
        local_var += d * d;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] / (float)dim + eps);
    __syncthreads();

    for (int j = tid; j < dim; j += nthreads) {
        dst[j] = (src[j] - mean) * inv * (1.0f + sc[j]) + sh[j];
    }
}
// Attention q/k preparation in one pass: RMSNorm over the full projection
// width (heads * d), optional interleaved RoPE per head, written directly in
// BHSD layout. src [batch, seq, src_width]; the projection is the slice
// [col_off, col_off + heads*d) of each row (fused QKV output). cos/sin [seq, d]
// (Wan layout: cos from the even slot, sin from the odd slot).
extern "C" __global__ void qk_norm_rope_bhsd(
    const float* src, const float* w, const float* cos_t, const float* sin_t, float* out,
    int batch, int seq, int heads, int d, int src_width, int col_off, float eps, int use_rope
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= batch * seq) return;
    int b = row / seq;
    int s = row % seq;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int width = heads * d;
    const float* x = src + (long)row * src_width + col_off;

    float local = 0.0f;
    for (int j = tid; j < width; j += nthreads) { float v = x[j]; local += v * v; }
    sdata[tid] = local;
    __syncthreads();
    for (int k = nthreads >> 1; k > 0; k >>= 1) {
        if (tid < k) sdata[tid] += sdata[tid + k];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] / (float)width + eps);
    __syncthreads();

    for (int j = tid; j < width; j += nthreads) {
        int h = j / d;
        int p = j - h * d;
        long o = (((long)b * heads + h) * seq + s) * d + p;
        if (use_rope) {
            int even = p - (p & 1);
            int j0 = h * d + even;
            float x1 = x[j0] * inv * w[j0];
            float x2 = x[j0 + 1] * inv * w[j0 + 1];
            float c = cos_t[(long)s * d + even];
            float sn = sin_t[(long)s * d + even + 1];
            out[o] = (p & 1) ? (x1 * sn + x2 * c) : (x1 * c - x2 * sn);
        } else {
            out[o] = x[j] * inv * w[j];
        }
    }
}
// [batch, seq, src_width] slice [col_off, col_off+heads*d) → BHSD (value heads).
extern "C" __global__ void split_heads_bhsd(
    const float* src, float* out, long n, long seq, long heads, long d, long src_width, long col_off
) {
    long i = IDX();
    if (i >= n) return;
    long p = i % d;
    long t = i / d;
    long s = t % seq;
    t /= seq;
    long h = t % heads;
    long b = t / heads;
    out[i] = src[(b * seq + s) * src_width + col_off + h * d + p];
}
// BHSD → [batch, seq, heads*d].
extern "C" __global__ void merge_heads(
    const float* src, float* out, long n, long seq, long heads, long d
) {
    long i = IDX();
    if (i >= n) return;
    long hd = heads * d;
    long j = i % hd;
    long row = i / hd;
    long s = row % seq;
    long b = row / seq;
    long h = j / d;
    long p = j % d;
    out[i] = src[((b * heads + h) * seq + s) * d + p];
}
// N-D permute (rank <= 6) as a strided gather: out coords unravel over
// out_shape; `str[k]` is the input stride of the axis feeding output axis k.
extern "C" __global__ void gather_nd(
    const float* in, float* out, long n, int rank,
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
    out[i] = in[src];
}
// Contiguous-block copy used by narrow/cat/pad on any axis: for each outer
// index o, copy `len` floats in[o*in_stride + in_offset + j] → out[o*out_stride + out_offset + j].
extern "C" __global__ void block_copy(
    const float* in, float* out,
    long outer, long len, long in_stride, long out_stride, long in_offset, long out_offset
) {
    long idx = IDX();
    if (idx >= outer * len) return;
    long o = idx / len;
    long j = idx - o * len;
    out[o * out_stride + out_offset + j] = in[o * in_stride + in_offset + j];
}
// Nearest-neighbour integer upsample of [nc, h, w] planes by (fy, fx).
extern "C" __global__ void upsample_nearest(
    const float* in, float* out, long n, long h, long w, long fy, long fx
) {
    long i = IDX();
    if (i >= n) return;
    long ow = w * fx;
    long oh = h * fy;
    long x = i % ow;
    long t = i / ow;
    long y = t % oh;
    long c = t / oh;
    out[i] = in[(c * h + y / fy) * w + x / fx];
}
// RMS over the channel axis of [n, c, spatial]; gamma length = c.
// `act != 0` applies SiLU to the normalized value. The kernel already reads x
// twice (sum of squares, then normalize), so the activation rides along for
// free and saves a whole read+write pass over a tensor that is hundreds of MB
// at VAE decode resolution.
extern "C" __global__ void rms_norm_channels(
    const float* x, const float* gamma, float* out,
    long n, long c, long spatial, float eps, int act
) {
    long idx = IDX();
    if (idx >= n * spatial) return;
    long ni = idx / spatial;
    long s = idx - ni * spatial;
    float acc = 0.0f;
    for (long ci = 0; ci < c; ++ci) {
        float v = x[(ni * c + ci) * spatial + s];
        acc += v * v;
    }
    float inv = rsqrtf(acc / (float)c + eps);
    for (long ci = 0; ci < c; ++ci) {
        long i = (ni * c + ci) * spatial + s;
        float v = x[i] * inv * gamma[ci];
        // Same expression as the standalone silu kernel, so fusing cannot move
        // the result.
        out[i] = act ? v / (1.0f + expf(-v)) : v;
    }
}
// Temporal unfold for conv3d-as-conv2d: [n, c, t, h, w] → [n*ot, c*kt, h, w]
// with window o covering input frames o*st .. o*st+kt-1 (channel c major).
extern "C" __global__ void temporal_unfold(
    const float* x, float* out, long total, long c, long t, long h, long w, long kt, long st, long ot
) {
    long i = IDX();
    if (i >= total) return;
    long xw = i % w;
    long r = i / w;
    long y = r % h;
    r /= h;
    long ck = r % (c * kt);
    long bo = r / (c * kt);
    long cc = ck / kt;
    long k = ck % kt;
    long o = bo % ot;
    long b = bo / ot;
    out[i] = x[(((b * c + cc) * t + o * st + k) * h + y) * w + xw];
}
// Embedding lookup: out[r, j] = table[idx[r], j].
extern "C" __global__ void index_select_rows(
    const float* table, const unsigned int* idx, float* out, long n, long d
) {
    long i = IDX();
    if (i >= n) return;
    long r = i / d;
    out[i] = table[(long)idx[r] * d + i % d];
}
// ---- Video Sparse Attention -------------------------------------------------
// Tiles are 64 padded slots; slot_src maps a slot to its token, or -1 for
// padding. Coarse means divide by the tile's REAL token count, so a partial
// tile is not diluted toward zero.
extern "C" __global__ void vsa_tile_mean(
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
            acc += xb[(long)tok * dim + d];
        }
        ob[d] = n > 0 ? acc / (float)n : 0.0f;
    }
}
// Round-to-nearest f32 -> bf16 bits, as cast_f32_bf16 does.
__device__ __forceinline__ unsigned short fv_to_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int hi = u >> 16;
    if ((u & 0x8000u) != 0u && (hi & 0x7F80u) != 0x7F80u && (hi & 0x7FFFu) != 0x7F7Fu) hi += 1u;
    return (unsigned short)hi;
}
// Order-preserving map from float to unsigned so integer compares sort floats.
__device__ __forceinline__ unsigned int fv_sortable(float x) {
    unsigned int u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// Top-k column indices per score row. A binary search on the sortable key
// (32 counting passes) beats an O(n*k) scan at n=819, k=164; the final gather
// is single-threaded so ties resolve to the lower index, matching the host
// reference exactly.
extern "C" __global__ void vsa_topk(
    const float* scores, unsigned int* out, int rows, int n, int k
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* s = scores + (long)row * n;
    extern __shared__ int counts[];
    __shared__ unsigned int lo_s;
    int tid = threadIdx.x;
    unsigned int lo = 0u, hi = 0xFFFFFFFFu;
    // Largest threshold T with count(key >= T) >= k.
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2u + ((hi - lo) & 1u);
        int local = 0;
        for (int i = tid; i < n; i += blockDim.x) local += (fv_sortable(s[i]) >= mid) ? 1 : 0;
        counts[tid] = local;
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) counts[tid] += counts[tid + off];
            __syncthreads();
        }
        int total = counts[0];
        __syncthreads();
        if (total >= k) lo = mid; else hi = mid - 1u;
    }
    if (tid == 0) lo_s = lo;
    __syncthreads();
    unsigned int thr = lo_s;
    if (tid == 0) {
        unsigned int* dst = out + (long)row * k;
        int written = 0;
        for (int i = 0; i < n && written < k; i++) if (fv_sortable(s[i]) > thr) dst[written++] = (unsigned int)i;
        for (int i = 0; i < n && written < k; i++) if (fv_sortable(s[i]) == thr) dst[written++] = (unsigned int)i;
        for (; written < k; written++) dst[written] = 0u;
    }
}
// Gather the selected tiles' K and V rows into a dense per-query-tile buffer
// so the fine stage is a batched GEMM. Padding slots are zeroed; the score
// mask, not the zeros, is what keeps them out of the softmax.
extern "C" __global__ void vsa_gather_kv(
    const float* k, const float* v, const unsigned int* selected, const int* slot_src,
    unsigned short* kg, unsigned short* vg, long seq, int dim, int topk, int tile_elems,
    int q_base, int num_tiles
) {
    long slot = (long)blockIdx.x * blockDim.y + threadIdx.y;   // slot within the gathered list
    long g = blockIdx.y;                                        // query tile within the group
    long bh = blockIdx.z;
    long len = (long)topk * tile_elems;
    if (slot >= len) return;
    int tile_pos = (int)(slot / tile_elems), within = (int)(slot % tile_elems);
    unsigned int kt = selected[((bh * (long)num_tiles) + q_base + g) * (long)topk + tile_pos];
    int src = slot_src[(long)kt * tile_elems + within];
    long dst = ((bh * gridDim.y + g) * len + slot) * (long)dim;
    const float* kb = k + bh * seq * (long)dim;
    const float* vb = v + bh * seq * (long)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float kv = 0.0f, vv = 0.0f;
        if (src >= 0) { kv = kb[(long)src * dim + d]; vv = vb[(long)src * dim + d]; }
        kg[dst + d] = fv_to_bf16(kv);
        vg[dst + d] = fv_to_bf16(vv);
    }
}
// Gather one group of query tiles into padded slot order, bf16.
extern "C" __global__ void vsa_gather_q(
    const float* q, const int* slot_src, unsigned short* out,
    long seq, int dim, int tile_elems, int q_base
) {
    long slot = (long)blockIdx.x * blockDim.y + threadIdx.y;
    long g = blockIdx.y;
    long bh = blockIdx.z;
    if (slot >= tile_elems) return;
    int tile = q_base + (int)g;
    int src = slot_src[(long)tile * tile_elems + slot];
    long dst = ((bh * gridDim.y + g) * (long)tile_elems + slot) * (long)dim;
    const float* qb = q + bh * seq * (long)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        out[dst + d] = fv_to_bf16(src >= 0 ? qb[(long)src * dim + d] : 0.0f);
    }
}
// Fused block-sparse attention: one CUDA block per (query tile, batch-head).
//
// The gather path materialises every query tile's selected K/V into a dense
// buffer so cuBLAS can run the fine stage on tensor cores. This streams the
// same tiles straight out of the tiled layout instead, trading ~80 GB/layer of
// traffic for scalar math. Whether that is a win is a measurement, not a
// theorem: cuBLAS bf16 runs at roughly twice the scalar f32 rate.
//
// 256 threads = 8 warps; warp w owns queries [8w, 8w+8), lane l owns key l of
// the current 32-key half tile and dims [4l, 4l+4) of the output. Every K/V
// tile load is therefore amortised over all 64 queries — the thing the earlier
// flash kernel got wrong by giving each block a single query.
#define VSA_Q 64
#define VSA_KH 32
__device__ __forceinline__ float fv_bf16_to_f32(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}
extern "C" __global__ void vsa_fused_attn(
    const float* q, const float* k, const float* v, const unsigned int* selected,
    const int* slot_src, const int* block_sizes, float* out,
    long seq, int dim, int topk, int num_tiles, float scale
) {
    int qt = blockIdx.x;
    long bh = blockIdx.y;
    if (qt >= num_tiles) return;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // Output dims are split across the warp's 32 lanes; dim=128 gives 4 each,
    // dim=64 gives 2. Assuming 4 reads past the row for anything narrower.
    int dpl = dim >> 5;

    extern __shared__ unsigned char vsa_smem[];
    unsigned short* Qs = (unsigned short*)vsa_smem;
    unsigned short* Ks = Qs + (long)VSA_Q * dim;
    unsigned short* Vs = Ks + (long)VSA_KH * dim;
    float* Ps = (float*)(Vs + (long)VSA_KH * dim);

    const float* qb = q + bh * seq * (long)dim;
    const float* kb = k + bh * seq * (long)dim;
    const float* vb = v + bh * seq * (long)dim;

    // Q tile stays resident for the whole block: it is read once per kv tile.
    for (int i = tid; i < VSA_Q * dim; i += blockDim.x) {
        int slot = i / dim, d = i - slot * dim;
        int src = slot_src[(long)qt * VSA_Q + slot];
        float val = src >= 0 ? qb[(long)src * dim + d] : 0.0f;
        Qs[i] = fv_to_bf16(val);
    }
    __syncthreads();

    float acc[8][4];
    float m_run[8], l_run[8];
    for (int qi = 0; qi < 8; qi++) {
        m_run[qi] = -3.402823466e+38f;
        l_run[qi] = 0.0f;
        for (int dd = 0; dd < 4; dd++) acc[qi][dd] = 0.0f;   // dpl <= 4 entries used
    }
    const float neg_inf = __int_as_float(0xff800000);

    for (int slot_i = 0; slot_i < topk; slot_i++) {
        unsigned int kt = selected[((bh * (long)num_tiles) + qt) * (long)topk + slot_i];
        int valid = block_sizes[kt];
        for (int half = 0; half < 2; half++) {
            int base = half * VSA_KH;
            for (int i = tid; i < VSA_KH * dim; i += blockDim.x) {
                int slot = i / dim, d = i - slot * dim;
                int src = slot_src[(long)kt * VSA_Q + base + slot];
                float kv = 0.0f, vv = 0.0f;
                if (src >= 0) { kv = kb[(long)src * dim + d]; vv = vb[(long)src * dim + d]; }
                Ks[i] = fv_to_bf16(kv);
                Vs[i] = fv_to_bf16(vv);
            }
            __syncthreads();

            // Scores: each lane takes one key, each warp eight queries.
            float s[8];
            int key_ok = (base + lane) < valid;
            for (int qi = 0; qi < 8; qi++) {
                int qrow = warp * 8 + qi;
                float dot = 0.0f;
                const unsigned short* qrow_p = Qs + (long)qrow * dim;
                const unsigned short* krow_p = Ks + (long)lane * dim;
                for (int d = 0; d < dim; d++) dot += fv_bf16_to_f32(qrow_p[d]) * fv_bf16_to_f32(krow_p[d]);
                s[qi] = key_ok ? dot * scale : neg_inf;
            }
            // Online softmax over this half tile, reduced across the warp's lanes.
            for (int qi = 0; qi < 8; qi++) {
                float m_tile = s[qi];
                for (int off = 16; off > 0; off >>= 1) m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, off));
                float m_new = fmaxf(m_run[qi], m_tile);
                float e = (s[qi] == neg_inf) ? 0.0f : expf(s[qi] - m_new);
                Ps[(warp * 8 + qi) * VSA_KH + lane] = e;
                float sum = e;
                for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
                float corr = (m_run[qi] == -3.402823466e+38f) ? 0.0f : expf(m_run[qi] - m_new);
                l_run[qi] = l_run[qi] * corr + sum;
                for (int dd = 0; dd < dpl; dd++) acc[qi][dd] *= corr;
                m_run[qi] = m_new;
            }
            __syncthreads();
            // P @ V: lane owns four output dims, loops the 32 keys.
            for (int qi = 0; qi < 8; qi++) {
                int qrow = warp * 8 + qi;
                const float* prow = Ps + (long)qrow * VSA_KH;
                for (int key = 0; key < VSA_KH; key++) {
                    float p = prow[key];
                    if (p == 0.0f) continue;
                    const unsigned short* vrow = Vs + (long)key * dim;
                    for (int dd = 0; dd < dpl; dd++) acc[qi][dd] += p * fv_bf16_to_f32(vrow[lane * dpl + dd]);
                }
            }
            __syncthreads();
        }
    }
    // Write the tile's rows in padded slot order; vsa_combine scatters them.
    for (int qi = 0; qi < 8; qi++) {
        int qrow = warp * 8 + qi;
        float inv = l_run[qi] > 0.0f ? 1.0f / l_run[qi] : 0.0f;
        float* orow = out + ((bh * (long)num_tiles + qt) * VSA_Q + qrow) * (long)dim;
        for (int dd = 0; dd < dpl; dd++) orow[lane * dpl + dd] = acc[qi][dd] * inv;
    }
}
// -inf the score columns that land on tile padding, so the softmax ignores
// them instead of treating a zeroed key as a real one scoring 0.
extern "C" __global__ void vsa_mask_pad(
    float* scores, const unsigned int* selected, const int* block_sizes,
    int rows_per_tile, int topk, int tile_elems, int q_base, int num_tiles
) {
    long col = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long g = blockIdx.y;
    long bh = blockIdx.z;
    long len = (long)topk * tile_elems;
    if (col >= len) return;
    int tile_pos = (int)(col / tile_elems), within = (int)(col % tile_elems);
    unsigned int kt = selected[((bh * (long)num_tiles) + q_base + g) * (long)topk + tile_pos];
    if (within < block_sizes[kt]) return;
    float* base = scores + (bh * gridDim.y + g) * (long)rows_per_tile * len;
    // NVRTC compiles without <math.h>, so spell -inf as its bit pattern.
    float neg_inf = __int_as_float(0xff800000);
    for (int r = 0; r < rows_per_tile; r++) base[(long)r * len + col] = neg_inf;
}
// out[token] = coarse[tile] * gate[token] + sparse[slot], scattering the
// padded tile layout back to token order. gate == null means a gate of one.
extern "C" __global__ void vsa_combine(
    const float* sparse, const float* coarse, const float* gate, const int* slot_src,
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
    const float* ga = has_gate ? gate + bh * seq * (long)dim + (long)src * dim : nullptr;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        ob[d] = co[d] * (ga ? ga[d] : 1.0f) + sp[d];
    }
}
// Tiled flash attention (online softmax, O(d) memory per block). Opt-in via
// FASTVIDEO_SDPA=flash: the barrier syncs per key tile make it slower than
// dense cuBLAS attention at the sequence lengths this crate runs.
#define FA_BK 32
#define FA_WARP 32
extern "C" __global__ void flash_attn_f32(
    const float* Q, const float* K, const float* V, float* O,
    int bh, int sq, int sk, int d, float scale
) {
    int bh_idx = blockIdx.x / sq;
    int q_i    = blockIdx.x % sq;
    if (bh_idx >= bh) return;
    int tid     = threadIdx.x;
    int n_warps = d / FA_WARP;
    int warp_id = tid / FA_WARP;
    int lane    = tid % FA_WARP;

    extern __shared__ float smem[];
    float* Ksh       = smem;
    float* Vsh       = smem + FA_BK * d;
    float* sc_partial = smem + 2 * FA_BK * d;

    float q_val = Q[((long)bh_idx * sq + q_i) * d + tid];
    float o_val = 0.0f;
    float m_val = -3.402823466e+38f;
    float l_val = 0.0f;

    for (int k0 = 0; k0 < sk; k0 += FA_BK) {
        int klen = (sk - k0 < FA_BK) ? (sk - k0) : FA_BK;
        for (int ki = 0; ki < klen; ki++) {
            long base = ((long)bh_idx * sk + k0 + ki) * d;
            Ksh[ki * d + tid] = K[base + tid];
            Vsh[ki * d + tid] = V[base + tid];
        }
        __syncthreads();
        for (int ki = 0; ki < klen; ki++) {
            float dot = q_val * Ksh[ki * d + tid];
            for (int offset = FA_WARP / 2; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            if (lane == 0) sc_partial[ki * n_warps + warp_id] = dot;
        }
        __syncthreads();
        if (tid < klen) {
            float s = 0.0f;
            for (int w = 0; w < n_warps; w++) s += sc_partial[tid * n_warps + w];
            Ksh[tid] = s * scale;
        }
        __syncthreads();
        float m_tile = -3.402823466e+38f;
        for (int ki = 0; ki < klen; ki++) m_tile = fmaxf(m_tile, Ksh[ki]);
        float m_new      = fmaxf(m_val, m_tile);
        float exp_rescale = expf(m_val - m_new);
        if (tid < klen) Ksh[tid] = expf(Ksh[tid] - m_new);
        __syncthreads();
        o_val *= exp_rescale;
        for (int ki = 0; ki < klen; ki++)
            o_val += Ksh[ki] * Vsh[ki * d + tid];
        float sum_exp = 0.0f;
        for (int ki = 0; ki < klen; ki++) sum_exp += Ksh[ki];
        l_val = l_val * exp_rescale + sum_exp;
        m_val = m_new;
        __syncthreads();
    }
    O[((long)bh_idx * sq + q_i) * d + tid] = o_val / l_val;
}

// ---- VSA fine stage on tensor cores ----------------------------------------
//
// The reference block-sparse kernels (FastVideo's Triton `_attn_fwd_sparse`,
// its ThunderKittens sm90 kernel, FlashAttention-4's block sparsity) never
// gather the selected K/V tiles. They permute Q/K/V into tile-contiguous order
// ONCE, so tile j is the contiguous slab of rows [64j, 64j+64), then one CUDA
// block per query tile walks its top-k list re-pointing at each slab and folds
// it into an online softmax. Our previous fused attempt had that structure and
// still lost 8x, because it did the math in scalar f32; every reference kernel
// does it on tensor cores. This one does too: `mma.sync.m16n8k16` bf16 with
// `ldmatrix` fragment loads and `cp.async` double buffering, which is the
// sm80+ version of what the ThunderKittens kernel does with wgmma and TMA.
//
// Layout of one tile in shared memory: 64 rows x 128 bf16 = 256 B/row, as 16
// chunks of 16 B. Chunk c of row r is stored at chunk (c ^ (r & 7)): without
// that XOR every row of an 8x8 ldmatrix starts on the same bank.
//
// Fragment maps (PTX ISA, m16n8k16, lane l, g = l>>2, t = l&3):
//   A 16x16 row: reg0 = (g, 2t..2t+1)  reg1 = (g+8, 2t..)  reg2 = (g, 2t+8..)  reg3 = (g+8, 2t+8..)
//   B 16x8 col:  reg0 = (k=2t..2t+1, n=g)  reg1 = (k=2t+8.., n=g)
//   C 16x8 f32:  c0,c1 = (g, 2t..2t+1)  c2,c3 = (g+8, 2t..2t+1)
// The C layout of S lines up with the A layout of P for the PV product, so P
// never leaves registers: n-tiles (2j, 2j+1) of S become k-chunk j of P.

#define MMA_TILE 64
#define MMA_DIM 128
#define MMA_ROWB 256   // bytes per tile row
#define MMA_TILEB (MMA_TILE * MMA_ROWB)

// Byte offset of (row, col) in a swizzled tile; `col` in bf16 elements, and a
// multiple of 8 wherever ldmatrix reads it.
__device__ __forceinline__ unsigned int mma_swz(int row, int col) {
    return (unsigned int)row * MMA_ROWB + ((((unsigned int)col >> 3) ^ ((unsigned int)row & 7u)) << 4) + (((unsigned int)col & 7u) << 1);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned int mma_smem_u32(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mma_cp_async16(unsigned int smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem));
}
__device__ __forceinline__ void mma_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void mma_cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void mma_ldm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_ldm_x2(unsigned int addr, unsigned int& r0, unsigned int& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void mma_ldm_x2_trans(unsigned int addr, unsigned int& r0, unsigned int& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const unsigned int* a, const unsigned int* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned int mma_pack_bf16(float lo, float hi) {
    return (unsigned int)fv_to_bf16(lo) | ((unsigned int)fv_to_bf16(hi) << 16);
}
// Copy one 64x128 bf16 tile (row-major, 256 B rows) from global into a
// swizzled shared buffer: 1024 chunks of 16 B over 128 threads.
__device__ __forceinline__ void mma_load_tile(unsigned int smem, const unsigned short* g, int tid) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int chunk = tid + i * 128;
        int row = chunk >> 4, c = chunk & 15;
        mma_cp_async16(smem + mma_swz(row, c * 8), g + row * MMA_DIM + c * 8);
    }
}
#endif

// One CUDA block per (query tile, batch*head); 4 warps, warp w owns query rows
// [16w, 16w+16). Q/K/V are the tile-ordered bf16 tensors from vsa_tile_qkv,
// `selected` is the top-k key-tile list per query tile, `block_sizes` the real
// token count per tile (padding columns are masked to -inf, as the reference
// does with `right_fill`). Output is f32 in the same tile-slot order the
// gather path produces, so `vsa_combine` is shared.
extern "C" __global__ void __launch_bounds__(128, 2) vsa_mma_attn(
    const unsigned short* __restrict__ qt, const unsigned short* __restrict__ kt,
    const unsigned short* __restrict__ vt, const unsigned int* __restrict__ selected,
    const int* __restrict__ block_sizes, float* __restrict__ out,
    int num_tiles, int topk, float scale_log2
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(128) unsigned char mma_smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int qtile = blockIdx.x, bh = blockIdx.y;
    const long padded = (long)num_tiles * MMA_TILE;
    const unsigned short* qb = qt + (bh * padded + (long)qtile * MMA_TILE) * MMA_DIM;
    const unsigned short* kb = kt + bh * padded * MMA_DIM;
    const unsigned short* vb = vt + bh * padded * MMA_DIM;
    const unsigned int* sel = selected + ((long)bh * num_tiles + qtile) * topk;

    // Buffers: K[2], V[2] — 4 x 16 KB.
    const unsigned int s_base = mma_smem_u32(mma_smem);
    const unsigned int sK[2] = { s_base, s_base + MMA_TILEB };
    const unsigned int sV[2] = { s_base + 2 * MMA_TILEB, s_base + 3 * MMA_TILEB };

    // Stage Q through K[0] into registers once: 8 k-chunks of A fragments.
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

    // Prime the pipeline with tile 0.
    unsigned int kt0 = sel[0];
    mma_load_tile(sK[0], kb + (long)kt0 * MMA_TILE * MMA_DIM, tid);
    mma_load_tile(sV[0], vb + (long)kt0 * MMA_TILE * MMA_DIM, tid);
    mma_cp_commit();

    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    const float NEG = __int_as_float(0xff800000);
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;   // rows g and g+8

    for (int i = 0; i < topk; i++) {
        const int buf = i & 1;
        // Issue the next tile's loads before touching this one, as the
        // reference does: index i+1 is read while i computes.
        if (i + 1 < topk) {
            unsigned int kn = sel[i + 1];
            mma_load_tile(sK[buf ^ 1], kb + (long)kn * MMA_TILE * MMA_DIM, tid);
            mma_load_tile(sV[buf ^ 1], vb + (long)kn * MMA_TILE * MMA_DIM, tid);
            mma_cp_commit();
            mma_cp_wait<1>();
        } else {
            mma_cp_wait<0>();
        }
        __syncthreads();

        // S = Q K^T for this warp's 16 rows x 64 keys: 8 n-tiles x 8 k-chunks.
        float s[8][4];
        #pragma unroll
        for (int n = 0; n < 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
        #pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                unsigned int b[2];
                // 8 keys (rows n*8..) x 16 dims (cols kc*16..), no transpose.
                mma_ldm_x2(sK[buf] + mma_swz(n * 8 + (lane & 7), kc * 16 + ((lane >> 3) & 1) * 8), b[0], b[1]);
                mma_bf16(s[n], qf[kc], b);
            }
        }

        // Mask padding columns of this key tile, then the online softmax in
        // the reference's order: new max, rescale old, exp2, accumulate.
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
        // A row with no valid key yet keeps its scale at 0 rather than NaN.
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

        // O += P V: P re-packed from C layout into A layout, V via ldmatrix.trans.
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
                // 16 keys (rows kc*16..) x 8 dims (cols n*8..), transposed.
                mma_ldm_x2_trans(sV[buf] + mma_swz(kc * 16 + (lane & 15), n * 8), b[0], b[1]);
                mma_bf16(o[n], pa, b);
            }
        }
        __syncthreads();
    }

    // Normalise and write this warp's 16 rows, f32, tile-slot order.
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
    // sm75 has no bf16 mma; the dispatcher never launches this there.
    (void)qt; (void)kt; (void)vt; (void)selected; (void)block_sizes; (void)out;
    (void)num_tiles; (void)topk; (void)scale_log2;
#endif
}

// The reference's `tile()`: permute one f32 [bh, seq, dim] tensor into
// tile-contiguous bf16 [bh, padded, dim] once, zero-filling padding slots.
// O(padded x dim) — versus the per-query-tile gather's O(tiles x topk x 64 x dim).
extern "C" __global__ void vsa_tile_qkv(
    const float* __restrict__ x, const int* __restrict__ slot_src, unsigned short* __restrict__ xt,
    long seq, long padded, int dim
) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)gridDim.z * padded * dim;
    if (i >= total) return;
    long bh = i / (padded * dim);
    long rem = i - bh * padded * dim;
    long slot = rem / dim;
    int d = (int)(rem - slot * dim);
    int src = slot_src[slot];
    float v = src >= 0 ? x[(bh * seq + src) * dim + d] : 0.f;
    xt[i] = fv_to_bf16(v);
}

// ---- FP8 E4M3 ------------------------------------------------------------
// NVRTC here has no cuda_fp8.h, so the conversion is open-coded. This mirrors
// fastvideo_ops::fp8::f32_to_e4m3 line for line; that reference is exhaustively
// tested over all 256 codes and the kernels tier checks this against it.
// Saturating (SATFINITE): out-of-range clamps to +-448 rather than becoming
// NaN, because one NaN poisons every remaining denoising step.
__device__ __forceinline__ unsigned char fv_to_e4m3(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int sign = (u >> 24) & 0x80u;
    unsigned int mag = u & 0x7FFFFFFFu;
    if (mag >= 0x7F800000u || mag >= 0x43E00000u) return (unsigned char)(sign | 0x7Eu);
    int exp = (int)(mag >> 23) - 127;
    unsigned int man = mag & 0x007FFFFFu;
    unsigned int out;
    if (exp >= -6) {
        unsigned int m = man >> 20;
        unsigned int rem = man & 0x000FFFFFu;
        unsigned int half = 1u << 19;
        if (rem > half || (rem == half && (m & 1u) == 1u)) m += 1u;
        unsigned int e = (unsigned int)(exp + 7);
        if (m == 8u) { m = 0u; e += 1u; }
        if (e > 15u || (e == 15u && m >= 7u)) return (unsigned char)(sign | 0x7Eu);
        out = (e << 3) | m;
    } else {
        unsigned int shift = (unsigned int)(20 + (-6 - exp));
        if (shift > 31u) return (unsigned char)sign;
        unsigned int full = (1u << 23) | man;
        unsigned int m = full >> shift;
        unsigned int rem = full & ((1u << shift) - 1u);
        unsigned int half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (m & 1u) == 1u)) m += 1u;
        out = m;
    }
    return (unsigned char)(sign | out);
}

__device__ __forceinline__ float fv_e4m3_to_f32(unsigned char b) {
    float sign = (b & 0x80u) ? -1.0f : 1.0f;
    int e = (int)((b >> 3) & 0x0Fu);
    unsigned int m = (unsigned int)(b & 0x07u);
    if (e == 0) return sign * (float)m * (1.0f / 512.0f);
    float frac = 1.0f + (float)m * 0.125f;
    return sign * frac * __int_as_float((e - 7 + 127) << 23);
}

// x * inv_scale -> E4M3. `inv_scale` is a device scalar so the scale can come
// from amax_abs without a host round trip.
extern "C" __global__ void quantize_e4m3(
    const float* a, unsigned char* out, const float* inv_scale, long n
) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fv_to_e4m3(a[i] * *inv_scale);
}

// Dequantize for the reference path and for checking the kernel against it.
extern "C" __global__ void dequantize_e4m3(
    const unsigned char* a, float* out, const float* scale, long n
) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fv_e4m3_to_f32(a[i]) * *scale;
}

// max(|a|) over the whole tensor, block-reduced then atomically combined.
// `out` must be zeroed by the caller.
extern "C" __global__ void amax_abs(const float* a, float* out, long n) {
    extern __shared__ float sm[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (long i = blockIdx.x * (long)blockDim.x + tid; i < n; i += (long)gridDim.x * blockDim.x) {
        float v = fabsf(a[i]);
        // A NaN weight must not silently become a 0 scale; propagate it so the
        // gate downstream can refuse rather than quantize garbage.
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

// amax -> (scale, inv_scale) on the device, so the GEMM's scale pointers can be
// filled without stalling on a download.
extern "C" __global__ void e4m3_scale_from_amax(const float* amax, float* scale, float* inv_scale) {
    float a = *amax;
    if (!(a > 0.0f) || !isfinite(a)) { *scale = 1.0f; *inv_scale = 1.0f; return; }
    float s = a * (1.0f / 448.0f);
    *scale = s;
    *inv_scale = 1.0f / s;
}
"#;

/// Declares [`KernelFns`] and [`KERNEL_NAMES`] from one list, so a kernel
/// can't be compiled but not loaded (or vice versa).
macro_rules! kernel_fns {
    ($($name:ident),+ $(,)?) => {
        pub struct KernelFns {
            $(pub $name: CudaFunction,)+
        }

        /// Every `__global__` entry point in [`KERNEL_SRC`], in load order.
        pub const KERNEL_NAMES: &[&str] = &[$(stringify!($name)),+];

        impl KernelFns {
            fn load(module: &Arc<cudarc::driver::CudaModule>) -> Result<Self> {
                Ok(Self {
                    $($name: module.load_function(stringify!($name))?,)+
                })
            }
        }
    };
}

kernel_fns!(
    vsa_mma_attn,
    vsa_tile_qkv,
    tanh_scaled,
    quantize_e4m3,
    dequantize_e4m3,
    amax_abs,
    e4m3_scale_from_amax,
    elem_add,
    elem_mul,
    elem_sub,
    mul_scalar,
    add_scalar,
    silu,
    gelu_tanh,
    clamp_f,
    fill_f,
    lincomb3,
    bcast_binary,
    add_bias_inplace,
    bias_gelu_inplace,
    cast_f32_bf16,
    cast_bf16_f32_bias_act,
    residual_gate_add_e,
    softmax_last,
    softmax_last_bf16,
    rms_norm_last,
    layer_norm_last,
    ln_adaln_e,
    qk_norm_rope_bhsd,
    split_heads_bhsd,
    merge_heads,
    gather_nd,
    block_copy,
    upsample_nearest,
    rms_norm_channels,
    temporal_unfold,
    index_select_rows,
    flash_attn_f32,
    vsa_tile_mean,
    vsa_topk,
    vsa_gather_kv,
    vsa_gather_q,
    vsa_fused_attn,
    vsa_mask_pad,
    vsa_combine,
);

/// NVRTC-compile the kernel module for `sm_major.sm_minor` without touching a
/// GPU. NVRTC only needs `libnvrtc`, so this runs on any Linux box with the
/// CUDA runtime libraries — a free gate before renting hardware.
pub fn compile_ptx(sm_major: i32, sm_minor: i32) -> Result<cudarc::nvrtc::Ptx> {
    let arch = super::hopper::nvrtc_arch(sm_major, sm_minor);
    let opts = CompileOptions {
        arch,
        use_fast_math: Some(true),
        ftz: Some(true),
        // Do not also set `fmad`: use_fast_math already injects --fmad=true.
        ..Default::default()
    };
    compile_ptx_with_opts(KERNEL_SRC, opts).map_err(|e| {
        DeviceError::Message(format!("nvrtc compile failed (arch={arch:?}): {e}"))
    })
}

impl KernelFns {
    pub fn compile(
        ctx: &Arc<cudarc::driver::CudaContext>,
        sm_major: i32,
        sm_minor: i32,
    ) -> Result<Self> {
        let ptx = compile_ptx(sm_major, sm_minor)?;
        let module = ctx.load_module(ptx)?;
        Self::load(&module)
    }
}

/// One thread per element, 1024 threads per block.
pub fn cfg_n(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n.max(1) as u32)
}

/// One block per row, `ROW_BLOCK_THREADS` threads/block, with dynamic shared
/// memory for the tree reduction. 256 is a power of two (the reduction needs
/// one) and a good fit for the hidden widths this crate sees.
pub const ROW_BLOCK_THREADS: u32 = 256;

pub fn cfg_rows(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.max(1) as u32, 1, 1),
        block_dim: (ROW_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: ROW_BLOCK_THREADS * std::mem::size_of::<f32>() as u32,
    }
}

/// Tiled flash attention: grid = bh*sq blocks, block = d threads.
pub fn cfg_flash(bh: usize, sq: usize, d: usize) -> LaunchConfig {
    let d_u = d as u32;
    let smem_bytes = (2 * 32 * d_u + 32 * (d_u / 32)) * std::mem::size_of::<f32>() as u32;
    LaunchConfig {
        grid_dim: ((bh * sq).max(1) as u32, 1, 1),
        block_dim: (d_u, 1, 1),
        shared_mem_bytes: smem_bytes,
    }
}

/// Launch `$f` on `$stream` with `$cfg`, pushing each argument in order.
/// Scalars are passed by reference (`&n`), buffers as `&slice`/`&mut slice`.
macro_rules! launch {
    ($stream:expr, $f:expr, $cfg:expr; $($arg:expr),+ $(,)?) => {
        ({
            use cudarc::driver::PushKernelArg as _;
            super::stats::record_launch();
            let mut builder = $stream.launch_builder($f);
            $( builder.arg($arg); )+
            unsafe { builder.launch($cfg) }.map(|_| ()).map_err(super::device::DeviceError::from)
        })
    };
}
pub(crate) use launch;
