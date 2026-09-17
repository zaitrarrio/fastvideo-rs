//! NVRTC-compiled CUDA kernels for elementwise / reduction ops.
//!
//! Compiled once into [`crate::wan::device::DeviceContext`] when CUDA is initialized.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::device::{DeviceError, Result};

const KERNEL_SRC: &str = r#"
extern "C" __global__ void elem_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void elem_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
extern "C" __global__ void elem_sub(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}
extern "C" __global__ void mul_scalar(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}
extern "C" __global__ void add_scalar(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + s;
}
extern "C" __global__ void silu(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        out[i] = x / (1.0f + expf(-x));
    }
}
extern "C" __global__ void gelu_tanh(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        const float k = 0.7978845608028654f; // sqrt(2/pi)
        float u = k * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(u));
    }
}
extern "C" __global__ void clamp_f(const float* a, float lo, float hi, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        out[i] = fminf(hi, fmaxf(lo, x));
    }
}
// Softmax along contiguous trailing axis of length `width` (rows = n / width).
// One block per row, `blockDim.x` threads cooperating via shared-memory tree
// reduction. Each thread strides across the row (`j += nthreads`), so
// adjacent threads touch adjacent addresses (coalesced), and the max/sum
// passes are parallel instead of one thread serially walking `width`
// elements three times. `width` can exceed `blockDim.x`; the grid-stride
// loop handles that. Dynamic shared memory: `blockDim.x * sizeof(float)`.
extern "C" __global__ void softmax_last(const float* a, float* out, int rows, int width) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = a + (size_t)row * (size_t)width;
    float* dst = out + (size_t)row * (size_t)width;

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
// RMS norm over last dim `width` (rows = n / width). weight length = width.
// One block per row (see `softmax_last` for the reduction/coalescing rationale).
extern "C" __global__ void rms_norm_last(
    const float* a, const float* w, float* out, int rows, int width, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = a + (size_t)row * (size_t)width;
    float* dst = out + (size_t)row * (size_t)width;

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
// LayerNorm over last dim `width` (rows = n / width). `weight`/`bias` may be
// null (unaffine norm) — `has_affine` gates that branch uniformly across the
// block, so it costs a predictable predicate, not divergence. One block per
// row, same reduction/coalescing shape as `softmax_last`/`rms_norm_last`.
extern "C" __global__ void layer_norm_last(
    const float* a, const float* w, const float* b, float* out,
    int rows, int width, float eps, int has_affine
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = a + (size_t)row * (size_t)width;
    float* dst = out + (size_t)row * (size_t)width;

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
        for (int j = tid; j < width; j += nthreads) {
            dst[j] = (src[j] - mean) * inv * w[j] + b[j];
        }
    } else {
        for (int j = tid; j < width; j += nthreads) {
            dst[j] = (src[j] - mean) * inv;
        }
    }
}
// Fused AdaLN modulate: out[b,l,d] = x[b,l,d] * (1 + scale[b,d]) + shift[b,d].
// `x` is [batch, seq, dim] contiguous; `scale`/`shift` are [batch, dim]
// (broadcast over `seq`). One thread per output element; batch/seq/dim index
// derived from the flat index so this replaces the CPU `broadcast_bin` path
// used for AdaLN's `mul(scale+1).add(shift)` pair with a single kernel launch
// and no host round trip.
extern "C" __global__ void modulate_scale_shift_last(
    const float* x, const float* scale, const float* shift, float* out,
    int batch, int seq, int dim
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)batch * seq * dim;
    if (idx >= total) return;
    int d = idx % dim;
    long bl = idx / dim;
    int b = bl / seq;
    long bd = (long)b * dim + d;
    out[idx] = x[idx] * (1.0f + scale[bd]) + shift[bd];
}
// Broadcast multiply: out[b,l,d] = x[b,l,d] * gate[b,d]. Same broadcast shape
// as `modulate_scale_shift_last`; used for AdaLN's gated residual add.
extern "C" __global__ void broadcast_mul_last(
    const float* x, const float* gate, float* out,
    int batch, int seq, int dim
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)batch * seq * dim;
    if (idx >= total) return;
    int d = idx % dim;
    long bl = idx / dim;
    int b = bl / seq;
    long bd = (long)b * dim + d;
    out[idx] = x[idx] * gate[bd];
}
// In-place add bias along last dim: out[i] += bias[i % width].
extern "C" __global__ void add_bias_last(float* out, const float* bias, int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += bias[i % width];
}
// Cast f32 -> bf16 bits stored as ushort. bfloat16 = f32's sign + 8-bit
// exponent + top 7 mantissa bits; round to nearest on the dropped bits (plain
// truncation biases every value toward zero). Carry into the exponent is the
// correct round-up; inf/NaN and the largest finite value are left unrounded.
extern "C" __global__ void f32_to_bf16(const float* a, unsigned short* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int u = __float_as_uint(a[i]);
        unsigned int hi = u >> 16;
        if ((u & 0x8000u) != 0u && (hi & 0x7F80u) != 0x7F80u && (hi & 0x7FFFu) != 0x7F7Fu) {
            hi += 1u;
        }
        out[i] = (unsigned short)hi;
    }
}
extern "C" __global__ void bf16_to_f32(const unsigned short* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int u = ((unsigned int)a[i]) << 16;
        out[i] = __uint_as_float(u);
    }
}
// Dense causal conv3d for small kernels: out[n,oc,ot,oh,ow]
// weight [oc, ic, kt, kh, kw], input already padded NCDHW.
extern "C" __global__ void causal_conv3d_f32(
    const float* x, const float* w, const float* bias, float* out,
    int n, int ic, int it, int ih, int iw,
    int oc, int kt, int kh, int kw,
    int ot, int oh, int ow,
    int st, int sh, int sw
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * oc * ot * oh * ow;
    if (idx >= total) return;
    int ow_i = idx % ow; int t1 = idx / ow;
    int oh_i = t1 % oh; t1 /= oh;
    int ot_i = t1 % ot; t1 /= ot;
    int oc_i = t1 % oc; int n_i = t1 / oc;
    float acc = bias ? bias[oc_i] : 0.0f;
    for (int c = 0; c < ic; ++c) {
        for (int kt_i = 0; kt_i < kt; ++kt_i) {
            int it_i = ot_i * st + kt_i;
            if (it_i >= it) continue;
            for (int kh_i = 0; kh_i < kh; ++kh_i) {
                int ih_i = oh_i * sh + kh_i;
                if (ih_i >= ih) continue;
                for (int kw_i = 0; kw_i < kw; ++kw_i) {
                    int iw_i = ow_i * sw + kw_i;
                    if (iw_i >= iw) continue;
                    int x_i = ((((n_i * ic + c) * it + it_i) * ih + ih_i) * iw + iw_i);
                    int w_i = ((((oc_i * ic + c) * kt + kt_i) * kh + kh_i) * kw + kw_i);
                    acc += x[x_i] * w[w_i];
                }
            }
        }
    }
    out[idx] = acc;
}
// out[i] = in[permuted coords]; dims[o] = input axis feeding output axis o.
extern "C" __global__ void permute_4d(
    const float* in, float* out,
    int is0, int is1, int is2, int is3,
    int d0, int d1, int d2, int d3
) {
    int in_shape[4] = {is0, is1, is2, is3};
    int dims[4] = {d0, d1, d2, d3};
    int out_shape[4] = {in_shape[d0], in_shape[d1], in_shape[d2], in_shape[d3]};
    int total = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int t = idx;
    int oc3 = t % out_shape[3]; t /= out_shape[3];
    int oc2 = t % out_shape[2]; t /= out_shape[2];
    int oc1 = t % out_shape[1]; t /= out_shape[1];
    int oc0 = t;
    int ic[4];
    ic[d0] = oc0; ic[d1] = oc1; ic[d2] = oc2; ic[d3] = oc3;
    int in_idx = ((ic[0] * is1 + ic[1]) * is2 + ic[2]) * is3 + ic[3];
    out[idx] = in[in_idx];
}
// Contiguous-block gather used by narrow/cat/pad when trailing dims are dense.
// For each outer index o in [0, outer): copy `len` floats from
//   in[o * in_stride + in_offset] → out[o * out_stride + out_offset]
extern "C" __global__ void block_copy(
    const float* in, float* out,
    int outer, int len, int in_stride, int out_stride, int in_offset, int out_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * len;
    if (idx >= total) return;
    int o = idx / len;
    int j = idx - o * len;
    out[o * out_stride + out_offset + j] = in[o * in_stride + in_offset + j];
}
// Interleaved RoPE: last dim even. cos/sin stored with repeated pairs (Wan layout).
extern "C" __global__ void rope_interleaved(
    const float* x, const float* cos, const float* sin, float* out, int rows, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = dim >> 1;
    int total = rows * pairs;
    if (i >= total) return;
    int row = i / pairs;
    int j = i - row * pairs;
    int base = row * dim + (j << 1);
    float x1 = x[base];
    float x2 = x[base + 1];
    // Matches transformer::apply_rotary: cos from the even slot, sin from the odd slot.
    float c = cos[base];
    float s = sin[base + 1];
    out[base] = x1 * c - x2 * s;
    out[base + 1] = x1 * s + x2 * c;
}
// RMS over channel axis for NCHW / NCtHW-as-NCspatial: gamma length = c.
extern "C" __global__ void rms_norm_channels(
    const float* x, const float* gamma, float* out,
    int n, int c, int spatial, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * spatial;
    if (idx >= total) return;
    int ni = idx / spatial;
    int s = idx - ni * spatial;
    float acc = 0.0f;
    for (int ci = 0; ci < c; ++ci) {
        float v = x[(ni * c + ci) * spatial + s];
        acc += v * v;
    }
    float inv = rsqrtf(acc / (float)c + eps);
    for (int ci = 0; ci < c; ++ci) {
        int i = (ni * c + ci) * spatial + s;
        out[i] = x[i] * inv * gamma[ci];
    }
}
// In-place GELU-tanh on BF16 (stored as ushort) — used by chained BF16 FFN path.
extern "C" __global__ void gelu_tanh_bf16(unsigned short* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int u = ((unsigned int)a[i]) << 16;
    float x = __uint_as_float(u);
    const float k0 = 0.7978845608028654f;
    float t = k0 * (x + 0.044715f * x * x * x);
    float y = 0.5f * x * (1.0f + tanhf(t));
    a[i] = (unsigned short)(__float_as_uint(y) >> 16);
}
// In-place add F32 bias to a BF16 (ushort) buffer along last dim `width`.
// Used by `forward_into_bf16` so the chained FFN includes the projection bias.
extern "C" __global__ void add_bias_bf16_last(unsigned short* out, const float* bias, int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int u = ((unsigned int)out[i]) << 16;
    float val = __uint_as_float(u) + bias[i % width];
    out[i] = (unsigned short)(__float_as_uint(val) >> 16);
}
// Tiled flash attention (online softmax, O(d) peak memory per block).
// Grid: (bh * sq), Block: (d) where d = head_dim (multiple of 32).
// Shared mem: (2*FA_BK*d + FA_BK*n_warps) * sizeof(float), passed at launch.
// Uses warp shuffle for intra-warp dot-product reduction and re-uses the K
// tile buffer for scores once the K tile has been consumed, avoiding any
// extra allocation. No O(Sq*Sk) intermediate buffer is needed.
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
    float* Ksh       = smem;                          // [FA_BK * d] — K tile, reused for scores
    float* Vsh       = smem + FA_BK * d;              // [FA_BK * d]
    float* sc_partial = smem + 2 * FA_BK * d;         // [FA_BK * n_warps]

    float q_val = Q[((long)bh_idx * sq + q_i) * d + tid];
    float o_val = 0.0f;
    float m_val = -3.402823466e+38f;
    float l_val = 0.0f;

    for (int k0 = 0; k0 < sk; k0 += FA_BK) {
        int klen = (sk - k0 < FA_BK) ? (sk - k0) : FA_BK;

        // Step 1: load K and V tiles cooperatively (each thread loads column tid).
        for (int ki = 0; ki < klen; ki++) {
            long base = ((long)bh_idx * sk + k0 + ki) * d;
            Ksh[ki * d + tid] = K[base + tid];
            Vsh[ki * d + tid] = V[base + tid];
        }
        __syncthreads();  // SYNC-A: K/V tiles visible to all threads

        // Step 2: compute klen dot products using warp-shuffle intra-warp reduction.
        for (int ki = 0; ki < klen; ki++) {
            float dot = q_val * Ksh[ki * d + tid];
            for (int offset = FA_WARP / 2; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            if (lane == 0) sc_partial[ki * n_warps + warp_id] = dot;
        }
        __syncthreads();  // SYNC-B: warp partial sums visible

        // Step 3: thread tid < klen aggregates warp partials → Ksh[tid] stores score.
        if (tid < klen) {
            float s = 0.0f;
            for (int w = 0; w < n_warps; w++) s += sc_partial[tid * n_warps + w];
            Ksh[tid] = s * scale;
        }
        __syncthreads();  // SYNC-C: scores in Ksh[0..klen]

        // Step 4: online softmax — all threads compute same m_tile and m_new.
        float m_tile = -3.402823466e+38f;
        for (int ki = 0; ki < klen; ki++) m_tile = fmaxf(m_tile, Ksh[ki]);
        float m_new      = fmaxf(m_val, m_tile);
        float exp_rescale = expf(m_val - m_new);

        // Overwrite Ksh[0..klen] with softmax weights.
        if (tid < klen) Ksh[tid] = expf(Ksh[tid] - m_new);
        __syncthreads();  // SYNC-D: softmax weights in Ksh[0..klen]

        // Step 5: rescale running output and accumulate weighted V.
        o_val *= exp_rescale;
        for (int ki = 0; ki < klen; ki++)
            o_val += Ksh[ki] * Vsh[ki * d + tid];

        // Update running statistics (same value computed by every thread).
        float sum_exp = 0.0f;
        for (int ki = 0; ki < klen; ki++) sum_exp += Ksh[ki];
        l_val = l_val * exp_rescale + sum_exp;
        m_val = m_new;
        __syncthreads();  // SYNC-E: done with Ksh/Vsh before next tile overwrites them
    }

    // Normalise and write output.
    O[((long)bh_idx * sq + q_i) * d + tid] = o_val / l_val;
}
// Fused unaffine LayerNorm + AdaLN modulate (one-pass, one kernel launch).
// Replaces the `layer_norm(None,None)` + `modulate(scale, shift)` pair in
// each WanBlock pre-attention and pre-FFN norm step. Scale/shift are [batch,
// dim]; x/out are [batch, seq, dim]. One block per (batch*seq) row,
// blockDim=256, dynamic shared memory for the tree reductions.
extern "C" __global__ void layer_norm_adaln_fused(
    const float* x, const float* scale_w, const float* shift_w, float* out,
    int batch, int seq, int dim, float eps
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= batch * seq) return;
    int batch_i = row / seq;
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;
    const float* src = x + (long)row * dim;
    float*       dst = out + (long)row * dim;
    const float* sc  = scale_w + (long)batch_i * dim;
    const float* sh  = shift_w + (long)batch_i * dim;

    // Compute mean.
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

    // Compute variance.
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

    // Write LayerNorm output with AdaLN modulation: out = norm * (1+scale) + shift.
    for (int j = tid; j < dim; j += nthreads) {
        float n = (src[j] - mean) * inv;
        dst[j] = n * (1.0f + sc[j]) + sh[j];
    }
}
"#;

pub struct KernelFns {
    pub elem_add: CudaFunction,
    pub elem_mul: CudaFunction,
    pub elem_sub: CudaFunction,
    pub mul_scalar: CudaFunction,
    pub add_scalar: CudaFunction,
    pub silu: CudaFunction,
    pub gelu_tanh: CudaFunction,
    pub clamp_f: CudaFunction,
    pub softmax_last: CudaFunction,
    pub rms_norm_last: CudaFunction,
    pub layer_norm_last: CudaFunction,
    pub modulate_scale_shift_last: CudaFunction,
    pub broadcast_mul_last: CudaFunction,
    pub add_bias_last: CudaFunction,
    pub f32_to_bf16: CudaFunction,
    pub bf16_to_f32: CudaFunction,
    pub causal_conv3d_f32: CudaFunction,
    pub permute_4d: CudaFunction,
    pub block_copy: CudaFunction,
    pub rope_interleaved: CudaFunction,
    pub rms_norm_channels: CudaFunction,
    pub gelu_tanh_bf16: CudaFunction,
    pub add_bias_bf16_last: CudaFunction,
    pub flash_attn_f32: CudaFunction,
    pub layer_norm_adaln_fused: CudaFunction,
}


/// Every `__global__` entry point in [`KERNEL_SRC`], in load order.
pub const KERNEL_NAMES: &[&str] = &[
    "elem_add",
    "elem_mul",
    "elem_sub",
    "mul_scalar",
    "add_scalar",
    "silu",
    "gelu_tanh",
    "clamp_f",
    "softmax_last",
    "rms_norm_last",
    "layer_norm_last",
    "modulate_scale_shift_last",
    "broadcast_mul_last",
    "add_bias_last",
    "f32_to_bf16",
    "bf16_to_f32",
    "causal_conv3d_f32",
    "permute_4d",
    "block_copy",
    "rope_interleaved",
    "rms_norm_channels",
    "gelu_tanh_bf16",
    "add_bias_bf16_last",
    "flash_attn_f32",
    "layer_norm_adaln_fused",
];

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
        Ok(Self {
            elem_add: module.load_function("elem_add")?,
            elem_mul: module.load_function("elem_mul")?,
            elem_sub: module.load_function("elem_sub")?,
            mul_scalar: module.load_function("mul_scalar")?,
            add_scalar: module.load_function("add_scalar")?,
            silu: module.load_function("silu")?,
            gelu_tanh: module.load_function("gelu_tanh")?,
            clamp_f: module.load_function("clamp_f")?,
            softmax_last: module.load_function("softmax_last")?,
            rms_norm_last: module.load_function("rms_norm_last")?,
            layer_norm_last: module.load_function("layer_norm_last")?,
            modulate_scale_shift_last: module.load_function("modulate_scale_shift_last")?,
            broadcast_mul_last: module.load_function("broadcast_mul_last")?,
            add_bias_last: module.load_function("add_bias_last")?,
            f32_to_bf16: module.load_function("f32_to_bf16")?,
            bf16_to_f32: module.load_function("bf16_to_f32")?,
            causal_conv3d_f32: module.load_function("causal_conv3d_f32")?,
            permute_4d: module.load_function("permute_4d")?,
            block_copy: module.load_function("block_copy")?,
            rope_interleaved: module.load_function("rope_interleaved")?,
            rms_norm_channels: module.load_function("rms_norm_channels")?,
            gelu_tanh_bf16: module.load_function("gelu_tanh_bf16")?,
            add_bias_bf16_last: module.load_function("add_bias_bf16_last")?,
            flash_attn_f32: module.load_function("flash_attn_f32")?,
            layer_norm_adaln_fused: module.load_function("layer_norm_adaln_fused")?,
        })
    }
}

fn cfg_n(n: u32) -> LaunchConfig {
    LaunchConfig::for_num_elems(n.max(1))
}

/// One block per row, `ROW_BLOCK_THREADS` threads/block, with dynamic shared
/// memory for the tree reduction (`softmax_last` / `rms_norm_last` /
/// `layer_norm_last`). 256 is a safe default block size across sm_75..sm_90
/// (8 warps/block; good occupancy without over-provisioning for the common
/// hidden-dim / attention-Sk widths this crate sees) and is a power of two,
/// which the shared-memory tree reduction requires.
const ROW_BLOCK_THREADS: u32 = 256;

fn cfg_rows(rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.max(1), 1, 1),
        block_dim: (ROW_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: ROW_BLOCK_THREADS * std::mem::size_of::<f32>() as u32,
    }
}

pub unsafe fn launch_binary(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(b)
            .arg(out)
            .arg(&n)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_unary(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(out)
            .arg(&n)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_unary_scalar(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<f32>,
    s: f32,
    out: &mut CudaSlice<f32>,
    n: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(&s)
            .arg(out)
            .arg(&n)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_clamp(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<f32>,
    lo: f32,
    hi: f32,
    out: &mut CudaSlice<f32>,
    n: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(&lo)
            .arg(&hi)
            .arg(out)
            .arg(&n)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_softmax_last(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    rows: i32,
    width: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(out)
            .arg(&rows)
            .arg(&width)
            .launch(cfg_rows(rows as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_rms_norm_last(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    rows: i32,
    width: i32,
    eps: f32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(w)
            .arg(out)
            .arg(&rows)
            .arg(&width)
            .arg(&eps)
            .launch(cfg_rows(rows as u32))?;
    }
    Ok(())
}

/// `weight`/`bias` are `None` for an unaffine LayerNorm (e.g. Wan's AdaLN
/// pre-norms, which apply scale/shift separately via `modulate_scale_shift_last`).
/// When absent, a zero-length placeholder slice is passed and the kernel's
/// `has_affine` flag keeps it unread.
pub unsafe fn launch_layer_norm_last(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    rows: i32,
    width: i32,
    eps: f32,
    has_affine: bool,
) -> Result<()> {
    let has_affine_i = if has_affine { 1i32 } else { 0i32 };
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(weight)
            .arg(bias)
            .arg(out)
            .arg(&rows)
            .arg(&width)
            .arg(&eps)
            .arg(&has_affine_i)
            .launch(cfg_rows(rows as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_modulate_scale_shift_last(
    stream: &CudaStream,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    scale: &CudaSlice<f32>,
    shift: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    batch: i32,
    seq: i32,
    dim: i32,
) -> Result<()> {
    let total = (batch as i64) * (seq as i64) * (dim as i64);
    unsafe {
        stream
            .launch_builder(f)
            .arg(x)
            .arg(scale)
            .arg(shift)
            .arg(out)
            .arg(&batch)
            .arg(&seq)
            .arg(&dim)
            .launch(cfg_n(total.max(1) as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_broadcast_mul_last(
    stream: &CudaStream,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    batch: i32,
    seq: i32,
    dim: i32,
) -> Result<()> {
    let total = (batch as i64) * (seq as i64) * (dim as i64);
    unsafe {
        stream
            .launch_builder(f)
            .arg(x)
            .arg(gate)
            .arg(out)
            .arg(&batch)
            .arg(&seq)
            .arg(&dim)
            .launch(cfg_n(total.max(1) as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_add_bias_last(
    stream: &CudaStream,
    f: &CudaFunction,
    out: &mut CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    n: i32,
    width: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(out)
            .arg(bias)
            .arg(&n)
            .arg(&width)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}


pub unsafe fn launch_f32_to_bf16(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(out)
            .arg(&n)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_bf16_to_f32(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    n: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(out)
            .arg(&n)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_causal_conv3d(
    stream: &CudaStream,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    out: &mut CudaSlice<f32>,
    n: i32, ic: i32, it: i32, ih: i32, iw: i32,
    oc: i32, kt: i32, kh: i32, kw: i32,
    ot: i32, oh: i32, ow: i32,
    st: i32, sh: i32, sw: i32,
) -> Result<()> {
    let total = n * oc * ot * oh * ow;
    // Optional bias: pass null via a zero-length sentinel — callers always provide bias slice.
    let bias_ref = bias.expect("bias required for launch_causal_conv3d");
    unsafe {
        stream
            .launch_builder(f)
            .arg(x)
            .arg(w)
            .arg(bias_ref)
            .arg(out)
            .arg(&n).arg(&ic).arg(&it).arg(&ih).arg(&iw)
            .arg(&oc).arg(&kt).arg(&kh).arg(&kw)
            .arg(&ot).arg(&oh).arg(&ow)
            .arg(&st).arg(&sh).arg(&sw)
            .launch(cfg_n(total.max(1) as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_permute_4d(
    stream: &CudaStream,
    f: &CudaFunction,
    input: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    in_shape: [i32; 4],
    dims: [i32; 4],
) -> Result<()> {
    let total = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];
    let (is0, is1, is2, is3) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (d0, d1, d2, d3) = (dims[0], dims[1], dims[2], dims[3]);
    unsafe {
        stream
            .launch_builder(f)
            .arg(input)
            .arg(out)
            .arg(&is0)
            .arg(&is1)
            .arg(&is2)
            .arg(&is3)
            .arg(&d0)
            .arg(&d1)
            .arg(&d2)
            .arg(&d3)
            .launch(cfg_n(total.max(1) as u32))?;
    }
    Ok(())
}

pub unsafe fn launch_block_copy(
    stream: &CudaStream,
    f: &CudaFunction,
    input: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    outer: i32,
    len: i32,
    in_stride: i32,
    out_stride: i32,
    in_offset: i32,
    out_offset: i32,
) -> Result<()> {
    let total = (outer * len).max(1) as u32;
    unsafe {
        stream
            .launch_builder(f)
            .arg(input)
            .arg(out)
            .arg(&outer)
            .arg(&len)
            .arg(&in_stride)
            .arg(&out_stride)
            .arg(&in_offset)
            .arg(&out_offset)
            .launch(cfg_n(total))?;
    }
    Ok(())
}

pub unsafe fn launch_rope_interleaved(
    stream: &CudaStream,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    rows: i32,
    dim: i32,
) -> Result<()> {
    let total = (rows * (dim / 2)).max(1) as u32;
    unsafe {
        stream
            .launch_builder(f)
            .arg(x)
            .arg(cos)
            .arg(sin)
            .arg(out)
            .arg(&rows)
            .arg(&dim)
            .launch(cfg_n(total))?;
    }
    Ok(())
}

pub unsafe fn launch_rms_norm_channels(
    stream: &CudaStream,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: i32,
    c: i32,
    spatial: i32,
    eps: f32,
) -> Result<()> {
    let total = (n * spatial).max(1) as u32;
    unsafe {
        stream
            .launch_builder(f)
            .arg(x)
            .arg(gamma)
            .arg(out)
            .arg(&n)
            .arg(&c)
            .arg(&spatial)
            .arg(&eps)
            .launch(cfg_n(total))?;
    }
    Ok(())
}

/// In-place BF16 GELU-tanh. `a` is a slice of ushort BF16 bits.
pub unsafe fn launch_gelu_tanh_bf16(
    stream: &CudaStream,
    f: &CudaFunction,
    a: &mut CudaSlice<u16>,
    n: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(a)
            .arg(&n)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}

/// In-place add F32 bias to a BF16 (ushort) buffer along last dim `width`.
pub unsafe fn launch_add_bias_bf16_last(
    stream: &CudaStream,
    f: &CudaFunction,
    out: &mut CudaSlice<u16>,
    bias: &CudaSlice<f32>,
    n: i32,
    width: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(f)
            .arg(out)
            .arg(bias)
            .arg(&n)
            .arg(&width)
            .launch(cfg_n(n as u32))?;
    }
    Ok(())
}

/// Tiled flash attention: O(d) peak memory. Grid = bh*sq blocks, Block = d threads.
/// `smem_bytes` = (2*FA_BK*d + FA_BK*n_warps) * 4; passed by caller.
pub unsafe fn launch_flash_attn_f32(
    stream: &CudaStream,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    bh: i32,
    sq: i32,
    sk: i32,
    d: i32,
    scale: f32,
) -> Result<()> {
    let fa_bk = 32u32;
    let d_u = d as u32;
    let n_warps = d_u / 32;
    let smem_bytes = (2 * fa_bk * d_u + fa_bk * n_warps) * std::mem::size_of::<f32>() as u32;
    let cfg = LaunchConfig {
        grid_dim: ((bh * sq).max(1) as u32, 1, 1),
        block_dim: (d_u, 1, 1),
        shared_mem_bytes: smem_bytes,
    };
    unsafe {
        stream
            .launch_builder(f)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(&bh)
            .arg(&sq)
            .arg(&sk)
            .arg(&d)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Fused unaffine LayerNorm + AdaLN modulate.
/// `scale_w`/`shift_w`: [batch, dim]; `x`/`out`: [batch, seq, dim].
/// One block per (batch*seq) row, 256 threads, dynamic shared memory.
pub unsafe fn launch_layer_norm_adaln_fused(
    stream: &CudaStream,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    scale_w: &CudaSlice<f32>,
    shift_w: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    batch: i32,
    seq: i32,
    dim: i32,
    eps: f32,
) -> Result<()> {
    let rows = (batch * seq).max(1) as u32;
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (ROW_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: ROW_BLOCK_THREADS * std::mem::size_of::<f32>() as u32,
    };
    unsafe {
        stream
            .launch_builder(f)
            .arg(x)
            .arg(scale_w)
            .arg(shift_w)
            .arg(out)
            .arg(&batch)
            .arg(&seq)
            .arg(&dim)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}
