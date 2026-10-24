// ==== region: moe ====
#ifndef IDX
#define IDX() ((long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x)
#endif

// Real-interleaved RoPE (Diffusers apply_rotary_emb use_real=True, unbind -2):
// pair (x0, x1) at even/odd channels with cos/sin at the even table index.
// x: [B, H, S, D] BHSD; cs/sn: [S, D].
extern "C" __global__ void rope_real(
    const float* x, const float* cs, const float* sn, float* out, long s, long d, long n
) {
    long i = IDX();
    if (i >= n) return;
    long j = i % d;
    if ((j & 1) && j + 1 > d) {
        out[i] = x[i];
        return;
    }
    long p = (i / d) % s;
    long pair = j & ~1L;
    if (pair + 1 >= d) {
        out[i] = x[i];
        return;
    }
    float x0 = x[i - (j - pair)];
    float x1 = x[i - (j - pair) + 1];
    float c = cs[p * d + pair];
    float si = sn[p * d + pair];
    out[i] = (j & 1) ? (x0 * si + x1 * c) : (x0 * c - x1 * si);
}

extern "C" __global__ void sigmoid_f(const float* a, float* out, long n) {
    long i = IDX();
    if (i < n) out[i] = 1.0f / (1.0f + expf(-a[i]));
}

extern "C" __global__ void copy_f(const float* a, float* out, long n) {
    long i = IDX();
    if (i < n) out[i] = a[i];
}

// One block per row. `width` is the expert count (small); thread 0 walks it.
// Writes descending top-k into idx/val. `norm != 0` L1-normalizes the k values.
extern "C" __global__ void topk_last(
    const float* scores, unsigned int* idx, float* val, int rows, int width, int k, int norm
) {
    int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    const float* src = scores + (long)row * width;
    unsigned int* oi = idx + (long)row * k;
    float* ov = val + (long)row * k;
    for (int t = 0; t < k; t++) {
        oi[t] = 0;
        ov[t] = -1e30f;
    }
    for (int e = 0; e < width; e++) {
        float v = src[e];
        int slot = k;
        for (int t = 0; t < k; t++) {
            if (v > ov[t]) {
                slot = t;
                break;
            }
        }
        if (slot == k) continue;
        for (int t = k - 1; t > slot; t--) {
            ov[t] = ov[t - 1];
            oi[t] = oi[t - 1];
        }
        ov[slot] = v;
        oi[slot] = (unsigned int)e;
    }
    if (norm) {
        float z = 0.0f;
        for (int t = 0; t < k; t++) z += ov[t];
        z = fmaxf(z, 1e-20f);
        for (int t = 0; t < k; t++) ov[t] /= z;
    }
}

// out[idx[r], j] += src[r, j] * weight[r]. Caller copies `base` into `out` first.
// Rows of one expert are unique, so no atomics.
extern "C" __global__ void scatter_add_rows(
    const float* src, const unsigned int* idx, const float* weight, float* out, long n, long d
) {
    long i = IDX();
    if (i >= n * d) return;
    long r = i / d;
    out[(long)idx[r] * d + (i % d)] += src[i] * weight[r];
}

// LingBot / DeepSeek-V3 group-limited router (`fastvideo_models::lingbot::routing`).
// One thread per token row of `logits` [rows, e] (f32). Selection uses
// score + bias; the n_group groups are ranked by the sum of their two best
// biased scores and only the topk_group best stay eligible; the k chosen
// experts get their bias-free scores, L1-normalized (+1e-20) when `norm`,
// times `scale`, rounded to bf16 when `round16`. Ties keep the lower index
// (repeated strict argmax), like the host's stable sort. e <= 256, n_group <= 64.
__device__ __forceinline__ float fv_moe_round_bf16(float x) {
    unsigned int b = __float_as_uint(x);
    if ((b & 0x7f800000u) == 0x7f800000u) return x;
    b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(b);
}

extern "C" __global__ void moe_group_topk(
    const float* logits, const float* bias, unsigned int* idx, float* val,
    int rows, int e, int k, int n_group, int topk_group, int softmax, int norm,
    float scale, int round16
) {
    int row = (int)IDX();
    if (row >= rows || e > 256 || n_group > 64) return;
    const float* src = logits + (long)row * e;
    float score[256];
    float choice[256];
    if (softmax) {
        float m = -3.402823466e38f;
        for (int j = 0; j < e; j++) m = fmaxf(m, src[j]);
        float z = 0.0f;
        for (int j = 0; j < e; j++) {
            score[j] = expf(src[j] - m);
            z += score[j];
        }
        for (int j = 0; j < e; j++) score[j] /= z;
    } else {
        for (int j = 0; j < e; j++) score[j] = 1.0f / (1.0f + expf(-src[j]));
    }
    for (int j = 0; j < e; j++) choice[j] = score[j] + bias[j];
    if (n_group > 1) {
        int per = e / n_group;
        float gscore[64];
        for (int g = 0; g < n_group; g++) {
            float m1 = -3.402823466e38f, m2 = -3.402823466e38f;
            int i1 = -1;
            for (int j = 0; j < per; j++) {
                float v = choice[g * per + j];
                if (i1 < 0 || v > m1) { m1 = v; i1 = j; }
            }
            for (int j = 0; j < per; j++) {
                float v = choice[g * per + j];
                if (j != i1 && v > m2) m2 = v;
            }
            gscore[g] = per >= 2 ? m1 + m2 : m1;
        }
        unsigned long long keep = 0ull;
        for (int t = 0; t < topk_group; t++) {
            int best = -1;
            float bv = 0.0f;
            for (int g = 0; g < n_group; g++) {
                if ((keep >> g) & 1ull) continue;
                if (best < 0 || gscore[g] > bv) { best = g; bv = gscore[g]; }
            }
            if (best >= 0) keep |= 1ull << best;
        }
        for (int g = 0; g < n_group; g++) {
            if ((keep >> g) & 1ull) continue;
            for (int j = 0; j < per; j++) choice[g * per + j] = __int_as_float(0xff800000);
        }
    }
    unsigned int* oi = idx + (long)row * k;
    float* ov = val + (long)row * k;
    float z = 0.0f;
    for (int t = 0; t < k; t++) {
        int best = -1;
        float bv = 0.0f;
        for (int j = 0; j < e; j++) {
            float v = choice[j];
            if (v == __int_as_float(0xff800000) && best >= 0) continue;
            if (best < 0 || v > bv) { best = j; bv = v; }
        }
        oi[t] = (unsigned int)best;
        ov[t] = score[best];
        z += score[best];
        choice[best] = __int_as_float(0xff800000);
    }
    for (int t = 0; t < k; t++) {
        float w = ov[t];
        if (k > 1 && norm) w = w / (z + 1e-20f);
        w *= scale;
        ov[t] = round16 ? fv_moe_round_bf16(w) : w;
    }
}

// out[t, j] = sum_s w[t*k+s] * ys[pos[t*k+s], j]: the MoE combine after the
// expert-sorted GEMMs (gather form, no atomics). One thread per output.
extern "C" __global__ void moe_combine(
    const float* ys, const unsigned int* pos, const float* w, float* out,
    long n, long k, long d
) {
    long i = IDX();
    if (i >= n * d) return;
    long t = i / d, j = i % d;
    float acc = 0.0f;
    for (long s = 0; s < k; s++) {
        long r = t * k + s;
        acc += w[r] * ys[(long)pos[r] * d + j];
    }
    out[i] = acc;
}
// ==== end region: moe ====

