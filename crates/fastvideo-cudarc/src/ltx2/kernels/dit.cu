// ==== region: ltx2 (FBCache distance, midpoint prune gather/scatter) ====
// Appended as one block so other sections can change without conflicts.

// sum |a - b| and sum |b| over n values, one pair per block at
// out[2 * blockIdx.x ..] (no atomics; the caller adds the pairs in block
// order, so the FBCache distance is the same bits in every run); 256 threads
// per block, grid-stride.
extern "C" __global__ void ltx_abs_diff_sums(
    const float* a, const float* b, double* out, long n
) {
    __shared__ double s_num[256];
    __shared__ double s_den[256];
    double num = 0.0, den = 0.0;
    long stride = (long)gridDim.x * (long)blockDim.x;
    for (long i = IDX(); i < n; i += stride) {
        double x = (double)a[i], y = (double)b[i];
        num += fabs(x - y);
        den += fabs(y);
    }
    s_num[threadIdx.x] = num;
    s_den[threadIdx.x] = den;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_num[threadIdx.x] += s_num[threadIdx.x + s];
            s_den[threadIdx.x] += s_den[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[2 * blockIdx.x] = s_num[0];
        out[2 * blockIdx.x + 1] = s_den[0];
    }
}

// out[r] = sum_c x[r, c]^2 for x [rows, d]; one 256-thread block per row.
extern "C" __global__ void ltx_row_sumsq(
    const float* x, float* out, long rows, long d
) {
    __shared__ float s[256];
    long r = blockIdx.x;
    if (r >= rows) return;
    float acc = 0.0f;
    for (long c = threadIdx.x; c < d; c += blockDim.x) {
        float v = x[r * d + c];
        acc += v * v;
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[r] = s[0];
}

// out[idx[j], :] = src[j, :] for src [k, d]; out already holds the base rows.
extern "C" __global__ void ltx_index_copy_rows(
    const float* src, const unsigned int* idx, float* out, long n, long d
) {
    long i = IDX();
    if (i >= n) return;
    out[(long)idx[i / d] * d + i % d] = src[i];
}

// rows[h * k + j] = h * tokens + idx[j]: a head-major rotary table's rows for
// the kept tokens.
extern "C" __global__ void ltx_rope_rows(
    const unsigned int* idx, unsigned int* rows, long k, long heads, long tokens
) {
    long i = IDX();
    if (i >= k * heads) return;
    rows[i] = (unsigned int)((i / k) * tokens + (long)idx[i % k]);
}

// A split rotary table in rope_half's layout — cos/sin [heads * tokens,
// 2 * half], head-major, each pair's value in both halves — gathered from its
// factored form (fastvideo_models::ltx2::SplitRopeLut): slot m = h * half + j
// is identity below `pad`, else frequency (m - pad) / axes on axis
// (m - pad) % axes, whose value row is index[t * axes + axis]. Pure copies,
// so the table carries the host's bits.
extern "C" __global__ void ltx_split_rope(
    const unsigned int* index, const float* lut_cos, const float* lut_sin,
    float* cos_out, float* sin_out,
    long heads, long tokens, long half, long axes, long pad, long n
) {
    long i = IDX();
    long d = 2 * half;
    if (i >= heads * tokens * d) return;
    long row = i / d, c = i % d;
    long j = c < half ? c : c - half;
    long h = row / tokens, t = row % tokens;
    long m = h * half + j;
    if (m < pad) {
        cos_out[i] = 1.0f;
        sin_out[i] = 0.0f;
        return;
    }
    long k = (m - pad) / axes, a = (m - pad) % axes;
    long u = (long)index[t * axes + a];
    cos_out[i] = lut_cos[u * n + k];
    sin_out[i] = lut_sin[u * n + k];
}
// ==== endregion: ltx2 ====

