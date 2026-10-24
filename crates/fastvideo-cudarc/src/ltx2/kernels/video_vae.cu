// ==== region: ltx video vae (channels-last bf16 conv decoder) ====
// The LTX-2.5 conv VAE decoder the way the reference runs it (ltx_core
// memory_efficient_decode.py: bf16 weights and activations, channels_last_3d
// workspaces, cuDNN convolutions): activations are bf16 [T, H*W, C] frames
// (NDHWC, batch 1), and every buffer a 3x3x3 conv reads carries one extra
// frame on each side holding the replicate-padded edge frames, so a temporal
// chunk of the conv input is one contiguous slice. The glue between convs is
// fused here with the reference's rounding points: the conv output rounds to
// bf16, + bias rounds, + residual rounds; PixelNorm rounds x*x, the mean, the
// mean + eps, the sqrt and x / rms; SiLU rounds its output. Each value goes
// through the same f32 operations as torch's per-op kernels (explicit _rn
// intrinsics); only the channel sum may add in a different order.

#define LTXV_F_Y 1
#define LTXV_F_BIAS 2
#define LTXV_F_READ_X 4
#define LTXV_F_WRITE_X 8
#define LTXV_F_WRITE_P 16
#define LTXV_F_X_EDGES 32

__device__ __forceinline__ float ltxv_r(float v) {
    return __uint_as_float((unsigned int)fv_bf16_rne(v) << 16);
}

// One warp per pixel (8 per 256-thread block), C a multiple of 128 and at
// most 1024: lane l holds channels 128k + 4l .. +3.
//   v = y (F_Y), bf16(v + bias) (F_BIAS), bf16(x + v) (F_READ_X with F_Y), or
//   x alone (F_READ_X without F_Y); stored to x (F_WRITE_X, in place is fine;
//   F_X_EDGES also writes the clip's first / last frame one frame before /
//   after, x being the interior of a padded buffer an upsampler conv reads);
//   silu(pixelnorm(v)) to the padded buffer p (F_WRITE_P), frame t at t + 1
//   and the clip's first/last frame also at 0 / T + 1 (replicate padding).
// y is the chunk [nt, P, C]; x points at frame 0 of a [T, P, C] interior; the
// chunk covers frames t0 .. t0 + nt of T.
extern "C" __global__ void ltxv_norm_silu(
    const unsigned short* __restrict__ y, const float* __restrict__ bias,
    unsigned short* x, unsigned short* __restrict__ p,
    long P, int C, int t0, int nt, int T, float eps, int flags
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    long pix = (long)blockIdx.x * (long)(blockDim.x >> 5) + warp;
    if (pix >= (long)nt * P) return;
    long tl = pix / P;
    long q = pix - tl * P;
    long t = (long)t0 + tl;
    int per = C >> 7;
    long yoff = pix * (long)C;
    long xoff = (t * P + q) * (long)C;
    float v[32];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < per) {
            int c = (k << 7) + (lane << 2);
            float a[4];
            if (flags & LTXV_F_Y) {
                uint2 u = *(const uint2*)(y + yoff + c);
                a[0] = __uint_as_float(u.x << 16);
                a[1] = __uint_as_float(u.x & 0xFFFF0000u);
                a[2] = __uint_as_float(u.y << 16);
                a[3] = __uint_as_float(u.y & 0xFFFF0000u);
                if (flags & LTXV_F_BIAS) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) a[j] = ltxv_r(__fadd_rn(a[j], bias[c + j]));
                }
            }
            if (flags & LTXV_F_READ_X) {
                uint2 u = *(const uint2*)(x + xoff + c);
                float b[4];
                b[0] = __uint_as_float(u.x << 16);
                b[1] = __uint_as_float(u.x & 0xFFFF0000u);
                b[2] = __uint_as_float(u.y << 16);
                b[3] = __uint_as_float(u.y & 0xFFFF0000u);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    a[j] = (flags & LTXV_F_Y) ? ltxv_r(__fadd_rn(b[j], a[j])) : b[j];
            }
            if (flags & LTXV_F_WRITE_X) {
                uint2 u;
                u.x = (unsigned int)fv_bf16_rne(a[0]) | ((unsigned int)fv_bf16_rne(a[1]) << 16);
                u.y = (unsigned int)fv_bf16_rne(a[2]) | ((unsigned int)fv_bf16_rne(a[3]) << 16);
                *(uint2*)(x + xoff + c) = u;
                if ((flags & LTXV_F_X_EDGES) && t == 0) *(uint2*)(x + xoff - P * C + c) = u;
                if ((flags & LTXV_F_X_EDGES) && t == T - 1) *(uint2*)(x + xoff + P * C + c) = u;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[4 * k + j] = a[j];
                ss += ltxv_r(__fmul_rn(a[j], a[j]));
            }
        }
    }
    if (!(flags & LTXV_F_WRITE_P)) return;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    float mean = ltxv_r(__fmul_rn(ss, 1.0f / (float)C));
    float rms = ltxv_r(sqrtf(ltxv_r(__fadd_rn(mean, eps))));
    long pbase = ((t + 1) * P + q) * (long)C;
    long pfirst = q * (long)C;
    long plast = ((long)(T + 1) * P + q) * (long)C;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < per) {
            int c = (k << 7) + (lane << 2);
            unsigned short o[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float n = ltxv_r(__fdiv_rn(v[4 * k + j], rms));
                o[j] = fv_bf16_rne(__fdiv_rn(n, __fadd_rn(1.0f, expf(-n))));
            }
            uint2 u;
            u.x = (unsigned int)o[0] | ((unsigned int)o[1] << 16);
            u.y = (unsigned int)o[2] | ((unsigned int)o[3] << 16);
            *(uint2*)(p + pbase + c) = u;
            if (t == 0) *(uint2*)(p + pfirst + c) = u;
            if (t == T - 1) *(uint2*)(p + plast + c) = u;
        }
    }
}

// Latents f32 [C, F, P] (one tile, contiguous) to the padded bf16 [F + 2, P, C]
// the first conv reads: bf16(bf16(bf16(z) * std) + mean) per channel (the
// reference casts the latent to bf16, then un_normalize's two ops), frame 0
// and F + 1 replicate the first / last frame. std and mean hold bf16 values.
extern "C" __global__ void ltxv_latent_in(
    const float* __restrict__ z, const float* __restrict__ std_, const float* __restrict__ mean,
    unsigned short* __restrict__ out, long P, int C, int F
) {
    long i = IDX();
    long n = (long)(F + 2) * P * C;
    if (i >= n) return;
    int c = (int)(i % C);
    long r = i / C;
    long q = r % P;
    long tp = r / P;
    long t = tp - 1;
    if (t < 0) t = 0;
    if (t > F - 1) t = F - 1;
    float v = ltxv_r(z[((long)c * F + t) * P + q]);
    v = ltxv_r(__fmul_rn(v, std_[c]));
    out[i] = fv_bf16_rne(__fadd_rn(v, mean[c]));
}

// Depth-to-space upsampler output: conv chunk y [nt, H, W, C*st*sh*sw] (input
// frames t0 .. t0 + nt) plus bias, rounded, rearranged
// "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)" into out, frame 0 of a
// [T', sh*H, sw*W, C] interior; drop = 1 drops the first output frame (the
// reference's x[:, :, 1:] after a temporal x2).
extern "C" __global__ void ltxv_d2s_bias(
    const unsigned short* __restrict__ y, const float* __restrict__ bias,
    unsigned short* __restrict__ out,
    int H, int W, int C, int st, int sh, int sw, int t0, int nt, int drop
) {
    long i = IDX();
    long HO = (long)sh * H, WO = (long)sw * W;
    long n = (long)nt * st * HO * WO * C;
    if (i >= n) return;
    int c = (int)(i % C);
    long r = i / C;
    long wo = r % WO;
    r /= WO;
    long ho = r % HO;
    r /= HO;
    int s = (int)(r % st);
    long tl = r / st;
    long of = (long)st * (t0 + tl) + s - drop;
    if (of < 0) return;
    long h = ho / sh, w = wo / sw;
    int j = (int)(ho - h * sh), k = (int)(wo - w * sw);
    int prod = st * sh * sw;
    long cy = (long)c * prod + (long)s * sh * sw + (long)j * sw + k;
    long cyn = (long)C * prod;
    float v = __fadd_rn(fv_bf16_to_f32(y[((tl * H + h) * W + w) * cyn + cy]), bias[cy]);
    out[((of * HO + ho) * WO + wo) * C + c] = fv_bf16_rne(v);
}

// conv_out chunk y [nt, H, W, 3*p*p] plus bias, rounded, unpatchified
// ("b (c p r q) f h w -> b c (f p) (h q) (w r)", patch_size_t = 1) into the
// tile's [T, 3, p*H, p*W] frames starting at frame t0:
// out[f, c, p*h + b, p*w + a] = y[f, h, w, c*p*p + a*p + b].
extern "C" __global__ void ltxv_out_unpatch(
    const unsigned short* __restrict__ y, const float* __restrict__ bias,
    unsigned short* __restrict__ out, int H, int W, int CO, int pz, int t0, int nt
) {
    long i = IDX();
    long HO = (long)pz * H, WO = (long)pz * W;
    long n = (long)nt * CO * HO * WO;
    if (i >= n) return;
    long wo = i % WO;
    long r = i / WO;
    long ho = r % HO;
    r /= HO;
    int c = (int)(r % CO);
    long tl = r / CO;
    long h = ho / pz, w = wo / pz;
    int b = (int)(ho - h * pz), a = (int)(wo - w * pz);
    long cyn = (long)CO * pz * pz;
    long cy = (long)c * pz * pz + (long)a * pz + b;
    float v = __fadd_rn(fv_bf16_to_f32(y[((tl * H + h) * W + w) * cyn + cy]), bias[cy]);
    out[((((long)t0 + tl) * CO + c) * HO + ho) * WO + wo] = fv_bf16_rne(v);
}
// ==== endregion: ltx video vae ====

