// ==== region: adapter fuse on the device (appended) ====
// One weight of a low-rank adapter merge, the host's arithmetic element for
// element (fastvideo_models::h3::lora::add_low_rank / add_diff, then
// half::bf16::from_f32): every product and sum rounded on its own
// (__fmul_rn / __fadd_rn, so no FMA contraction), the rank terms in
// ascending order, a zero `mult * B[o, r]` skipped as the host skips it, the
// `.diff` added last, and the round to bf16 is fv_bf16_rne (the host's RNE,
// NaN kept quiet). base_mode: 0 = bf16 base bits, 1 = f32 base, 2 = f32
// `.set_weight` replacement (w = base32 * rscale). `out` may be a row block of
// a stacked weight.
extern "C" __global__ void lora_fuse_bf16(
    const unsigned short* base16, const float* base32, int base_mode, float rscale,
    const float* a, const float* b, int rank, float mult, int has_pair,
    const float* diff, float dscale, int has_diff,
    unsigned short* out, long rows, long cols
) {
    long i = IDX();
    if (i >= rows * cols) return;
    long o = i / cols;
    long c = i - o * cols;
    float w;
    if (base_mode == 0) {
        w = __uint_as_float(((unsigned int)base16[i]) << 16);
    } else if (base_mode == 1) {
        w = base32[i];
    } else {
        w = __fmul_rn(base32[i], rscale);
    }
    if (has_pair) {
        for (int r = 0; r < rank; ++r) {
            float s = __fmul_rn(mult, b[o * (long)rank + r]);
            if (s == 0.0f) continue;
            w = __fadd_rn(w, __fmul_rn(s, a[(long)r * cols + c]));
        }
    }
    if (has_diff) {
        w = __fadd_rn(w, __fmul_rn(dscale, diff[i]));
    }
    out[i] = fv_bf16_rne(w);
}
// ==== endregion: adapter fuse on the device ====
