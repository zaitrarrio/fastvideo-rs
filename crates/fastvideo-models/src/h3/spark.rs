//! Spark stage-2 latent bridge from NVlabs/Sana `sol-engine`
//! `models/minimax_h3/Sol-H3-Spark`.
//!
//! Stage 1 emits a normalized H3 latent. The published bridge is
//! `D(model(N(h3)))` (`runtime/stage2_ops/h3_upscale.py`, author node around
//! an already-normalized Comfy latent) then
//! `H3ToLTXAdapter.convert` (`runtime/stage2_ops/h3_ltx_adapter`) and a crop
//! of the time axis to 16 (`stage2.py` `latent[:, :, :16]`).
//!
//! Official canvas: H3 `(1, 24, 37, 24, 42)` → upscaled `(1, 24, 37, 48, 84)`
//! → adapter `(1, 128, 17, 24, 42)` → refiner `(1, 128, 16, 24, 42)`, with
//! `pixel_frames=124`, `pixel_height=768`, `pixel_width=1344`.
//!
//! Stage-1 stays BF16 with FastH3_VSA_DataFree at strength 1.0. Upstream
//! W8A8 FP8 after the LoRA merge is not enabled.

use super::lora::SolH3AdapterSpec;

/// Stage-2 generic prompt from `Sol-H3-Spark/runtime/prompt_cache.py`.
pub const FIXED_PROMPT: &str =
    "4K, refined, high quality, cinematic detail, clean textures, natural motion.";

/// Volume dest from Phase 0, then local aliases, then dense-datafree fallback.
pub const SPARK_LORA_PATHS: &[&str] = &[
    "FastH3-4-step-Preview-v1-LoRA/vsa-datafree/adapter_model.safetensors",
    "FastH3-4-step-Preview-v1-VSA-DataFree/adapter_model.safetensors",
    "vsa-datafree/adapter_model.safetensors",
    "adapter/vsa-datafree/adapter_model.safetensors",
    "adapter/dense-datafree/adapter_model.safetensors",
    "dense-datafree/adapter_model.safetensors",
    "FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors",
];

/// Spark Stage-1 adapter: FastH3_VSA_DataFree preferred, strength 1.0.
pub fn spark_adapter_spec() -> SolH3AdapterSpec {
    SolH3AdapterSpec {
        alpha: 64,
        scale: 1.0,
        relative_paths: SPARK_LORA_PATHS,
    }
}

/// Pinned author-node statistics (`LATENTS_MEAN` / `LATENTS_STD` in
/// `minimax_h3_latent_upscaler_3d.py` at `d7c01b9011f2e8439493f6c02c29995a27df276f`).
pub const LATENTS_MEAN: [f64; 24] = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
];

pub const LATENTS_STD: [f64; 24] = [
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
];

pub const H3_INPUT: [usize; 4] = [24, 37, 24, 42];
pub const H3_UPSCALED: [usize; 4] = [24, 37, 48, 84];
pub const ADAPTER_OUTPUT: [usize; 4] = [128, 17, 24, 42];
pub const REFINER_INPUT: [usize; 4] = [128, 16, 24, 42];

pub const PIXEL_FRAMES: usize = 124;
pub const PIXEL_HEIGHT: usize = 768;
pub const PIXEL_WIDTH: usize = 1344;

pub const UPSCALER_FILE: &str = "minimax_h3_latent_upscaler_3d_bf16.safetensors";
pub const ADAPTER_WEIGHTS: &str = "model.safetensors";
pub const ADAPTER_CONFIG: &str = "config.json";

const H3_TEMPORAL_COMPRESSION: usize = 4;
const H3_CLIP_LENGTH: usize = 17;
const H3_TOKEN_DROP: usize = 3;
const LTX_TEMPORAL_COMPRESSION: usize = 8;

/// `N` then, after the network, `D`. Layout is `[B, C, T, H, W]`. The incoming
/// latent is already normalized; the author node still applies both transforms.
pub fn author_node_input(normalized: &[f32], shape: [usize; 5]) -> Result<Vec<f32>, String> {
    affine(normalized, shape, true)
}

pub fn author_node_output(network: &[f32], shape: [usize; 5]) -> Result<Vec<f32>, String> {
    affine(network, shape, false)
}

fn affine(samples: &[f32], shape: [usize; 5], into_network: bool) -> Result<Vec<f32>, String> {
    let [b, c, t, h, w] = shape;
    if c != LATENTS_MEAN.len() || samples.len() != b * c * t * h * w {
        return Err(format!(
            "h3 norm: {} values for shape {shape:?}",
            samples.len()
        ));
    }
    let plane = t * h * w;
    let mut out = vec![0.0; samples.len()];
    for batch in 0..b {
        for ch in 0..c {
            let mean = LATENTS_MEAN[ch] as f32;
            let std = LATENTS_STD[ch] as f32;
            if std == 0.0 {
                return Err("h3 norm: zero std".into());
            }
            let start = (batch * c + ch) * plane;
            for i in 0..plane {
                let v = samples[start + i];
                out[start + i] = if into_network {
                    (v - mean) / std
                } else {
                    v * std + mean
                };
            }
        }
    }
    Ok(out)
}

/// `h3_temporal_positions` from `h3_ltx_adapter/geometry.py`.
pub fn h3_temporal_positions(num_pixel_frames: usize) -> Result<Vec<f32>, String> {
    if num_pixel_frames == 0 {
        return Err("num_pixel_frames must be positive".into());
    }
    let tokens_per_chunk = (H3_CLIP_LENGTH + H3_TEMPORAL_COMPRESSION - 1) / H3_TEMPORAL_COMPRESSION;
    let num_chunks = (num_pixel_frames + H3_CLIP_LENGTH - 1) / H3_CLIP_LENGTH;
    let mut positions = Vec::with_capacity(num_chunks * tokens_per_chunk);
    for chunk in 0..num_chunks {
        for token in 0..tokens_per_chunk {
            let at = chunk * H3_CLIP_LENGTH + token * H3_TEMPORAL_COMPRESSION;
            positions.push(at.min(num_pixel_frames - 1) as f32);
        }
    }
    if H3_TOKEN_DROP > 0 {
        positions.truncate(positions.len().saturating_sub(H3_TOKEN_DROP));
    }
    Ok(positions)
}

pub fn padded_ltx_pixel_frames(num_pixel_frames: usize) -> Result<usize, String> {
    if num_pixel_frames == 0 {
        return Err("num_pixel_frames must be positive".into());
    }
    Ok(
        ((num_pixel_frames - 1 + LTX_TEMPORAL_COMPRESSION - 1) / LTX_TEMPORAL_COMPRESSION)
            * LTX_TEMPORAL_COMPRESSION
            + 1,
    )
}

pub fn ltx_temporal_positions(num_pixel_frames: usize) -> Result<Vec<f32>, String> {
    if num_pixel_frames == 0 {
        return Err("num_pixel_frames must be positive".into());
    }
    let latent_frames = (num_pixel_frames - 1) / LTX_TEMPORAL_COMPRESSION + 1;
    Ok((0..latent_frames)
        .map(|index| (index * LTX_TEMPORAL_COMPRESSION).min(num_pixel_frames - 1) as f32)
        .collect())
}

fn searchsorted_left(source: &[f32], target: f32) -> usize {
    let at = source.partition_point(|value| *value < target);
    at.min(source.len().saturating_sub(1))
}

/// Channel-first `[B, C, T, H, W]`.
pub fn temporal_resample(
    latent: &[f32],
    shape: [usize; 5],
    source_positions: &[f32],
    target_positions: &[f32],
) -> Result<Vec<f32>, String> {
    let [b, c, t, h, w] = shape;
    if latent.len() != b * c * t * h * w {
        return Err("temporal resample: length does not match shape".into());
    }
    if source_positions.len() != t || source_positions.is_empty() {
        return Err("temporal resample: source positions must match T".into());
    }
    let spatial = h * w;
    let mut out = vec![0.0; b * c * target_positions.len() * spatial];
    for (ti, &target) in target_positions.iter().enumerate() {
        let right = searchsorted_left(source_positions, target);
        let left = right.saturating_sub(1);
        let denom = source_positions[right] - source_positions[left];
        let mut weight = if denom > 0.0 {
            (target - source_positions[left]) / denom
        } else {
            0.0
        };
        weight = weight.clamp(0.0, 1.0);
        for batch in 0..b {
            for ch in 0..c {
                let base = ((batch * c + ch) * t) * spatial;
                let dst = ((batch * c + ch) * target_positions.len() + ti) * spatial;
                let l = &latent[base + left * spatial..base + (left + 1) * spatial];
                let r = &latent[base + right * spatial..base + (right + 1) * spatial];
                for i in 0..spatial {
                    out[dst + i] = l[i] + (r[i] - l[i]) * weight;
                }
            }
        }
    }
    Ok(out)
}

/// PyTorch `pixel_unshuffle` for NCHW, `downscale_factor = 2`.
pub fn pixel_unshuffle2(
    nchw: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>, String> {
    if h % 2 != 0 || w % 2 != 0 || nchw.len() != n * c * h * w {
        return Err("pixel_unshuffle expects even H and W".into());
    }
    let (oh, ow) = (h / 2, w / 2);
    let oc = c * 4;
    let mut out = vec![0.0; n * oc * oh * ow];
    for bi in 0..n {
        for ch in 0..c {
            for i in 0..2 {
                for j in 0..2 {
                    let oc_i = ch * 4 + i * 2 + j;
                    for y in 0..oh {
                        for x in 0..ow {
                            let src = (((bi * c + ch) * h + y * 2 + i) * w) + x * 2 + j;
                            let dst = (((bi * oc + oc_i) * oh + y) * ow) + x;
                            out[dst] = nchw[src];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `latent[:, :, :16]` after the adapter. The native output has 17 frames.
pub fn crop_refiner_time(
    latent: &[f32],
    shape: [usize; 5],
) -> Result<(Vec<f32>, [usize; 5]), String> {
    let [b, c, t, h, w] = shape;
    if t < 16 || latent.len() != b * c * t * h * w {
        return Err(format!("refiner crop expected T>=16, got {shape:?}"));
    }
    let spatial = h * w;
    let mut out = Vec::with_capacity(b * c * 16 * spatial);
    for batch in 0..b {
        for ch in 0..c {
            let start = ((batch * c + ch) * t) * spatial;
            out.extend_from_slice(&latent[start..start + 16 * spatial]);
        }
    }
    Ok((out, [b, c, 16, h, w]))
}

/// `temporal_nearest_pack` from `h3_ltx_adapter/geometry.py`.
/// Output layout is `[B, slots * C, T_target, H, W]`.
pub fn temporal_nearest_pack(
    latent: &[f32],
    shape: [usize; 5],
    source_positions: &[f32],
    target_positions: &[f32],
    slots: usize,
) -> Result<Vec<f32>, String> {
    let [b, c, t, h, w] = shape;
    if latent.len() != b * c * t * h * w || source_positions.len() != t {
        return Err("temporal pack: length does not match shape".into());
    }
    if slots == 0 || target_positions.is_empty() {
        return Err("temporal pack: slots and targets must be non-empty".into());
    }
    let tgt = target_positions.len();
    let mut assignment = vec![0usize; t];
    for (i, &source) in source_positions.iter().enumerate() {
        let mut best = 0;
        let mut best_d = f32::INFINITY;
        for (j, &target) in target_positions.iter().enumerate() {
            let d = (source - target).abs();
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        assignment[i] = best;
    }
    let mut counts = vec![0usize; tgt];
    for &at in &assignment {
        counts[at] += 1;
    }
    let required = counts.iter().copied().max().unwrap_or(0);
    if required > slots {
        return Err(format!(
            "temporal packing needs {required} slots, checkpoint provides {slots}"
        ));
    }
    let mut indices = vec![0usize; tgt * slots];
    let mut occupied = vec![false; tgt * slots];
    let mut filled = vec![0usize; tgt];
    for (src_i, &tgt_i) in assignment.iter().enumerate() {
        let slot = filled[tgt_i];
        indices[tgt_i * slots + slot] = src_i;
        occupied[tgt_i * slots + slot] = true;
        filled[tgt_i] += 1;
    }
    let spatial = h * w;
    let out_c = slots * c;
    let mut out = vec![0.0; b * out_c * tgt * spatial];
    for batch in 0..b {
        for slot in 0..slots {
            for ch in 0..c {
                let oc = slot * c + ch;
                for ti in 0..tgt {
                    if !occupied[ti * slots + slot] {
                        continue;
                    }
                    let src_t = indices[ti * slots + slot];
                    let src = ((batch * c + ch) * t + src_t) * spatial;
                    let dst = ((batch * out_c + oc) * tgt + ti) * spatial;
                    out[dst..dst + spatial].copy_from_slice(&latent[src..src + spatial]);
                }
            }
        }
    }
    Ok(out)
}

/// `align_h3_to_ltx`: linear resample, nearest pack, channel concat, 2× unshuffle.
pub fn align_h3_to_ltx(
    latent: &[f32],
    shape: [usize; 5],
    pixel_frames: usize,
    target_height: usize,
    target_width: usize,
    slots: usize,
) -> Result<(Vec<f32>, [usize; 5]), String> {
    let [b, c, t, h, w] = shape;
    if latent.len() != b * c * t * h * w {
        return Err("align: length does not match shape".into());
    }
    let source = h3_temporal_positions(pixel_frames)?;
    if source.len() != t {
        return Err(format!(
            "H3 latent T={t} does not match pixel_frames={pixel_frames}; expected T={}",
            source.len()
        ));
    }
    let target = ltx_temporal_positions(padded_ltx_pixel_frames(pixel_frames)?)?;
    let linear = temporal_resample(latent, shape, &source, &target)?;
    let packed = temporal_nearest_pack(latent, shape, &source, &target, slots)?;
    let tgt = target.len();
    let pack_c = slots * c;
    let cat_c = c + pack_c;
    let plane = tgt * h * w;
    let mut cat = vec![0.0; b * cat_c * plane];
    for batch in 0..b {
        let dst = batch * cat_c * plane;
        let lin = &linear[batch * c * plane..(batch + 1) * c * plane];
        cat[dst..dst + c * plane].copy_from_slice(lin);
        let pack = &packed[batch * pack_c * plane..(batch + 1) * pack_c * plane];
        cat[dst + c * plane..dst + cat_c * plane].copy_from_slice(pack);
    }
    let (scale_h, rem_h) = (h / target_height, h % target_height);
    let (scale_w, rem_w) = (w / target_width, w % target_width);
    if rem_h != 0 || rem_w != 0 || scale_h != 2 || scale_w != 2 {
        return Err(format!(
            "frozen adapter requires 2x pixel-unshuffle: source=({h}, {w}), target=({target_height}, {target_width})"
        ));
    }
    let n = b * tgt;
    let spatial = h * w;
    let mut nchw = vec![0.0; n * cat_c * spatial];
    for batch in 0..b {
        for ti in 0..tgt {
            for ch in 0..cat_c {
                let src = ((batch * cat_c + ch) * tgt + ti) * spatial;
                let dst = ((batch * tgt + ti) * cat_c + ch) * spatial;
                nchw[dst..dst + spatial].copy_from_slice(&cat[src..src + spatial]);
            }
        }
    }
    let unshuffled = pixel_unshuffle2(&nchw, n, cat_c, h, w)?;
    let oc = cat_c * 4;
    let (oh, ow) = (target_height, target_width);
    let plane_out = oh * ow;
    let mut out = vec![0.0; b * oc * tgt * plane_out];
    for batch in 0..b {
        for ti in 0..tgt {
            for ch in 0..oc {
                let src = ((batch * tgt + ti) * oc + ch) * plane_out;
                let dst = ((batch * oc + ch) * tgt + ti) * plane_out;
                out[dst..dst + plane_out].copy_from_slice(&unshuffled[src..src + plane_out]);
            }
        }
    }
    Ok((out, [b, oc, tgt, oh, ow]))
}

/// `F.interpolate(..., mode="trilinear", align_corners=False)`.
pub fn trilinear_ncdhw(
    src: &[f32],
    shape: [usize; 5],
    out_size: [usize; 3],
) -> Result<Vec<f32>, String> {
    let [b, c, t, h, w] = shape;
    let [ot, oh, ow] = out_size;
    if src.len() != b * c * t * h * w || ot == 0 || oh == 0 || ow == 0 || t == 0 || h == 0 || w == 0
    {
        return Err(format!(
            "trilinear: {shape:?} → {out_size:?} with {} values",
            src.len()
        ));
    }
    let st = t as f32 / ot as f32;
    let sh = h as f32 / oh as f32;
    let sw = w as f32 / ow as f32;
    let mut out = vec![0.0; b * c * ot * oh * ow];
    for batch in 0..b {
        for ch in 0..c {
            for zi in 0..ot {
                let (z0, z1, wz) = axis_sample(st * (zi as f32 + 0.5) - 0.5, t);
                for yi in 0..oh {
                    let (y0, y1, wy) = axis_sample(sh * (yi as f32 + 0.5) - 0.5, h);
                    for xi in 0..ow {
                        let (x0, x1, wx) = axis_sample(sw * (xi as f32 + 0.5) - 0.5, w);
                        let at = |z, y, x| src[(((batch * c + ch) * t + z) * h + y) * w + x];
                        let c00 = at(z0, y0, x0) * (1.0 - wx) + at(z0, y0, x1) * wx;
                        let c01 = at(z0, y1, x0) * (1.0 - wx) + at(z0, y1, x1) * wx;
                        let c10 = at(z1, y0, x0) * (1.0 - wx) + at(z1, y0, x1) * wx;
                        let c11 = at(z1, y1, x0) * (1.0 - wx) + at(z1, y1, x1) * wx;
                        let c0 = c00 * (1.0 - wy) + c01 * wy;
                        let c1 = c10 * (1.0 - wy) + c11 * wy;
                        let dst = (((batch * c + ch) * ot + zi) * oh + yi) * ow + xi;
                        out[dst] = c0 * (1.0 - wz) + c1 * wz;
                    }
                }
            }
        }
    }
    Ok(out)
}

fn axis_sample(coord: f32, size: usize) -> (usize, usize, f32) {
    let low = coord.floor();
    let low_i = low as isize;
    let high_i = low_i + 1;
    let l = low_i.clamp(0, size as isize - 1) as usize;
    let h = high_i.clamp(0, size as isize - 1) as usize;
    (l, h, coord - low)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_canvas_frame_counts() {
        let h3 = h3_temporal_positions(PIXEL_FRAMES).unwrap();
        assert_eq!(h3.len(), H3_INPUT[1]);
        let padded = padded_ltx_pixel_frames(PIXEL_FRAMES).unwrap();
        assert_eq!(padded, 129);
        let ltx = ltx_temporal_positions(padded).unwrap();
        assert_eq!(ltx.len(), ADAPTER_OUTPUT[1]);
        assert_eq!(REFINER_INPUT[1], 16);
        assert_eq!(
            (PIXEL_HEIGHT / 16, PIXEL_WIDTH / 16),
            (H3_UPSCALED[2], H3_UPSCALED[3])
        );
        assert_eq!(
            (PIXEL_HEIGHT / 32, PIXEL_WIDTH / 32),
            (ADAPTER_OUTPUT[2], ADAPTER_OUTPUT[3])
        );
    }

    #[test]
    fn author_node_roundtrip_is_identity() {
        let samples: Vec<f32> = (0..24).map(|i| i as f32 * 0.1 - 1.0).collect();
        let shape = [1, 24, 1, 1, 1];
        let network = author_node_input(&samples, shape).unwrap();
        let back = author_node_output(&network, shape).unwrap();
        for (got, want) in back.iter().zip(&samples) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }

    #[test]
    fn pixel_unshuffle_matches_pytorch_channel_order() {
        // NCHW 1x1x2x2 = [[1, 2], [3, 4]] → 1x4x1x1 channels [1, 2, 3, 4]
        // pytorch: out[:, c*r*r + i*r + j] = in[:, c, y*r+i, x*r+j]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = pixel_unshuffle2(&input, 1, 1, 2, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn resample_lerps_between_source_frames() {
        // T=2, one value per frame: 0 then 10. Target halfway.
        let latent = vec![0.0, 10.0];
        let out = temporal_resample(&latent, [1, 1, 2, 1, 1], &[0.0, 4.0], &[2.0]).unwrap();
        assert_eq!(out, vec![5.0]);
    }

    #[test]
    fn crop_keeps_the_first_sixteen_frames() {
        let latent: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let (out, shape) = crop_refiner_time(&latent, [1, 1, 20, 1, 1]).unwrap();
        assert_eq!(shape, [1, 1, 16, 1, 1]);
        assert_eq!(out, (0..16).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn nearest_pack_fills_slots_in_source_order() {
        // sources 0, 1, 10 against targets 0, 10. Two sources land on the first target.
        let latent = vec![10.0, 20.0, 30.0];
        let out =
            temporal_nearest_pack(&latent, [1, 1, 3, 1, 1], &[0.0, 1.0, 10.0], &[0.0, 10.0], 2)
                .unwrap();
        // channels are slots: slot0 = [10, 30], slot1 = [20, 0]
        assert_eq!(out, vec![10.0, 30.0, 20.0, 0.0]);
    }

    #[test]
    fn trilinear_midpoint_of_two_frames_is_the_average() {
        let out = trilinear_ncdhw(&[0.0, 10.0], [1, 1, 2, 1, 1], [1, 1, 1]).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-5, "{}", out[0]);
    }

    #[test]
    fn align_rejects_a_spatial_size_that_is_not_2x() {
        let err = align_h3_to_ltx(&[0.0, 0.0], [1, 1, 2, 1, 1], 1, 1, 1, 3).unwrap_err();
        assert!(err.contains("2x"), "{err}");
    }

    #[test]
    fn spark_lora_prefers_vsa_datafree_over_dense() {
        let spec = spark_adapter_spec();
        assert_eq!(spec.scale, 1.0);
        assert_eq!(
            spec.relative_paths[0],
            "FastH3-4-step-Preview-v1-LoRA/vsa-datafree/adapter_model.safetensors"
        );
        let vsa = spec
            .relative_paths
            .iter()
            .position(|p| p.contains("VSA-DataFree") || p.contains("vsa-datafree"))
            .unwrap();
        let dense = spec
            .relative_paths
            .iter()
            .position(|p| p.contains("dense-datafree"))
            .unwrap();
        assert!(vsa < dense);
        assert!(FIXED_PROMPT.starts_with("4K, refined,"));
    }
}
