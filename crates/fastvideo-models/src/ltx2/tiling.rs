//! Conv video VAE decode tiling, as `ltx_core` plans it (`ltx_core/tiling.py`,
//! `model/video_vae/video_vae.py:549-592`, `conv_video_decoder.py:359-381`).
//!
//! The reference never decodes a clip in one piece. It cuts the latent into
//! overlapping tiles on the time, height and width axes, decodes each tile on
//! its own (the convolutions see the tile border, not the neighbour), weights
//! each decoded tile by a separable trapezoid and sums the overlaps. Seams are
//! therefore part of the output, so the port plans exactly the same tiles and
//! masks. The plan is host arithmetic; the decode and the blend run on the
//! device (`fastvideo-cudarc::ltx2::vae`).
//!
//! Layouts used by the references:
//! * [`TileSizeConfig::conv_auto`] — `AUTO_TILING` for the Conv VAE
//!   (`ltx_pipelines/utils/helpers.py:60-97`): a 768 px / 64 px long-side tile
//!   scaled to the aspect ratio, and 80 / 24 frames in time.
//! * [`TileSizeConfig::spark`] — the Spark refiner's explicit tiles
//!   (`Sol-H3-Spark/runtime/stage2_ops/models.py:20-22`): 128 / 24 frames,
//!   448 / 64 height, 768 / 64 width.

/// One tile along one axis, in latent units (`DimensionInterval`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub start: usize,
    pub end: usize,
    pub left_ramp: usize,
    pub right_ramp: usize,
}

/// `DimensionSizeConfig`: pixel / frame units; `tile_size == 0` is untiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DimSize {
    pub tile_size: usize,
    pub overlap: usize,
}

impl DimSize {
    pub const UNTILED: Self = Self {
        tile_size: 0,
        overlap: 0,
    };

    pub fn new(tile_size: usize, overlap: usize) -> Self {
        Self { tile_size, overlap }
    }

    pub fn is_tiled(&self) -> bool {
        self.tile_size > 0
    }
}

/// `TileSizeConfig` for a `(frames, height, width)` video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileSizeConfig {
    pub frames: DimSize,
    pub height: DimSize,
    pub width: DimSize,
}

/// The Conv VAE's `(time, height, width)` scale factors (`VIDEO_SCALE_FACTORS`).
pub const VIDEO_SCALE: [usize; 3] = [8, 32, 32];

impl TileSizeConfig {
    /// `TileSizeConfig.from_long_side` (`tiling.py:755-795`): the long side's
    /// tile scaled in *latent* units, `round(size · axis / long)` with Python's
    /// half-to-even `round`, both axes sharing the long side's overlap.
    pub fn from_long_side(
        long_side: DimSize,
        height: usize,
        width: usize,
        scale_hw: [usize; 2],
        frames: DimSize,
    ) -> Result<Self, String> {
        if height == 0 || width == 0 || !long_side.is_tiled() {
            return Err(format!(
                "ltx2 tiling: from_long_side needs a tiled long side and a positive {width}x{height}"
            ));
        }
        let span = height.max(width);
        let axis_size = |axis_len: usize, factor: usize| -> usize {
            let axis_lat = axis_len / factor;
            let long_lat = span / factor;
            let size_lat = long_side.tile_size / factor;
            let overlap_lat = long_side.overlap / factor;
            let lower = 2usize.max(overlap_lat + 1);
            let scaled = python_round(size_lat as f64 * axis_lat as f64 / long_lat as f64);
            let tile_lat = lower.max(scaled);
            (tile_lat * factor).max((2 * factor).max(long_side.overlap + factor))
        };
        Ok(Self {
            frames,
            height: DimSize::new(axis_size(height, scale_hw[0]), long_side.overlap),
            width: DimSize::new(axis_size(width, scale_hw[1]), long_side.overlap),
        })
    }

    /// `AUTO_TILING` for the Conv VAE (`helpers.py:60-97`).
    pub fn conv_auto(height: usize, width: usize) -> Result<Self, String> {
        Self::from_long_side(
            DimSize::new(768, 64),
            height,
            width,
            [VIDEO_SCALE[1], VIDEO_SCALE[2]],
            DimSize::new(80, 24),
        )
    }

    /// The Spark refiner's explicit Conv tiles (`stage2_ops/models.py:20-22`).
    pub fn spark() -> Self {
        Self {
            frames: DimSize::new(128, 24),
            height: DimSize::new(448, 64),
            width: DimSize::new(768, 64),
        }
    }

    /// `TileSizeConfig.validate` (`tiling.py:733-744`, `_validate_size_axis`).
    pub fn validate(&self, scale: [usize; 3]) -> Result<(), String> {
        for (name, cfg, factor) in [
            ("frames", self.frames, scale[0]),
            ("height", self.height, scale[1]),
            ("width", self.width, scale[2]),
        ] {
            if !cfg.is_tiled() {
                if cfg.overlap != 0 {
                    return Err(format!("ltx2 tiling: untiled {name} has an overlap"));
                }
                continue;
            }
            if cfg.overlap >= cfg.tile_size {
                return Err(format!("ltx2 tiling: {name} overlap >= tile size"));
            }
            if cfg.tile_size < 2 * factor
                || cfg.tile_size % factor != 0
                || cfg.overlap % factor != 0
            {
                return Err(format!(
                    "ltx2 tiling: {name} tile {} / overlap {} is not on the x{factor} grid",
                    cfg.tile_size, cfg.overlap
                ));
            }
        }
        Ok(())
    }
}

/// Python 3 `round`: half to even.
fn python_round(x: f64) -> usize {
    let r = x.round();
    let r = if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    };
    r.max(0.0) as usize
}

fn whole(len: usize) -> Vec<Interval> {
    vec![Interval {
        start: 0,
        end: len,
        left_ramp: 0,
        right_ramp: 0,
    }]
}

/// `split_by_size(size, overlap)` with `min_tile_size=None` (`tiling.py:173-222`).
pub fn split_by_size(len: usize, size: usize, overlap: usize) -> Vec<Interval> {
    if size == 0 || overlap >= size || len <= size {
        return whole(len);
    }
    let amount = (len + size - 2 * overlap - 1) / (size - overlap);
    let mut out = vec![Interval {
        start: 0,
        end: size,
        left_ramp: 0,
        right_ramp: overlap,
    }];
    for i in 1..amount.saturating_sub(1) {
        out.push(Interval {
            start: i * (size - overlap),
            end: i * (size - overlap) + size,
            left_ramp: overlap,
            right_ramp: overlap,
        });
    }
    out.push(Interval {
        start: (amount - 1) * (size - overlap),
        end: len,
        left_ramp: overlap,
        right_ramp: 0,
    });
    out
}

/// `split_temporal_causal` (`tiling.py:225-251`): every tile after the first
/// starts one latent frame earlier, with a ramp one longer.
pub fn split_temporal_causal(len: usize, size: usize, overlap: usize) -> Vec<Interval> {
    if len <= size {
        return whole(len);
    }
    let mut out = split_by_size(len, size, overlap);
    for iv in out.iter_mut().skip(1) {
        iv.start -= 1;
        iv.left_ramp += 1;
    }
    out
}

/// `torch.linspace(start, end, steps)` on the CPU in float32: the first half
/// counts up from `start`, the second half down from `end`.
pub fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    match steps {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) / (steps - 1) as f32;
            let half = steps / 2;
            (0..steps)
                .map(|i| {
                    if i < half {
                        start + step * i as f32
                    } else {
                        end - step * (steps - i - 1) as f32
                    }
                })
                .collect()
        }
    }
}

/// `compute_trapezoidal_mask_1d` (`tiling.py:13-49`).
pub fn trapezoidal_mask(
    length: usize,
    ramp_left: usize,
    ramp_right: usize,
    left_starts_from_0: bool,
) -> Vec<f32> {
    let ramp_left = ramp_left.min(length);
    let ramp_right = ramp_right.min(length);
    let mut mask = vec![1.0f32; length];
    if ramp_left > 0 {
        let interval = if left_starts_from_0 {
            ramp_left + 1
        } else {
            ramp_left + 2
        };
        let mut fade_in = linspace_f32(0.0, 1.0, interval);
        fade_in.pop();
        if !left_starts_from_0 {
            fade_in.remove(0);
        }
        for (m, f) in mask.iter_mut().zip(fade_in) {
            *m *= f;
        }
    }
    if ramp_right > 0 {
        let fade_out = linspace_f32(1.0, 0.0, ramp_right + 2);
        let fade_out = &fade_out[1..fade_out.len() - 1];
        for (m, f) in mask[length - ramp_right..].iter_mut().zip(fade_out) {
            *m *= f;
        }
    }
    mask.iter_mut().for_each(|m| *m = m.clamp(0.0, 1.0));
    mask
}

/// One tile along one axis: where it is cut from the latent, where its decode
/// lands in the video, and its blend weights over that output span. `mask`
/// is `None` on an untiled axis (a length-1 ones mask in the reference).
#[derive(Debug, Clone, PartialEq)]
pub struct AxisTile {
    pub latent: std::ops::Range<usize>,
    pub out: std::ops::Range<usize>,
    pub mask: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// `map_temporal_slice` (`video_vae.py:549-555`): `8(n−1)+1` frames.
    Time,
    /// `map_spatial_slice` (`video_vae.py:586-592`).
    Space,
}

/// The tiles of one axis for a latent of `len` cells: the split of
/// `TileSizeConfig.to_splitters` (`tiling.py:797-833`) and the mapping of
/// `ConvVideoDecoder._prepare_tiles` (`conv_video_decoder.py:359-381`).
pub fn axis_tiles(len: usize, cfg: DimSize, factor: usize, axis: Axis) -> Vec<AxisTile> {
    let out_len = match axis {
        Axis::Time => (len.max(1) - 1) * factor + 1,
        Axis::Space => len * factor,
    };
    if !cfg.is_tiled() {
        return vec![AxisTile {
            latent: 0..len,
            out: 0..out_len,
            mask: None,
        }];
    }
    let size = cfg.tile_size / factor;
    let overlap = cfg.overlap / factor;
    let tile = size.max(2usize.max(overlap + 1));
    let intervals = match axis {
        Axis::Time => split_temporal_causal(len, tile, overlap),
        Axis::Space => split_by_size(len, tile, overlap),
    };
    intervals
        .into_iter()
        .map(|iv| {
            let (out, mask) = match axis {
                Axis::Time => {
                    let start = iv.start * factor;
                    let stop = 1 + (iv.end - 1) * factor;
                    let left = if iv.left_ramp == 0 {
                        0
                    } else {
                        1 + (iv.left_ramp - 1) * factor
                    };
                    let right = iv.right_ramp * factor;
                    (
                        start..stop,
                        trapezoidal_mask(stop - start, left, right, true),
                    )
                }
                Axis::Space => {
                    let (start, stop) = (iv.start * factor, iv.end * factor);
                    (
                        start..stop,
                        trapezoidal_mask(
                            stop - start,
                            iv.left_ramp * factor,
                            iv.right_ramp * factor,
                            false,
                        ),
                    )
                }
            };
            AxisTile {
                latent: iv.start..iv.end,
                out,
                mask: Some(mask),
            }
        })
        .collect()
}

/// Per-output-position sum of every tile's weight along one axis (the
/// separable factor of `compute_summed_weights`).
pub fn axis_weight_sum(tiles: &[AxisTile], out_len: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; out_len];
    for t in tiles {
        match &t.mask {
            Some(m) => {
                for (a, w) in acc[t.out.clone()].iter_mut().zip(m) {
                    *a += w;
                }
            }
            None => acc[t.out.clone()].iter_mut().for_each(|a| *a += 1.0),
        }
    }
    acc
}

/// `masks_are_complementary` for one axis (`tiling.py:421-455`, `atol=1e-5`).
pub fn axis_complementary(tiles: &[AxisTile], out_len: usize) -> bool {
    axis_weight_sum(tiles, out_len)
        .iter()
        .all(|&s| (s - 1.0).abs() <= 1e-5)
}

/// The whole plan for a `[F, H, W]` latent grid.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodePlan {
    pub time: Vec<AxisTile>,
    pub height: Vec<AxisTile>,
    pub width: Vec<AxisTile>,
    /// Output `[frames, height, width]`.
    pub out: [usize; 3],
    /// Every axis sums to one: the reference then skips the weight division.
    pub complementary: bool,
}

impl DecodePlan {
    pub fn new(grid: [usize; 3], cfg: &TileSizeConfig, scale: [usize; 3]) -> Result<Self, String> {
        cfg.validate(scale)?;
        let [f, h, w] = grid;
        if f == 0 || h == 0 || w == 0 {
            return Err(format!("ltx2 tiling: empty latent grid {grid:?}"));
        }
        let time = axis_tiles(f, cfg.frames, scale[0], Axis::Time);
        let height = axis_tiles(h, cfg.height, scale[1], Axis::Space);
        let width = axis_tiles(w, cfg.width, scale[2], Axis::Space);
        let out = [(f - 1) * scale[0] + 1, h * scale[1], w * scale[2]];
        let complementary = axis_complementary(&time, out[0])
            && axis_complementary(&height, out[1])
            && axis_complementary(&width, out[2]);
        Ok(Self {
            time,
            height,
            width,
            out,
            complementary,
        })
    }

    pub fn tiles(&self) -> usize {
        self.time.len() * self.height.len() * self.width.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_torch_endpoints_and_halves() {
        assert_eq!(linspace_f32(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace_f32(1.0, 0.0, 3), vec![1.0, 0.5, 0.0]);
        assert_eq!(linspace_f32(2.0, 3.0, 1), vec![2.0]);
    }

    #[test]
    fn trapezoid_ramps_partition_unity_across_an_overlap() {
        // Spatial: two tiles overlapping by 64 px, ramps exclude 0 and 1.
        let a = trapezoidal_mask(448, 0, 64, false);
        let b = trapezoidal_mask(384, 64, 0, false);
        assert_eq!(a[..384], vec![1.0; 384][..]);
        for k in 0..64 {
            assert!((a[384 + k] + b[k] - 1.0).abs() < 1e-6, "k={k}");
        }
        assert!(b[0] > 0.0 && b[63] < 1.0 && b[64] == 1.0);
        // Temporal left ramps start from 0.
        let t = trapezoidal_mask(73, 25, 0, true);
        assert_eq!(t[0], 0.0);
        assert!((t[1] - 0.04).abs() < 1e-6);
        assert_eq!(t[25], 1.0);
    }

    #[test]
    fn conv_auto_couples_the_long_side_to_the_aspect() {
        // 1344x768: long side 768 px (24 latents) on the width; height
        // round(24·24/42) = 14 latents = 448 px — the Spark tile.
        let c = TileSizeConfig::conv_auto(768, 1344).unwrap();
        assert_eq!(c.width, DimSize::new(768, 64));
        assert_eq!(c.height, DimSize::new(448, 64));
        assert_eq!(c.frames, DimSize::new(80, 24));
        // Square: both 768.
        let s = TileSizeConfig::conv_auto(1024, 1024).unwrap();
        assert_eq!((s.height.tile_size, s.width.tile_size), (768, 768));
        // Portrait mirrors landscape.
        let p = TileSizeConfig::conv_auto(1344, 768).unwrap();
        assert_eq!((p.height.tile_size, p.width.tile_size), (768, 448));
        assert!(TileSizeConfig::spark().validate(VIDEO_SCALE).is_ok());
        assert!(TileSizeConfig {
            height: DimSize::new(100, 64),
            ..TileSizeConfig::spark()
        }
        .validate(VIDEO_SCALE)
        .is_err());
    }

    #[test]
    fn python_round_is_half_even() {
        assert_eq!(python_round(2.5), 2);
        assert_eq!(python_round(3.5), 4);
        assert_eq!(python_round(13.71), 14);
    }

    #[test]
    fn split_by_size_matches_the_reference_layout() {
        assert_eq!(
            split_by_size(24, 14, 2),
            vec![
                Interval {
                    start: 0,
                    end: 14,
                    left_ramp: 0,
                    right_ramp: 2
                },
                Interval {
                    start: 12,
                    end: 24,
                    left_ramp: 2,
                    right_ramp: 0
                },
            ]
        );
        assert_eq!(split_by_size(10, 24, 2), whole(10));
        // Three tiles: 0..24, 22..46, 44..60.
        let t = split_by_size(60, 24, 2);
        assert_eq!(t.len(), 3);
        assert_eq!(
            (t[1].start, t[1].end, t[2].start, t[2].end),
            (22, 46, 44, 60)
        );
    }

    #[test]
    fn a_121_frame_clip_splits_into_two_causal_temporal_tiles() {
        // 16 latent frames, 80/24 frames → 10/3 latents.
        let t = axis_tiles(16, DimSize::new(80, 24), 8, Axis::Time);
        assert_eq!(t.len(), 2);
        assert_eq!((t[0].latent.clone(), t[0].out.clone()), (0..10, 0..73));
        // Shifted back one latent, ramp one longer: 25 frames from 0.
        assert_eq!((t[1].latent.clone(), t[1].out.clone()), (6..16, 48..121));
        let (m0, m1) = (t[0].mask.as_ref().unwrap(), t[1].mask.as_ref().unwrap());
        assert_eq!(m0.len(), 73);
        assert_eq!(m1.len(), 73);
        assert!(axis_complementary(&t, 121));
        // Untiled when the clip fits: the Spark 128-frame tile.
        let s = axis_tiles(16, DimSize::new(128, 24), 8, Axis::Time);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].out, 0..121);
        assert!(s[0].mask.as_ref().unwrap().iter().all(|&m| m == 1.0));
        assert_eq!(
            axis_tiles(16, DimSize::UNTILED, 8, Axis::Time)[0].mask,
            None
        );
    }

    #[test]
    fn spark_plan_is_two_by_two_spatial_tiles() {
        // 1344x768 x121 → latent [16, 24, 42].
        let plan = DecodePlan::new([16, 24, 42], &TileSizeConfig::spark(), VIDEO_SCALE).unwrap();
        assert_eq!(plan.out, [121, 768, 1344]);
        assert_eq!(
            (plan.time.len(), plan.height.len(), plan.width.len()),
            (1, 2, 2)
        );
        assert_eq!(plan.height[0].out, 0..448);
        assert_eq!(plan.height[1].out, 384..768);
        assert_eq!(plan.width[1].latent, 22..42);
        assert!(plan.complementary);
        let auto = DecodePlan::new(
            [16, 24, 42],
            &TileSizeConfig::conv_auto(768, 1344).unwrap(),
            VIDEO_SCALE,
        )
        .unwrap();
        assert_eq!(auto.tiles(), 2 * 2 * 2);
        assert!(auto.complementary);
    }
}
