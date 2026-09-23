//! TAEHV — the tiny autoencoder decoder, as an alternative to a full video VAE.
//!
//! Ported from <https://github.com/madebyollin/taehv> (`taehv.py`). Decoder only:
//! text-to-video never encodes. Two checkpoints share the same block list and
//! differ in the first/last conv, the spatial patch, and the temporal wrap:
//!
//! * [`TaeArch::Wan`] — `taew2_1.safetensors`, 16 channels, patch 1. Trim the
//!   first `t_upscale - 1` frames of the whole clip → Wan's `4n+1`.
//! * [`TaeArch::H3`] — `taeh3.safetensors`, 24 channels, patch 2. After the
//!   same 4× time grow, wrap like MiniMax-H3's 17-frame clips
//!   (`taehv.py` `_decode_h3_video`) → H3's `17n+5`.
//!
//! The Wan VAE is the largest single component of a clip we have not attacked —
//! 3.9 s of a 23.7 s 8-second clip on an H100. TAEHV trades quality for roughly
//! an order of magnitude less work.
//!
//! ## Architecture (taew2_1: `patch_size=1`, `latent_channels=16`)
//!
//! `nn.Sequential`, indices matching the checkpoint keys exactly, because the
//! keys are positional (`decoder.9.conv.0.weight`) and a renumbering would load
//! silently wrong weights:
//!
//! ```text
//!  0 Clamp                tanh(x/3)*3        (no parameters)
//!  1 conv 16   -> 256     3x3, bias
//!  2 ReLU
//!  3..5 MemBlock(256)
//!  6 Upsample x2          nearest
//!  7 TGrow  stride 1      1x1 256 -> 256     (time upscale off for this stage)
//!  8 conv 256  -> 128     3x3, no bias
//!  9..11 MemBlock(128)
//! 12 Upsample x2
//! 13 TGrow  stride 2      1x1 128 -> 256
//! 14 conv 128  -> 64      3x3, no bias
//! 15..17 MemBlock(64)
//! 18 Upsample x2
//! 19 TGrow  stride 2      1x1 64  -> 128
//! 20 conv 64   -> 64      3x3, no bias
//! 21 ReLU
//! 22 conv 64   -> 3       3x3, bias
//! ```
//!
//! Three spatial upsamples give 8x, two stride-2 TGrows give 4x in time —
//! matching Wan's VAE, so the same latent grid decodes to the same video shape.
//!
//! ## Temporal structure
//!
//! Every `MemBlock` reads the *previous timestep's* input to that same block.
//! The reference calls this `past`, and in its parallel path it is just the
//! input shifted one step along time with a zero frame in front. That makes the
//! whole decoder parallel over frames, which is what we implement: no
//! sequential feature cache, unlike the Wan VAE.
//!
//! `TGrow` with stride 2 turns one frame into two by splitting the channel
//! dimension of a 1x1 convolution, so the last stages run at the *output* frame
//! rate. Because the first stage's growth is disabled, the decoder emits
//! `4*T` frames for `T` latent frames and the first `t_upscale - 1 = 3` are
//! dropped — Wan's `4n+1` frame count falls out of that trim.

use super::tensor::{CudaTensor, Result, TensorError};
use super::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// `tanh(x/3) * 3`.
const CLAMP_SCALE: f32 = 3.0;
/// Frames the decoder emits before the first real one (`t_upscale - 1`).
const FRAMES_TO_TRIM: usize = 3;
/// Two stride-2 TGrows; the first TGrow is disabled (`stride == 1`).
const T_UPSCALE: usize = 4;
/// H3 encoder clips are 17 pixel frames = 5 latent tokens (`5 * t_upscale`).
const H3_CHUNK_FRAMES: usize = 5 * T_UPSCALE;
/// Trailing latent tokens the H3 encoder drops, in pixel frames (`3 * t_upscale`).
const H3_TOKEN_DROP_FRAMES: usize = 3 * T_UPSCALE;

/// Which checkpoint the sequential decoder was built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaeArch {
    /// Wan 2.1: 16 latent channels, 8× spatial, `4T − 3` frames.
    Wan,
    /// MiniMax-H3: 24 latent channels, 16× spatial (`8×` upsample + pixel-shuffle 2),
    /// then the 17-frame chunk wrap.
    H3,
}

impl TaeArch {
    pub fn latent_channels(self) -> usize {
        match self {
            Self::Wan => 16,
            Self::H3 => 24,
        }
    }

    /// Pixel-shuffle factor after the last conv. H3's 16× spatial is 8× nearest
    /// plus this; Wan's last conv already emits RGB.
    pub fn patch_size(self) -> usize {
        match self {
            Self::Wan => 1,
            Self::H3 => 2,
        }
    }

    fn first_conv_in(self) -> usize {
        self.latent_channels()
    }

    fn last_conv_out(self) -> usize {
        3 * self.patch_size() * self.patch_size()
    }
}

struct Conv {
    weight: CudaTensor,
    bias: Option<CudaTensor>,
    pad: usize,
}

impl Conv {
    fn load(
        map: &WeightMap,
        prefix: &str,
        out_c: usize,
        in_c: usize,
        k: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = super::weights::cuda_tensor_shaped(
            map,
            &format!("{prefix}.weight"),
            &[out_c, in_c, k, k],
        )?;
        let bias = if bias {
            Some(super::weights::cuda_tensor_shaped(
                map,
                &format!("{prefix}.bias"),
                &[out_c],
            )?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            pad: k / 2,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        x.conv2d(&self.weight, self.bias.as_ref(), self.pad, 1)
    }
}

/// `relu(conv(cat([x, past])) + x)` — the skip is the identity here because
/// every MemBlock in this checkpoint has `n_in == n_out`.
struct MemBlock {
    c0: Conv,
    c2: Conv,
    c4: Conv,
}

impl MemBlock {
    fn load(map: &WeightMap, prefix: &str, n: usize) -> Result<Self> {
        Ok(Self {
            c0: Conv::load(map, &format!("{prefix}.conv.0"), n, n * 2, 3, true)?,
            c2: Conv::load(map, &format!("{prefix}.conv.2"), n, n, 3, true)?,
            c4: Conv::load(map, &format!("{prefix}.conv.4"), n, n, 3, true)?,
        })
    }

    /// `x` and `past` are `[frames, c, h, w]`.
    fn forward(&self, x: &CudaTensor, past: &CudaTensor) -> Result<CudaTensor> {
        let h = CudaTensor::cat(&[x, past], 1)?;
        let h = relu(&self.c0.forward(&h)?);
        let h = relu(&self.c2.forward(&h)?);
        let h = self.c4.forward(&h)?;
        Ok(relu(&h.add(x)?))
    }
}

/// ReLU as a clamp: no dedicated kernel needed, and `clamp` is already device-side.
fn relu(x: &CudaTensor) -> CudaTensor {
    x.clamp(0.0, f32::INFINITY)
}

enum Block {
    Clamp,
    Conv(Conv),
    Relu,
    Mem(MemBlock),
    Upsample2,
    /// 1x1 convolution to `c * stride` channels, then split into `stride` frames.
    TGrow {
        conv: Conv,
        stride: usize,
    },
}

pub struct TaeHv {
    arch: TaeArch,
    blocks: Vec<Block>,
}

impl TaeHv {
    /// Load `taew2_1` from a directory holding `taew2_1.safetensors` (or any
    /// `WeightMap` whose keys are the reference's positional `decoder.N...`).
    pub fn load(map: &WeightMap) -> Result<Self> {
        Self::load_arch(map, TaeArch::Wan)
    }

    /// Same sequential decoder as [`Self::load`], with the first and last conv
    /// sized for `arch`. Keys stay `decoder.N...` — a renumbering would load
    /// silently wrong weights.
    pub fn load_arch(map: &WeightMap, arch: TaeArch) -> Result<Self> {
        use Block::*;
        let mem = |i: usize, n: usize| -> Result<Block> {
            Ok(Mem(MemBlock::load(map, &format!("decoder.{i}"), n)?))
        };
        let conv = |i: usize, o: usize, ic: usize, bias: bool| -> Result<Block> {
            Ok(Conv(self::Conv::load(
                map,
                &format!("decoder.{i}"),
                o,
                ic,
                3,
                bias,
            )?))
        };
        let tgrow = |i: usize, c: usize, stride: usize| -> Result<Block> {
            Ok(TGrow {
                conv: self::Conv::load(map, &format!("decoder.{i}.conv"), c * stride, c, 1, false)?,
                stride,
            })
        };
        Ok(Self {
            arch,
            blocks: vec![
                Clamp,
                conv(1, 256, arch.first_conv_in(), true)?,
                Relu,
                mem(3, 256)?,
                mem(4, 256)?,
                mem(5, 256)?,
                Upsample2,
                tgrow(7, 256, 1)?,
                conv(8, 128, 256, false)?,
                mem(9, 128)?,
                mem(10, 128)?,
                mem(11, 128)?,
                Upsample2,
                tgrow(13, 128, 2)?,
                conv(14, 64, 128, false)?,
                mem(15, 64)?,
                mem(16, 64)?,
                mem(17, 64)?,
                Upsample2,
                tgrow(19, 64, 2)?,
                conv(20, 64, 64, false)?,
                Relu,
                conv(22, arch.last_conv_out(), 64, true)?,
            ],
        })
    }

    /// `path` is a directory of safetensors or a single `.safetensors` file.
    pub fn load_from_path(path: &std::path::Path, arch: TaeArch) -> Result<Self> {
        let map = if path.is_file() {
            super::weights::WeightMap::open_files(&[path.to_path_buf()])?
        } else {
            super::weights::WeightMap::from_dir(path)?
        };
        Self::load_arch(&map, arch)
    }

    pub fn arch(&self) -> TaeArch {
        self.arch
    }

    /// `[1, 16, T, H, W]` latents → `[1, 3, 4T-3, 8H, 8W]` in `[-1, 1]`.
    ///
    /// The reference emits `[0, 1]`; this returns `[-1, 1]` to match what the
    /// Wan VAE decode gives the frame writer, so the two are interchangeable.
    ///
    /// Takes latents in **DiT space** — the reference documents its input as
    /// "~Gaussian", which is the denoiser's own scale, *before* the per-channel
    /// `latents_mean`/`latents_std` un-normalisation that the Wan VAE expects.
    /// That difference is the easiest thing to get silently wrong here, so the
    /// oracle stage checks it rather than trusting this comment.
    pub fn decode(&self, z: &CudaTensor) -> Result<CudaTensor> {
        self.decode_streaming(z, &mut |_, _| Ok(()))
    }

    /// `decode`, handing each chunk of finished frames to `sink(frame_offset,
    /// frames)` as soon as it exists — `[frames, 3, 8H, 8W]` in `[-1, 1]`, in
    /// order — so PNG encoding and the mp4 mux can run while the GPU is still
    /// on the next chunk. Returns the whole video exactly as `decode` does.
    pub fn decode_streaming(
        &self,
        z: &CudaTensor,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<CudaTensor> {
        let [n, c, t, h, w] = z.shape[..] else {
            return Err(msg(format!(
                "taehv expects [N, C, T, H, W] latents, got {:?}",
                z.shape
            )));
        };
        let want_c = self.arch.latent_channels();
        if n != 1 || c != want_c {
            return Err(msg(format!(
                "taehv ({:?}) expects [1, {want_c}, T, H, W], got {:?}",
                self.arch, z.shape
            )));
        }
        if self.arch == TaeArch::H3 {
            return self.decode_streaming_h3(z, c, t, h, w, sink);
        }

        // Being parallel over frames is what makes TAEHV fast and what makes it
        // run out of memory: every stage materialises every frame, and the last
        // one is 4T frames of 64 channels at full resolution — ~12.6 GB for a
        // 129-frame clip, on top of a resident DiT. So decode in chunks of
        // latent frames, carrying each MemBlock's boundary frame across the
        // seam, which is exactly what the reference's sequential path does with
        // its `memory[i]`. Results are identical to decoding in one go.
        let chunk = super::envflag::usize_flag("FASTVIDEO_TAEHV_CHUNK", 4).max(1);
        // One saved frame per block, at that block's own frame rate — the rate
        // differs after each TGrow, which is why this is indexed by block.
        let mut memory: Vec<Option<CudaTensor>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut out_chunks: Vec<CudaTensor> = Vec::new();
        let mut trimmed = false;
        let mut emitted = 0usize;

        let mut start = 0usize;
        while start < t {
            let len = chunk.min(t - start);
            let zc = z.narrow(2, start, len)?;
            let piece = self.decode_chunk(&zc, c, len, h, w, &mut memory)?;
            // Only the very first output frames are priming frames.
            let piece = if !trimmed {
                trimmed = true;
                let f = piece.shape[0];
                if f <= FRAMES_TO_TRIM {
                    return Err(msg(format!(
                        "taehv first chunk produced {f} frames, needs more than {FRAMES_TO_TRIM}; \
                         raise FASTVIDEO_TAEHV_CHUNK"
                    )));
                }
                piece.narrow(0, FRAMES_TO_TRIM, f - FRAMES_TO_TRIM)?
            } else {
                piece
            };
            // Clamp to the reference's [0, 1], then map to the [-1, 1] the
            // rest of the pipeline uses. Per chunk, so the sink sees finished
            // frames; elementwise, so it is the same as mapping after the cat.
            let piece = piece.clamp(0.0, 1.0).mul_scalar(2.0).add_scalar(-1.0);
            sink(emitted, &piece)?;
            emitted += piece.shape[0];
            out_chunks.push(piece);
            start += len;
        }

        let refs: Vec<&CudaTensor> = out_chunks.iter().collect();
        let x = CudaTensor::cat(&refs, 0)?;
        let (frames, oc, oh, ow) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        x.reshape(vec![1, frames, oc, oh, ow])?
            .permute(&[0, 2, 1, 3, 4])
    }

    /// One chunk of latent frames through every block, updating the per-block
    /// boundary memory. Returns `[frames, 3, 8H, 8W]`, untrimmed.
    fn decode_chunk(
        &self,
        zc: &CudaTensor,
        c: usize,
        len: usize,
        h: usize,
        w: usize,
        memory: &mut [Option<CudaTensor>],
    ) -> Result<CudaTensor> {
        let mut x = zc.reshape(vec![c, len, h, w])?.permute(&[1, 0, 2, 3])?;
        let mut frames = len;

        for (i, block) in self.blocks.iter().enumerate() {
            x = match block {
                Block::Clamp => tanh_scaled(&x, CLAMP_SCALE)?,
                Block::Relu => relu(&x),
                Block::Conv(cv) => cv.forward(&x)?,
                Block::Upsample2 => {
                    let (hh, ww) = (x.shape[2], x.shape[3]);
                    x.upsample_nearest2d(hh * 2, ww * 2)?
                }
                Block::Mem(m) => {
                    let past = match &memory[i] {
                        // A one-frame chunk's `past` is the carry alone — there
                        // is no earlier frame in this chunk to shift in, and
                        // narrowing to zero frames is not a valid tensor. A
                        // chunk size that does not divide the latent frame
                        // count always ends in a short chunk, and 33 frames
                        // with the default chunk of 4 ends in exactly one.
                        Some(prev) if frames == 1 => prev.clone(),
                        // Otherwise carry the previous chunk's last input frame
                        // into this chunk's first, so the seam is invisible.
                        Some(prev) => CudaTensor::cat(&[prev, &x.narrow(0, 0, frames - 1)?], 0)?,
                        None => shift_one_frame(&x, frames)?,
                    };
                    memory[i] = Some(x.narrow(0, frames - 1, 1)?);
                    m.forward(&x, &past)?
                }
                Block::TGrow { conv, stride } => {
                    let y = conv.forward(&x)?;
                    if *stride == 1 {
                        y
                    } else {
                        let (fc, hh, ww) = (y.shape[1] / stride, y.shape[2], y.shape[3]);
                        frames *= stride;
                        y.reshape(vec![frames, fc, hh, ww])?
                    }
                }
            };
        }
        Ok(x)
    }

    /// H3: grow `4T` frames, pixel-shuffle to 16× spatial, then the reference's
    /// 17-frame wrap (`taehv.py` `_decode_h3_video`). The wrap reads the whole
    /// clip (pad to a multiple of 20, drop 3 prefix frames per group, drop the
    /// last 12), so frames reach the sink after the decoder finishes.
    fn decode_streaming_h3(
        &self,
        z: &CudaTensor,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<CudaTensor> {
        let chunk = super::envflag::usize_flag("FASTVIDEO_TAEHV_CHUNK", 4).max(1);
        let mut memory: Vec<Option<CudaTensor>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut raw: Vec<CudaTensor> = Vec::new();
        let mut start = 0usize;
        while start < t {
            let len = chunk.min(t - start);
            let zc = z.narrow(2, start, len)?;
            let piece = pixel_shuffle2(&self.decode_chunk(&zc, c, len, h, w, &mut memory)?)?;
            raw.push(piece);
            start += len;
        }
        let refs: Vec<&CudaTensor> = raw.iter().collect();
        let mut frames = apply_h3_wrap(&CudaTensor::cat(&refs, 0)?)?;
        frames = frames.clamp(0.0, 1.0).mul_scalar(2.0).add_scalar(-1.0);
        // Hand the writer 17-frame groups so mux overlaps the last copy.
        let mut emitted = 0usize;
        while emitted < frames.shape[0] {
            let n = (frames.shape[0] - emitted).min(H3_CHUNK_FRAMES - FRAMES_TO_TRIM);
            let batch = frames.narrow(0, emitted, n)?;
            sink(emitted, &batch)?;
            emitted += n;
        }
        let (f, oc, oh, ow) = (
            frames.shape[0],
            frames.shape[1],
            frames.shape[2],
            frames.shape[3],
        );
        frames
            .reshape(vec![1, f, oc, oh, ow])?
            .permute(&[0, 2, 1, 3, 4])
    }
}

/// Pixel frames TAEH3 emits for `latent_frames` DiT tokens. For every
/// `5n + 2` latent count H3 requests, this is `17n + 5`.
pub fn h3_decoded_frames(latent_frames: usize) -> usize {
    h3_wrap_frame_count(latent_frames * T_UPSCALE)
}

/// After `4T` grown frames: pad T by `(-T) % 20` (Python unary-minus then
/// modulo — pad, not crop), drop the first 3 of each 20, drop the last 12.
fn h3_wrap_frame_count(raw_frames: usize) -> usize {
    let pad = (H3_CHUNK_FRAMES - raw_frames % H3_CHUNK_FRAMES) % H3_CHUNK_FRAMES;
    let groups = (raw_frames + pad) / H3_CHUNK_FRAMES;
    groups * (H3_CHUNK_FRAMES - FRAMES_TO_TRIM) - H3_TOKEN_DROP_FRAMES
}

/// `F.pixel_shuffle(x, 2)` on `[F, C*4, H, W]` → `[F, C, 2H, 2W]`.
fn pixel_shuffle2(x: &CudaTensor) -> Result<CudaTensor> {
    let [f, c4, h, w] = x.shape[..] else {
        return Err(msg(format!(
            "pixel_shuffle2 expects [F, C, H, W], got {:?}",
            x.shape
        )));
    };
    if c4 % 4 != 0 {
        return Err(msg(format!(
            "pixel_shuffle2: {c4} channels is not 4× a channel count"
        )));
    }
    let c = c4 / 4;
    x.reshape(vec![f, c, 2, 2, h, w])?
        .permute(&[0, 1, 4, 2, 5, 3])?
        .reshape(vec![f, c, h * 2, w * 2])
}

/// `taehv.py` `_decode_h3_video` after the sequential decoder, before
/// `postprocess_output_frames`. `x` is `[F, C, H, W]`.
fn apply_h3_wrap(x: &CudaTensor) -> Result<CudaTensor> {
    let [frames, c, h, w] = x.shape[..] else {
        return Err(msg(format!(
            "h3 wrap expects [F, C, H, W], got {:?}",
            x.shape
        )));
    };
    let pad = (H3_CHUNK_FRAMES - frames % H3_CHUNK_FRAMES) % H3_CHUNK_FRAMES;
    let x = if pad == 0 {
        x.clone()
    } else {
        let zeros = CudaTensor::zeros(&[pad, c, h, w]).to_device()?;
        CudaTensor::cat(&[x, &zeros], 0)?
    };
    let groups = x.shape[0] / H3_CHUNK_FRAMES;
    let body = x.reshape(vec![groups, H3_CHUNK_FRAMES, c, h, w])?.narrow(
        1,
        FRAMES_TO_TRIM,
        H3_CHUNK_FRAMES - FRAMES_TO_TRIM,
    )?;
    let kept = groups * (H3_CHUNK_FRAMES - FRAMES_TO_TRIM);
    let x = body.reshape(vec![kept, c, h, w])?;
    if kept <= H3_TOKEN_DROP_FRAMES {
        return Err(msg(format!(
            "h3 wrap: {frames} grown frames keep {kept}, cannot drop {H3_TOKEN_DROP_FRAMES}"
        )));
    }
    x.narrow(0, 0, kept - H3_TOKEN_DROP_FRAMES)
}

/// The previous frame's value at each position, with zeros before the first.
fn shift_one_frame(x: &CudaTensor, frames: usize) -> Result<CudaTensor> {
    let (c, h, w) = (x.shape[1], x.shape[2], x.shape[3]);
    let zero = CudaTensor::zeros(&[1, c, h, w]).to_device()?;
    if frames == 1 {
        return Ok(zero);
    }
    CudaTensor::cat(&[&zero, &x.narrow(0, 0, frames - 1)?], 0)
}

fn tanh_scaled(x: &CudaTensor, s: f32) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    if let Some(d) = x.dev()? {
        return CudaTensor::from_dev_result(
            super::ops::tanh_scaled_device(&d, s)?,
            x.shape.clone(),
        );
    }
    let host = x.host_cow()?;
    let data: Vec<f32> = host.iter().map(|v| (v / s).tanh() * s).collect();
    CudaTensor::from_vec(data, x.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic weights: this exercises shapes and the frame
    /// arithmetic, which is what the port can get wrong without any GPU.
    fn tiny_map() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let n: usize = shape.iter().product();
            let seed = key
                .bytes()
                .fold(17u64, |a, b| a.wrapping_mul(31).wrapping_add(u64::from(b)));
            (0..n)
                .map(|i| {
                    let x = seed
                        .wrapping_add(i as u64)
                        .wrapping_mul(6364136223846793005)
                        >> 33;
                    ((x % 2000) as f32 / 1000.0 - 1.0) * 0.05
                })
                .collect()
        })
    }

    /// Wan's 4n+1 frame count has to fall out of `4T` frames minus the 3
    /// priming frames — if the trim or a TGrow stride were wrong this is the
    /// check that catches it, and it needs no weights to be meaningful.
    #[test]
    fn frame_count_is_four_n_plus_one() {
        let tae = TaeHv::load(&tiny_map()).expect("load");
        for (latent_t, want_frames) in [(1usize, 1usize), (2, 5), (3, 9)] {
            let z = CudaTensor::zeros(&[1, 16, latent_t, 2, 2]);
            let out = tae.decode(&z).expect("decode");
            assert_eq!(
                out.shape,
                vec![1, 3, want_frames, 16, 16],
                "T={latent_t} should decode to {want_frames} frames at 8x spatial"
            );
        }
    }

    /// A single latent frame decodes to exactly one output frame: 4 grown
    /// frames minus 3 trimmed. Off-by-one here would silently shorten clips.
    #[test]
    fn one_latent_frame_survives_the_trim() {
        let tae = TaeHv::load(&tiny_map()).expect("load");
        let out = tae
            .decode(&CudaTensor::zeros(&[1, 16, 1, 2, 2]))
            .expect("decode");
        assert_eq!(out.shape, vec![1, 3, 1, 16, 16]);
    }

    /// `past` is the previous frame with zeros before the first — the whole
    /// temporal structure of the decoder rests on this one shift.
    #[test]
    fn shift_one_frame_zero_pads_the_front() {
        let x = CudaTensor::from_vec((0..6).map(|v| v as f32).collect(), vec![3, 2, 1, 1]).unwrap();
        let s = shift_one_frame(&x, 3).unwrap();
        assert_eq!(
            s.host_cow().unwrap().as_ref(),
            &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
        );
    }

    /// The whole point of the boundary carry: chunking must be invisible.
    /// Without it each chunk would restart every MemBlock's `past` from zero
    /// and leave a seam every `chunk` latent frames — a defect that looks like
    /// a periodic flicker in the video and like nothing at all in the shapes.
    #[test]
    fn chunking_does_not_change_the_result() {
        let tae = TaeHv::load(&tiny_map()).expect("load");
        let z = CudaTensor::from_vec(
            (0..(16 * 6 * 2 * 2))
                .map(|i| ((i % 37) as f32 / 37.0) - 0.5)
                .collect(),
            vec![1, 16, 6, 2, 2],
        )
        .unwrap();

        let whole = {
            std::env::set_var("FASTVIDEO_TAEHV_CHUNK", "64");
            tae.decode(&z).expect("whole")
        };
        let chunked = {
            std::env::set_var("FASTVIDEO_TAEHV_CHUNK", "2");
            tae.decode(&z).expect("chunked")
        };
        std::env::remove_var("FASTVIDEO_TAEHV_CHUNK");

        assert_eq!(
            whole.shape, chunked.shape,
            "chunking changed the frame count"
        );
        let (a, b) = (whole.host_cow().unwrap(), chunked.host_cow().unwrap());
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "chunk seam changed the output by {worst}");
    }

    /// The streaming sink sees exactly the frames `decode` returns, in order,
    /// already mapped to [-1, 1]: the writer thread must never get a frame
    /// the video does not contain, nor a frame in the reference's [0, 1].
    #[test]
    fn streaming_sink_sees_every_frame_in_order() {
        let tae = TaeHv::load(&tiny_map()).expect("load");
        let z = CudaTensor::from_vec(
            (0..(16 * 5 * 2 * 2))
                .map(|i| ((i % 41) as f32 / 41.0) - 0.5)
                .collect(),
            vec![1, 16, 5, 2, 2],
        )
        .unwrap();
        std::env::set_var("FASTVIDEO_TAEHV_CHUNK", "2");
        let mut seen: Vec<(usize, CudaTensor)> = Vec::new();
        let out = tae
            .decode_streaming(&z, &mut |off, frames| {
                seen.push((off, frames.clone()));
                Ok(())
            })
            .expect("streamed");
        std::env::remove_var("FASTVIDEO_TAEHV_CHUNK");

        assert!(
            seen.len() > 1,
            "a 5-frame latent in chunks of 2 must stream more than one batch"
        );
        let mut next = 0;
        let refs: Vec<CudaTensor> = seen
            .iter()
            .map(|(off, t)| {
                assert_eq!(*off, next, "batches must arrive in frame order");
                next += t.shape[0];
                t.clone()
            })
            .collect();
        assert_eq!(
            next, out.shape[2],
            "streamed frame count != decoded frame count"
        );
        let refs: Vec<&CudaTensor> = refs.iter().collect();
        let streamed = CudaTensor::cat(&refs, 0).unwrap();
        // [F, 3, H, W] vs the video's [1, 3, F, H, W].
        let video = out
            .reshape(vec![3, out.shape[2], out.shape[3], out.shape[4]])
            .unwrap()
            .permute(&[1, 0, 2, 3])
            .unwrap();
        let (a, b) = (streamed.host_cow().unwrap(), video.host_cow().unwrap());
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(b.iter()).all(|(x, y)| x == y),
            "streamed frames differ from the video"
        );
    }

    /// A chunk size that does not divide the latent frame count leaves a short
    /// final chunk, and Wan's 33 latent frames with the default chunk of 4 end
    /// in a chunk of exactly one — where there is no earlier frame to shift in.
    /// This is what a full 129-frame clip hits.
    #[test]
    fn a_ragged_final_chunk_still_decodes() {
        let tae = TaeHv::load(&tiny_map()).expect("load");
        let z = CudaTensor::from_vec(
            (0..(16 * 5 * 2 * 2))
                .map(|i| ((i % 13) as f32 / 13.0) - 0.5)
                .collect(),
            vec![1, 16, 5, 2, 2],
        )
        .unwrap();
        std::env::set_var("FASTVIDEO_TAEHV_CHUNK", "4"); // 5 = 4 + 1
        let ragged = tae.decode(&z).expect("ragged tail");
        std::env::set_var("FASTVIDEO_TAEHV_CHUNK", "64");
        let whole = tae.decode(&z).expect("whole");
        std::env::remove_var("FASTVIDEO_TAEHV_CHUNK");
        assert_eq!(ragged.shape, whole.shape);
        let (a, b) = (ragged.host_cow().unwrap(), whole.host_cow().unwrap());
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "ragged tail changed the output by {worst}");
    }

    #[test]
    fn output_is_in_minus_one_to_one() {
        let tae = TaeHv::load(&tiny_map()).expect("load");
        let out = tae
            .decode(&CudaTensor::zeros(&[1, 16, 2, 2, 2]))
            .expect("decode");
        let host = out.host_cow().unwrap();
        assert!(
            host.iter().all(|v| (-1.0..=1.0).contains(v)),
            "output escaped [-1, 1]"
        );
    }

    #[test]
    fn wrong_latent_channels_is_an_error() {
        let tae = TaeHv::load(&tiny_map()).expect("load");
        assert!(tae.decode(&CudaTensor::zeros(&[1, 32, 2, 2, 2])).is_err());
    }

    /// `(-T) % 20` pad, per-group prefix trim, drop last 12. The three
    /// request lengths H3 accepts (`5n+2` latents) must land on `17n+5`.
    #[test]
    fn h3_wrap_matches_official_frame_counts() {
        assert_eq!(h3_decoded_frames(2), 5);
        assert_eq!(h3_decoded_frames(7), 22);
        assert_eq!(h3_decoded_frames(37), 124);
        assert_eq!(h3_decoded_frames(72), 243);
        assert_eq!(h3_decoded_frames(107), 362);
    }

    /// Wrap is a gather: 28 grown frames → 22 kept, the pad zeros are exactly
    /// the 12 that get dropped, so the output is raw frames 3..19 and 23..27.
    #[test]
    fn h3_wrap_keeps_the_reference_indices() {
        let raw: Vec<f32> = (0..28).map(|v| v as f32).collect();
        let x = CudaTensor::from_vec(raw, vec![28, 1, 1, 1]).unwrap();
        let wrapped = apply_h3_wrap(&x).unwrap();
        let got = wrapped.host_cow().unwrap();
        let want: Vec<f32> = (3..20).chain(23..28).map(|v| v as f32).collect();
        assert_eq!(got.as_ref(), want.as_slice());
    }

    #[test]
    fn pixel_shuffle2_is_channel_to_spatial() {
        // Channel-major 2×2 tiles: values 0..11 in [1, 12, 1, 1] become
        // [1, 3, 2, 2] with each RGB channel a 2×2 of consecutive numbers.
        let x =
            CudaTensor::from_vec((0..12).map(|v| v as f32).collect(), vec![1, 12, 1, 1]).unwrap();
        let y = pixel_shuffle2(&x).unwrap();
        assert_eq!(y.shape, vec![1, 3, 2, 2]);
        assert_eq!(
            y.host_cow().unwrap().as_ref(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
    }

    #[test]
    fn h3_two_latents_decode_to_five_frames_at_16x() {
        let tae = TaeHv::load_arch(&tiny_map(), TaeArch::H3).expect("load");
        let out = tae
            .decode(&CudaTensor::zeros(&[1, 24, 2, 2, 2]))
            .expect("decode");
        assert_eq!(
            out.shape,
            vec![1, 3, 5, 32, 32],
            "2 latents, 2×2, patch 2 → 5 frames at 16×"
        );
        let host = out.host_cow().unwrap();
        assert!(host.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn h3_rejects_wan_channel_count() {
        let tae = TaeHv::load_arch(&tiny_map(), TaeArch::H3).expect("load");
        assert!(tae.decode(&CudaTensor::zeros(&[1, 16, 2, 2, 2])).is_err());
    }
}
