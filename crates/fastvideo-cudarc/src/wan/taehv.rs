//! TAEHV — madebyollin's tiny video autoencoders, as an alternative to a full
//! video VAE.
//!
//! Ported from <https://github.com/madebyollin/taehv> (`taehv.py`; the copies
//! sol-engine vendors are cited as `ltx:taehv.py:N` —
//! `models/ltx2.5-refiner/GB200/vendor/taehv/taehv.py`, commit 32ac014 — and
//! `h3:taehv.py:N` — `models/minimax_h3/super_acceleration/vendor/taeh3/taehv.py`,
//! commit e589fdd). Three checkpoints, one block vocabulary:
//!
//! * [`TaeArch::Wan`] — `taew2_1.safetensors`, 16 channels, patch 1. Trim the
//!   first `t_upscale - 1` frames of the whole clip → Wan's `4n+1`.
//! * [`TaeArch::H3`] — `taeh3.safetensors`, 24 channels, patch 2. After the
//!   same 4× time grow, wrap like MiniMax-H3's 17-frame clips
//!   (`h3:taehv.py:279-288` `_decode_h3_video`) → H3's `17n+5`. Decode only:
//!   sol-engine's H3 runs (`super_acceleration/stage1/taeh3_decoder_telemetry_overlay.py`)
//!   use TAEH3 solely as the video decoder, never its encoder.
//! * [`TaeArch::LtxWide`] — `taeltx2_3_wide.safetensors`, 128 channels,
//!   patch 4, all three TGrows and TPools at stride 2 (`ltx:taehv.py:204-205`),
//!   and the *wide* decoder (`ltx:taehv.py:221-229`): widths
//!   `[1024, 512, 256, 64]` with [`WideMemBlock`]s. Encode **and** decode, as
//!   sol-engine's LTX-2.5 refiner uses it (`refiner_head_cp.py:520-525`
//!   encode, `:567-578` decode). 32× spatial, 8× time: `8n+1` frames.
//!
//! The Wan VAE is the largest single component of a clip we have not attacked —
//! 3.9 s of a 23.7 s 8-second clip on an H100. TAEHV trades quality for roughly
//! an order of magnitude less work.
//!
//! ## Decoder (taew2_1 / taeh3: `n_f = [256, 128, 64, 64]`)
//!
//! `nn.Sequential`, indices matching the checkpoint keys exactly, because the
//! keys are positional (`decoder.9.conv.0.weight`) and a renumbering would load
//! silently wrong weights:
//!
//! ```text
//!  0 Clamp                tanh(x/3)*3        (no parameters)
//!  1 conv C    -> n0      3x3, bias
//!  2 ReLU
//!  3..5 MemBlock(n0)      (WideMemBlock for taeltx2_3_wide)
//!  6 Upsample x2          nearest
//!  7 TGrow  stride s0     1x1 n0 -> n0*s0    (s0 = 1 for Wan/H3, 2 for LTX)
//!  8 conv n0   -> n1      3x3, no bias
//!  9..11 MemBlock(n1)
//! 12 Upsample x2
//! 13 TGrow  stride 2      1x1 n1 -> 2*n1
//! 14 conv n1   -> n2      3x3, no bias
//! 15..17 MemBlock(n2)
//! 18 Upsample x2
//! 19 TGrow  stride 2      1x1 n2 -> 2*n2
//! 20 conv n2   -> 64      3x3, no bias
//! 21 ReLU
//! 22 conv 64   -> 3*p*p   3x3, bias, then pixel-shuffle p
//! ```
//!
//! ## Encoder (`ltx:taehv.py:206-212`)
//!
//! ```text
//!  0 conv 3*p*p -> 64     3x3, bias   (after pixel-unshuffle p)
//!  1 ReLU
//!  2 TPool stride t0      1x1 64*t0 -> 64
//!  3 conv 64 -> 64        3x3 stride 2, no bias
//!  4..6 MemBlock(64)
//!  7..11, 12..16          the same stage twice more
//! 17 conv 64 -> C         3x3, bias
//! ```
//!
//! ## Temporal structure
//!
//! Every `MemBlock` reads the *previous timestep's* input to that same block.
//! The reference calls this `past`, and in its parallel path it is just the
//! input shifted one step along time with a zero frame in front
//! (`ltx:taehv.py:83-88`). The sequential path (`ltx:taehv.py:95-138`, which
//! the refiner runs: `refiner_head_cp.py:521,568` `parallel=False`) keeps the
//! same frame in `memory[i]`; both are the same function of the input. We run
//! chunks of frames in parallel and carry each block's last input frame across
//! the seam, which is exactly the sequential path's `memory[i]` — results are
//! identical to decoding in one go (tested against [`super::taehv_ref`], a
//! plain transcription of the parallel path).
//!
//! `TGrow` with stride 2 turns one frame into two by splitting the channel
//! dimension of a 1x1 convolution, so the last stages run at the *output* frame
//! rate; `TPool` is its inverse on the encoder side.
//!
//! ## Latent space
//!
//! Both directions work in the **DiT-normalised** latent space: the reference
//! documents decode input as "~Gaussian", and sol-engine's refiner feeds the
//! stage-2 output straight in (`refiner_head_cp.py:567-570`) and de-normalises
//! the encoder output with the LTX VAE's per-channel statistics before the
//! latent upsampler (`:527-529`). Our LTX `normalize` is the same map
//! (`scaling_factor` 1.0).
//!
//! ## Precision
//!
//! Weights ship in f16; we hold and compute them in f32 (cuDNN, TF32 under
//! `--mode fast`). The reference refiner runs bf16 (`refiner_head_cp.py:429`),
//! so ours is the more precise of the two; parity checks here are against the
//! f32 reference transcription.

use super::tensor::{CudaTensor, Result, TensorError};
use super::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// `tanh(x/3) * 3`.
const CLAMP_SCALE: f32 = 3.0;
/// H3 encoder clips are 17 pixel frames = 5 latent tokens (`5 * t_upscale`).
const H3_CHUNK_LATENTS: usize = 5;
/// Trailing latent tokens the H3 encoder drops.
const H3_TOKEN_DROP: usize = 3;

/// Which checkpoint the sequential decoder was built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaeArch {
    /// Wan 2.1: 16 latent channels, 8× spatial, `4T − 3` frames.
    Wan,
    /// MiniMax-H3: 24 latent channels, 16× spatial (`8×` upsample + pixel-shuffle 2),
    /// then the 17-frame chunk wrap.
    H3,
    /// LTX-2.3/2.5 `taeltx2_3_wide`: 128 latent channels, 32× spatial (`8×`
    /// upsample + pixel-shuffle 4), `8T − 7` frames, wide decoder.
    LtxWide,
}

impl TaeArch {
    pub fn latent_channels(self) -> usize {
        match self {
            Self::Wan => 16,
            Self::H3 => 24,
            Self::LtxWide => 128,
        }
    }

    /// Pixel-(un)shuffle factor around the network (`ltx:taehv.py:204-205`,
    /// `h3:taehv.py:203-204`).
    pub fn patch_size(self) -> usize {
        match self {
            Self::Wan => 1,
            Self::H3 => 2,
            Self::LtxWide => 4,
        }
    }

    /// Decoder `n_f` (`ltx:taehv.py:213,222`).
    pub fn decoder_widths(self) -> [usize; 4] {
        match self {
            Self::Wan | Self::H3 => [256, 128, 64, 64],
            Self::LtxWide => [1024, 512, 256, 64],
        }
    }

    /// Whether the decoder's memblocks are `WideMemBlock`s.
    pub fn wide_decoder(self) -> bool {
        self == Self::LtxWide
    }

    /// TGrow strides per decoder stage. The default
    /// `decoder_time_upscale=(False, True, True)`; `taeltx` sets all three
    /// (`ltx:taehv.py:204-205`).
    pub fn decoder_time_upscale(self) -> [usize; 3] {
        match self {
            Self::Wan | Self::H3 => [1, 2, 2],
            Self::LtxWide => [2, 2, 2],
        }
    }

    /// TPool strides per encoder stage (`encoder_time_downscale`; the
    /// checkpoints agree: `encoder.12.conv` is `[64, 128]` for taeltx and
    /// `[64, 64]` for taeh3).
    pub fn encoder_time_downscale(self) -> [usize; 3] {
        match self {
            Self::Wan | Self::H3 => [2, 2, 1],
            Self::LtxWide => [2, 2, 2],
        }
    }

    /// Output frames per latent frame (`ltx:taehv.py:233`).
    pub fn t_upscale(self) -> usize {
        self.decoder_time_upscale().iter().product()
    }

    /// Input frames per latent frame (`ltx:taehv.py:232`).
    pub fn t_downscale(self) -> usize {
        self.encoder_time_downscale().iter().product()
    }

    /// Priming frames the decoder emits first (`ltx:taehv.py:234`).
    pub fn frames_to_trim(self) -> usize {
        self.t_upscale() - 1
    }

    /// Pixel size of one latent cell.
    pub fn spatial_scale(self) -> usize {
        8 * self.patch_size()
    }

    /// Whether this port runs the encoder for `self` (only where a reference
    /// pipeline uses it: the LTX refiner).
    pub fn has_encoder(self) -> bool {
        self == Self::LtxWide
    }

    /// Default latent frames per decode chunk. The wide LTX decoder's last
    /// stages run 8 frames per latent at 32× resolution (a 4K chunk of one
    /// latent is ~4 GB of activations), so it decodes one latent at a time.
    fn default_decode_chunk(self) -> usize {
        match self {
            Self::Wan | Self::H3 => 4,
            Self::LtxWide => 1,
        }
    }

    /// Guess from a weight file name, as `taehv.py`'s constructor does
    /// (`ltx:taehv.py:198-229`: substring tests on the checkpoint path).
    pub fn from_file_name(path: &std::path::Path) -> Option<Self> {
        let name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
        if name.contains("taeltx2_3_wide") {
            Some(Self::LtxWide)
        } else if name.contains("taeh3") {
            Some(Self::H3)
        } else if name.contains("taew2_1") {
            Some(Self::Wan)
        } else {
            None
        }
    }

    /// The weight file name this arch loads from a directory.
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Wan => "taew2_1.safetensors",
            Self::H3 => "taeh3.safetensors",
            Self::LtxWide => "taeltx2_3_wide.safetensors",
        }
    }
}

struct Conv {
    weight: CudaTensor,
    bias: Option<CudaTensor>,
    pad: usize,
    stride: usize,
    groups: usize,
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
        Self::load_ext(map, prefix, out_c, in_c, k, bias, 1, 1)
    }

    /// `in_c` is the full input width; the stored weight is `[out, in/groups, k, k]`.
    /// Weights are pinned to the device on a GPU run (a no-op on CPU), so a
    /// decode never re-uploads them.
    #[allow(clippy::too_many_arguments)]
    fn load_ext(
        map: &WeightMap,
        prefix: &str,
        out_c: usize,
        in_c: usize,
        k: usize,
        bias: bool,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let mut weight = super::weights::cuda_tensor_shaped(
            map,
            &format!("{prefix}.weight"),
            &[out_c, in_c / groups, k, k],
        )?;
        weight.pin_device()?;
        let bias = if bias {
            let mut b =
                super::weights::cuda_tensor_shaped(map, &format!("{prefix}.bias"), &[out_c])?;
            b.pin_device()?;
            Some(b)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            pad: k / 2,
            stride,
            groups,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        x.conv2d_groups(
            &self.weight,
            self.bias.as_ref(),
            self.pad,
            self.stride,
            self.groups,
        )
    }
}

/// `relu(conv(cat([x, past])) + x)` — the skip is the identity here because
/// every MemBlock in these checkpoints has `n_in == n_out` (`ltx:taehv.py:21-28`).
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

/// `WideMemBlock(n, n)` (`ltx:taehv.py:30-43`): a 1x1 squeeze of the
/// concatenated `[x, past]`, a grouped 3x3 (`groups = max(1, n // 64)`, so
/// 64 channels per group), a 1x1, another grouped 3x3 — ReLU between each —
/// then `relu(. + x)` with the identity skip (`n_in == n_out` throughout).
struct WideMemBlock {
    c0: Conv,
    c2: Conv,
    c4: Conv,
    c6: Conv,
}

impl WideMemBlock {
    fn load(map: &WeightMap, prefix: &str, n: usize) -> Result<Self> {
        let g = (n / 64).max(1);
        if !n.is_multiple_of(g) {
            return Err(msg(format!("WideMemBlock({n}): {n} % {g} != 0")));
        }
        Ok(Self {
            c0: Conv::load(map, &format!("{prefix}.conv.0"), n, n * 2, 1, true)?,
            c2: Conv::load_ext(map, &format!("{prefix}.conv.2"), n, n, 3, true, 1, g)?,
            c4: Conv::load(map, &format!("{prefix}.conv.4"), n, n, 1, true)?,
            c6: Conv::load_ext(map, &format!("{prefix}.conv.6"), n, n, 3, true, 1, g)?,
        })
    }

    fn forward(&self, x: &CudaTensor, past: &CudaTensor) -> Result<CudaTensor> {
        let h = CudaTensor::cat(&[x, past], 1)?;
        let h = relu(&self.c0.forward(&h)?);
        let h = relu(&self.c2.forward(&h)?);
        let h = relu(&self.c4.forward(&h)?);
        let h = self.c6.forward(&h)?;
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
    WideMem(WideMemBlock),
    Upsample2,
    /// 1x1 convolution to `c * stride` channels, then split into `stride` frames.
    TGrow {
        conv: Conv,
        stride: usize,
    },
    /// `stride` consecutive frames stacked on channels, then a 1x1
    /// convolution back to `c` (`ltx:taehv.py:45-52`).
    TPool {
        conv: Conv,
        stride: usize,
    },
}

pub struct TaeHv {
    arch: TaeArch,
    blocks: Vec<Block>,
    /// Loaded only for [`TaeArch::has_encoder`] archs.
    encoder: Option<Vec<Block>>,
}

impl TaeHv {
    /// Load `taew2_1` from a directory holding `taew2_1.safetensors` (or any
    /// `WeightMap` whose keys are the reference's positional `decoder.N...`).
    pub fn load(map: &WeightMap) -> Result<Self> {
        Self::load_arch(map, TaeArch::Wan)
    }

    /// The sequential decoder (and, for [`TaeArch::LtxWide`], the encoder)
    /// sized for `arch`. Keys stay `decoder.N...` / `encoder.N...` — a
    /// renumbering would load silently wrong weights.
    pub fn load_arch(map: &WeightMap, arch: TaeArch) -> Result<Self> {
        use Block::*;
        let nf = arch.decoder_widths();
        let tu = arch.decoder_time_upscale();
        let mem = |i: usize, n: usize| -> Result<Block> {
            let key = format!("decoder.{i}");
            Ok(if arch.wide_decoder() {
                WideMem(WideMemBlock::load(map, &key, n)?)
            } else {
                Mem(MemBlock::load(map, &key, n)?)
            })
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
            // The file's TGrow weight is always the stride it was trained
            // with; `patch_tgrow_layers` (`ltx:taehv.py:239-252`) only
            // matters when upscaling is disabled, which no caller here does.
            Ok(TGrow {
                conv: self::Conv::load(map, &format!("decoder.{i}.conv"), c * stride, c, 1, false)?,
                stride,
            })
        };
        let mut blocks = vec![Clamp, conv(1, nf[0], arch.latent_channels(), true)?, Relu];
        for s in 0..3 {
            let base = 3 + 6 * s;
            blocks.push(mem(base, nf[s])?);
            blocks.push(mem(base + 1, nf[s])?);
            blocks.push(mem(base + 2, nf[s])?);
            blocks.push(Upsample2);
            blocks.push(tgrow(base + 4, nf[s], tu[s])?);
            blocks.push(conv(base + 5, nf[s + 1], nf[s], false)?);
        }
        blocks.push(Relu);
        blocks.push(conv(
            22,
            3 * arch.patch_size() * arch.patch_size(),
            nf[3],
            true,
        )?);
        let encoder = if arch.has_encoder() {
            Some(Self::load_encoder(map, arch)?)
        } else {
            None
        };
        Ok(Self {
            arch,
            blocks,
            encoder,
        })
    }

    fn load_encoder(map: &WeightMap, arch: TaeArch) -> Result<Vec<Block>> {
        use Block::*;
        let key = |i: usize| format!("encoder.{i}");
        let cin = 3 * arch.patch_size() * arch.patch_size();
        let mut blocks = vec![
            Conv(self::Conv::load(map, &key(0), 64, cin, 3, true)?),
            Relu,
        ];
        for (s, &stride) in arch.encoder_time_downscale().iter().enumerate() {
            let base = 2 + 5 * s;
            blocks.push(TPool {
                conv: self::Conv::load(
                    map,
                    &format!("{}.conv", key(base)),
                    64,
                    64 * stride,
                    1,
                    false,
                )?,
                stride,
            });
            blocks.push(Conv(self::Conv::load_ext(
                map,
                &key(base + 1),
                64,
                64,
                3,
                false,
                2,
                1,
            )?));
            for i in base + 2..base + 5 {
                blocks.push(Mem(MemBlock::load(map, &key(i), 64)?));
            }
        }
        blocks.push(Conv(self::Conv::load(
            map,
            &key(17),
            arch.latent_channels(),
            64,
            3,
            true,
        )?));
        Ok(blocks)
    }

    /// `path` is a directory holding [`TaeArch::file_name`] (or any
    /// safetensors) or a single `.safetensors` file.
    pub fn load_from_path(path: &std::path::Path, arch: TaeArch) -> Result<Self> {
        let map = if path.is_file() {
            super::weights::WeightMap::open_files(&[path.to_path_buf()])?
        } else if path.join(arch.file_name()).is_file() {
            super::weights::WeightMap::open_files(&[path.join(arch.file_name())])?
        } else {
            super::weights::WeightMap::from_dir(path)?
        };
        Self::load_arch(&map, arch)
    }

    pub fn arch(&self) -> TaeArch {
        self.arch
    }

    /// Latents → video in `[-1, 1]`: `[1, 16, T, H, W]` → `[1, 3, 4T-3, 8H, 8W]`
    /// (Wan), `[1, 24, T, H, W]` → `[1, 3, 17n+5, 16H, 16W]` (H3),
    /// `[1, 128, T, H, W]` → `[1, 3, 8T-7, 32H, 32W]` (LTX).
    ///
    /// The reference emits `[0, 1]`; this returns `[-1, 1]` to match what the
    /// full VAE decode gives the frame writer, so the two are interchangeable.
    ///
    /// Takes latents in **DiT space** — the reference documents its input as
    /// "~Gaussian", which is the denoiser's own scale, *before* the per-channel
    /// `latents_mean`/`latents_std` un-normalisation that the full VAE expects.
    /// That difference is the easiest thing to get silently wrong here, so the
    /// oracle stage checks it rather than trusting this comment.
    pub fn decode(&self, z: &CudaTensor) -> Result<CudaTensor> {
        self.decode_streaming(z, &mut |_, _| Ok(()))
    }

    /// `decode`, handing each chunk of finished frames to `sink(frame_offset,
    /// frames)` as soon as it exists — `[frames, 3, H, W]` in `[-1, 1]`, in
    /// order — so PNG encoding and the mp4 mux can run while the GPU is still
    /// on the next chunk. Returns the whole video exactly as `decode` does.
    pub fn decode_streaming(
        &self,
        z: &CudaTensor,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<CudaTensor> {
        self.decode_streaming_chunked(z, self.decode_chunk_len(), sink)
    }

    /// [`Self::decode_streaming`] with the latent frames per chunk given
    /// (`FASTVIDEO_TAEHV_CHUNK` otherwise). Any value gives the same video.
    pub fn decode_streaming_chunked(
        &self,
        z: &CudaTensor,
        chunk: usize,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<CudaTensor> {
        let chunk = chunk.max(1);
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
            return self.decode_streaming_h3(z, [c, t, h, w], chunk, sink);
        }

        // Being parallel over frames is what makes TAEHV fast and what makes it
        // run out of memory: every stage materialises every frame, and the last
        // one is 4T frames of 64 channels at full resolution — ~12.6 GB for a
        // 129-frame clip, on top of a resident DiT. So decode in chunks of
        // latent frames, carrying each MemBlock's boundary frame across the
        // seam, which is exactly what the reference's sequential path does with
        // its `memory[i]`. Results are identical to decoding in one go.
        // One saved frame per block, at that block's own frame rate — the rate
        // differs after each TGrow, which is why this is indexed by block.
        let mut memory: Vec<Option<CudaTensor>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut out_chunks: Vec<CudaTensor> = Vec::new();
        // Only the very first output frames are priming frames
        // (`ltx:taehv.py:300` `x[:, frames_to_trim:]`); a chunk shorter than
        // the trim is dropped whole and the rest comes off the next.
        let mut to_trim = self.arch.frames_to_trim();
        let mut emitted = 0usize;

        let mut start = 0usize;
        while start < t {
            let len = chunk.min(t - start);
            let zc = z.narrow(2, start, len)?;
            let piece = pixel_shuffle(
                &run_blocks(
                    &self.blocks,
                    zc.reshape(vec![c, len, h, w])?.permute(&[1, 0, 2, 3])?,
                    &mut memory,
                )?,
                self.arch.patch_size(),
            )?;
            start += len;
            let f = piece.shape[0];
            let skip = to_trim.min(f);
            to_trim -= skip;
            if skip == f {
                continue;
            }
            let piece = if skip > 0 {
                piece.narrow(0, skip, f - skip)?
            } else {
                piece
            };
            // Clamp to the reference's [0, 1] (`ltx:taehv.py:280`), then map
            // to the [-1, 1] the rest of the pipeline uses. Per chunk, so the
            // sink sees finished frames; elementwise, so it is the same as
            // mapping after the cat.
            let piece = piece.clamp(0.0, 1.0).mul_scalar(2.0).add_scalar(-1.0);
            sink(emitted, &piece)?;
            emitted += piece.shape[0];
            out_chunks.push(piece);
        }
        if out_chunks.is_empty() {
            return Err(msg(format!(
                "taehv: {t} latent frames decode to no frames after the {}-frame trim",
                self.arch.frames_to_trim()
            )));
        }

        let refs: Vec<&CudaTensor> = out_chunks.iter().collect();
        let x = CudaTensor::cat(&refs, 0)?;
        let (frames, oc, oh, ow) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        x.reshape(vec![1, frames, oc, oh, ow])?
            .permute(&[0, 2, 1, 3, 4])
    }

    fn decode_chunk_len(&self) -> usize {
        super::envflag::usize_flag("FASTVIDEO_TAEHV_CHUNK", self.arch.default_decode_chunk()).max(1)
    }

    /// H3: grow `4T` frames, pixel-shuffle to 16× spatial, then the reference's
    /// 17-frame wrap (`h3:taehv.py:279-288` `_decode_h3_video`). The wrap reads
    /// the whole clip (pad to a multiple of 20, drop 3 prefix frames per group,
    /// drop the last 12), so frames reach the sink after the decoder finishes.
    fn decode_streaming_h3(
        &self,
        z: &CudaTensor,
        [c, t, h, w]: [usize; 4],
        chunk: usize,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<CudaTensor> {
        let mut memory: Vec<Option<CudaTensor>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut raw: Vec<CudaTensor> = Vec::new();
        let mut start = 0usize;
        while start < t {
            let len = chunk.min(t - start);
            let zc = z.narrow(2, start, len)?;
            let x = zc.reshape(vec![c, len, h, w])?.permute(&[1, 0, 2, 3])?;
            raw.push(pixel_shuffle(
                &run_blocks(&self.blocks, x, &mut memory)?,
                2,
            )?);
            start += len;
        }
        let refs: Vec<&CudaTensor> = raw.iter().collect();
        let mut frames = apply_h3_wrap(&CudaTensor::cat(&refs, 0)?)?;
        frames = frames.clamp(0.0, 1.0).mul_scalar(2.0).add_scalar(-1.0);
        // Hand the writer 17-frame groups so mux overlaps the last copy.
        let group = H3_CHUNK_LATENTS * TaeArch::H3.t_upscale() - TaeArch::H3.frames_to_trim();
        let mut emitted = 0usize;
        while emitted < frames.shape[0] {
            let n = (frames.shape[0] - emitted).min(group);
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

    /// Video `[1, 3, F, H, W]` in `[-1, 1]` → DiT-space latents
    /// `[1, C, ceil(F / t_down), H / (8p), W / (8p)]` (LTX: `[1, 128,
    /// ceil(F/8), H/32, W/32]`).
    ///
    /// sol-engine's refiner maps its `[-1, 1]` LTX pixels with
    /// `add(1).mul(0.5).clamp(0, 1)` (`refiner_head_cp.py:240-243`); `encode_video`
    /// then pads the end to a multiple of `t_downscale` by repeating the last
    /// frame (`ltx:taehv.py:268-274`) — 241 frames become 248, i.e. 31 latents,
    /// the refiner's `(1, 31, 128, 17, 30)` at 960x544 (`refiner_head_cp.py:522`).
    ///
    /// Streams `FASTVIDEO_TAEHV_ENC_CHUNK` input frames (rounded to a multiple
    /// of `t_downscale`, default `2 * t_downscale`) at a time with the same
    /// per-block carry as decode; every TPool then sees whole groups.
    pub fn encode(&self, video: &CudaTensor) -> Result<CudaTensor> {
        let td = self.arch.t_downscale();
        self.encode_chunked(
            video,
            super::envflag::usize_flag("FASTVIDEO_TAEHV_ENC_CHUNK", 2 * td),
        )
    }

    /// [`Self::encode`] with the input frames per chunk given (rounded up to
    /// a multiple of `t_downscale`). Any value gives the same latents.
    pub fn encode_chunked(&self, video: &CudaTensor, chunk: usize) -> Result<CudaTensor> {
        let Some(encoder) = self.encoder.as_ref() else {
            return Err(msg(format!(
                "taehv ({:?}): the encoder is not ported for this checkpoint",
                self.arch
            )));
        };
        let [n, c, f, h, w] = video.shape[..] else {
            return Err(msg(format!(
                "taehv encode expects [1, 3, F, H, W] video, got {:?}",
                video.shape
            )));
        };
        let scale = self.arch.spatial_scale();
        if n != 1 || c != 3 || f == 0 || h % scale != 0 || w % scale != 0 {
            return Err(msg(format!(
                "taehv ({:?}) encode expects [1, 3, F, H, W] with H, W multiples of {scale}, got {:?}",
                self.arch, video.shape
            )));
        }
        let td = self.arch.t_downscale();
        let chunk = chunk.div_ceil(td).max(1) * td;
        let frames = video
            .reshape(vec![3, f, h, w])?
            .permute(&[1, 0, 2, 3])?
            .add_scalar(1.0)
            .mul_scalar(0.5)
            .clamp(0.0, 1.0);
        let pad = (td - f % td) % td;
        let frames = if pad == 0 {
            frames
        } else {
            let last = frames.narrow(0, f - 1, 1)?;
            let mut parts: Vec<&CudaTensor> = vec![&frames];
            parts.extend(std::iter::repeat_n(&last, pad));
            CudaTensor::cat(&parts, 0)?
        };
        let total = f + pad;
        let mut memory: Vec<Option<CudaTensor>> = (0..encoder.len()).map(|_| None).collect();
        let mut out: Vec<CudaTensor> = Vec::new();
        let mut start = 0usize;
        while start < total {
            let len = chunk.min(total - start);
            let x = pixel_unshuffle(&frames.narrow(0, start, len)?, self.arch.patch_size())?;
            out.push(run_blocks(encoder, x, &mut memory)?);
            start += len;
        }
        let refs: Vec<&CudaTensor> = out.iter().collect();
        let z = CudaTensor::cat(&refs, 0)?;
        let (t, lc, lh, lw) = (z.shape[0], z.shape[1], z.shape[2], z.shape[3]);
        z.reshape(vec![1, t, lc, lh, lw])?.permute(&[0, 2, 1, 3, 4])
    }
}

/// One chunk of frames `[frames, c, h, w]` through every block, updating the
/// per-block boundary memory. The frame count changes at TGrow / TPool.
fn run_blocks(
    blocks: &[Block],
    mut x: CudaTensor,
    memory: &mut [Option<CudaTensor>],
) -> Result<CudaTensor> {
    let mut frames = x.shape[0];
    for (i, block) in blocks.iter().enumerate() {
        x = match block {
            Block::Clamp => tanh_scaled(&x, CLAMP_SCALE)?,
            Block::Relu => relu(&x),
            Block::Conv(cv) => cv.forward(&x)?,
            Block::Upsample2 => {
                let (hh, ww) = (x.shape[2], x.shape[3]);
                x.upsample_nearest2d(hh * 2, ww * 2)?
            }
            Block::Mem(_) | Block::WideMem(_) => {
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
                match block {
                    Block::Mem(m) => m.forward(&x, &past)?,
                    Block::WideMem(m) => m.forward(&x, &past)?,
                    _ => unreachable!(),
                }
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
            Block::TPool { conv, stride } => {
                if !frames.is_multiple_of(*stride) {
                    return Err(msg(format!(
                        "taehv TPool: {frames} frames in a chunk, stride {stride}"
                    )));
                }
                let (fc, hh, ww) = (x.shape[1], x.shape[2], x.shape[3]);
                frames /= stride;
                conv.forward(&x.reshape(vec![frames, fc * stride, hh, ww])?)?
            }
        };
    }
    Ok(x)
}

/// Pixel frames TAEH3 emits for `latent_frames` DiT tokens. For every
/// `5n + 2` latent count H3 requests, this is `17n + 5`.
pub fn h3_decoded_frames(latent_frames: usize) -> usize {
    h3_wrap_frame_count(latent_frames * TaeArch::H3.t_upscale())
}

/// After `4T` grown frames: pad T by `(-T) % 20` (Python unary-minus then
/// modulo — pad, not crop), drop the first 3 of each 20, drop the last 12.
fn h3_wrap_frame_count(raw_frames: usize) -> usize {
    let group = H3_CHUNK_LATENTS * TaeArch::H3.t_upscale();
    let pad = (group - raw_frames % group) % group;
    let groups = (raw_frames + pad) / group;
    groups * (group - TaeArch::H3.frames_to_trim()) - H3_TOKEN_DROP * TaeArch::H3.t_upscale()
}

/// `F.pixel_shuffle(x, r)` on `[F, C*r*r, H, W]` → `[F, C, rH, rW]`.
fn pixel_shuffle(x: &CudaTensor, r: usize) -> Result<CudaTensor> {
    if r == 1 {
        return Ok(x.clone());
    }
    let [f, crr, h, w] = x.shape[..] else {
        return Err(msg(format!(
            "pixel_shuffle expects [F, C, H, W], got {:?}",
            x.shape
        )));
    };
    if crr % (r * r) != 0 {
        return Err(msg(format!(
            "pixel_shuffle: {crr} channels is not {}× a channel count",
            r * r
        )));
    }
    let c = crr / (r * r);
    x.reshape(vec![f, c, r, r, h, w])?
        .permute(&[0, 1, 4, 2, 5, 3])?
        .reshape(vec![f, c, h * r, w * r])
}

/// `F.pixel_unshuffle(x, r)` on `[F, C, rH, rW]` → `[F, C*r*r, H, W]`.
fn pixel_unshuffle(x: &CudaTensor, r: usize) -> Result<CudaTensor> {
    if r == 1 {
        return Ok(x.clone());
    }
    let [f, c, hr, wr] = x.shape[..] else {
        return Err(msg(format!(
            "pixel_unshuffle expects [F, C, H, W], got {:?}",
            x.shape
        )));
    };
    if hr % r != 0 || wr % r != 0 {
        return Err(msg(format!(
            "pixel_unshuffle: {hr}x{wr} not a multiple of {r}"
        )));
    }
    let (h, w) = (hr / r, wr / r);
    x.reshape(vec![f, c, h, r, w, r])?
        .permute(&[0, 1, 3, 5, 2, 4])?
        .reshape(vec![f, c * r * r, h, w])
}

/// `h3:taehv.py:279-288` after the sequential decoder, before
/// `postprocess_output_frames`. `x` is `[F, C, H, W]`.
fn apply_h3_wrap(x: &CudaTensor) -> Result<CudaTensor> {
    let [frames, c, h, w] = x.shape[..] else {
        return Err(msg(format!(
            "h3 wrap expects [F, C, H, W], got {:?}",
            x.shape
        )));
    };
    let group = H3_CHUNK_LATENTS * TaeArch::H3.t_upscale();
    let trim = TaeArch::H3.frames_to_trim();
    let drop = H3_TOKEN_DROP * TaeArch::H3.t_upscale();
    let pad = (group - frames % group) % group;
    let x = if pad == 0 {
        x.clone()
    } else {
        let zeros = CudaTensor::zeros(&[pad, c, h, w]).to_device()?;
        CudaTensor::cat(&[x, &zeros], 0)?
    };
    let groups = x.shape[0] / group;
    let body = x
        .reshape(vec![groups, group, c, h, w])?
        .narrow(1, trim, group - trim)?;
    let kept = groups * (group - trim);
    let x = body.reshape(vec![kept, c, h, w])?;
    if kept <= drop {
        return Err(msg(format!(
            "h3 wrap: {frames} grown frames keep {kept}, cannot drop {drop}"
        )));
    }
    x.narrow(0, 0, kept - drop)
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
    fn pixel_shuffle_2_is_channel_to_spatial() {
        // Channel-major 2×2 tiles: values 0..11 in [1, 12, 1, 1] become
        // [1, 3, 2, 2] with each RGB channel a 2×2 of consecutive numbers.
        let x =
            CudaTensor::from_vec((0..12).map(|v| v as f32).collect(), vec![1, 12, 1, 1]).unwrap();
        let y = pixel_shuffle(&x, 2).unwrap();
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

    // ---- against the plain transcription of taehv.py ----------------------

    use super::super::taehv_ref as r;

    /// Uniform weights scaled by `sqrt(3 / fan_in)` (unit-variance outputs),
    /// deterministic per key: deep stacks of real-width blocks stay in a range
    /// where a wrong index shows up as a large error, not as saturation.
    fn scaled(key: &str, shape: &[usize]) -> Vec<f32> {
        let n: usize = shape.iter().product();
        let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
        let amp = if key.ends_with(".bias") {
            0.1
        } else {
            (3.0 / fan_in as f32).sqrt()
        };
        let seed = key.bytes().fold(0x9e37u64, |a, b| {
            a.wrapping_mul(131).wrapping_add(u64::from(b))
        });
        (0..n)
            .map(|i| {
                let x = seed
                    .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_mul(6364136223846793005)
                    >> 40;
                ((x % 20001) as f32 / 10000.0 - 1.0) * amp
            })
            .collect()
    }

    fn scaled_map() -> WeightMap {
        WeightMap::generated(scaled)
    }

    fn ws() -> impl Fn(&str, &[usize]) -> Vec<f32> {
        scaled
    }

    fn noise(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = seed
                    .wrapping_add(i as u64)
                    .wrapping_mul(0x2545_F491_4F6C_DD1D)
                    .rotate_left(17)
                    .wrapping_mul(6364136223846793005)
                    >> 40;
                lo + (hi - lo) * ((x % 100_000) as f32 / 100_000.0)
            })
            .collect()
    }

    fn act(shape: [usize; 4], seed: u64) -> (CudaTensor, r::Act) {
        let data = noise(shape.iter().product(), seed, -1.0, 1.0);
        (
            CudaTensor::from_vec(data.clone(), shape.to_vec()).unwrap(),
            r::Act::new(data, shape),
        )
    }

    fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
        assert_eq!(got.len(), want.len(), "length");
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (a, b) in got.iter().zip(want) {
            num += f64::from(a - b).powi(2);
            den += f64::from(*b).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    fn assert_close(what: &str, got: &CudaTensor, want: &r::Act) {
        assert_eq!(got.shape, want.shape.to_vec(), "{what}: shape");
        let e = rel_l2(&got.host_cow().unwrap(), &want.data);
        assert!(e < 1e-5, "{what}: rel_l2 {e}");
    }

    /// Our `[1, 3, F, H, W]` in [-1, 1] → the reference's `[F, 3, H, W]` in [0, 1].
    fn as_reference_frames(video: &CudaTensor) -> CudaTensor {
        let [_, c, f, h, w] = video.shape[..] else {
            panic!("video shape {:?}", video.shape)
        };
        video
            .reshape(vec![c, f, h, w])
            .unwrap()
            .permute(&[1, 0, 2, 3])
            .unwrap()
            .add_scalar(1.0)
            .mul_scalar(0.5)
    }

    #[test]
    fn memblock_matches_reference() {
        let map = scaled_map();
        for (key, n) in [("decoder.9", 128usize), ("encoder.4", 64)] {
            let m = MemBlock::load(&map, key, n).unwrap();
            let (x, xr) = act([3, n, 3, 5], 11);
            let past = shift_one_frame(&x, 3).unwrap();
            let got = m.forward(&x, &past).unwrap();
            let want = r::memblock(&ws(), key, n, &xr, &r::past_of(&xr));
            assert_close(key, &got, &want);
        }
    }

    /// Real widths of `taeltx2_3_wide` (16, 8 and 4 groups of 64).
    #[test]
    fn wide_memblock_matches_reference_at_every_width() {
        let map = scaled_map();
        for (key, n) in [
            ("decoder.3", 1024usize),
            ("decoder.10", 512),
            ("decoder.17", 256),
        ] {
            let m = WideMemBlock::load(&map, key, n).unwrap();
            let (x, xr) = act([2, n, 2, 3], 7 + n as u64);
            let past = shift_one_frame(&x, 2).unwrap();
            let got = m.forward(&x, &past).unwrap();
            let want = r::wide_memblock(&ws(), key, n, &xr, &r::past_of(&xr));
            assert_close(key, &got, &want);
        }
    }

    #[test]
    fn tgrow_and_tpool_match_reference() {
        let map = scaled_map();
        let grow = Block::TGrow {
            conv: Conv::load(&map, "decoder.7.conv", 2048, 1024, 1, false).unwrap(),
            stride: 2,
        };
        let (x, xr) = act([3, 1024, 2, 2], 3);
        let mut mem = vec![None];
        let got = run_blocks(std::slice::from_ref(&grow), x, &mut mem).unwrap();
        assert_close("tgrow", &got, &r::tgrow(&ws(), "decoder.7", 1024, 2, &xr));

        let pool = Block::TPool {
            conv: Conv::load(&map, "encoder.12.conv", 64, 128, 1, false).unwrap(),
            stride: 2,
        };
        let (x, xr) = act([4, 64, 3, 2], 5);
        let mut mem = vec![None];
        let got = run_blocks(std::slice::from_ref(&pool), x, &mut mem).unwrap();
        assert_close("tpool", &got, &r::tpool(&ws(), "encoder.12", 64, 2, &xr));
    }

    #[test]
    fn strided_and_grouped_convs_match_reference() {
        let map = scaled_map();
        let c = Conv::load_ext(&map, "encoder.3", 64, 64, 3, false, 2, 1).unwrap();
        let (x, xr) = act([2, 64, 5, 6], 9);
        let w = scaled("encoder.3.weight", &[64, 64, 3, 3]);
        assert_close(
            "stride-2 conv",
            &c.forward(&x).unwrap(),
            &r::conv2d(&xr, &w, 64, 3, None, 2, 1),
        );
        let c = Conv::load_ext(&map, "decoder.9.conv.2", 512, 512, 3, true, 1, 8).unwrap();
        let (x, xr) = act([1, 512, 3, 3], 13);
        let w = scaled("decoder.9.conv.2.weight", &[512, 64, 3, 3]);
        let b = scaled("decoder.9.conv.2.bias", &[512]);
        assert_close(
            "grouped conv",
            &c.forward(&x).unwrap(),
            &r::conv2d(&xr, &w, 512, 3, Some(&b), 1, 8),
        );
    }

    #[test]
    fn pixel_shuffle_4_and_unshuffle_match_reference() {
        let (x, xr) = act([2, 48, 3, 2], 21);
        assert_close(
            "shuffle",
            &pixel_shuffle(&x, 4).unwrap(),
            &r::pixel_shuffle(&xr, 4),
        );
        let (y, yr) = act([2, 3, 12, 8], 22);
        let un = pixel_unshuffle(&y, 4).unwrap();
        assert_close("unshuffle", &un, &r::pixel_unshuffle(&yr, 4));
        let back = pixel_shuffle(&un, 4).unwrap();
        assert_eq!(
            back.host_cow().unwrap().as_ref(),
            y.host_cow().unwrap().as_ref()
        );
    }

    fn check_decode(arch: TaeArch, latent: [usize; 3], chunks: &[usize]) {
        let tae = TaeHv::load_arch(&scaled_map(), arch).expect("load");
        let c = arch.latent_channels();
        let [t, h, w] = latent;
        let data = noise(c * t * h * w, 31, -2.0, 2.0);
        let z = CudaTensor::from_vec(data.clone(), vec![1, c, t, h, w]).unwrap();
        // Reference takes [T, C, H, W].
        let zr = CudaTensor::from_vec(data, vec![c, t, h, w])
            .unwrap()
            .permute(&[1, 0, 2, 3])
            .unwrap();
        let zr = r::Act::new(zr.host_cow().unwrap().into_owned(), [t, c, h, w]);
        let want = r::decode(arch, &ws(), &zr);
        for &chunk in chunks {
            let got = tae
                .decode_streaming_chunked(&z, chunk, &mut |_, _| Ok(()))
                .expect("decode");
            assert_close(
                &format!("{arch:?} decode, chunk {chunk}"),
                &as_reference_frames(&got),
                &want,
            );
        }
    }

    /// The whole wide decoder at real widths: every block, the stride-2
    /// TGrow at stage 0, pixel-shuffle 4, the 7-frame trim — and chunking by
    /// 1 (the default), 2 and all at once, which must all equal the
    /// reference's parallel pass.
    #[test]
    fn ltx_wide_decode_matches_reference() {
        check_decode(TaeArch::LtxWide, [3, 1, 2], &[1, 2, 64]);
    }

    #[test]
    fn wan_decode_matches_reference() {
        check_decode(TaeArch::Wan, [3, 2, 2], &[1, 4]);
    }

    #[test]
    fn h3_decode_matches_reference() {
        check_decode(TaeArch::H3, [7, 1, 2], &[2, 64]);
    }

    /// The encoder end to end: [-1, 1] → [0, 1] mapping, last-frame padding
    /// to a multiple of 8, pixel-unshuffle 4, three TPool/stride-2 stages —
    /// in 8-, 16- and whole-clip chunks against the reference's parallel pass.
    #[test]
    fn ltx_encode_matches_reference() {
        let tae = TaeHv::load_arch(&scaled_map(), TaeArch::LtxWide).expect("load");
        let (f, h, w) = (17usize, 32usize, 64usize);
        let data = noise(3 * f * h * w, 41, -1.1, 1.1);
        let video = CudaTensor::from_vec(data.clone(), vec![1, 3, f, h, w]).unwrap();
        let frames = as_reference_frames(&video).clamp(0.0, 1.0);
        let pr = r::Act::new(frames.host_cow().unwrap().into_owned(), [f, 3, h, w]);
        let want = r::encode(TaeArch::LtxWide, &ws(), &pr);
        assert_eq!(want.shape, [3, 128, 1, 2]);
        for chunk in [8usize, 16, 1000] {
            let got = tae.encode_chunked(&video, chunk).expect("encode");
            assert_eq!(got.shape, vec![1, 128, 3, 1, 2]);
            let got = got
                .reshape(vec![128, 3, 1, 2])
                .unwrap()
                .permute(&[1, 0, 2, 3])
                .unwrap();
            assert_close(&format!("ltx encode, chunk {chunk}"), &got, &want);
        }
    }

    /// LTX's `8n + 1` frames at 32× spatial, from the 7-frame trim of 8T.
    #[test]
    fn ltx_frame_counts_are_eight_n_plus_one() {
        let tae = TaeHv::load_arch(&tiny_map(), TaeArch::LtxWide).expect("load");
        for (t, want) in [(1usize, 1usize), (2, 9), (4, 25)] {
            let out = tae
                .decode(&CudaTensor::zeros(&[1, 128, t, 1, 1]))
                .expect("decode");
            assert_eq!(out.shape, vec![1, 3, want, 32, 32], "T={t}");
        }
        // The refiner's 241 frames: padded to 248, 31 latents.
        assert_eq!(241usize.div_ceil(TaeArch::LtxWide.t_downscale()), 31);
    }

    #[test]
    fn encoder_is_only_loaded_where_a_reference_uses_it() {
        let wan = TaeHv::load(&tiny_map()).expect("load");
        assert!(wan.encode(&CudaTensor::zeros(&[1, 3, 4, 8, 8])).is_err());
        let ltx = TaeHv::load_arch(&tiny_map(), TaeArch::LtxWide).expect("load");
        assert!(
            ltx.encode(&CudaTensor::zeros(&[1, 3, 8, 30, 32])).is_err(),
            "H not a multiple of 32"
        );
    }

    /// The real checkpoints load at the shapes this port declares, and the
    /// wide decoder + encoder agree with the reference on them. Runs only when
    /// `FASTVIDEO_TAE_TEST_DIR` holds `taeh3.safetensors` /
    /// `taeltx2_3_wide.safetensors` (`scripts/gpu/fetch-tae.sh`).
    #[test]
    fn real_checkpoints_match_reference_when_present() {
        let Some(dir) = std::env::var_os("FASTVIDEO_TAE_TEST_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        for arch in [TaeArch::H3, TaeArch::LtxWide] {
            let file = dir.join(arch.file_name());
            if !file.is_file() {
                continue;
            }
            let map = WeightMap::open_files(std::slice::from_ref(&file)).expect("open");
            let tae = TaeHv::load_arch(&map, arch).expect("load real weights");
            let ws = |k: &str, s: &[usize]| -> Vec<f32> {
                crate::wan::weights::cuda_tensor_shaped(&map, k, s)
                    .unwrap()
                    .host_cow()
                    .unwrap()
                    .into_owned()
            };
            let c = arch.latent_channels();
            let (t, h, w) = (2usize, 1usize, 2usize);
            let data = noise(c * t * h * w, 77, -2.0, 2.0);
            let z = CudaTensor::from_vec(data.clone(), vec![1, c, t, h, w]).unwrap();
            let zr = CudaTensor::from_vec(data, vec![c, t, h, w])
                .unwrap()
                .permute(&[1, 0, 2, 3])
                .unwrap();
            let want = r::decode(
                arch,
                &ws,
                &r::Act::new(zr.host_cow().unwrap().into_owned(), [t, c, h, w]),
            );
            let got = tae
                .decode_streaming_chunked(&z, 1, &mut |_, _| Ok(()))
                .unwrap();
            assert_close(
                &format!("{arch:?} real decode"),
                &as_reference_frames(&got),
                &want,
            );
            if arch.has_encoder() {
                let video = CudaTensor::from_vec(
                    noise(3 * 9 * 32 * 64, 5, -1.0, 1.0),
                    vec![1, 3, 9, 32, 64],
                )
                .unwrap();
                let frames = as_reference_frames(&video).clamp(0.0, 1.0);
                let want = r::encode(
                    arch,
                    &ws,
                    &r::Act::new(frames.host_cow().unwrap().into_owned(), [9, 3, 32, 64]),
                );
                let got = tae.encode_chunked(&video, 8).unwrap();
                let got = got
                    .reshape(vec![c, 2, 1, 2])
                    .unwrap()
                    .permute(&[1, 0, 2, 3])
                    .unwrap();
                assert_close("ltx real encode", &got, &want);
            }
        }
    }

    #[test]
    fn arch_from_file_name() {
        use std::path::Path;
        assert_eq!(
            TaeArch::from_file_name(Path::new("/x/taeltx2_3_wide.safetensors")),
            Some(TaeArch::LtxWide)
        );
        assert_eq!(
            TaeArch::from_file_name(Path::new("taeh3.safetensors")),
            Some(TaeArch::H3)
        );
        assert_eq!(
            TaeArch::from_file_name(Path::new("taeltx_2.safetensors")),
            None
        );
        assert_eq!(TaeArch::LtxWide.t_upscale(), 8);
        assert_eq!(TaeArch::LtxWide.frames_to_trim(), 7);
        assert_eq!(TaeArch::Wan.t_upscale(), 4);
    }
}
