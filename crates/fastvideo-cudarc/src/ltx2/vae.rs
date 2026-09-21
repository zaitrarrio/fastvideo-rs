//! `AutoencoderKLLTX2Video`, decoder half: `[1, 128, F, H, W]` latents →
//! `[1, 3, 8(F-1)+1, 32H, 32W]` video in (nominally) `[-1, 1]`.
//!
//! Every learned layer is a 3×3×3 convolution. Around them:
//!
//! * **padding** — the decoder is *non-causal*: time is padded with one copy of
//!   the first frame in front and one of the last frame behind; space is padded
//!   by reflection (`x[-1] = x[1]`). The backend's conv3d pads with zeros only,
//!   so both are done with `pad` and the conv runs unpadded.
//! * **norm** — `PerChannelRMSNorm`: RMS over the channel axis at every
//!   (t, h, w), no weight, eps 1e-8 (not the config's `resnet_norm_eps`).
//! * **upsampling** — a conv to `4·C` channels, a (2, 2, 2) depth-to-space to
//!   `C/2`, and the *first output frame dropped* (`F → 2F - 1`), plus a residual
//!   that is the same depth-to-space of the input itself (→ `C/8` channels),
//!   channel-tiled ×4. The upsampler runs *before* its block's resnets.
//! * **unpatchify** — a 4×4 spatial depth-to-space whose axis pairing is the
//!   reverse of the upsampler's: the more significant patch index goes to
//!   *width*.
//!
//! There is no timestep path and no noise injection in the LTX-2.0 config.
//!
//! # Streaming, exactly
//!
//! diffusers' own tiling (spatial or temporal) blends overlapping decodes and
//! is an approximation: with replicate padding at every tile edge and a frame
//! dropped per stage, a tile is not a window of the full result. The receptive
//! field does not rescue input-side chunking either — 41 convs deep it spans
//! about 22 latent frames, more than a 16-frame clip has.
//!
//! What *is* exact is streaming each convolution. Apart from the convs'
//! temporal kernel every op here is per-frame, and a radius-1 temporal conv
//! needs only a two-frame memory: [`TemporalConv`] keeps the last two frames it
//! saw, emits each output frame as soon as its right neighbour arrives, and
//! replicate-pads only at the true ends of the clip. Skip paths buffer their
//! frames until the delayed main path catches up. The result is the same sum
//! of the same products as the one-shot decode, in a different order of
//! arrival. The low-resolution stages are cheap and run whole; the last
//! up-block and the output head — where a full-clip activation is 1.4 GB and
//! there are a dozen of them — run in chunks of `FASTVIDEO_LTX2_VAE_CHUNK`
//! frames (default 8, giving 16 output frames per chunk), and finished frames
//! reach the sink while later ones are still being computed.
//! See docs/ports/ltx2.md §c.

use fastvideo_models::ltx2::config::Ltx2VideoVaeConfig;

use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::{msg, ones};

/// A 3×3×3 conv: reflect- or zero-padded in space, streamed in time.
struct TemporalConv {
    weight: CudaTensor,
    bias: CudaTensor,
    spatial_pad: PadMode,
}

/// What a [`TemporalConv`] remembers between chunks: the last two frames of its
/// input stream (the first chunk's left replicate included).
#[derive(Default)]
struct ConvState {
    carry: Option<CudaTensor>,
}

fn frames(x: &CudaTensor) -> usize {
    x.shape[2]
}

fn cat_time(parts: &[&CudaTensor]) -> Result<CudaTensor> {
    CudaTensor::cat(parts, 2)
}

impl TemporalConv {
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize, spatial_pad: PadMode) -> Result<Self> {
        let mut weight = cuda_tensor_shaped(map, &format!("{prefix}.conv.weight"), &[cout, cin, 3, 3, 3])?;
        let mut bias = cuda_tensor_shaped(map, &format!("{prefix}.conv.bias"), &[cout])?;
        weight.pin_device()?;
        bias.pin_device()?;
        Ok(Self { weight, bias, spatial_pad })
    }

    /// Feed the next frames of the stream (`None`: no new frames), `last` when
    /// the stream ends with them. Returns the output frames that became
    /// computable: output `t` needs inputs `t-1, t, t+1`, so everything but the
    /// newest frame — and that one too once `last` replicates it.
    fn push(&self, state: &mut ConvState, x: Option<CudaTensor>, last: bool) -> Result<Option<CudaTensor>> {
        let seq = match (state.carry.take(), x) {
            (Some(carry), Some(x)) => cat_time(&[&carry, &x])?,
            (Some(carry), None) => carry,
            // Start of the stream: one replicate of the first frame in front.
            (None, Some(x)) => cat_time(&[&x.narrow(2, 0, 1)?, &x])?,
            (None, None) => return Ok(None),
        };
        let n = frames(&seq);
        let seq = if last { cat_time(&[&seq, &seq.narrow(2, n - 1, 1)?])? } else { seq };
        let n = frames(&seq);
        if !last {
            state.carry = Some(seq.narrow(2, n - 2, 2)?);
        }
        if n < 3 {
            return Ok(None);
        }
        let padded = seq.pad(3, 1, 1, self.spatial_pad)?.pad(4, 1, 1, self.spatial_pad)?;
        drop(seq);
        padded.conv3d(&self.weight, Some(&self.bias), [0, 0, 0], [1, 1, 1]).map(Some)
    }
}

/// Frames waiting on a skip path for the delayed main path to catch up.
#[derive(Default)]
struct Fifo {
    pending: Option<CudaTensor>,
}

impl Fifo {
    fn push(&mut self, x: Option<&CudaTensor>) -> Result<()> {
        self.pending = match (self.pending.take(), x) {
            (Some(p), Some(x)) => Some(cat_time(&[&p, x])?),
            (Some(p), None) => Some(p),
            (None, x) => x.cloned(),
        };
        Ok(())
    }

    fn pop(&mut self, n: usize) -> Result<CudaTensor> {
        let p = self.pending.take().ok_or_else(|| msg("ltx2 vae: skip path ran dry"))?;
        let have = frames(&p);
        if n > have {
            return Err(msg(format!("ltx2 vae: main path produced {n} frames, skip path holds {have}")));
        }
        if n < have {
            self.pending = Some(p.narrow(2, n, have - n)?);
        }
        p.narrow(2, 0, n)
    }
}

struct Resnet {
    conv1: TemporalConv,
    conv2: TemporalConv,
    ones: CudaTensor,
}

#[derive(Default)]
struct ResnetState {
    conv1: ConvState,
    conv2: ConvState,
    skip: Fifo,
}

impl Resnet {
    fn load(map: &WeightMap, prefix: &str, ch: usize, spatial_pad: PadMode) -> Result<Self> {
        Ok(Self {
            conv1: TemporalConv::load(map, &format!("{prefix}.conv1"), ch, ch, spatial_pad)?,
            conv2: TemporalConv::load(map, &format!("{prefix}.conv2"), ch, ch, spatial_pad)?,
            ones: ones(ch)?,
        })
    }

    fn norm_act(&self, x: Option<CudaTensor>, eps: f32) -> Result<Option<CudaTensor>> {
        x.map(|x| x.rms_norm_channels_act(&self.ones, eps, true)).transpose()
    }

    fn push(&self, state: &mut ResnetState, x: Option<CudaTensor>, last: bool, eps: f32) -> Result<Option<CudaTensor>> {
        state.skip.push(x.as_ref())?;
        let h = self.conv1.push(&mut state.conv1, self.norm_act(x, eps)?, last)?;
        let h = self.conv2.push(&mut state.conv2, self.norm_act(h, eps)?, last)?;
        h.map(|h| state.skip.pop(frames(&h))?.add(&h)).transpose()
    }
}

/// `[1, 8C, f, h, w]` → `[1, C, 2f, 2h, 2w]` — spatiotemporal (2, 2, 2).
fn depth_to_space_spatiotemporal(y: &CudaTensor) -> Result<CudaTensor> {
    let [b, c8, f, h, w] = y.shape[..] else {
        return Err(msg(format!("depth_to_space expects [1, 8C, F, H, W], got {:?}", y.shape)));
    };
    if b != 1 || c8 % 8 != 0 {
        return Err(msg(format!("depth_to_space expects [1, 8C, F, H, W], got {:?}", y.shape)));
    }
    let c = c8 / 8;
    let spatial = y.reshape(vec![c * 2, 2, 2, f, h, w])?.permute(&[0, 3, 4, 1, 5, 2])?.reshape(vec![c, 2, f, 4 * h * w])?;
    spatial.permute(&[0, 2, 1, 3])?.reshape(vec![1, c, 2 * f, 2 * h, 2 * w])
}

/// `[1, C·st·sh·sw, f, h, w]` → `[1, C, st·f, sh·h, sw·w]`.
fn depth_to_space(y: &CudaTensor, stride: (usize, usize, usize)) -> Result<CudaTensor> {
    let (st, sh, sw) = stride;
    if (st, sh, sw) == (2, 2, 2) {
        return depth_to_space_spatiotemporal(y);
    }
    let prod = st * sh * sw;
    let [b, cprod, f, h, w] = y.shape[..] else {
        return Err(msg(format!("depth_to_space expects [1, C·stride, F, H, W], got {:?}", y.shape)));
    };
    if b != 1 || cprod % prod != 0 {
        return Err(msg(format!("depth_to_space expects [1, C·stride, F, H, W], got {:?}", y.shape)));
    }
    let c = cprod / prod;
    match (st, sh, sw) {
        (2, 1, 1) => y.reshape(vec![c, 2, f, h, w])?.permute(&[0, 2, 1, 3, 4])?.reshape(vec![1, c, 2 * f, h, w]),
        (1, 2, 2) => {
            let spatial = y.reshape(vec![c * 2, 2, 2, f, h, w])?.permute(&[0, 3, 4, 1, 5, 2])?.reshape(vec![c, 2, f, 4 * h * w])?;
            spatial.permute(&[0, 2, 1, 3])?.reshape(vec![1, c, f, 2 * h, 2 * w])
        }
        _ => Err(msg(format!("depth_to_space: unsupported stride ({st}, {sh}, {sw})"))),
    }
}

struct Upsampler {
    conv: TemporalConv,
    stride: (usize, usize, usize),
    residual: bool,
    drop_first_frame: bool,
}

#[derive(Default)]
struct UpsamplerState {
    conv: ConvState,
    skip: Fifo,
    dropped_first: bool,
}

impl Upsampler {
    fn push(&self, state: &mut UpsamplerState, x: Option<CudaTensor>, last: bool) -> Result<Option<CudaTensor>> {
        let skip_in = if self.residual { x.as_ref() } else { None };
        state.skip.push(skip_in)?;
        let Some(y) = self.conv.push(&mut state.conv, x, last)? else { return Ok(None) };
        let mut out = depth_to_space(&y, self.stride)?;
        if self.residual {
            let residual = depth_to_space(&state.skip.pop(frames(&y))?, self.stride)?;
            let main_ch = out.shape[1];
            let res_ch = residual.shape[1];
            if !main_ch.is_multiple_of(res_ch) {
                return Err(msg(format!("ltx2 vae upsampler: {main_ch} channels vs residual {res_ch}")));
            }
            let tiles = main_ch / res_ch;
            let tiled = CudaTensor::cat(&(0..tiles).map(|_| &residual).collect::<Vec<_>>(), 1)?;
            out = out.add(&tiled)?;
        }
        if !self.drop_first_frame {
            return Ok(Some(out));
        }
        if state.dropped_first {
            return Ok(Some(out));
        }
        state.dropped_first = true;
        let n = frames(&out);
        if n == 1 {
            return Ok(None);
        }
        out.narrow(2, 1, n - 1).map(Some)
    }
}

struct Block {
    upsampler: Option<Upsampler>,
    resnets: Vec<Resnet>,
}

#[derive(Default)]
struct BlockState {
    upsampler: UpsamplerState,
    resnets: Vec<ResnetState>,
}

impl Block {
    fn push(&self, state: &mut BlockState, x: Option<CudaTensor>, last: bool, eps: f32) -> Result<Option<CudaTensor>> {
        if state.resnets.is_empty() {
            state.resnets = self.resnets.iter().map(|_| ResnetState::default()).collect();
        }
        let mut x = match &self.upsampler {
            Some(up) => up.push(&mut state.upsampler, x, last)?,
            None => x,
        };
        for (resnet, st) in self.resnets.iter().zip(&mut state.resnets) {
            x = resnet.push(st, x, last, eps)?;
        }
        Ok(x)
    }

    /// The whole clip in one go: a stream of one chunk.
    fn forward(&self, x: CudaTensor, eps: f32) -> Result<CudaTensor> {
        self.push(&mut BlockState::default(), Some(x), true, eps)?.ok_or_else(|| msg("ltx2 vae: a block produced no frames"))
    }
}

pub struct VideoDecoder {
    cfg: Ltx2VideoVaeConfig,
    /// `[1, C, 1, 1, 1]` each.
    latents_mean: CudaTensor,
    latents_std: CudaTensor,
    conv_in: TemporalConv,
    /// `mid_block`, then `up_blocks.0 …`.
    blocks: Vec<Block>,
    ones_out: CudaTensor,
    conv_out: TemporalConv,
}

impl VideoDecoder {
    /// `map` is the diffusers `vae/` folder.
    pub fn load(map: &WeightMap, cfg: &Ltx2VideoVaeConfig) -> Result<Self> {
        if cfg.timestep_conditioning || cfg.decoder_causal || cfg.patch_size_t != 1 {
            return Err(msg("ltx2 vae: timestep conditioning, causal decode, or patch_size_t != 1 are not supported"));
        }
        if cfg.decoder_inject_noise.iter().any(|&n| n) {
            return Err(msg("ltx2 vae: decoder noise injection is not supported"));
        }
        if cfg.upsample_type.len() != cfg.upsample_factor.len() || cfg.upsample_residual.len() != cfg.upsample_factor.len() {
            return Err(msg("ltx2 vae: upsample_type / upsample_residual length must match upsample_factor"));
        }
        let spatial_pad = if cfg.decoder_reflect_padding { PadMode::Reflect } else { PadMode::Zeros };
        let stages = cfg.decoder_stages();
        let mut blocks = Vec::with_capacity(stages.len());
        let mut cin = stages[0].channels;
        for (i, stage) in stages.iter().enumerate() {
            let prefix = if i == 0 { "decoder.mid_block".to_string() } else { format!("decoder.up_blocks.{}", i - 1) };
            let upsampler = match &stage.upsampler {
                Some(up) => {
                    if up.in_channels != cin {
                        return Err(msg(format!("ltx2 vae stage {i}: upsampler expects {} in, block has {cin}", up.in_channels)));
                    }
                    Some(Upsampler {
                        conv: TemporalConv::load(map, &format!("{prefix}.upsamplers.0.conv"), up.in_channels, up.conv_out_channels, spatial_pad)?,
                        stride: up.stride,
                        residual: up.residual,
                        drop_first_frame: up.drop_first_frame,
                    })
                }
                None => None,
            };
            let resnets = (0..stage.resnet_layers)
                .map(|n| Resnet::load(map, &format!("{prefix}.resnets.{n}"), stage.channels, spatial_pad))
                .collect::<Result<Vec<_>>>()?;
            blocks.push(Block { upsampler, resnets });
            cin = stage.channels;
        }
        let z = cfg.latent_channels;
        let stat = |key: &str| -> Result<CudaTensor> {
            let mut t = cuda_tensor_shaped(map, key, &[z])?.reshape(vec![1, z, 1, 1, 1])?;
            t.pin_device()?;
            Ok(t)
        };
        Ok(Self {
            latents_mean: stat("latents_mean")?,
            latents_std: stat("latents_std")?,
            conv_in: TemporalConv::load(map, "decoder.conv_in", z, stages[0].channels, spatial_pad)?,
            blocks,
            ones_out: ones(cin)?,
            conv_out: TemporalConv::load(map, "decoder.conv_out", cin, cfg.out_channels * cfg.patch_size * cfg.patch_size, spatial_pad)?,
            cfg: cfg.clone(),
        })
    }

    /// `out[c, f, p·h + b, p·w + a] = y[c·p² + a·p + b, f, h, w]` — the more
    /// significant patch index lands on *width*.
    fn unpatchify(&self, y: &CudaTensor) -> Result<CudaTensor> {
        let (p, c) = (self.cfg.patch_size, self.cfg.out_channels);
        let [_, _, f, h, w] = y.shape[..] else { return Err(msg("ltx2 vae: unpatchify expects rank 5")) };
        // [c, a, b, f, h, w] → [c, f, h, b, w, a].
        y.reshape(vec![c, p, p, f, h, w])?.permute(&[0, 3, 4, 2, 5, 1])?.reshape(vec![1, c, f, p * h, p * w])
    }

    /// DiT-space (normalised) latents `[1, C, F, H, W]` → video
    /// `[1, 3, 8(F-1)+1, 32H, 32W]`, handing each run of finished frames to
    /// `sink(frame_offset, frames)` as `[frames, 3, H, W]`, in order, while the
    /// rest is still being decoded. The values are the decoder's own — nominally
    /// `[-1, 1]`, not clamped. Returns the number of frames produced.
    pub fn decode_streaming(&self, latents: &CudaTensor, sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>) -> Result<usize> {
        let chunk = crate::wan::envflag::usize_flag("FASTVIDEO_LTX2_VAE_CHUNK", 8);
        self.decode_streaming_chunked(latents, chunk, sink)
    }

    /// [`Self::decode_streaming`] with the chunk size — frames of the last
    /// up-block's input per step — given explicitly. Any value gives the same
    /// video; it trades peak memory against launch overhead.
    pub fn decode_streaming_chunked(
        &self,
        latents: &CudaTensor,
        chunk: usize,
        sink: &mut dyn FnMut(usize, &CudaTensor) -> Result<()>,
    ) -> Result<usize> {
        let [b, c, f, _, _] = latents.shape[..] else {
            return Err(msg(format!("ltx2 vae expects [1, C, F, H, W] latents, got {:?}", latents.shape)));
        };
        if b != 1 || c != self.cfg.latent_channels || f == 0 || latents.shape[3] < 2 || latents.shape[4] < 2 {
            return Err(msg(format!("ltx2 vae expects [1, {}, F, H>=2, W>=2] latents, got {:?}", self.cfg.latent_channels, latents.shape)));
        }
        let eps = self.cfg.pixel_norm_eps as f32;
        let z = latents.mul(&self.latents_std)?.mul_scalar(1.0 / self.cfg.scaling_factor as f32).add(&self.latents_mean)?;
        let mut x = self.conv_in.push(&mut ConvState::default(), Some(z), true)?.ok_or_else(|| msg("ltx2 vae: conv_in produced no frames"))?;
        let (last_block, early) = self.blocks.split_last().ok_or_else(|| msg("ltx2 vae: no blocks"))?;
        for block in early {
            x = block.forward(x, eps)?;
        }

        let chunk = chunk.max(1);
        let total = frames(&x);
        let (mut state, mut head) = (BlockState::default(), ConvState::default());
        let (mut start, mut emitted) = (0usize, 0usize);
        while start < total {
            let len = chunk.min(total - start);
            let last = start + len == total;
            let piece = x.narrow(2, start, len)?;
            start += len;
            let h = last_block.push(&mut state, Some(piece), last, eps)?;
            let h = h.map(|h| h.rms_norm_channels_act(&self.ones_out, eps, true)).transpose()?;
            let Some(y) = self.conv_out.push(&mut head, h, last)? else { continue };
            let video = self.unpatchify(&y)?;
            let (n, hh, ww) = (frames(&video), video.shape[3], video.shape[4]);
            // [1, 3, n, H, W] → [n, 3, H, W].
            let by_frame = video.reshape(vec![self.cfg.out_channels, n, hh, ww])?.permute(&[1, 0, 2, 3])?;
            sink(emitted, &by_frame)?;
            emitted += n;
        }
        let want = self.cfg.decoded_frames(f);
        if emitted != want {
            return Err(msg(format!("ltx2 vae: streamed {emitted} frames, {f} latent frames decode to {want}")));
        }
        Ok(emitted)
    }

    /// The whole video as `[1, 3, F', H', W']`.
    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let mut pieces: Vec<CudaTensor> = Vec::new();
        self.decode_streaming(latents, &mut |_, frames| {
            pieces.push(frames.clone());
            Ok(())
        })?;
        let all = CudaTensor::cat(&pieces.iter().collect::<Vec<_>>(), 0)?;
        let (n, c, h, w) = (all.shape[0], all.shape[1], all.shape[2], all.shape[3]);
        all.permute(&[1, 0, 2, 3])?.reshape(vec![1, c, n, h, w])
    }
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::{get, weights};
    use super::*;

    /// `[c, f, h, w]` volume for the loop reference.
    #[derive(Clone)]
    struct Vol {
        c: usize,
        f: usize,
        h: usize,
        w: usize,
        v: Vec<f32>,
    }

    impl Vol {
        fn zeros(c: usize, f: usize, h: usize, w: usize) -> Self {
            Self { c, f, h, w, v: vec![0.0; c * f * h * w] }
        }
        fn idx(&self, c: usize, f: usize, h: usize, w: usize) -> usize {
            ((c * self.f + f) * self.h + h) * self.w + w
        }
        /// Replicate in time, reflect (without repeating the edge) in space.
        fn padded(&self, c: usize, f: isize, h: isize, w: isize) -> f32 {
            let reflect = |i: isize, n: usize| -> usize {
                if i < 0 {
                    (-i) as usize
                } else if i as usize >= n {
                    2 * n - 2 - i as usize
                } else {
                    i as usize
                }
            };
            self.v[self.idx(c, f.clamp(0, self.f as isize - 1) as usize, reflect(h, self.h), reflect(w, self.w))]
        }
    }

    fn conv3(map: &WeightMap, prefix: &str, x: &Vol, cout: usize) -> Vol {
        let w = get(map, &format!("{prefix}.conv.weight"), &[cout, x.c, 3, 3, 3]);
        let b = get(map, &format!("{prefix}.conv.bias"), &[cout]);
        let mut out = Vol::zeros(cout, x.f, x.h, x.w);
        for o in 0..cout {
            for f in 0..x.f {
                for h in 0..x.h {
                    for ww in 0..x.w {
                        let mut acc = b[o];
                        for i in 0..x.c {
                            for (k, wk) in w[(o * x.c + i) * 27..(o * x.c + i + 1) * 27].iter().enumerate() {
                                let (kf, kh, kw) = ((k / 9) as isize - 1, (k / 3 % 3) as isize - 1, (k % 3) as isize - 1);
                                acc += wk * x.padded(i, f as isize + kf, h as isize + kh, ww as isize + kw);
                            }
                        }
                        let at = out.idx(o, f, h, ww);
                        out.v[at] = acc;
                    }
                }
            }
        }
        out
    }

    fn norm_silu(x: &Vol) -> Vol {
        let mut out = x.clone();
        let cells = x.f * x.h * x.w;
        for p in 0..cells {
            let ms = (0..x.c).map(|c| x.v[c * cells + p].powi(2)).sum::<f32>() / x.c as f32;
            for c in 0..x.c {
                let n = x.v[c * cells + p] / (ms + 1e-8).sqrt();
                out.v[c * cells + p] = n / (1.0 + (-n).exp());
            }
        }
        out
    }

    fn resnet(map: &WeightMap, prefix: &str, x: &Vol) -> Vol {
        let h = conv3(map, &format!("{prefix}.conv1"), &norm_silu(x), x.c);
        let h = conv3(map, &format!("{prefix}.conv2"), &norm_silu(&h), x.c);
        Vol { v: x.v.iter().zip(&h.v).map(|(a, b)| a + b).collect(), ..h }
    }

    /// The spec's index map, literally.
    fn d2s(y: &Vol, stride: (usize, usize, usize)) -> Vol {
        let (st, sh, sw) = stride;
        let prod = st * sh * sw;
        let mut out = Vol::zeros(y.c / prod, st * y.f, sh * y.h, sw * y.w);
        for c in 0..out.c {
            for f in 0..y.f {
                for h in 0..y.h {
                    for w in 0..y.w {
                        for i in 0..st {
                            for j in 0..sh {
                                for k in 0..sw {
                                    let ijk = i * (sh * sw) + j * sw + k;
                                    let at = out.idx(c, st * f + i, sh * h + j, sw * w + k);
                                    out.v[at] = y.v[y.idx(c * prod + ijk, f, h, w)];
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn drop_first_frame(x: &Vol) -> Vol {
        let mut out = Vol::zeros(x.c, x.f - 1, x.h, x.w);
        for c in 0..x.c {
            let per = x.h * x.w;
            out.v[c * (x.f - 1) * per..(c + 1) * (x.f - 1) * per].copy_from_slice(&x.v[(c * x.f + 1) * per..(c + 1) * x.f * per]);
        }
        out
    }

    fn tiny() -> Ltx2VideoVaeConfig {
        Ltx2VideoVaeConfig {
            latent_channels: 4,
            decoder_block_out_channels: vec![16, 32, 64],
            decoder_layers_per_block: vec![1, 1, 1, 1],
            patch_size: 2,
            ..Ltx2VideoVaeConfig::ltx2_19b()
        }
    }

    fn reference(map: &WeightMap, cfg: &Ltx2VideoVaeConfig, z: &Vol) -> Vol {
        let (mean, std) = (get(map, "latents_mean", &[z.c]), get(map, "latents_std", &[z.c]));
        let mut x = z.clone();
        let per = z.f * z.h * z.w;
        for c in 0..z.c {
            x.v[c * per..(c + 1) * per].iter_mut().for_each(|v| *v = *v * std[c] + mean[c]);
        }
        let stages = cfg.decoder_stages();
        let mut x = conv3(map, "decoder.conv_in", &x, stages[0].channels);
        for (i, stage) in stages.iter().enumerate() {
            let prefix = if i == 0 { "decoder.mid_block".to_string() } else { format!("decoder.up_blocks.{}", i - 1) };
            if let Some(up) = &stage.upsampler {
                let main = d2s(&conv3(map, &format!("{prefix}.upsamplers.0.conv"), &x, up.conv_out_channels), up.stride);
                let main = if up.drop_first_frame { drop_first_frame(&main) } else { main };
                let mut sum = main;
                if up.residual {
                    let res = d2s(&x, up.stride);
                    let res = if up.drop_first_frame { drop_first_frame(&res) } else { res };
                    let tiles = sum.c / res.c;
                    let per = sum.f * sum.h * sum.w;
                    for c in 0..sum.c {
                        for p in 0..per {
                            sum.v[c * per + p] += res.v[(c % res.c) * per + p];
                        }
                    }
                    assert_eq!(tiles, sum.c / res.c);
                }
                x = sum;
            }
            for n in 0..stage.resnet_layers {
                x = resnet(map, &format!("{prefix}.resnets.{n}"), &x);
            }
        }
        let y = conv3(map, "decoder.conv_out", &norm_silu(&x), 3 * cfg.patch_size * cfg.patch_size);
        let p = cfg.patch_size;
        let mut out = Vol::zeros(3, y.f, p * y.h, p * y.w);
        for c in 0..3 {
            for f in 0..y.f {
                for h in 0..y.h {
                    for w in 0..y.w {
                        for a in 0..p {
                            for b in 0..p {
                                let at = out.idx(c, f, p * h + b, p * w + a);
                                out.v[at] = y.v[y.idx(c * p * p + a * p + b, f, h, w)];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn latent(f: usize) -> Vol {
        let mut z = Vol::zeros(4, f, 2, 3);
        z.v.iter_mut().enumerate().for_each(|(i, v)| *v = (i as f32 * 0.73).sin());
        z
    }

    fn assert_same(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() < 2e-4 * (1.0 + b.abs()), "{what}[{i}]: {a} vs {b}");
        }
    }

    /// The sizes of the runs of frames the sink saw, and the video as `[3, F, H, W]`.
    fn decode_with_chunk(dec: &VideoDecoder, z: &Vol, chunk: usize) -> (Vec<usize>, CudaTensor) {
        let mut runs = Vec::new();
        let zt = CudaTensor::from_vec(z.v.clone(), vec![1, z.c, z.f, z.h, z.w]).unwrap();
        let mut pieces = Vec::new();
        dec.decode_streaming_chunked(&zt, chunk, &mut |offset, frames| {
            assert_eq!(offset, runs.iter().sum::<usize>(), "frames arrive in order");
            runs.push(frames.shape[0]);
            pieces.push(frames.clone());
            Ok(())
        })
        .unwrap();
        let all = CudaTensor::cat(&pieces.iter().collect::<Vec<_>>(), 0).unwrap();
        (runs, all.permute(&[1, 0, 2, 3]).unwrap())
    }

    #[test]
    fn decoder_matches_a_loop_reference() {
        let cfg = tiny();
        let map = weights();
        let dec = VideoDecoder::load(&map, &cfg).unwrap();
        let z = latent(2);
        let want = reference(&map, &cfg, &z);
        assert_eq!((want.f, want.h, want.w), (9, 2 * 8 * 2, 3 * 8 * 2));
        let (_, got) = decode_with_chunk(&dec, &z, 64);
        assert_eq!(got.shape, vec![3, 9, 32, 48]);
        assert_same(&got.host_cow().unwrap(), &want.v, "whole-clip decode");
    }

    /// The point of the streaming formulation: any chunk size gives the
    /// one-shot result, and frames leave before the clip is finished.
    #[test]
    fn streaming_in_chunks_equals_the_one_shot_decode() {
        let cfg = tiny();
        let dec = VideoDecoder::load(&weights(), &cfg).unwrap();
        let z = latent(3);
        let (whole_runs, whole) = decode_with_chunk(&dec, &z, 1000);
        assert_eq!(whole_runs, vec![17]);
        let whole = whole.host_cow().unwrap().into_owned();
        for chunk in [1usize, 2, 3, 5] {
            let (runs, got) = decode_with_chunk(&dec, &z, chunk);
            assert!(runs.len() > 1, "chunk {chunk} must stream in several runs, got {runs:?}");
            assert_eq!(runs.iter().sum::<usize>(), 17);
            let got = got.host_cow().unwrap();
            for (i, (a, b)) in got.iter().zip(&whole).enumerate() {
                assert!((a - b).abs() < 1e-5 * (1.0 + b.abs()), "chunk {chunk} value {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn a_single_latent_frame_decodes_to_a_single_image() {
        let dec = VideoDecoder::load(&weights(), &tiny()).unwrap();
        let (runs, got) = decode_with_chunk(&dec, &latent(1), 8);
        assert_eq!(runs, vec![1]);
        assert_eq!(got.shape, vec![3, 1, 32, 48]);
    }

    #[test]
    fn depth_to_space_follows_the_documented_index_map() {
        let y = Vol { c: 16, f: 2, h: 2, w: 3, v: (0..16 * 2 * 2 * 3).map(|i| i as f32).collect() };
        let got = depth_to_space(&CudaTensor::from_vec(y.v.clone(), vec![1, 16, 2, 2, 3]).unwrap(), (2, 2, 2)).unwrap();
        assert_eq!(got.shape, vec![1, 2, 4, 4, 6]);
        assert_eq!(&*got.host_cow().unwrap(), &d2s(&y, (2, 2, 2)).v[..]);
    }

    #[test]
    fn ltx25_config_loads_synthetic_weights() {
        let cfg = Ltx2VideoVaeConfig::ltx2_5_22b();
        let _ = cfg.decoder_stages();
        VideoDecoder::load(&weights(), &cfg).expect("load");
    }

    #[test]
    fn latents_of_the_wrong_shape_are_refused() {
        let dec = VideoDecoder::load(&weights(), &tiny()).unwrap();
        assert!(dec.decode(&CudaTensor::zeros(&[1, 5, 2, 2, 2])).is_err());
        assert!(dec.decode(&CudaTensor::zeros(&[2, 4, 2, 2, 2])).is_err());
        // Reflect padding needs two cells to reflect across.
        assert!(dec.decode(&CudaTensor::zeros(&[1, 4, 2, 1, 2])).is_err());
    }
}
