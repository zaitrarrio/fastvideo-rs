# Real-ESRGAN compact video upscaler (`realesr-animevideov3`)

Date: 2026-11-18. Branch `feat/esrgan-upscaler`. Code:
`crates/fastvideo-cudarc/src/rt/esrgan.rs`; kernels `bias_prelu_inplace` and
`esrgan_pack_rgb8` in `crates/fastvideo-cudarc/src/rt/kernels/core.cu` with
their host twins in `rt/ops.rs` (`ops::host::bias_prelu`,
`ops::host::esrgan_pack_rgb8`); parity group `esrgan` in
`crates/fastvideo-gpucheck/src/kernels.rs`; the numpy reference
`scripts/ref/esrgan_parity.py`.

## What

A ×2 / ×3 / ×4 pixel upscaler for finished clips, run after the pipeline
has written its frames, for every family whose output is PNG frames (Wan,
H3, LTX, the rest). It is Real-ESRGAN's compact network, SRVGGNetCompact,
with the `realesr-animevideov3` checkpoint: **2.5 MB** of f32 weights,
**BSD-3-Clause** (xinntao/Real-ESRGAN; the checkpoint is release v0.2.5.0,
pinned by URL, sha256 and size in `scripts/gpu/fetch-tae.sh` and
`scripts/gpu/weights-manifest.tsv` `auxiliary/upscalers/`).

This is a different, cheap option from the diffusion video super-resolution
models `docs/serve/h3-1080p-and-upscaler.md` decided (2026-09-28) **not** to
port (SeedVR2 3B, FlashVSR: a second model with about the generation's own
GPU time). It invents no detail: it sharpens edges and keeps the picture
steady, at a cost of tens of milliseconds per 1080p frame. fastwan-rs's
comparison (`fastwan-rs/docs/runs/2026-11-12-upscalers/README.md`) found it
the fastest and steadiest learned option on generated 480×272 clips: 0.22 s
for 81 frames to 1920×1088 (fp16, RTX 4090), against 7.34 s for
`RealESRGAN_x4plus` and 2.6–3.0 s for RealBasicVSR, which over-processes
generated video into an oil-paint look.

Ported from fastwan-rs's `crates/fastwan-model/src/upscale.rs` (the network,
the device-side pack, the batch rule) and h3-rs's
`crates/h3-model/src/esrgan.rs` / `crates/h3-cli/src/finish.rs` (the ×2 / ×3
rule, `cli_factor`, `serve_factor`), both of which match a numpy
SRVGGNetCompact on the real weights.

## The network

```
conv3×3 3→64, PReLU
16 × (conv3×3 64→64, PReLU)
conv3×3 64→48
pixel-shuffle ×4  (48 = 3 · 4²)
+ the input upsampled ×4 (nearest), clamp [0, 255], round
```

Keys are the `nn.Sequential` positions of `body`: `body.{2i}.weight` /
`.bias` for conv `i`, `body.{2i+1}.weight` its PReLU slope (`esrgan_manifest`
lists all 53, 1 184 972 floats). The `.pth` is read directly by
`fastvideo_loader::pth::read_pth_nested` under `params_ema` / `params` (no
conversion, no Python); a safetensors with the same keys also loads. A
missing key is an error.

On the device (`Esrgan::upscale_rgb8`): the 8-bit frames go up as f32
NCHW in [0, 1] plus the exact 8-bit bytes for the residual; every conv runs
through `rt::conv::cudnn_conv` (f32, the plan cache) followed by the fused
`bias_prelu_inplace` epilogue (the last conv `add_bias_inplace`); then
`esrgan_pack_rgb8` does the pixel shuffle, the residual add from the 8-bit
input, the clamp and the rounding and writes interleaved RGB8 on the
device, so a frame comes down as 3 bytes per output pixel (12 for the f32
map). Frames run in batches sized by `Esrgan::batch_frames`: per frame
`(3 + 2·64 + 48)` f32 planes plus the 8-bit input and output, under a 2 GiB
budget (`FASTVIDEO_ESRGAN_BUDGET_GIB`): 21 frames per batch at 480×272, 7
at 832×480.

The plain-Rust path (`Esrgan::upscale_rgb8_cpu`, convs accumulating in f64
and storing f32 per layer, the pack exact in f64) is the reference the
device path is checked against, and what runs when no device is live.

## ×2 and ×3

- **×4** is the network.
- **×2** is the same ×4 network with its clamped ×4 picture box-filtered
  2×2 before the rounding, inside `esrgan_pack_rgb8` (`down = 2`). fastwan-rs
  found no fast ×2 model better than that (`RealESRGAN_x2plus` was 10×
  slower and barely distinguishable from bicubic ×2), and the network is
  the cost either way, so ×2 costs about what ×4 does and writes a quarter
  of the pixels.
- **×3** is the ×4 frames Lanczos-3 resampled on the host (`resize_lanczos3`,
  separable, row-parallel), like Real-ESRGAN's `--outscale` and h3-rs's rule.
  It is the slowest factor: the host resample dominates (h3-rs: 768p ×2 by
  resampling 18.5 s, "mostly host Lanczos", which is why ×2 here is the box
  filter on the device instead).

Rules shared with the CLIs: `default_factor(w, h)` (h3-rs's `cli_factor`:
×4 up to 832×480, ×2 above) and `serve_factor(w, h, scale)` (the largest
factor whose output fits 2560×1440: 480×272 → ×4, 832×480 → ×3, 1344×768 →
×2). Neither is applied automatically yet: the flags take an explicit factor.

## Flags, environment, files

- `fv-cli generate --upscale 2|3|4 [--upscale-weights PATH]` (TOML overlay
  keys `upscale`, `upscale_weights`): after the family wrote
  `frame-NNN.png` (+ `output.mp4`), the frames are upscaled into
  `frames-up/frame-NNN.png` and `output-up.mp4` next to them (the mp4 at the
  original's frame rate, probed with `ffprobe`; H3's `audio.wav` muxed in
  through the pipelines' own `VideoWriter`), and the output size and the
  upscale time are printed.
- `fv-gpucheck wan gen --upscale N [--upscale-weights PATH]` and
  `fv-gpucheck h3 gen --upscale N [--upscale-weights PATH]`: the same
  through `rt::esrgan::upscale_dir`, recording `upscale` (factor, in / out
  sizes, batch, device, read / load / upscale / write seconds, fps, mp4) and
  `upscale_s` in `benchmark.json`.
- Weights, in order: `--upscale-weights`, `FASTVIDEO_ESRGAN_WEIGHTS`,
  `$FASTVIDEO_TAE_DIR/realesr-animevideov3.pth`,
  `auxiliary/upscalers/realesr-animevideov3.pth` under `--weights`, its
  parent, `FASTVIDEO_WEIGHTS`, `FV_WEIGHTS`. A missing file names every path
  tried and the pinned URL.
- Pods: `scripts/gpu/fetch-tae.sh` fetches the `.pth` (sha256-checked)
  beside the TAE files onto the container disk (the volume's
  `auxiliary/upscalers/` copy first), so a lab arm with
  `tae = "realesr-animevideov3.pth"` reads `{tae}/realesr-animevideov3.pth`;
  `verify-weights.sh aux` checks the manifest row.
- Lab: `lab/cells/upscale.toml` (`upscale-h3-480p`: FastH3 4-step VSA 480p
  + TAEH3, ×2; `upscale-wan-480p`: FastWan2.1 1.3B 480p + TAEHV, ×4). Not in
  a tier yet.

## Checks

- `cargo test -p fastvideo-cudarc esrgan`: the manifest, a missing key, the
  batch rule, the factors, the tiny config end to end at every factor, ×2
  as the box filter of ×4, the Lanczos resample, the pack and PReLU twins,
  a tiny network against an independently written plain loop, and, with
  `FASTVIDEO_ESRGAN_WEIGHTS` set, the real checkpoint on a 24×16 LCG frame:
  byte-identical to the plain loop and to `scripts/ref/esrgan_parity.py`'s
  numpy SRVGGNetCompact (sha256 of the ×4 bytes pinned in the test). Run on
  2026-11-18 on the real weights: 96×64 output, sha256
  `23224e6c…9041f9`, exact.
- `fv-gpucheck kernels --groups esrgan` (a GPU): `bias_prelu_inplace` and
  `esrgan_pack_rgb8` bit-exact against their twins (three shapes, ×4 and
  the ×2 box, ×2 network), the tiny network on the device against the host
  reference at ×4 / ×3 / ×2 (≤ 1 level, < 1 % of bytes), the real checkpoint
  the same when it is reachable, and a timing of 81 frames 480×272 →
  1920×1088 (`esrgan_480x272x81_x4`).

## Reference numbers (other repos; not yet measured here)

- fastwan-rs, RTX 4090, CUDA: 81 frames 480×272 → 1920×1088 in **0.87 s**,
  of which 285 ms is GPU (`fastwan-rs/CLAUDE.md`, the "Upscaler" bullet;
  `docs/runs/2026-11-13-vast-4090-gpu-pack-browser`).
- h3-rs, RTX PRO 6000 Server Edition: a 272p draft ×4 to 1920×1088 **1.7 s**,
  480p ×4 to 3328×1920 **4.4 s**, 768p ×2 (resampled) 18.5 s, mostly host
  Lanczos (`h3-rs/CLAUDE.md`, "Sizes off the 32-pixel grid";
  `docs/runs/2026-11-17-vast-pro6000-ref2va-upscale`).

## Unverified

No GPU run of this port yet: the two kernels are written against their host
twins and the `esrgan` parity group, which is how this repository proves
kernels, but neither has compiled or run on a device, and no fastvideo-rs
timing exists. The cuDNN path is f32 (TF32 under the default math) and is
compared to the f64 host reference with a 1-level tolerance; the ×3 Lanczos
is Lanczos-3 in f32, not OpenCV's `INTER_LANCZOS4`, so ×3 frames are not
byte-identical to Real-ESRGAN's `--outscale 3`.
