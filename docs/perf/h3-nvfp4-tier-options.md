# H3 NVFP4 tier: three speed / quality options (2026-10-14)

The owner asked for three options for the H3 NVFP4 tier, measured against upstream FastVideo, before choosing what
to ship. All three ran on **one RTX PRO 6000 Blackwell pod** (EU-RO-1, $2.49/hr, image
`fastvideo-rs-runtime:sha-b564268` for its CUDA libraries, this branch's `fv-gpucheck` at `c56aac4`). Every arm used
the same image, binary, prompts and seeds. One R2 fetch served all of them: 109 GB in 129 s, with h3-base read only from
the EU volume.

- **Option 5:** TAEH3 decode on V2-NVFP4.
- **Option 6:** the published NVFP4 VAE decoder linears, new in this branch behind `FASTVIDEO_H3_VAE_NVFP4=1`.
- **Option 7:** the Trim 8-step NVFP4 checkpoint, with and without sparse FP4 attention.

The base is main plus PR #113 (sparse FP4 VSA attention, the NVFP4 default). PR #114 (`FASTVIDEO_H3_VAE_AB`) had not
merged, so each decoder ran in its own process. Two things show that the denoised latents were identical across those
processes:
- The bf16 and MXFP8 decodes of the same request agree at 44.6-50.3 dB PSNR and LPIPS 0.001-0.004.
- A different sample of the same model scores 10-14 dB.

**Request.** All five prompts of `scripts/gpu/prompts-hd5.json`, each with its own seed (talking-head 1024, three
at 42, h3-demo 0). Canvases 832x480 and 1344x768, 124 frames (5 s).

**Process.** One process per arm and canvas, `--warm` (one untimed generation first). The bf16 text encoder streamed
once per prompt; after that every arm read the shared conditioning cache.

**Metrics:**
- Times are medians over the five prompts.
- "End to end" is `total_s` minus text time. The first cell, `v2-mxvae-480`, streamed the encoder for four prompts
  at 228 s each, so its raw totals are not comparable. Every other cell's text time was below 0.3 s.
- Peak is the in-process device maximum. It agrees with nvidia-smi sampled every second.

**Quality** is `fv-gpucheck compare-clips` (PSNR, LPIPS alex, sharpness and temporal-jitter ratios) per prompt,
followed by `gate` with `scripts/gpu/gate-policy.toml` (kind `lossy`). Its hard quality checks are frame count,
sharpness ratio 0.95-1.08, jitter ratio 0.85-1.20 and LPIPS available.

The gate's **performance** section asks for at least 1.03x on `denoise_s`. A decoder option leaves the denoise
untouched (1.00x), so every decoder-only pair below reads "fail" overall. **The quality column is the verdict that
matters for options 5 and 6.**

Scripts: `scripts/gpu/h3-tier-options-{validate,pod}.sh` and `h3-tier-options-summary.py`. Raw results are in
`artifacts/perf/h3-nvfp4-tier-options/` (`summary.md`, per-cell `benchmark.json` and logs, compare / gate reports,
middle-frame sheets `frame062-480p-*.jpg`).

Upstream reference (docs/ports/fasth3-nvfp4.md §5 and docs/ports/fasth3-sparse-fp4-attn.md §3.3, same GPU model):
- V2-NVFP4 denoise: 10.25 s at 480p and 29.3 s at 768p.
- Warm VAE decode: about 8.8 s, from its compiled VAE. Upstream's 768p VAE times are missing because of its
  compile bug.

## Option 5: TAEH3 decode on V2-NVFP4

| canvas | decoder | denoise s | VAE decode s | end to end s | peak GiB | PSNR / LPIPS vs official VAE | sharpness / jitter ratio | gate quality |
|---|---|---:|---:|---:|---:|---|---|---|
| 480p | official 36-block, MXFP8 linears (default) | 11.86 | 1.96 | 13.93 | 32.1 | — | — | — |
| 480p | official, bf16 linears | 11.94 | 2.63 | 14.74 | 29.8 | 45.9 dB / 0.002 (vs MXFP8) | 1.00 / 1.00 | pass |
| 480p | **TAEH3** | 11.92 | **0.14** | **12.19** | 26.8 | 24.8 dB / 0.170 [0.11-0.20] | **0.894** [0.855-0.905] / 0.989 | **FAIL** (sharpness, 5 of 5 prompts) |
| 768p | official, MXFP8 (default) | 31.93 | 4.64 | 36.76 | 38.5 | — | — | — |
| 768p | official, bf16 | 31.97 | 6.20 | 38.35 | 36.0 | 47.8 dB / 0.002 | 1.00 / 1.00 | pass |
| 768p | **TAEH3** | 31.88 | **0.44** | **32.95** | 34.2 | 27.3 dB / 0.149 [0.10-0.21] | **0.882** [0.817-0.901] / 0.965 | **FAIL** (sharpness, 5 of 5) |

- **Speed.** TAEH3 cuts the decode 14x / 11x, from 1.96 / 4.64 s to 0.14 / 0.44 s. End to end it saves 1.7 s at
  480p (-12%) and 3.8 s at 768p (-10%). It also frees 5.3 / 4.3 GiB of peak.
- **Quality.** It fails the gate's sharpness floor on every prompt at both canvases: 0.82-0.91 against 0.95.
  - The frames are the same picture, only softer (`frame062-480p-*.jpg`). Jitter is within the window.
  - This repeats docs/techniques.md (TAEH3 on the bf16 recipes: sharpness 0.84-0.93 against its twin).
- **Recommendation:** offer TAEH3 as an explicit **"fast preview"** option only (draft / scrubbing, where a sharp final
  frame is not the point). Do not make it the default, because it fails the gate. Against upstream, our official
  decode (1.96 / 4.64 s) is already well under upstream's warm ~8.8 s. That makes TAEH3 a preview tool, not
  something parity needs.

## Option 6: the published NVFP4 VAE decoder linears (`FASTVIDEO_H3_VAE_NVFP4=1`)

**What the checkpoint holds.** The V2-NVFP4 `vae/` header was read from R2 and dequantized against h3-base (numpy, four
linears of blocks 0 / 5 / 17 / 35):
- **Every** decoder block linear is ModelOpt NVFP4: `to_q`, `to_k` and `to_v` separately, `to_out.0`, `ff.net.0.proj`
  and `ff.net.2`, for all 36 blocks.
- The encoding is U8 E2M1 codes with **the low nibble first**, **row-major** F8_E4M3 per-16 scales, F32
  `weight_scale_2`, a calibrated F32 `input_scale`, and F32 biases.
- `quantization_config` ignores `proj_in` and `proj_out`, which stay f32. The encoder convolutions are f32.
- Read this way, each weight is 0.081 rel-L2 from h3-base's f32 weight. The DiT's ModelOpt FFN gets the same figure.
  Reading the high nibble first gives 1.41.

**Code** (`crates/fastvideo-cudarc/src/h3/vae.rs`). The seam is the MXFP8 one, behind a new opt-in flag:

| | |
|---|---|
| `FASTVIDEO_H3_VAE_NVFP4=1` | the block linears are read as published (`h3::nvfp4::PublishedLinear::read_modelopt`, `Nvfp4Operand::from_host_layout`). q / k / v form one `Nvfp4Stack`: the activation is quantized once with the largest of the three calibrated scales, and each part keeps its own global scale. Biases are f32 as published. The GEMM is cuBLASLt `VEC16_UE4M3` W4A4 (the DiT's path), bf16 out, with the bias fused into the next kernel as on the bf16 path (`Linear::gemm_nvfp4_bf16`) |
| valid only when | the `vae/` carries the weights (`decoder.transformer_blocks.0.attn.to_q.weight_scale` is in the opened files) **and** the GPU is sm_100 / sm_120. `=1` on a bf16 `vae/` is an error; a ModelOpt `vae/` without the flag is refused with what to do, because the tree has no bf16 copy. `MXFP8=1` together with `NVFP4=1` is an error, and MXFP8 is otherwise off with NVFP4 |
| `FASTVIDEO_H3_VAE_NVFP4_ACT=dynamic` | measure the activation amax per call instead of using the calibrated `input_scale` |
| `FASTVIDEO_H3_VAE_CHECK=1` + `FASTVIDEO_H3_VAE_NVFP4_REF=<bf16 vae dir>` | on the first tile batch, the NVFP4 fused output is compared with the NVFP4 unfused output, and with the reference tree's bf16 unfused, bf16 fused and MXFP8 fused outputs. Sanity bound 0.25 rel-L2 |
| folder | the V2-NVFP4 `vae/` (config, index, shards 1-2) plus h3-base's byte-identical shard 3 |
| tests | `h3::vae::tests::vae_nvfp4_needs_the_published_linears_and_nvfp4_tensor_cores` (the resolution rules). `cargo test -p fastvideo-cudarc --lib -- h3::vae h3::nvfp4 h3::pipeline h3::manifest_tests`: 43 passed |

**Check** (`FASTVIDEO_H3_VAE_CHECK`: 6 tiles of grid [7, 16, 16], 480p, `check/{static,dynamic}/vae-lines.txt`):

| rel-L2 of the decoded tiles | static `input_scale` (default) | dynamic amax |
|---|---:|---:|
| NVFP4 fused vs NVFP4 unfused (the integration) | 3.4e-2 | 3.4e-2 |
| **NVFP4 vs bf16 unfused** (the reference) | **4.31e-2** | 4.26e-2 |
| NVFP4 vs MXFP8 fused | 4.34e-2 | — |
| for scale: MXFP8 vs bf16 unfused / bf16 fused vs unfused | 1.24e-2 / 1.9e-3 | |
| tile-batch time, synchronized per op: NVFP4 / MXFP8 / bf16 fused | 0.146 / 0.150 / 0.196 s | 0.145 s |

The check passes, well inside its sanity bound. NVFP4 is about 3.5x further from bf16 than MXFP8 is. The calibrated
scales are as good as measuring the amax, so static stays the default.

**Cells** (same latents; decode only differs):

| canvas | decoder linears | VAE decode s | end to end s | peak GiB | decoder weights held | PSNR / LPIPS vs bf16 | sharpness / jitter vs bf16 | gate quality vs bf16 | vs MXFP8 |
|---|---|---:|---:|---:|---|---|---|---|---|
| 480p | bf16 | 2.63 | 14.74 | 29.8 | 4.5 GiB | — | — | — | |
| 480p | MXFP8 (default) | 1.96 | 13.93 | 32.1 | 4.5 + 2.3 GiB | 45.9 dB / 0.002 | 1.001 / 1.001 | pass | — |
| 480p | **NVFP4** | **1.77** | **13.89** | **27.1** | **1.27 GiB** | 35.9 dB / 0.019 | 1.027 / 1.016 | **pass** | pass (35.8 dB, 0.019; 1.026 / 1.019) |
| 768p | bf16 | 6.20 | 38.35 | 36.0 | | — | — | — | |
| 768p | MXFP8 (default) | 4.64 | 36.76 | 38.5 | | 47.8 dB / 0.002 | 1.003 / 1.001 | pass | — |
| 768p | **NVFP4** | **4.12** | **36.22** | **33.3** | | 37.6 dB / 0.018 | 1.028 / 1.020 | **pass** | pass (37.6 dB, 0.018; 1.025 / 1.020) |

- **Speed:** decode -10% / -11% against MXFP8 (-0.19 / -0.52 s) and -33% against bf16; end to end -0.3% / -1.5%. The
  ViT's GEMMs were already a minority of the decode at MXFP8 (the check's per-op table: SDPA, SwiGLU, norms and the
  activation quantization take the rest), so 4-bit weights buy little time.
- **Memory:** -5.0 / -5.2 GiB of peak against the MXFP8 default (which holds bf16 + MXFP8 copies), -2.7 GiB against
  bf16. This is the option's real gain.
- **Quality:** passes the gate's hard checks against both bf16 and MXFP8 on all five prompts at both canvases
  (sharpness 1.014-1.035, jitter 1.003-1.036). It is a visible-only-in-difference change: 34-40 dB PSNR, LPIPS
  0.010-0.030, 10x the MXFP8 decoder's LPIPS.
- **Recommendation:** for the NVFP4 tier, **make the published NVFP4 VAE the tier's decoder** when the pool is
  memory-bound (e.g. a 32 GB card, or co-residence). It passes the gate and saves 5 GiB, and it is never slower. As a
  pure speed lever it is worth ~0.5 s at 768p only; it stays opt-in (`FASTVIDEO_H3_VAE_NVFP4=1`) until the owner picks.
  It needs the V2-NVFP4 `vae/` shards in the tier's tree (1.7 GB, already in R2 under `fasth3-v2-nvfp4`).

## Option 7: FastH3 Trim 8-step NVFP4 (`fasth3-trim-8step-nvfp4`)

The DiT has **42 blocks** (the 5 NVFP4 linears of every block, 9.33 GiB resident). The 26 in "26 blocks" is the Trim's
lightweight **VAE**, which decodes bf16 here: the MXFP8 default is for the 36-block decoder only. V2-NVFP4 is 50 blocks,
FFN-only NVFP4, MXFP8 36-block VAE.

| canvas | arm | denoise s | s / fwd | VAE s | end to end s | peak GiB | vs V2-NVFP4 denoise |
|---|---|---:|---:|---:|---:|---:|---:|
| 480p | V2-NVFP4 + FP4 attn (tier default) | 11.86 | 1.485 | 1.96 | 13.93 | 32.1 | 1.00x |
| 480p | Trim bf16 (`FASTVIDEO_H3_QUANT=off`) | 17.17 | 2.145 | 1.92 | 19.19 | 44.1 | 0.69x |
| 480p | Trim NVFP4 (bf16 VSA, as wired by #105) | 11.02 | 1.374 | 1.91 | 13.04 | 20.3 | 1.08x |
| 480p | **Trim NVFP4 + FP4 attn** (`FASTVIDEO_H3_VSA_FP4=1`, new) | **9.21** | **1.155** | 1.91 | **11.23** | **19.7** | **1.29x** |
| 768p | V2-NVFP4 + FP4 attn | 31.93 | 3.991 | 4.64 | 36.76 | 38.5 | 1.00x |
| 768p | Trim bf16 | 48.55 | 6.068 | 4.51 | 53.24 | 51.3 | 0.66x |
| 768p | Trim NVFP4 | 32.65 | 4.080 | 4.51 | 37.35 | 27.9 | 0.98x |
| 768p | **Trim NVFP4 + FP4 attn** | **24.64** | **3.078** | 4.50 | **29.31** | **27.1** | **1.30x** |

Upstream V2-NVFP4 (10.25 / 29.3 s denoise) is beaten on denoise by Trim NVFP4 + FP4 attention at both canvases
(0.90x / 0.84x of its time). There is **no upstream Trim NVFP4 number**: FastVideo does not load the ComfyUI file
(docs/ports/h3-trim.md §5.4), so nothing upstream was run.

**Quality** (gate quality section; per prompt):

| baseline -> candidate | canvas | PSNR dB | LPIPS | sharpness | jitter | gate quality |
|---|---|---|---|---|---|---|
| Trim bf16 -> Trim NVFP4 | 480p | 12.1 [9.9-13.8] | 0.60 | 1.01 [0.96-1.43] | 1.08 [0.85-1.31] | FAIL (2 prompts) |
| Trim bf16 -> Trim NVFP4 | 768p | 11.6 [10.0-14.7] | 0.60 | 1.00 [0.80-1.21] | 1.29 [0.75-2.17] | FAIL (5 prompts) |
| Trim bf16 -> Trim NVFP4 + FP4 attn | 480p / 768p | 11.4 / 11.1 | 0.62 / 0.60 | 0.92 / 0.94 | 0.82 / 0.92 | FAIL / FAIL |
| Trim NVFP4 -> + FP4 attn | 480p / 768p | 13.0 / 13.4 | 0.55 / 0.51 | 0.91 / 0.93 | 0.83 / 0.83 | FAIL / FAIL |
| V2-NVFP4 -> Trim NVFP4 | 480p / 768p | 10.2 / 10.7 | 0.66 / 0.68 | 1.07 / 1.13 | 0.99 / 1.14 | FAIL / FAIL |
| V2-NVFP4 -> Trim NVFP4 + FP4 attn | 480p / 768p | 10.2 / 10.6 | 0.68 / 0.68 | 1.09 / 1.02 | 0.80 / 0.81 | FAIL / FAIL |

Every pair is a different sample. Against bf16 Trim, Trim NVFP4 fails as §5.4 of docs/ports/h3-trim.md found (here on
five prompts); FP4 attention moves it further (sharpness 0.91-0.93 and jitter 0.83 against Trim NVFP4, the same
softer / calmer direction FP4 attention took on V2, docs/ports/fasth3-sparse-fp4-attn.md §3.4). Against V2-NVFP4 the
Trim is another model: a different person and room for the same prompt (`frame062-480p-talking-head.jpg`), PSNR ~10 dB.
For scale, V2-NVFP4 itself fails the gate against bf16 V2 and against upstream's own output
(docs/ports/fasth3-nvfp4.md §6).

**FP4 attention and `is_modelopt_dir`.** As wired by #113 it did **not** apply: the Trim file is ComfyUI, not a ModelOpt
`transformer/`, and `vsa_fp4_choice` keyed on `is_modelopt_dir`. Nothing in the kernel depends on the checkpoint (same
56 x 128 heads, tile 64, VSA 0.8), so this branch lets `FASTVIDEO_H3_VSA_FP4=1` take it for the Trim NVFP4 DiT
(`h3/pipeline.rs`: `vsa_fp4_choice(modelopt, comfy)`); unset keeps bf16 VSA there (logged). It gave 1.20x / 1.33x denoise
on top of the Trim NVFP4.

**Recommendation:** Trim NVFP4 + FP4 attention is **the fastest H3 configuration measured** (9.2 / 24.6 s denoise,
11.2 / 29.3 s end to end, 20-27 GiB peak) and the only one under upstream's V2-NVFP4 denoise. It fails the gate
against everything it can be compared to, as does every NVFP4 tier so far, so it can only ship as an explicit opt-in
"fastest" tier judged by eye. Without FP4 attention the Trim NVFP4 is not worth offering over V2-NVFP4: level at 768p,
1.08x at 480p, a different and smaller model.

## Combined: the fastest tier that passes the gate

The gate here compares against the tier's current default (V2-NVFP4 + FP4 attention, official VAE with MXFP8 linears),
the reference every option was asked against:

| option | end to end 480p / 768p | vs tier default | gate quality vs default | verdict |
|---|---|---:|---|---|
| tier default (V2-NVFP4 + FP4 attn, MXFP8 VAE) | 13.93 / 36.76 s | 1.00x | — | baseline |
| **+ NVFP4 VAE (option 6)** | **13.89 / 36.22 s** | 1.00x / 1.01x | **pass** | **fastest that passes; -5 GiB** |
| + TAEH3 (option 5) | 12.19 / 32.95 s | 1.14x / 1.12x | fail (sharpness) | preview only |
| Trim NVFP4 + FP4 attn (option 7) | 11.23 / 29.31 s | 1.24x / 1.25x | fail (different model) | opt-in "fastest" |
| Trim NVFP4 + FP4 attn + NVFP4 VAE | not applicable: the Trim's 26-block VAE has no published NVFP4 form | | | |

- **Ship:** V2-NVFP4 + FP4 attention with the **published NVFP4 VAE** as the NVFP4 tier (gate pass, same speed, 5 GiB
  less), once the owner accepts a tier decoder that only that tree carries.
- **Offer as explicit options:** TAEH3 "fast preview" (any tier; -10 to -12% end to end, softer frames) and Trim NVFP4
  + FP4 attention "fastest" (1.25x the tier end to end; a different model's sample).
- Against upstream (V2-NVFP4, 10.25 / 29.3 s denoise, ~8.8 s warm decode): our tier default is 1.16x / 1.09x behind on
  denoise but ahead end to end wherever upstream's decode counts (13.9 s against ~19 s at 480p).

## Cost and clean-up

- GPU: one pod, `s9mm8tx90r8llm` (EU-RO-1, $2.49/hr), 16:22:38-17:34, about **$2.97**. The first cell spent 15 min
  streaming the bf16 text encoder from the network volume for four new prompts (228 s each); a pod with the FP8 encoder
  resident or a warm text cache would save that.
- Build: the shared build pod (`fv-build-eu-083e2c`, already running): check, tests, one release build.
- Clean-up: the pod deleted and checked gone (`GET /pods/s9mm8tx90r8llm`: 404); its backstop (`fv-h3opts-backstop-…`)
  stopped by its exact PID; results fetched and checked (`gzip -t`, `tar -t`) before the delete. The staged bundle
  `fv-weights-staging/tmp/h3-tier-options-bundle-64d01601…` is removed by its exact key. Nothing was written to R2's
  weights bucket or to a volume.
