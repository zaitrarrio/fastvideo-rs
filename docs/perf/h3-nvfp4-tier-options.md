# H3 NVFP4 tier: three speed / quality options (2026-10-14)

**Decided (owner, 2026-10-14) and shipped:** the NVFP4 VAE is the NVFP4 tier's default decoder; TAEH3 is an explicit
"fast preview" choice; Trim NVFP4 + sparse FP4 attention is an opt-in "fastest" tier; the training pilot is declined.
See [Decisions and what shipped](#decisions-2026-10-14-and-what-shipped). The measurements below are the basis.

The owner asked for three options for the H3 NVFP4 tier, measured against upstream FastVideo, before choosing what
to ship. All three ran on **one RTX PRO 6000 Blackwell pod** (EU-RO-1, $2.49/hr, image
`fastvideo-rs-runtime:sha-b564268` for its CUDA libraries, this branch's `fv-gpucheck` at `c56aac4`). Every arm used
the same image, binary, prompts and seeds. One R2 fetch served all of them: 109 GB in 129 s, with h3-base read only from
the EU volume.

- **Option 5:** TAEH3 decode on V2-NVFP4.
- **Option 6:** the published NVFP4 VAE decoder linears, measured behind `FASTVIDEO_H3_VAE_NVFP4=1` (now the
  default of a `vae/` that carries them).
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

Scripts: `scripts/gpu/experiments/h3-tier-options-{validate,pod}.sh` (deleted in D1e, docs/adr/0019-fv-lab.md §14; in git at `22a5fd7b`) and `h3-tier-options-summary.py` (since D1c `fv-lab report summary --kind h3-tier-options DIR --json OUT`). Raw results are in
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

## Option 6: the published NVFP4 VAE decoder linears (`FASTVIDEO_H3_VAE_NVFP4`)

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
| valid only when | the `vae/` carries the weights (`decoder.transformer_blocks.0.attn.to_q.weight_scale` is in the opened files) **and** the GPU is sm_100 / sm_120. `=1` on a bf16 `vae/` is an error; (as measured; since the decision a ModelOpt `vae/` decodes with them by default and `=0` takes h3-base's `vae/`, see below). `MXFP8=1` together with `NVFP4=1` is an error, and MXFP8 is otherwise off with NVFP4 |
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
  pure speed lever it is worth ~0.5 s at 768p only; it was opt-in (`FASTVIDEO_H3_VAE_NVFP4=1`) until the owner picked it (now the default, see the decisions below).
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

## Decisions (2026-10-14) and what shipped

| decision | what it is now | how to choose it |
|---|---|---|
| **1. The NVFP4 VAE is the NVFP4 tier's decoder** | a `vae/` that carries the published NVFP4 decoder linears (the V2 / 4-step NVFP4 releases') decodes with them **by default**; `FASTVIDEO_H3_VAE_NVFP4=0` decodes with h3-base's `vae/` (bf16 + MXFP8 copies), the decoder the tier ran before | served recipes `v2-nvfp4` (catalog `fasth3-v2-nvfp4`, served `h3-v2-nvfp4`, preset `fasth3-v2-nvfp4`) and `4step-vsa-nvfp4` (catalog `fasth3-4step-vsa-nvfp4`, served `h3-turbo-nvfp4`, preset `fasth3-4step-nvfp4`); untiered and opt-in, `h3-turbo` stays the default H3 tier |
| **2. TAEH3 is an explicit "fast preview"** | the same recipe decoded by TAEH3 at its canvases (768p default, 480p), a model id of its own; never a default. It **fails the sharpness gate** (0.82-0.91 against the official decoder: softer frames of the same picture) and saves **about 10-12% end to end** (option 5) | `h3-v2-nvfp4-preview` (`fasth3-v2-nvfp4-taeh3`), `h3-turbo-nvfp4-preview` (`fasth3-4step-vsa-nvfp4-taeh3`), `h3-turbo-preview` (`fasth3-4step-vsa-taeh3`); presets of the same names (`fasth3-v2-nvfp4-preview`, `fasth3-4step-nvfp4-preview`, `h3-turbo-preview`). `h3-draft` (the 480p draft tier) is unchanged |
| **3. Trim NVFP4 + sparse FP4 attention is an opt-in "fastest" tier** | recipe `trim-8step-nvfp4-fp4attn` turns the FP4 VSA fine stage on (`FASTVIDEO_H3_VSA_FP4=1` semantics: an error without the sm_120 kernel; `=0` still turns it off), no process-wide variable. **A different model from V2, not gate-equivalent**: 9.21 / 24.64 s denoise at 480p / 768p, about 20 / 27 GiB peak | catalog `fasth3-trim-nvfp4-fast`, served `h3-trim-nvfp4-fast`, preset `fasth3-trim-nvfp4-fast` (R2 trees `fasth3-trim-8step`, `fasth3-trim-8step-nvfp4`, `h3-base`) |
| 4. Training pilot | declined | — |

**Why model ids for the preview, not a request field.** The serving API's per-request knobs are sampling knobs
(`steps`, `guidance`, ...; `SamplingOverrides`), and every draft decoder already is a model id of its own (`h3-draft`,
`ltx-draft`, `wan-draft`), which every API (native, OpenAI videos, MiniMax, fal) and fv-control's pools, standalone
pods and serverless endpoints address the same way. A request field would also need a second decoder resident beside
each pipeline. So a preview is chosen by model id, and its pool runs it alone.

**Decoder resolution** (`crates/fastvideo-cudarc/src/h3/vae.rs` `resolve_nvfp4`, `h3/pipeline.rs`
`resolve_video_vae`), never a silent fallback:

| `vae/` of the tree | `FASTVIDEO_H3_VAE_NVFP4` | decoder |
|---|---|---|
| carries the NVFP4 linears | unset or `1` | the published NVFP4 linears, sm_100 / sm_120; elsewhere an error naming `=0` |
| carries the NVFP4 linears | `0` | h3-base's `vae/` (the text weights' tree), bf16 + MXFP8; an error naming `--text-weights` when there is none |
| bf16 (h3-base, the Trim) | unset or `0` | unchanged (bf16 + MXFP8 as `FASTVIDEO_H3_VAE_MXFP8` resolves) |
| bf16 | `1` | an error (no NVFP4 linears) |

The R2 trees `fasth3-v2-nvfp4` / `fasth3-4step-vsa-nvfp4` carry `vae/` shards 1-2 (the NVFP4 linears), its config and
index; the index's third shard, byte-identical to h3-base's, is read from h3-base's `vae/`
(`completed_vae_files`, logged), so no composed folder is needed. A shard in neither is an error that says to fetch
h3-base beside the tree. `FASTVIDEO_H3_VAE_MXFP8=1` with the NVFP4 decoder is an error.

**Memory plan** (`crates/fastvideo-models/src/h3/memory.rs` `H3VaeLinears`): the video decoder is priced as held.

| decoder | priced | of which block linears |
|---|---:|---:|
| bf16 (`MXFP8=0`, pre-sm_100, TAEH3 as the safe side) | 4.50 GiB | 4.50 GiB bf16 |
| bf16 + MXFP8 copies (the default elsewhere) | 6.83 GiB | + 2.32 GiB MXFP8 |
| **published NVFP4 (the NVFP4 tier's default)** | **1.27 GiB** | **1.27 GiB** (E2M1 + E4M3 per 16; the load log's "1.27 GiB with biases") |

The NVFP4 decoder's plan is 5.55 GiB under the MXFP8 default's; the measured peak dropped 5.0 / 5.2 GiB at 480p / 768p.

**Trees each preset needs** (`control/src/recipe-trees.ts`, checked against the configs and the Rust sources by
`control/test/unit/preset-trees.test.ts`): the NVFP4 tier and its previews read their release tree (the config's
`weights`) plus `h3-base` (text encoder, tokenizer, audio VAE, `vae/` shard 3), prequant `h3-base/text_encoder_fp8`;
the Trim fastest tier `fasth3-trim-8step`, `fasth3-trim-8step-nvfp4`, `h3-base`; `h3-turbo-preview` the same as
`h3-turbo`; TAEH3 is an aux file every boot fetches. The release trees are in R2 only (not on the EU volume), so those
presets default to `weights_source r2`; `h3-turbo-preview` is all on the EU volume.

**Code:** `fastvideo-models` `h3::config` (`nvfp4_release_contract`, `is_trim_nvfp4_fp4attn_recipe`),
`h3::memory`; `fastvideo-cudarc` `h3::vae`, `h3::pipeline` (`resolve_video_vae`, `vsa_fp4_choice(.., recipe_fp4)`;
a release recipe on a non-ModelOpt `transformer/` is refused); `fastvideo-engine-service` `cuda::caps` (catalog
entries, NVFP4 requirement, T2V only for the releases); `fastvideo-serve` `tokens` (the release trees' tokenizer is
h3-base's); `configs/serve/runpod-h3-{v2-nvfp4,turbo-nvfp4,v2-nvfp4-preview,turbo-nvfp4-preview,turbo-preview,trim-nvfp4-fast}.toml`;
fv-control presets, recipe trees, catalog, serverless recipes, `scripts/serve/variants.sh`.

**GPU smoke of what shipped** (one RTX PRO 6000, pod `2d8zzz9gsqsop3`, US-NC-1, r2 mode, image
`fastvideo-rs-runtime:sha-b564268`, this branch's and main's (`cf8a20d`) release `fv-gpucheck`;
`scripts/gpu/experiments/h3-tier-ship-{validate,pod}.sh`, deleted in D1e, docs/adr/0019-fv-lab.md §14; in git at `22a5fd7b`; results in `artifacts/perf/h3-nvfp4-tier-ship/`). One prompt
(talking-head, seed 1024), 832x480, 124 frames, `--warm`, one process each. V2 ran on the R2 tree as fetched
(`--h3-recipe v2-nvfp4 --text-weights h3-base`); main ran it from a folder holding the tree's `transformer/` and
h3-base's `vae/`, as option 6 measured it.

| run | decoder (log line) | denoise s | VAE s | total s | peak (device max) |
|---|---|---:|---:|---:|---:|
| main, default | MXFP8 36 blocks (h3-base `vae/`) | 9.95 | 1.97 | 12.22 | 30.7 GiB |
| **this branch, default** | **"NVFP4 decoder (FASTVIDEO_H3_VAE_NVFP4 default: the tree carries them) ... 1.27 GiB"**; shard 3 "from /fvw/h3-base/vae/..." | 10.02 | **1.77** | 12.06 | **25.7 GiB** |
| this branch, `FASTVIDEO_H3_VAE_NVFP4=0` | "h3-base's vae/ ... (bf16 / MXFP8 linears)"; MXFP8 36 blocks | 10.06 | 1.99 | 12.33 | 30.7 GiB |
| this branch, preview (TAEH3) | "h3 vae=taeh3" | 10.09 | 0.13 | 10.55 | 25.5 GiB |
| this branch, `trim-8step-nvfp4-fp4attn` | "VSA fine stage on sparse FP4 tensor cores (FASTVIDEO_H3_VSA_FP4=1 by the recipe ...)"; MXFP8 26 blocks | **7.53** | 1.45 | 9.27 | 21.1 GiB |

- The `=0` clip is **byte-identical** to main's default: all 250 output files (124 PNG frames and the audio, the warm
  and the timed generation) have the same SHA-256.
- The NVFP4 default decodes in 1.77 s, against option 6's 1.77 s and about 1.8 s expected, with 5.0 GiB less peak.
- The Trim fastest recipe's denoise (7.53 s) is under option 7's 9.21 s, because the glue fusions and lazy step events
  merged since. It ran with the FP4 attention on from the recipe alone, with no variable set.

## Cost and clean-up

**Tier-ship smoke (2026-10-14):** pod `2d8zzz9gsqsop3`, $2.49/hr, 22:22-22:33 UTC: **about $0.46**.
- The run order was: fetch (187 GB in 140 s), runs, pack.
- The samplers were stopped by their PIDs before the tar. The archive was checked (`gzip -t`, `tar -t`) and fetched, and only then was the pod deleted.
- The delete was checked: `GET /pods/2d8zzz9gsqsop3` returned 404.
- The backstop (`fv-h3ship-backstop-2d8zzz9gsqsop3`) was stopped by its exact PID.
- The staged bundle `fv-weights-staging/tmp/h3-tier-ship-bundle-fd0f5f03…` was removed by its exact key.
- Builds ran on the shared build pod.
- R2 and the volumes were only read.

**Options measurement:**

- GPU: one pod, `s9mm8tx90r8llm` (EU-RO-1, $2.49/hr), 16:22:38-17:34, about **$2.97**. The first cell spent 15 min
  streaming the bf16 text encoder from the network volume for four new prompts (228 s each); a pod with the FP8 encoder
  resident or a warm text cache would save that.
- Build: the shared build pod (`fv-build-eu-083e2c`, already running): check, tests, one release build.
- Clean-up: the pod deleted and checked gone (`GET /pods/s9mm8tx90r8llm`: 404); its backstop (`fv-h3opts-backstop-…`)
  stopped by its exact PID; results fetched and checked (`gzip -t`, `tar -t`) before the delete. The staged bundle
  `fv-weights-staging/tmp/h3-tier-options-bundle-64d01601…` is removed by its exact key. Nothing was written to R2's
  weights bucket or to a volume.
