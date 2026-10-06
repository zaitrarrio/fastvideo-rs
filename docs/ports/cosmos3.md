# Cosmos3-Super — port specification

FastVideo family `cosmos3`: NVIDIA Cosmos3-Super, a 64B omnimodal
Mixture-of-Transformers; this port is its **text-to-video** path, single GPU.
Distinct from Predict2 Video2World (`docs/ports/cosmos.md`, EDM I2V).

Sources (read 2026-10-06):

- Hub `nvidia/Cosmos3-Super` @ `f543c56225b2e04d0ad141e29655be3a45d9c455`
  (OpenMDW 1.1, not gated): `transformer/config.json`, the 1430-key shard
  index, the headers of shards 1 and 27, `scheduler/scheduler_config.json`,
  `vae/config.json`, `text_tokenizer/` template — vendored in
  `crates/fastvideo-models/src/cosmos3/hub/` (configs, key list).
- diffusers `main` (2026-10-06, commit bc5e3bd): `transformer_cosmos3.py`,
  `pipeline_cosmos3_omni.py`, `scheduling_unipc_multistep.py`. The sol-engine
  runtime (SGLang, `b0b7eb4d0`) is not public; its profile
  (`models/cosmos3.toml`, `models/cosmos3/{baseline,optimized}/env.sh`) fixes
  the official config and the arms.

---

## Architecture

`Cosmos3OmniTransformer`, 64 layers, hidden 5120, 64 query / 8 KV heads of
128, SwiGLU 25600, RMS eps 1e-6. Every layer has **two towers** over one
packed sequence `[und (text) | gen (vision)]` — "use_moe" in the config is this
modality routing, not routed experts:

| | und (text) | gen (vision) |
|---|---|---|
| norms | `input_layernorm`, `post_attention_layernorm` | `…_moe_gen` |
| q / k / v / out | `to_q/k/v`, `to_out` | `add_q/k/v_proj`, `to_add_out` |
| q/k norm (per head) | `norm_q/k` | `norm_added_q/k` |
| attention | **causal**, text keys only | full, **text keys + vision keys** |
| MLP | `mlp` | `mlp_moe_gen` |

Embeddings: text ids through `embed_tokens`; vision latents (Wan 2.2, 48 ch,
×4 time, ×16 space) padded to even H/W and packed in 2×2 patches
(`cthpwq → thwpqc`, 192 → `proj_in`), plus `time_embedder(time_proj(t·0.001))`
on every noisy token. Output: `norm_moe_gen` → `proj_out` → unpatchify.

Rotary: unified 3D M-RoPE, theta 5e6, sections 24/20/20 **interleaved**
(frequency `j` reads H when `j ≡ 1 mod 3, j < 60`, W when `j ≡ 2 mod 3,
j < 60`, else T), rotate-half. Text positions `(i, i, i)`; the vision time
axis starts at `und_len + 15000` with fps modulation (`f / (fps/4) · (24/4)`,
i.e. `f` at 24 fps), H/W restart at 0.

Unused by T2V: `lm_head`, the und `norm`, the sound (`audio_*`) and action
(`action_*`) heads, `vision_encoder/`, `sound_tokenizer/`.

### The text-K/V cache (exact)

The text tower never attends to vision tokens, so its per-layer K/V for a
prompt are constant across denoising steps. `Cosmos3Transformer::und_caches`
runs it once per request (cond and uncond in one pass over the layers) and
each step runs only the 31.2B **gen** tower against the cached keys. This is
mathematically identical to the reference's per-step joint pass (the golden
test runs the joint pass) and halves the per-step weights touched.

## Text, schedule, VAE

| piece | notes |
|---|---|
| prompt | `"{prompt}. The video is 7.9 seconds long and is of 24 FPS. This video is of 720x1280 resolution."` (negative: the "is not" forms, `""` by default), Qwen chat template with the video system prompt and a generation turn, then `<|im_end|>`, `<|vision_start|>` (`fastvideo-models::cosmos3::prompt`) |
| tokenizer | `text_tokenizer/tokenizer.json` (Qwen2) |
| schedule | Hub `UniPCMultistepScheduler`: **Karras** flow sigmas (σ_min 0.147, σ_max 200, ρ 7 → σ/(σ+1)), bh2, order 2, predict-x0, final 0; timesteps `int64(σ·1000)`; the transformer sees `t·0.001` |
| flow_shift | the HF example and `models/cosmos3.toml` pass `flow_shift=10.0`; under `use_karras_sigmas` Diffusers' `set_timesteps` never reads it, so it is inert here too (risk: the SGLang runtime may differ) |
| VAE | `AutoencoderKLWan` Wan 2.2 (z 48, residual, patch 2), bf16 on the Hub (not byte-identical to the Wan2.2-TI2V-5B f32 file, so it is downloaded) |

## Sol-engine arms

| arm | upstream (`env.sh`) | here |
|---|---|---|
| baseline | 35 steps, seams off | `--arm baseline` |
| fullopt | TeaCache `teacache_c115_s10_m3` + step-selective NVFP4 (gate_up, down, qkv, out; first and last 3 steps dense) | `--arm teacache` (`SolCosmosTea`: threshold 1.15, start 10, ≤ 3 in a row, signal = the step's time embedding, residual of the gen layer stack per CFG branch); precision via `FASTVIDEO_FP8=1` (W8A8 tensorwise, **all** steps) as the cell `teacache-fp8` |

NVFP4 is not ported as a speed path: this repo's NVFP4 linear is a numerics
emulation (dequant beforehand), so a step-selective NVFP4 arm would measure
nothing. FP8 W8A8 on Blackwell FP8 tensor cores is the nearest real
precision speedup; it differs from the published recipe and is labelled so.

## Memory (one GPU)

| GPU | plan |
|---|---|
| B200 (192 GB) | `FASTVIDEO_COSMOS3_UND=resident`: both towers resident (2 × 62.4 GB bf16) |
| RTX PRO 6000 (96 GB) | default: gen tower resident (62.4 GB bf16), text tower **streamed** layer by layer from the mapped checkpoint once per request (31 GB read; page cache after the first). Activations at 44 160 vision tokens: ≈ 1 GB hidden, ≈ 5 GB attention (K/V repeated to 64 heads), MLP chunked (`FASTVIDEO_COSMOS3_MLP_CHUNK`, 8192 tokens) |

No FP8 / NVFP4 is needed to fit either card.

## Comparison framing

Published: **4× GB200**, sequence parallel, warmup excluded: baseline
130.41 s (denoise 121.42, decode 5.80), fullopt 2.26×. Ours: one GPU, same
canvas / steps / guidance / prompts (`scripts/gpu/sol/cosmos3-*.txt`), one
warmup request, then the timed request. Report `request_s` and `denoise_s`
beside theirs and as GPU-seconds (130.41 × 4 = 521.6 GPU-s), plus the
baseline / teacache / teacache-fp8 ratios against 2.26× (theirs includes
NVFP4; ours FP8, so a lower precision gain is expected).

## Tests (CPU)

- `fastvideo-models::cosmos3::config` — preset = Hub config; T2V keys +
  the listed unused keys = the 1430-key index; shapes = shard headers;
  tower size 31.2B.
- `rope` — interleave slices, official positions (48×23×40 grid, offset
  `und_len + 15000`), fps modulation, text tables = plain RoPE.
- `schedule` — Hub scheduler flags, the 35-step Karras flow sigmas.
- `prompt` — the templates.
- `fastvideo-cudarc::cosmos3::transformer::golden_tiny_matches_numpy_reference`
  — tiny random MoT vs `scripts/ref/cosmos3_reference.py`, a NumPy
  transcription of diffusers' joint forward: max |Δ| < 2e-4. Plus the text
  cache = streamed = resident, causality, batched = single passes, TeaCache
  reuse reproduces the residual.

## Weights

On the EU volume `jg48s6o1w0` since 2026-10-06 (PR #35): `weights-manifest.tsv`
row `cosmos3-super` @ `f543c56225b2e04d0ad141e29655be3a45d9c455`
(`weights-revisions.tsv`, hashes in `weights-sha256.tsv`): `transformer/`
128.04 GB, `vae/` 1.41 GB, `text_tokenizer/` + `scheduler/` ≈ 0.02 GB =
**129.47 GB**. Verify cell `cosmos3-super`.

## GPU benchmark arm (not run)

`FV_FAMILY=sol-cosmos3 scripts/gpu/runpod-http.sh run <sha>` (runpod-matrix
family `sol-cosmos3`): cells `cosmos3-baseline`, `cosmos3-teacache`,
`cosmos3-teacache-fp8` (`FV_COSMOS3_ARMS`), each `fv-gpucheck sol
cosmos3-gen --warm`. B200: add `FV_EXTRA_ENV="FASTVIDEO_COSMOS3_UND=resident"`.
Estimates in docs/perf/sol-lingbot-cosmos3-plan.md.

## Port status

| layer | status |
|---|---|
| config / key map / shapes vs Hub | landed, CPU-tested |
| MoT DiT (both towers, GQA, M-RoPE, patching) + text-K/V cache | landed, golden-tested |
| prompt templates + tokenizer | landed (tokenizer file read at run time) |
| UniPC Karras flow schedule | landed |
| Wan 2.2 VAE decode | reused (`wan::vae22`) |
| TeaCache arm; FP8 cell | landed |
| CLI registry (`fastvideo generate`) | **not wired** (benchmark goes through `fv-gpucheck sol`) |
| I2V / V2V, sound, action, multi-GPU SP | out of scope |
| GPU run | **not run** |
