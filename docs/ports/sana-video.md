# SANA-Video 2B (480p) port

Status (2026-10-06): **code complete, CPU-tested, GPU-unmeasured.** No weights
on the EU volume yet. HunyuanVideo-13B was dropped from this work by owner
decision on 2026-10-06.

## What Sol-Engine publishes

| | |
|---|---|
| Model | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` (Apache-2.0) |
| Claim | ~2.77x warm end to end, EasyCache 0.1 + linear-attention BF16 + QKV merge + `torch.compile` (README; `site_docs/pipelines/sana.md`) |
| Canvas | 832x480, 81 frames, 50 steps, one GB200 |
| Not published | the run script it names (`scripts/sana/sana_video_sglang_run.py` is not in the branch), absolute seconds, guidance scale, EasyCache warmup / cooldown |

**Mismatch with the Sol-Engine profile.** `models/sana_video` on the branch
is a *5B* model in a private bundle (`yitongl/sana_video`: Qwen text encoder,
LTX-2.3 VAE, 1280x736x193, cfg 8, shift 12, measured 313 s on 4 GPUs). That
model is not public. We port the public 2B, which is what the 2.77x claim is
about.

## Architecture (Diffusers reference, `transformer_sana_video.py`)

- **DiT**, 20 blocks, dim 2240 (20 x 112), patch (1, 2, 2), in/out 16:
  AdaLN-single timestep modulation (`scale_shift_table` + `t_mod`);
  **ReLU linear self-attention** (q/k RMS-normed across heads, eps 1e-5;
  RoPE on the ReLU features; normaliser from the *unrotated* features:
  `out = q_rot (k_rotᵀ v) / (q · Σk + 1e-15)`, f32); softmax cross-attention
  to the caption (no norm, no gate); **GLUMBTempConv** FFN (1x1 to 2x6720,
  SiLU, depthwise 3x3, GLU, 1x1 to 2240, then `x + conv_temp(x)` over frames
  with a (3, 1) kernel).
- **RoPE**: Wan-style 3 axes, head 112 = 40 (frames) + 36 + 36, theta 1e4,
  float64 tables, interleaved pairs.
- **Text**: Gemma-2-2B-it (`Gemma2Model`, last hidden state after the final
  norm), with the "complex human instruction" prefix on the positive prompt,
  `select_index = [0] + range(-299, 0)`, cross-attention masks the padding.
  Gemma-2 needs **attention logit soft-capping** (cap 50) - added to
  `crate::llm` as `DecoderConfig::attn_softcap` (None for every other encoder).
- **VAE**: `AutoencoderKLWan`, the Wan 2.1 VAE. The safetensors file is
  byte-identical to `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae` (LFS sha256
  `d6e524b3…ed19e793`, 507 591 892 B), so our Wan decoder is reused as is.
  (The 2026-09-25 gap doc said DC-AE; corrected.)
- **Sampler**: `DPMSolverMultistepScheduler`, dpmsolver++, order 2, midpoint,
  flow prediction, `use_flow_sigmas`, flow_shift 8, final sigma 0,
  lower_order_final. Timesteps are `int64(sigma * 1000)`. CFG 6 by default.

## Code

| Piece | Where |
|---|---|
| configs (parsed from the tree), schedule plan, rope, prompt layout, arm switches, key spec | `crates/fastvideo-models/src/sana_video/` |
| DiT, Gemma-2 encode, pipeline (CFG, sampler, EasyCache, Wan VAE) | `crates/fastvideo-cudarc/src/sana_video/` |
| softcapped GQA attention | `crates/fastvideo-cudarc/src/llm/attn.rs` |
| benchmark stage | `fv-gpucheck sana-video {info,gen}` (`crates/fastvideo-gpucheck/src/sana_video_stage.rs`) |
| benchmark arm | `scripts/gpu/runpod-matrix.sh sana-video` |

Batch is 1: the two CFG branches are two transformer calls (the reference
batches them; a batched path is a later optimisation).

### Optimized arm (`--arm full`, or `FASTVIDEO_SANA_OPT=full`)

| Sol-Engine | Here |
|---|---|
| EasyCache 0.1 | `fastvideo_models::wan::sol_cache::EasyCache`, threshold 0.1, retain 7, cooldown 1 (Sol-Engine's own EasyCache runtime defaults; the SANA values are unpublished). Reuses the guided output residual `noise_pred - x`. |
| QKV merge | one fused GEMM for self-attention q/k/v (exact; tested equal to the separate path) |
| linear-attn BF16 | q_rot, k_rot, v rounded to bf16 before the f32 products (numerics only; no bf16 kernel yet) |
| torch.compile | no Rust counterpart; the DiT glue is hand-fused in both arms |

Overrides: `FASTVIDEO_SANA_EASYCACHE=<thr|0>`, `FASTVIDEO_SANA_QKV_MERGE=0|1`,
`FASTVIDEO_SANA_LINATTN_BF16=0|1`.

## Tests (CPU, `cargo test -p fastvideo-models -p fastvideo-cudarc --lib -- sana_video`)

- Hub config parsing (transformer, Gemma-2) and refusal of unported variants.
- **Key mapping**: every tensor the loaders read, with shape, equals the real
  Hub safetensors headers (transformer 496 tensors, text encoder 288), from
  `fixtures/hub_*_keys.tsv` fetched by HTTP range request; the device loaders
  read exactly that spec (recording weight map, tiny config, both arms).
- Sampler: endpoints of the 50-step table, order sequence (1, 2 x 48, 1),
  final step = data prediction, exact integration of a straight flow.
- RoPE layout, CHI / `select_index` layout, aspect-ratio bins.
- Linear attention against a direct per-token evaluation of the formula.
- Tiny random-weight DiT forward (finite, shaped) and QKV merge == separate.
- Tiny Gemma-2 encode through the streamed decoder; soft-capping limits.

## Weights to download (owner approval needed)

| dest (EU volume) | repo @ revision | globs | size |
|---|---|---|---|
| `sana-video-2b-480p` | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` @ `db5f398b13ca086d09a50ce156c20527773841b1` | `model_index.json scheduler/* tokenizer/* text_encoder/* transformer/* vae/*` | 18.0 GB (transformer 8.23 GB f32, text encoder 5.23 GB bf16, vae 0.51 GB, tokenizer 0.04 GB) |

Rows are in `scripts/gpu/weights-manifest.tsv` / `weights-revisions.tsv`;
check with `verify-weights.sh sana-video-2b-480p`. The vae/ subtree could be
skipped and `--vae /workspace/weights/wan21-t2v-14b/vae` passed instead
(identical file), but the tree is self-contained by default.

**Expected path on the pod:** `/workspace/weights/sana-video-2b-480p`
(override with `FV_SANA_WEIGHTS`).

## Benchmark arm (ready, not run)

From the CI runtime image (no compile, no download on the pod):

```bash
bash scripts/gpu/runpod-matrix.sh sana-video        # cells sana-baseline, sana-full
# one cell by hand:
fv-gpucheck --mode fast sana-video gen --weights /workspace/weights/sana-video-2b-480p \
  --prompt "a corgi running on the beach" --arm full --seed 42 --warm --clip out/frames
```

Each cell runs one cold and one timed warm generation; `benchmark.json` next to
the frames has `generate_s` (text encode through decode, frame writes
excluded: Sol-Engine's timing scope), per-step seconds, `reused_steps` and peak
MiB. The two cells are compared frame by frame (`compare_cells`). Speedup to
report: `sana-baseline.generate_s / sana-full.generate_s`, against 2.77x.

**Estimated cost on RTX PRO 6000 ($2.09/h):** the 2B DiT is ~130 TFLOP per
forward at 32 760 tokens, 100 forwards per clip; at a realistic 30-50 % of
peak for this port that is 2-5 min per clip. Two generations per cell plus
load: ~10-15 min (baseline), ~8-12 min (full); plus pod start. **Budget
~45 GPU minutes (~$1.60), hard cap 2 x 40 min (`FV_GEN_TIMEOUT_S=2400`).**

## GPU parity plan (later)

1. Run the Diffusers pipeline (bf16 DiT, as the model card) once with
   `output_type="latent"` and hooks dumping: Gemma last hidden state and the
   selected caption tokens, the initial `randn` latents, `transformer` output
   at steps 0, 1, 25, 49, the sampler sigmas/timesteps, the final latents and
   the decoded frames.
2. Feed the same noise (`--noise <raw f32>`) and compare stage by stage:
   caption tokens (cosine > 0.999), step-0 velocity (rel L2 < 1e-2 bf16),
   final latents, then frames by PSNR / LPIPS (`fv-gpucheck compare-clips`).
3. Check the CHI token count and `select_index` on the real tokenizer, and that
   the sampler table matches the reference's 51 sigmas exactly.
4. Then the full arm vs baseline: reuse count and LPIPS, as Sol-Engine gates.
