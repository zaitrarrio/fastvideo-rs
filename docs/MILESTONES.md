# Milestones

Chronological record of what shipped, what it cost, and what each step proved.

All times are **local (CDT, UTC−5)** to match `git log`. Run IDs under
`artifacts/gpucheck/runs/` are UTC, so `175630Z` is 12:56 here. For *why* a
choice was made rather than *what* changed, see
[decision-log.md](../decision-log.md) and [docs/adr](adr/).

## 2026-09-09 — Scaffold to first frames

| Commit | Milestone |
| --- | --- |
| `a47b2d0` | Phase 0 workspace scaffolded |
| `bd1aa2a` | Wan transformer, UMT5 and VAE land; PNG output |
| `d6d9dd5` | Candle CUDA inference for Vast.ai GPUs |
| `bfeb341` | Docker CUDA builder and Vast cargo cache |
| `6ae69d7` | fastvideo-ops integrated for tensor operations |
| `f697f5b` | I2V support |
| `ceead5b` | Benchmarking for video generation |

## 2026-09-17, 00:22–04:56 — cudarc becomes the primary path

| Commit | Milestone |
| --- | --- |
| `524e930` | Full Wan modules on the cudarc backend |
| `0b7a6ec`, `aef03dd`, `9aefb63` | GPU-optimized Wan pipeline: flash attention, BF16 FFN, fused kernels, plus an A100 smoke suite |
| `6dc5e8b` | A100 smoke suite retired in favour of a real check harness |
| `2c9cb1e` | **`fastvideo-gpucheck` crate** — staged, fail-fast GPU validation |

The harness shape that made everything afterwards cheap: rent the smallest
capable GPU, run stages in order, stop at the first failure, pull artifacts,
destroy the instance. Tiers are `kernels` (T1), `parity` (T2), `clip` (T3).

## 2026-09-17, 05:10–08:03 — Harness hardening, T1 and T2 go green

Harness work and paid runs interleaved, each failure feeding the next fix.

| Commit | Milestone |
| --- | --- |
| `feb9637`, `09794b3` | Docker and GPU scripts; cuDNN integration |
| `13e5d6a` | BF16 casting and kernel accuracy fixes |
| `0178839`, `e329d01` | **GHCR runtime image built in CI**; renting skips known-bad hosts |
| `198b663` | Device-fresh tensors read through `host_cow` in host fallbacks |
| `db9cb8f` | **Every test emits an mp4 and a generation time**; CPU reference runs in parallel |
| `a8ab946` | No flash attention above head dim 128 |

**T1 kernels tier** — seven runs to green:

| Run (UTC id) | Local | Result | Cost |
| --- | --- | --- | ---: |
| `101246Z` … `120552Z` (6 runs) | 05:12–07:05 | fail | $0.109 |
| `121607Z` | 07:16 | **pass**, 9 stages, 6 min | $0.007 |

**T2 parity tier** — real 1.3B weights against a CPU reference:

| Run | Local | Result | Cost |
| --- | --- | --- | ---: |
| `122337Z` | 07:23 | pass, 11 stages | $0.044 |
| `124601Z` | 07:46 | fail (RTX A4000) | $0.036 |
| `130350Z` | 08:03 | **pass**, 14 stages, 37 min | $0.112 |

Correct, but slow. Baseline on an RTX 5060 Ti:

| | exact |
| --- | ---: |
| DiT forward | 5.263 s |
| VAE decode | 27.570 s |
| UniPC 2-step | 18.642 s |
| UniPC video | 46.189 s |

## 2026-09-17, 11:19 — The performance pass

`7566769` — *device-only tensors, fused kernels, cuDNN conv3d, upstream
samplers.* The single largest change in the project:

- **Device-only tensors.** `CudaTensor` keeps a host copy only while it is valid; `pin_device` frees it. No host round trip inside a forward pass.
- **30 fused NVRTC kernels**, including `qk_norm_rope_bhsd`, `ln_adaln_e`, `residual_gate_add_e`, `gather_nd`, `block_copy`.
- **cuDNN N-D convolution** with a shape-keyed plan cache, plus a temporal-unfold alternative.
- **Upstream-aligned samplers** — DMD corrections and UniPC order-2.
- **No CPU fallback.** `stats::host_fallback` errors whenever a device is expected, so a missing device path fails loudly instead of quietly computing on the host.

Verified on the same RTX 5060 Ti (run `162416Z`, 11:24, 14 stages, $0.046):

| | before | after | speedup |
| --- | ---: | ---: | ---: |
| DiT forward | 5.263 s | 0.112 s | **47×** |
| VAE decode | 27.570 s | 0.186 s | **148×** |
| UniPC 2-step | 18.642 s | 0.446 s | **42×** |
| UniPC video | 46.189 s | 0.632 s | **73×** |

## 2026-09-17, 11:23–11:51 — Cheap runs, honest references

| Commit | Milestone |
| --- | --- |
| `c48b99f` | Broadcast-refusal tensor sized to its shape — a test bug that had failed a paid run |
| `2cd3161` | **CPU references cached by source hash.** The 648 s CPU parity reference is computed once per source key and restored on later runs; GPU-only files are excluded from the key, so kernel work never invalidates it |
| `5c03514` | **cuBLAS math probe tier**; the UMT5 embedding table stays on host |

## 2026-09-17, 11:50–12:03 — The cuBLAS 12.4 discovery

Fast mode was producing output **bit-identical** to exact mode, which meant bf16
was never running. The math probe tier (runs `165231Z`, `165241Z`) timed each
GEMM math option against FP32 and compared the results:

- **cuBLAS 12.4 predates Blackwell.** It silently ignored TF32 and BF16 compute types, and its FP32 was 2.6× slower than it should be.
- **cuBLAS 12.9 honours them**, and bf16 buffers give roughly 4× FP32 throughput.

`c5ca5a9` made fast mode store and multiply real bf16 buffers and required
cuBLAS 12.9, with bootstrap auto-installing it.

## 2026-09-17, 12:32 — Per-shape conv3d backend selection

`d70e13c` — cuDNN and temporal unfold each win on different hardware, different
shapes, and the ranking flips with the math mode:

| | exact | fast |
| --- | --- | --- |
| A5000, VAE shapes | unfold, ~18 ms vs 35 ms | cuDNN, ~8.5 ms (TF32) |
| Blackwell | cuDNN 33 ms vs unfold 53 ms | — |

The cache times both once per shape — keyed on shape *and* math flag — then
keeps the winner.

## 2026-09-17, 12:04–12:56 — T3 clip tier: 8-second videos

| Run | Local | Result |
| --- | --- | --- |
| `170404Z` | 12:04 | fail, 7 stages — cast-length test bug |
| `173304Z` | 12:33 | fail, 22 stages — every stage passed except the final fast-vs-exact compare |
| `175630Z` | 12:56 | **pass**, 22 stages, 19 min, $0.066 |

**The compare failure was a bad gate, not a bad kernel.** Its limits (rel_l2
0.15, 30 dB) were set while fast mode was silently FP32, so they had never been
measured against real bf16. The divergence was even across every latent frame
with no growth along the sequence, concentrated in high frequencies, and both
contact sheets showed the same scene and motion — trajectory drift, not a bug.

`9bb8bf6` split the gate to match the physics:

- **Step-1 latents, rel_l2 ≤ 0.05.** Both paths start step 1 from identical noise, so this isolates single-forward bf16 error. Measured: **0.022**.
- **Final latents and frames, loose.** Over 3 DMD steps that error compounds ~10× to 0.23 (23 dB); the limits are now 0.35 and 20 dB.

`eba8263` fixed clip retention: the per-stage pull excluded `clips/*/frames`,
where `output.mp4` lives, so clips only arrived at teardown and were lost if a
run died first. Clips now land in `artifacts/clips/<run>/` as each stage
completes.

### Final numbers — RTX A5000, real 1.3B weights

| | exact | fast | speedup |
| --- | ---: | ---: | ---: |
| DiT forward | 52 ms | 20 ms | 2.6× |
| VAE decode | 147 ms (131 dB) | 53 ms (74 dB) | 2.8× |
| UniPC 2-step | 214 ms | 85 ms | 2.5× |

**8-second clips** — 448×832, 129 frames, DMD 3 steps, fast mode, 48k tokens per
forward pass:

| clip | denoise | VAE | total |
| --- | ---: | ---: | ---: |
| beach_dog | 152.8 s (50.9 s/step) | 27.6 s | 197.1 s |
| city_rain | 153.2 s (51.1 s/step) | 27.7 s | 198.5 s |

About 24.5 s of compute per second of video, on a $0.20/hr GPU.

## 2026-09-17, 13:32 — Flash attention measured and rejected

The opt-in `flash_attn_f32` kernel had never been timed at clip scale, where
attention dominates a 43 s DiT forward. An A/B on a pinned A5000
(`FV_STAGE_ENV="FASTVIDEO_SDPA=flash"`, run `183200Z`) settled it:

| Tokens | Dense | Flash | Penalty |
| --- | ---: | ---: | ---: |
| 1,456 | 0.13 s | 1.64 s | 12.6× |
| 4,368 | 0.56 s | 11.8 s | 21× |
| 13,104 | 3.74 s | 134 s | 35.8× |

The penalty grows with sequence length — backwards for a flash-style kernel.
The cause is structural: the launch config gives one block per query row, so at
48k tokens 576,576 blocks each stream all of K and V through shared memory.
That trades the materialized score matrix (110 GB per layer) for K/V re-reads
hundreds of times larger. The budget gate stopped the run at $0.053, projecting
1,795 s per forward against 43 s for dense.

Flash stays off. Making it competitive means writing FlashAttention-2 properly
(query tiles, tensor-core MMA, double-buffered loads); the larger win is
porting upstream's sparse attention, which cuts the S² term instead of fighting
it. The cheap intermediate — bf16 probabilities in the dense path — is next.

## 2026-09-17, 14:06 — bf16 attention probabilities: 36% off an 8s clip

With flash ruled out, the target was the probability matrix dense attention
writes and reads back. Under bf16 GEMM math cuBLAS already rounds its F32
operands to bf16 for the tensor-core op, so storing probabilities that way is
the same arithmetic over half the bytes.

Measured as a same-instance A/B (one kept box, only the dtype differs):

| | F32 probs | bf16 probs | Gain |
| --- | ---: | ---: | ---: |
| Forward @ 1,456 tokens | 0.117 s | 0.111 s | 5.4% |
| Forward @ 4,368 tokens | 0.489 s | 0.414 s | 15.3% |
| Forward @ 13,104 tokens | 3.064 s | 2.304 s | 24.8% |
| **8s clip denoise** | **128.5 s** | **82.7 s** | **35.6%** |
| 8s clip, end to end | 161.9 s | 115.8 s | 28.5% |
| Peak denoise memory | 9,859 MiB | 9,539 MiB | — |

The gain grows with sequence length, exactly as the traffic argument predicts.
Accuracy did not move: parity fast rel_l2 0.004515 → 0.004498, exact mode
bit-identical, and every fast-vs-exact gate unchanged in character (step-1
0.0166 → 0.0180 against a 0.05 limit).

`FASTVIDEO_ATTN_PROBS_BF16=0` opts out. Two harness fixes came with it:
`FV_STAGE_ENV` to A/B a setting without touching code, and a disk precheck that
counts already-downloaded weights so a kept instance can be reused.

## 2026-09-17, 17:27 — Head to head with upstream FastVideo

A new `compare` tier runs our clip stages and then upstream FastVideo on the
**same rented box**, so the hardware is identical by construction rather than by
matching model names. Same weights, same 8s clip, same DMD timesteps.

| RTX 3090 Ti | Ours (dense) | Upstream (VSA) |
| --- | ---: | ---: |
| Generation | ~108.4 s | **55.3 s** |
| — denoise | 82.7 s | — |
| — VAE decode | 21.9 s | — |
| Model load (excluded) | 12 s | 49.9 s |
| First run (warm-up) | 82.7 s | 95.9 s |

Upstream is **1.96× faster**, and the gap is precisely the optimization we chose
not to port: VSA cuts the quadratic attention term, while our 48k-token forward
is dominated by dense attention.

Two details that matter for reading this honestly. Upstream's first generation
takes 95.9 s against 55.3 s steady-state, because Triton compiles its kernels on
first use — a single-shot benchmark would have reported upstream as *slower*
than us. And the like-for-like dense number does not exist on their side:
`TORCH_SDPA` cannot load this checkpoint at all, since FastWan ships VSA gate
weights (`to_gate_compress`) their dense model class does not define.

## 2026-09-18, 00:34 — Video Sparse Attention ported: 1.81x on an 8s clip

The head-to-head put upstream 1.96x ahead, and the gap was the optimization
we had not ported. So we ported it, reading their implementation rather than
inferring it: `(4,4,4)` tiles, a coarse stage that attends over tile means, a
top-k that picks which tiles the fine stage sees at full resolution, and the
checkpoint's gate scaling the coarse term.

| RTX 3090 Ti, 8s clip | Dense | VSA |
| --- | ---: | ---: |
| Denoise | 82.7 s | **45.6 s** |
| Per step | 27.6 s | 15.1 s |

Speed scales exactly as the algorithm predicts — VSA *loses* where its fixed
overhead outweighs the sparsity, and wins as the quadratic term takes over:

| Tokens | Dense | VSA |
| --- | ---: | ---: |
| 1,456 | 0.115 s | 0.227 s (2x slower) |
| 4,368 | 0.415 s | 0.412 s (parity) |
| 13,104 | 2.309 s | 1.931 s (16% faster) |
| 48,048 | 27.6 s | 15.1 s (**1.81x faster**) |

Two decisions worth keeping. The fine stage **gathers** selected K/V and runs
batched GEMMs instead of using a fused kernel: it reuses the tensor-core path
and still moves an order of magnitude less than dense attention's score matrix.
And the coarse stage is **pinned to F32** — tile selection is discrete, so
letting bf16 rounding flip a near-tie changes which tile is attended and moved
the largest test grid from rel_l2 0.003 to 0.043.

The load-bearing test: with every tile selected and a zero gate, the host
reference is exactly dense attention, so tiling, padding and the online softmax
are verified against an independent oracle before any kernel runs.

Quality: composition survives under the same seed, but VSA is a different sample
with more saturated colour, and that has not been checked against upstream's own
VSA output.

## 2026-09-18, 01:44 — VAE decode: 21.9 s to 17.8 s

With attention handled, VAE decode was the largest single cost in an 8s clip.
The convolutions turned out to be a dead end — the dominant one already runs at
this card's TF32 peak — so the wins came from memory traffic and from how the
decode is structured.

| 8s clip | VAE decode |
| --- | ---: |
| Baseline | 21.86 s |
| + SiLU folded into the RMS norm | 19.49 s (-10.8%) |
| + 2 latent frames per pass | **17.81 s** (-18.5%) |
| 4 latent frames per pass | out of memory |

Both changes are verified equivalent rather than argued: the fusion matches
norm-then-silu at rel_l2 8e-8, and chunking produces 129 frames with a clipped
fraction identical to eighteen digits.

Chunking helped less than expected. Thirty-three sequential passes suggested
launch overhead was the problem, but halving the passes bought only 8.6% — the
decode tracks the work, not the number of launches.

One near-miss worth recording: at four frames per pass the decode "finished" in
1.98 seconds, which would have read as a 10x win. It was an out-of-memory
failure part-way through. Removing 3.7x of the passes cannot produce 10x, and
disbelieving the number is what surfaced the error.

## Totals

- **42 validation runs**, **$1.79** of GPU time end to end.
- A full T3 tier — kernels, models, parity, text encoding, two 8s clips and a precision comparison — costs **$0.066** and 19 minutes.
- Cached CPU references save 648 s of billed CPU work per run.
