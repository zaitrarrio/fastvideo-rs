# Flux2 generate gap: rust cudarc vs upstream TORCH_SDPA

Investigation of why `fastvideo-rs` Flux2 (FLUX.2-klein-4B) cudarc generate is
~3.1× slower than upstream FastVideo `TORCH_SDPA` on the same Vast RTX 3090
workload (1024×1024, 4 steps, guidance 1.0). Measured after the I64 loader skip
(`bn.num_batches_tracked`) on `cursor/skip-safetensors-i64-2339`:

| Side | Load | Generate |
| --- | ---: | ---: |
| Upstream TORCH_SDPA | 70.4 s | median 7.57 s (min 7.55 s) |
| Rust cudarc | 52.0 s | **23.436 s** (~3.1×) |

Log: `sdpa device dense B=1 H=24 Sq=4608 Sk=4608 D=128 query_chunk=2427; gemm=Bf16 resident=true`.

This note does **not** pick a single root cause. It ranks several, with code
evidence. P0 (device DiT RoPE + cached tables + on-device Euler) is in tree;
re-run `compare-flux2` to measure the remaining gap. Do not treat the 23.4 s
figure as post-P0.

## 1. Timing scopes (apples-to-apples)

They measure the same pipeline window. Scope mismatch does **not** explain 3.1×.

### Rust `generate_ms`

`VideoGenerator::run_cudarc_flux2` times only `Flux2Pipeline::generate` after
weights are resident (`crates/fastvideo-core/src/generator.rs`). That function
does, in order:

1. CPU Gaussian noise → upload packed latents `[1, 128, 1, 64, 64]`
2. Tokenize + **Qwen3-4B text encode** (512-token pad, stack layers 9/18/27) unless `FASTVIDEO_FLUX2_DUMMY_TEXT=1`
3. **4 DiT forwards** (5 double + 20 single blocks) + **host Euler** (`host_cow` latents+velocity, `step_euler` on `Vec<f32>`, `from_vec` back)
4. Host `unpatchify_2x2`
5. **Full 2D VAE decode**
6. `write_frames`: device→host + one 1024×1024 PNG

`fastvideo bench` / `remote.sh flux2-rust-bench` is **one cold generate**. No
warmup, no explicit `cudaDeviceSynchronize` (PNG download is a sync).

### Upstream `median_seconds`

`scripts/gpu/upstream_bench.py` times `VideoGenerator.generate_video(...)` with
`save_video=True`, `torch.cuda.synchronize()` before/after. Load is excluded.
It does one warmup, then `FV_UPSTREAM_RUNS` (default 2) timed runs, and reports
the median.

`generate_video` is the same user-visible work: text encode → denoise → VAE
decode → save. `num_frames=1` for `--workload t2i`.

### What is *not* a 3× artifact

| Difference | Direction | Likely size |
| --- | --- | --- |
| Rust cold vs upstream post-warmup median | rust slower | tenths–2 s (first-use kernels / allocator), not 16 s |
| Explicit CUDA sync on upstream | upstream slightly more conservative | already in both via D2H / save |
| PNG vs FastVideo save | both write pixels | ≪1 s |
| Load times (52 s vs 70 s) | **outside** generate | ignore |

Klein 4B packed seq is `(1024/8/2)² = 64×64 = 4096` image tokens + 512 text =
**4608**, matching the SDPA log.

---

## 2. Existing timers (and what was missing)

Wan already has `StepTimer` spans (`generate`, per-step denoise, `vae.decode`)
and optional `FASTVIDEO_DEVICE_STATS=1` transfer dumps. **Flux2 generate had
none of that** — only the one-shot `generate_ms` and a once-per-process dense
SDPA line.

Profiling (PR #5) added Wan-style spans plus `profile.json` next to the PNG:

- `text_encode_ms`, `denoise_ms`, `steps_ms[]`, `euler_host_ms`
- `unpack_ms`, `vae_decode_ms`, `write_frames_ms`
- **RoPE**: `rope_host_apply_calls/ms/elems`, `rope_device_apply_calls/elems`,
  `rope_table_calls/ms`
- generate-window `h2d` / `d2h` counts and MiB, plus `host_fallbacks`

On a cudarc GPU generate, **`rope_host_apply_*` must be ~0** and
`rope_device_apply_calls` ≈ `2 × (num_layers + num_single_layers) × steps`
(200 for Klein 4B / 4 steps). A non-zero host apply counter means Q/K are
still leaving the device. CPU unit tests still take the host twin and increment
`rope_host_*`.

---

## 3. Flux2 cudarc path (what actually runs)

### Dense SDPA (confirmed)

`nn::scaled_dot_product_attention` defaults to `FASTVIDEO_SDPA=dense`:
strided-batched cuBLAS `Q@Kᵀ` + softmax + `P@V`, query-chunked so the score
buffer stays under 256M elements (`query_chunk = 256M / (B·H·Sk) = 2427` at
this shape). bf16 probabilities when `gemm=Bf16`.

The in-tree `FASTVIDEO_SDPA=flash` kernel is **not** a win. Wan clip A/B
(`docs/MILESTONES.md`, 2026-09-17) measured **12–35× slower** than dense
because it launches one block per query row and re-reads K/V. Do not flip
that flag for Flux2.

### DiT RoPE (P0 done — was the strongest code-level finding)

`apply_rotary` used to `host_cow` Q/K `[B,S,H,D]`, pair-rotate on CPU, and
`from_vec` back. Klein 4B: `2 (q,k) × (5 double + 20 single) × 4 steps = 200`
round-trips of `[1, 4608, 24, 128]` (~56.6 MiB each) plus `[4608, 128]`
tables — on the order of **20+ GiB** of PCIe and a sync before every
attention. That path also bypassed `stats::host_fallback`.

**Now:** `CudaTensor::apply_rotary_bshd` launches the NVRTC sibling
`apply_rotary_bshd` (rotate half of Wan `qk_norm_rope_bhsd`, Flux2 layout:
BSHD in/out, even-slot cos/sin). Host twin lives in `ops::host` for CPU
tests and `fv-gpucheck`. Wan’s fused kernel is **not** reused as-is: it
RMSNorms over `heads*d` and writes BHSD with sin from the odd slot. Flux2
RMSNorms per head (`rms_heads`) and stores `repeat_interleave(2)` tables.

`flux2_rope` tables are built once per `(text_len, img_h, img_w)` on
`Flux2Transformer2D` and pinned. `rope_table_calls` should be **1** per
generate, not 4.

### Euler (P0 small extra)

Was: each step downloaded packed latents + velocity (`[1,128,1,64,64]` ≈
2 MiB each) for `step_euler` on `Vec<f32>`. Now: `take_euler_dt` +
`CudaTensor::lincomb` (`x + dt * v`) on device. `euler_host_ms` is launch
time only (no D2H/H2D); expect ~0 on GPU. Candle generate still uses host
`step_euler`.

### Text (Qwen3-4B, once)

36 layers, hidden 2560, GQA 32/8, 512 tokens. Device GEMMs + SDPA, but:

- `apply_rope_neox` builds cos/sin on the host every layer
- `causal_pad_mask` uploads a fresh `[1,32,512,512]` mask every layer (~32 MiB × 36)
- `stack_selected` downloads hidden states 9/18/27 and stacks on the host

This runs **once** per generate. It is inside both rust `generate_ms` and
upstream `median_seconds`. Extra rust cost is overhead, not “text is missing
from upstream.”

### VAE decode (inside generate, usually not dominant)

Full 2D decoder: `post_quant` → mid ResNets + **spatial attn** → up-blocks +
nearest/conv upsample. Mid attention is implemented as **one head, D=512**
at 128×128 (`Sq=Sk=16384`):

```203:214:crates/fastvideo-cudarc/src/flux2/vae.rs
        let q = self.to_q.forward(&seq)?.reshape(vec![b, 1, h * w, c])?;
        // ...
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
```

That is a second, larger dense SDPA than the DiT line in the log
(`info_once` only prints the **first** SDPA, which is DiT). Compute is ~0.5
TFLOP once — plausible 0.5–2 s, not 16 s. Tiling/slicing is not ported
(upstream has it, off by default).

---

## 4. Upstream FastVideo Flux2

Upstream `fastvideo/models/dits/flux_2.py`:

- `LocalAttention` → selected backend. `TORCH_SDPA` is
  `torch.nn.functional.scaled_dot_product_attention` after a BHSD transpose
  (`fastvideo/attention/backends/sdpa.py`).
- On Ampere (3090, sm_86) with **no mask** and `head_dim=128`, PyTorch 2.x
  picks FlashAttention and/or mem-efficient SDPA. It does **not** materialize
  a 4608×4608 score matrix the way rust dense does.
- RoPE is `apply_rotary_emb` (GPU).
- Double-stream QKV and single-stream `to_qkv_mlp_proj` stay on device;
  single-stream QKV+MLP is already fused (rust matches that fusion).
- `_compile_conditions` exist on the DiT config (torch.compile eligible).
- VAE is the same AutoencoderKLFlux2 graph; tiling exists but is off.

Rust GEMM path is already bf16-resident (`gemm=Bf16 resident=true`), so this
is not “we are in FP32 and they are not.”

---

## 5. Ranked bottlenecks

Estimates assume the 15.9 s gap (23.4 − 7.6) is almost all inside generate.
FLOPs alone do **not** explain 23 s: four Klein forwards are ~140 TFLOP;
a 3090’s 142 TFLOP/s bf16 peak would finish the math in a few seconds at
even modest efficiency. The gap is **host syncs, traffic, and unfused
launches**, not missing FLOPs.

| Rank | Cause | Evidence | Expected share of the 23.4 s | Notes |
| ---: | --- | --- | --- | --- |
| 1 | **Host DiT RoPE** (`apply_rotary`) — **fixed in P0** | Was 200 D2H/H2D of `[1,4608,24,128]`. Now `apply_rotary_bshd` on GPU; tables cached. | **~3–8 s** plus GPU idle (hypothesis; not re-measured here) | Confirm with `rope_host_apply_ms ≈ 0` and `rope_device_apply_calls = 200`. |
| 2 | **Dense SDPA vs PyTorch fused SDPA** at 4608 | Log: device dense, chunk 2427; upstream `F.sdpa` flash/mem-efficient | **~1–4 s** of remaining DiT time | Same FLOP count; rust pays HBM for ~1 GiB scores × 2 chunks × 100 attentions. In-tree `flash` is **worse** (12–35× on Wan). |
| 3 | **Unfused double-stream QKV + kernel-launch tax** | 6 separate linears + `cat` + transpose + narrow per double block; many 1-op kernels vs PyTorch fused graphs / compile | **~2–5 s** | Shows up as leftover `denoise_ms` after P0. |
| 4 | **Qwen3 encode overhead** | Host NeoX tables + 36 mask uploads + host layer stack | **~1–3 s** extra vs HF | Inside both timings; rust-only waste is the host tables/masks. |
| 5 | **VAE mid attn as H=1,D=512 @ 16k tokens** | First SDPA log hides this; one 16384²×512 dense | **~0.5–2 s** | Once per generate. |
| 6 | **Cold bench vs warmed median** | rust 1 run; upstream warmup + median of 2 | **~0.5–2 s** | Re-bench rust with a discarded warmup to isolate. |
| 7 | Euler + unpack + PNG | Device Euler in P0; unpack + one PNG remain | **≪0.3 s** | Real, not ranked as the gap. |

Discarded as *the* explanation: “generate includes VAE/text and upstream
doesn’t” — both include them. “Need `FASTVIDEO_SDPA=flash`” — measured
regression on this stack.

---

## 6. Fix plan

### P0 — done (device RoPE + cached tables + device Euler)

1. **Device Flux2 RoPE.** `apply_rotary_bshd` pair-rotates `[B,S,H,D]` with
   `[S,D]` tables on GPU. Host twin + `fv-gpucheck` parity. **Expected: 3–8 s
   off, maybe more** once attention no longer syncs 200 times — verify on Vast,
   do not quote the 23.4 s row as post-fix.
2. **Profile kept.** `rope_host_*` stays; `rope_device_apply_*` added.
3. **Device Euler** via `take_euler_dt` + `lincomb`. Tiny; removes a sync.

Re-run compare (same box / prompt / steps as the 23.4 s row):

```bash
scripts/gpu/validate.sh offers compare-flux2
scripts/gpu/validate.sh run compare-flux2
```

Read `remote/flux2-rust/profile.json` (and `flux2.profile` in the rust log).
Overrides: `FV_FLUX2_STEPS=4`, `FV_FLUX2_HEIGHT=1024`, `FV_FLUX2_WIDTH=1024`.
Do **not** set `FASTVIDEO_SDPA=flash`.

### P1 — after profile says RoPE is gone and denoise is still ~2×

1. **Do not enable in-tree flash.** Write a real FA-2 / cuDNN SDPA, or bind
   PyTorch’s mem-efficient path, if `denoise_ms` is still SDPA-bound at 4608.
2. Fuse double-stream `to_q/k/v` and `add_q/k/v` (Wan already has
   `Linear::load_fused`).
3. Text: build the causal mask once; move NeoX tables to device; `cat` stacked
   layers on device.
4. Add a rust warmup (or `bench --runs N`) so the headline number matches
   upstream `median_seconds`.

### P2 — later / quality

1. VAE mid attention as multi-head (D=64/128), not one 512-wide head.
2. VAE tiling only if 1024 decode shows up in `vae_decode_ms`.
3. torch.compile-equivalent fusion is out of scope; prefer kernels you already
   have.

### How to read the next Vast artifacts

`remote/flux2-rust/profile.json` (and the `flux2.profile ...` log line):

- `rope_host_apply_calls` / `rope_host_apply_ms` ≈ 0 and
  `rope_device_apply_calls` = 200 → P0 RoPE is on device (expected after this
  change).
- `rope_host_apply_ms` still seconds → Q/K are leaving the device; regress.
- `rope_table_calls` should be 1 (cached). >1 means a size change or cache miss.
- `rope_host_apply_ms` ≈ 0 but `denoise_ms` still ≈ 18 s+ → SDPA / GEMM launch (P1).
- `text_encode_ms` ≈ 5 s+ → text host masks/tables.
- `vae_decode_ms` ≈ 5 s+ → mid-attn / conv, not DiT.
- `euler_host_ms` ≈ 0 → device lincomb (no latent D2H). Non-zero + matching
  `d2h_mib` means Euler fell back to host.

`generate_ms` will stay inclusive of text+VAE+PNG; compare
`denoise_ms + text_encode_ms + vae_decode_ms` to upstream’s 7.57 s, not load.
