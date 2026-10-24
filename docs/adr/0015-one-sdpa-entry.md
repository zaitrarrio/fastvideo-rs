# ADR-0015: One `sdpa(q, k, v, &Plan)` entry, `sdpa_rule.rs` the one chooser

- Status: accepted (plan B5, `docs/plan/2026-10-23-execution-plan.md`)
- Date: 2026-10-24
- Source: `docs/gaps/2026-10-22-architecture-review.md` §4.2 "Duplication on
  the hot path" and "Host twins", §5.1, §5.2, §7.3 items 4 and 6

## Context

The review counted "seven or more dense-attention backends (25 `sdpa*` /
`attn*` functions)" and "five NVFP4 modules plus the oxide crate for a feature
that is off". The inventory below, taken on `326c879a` (main before B5),
separates three things the count mixed: the **entry points** model code calls
(where a backend is chosen), the **kernels** they launch (one function per
kernel, which stays), and **attention algorithms that are not dense SDPA**
(sparse, linear, GQA, neighbourhood), which have their own kernels and are out
of this ADR's scope except where they call the dense entry.

### Entry points before B5 (where the choice was made)

| Entry | File | Callers (families) | Chose between | dtype / layout |
|---|---|---|---|---|
| `nn::scaled_dot_product_attention(q,k,v,scale)` | `rt/nn.rs` | cosmos, cosmos3, flux, flux2 (DiT, VAE), H3 (vision, VAE unfused path, VSA text rows), hunyuan15, kandinsky5, lingbot, longcat, LTX-2 (`VideoAttnKernel::Dense`), mmaudio, sana_video, Sol / PISA dense splices, Wan (CLIP, VAE, AR cache); gpucheck | `FASTVIDEO_SDPA` (`sparse` / `host` / `flash` / `cublas` / dense), Sage (recipe or `FASTVIDEO_ATTN_SAGE`, `MIN_SEQ`), fused mma, cuBLAS, composed; sequence-parallel shards | BHSD, f32 or bf16 (rounded to bf16 under bf16 activations, output bf16 then) |
| `nn::scaled_dot_product_attention_masked(..., mask)` | `rt/nn.rs` | the above with `None`; additive masks in flux / kandinsky5 text, hunyuan15 / kandinsky5 VAE, mmaudio CLIP, the LLM encoders' host GQA path | as above; a mask goes straight to composed | BHSD + `[.., sq, sk]` additive f32 |
| `h3::transformer::dense_attention` | `h3/transformer.rs` | H3 dense blocks | FP8 `Q K^T` (`attn_fp8`, `fp8_attention`) first, then the entry above | BHSD bf16 |
| `nn::sdpa_block_causal(..., BlockCausal)` | `rt/nn.rs` | Wan causal (SF-Wan, LongLive) | `flash_mma_fwd2_causal_d*` (`FASTVIDEO_WAN_CAUSAL_FLASH`, SP world 1, fused context) else the dense mask through the masked entry | BHSD, d 64 / 128 |
| `nn::sdpa_vdn_window(..., VdnWindow)` | `rt/nn.rs` | VDN-H3 | `flash_mma_fwd2_window_d128` else the dense mask (small shapes only) | BHSD, d 128 |
| `nn::sdpa_kv_window` / `_with(flash)` | `rt/nn.rs` | Wan causal AR (KV-cache window) | the dense entry, or the composed f32 reference (`FASTVIDEO_WAN_CAUSAL_FLASH=0`) | BHSD |
| `attn::device_mma_sdpa` + `flash_kernel_for`, `flash_kernel_choice`, `dc_default`, `cudnn_default`, `auto_cudnn_or_v2`, `timed_cudnn_or_v2` | `rt/attn.rs` | the dense entry; the H3 ViT decoder's fused chain; gpucheck | `[kernels] dense_attention` / `FASTVIDEO_FLASH_KERNEL` (V1 / V2 / V3 / V3s / cuDNN / attn_dc), V2-fills-the-grid, the sm_12x cuDNN-or-fwd2 rule (`sdpa_rule::pick`, cuDNN plan probe), `FASTVIDEO_SDPA_AUTO=timed` | bf16 operands, f32 or bf16 out |
| `h3::vae::sdpa_kernel` (`FASTVIDEO_H3_VAE_SDPA`) | `h3/vae.rs` | H3 ViT decoder (fused) | its own kernel override, else `device_mma_sdpa` | bf16 |
| `flux2::text::text_sdpa` | `flux2/text.rs` | FLUX.2 text encoder | a copy of the composed SDPA (no rounding, no shards) | f32 + causal / pad mask |
| `ops::scaled_dot_product_attention` | `rt/ops.rs` | none (a wrapper of the entry) | — | — |
| `attn::flash_style_sdpa_host` | `rt/attn.rs` | `FASTVIDEO_SDPA=host` | host online softmax (host only: refused with a live device) | f32 host |
| `attn::block_sparse_sdpa` | `rt/attn.rs` | `FASTVIDEO_SDPA=sparse` (+ `FASTVIDEO_VSA_WINDOW`) | host window prototype (host only: refused with a live device) | f32 host |

### Kernels (one function each; unchanged)

`attn::device_mma_sdpa_with` (V1 `flash_mma_fwd_d{64,128}`, V2
`flash_mma_fwd2_d{64,128}`, V3 / V3s `flash_mma_fwd3{,s}_d128`, cuDNN
`cudnn_sdpa::sdpa_bf16`, attn_dc `attn_dc::dense_fwd` on sm_90 / sm_100),
`attn::device_mma_sdpa_causal`, `attn::device_mma_sdpa_window`,
`attn::device_dense_sdpa(_with_budget)` (cuBLAS, query-chunked),
`attn::device_flash_sdpa` (`flash_attn_f32`), `attn_sage::dense_sdpa` (INT8 QK /
FP8 PV), `attn_fp8::dense_sdpa`, the composed tensor-op SDPA. The kernel checks
in `fv-gpucheck` (`flash_mma`, `attn2`, `attn3`, `sdpa_rule`, `attn_fp8`,
`attn_sage`, `wan_causal_attn`, `vdn`, `h3vae`, `determinism`) call these
directly, which is what they are for.

### Not dense SDPA (own algorithms; out of scope)

Sparse: `rt::vsa` (Wan VSA device kernels + `vsa_attention_host` oracle),
`h3::vsa::H3Vsa` (+ `attention_host` oracle), `rt::sla::sla_attention` (+ host
oracle), `sol_attn` / `rt::sol_ops` (Sol-Attn), `pisa_attn`, `longcat::bsa`,
`attn_fp8::vsa_fine`, `attn_fp4::vsa_fine`, `attn_dc::vsa_*`. Their dense parts
(Sol / PISA splices, VSA text rows, BSA's dense fallback) call the dense entry.
Others: `llm::attn` GQA (`device_gqa_masked`; its host fallback calls the dense
entry), the Gemma-2 softcap attention, SANA's linear attention, LTX-2.5's
diffusion-decoder neighbourhood attention (`neighborhood_attn_host`, host
only, opt-in `--diff-vae`, no device kernel), H3's audio VAE causal pre-block
(host), `fastvideo-ops::TensorBackend` (dead, B6), and the plain-Rust oracles in
`fastvideo-models` (`sol_attn`, `pisa_attn`) and gpucheck (`ref_sdpa`,
`sdpa_host`).

## Decision

1. **One entry**, `rt::sdpa::sdpa(q, k, v, &Plan)` (`crates/fastvideo-cudarc/src/rt/sdpa.rs`).
   A `Plan` says what the call site computes: `Plan::dense(scale)`,
   `Plan::masked(scale, mask)`, `.fp8_first()` (the H3 dense blocks),
   `Plan::block_causal`, `Plan::vdn_window`, `Plan::kv_window()` /
   `kv_window_with(flash)`, and `Plan::composed(scale, mask)` (the tensor ops
   alone, for FLUX.2's text encoder). Every family's model code calls it; the
   six `nn` functions, the `ops` wrapper, H3's `dense_attention` chooser,
   H3 VAE's `sdpa_kernel` and FLUX.2's `text_sdpa` copy are gone. The H3 ViT
   decoder keeps launching the fused kernel inside its fused block chain
   (`attn::device_mma_sdpa`, whose kernel is the rule's).
2. **`sdpa_rule.rs` is the one chooser.** `dense_route(shape, masked,
   out_bf16, probe)` returns the kernels a dense call tries in order
   (`Retired` / `FlashF32` / `Sage` / `Mma` / `Cublas` / `Composed`);
   `mma_kernel(dims, out_bf16, probe)` the fused kernel (`Fixed`, the
   per-shape `Rule`, or `Timed`); `fp8_first`, `block_causal_first`,
   `vdn_window_first`, `kv_window_flash` the masked routes. The settings
   these read live there (`FASTVIDEO_SDPA`, `FASTVIDEO_SDPA_AUTO`,
   `FASTVIDEO_WAN_CAUSAL_FLASH`, `FASTVIDEO_H3_VAE_SDPA`); the rest come in
   through the `Probe` trait (`rt::sdpa::Live` in the runtime). The
   functions are pure over the probe; the two probes with effects (loading
   attn_dc, building a cuDNN plan) are asked only when the `Mma` step runs,
   as before. Kernels that cannot run a call still decline it (`None`), and
   `run_route` takes the next step, exactly as the old `if let Some(out)`
   chain did.
3. **Byte identity** is pinned on the CPU by
   `rt/sdpa_rule_identity.rs`: a transcription of the merge-base choosers
   (`nn.rs` `run` closure, `sdpa_block_causal`, `sdpa_vdn_window`,
   `sdpa_kv_window(_with)`; `attn.rs` `device_mma_sdpa`, `flash_kernel_for`,
   `auto_cudnn_or_v2`; `h3/transformer.rs` `dense_attention`) and the new
   chooser run against one fake world. The fused-kernel pick is checked over
   every value it reads (8 kernel choices, 5 device generations, 3 SM
   counts, attn_dc, auto mode, cuDNN plan, both output types, ten shapes:
   the H3 / LTX / Wan production shapes, the rule edges, d 64 / 80 / 128,
   mismatched K/V; 19 200 cases); every entry over the route settings (5
   backends, SP world, fused context, Sage, FP8, causal flash), five
   fused-kernel contexts, the ten shapes, both output types and every set of
   declining kernels among those the entry can try (2.8 M cases). Each side
   records every kernel tried with its arguments and every effectful probe,
   in order; the traces are equal. The whole `sdpa_rule` test set, the full
   enumeration included, takes 2.0 s in a debug build on the build pod, so
   it runs in the default `cargo test`.
4. **Host twins.** Retired: `flash_style_sdpa_host` (`FASTVIDEO_SDPA=host`)
   and `block_sparse_sdpa` (`FASTVIDEO_SDPA=sparse`). Both were host-only
   and already refused with a live device (`host_only_op`), so no CUDA run
   could use them; on a CPU run `FASTVIDEO_SDPA=host|sparse` is now refused
   with a message naming the replacement (the composed reference is the CPU
   path, `FASTVIDEO_VSA=1` the device VSA), and a masked call under either
   value keeps the composed path, as before. `host_only_op` had no other
   user and is gone. Kept, and why:
   - the composed tensor-op SDPA: the CPU / tiny path of every family and
     the masked path (every op has a device kernel, so it stays on the device);
   - `vsa_attention_host`, `h3::vsa::attention_host`, `sla_attention_host` and
     the Sol / PISA host math: the oracles `fv-gpucheck` and the CPU tests judge
     the device kernels against, and the CPU path of those routes; each is
     guarded by `host_fallback` / `host_algorithm`, an error with a live device;
   - the LLM encoders' host GQA path and `neighborhood_attn_host`: no device
     twin exists to retire them for.
   `fv-gpucheck kernels`' `no_host_fallback` group now runs every `Plan` the
   families use on device operands (self- and cross-attention, d 128 / 64 /
   80, f32 and bf16 activations) and requires zero counted host fallbacks
   and a device-resident output: the GPU evidence for the retirement.
5. **Features.** `fastvideo-cudarc/nvfp4-gemm` gates `rt::nvfp4_gemm`
   (cuBLASLt block-scaled FP4, the scale swizzle, the device quantizer: no
   served path calls them); `fastvideo-cudarc/oxide` (implies `nvfp4-gemm`)
   gates the oxide Tile-IR path: build.rs embedding the `fv-oxide-aot` cubins
   (and checking `FV_REQUIRE_OXIDE`), their load per device, and the
   `FASTVIDEO_NVFP4_OXIDE_GEMM` route of the LongLive NVFP4 linear. Both are
   off by default: no `configs/serve/*.toml`, profile or family manifest
   turns the setting or `[kernels] nvfp4_gemm` on. `fastvideo-gpucheck`'s
   `cuda` feature enables `oxide`, so every fv-gpucheck build (CI
   `gpucheck-t0`, the release pod, the gpucheck image) compiles and embeds
   what it did; the serve binaries (`fastvideo-engine-service/cuda`) no longer
   embed the cubins, and `FASTVIDEO_NVFP4_OXIDE_GEMM=1` there falls back to the
   in-tree W4A4 kernel, as any build without the cubins always did (a build
   without the feature that sees `FV_REQUIRE_OXIDE` prints a warning;
   `release-artifacts-pod.sh` no longer passes the oxide environment to the
   serve builds). Because fv-gpucheck's `cuda` turns `oxide` on for any build
   it is part of, `gpucheck-t0`'s `serve-cuda-check` job type-checks the serve
   feature set on its own: `cargo check -p fastvideo-cudarc --features cuda`
   and `cargo check -p fastvideo-engine-service --features cuda`.

## Consequences

- One place decides which attention kernel runs; a new kernel is a `Step` or
  a `FlashKernel` and a row in the identity test, not a branch in a family.
- `nn::sdpa_backend()` became `sdpa_rule::backend_setting()` (the raw value,
  for the device banner and the FLUX reports) and `sdpa_rule::backend()`;
  `attn::FlashKernel` and `attn::MMA_TILE` are re-exports of `sdpa_rule`'s.
- `FASTVIDEO_VSA_WINDOW` is no longer read (its row in
  `techniques::settings` is stale and is named only by the refusal message);
  the row and `fastvideo-models/src/techniques/kernels.rs`' pointer to
  `rt::attn::flash_kernel_for` (now `rt::sdpa_rule::flash_kernel_for`) are
  for the settings work package (B3b) to update.
- Not done: moving the sparse routes' dense calls behind a `Plan` variant of
  their own, and a device kernel for LTX-2.5's neighbourhood attention.
