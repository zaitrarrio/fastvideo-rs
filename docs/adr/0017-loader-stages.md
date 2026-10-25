# ADR-0017: The H3 and LTX-2 loads in four stages: plan, open, place, fuse

- Status: Accepted (plan B6, `docs/plan/2026-10-23-execution-plan.md`)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-loader-stages
- Source: `docs/gaps/2026-10-22-architecture-review.md` §4.2 ("`H3Pipeline::load`
  is ~830 lines in one function"), §7.3 items 5 and 6
- Code: `crates/fastvideo-cudarc/src/h3/pipeline/load.rs` (H3),
  `crates/fastvideo-cudarc/src/ltx2/pipeline/load.rs` (LTX-2),
  `crates/fastvideo-cudarc/src/rt/load_stages.rs` (`DeviceFacts`)
- Tests: `crates/fastvideo-cudarc/tests/load_stages.rs`, goldens in
  `crates/fastvideo-cudarc/tests/goldens/{h3,ltx2}_load_plan.txt`

## Context

`H3Pipeline::load_with_settings` was one function of about 880 lines and
`Ltx2Pipeline::load_with_settings` one of about 230. Each interleaved four
kinds of work: deciding (the recipe, the contract, the quant mode, the text
encoder, the DiT placement, the adapter), reading the device to decide (free
memory, the compute capability, bf16 activations, the linear route), mapping
files, and uploading. A decision could not be inspected without a GPU and the
weights, and nothing tested what a served configuration would load where.
B2 (ADR-0013) already split *when* a family loads ("`load` plans, `generate`
loads") without changing the loaders; B3b (ADR-0010 amendment) passed the
settings snapshot into them and left their shape to this package.

The non-negotiable is byte identity: the same tensors in the same order and
precision, the same placement decisions, the same `load/io` lines and ledger
bookings.

## Decision

### The stages and what each may touch

| stage | function | input → output | may touch |
|---|---|---|---|
| **plan** | `H3LoadPlan::new`, `Ltx2LoadPlan::new` | options, `&Snapshot`, the active profile, `DeviceFacts` → `H3LoadPlan` / `Ltx2LoadPlan` | the settings snapshot (a1 by parameter; a2 callees through a nested `Snapshot::scope`), configs and manifests on disk (`config.json`, `fastvideo_inference.json`, `mlx_h3_dit.json`, the pre-quantized tree's manifest, safetensors headers, directory listings), the log. **No device, no tensor byte.** |
| **open** | `plan.open()` | `&plan` → `H3Opened` / `Ltx2Opened` | maps the checkpoint files (headers) and queues their read-ahead in load order; reads the bytes it must check (the NVFP4 DiT against the bf16 tree). **No device.** Starts the `load/io` counters. |
| **place** | `plan.place(opened)` | `&plan`, the maps → `H3Placed` / `Ltx2Placed` | the device: uploads, quantizes, offloads, books the ledger, measures free memory where a decision depends on what is already placed; merges the planned adapter into each linear as it loads; writes the `load/io` lines. |
| **fuse** | `plan.fuse(placed)` | the plan, the placed models → `H3Pipeline` / `Ltx2Pipeline` | the passes over the placed model and the pipeline's construction. |

`load_with_settings` is their composition, inside the load's settings scope:

```rust
let _load_scope = settings.scope();
crate::rt::tensor::default_bf16_activations();   // (LTX-2: + the NVFP4 video-FFN scope)
let plan = H3LoadPlan::new(root, options, settings, settings::active(), DeviceFacts::measure())?;
let _quant_default = plan.quant_default_guard(); // H3: VDN-H3's bf16 default
let opened = plan.open()?;
let placed = plan.place(opened)?;
plan.fuse(placed)
```

`load` (captures a snapshot) and `load_with_settings` keep their signatures;
no caller changed (`fastvideo-engine-service`, `fastvideo-core`'s families,
`fv-gpucheck`).

### `DeviceFacts`

What a plan needs to know about the card, measured once before it
(`DeviceFacts::measure`) or given (tests, a dry run): `free` (free memory after
the device budget), `device_sm` (the global device's capability),
`device_expected` (ops run on the device), `bf16_activations`, `linear_route`
(how a `Linear` loads, as a pre-quantized tree records it). The runtime's
device-dependent rules gained pure forms over these facts, and the live
functions are now those forms applied to the live facts (same expressions):
`QuantMode::resolve_on` (`from_settings` = it with the thread's scoped default
and the live capability, plus the once-per-process downgrade log),
`h3::vae::mxfp8_wanted_on` / `nvfp4_setting_from`,
`VideoVaeSource::planned_linears_on`, `h3::nvfp4::require_tensor_cores_on`,
`rt::load_stages::linear_route`.

### What each stage holds, per family

**H3** (`h3/pipeline/load.rs`):

- plan: the recipe (the profile's when the caller named none), Ref2VA and the
  reference fit; the contract and schedule; the VDN checkpoint and its bf16
  default; `FASTVIDEO_H3_QUANT` on this device; the DiT config; the video VAE
  source and how its linears load; the techniques, the profile log, `dense`;
  the TAEH3 weights; the `Auto` text encoder decision (the memory plan at the
  facts' free memory) and whether a resident encoder loads; the DiT's source
  (`transformer/`, `transformer_ref/` or the MLX export, its manifest read);
  the NVFP4 plan (Comfy file or ModelOpt `transformer/`, the tensor-core and
  recipe checks); the pre-quantized tree (`dit_tree::select`); the VSA gate;
  the placement inputs (`dit_policy`, the default request's resident and
  streamed needs); the adapter merge (`H3AdapterPlan`: tree, Preview,
  Sol-H3, VDN, Plug, Turbo, none); the TeaCache state; the runtime's block
  fusions as the snapshot sets them (`FASTVIDEO_H3_FUSE`, `FASTVIDEO_H3_GLUE`).
  `residency(free)` and `i2v_tower(..)` are the plan's placement rules as pure
  functions of the free memory place measures.
- open: the resident encoder's map, the DiT's map (the MLX export's sets
  `FASTVIDEO_H3_AFFINE` through `settings::set_env`, as before), the NVFP4
  DiT, the pre-quantized tree, the read-ahead groups, the decoders' maps.
- place: the text encoder (`load/io h3 text_encoder`), the adapter's factors
  (read and checked against the DiT's map), the image-to-video tower, the
  residency (logged), the refiner and DiT (the adapter merged per linear,
  before quantization), the VDN branch, the merge's completeness check
  (`H3LoraFuse::finish`), `load/io h3 dit`, the decoders, `load/io h3 total`,
  the sparse FP4 VSA choice (it needs the device's kernels).
- fuse: TeaCache installed, the merge's report (`h3 lora merge: device|host`,
  `load/verify h3 lora`), the pipeline.

**LTX-2** (`ltx2/pipeline/load.rs`):

- plan: the distilled LoRA of a dev DiT, the IC-LoRA and its header, the
  profile log and the TAEHV check, the `--offload` placement, the text
  residency, the DiT policy, the default workload's need and the residency
  at the facts' free memory, the upsampler, the tiny autoencoder and its
  channel check, `FASTVIDEO_LTX2_LORA_BASE`, the NVFP4 video-FFN rule and
  `FASTVIDEO_LTX_FUSE` as the snapshot sets them.
- open: the DiT's map with its read-ahead (a pipeline without a LoRA).
- place: the DiT (the IC-LoRA's factors attached at strength 0), the decoders
  unless the DiT streams, the tiny autoencoder, `load/io ltx2 load`; then the
  read-ahead of what the first request places, queued behind this placement
  as before (Gemma, a LoRA pipeline's DiT). Placement decides those are
  deferred, so it is placement that queues their reads.
- fuse: LTX-2 fuses its LoRAs per request (the distilled LoRA at each stage's
  strength in `dit_for`, the IC-LoRA in stage 1 in `set_ic_strength`), so the
  load's fuse records where they start (strength 0, the base) and builds the
  pipeline.

### Why the fusion is split across place and fuse

An adapter merge cannot be a pass after placement without changing numerics
or memory: H3 merges each LoRA into the bf16 weight before that linear is
quantized (MXFP8 / W8A8 / NVFP4), and holding the unmerged placed weights for
a later pass would need a second copy of the DiT. So the plan decides the
merge, place applies it per linear as it loads (and checks every target was
applied), and fuse holds what runs over the finished model. The kernel-level
block fusions (`h3/fused16.rs`, `ltx2/fuse.rs`) are chosen per call at run
time; the plan records the setting so a golden shows it, and since B4b the
fuse stage installs that value on the DiT, whose forwards run under it
(`rt::fuse::BlockFusion`, ADR-0022).

### Byte identity

Every statement of the two loads kept its order inside its stage, and the
stages run in the order the statements did, with these differences, none of
which changes a tensor, a precision, a placement or an evidence line:

- `DeviceFacts::measure` runs before the plan. H3 measured free memory after
  resolving the recipe, contract, VAE source and techniques, LTX-2 after
  resolving its LoRAs; none of that allocates on the device, so the value is
  the same. `bf16_activations`, the linear route and the capability are read
  once instead of at each site; they do not change during a load (cached
  getters, the global device).
- Log lines that are not evidence moved: into the plan (H3 `h3 recipe=..`,
  `h3 sol-attn: ..`, the pre-quantized tree selection; LTX-2 none) and out
  of H3's DiT section into fuse (the TeaCache line, `h3 lora merge`,
  `load/verify h3 lora`, now after `load/io h3 total`). The `load/io` lines
  keep their sequence and their counters: no file read moved across one (the
  VDN branch is read in place, before `load/io h3 dit`, as before).
- Errors the plan decides now stop a load before anything loads instead of
  after the text encoder or the DiT did (a missing `transformer_ref/`, the
  NVFP4 checks, a Turbo step count, a Plug or Turbo recipe with Ref2VA, the
  TeaCache state, the VAE MXFP8 setting, the DiT offload setting, LTX-2's
  TAEHV channel count and `FASTVIDEO_LTX2_LORA_BASE`); the messages are
  unchanged. An invalid `FASTVIDEO_H3_QUANT` is kept as the plan's error value
  and raised where the load read it before.
- The once-per-process `FASTVIDEO_H3_QUANT=mxfp8 needs sm_100+` line is now
  written by the first live read (the DiT load), not by the plan.
- The wall seconds of the `load/io h3 ..` lines no longer include the
  pre-quantized tree's selection (its manifest and the source shard sizes
  are read by the plan, before the I/O clock starts); their byte counters
  are unchanged (those reads go through no weight map).
- `dit_tree_identity_for` (`fv-gpucheck quantize-dit`) resolves the quant
  mode before its early returns instead of at the end; an invalid
  `FASTVIDEO_H3_QUANT` is still reported only where it was.

**GPU confirmation.** The `techniques` byte-identity family:
fasth3-4step-vsa-768p (on the EU volume it loads the served
`transformer_prequant_4step-vsa_mxfp8` tree, so the tree path and the auto
encoder), h3-480p-fullopt (`sol-h3-rtx`: no adapter, since
`is_sol_h3_recipe` excludes `-rtx`; it covers the plan's quant and placement
and TeaCache installed in fuse), ltx25-512p-sol (LTX-2.5 two-stage). The
paths those cells do not reach run in the `b6` family (first of
`scripts/gpu/runpod-matrix.sh`; the `b6` tier of `lab/` since, fv-lab only after
its GPU gate, ADR-0019 §10) (baseline binary against this build, the same
command line, `compare-clips --off-identity`), on trees of the EU volume
only:

| cell | what it loads |
|---|---|
| `b6-solh3-tree` | `sol-h3` from `h3-base/transformer_prequant_sol-h3_mxfp8` (tree selection on the linear route and the quant mode) |
| `b6-solh3-merge` | `sol-h3` with `FASTVIDEO_H3_DIT_TREE=off`: the Sol-H3 adapter merged in place |
| `b6-4stepvsa-merge` | `4step-vsa` with `FASTVIDEO_H3_DIT_TREE=off`: the Preview LoRA merged at load |
| `b6-h3-streamed` | `4step-vsa` under `FASTVIDEO_DEVICE_BUDGET_GIB=32`: streamed encoder and DiT, decoders per decode |
| `b6-h3-ref2va` | `transformer_ref` from `h3-ref2va`, one image reference, the i2v tower beside the auto encoder |
| `b6-ltx25-ref2v` | the Ingredients IC-LoRA attached at load, fused at stage 1 |
| `b6-ltx25-a2v-guided` | the dev DiT with the distilled LoRA: DiT on the first request, base in host memory |
| `b6-ltx25-offload-cpu` | `--offload cpu`: DiT and Gemma streamed, decoders around their calls |

No cell covers the NVFP4 loads (the Trim's ComfyUI file, the ModelOpt
`transformer/`), VDN-H3 or the MLX export: none of their trees is on the EU
volume (`scripts/gpu/volumes.sh` `FV_EU_VOLUME_TREES`). Their plans are
pinned by the goldens below; their open and place code moved verbatim.

### Testing on the CPU

- Plan goldens (`h3_plans_are_the_golden_decisions`,
  `ltx2_plans_are_the_golden_decisions`), on a synthetic weights directory
  with the EU volume's H3 layout (names and markers, no tensor bytes): the
  served pre-quantized trees with the manifests `quantize-dit` writes, the
  Preview LoRAs, a Trim NVFP4 file, a ModelOpt NVFP4 `transformer/`, a VDN
  stage-B checkpoint. H3: `4step-vsa` (served profile and the `techniques`
  arm: the served tree; tree off: the Preview merge; B200: the tree; H100 and
  CPU: bf16, no tree), `sol-h3` (served profile: its tree; H100 with
  `mxfp8` asked: W8A8, no tree), `8step` (RTX PRO 6000, a 32 GiB card,
  streamed + W8A8), `sol-h3-rtx` + TeaCache (the `fullopt` cell's two arms),
  `trim-8step-nvfp4` (RTX PRO 6000; H100 refused), `4step-vsa-nvfp4`
  (ModelOpt, B200), `vdn-h3-50step`. LTX-2.5: distilled two-stage (RTX PRO
  6000, a 32 GiB card, `--offload cpu`, TAEHV + NVFP4, the CPU),
  reference-to-video (IC-LoRA), dev audio-to-video (distilled LoRA, base on
  the host). Each pins what loads, the quant mode, the offload, the adapter
  and step-cache fusions. H3's DiT placement line shows the rule at the
  load-start free memory; the place stage applies it to the free memory it
  measures after the text encoder and the i2v tower. `FV_GOLDEN_BLESS=1`
  rewrites the goldens.
- Open on a tiny synthetic snapshot (H3, the reference fixtures).
- The full LTX-2 lifecycle (plan, open, place, fuse) on the tiny fixtures at
  a tiny config, to a pipeline.
- Not on the CPU: H3's place and fuse (`transformer_config` accepts only the
  production dims, so a tiny H3 DiT cannot load through the pipeline), and
  either family's generate (their text encoders are the production
  Qwen3-VL-32B / Gemma configurations with their tokenizers). Wan is the only
  family with a CPU generation (`--tiny`).

### Retired with this package (review §7.3 item 6)

- `fastvideo-ops::TensorBackend`, its host reference (`HostBackend`,
  `HostTensor`, `Device`, `DType`, `OpsError`) and `fastvideo-cudarc`'s
  `CudarcBackend`: no user (the one implementation delegated to the host and
  only its own tests called it). `fastvideo-ops` keeps `fp8`, which the
  runtime and the model crate use; the unused `fastvideo-ops` dependency of
  `fastvideo-core` and `fastvideo-gpucheck` and its `thiserror` dependency
  went with it.
- The `fastvideo-mlx` crate: an Apple Silicon scaffold (host stubs, an
  optional `mlx-rs` gate, no DiT or TAEHV graph) that nothing depended on.
  Its lockfile entries (`mlx-rs` and its build tree) are gone.

Kept, because they are reachable:

- The CUDA runtime's reader of FastVideo's MLX H3 export (`h3/mlx.rs`,
  `WeightMap::with_mlx_h3_aliases`, `rt::affine`, the fused affine GEMM):
  any H3 load whose `transformer/` (or root) holds `mlx_h3_dit.json` and
  `mlx_h3_dit.safetensors` takes it (`H3DitSource::Mlx`), from the CLI's
  `--weights` or a serve config's weights directory; it is chosen by the
  checkpoint, not by a setting. No served family variant ships such a tree
  today and the EU volume holds none, so it has no GPU cell; `fv-gpucheck`'s
  affine kernel checks cover the GEMM, and `FASTVIDEO_H3_AFFINE` (on-the-fly
  affine quantization of any H3 tree) uses the same kernels.
- `nvfp4_gemm.rs` and the oxide AOT path: behind features since B5.

### How B4 and a lazy cold start plug in

- B4's one `fuse.rs` (a rounding-point policy) becomes a value the plan
  decides (today `kernel_fusions`, read from the snapshot) and the fuse stage
  installs on the pipeline, instead of the process-wide cached getters
  (`fused16::fuse_enabled`, `ltx2::fuse::enabled`). Done in B4b
  (ADR-0022): `kernel_fusions` is an `rt::fuse::BlockFusion`, installed on
  the H3 DiT by the fuse stage and on every LTX-2 DiT the pipeline holds;
  the cached process value remains for DiTs built outside a plan. B4's `StepCache` trait
  replaces `H3LoadPlan::teacache` with a boxed step cache the fuse stage
  installs; LTX-2's FBCache and the Wan caches can take the same seam.
- A lazy or streamed cold start needs exactly this split: the plan runs
  before the device exists (a dry run, `fv-boot`'s weights plan) and names
  every file place reads, so a registry `hydrate` can fetch those chunks
  first and in load order; open is independent of the device and can run
  while the context initializes; place already goes component by component
  and LTX-2 already defers its DiT and Gemma to the first request, so a
  placement that streams DiT blocks as their chunks arrive is a change inside
  place, with the plan and the pipeline unchanged.

## Consequences

- The plan of a served configuration is a value: what a load would do on a
  given card is testable without a GPU or the weights, and a decision change
  shows in a golden.
- The device reads a decision depends on are in one place (`DeviceFacts`),
  with pure forms of the precision rules beside the live ones.
- Wan is not restructured (it already loads apart from generating, ADR-0013).
- Adding a component to a load means placing it in one stage; the stage
  contract above says which.
