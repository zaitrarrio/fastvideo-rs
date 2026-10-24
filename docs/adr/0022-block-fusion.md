# ADR-0022: One block-fusion module with a rounding policy

- Status: Accepted (plan B4b, `docs/plan/2026-10-23-execution-plan.md`)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-block-fusion
- Source: `docs/gaps/2026-10-22-architecture-review.md` §3 ("the block fusion
  written three times (`wan/fuse.rs` 619, `h3/fused16.rs` 1,009,
  `ltx2/fuse.rs` 321)"), §5.2 ("One `fuse.rs` parameterized by rounding
  policy"), §7.3 item 4; ADR-0017 "How B4 and a lazy cold start plug in"
- Code: `crates/fastvideo-cudarc/src/rt/fuse/` (`policy.rs`: the rounding
  policies and the kernel each selects; `host.rs`: the one host reference;
  `switch.rs`: the switches and `BlockFusion`); the family modules
  `wan/fuse.rs`, `h3/fused16.rs`, `ltx2/fuse.rs` (launchers and
  family-only ops); the seam in `h3/pipeline/load.rs`, `h3/transformer.rs`,
  `ltx2/pipeline/load.rs`, `ltx2/pipeline.rs`, `ltx2/transformer.rs`;
  the four settings' owners in `RUNTIME_FAMILY_ROWS`
  (`fastvideo-models` `techniques/settings/workspace_check.rs`: their reads
  moved from the family directories into `rt/`)
- Tests: `rt/fuse/equality.rs` (new against old, bit for bit) with the
  pre-B4b code verbatim in `rt/fuse/legacy.rs`; `rt/fuse/switch.rs` unit
  tests; the families' existing tests; GPU: `scripts/gpu/runpod-matrix.sh
  b4b`

Line references to "pre-B4b" code are to main `17e4a1dc`.

## Context

Three families fuse the elementwise chain of their DiT block. Each wrote it
alone: its own enable getter (a thread-local override over a process-wide
cached read), its own host twins, its own launchers. The plan's first rule
is byte identity (§1), and the NVRTC source is pinned byte for byte
(`rt::kernel_src`, `KERNELS_CU_SHA256`): B4b may not move an output bit, and
it changes no kernel.

### Inventory

Rounding notation: `r16(v)` = round to bf16 (stored or immediately
re-read as bf16); everything else is f32. `x` the residual stream, `u` the
branch, `g` its gate row, `w`/`b` the norm affine, `sc`/`sh` the
modulation rows.

#### Residual + modulated norm, and the residual alone

| family op (pre-B4b) | inputs → outputs | rounding points | kernel (source) | host twin | switch | callers |
|---|---|---|---|---|---|---|
| Wan `self_residual_norm` (`wan/fuse.rs:100-146`) | `h`, `attn` `[B,S,D]` bf16, table `e` `[B,6,D]`, `norm2` `w`,`b` → `(hidden, normed)` bf16 | `s = h + a·g` (mul, then add, `_rn`, no FMA); `hidden = r16(s)`; LayerNorm of the **f32** `s` (two passes); `normed = r16(n·w + b)` | `wan_res_ln` res_mode 0, ln_mode 0 (`wan/kernels/block.cu`); off: `wan_res_gate` (f32 copy) + `wan_ln` | yes, sequential sums, `1/sqrt` (≤ 1 bf16 ulp from the kernel) | `FASTVIDEO_WAN_FUSE` (fused vs two kernels, same math) | `wan/transformer.rs:964` |
| Wan `cross_residual_norm_mod` (`:150-196`) | `h`, `cross`, `e` → `(hidden, normed)` | `hidden = r16(h + c)`; LayerNorm of the stored bf16; `normed = r16(r16(n)·(1 + sc) + sh)` | `wan_res_ln` 1, 1; off: `wan_res_gate` + `wan_ln` | yes | as above | `:979` |
| Wan `gate_residual` (`:199-231`) | `h`, `ff`, `e` → hidden | `r16(h + ff·g)` (mul then add) | `wan_res_gate` res_mode 0 | yes | none | `:990` |
| H3 `norm_mod` (`h3/fused16.rs:72-106`) | `x` `[.., H]`, `w`, AdaLN rows (f32 `[T,6,H]`, `1 + scale` pre-formed) → normed bf16 or MXFP8 | RMS over the row in the kernel's lane order, `__frsqrt_rn`; `r16(fma(x·r·w, sc1p, sh))` | `h3_norm_mod` (`h3/kernels/block_ops.cu`) | yes, **bit-exact** (`quant::row_rsqrt`) | none (always under bf16 activations) | `h3/transformer.rs:2568`, `:2601` |
| H3 `res_gate_norm_mod` (`:110-164`) | `res`, `branch`, `w`, rows → `(hidden, normed)` | `s = fma(g, u, x)`; `hidden = r16(s)`; norm of the **f32** `s`, modulation as above | `h3_res_gate_norm_mod` | yes, bit-exact | none | `:2624` |
| H3 `gate_residual` (`:167-191`) | `res`, `branch`, rows → hidden | `r16(x + r16(g·u))` (eager torch) | `h3_gate_residual` round_product 1 | yes | none | `:2599`, `:2648` |
| H3 `gate_res_norm_mod` (`:366-412`) | as `gate_residual`, then the next block's `norm_mod` | `hidden = r16(x + r16(g·u))`; norm of the **stored** hidden | `fvf_h3_gate_res_norm_mod` (`rt/kernels/dit_fusions.cu`) | none (host runs the two twins) | `FASTVIDEO_H3_FUSE` | `:2585` |
| LTX `norm_mod` / `res_norm_mod` (`ltx2/fuse.rs:102-127`, dev `:172-245`) | `x` bf16, `Residual` none / `Plain(u)` / `Gated{u, gates, row}`, `Norm::AdaLn(mod)` / `ThenMod(w, mod)`, bf16 `[rows, W]` tables → normed (and hidden) | `hidden = r16(x + u)` or `r16(x + r16(u·g))`; norm of the stored hidden, `rsqrtf`; AdaLn: `r16(r16(n·r16(1 + sc)) + sh)`; ThenMod: `r16(r16(r16(n·w)·r16(1 + sc)) + sh)` | `fvf_ltx_res_norm_mod` res_mode 0/1/2, mode 0/1 | none: CPU runs the op chain | `FASTVIDEO_LTX_FUSE` | `ltx2/transformer.rs:165`, `:197`, `:254`, `:2563`, `:2579` |

#### q/k RMSNorm + RoPE

| family op | rounding points and pairing | kernel | host twin | callers |
|---|---|---|---|---|
| Wan `qk_norm_rope` (`wan/fuse.rs:236-305`) | RMS **across heads**; `r16(r16(x·inv)·w)`; **interleaved** pairs `(2i, 2i+1)`, `[S, d]` f32 tables (cos at the even slot, sin at the odd), mul / mul / add, one rounding | `wan_qk_norm_rope16` | yes, sequential sum | `wan/transformer.rs:174` |
| H3 `qk_norm_rope` (`h3/fused16.rs:197-240`) | RMS **per head**; `x·r·w` unrounded; **partial rotate_half** over the first `r` channels, `[S, r]` tables, one rounding | `h3_qk_norm_rope`; glue: `fvf_h3_qkvg_heads` (q, k, v, gate in one launch, the same rounding) | yes, sequential sum (q/k only; ≤ 1 ulp from the kernel's warp tree, `rsqrtf`) | `h3/transformer.rs:1801`, `:1821`, `:1881`, `:2026`, `:2029`; f32 out `:1967`, `:1973`; glue `:2012` |
| LTX `qk_norm_rope` (`ltx2/fuse.rs:129-145`) | RMS across heads, `rsqrtf`; `r16(x·rs·w)`; **rotate_half within each head, head-major `[H·S, r]` tables** (`dup`); `fma(n, cos, other·sin)`, one rounding | `fvf_ltx_qk_norm_rope` | none | `ltx2/attention.rs:336` |

#### One family's alone

- LTX `gate_merge` (`fvf_ltx_gate_merge`): LTX-2.5's per-head `2σ` output
  gates and the head merge (`ltx2/attention.rs:372`, `:400`).
- H3 `merge_heads_mx` (`fvf_merge_heads_mx`), `merge_heads_nvfp4`
  (`fvf_h3_merge_heads_nvfp4`, `fvf_amax_f32_r16`), `swiglu_mx`
  (`h3_swiglu_mx`), `swiglu_nvfp4` (`fvf_h3_swiglu_nvfp4`), `qkvg_heads`
  (`fvf_h3_qkvg_heads`), `split_heads_f32`, `qk_norm_rope_f32`: H3's MXFP8 /
  NVFP4 producers and VSA feeds.
- Wan's step 1, `ln_adaln_e` (`rt/fused.rs`), was already one shared op.

#### Switches (pre-B4b)

| getter | setting | override | cache |
|---|---|---|---|
| `fused16::fuse_enabled` (`h3/fused16.rs:303`) | `FASTVIDEO_H3_FUSE` | `with_fuse` | `OnceLock` |
| `fused16::glue_enabled` (`:328`) | `FASTVIDEO_H3_GLUE`, only while the fusions are on | `with_glue` | `OnceLock` |
| `ltx2::fuse::enabled` (`ltx2/fuse.rs:34`) | `FASTVIDEO_LTX_FUSE` | `with_fuse` | `OnceLock` |
| `wan::fuse::enabled` (`wan/fuse.rs:42`) | `FASTVIDEO_WAN_FUSE` | `with_fuse` | `CachedBool` |

All four: `Layered` rows, default on, `Process` timing (ADR-0010); read
through `envflag::bool_flag`. The H3 and LTX-2 load plans already recorded
the same settings from their snapshot (`kernel_fusions`), for the golden
only.

### What is the same and what is not

The residual + modulated norm is one computation in all three families:
`hidden = R(combine(x, u, g))`, `normed = R(modulate(normalize(read)))`.
What differs is exactly where values are rounded and how the residual is
contracted (`Residual`: none, add, mul-then-add, FMA, rounded product;
`NormInput`: the f32 sum or the stored bf16; `Modulation`: five forms) —
plus two things that are not rounding points:

- **the normalizer**: Wan's is a LayerNorm (mean and variance), H3's and
  LTX's an RMSNorm. Same skeleton, different statistic; kept as
  `Normalizer`.
- **the reduction order and rsqrt**: Wan sums with `__fadd_rn` and uses
  `rsqrtf`; H3's norm kernels use per-lane FMA partials and `__frsqrt_rn`
  (so its host twin is bit-exact); LTX uses FMA partials and `rsqrtf`.
  Unifying the kernels would change these orders, i.e. the output bits.

The q/k ops are one shape (RMS → weight → rotate → round) with different
rounding points, but also a different **pairing** (interleaved, partial
rotate_half, head-major rotate_half) and norm **scope** (across heads or
per head): parameters of the policy, not rounding variants alone.

## Decision

### One module, `rt::fuse`, with the rounding points as a value

`RoundingPolicy { residual, norm: Option<Norm { input, normalizer,
modulation }>, host }` and `QkPolicy { scope, round_normalized,
round_weighted, pairing, combine, host, kernel }`, one named constant per
family op (`WAN_SELF_RESIDUAL_NORM`, `WAN_CROSS_RESIDUAL_NORM_MOD`,
`WAN_GATE_RESIDUAL`, `H3_NORM_MOD`, `H3_RES_GATE_NORM_MOD`,
`H3_GATE_RESIDUAL`, `H3_GATE_RES_NORM_MOD`, `ltx_res_norm_mod(residual,
then_mod)`, `WAN_QK_NORM_ROPE`, `H3_QK_NORM_ROPE`, `LTX_QK_NORM_ROPE`).

**Unified:**

1. **The rounding policy selects the kernel.** `RoundingPolicy::kernel()`
   maps a policy onto the one kernel instantiation implementing it, with its
   mode arguments (`Kernel::WanResLn { res_mode, ln_mode }`,
   `LtxResNormMod { res_mode, mode }`, `H3GateResidual { round_product }`,
   ...); `split()` gives Wan's two-kernel form. The launchers take their mode
   arguments from it instead of literals; a policy no kernel implements is a
   typed `PolicyError`. Kernel sources and launch configurations are
   unchanged. `QkPolicy` is descriptive on the device: each family has one
   q/k kernel and its launcher has no mode argument a policy could set
   (`use_rope`, the output dtype and the tables follow the call), so the
   policy names that kernel (a test pins the name against the legacy
   launch and the pinned source) and drives the host twins, but selects
   nothing at launch.
2. **One host reference** (`rt::fuse::host`): `res_norm_row(policy, row
   operands)` and `qk_norm` / `qk_rope_head(policy, ...)`, built from small
   primitives (`combine`, `normalize_ln_sequential`, `normalize_rms_lanes`,
   `modulate_*`). Wan's and H3's CPU twins call it with their policy;
   `quant::norm_mod_row` and `quant::gate_residual_eager` are thin wrappers
   over the same primitives (their other callers, gpucheck and
   `rt/fused.rs`, unchanged); `quant::res_gate_norm_mod_row`,
   `quant::qk_norm_rope_row`, `rms_factor` and Wan's `ln_stats` are gone.
   `HostTwin` records each family's host reading (Wan: operands as stored,
   sequential sums; H3: operands rounded to bf16, the kernel's lane order;
   LTX: none, because `rsqrtf` cannot be reproduced on the host, so its CPU
   path stays the op chain).
3. **One switch implementation** (`rt::fuse::switch`): `Switch::{H3,
   H3Glue, Ltx, Wan}`, `enabled`, `with_forced`; the families'
   `fuse_enabled` / `glue_enabled` / `enabled` / `with_fuse` / `with_glue`
   keep their names and delegate (gpucheck unchanged). Names, defaults,
   parse and the glue's dependency on the fusions are unchanged.

**Kept separate, and why:**

- **The kernels.** Their source is pinned and their reduction orders and
  rsqrt differ per family; one templated kernel would change bits. The
  policy chooses among the existing instantiations.
- **The launchers** (`dev` modules): operand checks, the families' table
  layouts (Wan `[B, 6, D]` indexed by `row / seq`; H3 f32 `[T, 6, H]` by
  `idx[row] + base`; LTX bf16 `[rows, W]` by row number), H3's MXFP8 /
  NVFP4 outputs.
- **The family-only ops** listed in the inventory: nothing to unify with.
- **LTX's host twin**: none before, none now (adding one would change what
  a CPU run executes).

### The plan's value installed on the DiT (ADR-0017)

`rt::fuse::BlockFusion` holds the switches a load plan decided
(`BlockFusion::h3(settings)`, `BlockFusion::ltx(settings)`, from the load's
snapshot; `on()` gives what the plan descriptions print, so
`tests/goldens/{h3,ltx2}_load_plan.txt` are unchanged). The fuse stage
installs it: H3's fuse stage on the DiT and the text refiner
(`H3Transformer::install_kernel_fusion`,
`H3TextRefiner::install_kernel_fusion`); LTX's fuse stage on the placed DiT
and `Ltx2Pipeline` on every DiT it loads later
(`Ltx2Transformer::install_kernel_fusion`), and the pipeline runs its text
encodes (the connectors' q/k norms and output gates) under the same value.
The DiT's `forward` / `forward_probed`, the refiner's `forward` and the
encodes run inside `rt::fuse::installed(value, ..)`, so no fused call of a
planned pipeline reads the process cache and one pipeline never mixes two
values. `installed` and `with_forced` restore the previous value through a
drop guard, so a caught panic cannot leave a value installed on a reused
thread. A switch resolves: forced (`with_forced`, the parity checks) >
installed > the process setting read once (DiTs built outside a plan:
gpucheck stages, tests). Wan has no loader stages (ADR-0017 covers H3 and
LTX-2): its switch keeps the process value.

Why this is behaviour-preserving: the plan reads the same `Layered` row
through the same parser as the cached getter; every process sets these
names before a load (CLI flags, `fv-gpucheck --mode`, fv-serve's process
plan), and fv-serve refuses to serve, in one process, models whose
process-wide settings (these included) differ (`ProcessPlan::for_models`;
only `FASTVIDEO_H3_QUANT` may differ per model). So the installed value
equals the value the cached getter would have returned. A thread spawned
inside a forward reads the process value: the same value.

## Evidence

- **CPU** (`rt/fuse/equality.rs`, against `legacy.rs`, bit for bit):
  Wan's three residual ops over five shapes (widths 8 to 1536, batch 1-2,
  rounded and unrounded inputs, two slot choices each) and its q/k op (four
  geometries, three column offsets, with and without RoPE); H3's
  `norm_mod`, `res_gate_norm_mod`, `gate_residual` over five shapes (32 to
  5120 wide, across the 256-lane stride, two block bases) and `qk_norm_rope`
  (three geometries, `r` 0 / 8 / `d`, three column offsets);
  `quant::norm_mod_row` and `gate_residual_eager` over six widths; H3's
  `fvf_h3_gate_res_norm_mod` policy against the legacy two twins in a row;
  a contraction fixture whose sums sit on bf16 rounding midpoints, so the
  FMA (H3) against product-then-sum (Wan) decides every stored hidden
  (random inputs cross such a boundary about once in 2^16 elements); every
  policy's kernel and mode arguments against the legacy literals (Wan fused
  and split, H3, all six LTX residual × norm combinations, through
  `ltx2::fuse::policy` from its operand types); every selected kernel and
  q/k kernel present in the pinned NVRTC source; the policies pairwise
  distinct; the plans' reads against the old reads for nine spellings of
  each setting; the switches against the old getters (unforced, every
  forced combination and nesting, under no plan / an H3 plan / an LTX plan
  / both).
- **Mutations** (`scripts/dev/b4b-fuse-mutations.sh` applies each alone as
  an exact-text edit, reruns the fusion tests and restores the file; all
  eleven caught): Wan's cross residual reading the f32 sum; H3's residual
  product-then-sum instead of FMA (caught by the kernel selection and the
  contraction fixture); the host's product-then-sum as an FMA (the
  contraction fixture); the rounded-norm modulation without its inner
  rounding; Wan q/k without the normalized rounding; LTX's `res_mode` 1
  and 2 swapped; the glue switch not needing the fusions; an installed
  value beating a forced one; H3's gate residual given Wan's policy; the
  interleaved RoPE reading sin at the even slot; `installed` restoring
  without a guard (lost on unwind).
- **GPU** (`scripts/gpu/runpod-matrix.sh b4b`, `FV_BASELINE_SHA` = the
  commit before B4b): per cell, the baseline binary and this build with the
  same argv and env, `compare-clips --off-identity`, each arm's
  `frames.sha256.json`. Every arm runs with `FASTVIDEO_GPU_TRACE=1`: the
  CUPTI kernel table of one traced denoise step is the evidence of which
  fused kernels ran (no log line names them, and a new line would exist in
  the PR arm only). The table used to list the step's 25 slowest kernels,
  which cannot prove an absence (an `off` cell) or that every fused kernel
  ran (an `on` cell could pass on one kernel while another was cut). So
  B4b adds `FASTVIDEO_GPU_TRACE_TOP` (a `Layered` `trace` row, default 25:
  how many kernels the report lists; it changes the report only), and the
  cells set it to list every kernel. The baseline (main `17e4a1dc`)
  predates the setting and keeps its 25, so the checks that need a
  complete table rest on the PR arm, and the base arm must agree with it
  wherever it can be compared: `b4b_verdict` (`scripts/gpu/b4-verdict.sh`,
  held to stub trees by `scripts/gpu/tests/b4-verdict.test.sh`) passes only
  when
  - both arms exited 0 and the compare passed `off_identity` and
    `off_identity_frame_count` (not skipped);
  - both arms logged a trace with a kernel table, `dropped_records` 0 and
    the same `kernel_count` (the step's launches);
  - the PR arm's table is complete: its calls sum to `kernel_count`;
  - every fused or bf16-chain kernel (`fvf_*`, `h3_*`, `wan_*`, `mx_*`) the
    base arm lists is in the PR arm with the same calls;
  - rule `on`: every named switch-gated kernel ran in the PR arm (H3:
    `fvf_h3_gate_res_norm_mod`, `fvf_h3_qkvg_heads`; LTX:
    `fvf_ltx_res_norm_mod`, `fvf_ltx_qk_norm_rope`, `fvf_ltx_gate_merge`;
    fullopt: `fvf_merge_heads_mx`); rule `off`: no switch-gated kernel
    (`fvf_*` but `fvf_split_heads_rows`) ran in the PR arm, and the
    unfused path's kernels did (H3: `h3_norm_mod`, `h3_gate_residual`,
    `h3_qk_norm_rope`; LTX: `mx_rms_norm`, `mx_rope_half`);
  - for H3, the PR arm's calls keep the block structure: one
    `h3_res_gate_norm_mod` per block; fused, `h3_norm_mod` +
    `fvf_h3_gate_res_norm_mod` and `h3_gate_residual` +
    `fvf_h3_gate_res_norm_mod` each equal it (block 0's norm and the last
    block's residual run alone, every other block's residual is deferred);
    unfused, `h3_norm_mod` and `h3_gate_residual` each equal it.
  Cells: `b4b-h3-4stepvsa` (FastH3 4-step VSA, 480p / 5 s,
  `FASTVIDEO_H3_FUSE=1`), `b4b-h3-4stepvsa-off` (`=0`), `b4b-ltx25-512p`
  (LTX-2.5 two-stage 512p / 49 frames, `FASTVIDEO_LTX_FUSE=1`),
  `b4b-ltx25-512p-off` (`=0`), and, named only, `b4b-h3-fullopt` (the
  MXFP8 route). `b4b.verdict.json` fails when a cell named in
  `FV_B4B_CELLS` has no passing verdict. A named kernel the route does not
  in fact launch on the card (e.g. `fvf_h3_qkvg_heads` when VSA does not
  take bf16 there) fails the cell and names it; the list is set from the
  code paths, not yet from a GPU run. Wan has no tree on the EU volume
  (`volumes.sh` `FV_EU_VOLUME_TREES`): its evidence is the CPU fixtures.

## Consequences

- A new family's fusion is a policy (or an existing one) and a launcher;
  the host twin, the switch and the plan seam are not written again. A new
  rounding variant is a new named policy value with an equality fixture,
  never an edit of an existing constant.
- `legacy.rs` is test-only and frozen.
- H3's host `gate_residual` now returns an error for operands that are not
  whole rows of `hidden` (or differ in length), where the old twin could
  index past the row index (a panic); well-formed inputs are unchanged.
- The cached process value remains as the fallback for DiTs built outside
  a plan; removing it needs every such caller (gpucheck stages, Wan) to
  take a plan value first.
- The kernels stay three families of instantiations. Collapsing them
  (e.g. one templated residual + norm kernel) would change reduction orders
  and is a numerics change with its own cells, not part of B4b.
