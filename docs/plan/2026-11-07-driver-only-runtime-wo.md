# Work orders: driver-only runtime, step 1 (ADR-0024)

Two increments, in order. WO-1b starts only after WO-1a is merged.
Owner decisions recorded: D1 (2026-11-12, FVID-2026-11-12-moe-aot-build-time-nvrtc,
replacing FVID-2026-11-07-moe-aot-flags) and D3 (2026-11-07,
FVID-2026-11-07-no-jit-build-gate). Both work orders are ready to run.

---

```
## WORK ORDER 1a: AOT cubins for the MoE kernel module

### Objective
The MoE kernels (rope_real, sigmoid_f, copy_f, topk_last, scatter_add_rows,
moe_group_topk, moe_combine) load from cubins/PTX embedded at build time,
like attn_fp8, instead of always NVRTC-compiling at first use. Outputs are
bit-identical to today's NVRTC module. NVRTC stays as the last fallback
(WO-1b gates it).

### Acceptance criteria (executable)
- [ ] `task check` exits 0 (fmt, clippy -D warnings, tests, NVRTC gate, cuda type-check)
- [ ] `task nvrtc` exits 0 and its report lists `moe` for every SM in FV_CUBIN_SMS
- [ ] Test `moe_aot_source_is_moe_kernel_src` passes: the bytes build.rs hands
      nvcc for the MoE module == `rt::ops::moe_kernel_src()`
- [ ] Test `moe_aot_table_covers_every_built_sm` passes (when nvcc was present
      at build: AOT_MOE has a cubin and PTX for each SM in FV_CUBIN_SMS)
- [ ] fv-gpucheck kernel group `moe_aot` passes on one RTX PRO 6000
      (`scripts/gpu/runpod-http.sh kernels <sha>`): every MoE entry point,
      run on fixed seeded inputs through the AOT module and through the
      NVRTC module, gives bit-equal outputs
- [ ] With the AOT build on the GPU, the log line `moe kernels: cubin sm120`
      appears (not `nvrtc`)

### Context
- Repo: zaitrarrio/fastvideo-rs. Rust stable (rust-toolchain.toml), cudarc
  0.17 with features std, driver, cublas, cublaslt, cudnn, nvrtc, f16,
  cuda-version-from-build-system, dynamic-loading (workspace Cargo.toml).
  CUDA 13.4 pins: scripts/gpu/cuda-13.pins.
- Build and test on the shared build pod (scripts/dev/build-pod.sh,
  docs/dev/build-pod.md), not in a small container. GPU work on Runpod per
  CLAUDE.md: wall-clock backstop, delete the pod, stop before balance < $8.
- crates/fastvideo-cudarc/build.rs: KERNEL_PARTS (includes
  src/rt/kernels/moe.cu) -> $OUT_DIR/kernels.cu -> main AOT table, compiled
  with reference flags (no fast math). Per-module tables: SRC_FP8 /
  SRC_SAGE / SRC_VDN via `module_entries` (sm >= 89 only, --use_fast_math,
  -std=c++17), written with `write_module_table(path, NAME, entries)`.
- src/rt/ops.rs: `moe_kernel_src()` returns IDX-define prefix + the
  `// ==== region: moe ====` .. `// ==== end region: moe ====` slice of
  KERNEL_SRC. `load_moe_kernels()` compiles it via
  `rt::nvrtc_sass::load_for_device` with CompileOptions { use_fast_math:
  Some(true), ftz: Some(true), ..Default }.
- Pattern to copy: src/rt/attn_fp8.rs `load_embedded` (FASTVIDEO_KERNELS=nvrtc
  -> skip; exact-SM cubin via temp file + `Ptx::from_file`; else same-major
  PTX <= device SM via `Ptx::from_src`; else None -> NVRTC).
- Invariant: plan §1 byte identity. No served output may move a bit.

### Interfaces & contracts (pinned — do not redesign)
- build.rs writes `$OUT_DIR/moe_aot.cu` containing exactly the string
  `moe_kernel_src()` returns, and `$OUT_DIR/aot_moe.rs` defining
  `pub static AOT_MOE: &[(u32, &[u8], &str)]` — (sm, cubin bytes, PTX text),
  same shape as AOT_ATTN_FP8.
- SMs: every SM in FV_CUBIN_SMS (default "75,80,86,89,90,100,120"); no
  sm >= 89 filter for this module.
- Compiler (owner decision D1, revised 2026-11-12,
  FVID-2026-11-12-moe-aot-build-time-nvrtc): libnvrtc at build time, via a
  cudarc build-dependency (features std, nvrtc, dynamic-loading,
  cuda-13000), with `rt/moe_nvrtc_opts.rs` `MOE_NVRTC_OPTIONS`
  (`--ftz=true --fmad=true`) + `--gpu-architecture=sm_XY` (cubin) /
  `compute_XY` (PTX). Test `moe_nvrtc_options_match_runtime` pins those to
  what the runtime path passes. The first attempt (nvcc `--use_fast_math
  --ftz=true`) failed the GPU gate in 4 of 12 checks.
- `fn load_moe_kernels() -> Result<Rc<MoeKernels>>` keeps its signature;
  the log line keeps the form `moe kernels: {origin}` with origin
  `cubin sm{N}` | `ptx compute{N}` | the nvrtc_sass origin.

### Steps
1. Add the two CPU tests above (source equality; table coverage gated on a
   build-time `cfg`/const telling whether nvcc ran). Watch them fail.
2. build.rs: write moe_aot.cu by reproducing moe_kernel_src()'s
   construction from the concatenated KERNEL_PARTS (same prefix, same
   markers); add `rerun-if-changed` for moe.cu (already in KERNEL_PARTS) ;
   compile cubin + PTX per SM with a new helper (do not loosen
   `module_entries`' sm >= 89 rule for the attention modules); write
   aot_moe.rs. Without nvcc: empty table, as the other modules do.
3. ops.rs: include the table; add `load_embedded` for MoE mirroring attn_fp8;
   call it first in `load_moe_kernels`; NVRTC path unchanged after it.
4. fv-gpucheck: add group `moe_aot` (compiles the NVRTC module explicitly,
   loads the AOT one, compares every entry point bit for bit on seeded
   inputs, incl. a topk tie case).
5. Run `task check`, `task nvrtc`, then the GPU group on one RTX PRO 6000.

### Non-goals
- Gating or removing NVRTC anywhere (WO-1b).
- Changing the main KERNEL_PARTS flags or loading MoE from the main module.
- cuDNN, static linking, image changes.

### Decision points — STOP and ask
- Any new dependency.
- GPU bit-equality fails with the chosen flags
  (report which entry point differs and by how much; do not tune flags
  until the owner decides).
- Binary size growth > 5 MB.

### Verification
- Gates above with pasted output. Mark as UNVERIFIED anything about nvcc vs
  NVRTC flag equivalence or cudarc API behaviour not read from the
  cudarc 0.17 source in Cargo.lock.
- Rollback: revert the commit; the NVRTC path is untouched.

### Decision log entry (proposed)
FVID · <date> · FVID-<date>-moe-aot: MoE kernels load embedded cubins/PTX
(AOT_MOE, flags per D1), NVRTC last; bit-equal on RTX PRO 6000 (gpucheck
moe_aot). ADR: docs/adr/0024-driver-only-runtime.md (step 1a)
```

---

```
## WORK ORDER 1b: `jit` feature — no runtime compilation in served builds

### Objective
A `fastvideo-cudarc` build without the new `jit` feature never compiles a
kernel at runtime. If the binary has no cubin or PTX for the device, module
load fails with an error naming the module, the device SM and the SMs the
binary was built for. Builds with `jit` behave exactly as today.

### Acceptance criteria (executable)
- [ ] `task check` exits 0, with `task build:cuda` running
      `cargo check -p fastvideo-cudarc -p fastvideo-gpucheck --features cuda,jit`
- [ ] `cargo test -p fastvideo-cudarc` passes with and without `--features jit`
- [ ] Test `no_jit_refuses_nvrtc_override` passes: without `jit`,
      FASTVIDEO_KERNELS=nvrtc returns an error naming feature `jit`
- [ ] Test `no_jit_missing_sm_is_named` passes: without `jit`, a load for an
      SM absent from every table returns `KernelsUnavailable` with
      module, device SM and built SMs
- [ ] `rg 'compile_ptx|load_for_device|nvrtc_sass::load' crates/fastvideo-cudarc/src`
      — every hit is under `#[cfg(feature = "jit")]` (a CPU test enforces this
      with a source scan, like the existing kernel_src tests)
- [ ] fv-gpucheck builds with `jit`; `task nvrtc` still passes
- [ ] On one RTX PRO 6000, the served image (no `jit`) runs one H3, one
      FastH3 and one LTX-2.5 cell of `task rtx6000` with outputs byte-equal
      to the previous image

### Context
- Runtime NVRTC call sites today: rt/kernels.rs (`compile_ptx`, `load_for`
  fallback, `compile`), rt/nvrtc_sass.rs, rt/attn_fp8.rs, rt/attn_sage.rs,
  rt/attn_fp4.rs, rt/attn_dc.rs, h3/vdn.rs, rt/ops.rs (`load_moe_kernels`
  fallback after WO-1a).
- The cudarc `nvrtc` cargo feature must stay enabled: `cudarc::nvrtc::Ptx`
  is how cubins are loaded (`Ptx::from_file`). Gate only our compile calls.
- build.rs currently warns and produces a JIT-only binary when no nvcc is
  found. fv-gpucheck depends on fastvideo-cudarc with features cuda + oxide.
- engine-service / core / cli enable `fastvideo-cudarc/cuda`, not `jit`.

### Interfaces & contracts (pinned — do not redesign)
- crates/fastvideo-cudarc/Cargo.toml: `jit = ["cuda"]`.
- fastvideo-gpucheck Cargo.toml adds "fastvideo-cudarc/jit" to its cuda feature.
- New error variant in the crate's kernel-load error type:
  `KernelsUnavailable { module: &'static str, device_sm: u32, built_sms: Vec<u32> }`
  Display: "no kernels for {module} on sm_{device_sm}; this binary was built
  for {built_sms:?}; rebuild with FV_CUBIN_SMS including {device_sm}, or
  build with feature `jit`"
- FASTVIDEO_KERNELS=nvrtc without `jit`: error
  "FASTVIDEO_KERNELS=nvrtc needs a build with feature `jit`".

### Steps
1. Write the three CPU tests (two behaviour, one source scan). Watch them fail.
2. Add the feature, the error variant, and `#[cfg(feature = "jit")]` around
   every NVRTC compile path; the non-jit branch returns the pinned errors.
3. build.rs (D3, decided): panic for `cuda` without `jit` when no nvcc is
   found, message naming NVCC and the `jit` feature.
4. Taskfile `build:cuda` -> `--features cuda,jit` (D3); gpucheck feature.
5. Gates, then the GPU parity run with the new served image.

### Non-goals
- Removing cuDNN or the cuDNN SDPA candidate; static linking.
- Disabling the cudarc `nvrtc` feature.
- Measuring whether cuBLASLt still dlopens libnvrtc (record it if seen,
  but it is step 2's measurement).

### Decision points — STOP and ask
- Any served path that turns out to reach NVRTC with an AOT build (a
  missing table, an SM not built): report it, do not add a fallback.
- Any change to served outputs.

### Verification
- Gates with pasted output; GPU byte-equality on the three cells.
- Rollback: build images with `--features jit` (one line), behaviour as before.

### Decision log entry (proposed)
FVID · <date> · FVID-<date>-no-jit-served: `jit` feature gates all runtime
NVRTC; served builds hard-fail with KernelsUnavailable instead of JIT;
gpucheck keeps `jit`. ADR: docs/adr/0024-driver-only-runtime.md (step 1b)
```
