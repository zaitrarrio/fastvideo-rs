# fv-lab: run a tier's cells on a GPU pod

`fv-lab` runs the lab's run plan (`lab/tiers.toml`, `lab/cells/*.toml`) on a GPU box and makes that box: a Runpod pod
on fv-deploy's guards (price cap, $8 floor, EU volume, detached backstop, ready within 600 s in at most 3 cycles).
Design: [ADR-0019](../adr/0019-fv-lab.md). Today it carries twenty tiers that were `runpod-matrix.sh` families (and `hd5`, a sibling frozen with `hd`). Each
passed its GPU gate (fv-lab's run and the script's of one commit, frames byte-identical), so its script branch is
deleted and the tier is fv-lab only ([The families](#the-families), [ADR-0019 §10](../adr/0019-fv-lab.md)). Each is
held to the script's last dry run by a frozen golden ([below](#keeping-it-equal-to-the-scripts)). The other families
stay in the script and move in later slices. `b4a` (D1d) is the first whose cells carry a verdict step ([Verdicts
and `fv-lab gate`](#verdicts-and-fv-lab-gate); its GPU gate passed on `e82b6b7`, ADR-0019 §11.6); `det-short` and
`determinism` (D1d) the last, with their `[repeats]` frame hashes (GPU checks passed on `f93a961`, §15.10). `oracle`
is a tier whose script branch stays whole ([below](#determinism-and-the-oracle-d1d-second-slice)), and `fuse`,
`attn3` and `ltx` (D1f) are retired too (GPU checks passed on `762ce4d`, §17.6; [below](#fuse-attn3-ltx-d1f)), and so
are `hd` with its 1080P-gate sibling `hd5` (D1 S2, [below](#hd-and-hd5-d1-s2)), `serve-engine` (D1 S3,
[below](#serve-engine-d1-s3)), `sfstream` (D1 S4, [below](#sfstream-d1-s4)) and `mmaudio` (D1 S5,
[below](#mmaudio-d1-s5)), each after its GPU check (ADR-0019 §22.10, §22.12, §22.14, §22.16; retired in S7, §22.19).
`wan` (D1 S6, [below](#wan-d1-s6)) is a tier whose script branch stays until `sfwan` goes with `oracle` (S8). `gold`
is fv-lab's own: every gold variant's canonical cell (ADR-0023).
`precision-nvfp4` and `ltxvae-gen` are two modes of retired families, restored as tiers ([ADR-0019 §12](../adr/0019-fv-lab.md)).
Eleven more families left the script with no tier (`h3`, `hunyuan`, `sana-video`, `b200`, `writer`, `precision-debug`,
`identity`, `b4b`, `b4c`, `b4c2`, `h3attn`; D1g, ADR-0019 §18), and nine after them (`sfquality`, `headline`,
`speechtest`, `cold`, `eval`, `ltxoffload`, `solbench`, `sol-lingbot`, `sol-cosmos3`; ADR-0019 §22, S1): every driver
refuses them ([below](#retired-with-no-tier-d1g)).

## Build

On the build pod (CLAUDE.md "Builds"), then fetch the binary:

```sh
bash scripts/dev/build-pod.sh run . -- cargo build --release --locked -p fastvideo-lab --features cli --bin fv-lab
bash scripts/dev/build-pod.sh fetch . release/fv-lab ./fv-lab
```

The tools release carries it in its `gpucheck` set, and the runtime image `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>`
at
`/opt/fastvideo-rs/target/release/fv-lab` with `lab/` at `/opt/fastvideo-rs/lab` (and the image workflow uploads it
beside `fv-gpucheck` as an artifact). `fv-lab` finds `lab/tiers.toml` above the current directory, then above its own
binary; `--root DIR` (or `FV_LAB_ROOT`) elsewhere.

## The run plan

```sh
./fv-lab check                    # parse and validate lab/; one line per tier
./fv-lab schema tiers             # the JSON Schemas (crates/fastvideo-lab/schemas/{tiers,cell}.v1.json)
./fv-lab run rtx6000 --dry-run --work /workspace --scratch /fvscratch --tag test
```

`--dry-run` prints every arm's command line exactly as `FV_DRY_RUN=1 runpod-matrix.sh rtx6000` logged it before the
branch was deleted (the frozen golden), and writes nothing. A tier lists cells; a cell (`lab/cells/<name>.toml`) is one workload (prompt, seed, length, canvas) and its
arms; an arm is one run directory, one fv-gpucheck command (with `{weights}`, `{runs}`, `{prompt}`, … placeholders),
gated on a `verify-weights.sh` cell and naming the weight trees it reads. Edit a cell, then `fv-lab check` and the
dry run show what will run.

## The families

| tier | cells | arms | what it runs | notes |
|---|---|---|---|---|
| `rtx6000` | `rtx6000-{h3,ltx25}` | 7 | the RTX PRO 6000 parity baseline: H3, FastH3, LTX-2.5 | D1a; `runpod-http.sh`'s default |
| `techniques` | `techniques-*` (4) | 13 | technique profiles against the baseline binary and this build, byte identity | needs `--baseline-sha` |
| `b6` | `b6-*` (8) | 16 | the loader split's load paths, baseline binary vs this build, byte identity | needs `--baseline-sha` |
| `rtx5090` | `rtx5090-*` (5) | 17 | the sol-engine RTX 5090 suite (dense, Sol, fullopt, TAE decoders) | resident on the PRO 6000 |
| `fastvideo` | `fastvideo-*` (6) | 17 | our stack without sol-engine techniques, TAE twins, lossy gates | |
| `h3arms` | `h3arms-*` (3) | 13 | FastH3 Sol / TeaCache arms gated against the plain recipe | 11 arms skip on the EU volume (`h3-8step`) |
| `trace` | `trace-{h3,ltx25}` | 5 | one CUPTI-traced warm step per workload | |
| `ltxfps` | `ltxfps-{ltx25,h3}` | 5 | LTX-2.5 at 25 / 48 / 50 fps and silent, the frame-sink checks | |
| `ltxvae` | `ltxvae-*` (4) | 12 | the LTX-2.5 VAE decode alone (`vae-bench`) | |
| `precision` | `precision-{h3,ltx25}` | 8 | precision switches against f32 activations | the script's default mode only |
| `b4a` | `b4a-h3-*` (3) | 6 | the H3 TeaCache step cache, baseline binary vs this build: byte identity and equal cache decisions, a verdict per cell | D1d; needs `--baseline-sha`; retired after its GPU gate (§11.6) |
| `precision-nvfp4` | `precision-nvfp4-ltx25-{4k5s,1080p20s}` | 4 | NVFP4 video FFN against bf16, LTX-2.5 Sol, compare + lossy gate | was `precision`'s `FV_PRECISION_ARM=ltx-nvfp4` mode; restored (§12) |
| `ltxvae-gen` | `ltxvae`'s 4 + `ltxvae-gen-{4k5s,1080p20s}` | 16 | `ltxvae`, plus per workload a cold gen saving its latents and the streaming decoder on them | was `ltxvae`'s `FV_VAE_GEN=1`; restored (§12); shares `ltxvae`'s cells |
| `det-short` | `det-short-{h3,wan,ltx,ltx-sage}` | 15 | byte identity across fresh processes at short configs, three runs per group and the timed-pick control; `determinism.txt` | D1d (§15); retired after its GPU check (§15.10); `wan` skips on the EU volume |
| `determinism` | `determinism-{ltx-t2v,ltx-i2v,h3}` | 9 | byte identity of 1080p runs within one process and across two; prompt sets as files, hashes per run, stage digests | D1d (§15); retired after its GPU check (§15.10) |
| `oracle` | `oracle-<target>` (9), and 10 extra cells | 32 (+46) | the GPU oracle diff of the nine tensor-diff targets against the upstream references (`FV_ORACLE_URL`); on request (`extra_cells`, run when `--cells` names their arms) the ten LTX clip-metric targets with their ffmpeg / ffprobe `[[metric]]` reports (D1i, §20) | D1d (§15); needs `--extra-env FV_ORACLE_URL=…` and `--cells`; GPU check passed (`f93a961`; the clip-metric targets': §20.7), but **not retired**: the script branch stays whole for `sfwan13` and `oracle.sh` (§15.10, §20.5) |
| `oracle-r2` | `oracle-sfwan13` | 10 | the oracle diff of SF-Wan 1.3B (`sfwan13`), its tree fetched from R2 (`weights = "volume+r2"`): tensors, the f32 floor, the frames against the reference's decoded clip and the codec floor (`decode` metrics, directory `[[compare]]`s), the pre-fix and composed paths | D1j ([§21](../adr/0019-fv-lab.md)); a sibling of the `oracle` family (`FV_LAB_SIBLINGS`); `fv-lab oracle --family oracle-r2`; GPU check §21.7, pending §21.8 |
| `fuse` | `fuse-{kernels,trace,switch}` | 10 | DiT block fusions: the kernels bit for bit (ungated), a traced step per workload, fusions off against on (`--off-identity`, `exact` gate) | D1f ([§17](../adr/0019-fv-lab.md)); retired after its GPU check (§17.6); 3 arms skip on the EU volume (`h3-8step`) |
| `attn3` | `attn3-h3-ltx25` | 6 | flash v2 against cuDNN SDPA on FastH3 4-step dense and LTX-2.5 1080p20s dense, VSA bf16 in place against f32 | D1f (§17); retired after its GPU check (§17.6); `FV_ATTN3_ARMS` / `FV_ATTN3_VSA` refused (select by arm) |
| `ltx` | `ltx-versions` | 3 | LTX-2.5 two-stage, LTX-2.3 two-stage, LTX-2.0 19B distilled | D1f (§17); retired after its GPU check (§17.6; the script gated its arms from D1f); `ltx23` / `ltx2` skip on the EU volume |
| `hd` | `hd-canvases`, `hd-10s` | 8 | H3 at the served 1080P canvas: h3-turbo and h3-max at 768p and 1920x1088 / 1088x1920 (`--oversize-canvas`) over `prompts-hd.json`, both at 1920x1088 for 10 s; time and peak memory | D1 S2 ([§22](../adr/0019-fv-lab.md)); the script's derived step (Lanczos compares, post script, keyframe prune) dropped in lockstep; retired after its GPU check (§22.10, §22.19) |
| `hd5` | `hd5-canvases`, `hd-10s` | 8 | `hd` over the 1080P serve gate's five prompts (`prompts-hd5.json`); the gate is `--cells turbo-768p,turbo-1080p,max-768p,max-1080p,turbo-1080p-10s,max-1080p-10s` | D1 S2 (§22.9); a sibling of the `hd` family (the script with `FV_PROMPTS=…/prompts-hd5.json`), frozen with it (`FV_LAB_RETIRED_SIBLINGS`, §22.19); shares `hd-10s` |
| `serve-engine` | `serve-engine-{h3-turbo,ltx-turbo,fastwan21-1.3b}` | 6 | the serve engine's GPU check: h3-turbo, ltx-turbo and fastwan21-1.3b each generated by the CLI, then through `EngineService` + `CudaBackend` (`fv-gpucheck engine --reference`: frames byte-compared with the CLI's, a second job cancelled mid-run) | D1 S3 ([§22](../adr/0019-fv-lab.md)); `--tae-dir` dropped from the script's engine arms in lockstep; weights `volume+r2` (`fastwan21-1.3b` from R2), 150 GB; `FV_ENGINE_CELLS` refused (select by arm); retired after its GPU check (§22.12, §22.19) |
| `sfstream` | `sfstream-rollout` | 2 | SF-Wan 1.3B's open-ended causal rollout (`fv-gpucheck wan stream`, TAEHV per block): `--parity` against the bounded 81-frame path, a 120 s rebased sink-15 run, keep / reset prompt switches (`sfstream-main`); a 600 s memory-growth run (`sfstream-10min`) | D1 S4 ([§22](../adr/0019-fv-lab.md)); its arms read `{tae}` without a TAE gate (Addition 16), the script's `export FASTVIDEO_TAE_DIR` moved into each arm's env in lockstep; weights `volume+r2` (`sfwan21-1.3b` from R2), 150 GB; `FV_SFSTREAM_RUNS`, `FV_SF_PROMPT`, `FV_SF_SWITCH` refused (the cell is their defaults); retired after its GPU check (§22.14, §22.19) |
| `mmaudio` | `mmaudio-sidecar` | 3 | the MMAudio V2A sidecar (`wan gen --audio mmaudio`) on FastWan 1.3B DMD and SF-Wan 1.3B at 480x832x81 (`wan13-audio`, `sfwan13-audio`), then our port alone on the FastWan clip, timed (`ours-v2a`); no upstream parity | D1 S5 ([§22](../adr/0019-fv-lab.md)); the script's branch cut to these arms in lockstep (the upstream, inject and waveform arms gone; their results stand at `778c65e`), its `export`s moved into each Wan arm's env, composite gates (`fastwan21-1.3b-mmaudio`, `sfwan21-1.3b-mmaudio`); weights `volume+r2` (all three trees from R2), 180 GB; `FV_MMAUDIO_PROMPT` refused; retired after its GPU check (§22.16, §22.19) |
| `wan` | `wan-fastwan21-1.3b`, `wan-wan21-t2v-14b`, `wan-wan22-ti2v-oracle`, `wan-wan22-ti2v-5b`, `wan-fastwan22-ti2v-oracle`, `wan-fastwan22-ti2v-5b` | 49 | the Wan family's kernel, decoder, cache and parity regression check: FastWan2.1 1.3B (VSA / dense, the kernel arms with lossy and exact gates, full VAE, no-cache identity, Sol, TeaCache), `kernels-wan` (ungated), SF-Wan 1.3B 81 frames, the TI2V-5B / T2V-14B kernel A/B, T2V-14B's caches and Sol at 50 steps, the TI2V-5B recipe, two module-parity oracle cells (`FV_ORACLE_URL`), fal's canvases, wan-turbo's CLI / engine pair | D1 S6 ([§22](../adr/0019-fv-lab.md)); its `export FASTVIDEO_TAE_DIR` moved into each gated arm's env and `engine-fw22`'s `--tae-dir` dropped in lockstep; weights `volume+r2` (every Wan tree from R2, eleven fetches in a full run, at most 80.41 GB at once), 210 GB, `min_ram_gb = 256`; `FV_SFWAN_CELLS` refused; branch kept until S8 (`sfwan` runs it; §22.18) |
| `gold` | `gold-*` (32) | 32 | every gold variant's canonical cell: the served variant through the serve engine at its gold canvas | fv-lab's own, generated from `lab/gold/` ([below](#the-gold-files-ci-and-the-tier-table)); weights `volume+r2` ([below](#weights-from-the-volume-and-r2-volumer2)) |
| `h3-graphs` | `h3-graphs-turbo{-trace,}` | 4 | h3-turbo at 768p: a traced process with the step graphs on, then per-step sync, the lazy step clock and CUDA graphs of the steps, timed warm, `--off-identity` against lazy | D1e: was `scripts/gpu/experiments/h3-graphs-*.sh` (`FV_CKS=turbo`, deleted after the GPU check, §14.4), [below](#port-one-offs-the-experiment-tiers) |

All eleven above `precision-nvfp4`, `det-short` and `determinism`, `fuse`, `attn3` and `ltx`, and `hd`, `hd5`,
`serve-engine`, `sfstream` and `mmaudio` are **retired from the script** (`scripts/gpu/fv-lab-retired.sh`, ADR-0019
§10; `b4a` §11.6; `det-short`, `determinism` §15.10; `fuse`, `attn3`, `ltx` §17.6; the last five §22.19): their
`runpod-matrix.sh` branches are deleted, and `runpod-matrix.sh <tier>` refuses them (exit 2). The script keeps three
other families, `sfwan`, `wan` and `oracle` (the twenty retired with no tier went in D1g and §22's S1,
[below](#retired-with-no-tier-d1g)).
ADR-0019 §9.1 says why for each and when its branch retires. An arm whose tree is not on the EU volume (`h3-8step`)
is recorded as skipped (`weights incomplete`), as the script recorded it.

What the retired branches had and the tiers do not, dropped with the branches: the `FV_VAE_*`, `FV_LTXFPS*`,
`FV_TECH_CELLS`, `FV_B6_CELLS`, `FV_B4A_CELLS`, `FV_DET_*`, `FV_ATTN3_*`, `FV_ENGINE_CELLS`, `FV_SFSTREAM_RUNS`,
`FV_SF_PROMPT`, `FV_SF_SWITCH` and `FV_MMAUDIO_PROMPT` knobs (ADR-0019 §9.4, §11.5, §15.10, §17.6, §22.19) and the
summaries the branches copied into `live.log`. The two
modes dropped with them are tiers again (owner decision 2026-10-25, [ADR-0019 §12](../adr/0019-fv-lab.md)):

```sh
./fv-lab run precision-nvfp4 --dry-run      # was FV_PRECISION_ARM=ltx-nvfp4 runpod-matrix.sh precision
./fv-lab run ltxvae-gen --dry-run           # was FV_VAE_GEN=1 runpod-matrix.sh ltxvae (all 16 arms)
./fv-lab run ltxvae-gen --cells ltx25-4k5s-gen,ltx25-4k5s-gen-streaming   # only the 4k5s gen pair (no knob did this)
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family precision-nvfp4 --lpips --yes --wait
```

The old knobs pick arms, and `--cells` (`FV_CELLS`) names the same arms. Each list below assumes the mode's other knobs
at their defaults:

| old setting | `--cells` |
|---|---|
| `FV_NVFP4_WORKLOADS=4k5s` | `ltx25-4k5s-sol-bf16,ltx25-4k5s-sol-nvfp4` |
| `FV_NVFP4_WORKLOADS=1080p20s` | `ltx25-1080p20s-sol-bf16,ltx25-1080p20s-sol-nvfp4` |
| `FV_VAE_WORKLOADS=4k5s` (the 8 arms of 4k5s) | `ltxvae-4k5s-fast,ltxvae-4k5s-streaming,ltxvae-4k5s-fast-untraced,ltxvae-4k5s-streaming-untraced,ltxvae-4k5s-fast-mp4,ltxvae-4k5s-fast-tune,ltx25-4k5s-gen,ltx25-4k5s-gen-streaming` |
| `FV_VAE_WORKLOADS=1080p20s` | the same eight with `1080p20s` |
| `FV_VAE_UNTRACED=0` (traced and gen arms, both workloads) | `ltxvae-4k5s-fast,ltxvae-4k5s-streaming,ltx25-4k5s-gen,ltx25-4k5s-gen-streaming,ltxvae-1080p20s-fast,ltxvae-1080p20s-streaming,ltx25-1080p20s-gen,ltx25-1080p20s-gen-streaming` |
| `FV_VAE_UNTRACED=0 FV_VAE_WORKLOADS=4k5s` | `ltxvae-4k5s-fast,ltxvae-4k5s-streaming,ltx25-4k5s-gen,ltx25-4k5s-gen-streaming` |

`--cells` runs the same arms, but the run is not identical. The script never reached the arms its knobs left out. fv-lab
logs each unselected arm as `skip <arm> (not in FV_CELLS)` and a compare of two unselected arms as `skip compare
(frames missing)`. The arm directories, frames and reports are the same.

Their goldens are what the script wrote in that mode at `5987e6a4` (`scripts/gpu/tests/fv-lab-restored-goldens.sh
--check` reproduces them from git history), frozen from the start; fv-lab's dry run and pod equal them with the
family named by the tier and without the mode's `FV_EXTRA_ENV` setting. `FV_VAE_TRACE=0` has no tier. NVFP4 needs
sm_100 / sm_120: the tier is the PRO 6000, as the mode was.

### Port one-offs: the experiment tiers

D1e ([ADR-0019 §14](../adr/0019-fv-lab.md)) decided the ten port experiments F1 moved to `scripts/gpu/experiments/`:
nine are deleted (finished; results in `docs/ports/*.md`, `docs/perf/*.md`, evidence in `artifacts/`, scripts in git at
`22a5fd7b`), and `h3-graphs` is the tier above, for the open item of `docs/perf/h3-denoise-graphs.md` §5 (h3-turbo
was never measured; measured on `f93a961`, ADR-0019 §14.4, and recorded there). Such a tier is listed in `FV_LAB_EXPERIMENTS` (`scripts/gpu/fv-lab-retired.sh`): never a
`runpod-matrix.sh` family, so `runpod-http.sh run` / `all` refuse it as they refuse `gold`. Its commands are the
experiment pod script's own:

```sh
bash scripts/gpu/tests/fv-lab-experiment-goldens.sh [--check]   # cut from h3-graphs-pod.sh: fixtures/experiments/h3-graphs-script.txt, the prompt sets
./fv-lab run h3-graphs --dry-run                                 # fv-lab's own golden: fixtures/h3-graphs-dry-run.txt
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family h3-graphs --wall-cap-s 3600 \
  --fetch-tree --fetch-skip '/frames/|\.cache$' --yes --wait    # the GPU check, ADR-0019 §14.4 (~35 min, ~$1.45)
```

`tests/experiment_tiers.rs` holds each arm to the script's command word for word after the path mapping of §14.3
(the run directory for `/srv/…`, `frames/` for `clip/`, the tier's cache paths for `/fvw/…`, the committed prompt sets,
the runner's cap for `timeout 2400`). The experiment's own scripts were deleted once that GPU check passed (with
`scripts/gpu/experiments/`); the goldens script reads `h3-graphs-pod.sh` from the parent of the deleting commit, and
in a shallow clone keeps the fixtures as they are.

`sol-bench-pod.sh` (`docs/perf/sol-bench.md`) is to become the next experiment tier (owner, 2026-10-28). It is not
one yet: [ADR-0019 §23](../adr/0019-fv-lab.md) gives the rules a tier cannot keep exactly today, the two proposed
cell additions (`estimate_s` / `budget_s`, `env -u`) and the owner's open questions. Until then, run it through
`runpod-http.sh` `FV_POD_SCRIPT` as that doc says.

### `runpod-http.sh` for a retired tier

`runpod-http.sh run [sha]` with `FV_FAMILY` (default `rtx6000`) set to a retired tier runs
`fv-lab pod up --sha <sha> --family <tier> --root <repo> [--cells …] [--baseline-sha …] [--extra-env …] [--lpips] --yes --wait`
instead of its own pod, mapping `FV_CELLS`, `FV_BASELINE_SHA`, `FV_EXTRA_ENV`, `FV_LPIPS=1` and the tree fetch
(`FV_FETCH_TREE=1` → `--fetch-tree`, `FV_FETCH_SKIP` → `--fetch-skip`, `FV_FETCH_MAX_MB` → `--fetch-max-mb`; `run` only) across; `plan [sha]
[tag]` prints fv-lab's plan (no `--yes`). The other families keep the script path. `FV_POD_CAP_S` below the tier's
`wall_cap_s` maps to `pod up --wall-cap-s` (a cap is lowered, never raised). Every other setting of the script path
that fv-lab would not honour is refused, not dropped, unless fv-lab does the same thing anyway:

- no equivalent, refused when set: `FV_POD_FILES`, `FV_POD_SCRIPT`, `FV_POD_FILE`, `FV_PROMPTS`, `FV_PROMPT`,
  `FV_SEED`, `FV_AVOID_MACHINES`, `FV_CREATE_WAIT_S`, `RUNPOD_ALLOWED_CUDA`,
  `FV_KEEP_POD=1`, `RUNPOD_NO_VOLUME=1`, `FV_FETCH_SKIP` / `FV_FETCH_MAX_MB` without `FV_FETCH_TREE=1`, and the retired pair selectors `FV_TECH_CELLS`,
  `FV_B6_CELLS`, `FV_B4A_CELLS`, `FV_DET_CELLS`, arm pickers `FV_ATTN3_ARMS`, `FV_ATTN3_VSA` and group picker
  `FV_ENGINE_CELLS` (select by arm with `FV_CELLS`), `FV_R2_TREES` (the script side's GPU-check form; fv-lab
  fetches a `volume+r2` tier's trees itself), and an `FV_EXTRA_ENV` holding a
  newline (it is read as one line; later entries would be dropped);
- refused unless equal to the tier's value in `lab/tiers.toml`: `FV_GEN_TIMEOUT_S` (`cell_timeout_s`),
  `FV_CONTAINER_DISK_GB`, `RUNPOD_CLOUD_TYPE`, `RUNPOD_GPU_TYPE`, `FV_MIN_RAM_GB` (`min_ram_gb`; refused on a
  tier without one), `FV_SKIP_TAE=1` on a `tae = true` tier, and `RUNPOD_GPU_MAX_DPH` below the tier's
  `max_usd_per_hr` (above it is moot: fv-deploy holds the tier's);
- refused unless equal to fv-lab's fixed value: `FV_IDLE_GPU_MIN` 10, `FV_IDLE_GPU_GRACE_MIN` 30,
  `FV_IDLE_NET_BYTES` 262144, `FV_MIN_DRIVER` 580, `FV_BOOT_WAIT_S` 600, `RUNPOD_VOLUME_ID` / `RUNPOD_VOLUME_NAME` /
  `FV_PLAN_VOLUME` the EU volume, `RUNPOD_IMAGE` the commit's runtime image, `FV_POD_PREFIX` `fv-lab`;
- `FV_EXTRA_ENV` entries no tier reads: the dropped modes' knobs (`FV_PRECISION_ARM`, `FV_NVFP4_WORKLOADS`,
  `FV_VAE_*`, `FV_LTXFPS*`, `FV_TECH_CELLS`, `FV_B6_CELLS`, `FV_B4A_CELLS`, `FV_DET_*`, `FV_ATTN3_*`,
  `FV_ENGINE_CELLS`, `FV_SFSTREAM_RUNS`, `FV_SF_PROMPT`, `FV_SF_SWITCH`, `FV_MMAUDIO_PROMPT`), `hd`'s dropped post-script hook
  (`FV_HD_POST_URL`, `FV_HD_POST_TIMEOUT_S`, ADR-0019 §22; refused on the script path too), `FV_PROMPTS` / `FV_PROMPT` / `FV_SEED`, and what the
  start command sets after them (`FV_WORK`, `FV_SCRATCH`, `FV_RUN_TAG`, `FV_CELLS`, `FV_GEN_TIMEOUT_S`). The two
  restored modes' settings name their tier: `FV_PRECISION_ARM=ltx-nvfp4` says `use FV_FAMILY=precision-nvfp4`,
  `FV_VAE_GEN=1` says `use FV_FAMILY=ltxvae-gen` (refused, not routed: the run would land under another run
  directory and pod name than `FV_FAMILY`'s); `FV_NVFP4_WORKLOADS`, `FV_VAE_WORKLOADS`, `FV_VAE_UNTRACED` say to
  narrow by arm with `FV_CELLS`, and `FV_B4A_CELLS` to select `b4a`'s pair by its arms (`FV_CELLS="<cell>-base
  <cell>-pr"`). A mode setting is answered before any other refusal, and one message lists every
  refused `FV_EXTRA_ENV` key. `fv-lab` itself refuses the same: `pod up --extra-env` this whole set, `fv-lab run`
  the mode knobs in its environment (its own inputs `FV_WORK`, … are not refused, and `FV_PROMPTS` / `FV_PROMPT` /
  `FV_SEED` keep their warning); `runpod-matrix.sh precision` with `FV_PRECISION_ARM=ltx-nvfp4` (or `ltxvae` with
  `FV_VAE_GEN=1`, or the tier's own name) names the tier.

`FV_CELLS` and `FV_EXTRA_ENV` are split on spaces (and `FV_CELLS` on commas) without globbing. `all` mode is refused
(run `kernels`, then `run`), and `FV_FAMILY=gold` is refused for `run` / `all` (no script family: `fv-lab pod up
--family gold`; its `plan` stays the script's body, the payload golden). The binary is `FV_LAB_BIN`, else `fv-lab`
on `PATH`, else `./fv-lab` at the repository root ([Build](#build)); without one the driver says how to build it.
`scripts/gpu/tests/fv-lab-route.test.sh` (CI: `control.yml`) holds all of this, with a fake `FV_LAB_BIN`.

```sh
FV_DRY_RUN=1 FV_FAMILY=b6 FV_BASELINE_SHA=e93e357 bash scripts/gpu/runpod-http.sh run <7>   # prints the fv-lab command
FV_DRY_RUN=1 FV_FAMILY=b4a FV_CELLS="b4a-h3-4step-tea-base b4a-h3-4step-tea-pr" FV_BASELINE_SHA=e93e357 \
  bash scripts/gpu/runpod-http.sh run <7>                           # b4a's GPU gate cell, through fv-lab
FV_DRY_RUN=1 FV_FAMILY=wan bash scripts/gpu/runpod-http.sh run <7>    # a script family: says what it would run
FV_DRY_RUN=1 FV_FAMILY=h3 bash scripts/gpu/runpod-http.sh run <7>     # retired with no tier: refused (exit 2), names rtx6000
```

`FV_DRY_RUN=1` never creates a pod and needs no key. `runpod.sh rtx6000` is retired with the branch (the Taskfile's
`rtx6000` task runs `runpod-http.sh run`). The script's `headline` family, which ran its cells of `fastvideo`,
`rtx6000`, `rtx5090` and `precision` through `fv-lab run <tier> --cells …` into its own run directory, is retired with
no tier ([ADR-0019 §22](../adr/0019-fv-lab.md), S1): the same nine arms are four `fv-lab run <tier> --cells …` calls,
or `fv-lab pod up --family <tier> --cells …` from a host, and the refusal lists them. `--runs-dir` / `FV_RUNS_DIR`
stays.

Cell fields beyond D1a ([ADR-0019 §9.2](../adr/0019-fv-lab.md)):

```toml
[[arm]]
name = "b6-solh3-tree-base"
bin = "baseline"            # FV_BASELINE_SHA's fv-gpucheck, fetched into <scratch>/baseline-<sha>/
env = []                    # a bare `env` before the binary, as the script writes it
tae = "taeh3.safetensors"   # read as {tae}/taeh3.safetensors; fetched once, else the arm is skipped
...

[[arm]]
name = "sfstream-main"      # {tae} without `tae` (ADR-0019 §22.4, Addition 16): not TAE-gated, gated on its
env = ["FASTVIDEO_TAE_DIR={tae}"]   # trees only; allowed only in a tier with `tae = true` (Lab::load refuses others)
...

[[compare]]
baseline = "a"
candidate = "b"
gate = { kind = "lossy" }   # this pair's gate, right after the compare
```

```sh
./fv-lab run b6 --dry-run --baseline-sha <7>      # the baseline tiers need the commit
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family b6 --baseline-sha <base7> --yes --wait
./fv-lab run rtx5090 --dry-run --cells h3-480p-fullopt,h3-480p-fullopt-taeh3   # a subset, by arm
```

## Verdicts and `fv-lab gate`

A cell may carry a `[verdict]` (D1d, [ADR-0019 §11](../adr/0019-fv-lab.md)): what the script's `b4a` family did
around its arms, as data (the branch is retired; `crates/fastvideo-lab/tests/verdict.rs` holds fv-lab to a frozen copy
of it, `fixtures/b4a-script/`). The verdict's arms start from an empty directory; right after each, its exit (`summary.json`),
its evidence files (the `stderr.log` lines matching patterns, the log's `[…]` prefixes stripped) and its line counts
are checked; after the cell's compares, its `--off-identity` pairs and its evidence diffs. The run directory then
holds `<family>-<cell>.verdict.json` (`{"cell","verdict":"pass|fail","why"}`) and, at the end, `<family>.verdict.json`
over the cells that ran; the log says `<family> <cell>: pass|fail (<reasons>)`.

```toml
[verdict]
family = "b4a"
arms = ["b4a-h3-4step-tea-base", "b4a-h3-4step-tea-pr"]     # <cell>-<label>; reasons say "pr:exit"
describe = "verdict also needs both arms exit 0 (summary.json) and the --off-identity compare passed"

[[verdict.evidence]]
file = "cache-evidence.log"
lines = ["h3 sol teacache", "h3 teacache:", "h3 teacache step", "fuse step cache"]   # substrings, * = anything
strip = "bracket-prefixes"
diff = "evidence.diff"            # the two arms' files must be equal

[[verdict.count]]
what = "teacache reused forwards"
lines = ["h3 teacache step *REUSE"]
min = 1
why = "no-reuse"

[[verdict.identical]]
baseline = "b4a-h3-4step-tea-base"
candidate = "b4a-h3-4step-tea-pr"
```

A verdict cell runs only when one of its verdict arms is selected (`--cells`); `fv-lab run b4a --cells
b4a-h3-4step-tea-base,b4a-h3-4step-tea-pr` is what the retired script ran as `FV_B4A_CELLS=b4a-h3-4step-tea`.

```sh
./fv-lab run b4a --dry-run --baseline-sha <7>          # the script's dry run, verdict lines included
./fv-lab gate artifacts/runpod/b4a/<tag>               # verdicts and gates again, offline, into <tag>.regate
./fv-lab gate artifacts/runpod/b4a/<tag> --check --out /tmp/regate   # and compare with the run's own files
./fv-lab gate artifacts/runpod/fastvideo/<tag> --gpucheck ./fv-gpucheck --out /tmp/regate   # a tier with gates
```

`fv-lab gate <run dir>` evaluates a finished run's promotion gates (`fv-gpucheck gate`, CPU only, so any build of
fv-gpucheck) and verdicts again from its files, with no GPU and without running an arm or a compare
([ADR-0019 §11.3](../adr/0019-fv-lab.md)):

- the tier is `--family` or the directory's parent;
- the arms are `--cells`, else the run's `<tier>.arms.json` (the selection `fv-lab run` records), else all of the
  tier's when the run holds them all; a run without the record that holds fewer (a script run with `FV_B4A_CELLS` from
  before the branch went, or one cut short) is refused until `--cells` names its selection, because a guess would pass the cells it never
  finished;
- the output goes to `--out`, default `<run dir>.regate`, never into the run directory;
- `--check` compares each derived file with the run's own (gate reports by status) and exits 1 on any difference,
  whether or not the verdicts pass;
- a gate whose arms did not both run is skipped and counted; one whose arms ran without a `benchmark.json` fails.

Exit 0 when all passed, 1 when one failed or nothing could be evaluated. `pod fetch` takes the verdict files
(`*.verdict.json`, `*.diff`, `<tier>.arms.json`, `<arm>/*-evidence.log`); `runpod-http.sh` only with
`FV_FETCH_TREE=1`. `fv-lab-compare-runs.sh` compares them byte for byte when either run has them (a family verdict's
cell list as a set), and `--expect-verdicts N` fails unless at least N compared the same. `FV_B4A_CELLS` is refused
by `fv-lab run`, `pod up --extra-env` and `runpod-http.sh`'s `FV_EXTRA_ENV`: select a pair by its arms (`--cells
<cell>-base <cell>-pr`, `FV_CELLS`).

## Determinism and the oracle (D1d, second slice)

[ADR-0019 §15](../adr/0019-fv-lab.md). Three tiers keep their `runpod-matrix.sh` branches until each one's GPU check
through both runners (§15.7); their goldens come from the live branches (`fv-lab-goldens.sh`).

- **`[repeats]`** (det-short, determinism): after each arm its frames are hashed (`hash = "arm"`: one hash per arm,
  `<arm>/hashes.json` and `<runs>/hashes.txt`; `hash = "run"`: one per prompt-set run, `<arm>/hashes.txt` and the
  stage digests in `<arm>/digests.txt`) and pruned to `keep`; after every cell `<runs>/determinism.txt` reports each
  hash against the reference (`IDENTICAL` / `DIFFERENT`), as the script writes it. The typed verdict: a cell passes
  when every `judged` arm's hashes equal the reference's (the timed-pick `controls` are reported only); the log says
  `repeats <cell>: pass|fail (…)` and `fv-lab gate <run dir>` decides it again from the fetched hashes (no frames
  needed). No verdict file is written (the script writes none; the GPU comparison is file for file).
- **`[[prompt_file]]`** (determinism): the prompt set an arm reads as `{runs}/<file>`, written as the script's `det_set`.
- **`[reference]`** (oracle): the upstream dump of `oracle-<target>`, fetched from `FV_ORACLE_URL` (space or comma
  separated `http(s)://` base URLs, anything else refused; `FV_ORACLE_WAIT_S`, default 3 h) in the background from the start, read as `{ref}`; the run
  directory gets `ORACLE_REFS_DONE` once every selected reference is here. Unavailable: the cell's first arm is
  recorded as skipped (`reference dump unavailable`). After the cell the dumps (`fresh`) and the reference are removed.
- **Ungated arms** (no `gate`): `compare-dumps` of two dumps, run only when both exist (`requires`), their
  `compare-dumps` lines copied into the log (`echo`).

```sh
./fv-lab run det-short --dry-run                         # the script's dry run
./fv-lab run det-short --cells h3-r1,h3-r2,h3-r3,h3-timed   # was FV_DET_CELLS=h3
./fv-lab run determinism --cells h3-ab,h3-c,h3-timed     # was FV_DET_CELLS="h3-ab h3-c h3-timed"
./fv-lab gate artifacts/runpod/det-short/<tag> --check --out /tmp/regate   # determinism.txt again, and the verdicts
# oracle: the upstream pods first (scripts/gpu/oracle.sh, or runpod-http.sh upstream with FV_KEEP_POD=1), then
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family oracle \
  --cells oracle-fasth3-4step-vsa,oracle-fasth3-4step-vsa-diff,oracle-fasth3-4step-vsa-f32,oracle-fasth3-4step-vsa-f32-diff,oracle-fasth3-4step-vsa-bf16-vs-f32 \
  --extra-env FV_ORACLE_URL=https://<upstream pod>-8000.proxy.runpod.net/upstream/<tag> --yes --wait
```

`FV_ORACLE_TARGETS=<t…>` is `--cells` with the arms of `oracle-<t>` (each target's main arm, its `-diff`, and for the
H3 targets `-f32`, `-f32-diff`, `-bf16-vs-f32`; for Ref2VA `-cond`, `-cond-diff`). Refused by `fv-lab run` and `pod
up --extra-env`: `FV_DET_CELLS`, `FV_DET_REPEATS`, `FV_DET_PROMPT`, `FV_DET_T2V_PROMPT`, `FV_DET_I2V_PROMPT` (also by
`runpod-http.sh`'s `FV_EXTRA_ENV` since the two families retired, ADR-0019 §15.10), `FV_ORACLE_TARGETS`,
`FV_ORACLE_F32`, `FV_ORACLE_OWN_TEXT`, `FV_ORACLE_PREFETCH`, `FV_REF2VA_PROMPT` (the script's `oracle` branch still
reads them), for `oracle` a `FASTVIDEO_DUMP_OPS` other than `0,1,24,47`, and for `determinism` an `FV_ORACLE_URL` (the
script ran the oracle first with it; the tier does not). `pod fetch`
takes `determinism.txt`, `hashes.txt` and each arm's `hashes.txt` / `digests.txt`; `fv-lab-compare-runs.sh` compares
the hashes files byte for byte and each arm's `compare-dumps` lines, except a `[repeats] controls` arm's (read from
`lab/cells`), which are reported, not judged (ADR-0019 §15.7). Still in the script (§15.6): `sfwan13` until §21's
check (it is the `oracle-r2` tier's cell since §21; the
ten LTX clip-metric targets are the tier's extra cells since §20, below), `FV_ORACLE_OWN_TEXT=1`, and
`runpod-http.sh upstream` (the reference pods; `fv-lab oracle`, below, brings them up as fv-lab pods for the
targets that are cells). So the `oracle` branch stays
whole and `oracle.sh` keeps driving it (`FV_ORACLE_TARGETS`, `FV_POD_FILE`, `FV_CREATE_WAIT_S`, which `fv-lab pod up`
lacks); it retires once the driver and the clip-metric cells exist (ADR-0019 §15.10). det-short's opt-in `ltx-sol`
group and other `FV_DET_REPEATS` went with its branch (a tier when wanted). The two retired families' script side is
frozen for `tests/repeats_oracle.rs` at `0f63c56b` (`crates/fastvideo-lab/fixtures/det-script/`).

## An oracle run: `fv-lab oracle` (D1h)

[ADR-0019 §19](../adr/0019-fv-lab.md) (proposed until its GPU check, §19.7). The driver of an oracle run on
fv-deploy's guards: the upstream reference pods of `lab/upstream.toml` (one per stack: `fastvideo` for the H3
targets, with its 192 GB host-RAM floor; `sol-ltx25` for the LTX ones), the wait for their references, the `oracle`
tier's runtime pod with `FV_ORACLE_URL`, and the release (fetch, then delete) of the upstream pods once the runtime
pod has every reference (`ORACLE_REFS_DONE`), when no runtime pod is ready within `--stock-wait-s` (2,100 s), or on any
failure or interrupt.

```sh
./fv-lab oracle --sha <7> --cells oracle-fasth3-4step-vsa,oracle-fasth3-4step-vsa-diff   # the plan, no key
RUNPOD_API_KEY=… ./fv-lab oracle --sha <7> \
  --cells oracle-fasth3-4step-vsa,oracle-fasth3-4step-vsa-diff,oracle-fasth3-4step-vsa-f32,oracle-fasth3-4step-vsa-f32-diff,oracle-fasth3-4step-vsa-bf16-vs-f32 \
  [--upstream-wall-cap-s 7200] [--wall-cap-s 5400] [--up-sha <commit>] [--upstream-image-tag latest|sha-<7>] --yes
./fv-lab check                     # also validates lab/upstream.toml against the oracle cells
./fv-lab schema upstream           # schemas/upstream.v1.json
```

- `--cells` names arms, as `pod up`'s; their cells' `[reference] target`s pick the stacks. `FV_ORACLE_URL` is the
  driver's (refused in `--extra-env`); the runtime side is `fv-lab pod up --wait`, so its refusals apply.
- The upstream pods clone `--up-sha` (default `--sha`; it must be on GitHub) for `scripts/gpu/upstream` and boot
  `ghcr.io/zaitrarrio/fastvideo-rs-upstream-<image>:<--upstream-image-tag>` (default `latest`, pinned to its digest).
- `--refs-wait-s` (default what the upstream cap leaves after the stock wait and 30 minutes, counted from the first
  upstream create) bounds the wait for the references; a target not served by then (or whose upstream run ended without it) is not run, and fails the run.
- Before the first create, the balance less what the account already spends over the run's longest cap, less every
  pod of the run at its price cap over its wall cap and two replaced readiness cycles, must stay at $8.
- `--url-file F` writes the served bases; `--hold-upstream F` keeps the upstream pods until `F` exists (at most their
  cap less 15 min), for a second reader of the same references (§19.7's script side); a failed runtime side, the
  stock wait or an interrupt release them at once.
- Without `ORACLE_REFS_DONE` 1,800 s after the runtime pod is ready the upstream pods go all the same. An interrupt
  stops the runtime side (SIGTERM), deletes the upstream pods without fetching them, then waits for it (600 s, then
  it is killed and its pod deleted). A reused `--tag` is refused. The exit status is the runtime side's when it
  failed with one.
- Results: the runtime run in `artifacts/runpod/oracle/<tag>/` (`pod up`'s fetch), each upstream run in
  `artifacts/runpod/upstream/<tag>/` (without dumps or clips), the JSON report on stdout. `pod reap --account`
  covers `fv-lab-upstream-…` pods.

The upstream pod's body is `runpod-http.sh upstream`'s (`runpod-http.sh plan-upstream`, the goldens
`fixtures/runpod-http-upstream-<stack>.json`) but for the wall-cap watchdog every fv-lab pod carries.
`--fetch-tree [--fetch-skip ERE] [--fetch-max-mb N]` is passed to the runtime side's `pod up` (the whole run
directory: the clip-metric targets' `metrics.txt` and clip copies, below). `sfwan13`: `fv-lab oracle --family
oracle-r2 --cells <its arms>` (ADR-0019 §21); its upstream target fetches `sfwan21-1.3b` from R2 (`lab/upstream.toml`
`r2`, the pod's disk raised to the target's 180 GB, the R2 key as Runpod secret references).

## Derived-clip metrics and extra cells (D1i)

[ADR-0019 §20](../adr/0019-fv-lab.md) (proposed until its GPU check, §20.7). The ten LTX clip-metric targets of the
oracle (`ltx25-{i2v,kf,ref2v,a2v,a2v-i2v,retake,retake-v,retake-a,extend,a2v-guided}`) are cells of the `oracle`
tier, listed under its **`extra_cells`**: a tier's extra cells run after its `cells`, and only when `--cells` /
`FV_CELLS` names one of their arms, so `fv-lab run oracle` without a selection (and its goldens) is the nine
tensor-diff targets as before.

```sh
./fv-lab oracle --sha <7> --cells oracle-ltx25-retake,oracle-ltx25-retake-diff,oracle-ltx25-retake-ownenc,oracle-ltx25-retake-ownenc-diff \
  --fetch-tree --fetch-skip '/frames/'                       # the plan: one sol-ltx25 pod, UP_STEPS weights:ltx25 oracle:ltx25-retake
FV_CELLS=oracle-ltx25-kf,oracle-ltx25-kf-diff ./fv-lab run oracle --dry-run   # the arms and the metric commands
```

After its arms a clip-metric cell runs its **`[[metric]]`** list: each a pass over clips (`{label, path}`) into a
report directory (`{runs}/oracle-<target>-frames/`): a clip that is not there writes `<label> missing <path>` (with
`missing = true`), a present one is copied to `<label>.mp4` (`copy = true`) and measured, in order, by `probe`
(ffprobe stream lines), `clip-vs-clip` (`against`), `kept-frames-vs-source` (`source`, an ffmpeg `select`, `fps`) or
`frame-vs-image` (`frame`, `image`, `width`, `height`: cover and centre crop); each comparison writes `ssim` then
`psnr` lines. `echo = true` copies the whole `metrics.txt` into the log as `[<tag>] <line>`. The commands are the
script's, word for word, and a dry run lists them (`dry-run metric <tag>: ffmpeg …`). `metrics.txt` is a report:
nothing judges it. `fv-lab-compare-runs.sh` compares it with the run tag read alike: the same, or reported when only
ffmpeg's SSIM / PSNR values differ (the clips' mp4 encodes); a probe line, an empty or failed value, or another line
set is a difference (ADR-0019 §20.6).

The goldens: `oracle-extra-dry-run.txt` and `oracle-extra-gates.tsv` are the script's dry run with
`FV_ORACLE_TARGETS` naming the ten (`tests/dry_run.rs`); `tests/repeats_oracle.rs` runs both runners on fakes of
ffmpeg and ffprobe and compares every report and copy.

## `fuse`, `attn3`, `ltx` (D1f)

[ADR-0019 §17](../adr/0019-fv-lab.md). Three more families as tiers. All three passed their GPU check through both
runners on `762ce4d` (§17.5 "Results": every arm that ran byte-identical, `kernels-fuse` 90 `[PASS]`) and are retired
(§17.6): their `runpod-matrix.sh` branches are deleted, `runpod-http.sh run` / `plan` call `fv-lab pod up`, and their
goldens are frozen as the script last wrote them. No new
cell field: `fuse`'s kernel check is an ungated arm (§15), and a cell whose compares follow arms of two pipelines
names both in its descriptive `model` (`model = "h3 ltx2"`). The script's `ltx` branch gated its three arms from
this slice until it went (it had run them ungated, so the LTX-2.3 and LTX-2.0 arms failed on the volume, where their
trees are not; both runners then recorded them as skipped).

```sh
./fv-lab run fuse --dry-run                     # the script's dry run, kernels-fuse first
./fv-lab run fuse --cells kernels-fuse,fasth3-4step-vsa-768p-off,fasth3-4step-vsa-768p-on   # the kernel check and one switch pair
./fv-lab run attn3 --cells fasth3-4step-dense-768p-v2,fasth3-4step-dense-768p-cudnn          # the H3 dense A/B only
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family ltx --yes --wait
```

`FV_ATTN3_ARMS` / `FV_ATTN3_VSA` are refused by `fv-lab run` and `pod up --extra-env`, and since the branch went by
`runpod-http.sh` too (in `FV_EXTRA_ENV` and the host environment); `--cells` / `FV_CELLS` names the arms. The script's `[PASS]` / `[FAIL]` and `/gpu_trace` lines in `live.log` are not
copied (log only: both are in the arms' `stderr.log`).

Not moved (§17.3): `b4b`, `b4c`, `b4c2`. Each cell's verdict reads what no `[verdict]` list types today
(`benchmark.json` fields, cache-directory listings under `$SCRATCH`, CUPTI kernel tables). They and `h3` (§17.4:
`rtx6000` covers it but for the refiner-less Spark fallback, which nothing serves) are retired with no tier
([below](#retired-with-no-tier-d1g)).

## `hd` and `hd5` (D1, S2)

[ADR-0019 §22](../adr/0019-fv-lab.md) (S2, §22.9). `runpod-matrix.sh hd` as a tier: its eight arms
(`lab/cells/hd-canvases.toml`, `hd-10s.toml`), held to the branch's dry run by `fv-lab-goldens.sh`. The branch's
derived step left the script in the same change, so the two runners run the same arms and nothing else: the 768p clips'
ffmpeg Lanczos upscale and its compares with the native 1080p clips, the `FV_HD_POST_URL` post script and the keyframe
prune are gone (their findings are in `docs/serve/h3-1080p-and-upscaler.md`; their commands: `git show
4f8822a0:scripts/gpu/runpod-matrix.sh`). `FV_HD_POST_URL` and `FV_HD_POST_TIMEOUT_S` are refused by fv-lab
(`run`'s environment, `pod up --extra-env`) and `runpod-http.sh` (both paths). Both tiers are **retired from the
script** since their GPU check passed (§22.10 on `d53ea62`; S7, §22.19): `runpod-matrix.sh hd` and `hd5` refuse (exit
2), `FV_FAMILY=hd runpod-http.sh run` (or `FV_FAMILY=hd5`) runs `fv-lab pod up`, and the frozen sibling `hd5` keeps
the fixtures the script wrote with `FV_PROMPTS=…/prompts-hd5.json` (`FV_LAB_RETIRED_SIBLINGS`, which carries that
mode). That setting with `hd` is refused naming `hd5` (`runpod-matrix.sh`, `runpod-http.sh run` / `all` / `plan`
from the host environment or `FV_EXTRA_ENV`, fv-lab's `--extra-env` and `run`): dropping it alone would run `hd`'s
own prompt set under the same arm names. Use `FV_FAMILY=hd5` (`fv-lab pod up --family hd5`).

`hd5` is the 1080P serve gate: the script ran it as `FV_PROMPTS=/opt/fastvideo-rs/scripts/gpu/prompts-hd5.json`, which
fv-lab does not read (`fv-lab run` ignores it with a warning, `pod up --extra-env` refuses it: a cell's workload is its
prompts, ADR-0019 §7), so the prompt set is a cell's `workload.prompt_set`
(`hd5-canvases`). It is a tier of its own, not extra cells of `hd`: its arms keep the script's names, which
`hd_report.py` and `fv-lab report summary --kind hd` read, and a tier's arm names are unique. Its 10 s arms are `hd`'s
cell (`FV_PROMPTS` never reached them).

```sh
./fv-lab run hd --dry-run                       # the script's eight arms, nothing else
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family hd --yes --wait
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family hd5 \
  --cells turbo-768p,turbo-1080p,max-768p,max-1080p,turbo-1080p-10s,max-1080p-10s --yes --wait   # the 1080P gate
./fv-lab report summary --kind hd artifacts/runpod/hd/<tag>    # the gate's run: artifacts/runpod/hd5/<tag>
```

The script's per-arm timing line in `live.log` (`<arm> "denoise_s": …`) is log only and not copied: the numbers are each
arm's `benchmark.json`, which `report summary --kind hd` prints.

## `serve-engine` (D1, S3)

[ADR-0019 §22](../adr/0019-fv-lab.md) (S3, §22.11). `runpod-matrix.sh serve-engine` as a tier: one cell per group
(`lab/cells/serve-engine-h3-turbo.toml`, `-ltx-turbo.toml`, `-fastwan21-1.3b.toml`), each the CLI arm and then the
engine arm that reads the CLI arm's frames as `--reference`, held to the branch's dry run by `fv-lab-goldens.sh`. In the
same change the branch's engine arms lost `--tae-dir "$TAE"`, as the gold engine cells never pass it: `fv-gpucheck
engine` then takes `<weights root>/auxiliary/tae` (the script's `$TAE` only when the volume's TAE copy is complete),
and none of the three recipes reads a tiny decoder (h3-turbo and ltx-turbo name none, fastwan21-1.3b decodes with the
full VAE), so the frames do not change ([ADR-0019 §22.11](../adr/0019-fv-lab.md)). The tier is `weights = "volume+r2"`: `fastwan21-1.3b` is R2-only
and is fetched before `cli-wan-turbo` onto a 150 GB container disk. The branch's `FV_ENGINE_CELLS` (groups `h3 ltx
wan`) is `--cells` by arm; fv-lab refuses it (`fv-lab run`'s environment, `pod up --extra-env`). The branch's removal
of the PNG frames after the family is not copied (disk only: a CLI arm's frames are in its `frames.sha256.json`, an
engine arm's in its `engine/frames` check's `frames_sha256`).

```sh
./fv-lab run serve-engine --dry-run             # the script's six arms, nothing else (plus the R2 plan)
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family serve-engine --yes --wait
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family serve-engine --cells cli-h3-turbo,engine-h3-turbo --yes --wait
FV_FAMILY=serve-engine bash scripts/gpu/runpod-http.sh run <7>   # the same fv-lab pod up (retired, §22.19)
```

Retired from the script since its GPU check passed (§22.12 on `3bc6d03`; S7, §22.19): `runpod-matrix.sh
serve-engine` refuses (exit 2), and `FV_ENGINE_CELLS` is refused by `runpod-http.sh` too (`FV_LAB_NOT_READ`, and in
the host environment), with the arms to pass as `FV_CELLS`. The script side's GPU-check form (`FV_R2_TREES=…`, §22.12)
is refused on the fv-lab route; its body stays a frozen fixture (`runpod-http-serve-engine-r2.json`).

Each engine arm's own checks (`engine/identical_to_cli`, `engine/cancel`) are in its `stderr.log` and
`gpucheck-out/engine.json`; the engine writes its frames under `<arm>/out/<job id>/frames`, so only the CLI arms
have a `frames.sha256.json`.

## `sfstream` (D1, S4)

[ADR-0019 §22](../adr/0019-fv-lab.md) (S4, §22.13). `runpod-matrix.sh sfstream` as a tier: one cell
(`lab/cells/sfstream-rollout.toml`) with the branch's two arms, `sfstream-main` (`--parity`, a 120 s rebased sink-15
run, the keep and reset prompt switches) and `sfstream-10min` (600 s in 30 s windows), held to the branch's dry run by
`fv-lab-goldens.sh`. `wan stream` decodes every block with TAEHV (taew2_1), found through `FASTVIDEO_TAE_DIR`: the arms
set `FASTVIDEO_TAE_DIR={tae}` without a `tae` file, which §22.4's Addition 16 allows. Such an arm is not TAE-gated (no
`fetch-tae.sh`, no `tae weights missing` skip) and gates on its tree only, as the script's `gated_cell` does; it is
allowed only in a tier with `tae = true`, whose pod fetches the files when the volume lacks them. In the same change the
branch's `export FASTVIDEO_TAE_DIR="$TAE"` became each arm's `env FASTVIDEO_TAE_DIR="$TAE"`, so the dry run (and the
golden) shows it, and the branch's taew2_1 fetch is skipped in a dry run. The tier is `weights = "volume+r2"`:
`sfwan21-1.3b` (28.93 GB) is R2-only and is fetched before `sfstream-main` onto a 150 GB container disk. The branch's
`FV_SFSTREAM_RUNS`, `FV_SF_PROMPT` and `FV_SF_SWITCH` are the cell's defaults; fv-lab refuses them (another run list or
prompt is another cell).

```sh
./fv-lab run sfstream --dry-run                 # the script's two arms, each with its FASTVIDEO_TAE_DIR (plus the R2 plan)
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family sfstream --cells sfstream-main --wall-cap-s 3600 \
  --fetch-tree --fetch-skip '\.png$|\.cache$' --yes --wait
FV_FAMILY=sfstream FV_CELLS=sfstream-main bash scripts/gpu/runpod-http.sh run <7>   # the same fv-lab pod up (§22.19)
```

Retired from the script since its GPU check passed (§22.14 on `b7ddf5f`; S7, §22.19): `runpod-matrix.sh sfstream`
refuses (exit 2), and `FV_SFSTREAM_RUNS`, `FV_SF_PROMPT` and `FV_SF_SWITCH` are refused by `runpod-http.sh` too
(`FV_LAB_NOT_READ`). Its script-side GPU-check body stays a frozen fixture (`runpod-http-sfstream-r2.json`).

`wan stream` writes no `<arm>/frames` (its report is `gpucheck-out/wan.json`, its window sheets `gpucheck-out/sheets/*.jpg`), so the runs'
compare (`fv-lab-compare-runs.sh`) counts each arm as the same with no frames and exits 1 ("nothing was
compared"). The tier's GPU check reads the runners' identity from the arm's own records instead: `wan.json`'s checks,
its `run/<name>` notes (each window's latent hash, decoded-pixel statistics and fresh-decode differences), its
`parity/absolute` note, and the window sheets `gpucheck-out/sheets/*.jpg`, each equal across the two pods
([ADR-0019 §22.14](../adr/0019-fv-lab.md), with the exact commands).

Addition 16 in practice: an arm that reads `{tae}` without a `tae` file is not TAE-gated, so a TAE file missing when
it runs is its failure (a `wan gen` arm in the `auto` decoder mode would silently decode with the Wan VAE instead).
The tier's `tae = true` only makes the pod start run `fetch-tae.sh` once; fv-lab does not retry per arm as the
script's branches do with `fetch_taehv.sh`. Every `{tae}/<file>` such an arm names must be one `fetch-tae.sh` fetches
(`tests/lab_files.rs`), and `taew2_2` is not one yet (S6 adds it before `wan` reads it).

## `mmaudio` (D1, S5)

[ADR-0019 §22](../adr/0019-fv-lab.md) (S5, §22.15). `runpod-matrix.sh mmaudio` as a small tier, without upstream
parity: one cell (`lab/cells/mmaudio-sidecar.toml`) with `wan13-audio` and `sfwan13-audio` (FastWan 1.3B DMD and
SF-Wan 1.3B with `--audio mmaudio`: the soundtrack written as `frames/output.wav` and muxed into `frames/output.mp4`,
the split in `benchmark.json`'s `audio`) and `ours-v2a` (our port alone, `fv-gpucheck mmaudio v2a`, on
`{runs}/wan13-audio/frames/output.mp4`, three timed runs). In the same change the script's branch was cut to these
three arms (the upstream venv and clone, the timed upstream sidecar, the inject arms with `compare-dumps` and the
waveform comparisons went: a real-run change; their results stand at `778c65e`, `docs/ports/mmaudio.md`), its
`export FASTVIDEO_MMAUDIO_WEIGHTS` / `FASTVIDEO_TAE_DIR` became each Wan arm's env entries, `ours-v2a` is gated on
`mmaudio-44k-v2`, and its taew2_1 fetch is skipped in a dry run. An arm's `trees` must be exactly its gate cell's
(`tests/dry_run.rs`), and the Wan arms read `mmaudio-44k-v2` too, so they gate on the composite `verify-weights.sh`
cells `fastwan21-1.3b-mmaudio` and `sfwan21-1.3b-mmaudio`. The tier is `weights = "volume+r2"`: all three trees are
R2-only; fv-lab fetches each before the first arm that reads it and drops a tree no later arm reads to make room, at
most 50.68 GB at once on a 180 GB disk. The branch's `FV_MMAUDIO_PROMPT` is the cell's prompt; fv-lab refuses it.
The MMAudio checkpoints are CC-BY-NC 4.0: the tier reads the owner's R2 copy and redistributes nothing; whether the
user-facing knob (`FASTVIDEO_WAN_AUDIO`, `--audio mmaudio`) stays offered is an open owner question (§22.6).

```sh
./fv-lab run mmaudio --dry-run                  # the cut branch's three arms (plus the R2 plan)
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family mmaudio --cells wan13-audio,ours-v2a --wall-cap-s 3600 \
  --fetch-tree --fetch-skip '\.png$|\.cache$' --yes --wait
FV_FAMILY=mmaudio FV_CELLS="wan13-audio ours-v2a" bash scripts/gpu/runpod-http.sh run <7>   # the same fv-lab pod up (§22.19)
```

Retired from the script since its GPU check passed (§22.16 on `9e6d09a`; S7, §22.19): `runpod-matrix.sh mmaudio`
refuses (exit 2), and `FV_MMAUDIO_PROMPT` is refused by `runpod-http.sh` too (`FV_LAB_NOT_READ`). Its script-side
GPU-check body stays a frozen fixture (`runpod-http-mmaudio-r2.json`).

`wan13-audio` writes PNG frames, so `fv-lab-compare-runs.sh` compares them; `ours-v2a` writes none ("no frames").
The audio has no frames record: the tier's GPU check reads the runners' identity from the arms' own records instead,
each equal across the two pods: `wan13-audio`'s `gpucheck-out/wan.json` (status, checks, the `FASTVIDEO_*` env) and
`benchmark.json` (`decoder`, `audio`), its soundtrack `frames/output.wav` and `frames/cold/output.wav`, and
`ours-v2a`'s `mmaudio.wav` / `mmaudio.f32` / `mmaudio.json` ([ADR-0019 §22.16](../adr/0019-fv-lab.md), with the exact
commands).

## `wan` (D1, S6)

[ADR-0019 §22](../adr/0019-fv-lab.md) (S6, §22.17). `runpod-matrix.sh wan` as a tier of six cells, the branch's 49
commands (45 gated arms, `kernels-wan` and three `compare-dumps`), 29 compares, 20 gates and 2 references in its order.
Six, not one per helper call: a compare or gate reads arms of its own cell, and the 1.3B baseline `wan13-dmd` is
compared and gated against arms 18 to 22, so the first 22 arms are one cell (`wan-fastwan21-1.3b`, its compares placed
with `after`); the two oracle blocks are `[reference]` cells (`wan22-ti2v`, `fastwan22-ti2v`: `lab/upstream.toml`'s
`fastvideo` stack, their trees fetched from R2 upstream); two compares across cells are directory compares
(`wan5b--fw22`, `sfwan13-81f-fullvae--sfwan13-81f-flash`, in the last cell). In the same change the script's
`export FASTVIDEO_TAE_DIR="$TAE"` became each gated arm's env entry, `engine-fw22` lost its `--tae-dir` (as
`serve-engine`'s), and its TAE fetch is skipped in a dry run. The arms read `{tae}` without a TAE gate (Addition 16):
`fetch-tae.sh` fetches `taew2_2` too since S6 (the oracle arms' `--taehv`), and `benchmark.json`'s `decoder` shows
whether the distilled presets decoded with TAEHV or fell back to the Wan VAE. Every Wan tree is R2-only: fv-lab drops
and fetches per arm (eleven fetches, 428.10 GB in a full run, at most 80.41 GB at once on 210 GB); `min_ram_gb = 256`
for the 14B arms. `runpod.sh matrix` runs it through the image's fv-lab, so it needs an image whose `lab/` has the
tier (`runpod.sh` ships only `scripts/gpu/`; an image from before S6 stops with fv-lab's "no tier or cell named", as `ltx` since §17.6). `sfwan` still runs the branch's SF-Wan arms
(`FV_SFWAN_CELLS`, which fv-lab refuses: `--cells` on this tier).

```sh
./fv-lab run wan --dry-run                      # the branch's 49 commands in its order (plus the R2 plan)
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family wan \
  --cells wan13-dmd,wan13-dmd-dense,kernels-wan,cli-fw22,engine-fw22 --wall-cap-s 3600 \
  --fetch-tree --fetch-skip '\.png$|\.cache$' --yes --wait
# the script's side of the same run (until S8): the two trees those arms read, staged at once (53.41 GB + 120 GB)
RUNPOD_API_KEY=… FV_FAMILY=wan FV_CELLS="wan13-dmd wan13-dmd-dense kernels-wan cli-fw22 engine-fw22" \
  FV_R2_TREES="fastwan21-1.3b fastwan22-ti2v-5b" FV_CONTAINER_DISK_GB=174 FV_MIN_RAM_GB=256 FV_FETCH_TREE=1 \
  FV_FETCH_SKIP='\.png$|\.cache$' bash scripts/gpu/runpod-http.sh run <7>
# the two oracle cells: one fastvideo upstream pod making both references (ADR-0019 §22.18)
RUNPOD_API_KEY=… ./fv-lab oracle --family wan --sha <7> \
  --cells wan5b-oracle,wan5b-oracle-diff,wan5b-oracle-exact,wan5b-oracle-exact-diff,fw22-oracle,fw22-oracle-diff --yes
```

## Retired with no tier (D1g)

[ADR-0019 §18](../adr/0019-fv-lab.md) (coordinator decision 2026-10-26). Eleven `runpod-matrix.sh` families left the
script without becoming tiers, listed in `FV_SCRIPT_RETIRED` (`scripts/gpu/fv-lab-retired.sh`), which says what
replaces each (`fv_script_retired_why`):

| family | what replaces it |
|---|---|
| `h3` | the `rtx6000` tier runs every arm but the refiner-less Spark fallback and the cold streamed-encoder runs, which nothing serves; the `h3` commands: `git show b019ccb2:scripts/gpu/runpod-matrix.sh` |
| `b4b`, `b4c`, `b4c2` | nothing: finished experiments, their GPU gates passed when their changes merged (#179, #177, #178; `artifacts/runpod/{b4b,b4c,b4c2}/`); a later change to the fusions or the prompt cache gets its own byte-identity run (the `b4a` / `techniques` shape) and the CPU fixtures (`rt/prompt_cache/equality.rs`, `rt/fuse/equality.rs`) |
| `hunyuan`, `sana-video` | nothing runs them: trees not on the EU volume, no variant or gold file serves them; their arms by hand: `docs/ports/sana-video.md` (`sana-video`), or `git show b019ccb2:scripts/gpu/runpod-matrix.sh` |
| `b200` | nothing: the US volume is gone (`runpod.sh b200` is refused too) |
| `writer` | nothing: a one-time measurement (commit `a8ecfec6`) |
| `precision-debug` | the `precision` tiers run the same precision variants, but not its per-step CUPTI trace (`FASTVIDEO_GPU_TRACE_STEP`) or its `FASTVIDEO_DUMP_DIR` + `compare-dumps` diffs; those by hand: `git show b019ccb2:scripts/gpu/runpod-matrix.sh` |
| `identity` | the `det-short` / `determinism` tiers |
| `h3attn` | nothing: its arms are recorded in `docs/techniques.md` "H3 attention arms", none open |

They are not fv-lab tiers, so nothing routes them to fv-lab: `runpod-matrix.sh <family>`, `runpod-http.sh run` / `all`
/ `plan` (dry run or not, before the key) and `runpod.sh gpu` / `matrix` / `plan` refuse them with exit 2 and that
pointer. `runpod.sh gpu` starts `ltx` only now and `matrix` runs `ltx` then `wan`. Their branches are in
`git show b019ccb2:scripts/gpu/runpod-matrix.sh`; `b4-verdict.sh` went with the B4 branches (`b4a`'s verdict is held
to its frozen copy, `fixtures/b4a-script/`). `fv-lab-route.test.sh`, `runpod-plan.test.sh` and `tests/dry_run.rs` hold
this.

Nine more joined the list in [ADR-0019 §22](../adr/0019-fv-lab.md)'s first slice (S1, FVID-2026-10-26-d1-remaining),
the same way: refused by every driver, branches deleted, nothing routed to fv-lab. Their pointers also say what is
lost; their commands are in `git show 4f8822a0:scripts/gpu/runpod-matrix.sh`.

| family | what replaces it | lost |
|---|---|---|
| `sfquality` | nothing: the R12 long-rollout study is finished and its outcome shipped (the 15-frame sink, the causal session caps; `docs/ports/wan.md` R12); the `sfstream` tier keeps a pod check of the rollout | the sweep in one command, the f32 and second-seed rollouts (by hand: `fv-gpucheck wan stream --run <spec>`) |
| `headline` | its nine arms as four `fv-lab run <tier> --cells …` calls: `fastvideo`, `rtx6000`, `rtx5090`, `precision` (the refusal lists the arms) | the nine arms on one pod, and its use on a new GPU type (each tier fixes its `gpu`) |
| `speechtest` | nothing: `docs/ports/mmaudio.md` "Speech" answers it; the `mmaudio` tier keeps the sidecar checked on a pod | the Whisper check in one command (`scripts/gpu/speech_transcribe.py` by hand) |
| `cold` | nothing: no result was recorded | the page-cache-cold end-to-end time with load included (sol-engine's e2e) |
| `eval` | `precision`'s f32act / bf16act pairs with lossy gates, `fuse`'s `exact` gates with an OFF arm, prompt sets in cells, `--lpips` on any run | the combination in one run, the `BF16_ACT=0` OFF-identity gate, the five-prompt LPIPS pairs, the LPIPS self-test on a pod (by hand: `fv-gpucheck lpips --weights <dir>`) |
| `ltxoffload` | nothing: its numbers are in `docs/ports/ltx25.md`; `b6-ltx25-offload-cpu` still checks `--offload cpu` across builds | LTX `cpu`-against-resident identity, the 4K `cpu` peaks |
| `solbench`, `sol-lingbot`, `sol-cosmos3` | `scripts/gpu/sol-bench-pod.sh`, which the sol-engine benchmark runs (`docs/perf/sol-bench.md`) | per-arm compares and lossy gates, `wan13-sol-easycache`, `wan13-sol-attn`, `a14b-sol-mxfp8`, LingBot on three prompts, LingBot's B200 `FV_LINGBOT_RESIDENCY=both` path (`sol-bench-pod.sh` hard-codes `--residency swap`) (none ever recorded) |

## On a GPU box: `fv-lab run`

```sh
FV_WORK=/workspace FV_SCRATCH=/fvscratch FV_RUN_TAG=<tag> fv-lab run rtx6000      # every arm
fv-lab run rtx6000 --cells sol-h3,ltx25-two-stage                                 # a subset (FV_CELLS)
fv-lab run rtx6000-ltx25                                                          # one cell
```

The pod's start command does this. It writes `runpod-matrix.sh`'s run directory (`<scratch>/runs/<tier>/<tag>/`:
`live.log`, per arm `summary.json`, `stdout.log`, `stderr.log`, `nvidia-smi-{start,end}.txt`, frames,
`benchmark.json`, `compare/`, `gate/`, `done.json`). It also records the pod's memory, from its memory cgroup
([ADR-0019 §16.10](../adr/0019-fv-lab.md#1610-the-host-ram-floor-min_ram_gb-2026-10-26)):

- at start, `box.txt` gets `cpus=… mem=…G` and `cgroup=v2 <dir> pod_ram_limit_mib=… oom_kills=…`: the pod's RAM
  limit, not the host's `free`;
- each arm's `summary.json` gains `peak_ram_mib`, `ram_limit_mib`, `pod_peak_mib_cumulative` and `oom_kills`, and
  `live.log` a `ram <arm>: …` line. `peak_ram_mib` is sampled each second as the whole cgroup's anon + shmem (no page
  cache), so it includes the runner and anything running beside the arm, not the arm alone. `ram_limit_mib` is the
  limit of the runner's own cgroup only (an ancestor's tighter limit is not read). `pod_peak_mib_cumulative` is the
  cgroup's peak since the pod started, page cache included, and `null` on cgroup v2 before Linux 5.19 (no
  `memory.peak`). `oom_kills` counts the kernel's OOM kills in the cgroup during the arm;
- an arm the kernel OOM-killed (exit 137, often nothing printed) gets `OOM <arm>: the kernel OOM-killed <n>
  process(es) in this pod during the arm (host RAM, not GPU …)`;
- without a memory cgroup the fields are left out.

Settings, as flags or the script's variables: `--work`
(`FV_WORK`), `--scratch` (`FV_SCRATCH`), `--tag` (`FV_RUN_TAG`), `--runs-dir` (`FV_RUNS_DIR`, the run directory
itself instead of `<scratch>/runs/<tier>/<tag>`), `--cells` (`FV_CELLS`), `--gpucheck`
(`FV_GPUCHECK`), `--cell-timeout-s` (`FV_GEN_TIMEOUT_S`, default the tier's), `FV_LPIPS=1`, `FV_GATE_POLICY`. Exit 2
when the box cannot run anything (no fv-gpucheck, no nvidia-smi, no cuDNN), as the script's FATAL.

## From a laptop or the session container: `fv-lab pod`

```sh
# The plan: the Launch and the exact POST /pods body. No key, no network.
./fv-lab pod up --sha <7> --family rtx6000

# Create the pod (runtime image at <7>, pinned to its digest), wait until it serves its run directory:
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family rtx6000 --yes
# … it prints the pod id and the tag; the pod runs `fv-lab run rtx6000` on its own.

RUNPOD_API_KEY=… ./fv-lab pod status <pod id>           # provider state, server, last log lines
RUNPOD_API_KEY=… ./fv-lab pod fetch <pod id> --wait --down   # wait for DONE, fetch, delete
RUNPOD_API_KEY=… ./fv-lab pod down <pod id>             # delete now (no pod record needed)
RUNPOD_API_KEY=… ./fv-lab pod down --from-ledger        # every pod the ledger shows created and not deleted
RUNPOD_API_KEY=… ./fv-lab pod reap [--yes]              # pods recorded here, past their cap + 10 min
RUNPOD_API_KEY=… ./fv-lab pod reap --account [--yes]    # any fv-lab-<tier>-<sha7>-<MMDDHHMMSS>-<hex4> pod past its tier's cap

# Or all of it in one command (as runpod-http.sh run):
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family rtx6000 --yes --wait
```

Options of `up`: `--cells a b` (only these arms), `--extra-env K=V …` (`FV_EXTRA_ENV`), `--lpips`, `--tag`,
`--state-dir`. The run tag is `<sha>-<MMDDHHMMSS>-<4 random hex>`, and `up` refuses to create when a live pod already
has the name. Results land in `artifacts/runpod/<tier>/<tag>/` (the same place and files as
`runpod-http.sh`, plus `pod.json`); the ledger, pod records and backstop logs in `artifacts/fv-lab/`. Run `up` from a
host that outlives the tier's wall cap (4 h for `rtx6000`): the backstop process runs there. The pod also deletes
itself after 10 idle GPU minutes (the start command's idle watchdog, with the pod-scoped key Runpod injects),
and at the tier's wall cap after its container starts (the cap watchdog, ADR-0023 §4), whatever happens to that host.
Both use GraphQL `podTerminate`: the pod-scoped key gets HTTP 403 on a REST DELETE.
After an interrupted `up` (its pod record is only written once the pod is ready), `pod down --from-ledger` deletes
what the ledger shows created and not deleted.

`fetch` (and `up --wait`) take the summaries as `runpod-http.sh` does; with `--fetch-tree` the whole run directory
instead (`FV_FETCH_TREE=1`: frames, mp4s, wavs, every file), less what `--fetch-skip` matches (paths
`<tier>/<tag>/…`; an ERE of alternatives of literal text, `.`, `\`-escaped punctuation, `^` and `$`, e.g.
`'/frames/'` or `'\.png$|\.cache$'`; anything else is refused before the pod is created) and files above
`--fetch-max-mb` MiB (default 400, 0 no limit; checked while the file streams). `up` takes them only with `--wait`.
A pattern starting with `-` is refused (clap reads it as an option). The pod's listing is not trusted: an entry that
is not a single name inside the run directory (absolute, `..`, `.`, a `/`, `\` or NUL once decoded, a malformed `%`
escape) is refused and logged, and the walk stops 16 levels down. The top-level listing is retried 5 times, and
`RUN_FILES` and `DONE` (what the run is judged by) are fetched whatever the skip says (ADR-0019 §13.3).

`fetch` (and `up --wait`) fail (exit 1) unless the fetched `DONE` says `done <tag>` and `matrix.out` holds
`matrix_exit=0`, and when the host's driver was too old; the files fetched so far are kept. `--down` and `up --wait`
delete the pod whatever happened, then report.

Every arm leaves `<arm>/frames.sha256.json` (each frame's SHA-256, written by `fv-lab run` and by `runpod-matrix.sh`
alike, fetched by both drivers). Two runs of one commit compare with

```sh
bash scripts/gpu/fv-lab-compare-runs.sh <fv-lab run dir> <runpod-http run dir>
```

(same arms, same exits and skips, byte-identical frames records, and for the D1b families the same compare verdicts
and the same gate reports); the D1a GPU gate is that command after one `fv-lab
pod up --sha <7> --family rtx6000 --yes --wait` and one `runpod-http.sh run <7>` of the same commit
([ADR-0019 §8](../adr/0019-fv-lab.md)). The commit must be one whose runtime image carries fv-lab and the frames record:
the final head of PR #171 or later. ADR-0019 §9.5 has the D1b families' runs: the cheapest cells per family and their
cost.

## Reports: `fv-lab report summary` and `report table`

What the report scripts printed, read from a fetched run directory's files (ADR-0019 §13); no GPU, no network.

```sh
./fv-lab report summary artifacts/runpod/b6/<tag>        # the lines b6's script branch copied into live.log
./fv-lab report summary artifacts/runpod/<dir> --family h3arms   # the tier when the parent is not its name
./fv-lab report summary --kind hd artifacts/runpod/hd/<tag>   # hd_report.py, without keyframe sharpness (no --jpg-out)
./fv-lab report summary --kind h3-graphs DIR             # was experiments/h3-graphs-summary.py
./fv-lab report summary --kind h3-tier-options DIR --json OUT   # was h3-tier-options-summary.py
./fv-lab report summary --kind h3-trim-nvfp4 DIR --json OUT     # was h3-trim-nvfp4-summary.py
./fv-lab report summary --kind fasth3-nvfp4 DIR --json OUT      # was fasth3-nvfp4-summary.py
./fv-lab report table --kind sol-bench artifacts/runpod/sol-bench/<date>/*   # was sol_bench_table.py; results.json beside them (--json F, --stdout)
./fv-lab report table --kind timings <validate.sh run dir>     # timings.sh: <run>/timings.md, printed (--out F, --stdout)
```

`--kind tier` (the default) prints, for `techniques`, `b6`, `b4a`, `h3arms`, `trace`, `ltxfps`, `ltxvae` and
`ltxvae-gen`, the summary lines their `runpod-matrix.sh` branches wrote into `live.log` (compare fields, arm timings,
load-stage lines, reuse counts, gate verdicts, the `/gpu_trace` line, sink checks, decode lines), without the
`[HH:MM:SS] [family] ` stamp, for runs of either runner. The other tiers' branches printed none. `hd_report.py`
(the keyframe sharpness needs numpy) and `timings.sh` (`validate.sh` calls it) stay; `sol_bench_table.py` and the four
experiment `*-summary.py` are deleted (ADR-0019 §13.5).

## The gold files: CI and the tier table

Design: [ADR-0023](../adr/0023-lab-report-and-ci-bridge.md) (plan D3, D4).

```sh
./fv-lab gold check                         # parse, resolve against families/, validated commits exist, the age table
./fv-lab gold check --max-age-days 30       # also fail a validated variant older than 30 days (default: warn only)
./fv-lab gold cells --write                 # regenerate lab/cells/gold-*.toml after a gold or family file changes
./fv-lab gold cells --check                 # CI: the committed gold cells are current
./fv-lab run gold --all --dry-run           # the gold tier: one serve-engine arm per gold variant
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family gold --yes --wait   # the gold runs -> artifacts/runpod/gold/<tag>/
./fv-lab report tiers --runs artifacts/runpod/gold/<tag>                # docs/perf/tiers.md from that run set
./fv-lab report tiers --runs artifacts/runpod/gold/<tag1>,artifacts/runpod/gold/<tag2>   # a gold run split with --cells, as one
./fv-lab report tiers --check               # CI: docs/perf/tiers.md is what the run set its header names renders to
```

`gold check` prints Markdown (the table CI appends to the job summary) and exits 1 on a gold file that does not parse
or resolve, or a `[last_validated].commit` that is not a commit here (outside a git work tree the commits are not
checked, and it says so). A variant never validated is a warning.

`report tiers` writes one row per gold file. A variant is measured only by the arm of its name in the run set (the gold
tier names its arms by variant id), with every value read from that arm's `summary.json`, `benchmark.json` or
`gpucheck-out/engine.json` and the run's `pod.json`; anything else is "not measured" with the reason. Options:
`--gold DIR` (other gold files), `--out FILE`, `--stdout`.

`--runs` takes several run directories (comma separated or repeated) of one tier and one commit, e.g. a gold run
split in two with `--cells` ([ADR-0019 §16.7](../adr/0019-fv-lab.md#167-the-gpu-check-the-coordinators)), and
renders them as one run set: the arms are merged, each row reads its own run's `pod.json` and counts its validation
age to that run's end, and the header names every run (its tag, the arms shown from it, end, GPU and price, and the
copies shown from another run). Run directories must not contain commas, since `--runs` splits on them. Refused:
runs of different tiers; with several runs, a run whose commit is not known (no `pod.json` sha and no tag head of at
least 7 hex digits) or runs of different commits; the same directory twice (`./a` is `a`); an arm that passed (exit
0) in more than one run; and a `--runs` that names no directory. An arm in several runs is shown from the copy that
passed, else the last that failed, else the last skipped, in `--runs` order: a failed arm rerun in a later run shows
the rerun. `--check` reads the runs back from the header (`` Run set: `a`, `b` ``). One run renders exactly as before.

Workflows: `gold-check.yml` runs the three checks on lab, family and lab-crate changes; `gold-run.yml`
(`workflow_dispatch`, input `confirm: spend`) runs `pod up --family gold --yes --wait` and renders the table as an
artifact. It needs the `RUNPOD_API_KEY` repository secret, which is the owner's to add; without it the job refuses.

## Weights from the volume and R2: `volume+r2`

Design: [ADR-0019 §16](../adr/0019-fv-lab.md#16-weights-by-tree-from-two-places-weights--volumer2). A tier's
`weights = "volume+r2"` (the `gold`, `oracle-r2`, `serve-engine`, `sfstream`, `mmaudio` and `wan` tiers) reads every tree the EU volume holds from the volume and fetches every other
tree from the R2 weights store onto the container disk, right before the first arm that reads it:

- `{weights}` is `/fvscratch/weights`: a symlink per volume tree into `/workspace/weights`, the fetched trees beside
  them. The volume is never written.
- Per arm, when a tree it reads is not on the disk yet: the fetched trees it does not read are removed, then
  `r2-weights.py plan <dest>@<rev> --pin weights-r2.tsv` (the revisions and manifest SHA-256s of
  `scripts/gpu/weights-r2.tsv`) and `r2-fetch.py` fetch the missing ones under the arm's cap. Logs: `<arm>/r2-fetch.log`,
  `<arm>/r2-fetch.json`, the `r2 <arm>: …` lines of `live.log`. The runtime image has no Python (by design), so the
  first fetch installs `python3` with apt (`r2-apt.log`, at most 10 min; a failure is retried at the next fetch).
- A failed fetch records the arm as skipped (`r2 fetch failed`); a tree in neither place as `weights incomplete`.
- The root is never on the volume: run by hand, pass `--scratch` (or `FV_SCRATCH`) on the container disk, e.g.
  `fv-lab run gold --all --work /workspace --scratch /fvscratch`; without it (the scratch disk is then the work
  directory), or with a scratch under `/workspace`, the run is refused (exit 2).
- The idle watchdog of a `volume+r2` pod waits 85 minutes at 0 % GPU (a fetch uses none), not 10.
- The gold tier's pod asks for at least 256 GB of host RAM (`min_ram_gb = 256`, Runpod `minRAMPerGPU`): its 14B arm
  peaked at 156,004 MiB of process memory at `7e580e5` (the f32 DiT and text encoder, each held twice on the host)
  right after fetching 84 GB of trees, and Runpod's GB are 10^9 bytes (ADR-0019 §16.10 "Raised to 256"). Fewer hosts
  qualify; a create none can place is fv-deploy's `no capacity` (nothing created, nothing billed). A tier
  without `min_ram_gb` gets Runpod's default floor (8 GB per GPU) and the host's share. `pod up` logs the floor, the
  pod's `ram=…GB vcpu=…` from Runpod's answer, and records them in `pod.json` (ADR-0019 §16.10).
- Any pod's run gets `FV_RUN_DEADLINE` from `pod up` (the cap less 20 min): arms that could not finish by then are
  skipped as `wall cap near`, so `DONE` is written, and `pod up --wait` stops waiting 15 min before the cap and
  fetches what is there.
- The R2 key is two Runpod secret references in the pod's env (`fv_r2_access_key_id`, `fv_r2_secret_access_key`, filled
  in by Runpod); `fv-lab run` gives it to the plan step only. Nothing to set locally. The secrets the pods reference
  (and the upstream pods' `fv_hf_token`): docs/ops/runpod-secrets.md.
- `fv-lab run gold --all --dry-run …` prints the plan: one `dry-run weights …` line (18 trees, 530.74 GB, at most
  147.86 GB at once) and a `dry-run r2 <arm>: drop …; fetch …` line before each arm that fetches. The gold tier's
  container disk (300 GB) holds the largest arm (`h3-8step`, 147.86 GB) plus the 120 GB every volume tier has
  (`tests/gold_tier.rs` checks it).

The full gold run, its smoke run and the cost: ADR-0019 §16.7 (about 1.5-3.1 h, $3.70-$7.70, at most $9.96 at the 4 h
cap).

## Keeping it equal to the scripts

```sh
bash scripts/gpu/tests/fv-lab-goldens.sh           # regenerate crates/fastvideo-lab/fixtures/ from the scripts
bash scripts/gpu/tests/fv-lab-goldens.sh --check   # CI (control.yml)
bash scripts/gpu/tests/fv-lab-report-goldens.sh [--check]   # the reports' fixtures (fixtures/report/, ADR-0019 §13.2)
bash scripts/gpu/tests/fv-lab-experiment-goldens.sh [--check]   # the experiment tiers' commands (ADR-0019 §14.3)
```

The report goldens are the report scripts' output on committed run directories (`python3 -I`, `env -i`,
`LC_ALL=C`); `tests/report.rs` compares fv-lab with each byte for byte. The retired scripts run from `f9e57ea2` when
the history has it (a full clone); in a shallow clone their fixtures are kept as frozen. The deleted branches' summary
lines are frozen in `fixtures/report/summary-script.sh`, each block checked against `runpod-matrix.sh` at `5987e6a4`
when the history has it, and every line of the script-side runs is checked to be in that run's `live.log`.

For every tier of `lab/tiers.toml`, the fixtures are the script's dry run (`<tier>-dry-run.txt`), its weight gates
(`<tier>-gates.tsv`: the cells the dry run gated, with their `verify-weights.sh --list` needs) and `runpod-http.sh
plan`'s pod body (`runpod-http-<tier>.json`); for a tier with `extra_cells`, the script's dry run and gates under
their settings too (`<tier>-extra-dry-run.txt`, `<tier>-extra-gates.tsv`; `oracle`'s ten clip-metric targets, ADR-0019
§20.6). For every stack of `lab/upstream.toml`, `runpod-http.sh
plan-upstream`'s pod body with `oracle.sh`'s settings (`runpod-http-upstream-<stack>.json`, ADR-0019 §19.3;
`tests/oracle_driver.rs`).

**Frozen since the GPU gate.** Once a tier passed its GPU gate and its script branch is deleted (it is listed in
`scripts/gpu/fv-lab-retired.sh`), the scripts can no longer produce its fixtures. They stay as they were written
before the branch went and are fv-lab's own goldens from then on: the goldens script does not regenerate them, only
checks that they exist, that the tier is in `lab/tiers.toml` and that its branch is gone (and that every other script
tier still has one); `tests/dry_run.rs` checks the same. The Rust tests still compare fv-lab's dry run, gates and pod
body with them, so a change to a frozen tier's commands or pod is deliberate: edit the cell (or `pod-start.sh.in`)
and the fixture in the same PR, and say why in the PR. The twenty retired tiers are frozen (`b4a`'s since its GPU
gate, ADR-0019 §11.6; `det-short`'s and `determinism`'s since theirs, §15.10; `fuse`'s, `attn3`'s and `ltx`'s,
§17.6; `hd`'s, `serve-engine`'s, `sfstream`'s and `mmaudio`'s, §22.19), with `hd`'s sibling `hd5` (its dry run the
`hd` family's with `FV_PROMPTS=…/prompts-hd5.json`, `FV_LAB_RETIRED_SIBLINGS`) and the script sides' GPU-check
bodies of the last three (`runpod-http-{serve-engine,sfstream,mmaudio}-r2.json`), and so are the two
restored modes (`precision-nvfp4`, `ltxvae-gen`), whose fixtures come from the script at `5987e6a4` in that mode
(`FV_LAB_RESTORED` in `fv-lab-retired.sh`; `fv-lab-restored-goldens.sh --check` reproduces them); `gold` was never a
script family (only its pod body comes from `runpod-http.sh plan`). The `gold` tier is no script family: only its pod body is written there, and its dry run is fv-lab's own golden
(`fixtures/gold-dry-run.txt`, `tests/gold_tier.rs`). The experiment tiers (`FV_LAB_EXPERIMENTS`, D1e) are the same: only
their pod body is written there, their dry run is fv-lab's own golden (`fixtures/<tier>-dry-run.txt`) and
`fv-lab-experiment-goldens.sh` holds their commands to the experiment's pod script (`tests/experiment_tiers.rs`). Once, there is also its `frames_digest` of a sample. A new tier fails
`--check` until its fixtures are written; `tests/{dry_run,payload_golden,frames_digest,fv_lab_bin}.rs` hold fv-lab to
them. The script runs every script under `env -i`, so your `FV_*` / `RUNPOD_*` settings do not leak in.
