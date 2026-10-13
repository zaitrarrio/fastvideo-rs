# Wan lazy DiT under the streaming fetch (2026-10-13)

This run tests the fix proposed in docs/ops/r2-bench-2026-10-13-stream.md finding 2:
`FASTVIDEO_WAN_DIT_LAZY=1` makes the one-DiT Wan pipeline open its transformer with the lazy map
(`WeightMap::open`, crates/fastvideo-cudarc/src/wan/pipeline.rs ~L453). The lazy map's `LazyStore` views wait
per part of the streamed file. The eager map (`WeightMap::from_dir`) waits for the whole 10 GB file.

**Result.** With the flag, fv-serve was ready **3.8 s after the last byte**, against 81.7 s for eager in this run
(87.9 s in the stream doc). Container start → ready was **106.6 s**, against 212.6 s for eager in this run (196.2 s
in the stream doc). The output was **byte-identical** to eager, and the host peak RSS was 9.3 GiB lower.

## Setup

The setup is the same as the stream doc's, with one endpoint per arm, run one after the other:

- Staging fv-control, driven with `scripts/serve/fv-control.sh endpoint create <spec.json>`.
- Preset `wan` (variant `wan5b`), `weights_source: r2`, `allowed_cuda` `["13.0"]`.
- GPU types: H100 80GB HBM3 / H200 / RTX PRO 6000 Blackwell Server Edition, in any DC.
- `workers_min` 0, `workers_max` 1, `idle_timeout_s` 5, FlashBoot on, and a 40-45 min delete backstop.
- `image: {sha: 680196f}`, the stream doc's image (serve-image run 38368357378), with the same digest
  `sha256:6f3eac47…`. The doc's spec was read back from fv-control (`bench-wan-stream`, `50mt8ldz0z9n1t`).
- The only difference between the arms is the spec `env`:
  - `bench-wan-lazy` had `{"FASTVIDEO_WAN_DIT_LAZY": "1"}`.
  - `bench-wan-eager` had `{"FASTVIDEO_WAN_DIT_LAZY": "0"}`, the default for a one-DiT checkpoint, set explicitly.
- Each endpoint got one cold invoke from zero, with the stream doc's job: `fastwan22-ti2v-5b`, "A red fox trots
  through fresh snow at dawn, low sun, gentle camera push in", seed 1.

**Image check.** fv-serve comes from the image's layer `sha256:624e3589…`, the same layer as in the stream doc. It
is `fv-serve 0.1.8 (5dcc5a0)`, from tools-v0.1.8.

- The layer was pulled from ghcr and grepped. It contains `FASTVIDEO_WAN_DIT_LAZY`, `fv-streaming-fetch/1` and the
  `wan load: host peak RSS` line.
- The commit that added the flag (`ad94673`) is an ancestor of `5dcc5a0`.
- The boot logs confirm which path each arm took: `wan load: … (lazy dit)` and `(eager dit)`.

Both workers were sm_90 at $3.99/h (H100 80GB, like the stream doc's worker). Both had a cache miss and pulled the
image new to the host. This time Runpod placed both workers at once (no stuck "initializing" worker).

## Results

Times are UTC. "Start" is the container start, taken from the boot log's `+s` stamps. "Ready" is `FV-SERVE READY`.

| | **lazy** (`hp745cqzcimu6b`) | eager, this run (`vl1s0rggaai0hl`) | eager, stream doc (`49yidzr9y4w3qk`) |
|---|---|---|---|
| Image pull (new to host) | 68 s | 55 s | — |
| `stream_ready` | 14.8 s (+19.8 s) | 13.8 s (+18.4 s) | 13.6 s (+18.2 s) |
| fv-serve start | +24.5 s | +22.8 s | +22.8 s |
| Fetch | 58.78 GB in 95.3 s, **617 MB/s** (66 slow restarts, 0 retries / bad parts) | 58.78 GB in 124.5 s, **472 MB/s** (57 slow restarts, 1 bad part) | 58.78 GB in 101.7 s, 578 MB/s |
| Fetch end | +102.8 s | +130.9 s | +108.3 s |
| `load_overlap` | 80.4 s | 110.1 s | 88.5 s |
| Model resident (load start → resident) | **75.4 s** | 183.3 s | 166.8 s |
| fv-serve `ready_after_s` | **82.0** | 189.7 | 173.3 |
| **Fetch end → ready** | **3.8 s** | **81.7 s** | **87.9 s** |
| **Container start → ready** | **106.6 s** | **212.6 s** | **196.2 s** |
| Host peak RSS at `wan load` | **26.5 GiB** (lazy dit) | 35.8 GiB (eager dit) | (not recorded) |
| Job delay / exec | 196.1 s / 17.1 s (run 13.3 s: text 0.75, denoise 2.97, decode 9.40) | 301.3 s / 18.3 s (run 13.6 s: text 0.93, denoise 3.02, decode 9.43) | 942.5 s / 17.3 s (run 13.3 s) |
| Container start → job done | 124.5 s | 244.7 s | (216.5 s to the mp4's last byte) |
| Click → job done | **213.5 s** | 333.0 s | 961.7 s (placement; see the stream doc) |

In this run, the eager fetch was 29 s slower than the lazy one (472 against 617 MB/s, the same tree on another
host). That alone inflates its container start → ready. **Fetch end → ready** does not depend on the fetch rate:

- eager waits about 82-88 s after the last byte (two runs);
- lazy waits 3.8 s.

At the same fetch time, lazy saves about **80 s**, which is the whole DiT load. Against the stream doc's 196.2 s
(fetch 101.7 s), lazy's 106.6 s (fetch 95.3 s) is **−90 s (−46 %)**. About 6 s of that comes from the faster fetch.

### Output

- All three mp4s are the same bytes, sha256 `64b6c3d0e8d9446df76606b4b196e891ab222b15d233ea6105dd6a488d4b58d1`:
  - lazy (this run);
  - eager (this run);
  - eager from the stream doc (its 24 h URL was still valid).
- Each is 2,765,919 bytes, h264, 1248×704, 121 frames at 24 fps (ffprobe). Frame 60 has a mean luma of about 89,
  so it is not blank.
- Lazy parity is exact, so a PSNR check is moot. The lazy path uploads the same bf16 bits (`weights::fill_bf16`).

### Errors, memory and VRAM

- **No errors.** Neither boot report has an error. No panics, no `ERROR` lines and no retries.
- The only warnings were the usual ones on both arms: NVENC unavailable → CPU encoder, and `FV_URL_SIGNING_KEY`
  unset.
- The lazy arm's loader waited only on DiT part 0 (the safetensors header, at open) and then on VAE parts 1-2.
  UMT5 and the VAE are still eager. They load first, then the DiT is read part by part as the parts arrive.
- **Host memory:** the peak RSS was 26.5 GiB lazy against 35.8 GiB eager, **−9.3 GiB**. The eager map copies the
  10 GB DiT into host memory. The lazy map keeps it in the mapped file.
- **VRAM: not measured.** fv-serve reports `peak_memory_mb: null`, and the boot log has no VRAM line after the
  load. Both arms had `free_gb` 84.38 at load start. Their denoise times were the same (2.97 / 3.02 s for 3 steps)
  with no swap or offload line. A VRAM difference is not expected, since the same bf16 tensors end up resident, but
  that is **UNVERIFIED**.

## Findings

1. **The fix works.** On a one-DiT Wan checkpoint, `FASTVIDEO_WAN_DIT_LAZY=1` overlaps the DiT load with the
   streaming fetch, as H3 already does.
   - Fetch end → ready drops from about 85 s to 3.8 s, a tail even shorter than H3's 10.5 s.
   - The output is bit-identical.
   - The host peak RSS is 9.3 GiB lower.
2. **The Wan cold start on r2 is now the fetch plus about 11 s:** about 20 s to `stream_ready` and fv-serve, then
   the fetch, then a 4 s tail.
   - Container start → ready was 106.6 s. The EU volume baseline in the stream doc's table is 119 s. This is one
     sample, and the fetch rate varies by host (472-617 MB/s across the three runs).
3. **The next Wan lever is the tree order** (stream doc finding 3). The boot model's tree `fastwan22-ti2v-5b`
   (24.2 GB) still finished last (95 s), because it shared the streams with `wan22-ti2v-5b` (34.2 GB), which the
   boot does not load. Fetching the boot tree first could put ready at about 24.2 GB / 0.6 GB/s ≈ 40 s of fetch +
   20 s. That is an estimate, **UNVERIFIED**.

## Proposal (not made in this PR)

> **Adopted 2026-10-13** (owner: "make lazy wan the default for r2"): `WanPipeline::load_with_dit` now
> defaults `FASTVIDEO_WAN_DIT_LAZY` to `two_experts || streaming`. The volume path is unchanged.

**Make the lazy DiT the default for one-DiT Wan loads under the streaming fetch.** For example, in
`WanPipeline::load_with_dit`, change the default of `FASTVIDEO_WAN_DIT_LAZY` from `two_experts` to
`two_experts || streaming`, where `streaming` is true while `FASTVIDEO_STREAMING_FETCH=1` (the loader's
`fv-streaming-fetch/1` mode). `=0` would still force eager.

- **For:**
  - about −80 s cold start on every r2 Wan boot;
  - identical output;
  - −9.3 GiB host RSS;
  - the A14B already runs this path by default.
- **Limits of this evidence:**
  - one sample per arm, on H100 only;
  - one-DiT Wan 2.2 TI2V-5B (`fastwan22-ti2v-5b`) only;
  - lazy on the volume (non-streaming) path and on Wan 2.1 / LongLive (`load_pipeline_with_dit` with a given map)
    was not tested here.

  That is why the proposal gates on streaming instead of making lazy the default everywhere. A volume run would
  show whether lazy helps or hurts there: page-cache reads from a network volume, where eager copies once.

## Spend

- **Runpod:** balance **$19.67** before (09:52 UTC) and **$19.15** at 10:03, both endpoints deleted. That is about
  **$0.52**, including the account's baseline of 0.083/h, and billing may settle a little. The budget aim was
  $0.50, the hard stop $2 and the floor $12.
- **Left running:** nothing. `endpoint list` shows `bench-wan-lazy` (`5ndvui4y0jnp68`) and `bench-wan-eager`
  (`04ozf41pu4zmbk`) as `deleted`, with workers 0.
  - The eager endpoint's first delete returned Runpod's "Failed to terminate resources. Try again." while its
    worker was still exiting. fv-control's cron retried it and finished within about a minute.
  - `status` showed `running_pods` 0. No pods were created.
