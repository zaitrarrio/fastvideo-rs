# H3 `auto` text encoder: residency from the memory plan

**Rule (2026-10-14, owner-approved).** `--text-encoder auto` (the serve
default) keeps the FP8 text encoder (22.7 GiB) resident when the free device
memory measured before any load covers

    fp8 encoder
    + max( resident weights + denoise reserve(planned request) + runtime 2 GiB,
           the plan's own peak,
           the DiT placement's need )

where *resident weights* is the plan's denoise stage
(`fastvideo_models::h3::memory::plan`) with the DiT's block linears priced at
the precision the load uses (`H3DitWeights`: bf16, MXFP8 blocks 2-46, W8A8,
NVFP4), the decoders with the VAE's MXFP8 copies, the refiner, and the
image-to-video tower's 2 GiB reserve when it may load. The *denoise reserve*
is the measured bound (640 KiB per packed row under bf16 activations + 4 GiB,
`memory::denoise_reserve_bytes`) of the planned request: the recipe's default
canvas at 5 s (serve), else 1344x768 for 5 s. The *DiT placement's need*
(`dit_placement_need`: the bf16-priced resident plan the `dit_offload = auto`
placement already uses) is a floor whenever that placement would keep the DiT
resident: the placement is measured after the encoder loads, so the encoder
can never turn a resident DiT into a streamed one. Otherwise the encoder
streams. The decision is logged once:

    h3 text encoder: auto -> resident-fp8: 94.3 GiB free before load >= need 91.7 GiB = fp8 encoder 22.7 + max(...)

Code: `crates/fastvideo-models/src/h3/memory.rs` (`auto_text_encoder`),
`crates/fastvideo-cudarc/src/h3/pipeline.rs` (`auto_text_encoder_plan`).

**Larger requests** (the 1080P tier, long clips) are not planned for: before
their denoise the post-encode check (`keep_auto_encoder`, now counting
pool-cached bytes as available) releases the encoder when the request's
reserve does not fit, and later prompts stream.

**Numbers.** Residency does not change the conditioning: on a card with
85 GB free before the load (`AUTO_FP8_NUMBERS_FREE_BYTES`, the 96 GB class)
`auto` computes with the FP8 rows whether the encoder is resident, streamed
from the start or released; below that it streams bf16 as before
(docs/perf/determinism.md).

## Decisions (unit tests, `auto_text_encoder_table`)

Free before load: RTX PRO 6000 94.35 GiB (measured), RTX 5090 31.3, an 80 GB
card 79.1, the PRO 6000 beside the 11 GB upsampler sidecar 84.1. Served
settings: `dit_offload = auto`, bf16 activations, VAE MXFP8 copies, the i2v
tower's reserve, 768p 5 s planned.

| Card | DiT | need GiB | `auto` | before (85 GB rule) |
|---|---|---:|---|---|
| 96 GB | MXFP8, VSA gate (h3-turbo) | 91.7 | **resident** | resident |
| 96 GB | MXFP8, no gate (h3-max) | 87.0 | **resident** | resident |
| 96 GB | NVFP4 | 74.0 | **resident** | resident |
| 96 GB | bf16 (gated / dense) | 102.4 / 98.8 | streamed (FP8 numbers) | resident, released before the first 768p denoise |
| RTX 5090 | MXFP8 / NVFP4 | 56.0 / 74.0 | streamed | streamed |
| 80 GB | MXFP8 / W8A8 / bf16 | 91.7 / 91.7 / 102.4 | streamed | streamed |
| 80 GB | NVFP4 | 74.0 | **resident** | streamed |
| 96 GB + sidecar | MXFP8 turbo / max | 91.7 / 87.0 | streamed (FP8 numbers); DiT resident | resident, and the DiT then **streamed** (60.6 GiB free after the encoder < its 64.6 GiB plan, measured 2026-10-13) |
| 96 GB + sidecar | NVFP4 | 74.0 | **resident** | resident |
| 96 GB, planned 1080p 5 s | MXFP8 | 110.1 | streamed | — |

On the 96 GB card with MXFP8 the binding term is the DiT placement's floor
(66.9 GiB + the tower + the encoder); the plan's own terms need 86.7 GiB.
After the load and a 768p 5 s encode about 32 GiB stay free, over the 27.6
GiB reserve, so the encoder stays; a 1080p 5 s (50.9 GiB) or 768p 15 s
request releases it (`larger_requests_release_the_auto_encoder_after_encoding`).

## GPU check

Pending.
