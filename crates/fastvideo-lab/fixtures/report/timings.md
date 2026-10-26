# Test timings — timings-run

Run **pass** in 42 min, ~$0.364 at $0.52/hr (instance 31415926).

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (97887 MiB, sm 12.0, driver CUDA 13.0), 32 cores, 188 GB RAM.

## Stages

| stage | status | wall s |
|---|---|---:|
| clip-ltx-cold | pass | 5 |
| clip-wan-t2v | pass | 99.9 |
| engine | pass | 135.3 |
| model-wan | pass | 123.5 |
| parity-wan | fail | 300 |
| probe-empty | fail | 0.4 |
| probe-wan14 | pass | 2.5 |
| sol | error | 145.8 |
| wan-wan-cpu-ref | cached (earlier run) | 10 |
| wan | pass | 102.9 |

## Generated outputs

Time is end-to-end generation on the stage's device (sampler videos: denoise + VAE decode). GPU rows compare against the CPU-path reference.

| stage | output | seconds | rel_l2 | PSNR dB | mp4 |
|---|---|---:|---:|---:|---|
| model-wan | t2v | 12.346 | 0.0123 | 38.123 | model-wan/t2v.mp4 |
| model-wan | i2v | 7.5 | — | — | encode failed |
| model-wan | latents | 1.25 | 0.5 | 21 | — |
| parity-wan | step0 | 0.004 | 0.00123 | — | — |

## Clips

| clip | frames | resolution | steps | warm | load s | denoise s | per step s | decode s | write s | generate s | s per video-second | peak MiB | mp4 |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clip-ltx-cold | — | 768x? | — | cold | — | — | — | — | — | 61.5 | — | — | — |
| clip-wan-t2v | 81 | 832x480 | 50 | yes | 12.345 | 45.6789 | 0.91357 | 3.21 | 0.5 | 50.1234 | 10.024 | 40960 | clip-wan-t2v/out.mp4 |

## Projections

- probe-empty: fail; projected total ? s (DiT forward ? s at ? tokens), peak ? MiB
- probe-wan14: pass; projected total 1234.56 s (DiT forward 12.3456 s at 75600 tokens), peak 81234.5 MiB
