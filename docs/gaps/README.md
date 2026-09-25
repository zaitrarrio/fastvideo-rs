# Gap analyses

Every gap analysis run against this tree, in date order. Each file is a
self-contained record of what was compared, what the numbers were, and what
was concluded on that day. They are not kept current; later entries supersede
earlier ones where they overlap, and `decision-log.md` records what was done
about each.

| Date | File | Compares | Headline |
|---|---|---|---|
| 2026-09-10 | [cudarc-vs-upstream-parity.md](2026-09-10-cudarc-vs-upstream-parity.md) | cudarc Wan/FastWan surface vs hao-ai-lab/FastVideo feature list | 100% in-scope feature parity; Flash / VSA / SP are in-tree approximations, not vendor binaries |
| 2026-09-17 | [upstream-head-to-head.md](2026-09-17-upstream-head-to-head.md) | Ours vs upstream FastVideo on one rented RTX 3090 Ti | Upstream 1.96x faster; the gap is VSA. Closed to 1.81x denoise speedup the next day |
| 2026-09-24 | [codebase-review-gpu-sol-engine-oxide.md](2026-09-24-codebase-review-gpu-sol-engine-oxide.md) | GPU utilisation, sol-engine spec alignment, cuda-oxide / pliron footprint | Sol-Attn / PISA / SLA ran on the CPU; f32 activation sandwich; oxide footprint zero |
| 2026-09-24 | [phase3-vs-published.md](2026-09-24-phase3-vs-published.md) | Phase 3 gate on RTX PRO 6000 vs published FastH3 / FastWan-QAD numbers | 10.8 s/step is not 16.2 s E2E; no B200 gen, no H3 decode reached |
| 2026-09-24 | [decoder-gap.md](2026-09-24-decoder-gap.md) | Official video VAEs vs tiny decoders vs published E2E | Official H3 decode (23-29 s) alone exceeds the published 16.2 s clip; TAEH3 closes it |
| 2026-09-25 | [sol-engine-docs-level.md](2026-09-25-sol-engine-docs-level.md) | fastvideo-rs vs the published NVlabs Sol-Engine docs (pipelines, techniques, workflow) | Contracts ported, 0/6 pipelines GPU-measured with the optimized line on, 0/3 quality gates |
| 2026-09-25 | [sol-engine-code-level.md](2026-09-25-sol-engine-code-level.md) | fastvideo-rs source vs NVlabs/Sana `sol-engine` source (HEAD 6c2f582) | Controllers match line for line; six verified defects in clocks, CFG batching, sink layout; Sol kernel 10x structural |

Related reports that are measurements rather than gap analyses live in the
decision log and `docs/MILESTONES.md`: the H3 encoder matrix report
(2026-09-20), model support by platform (2026-09-20), the H200 and B200 warm
suites (2026-09-24 / 25).
