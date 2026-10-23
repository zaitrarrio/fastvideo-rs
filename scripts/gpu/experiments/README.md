# Port experiment drivers

These are the per-port experiment drivers: one-off GPU runs that each H3 port
or optimisation used to prove itself on a Runpod RTX PRO 6000, kept here until
the validation driver `fv-lab` (execution plan WP-D1,
`docs/plan/2026-10-23-execution-plan.md`) absorbs them as data cells. Each
experiment is a triplet. `<name>-validate.sh` runs on the laptop: it checks
stock and price, builds a presigned R2 weights plan (`../r2-weights.py`),
stages a bundle (the pod script, `../r2-fetch.py`, the prompts and the
branch's `fv-gpucheck`) to `fv-weights-staging`, creates the pod with a
wall-clock backstop, polls its `status.tsv` over the HTTPS proxy, fetches the
results and deletes the pod (`up | status | fetch | down`). `<name>-pod.sh`
runs on the pod: it fetches the trees from the plan, runs the cells
(generations, parity, LPIPS / PSNR, timings) and writes `status.tsv` beside
the artifacts. `<name>-summary.py` (and `<name>-plan.py` where the plan is
shaped per arm) turns a fetched results directory into the tables quoted in
the port's write-up. Shared tooling stays one level up in `scripts/gpu/`
(`lib.sh`, `volumes.sh`, `runpod-cuda.sh`, `r2-*.py`, `upstream/`); the
drivers reach it through `$GPU`. What each experiment measured and found is
in `docs/ports/*.md` (`h3-trim.md`, `vdn-h3.md`, `h3-turbo-loras.md`,
`fasth3-nvfp4.md`, `fasth3-sparse-fp4-attn.md`) and `docs/perf/*.md`
(`h3-glue-fusion.md`, `h3-denoise-graphs.md`, `h3-nvfp4-tier-options.md`).
