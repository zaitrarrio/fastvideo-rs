# Port experiment drivers (D1e: one left)

These were the per-port experiment drivers: one-off GPU runs that each H3 port
or optimisation used to prove itself on a Runpod RTX PRO 6000. D1e
(`docs/adr/0019-fv-lab.md` §14) decided each one: nine are deleted (finished
experiments whose results are in `docs/ports/*.md` and `docs/perf/*.md`, the
evidence in `artifacts/`; the scripts are in git at `22a5fd7b`), and one,
`h3-graphs`, became the fv-lab tier `h3-graphs` (`lab/tiers.toml`,
`lab/cells/h3-graphs-turbo{,-trace}.toml`): its h3-turbo cells, the open item
of `docs/perf/h3-denoise-graphs.md` §5, with the commands cut from
`h3-graphs-pod.sh` itself and held word for word by
`scripts/gpu/tests/fv-lab-experiment-goldens.sh` and
`crates/fastvideo-lab/tests/experiment_tiers.rs`.

`h3-graphs-validate.sh` (on the laptop: stock and price, the R2 plan, the pod
with a wall-clock backstop, `up | status | fetch | down | print-start`) and
`h3-graphs-pod.sh` (on the pod) stay until the tier's GPU check (ADR-0019
§14.4) has passed; then both go, and the goldens script reads the pod script
from the history. That same commit drops `h3-graphs-validate.sh` from `LOCAL`
in `scripts/gpu/tests/self-delete.test.sh` and lowers its launcher floor from
3 to 2, or that test fails. Its tables come from `fv-lab report summary --kind h3-graphs
DIR` (D1c). Shared tooling stays one level up in `scripts/gpu/` (`volumes.sh`,
`r2-*.py`, `lib/self-delete.sh`); the drivers reach it through `$GPU`.
