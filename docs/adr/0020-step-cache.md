# ADR-0020: One `StepCache` trait, one implementation per algorithm

- Status: Accepted (plan B4a, `docs/plan/2026-10-23-execution-plan.md`)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-step-cache
- Source: `docs/gaps/2026-10-22-architecture-review.md` §3 ("step caches
  reimplemented per family (EasyCache x3, TeaCache x3, FbCache, Taylor)"),
  §5.2 ("one `StepCache` trait with `EasyCache`/`TeaCache`/`FbCache`/`Taylor`
  implementations in `techniques/`"), §7.3 item 4
- Code: `crates/fastvideo-models/src/techniques/step_cache/`
  (`mod.rs`: trait and shared types; `easycache.rs`, `teacache.rs`,
  `fbcache.rs`, `taylor.rs`); the family parameter sets in
  `wan/sol_cache.rs`, `h3/sol.rs`, `cosmos/sol.rs`, `ltx2/fbcache.rs`,
  `lingbot/sol.rs` (fastvideo-models)
- Tests: `step_cache/equality.rs` (old against new, bit for bit, on
  synthetic tensors) with the pre-B4a controllers verbatim in
  `step_cache/legacy.rs`; a decision-sequence test in each algorithm's
  module; GPU: `scripts/gpu/runpod-matrix.sh b4a`, since ADR-0019 §11.6 the
  fv-lab tier `b4a` (`lab/cells/b4a-h3-*.toml`; the script branch retired)

Line references to "pre-B4a" code are to main `2e3c71b1`, where the
copies lived; `legacy.rs` holds them verbatim under the same names, except
TeaCache4Wan2.1, which lived inside a cudarc runtime with its tensors and
is there as a hand reduction to its decision (every arithmetic line as
written).

## Context

Nine step-cache controllers lived in five families' modules, each with its
own state machine, its own decision struct and its own arithmetic:

| copy (pre-B4a) | algorithm | family runtimes |
|---|---|---|
| `wan/sol_cache.rs:157-321` `EasyCache` | EasyCache | Wan 5B / 14B (`wan/pipeline.rs:1802-1895`), SANA-Video (`sana_video/pipeline.rs:218-280`) |
| `wan/sol_cache.rs:715-850` `A14bCacheController` | EasyCache | Wan 2.2 A14B (`wan/transformer.rs:1827-1906`) |
| `lingbot/sol.rs:149-208` `EasyCache` | EasyCache | LingBot (`lingbot/pipeline.rs:534-570`) |
| `wan/sol_cache.rs:472-596` `SolTeaCache` | TeaCache | Wan Sol (`wan/transformer.rs:1223-1303, :1634-1681`) |
| `h3/sol.rs:421-558` `H3TeaCache` | TeaCache | H3 (`h3/transformer.rs:3258-3331`, installed in the load's `fuse` stage, ADR-0017) |
| `cosmos/sol.rs:66-137` `TeaCacheWindow`, `SolCosmosTea` | TeaCache | Cosmos (`cosmos/transformer.rs:539-593`), Cosmos3 (`cosmos3/pipeline.rs:242-276`) |
| `wan/pipeline.rs:1571-1671` (cudarc) `TeaCache` | TeaCache | Wan `FASTVIDEO_TEACACHE` (TeaCache4Wan2.1) |
| `ltx2/fbcache.rs:84-190` `FbCache` | FbCache | LTX-2.5 stage 1 (`ltx2/transformer.rs:1439-1512`) |
| `wan/sol_cache.rs:30-146` `TaylorSchedule`, `taylor_update`, `taylor_predict` | Taylor | Wan (`wan/transformer.rs:1305-1338, :1908-1960`) |

They are one idea each, but the copies are not arithmetically the same:
where one clears its accumulator, whether it compares `>=` or `<`, whether
it accumulates in f32, how it rescales, which steps it forces. The plan's
first rule is byte identity (§1): unifying them may not move a single
output bit, so every difference has to survive as a named parameter.

## Decision

### Where it lives: `fastvideo_models::techniques::step_cache`

Not in fastvideo-cudarc (`rt/` or a `step_cache` module there), because:

- the controllers hold only scalars (flags, accumulators, counters,
  EasyCache's `k`); every copy already lived in fastvideo-models, which
  compiles and tests without the `cuda` feature or a device;
- the techniques layer owns the `step_output` seam the caches write
  (`Seam::StepOutput`, the `teacache` technique, `H3Techniques::teacache_state`)
  and the settings that configure them (ADR-0010); the review (§5.2) puts
  the trait in `techniques/`;
- cudarc depends on fastvideo-models, not the reverse, so the family
  runtimes there call it directly.

The tensors a cache keeps between forwards (the previous signal, the
residuals, Taylor's factors) and the device reductions stay in each
family's cudarc runtime: their placement, booking (`rt::ledger::STEP_CACHE`)
and freeing (`end_denoise`, `clear_step_caches`) are family business, and
the reductions are kernels (B4a changes no kernel).

### The interface

```rust
pub trait StepCache {
    fn name(&self) -> &'static str;                // log / error prefix
    fn algorithm(&self) -> Algorithm;
    fn needs_signal(&self, lane: Lane, step: usize) -> bool;
    fn decide(&mut self, lane: Lane, step: usize, signal: Signal)
        -> Result<Decision, StepCacheError>;
    fn note_computed(&mut self, lane: Lane, refresh: Refresh)
        -> Result<(), StepCacheError>;
    fn note_reused(&mut self, lane: Lane);
    fn reset(&mut self);
}
```

- **The per-step decision** is `Decision { action, reason, step, distance,
  indicator, accumulator, k }` with `action` one of `Compute`, `Reuse`
  (add the stored residual) or `Extrapolate` (Taylor's forecast). `reason`
  is the family's string (`"warmup"`, `"below_threshold"`,
  `"online_error_seed"`, ...), so every log line that printed it prints the
  same text.
- **Lanes** are independent states: a CFG branch (Wan Sol: `Lane::COND`,
  `Lane::UNCOND`), a CFG pass ordinal (LTX: `Lane(pass)`), or `Lane::ONLY`.
  Created on first use. A14B keeps one controller per expert, not one lane
  per expert: each expert's transformer owns its runtime and controller,
  and both call it with `A14bExpert::HighNoise.lane()`, as the pre-B4a
  runtime passed `A14bExpert::HighNoise` (`wan/transformer.rs:1838-1898`).
  The controller's per-expert lanes (`A14bExpert::lane()`) are exercised
  only by the CPU fixture.
- **The protocol**, per forward: `needs_signal` (measure only when true;
  a forced step's signal is never read, as before), `decide`, then
  `note_computed` after a computed forward or `note_reused` after a reused
  one; `reset` when the denoise's buffers are freed (H3 `end_denoise`).
- **Errors** are typed (`StepCacheError`, `thiserror`): bad parameters with
  the old messages (`"h3 teacache requires a positive threshold"`), a needed
  signal of the wrong kind (`WrongSignal`, where the old runtimes passed `0.0`
  or `expect`ed), a missing refresh (`WrongRefresh`), Taylor's zero delta
  and forecast-before-update.

### The state each keeps

| type | per lane | global |
|---|---|---|
| `TeaCache` | has signal, has residual, accumulator, consecutive hits, reuses since the last compute | the parameters |
| `FbCache` | (a `TeaCache` rule) | the step index and the pass ordinal of the step |
| `EasyCache` | accumulator, `k`, has previous input, has payload (residual), stored output norm, consecutive reuses, last decision | the parameters |
| `Taylor` | — | the forward counter (from −1), the forward of the last factor update |

### How a family feeds the signal

| family | signal | reduction (pre-B4a, unchanged) | `Signal` |
|---|---|---|---|
| H3 TeaCache | block 0's AdaLN-modulated RMS-norm of the hidden (`h3/transformer.rs:3269-3274`) | `Σ|a−b| / max(Σ|b|, 1e-8)`, sums in f64 on the device in block order (`rt/ops.rs:778`, `h3/transformer.rs:431-451`) | `Distance` |
| Wan Sol TeaCache | the timestep projection (`wan/transformer.rs:2277`), per CFG row | `(Σ|d|/n) / max(Σ|b|/n, 1e-8)` (`wan/transformer.rs:1202-1206`) | `Distance` |
| Cosmos / Cosmos3 TeaCache | the DiT time embed (`cosmos/transformer.rs:698`, `cosmos3/pipeline.rs:246`), on the host | `(Σ|d|/n) / max(Σ|b|/n, 1e-8)` (`cosmos/sol.rs:54-63`); `0.0` before step 10 or without a previous embed | `Distance` |
| TeaCache4Wan2.1 | the latents (`wan/pipeline.rs:1642-1645`) | `Σ|d| / (Σ|b| + 1e-6)` | `Distance` |
| LTX FbCache | block 0's residual `hidden − input` (`ltx2/transformer.rs:1452-1464`) | device `Σ|d| / Σ|b|` (`inf` at 0, `ltx2/transformer.rs:397-408`, `rt/ops.rs:4979`); host `(Σ|d|/n) / (Σ|b|/n)` | `Distance` |
| Wan / SANA EasyCache | mean `|x − x_prev|` and mean `|out_last|` (`wan/pipeline.rs:1816-1823`, `sana_video/pipeline.rs:243-251`) | `wan/sol_cache.rs:128-137` (cudarc) | `Drift` |
| A14B EasyCache | mean `|prefix − prefix_prev|` of block 0's output (`wan/transformer.rs:1836-1843`) | as above | `Distance` |
| LingBot EasyCache | the latent against the last computed step's latent (`lingbot/pipeline.rs:539-541`) | `(Σ f64(|a−b|_f32)/n) / (Σ|b|/n + 1e-8)` (`lingbot/sol.rs:172-182`) | `Distance` |
| Taylor | none; the factor history `proj_out` (`wan/transformer.rs:2371`) | — | `None` |

The five relative-L1 formulas are kept as they are (they differ in sum vs
mean, the clamp and where the `f32` difference is taken). The host ones
are named in `step_cache::distance`; the family wrappers
(`h3::sol::relative_l1`, `cosmos::sol::relative_l1`,
`ltx2::fbcache::relative_l1`, `lingbot::sol::EasyCache::relative_l1`)
compute exactly what they did.

`Refresh` carries what a computed step hands back: EasyCache's `k`
refresh (`full_input_change`, `output_change`, and for A14B the new
output's mean `|·|`); `Refresh::None` for the others.

### The arithmetic each implementation preserves

#### TeaCache (`teacache.rs`)

Per lane: a forced step computes; otherwise `indicator = rescale(distance)`,
`acc = acc + indicator`, and the comparison decides. Forced reasons, in
order: `warmup` (`step < warmup`), `cooldown` (the window's tail),
`initialize` (`require_signal` and no signal or no residual yet),
`periodic` (`since >= periodic`), the hit cap (`hits >= limit`, its own
reason string). The order is `wan/sol_cache.rs:564-579`.

| parameter | H3 | Wan Sol | Cosmos / Cosmos3 | TeaCache4Wan2.1 | LTX FbCache |
|---|---|---|---|---|---|
| window | `[retain, num_forwards − cooldown)` (`h3/sol.rs:522-526`) | `[start, end)` (`wan/sol_cache.rs:566-569`) | `step >= 10`, no tail (`cosmos/sol.rs:89`) | the first signalled decision is warm-up (`wan/pipeline.rs:1647-1651`) | `step >= warmup`, no tail (`ltx2/fbcache.rs:132`) |
| `Rescale` | `Horner` from 0 (`h3/sol.rs:552-558`) | `Horner` from 0 (`wan/sol_cache.rs:590-596`) | `Identity` (`cosmos/sol.rs:90`) | `HornerLeading` (`wan/pipeline.rs:1627-1630`) | `Identity` (`ltx2/fbcache.rs:139`) |
| `Comparison` | `acc >= thr` computes (`h3/sol.rs:484`) | `acc >= thr` (`wan/sol_cache.rs:527`) | `acc >= thr` (`cosmos/sol.rs:91`) | `acc < thr` reuses (`wan/pipeline.rs:1652`) | `acc < thr` reuses (`ltx2/fbcache.rs:140`) |
| `Accumulate` | f64 | f64 | f64 | **f32**: `acc_f32 += poly(rel) as f32`, `acc_f32 < thr_f32` (`wan/pipeline.rs:1575, :1646`) | f64 |
| `ResetOn` | `Decision`: every compute decision clears (warm-up / cool-down `h3/sol.rs:477-479`, threshold `:485-487`), so a threshold compute reports `acc 0` | `Computed`: only `note_computed` clears (`wan/sol_cache.rs:550-556`); the decision reports the crossing value | `Decision` (`cosmos/sol.rs:93-96`) | `Threshold`: the warm-up decision accumulates too and is not cleared; only a threshold compute clears; `store` does not (`wan/pipeline.rs:1646-1658`) | `Computed` (`ltx2/fbcache.rs:155-160`) |
| `require_signal` | yes | yes | no (the caller passes `0.0`) | no (the runtime decides only with a previous step) | yes |
| hit cap | — | `max_hits`, 0 = off (`Cap::nonzero`) | 3, **no zero guard** (`hits >= max_consecutive`) | — | 10, no zero guard |
| periodic | — | `periodic`, 0 = off | — | — | — |
| lanes | `ONLY` | `COND` / `UNCOND`; a CFG batch decides both rows and reuses only when both reuse (`wan/transformer.rs:1262-1303`) | `ONLY`; uncond follows | `ONLY` | one per CFG pass (`FbCache`) |

`Comparison::ComputeAtOrAbove` and `ReuseBelow` differ only on NaN (`acc >=
thr` reuses a NaN, `!(acc < thr)` computes it); `Horner` and
`HornerLeading` differ only where `0 · x` is NaN (`x` infinite, which LTX's
`inf` distance could produce, but LTX does not rescale). Both pairs stay.

#### FbCache (`fbcache.rs`)

`FbCache` is the TeaCache rule above with the LTX column's parameters
(`Identity`, `ReuseBelow`, `Computed`, cap `max_consecutive` reason
`"max_consecutive"`), plus the pass ordinal: `begin_step(i)` resets it,
`decide_next` decides `Lane(pass)` and advances (`ltx2/fbcache.rs:105-153`).

#### EasyCache (`easycache.rs`)

Forced reasons, in order: warm-up, cool-down (`Last { steps, of }`), seed
(no residual yet), the reuse cap, online seed (no previous input or `k`,
and for A14B no stored norm) (`wan/sol_cache.rs:825-841`). Otherwise
`estimate` per `Estimator`, accumulated, compared.

| parameter | Wan / SANA (`wan/sol_cache.rs:258-321`) | A14B (`wan/sol_cache.rs:775-841`) | LingBot (`lingbot/sol.rs:185-207`) |
|---|---|---|---|
| `Estimator` | `KScaled`: `k · input_change / max(output_norm, 1e-8)`, the norm measured at the decision (`:273-274`) | `KScaledStoredNorm`: `k · input_change / max(norm_last, 1e-8)`, the norm stored by `note_computed` (`:781-782`) | `Raw`: the distance itself, **not accumulated** (`:194`) |
| `k` refresh | `k = Δout / max(Δin, 1e-8)` once a payload exists (`:301-304`) | the same (`:811-813`) | — |
| `Comparison` | `acc >= thr` computes (`:276`) | `acc < thr` reuses (`:786`) | `rel < thr` reuses (`:194`) |
| accumulator clears | warm-up, cool-down, threshold (`:265, :268, :277`) | every guard and the refresh (`:777, :789`), and `note_computed` (`:817`) | — |
| reuse cap | — | `max_reuse`, 0 = off (`:833`) | `max_reuse`, **no zero guard** (`run >= max_reuse`, `:191`) |
| reasons | `EasyLabels::WAN` | `EasyLabels::A14B` (`warmup_guard`, `tail_guard`, `payload_seed`, `max_reuse_guard`, `online_error_seed`, `online_error_refresh`, `online_error_below_threshold`) | not logged |
| uncond | follows the cond lane's last decision unless it has no residual (`uncond_compute`, `:314-316`) | the runtime reuses the cond decision per branch (`wan/transformer.rs:1849-1865`) | the whole CFG velocity is reused |

Two places where the one implementation is written uniformly and the old
copies were not, both shown to be the same numbers:

- Wan's `initialize` did not clear the accumulator (`wan/sol_cache.rs:270-271`),
  the unified seed / online-seed guards do. `initialize` happens only while
  `has_previous_step && has_residual && k` is not yet true; those flags
  never go back, and accumulation needs all three, so the accumulator is
  `0.0` at every `initialize` and clearing it changes nothing.
- The cosmos window counted a reuse inside `decide` (`cosmos/sol.rs:98`);
  `SolCosmosTea::decide` now calls `note_reused` right after a reuse, the
  same increment at the same point.

#### Taylor (`taylor.rs`)

`decide` advances the forward counter (starting at −1) and computes on
`step < warmup` (`"warmup"`), `step >= cooldown_start` (`"cooldown"`) or
`(step − warmup − 1).rem_euclid(interval) == 0` (`"interval_refresh"`),
else `Extrapolate` (`"taylor_forecast"`) (`wan/sol_cache.rs:70-89`).
`note_computed` records the update forward; `update_delta` is `current −
last` (1 for the first, an error for 0) and `forecast_offset` `current −
last`. The coefficients both factor paths use: the divided-difference
scale `1.0 / delta as f32` (`wan/sol_cache.rs:107`, `wan/transformer.rs:1315`)
and the forecast coefficients `(pow / fact) as f32` with `pow *= offset as
f64`, `fact *= order as f64` (`wan/sol_cache.rs:130-140`,
`wan/transformer.rs:1330-1335`). The host forecast accumulates from zero
including order 0 (`host_predict`); the device forecast starts from
`base · 1.0` and adds the higher orders (`predict_factors`). Both orders
of summation are kept as they were.

### How each family's settings map onto it

| family | settings (names and defaults unchanged, ADR-0010) | constructor |
|---|---|---|
| Wan EasyCache | `FASTVIDEO_WAN_SOL_CACHE=easycache`; `FASTVIDEO_WAN_EASYCACHE_PROFILE` (code-default 0.05/7/1, fullstack 0.036/7/1, 14b-tuned 0.10/5/1), `_THRESH`, `_RETAIN`, `_COOLDOWN` (with the `WAN22_*` fallbacks) | `wan::sol_cache::easycache(steps, threshold, retain, cooldown)`, `easycache_from_profile` |
| Wan A14B | the same flag on an MoE model; the published 0.30 / 5 / 3 / 1 | `a14b_easycache_official(steps)` |
| SANA-Video | `FASTVIDEO_SANA_EASYCACHE` / `FASTVIDEO_SANA_OPT`; retain 7, cooldown 1 | `easycache(...)` |
| LingBot | `FASTVIDEO_LINGBOT_SOL` (`cache` / `easycache`); `EasyCacheConfig::BASE` / `REFINER` | `lingbot::sol::EasyCache::new(cfg, steps)` |
| Wan Sol TeaCache | `FASTVIDEO_WAN_SOL_CACHE=teacache`; `FASTVIDEO_WAN_TEACACHE_THRESH` 0.12, `_START` 2, `_END` steps−2, `_MAX_HITS` 0, `_PERIODIC` 0, `_COEFFS` `1.0,0.0` | `wan::sol_cache::sol_teacache(...)` |
| TeaCache4Wan2.1 | `FASTVIDEO_TEACACHE`, `FASTVIDEO_TEACACHE_THRESH` 0.08 (parsed as f32), `FASTVIDEO_TEACACHE_RET` | `wan::sol_cache::TeaCache4Wan21::new(threshold, ret)` |
| H3 | `FASTVIDEO_H3_SOL_CACHE=teacache` and the `teacache` technique (`threshold`, `retain_steps`, `cooldown_steps`, `num_forwards`, `coefficients`; profiles `h3/rtx5090_fullopt*`, `h3/fasth3_8step*_teacache`) | `h3::sol::teacache(...)`, `rtx_teacache(forwards)`, via `H3Techniques::teacache_state` in the `fuse` stage (ADR-0017) |
| Cosmos / Cosmos3 | `FASTVIDEO_COSMOS_SOL=teacache`, Cosmos3 `--arm teacache`; 1.15 / 10 / 3 | `cosmos::sol::SolCosmosTea::official()`, `teacache_window(...)` |
| LTX-2.5 | `FASTVIDEO_LTX2_FBCACHE`; 0.08 / warmup 1 / 10; only GB200 dev guided stage 1 (`fbcache::scope`) | `ltx2::fbcache::official()` |
| Wan Taylor | `FASTVIDEO_WAN_SOL_CACHE=taylorseer`; `FASTVIDEO_WAN_TAYLOR_INTERVAL` 3, `_WARMUP` 3, `_COOLDOWN` steps−2, `_ORDER` 1, `_LITE` (with the `WAN22_TAYLOR_*` fallbacks) | `wan::sol_cache::taylor(...)` |

No setting, default or profile changed. The evidence lines are the same
text: `h3 sol teacache: ...` / `h3 teacache: threshold ...` at install,
`h3 teacache step N: compute|REUSE (reason; rel_l1 .. indicator .. acc ..)`
per decision and the `benchmark.json` `teacache` block, the load plan's
`fuse step cache: teacache threshold=.. retain=.. cooldown=..` (a golden in
`tests/goldens/h3_load_plan.txt`), `wan sol teacache: ...`, `wan sol
easycache: ...`, `wan a14b easycache: ...`, `wan sol taylorseer: ...`,
`ltx2 sol: GB200 FBCache ...`, the debug reuse lines, and the reuse counts
(SANA `reused_steps`, Cosmos3 `steps_reused`, LingBot `reused`).

### Evidence

- **CPU, every algorithm and family** (`step_cache/equality.rs`): the
  pre-B4a controller (`legacy.rs`) and the new one are driven through the
  family's runtime protocol on the same synthetic tensors (a stand-in
  transformer whose outputs feed the next step's latent, so one diverging
  decision diverges everything after it); every decision (compute or
  reuse, reason, accumulator and indicator bits) and the final latent's
  bits must be equal. Wan / SANA EasyCache (five profiles), A14B (four,
  both experts), LingBot (four configs, two schedule lengths), Wan Sol
  TeaCache (one-branch and CFG-batch forwards, hit cap, periodic, four
  polynomials), H3 (five parameter sets, two requests with `reset`),
  Cosmos and the bare window (zero cap included), TeaCache4Wan2.1 (both
  coefficient sets, five thresholds including 0), LTX FbCache (three
  passes, an `inf` distance, zero cap), Taylor (four schedules, orders
  1-3). Each suite also asserts that reuse actually happens. Two tests
  pin the comparisons themselves: one exact tie (`acc == threshold` on
  dyadic values) per comparison family, old against new (H3, Wan Sol,
  Cosmos, Wan EasyCache with `>=`; LTX FBCache, A14B, LingBot with
  `!(<)`), and TeaCache4Wan2.1 on a signal where its f32 running sum lies
  above the f64 one with the threshold exactly on the f32 sum (the f32
  rule computes on the tie, an f64 accumulator would reuse); its fixture
  also compares the f32 accumulator's bits per decision. Mutations caught:
  H3's `ResetOn`, A14B's `Estimator`, f64 instead of f32 accumulation,
  `>=` → `>` (f64 and f32), and the `!(acc < thr)` tie flipped (f64 and
  f32).
- **GPU** (`scripts/gpu/runpod-matrix.sh b4a`): the baseline binary and the
  PR build with the same argv and env, `compare-clips --off-identity`, each
  arm's `frames.sha256.json`; a cell also fails when either arm reused no
  forward or the arms' step-cache evidence lines (install line, every
  `h3 teacache step` decision, the load plan's `fuse step cache`) differ.
  Since the B4c review (PR #177) the verdict also needs both arms' exit 0
  and the `--off-identity` compare passed and not skipped
  (`b4-verdict.sh`, frozen in `crates/fastvideo-lab/fixtures/b4a-script/b4-verdict.sh`
  since ADR-0019 §11.6 / §18), and `b4a.verdict.json` fails when a cell
  named in `FV_B4A_CELLS` has no passing verdict. The family is an fv-lab
  tier since ADR-0019 §11 (its GPU gate passed on `e82b6b7`, §11.6, and
  the script branch is deleted): `fv-lab pod up --family b4a --cells
  <cell>-base <cell>-pr --baseline-sha <7>` runs it, with the same files.
  Three H3 TeaCache cells: the legacy flag, the profile, and a 4-step
  reuse fixture. **Not on the GPU:**
  - LTX-2.5 FBCache has no reachable path today: `fbcache::scope` admits
    only LTX-2.5 dev *guided* stage 1, but the only guided LTX-2.5 route,
    A2V, runs with `use_cfg = false` (`ltx2/pipeline.rs:3452`) so
    `arm_requested(_, false)` is refused, every other guided LTX-2.5
    request fails `check_generate_contract`, and `denoise_guided_cond`
    never calls `begin_fbcache_step`. `scope` is not widened here (that
    would change outputs); FBCache's evidence is the CPU fixture.
  - Wan (EasyCache, A14B, Sol TeaCache, TeaCache4Wan2.1, Taylor), SANA,
    LingBot, Cosmos and Cosmos3 have no tree on the EU volume (`volumes.sh`
    `FV_EU_VOLUME_TREES`); their evidence is the CPU equality fixtures.

## Consequences

- A new cache, or a family adopting one, is a parameter set and a runtime
  that measures the signal; the decision rule is not written again. A new
  arithmetic variant must be a new named parameter with an equality
  fixture, not an edit of an existing one.
- `H3TeaCache`, `SolTeaCache`, `TeaBranch`, `A14bCacheController`,
  `TeaCacheWindow`, `TaylorSchedule`, `taylor_update` / `taylor_predict`,
  `FbDecision` and the per-family decision structs are gone; their callers
  (cudarc runtimes, tests) use the trait. `SolCosmosTea::decide` and
  LingBot's `reuse` / `computed` return `Result`.
- Where a runtime `expect`ed the previous signal a decision needs (H3, Wan
  Sol, Wan / SANA EasyCache, A14B, LTX), it now hands `Signal::None` and
  the cache returns a typed `WrongSignal` error naming the step. The
  `expect`s on stored residuals and pending tensors in the reuse paths are
  unchanged.
- `legacy.rs` is test-only and frozen: it is the reference the fixtures
  hold the implementation to, not code to maintain.
- Fusion (B4b) and the disk prompt cache (B4c) are separate packages.
