# H3: fusing the MXFP8 quantize into the Sol row gather (phase 0: not worth it)

Lever evaluated on 2026-10-14: on Sol-Attn layers with a row permutation, write the permuted attention input once,
already MXFP8, instead of a bf16 gather followed by the QKV linear's own quantize pass. **Verdict: stop at phase 0.
The addressable time is 0% of a denoise step on h3-turbo, h3-max and the NVFP4 tier, because none of them runs a
permuted layer.** Nothing was implemented and no GPU time was spent.

## 1. Where the gather and the separate quantize are

`crates/fastvideo-cudarc/src/h3/transformer.rs`:

- `Attention::forward` gathers the input rows (`RowGather::apply`, phase `h3_attn_sol_gather`) only when
  `AttnMode::SolLayer`'s policy has a `permutation`. `project_out` runs the inverse gather after `to_out`.
- `Attention::accepts_mx` refuses the fused norm+modulate's MXFP8 output on those layers (`gathered`), so they get a
  bf16 activation and `qkvg.forward` quantizes it: that is the separate pass this lever would remove.
- `H3SolPolicy::from_technique` builds the permutation only from `sink_spec_for(SinkMode::Suffix)`, the Spark
  `[visual | text+audio]` plan (`fastvideo_models::h3::sol::sink_plan`).

`SinkMode::Suffix` (the Spark policy, `FASTVIDEO_H3_SOL_ATTN=spark` or `sink = "suffix"`) is rejected for every recipe
except `sol-h3-spark` (`recipe_sol_attn_policy`, `h3::techniques`), and `sol-h3-spark` is not in the CUDA catalog
(`crates/fastvideo-engine-service/src/cuda/caps.rs`).

## 2. Per tier

| Tier | Recipe / profile | Attention | Permuted layers | Separate MXFP8 quantize next to a gather | Addressable share |
|---|---|---|---|---|---:|
| `h3-turbo` | `fasth3-4step-vsa` / `h3/fasth3_4step_vsa` | VSA-H3 | none (no Sol route) | none | 0% |
| `h3-max` | `sol-h3` / `h3/sol_h3_4step_engine_ladder` | Sol engine route, `sink = "prefix"` | none (`plan: None`; the prefix sink is contiguous in packed order) | none: `accepts_mx` is true, the norm+modulate already writes the QKV operand as MXFP8 | 0% |
| NVFP4 (V2-NVFP4, 4-step NVFP4) | ModelOpt `transformer/` (docs/ports/fasth3-nvfp4.md) | VSA-H3 | none | none (see below) | 0% |

**NVFP4.** The existing CUPTI trace of one warm 480p V2 step
(`artifacts/ports/fasth3-nvfp4/run-2026-10-14/ours/v2-profile-gpu-trace-480p.json`, on the branch merged as the NVFP4
apples-to-apples port) has one row gather per block: `index_select_rows`, 50 calls, 2.05 ms of 1.69 s busy (0.12%).
That is VSA's `prefix_branch_rows` read of the compressed branch for the text/audio query rows (`h3/vsa.rs`),
followed by bf16 roundings, not by an FP quantize. The NVFP4 activation passes (`nvfp4_quant_bf16_sw` 5.4%,
`nvfp4_amax_bf16` 0.9%) feed the FFN / attention GEMMs from the norm, the SwiGLU and the head merge; no gather sits
before them. Fusing those into their producers is the glue-fusion work (branch `claude/h3-glue-fusion`), not this
lever. VSA's own tile gather (`vsa_tile_qkv`, 7.5%) feeds a bf16 kernel; it would only meet a quantize under the
opt-in FP4 VSA attention.

## 3. Why no GPU trace for h3-turbo and h3-max

A trace measures how long the gather and its quantize take. On these tiers the gather is not launched: the only code
path that builds a permutation is refused at technique resolution for both recipes. A trace would show a share of
exactly zero and could not change the verdict, so the budget was not spent on it.

## 4. Nearest real opportunity (not evaluated here)

On `h3-turbo` the QKV projection carries the bf16 `to_gate_compress` rows (`has_gate`), so `accepts_mx` is false
there too. The norm+modulate writes bf16, and `qkvg.forward` quantizes the activation for its three MXFP8 sections
in a separate pass on all 50 blocks. A norm+modulate that writes both the bf16 rows (for the gate) and the MXFP8
operand (for Q/K/V) would remove that pass on the served turbo tier. It needs its own phase-0 trace.
