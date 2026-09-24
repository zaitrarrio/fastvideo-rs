# Cosmos3-Super — port specification

FastVideo family `cosmos3`: 64B text-to-video Super. Distinct from Predict2
Video2World (`docs/ports/cosmos.md`), which is EDM I2V. Super is the
sol-engine FlowMatch profile (`models/cosmos3.toml` /
`models/cosmos3/optimized/env.sh` on NVlabs/Sana `sol-engine`).

Upstream serves Super with **4-GPU sequence parallel**. This port is
**single-GPU only**. Multi-GPU SP is out of scope.

Sources in this tree (read 2026-09-24): `docs/scope.md` Cosmos3-Super row;
`crates/fastvideo-models/src/cosmos/sol.rs`; `FASTVIDEO_COSMOS3_OFFICIAL`;
`FASTVIDEO_NVFP4` / `FASTVIDEO_NVFP4_COSMOS_STEPS` (WS-H). Super DiT
`config.json` is **not vendored**.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `cosmos3_super_64b_t2v` | **TODO(upstream)** | 64B T2V Super. Not registered until a Hub id is published in-tree |

No CLI / gpucheck hook yet: Predict2 registry rows stay I2V EDM and must
not swallow this T2V FlowMatch SKU.

---

## Canvas / sampling (published)

`FASTVIDEO_COSMOS3_OFFICIAL=1` (or `official`). From `models/cosmos3.toml`
via `cosmos::sol`:

| piece | value |
|---|---|
| resolution | **1280 × 720** |
| frames | **189** |
| steps | **35** |
| guidance | **6** |
| fps | **24** |
| flow-shift | **10** — applied on this FlowMatch path. Predict2 EDM records it and does not apply it |

---

## DiT — Super 64B

**TODO(upstream).** No `num_attention_heads`, `attention_head_dim`,
`num_layers`, `in_channels` / `out_channels`, `patch_size`, text width, or
VAE family is in this tree or a vendored Hub `config.json`.
`Cosmos3TransformerConfig::super_64b()` returns `None`. `tiny()` is a
unit-test graph only.

Block sketch (Predict2-shaped, unconfirmed for Super): AdaLN → self-attn →
cross-attn → GELU FF → AdaLN unpatch. Confirm against upstream before a
weight load.

---

## TeaCache (reused, not reinvented)

Same controller as Predict2 (`cosmos::sol`, `FASTVIDEO_COSMOS_SOL=teacache`):

| knob | value |
|---|---|
| threshold | **1.15** |
| first eligible step | **10** |
| max consecutive reuses | **3** |
| signal | DiT time embed (`temb`), mean-absolute relative L1 |

---

## Single-GPU NVFP4 / FP8

One 96 GB PRO 6000 needs quantized weights. Plan (WS-H already landed):

| switch | effect |
|---|---|
| `FASTVIDEO_NVFP4=1` | TransformerEngine `NVFP4BlockScaling` (`static_6`) |
| `FASTVIDEO_NVFP4=mse` | FourOverSix opt-in |
| `FASTVIDEO_NVFP4_COSMOS_STEPS=1` | middle denoising steps only (`fp4_linear`: skip first **3** + last **3**) |

`fp4_linear` names those steps. Linears are not quantized in this scaffold.
FP8 (`FASTVIDEO_FP8`) is the existing per-tensor E4M3 path. Tile-IR W4A4
GEMM stays off. Multi-GPU sequence parallel is **out of scope**.

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::cosmos3` canvas / FlowMatch / TeaCache re-export | landed |
| Super 64B DiT dims / Hub id / text / VAE | **TODO(upstream)** |
| cudarc tiny T2V forward | landed (zeros graph) |
| Registry / CLI / gpucheck | **TODO** (no published Hub id) |
| Weight load / generate | **TODO** |
| Multi-GPU SP | out of scope |
