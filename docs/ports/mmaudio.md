# MMAudio — port specification

FastVideo family `mmaudio`: multimodal audio DiT for **V2A** / **T2A**.

Public upstream Hub id: **`hkchengrex/MMAudio`** (native `.pth` layout, not
Diffusers). Diffusers conversion id `FastVideo/MMAudio-large-44k-v2-Diffusers`
remains reserved. Convert locally and set `MMAUDIO_MODEL_PATH` to a Diffusers-
shaped root (`transformer/`, `text_encoder/`, …).

Sources (read 2026-09-22): FastVideo support matrix; hkchengrex/MMAudio.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `mmaudio_large_44k_v2` | `hkchengrex/MMAudio` | public native weights |
| (alias) | `FastVideo/MMAudio-large-44k-v2-Diffusers` | reserved Diffusers pack |

Sample rate **44100**, workloads **V2A** + **T2A**.

---

## DiT scaffold dims (host)

Until public Diffusers config lands, scaffold uses:

| piece | value |
|---|---|
| latent channels | **64** |
| layers / heads / head_dim | **28** / **16** / **64** |
| text / visual cond dim | **1024** / **1024** |
| latent length (10 s @ hop 512) | **860** |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::mmaudio` | landed |
| Registry + generate (wav scaffold) | landed |
| cudarc DiT (tiny zeros + load hook) | landed |
| Text encode (T5-11B / CLIP when dirs present) | landed |
| Synchformer visual encode | external (clear error if `image_encoder/` present without graph) |
| Public Hub Diffusers conversion | use `MMAUDIO_MODEL_PATH` or convert from `hkchengrex/MMAudio` |
