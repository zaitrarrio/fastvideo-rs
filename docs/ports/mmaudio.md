# MMAudio — port specification

FastVideo family `mmaudio`: multimodal audio DiT for **V2A** / **T2A**.
Registered Hub id `FastVideo/MMAudio-large-44k-v2-Diffusers` is reserved but
not yet public — convert official weights locally and set `MMAUDIO_MODEL_PATH`.

Sources (read 2026-09-22): FastVideo support matrix note on MMAudio.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `mmaudio_large_44k_v2` | `FastVideo/MMAudio-large-44k-v2-Diffusers` | reserved; use `MMAUDIO_MODEL_PATH` |

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
| Public Hub weights / Synchformer+CLIP | external blocker (`MMAUDIO_MODEL_PATH`) |
