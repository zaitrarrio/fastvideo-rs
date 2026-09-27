#!/usr/bin/env python3
"""Wan 2.2 TI2V-5B module oracle: Diffusers reference tensors in our dump format.

Writes into ``$FV_ORACLE_DUMP_DIR`` (or ``--out``) what
``fv-gpucheck wan oracle`` reads and diffs (crates/fastvideo-gpucheck/src/wan_oracle.rs):

* ``vae_enc_in`` (a 9-frame 704x1280 clip in [-1, 1], a slow pan over
  ``scripts/gpu/fixtures/ti2v-beach-832x480.jpg``), ``vae_enc_out`` (the
  posterior mean, ``AutoencoderKLWan.encode(x).latent_dist.mode()``),
  ``vae_dec_in`` (= that mean, VAE space) and ``vae_dec_out``
  (``AutoencoderKLWan.decode(z).sample``). fp32, Diffusers' own chunking
  (encode: frame 0, then 4 frames per pass; decode: one latent frame per pass).
* Per DiT case ``t2v`` (one timestep) and ``i2v`` (the TI2V
  ``expand_timesteps`` input: frame 0's tokens at timestep 0): the inputs
  ``<case>_dit_latents`` (seeded N(0, 1), 704x1280x121's 48x31x44x80),
  ``<case>_dit_encoder`` (512 tokens, the first 48 N(0, 0.25^2), the rest
  zero as UMT5 padding), ``<case>_dit_timestep`` / ``<case>_dit_timestep_frames``
  (ours: one value per latent frame), and the outputs ``<case>_patch_embed``,
  ``<case>_timestep_proj`` (``[frames or 1, 6, dim]``), ``<case>_block_<i>``
  (every 64th token row, as dump.rs) and ``<case>_dit_out``.
  ``WanTransformer3DModel`` in bf16, as the pipeline runs it.

The weights are the volume's ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` copy, read only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

STRIDE = 64  # dump.rs BLOCK_ROW_STRIDE


def dump(out: Path, name: str, t) -> None:
    a = t.detach().to(torch.float32).cpu().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t, np.float32)
    a.astype("<f4").tofile(out / f"{name}.f32")
    (out / f"{name}.shape").write_text(" ".join(str(d) for d in a.shape))


def clip_from_image(path: Path, height: int, width: int, frames: int) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    pan = 4 * (frames - 1)
    scale = max(width / img.width, (height + pan) / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x0 = (img.width - width) // 2
    out = []
    for k in range(frames):
        y0 = 4 * k
        a = np.asarray(img.crop((x0, y0, x0 + width, y0 + height)), dtype=np.float32) / 127.5 - 1.0
        out.append(torch.from_numpy(a).permute(2, 0, 1))
    return torch.stack(out, dim=1).unsqueeze(0)  # [1, 3, F, H, W]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Diffusers root (vae/, transformer/)")
    ap.add_argument("--out", default=os.environ.get("FV_ORACLE_DUMP_DIR", ""))
    ap.add_argument("--image", required=True)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--vae-frames", type=int, default=9)
    ap.add_argument("--dit-frames", type=int, default=121)
    ap.add_argument("--timestep", type=float, default=781.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-dit", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    meta: dict = {"model": a.model, "args": vars(a), "torch": torch.__version__}
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    import diffusers

    meta["diffusers"] = diffusers.__version__

    # ---- VAE (fp32)
    vae = AutoencoderKLWan.from_pretrained(Path(a.model) / "vae", torch_dtype=torch.float32).to(dev).eval()
    video = clip_from_image(Path(a.image), a.height, a.width, a.vae_frames)
    dump(out, "vae_enc_in", video)
    with torch.no_grad():
        t0 = time.perf_counter()
        mu = vae.encode(video.to(dev)).latent_dist.mode()
        torch.cuda.synchronize()
        meta["vae_encode_s"] = time.perf_counter() - t0
        dump(out, "vae_enc_out", mu)
        dump(out, "vae_dec_in", mu)
        t0 = time.perf_counter()
        rec = vae.decode(mu).sample
        torch.cuda.synchronize()
        meta["vae_decode_s"] = time.perf_counter() - t0
        dump(out, "vae_dec_out", rec)
    meta["vae_latent_shape"] = list(mu.shape)
    meta["vae_recon_psnr_db"] = float(10 * torch.log10(4.0 / ((rec.cpu() - video) ** 2).mean()))
    del vae, mu, rec
    torch.cuda.empty_cache()

    # ---- DiT (bf16)
    if not a.skip_dit:
        tr = WanTransformer3DModel.from_pretrained(Path(a.model) / "transformer", torch_dtype=torch.bfloat16).to(dev).eval()
        cfg = tr.config
        dim = cfg.num_attention_heads * cfg.attention_head_dim
        g = torch.Generator("cpu").manual_seed(a.seed)
        T = (a.dit_frames - 1) // 4 + 1
        h, w = a.height // 16, a.width // 16
        latents = torch.randn(1, cfg.in_channels, T, h, w, generator=g, dtype=torch.float32)
        enc = torch.zeros(1, 512, cfg.text_dim)
        enc[:, :48] = torch.randn(1, 48, cfg.text_dim, generator=g) * 0.25
        tok_per_frame = (h // cfg.patch_size[1]) * (w // cfg.patch_size[2])
        state: dict = {"case": None}

        def hook_patch(_m, _i, o):
            dump(out, f"{state['case']}_patch_embed", o.flatten(2).transpose(1, 2)[0, ::STRIDE])

        def hook_cond(_m, _i, o):
            tp = o[1]
            if tp.dim() == 3:  # [1, seq, 6*dim]: one row per latent frame
                tp = tp[0, ::tok_per_frame]
            dump(out, f"{state['case']}_timestep_proj", tp.reshape(-1, 6, dim))

        def hook_block(i):
            def f(_m, _i, o):
                dump(out, f"{state['case']}_block_{i}", o[0, ::STRIDE])
            return f

        hooks = [tr.patch_embedding.register_forward_hook(hook_patch),
                 tr.condition_embedder.register_forward_hook(hook_cond)]
        hooks += [b.register_forward_hook(hook_block(i)) for i, b in enumerate(tr.blocks)]
        for case in ("t2v", "i2v"):
            state["case"] = case
            dump(out, f"{case}_dit_latents", latents)
            dump(out, f"{case}_dit_encoder", enc)
            if case == "t2v":
                ts = torch.tensor([a.timestep], dtype=torch.float32)
                dump(out, f"{case}_dit_timestep", ts)
            else:
                mask = torch.ones(T, h, w)
                mask[0] = 0
                ts = (mask[:, ::2, ::2] * a.timestep).flatten().unsqueeze(0)
                frames = torch.full((T,), a.timestep)
                frames[0] = 0
                dump(out, f"{case}_dit_timestep_frames", frames)
            with torch.no_grad():
                t0 = time.perf_counter()
                o = tr(hidden_states=latents.to(dev, torch.bfloat16), timestep=ts.to(dev),
                       encoder_hidden_states=enc.to(dev, torch.bfloat16), return_dict=False)[0]
                torch.cuda.synchronize()
                meta[f"{case}_dit_s"] = time.perf_counter() - t0
            dump(out, f"{case}_dit_out", o)
        for hk in hooks:
            hk.remove()
    (out / "oracle_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
