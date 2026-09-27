#!/usr/bin/env python3
"""CPU reference fixtures for the MMAudio port: MMAudio's own classes, seeded
random weights, float32, one run each.

    PYTHONPATH=<MMAudio checkout @974010a> python scripts/gpu/mmaudio_tiny_reference.py \
        --out <dir> [--tokenizer <DFN5B tokenizer.json dir>]

Writes <dir>/<component>.safetensors (weights, `__in_*` inputs and `__out_*`
outputs). `FV_MMAUDIO_FIXTURES=<dir> cargo test -p fastvideo-cudarc --release
mmaudio::reference_tests -- --ignored` loads them through the production
loaders. Toy sizes except Synchformer (its geometry is fixed: 224 px, 16
frames, ViT-B) and the frame preprocessing (a real 480x832 frame size).
The configs are mirrored in crates/fastvideo-cudarc/src/mmaudio/reference_tests.rs.
"""
import argparse
import os
import zlib

import numpy as np
import torch
from safetensors.torch import save_file


def randomize(module, gen, scale=0.2):
    with torch.no_grad():
        for _, p in module.named_parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)


def save(out, name, sd, ins, outs):
    d = {k: v.detach().float().contiguous().clone() for k, v in sd.items()}
    for k, v in ins.items():
        d[f"__in_{k}"] = v.detach().float().contiguous().clone()
    for k, v in outs.items():
        d[f"__out_{k}"] = v.detach().float().contiguous().clone()
    save_file(d, os.path.join(out, f"{name}.safetensors"))
    print(name, {k: tuple(v.shape) for k, v in outs.items()})


def dit(out, gen):
    from mmaudio.model.networks import MMAudio
    net = MMAudio(latent_dim=8, clip_dim=12, sync_dim=10, text_dim=12, hidden_dim=32, depth=3,
                  fused_depth=1, num_heads=2, latent_seq_len=11, clip_seq_len=4, sync_seq_len=16,
                  text_seq_len=5, v2=True).eval()
    randomize(net, gen)
    clip = torch.randn(1, 4, 12, generator=gen)
    sync = torch.randn(1, 16, 10, generator=gen)
    text = torch.randn(1, 5, 12, generator=gen)
    lat = torch.randn(1, 11, 8, generator=gen)
    t = torch.tensor([0.37])
    with torch.no_grad():
        cond = net.preprocess_conditions(clip, sync, text)
        flow = net.predict_flow(lat, t, cond)
    sd = {k: v for k, v in net.state_dict().items() if k not in ("latent_rot", "clip_rot", "t_embed.freqs")}
    save(out, "dit", sd, {"clip": clip, "sync": sync, "text": text, "latent": lat, "t": t},
         {"flow": flow, "clip_f": cond.clip_f, "sync_f": cond.sync_f, "text_f": cond.text_f})


def vae(out, gen):
    from mmaudio.ext.autoencoder.vae import Decoder1D
    from mmaudio.ext.autoencoder.edm2_utils import MPConv1D
    dec = Decoder1D(dim=8, ch_mult=(1, 2, 4), num_res_blocks=2, attn_layers=[3], down_layers=[0],
                    in_dim=6, out_dim=6, embed_dim=4).eval()
    randomize(dec, gen, 1.0)
    sd = {f"decoder.{k}": v.clone() for k, v in dec.state_dict().items()}
    for m in dec.modules():
        if isinstance(m, MPConv1D):
            m.remove_weight_norm()
    z = torch.randn(1, 4, 9, generator=gen)
    with torch.no_grad():
        y = dec(z)
    save(out, "vae", sd, {"z": z}, {"mel": y})


def bigvgan(out, gen):
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN
    from mmaudio.ext.bigvgan_v2.env import AttrDict
    h = AttrDict(dict(num_mels=6, upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4],
                      upsample_initial_channel=16, resblock="1", resblock_kernel_sizes=[3],
                      resblock_dilation_sizes=[[1, 3]], activation="snakebeta", snake_logscale=True,
                      use_tanh_at_final=False, use_bias_at_final=False))
    net = BigVGAN(h).eval()
    randomize(net, gen, 0.3)
    mel = torch.randn(1, 6, 13, generator=gen)
    with torch.no_grad():
        wav = net(mel)
    save(out, "bigvgan", net.state_dict(), {"mel": mel}, {"wav": wav})


def synchformer(out, gen):
    from mmaudio.ext.synchformer import Synchformer
    net = Synchformer().eval()
    randomize(net, gen, 0.02)
    frames = torch.rand(24, 3, 224, 224, generator=gen) * 2 - 1
    x = frames.unsqueeze(0)
    segs = torch.stack([x[:, i * 8:i * 8 + 16] for i in range(2)], dim=1)
    with torch.no_grad():
        f = net(segs)  # (1, 2, 8, 768)
    sd = {k: v for k, v in net.state_dict().items()}
    save(out, "synchformer", sd, {"frames": frames}, {"feat": f.reshape(1, 16, 768)})


def clip(out, gen, tok_dir):
    import open_clip
    from open_clip.model import CLIP
    from mmaudio.model.utils.features_utils import patch_clip
    import torch.nn.functional as F
    m = CLIP(embed_dim=16, vision_cfg=dict(image_size=28, layers=2, width=32, head_width=16, patch_size=14),
             text_cfg=dict(context_length=77, vocab_size=49408, width=32, heads=2, layers=2),
             quick_gelu=True).eval()
    randomize(m, gen, 0.2)
    m = patch_clip(m)
    px = torch.randn(3, 3, 30, 30, generator=gen)
    tok = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    prompt = "A red fox running through tall grass at dusk. <S>The fox cuts through.<E>\nAudio: wind, crickets"
    ids = tok([prompt, ""])
    with torch.no_grad():
        img = m.encode_image(px, normalize=True)
        txt = m.encode_text(ids, normalize=True)
    sd = {k: v for k, v in m.state_dict().items() if k != "attn_mask"}
    save(out, "clip", sd, {"pixels": px, "ids": ids.float()}, {"image": img, "text": txt})
    with open(os.path.join(out, "clip_prompt.txt"), "w") as f:
        f.write(prompt)


def frames(out, gen):
    from torchvision.transforms import v2
    clip_t = v2.Compose([v2.Resize((384, 384), interpolation=v2.InterpolationMode.BICUBIC), v2.ToImage(),
                         v2.ToDtype(torch.float32, scale=True)])
    sync_t = v2.Compose([v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), v2.CenterCrop(224),
                         v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                         v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # Smooth content plus noise, like a decoded video frame.
    yy, xx = np.meshgrid(np.arange(480), np.arange(832), indexing="ij")
    base = np.stack([128 + 100 * np.sin(xx / 37.0 + c) * np.cos(yy / 23.0 - c) for c in range(3)], -1)
    noise = torch.randint(-20, 21, (2, 480, 832, 3), generator=gen).numpy()
    raw = np.clip(base[None] + noise, 0, 255).astype(np.uint8)
    chunk = torch.from_numpy(raw).permute(0, 3, 1, 2)
    save(out, "frames", {}, {"rgb": torch.from_numpy(raw).float()},
         {"clip": clip_t(chunk), "sync": sync_t(chunk)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="dit,vae,bigvgan,synchformer,clip,frames")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    only = a.only.split(",")
    for name, fn in [("dit", dit), ("vae", vae), ("bigvgan", bigvgan), ("synchformer", synchformer),
                     ("frames", frames)]:
        if name in only:
            fn(a.out, torch.Generator().manual_seed(zlib.crc32(name.encode()) % 1000))
    if "clip" in only:
        clip(a.out, torch.Generator().manual_seed(7), None)


if __name__ == "__main__":
    main()
