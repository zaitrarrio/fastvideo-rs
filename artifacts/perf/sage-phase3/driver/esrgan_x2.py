#!/usr/bin/env python3
"""Real-ESRGAN x2plus per frame (RRDBNet inline, no basicsr).

    python esrgan_x2.py <RealESRGAN_x2plus.pth> <out root> <src dir>...

For each src dir of frame-NNN.png, writes <out root>/<basename>/frame-NNN.png
and prints one JSON line: model seconds (CUDA synchronised, fp16, batches of
8 frames), PNG read/write seconds, peak memory. The first dir is run twice
(the first pass warms cuDNN autotuning); the second is the reported one.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNetX2(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3 * 4, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(F.pixel_unshuffle(x, 2))
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


ckpt, out_root, srcs = sys.argv[1], sys.argv[2], sys.argv[3:]
net = RRDBNetX2()
sd = torch.load(ckpt, map_location="cpu")
net.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=True)
net = net.half().cuda().eval()
torch.backends.cudnn.benchmark = True


def run(src, write):
    names = sorted(p for p in os.listdir(src) if p.endswith(".png"))
    t0 = time.time()
    frames = [np.asarray(Image.open(os.path.join(src, n)).convert("RGB")) for n in names]
    t1 = time.time()
    outs = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t2 = time.time()
    with torch.no_grad():
        for i in range(0, len(frames), 8):
            x = torch.from_numpy(np.stack(frames[i:i + 8])).cuda().permute(0, 3, 1, 2).half() / 255.0
            y = net(x).clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1)
            outs.append(y.cpu())
    torch.cuda.synchronize()
    t3 = time.time()
    if write:
        d = os.path.join(out_root, os.path.basename(src.rstrip("/")))
        os.makedirs(d, exist_ok=True)
        k = 0
        for y in outs:
            for f in y.numpy():
                Image.fromarray(f).save(os.path.join(d, names[k]))
                k += 1
    t4 = time.time()
    h, w = frames[0].shape[:2]
    print(json.dumps({"src": src, "write": write, "frames": len(frames), "in": f"{w}x{h}", "out": f"{2 * w}x{2 * h}",
                      "read_s": round(t1 - t0, 2), "model_s": round(t3 - t2, 3), "write_s": round(t4 - t3, 2),
                      "fps": round(len(frames) / (t3 - t2), 1),
                      "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)


run(srcs[0], False)
for s in srcs:
    run(s, True)
