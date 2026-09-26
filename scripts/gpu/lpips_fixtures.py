#!/usr/bin/env python3
"""Write the LPIPS validation images (crates/fastvideo-gpucheck/fixtures/lpips).

Deterministic (numpy only, PNG written with zlib): two synthetic scenes plus
perturbations of the kind a lossy generation switch produces (noise, blur,
shift, tone change, posterization). `lpips_ref.py` scores the pairs listed in
`pairs.json` with the official `lpips` package; the Rust port
(`fv-gpucheck lpips`) is pinned to those numbers.

    python3 scripts/gpu/lpips_fixtures.py [out_dir]
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def scene(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3))
    for c in range(3):
        a, b, f = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.02, 0.12)
        img[..., c] = 128 + 60 * np.sin(f * (a * x + b * y) + c) + 30 * np.cos(0.05 * x * (c + 1))
    for _ in range(6):
        cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(6, h / 3)
        col = rng.uniform(0, 255, 3)
        m = (y - cy) ** 2 + (x - cx) ** 2 < r * r
        img[m] = 0.3 * img[m] + 0.7 * col
    for _ in range(4):
        y0, x0 = rng.randint(0, h - 8), rng.randint(0, w - 8)
        y1, x1 = y0 + rng.randint(4, h // 3), x0 + rng.randint(4, w // 3)
        img[y0:y1, x0:x1] = rng.uniform(0, 255, 3)
    # fine texture: stripes in one region
    img[: h // 3, w // 2 :, 1] += 25 * np.sign(np.sin(0.9 * x[: h // 3, w // 2 :]))
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def box_blur(img: np.ndarray, k: int) -> np.ndarray:
    p = k // 2
    f = np.pad(img.astype(np.float64), ((p, p), (p, p), (0, 0)), mode="edge")
    out = np.zeros(img.shape)
    for dy in range(k):
        for dx in range(k):
            out += f[dy : dy + img.shape[0], dx : dx + img.shape[1]]
    return np.clip(np.round(out / (k * k)), 0, 255).astype(np.uint8)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "crates/fastvideo-gpucheck/fixtures/lpips")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    a = scene(96, 128, 0)
    imgs = {
        "a": a,
        "a_noise": np.clip(a.astype(np.float64) + rng.normal(0, 10, a.shape), 0, 255).round().astype(np.uint8),
        "a_blur": box_blur(a, 5),
        "a_shift": np.roll(a, 3, axis=1),
        "a_tone": np.clip(a.astype(np.float64) * 0.8 + 30, 0, 255).round().astype(np.uint8),
        "b": scene(96, 128, 1),
    }
    c = scene(160, 224, 2)
    imgs["c"] = c
    imgs["c_poster"] = (c // 32 * 32 + 16).astype(np.uint8)
    for name, img in imgs.items():
        write_png(out / f"{name}.png", img)
    pairs = [["a", "a"], ["a", "a_noise"], ["a", "a_blur"], ["a", "a_shift"], ["a", "a_tone"], ["a", "b"], ["c", "c_poster"]]
    (out / "pairs.json").write_text(json.dumps({"pairs": pairs}, indent=1) + "\n")
    print(f"wrote {len(imgs)} images, {len(pairs)} pairs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
