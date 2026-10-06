#!/usr/bin/env python3
"""Contact sheet: rows = prompts, columns = cells, one frame each, 256 px wide.
   sheet.py <workdir> <name> <frame index> <cell...>  -> <workdir>/sheets/<name>.jpg"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
wk, name, fr, cells = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4:]
prompts = sorted(p.name for p in (wk / cells[0] / "frames").iterdir() if p.is_dir() and p.name not in ("cold", "warmup"))
W = 256
tiles = []
for p in prompts:
    row = []
    for c in cells:
        fs = sorted((wk / c / "frames" / p).glob("*.png"))
        im = Image.open(fs[min(fr, len(fs) - 1)]).convert("RGB") if fs else Image.new("RGB", (W, 144))
        im = im.resize((W, round(im.height * W / im.width)))
        ImageDraw.Draw(im).text((4, 4), f"{c} | {p}", fill=(255, 255, 0))
        row.append(im)
    tiles.append(row)
h = max(t.height for r in tiles for t in r)
out = Image.new("RGB", (W * len(cells), h * len(prompts)))
for i, r in enumerate(tiles):
    for j, t in enumerate(r):
        out.paste(t, (j * W, i * h))
(wk / "sheets").mkdir(exist_ok=True)
out.save(wk / "sheets" / f"{name}.jpg", quality=78)
print(name, out.size)
