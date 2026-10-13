#!/usr/bin/env python3
"""Video and audio PSNR between two clips (a reference and a candidate), through ffmpeg.

    av_psnr.py REF CAND [--json OUT]

REF / CAND: an mp4 with an audio track, or a directory holding `output.mp4` (and `audio.wav`, used in place of
the mp4's audio when present: our runs write the WAV before muxing). Video: every decoded frame as rgb24, PSNR
over the whole clip (MSE pooled over all pixels) and the per-frame mean; frames are paired in order and the
shorter clip bounds the count. Audio: both tracks decoded to float32 stereo at 32 kHz (H3's native rate),
PSNR against a peak of 1.0 over the common length, plus the best lag within +-0.1 s (AAC's encoder delay
shifts an mp4's audio a few ms). numpy only (any FastVideo venv has it).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

RATE = 32_000


def _paths(p: str) -> tuple[Path, Path]:
    q = Path(p)
    if q.is_dir():
        mp4 = q / "output.mp4"
        wav = q / "audio.wav"
        return mp4, wav if wav.is_file() else mp4
    return q, q


def _probe(mp4: Path) -> tuple[int, int]:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", str(mp4)], text=True)
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def _frames(mp4: Path, w: int, h: int) -> np.ndarray:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def _audio(src: Path) -> np.ndarray:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", str(RATE),
                                   "-f", "f32le", "-"])
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def psnr(mse: float, peak: float) -> float:
    return float("inf") if mse <= 0 else 10.0 * math.log10(peak * peak / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    (rv, ra), (cv, ca) = _paths(a.ref), _paths(a.cand)
    res: dict = {"ref": str(rv), "cand": str(cv)}
    w, h = _probe(rv)
    cw, ch = _probe(cv)
    if (w, h) != (cw, ch):
        res["note"] = f"sizes differ: ref {w}x{h}, cand {cw}x{ch}; cand scaled to ref"
    fr, fc = _frames(rv, w, h), _frames(cv, w, h)
    n = min(len(fr), len(fc))
    d = fr[:n].astype(np.float64) - fc[:n].astype(np.float64)
    per = (d * d).reshape(n, -1).mean(axis=1)
    res["video"] = {"frames_ref": int(len(fr)), "frames_cand": int(len(fc)), "frames": int(n),
                    "psnr_db": psnr(float(per.mean()), 255.0),
                    "psnr_db_frame_mean": float(np.mean([psnr(float(m), 255.0) for m in per])),
                    "psnr_db_frame_min": float(min(psnr(float(m), 255.0) for m in per))}
    try:
        ar, ac = _audio(ra), _audio(ca)
        m = min(len(ar), len(ac))
        best = None
        for lag in range(-RATE // 10, RATE // 10 + 1, 8):
            x = ar[max(lag, 0):m + min(lag, 0)]
            y = ac[max(-lag, 0):m - max(lag, 0)]
            k = min(len(x), len(y))
            mse = float(((x[:k] - y[:k]) ** 2).mean())
            if best is None or mse < best[1]:
                best = (lag, mse)
        mse0 = float(((ar[:m] - ac[:m]) ** 2).mean())
        res["audio"] = {"samples_ref": int(len(ar)), "samples_cand": int(len(ac)), "rate": RATE,
                        "psnr_db": psnr(mse0, 1.0), "best_lag_samples": int(best[0]),
                        "psnr_db_best_lag": psnr(best[1], 1.0),
                        "rms_ref": float(np.sqrt((ar[:m] ** 2).mean())), "rms_cand": float(np.sqrt((ac[:m] ** 2).mean()))}
    except subprocess.CalledProcessError as e:
        res["audio"] = {"error": str(e)}
    text = json.dumps(res, indent=1)
    print(text)
    if a.json:
        Path(a.json).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
