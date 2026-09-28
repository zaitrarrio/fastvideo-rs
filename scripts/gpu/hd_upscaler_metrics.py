#!/usr/bin/env python3
"""No-reference quality metrics for the hd upscaler benchmark (hd-upscaler.sh).

    python hd_upscaler_metrics.py <RUNS> <clip>...   > metrics.json

For every clip, each available variant (frame PNGs under
<RUNS>/<variant>/frames/<clip>/) is scored:

- lapvar: variance of the Laplacian of the luma (0-255), mean over frames;
- grad: mean Sobel gradient magnitude;
- hf_share: share of spectral power (Hann window, DC removed) above the 768p
  band limit at that canvas, 0.5 * 768 / H cycles/px (every 4th frame). A
  Lanczos upscale of a 768p frame has almost none there by construction;
- warp_err: temporal flicker. Frame t-1 is warped onto frame t with dense
  optical flow (Farneback) measured on the content's own 768p-class frames
  (the 768p source for Lanczos and the upscalers, the native clip downscaled
  for native 1080p), and the mean absolute luma error is taken over pixels
  whose flow is forward-backward consistent. warp_err_hf does the same on
  the high-pass band (luma minus a 2 px Gaussian): shimmer of fine detail.
  The per-pair series' max / median ratio flags a jump (a batch seam);
- lum_flicker: standard deviation of the frame-mean luma differences;
- psnr_768: fidelity to the 768p input (the variant area-downscaled to
  1344x768 against the source frame); not defined for native 1080p.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = "turbo-768p"
# variant -> (flow source kind, canvas group)
VARIANTS = {
    "turbo-768p-lanczos": "src",
    "seedvr2-3b-1088": "src",
    "turbo-1080p": "self",
    "turbo-768p-lanczos1440": "src",
    "seedvr2-3b-1440": "src",
    "flashvsr-v1.1-x1.5": "src",
    "lanczos-x1.5": "src",
    "flashvsr-v1.1-x2": "src",
    "lanczos-x2": "src",
}


def frames(d: Path):
    return sorted(d.glob("frame-*.png"))


def luma(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def hf_share(y: np.ndarray, cutoff: float) -> float:
    h, w = y.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    f = np.fft.rfft2((y - y.mean()) * win)
    p = (f.real**2 + f.imag**2)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    r = np.sqrt(fx**2 + fy**2)
    p[0, 0] = 0
    return float(p[r > cutoff].sum() / p.sum())


def flows(small):
    """Backward flows (t -> t-1) and a consistency mask per pair, on 768p-class
    luma at half size (672x384; the flow is smooth, and this keeps it cheap)."""
    out = []
    half = [cv2.resize(y, (y.shape[1] // 2, y.shape[0] // 2), interpolation=cv2.INTER_AREA) for y in small]
    for t in range(1, len(half)):
        a, b = half[t].astype(np.uint8), half[t - 1].astype(np.uint8)
        fb = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        ff = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        h, w = a.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mx, my = gx + fb[..., 0], gy + fb[..., 1]
        back = cv2.remap(ff, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.hypot(fb[..., 0] + back[..., 0], fb[..., 1] + back[..., 1])
        mag = np.hypot(fb[..., 0], fb[..., 1])
        ok = (err < 0.5 + 0.05 * mag) & (mx >= 1) & (mx < w - 2) & (my >= 1) & (my < h - 2)
        out.append((fb, ok))
    return out


def warp_errors(ys, fl, crop=(0.0, 1.0)):
    """crop: the horizontal part of the flow's frame the variant covers (a
    centre-cropped output), as fractions of the width (start, width)."""
    h, w = ys[0].shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    e, ehf = [], []
    for t in range(1, len(ys)):
        fb, ok = fl[t - 1]
        if crop != (0.0, 1.0):
            fw = fb.shape[1]
            a, b = int(round(crop[0] * fw)), int(round((crop[0] + crop[1]) * fw))
            fb, ok = fb[:, a:b], ok[:, a:b]
        sy, sx = h / fb.shape[0], w / fb.shape[1]
        f = cv2.resize(fb, (w, h), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(ok.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mx, my = gx + f[..., 0] * sx, gy + f[..., 1] * sy
        prev = cv2.remap(ys[t - 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = np.abs(ys[t] - prev)
        e.append(float(d[m].mean()))
        hp_t = ys[t] - cv2.GaussianBlur(ys[t], (0, 0), 2)
        hp_p = prev - cv2.GaussianBlur(prev, (0, 0), 2)
        ehf.append(float(np.abs(hp_t - hp_p)[m].mean()))
    return e, ehf


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    runs = Path(sys.argv[1])
    clips = sys.argv[2:]
    res = {}
    for clip in clips:
        src = frames(runs / SRC / "frames" / clip)
        if not src:
            continue
        src_y = [luma(p) for p in src]
        src_fl = flows(src_y)
        res[clip] = {}
        for var, kind in VARIANTS.items():
            fs = frames(runs / var / "frames" / clip)
            if not fs:
                continue
            ys = [luma(p) for p in fs]
            n = min(len(ys), len(src_y))
            ys = ys[:n]
            h, w = ys[0].shape
            cutoff = 0.5 * 768 / min(h, w)
            lap = [float(cv2.Laplacian(y, cv2.CV_32F).var()) for y in ys]
            grad = [float(np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1)).mean()) for y in ys]
            hfs = [hf_share(ys[i], cutoff) for i in range(0, n, 4)]
            crop = (0.0, 1.0)
            if kind == "src":
                fl = src_fl[: n - 1]
                sh, sw = src_y[0].shape
                cw = int(round(w * sh / h))
                if abs(cw - sw) >= 16:
                    crop = (((sw - cw) // 2) / sw, cw / sw)
            else:
                small = [cv2.resize(y, (src_y[0].shape[1], src_y[0].shape[0]), interpolation=cv2.INTER_AREA) for y in ys]
                fl = flows(small)
            e, ehf = warp_errors(ys, fl, crop)
            means = np.array([y.mean() for y in ys])
            r = {
                "size": f"{w}x{h}",
                "frames": n,
                "lapvar": float(np.mean(lap)),
                "grad": float(np.mean(grad)),
                "hf_share": float(np.mean(hfs)),
                "hf_cutoff": cutoff,
                "warp_err": float(np.mean(e)),
                "warp_err_hf": float(np.mean(ehf)),
                "warp_err_max_over_median": float(np.max(e) / np.median(e)),
                "warp_err_argmax": int(np.argmax(e)) + 1,
                "warp_err_series": [round(x, 3) for x in e],
                "lum_flicker": float(np.std(np.diff(means))),
            }
            if kind == "src":
                sh, sw = src_y[0].shape
                # A centre-cropped output (FlashVSR's multiples of 128) is
                # compared with the same centre crop of the source.
                cw = int(round(w * sh / h))
                cw = sw if abs(cw - sw) < 16 else cw
                x0 = (sw - cw) // 2
                ps = [psnr(cv2.resize(ys[i], (cw, sh), interpolation=cv2.INTER_AREA), src_y[i][:, x0:x0 + cw]) for i in range(n)]
                r["psnr_768"] = float(np.mean(ps))
            res[clip][var] = r
            print(f"{clip} {var} {r['size']} lapvar={r['lapvar']:.1f} hf={r['hf_share']:.2e} warp={r['warp_err']:.3f} "
                  f"warp_hf={r['warp_err_hf']:.3f} psnr768={r.get('psnr_768', float('nan')):.2f}", file=sys.stderr, flush=True)
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
