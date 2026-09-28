#!/usr/bin/env python3
"""Audio-visual sync proxy for speech clips (no SyncNet weights).

    python3 scripts/gpu/lipsync_proxy.py [--json out.json] [--pool LABEL] clip.mp4 [clip2.mp4 ...]

SyncNet needs its own checkpoint, and new weights must go on both network
volumes with the owner's approval (CLAUDE.md), so this is a documented proxy
instead (docs/serve/h3-1080p-and-upscaler.md, "Lip sync"):

1. Face: OpenCV's Haar frontal-face detector (bundled with
   opencv-python-headless 4.x) on every frame, downscaled to 640 px wide; the
   largest face, its box median-smoothed over 3 frames.
2. Articulation: the mouth region (face box x 25-75 %, y 68-95 %) and the
   upper face (x 15-85 %, y 20-50 %), each resized to 96x48 so clips of
   different resolutions are measured alike; per frame, the mean absolute
   change of the mouth minus that of the upper face (head motion, blinks and
   camera moves cancel, speech movements remain).
3. Audio: the MP4's audio in the speech band (300-3400 Hz), 16 kHz mono, RMS
   per video frame (1/fps s).
4. Pearson correlation of articulation with the audio envelope at lags
   -6..+6 frames (+-250 ms at 24 fps). Reported: the best lag (positive =
   video lags audio), r there, r at lag 0, a confidence (best r minus the
   median r over all lags, the analogue of SyncNet's min-vs-median
   distance), and the speech contrast (articulation in the loudest 30 % of
   frames minus the quietest 30 %, in standard deviations).

One 5 s clip gives about 124 samples, so single-clip lags are noisy;
`--pool LABEL` also reports the correlation over all given clips
concatenated (each clip z-scored on its own), which is what to compare
between arms (768p vs 1080p of the same prompts and seeds). In sync: the
pooled best lag within +-2 frames (+-83 ms), positive r and speech contrast.
Needs numpy, opencv-python-headless 4.x (the 5.x wheel drops the cascades),
ffmpeg on PATH.
"""
import argparse
import json
import subprocess
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("lipsync_proxy.py needs opencv-python-headless 4.x (pip install 'opencv-python-headless<5')")

MAX_LAG = 6


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         stdout=subprocess.PIPE)
    n = w * h
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w)
    p.wait()


def audio_envelope(path, fps, count):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
         "-af", "highpass=f=300,lowpass=f=3400", "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    a = np.frombuffer(raw, np.float32).astype(np.float64)
    hop = 16000.0 / fps
    env = np.zeros(count)
    for i in range(count):
        seg = a[int(i * hop):int((i + 1) * hop)]
        env[i] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    return env


def smooth_boxes(boxes, k=3):
    idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        return None
    arr = np.array([boxes[i] if boxes[i] is not None else boxes[min(idx, key=lambda j: abs(j - i))]
                    for i in range(len(boxes))], dtype=np.float64)
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i - k // 2), min(len(arr), i + k // 2 + 1)
        out[i] = np.median(arr[lo:hi], axis=0)
    return out


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def lag_corr(v, a, lag):
    """Pearson r with the video signal shifted: positive lag = video follows audio."""
    if lag > 0:
        v, a = v[lag:], a[:-lag]
    elif lag < 0:
        v, a = v[:lag], a[-lag:]
    if len(v) < 10 or v.std() == 0 or a.std() == 0:
        return 0.0
    return float(np.corrcoef(v, a)[0, 1])


def sync_stats(v, env, fps, segments=None):
    """Lag scan of `v` against `env`. `segments`: clip boundaries of a pooled
    signal, so a shift never pairs one clip's video with another's audio."""
    segments = segments or [(0, len(v))]

    def corr(lag):
        vs, es = [], []
        for s, e in segments:
            cv_, ce = v[s:e], env[s:e]
            if lag > 0:
                cv_, ce = cv_[lag:], ce[:-lag]
            elif lag < 0:
                cv_, ce = cv_[:lag], ce[-lag:]
            vs.append(cv_)
            es.append(ce)
        return lag_corr(np.concatenate(vs), np.concatenate(es), 0)

    rs = {lag: corr(lag) for lag in range(-MAX_LAG, MAX_LAG + 1)}
    best = max(rs, key=rs.get)
    loud, quiet = env >= np.quantile(env, 0.7), env <= np.quantile(env, 0.3)
    return {
        "best_lag_frames": best,
        "best_lag_ms": round(1000.0 * best / fps, 1),
        "r_best": round(rs[best], 3),
        "r_lag0": round(rs[0], 3),
        "confidence": round(rs[best] - float(np.median(list(rs.values()))), 3),
        "speech_contrast": round(float(v[loud].mean() - v[quiet].mean()), 3),
        "r_by_lag": {str(k): round(x, 3) for k, x in rs.items()},
    }


def signals(path):
    w, h, fps = probe(path)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    scale = 640.0 / w
    grays, boxes = [], []
    for g in frames(path, w, h):
        small = cv2.resize(g, (640, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        faces = det.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        boxes.append(None if len(faces) == 0 else tuple(v / scale for v in max(faces, key=lambda f: f[2] * f[3])))
        grays.append(g)
    n = len(grays)
    found = sum(b is not None for b in boxes)
    info = {"clip": path, "size": f"{w}x{h}", "fps": fps, "frames": n, "face_frames": found}
    sm = smooth_boxes(boxes)
    if sm is None or found < n // 2:
        return info, None, None

    def region(g, box, x0, x1, y0, y1):
        x, y, bw, bh = box
        crop = g[int(y + y0 * bh):int(min(h, y + y1 * bh)), int(x + x0 * bw):int(x + x1 * bw)]
        return cv2.resize(crop, (96, 48), interpolation=cv2.INTER_AREA).astype(np.float64)

    art, prev = [], None
    for g, box in zip(grays, sm):
        cur = (region(g, box, 0.25, 0.75, 0.68, 0.95), region(g, box, 0.15, 0.85, 0.20, 0.50))
        art.append(0.0 if prev is None else
                   float(np.abs(cur[0] - prev[0]).mean() - np.abs(cur[1] - prev[1]).mean()))
        prev = cur
    if n > 1:
        art[0] = art[1]
    return info, zscore(art), zscore(audio_envelope(path, fps, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--pool", metavar="LABEL", help="also report the pooled statistics under this label")
    args = ap.parse_args()
    out, vs, es, segs, fps = [], [], [], [], 24.0
    for c in args.clips:
        info, v, env = signals(c)
        if v is None:
            info["verdict"] = "no stable face"
        else:
            info.update(sync_stats(v, env, info["fps"]))
            s0 = sum(len(x) for x in vs)
            vs.append(v)
            es.append(env)
            segs.append((s0, s0 + len(v)))
            fps = info["fps"]
        out.append(info)
        print(f"{c}: {info['size']} faces {info['face_frames']}/{info['frames']} "
              f"lag {info.get('best_lag_frames')} r {info.get('r_best')} (lag0 {info.get('r_lag0')}) "
              f"conf {info.get('confidence')} speech {info.get('speech_contrast')}")
    report = {"clips": out}
    if args.pool and vs:
        pooled = sync_stats(np.concatenate(vs), np.concatenate(es), fps, segs)
        pooled["clips"] = len(vs)
        report["pooled"] = {"label": args.pool, **pooled}
        print(f"POOLED {args.pool} ({len(vs)} clips): lag {pooled['best_lag_frames']} ({pooled['best_lag_ms']} ms) "
              f"r {pooled['r_best']} (lag0 {pooled['r_lag0']}) conf {pooled['confidence']} speech {pooled['speech_contrast']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
