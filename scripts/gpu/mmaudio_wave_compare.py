#!/usr/bin/env python3
"""Waveform similarity of two MMAudio outputs (raw little-endian f32 mono,
44.1 kHz): sample-domain rel-L2 / cosine, and spectral measures that do not
punish phase drift — log-mel L1 (dB), spectral convergence of the STFT
magnitude, the correlation of the log-mel images and the cosine of the
per-band mel energy profile.

    python mmaudio_wave_compare.py --ref a.f32 --cand b.f32 [--label x] [--out x.json]
"""
import argparse
import json

import numpy as np


def load(p):
    return np.fromfile(p, dtype="<f4").astype(np.float64)


def measures(ref, cand, sr=44100):
    import librosa
    n = min(len(ref), len(cand))
    r, c = ref[:n], cand[:n]
    out = {"samples_ref": len(ref), "samples_cand": len(cand), "compared": n,
           "rel_l2": float(np.linalg.norm(c - r) / max(np.linalg.norm(r), 1e-12)),
           "cosine": float(np.dot(r, c) / max(np.linalg.norm(r) * np.linalg.norm(c), 1e-12)),
           "rms_ref": float(np.sqrt(np.mean(r ** 2))), "rms_cand": float(np.sqrt(np.mean(c ** 2)))}
    mr = np.abs(librosa.stft(r, n_fft=2048, hop_length=512))
    mc = np.abs(librosa.stft(c, n_fft=2048, hop_length=512))
    out["spectral_convergence"] = float(np.linalg.norm(mc - mr) / max(np.linalg.norm(mr), 1e-12))
    melr = librosa.feature.melspectrogram(S=mr ** 2, sr=sr, n_mels=128)
    melc = librosa.feature.melspectrogram(S=mc ** 2, sr=sr, n_mels=128)
    lr = np.maximum(librosa.power_to_db(melr, ref=1.0, top_db=None), -100)
    lc = np.maximum(librosa.power_to_db(melc, ref=1.0, top_db=None), -100)
    out["log_mel_l1_db"] = float(np.mean(np.abs(lr - lc)))
    er, ec = melr.mean(axis=1), melc.mean(axis=1)
    out["mel_band_energy_cosine"] = float(np.dot(er, ec) / max(np.linalg.norm(er) * np.linalg.norm(ec), 1e-12))
    fr, fc = lr.flatten() - lr.mean(), lc.flatten() - lc.mean()
    out["log_mel_corr"] = float(np.dot(fr, fc) / max(np.linalg.norm(fr) * np.linalg.norm(fc), 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    m = measures(load(a.ref), load(a.cand))
    m["label"] = a.label
    print(json.dumps(m, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
