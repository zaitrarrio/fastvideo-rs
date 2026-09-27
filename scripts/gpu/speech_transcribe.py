#!/usr/bin/env python3
"""Whisper intelligibility check for generated audio (runpod-matrix.sh speechtest).

Reads a TSV manifest (id, wav, intended line, note) and transcribes every wav
with openai-whisper (English forced, temperature 0). Per clip it writes the
transcript, word error rate against the intended line (lowercased, punctuation
stripped, small numbers spelled out), Whisper's no_speech_prob (first segment
and mean), avg_logprob, and the unforced language-ID probability of English.

  speech_transcribe.py --manifest manifest.tsv --model large-v3 --out speech.json
"""
import argparse
import json
import re
import sys

NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
       "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def norm(text):
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = [NUM.get(w, w) for w in text.split()]
    return [w.strip("'") for w in words if w.strip("'")]


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import whisper  # noqa: E402

    model = whisper.load_model(a.model)
    rows = []
    for line in open(a.manifest):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        cid, wav, intended, note = (line.split("\t") + ["", "", "", ""])[:4]
        try:
            audio = whisper.load_audio(wav)
        except Exception as e:  # missing / unreadable output
            rows.append({"id": cid, "wav": wav, "error": str(e)})
            print(f"{cid}: ERROR {e}", file=sys.stderr)
            continue
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        res = model.transcribe(audio, language="en", temperature=0.0, condition_on_previous_text=False)
        segs = res.get("segments", [])
        text = res.get("text", "").strip()
        row = {
            "id": cid,
            "wav": wav,
            "intended": intended,
            "note": note,
            "seconds": len(audio) / 16000.0,
            "transcript": text,
            "wer": wer(intended, text) if intended else None,
            "no_speech_prob": segs[0]["no_speech_prob"] if segs else 1.0,
            "no_speech_prob_mean": sum(s["no_speech_prob"] for s in segs) / len(segs) if segs else 1.0,
            "avg_logprob": sum(s["avg_logprob"] for s in segs) / len(segs) if segs else None,
            "p_english": float(probs.get("en", 0.0)),
            "top_language": max(probs, key=probs.get),
            "segments": [{k: s[k] for k in ("start", "end", "text", "no_speech_prob", "avg_logprob", "compression_ratio")} for s in segs],
        }
        rows.append(row)
        print(f"{cid}: wer={row['wer']} nsp={row['no_speech_prob']:.3f} en={row['p_english']:.2f} | {text}", file=sys.stderr)
    json.dump({"model": a.model, "clips": rows}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
