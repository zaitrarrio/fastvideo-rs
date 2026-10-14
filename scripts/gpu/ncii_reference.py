#!/usr/bin/env python3
"""Reference scores of the NCII prompt guard for the Rust port (fastvideo_models::ncii).

Runs `hfmlsoc/ncii-light-guard-v01` exactly as the MiniMax-H3-Turbo-Lora Space's `ncii_guard.py` does
(`transformers.pipeline("text-classification", device="cpu")`, `truncation=True`) on a few prompts and writes
`crates/fastvideo-models/tests/fixtures/ncii_reference.json`: the token ids, the pipeline's top label and score,
and the `ncii` probability. The Rust test `ncii::tests::matches_transformers_when_weights_are_present`
(`FV_NCII_DIR=<dir> cargo test -p fastvideo-models ncii -- --ignored`) compares against it.

    python3 scripts/gpu/ncii_reference.py <guard dir> [out.json]

Needs torch (CPU) and transformers 5.x. The prompts are deliberately mild (the model card's own example and
ordinary edit requests); docs/ports/h3-turbo-loras.md "NCII guard".
"""

import json
import sys

PROMPTS = [
    "skirt riding up the thigh",  # the model card's example (ncii)
    "make the sky a warm sunset orange",
    "replace the background with a snowy mountain range",
    "the woman turns to the camera and smiles, then waves",
    "a corgi in a chef hat flipping a pancake in a sunlit kitchen, sizzling sounds and a cheerful bark, cinematic",
    "change her jacket to a red raincoat",
    "remove the people in the background",
    # A long prompt: more than the tokenizer's 256-token truncation.
    " ".join(["A slow dolly shot along a rainy neon street at night, reflections on the wet asphalt,"] * 24),
]


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, pipeline

    guard = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "crates/fastvideo-models/tests/fixtures/ncii_reference.json"
    clf = pipeline("text-classification", model=guard, device="cpu")
    tok = AutoTokenizer.from_pretrained(guard)
    model = clf.model
    cases = []
    for p in PROMPTS:
        top = clf(p, truncation=True)[0]
        enc = tok(p, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0].float()
        probs = torch.softmax(logits, -1)
        ncii = float(probs[model.config.label2id["ncii"]])
        cases.append({
            "prompt": p,
            "ids": enc["input_ids"][0].tolist(),
            "label": top["label"],
            "score": float(top["score"]),
            "ncii": ncii,
            "logits": [float(x) for x in logits],
        })
        print(f"{top['label']:5s} {top['score']:.4f} ncii={ncii:.4f} tokens={enc['input_ids'].shape[1]}  {p[:60]}")
    doc = {
        "model": "hfmlsoc/ncii-light-guard-v01",
        "revision": "6a44e8b4b608078db164b0c946947922b2b98eab",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": str(model.dtype),
        "cases": cases,
    }
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
