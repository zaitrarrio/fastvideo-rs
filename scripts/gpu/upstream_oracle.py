#!/usr/bin/env python3
"""An external oracle for the text encoder and one DiT step.

Every other gate we have compares fastvideo-rs against fastvideo-rs: `parity`
is GPU vs our own CPU path, `compare` diffs two of our own clip dirs. A shared
algorithmic error is invisible to all of them, which is how the port shipped
for days rendering the subject of a prompt and none of its scene.

This runs the *reference* implementations — transformers' UMT5 and diffusers'
WanTransformer3DModel — on the same weights, and writes the inputs and outputs
so `fv-gpucheck oracle` can diff our stack against them:

    noise        [1, 16, Tl, Hl, Wl]  latent input, from a fixed seed
    timestep     [1]
    text         [1, 512, 4096]       oracle prompt embedding (conditional)
    text_neg     [1, 512, 4096]       oracle negative embedding
    dit          [1, 16, Tl, Hl, Wl]  oracle DiT output for (noise, t, text)

Both sides then run on byte-identical inputs, so the three comparisons
attribute error rather than just detecting it:

    text    ours vs oracle embedding             → the UMT5 port
    dit     our DiT on the *oracle* embedding    → the DiT port
    e2e     our DiT on *our* embedding           → what a clip actually gets

Everything is float32: our exact mode is float32, and an oracle that is looser
than the thing it judges cannot set a limit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-weights", required=True, help="Wan2.1 dir with text_encoder/ and tokenizer/")
    ap.add_argument("--dit-weights", required=True, help="FastWan dir with transformer/")
    ap.add_argument("--prompts", required=True, help="prompt JSON, same file the embed stage reads")
    ap.add_argument("--name", default=None, help="which prompt to use (default: the first)")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument("--timestep", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--text-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    spec = json.load(open(args.prompts))
    entry = spec["prompts"][0] if args.name is None else next(p for p in spec["prompts"] if p["name"] == args.name)
    prompt, negative = entry["prompt"], spec["negative"]

    meta: dict[str, object] = {
        "prompt": prompt,
        "name": entry["name"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "spec": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "timestep": args.timestep,
            "seed": args.seed,
            "text_dtype": args.text_dtype,
        },
    }
    out: dict[str, "torch.Tensor"] = {}

    # --- text encoder -----------------------------------------------------
    # diffusers' WanPipeline._get_t5_prompt_embeds, reproduced exactly: encode
    # the padded batch *with* its attention mask, then keep the real tokens and
    # zero-fill to max_sequence_length.
    from transformers import AutoTokenizer, UMT5EncoderModel

    t_dtype = getattr(torch, args.text_dtype)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_weights, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.base_weights, subfolder="text_encoder", torch_dtype=t_dtype
    ).eval().to("cuda")
    meta["load_text_s"] = time.time() - t0

    @torch.no_grad()
    def encode(text: str) -> tuple["torch.Tensor", int]:
        ti = tokenizer(
            [text],
            padding="max_length",
            max_length=args.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = ti.input_ids.to("cuda"), ti.attention_mask.to("cuda")
        seq_len = int(mask.gt(0).sum())
        hs = text_encoder(ids, attention_mask=mask).last_hidden_state.float()
        kept = hs[0, :seq_len]
        pad = kept.new_zeros(args.max_sequence_length - seq_len, kept.size(1))
        return torch.cat([kept, pad]).unsqueeze(0).contiguous(), seq_len

    t0 = time.time()
    text, n_tok = encode(prompt)
    text_neg, n_tok_neg = encode(negative)
    meta["encode_s"] = time.time() - t0
    meta["tokens"] = {"prompt": n_tok, "negative": n_tok_neg}
    out["text"] = text.cpu()
    out["text_neg"] = text_neg.cpu()

    # The DiT needs the memory back; 5.6B params in float32 is most of the card.
    del text_encoder
    torch.cuda.empty_cache()

    # --- one DiT step -----------------------------------------------------
    from diffusers import WanTransformer3DModel

    t0 = time.time()
    dit = WanTransformer3DModel.from_pretrained(
        args.dit_weights, subfolder="transformer", torch_dtype=torch.float32
    ).eval().to("cuda")
    meta["load_dit_s"] = time.time() - t0

    # Wan's VAE is 8x spatial and 4x temporal with the first frame kept whole.
    lat_t = (args.num_frames - 1) // 4 + 1
    lat_h, lat_w = args.height // 8, args.width // 8
    shape = (1, 16, lat_t, lat_h, lat_w)
    meta["latent_shape"] = list(shape)

    # Generated on the CPU so the saved tensor is what both sides consume, with
    # no dependence on anyone's device RNG.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(shape, generator=g, dtype=torch.float32)
    out["noise"] = noise
    out["timestep"] = torch.tensor([args.timestep], dtype=torch.float32)

    with torch.no_grad():
        t0 = time.time()
        sample = dit(
            hidden_states=noise.to("cuda"),
            timestep=out["timestep"].to("cuda"),
            encoder_hidden_states=text.to("cuda"),
            return_dict=True,
        ).sample
        torch.cuda.synchronize()
        meta["dit_forward_s"] = time.time() - t0
    out["dit"] = sample.float().cpu().contiguous()

    meta["stats"] = {
        k: {"mean": float(v.mean()), "std": float(v.std()), "absmax": float(v.abs().max())}
        for k, v in out.items()
        if v.numel() > 1
    }
    save_file(out, args.out)
    json.dump(meta, open(args.meta, "w"), indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
