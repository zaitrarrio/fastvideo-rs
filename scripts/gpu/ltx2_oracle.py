#!/usr/bin/env python3
"""An external oracle for every stage of the LTX-2 port.

Same contract as `upstream_oracle.py`: run the *reference* implementation
(transformers' Gemma-3 and diffusers' LTX-2 classes) on the released weights,
save both sides of every call, and let our stack be diffed against it stage by
stage, so an error is attributed rather than merely detected. Inputs that are
random are drawn on the CPU from a fixed seed and saved, so nothing depends on
anyone's device RNG.

Stages (each can be skipped; each frees its model before the next loads, so the
whole thing fits one 96 GB card: Gemma-3-12B bf16 ~24 GB, DiT bf16 ~38 GB):

    text      input_ids/attention_mask, the 49 stacked Gemma hidden states for
              the real tokens                                  → the Gemma port
    conn      text_proj_in output, video/audio connector outputs, in the
              pipeline's own bf16 and again in float32       → the connectors
    dit       RoPE tables, one forward on seeded latents for both streams,
              with block 0 / mid / last outputs               → the DiT port
    vae       decode of a fixed small latent                  → the video VAE
    audio     audio-VAE mel and vocoder waveform of a fixed latent
    sample    (--sample) the full 8-step distilled loop, every step's latents

Tensor names carry the stage as a prefix. Everything is saved as float32 (ids
and masks as int32); `--meta` records shapes, dtypes, timings and versions.

`--llm-out` additionally writes the file `fv-gpucheck llm --family gemma3-12b
--oracle <file>` reads (crates/fastvideo-gpucheck/src/llm_oracle.rs), all float32:

    input_ids   [S]           token ids, left-padded to S = 1024
    positions   [S]           the rotary position transformers used for each slot
    attend      [S]           1.0 where the slot may be a key, 0.0 for padding
    hidden_<k>  [1, S, 3840]  output_hidden_states[k]; 0 = scaled embeddings,
                              k = output of decoder layer k, 48 = after the final norm

LTX-2 consumes **all 49** states (pipeline_ltx2.py:350-352 stacks the whole tuple;
connectors.py:427 unflattens it to [B, S, 3840, 49]), so `--llm-taps all` is the
default; taps 1 and 6 (6 = output of the first global-attention layer) are the
early-divergence probes and are always included.

Line references are to diffusers `main` as read for docs/ports/ltx2.md:
`pipeline_ltx2.py` = src/diffusers/pipelines/ltx2/pipeline_ltx2.py,
`connectors.py` = src/diffusers/pipelines/ltx2/connectors.py,
`transformer_ltx2.py` = src/diffusers/models/transformers/transformer_ltx2.py.

`--weights` must be a *distilled* diffusers layout (the 8-sigma schedule is
meaningless on the dev transformer): `rootonchair/LTX-2-19b-distilled`, or a
local conversion of `ltx-2-19b-distilled.safetensors`. `Lightricks/LTX-2`'s own
`transformer/` and `connectors/` are the dev model — both differ by hash.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time

# diffusers pipelines/ltx2/utils.py:27 — the schedule the checkpoint was distilled against.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="rootonchair/LTX-2-19b-distilled", help="distilled diffusers layout (dir or hub id)")
    ap.add_argument(
        "--model-version",
        default="2.0",
        choices=["2.0", "2.5"],
        help="2.0: LTX-2 19B distilled (default). 2.5: LTX-2.5 22B — TODO: load Lightricks/LTX-2.5-Diffusers, Gemma4, ancestral sample loop",
    )
    ap.add_argument("--prompts", required=True, help="prompt JSON, same file the embed stage reads")
    ap.add_argument("--name", default=None, help="which prompt to use (default: the first)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--sigma", type=float, default=0.725, help="noise level of the single DiT forward (a schedule entry)")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--max-sequence-length", type=int, default=1024)
    ap.add_argument("--text-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument(
        "--dit-dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "both"],
        help="float32 needs ~76 GB for weights; 'both' runs bf16 (dit.*, sample.*) then float32 (dit32.*, sample32.*) and records the distance between them",
    )
    ap.add_argument("--tap-stride", type=int, default=16, help="keep every Nth video token of the sub-layer taps (100 MB each otherwise)")
    ap.add_argument("--vae-latent", default="3,8,12", help="F,H,W of the fixed latent for the VAE stage")
    ap.add_argument("--audio-latent-frames", type=int, default=26)
    ap.add_argument("--skip", default="", help="comma list of stages to skip: text,conn,dit,vae,audio")
    ap.add_argument("--sample", action="store_true", help="also run the full 8-step distilled loop")
    ap.add_argument("--llm-out", default=None, help="also write the fv-gpucheck llm oracle file here")
    ap.add_argument("--llm-taps", default="all", help="'all' (0..48, what LTX-2 consumes) or a comma list; 0,1,6,48 always kept")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}
    if "text" in skip and args.llm_out:
        ap.error("--llm-out needs the text stage")
    if "text" in skip and ("conn" not in skip or "dit" not in skip or args.sample):
        # conn consumes text; dit and sample consume conn. Keeping the chain whole
        # is what makes "our DiT on the oracle's embedding" a meaningful diff.
        ap.error("--skip text requires --skip conn,dit and no --sample")
    if "conn" in skip and ("dit" not in skip or args.sample):
        ap.error("--skip conn requires --skip dit and no --sample")

    import torch
    from safetensors.torch import save_file

    dev = "cuda"
    spec = json.load(open(args.prompts))
    entry = spec["prompts"][0] if args.name is None else next(p for p in spec["prompts"] if p["name"] == args.name)
    prompt = entry["prompt"]

    import diffusers
    import transformers

    meta: dict[str, object] = {
        "prompt": prompt,
        "name": entry["name"],
        "weights": args.weights,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "spec": {k: v for k, v in vars(args).items() if k not in ("out", "meta", "prompts", "llm_out")},
        "timings": {},
    }
    out: dict[str, "torch.Tensor"] = {}
    timings: dict[str, float] = meta["timings"]  # type: ignore[assignment]

    def free() -> None:
        # Call *after* `del`-ing the model in the caller's scope; a reference passed
        # in here would keep the weights alive through the collection.
        gc.collect()
        torch.cuda.empty_cache()

    def f32(t: "torch.Tensor") -> "torch.Tensor":
        return t.detach().float().cpu().contiguous()

    def cpu_randn(shape: tuple[int, ...], offset: int) -> "torch.Tensor":
        # One generator per tensor, seed + offset, so adding a stage never shifts
        # the draws of another.
        g = torch.Generator(device="cpu").manual_seed(args.seed + offset)
        return torch.randn(shape, generator=g, dtype=torch.float32)

    def load(cls, subfolder: str, dtype: "torch.dtype", lib: str):
        """from_pretrained across library generations.

        transformers >= 4.56 spells the dtype argument `dtype` and deprecates
        `torch_dtype`; diffusers spells it `torch_dtype`. Each library is tried
        with its own spelling first, then the other. An unknown keyword is not
        always an error — it can be swallowed and the model loaded in float32 — so
        the resulting parameter dtype is checked, not trusted.
        """
        order = ("dtype", "torch_dtype") if lib == "transformers" else ("torch_dtype", "dtype")
        model = None
        for i, kw in enumerate(order):
            try:
                model = cls.from_pretrained(args.weights, subfolder=subfolder, **{kw: dtype})
                break
            except TypeError as e:
                if i + 1 == len(order) or "dtype" not in str(e):
                    raise
        got = next(model.parameters()).dtype
        if got != dtype:
            print(f"[oracle] {cls.__name__} loaded as {got}; casting to {dtype}", file=sys.stderr)
            model = model.to(dtype)
        return model.eval().to(dev)

    # --- text: Gemma-3-12B ---------------------------------------------------
    prompt_embeds = prompt_mask = None
    if "text" not in skip:
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        t_dtype = getattr(torch, args.text_dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.weights, subfolder="tokenizer")
        # pipeline_ltx2.py:327-331 — left padding, pad falls back to eos (Gemma has <pad>=0).
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        t0 = time.time()
        # The shards are float32 on disk (48.7 GB); torch_dtype casts on load, which
        # is what LTX2Pipeline.from_pretrained(torch_dtype=bfloat16) does too.
        text_encoder = load(Gemma3ForConditionalGeneration, "text_encoder", t_dtype, "transformers")
        timings["load_text_s"] = time.time() - t0

        # pipeline_ltx2.py:333-341 — no chat template, no system prompt: the stripped
        # prompt, <bos> prepended by add_special_tokens, padded to max_length.
        ti = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=args.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids, mask = ti["input_ids"].to(dev), ti["attention_mask"].to(dev)
        n_tok = int(mask.sum())
        # Everything downstream (positions, the slice of real tokens, the connectors'
        # padding_side="left") assumes the real tokens are the *last* n. A tokenizer
        # generation that ignores `padding_side` set as an attribute must not pass silently.
        if ids.shape[1] != args.max_sequence_length or not bool(mask[0, -n_tok:].all()) or int(mask[0, : ids.shape[1] - n_tok].sum()) != 0:
            raise SystemExit(f"[oracle] tokenizer did not left-pad to {args.max_sequence_length}: mask sum {n_tok}, shape {tuple(ids.shape)}")
        with torch.no_grad():
            t0 = time.time()
            # pipeline_ltx2.py:347-349 — the multimodal wrapper, text only; position_ids are
            # left to default, i.e. arange(1024) *including* the pad slots
            # (modeling_gemma3.py:529-530), so real tokens sit at 1024-n … 1023.
            enc = text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hidden_states = getattr(enc, "hidden_states", None)
            if hidden_states is None:
                # A wrapper generation that does not surface the decoder's states:
                # ask the language model itself (where it lives has moved between
                # `model.language_model` and `language_model.model`).
                lm = text_encoder.get_decoder() if hasattr(text_encoder, "get_decoder") else None
                for path in ("model.language_model", "language_model.model", "language_model"):
                    if lm is not None:
                        break
                    obj = text_encoder
                    for part in path.split("."):
                        obj = getattr(obj, part, None)
                        if obj is None:
                            break
                    lm = obj
                if lm is None:
                    raise SystemExit("[oracle] no hidden_states on the output and no language model found on the wrapper")
                hidden_states = lm(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
            hidden_states = tuple(hidden_states)
            torch.cuda.synchronize()
            timings["encode_s"] = time.time() - t0
            n_layers = text_encoder.config.get_text_config().num_hidden_layers
            if len(hidden_states) != n_layers + 1:
                raise SystemExit(f"[oracle] expected {n_layers + 1} hidden states (embeddings + every layer), got {len(hidden_states)}")
            # Two facts the port relies on, measured rather than assumed, because how the
            # tuple is assembled changed between transformers generations (explicit loop
            # in 4.x, output-recording hooks in 5.x): state 0 is the *scaled* embedding,
            # and the last state is the final norm's output.
            checks: dict[str, float] = {}
            try:
                emb = text_encoder.get_input_embeddings()(ids)
                real = slice(args.max_sequence_length - n_tok, None)
                checks["state0_vs_embedding_max_abs"] = float((hidden_states[0][0, real].float() - emb[0, real].float()).abs().max())
                last = getattr(enc, "last_hidden_state", None)
                if last is not None:
                    checks["state_last_vs_last_hidden_state_max_abs"] = float((hidden_states[-1][0, real].float() - last[0, real].float()).abs().max())
            except Exception as e:  # diagnostics only
                checks["error"] = repr(e)  # type: ignore[assignment]
            meta["text_checks"] = checks
        # pipeline_ltx2.py:350-352 — 49 states (embeddings, 47 raw layer outputs, and the
        # last one *after* the final norm, modeling_gemma3.py:588-591) stacked on a new
        # last axis, then flattened: feature index = channel * 49 + layer.
        stacked = torch.stack(hidden_states, dim=-1)  # [1, 1024, 3840, 49]
        prompt_embeds = stacked.flatten(2, 3).to(dtype=t_dtype)  # [1, 1024, 188160]
        prompt_mask = mask

        out["text.input_ids"] = ids.to(torch.int32).cpu()
        out["text.attention_mask"] = mask.to(torch.int32).cpu()
        # Pad rows are attention-masked garbage the connectors overwrite; only the
        # real tokens (the last n, left padding) are a fair comparison. ~0.75 MB/token.
        out["text.hidden_states"] = f32(stacked[0, args.max_sequence_length - n_tok :])  # [n, 3840, 49]
        meta["tokens"] = n_tok

        if args.llm_out:
            n_states = len(hidden_states)  # 49
            want = set(range(n_states)) if args.llm_taps == "all" else {int(t) for t in args.llm_taps.split(",")}
            want |= {0, 1, 6, n_states - 1}
            s_len = ids.shape[1]
            llm = {
                "input_ids": ids[0].float().cpu(),  # exact in f32: vocab 262208 < 2^24
                # transformers builds position_ids = arange(S) over the *padded* sequence
                # when none are passed (modeling_gemma3.py:521-530) — it does not restart
                # at the first real token — so slot j rotates by j, pads included.
                "positions": torch.arange(s_len, dtype=torch.float32),
                "attend": mask[0].float().cpu(),
            }
            for k in sorted(want):
                llm[f"hidden_{k}"] = f32(hidden_states[k])  # [1, S, 3840]
            save_file(llm, args.llm_out)
            meta["llm_oracle"] = {
                "file": args.llm_out,
                "taps": sorted(want),
                "reference": f"transformers {transformers.__version__} Gemma3ForConditionalGeneration, {args.text_dtype} on cuda",
                "dtype": args.text_dtype,
                "device": "cuda",
                "saved_as": "float32",
                "padding": "left",
                "positions": "arange(S) over the padded sequence; real tokens occupy the last n slots",
                # In bf16 the sqrt(3840) embedding multiplier is rounded to the weight
                # dtype first (modeling_gemma3.py:107): 62.0, not 61.967735.
                "embed_scale_used": float(torch.tensor(3840.0**0.5).to(t_dtype)),
                "attended": n_tok,
            }
        meta["text_hidden_states_layout"] = "[n_real_tokens, hidden=3840, state=49]; state 0 = scaled embeddings, 48 = post-norm"
        del text_encoder, enc, stacked, hidden_states
        free()

    # --- conn: LTX2TextConnectors -------------------------------------------
    conn_video = conn_audio = conn_mask = None
    if "conn" not in skip:
        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        t0 = time.time()
        connectors = load(LTX2TextConnectors, "connectors", prompt_embeds.dtype, "diffusers")
        timings["load_conn_s"] = time.time() - t0

        captured: dict[str, "torch.Tensor"] = {}
        # text_proj_in is the 188160→3840 bias-free Linear applied after the per-layer
        # masked mean/range normalisation (connectors.py:451-459); its output is the
        # first thing worth diffing, before any attention is involved.
        hook = connectors.text_proj_in.register_forward_hook(lambda _m, _i, o: captured.__setitem__("proj", o))
        with torch.no_grad():
            # pipeline_ltx2.py:1237-1242 — returns (video [B,1024,3840], audio [B,1024,3840],
            # mask [B,1024]). The registers replace every pad slot, so the mask comes
            # back all ones (connectors.py:318, 472): the DiT never masks text.
            conn_video, conn_audio, conn_mask = connectors(prompt_embeds, prompt_mask, padding_side="left")
        hook.remove()
        out["conn.proj"] = f32(captured["proj"])
        out["conn.video"] = f32(conn_video)
        out["conn.audio"] = f32(conn_audio)
        out["conn.mask"] = conn_mask.to(torch.int32).cpu()

        # The pipeline runs the normalisation — a sum over up to 1024×3840 values — in
        # bf16. Re-running in float32 measures how much of our diff is that choice
        # rather than a port error. Float32 weights are an exact widening of bf16.
        connectors.float()
        with torch.no_grad():
            v32, a32, _ = connectors(prompt_embeds.float(), prompt_mask, padding_side="left")
        out["conn.video_f32"] = f32(v32)
        out["conn.audio_f32"] = f32(a32)
        del connectors, v32, a32
        free()

    # --- latent geometry, shared by dit and sample --------------------------
    # pipeline_ltx2.py:1261-1263 — 8x temporal (first frame kept whole), 32x spatial.
    lat_f = (args.num_frames - 1) // 8 + 1
    lat_h, lat_w = args.height // 32, args.width // 32
    # pipeline_ltx2.py:1295-1299 — 16000/160/4 = 25 audio latents per second; Python
    # round() is half-to-even.
    audio_n = round(args.num_frames / args.frame_rate * (16000 / 160 / 4.0))
    meta["latent_grid"] = [lat_f, lat_h, lat_w]
    meta["video_tokens"] = lat_f * lat_h * lat_w
    meta["audio_tokens"] = audio_n

    def pack_video(x: "torch.Tensor") -> "torch.Tensor":
        # LTX2Pipeline._pack_latents with patch 1/1 (pipeline_ltx2.py:648-668):
        # [B,C,F,H,W] → [B, F*H*W, C], token order f-major, then h, then w.
        return x.permute(0, 2, 3, 4, 1).flatten(1, 3)

    def pack_audio(x: "torch.Tensor") -> "torch.Tensor":
        # LTX2Pipeline._pack_audio_latents, patch-less branch (pipeline_ltx2.py:741-742):
        # [B,C,L,M] → [B, L, C*M], feature index = channel * 16 + mel_bin.
        return x.transpose(1, 2).flatten(2, 3)

    transformer = None
    sample_video = sample_audio = None
    if "dit" not in skip or args.sample:
        import contextlib

        from diffusers import LTX2VideoTransformer3DModel

        # "both": the product's bf16 first (tensors `dit.*` / `sample.*`), then the same
        # module widened to float32 (`dit32.*` / `sample32.*`). bf16 weights widen
        # exactly, so the float32 pass is the same function without rounding: the
        # distance between the two passes is the reference's own noise floor, and a
        # port is only wrong by what it differs from float32 *beyond* that.
        passes = ["bfloat16", "float32"] if args.dit_dtype == "both" else [args.dit_dtype]
        t0 = time.time()
        transformer = load(LTX2VideoTransformer3DModel, "transformer", getattr(torch, passes[0]), "diffusers")
        timings["load_dit_s"] = time.time() - t0
        accepted = set(inspect.signature(transformer.forward).parameters)
        current = {"dtype": getattr(torch, passes[0]), "sdpa": "default"}

        # pipeline_ltx2.py:1369-1374 — [B,3,S,2] and [B,1,L,2] patch bounds in seconds /
        # pixels; computed once, identical for every step.
        video_coords = transformer.rope.prepare_video_coords(1, lat_f, lat_h, lat_w, dev, fps=args.frame_rate)
        audio_coords = transformer.audio_rope.prepare_audio_coords(1, audio_n, dev)

        def sdpa_context():
            # The float32 pass wants the plain softmax(QK^T)V, not a fused kernel's
            # reordering. It costs 32*S*S*4 bytes of scores (4.8 GB at 6144 tokens).
            if current["sdpa"] != "math":
                return contextlib.nullcontext()
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(SDPBackend.MATH)

        def dit(video: "torch.Tensor", audio: "torch.Tensor", timestep: "torch.Tensor"):
            """One unguided call, argument for argument pipeline_ltx2.py:1399-1421."""
            d_dtype = current["dtype"]
            kwargs = dict(
                hidden_states=video.to(dev, d_dtype),  # :1389 latents are fp32, cast per call
                audio_hidden_states=audio.to(dev, d_dtype),
                encoder_hidden_states=conn_video.to(d_dtype),
                audio_encoder_hidden_states=conn_audio.to(d_dtype),
                timestep=timestep,  # already sigma*1000 (transformer_ltx2.py:1406-1408)
                sigma=timestep,  # LTX-2.3 prompt modulation only; inert for 2.0
                encoder_attention_mask=conn_mask,
                audio_encoder_attention_mask=conn_mask,
                num_frames=lat_f,
                height=lat_h,
                width=lat_w,
                fps=args.frame_rate,
                audio_num_frames=audio_n,
                video_coords=video_coords,
                audio_coords=audio_coords,
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                # Pipeline default True. With one shared sigma for both streams the
                # "cross" timestep equals the own timestep, so 2.0 weights see the
                # same numbers either way (transformer_ltx2.py:1560, 1576).
                use_cross_timestep=True,
                attention_kwargs=None,
                return_dict=False,
            )
            # Older diffusers releases predate some of these keywords.
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
            try:
                with torch.no_grad(), sdpa_context():
                    v, a = transformer(**kwargs)
            except torch.OutOfMemoryError:
                if current["sdpa"] != "math":
                    raise
                # 76 GB of float32 weights leave little room; the fused kernels are
                # still float32 arithmetic, just not the textbook order.
                print("[oracle] math SDPA ran out of memory in float32; falling back to the default kernel", file=sys.stderr)
                current["sdpa"] = "default (math SDPA OOM)"
                free()
                with torch.no_grad():
                    v, a = transformer(**kwargs)
            # pipeline_ltx2.py:1422-1423 — everything outside the model is float32.
            return v.float(), a.float()

        def strided(t: "torch.Tensor") -> "torch.Tensor":
            # Sub-layer taps of the video stream are [1, 6144, 4096] each — 100 MB — and
            # there are two dozen of them per pass. Every `tap_stride`-th token is as
            # good a sample for a relative error; the audio stream is kept whole.
            return f32(t[:, :: args.tap_stride] if t.shape[1] > 1024 else t)

        def first(args_, kwargs_):
            return args_[0] if args_ else kwargs_["hidden_states"]

        def install_taps(prefix: str, n_blocks: int):
            hooks = []
            taps = sorted({0, n_blocks // 2, n_blocks - 1})
            for i in taps:
                # Each block returns (video [B,S,4096], audio [B,L,2048]) (transformer_ltx2.py:811).
                def tap(_m, _i, o, i=i):
                    out[f"{prefix}.block{i:02d}.video"] = f32(o[0])
                    out[f"{prefix}.block{i:02d}.audio"] = f32(o[1])

                hooks.append(transformer.transformer_blocks[i].register_forward_hook(tap))
            # Inside the first and the last block: for each sub-layer its input (the
            # modulated norm), its output (before the gate) and the residual stream
            # right after it — read as the input of the *next* norm, which is the only
            # place the stream is visible from outside the block's forward().
            streams = {
                "video": [("attn1", "norm2"), ("attn2", "audio_to_video_norm"), ("audio_to_video_attn", "norm3"), ("ff", None)],
                "audio": [("audio_attn1", "audio_norm2"), ("audio_attn2", "video_to_audio_norm"), ("video_to_audio_attn", "audio_norm3"), ("audio_ff", None)],
            }
            short = {"attn1": "attn1", "attn2": "attn2", "audio_to_video_attn": "av", "ff": "ff", "audio_attn1": "attn1", "audio_attn2": "attn2", "video_to_audio_attn": "av", "audio_ff": "ff"}
            for i in sorted({0, n_blocks - 1}):
                block = transformer.transformer_blocks[i]
                for stream, layers in streams.items():
                    for layer, next_norm in layers:
                        base = f"{prefix}.block{i:02d}.{stream}.{short[layer]}"

                        def pre(_m, a, k, base=base):
                            out[f"{base}_in"] = strided(first(a, k))

                        def post(_m, _a, _k, o, base=base):
                            out[f"{base}_out"] = strided(o)

                        hooks.append(getattr(block, layer).register_forward_pre_hook(pre, with_kwargs=True))
                        hooks.append(getattr(block, layer).register_forward_hook(post, with_kwargs=True))
                        if next_norm is not None:

                            def after(_m, a, k, base=base):
                                out[f"{base}_after"] = strided(first(a, k))

                            hooks.append(getattr(block, next_norm).register_forward_pre_hook(after, with_kwargs=True))
            # The output heads in pieces: LayerNorm out, then (1 + scale)·x + shift as
            # proj_out receives it; proj_out's own output is `{prefix}.video_out`.
            for stream, norm, proj in [("video", "norm_out", "proj_out"), ("audio", "audio_norm_out", "audio_proj_out")]:

                def normed(_m, _a, _k, o, stream=stream):
                    out[f"{prefix}.head.{stream}.norm"] = strided(o)

                def modulated(_m, a, k, stream=stream):
                    out[f"{prefix}.head.{stream}.modulated"] = strided(first(a, k) if a or "hidden_states" in k else k["input"])

                hooks.append(getattr(transformer, norm).register_forward_hook(normed, with_kwargs=True))
                hooks.append(getattr(transformer, proj).register_forward_pre_hook(modulated, with_kwargs=True))
            return hooks, taps

        def run_sample(prefix: str):
            """The 8-step distilled loop in the current dtype; tensors `{prefix}.*`."""
            # The distilled scheduler config — no dynamic shift, no terminal stretch — makes
            # set_timesteps(sigmas=…) return the list untouched plus a trailing 0, with
            # timesteps = float32(sigma) * 1000 (scheduling_flow_match_euler_discrete.py:348-377).
            # That is three lines of arithmetic, so it is done here and the scheduler class
            # is only asked to agree: a constructor/keyword change in diffusers then costs a
            # note in the meta file, not the run, and a dev-configured scheduler/ folder
            # cannot silently shift the sigmas either way.
            sigmas = torch.tensor(DISTILLED_SIGMA_VALUES + [0.0], dtype=torch.float32, device=dev)
            timesteps = sigmas[:-1] * 1000.0
            meta["sample_sigmas"] = [float(s) for s in sigmas]
            if "sample_scheduler_check" not in meta:
                try:
                    from diffusers import FlowMatchEulerDiscreteScheduler

                    sched = FlowMatchEulerDiscreteScheduler(
                        num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False, shift_terminal=None
                    )
                    sched.set_timesteps(sigmas=DISTILLED_SIGMA_VALUES, device=dev)
                    agree = bool(torch.equal(sched.sigmas.to(sigmas), sigmas) and torch.equal(sched.timesteps.to(timesteps), timesteps))
                    meta["sample_scheduler_check"] = {"agrees_with_diffusers": agree}
                    if not agree:
                        raise SystemExit(f"[oracle] diffusers scheduler disagrees: sigmas {sched.sigmas.tolist()}, timesteps {sched.timesteps.tolist()}")
                except SystemExit:
                    raise
                except Exception as e:  # API drift in the scheduler must not cost the trajectory
                    meta["sample_scheduler_check"] = {"error": repr(e)}

            # pipeline_ltx2.py:804-807, 844-845 — N(0,1) in float32, packed. The pipeline draws
            # video then audio from one generator; here each has its own saved draw.
            lat = pack_video(cpu_randn((1, 128, lat_f, lat_h, lat_w), 10)).to(dev)
            aud = pack_audio(cpu_randn((1, 8, audio_n, 16), 11)).to(dev)
            out["sample.video_noise"] = f32(lat)
            out["sample.audio_noise"] = f32(aud)
            t0 = time.time()
            for i, t in enumerate(timesteps):
                v, a = dit(lat, aud, t.expand(1))
                # Unguided (guidance_scale=1, stg_scale=0, modality_scale=1, rescale=0), the
                # pipeline still round-trips v → x0 → v in float32 (pipeline_ltx2.py:1466-1467,
                # 1573-1574); kept so the step is bit-comparable with a real pipeline run.
                sigma = sigmas[i]
                v = (lat - (lat - v * sigma)) / sigma
                a = (aud - (aud - a * sigma)) / sigma
                # pipeline_ltx2.py:1577-1580 — scheduler.step is x ← x + (sigma_next - sigma) · v
                # in float32, the same rule and the same sigmas for both streams.
                dt = sigmas[i + 1] - sigma
                lat = lat + dt * v
                aud = aud + dt * a
                out[f"{prefix}.step{i}.video"] = f32(lat)
                out[f"{prefix}.step{i}.audio"] = f32(aud)
            torch.cuda.synchronize()
            timings[f"{prefix}_s"] = time.time() - t0
            return lat, aud

    video_in = audio_in = timestep = None
    for n_pass, pass_dtype in enumerate(passes if transformer is not None else []):
        # The first pass keeps the historical names; a second pass can only be float32.
        dit_prefix, sample_prefix = ("dit", "sample") if n_pass == 0 else ("dit32", "sample32")
        current["dtype"] = getattr(torch, pass_dtype)
        if pass_dtype == "float32":
            # TF32 would make "float32" a 19-bit format on exactly the matmuls that matter.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            current["sdpa"] = "math"
            if n_pass > 0:
                transformer.to(torch.float32)
                free()
        meta.setdefault("dit_passes", {})[dit_prefix] = {"dtype": pass_dtype, "tf32": pass_dtype != "float32" and bool(torch.backends.cuda.matmul.allow_tf32)}

        # --- dit: one forward ------------------------------------------------
        if "dit" not in skip:
            if n_pass == 0:
                with torch.no_grad():
                    # The four RoPE tables, exactly as forward() builds them
                    # (transformer_ltx2.py:1505-1511): self-attn tables are [1,32,S,64]
                    # (video, 3 axes) and [1,32,L,32] (audio); the a↔v cross tables are
                    # time-only, [1,32,S,32] and [1,32,L,32]. They are float32 whatever
                    # the module dtype is.
                    for name, (cos, sin) in {
                        "video": transformer.rope(video_coords, device=dev),
                        "audio": transformer.audio_rope(audio_coords, device=dev),
                        "cross_video": transformer.cross_attn_rope(video_coords[:, 0:1, :], device=dev),
                        "cross_audio": transformer.cross_attn_audio_rope(audio_coords[:, 0:1, :], device=dev),
                    }.items():
                        out[f"dit.rope.{name}.cos"] = f32(cos)
                        out[f"dit.rope.{name}.sin"] = f32(sin)
                        meta.setdefault("rope_dtypes", {})[name] = str(cos.dtype)
                out["dit.video_coords"] = f32(video_coords)
                out["dit.audio_coords"] = f32(audio_coords)

                video_in = pack_video(cpu_randn((1, 128, lat_f, lat_h, lat_w), 0))  # [1, S, 128]
                audio_in = pack_audio(cpu_randn((1, 8, audio_n, 16), 1))  # [1, L, 128]
                # The scheduler's timesteps are float32(sigma) * 1000
                # (scheduling_flow_match_euler_discrete.py:366-367); (B,) like pipeline :1396.
                timestep = (torch.tensor([args.sigma], dtype=torch.float32) * 1000.0).to(dev)
                out["dit.video_in"] = video_in
                out["dit.audio_in"] = audio_in
                out["dit.timestep"] = timestep.cpu()
                out["dit.tap_stride"] = torch.tensor([float(args.tap_stride)])

            hooks, taps = install_taps(dit_prefix, len(transformer.transformer_blocks))
            t0 = time.time()
            v, a = dit(video_in, audio_in, timestep)
            torch.cuda.synchronize()
            timings[f"{dit_prefix}_forward_s"] = time.time() - t0
            for h in hooks:
                h.remove()
            out[f"{dit_prefix}.video_out"] = f32(v)  # velocity, [1, S, 128]
            out[f"{dit_prefix}.audio_out"] = f32(a)  # velocity, [1, L, 128]
            meta["dit_taps"] = taps
            meta["dit_passes"][dit_prefix]["sdpa"] = current["sdpa"]

        # --- sample: the full distilled loop ----------------------------------
        if args.sample:
            lat, aud = run_sample(sample_prefix)
            if n_pass == 0:
                # What gets decoded below is the product's trajectory.
                sample_video, sample_audio = lat, aud

    if transformer is not None and len(passes) == 2 and "dit" not in skip:
        # The reference's own noise floor, tap by tap: rel-L2 of bf16 against float32.
        floor = {}
        for name in sorted(k for k in out if k.startswith("dit32.")):
            a, b = out["dit." + name[len("dit32.") :]], out[name]
            floor[name[len("dit32.") :]] = float((a - b).norm() / b.norm().clamp_min(1e-30))
        for name in sorted(k for k in out if k.startswith("sample32.step")):
            a, b = out["sample." + name[len("sample32.") :]], out[name]
            floor["sample." + name[len("sample32.") :]] = float((a - b).norm() / b.norm().clamp_min(1e-30))
        meta["dit_bf16_vs_f32"] = floor

    if transformer is not None:
        # Every closure over the module has to go, or its 38-76 GB stay resident
        # through the VAE decodes.
        del transformer, dit, install_taps, run_sample, sdpa_context
        free()

    # --- vae: AutoencoderKLLTX2Video decode ---------------------------------
    if "vae" not in skip or sample_video is not None:
        from diffusers import AutoencoderKLLTX2Video

        vae = load(AutoencoderKLLTX2Video, "vae", torch.float32, "diffusers")
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).float()
        std = vae.latents_std.view(1, -1, 1, 1, 1).float()
        out["vae.latents_mean"] = f32(vae.latents_mean)
        out["vae.latents_std"] = f32(vae.latents_std)

        def decode(z_norm: "torch.Tensor") -> "torch.Tensor":
            # pipeline_ltx2.py:1638-1643 — z·std/scaling_factor + mean, then decode with
            # timestep=None (config.timestep_conditioning is false, :1621-1622) and
            # causal left to the config default (decoder_causal=false).
            z = z_norm.to(dev) * std / vae.config.scaling_factor + mean
            with torch.no_grad():
                return vae.decode(z, None, return_dict=False)[0]

        if "vae" not in skip:
            f, h, w = (int(x) for x in args.vae_latent.split(","))
            z = cpu_randn((1, 128, f, h, w), 20)  # DiT-space (normalised) latent
            t0 = time.time()
            video = decode(z)
            torch.cuda.synchronize()
            timings["vae_decode_s"] = time.time() - t0
            out["vae.latent"] = z
            out["vae.video"] = f32(video)  # [1, 3, 8(f-1)+1, 32h, 32w], nominally [-1, 1]
        if sample_video is not None:
            # pipeline_ltx2.py:1598-1605 — unpack [1,S,128] → [1,128,F,H,W].
            z = sample_video.float().cpu().reshape(1, lat_f, lat_h, lat_w, 128).permute(0, 4, 1, 2, 3)
            video = decode(z)
            keep = sorted({0, video.shape[2] // 2, video.shape[2] - 1})
            out["sample.frames"] = f32(video[:, :, keep])  # 3 of 121 frames; the rest is 570 MB
            meta["sample_frames_kept"] = keep
        del vae, decode
        free()

    # --- audio: AutoencoderKLLTX2Audio decode + LTX2Vocoder -----------------
    if "audio" not in skip or sample_audio is not None:
        from diffusers import AutoencoderKLLTX2Audio
        from diffusers.pipelines.ltx2 import LTX2Vocoder

        audio_vae = load(AutoencoderKLLTX2Audio, "audio_vae", torch.float32, "diffusers")
        vocoder = load(LTX2Vocoder, "vocoder", torch.float32, "diffusers")
        out["audio.latents_mean"] = f32(audio_vae.latents_mean)  # [128] = 8 channels × 16 bins
        out["audio.latents_std"] = f32(audio_vae.latents_std)
        meta["vocoder_sample_rate"] = int(vocoder.config.output_sampling_rate)

        def decode_audio(packed_norm: "torch.Tensor"):
            # pipeline_ltx2.py:1607-1610 — de-normalise while still packed [B,L,128] (the
            # statistics are per packed feature), then unpack to [B,8,L,16].
            z = packed_norm.to(dev) * audio_vae.latents_std.float() + audio_vae.latents_mean.float()
            z = z.unflatten(2, (-1, 16)).transpose(1, 2)
            with torch.no_grad():
                # pipeline_ltx2.py:1647-1648 — mel [B,2,4L-3,64] → waveform [B,2,240·(4L-3)]
                # at vocoder.config.output_sampling_rate (24 kHz; the mel is 16 kHz/hop 160).
                mel = audio_vae.decode(z, return_dict=False)[0]
                return mel, vocoder(mel)

        if "audio" not in skip:
            packed = pack_audio(cpu_randn((1, 8, args.audio_latent_frames, 16), 30))
            t0 = time.time()
            mel, wave = decode_audio(packed)
            torch.cuda.synchronize()
            timings["audio_decode_s"] = time.time() - t0
            out["audio.latent"] = packed
            out["audio.mel"] = f32(mel)
            out["audio.wave"] = f32(wave)
        if sample_audio is not None:
            mel, wave = decode_audio(sample_audio.float())
            out["sample.mel"] = f32(mel)
            out["sample.wave"] = f32(wave)
        del audio_vae, vocoder, decode_audio
        free()

    meta["shapes"] = {k: list(v.shape) for k, v in out.items()}
    meta["stats"] = {
        k: {"mean": float(v.float().mean()), "std": float(v.float().std()), "absmax": float(v.float().abs().max())}
        for k, v in out.items()
        if v.numel() > 1
    }
    save_file({k: v.contiguous() for k, v in out.items()}, args.out)
    json.dump(meta, open(args.meta, "w"), indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in meta.items() if k != "stats"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
