#!/usr/bin/env python3
"""An external oracle for FastH3: text encoder, one DiT forward, the 8-step loop, both VAE decoders.

Same contract as `upstream_oracle.py`: run the *reference* implementations
(transformers' Qwen3-VL, diffusers' MiniMax-H3 classes) on the same weights,
with every random input drawn from a seeded CPU generator and saved next to the
outputs, so `fv-gpucheck` diffs our stack on byte-identical inputs and each
stage's error is attributed rather than merely detected.

Stages, each freed before the next so the whole script fits one 96 GB card:

    1 text   Qwen3-VL-32B in bf16 (66.7 GB)        -> text_ids, text, text_h0, text_h1
             and, with --llm-out, the file `fv-gpucheck llm --family qwen3-vl-32b` reads
    2 dit    MiniMaxH3Transformer3DModel in bf16    -> dit_*, temb, adaln_block0, text_refined, block_*
             (66.2 GB resident: diffusers ignores the 3.85 GB of to_gate_compress)
    3 loop   the 8-forward FastH3 ladder, dense      -> loop_video, loop_audio
    4 vae    AutoencoderKLMiniMaxH3, float32        -> vae_latent, vae_video_raw
    5 audio  AutoencoderKLMiniMaxH3Audio, float32   -> audio_latent, audio_wave

What this oracle is NOT, and the port must not pretend otherwise:

  * diffusers has no VSA-H3 backend and no `to_gate_compress`. Stages 2 and 3 are
    **dense attention with the compression-gate branch absent**. The checkpoint
    was distilled *with* VSA at sparsity 0.8 and ships trained gates, so these
    are the reference for our dense mode (`H3_VSA=off`), which validates every
    weight, the packing, the RoPE, the AdaLN table and the scheduler. VSA-H3
    itself has to be judged by (a) our VSA at sparsity 0 without the gate == our
    dense, and (b) FastVideo's own pipeline; see docs/ports/h3.md section j.
  * The DiT runs in bfloat16 (its float32 form is 140 GB). Everything is *saved*
    as float32, which is lossless, but stage 2/3 limits have to allow for bf16
    accumulation over 50 blocks. The text and VAE stages are the precision the
    reference ships: bf16 for Qwen3-VL, float32 for both VAEs.

Every tensor is written as float32, including ids, tags and indices (exact: all
are far below 2^24): `fastvideo-gpucheck`'s safetensors reader accepts nothing
else. `position_ids` are built in float64 upstream and cast to float32 as the
first thing the rope does (FastVideo minimax_h3.py:85), so float32 is what the
model consumes.

Nothing here is run on the dev machine; every non-obvious line cites the
upstream line it mirrors (diffusers `main`, FastVideo `main`, 2026-09-19).
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time

REPO = "FastVideo/FastVideo-FastH3-8-Step-V2"

# fastvideo_inference.json of the checkpoint; the diffusers scheduler configs
# carry only the two shifts, so the ladder has to come from here.
DMD_RUNGS = [999, 874, 749, 624, 500, 375, 250, 125]

# diffusers modular_pipeline.py:24-26 / FastVideo packing.py:17-19.
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
AUDIO_CHANNELS = 2  # modular_pipeline.py:39
TEXT_ENCODER_LAYER = 50  # encoders.py:35, packing.py:31 -> hidden_states[50]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=REPO, help="Hub id or a local snapshot dir of the FastH3 repo")
    ap.add_argument("--prompts", required=True, help="prompt JSON, same file the embed stage reads")
    ap.add_argument("--name", default=None, help="which prompt to use (default: the first)")
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--num-frames", type=int, default=124, help="17n+5; 124 = 5 s. 56 is a cheap 17-latent variant")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--dit-step", type=int, default=0, help="ladder index whose timesteps the single forward uses")
    ap.add_argument("--dump-blocks", default="0,24,49", help="transformer blocks whose output is dumped, row-strided")
    ap.add_argument("--block-row-stride", type=int, default=16, help="keep every k-th packed row of a block dump")
    ap.add_argument("--vae-latent", default="12,32,48", help="T,H,W of the fixed video latent (2 chunks, 3x4 tiles)")
    ap.add_argument("--audio-latents", type=int, default=50, help="length of the fixed audio latent, per channel")
    ap.add_argument("--stages", default="text,dit,loop,vae,audio")
    ap.add_argument("--device-map", default="cuda", help="passed to from_pretrained so shards stream to the GPU")
    ap.add_argument("--llm-out", default=None, help="second file, in the `fv-gpucheck llm` oracle format")
    ap.add_argument("--llm-taps", default="1,8,50", help="output_hidden_states indices written to --llm-out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    import torch
    import diffusers
    import transformers
    from safetensors.torch import save_file

    spec = json.load(open(args.prompts))
    entry = spec["prompts"][0] if args.name is None else next(p for p in spec["prompts"] if p["name"] == args.name)
    prompt = entry["prompt"]

    meta: dict[str, object] = {
        "prompt": prompt,
        "name": entry["name"],
        "weights": args.weights,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "spec": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "seed": args.seed,
            "dit_step": args.dit_step,
            "dmd_rungs": DMD_RUNGS,
            "block_row_stride": args.block_row_stride,
            "attention": "dense (diffusers native SDPA); no VSA, no to_gate_compress",
            "dit_dtype": "bfloat16 with proj_in/audio_proj_in/time_embedder/proj_out/audio_proj_out in float32",
        },
    }
    # A "float32 reference" is only float32 if the library is told so: PyTorch
    # ships with TF32 enabled for cuDNN convolutions (10-bit mantissa inside a
    # float32 API), and the audio decoder is nothing but ~130 convolutions.
    # Pin real float32 everywhere; the bf16 stages are unaffected.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    meta["precision"] = {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_version": torch.backends.cudnn.version(),
        "vae_float32_sdpa": "math backend (no fused kernel) for the float32 decode",
    }

    def math_sdpa():
        """Context that forces the unfused SDPA for a float32 stage; a no-op on a torch without the API."""
        import contextlib

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(SDPBackend.MATH)
        except ImportError:
            meta["precision"]["vae_float32_sdpa"] = "torch.nn.attention.sdpa_kernel unavailable; default kernel"
            return contextlib.nullcontext()

    out: dict[str, "torch.Tensor"] = {}
    dev = "cuda"

    def release() -> None:
        # The caller `del`s its own names first; a helper cannot drop them for it.
        gc.collect()
        torch.cuda.empty_cache()

    # --- tolerance for two API generations --------------------------------------
    # transformers 5 / diffusers 0.41 renamed `torch_dtype=` to `dtype=`. Both
    # loaders swallow unknown kwargs in some releases, and a silently ignored
    # dtype means a 133 GB float32 load, so: pick the spelling the installed
    # `from_pretrained` actually pops, and verify the result.
    def load_pretrained(cls, *pos, dtype, check: str | None = None, **kw):
        import re

        try:
            src = inspect.getsource(cls.from_pretrained)
        except (OSError, TypeError):
            src = ""
        names = ["dtype", "torch_dtype"] if re.search(r"""pop\(\s*["']dtype["']""", src) else ["torch_dtype", "dtype"]
        last: Exception | None = None
        for name in names:
            try:
                model = cls.from_pretrained(*pos, **{name: dtype}, **kw)
            except (TypeError, torch.cuda.OutOfMemoryError) as e:
                # TypeError: "unexpected keyword argument". OOM: the kwarg was
                # swallowed and the model came up in float32 (twice the size).
                last = e
                release()
                continue
            # The widest matrix decides: norms and `_keep_in_fp32_modules` may legitimately differ.
            params = [p for n, p in model.named_parameters() if check is None or check in n]
            got = max(params, key=lambda p: p.numel()).dtype
            if got == dtype:
                meta.setdefault("dtype_kwarg", {})[cls.__name__] = name
                return model
            last = RuntimeError(f"{cls.__name__}.from_pretrained({name}={dtype}) produced {got}")
            del model
            release()
        raise RuntimeError(f"could not load {cls.__name__} as {dtype}: {last}")

    def resolve(module_names: list[str], attr: str):
        """`attr` from the first module that has it: top-level export first, defining module as a fallback."""
        import importlib

        errors = []
        for name in module_names:
            try:
                return getattr(importlib.import_module(name), attr)
            except (ImportError, AttributeError) as e:
                errors.append(f"{name}: {e}")
        raise ImportError(f"{attr} not found; tried {errors}. diffusers {diffusers.__version__} may predate MiniMax-H3.")

    # --- 1. text encoder ----------------------------------------------------
    # diffusers encoders.py:192 and FastVideo minimax_h3_conditioning.py:187 agree:
    # the prompt verbatim, NO chat template, NO special tokens, no padding, no
    # truncation. One sequence, so the attention mask is all ones.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.weights, subfolder="tokenizer")
    # tokenizer_config.json lists seven `additional_special_tokens` (`<d>`, `</d>`,
    # `<|cutoff|>`, `<|lyrics_*|>`, `<|caption_*|>`) that tokenizer.json does not
    # define. transformers 4 appends them to the vocabulary at load time, which is
    # what FastVideo trained and serves with; H3 prompts wrap dialogue in
    # `<d>...</d>`. Read the list from the config file rather than from the
    # tokenizer object: transformers 5 removed `additional_special_tokens` from
    # the fast tokenizer classes, and the config file is what a port reads anyway.
    import os

    if os.path.isdir(args.weights):
        tok_cfg_path = os.path.join(args.weights, "tokenizer", "tokenizer_config.json")
    else:
        from huggingface_hub import hf_hub_download

        tok_cfg_path = hf_hub_download(args.weights, "tokenizer_config.json", subfolder="tokenizer")
    with open(tok_cfg_path) as fh:
        tok_cfg = json.load(fh)
    listed = tok_cfg.get("additional_special_tokens") or []
    if not listed and isinstance(tok_cfg.get("extra_special_tokens"), (dict, list)):
        extra = tok_cfg["extra_special_tokens"]
        listed = list(extra.values()) if isinstance(extra, dict) else list(extra)
    listed = [t if isinstance(t, str) else t.get("content", "") for t in listed]
    markers = [t for t in listed if t.startswith(("<d>", "</d>", "<|cutoff", "<|lyrics", "<|caption"))]

    def marker_id(tok: str):
        # None, or the unk id, means "not in the vocabulary" depending on the release.
        i = tokenizer.convert_tokens_to_ids(tok)
        return None if i is None or i == getattr(tokenizer, "unk_token_id", None) else int(i)

    # A loader that did NOT register the markers (a transformers release that
    # ignores `additional_special_tokens` in the config) would split `<d>` into
    # pieces. Register them the way transformers 4 does - in the listed order, at
    # the first free ids - so the reference tokenization is the trained one on
    # every release, and say which path produced the ids.
    missing = [t for t in markers if marker_id(t) is None]
    meta["markers_registered_by_loader"] = [t for t in markers if t not in missing]
    meta["markers_registered_by_oracle"] = missing
    if missing:
        from tokenizers import AddedToken

        tokenizer.add_tokens([AddedToken(t, special=True, normalized=False) for t in missing], special_tokens=True)
    meta["added_special_token_ids"] = {tok: marker_id(tok) for tok in markers}
    assert all(v is not None for v in meta["added_special_token_ids"].values()), meta["added_special_token_ids"]
    meta["tokenizer_len"] = len(tokenizer)
    meta["transformers_version"] = transformers.__version__

    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    meta["tokens"] = len(token_ids)
    out["text_ids"] = torch.tensor(token_ids, dtype=torch.float32)  # ids < 151936 < 2^24: exact

    if "text" in stages:
        from transformers import Qwen3VLForConditionalGeneration

        t0 = time.time()
        text_encoder = load_pretrained(
            Qwen3VLForConditionalGeneration,
            args.weights,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            check="language_model",
            device_map=args.device_map,
        ).eval()
        meta["load_text_s"] = time.time() - t0

        # encoders.py:63-70: a stack truncated to exactly 50 layers would return a
        # post-norm last state, which is not hidden_states[50].
        assert text_encoder.config.text_config.num_hidden_layers > TEXT_ENCODER_LAYER

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=dev)
        kwargs = {}
        # encoders.py:73,94: mm_token_type_ids is 0 for every text token. Older
        # transformers releases have no such argument; for text-only ids the
        # rotary layout is the same 1-D arange either way.
        if "mm_token_type_ids" in inspect.signature(text_encoder.model.forward).parameters:
            kwargs["mm_token_type_ids"] = torch.zeros_like(input_ids)
        # `output_hidden_states` is plumbed differently in transformers 4 (explicit
        # tuples) and 5 (output-recording hooks on the inner text model), and the
        # outer Qwen3VLModel has at times dropped the field. Forward hooks on the
        # embedding and on decoder layer k-1 capture hidden_states[k] by
        # construction on any release; they back the tuple up and cross-check it.
        llm_taps = [int(k) for k in args.llm_taps.split(",") if k != ""] if args.llm_out else []
        wanted_taps = sorted({0, 1, TEXT_ENCODER_LAYER, *llm_taps})
        language_model = getattr(text_encoder.model, "language_model", None) or text_encoder.model
        num_layers = len(language_model.layers)
        assert max(wanted_taps) < num_layers, "the last tap is post-norm in HF's tuple; hooks cannot see that"
        hooked: dict[int, "torch.Tensor"] = {}

        def grab(k: int):
            def hook(_module, _inputs, output):
                hooked[k] = (output[0] if isinstance(output, (tuple, list)) else output).detach()

            return hook

        hook_handles = [
            (language_model.embed_tokens if k == 0 else language_model.layers[k - 1]).register_forward_hook(grab(k))
            for k in wanted_taps
        ]
        with torch.no_grad():
            t0 = time.time()
            # encoders.py:91-98: `.model`, not the top-level module, so the
            # vocabulary projection never runs; causal LM attention throughout.
            outputs = text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_hidden_states=True,
                **kwargs,
            )
            torch.cuda.synchronize()
            meta["encode_s"] = time.time() - t0
        for h in hook_handles:
            h.remove()
        hs = getattr(outputs, "hidden_states", None)
        if hs is not None and len(hs) == num_layers + 1:
            meta["hidden_states_source"] = "output_hidden_states tuple"
            meta["hook_vs_tuple_max_abs"] = {
                str(k): float((hooked[k].float() - hs[k].float()).abs().max()) for k in wanted_taps
            }
        else:
            meta["hidden_states_source"] = "forward hooks (the model returned no usable hidden_states tuple)"
            hs = hooked  # indexed by tap, like the tuple
        # HF convention: hidden_states[0] is the embedding output and
        # hidden_states[i] the residual stream after decoder layer i-1. Index 50
        # is therefore after layer **49** and has not been through `norm`
        # (FastVideo minimax_h3_qwen3_vl.py:335 returns at layer_index + 1 == 50).
        meta["num_hidden_states"] = len(hs)
        prompt_embeds = hs[TEXT_ENCODER_LAYER]  # [1, N, 5120] bf16, encoders.py:99
        out["text"] = prompt_embeds.float().cpu().contiguous()
        out["text_h0"] = hs[0].float().cpu().contiguous()  # token embeddings: isolates the table gather
        out["text_h1"] = hs[1].float().cpu().contiguous()  # after layer 0: isolates one GQA/RoPE/SwiGLU layer

        if args.llm_out:
            # crates/fastvideo-gpucheck/src/llm_oracle.rs. `hidden_<k>` is
            # output_hidden_states[k]: k = 0 the embeddings, k the stream after k
            # decoder layers (zero-based layer k-1), k = 64 the only normed entry.
            # H3 consumes k = 50. For a text-only prompt the three mrope axes
            # share the 1-D positions 0..S-1 (minimax_h3_qwen3_vl.py:584-587),
            # and with one unpadded sequence every token may be attended.
            n_tok = len(token_ids)
            llm = {
                "input_ids": torch.tensor(token_ids, dtype=torch.float32),
                "positions": torch.arange(n_tok, dtype=torch.float32),
                "attend": torch.ones(n_tok, dtype=torch.float32),
            }
            taps = llm_taps
            assert TEXT_ENCODER_LAYER in taps, "the tap H3 conditions on must be among --llm-taps"
            for k in taps:
                llm[f"hidden_{k}"] = hs[k].float().cpu().contiguous()  # [1, S, 5120]
            save_file(llm, args.llm_out)
            meta["llm_oracle"] = {
                "file": args.llm_out,
                "taps": taps,
                "consumed_tap": TEXT_ENCODER_LAYER,
                "reference_dtype": "bfloat16 on GPU (float32 is 133 GB and does not fit); saved as float32",
            }
        prompt_embeds = prompt_embeds.detach().clone()
        del text_encoder, outputs, hs
        release()
    else:
        # Later stages still need an embedding; a seeded stand-in keeps them runnable alone.
        g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        prompt_embeds = torch.randn((1, len(token_ids), 5120), generator=g, dtype=torch.float32).to(dev, torch.bfloat16)
        out["text"] = prompt_embeds.float().cpu().contiguous()
        meta["text_is_synthetic"] = True

    # --- geometry and the packed layout --------------------------------------
    # Uses diffusers' own builders rather than a transcription of them; FastVideo
    # packing.py:227-298 is line-for-line the same layout.
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )

    assert args.num_frames % 17 == 5, "num_frames must be 17n + 5"
    lat_t = (args.num_frames - 5) // 17 * 5 + 2  # packing.py:128-131
    lat_h, lat_w = args.height // 16, args.width // 16
    n_audio = int(round(args.num_frames / 24 * 40))  # packing.py:134-135
    patch = (1, 2, 2)
    text_tags = torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long)  # encoders.py:205
    (position_ids, token_tags, video_indices, audio_indices, text_indices, n_cond_v, n_cond_a) = (
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_tags, lat_t, lat_h, lat_w, n_audio, patch, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG, ()
        )
    )
    assert n_cond_v == 0 and n_cond_a == 0  # T2AV has no keyframe or reference rows
    seq = position_ids.shape[0]
    meta["latent_shape"] = [1, 24, lat_t, lat_h, lat_w]
    meta["audio_latents_per_channel"] = n_audio
    meta["sequence_length"] = seq
    out["position_ids"] = position_ids.float().contiguous()  # [S, 3] (t, h, w); float64 upstream, float32 at the rope
    out["token_tags"] = token_tags.float().contiguous()  # 0 video, 1 text, 2 audio

    # One request generator, video noise first, then the audio rows
    # (before_denoise.py:847-873, minimax_h3_latent_preparation.py:321-343). Drawn
    # on the CPU, which is also what randn_tensor does with a CPU generator.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    video_noise = torch.randn((1, 24, lat_t, lat_h, lat_w), generator=g, dtype=torch.float32)
    audio_noise = torch.randn((n_audio * AUDIO_CHANNELS, 32), generator=g, dtype=torch.float32)
    out["video_noise"] = video_noise  # un-patchified, so our patchify is under test too
    out["audio_noise"] = audio_noise  # rows: channel 0 latents 0..n-1, then channel 1

    # --- the two schedules -----------------------------------------------------
    # FastVideo minimax_h3_denoising.py:81-87. diffusers' own set_timesteps(9)
    # would build linspace(1, 0, 9), whose first four sigmas are NOT the trained
    # rungs (1.0 vs 0.999, ...), so the explicit-sigmas entry point is used.
    MiniMaxH3Scheduler = resolve(["diffusers", "diffusers.schedulers.scheduling_minimax_h3"], "MiniMaxH3Scheduler")

    scheduler = MiniMaxH3Scheduler.from_pretrained(args.weights, subfolder="scheduler")
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(args.weights, subfolder="audio_scheduler")
    assert scheduler.shift == 10.0 and audio_scheduler.shift == 3.0, (scheduler.shift, audio_scheduler.shift)
    base = torch.tensor([s / 1000.0 for s in DMD_RUNGS] + [0.0], dtype=torch.float32)
    for sch in (scheduler, audio_scheduler):
        shift = float(sch.shift)
        sch.set_timesteps(sigmas=shift * base / (1 + (shift - 1) * base), device=dev)
    out["video_sigmas"] = scheduler.sigmas.float().cpu()
    out["audio_sigmas"] = audio_scheduler.sigmas.float().cpu()
    out["video_timesteps"] = scheduler.timesteps.float().cpu()  # t = 1 - sigma, what the DiT is conditioned on
    out["audio_timesteps"] = audio_scheduler.timesteps.float().cpu()

    def row_plan(i: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        v, a = float(scheduler.timesteps[i].item()), float(audio_scheduler.timesteps[i].item())
        # minimax_h3_denoising.py:136-142; the two condition timesteps address no rows here.
        unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices, audio_indices, 0, 0, len(token_ids), v, a, max(v, 0.999), 1.0
        )
        return unique.to(dev), inverse.to(dev)

    # --- 2. one DiT forward ------------------------------------------------------
    transformer = None
    if stages & {"dit", "loop"}:
        MiniMaxH3Transformer3DModel = resolve(
            ["diffusers", "diffusers.models.transformers.transformer_minimax_h3"], "MiniMaxH3Transformer3DModel"
        )

        t0 = time.time()
        # bf16 request + _keep_in_fp32_modules (transformer_minimax_h3.py:444-451)
        # reproduces FastVideo's precision split (minimax_h3.py:583-607). The
        # checkpoint's 50 `attn.to_gate_compress.weight` tensors have no module
        # here and are reported as unexpected keys, not loaded.
        transformer = load_pretrained(
            MiniMaxH3Transformer3DModel,
            args.weights,
            subfolder="transformer",
            dtype=torch.bfloat16,
            check="transformer_blocks",  # proj_in/out and the time embedder are pinned float32 on purpose
            device_map=args.device_map,
        ).eval()
        # The precision split the limits assume; a loader that drops `_keep_in_fp32_modules` would change them.
        meta["dit_param_dtypes"] = {
            n: str(next(getattr(transformer, n).parameters()).dtype)
            for n in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "context_embedder")
        }
        meta["load_dit_s"] = time.time() - t0

    layout = {
        "token_tags": token_tags.to(dev),
        "position_ids": position_ids.to(dev),
        "video_indices": video_indices.to(dev),
        "audio_indices": audio_indices.to(dev),
        "text_indices": text_indices.to(dev),
    }
    video_rows = patchify_video_latents(video_noise, patch).to(dev)  # [Nv, 96], channel-major patch features
    audio_rows = audio_noise.to(dev)

    def forward(v_rows: "torch.Tensor", a_rows: "torch.Tensor", i: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        unique, inverse = row_plan(i)
        # denoise.py:121-130: batch axis added here, latents stay float32, the
        # prompt embedding stays in the conditioner's bf16.
        return transformer(
            hidden_states=v_rows[None],
            audio_hidden_states=a_rows[None],
            encoder_hidden_states=prompt_embeds.to(dev),
            timestep=unique,
            timestep_indices=inverse,
            return_dict=False,
            **layout,
        )

    if "dit" in stages:
        stride = args.block_row_stride
        hooks = []

        def keep(name: str, rows: bool = False):
            def hook(_module, _inputs, output):
                # adaln_proj returns its six chunks as a tuple; everything else hooked here is one tensor.
                t = torch.stack(output) if isinstance(output, tuple) else output
                t = t[:, ::stride] if rows else t  # block outputs are [1, S, 5376]
                out[name] = t.detach().float().cpu().contiguous()

            return hook

        # [n_t, 2688] float32: the one input every AdaLN projection shares.
        hooks.append(transformer.time_embedder.register_forward_hook(keep("temb")))
        # [6, n_t*3, 5376]: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
        # rows [t0_video, t0_text, t0_audio, t1_video, ...]. The precomputed table is checked against this.
        hooks.append(transformer.transformer_blocks[0].adaln_proj.register_forward_hook(keep("adaln_block0")))
        hooks.append(transformer.token_refiner.register_forward_hook(keep("text_refined")))  # [1, N, 5376]
        for b in [int(x) for x in args.dump_blocks.split(",") if x != ""]:
            hooks.append(transformer.transformer_blocks[b].register_forward_hook(keep(f"block_{b}", rows=True)))

        torch.cuda.reset_peak_memory_stats()  # otherwise this reports the text encoder's peak
        unique, inverse = row_plan(args.dit_step)
        out["dit_timesteps"] = unique.float().cpu()
        out["dit_timestep_indices"] = inverse.float().cpu()  # per row, index into dit_timesteps
        with torch.no_grad():
            t0 = time.time()
            v_out, a_out = forward(video_rows, audio_rows, args.dit_step)
            torch.cuda.synchronize()
            meta["dit_forward_s"] = time.time() - t0
        for h in hooks:
            h.remove()
        out["dit_video"] = v_out.float().cpu().contiguous()  # [1, Nv, 96] data-ward velocity
        out["dit_audio"] = a_out.float().cpu().contiguous()  # [1, 2*Na, 32]
        meta["dit_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30

    # --- 3. the 8-forward loop ---------------------------------------------------
    if "loop" in stages:
        v_rows, a_rows = video_rows.clone(), audio_rows.clone()
        per_step_v, per_step_a = [], []
        with torch.no_grad():
            t0 = time.time()
            for i, (tv, ta) in enumerate(zip(scheduler.timesteps, audio_scheduler.timesteps)):
                v_out, a_out = forward(v_rows, a_rows, i)
                # denoise.py:225-237 / minimax_h3_denoising.py:226-237: float32
                # velocity, the *timestep* (not the index) selects the step.
                v_rows = scheduler.step(v_out[0].float(), tv, v_rows, return_dict=False)[0]
                a_rows = audio_scheduler.step(a_out[0].float(), ta, a_rows, return_dict=False)[0]
                per_step_v.append(v_rows.float().cpu())
                per_step_a.append(a_rows.float().cpu())
            torch.cuda.synchronize()
            meta["loop_s"] = time.time() - t0
        out["loop_video"] = torch.stack(per_step_v).contiguous()  # [8, Nv, 96]; [-1] is the clean latent rows
        out["loop_audio"] = torch.stack(per_step_a).contiguous()  # [8, 2*Na, 32]

    if transformer is not None:
        del transformer
        release()

    # --- 4. video VAE decode -------------------------------------------------------
    if "vae" in stages:
        AutoencoderKLMiniMaxH3 = resolve(
            ["diffusers", "diffusers.models.autoencoders.autoencoder_kl_minimax_h3"], "AutoencoderKLMiniMaxH3"
        )

        t0 = time.time()
        # float32 weights on disk and pinned float32 by _keep_in_fp32_modules
        # (autoencoder_kl_minimax_h3.py:529-532). Tiling is on by default and is
        # part of the released output (:520-523, :608-615); leave it on.
        vae = load_pretrained(AutoencoderKLMiniMaxH3, args.weights, subfolder="vae", dtype=torch.float32, check="decoder")
        vae = vae.eval().to(dev)
        meta["load_vae_s"] = time.time() - t0
        assert vae.use_tiling and vae.tile_sample_min_height == 256 and vae.tile_sample_min_overlap_height == 64

        vt, vh, vw = (int(x) for x in args.vae_latent.split(","))
        g = torch.Generator(device="cpu").manual_seed(args.seed + 2)
        latent = torch.randn((1, 24, vt, vh, vw), generator=g, dtype=torch.float32)
        out["vae_latent"] = latent  # DiT-space (normalized); the port applies mean/std itself
        mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
        z = latent.to(dev) * std + mean  # decoders.py:183-185
        with torch.no_grad():
            t0 = time.time()
            with math_sdpa():
                raw = vae.decode(z, return_dict=False)[0]  # float32 end to end: the limit-setting reference
            torch.cuda.synchronize()
            meta["vae_decode_s"] = time.time() - t0
            # decoders.py:187-188: the shipped recipe is float16 *autocast over float32 weights*.
            # Measured here so the port's limit can be set against a known float16 cost.
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                raw16 = vae.decode(z, return_dict=False)[0].float()
        # ImageNet-normalized RGB, *before* `raw * pixel_std + pixel_mean` and the
        # clamp to [0, 1] (decoders.py:189-191): a clamp would hide errors.
        out["vae_video_raw"] = raw.float().cpu().contiguous()  # [1, 3, F, 16*vh, 16*vw]
        meta["vae_video_shape"] = list(raw.shape)
        meta["vae_fp16_autocast_vs_fp32"] = {
            "max_abs": float((raw16 - raw).abs().max()),
            "mean_abs": float((raw16 - raw).abs().mean()),
        }
        meta["vae_postprocess"] = "rgb01 = clamp(raw * [0.229,0.224,0.225] + [0.485,0.456,0.406], 0, 1)"
        del vae, raw, raw16
        release()

    # --- 5. audio VAE decode --------------------------------------------------------
    if "audio" in stages:
        AutoencoderKLMiniMaxH3Audio = resolve(
            ["diffusers", "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio"], "AutoencoderKLMiniMaxH3Audio"
        )

        # float32: the DAC/BigVGAN stack loses ~20 dB under bfloat16
        # (autoencoder_kl_minimax_h3_audio.py:531-534).
        audio_vae = load_pretrained(
            AutoencoderKLMiniMaxH3Audio, args.weights, subfolder="audio_vae", dtype=torch.float32, check="decoder"
        )
        audio_vae = audio_vae.eval().to(dev)
        g = torch.Generator(device="cpu").manual_seed(args.seed + 3)
        # Stereo is two mono batch items through the same weights (:26-27).
        a_latent = torch.randn((AUDIO_CHANNELS, 32, args.audio_latents), generator=g, dtype=torch.float32)
        out["audio_latent"] = a_latent
        a_mean = torch.tensor(audio_vae.config.latents_mean, device=dev, dtype=torch.float32).view(1, -1, 1)
        a_std = torch.tensor(audio_vae.config.latents_std, device=dev, dtype=torch.float32).view(1, -1, 1)
        # Where an error enters matters more than that it exists: dump the stream
        # after conv_pre, after every transposed conv (`up_i`) and after every
        # averaged AMP stage (`stage_i`; stage_6 feeds the final activation).
        # The stage average has no module of its own, so it is rebuilt from the
        # three AMP blocks' outputs exactly as the decoder's forward does.
        bigvgan = audio_vae.decoder
        audio_hooks, amp_outputs = [], {}

        def dump(name: str):
            def hook(_module, _inputs, output):
                out[name] = output.detach().float().cpu().contiguous()

            return hook

        def collect(index: int):
            def hook(_module, _inputs, output):
                amp_outputs[index] = output.detach()

            return hook

        audio_hooks.append(bigvgan.conv_pre.register_forward_hook(dump("audio_conv_pre")))
        for i in range(bigvgan.num_upsamples):
            audio_hooks.append(bigvgan.ups[i][0].register_forward_hook(dump(f"audio_up_{i}")))
        for r, block in enumerate(bigvgan.resblocks):
            audio_hooks.append(block.register_forward_hook(collect(r)))
        # Are the 254 anti-aliasing filter buffers really one filter? The port loads a single copy.
        filters = [b for n, b in bigvgan.named_buffers() if n.endswith("filter")]
        meta["audio_filters"] = {
            "count": len(filters),
            "all_bit_equal": all(torch.equal(f, filters[0]) for f in filters),
            "taps": [float(x) for x in filters[0].flatten()],
        }

        with torch.no_grad():
            t0 = time.time()
            wave = audio_vae.decode(a_latent.to(dev) * a_std + a_mean, return_dict=False)[0]  # decoders.py:243-247
            torch.cuda.synchronize()
            meta["audio_decode_s"] = time.time() - t0
        assert wave.shape == (AUDIO_CHANNELS, 1, args.audio_latents * 800), wave.shape
        for h in audio_hooks:
            h.remove()
        k = bigvgan.num_kernels
        for i in range(bigvgan.num_upsamples):
            stage = sum(amp_outputs[i * k + j] for j in range(k)) / k
            out[f"audio_stage_{i}"] = stage.float().cpu().contiguous()
        out["audio_wave"] = wave[:, 0].float().cpu().contiguous()  # [2, 800*n], already clamped to [-1, 1]
        del audio_vae, wave
        release()

    meta["stats"] = {
        k: {"mean": float(v.double().mean()), "std": float(v.double().std()), "absmax": float(v.double().abs().max())}
        for k, v in out.items()
        if v.numel() > 1
    }
    meta["shapes"] = {k: [str(v.dtype), list(v.shape)] for k, v in out.items()}
    save_file(out, args.out)
    json.dump(meta, open(args.meta, "w"), indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
