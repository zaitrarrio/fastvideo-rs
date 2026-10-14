#!/usr/bin/env python3
"""The MiniMax-H3-Turbo-Lora Space as a parity reference for our Turbo LoRA recipes (docs/ports/h3-turbo-loras.md).

Runs what the Space runs, from its own files (`h3_lora.py`, `h3_split_blocks.py` at Space revision 9a76d679, fetched
verbatim by the pod script): diffusers' MiniMax-H3 modular pipeline at the Space's pinned commit (huggingface/diffusers
#14371 @ 665f5782), the denoising half (`MiniMaxH3GeneratorBlocks`), bf16 transformer, float32 VAEs, with the Space's
`apply_lora` folding the larry LoRA into the bf16 weights at load and `set_active` switching to the lightx2v set.

What it writes, in our dump format (`<name>.f32` raw little-endian float32 + `<name>.shape`):

  <out>/larry/merged.<prefix>, base.<prefix>   the Space's larry fold of each `--params` linear, and the base weight
  <out>/lightx/merged.<prefix>                 the Space's lightx2v fold of the same linears
  <out>/larry/text_hidden                      the conditioning it was given (ours, so text is out of the comparison)
  <out>/larry/{video,audio}_step00_in          its starting noise rows (torch's draws off the request's generator)
  <out>/larry/{video,audio}_{vel_,}stepNN      each step's velocity and latents rows (NN = 01 after the first step)
  <out>/larry/{video,audio}_sigmas
  <clip>/output.mp4, <clip>/audio.wav          the decoded clip (the Space's `encode_video`) and its soundtrack

Our side then runs with `FASTVIDEO_INJECT_DIR=<out>/larry` (the noise and text) and `FASTVIDEO_DUMP_DIR`, and
`fv-gpucheck compare-dumps` / `av_psnr.py` compare. The text conditioning is ours (`--text-hidden`, the Qwen3-VL
layer-50 rows of our first run): the Space gets it from a remote conditioner Space, which is not part of this check.

    h3_turbo_lora_ref.py --base <overlay> --space <dir with h3_lora.py, h3_split_blocks.py> \\
        --larry <file> --lightx <file> --text-hidden <ours>/text_hidden --out <dir> --clip <dir> \\
        --height 480 --width 832 --num-frames 124 --steps 6 --seed 1024 --params p1,p2,...

`--steps` is the Space's (diffusers `num_inference_steps`, grid points): 6 is 5 forwards (`larry-turbo-5step`).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path


def dump(out: Path, name: str, t) -> None:
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    a = t.detach().float().cpu().contiguous().numpy().astype(np.float32)
    (out / f"{name}.f32").write_bytes(a.tobytes())
    (out / f"{name}.shape").write_text(" ".join(str(d) for d in a.shape))


def read_dump(prefix: Path):
    import numpy as np

    shape = [int(x) for x in Path(f"{prefix}.shape").read_text().split()]
    data = np.fromfile(f"{prefix}.f32", dtype="<f4")
    return data.reshape(shape)


def write_wav_f32(path: Path, audio, rate: int) -> None:
    """IEEE-float WAV (format 3), interleaved; `audio` is (channels, samples)."""
    import numpy as np

    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    ch, n = a.shape
    data = a.T.reshape(-1).astype("<f4").tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 3, ch, rate, rate * ch * 4, ch * 4, 32)
    hdr += b"data" + struct.pack("<I", len(data))
    path.write_bytes(hdr + data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="diffusers MiniMax-H3 dir (transformer/ vae/ audio_vae/ schedulers, local index)")
    ap.add_argument("--space", required=True, help="dir holding the Space's h3_lora.py and h3_split_blocks.py")
    ap.add_argument("--larry", required=True)
    ap.add_argument("--lightx", required=True)
    ap.add_argument("--text-hidden", required=True, help="our text_hidden dump prefix (rows x 5120)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=6, help="diffusers num_inference_steps (grid points)")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--params", default="", help="comma-separated linear prefixes whose weights are dumped")
    ap.add_argument("--active", choices=("larry", "lightx"), default="larry",
                    help="larry: the Space's default (larry folded, lightx switched to and back for its weights); lightx: "
                         "only the lightx2v set loaded (H3_LORA=off), folded once, and the denoise runs with it")
    ap.add_argument("--attention", default="_native_cudnn", help="the Space's H3_ATTENTION ('' keeps the default)")
    a = ap.parse_args()

    out = Path(a.out)
    meta: dict[str, object] = {"args": vars(a)}
    t0 = time.time()

    # The Space's files fetch the LoRAs with hf_hub_download(repo, file): answer from the local copies instead.
    import huggingface_hub

    local = {Path(a.larry).name: a.larry, Path(a.lightx).name: a.lightx}
    real = huggingface_hub.hf_hub_download

    def hub(repo_id, filename, *args, **kw):
        if filename in local:
            return local[filename]
        return real(repo_id, filename, *args, **kw)

    huggingface_hub.hf_hub_download = hub
    os.environ["H3_LORA"] = Path(a.larry).name if a.active == "larry" else "off"
    os.environ["H3_LIGHTX_FILE"] = Path(a.lightx).name
    os.environ.setdefault("H3_LORA_STRENGTH", "1.0")
    os.environ["H3_LORA_DEFAULT"] = a.active
    run = out / a.active
    sys.path.insert(0, a.space)

    import diffusers
    import torch
    import transformers

    import h3_lora
    from h3_split_blocks import MiniMaxH3GeneratorBlocks

    meta.update(torch=torch.__version__, cuda=torch.version.cuda, diffusers=diffusers.__version__, transformers=transformers.__version__)

    pipe = MiniMaxH3GeneratorBlocks().init_pipeline(a.base)
    pipe.load_components(dtype=torch.bfloat16)
    meta["load_s"] = time.time() - t0
    print(f"[ref] loaded in {meta['load_s']:.0f}s", flush=True)

    params = dict(pipe.transformer.named_parameters())
    wanted = [p.strip() for p in a.params.split(",") if p.strip()]
    for p in wanted:
        if f"{p}.weight" not in params:
            raise KeyError(f"--params: {p}.weight is not a transformer parameter")
        dump(run, f"base.{p}", params[f"{p}.weight"])

    # The Space's fold, verbatim (h3_lora.apply_lora). The Space folds with the transformer still on the host; here it
    # is on the card first (the pod's RAM need not hold 62 GB twice). The arithmetic is the same: `_apply` computes
    # each delta from the CPU-resident factors on the CPU and adds it to the parameter in float32 where it lives.
    pipe.transformer.to("cuda")
    t1 = time.time()
    status = h3_lora.apply_lora(pipe.transformer)
    meta["lora_status"] = status
    meta["fold_s"] = time.time() - t1
    print(f"[ref] {status} ({meta['fold_s']:.0f}s)", flush=True)
    params = dict(pipe.transformer.named_parameters())
    for p in wanted:
        dump(run, f"merged.{p}", params[f"{p}.weight"])
    if a.active == "larry" and h3_lora.set_active(pipe.transformer, "lightx") != "lightx":
        raise RuntimeError("the lightx set did not load")
    params = dict(pipe.transformer.named_parameters())
    for p in wanted if a.active == "larry" else []:
        dump(out / "lightx", f"merged.{p}", params[f"{p}.weight"])
    # Back to larry: the Space's switch unfolds lightx and refolds larry through one more bf16 rounding; the
    # dumped larry weights above are the startup fold. The run below uses the switched-back weights, as a
    # Space request after a lightx one would.
    if a.active == "larry":
        assert h3_lora.set_active(pipe.transformer, "larry") == "larry"
        params = dict(pipe.transformer.named_parameters())
        for p in wanted:
            dump(run, f"merged_after_switch.{p}", params[f"{p}.weight"])

    if a.attention:
        try:
            pipe.transformer.set_attention_backend(a.attention)
            meta["attention"] = a.attention
        except Exception as e:  # noqa: BLE001 - a missing backend falls back to the default SDPA
            meta["attention"] = f"default ({a.attention}: {e})"
    pipe.to("cuda")

    # The conditioning: ours, as the remote conditioner's `prompt_embeds` (bf16) + all-text `text_token_tags`.
    hidden = torch.from_numpy(read_dump(Path(a.text_hidden)).copy())
    hidden = hidden.reshape(-1, hidden.shape[-1])
    prompt_embeds = hidden.to(torch.bfloat16)[None].to("cuda")
    text_tag = int(getattr(pipe, "text_tag", 1))
    text_token_tags = torch.full((hidden.shape[0],), text_tag, dtype=torch.long)
    dump(run, "text_hidden", hidden)

    # Hooks: the starting rows after the noise draw, each step's velocity and latents.
    from diffusers.modular_pipelines.minimax_h3 import before_denoise, denoise

    prep_call = before_denoise.MiniMaxH3PrepareLatentsStep.__call__

    def prep(self, components, state):
        components, state = prep_call(self, components, state)
        bs = self.get_block_state(state)
        dump(run, "video_step00_in", bs.latents)
        dump(run, "audio_step00_in", bs.audio_latents)
        return components, state

    before_denoise.MiniMaxH3PrepareLatentsStep.__call__ = prep
    step_call = denoise.MiniMaxH3LoopSchedulerStep.__call__

    def step(self, components, block_state, i, t):
        cv, ca = block_state.num_condition_video_rows, block_state.num_condition_audio_rows
        if i == 0:
            dump(run, "video_sigmas", components.scheduler.sigmas)
            dump(run, "audio_sigmas", components.audio_scheduler.sigmas)
        dump(run, f"video_vel_step{i + 1:02}", block_state.noise_pred[0, cv:])
        dump(run, f"audio_vel_step{i + 1:02}", block_state.audio_noise_pred[0, ca:])
        components, block_state = step_call(self, components, block_state, i, t)
        dump(run, f"video_step{i + 1:02}", block_state.latents[cv:])
        dump(run, f"audio_step{i + 1:02}", block_state.audio_latents[ca:])
        return components, block_state

    denoise.MiniMaxH3LoopSchedulerStep.__call__ = step

    t2 = time.time()
    torch.cuda.reset_peak_memory_stats()
    state = pipe(
        prompt_embeds=prompt_embeds,
        text_token_tags=text_token_tags,
        image=None,
        last_image=None,
        height=a.height,
        width=a.width,
        num_frames=a.num_frames,
        num_inference_steps=a.steps,
        generator=torch.Generator("cpu").manual_seed(a.seed),
    )
    torch.cuda.synchronize()
    meta["generate_s"] = time.time() - t2
    meta["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    frames, audio, rate = state.get("videos")[0], state.get("audio")[0].cpu(), state.get("sampling_rate")
    clip = Path(a.clip)
    clip.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import encode_video

    encode_video(frames, fps=24, output_path=str(clip / "output.mp4"), audio=audio, audio_sample_rate=rate)
    write_wav_f32(clip / "audio.wav", audio.float().numpy(), int(rate))
    meta["sampling_rate"] = int(rate)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[ref] generate {meta['generate_s']:.0f}s, peak {meta['peak_allocated_gib']:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
