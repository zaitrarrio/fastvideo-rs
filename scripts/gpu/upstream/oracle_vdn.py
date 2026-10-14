#!/usr/bin/env python3
"""VDN-H3 upstream reference for the parity run (docs/ports/vdn-h3.md §5).

The published diffusers component of OpenVDN/vdn-minimax-h3 (`<stage>/diffusers/modeling_vdn_h3.py`, the patched
diffusers of the code repo's scripts/setup_diffusers.sh) driven by the code repo's own sampler
(`src/inference/render.py` `generate_latents`, re-spelled here without the repo's package): the packed T2VA
layout from diffusers' MiniMaxH3PrepareLayoutStep, MiniMaxH3Scheduler with shifts 12 / 3 on the uniform
`steps + 1` grid, one forward per step (no CFG), Euler steps.

Inputs are made comparable with ours: the prompt embeddings are OUR text encoder's (`--text-hidden DIR`, the
`text_hidden` of a FASTVIDEO_DUMP_DIR run, cast to bf16 as the cache format holds them), so the text is identical
on both sides; the noise is torch's and is dumped under the names FASTVIDEO_INJECT_DIR reads back.

Dumps (oracle_dump.py format, FV_ORACLE_DUMP_DIR or --dump): text_hidden, video_step00_in, audio_step00_in,
{video,audio}_{sigmas,timesteps}, {video,audio}_vel_stepNN, {video,audio}_stepNN, step00_block_<i> (every 64th
row), and the decoded clip (`--out DIR/output.mp4`, audio also as audio.wav) for av_psnr.py.

    oracle_vdn.py --base-parent /w --stage /w/vdn-h3-stage-b-2000/<rev>/stage-b-step-2000 \\
        --text-hidden ours-dump --height 480 --width 832 --frames 124 --steps 8 --seed 1024 \\
        --dump up/dump --out up --kernels eager --softmax ref

`--kernels eager --softmax ref` (default) is the released arithmetic spelled eagerly with the reference window
softmax (`M:window_softmax_reference`): no torch.compile and no FlexAttention / varlen kernels.
`--kernels inference --softmax decomposed` is the production inference body.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_dump as od  # noqa: E402

FPS = 24
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def log(msg: str) -> None:
    print(f"[oracle_vdn {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_f32(d: Path, name: str) -> torch.Tensor:
    shape = [int(x) for x in (d / f"{name}.shape").read_text().split()]
    a = np.fromfile(d / f"{name}.f32", dtype="<f4").reshape(shape)
    return torch.from_numpy(a)


def load_vdn(base_parent: Path, stage: Path, work: Path, kernels: str, softmax: str):
    """The diffusers component from local files: an overlay whose config points `vdn.base.source` at the
    directory holding `h3-base/` (instead of the Hub), with the branch and adapters linked in."""
    overlay = work / "vdn-overlay"
    (overlay / "diffusers").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((stage / "diffusers/config.json").read_text())
    cfg["vdn"]["base"]["source"] = str(base_parent)
    cfg["vdn"]["inference_kernels"] = kernels == "inference"
    (overlay / "diffusers/config.json").write_text(json.dumps(cfg, indent=1))
    shutil.copy(stage / "diffusers/modeling_vdn_h3.py", overlay / "diffusers/modeling_vdn_h3.py")
    for name in ("linear_branch", "adapters"):
        link = overlay / name
        if not link.exists():
            link.symlink_to(stage / name)
    spec = importlib.util.spec_from_file_location("modeling_vdn_h3", overlay / "diffusers/modeling_vdn_h3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modeling_vdn_h3"] = mod
    spec.loader.exec_module(mod)
    t0 = time.time()
    model = mod.VDNMiniMaxH3Transformer3DModel.from_pretrained(
        str(overlay), subfolder="diffusers", softmax_backend=softmax if kernels == "inference" else None)
    if kernels != "inference":
        # The eager bodies; the window kernel chosen explicitly (`ref`: the per-frame SDPA reference).
        mod.set_softmax_backend(model, softmax)
    log(f"loaded {stage} ({time.time() - t0:.0f}s): kernels={kernels} softmax="
        f"{next(mod.iter_hybrids(model)).softmax_impl}")
    return mod, model


@torch.no_grad()
def generate(model, prompt_embeds, args, device):
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep, patchify_video_latents)
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_CHANNELS as AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG as AUDIO_TAG,
        MINIMAX_H3_VIDEO_TAG as VIDEO_TAG, align_num_frames, audio_latent_num_frames,
        video_latent_num_frames)

    num_frames = align_num_frames(args.frames, 17, 5)
    num_latent_frames = video_latent_num_frames(num_frames, 17, 5)
    num_audio_latents = audio_latent_num_frames(num_frames)
    lat_h, lat_w = args.height // 16, args.width // 16
    patch = tuple(model.config.patch_size)
    channels = model.config.in_channels
    text_token_tags = torch.full((prompt_embeds.shape[0],), 1, dtype=torch.long)
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags, num_latent_frames, lat_h, lat_w, num_audio_latents, patch, AUDIO_CHANNELS,
        AUDIO_TAG, VIDEO_TAG, keyframe_anchors=())
    position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, _ = layout
    position_ids, token_tags = position_ids.to(device), token_tags.to(device)
    video_indices, audio_indices, text_indices = (
        video_indices.to(device), audio_indices.to(device), text_indices.to(device))
    log(f"layout: {num_frames} frames -> {num_latent_frames} latent, {lat_h}x{lat_w}, "
        f"{position_ids.shape[0]} rows ({prompt_embeds.shape[0]} text, {audio_indices.numel()} audio, "
        f"{video_indices.numel()} video)")

    scheduler = MiniMaxH3Scheduler(shift=args.video_shift)
    audio_scheduler = MiniMaxH3Scheduler(shift=args.audio_shift)
    scheduler.set_timesteps(args.steps + 1, device=device)
    audio_scheduler.set_timesteps(args.steps + 1, device=device)

    generator = torch.Generator(device).manual_seed(args.seed)
    latents = torch.randn((1, channels, num_latent_frames, lat_h, lat_w), generator=generator,
                          device=device, dtype=torch.float32)
    video_rows = patchify_video_latents(latents, patch)
    audio_rows = torch.randn((num_audio_latents * AUDIO_CHANNELS, 32), generator=generator,
                             device=device, dtype=torch.float32)
    od.write("text_hidden", prompt_embeds.float())
    od.write("video_step00_in", video_rows)
    od.write("audio_step00_in", audio_rows)
    for tag, s in (("video", scheduler), ("audio", audio_scheduler)):
        od.write(f"{tag}_sigmas", s.sigmas.float())
        od.write(f"{tag}_timesteps", s.timesteps.float())

    seq_len = position_ids.shape[0]
    step_s = []
    for i, (t, audio_t) in enumerate(zip(scheduler.timesteps, audio_scheduler.timesteps)):
        handles = []
        if i == 0:
            for b, block in enumerate(model.transformer_blocks):
                handles.append(block.register_forward_hook(
                    lambda m, a, out, b=b: od.rows_strided(f"step00_block_{b}", out)))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        row_timesteps = torch.full((seq_len,), float(t), dtype=torch.float32, device=device)
        row_timesteps[audio_indices] = float(audio_t)
        timestep, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        noise_pred, audio_noise_pred = model(
            hidden_states=video_rows[None], audio_hidden_states=audio_rows[None],
            encoder_hidden_states=prompt_embeds[None].to(device), timestep=timestep,
            timestep_indices=timestep_indices, token_tags=token_tags, position_ids=position_ids,
            video_indices=video_indices, audio_indices=audio_indices, text_indices=text_indices,
            return_dict=False)
        for h in handles:
            h.remove()
        video_rows = scheduler.step(noise_pred[0].float(), t, video_rows, return_dict=False)[0]
        audio_rows = audio_scheduler.step(audio_noise_pred[0].float(), audio_t, audio_rows, return_dict=False)[0]
        torch.cuda.synchronize()
        step_s.append(time.perf_counter() - t0)
        od.write(f"video_vel_step{i + 1:02d}", noise_pred[0])
        od.write(f"video_step{i + 1:02d}", video_rows)
        od.write(f"audio_vel_step{i + 1:02d}", audio_noise_pred[0])
        od.write(f"audio_step{i + 1:02d}", audio_rows)
        log(f"step {i + 1}/{args.steps}: {step_s[-1]:.2f}s")

    rows = video_rows.reshape(-1, num_latent_frames, lat_h // patch[1], lat_w // patch[2], channels, *patch)
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = rows.reshape(-1, channels, num_latent_frames, lat_h, lat_w).contiguous()
    audio_latents = audio_rows.reshape(AUDIO_CHANNELS, num_audio_latents, 32).permute(0, 2, 1).contiguous()
    return latents, audio_latents, step_s, num_frames


@torch.no_grad()
def decode(base: Path, latents, audio_latents, out: Path, device):
    """render.py decode_and_save: the video VAE under fp16 autocast, the audio VAE, H.264 + AAC."""
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    from diffusers.utils.export_utils import encode_video

    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(base / "vae")).to(device)
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        video = vae.decode(latents * std + mean, return_dict=False)[0]
    pm = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    ps = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * ps + pm).clamp(0, 1)
    del vae
    torch.cuda.empty_cache()
    avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(base / "audio_vae")).to(device)
    am = torch.tensor(avae.config.latents_mean, device=device).view(1, -1, 1)
    asd = torch.tensor(avae.config.latents_std, device=device).view(1, -1, 1)
    audio = avae.decode(audio_latents * asd + am, return_dict=False)[0]
    audio = audio.float().permute(1, 0, 2)[0]
    rate = avae.config.sampling_rate
    frames = (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()
    out.mkdir(parents=True, exist_ok=True)
    encode_video(frames, fps=FPS, output_path=str(out / "output.mp4"), audio=audio.cpu(),
                 audio_sample_rate=rate)
    try:
        import wave

        pcm = (audio.clamp(-1, 1).cpu().numpy().T * 32767).round().astype("<i2")
        with wave.open(str(out / "audio.wav"), "wb") as w:
            w.setnchannels(pcm.shape[1])
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
    except Exception as e:  # noqa: BLE001  (the mp4 carries the audio as well)
        log(f"audio.wav not written: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-parent", type=Path, required=True, help="the directory holding h3-base/")
    p.add_argument("--stage", type=Path, required=True, help="the VDN checkpoint (stage-b-step-2000/)")
    p.add_argument("--text-hidden", type=Path, required=True, help="a dump dir with text_hidden.f32/.shape")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--video-shift", type=float, default=12.0)
    p.add_argument("--audio-shift", type=float, default=3.0)
    p.add_argument("--kernels", choices=("eager", "inference"), default="eager")
    p.add_argument("--softmax", choices=("ref", "decomposed", "flex"), default="ref")
    p.add_argument("--dump", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("/fvw/vdn-up"))
    p.add_argument("--no-decode", action="store_true")
    args = p.parse_args()
    if args.dump is not None:
        os.environ["FV_ORACLE_DUMP_DIR"] = str(args.dump)
    device = "cuda"
    log(f"torch {torch.__version__} cuda {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
    hidden = read_f32(args.text_hidden, "text_hidden").reshape(-1, 5120).to(torch.bfloat16)
    mod, model = load_vdn(args.base_parent, args.stage, args.work, args.kernels, args.softmax)
    model.to(device)
    t0 = time.time()
    latents, audio_latents, step_s, num_frames = generate(model, hidden, args, device)
    denoise = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    if not args.no_decode:
        decode(args.base_parent / "h3-base", latents, audio_latents, args.out, device)
    result = {"frames": num_frames, "height": args.height, "width": args.width, "steps": args.steps,
              "seed": args.seed, "kernels": args.kernels, "softmax": args.softmax, "denoise_s": denoise,
              "step_s": step_s, "peak_gib": peak, "torch": torch.__version__}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    od.meta(vdn=result)
    log(f"done: denoise {denoise:.1f}s, peak {peak:.1f} GiB")


if __name__ == "__main__":
    main()
