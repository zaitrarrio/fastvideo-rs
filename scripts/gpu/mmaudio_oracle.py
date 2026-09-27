#!/usr/bin/env python3
"""Upstream MMAudio (hkchengrex/MMAudio 974010a, the commit strobe pins) as
strobe's `scripts/batch/sidecar-audio.py` runs it: large_44k_v2, bf16 on
CUDA, Euler 25 steps, CFG 4.5, negative text "", duration = the clip's.

    PYTHONPATH=<MMAudio checkout> python mmaudio_oracle.py --weights <root> \
        --video clip.mp4 --prompt "..." --out <dir> [--runs 3] [--dump <dir>]

Weights come from the local tree of scripts/gpu/fetch-mmaudio.py (the hub
loaders are pointed at it; nothing is downloaded). `--runs`: timed
generations, strobe's `audio_s` (the `generate` call through a CUDA sync,
`load_video` excluded; reported separately). `--dump`: the oracle dump
(docs/oracle.md format, `mm_*` names, see crates/fastvideo-cudarc/src/mmaudio/
pipeline.rs) of one more generation, with the sampler written out step by step
(the same arithmetic as `eval_utils.generate` + `FlowMatching.to_data`), plus
an f32 VAE decode and vocode of the same latent (`mm_mel_f32`, `mm_wave_f32`).
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch


def write(dump, name, t):
    t = t.detach().float().cpu().contiguous()
    (dump / f"{name}.f32").write_bytes(t.numpy().tobytes())
    (dump / f"{name}.shape").write_text(" ".join(str(d) for d in t.shape))


def setup(root: Path):
    import open_clip
    import mmaudio.model.utils.features_utils as fu
    import mmaudio.ext.autoencoder.autoencoder as ae

    clip_dir = root / "DFN5B-CLIP-ViT-H-14-384"

    def local_clip(name, return_transform=False):
        assert name == "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", name
        return open_clip.create_model_from_pretrained(
            "ViT-H-14-378-quickgelu", pretrained=str(clip_dir / "open_clip_pytorch_model.bin"),
            return_transform=False)

    fu.create_model_from_pretrained = local_clip
    orig = ae.BigVGANv2.from_pretrained

    def local_vocoder(name, **kw):
        assert name == "nvidia/bigvgan_v2_44khz_128band_512x", name
        return orig(str(root / "bigvgan_v2_44khz_128band_512x"), **kw)

    ae.BigVGANv2.from_pretrained = local_vocoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    root = Path(a.weights).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup(root)
    import soundfile  # torchaudio.save needs torchcodec on current torchaudio
    from mmaudio.eval_utils import all_model_cfg, generate, load_video
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    model = all_model_cfg["large_44k_v2"]
    seq_cfg = model.seq_cfg
    device, dtype = "cuda", torch.bfloat16
    t0 = time.monotonic()
    net = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(root / "weights/mmaudio_large_44k_v2.pth", map_location=device, weights_only=True))
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=a.steps)
    feature_utils = FeaturesUtils(tod_vae_ckpt=str(root / "ext_weights/v1-44.pth"),
                                  synchformer_ckpt=str(root / "ext_weights/synchformer_state_dict.pth"),
                                  enable_conditions=True, mode=model.mode, bigvgan_vocoder_ckpt=None,
                                  need_vae_encoder=False).to(device, dtype).eval()
    torch.cuda.synchronize()
    load_s = time.monotonic() - t0

    import av
    with av.open(a.video) as c:
        st = c.streams.video[0]
        nframes, fps = st.frames, float(st.guessed_rate)
    duration = a.duration or (nframes / fps)
    t0 = time.monotonic()
    vi = load_video(Path(a.video), duration)
    load_video_s = time.monotonic() - t0
    clip_frames = vi.clip_frames.unsqueeze(0)
    sync_frames = vi.sync_frames.unsqueeze(0)
    seq_cfg.duration = float(vi.duration_sec)
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    torch.cuda.reset_peak_memory_stats()
    runs = []
    audio = None
    for i in range(a.runs):
        rng = torch.Generator(device=device)
        rng.manual_seed(a.seed)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.inference_mode():
            audio = generate(clip_frames, sync_frames, [a.prompt], negative_text=[""], feature_utils=feature_utils,
                             net=net, fm=fm, rng=rng, cfg_strength=a.cfg)
        torch.cuda.synchronize()
        runs.append(time.monotonic() - t0)
        print(f"upstream generate run {i}: {runs[-1]:.3f}s", flush=True)
    wave = audio.float().cpu()[0]
    soundfile.write(str(out / "upstream.wav"), wave[0].numpy(), seq_cfg.sampling_rate, subtype="FLOAT")
    (out / "upstream.f32").write_bytes(wave[0].numpy().tobytes())
    audio_s = runs[-1]
    doc = {"pipeline": "MMAudio (upstream Python, strobe sidecar settings)", "commit": "974010a",
           "variant": "large_44k_v2", "dtype": "bfloat16", "steps": a.steps, "cfg": a.cfg, "seed": a.seed,
           "clip_s": duration, "duration_s": float(vi.duration_sec), "load_s": load_s,
           "load_video_s": load_video_s, "runs_s": runs, "audio_s": audio_s, "audio_rtf": audio_s / duration,
           "peak_allocated_mib": torch.cuda.max_memory_allocated() >> 20,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "latent_seq_len": seq_cfg.latent_seq_len, "clip_seq_len": seq_cfg.clip_seq_len,
           "sync_seq_len": seq_cfg.sync_seq_len, "samples": int(wave.shape[-1])}
    print(json.dumps(doc, indent=1))
    (out / "upstream.json").write_text(json.dumps(doc, indent=1))
    if a.dump:
        dump_run(Path(a.dump), a, clip_frames, sync_frames, feature_utils, net, fm)


@torch.inference_mode()
def dump_run(dump, a, clip_frames, sync_frames, fu, net, fm):
    dump.mkdir(parents=True, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16
    cf = clip_frames.to(device, dtype)
    write(dump, "mm_clip_pixels", fu.clip_preprocess(cf)[0])
    write(dump, "mm_sync_pixels", sync_frames[0])
    clip_f = fu.encode_video_with_clip(cf, batch_size=40)
    sync_f = fu.encode_video_with_sync(sync_frames.to(device, dtype), batch_size=40)
    text_f = fu.encode_text([a.prompt])
    neg_f = fu.encode_text([""])
    for n, t in [("mm_clip_f", clip_f), ("mm_sync_f", sync_f), ("mm_text_f", text_f), ("mm_neg_text_f", neg_f)]:
        write(dump, n, t)
    rng = torch.Generator(device=device)
    rng.manual_seed(a.seed)
    x0 = torch.randn(1, net.latent_seq_len, net.latent_dim, device=device, dtype=dtype, generator=rng)
    write(dump, "mm_x0", x0)
    cond = net.preprocess_conditions(clip_f, sync_f, text_f)
    empty = net.get_empty_conditions(1, negative_text_features=neg_f)
    hooks = []
    for i, b in enumerate(net.joint_blocks):
        hooks.append(b.register_forward_hook(lambda m, inp, o, i=i: write(dump, f"mm_step00_block_{i}", o[0])))
    for j, b in enumerate(net.fused_blocks):
        k = len(net.joint_blocks) + j
        hooks.append(b.register_forward_hook(lambda m, inp, o, k=k: write(dump, f"mm_step00_block_{k}", o)))
    x = x0
    steps = torch.linspace(0, 1 - fm.min_sigma, fm.num_steps + 1)
    for ti, t in enumerate(steps[:-1]):
        tt = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
        fc = net.predict_flow(x, tt, cond)
        for h in hooks:
            h.remove()
        hooks = []
        fu_ = net.predict_flow(x, tt, empty)
        if ti == 0:
            write(dump, "mm_flow_c_step00", fc)
            write(dump, "mm_flow_u_step00", fu_)
        flow = a.cfg * fc + (1 - a.cfg) * fu_
        dt = steps[ti + 1] - t
        x = x + dt * flow
        write(dump, f"mm_x_step{ti:02d}", x)
    x1 = net.unnormalize(x)
    write(dump, "mm_x1", x1)
    mel = fu.decode(x1)
    write(dump, "mm_mel", mel)
    wave = fu.vocode(mel)
    write(dump, "mm_wave", wave.reshape(1, -1))
    # f32 decode + vocode of the same latent from freshly loaded f32 weights
    # (fu.tod's were rounded by .to(bf16)): the clean reference for our f32 modules.
    from mmaudio.ext.autoencoder import AutoEncoderModule
    tod = AutoEncoderModule(vae_ckpt_path=str(Path(a.weights) / "ext_weights/v1-44.pth"), vocoder_ckpt_path=None,
                            mode="44k", need_vae_encoder=False).to(device).eval()
    mel32 = tod.decode(x1.float().transpose(1, 2))
    write(dump, "mm_mel_f32", mel32)
    write(dump, "mm_wave_f32", tod.vocode(mel32).reshape(1, -1))
    del tod
    print("dumped", len(list(dump.glob("*.f32"))), "tensors to", dump)


if __name__ == "__main__":
    main()
