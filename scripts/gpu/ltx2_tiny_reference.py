#!/usr/bin/env python3
"""CPU reference fixtures for the LTX-2 port: diffusers' own classes at toy sizes.

`ltx2_oracle.py` needs the 19B checkpoint and a 96 GB card. Most of what it can
find — a swapped shift/scale, a wrong rotary layout, a mis-paired depth-to-space —
does not depend on the size of the model. This script instantiates each diffusers
LTX-2 class with a tiny config and seeded random weights, runs it once in float32
on the CPU, and writes weights + input + output (+ block taps) to one safetensors
file per component. The Rust tests in `crates/fastvideo-cudarc/src/ltx2/
reference_tests.rs` load the same weights through the production loaders and must
reproduce the outputs. No GPU, no download.

    python scripts/gpu/ltx2_tiny_reference.py --out crates/fastvideo-cudarc/src/ltx2/fixtures

Needs torch + diffusers (git main). On a torch too old for diffusers main to
import (e.g. 2.2 on Intel macOS) pass `--shim-old-torch`: the missing modules are
stubbed, none of which the LTX-2 math touches. The configs below are mirrored in
the Rust tests; change both together.
"""

from __future__ import annotations

import argparse
import os
import sys


def shim_old_torch() -> None:
    import contextlib
    import enum
    import types

    import torch
    import torch.distributed

    class _Any:
        def __getattr__(self, n):
            return lambda *a, **k: None

    for name in ("xpu", "mtia"):
        if not hasattr(torch, name):
            setattr(torch, name, _Any())
    if not hasattr(torch.distributed, "device_mesh"):
        m = types.ModuleType("torch.distributed.device_mesh")
        m.DeviceMesh = type("DeviceMesh", (), {})
        torch.distributed.device_mesh = m
        sys.modules["torch.distributed.device_mesh"] = m
    if not hasattr(torch.nn, "RMSNorm"):
        # torch >= 2.4's module, verbatim semantics: x * rsqrt(mean(x^2) + eps) * weight.
        class RMSNorm(torch.nn.Module):
            def __init__(self, normalized_shape, eps=None, elementwise_affine=True):
                super().__init__()
                self.eps = eps
                self.weight = torch.nn.Parameter(torch.ones(normalized_shape)) if elementwise_affine else None

            def forward(self, x):
                eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
                y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
                return y if self.weight is None else y * self.weight

        torch.nn.RMSNorm = RMSNorm
    import torch.nn.functional as F

    try:
        accepts_gqa = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")
    except Exception:
        accepts_gqa = False
    if not accepts_gqa:
        native = F.scaled_dot_product_attention

        def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            assert not enable_gqa or query.shape[1] == key.shape[1]
            return native(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        F.scaled_dot_product_attention = sdpa
    try:
        import torch.nn.attention  # noqa: F401
    except ModuleNotFoundError:
        a = types.ModuleType("torch.nn.attention")

        class SDPBackend(enum.Enum):
            MATH = 0
            FLASH_ATTENTION = 1
            EFFICIENT_ATTENTION = 2
            CUDNN_ATTENTION = 3

        a.SDPBackend = SDPBackend
        a.sdpa_kernel = lambda *x, **k: contextlib.nullcontext()
        torch.nn.attention = a
        sys.modules["torch.nn.attention"] = a
        f = types.ModuleType("torch.nn.attention.flex_attention")
        f.flex_attention = None
        f.BlockMask = type("BlockMask", (), {})
        f.create_block_mask = None
        sys.modules["torch.nn.attention.flex_attention"] = f


TRANSFORMER = dict(
    in_channels=6, out_channels=6, patch_size=1, patch_size_t=1, num_attention_heads=4, attention_head_dim=8,
    cross_attention_dim=32, vae_scale_factors=(8, 32, 32), pos_embed_max_pos=20, base_height=2048, base_width=2048,
    audio_in_channels=5, audio_out_channels=5, audio_patch_size=1, audio_patch_size_t=1, audio_num_attention_heads=4,
    audio_attention_head_dim=4, audio_cross_attention_dim=16, audio_scale_factor=4, audio_pos_embed_max_pos=20,
    audio_sampling_rate=16000, audio_hop_length=160, num_layers=2, activation_fn="gelu-approximate",
    qk_norm="rms_norm_across_heads", norm_elementwise_affine=False, norm_eps=1e-6, caption_channels=12,
    attention_bias=True, attention_out_bias=True, rope_theta=10000.0, rope_double_precision=True, causal_offset=1,
    timestep_scale_multiplier=1000, cross_attn_timestep_scale_multiplier=1000, rope_type="split",
)
CONNECTORS = dict(
    caption_channels=8, text_proj_in_factor=3, video_connector_num_attention_heads=2, video_connector_attention_head_dim=4,
    video_connector_num_layers=2, video_connector_num_learnable_registers=4, audio_connector_num_attention_heads=2,
    audio_connector_attention_head_dim=4, audio_connector_num_layers=1, audio_connector_num_learnable_registers=4,
    connector_rope_base_seq_len=16, rope_theta=10000.0, rope_double_precision=True, causal_temporal_positioning=False,
    rope_type="split",
)
VAE = dict(
    in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 16, 32, 64), decoder_block_out_channels=(8, 16, 32),
    layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1), spatio_temporal_scaling=(True, True, True, True),
    decoder_spatio_temporal_scaling=(True, True, True), decoder_inject_noise=(False, False, False, False),
    upsample_residual=(True, True, True), upsample_factor=(2, 2, 2), timestep_conditioning=False, patch_size=2, patch_size_t=1,
    encoder_causal=True, decoder_causal=False, encoder_spatial_padding_mode="zeros", decoder_spatial_padding_mode="reflect",
)
AUDIO_VAE = dict(
    # diffusers sizes latents_mean/std by `base_channels`; it must equal latent_channels * (mel_bins / 4).
    base_channels=4, output_channels=2, ch_mult=(1, 2, 4), num_res_blocks=1, attn_resolutions=None, in_channels=2, resolution=256,
    latent_channels=2, norm_type="pixel", causality_axis="height", dropout=0.0, mid_block_add_attention=False,
    sample_rate=16000, mel_hop_length=160, is_causal=True, mel_bins=8, double_z=True,
)
VOCODER = dict(
    in_channels=16, hidden_channels=64, out_channels=2, upsample_kernel_sizes=[7, 4, 4, 4, 4], upsample_factors=[3, 2, 2, 2, 2],
    resnet_kernel_sizes=[3, 7, 11], resnet_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], leaky_relu_negative_slope=0.1,
    output_sampling_rate=24000,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shim-old-torch", action="store_true")
    ap.add_argument("--only", default="", help="comma list: dit,connectors,vae,audio")
    args = ap.parse_args()
    if args.shim_old_torch:
        shim_old_torch()
    import torch
    from safetensors.torch import save_file

    only = {s for s in args.only.split(",") if s}
    os.makedirs(args.out, exist_ok=True)

    def randomise(model: "torch.nn.Module", seed: int, scale: float = 1.0) -> None:
        # Default inits leave zero biases and unit-ish tables, which would hide a
        # dropped bias or a swapped table row. Everything gets seeded noise.
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in sorted(model.named_parameters()):
                std = scale * (0.5 if p.ndim <= 1 or "table" in name or "registers" in name else 1.5 / max(p.shape[1:].numel(), 1) ** 0.5)
                p.copy_(torch.randn(p.shape, generator=g) * std)
                if name.endswith(("norm_q.weight", "norm_k.weight")):
                    p.add_(1.0)
            for name, b in sorted(model.named_buffers()):
                if name.endswith(("latents_mean", "latents_std")):
                    b.copy_(torch.randn(b.shape, generator=g) * 0.3 + (1.0 if name.endswith("std") else 0.0))

    def dump(name: str, model: "torch.nn.Module", extra: dict, keep=lambda k: True) -> None:
        tensors = {k: v.detach().float().contiguous() for k, v in model.state_dict().items() if keep(k)}
        for k, v in extra.items():
            assert k not in tensors
            tensors[k] = v.detach().float().contiguous()
        # `.st`: the repository ignores *.safetensors, and these fixtures are committed.
        path = os.path.join(args.out, f"{name}.st")
        save_file(tensors, path)
        print(name, len(tensors), os.path.getsize(path))

    def randn(shape, seed):
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed))

    if not only or "dit" in only:
        from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

        model = LTX2VideoTransformer3DModel(**TRANSFORMER).eval()
        randomise(model, 1)
        grid, audio_n, text_n, fps, timestep = (3, 2, 3), 5, 7, 24.0, 725.0
        video = randn((1, grid[0] * grid[1] * grid[2], 6), 2)
        audio = randn((1, audio_n, 5), 3)
        ctx_v, ctx_a = randn((1, text_n, 12), 4), randn((1, text_n, 12), 5)
        taps = {}
        hooks = []
        for i, block in enumerate(model.transformer_blocks):
            def tap(_m, _i, o, i=i):
                taps[f"ref.block{i}.video"], taps[f"ref.block{i}.audio"] = o[0], o[1]
            hooks.append(block.register_forward_hook(tap))

        # The same sub-layer taps, by the same hooks, as ltx2_oracle.py `install_taps`
        # (kept in step by hand): running them here is also the only place those hooks
        # execute without a 96 GB card.
        def first(a, k):
            return a[0] if a else k["hidden_states"]

        streams = {
            "video": [("attn1", "norm2"), ("attn2", "audio_to_video_norm"), ("audio_to_video_attn", "norm3"), ("ff", None)],
            "audio": [("audio_attn1", "audio_norm2"), ("audio_attn2", "video_to_audio_norm"), ("video_to_audio_attn", "audio_norm3"), ("audio_ff", None)],
        }
        short = {"attn1": "attn1", "attn2": "attn2", "audio_to_video_attn": "av", "ff": "ff", "audio_attn1": "attn1", "audio_attn2": "attn2", "video_to_audio_attn": "av", "audio_ff": "ff"}
        for i, block in enumerate(model.transformer_blocks):
            for stream, layers in streams.items():
                for layer, next_norm in layers:
                    base = f"ref.block{i:02d}.{stream}.{short[layer]}"

                    def pre(_m, a, k, base=base):
                        taps[f"{base}_in"] = first(a, k)

                    def post(_m, _a, _k, o, base=base):
                        taps[f"{base}_out"] = o

                    hooks.append(getattr(block, layer).register_forward_pre_hook(pre, with_kwargs=True))
                    hooks.append(getattr(block, layer).register_forward_hook(post, with_kwargs=True))
                    if next_norm is not None:

                        def after(_m, a, k, base=base):
                            taps[f"{base}_after"] = first(a, k)

                        hooks.append(getattr(block, next_norm).register_forward_pre_hook(after, with_kwargs=True))
        for stream, norm, proj in [("video", "norm_out", "proj_out"), ("audio", "audio_norm_out", "audio_proj_out")]:

            def normed(_m, _a, _k, o, stream=stream):
                taps[f"ref.head.{stream}.norm"] = o

            def modulated(_m, a, k, stream=stream):
                taps[f"ref.head.{stream}.modulated"] = first(a, k) if a or "hidden_states" in k else k["input"]

            hooks.append(getattr(model, norm).register_forward_hook(normed, with_kwargs=True))
            hooks.append(getattr(model, proj).register_forward_pre_hook(modulated, with_kwargs=True))
        with torch.no_grad():
            t = torch.tensor([timestep])
            v, a = model(
                hidden_states=video, audio_hidden_states=audio, encoder_hidden_states=ctx_v, audio_encoder_hidden_states=ctx_a,
                timestep=t, sigma=t, encoder_attention_mask=torch.ones(1, text_n), audio_encoder_attention_mask=torch.ones(1, text_n),
                num_frames=grid[0], height=grid[1], width=grid[2], fps=fps, audio_num_frames=audio_n,
                use_cross_timestep=True, return_dict=False,
            )
            coords = model.rope.prepare_video_coords(1, *grid, "cpu", fps=fps)
            cos, sin = model.rope(coords)
        dump("dit_tiny", model, {
            "ref.video_in": video, "ref.audio_in": audio, "ref.ctx_video": ctx_v, "ref.ctx_audio": ctx_a,
            "ref.video_out": v, "ref.audio_out": a, "ref.rope.video.cos": cos, "ref.rope.video.sin": sin, **taps,
        })

    if not only or "connectors" in only:
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

        model = LTX2TextConnectors(**CONNECTORS).eval()
        randomise(model, 11)
        total, real = 8, 3
        states = randn((1, total, 8, 3), 12) * 2.0 + 0.2
        mask = torch.zeros(1, total, dtype=torch.long)
        mask[0, total - real :] = 1
        with torch.no_grad():
            v, a, m = model(states, mask, padding_side="left")
        assert bool(m.all())
        dump("connectors_tiny", model, {"ref.hidden_states": states[0, total - real :], "ref.video": v, "ref.audio": a})

    if not only or "vae" in only:
        from diffusers import AutoencoderKLLTX2Video

        model = AutoencoderKLLTX2Video(**VAE).eval()
        randomise(model, 21, scale=1.5)
        z = randn((1, 4, 3, 2, 2), 22)
        with torch.no_grad():
            mean = model.latents_mean.view(1, -1, 1, 1, 1)
            std = model.latents_std.view(1, -1, 1, 1, 1)
            video = model.decode(z * std / model.config.scaling_factor + mean, None, return_dict=False)[0]
        dump("vae_tiny", model, {"ref.latent": z, "ref.video": video}, keep=lambda k: k.startswith("decoder.") or k.startswith("latents_"))

    if not only or "audio" in only:
        from diffusers import AutoencoderKLLTX2Audio
        from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

        vae = AutoencoderKLLTX2Audio(**AUDIO_VAE).eval()
        randomise(vae, 31, scale=1.5)
        packed = randn((1, 4, 4), 32)
        with torch.no_grad():
            z = packed * vae.latents_std + vae.latents_mean
            z = z.unflatten(2, (-1, 2)).transpose(1, 2)
            mel = vae.decode(z, return_dict=False)[0]
        dump("audio_vae_tiny", vae, {"ref.latent": packed, "ref.mel": mel}, keep=lambda k: k.startswith("decoder.") or k.startswith("latents_"))
        voc = LTX2Vocoder(**VOCODER).eval()
        randomise(voc, 41, scale=1.2)
        with torch.no_grad():
            wave = voc(mel)
        dump("vocoder_tiny", voc, {"ref.mel": mel, "ref.wave": wave})
    return 0


if __name__ == "__main__":
    sys.exit(main())
