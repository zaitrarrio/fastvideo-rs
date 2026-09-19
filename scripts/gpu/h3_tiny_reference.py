#!/usr/bin/env python3
"""CPU reference fixtures for the FastH3 port: diffusers' own classes at toy sizes.

`h3_oracle.py` needs the 70 GB checkpoint and a 96 GB card. Most of what it can
find - a swapped shift/scale, the AdaLN row layout, the rotary channel map, a
mis-paired unpatchify, the tile cross-fade, the weight-norm axis of a transposed
conv, the sign of the scheduler step - does not depend on the size of the model.
This script instantiates each diffusers MiniMax-H3 class with a tiny config and
seeded random weights, runs it once in float32 on the CPU, and writes weights +
inputs + outputs (+ taps) to one file per component. The Rust tests in
`crates/fastvideo-cudarc/src/h3/reference_tests.rs` load the same weights through
the production loaders and must reproduce the outputs. No GPU, no download.

    python scripts/gpu/h3_tiny_reference.py --out crates/fastvideo-cudarc/src/h3/fixtures

Needs torch + diffusers (git main). On a torch too old for diffusers main to
import (2.2 is the last Intel-macOS build) pass `--shim-old-torch`: the missing
modules are stubbed, none of which the H3 math touches. The configs below are
mirrored in the Rust tests; change both together.
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



DMD_RUNGS = [999, 874, 749, 624, 500, 375, 250, 125]
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2

TRANSFORMER = dict(
    num_attention_heads=3, attention_head_dim=16, hidden_size=20, num_layers=3, num_refiner_layers=2, ffn_dim=14,
    in_channels=2, audio_in_channels=3, patch_size=(1, 2, 2), text_dim=7, freq_dim=8, time_embed_hidden_dim=9,
    time_embed_dim=5, rope_freq_dim=2, rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)
# Request geometry of the DiT fixture: 5 text tokens, 3 audio latents per channel, a 3 x 4 x 6 latent.
DIT_TEXT, DIT_AUDIO, DIT_LATENT = 5, 3, (3, 4, 6)
VAE = dict(
    in_channels=3, out_channels=3, latent_channels=5, block_out_channels=(4, 4, 4, 4, 4, 4), layers_per_block=1,
    spatial_downsample_factors=(2, 2, 1, 1, 1, 1), temporal_downsample_factors=(1, 2, 2, 1, 1, 1), norm_num_groups=2,
    decoder_num_layers=2, decoder_num_attention_heads=2, decoder_attention_head_dim=16, decoder_num_register_tokens=4,
    decoder_ffn_mult=2, decoder_rope_theta=100.0, decoder_rope_dim_ratio=0.75, decoder_norm_eps=1e-5, clip_length=17,
    token_drop=3,
)
VAE_TILE, VAE_OVERLAP, VAE_LATENT = 8, 4, (12, 3, 5)  # px, px, latent (T, H, W): 2 chunks, 2 x 4 tiles
AUDIO_VAE = dict(
    encoder_dim=4, encoder_rates=(2, 4, 4, 2, 5), latent_dim=8, latent_channels=4, num_attention_heads=2, decoder_dim=128,
    decoder_rates=(5, 2, 2, 2, 2, 2, 2), decoder_kernel_sizes=(9, 4, 4, 4, 4, 4, 4), resblock_kernel_sizes=(3, 7, 11),
    resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)), sampling_rate=32000,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shim-old-torch", action="store_true")
    ap.add_argument("--only", default="", help="comma list: dit,vae,audio")
    args = ap.parse_args()
    if args.shim_old_torch:
        shim_old_torch()
    import torch
    from safetensors.torch import save_file

    only = {s for s in args.only.split(",") if s}
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    def randomise(model: "torch.nn.Module", seed: int, weight_norm_gain: float = 1.0) -> None:
        # Default inits leave zero biases, zero LayerScales and unit norms, which
        # would hide a dropped bias, a skipped branch or a swapped table row.
        g = torch.Generator().manual_seed(seed)
        for name, p in sorted(model.named_parameters()):
            if p.ndim <= 1 or p.shape[1:].numel() == 1:
                noise = torch.randn(p.shape, generator=g) * 0.3
                # Norm weights, LayerScales and weight-norm gains sit around one.
                leaf = name.rsplit(".", 1)[-1]
                if leaf == "weight_g":
                    # A weight-normed row has norm |g| whatever `weight_v` is, so g IS the gain.
                    p.copy_(noise * 0.3 + weight_norm_gain)
                else:
                    p.copy_(noise + (1.0 if "norm" in name or leaf in ("scale1", "scale2") else 0.0))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * (1.2 / p.shape[1:].numel() ** 0.5))

    def dump(name: str, model: "torch.nn.Module", extra: dict, keep=lambda k: True) -> None:
        tensors = {k: v.detach().float().contiguous() for k, v in model.state_dict().items() if keep(k)}
        for k, v in extra.items():
            assert k not in tensors, k
            tensors[k] = v.detach().float().contiguous()
        # `.st`: the repository ignores *.safetensors, and these fixtures are committed.
        path = os.path.join(args.out, f"{name}.st")
        save_file(tensors, path)
        print(name, len(tensors), "tensors", os.path.getsize(path), "bytes")

    def randn(shape, seed):
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed))

    # --- DiT: one forward with taps, then the 8-step ladder ----------------------------------------
    if not only or "dit" in only:
        from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
            patchify_video_latents,
        )

        model = MiniMaxH3Transformer3DModel(**TRANSFORMER).eval().float()
        randomise(model, 11)
        patch = tuple(TRANSFORMER["patch_size"])
        lt, lh, lw = DIT_LATENT
        text_tags = torch.full((DIT_TEXT,), TEXT_TAG, dtype=torch.long)
        position_ids, token_tags, video_idx, audio_idx, text_idx, ncv, nca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_tags, lt, lh, lw, DIT_AUDIO, patch, 2, AUDIO_TAG, VIDEO_TAG, ()
        )
        assert ncv == 0 and nca == 0
        layout = dict(token_tags=token_tags, position_ids=position_ids, video_indices=video_idx, audio_indices=audio_idx, text_indices=text_idx)

        schedulers = []
        base = torch.tensor([s / 1000.0 for s in DMD_RUNGS] + [0.0], dtype=torch.float32)
        for shift in (10.0, 3.0):
            sch = MiniMaxH3Scheduler(shift=shift)
            sch.set_timesteps(sigmas=shift * base / (1 + (shift - 1) * base))
            schedulers.append(sch)
        video_sch, audio_sch = schedulers

        def row_plan(i: int):
            v, a = float(video_sch.timesteps[i]), float(audio_sch.timesteps[i])
            return MiniMaxH3SetTimestepsStep.build_row_timesteps(video_idx, audio_idx, 0, 0, DIT_TEXT, v, a, max(v, 0.999), 1.0)

        text = randn((1, DIT_TEXT, TRANSFORMER["text_dim"]), 12)
        video_noise = randn((1, TRANSFORMER["in_channels"], lt, lh, lw), 13)
        audio_rows = randn((2 * DIT_AUDIO, TRANSFORMER["audio_in_channels"]), 14)
        video_rows = patchify_video_latents(video_noise, patch)

        def forward(v_rows, a_rows, i):
            unique, inverse = row_plan(i)
            return model(
                hidden_states=v_rows[None], audio_hidden_states=a_rows[None], encoder_hidden_states=text,
                timestep=unique, timestep_indices=inverse, return_dict=False, **layout,
            )

        taps, hooks = {}, []

        def keep(name):
            def hook(_m, _i, output):
                taps[name] = (torch.stack(output) if isinstance(output, tuple) else output).detach().clone()

            return hook

        step = 3
        hooks.append(model.time_embedder.register_forward_hook(keep("ref.temb")))
        hooks.append(model.token_refiner.register_forward_hook(keep("ref.text_refined")))
        for b, block in enumerate(model.transformer_blocks):
            hooks.append(block.register_forward_hook(keep(f"ref.block_{b}")))
            hooks.append(block.adaln_proj.register_forward_hook(keep(f"ref.adaln_{b}")))
        v_out, a_out = forward(video_rows, audio_rows, step)
        for h in hooks:
            h.remove()

        v, a = video_rows.clone(), audio_rows.clone()
        loop_v, loop_a = [], []
        for i, (tv, ta) in enumerate(zip(video_sch.timesteps, audio_sch.timesteps)):
            vo, ao = forward(v, a, i)
            v = video_sch.step(vo[0].float(), tv, v, return_dict=False)[0]
            a = audio_sch.step(ao[0].float(), ta, a, return_dict=False)[0]
            loop_v.append(v.clone())
            loop_a.append(a.clone())

        dump("dit_tiny", model, {
            "in.text": text, "in.video_noise": video_noise, "in.audio_rows": audio_rows, "in.step": torch.tensor([float(step)]),
            "ref.position_ids": position_ids.float(), "ref.token_tags": token_tags.float(),
            "ref.video_rows": video_rows, "ref.video": v_out, "ref.audio": a_out,
            "ref.loop_video": torch.stack(loop_v), "ref.loop_audio": torch.stack(loop_a), **taps,
        })

    # --- video VAE: one tile through the ViT, then the tiled, chunked decode -------------------------
    if not only or "vae" in only:
        from diffusers import AutoencoderKLMiniMaxH3

        mean = [0.1 * (i - 2) for i in range(VAE["latent_channels"])]
        std = [1.0 + 0.2 * i for i in range(VAE["latent_channels"])]
        vae = AutoencoderKLMiniMaxH3(**VAE, latents_mean=tuple(mean), latents_std=tuple(std)).eval().float()
        randomise(vae, 21)
        vae.enable_tiling(
            tile_sample_min_height=VAE_TILE, tile_sample_min_width=VAE_TILE,
            tile_sample_min_overlap_height=VAE_OVERLAP, tile_sample_min_overlap_width=VAE_OVERLAP,
        )
        assert vae.use_tiling and vae.tile_sample_min_height == VAE_TILE and vae.tile_sample_min_overlap_width == VAE_OVERLAP
        t, h, w = VAE_LATENT
        latent = randn((1, VAE["latent_channels"], t, h, w), 22)  # DiT space; the port applies mean/std itself
        z = latent * torch.tensor(std).view(1, -1, 1, 1, 1) + torch.tensor(mean).view(1, -1, 1, 1, 1)
        video = vae.decode(z, return_dict=False)[0]
        # One tile, denormalized already: the ViT alone, no stitching.
        tile = randn((1, VAE["latent_channels"], 7, 2, 2), 23)
        blocks = {}
        hooks = [
            blk.register_forward_hook(lambda _m, _i, o, b=b: blocks.__setitem__(f"ref.tile_block_{b}", o.detach().clone()))
            for b, blk in enumerate(vae.decoder.transformer_blocks)
        ]
        tile_out = vae.decoder(vae.post_quant_conv(tile))
        for hk in hooks:
            hk.remove()
        dump("vae_tiny", vae, {
            "in.latent": latent, "in.tile": tile, "in.latents_mean": torch.tensor(mean), "in.latents_std": torch.tensor(std),
            "ref.video": video, "ref.tile": tile_out, **blocks,
        }, keep=lambda k: k.startswith(("decoder.", "post_quant_conv.")))

    # --- audio VAE decoder ---------------------------------------------------------------------------
    if not only or "audio" in only:
        from diffusers import AutoencoderKLMiniMaxH3Audio

        lc = AUDIO_VAE["latent_channels"]
        mean = [0.05 * (i - 1) for i in range(lc)]
        std = [1.5 + 0.25 * i for i in range(lc)]
        audio = AutoencoderKLMiniMaxH3Audio(**AUDIO_VAE, latents_mean=mean, latents_std=std).eval().float()
        # Unit gains through 7 stages of 3 residual AMP blocks reach |x| ~ 300 and a
        # waveform that is railed at the clamp, which tests nothing and makes
        # sin^2(alpha x) amplify float32 rounding. Keep the stream O(1).
        randomise(audio, 31, weight_norm_gain=0.7)
        latent = randn((2, lc, 3), 32)  # stereo = a batch of two
        dec = audio.decoder
        taps, amp = {}, {}
        hooks = [dec.conv_pre.register_forward_hook(lambda _m, _i, o: taps.__setitem__("ref.conv_pre", o.detach().clone()))]
        for i in range(dec.num_upsamples):
            hooks.append(dec.ups[i][0].register_forward_hook(lambda _m, _i, o, i=i: taps.__setitem__(f"ref.up_{i}", o.detach().clone())))
        for r, blk in enumerate(dec.resblocks):
            hooks.append(blk.register_forward_hook(lambda _m, _i, o, r=r: amp.__setitem__(r, o.detach().clone())))
        z = latent * torch.tensor(std).view(1, -1, 1) + torch.tensor(mean).view(1, -1, 1)
        wave = audio.decode(z, return_dict=False)[0]
        railed = float((wave.abs() >= 1.0).float().mean())
        assert railed < 0.05 and float(wave.std()) > 0.02, f"toy waveform is railed ({railed:.2%}) or silent; retune the gains"
        for hk in hooks:
            hk.remove()
        k = dec.num_kernels
        for i in range(dec.num_upsamples):
            taps[f"ref.stage_{i}"] = sum(amp[i * k + j] for j in range(k)) / k
        # Rows as the DiT emits them: [2 * Na, C], the left channel's latents then the right's.
        rows = latent.permute(0, 2, 1).reshape(-1, lc)
        dump("audio_vae_tiny", audio, {
            "in.latent": latent, "in.rows": rows, "in.latents_mean": torch.tensor(mean), "in.latents_std": torch.tensor(std),
            "ref.wave": wave[:, 0], **taps,
        }, keep=lambda key: key.startswith(("decoder.", "dec_in_proj.")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
