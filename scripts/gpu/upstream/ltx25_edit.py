"""Retake and extend on the LTX-2.5 distilled weights (the oracle reference of
fastvideo-rs's `Ltx2Request::edit`, docs/oracle.md "LTX-2.5 retake and
extend").

Lightricks/LTX-2 `fd4ded7` ships retake as `RetakePipeline`
(`ltx_pipelines/retake.py`): the source video and audio VAE-encoded
(`video_latent_from_file`, `audio_latent_from_file`), one `DiffusionStage` at
the source size with `DISTILLED_SIGMAS`, a `TemporalRegionMask` on each
regenerated modality and `frozen=True` on the other, and the whole clip
decoded. Extend is not in LTX-2; Lightricks' LTX-Desktop (`68cd86c`,
`backend/services/retake_pipeline/ltx_retake_pipeline.py`) runs it on the same
flow: both latents zero-padded by the new latent frames at the start or the
end, the regenerated window the new frames plus a 0.5 s feather into the
source (`_EXTEND_MASK_DELTA_SECONDS`), and, for 2.5 checkpoints, the ancestral
stage sampler (`distilled_stage_sampler_kwargs`: "Distilled 2.5+ checkpoints
need the ancestral sampler"; `retake.py` itself does not pass it).

This module is that flow, line for line, on `DistilledPipeline`'s own blocks
(prompt encoder, image conditioner as the video encoder, stage, decoders),
with the `AudioConditioner` built as in `retake.py`, `video_latent_from_file`
with its default tiling (`TileSizeConfig.default()`, as `retake.py`; LTX-Desktop
passes smaller tiles to save VRAM), and the ancestral sampler of
`DistilledPipeline._stage_1_sampler_kwargs` (the same stepper and loop as
LTX-Desktop's). The source is the whole file (`get_videostream_metadata`),
cut to `8k + 1` frames.

`install(spec, audio_vae_path)` swaps `DistilledPipeline.__call__`; `spec` is
`retake:SOURCE:START:END:MODE` (MODE `av`, `v` or `a`: the modalities
regenerated) or `extend:SOURCE:FRAMES:start|end`. With FV_ORACLE_DUMP_DIR it
also writes `v2v_video_latent` (`[F*H*W, 128]`, the source latent before
padding), `v2v_audio_wave` (`[C, N]`) and `v2v_audio_latent` (`[L, 128]`,
before padding), `v2v_frame0_pixels` (`[3, H, W]`), and the region
(`v2v_region`, `[start_s, end_s]`), the names ltx2/v2v.rs dumps and injects.
"""

from __future__ import annotations

import os

EXTEND_MASK_DELTA_SECONDS = 0.5


def parse_spec(spec: str) -> dict:
    kind, rest = spec.split(":", 1)
    if kind == "retake":
        src, start, end, mode = rest.rsplit(":", 3)
        if mode not in ("av", "v", "a"):
            raise ValueError(f"retake mode {mode}: av, v or a")
        return {"kind": kind, "source": src, "start": float(start), "end": float(end),
                "video": "v" in mode, "audio": "a" in mode}
    if kind == "extend":
        src, frames, at = rest.rsplit(":", 2)
        if at not in ("start", "end") or int(frames) % 8 != 0:
            raise ValueError(f"extend {frames}:{at}: a multiple of 8 frames, start or end")
        return {"kind": kind, "source": src, "frames": int(frames), "at": at}
    raise ValueError(f"edit spec {spec!r}: retake:… or extend:…")


def install(spec: str, audio_vae_path: str) -> None:
    import torch
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
    from ltx_core.model.video_vae import AUTO_TILING
    from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
    from ltx_pipelines import distilled
    from ltx_pipelines.utils.blocks import AudioConditioner
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.helpers import (
        audio_latent_from_file,
        ensure_tiling_config,
        tiling_scale_factors_for_vae,
        video_latent_from_file,
    )
    from ltx_pipelines.utils.media_io import decode_audio_from_file, get_videostream_metadata
    from ltx_pipelines.utils.media_io.decode import decode_video_from_file, video_preprocess
    from ltx_pipelines.utils.types import ModalitySpec

    s = parse_spec(spec)

    def dump(name, t):
        if not os.environ.get("FV_ORACLE_DUMP_DIR"):
            return
        try:
            import oracle_dump

            oracle_dump.write(name, t)
        except Exception as e:  # noqa: BLE001
            print(f"[ltx25_edit] dump {name}: {type(e).__name__}: {e}", flush=True)

    def pad(latent, frames, at):
        """`LTXRetakePipeline._pad_latent_frames`."""
        if frames <= 0:
            return latent
        shape = list(latent.shape)
        shape[2] = frames
        z = torch.zeros(shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([z, latent] if at == "start" else [latent, z], dim=2)

    def call(  # noqa: PLR0913, PLR0915
        self,
        prompt,
        seed,
        height,
        width,
        frame_rate,
        images,
        num_frames=None,
        vae_dtype=None,
        tiling_config=AUTO_TILING,
        enhance_prompt=False,
        enhance_static_cache=False,
        stage_1_sigmas=DISTILLED_SIGMAS,
        stage_2_sigmas=None,
        color_space=None,
        generated_keyframes=0,
    ):
        if images:
            raise ValueError("retake/extend takes no image conditioning")
        src = s["source"]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        # LTX-Desktop `_run`: the source's shape at the requested size, its
        # frame count cut to 8k+1 (`correct_frame_count`).
        meta = get_videostream_metadata(src)
        source_frames = (meta.frames - 1) // 8 * 8 + 1
        output_shape = meta._replace(width=width, height=height, frames=source_frames)
        fps = output_shape.fps
        if abs(fps - frame_rate) > 1e-6:
            raise ValueError(f"--frame-rate {frame_rate} must be the source's {fps}")

        initial_video_latent = self.image_conditioner(
            lambda enc: video_latent_from_file(
                video_encoder=enc, file_path=src, output_shape=output_shape, dtype=dtype, device=self.device
            )
        )
        dump("v2v_video_latent", initial_video_latent[0].permute(1, 2, 3, 0).reshape(-1, initial_video_latent.shape[1]))
        try:
            f0 = next(iter(decode_video_from_file(path=src, device=self.device, start_time=0.0, max_duration=1.0 / fps)))
            dump("v2v_frame0_pixels", video_preprocess(iter([f0]), height, width, dtype, self.device)[0, :, 0])
        except Exception as e:  # noqa: BLE001
            print(f"[ltx25_edit] frame 0 pixels: {type(e).__name__}: {e}", flush=True)

        audio_conditioner = AudioConditioner(audio_vae_path, dtype, self.device)
        wave = decode_audio_from_file(src, self.device, 0.0, output_shape.frames / fps)
        if wave is not None:
            dump("v2v_audio_wave", wave.waveform[0])
        initial_audio_latent = audio_conditioner(
            lambda enc: audio_latent_from_file(
                audio_encoder=enc, file_path=src, output_shape=output_shape, dtype=dtype, device=self.device
            )
        )
        if initial_audio_latent is not None:
            a = initial_audio_latent
            dump("v2v_audio_latent", a[0].permute(1, 0, 2).reshape(a.shape[2], -1))

        if s["kind"] == "extend":
            ext, at = s["frames"], s["at"]
            target_shape = output_shape._replace(frames=output_shape.frames + ext)
            pad_v = (
                VideoLatentShape.from_pixel_shape(target_shape).frames
                - VideoLatentShape.from_pixel_shape(output_shape).frames
            )
            initial_video_latent = pad(initial_video_latent, pad_v, at)
            if initial_audio_latent is not None:
                pad_a = (
                    AudioLatentShape.from_video_pixel_shape(target_shape).frames
                    - AudioLatentShape.from_video_pixel_shape(output_shape).frames
                )
                initial_audio_latent = pad(initial_audio_latent, pad_a, at)
            delta = round(EXTEND_MASK_DELTA_SECONDS * fps)
            if at == "start":
                region_start, region_end = 0.0, min(target_shape.frames, ext + delta) / fps
            else:
                region_start, region_end = max(0, output_shape.frames - delta) / fps, target_shape.frames / fps
            regenerate_video = regenerate_audio = True
        else:
            target_shape = output_shape
            region_start, region_end = s["start"], s["end"]
            regenerate_video, regenerate_audio = s["video"], s["audio"]
        if num_frames is not None and num_frames != target_shape.frames:
            raise ValueError(f"--num-frames {num_frames}: the edit makes {target_shape.frames}")
        dump("v2v_region", torch.tensor([region_start, region_end], dtype=torch.float64))
        print(
            f"[ltx25_edit] {s['kind']} {src}: {meta.width}x{meta.height} {meta.frames}f @ {fps} -> "
            f"{width}x{height} {target_shape.frames}f, region [{region_start:.4f}, {region_end:.4f}) s, "
            f"video {regenerate_video} audio {regenerate_audio} (source audio {initial_audio_latent is not None})",
            flush=True,
        )

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
        )
        v_ctx, a_ctx = ctx_p.video_encoding, ctx_p.audio_encoding

        mask = TemporalRegionMask(start_time=region_start, end_time=region_end, fps=fps)
        video = ModalitySpec(
            context=v_ctx,
            conditionings=[mask] if regenerate_video else [],
            initial_latent=initial_video_latent,
            frozen=not regenerate_video,
        )
        audio = ModalitySpec(
            context=a_ctx,
            conditionings=[mask] if (initial_audio_latent is not None and regenerate_audio) else [],
            initial_latent=initial_audio_latent,
            frozen=initial_audio_latent is not None and not regenerate_audio,
        )

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=target_shape.frames, height=height, width=width, fps=fps),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )
        sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(v_ctx, a_ctx),
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=target_shape.frames,
            fps=fps,
            video=video,
            audio=audio,
            **self._stage_1_sampler_kwargs(seed),
        )
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio, target_shape.frames, tiling_config

    distilled.DistilledPipeline.__call__ = call
    print(f"[ltx25_edit] DistilledPipeline.__call__ -> {s['kind']} on {s['source']}", flush=True)
