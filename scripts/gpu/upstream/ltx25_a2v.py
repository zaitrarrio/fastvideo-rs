"""Audio-to-video on the LTX-2.5 distilled two-stage (the oracle reference of
fastvideo-rs's `Ltx2Request::audio`, docs/oracle.md "LTX-2.5 audio-to-video").

Lightricks/LTX-2 `fd4ded7` ships audio-to-video as `A2VidPipelineTwoStage`
(`ltx_pipelines/a2vid_two_stage.py`), which runs the *dev* transformer with
CFG/STG at stage 1 and the distilled LoRA at stage 2. The weights we serve are
the distilled transformer, which upstream drives with `DistilledPipeline`.
This module is `DistilledPipeline.__call__` with `a2vid_two_stage.py`'s audio
handling put in, line for line, and nothing else changed:

* `decode_audio_from_file(audio_path, device, 0.0, num_frames / frame_rate)`,
* `AudioConditioner(audio_vae)(lambda enc: vae_encode_audio(decoded_audio, enc, None))`,
  cut to `AudioLatentShape.from_duration(num_frames / frame_rate).frames`,
* `audio=ModalitySpec(context=a_ctx, frozen=True, noise_scale=0.0,
  initial_latent=encoded_audio_latent)` on both stages (instead of the
  distilled pipeline's generated audio),
* the returned audio is the decoded input waveform.

`install(audio_path, audio_vae_path)` swaps `DistilledPipeline.__call__`, so
sol-engine's RTX5090 `gpu_infer.py` (through bench_ltx25.py) builds, times and
encodes as for every other LTX-2.5 oracle target. With FV_ORACLE_DUMP_DIR it
also writes `a2v_audio_wave` ([C, N], the decoded waveform) and
`a2v_audio_latent` ([T, 128], the encoded latent patchified as the DiT sees
it), the names ltx2/pipeline.rs `encode_driving_audio` dumps and injects.
"""

from __future__ import annotations

import os


def install(audio_path: str, audio_vae_path: str) -> None:
    import torch
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.model.video_vae import AUTO_TILING
    from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
    from ltx_pipelines import distilled
    from ltx_pipelines.utils.blocks import AudioConditioner
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.helpers import (
        assert_resolution,
        combined_image_conditionings,
        ensure_tiling_config,
        tiling_scale_factors_for_vae,
    )
    from ltx_pipelines.utils.media_io import decode_audio_from_file
    from ltx_pipelines.utils.types import ModalitySpec

    def dump(name, t):
        if not os.environ.get("FV_ORACLE_DUMP_DIR"):
            return
        try:
            import oracle_dump

            oracle_dump.write(name, t)
        except Exception as e:  # noqa: BLE001
            print(f"[ltx25_a2v] dump {name}: {type(e).__name__}: {e}", flush=True)

    def call(  # noqa: PLR0913
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
        stage_2_sigmas=STAGE_2_DISTILLED_SIGMAS,
        color_space=None,
        generated_keyframes=0,
    ):
        if not isinstance(num_frames, int):
            raise ValueError("audio-to-video needs an explicit --num-frames")
        if distilled.has_generated_keyframes(generated_keyframes):
            raise ValueError("audio-to-video takes no generated keyframes")
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        # a2vid_two_stage.py: encode audio.
        decoded_audio = decode_audio_from_file(audio_path, self.device, 0.0, num_frames / frame_rate)
        if decoded_audio is None:
            raise ValueError(f"Failed to decode audio from {audio_path}.")
        dump("a2v_audio_wave", decoded_audio.waveform[0])
        audio_conditioner = AudioConditioner(audio_vae_path, dtype, self.device)
        encoded_audio_latent = audio_conditioner(lambda enc: vae_encode_audio(decoded_audio, enc, None))
        audio_shape = AudioLatentShape.from_duration(
            batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16
        )
        encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
        # "b c t f -> b t (c f)": the DiT's audio tokens.
        dump("a2v_audio_latent", encoded_audio_latent[0].permute(1, 0, 2).reshape(encoded_audio_latent.shape[2], -1))
        print(
            f"[ltx25_a2v] {audio_path}: {decoded_audio.sampling_rate} Hz {list(decoded_audio.waveform.shape)} "
            f"-> latent {list(encoded_audio_latent.shape)}",
            flush=True,
        )

        def frozen_audio():
            return ModalitySpec(
                context=audio_context,
                frozen=True,
                noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            )

        # Stage 1 (DistilledPipeline), the audio frozen.
        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_w, stage_1_h = width // 2, height // 2
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=frozen_audio(),
            **self._stage_1_sampler_kwargs(seed),
        )

        # Stage 2 (DistilledPipeline), the audio frozen again.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=frozen_audio(),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
        # a2vid_two_stage.py: the original input audio, not the VAE-decoded one.
        original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)
        return decoded_video, original_audio, num_frames, tiling_config

    distilled.DistilledPipeline.__call__ = call
    print(f"[ltx25_a2v] DistilledPipeline.__call__ -> audio-to-video on {audio_path}", flush=True)
