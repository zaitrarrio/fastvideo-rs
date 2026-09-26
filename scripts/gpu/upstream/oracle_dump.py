#!/usr/bin/env python3
"""GPU oracle dumps from the Python references, in fastvideo-rs's dump format.

Writes exactly what our ``FASTVIDEO_DUMP_DIR`` writes
(crates/fastvideo-cudarc/src/wan/dump.rs): ``<dir>/<name>.f32`` (raw
little-endian float32, a bf16 tensor widened exactly) plus ``<dir>/<name>.shape``
(space-separated dims), under the same names and with the same row subsampling,
so ``fv-gpucheck compare-dumps`` diffs a reference run against ours tensor by
tensor. The reference's own inputs (initial noise, text hidden states, per-step
noise of ancestral samplers) are dumped under the names our
``FASTVIDEO_INJECT_DIR`` reads back, so the Rust run starts from torch's draws.

Activated by ``FV_ORACLE_DUMP_DIR``. ``install()`` registers post-import
patches, so it works whether the reference modules are imported before or
after it, and in every process: ``oracle_site/sitecustomize.py`` calls it at
interpreter start when the variable is set (FastVideo runs the DiT in
multiprocessing workers). Nothing is patched when the variable is unset.

Targets
-------
* FastVideo MiniMax-H3 (FastH3 8-step V2, FastH3 Preview LoRAs):
  ``MiniMaxH3DenoisingStage.forward`` (inputs, both schedulers' steps),
  ``MiniMaxH3Transformer3DModel`` (refined text, first-step block outputs,
  the inside of ``FASTVIDEO_DUMP_OPS`` blocks). Run the strict eager route
  (``--profile strict --no-inference-torch-compile``): hooks inside a
  fullgraph-compiled block would break its graph, and the H3 fusions are
  FastVideo's own report-only reorderings.
* LTX-2 (``ltx_pipelines`` / ``ltx_core``, driven by sol-engine's LTX-2.5
  RTX5090 ``gpu_infer.py``): see ``_patch_ltx_*`` below.

Names (step numbers are 1-based after the step, as ours):
  text_hidden, text_refined, video_step00_in, audio_step00_in,
  {video,audio}_{sigmas,timesteps}, {video,audio}_vel_stepNN, {video,audio}_stepNN,
  step00_packed_in, rope_cos, rope_sin, step00_block_<i>,
  step00_b<i>_{adaln,attn_in,attn_out,resid_msa,ffn_in,ffn_out}
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
from pathlib import Path

BLOCK_ROW_STRIDE = 64  # dump.rs BLOCK_ROW_STRIDE


def _dir() -> Path | None:
    d = os.environ.get("FV_ORACLE_DUMP_DIR", "")
    return Path(d) if d else None


def op_blocks() -> list[int]:
    raw = os.environ.get("FASTVIDEO_DUMP_OPS", "0")
    out = []
    for s in raw.split(","):
        s = s.strip()
        if s.isdigit():
            out.append(int(s))
    return out


def _rank0() -> bool:
    for k in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v not in (None, "", "0"):
            return False
    return True


_LOG = []


def _note(msg: str) -> None:
    print(f"[oracle_dump] {msg}", file=sys.stderr, flush=True)
    _LOG.append(msg)


def write(name: str, t) -> None:
    """The whole tensor as float32 (bf16/fp16 widen exactly)."""
    d = _dir()
    if d is None or not _rank0():
        return
    import numpy as np
    import torch

    if isinstance(t, torch.Tensor):
        a = t.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    else:
        a = np.asarray(t, dtype=np.float32)
    d.mkdir(parents=True, exist_ok=True)
    a.astype("<f4", copy=False).tofile(d / f"{name}.f32")
    (d / f"{name}.shape").write_text(" ".join(str(int(x)) for x in a.shape))
    _meta_add(name, list(a.shape), str(getattr(t, "dtype", "f32")))


def rows_strided(name: str, t, stride: int = BLOCK_ROW_STRIDE) -> None:
    """Every ``stride``-th row of ``t`` viewed as ``[rows, last_dim]`` (dump.rs rows_strided)."""
    if _dir() is None:
        return
    w = t.shape[-1]
    write(name, t.reshape(-1, w)[::stride])


_META: dict = {"tensors": {}}
_META_LOCK = threading.Lock()


def _meta_add(name: str, shape: list, dtype: str) -> None:
    with _META_LOCK:
        _META["tensors"][name] = {"shape": shape, "dtype": dtype}
        d = _dir()
        if d is not None:
            (d / "oracle_meta.json").write_text(json.dumps(_META, indent=1, default=str))


def meta(**kv) -> None:
    with _META_LOCK:
        _META.update(kv)


# --------------------------------------------------------------- import hooks
class _PatchLoader(importlib.abc.Loader):
    def __init__(self, inner, name, patch):
        self.inner, self.name, self.patch = inner, name, patch

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            self.patch(module)
            _note(f"patched {self.name}")
        except Exception as e:  # noqa: BLE001
            _note(f"patch of {self.name} FAILED: {type(e).__name__}: {e}")
            raise


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, targets):
        self.targets = targets

    def find_spec(self, name, path, target=None):
        patch = self.targets.get(name)
        if patch is None:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None:
            return None
        spec.loader = _PatchLoader(spec.loader, name, patch)
        return spec


_INSTALLED = False


def install() -> None:
    """Register the patches (idempotent; no-op without FV_ORACLE_DUMP_DIR)."""
    global _INSTALLED
    if _INSTALLED or _dir() is None:
        return
    _INSTALLED = True
    targets = {
        "fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising": _patch_fv_denoise,
        "fastvideo.models.dits.minimax_h3": _patch_fv_dit,
    }
    targets.update(_LTX_TARGETS)
    for name, patch in list(targets.items()):
        if name in sys.modules:  # imported before install()
            patch(sys.modules[name])
            _note(f"patched {name} (already imported)")
            targets.pop(name)
    sys.meta_path.insert(0, _Finder(targets))
    meta(pid_start=os.getpid(), started=time.time(), argv=sys.argv, op_blocks=op_blocks())


# ------------------------------------------------------------------ state
class _State:
    """Which step the reference is on. ``step`` is 0 during the first forward."""

    def __init__(self):
        self.active = False
        self.done = False
        self.step = 0


S = _State()


# ------------------------------------------------------------- FastVideo H3
def _patch_fv_denoise(mod) -> None:
    stage_cls = mod.MiniMaxH3DenoisingStage
    orig_forward = stage_cls.forward

    def forward(self, batch, fastvideo_args):
        if S.done or S.active:
            return orig_forward(self, batch, fastvideo_args)
        S.active, S.step = True, 0
        write("video_step00_in", batch.latents)
        write("audio_step00_in", batch.audio_latents)
        write("text_hidden", batch.prompt_embeds[0])
        layout = batch.extra.get("minimax_h3_layout")
        if layout is not None:
            meta(layout={
                "sequence_length": int(layout.sequence_length),
                "text": int(layout.text_indices.numel()),
                "audio": int(layout.audio_indices.numel()),
                "video": int(layout.video_indices.numel()),
                "num_condition_video_rows": int(layout.num_condition_video_rows),
            })
        meta(env={k: v for k, v in os.environ.items() if k.startswith("FASTVIDEO_")},
             vsa_sparsity=getattr(batch, "VSA_sparsity", None))
        wrapped = []
        for sched, tag in ((self.scheduler, "video"), (self.audio_scheduler, "audio")):
            wrapped.append(sched)
            sched.step = _wrap_h3_step(sched, tag)
        try:
            return orig_forward(self, batch, fastvideo_args)
        finally:
            for sched in wrapped:
                sched.__dict__.pop("step", None)
            S.active, S.done = False, True
            _note(f"denoise done after {S.step} steps")

    stage_cls.forward = forward


def _wrap_h3_step(sched, tag):
    bound = type(sched).step
    n = [0]

    def step(model_output, timestep, sample, return_dict=True, **kw):
        out = bound(sched, model_output, timestep, sample, return_dict=return_dict, **kw)
        if n[0] == 0:
            write(f"{tag}_sigmas", sched.sigmas)
            write(f"{tag}_timesteps", sched.timesteps)
        n[0] += 1
        prev = out[0] if isinstance(out, tuple) else out.prev_sample
        write(f"{tag}_vel_step{n[0]:02d}", model_output)
        write(f"{tag}_step{n[0]:02d}", prev)
        if tag == "video":
            S.step = n[0]
        return out

    return step


def _patch_fv_dit(mod) -> None:
    import torch

    cls = mod.MiniMaxH3Transformer3DModel
    orig_forward = cls.forward
    orig_refined = cls._refined_text

    def _refined_text(self, encoder_hidden_states):
        out = orig_refined(self, encoder_hidden_states)
        if S.active and S.step == 0:
            write("text_refined", out)
        return out

    def forward(self, *args, **kwargs):
        if not (S.active and S.step == 0):
            return orig_forward(self, *args, **kwargs)
        handles = []
        ops = set(op_blocks())
        for i, block in enumerate(self.transformer_blocks):
            handles.append(block.register_forward_hook(
                lambda m, a, out, i=i: rows_strided(f"step00_block_{i}", out)))
            if i == 0:
                def pre0(m, a):
                    hs, rot = a[0], a[3]
                    rows_strided("step00_packed_in", hs)
                    rows_strided("rope_cos", rot[0])
                    rows_strided("rope_sin", rot[1])
                handles.append(block.register_forward_pre_hook(pre0))
            if i in ops:
                handles += _h3_op_hooks(block, i, torch)
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            for h in handles:
                h.remove()

    cls.forward = forward
    cls._refined_text = _refined_text


def _h3_op_hooks(block, i, torch):
    """Inside block i (strict, unfused path): the six AdaLN vectors gathered per
    row, the modulated norms feeding attention and FFN, both branch outputs, and
    the post-attention residual recomputed exactly as the block does."""
    ctx: dict = {}
    idx_rows = slice(None, None, BLOCK_ROW_STRIDE)

    def pre(m, a):
        ctx["hs"], ctx["idx"] = a[0], a[2]

    def ada(m, a, out):
        ctx["ada"] = out
        idx = ctx["idx"][idx_rows]
        write(f"step00_b{i}_adaln", torch.cat([t.index_select(0, idx) for t in out], dim=-1))

    def attn_pre(m, a):
        rows_strided(f"step00_b{i}_attn_in", a[0])

    def attn_out(m, a, out):
        rows_strided(f"step00_b{i}_attn_out", out)
        hs = ctx["hs"]
        gate = ctx["ada"][2].to(hs.dtype)
        resid = hs + gate.index_select(0, ctx["idx"]) * out
        rows_strided(f"step00_b{i}_resid_msa", resid)

    def ff_pre(m, a):
        rows_strided(f"step00_b{i}_ffn_in", a[0])

    def ff_out(m, a, out):
        rows_strided(f"step00_b{i}_ffn_out", out)

    return [
        block.register_forward_pre_hook(pre),
        block.adaln_proj.register_forward_hook(ada),
        block.attn.register_forward_pre_hook(attn_pre),
        block.attn.register_forward_hook(attn_out),
        block.ff.register_forward_pre_hook(ff_pre),
        block.ff.register_forward_hook(ff_out),
    ]


# ------------------------------------------------------------------ LTX-2
# ltx_pipelines DistilledPipeline (LTX-2.5 distilled two-stage, as sol-engine's
# RTX5090 gpu_infer.py drives it). Names carry the stage: s1_ (half-resolution
# ancestral stage), s2_ (full-resolution Euler refine); see ltx2/pipeline.rs.
#   text_{video,audio}_ctx                PromptEncoder output (the DiT's contexts)
#   noise_seed<seed>_<k>                  k-th torch.randn on the generator seeded <seed>
#                                         (initial noise, ancestral per-step noise,
#                                         stage-2 renoise), replayed by our NoiseStream
#   s{n}_sigmas, s{n}_{video,audio}_step00_in, s{n}_{video,audio}_{vel,x0}_stepNN,
#   s{n}_{video,audio}_stepNN, s{n}_step00_{video,audio}_in,
#   s{n}_step00_{video,audio}_block_<i>   (video every 64th row, audio whole)
#   s2_upsampled (packed, every 64th row), s2_entry_{video,audio}
class _Ltx:
    stage = 0
    step = -1
    in_stage = False
    draws: dict = {}


def _ltx_pack5(t):
    """[1, C, F, H, W] -> [F*H*W, C] (the patch-1 token order)."""
    return t[0].permute(1, 2, 3, 0).reshape(-1, t.shape[1])


def _patch_ltx_blocks(mod) -> None:
    import torch

    orig_randn = torch.randn

    def randn(*a, **kw):
        out = orig_randn(*a, **kw)
        g = kw.get("generator")
        if g is not None and _Ltx.in_stage:
            seed = int(g.initial_seed())
            k = _Ltx.draws.get(seed, 0)
            _Ltx.draws[seed] = k + 1
            write(f"noise_seed{seed}_{k:03d}", out)
        return out

    torch.randn = randn

    stage_cls = mod.DiffusionStage
    orig_call = stage_cls.__call__

    def call(self, *args, **kwargs):
        _Ltx.stage += 1
        n = _Ltx.stage
        p = f"s{n}_"
        inner = kwargs.get("loop") or mod.euler_denoising_loop

        def loop(*la, **lk):
            den = lk["denoiser"]
            sig = lk["sigmas"]

            def denoiser(transformer, vs, as_, sigmas, step_idx):
                _Ltx.step = step_idx
                tag = "step00_in" if step_idx == 0 else f"step{step_idx:02d}"
                if step_idx == 0:
                    write(p + "sigmas", sigmas)
                    if n == 2:
                        write("s2_entry_video", vs.latent)
                        write("s2_entry_audio", as_.latent)
                write(f"{p}video_{tag}", vs.latent)
                write(f"{p}audio_{tag}", as_.latent)
                vr, ar = den(transformer, vs, as_, sigmas, step_idx)
                if vr is not None:
                    write(f"{p}video_x0_step{step_idx + 1:02d}", vr.denoised)
                if ar is not None:
                    write(f"{p}audio_x0_step{step_idx + 1:02d}", ar.denoised)
                return vr, ar

            lk["denoiser"] = denoiser
            vs, as_ = inner(*la, **lk)
            last = len(sig) - 1
            write(f"{p}video_step{last:02d}", vs.latent)
            write(f"{p}audio_step{last:02d}", as_.latent)
            return vs, as_

        kwargs["loop"] = loop
        _Ltx.in_stage, _Ltx.step = True, -1
        try:
            return orig_call(self, *args, **kwargs)
        finally:
            _Ltx.in_stage = False
            _note(f"ltx stage {n} done")

    stage_cls.__call__ = call

    enc_cls = mod.PromptEncoder
    orig_enc = enc_cls.__call__

    def enc(self, *args, **kwargs):
        out = orig_enc(self, *args, **kwargs)
        if not getattr(_Ltx, "text_done", False):
            _Ltx.text_done = True
            write("text_video_ctx", out[0].video_encoding)
            write("text_audio_ctx", out[0].audio_encoding)
            try:
                # The connector output mask is all ones (registers fill the pads):
                # the real rows are the tokenizer mask's count, front-aligned.
                n = int(getattr(_Ltx, "text_real", 0)) or int(out[0].attention_mask[0].sum())
                write("text_video_ctx_real", out[0].video_encoding[0, :n])
                write("text_audio_ctx_real", out[0].audio_encoding[0, :n])
            except Exception as e:  # noqa: BLE001
                _note(f"text ctx rows: {type(e).__name__}: {e}")
        return out

    enc_cls.__call__ = enc

    up_cls = mod.VideoUpsampler
    orig_up = up_cls.__call__

    def up(self, latent):
        out = orig_up(self, latent)
        rows_strided("s2_upsampled", _ltx_pack5(out))
        return out

    up_cls.__call__ = up


def _patch_ltx_model(mod) -> None:
    cls = mod.LTXModel
    orig_forward = cls.forward

    def forward(self, video, audio, perturbations):
        n, k = _Ltx.stage, _Ltx.step
        p = f"s{n}_"
        handles = []
        if _Ltx.in_stage and k == 0:
            for i, block in enumerate(self.transformer_blocks):
                def hook(m, a, out, i=i):
                    v, au = out
                    if v is not None:
                        rows_strided(f"{p}step00_video_block_{i}", v.x)
                    if au is not None:
                        rows_strided(f"{p}step00_audio_block_{i}", au.x, 1)
                handles.append(block.register_forward_hook(hook))

            def pre0(m, a, kw):
                v, au = kw.get("video"), kw.get("audio")
                if v is not None:
                    rows_strided(f"{p}step00_video_in", v.x)
                if au is not None:
                    rows_strided(f"{p}step00_audio_in", au.x, 1)
            handles.append(self.transformer_blocks[0].register_forward_pre_hook(pre0, with_kwargs=True))
        try:
            vx, ax = orig_forward(self, video, audio, perturbations)
        finally:
            for h in handles:
                h.remove()
        if _Ltx.in_stage and k >= 0:
            if vx is not None:
                write(f"{p}video_vel_step{k + 1:02d}", vx)
            if ax is not None:
                write(f"{p}audio_vel_step{k + 1:02d}", ax)
        return vx, ax

    cls.forward = forward


# The text path, stage by stage (the first prompt of the first encode only),
# every tensor restricted to the real (mask == 1) tokens in order, as ours:
#   text_input_ids, text_attention_mask   [1024] (the tokenizer's, left-padded)
#   text_hidden_<k>                       [n, 3840] Gemma hidden state k (HF numbering:
#                                         0 = scaled embeddings, 48 = normed last)
#   text_{video,audio}_feats              [n, D] FeatureExtractor output (the aggregate
#                                         embeds, before the connectors)
#   text_{video,audio}_ctx_real           [n, D] connector output rows of the real
#                                         tokens (front-aligned after the right-pad sort)
# plus oracle_meta.json "gemma": layer_scalar values, norm-weight means, the
# attention implementation, the full-attention RoPE frequencies.
TEXT_TAPS = (0, 1, 2, 5, 6, 7, 12, 24, 36, 47, 48)


def _patch_ltx_gemma_encoder(mod) -> None:
    cls = mod.LTXGemmaTextEncoder
    orig = cls.encode

    def encode(self, prompts):
        out = orig(self, prompts)
        if getattr(_Ltx, "hidden_done", False) or not out:
            return out
        _Ltx.hidden_done = True
        try:
            toks = self.tokenizer.tokenize_with_weights(prompts[0])["gemma"]
            write("text_input_ids", [int(t) for t, _ in toks])
            write("text_attention_mask", [int(w) for _, w in toks])
        except Exception as e:  # noqa: BLE001
            _note(f"text ids: {type(e).__name__}: {e}")
        hs, mask = out[0]
        keep = mask[0].bool()
        for k in TEXT_TAPS:
            if k < len(hs):
                write(f"text_hidden_{k}", hs[k][0][keep])
        _Ltx.text_real = int(keep.sum())
        info: dict = {"num_hidden_states": len(hs), "real_tokens": _Ltx.text_real}
        try:
            lm = self.model.model.language_model
            info["attn_implementation"] = str(getattr(self.model.config, "_attn_implementation", None))
            info["layer_scalar"] = [float(l.layer_scalar.float().reshape(-1)[0]) for l in lm.layers]
            l0 = lm.layers[0]
            info["norm_weight_mean"] = {
                n: float(getattr(l0, n).weight.float().mean())
                for n in ("input_layernorm", "post_attention_layernorm",
                          "pre_feedforward_layernorm", "post_feedforward_layernorm")
            }
            info["q_norm_mean"] = float(l0.self_attn.q_norm.weight.float().mean())
            info["final_norm_mean"] = float(lm.norm.weight.float().mean())
            info["embed_scale"] = float(lm.embed_tokens.embed_scale.to(lm.embed_tokens.weight.dtype))
            info["layer5_has_v_proj"] = lm.layers[5].self_attn.v_proj is not None
            info["layer0_has_v_proj"] = lm.layers[0].self_attn.v_proj is not None
            info["scaling"] = float(l0.self_attn.scaling)
            re = lm.rotary_emb
            info["full_inv_freq_head"] = [float(x) for x in re.full_attention_inv_freq[:4]]
            info["full_inv_freq_len"] = int(re.full_attention_inv_freq.numel())
            info["full_inv_freq_nonzero"] = int((re.full_attention_inv_freq != 0).sum())
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"
        meta(gemma=info)
        _note(f"gemma: {info}")
        return out

    cls.encode = encode


def _patch_ltx_feature_extractor(mod) -> None:
    for name in ("FeatureExtractorV1", "FeatureExtractorV2"):
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        orig = cls.forward

        def forward(self, hidden_states, attention_mask, *a, _orig=orig, **kw):
            v, au = _orig(self, hidden_states, attention_mask, *a, **kw)
            if not getattr(_Ltx, "feats_done", False):
                _Ltx.feats_done = True
                keep = attention_mask[0].bool()
                write("text_video_feats", v[0][keep])
                if au is not None:
                    write("text_audio_feats", au[0][keep])
            return v, au

        cls.forward = forward


_LTX_TARGETS: dict = {
    "ltx_pipelines.utils.blocks": _patch_ltx_blocks,
    "ltx_core.model.transformer.model": _patch_ltx_model,
    "ltx_core.text_encoders.gemma.encoders.base_encoder": _patch_ltx_gemma_encoder,
    "ltx_core.text_encoders.gemma.feature_extractor": _patch_ltx_feature_extractor,
}
