#!/usr/bin/env python3
"""Rebuild a released single-file / native-layout safetensors file, byte for byte,
from the Diffusers-layout copy of the same weights already on the volume.

The upstream runtimes we benchmark load native checkpoints (SGLang's MiniMax-H3
`FL2VA` partition, Lightricks' single-file LTX-2.5 packs), while the volume
holds the Diffusers layout. The tensors are the same numbers under other names
(plus a few fused/reordered projections), so instead of downloading them again
we:

  1. fetch only the target file's safetensors header (an HTTP range read),
  2. write that exact header and, at each tensor's recorded offset, the bytes
     produced by a recipe over the local Diffusers tensors,
  3. sha256 the result and require it to equal the Hub's LFS oid.

Step 3 makes the conversion self-verifying: a wrong rename, fusion order or
dtype cannot pass. Tiny tensors with no Diffusers counterpart (e.g. a RoPE
buffer) are range-read from the Hub (`remote` recipe, capped in size).

Recipes (per target key):  copy <src> | qkv <q> <k> <v> <head_dim> |
swap_halves <src> | remote

Usage:
  reconstruct.py --plan h3_fl2va_dit --src /workspace/weights/h3-base \
      --repo MiniMaxAI/MiniMax-H3 --revision <sha> --out <dir>
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REMOTE_CAP = 64 << 20  # never range-read more than this per tensor
ELEM = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- hub
def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    for p in (os.environ.get("HF_HOME", "") + "/token", "/workspace/hf/token", os.path.expanduser("~/.cache/huggingface/token")):
        if p and os.path.isfile(p):
            return Path(p).read_text().strip()
    return None


def _curl(args: list[str]) -> bytes:
    tok = hf_token()
    hdr = ["-H", f"Authorization: Bearer {tok}"] if tok else []
    last = None
    for attempt in range(5):
        r = subprocess.run(["curl", "-sSfL", "--retry", "3", *hdr, *args], capture_output=True)
        if r.returncode == 0:
            return r.stdout
        last = r.stderr.decode(errors="replace")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"curl failed: {args[-1]}: {last}")


def resolve_url(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def range_read(repo: str, rev: str, path: str, start: int, end_excl: int) -> bytes:
    if end_excl <= start:
        return b""
    return _curl(["-r", f"{start}-{end_excl - 1}", resolve_url(repo, rev, path)])


def remote_header(repo: str, rev: str, path: str) -> tuple[bytes, dict]:
    n = struct.unpack("<Q", range_read(repo, rev, path, 0, 8))[0]
    raw = range_read(repo, rev, path, 0, 8 + n)
    return raw, json.loads(raw[8:])


def lfs_oid(repo: str, rev: str, path: str) -> tuple[str | None, int | None]:
    body = json.dumps({"paths": [path], "expand": True})
    out = _curl(["-X", "POST", "-H", "content-type: application/json", "-d", body,
                 f"https://huggingface.co/api/models/{repo}/paths-info/{rev}"])
    info = json.loads(out)
    for e in info:
        if e.get("path") == path:
            lfs = e.get("lfs") or {}
            return lfs.get("oid"), e.get("size")
    return None, None


# --------------------------------------------------------------------- local src
class Sources:
    """Every tensor in every *.safetensors under the given roots, by key."""

    def __init__(self, roots: list[Path], prefer: list[str] | None = None) -> None:
        self.meta: dict[str, tuple[Path, int, dict]] = {}
        self._maps: dict[Path, mmap.mmap] = {}
        self._ns: dict[Path, str] = {}
        files: list[Path] = []
        spec: list[tuple[str, Path]] = []
        for root in roots:
            ns, _, path = str(root).rpartition("=")
            spec.append((ns, Path(path)))
        for ns, root in spec:
            before = len(files)
            if root.is_file():
                files.append(root)
                self._ns.update({root: ns})
                continue
            # Prefer the file set an index.json names (dirs may hold stale shard sets).
            idx = sorted(root.glob("*.safetensors.index.json"))
            if idx:
                names = sorted(set(json.loads(idx[0].read_text())["weight_map"].values()))
                files += [root / n for n in names]
            else:
                files += sorted(root.glob("*.safetensors"))
            for f in files[before:]:
                self._ns[f] = ns
        for f in files:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            ns = self._ns.get(f, "")
            for k, v in h.items():
                k = f"{ns}:{k}" if ns else k
                if k in self.meta:
                    continue
                self.meta[k] = (f, 8 + n, v)

    def get(self, key: str) -> tuple[np.ndarray, dict]:
        f, base, v = self.meta[key]
        m = self._maps.get(f)
        if m is None:
            fh = open(f, "rb")
            m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps[f] = m
        a, b = v["data_offsets"]
        return np.frombuffer(m, dtype=np.uint8, count=b - a, offset=base + a), v


# ------------------------------------------------------------------------ plans
def _h3_dit_name(k: str) -> str:
    r = k
    if r.startswith("token_refiner.blocks."):
        r = r.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif r.startswith("blocks."):
        r = "transformer_blocks." + r[len("blocks."):]
    for a, b in ((".attn.q_norm.", ".attn.norm_q."), (".attn.k_norm.", ".attn.norm_k."),
                 (".attn.out_proj.", ".attn.to_out.0."), (".mlp.fc1.", ".ff.net.0.proj."),
                 (".mlp.fc2.", ".ff.net.2.")):
        r = r.replace(a, b)
    top = {"audio_patch_proj": "audio_proj_in", "video_patch_proj": "proj_in", "condition_proj": "context_embedder",
           "final_layer.adaln_proj.linear": "norm_out.linear", "final_layer.norm": "norm_out.norm",
           "final_layer.audio_out": "audio_proj_out", "final_layer.video_out": "proj_out",
           "time_embedder.proj_in": "time_embedder.linear_1", "time_embedder.proj_out": "time_embedder.linear_2"}
    for a, b in top.items():
        if r.startswith(a + "."):
            r = b + r[len(a):]
    return r


def plan_h3_fl2va_dit(key: str) -> tuple:
    """SGLang/native MiniMax-H3 DiT key -> Diffusers MiniMaxH3Transformer3DModel."""
    if key == "rope.inv_freq":
        return ("remote",)
    if key.endswith(".qkv_proj.weight"):
        base = _h3_dit_name(key)[: -len(".qkv_proj.weight")]
        return ("qkv", f"{base}.to_q.weight", f"{base}.to_k.weight", f"{base}.to_v.weight", 128)
    if key.endswith(".mlp.fc1.weight"):
        # native = [up; gate] where Diffusers stores [gate; up] (or vice versa)
        return ("swap_halves", _h3_dit_name(key))
    return ("copy", _h3_dit_name(key))


def plan_h3_fl2va_video_vae(key: str) -> tuple:
    """Native MiniMax-H3 video VAE (FL2VA/video_vae/source) -> Diffusers AutoencoderKLMiniMaxH3."""
    if key == "decoder.mask_token":
        return ("remote",)
    m = re.match(r"encoder\.down\.(\d+)\.block\.(\d+)\.(.*)$", key)
    if m:
        rest = m.group(3).replace("nin_shortcut", "conv_shortcut")
        return ("copy", f"encoder.down_blocks.{m.group(1)}.resnets.{m.group(2)}.{rest}")
    m = re.match(r"encoder\.down\.(\d+)\.downsample\.(.*)$", key)
    if m:
        return ("copy", f"encoder.down_blocks.{m.group(1)}.downsamplers.0.{m.group(2)}")
    if key.startswith("decoder.x_embedder."):
        return ("copy", key.replace("decoder.x_embedder.", "decoder.proj_in."))
    m = re.match(r"(decoder\.transformer_blocks\.\d+)\.(.*)$", key)
    if m:
        b, rest = m.groups()
        if rest.startswith("attn.to_qkv."):
            s = rest.split(".")[-1]
            return ("qkv", f"{b}.attn.to_q.{s}", f"{b}.attn.to_k.{s}", f"{b}.attn.to_v.{s}", 64)
        if rest.startswith("attn.to_out."):
            return ("copy", f"{b}.attn.to_out.0.{rest.split('.')[-1]}")
        if rest.startswith("ff.w1."):
            return ("swap_halves", f"{b}.ff.net.0.proj.{rest.split('.')[-1]}")
        if rest.startswith("ff.w2."):
            return ("copy", f"{b}.ff.net.2.{rest.split('.')[-1]}")
    return ("copy", key)


# ---- LTX-2.5 single-file packs (Lightricks/LTX-2.5) from LTX-2.5-Diffusers.
# Renames are the inverse of diffusers' scripts/convert_ltx2_to_diffusers.py
# (LTX 2.3/2.5 dicts); every tensor keeps its dtype and layout.
_LTX_TR = [("patchify_proj", "proj_in"), ("audio_patchify_proj", "audio_proj_in"),
           ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
           ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
           ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
           ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
           ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
           ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
           ("q_norm", "norm_q"), ("k_norm", "norm_k"),
           ("audio_prompt_adaln_single", "audio_prompt_adaln"), ("prompt_adaln_single", "prompt_adaln")]
_LTX_CONN = [("connectors.", ""), ("video_embeddings_connector", "video_connector"),
             ("audio_embeddings_connector", "audio_connector"), ("transformer_1d_blocks", "transformer_blocks"),
             ("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in"),
             ("text_embedding_projection.video_aggregate_embed", "video_text_proj_in"),
             ("q_norm", "norm_q"), ("k_norm", "norm_k")]
_LTX_VAE_BLOCKS = {"down_blocks.0": "down_blocks.0", "down_blocks.1": "down_blocks.0.downsamplers.0",
                   "down_blocks.2": "down_blocks.1", "down_blocks.3": "down_blocks.1.downsamplers.0",
                   "down_blocks.4": "down_blocks.2", "down_blocks.5": "down_blocks.2.downsamplers.0",
                   "down_blocks.6": "down_blocks.3", "down_blocks.7": "down_blocks.3.downsamplers.0",
                   "down_blocks.8": "mid_block", "up_blocks.0": "mid_block", "up_blocks.1": "up_blocks.0.upsamplers.0",
                   "up_blocks.2": "up_blocks.0", "up_blocks.3": "up_blocks.1.upsamplers.0", "up_blocks.4": "up_blocks.1",
                   "up_blocks.5": "up_blocks.2.upsamplers.0", "up_blocks.6": "up_blocks.2",
                   "up_blocks.7": "up_blocks.3.upsamplers.0", "up_blocks.8": "up_blocks.3"}
_LTX_STATS = [("per_channel_statistics.mean-of-means", "latents_mean"),
              ("per_channel_statistics.std-of-means", "latents_std")]


def _ren(k: str, rules) -> str:
    for a, b in rules:
        k = k.replace(a, b)
    return k


def plan_ltx25_transformer(key: str) -> tuple:
    k = key[len("model.diffusion_model."):]
    if k == "keyframes_abs_pos_embedding":
        return ("remote",)  # 8 KB; not carried by the Diffusers transformer
    if k.startswith(("video_embeddings_connector", "audio_embeddings_connector")):
        return ("copy", "C:" + _ren(k, _LTX_CONN))
    k = _ren(k, _LTX_TR)
    if k.startswith("adaln_single."):
        k = "time_embed." + k[len("adaln_single."):]
    elif k.startswith("audio_adaln_single."):
        k = "audio_time_embed." + k[len("audio_adaln_single."):]
    return ("copy", "T:" + k)


def plan_ltx25_text_encoder(key: str) -> tuple:
    if key.startswith("hf_asset__") or key == "tokenizer_json":
        return ("remote",)  # packed tokenizer / processor sidecars (U8, <= 32 MB)
    if key.startswith("text_embedding_projection."):
        return ("copy", "C:" + _ren(key, _LTX_CONN))
    for a, b in (("model.layers.", "model.language_model.layers."),
                 ("model.embed_tokens.", "model.language_model.embed_tokens."),
                 ("model.norm.", "model.language_model.norm."), ("vision_model.", "model.vision_embedder."),
                 ("multi_modal_projector.", "model.embed_vision."), ("audio_projector.", "model.embed_audio.")):
        if key.startswith(a):
            return ("copy", "G:" + b + key[len(a):])
    return ("copy", "G:" + key)


def plan_ltx25_video_vae(key: str) -> tuple:
    m = re.match(r"(encoder|decoder)\.((?:down|up)_blocks\.\d+)\.(.*)$", key)
    k = f"{m.group(1)}.{_LTX_VAE_BLOCKS[m.group(2)]}.{m.group(3)}" if m else key
    k = _ren(k, [("last_time_embedder", "time_embedder"), ("last_scale_shift_table", "scale_shift_table"),
                 ("res_blocks", "resnets"), *_LTX_STATS])
    return ("copy", k)


def plan_ltx25_audio_vae(key: str) -> tuple:
    if key.startswith("audio_vae."):
        return ("copy", "A:" + _ren(key[len("audio_vae."):], _LTX_STATS))
    k = _ren(key[len("vocoder."):], [("resblocks", "resnets"), ("conv_pre", "conv_in"), ("conv_post", "conv_out"),
                                      ("act_post", "act_out"), ("downsample.lowpass", "downsample")])
    return ("copy", "V:" + k.replace(".ups.", ".upsamplers."))


PLANS = {
    "ltx25_transformer": plan_ltx25_transformer,
    "ltx25_text_encoder": plan_ltx25_text_encoder,
    "ltx25_video_vae": plan_ltx25_video_vae,
    "ltx25_audio_vae": plan_ltx25_audio_vae,
    "ltx25_upsampler": lambda k: ("copy", k),
    "h3_fl2va_dit": plan_h3_fl2va_dit,
    "h3_fl2va_video_vae": plan_h3_fl2va_video_vae,
}


# ---------------------------------------------------------------------- recipes
def build_tensor(recipe: tuple, meta: dict, src: Sources, remote) -> bytes | np.ndarray:
    kind = recipe[0]
    nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
    if kind == "remote":
        if nbytes > REMOTE_CAP:
            raise RuntimeError(f"remote tensor too large ({nbytes} bytes)")
        return remote(meta)
    if kind == "copy":
        a, v = src.get(recipe[1])
        if v["dtype"] != meta["dtype"] and list(v["shape"]) == list(meta["shape"]) and nbytes <= REMOTE_CAP:
            # The Diffusers copy was saved in another dtype (e.g. BF16 for an F32
            # modulation table): the original bytes are not derivable, fetch them.
            return remote(meta)
        if v["dtype"] != meta["dtype"] or list(v["shape"]) != list(meta["shape"]):
            raise RuntimeError(f"copy {recipe[1]}: {v['dtype']}{v['shape']} != {meta['dtype']}{meta['shape']}")
        return a
    if kind == "swap_halves":
        a, v = src.get(recipe[1])
        if v["dtype"] != meta["dtype"] or list(v["shape"]) != list(meta["shape"]):
            raise RuntimeError(f"swap {recipe[1]}: shape/dtype mismatch")
        half = a.size // 2
        return np.concatenate([a[half:], a[:half]])
    if kind == "qkv":
        _, qk, kk, vk, hd = recipe
        parts = [src.get(x) for x in (qk, kk, vk)]
        rows = parts[0][1]["shape"][0]
        heads = rows // hd
        rowbytes = parts[0][0].size // rows
        stacked = np.stack([p[0].reshape(heads, hd * rowbytes) for p in parts], axis=1)
        out = stacked.reshape(-1)
        if out.size != nbytes:
            raise RuntimeError(f"qkv size {out.size} != {nbytes}")
        return out
    raise RuntimeError(f"unknown recipe {recipe}")


# The volume is shared with other pods: cap our read+write rate.
_THR = {"t0": None, "bytes": 0, "mbps": float(os.environ.get("RECON_MAX_MBPS", "250" if os.environ.get("UP_LOCAL") != "1" else "500"))}


def throttle(nbytes: int) -> None:
    if _THR["mbps"] <= 0:
        return
    if _THR["t0"] is None:
        _THR["t0"] = time.time()
    _THR["bytes"] += 2 * nbytes  # read + write
    ahead = _THR["bytes"] / (_THR["mbps"] * 1e6) - (time.time() - _THR["t0"])
    if ahead > 0:
        time.sleep(ahead)


def volume_used_bytes(root: str) -> int:
    """Bytes stored on the volume (du; df reports the whole shared filesystem)."""
    out = subprocess.run(["du", "-s", "--block-size=1", root], capture_output=True, text=True).stdout
    return int(out.split()[0]) if out.strip() else 0


def repair(tmp: Path, header: dict, hlen: int, remote) -> list:
    """Compare the head and tail of every tensor with the Hub; refetch small ones that differ."""
    def check(item):
        key, m = item
        a, b = m["data_offsets"]
        n = min(4096, b - a)
        with open(tmp, "rb") as fh:
            fh.seek(hlen + a)
            head = fh.read(n)
            fh.seek(hlen + b - n)
            tail = fh.read(n)
        want_h = remote({"data_offsets": [a, a + n]})
        want_t = remote({"data_offsets": [b - n, b]})
        return key, (head != want_h or tail != want_t)

    bad = []
    with cf.ThreadPoolExecutor(16) as ex:
        for key, differs in ex.map(check, header.items()):
            if differs:
                bad.append(key)
    out = []
    with open(tmp, "r+b") as fh:
        for key in bad:
            m = header[key]
            a, b = m["data_offsets"]
            if b - a <= REMOTE_CAP:
                fh.seek(hlen + a)
                fh.write(remote(m))
                out.append([key, b - a, "refetched"])
            else:
                out.append([key, b - a, "too large to refetch"])
    return out


def reconstruct(repo: str, rev: str, path: str, out: Path, plan, src: Sources, verify: bool = True) -> dict:
    t0 = time.time()
    raw, header = remote_header(repo, rev, path)
    meta = header.pop("__metadata__", None)
    oid, size = lfs_oid(repo, rev, path)
    hlen = len(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and size and out.stat().st_size == size and (out.with_suffix(out.suffix + ".sha256")).exists():
        have = out.with_suffix(out.suffix + ".sha256").read_text().strip()
        if have == oid:
            log(f"{path}: already reconstructed and verified")
            return {"path": path, "ok": True, "cached": True}

    def remote(m: dict) -> bytes:
        a, b = m["data_offsets"]
        return range_read(repo, rev, path, hlen + a, hlen + b)

    quota = float(os.environ.get("VOLUME_QUOTA_GB", "1000")) * 1e9
    reserve = float(os.environ.get("VOLUME_RESERVE_GB", "60")) * 1e9
    vroot = os.environ.get("VOLUME_ROOT", "/workspace" if str(out).startswith("/workspace") else "")
    if vroot and os.path.isdir(vroot) and size:
        used = volume_used_bytes(vroot)
        if used + size > quota - reserve:
            raise RuntimeError(f"not enough volume space: used {used / 1e9:.0f} GB + {size / 1e9:.0f} GB "
                               f"> {quota / 1e9:.0f} GB quota - {reserve / 1e9:.0f} GB reserve")
        log(f"{path}: volume used {used / 1e9:.0f} GB, writing {size / 1e9:.1f} GB")
    tmp = out.with_suffix(out.suffix + ".partial")
    total = hlen + max(v["data_offsets"][1] for v in header.values())
    missing = []
    with open(tmp, "wb") as fh:
        fh.truncate(total)
        fh.seek(0)
        fh.write(raw)
        for key, m in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            recipe = plan(key)
            try:
                data = build_tensor(recipe, m, src, remote)
            except KeyError as e:
                missing.append(f"{key} <- {recipe} (no source {e})")
                continue
            fh.seek(hlen + m["data_offsets"][0])
            fh.write(data.tobytes() if isinstance(data, np.ndarray) else data)
            throttle(m["data_offsets"][1] - m["data_offsets"][0])
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} tensors without a source, e.g. {missing[:5]}")
    res = {"path": path, "bytes": total, "expected_size": size, "expected_oid": oid, "metadata": meta,
           "build_s": round(time.time() - t0, 1)}
    if verify:
        h = hashlib.sha256()
        with open(tmp, "rb") as fh:
            while True:
                b = fh.read(64 << 20)
                if not b:
                    break
                h.update(b)
        res["sha256"] = h.hexdigest()
        res["ok"] = res["sha256"] == oid
        if not res["ok"]:
            log(f"{path}: SHA MISMATCH {res['sha256']} != {oid}; sampling every tensor against the Hub")
            res["diff"] = repair(tmp, header, hlen, remote)
            h = hashlib.sha256()
            with open(tmp, "rb") as fh:
                while True:
                    b = fh.read(64 << 20)
                    if not b:
                        break
                    h.update(b)
            res["sha256"] = h.hexdigest()
            res["ok"] = res["sha256"] == oid
            if not res["ok"]:
                log(f"{path}: still mismatched after repair: {res['diff'][:20]}")
                return res
            log(f"{path}: repaired {len(res['diff'])} tensors from the Hub")
    os.replace(tmp, out)
    if verify:
        out.with_suffix(out.suffix + ".sha256").write_text(res["sha256"] + "\n")
    res["total_s"] = round(time.time() - t0, 1)
    log(f"{path}: ok ({total / 1e9:.1f} GB, {res['total_s']} s)")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=sorted(PLANS))
    ap.add_argument("--src", required=True, action="append", help="source dir/file (repeatable)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--files", nargs="+", required=True, help="target paths inside the repo")
    ap.add_argument("--out-root", required=True, help="written as <out-root>/<file path minus --strip>")
    ap.add_argument("--strip", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    src = Sources([Path(s) for s in args.src])
    log(f"{len(src.meta)} source tensors")
    plan = PLANS[args.plan]

    def one(f: str) -> dict:
        rel = f[len(args.strip):] if args.strip and f.startswith(args.strip) else f
        try:
            return reconstruct(args.repo, args.revision, f, Path(args.out_root) / rel, plan, src)
        except Exception as e:  # noqa: BLE001
            log(f"{f}: FAILED {e}")
            return {"path": f, "ok": False, "error": str(e)}

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(one, args.files))
    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2))
    bad = [r for r in results if not r.get("ok")]
    log(f"{len(results) - len(bad)}/{len(results)} files verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
