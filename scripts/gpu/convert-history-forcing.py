#!/usr/bin/env python3
"""Convert History Forcing's `dmd/model.pt` to safetensors, without torch.

    python3 -I scripts/gpu/convert-history-forcing.py <dmd/model.pt> <out dir> \
        [--revision <hub rev>] [--expect-sha256 <hex>] [--sub generator_ema|generator]

<dmd/model.pt> is the Hub's `xzxxntxdy/History-Forcing` `dmd/model.pt`
(11 352 569 509 bytes at 8604aafe4fada0311c7f5a993e7e620473c8e7ac): a training
checkpoint with the Wan2.1-T2V-1.3B generator twice, `generator` (raw) and
`generator_ema` (what the released inference runs, `inference.yaml`
`weights: ema`), f32, original Wan names behind FSDP wrappers
(`model._fsdp_wrapped_module.blocks.0.self_attn.q.weight`), each tensor
saved with attributes (`torch._tensor._rebuild_from_type_v2`). Ported from
fastwan-rs `scripts/convert_self_forcing.py`.

Writes, into <out dir> (which must not exist: written as
<out>.partial-<stamp> and renamed at the end, add-only):

- generator_ema.safetensors: the `generator_ema` sub-dict (`--sub generator`:
  the raw one, as generator.safetensors), wrappers stripped but the ORIGINAL
  Wan names kept (`blocks.0.self_attn.q.weight`, `head.modulation`, ...):
  crates/fastvideo-cudarc/src/wan/longlive.rs renames to Diffusers at load,
  as it does for LongLive's longlive_base.safetensors. Dtypes as stored (f32
  for this checkpoint; the Rust loader narrows 2-D weights to bf16 at load).
  Header metadata: source file and sub-dict, the source's SHA-256, the Hub
  repo and revision, the licence note ("none stated: evaluation only") and
  this converter's name;
- keys.txt (key, shape, dtype), sha256.txt (the output and the source),
  .complete.

The pickle is read with an unpickler that resolves only tensor rebuilds,
storage types and OrderedDict and refuses everything else; tensor bytes are
streamed from the zip in 16 MiB chunks, so memory stays small whatever the
checkpoint's size. The written file is read back and every tensor compared
with the source bytes before the rename.
"""
import collections
import hashlib
import io
import json
import pickle
import struct
import sys
import time
import zipfile
from pathlib import Path

HUB_REPO = "xzxxntxdy/History-Forcing"
HUB_FILE = "dmd/model.pt"
LICENCE = "none stated on the Hub card or in the repo (evaluation only)"
DTYPES = {"BFloat16Storage": ("BF16", 2), "HalfStorage": ("F16", 2), "FloatStorage": ("F32", 4)}
# The wrappers a Self-Forcing / History Forcing state dict carries, stripped
# per key as many times as they apply (longlive.rs strip_wrappers).
WRAPPERS = ["base_model.model.", "_fsdp_wrapped_module.", "model.", "module.", "generator."]
CHUNK = 16 << 20


class Lazy:
    """A tensor described by its storage; bytes are read on demand."""

    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)
        self.numel = 1
        for d in self.size:
            self.numel *= d


def rebuild(storage, offset, size, stride, *_):
    exp, acc = [], 1
    for d in reversed(size):
        exp.insert(0, acc)
        acc *= d
    if list(stride) != exp and any(s > 1 for s in size):
        raise ValueError(f"non-contiguous tensor {size} {stride}")
    return Lazy(storage, offset, size, stride)


TENSOR = object()  # stands in for the torch.Tensor class


def rebuild_from_type(func, new_type, args, state):
    # A tensor saved with Python attributes: the plain tensor, attributes
    # dropped. Only plain Tensor is accepted.
    if func is not rebuild or new_type is not TENSOR:
        raise pickle.UnpicklingError("unexpected tensor subclass")
    return rebuild(*args)


def rebuild_parameter(data, requires_grad, hooks, *_):
    if not isinstance(data, Lazy):
        raise pickle.UnpicklingError("parameter without a tensor")
    return data


class Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild
        if (mod, name) == ("torch._tensor", "_rebuild_from_type_v2"):
            return rebuild_from_type
        if (mod, name) == ("torch._utils", "_rebuild_parameter"):
            return rebuild_parameter
        if (mod, name) == ("torch", "Tensor"):
            return TENSOR
        if mod == "torch" and name in DTYPES:
            return name
        raise pickle.UnpicklingError(f"refusing global {mod}.{name}")

    def persistent_load(self, pid):
        kind, cls, key, _loc, _n = pid
        if kind != "storage" or cls not in DTYPES:
            raise pickle.UnpicklingError(f"unexpected storage {pid}")
        return (cls, key)


def strip_wrappers(key):
    while True:
        before = key
        for p in WRAPPERS:
            if key.startswith(p):
                key = key[len(p):]
        if key == before:
            return key


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tensor(z, prefix, t, dst, h):
    """Append tensor `t`'s bytes from the zip to `dst` (and hash `h`), chunked."""
    nb = DTYPES[t.storage[0]][1]
    left = t.numel * nb
    with z.open(f"{prefix}/data/{t.storage[1]}") as s:
        s.seek(t.offset * nb)
        while left:
            b = s.read(min(CHUNK, left))
            if not b:
                raise SystemExit(f"storage {t.storage[1]}: short read")
            dst.write(b)
            h.update(b)
            left -= len(b)


def same_bytes(z, prefix, t, f, start, end):
    """The written bytes [start, end) of `f` equal tensor `t`'s source bytes."""
    nb = DTYPES[t.storage[0]][1]
    left = end - start
    f.seek(start)
    with z.open(f"{prefix}/data/{t.storage[1]}") as s:
        s.seek(t.offset * nb)
        while left:
            n = min(CHUNK, left)
            if f.read(n) != s.read(n):
                return False
            left -= n
    return True


def main(argv):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="the checkpoint (dmd/model.pt)")
    ap.add_argument("out", help="the output directory (must not exist)")
    ap.add_argument("--revision", default="unpinned", help="the Hub revision the checkpoint was fetched at (metadata)")
    ap.add_argument("--expect-sha256", help="refuse a source whose SHA-256 differs (the Hub's LFS hash)")
    ap.add_argument("--sub", default="generator_ema", choices=("generator_ema", "generator"),
                    help="the sub-dict to write (default generator_ema; falls back to generator when absent)")
    a = ap.parse_args(argv)
    src, out, sub = Path(a.src), Path(a.out), a.sub
    opts = {"revision": a.revision, "expect-sha256": a.expect_sha256}
    if out.exists():
        raise SystemExit(f"{out} exists: add-only, not touching it")
    t0 = time.time()
    print(f"sha256 of {src} ...", flush=True)
    src_sha = sha256_file(src)
    want = opts.get("expect-sha256")
    if want and want.lower() != src_sha:
        raise SystemExit(f"{src}: sha256 {src_sha}, expected {want}")
    print(f"  {src_sha}  {src.name} ({src.stat().st_size} bytes, {time.time() - t0:.0f} s)", flush=True)

    z = zipfile.ZipFile(src)
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    prefix = pkl[: -len("/data.pkl")]
    obj = Unpickler(io.BytesIO(z.read(pkl))).load()
    if not isinstance(obj, dict):
        raise SystemExit(f"{src}: the root is a {type(obj).__name__}, not a dict")
    print("top-level keys:", list(obj)[:10], flush=True)
    picked = sub
    if not isinstance(obj.get(picked), dict):
        picked = "generator" if sub == "generator_ema" and isinstance(obj.get("generator"), dict) else None
        if picked is None:
            raise SystemExit(f"{src}: no `{sub}` sub-dict (keys {list(obj)[:10]})")
        print(f"no `{sub}`: falling back to `{picked}`", flush=True)
    sd = obj[picked]
    state, skipped = {}, []
    for k, v in sd.items():
        if isinstance(v, Lazy):
            nk = strip_wrappers(k)
            if nk in state:
                raise SystemExit(f"{k}: `{nk}` twice after stripping the wrappers")
            state[nk] = v
        else:
            skipped.append(k)
    if not state:
        raise SystemExit(f"{src}: `{picked}` holds no tensors")
    names = sorted(state)
    name = f"{picked}.safetensors"

    tmp = out.parent / f".{out.name}.partial-{time.strftime('%Y%m%d%H%M%S')}"
    tmp.mkdir(parents=True)
    # The Hub path, not the local file name, so the output (and its SHA-256)
    # is the same wherever the checkpoint was saved.
    header = {
        "__metadata__": {
            "converter": "fastvideo-rs scripts/gpu/convert-history-forcing.py",
            "hub_repo": HUB_REPO,
            "hub_revision": opts.get("revision", "unpinned"),
            "licence": LICENCE,
            "source": f"{HUB_FILE}#{picked}",
            "source_sha256": src_sha,
        }
    }
    off = 0
    for k in names:
        t = state[k]
        dt, nb = DTYPES[t.storage[0]]
        header[k] = {"dtype": dt, "shape": list(t.size), "data_offsets": [off, off + t.numel * nb]}
        off += t.numel * nb
    # Sorted keys, no spaces: the same header (and file SHA-256) every run.
    h = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    h += b" " * (-len(h) % 8)
    path = tmp / name
    hasher = hashlib.sha256()
    with open(path, "wb") as f:
        head = struct.pack("<Q", len(h)) + h
        f.write(head)
        hasher.update(head)
        for k in names:
            copy_tensor(z, prefix, state[k], f, hasher)
    out_sha = hasher.hexdigest()
    size = path.stat().st_size
    if size != 8 + len(h) + off:
        raise SystemExit(f"{path}: {size} bytes, expected {8 + len(h) + off}")
    print(f"wrote {name}: {len(names)} tensors, {off} tensor bytes, {size} bytes, "
          f"dtypes {sorted({DTYPES[t.storage[0]][0] for t in state.values()})}, skipped {skipped}", flush=True)

    # Read back: the header round-trips and every tensor equals its source.
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        back = json.loads(f.read(n))
        if {k: v for k, v in back.items() if k != "__metadata__"} != {k: v for k, v in header.items() if k != "__metadata__"}:
            raise SystemExit(f"{path}: the header differs after the round trip")
        base = 8 + n
        for k in names:
            a, b = back[k]["data_offsets"]
            if not same_bytes(z, prefix, state[k], f, base + a, base + b):
                raise SystemExit(f"{path}: {k} differs from the source")
    print(f"read back: {len(names)} tensors equal the source", flush=True)

    with open(tmp / "keys.txt", "w") as fk:
        for k in names:
            fk.write(f"{k}\t{list(state[k].size)}\t{header[k]['dtype']}\n")
    (tmp / "sha256.txt").write_text(f"{out_sha}  {name}\n{src_sha}  {HUB_FILE} (the source, {HUB_REPO})\n")
    (tmp / ".complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    tmp.rename(out)
    print(f"{out_sha}  {out / name}")
    print(f"done in {time.time() - t0:.0f} s: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
