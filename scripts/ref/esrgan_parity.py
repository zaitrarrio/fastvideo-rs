"""Independent numpy SRVGGNetCompact (Real-ESRGAN compact) for the parity
test of `fastvideo_cudarc::rt::esrgan`.

    python3 -I scripts/ref/esrgan_parity.py realesr-animevideov3.pth [W H SEED]

Reads the torch zip without torch (an unpickler that resolves only the
globals a plain state dict needs, as fastwan-rs's convert_esrgan.py), builds
the same deterministic LCG frame as the Rust test `lcg_frame(h, w, seed)`
(default 24x16, seed 2026), runs the network straight from the Real-ESRGAN
definition (conv3x3 + PReLU x17, conv -> pixel_shuffle(4) + nearest x4 of
the input, clamp, round) with the reference's numerics (f64 accumulation,
f32 stored per layer, f32 PReLU, the final stage exact in f64), and prints
the SHA-256 of the x4 RGB8 bytes, which the Rust test pins. With a fourth
argument OUT, also writes the raw bytes there.
"""
import collections, hashlib, io, pickle, struct, sys, zipfile

import numpy as np

DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}


class Storage:
    def __init__(self, kind, key):
        self.kind, self.key = kind, key


def rebuild(storage, offset, size, stride, *_):
    a = storage.data[offset:]
    n = int(np.prod(size)) if size else 1
    exp, acc = [], 1
    for d in reversed(size):
        exp.insert(0, acc)
        acc *= d
    assert list(stride) == exp or n <= 1, f"non-contiguous tensor {size} {stride}"
    return a[:n].reshape(size).astype(np.float32)


class Unpickler(pickle.Unpickler):
    def __init__(self, f, z, prefix):
        super().__init__(f)
        self.z, self.prefix = z, prefix

    def find_class(self, mod, name):
        if (mod, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild
        if mod == "torch" and name in DT:
            return name
        raise pickle.UnpicklingError(f"refusing global {mod}.{name}")

    def persistent_load(self, pid):
        kind, cls, key, _loc, _n = pid
        assert kind == "storage" and cls in DT, pid
        s = Storage(cls, key)
        s.data = np.frombuffer(self.z.read(f"{self.prefix}/data/{key}"), dtype=DT[cls])
        return s


def load(path):
    z = zipfile.ZipFile(path)
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    prefix = pkl[: -len("/data.pkl")]
    sd = Unpickler(io.BytesIO(z.read(pkl)), z, prefix).load()
    for k in ("params_ema", "params"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    return sd


def lcg_frame(h, w, seed):
    out = np.empty(h * w * 3, np.uint8)
    x = seed
    for i in range(h * w * 3):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = (x >> 16) & 0xFF
    return out.reshape(h, w, 3)


def conv3(x, w, b):  # x [C,H,W] f32, w [O,C,3,3], b [O]: f64 sums, f32 out
    c, hh, ww = x.shape
    p = np.pad(x.astype(np.float64), ((0, 0), (1, 1), (1, 1)))
    cols = np.stack([p[:, i : i + hh, j : j + ww] for i in range(3) for j in range(3)], 1)  # C,9,H,W
    y = np.einsum("ckhw,ock->ohw", cols, w.reshape(w.shape[0], c, 9).astype(np.float64)) + b.astype(np.float64)[:, None, None]
    return y.astype(np.float32)


def srvgg(sd, x8, scale=4):
    x = x8.transpose(2, 0, 1).astype(np.float32) / 255.0
    y, i = x, 0
    while f"body.{i}.weight" in sd and sd[f"body.{i}.weight"].ndim == 4:
        y = conv3(y, sd[f"body.{i}.weight"], sd[f"body.{i}.bias"])
        a = sd.get(f"body.{i + 1}.weight")
        if a is not None and a.ndim == 1:
            y = np.where(y >= 0, y, y * a[:, None, None]).astype(np.float32)
            i += 2
        else:
            i += 1
    c, h, w = 3, x.shape[1], x.shape[2]
    y = y.reshape(c, scale, scale, h, w).transpose(0, 3, 1, 4, 2).reshape(c, h * scale, w * scale)
    base = x8.transpose(2, 0, 1).repeat(scale, 1).repeat(scale, 2).astype(np.float64)
    v = y.astype(np.float64) * 255.0 + base + 0.5
    return np.clip(v, 0, 255).astype(np.uint8).transpose(1, 2, 0)


args = sys.argv[1:]
wts = args[0]
w, h, seed = (int(args[1]), int(args[2]), int(args[3])) if len(args) >= 4 else (24, 16, 2026)
out = srvgg(load(wts), lcg_frame(h, w, seed))
raw = np.ascontiguousarray(out).tobytes()
print(f"{w}x{h} seed {seed} -> {out.shape[1]}x{out.shape[0]}: sha256 {hashlib.sha256(raw).hexdigest()}")
if len(args) >= 5:
    open(args[4], "wb").write(raw)
