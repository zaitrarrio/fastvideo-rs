#!/usr/bin/env python3
"""Independent oracle for the manifest, key and lineage fixtures.

Regenerates fixtures/manifests/, fixtures/keys/ and fixtures/lineage/ from a
second implementation of the formats (Python: `blake3` for digests,
`cryptography` for Ed25519, `json` for the RFC 8785 subset), so the Rust
crate's tests check agreement with code that shares nothing with it.

TEST ONLY: the keys are derived from fixed seeds and are public knowledge.

    python3 -I -c "import sys; sys.path.insert(0, '<dir with blake3>'); \
        exec(open('fixtures/gen_fixtures.py').read())"
or, with blake3 and cryptography importable:  python3 fixtures/gen_fixtures.py
Run from the crate root (crates/fastvideo-manifest).
"""
import base64
import json
import os

import blake3
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DOMAIN = b"gia-manifest-v1\0"
CREATED = 1761264000  # 2026-10-23T00:00:00Z
CHUNKING = {"algorithm": "fastcdc-2020", "min": 16, "avg": 64, "max": 256}


def b3(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def jcs(value) -> bytes:
    """RFC 8785 subset: sorted keys, no whitespace, integers only."""
    def check(v, path):
        if isinstance(v, float):
            raise ValueError(f"float at {path}")
        if isinstance(v, dict):
            for k, x in v.items():
                check(x, f"{path}.{k}")
        if isinstance(v, list):
            for i, x in enumerate(v):
                check(x, f"{path}[{i}]")
    check(value, "$")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "signatures"}
    return b3(jcs(body))


def key(seed_byte: int) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes([seed_byte]) * 32)


KEYS = {"test-key-1": key(0x11), "test-key-2": key(0x22)}


def sign(manifest: dict, kid: str) -> None:
    sig = KEYS[kid].sign(DOMAIN + bytes.fromhex(digest(manifest)))
    manifest.setdefault("signatures", [])
    manifest["signatures"] = [s for s in manifest["signatures"] if s["kid"] != kid]
    manifest["signatures"].append(
        {"kid": kid, "alg": "ed25519", "sig": base64.urlsafe_b64encode(sig).rstrip(b"=").decode()}
    )


def file_entry(path: str, content: bytes, cut: int) -> dict:
    chunks, off = [], 0
    while off < len(content):
        piece = content[off:off + cut]
        chunks.append({"id": b3(piece), "offset": off, "len": len(piece)})
        off += len(piece)
    return {"path": path, "size": len(content), "blake3": b3(content), "chunks": chunks}


def manifest(name: str, toml_text: str, files: list, parent=None) -> dict:
    return {
        "schema": 1,
        "name": name,
        "parent": parent,
        "model_toml_blake3": b3(toml_text.encode()),
        "files": sorted(files, key=lambda f: f["path"].encode()),
        "chunking": dict(CHUNKING),
        "created_at_unix": CREATED,
        "signatures": [],
    }


def write(rel: str, text: str) -> None:
    p = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def dump(m: dict) -> str:
    return json.dumps(m, indent=2, ensure_ascii=False) + "\n"


# ---- synthetic file contents (deterministic) --------------------------------
WAN_FILES = [
    file_entry("model.toml", b"# placeholder: the real bytes are hashed into model_toml_blake3\n", 64),
    file_entry("weights/text_encoder.safetensors", bytes(range(256)) * 3, 200),
    file_entry("weights/transformer.safetensors", b"transformer-" * 40, 100),
    file_entry("weights/vae.safetensors", b"vae!" * 17, 50),
]
DMD_FILES = [
    WAN_FILES[0],
    file_entry("weights/text_encoder.safetensors", b"fp8-encoder" * 30, 120),
    file_entry("weights/transformer.safetensors", b"dmd-distilled-" * 33, 150),
    WAN_FILES[3],
]
NF4_FILES = [
    WAN_FILES[0],
    file_entry("weights/text_encoder.safetensors", b"nf4" * 90, 90),
    file_entry("weights/transformer.safetensors", b"dmd-2step-" * 42, 140),
    WAN_FILES[3],
]

# ---- model.toml texts for the lineage chains --------------------------------
def vendor_toml(redistribution: bool) -> str:
    return f'''schema = 1
name = "wan-ai/wan2.2-t2v"
license = {{ spdx = "Apache-2.0", redistribution = {str(redistribution).lower()}, url = "https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt" }}

[runtime]
engine = "fastvideo-rs"
min_version = "0.1.0"
pipeline = "wan"

[defaults]
steps = 40
guidance = 5.0
width = 832
height = 480
frames = 81
fps = 16

[components]
transformer = {{ path = "weights/transformer.safetensors", dtype = "bf16" }}
text_encoder = {{ path = "weights/text_encoder.safetensors", dtype = "bf16" }}
vae = {{ path = "weights/vae.safetensors", dtype = "bf16" }}
'''


def fork1_toml(variant: str, parent_rev: str, redistribution: bool = True) -> str:
    return f'''schema = 1
name = "zay/wan2.2-t2v/{variant}"
parent = "wan-ai/wan2.2-t2v@{parent_rev}"
license = {{ spdx = "Apache-2.0", redistribution = {str(redistribution).lower()} }}

[runtime]
engine = "fastvideo-rs"
min_version = "0.1.0"
pipeline = "wan"
recipe = "dmd-4step"
profile = "wan/fastwan_dmd_4step"

[defaults]
steps = 4
guidance = 1.0
width = 832
height = 480
frames = 81
fps = 16

[components]
transformer = {{ path = "weights/transformer.safetensors", dtype = "bf16" }}
text_encoder = {{ path = "weights/text_encoder.safetensors", dtype = "fp8" }}
vae = {{ path = "weights/vae.safetensors", dtype = "bf16" }}

[change]
summary = "Distilled to 4 steps; fp8 text encoder"
reason = "Drafts in a fraction of the time"

[variant]
attention_route = "dense"
dit_quant = "bf16"
decoder = "full"
stages = 1
tiers = ["turbo"]
'''


def fork2_toml(parent_rev: str) -> str:
    return f'''schema = 1
name = "bo/wan2.2-t2v/dmd:2:nf4"
parent = "zay/wan2.2-t2v/dmd:4:fp8@{parent_rev}"
license = {{ spdx = "Apache-2.0", redistribution = true }}

[runtime]
engine = "fastvideo-rs"
min_version = "0.1.1"
pipeline = "wan"

[defaults]
steps = 2

[components]
transformer = {{ path = "weights/transformer.safetensors", dtype = "bf16" }}
text_encoder = {{ path = "weights/text_encoder.safetensors", dtype = "nf4" }}
vae = {{ path = "weights/vae.safetensors", dtype = "bf16" }}

[change]
summary = "Two-step distillation, nf4 text encoder"
reason = "Fits a 24 GB card"
'''


def chain(variant_in_toml="dmd:4:fp8", vendor_redistribution=True, fork1_files=DMD_FILES, break_link=False):
    v_toml = vendor_toml(vendor_redistribution)
    vendor = manifest("wan-ai/wan2.2-t2v", v_toml, WAN_FILES)
    sign(vendor, "test-key-1")
    v_digest = digest(vendor)

    f1_toml = fork1_toml(variant_in_toml, v_digest[:12])
    fork1 = manifest(
        "zay/wan2.2-t2v/dmd:4:fp8", f1_toml, fork1_files,
        parent={"name": "wan-ai/wan2.2-t2v", "digest": v_digest},
    )
    sign(fork1, "test-key-1")
    f1_digest = digest(fork1)

    link_digest = f1_digest
    if break_link:
        link_digest = "0" * 64
    f2_toml = fork2_toml(link_digest[:12])
    fork2 = manifest(
        "bo/wan2.2-t2v/dmd:2:nf4", f2_toml, NF4_FILES,
        parent={"name": "zay/wan2.2-t2v/dmd:4:fp8", "digest": link_digest},
    )
    sign(fork2, "test-key-2")
    return [("vendor", vendor, v_toml), ("fork1", fork1, f1_toml), ("fork2", fork2, f2_toml)]


def write_chain(dirname: str, entries) -> None:
    for stem, m, toml_text in entries:
        write(f"lineage/{dirname}/{stem}.json", dump(m))
        write(f"lineage/{dirname}/{stem}.model.toml", toml_text)
        write(f"lineage/{dirname}/{stem}.digest", digest(m) + "\n")


def main() -> None:
    # keys
    for kid, k in KEYS.items():
        priv = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                               serialization.NoEncryption()).decode()
        pub = k.public_key().public_bytes(serialization.Encoding.PEM,
                                          serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        write(f"keys/{kid}.pem", priv)
        write(f"keys/{kid}.pub.pem", pub)

    # manifests: a vendor ref (unsigned) and a fork (signed), plus the fork
    # re-serialised with reversed key order and other whitespace.
    ok = chain()
    vendor = dict(ok[0][1])
    vendor["signatures"] = []
    write("manifests/vendor.json", dump(vendor))
    write("manifests/vendor.digest", digest(vendor) + "\n")
    write("manifests/vendor.canonical.json", jcs({k: v for k, v in vendor.items() if k != "signatures"}).decode() + "\n")
    fork = ok[1][1]
    write("manifests/fork.json", dump(fork))
    write("manifests/fork.digest", digest(fork) + "\n")

    def reverse_keys(v):
        if isinstance(v, dict):
            return {k: reverse_keys(v[k]) for k in sorted(v.keys(), reverse=True)}
        if isinstance(v, list):
            return [reverse_keys(x) for x in v]
        return v
    write("manifests/fork-shuffled.json", json.dumps(reverse_keys(fork), indent=4) + "\n")
    write("manifests/fork-shuffled.digest", digest(fork) + "\n")

    # invalid manifests
    bad = json.loads(dump(vendor))
    bad["files"][1]["size"] = float(bad["files"][1]["size"])
    write("manifests/invalid/float-size.json", json.dumps(bad, indent=2) + "\n")
    bad = json.loads(dump(vendor))
    bad["licence"] = "Apache-2.0"
    write("manifests/invalid/unknown-key.json", dump(bad))
    bad = json.loads(dump(vendor))
    bad["files"].reverse()
    write("manifests/invalid/unsorted-files.json", dump(bad))
    bad = json.loads(dump(vendor))
    bad["files"][1]["chunks"][0]["offset"] = 1
    write("manifests/invalid/chunks-gap.json", dump(bad))
    bad = json.loads(dump(vendor))
    bad["files"][1]["path"] = "weights/../text_encoder.safetensors"
    write("manifests/invalid/dotdot-path.json", dump(bad))
    bad = json.loads(dump(vendor))
    bad["name"] = "wan-ai/wan2.2-t2v@" + digest(vendor)[:12]
    write("manifests/invalid/name-with-rev.json", dump(bad))

    # lineage chains
    write_chain("ok", ok)
    write_chain("broken-link", chain(break_link=True))
    write_chain("variant-mismatch", chain(variant_in_toml="dmd:8:fp8"))
    write_chain("licence-violation", chain(vendor_redistribution=False, fork1_files=WAN_FILES))
    print("fixtures written under", ROOT)


if __name__ == "__main__":
    main()
