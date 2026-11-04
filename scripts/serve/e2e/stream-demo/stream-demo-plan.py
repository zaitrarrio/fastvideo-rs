#!/usr/bin/env python3
"""The stream demo's launch plan: a preset's worker config and weight trees,
resolved against one weights source (stream-demo-pod.sh calls this; stdlib only).

    stream-demo-plan.py --root <repo> --preset <id|configs/serve file> --source volume|r2|hub

Prints one JSON object:

    preset, config (path), config_b64, variants, min_vram_gb (the preset's, 0 when
    unset), trees (every tree the config
    loads: the variants' trees from families/*.toml, ${FV_WEIGHTS}/<tree>
    references, the preset's pre-quantized trees), aux (the auxiliary/ files
    among them), bytes (the trees' size per scripts/gpu/weights-r2.tsv),
    disk_gb (the container disk), source, and per source:
      volume: {id, mount, dc}            every tree must be on the EU volume
                                         (scripts/gpu/volumes.sh FV_EU_VOLUME_TREES)
      r2:     {trees}                    every tree must be published (weights-r2.tsv);
                                         the aux trees ride along as fv-control does
      hub:    {trees, rows_b64, sha_b64, skipped}
                                         FV_WEIGHTS_TREES_B64 / FV_HUB_SHA256_B64 for
                                         fv-boot's hub source (pinned revisions from
                                         weights-revisions.tsv; aux rows from the
                                         manifest's url: rows); pre-quantized trees
                                         have no Hub source and are skipped

Exit 2 with the reason on stderr when the source cannot serve the preset.
"""

import argparse
import base64
import json
import os
import re
import sys
import tomllib

MAX_DISK_GB = 500
MIN_DISK_GB = 40
VOLUME_MOUNT = "/workspace"


def die(msg):
    print(f"stream-demo-plan: {msg}", file=sys.stderr)
    sys.exit(2)


def tsv_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            yield line.split("\t")


def tree_covers(tree, rel):
    return rel == tree or rel.startswith(tree + "/")


def eu_volume(root):
    text = open(os.path.join(root, "scripts/gpu/volumes.sh"), encoding="utf-8").read()
    def var(name):
        m = re.search(rf'^{name}="?([^"\n]*)"?$', text, re.M)
        return m.group(1) if m else ""
    m = re.search(r'FV_EU_VOLUME_TREES="((?:[^"\\]|\\\n)*)"', text)
    trees = m.group(1).replace("\\\n", " ").split() if m else []
    return {"id": var("FV_EU_VOLUME_ID"), "dc": var("FV_EU_DC"), "gpus": var("FV_EU_GPUS"), "trees": trees}


def families(root):
    out = {}
    d = os.path.join(root, "families")
    for n in sorted(os.listdir(d)):
        if not n.endswith(".toml"):
            continue
        with open(os.path.join(d, n), "rb") as f:
            fam = tomllib.load(f)
        for v in fam.get("variants", []):
            out[v["id"]] = v.get("trees", [])
    return out


def preset_of(root, name):
    with open(os.path.join(root, "deploy/deploy.toml"), "rb") as f:
        deploy = tomllib.load(f)
    # The model presets and the family presets (every variant of a family on one pod).
    presets = {p["id"]: p for p in deploy.get("presets", []) + deploy.get("family_presets", [])}
    if name in presets:
        p = presets[name]
        return name, os.path.join(root, "configs/serve", p["config"]), p.get("prequant", []), p.get("container_disk_gb"), p.get("min_vram_gb", 0)
    path = name if os.path.isfile(name) else os.path.join(root, "configs/serve", name)
    if not os.path.isfile(path):
        die(f"preset {name}: not a deploy/deploy.toml preset id ({', '.join(sorted(presets))}) nor a configs/serve file")
    base = os.path.basename(path)
    for p in presets.values():
        if p["config"] == base:
            return p["id"], path, p.get("prequant", []), p.get("container_disk_gb"), p.get("min_vram_gb", 0)
    return os.path.splitext(base)[0], path, [], None, 0


def config_trees(path, fams):
    text = open(path, encoding="utf-8").read()
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    variants = [m["variant"] for m in cfg.get("models", []) if "variant" in m]
    if not variants:
        die(f"{path}: no [[models]] variant")
    trees = []
    for v in variants:
        if v not in fams:
            die(f"{path}: variant {v} is not in families/*.toml")
        trees += fams[v]
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"\$\{FV_WEIGHTS\}/([A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?)", line):
            trees.append(m.group(1))
    return cfg, variants, sorted(set(trees))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--source", required=True, choices=["volume", "r2", "hub"])
    a = ap.parse_args()
    root = a.root
    gpu = os.path.join(root, "scripts/gpu")

    preset, cfg_path, prequant, disk_override, min_vram_gb = preset_of(root, a.preset)
    cfg, variants, trees = config_trees(cfg_path, families(root))
    manifest = {r[0]: r for r in tsv_rows(os.path.join(gpu, "weights-manifest.tsv"))}
    aux = {d: r for d, r in manifest.items() if len(r) > 1 and r[1].startswith("url:")}
    r2 = {r[0]: r for r in tsv_rows(os.path.join(gpu, "weights-r2.tsv"))}
    revs = {}
    for r in tsv_rows(os.path.join(gpu, "weights-revisions.tsv")):
        revs.setdefault(r[0], []).append(r)

    aux_files = [t for t in trees if t in aux]
    weight_trees = [t for t in trees if t not in aux]
    out = {
        "preset": preset,
        "config": os.path.relpath(cfg_path, root),
        "config_b64": base64.b64encode(open(cfg_path, "rb").read()).decode(),
        "variants": variants,
        "trees": weight_trees + prequant,
        "aux": aux_files,
        "prequant": prequant,
        "min_vram_gb": min_vram_gb,
        "source": a.source,
    }

    def r2_bytes(ts):
        return sum(int(r2[t][3]) for t in ts if t in r2)

    if a.source == "volume":
        vol = eu_volume(root)
        need = weight_trees + prequant + aux_files
        missing = [t for t in need if not any(tree_covers(e, t) for e in vol["trees"])]
        if missing:
            die(
                f"preset {preset}: not on the EU volume {vol['id']} ({', '.join(missing)}; the volume holds "
                f"H3 + LTX trees only, scripts/gpu/volumes.sh FV_EU_VOLUME_TREES). Set FV_DEMO_WEIGHTS=r2 "
                f"(the R2 weights store) or FV_DEMO_WEIGHTS=hub (the Hub at the pinned revisions)."
            )
        out["volume"] = {"id": vol["id"], "mount": VOLUME_MOUNT, "dc": vol["dc"], "gpus": vol["gpus"]}
        out["weights"] = f"{VOLUME_MOUNT}/weights"
        out["bytes"] = 0
        out["disk_gb"] = disk_override or MIN_DISK_GB
    elif a.source == "r2":
        fetch = list(weight_trees)
        for t in prequant:
            if t in r2 and t not in fetch:
                fetch.append(t)
        unpublished = [t for t in weight_trees if t not in r2]
        if unpublished:
            die(f"preset {preset}: not in the R2 weights store ({', '.join(unpublished)}; scripts/gpu/weights-r2.tsv)")
        # The aux trees (TAE decoders, LPIPS) come with the weights, as fv-control's r2 boots fetch them.
        for t in ("auxiliary/tae", "auxiliary/lpips"):
            if t in r2 and t not in fetch:
                fetch.append(t)
        out["r2"] = {"trees": fetch}
        out["weights"] = "/fvscratch/weights"
        out["bytes"] = r2_bytes(fetch)
        out["disk_gb"] = max(MIN_DISK_GB, min(MAX_DISK_GB, int(out["bytes"] * 1.15 / 1e9) + 30))
        if out["bytes"] * 1.15 / 1e9 + 30 > MAX_DISK_GB:
            die(f"preset {preset}: {out['bytes'] / 1e9:.0f} GB of trees need more than Runpod's {MAX_DISK_GB} GB container disk: use the volume")
    else:
        rows, sha_rows, skipped, hub_trees = [], [], [], []
        for t in weight_trees + prequant:
            rv = [r for r in revs.get(t, []) if len(r) >= 5 and r[4] == "hub" and re.fullmatch(r"[0-9a-f]{40}", r[2])]
            m = manifest.get(t)
            if not m or not rv or rv[0][1] != m[1]:
                if t in prequant:
                    skipped.append(t)  # no Hub source (an EU-volume / R2 tree): the engine quantizes at load
                    continue
                die(f"preset {preset}: {t} has no pinned Hub revision (scripts/gpu/weights-revisions.tsv, fetcher hub): use FV_DEMO_WEIGHTS=r2 or the volume")
            rows.append("\t".join(["tree", t, m[1], rv[0][2], m[2] if len(m) > 2 else ""]))
            hub_trees.append(t)
        # Every aux file (the TAE decoders, LPIPS): small, pinned by SHA-256, as fv-control's hub boots send them.
        for t in sorted(aux):
            r = aux[t]
            meta = " ".join(r[2:])
            sha = re.search(r"sha256:([0-9a-f]{64})", meta)
            size = re.search(r"size:(\d+)", meta)
            if not (sha and size):
                die(f"{t}: the manifest's url: row has no sha256/size")
            rows.append("\t".join(["aux", t, r[1][4:], sha.group(1), size.group(1)]))
        for r in tsv_rows(os.path.join(gpu, "weights-sha256.tsv")):
            if len(r) >= 4 and r[0] in hub_trees and r[3].startswith("sha256:"):
                sha_rows.append("\t".join([r[0], r[1], r[2] if r[2].strip() else "-", r[3][7:]]))
        out["hub"] = {
            "trees": hub_trees,
            "skipped": skipped,
            "rows_b64": base64.b64encode(("\n".join(rows) + "\n").encode()).decode(),
            "sha_b64": base64.b64encode(("\n".join(sha_rows) + "\n").encode()).decode(),
        }
        out["weights"] = "/fvscratch/weights"
        out["bytes"] = r2_bytes(hub_trees)
        out["disk_gb"] = max(MIN_DISK_GB, min(MAX_DISK_GB, int(out["bytes"] * 1.15 / 1e9) + 30))
        if out["bytes"] * 1.15 / 1e9 + 30 > MAX_DISK_GB:
            die(f"preset {preset}: {out['bytes'] / 1e9:.0f} GB of trees need more than Runpod's {MAX_DISK_GB} GB container disk: use the volume")
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
