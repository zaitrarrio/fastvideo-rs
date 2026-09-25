#!/usr/bin/env python3
"""Mirror a Hub repo revision as a local directory without re-downloading weights.

For every file of <repo>@<rev> (optionally filtered by path prefix):
  * a volume copy with the same relative path and byte size  -> symlink
  * otherwise, if it is small (<= --max-download-mb)         -> download it
  * otherwise                                                 -> reported missing

`--map SRC_PREFIX=LOCAL_DIR` lets a repo subtree resolve against another local
directory (e.g. FL2VA/text_encoder -> the root text_encoder already on disk:
the Hub stores the same LFS objects under both paths).

Prints a JSON report; exit 1 when anything is missing (unless --allow-missing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct import _curl, resolve_url  # noqa: E402


def tree(repo: str, rev: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    return [e for e in json.loads(_curl([url])) if e.get("type") == "file"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local", default=None, help="local dir holding the same relative paths")
    ap.add_argument("--map", action="append", default=[], help="REPO_PREFIX=LOCAL_DIR")
    ap.add_argument("--include", action="append", default=[], help="repo path prefixes to mirror")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--strip", default="", help="repo prefix removed in the output path")
    ap.add_argument("--max-download-mb", type=float, default=100)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    maps = [tuple(m.split("=", 1)) for m in args.map]
    report = {"linked": 0, "downloaded": [], "present": 0, "missing": []}
    for e in tree(args.repo, args.rev):
        p, size = e["path"], e.get("size", 0)
        if args.include and not any(p == i or p.startswith(i) for i in args.include):
            continue
        if any(p.startswith(x) for x in args.exclude):
            continue
        rel = p[len(args.strip):] if args.strip and p.startswith(args.strip) else p
        dest = out / rel
        if dest.exists() and dest.stat().st_size == size:
            report["present"] += 1
            continue
        cand = []
        for pre, loc in maps:
            if p.startswith(pre):
                cand.append(Path(loc) / p[len(pre):].lstrip("/"))
        if args.local:
            cand.append(Path(args.local) / p)
        src = next((c for c in cand if c.is_file() and c.stat().st_size == size), None)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src is not None:
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            os.symlink(os.path.realpath(src), dest)
            report["linked"] += 1
        elif size <= args.max_download_mb * 1e6:
            tmp = dest.with_name(dest.name + ".part")
            data = _curl([resolve_url(args.repo, args.rev, p)])
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            report["downloaded"].append([p, size])
        else:
            report["missing"].append([p, size])
    print(json.dumps(report, indent=1))
    return 1 if report["missing"] and not args.allow_missing else 0


if __name__ == "__main__":
    sys.exit(main())
