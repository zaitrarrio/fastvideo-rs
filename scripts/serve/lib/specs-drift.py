#!/usr/bin/env python3
"""Drift check for the upstream API specifications vendored under
docs/serve/specs/ (plan E1; docs/gaps/2026-10-22-architecture-review.md §4.4).

Every `docs/serve/specs/<upstream>/SOURCES.toml` lists the files of that
directory as `[[source]]` tables:

    file       path relative to the directory
    status     "ok" (vendored), "unavailable" (the URL gave no usable
               document; re-probed every run) or "derived" (hand-written
               from another vendored file; integrity-checked only)
    url        where the stored bytes came from (a pinned, immutable URL
               where the upstream offers one)
    drift_url  optional: the moving URL (branch head) to re-fetch for the
               drift check; defaults to `url`
    normalize  "json" (pretty-printed, sorted keys), "verbatim" (bytes as
               served) or "openai-videos" (the `/videos*` trim of the OpenAI
               `openapi.yaml`, re-applied on every fetch)
    fetched_at UTC time the stored bytes were fetched
    sha256     of the file as stored
    note       free text (what was trimmed, why unavailable, ...)

Without `--update` the script re-fetches every entry that has a URL,
normalises the download the same way, compares it with the stored file,
verifies every stored file's sha256 and exits 1 when anything drifted,
is unreachable, has become available or no longer matches its digest.
`--update` rewrites the drifted files and their SOURCES.toml entries.

Standard library only. Downloads are untrusted data: they are parsed, never
executed, and this file is meant to run under `python3 -I`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib

KEY_ORDER = ("file", "status", "url", "drift_url", "normalize", "fetched_at", "sha256", "note")
DIFF_LINES = 60
CURL_TIMEOUT_S = "120"


# --- SOURCES.toml -----------------------------------------------------------


def load_sources(path: str) -> tuple[str, list[dict]]:
    """Return (header comment, entries). The header is every line before the
    first `[[source]]`, kept verbatim when the file is rewritten."""
    with open(path, "rb") as fh:
        raw = fh.read()
    data = tomllib.loads(raw.decode("utf-8"))
    entries = data.get("source", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: `source` must be an array of tables")
    header_lines = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip() == "[[source]]":
            break
        header_lines.append(line)
    header = "\n".join(header_lines).rstrip("\n")
    for e in entries:
        for key in ("file", "status"):
            if key not in e:
                raise ValueError(f"{path}: every [[source]] needs `{key}`")
        if e["status"] not in ("ok", "unavailable", "derived"):
            raise ValueError(f"{path}: {e['file']}: unknown status {e['status']!r}")
        if e.get("normalize", "json") not in ("json", "verbatim", "openai-videos"):
            raise ValueError(f"{path}: {e['file']}: unknown normalize {e['normalize']!r}")
        for k in e:
            if k not in KEY_ORDER:
                raise ValueError(f"{path}: {e['file']}: unknown key {k!r}")
    return header, entries


def toml_string(value: str) -> str:
    # TOML basic strings accept the escapes json.dumps emits.
    return json.dumps(value, ensure_ascii=False)


def dump_sources(path: str, header: str, entries: list[dict]) -> None:
    out = [header, ""] if header else []
    for e in entries:
        out.append("[[source]]")
        for key in KEY_ORDER:
            if key in e and e[key] is not None:
                out.append(f"{key} = {toml_string(str(e[key]))}")
        out.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out).rstrip("\n") + "\n")


# --- fetching and normalising ------------------------------------------------


def fetch(url: str, dest: str) -> tuple[bool, str]:
    """curl the URL to `dest`. Returns (ok, detail). TLS verification stays on
    and the environment's proxy settings apply."""
    cmd = [
        "curl", "-sS", "-L", "--fail", "--max-time", CURL_TIMEOUT_S, "--retry", "2",
        "--retry-delay", "3", "-o", dest, "-w", "%{http_code}", url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False, "curl not found"
    code = proc.stdout.strip()
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        return False, f"HTTP {code or '?'}: {err[-1] if err else f'curl exit {proc.returncode}'}"
    return True, f"HTTP {code}"


def normalize_json(raw: bytes) -> bytes:
    obj = json.loads(raw.decode("utf-8"))
    if obj is None:
        raise ValueError("document is JSON null")
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


_REF = re.compile(r"#/components/schemas/([A-Za-z0-9_.\-]+)")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_end(lines: list[str], start: int) -> int:
    """End (exclusive) of the YAML block whose first line is `start`: the
    following lines that are blank or indented deeper than the first."""
    base = _indent(lines[start])
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line.strip() == "" or _indent(line) > base:
            i += 1
            continue
        break
    while i > start + 1 and lines[i - 1].strip() == "":
        i -= 1
    return i


def _top_block(lines: list[str], key: str) -> tuple[int, int]:
    for i, line in enumerate(lines):
        if line == f"{key}:" or line.startswith(f"{key}: "):
            return i, _block_end(lines, i)
    raise ValueError(f"top-level `{key}` not found")


def _children(lines: list[str], start: int, end: int, indent: int) -> list[tuple[str, int, int]]:
    """(key, start, end) of every mapping entry at exactly `indent` spaces
    inside lines[start:end]."""
    out = []
    i = start
    while i < end:
        line = lines[i]
        if line.strip() and _indent(line) == indent and not line.lstrip().startswith("- "):
            m = re.match(r"^ *(['\"]?)([^'\":]+)\1:(?: .*)?$", line)
            if m:
                e = min(_block_end(lines, i), end)
                out.append((m.group(2), i, e))
                i = e
                continue
        i += 1
    return out


def trim_openai_videos(text: str) -> str:
    """Keep the `openapi`, `info`, `servers` and `security` preamble, every
    `paths` entry under `/videos`, the `components.schemas` closure those
    paths reference (transitively) and `components.securitySchemes`. Blocks
    are copied as text, so the output diffs cleanly against upstream."""
    lines = text.split("\n")
    out: list[str] = [
        "# TRIMMED: the `/videos*` subset of the OpenAI `openapi.yaml`, extracted by",
        "# scripts/serve/lib/specs-drift.py (trim_openai_videos). Source and date are in",
        "# SOURCES.toml. Do not edit; `bash scripts/serve/specs-drift.sh --update` rewrites it.",
    ]
    for key in ("openapi", "info", "servers", "security"):
        try:
            s, e = _top_block(lines, key)
        except ValueError:
            continue
        out.extend(lines[s:e])

    ps, pe = _top_block(lines, "paths")
    path_blocks = [(k, s, e) for (k, s, e) in _children(lines, ps + 1, pe, 2) if k.startswith("/videos")]
    if not path_blocks:
        raise ValueError("no `/videos*` paths found")
    out.append("paths:")
    for _, s, e in path_blocks:
        out.extend(lines[s:e])

    cs, ce = _top_block(lines, "components")
    comps = {k: (s, e) for (k, s, e) in _children(lines, cs + 1, ce, 2)}
    if "schemas" not in comps:
        raise ValueError("`components.schemas` not found")
    ss, se = comps["schemas"]
    schemas = {k: (s, e) for (k, s, e) in _children(lines, ss + 1, se, 4)}

    wanted: set[str] = set()
    pending = set()
    for _, s, e in path_blocks:
        pending.update(_REF.findall("\n".join(lines[s:e])))
    while pending:
        name = pending.pop()
        if name in wanted:
            continue
        if name not in schemas:
            raise ValueError(f"schema `{name}` referenced but not defined")
        wanted.add(name)
        s, e = schemas[name]
        pending.update(_REF.findall("\n".join(lines[s:e])))

    out.append("components:")
    out.append("  schemas:")
    for name, (s, e) in schemas.items():  # upstream order
        if name in wanted:
            out.extend(lines[s:e])
    if "securitySchemes" in comps:
        s, e = comps["securitySchemes"]
        out.extend(lines[s:e])
    return "\n".join(out).rstrip("\n") + "\n"


def normalize(kind: str, raw: bytes) -> bytes:
    if kind == "json":
        return normalize_json(raw)
    if kind == "verbatim":
        return raw
    if kind == "openai-videos":
        return trim_openai_videos(raw.decode("utf-8")).encode("utf-8")
    raise ValueError(f"unknown normalize {kind!r}")


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the check ---------------------------------------------------------------


def diff_head(old: bytes, new: bytes, name: str) -> list[str]:
    try:
        a = old.decode("utf-8").splitlines()
        b = new.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return ["  (binary difference)"]
    d = list(difflib.unified_diff(a, b, f"stored/{name}", f"fetched/{name}", lineterm="", n=1))
    plus = sum(1 for l in d[2:] if l.startswith("+"))
    minus = sum(1 for l in d[2:] if l.startswith("-"))
    head = [f"  {plus} line(s) added, {minus} removed"]
    head += ["  " + l for l in d[:DIFF_LINES]]
    if len(d) > DIFF_LINES:
        head.append(f"  ... ({len(d) - DIFF_LINES} more diff lines)")
    return head


def run(specs_dir: str, update: bool, only: list[str], tmp: str) -> int:
    problems: list[str] = []
    checked = 0
    dirs = sorted(d for d in os.listdir(specs_dir) if os.path.isfile(os.path.join(specs_dir, d, "SOURCES.toml")))
    if only:
        dirs = [d for d in dirs if d in only]
        missing = sorted(set(only) - set(dirs))
        if missing:
            print(f"no SOURCES.toml for: {', '.join(missing)}", file=sys.stderr)
            return 2
    n = 0
    for d in dirs:
        toml_path = os.path.join(specs_dir, d, "SOURCES.toml")
        header, entries = load_sources(toml_path)
        changed = False
        for e in entries:
            n += 1
            name = f"{d}/{e['file']}"
            path = os.path.join(specs_dir, d, e["file"])
            kind = e.get("normalize", "json")
            url = e.get("drift_url") or e.get("url")
            stored = None
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    stored = fh.read()

            # 1. Integrity of what is stored.
            if e["status"] in ("ok", "derived"):
                if stored is None and not (update and url and e["status"] == "ok"):
                    problems.append(f"MISSING     {name}: listed in SOURCES.toml but not on disk")
                    print(f"MISSING     {name}")
                    continue
                digest = sha256_of(stored) if stored is not None else None
                if stored is not None and e.get("sha256") and digest != e["sha256"] and not update:
                    problems.append(f"SHA256      {name}: stored file does not match SOURCES.toml")
                    print(f"SHA256      {name}: stored {digest[:12]} != recorded {str(e.get('sha256'))[:12]}")
                    continue
            elif stored is not None:
                problems.append(f"UNEXPECTED  {name}: status unavailable but a file exists")
                print(f"UNEXPECTED  {name}: status unavailable but a file exists")
                continue

            if e["status"] == "derived" or not url:
                print(f"derived     {name}")
                checked += 1
                continue

            # 2. Re-fetch and normalise.
            dest = os.path.join(tmp, f"{n}.bin")
            ok, detail = fetch(url, dest)
            if not ok:
                if e["status"] == "unavailable":
                    print(f"unavailable {name} ({detail}; still unavailable)")
                    checked += 1
                else:
                    problems.append(f"UNREACHABLE {name}: {detail} ({url})")
                    print(f"UNREACHABLE {name}: {detail}")
                continue
            with open(dest, "rb") as fh:
                raw = fh.read()
            try:
                fresh = normalize(kind, raw)
            except (ValueError, UnicodeDecodeError) as exc:
                if e["status"] == "unavailable":
                    print(f"unavailable {name} ({detail}, {exc}; still unavailable)")
                    checked += 1
                else:
                    problems.append(f"UNPARSABLE  {name}: {exc} ({url})")
                    print(f"UNPARSABLE  {name}: {exc}")
                continue

            if e["status"] == "unavailable":
                problems.append(f"AVAILABLE   {name}: the URL now serves a parsable document ({len(fresh)} bytes); vendor it by hand")
                print(f"AVAILABLE   {name}: now serves {len(fresh)} bytes; vendor it by hand")
                continue

            # 3. Compare.
            if fresh == stored:
                print(f"ok          {name}")
                checked += 1
                continue
            lines = diff_head(stored, fresh, e["file"]) if stored is not None else [f"  new file, {len(fresh)} bytes"]
            if update:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as fh:
                    fh.write(fresh)
                e["sha256"] = sha256_of(fresh)
                e["fetched_at"] = utc_now()
                changed = True
                print(f"UPDATED     {name}")
                for l in lines:
                    print(l)
                checked += 1
            else:
                problems.append(f"DRIFT       {name}: {lines[0].strip()} ({url})")
                print(f"DRIFT       {name}")
                for l in lines:
                    print(l)
        if update:
            # Record digests of files that had none yet (first vendoring).
            for e in entries:
                path = os.path.join(specs_dir, d, e["file"])
                if e["status"] in ("ok", "derived") and os.path.isfile(path):
                    with open(path, "rb") as fh:
                        digest = sha256_of(fh.read())
                    if e.get("sha256") != digest:
                        e["sha256"] = digest
                        e.setdefault("fetched_at", utc_now())
                        changed = True
            if changed:
                dump_sources(toml_path, header, entries)
                print(f"wrote       {d}/SOURCES.toml")

    print()
    if problems:
        print(f"{len(problems)} problem(s), {checked} spec(s) unchanged:")
        for p in problems:
            print(f"- {p}")
        if not update:
            print("\nRun `bash scripts/serve/specs-drift.sh --update` to accept the upstream changes,")
            print("then review the diff and update the prose research docs that cite these files.")
        return 1
    print(f"no drift: {checked} spec(s) checked")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs-dir", required=True, help="docs/serve/specs")
    ap.add_argument("--update", action="store_true", help="rewrite drifted files and SOURCES.toml")
    ap.add_argument("--only", action="append", default=[], metavar="DIR", help="check only this upstream directory (repeatable)")
    ap.add_argument("--trim-openai", metavar="OPENAPI_YAML", help="print the `/videos*` trim of this local file and exit")
    args = ap.parse_args(argv)
    if args.trim_openai:
        with open(args.trim_openai, "r", encoding="utf-8") as fh:
            sys.stdout.write(trim_openai_videos(fh.read()))
        return 0
    with tempfile.TemporaryDirectory(prefix="specs-drift-") as tmp:
        return run(args.specs_dir, args.update, args.only, tmp)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
