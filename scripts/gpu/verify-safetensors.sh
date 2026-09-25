#!/usr/bin/env bash
# Verify one or more .safetensors files are complete (header + data length).
# Usage: verify-safetensors.sh <file> [<file>...]
#        verify-safetensors.sh --dir <dir>   # every *.safetensors under dir
set -euo pipefail

verify_one() {
  local p="$1"
  [[ -f "$p" ]] || { echo "missing $p" >&2; return 1; }
  local size hlen end
  size=$(wc -c <"$p" | tr -d ' ')
  # header length: first 8 bytes LE
  hlen=$(od -An -t u8 -N 8 -j 0 "$p" | tr -d ' ')
  if [[ -z "$hlen" ]] || (( size < 8 || hlen > size - 8 )); then
    echo "truncated header $p: size=$size header_len=${hlen:-none}" >&2
    return 1
  fi
  # Parse max data_offsets[1] from the JSON header with a tiny awk/python-free path:
  # read header bytes and find the largest "data_offsets":[a,b] second value.
  local header
  header=$(dd if="$p" bs=1 skip=8 count="$hlen" 2>/dev/null)
  end=$(printf '%s' "$header" | grep -oE '"data_offsets"[[:space:]]*:[[:space:]]*\[[[:space:]]*[0-9]+[[:space:]]*,[[:space:]]*[0-9]+' \
    | grep -oE '[0-9]+$' | sort -n | tail -1 || true)
  [[ -n "$end" ]] || { echo "no data_offsets in $p" >&2; return 1; }
  local want=$((8 + hlen + end))
  if [[ "$size" -ne "$want" ]]; then
    echo "truncated $p: size=$size want=$want" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--dir" ]]; then
  dir="${2:?}"
  n=0
  # HF cache snapshots point at blobs via symlinks; follow them (-L).
  while IFS= read -r -d '' f; do
    verify_one "$f"
    n=$((n + 1))
  done < <(find -L "$dir" -type f -name '*.safetensors' -print0)
  (( n > 0 )) || { echo "no safetensors under $dir" >&2; exit 1; }
  echo "weights ok: $n shards verified under $dir"
  exit 0
fi

(( $# >= 1 )) || { echo "usage: $0 <file>... | --dir <dir>" >&2; exit 1; }
for f in "$@"; do verify_one "$f"; done
echo "weights ok: $# file(s) verified"
