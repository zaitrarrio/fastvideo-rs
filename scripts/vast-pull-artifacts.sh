#!/usr/bin/env bash
# Copy GPU bench artifacts from a Vast instance onto this machine.
# Merges into ./artifacts/ (never --delete). Also imports leftover
# /workspace/fastvideo-* dirs from earlier runs that used fixed paths.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_ID="${VAST_INSTANCE_ID:-50416610}"
SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
LOCAL="${VAST_ARTIFACT_DIR:-$ROOT/artifacts}"

url="$(vastai ssh-url "$INSTANCE_ID")"
hostport="${url#ssh://root@}"
host="${hostport%:*}"
port="${hostport##*:}"

mkdir -p "$LOCAL"

echo "pull artifacts root@$host:$port:/workspace/artifacts -> $LOCAL"
ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=accept-new "root@$host" bash -s <<'EOF'
set -euo pipefail
GPU_SLUG="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//' || echo unknown)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p /workspace/artifacts
# Older benches wrote fixed paths. Snapshot them once if they still have frames.
legacy_root="/workspace/artifacts/${STAMP}-${GPU_SLUG}-imported"
copied=0
for src in \
  /workspace/fastvideo-tiny \
  /workspace/fastvideo-bench \
  /workspace/fastvideo-bench-480-canary \
  /workspace/fastvideo-bench-4s \
  /workspace/fastvideo-out \
  /workspace/fastvideo-gpu-smoke
do
  name="${src##*/}"
  if [ -d "$src" ] && ls "$src"/frame-*.png >/dev/null 2>&1; then
    dest="$legacy_root/$name"
    if [ ! -d "$dest" ]; then
      mkdir -p "$dest"
      cp -a "$src"/. "$dest"/
      copied=1
      echo "imported $src -> $dest"
    fi
  fi
done
if [ "$copied" -eq 1 ]; then
  echo "$STAMP imported leftover /workspace/fastvideo-* dirs" > "$legacy_root/README.txt"
fi
ls -la /workspace/artifacts 2>/dev/null || true
EOF

rsync -az \
  -e "ssh -i $SSH_KEY -p $port -o StrictHostKeyChecking=accept-new" \
  "root@$host:/workspace/artifacts/" "$LOCAL/"
echo "local artifacts:"
find "$LOCAL" -maxdepth 3 -type f \( -name 'bench.json' -o -name 'clip.json' -o -name 'run.json' -o -name 'frame-000.png' \) | sort
