#!/usr/bin/env bash
# Sync this repo onto a running Vast instance over SSH.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_ID="${VAST_INSTANCE_ID:-50512723}"
SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
REMOTE_DIR="${VAST_REMOTE_DIR:-/workspace/fastvideo-rs}"

url="$(vastai ssh-url "$INSTANCE_ID")"
# ssh://root@HOST:PORT
hostport="${url#ssh://root@}"
host="${hostport%:*}"
port="${hostport##*:}"

echo "sync $ROOT -> root@$host:$port:$REMOTE_DIR"
ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "root@$host" "mkdir -p $REMOTE_DIR"
rsync -az --delete \
  --exclude target \
  --exclude target-linux \
  --exclude .git \
  --exclude 'outputs' \
  --exclude 'artifacts' \
  -e "ssh -i $SSH_KEY -p $port -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=3" \
  "$ROOT/" "root@$host:$REMOTE_DIR/"
echo "synced"
