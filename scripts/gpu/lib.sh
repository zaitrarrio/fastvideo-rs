#!/usr/bin/env bash
# Shared helpers for scripts/gpu/*.sh. Source, don't execute.
# shellcheck shell=bash disable=SC2034

FV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit "${2:-2}"; }

# Load KEY=VALUE pairs from the repo's .env (or $FV_ENV_FILE) without
# evaluating it. Variables already set in the environment win. Values are
# exported so `vastai` (which reads VAST_API_KEY) and the detached watchdog
# see them without the key ever appearing on a command line.
fv_load_env() {
  local file="${FV_ENV_FILE:-$FV_ROOT/.env}"
  [[ -f "$file" ]] || return 0
  local perms
  perms="$(stat -f '%Lp' "$file" 2>/dev/null || stat -c '%a' "$file" 2>/dev/null || echo 600)"
  if [[ "${perms: -2}" != "00" ]]; then
    log "warning: $file is readable by other users (mode $perms); run: chmod 600 $file"
  fi
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    line="${line#export }"
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]] || {
      log "warning: ignoring malformed line in $file"
      continue
    }
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then
      value="${BASH_REMATCH[1]}"
    fi
    [[ -n "${!key+x}" ]] || export "$key=$value"
  done <"$file"
}
fv_load_env

FV_LABEL_PREFIX="fvgpu"
FV_SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
FV_REMOTE_DIR="${VAST_REMOTE_DIR:-/workspace/fastvideo-rs}"

require_tools() {
  local missing=()
  for t in "$@"; do command -v "$t" >/dev/null 2>&1 || missing+=("$t"); done
  [[ ${#missing[@]} -eq 0 ]] || die "missing tools: ${missing[*]}"
}

# Portable `timeout` (GNU coreutils on macOS is `gtimeout`).
fv_timeout() {
  if command -v timeout >/dev/null 2>&1; then timeout "$@"
  elif command -v gtimeout >/dev/null 2>&1; then gtimeout "$@"
  else
    local secs="$1"; shift
    "$@" & local pid=$!
    ( sleep "$secs"; kill -TERM "$pid" 2>/dev/null; sleep 30; kill -KILL "$pid" 2>/dev/null ) & local w=$!
    wait "$pid"; local rc=$?
    kill "$w" 2>/dev/null; wait "$w" 2>/dev/null
    return "$rc"
  fi
}

# --- vast ---------------------------------------------------------------------

vast_check_auth() {
  [[ -n "${VAST_API_KEY:-}" ]] \
    || die "VAST_API_KEY is not set: cp .env.example .env, add your key, chmod 600 .env"
  local out
  if ! out="$(vastai show user --raw 2>&1)"; then
    die "vastai rejected VAST_API_KEY from ${FV_ENV_FILE:-.env}: $(head -c 300 <<<"$out")"
  fi
  jq -r '"vast user ok: balance=$\(.credit // .balance // "?")"' <<<"$out" 2>/dev/null >&2 || true
}

vast_balance() {
  vastai show user --raw 2>/dev/null | jq -r '.credit // .balance // empty' 2>/dev/null
}

# vast_ssh_target ID → "host port"
vast_ssh_target() {
  local id="$1" url hostport
  url="$(vastai ssh-url "$id" 2>/dev/null)" || return 1
  [[ "$url" == ssh://* ]] || return 1
  hostport="${url#ssh://root@}"
  printf '%s %s\n' "${hostport%:*}" "${hostport##*:}"
}

vast_destroy() {
  local id="$1"
  [[ -n "$id" ]] || return 0
  log "destroying instance $id"
  local i
  for i in 1 2 3; do
    if vastai destroy instance "$id" -y >/dev/null 2>&1; then
      log "instance $id destroyed"
      return 0
    fi
    sleep $((i * 5))
  done
  log "WARNING: could not confirm destroy of $id — check: vastai show instances --label ${FV_LABEL_PREFIX}*"
  return 1
}

# --- ssh / rsync ----------------------------------------------------------------

FV_SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=15 -o BatchMode=yes)

fv_ssh() {
  local host="$1" port="$2"; shift 2
  ssh -i "$FV_SSH_KEY" -p "$port" "${FV_SSH_OPTS[@]}" "root@$host" "$@"
}

fv_rsync_to() {
  local host="$1" port="$2" src="$3" dst="$4"; shift 4
  rsync -az "$@" -e "ssh -i $FV_SSH_KEY -p $port ${FV_SSH_OPTS[*]}" "$src" "root@$host:$dst"
}

fv_rsync_from() {
  local host="$1" port="$2" src="$3" dst="$4"; shift 4
  rsync -az "$@" -e "ssh -i $FV_SSH_KEY -p $port ${FV_SSH_OPTS[*]}" "root@$host:$src" "$dst"
}
