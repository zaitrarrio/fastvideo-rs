#!/usr/bin/env bash
# scripts/serve/ci-images.sh `smoke` cannot hang: every `docker run` is bounded (FV_SMOKE_TIMEOUT_S), a container
# that outlives it is removed, and the refusal check says what went wrong. serve-image run 286 waited hours on an
# image whose fv-boot (a stale tools release) served fv-serve's default instead of refusing a boot without a config.
# A stand-in `docker` (and `curl`) on PATH plays the image; no Docker, no network.
#
#   bash scripts/serve/tests/ci-images.test.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
PASS=0 FAIL=0
ok() { PASS=$((PASS + 1)); printf 'ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$1"; [[ -z "${2:-}" ]] || printf '%s\n' "$2" | sed 's/^/     /' | tail -20; }

mkdir -p "$T/bin" "$T/run"
# FAKE_BOOT: what the image does when started without a config: refuse | hang | boot.
# A named container is "running" (docker container inspect finds it) while $T/run/<name> exists.
cat >"$T/bin/docker" <<'SH'
#!/usr/bin/env bash
echo "docker $*" >>"$FAKE_DIR/calls"
case "$1" in
  pull) exit 0 ;;
  image)
    [[ "$*" == *'json .Config.Entrypoint}}'* && "$*" != *size=* ]] && { echo '["/opt/fastvideo-rs/bin/fv-boot"]'; exit 0; }
    echo "size=1 entrypoint=[] env=[]"; exit 0 ;;
  container) [[ "$2" == inspect && -e "$FAKE_DIR/run/${*: -1}" ]] && exit 0; exit 1 ;;
  rm) name="${*: -1}"; [[ -e "$FAKE_DIR/run/$name" ]] && { rm -f "$FAKE_DIR/run/$name"; echo "removed $name" >>"$FAKE_DIR/calls"; }; exit 0 ;;
  logs) echo "fake logs"; exit 0 ;;
  run)
    shift; name=""; detach=0; args=" $* "
    while (($#)); do case "$1" in --name) name="$2"; shift 2 ;; -d) detach=1; shift ;; *) shift ;; esac; done
    ((detach)) && { echo cid123; exit 0; }
    [[ -n "$name" ]] && : >"$FAKE_DIR/run/$name"
    done_() { [[ -z "$name" ]] || rm -f "$FAKE_DIR/run/$name"; }
    if [[ "$args" == *" --version "* ]]; then echo "fv-serve 0.0.0"; done_; exit 0; fi
    if [[ "$args" == *" --entrypoint bash "* ]]; then echo "smoke ok"; done_; exit 0; fi
    if [[ "$args" == *" FV_FAIL_SLEEP_S=0 "* ]]; then
      case "$FAKE_BOOT" in
        refuse) echo "[fv-boot] 00:00:00 +0.0s FAILED: no worker config: the image serves no default; send ..." >&2; done_; exit 1 ;;
        boot) echo "[fv-boot] exec fv-serve"; done_; exit 0 ;;
        hang) echo "[fv-boot] 00:00:00 +0.0s config ok: default (fv-serve's default)"; exec sleep 300 ;;
      esac
    fi
    done_; exit 0 ;;
esac
exit 0
SH
printf '#!/bin/sh\necho ok\n' >"$T/bin/curl"
chmod +x "$T/bin/docker" "$T/bin/curl"

smoke() { # <FAKE_BOOT> -> sets RC, OUT, TOOK
  : >"$T/calls"
  local t0=$SECONDS
  OUT="$(PATH="$T/bin:$PATH" FAKE_DIR="$T" FAKE_BOOT="$1" FV_SMOKE_TIMEOUT_S=2 \
    timeout 60 bash "$ROOT/scripts/serve/ci-images.sh" smoke cpu ghcr.io/x/serve@sha256:abc 2>&1)"
  RC=$?
  TOOK=$((SECONDS - t0))
}

smoke refuse
[[ "$RC" == 0 && "$OUT" == *"refused without a config (exit 1)"* ]] && ok "an image that refuses passes the smoke" || bad "refuse: rc=$RC" "$OUT"

smoke hang
if [[ "$RC" != 0 && "$RC" != 124 && "$TOOK" -lt 30 && "$OUT" == *"did not refuse within 2s"* ]]; then
  ok "an image that serves instead of refusing fails the smoke within FV_SMOKE_TIMEOUT_S (rc=$RC, ${TOOK}s)"
else
  bad "hang: rc=$RC after ${TOOK}s (124: the smoke itself hung)" "$OUT"
fi
grep -q "^removed fv-smoke-" "$T/calls" && ok "the timed-out container is removed (docker rm -f by name)" || bad "hang: container not removed" "$(cat "$T/calls")"
[[ -z "$(ls "$T/run")" ]] && ok "no container left running" || bad "containers left: $(ls "$T/run")"

smoke boot
[[ "$RC" == 2 && "$OUT" == *"it booted (exit 0)"* ]] && ok "an image that boots without a config fails the smoke" || bad "boot: rc=$RC" "$OUT"

grep -E '^docker run ' "$T/calls" | grep -v -- ' -d ' | grep -v -- '--name fv-smoke-' >"$T/unbounded" || true
[[ ! -s "$T/unbounded" ]] && ok "every foreground docker run of the smoke is named (bounded)" || bad "unbounded docker run" "$(cat "$T/unbounded")"

printf '\nci-images.test: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" == 0 ]]
