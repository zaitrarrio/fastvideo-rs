#!/usr/bin/env bash
# fv-boot GPU smoke (plan C2 gate; docs/adr/0014-fv-boot.md §8). Runs ON a GPU pod started from a serve image
# (ghcr.io/zaitrarrio/fastvideo-rs-serve:<variant>-…, e.g. ltx or h3-turbo) with the EU weights volume mounted at
# /workspace, with a release `fv-boot` built on the build pod (and optionally a `fv-serve`) downloaded onto it. The
# pod's own entrypoint must not be serving (start it with a command that fetches the binaries and runs this script).
#
# It runs fv-boot exactly as an image entrypoint would (env only, plus `--config`): driver check, volume, config
# materialization and `fv-serve --print-config`, the weights plan, then exec fv-serve; waits for /ping 200 (ready),
# reads /healthz, runs one tiny generation (unless FV_SMOKE_GENERATE=0), records fv-boot's phase reports (a local
# report sink stands in for fv-control's /ingest/v1/endpoint) and the timings in summary.json, stops fv-serve and
# exits 0 when the boot reached ready (and the job, when asked, succeeded).
#
# Env (inputs):
#   FV_BOOT_BIN          the fv-boot binary (required; chmod +x is done here)
#   FV_SMOKE_SERVE_BIN   a fv-serve to run instead of the image's (/opt/fastvideo-rs/bin/fv-serve)
#   FV_SMOKE_CONFIG      the config (default: the image's FV_CONFIG)
#   FV_SMOKE_SOURCE      FV_WEIGHTS_SOURCE (default volume; the EU volume holds H3 + LTX trees)
#   FV_SMOKE_WEIGHTS     FV_WEIGHTS (default /workspace/weights)
#   FV_SMOKE_GENERATE    1 (default): one tiny job through the native API; 0: boot only
#   FV_SMOKE_MODEL       the job's model (default: the first model of /fv/v1/capabilities)
#   FV_SMOKE_PARAMS      JSON merged into the job body (default: seed 1, short_edge 480, 16:9)
#   FV_SMOKE_WAIT_S      how long to wait for ready (default 900)
#   FV_SMOKE_JOB_WAIT_S  how long to wait for the job (default 600)
#   FV_SMOKE_OUT         output directory (default /tmp/fv-boot-smoke; never on the volume)
# Output: $FV_SMOKE_OUT/{summary.json,boot.log,reports.jsonl,healthz.json,job.json}.
set -uo pipefail

: "${FV_BOOT_BIN:?FV_BOOT_BIN: the fv-boot binary}"
OUT="${FV_SMOKE_OUT:-/tmp/fv-boot-smoke}"
case "$OUT" in /workspace/*) echo "FV_SMOKE_OUT must not be on the weights volume" >&2; exit 2 ;; esac
CFG="${FV_SMOKE_CONFIG:-${FV_CONFIG:-}}"
WAIT_S="${FV_SMOKE_WAIT_S:-900}"
JOB_WAIT_S="${FV_SMOKE_JOB_WAIT_S:-600}"
SINK_PORT=18765
BASE=http://127.0.0.1:8000
mkdir -p "$OUT/state"
chmod +x "$FV_BOOT_BIN"
[ -n "${FV_SMOKE_SERVE_BIN:-}" ] && chmod +x "$FV_SMOKE_SERVE_BIN"
command -v python3 >/dev/null || { echo "python3 is not in this image" >&2; exit 2; }
now() { date +%s.%N; }
log() { echo "[fv-boot-smoke] $(date -u +%H:%M:%S) $*" >&2; }

# The report sink: fv-control's /ingest/v1/endpoint stand-in, one JSON body per line.
python3 - "$SINK_PORT" "$OUT/reports.jsonl" >"$OUT/sink.log" 2>&1 <<'PY' &
import http.server, json, sys, time
port, path = int(sys.argv[1]), sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace")
        try:
            rec = {"at": time.time(), "auth": self.headers.get("authorization", ""), "body": json.loads(body)}
        except ValueError:
            rec = {"at": time.time(), "raw": body}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a):
        pass
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
PY
SINK=$!
: >"$OUT/reports.jsonl"

# fv-boot's environment: the image's own plus the pod-protocol report sink and a standalone fv-serve (no keys, jobs
# in memory, artifacts on local disk, state off the volume).
export FV_ENDPOINT_REPORT_URL="http://127.0.0.1:$SINK_PORT/ingest/v1/endpoint"
export FV_ENDPOINT_REPORT_TOKEN=smoke
export FV_POD_ID="smoke-${RUNPOD_POD_ID:-local}"
export FV_WEIGHTS_SOURCE="${FV_SMOKE_SOURCE:-volume}"
export FV_WEIGHTS="${FV_SMOKE_WEIGHTS:-/workspace/weights}"
export FV_STATE_DIR="$OUT/state"
export FV_AUTH_MODE="${FV_AUTH_MODE:-none}" FV_JOB_STORE="${FV_JOB_STORE:-memory}" FV_ARTIFACTS="${FV_ARTIFACTS:-local}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
[ -n "${FV_SMOKE_SERVE_BIN:-}" ] && export FV_SERVE_BIN="$FV_SMOKE_SERVE_BIN"
args=()
[ -n "$CFG" ] && args=(--config "$CFG")

log "fv-boot $("$FV_BOOT_BIN" version 2>&1) ${args[*]} (weights $FV_WEIGHTS_SOURCE at $FV_WEIGHTS)"
T0="$(now)"
"$FV_BOOT_BIN" "${args[@]}" >"$OUT/serve.out" 2>"$OUT/boot.log" &
PID=$!
cleanup() {
  kill -TERM "$PID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$PID" 2>/dev/null
  kill "$SINK" 2>/dev/null
}
trap cleanup EXIT

T_FIRST=""; T_READY=""; code=""; why=""
while :; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/ping" || true)"
  if [ -z "$T_FIRST" ] && { [ "$code" = 204 ] || [ "$code" = 200 ]; }; then T_FIRST="$(now)"; log "fv-serve answers /ping ($code)"; fi
  [ "$code" = 200 ] && { T_READY="$(now)"; break; }
  if ! kill -0 "$PID" 2>/dev/null; then why="the boot exited: $(tail -n 3 "$OUT/boot.log" | tr '\n"' '  ')"; break; fi
  if [ "$(awk -v a="$T0" -v b="$(now)" 'BEGIN{print (b-a > '"$WAIT_S"')}')" = 1 ]; then why="not ready after ${WAIT_S}s (last /ping $code)"; break; fi
  sleep 2
done
[ -n "$T_READY" ] && log "ready" || log "FAILED: $why"
curl -s --max-time 10 "$BASE/healthz" >"$OUT/healthz.json" || true

JOB_STATUS="skipped"; T_JOB0=""; T_JOB1=""
if [ -n "$T_READY" ] && [ "${FV_SMOKE_GENERATE:-1}" = 1 ]; then
  model="${FV_SMOKE_MODEL:-$(curl -s --max-time 10 "$BASE/fv/v1/capabilities" | python3 -c 'import json,sys
m = (json.load(sys.stdin).get("models") or [{}])[0]
print(m.get("id") or m.get("name") or "")' 2>/dev/null)}"
  body="$(python3 -c 'import json,sys
b = {"model": sys.argv[1], "prompt": "a red fox trotting through fresh snow", "seed": 1, "short_edge": 480, "aspect_ratio": "16:9"}
b.update(json.loads(sys.argv[2] or "{}"))
print(json.dumps(b))' "$model" "${FV_SMOKE_PARAMS:-}")"
  log "job: $body"
  T_JOB0="$(now)"
  id="$(curl -s --max-time 30 -H 'content-type: application/json' -d "$body" "$BASE/fv/v1/jobs" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id") or "")' 2>/dev/null)"
  JOB_STATUS="submit_failed"
  if [ -n "$id" ]; then
    for _ in $(seq 1 "$JOB_WAIT_S"); do
      curl -s --max-time 10 "$BASE/fv/v1/jobs/$id" >"$OUT/job.json" || true
      JOB_STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$OUT/job.json" 2>/dev/null)"
      case "$JOB_STATUS" in succeeded | failed | cancelled) break ;; esac
      sleep 1
    done
  fi
  T_JOB1="$(now)"
  log "job $id: $JOB_STATUS"
fi

python3 - "$OUT" "$T0" "${T_FIRST:-}" "${T_READY:-}" "${T_JOB0:-}" "${T_JOB1:-}" "$JOB_STATUS" "$why" "$CFG" "$FV_WEIGHTS_SOURCE" <<'PY' | tee "$OUT/summary.json"
import json, os, re, sys
out, t0, tf, tr, j0, j1, job, why, cfg, src = sys.argv[1:11]
f = lambda x: float(x) if x else None
t0 = float(t0)
rel = lambda x: round(f(x) - t0, 1) if x else None
lines = open(os.path.join(out, "boot.log"), errors="replace").read().splitlines()
steps, markers, failed = {}, [], None
for l in lines:
    m = re.match(r"^\[fv-boot\] \d\d:\d\d:\d\d \+([\d.]+)s (.*)$", l)
    if m:
        el, msg = float(m.group(1)), m.group(2)
        s = re.match(r"^step (\S+(?: \S+)?): ([\d.]+)s", msg)
        if s:
            steps[s.group(1)] = float(s.group(2))
        if msg.startswith("FAILED:"):
            failed = msg
        if msg.startswith(("config ok", "exec ", "driver ", "weights")):
            markers.append({"at_s": el, "line": msg[:300]})
    elif l.startswith("[fv-boot] "):
        markers.append({"at_s": None, "line": l[10:300]})
reports = []
for l in open(os.path.join(out, "reports.jsonl")):
    try:
        r = json.loads(l)
        reports.append({"at_s": round(r["at"] - t0, 1), **r.get("body", {})})
    except (ValueError, KeyError):
        pass
def load(p):
    try:
        return json.load(open(os.path.join(out, p)))
    except (OSError, ValueError):
        return None
hz = load("healthz.json")
j = load("job.json") or {}
print(json.dumps({
    "target": "fv-boot-smoke", "pod": os.environ.get("RUNPOD_POD_ID"), "config": cfg, "weights_source": src,
    "image_variant": os.environ.get("FV_VARIANT"), "fv_boot_failed": failed, "why": why or None,
    "timings_s": {"first_ping": rel(tf), "ready": rel(tr), "job_wall": round(f(j1) - f(j0), 1) if j0 and j1 else None},
    "boot_steps_s": steps, "boot_markers": markers, "reports": reports,
    "healthz": {k: hz.get(k) for k in ("state", "loaded", "loading")} if isinstance(hz, dict) else None,
    "job": {"status": job, "id": j.get("id"), "model": j.get("model"), "error": j.get("error")},
}, indent=2))
PY
[ -n "$T_READY" ] || exit 1
case "${FV_SMOKE_GENERATE:-1}:$JOB_STATUS" in 0:* | 1:succeeded) exit 0 ;; *) exit 1 ;; esac
