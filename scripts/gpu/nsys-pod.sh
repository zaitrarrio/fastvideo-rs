#!/bin/bash
# On-pod driver for docs/perf/datacenter-profile.md: fv-serve (or fv-gpucheck)
# under Nsight Systems, one trace per job. Runs on an E2E pod
# (scripts/serve/e2e/pod.sh up ...), uploaded to /e2e with this repo's
# scripts/gpu/nsys_profile.py and scripts/serve/e2e/bench.py.
#
#   nsys-pod.sh install              Nsight Systems CLI from NVIDIA's devtools repo
#   nsys-pod.sh key                  write /e2e/prof/k.json from $FV_KEY (mode 600)
#   nsys-pod.sh serve <config>       (re)start fv-serve under `nsys launch` (session fv)
#   nsys-pod.sh ready [timeout_s]    wait for /ping 200 on 127.0.0.1:8000
#   nsys-pod.sh warm <label> <body>  one untraced job
#   nsys-pod.sh job <label> <body>   one traced job: nsys start, job, nsys stop,
#                                    then export + analyze in the background
#   nsys-pod.sh gpucheck <label> <tail_s> <args...>
#                                    fv-gpucheck under `nsys profile`, analyze the
#                                    last <tail_s> s of device activity
#   nsys-pod.sh stop                 stop fv-serve (and the nsys session)
#
# Traces: CUDA runtime + driver API, kernels, memcpy/memset, cuBLAS and cuDNN
# ranges, CUDA graphs per node. No CPU sampling (no perf access in the pod).
set -u
P=/e2e/prof
mkdir -p "$P/out"
NSYS_DEB=NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb
# `nsys launch` takes the trace switches; --sample / --cpuctxsw belong to start / profile.
TRACE=(--trace=cuda,cublas,cudnn --cuda-graph-trace=node)
NOCPU=(--sample=none --cpuctxsw=none)
BIN=/opt/fastvideo-rs/bin

stop_serve() {
  nsys shutdown --session=fv --kill sigterm >/dev/null 2>&1
  if [ -s /e2e/serve.pid ]; then kill -TERM "$(cat /e2e/serve.pid)" 2>/dev/null; fi
  for _ in $(seq 60); do pidof fv-serve >/dev/null || break; sleep 1; done
  for p in $(pidof fv-serve); do kill -KILL "$p" 2>/dev/null; done
  : > /e2e/serve.pid
  sleep 3  # let the old nsys session go before a new one takes its name
}

analyze() {  # label [tail_s]
  local l=$1 tail=${2:-}
  nsys export --type sqlite --force-overwrite=true -o "$P/$l.sqlite" "$P/$l.nsys-rep" > "$P/$l.analyze.log" 2>&1
  local meta; meta=$(grep "\"label\": \"$l\"" "$P/runs.jsonl" 2>/dev/null | tail -1)
  python3 /e2e/scripts/gpu/nsys_profile.py "$P/$l.sqlite" --label "$l" --out-dir "$P/out/$l" \
    --meta "{\"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\", \"job\": ${meta:-null}}" \
    ${tail:+--tail-s "$tail"} >> "$P/$l.analyze.log" 2>&1
  echo "analyzed rc=$?" >> "$P/$l.analyze.log"
}

case "${1:-}" in
  install)
    command -v nsys >/dev/null && { nsys --version; exit 0; }
    wget -q -O /tmp/nsys.deb "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/$NSYS_DEB" \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends /tmp/nsys.deb > "$P/nsys-install.log" 2>&1
    rm -f /tmp/nsys.deb
    nsys --version ;;
  key)
    ( umask 077; printf '{"key": "%s"}' "${FV_KEY:?}" > "$P/k.json" ); echo "key file written" ;;
  serve)
    cfg="${2:-/e2e/fv.toml}"
    stop_serve
    echo "=== $(date -u +%FT%TZ) start $cfg (nsys launch)" >> /e2e/serve.log
    env -u RUNPOD_POD_ID -u RUNPOD_PUBLIC_IP \
      FV_PUBLIC_BASE_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net" FV_WORKER_ID="${RUNPOD_POD_ID}" \
      nsys launch --session-new=fv "${TRACE[@]}" "$BIN/fv-serve" --config "$cfg" >> /e2e/serve.log 2>&1 &
    sleep 5; pidof fv-serve > /e2e/serve.pid; echo "fv-serve pid $(cat /e2e/serve.pid)" ;;
  ready)
    t0=$(date +%s)
    for _ in $(seq "${2:-600}"); do
      pidof fv-serve >/dev/null || { echo "fv-serve not running"; tail -5 /e2e/serve.log; exit 1; }
      c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/ping)
      [ "$c" = 200 ] && { echo "ready after $(( $(date +%s) - t0 )) s"; exit 0; }
      sleep 1
    done
    echo "not ready ($c)"; tail -5 /e2e/serve.log; exit 1 ;;
  warm)
    python3 /e2e/scripts/serve/e2e/bench.py --base http://127.0.0.1:8000 --key-file "$P/k.json" \
      --out "$P/warm.jsonl" --label "$2" --body "$3" --timeout 900 ;;
  job)
    l=$2
    nsys start --session=fv "${NOCPU[@]}" --output="$P/$l.nsys-rep" --force-overwrite=true
    python3 /e2e/scripts/serve/e2e/bench.py --base http://127.0.0.1:8000 --key-file "$P/k.json" \
      --out "$P/runs.jsonl" --label "$l" --body "$3" --timeout 900
    rc=$?
    nsys stop --session=fv > "$P/$l.stop.log" 2>&1
    ( analyze "$l" ) > /dev/null 2>&1 &
    exit $rc ;;
  gpucheck)
    l=$2; tail=$3; shift 3
    gc=$(ls "$BIN/fv-gpucheck" /opt/fastvideo-rs/target/release/fv-gpucheck 2>/dev/null | head -1)
    nsys profile "${TRACE[@]}" "${NOCPU[@]}" --force-overwrite=true -o "$P/$l" "$gc" "$@" > "$P/$l.log" 2>&1
    echo "gpucheck rc=$?"; grep -a 'wan/run' "$P/$l.log" | head -c 400; echo
    ( analyze "$l" "$tail" ) > /dev/null 2>&1 & ;;
  stop) stop_serve; echo stopped ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
