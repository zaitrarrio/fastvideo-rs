#!/usr/bin/env bash
# Phase-1 batch on a rented GPU (runs inside the container; stdout is the
# container log that vast-run.sh reads back). Every result line starts "R ".
# Env: SAGE_SHAPES, SAGE_MODES, SAGE_KERNELS, SAGE_REAL (1 = capture FastWan
# Q/K/V and bench them), SAGE_STEPS (setup,capture,bench,real).
set -uo pipefail
W=/root/w
cd "$W"
T0=$(date +%s)
say() { echo "S $(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s $*"; }
STEPS="${SAGE_STEPS:-setup,capture,bench,real}"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv,noheader | sed 's/^/S gpu /'
if [[ ",$STEPS," == *,setup,* ]]; then
  say setup start
  bash setup.sh 2>&1 | sed 's/^/S setup /'
  for f in logs/sage2.log logs/sage3.log logs/ours.log logs/fetch.log; do
    [[ -f $f ]] || continue
    if ! tail -n1 "$f" | grep -q -- '-ok$'; then
      say "FAILED $f (tail):"; grep -v '^\s*$' "$f" | grep -iE 'error|fatal|fail' | tail -n 25 | sed 's/^/S   /'
      tail -n 15 "$f" | sed 's/^/S   /'
    fi
  done
  grep -A4 "Compiling entry.*\(attn_sage_fwd\|flash_mma_fwd2_d128\)" logs/ours.log | grep -i "Compiling entry\|registers\|spill" | sed 's/^/S ptxas /'
  say setup done
fi
python - <<'EOF' 2>&1 | sed 's/^/S env /'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
for m in ("sageattention", "sageattn3"):
    try:
        __import__(m); print(m, "import ok")
    except Exception as e:
        print(m, "import FAILED:", repr(e)[:300])
EOF
if [[ ",$STEPS," == *,capture,* && -d fastwan/transformer ]]; then
  say capture start
  python capture.py "$W/fastwan" "$W/cap" 2>&1 | grep -v Warning | tail -n 12 | sed 's/^/S cap /'
  say capture done
fi
if [[ ",$STEPS," == *,bench,* ]]; then
  say bench start
  python bench.py --cubins "$W" --out "$W/results.jsonl" \
    ${SAGE_SHAPES:+--shapes "$SAGE_SHAPES"} --modes "${SAGE_MODES:-normal,peaked,outlier}" \
    ${SAGE_KERNELS:+--kernels "$SAGE_KERNELS"} 2> >(sed 's/^/S bench-err /' | tail -n 30) | sed 's/^/R /'
  say bench done
fi
if [[ ",$STEPS," == *,real,* ]] && ls cap/*.pt >/dev/null 2>&1; then
  say real start
  python bench.py --cubins "$W" --out "$W/results.jsonl" --real "$(ls cap/*.pt | paste -sd,)" \
    ${SAGE_KERNELS:+--kernels "$SAGE_KERNELS"} --no-time 2> >(sed 's/^/S real-err /' | tail -n 30) | sed 's/^/R /'
  say real done
fi
say FV-SAGE-DONE
