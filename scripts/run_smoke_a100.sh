#!/usr/bin/env bash
# run_smoke_a100.sh — A100 smoke-test matrix for fastvideo-cudarc.
#
# Usage:
#   FASTVIDEO_WEIGHTS=/mnt/wan bash scripts/run_smoke_a100.sh
#
# Or:
#   bash scripts/run_smoke_a100.sh /mnt/wan
#
# Optional env knobs:
#   SMOKE_FILTER     — pass to 'cargo test --' to run only matching tests
#   SMOKE_FULL=1     — include the slow 81-frame test (takes ~5–10 min)
#   CARGO_FLAGS      — extra cargo flags, e.g. "--release"
#
# Each variant runs in a separate subprocess so that cached env flags
# (CachedBool / CachedString) are read fresh for every configuration.
# Results are collected into a timing summary printed at the end.

set -euo pipefail

WEIGHTS="${FASTVIDEO_WEIGHTS:-${1:-}}"
if [[ -z "$WEIGHTS" ]]; then
    echo "ERROR: set FASTVIDEO_WEIGHTS=/path/to/wan or pass it as first argument." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CARGO="${CARGO:-cargo}"
CARGO_FLAGS="${CARGO_FLAGS:-}"
SMOKE_FULL="${SMOKE_FULL:-0}"
SMOKE_FILTER="${SMOKE_FILTER:-}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${REPO_ROOT}/target/smoke-logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Columns: variant-name | extra env vars
VARIANTS=(
    # name                   extra env vars (KEY=VAL pairs, space-separated)
    "t2v_default             "
    "t2v_dmd_1step           "
    "t2v_dmd_4step           "
    "t2v_cfg_guidance_5      "
    "t2v_seed_determinism    "
    "t2v_9frames             "
    "t2v_hd_720p             "
    "t2v_dense_sdpa          FASTVIDEO_SDPA=dense"
    "t2v_f32_path            FASTVIDEO_BF16=0"
    "t2v_teacache_default    FASTVIDEO_TEACACHE=1 FASTVIDEO_TEACACHE_THRESH=0.08"
    "t2v_teacache_aggressive FASTVIDEO_TEACACHE=1 FASTVIDEO_TEACACHE_THRESH=0.15"
    "t2v_maxperf_combined    FASTVIDEO_TEACACHE=1 FASTVIDEO_TEACACHE_THRESH=0.08 FASTVIDEO_BF16=1 FASTVIDEO_SDPA=flash FASTVIDEO_TF32=1"
    "bench_step_latency_16step "
)

if [[ "$SMOKE_FULL" == "1" ]]; then
    VARIANTS+=("t2v_full_81frame FASTVIDEO_SMOKE_FULL=1")
fi

# ─── Build once ──────────────────────────────────────────────────────────────
echo "═══ Building fastvideo-cudarc (features=cuda) ═══"
${CARGO} build -p fastvideo-cudarc --features cuda --test smoke_a100 ${CARGO_FLAGS} 2>&1 \
    | tee "${LOG_DIR}/build.log"
echo ""

# ─── Run variants ─────────────────────────────────────────────────────────────
declare -a RESULTS   # "NAME STATUS ELAPSED"

run_variant() {
    local raw="$1"
    # Parse: first token = test name, rest = KEY=VAL env pairs
    local name
    name="$(echo "$raw" | awk '{print $1}')"
    local extra_env
    extra_env="$(echo "$raw" | cut -d' ' -f2-)"

    # Apply SMOKE_FILTER
    if [[ -n "$SMOKE_FILTER" && "$name" != *"$SMOKE_FILTER"* ]]; then
        RESULTS+=("${name} SKIPPED 0")
        return
    fi

    local log="${LOG_DIR}/${name}.log"
    local start end elapsed_s status

    echo "─── ${name} ───────────────────────────────────────"
    if [[ -n "$extra_env" && "$extra_env" != " " ]]; then
        echo "    env: ${extra_env}"
    fi

    # Build the env string for eval
    local env_prefix="FASTVIDEO_WEIGHTS=${WEIGHTS}"
    for kv in $extra_env; do
        env_prefix="${env_prefix} ${kv}"
    done

    start=$(date +%s%3N)
    if env ${env_prefix} \
        ${CARGO} test -p fastvideo-cudarc --features cuda --test smoke_a100 \
            ${CARGO_FLAGS} \
            -- --test-threads=1 --nocapture "${name}" 2>&1 \
        | tee "${log}"; then
        status="PASS"
    else
        status="FAIL"
    fi
    end=$(date +%s%3N)
    elapsed_s=$(( (end - start) ))

    RESULTS+=("${name} ${status} ${elapsed_s}")
    echo "    → ${status} in ${elapsed_s}ms"
    echo ""
}

for variant in "${VARIANTS[@]}"; do
    run_variant "$variant"
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo " SMOKE RESULTS  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo " Weights: ${WEIGHTS}"
echo "═══════════════════════════════════════════════════════"
printf "%-42s  %-8s  %s\n" "TEST" "STATUS" "ELAPSED(ms)"
printf "%-42s  %-8s  %s\n" "$(printf '%.0s-' {1..42})" "--------" "-----------"

PASS=0 FAIL=0 SKIP=0
for r in "${RESULTS[@]}"; do
    n="$(echo "$r" | awk '{print $1}')"
    s="$(echo "$r" | awk '{print $2}')"
    e="$(echo "$r" | awk '{print $3}')"
    printf "%-42s  %-8s  %s\n" "$n" "$s" "$e"
    case "$s" in
        PASS) (( PASS++ )) ;;
        FAIL) (( FAIL++ )) ;;
        SKIPPED) (( SKIP++ )) ;;
    esac
done

echo "═══════════════════════════════════════════════════════"
echo " PASS=${PASS}  FAIL=${FAIL}  SKIPPED=${SKIP}"
echo " Logs: ${LOG_DIR}"
echo "═══════════════════════════════════════════════════════"

if (( FAIL > 0 )); then
    echo "FAILED variants:"
    for r in "${RESULTS[@]}"; do
        s="$(echo "$r" | awk '{print $2}')"
        n="$(echo "$r" | awk '{print $1}')"
        if [[ "$s" == "FAIL" ]]; then
            echo "  ${n}  →  ${LOG_DIR}/${n}.log"
        fi
    done
    exit 1
fi
exit 0
