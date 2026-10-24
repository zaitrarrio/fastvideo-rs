#!/usr/bin/env bash
# The mutation checks of plan B4b (docs/adr/0022-block-fusion.md, "Evidence"):
# each mutation alone is applied to the worktree, the fusion tests run, and
# the mutation must make at least one of them fail; the file is restored
# whatever happens. Prints one line per mutation and exits 1 when a
# mutation survives (or no longer applies: the code moved, update the
# table).
#
#   bash scripts/dev/b4b-fuse-mutations.sh            # tests on the build pod
#   FV_MUT_RUNNER="cargo" bash scripts/dev/b4b-fuse-mutations.sh   # locally
#
# FV_MUT_RUNNER: the command that runs `test --locked -p fastvideo-cudarc
# --lib -- <filters>` (default: `bash scripts/dev/build-pod.sh run <this
# worktree's name> -- cargo`). Needs python3 for the exact-text edits.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
SRC=crates/fastvideo-cudarc/src
if [[ -n "${FV_MUT_RUNNER:-}" ]]; then
  # shellcheck disable=SC2206
  RUNNER=($FV_MUT_RUNNER)
else
  RUNNER=(bash scripts/dev/build-pod.sh run "$(basename "$ROOT")" -- cargo)
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
survived=0

# replace <file> <old> <new>: the one occurrence of <old> becomes <new>.
replace() {
  OLD="$2" NEW="$3" python3 - "$1" <<'PY'
import os, sys
path = sys.argv[1]
old, new = os.environ["OLD"], os.environ["NEW"]
text = open(path).read()
if text.count(old) != 1:
    sys.exit(f"{path}: the mutation's text occurs {text.count(old)} times")
open(path, "w").write(text.replace(old, new))
PY
}

# mutate <label> <file> <old> <new>
mutate() {
  local label="$1" file="$2" status failed
  cp "$file" "$TMP/backup"
  if ! replace "$file" "$3" "$4"; then
    echo "NOT APPLIED $label"
    survived=$((survived + 1))
    return
  fi
  "${RUNNER[@]}" test --locked -p fastvideo-cudarc --lib -- rt::fuse wan::fuse h3::fused16 rt::quant \
    >"$TMP/$label.log" 2>&1
  status=$?
  cp "$TMP/backup" "$file"
  failed="$(grep -E '^test .* FAILED$' "$TMP/$label.log" | sed -E 's/^test (.*) \.\.\. FAILED$/\1/' | tr '\n' ' ')"
  if [[ $status -ne 0 && -n "$failed" ]]; then
    echo "caught   $label: $failed"
  else
    echo "SURVIVED $label (exit $status)"
    survived=$((survived + 1))
  fi
}

mutate wan-cross-reads-f32-sum "$SRC/rt/fuse/policy.rs" \
  $'    residual: Residual::Add,\n    norm: norm(\n        NormInput::StoredBf16,' \
  $'    residual: Residual::Add,\n    norm: norm(\n        NormInput::F32Sum,'
mutate h3-res-gate-muladd-not-fma "$SRC/rt/fuse/policy.rs" \
  $'pub const H3_RES_GATE_NORM_MOD: RoundingPolicy = RoundingPolicy {\n    residual: Residual::Fma,' \
  $'pub const H3_RES_GATE_NORM_MOD: RoundingPolicy = RoundingPolicy {\n    residual: Residual::MulAdd,'
mutate host-muladd-as-fma "$SRC/rt/fuse/host.rs" \
  'Residual::MulAdd => x + u * g,' 'Residual::MulAdd => g.mul_add(u, x),'
mutate rounded-norm-drops-norm-rounding "$SRC/rt/fuse/host.rs" \
  'bf16_round(bf16_round(n[j]) * (1.0 + scale[j]) + shift[j])' 'bf16_round(n[j] * (1.0 + scale[j]) + shift[j])'
mutate wan-qk-no-normalized-rounding "$SRC/rt/fuse/policy.rs" \
  $'    scope: QkScope::AcrossHeads,\n    round_normalized: true,' \
  $'    scope: QkScope::AcrossHeads,\n    round_normalized: false,'
mutate ltx-res-modes-swapped "$SRC/rt/fuse/policy.rs" \
  $'R::RoundedProduct => 1,\n                    _ => 2,' $'R::RoundedProduct => 2,\n                    _ => 1,'
mutate glue-without-fuse "$SRC/rt/fuse/switch.rs" \
  $'pub fn enabled(switch: Switch) -> bool {\n    match switch {\n        Switch::H3Glue => raw(Switch::H3) && raw(Switch::H3Glue),' \
  $'pub fn enabled(switch: Switch) -> bool {\n    match switch {\n        Switch::H3Glue => raw(Switch::H3Glue),'
mutate installed-beats-forced "$SRC/rt/fuse/switch.rs" \
  $'    if let Some(v) = FORCED.with(|c| c.get()[i]) {\n        return v;\n    }\n    if let Some(v) = INSTALLED.with(|c| c.get().values[i]) {\n        return v;\n    }' \
  $'    if let Some(v) = INSTALLED.with(|c| c.get().values[i]) {\n        return v;\n    }\n    if let Some(v) = FORCED.with(|c| c.get()[i]) {\n        return v;\n    }'
mutate h3-gate-residual-gets-wan-policy "$SRC/h3/fused16.rs" \
  $'            &H3_GATE_RESIDUAL,\n            &RowOperands {' \
  $'            &crate::rt::fuse::policy::WAN_GATE_RESIDUAL,\n            &RowOperands {'
mutate interleaved-sin-at-even-slot "$SRC/rt/fuse/host.rs" \
  'let (cv, sv) = (c[even], sn[even + 1]);' 'let (cv, sv) = (c[even], sn[even]);'
mutate no-restore-on-unwind "$SRC/rt/fuse/switch.rs" \
  $'    let _restore = InstalledGuard(INSTALLED.with(|c| c.replace(value)));\n    f()' \
  $'    let prev = INSTALLED.with(|c| c.replace(value));\n    let out = f();\n    INSTALLED.with(|c| c.set(prev));\n    out'

if ((survived)); then
  echo "b4b mutations: $survived survived or did not apply" >&2
  exit 1
fi
echo "b4b mutations: all caught"
