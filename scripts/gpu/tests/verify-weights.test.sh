#!/usr/bin/env bash
# verify-weights.sh's index check on a stand-in weights root (small but valid
# safetensors files, no network, no GPU):
#
#   1. the NVFP4 release trees (fasth3-v2-nvfp4, fasth3-4step-vsa-nvfp4) carry
#      vae/ shards 1-2 and an index naming shard 3, which is h3-base's
#      (docs/ops/r2-weights.md §15): `variant:<id>` passes with h3-base's
#      shard 3 present and complete, and says where shard 3 is read from;
#   2. it fails, naming h3-base's path, when h3-base's shard 3 is missing or
#      truncated;
#   3. any other shard an index names but the tree lacks still fails
#      (h3-base's own vae/ without its shard 3, a missing shard 2 of an NVFP4
#      tree: only the listed shared files may be absent).
#
#   bash scripts/gpu/tests/verify-weights.test.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VERIFY="$ROOT/scripts/gpu/verify-weights.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
bad() { printf 'verify-weights.test: %s\n' "$*" >&2; fail=1; }

# A complete one-tensor safetensors file (header length < 256).
st() {
  local p="$1" h='{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
  mkdir -p "$(dirname "$p")"
  { printf "\\$(printf '%03o' "${#h}")\0\0\0\0\0\0\0"; printf '%s' "$h"; printf '\0\0\0\0'; } >"$p"
}
SHARD() { printf 'diffusion_pytorch_model-0000%s-of-00003.safetensors' "$1"; }
INDEX='{"weight_map": {"a": "'"$(SHARD 1)"'", "b": "'"$(SHARD 2)"'", "c": "'"$(SHARD 3)"'"}}'

# h3-base: every tree the variant reads, its vae/ complete.
fill_base() {
  local W="$1" d
  for d in transformer audio_vae text_encoder; do st "$W/h3-base/$d/model.safetensors"; done
  mkdir -p "$W/h3-base/tokenizer" && echo '{}' >"$W/h3-base/tokenizer/tokenizer.json"
  mkdir -p "$W/h3-base/vae" && echo '{}' >"$W/h3-base/vae/config.json"
  printf '%s' "$INDEX" >"$W/h3-base/vae/diffusion_pytorch_model.safetensors.index.json"
  for i in 1 2 3; do st "$W/h3-base/vae/$(SHARD "$i")"; done
}
# An NVFP4 release tree as published in R2: vae/ shards 1-2 and the index.
fill_release() {
  local W="$1" t="$2"
  st "$W/$t/transformer/diffusion_pytorch_model-00001-of-00001.safetensors"
  mkdir -p "$W/$t/vae" && echo '{}' >"$W/$t/vae/config.json"
  printf '%s' "$INDEX" >"$W/$t/vae/diffusion_pytorch_model.safetensors.index.json"
  for i in 1 2; do st "$W/$t/vae/$(SHARD "$i")"; done
}
run() { FV_WEIGHTS="$1" bash "$VERIFY" "${@:2}" >"$TMP/out" 2>&1; }

for t in fasth3-v2-nvfp4 fasth3-4step-vsa-nvfp4; do
  W="$TMP/$t-ok"
  fill_base "$W"
  fill_release "$W" "$t"
  # 1. As published: passes, shard 3 from h3-base.
  if run "$W" "variant:$t"; then
    grep -q "shared shard $W/$t/vae/$(SHARD 3): $W/h3-base/vae/$(SHARD 3)" "$TMP/out" \
      || bad "$t: no line naming h3-base's shard 3: $(cat "$TMP/out")"
  else
    bad "$t: variant:$t failed on the tree as published: $(cat "$TMP/out")"
  fi

  # 2a. h3-base's shard 3 missing: fails, naming it.
  W="$TMP/$t-nobase3"
  fill_base "$W"
  fill_release "$W" "$t"
  rm "$W/h3-base/vae/$(SHARD 3)"
  if run "$W" "variant:$t"; then
    bad "$t: passed without h3-base's shard 3"
  else
    grep -q "MISSING shard $W/$t/vae/$(SHARD 3) .*the engine reads $W/h3-base/vae/$(SHARD 3), which is missing" "$TMP/out" \
      || bad "$t: the failure does not name h3-base's shard 3: $(cat "$TMP/out")"
  fi

  # 2b. h3-base's shard 3 truncated: fails.
  W="$TMP/$t-trunc3"
  fill_base "$W"
  fill_release "$W" "$t"
  truncate -s 20 "$W/h3-base/vae/$(SHARD 3)"
  run "$W" "variant:$t" && bad "$t: passed with h3-base's shard 3 truncated"

  # 3. A shard the tree must carry (shard 2) missing: fails as before.
  W="$TMP/$t-no2"
  fill_base "$W"
  fill_release "$W" "$t"
  rm "$W/$t/vae/$(SHARD 2)"
  if run "$W" "variant:$t"; then
    bad "$t: passed without its own vae shard 2"
  else
    grep -q "MISSING shard $W/$t/vae/$(SHARD 2) (listed in" "$TMP/out" \
      || bad "$t: the failure does not name shard 2: $(cat "$TMP/out")"
  fi
done

# 3. h3-base's own vae/ is not completed from anywhere.
W="$TMP/base-no3"
fill_base "$W"
rm "$W/h3-base/vae/$(SHARD 3)"
if run "$W" h3-base; then
  bad "h3-base passed without its vae shard 3"
else
  grep -q "MISSING shard $W/h3-base/vae/$(SHARD 3) (listed in" "$TMP/out" \
    || bad "h3-base: the failure does not name shard 3: $(cat "$TMP/out")"
fi

if (( fail )); then
  echo "verify-weights.test FAILED" >&2
  exit 1
fi
echo "verify-weights.test ok"
