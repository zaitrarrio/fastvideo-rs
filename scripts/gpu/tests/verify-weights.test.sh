#!/usr/bin/env bash
# verify-weights.sh's index check on a stand-in weights root (small but valid
# safetensors files, no network, no GPU). The script and verify-safetensors.sh
# run from a copy whose recorded size / SHA-256 of h3-base's vae shard 3
# (H3_VAE_SHARD3_SIZE / _SHA256) are the stand-in's; everything else is the
# repository's script as is.
#
#   1. the NVFP4 release trees (fasth3-v2-nvfp4, fasth3-4step-vsa-nvfp4) carry
#      vae/ shards 1-2 and an index naming shard 3, which is h3-base's
#      (docs/ops/r2-weights.md §15): `variant:<id>` and the hand-written cell
#      pass with h3-base's shard 3 complete, of the recorded size and SHA-256,
#      and say where shard 3 is read from;
#   2. they fail, naming h3-base's copy, when it is missing, truncated, of
#      another size or another SHA-256;
#   3. any other shard an index names but its tree lacks still fails: a
#      missing shard 2 of an NVFP4 tree, h3-base's own shard 3, and h3-8step's
#      shard 3 under variant:fasth3-8step-dense with h3-base complete (only the
#      listed shared files may be absent).
#
#   bash scripts/gpu/tests/verify-weights.test.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
bad() { printf 'verify-weights.test: %s\n' "$*" >&2; fail=1; }

# A complete safetensors file of one F32 tensor of $2 elements (default 1),
# its data bytes all $3 (an octal escape, default \0). Header length < 256.
st() {
  local p="$1" n="${2:-1}" byte="${3:-\\0}" h i
  h='{"w":{"dtype":"F32","shape":['"$n"'],"data_offsets":[0,'"$((4 * n))"']}}'
  mkdir -p "$(dirname "$p")"
  {
    printf "\\$(printf '%03o' "${#h}")\0\0\0\0\0\0\0"
    printf '%s' "$h"
    for ((i = 0; i < 4 * n; i++)); do printf "$byte"; done
  } >"$p"
}
SHARD() { printf 'diffusion_pytorch_model-0000%s-of-00003.safetensors' "$1"; }
INDEX='{"weight_map": {"a": "'"$(SHARD 1)"'", "b": "'"$(SHARD 2)"'", "c": "'"$(SHARD 3)"'"}}'

# The script under test, its recorded shard-3 size / SHA-256 set to the stand-in's.
GPU="$TMP/gpu"
mkdir -p "$GPU"
cp "$ROOT/scripts/gpu/verify-safetensors.sh" "$GPU/"
st "$TMP/ref3.safetensors"
REF_SIZE="$(wc -c <"$TMP/ref3.safetensors" | tr -d ' ')"
REF_SHA="$(sha256sum "$TMP/ref3.safetensors" | awk '{print $1}')"
for k in SIZE SHA256; do
  n="$(grep -c "^H3_VAE_SHARD3_$k=" "$ROOT/scripts/gpu/verify-weights.sh")"
  [[ "$n" == 1 ]] || bad "verify-weights.sh: $n lines H3_VAE_SHARD3_$k=, expected 1"
done
sed -e "s/^H3_VAE_SHARD3_SIZE=.*/H3_VAE_SHARD3_SIZE=$REF_SIZE/" \
  -e "s/^H3_VAE_SHARD3_SHA256=.*/H3_VAE_SHARD3_SHA256=$REF_SHA/" \
  "$ROOT/scripts/gpu/verify-weights.sh" >"$GPU/verify-weights.sh"

# A tree with an H3 vae/ (index naming 3 shards) holding shards $3 (e.g. "1 2 3").
fill_vae() {
  local d="$1/vae" i
  mkdir -p "$d" && echo '{}' >"$d/config.json"
  printf '%s' "$INDEX" >"$d/diffusion_pytorch_model.safetensors.index.json"
  for i in $2; do st "$d/$(SHARD "$i")"; done
}
# h3-base / h3-8step: every component the cells read, vae/ complete.
fill_h3() {
  local t="$1/$2" d
  for d in transformer audio_vae text_encoder; do st "$t/$d/model.safetensors"; done
  mkdir -p "$t/tokenizer" && echo '{}' >"$t/tokenizer/tokenizer.json"
  fill_vae "$t" "1 2"
  cp "$TMP/ref3.safetensors" "$t/vae/$(SHARD 3)"
}
# An NVFP4 release tree as published in R2: transformer/, vae/ shards 1-2 and the index.
fill_release() {
  st "$1/$2/transformer/diffusion_pytorch_model-00001-of-00001.safetensors"
  fill_vae "$1/$2" "1 2"
}
run() { FV_WEIGHTS="$1" bash "$GPU/verify-weights.sh" "${@:2}" >"$TMP/out" 2>&1; }
setup() { # <case> <release tree>: a fresh root with h3-base and the release tree
  W="$TMP/$1"
  fill_h3 "$W" h3-base
  fill_release "$W" "$2"
}

# The hand-written cells' sha:<dest> lists: the stand-in files (all the same bytes).
for t in fasth3-v2-nvfp4 fasth3-4step-vsa-nvfp4; do
  for f in transformer/diffusion_pytorch_model-00001-of-00001.safetensors "vae/$(SHARD 1)" "vae/$(SHARD 2)"; do
    printf '%s\t%s\t%s\tsha256:%s\n' "$t" "$f" "$REF_SIZE" "$REF_SHA"
  done
done >"$GPU/weights-sha256.tsv"

for t in fasth3-v2-nvfp4 fasth3-4step-vsa-nvfp4; do
  # 1. As published: passes (the variant cell and the hand-written one), shard 3 from h3-base.
  for cell in "variant:$t" "$t"; do
    setup "$t-ok" "$t"
    B3="$W/h3-base/vae/$(SHARD 3)"
    if run "$W" "$cell"; then
      grep -q "shared shard $W/$t/vae/$(SHARD 3): $B3 (complete, $REF_SIZE bytes" "$TMP/out" \
        || bad "$cell: no line naming h3-base's shard 3: $(cat "$TMP/out")"
    else
      bad "$cell failed on the tree as published: $(cat "$TMP/out")"
    fi
  done
  # The hand-written cell reads h3-base, not h3-8step (none here).
  grep -q h3-8step "$TMP/out" && bad "$t: the hand-written cell names h3-8step: $(cat "$TMP/out")"

  # 2a. h3-base's shard 3 missing: fails, naming it.
  setup "$t-missing" "$t"
  B3="$W/h3-base/vae/$(SHARD 3)"
  rm "$B3"
  if run "$W" "variant:$t"; then
    bad "$t: passed without h3-base's shard 3"
  else
    grep -q "MISSING shard $W/$t/vae/$(SHARD 3) .*the engine reads $B3, which is missing" "$TMP/out" \
      || bad "$t: the failure does not name h3-base's shard 3: $(cat "$TMP/out")"
  fi

  # 2b. Truncated: the shared copy's own check fails (not only h3-base's tree check).
  setup "$t-trunc" "$t"
  B3="$W/h3-base/vae/$(SHARD 3)"
  truncate -s 20 "$B3"
  if run "$W" "variant:$t"; then
    bad "$t: passed with h3-base's shard 3 truncated"
  else
    grep -q "shared copy $B3 incomplete" "$TMP/out" \
      || bad "$t: truncated shard 3: no 'shared copy … incomplete' line: $(cat "$TMP/out")"
  fi

  # 2c. A complete safetensors file of another size: fails on the size.
  setup "$t-size" "$t"
  B3="$W/h3-base/vae/$(SHARD 3)"
  st "$B3" 2
  if run "$W" "variant:$t"; then
    bad "$t: passed with h3-base's shard 3 of another size"
  else
    grep -q "shared copy $B3: $(wc -c <"$B3" | tr -d ' ') bytes, expected $REF_SIZE" "$TMP/out" \
      || bad "$t: shard 3 of another size: no size line: $(cat "$TMP/out")"
  fi

  # 2d. The same size, other bytes: fails on the SHA-256.
  setup "$t-sha" "$t"
  B3="$W/h3-base/vae/$(SHARD 3)"
  st "$B3" 1 '\001'
  if run "$W" "variant:$t"; then
    bad "$t: passed with h3-base's shard 3 of another SHA-256"
  else
    grep -q "shared copy $B3: SHA-256 [0-9a-f]*, expected $REF_SHA" "$TMP/out" \
      || bad "$t: shard 3 of another SHA-256: no SHA-256 line: $(cat "$TMP/out")"
  fi

  # 3a. A shard the tree must carry (shard 2) missing: fails as before.
  setup "$t-no2" "$t"
  rm "$W/$t/vae/$(SHARD 2)"
  if run "$W" "variant:$t"; then
    bad "$t: passed without its own vae shard 2"
  else
    grep -q "MISSING shard $W/$t/vae/$(SHARD 2) (listed in" "$TMP/out" \
      || bad "$t: the failure does not name shard 2: $(cat "$TMP/out")"
  fi
done

# 3b. h3-base's own vae/ is not completed from anywhere.
W="$TMP/base-no3"
fill_h3 "$W" h3-base
rm "$W/h3-base/vae/$(SHARD 3)"
if run "$W" h3-base; then
  bad "h3-base passed without its vae shard 3"
else
  grep -q "MISSING shard $W/h3-base/vae/$(SHARD 3) (listed in" "$TMP/out" \
    || bad "h3-base: the failure does not name shard 3: $(cat "$TMP/out")"
fi

# 3c. A tree outside the table (h3-8step) is not completed from h3-base either,
# though h3-base is in the variant and holds the same file name.
W="$TMP/8step-no3"
fill_h3 "$W" h3-base
fill_h3 "$W" h3-8step
rm "$W/h3-8step/vae/$(SHARD 3)"
if run "$W" variant:fasth3-8step-dense; then
  bad "variant:fasth3-8step-dense passed without h3-8step's vae shard 3"
else
  grep -q "MISSING shard $W/h3-8step/vae/$(SHARD 3) (listed in" "$TMP/out" \
    || bad "fasth3-8step-dense: the failure does not name h3-8step's shard 3: $(cat "$TMP/out")"
fi

if (( fail )); then
  echo "verify-weights.test FAILED" >&2
  exit 1
fi
echo "verify-weights.test ok"
